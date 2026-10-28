# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The CPU-only dump compiles exactly the TTGIR a real GPU launch compiles.

Runs on a Blackwell GPU host (BUCK: remote execution on GB300, i.e. sm_103; a B200 host
checks sm_100). The TLX tutorial GEMM is launched through its host function on the GPU;
its autotuner compiles every config. Then the same host call is captured by dump.run on
the CPU-only path (gpu_free + capture) for the GPU's target, and every compiled TTGIR is
compared with the GPU's. This checks the specialization (signature, divisibility attrs,
constexprs), the backend options and the target that the CPU path derives.
"""

import importlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from triton import knobs

from ...frontend.ttgir import dump, gpu_free

KERNEL = "matmul_kernel_tma_pipelined_blackwell"


class GpuParityTest(unittest.TestCase):
    def test_tutorial_gemm_ttgir_matches_the_gpu_compile(self):
        if not torch.cuda.is_available():
            self.skipTest(
                "needs a Blackwell GPU (run through the BUCK target's remote execution)"
            )
        target = "sm_{}{}".format(*torch.cuda.get_device_capability())
        if target not in gpu_free.TARGETS:
            self.skipTest(
                f"{target} is not an analyzed target ({sorted(gpu_free.TARGETS)})"
            )
        root = Path(tempfile.mkdtemp(prefix="lkv_gpu_parity_"))
        self.addCleanup(shutil.rmtree, root)
        tutorial = importlib.util.find_spec(
            "triton.language.extra.tlx.tutorials.blackwell_gemm_pipelined"
        )
        shutil.copy(tutorial.origin, root / "lkv_parity_gemm.py")
        # The capture needs JITFunction.run, which the C launch cache would bypass; the knob
        # is read when the kernel is decorated (it does not change the compiled IR).
        knobs.nvidia.enable_c_cache = False
        sys.path.insert(0, str(root))
        module = importlib.import_module("lkv_parity_gemm")

        a = torch.randn(256, 256, dtype=torch.float16, device="cuda")
        b = torch.randn(256, 256, dtype=torch.float16, device="cuda")
        module.matmul(a, b)
        torch.cuda.synchronize()
        jit = getattr(module, KERNEL).fn
        compiled = [
            k for cache in jit.device_caches.values() for k in cache[0].values()
        ]
        gpu = sorted(
            (k.result() if hasattr(k, "result") else k).asm["ttgir"] for k in compiled
        )
        self.assertEqual(len(gpu), len(module.get_cuda_autotune_config()))

        # The CPU path builds its own binder for the CPU driver's target.
        jit.device_caches.clear()
        case = {
            "name": "gpu_parity",
            "src_root": str(root),
            "module": "lkv_parity_gemm",
            "entry": "matmul",
            "kernel": KERNEL,
            "num_sms": torch.cuda.get_device_properties(0).multi_processor_count,
            "inputs": [
                {
                    "name": "square",
                    "args": {
                        "a": {"dtype": "float16", "shape": [256, 256]},
                        "b": {"dtype": "float16", "shape": [256, 256]},
                    },
                }
            ],
        }
        out = root / "dump"
        dump.run(case, out, target=target)
        specs = json.loads((out / "dump.json").read_text())["specs"]
        cpu = sorted((out / s["ttgir"]).read_text() for s in specs.values())
        self.assertEqual(cpu, gpu)


if __name__ == "__main__":
    unittest.main()
