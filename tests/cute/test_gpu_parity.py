# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The CPU-only dump traces exactly the CuTe MLIR a real GPU launch compiles, and a
whole-warp tcgen05.commit arrives once per lane.

Runs on a Blackwell GPU host (buck target :test_gpu_parity: remote execution on a GB300);
without one it skips. kernels.py's TMA pipeline is compiled and launched on the GPU through
its host function, with the DSL's compile_and_cache spied on for the initial MLIR and the
pass pipeline it compiles; then the same host call is dumped on the CPU-only path
(dump.run, host_patches.py) and the two are compared. This checks the host patches this
host call reaches: CPU tensors typed as CUDA tensors through DLPack (P4) give the argument
types, layouts and constants of the GPU compile, and the recorded pass pipeline is the
DSL's own (P5); the PTX the dump keeps is that compile's, so it is the GPU's too. Its host
makes no device query, so the device patches (P1-P3) and the recorded launches (P6, P9)
are not compared. The DSL reads its environment once, at its first compile, so both
compiles are for the GPU's arch (sm_103a on a GB300, the analyzed sm_100a on a B200).

The op table counts a tcgen05.commit once per executing lane (ir_ops_cute.md, "Issuing
threads"; the DSL does not elect it, test_frontend `test_issuing_threads_match_the_ptx`):
kernels.py's UMMA kernels commit from all 32 lanes of a warp to a count-32 mbarrier,
which completes only if every lane's commit arrives (an elected commit would hang).
"""

import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from ...frontend.cute import dump

HERE = Path(__file__).resolve().parent


def blackwell(test) -> tuple:
    if not torch.cuda.is_available():
        test.skipTest(
            "needs a Blackwell GPU (run through the buck target's remote execution)"
        )
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        test.skipTest(f"sm_{major}{minor} is not an SM100-family GPU")
    return major, minor


def kernels():
    """kernels.py, imported as dump.py imports it, so that MLIR locations name one file."""
    sys.path.insert(0, str(HERE))
    # tests/cute/kernels.py (a resource), not the pypi package autodeps guesses
    return importlib.import_module("kernels")  # @manual


class GpuParityTest(unittest.TestCase):
    def test_tma_pipeline_mlir_matches_the_gpu_compile(self):
        major, minor = blackwell(self)
        root = Path(tempfile.mkdtemp(prefix="lkv_cute_gpu_parity_"))
        self.addCleanup(shutil.rmtree, root)
        os.environ["CUTE_DSL_ARCH"] = (
            f"sm_{major}{minor}a"  # before the DSL's first compile
        )
        from cutlass.base_dsl.dsl import (  # noqa: PLC0415 - after the environment
            BaseDSL,
        )

        from ...frontend.cute.host_patches import (  # noqa: PLC0415 - imports the DSL
            GENERIC_FORM,
        )

        real, gpu = BaseDSL.compile_and_cache, []

        def spy(dsl, module, module_hash, function_name, pipeline, *a, **k):
            # What host_patches.py's P5 records, then the real compile.
            arch = dsl.compile_options.gpu_arch or dsl.envar.arch
            text = module.operation.get_asm(
                enable_debug_info=True, print_generic_op_form=GENERIC_FORM
            )
            gpu.append((text + "\n", dsl.preprocess_pipeline(dsl._get_pipeline(pipeline), arch)))  # fmt: skip
            return real(dsl, module, module_hash, function_name, pipeline, *a, **k)

        k = kernels()
        a = torch.randn(100, k.TILE, dtype=torch.bfloat16)
        out = torch.zeros(100, device="cuda")
        with mock.patch.object(BaseDSL, "compile_and_cache", spy):
            k.run_tma_pipeline(
                a.cuda(), out, elect_producer=True, extra_rows=0, copy_lanes=32
            )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out.cpu(), a[:, 0].float()))  # the kernel ran

        case = {
            "name": "gpu_parity",
            "source": {"package": "kernels.py", "src": str(HERE)},
            "env": {},
            "entry": "kernels:run_tma_pipeline",
            "builder": f"{HERE / 'kernels.py'}:args_tma_pipeline",
            "num_sms": torch.cuda.get_device_properties(0).multi_processor_count,
            "inputs": [{"name": "rows100", "rows": 100}],
        }
        path = root / "gpu_parity.json"
        path.write_text(json.dumps(case))
        (c,) = dump.run(path, root / "out")["compiles"]
        self.assertEqual([(Path(c["ir"]).read_text(), c["pipeline"])], gpu)


class WholeWarpCommitOnGpuTest(unittest.TestCase):
    def test_whole_warp_commits_complete_a_count_32_barrier(self):
        blackwell(self)
        k = kernels()
        for block_scaled in (False, True):
            k.run_umma(torch.zeros(1, device="cuda"), block_scaled)
        torch.cuda.synchronize()  # returns: the barriers completed (a hang times out)
        # both kernels were compiled and launched
        self.assertEqual(sorted(k._UMMA_HOSTS), [False, True])


if __name__ == "__main__":
    unittest.main()
