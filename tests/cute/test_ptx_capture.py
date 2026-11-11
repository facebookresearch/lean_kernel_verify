# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The PTX capture of the CuTe dump (host_patches.py P5/P6/P9/P10) on the CPU.

The e2e probe kernel (tests/e2e/cases/probe_cute.json) and a small TMA kernel compiled
with and without TVM-FFI run through dump.run: every compile keeps the initial MLIR the
variant digest is taken from (the PTX capture must not change it) and gets its PTX and
cubin, and every launch gets the LaunchRecord its compiled host code makes, with the
parameter bytes it built, the CUtensorMap decoded and addresses resolved to the named
tensors. Needs the CuTe DSL (no GPU): buck target :test_ptx_capture.
"""

import hashlib
import importlib.metadata
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

if __package__ in (None, ""):
    # Run as a script: the project root becomes the package `lkv_root`, so the relative
    # imports below resolve as they do in the buck target.
    sys.dont_write_bytecode = True
    _root = ModuleType("lkv_root")
    _root.__path__ = [str(Path(__file__).resolve().parents[2])]
    sys.modules["lkv_root"] = _root
    __package__ = "lkv_root.tests.cute"  # noqa: A001

from ...frontend.cute import dump  # noqa: E402

# host_patches imports the DSL, which reads its environment (CUTE_DSL_ARCH) once: it is
# imported only after a dump pinned it (dump.run), in the tests below.

SKILL = Path(__file__).resolve().parents[2]
DSL = importlib.metadata.version("nvidia-cutlass-dsl")  # this test binary's

# A TMA descriptor over a column view of a bf16 matrix (its address is 32 bytes into the
# storage) and a tensor the kernel updates; written to a temporary source directory.
TMA_KERNELS = """
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def touch(atom: cute.CopyAtom, gA: cute.Tensor, mB: cute.Tensor, n: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    cpasync.prefetch_descriptor(atom)
    if tidx < n:
        mB[tidx] = mB[tidx] + 1.0


@cute.jit
def host(mA: cute.Tensor, mB: cute.Tensor, n: cutlass.Int32):
    smem = cute.make_layout((64, 32), stride=(32, 1))
    atom, gA = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mA, smem, (64, 32)
    )
    touch(atom, gA, mB, n).launch(
        grid=(3, 1, 1), block=(128, 1, 1), cluster=(1, 1, 1), smem=1024
    )


def run(a, b, n, tvm_ffi):
    ma = from_dlpack(a, assumed_align=16, enable_tvm_ffi=tvm_ffi)
    mb = from_dlpack(b, assumed_align=16, enable_tvm_ffi=tvm_ffi)
    options = {"options": "--enable-tvm-ffi"} if tvm_ffi else {}
    compiled = cute.compile(
        host,
        ma.mark_layout_dynamic(leading_dim=1),
        mb.mark_layout_dynamic(leading_dim=0),
        n,
        **options,
    )
    compiled(a, b, n)  # torch tensors, as TVM-FFI callers pass them


def args(inp):
    base = torch.zeros(256, 96, dtype=torch.bfloat16)
    return [base[:, 16:80], torch.zeros(128), inp["n"], inp["tvm_ffi"]], {}, {}
"""


def run_dump(case: dict, src: Path) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix=f"lkv_ptx_capture_{case['name']}_"))
    path = tmp / "case.json"
    path.write_text(json.dumps(case))
    return dump.run(path, tmp / "out", str(src), lineinfo_check=DSL == "4.4.2")


def entries(ptx: str) -> list:
    return re.findall(r"\.entry\s+(\w+)", ptx)


class PtxCaptureTest(unittest.TestCase):
    def assert_compiles(self, recorded: dict, tvm_ffi: list):
        self.assertEqual([c["ptx"]["tvm_ffi"] for c in recorded["compiles"]], tvm_ffi)
        for c in recorded["compiles"]:
            self.assertNotIn("ptx_error", c)
            # The variant digest is still the initial MLIR's.
            mlir = Path(c["ir"]).read_text()
            self.assertEqual(
                c["digest"], hashlib.sha256(mlir.encode()).hexdigest()[:16]
            )
            ptx = Path(c["ptx"]["file"]).read_text()
            self.assertEqual(
                c["ptx"]["sha256"], hashlib.sha256(ptx.encode()).hexdigest()
            )
            self.assertEqual(c["ptx"]["target"], "sm_100a")
            self.assertRegex(ptx, r"\n\s*\.loc\s")  # compiled with line info
            if DSL == "4.4.2":
                self.assertTrue(c["ptx"]["lineinfo_check"]["identical"], c["ptx"])

    def assert_static_smem(self, record: dict, compile_: dict):
        from ...frontend.cute.host_patches import static_smem  # noqa: PLC0415

        cubin = Path(compile_["ptx"]["cubin"]).read_bytes()
        static = static_smem(cubin, record["entry"])
        # The host asks for the opt-in maximum minus the kernel's static shared memory.
        self.assertEqual(record["kernel_attrs"], {"8": 232448 - static, "14": 1})

    def test_probe_kernel(self):
        case = json.loads((SKILL / "tests/e2e/cases/probe_cute.json").read_text())
        case |= {
            "dsl": DSL,
            "builder": str(SKILL / "tests/e2e/kernels/lkv_e2e_probe_cute.py") + ":args",
        }
        recorded = run_dump(case, SKILL / "tests/e2e/kernels")
        self.assert_compiles(recorded, [False, False, False])
        # The three inputs compile the same module: one digest (line info must not leak
        # from one compile into the next initial MLIR).
        self.assertEqual(len({c["digest"] for c in recorded["compiles"]}), 1)
        inputs = {i["name"]: i for i in recorded["inputs"]}
        self.assertEqual(inputs["n200"]["launches"], [])
        for name, n in (("n128", 128), ("n100", 100), ("n0", 0)):
            (launch,) = inputs[name]["launches"]
            self.assertNotIn("capture_error", launch)
            (r,) = launch["launch_records"]
            c = recorded["compiles"][launch["compile"]]
            self.assertEqual(entries(Path(c["ptx"]["file"]).read_text()), [r["entry"]])
            self.assertEqual(
                (r["kernel"], r["grid"], r["block"], r["cluster"], r["smem_dynamic"]),
                ("host", [1, 1, 1], [128, 1, 1], [1, 1, 1], 0),
            )
            self.assertEqual(r["attrs"], {"pdl": False, "cooperative": False})
            self.assert_static_smem(r, c)
            # The tensor is a struct: its data pointer, then its dynamic extent.
            (p,) = r["params"]
            self.assertEqual((p["kind"], p["size"]), ("aggregate", 16))
            raw = bytes.fromhex(p["bytes"])
            self.assertEqual(int.from_bytes(raw[8:12], "little"), n)
            (ptr,) = p["fields"]  # the kernel reads the pointer only
            self.assertEqual((ptr["off"], ptr["size"], ptr["name"]), (0, 8, "param0_0"))
            arg = launch["args"][0]
            self.assertEqual(ptr["value"], arg["storage"][0] + arg["byte_offset"])
            if n:
                self.assertEqual(
                    ptr["tensor"],
                    [{"tensor": "m", "storage": [arg["storage"][0], arg["storage"][0] + 4 * n],
                      "elem_off": 0, "match": "exact"}],
                )  # fmt: skip

    def test_tma_descriptor_with_and_without_tvm_ffi(self):
        src = Path(tempfile.mkdtemp(prefix="lkv_ptx_capture_src_"))
        (src / "lkv_tma_kernels.py").write_text(TMA_KERNELS)
        case = {
            "name": "tma",
            "source": {"package": "lkv_tma_kernels.py", "src": str(src)},
            "entry": "lkv_tma_kernels:run",
            "builder": f"{src / 'lkv_tma_kernels.py'}:args",
            "num_sms": 2,
            "dsl": DSL,
            "inputs": [
                {"name": "plain", "n": 7, "tvm_ffi": False},
                {"name": "tvm_ffi", "n": 7, "tvm_ffi": True},
            ],
        }
        recorded = run_dump(case, src)
        self.assert_compiles(recorded, [False, True])
        for inp in recorded["inputs"]:
            (launch,) = inp["launches"]
            self.assertNotIn("capture_error", launch)
            (r,) = launch["launch_records"]
            self.assertEqual(
                (r["grid"], r["block"], r["cluster"], r["smem_dynamic"]),
                ([3, 1, 1], [128, 1, 1], [1, 1, 1], 1024),
            )
            self.assert_static_smem(r, recorded["compiles"][launch["compile"]])
            tmap, coords, b, n = r["params"]
            self.assertEqual(
                [p["kind"] for p in r["params"]],
                ["tensormap", "aggregate", "aggregate", "scalar"],
            )
            t = tmap["tensormap"]
            self.assertEqual(
                (t["dtype"], t["rank"], t["dims"], t["strides_bytes"], t["box"]),
                ("BFLOAT16", 2, [64, 256], [192], [32, 64]),
            )
            self.assertEqual(
                t["unwritten_words"], [] if DSL == "4.4.2" else list(range(8, 16))
            )
            mA, mB = launch["args"][0], launch["args"][1]
            self.assertEqual(t["address"], mA["storage"][0] + mA["byte_offset"])
            self.assertEqual(mA["byte_offset"], 32)
            self.assertEqual(
                [(x["tensor"], x["elem_off"], x["match"]) for x in t["base"]],
                [("mA", 0, "exact")],
            )
            self.assertEqual(
                [
                    (f["off"], [x["tensor"] for x in f.get("tensor", [])])
                    for f in b["fields"]
                ],
                [(0, ["mB"])],
            )
            self.assertEqual(b["fields"][0]["value"], mB["storage"][0])
            self.assertEqual((n["type"], n["value"]), ("u32", 7))

    def test_unmodeled_runtime_functions_stop_the_engine(self):
        # (Runs after the dumps above: test methods run in name order.)
        from ...frontend.cute import host_patches  # noqa: PLC0415

        class Engine:
            def raw_register_runtime(self, name, ptr):
                raise AssertionError("nothing is registered when a function is unknown")

        runtime = host_patches._HostRuntime(num_sms=2)
        with self.assertRaisesRegex(
            RuntimeError, r"declares _cudaMalloc, _cudaMemcpy, which"
        ):
            runtime.register(
                Engine(),
                ["_cudaLaunchKernelEx", "printf", "_cudaMalloc", "_cudaMemcpy"],
            )


if __name__ == "__main__":
    unittest.main()
