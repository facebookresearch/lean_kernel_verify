# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The TTGIR dump's PTX capture for the PTX frontend, on small TLX kernels.

    buck2 test fbcode//lean_kernel_verify/tests/ttgir:test_ptx_capture

Each test writes a kernel module plus its host function into a temporary source tree, runs
the real dump (CPU host run, launch capture, compile for sm_100) and checks what it adds
for the PTX frontend: <spec>.ptx with its sha256 and source paths kept, the compiled
kernel's metadata per spec, and per launch the LaunchRecord whose device parameters
(Triton's own expand_signature / wrap_handle_tensordesc, constexprs dropped, the two
scratch pointers appended) agree with the PTX `.entry` (frontend/ptx/launch.py).
"""

import hashlib
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import ModuleType

if __package__ in (None, ""):
    # Run as a script (with torch and Triton importable): the project root becomes the
    # package `lkv_root`, so the relative imports below resolve as in the buck target.
    sys.dont_write_bytecode = True
    _root = ModuleType("lkv_root")
    _root.__path__ = [str(Path(__file__).resolve().parents[2])]
    sys.modules["lkv_root"] = _root
    __package__ = "lkv_root.tests.ttgir"  # noqa: A001

from ...frontend.ptx import launch as ptx_launch
from ...frontend.ttgir import capture, dump

KERNELS = """
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def probe(x_ptr, n, BLOCK: tl.constexpr):
    bars = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    b = tlx.local_view(bars, 0)
    with tlx.async_tasks():
        with tlx.async_task("default"):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            v = tl.load(x_ptr + offs, mask=offs < n)
            tl.store(x_ptr + offs, v + 1, mask=offs < n)
            tlx.barrier_arrive(b)
        with tlx.async_task(num_warps=1):
            tlx.barrier_wait(b, 0)


def run(x, BLOCK=128):
    probe[(triton.cdiv(x.numel(), BLOCK),)](x, x.numel(), BLOCK=BLOCK)


@triton.jit
def desc_copy(in_desc, out_ptr, n, flag, scale, big, one, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK
    tile = in_desc.load([row, 0])
    if flag:
        tile = tile * scale + (big % 7).to(tl.float32) * one
    out = tl.make_tensor_descriptor(
        out_ptr, shape=[n, BLOCK], strides=[BLOCK, 1], block_shape=[BLOCK, BLOCK]
    )
    out.store([row, 0], tile)


def run_desc(x, out, BLOCK=32):
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8))
    desc = TensorDescriptor.from_tensor(x, [BLOCK, BLOCK])
    n = x.shape[0]
    desc_copy[(triton.cdiv(n, BLOCK),)](desc, out, n, True, 0.5, 2**40, 1, BLOCK=BLOCK)
"""


def case(root: Path, entry: str, kernel: str, inputs: list) -> dict:
    return {
        "name": f"test_ptx_{kernel}",
        "src_root": str(root),
        "module": "lkv_ptx_capture_kernels",
        "entry": entry,
        "kernel": kernel,
        "num_sms": 2,
        "inputs": inputs,
    }


class PtxCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="lkv_ptx_src_"))
        cls.out = Path(tempfile.mkdtemp(prefix="lkv_ptx_out_"))
        (cls.root / "lkv_ptx_capture_kernels.py").write_text(textwrap.dedent(KERNELS))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)
        shutil.rmtree(cls.out)

    def dump(self, c: dict) -> tuple:
        out = self.out / c["name"]
        return dump.run(c, out), out

    def assert_matches_ptx(self, launch: dict, ptx: str):
        decls = ptx_launch.ptx_params(ptx, launch["entry"])
        self.assertEqual(len(launch["params"]), len(decls))
        for p, d in zip(launch["params"], decls):
            self.assertEqual(p["decl"], d["decl"])
            self.assertEqual(
                ({"tensormap": "bytes"}.get(p["kind"], p["kind"]), p["size"]),
                (d["kind"], d["size"]),
                p["name"],
            )

    def test_probe_ptx_and_launch(self):
        c = case(
            self.root,
            "run",
            "probe",
            [
                {
                    "name": "partial",
                    "args": {"x": {"dtype": "float32", "shape": [300]}},
                },
                {"name": "empty", "args": {"x": {"dtype": "float32", "shape": [0]}}},
            ],
        )
        d, out = self.dump(c)
        for key, spec in d["specs"].items():
            ptx = (out / spec["ptx"]).read_text()
            self.assertEqual(spec["ptx"], f"{key}.ptx")
            self.assertEqual(
                spec["ptx_sha256"], hashlib.sha256(ptx.encode()).hexdigest()
            )
            self.assertTrue((out / spec["ttgir"]).is_file())  # legacy output unchanged
            # TRITON_PRESERVE_PTX_SOURCE_PATHS: .file names the exported source.
            self.assertIn(f'"{self.root / "lkv_ptx_capture_kernels.py"}"', ptx)
            self.assertEqual(spec["tensordesc_meta"], [])
            self.assertEqual(spec["global_scratch"]["size"], 0)
            self.assertEqual(spec["profile_scratch_size"], 0)
            self.assertTrue(spec["ptxas"]["version"])
        self.assertEqual(d["env"]["own"]["TRITON_PRESERVE_PTX_SOURCE_PATHS"], "1")
        grids = {}
        for inp in d["inputs"]:
            (la,) = inp["launches"]
            rec = la["launch"]
            self.assert_matches_ptx(rec, (out / f"{la['spec']}.ptx").read_text())
            self.assertEqual(
                [(p["name"], p["kind"]) for p in rec["params"]],
                [
                    ("x_ptr", "ptr"),
                    ("n", "scalar"),  # BLOCK (constexpr) is not a parameter
                    ("__global_scratch", "ptr"),
                    ("__profile_scratch", "ptr"),
                ],
            )
            x_ptr, n, gs, ps = (p["value"] for p in rec["params"])
            self.assertEqual(x_ptr, la["args"]["x_ptr"]["data_ptr"])
            self.assertEqual(n, la["args"]["n"]["value"])
            self.assertEqual((gs, ps), (0, 0))  # no scratch: NULL
            self.assertEqual(rec["entry"], "probe")
            self.assertEqual(rec["cluster"], [1, 1, 1])
            self.assertEqual(rec["block"][1:], [1, 1])
            self.assertIn(
                f".reqntid {rec['block'][0]}", (out / f"{la['spec']}.ptx").read_text()
            )
            spec = d["specs"][la["spec"]]
            self.assertEqual(rec["smem_dynamic"], spec["shared_bytes"])
            grids[inp["name"]] = rec["grid"]
        self.assertEqual(grids, {"partial": [3, 1, 1], "empty": [0, 1, 1]})

    def test_host_descriptor_scalars_and_scratch(self):
        c = case(
            self.root,
            "run_desc",
            "desc_copy",
            [
                {
                    "name": "rows96",
                    "args": {
                        "x": {"dtype": "float32", "shape": [96, 32]},
                        "out": {"dtype": "float32", "shape": [96, 32]},
                    },
                }
            ],
        )
        d, out = self.dump(c)
        (la,) = d["inputs"][0]["launches"]
        rec = la["launch"]
        spec = d["specs"][la["spec"]]
        self.assert_matches_ptx(rec, (out / spec["ptx"]).read_text())
        params = {p["name"]: p for p in rec["params"]}
        self.assertEqual(
            list(params),
            [
                "in_desc",
                "in_desc.shape.0",
                "in_desc.shape.1",
                "in_desc.stride.0",
                "in_desc.stride.1",
                "out_ptr",
                "n",
                "flag",
                "scale",
                "big",  # one == 1 is specialized as a constexpr: no parameter
                "__global_scratch",
                "__profile_scratch",
            ],
        )
        self.assertEqual(
            [(p["type"], p["size"]) for p in rec["params"][:5]],
            [("nvTmaDesc", 128), ("i32", 4), ("i32", 4), ("i64", 8), ("i64", 8)],
        )
        self.assertEqual(
            [(params[k]["type"], params[k]["size"]) for k in ("flag", "scale", "big")],
            [("u1", 1), ("fp32", 4), ("i64", 8)],  # a bool is u1
        )
        self.assertEqual(
            [params[k]["value"] for k in ("n", "flag", "scale", "big")],
            [96, True, 0.5, 2**40],
        )
        # The host descriptor: what the launcher would encode, from the snapshot plus
        # tensordesc_meta (box, swizzle, element type), not an encoded CUtensorMap.
        snap = la["args"]["in_desc"]
        (meta,) = spec["tensordesc_meta"]
        tmap = params["in_desc"]["value"]
        self.assertEqual(tmap["address"], snap["base"]["data_ptr"])
        self.assertEqual(
            (tmap["shape"], tmap["strides"]), (snap["shape"], snap["strides"])
        )
        self.assertEqual(tmap["box"], meta["block_size"])
        self.assertEqual(tmap["swizzle"], meta["swizzle"])
        self.assertEqual(tmap["elem_size"], 4)
        self.assertEqual(
            [
                params[f"in_desc.{f}.{i}"]["value"]
                for f in ("shape", "stride")
                for i in (0, 1)
            ],
            [96, 32, 32, 1],
        )
        self.assertEqual(params["out_ptr"]["value"], la["args"]["out_ptr"]["data_ptr"])
        # The in-kernel descriptor lives in the global scratch: size per program x grid.
        gs = spec["global_scratch"]
        self.assertGreater(gs["size"], 0)
        self.assertIsNone(params["__global_scratch"]["value"])
        self.assertEqual(
            params["__global_scratch"]["scratch"],
            {"bytes": 3 * gs["size"], "align": gs["align"]},
        )
        self.assertEqual(rec["grid"], [3, 1, 1])

    def test_parameters_disagreeing_with_the_ptx_fail(self):
        decls = ptx_launch.ptx_params(
            ".visible .entry k(.param .u64 .ptr .global .align 1 k_param_0, "
            ".param .b32 k_param_1)"
        )
        ours = [
            {"name": "x_ptr", "type": "*fp32", "kind": "ptr", "size": 8, "align": 8},
            {"name": "n", "type": "i64", "kind": "scalar", "size": 8, "align": 8},
        ]
        with self.assertRaisesRegex(RuntimeError, r"do not match \.entry k(.|\n)*1: n"):
            capture._check_params("k", "k", ours, decls)
        with self.assertRaisesRegex(RuntimeError, "do not match"):
            capture._check_params("k", "k", ours[:1], decls)
        ours[1].update(type="i32", size=4, align=4)
        capture._check_params("k", "k", ours, decls)
        self.assertEqual(
            [(p["i"], p["decl"]) for p in ours],
            [(i, d["decl"]) for i, d in enumerate(decls)],
        )


if __name__ == "__main__":
    unittest.main()
