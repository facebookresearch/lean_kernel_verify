# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end tests of the TTGIR frontend (dump -> to_kir) on small TLX kernels.

Each test writes a kernel module plus its host function into a temporary source tree
(the same layout an `sl cat` export has), runs the real dump (CPU host run, launch
capture, compile for sm_100 or sm_103) and translates every specialization to KIR. The
KIR is validated with frontend/kir.py (the kir.md contract check).
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

import torch

from ...frontend import kir as kir_contract
from ...frontend.ttgir import capture, dump, host_env, to_kir
from ...frontend.ttgir.host_reads import HostAliasError
from ...frontend.ttgir.slicing import SliceError

PROBE = """
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


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


def tail_view(x):
    return x[1:]  # data_ptr 4 bytes past a 16-byte boundary


def _same_config(nargs):
    SEEN_BEST.append(tuned.best_config.kwargs["BLOCK"] == nargs["BLOCK"])


SEEN_BEST = []


@triton.autotune(configs=[triton.Config({"BLOCK": 16}, pre_hook=_same_config),
                          triton.Config({"BLOCK": 32}, pre_hook=_same_config)],
                 key=["n"], reset_to_zero=["start_ptr"])
@triton.jit
def tuned(start_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.load(start_ptr) + tl.arange(0, BLOCK)  # integer metadata
    tl.store(out_ptr + offs, offs, mask=offs < n)


def run_tuned(start, out):
    tuned[(1,)](start, out, out.numel())


@triton.jit
def float_branch(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    if tl.load(x_ptr) > 0.0:  # seeded data -> control dependency
        tl.store(out_ptr + offs, offs)


def run_float_branch(x, out):
    float_branch[(1,)](x, out, BLOCK=32)


@triton.jit
def float_index(x_ptr, out_ptr):
    idx = tl.load(x_ptr).to(tl.int32)  # seeded data -> address dependency
    tl.store(out_ptr + idx, 1.0)


def run_float_index(x, out):
    float_index[(1,)](x, out)


@triton.jit
def int_index(idx_ptr, out_ptr, BLOCK: tl.constexpr):
    row = tl.load(idx_ptr + tl.program_id(0))  # integer metadata: becomes a KIR load
    offs = row * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, offs)


def run_int_index(idx, out):
    int_index[(idx.numel(),)](idx, out, BLOCK=16)


@triton.jit
def early_exit(off_ptr, out_ptr, BLOCK: tl.constexpr):
    begin = tl.load(off_ptr + tl.program_id(0))
    end = tl.load(off_ptr + tl.program_id(0) + 1)
    if end - begin == 0:  # lowers to cf.cond_br + tt.return
        return
    offs = begin + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, offs, mask=offs < end)


def run_early_exit(off, out):
    early_exit[(off.numel() - 1,)](off, out, BLOCK=16)


def run_checked(x, check, BLOCK=128):
    # The host validates x before the launch, or returns early on an empty x.
    too_long = x.numel() > 256
    if check == "raise" and too_long:
        raise ValueError("run_checked: x is too long")
    if check == "assert_async":
        torch._assert_async(torch.tensor(not too_long), "run_checked: x is too long")
    if check == "check":
        torch._check(not too_long, lambda: "run_checked: x is too long")
    if check == "torch_error" and too_long:
        torch.cat([x, torch.zeros(2, 2)])  # a failing torch op is not a validation
    if check == "skip_empty" and x.numel() == 0:
        return
    probe[(triton.cdiv(x.numel(), BLOCK),)](x, x.numel(), BLOCK=BLOCK)


def run_env(x):
    BLOCK = int(os.environ.get("LKV_TEST_BLOCK", "128"))  # a variant switch read from the environment
    probe[(triton.cdiv(x.numel(), BLOCK),)](x, x.numel(), BLOCK=BLOCK)


def run_warmup(x, BLOCK=128):
    probe.warmup(x, x.numel(), BLOCK=BLOCK, grid=(1,))  # compiles, does not launch
    probe[(triton.cdiv(x.numel(), BLOCK),)](x, x.numel(), BLOCK=BLOCK)


def run_chain(meta, buf, meta2, out, BLOCK=16):
    # Launch 0 writes buf; launch 1 reads meta2 (buf itself, or another tensor) as metadata.
    int_index[(meta.numel(),)](meta, buf, BLOCK=BLOCK)
    int_index[(2,)](buf if meta2 is None else meta2, out, BLOCK=BLOCK)


def run_derived(meta, buf, out, src, BLOCK=16):
    # Launch 1's metadata is a host copy of buf (written by launch 0) or of meta (not written).
    int_index[(meta.numel(),)](meta, buf, BLOCK=BLOCK)
    rows = (buf if src == "buf" else meta)[:2].clone()
    int_index[(2,)](rows, out, BLOCK=BLOCK)


@triton.jit
def copy_count(src_ptr, cnt_ptr):
    tl.store(cnt_ptr, tl.load(src_ptr))


@triton.jit
def fill(out_ptr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, offs)


def run_host_reads(src, out, read, how, BLOCK=16):
    # A kernel writes cnt; the host sizes the grid of fill from cnt (stale on CPU) or from src
    # (only read by the kernel): with item(), the shape of nonzero(), or in the grid callable.
    cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
    copy_count[(1,)](src, cnt)
    data = cnt if read == "cnt" else src
    if how == "grid_callable":
        fill[lambda META: (int(data.item()),)](out, BLOCK=BLOCK)
    else:
        n = torch.nonzero(torch.arange(8) < data).shape[0] if how == "nonzero" else int(data.item())
        fill[(n,)](out, BLOCK=BLOCK)


@triton.jit
def scatter_rows(ptrs_ptr, n):
    # Grouped-GEMM style: the kernel writes through an address read from a pointer array.
    dst = tl.load(ptrs_ptr).to(tl.pointer_type(tl.int32))
    tl.store(dst + tl.program_id(0), n - 1 - tl.program_id(0))


def run_cpu_alias(meta, out, write, BLOCK=16):
    # The host adjusts its own copy of the metadata (on the GPU, .cpu() copies), or only reads it.
    shifted = meta.cpu()
    if write:
        shifted -= 2
    int_index[(2,)](meta, out, BLOCK=BLOCK)


def run_ptr_array(meta, other, out, target, use, BLOCK=16):
    # scatter_rows writes meta (or another buffer) through its address; then meta is metadata
    # of int_index, or the host reads it.
    ptrs = torch.tensor([(meta if target == "meta" else other).data_ptr()], dtype=torch.int64, device="cuda")
    scatter_rows[(2,)](ptrs, 2)
    if use == "read":
        fill[(int(meta.sum().item()) + 1,)](out, BLOCK=BLOCK)
    else:
        int_index[(2,)](meta, out, BLOCK=BLOCK)
"""

# Module-level caches persist from one input of a run to the next (one process).
CACHE = """
from lkv_test_kernels import copy_count, int_index

_ROWS = {}


def run_cached(meta, out, cache, BLOCK):
    # The first call computes the rows, with a kernel or on the host, and caches them.
    rows = _ROWS.get(cache)
    if rows is None:
        rows = meta.clone()
        if cache == "kernel":
            copy_count[(1,)](meta, rows)
        _ROWS[cache] = rows
    int_index[(2,)](rows, out, BLOCK=BLOCK)
"""

# An environment read the source scan does not see (an alias of os.environ).
ENV_ALIAS = """
from os import environ as _env

import triton
from lkv_test_kernels import probe


def block():
    return int(_env.get("LKV_TEST_ALIAS_BLOCK", "128"))


def run_alias(x):
    probe[(triton.cdiv(x.numel(), block()),)](x, x.numel(), BLOCK=block())
"""


def case(
    root: Path, module: str, entry: str, kernel: str, inputs: list, **extra
) -> dict:
    return {
        "name": f"test_{kernel}",
        "src_root": str(root),
        "module": module,
        "entry": entry,
        "kernel": kernel,
        "num_sms": 2,
        "inputs": inputs,
        **extra,
    }


class TtgirFrontendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="lkv_ttgir_src_"))
        cls.out = Path(tempfile.mkdtemp(prefix="lkv_ttgir_out_"))
        for name in ("lkv_test_kernels", "lkv_test_kernels_sm103"):
            (cls.root / f"{name}.py").write_text(textwrap.dedent(PROBE))
        (cls.root / "lkv_test_cache.py").write_text(CACHE)
        (cls.root / "lkv_test_env_alias.py").write_text(ENV_ALIAS)
        tutorial = importlib.util.find_spec(
            "triton.language.extra.tlx.tutorials.blackwell_gemm_pipelined"
        ).origin
        shutil.copy(tutorial, cls.root / "lkv_tutorial_gemm.py")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)
        shutil.rmtree(cls.out)

    def translate(self, c: dict) -> dict:
        out = self.out / c["name"]
        dump.run(c, out)
        return to_kir.translate(out / "dump.json", c)

    def assert_valid(self, results: dict) -> list:
        kirs = []
        for key, r in results.items():
            if isinstance(r, Exception):
                self.fail(f"{key}: {r}")
            self.assertEqual(kir_contract.validate(r), [], key)
            kirs.append(r)
        return kirs

    def test_probe_kernel(self):
        c = case(
            self.root,
            "lkv_test_kernels",
            "run",
            "probe",
            [
                {
                    "name": "partial",
                    "args": {"x": {"dtype": "float32", "shape": [300]}},
                },
                {
                    "name": "aligned",
                    "args": {"x": {"dtype": "float32", "shape": [256]}},
                },
                {"name": "empty", "args": {"x": {"dtype": "float32", "shape": [0]}}},
                {
                    "name": "below_block",
                    "args": {"x": {"dtype": "float32", "shape": [100]}},
                },
                # A view 4 bytes into its storage: not 16-byte divisible, on the GPU as here.
                {
                    "name": "unaligned",
                    "args": {"x": {"ref": "xv"}},
                    "setup": [
                        {
                            "call": "tail_view",
                            "args": {"x": {"dtype": "float32", "shape": [301]}},
                            "bind": ["xv"],
                        }
                    ],
                },
            ],
            intended_writes={"x_ptr": "all"},
        )
        kirs = self.assert_valid(self.translate(c))
        kir = kirs[0]
        self.assertEqual(kir["target"], "sm_100")
        self.assertEqual(
            [(r["name"], r["warps"]) for r in kir["roles"]],
            [("default", 4), ("partition0", 1)],
        )
        events = [e for r in kir["roles"] for e in _events(r["body"])]
        kinds = [e["ev"] for e in events]
        self.assertEqual(kinds.count("bar_init"), 1)
        for kind in ("arrive", "wait", "gload", "gstore"):
            self.assertIn(kind, kinds)
        store = next(e for e in events if e["ev"] == "gstore")
        self.assertEqual(store["tensor"], "x_ptr")
        self.assertEqual(store["mask"]["t"], "cmp")
        self.assertEqual(store["mask"]["o"], "slt")
        self.assertTrue(store["loc"].endswith("lkv_test_kernels.py:18"), store["loc"])
        grids = {i["name"]: i["grid"][0] for k in kirs for i in k["case"]["inputs"]}
        self.assertEqual(
            grids,
            {"partial": 3, "aligned": 2, "empty": 0, "below_block": 1, "unaligned": 3},
        )
        # The specialization the GPU would derive: data_ptr and n divisible by 16 or not.
        d = json.loads((self.out / c["name"] / "dump.json").read_text())
        by_input = {
            i["name"]: d["specs"][i["launches"][0]["spec"]]["specialization"]
            for i in d["inputs"]
        }
        self.assertEqual(by_input["aligned"]["x_ptr"], ["*fp32", "D"])
        self.assertEqual(by_input["aligned"]["n"], ["i32", "D"])
        self.assertEqual(by_input["partial"]["n"], ["i32", ""])
        self.assertEqual(by_input["unaligned"]["x_ptr"], ["*fp32", ""])
        self.assertEqual(by_input["aligned"]["BLOCK"], ["constexpr", 128])

    def test_target_sm103(self):
        # Own module: a JITFunction keeps the backend (target) of its first launch.
        c = case(
            self.root,
            "lkv_test_kernels_sm103",
            "run",
            "probe",
            [
                {
                    "name": "aligned",
                    "args": {"x": {"dtype": "float32", "shape": [256]}},
                },
            ],
            intended_writes={"x_ptr": "all"},
            name="test_probe_sm103",
        )
        out = self.out / c["name"]
        d = dump.run(c, out, target="sm_103")
        self.assertEqual(
            d["target"], {"name": "sm_103", "arch": 103, "max_shared_mem": 232448}
        )
        (spec,) = d["specs"].values()
        self.assertIn('ttg.target = "cuda:103"', (out / spec["ttgir"]).read_text())
        (kir,) = self.assert_valid(to_kir.translate(out / "dump.json", c))
        self.assertEqual(kir["target"], "sm_103")

    def test_autotuner_resets_and_sets_best_config(self):
        mod = sys.modules.get("lkv_test_kernels")
        if mod is not None:
            mod.SEEN_BEST.clear()
        c = case(
            self.root,
            "lkv_test_kernels",
            "run_tuned",
            "tuned",
            [
                {
                    "name": "start7",
                    "args": {
                        "start": {"dtype": "int32", "values": [7]},
                        "out": {"dtype": "int32", "shape": [64]},
                    },
                },
            ],
            intended_writes={"out_ptr": "all"},
        )
        kirs = self.assert_valid(self.translate(c))
        self.assertEqual(len(kirs), 2)
        # Autotuner.run zeroes reset_to_zero arguments after benchmarking, before the launch.
        for kir in kirs:
            self.assertEqual(kir["case"]["inputs"][0]["metadata"]["start_ptr"], [0])
        self.assertEqual(sys.modules["lkv_test_kernels"].SEEN_BEST, [True, True])

    def test_misaligned_storage_is_refused(self):
        t = torch.frombuffer(bytearray(64), dtype=torch.uint8, offset=4)
        self.assertNotEqual(t.untyped_storage().data_ptr() % 16, 0)
        with self.assertRaisesRegex(RuntimeError, "not 16-byte aligned"):
            capture._tensor(t)

    def test_modules_must_come_from_the_export(self):
        dump.check_exported(self.root)
        stray = types.ModuleType("lkv_test_kernels.stray")
        stray.__file__ = "/elsewhere/lkv_test_kernels/stray.py"
        sys.modules[stray.__name__] = stray
        try:
            with self.assertRaisesRegex(
                RuntimeError, "lkv_test_kernels.stray was loaded from /elsewhere"
            ):
                dump.check_exported(self.root)
        finally:
            del sys.modules[stray.__name__]

    def test_tutorial_gemm(self):
        c = case(
            self.root,
            "lkv_tutorial_gemm",
            "matmul",
            "matmul_kernel_tma_pipelined_blackwell",
            [
                {
                    "name": "square",
                    "args": {
                        "a": {"dtype": "float16", "shape": [256, 256]},
                        "b": {"dtype": "float16", "shape": [256, 256]},
                    },
                },
            ],
            intended_writes={"c_ptr": "all"},
        )
        kirs = self.assert_valid(self.translate(c))
        self.assertEqual(
            len(kirs), 8
        )  # every autotune config the tuner would benchmark
        for kir in kirs:
            kinds = {e["ev"] for r in kir["roles"] for e in _events(r["body"])}
            self.assertTrue(
                {"tma_load", "expect_tx", "wait", "commit_arrive", "gstore"} <= kinds
            )
            # tl.make_tensor_descriptor with host-side extents ([M, K], [K, N]) and loads:
            # desc tensors whose shapes are expressions over the kernel params and whose
            # box is the block of the descriptor type.
            descs = {t["name"]: t for t in kir["tensors"] if t["kind"] == "desc"}
            self.assertEqual(len(descs), 2)
            for t in descs.values():
                self.assertEqual(
                    {e["name"] for e in _vars(t["shape"])} - {"M", "N", "K"}, set()
                )
            # Every barrier_expect announces exactly the bytes of the TMA loads it covers.
            for role in kir["roles"]:
                pending = {}
                for e in _events(role["body"]):
                    if e["ev"] == "expect_tx":
                        self.assertEqual(
                            pending.get(e["bar"], 0), 0, kir["case"]["name"]
                        )
                        pending[e["bar"]] = e["bytes"]["v"]
                    elif e["ev"] == "tma_load":
                        t = descs[e["desc"]]
                        pending[e["bar"]] -= t["box"][0] * t["box"][1] * t["elem_bytes"]
                self.assertEqual(
                    set(pending.values()) - {0}, set(), kir["case"]["name"]
                )
            # The TMA store of C through an in-kernel descriptor is a gstore of c_ptr, so
            # P0-3 can be stated for c_ptr.
            (store,) = [
                e
                for r in kir["roles"]
                for e in _events(r["body"])
                if e["ev"] == "gstore"
            ]
            self.assertEqual(store["tensor"], "c_ptr")
            self.assertEqual(
                kir["case"]["inputs"][0]["intended_writes"], {"c_ptr": "all"}
            )

    def test_integer_metadata_becomes_load(self):
        c = case(
            self.root,
            "lkv_test_kernels",
            "run_int_index",
            "int_index",
            [
                {
                    "name": "rows",
                    "args": {
                        "idx": {"dtype": "int32", "values": [2, 0, 1]},
                        "out": {"dtype": "int32", "shape": [48]},
                    },
                },
            ],
            intended_writes={"out_ptr": "all"},
        )
        kir = self.assert_valid(self.translate(c))[0]
        idx = next(t for t in kir["tensors"] if t["name"] == "idx_ptr")
        self.assertTrue(idx["metadata"])
        self.assertEqual(kir["case"]["inputs"][0]["metadata"]["idx_ptr"], [2, 0, 1])
        self.assertIn('"e": "load"', json.dumps(kir["roles"]))

    def test_early_exit_becomes_if(self):
        c = case(
            self.root,
            "lkv_test_kernels",
            "run_early_exit",
            "early_exit",
            [
                {
                    "name": "empty_middle",
                    "args": {
                        "off": {"dtype": "int32", "values": [0, 5, 5, 12]},
                        "out": {"dtype": "int32", "shape": [12]},
                    },
                },
            ],
            intended_writes={"out_ptr": "all"},
        )
        kir = self.assert_valid(self.translate(c))[0]
        (role,) = kir["roles"]
        branch = next(s for s in role["body"] if s["op"] == "if")
        self.assertEqual(branch["then"], [])
        self.assertEqual([e["ev"] for e in _events(branch["else"])], ["gstore"])

    def test_environment_is_pinned_to_the_case(self):
        # The shell's values are ignored and recorded; the case's env applies. So are compiler
        # switches outside TRITON_* (a triton.knobs one and one only libtriton reads).
        saved = dict(os.environ)
        try:
            inputs = [
                {"name": "n256", "args": {"x": {"dtype": "float32", "shape": [256]}}}
            ]
            for env, block in (({}, 128), ({"LKV_TEST_BLOCK": "64"}, 64)):
                with self.subTest(env=env):
                    os.environ.update(
                        {
                            "LKV_TEST_BLOCK": "64",
                            "TRITON_LKV_TEST_SWITCH": "1",  # the shell's
                            "USE_IR_LOC": "ttir",
                            "DISABLE_MMA_V5": "1",
                        }
                    )
                    c = case(
                        self.root,
                        "lkv_test_kernels",
                        "run_env",
                        "probe",
                        inputs,
                        env=env,
                        intended_writes={"x_ptr": "all"},
                        name=f"test_env_{block}",
                    )
                    d = dump.run(c, self.out / c["name"])
                    (spec,) = d["specs"].values()
                    self.assertEqual(spec["constexprs"]["BLOCK"], block)
                    self.assertEqual(
                        d["inputs"][0]["launches"][0]["grid"], [256 // block, 1, 1]
                    )
                    self.assertEqual(d["env"]["case"], env)
                    self.assertEqual(
                        {
                            k: d["env"]["unset"].get(k)
                            for k in (
                                "TRITON_LKV_TEST_SWITCH",
                                "USE_IR_LOC",
                                "DISABLE_MMA_V5",
                            )
                        },
                        {
                            "TRITON_LKV_TEST_SWITCH": "1",
                            "USE_IR_LOC": "ttir",
                            "DISABLE_MMA_V5": "1",
                        },
                    )
                    self.assertEqual("LKV_TEST_BLOCK" in d["env"]["unset"], not env)
                    self.assertIn("LKV_TEST_BLOCK", d["env"]["read_by_sources"])
                    self.assertIn("TRITON_CACHE_DIR", d["env"]["effective"])
                    self.assertEqual(d["env"]["triton_cache_invalidating"], {})
                    self.assertEqual(d["compiler"]["name"], "triton")
                    (kir,) = self.assert_valid(
                        to_kir.translate(self.out / c["name"] / "dump.json", c)
                    )
                    self.assertEqual(
                        kir["compiler"],
                        {"name": "triton", "version": d["compiler"]["version"]},
                    )
        finally:
            os.environ.clear()
            os.environ.update(saved)

    def test_rewritten_locations_are_refused(self):
        # A case env with USE_IR_LOC points every location at a scratch IR file.
        c = case(
            self.root,
            "lkv_test_kernels",
            "run",
            "probe",
            [{"name": "n256", "args": {"x": {"dtype": "float32", "shape": [256]}}}],
            intended_writes={"x_ptr": "all"},
            env={"USE_IR_LOC": "ttir"},
            name="test_ir_loc",
        )
        (err,) = self.translate(c).values()
        self.assertIsInstance(err, to_kir.KirError)
        self.assertRegex(
            str(err),
            r"has location '/\S+\.ttir:\d+', not a line of the exported sources",
        )

    def test_environment_reads_the_scan_cannot_see_are_refused(self):
        # The shell's value would be used; the check ends with the dump.
        saved = dict(os.environ)
        c = case(
            self.root,
            "lkv_test_env_alias",
            "run_alias",
            "probe",
            [{"name": "n256", "args": {"x": {"dtype": "float32", "shape": [256]}}}],
            intended_writes={"x_ptr": "all"},
            name="test_env_alias",
        )
        try:
            os.environ["LKV_TEST_ALIAS_BLOCK"] = "64"
            with self.assertRaisesRegex(
                host_env.EnvError,
                r"lkv_test_env_alias.py:9 reads environment variable "
                "LKV_TEST_ALIAS_BLOCK, which the source scan did not find",
            ):
                dump.run(c, self.out / c["name"])
            self.assertEqual(sys.modules["lkv_test_env_alias"].block(), 64)
        finally:
            os.environ.clear()
            os.environ.update(saved)

    def test_inputs_the_host_rejects_are_recorded(self):
        # An input the host refuses (a raise in the sources, a failed torch check) or does not
        # launch is recorded with the reason, and the next input still runs and is translated.
        def checked(check, first):
            inputs = [
                {
                    "name": "first",
                    "args": {
                        "x": {"dtype": "float32", "shape": [first]},
                        "check": check,
                    },
                },
                {
                    "name": "ok",
                    "args": {"x": {"dtype": "float32", "shape": [256]}, "check": check},
                },
            ]
            return case(
                self.root,
                "lkv_test_kernels",
                "run_checked",
                "probe",
                inputs,
                intended_writes={"x_ptr": "all"},
                name=f"test_checked_{check}",
            )

        too_long = r"run_checked: x is too long at lkv_test_kernels.py:\d+$"
        for check, first, error in (
            ("raise", 300, "ValueError: " + too_long),
            ("assert_async", 300, "RuntimeError: " + too_long),
            ("check", 300, "RuntimeError: " + too_long),
            ("skip_empty", 0, r"no launch of probe \(the host launched: nothing\)$"),
        ):
            with self.subTest(check=check):
                c = checked(check, first)
                d = dump.run(c, self.out / c["name"])
                rejected, ok = d["inputs"]
                self.assertRegex(rejected["host_error"], error)
                self.assertEqual(rejected["launches"], [])
                self.assertNotIn("host_error", ok)
                self.assertEqual([la["grid"] for la in ok["launches"]], [[2, 1, 1]])
                (kir,) = self.assert_valid(
                    to_kir.translate(self.out / c["name"] / "dump.json", c)
                )
                self.assertEqual([i["name"] for i in kir["case"]["inputs"]], ["ok"])
        # A torch op failing in the host code is not the host refusing the input.
        c = checked("torch_error", 300)
        with self.assertRaisesRegex(
            RuntimeError, "Tensors must have same number of dimensions"
        ):
            dump.run(c, self.out / c["name"])

    def test_warmup_is_not_a_launch(self):
        c = case(
            self.root,
            "lkv_test_kernels",
            "run_warmup",
            "probe",
            [{"name": "n256", "args": {"x": {"dtype": "float32", "shape": [256]}}}],
            intended_writes={"x_ptr": "all"},
            name="test_warmup",
        )
        d = dump.run(c, self.out / c["name"])
        self.assertEqual([la["grid"] for la in d["inputs"][0]["launches"]], [[2, 1, 1]])

    def test_metadata_written_by_an_earlier_launch_of_the_kernel_is_refused(self):
        # Launch 0 of int_index writes buf, launch 1 reads it as metadata.
        def chain(meta2):
            args = {
                "meta": {"dtype": "int32", "values": [1, 0]},
                "buf": {"dtype": "int32", "shape": [32]},
                "meta2": meta2,
                "out": {"dtype": "int32", "shape": [32]},
            }
            return case(
                self.root,
                "lkv_test_kernels",
                "run_chain",
                "int_index",
                [{"name": "two", "args": args}],
                intended_writes={"out_ptr": "all"},
                name=f"test_chain_{meta2 is None}",
            )

        (err,) = self.translate(chain(None)).values()
        self.assertIsInstance(err, to_kir.KirError)
        self.assertIn(
            "input two#1: metadata tensor idx_ptr may hold data that launch 0 of int_index (argument "
            "out_ptr) writes before this launch",
            str(err),
        )
        (kir,) = self.assert_valid(
            self.translate(chain({"dtype": "int32", "values": [0, 1]}))
        )
        self.assertEqual([i["name"] for i in kir["case"]["inputs"]], ["two#0", "two#1"])

    def test_metadata_computed_on_the_host_from_a_skipped_write_is_refused(self):
        def derived(src):
            args = {
                "meta": {"dtype": "int32", "values": [1, 0]},
                "buf": {"dtype": "int32", "shape": [32]},
                "out": {"dtype": "int32", "shape": [32]},
                "src": src,
            }
            return case(
                self.root,
                "lkv_test_kernels",
                "run_derived",
                "int_index",
                [{"name": "two", "args": args}],
                intended_writes={"out_ptr": "all"},
                name=f"test_derived_{src}",
            )

        (err,) = self.translate(derived("buf")).values()
        self.assertIn(
            "input two#1: metadata tensor idx_ptr may hold data that launch 0 of int_index (argument "
            "out_ptr) writes before this launch (directly or through host computation)",
            str(err),
        )
        self.assert_valid(self.translate(derived("meta")))

    def test_host_read_of_a_skipped_write_is_refused(self):
        # The host sizes the grid of `fill` with int(cnt.item()), with the shape of nonzero(),
        # or in the grid callable.
        def reads(read, how):
            args = {
                "src": {"dtype": "int32", "values": [4]},
                "out": {"dtype": "int32", "shape": [64]},
                "read": read,
                "how": how,
            }
            return case(
                self.root,
                "lkv_test_kernels",
                "run_host_reads",
                "fill",
                [{"name": "n64", "args": args}],
                intended_writes={"out_ptr": "all"},
                name=f"test_host_reads_{read}_{how}",
            )

        for how, op in (
            ("item", "TensorBase.item"),
            ("nonzero", "aten.nonzero.default"),
            ("grid_callable", "TensorBase.item"),
        ):
            with self.subTest(how=how):
                (err,) = self.translate(reads("cnt", how)).values()
                self.assertIsInstance(err, to_kir.KirError)
                self.assertRegex(
                    str(err),
                    rf"input n64: before this launch the host reads data \({op} at "
                    r"lkv_test_kernels.py:\d+\) that launch 0 of copy_count \(argument cnt_ptr\) "
                    "may write",
                )
                (kir,) = self.assert_valid(
                    self.translate(reads("src", how))
                )  # copy_count only reads src
                self.assertEqual(kir["case"]["inputs"][0]["grid"], [4, 1, 1])

    def test_data_cached_from_an_earlier_input_is_checked(self):
        # The inputs of a run share the process; the second input reuses rows a kernel
        # computed in the first (refused), or rows the host computed (accepted).
        def cached(cache):
            inputs = [
                {
                    "name": name,
                    "args": {
                        "meta": {"dtype": "int32", "values": [1, 0]},
                        "out": {"dtype": "int32", "shape": [32]},
                        "cache": cache,
                        "BLOCK": block,
                    },
                }
                for name, block in (("first", 16), ("second", 8))
            ]
            return case(
                self.root,
                "lkv_test_cache",
                "run_cached",
                "int_index",
                inputs,
                intended_writes={"out_ptr": "unchecked"},
                name=f"test_cached_{cache}",
            )

        errors = sorted(str(e) for e in self.translate(cached("kernel")).values())
        self.assertEqual(len(errors), 2)
        for err, name in zip(errors, ("first", "second")):
            self.assertIn(
                f"input {name}: metadata tensor idx_ptr may hold data that launch 0 of copy_count "
                "(argument cnt_ptr) writes before this launch",
                err,
            )
        self.assertEqual(len(self.assert_valid(self.translate(cached("host")))), 2)

    def test_writes_to_what_a_device_move_returned_are_refused(self):
        # The CPU run aliases what the GPU copies.
        def cpu_alias(write):
            args = {
                "meta": {"dtype": "int32", "values": [2, 3]},
                "out": {"dtype": "int32", "shape": [64]},
                "write": write,
            }
            return case(
                self.root,
                "lkv_test_kernels",
                "run_cpu_alias",
                "int_index",
                [{"name": "two", "args": args}],
                intended_writes={"out_ptr": "unchecked"},
                name=f"test_cpu_alias_{write}",
            )

        with self.assertRaisesRegex(
            HostAliasError,
            r"lkv_test_kernels.py:\d+ writes \(aten.sub_.Tensor\) data that "
            r"lkv_test_kernels.py:\d+ moved between host and device",
        ):
            dump.run(cpu_alias(True), self.out / "test_cpu_alias_True")
        (kir,) = self.assert_valid(self.translate(cpu_alias(False)))
        self.assertEqual(kir["case"]["inputs"][0]["metadata"]["idx_ptr"], [2, 3])

    def test_writes_through_an_address_argument_are_tracked(self):
        # scatter_rows writes meta through a pointer array, then meta is metadata, or the host
        # reads it; a pointer to another buffer leaves meta fresh.
        def ptr_array(target, use):
            args = {
                "meta": {"dtype": "int32", "values": [0, 1]},
                "other": {"dtype": "int32", "shape": [4]},
                "out": {"dtype": "int32", "shape": [64]},
                "target": target,
                "use": use,
            }
            kernel = "fill" if use == "read" else "int_index"
            return case(
                self.root,
                "lkv_test_kernels",
                "run_ptr_array",
                kernel,
                [{"name": "two", "args": args}],
                intended_writes={"out_ptr": "unchecked"},
                name=f"test_ptr_array_{target}_{use}",
            )

        writer = r"launch 0 of scatter_rows \(argument ptrs_ptr\)"
        for use, msg in (
            (
                "metadata",
                f"input two: metadata tensor idx_ptr may hold data that {writer} writes",
            ),
            (
                "read",
                r"input two: before this launch the host reads data \(TensorBase.item at "
                rf"lkv_test_kernels.py:\d+\) that {writer} may write",
            ),
        ):
            with self.subTest(use=use):
                (err,) = self.translate(ptr_array("meta", use)).values()
                self.assertIsInstance(err, to_kir.KirError)
                self.assertRegex(str(err), msg)
                self.assert_valid(self.translate(ptr_array("other", use)))

    def test_slicing_rejects_float_branch(self):
        c = case(
            self.root,
            "lkv_test_kernels",
            "run_float_branch",
            "float_branch",
            [
                {
                    "name": "one",
                    "args": {
                        "x": {"dtype": "float32", "shape": [1]},
                        "out": {"dtype": "int32", "shape": [32]},
                    },
                },
            ],
        )
        (err,) = self.translate(c).values()
        self.assertIsInstance(err, SliceError)
        self.assertIn("arith.cmpf", str(err))
        self.assertIn("control operand of `scf.if`", str(err))
        self.assertIn("lkv_test_kernels.py:", str(err))

    def test_slicing_rejects_float_address(self):
        c = case(
            self.root,
            "lkv_test_kernels",
            "run_float_index",
            "float_index",
            [
                {
                    "name": "one",
                    "args": {
                        "x": {"dtype": "float32", "shape": [1]},
                        "out": {"dtype": "float32", "shape": [8]},
                    },
                },
            ],
        )
        (err,) = self.translate(c).values()
        self.assertIsInstance(err, SliceError)
        self.assertIn("arith.fptosi", str(err))
        self.assertIn("pointer of `tt.store`", str(err))


def _events(stmts: list):
    for s in stmts:
        if s["op"] == "event":
            yield s
        for key in ("body", "then", "else"):
            yield from _events(s.get(key, []))


def _vars(exprs: list):
    for e in exprs:
        if e.get("e") == "var":
            yield e
        yield from _vars([e[k] for k in ("a", "b", "c") if k in e])


if __name__ == "__main__":
    unittest.main()
