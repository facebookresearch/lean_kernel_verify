# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests of the TTGIR -> KIR translation on handwritten TTGIR (no compiler, no GPU).

Each test builds a small module in the printed form Triton emits, translates it with
to_kir.build against a minimal dump record, and checks the KIR or the hard error. The
sources of the op semantics are cited in references/ir_ops.md.
"""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import torch

from ...frontend import kir as kir_contract
from ...frontend.ttgir import host_env, kir_build as K, provenance, to_kir, var_names
from ...frontend.ttgir.export_src import imported_modules, package_of
from ...frontend.ttgir.host_reads import HostAliasError, HostReads
from ...frontend.ttgir.mlir_text import parse, walk
from ...frontend.ttgir.slicing import SliceError

HEADER = """#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, fp4Padded = true}>
#smem = #ttg.shared_memory
#loc = loc("/src/k.py":10:0)
#loc1 = loc("/src/k.py":11:0)
#loc2 = loc("/src/k.py":12:0)
#loc3 = loc("/src/k.py":13:0)
#loc4 = loc("/src/k.py":14:0)
"""
ATTRS = (
    '"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", '
    '"ttg.threads-per-warp" = 32 : i32'
)
BAR = "!ttg.memdesc<1xi64, #shared1, #smem, mutable>"
PRELUDE = f"""    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %c1_i32 = arith.constant 1 : i32 loc(#loc1)
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %c128_i32 = arith.constant 128 : i32 loc(#loc1)
    %c2_i64 = arith.constant 2 : i64 loc(#loc1)
    %true = arith.constant true loc(#loc1)
    %cst = arith.constant 1.000000e+00 : f32 loc(#loc1)
    %bars = ttg.local_alloc : () -> {BAR} loc(#loc1)
    %b0 = ttg.memdesc_index %bars[%c0_i32] : {BAR} -> {BAR} loc(#loc1)
    ttng.init_barrier %b0, 1 : {BAR} loc(#loc1)
"""


def ttgir(args: dict, body: str, attrs: str = ATTRS) -> str:
    sig = ", ".join(f'%{n}: {t} loc("{n}"(#loc))' for n, t in args.items())
    return (
        HEADER + f"module attributes {{{attrs}}} {{\n"
        f"  tt.func public @k({sig}) attributes {{noinline = false}} {{\n"
        + PRELUDE
        + body
        + "    tt.return loc(#loc)\n  } loc(#loc)\n} loc(#loc)\n"
    )


def tensor(lo: int, shape: list, elem: int = 4, values: list | None = None) -> dict:
    n = 1
    for d in shape:
        n *= d
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    rec = {
        "kind": "tensor",
        "dtype": "int32",
        "elem_bytes": elem,
        "shape": shape,
        "strides": strides,
        "storage_offset": 0,
        "storage": [lo, lo + n * elem],
        "data_ptr": lo,
    }
    if values is not None:
        rec["values"] = values
    return rec


def scalar(v: int) -> dict:
    return {"kind": "scalar", "value": v}


def dump_of(
    snap: dict, target: str = "sm_100", grid=(1, 1, 1), stale=None, stale_reads=()
) -> dict:
    """A dump.json record with one input "i" whose launch (index 1) of spec "s" has `snap`."""
    arch = {"sm_100": 100, "sm_103": 103}[target]
    launch = {
        "spec": "s",
        "index": 1,
        "grid": list(grid),
        "args": snap,
        "stale": stale or {},
    }
    return {
        "case": "t",
        "kernel": "k",
        "src_root": "/src",
        "num_sms": 2,
        "target": {"name": target, "arch": arch, "max_shared_mem": 232448},
        "compiler": {"name": "triton", "version": "3.9.0"},
        "specs": {"s": {"constexprs": {}, "shared_bytes": 0}},
        "inputs": [
            {"name": "i", "launches": [launch], "stale_reads": list(stale_reads)}
        ],
    }


def build(text: str, snap: dict, target: str = "sm_100", **case) -> dict:
    """to_kir.build for one launch (input "i", grid 1) of the module `text`."""
    return to_kir.build(
        text, dump_of(snap, target), "s", {"inputs": [{"name": "i"}], **case}
    )


def events(stmts: list):
    for s in stmts:
        if s["op"] == "event":
            yield s
        for key in ("body", "then", "else"):
            yield from events(s.get(key, []))


def store_offsets(stmts: list, env: dict) -> list:
    """The gstore offsets (scalar ones) of a role body run on `env`: a reference run of the
    KIR's lets, loops and ifs with K.evaluate."""
    out = []
    for s in stmts:
        if s["op"] == "let":
            env[s["var"]] = K.evaluate(s["expr"], env)
        elif s["op"] == "for":
            ub, step = K.evaluate(s["ub"], env), K.evaluate(s["step"], env)
            for i in range(K.evaluate(s["lb"], env), ub, step):
                env[s["var"]] = i
                out += store_offsets(s["body"], env)
        elif s["op"] == "while":
            while K.evaluate(s["cond"], env):
                out += store_offsets(s["body"], env)
        elif s["op"] == "if":
            branch = "then" if K.evaluate(s["cond"], env) else "else"
            out += store_offsets(s[branch], env)
        elif s.get("ev") == "gstore":
            out.append(K.evaluate(s["offsets"]["a"], env))
    return out


def lets(stmts: list) -> dict:
    out = {}
    for s in stmts:
        if s["op"] == "let":
            out[s["var"]] = s["expr"]
        for key in ("body", "then", "else"):
            out.update(lets(s.get(key, [])))
    return out


def addr_marked(e: dict) -> bool:
    return bool(e.get("addr")) or any(
        addr_marked(e[k]) for k in ("a", "b", "c") if k in e
    )


def texpr_nodes(t: dict):
    """The TExpr nodes of a TExpr (not the scalar Exprs under its splats)."""
    yield t
    for k in ("a", "b", "c"):
        if isinstance(t.get(k), dict) and "t" in t[k]:
            yield from texpr_nodes(t[k])


# An in-kernel descriptor over %a_ptr (f16, extents [M, K], row stride K): the tensormap
# box {BOX} (innermost first) and the block 64x128 of its type.
DEVICE_DESC = """    %ks = arith.extsi %K : i32 to i64 loc(#loc2)
    %kb = arith.muli %ks, %c2_i64 : i64 loc(#loc2)
    %scr = ttg.global_scratch_alloc {{alignment = 128 : i32, nbytes = 128 : i32}} : !tt.ptr<i8> loc(#loc2)
    ttng.tensormap_create %scr, %a_ptr, [{BOX}], [%K, %M], [%kb], [%c1_i32, %c1_i32] {{elem_type = {ETYPE} : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32}} : (!tt.ptr<i8>, !tt.ptr<f16>, i32, i32, i32, i32, i64, i32, i32) -> () loc(#loc2)
    %desc = ttng.reinterpret_tensor_descriptor %scr : !tt.ptr<i8> to !tt.tensordesc<64x128xf16, {ENC}> loc(#loc2)
"""
TMA_LOAD = """    %buf = ttg.local_alloc : () -> !ttg.memdesc<{TILE}xf16, {SENC}, #smem, mutable> loc(#loc3)
    ttng.barrier_expect %b0, 16384, %true : {BAR} loc(#loc3)
    ttng.async_tma_copy_global_to_local %desc[%c0_i32, %c0_i32] %buf, %b0, %true : !tt.tensordesc<64x128xf16, {ENC}>, {BAR} -> !ttg.memdesc<{TILE}xf16, {SENC}, #smem, mutable> loc(#loc3)
    ttng.wait_barrier %b0, %c0_i32 : {BAR} loc(#loc4)
"""
TMA_STORE = """    %cbuf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable> loc(#loc3)
    ttng.async_tma_copy_local_to_global %desc[%c0_i32, %c0_i32] %cbuf : !tt.tensordesc<64x128xf16, #shared>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable> loc(#loc3)
"""
GEMM_ARGS = {"a_ptr": "!tt.ptr<f16>", "M": "i32", "K": "i32"}
GEMM_SNAP = {
    "a_ptr": {**tensor(1 << 20, [256, 256], 2), "dtype": "float16"},
    "M": scalar(256),
    "K": scalar(256),
}


def tma(
    box="%c64_i32, %c64_i32",
    etype=6,
    enc="#shared",
    tile="64x128",
    senc=None,
    store=False,
) -> str:
    body = DEVICE_DESC.format(BOX=box, ETYPE=etype, ENC=enc)
    if store:
        return body + TMA_STORE
    return body + TMA_LOAD.format(TILE=tile, SENC=senc or enc, ENC=enc, BAR=BAR)


class TmaFootprintTest(unittest.TestCase):
    """The KIR box of a TMA op is the block of its descriptor type."""

    def test_box_is_the_block_and_bytes_match_expect(self):
        kir = build(ttgir(GEMM_ARGS, tma()), GEMM_SNAP)
        self.assertEqual(kir_contract.validate(kir), [])
        desc = next(t for t in kir["tensors"] if t["kind"] == "desc")
        self.assertEqual(
            desc["box"], [64, 128]
        )  # the tensormap box is [64, 64] (128B swizzle)
        evs = list(events(kir["roles"][0]["body"]))
        load = next(e for e in evs if e["ev"] == "tma_load")
        expect = next(e for e in evs if e["ev"] == "expect_tx")
        self.assertEqual(64 * 128 * desc["elem_bytes"], expect["bytes"]["v"])
        self.assertEqual(load["desc"], desc["name"])

    def test_tensormap_box_not_tiling_the_block_is_refused(self):
        with self.assertRaisesRegex(
            to_kir.KirError, r"tensormap box \[64, 128\].*does not tile"
        ):
            build(ttgir(GEMM_ARGS, tma(box="%c128_i32, %c64_i32")), GEMM_SNAP)

    def test_tile_other_than_the_block_is_refused(self):
        with self.assertRaisesRegex(
            to_kir.KirError, r"tile \[64, 64\] is not the descriptor block"
        ):
            build(ttgir(GEMM_ARGS, tma(tile="64x64")), GEMM_SNAP)

    def test_encoding_mismatch_is_refused(self):
        with self.assertRaisesRegex(
            to_kir.KirError, "differs from the shared-memory encoding"
        ):
            build(ttgir(GEMM_ARGS, tma(senc="#shared2")), GEMM_SNAP)

    def test_host_descriptor_block_shape_must_match_its_type(self):
        args = {
            "d": "!tt.tensordesc<64x128xf16, #shared>",
            "d.shape.0": "i32",
            "d.shape.1": "i32",
            "d.stride.0": "i64",
            "d.stride.1": "i64",
        }
        body = TMA_LOAD.format(
            TILE="64x128", SENC="#shared", ENC="#shared", BAR=BAR
        ).replace("%desc", "%d")
        base = {**tensor(1 << 20, [256, 256], 2), "dtype": "float16"}
        snap = {
            "d": {
                "kind": "desc",
                "dtype": "float16",
                "elem_bytes": 2,
                "shape": [256, 256],
                "strides": [256, 1],
                "block_shape": [64, 128],
                "padding": "zero",
                "base": base,
            }
        }
        kir = build(ttgir(args, body), snap)
        self.assertEqual(
            next(t for t in kir["tensors"] if t["name"] == "d")["box"], [64, 128]
        )
        snap["d"]["block_shape"] = [64, 64]
        with self.assertRaisesRegex(
            to_kir.KirError, "host descriptor d .* does not match its type"
        ):
            build(ttgir(args, body), snap)


HOST_DESC_ARGS = {
    "d": "!tt.tensordesc<64x128xf16, #shared>",
    "d.shape.0": "i32",
    "d.shape.1": "i32",
    "d.stride.0": "i64",
    "d.stride.1": "i64",
}
# A TMA store at row 300 through host descriptor %d.
HOST_DESC_STORE = TMA_STORE.replace(
    "%desc[%c0_i32, %c0_i32]", "%d[%c300_i32, %c0_i32]"
).replace(
    "    %cbuf", "    %c300_i32 = arith.constant 300 : i32 loc(#loc3)\n    %cbuf", 1
)


def host_desc(shape: list, strides: list, base_shape: list, base_strides: list) -> dict:
    base = {
        **tensor(1 << 20, [256, 256], 2),
        "dtype": "float16",
        "shape": base_shape,
        "strides": base_strides,
    }
    return {
        "d": {
            "kind": "desc",
            "dtype": "float16",
            "elem_bytes": 2,
            "shape": shape,
            "strides": strides,
            "block_shape": [64, 128],
            "padding": "zero",
            "base": base,
        }
    }


class TmaExtentTest(unittest.TestCase):
    """The extents of a descriptor lie inside the tensor behind it."""

    def store(self, *dims) -> dict:
        return build(
            ttgir(HOST_DESC_ARGS, HOST_DESC_STORE),
            host_desc(*dims),
            intended_writes={"d": "unchecked"},
        )

    def test_host_descriptor_beyond_its_base_is_refused(self):
        # 512 rows declared over a 256-row tensor, a store at row 300.
        with self.assertRaisesRegex(
            to_kir.KirError,
            r"input i: TMA descriptor d \(shape \[512, 256\], strides "
            r"\[256, 1\]\) has in-extent elements outside its base tensor .* element offset 65536",
        ):
            self.store([512, 256], [256, 1], [256, 256], [256, 1])
        self.assertEqual(
            kir_contract.validate(
                self.store([256, 256], [256, 1], [256, 256], [256, 1])
            ),
            [],
        )

    def test_extents_are_compared_element_by_element(self):
        # A column slice of a [256, 256] storage: its padding is outside the base tensor.
        self.store([256, 128], [256, 1], [256, 128], [256, 1])
        with self.assertRaisesRegex(
            to_kir.KirError, "outside its base tensor .* element offset 128"
        ):
            self.store([256, 192], [256, 1], [256, 128], [256, 1])
        # A 2-D descriptor over a 3-D tensor with the same elements (merged dims).
        self.store([256, 256], [256, 1], [64, 4, 256], [1024, 256, 1])
        with self.assertRaisesRegex(to_kir.KirError, "element offset 65536"):
            self.store([257, 256], [256, 1], [64, 4, 256], [1024, 256, 1])

    def test_in_kernel_descriptor_loads_are_checked_against_the_base_pointer(self):
        snap = {**GEMM_SNAP, "M": scalar(512)}
        with self.assertRaisesRegex(
            to_kir.KirError,
            r"TMA descriptor scr \(shape \[512, 256\], strides \[256, 1\]\) "
            "has in-extent elements outside a_ptr",
        ):
            build(ttgir(GEMM_ARGS, tma()), snap)
        # Stores through it are footprint gstores on a_ptr (Layer 2 checks every element).
        build(ttgir(GEMM_ARGS, tma(store=True)), snap, intended_writes={"a_ptr": "all"})

    def test_in_kernel_descriptor_extents_are_unsigned(self):
        # The tensormap global_dim is a b32 that TMA reads as unsigned, so M = -64 is
        # 2^32 - 64 rows (no clipping), not an empty descriptor.
        with self.assertRaisesRegex(
            to_kir.KirError, r"TMA descriptor scr \(shape \[4294967232, 256\]"
        ):
            build(ttgir(GEMM_ARGS, tma()), {**GEMM_SNAP, "M": scalar(-64)})


class DescriptorTypeTest(unittest.TestCase):
    """fp4Padded and tensormap element types."""

    def test_fp4_padded_is_refused(self):
        with self.assertRaisesRegex(to_kir.KirError, "fp4Padded"):
            build(ttgir(GEMM_ARGS, tma(enc="#shared3")), GEMM_SNAP)

    def test_packed_elem_type_is_refused(self):
        with self.assertRaisesRegex(to_kir.KirError, "elem_type 13"):
            build(ttgir(GEMM_ARGS, tma(etype=13)), GEMM_SNAP)

    def test_elem_type_size_must_match_the_base(self):
        with self.assertRaisesRegex(to_kir.KirError, "elem_type 7 has 4-byte elements"):
            build(ttgir(GEMM_ARGS, tma(etype=7)), GEMM_SNAP)
        build(ttgir(GEMM_ARGS, tma(etype=10)), GEMM_SNAP)  # bf16: same size as f16


class DeviceDescriptorStoreTest(unittest.TestCase):
    """Stores through in-kernel descriptors are global stores of the base pointer."""

    def test_store_becomes_masked_gstore_on_base(self):
        kir = build(
            ttgir(GEMM_ARGS, tma(store=True)),
            GEMM_SNAP,
            intended_writes={"a_ptr": "all"},
        )
        self.assertEqual(kir_contract.validate(kir), [])
        self.assertFalse([t for t in kir["tensors"] if t["kind"] == "desc"])
        (store,) = [e for e in events(kir["roles"][0]["body"]) if e["ev"] != "bar_init"]
        self.assertEqual((store["ev"], store["tensor"]), ("gstore", "a_ptr"))
        self.assertEqual(store["offsets"]["shape"], [64, 128])
        self.assertEqual(store["mask"]["t"], "bin")
        a = next(t for t in kir["tensors"] if t["name"] == "a_ptr")
        self.assertEqual(a["role"], "output")
        self.assertEqual(kir["case"]["inputs"][0]["intended_writes"], {"a_ptr": "all"})
        # The extents in the footprint mask are zero-extended (unsigned).
        extents = [
            n["a"]
            for n in texpr_nodes(store["mask"])
            if n["t"] == "splat" and n["a"]["e"] == "ext" and n["a"]["a"]["e"] == "var"
        ]
        self.assertEqual(
            [(e["a"]["name"], e["signed"]) for e in extents],
            [("M", False), ("K", False)],
        )
        self.assertEqual(K.evaluate(extents[0], {"M": -64}), 2**32 - 64)


class MetadataTest(unittest.TestCase):
    """Metadata written in the same launch, and masked metadata loads."""

    ARGS = {"i_ptr": "!tt.ptr<i32>", "o_ptr": "!tt.ptr<i32>", "n": "i32"}
    READ = f"""    %pid = tt.get_program_id x : i32 loc(#loc2)
    %ip = tt.addptr %i_ptr, %pid : !tt.ptr<i32>, i32 loc(#loc2)
    %inb = arith.cmpi slt, %pid, %n : i32 loc(#loc2)
    %v = tt.load %ip, %inb, %c0_i32 : !tt.ptr<i32> loc(#loc2)
    %b1 = ttg.memdesc_index %bars[%v] : {BAR} -> {BAR} loc(#loc3)
    ttng.arrive_barrier %b1, 1 : {BAR} loc(#loc3)
"""
    WRITE_O = """    %op = tt.addptr %o_ptr, %c0_i32 : !tt.ptr<i32>, i32 loc(#loc4)
    tt.store %op, %n : !tt.ptr<i32> loc(#loc4)
"""

    def snap(self, o_lo: int) -> dict:
        return {
            "i_ptr": tensor(4096, [4], values=[0, 0, 0, 0]),
            "o_ptr": tensor(o_lo, [4]),
            "n": scalar(4),
        }

    def test_masked_load_guards_its_index(self):
        kir = build(ttgir(self.ARGS, self.READ + self.WRITE_O), self.snap(8192))
        self.assertEqual(kir_contract.validate(kir), [])
        v = lets(kir["roles"][0]["body"])["v"]
        self.assertEqual((v["e"], v["a"]["e"]), ("select", "load"))
        (index,) = v["a"]["index"]
        self.assertEqual(
            (index["e"], index["c"], index["b"]),
            ("select", v["c"], {"e": "const", "v": 0, "w": 64}),
        )

    def test_metadata_written_by_the_kernel_is_refused(self):
        write_i = "    tt.store %i_ptr, %n : !tt.ptr<i32> loc(#loc4)\n"
        with self.assertRaisesRegex(
            to_kir.KirError, r"metadata tensor\(s\) \['i_ptr'\] are written by k"
        ):
            build(ttgir(self.ARGS, write_i + self.READ), self.snap(8192))

    def test_metadata_sharing_storage_with_a_written_tensor_is_refused(self):
        with self.assertRaisesRegex(to_kir.KirError, "i_ptr shares storage with o_ptr"):
            build(ttgir(self.ARGS, self.READ + self.WRITE_O), self.snap(4096 + 8))

    def test_strided_metadata_is_refused(self):
        # Contents are recorded in index order, loads index by offset.
        snap = self.snap(8192)
        snap["i_ptr"] = {**snap["i_ptr"], "strides": [2], "storage": [4096, 4096 + 32]}
        with self.assertRaisesRegex(
            to_kir.KirError, r"metadata tensor i_ptr has strides \[2\]"
        ):
            build(ttgir(self.ARGS, self.READ + self.WRITE_O), snap)

    WRITER = [{"launch": 0, "kernel": "producer", "param": "out"}]

    def translate(self, **dump) -> dict:
        return to_kir.build(
            ttgir(self.ARGS, self.READ + self.WRITE_O),
            dump_of(self.snap(8192), **dump),
            "s",
            {"inputs": [{"name": "i"}]},
        )

    def test_metadata_an_earlier_launch_may_write_is_refused(self):
        # dump.json "stale" (any earlier launch, through host code too).
        with self.assertRaisesRegex(
            to_kir.KirError,
            "metadata tensor i_ptr may hold data that launch 0 of producer "
            r"\(argument out\) writes before this launch",
        ):
            self.translate(stale={"i_ptr": self.WRITER})
        self.translate(stale={"o_ptr": self.WRITER})  # not metadata

    def test_host_read_of_stale_data_before_the_launch_is_refused(self):
        read = {
            "op": "TensorBase.item",
            "loc": "host.py:9",
            "after": 1,
            "writers": self.WRITER,
        }
        with self.assertRaisesRegex(
            to_kir.KirError,
            r"before this launch the host reads data \(TensorBase.item at "
            r"host.py:9\) that launch 0 of producer \(argument out\) may write",
        ):
            self.translate(stale_reads=[read])
        self.translate(
            stale_reads=[{**read, "after": 2}]
        )  # read after the launch (index 1)


class StructureTest(unittest.TestCase):
    """Module attributes (num-warps, threads-per-warp, target), event locations, elision
    flags, loop and scratch-buffer constructs, CLC queries."""

    def test_missing_num_warps_is_an_error(self):
        attrs = ATTRS.replace('"ttg.num-warps" = 4 : i32, ', "")
        with self.assertRaisesRegex(to_kir.KirError, "ttg.num-warps"):
            build(ttgir({}, "", attrs), {})

    def test_threads_per_warp_must_be_32(self):
        with self.assertRaisesRegex(to_kir.KirError, "threads-per-warp"):
            build(ttgir({}, "", ATTRS.replace("= 32 : i32", "= 64 : i32")), {})

    def test_target_is_recorded_and_checked(self):
        self.assertEqual(build(ttgir({}, ""), {})["target"], "sm_100")
        self.assertEqual(
            build(ttgir({}, "", ATTRS.replace("cuda:100", "cuda:103")), {}, "sm_103")[
                "target"
            ],
            "sm_103",
        )
        with self.assertRaisesRegex(to_kir.KirError, "dump is for sm_103"):
            build(ttgir({}, ""), {}, "sm_103")

    def test_events_outside_the_exported_sources_are_refused(self):
        # E.g. USE_IR_LOC points every location at a scratch IR file.
        text = ttgir(GEMM_ARGS, tma(store=True)).replace(
            '"/src/k.py"', '"/tmp/lkv_triton_cache_x/k.ttir"'
        )
        with self.assertRaisesRegex(
            to_kir.KirError,
            r"bar_init in role default has location "
            r"'/tmp/lkv_triton_cache_x/k.ttir:11', not a line of the exported sources",
        ):
            build(text, GEMM_SNAP, intended_writes={"a_ptr": "all"})

    def test_elision_flags_reach_the_kir_case(self):
        dump = dump_of({})
        dump["specs"]["s"]["constexprs"] = {"FULL_TILES": True, "BLOCK": 64}
        kir = to_kir.build(
            ttgir({}, ""),
            dump,
            "s",
            {"inputs": [{"name": "i"}], "elision_flags": ["FULL_TILES"]},
        )
        self.assertEqual(kir_contract.validate(kir), [])
        self.assertEqual(
            (kir["case"]["elision_flags"], kir["case"]["inputs"][0]["flags"]),
            (["FULL_TILES"], {"FULL_TILES": True}),
        )
        self.assertNotIn("elision_flags", build(ttgir({}, ""), {})["case"])
        with self.assertRaisesRegex(
            to_kir.KirError,
            r"case `elision_flags` names \['BLOCK'\], which are not "
            "boolean constexprs of k",
        ):
            to_kir.build(
                ttgir({}, ""),
                dump,
                "s",
                {"inputs": [{"name": "i"}], "elision_flags": ["BLOCK"]},
            )

    def test_unsigned_for_is_refused(self):
        body = f"""    scf.for unsigned %iv = %c0_i32 to %n step %c1_i32  : i32 {{
      ttng.arrive_barrier %b0, 1 : {BAR} loc(#loc3)
    }} loc(#loc3)
"""
        with self.assertRaisesRegex(to_kir.KirError, "unsignedCmp"):
            build(ttgir({"n": "i32"}, body), {"n": scalar(2)})
        build(ttgir({"n": "i32"}, body.replace("unsigned ", "")), {"n": scalar(2)})

    def test_tensor_carried_by_a_loop_is_named(self):
        body = """    %r0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked> loc(#loc2)
    %c16 = arith.constant dense<16> : tensor<16xi32, #blocked> loc(#loc2)
    %ps = tt.splat %x_ptr : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #blocked> loc(#loc2)
    %res = scf.for %iv = %c0_i32 to %n step %c1_i32 iter_args(%offs = %r0) -> (tensor<16xi32, #blocked>)  : i32 {
      %pp = tt.addptr %ps, %offs : tensor<16x!tt.ptr<i32>, #blocked>, tensor<16xi32, #blocked> loc(#loc3)
      tt.store %pp, %offs : tensor<16x!tt.ptr<i32>, #blocked> loc(#loc3)
      %o2 = arith.addi %offs, %c16 : tensor<16xi32, #blocked> loc(#loc3)
      scf.yield %o2 : tensor<16xi32, #blocked> loc(#loc3)
    } loc(#loc3)
"""
        snap = {"x_ptr": tensor(4096, [64]), "n": scalar(4)}
        with self.assertRaisesRegex(
            to_kir.KirError,
            r"%offs of type tensor<16xi32, #blocked> carried by `scf.for` at "
            "fbcode/k.py:13",
        ):
            build(
                ttgir({"x_ptr": "!tt.ptr<i32>", "n": "i32"}, body),
                snap,
                intended_writes={"x_ptr": "all"},
            )
        scalar_body = """    %res = scf.for %iv = %c0_i32 to %n step %c1_i32 iter_args(%o = %c0_i32) -> (i32)  : i32 {
      %p = tt.addptr %x_ptr, %o : !tt.ptr<i32>, i32 loc(#loc3)
      tt.store %p, %o : !tt.ptr<i32> loc(#loc3)
      %o2 = arith.addi %o, %c1_i32 : i32 loc(#loc3)
      scf.yield %o2 : i32 loc(#loc3)
    } loc(#loc3)
"""
        build(
            ttgir({"x_ptr": "!tt.ptr<i32>", "n": "i32"}, scalar_body),
            snap,
            intended_writes={"x_ptr": "all"},
        )

    def test_second_tensormap_into_one_scratch_is_refused(self):
        create = next(
            line for line in DEVICE_DESC.splitlines() if "tensormap_create" in line
        )
        body = tma().replace(
            create.format(BOX="%c64_i32, %c64_i32", ETYPE=6) + "\n",
            (create.format(BOX="%c64_i32, %c64_i32", ETYPE=6) + "\n") * 2,
        )
        with self.assertRaisesRegex(
            to_kir.KirError,
            r"\(ttgir line 28\) writes a second TMA descriptor into %scr "
            r"\(first at `ttng.tensormap_create` at fbcode/k.py:12 \(ttgir line 27\)\)",
        ):
            build(ttgir(GEMM_ARGS, body), GEMM_SNAP)

    def test_one_name_for_two_values_in_sibling_partitions_is_refused(self):
        # The printer may reuse an SSA name in sibling regions, but KIR barrier and
        # tensor names are kernel-wide: partition1's own %bar (or %scr) is not
        # partition0's.
        caps = {"arg3": "!tt.ptr<f16>", "arg4": "i32", "arg5": "i32", "arg6": BAR}
        consts = "".join(PRELUDE.splitlines(True)[:6])  # isolated: no outer constants
        own = f"""      %bar = ttg.local_alloc : () -> {BAR} loc(#loc2)
      ttng.init_barrier %bar, 1 : {BAR} loc(#loc2)
"""
        load = tma()
        for old, new in zip(("%a_ptr", "%M", "%K", "%b0"), caps):
            load = load.replace(old, "%" + new)
        cases = {
            "barrier": (
                own + f"      ttng.arrive_barrier %bar, 1 : {BAR} loc(#loc3)\n",
                own + f"      ttng.wait_barrier %bar, %c0_i32 : {BAR} loc(#loc4)\n",
                "%bar",
            ),
            "descriptor": (load, load, "%scr"),
        }
        sig = ", ".join(f"%{n}: {t} loc(unknown)" for n, t in caps.items())
        for kind, (p0, p1, name) in cases.items():
            for p1_name in (name, name + "_0"):
                parts = [
                    f"    partition{i}({sig}) num_warps(1) {{\n{consts}{p}"
                    "      ttg.warp_return loc(#loc1)\n    }"
                    for i, p in enumerate((p0, p1.replace(name, p1_name)))
                ]
                body = (
                    "    ttg.warp_specialize(%a_ptr, %M, %K, %b0)\n    default {\n"
                    "      ttg.warp_yield loc(#loc1)\n    }\n"
                    + "\n".join(parts)
                    + f" : ({', '.join(caps.values())}) -> () loc(#loc1)\n"
                )
                with self.subTest(kind=kind, p1_name=p1_name):
                    if p1_name != name:
                        kir = build(ttgir(GEMM_ARGS, body), GEMM_SNAP)
                        self.assertEqual(kir_contract.validate(kir), [])
                        names = {x["name"] for x in kir["barriers"] + kir["tensors"]}
                        self.assertLessEqual({name[1:], p1_name[1:]}, names)
                        continue
                    with self.assertRaisesRegex(
                        to_kir.KirError, rf"{name} also names the {kind} of .* line"
                    ):
                        build(ttgir(GEMM_ARGS, body), GEMM_SNAP)

    def test_clc_query_in_a_branch_without_events_advances_the_counter(self):
        body = f"""    %resp = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable> loc(#loc2)
    %pos = arith.cmpi sgt, %n, %c0_i32 : i32 loc(#loc2)
    scf.if %pos {{
      %x0, %y0, %z0 = ttng.clc_query_cancel %resp : (!ttg.memdesc<1xi64, #shared1, #smem, mutable>) -> (i32, i32, i32) loc(#loc3)
    }} loc(#loc3)
    %x1, %y1, %z1 = ttng.clc_query_cancel %resp : (!ttg.memdesc<1xi64, #shared1, #smem, mutable>) -> (i32, i32, i32) loc(#loc3)
    %b1 = ttg.memdesc_index %bars[%x1] : {BAR} -> {BAR} loc(#loc3)
    ttng.arrive_barrier %b1, 1 : {BAR} loc(#loc4)
"""
        kir = build(ttgir({"n": "i32"}, body), {"n": scalar(1)})
        (role,) = kir["roles"]
        launched = next(
            s for s in role["body"] if s["op"] == "if"
        )  # `if __clc_launched[pid.x] != 0`
        branch = next(
            s for s in launched["then"] if s["op"] == "if" and s["cond"]["e"] == "var"
        )
        self.assertEqual([s["var"] for s in branch["then"]], ["__clc_k"])

    def test_loop_carried_pointer_is_named(self):
        body = """    %r = scf.for %iv = %c0_i32 to %n step %c1_i32 iter_args(%p = %x_ptr) -> (!tt.ptr<f32>)  : i32 {
      tt.store %p, %cst : !tt.ptr<f32> loc(#loc3)
      %p2 = tt.addptr %p, %c1_i32 : !tt.ptr<f32>, i32 loc(#loc3)
      scf.yield %p2 : !tt.ptr<f32> loc(#loc3)
    } loc(#loc3)
"""
        with self.assertRaisesRegex(
            to_kir.KirError, r"pointer %p carried by `scf.for` .*ptrs \+= BLOCK"
        ):
            build(
                ttgir({"x_ptr": "!tt.ptr<f32>", "n": "i32"}, body),
                {"x_ptr": tensor(4096, [8]), "n": scalar(2)},
            )

    def test_loop_result_survives_a_sibling_loop_reusing_its_arg_names(self):
        # The printer names the args of both loops %arg10, %arg11, and the second loop's
        # %arg11 is live: its pre-let rebinds arg11 while %r1 (= 4 * 64) is the second
        # loop's init at another position and is stored to again after it.
        body = """    %r1 = scf.for %arg10 = %c0_i32 to %n step %c1_i32 iter_args(%arg11 = %c0_i32) -> (i32)  : i32 {
      %t = arith.addi %arg11, %c64_i32 : i32 loc(#loc2)
      scf.yield %t : i32 loc(#loc2)
    } loc(#loc2)
    %r2:2 = scf.for %arg10 = %c0_i32 to %n step %c1_i32 iter_args(%arg11 = %c0_i32, %arg12 = %r1) -> (i32, i32)  : i32 {
      %o = arith.addi %arg11, %arg12 : i32 loc(#loc3)
      %p = tt.addptr %x_ptr, %o : !tt.ptr<i32>, i32 loc(#loc3)
      tt.store %p, %c0_i32 : !tt.ptr<i32> loc(#loc3)
      %t = arith.addi %arg11, %c1_i32 : i32 loc(#loc3)
      scf.yield %t, %arg12 : i32, i32 loc(#loc3)
    } loc(#loc3)
    %q = tt.addptr %x_ptr, %r1 : !tt.ptr<i32>, i32 loc(#loc4)
    tt.store %q, %c0_i32 : !tt.ptr<i32> loc(#loc4)
"""
        kir = build(
            ttgir({"x_ptr": "!tt.ptr<i32>", "n": "i32"}, body),
            {"x_ptr": tensor(4096, [64]), "n": scalar(4)},
            intended_writes={"x_ptr": "unchecked"},
        )
        self.assertEqual(kir_contract.validate(kir), [])
        stores = store_offsets(kir["roles"][0]["body"], {"n": 4})
        self.assertEqual(stores, [256, 257, 258, 259, 256])  # x_ptr has 64 elements

    def test_while_forwarding_its_args_permuted_reads_their_old_values(self):
        # The after args are printed with the before args' names; scf.condition swaps
        # them.
        body = """    %cm1 = arith.constant -1 : i32 loc(#loc1)
    %r:2 = scf.while (%arg10 = %cm1, %arg11 = %c64_i32) : (i32, i32) -> (i32, i32) {
      %c = arith.cmpi slt, %arg10, %n : i32 loc(#loc2)
      scf.condition(%c) %arg11, %arg10 : i32, i32 loc(#loc2)
    } do {
    ^bb0(%arg10: i32 loc(#loc3), %arg11: i32 loc(#loc3)):
      %p = tt.addptr %x_ptr, %arg11 : !tt.ptr<i32>, i32 loc(#loc3)
      tt.store %p, %c0_i32 : !tt.ptr<i32> loc(#loc3)
      %q = tt.addptr %x_ptr, %arg10 : !tt.ptr<i32>, i32 loc(#loc4)
      tt.store %q, %c0_i32 : !tt.ptr<i32> loc(#loc4)
      %t = arith.addi %arg11, %c1_i32 : i32 loc(#loc3)
      scf.yield %t, %arg10 : i32, i32 loc(#loc3)
    } loc(#loc2)
"""
        kir = build(
            ttgir({"x_ptr": "!tt.ptr<i32>", "n": "i32"}, body),
            {"x_ptr": tensor(4096, [128]), "n": scalar(1)},
            intended_writes={"x_ptr": "unchecked"},
        )
        self.assertEqual(kir_contract.validate(kir), [])
        stores = store_offsets(kir["roles"][0]["body"], {"n": 1})
        self.assertEqual(stores, [-1, 64, 0, 64])  # x_ptr[-1] is out of bounds

    def test_while_without_carried_values(self):
        # A loop carrying nothing prints without an init list (`scf.while : () -> ()`),
        # e.g. a spin loop on a flag; one carrying nothing but producing a result too.
        spin = f"""    scf.while : () -> () {{
      %v = tt.load %flag : !tt.ptr<i32> loc(#loc2)
      %c = arith.cmpi slt, %v, %n : i32 loc(#loc2)
      scf.condition(%c) loc(#loc2)
    }} do {{
      ttng.arrive_barrier %b0, 1 : {BAR} loc(#loc3)
      scf.yield loc(#loc3)
    }} loc(#loc2)
"""
        result = """    %r = scf.while : () -> i32 {
      %v = tt.load %flag : !tt.ptr<i32> loc(#loc2)
      %c = arith.cmpi slt, %v, %n : i32 loc(#loc2)
      scf.condition(%c) %v : i32 loc(#loc2)
    } do {
    ^bb0(%arg10: i32 loc(#loc3)):
      %p = tt.addptr %x_ptr, %arg10 : !tt.ptr<i32>, i32 loc(#loc3)
      tt.store %p, %c0_i32 : !tt.ptr<i32> loc(#loc3)
      scf.yield loc(#loc3)
    } loc(#loc2)
    %q = tt.addptr %x_ptr, %r : !tt.ptr<i32>, i32 loc(#loc4)
    tt.store %q, %c0_i32 : !tt.ptr<i32> loc(#loc4)
"""
        args = {"x_ptr": "!tt.ptr<i32>", "flag": "!tt.ptr<i32>", "n": "i32"}
        snap = {
            "x_ptr": tensor(8192, [64]),
            "flag": tensor(4096, [1], values=[5]),
            "n": scalar(3),
        }
        for name, body, results, after_args in (
            ("spin", spin, 0, 0),
            ("result", result, 1, 1),
        ):
            with self.subTest(name):
                text = ttgir(args, body)
                (loop,) = [
                    op
                    for op in walk(parse(text).func.regions[0])
                    if op.name == "scf.while"
                ]
                self.assertEqual(
                    (len(loop.operands), len(loop.results), len(loop.regions[0].args)),
                    (0, results, 0),
                )
                self.assertEqual(len(loop.regions[1].args), after_args)
                writes = {"x_ptr": "unchecked"} if results else {}
                kir = build(text, snap, intended_writes=writes)
                self.assertEqual(kir_contract.validate(kir), [])
                body_stmts = kir["roles"][0]["body"]
                (w,) = [s for s in body_stmts if s["op"] == "while"]
                self.assertEqual(w["cond"], {"e": "var", "name": "c", "w": 1})
                # The before region runs before the loop and again after each body.
                self.assertEqual([s["var"] for s in w["body"][-2:]], ["v", "c"])
                ev = "arrive" if name == "spin" else "gstore"
                self.assertEqual([s["ev"] for s in events(w["body"])], [ev, "gload"])


class VariableNamesTest(unittest.TestCase):
    """KIR variables of SSA values whose printed names repeat in sibling regions
    (var_names): a value keeps its printed name unless sharing that variable with
    another value's would not be exact; then it is named `<name>@<ttgir line>`."""

    ARGS = {"x_ptr": "!tt.ptr<i32>", "n": "i32"}
    SNAP = {"x_ptr": tensor(8192, [128]), "n": scalar(3)}

    @staticmethod
    def line_of(text: str, op: str) -> int:
        return next(i for i, ln in enumerate(text.splitlines(), 1) if op in ln)

    def check(self, body: str, n: int) -> tuple[str, dict, list]:
        text = ttgir(self.ARGS, body)
        kir = build(text, self.SNAP, intended_writes={"x_ptr": "unchecked"})
        self.assertEqual(kir_contract.validate(kir), [])
        return text, kir, store_offsets(kir["roles"][0]["body"], {"n": n})

    def test_while_regions_reusing_a_number_with_another_width(self):
        # The printer numbers both regions from the same counter: the condition %0 of the
        # before region (i1, assigned before the loop) and the after region's %0 (i32).
        body = """    %r:2 = scf.while (%arg10 = %c0_i32, %arg11 = %c0_i32) : (i32, i32) -> (i32, i32) {
      %0 = arith.cmpi slt, %arg10, %n : i32 loc(#loc2)
      scf.condition(%0) %arg10, %arg11 : i32, i32 loc(#loc2)
    } do {
    ^bb0(%arg10: i32 loc(#loc3), %arg11: i32 loc(#loc3)):
      %first = arith.cmpi slt, %arg10, %c1_i32 : i32 loc(#loc3)
      %0 = scf.if %first -> (i32) {
        scf.yield %c64_i32 : i32 loc(#loc3)
      } else {
        scf.yield %arg11 : i32 loc(#loc3)
      } loc(#loc3)
      %p = tt.addptr %x_ptr, %0 : !tt.ptr<i32>, i32 loc(#loc4)
      tt.store %p, %c0_i32 : !tt.ptr<i32> loc(#loc4)
      %t = arith.addi %arg10, %c1_i32 : i32 loc(#loc3)
      %u = arith.addi %0, %c1_i32 : i32 loc(#loc3)
      scf.yield %t, %u : i32, i32 loc(#loc3)
    } loc(#loc2)
"""
        text, kir, stores = self.check(body, 3)
        self.assertEqual(stores, [64, 65, 66])
        names = lets(kir["roles"][0]["body"])
        line = self.line_of(text, "%0 = scf.if")
        self.assertEqual(names["0"]["e"], "cmp")  # the condition keeps its name
        self.assertEqual(names[f"0@{line}"]["w"], 32)
        # The after args share the before args' variables (exact: the carry reassigns
        # the before args before any read).
        self.assertNotIn(f"arg10@{self.line_of(text, '^bb0')}", names)

    def test_sibling_loops_carrying_a_name_with_another_width(self):
        # Both loops carry %arg11 (the printer numbers sibling bodies alike), assigned
        # before each loop: an i1 in the first, an i32 in the second.
        body = """    %r1 = scf.for %arg10 = %c0_i32 to %n step %c1_i32 iter_args(%arg11 = %true) -> (i1)  : i32 {
      %t = arith.xori %arg11, %true : i1 loc(#loc2)
      scf.yield %t : i1 loc(#loc2)
    } loc(#loc2)
    %r2 = scf.for %arg10 = %c0_i32 to %n step %c1_i32 iter_args(%arg11 = %c0_i32) -> (i32)  : i32 {
      %p = tt.addptr %x_ptr, %arg11 : !tt.ptr<i32>, i32 loc(#loc3)
      tt.store %p, %c0_i32 : !tt.ptr<i32> loc(#loc3)
      %t = arith.addi %arg11, %c1_i32 : i32 loc(#loc3)
      scf.yield %t : i32 loc(#loc3)
    } loc(#loc3)
    %q = arith.select %r1, %c64_i32, %c128_i32 : i32 loc(#loc4)
    %pq = tt.addptr %x_ptr, %q : !tt.ptr<i32>, i32 loc(#loc4)
    tt.store %pq, %c0_i32 : !tt.ptr<i32> loc(#loc4)
"""
        text, kir, stores = self.check(body, 2)
        self.assertEqual(stores, [0, 1, 64])  # r1 = true xor true xor true
        names = lets(kir["roles"][0]["body"])
        line = self.line_of(text, "%r2 = scf.for")
        self.assertEqual((names["arg11"]["w"], names[f"arg11@{line}"]["w"]), (1, 32))
        loops = [s for s in kir["roles"][0]["body"] if s["op"] == "for"]
        self.assertEqual([s["var"] for s in loops], ["arg10", "arg10"])

    def test_exact_sharing_keeps_the_printed_names(self):
        # Same widths, and the before region's %0 is dead in the after region (it is
        # assigned again at the end of the body before the next test): one variable, as
        # in the KIR of every TLX CLC loop.
        body = """    %r = scf.while (%arg10 = %c0_i32) : (i32) -> i32 {
      %0 = arith.cmpi slt, %arg10, %n : i32 loc(#loc2)
      scf.condition(%0) %arg10 : i32 loc(#loc2)
    } do {
    ^bb0(%arg10: i32 loc(#loc3)):
      %0 = arith.cmpi slt, %arg10, %c1_i32 : i32 loc(#loc3)
      %o = arith.select %0, %c64_i32, %arg10 : i32 loc(#loc3)
      %p = tt.addptr %x_ptr, %o : !tt.ptr<i32>, i32 loc(#loc4)
      tt.store %p, %c0_i32 : !tt.ptr<i32> loc(#loc4)
      %t = arith.addi %arg10, %c1_i32 : i32 loc(#loc3)
      scf.yield %t : i32 loc(#loc3)
    } loc(#loc2)
"""
        _, kir, stores = self.check(body, 3)
        self.assertEqual(stores, [64, 1, 2])
        self.assertFalse(any("@" in x for x in lets(kir["roles"][0]["body"])))

    def test_name_role(self):
        t0, t1 = var_names.token(0), var_names.token(1)

        def let(x, e):
            return {"op": "let", "var": x, "expr": e}

        def use(x):
            return {"op": "assert", "cond": K.var(x, 1), "msg": ""}

        c1, c32 = K.const(1, 1), K.const(1, 32)
        loop = {"op": "for", "var": t1, "lb": c32, "ub": c32, "step": c32, "body": []}
        cases = {
            # t0 is read after t1 is assigned: sharing would clobber it.
            "live": ([let(t0, c1), let(t1, c1), use(t0)], ["x", "x@9"]),
            # t0 is never read after t1 is assigned: one variable.
            "dead": ([let(t0, c1), use(t0), let(t1, c1), use(t1)], ["x", "x"]),
            # Visible at once with other widths.
            "width": ([let(t0, c1), let(t1, c32)], ["x", "x@9"]),
            # Never visible at once (t0 is local to the branch): any widths.
            "scopes": (
                [
                    {"op": "if", "cond": c1, "then": [let(t0, c1)], "else": []},
                    let(t1, c32),
                ],
                ["x", "x"],
            ),
            # A `for` variable is read-only: never shared with a visible variable.
            "for": ([let(t0, K.const(1, 32)), dict(loop)], ["x", "x@9"]),
            # Taken by a param.
            "param": ([let(t1, c1)], [None, "x@9"]),
            # t0 is read again on the next trip, after t1 is assigned: the back edge
            # keeps it live.
            "loop": (
                [
                    let(t0, c1),
                    {
                        "op": "for",
                        "var": "i",
                        "lb": c32,
                        "ub": K.const(3, 32),
                        "step": c32,
                        "body": [use(t0), let(t1, c1)],
                    },
                ],
                ["x", "x@9"],
            ),
            "while": (
                [
                    let(t0, c1),
                    {
                        "op": "while",
                        "cond": c1,
                        "max_iters": 3,
                        "body": [use(t0), let(t1, c1)],
                    },
                ],
                ["x", "x@9"],
            ),
            # A work field named like a nested statement list is read too.
            "work_field": (
                [
                    let(t0, c32),
                    let(t1, c32),
                    {"op": "event", "ev": "work", "fields": {"body": K.var(t0, 32)}},
                ],
                ["x", "x@9"],
            ),
        }

        def pairs(out, body):  # (named statement, tokenized statement)
            for s, s0 in zip(out, body):
                yield s, s0
                for k in ("body", "then", "else"):
                    yield from pairs(s.get(k, []), s0.get(k, []))

        for name, (body, want) in cases.items():
            with self.subTest(name):
                reserved = {"x"} if name == "param" else set()
                out = var_names.name_role(body, ["x", "x"], [5, 9], reserved)
                got = {t0: None, t1: None}
                for s, s0 in pairs(out, body):
                    if s0.get("var") in got:
                        got[s0["var"]] = s["var"]
                self.assertEqual([got[t0], got[t1]], want)


class ErrorsTest(unittest.TestCase):
    """Malformed or unsupported constructs are KirErrors naming the op and its loc."""

    def test_unreadable_range_is_a_kir_error(self):
        body = "    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32, #blocked> loc(#loc3)\n"
        body += "    %ps = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>, #blocked> loc(#loc3)\n"
        body += "    %pp = tt.addptr %ps, %r : tensor<4x!tt.ptr<f32>, #blocked>, tensor<4xi32, #blocked> loc(#loc3)\n"
        body += "    %cv = arith.constant dense<1.000000e+00> : tensor<4xf32, #blocked> loc(#loc3)\n"
        body += "    tt.store %pp, %cv : tensor<4x!tt.ptr<f32>, #blocked> loc(#loc4)\n"
        with self.assertRaisesRegex(
            to_kir.KirError,
            r"cannot read the range of `tt.make_range` at fbcode/k.py:13",
        ):
            build(ttgir({"x_ptr": "!tt.ptr<f32>"}, body), {"x_ptr": tensor(4096, [4])})

    def test_unexpected_operand_is_a_kir_error_with_loc(self):
        body = "    %w = tt.get_program_id w : i32 loc(#loc3)\n"
        body += f"    %b1 = ttg.memdesc_index %bars[%w] : {BAR} -> {BAR} loc(#loc3)\n"
        body += f"    ttng.arrive_barrier %b1, 1 : {BAR} loc(#loc4)\n"
        with self.assertRaisesRegex(
            to_kir.KirError,
            r"cannot translate `tt.get_program_id` at fbcode/k.py:13 .*ValueError",
        ):
            build(ttgir({}, body), {})

    def test_pointer_element_types(self):
        body = "    tt.store %x_ptr, %v : !tt.ptr<{T}> loc(#loc4)\n"
        kir = build(
            ttgir({"x_ptr": "!tt.ptr<ui8>", "v": "i32"}, body.format(T="ui8")),
            {"x_ptr": tensor(4096, [4], 1), "v": scalar(0)},
        )
        self.assertEqual(
            next(t for t in kir["tensors"] if t["name"] == "x_ptr")["elem_bytes"], 1
        )
        with self.assertRaisesRegex(to_kir.KirError, "f4E2M1FN .not whole bytes"):
            build(
                ttgir(
                    {"x_ptr": "!tt.ptr<f4E2M1FN>", "v": "i32"},
                    body.format(T="f4E2M1FN"),
                ),
                {"x_ptr": tensor(4096, [4], 1), "v": scalar(0)},
            )


class LocAndAddrTest(unittest.TestCase):
    """Locs of generic-form region ops, and `addr` marks on address paths only."""

    REDUCE = f"""    %tm = ttng.tmem_alloc : () -> !ttg.memdesc<1xi32, #shared1, #smem, mutable> loc(#loc2)
    %tv = ttng.tmem_load %tm : !ttg.memdesc<1xi32, #shared1, #smem, mutable> -> tensor<1xi32, #blocked> loc(#loc2)
    %s = "tt.reduce"(%tv) <{{axis = 0 : i32}}> ({{
    ^bb0(%a: i32 loc(#loc2), %b: i32 loc(#loc2)):
      %m = arith.addi %a, %b : i32 loc(#loc2)
      tt.reduce.return %m : i32 loc(#loc2)
    }}) : (tensor<1xi32, #blocked>) -> i32 loc(#loc3)
    %c = arith.cmpi sgt, %s, %c0_i32 : i32 loc(#loc3)
    scf.if %c {{
      ttng.arrive_barrier %b0, 1 : {BAR} loc(#loc4)
    }} loc(#loc4)
"""

    def test_generic_region_op_has_its_loc(self):
        module = parse(ttgir({}, self.REDUCE), "/src")
        (reduce,) = [
            op for op in walk(module.func.regions[0]) if op.name == "tt.reduce"
        ]
        self.assertEqual(reduce.loc, "fbcode/k.py:13")
        with self.assertRaisesRegex(
            SliceError, r"dropped op `tt.reduce` at fbcode/k.py:13"
        ):
            build(ttgir({}, self.REDUCE), {})

    def test_approx_divf_is_dropped(self):
        # Newer Triton builds emit tt.approx_divf for a float division: float math, kept
        # off the slice like tt.precise_divf.
        div = "    %q = tt.approx_divf %cst, %cst : f32 loc(#loc2)\n"
        kir = build(ttgir({}, div), {})
        self.assertEqual(
            [e["ev"] for e in events(kir["roles"][0]["body"])], ["bar_init"]
        )
        decides = f"""    %i = arith.fptosi %q : f32 to i32 loc(#loc3)
    %c = arith.cmpi sgt, %i, %c0_i32 : i32 loc(#loc3)
    scf.if %c {{
      ttng.arrive_barrier %b0, 1 : {BAR} loc(#loc4)
    }} loc(#loc4)
"""
        with self.assertRaisesRegex(SliceError, "dropped op"):
            build(ttgir({}, div + decides), {})

    def test_addr_marks_address_arithmetic_only(self):
        body = """    %pid = tt.get_program_id x : i32 loc(#loc2)
    %off = arith.muli %pid, %c64_i32 : i32 loc(#loc2)
    %lim = arith.muli %n, %c64_i32 : i32 loc(#loc2)
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked> loc(#loc2)
    %s = tt.splat %off : i32 -> tensor<64xi32, #blocked> loc(#loc2)
    %o = arith.addi %s, %r : tensor<64xi32, #blocked> loc(#loc2)
    %ls = tt.splat %lim : i32 -> tensor<64xi32, #blocked> loc(#loc2)
    %m = arith.cmpi slt, %o, %ls : tensor<64xi32, #blocked> loc(#loc2)
    %ps = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked> loc(#loc3)
    %pp = tt.addptr %ps, %o : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked> loc(#loc3)
    %cv = arith.constant dense<1.000000e+00> : tensor<64xf32, #blocked> loc(#loc3)
    tt.store %pp, %cv, %m : tensor<64x!tt.ptr<f32>, #blocked> loc(#loc4)
"""
        kir = build(
            ttgir({"x_ptr": "!tt.ptr<f32>", "n": "i32"}, body),
            {"x_ptr": tensor(4096, [128]), "n": scalar(2)},
        )
        bound = lets(kir["roles"][0]["body"])
        self.assertTrue(addr_marked(bound["off"]))
        self.assertFalse(addr_marked(bound["lim"]))
        (store,) = [e for e in events(kir["roles"][0]["body"]) if e["ev"] == "gstore"]
        self.assertTrue(store["offsets"].get("addr"))


class ProvenanceTest(unittest.TestCase):
    """The write set of side-effecting asm: its stores through pointer operands, else
    unknown."""

    ASM = """    %r = tt.make_range {{end = 4 : i32, start = 0 : i32}} : tensor<4xi32, #blocked> loc(#loc3)
    %ys = tt.splat %y_ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #blocked> loc(#loc3)
    %v = tt.elementwise_inline_asm "{TEXT}" {{constraints = "=r,l,r", packed_element = 1 : i32, pure = {PURE}}} %ys, %r : tensor<4x!tt.ptr<i32>, #blocked>, tensor<4xi32, #blocked> -> tensor<4xi32, #blocked> loc(#loc3)
    %p = tt.addptr %x_ptr, %c0_i32 : !tt.ptr<i32>, i32 loc(#loc4)
    tt.store %p, %c0_i32 : !tt.ptr<i32> loc(#loc4)
"""
    ARGS = {"x_ptr": "!tt.ptr<i32>", "y_ptr": "!tt.ptr<i32>"}

    def written(self, text: str, pure: str = "false"):
        return provenance.written_params(
            ttgir(self.ARGS, self.ASM.format(TEXT=text, PURE=pure))
        )

    def test_side_effecting_inline_asm(self):
        store = "{ .reg .pred p; setp.ne.b32 p, $2, 0; @p st.global.b32 [$1+4], $2; mov.u32 $0, 0; }"
        self.assertEqual(self.written(store, "true"), {"x_ptr"})
        self.assertEqual(self.written(store), {"x_ptr", "y_ptr"})
        self.assertEqual(self.written("mov.u32 $0, $2;"), {"x_ptr"})  # no memory write
        self.assertEqual(
            self.written("st.shared.b32 [$2], $2; mov.u32 $0, 0;"), {"x_ptr"}
        )
        for unknown in ("st.global.b32 [$2], $2; mov.u32 $0, 0;",  # not a pointer operand
                        "cp.async.bulk.global.shared::cta.bulk_group [$1], [$2], 16;",
                        "membar.gl; mov.u32 $0, $2;",
                        "ld.global.u64 $1, [$1]; st.global.b32 [$1], $2; mov.u32 $0, 0;"):  # fmt: skip
            self.assertIsNone(self.written(unknown), unknown)  # (the last rewrites $1)
        # an MLIR tab escape between instructions
        self.assertEqual(self.written("st.global.b32 [$1], $2;\\0A\\09mov.u32 $0, 0;"),
                         {"x_ptr", "y_ptr"})  # fmt: skip

    TWO = """    %xs = tt.splat %x_ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #blocked> loc(#loc3)
    %ys = tt.splat %y_ptr : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>, #blocked> loc(#loc3)
    %r = tt.make_range {{end = 4 : i32, start = 0 : i32}} : tensor<4xi32, #blocked> loc(#loc3)
    %h = arith.trunci %r : tensor<4xi32, #blocked> to tensor<4xi16, #blocked> loc(#loc3)
    %a = arith.extsi %r : tensor<4xi32, #blocked> to tensor<4xi64, #blocked> loc(#loc3)
    %v = tt.elementwise_inline_asm "{TEXT}" {{constraints = "{CONS}", packed_element = {N} : i32, pure = false}} {OPS} : {TYPES} -> tensor<4xi32, #blocked> loc(#loc3)
"""

    def test_operand_registers(self):
        # Triton packs the outputs, then each operand: a pointer takes packed_element
        # registers; a tied (matching) constraint makes the write set unknown
        ptr, i32 = "tensor<4x!tt.ptr<i32>, #blocked>", "tensor<4xi32, #blocked>"

        def written(text, cons, n, ops):
            body = self.TWO.format(TEXT=text, CONS=cons, N=n, OPS=", ".join(o for o, _ in ops),
                                   TYPES=", ".join(t for _, t in ops))  # fmt: skip
            return provenance.written_params(ttgir(self.ARGS, body))

        two = [("%xs", ptr), ("%ys", ptr)]
        self.assertEqual(written("st.global.b32 [$3], 0; mov.u32 $0, 0;", "=r,=r,l,l,l,l", 2, two),
                         {"x_ptr"})  # fmt: skip
        self.assertEqual(written("st.global.b32 [$4], 0; mov.u32 $0, 0;", "=r,=r,l,l,l,l", 2, two),
                         {"y_ptr"})  # fmt: skip
        mixed = [("%r", i32), ("%ys", ptr)]
        self.assertEqual(written("st.global.b32 [$2], $1; mov.u32 $0, 0;", "=r,r,l", 1, mixed),
                         {"y_ptr"})  # fmt: skip
        self.assertIsNone(
            written("st.global.b32 [$1], $2; mov.u32 $0, 0;", "=r,r,l", 1, mixed)
        )
        self.assertIsNone(
            written("st.global.b32 [$2], 0; mov.u32 $0, 0;", "=r,l,0", 1, two)
        )
        # a 32-element row store (pack 32): 8 outputs, 16 registers of i16 values, the pointer's 32,
        # a 32-bit mask's 32; the stores go through $24, the pointer's first register
        st = ("{ .reg .pred p; setp.ne.b32 p, $56, 0; "
              "@p st.global.v8.b32 [$24], {$8, $9, $10, $11, $12, $13, $14, $15}; "
              "@p st.global.v8.b32 [$24+32], {$16, $17, $18, $19, $20, $21, $22, $23}; "
              + " ".join("mov.b32 $%d, 0;" % k for k in range(8)) + " }")  # fmt: skip
        cons = ",".join(["=r"] * 8 + ["r"] * 16 + ["l"] * 32 + ["r"] * 32)
        b16 = [("%h", "tensor<4xi16, #blocked>"), ("%ys", ptr), ("%r", i32)]
        self.assertEqual(written(st, cons, 32, b16), {"y_ptr"})
        # the i64 address $2 is no pointer operand's register (x_ptr is tied to output 0)
        tied = [("%a", "tensor<4xi64, #blocked>"), ("%xs", ptr)]
        self.assertIsNone(written("atom.global.exch.b32 $1, [$2], 1; mov.u64 $0, $3;",
                                  "=l,=r,l,0", 1, tied))  # fmt: skip


class GlobalSyncInputTest(unittest.TestCase):
    """KIR change 18 input fields of a launch: its co-resident capacity, the case's
    metadata_contents and the intended writes of tensors a KIR does not write."""

    def test_capacity_needs_one_cta_per_sm(self):
        dump = {"num_sms": 6, "target": {"max_shared_mem": 232448}}
        self.assertEqual(
            to_kir._capacity(dump, {"shared_bytes": 200000}), {"capacity": {"ctas": 6}}
        )
        self.assertEqual(to_kir._capacity(dump, {"shared_bytes": 100000}), {})
        self.assertEqual(to_kir._capacity(dump, {"shared_bytes": 0}), {})

    def test_given_contents(self):
        case = {"inputs": [{"name": "a", "metadata_contents": {"T": [1, 2]}}, {"name": "b"}],
                "metadata_contents": {"T": [7]}}  # fmt: skip
        self.assertEqual(
            to_kir._given_contents(case, {"name": "a"}, "T", [4]), [1, 2, 0, 0]
        )
        self.assertEqual(to_kir._given_contents(case, {"name": "b"}, "T", [2]), [7, 0])
        self.assertIsNone(to_kir._given_contents(case, {"name": "b"}, "U", [2]))
        with self.assertRaisesRegex(to_kir.KirError, "has 2 elements, the tensor 1"):
            to_kir._given_contents(case, {"name": "a"}, "T", [1])

    def test_unchecked_writes_of_other_tensors_are_dropped(self):
        case = {"inputs": [{"name": "a", "intended_writes": {"o": "unchecked", "w": "unchecked",
                                                             "x": "all"}}]}  # fmt: skip
        tensors = [{"name": "o", "role": "output"}, {"name": "w", "role": "input"}]
        self.assertEqual(
            to_kir._intended(case, {"name": "a"}, tensors),
            {"o": "unchecked", "x": "all"},
        )


class EvaluateTest(unittest.TestCase):
    """kir_build.evaluate: host-side extents in the exact BitVec semantics of kir.md."""

    def test_bitvec_semantics(self):
        c32, c64 = (lambda v: K.const(v, 32)), (lambda v: K.const(v, 64))
        cases = [
            (K.bin_("sdiv", c32(-7), c32(2), 32), -3), (K.bin_("srem", c32(-7), c32(2), 32), -1),
            (K.bin_("udiv", c32(-8), c32(2), 32), 2**31 - 4),
            (K.bin_("add", c32(2**31 - 1), c32(1), 32), -(2**31)), (K.bin_("shl", c32(1), c32(32), 32), 0),
            (K.bin_("ashr", c32(-8), c32(40), 32), -1), (K.bin_("lshr", c32(-1), c32(28), 32), 15),
            (K.bin_("umin", c32(-1), c32(3), 32), 3), (K.bin_("smin", c32(-1), c32(3), 32), -1),
            (K.trunc(c64(2**32 + 5), 32), 5), (K.ext(c32(-1), 64, False), 2**32 - 1), (K.ext(c32(-1), 64, True), -1),
            (K.select(K.cmp("ult", c32(-1), c32(1)), c32(10), c32(20)), 20),
            (K.bin_("sdiv", K.ext(K.var("s", 32), 64, True), c64(2), 64), 384),
        ]  # fmt: skip
        for e, want in cases:
            self.assertEqual(K.evaluate(e, {"s": 768}), want, e)
        with self.assertRaisesRegex(ValueError, "division by zero"):
            K.evaluate(K.bin_("sdiv", c32(1), K.var("z", 32), 32), {"z": 0})


class HostEnvTest(unittest.TestCase):
    """Environment pinning (host_env.py)."""

    SRC = """import os
from os import environ, getenv
FLAG = "LKV_T_CONST"
a = os.environ.get("LKV_T_GET", "0")
b = os.getenv(FLAG)
c = environ["LKV_T_ITEM"] if "LKV_T_IN" in os.environ else getenv("LKV_T_GETENV")
os.environ["LKV_T_STORE"] = "1"
"""

    def root(self, src: str) -> Path:
        d = Path(tempfile.mkdtemp(prefix="lkv_env_test_"))
        (d / "pkg").mkdir()
        (d / "pkg" / "m.py").write_text(src)
        return d

    def test_reads_of_the_sources_are_found(self):
        self.assertEqual(
            host_env.source_reads([self.root(self.SRC)]),
            {"LKV_T_CONST", "LKV_T_GET", "LKV_T_ITEM", "LKV_T_IN", "LKV_T_GETENV"},
        )
        with self.assertRaisesRegex(
            host_env.EnvError,
            r"cannot tell which environment variable is read at .*m.py:3",
        ):
            host_env.source_reads(
                [self.root("import os\nname = 'X'\nv = os.environ.get(name.lower())\n")]
            )

    def test_pin_drops_shell_values_unless_the_case_sets_them(self):
        saved = dict(os.environ)
        try:
            os.environ.update(
                {
                    "LKV_T_GET": "shell",
                    "TRITON_LKV_T": "1",
                    "MLIR_LKV_T": "1",
                    "LKV_T_OTHER": "kept",
                }
            )
            pinned = host_env.pin(
                {"LKV_T_CONST": 64, "MLIR_LKV_T": "2"},
                [self.root(self.SRC)],
                {"CUTE_DSL_LKV_T": "own"},
            )
            self.assertEqual(
                {k: v for k, v in pinned["unset"].items() if "LKV_T" in k},
                {"LKV_T_GET": "shell", "TRITON_LKV_T": "1"},
            )
            self.assertEqual(pinned["case"], {"LKV_T_CONST": "64", "MLIR_LKV_T": "2"})
            self.assertEqual(
                (
                    os.environ.get("LKV_T_GET"),
                    os.environ.get("TRITON_LKV_T"),
                    os.environ["LKV_T_OTHER"],
                ),
                (None, None, "kept"),
            )
            rec = host_env.record(pinned)
            self.assertEqual(
                {k: v for k, v in rec["effective"].items() if "LKV_T" in k},
                {"CUTE_DSL_LKV_T": "own", "LKV_T_CONST": "64", "MLIR_LKV_T": "2"},
            )
            with self.assertRaisesRegex(
                host_env.EnvError, r"sets \['CUTE_DSL_LKV_T'\], which the frontend sets"
            ):
                host_env.pin({"CUTE_DSL_LKV_T": "x"}, [], {"CUTE_DSL_LKV_T": "own"})
        finally:
            host_env.release()
            os.environ.clear()
            os.environ.update(saved)

    ALIAS = """import os
import os as _os
from os import environ as _env


def by_alias():
    return _os.environ["LKV_T_ALIAS"]


def by_imported_environ():
    return _env.get("LKV_T_ALIAS")


def enumerated():
    return dict(_os.environ)


def by_name():
    return os.environ.get("LKV_T_GET")


def by_library():
    return os.path.expanduser("~")
"""

    def test_reads_the_scan_cannot_see_raise_until_release(self):
        # Reads through aliases and enumerations would take the shell's value.
        root = self.root(self.ALIAS)
        spec = importlib.util.spec_from_file_location(
            "lkv_t_alias", root / "pkg" / "m.py"
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        saved = dict(os.environ)
        try:
            os.environ.update({"LKV_T_ALIAS": "shell", "LKV_T_GET": "shell"})
            host_env.pin({}, [root], {})
            for read, line in ((m.by_alias, 7), (m.by_imported_environ, 11)):
                with self.assertRaisesRegex(
                    host_env.EnvError,
                    rf"pkg/m.py:{line} reads environment variable "
                    "LKV_T_ALIAS, which the source scan did not find",
                ):
                    read()
            with self.assertRaisesRegex(
                host_env.EnvError, r"pkg/m.py:15 enumerates the environment"
            ):
                m.enumerated()
            self.assertIsNone(m.by_name())  # read by name: pinned (unset), not refused
            m.by_library()  # posixpath reads HOME on the code's behalf: a library read, not checked
            self.assertIn(
                "LKV_T_ALIAS", dict(os.environ)
            )  # code outside the roots is not checked
        finally:
            host_env.release()
            os.environ.clear()
            os.environ.update(saved)
        self.assertEqual(m.by_imported_environ(), saved.get("LKV_T_ALIAS"))

    def test_compiler_switches(self):
        # Every triton.knobs variable is a compiler switch (except CC, the launcher's C
        # compiler, and AMD's AMDGCN_*); so are LLVM_* and NVPTX_* variables.
        from triton import knobs  # noqa: PLC0415

        keys = {
            d.key
            for c in vars(knobs).values()
            if isinstance(c, type) and issubclass(c, knobs.base_knobs)
            for d in vars(c).values()
            if isinstance(getattr(d, "key", None), str)
        }
        self.assertIn("USE_IR_LOC", keys)
        self.assertEqual(
            {k for k in keys if not host_env.is_compiler_switch(k)},
            {"CC"} | {k for k in keys if k.startswith("AMDGCN_")},
        )
        self.assertTrue(host_env.is_compiler_switch("LLVM_IR_ENABLE_DUMP"))

    def test_drop_unsets_switches_found_after_the_pin(self):
        # libtriton's cache-invalidating variables outside the switch names (TTGIR dump.py).
        saved = dict(os.environ)
        try:
            os.environ.update({"DISABLE_MMA_V5": "1", "LKV_T_CASE": "shell"})
            pinned = host_env.pin({"LKV_T_CASE": "case"}, [], {})
            host_env.drop(pinned, ["DISABLE_MMA_V5", "LKV_T_CASE", "LKV_T_NOT_SET"])
            self.assertEqual(pinned["unset"].get("DISABLE_MMA_V5"), "1")
            self.assertNotIn("LKV_T_NOT_SET", pinned["unset"])
            self.assertEqual(
                (os.environ.get("DISABLE_MMA_V5"), os.environ["LKV_T_CASE"]),
                (None, "case"),
            )
        finally:
            host_env.release()
            os.environ.clear()
            os.environ.update(saved)


class HostReadsTest(unittest.TestCase):
    """Where host data comes from (host_reads.py)."""

    def test_sources_and_host_reads(self):
        reads = HostReads(Path(__file__).resolve().parent)
        launched, fresh = (
            torch.zeros(4, dtype=torch.int32),
            torch.ones(4, dtype=torch.int32),
        )
        key = launched.untyped_storage().data_ptr()
        with reads:
            with reads.paused():
                self.assertEqual(reads.launch([launched]), [[]])
            meta = (
                launched.shape,
                launched.stride(),
                launched.data_ptr(),
                launched.view(2, 2),
            )
            fresh.tolist()  # never passed to a launch: not a read
            derived = launched + 1  # host computation: derived from launch 0's argument
            target = torch.zeros(4, dtype=torch.int32)
            target[0] = launched[1]  # __setitem__ writes its target
            n = int(derived.sum())
            with reads.paused():
                sources = reads.launch([derived, target, fresh])
        self.assertEqual(len(meta), 4)
        self.assertEqual(n, 4)
        self.assertEqual(sources, [[[key, 1]], [[key, 1]], []])
        (read,) = reads.reads
        self.assertEqual(
            (read["op"], read["after"], read["sources"]),
            ("TensorBase.__int__", 1, [[key, 1]]),
        )
        self.assertRegex(read["loc"], r"^test_translate.py:\d+$")

    def launched(self):
        reads = HostReads(Path(__file__).resolve().parent)
        t = torch.tensor([3, 0, 2, 5], dtype=torch.int32)
        with reads, reads.paused():
            reads.launch([t])
        return reads, t

    def test_data_dependent_shapes_are_host_reads(self):
        # The shape of nonzero(t) is t's data; one read per call.
        for name, read in (
            ("nonzero", lambda t: torch.nonzero(t).shape[0]),
            ("boolean index", lambda t: len(t[t > 0])),
            ("empty(t[0])", lambda t: torch.empty(t[0]).shape[0]),
        ):
            with self.subTest(name):
                reads, t = self.launched()
                with reads:
                    read(t)
                (r,) = reads.reads
                self.assertTrue(r["op"].startswith("aten."), r["op"])
        reads, t = self.launched()
        with reads:
            t[0].item()
            torch.arange(4).sum().item()  # data no launch saw
        self.assertEqual([r["op"] for r in reads.reads], ["TensorBase.item"])

    def test_writes_to_what_a_device_move_returned_are_refused(self):
        # On the GPU meta.cpu() is a copy; on the CPU host it is meta.
        reads = HostReads(Path(__file__).resolve().parent)
        meta = torch.tensor([2, 3], dtype=torch.int32)
        with reads:
            meta.cpu().tolist()  # a move with no later write
            meta.to(torch.int64).add_(
                1
            )  # a dtype conversion copies on the CPU host too
            shifted = meta.cpu()
            with self.assertRaisesRegex(
                HostAliasError,
                r"^test_translate.py:\d+ writes \(aten.sub_.Tensor\) data that "
                r"test_translate.py:\d+ moved between host and device",
            ):
                shifted -= 2
            on_host = torch.tensor([0, 1]).to("cpu")
            with self.assertRaisesRegex(HostAliasError, "aten.fill_.Scalar"):
                on_host[0] = 5
        self.assertEqual(meta.tolist(), [2, 3])

    def test_address_hints_make_their_pointees_sources(self):
        # A launch may write through an address an integer argument holds.
        reads = HostReads(Path(__file__).resolve().parent)
        meta = torch.zeros(4, dtype=torch.int32)
        key = meta.untyped_storage().data_ptr()
        with reads:
            ptrs = torch.tensor([meta.data_ptr() + 4], dtype=torch.int64)
            with reads.paused():
                hinted = reads.pointees([*ptrs.tolist(), key + 16, 7])
                reads.launch([ptrs], hinted)
            meta.sum().item()
        self.assertEqual(hinted, [key])
        (read,) = reads.reads
        self.assertEqual((read["op"], read["sources"]), ("TensorBase.item", [[key, 1]]))


class ExportTest(unittest.TestCase):
    """Relative imports are followed by the source export."""

    def test_relative_imports_resolve_against_the_package(self):
        src = "from . import a\nfrom .b import c\nfrom ..d import e\nimport x.y\n"
        self.assertEqual(
            imported_modules(src, package_of("p/q/m.py")),
            {"p.q", "p.q.a", "p.q.b", "p.q.b.c", "p.d", "p.d.e", "x.y"},
        )
        self.assertEqual(package_of("p/q/__init__.py"), "p.q")


class ClcScheduleTest(unittest.TestCase):
    """_with_clc builds the CLC tables from the number of resident CTAs (and clusters)."""

    @staticmethod
    def _input(ctas: int) -> dict:
        return {
            "name": "i",
            "grid": [ctas, 1, 1],
            "metadata": {},
            "params": {},
            "shapes": {},
        }

    def test_round_robin_cycles_pending_ctas_over_the_resident_ones(self):
        out = to_kir._with_clc(self._input(5), "round_robin", 2)
        self.assertEqual(out["metadata"]["__clc_launched"], [1, 1, 0, 0, 0])
        # CTA 0 steals 2 and 4, CTA 1 steals 3; rows are padded with -1 to 1 + max steals.
        self.assertEqual(out["shapes"]["__clc_schedule"], [5, 3])
        self.assertEqual(
            out["metadata"]["__clc_schedule"],
            [2, 4, -1, 3, -1, -1] + [-1] * 9,
        )

    def test_clusters_are_stolen_whole(self):
        # 2-CTA clusters, 4 resident CTAs (2 clusters), a grid of 5 clusters: round_robin
        # gives cluster 2 (CTAs 4, 5) and 4 to cluster 0, cluster 3 to cluster 1; each entry
        # is the stolen cluster's first CTA, the same for both CTAs of the thief
        out = to_kir._with_clc(self._input(10), "round_robin", 4, (2, 1, 1))
        self.assertEqual(out["metadata"]["__clc_launched"], [1] * 4 + [0] * 6)
        self.assertEqual(out["shapes"]["__clc_schedule"], [10, 3])
        row0, row1, idle = [4, 8, -1], [6, -1, -1], [-1, -1, -1]
        self.assertEqual(
            out["metadata"]["__clc_schedule"], row0 * 2 + row1 * 2 + idle * 6
        )
        greedy = to_kir._with_clc(self._input(10), "greedy", 4, (2, 1, 1))
        self.assertEqual(greedy["metadata"]["__clc_schedule"][:8], [4, 6, 8, -1] * 2)
        # fewer resident CTAs than a cluster holds: one cluster still starts
        one = to_kir._with_clc(self._input(4), "round_robin", 1, (2, 1, 1))
        self.assertEqual(one["metadata"]["__clc_launched"], [1, 1, 0, 0])
        self.assertEqual(
            to_kir._with_clc(self._input(5), "round_robin", 2, (1, 1, 1)),
            to_kir._with_clc(self._input(5), "round_robin", 2),
        )
        with self.assertRaisesRegex(to_kir.KirError, "cluster dims"):
            to_kir._with_clc(self._input(4), "none", 2, (2, 2, 1))

    def test_empty_grid_launches_nothing(self):
        out = to_kir._with_clc(self._input(0), "round_robin", 4)
        self.assertEqual(out["metadata"]["__clc_launched"], [])
        self.assertEqual(out["shapes"]["__clc_schedule"], [0, 1])

    def test_no_resident_cta_is_refused(self):
        for schedule in ("none", "round_robin", "greedy"):
            with (
                self.subTest(schedule=schedule),
                self.assertRaisesRegex(
                    to_kir.KirError, "at least one resident CTA, got 0"
                ),
            ):
                to_kir._with_clc(self._input(4), schedule, 0)


class PreferredGeometryTest(unittest.TestCase):
    """preferred_geometry: the launches' preferred cluster dims where they differ from
    the regular ones (the PTX translation's second KIR, <spec>@cluster<dims>)."""

    @staticmethod
    def _dump(*recs) -> dict:
        return {"inputs": [{"name": f"i{k}", "launches": [{"spec": "s", "launch": r}]}
                           for k, r in enumerate(recs)]}  # fmt: skip

    def test_preferred_geometry(self):
        pref = {"cluster": [2, 1, 1], "grid": [8, 1, 1],
                "attrs": {"preferred_cluster": [4, 1, 1]}}  # fmt: skip
        self.assertEqual(
            to_kir.preferred_geometry(self._dump(pref, pref), "s"), (4, 1, 1)
        )
        self.assertEqual(to_kir.geometry_suffix((4, 1, 1)), "@cluster4x1x1")
        for attrs in (
            {},
            {"preferred_cluster": None},
            {"preferred_cluster": [2, 1, 1]},
        ):
            rec = {"cluster": [2, 1, 1], "attrs": attrs}
            self.assertIsNone(to_kir.preferred_geometry(self._dump(rec), "s"), attrs)
        # regular dims 1 x 1 x 1 count as a regular cluster only as REQUIRED_CLUSTER
        single = {"cluster": [1, 1, 1], "grid": [6, 1, 1],
                  "attrs": {"preferred_cluster": [3, 1, 1],
                            "func_attributes": {"required_cluster": [1, 1, 1]}}}  # fmt: skip
        self.assertEqual(to_kir.preferred_geometry(self._dump(single), "s"), (3, 1, 1))
        del single["attrs"]["func_attributes"]
        with self.assertRaisesRegex(to_kir.KirError, "without a regular cluster"):
            to_kir.preferred_geometry(self._dump(single), "s")
        odd = {"cluster": [2, 1, 1], "attrs": {"preferred_cluster": [3, 1, 1]}}
        with self.assertRaisesRegex(to_kir.KirError, "no multiple"):
            to_kir.preferred_geometry(self._dump(odd), "s")
        with self.assertRaisesRegex(to_kir.KirError, "differ"):
            to_kir.preferred_geometry(self._dump(pref, {"cluster": [2, 1, 1]}), "s")


class RequiredClusterTest(unittest.TestCase):
    """check_required_cluster: a launch whose function's REQUIRED_CLUSTER attribute
    (cuFuncSetAttribute) differs from its cluster dims is refused (gap)."""

    @staticmethod
    def _dump(*recs) -> dict:
        return {"inputs": [{"name": f"i{k}", "launches": [{"spec": "s", "launch": r}]}
                           for k, r in enumerate(recs)]}  # fmt: skip

    def test_required_cluster(self):
        dump = self._dump
        for attrs in ({}, {"func_attributes": {}},
                      {"func_attributes": {"required_cluster": [2, 1, 1]}}):  # fmt: skip
            to_kir.check_required_cluster(
                dump({"cluster": [2, 1, 1], "attrs": attrs}), "s"
            )
        # a launch with regular cluster 1 (required 1) and preferred cluster 3
        pref3 = {"cluster": [1, 1, 1], "attrs": {"preferred_cluster": [3, 1, 1],
                                                 "func_attributes": {"required_cluster": [1, 1, 1]}}}  # fmt: skip
        to_kir.check_required_cluster(dump(pref3), "s")
        bad = {
            "cluster": [1, 1, 1],
            "attrs": {"func_attributes": {"required_cluster": [2, 1, 1]}},
        }
        with self.assertRaises(to_kir.KirError) as e:
            to_kir.check_required_cluster(dump(pref3, bad), "s")
        self.assertIn(
            "cluster dims (1, 1, 1) but its function's required cluster",
            str(e.exception),
        )
        self.assertEqual(to_kir.failure_kind(e.exception), "gap")


class PreferredClusterTest(unittest.TestCase):
    """check_preferred_cluster: the TTGIR translation refuses a launch with a preferred
    cluster other than its cluster dims (gap: it checks one geometry, the PTX
    translation both), after its own refusals."""

    def test_preferred_cluster(self):
        dump = dump_of({})
        la = dump["inputs"][0]["launches"][0]
        case = {"inputs": [{"name": "i"}]}
        for attrs in (
            {},
            {"preferred_cluster": None},
            {"preferred_cluster": [1, 1, 1]},
        ):
            la["launch"] = {"cluster": [1, 1, 1], "attrs": attrs}
            to_kir.build(ttgir({}, ""), dump, "s", case)
        la["launch"] = {"cluster": [1, 1, 1], "attrs": {"preferred_cluster": [3, 1, 1]}}
        with self.assertRaises(to_kir.KirError) as e:
            to_kir.build(ttgir({}, ""), dump, "s", case)
        self.assertIn(
            "asks for a preferred cluster (3, 1, 1) beside its cluster dims (1, 1, 1)",
            str(e.exception),
        )
        self.assertIn("the PTX translation checks both", str(e.exception))
        self.assertEqual(to_kir.failure_kind(e.exception), "gap")
        no_warps = ATTRS.replace('"ttg.num-warps" = 4 : i32, ', "")
        with self.assertRaisesRegex(to_kir.KirError, "ttg.num-warps"):
            to_kir.build(ttgir({}, "", no_warps), dump, "s", case)


class TmemRectTest(unittest.TestCase):
    """A tmem event's rects validate through frontend/kir.py."""

    @staticmethod
    def _kir(rects: list) -> dict:
        def c(v: int) -> dict:
            return {"e": "const", "v": v, "w": 32}

        return {
            "kir_version": 1,
            "kernel": "k",
            "source": "k.py",
            "ir": "ttgir",
            "grid": [c(1), c(1), c(1)],
            "params": [],
            "tensors": [],
            "barriers": [],
            "roles": [
                {
                    "name": "r",
                    "warps": 1,
                    "threads_per_issue": 1,
                    "body": [
                        {
                            "op": "event",
                            "ev": "tmem",
                            "inst": "ld",
                            "warp": c(0),
                            "rects": rects,
                            "loc": "k.py:1",
                        }
                    ],
                }
            ],
            "tmem": {"cols": 32, "base": "relative"},
            "case": {"name": "c", "inputs": [{"name": "i", "params": {}}]},
        }

    @staticmethod
    def _rect() -> dict:
        def c(v: int) -> dict:
            return {"e": "const", "v": v, "w": 32}

        return {
            "use": "data",
            "write": False,
            "lane": c(0),
            "nlanes": 16,
            "col": c(0),
            "ncols": 1,
        }

    def test_valid_ld_rects_validate_clean(self):
        self.assertEqual(kir_contract.validate(self._kir([self._rect()])), [])

    def test_unknown_rect_use_is_rejected(self):
        rect = self._rect()
        rect["use"] = "meta"
        errors = kir_contract.validate(self._kir([rect]))
        self.assertTrue(any("must be one of" in e for e in errors), errors)


if __name__ == "__main__":
    unittest.main()
