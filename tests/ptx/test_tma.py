# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of TMA and bulk copies in the PTX core (frontend/ptx/tma.py) through translate().

    python3 -I tests/ptx/test_tma.py      (or the buck target :test_tma)

Host descriptors: the TTGIR convention (tma_load/tma_store/tma_reduce on the jit-named
descriptor, the kernel's own shape/stride params, box from the record; reached by
`mov.b32` + `cvt` + `cvta.param` or `mov.b64` + `cvta.param`) and the CuTe one (tma<k>_<base>
with synthetic shape/stride params from the decoded CUtensorMap, a tma_load plus its gload
footprint, stores and reduces as gstore/gatomic footprints), coordinates reversed, packed
sub-byte formats (PackedSubByte: KIR in bytes, the dim-0 coordinate asserted a multiple of
the hardware's granularity and converted, a store or reduce the hardware traps on asserted
false, packed Triton and in-kernel descriptors refused), a zero-length tensor's
descriptor (extents 0, no footprint); the issuing threads (elect: one per warp; k = 1, 2 or
32 lanes of a warp: loads and reduces, tensor and bulk, once per lane with its own
operands, stores once per distinct (descriptor, coordinates, shared source) or (global
address, shared source), the cache policy not part of it, as tests/ptx/test_gpu_tma_issue.py
measures; refused unless the compiler is in tma.ISSUE_PTXAS and the target in
tma.ISSUE_TARGETS, and a store whose copies depend on the elected lane; a single thread's
per-warp operand splits the role; tkinfo_version); bulk copies
(g2s: gload + tma_load of bulk_<T>_<n>; s2g; reduce, its type checked); in-kernel descriptors (zeroed staging buffer, replace under a thread mask,
cp_fenceproxy to the scratch, a store through it clipped at the extents, one buffer reused
for two descriptors, other shared-memory writes closing the buffer (generic st/atom/red
through cvta.shared unless of static bytes outside it, stmatrix, cp.async, mbarrier),
and the refusals, a replace under a branch on data among them); the write set through
descriptors; clusters, multicast and generic access to a descriptor refused; desc_base
and segment_dims.
"""

import math
import struct
import sys
import unittest
from pathlib import Path
from types import ModuleType

if __package__ in (None, ""):
    sys.dont_write_bytecode = True
    _root = ModuleType("lkv_root")
    _root.__path__ = [str(Path(__file__).resolve().parents[2])]
    sys.modules["lkv_root"] = _root
    __package__ = "lkv_root.tests.ptx"  # noqa: A001

from ...frontend import kir  # noqa: E402
from ...frontend.ptx import tma, translate as T  # noqa: E402

SRC = "/src"
_DECL = {
    "tmap": ".param .align 64 .b8 k_param_{i}[128]",
    "ptr": ".param .u64 .ptr .global .align 1 k_param_{i}",
    "b32": ".param .b32 k_param_{i}",
    "b64": ".param .b64 k_param_{i}",
}
_SIZE = {"tmap": 128, "ptr": 8, "b32": 4, "b64": 8}
_KIND = {"tmap": "tensormap", "ptr": "ptr", "b32": "scalar", "b64": "scalar"}


def kernel(body: str, params: list, threads: int) -> str:
    decl = ", ".join(_DECL[k].format(i=i) for i, (k, _n, _v) in enumerate(params))
    return f"""
.version 9.3
.target sm_100a
.address_size 64
.extern .shared .align 1024 .b8 smem[];
.visible .entry k({decl})
.reqntid {threads}
{{
    .reg .pred %p<32>;
    .reg .b32 %r<100>;
    .reg .b64 %rd<64>;
    .loc 1 1 0
{body}
    ret;
}}
    .file 1 "{SRC}/k.py"
"""


def records(params: list, family: str, threads: int, n: int = 1) -> list:
    """LaunchRecords: a value given as a list has one entry per launch."""
    out = []
    for r in range(n):
        ps = []
        for i, (k, name, val) in enumerate(params):
            v = val[r] if isinstance(val, list) else val
            p = {
                "i": i,
                "ptx": f"k_param_{i}",
                "name": name,
                "kind": _KIND[k],
                "size": _SIZE[k],
            }
            if k == "tmap":
                p["tensormap" if family == "cute" else "value"] = v
            elif v == "scratch":
                p["value"], p["scratch"] = None, {"bytes": 1024, "align": 128}
            else:
                p["value"] = v
            ps.append(p)
        out.append(
            {
                "block": [threads, 1, 1],
                "cluster": [1, 1, 1],
                "grid": [1, 1, 1],
                "params": ps,
            }
        )
    return out


def run(body, params, family="ttgir", tensors=None, threads=32, launches=1, **pol):
    recs = records(params, family, threads, launches)
    table = T.param_table(recs, family, None, [])
    pol = T.policy(family, src_root=SRC, params=table,
                   tensors={n: {"elem_bytes": b} for n, b in (tensors or {}).items()}, **pol)  # fmt: skip
    return T.translate(kernel(body, params, threads), "k", recs, pol)


def writes(body, params, family="ttgir", threads=32):
    table = T.param_table(records(params, family, threads), family, None, [])
    return T.writes(kernel(body, params, threads), "k", table)


def statements(stmts):
    for s in stmts:
        yield s
        for k in ("then", "else", "body"):
            yield from statements(s.get(k, []))


def events(facts, ev=None):
    return [s for r in facts.roles for s in statements(r["body"])
            if s["op"] == "event" and ev in (None, s["ev"])]  # fmt: skip


def kir_errors(facts, tensors, metadata=None) -> list:
    """frontend/kir.py's errors for a KIR around the Facts: 1-D tensors of 1 << 20
    elements, the descriptors (a bulk one over its base), every param 1 unless the Facts'
    values give it; `metadata`: the contents of the metadata tensors."""
    one = {"e": "const", "w": 64, "v": 1}
    ts = [{"name": n, "kind": "global", "elem_bytes": t["elem_bytes"],
           "shape": [{"e": "const", "w": 64, "v": 1 << 20}], "strides": [one], "segments": None,
           "role": "output" if n in facts.written else "input", "metadata": t["metadata"]}
          for n, t in facts.tensors.items()]  # fmt: skip
    ts += [{"name": n, "kind": "desc", "elem_bytes": d["elem_bytes"], "box": d["box"],
            "shape": d["shape"] or [{"e": "const", "w": 64, "v": 1 << 20}], "strides": d["strides"] or [one],
            "segments": None, "role": "output" if n in facts.written else "input", "metadata": False}
           for n, d in facts.descs.items()]  # fmt: skip
    values = {
        **dict.fromkeys(facts.params, 1),
        **(facts.values[0] if facts.values else {}),
    }
    obj = {"kir_version": 1, "kernel": "k", "source": "k.py", "ir": "ptx", "target": "sm_100",
           "grid": [{"e": "const", "w": 32, "v": 1}] * 3,
           "params": [{"name": n, "type": f"i{w}"} for n, w in facts.params.items()],
           "tensors": ts, "barriers": facts.barriers,
           "roles": [{k: r[k] for k in ("name", "warps", "threads_per_issue", "body")} for r in facts.roles],
           "case": {"name": "t", "inputs": [{"name": "i", "params": values, "metadata": metadata or {},
                                              "intended_writes": dict.fromkeys(facts.written, "unchecked")}]}}  # fmt: skip
    return kir.validate(obj)


# ------------------------------------------------------------------ evaluation
#
# Footprints of a role on concrete params: the statements of these small kernels (lets,
# ifs, thread loops, events) and the TExprs of a TMA footprint (splat, full, range,
# expand, bcast, bin, cmp) evaluated element by element.

_OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
        "sdiv": lambda a, b: int(a / b), "srem": lambda a, b: a - b * int(a / b),
        "and": lambda a, b: a & b, "or": lambda a, b: a | b, "shl": lambda a, b: a << b,
        "xor": lambda a, b: a ^ b, "eq": lambda a, b: a == b, "ne": lambda a, b: a != b,
        "slt": lambda a, b: a < b, "sge": lambda a, b: a >= b, "ult": lambda a, b: a < b,
        "urem": lambda a, b: a % b, "udiv": lambda a, b: a // b, "lshr": lambda a, b: a >> b}  # fmt: skip


def footprints(facts, params, role=0, tables=None):  # noqa: C901 (one case per KIR node and statement)
    """[(event, tensor or desc, sorted element offsets or coords)] of one role; `tables`:
    the contents of the tensors it loads from."""
    env, out = dict(params), []

    def ex(e):
        k = e["e"]
        if k == "const":
            return e["v"]
        if k == "var":
            return env[e["name"]]
        if k in ("ext", "trunc"):
            return ex(e["a"])
        if k == "select":
            return ex(e["a"]) if ex(e["c"]) else ex(e["b"])
        if k == "pid":
            return 0
        if k == "load":
            return tables[e["tensor"]][ex(e["index"][0])]
        return int(_OPS[e["o"]](ex(e["a"]), ex(e["b"])))

    def at(t, idx):
        k = t["t"]
        if k == "splat":
            return ex(t["a"])
        if k == "full":
            return t["v"]
        if k == "range":
            return t["start"] + idx[0]
        if k in ("expand", "bcast"):
            ax = t["axis"]
            return at(
                t["a"],
                idx[:ax] + idx[ax + 1 :]
                if k == "expand"
                else idx[:ax] + (0,) + idx[ax + 1 :],
            )
        return int(_OPS[t["o"]](at(t["a"], idx), at(t["b"], idx)))

    def go(stmts):
        for s in stmts:
            if s["op"] == "let":
                env[s["var"]] = ex(s["expr"])
            elif s["op"] == "if":
                go(s["then"] if ex(s["cond"]) else s["else"])
            elif s["op"] == "for":
                for v in range(ex(s["lb"]), ex(s["ub"]), ex(s["step"])):
                    env[s["var"]] = v
                    go(s["body"])
            elif s["op"] == "event" and "offsets" in s:
                shape = s["offsets"]["shape"]
                idxs = [()]
                for n in shape:
                    idxs = [i + (j,) for i in idxs for j in range(n)]
                got = [at(s["offsets"], i) for i in idxs
                       if s["mask"] is None or at(s["mask"], i)]  # fmt: skip
                out.append((s["ev"], s["tensor"], sorted(got)))
            elif s["op"] == "event" and "coords" in s:
                out.append((s["ev"], s["desc"], [ex(c) for c in s["coords"]]))

    go(facts.roles[role]["body"])
    return out


# ---------------------------------------------------------------- fixtures

META = {"address": 0x10000, "host_dtype": 10, "elem_size": 2, "shape": [256, 128],
        "strides": [128, 1], "box": [64, 32], "swizzle": 3, "padding": 0}  # fmt: skip
TTGIR_PARAMS = [("tmap", "desc_x", META), ("b32", "desc_x.shape.0", 256),
                ("b32", "desc_x.shape.1", 128), ("b64", "desc_x.stride.0", 128),
                ("b64", "desc_x.stride.1", 1), ("b32", "n", 7)]  # fmt: skip


def decoded(
    dims, strides_bytes, box, base="x", elem_bits=16, address=0x10000, elem_off=0
):
    """A launch.decode result (the fields kir_desc reads) with its address candidates."""
    cands = [{"tensor": base, "storage": [address, address + (1 << 20)], "elem_off": elem_off,
              "match": "exact"}] if base else []  # fmt: skip
    return {"dtype": "X", "elem_bits": elem_bits, "rank": len(dims), "address": address,
            "dims": dims, "strides_bytes": strides_bytes, "box": box,
            "estrides": [1] * len(dims), "interleave": "NONE", "base": cands}  # fmt: skip


def packed(dtype, dims, strides_bytes, box, **kw):
    """A decoded packed sub-byte descriptor (4- or 6-bit elements) over the byte tensor q."""
    bits = 6 if dtype == "16U6_ALIGN16B" else 4
    return {**decoded(dims, strides_bytes, box, "q", bits, **kw), "dtype": dtype}


CUTE_PARAMS = [("tmap", "t", decoded([128, 256], [256], [32, 64])), ("b32", "n", 7)]

PRE = """
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, smem;
    setp.eq.u32 %p1, %r1, 0;
    @%p1 mbarrier.init.shared::cta.b64 [%r2], 1;
    ld.param.b32 %r5, [k_param_{n}];
    mov.b32 %r7, 32;
    add.s32 %r6, %r2, 1024;
    elect.sync %r3|%p2, -1;
"""
TMAP32 = """
    mov.b32 %r10, k_param_0;
    cvt.u64.u32 %rd1, %r10;
    cvta.param.u64 %rd2, %rd1;
"""
TMAP64 = """
    mov.b64 %rd1, k_param_0;
    cvta.param.u64 %rd2, %rd1;
    prefetch.tensormap [%rd2];
"""
LOAD = """
    @%p2 mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r2], {nbytes};
    @%p2 cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%r6], [%rd2, {{%r7, %r5}}], [%r2];
"""
STORE = "    @%p2 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd2, {%r7, %r5}], [%r6];\n"
REDUCE = "    @%p2 cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group [%rd2, {%r7, %r5}], [%r6];\n"


class HostTtgir(unittest.TestCase):
    def test_load_on_the_descriptor(self):
        for tmap in (TMAP32, TMAP64):
            f = run(PRE.format(n=5) + tmap + LOAD.format(nbytes=4096), TTGIR_PARAMS)
            self.assertEqual(f.refusals, [])
            (ev,) = events(f, "tma_load")
            self.assertEqual((ev["desc"], ev["bar"]), ("desc_x", "mbar_0"))
            # PTX {32, n} innermost first -> KIR [n, 32]
            self.assertEqual(
                ev["coords"],
                [{"e": "var", "w": 32, "name": "n"}, {"e": "const", "w": 32, "v": 32}],
            )
            self.assertEqual(events(f, "gload"), [])  # no base footprint
            d = f.descs["desc_x"]
            self.assertEqual(
                (d["kind"], d["box"], d["elem_bytes"], d["uses"]),
                ("host", [64, 32], 2, ["tma_load"]),
            )
            self.assertEqual([x["name"] for x in d["shape"] + d["strides"]],
                             ["desc_x.shape.0", "desc_x.shape.1", "desc_x.stride.0", "desc_x.stride.1"])  # fmt: skip
            self.assertEqual(
                (f.params["desc_x.shape.0"], f.params["desc_x.stride.0"]), (32, 64)
            )
            self.assertEqual(f.values[0]["desc_x.shape.0"], 256)
            self.assertEqual(kir_errors(f, {}), [])

    def test_store_and_reduce_on_the_descriptor(self):
        f = run(PRE.format(n=5) + TMAP32 + STORE + REDUCE, TTGIR_PARAMS)
        self.assertEqual(
            [e["ev"] for e in events(f)], ["bar_init", "tma_store", "tma_reduce"]
        )
        self.assertEqual(f.written, ["desc_x"])
        self.assertEqual(f.descs["desc_x"]["uses"], ["tma_reduce", "tma_store"])
        self.assertEqual(kir_errors(f, {}), [])
        self.assertEqual(
            writes(PRE.format(n=5) + TMAP32 + STORE, TTGIR_PARAMS), ["desc_x"]
        )

    def test_block_of_several_boxes(self):
        # Triton cuts a block [64, 64] into two messages of box [64, 32] (inner coordinate
        # +32): one event of the block at the first message's coordinates, as the TTGIR
        # frontend's; messages at no other box of the block, or too few, are refused.
        second = (
            "    add.s32 %r8, %r7, 32;\n    @%p2 cp.async.bulk.tensor.2d.shared::cta."
            "global.mbarrier::complete_tx::bytes [%r6], [%rd2, {%r8, %r5}], [%r2];\n"
        )
        blocks = {"desc_x": [64, 64]}
        body = PRE.format(n=5) + TMAP32 + LOAD.format(nbytes=8192)
        f = run(body + second, TTGIR_PARAMS, blocks=blocks)
        self.assertEqual(f.refusals, [])
        (ev,) = events(f, "tma_load")
        self.assertEqual(ev["coords"][1], {"e": "const", "w": 32, "v": 32})
        self.assertEqual(f.descs["desc_x"]["box"], [64, 64])
        self.assertEqual(kir_errors(f, {}), [])
        f = run(body + second.replace("32;", "16;"), TTGIR_PARAMS, blocks=blocks)
        self.assertIn("no other box of its block", f.refusals[0]["msg"])
        f = run(body, TTGIR_PARAMS, blocks=blocks)
        self.assertIn("its block [64, 64] has 2", f.refusals[0]["msg"])
        st = PRE.format(n=5) + TMAP32 + STORE + "    add.s32 %r8, %r7, 32;\n"
        f = run(st + STORE.replace("{%r7,", "{%r8,"), TTGIR_PARAMS, blocks=blocks)
        self.assertEqual(len(events(f, "tma_store")), 1)
        # one instruction, the elected lane of each of two warps: warp parity picks the
        # box (TLX's dQ reduce); together one block, one event at its origin
        parity = (
            "    shr.u32 %r11, %r1, 5;\n    and.b32 %r12, %r11, 1;\n"
            "    shl.b32 %r13, %r12, 5;\n    add.s32 %r8, %r7, %r13;\n"
        )
        body = PRE.format(n=5) + TMAP32 + parity + REDUCE.replace("{%r7,", "{%r8,")
        f = run(body, TTGIR_PARAMS, blocks=blocks, threads=64)
        self.assertEqual(f.refusals, [])
        (ev,) = events(f, "tma_reduce")
        self.assertEqual(ev["coords"][1], {"e": "const", "w": 32, "v": 32})
        self.assertEqual(kir_errors(f, {}), [])

    def test_coordinate_marked_as_address(self):
        body = (
            PRE.format(n=5)
            + TMAP32
            + "    add.s32 %r9, %r5, 64;\n"
            + STORE.replace("%r5}", "%r9}")
        )
        f = run(body, TTGIR_PARAMS)
        nodes = [s["expr"] for s in statements(f.roles[0]["body"]) if s["op"] == "let"]
        nodes += [c for e in events(f, "tma_store") for c in e["coords"]]
        self.assertTrue(any(x.get("o") == "add" and x.get("addr") for x in nodes))


class HostCute(unittest.TestCase):
    def test_load_with_footprint(self):
        f = run(
            PRE.format(n=1) + TMAP64 + LOAD.format(nbytes=4096),
            CUTE_PARAMS,
            "cute",
            {"x": 2},
        )
        self.assertEqual(f.refusals, [])
        (ld,) = events(f, "tma_load")
        self.assertEqual(ld["desc"], "tma0_x")
        d = f.descs["tma0_x"]
        self.assertEqual((d["box"], d["base"], d["elem_bytes"]), ([64, 32], "x", 2))
        self.assertEqual(f.values[0], {"tma0_x.shape.0": 256, "tma0_x.shape.1": 128,
                                       "tma0_x.stride.0": 128, "tma0_x.stride.1": 1})  # fmt: skip
        self.assertEqual(f.tensors["x"]["uses"], ["gload"])
        params = {**f.values[0], "param1": 250}
        (_l, (ev, t, offs)) = footprints(f, params)
        # rows 250..255 of 256 (box 64 rows clipped), columns 32..63
        want = sorted(r * 128 + c for r in range(250, 256) for c in range(32, 64))
        self.assertEqual((ev, t, offs), ("gload", "x", want))
        self.assertEqual(kir_errors(f, {"x": 2}), [])

    def test_store_and_reduce_footprints(self):
        f = run(
            PRE.format(n=1) + TMAP64 + STORE + REDUCE, CUTE_PARAMS, "cute", {"x": 2}
        )
        self.assertEqual([e["ev"] for e in events(f)][1:], ["gstore", "gatomic"])
        self.assertEqual((f.written, f.descs), (["x"], {}))
        self.assertEqual(
            writes(PRE.format(n=1) + TMAP64 + STORE, CUTE_PARAMS, "cute"), ["x"]
        )

    def test_zero_length_descriptor(self):
        p = [
            (
                "tmap",
                "t",
                decoded([1 << 32, 1 << 32], [256], [32, 64], None, address=0),
            ),
            ("b32", "n", 7),
        ]
        f = run(
            PRE.format(n=1) + TMAP64 + LOAD.format(nbytes=4096), p, "cute", {"x": 2}
        )
        self.assertEqual(f.refusals, [])
        self.assertEqual(f.descs["tma0"]["base"], None)
        self.assertEqual(
            (f.values[0]["tma0.shape.0"], f.values[0]["tma0.shape.1"]), (0, 0)
        )
        self.assertEqual(
            [e["ev"] for e in events(f)],
            ["bar_init", "expect_tx", "arrive", "tma_load"],
        )
        # The hardware reads the extents 2^32: an issued copy accesses address 0, so the
        # issuing thread asserts false.
        (check,) = [s for s in statements(f.roles[0]["body"]) if s["op"] == "assert"]
        self.assertIn("zero-length tensor's descriptor", check["msg"])
        self.assertEqual(check["cond"], {"e": "const", "w": 1, "v": 0})

    def test_shifted_base_offset(self):
        p = [("tmap", "t", [decoded([128, 256], [256], [32, 64], elem_off=-64),
                            decoded([128, 256], [256], [32, 64], elem_off=0)]), ("b32", "n", 7)]  # fmt: skip
        f = run(
            PRE.format(n=1) + TMAP64 + LOAD.format(nbytes=4096),
            p,
            "cute",
            {"x": 2},
            launches=2,
        )
        self.assertEqual([v["tma0_x.off"] for v in f.values], [-64, 0])
        self.assertEqual(kir_errors(f, {"x": 2}), [])


class Issue(unittest.TestCase):
    def test_elected_issuer_per_warp(self):
        f = run(
            PRE.format(n=5) + TMAP32 + LOAD.format(nbytes=4096),
            TTGIR_PARAMS,
            threads=64,
        )
        (loop,) = [s for s in statements(f.roles[0]["body"])
                   if s["op"] == "for" and s["body"][0]["ev"] == "tma_load"]  # fmt: skip
        self.assertEqual(loop["ub"]["v"], 2)  # one elected lane per warp

    def test_single_thread_operand_per_warp_splits_the_role(self):
        # thread 0 of two warps loads at coordinate tid / 32: CuTe splits the role per
        # warp on it, as before D15 (not a thread loop over each thread's own operands)
        body = PRE.format(n=1) + TMAP64 + "    shr.u32 %r7, %r1, 5;\n"
        body += LOAD.format(nbytes=4096).replace("@%p2", "@%p1")
        f = run(body, CUTE_PARAMS, "cute", {"x": 2}, threads=64)
        self.assertEqual(f.refusals, [])
        self.assertEqual([r["name"] for r in f.roles], ["warps0_0", "warps1_1"])
        self.assertEqual(len(events(f, "tma_load")), 1)


# k lanes of one warp (%p3: lane < k; `full`: the whole warp, unpredicated) execute one
# copy whose operands are the same on every lane or differ per lane, the kinds
# tests/ptx/test_gpu_tma_issue.py measures on hardware: coordinate %r1 (the lane) instead
# of %r7 (32); shared address %r16 (+ 128 * lane) instead of %r6; cache policy %rd10 (one
# of two by lane parity, %p4); coordinate %r13 = tab[lane] (`same_dyn`: equal values in
# lane-varying registers when tab is constant); descriptor %rd3 (the second one, u) on the
# odd lanes (%p5) and %rd2 on the even ones (%p6), as two instructions.
LANES = """
    setp.lt.u32 %p3, %r1, {k};
    and.b32 %r14, %r1, 1;
    setp.eq.u32 %p4, %r14, 1;
    shl.b32 %r15, %r1, 7;
    add.s32 %r16, %r6, %r15;
    selp.b64 %rd10, 1, 2, %p4;
    and.pred %p5, %p3, %p4;
    xor.pred %p6, %p4, 1;
    and.pred %p6, %p3, %p6;
"""
TAB = """
    ld.param.u64 %rd20, [k_param_3];
    mul.wide.u32 %rd21, %r1, 4;
    add.s64 %rd22, %rd20, %rd21;
    ld.global.u32 %r13, [%rd22];
"""
TMAP_U = "    mov.b64 %rd8, k_param_1;\n    cvta.param.u64 %rd3, %rd8;\n"
VARIANT = {"same": {}, "coord": {"c": "%r1"}, "smem": {"src": "%r16"}, "hint": {"hint": True},
           "same_dyn": {"c": "%r13"}}  # fmt: skip
TWO = [("tmap", "t", decoded([128, 256], [256], [8, 4])),
       ("tmap", "u", decoded([128, 256], [256], [8, 4], "y", address=0x200000)),
       ("b32", "n", 7), ("ptr", "tab", 0x300000)]  # fmt: skip
TWO_T = {"x": 2, "y": 2, "param3": 4}  # a CuTe pointer parameter is param<i>
OK = "13.3.33"  # a validated compiler (tma.ISSUE_PTXAS)


def tma_op(form, c="%r7", src="%r6", hint=False, desc="%rd2"):
    h, ho = (".L2::cache_hint", ", %rd10") if hint else ("", "")
    crd = f"[{desc}, {{{c}, %r5}}]"
    return {"load": f"cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes{h} [{src}], {crd}, [%r2]{ho}",
            "store": f"cp.async.bulk.tensor.2d.global.shared::cta.bulk_group{h} {crd}, [{src}]{ho}",
            "reduce": f"cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group{h} {crd}, [{src}]{ho}"}[form]  # fmt: skip


def lanes(form, var, k, full=False, ptxas=OK):
    """The CuTe kernel (descriptors t over x and u over y) of `var` with k lanes."""
    pred = "@%p6 " if var == "desc" else "" if full else "@%p3 "
    body = (
        PRE.format(n=2)
        + TMAP64
        + TMAP_U
        + LANES.format(k=k)
        + (TAB if var == "same_dyn" else "")
    )
    body += f"    {pred}{tma_op(form, **VARIANT.get(var, {}))};\n"
    if var == "desc":
        body += f"    @%p5 {tma_op(form, desc='%rd3')};\n"
    return run(body, TWO, "cute", TWO_T, ptxas=ptxas)


def meta(var):
    return {"param3": [5] * 32} if var == "same_dyn" else None


def copies(f, tab=None):
    """[(event, tensor or desc, offsets or coords)] of the copies of a lanes() kernel (the
    tab load left out), with n = 7 and tab = `tab` (default: 32 x 5)."""
    params = {**f.values[0], "param2": 7}
    got = footprints(f, params, tables={"param3": tab or [5] * 32})
    return [g for g in got if g[1] != "param3"]


class Lanes(unittest.TestCase):
    """The issuing rules for several lanes of a warp (tma.py, "Issuing threads")."""

    def test_store_once_per_distinct_key(self):
        # (descriptor, coordinates, shared source); not the cache policy
        keys = {"same": 1, "same_dyn": 1, "hint": 1, "coord": 32, "smem": 32, "desc": 2}
        for var, n in keys.items():
            for k, full in ((1, False), (2, False), (32, False), (32, True)):
                with self.subTest(var=var, k=k, full=full):
                    f = lanes("store", var, k, full)
                    self.assertEqual(f.refusals, [])
                    got = copies(f)
                    self.assertEqual(len(got), min(k, n))
                    self.assertTrue(all(g[0] == "gstore" for g in got))
                    if var == "desc" and k > 1:
                        self.assertEqual(sorted(g[1] for g in got), ["x", "y"])
                    self.assertEqual(kir_errors(f, TWO_T, meta(var)), [])
        # equal values only at run time: one copy if tab is constant, else one per lane
        f = lanes("store", "same_dyn", 32)
        self.assertEqual(len(copies(f, list(range(32)))), 32)
        self.assertEqual(len(copies(f, [3, 4] * 16)), 2)

    def test_coordinates_of_each_issuing_lane(self):
        f = lanes("store", "coord", 2)
        cols = [sorted({o % 128 for o in g[2]}) for g in copies(f)]
        # lane L's box starts at column L (KIR coords [n, L]), 8 columns
        self.assertEqual(cols, [list(range(0, 8)), list(range(1, 9))])

    def test_load_and_reduce_once_per_lane(self):
        for form, evs in (("load", ("tma_load", "gload")), ("reduce", ("gatomic",))):
            for var in ("same", "same_dyn", "coord", "smem", "hint", "desc"):
                for k, full in ((1, False), (2, False), (32, False), (32, True)):
                    with self.subTest(form=form, var=var, k=k, full=full):
                        f = lanes(form, var, k, full)
                        self.assertEqual(f.refusals, [])
                        got = copies(f)
                        for ev in evs:
                            self.assertEqual(len([g for g in got if g[0] == ev]), k)
                        if var == "desc":
                            on_y = [g for g in got if g[0] == evs[-1] and g[1] == "y"]
                            self.assertEqual(len(on_y), k // 2)
                        self.assertEqual(kir_errors(f, TWO_T, meta(var)), [])

    def test_load_coordinates_and_bytes_per_lane(self):
        # each lane's tma_load at its own coordinates: the lanes' complete_tx bytes are
        # k boxes (P0-4)
        f = lanes("load", "coord", 32)
        loads = [g for g in copies(f) if g[0] == "tma_load"]
        self.assertEqual([g[2] for g in loads], [[7, lane] for lane in range(32)])
        self.assertEqual(math.prod(f.descs["tma0_x"]["box"]) * 2, 64)
        (loop,) = [s for s in f.roles[0]["body"] if s["op"] == "for"
                   and any(x.get("ev") == "tma_load" for x in statements(s["body"]))]  # fmt: skip
        self.assertEqual(loop["ub"]["v"], 32)  # a thread loop: each lane's coordinates
        f = lanes("load", "same", 32)
        (loop,) = [s for s in f.roles[0]["body"] if s["op"] == "for"
                   and s["body"][0].get("ev") == "tma_load"]  # fmt: skip
        # the same operands: 32 issues of one event
        self.assertEqual(loop["ub"]["v"], 32)

    def test_packed_sub_byte_load_per_lane(self):
        p = [("tmap", "t", packed("16U4_ALIGN8B", [256, 64], [128], [128, 16])),
             ("b32", "n", 7)]  # fmt: skip
        body = PRE.format(n=1) + TMAP64 + LANES.format(k=2)
        body += "    shl.b32 %r18, %r1, 5;\n    add.s32 %r17, %r5, %r18;\n"
        body += f"    @%p3 {tma_op('load', c='%r5').replace('{%r5, %r5}', '{%r17, %r5}')};\n"
        f = run(body, p, "cute", {"q": 1}, ptxas=OK)
        self.assertEqual(f.refusals, [])
        got = footprints(f, {**f.values[0], "param1": 64})
        # lane L at element column n + 32 L, byte column (n + 32 L) / 2 (dim 0 in bytes)
        self.assertEqual(
            [g[2] for g in got if g[0] == "tma_load"], [[64, 32], [64, 48]]
        )
        asserts = [s for s in statements(f.roles[0]["body"]) if s["op"] == "assert"]
        self.assertIn("multiple of 32 16U4_ALIGN8B", asserts[0]["msg"])

    def test_two_descriptors_in_one_instruction_refused(self):
        body = PRE.format(n=2) + TMAP64 + TMAP_U + LANES.format(k=32)
        body += "    selp.b64 %rd9, %rd3, %rd2, %p4;\n"
        body += f"    @%p3 {tma_op('store', desc='%rd9')};\n"
        f = run(body, TWO, "cute", TWO_T, ptxas=OK)
        self.assertIn("unknown provenance", f.refusals[0]["msg"])

    def test_store_depending_on_the_elected_lane_refused(self):
        # H1, as for loads and reduces: under `@!elect` (every lane but the elected one)
        # or `elect && tab[lane] != 0`, which copies a store at the lane's coordinate
        # (bulk: address) makes depends on the lane elect.sync picks (unspecified)
        guard = TAB + "    setp.ne.u32 %p7, %r13, 0;\n    and.pred %p8, %p2, %p7;\n"
        bp = [
            ("ptr", "x", 0x10000),
            ("b32", "n", 7),
            ("b32", "m", 0),
            ("ptr", "tab", 0x300000),
        ]
        s2g = "    {p} cp.async.bulk.global.shared::cta.bulk_group [{g}], [%r6], 512;\n"
        for p in ("@!%p2", "@%p8"):
            for c, g, ok in (("%r1", "%rd5", False), ("%r7", "%rd3", p == "@!%p2")):
                with self.subTest(p=p, coordinate=c):
                    pre = PRE.format(n=2) + TMAP64 + LANES.format(k=32) + guard
                    f = run(
                        pre + f"    {p} {tma_op('store', c=c)};\n",
                        TWO,
                        "cute",
                        TWO_T,
                        ptxas=OK,
                    )
                    pre = PRE.format(n=1) + LANES.format(k=32) + guard + BULK_LANES
                    b = run(
                        pre + s2g.format(p=p, g=g),
                        bp,
                        "cute",
                        {"param0": 4, "param3": 4},
                        ptxas=OK,
                    )
                    if ok:  # the same key on every lane: one copy, whichever lane is elected
                        self.assertEqual((f.refusals, b.refusals), ([], []))
                        self.assertEqual(len(copies(f)), 1)
                        got = footprints(b, {"param1": 7}, tables={"param3": [5] * 32})
                        self.assertEqual(len([x for x in got if x[0] == "gstore"]), 1)
                    else:
                        for r in (f, b):
                            self.assertIn("elect", str(r.refusals))

    def test_ttgir_events(self):
        # desc events: a tma_store per distinct key, a tma_reduce and tma_load per lane
        body = PRE.format(n=5) + TMAP32 + LANES.format(k=32)
        for form, ev, var, n in (("store", "tma_store", "same", 1), ("store", "tma_store", "coord", 32),
                                 ("reduce", "tma_reduce", "same", 32), ("load", "tma_load", "coord", 32)):  # fmt: skip
            with self.subTest(form=form, var=var):
                f = run(
                    body + f"    {tma_op(form, **VARIANT[var])};\n",
                    TTGIR_PARAMS,
                    ptxas=OK,
                )
                self.assertEqual(f.refusals, [])
                got = [g for g in footprints(f, {"n": 5}) if g[0] == ev]
                self.assertEqual(len(got), n)
                self.assertEqual(kir_errors(f, {}), [])


BULK_LANES = """
    ld.param.u64 %rd3, [k_param_0];
    mul.wide.u32 %rd4, %r1, 512;
    add.s64 %rd5, %rd3, %rd4;
    selp.u32 %r17, 256, 512, %p4;
"""


def bulk(form, var, k, full=False, ptxas=OK):
    """A CuTe kernel: k lanes copy 512 bytes (128 elements of x) at x (or x + 512 lane:
    `dst`) from smem + 1024 (+ 128 lane: `smem`), `size`: 256 bytes on odd lanes."""
    g = "%rd5" if var == "dst" else "%rd3"
    src = "%r16" if var == "smem" else "%r6"
    size = "%r17" if var == "size" else "512"
    h, ho = (".L2::cache_hint", ", %rd10") if var == "hint" else ("", "")
    op = {"load": f"cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes{h} [{src}], [{g}], {size}, [%r2]{ho}",
          "store": f"cp.async.bulk.global.shared::cta.bulk_group{h} [{g}], [{src}], {size}{ho}",
          "reduce": f"cp.reduce.async.bulk.global.shared::cta.bulk_group{h}.add.u32 [{g}], [{src}], {size}{ho}"}[form]  # fmt: skip
    body = (
        PRE.format(n=1)
        + LANES.format(k=k)
        + BULK_LANES
        + f"    {'' if full else '@%p3 '}{op};\n"
    )
    return run(
        body,
        [("ptr", "x", 0x10000), ("b32", "n", 7)],
        "cute",
        {"param0": 4},
        ptxas=ptxas,
    )


class BulkLanes(unittest.TestCase):
    def test_counts(self):
        # g2s and reduce once per lane; s2g once per distinct (dst, src), not the policy
        keys = {"same": 1, "hint": 1, "dst": 32, "smem": 32}
        for form, ev in (("load", "gload"), ("reduce", "gatomic"), ("store", "gstore")):
            for var, n in keys.items():
                for k, full in ((1, False), (2, False), (32, False), (32, True)):
                    with self.subTest(form=form, var=var, k=k, full=full):
                        f = bulk(form, var, k, full)
                        self.assertEqual(f.refusals, [])
                        got = footprints(f, {"param1": 7})
                        want = min(k, n) if form == "store" else k
                        self.assertEqual(len([g for g in got if g[0] == ev]), want)
                        if form == "load":
                            loads = [g for g in got if g[0] == "tma_load"]
                            self.assertEqual(len(loads), k)
                            if var == "dst":  # each lane's element offset
                                self.assertEqual([g[2] for g in loads],
                                                 [[128 * lane] for lane in range(k)])  # fmt: skip
                        self.assertEqual(kir_errors(f, {"param0": 4}), [])

    def test_size_differing_per_lane_refused(self):
        # (the size is part of the s2g key on hardware; KIR needs one size per copy)
        for form in ("load", "store", "reduce"):
            f = bulk(form, "size", 32)
            self.assertIn("not a constant number", f.refusals[0]["msg"])


class Gate(unittest.TestCase):
    """Several lanes of a warp: translated only for a compiler in tma.ISSUE_PTXAS."""

    def test_unvalidated_compiler_refused(self):
        for ptxas, named in ((None, "(version unknown)"), ("13.3.99", "13.3.99")):
            for form in ("load", "store", "reduce"):
                with self.subTest(ptxas=ptxas, form=form):
                    f = lanes(form, "same", 2, ptxas=ptxas)
                    self.assertEqual(f.refusals[0]["kind"], "gap")
                    self.assertIn(named, f.refusals[0]["msg"])
                    self.assertIn("tma.ISSUE_PTXAS", f.refusals[0]["msg"])
                    f = bulk(form, "same", 32, ptxas=ptxas)
                    self.assertIn(named, f.refusals[0]["msg"])

    def test_one_lane_per_warp_needs_no_gate(self):
        # k = 1, and the elected lane of each of two warps: today's KIR, any compiler
        self.assertEqual(lanes("store", "coord", 1, ptxas=None).refusals, [])
        for body, p, fam, ts in ((PRE.format(n=5) + TMAP32 + LOAD.format(nbytes=4096), TTGIR_PARAMS, "ttgir", None),
                                 (PRE.format(n=1) + TMAP64 + STORE + REDUCE, CUTE_PARAMS, "cute", {"x": 2})):  # fmt: skip
            a = run(body, p, fam, ts, threads=64)
            b = run(body, p, fam, ts, threads=64, ptxas=OK)
            self.assertEqual((a.refusals, a.roles), ([], b.roles))

    def test_policy_default_from_the_launch_records(self):
        # TTGIR: frontend/ttgir/dump.py records `ptxas` in every LaunchRecord
        body = (
            PRE.format(n=5)
            + TMAP32
            + LANES.format(k=2)
            + f"    @%p3 {tma_op('store')};\n"
        )
        recs = records(TTGIR_PARAMS, "ttgir", 32)
        table = T.param_table(recs, "ttgir", None, [])
        for v, ok in ((OK, True), ("13.3.99", False)):
            pol = T.policy("ttgir", src_root=SRC, params=table, tensors={})
            f = T.translate(kernel(body, TTGIR_PARAMS, 32), "k",
                            [{**r, "ptxas": v} for r in recs], pol)  # fmt: skip
            self.assertEqual(f.refusals == [], ok)

    def test_unvalidated_target_refused(self):
        # measured on sm_100a (GB200) and sm_103a (GB300) only
        body = (
            PRE.format(n=2)
            + TMAP64
            + LANES.format(k=2)
            + f"    @%p3 {tma_op('store')};\n"
        )
        recs = records(TWO, "cute", 32)
        pol = T.policy("cute", src_root=SRC, params=T.param_table(recs, "cute", None, []),
                       tensors={n: {"elem_bytes": b} for n, b in TWO_T.items()}, ptxas=OK)  # fmt: skip
        for target, ok in (("sm_100a", True), ("sm_103a", True), ("sm_120a", False)):
            f = T.translate(
                kernel(body, TWO, 32).replace("sm_100a", target), "k", recs, pol
            )
            self.assertEqual(f.refusals == [], ok)
            if not ok:
                self.assertIn("for sm_120", f.refusals[0]["msg"])
                self.assertIn("tma.ISSUE_TARGETS", f.refusals[0]["msg"])

    def test_tkinfo_version(self):
        note = b"\x06\0\0\0\x30\0\0\0\xe8\x03\0\0NVIDIA\0\0ptxas | release 12.9, V12.9.83 | -O 3"
        self.assertEqual(tma.tkinfo_version(_elf({".text": b"\0" * 8, ".note.nv.tkinfo": note})),
                         "12.9.83")  # fmt: skip
        self.assertIsNone(tma.tkinfo_version(_elf({".text": b"\0" * 8})))
        self.assertIsNone(tma.tkinfo_version(b"\x7fELF"))


def _elf(sections: dict) -> bytes:
    """A minimal ELF64: the named sections and .shstrtab, section headers last."""
    blobs = list(sections.items()) + [(".shstrtab", b"")]
    names = b"\0" + b"".join(n.encode() + b"\0" for n, _ in blobs)
    blobs[-1] = (".shstrtab", names)
    out, heads, at = bytearray(64), [bytes(64)], 1
    for name, data in blobs:
        heads.append(
            struct.pack("<IIQQQQIIQQ", at, 7, 0, 0, len(out), len(data), 0, 0, 1, 0)
        )
        out += data
        at += len(name) + 1
    struct.pack_into("<Q", out, 0x28, len(out))
    struct.pack_into("<HHH", out, 0x3A, 64, len(heads), len(heads) - 1)
    return bytes(out) + b"".join(heads)


BULK = """
    ld.param.u64 %rd3, [k_param_0];
    mul.wide.s32 %rd4, %r5, 4;
    add.s64 %rd5, %rd3, %rd4;
    mov.u32 %r20, %cluster_ctarank;
    add.s32 %r21, %r2, 2048;
    mapa.shared::cluster.u32 %r22, %r21, %r20;
    mov.b32 %r23, 512;
"""


class Bulk(unittest.TestCase):
    P = [("ptr", "x", 0x10000), ("b32", "n", 7)]

    def test_g2s(self):
        body = (
            PRE.format(n=1)
            + BULK
            + """
    @%p2 mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r2], 512;
    @%p2 cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%r22], [%rd5], %r23, [%r2];
"""
        )
        f = run(body, self.P, tensors={"x": 4})
        self.assertEqual(f.refusals, [])
        (g,) = events(f, "gload")
        self.assertEqual(g["offsets"]["shape"], [128])
        self.assertEqual(f.descs["bulk_x_128"], {"kind": "bulk", "base": "x", "elem_bytes": 4, "box": [128],
                                                 "uses": ["tma_load"], "shape": None, "strides": None})  # fmt: skip
        (ld,) = events(f, "tma_load")
        self.assertEqual(
            (ld["desc"], footprints(f, {"n": 3})[-1]),
            ("bulk_x_128", ("tma_load", "bulk_x_128", [3])),
        )
        self.assertEqual(kir_errors(f, {"x": 4}), [])

    def test_s2g_and_reduce(self):
        body = (
            PRE.format(n=1)
            + BULK
            + """
    @%p2 cp.async.bulk.global.shared::cta.bulk_group [%rd5], [%r21], 512;
    @%p2 cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%rd5], [%r21], 512;
"""
        )
        f = run(body, self.P, tensors={"x": 4})
        self.assertEqual([e["ev"] for e in events(f)][1:], ["gstore", "gatomic"])
        self.assertEqual(writes(body, self.P), ["x"])
        f = run(body.replace("add.f32", "add.u64"), self.P, tensors={"x": 4})
        self.assertIn("bulk reduce of u64 values", f.refusals[0]["msg"])
        f = run(body.replace("%r23, [%r2]", "%r5, [%r2]").replace("[%r21], 512", "[%r21], %r5"),
                self.P, tensors={"x": 4})  # fmt: skip
        self.assertIn("not a constant number", f.refusals[0]["msg"])

    def test_reduce_add_noftz(self):
        # f16/bf16 .add.noftz (subnormals kept) is an add: a gatomic of the 2-byte
        # elements (FA4's f16 dQ accumulator); its type still has to be the elements'
        red = "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16"
        body = PRE.format(n=1) + BULK + f"    @%p2 {red} [%rd5], [%r21], 512;\n"
        f = run(body, self.P, tensors={"x": 2})
        self.assertEqual(f.refusals, [])
        (g,) = events(f, "gatomic")
        self.assertEqual(g["offsets"]["shape"], [256])
        f = run(body, self.P, tensors={"x": 4})
        self.assertIn("bulk reduce of f16 values", f.refusals[0]["msg"])


# In-kernel descriptor: 32 threads zero the staging buffer smem+4096, thread 0 replaces
# the fields (a masked branch), the warp copies it to this CTA's scratch (+ OFF), and the
# elected lane stores the box at (col 0, row ROW) through its generic address.
STAGE = """
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, smem;
    ld.param.u64 %rd10, [k_param_3];
    mov.u32 %r30, %ctaid.x;
    mul.wide.u32 %rd11, %r30, 256;
    add.s64 %rd12, %rd10, %rd11;
    ld.param.b32 %r5, [k_param_2];
    add.s32 %r40, %r2, 4096;
    shl.b32 %r41, %r1, 2;
    add.s32 %r42, %r40, %r41;
    mov.b32 %r43, 0;
    setp.eq.u32 %p1, %r1, 0;
    elect.sync %r3|%p2, -1;
"""
FILL = """
    st.shared.b32 [%r42], %r43;
    @!%p1 bra SKIP{k};
    cvt.u64.u32 %rd15, %r40;
    ld.param.u64 %rd14, [k_param_{p}];
    tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%rd15], %rd14;
    tensormap.replace.tile.rank.shared::cta.b1024.b32 [%rd15], 0x1;
    tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%rd15], 0x0, 32;
    tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%rd15], 0x1, 64;
    tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%rd15], 0x0, 128;
    tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%rd15], 0x1, %r5;
    tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%rd15], 0x0, 256;
    tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [%rd15], 0x0, 1;
    tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [%rd15], 0xa;
    tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [%rd15], 0x2;
SKIP{k}:
    cvt.u64.u32 %rd16, %r40;
    add.s64 %rd{d}, %rd12, {off};
    tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%rd{d}], [%rd16], 0x80;
"""
USE = """
    cvta.global.u64 %rd{g}, %rd{d};
    mov.b32 %r50, 0;
    mov.b32 %r51, {row};
    add.s32 %r52, %r2, 8192;
    @%p2 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd{g}, {{%r50, %r51}}], [%r52];
"""
KP = [
    ("ptr", "dv", 0x20000),
    ("ptr", "dk", 0x40000),
    ("b32", "rows", 70),
    ("ptr", "__global_scratch", "scratch"),
]
KT = {"dv": 2, "dk": 2}


def fill(k, p, d, off):
    return FILL.format(k=k, p=p, d=d, off=off)


def use(d, g, row=0):
    return USE.format(d=d, g=g, row=row)


class InKernel(unittest.TestCase):
    def test_clipped_store(self):
        f = run(STAGE + fill(0, 0, 20, 0) + use(20, 21, row=32), KP, tensors=KT)
        self.assertEqual(f.refusals, [])
        self.assertEqual((f.written, f.descs), (["dv"], {}))
        ((ev, t, offs),) = footprints(f, {"rows": 70})
        # rows 32..69 (box 64 rows clipped at the extent 70), columns 0..31, row stride 128
        self.assertEqual(
            (ev, t, offs),
            (
                "gstore",
                "dv",
                sorted(r * 128 + c for r in range(32, 70) for c in range(32)),
            ),
        )
        self.assertEqual(kir_errors(f, KT), [])
        self.assertEqual(writes(STAGE + fill(0, 0, 20, 0) + use(20, 21), KP), ["dv"])

    def test_buffer_reused_for_two_descriptors(self):
        body = (
            STAGE + fill(0, 0, 20, 0) + fill(1, 1, 22, 128) + use(22, 23) + use(20, 21)
        )
        f = run(body, KP, tensors=KT)
        self.assertEqual(f.refusals, [])
        self.assertEqual([t for _e, t, _o in footprints(f, {"rows": 70})], ["dk", "dv"])
        self.assertEqual(writes(body, KP), ["dk", "dv"])

    def test_refusals(self):
        f = run(STAGE + fill(0, 0, 20, 0) + use(24, 21).replace("cvta.global.u64 %rd21, %rd24", "add.s64 %rd24, %rd12, 128;\n    cvta.global.u64 %rd21, %rd24"), KP, tensors=KT)  # fmt: skip
        self.assertIn("unknown provenance", f.refusals[0]["msg"])
        f = run(STAGE + fill(0, 0, 20, 0).replace("    st.shared.b32 [%r42], %r43;\n", "") + use(20, 21), KP, tensors=KT)  # fmt: skip
        self.assertIn("not zeroed by one warp", f.refusals[0]["msg"])
        branchy = (
            fill(0, 0, 20, 0).replace(
                "SKIP0:\n", "SKIP0:\n    setp.eq.s32 %p5, %r5, 3;\n    @%p5 bra OUT;\n"
            )
            + "OUT:\n"
        )
        f = run(STAGE + branchy + use(20, 21), KP, tensors=KT)
        self.assertIn("another branch or loop", f.refusals[0]["msg"])
        load = "    @%p2 cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%r52], [%rd21, {%r50, %r51}], [%r2];\n"
        f = run(
            STAGE + fill(0, 0, 20, 0) + use(20, 21).split("    @%p2")[0] + load,
            KP,
            tensors=KT,
        )
        self.assertIn("in-kernel descriptor (gap", f.refusals[0]["msg"])
        bad = KP[:3] + [("ptr", "__global_scratch", 0x80000)]  # not marked scratch
        self.assertIsNone(writes(STAGE + fill(0, 0, 20, 0) + use(20, 21), bad))
        # a replace under a branch on data is no branch whose arms emit nothing
        # (interp._inert): taking it as unconditional would clip the store at 40 rows
        cut = "    cvt.u64.u32 %rd16, %r40;\n"
        data = "    ld.shared.u32 %r60, [%r2+64];\n    setp.eq.s32 %p6, %r60, 0;\n" \
            "    @%p6 bra KEEP;\n    tensormap.replace.tile.global_dim.shared::cta.b1024.b32" \
            " [%rd16], 0x1, 40;\nKEEP:\n"  # fmt: skip
        f = run(STAGE + fill(0, 0, 20, 0).replace(cut, cut + data) + use(20, 21, row=32),
                KP, tensors=KT)  # fmt: skip
        self.assertIn("reaches a branch condition", f.refusals[0]["msg"])
        # a tensormap edited in place in global memory (quack's GemmSm100): no fields known
        glob = fill(0, 0, 20, 0).replace(
            "global_dim.shared::cta.b1024.b32 [%rd15], 0x1, %r5",
            "global_dim.global.b1024.b32 [%rd14], 0x1, %r5",
        )
        f = run(STAGE + glob + use(20, 21), KP, tensors=KT)
        self.assertIn(
            "global_dim of a tensormap in global memory", f.refusals[0]["msg"]
        )

    def test_one_dimensional_without_rank(self):
        # Triton replaces no rank for a 1-D descriptor: the zeroed field, rank - 1 = 0
        one = fill(0, 0, 20, 0).replace("0x1, %r5", "0x0, %r5")
        for line in ("rank.shared::cta.b1024.b32 [%rd15], 0x1",
                     "box_dim.shared::cta.b1024.b32 [%rd15], 0x1, 64",
                     "global_dim.shared::cta.b1024.b32 [%rd15], 0x0, 128",
                     "global_stride.shared::cta.b1024.b64 [%rd15], 0x0, 256"):  # fmt: skip
            one = one.replace(f"    tensormap.replace.tile.{line};\n", "")
        store = "    @%p2 cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [%rd21, {%r51}], [%r52];\n"  # fmt: skip
        f = run(STAGE + one + use(20, 21, row=48).split("    @%p2")[0] + store,
                KP, tensors=KT)  # fmt: skip
        self.assertEqual(f.refusals, [])
        # elements 48..79 of the 32-element box, clipped at the extent 70
        self.assertEqual(
            footprints(f, {"rows": 70}), [("gstore", "dv", list(range(48, 70)))]
        )

    def test_other_shared_writes_close_the_buffer(self):
        # a generic st/atom/red reaching the buffer through cvta.shared (its rank word:
        # 16 = rank 2 - 1), or an atom/red.shared, closes it, as a st.shared does; one
        # of static bytes outside it does not; any write of tma.SHARED_WRITES closes it;
        # so does a st.shared, stmatrix or atom/red.shared under a data predicate (which
        # skips the instruction's execution)
        zero = "    st.shared.b32 [%r42], %r43;\n"
        fence = "    cvt.u64.u32 %rd16, %r40;\n"
        gen = "    cvt.u64.u32 %rd30, %r40;\n    cvta.shared.u64 %rd31, %rd30;\n"
        far = "    mul.wide.u32 %rd32, %r5, 4;\n    add.s64 %rd33, %rd31, %rd32;\n"
        dv = "    ld.param.u64 %rd34, [k_param_0];\n"
        data = "    ld.shared.u32 %r60, [%r2+64];\n    setp.eq.s32 %p6, %r60, 0;\n"
        for at, extra, ok in (
            (zero, data + "    @%p6 st.shared.u32 [%r40+4], 16;\n", False),
            (zero, data + "    @%p6 st.shared.u32 [%r40+128], 16;\n", True),
            (
                zero,
                data
                + "    @%p6 stmatrix.sync.aligned.m8n8.x1.shared.b16 [%r40], {%r43};\n",
                False,
            ),  # fmt: skip
            (zero, gen + "    @%p1 st.u32 [%rd31+4], 16;\n", False),
            (fence, gen + "    @%p1 st.u32 [%rd31+4], 16;\n", False),
            (zero, gen + "    @%p1 red.add.u32 [%rd31+4], 16;\n", False),
            (fence, gen + "    @%p1 atom.add.u32 %r61, [%rd31+4], 16;\n", False),
            (zero, "    @%p1 atom.shared.add.u32 %r61, [%r40+4], 16;\n", False),
            (fence, "    @%p1 red.shared::cta.add.u32 [%r40+4], 16;\n", False),
            (zero, data + "    @%p6 red.shared.or.b32 [%r40+4], 16;\n", False),
            (zero, "    @%p1 atom.shared::cta.exch.b32 %r61, [%r40+128], 16;\n", True),
            (zero, gen + far + "    @%p1 st.u32 [%rd33], 16;\n", False),  # not static
            (zero, gen + "    @%p1 st.u32 [%rd31+128], 16;\n", True),
            (
                zero,
                "    stmatrix.sync.aligned.m8n8.x1.shared.b16 [%r2], {%r43};\n",
                False,
            ),  # fmt: skip
            (zero, dv + "    cp.async.ca.shared.global [%r2], [%rd34], 4;\n", False),
            (fence, "    mbarrier.init.shared::cta.b64 [%r2+64], 1;\n", False),
        ):
            with self.subTest(extra=extra.split("\n")[-2]):
                body = STAGE + fill(0, 0, 20, 0).replace(at, at + extra) + use(20, 21)
                f = run(body, KP, tensors=KT)
                if ok:
                    self.assertEqual(f.refusals, [])
                else:
                    self.assertIn("not zeroed by one warp first", f.refusals[0]["msg"])

    def test_zeroed_through_a_register_known_zero(self):
        # LLVM zeroes the buffer with a register an earlier loop's body sets to 0, read
        # after the loop only: every edge leaves it at 0, so it is 0 there (interp.py)
        loop = "    setp.ne.s32 %p8, %r5, 3;\n    mov.b32 %r44, 0;\nZL:\n" \
            "    mov.b32 %r43, 0;\n    add.s32 %r44, %r44, 1;\n" \
            "    setp.lt.s32 %p7, %r44, %r5;\n    @%p7 bra ZL;\n"  # fmt: skip
        zero = "    mov.b32 %r43, 0;\n"
        f = run(STAGE.replace(zero, loop) + fill(0, 0, 20, 0) + use(20, 21, row=32),
                KP, tensors=KT)  # fmt: skip
        self.assertEqual(f.refusals, [])
        self.assertEqual([t for _e, t, _o in footprints(f, {"rows": 70})], ["dv"])
        # another value written after the loop, or 0 under a predicate in it: refused
        for bad in (loop + "    @%p8 mov.b32 %r43, 1;\n",
                    loop.replace("ZL:\n" + zero, "ZL:\n    @%p8 mov.b32 %r43, 0;\n")):  # fmt: skip
            f = run(
                STAGE.replace(zero, bad) + fill(0, 0, 20, 0) + use(20, 21),
                KP,
                tensors=KT,
            )
            self.assertIn("not zeroed by one warp", f.refusals[0]["msg"])


class PackedSubByte(unittest.TestCase):
    """Packed sub-byte CuTe descriptors (tma._PACKED, measured by test_gpu_tma_subbyte.py):
    KIR in bytes of the byte tensor q, the dim-0 coordinate (here param1) asserted a multiple
    of the format's granularity and converted, the packed bytes as transaction bytes; a
    store or reduce the hardware traps on is an assert false."""

    def run_q(self, dtype, dims, strides, box, body, launches=None):
        t = launches or packed(dtype, dims, strides, box)
        p = [("tmap", "t", t), ("b32", "n", 7)]
        body = PRE.format(n=1) + TMAP64 + body.replace("{%r7, %r5}", "{%r5, %r7}")
        return run(body, p, "cute", {"q": 1}, launches=len(launches or [0]))

    def check_load(self, dtype, strides, box, kir_box, row, n, cols):
        f = self.run_q(dtype, [256, 64], strides, [128, 16], LOAD.format(nbytes=0))
        self.assertEqual(f.refusals, [])
        d = f.descs["tma0_q"]
        self.assertEqual((d["box"], d["elem_bytes"]), (kir_box, 1))  # tx: prod(box)
        self.assertEqual(f.values[0]["tma0_q.shape.1"], row)  # dim-0 extent in bytes
        (check,) = [s for s in statements(f.roles[0]["body"]) if s["op"] == "assert"]
        self.assertIn(f"not a multiple of {box} {dtype} elements", check["msg"])
        self.assertEqual(check["cond"]["a"]["o"], "srem")  # on the format's coordinate
        self.assertEqual(check["cond"]["a"]["b"]["v"], box)
        got = footprints(f, {**f.values[0], "param1": n})
        # rows 32..47 (dim-1 coordinate 32, 16 box rows), the byte columns of the box
        want = sorted(r * strides[0] + c for r in range(32, 48) for c in cols)
        self.assertEqual([g[2] for g in got if g[0] == "gload"], [want])
        self.assertEqual(kir_errors(f, {"q": 1}), [])

    def test_align8b_load(self):
        # 128 4-bit elements = 64 bytes a row; element column 64 is byte 32
        self.check_load("16U4_ALIGN8B", [128], 32, [16, 64], 128, 64, range(32, 96))

    def test_align16b_load(self):
        # packed bytes too (its shared-memory copy is twice as wide); granularity 128
        self.check_load("16U4_ALIGN16B", [128], 128, [16, 64], 128, 128, range(64, 128))

    def test_u6_load(self):
        # 6-bit: 128 elements = 96 bytes; element column 128 is byte 96 (x 3 / 4)
        self.check_load("16U6_ALIGN16B", [192], 128, [16, 96], 192, 128, range(96, 192))

    def test_store_and_reduce(self):
        f = self.run_q("16U4_ALIGN8B", [256, 64], [128], [128, 16], STORE + REDUCE)
        self.assertEqual(f.refusals, [])
        self.assertEqual([e["ev"] for e in events(f)][1:], ["gstore"])
        traps = [s for s in statements(f.roles[0]["body"]) if s["op"] == "assert"
                 and s["cond"] == {"e": "const", "w": 1, "v": 0}]  # fmt: skip
        self.assertEqual(len(traps), 1)
        self.assertIn("TMA reduce through tma0_q, a 16U4_ALIGN8B descriptor: the hardware traps",
                      traps[0]["msg"])  # fmt: skip
        f = self.run_q("16U4_ALIGN16B", [256, 64], [128], [128, 16], STORE)
        self.assertEqual([e["ev"] for e in events(f)][1:], [])
        self.assertIn("TMA store through tma0_q, a 16U4_ALIGN16B descriptor",
                      [s for s in statements(f.roles[0]["body"]) if s["op"] == "assert"][-1]["msg"])  # fmt: skip

    def test_launches_with_two_formats_refused(self):
        two = [
            packed(dt, [256, 64], [128], [128, 16])
            for dt in ("16U4_ALIGN8B", "16U4_ALIGN16B")
        ]
        f = self.run_q(None, None, None, None, LOAD.format(nbytes=0), launches=two)
        self.assertIn("differ in the descriptor's box or format", f.refusals[0]["msg"])

    def test_triton_and_in_kernel_packed_refused(self):
        p = [("tmap", "desc_x", {**META, "host_dtype": 14})] + TTGIR_PARAMS[1:]
        f = run(PRE.format(n=5) + TMAP32 + LOAD.format(nbytes=4096), p)
        self.assertIn("packed sub-byte Triton descriptor", f.refusals[0]["msg"])
        u4 = fill(0, 0, 20, 0).replace("[%rd15], 0xa;", "[%rd15], 0xd;")
        f = run(STAGE + u4 + use(20, 21), KP, tensors=KT)
        self.assertIn("elemtype 13 (13-15, packed sub-byte: gap)", f.refusals[0]["msg"])


class Refusals(unittest.TestCase):
    def test_cluster_and_multicast(self):
        mc = LOAD.format(nbytes=4096).replace("bytes [%r6]", "bytes.multicast::cluster [%r6]").replace(
            "[%r2];\n", "[%r2], %rs1;\n").replace("cp.async.bulk.tensor.2d.shared::cta", "cp.async.bulk.tensor.2d.shared::cluster")  # fmt: skip
        f = run(
            PRE.format(n=5).replace("    elect", "    .reg .b16 %rs<2>;\n    elect")
            + TMAP32
            + mc,
            TTGIR_PARAMS,
        )
        self.assertIn("multicast TMA load outside a cluster", f.refusals[0]["msg"])

    def test_generic_access_through_parameters(self):
        f = run(
            PRE.format(n=5)
            + TMAP32
            + "    ld.u64 %rd9, [%rd2+8];\n    st.global.b64 [%rd9], %rd9;\n",
            TTGIR_PARAMS,
        )
        self.assertIn("only TMA instructions", f.refusals[0]["msg"])
        f = run(PRE.format(n=5) + "    ld.param.b32 %r9, [k_param_0];\n", TTGIR_PARAMS)
        self.assertIn("tensormap parameter", f.refusals[0]["msg"])
        body = PRE.format(n=5) + "    mov.b64 %rd1, k_param_5;\n    cvta.param.u64 %rd2, %rd1;\n    ld.u32 %r9, [%rd2];\n    st.shared.b32 [%r2+64], %r9;\n" + TMAP32 + STORE.replace("%r5}", "%r9}")  # fmt: skip
        self.assertIn("slice soundness", run(body, TTGIR_PARAMS).refusals[0]["msg"])


class Helpers(unittest.TestCase):
    def test_desc_base(self):
        def rec(*cands):
            return {"params": [{"tensormap": {"base": [{"tensor": t, "storage": s, "elem_off": 0, "match": m}
                                                       for t, s, m in cands]}}]}  # fmt: skip

        one = rec(("x", [0, 8], "exact"), ("y", [0, 64], "inside"))
        self.assertEqual(tma.desc_base([one], 0, [], {}), ("x", []))
        alias = rec(("mK", [0, 8], "exact"), ("mK_dQ", [0, 8], "exact"))
        taken: dict = {}
        leaves = ["mQ", "mK", "mK_dQ"]
        self.assertEqual(
            [tma.desc_base([alias], 0, leaves, taken) for _ in range(2)],
            [("mK", ["mK", "mK_dQ"]), ("mK_dQ", ["mK", "mK_dQ"])],
        )
        two = rec(("a", [0, 8], "exact"), ("b", [0, 16], "exact"))
        self.assertEqual(tma.desc_base([two], 0, ["a", "b"], {}), (None, []))
        self.assertEqual(
            tma.desc_base([one, rec(("z", [0, 8], "exact"))], 0, [], {}), (None, [])
        )
        self.assertEqual(tma.desc_base([rec()], 0, [], {}), (None, []))

    def test_aliased_base_by_layout(self):
        # one storage passed as acc [2, 4096] and acc_tma [64, 128]: the descriptor of
        # extents [64, 128] is over acc_tma, though acc comes first in jit order
        def cands(addr):
            return [{"tensor": n, "storage": [addr, addr + 32768], "elem_off": 0, "match": "exact"}
                    for n in ("acc", "acc_tma")]  # fmt: skip

        dec = {
            **decoded([128, 64], [512], [32, 16], elem_bits=32),
            "base": cands(0x10000),
        }
        p = [("tmap", "t", dec), ("b32", "n", 7)]
        recs = records(p, "cute", 32)
        table = T.param_table(recs, "cute", None, ["acc", "acc_tma"])
        self.assertEqual(
            (table[0]["base"], table[0]["aliases"]), ("acc", ["acc", "acc_tma"])
        )
        shapes = {"acc": {"elem_bytes": 4, "shape": [2, 4096], "strides": [4096, 1]},
                  "acc_tma": {"elem_bytes": 4, "shape": [64, 128], "strides": [128, 1]}}  # fmt: skip
        pol = T.policy("cute", src_root=SRC, params=table, tensors=shapes)
        body = PRE.format(n=1) + TMAP64 + REDUCE
        f = T.translate(kernel(body, p, 32), "k", recs, pol)
        self.assertEqual((f.refusals, f.written), ([], ["acc_tma"]))

    def test_segment_dims(self):
        desc = [([2, 300, 64], [64, 128, 1]), ([2, 500, 64], [64, 128, 1])]
        tensor = [([300, 2, 64], [128, 64, 1]), ([500, 2, 64], [128, 64, 1])]
        self.assertEqual(tma.segment_dims(desc, tensor), [1, 0, 2])
        self.assertEqual(tma.segment_dims([([4], [1])], [([4, 4], [1, 1])]), [None])


if __name__ == "__main__":
    unittest.main()
