# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the warp-specialize captures (frontend/ptx/interp.py, frontend_ptx.md
"Threads and roles") against every kind of shared-memory write, through translate().

    python3 -I tests/ptx/test_captures.py      (or the buck target :test_captures)

A Triton direct dispatch (8 warps: the default 0-3, partition0 = warp 6, partition1 =
warp 7, padding 4-5): the default stores `n + 7` at global_smem+32 before its first fork
barrier, partition0 reads it between its fork barriers and stores at x[capture], so the
capture forwarded translates and an unknown one is Data that reaches an address. Each
write kind over the capture's bytes: by the default before its capture store (an
asynchronous write may land after it), between the store and its first fork barrier
(the capture becomes unknown), and by partition1 before its second fork barrier after
partition0 read it (refused); a write elsewhere leaves the capture alone. Kinds: a TMA
landing, a cp.async and a bulk copy, a generic store through cvta.shared (the direct
dispatch does not refuse it) or a data address (any byte), atom/red.shared, stmatrix, tcgen05.alloc's address slot, an
st.async landing or a shared::cluster bulk reduce from the other CTA of a cluster (after
the writer's second fork barrier: no barrier of this CTA orders it), a CLC response, mbarrier objects (their arrays, after
every role). A footprint through a ring-buffer index (x % k as LLVM writes it) is bounded,
unless the arithmetic may wrap at its width; one through a strength-reduced loop-carried
offset keeps the capture under a KIR assume; one under elect.sync covers every lane that
may be elected; a TMA landing through a packed sub-byte descriptor may touch any byte.
Each arm of an if (on a uniform value or per warp) counts the default's fork barriers
from the If. The op table's shared-memory writers are all covered: each is one of
interp.CAPTURE_WRITERS and reaches a capture hook through the dispatch of its class.
"""

import re
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest import mock

if __package__ in (None, ""):
    sys.dont_write_bytecode = True
    _root = ModuleType("lkv_root")
    _root.__path__ = [str(Path(__file__).resolve().parents[2])]
    sys.modules["lkv_root"] = _root
    __package__ = "lkv_root.tests.ptx"  # noqa: A001

from ...frontend.ptx import (  # noqa: E402
    cluster,
    expr as E,
    interp,
    ops,
    tma as ptx_tma,
    translate as T,
)

SRC = "/src"
META = {"address": 0x10000, "host_dtype": 10, "elem_size": 2, "shape": [256, 128],
        "strides": [128, 1], "box": [64, 32], "swizzle": 3, "padding": 0}  # fmt: skip
# (kind, name, value): a host TMA descriptor (box 64 x 32 bf16: 4096 bytes), x, n
PARAMS = [("tmap", "desc_x", META), ("b32", "desc_x.shape.0", 256),
          ("b32", "desc_x.shape.1", 128), ("b64", "desc_x.stride.0", 128),
          ("b64", "desc_x.stride.1", 1), ("ptr", "x", 0x20000), ("b32", "n", 7)]  # fmt: skip
_DECL = {"tmap": ".param .align 64 .b8 k_param_{i}[128]",
         "ptr": ".param .u64 .ptr .global .align 1 k_param_{i}",
         "b32": ".param .b32 k_param_{i}", "b64": ".param .b64 k_param_{i}"}  # fmt: skip
_SIZE = {"tmap": 128, "ptr": 8, "b32": 4, "b64": 8}
_KIND = {"tmap": "tensormap", "ptr": "ptr", "b32": "scalar", "b64": "scalar"}

BODY = """
    mov.u32 %r1, %tid.x;
    shr.u32 %r2, %r1, 5;
    shfl.sync.idx.b32 %r3, %r2, 0, 31, -1;
    ld.param.u64 %rd1, [k_param_5];
    ld.param.b32 %r20, [k_param_6];
    mov.b64 %rd5, k_param_0;
    cvta.param.u64 %rd6, %rd5;
    mov.u32 %r30, global_smem;
    cvt.u64.u32 %rd8, %r30;
    cvta.shared.u64 %rd9, %rd8;
    setp.lt.u32 %p1, %r3, 4;
    @!%p1 bra $HDR;
    setp.eq.u32 %p5, %r1, 0;
    elect.sync %r31|%p6, -1;
    setp.eq.u32 %p7, %r3, 0;
    and.pred %p8, %p7, %p6;
    @%p5 mbarrier.init.shared::cta.b64 [%r30+64], 1;
{early}    add.s32 %r40, %r20, 7;
    st.shared.b32 [global_smem+32], %r40;
{default}    barrier.sync 1;
    barrier.sync 1;
    @%p5 st.global.b32 [%rd1], %r1;
    bra.uni $RET;
$HDR:
    barrier.sync 1;
    setp.gt.u32 %p2, %r3, 6;
    @!%p2 bra $L2;
    setmaxnreg.dec.sync.aligned.u32 24;
    elect.sync %r32|%p9, -1;
    setp.eq.u32 %p10, %r1, 224;
{late}    barrier.sync 1;
{after}    st.global.b32 [%rd1+4], %r3;
    bra.uni $RET;
$L2:
    setp.eq.b32 %p3, %r3, 6;
    @!%p3 bra $PAD;
    setmaxnreg.dec.sync.aligned.u32 24;
    ld.shared.b32 %r4, [global_smem+32];
    barrier.sync 1;
    mul.wide.s32 %rd2, %r4, 4;
    add.s64 %rd3, %rd1, %rd2;
    setp.eq.u32 %p4, %r1, 192;
    @%p4 st.global.b32 [%rd3], %r3;
    bra.uni $RET;
$PAD:
    setmaxnreg.dec.sync.aligned.u32 24;
    barrier.sync 1;
$RET:
    ret;
"""


# per hook: the predicate of one thread (`{one}`) and of an elected lane (`{elect}`) there
_PREDS = {"early": ("%p5", "%p8"), "default": ("%p5", "%p8"), "late": ("%p10", "%p9"),
          "after": ("%p10", "%p9")}  # fmt: skip


def kernel(cluster: int = 1, text: str = BODY, **hooks) -> str:
    decl = ", ".join(_DECL[k].format(i=i) for i, (k, _n, _v) in enumerate(PARAMS))
    body = text.format(**{h: hooks.get(h, "").replace("{one}", _PREDS[h][0]).replace(
        "{elect}", _PREDS[h][1]) for h in _PREDS})  # fmt: skip
    req = f".reqnctapercluster {cluster}, 1, 1\n" if cluster > 1 else ""
    return f"""
.version 9.3
.target sm_100a
.address_size 64
.extern .shared .align 1024 .b8 global_smem[];
.visible .entry k({decl})
.reqntid 256
{req}{{
    .reg .pred %p<32>;
    .reg .b32 %r<100>;
    .reg .b64 %rd<64>;
    .loc 1 1 0
{body}
}}
    .file 1 "{SRC}/k.py"
"""  # fmt: skip


def run(cluster: int = 1, text: str = BODY, **hooks):
    params = []
    for i, (k, name, v) in enumerate(PARAMS):
        p = {"i": i, "ptx": f"k_param_{i}", "name": name, "kind": _KIND[k],
             "size": _SIZE[k], "value": v}  # fmt: skip
        params.append(p)
    recs = [{"block": [256, 1, 1], "cluster": [cluster, 1, 1], "grid": [2, 1, 1],
             "params": params}]  # fmt: skip
    table = T.param_table(recs, "ttgir", None, [])
    pol = T.policy(
        "ttgir", src_root=SRC, params=table, tensors={"x": {"elem_bytes": 4}}
    )
    return T.translate(kernel(cluster, text, **hooks), "k", recs, pol)


def _statements(stmts):
    for s in stmts:
        yield s
        for k in ("then", "else", "body"):
            yield from _statements(s.get(k, []))


def refusal(f) -> str:
    return f.refusals[0]["msg"] if f.refusals else ""


def tma(at: int) -> str:
    return ("    @{elect} cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
            f"[global_smem+{at}], [%rd6, {{%r20, %r20}}], [global_smem+64];\n")  # fmt: skip


def reaches_a_hook(mn: str, cls: str) -> bool:
    """mn, an op-table row of class cls, reaches _capture_store or _capture_clobber at a
    shared address through the dispatch Translator.execute takes for its class (event:
    _capture_writes, drop: _shared_drop)."""
    hits = []
    smem = E.var("%s:global_smem", 32)
    tma_ = SimpleNamespace(staging_store=lambda *a: None, close_staging=lambda *a: None)
    mem = SimpleNamespace(address=lambda *a: smem, execute=lambda *a: None, tma=tma_)
    stub = SimpleNamespace(k=SimpleNamespace(mem=mem, captures={}), uniform=lambda v: v,
                           operand=lambda *a: E.const(16, 32),
                           _footprint=lambda *a: ("global_smem", 0, 16),
                           _capture_clobber=lambda *a, **kw: hits.append(mn),
                           _capture_store=lambda *a: hits.append(mn),
                           _handoff_store=lambda *a: None, _tracked=lambda: True)  # fmt: skip
    tr = cast(interp.Translator, stub)  # only the attributes the two dispatchers read
    i = SimpleNamespace(opcode=mn + ".b32", operands=(("mem", None),) * 3)
    with mock.patch.object(cluster, "_address", lambda *a: smem):
        if cls == "event":
            interp.Translator._capture_writes(tr, i, mn)
        elif cls == "drop":
            interp.Translator._shared_drop(tr, i, mn)
    return bool(hits)


UNKNOWN = "reaches"  # the capture is Data: partition0's store address
OVERWRITE = "may overwrite a warp-specialize capture"


class Captures(unittest.TestCase):
    def check(
        self, write: str, overlaps: bool = True, asynchronous: bool = False, hooks=None
    ):
        """The write `write` at each hook: refused / Data where it overlaps, else no
        refusal (the capture forwarded); a synchronous write before the capture store
        is overwritten by it."""
        for hook in hooks or ("early", "default", "late"):
            f = run(1, **{hook: write})
            hit = overlaps and (asynchronous or hook != "early")
            want = (
                "" if not hit else OVERWRITE if hook in ("late", "after") else UNKNOWN
            )
            if want:
                self.assertIn(want, refusal(f), (hook, write))
            else:
                self.assertEqual(f.refusals, [], (hook, write))

    def test_forwarded(self):
        f = run()
        self.assertEqual(f.refusals, [])
        self.assertEqual(
            [r["name"] for r in f.roles], ["default", "partition0", "partition1"]
        )

    def test_tma_landing(self):
        # box 64 x 32 bf16 = 4096 bytes from the destination: [0, 4096) holds the capture
        self.check(tma(0), asynchronous=True)
        self.check(tma(1024), overlaps=False)

    def test_packed_tma_landing(self):
        # a packed sub-byte format's shared-memory copy may be wider than its packed bytes
        # (16U4_ALIGN16B, 16U6_ALIGN16B): such a landing reports no byte count, so it may
        # touch every capture; a plain one reports its box (16 x 64 bytes)
        class Stop(Exception):
            pass

        got: list[int | None] = []

        def landing(i, a, nbytes, remote):
            got.append(nbytes)
            raise Stop

        cases = {None: 1024, "16U4_ALIGN16B": None, "16U4_ALIGN8B": None}
        for packed, want in cases.items():
            got.clear()
            d = ptx_tma.Desc("t", "host", "q", 1, [16, 64], packed=packed)
            tm = SimpleNamespace(_desc=lambda *a, d=d: d, _coords=lambda *a: [])
            tr = SimpleNamespace(capture_landing=landing)
            i = SimpleNamespace(operands=(None, None, None))
            with (
                mock.patch.object(cluster, "load_slot", lambda *a: (None, None, {})),
                mock.patch.object(cluster, "_address", lambda *a: None),
                self.assertRaises(Stop),
            ):
                ptx_tma.Tma._tensor_load(
                    cast(ptx_tma.Tma, tm), cast(interp.Translator, tr), i
                )
            self.assertEqual(got, [want], packed)

    def test_cp_async_and_bulk(self):
        cp = "    @{one} cp.async.ca.shared.global [global_smem+AT], [%rd1], 4;\n"
        self.check(cp.replace("AT", "32"), asynchronous=True)
        self.check(cp.replace("AT", "36"), overlaps=False)
        bulk = ("    @{elect} cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                "[global_smem+AT], [%rd1], 16, [global_smem+64];\n")  # fmt: skip
        self.check(bulk.replace("AT", "24"), asynchronous=True)  # [24, 40)
        self.check(bulk.replace("AT", "16"), overlaps=False)  # [16, 32)
        self.check(bulk.replace("AT", "128"), overlaps=False)

    def test_generic_store_in_a_direct_dispatch(self):
        self.check("    @{one} st.u32 [%rd9+32], %r1;\n")
        self.check("    @{one} st.u32 [%rd9+256], %r1;\n", overlaps=False)
        self.check("    @{one} atom.add.u32 %r60, [%rd9+34], 1;\n")

    def test_generic_store_through_data(self):
        # a generic store whose address is data may write any shared byte: refused once a
        # worker read a capture, whatever the memory model does with the access itself
        w = "    ld.shared.u64 %rd30, [global_smem+512];\n    @{one} st.u32 [%rd30], %r1;\n"
        self.assertIn(OVERWRITE, refusal(run(late=w)))

    def test_elected_lane_footprint(self):
        # a write under elect.sync covers every lane that may be elected (unspecified),
        # not lane 0's bytes: at 60 - 4 lane (generic and st.shared), the elected lane 7
        # writes the capture; one at an offset that depends on the elected lane (28 + 4
        # lane if elected, else 0) touches any byte
        lane = ("    and.b32 %r50, %r1, 31;\n    mul.lo.s32 %r51, %r50, -4;\n"
                "    add.s32 %r52, %r51, 60;\n")  # fmt: skip
        self.check(lane + "    cvt.u64.u32 %rd50, %r52;\n    add.u64 %rd51, %rd9, %rd50;\n"
                   "    @{elect} st.b32 [%rd51], %r20;\n")  # fmt: skip
        self.check(lane + "    add.s32 %r53, %r30, %r52;\n"
                   "    @{elect} st.shared.b32 [%r53], %r20;\n")  # fmt: skip
        self.check(lane.replace("-4", "4").replace("60", "28") + "    selp.b32 %r54, %r52, 0, %p6;\n"
                   "    add.s32 %r55, %r30, %r54;\n    @%p8 st.shared.b32 [%r55], %r20;\n",
                   hooks=("default",))  # fmt: skip

    def test_shared_atomics_and_stmatrix(self):
        self.check("    @{one} red.shared.add.u32 [global_smem+32], 1;\n")
        self.check(
            "    @{one} red.shared.add.u32 [global_smem+40], 1;\n", overlaps=False
        )
        # each address-giving lane writes a 16-byte row
        rows = ("    mov.u32 %r61, global_smem;\n    add.s32 %r62, %r61, AT;\n"
                "    stmatrix.sync.aligned.m8n8.x1.shared.b16 [%r62], {%r1};\n")  # fmt: skip
        self.check(rows.replace("AT", "24"), hooks=("default", "late"))
        self.check(rows.replace("AT", "16"), overlaps=False, hooks=("default", "late"))

    def test_tcgen05_alloc_slot(self):
        alloc = "    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [global_smem+AT], 32;\n"
        self.check(alloc.replace("AT", "32"), hooks=("default", "late"))
        self.check(alloc.replace("AT", "96"), overlaps=False, hooks=("default", "late"))

    def test_clc_response_and_mbarrier_objects(self):
        # a CLC response (16 bytes, asynchronous) and an mbarrier object (8 bytes) at 32
        clc = ("    @{elect} clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::"
               "complete_tx::bytes.b128 [global_smem+AT], [global_smem+64];\n")  # fmt: skip
        self.check(clc.replace("AT", "32"), asynchronous=True)
        self.check(clc.replace("AT", "128"), overlaps=False)
        # an mbarrier object changes at its events (some asynchronous, at indices that
        # may be loop-carried): its whole array must miss every capture a worker read
        init = "    @{one} mbarrier.init.shared::cta.b64 [global_smem+AT], 1;\n"
        for hook in ("early", "default", "late"):
            f = run(**{hook: init.replace("AT", "32")})
            self.assertIn(
                "lies in an mbarrier array (global_smem + 32, 1 entries)", refusal(f)
            )
        self.check(init.replace("AT", "40"), overlaps=False)

    def test_ring_buffer_index_bounds(self):
        # `x % k` as LLVM writes it for any x: x - ((x + b) & -k) and x - q k with q the
        # magic-number quotient (signed mul.hi, its sign bit added): within (-k, k)
        x = E.var("c", 32)
        b = E.bin("lshr", E.bin("ashr", x, E.const(31, 32)), E.const(30, 32))
        pow2 = E.bin("sub", x, E.bin("and", E.bin("add", x, b), E.const(-4, 32)))
        self.assertEqual(interp._capture_iv(pow2), (-3, 3))
        h = E.trunc(E.bin("lshr", E.bin("mul", E.ext(x, 64, True), E.const(715827883, 64)),
                          E.const(32, 64)), 32)  # fmt: skip
        q = E.bin("add", h, E.bin("lshr", h, E.const(31, 32)))
        magic = E.bin("sub", x, E.bin("mul", q, E.const(6, 32)))
        self.assertEqual(interp._capture_iv(magic), (-5, 5))
        self.assertIsNone(interp._capture_iv(E.bin("add", x, E.const(4, 32))))
        # no bounds where the operation may wrap at its width (a sum of two selects of
        # 0x7FFFFFF0 or 0, plus 64, is 32 if both are taken), nor from a multiplier
        # that is no mul.hi.s32 operand (2^52 / 6 rounded up, shifted right by 20)
        big, z = E.const(0x7FFFFFF0, 32), E.const(0, 32)
        p, r = (E.cmp("ne", E.var(v, 32), z) for v in ("p", "r"))
        two = E.bin("add", E.select(p, big, z), E.select(r, big, z))
        self.assertIsNone(interp._capture_iv(E.bin("add", two, E.const(64, 32))))
        h20 = E.bin("ashr", E.trunc(E.bin("lshr", E.bin("mul", E.ext(x, 64, True),
                    E.const(-(-(1 << 52) // 6), 64)), E.const(32, 64)), 32), E.const(20, 32))  # fmt: skip
        q20 = E.bin("add", h20, E.bin("lshr", h20, E.const(31, 32)))
        self.assertIsNone(
            interp._capture_iv(E.bin("sub", x, E.bin("mul", q20, E.const(6, 32))))
        )
        # a mask keeps only the bits x can have: (x in [0, 1]) & 126 is 0
        bit = E.bin("lshr", E.bin("urem", E.pid(0), E.const(4, 32)), E.const(1, 32))
        self.assertEqual(
            interp._capture_iv(E.bin("and", bit, E.const(126, 32))), (0, 0)
        )

    def test_zero_extension_of_a_negative_value(self):
        # x in [-1, 6] at 16 bits: its zero extension is 0..6 or 65535, its sign extension
        # -1..6, so a footprint 4 * zext(x) reaches 262140
        x = E.bin("sub", E.bin("and", E.var("n", 16), E.const(7, 16)), E.const(1, 16))
        self.assertEqual(interp._capture_iv(x), (-1, 6))
        z = E.ext(x, 32, False)
        self.assertEqual(interp._capture_iv(z), (0, 65535))
        self.assertEqual(
            interp._capture_iv(E.bin("mul", z, E.const(4, 32))), (0, 262140)
        )
        self.assertEqual(interp._capture_iv(E.ext(x, 32, True)), (-1, 6))
        self.assertEqual(
            interp._capture_iv(E.ext(E.bin("and", x, E.const(7, 16)), 32, False)),
            (0, 7),
        )

    def test_sign_extension_of_a_bit(self):
        # a 1-bit value is 0 or 1 and its sign extension 0 or -1, so a footprint
        # 4 * sext(p) reaches -4
        p = E.cmp("slt", E.var("n", 32), E.const(4, 32))
        s = E.ext(p, 32, True)
        self.assertEqual(interp._capture_iv(s), (-1, 0))
        self.assertEqual(interp._capture_iv(E.ext(p, 32, False)), (0, 1))
        self.assertEqual(interp._capture_iv(E.bin("mul", s, E.const(4, 32))), (-4, 0))

    def test_strength_reduced_offset_is_assumed(self):
        # an offset carried through a loop (no static bounds): the capture stays known
        # under a KIR assume (change 16) that the landing misses it, evaluated per input
        loop = ("    mov.b32 %r70, 0;\n    mov.b32 %r71, 0;\n$RING:\n"
                "    add.s32 %r72, %r30, %r71;\n"
                "    @{elect} cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                "[%r72], [%rd1], 16, [global_smem+64];\n"
                "    add.s32 %r71, %r71, 4096;\n    add.s32 %r70, %r70, 1;\n"
                "    setp.lt.s32 %p11, %r70, %r20;\n    @%p11 bra $RING;\n")  # fmt: skip
        f = run(late=loop)
        self.assertEqual(f.refusals, [])
        stmts = [s for r in f.roles for s in _statements(r["body"])]
        msgs = [s["msg"] for s in stmts if s["op"] == "assume"]
        self.assertIn(
            "a shared write misses the warp-specialize captures [32, 36)", msgs
        )
        self.assertFalse(
            any(s["op"] == "assert" for s in stmts)
        )  # never a certain error

    def test_st_async_from_another_cta(self):
        # the other CTA of the pair writes this CTA's capture: even after the writer's
        # second fork barrier (no barrier of this CTA orders it)
        st = ("    mov.u32 %r50, %cluster_ctarank;\n    xor.b32 %r51, %r50, 1;\n"
              "    add.s32 %r52, %r30, AT;\n    mapa.shared::cluster.u32 %r53, %r52, %r51;\n"
              "    add.s32 %r54, %r30, 64;\n    mapa.shared::cluster.u32 %r55, %r54, %r51;\n"
              "    @{elect} st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%r53], %r1, [%r55];\n")  # fmt: skip
        for hook in ("late", "after"):
            self.assertIn(
                OVERWRITE, refusal(run(2, **{hook: st.replace("AT", "32")})), hook
            )
            self.assertEqual(
                run(2, **{hook: st.replace("AT", "128")}).refusals, [], hook
            )

    def test_bulk_reduce_from_another_cta(self):
        # cp.reduce.async.bulk .shared::cluster combines bytes into the other CTA's shared
        # memory, as st.async: even after the writer's second fork barrier
        red = ("    mov.u32 %r50, %cluster_ctarank;\n    xor.b32 %r51, %r50, 1;\n"
               "    add.s32 %r52, %r30, AT;\n    mapa.shared::cluster.u32 %r53, %r52, %r51;\n"
               "    add.s32 %r54, %r30, 64;\n    mapa.shared::cluster.u32 %r55, %r54, %r51;\n"
               "    add.s32 %r56, %r30, 256;\n"
               "    @{elect} cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u32 [%r53], [%r56], 16, [%r55];\n")  # fmt: skip
        for hook in ("late", "after"):
            self.assertIn(
                OVERWRITE, refusal(run(2, **{hook: red.replace("AT", "24")})), hook
            )
            self.assertEqual(
                run(2, **{hook: red.replace("AT", "128")}).refusals, [], hook
            )

    def test_fork_barriers_in_both_arms(self):
        # the default runs its two fork barriers in both arms of an if, on n (a KIR `if`)
        # or on its warp (masked), and writes the capture in one arm before them: each
        # arm counts its fork barriers from the If, whichever arm is translated first
        forks = "    barrier.sync 1;\n    barrier.sync 1;\n"
        for cond in ("%r20, 3", "%r3, 0"):
            for write in ("st.shared.b32 [global_smem+32]", "st.u32 [%rd9+32]"):
                for arms in ((write, ""), ("", write)):
                    a, b = (f"    {w}, %r1;\n" if w else "" for w in arms)
                    text = BODY.replace("{default}" + forks, (
                        f"    setp.gt.s32 %p20, {cond};\n    @%p20 bra $THEN;\n{a}{forks}"
                        f"    bra.uni $JOIN;\n$THEN:\n{b}{forks}$JOIN:\n"))  # fmt: skip
                    self.assertIn(UNKNOWN, refusal(run(text=text)), (cond, write, arms))

    def test_writes_after_the_second_fork_barrier(self):
        # this CTA's own writes after partition1's second fork barrier follow every read
        self.assertEqual(run(after="    @{one} st.u32 [%rd9+32], %r1;\n").refusals, [])
        self.assertEqual(run(after=tma(0)).refusals, [])

    def test_op_table_writers_covered(self):
        # every op-table row (or prefix family) that may write shared memory is one of
        # CAPTURE_WRITERS, and reaches a capture hook through the dispatch execute() takes
        # for its class, but TMA tensor loads (tma.py reports their landings) and the
        # mbarrier objects (their arrays, check_capture_barriers)
        reads = re.compile(
            r"ld\.|cvta\.|mapa\.|(cp|cp\.reduce)\.async\.bulk(\.tensor\.\dd)?\.global\.shared"
            r"|cp\.reduce\.async\.bulk\.tensor\.\dd\.global\.shared|tensormap\.cp_fenceproxy"
            r"|(st|atom|red)\.(global|local|param)|red\.async\.global|mbarrier\.(try|test)_wait"
            r"|tensormap\.replace\.\w+\.global$|clusterlaunchcontrol\.query"
        )  # fmt: skip
        maybe = re.compile(r"shared|^stmatrix|^(st|atom|red)(\.|$)|^mbarrier\.|^tcgen05\."
                           r"(alloc|commit)|^clusterlaunchcontrol|^tensormap\.replace")  # fmt: skip
        elsewhere = re.compile(r"cp\.async\.bulk\.tensor\.\dd\.shared|tcgen05\.commit"
                               r"|mbarrier\.(init|arrive|expect_tx|complete_tx)")  # fmt: skip
        writers = [re.compile(p) for p in interp.CAPTURE_WRITERS]
        rows = [(k, cls) for k, (cls, _r) in ops.TABLE.items()]
        rows += [(k, cls) for cls, _r, keys in ops._FAMILIES for k in keys]
        rows = [(k, cls) for k, cls in rows
                if cls != "refuse" and maybe.search(k) and not reads.match(k)]  # fmt: skip
        self.assertEqual(
            [k for k, _c in rows if not any(w.match(k) for w in writers)], []
        )
        self.assertEqual(
            [k for k, c in rows if not elsewhere.match(k) and not reaches_a_hook(k, c)],
            [],
        )


if __name__ == "__main__":
    unittest.main()
