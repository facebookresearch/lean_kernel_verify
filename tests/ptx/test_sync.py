# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the PTX core's synchronization rules (frontend/ptx/sync.py and the mbarrier,
named-barrier and loop rules M2 kernels need) through translate() on hand-written PTX.

    python3 -I tests/ptx/test_sync.py      (or the buck target :test_sync)

tcgen05: alloc/dealloc/relinquish/ld/st/wait/cp/shift/mma emit nothing in P0, ld's
registers are Data, a .sync.aligned form under a lane condition is refused, commit is a
commit_arrive per issuing thread, a multicast commit and cta_group::2 need a cluster
(test_cluster.py). CLC: the
try_cancel/query/get_first_ctaid sequence of Triton's CLC (evaluated over a schedule),
the role prologue, refusals of other response uses (a response read elsewhere than the
try_cancel's address or before a wait on its mbarrier, an unpredicated get_first_ctaid)
and of CuTe CLC. Named barriers,
fences and griddepcontrol emit nothing; the cluster-scope forms are gaps in the Triton
family (test_cluster.py: the CuTe family); relaxed arrive
and inval are refused; parts of a CuTe role that emit nothing are merged. Pipelines: a
wait in a loop with stage and phase slots, the LSR form of `fulls[blk % 2]`, a barrier
index computed after a loop from registers the loop defines, an smem descriptor (bit
operations on a shared address) carried in a loop.
"""

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

from .fixtures.kernels import evaluate, events, kir_errors, run  # noqa: E402

# thread id, n, the shared array, warp, thread 0, warp 0, the pointer x.
PRE = """
    mov.u32 %r1, %tid.x;
    ld.param.b32 %r2, [k_param_1];
    mov.u32 %r10, smem;
    shr.u32 %r3, %r1, 5;
    setp.eq.s32 %p12, %r1, 0;
    setp.eq.s32 %p11, %r3, 0;
    ld.param.u64 %rd1, [k_param_0];
"""


def sync(body, **kw):
    return run(PRE + body + "\n    ret;\n", **kw)


def refusal(f) -> str:
    return f.refusals[0]["msg"] if f.refusals else ""


TCGEN05 = """
    @%p12 mbarrier.init.shared::cta.b64 [%r10], 1;
    @%p11 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r10+16], 128;
    @%p11 tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
    ld.shared.b32 %r20, [%r10+16];
    tcgen05.ld.sync.aligned.32x32b.x2.b32 {%r21, %r22}, [%r20];
    tcgen05.wait::ld.sync.aligned;
    tcgen05.st.sync.aligned.32x32b.x1.b32 [%r20], {%r21};
    tcgen05.wait::st.sync.aligned;
    tcgen05.fence::before_thread_sync;
    @%p12 tcgen05.mma.cta_group::1.kind::f16 [%r20], %rd5, %rd6, %r24, %p14;
    @%p12 tcgen05.cp.cta_group::1.128x256b [%r20], %rd5;
    @%p12 tcgen05.shift.cta_group::1.down [%r20];
    @%p12 tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r10];
    @%p11 tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r20, 128;
"""


class Tcgen05(unittest.TestCase):
    def test_p0_no_ops_and_commit(self):
        for family in ("ttgir", "cute"):
            f = sync(TCGEN05, family=family)
            self.assertEqual(f.refusals, [])
            evs = [(e["ev"], e["bar"]) for e in events(f)]
            self.assertEqual(evs, [("bar_init", "mbar_0"), ("commit_arrive", "mbar_0")])
            self.assertEqual(kir_errors(f), [])

    def test_commit_per_issuing_thread(self):
        f = sync("""
    @%p12 mbarrier.init.shared::cta.b64 [%r10], 4;
    elect.sync %r23|%p13, -1;
    @%p13 tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r10];
""")
        loop = [s for s in f.roles[0]["body"] if s["op"] == "for"]
        self.assertEqual(loop[0]["ub"]["v"], 4)  # one per warp: the elected lanes
        self.assertEqual(loop[0]["body"][0]["ev"], "commit_arrive")
        self.assertEqual(kir_errors(f), [])

    def test_ld_registers_are_data(self):
        f = sync("""
    @%p12 mbarrier.init.shared::cta.b64 [%r10], 1;
    tcgen05.ld.sync.aligned.32x32b.x1.b32 {%r21}, [%r20];
    add.s32 %r25, %r10, %r21;
    @%p12 mbarrier.arrive.shared::cta.b64 _, [%r25];
""")
        self.assertIn("tcgen05.ld", refusal(f))
        self.assertIn("slice soundness", refusal(f))

    def test_refusals(self):
        f = sync("""
    setp.lt.u32 %p16, %r1, 16;
    @%p16 tcgen05.wait::ld.sync.aligned;
""")
        self.assertIn("differs between the lanes of a warp", refusal(f))
        f = sync("""
    @%p12 mbarrier.init.shared::cta.b64 [%r10], 1;
    @%p12 tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%r10], %rs1;
""")
        self.assertIn("multicast tcgen05.commit outside a cluster", refusal(f))
        f = sync(
            "    @%p12 tcgen05.mma.cta_group::2.kind::f16 [%r20], %rd5, %rd6, %r24, %p14;"
        )
        self.assertIn("cta_group::2", refusal(f))


# Triton's CLC loop: try_cancel by thread 0 (of the CTA ranked 0 in its cluster), wait,
# query, and the tile it returns stored by thread 0 until the schedule says -1.
CLC = """
    @%p12 mbarrier.init.shared::cta.b64 [%r10], 1;
    mov.b32 %r30, 0;
LOOP:
    @%p12 mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r10], 16;
{
    .reg .u32 first;
    .reg .pred pf;
    .reg .pred pi;
    mov.u32 first, %cluster_ctaid.x;
    setp.u32.eq pf, first, 0x0;
    and.pred pi, %p12, pf;
    @pi clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%r10+16], [%r10];
}
{
    .reg .pred P1;
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 P1, [%r10], %r30;
    @!P1 bra.uni WAIT;
}
    xor.b32 %r30, %r30, 1;
    mov.u32 %r31, 0;
    mov.u32 %r32, 0;
{
    .reg .b128 clc_result;
    .reg .pred p1;
    mov.s32 %r31, -1;
    mov.s32 %r32, -1;
    ld.shared.b128 clc_result, [%r10+16];
    clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;
    @p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%r31, %r32, %r33, _}, clc_result;
}
    setp.ne.s32 %p17, %r31, -1;
    and.pred %p18, %p17, %p12;
    add.s32 %r34, %r31, %r32;
    mul.wide.s32 %rd2, %r34, 4;
    add.s64 %rd3, %rd1, %rd2;
    @%p18 st.global.b32 [%rd3], %r1;
    @%p17 bra LOOP;
"""


class Clc(unittest.TestCase):
    def test_schedule_model(self):
        f = sync(CLC)
        self.assertEqual(f.refusals, [])
        self.assertTrue(f.clc)
        body = f.roles[0]["body"]
        self.assertEqual(
            body[0],
            {"op": "let", "var": "__clc_k", "expr": {"e": "const", "w": 32, "v": 0}},
        )
        self.assertEqual(
            (body[1]["op"], body[1]["cond"]["a"]["tensor"]), ("if", "__clc_launched")
        )
        (load,) = events(f, "tma_load")
        self.assertEqual(
            (load["desc"], load["coords"]),
            ("__clc_response", [{"e": "const", "w": 32, "v": 0}]),
        )
        self.assertEqual(kir_errors(f), [])
        sched = {(0, 0): 5, (0, 1): 7, (0, 2): -1}
        trace = evaluate(
            f, {"n": 1}, metadata={"__clc_launched": [1], "__clc_schedule": sched}
        )
        self.assertEqual(
            [t for t in trace if t[0] == "gstore"],
            [("gstore", "x", [5]), ("gstore", "x", [7])],
        )
        self.assertEqual(
            [t[3] for t in trace if t[0] == "wait"], [0, 1, 0]
        )  # phase flips
        self.assertEqual(len([t for t in trace if t[0] == "tma_load"]), 3)
        self.assertEqual(
            evaluate(
                f, {"n": 1}, metadata={"__clc_launched": [0], "__clc_schedule": sched}
            ),
            [],
        )

    def test_refusals(self):
        self.assertIn(
            "CLC outside the Triton family", refusal(sync(CLC, family="cute"))
        )
        bad = CLC.replace(
            "ld.shared.b128 clc_result", "mov.b64 %rd9, 0;\n    ld.shared.b64 %rd9"
        )
        self.assertIn("not a CLC response", refusal(sync(bad)))
        bad = CLC.replace(
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;",
            "",
        )
        bad = bad.replace("@p1 clusterlaunchcontrol", "clusterlaunchcontrol")
        self.assertIn("no is_canceled queried", refusal(sync(bad)))
        bad = CLC.replace(
            "clusterlaunchcontrol.query_cancel.is_canceled",
            "@%p12 clusterlaunchcontrol.query_cancel.is_canceled",
        )
        self.assertIn("CLC query under a condition", refusal(sync(bad)))

    def test_response_read_where_and_when_try_cancel_writes_it(self):
        """The queried response must be the try_cancel's: read from its address, after a
        wait on its mbarrier; get_first_ctaid must be predicated by is_canceled's p."""
        bad = CLC.replace(
            "[%r10+16];\n    clusterlaunchcontrol.query",
            "[%r10+48];\n    clusterlaunchcontrol.query",
        )
        self.assertIn("from an address no try_cancel writes", refusal(sync(bad)))
        wait = CLC[CLC.index("{\n    .reg .pred P1;") : CLC.index("    xor.b32 %r30")]
        read = CLC[
            CLC.index("    mov.u32 %r31, 0;") : CLC.index("    setp.ne.s32 %p17")
        ]
        flip = "    xor.b32 %r30, %r30, 1;\n"
        moved = CLC.replace(wait + flip + read, read + wait + flip)
        self.assertNotEqual(moved, CLC)
        self.assertIn(
            "before a wait on its try_cancel's mbarrier", refusal(sync(moved))
        )
        bad = CLC.replace(
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid",
            "clusterlaunchcontrol.query_cancel.get_first_ctaid",
        )
        self.assertIn("not predicated by the is_canceled result", refusal(sync(bad)))


class Barriers(unittest.TestCase):
    def test_named_barriers_fences_griddepcontrol_emit_nothing(self):
        body = """
    bar.sync 1, 64;
    barrier.sync 2, 128;
    bar.arrive 3, 64;
    barrier.arrive 4, 64;
    barrier.sync.aligned 0;
    bar.warp.sync -1;
    fence.proxy.async.shared::cta;
    fence.mbarrier_init.release.cluster;
    griddepcontrol.wait;
    griddepcontrol.launch_dependents;
"""
        for family in ("ttgir", "cute"):
            f = sync(body, family=family)
            self.assertEqual((f.refusals, events(f)), ([], []))
        f = sync("    setp.lt.u32 %p16, %r1, 16;\n    @%p16 bar.sync 1, 64;")
        self.assertIn("differs between the lanes of a warp", refusal(f))
        f = sync("    elect.sync %r23|%p13, -1;\n    @%p13 barrier.sync.aligned 1, 64;")
        self.assertIn("differs between the lanes of a warp", refusal(f))

    def test_gaps_and_refused_forms(self):
        init = "    @%p12 mbarrier.init.shared::cta.b64 [%r10], 1;\n"
        for stmt, why in [
            ("barrier.cluster.arrive.aligned;", "Triton family is not in v1"),
            (
                "@%p12 mbarrier.arrive.shared::cluster.b64 _, [%r10];",
                "Triton family is not in v1",
            ),
            (
                "@%p12 mbarrier.complete_tx.shared::cta.b64 [%r10], 16;",
                "Triton family is not in v1",
            ),
            (
                "@%p12 mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%r10];",
                "relaxed arrive",
            ),
            (
                "@%p12 mbarrier.inval.shared::cta.b64 [%r10];",
                "relaxed arrive and invalidation",
            ),
        ]:
            f = sync(init + "    " + stmt)
            self.assertIn(why, refusal(f), stmt)
            self.assertEqual(f.refusals[0]["kind"], "gap")

    def test_idle_warps_are_one_role(self):
        # CuTe policy: warps 2 and 3 only park, tested in two branches (as LLVM splits
        # `warp in (2, 3)`); their parts emit nothing and are one role, as in the CuTe
        # frontend's.
        f = sync(
            """
    @%p12 mbarrier.init.shared::cta.b64 [%r10], 64;
    setp.eq.s32 %p30, %r3, 2;
    @%p30 bra END;
    setp.eq.s32 %p31, %r3, 3;
    @%p31 bra PARK;
    mbarrier.arrive.shared::cta.b64 _, [%r10];
    bra END;
PARK:
    setmaxnreg.dec.sync.aligned.u32 24;
END:
""",
            family="cute",
        )
        roles = [(r["name"], r["threads_per_issue"]) for r in f.roles]
        self.assertEqual(roles, [("warps0_1", 64), ("warps2_3", 1)])
        self.assertEqual(kir_errors(f), [])

    def test_wait_in_a_loop_with_stage_and_phase(self):
        # A 2-stage pipeline: index + 1 wraps to 0 by a select, the phase flips with it.
        body = """
    @%p12 mbarrier.init.shared::cta.b64 [%r10], 1;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+8], 1;
    mov.b32 %r40, 0;
    mov.b32 %r41, 0;
    mov.b32 %r42, 0;
LOOP:
    shl.b32 %r43, %r40, 3;
    add.s32 %r44, %r10, %r43;
{
    .reg .pred P1;
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 P1, [%r44], %r41;
    @!P1 bra.uni WAIT;
}
    add.s32 %r40, %r40, 1;
    setp.eq.s32 %p20, %r40, 2;
    selp.b32 %r40, 0, %r40, %p20;
    xor.b32 %r45, %r41, 1;
    selp.b32 %r41, %r45, %r41, %p20;
    add.s32 %r42, %r42, 1;
    setp.lt.s32 %p21, %r42, %r2;
    @%p21 bra LOOP;
"""
        for family in ("ttgir", "cute"):
            f = sync(body, family=family)
            self.assertEqual(f.refusals, [])
            self.assertEqual(
                [(b["name"], b["num"]["v"]) for b in f.barriers], [("mbar_0", 2)]
            )
            waits = [t[2:] for t in evaluate(f, {"n": 5}) if t[0] == "wait"]
            self.assertEqual(waits, [(0, 0), (1, 0), (0, 1), (1, 1), (0, 0)])
            self.assertEqual(kir_errors(f), [])

    def test_cute_split_by_stage_ranges(self):
        # Two pipelines initialized in one block at one .loc (DSL 4.4.2): a 2-stage one
        # (full 0-8, empty 16-24; the stage wraps by a select) and a 1-stage one (full
        # 32 with its transaction bytes, empty 40). The stage slot's range [0, 1] splits
        # the run as the CuTe frontend does; 32 and 40, named by constants and used
        # differently, are two barriers. 48-56 are used through an address slot LLVM
        # rematerializes (invariant: a point range keeps that run whole, as before).
        # 64-72: a 2-stage pipeline whose stage index LLVM folded in unrolled copies
        # (constants, the same role and events on both): one array, as the DSL code's
        # dynamic index makes it for the CuTe frontend.
        body = """
    @%p12 mbarrier.init.shared::cta.b64 [%r10], 1;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+8], 1;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+16], 1;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+24], 1;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+32], 1;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+40], 1;
    .loc 1 2 0
    @%p12 mbarrier.init.shared::cta.b64 [%r10+48], 1;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+56], 1;
    .loc 1 4 0
    @%p12 mbarrier.init.shared::cta.b64 [%r10+64], 1;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+72], 1;
    .loc 1 3 0
    mov.b32 %r40, 0;
    mov.b32 %r41, 0;
    mov.b32 %r42, 0;
    mov.u32 %r50, smem;
LOOP:
    shl.b32 %r43, %r40, 3;
    add.s32 %r44, %r10, %r43;
    @%p12 mbarrier.arrive.shared::cta.b64 _, [%r44];
    @%p12 mbarrier.arrive.shared::cta.b64 _, [%r44+16];
    @%p12 mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r10+32], 16;
    @%p12 mbarrier.arrive.shared::cta.b64 _, [%r10+40];
    @%p12 mbarrier.arrive.shared::cta.b64 _, [%r50+48];
    @%p12 mbarrier.arrive.shared::cta.b64 _, [%r50+56];
    @%p12 mbarrier.arrive.shared::cta.b64 _, [%r10+64];
    @%p12 mbarrier.arrive.shared::cta.b64 _, [%r10+72];
    add.s32 %r40, %r40, 1;
    setp.eq.s32 %p20, %r40, 2;
    selp.b32 %r40, 0, %r40, %p20;
    mov.u32 %r50, smem;
    add.s32 %r42, %r42, 1;
    setp.lt.s32 %p21, %r42, %r2;
    @%p21 bra LOOP;
"""
        f = sync(body, family="cute")
        self.assertEqual(f.refusals, [])
        arrays = [(b["name"], b["num"]["v"]) for b in f.barriers]
        want = [
            ("mbar_0", 2),
            ("mbar_16", 2),
            ("mbar_32", 1),
            ("mbar_40", 1),
            ("mbar_48", 2),
            ("mbar_64", 2),
        ]
        self.assertEqual(arrays, want)
        self.assertEqual(kir_errors(f), [])
        trace = [t[1:3] for t in evaluate(f, {"n": 2}) if t[0] == "arrive"]
        self.assertEqual(
            trace[:2] + trace[6:10],
            [("mbar_0", 0), ("mbar_16", 0), ("mbar_64", 0), ("mbar_64", 1), ("mbar_0", 1),
             ("mbar_16", 1)],
        )  # fmt: skip
        ttgir = [(b["name"], b["num"]["v"]) for b in sync(body).barriers]
        self.assertEqual(
            ttgir, [("mbar_0", 6), ("mbar_48", 2), ("mbar_64", 2)]
        )  # no split (TLX)

    def test_array_of_a_scaled_srem_index(self):
        # `empties[(n - 1) % 2]` as LLVM writes it: 16 + ((n-1) - ((n-1 + sign) & -2)) << 3.
        # The -1 inside the scaled index is the index's: the array is the one at 16.
        f = sync("""
    .loc 1 2 0
    @%p12 mbarrier.init.shared::cta.b64 [%r10], 1;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+8], 1;
    .loc 1 3 0
    @%p12 mbarrier.init.shared::cta.b64 [%r10+16], 1;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+24], 1;
    add.s32 %r80, %r2, -1;
    shr.u32 %r81, %r80, 31;
    add.s32 %r82, %r80, %r81;
    and.b32 %r83, %r82, -2;
    sub.s32 %r84, %r80, %r83;
    shl.b32 %r85, %r84, 3;
    add.s32 %r86, %r10, %r85;
    @%p12 mbarrier.arrive.shared::cta.b64 _, [%r86+16];
""")
        self.assertEqual(f.refusals, [])
        for n, idx in ((1, 0), (2, 1), (5, 0)):
            arrive = [t[1:3] for t in evaluate(f, {"n": n}) if t[0] == "arrive"]
            self.assertEqual(arrive, [("mbar_16", idx)])

    def test_lsr_slot_with_a_folded_stage(self):
        # TLX `fulls[blk % 2]` after LSR: a slot smem + 64 + 8*blk (+8 per trip) minus
        # 16 * (blk sdiv 2); the slot keeps smem + 64 outside, trailing zeros prove idx.
        # The same with the slot an integer 8*blk added to smem after (the trailing zeros
        # of an integer slot too).
        loop = """
    @%p12 mbarrier.init.shared::cta.b64 [%r10+64], 1;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+72], 1;
    mov.b32 %r51, 0;
    {init}
LOOP:
    shr.u32 %r52, %r51, 31;
    add.s32 %r53, %r51, %r52;
    shr.s32 %r54, %r53, 1;
    shl.b32 %r55, %r54, 4;
    sub.s32 %r56, %r50, %r55;
    {use}
    add.s32 %r50, %r50, 8;
    add.s32 %r51, %r51, 1;
    setp.lt.s32 %p21, %r51, %r2;
    @%p21 bra LOOP;
"""
        for init, use in [
            ("add.s32 %r50, %r10, 64;", "@%p12 mbarrier.arrive.shared::cta.b64 _, [%r56];"),
            ("shl.b32 %r50, %r51, 3;", "add.s32 %r57, %r10, %r56;\n"
             "    @%p12 mbarrier.arrive.shared::cta.b64 _, [%r57+64];"),
        ]:  # fmt: skip
            f = sync(loop.format(init=init, use=use))
            self.assertEqual(f.refusals, [])
            arrives = [t[2] for t in evaluate(f, {"n": 4}) if t[0] == "arrive"]
            self.assertEqual(arrives, [0, 1, 0, 1])
            self.assertEqual(kir_errors(f), [])

    def test_index_computed_after_a_loop(self):
        # %r61 is written before it is read in every trip and read after the loop only
        # (undefined before it), %r64 too, an address (its slot keeps its base); a float
        # register like them stays data after the loop.
        f = sync("""
    @%p12 mbarrier.init.shared::cta.b64 [%r10], 1;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+8], 1;
    mov.b32 %r60, 0;
LOOP:
    add.s32 %r61, %r60, 1;
    and.b32 %r61, %r61, 1;
    mov.u32 %r64, smem;
    mov.f32 %f1, 0f3F800000;
    add.s32 %r60, %r60, 1;
    setp.lt.s32 %p22, %r60, %r2;
    @%p22 bra LOOP;
    shl.b32 %r62, %r61, 3;
    add.s32 %r63, %r10, %r62;
    @%p12 mbarrier.arrive.shared::cta.b64 _, [%r63];
    @%p12 mbarrier.arrive.shared::cta.b64 _, [%r64+8];
    @%p12 st.global.f32 [%rd1], %f1;
""")
        self.assertEqual(f.refusals, [])
        for n, idx in ((3, 1), (4, 0)):
            arrives = [t[2] for t in evaluate(f, {"n": n}) if t[0] == "arrive"]
            self.assertEqual(arrives, [idx, 1])

    def test_smem_descriptor_is_data(self):
        # tcgen05 smem descriptors: bit operations on a shared address depend on where
        # the buffer is; carried in a loop they are data, refused only at a root.
        body = """
    shr.u32 %r70, %r10, 4;
    and.b32 %r71, %r70, 16383;
    mov.b32 %r72, 0;
LOOP:
    add.s32 %r71, %r71, 64;
    add.s32 %r72, %r72, 1;
    setp.lt.s32 %p23, %r72, %r2;
    @%p23 bra LOOP;
"""
        self.assertEqual(sync(body).refusals, [])
        f = sync(body + "    @%p12 mbarrier.arrive.shared::cta.b64 _, [%r71];")
        self.assertIn("%r71 after the loop", refusal(f))
        f = sync(body + "    @%p12 mbarrier.arrive.shared::cta.b64 _, [%r70];")
        self.assertIn("an address transformed", refusal(f))


if __name__ == "__main__":
    unittest.main()
