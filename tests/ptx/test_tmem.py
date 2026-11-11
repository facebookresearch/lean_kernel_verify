# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the PTX frontend's P1 pass (frontend/ptx/tmem.py: P1-1 events on PTX)
through translate() with policy p1 on hand-written PTX.

    python3 -I tests/ptx/test_tmem.py      (or the buck target :test_tmem)

Events: tcgen05.ld/st per issuing warp (a `for` over the warps when all issue), wait::ld
and ::st, cp and shift and mma (dense with a TMEM A, block-scaled with its scale factors,
an instruction descriptor known only in its shape bits, cta_group::2 with the pair's
cta_mask) by one thread, named barriers; their order next to the mbarrier events; the
allocation and the KIR `tmem`; the role split by an mbarrier event part of a role
issues. Refusals R1, R2, R3, R5, R6, R7 and R10, the Triton family, and the P0 pass
unchanged.
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

from .fixtures.kernels import events, kir_errors, run, statements  # noqa: E402

# thread id, n, the shared array, warp, thread 0, warp 0, the pointer x, an elected lane,
# and the 512-column allocation by warp 0 (its address at smem+16).
PRE = """
    mov.u32 %r1, %tid.x;
    ld.param.b32 %r2, [k_param_1];
    mov.u32 %r10, smem;
    shr.u32 %r3, %r1, 5;
    setp.eq.s32 %p12, %r1, 0;
    setp.eq.s32 %p11, %r3, 0;
    ld.param.u64 %rd1, [k_param_0];
    @%p12 mbarrier.init.shared::cta.b64 [%r10], 1;
"""
ALLOC = """
    mov.b32 %r9, 512;
    @%p11 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r10+16], %r9;
"""


def p1(body, alloc=ALLOC, threads=32, **kw):
    return run(PRE + alloc + body + "\n    ret;\n", family="cute", p1=True,
               threads=threads, **kw)  # fmt: skip


def refusal(f) -> str:
    return f.refusals[0]["msg"] if f.refusals else ""


def value(e: dict, env: dict) -> int:
    """A KIR Expr's value (the few nodes TMEM addresses use), unsigned at its width."""
    k, w = e["e"], e["w"]
    if k == "const":
        return e["v"] % (1 << w)
    if k == "var":
        return env[e["name"]]
    a, b = value(e["a"], env), value(e["b"], env)
    r = {"add": a + b, "sub": a - b, "mul": a * b, "and": a & b, "or": a | b,
         "shl": a << b, "lshr": a >> b}[e["o"]]  # fmt: skip
    return r % (1 << w)


def rects(ev: dict, env: dict | None = None) -> list:
    """(use, write, lane, nlanes, col, ncols) of a tmem event, values under env."""
    return [(r["use"], r["write"], value(r["lane"], env or {}), r["nlanes"],
             value(r["col"], env or {}), r["ncols"]) for r in ev["rects"]]  # fmt: skip


class LdStWait(unittest.TestCase):
    # every warp's lane quarter, columns 64..: ld x2, wait, st x1, wait
    BODY = """
    bar.sync 0;
    and.b32 %r20, %r3, 3;
    shl.b32 %r21, %r20, 21;
    or.b32 %r22, %r21, 64;
    tcgen05.ld.sync.aligned.32x32b.x2.b32 {%r30, %r31}, [%r22];
    tcgen05.wait::ld.sync.aligned;
    tcgen05.st.sync.aligned.32x32b.x1.b32 [%r22+8], {%r30};
    tcgen05.wait::st.sync.aligned;
    tcgen05.fence::before_thread_sync;
"""

    def test_per_warp_loop(self):
        f = p1(self.BODY, threads=128)
        self.assertEqual(f.refusals, [])
        self.assertEqual(f.tmem, {"cols": 512, "base": "zero"})
        self.assertEqual(
            [(e["ev"], e.get("inst", e.get("ld"))) for e in events(f) if e["ev"] != "bar_init"],
            [("named_bar", None), ("tmem", "ld"), ("tmem_wait", True), ("tmem", "st"),
             ("tmem_wait", False)],
        )  # fmt: skip
        bar = events(f, "named_bar")[0]
        self.assertEqual(
            (bar["id"]["v"], bar["count"]["v"], bar["sync"]), (0, 128, True)
        )
        loops = [s for s in statements(f.roles[0]["body"]) if s["op"] == "for"]
        self.assertEqual([lp["ub"]["v"] for lp in loops], [4, 4])  # one per warp
        for lp, (write, col, n) in zip(loops, ((False, 64, 2), (True, 72, 1))):
            ev = lp["body"][-1]
            for w in range(4):
                env = {lp["var"]: w}
                env.update({s["var"]: value(s["expr"], env) for s in lp["body"][:-1]})
                self.assertEqual(value(ev["warp"], env), w)
                self.assertEqual(rects(ev, env), [("data", write, 32 * w, 32, col, n)])
        self.assertEqual(kir_errors(f), [])

    def test_one_warp_and_p0(self):
        f = p1(self.BODY)
        ev = events(f, "tmem")
        self.assertEqual([e["warp"]["v"] for e in ev], [0, 0])
        self.assertEqual(rects(ev[0]), [("data", False, 0, 32, 64, 2)])
        self.assertEqual(kir_errors(f), [])
        p0 = run(PRE + ALLOC + self.BODY + "\n    ret;\n", family="cute", threads=32)
        self.assertEqual(p0.refusals, [])
        self.assertIsNone(p0.tmem)
        self.assertEqual([e["ev"] for e in events(p0)], ["bar_init"])

    def test_refusals(self):
        f = p1("    tcgen05.ld.sync.aligned.16x64b.x2.b32 {%r30, %r31}, [%r9];")
        self.assertIn("R5:", refusal(f))
        f = p1("    tcgen05.ld.sync.aligned.32x32b.x1.b32 {%r30}, [%r1];")
        self.assertIn("R6: tcgen05.ld: an operand varies across the lanes", refusal(f))
        f = p1("""
    setp.lt.u32 %p13, %r1, 16;
    @%p13 tcgen05.ld.sync.aligned.32x32b.x1.b32 {%r30}, [%r9];""")
        self.assertIn("R6: tcgen05.ld with only part of a warp", refusal(f))
        f = p1("""
    ld.global.u32 %r20, [%rd1];
    setp.lt.u32 %p13, %r1, %r20;
    @%p13 tcgen05.ld.sync.aligned.32x32b.x1.b32 {%r30}, [%r9];""")
        self.assertIn(
            "R6: tcgen05.ld under a lane condition that depends on runtime", refusal(f)
        )
        f = p1("    @%p11 tcgen05.wait::ld.sync.aligned;", threads=64)
        self.assertIn(
            "R7: tcgen05.wait::ld reached by only part of the role", refusal(f)
        )
        f = p1("""
    ld.shared.u32 %r20, [%r10+16];
    tcgen05.ld.sync.aligned.32x32b.x1.b32 {%r30}, [%r20];""")
        self.assertIn("R1: tcgen05.ld: data value from ld.shared", refusal(f))
        f = p1("    tcgen05.ld.sync.aligned.32x32b.x1.b32 {%r30}, [%r10];")
        self.assertIn(
            "R1: tcgen05.ld: a memory address used as a TMEM address", refusal(f)
        )


# One warp's MMA pipeline: two scale-factor copies, a block-scaled MMA (idesc known in
# its shape bits only), a dense MMA with A in TMEM, a shift, and a commit, all by
# elected lanes. Block-scaled idesc: M >> 7 = 1 at bits 27-28, N >> 3 = 32 (N 256) at
# 17-22, bit 13 runtime; dense 0x8200490: M >> 4 = 8 at 24-28, N >> 3 = 16 (N 128).
MMA = """
    setp.eq.s32 %p15, %r2, 1;
    elect.sync %r23|%p13, -1;
    mov.b32 %r40, 384;
    @%p13 tcgen05.cp.cta_group::1.32x128b.warpx4 [%r40], %rd5;
    mov.b32 %r41, 392;
    @%p13 tcgen05.cp.cta_group::1.128x256b [%r41], %rd5;
    tcgen05.wait::st.sync.aligned;
    selp.b32 %r44, 138420224, 138412032, %p15;
    mov.b32 %r45, 1073742208;
    mov.b32 %r42, 0;
    @%p13 tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [%r42], %rd5, %rd6, %r44, [%r45], [%r41], %p14;
    mov.b32 %r46, 0x8200490;
    mov.b32 %r43, 256;
    @%p13 tcgen05.mma.cta_group::1.kind::f16 [%r43], [%r42+8], %rd6, %r46, {%r9, %r9, %r9, %r9}, 1;
    @%p13 tcgen05.shift.cta_group::1.down [%r40];
    elect.sync %r24|%p16, -1;
    @%p16 tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r10];
"""


class CpMma(unittest.TestCase):
    def test_pipeline(self):
        f = p1(MMA)
        self.assertEqual(f.refusals, [])
        evs = [
            e for e in events(f) if e["ev"] in ("tmem", "tmem_wait", "commit_arrive")
        ]
        self.assertEqual([(e["ev"], e.get("inst")) for e in evs], [
            ("tmem", "cp"), ("tmem", "cp"), ("tmem_wait", None), ("tmem", "mma"),
            ("tmem", "mma"), ("tmem", "cp"), ("commit_arrive", None)])  # fmt: skip
        self.assertEqual(rects(evs[0]), [("data", True, 0, 128, 384, 4)])
        self.assertEqual(rects(evs[1]), [("data", True, 0, 128, 392, 8)])
        # SFA's byte selector (bits 30-31 of its address) dropped; SFB 8 columns (N 256)
        self.assertEqual(rects(evs[3]), [("d", True, 0, 128, 0, 256), ("sf", False, 0, 128, 384, 4),
                                         ("sf", False, 0, 128, 392, 8)])  # fmt: skip
        self.assertEqual(
            rects(evs[4]), [("d", True, 0, 128, 256, 128), ("a", False, 0, 128, 8, 8)]
        )
        self.assertEqual(rects(evs[5]), [("data", True, 0, 128, 384, 8)])  # the shift
        self.assertTrue(
            all(e["warp"]["v"] == 0 and "cta_mask" not in e for e in evs[:2])
        )
        self.assertEqual(kir_errors(f), [])

    def test_cta_pair(self):
        f = p1(
            """
    elect.sync %r23|%p13, -1;
    mov.b32 %r46, 0x8200490;
    mov.b32 %r43, 256;
    @%p13 tcgen05.mma.cta_group::2.kind::f16 [%r43], %rd5, %rd6, %r46, 1;
    mov.b32 %r40, 384;
    @%p13 tcgen05.cp.cta_group::2.32x128b.warpx4 [%r40], %rd5;""",
            cluster=(2, 1, 1),
        )
        self.assertEqual(f.refusals, [])
        mma, cp = events(f, "tmem")
        self.assertEqual(rects(mma), [("d", True, 0, 128, 256, 64)])  # M 128: N / 2
        for e in (mma, cp):  # 3 << (rank & ~1), rank = pid.x % 2
            self.assertEqual(e["cta_mask"]["o"], "shl")
            self.assertEqual(e["cta_mask"]["a"]["v"], 3)
        self.assertEqual(kir_errors(f), [])
        f = p1("    @%p12 tcgen05.cp.cta_group::2.32x128b.warpx4 [%r9], %rd5;")
        self.assertIn("R5: cp cta_group::2 without a cluster of CTA pairs", refusal(f))

    def test_refusals(self):
        bad = {
            "    @%p12 tcgen05.mma.cta_group::1.kind::f16 [%r9], %rd5, %rd6, %r2, 1;":
                "R5: tcgen05.mma instruction descriptor whose shape bits are not known",
            "    mov.b32 %r46, 0x4200490;\n    @%p12 tcgen05.mma.cta_group::1.kind::f16 [%r9], %rd5, %rd6, %r46, 1;":
                "R5: tcgen05.mma.cta_group::1.kind::f16 64x128",
            "    mov.b32 %r46, 0x8420000;\n    @%p12 tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%r9], %rd5, %rd6, %r46, [%r9], [%r9], 1;":
                "R5: tcgen05.mma.cta_group::1.kind::mxf4 128x264",
            "    @%p12 tcgen05.cp.cta_group::1.4x256b [%r9], %rd5;": "R5: tcgen05.cp.cta_group::1.4x256b",
            "    setp.lt.u32 %p13, %r1, 2;\n    @%p13 tcgen05.cp.cta_group::1.128x128b [%r9], %rd5;":
                "R6: tcgen05.cp issued by 2 threads",
            "    @%p12 tcgen05.cp.cta_group::1.128x128b [%r1], %rd5;":
                "R6: tcgen05.cp: an operand varies across the lanes",
        }  # fmt: skip
        for body, want in bad.items():
            with self.subTest(body=body):
                self.assertIn(want, refusal(p1(body)))
        f = p1("    @%p12 tcgen05.cp.cta_group::1.128x128b [%r9], %rd5;", threads=64)
        self.assertIn("R6: tcgen05.cp in a role of 2 warps", refusal(f))

    def test_commit_issuers(self):
        # an elected lane's MMA and thread 0's commit: P1-1 needs the same thread (R10)
        f = p1("""
    elect.sync %r23|%p13, -1;
    mov.b32 %r46, 0x8200490;
    @%p13 tcgen05.mma.cta_group::1.kind::f16 [%r9], %rd5, %rd6, %r46, 1;
    @%p12 tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r10];""")
        self.assertIn("R10:", refusal(f))
        f = p1("""
    mov.b32 %r46, 0x8200490;
    @%p12 tcgen05.mma.cta_group::1.kind::f16 [%r9], %rd5, %rd6, %r46, 1;
    @%p12 tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r10];""")
        self.assertEqual(f.refusals, [])  # one fixed thread issues both


class Allocation(unittest.TestCase):
    ACCESS = "    mov.b32 %r25, 0;\n    tcgen05.ld.sync.aligned.32x32b.x1.b32 {%r30}, [%r25];"

    def test_rules(self):
        cases = {
            "": "R2: 0 TMEM allocations",
            "    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r10+16], 512;":
                "R2: 2 TMEM allocations",  # warps 0 and 1
            "    @%p11 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r10+16], 128;":
                "R3: absolute TMEM addresses with a 128-column allocation",
            "    @%p11 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r10+16], %r2;":
                "R2: tcgen05.alloc with a non-constant column count",
            "    @%p11 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r10+16], 96;":
                "R2: 96 columns",
            "    ld.global.u32 %r20, [%rd1];\n    setp.lt.u32 %p13, %r1, %r20;\n"
            "    @%p13 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r10+16], 512;":
                "R2: tcgen05.alloc under a runtime lane condition",
        }  # fmt: skip
        for alloc, want in cases.items():
            with self.subTest(alloc=alloc):
                self.assertIn(want, refusal(p1(self.ACCESS, alloc=alloc, threads=64)))
        f = p1(
            """
    mov.b32 %r20, 0;
L:
    @%p11 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r10+16], 512;
    add.s32 %r20, %r20, 1;
    setp.lt.s32 %p13, %r20, %r2;
    @%p13 bra L;""",
            alloc="",
        )
        self.assertIn("R2: tcgen05.alloc inside a loop", refusal(f))

    def test_no_tmem(self):
        f = p1("    bar.sync 0;", alloc="")
        self.assertEqual((f.refusals, f.tmem), ([], None))  # no tcgen05: no P1 events
        self.assertEqual([e["ev"] for e in events(f)], ["bar_init"])
        f = p1("    @%p12 tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r10];\n"
               "    bar.sync 0;", alloc="")  # fmt: skip
        self.assertEqual(
            (f.refusals, f.tmem), ([], None)
        )  # nothing allocated or accessed

    def test_triton_family(self):
        f = run(PRE + ALLOC + "    ret;\n", family="ttgir", p1=True)
        self.assertIn("P1-1 on Triton kernels is M5", refusal(f))


class NamedBarriers(unittest.TestCase):
    def test_ids_and_counts(self):
        f = p1(
            """
    bar.sync 1, 64;
    barrier.sync.aligned 2;
    mov.b32 %r20, 3;
    bar.arrive %r20, 96;
    barrier.red.popc.aligned.u32 %r21, 4, %p12;
    bar.red.and.pred %p13, 5, 32, %p12;""",
            threads=64,
        )
        self.assertEqual(f.refusals, [])
        got = [
            (e["id"]["v"], e["count"]["v"], e["sync"]) for e in events(f, "named_bar")
        ]
        self.assertEqual(
            got,
            [
                (1, 64, True),
                (2, 64, True),
                (3, 96, False),
                (4, 64, True),
                (5, 32, True),
            ],
        )
        self.assertEqual(kir_errors(f), [])

    def test_per_warp_id_splits(self):
        f = p1("    bar.sync %r3, 32;", threads=64)
        self.assertEqual(f.refusals, [])
        self.assertEqual(
            [(r["name"], r["warps"]) for r in f.roles],
            [("warps0_0", 1), ("warps1_1", 1)],
        )
        self.assertEqual([e["id"]["v"] for e in events(f, "named_bar")], [0, 1])

    def test_refusals(self):
        for body, want in (
            ("    bar.sync 1, 48;", "R7: named barrier thread count 48"),
            ("    bar.sync %r2, 64;", "R7: named barrier id is not a constant"),
            (
                "    @%p11 bar.sync 1, 32;",
                "R7: bar.sync reached by only part of the role",
            ),
        ):
            with self.subTest(body=body):
                self.assertIn(want, refusal(p1(body, threads=64)))


class LoopExits(unittest.TestCase):
    # LLVM rematerializes `mov r, %tid.x` in a loop and reads r after it only: its value
    # there is the loop-invariant tid (interp.run_loop, P1 pass), so the TMEM address
    # built from it is warp-uniform; one derived from a `for` variable stays a slot.
    def test_invariant_value_after_loop(self):
        f = p1(
            """
    mov.b32 %r20, 0;
L:
    mov.u32 %r25, %tid.x;
    add.s32 %r20, %r20, 1;
    setp.lt.s32 %p13, %r20, %r2;
    @%p13 bra L;
    shr.u32 %r26, %r25, 5;
    shl.b32 %r27, %r26, 21;
    tcgen05.ld.sync.aligned.32x32b.x1.b32 {%r30}, [%r27];""",
            threads=64,
        )
        self.assertEqual(f.refusals, [])
        self.assertEqual(kir_errors(f), [])
        p0 = run(PRE + ALLOC + "    ret;\n", family="cute", threads=64)
        self.assertEqual(p0.refusals, [])

    def test_for_variable_not_used_after_loop(self):
        f = p1("""
    mov.b32 %r20, 0;
L:
    shl.b32 %r27, %r20, 21;
    add.s32 %r20, %r20, 1;
    setp.lt.s32 %p13, %r20, 4;
    @%p13 bra L;
    tcgen05.ld.sync.aligned.32x32b.x1.b32 {%r30}, [%r27];""")
        self.assertEqual(f.refusals, [])
        loop = next(s for s in statements(f.roles[0]["body"]) if s["op"] == "for")
        ev = events(f, "tmem")[0]
        self.assertNotIn(loop["var"], str(ev))  # the slot the loop left, not `i << 21`
        self.assertEqual(kir_errors(f), [])


class SplitByIssuers(unittest.TestCase):
    ARRIVE = "    @%p12 mbarrier.arrive.shared::cta.b64 _, [%r10];\n    tcgen05.wait::ld.sync.aligned;"

    def test_thread0_arrive_splits(self):
        f = p1(self.ARRIVE, threads=128)
        self.assertEqual(f.refusals, [])
        self.assertEqual(
            [(r["name"], r["warps"]) for r in f.roles],
            [("warps0_0", 1), ("warps1_3", 3)],
        )
        self.assertEqual([len(events(f, "arrive", r["name"])) for r in f.roles], [1, 0])
        self.assertEqual(kir_errors(f), [])
        p0 = run(PRE + ALLOC + self.ARRIVE + "\n    ret;\n", family="cute", threads=128)
        self.assertEqual(
            [r["name"] for r in p0.roles], ["warps0_3"]
        )  # P0 keeps its role

    def test_runtime_condition(self):
        f = p1(
            """
    ld.global.u32 %r20, [%rd1];
    setp.lt.u32 %p13, %r1, %r20;
    @%p13 mbarrier.arrive.shared::cta.b64 _, [%r10];""",
            threads=64,
        )
        self.assertIn("R7: mbarrier arrive under a lane condition", refusal(f))


if __name__ == "__main__":
    unittest.main()
