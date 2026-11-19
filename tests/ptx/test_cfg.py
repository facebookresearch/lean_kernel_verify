# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the PTX control flow: basic blocks, dominators, loops, the shape check and
the Triton warp-specialize dispatch recognizer.

    python3 -I tests/ptx/test_cfg.py      (or the buck target :test_cfg)

On hand-written kernels: blocks, successors and Cooper-Harvey-Kennedy dominators of a
loop with an if inside, inline-asm wait loops (both branch forms) as one-exit loops;
a loop with two exit edges into one block (`while (a && b)`) admitted; refusals of
irreducible flow, a loop with exits to two blocks, two rets, a guarded ret,
falling off the end and an unknown label. A kernel without brx.idx is one role. The
recognizer on a warp-specialized kernel in each branch polarity (workers on @!p bra with
the default falling through and padding states; the default on @p bra with unrelated
instructions in between, a worker preamble, setmaxnreg and a partition loop), checking
the warp -> state map, padding, exit state and each role's control flow (one pass of
the switch loop header, whose definitions the regions read); the state table written as
a vector of immediates; the dispatch as a compare chain without brx.idx (each role
passing the links up to its own; looked for in Triton kernels only); padding states'
roles (marked, translate.py keeps them only if they emit); the direct dispatch of the
single-warp-specialize lowering (warp-id links, regions ending in ret, numbering, its
refusals, a padding target reached through two links: a role per path, each translated);
fork barriers in each spelling (`bar.sync 1`, `barrier.sync.aligned 1` count too); and one
refusal per protocol item (entry, header, table, exit, forks), each
naming its item, among them branches on the complement result q of a `setp p|q` (not
read as the comparison).
"""

import sys
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
    __package__ = "lkv_root.tests.ptx"  # noqa: A001

from ...frontend.ptx import cfg, parse, regions, translate as T  # noqa: E402

HEAD = """
.version 9.3
.target sm_100a
.address_size 64

.extern .shared .align 16 .b8 global_smem[];

.visible .entry k(
    .param .u64 .ptr .global .align 1 k_param_0,
    .param .b32 k_param_1
)
.reqntid %d
{
    .reg .pred %%p<16>;
    .reg .b32 %%r<64>;
    .reg .b64 %%rd<8>;
"""


def kernel(body, threads=128):
    return HEAD % threads + body + "\n}\n"


def build(text):
    return cfg.build(parse.parse(text).function("k"))


def holding(g, text):
    """The block holding the one instruction whose text starts with `text`."""
    (name,) = [
        b for b, blk in g.blocks.items() for i in blk.instrs if i.text.startswith(text)
    ]
    return name


# Warp-specialized, workers on `@!p bra`, the default falling through (8 warps, 4 of them
# default). States: warps 4, 5 -> 1 (padding), 6 -> 0, 7 -> 2 (padding), exit 3.
WS_FALL = kernel(
    """
JT:
    .branchtargets
        $P0,
        $PAD1,
        $PAD2,
        $RET;
    mov.u32 %r1, %tid.x;
    shr.u32 %r2, %r1, 5;
    shfl.sync.idx.b32 %r3, %r2, 0, 31, -1;
    setp.lt.u32 %p1, %r3, 4;
    @!%p1 bra $LOOP;
    ld.param.u64 %rd1, [k_param_0];
    st.shared.b32 [global_smem+24], 33554689;
    barrier.sync 1;
    barrier.sync 1;
    st.global.b32 [%rd1], %r1;
    barrier.sync 1;
    st.shared.b32 [global_smem+24], 50529027;
    barrier.sync 1;
$RET:
    ret;
$PAD2:
    barrier.sync 1;
    barrier.sync 1;
$LOOP:
    barrier.sync 1;
    mov.b32 %r6, global_smem;
    add.s32 %r7, %r6, %r3;
    ld.shared.b8 %r5, [%r7+20];
    setp.gt.u32 %p2, %r5, 3;
    @%p2 bra $BYPASS;
    brx.idx %r5, JT;
$P0:
    barrier.sync 1;
    add.s32 %r8, %r6, 16;
    mov.b32 %r9, 0;
    {
    .reg .pred complete;
    waitLoop:
    mbarrier.try_wait.parity.shared::cta.b64 complete, [%r8], %r9;
    @!complete bra.uni waitLoop;
    }
    barrier.sync 1;
    bra.uni $LOOP;
$PAD1:
    barrier.sync 1;
    barrier.sync 1;
    bra.uni $LOOP;
$BYPASS:
    barrier.sync 1;
    barrier.sync 1;
    bra.uni $LOOP;
""",
    threads=256,
)

# Warp-specialized, the default on `@p bra`, unrelated instructions between the setp and
# the bra, a worker preamble before the header, setmaxnreg, a 4-warp partition with a
# loop and a shared tail block (12 warps, 4 default). States: warps 4-7 -> 0, 8, 9 -> 3
# (padding), 10 -> 1, 11 -> 2, exit 4: bytes 0,0,0,0,3,3,1,2 and 4 x 8.
WS_TAKEN = kernel(
    """
JT:
    .branchtargets
        $S0,
        $S1,
        $S2,
        $S3,
        $RET;
    ld.param.u64 %rd1, [k_param_0];
$TOP:
    mov.u32 %r1, %tid.x;
    shr.u32 %r2, %r1, 5;
    shfl.sync.idx.b32 %r3, %r2, 0, 31, -1;
    setp.lt.u32 %p1, %r3, 4;
    mov.b32 %r40, global_smem;
    and.b32 %r41, %r1, 511;
    @%p1 bra $DEF;
    ld.param.b32 %r42, [k_param_1];
    bra.uni $HDR;
$TAIL:
    barrier.sync 1;
    setmaxnreg.inc.sync.aligned.u32 120;
$HDR:
    setmaxnreg.dec.sync.aligned.u32 120;
    barrier.sync 1;
    mov.b32 %r60, global_smem;
    add.s32 %r61, %r60, %r3;
    ld.shared.b8 %r59, [%r61+1028];
    setp.gt.u32 %p3, %r59, 4;
    @%p3 bra $BYP;
    setp.lt.s32 %p6, %r42, 256;
    brx.idx %r59, JT;
$S0:
    setmaxnreg.dec.sync.aligned.u32 64;
    barrier.sync 1;
    mov.b32 %r10, 0;
$S0L:
    add.s32 %r10, %r10, 1;
    st.global.b32 [%rd1], %r10;
    setp.lt.s32 %p4, %r10, %r42;
    @%p4 bra $S0L;
    bra.uni $TAIL;
$S1:
    barrier.sync 1;
    st.global.b32 [%rd1+4], %r42;
    barrier.sync 1;
    bra.uni $HDR;
$S2:
    barrier.sync 1;
    st.global.b32 [%rd1+8], %r42;
    bra.uni $TAIL;
$S3:
    setmaxnreg.dec.sync.aligned.u32 104;
    barrier.sync 1;
    barrier.sync 1;
    setmaxnreg.inc.sync.aligned.u32 120;
    bra.uni $HDR;
$BYP:
    barrier.sync 1;
    barrier.sync 1;
    bra.uni $HDR;
$DEF:
    setmaxnreg.inc.sync.aligned.u32 256;
    st.shared.b64 [global_smem+1032], 144399974472351744;
    barrier.sync 1;
    setmaxnreg.dec.sync.aligned.u32 240;
    barrier.sync 1;
    mov.b32 %r20, 0;
$DL:
    add.s32 %r20, %r20, 1;
    setp.lt.s32 %p5, %r20, 8;
    @%p5 bra $DL;
    barrier.sync 1;
    st.shared.b64 [global_smem+1032], 289360691352306692;
    barrier.sync 1;
$RET:
    ret;
""",
    threads=384,
)


# Warp-specialized with at most 3 worker states: LLVM compares the state byte in a chain
# instead of a brx.idx jump table (the last link negated). 12 warps, 4 default: warps
# 4-7 -> 0, 8-11 -> 1, exit 2.
WS_CHAIN = kernel(
    """
    mov.u32 %r1, %tid.x;
    shr.u32 %r2, %r1, 5;
    shfl.sync.idx.b32 %r3, %r2, 0, 31, -1;
    ld.param.u64 %rd1, [k_param_0];
    setp.lt.u32 %p1, %r3, 4;
    @!%p1 bra $LOOP;
    st.shared.b32 [global_smem+28], 0;
    st.shared.b32 [global_smem+32], 16843009;
    barrier.sync 1;
    barrier.sync 1;
    st.global.b32 [%rd1], %r1;
    barrier.sync 1;
    st.shared.b32 [global_smem+32], 33686018;
    st.shared.b32 [global_smem+28], 33686018;
    barrier.sync 1;
$RET:
    ret;
$S1:
    barrier.sync 1;
    st.global.b32 [%rd1+4], %r6;
    barrier.sync 1;
$LOOP:
    barrier.sync 1;
    mov.b32 %r6, global_smem;
    add.s32 %r7, %r6, %r3;
    ld.shared.b8 %r5, [%r7+24];
    setp.eq.b32 %p2, %r5, 0;
    @%p2 bra $S0;
    setp.eq.b32 %p3, %r5, 1;
    @%p3 bra $S1;
    setp.ne.b32 %p4, %r5, 2;
    @!%p4 bra $RET;
    barrier.sync 1;
    barrier.sync 1;
    bra.uni $LOOP;
$S0:
    barrier.sync 1;
    st.global.b32 [%rd1+8], %r6;
    barrier.sync 1;
    bra.uni $LOOP;
""",
    threads=384,
)


# Direct dispatch (Triton's single-warp-specialize lowering): no state bytes, no switch
# loop; the workers branch on their warp id (largest start first, LLVM turning `wid >= 6`
# into `wid == 6` below 7) and every region ends in ret. 8 warps, 4 default: warp 7 -> a
# region, warp 6 -> a region with a loop that loads a capture first, warps 4, 5 -> padding.
WS_DIRECT = kernel(
    """
    mov.u32 %r1, %tid.x;
    shr.u32 %r2, %r1, 5;
    shfl.sync.idx.b32 %r3, %r2, 0, 31, -1;
    ld.param.u64 %rd1, [k_param_0];
    setp.lt.u32 %p1, %r3, 4;
    @!%p1 bra $HDR;
    st.shared.b32 [global_smem+8], %r1;
    barrier.sync 1;
    barrier.sync 1;
    st.global.b32 [%rd1], %r1;
    bra.uni $RET;
$HDR:
    barrier.sync 1;
    setp.gt.u32 %p2, %r3, 6;
    @!%p2 bra $L2;
    setmaxnreg.dec.sync.aligned.u32 24;
    barrier.sync 1;
    st.global.b32 [%rd1+4], %r3;
    bra.uni $RET;
$L2:
    setp.eq.b32 %p3, %r3, 6;
    @!%p3 bra $PAD;
    setmaxnreg.dec.sync.aligned.u32 24;
    ld.shared.b32 %r4, [global_smem+8];
    barrier.sync 1;
    mov.b32 %r10, 0;
$P0L:
    add.s32 %r10, %r10, 1;
    st.global.b32 [%rd1+8], %r10;
    setp.lt.s32 %p4, %r10, 4;
    @%p4 bra $P0L;
    bra.uni $RET;
$PAD:
    setmaxnreg.dec.sync.aligned.u32 24;
    barrier.sync 1;
$RET:
    ret;
""",
    threads=256,
)


# WS_DIRECT without the capture and the loop, whose padding target $PAD is reached through
# two links: warp 5 takes $L2's branch, warp 4 the next link's (`wid != 6`), warp 6 falls
# through to its region.
WS_DIRECT_PAD_TWO_LINKS = kernel(
    """
    mov.u32 %r1, %tid.x;
    shr.u32 %r2, %r1, 5;
    shfl.sync.idx.b32 %r3, %r2, 0, 31, -1;
    ld.param.u64 %rd1, [k_param_0];
    setp.lt.u32 %p1, %r3, 4;
    @!%p1 bra $HDR;
    barrier.sync 1;
    barrier.sync 1;
    st.global.b32 [%rd1], %r1;
    bra.uni $RET;
$HDR:
    barrier.sync 1;
    setp.gt.u32 %p2, %r3, 6;
    @!%p2 bra $L2;
    barrier.sync 1;
    st.global.b32 [%rd1+4], %r3;
    bra.uni $RET;
$L2:
    setp.eq.b32 %p3, %r3, 5;
    @%p3 bra $PAD;
    setp.eq.b32 %p4, %r3, 6;
    @!%p4 bra $PAD;
    barrier.sync 1;
    st.global.b32 [%rd1+8], %r3;
    bra.uni $RET;
$PAD:
    barrier.sync 1;
$RET:
    ret;
""",
    threads=256,
)


class Shape(unittest.TestCase):
    def test_blocks_dominators_loops(self):
        g = build(
            kernel(
                """
    mov.u32 %r1, %tid.x;
    mov.b32 %r2, 0;
$L:
    setp.eq.u32 %p1, %r1, 0;
    @%p1 bra $ELSE;
    add.s32 %r2, %r2, 1;
    bra.uni $JOIN;
$ELSE:
$ELSE2:
    add.s32 %r2, %r2, 2;
$JOIN:
    setp.lt.s32 %p2, %r2, 100;
    @%p2 bra $L;
    ret;
"""
            )
        )
        b, e = g.blocks, g.entry
        then, after = holding(g, "add.s32 %r2, %r2, 1"), holding(g, "ret")
        # $ELSE and $ELSE2 name one block; the fall-through of a guarded bra and the
        # code after the loop are named after the line of their first instruction.
        self.assertEqual(list(b), [e, "$L", then, "$ELSE", "$JOIN", after])
        self.assertEqual(
            (e, then), (f"@{b[e].instrs[0].line}", f"@{b[then].instrs[0].line}")
        )
        self.assertEqual(b["$L"].succs, ("$ELSE", then))
        self.assertEqual(b[then].succs, ("$JOIN",))
        self.assertEqual(b["$JOIN"].succs, ("$L", after))
        self.assertEqual(b[after].succs, ())
        self.assertEqual(sorted(b["$JOIN"].preds), sorted(["$ELSE", then]))
        idom = {e: e, "$L": e, then: "$L", "$ELSE": "$L", "$JOIN": "$L", after: "$JOIN"}
        self.assertEqual(g.idom, idom)
        self.assertTrue(g.dominates("$L", after) and not g.dominates(then, "$JOIN"))
        self.assertEqual(g.loops, {"$L": {"$L", then, "$ELSE", "$JOIN"}})

    def test_asm_wait_loops(self):
        # Both forms of an inline-asm wait loop, their labels scoped per asm block.
        g = build(
            kernel(
                """
    mov.b32 %r8, global_smem;
    mov.b32 %r9, 0;
    {
    .reg .pred complete;
    waitLoop:
    mbarrier.try_wait.parity.shared::cta.b64 complete, [%r8], %r9;
    @!complete bra.uni waitLoop;
    }
    {
    .reg .pred P1;
    LAB_WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 P1, [%r8], %r9, 10000000;
    @P1 bra.uni DONE;
    bra.uni LAB_WAIT;
    DONE:
    }
    ret;
"""
            )
        )
        back = holding(g, "bra.uni LAB_WAIT")
        self.assertEqual(
            g.loops, {"waitLoop#1": {"waitLoop#1"}, "LAB_WAIT#2": {"LAB_WAIT#2", back}}
        )
        self.assertEqual(g.blocks["waitLoop#1"].succs, ("waitLoop#1", "LAB_WAIT#2"))
        self.assertEqual(g.blocks["LAB_WAIT#2"].succs, ("DONE#2", back))

    def test_unreachable_code_is_dropped(self):
        g = build(
            kernel("    bra.uni $A;\n    add.s32 %r1, %r1, 1;\n$A:\n    ret;\n$END:\n")
        )
        self.assertEqual(list(g.blocks), [g.entry, "$A"])
        self.assertEqual(len(g.blocks[g.entry].instrs), 1)

    def test_two_exits_into_one_block(self):
        g = build(
            kernel("""
$L:
    add.s32 %r2, %r2, 1;
    @%p1 bra $X;
    add.s32 %r2, %r2, 3;
    @%p2 bra $L;
$X:
    ret;
""")
        )
        self.assertEqual(list(g.loops), ["$L"])

    def test_trap_goes_on_after_its_branch(self):
        # Triton's device asserts after the ret (`call __assertfail; trap;`), CuTe's
        # `if x: trap()`: a trap block goes on where its branch's other side does, as if
        # the trap were not reached; a guarded trap ends no block
        g = build(
            kernel("""
    @%p1 bra $A1;
    add.s32 %r2, %r2, 1;
    @%p2 bra $J;
    trap;
$J:
    @%p3 bra $A2;
    @%p4 trap;
    add.s32 %r2, %r2, 2;
    ret;
$A1:
    trap;
$A2:
    trap;
""")
        )
        b1, b2 = holding(g, "add.s32 %r2, %r2, 1"), holding(g, "@%p4 trap")
        self.assertEqual(g.blocks["$A1"].succs, (b1,))
        (cute,) = [b for b in g.blocks[b1].succs if b != "$J"]
        self.assertEqual(g.blocks[cute].succs, ("$J",))
        self.assertEqual(g.blocks["$A2"].succs, (b2,))
        self.assertEqual(g.blocks[b2].instrs[-1].base, "ret")
        # reached otherwise than by one conditional branch: refused
        for body in ("    bra.uni $A;\n    ret;\n$A:\n    trap;\n",
                     "    @%p1 bra $A;\n    @%p2 bra $A;\n    ret;\n$A:\n    trap;\n"):  # fmt: skip
            with self.assertRaisesRegex(cfg.CfgError, "not reached by one conditional"):
                build(kernel(body))

    def test_refusals(self):
        cases = {
            "irreducible control flow": """
    setp.eq.u32 %p1, %r1, 0;
    @%p1 bra $B;
$A:
    add.s32 %r2, %r2, 1;
    @%p2 bra $X;
$B:
    add.s32 %r2, %r2, 2;
    @%p3 bra $A;
$X:
    ret;
""",
            "has 2 exit edges": """
$L:
    add.s32 %r2, %r2, 1;
    @%p1 bra $X;
    @%p2 bra $L;
    add.s32 %r2, %r2, 3;
$X:
    ret;
""",
            "2 ret/exit": "    @%p1 bra $A;\n    ret;\n$A:\n    ret;\n",
            "guarded ret": "    @%p1 ret;\n    ret;\n",
            "falls off the end": "    add.s32 %r1, %r1, 1;\n",
            "no such label": "    bra.uni $NOWHERE;\n    ret;\n",
        }
        for msg, body in cases.items():
            with self.subTest(msg), self.assertRaisesRegex(cfg.CfgError, msg):
                build(kernel(body))

    def test_one_role_without_brx(self):
        g = build(kernel("    mov.u32 %r1, %tid.x;\n    ret;\n"))
        rs, d = cfg.roles(g)
        self.assertIsNone(d)
        self.assertEqual(
            [(r.name, r.warps, r.entry) for r in rs],
            [("default", (0, 1, 2, 3), g.entry)],
        )
        self.assertEqual(rs[0].succs, {g.entry: ()})


class WarpSpecialize(unittest.TestCase):
    def test_fall_through_polarity_and_padding(self):
        g = build(WS_FALL)
        rs, d = cfg.roles(g)
        self.assertEqual(d.default_warps, 4)
        self.assertEqual(d.warps, {4: 1, 5: 1, 6: 0, 7: 2})
        self.assertEqual((d.exit_state, d.padding, d.header), (3, (1, 2), "$LOOP"))
        # padding states have roles too, which translate.py keeps only if they emit
        self.assertEqual(
            [(r.name, r.warps, r.padding) for r in rs],
            [("default", (0, 1, 2, 3), False), ("partition0", (6,), False),
             ("partition1", (4, 5), True), ("partition2", (7,), True)],
        )  # fmt: skip
        entry = d.dispatch
        self.assertEqual(entry, g.entry)
        # The default falls through. Partition 0 jumps from the dispatch to the header
        # (no worker preamble), passes it once with the bypass and brx.idx resolved to
        # its target, and its back edge into the header becomes ret; the region reads
        # %r6, which the header defines.
        default, p0 = rs[0].succs, rs[1].succs
        fall, tail = g.blocks[entry].succs[1], g.blocks["waitLoop#1"].succs[1]
        xb = g.blocks["$LOOP"].succs[1]
        self.assertEqual(default[entry], (fall,))
        self.assertEqual(list(default), [entry, fall, "$RET"])
        self.assertEqual(p0[entry], ("$LOOP",))
        self.assertEqual((p0["$LOOP"], p0[xb]), ((xb,), ("$P0",)))
        self.assertEqual(p0[tail], ("$RET",))
        self.assertEqual(p0["$RET"], ())
        self.assertEqual(
            set(p0), {entry, "$LOOP", xb, "$P0", "waitLoop#1", tail, "$RET"}
        )
        # A 2-CTA kernel ends every worker region, padding included, with a cluster
        # arrive: still padding.
        text = WS_FALL.replace(
            "$PAD1:\n", "$PAD1:\n    barrier.cluster.arrive.aligned;\n"
        )
        self.assertEqual(cfg.roles(build(text))[1].padding, (1, 2))

    def test_taken_polarity_preamble_and_loops(self):
        g = build(WS_TAKEN)
        rs, d = cfg.roles(g)
        self.assertEqual(d.warps, {4: 0, 5: 0, 6: 0, 7: 0, 8: 3, 9: 3, 10: 1, 11: 2})
        self.assertEqual((d.default_warps, d.exit_state, d.padding), (4, 4, (3,)))
        self.assertEqual((d.dispatch, d.header), ("$TOP", "$HDR"))
        names = [(r.name, r.warps) for r in rs if not r.padding]
        self.assertEqual(
            names,
            [
                ("default", (0, 1, 2, 3)),
                ("partition0", (4, 5, 6, 7)),
                ("partition1", (10,)),
                ("partition2", (11,)),
            ],
        )
        self.assertEqual([(r.name, r.warps) for r in rs if r.padding],
                         [("partition3", (8, 9))])  # fmt: skip
        default, p0, p1 = (r.succs for r in rs[:3])
        self.assertEqual(default["$TOP"], ("$DEF",))
        after = g.blocks["$DL"].succs[1]
        self.assertEqual(list(default), [g.entry, "$TOP", "$DEF", "$DL", after, "$RET"])
        # Workers: preamble, one header pass to the state's target (the brx.idx block
        # also holds an instruction LLVM sank there), the tail block shared by states 0
        # and 2 ends in ret. The bypass target is in no role.
        preamble, xb = g.blocks["$TOP"].succs[1], g.blocks["$HDR"].succs[1]
        self.assertEqual(len(g.blocks[xb].instrs), 2)
        self.assertEqual(p0["$TOP"], (preamble,))
        self.assertEqual(
            (p0[preamble], p0["$HDR"], p0[xb]), (("$HDR",), (xb,), ("$S0",))
        )
        self.assertEqual(p0["$TAIL"], ("$RET",))
        self.assertEqual(p0["$S0L"], ("$S0L", g.blocks["$S0L"].succs[1]))
        self.assertEqual(p1[xb], ("$S1",))
        self.assertEqual(p1["$S1"], ("$RET",))
        self.assertNotIn("$TAIL", p1)
        self.assertEqual(rs[3].succs["$TAIL"], ("$RET",))
        for r in rs:
            self.assertNotIn("$BYP", r.succs)
        self.assertNotIn("$HDR", default)

    def test_vector_of_immediates_table(self):
        # LLVM merges the state stores of adjacent warps into one vector store
        text = WS_TAKEN.replace(
            "st.shared.b64 [global_smem+1032], 144399974472351744",
            "st.shared.v2.b32 [global_smem+1032], {0, 33620739}",
        ).replace(
            "st.shared.b64 [global_smem+1032], 289360691352306692",
            "st.shared.v2.b32 [global_smem+1032], {67372036, 67372036}",
        )
        rs, d = cfg.roles(build(text))
        self.assertEqual(d.warps, {4: 0, 5: 0, 6: 0, 7: 0, 8: 3, 9: 3, 10: 1, 11: 2})
        self.assertEqual((d.exit_state, d.padding), (4, (3,)))
        self.assertEqual(
            [r.name for r in rs], [r.name for r in cfg.roles(build(WS_TAKEN))[0]]
        )
        with self.assertRaisesRegex(
            cfg.CfgError, r"dispatch \(table\): .*not as an immediate"
        ):
            cfg.roles(build(text.replace("{0, 33620739}", "{%r1, 33620739}")))

    def test_compare_chain(self):
        g = build(WS_CHAIN)
        rs, d = cfg.roles(g)
        self.assertEqual(
            d.warps, {**dict.fromkeys(range(4, 8), 0), **dict.fromkeys(range(8, 12), 1)}
        )
        self.assertEqual(
            (d.default_warps, d.exit_state, d.padding, d.header), (4, 2, (), "$LOOP")
        )
        self.assertEqual(
            [(r.name, r.warps) for r in rs],
            [("default", (0, 1, 2, 3)), ("partition0", (4, 5, 6, 7)),
             ("partition1", (8, 9, 10, 11))],
        )  # fmt: skip
        # each role passes the links up to its own, then its region, whose back edge
        # (S1 falls through into the header) becomes ret; the bypass is in no role
        link2 = g.blocks["$LOOP"].succs[1]
        link3 = g.blocks[link2].succs[1]
        p0, p1 = rs[1].succs, rs[2].succs
        self.assertEqual(
            (p0["$LOOP"], p0["$S0"], p0["$RET"]), (("$S0",), ("$RET",), ())
        )
        self.assertEqual(
            (p1["$LOOP"], p1[link2], p1["$S1"]), ((link2,), ("$S1",), ("$RET",))
        )
        self.assertNotIn(link2, p0)
        bypass = g.blocks[link3].succs[1]
        for r in rs:
            self.assertNotIn(link3, r.succs)
            self.assertNotIn(bypass, r.succs)
        bad = {
            "has `st.shared": WS_CHAIN.replace(
                "    setp.eq.b32 %p3",
                "    st.shared.b32 [global_smem], 0;\n    setp.eq.b32 %p3",
            ),
            "compares the state byte with 256": WS_CHAIN.replace(
                "%r5, 1;", "%r5, 256;"
            ),
        }
        for msg, text in bad.items():
            with (
                self.subTest(msg),
                self.assertRaisesRegex(cfg.CfgError, rf"dispatch \(header\): .*{msg}"),
            ):
                cfg.roles(build(text))

        # a link on the complement result q of `setp.eq p|q` (taken when s != v) is no
        # link: on the first, no chain (one role); on another, the chain ends early
        def pair(v):
            p = f"%p{v + 2}"
            return WS_CHAIN.replace(
                f"{p}, %r5, {v};\n    @{p} bra", f"{p}|%p9, %r5, {v};\n    @%p9 bra"
            )

        rs, d = cfg.roles(build(WS_CHAIN), "cute")  # a chain is Triton's lowering only
        self.assertEqual(([r.name for r in rs], d), (["default"], None))
        rs, d = cfg.roles(build(pair(0)))
        self.assertEqual(([r.name for r in rs], d), (["default"], None))
        with self.assertRaisesRegex(cfg.CfgError, r"dispatch \(exit\): .*not the one"):
            cfg.roles(build(pair(1)))

    def test_direct_dispatch(self):
        g = build(WS_DIRECT)
        rs, d = cfg.roles(g)
        self.assertTrue(d.direct)
        self.assertEqual((d.default_warps, d.exit_state, d.header), (4, None, "$HDR"))
        self.assertEqual((d.warps, d.padding), ({4: 2, 5: 2, 6: 0, 7: 1}, (2,)))
        # partitions numbered by first warp, those that do more than sync first
        self.assertEqual(
            [(r.name, r.warps, r.padding) for r in rs],
            [("default", (0, 1, 2, 3), False), ("partition0", (6,), False),
             ("partition1", (7,), False), ("partition2", (4, 5), True)],
        )  # fmt: skip
        default, p0, p1, pad = (r.succs for r in rs)
        fall = g.blocks[g.entry].succs[1]
        r7 = g.blocks["$HDR"].succs[1]
        r6 = g.blocks["$L2"].succs[1]
        self.assertEqual(default[g.entry], (fall,))
        self.assertNotIn("$HDR", default)
        # each worker passes the links up to its region, which ends in ret
        self.assertEqual(
            (p0[g.entry], p0["$HDR"], p0["$L2"]), (("$HDR",), ("$L2",), (r6,))
        )
        self.assertEqual((p1["$HDR"], p1[r7], p1["$RET"]), ((r7,), ("$RET",), ()))
        self.assertNotIn("$L2", p1)
        self.assertEqual((pad["$L2"], pad["$PAD"]), (("$PAD",), ("$RET",)))
        self.assertIn("$P0L", p0)
        for r in (p1, pad):
            self.assertNotIn("$P0L", r)
        # looked for in the Triton family only
        rs, d = cfg.roles(g, "cute")
        self.assertEqual(([r.name for r in rs], d), (["default"], None))
        # the comparison may be on any form of the warp id's test (here `wid >= 7`)
        rs, d = cfg.roles(build(WS_DIRECT.replace("setp.gt.u32 %p2, %r3, 6",
                                                  "setp.ge.s32 %p2, %r3, 7")))  # fmt: skip
        self.assertEqual(d.warps, {4: 2, 5: 2, 6: 0, 7: 1})
        cases = [
            (
                "forks",
                "2 fork barriers in the header",
                WS_DIRECT.replace("$HDR:\n    barrier.sync 1;\n",
                                  "$HDR:\n    barrier.sync 1;\n    barrier.sync 1;\n"),
            ),
            (
                "forks",
                "reaches ret after 0 fork barriers, not 1",
                WS_DIRECT.replace("    setmaxnreg.dec.sync.aligned.u32 24;\n    barrier.sync 1;\n$RET:",
                                  "    setmaxnreg.dec.sync.aligned.u32 24;\n$RET:"),
            ),
            (
                "forks",
                "the default reaches ret after 1 fork barriers, not 2",
                WS_DIRECT.replace("    barrier.sync 1;\n    barrier.sync 1;\n    st.global",
                                  "    barrier.sync 1;\n    st.global"),
            ),
            (
                "forks",
                "share",
                WS_DIRECT.replace("    st.global.b32 [%rd1], %r1;\n    bra.uni $RET;",
                                  "    st.global.b32 [%rd1], %r1;\n    bra.uni $P0L;"),
            ),
            (
                "header",
                "unexpected `st.global",
                WS_DIRECT.replace("$HDR:\n    barrier.sync 1;\n",
                                  "$HDR:\n    barrier.sync 1;\n    st.global.b32 [%rd1], %r3;\n"),
            ),
            (
                "entry",
                "taken when wid < 4",
                WS_DIRECT.replace("@!%p1 bra $HDR", "@%p1 bra $HDR"),
            ),
        ]  # fmt: skip
        for item, msg, text in cases:
            with self.subTest(f"{item}: {msg}"):
                with self.assertRaisesRegex(
                    cfg.CfgError, rf"dispatch \({item}\): .*{msg}"
                ):
                    cfg.roles(build(text))

    def test_direct_padding_through_two_links(self):
        g = build(WS_DIRECT_PAD_TWO_LINKS)
        rs, d = cfg.roles(g)
        self.assertEqual((d.warps, d.padding), ({4: 2, 5: 3, 6: 0, 7: 1}, (2, 3)))
        self.assertEqual(
            [(r.name, r.warps, r.padding) for r in rs],
            [("default", (0, 1, 2, 3), False), ("partition0", (6,), False),
             ("partition1", (7,), False), ("partition2", (4,), True),
             ("partition3", (5,), True)],
        )  # fmt: skip
        # each padding role passes its own links (one role for both passed none, and
        # its region tree failed with a KeyError)
        link = g.blocks["$L2"].succs[1]
        pad4, pad5 = rs[3].succs, rs[4].succs
        self.assertEqual(
            (pad4["$HDR"], pad4["$L2"], pad4[link]), (("$L2",), (link,), ("$PAD",))
        )
        self.assertEqual((pad5["$HDR"], pad5["$L2"]), (("$L2",), ("$PAD",)))
        self.assertNotIn(link, pad5)
        for r in rs:
            regions.build(g, r)
        params = [{"ptx": "k_param_0", "kind": "ptr", "name": "x", "size": 8, "fields": {}},
                  {"ptx": "k_param_1", "kind": "scalar", "name": "n", "size": 4, "fields": {}}]  # fmt: skip
        pol = T.policy("ttgir", params=params, tensors={"x": {"elem_bytes": 4}})
        f = T.translate(WS_DIRECT_PAD_TWO_LINKS, "k", [{"block": [256, 1, 1]}], pol)
        self.assertEqual(f.refusals, [])
        self.assertEqual(
            [r["name"] for r in f.roles], ["default", "partition0", "partition1"]
        )
        # a target that does more than sync, reached through two links: refused
        text = WS_DIRECT_PAD_TWO_LINKS.replace(
            "$PAD:\n    barrier.sync 1;\n",
            "$PAD:\n    barrier.sync 1;\n    st.global.b32 [%rd1+12], %r3;\n",
        )
        with self.assertRaisesRegex(
            cfg.CfgError, r"dispatch \(chain\): warps \[4, 5\] reach the partition at "
        ):
            cfg.roles(build(text))

    def test_fork_barrier_spellings(self):
        # bar.sync 1 and barrier.sync.aligned 1 are fork barriers too, as the interpreter
        # counts them: the same dispatch, and one more in the default breaks the count
        alt = WS_DIRECT.replace(
            "$HDR:\n    barrier.sync 1;\n", "$HDR:\n    bar.sync 1;\n"
        )
        alt = alt.replace(
            "    barrier.sync 1;\n$RET:", "    barrier.sync.aligned 1;\n$RET:"
        )
        self.assertEqual(cfg.roles(build(alt))[1].warps, {4: 2, 5: 2, 6: 0, 7: 1})
        two = "    barrier.sync 1;\n    barrier.sync 1;\n    st.global"
        three = (
            "    barrier.sync 1;\n    bar.sync 1;\n    barrier.sync 1;\n    st.global"
        )
        for text, msg in ((WS_DIRECT, "after 3 fork barriers, not 2"),
                          (WS_FALL, "after 5 fork barriers, not 4")):  # fmt: skip
            with self.subTest(msg):
                with self.assertRaisesRegex(
                    cfg.CfgError, rf"dispatch \(forks\): .*{msg}"
                ):
                    cfg.roles(build(text.replace(two, three)))

    def test_protocol_refusals(self):
        def drop(text, line):
            assert line in text, line
            return text.replace(line, "", 1)

        table = "    st.shared.b32 [global_smem+24], 33554689;\n"
        exit_ = "    st.shared.b32 [global_smem+24], 50529027;\n"
        cases = [
            # entry
            ("entry", "is not on wid < D", WS_FALL.replace("setp.lt", "setp.eq", 1)),
            (
                "entry",
                "not shfl.sync.idx",
                WS_FALL.replace("setp.lt.u32 %p1, %r3, 4", "setp.lt.u32 %p1, %r2, 4"),
            ),
            (
                "entry",
                "not shfl.sync.idx",
                WS_FALL.replace("shfl.sync.idx.b32", "shfl.sync.bfly.b32"),
            ),
            # a guarded definition may not execute: it reaches nothing for sure
            (
                "entry",
                "is not on wid < D",
                WS_FALL.replace("    setp.lt.u32 %p1", "    @%p7 setp.lt.u32 %p1"),
            ),
            (
                "entry",
                "taken when wid < 4",
                WS_FALL.replace("@!%p1 bra $LOOP", "@%p1 bra $LOOP"),
            ),
            # the guard is the complement result of a setp pair (and not even read)
            (
                "entry",
                "is not on wid < D",
                WS_CHAIN.replace(
                    "setp.lt.u32 %p1, %r3, 4;\n    @!%p1 bra",
                    "setp.lt.u32 %p1|%p8, %r3, 4;\n    @!%p8 bra",
                ),
            ),
            (
                "entry",
                "taken when wid < 4",
                WS_TAKEN.replace("setp.lt.u32 %p1, %r3, 4", "setp.ge.u32 %p1, %r3, 4"),
            ),
            # header
            (
                "header",
                "unexpected `ld.shared.b8",
                WS_FALL.replace("$LOOP:\n    barrier.sync 1;\n", "$LOOP:\n"),
            ),
            (
                "header",
                r"is not \[smem \+ wid \+ K\]",
                WS_FALL.replace("add.s32 %r7, %r6, %r3", "add.s32 %r7, %r6, %r2"),
            ),
            # the only definition of %r30 does not dominate its use
            (
                "header",
                r"is not \[smem \+ wid \+ K\]",
                WS_FALL.replace(
                    "add.s32 %r7, %r6, %r3", "add.s32 %r7, %r6, %r30"
                ).replace("$PAD1:\n", "$PAD1:\n    mov.u32 %r30, %r3;\n"),
            ),
            (
                "header",
                r"`@\(state > 3\) bra`",
                WS_FALL.replace("setp.gt.u32 %p2, %r5, 3", "setp.gt.u32 %p2, %r5, 2"),
            ),
            (
                "header",
                r"`@\(state > 3\) bra`",
                WS_FALL.replace(
                    "setp.gt.u32 %p2, %r5, 3;\n    @%p2 bra",
                    "setp.gt.u32 %p2|%p9, %r5, 3;\n    @%p9 bra",
                ),
            ),
            (
                "header",
                "unexpected",
                WS_FALL.replace(
                    "    mov.b32 %r6, global_smem;",
                    "    st.shared.b32 [global_smem], 0;\n    mov.b32 %r6, global_smem;",
                ),
            ),
            (
                "header",
                "unexpected `st.global",
                WS_FALL.replace(
                    "    brx.idx %r5, JT;",
                    "    st.global.b32 [%rd1], %r5;\n    brx.idx %r5, JT;",
                ),
            ),
            # table and exit
            ("table", r"warps \[4, 5, 6, 7\] is not written", drop(WS_FALL, table)),
            (
                "table",
                "after 1 fork barriers",
                drop(WS_FALL, table).replace(
                    "    barrier.sync 1;\n", "    barrier.sync 1;\n" + table, 1
                ),
            ),
            ("table", "start in states", WS_FALL.replace("33554689", "33554691")),
            ("exit", r"warps \[4, 5, 6, 7\] is not written", drop(WS_FALL, exit_)),
            (
                "exit",
                "written twice",
                WS_FALL.replace(
                    exit_, exit_ + "    st.shared.b8 [global_smem+25], 3;\n"
                ),
            ),
            (
                "table",
                "under a guard",
                WS_FALL.replace(
                    table, "    @%p1 st.shared.b32 [global_smem+24], 33554689;\n"
                ),
            ),
            (
                "table",
                "outside the default",
                WS_FALL.replace(
                    "$PAD1:\n", "$PAD1:\n    st.shared.b8 [global_smem+26], 1;\n"
                ),
            ),
            (
                "exit",
                "not on every path",
                WS_TAKEN.replace(
                    "    barrier.sync 1;\n    st.shared.b64 [global_smem+1032], 289360691352306692;\n",
                    "    barrier.sync 1;\n    @%p5 bra $SKIP;\n"
                    "    st.shared.b64 [global_smem+1032], 289360691352306692;\n$SKIP:\n",
                ),
            ),
            ("exit", "not the one state", WS_FALL.replace("50529027", "33686018")),
            # forks
            (
                "forks",
                "after 1 fork barriers, not 2",
                WS_FALL.replace(
                    "    barrier.sync 1;\n    bra.uni $LOOP;\n$PAD1:",
                    "    bra.uni $LOOP;\n$PAD1:",
                ),
            ),
            (
                "forks",
                "depend on the path",
                WS_TAKEN.replace("$DL:\n", "$DL:\n    barrier.sync 1;\n"),
            ),
            ("forks", "not 4", WS_FALL.replace(exit_ + "    barrier.sync 1;\n", exit_)),
        ]
        for item, msg, text in cases:
            with self.subTest(f"{item}: {msg}"):
                with self.assertRaisesRegex(
                    cfg.CfgError, rf"dispatch \({item}\): .*{msg}"
                ):
                    cfg.roles(build(text))

    def test_two_brx(self):
        text = WS_FALL.replace(
            "$RET;\n", "$RET;\nJT2:\n    .branchtargets\n        $D1;\n", 1
        ).replace(
            "    st.global.b32", "    brx.idx %r1, JT2;\n$D1:\n    st.global.b32", 1
        )
        with self.assertRaisesRegex(cfg.CfgError, "only one dispatch"):
            cfg.roles(build(text))


if __name__ == "__main__":
    unittest.main()
