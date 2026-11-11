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
the switch loop header, whose definitions the regions read); and one
refusal per protocol item (entry, header, table, exit, forks), each naming its item.
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

from ...frontend.ptx import cfg, parse  # noqa: E402

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
        self.assertEqual(
            [(r.name, r.warps) for r in rs],
            [("default", (0, 1, 2, 3)), ("partition0", (6,))],
        )
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
        names = [(r.name, r.warps) for r in rs]
        self.assertEqual(
            names,
            [
                ("default", (0, 1, 2, 3)),
                ("partition0", (4, 5, 6, 7)),
                ("partition1", (10,)),
                ("partition2", (11,)),
            ],
        )
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
