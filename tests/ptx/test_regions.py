# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the region tree of a role (frontend/ptx/regions.py).

    python3 -I tests/ptx/test_regions.py      (or the buck target :test_regions)

On hand-written kernels: an if/else diamond (the branch target is `then`, the join is
the immediate post-dominator), an if without else, a loop whose two exit edges reach one
block (BREAK on both paths), the two inline-asm wait-loop forms as one Wait each, a
try_wait whose predicate is read elsewhere (refused), a try_wait loop of another shape
(left to the op table), and register liveness (a guarded write does not kill).
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

from ...frontend.ptx import cfg, parse, regions as R  # noqa: E402

HEAD = """
.version 9.3
.target sm_100a
.address_size 64
.extern .shared .align 16 .b8 smem[];
.visible .entry k(.param .b32 k_param_0)
.reqntid 128
{
    .reg .pred %p<8>;
    .reg .b32 %r<32>;
"""


def tree(body):
    fn = parse.parse(HEAD + body + "\n}\n").function("k")
    g = cfg.build(fn)
    (role,), _ = cfg.roles(g)
    return R.build(g, role)


def kinds(seq):
    return [type(n).__name__ if not isinstance(n, str) else n for n in seq]


class RegionTree(unittest.TestCase):
    def test_diamond(self):
        t = tree("""
    ld.param.b32 %r1, [k_param_0];
    setp.lt.s32 %p1, %r1, 4;
    @%p1 bra THEN;
    mov.b32 %r2, 1;
    bra JOIN;
THEN:
    mov.b32 %r2, 2;
JOIN:
    add.s32 %r3, %r2, 1;
    ret;
""")
        self.assertEqual(kinds(t.body), ["Block", "If", "Block"])
        node = t.body[1]
        self.assertEqual(node.join, "JOIN")
        self.assertEqual(node.then[0].name, "THEN")  # the branch target
        self.assertEqual([i.text for i in node.then[0].instrs], ["mov.b32 %r2, 2"])
        self.assertIn("%r2", t.live_in["JOIN"])
        self.assertNotIn("%r1", t.live_in["JOIN"])

    def test_if_without_else(self):
        t = tree("""
    ld.param.b32 %r1, [k_param_0];
    setp.lt.s32 %p1, %r1, 4;
    @!%p1 bra SKIP;
    mov.b32 %r2, 2;
SKIP:
    ret;
""")
        node = t.body[1]
        self.assertEqual(node.join, "SKIP")
        self.assertEqual(node.then, [])  # the target is the join: nothing to run
        self.assertEqual(kinds(node.else_), ["Block"])

    def test_loop_with_two_exit_edges(self):
        t = tree("""
    ld.param.b32 %r1, [k_param_0];
    mov.b32 %r2, 0;
HEAD:
    setp.ge.s32 %p1, %r2, %r1;
    @%p1 bra OUT;
    add.s32 %r2, %r2, 1;
    setp.eq.s32 %p2, %r2, 7;
    @%p2 bra OUT;
    bra HEAD;
OUT:
    ret;
""")
        loop = t.body[1]
        self.assertIsInstance(loop, R.Loop)
        self.assertEqual(loop.exit, "OUT")
        self.assertIn("%r2", t.defs["HEAD"])

        def ends(seq):
            out = []
            for n in seq:
                if isinstance(n, R.If):
                    out += ends(n.then) + ends(n.else_)
                elif n in (R.CONTINUE, R.BREAK):
                    out.append(n)
            return out

        self.assertEqual(sorted(ends(loop.body)), [R.BREAK, R.BREAK, R.CONTINUE])

    def test_wait_loops(self):
        for asm in (
            "{\n.reg .pred P1;\nWAIT:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%r1], %r2;\n"
            "@!P1 bra.uni WAIT;\n}",
            "{\n.reg .pred P1;\nLAB_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%r1], %r2, "
            "10000000;\n@P1 bra.uni DONE;\nbra.uni LAB_WAIT;\nDONE:\n}",
        ):
            with self.subTest(asm=asm.split()[3]):
                t = tree(f"""
    mov.u32 %r1, smem;
    mov.b32 %r2, 0;
{asm}
    ret;
""")
                waits = [n for n in t.body if isinstance(n, R.Wait)]
                self.assertEqual(len(waits), 1)
                self.assertEqual(waits[0].instr.operands[1][1][1], "%r1")

    def test_wait_result_used_is_refused(self):
        with self.assertRaisesRegex(R.RegionError, "only the blocking wait loop"):
            tree("""
    mov.u32 %r1, smem;
    mov.b32 %r2, 0;
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r1], %r2;
    @!%p1 bra WAIT;
    selp.b32 %r3, 1, 0, %p1;
    ret;
""")

    def test_other_try_wait_loop_is_a_loop(self):
        t = tree("""
    mov.u32 %r1, smem;
    mov.b32 %r2, 0;
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r1], %r2;
    add.s32 %r2, %r2, 1;
    @!%p1 bra WAIT;
    ret;
""")
        self.assertIsInstance(t.body[1], R.Loop)  # the op table refuses its try_wait

    def test_guarded_write_does_not_kill(self):
        t = tree("""
    ld.param.b32 %r1, [k_param_0];
    setp.lt.s32 %p1, %r1, 4;
    mov.b32 %r2, 0;
    @%p1 bra B;
    bra B;
B:
    @%p1 mov.b32 %r2, 5;
    add.s32 %r3, %r2, 1;
    ret;
""")
        self.assertIn("%r2", t.live_in["B"])


if __name__ == "__main__":
    unittest.main()
