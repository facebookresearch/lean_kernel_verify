# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the work anchors (frontend/ptx/anchors.py).

    python3 -I tests/ptx/test_anchors.py      (or the buck target :test_anchors)

On hand-written PTX over tiny Python sources written to a temporary export root (the
.file paths name them; columns 1-based as Triton writes them). Triton assignment
anchors: a plain name whose region also holds ld.params carrying the expression's .loc
(never the value, even when used elsewhere), whose value survives a guarded rewrite, and
which does not give way to an instruction one column to its left; a stray result in the
region, also a predicate read only as a guard, is a case error naming both; a tuple
target through a conditional expression and an inlined helper in another file, whose
loads share a scaled index (the shared value gives way to the load) and whose dead
element gets a "novalue" site at the last of the other fields' sites (alone: no site
anywhere, an error); the other branch, where two elements are one name of the helper and
the third is computed from it; a tie between two independent results broken by the root
expression's position; a statement without an instruction (LLVM computed it under a
later .loc) next to one with a site: "novalue"; an occurrence in a dropped constexpr
branch (a note) and a loop-carried AugAssign whose register has two definitions
(followed over the control flow, through a guarded branch or ret); a masked load (a
default, then a guarded load into one register: one result); a variable never assigned,
or bound in an unsupported form (a nested tuple target); an annotated assignment, and the
kernel's file outside the roots or with two functions of its name. Load anchors: CuTe
DSL 4.4.2 line-only matching against a wrong .file index, by statement text and by
relpath:line, global loads only, one site per load (its outermost matching frame), every
match kept with a note when they sit on different lines, text matching no source line,
file-and-line matching refusing the wrong file, and the Triton list form (decision D2)
matching an inlined helper's frame, with two export roots. Bad annotations raise.
"""

import sys
import tempfile
import textwrap
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

from ...frontend.ptx import anchors, parse  # noqa: E402

# Line numbers matter: the PTX .locs below point into these files.
KERN = """\
import triton
import triton.language as tl
from helper import calc, lookup


@triton.jit
def plain(O_offsets, H):
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    begin = tl.load(O_offsets + off_z)
    tl.store(O_offsets, begin)


@triton.jit
def tiles(tb, th, tk, out, idx, n, LB: tl.constexpr):
    off_z, off_h, pid = (
        lookup(tb, th, tk, idx)
        if LB
        else calc(idx, n)
    )
    tl.store(out + off_z + pid, 0)


@triton.jit
def persistent(out, n, FIRST: tl.constexpr):
    if FIRST:
        pid = tl.program_id(0)
    else:
        pid = tl.program_id(1)
    while pid < n:
        tl.store(out + pid, 0)
        pid += tl.num_programs(0)


@triton.jit
def masked(x, n):
    v = tl.load(x, mask=n > 0, other=0)
    tl.store(x, v)


@triton.jit
def nested(x):
    for pid in range(3):
        pass
    a, (off_z, b) = 1, (2, 3)


@triton.jit
def split(t, n, H, out):
    s = t // n
    off_z = s // H
    pid = t % n
    tl.store(out + off_z + pid, 0)
"""

HELPER = """\
import triton
import triton.language as tl


@triton.jit
def lookup(tb, th, tk, i):
    off_z = tl.load(tb + i)
    off_h = tl.load(th + i)
    pid = tl.load(tk + i)
    return off_z, off_h, pid


@triton.jit
def calc(i, n):
    off_z = i // n
    return off_z, off_z, i % n
"""

# CuTe DSL sources: the PTX names kern_cute.py (its one .file) but the lines are
# seqlen.py's, as DSL 4.4.2 writes them.
SEQLEN = """\
class SeqlenInfo:
    def __init__(self, cu_seqlens, batch_idx):
        offset = 0 if cu_seqlens is None else cu_seqlens[batch_idx]
        self.offset = offset
        seqlen = cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]
"""
KERN_CUTE = """\
import cutlass


def kernel(self):
    pass
"""

LLVM_HEAD = """
.version 9.3
.target sm_100a
.address_size 64

"""

# As in the real preprocess kernels: an ld.param of another parameter under the
# expression's .loc, used elsewhere.
PLAIN = """
.visible .entry plain(
    .param .u64 .ptr .global .align 1 plain_param_0,
    .param .b32 plain_param_1,
    .param .b32 plain_param_2
)
.reqntid 128
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;
    .loc 1 7 0
$L__func_begin0:
    ld.param.u64 %rd1, [plain_param_0];
    .loc 1 8 14
    mov.u32 %r1, %ctaid.y;
    .loc 1 9 13
    ld.param.u32 %r2, [plain_param_1];
    div.s32 %r3, %r1, %r2;
    ld.param.u32 %r8, [plain_param_2];
    .loc 1 10 21
    mul.wide.s32 %rd2, %r3, 4;
    add.s64 %rd3, %rd1, %rd2;
    .loc 1 10 13
    ld.global.b32 %r4, [%rd3];
    .loc 1 11 5
    st.global.b32 [%rd1], %r4;
    st.global.b32 [%rd1+4], %r8;
    ret;
}
"""

# tiles compiled with LB: lookup inlined at the call in line 17 (column 9).
TILES_LB = """
.visible .entry tiles(
    .param .u64 .ptr .global .align 1 tiles_param_0,
    .param .u64 .ptr .global .align 1 tiles_param_1,
    .param .u64 .ptr .global .align 1 tiles_param_2,
    .param .u64 .ptr .global .align 1 tiles_param_3,
    .param .b32 tiles_param_4,
    .param .b32 tiles_param_5
)
.reqntid 128
{
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;
    .loc 1 15 0
$L__func_begin0:
    ld.param.u64 %rd1, [tiles_param_0];
    ld.param.u64 %rd2, [tiles_param_2];
    ld.param.u64 %rd3, [tiles_param_3];
    ld.param.u32 %r1, [tiles_param_4];
    .loc 2 7 21, function_name $L__info_string0, inlined_at 1 17 9
    mul.wide.s32 %rd4, %r1, 4;
    add.s64 %rd5, %rd1, %rd4;
    .loc 2 7 13, function_name $L__info_string0, inlined_at 1 17 9
    ld.global.b32 %r2, [%rd5];
    .loc 2 9 19, function_name $L__info_string0, inlined_at 1 17 9
    add.s64 %rd6, %rd2, %rd4;
    .loc 2 9 11, function_name $L__info_string0, inlined_at 1 17 9
    ld.global.b32 %r3, [%rd6];
    .loc 1 21 5
    add.s32 %r4, %r2, %r3;
    mul.wide.s32 %rd7, %r4, 4;
    add.s64 %rd8, %rd3, %rd7;
    st.global.b32 [%rd8], 0;
    ret;
}
"""

# tiles compiled without LB: calc inlined at the call in line 19 (column 14).
TILES_CALC = """
.visible .entry tiles(
    .param .u64 .ptr .global .align 1 tiles_param_0,
    .param .u64 .ptr .global .align 1 tiles_param_1,
    .param .u64 .ptr .global .align 1 tiles_param_2,
    .param .u64 .ptr .global .align 1 tiles_param_3,
    .param .b32 tiles_param_4,
    .param .b32 tiles_param_5
)
.reqntid 128
{
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;
    .loc 1 15 0
$L__func_begin0:
    ld.param.u64 %rd3, [tiles_param_3];
    ld.param.u32 %r1, [tiles_param_4];
    ld.param.u32 %r5, [tiles_param_5];
    .loc 2 15 13, function_name $L__info_string0, inlined_at 1 19 14
    div.s32 %r2, %r1, %r5;
    .loc 2 16 26, function_name $L__info_string0, inlined_at 1 19 14
    mul.lo.s32 %r6, %r2, %r5;
    sub.s32 %r3, %r1, %r6;
    .loc 1 21 5
    add.s32 %r4, %r2, %r3;
    mul.wide.s32 %rd7, %r4, 4;
    add.s64 %rd8, %rd3, %rd7;
    st.global.b32 [%rd8], 0;
    ret;
}
"""

# persistent compiled with FIRST: pid lives in %r1, written before the loop and by the
# AugAssign at its latch.
PERSISTENT = """
.visible .entry persistent(
    .param .u64 .ptr .global .align 1 persistent_param_0,
    .param .b32 persistent_param_1
)
.reqntid 128
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;
    .loc 1 25 0
$L__func_begin0:
    ld.param.u64 %rd1, [persistent_param_0];
    ld.param.u32 %r2, [persistent_param_1];
    .loc 1 27 15
    mov.u32 %r1, %ctaid.x;
    .loc 1 30 11
    setp.ge.s32 %p1, %r1, %r2;
    @%p1 bra $L__BB0_2;
$L__BB0_1:
    .loc 1 31 9
    mul.wide.s32 %rd2, %r1, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], 0;
    .loc 1 32 16
    mov.u32 %r3, %nctaid.x;
    .loc 1 32 9
    add.s32 %r1, %r1, %r3;
    .loc 1 30 11
    setp.lt.s32 %p2, %r1, %r2;
    @%p2 bra $L__BB0_1;
$L__BB0_2:
    ret;
}
"""

# A Triton masked load: a default value, then the load under the mask, into one register.
MASKED = """
.visible .entry masked(
    .param .u64 .ptr .global .align 1 masked_param_0,
    .param .b32 masked_param_1
)
.reqntid 128
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;
    .loc 1 36 0
$L__func_begin0:
    ld.param.u64 %rd1, [masked_param_0];
    ld.param.u32 %r1, [masked_param_1];
    .loc 1 37 25
    setp.gt.s32 %p1, %r1, 0;
    .loc 1 37 9
    // begin inline asm
    mov.u32 %r2, 0x0;
    @%p1 ld.global.b32 { %r2 }, [ %rd1 + 0 ];
    // end inline asm
    .loc 1 38 5
    st.global.b32 [%rd1], %r2;
    ret;
}
"""

NESTED = """
.visible .entry nested(
    .param .u64 .ptr .global .align 1 nested_param_0
)
.reqntid 128
{
    .loc 1 42 0
$L__func_begin0:
    ret;
}
"""

# LLVM computed pid's remainder under the store's .loc: pid has no instruction of its own.
SPLIT = """
.visible .entry split(
    .param .b32 split_param_0,
    .param .b32 split_param_1,
    .param .b32 split_param_2,
    .param .u64 .ptr .global .align 1 split_param_3
)
.reqntid 128
{
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;
    .loc 1 49 0
$L__func_begin0:
    ld.param.u32 %r1, [split_param_0];
    ld.param.u32 %r2, [split_param_1];
    ld.param.u32 %r3, [split_param_2];
    ld.param.u64 %rd1, [split_param_3];
    .loc 1 50 9
    div.s32 %r4, %r1, %r2;
    .loc 1 51 13
    div.s32 %r5, %r4, %r3;
    .loc 1 53 14
    rem.s32 %r6, %r1, %r2;
    add.s32 %r7, %r5, %r6;
    mul.wide.s32 %rd2, %r7, 4;
    add.s64 %rd3, %rd1, %rd2;
    .loc 1 53 5
    st.global.b32 [%rd3], 0;
    ret;
}
"""

# CuTe DSL 4.4.2 style: one .file (the kernel's), lines of other modules.
CUTE = """
.visible .entry kernel_cute_0(
    .param .u64 .ptr .global .align 1 kernel_cute_0_param_0
)
.reqntid 128
{
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;
    .loc 1 4 0
    ld.param.u64 %rd1, [kernel_cute_0_param_0];
    mov.u32 %r1, %ctaid.z;
    mul.wide.s32 %rd2, %r1, 4;
    add.s64 %rd3, %rd1, %rd2;
    .loc 1 3 58
    ld.global.u32 %r2, [%rd3];
    .loc 1 5 21
    ld.global.u32 %r3, [%rd3+4];
    sub.s32 %r4, %r3, %r2;
    st.global.u32 [%rd1], %r4;
    ret;
}
"""


class AnchorsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name).resolve()
        for name, text in (("kern.py", KERN), ("helper.py", HELPER),
                           ("cute/seqlen.py", SEQLEN), ("cute/kern_cute.py", KERN_CUTE)):  # fmt: skip
            (cls.root / name).parent.mkdir(exist_ok=True)
            (cls.root / name).write_text(text)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def fn(self, body, name, files=("kern.py", "helper.py"), root=None):
        table = "".join(
            f'.file {k} "{(root or self.root) / f}"\n' for k, f in enumerate(files, 1)
        )
        text = LLVM_HEAD + textwrap.dedent(body) + table
        # .debug_str names the inlined helpers' function_name (parse.py).
        text += ".section .debug_str\n{\n$L__info_string0:\n.b8 104,0\n}\n"
        return parse.parse(text).function(name)

    def resolve(self, work, fn, family="ttgir", mode="file_line"):
        return anchors.resolve(work, family, fn, self.root, mode)

    @staticmethod
    def text(fn, site):
        name, k, line = site.instr
        i = fn.instructions()[k]
        assert (name, i.line) == (fn.name, line), site
        return i.text

    # -- assignment anchors ---------------------------------------------------------------

    def test_plain_name_skips_ld_param(self):
        fn = self.fn(PLAIN, "plain")
        (site,), notes = self.resolve({"batch": "off_z"}, fn)
        self.assertEqual((site.field, site.kind), ("batch", "assign"))
        self.assertEqual(self.text(fn, site), "div.s32 %r3, %r1, %r2")
        self.assertEqual(site.detail, {"var": "off_z", "stmt": "kern.py:9", "element": None, "reg": "%r3"})  # fmt: skip
        self.assertEqual(notes, [])
        # A guarded rewrite of %r3 on the way does not end its value: still the result.
        guarded = PLAIN.replace(
            "mul.wide.s32 %rd2, %r3, 4;",
            "setp.eq.s32 %p1, %r1, 0;\n    @%p1 mov.u32 %r3, 0;\n    mul.wide.s32 %rd2, %r3, 4;",
        )
        fn = self.fn(guarded, "plain")
        (site,), _ = self.resolve({"batch": "off_z"}, fn)
        self.assertEqual(self.text(fn, site), "div.s32 %r3, %r1, %r2")
        # Columns are 1-based: an instruction at column 12, the space before the value
        # (ast col_offset 12), is not the value's, so the value does not give way to it.
        left = PLAIN.replace(
            "ld.param.u32 %r8, [plain_param_2];",
            ".loc 1 9 12\n    add.s32 %r8, %r3, 1;",
        )
        fn = self.fn(left, "plain")
        (site,), _ = self.resolve({"batch": "off_z"}, fn)
        self.assertEqual(self.text(fn, site), "div.s32 %r3, %r1, %r2")

    def test_stray_result_is_an_error(self):
        bad = PLAIN.replace(
            "div.s32 %r3, %r1, %r2;", "div.s32 %r3, %r1, %r2;\n    mov.u32 %r5, %tid.x;"
        ).replace("st.global.b32 [%rd1], %r4;", "add.s32 %r6, %r4, %r5;\n    st.global.b32 [%rd1], %r6;")  # fmt: skip
        with self.assertRaisesRegex(anchors.AnchorError, r"2 results .*-> %r3.*-> %r5"):
            self.resolve({"batch": "off_z"}, self.fn(bad, "plain"))
        # A guard is a read: a predicate of the region guarding a later store leaves it.
        bad = PLAIN.replace(
            "div.s32 %r3, %r1, %r2;", "div.s32 %r3, %r1, %r2;\n    setp.eq.s32 %p1, %r1, 0;"
        ).replace("st.global.b32 [%rd1+4], %r8;", "@%p1 st.global.b32 [%rd1+4], %r8;")  # fmt: skip
        with self.assertRaisesRegex(anchors.AnchorError, r"2 results .*-> %r3.*-> %p1"):
            self.resolve({"batch": "off_z"}, self.fn(bad, "plain"))

    def test_tuple_from_helper(self):
        fn = self.fn(TILES_LB, "tiles")
        sites, notes = self.resolve({"batch": "off_z", "block": "pid"}, fn)
        got = {
            s.field: (self.text(fn, s), s.detail["reg"], s.detail["element"])
            for s in sites
        }
        # The scaled index %rd4 also feeds pid's load, but off_z's load is computed from it.
        self.assertEqual(got, {"batch": ("ld.global.b32 %r2, [%rd5]", "%r2", 0),
                               "block": ("ld.global.b32 %r3, [%rd6]", "%r3", 2)})  # fmt: skip
        self.assertEqual(notes, [])
        # off_h is unused, so LLVM dropped its load. Its statement still runs: with the other
        # fields asked too, a "novalue" site at the last of their sites.
        sites, notes = self.resolve(
            {"batch": "off_z", "head": "off_h", "block": "pid"}, fn
        )
        (dead,) = [s for s in sites if s.kind == "novalue"]
        self.assertEqual(
            (dead.field, self.text(fn, dead)), ("head", "ld.global.b32 %r3, [%rd6]")
        )
        self.assertEqual(dead.detail, {"var": "off_h", "stmt": "kern.py:17", "element": 1, "reg": None})  # fmt: skip
        self.assertEqual(notes, [])
        # Alone it has no site at all: the field resolves nowhere.
        with self.assertRaisesRegex(
            anchors.AnchorError,
            r"'head'.*resolves to no instruction.*off_h at kern.py:17, element 1.*unused",
        ):
            self.resolve({"head": "off_h"}, fn)

    def test_tie_goes_to_the_root_expression(self):
        # The index sign-extended under the address subexpression's .loc (7:21) and used
        # outside too: two independent results; the load sits at the root, tl.load (7:13).
        text = TILES_LB.replace(
            "mul.wide.s32 %rd4, %r1, 4;", "mul.wide.s32 %rd4, %r1, 4;\n    cvt.s64.s32 %rd9, %r1;"
        ).replace("st.global.b32 [%rd8], 0;", "add.s64 %rd10, %rd8, %rd9;\n    st.global.b32 [%rd10], 0;")  # fmt: skip
        fn = self.fn(text, "tiles")
        (site,), _ = self.resolve({"batch": "off_z"}, fn)
        self.assertEqual(self.text(fn, site), "ld.global.b32 %r2, [%rd5]")

    def test_tuple_other_branch(self):
        fn = self.fn(TILES_CALC, "tiles")
        sites, notes = self.resolve(
            {"batch": "off_z", "head": "off_h", "block": "pid"}, fn
        )
        got = {s.field: self.text(fn, s) for s in sites}
        # Elements 0 and 1 are calc's off_z; pid's remainder reads the division.
        self.assertEqual(got, {"batch": "div.s32 %r2, %r1, %r5", "head": "div.s32 %r2, %r1, %r5",
                               "block": "sub.s32 %r3, %r1, %r6"})  # fmt: skip
        self.assertEqual(notes, [])
        # The remainder's product also used outside, at the same (root) position: it is
        # computed into the sub, so it gives way.
        shared = TILES_CALC.replace(
            "add.s32 %r4, %r2, %r3;",
            "add.s32 %r7, %r2, %r3;\n    add.s32 %r4, %r7, %r6;",
        )
        fn = self.fn(shared, "tiles")
        (site,), _ = self.resolve({"block": "pid"}, fn)
        self.assertEqual(self.text(fn, site), "sub.s32 %r3, %r1, %r6")

    def test_dropped_branch_and_loop_carried_augassign(self):
        fn = self.fn(PERSISTENT, "persistent")
        sites, notes = self.resolve({"block": "pid"}, fn)
        self.assertEqual(
            [(self.text(fn, s), s.detail["stmt"]) for s in sites],
            [
                ("mov.u32 %r1, %ctaid.x", "kern.py:27"),
                ("add.s32 %r1, %r1, %r3", "kern.py:32"),
            ],
        )
        (note,) = notes
        self.assertIn("pid at kern.py:29", note)
        self.assertIn("no instruction", note)
        # pid's first value read only past a guarded branch, or a guarded ret: both fall
        # through, so it still leaves its code.
        for exit_ in ("@%p1 bra $L__BB0_2;", "@%p1 ret;"):
            text = PERSISTENT.replace(
                "setp.ge.s32 %p1, %r1, %r2;\n    @%p1 bra $L__BB0_2;",
                f"setp.ge.s32 %p1, %r2, 0;\n    {exit_}",
            )
            fn = self.fn(text, "persistent")
            sites, _ = self.resolve({"block": "pid"}, fn)
            self.assertEqual(
                [self.text(fn, s) for s in sites],
                ["mov.u32 %r1, %ctaid.x", "add.s32 %r1, %r1, %r3"],
                exit_,
            )

    def test_statement_without_instruction_in_a_live_block(self):
        fn = self.fn(SPLIT, "split")
        sites, notes = self.resolve({"batch": "off_z", "block": "pid"}, fn)
        self.assertEqual(
            [(s.field, s.kind, self.text(fn, s), s.detail["reg"]) for s in sites],
            [("batch", "assign", "div.s32 %r5, %r4, %r3", "%r5"),
             ("block", "novalue", "div.s32 %r5, %r4, %r3", None)],
        )  # fmt: skip
        self.assertEqual(notes, [])
        with self.assertRaisesRegex(
            anchors.AnchorError, "'block'.*pid at kern.py:52.*no instruction"
        ):
            self.resolve({"block": "pid"}, fn)

    def test_masked_load_is_one_result(self):
        fn = self.fn(MASKED, "masked")
        (site,), _ = self.resolve({"v": "v"}, fn)
        self.assertEqual(
            self.text(fn, site), "@%p1 ld.global.b32 { %r2 }, [ %rd1 + 0 ]"
        )

    def test_unassigned_or_unsupported(self):
        with self.assertRaisesRegex(
            anchors.AnchorError, "off_q is not assigned in plain"
        ):
            self.resolve({"batch": "off_q"}, self.fn(PLAIN, "plain"))
        fn = self.fn(NESTED, "nested")
        with self.assertRaisesRegex(
            anchors.AnchorError, "off_z is bound at line 45 in a form"
        ):
            self.resolve({"batch": "off_z"}, fn)
        # So is a plain name of that nested target: the target is not flat.
        with self.assertRaisesRegex(
            anchors.AnchorError, "a is bound at line 45 in a form"
        ):
            self.resolve({"batch": "a"}, fn)
        # A for-loop target is no occurrence (a block argument, never a NameLoc op).
        with self.assertRaisesRegex(
            anchors.AnchorError, "pid is not assigned in nested"
        ):
            self.resolve({"block": "pid"}, fn)

    def test_kernel_lookup_and_annotated_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            # An annotated assignment is an occurrence; its value starts at column 18.
            (root / "kern.py").write_text(
                KERN.replace("    off_z = off_hz // H", "    off_z: int = off_hz // H")
            )
            fn = self.fn(
                PLAIN.replace(".loc 1 9 13", ".loc 1 9 18"), "plain", root=root
            )
            (site,), _ = anchors.resolve(
                {"batch": "off_z"}, "ttgir", fn, root, "file_line"
            )
            self.assertEqual(self.text(fn, site), "div.s32 %r3, %r1, %r2")
            # The kernel's file outside every root, or two functions of its name in it
            # (another file name: sources are cached by path): case errors.
            with self.assertRaisesRegex(anchors.AnchorError, "are not in a file under"):
                anchors.resolve({"batch": "off_z"}, "ttgir", fn, self.root, "file_line")
            (root / "twice.py").write_text(KERN + "\n\ndef plain():\n    pass\n")
            fn = self.fn(PLAIN, "plain", files=("twice.py",), root=root)
            with self.assertRaisesRegex(
                anchors.AnchorError, "2 functions plain in twice.py"
            ):
                anchors.resolve({"batch": "off_z"}, "ttgir", fn, root, "file_line")

    # -- load anchors ---------------------------------------------------------------------

    def cute(self):
        return self.fn(CUTE, "kernel_cute_0", files=("cute/kern_cute.py",))

    def test_line_only_statement_text(self):
        fn = self.cute()
        where = "offset = 0 if cu_seqlens is None else cu_seqlens[batch_idx]"
        work = {"batch": [{"load": "mCuSeqlensQ", "where": where}]}
        (site,), notes = self.resolve(work, fn, "cute", "line_only")
        self.assertEqual(
            (site.kind, self.text(fn, site)), ("load", "ld.global.u32 %r2, [%rd3]")
        )
        self.assertEqual(site.detail, {"tensor": "mCuSeqlensQ", "where": where,
                                       "frame": "cute/kern_cute.py:3", "source": ["cute/seqlen.py:3"]})  # fmt: skip
        self.assertEqual(notes, [])
        # Comparing files too finds nothing: the .file index is the kernel's. For CuTe that
        # is a note, as in the legacy frontend.
        sites, (note,) = self.resolve(work, fn, "cute", "file_line")
        self.assertEqual(sites, [])
        self.assertIn("no global load", note)
        # Only global loads are candidates: the ld.param on line 4 is no load of
        # seqlen.py:4's statement.
        work = {"batch": [{"load": "T", "where": "self.offset = offset"}]}
        sites, (note,) = self.resolve(work, fn, "cute", "line_only")
        self.assertEqual(sites, [])
        self.assertIn("no global load", note)

    def test_line_only_relpath_and_several_lines(self):
        fn = self.cute()
        (site,), _ = self.resolve(
            {"batch": [{"load": "T", "where": "seqlen.py:5"}]}, fn, "cute", "line_only"
        )
        self.assertEqual(self.text(fn, site), "ld.global.u32 %r3, [%rd3+4]")
        sites, (note,) = self.resolve(
            {"batch": [{"load": "T", "where": "cu_seqlens[batch_idx]"}]},
            fn,
            "cute",
            "line_only",
        )
        self.assertEqual([self.text(fn, s).split()[1] for s in sites], ["%r2,", "%r3,"])
        self.assertIn("different lines [3, 5]", note)
        # A load is one site per entry, at its outermost matching frame, even when an
        # inner (inlined) frame matches too.
        inlined = CUTE.replace(
            ".loc 1 5 21",
            ".loc 1 5 21, function_name $L__info_string0, inlined_at 1 3 9",
        )
        fn2 = self.fn(inlined, "kernel_cute_0", files=("cute/kern_cute.py",))
        sites, notes = self.resolve(
            {"batch": [{"load": "T", "where": "cu_seqlens[batch_idx]"}]},
            fn2,
            "cute",
            "line_only",
        )
        self.assertEqual(
            [self.text(fn2, s).split()[1] for s in sites], ["%r2,", "%r3,"]
        )
        self.assertEqual(notes, [])
        with self.assertRaisesRegex(anchors.AnchorError, "matches no source line"):
            self.resolve(
                {"batch": [{"load": "T", "where": "nowhere()"}]},
                fn,
                "cute",
                "line_only",
            )

    def test_triton_load_form(self):
        fn = self.fn(TILES_LB, "tiles")
        work = {"block": [{"load": "tk", "where": "tl.load(tk + i)"}]}
        with tempfile.TemporaryDirectory() as other:  # a second root, without the line
            (Path(other) / "x.py").write_text("tl.load(tk)\n")
            (site,), _ = anchors.resolve(
                work, "ttgir", fn, [other, self.root], "file_line"
            )
        self.assertEqual(self.text(fn, site), "ld.global.b32 %r3, [%rd6]")
        self.assertEqual(site.detail["frame"], "helper.py:9")
        # For Triton a field without any site is a case error, as in the TTGIR frontend.
        with self.assertRaisesRegex(anchors.AnchorError, "resolves to no instruction"):
            self.resolve({"block": [{"load": "tk", "where": "pid < n"}]}, fn)

    def test_bad_annotations(self):
        fn = self.cute()
        self.assertEqual(self.resolve(None, fn, "cute", "line_only"), ([], []))
        for work in ({"batch": "off_z"}, {"batch": []}, {"batch": [{"load": "T"}]}):
            with self.assertRaises(anchors.AnchorError):
                self.resolve(work, fn, "cute", "line_only")
        with self.assertRaises(ValueError):
            self.resolve({}, fn, "tlx", "file_line")
        with self.assertRaises(ValueError):
            self.resolve(
                {"batch": "off_z"}, self.fn(PLAIN, "plain"), "ttgir", "line_only"
            )


if __name__ == "__main__":
    unittest.main()
