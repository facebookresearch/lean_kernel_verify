# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the PTX core's thread model and value semantics (frontend/ptx/interp.py,
expr.py), through translate() on hand-written kernels.

    python3 -I tests/ptx/test_interp.py      (or the buck target :test_interp)

Covers: the PTX integer expansions against brute force (mul.hi, bfe, bfi, prmt, lop3,
shf, popc, clz, brev, cvt, div_exact), the test of one bit of a packed predicate mask,
the constant added to a shifted value LLVM shrank taken back to the closest to 0;
per-thread events (Triton: one event over the threads, a replicated load over one thread
per class; CuTe: a thread loop), a uniform predicate as a KIR `if`, a lane-varying one as
a mask; elect.sync and holes H1-H3 (an elect-guarded event issued once per warp, the
elected lane under another lane condition refused, an operand depending on the lane
refused, an elect symbol carried in a per-thread variable refused, a full-mask shfl on
part of a warp refused, on a runtime part asserted); shfl/vote; a divergent branch
merging by select, a uniform one as `if`; a loop as `while` with carried slots; partition
refinement (CuTe roles warps<a>_<b>) against a mask (Triton); a predicated metadata load
kept inside its select arm (R1); special registers; %warpid refused. Review: values
(fixtures evaluate()) of an If joining at the loop's back edge and of a uniform If's slot
whose branch holds a loop; a value of the elected lane as an event operand or a branch
refused; a loop exit under a lane condition nested in a uniform If refused; a masked lane
table loading on no masked-off lane; select arms' wraps, loads and divisions dead (R1).
"""

import json
import random
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

from ...frontend.ptx import expr as E  # noqa: E402
from .fixtures.kernels import (  # noqa: E402
    evaluate,
    events,
    kir_errors,
    run,
    statements,
)

PRE = """
    mov.u32 %r1, %tid.x;
    ld.param.b32 %r2, [k_param_1];
    ld.param.u64 %rd1, [k_param_0];
"""


def c32(v):
    return E.const(v, 32)


def u32(e):
    return e[2] & 0xFFFFFFFF


class Expansions(unittest.TestCase):
    """Each expansion of expr.py folds on constants to the ISA's value."""

    def setUp(self):
        self.rng = random.Random(7)

    def vals(self, n=300):
        edge = [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x12345678]
        return edge + [self.rng.getrandbits(32) for _ in range(n)]

    def test_mul_hi_and_wide(self):
        for a in self.vals(100):
            for b in self.vals(3):
                sa, sb = a - (a >> 31 << 32), b - (b >> 31 << 32)
                self.assertEqual(u32(E.mul_hi(c32(a), c32(b), False)), (a * b) >> 32)
                self.assertEqual(
                    u32(E.mul_hi(c32(a), c32(b), True)), ((sa * sb) >> 32) & 0xFFFFFFFF
                )
                w = E.mul_wide(c32(a), c32(b), True)
                self.assertEqual(w[2] & (2**64 - 1), (sa * sb) & (2**64 - 1))

    def test_bfe_bfi(self):
        for a in self.vals(60):
            for pos in (0, 3, 31, 33, 255):
                for ln in (0, 1, 5, 32, 40):
                    bits = [
                        (a >> min(pos + i, 31)) & 1 if pos + i <= 31 else None
                        for i in range(32)
                    ]
                    u = sum(b << i for i, b in enumerate(bits[:ln]) if b)
                    self.assertEqual(u32(E.bfe(c32(a), pos, ln, False)), u)
                    sbit = 0 if ln == 0 else (a >> min(pos + ln - 1, 31)) & 1
                    s = sum(
                        (bits[i] if i < ln and bits[i] is not None else sbit) << i
                        for i in range(32)
                    )
                    self.assertEqual(u32(E.bfe(c32(a), pos, ln, True)), s)
                    b = self.rng.getrandbits(32)
                    want = b
                    for i in range(ln):
                        if pos + i <= 31:
                            want = want & ~(1 << (pos + i)) | (
                                ((a >> i) & 1) << (pos + i)
                            )
                    self.assertEqual(
                        u32(E.bfi(c32(a), c32(b), pos, ln)), want & 0xFFFFFFFF
                    )

    def test_prmt_lop3_shf(self):
        for a in self.vals(40):
            b, c = self.rng.getrandbits(32), self.rng.getrandbits(32)
            sel = self.rng.getrandbits(16)
            src = list(a.to_bytes(4, "little") + b.to_bytes(4, "little"))
            want = 0
            for i in range(4):
                s = (sel >> 4 * i) & 15
                byte = src[s & 7]
                if s & 8:
                    byte = 0xFF if byte & 0x80 else 0
                want |= byte << 8 * i
            self.assertEqual(u32(E.prmt(c32(a), c32(b), sel)), want)
            lut = self.rng.getrandbits(8)
            want = sum(
                ((lut >> ((a >> i & 1) * 4 + (b >> i & 1) * 2 + (c >> i & 1))) & 1) << i
                for i in range(32)
            )
            self.assertEqual(u32(E.lop3(c32(a), c32(b), c32(c), lut)), want)
            for n in (0, 1, 13, 31, 32, 40):
                wide = (b << 32) | a
                for left in (True, False):
                    for wrap in (True, False):
                        k = n & 31 if wrap else min(n, 32)
                        want = ((wide << k) >> 32) if left else (wide >> k)
                        self.assertEqual(
                            u32(E.shf(c32(a), c32(b), n, left, wrap)), want & 0xFFFFFFFF
                        )

    def test_bit_counts_and_cvt(self):
        for a in self.vals(60):
            self.assertEqual(u32(E.popc(c32(a))), bin(a).count("1"))
            self.assertEqual(u32(E.clz(c32(a))), 32 - a.bit_length())
            self.assertEqual(u32(E.brev(c32(a))), int(f"{a:032b}"[::-1], 2))
            self.assertEqual(
                u32(E.cvt_int(c32(a), 32, 16, True)),
                ((a & 0xFFFF) ^ 0x8000) - 0x8000 & 0xFFFFFFFF,
            )
            self.assertEqual(E.cvt_int(c32(a), 64, 32, False)[2], a)

    def test_div_exact(self):
        r = E.var("r", 32)
        off = E.bin(
            "add", E.bin("mul", E.ext(r, 64, True), E.const(4, 64)), E.const(16, 64)
        )
        self.assertIs(
            E.div_exact(off, 4), E.bin("add", E.ext(r, 64, True), E.const(4, 64))
        )
        x = E.var("x", 64)  # unbounded: the byte arithmetic stays, divided exactly
        self.assertEqual(E.div_exact(E.bin("shl", x, E.const(1, 64)), 2)[2], "sdiv")
        self.assertIsNone(E.div_exact(E.bin("add", x, E.const(1, 64)), 2))


class TableLowering(unittest.TestCase):
    def test_selects_over_distinct_entries(self):
        # entries[key] for every key in range: one select per entry besides the most
        # frequent, on `key urem p == r` for every p-th key from r, else key == i or'd.
        key = E.bin("urem", E.var("t", 32), c32(32))
        v, w, z = E.var("v", 32), E.var("w", 32), c32(0)
        for entries in ([v, z, z, z] * 8, [z] * 31 + [v], [v if i in (1, 5, 6) else z for i in range(32)],
                        [c32(3 * i) for i in range(32)], [v, w] * 16):  # fmt: skip
            got = E._table_select(key, tuple(entries))
            for k in range(32):
                self.assertIs(E.subst(got, {"t": c32(k)}), entries[k])
        lowered = E.to_json(E.table(key, tuple([v, z, z, z] * 8)))
        self.assertEqual(json.dumps(lowered).count('"select"'), 1)


class PackedPredicates(unittest.TestCase):
    def test_one_bit_of_a_packed_mask(self):
        # LLVM packs predicates into an i16 mask (selp of -1/1, shl, and 0xffff, disjoint
        # adds) and tests one bit per predicated load: the test folds to that predicate.
        c = [E.cmp("ult", E.var(f"x{i}", 32), E.var("n", 32)) for i in range(3)]

        def sel(p, v):
            return E.select(p, c32(v), c32(0))

        m = E.bin("and", E.bin("shl", sel(c[0], 0xFFFF), c32(3)), c32(0xFFFF))
        m = E.bin("add", m, E.bin("shl", sel(c[1], 1), c32(2)))
        m = E.bin("add", m, sel(c[2], 1))  # bits 3..15: c0, bit 2: c1, bit 0: c2

        def bit(k):
            return E.cmp("ne", E.bin("and", E.bin("lshr", m, c32(k)), c32(1)), c32(0))

        self.assertIs(bit(15), c[0])
        self.assertIs(bit(2), c[1])
        self.assertIs(bit(0), c[2])
        self.assertIs(bit(1), E.const(0, 1))
        self.assertIs(E.cmp("eq", E.bin("lshr", m, c32(15)), c32(0)), E.not1(c[0]))
        self.assertEqual(E.cmp("ne", m, c32(0))[0], "cmp")  # several bits: no fold

    def test_or_below_trailing_zeros(self):
        # (128 z + 32) | 64 = 128 z + 96: 32 and 64 lie below z's 7 trailing zeros
        z = E.bin("shl", E.var("z", 32), c32(7))
        x = E.bin("add", z, c32(32))
        self.assertIs(E.bin("or", x, c32(64)), E.bin("add", z, c32(96)))
        self.assertEqual(E.bin("or", x, c32(32))[2], "or")  # overlapping bits
        self.assertEqual(E.bin("or", x, c32(128))[2], "or")  # not below the zeros


class ShrunkConstants(unittest.TestCase):
    def test_shift_takes_the_constant_closest_to_zero(self):
        # LLVM writes (n - 2 - i) << 7 as (n + 0x1FFFFFE - i) << 7 (bf16_fwd_2cta PTX
        # line 315): the same value, with the constant whose wrap P0-1 would report
        n, i, k = E.var("n", 32), E.var("i", 32), c32(7)
        got = E.bin("shl", E.bin("sub", E.bin("add", n, c32(0x1FFFFFE)), i), k)
        self.assertIs(got, E.bin("shl", E.bin("sub", E.bin("add", n, c32(-2)), i), k))
        self.assertIs(  # in either operand, also of a sub
            E.bin("shl", E.bin("sub", i, E.bin("add", n, c32(0x1000001))), k),
            E.bin("shl", E.bin("sub", i, E.bin("add", n, c32(-0xFFFFFF))), k),
        )
        for c in (0xFFFFFF, -0x1000000, 5):  # already the closest: unchanged
            e = E.bin("shl", E.bin("add", n, c32(c)), k)
            self.assertIs(e[3][4], c32(c))
        for a in (0, 1, 0x7FFFFFFF, 0xFFFFFFFF, 0x12345678):
            for c in (0x1FFFFFE, 0x1000000, -0x7000000):
                got = E.bin("shl", E.bin("add", n, c32(c)), k)
                self.assertEqual(
                    E.subst(got, {"n": c32(a)})[2] & 0xFFFFFFFF,
                    ((a + c) << 7) & 0xFFFFFFFF,
                )


class LoopInvariants(unittest.TestCase):
    def test_bound_before_the_loop(self):
        # n * 12 cannot change in the loop: its let precedes the `while` (Lean evaluates
        # the body every trip); the load x[0] may fail, so it stays where the body uses it.
        f = run(
            PRE
            + """
    setp.eq.s32 %p9, %r1, 0;
    mul.lo.s32 %r3, %r2, 12;
    ld.global.b32 %r5, [%rd1];
    mov.b32 %r21, 0;
LOOP:
    add.s32 %r4, %r3, %r21;
    add.s32 %r6, %r4, %r5;
    mul.wide.s32 %rd2, %r6, 4;
    add.s64 %rd3, %rd1, %rd2;
    @%p9 st.global.b32 [%rd3], %r21;
    add.s32 %r21, %r21, 1;
    setp.lt.s32 %p13, %r21, %r2;
    @%p13 bra LOOP;
    ret;
"""
        )
        self.assertEqual(f.refusals, [])
        body = f.roles[0]["body"]
        (loop,) = [s for s in body if s["op"] == "while"]
        inside = [s["expr"] for s in statements(loop["body"]) if s["op"] == "let"]
        before = [s["expr"] for s in body if s["op"] == "let"]

        def twelve(es):
            return any(e.get("o") == "mul" and e["b"].get("v") == 12 for e in es)

        self.assertTrue(twelve(before))
        self.assertFalse(twelve(inside))
        self.assertTrue(any(e["e"] == "load" for e in inside))
        self.assertEqual(kir_errors(f), [])

    def test_not_bound_before_the_loop_from_a_branch(self):
        # The same n * 12 used only under `if n > 5` in the body stays in that branch:
        # Lean evaluates it (and its wrap check) only when the branch is taken (R1).
        f = run(
            PRE
            + """
    setp.eq.s32 %p9, %r1, 0;
    setp.gt.s32 %p2, %r2, 5;
    mul.lo.s32 %r3, %r2, 12;
    mov.b32 %r21, 0;
LOOP:
    @!%p2 bra SKIP;
    add.s32 %r4, %r3, %r21;
    mul.wide.s32 %rd2, %r4, 4;
    add.s64 %rd3, %rd1, %rd2;
    @%p9 st.global.b32 [%rd3], %r21;
SKIP:
    add.s32 %r21, %r21, 1;
    setp.lt.s32 %p13, %r21, %r2;
    @%p13 bra LOOP;
    ret;
"""
        )
        self.assertEqual(f.refusals, [])
        body = f.roles[0]["body"]
        (loop,) = [s for s in body if s["op"] == "while"]
        (arm,) = [a for s in loop["body"] if s["op"] == "if" for a in (s["then"], s["else"])
                  if any(x["op"] == "event" for x in a)]  # fmt: skip
        twelve = lambda ss: any(  # noqa: E731
            s["op"] == "let"
            and s["expr"].get("o") == "mul"
            and s["expr"]["b"].get("v") == 12
            for s in ss
        )
        self.assertFalse(twelve(body))
        self.assertTrue(twelve(arm))
        self.assertEqual(kir_errors(f), [])


class Threads(unittest.TestCase):
    def test_masked_store_per_family(self):
        body = (
            PRE
            + """
    setp.lt.s32 %p1, %r1, %r2;
    mul.wide.s32 %rd2, %r1, 4;
    add.s64 %rd3, %rd1, %rd2;
    mov.b32 %r3, 0;
    @%p1 st.global.b32 [%rd3], %r3;
    ret;
"""
        )
        tlx = run(body)
        (ev,) = events(tlx, "gstore")
        self.assertEqual(ev["offsets"]["shape"], [128])  # one event over the threads
        self.assertEqual(ev["mask"]["t"], "cmp")
        cute = run(body, family="cute")
        loops = [s for s in cute.roles[0]["body"] if s["op"] == "for"]
        self.assertEqual(len(loops), 1)  # a thread loop with the thread's mask as `if`
        self.assertEqual(loops[0]["ub"]["v"], 128)
        self.assertEqual(loops[0]["body"][-1]["op"], "if")
        self.assertEqual(kir_errors(tlx) + kir_errors(cute), [])

    def test_replicated_load_once_per_thread_class(self):
        # Each warpgroup half reads all of x[n + 0..1] (thread t: x[n + t/64 % 2], under
        # n > 0): the Triton event is over one thread per class, threads 0 and 64; the
        # elements, all P0-1 sees, are those of the 128 loads. x[t] keeps 128 threads.
        f = run(
            PRE
            + """
    shr.u32 %r3, %r1, 6;
    and.b32 %r4, %r3, 1;
    add.s32 %r5, %r4, %r2;
    mul.wide.s32 %rd2, %r5, 4;
    add.s64 %rd3, %rd1, %rd2;
    setp.gt.s32 %p1, %r2, 0;
    @%p1 ld.global.b32 %r6, [%rd3];
    mul.wide.u32 %rd4, %r1, 4;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.b32 %r7, [%rd5];
    add.s32 %r8, %r6, %r7;
    st.global.b32 [%rd5+1024], %r8;
    ret;
"""
        )
        self.assertEqual(f.refusals, [])
        rep, own = events(f, "gload")
        self.assertEqual(rep["offsets"]["shape"], [2])
        self.assertEqual(own["offsets"]["shape"], [128])
        loads = [o for e, _t, o in evaluate(f, {"n": 3}) if e == "gload"]
        self.assertEqual([sorted(o) for o in loads], [[3, 4], list(range(128))])
        self.assertEqual(kir_errors(f), [])

    def test_uniform_predicate_is_an_if(self):
        f = run(
            PRE
            + """
    setp.gt.s32 %p1, %r2, 0;
    setp.eq.s32 %p2, %r1, 0;
    and.pred %p3, %p1, %p2;
    @%p3 st.global.b32 [%rd1], %r2;
    ret;
"""
        )
        (s,) = [s for s in f.roles[0]["body"] if s["op"] != "let"]
        self.assertEqual(s["op"], "if")  # n > 0, then thread 0's single store
        self.assertEqual(s["then"][0]["offsets"]["shape"], [1])

    def test_elect_issues_once_per_warp(self):
        f = run(
            PRE
            + """
    elect.sync %r3|%p1, -1;
    shr.u32 %r4, %r1, 5;
    setp.eq.s32 %p2, %r4, 2;
    and.pred %p3, %p2, %p1;
    @%p3 st.global.b32 [%rd1], %r2;
    @%p1 red.global.add.u32 [%rd1+4], 1;
    ret;
"""
        )
        st, red = events(f)
        self.assertEqual(st["offsets"]["shape"], [1])  # warp 2's elected lane only
        self.assertEqual(red["ev"], "gatomic")
        self.assertEqual(
            red["offsets"]["shape"], [128]
        )  # one per warp: 4 lanes masked in
        self.assertEqual(kir_errors(f), [])

    def test_h1_elect_with_a_lane_condition(self):
        f = run(
            PRE
            + """
    elect.sync %r3|%p1, -1;
    setp.lt.u32 %p2, %r1, 16;
    and.pred %p3, %p2, %p1;
    @%p3 st.global.b32 [%rd1], %r2;
    ret;
"""
        )
        self.assertIn("depends on the lane", f.refusals[0]["msg"])
        f = run(
            PRE
            + """
    elect.sync %r3|%p1, -1;
    mov.u32 %r4, %laneid;
    mul.wide.u32 %rd2, %r4, 4;
    add.s64 %rd3, %rd1, %rd2;
    @%p1 st.global.b32 [%rd3], %r2;
    ret;
"""
        )
        self.assertIn("depends on which lane is elected", f.refusals[0]["msg"])

    def test_h2_elect_not_concretized_in_a_slot(self):
        f = run(
            PRE
            + """
    elect.sync %r3|%p1, -1;
    selp.b32 %r4, 1, 0, %p1;
    mov.b32 %r5, 0;
    mov.b32 %r6, 0;
LOOP:
    add.s32 %r5, %r5, %r4;
    add.s32 %r6, %r6, 1;
    setp.lt.s32 %p2, %r6, %r2;
    @%p2 bra LOOP;
    mul.wide.s32 %rd2, %r5, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r2;
    ret;
"""
        )
        self.assertIn("elect.sync picks is carried", f.refusals[0]["msg"])

    def test_h3_collective_on_part_of_a_warp(self):
        f = run(
            PRE
            + """
    mov.u32 %r4, %laneid;
    setp.lt.u32 %p1, %r4, 16;
    @!%p1 bra SKIP;
    shfl.sync.idx.b32 %r5, %r1, 0, 31, -1;
SKIP:
    ret;
"""
        )
        self.assertIn("only part of warp(s)", f.refusals[0]["msg"])
        f = run(
            PRE
            + """
    setp.lt.s32 %p1, %r1, %r2;
    @!%p1 bra SKIP;
    vote.sync.ballot.b32 %r5, %p1, -1;
SKIP:
    ret;
"""
        )
        (a,) = [s for s in statements(f.roles[0]["body"]) if s["op"] == "assert"]
        self.assertIn("all lanes of a warp or none", a["msg"])

    def test_h3_runtime_check_per_thread_under_cute(self):
        # Decision D8: the CuTe policy asserts per thread that its lane agrees with lane
        # 0 of its warp (one assert in a thread loop, not one conjunct per warp); n
        # splitting a warp fails it, whole warps (or none) pass.
        f = run(
            PRE
            + """
    setp.lt.s32 %p1, %r1, %r2;
    @!%p1 bra SKIP;
    vote.sync.ballot.b32 %r5, %p1, -1;
SKIP:
    ret;
""",
            family="cute",
        )
        self.assertEqual(f.refusals, [])
        (loop,) = [s for s in f.roles[0]["body"] if s["op"] == "for"]
        self.assertEqual(loop["body"][-1]["op"], "assert")
        self.assertEqual(kir_errors(f), [])
        for n, fails in ((64, False), (40, True), (0, False), (127, True)):
            got = [t for t in evaluate(f, {"n": n}) if t[0] == "assert"]
            self.assertEqual(bool(got), fails, n)

    def test_shfl_vote_popc(self):
        f = run(
            PRE
            + """
    shr.u32 %r3, %r1, 5;
    shfl.sync.idx.b32 %r4, %r3, 0, 31, -1;
    shfl.sync.down.b32 %r5, %r1, 1, 31, -1;
    setp.lt.s32 %p1, %r5, %r2;
    vote.sync.ballot.b32 %r6, %p1, -1;
    popc.b32 %r7, %r6;
    add.s32 %r8, %r7, %r4;
    mul.wide.s32 %rd2, %r8, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r2;
    ret;
"""
        )
        self.assertEqual(f.refusals, [])
        # The ballot reads every lane's shuffled value (a table): per-thread events.
        (loop,) = [s for s in f.roles[0]["body"] if s["op"] == "for"]
        self.assertEqual(loop["body"][-1]["ev"], "gstore")
        self.assertEqual(kir_errors(f), [])

    def test_branches(self):
        f = run(
            PRE
            + """
    setp.lt.s32 %p1, %r1, 64;
    mov.b32 %r3, 1;
    @%p1 bra A;
    mov.b32 %r3, 2;
A:
    setp.gt.s32 %p2, %r2, 5;
    @%p2 bra B;
    add.s32 %r3, %r3, %r2;
    st.global.b32 [%rd1+8], %r2;
B:
    mul.wide.s32 %rd2, %r3, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r2;
    ret;
"""
        )
        body = f.roles[0]["body"]
        ifs = [s for s in body if s["op"] == "if"]
        self.assertEqual(len(ifs), 1)  # n > 5 is uniform: a KIR if; tid < 64 a select
        self.assertEqual(kir_errors(f), [])

    def test_loop_is_a_while(self):
        f = run(
            PRE
            + """
    mov.b32 %r3, 0;
    mov.b32 %r4, %r1;
LOOP:
    add.s32 %r4, %r4, 128;
    add.s32 %r3, %r3, 1;
    setp.lt.s32 %p1, %r3, %r2;
    @%p1 bra LOOP;
    mul.wide.s32 %rd2, %r4, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r2;
    ret;
""",
            max_iters=77,
        )
        (w,) = [s for s in f.roles[0]["body"] if s["op"] == "while"]
        self.assertEqual(w["max_iters"], 77)
        self.assertEqual(kir_errors(f), [])

    def test_counted_loop_is_a_for(self):
        body = (
            PRE
            + """
    mov.b32 %r3, 0;
    mov.b32 %r5, 0;
LOOP:
    add.s32 %r5, %r5, %r1;
    add.s32 %r3, %r3, 2;
    setp.lt.s32 %p1, %r3, UB;
    @%p1 bra LOOP;
    mul.wide.s32 %rd2, %r5, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3+%s], %r3;
    ret;
"""
        )
        f = run(body.replace("UB", "7").replace("+%s", ""))
        loops = [s for s in f.roles[0]["body"] if s["op"] in ("for", "while")]
        loop = loops[0]  # then the per-thread store's thread loop
        self.assertEqual(
            (loop["op"], loop["lb"]["v"], loop["ub"]["v"], loop["step"]["v"]),
            ("for", 0, 7, 2),
        )
        self.assertEqual(kir_errors(f), [])
        f = run(
            body.replace("UB", "%r2").replace("+%s", "")
        )  # a runtime bound: no proof
        self.assertEqual(
            [s["op"] for s in f.roles[0]["body"] if s["op"] in ("for", "while")][0],
            "while",
        )

    def test_partition_refinement(self):
        body = (
            PRE
            + """
    shr.u32 %r3, %r1, 5;
    setp.eq.s32 %p1, %r3, 0;
    @!%p1 bra W;
    st.global.b32 [%rd1], %r2;
W:
    ret;
"""
        )
        cute = run(body, family="cute")
        self.assertEqual(
            [(r["name"], r["warps"]) for r in cute.roles],
            [("warps0_0", 1), ("warps1_3", 3)],
        )
        tlx = run(body)
        self.assertEqual([r["name"] for r in tlx.roles], ["default"])

    def test_predicated_metadata_load_stays_in_its_select_arm(self):
        f = run(
            """
    mov.u32 %r1, %tid.x;
    ld.param.b32 %r2, [k_param_2];
    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd4, [k_param_1];
    setp.lt.s32 %p1, %r1, %r2;
    mul.wide.s32 %rd2, %r1, 4;
    add.s64 %rd3, %rd4, %rd2;
    mov.b32 %r4, 0;
    @%p1 ld.global.u32 %r4, [%rd3];
    mul.wide.s32 %rd5, %r4, 4;
    add.s64 %rd6, %rd1, %rd5;
    st.global.b32 [%rd6], %r2;
    ret;
""",
            ptrs=("x", "meta"),
        )
        self.assertTrue(f.tensors["meta"]["metadata"])

        def loads_outside_arms(node, arm=False):
            if not isinstance(node, dict):
                return 0
            n = int(node.get("e") == "load" and not arm)
            for k in ("a", "b", "c"):
                sub = node.get(k)
                n += loads_outside_arms(
                    sub, arm or (node.get("e") == "select" and k in "ab")
                )
            return n + sum(loads_outside_arms(i, arm) for i in node.get("index", []))

        for s in statements(f.roles[0]["body"]):
            for v in s.values():
                self.assertEqual(loads_outside_arms(v), 0, s)
        self.assertEqual(kir_errors(f), [])

    def test_special_registers(self):
        f = run(
            PRE
            + """
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %nctaid.y;
    mov.u32 %r5, %cluster_ctarank;
    mov.u32 %r6, %ntid.x;
    add.s32 %r7, %r3, %r4;
    add.s32 %r7, %r7, %r5;
    add.s32 %r7, %r7, %r6;
    setp.eq.s32 %p1, %r1, 0;
    mul.wide.s32 %rd2, %r7, 4;
    add.s64 %rd3, %rd1, %rd2;
    @%p1 st.global.b32 [%rd3], %r2;
    ret;
""",
            cluster=(2, 1, 1),
            family="cute",
        )
        text = str(f.roles[0]["body"])
        self.assertIn("'e': 'pid', 'w': 32, 'axis': 0", text)
        self.assertIn("'e': 'nprog', 'w': 32, 'axis': 1", text)
        self.assertIn("'o': 'urem'", text)  # the CTA rank in a cluster of 2
        f = run(PRE + "    mov.u32 %r3, %warpid;\n    ret;\n")
        self.assertIn("%warpid may change", f.refusals[0]["msg"])


class Review(unittest.TestCase):
    """Regressions of the M1 adversarial reviews (values checked with evaluate())."""

    def store_offset(self, f, n, **kw):
        (st,) = [t for t in evaluate(f, {"n": n}, **kw) if t[0] == "gstore"]
        return st[2]

    def test_if_joining_at_the_back_edge_merges(self):
        # Both arms branch back to the header: join CONTINUE, r6 += 1 or 2 per trip.
        f = run(
            PRE
            + """
    mov.b32 %r3, 0;
    mov.b32 %r6, 0;
LOOP:
    setp.ge.s32 %p0, %r3, %r2;
    @%p0 bra OUT;
    add.s32 %r3, %r3, 1;
    setp.gt.s32 %p1, %r2, 5;
    @%p1 bra A;
    add.s32 %r6, %r6, 1;
    bra.uni LOOP;
A:
    add.s32 %r6, %r6, 2;
    bra.uni LOOP;
OUT:
    mul.wide.s32 %rd2, %r6, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r2;
    ret;
""",
            threads=32,
        )
        self.assertEqual(kir_errors(f), [])
        self.assertEqual(self.store_offset(f, 3), [3] * 32)
        self.assertEqual(self.store_offset(f, 7), [14] * 32)

    def test_value_of_the_elected_lane_is_not_lane_0s(self):
        # An event operand, or a branch, that depends on which lane elect.sync picks is
        # refused rather than evaluated for lane 0 (H1/H2), in both families.
        operand = """
    elect.sync %r3|%p1, -1;
    and.b32 %r4, %r1, 31;
    selp.b32 %r5, 32, 0, %p1;
    add.s32 %r6, %r4, %r5;
    mul.wide.u32 %rd2, %r6, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r2;
    ret;
"""
        branch = """
    elect.sync %r3|%p1, -1;
    and.b32 %r4, %r1, 31;
    selp.b32 %r5, %r4, 0, %p1;
    setp.ne.s32 %p2, %r5, 0;
    @!%p2 bra SKIP;
    st.global.b32 [%rd1], %r2;
SKIP:
    ret;
"""
        for fam in ("ttgir", "cute"):
            f = run(PRE + operand, family=fam, threads=32)
            self.assertIn(
                "depends on which lane elect.sync picks", f.refusals[0]["msg"]
            )
            f = run(PRE + branch, family=fam, threads=32)  # was folded away: no store
            self.assertIn("another condition on the elected lane", f.refusals[0]["msg"])

    def test_divergent_exit_in_a_nested_uniform_if(self):
        f = run(
            PRE
            + """
    and.b32 %r4, %r1, 31;
    mov.b32 %r3, 0;
LOOP:
    setp.lt.u32 %p1, %r4, 16;
    @%p1 bra A;
    setp.gt.s32 %p2, %r2, 5;
    @%p2 bra OUT;
    bra NEXT;
A:
    setp.gt.s32 %p3, %r2, 7;
    @%p3 bra OUT;
NEXT:
    add.s32 %r3, %r3, 1;
    setp.lt.s32 %p4, %r3, 10;
    @%p4 bra LOOP;
OUT:
    mul.wide.s32 %rd2, %r3, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r2;
    ret;
""",
            threads=32,
        )
        self.assertIn("thread-varying condition (divergent loop)", f.refusals[0]["msg"])

    def test_uniform_if_slot_is_declared_without_branch_names(self):
        # r3, live after the join, is a loop's carried slot inside the taken arm.
        f = run(
            PRE
            + """
    mov.b32 %r3, 0;
    setp.gt.s32 %p1, %r2, 0;
    @%p1 bra BODY;
    bra.uni J;
BODY:
    mov.b32 %r5, 0;
LOOP:
    add.s32 %r3, %r3, 4;
    add.s32 %r5, %r5, 1;
    setp.lt.s32 %p2, %r5, %r2;
    @%p2 bra LOOP;
J:
    mul.wide.s32 %rd2, %r3, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r2;
    ret;
""",
            threads=32,
        )
        self.assertEqual(kir_errors(f), [])
        self.assertEqual(self.store_offset(f, 3), [12] * 32)
        self.assertEqual(self.store_offset(f, -1), [0] * 32)

    def test_masked_lane_table_reads_no_off_lane(self):
        # if (lane < n) r = meta[lane]; shfl: the per-lane lets of r run unconditionally,
        # so a lane outside the mask must not load (an empty meta has no element 0).
        f = run(
            """
    mov.u32 %r1, %tid.x;
    ld.param.b32 %r2, [k_param_2];
    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd4, [k_param_1];
    and.b32 %r3, %r1, 31;
    mov.b32 %r4, 0;
    setp.lt.s32 %p1, %r3, %r2;
    @!%p1 bra SKIP;
    mul.wide.s32 %rd2, %r3, 4;
    add.s64 %rd3, %rd4, %rd2;
    ld.global.u32 %r4, [%rd3];
SKIP:
    shfl.sync.idx.b32 %r5, %r4, 0, 31, -1;
    setp.eq.s32 %p2, %r1, 0;
    mul.wide.s32 %rd5, %r5, 4;
    add.s64 %rd6, %rd1, %rd5;
    @%p2 st.global.b32 [%rd6], %r2;
    ret;
""",
            family="cute",
            ptrs=("x", "meta"),
            threads=32,
        )
        self.assertEqual(kir_errors(f), [])
        self.assertEqual(self.store_offset(f, 0, metadata={"meta": []}), [0])
        self.assertEqual(self.store_offset(f, 2, metadata={"meta": [9, 8]}), [9])

    def test_select_arm_side_conditions_are_dead(self):
        # R1: a predicated (or branch-local) value is a select arm; its address wrap and
        # its metadata load or division happen only where the arm is chosen.
        pred = (
            PRE
            + """
    mov.b32 %r3, 0;
    setp.gt.s32 %p1, %r2, 5;
    @%p1 add.s32 %r3, %r2, 2147483645;
    setp.eq.s32 %p2, %r1, 0;
    mul.wide.s32 %rd2, %r3, 4;
    add.s64 %rd3, %rd1, %rd2;
    @%p2 st.global.b32 [%rd3], %r2;
    ret;
"""
        )
        for fam in ("ttgir", "cute"):
            f = run(pred, family=fam, threads=32)
            self.assertNotIn(("overflow", "add"), evaluate(f, {"n": 3}))  # not run
            self.assertIn(("overflow", "add"), evaluate(f, {"n": 6}))
        branch = """
    mov.u32 %r1, %tid.x;
    ld.param.b32 %r2, [k_param_2];
    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd4, [k_param_1];
    mov.b32 %r4, 0;
    mov.b32 %r6, 0;
    setp.gt.s32 %p1, %r2, 1000;
    @!%p1 bra J;
    ld.global.u32 %r5, [%rd4];
    div.u32 %r4, %r5, %r2;
    add.s32 %r6, %r2, 2147483645;
J:
    add.s32 %r7, %r4, %r6;
    setp.eq.s32 %p2, %r1, 0;
    mul.wide.s32 %rd5, %r7, 4;
    add.s64 %rd6, %rd1, %rd5;
    @%p2 st.global.b32 [%rd6], %r2;
    ret;
"""
        f = run(branch, family="cute", ptrs=("x", "meta"), threads=32)
        self.assertEqual(kir_errors(f), [])
        trace = evaluate(f, {"n": 3}, metadata={"meta": []})
        self.assertEqual([t for t in trace if t[0] != "gstore"], [])  # no load, wrap
        self.assertEqual(trace[-1][2], [0])


if __name__ == "__main__":
    unittest.main()
