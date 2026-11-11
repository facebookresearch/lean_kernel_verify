# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the PTX core's thread model and value semantics (frontend/ptx/interp.py,
expr.py), through translate() on hand-written kernels.

    python3 -I tests/ptx/test_interp.py      (or the buck target :test_interp)

Covers: the PTX integer expansions against brute force (mul.hi, bfe, bfi, prmt and its
modes, lop3, shf, popc, clz, brev, cvt and cvt.sat, add/sub.sat, mul24, sad, div_exact),
the test of one bit of a packed predicate mask,
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

import itertools
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


def _wraps(e, env: dict) -> tuple[list, int]:
    """(the add/sub/mul/shl/trunc nodes of e whose signed result wraps (what an addr mark
    reports) on the variable values env, outside the sign extensions ashr(shl(x, k), k2)
    translate.mark_addr skips; e's value, evaluated exactly on the way)."""
    out = []

    def go(x):
        k, w = x[0], E.width(x)
        if k == "const":
            return x[2]
        if k == "var":
            return env[x[2]]
        if k == "ext":
            v = go(x[3])
            return v if x[2] else v & ((1 << E.width(x[3])) - 1)
        if k == "trunc":
            v = go(x[2])
            r = E._wrap(v, w)
            if r != v:
                out.append(x)
            return r
        if k == "select":
            return go(x[3]) if go(x[2]) else go(x[4])
        if k == "cmp":
            a, b = go(x[2]), go(x[3])
            if x[1][0] == "u":
                m = (1 << E.width(x[2])) - 1
                a, b = a & m, b & m
            op = {"eq": "__eq__", "ne": "__ne__", "lt": "__lt__", "le": "__le__",
                  "gt": "__gt__", "ge": "__ge__"}[x[1].lstrip("su")]  # fmt: skip
            return int(getattr(a, op)(b))
        op, a, b = x[2], x[3], x[4]
        if op == "ashr" and a[0] == "bin" and a[2] == "shl" and E.is_const(a[4]) \
                and E.is_const(b) and a[4][2] <= b[2]:  # fmt: skip
            return E._wrap(go(a[3]) << a[4][2], w) >> b[2]
        va, vb = go(a), go(b)
        ua, ub = va & ((1 << w) - 1), vb & ((1 << w) - 1)
        full = {"add": va + vb, "sub": va - vb, "mul": va * vb,
                "shl": va << ub if ub < w else 0, "lshr": ua >> ub if ub < w else 0,
                "ashr": va >> min(ub, w - 1), "and": va & vb, "or": va | vb,
                "xor": va ^ vb, "smin": min(va, vb), "smax": max(va, vb),
                "umin": min(ua, ub), "umax": max(ua, ub)}[op]  # fmt: skip
        r = E._wrap(full, w)
        if op in ("add", "sub", "mul", "shl") and r != full:
            out.append(x)
        return r

    return out, go(e)


class MoreExpansions(unittest.TestCase):
    """mul.hi at 16 and 64 bits, packed 16-bit lanes, bfind, dp2a, add/sub.sat, mul24,
    sad, cvt.sat and prmt's modes against brute force; the mul.hi and add/sub.sat
    expansions have no node an address mark would report (design §5.8), the others only
    where the result does not fit."""

    def setUp(self):
        self.rng = random.Random(11)

    def test_mul_hi_16_and_64_bits(self):
        for w in (16, 64):
            edge = [0, 1, 2, (1 << (w - 1)) - 1, 1 << (w - 1), (1 << w) - 1]
            vals = edge + [self.rng.getrandbits(w) for _ in range(150)]
            if w == 64:
                vals.append(1418980313362273202)  # ceil(2**64 / 13): x // 13
            a, b = E.var("a", w), E.var("b", w)
            for signed in (False, True):
                sym = E.lower_narrow(E.mul_hi(a, b, signed))
                self.assertTrue(all(E.width(x) in E.KIR_WIDTHS for x in _walk(sym)))
                for x in vals:
                    for y in vals[:: 7 if w == 64 else 3]:
                        sx, sy = (E._wrap(v, w) if signed else v for v in (x, y))
                        want = ((sx * sy) >> w) & ((1 << w) - 1)
                        self.assertEqual(
                            E.mul_hi(E.const(x, w), E.const(y, w), signed)[2]
                            & ((1 << w) - 1),
                            want,
                        )
                        env = {"a": E._wrap(x, 64), "b": E._wrap(y, 64)}
                        if w == 16:  # lowered: the i32 zero extensions
                            env = {"a": x, "b": y}
                        wraps, v = _wraps(sym, env)
                        self.assertEqual(wraps, [], (w, signed, x, y))
                        self.assertEqual(v & ((1 << w) - 1), want)

    def test_packed_lanes_bfind_dp2a(self):
        for _ in range(400):
            a, b, c = (self.rng.getrandbits(32) for _ in range(3))
            lo, hi = (x[2] & 0xFFFF for x in E.lanes16(E.const(a, 32)))
            self.assertEqual((lo, hi), (a & 0xFFFF, a >> 16))
            packed = E.pack16(E.const(b >> 16, 16), E.const(c >> 16, 16))
            self.assertEqual(packed[2] & 0xFFFFFFFF, (c >> 16 << 16) | (b >> 16))
            for hmode in (False, True):
                for sa in (False, True):
                    for sb in (False, True):
                        h = [E._wrap(a >> 16 * j, 16) if sa else a >> 16 * j & 0xFFFF
                             for j in range(2)]  # fmt: skip
                        k = 2 if hmode else 0
                        by = [E._wrap(b >> 8 * (k + j), 8) if sb else b >> 8 * (k + j) & 255
                              for j in range(2)]  # fmt: skip
                        got = E.dp2a(c32(a), c32(b), c32(c), hmode, sa, sb)
                        want = (c + h[0] * by[0] + h[1] * by[1]) & 0xFFFFFFFF
                        self.assertEqual(u32(got), want)
            for w in (32, 64):
                x = self.rng.getrandbits(w) >> self.rng.randrange(w + 1)
                for signed in (False, True):
                    v = E._wrap(x, w) if signed else x
                    y = ~v if v < 0 else v
                    for shiftamt in (False, True):
                        want = 0xFFFFFFFF if y == 0 else (
                            w - y.bit_length() if shiftamt else y.bit_length() - 1)  # fmt: skip
                        got = E.bfind(E.const(x, w), signed, shiftamt)
                        self.assertEqual(u32(got), want)

    def fits(self, sym, env: dict, want: int, w: int = 32) -> None:
        """sym evaluates to want modulo 2**w, and an address mark reports a node of it
        exactly when want does not fit a signed i32."""
        wraps, v = _wraps(sym, env)
        self.assertEqual(v & ((1 << w) - 1), want & ((1 << w) - 1))
        self.assertEqual(wraps != [], not -(2**31) <= want < 2**31, (env, want))

    def test_sat_mul24_sad(self):
        edge = [0, 1, 2, 5, 0x7FFFFF, 0x800000, 0xFFFFFF, 0x1000000, 0x7FFFFFFF,
                0x80000000, 0xFFFFFFFB, 0xFFFFFFFF]  # fmt: skip
        vals = edge + [self.rng.getrandbits(32) for _ in range(30)]
        a, b, c = (E.var(n, 32) for n in "abc")
        for x in vals:
            for y in vals:
                sx, sy = E._wrap(x, 32), E._wrap(y, 32)
                for sub in (False, True):  # add/sub.sat.s32: never a wrap
                    want = max(min(sx - sy if sub else sx + sy, 2**31 - 1), -(2**31))
                    self.fits(E.add_sat(a, b, sub), {"a": sx, "b": sy}, want)
                    self.assertEqual(E.add_sat(c32(x), c32(y), sub)[2], want)
                z = vals[(x ^ y) % len(vals)]
                for signed in (False, True):
                    # mul24: the 24-bit fields' product; .lo wraps as mul.lo does
                    fx, fy = (
                        E._wrap(v, 24) if signed else v & 0xFFFFFF for v in (x, y)
                    )
                    for hi, want in ((False, fx * fy), (True, fx * fy >> 16)):
                        env = {"a": sx, "b": sy}
                        self.fits(E.mul24(a, b, hi, signed), env, want)
                        got = E.mul24(c32(x), c32(y), hi, signed)
                        self.assertEqual(u32(got), want & 0xFFFFFFFF)
                    # sad: c + |a - b| by the type, computed where only the sum wraps
                    rx, ry, rz = (E._wrap(v, 32) if signed else v for v in (x, y, z))
                    want = rz + abs(rx - ry)
                    env = {"a": sx, "b": sy, "c": E._wrap(z, 32)}
                    self.fits(E.sad(a, b, c, signed), env, want)
                    self.assertEqual(u32(E.sad(c32(x), c32(y), c32(z), signed)),
                                     want & 0xFFFFFFFF)  # fmt: skip
        for w in (16, 64):  # sad at the other widths: values; no wrap on small operands
            m = (1 << w) - 1
            ws = [0, 1, 7, (1 << (w - 1)) - 1, 1 << (w - 1), m]
            ws += [self.rng.getrandbits(w) for _ in range(12)]
            av, bv, cv = (E.var(n, w) for n in "abc")
            for signed in (False, True):
                sym = E.lower_narrow(E.sad(av, bv, cv, signed))
                for x in ws:
                    for y in ws:
                        for z in ws[:6]:
                            rx, ry, rz = (
                                E._wrap(v, w) if signed else v for v in (x, y, z)
                            )
                            want = rz + abs(rx - ry)
                            env = {n: (v if w == 16 else E._wrap(v, 64))
                                   for n, v in zip("abc", (x, y, z))}  # fmt: skip
                            wraps, v = _wraps(sym, env)
                            self.assertEqual(v & m, want & m)
                            if w == 16 or all(abs(e) < 2**61 for e in env.values()):
                                self.assertEqual(wraps, [], (w, signed, x, y, z))

    def test_cvt_sat(self):
        # every pair of integer types: the source value clamped to the destination's
        # range; only a truncation to u32 of a 64-bit value of 2**31 or more is lossy
        for sk, sw, dk, dw in itertools.product("su", (8, 16, 32, 64), repeat=2):
            rw = max(sw, 32)  # the source register
            ss, ds = sk == "s", dk == "s"
            lo, hi = (
                (-(1 << (dw - 1)), (1 << (dw - 1)) - 1) if ds else (0, (1 << dw) - 1)
            )
            sym = E.lower_narrow(E.cvt_sat(E.var("a", rw), dw, ds, sw, ss))
            edge = [0, 1, (1 << (sw - 1)) - 1, 1 << (sw - 1), (1 << sw) - 1]
            for x in edge + [self.rng.getrandbits(sw) for _ in range(40)]:
                junk = self.rng.getrandbits(rw - sw) << sw if rw > sw else 0
                v = E._wrap(x, sw) if ss else x
                want = max(min(v, hi), lo)
                got = E.cvt_sat(E.const(x | junk, rw), dw, ds, sw, ss)
                self.assertEqual(got[2] & ((1 << dw) - 1), want & ((1 << dw) - 1))
                wraps, r = _wraps(sym, {"a": E._wrap(x | junk, rw)})
                self.assertEqual(r & ((1 << dw) - 1), want & ((1 << dw) - 1))
                lossy = sw == 64 and dw == 32 and not ds and want >= 2**31
                self.assertEqual(wraps != [], lossy, (sk, sw, dk, dw, x))

    def test_prmt_modes(self):
        # the ISA's selections by name: forward and backward 4 extract (bytes c, c +- 1,
        # ...), replicate 8 and 16, edge clamp left and right; constant or runtime c
        pick = {"f4e": lambda c, i: c + i, "b4e": lambda c, i: (c - i) % 8,
                "rc8": lambda c, i: c, "ecl": max, "ecr": min,
                "rc16": lambda c, i: 2 * (c & 1) + (i & 1)}  # fmt: skip
        cv = E.var("c", 32)
        for _ in range(60):
            a, b, c = (self.rng.getrandbits(32) for _ in range(3))
            src = (b << 32 | a).to_bytes(8, "little")
            for mode, f in pick.items():
                want = sum(src[f(c & 3, i)] << 8 * i for i in range(4))
                self.assertEqual(u32(E.prmt_mode(c32(a), c32(b), c32(c), mode)), want)
                sym = E.prmt_mode(c32(a), c32(b), cv, mode)
                self.assertEqual(
                    _wraps(sym, {"c": E._wrap(c, 32)})[1] & 0xFFFFFFFF, want
                )


def _walk(e):
    seen, stack = set(), [e]
    while stack:
        x = stack.pop()
        if x not in seen:
            seen.add(x)
            yield x
            stack.extend(E.children(x))


class MoreIntegerOps(unittest.TestCase):
    """Kernels using integer ops that real kernels use: their store offsets on concrete
    inputs (fixtures evaluate()), with no address wrap reported, and the refusals that
    remain."""

    def offsets(self, body, n, family="cute", **kw):
        f = run(PRE + body + "    ret;\n", family=family, **kw)
        self.assertEqual(f.refusals, [])
        self.assertEqual(kir_errors(f), [])
        trace = evaluate(f, {"n": n})
        self.assertEqual([t for t in trace if t[0] != "gstore"], [])  # no wrap
        return [o for t in trace for o in t[2]]

    def test_abs(self):
        body = """
    sub.s32 %r3, %r1, %r2;
    abs.s32 %r4, %r3;
    cvt.s64.s32 %rd2, %r3;
    abs.s64 %rd3, %rd2;
    cvt.u32.u64 %r5, %rd3;
    add.s32 %r6, %r4, %r5;
    mul.wide.s32 %rd4, %r6, 4;
    add.s64 %rd5, %rd1, %rd4;
    st.global.b32 [%rd5], %r2;
"""
        self.assertEqual(self.offsets(body, 50), [2 * abs(t - 50) for t in range(128)])

    def test_mul_hi_64_and_16_bits(self):
        # (t + n) // 13 by LLVM's magic number (qk_norm_rope.py:38), the signed high half
        # of (t - 64) * -2**62 (floor(-(t - 64) / 4)), and the 16-bit (500 t * 0xCCCD) >> 16.
        body = """
    cvt.u64.u32 %rd2, %r1;
    cvt.u64.u32 %rd3, %r2;
    add.s64 %rd4, %rd2, %rd3;
    mul.hi.u64 %rd5, %rd4, 1418980313362273202;
    sub.s32 %r3, %r1, 64;
    cvt.s64.s32 %rd6, %r3;
    mul.hi.s64 %rd7, %rd6, -4611686018427387904;
    add.s64 %rd8, %rd5, %rd7;
    mul.lo.s32 %r4, %r1, 500;
    cvt.u16.u32 %rs1, %r4;
    mul.hi.u16 %rs2, %rs1, 52429;
    cvt.u64.u16 %rd9, %rs2;
    add.s64 %rd10, %rd8, %rd9;
    shl.b64 %rd11, %rd10, 2;
    add.s64 %rd12, %rd1, %rd11;
    st.global.b32 [%rd12], %r2;
"""
        want = [(t + 1000) // 13 + (-(t - 64)) // 4 + (500 * t * 52429 >> 16)
                for t in range(128)]  # fmt: skip
        for fam in ("cute", "ttgir"):
            self.assertEqual(self.offsets(body, 1000, family=fam), want)

    def test_packed_16x2(self):
        # lanes (lo, hi) = (-1, t & 7) and (2, 3): add (1, (t & 7) + 3), max.u (0xFFFF,
        # max(t & 7, 3)), min.s (-1, min(t & 7, 3)), sub (-3, (t & 7) - 3); max.relu.s of
        # the first with itself (0, t & 7)
        body = """
    and.b32 %r3, %r1, 7;
    shl.b32 %r4, %r3, 16;
    or.b32 %r5, %r4, 65535;
    mov.b32 %r6, 196610;
    add.s16x2 %r7, %r5, %r6;
    max.u16x2 %r8, %r5, %r6;
    min.s16x2 %r9, %r5, %r6;
    max.relu.s16x2 %r10, %r5, %r5;
    sub.u16x2 %r11, %r5, %r6;
    mov.b32 {%rs1, %rs2}, %r7;
    mov.b32 {%rs3, %rs4}, %r8;
    mov.b32 {%rs5, %rs6}, %r9;
    mov.b32 {%rs7, %rs8}, %r10;
    mov.b32 {%rs9, %rs10}, %r11;
    cvt.s32.s16 %r12, %rs1;
    cvt.s32.s16 %r13, %rs2;
    cvt.u32.u16 %r14, %rs3;
    cvt.s32.s16 %r15, %rs4;
    cvt.s32.s16 %r16, %rs5;
    cvt.s32.s16 %r17, %rs6;
    cvt.s32.s16 %r18, %rs7;
    cvt.s32.s16 %r19, %rs8;
    cvt.s32.s16 %r20, %rs9;
    cvt.s32.s16 %r21, %rs10;
    add.s32 %r22, %r12, %r13;
    add.s32 %r22, %r22, %r14;
    add.s32 %r22, %r22, %r15;
    add.s32 %r22, %r22, %r16;
    add.s32 %r22, %r22, %r17;
    add.s32 %r22, %r22, %r18;
    add.s32 %r22, %r22, %r19;
    add.s32 %r22, %r22, %r20;
    add.s32 %r22, %r22, %r21;
    mul.wide.s32 %rd2, %r22, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r2;
"""
        want = []
        for t in range(128):
            h = t & 7
            want.append((1 + h + 3) + (0xFFFF + max(h, 3)) + (-1 + min(h, 3))
                        + (0 + h) + (-3 + h - 3))  # fmt: skip
        self.assertEqual(self.offsets(body, 0), want)

    def test_redux_sync(self):
        # add of the lane ids (496), max.u of tid (32 w + 31), min.s of tid - 100, or of
        # 1 << (lane % 4) (15): every lane gets its warp's reduction
        body = """
    mov.u32 %r3, %laneid;
    redux.sync.add.s32 %r4, %r3, -1;
    redux.sync.max.u32 %r5, %r1, -1;
    sub.s32 %r6, %r1, 100;
    redux.sync.min.s32 %r7, %r6, -1;
    and.b32 %r8, %r3, 3;
    shl.b32 %r9, 1, %r8;
    redux.sync.or.b32 %r10, %r9, -1;
    add.s32 %r11, %r4, %r5;
    add.s32 %r11, %r11, %r7;
    add.s32 %r11, %r11, %r10;
    mul.wide.s32 %rd2, %r11, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r2;
"""
        want = [496 + (t | 31) + (t // 32 * 32 - 100) + 15 for t in range(128)]
        for fam in ("cute", "ttgir"):
            self.assertEqual(self.offsets(body, 0, family=fam), want)
        # Of data: one value per warp, Data; refused only where it reaches a root.
        data = """
    ld.global.f32 %f1, [%rd1];
    mov.b32 %r3, %f1;
    redux.sync.add.s32 %r4, %r3, -1;
    st.global.b32 [%rd1], %r4;
"""
        self.assertEqual(run(PRE + data + "    ret;\n").refusals, [])
        f = run(PRE + data.replace("st.global.b32 [%rd1], %r4", """
    mul.wide.s32 %rd2, %r4, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r2""") + "    ret;\n")  # fmt: skip
        self.assertIn("data value from", f.refusals[0]["msg"])
        f = run(PRE + """
    mov.u32 %r3, %laneid;
    setp.lt.u32 %p1, %r3, 16;
    @!%p1 bra SKIP;
    redux.sync.add.s32 %r4, %r3, -1;
SKIP:
    ret;
""")  # fmt: skip
        self.assertIn("only part of warp(s)", f.refusals[0]["msg"])

    def test_relu_bfind_dp2a(self):
        body = """
    sub.s32 %r3, %r1, 64;
    max.relu.s32 %r4, %r3, -5;
    min.relu.s32 %r5, %r3, 10;
    bfind.u32 %r6, %r1;
    bfind.shiftamt.u32 %r7, %r1;
    bfind.s32 %r8, %r3;
    dp2a.lo.u32.u32 %r9, %r1, 770, %r2;
    dp2a.hi.s32.s32 %r10, %r3, -16777216, 0;
    add.s32 %r11, %r4, %r5;
    add.s32 %r11, %r11, %r6;
    add.s32 %r11, %r11, %r7;
    add.s32 %r11, %r11, %r8;
    add.s32 %r11, %r11, %r9;
    add.s32 %r11, %r11, %r10;
    mul.wide.s32 %rd2, %r11, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r2;
"""

        def msb(v):
            return v.bit_length() - 1 if v else -1

        want = []
        for t in range(128):
            x = t - 64
            hi = -1 if x < 0 else 0  # x's upper half, sign-extended
            want.append(max(x, -5, 0) + max(min(x, 10), 0) + msb(t)
                        + (31 - msb(t) if t else -1) + msb(~x if x < 0 else x)
                        + 7 + 2 * t + hi * -1)  # fmt: skip
        self.assertEqual(self.offsets(body, 7), want)

    def test_carry_chain(self):
        # 0xFFFFFFFF + t carries for t >= 1 (add.cc/addc); so does t * 2**24 + 0xFF000000
        # (mad.lo.cc), which madc.hi adds to the high word of t * 2**28, t >> 4.
        body = """
    mov.b32 %r3, -1;
    add.cc.u32 %r4, %r3, %r1;
    addc.cc.u32 %r5, %r2, 0;
    addc.u32 %r6, 0, 0;
    mad.lo.cc.u32 %r7, %r1, 16777216, -16777216;
    madc.hi.u32 %r8, %r1, 268435456, 0;
    add.s32 %r9, %r5, %r8;
    add.s32 %r9, %r9, %r6;
    mul.wide.s32 %rd2, %r9, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r2;
"""
        want = [5 + (t >= 1) + (t >> 4) + (t >= 1) for t in range(128)]
        self.assertEqual(self.offsets(body, 5), want)
        f = run(PRE + """
    setp.lt.s32 %p1, %r1, %r2;
    @%p1 add.cc.u32 %r4, %r1, 1;
    ret;
""")  # fmt: skip
        self.assertIn("carry-flag instruction under a predicate", f.refusals[0]["msg"])
        f = run(PRE + """
    add.cc.u32 %r4, %r1, 1;
    setp.lt.s32 %p1, %r1, %r2;
    @%p1 bra NEXT;
NEXT:
    addc.u32 %r5, %r2, 0;
    mul.wide.s32 %rd2, %r5, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r2;
    ret;
""")  # fmt: skip
        self.assertIn("not set in its block", f.refusals[0]["msg"])

    def test_borrow_chain(self):
        # sub.cc/subc(.cc): a - (b + CC.CF), the borrow out to CC.CF (ISA), against that
        # formula for every thread and several n (n - t - borrow wraps for t > n); the
        # differences reach the address through their low and top bits.
        body = """
    sub.cc.u32 %r4, %r1, 64;
    subc.cc.u32 %r5, %r2, %r1;
    subc.cc.u32 %r6, 1, 0;
    subc.u32 %r7, 3, 1;
    sub.cc.u32 %r8, %r2, %r2;
    subc.u32 %r9, 5, 0;
    and.b32 %r10, %r5, 7;
    shr.u32 %r11, %r4, 29;
    shr.u32 %r12, %r5, 28;
    add.s32 %r13, %r10, %r11;
    add.s32 %r13, %r13, %r12;
    add.s32 %r13, %r13, %r6;
    add.s32 %r13, %r13, %r7;
    add.s32 %r13, %r13, %r9;
    mul.wide.s32 %rd2, %r13, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r2;
"""

        def sub(a, b, cin):  # (difference mod 2**32, borrow out)
            return (a - b - cin) % 2**32, int(b + cin > a)

        for n in (0, 5, 64, 127, 1000, -1):
            want = []
            for t in range(128):
                d4, c = sub(t, 64, 0)
                d5, c = sub(n % 2**32, t, c)
                d6, c = sub(1, 0, c)
                d7, _ = sub(3, 1, c)
                d9 = 5  # sub.cc of n - n: no borrow
                want.append((d5 & 7) + (d4 >> 29) + (d5 >> 28) + d6 + d7 + d9)
            with self.subTest(n=n):
                self.assertEqual(self.offsets(body, n), want)
        f = run(PRE + """
    setp.lt.s32 %p1, %r1, %r2;
    @%p1 subc.u32 %r4, %r1, 1;
    ret;
""")  # fmt: skip
        self.assertIn("carry-flag instruction under a predicate", f.refusals[0]["msg"])
        f = run(PRE + """
    sub.cc.u32 %r4, %r1, 1;
    setp.lt.s32 %p1, %r1, %r2;
    @%p1 bra NEXT;
NEXT:
    subc.cc.u32 %r5, %r2, 0;
    mul.wide.s32 %rd2, %r5, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r2;
    ret;
""")  # fmt: skip
        self.assertIn("not set in its block", f.refusals[0]["msg"])

    def test_barrier_red(self):
        # popc of t < n over the 128 threads, and of t < n (n >= 128): thread 0 stores at
        # min(n, 128) + 1000 * (n >= 128); exact when the role is the whole block
        body = """
    setp.lt.s32 %p1, %r1, %r2;
    barrier.red.popc.aligned.u32 %r3, 0, %p1;
    barrier.red.and.aligned.pred %p2, 0, %p1;
    bar.red.or.pred %p3, 0, 128, !%p1;
    selp.b32 %r4, 1000, 0, %p2;
    selp.b32 %r5, 10000, 0, %p3;
    add.s32 %r6, %r3, %r4;
    add.s32 %r6, %r6, %r5;
    setp.eq.s32 %p4, %r1, 0;
    mul.wide.s32 %rd2, %r6, 4;
    add.s64 %rd3, %rd1, %rd2;
    @%p4 st.global.b32 [%rd3], %r2;
"""
        for fam in ("cute", "ttgir"):
            f = run(PRE + body + "    ret;\n", family=fam)
            self.assertEqual(f.refusals, [])
            for n, want in ((40, 10040), (0, 10000), (200, 1128)):
                self.assertEqual(evaluate(f, {"n": n})[-1][2], [want])
        # CuTe splits warps 0-1 from 2-3 (the branch): each role sees half the block.
        split = """
    shr.u32 %r7, %r1, 6;
    setp.eq.s32 %p5, %r7, 0;
    @%p5 bra A;
    st.global.b32 [%rd1+4], %r2;
A:
"""
        f = run(PRE + split + body + "    ret;\n", family="cute")
        self.assertIn("barrier.red.popc.aligned.u32 over threads outside the role",
                      f.refusals[0]["msg"])  # fmt: skip
        # bar.red is barrier.red.aligned (ISA): under a lane-varying condition, refused
        lane = """
    and.b32 %r7, %r1, 31;
    setp.lt.u32 %p5, %r7, 16;
    @!%p5 bra B;
    {bar} %r8, 0, %p5;
B:
    ret;
"""
        for bar in ("bar.red.popc.u32", "barrier.red.popc.aligned.u32"):
            f = run(PRE + lane.format(bar=bar))
            self.assertIn(f"{bar[:-4]} under a condition that differs between the lanes",
                          f.refusals[0]["msg"])  # fmt: skip
        f = run(PRE + lane.format(bar="barrier.red.popc.u32"))  # not aligned: allowed
        self.assertEqual(f.refusals, [])

    def test_sat_mul24_sad_cvt_prmt(self):
        # x = t - 64: min(x, 47) + max(x, -48) by add/sub.sat; -x by mul24.lo.s32 of
        # 0xFFFFFF (-1 as 24 bits); t by mul24.hi.u32 of 0x1010000 (bit 24 dropped); -x
        # by mul24.hi.s32 of -65536; n + |x - 5| and |t - 100| by sad; max(x, 0) and
        # clamp(4x, -128, 127) by cvt.sat; 4x wrapped to s8, sign-extended into the b32
        # register as ptxas does; bytes of 0x08070605:0x04030201 picked by prmt's modes
        body = """
    sub.s32 %r3, %r1, 64;
    add.sat.s32 %r4, %r3, 2147483600;
    sub.s32 %r4, %r4, 2147483600;
    sub.sat.s32 %r5, %r3, 2147483600;
    add.s32 %r5, %r5, 2147483600;
    mul24.lo.s32 %r6, %r3, 16777215;
    mul24.hi.u32 %r7, %r1, 16842752;
    mul24.hi.s32 %r8, %r3, -65536;
    sad.s32 %r9, %r3, 5, %r2;
    sad.u32 %r10, %r1, 100, 0;
    cvt.sat.u8.s32 %r11, %r3;
    shl.b32 %r12, %r3, 2;
    cvt.sat.s8.s32 %r13, %r12;
    cvt.s8.s32 %r14, %r12;
    mov.b32 %r15, 67305985;
    mov.b32 %r16, 134678021;
    and.b32 %r17, %r1, 3;
    prmt.b32.rc16 %r18, %r15, %r16, %r17;
    shr.u32 %r18, %r18, 24;
    prmt.b32.ecl %r19, %r15, %r16, %r17;
    and.b32 %r19, %r19, 255;
    prmt.b32.f4e %r20, %r15, %r16, 6;
    shr.u32 %r20, %r20, 24;
    add.s32 %r21, %r4, %r5;
    add.s32 %r21, %r21, %r6;
    add.s32 %r21, %r21, %r7;
    add.s32 %r21, %r21, %r8;
    add.s32 %r21, %r21, %r9;
    add.s32 %r21, %r21, %r10;
    add.s32 %r21, %r21, %r11;
    add.s32 %r21, %r21, %r13;
    add.s32 %r21, %r21, %r14;
    add.s32 %r21, %r21, %r18;
    add.s32 %r21, %r21, %r19;
    add.s32 %r21, %r21, %r20;
    mul.wide.s32 %rd2, %r21, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r2;
"""
        want = []
        for t in range(128):
            x = t - 64
            want.append(min(x, 47) + max(x, -48) - x + t - x + 9 + abs(x - 5)
                        + abs(t - 100) + max(x, 0) + max(min(4 * x, 127), -128)
                        + ((4 * x + 128) % 256 - 128) + (4 if t & 1 else 2)
                        + (t & 3) + 1 + 6)  # fmt: skip
        for fam in ("cute", "ttgir"):
            self.assertEqual(self.offsets(body, 9, family=fam), want)
        f = run(PRE + """
    prmt.b32 %r3, %r1, %r2, %r1;
    mul.wide.s32 %rd2, %r3, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r2;
    ret;
""")  # fmt: skip
        self.assertIn("prmt with a runtime selector", f.refusals[0]["msg"])

    def test_mma_sync_is_register_data(self):
        body = """
    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%f1, %f2, %f3, %f4}, {%r3, %r4, %r5, %r6}, {%r7, %r8}, {%f5, %f6, %f7, %f8};
    st.global.f32 [%rd1], %f1;
    ret;
"""
        self.assertEqual(run(PRE + body).refusals, [])
        f = run(PRE + body.replace("st.global.f32 [%rd1], %f1", """
    mov.b32 %r9, %f1;
    mul.wide.s32 %rd2, %r9, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r2"""))  # fmt: skip
        self.assertIn("mma.sync", f.refusals[0]["msg"])


if __name__ == "__main__":
    unittest.main()
