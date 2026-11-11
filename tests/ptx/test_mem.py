# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the PTX core's memory model (frontend/ptx/mem.py) through translate().

    python3 -I tests/ptx/test_mem.py      (or the buck target :test_mem)

Global accesses: provenance from a pointer parameter to element offsets (byte offsets
divided exactly), a byte offset that is no multiple of the element refused, loop-carried
pointers (uniform and per thread) whose slots' trailing zeros prove it, a vector access
as consecutive elements, a sub-element access refused, a generic load of shared memory
as data, an address of unknown provenance refused, a select between two tensors (the
CuTe policy splits the role by its warp-uniform condition), a per-lane table of
addresses of one base, unrolled loads of one instruction family merged into one event;
metadata loads (value in an address: load node, metadata tensor), strong
(.volatile/.relaxed/.acquire) loads as Data; cp.async with src-size and with a
cache-hint operand (R2); atom/red (gatomic, exch: gstore), on shared memory
(atom/red.shared, generic) no event and Data results. mbarriers:
init/arrive/expect_tx/wait resolved to arrays; runs split by init .loc (Triton) and by
uses (CuTe policy); the AST name of the init's line; an arrive count; threads_per_issue
(CuTe gcd) with an arrive counted in a thread loop; a loop-carried slot address keeping
its constant part (known trailing zeros prove the index exact); a non-constant init and
an unresolvable offset refused.
"""

import itertools
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

if __package__ in (None, ""):
    sys.dont_write_bytecode = True
    _root = ModuleType("lkv_root")
    _root.__path__ = [str(Path(__file__).resolve().parents[2])]
    sys.modules["lkv_root"] = _root
    __package__ = "lkv_root.tests.ptx"  # noqa: A001

from ...frontend.ptx import expr as E, mem as M  # noqa: E402
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
    setp.eq.s32 %p9, %r1, 0;
"""


def offset_text(ev):
    return str(ev["offsets"])


class Global(unittest.TestCase):
    def test_provenance_and_division(self):
        f = run(
            PRE
            + """
    mul.wide.s32 %rd2, %r2, 4;
    add.s64 %rd3, %rd1, %rd2;
    @%p9 ld.global.f32 %f1, [%rd3+8];
    ret;
"""
        )
        (ev,) = events(f, "gload")
        self.assertEqual(ev["tensor"], "x")
        a = ev["offsets"]["a"]  # (sext(n) * 4 + 8) / 4 = sext(n) + 2, exactly
        self.assertEqual((a["e"], a["o"], a["b"]["v"]), ("bin", "add", 2))
        self.assertEqual(kir_errors(f), [])

    def test_misaligned_and_sub_element(self):
        f = run(
            PRE
            + """
    cvt.u64.u32 %rd2, %r2;
    add.s64 %rd3, %rd1, %rd2;
    @%p9 ld.global.f32 %f1, [%rd3];
    ret;
"""
        )
        self.assertIn("not provably a multiple", f.refusals[0]["msg"])
        f = run(PRE + "    @%p9 ld.global.u8 %rs1, [%rd1];\n    ret;\n")
        self.assertIn("sub-element access", f.refusals[0]["msg"])

    def test_loop_carried_pointers(self):
        # LSR: x + 4*tid and x + 4*n step by 512 per trip (TLX's M and D pointers): both
        # are uniform slots, the first with its thread term 4*tid kept outside; their
        # trailing zeros prove the element offsets exact. A step of 514 does not; a step
        # of 4*tid makes the first a per-thread family.
        body = (
            PRE
            + """
    mul.wide.u32 %rd2, %r1, 4;
    add.s64 %rd3, %rd1, %rd2;
    mul.wide.s32 %rd4, %r2, 4;
    add.s64 %rd5, %rd1, %rd4;
    mov.b32 %r21, 0;
LOOP:
    ld.global.b32 %r5, [%rd3];
    @%p9 st.global.b32 [%rd5], %r5;
    add.s64 %rd3, %rd3, STEP;
    add.s64 %rd5, %rd5, 512;
    add.s32 %r21, %r21, 1;
    setp.lt.s32 %p13, %r21, %r2;
    @%p13 bra LOOP;
    ret;
"""
        )

        def family(f):  # per-thread slot variables c<k>_<thread>
            return [s["var"] for s in statements(f.roles[0]["body"])
                    if s["op"] == "let" and s["var"].count("_") == 1]  # fmt: skip

        f = run(body.replace("STEP", "512"))
        self.assertEqual(f.refusals, [])
        self.assertEqual(len(events(f, "gload")) + len(events(f, "gstore")), 2)
        self.assertEqual(family(f), [])
        self.assertEqual(kir_errors(f), [])
        f = run(body.replace("STEP", "514"))
        self.assertIn("not provably a multiple", f.refusals[0]["msg"])
        f = run(body.replace("STEP", "%rd2"))
        self.assertEqual(f.refusals, [])
        self.assertEqual(len(family(f)) % 128, 0)  # entry, temporaries, back edge
        self.assertTrue(family(f))
        self.assertEqual(kir_errors(f), [])

    def test_vector_and_generic_shared(self):
        f = run(
            PRE
            + """
    mov.u32 %r3, smem;
    cvt.u64.u32 %rd4, %r3;
    @%p9 ld.global.v4.b32 {%r4, %r5, %r6, %r7}, [%rd1+16];
    ld.u32 %r8, [%r3];
    ret;
""",
            elem={"x": 2},
        )
        (ev,) = events(f)
        self.assertEqual(ev["offsets"]["shape"], [8])  # 16 bytes of 2-byte elements
        self.assertEqual(ev["offsets"]["b"]["t"], "range")

    def test_unknown_provenance(self):
        f = run(
            PRE
            + """
    cvt.u64.u32 %rd2, %r2;
    @%p9 st.global.b32 [%rd2], %r2;
    ret;
"""
        )
        self.assertIn("no tensor provenance", f.refusals[0]["msg"])

    def test_select_between_tensors(self):
        # LLVM merges `if tid < 64: store x else: store y` into a selp of the base: the
        # CuTe policy splits the role by the warp-uniform condition, each part storing
        # one tensor; a condition that varies within a warp, or no splitting (Triton
        # partitions), is refused.
        body = """
    mov.u32 %r1, %tid.x;
    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd2, [k_param_1];
    setp.lt.u32 %p1, %r1, LIMIT;
    selp.b64 %rd3, %rd1, %rd2, %p1;
    mul.wide.u32 %rd4, %r1, 4;
    add.s64 %rd5, %rd3, %rd4;
    st.global.b32 [%rd5+64], %r1;
    ret;
"""
        f = run(body.replace("LIMIT", "64"), "cute", ptrs=("x", "y"), scalars=())
        self.assertEqual(
            [(r["name"], [e["tensor"] for e in statements(r["body"]) if e["op"] == "event"])
             for r in f.roles],
            [("warps0_1", ["x"]), ("warps2_3", ["y"])],
        )  # fmt: skip
        self.assertEqual(kir_errors(f), [])
        for limit, family in (("16", "cute"), ("64", "ttgir")):
            f = run(body.replace("LIMIT", limit), family, ptrs=("x", "y"), scalars=())
            self.assertIn("no tensor provenance", f.refusals[0]["msg"])

    def test_peel_table(self):
        # A per-lane table of addresses of one base is that base plus a table of offsets.
        x, y = E.var("%g:x", 64), E.var("%g:y", 64)
        a, b = E.var("a", 64), E.var("b", 64)
        t = E.table(E.LANE, (E.bin("add", x, a), E.bin("add", x, b)))
        self.assertEqual(M.peel(t), (x, E.table(E.LANE, (a, b))))
        self.assertIsNone(M.peel(E.table(E.LANE, (E.bin("add", x, a), y))))

    def test_unrolled_loads_merge(self):
        body = (
            PRE
            + """
    mul.wide.u32 %rd2, %r1, 4;
    add.s64 %rd3, %rd1, %rd2;
    setp.lt.s32 %p1, %r1, %r2;
    add.s32 %r3, %r1, 128;
    setp.lt.s32 %p2, %r3, %r2;
    add.s32 %r4, %r1, 256;
    setp.lt.s32 %p3, %r4, %r2;
    @%p1 ld.global.b32 %r5, [%rd3];
    @%p2 ld.global.b32 %r6, [%rd3+512];
    @%p3 ld.global.b32 %r7, [%rd3+1024];
    ret;
"""
        )
        (ev,) = events(run(body), "gload")
        self.assertEqual(ev["offsets"]["shape"], [3, 128])  # instructions x threads
        self.assertEqual(ev["mask"]["shape"], [3, 128])
        cute = run(body, family="cute")
        self.assertEqual(
            len(events(cute, "gload")), 3
        )  # per thread, in one thread loop
        self.assertEqual(len([s for s in cute.roles[0]["body"] if s["op"] == "for"]), 1)

    def test_metadata_load(self):
        f = run(
            """
    mov.u32 %r1, %tid.x;
    setp.eq.s32 %p9, %r1, 0;
    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd4, [k_param_1];
    ld.global.u32 %r3, [%rd4+4];
    mul.wide.s32 %rd2, %r3, 4;
    add.s64 %rd3, %rd1, %rd2;
    @%p9 st.global.b32 [%rd3], %r3;
    ret;
""",
            ptrs=("x", "seg"),
            scalars=(),
        )
        self.assertTrue(f.tensors["seg"]["metadata"])
        self.assertFalse(f.tensors["x"]["metadata"])
        self.assertEqual(f.written, ["x"])
        self.assertIn("'e': 'load'", str(f.roles[0]["body"]))
        self.assertEqual(kir_errors(f), [])

    def test_cp_async(self):
        f = run(
            PRE
            + """
    mov.u32 %r3, smem;
    @%p9 cp.async.cg.shared.global [%r3], [%rd1], 16, %r2;
    @%p9 cp.async.cg.shared.global.L2::cache_hint [%r3], [%rd1+16], 16, %rd1;
    @%p9 cp.async.ca.shared.global.L2::cache_hint [%r3], [%rd1+32], 16, %r2, %rd1;
    ret;
"""
        )
        a, b, c = events(f, "gload")
        self.assertEqual(a["offsets"]["shape"], [4])
        # %rd1 is the cache policy, not a src-size (R2); with both, %r2 is the src-size
        self.assertEqual(b["offsets"]["shape"], [4])
        self.assertEqual(c["offsets"]["shape"], [4])
        asserts = [s for s in statements(f.roles[0]["body"]) if s["op"] == "assert"]
        self.assertEqual(len(asserts), 2)  # src-size is 0 or 16, for a and c only
        self.assertEqual(kir_errors(f), [])

    def test_atomics(self):
        f = run(
            PRE
            + """
    @%p9 red.global.add.u32 [%rd1], 1;
    @%p9 atom.global.exch.b32 %r3, [%rd1+4], 1;
    @%p9 atom.global.add.u32 %r4, [%rd1+8], %r3;
    ret;
"""
        )
        self.assertEqual([e["ev"] for e in events(f)], ["gatomic", "gstore", "gatomic"])

    def test_atomic_orderings_scopes_and_vectors(self):
        # Triton writes a .sem and .scope on every atomic, inline asm in either order;
        # red.async .global names no mbarrier. A vector atomic covers its elements (one
        # was a single element before) and every result register is Data.
        f = run(
            PRE
            + """
    @%p9 atom.global.gpu.acq_rel.add.f32 %f1, [%rd1], %f2;
    @%p9 red.relaxed.gpu.global.add.u32 [%rd1+4], 1;
    @%p9 atom.global.release.gpu.exch.b32 %r3, [%rd1+8], 1;
    @%p9 red.global.cta.relaxed.or.b32 [%rd1+12], 1;
    @%p9 red.async.release.gpu.global.add.s32 [%rd1+16], 1;
    @%p9 atom.max.acquire.gpu.u32 %r4, [%rd1+20], 1;
    @%p9 red.global.gpu.relaxed.add.noftz.v8.bf16 [%rd1+32], {%r5, %r5, %r5, %r5, %r5, %r5, %r5, %r5};
    @%p9 atom.global.gpu.acq_rel.add.v2.f32 {%r6, %r7}, [%rd1+64], {%r5, %r5};
    ret;
""",
            elem={"x": 2},
        )
        self.assertEqual(f.refusals, [])
        evs = events(f)  # (consecutive ones of a kind merge)
        self.assertEqual([e["ev"] for e in evs].count("gstore"), 1)
        self.assertEqual([e["offsets"].get("shape") for e in evs[-2:]], [[8], [4]])
        mov = "    mul.wide.u32 %rd2, %r7, 4;\n    add.s64 %rd3, %rd1, %rd2;\n"
        f = run(PRE + "    atom.global.gpu.acq_rel.add.v2.f32 {%r6, %r7}, [%rd1], {%r5, %r5};\n"
                + mov + "    @%p9 st.global.b32 [%rd3], %r1;\n    ret;\n")  # fmt: skip
        self.assertIn("data value from atom.global", f.refusals[0]["msg"])

    def test_generic_atomic_on_shared(self):
        # A generic atomic through the generic address of shared memory is shared data.
        f = run(
            PRE
            + """
    mov.u32 %r3, smem;
    cvt.u64.u32 %rd2, %r3;
    cvta.shared.u64 %rd3, %rd2;
    @%p9 atom.add.relaxed.gpu.u32 %r4, [%rd3+8], 1;
    @%p9 atom.add.relaxed.gpu.u32 %r5, [%rd1+8], 1;
    ret;
"""
        )
        (ev,) = events(f)
        self.assertEqual(ev["ev"], "gatomic")
        self.assertEqual(f.refusals, [])
        # its result is Data: refused where it reaches an address
        f = run(PRE + SHARED + "    atom.add.relaxed.gpu.u32 %r4, [%rd5+8], 1;\n"
                "    mul.wide.u32 %rd6, %r4, 4;\n    add.s64 %rd7, %rd1, %rd6;\n"
                "    @%p9 st.global.b32 [%rd7], %r1;\n    ret;\n")  # fmt: skip
        self.assertIn("data value from atom.add.relaxed.gpu.u32", f.refusals[0]["msg"])

    def test_shared_atomics(self):
        # atom/red.shared (.shared::cta too) are shared memory: no event; an atom's
        # result is Data, refused where it reaches an address
        f = run(
            PRE
            + """
    mov.u32 %r3, smem;
    @%p9 atom.shared.add.u32 %r4, [%r3+8], 1;
    atom.shared::cta.cas.b32 %r5, [%r3+12], 0, %r1;
    red.shared.max.s32 [%r3+16], %r1;
    @%p9 red.shared::cta.add.f32 [%r3+20], 0f3F800000;
    @%p9 st.global.b32 [%rd1], %r1;
    ret;
"""
        )
        self.assertEqual(f.refusals, [])
        self.assertEqual([e["ev"] for e in events(f)], ["gstore"])
        f = run(PRE + """
    mov.u32 %r3, smem;
    atom.shared.add.u32 %r4, [%r3+8], 1;
    mul.wide.u32 %rd4, %r4, 4;
    add.s64 %rd5, %rd1, %rd4;
    @%p9 st.global.b32 [%rd5], %r1;
    ret;
""")  # fmt: skip
        self.assertIn("data value from atom.shared.add.u32", f.refusals[0]["msg"])

    def test_strong_loads_are_data(self):
        # A strong load (.volatile, .relaxed/.acquire.<scope>, explicitly global or
        # generic; also .cv) reads what another thread may write while the kernel runs:
        # its value is Data (still a gload), refused where it reaches an address or a
        # bound, never a metadata read; a weak one (.cg, .nc, .weak) stays a metadata
        # read.
        head = """
    mov.u32 %r1, %tid.x;
    setp.eq.s32 %p9, %r1, 0;
    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd4, [k_param_1];
    LOAD %r3, [%rd4+4];
"""
        addr = """
    mul.wide.s32 %rd2, %r3, 4;
    add.s64 %rd3, %rd1, %rd2;
    @%p9 st.global.b32 [%rd3], %r1;
    ret;
"""
        bound = """
    setp.lt.s32 %p1, %r1, %r3;
    @%p1 st.global.b32 [%rd1], %r1;
    ret;
"""
        kw = {"ptrs": ("x", "seg"), "scalars": ()}
        strong = "(a strong load: another thread may write it) reaches a global address"
        for ld, use, want in (
            ("ld.volatile.global.u32", addr, strong),
            ("ld.global.acquire.gpu.u32", bound, "reaches the predicate of an event"),
            ("ld.relaxed.gpu.global.b32", addr, strong),
            ("ld.acquire.sys.u32", bound, "reaches the predicate of an event"),
            ("ld.volatile.u32", addr, strong),
            ("ld.global.cv.u32", addr, strong),  # cache operator "fetch again"
        ):
            with self.subTest(ld):
                f = run(head.replace("LOAD", ld) + use, **kw)
                self.assertIn(want, f.refusals[0]["msg"])
                if use is addr:
                    self.assertIn(f"data value from {ld} of seg", f.refusals[0]["msg"])
        weak = ("ld.global.cg.u32", "ld.global.nc.u32", "ld.weak.global.u32")
        for ld, use in itertools.product(weak, (addr, bound)):
            with self.subTest(ld):
                f = run(head.replace("LOAD", ld) + use, **kw)
                self.assertEqual(f.refusals, [])
                self.assertTrue(f.tensors["seg"]["metadata"])
                self.assertEqual([e["ev"] for e in events(f)], ["gload", "gstore"])
        # used for data only, a strong load costs nothing
        f = run(head.replace("LOAD", "ld.volatile.global.u32")
                + "    @%p9 st.global.b32 [%rd1], %r3;\n    ret;\n", **kw)  # fmt: skip
        self.assertEqual(f.refusals, [])
        self.assertEqual([e["ev"] for e in events(f)], ["gload", "gstore"])

    def test_cache_operators_and_policies(self):
        # Cache operators, .volatile, .relaxed/.acquire.<scope> and an L2 cache policy
        # (createpolicy, the .L2::cache_hint operand) change no event; an integer ld
        # through .nc is still a metadata load where its value reaches a root.
        f = run(
            """
    mov.u32 %r1, %tid.x;
    setp.eq.s32 %p9, %r1, 0;
    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd4, [k_param_1];
    createpolicy.fractional.L2::evict_last.b64 %rd9, 1.0;
    ld.global.nc.L1::evict_last.L2::cache_hint.b32 %r3, [%rd4+4], %rd9;
    ld.volatile.global.b32 %r4, [%rd1];
    ld.relaxed.gpu.global.b32 %r5, [%rd1+4];
    ld.global.acquire.gpu.b32 %r6, [%rd1+8];
    ld.global.cg.v2.b32 {%r7, %r8}, [%rd1+16];
    mul.wide.s32 %rd2, %r3, 4;
    add.s64 %rd3, %rd1, %rd2;
    @%p9 st.global.cs.L2::cache_hint.b32 [%rd3], %r3, %rd9;
    @%p9 st.release.sys.global.b32 [%rd1+32], %r3;
    ret;
""",
            ptrs=("x", "seg"),
            scalars=(),
        )
        self.assertEqual(f.refusals, [])
        self.assertEqual({e["ev"] for e in events(f)}, {"gload", "gstore"})
        self.assertTrue(f.tensors["seg"]["metadata"])
        self.assertEqual(kir_errors(f), [])
        # the policy itself is no value any rule may read
        f = run(PRE + "    createpolicy.fractional.L2::evict_first.b64 %rd9, 1.0;\n"
                "    add.s64 %rd3, %rd1, %rd9;\n    @%p9 st.global.b32 [%rd3], %r1;\n    ret;\n")  # fmt: skip
        self.assertIn("data value from createpolicy", f.refusals[0]["msg"])


SHARED = """
    mov.u32 %r3, smem;
    cvt.u64.u32 %rd4, %r3;
    cvta.shared.u64 %rd5, %rd4;
"""


class DeviceAssert(unittest.TestCase):
    def test_assert_block_after_the_ret(self):
        # Triton's tl.device_assert(t < n): the failing threads branch to a block after
        # the ret that calls __assertfail (a KIR assert under their mask) and traps; the
        # run goes on after the branch as if the trap were not reached (cfg.py)
        f = run(PRE + """
    setp.lt.u32 %p1, %r1, %r2;
    @!%p1 bra $ASSERT;
    mul.wide.u32 %rd2, %r1, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r1;
    ret;
$ASSERT:
    {
    .param .b64 param0;
    st.param::func.b64 [param0], 1;
    call.uni __assertfail, (param0);
    }
    trap;
""")  # fmt: skip
        self.assertEqual((f.refusals, kir_errors(f)), ([], []))
        trace = evaluate(f, {"n": 100})
        self.assertEqual([t[0] for t in trace], ["assert"] * 28 + ["gstore"])
        self.assertEqual(len(trace[-1][2]), 128)
        self.assertEqual([t[0] for t in evaluate(f, {"n": 128})], ["gstore"])


class AddressSpaces(unittest.TestCase):
    def test_cvta_shared_round_trip(self):
        # CuTe DSL 4.6.1: shared::cta -> generic -> shared::cluster, then truncated: the
        # same shared address (an mbarrier slot here). A generic address truncated
        # without cvta.to.shared, or cvta.to.shared of a global one, is not one.
        f = bars(
            SHARED
            + """
    cvta.to.shared::cluster.u64 %rd6, %rd5;
    cvt.u32.u64 %r4, %rd6;
    @%p12 mbarrier.init.shared::cta.b64 [%r4+8], 1;
    cvta.to.shared::cta.u64 %rd7, %rd5;
    cvt.u32.u64 %r5, %rd7;
    @%p12 mbarrier.arrive.shared::cta.b64 _, [%r5+8];
"""
        )
        self.assertEqual(f.refusals, [])
        self.assertEqual([e["ev"] for e in events(f)], ["bar_init", "arrive"])
        self.assertEqual({e["bar"] for e in events(f)}, {"mbar_8"})
        f = bars(SHARED + "    cvt.u32.u64 %r4, %rd5;\n"
                 "    @%p12 mbarrier.init.shared::cta.b64 [%r4+8], 1;\n")  # fmt: skip
        self.assertIn("not a shared-memory symbol", f.refusals[0]["msg"])
        f = bars("    cvta.to.shared.u64 %rd6, %rd1;\n    cvt.u32.u64 %r4, %rd6;\n"
                 "    @%p12 mbarrier.init.shared::cta.b64 [%r4], 1;\n")  # fmt: skip
        self.assertIn("data value from cvta.to.shared", f.refusals[0]["msg"])

    def test_generic_shared_access_is_data(self):
        f = run(PRE + SHARED + "    ld.u32 %r6, [%rd5+4];\n    st.u32 [%rd5+8], %r6;\n"
                "    ret;\n")  # fmt: skip
        self.assertEqual((f.refusals, events(f)), ([], []))
        # an explicit .global access through a shared address (generic, or the window
        # address itself) faults on the hardware: an error, as for atom/red
        for line in ("st.global.u32 [%rd5+8], %r1;", "ld.global.u32 %r6, [%rd5+8];",
                     "st.global.u32 [%rd4+8], %r1;", "atom.global.add.u32 %r6, [%rd5+8], 1;"):  # fmt: skip
            with self.subTest(line):
                f = run(PRE + SHARED + f"    {line}\n    ret;\n")
                self.assertIn(
                    "on shared memory through a global address", f.refusals[0]["msg"]
                )
                self.assertEqual(f.refusals[0]["kind"], "error")


DEPOT = """
    .local .align 8 .b8 __local_depot0[16];
    .reg .b64 %SP;
    .reg .b32 %SPL;
    mov.b32 %SPL, __local_depot0;
    cvt.u64.u32 %SP, %SPL;
    cvta.local.u64 %SP, %SP;
"""


class LocalMemory(unittest.TestCase):
    def test_depot_bytes(self):
        # nvfp4x2_kv.py: two bytes stored through the generic depot address and loaded
        # as one b16 (Data: not one store's bytes), a printf's argument staging: no event.
        f = run(PRE + DEPOT + """
    cvt.u16.u32 %rs1, %r1;
    st.b8 [%SP+1], %rs1;
    st.b8 [%SP], %rs1;
    ld.b16 %rs2, [%SP];
    st.local.v2.b32 [%SPL+8], {%r1, %r2};
    ret;
""")  # fmt: skip
        self.assertEqual((f.refusals, events(f)), ([], []))

    def test_constant_offsets_forward_within_a_block(self):
        # A value stored at a constant offset reloads exactly later in its basic block
        # (here an element offset); across a block boundary, at another width, after a
        # predicated store or a store at a runtime offset, it is Data.
        tail = """
    mul.wide.s32 %rd2, %r9, 4;
    add.s64 %rd3, %rd1, %rd2;
    @%p9 st.global.b32 [%rd3], %r1;
    ret;
"""
        store = "    st.local.b32 [%SPL+4], %r2;\n"
        load = "    ld.local.b32 %r9, [%SPL+4];\n"
        f = run(PRE + DEPOT + store + load + tail)
        self.assertEqual(f.refusals, [])
        (ev,) = events(f)
        self.assertIn("'name': 'n'", offset_text(ev))
        self.assertEqual(kir_errors(f), [])
        for between in ("NEXT:\n", "    @%p9 st.local.b32 [%SPL+4], %r1;\n",
                        "    add.s32 %r8, %SPL, %r1;\n    st.local.b8 [%r8], %r1;\n"):  # fmt: skip
            f = run(PRE + DEPOT + store + between + load + tail)
            self.assertIn("data value from ld.local", f.refusals[0]["msg"])
        f = run(PRE + DEPOT + store + "    ld.local.b16 %rs9, [%SPL+4];\n"
                "    cvt.u32.u16 %r9, %rs9;\n" + tail)  # fmt: skip
        self.assertIn("data value from ld.local", f.refusals[0]["msg"])

    def test_stores_stay_inside_the_depot(self):
        def asserts(f):
            return [s for s in statements(f.roles[0]["body"]) if s["op"] == "assert"]

        f = run(PRE + DEPOT + "    st.local.b32 [%SPL+12], %r1;\n    ret;\n")
        self.assertEqual((f.refusals, asserts(f)), ([], []))
        f = run(PRE + DEPOT + "    st.local.b32 [%SPL+16], %r1;\n    ret;\n")
        (a,) = asserts(f)
        self.assertEqual(
            (a["cond"]["v"], a["msg"]), (0, "st.local inside __local_depot0")
        )
        f = run(PRE + DEPOT + "    and.b32 %r8, %r1, 7;\n    shl.b32 %r8, %r8, 2;\n"
                "    add.s32 %r8, %SPL, %r8;\n    st.local.b32 [%r8], %r1;\n    ret;\n")  # fmt: skip
        self.assertEqual(f.refusals, [])
        self.assertTrue(asserts(f))  # per thread
        self.assertEqual(kir_errors(f), [])
        f = run(PRE + DEPOT + "    ld.global.b32 %r8, [%SP];\n    ret;\n")
        self.assertIn("global access through L:", f.refusals[0]["msg"])
        f = run(PRE + "    st.local.b32 [%rd1], %r1;\n    ret;\n")
        self.assertIn("local depot", f.refusals[0]["msg"])


def bars(body, family="ttgir", **pol):
    return run(
        PRE
        + """
    mov.u32 %r10, smem;
    elect.sync %r11|%p10, -1;
    shr.u32 %r12, %r1, 5;
    setp.eq.s32 %p11, %r12, 0;
    and.pred %p12, %p11, %p10;
"""
        + body
        + "\n    ret;\n",
        family=family,
        **pol,
    )


class Barriers(unittest.TestCase):
    def test_events_and_arrays(self):
        f = bars("""
    .loc 1 3 0
    @%p12 mbarrier.init.shared::cta.b64 [%r10+8], 1;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+16], 1;
    .loc 1 4 0
    @%p12 mbarrier.init.shared::cta.b64 [%r10+24], 1;
    @%p9 mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r10+16], 128;
    @%p12 mbarrier.arrive.shared::cta.b64 _, [%r10+24], 2;
    mov.b32 %r13, 0;
{
    .reg .pred P1;
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 P1, [%r10+24], %r13;
    @!P1 bra.uni WAIT;
}
""")
        self.assertEqual(
            [(b["name"], b["num"]["v"]) for b in f.barriers],
            [("mbar_8", 2), ("mbar_24", 1)],
        )
        evs = [(e["ev"], e["bar"], e["idx"]["v"]) for e in events(f)]
        self.assertEqual(evs, [("bar_init", "mbar_8", 0), ("bar_init", "mbar_8", 1), ("bar_init", "mbar_24", 0),
                               ("expect_tx", "mbar_8", 1), ("arrive", "mbar_8", 1), ("arrive", "mbar_24", 0),
                               ("wait", "mbar_24", 0)])  # fmt: skip
        arrive = events(f, "arrive")[1]
        self.assertEqual(arrive["n"], 2)
        self.assertEqual(kir_errors(f), [])

    def test_threads_per_issue_counts_thread_loop_arrives(self):
        # CuTe policy: Lean multiplies every arrive by threads_per_issue, so an arrive
        # counted per thread in a thread loop (lanes known at runtime only) keeps it 1.
        body = """
    @%p12 mbarrier.init.shared::cta.b64 [%r10], 128;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+8], 64;
    mbarrier.arrive.shared::cta.b64 _, [%r10];
    setp.lt.s32 %p1, %r1, %r2;
    @%p1 mbarrier.arrive.shared::cta.b64 _, [%r10+8];
"""
        f = bars(body, family="cute")
        self.assertEqual(f.roles[0]["threads_per_issue"], 1)
        static, looped = events(f, "arrive")
        self.assertEqual((static["n"], looped.get("n", 1)), (128, 1))
        f = bars(body.replace("@%p1 ", ""), family="cute")  # both static: the gcd
        self.assertEqual(f.roles[0]["threads_per_issue"], 128)
        self.assertEqual(kir_errors(f), [])

    def test_ast_names(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "k.py").write_text(
                "x = 1\nfulls = tlx.alloc_barriers(2, 1)\nb = f(\n"
            )
            f = run(
                PRE
                + """
    mov.u32 %r10, smem;
    .loc 1 2 0
    @%p9 mbarrier.init.shared::cta.b64 [%r10], 1;
    .loc 1 3 0
    @%p9 mbarrier.init.shared::cta.b64 [%r10+16], 1;
    ret;
""",
                src=d,
            )
        self.assertEqual([b["name"] for b in f.barriers], ["fulls", "mbar_16"])
        self.assertEqual(f.roles[0]["body"][-1]["loc"], "fbcode/k.py:3")
        cute = bars("    @%p12 mbarrier.init.shared::cta.b64 [%r10], 1;", family="cute")
        self.assertEqual(cute.barriers[0]["name"], "mbar_0")

    def test_split_by_uses_under_the_cute_policy(self):
        body = """
    @%p12 mbarrier.init.shared::cta.b64 [%r10], 1;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+8], 1;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+16], 1;
    @%p9 mbarrier.arrive.shared::cta.b64 _, [%r10+16];
    and.b32 %r14, %r2, 1;
    shl.b32 %r15, %r14, 3;
    add.s32 %r16, %r10, %r15;
    @%p9 mbarrier.arrive.shared::cta.b64 _, [%r16];
"""
        self.assertEqual([b["num"]["v"] for b in bars(body).barriers], [3])
        cute = bars(body, family="cute")
        self.assertEqual(
            [(b["name"], b["num"]["v"]) for b in cute.barriers],
            [("mbar_0", 2), ("mbar_16", 1)],
        )
        self.assertEqual(kir_errors(cute), [])

    def test_loop_carried_slot_address(self):
        # LSR: the address of slot blk is smem + 64 + 8*blk, the loop adds 8 per trip;
        # the slot keeps smem + 64 outside, its trailing zeros prove idx = (addr - 64)/8.
        f = bars("""
    @%p12 mbarrier.init.shared::cta.b64 [%r10+64], 1;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+72], 1;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+80], 1;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+88], 1;
    add.s32 %r20, %r10, 64;
    mov.b32 %r21, 0;
LOOP:
    @%p9 mbarrier.arrive.shared::cta.b64 _, [%r20];
    add.s32 %r20, %r20, 8;
    add.s32 %r21, %r21, 1;
    setp.lt.s32 %p13, %r21, %r2;
    @%p13 bra LOOP;
""")
        self.assertEqual(f.refusals, [])
        (arrive,) = events(f, "arrive")
        self.assertEqual(arrive["bar"], "mbar_64")
        self.assertEqual(arrive["idx"]["o"], "sdiv")  # the slot variable / 8, exact
        self.assertEqual(kir_errors(f), [])

    def test_refusals(self):
        f = bars("""
    shl.b32 %r14, %r2, 3;
    add.s32 %r15, %r10, %r14;
    @%p12 mbarrier.init.shared::cta.b64 [%r15], 1;
""")
        self.assertIn("not a constant of the kernel", f.refusals[0]["msg"])
        f = bars("""
    @%p12 mbarrier.init.shared::cta.b64 [%r10], 1;
    add.s32 %r15, %r10, %r2;
    @%p9 mbarrier.arrive.shared::cta.b64 _, [%r15];
""")
        self.assertIn("multiple of 8", f.refusals[0]["msg"])
        f = bars("""
    @%p12 mbarrier.init.shared::cta.b64 [%r10], 1;
    @%p9 mbarrier.arrive.shared::cta.b64 _, [%r10+64];
""")
        self.assertIn("not an initialized mbarrier", f.refusals[0]["msg"])


if __name__ == "__main__":
    unittest.main()
