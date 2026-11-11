# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the PTX core's memory model (frontend/ptx/mem.py) through translate().

    python3 -I tests/ptx/test_mem.py      (or the buck target :test_mem)

Global accesses: provenance from a pointer parameter to element offsets (byte offsets
divided exactly), a byte offset that is no multiple of the element refused, a vector
access as consecutive elements, a sub-element access refused, a generic load of shared
memory as data, an address of unknown provenance refused, unrolled loads of one
instruction family merged into one event; metadata loads (value in an address: load
node, metadata tensor); cp.async with src-size and with a cache-hint operand (R2); atom/red
(gatomic, exch: gstore). mbarriers: init/arrive/expect_tx/wait resolved to arrays; runs
split by init .loc (Triton) and by uses (CuTe policy); the AST name of the init's line;
an arrive count; threads_per_issue (CuTe gcd) with an arrive counted in a thread loop;
a loop-carried slot address keeping its constant part (known trailing
zeros prove the index exact); a non-constant init and an unresolvable offset refused.
"""

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

from .fixtures.kernels import events, kir_errors, run, statements  # noqa: E402

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
