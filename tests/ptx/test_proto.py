# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of protocol regions (frontend/ptx/proto.py), tagged words and ring remainders under
policy gsync (frontend/ptx/gsev.py):

    python3 -I tests/ptx/test_proto.py      (or the buck target :test_proto)

A team join on a word shared by the 3 CTAs of a triple (lane 0's fetch-add broadcast by
shfl.sync, a poll loop that times out after 4 polls, a CAS that sets the abort bit, two exits)
in two orders of its tests: the count first (the abort bit missed: some CTAs of a triple may
form while another aborts) and the abort bit first (all or none form). The region is found
(one entry, its exits, live-ins), cut out (the rewritten function passes the CFG shape check),
its interleavings explored (the reachable outcome vectors), and refused where it is not one
word's morally strong protocol. The translation reads each CTA's exit from `__proto0` and
issues the word's footprint. Tagged words: a uniform tag spin is one `gwait` tag_eq binding
the value half; a tagged store with a data value half is a store of `tag << 32` with
`data_lo`; an exit condition that reads the value half is refused. A relay ring slot written as
`X - ((X + (X lshr 31)) & -2)` (C's `%` by 2) is keyed by X. A work cursor's fetch-add whose
old value is a work index is a `gclaim` under policy claims "representative", else refused.
"""

import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

if __package__ in (None, ""):
    sys.dont_write_bytecode = True
    _root = ModuleType("lkv_root")
    _root.__path__ = [str(Path(__file__).resolve().parents[2])]
    sys.modules["lkv_root"] = _root
    __package__ = "lkv_root.tests.ptx"  # noqa: A001

from ...frontend.ptx import cfg as C, parse, proto  # noqa: E402
from .fixtures.kernels import events, kernel, kir_errors, run  # noqa: E402

JOIN = """
    ld.param.u64 %rd1, [k_param_0];
    mov.u64 %rd2, %rd1;
    ld.param.u64 %rd3, [k_param_1];
    mov.u64 %rd4, %rd3;
    mov.u32 %r1, %ctaid.x;
    mul.hi.u32 %r2, %r1, 1431655766;
    mul.wide.u32 %rd5, %r2, 4;
    add.s64 %rd6, %rd2, %rd5;
    mov.b32 %r3, 1;
    // begin inline asm
    {
    .reg .pred p; .reg .b32 l, v;
    mov.u32 l, %laneid;
    setp.eq.u32 p, l, 0;
    mov.b32 v, 0;
    @p atom.relaxed.gpu.global.add.u32 v, [%rd6], %r3;
    shfl.sync.idx.b32 %r4, v, 0, 31, -1;
    }
    // end inline asm
    add.s32 %r5, %r4, 1;
    mov.b32 %r6, 0;
    and.b32 %r12, %r5, 256;
    setp.ne.s32 %p5, %r12, 0;
    @%p5 bra $ABORT;
$POLL:
TESTS
    setp.ge.s32 %p3, %r6, 4;
    @%p3 bra $CAS;
    // begin inline asm
    {
    nanosleep.u32 64;
    ld.relaxed.gpu.global.b32 %r5, [%rd6];
    }
    // end inline asm
    add.s32 %r6, %r6, 1;
    bra.uni $POLL;
$CAS:
    or.b32 %r9, %r5, 256;
    // begin inline asm
    {
    .reg .pred p; .reg .b32 l, v;
    mov.u32 l, %laneid;
    setp.eq.u32 p, l, 0;
    mov.b32 v, 0;
    @p atom.relaxed.gpu.global.cas.b32 v, [%rd6], %r5, %r9;
    shfl.sync.idx.b32 %r10, v, 0, 31, -1;
    }
    // end inline asm
    setp.eq.s32 %p4, %r10, %r5;
    mov.b32 %r5, %r10;
    @%p4 bra $ABORT;
    bra.uni $POLL;
$FORMED:
    mul.wide.u32 %rd7, %r1, 4;
    add.s64 %rd8, %rd4, %rd7;
    st.global.u32 [%rd8], %r3;
$ABORT:
    ret;
"""

COUNT = """    and.b32 %r7, %r5, 255;
    setp.eq.s32 %p1, %r7, 3;
    @%p1 bra $FORMED;
    and.b32 %r8, %r5, 256;
    setp.ne.s32 %p2, %r8, 0;
    @%p2 bra $ABORT;"""
ABORT_FIRST = """    and.b32 %r8, %r5, 256;
    setp.ne.s32 %p2, %r8, 0;
    @%p2 bra $ABORT;
    and.b32 %r7, %r5, 255;
    setp.eq.s32 %p1, %r7, 3;
    @%p1 bra $FORMED;"""

PTRS = ("team", "out")
PARAMS = [{"ptx": f"k_param_{i}", "kind": "ptr", "name": p, "size": 8, "fields": {}}
          for i, p in enumerate(PTRS)]  # fmt: skip


def region(tests=COUNT, body=JOIN):
    fn = parse.parse(kernel(body.replace("TESTS", tests), PTRS, (), 32)).function("k")
    cfg = C.build(fn, check_exits=False)
    roles, _ = C.roles(cfg, "ttgir")
    (r,) = proto.find(fn, cfg, roles, ["team"], PARAMS)
    return fn, r


def outcomes(r, n=3):
    group = [proto.Agent(c, (c, 0, 0), {r.addr[0]: proto.Ptr("team", 0), "%r3": 1}, ("team", 0))
             for c in range(n)]  # fmt: skip
    ex = proto.explore(r, group, 0, {"grid": (n, 1, 1), "block": (32, 1, 1), "warp": 0})
    return {"".join(str(e) for e, _ in v) for v in ex.outcomes}, ex


class Region(unittest.TestCase):
    def test_find(self):
        fn, r = region()
        self.assertEqual(r.tensor, "team")
        self.assertEqual(r.addr, ("%rd6", 0))
        self.assertEqual(r.width, 32)
        self.assertEqual(r.live_in, ["%r3", "%rd6"])
        self.assertEqual(r.live_out, [])
        self.assertEqual(r.exits, ["$ABORT", "$FORMED"])
        # the region starts at the inline asm block of the first access
        self.assertEqual(r.blocks[r.entry][0].opcode, "mov.u32")

    def test_rewrite_passes_the_shape_check(self):
        fn, r = region()
        with self.assertRaises(C.CfgError):
            C.build(fn)  # the poll loop has three exits
        fn2 = proto.rewrite(fn, [r])
        C.build(fn2)
        ops = [i.opcode for i in fn2.instructions()]
        self.assertIn("lkv.outcome.u32", ops)
        self.assertNotIn("atom.relaxed.gpu.global.cas.b32", ops)


class Explore(unittest.TestCase):
    def test_count_first_lets_one_cta_form_alone(self):
        _fn, r = region(COUNT)
        got, ex = outcomes(r)
        # exit 0 aborted, 1 formed: none, all, and one CTA alone (it saw the third
        # registration after the aborter set the abort bit)
        self.assertEqual(got, {"000", "111", "100", "010", "001"})
        self.assertTrue(ex.symmetric)
        self.assertEqual(ex.stuck, [])

    def test_abort_first_agrees(self):
        _fn, r = region(ABORT_FIRST)
        got, _ex = outcomes(r)
        # all or none
        self.assertEqual(r.exits, ["$ABORT", "$FORMED"])
        self.assertEqual(got, {"000", "111"})

    def test_a_lone_cta_never_forms(self):
        _fn, r = region(COUNT)
        got, _ex = outcomes(r, 1)
        self.assertEqual(got, {"0"})

    def test_state_budget(self):
        _fn, r = region(COUNT)
        old = proto.MAX_STATES
        proto.MAX_STATES = 10
        try:
            with self.assertRaises(proto.ProtocolError):
                outcomes(r)
        finally:
            proto.MAX_STATES = old


class Refusals(unittest.TestCase):
    def check(self, body, why, tests=COUNT):
        with self.assertRaises(proto.ProtocolError) as cm:
            region(tests, body)
        self.assertIn(why, str(cm.exception))

    def test_weak_load_of_the_word(self):
        self.check(JOIN.replace("ld.relaxed.gpu.global.b32 %r5", "ld.global.b32 %r5"),
                   "not an atomic or a strong")  # fmt: skip

    def test_access_outside_the_region(self):
        body = JOIN.replace(
            "    ret;\n", "    ld.relaxed.gpu.global.b32 %r12, [%rd2];\n    ret;\n"
        )
        self.check(body, "protocol region")

    def test_other_memory_in_the_region(self):
        body = JOIN.replace("    add.s32 %r6, %r6, 1;\n",
                            "    add.s32 %r6, %r6, 1;\n    st.global.u32 [%rd4], %r6;\n")  # fmt: skip
        self.check(body, "in a protocol region")

    def test_access_through_a_loaded_pointer(self):
        # a pointer read from memory may point at the word: an access outside the region
        body = JOIN.replace("    ret;\n", "    ld.global.u64 %rd9, [%rd4];\n"
                            "    ld.relaxed.gpu.global.b32 %r13, [%rd9];\n    ret;\n")  # fmt: skip
        self.check(body, "protocol region")

    def test_predicate_live_out(self):
        body = JOIN.replace(
            "    st.global.u32 [%rd8], %r3;\n", "    @%p1 st.global.u32 [%rd8], %r3;\n"
        )
        self.check(body, "live-out register %p1 is .pred, not a 32-bit integer")

    def test_provenance_of_loop_carried_registers(self):
        # a least fixpoint: the order the registers are asked in does not matter
        body = """
    ld.param.u64 %rd20, [k_param_0];
    mov.u32 %r1, 0;
$L:
    ld.global.b32 %r2, [%rd20];
    add.s64 %rd21, %rd20, 4;
    mov.u64 %rd20, %rd21;
    add.s32 %r1, %r1, 1;
    setp.lt.s32 %p1, %r1, 4;
    @%p1 bra $L;
    ret;
"""
        fn = parse.parse(kernel(body, PTRS, (), 32)).function("k")
        for order in (("%rd20", "%rd21"), ("%rd21", "%rd20")):
            prov = proto._Provenance(fn)
            self.assertEqual([prov.params(r) for r in order], [{"k_param_0"}] * 2)

    def test_divergent_branch(self):
        body = JOIN.replace("    setp.ge.s32 %p3, %r6, 4;",
                            "    mov.u32 %r12, %laneid;\n    setp.ge.s32 %p3, %r12, 4;")  # fmt: skip
        _fn, r = region(COUNT, body)
        with self.assertRaises(proto.ProtocolError) as cm:
            outcomes(r)
        self.assertIn("lanes take differently", str(cm.exception))


class Translation(unittest.TestCase):
    def facts(self, tests=COUNT):
        return run(JOIN.replace("TESTS", tests), family="ttgir", ptrs=PTRS, scalars=(), threads=32,
                   gsync=True, protocols=[{"tensor": "team"}])  # fmt: skip

    def test_outcome_and_footprint(self):
        f = self.facts()
        self.assertEqual(f.refusals, [])
        ((r, rec),) = f.protocols
        self.assertEqual(sorted(rec.live_in), ["%r3", "%rd6"])
        self.assertEqual(rec.conds, [[]])
        self.assertEqual(
            len([e for e in events(f, "gatomic") if e["tensor"] == "team"]), 1
        )
        loads = [
            s
            for role in f.roles
            for s in _lets(role["body"])
            if "__proto0" in str(s["expr"])
        ]
        self.assertEqual(len(loads), 1)
        self.assertNotIn("team", f.sync)

    def test_without_the_policy_the_loop_shape_refuses(self):
        f = run(JOIN.replace("TESTS", COUNT), family="ttgir", ptrs=PTRS, scalars=(), threads=32,
                gsync=True)  # fmt: skip
        self.assertTrue(f.refusals)

    def test_scenario_inputs(self):
        # the TTGIR assembly's inputs: one per product of the groups' reachable outcome vectors
        from ...frontend.ttgir import to_kir

        base = {
            "name": "in",
            "grid": [6, 1, 1],
            "params": {},
            "metadata": {},
            "shapes": {},
        }
        for tests, n, vec in ((COUNT, 25, "100-111"), (ABORT_FIRST, 4, "000-111")):
            f = self.facts(tests)
            got = to_kir._with_protocols(
                base, f.protocols, {"protocols": [{"tensor": "team"}]}, (32, 1, 1)
            )
            self.assertEqual(len(got), n)
            one = next(x for x in got if x["name"] == f"in@proto:team:{vec}")
            self.assertEqual(
                one["metadata"]["__proto0"], [int(c) for c in vec.replace("-", "")]
            )
            self.assertEqual(one["params"]["__proto0.shape.0"], 6)
            self.assertEqual(one["sync_groups"], [[0, 1, 2], [3, 4, 5]])
        with mock.patch.object(to_kir, "MAX_PROTOCOL_INPUTS", 10):
            with self.assertRaisesRegex(
                to_kir.KirError, "25 outcome vectors .* more than 10"
            ):
                to_kir._with_protocols(base, self.facts().protocols,
                                       {"protocols": [{"tensor": "team"}]}, (32, 1, 1))  # fmt: skip

    def test_agents_from_the_record(self):
        f = self.facts()
        ((r, rec),) = f.protocols
        groups = proto.agents(rec, (6, 1, 1), {}, {})
        self.assertEqual([[a.cta for a in g] for g in groups], [[0, 1, 2], [3, 4, 5]])
        self.assertEqual([g[0].word for g in groups], [("team", 0), ("team", 1)])
        self.assertEqual(groups[1][0].regs["%rd6"], proto.Ptr("team", 4))


def _lets(stmts):
    for s in stmts:
        if s.get("op") == "let":
            yield s
        for k in ("then", "else", "body"):
            yield from _lets(s.get(k, []))


TAKE = """
    ld.param.u64 %rd1, [k_param_0];
    mov.u64 %rd2, %rd1;
    ld.param.u64 %rd3, [k_param_1];
    mov.u64 %rd4, %rd3;
    mov.u32 %r1, %ctaid.x;
    setp.ne.s32 %p1, %r1, 0;
    @%p1 bra $TAKE;
    mov.b32 %r2, 7;
    mov.b32 %r3, 1;
    // begin inline asm
    {
    .reg .b64 d;
    mov.b64 d, {%r2, %r3};
    st.relaxed.gpu.global.b64 [%rd2], d;
    }
    // end inline asm
    bra.uni $END;
$TAKE:
    mov.b32 %r4, 1;
    // begin inline asm
    {
    .reg .pred p; .reg .b64 d; .reg .b32 t;
    XW1:
    ld.relaxed.gpu.global.b64 d, [%rd2];
    mov.b64 {%r5, t}, d;
    setp.ne.s32 p, t, %r4;
    @p bra XW1;
    }
    // end inline asm
    mul.wide.u32 %rd5, %r5, 4;
    add.s64 %rd6, %rd4, %rd5;
    st.global.u32 [%rd6], %r1;
$END:
    ret;
"""


def take(body=TAKE):
    return run(body, family="ttgir", ptrs=("mail", "out"), scalars=(), threads=32,
               elem={"mail": 8}, gsync=True)  # fmt: skip


class Tagged(unittest.TestCase):
    def test_uniform_take_binds_the_value(self):
        f = take()
        self.assertEqual(f.refusals, [])
        self.assertEqual(f.sync, {"mail": 64})
        (w,) = events(f, "gwait")
        self.assertEqual((w["w"], w["target"]["w"]), (32, 32))
        (st,) = [e for e in events(f, "gstore") if e["tensor"] == "out"]
        self.assertIn(w["var"], str(st["offsets"]))
        (s,) = events(f, "gsig")
        self.assertEqual(s["value"]["v"], (1 << 32) | 7)
        self.assertEqual(kir_errors(f), [])

    def test_data_value_half(self):
        body = TAKE.replace(
            "    mov.b32 %r2, 7;\n",
            "    mov.b32 %f1, 0f3F800000;\n    mov.b32 %r2, %f1;\n",
        )
        body = body.replace(
            "    mul.wide.u32 %rd5, %r5, 4;\n", "    mul.wide.u32 %rd5, %r1, 4;\n"
        )
        f = take(body)
        self.assertEqual(f.refusals, [])
        (s,) = events(f, "gsig")
        self.assertTrue(s.get("data_lo"))
        self.assertEqual(s["value"]["v"], 1 << 32)

    def test_exit_condition_on_the_value_half(self):
        f = take(TAKE.replace("setp.ne.s32 p, t, %r4;", "setp.ne.s32 p, %r5, %r4;"))
        self.assertTrue(f.refusals)


CLAIM = """
    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd3, [k_param_1];
    mov.u32 %r1, %tid.x;
    setp.eq.s32 %p1, %r1, 0;
$LOOP:
    mov.u32 %r2, 0;
    @%p1 atom.global.add.u32 %r2, [%rd1], 1;
    shfl.sync.idx.b32 %r3, %r2, 0, 31, -1;
    setp.ge.u32 %p2, %r3, 4;
    @%p2 bra $END;
    mul.wide.u32 %rd5, %r3, 4;
    add.s64 %rd6, %rd3, %rd5;
    @%p1 st.global.u32 [%rd6], %r3;
    bra.uni $LOOP;
$END:
    ret;
"""


class Claims(unittest.TestCase):
    def claim(self, **pol):
        return run(CLAIM, family="ttgir", ptrs=("ctr", "out"), scalars=(), threads=32,
                   gsync=True, **pol)  # fmt: skip

    def test_representative_claim(self):
        f = self.claim(claims="representative")
        self.assertEqual(f.refusals, [])
        self.assertEqual((f.sync, f.representative), ({"ctr": 32}, ["claims"]))
        (c,) = events(f, "gclaim")
        self.assertEqual((c["tensor"], c["value"]["v"], c["w"]), ("ctr", 1, 32))
        (st,) = [e for e in events(f, "gstore") if e["tensor"] == "out"]
        self.assertIn(c["var"], str(st["offsets"]))
        self.assertEqual(kir_errors(f), [])

    def test_claim_without_the_policy_is_refused(self):
        self.assertTrue(self.claim().refusals)


class RingRemainder(unittest.TestCase):
    def test_signed_remainder_slot_is_keyed_by_the_counter(self):
        from ...frontend.ptx import expr as E, gsev

        X = E.var("c", 32)
        r = E.bin("sub", X, E.bin("and", E.bin("add", X, E.bin("lshr", X, E.const(31, 32))),
                                  E.const(-2, 32)))  # fmt: skip
        terms = [
            (1, E.bin("shl", r, E.const(3, 32))),
            (-1, E.bin("shl", r, E.const(2, 32))),
        ]
        self.assertIs(gsev._ring_rem_terms(terms, 2, 4), X)
        self.assertIsNone(gsev._ring_rem_terms(terms, 2, 8))
        self.assertIsNone(gsev._ring_rem_terms(terms[:1], 2, 4))


if __name__ == "__main__":
    unittest.main()
