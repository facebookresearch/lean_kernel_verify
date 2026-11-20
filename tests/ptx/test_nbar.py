# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the P0-6 barrier pass (frontend/ptx/nbar.py) on hand-written PTX.

    python3 -I tests/ptx/test_nbar.py      (or the buck target :test_nbar)

Each kernel's sites (counted `named_bar` events or `nb_open` events with their kind and
certainty), roles and the KIR it makes: a uniform barrier; a barrier only warp 0 reaches
(the role splits, warp 0 alone counts it: Lean finds the count unmet); one only half of
each warp's lanes reach (certain lane divergence); a spin loop on a strong load around a
barrier (certain, kind loop); a counted loop on a parameter (an exact KIR loop); a
producer `bar.arrive` / consumer `bar.sync` pair (split roles); a warp-specialized
kernel (the dispatch's fork barriers emit nothing, padding roles); a barrier id read from
memory (not certain, kind operand); a count no warps meet (certain); the value an inline
spin accepted controlling a loop around a barrier (a published value: an nb_rep region)
against a single strong read doing the same (certain divergence); the roles split from
one role inside a loop on uniform data sharing its region key, also where the loop's
bound is a ballot of per-lane data alike in every warp; a loop bound read from a tensor
with recorded contents (an exact KIR loop over a metadata `load`, without contents a
repeated region, of a tensor the kernel writes undecided); a switch `brx.idx` lowered to
its compare chain (exact `if`s, a role split per warp, an index not provably in range
refused); a class loop that leaves its dead registers unclassified; a cycle entered at
two blocks (made reducible by node splitting), also before a loop whose rotated test
leaves it to a path back to its header (the path copied into the loop); a call (refused,
P0-6 undecided); a
warp-specialized kernel whose default may run no round (the workers' barriers under its
decision, refused on a decision that is data); a nest of loops translated by classes
(the final rounds the enclosing fixed-point rounds drop are skipped: the same KIR, fewer
passes); and the KIRs frontend/kir.py accepts.
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

from ...frontend import kir  # noqa: E402
from ...frontend.ptx import nbar, translate as T  # noqa: E402
from .fixtures.kernels import kernel, SRC  # noqa: E402

WS = """
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
    st.shared.b32 [smem+24], 33554689;
    barrier.sync 1;
    barrier.sync 1;
    st.global.b32 [%rd1], %r1;
    bar.sync 2, 128;
    barrier.sync 1;
    st.shared.b32 [smem+24], 50529027;
    barrier.sync 1;
$RET:
    ret;
$PAD2:
    barrier.sync 1;
    barrier.sync 1;
$LOOP:
    barrier.sync 1;
    mov.b32 %r6, smem;
    add.s32 %r7, %r6, %r3;
    ld.shared.b8 %r5, [%r7+20];
    setp.gt.u32 %p2, %r5, 3;
    @%p2 bra $BYPASS;
    brx.idx %r5, JT;
$P0:
    barrier.sync 1;
    bar.sync 3, 32;
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
"""


# WS whose default may run no round (cfg.py, item rounds): with n < 1 it stores the exit
# state for every worker before one fork barrier and returns; barrier 4 (160 threads) is
# the default's and partition0's on the round side only.
WS_ZERO = (
    WS.replace(
        "    st.shared.b32 [smem+24], 33554689;\n",
        "    ld.param.b32 %r4, [k_param_1];\n    setp.lt.s32 %p3, %r4, 1;\n"
        "    @%p3 bra $NONE;\n    st.shared.b32 [smem+24], 33554689;\n",
    )
    .replace("    bar.sync 2, 128;\n", "    bar.sync 2, 128;\n    bar.sync 4, 160;\n")
    .replace(
        "    st.shared.b32 [smem+24], 50529027;\n    barrier.sync 1;\n$RET:\n",
        "    st.shared.b32 [smem+24], 50529027;\n    barrier.sync 1;\n    bra.uni $RET;\n"
        "$NONE:\n    st.shared.b32 [smem+24], 50529027;\n    barrier.sync 1;\n$RET:\n",
    )
    .replace("    bar.sync 3, 32;\n", "    bar.sync 3, 32;\n    bar.sync 4, 160;\n")
)


def facts(body, threads=128, family="ttgir", meta=()):
    params = [{"ptx": "k_param_0", "kind": "ptr", "name": "x", "size": 8, "fields": {}},
              {"ptx": "k_param_1", "kind": "scalar", "name": "n", "size": 4, "fields": {}}]  # fmt: skip
    pol = T.policy(
        family, src_root=SRC, params=params, tensors={"x": {"elem_bytes": 4}}
    )
    pol["nb_meta"] = frozenset(meta)
    rec = {"block": [threads, 1, 1], "cluster": [1, 1, 1]}
    return nbar.translate(kernel(body, threads=threads), "k", [rec], pol)


def sites(f, status=None):
    return [s for s in f.sites if status in (None, s["status"])]


def doc(f):
    return nbar.document(f, "k", "k.py", "test", [{"name": "i", "params": {"n": 1, "x": 0},
                                                    "grid": [1, 1, 1]}])  # fmt: skip


class NbarTest(unittest.TestCase):
    def ok(self, f):
        self.assertEqual(f.refusals, [])
        self.assertEqual(kir.validate(doc(f)), [])

    def test_uniform(self):
        f = facts("    bar.sync 1, 128;\n    ret;")
        self.ok(f)
        (s,) = sites(f)
        self.assertEqual((s["status"], s["id"], s["count"], s["op"], s["role"]),
                         ("counted", 1, 128, "sync", "default"))  # fmt: skip

    def test_warp_divergent_splits_the_role(self):
        f = facts("""
    mov.u32 %r1, %tid.x;
    setp.lt.u32 %p1, %r1, 32;
    @!%p1 bra $END;
    bar.sync 1, 128;
$END:
    ret;""")
        self.ok(f)
        self.assertEqual([r["name"] for r in f.roles], ["default:w0-0", "default:w1-3"])
        (s,) = sites(f)
        self.assertEqual(
            (s["status"], s["role"], s["warps"]), ("counted", "default:w0-0", 1)
        )

    def test_lane_divergent_is_certain(self):
        f = facts("""
    mov.u32 %r1, %tid.x;
    and.b32 %r2, %r1, 31;
    setp.lt.u32 %p1, %r2, 16;
    @!%p1 bra $END;
    bar.sync 1, 128;
$END:
    ret;""")
        self.ok(f)
        (s,) = sites(f)
        self.assertEqual((s["status"], s["kind"], s["certain"]), ("open", "lane", True))

    def test_spin_loop_is_certain(self):
        f = facts("""
    ld.param.u64 %rd1, [k_param_0];
$LOOP:
    bar.sync 1, 128;
    ld.volatile.global.u32 %r1, [%rd1];
    setp.eq.s32 %p1, %r1, 0;
    @%p1 bra $LOOP;
    ret;""")
        self.ok(f)
        (s,) = sites(f)
        self.assertEqual((s["status"], s["kind"], s["certain"]), ("open", "loop", True))
        self.assertIn("ld.volatile.global.u32", s["why"])

    def test_carry_written_under_a_data_predicate(self):
        # the threads whose predicate (a strong load) is false keep another carry, so the
        # addc that reads it outside the predicate decides the barrier on that data
        f = facts("""
    ld.param.u64 %rd1, [k_param_0];
    ld.volatile.global.u32 %r1, [%rd1];
    setp.eq.s32 %p1, %r1, 0;
    @%p1 add.cc.u32 %r3, -1, 1;
    addc.u32 %r4, 0, 0;
    setp.eq.s32 %p2, %r4, 0;
    @!%p2 bra $SKIP;
    bar.sync 1, 96;
$SKIP:
    ret;""")
        self.ok(f)
        (s,) = sites(f)
        self.assertEqual((s["status"], s["certain"]), ("open", True))
        self.assertIn("ld.volatile.global.u32", s["why"])

    def test_counted_loop_is_exact(self):
        f = facts("""
    ld.param.u32 %r1, [k_param_1];
    mov.u32 %r2, 0;
    setp.lt.s32 %p1, %r1, 1;
    @%p1 bra $END;
$LOOP:
    bar.sync 1, 128;
    add.s32 %r2, %r2, 1;
    setp.lt.s32 %p2, %r2, %r1;
    @%p2 bra $LOOP;
$END:
    ret;""")
        self.ok(f)
        (s,) = sites(f)
        self.assertEqual((s["status"], s["rep"]), ("counted", False))
        ops = {st["op"] for r in f.roles for st in _stmts(r["body"])}
        self.assertTrue(ops & {"for", "while"}, ops)

    def test_producer_consumer(self):
        f = facts("""
    mov.u32 %r1, %tid.x;
    setp.lt.u32 %p1, %r1, 64;
    @%p1 bra $PROD;
    bar.sync 1, 128;
    bra.uni $END;
$PROD:
    bar.arrive 1, 128;
$END:
    ret;""")
        self.ok(f)
        got = sorted((s["role"], s["op"]) for s in sites(f, "counted"))
        self.assertEqual(got, [("default:w0-1", "arrive"), ("default:w2-3", "sync")])

    def test_dispatch_forks_and_padding(self):
        f = facts(WS, threads=256)
        self.ok(f)
        self.assertTrue(f.dispatch)
        roles = {r["name"]: (r["w0"], r["warps"]) for r in f.roles}
        # cfg.py's roles, the padding states' (partition1, partition2) included: each
        # warp in exactly one role
        self.assertEqual(
            roles,
            {"default": (0, 4), "partition0": (6, 1), "partition1": (4, 2),
             "partition2": (7, 1)},
        )  # fmt: skip
        got = sorted((s["role"], s["id"], s["count"]) for s in sites(f, "counted"))
        self.assertEqual(got, [("default", 2, 128), ("partition0", 3, 32)])
        self.assertEqual(sites(f, "open"), [])

    def test_dispatch_without_rounds(self):
        f = facts(WS_ZERO, threads=256)
        self.ok(f)
        got = sorted((s["role"], s["id"], s["count"]) for s in sites(f, "counted"))
        self.assertEqual(got, [("default", 2, 128), ("default", 4, 160),
                               ("partition0", 3, 32), ("partition0", 4, 160)])  # fmt: skip
        # partition0's barriers run under the default's decision (n >= 1), a KIR `if`
        # like the default's own
        body = {r["name"]: r["body"] for r in f.roles}
        (gate,) = [s for s in body["partition0"] if s["op"] == "if"]
        self.assertEqual([x["id"]["v"] for x in _stmts(gate["then"]) if x.get("ev") == "named_bar"],
                         [3, 4])  # fmt: skip
        self.assertFalse(any(x.get("ev") == "named_bar" for x in body["partition0"]))
        (dif,) = [s for s in body["default"] if s["op"] == "if"]
        self.assertEqual(sorted(x["id"]["v"] for side in ("then", "else")
                                for x in _stmts(dif[side]) if x.get("ev") == "named_bar"),
                         [2, 4])  # fmt: skip
        # a decision on data (a strong load) cannot be paired: refused
        g = facts(WS_ZERO.replace("    ld.param.b32 %r4, [k_param_1];\n",
                                  "    ld.relaxed.gpu.global.b32 %r4, [%rd1];\n"), threads=256)  # fmt: skip
        self.assertIn(
            "run a round depends on data", g.refusals[0]["msg"] if g.refusals else ""
        )

    def test_id_from_memory_is_not_certain(self):
        f = facts("""
    ld.param.u64 %rd1, [k_param_0];
    ld.global.u32 %r1, [%rd1];
    bar.sync %r1, 128;
    ret;""")
        self.ok(f)
        (s,) = sites(f)
        self.assertEqual(
            (s["status"], s["kind"], s["certain"]), ("open", "operand", False)
        )

    def test_count_no_warps_meet(self):
        f = facts("    bar.sync 1, 100;\n    ret;")
        self.ok(f)
        (s,) = sites(f)
        self.assertEqual(
            (s["status"], s["kind"], s["certain"]), ("open", "operand", True)
        )

    def test_barrier_without_a_count_says_all(self):
        # warps 2-3 exit, warps 0-1 sync barrier 0: without a thread count the exited
        # threads release it (PTX ISA exit; the KIR says `all`), with count 128 not
        body = """
    mov.u32 %r1, %tid.x;
    setp.ge.u32 %p1, %r1, 64;
    @%p1 bra $X;
    bar.sync 0{};
$X:
    ret;"""
        for count, every in (("", True), (", 128", False)):
            f = facts(body.replace("{}", count))
            self.ok(f)
            bars = [e for r in doc(f)["roles"] for e in _stmts(r["body"])
                    if e.get("ev") == "named_bar"]  # fmt: skip
            self.assertEqual(
                [(e["count"]["v"], "all" in e) for e in bars], [(128, every)]
            )

    def test_published_spin_value(self):
        f = facts("""
    ld.param.u64 %rd1, [k_param_0];
$CHECK:
    ld.acquire.gpu.global.u32 %r1, [%rd1];
    setp.ge.s32 %p1, %r1, 1;
    @!%p1 bra $CHECK;
    mov.u32 %r2, 0;
$LOOP:
    bar.sync 1, 128;
    add.s32 %r2, %r2, 1;
    setp.lt.s32 %p2, %r2, %r1;
    @%p2 bra $LOOP;
    ret;""")
        self.ok(f)
        (s,) = sites(f)
        self.assertEqual((s["status"], s["rep"]), ("counted", True))

    def test_single_strong_read_is_certain(self):
        f = facts("""
    ld.param.u64 %rd1, [k_param_0];
    ld.acquire.gpu.global.u32 %r1, [%rd1];
    mov.u32 %r2, 0;
$LOOP:
    bar.sync 1, 128;
    add.s32 %r2, %r2, 1;
    setp.lt.s32 %p2, %r2, %r1;
    @%p2 bra $LOOP;
    ret;""")
        self.ok(f)
        (s,) = sites(f)
        self.assertEqual((s["status"], s["kind"], s["certain"]), ("open", "loop", True))

    def test_split_roles_share_a_keyed_region(self):
        f = facts("""
    ld.param.u64 %rd1, [k_param_0];
    ld.global.u32 %r1, [%rd1];
    mov.u32 %r2, 0;
    mov.u32 %r3, %tid.x;
    shr.u32 %r4, %r3, 5;
    setp.lt.s32 %p1, %r1, 1;
    @%p1 bra $END;
$LOOP:
    setp.eq.u32 %p2, %r4, 0;
    @!%p2 bra $SKIP;
    bar.sync 2, 32;
$SKIP:
    bar.sync 1, 128;
    add.s32 %r2, %r2, 1;
    setp.lt.s32 %p3, %r2, %r1;
    @%p3 bra $LOOP;
$END:
    ret;""")
        self.ok(f)
        self.assertEqual([r["name"] for r in f.roles], ["default:w0-0", "default:w1-3"])
        keys = _keys(f)
        self.assertEqual(len(keys["default:w0-0"]), 2)  # the If on n < 1 and the loop
        self.assertEqual(keys["default:w0-0"], keys["default:w1-3"])
        self.assertNotIn("", keys["default:w0-0"])

    def test_metadata_bound_is_an_exact_loop(self):
        body = """
    ld.param.u64 %rd1, [k_param_0];
    ld.global.u32 %r1, [%rd1];
    mov.u32 %r2, 0;
    setp.lt.s32 %p1, %r1, 1;
    @%p1 bra $END;
$LOOP:
    bar.sync 1, 128;
    add.s32 %r2, %r2, 1;
    setp.lt.s32 %p2, %r2, %r1;
    @%p2 bra $LOOP;
$END:
    ret;"""
        f = facts(body, meta=["x"])
        self.assertEqual(f.refusals, [])
        self.assertEqual(f.meta, {"x": 4})
        (s,) = sites(f)
        self.assertEqual((s["status"], s["rep"]), ("counted", False))
        loads = [
            n for r in f.roles for n in T._expr_nodes(r["body"]) if n.get("e") == "load"
        ]
        self.assertTrue(loads)
        d = nbar.document(f, "k", "k.py", "test", [{"name": "i", "params": {"x.nb_len": 1},
                                                    "grid": [1, 1, 1], "metadata": {"x": [3]}}],
                          tensors=[{"name": "x", "kind": "global", "elem_bytes": 4,
                                    "shape": [{"e": "var", "w": 64, "name": "x.nb_len"}],
                                    "strides": [{"e": "const", "w": 64, "v": 1}],
                                    "segments": None, "role": "input", "metadata": True}],
                          tensor_params={"x.nb_len": 64})  # fmt: skip
        self.assertEqual(kir.validate(d), [])
        self.assertEqual(d["case"]["inputs"][0]["metadata"], {"x": [3]})
        # without recorded contents the bound is uniform data: a repeated region
        (s,) = sites(facts(body))
        self.assertEqual((s["status"], s["rep"]), ("counted", True))

    def test_a_written_tensor_is_no_metadata(self):
        f = facts(
            """
    ld.param.u64 %rd1, [k_param_0];
    ld.global.u32 %r1, [%rd1];
    st.global.u32 [%rd1+4], %r1;
    mov.u32 %r2, 0;
    setp.lt.s32 %p1, %r1, 1;
    @%p1 bra $END;
$LOOP:
    bar.sync 1, 128;
    add.s32 %r2, %r2, 1;
    setp.lt.s32 %p2, %r2, %r1;
    @%p2 bra $LOOP;
$END:
    ret;""",
            meta=["x"],
        )
        self.assertEqual(f.meta, {})
        (s,) = sites(f)
        self.assertEqual(
            (s["status"], s["kind"], s["certain"]), ("open", "loop", False)
        )

    def test_ballot_of_lanes_keys_the_split_roles(self):
        f = facts("""
    ld.param.u64 %rd1, [k_param_0];
    mov.u32 %r3, %tid.x;
    and.b32 %r5, %r3, 31;
    mul.wide.u32 %rd2, %r5, 4;
    add.s64 %rd3, %rd1, %rd2;
    ld.global.u32 %r6, [%rd3];
    setp.gt.s32 %p4, %r6, 0;
    vote.sync.ballot.b32 %r1, %p4, -1;
    mov.u32 %r2, 0;
    shr.u32 %r4, %r3, 5;
    setp.lt.s32 %p1, %r1, 1;
    @%p1 bra $END;
$LOOP:
    setp.eq.u32 %p2, %r4, 0;
    @!%p2 bra $SKIP;
    bar.sync 2, 32;
$SKIP:
    bar.sync 1, 128;
    add.s32 %r2, %r2, 1;
    setp.lt.u32 %p3, %r2, %r1;
    @%p3 bra $LOOP;
$END:
    ret;""")
        self.ok(f)
        keys = _keys(f)
        self.assertEqual(sorted(keys), ["default:w0-0", "default:w1-3"])
        self.assertEqual(keys["default:w0-0"], keys["default:w1-3"])
        self.assertTrue(keys["default:w0-0"] and "" not in keys["default:w0-0"])

    def test_switch_is_lowered(self):
        """A switch brx.idx (not the dispatch) is cfg.lower_switches' compare chain, as in
        the P0 translation: on pid.x % 4 every warp takes the same case (exact KIR ifs);
        on the warp index the role splits per warp; an index not provably in range is
        refused."""
        body = """
JT:
    .branchtargets C0, C1, C2, C3;
    mov.u32 %r2, %ctaid.x;
    and.b32 %r3, %r2, 3;
    brx.idx %r3, JT;
C0:
    bar.sync 1, 128;
    bra.uni J;
C1:
    bar.sync 2, 128;
    bra.uni J;
C2:
    bar.sync 1, 128;
    bar.sync 1, 128;
    bra.uni J;
C3:
J:
    ret;"""
        f = facts(body)
        self.ok(f)
        self.assertEqual([(s["status"], s["ptx_line"] > 0) for s in sites(f)],
                         [("counted", True)] * 4)  # fmt: skip
        self.assertEqual([r["name"] for r in f.roles], ["default"])
        g = facts(
            body.replace(
                "mov.u32 %r2, %ctaid.x;",
                "mov.u32 %r1, %tid.x;\n    shr.u32 %r2, %r1, 5;",
            )
        )
        self.ok(g)
        self.assertEqual(len(g.roles), 4)  # one case per warp
        h = facts(body.replace("and.b32 %r3, %r2, 3;", "and.b32 %r3, %r2, 7;"))
        self.assertIn("not provably in [0, 4)", h.refusals[0]["msg"])

    def test_dead_registers_are_not_classified(self):
        """A class loop classifies only the registers live at its header or after its
        exit: a per-thread temporary it writes and nothing reads later (the bulk of a
        real loop body) is never classified, and the verdict-relevant classes are the
        same."""
        body = """
    ld.param.u64 %rd1, [k_param_0];
    ld.global.u32 %r1, [%rd1];
    mov.u32 %r2, 0;
    mov.u32 %r3, %tid.x;
$LOOP:
    mul.lo.u32 %r9, %r3, 4;
    add.u32 %r10, %r9, %r2;
    st.shared.u32 [smem], %r10;
    bar.sync 1, 128;
    add.s32 %r2, %r2, 1;
    setp.lt.s32 %p2, %r2, %r1;
    @%p2 bra $LOOP;
    ret;"""
        seen = []
        klass = nbar._Nb.klass

        def spy(tr, v):
            seen.append(v)
            return klass(tr, v)

        with mock.patch.object(nbar._Nb, "klass", spy):
            f = facts(body)
        self.ok(f)
        (s,) = sites(f)
        self.assertEqual((s["status"], s["rep"]), ("counted", True))
        tid = [v for v in seen if v is not None and not isinstance(v, nbar.interp.Data)
               and "%lane" in nbar.E.free_vars(v) and "mul" in str(v)]  # fmt: skip
        self.assertEqual(tid, [])  # %r9 = tid * 4 and %r10 are dead: not classified

    def test_irreducible_cycle_is_split(self):
        f = facts("""
    ld.param.u32 %r1, [k_param_1];
    setp.eq.s32 %p1, %r1, 0;
    @%p1 bra $B;
$A:
    bar.sync 1, 128;
$B:
    add.s32 %r1, %r1, -1;
    setp.gt.s32 %p2, %r1, 0;
    @%p2 bra $A;
    ret;""")
        self.ok(f)
        self.assertTrue(f.split)
        self.assertEqual({s["status"] for s in sites(f)}, {"counted"})

    def test_rotated_exit_path_is_copied_into_its_loop(self):
        """After node splitting, a loop that exits to two blocks, one of them the start of
        a path back to its header ($X, $Y: the only edges into it from outside come
        before the loop), gets a copy of that path: the loop then exits to $OUT alone and
        its barrier is counted."""
        f = facts("""
    ld.param.u32 %r1, [k_param_1];
    mov.u32 %r3, 0;
    setp.eq.s32 %p1, %r1, 0;
    @%p1 bra $B;
$A:
    bar.sync 1, 128;
$B:
    add.s32 %r1, %r1, -1;
    setp.gt.s32 %p2, %r1, 0;
    @%p2 bra $A;
$X:
    add.s32 %r3, %r3, 1;
$Y:
    mov.u32 %r4, 0;
$H:
    bar.sync 2, 128;
    setp.ge.s32 %p3, %r3, 4;
    @%p3 bra $OUT;
    add.s32 %r4, %r4, 1;
    setp.lt.s32 %p4, %r4, 2;
    @%p4 bra $H;
    bra.uni $X;
$OUT:
    ret;""")
        self.ok(f)
        self.assertLessEqual({"$A", "$X", "$Y"}, {b.split("~")[0] for b in f.split})
        self.assertEqual(sorted((s["status"], s["id"], s["rep"]) for s in sites(f)),
                         [("counted", 1, False)] * 2 + [("counted", 2, False)])  # fmt: skip

    def test_call_is_refused(self):
        f = facts("""
    call.uni (_), helper, (0);
    bar.sync 1, 128;
    ret;""")
        self.assertEqual(len(f.refusals), 1)
        self.assertIn("call", f.refusals[0]["msg"])

    def test_split_roles_keys_follow_their_paths(self):
        """Roles split from one role share a region key only where their conditions
        came the same way: warp 0 reads its loop bound from x[0] and the other warps from
        x[1] (a branch that holds no barrier), or warp 0 adds 1 to it (a predicate). Each
        role's bound is one value over its own warps, but their keys differ (P0-6 is then
        undecided: Lean pairs no instance across nests)."""
        loop = """
    mov.u32 %r4, 0;
    setp.eq.u32 %p3, %r3, 0;
    @%p3 bra $E;
$L:
    bar.sync 1, 128;
    add.u32 %r4, %r4, 1;
    setp.lt.u32 %p2, %r4, %r3;
    @%p2 bra $L;
$E:
    ret;"""
        head = """
    mov.u32 %r1, %tid.x;
    shr.u32 %r2, %r1, 5;
    ld.param.u64 %rd1, [k_param_0];
    setp.eq.u32 %p1, %r2, 0;"""
        branch = (
            head
            + """
    @%p1 bra $W0;
    ld.global.nc.u32 %r3, [%rd1+4];
    bra $J;
$W0:
    ld.global.nc.u32 %r3, [%rd1];
$J:"""
        )
        pred = (
            head
            + """
    ld.global.nc.u32 %r3, [%rd1];
    @%p1 add.u32 %r3, %r3, 1;"""
        )
        for body in (branch + loop, pred + loop):
            f = facts(body)
            self.ok(f)
            keys = _keys(f)
            self.assertEqual(sorted(keys), ["default:w0-0", "default:w1-3"])
            self.assertEqual(len(keys["default:w0-0"]), 2)  # the If on n == 0, the loop
            for a, b in zip(keys["default:w0-0"], keys["default:w1-3"]):
                self.assertNotEqual(a, b)
                self.assertEqual(a.split("#")[0], b.split("#")[0])

    def test_split_roles_own_instances_key_apart(self):
        """A value a split role holds in a form only its own warps share keys its
        regions apart from the other roles': a branch on select(warp == 0, x[0], x[1]) (its
        instance in each role is one load), a KIR `if` on warp*4 < n picking a load's
        index, a loop carrying warp + i in a uniform slot, a shuffle of the warp index (a
        table of the role's threads) folding to one value; the barrier id per warp gives
        one role per warp."""
        start = """
    ld.param.u64 %rd1, [k_param_0];
    mov.u32 %r1, %tid.x;
    shr.u32 %r2, %r1, 5;"""
        loop = """
    mov.u32 %r13, 0;
    setp.eq.u32 %p7, %r10, 0;
    @%p7 bra $E;
$L:
    bar.sync 1, 128;
    add.u32 %r13, %r13, 1;
    setp.lt.u32 %p8, %r13, %r10;
    @%p8 bra $L;
$E:
    ret;"""
        per_warp = "\n    add.u32 %r20, %r2, 4;\n    bar.sync %r20, 32;"
        load = """
    mul.wide.u32 %rd2, %r9, 4;
    add.s64 %rd3, %rd1, %rd2;
    ld.global.nc.u32 %r10, [%rd3];"""
        select = (
            start
            + """
    ld.global.nc.u32 %r5, [%rd1];
    ld.global.nc.u32 %r6, [%rd1+4];
    setp.eq.u32 %p1, %r2, 0;
    selp.b32 %r3, %r5, %r6, %p1;
    setp.ne.u32 %p3, %r2, 0;
    @%p3 bra $S;
    bar.sync 2, 32;
$S:
    setp.eq.u32 %p4, %r3, 0;
    @%p4 bra $E;
    bar.sync 1, 128;
$E:
    ret;"""
        )
        kir_if = (
            start
            + per_warp
            + """
    ld.param.u32 %r30, [k_param_1];
    shl.b32 %r7, %r2, 2;
    setp.lt.u32 %p5, %r7, %r30;
    mov.u32 %r9, 2;
    @!%p5 bra $A;
    bar.sync 9, 32;
    mov.u32 %r9, 1;
$A:"""
            + load
            + loop
        )
        slot = (
            start
            + per_warp
            + """
    mov.u32 %r9, %r2;
    mov.u32 %r12, 0;
$K:
    add.u32 %r9, %r9, 1;
    add.s32 %r12, %r12, 1;
    setp.lt.s32 %p6, %r12, 3;
    @%p6 bra $K;"""
            + load
            + loop
        )
        shfl = (
            start
            + """
    ld.global.nc.u32 %r5, [%rd1];
    ld.global.nc.u32 %r6, [%rd1+4];
    shfl.sync.idx.b32 %r3, %r2, 0, 31, -1;
    setp.ge.u32 %p3, %r3, 2;
    @%p3 bra $S;
    bar.sync 2, 64;
$S:
    setp.lt.u32 %p1, %r3, 2;
    selp.b32 %r10, %r5, %r6, %p1;"""
            + loop
        )
        for body, nroles in ((select, 2), (kir_if, 4), (slot, 4), (shfl, 2)):
            f = facts(body)
            self.ok(f)
            keys = _keys(f)
            self.assertEqual(len(keys), nroles)
            self.assertTrue(all(keys.values()))
            self.assertEqual(len({tuple(k) for k in keys.values()}), nroles)

    def test_only_the_accepted_value_is_published(self):
        """A spin publishes the value it accepted, also through a copy, after a rotated
        guard testing some of its bits, or where it tests a uniform entry value first (a
        repeated region on it), not a value computed from it in the loop: here one per
        lane, so a branch on it is an open site, not certain (it differs between the
        lanes, but not on every value the spin may accept); nor where it tests a value
        per lane first (lanes that accept it leave with their own)."""
        spin = """
    ld.param.u64 %rd1, [k_param_0];
    mov.u32 %r8, %tid.x;
    and.b32 %r9, %r8, 31;
$S:
    ld.acquire.gpu.global.u32 %r1, [%rd1];
    xor.b32 %r2, %r1, %r9;
    mov.u32 %r3, %r1;
    setp.eq.s32 %p1, %r1, 0;
    @%p1 bra $S;"""
        f = facts(
            spin
            + """
    setp.eq.u32 %p2, %r2, 5;
    @%p2 bra $K;
    bar.sync 1, 128;
$K:
    ret;"""
        )
        self.ok(f)
        (s,) = sites(f)
        self.assertEqual((s["status"], s["certain"]), ("open", False))
        bounded = """
    mov.u32 %r4, 0;
$L:
    bar.sync 1, 128;
    add.s32 %r4, %r4, 1;
    setp.lt.s32 %p2, %r4, %r6;
    @%p2 bra $L;
    ret;"""
        rotated = """
    ld.param.u64 %rd1, [k_param_0];
    ld.relaxed.gpu.global.b32 %r1, [%rd1];
    and.b32 %r4, %r1, -32768;
    setp.eq.b32 %p1, %r4, 32768;
    @%p1 bra $G;
$S:
    ld.relaxed.gpu.global.b32 %r1, [%rd1];
    and.b32 %r5, %r1, -32768;
    setp.eq.b32 %p3, %r5, 32768;
    @!%p3 bra $S;
$G:
    and.b32 %r6, %r1, 32767;"""
        header = """
    ld.param.u64 %rd1, [k_param_0];
    ld.param.u32 %r1, [k_param_1];
$C:
    shr.u32 %r4, %r1, 14;
    setp.eq.u32 %p1, %r4, 3;
    @%p1 bra $D;
    ld.relaxed.gpu.global.b32 %r1, [%rd1];
    bra.uni $C;
$D:
    and.b32 %r6, %r1, 16383;"""
        for body in (spin + "\n    mov.u32 %r6, %r3;" + bounded, rotated + bounded,
                     header + bounded):  # fmt: skip
            (s,) = sites(facts(body))
            self.assertEqual((s["status"], s["rep"]), ("counted", True))
        lanes = header.replace(
            "ld.param.u32 %r1, [k_param_1];",
            "mov.u32 %r8, %tid.x;\n    and.b32 %r1, %r8, 31;",
        )
        (s,) = sites(facts(lanes + bounded))
        self.assertEqual((s["status"], s["certain"]), ("open", False))

    def test_strong_shared_load_is_no_uniform_data(self):
        """A strong shared load (.volatile, .relaxed, .acquire) may read another value in
        each warp, as a strong global load: a loop around a barrier exiting on one, and a
        branch on one around a barrier, are certain divergence."""
        f = facts("""
$L:
    bar.sync 1, 128;
    ld.volatile.shared.u32 %r1, [smem+8];
    setp.ne.s32 %p1, %r1, 0;
    @%p1 bra $L;
    ret;""")
        self.ok(f)
        (s,) = sites(f)
        self.assertEqual((s["status"], s["kind"], s["certain"]), ("open", "loop", True))
        g = facts("""
    ld.relaxed.cta.shared.u32 %r1, [smem+8];
    setp.eq.s32 %p1, %r1, 0;
    @%p1 bra $S;
    bar.sync 1, 128;
$S:
    ret;""")
        (s,) = sites(g)
        self.assertEqual((s["status"], s["kind"], s["certain"]), ("open", "cond", True))

    def test_certain_only_under_certain_conditions(self):
        """An open site is certain only if every divergent condition around it is and it
        lies in no repeated region (which may run no pass): a lane condition inside a branch
        on uniform data, an elected lane inside or around a branch on data the kernel
        writes, and a count no warps meet inside a branch on uniform data are not."""
        guard = """
    ld.param.u64 %rd1, [k_param_0];
    ld.global.nc.u32 %r7, [%rd1];
    setp.eq.s32 %p2, %r7, 0;
    @%p2 bra $E;"""
        lane = """
    mov.u32 %r1, %tid.x;
    and.b32 %r2, %r1, 31;
    setp.ge.u32 %p1, %r2, 16;
    @%p1 bra $E;"""
        written = """
    ld.param.u64 %rd1, [k_param_0];
    mov.u32 %r1, %tid.x;
    st.global.u32 [%rd1+64], %r1;"""
        data = """
    ld.global.u32 %r7, [%rd1];
    setp.ne.s32 %p2, %r7, 12345;
    @%p2 bra $E;"""
        elect = "\n    elect.sync %r9|%p1, -1;\n    @!%p1 bra $E;"
        tail = "\n    bar.sync 1, 128;\n$E:\n    ret;"
        for body in (guard + lane + tail, written + elect + data + tail,
                     written + data + elect + tail,
                     guard + "\n    bar.sync 1, 100;\n$E:\n    ret;"):  # fmt: skip
            f = facts(body)
            self.ok(f)
            (s,) = sites(f)
            self.assertEqual((s["status"], s["certain"]), ("open", False))

    def test_lanes_at_a_barrier_without_aligned(self):
        """Without .aligned each thread arrives on its own: lanes of a warp that reach a
        barrier.sync at two instructions, or only some lanes under a predicate, are open
        but not certain; the aligned bar.sync under the same predicate is certain."""
        f = facts("""
    mov.u32 %r1, %tid.x;
    and.b32 %r2, %r1, 31;
    setp.lt.u32 %p1, %r2, 16;
    @%p1 bra $LO;
    barrier.sync 3, 128;
    bra $END;
$LO:
    barrier.sync 3, 128;
$END:
    ret;""")
        self.ok(f)
        self.assertEqual([(s["status"], s["kind"], s["certain"]) for s in sites(f)],
                         [("open", "lane", False)] * 2)  # fmt: skip
        pred = """
    mov.u32 %r1, %tid.x;
    and.b32 %r2, %r1, 31;
    setp.lt.u32 %p1, %r2, 16;
    @%p1 {} 3, 128;
    ret;"""
        for op, certain in (("barrier.sync", False), ("bar.sync", True)):
            (s,) = sites(facts(pred.format(op)))
            self.assertEqual(
                (s["status"], s["kind"], s["certain"]), ("open", "lane", certain)
            )

    def test_barrier_cta_is_a_named_barrier(self):
        """bar/barrier's optional .cta changes nothing (PTX ISA): barrier.cta.sync,
        bar.cta.sync and barrier.cta.arrive.aligned are counted as their plain forms."""
        self.assertEqual(nbar.ops.mnemonic("barrier.cta.sync.aligned"),
                         "barrier.sync.aligned")  # fmt: skip
        self.assertEqual(nbar.ops.mnemonic("bar.cta.red.popc.u32"), "bar.red.popc")
        f = facts("""
    barrier.cta.sync 2, 128;
    bar.cta.sync 3, 128;
    mov.u32 %r1, %tid.x;
    setp.lt.u32 %p1, %r1, 64;
    @%p1 bra $P;
    barrier.cta.sync.aligned 4, 128;
    bra.uni $E;
$P:
    barrier.cta.arrive.aligned 4, 128;
$E:
    ret;""")
        self.ok(f)
        got = sorted((s["role"], s["op"], s["id"]) for s in sites(f, "counted"))
        self.assertEqual(got, [("default:w0-1", "arrive", 4), ("default:w0-1", "sync", 2),
                               ("default:w0-1", "sync", 3), ("default:w2-3", "sync", 2),
                               ("default:w2-3", "sync", 3), ("default:w2-3", "sync", 4)])  # fmt: skip

    def test_discarded_final_rounds_are_skipped(self):
        """A nest of loops bounded by a value read from x (uniform data: each loop is
        translated by classes, an nb_rep region): inside an enclosing loop's fixed-point
        round, whose work is discarded, a loop's final round is skipped
        (nbar._SKIP_DISCARDED). The KIR does not change (also the variable names of an
        exact loop inside the nest and of one after it: discarded work makes no names)
        and the innermost barrier is translated fewer times: 3^d passes at depth d
        without the cut (an exact attempt, one fixed-point round, the final round),
        81 -> 48 at depth 4."""
        passes = {}
        for skip in (False, True):
            seen = []
            named_bar = nbar._Nb.named_bar

            def spy(tr, i, mn, seen=seen, named_bar=named_bar):
                seen.append(i.line)
                return named_bar(tr, i, mn)

            with mock.patch.object(nbar, "_SKIP_DISCARDED", skip):
                with mock.patch.object(nbar._Nb, "named_bar", spy):
                    nest = facts(_nest(4))
                got = [nest, facts(NEST_NAMES)]
            for f in got:
                self.ok(f)
            passes[skip] = (len(seen), [doc(f) for f in got])
        self.assertEqual(passes[False][0], 81)
        self.assertEqual(passes[True][0], 48)
        self.assertEqual(passes[False][1], passes[True][1])  # the same KIRs
        names = [s.get("var") for s in _stmts(passes[True][1][1]["roles"][0]["body"])
                 if s["op"] in ("let", "for")]  # fmt: skip
        # the exact loop's variable in the nest, then the slots of the loop after it: the
        # first names of the role (the discarded rounds and attempts before them made
        # none, interp.Translator.restore)
        self.assertEqual(names[:3], ["i1", "c2", "go3"])


# The innermost barrier of NEST_NAMES is in an exact counted loop (a KIR `for`), and an
# exact loop on the parameter n follows the nest (`while` with its slots): both make
# fresh names, the first ones in the rounds the cut skips.
NEST_NAMES = """
    ld.param.u64 %rd1, [k_param_0];
    ld.global.u32 %r1, [%rd1];
    mov.u32 %r10, 0;
$L0:
    mov.u32 %r11, 0;
$L1:
    mov.u32 %r20, 0;
$E0:
    bar.sync 1, 128;
    add.s32 %r20, %r20, 1;
    setp.lt.s32 %p5, %r20, 4;
    @%p5 bra $E0;
    add.s32 %r11, %r11, 1;
    setp.lt.s32 %p2, %r11, %r1;
    @%p2 bra $L1;
    add.s32 %r10, %r10, 1;
    setp.lt.s32 %p1, %r10, %r1;
    @%p1 bra $L0;
    ld.param.u32 %r30, [k_param_1];
    mov.u32 %r31, 0;
$E1:
    bar.sync 2, 128;
    add.s32 %r31, %r31, 1;
    setp.lt.s32 %p6, %r31, %r30;
    @%p6 bra $E1;
    ret;"""


def _nest(depth: int) -> str:
    """depth loops, each counting to the value read from x, around one bar.sync."""
    out = "    ld.param.u64 %rd1, [k_param_0];\n    ld.global.u32 %r1, [%rd1];\n"
    for k in range(depth):
        out += f"    mov.u32 %r{10 + k}, 0;\n$L{k}:\n"
    out += "    bar.sync 1, 128;\n"
    for k in reversed(range(depth)):
        out += (f"    add.s32 %r{10 + k}, %r{10 + k}, 1;\n"
                f"    setp.lt.s32 %p{k + 1}, %r{10 + k}, %r1;\n    @%p{k + 1} bra $L{k};\n")  # fmt: skip
    return out + "    ret;"


def _keys(f) -> dict:
    """role -> the keys of its nb_rep begin events (the ones Lean reads), in order."""
    return {r["name"]: [e["key"] for e in _stmts(r["body"])
                        if e.get("ev") == "nb_rep" and e["begin"]]
            for r in f.roles}  # fmt: skip


def _stmts(stmts):
    for s in stmts:
        yield s
        for k in ("then", "else", "body"):
            yield from _stmts(s.get(k, []))


if __name__ == "__main__":
    unittest.main()
