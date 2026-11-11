# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the PTX core translator's contract (frontend/ptx/translate.py).

    python3 -I tests/ptx/test_translate.py      (or the buck target :test_translate)

Facts of a Triton warp-specialized kernel (roles default / partition<k> with their warps,
the barrier its default initializes and arrives on and its partition waits on), KIR-valid;
addr marks per role (a let name used in two roles is marked only where it feeds an
address, R5) and none on a mul.hi or sign-extension expansion; KIR locs (the innermost frame below the
source root with the family's prefix, inherited without a loc; DSL 4.4.2's line-only
rule); refusal kinds (an unknown instruction is an error, one outside the M1 set a gap,
a control-flow shape a gap, an exception of the core an error at its instruction); work
anchors under both rules (TTGIR gate: one event after the definition; CuTe each: -1 at
the role's start, then the anchored load's index), and an anchored load counting only
where its predicate holds. param_table names a struct's pointer field only by the tensor
its address starts in every record that has a non-null pointer.
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

from ...frontend.ptx import parse, translate as T  # noqa: E402
from .fixtures.kernels import (  # noqa: E402
    evaluate,
    events,
    kernel,
    kir_errors,
    run,
    statements,
)

# Warp-specialized (8 warps, the default 0-3 falling through): warps 4, 5 -> state 1, 6 ->
# 0, 7 -> 2, exit 3 (1 and 2 are padding). The default initializes smem+16 and arrives
# on it after a store; partition0 (warp 6) waits on it with a computed parity.
WS = kernel(
    """
JT:
    .branchtargets $P0, $PAD1, $PAD2, $RET;
    mov.u32 %r1, %tid.x;
    shr.u32 %r2, %r1, 5;
    shfl.sync.idx.b32 %r3, %r2, 0, 31, -1;
    setp.lt.u32 %p1, %r3, 4;
    @!%p1 bra $LOOP;
    ld.param.u64 %rd1, [k_param_0];
    ld.param.b32 %r20, [k_param_1];
    mov.b32 %r30, smem;
    setp.eq.s32 %p5, %r2, 0;
    elect.sync %r31|%p6, -1;
    and.pred %p7, %p5, %p6;
    @%p7 mbarrier.init.shared::cta.b64 [%r30+16], 1;
    st.shared.b32 [smem+24], 33554689;
    barrier.sync 1;
    barrier.sync 1;
    add.s32 %r21, %r20, 1;
    mul.wide.s32 %rd2, %r21, 4;
    add.s64 %rd3, %rd1, %rd2;
    setp.eq.s32 %p8, %r1, 0;
    @%p8 st.global.b32 [%rd3], %r1;
    @%p8 mbarrier.arrive.shared::cta.b64 _, [%r30+16];
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
    add.s32 %r8, %r6, 16;
    ld.param.b32 %r22, [k_param_1];
    add.s32 %r9, %r22, 1;
    and.b32 %r10, %r9, 1;
    {
    .reg .pred complete;
    waitLoop:
    mbarrier.try_wait.parity.shared::cta.b64 complete, [%r8], %r10;
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


def ws_facts(**pol):
    params = [{"ptx": "k_param_0", "kind": "ptr", "name": "x", "size": 8, "fields": {}},
              {"ptx": "k_param_1", "kind": "scalar", "name": "n", "size": 4, "fields": {}}]  # fmt: skip
    policy = T.policy(
        "ttgir", src_root="/src", params=params, tensors={"x": {"elem_bytes": 4}}, **pol
    )
    return T.translate(
        WS.replace("smem+24", "smem+24"), "k", [{"block": [256, 1, 1]}], policy
    )


class Contract(unittest.TestCase):
    def test_warp_specialized_facts(self):
        f = ws_facts()
        self.assertEqual(f.refusals, [])
        self.assertEqual([(r["name"], r["w0"], r["warps"]) for r in f.roles],
                         [("default", 0, 4), ("partition0", 6, 1)])  # fmt: skip
        self.assertEqual(
            [(b["name"], b["count"]["v"], b["num"]["v"]) for b in f.barriers],
            [("mbar_16", 1, 1)],
        )
        self.assertEqual(
            [e["ev"] for e in events(f, role="default")],
            ["bar_init", "gstore", "arrive"],
        )
        self.assertEqual([e["ev"] for e in events(f, role="partition0")], ["wait"])
        self.assertEqual(
            (f.target, f.block, f.params, f.written),
            ("sm_100", (256, 1, 1), {"n": 32}, ["x"]),
        )
        self.assertEqual(
            f.tensors, {"x": {"uses": ["gstore"], "metadata": False, "elem_bytes": 4}}
        )
        self.assertEqual(kir_errors(f), [])

    def test_addr_marks_per_role(self):
        f = ws_facts()
        lets = {
            r["name"]: [s for s in statements(r["body"]) if s["op"] == "let"]
            for r in f.roles
        }
        shared = {s["var"] for s in lets["default"]} & {
            s["var"] for s in lets["partition0"]
        }
        self.assertTrue(shared)  # one variable name in both roles
        marked = [s["var"] for s in lets["partition0"] if s["expr"].get("addr")]
        self.assertEqual(marked, [])  # the parity's add is no address (R5)
        self.assertTrue(any(s["expr"].get("addr") for s in lets["default"]))

    def test_mul_hi_is_not_marked(self):
        f = run("""
    mov.u32 %r1, %tid.x;
    ld.param.b32 %r2, [k_param_1];
    ld.param.u64 %rd1, [k_param_0];
    mul.hi.u32 %r3, %r2, -1431655765;
    shr.u32 %r4, %r3, 1;
    add.s32 %r5, %r4, 1;
    mul.wide.u32 %rd2, %r5, 4;
    add.s64 %rd3, %rd1, %rd2;
    setp.eq.s32 %p1, %r1, 0;
    @%p1 st.global.b32 [%rd3], %r1;
    ret;
""")
        text = str(f.roles[0]["body"])
        self.assertIn("'o': 'mul'", text)
        nodes = []

        def walk(n):
            if isinstance(n, dict):
                nodes.append(n)
                for v in n.values():
                    walk(v)
            elif isinstance(n, list):
                for v in n:
                    walk(v)

        walk(f.roles[0]["body"])
        hi = [n for n in nodes if T._mulhi_operands(n) is not None]
        self.assertEqual(len(hi), 1)
        self.assertFalse(hi[0].get("addr") or hi[0]["a"]["a"].get("addr"))
        self.assertTrue(
            any(n.get("addr") for n in nodes if n.get("e") == "bin" and n["o"] == "add")
        )

    def test_locs(self):
        f = run("""
    mov.u32 %r1, %tid.x;
    ld.param.u64 %rd1, [k_param_0];
    setp.eq.s32 %p1, %r1, 0;
    .loc 1 7 3
    @%p1 st.global.b32 [%rd1], %r1;
    .loc 1 0 3
    @%p1 st.global.b32 [%rd1+4], %r1;
    ret;
""")
        (ev,) = events(f)  # one instruction family at one loc: one event of two stores
        self.assertEqual((ev["loc"], ev["offsets"]["shape"]), ("fbcode/k.py:7", [2]))
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "k.py").write_text("x = 1\n\n    y = f(x)  # z\n")
            body = """
    mov.u32 %r1, %tid.x;
    ld.param.u64 %rd1, [k_param_0];
    setp.eq.s32 %p1, %r1, 0;
    .loc 1 3 4
    @%p1 st.global.b32 [%rd1], %r1;
    .loc 1 3 20
    @%p1 st.global.b32 [%rd1+4], %r1;
    .loc 1 2 0
    @%p1 st.global.b32 [%rd1+8], %r1;
    .loc 1 847 18
    @%p1 st.global.b32 [%rd1+12], %r1;
    ret;
"""
            f = run(body, family="cute", src=d)
        # columns 0-based: 3:4 is `y`, 3:20 a comment, line 2 blank, 847 beyond the file
        self.assertEqual([e["loc"] for e in events(f)], ["k.py:3"] * 4)

    def test_refusal_kinds(self):
        f = run("    mov.u32 %r1, %tid.x;\n    frobnicate.b32 %r2, %r1;\n    ret;\n")
        self.assertEqual(f.refusals[0]["kind"], "error")
        self.assertIn("no op table row", f.refusals[0]["msg"])
        f = run("    barrier.cluster.arrive.aligned;\n    ret;\n")
        self.assertEqual(f.refusals[0]["kind"], "gap")
        self.assertIn("not in v1", f.refusals[0]["msg"])
        f = run(
            "    mov.u32 %r1, %tid.x;\n    setp.eq.s32 %p1, %r1, 0;\n    @%p1 bra A;\n    ret;\nA:\n    ret;\n"
        )
        self.assertEqual((f.refusals[0]["kind"], f.roles), ("gap", []))

    def test_internal_error_is_a_refusal(self):
        # An exception of the core is an "error" refusal, never raised: in a role at the
        # instruction (a constant division by zero is an ExprError of the folder), else
        # by its type (an anchor site past the kernel's instructions).
        f = run("    mov.u32 %r1, 7;\n    div.u32 %r2, %r1, 0;\n    ret;\n")
        self.assertEqual((f.refusals[0]["kind"], f.roles), ("error", []))
        self.assertRegex(
            f.refusals[0]["msg"],
            r"^div\.u32 at fbcode/k\.py:1 \(PTX line \d+\): ExprError: constant udiv by zero$",
        )
        site = {"field": "b", "kind": "assign", "instr": ("k", 99, 1), "detail": {}}
        f = run("    ret;\n", work={"fields": ["b"], "mode": "gate", "sites": [site]})
        self.assertEqual(
            f.refusals,
            [{"kind": "error", "msg": "IndexError: list index out of range"}],
        )

    def work(self, family, kind):
        body = """
    mov.u32 %r1, %tid.x;
    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd4, [k_param_1];
    mov.u32 %r2, %ctaid.y;
    shr.u32 %r3, %r2, 1;
    mul.wide.u32 %rd5, %r3, 4;
    add.s64 %rd6, %rd4, %rd5;
    ld.global.u32 %r4, [%rd6];
    mul.wide.s32 %rd2, %r4, 4;
    add.s64 %rd3, %rd1, %rd2;
    setp.eq.s32 %p1, %r1, 0;
    @%p1 st.global.b32 [%rd3], %r1;
    ret;
"""
        fn = parse.parse(kernel(body, ptrs=("x", "seg"), scalars=())).function("k")
        instrs = fn.instructions()
        k = next(
            j
            for j, i in enumerate(instrs)
            if i.text.startswith("shr" if kind == "assign" else "ld.global")
        )
        detail = {"reg": "%r3"} if kind == "assign" else {"tensor": "seg"}
        site = {
            "field": "batch",
            "kind": kind,
            "instr": ("k", k, instrs[k].line),
            "detail": detail,
        }
        mode = "gate" if family == "ttgir" else "each"
        return run(body, family=family, ptrs=("x", "seg"), scalars=(),
                   work={"fields": ["batch"], "mode": mode, "sites": [site]})  # fmt: skip

    def test_work_anchors(self):
        f = self.work("ttgir", "assign")
        (w,) = events(f, "work")
        (let,) = [
            s
            for s in f.roles[0]["body"]
            if s.get("var") == w["fields"]["batch"].get("name")
        ]
        self.assertEqual(let["expr"]["o"], "lshr")  # ctaid.y >> 1
        self.assertEqual(f.work_events, 1)
        f = self.work("cute", "load")
        body = f.roles[0]["body"]
        self.assertEqual((body[0]["var"], body[0]["expr"]["v"]), ("work_batch", -1))
        (w,) = events(f, "work")
        self.assertEqual(
            w["fields"], {"batch": {"e": "var", "w": 64, "name": "work_batch"}}
        )
        self.assertEqual(kir_errors(f), [])

    def test_work_anchor_counts_only_where_its_load_runs(self):
        # Two anchored loads under complementary uniform predicates (pid < n): each sets
        # the field inside its `if`. Under lane-dependent runtime predicates both set it
        # (the CuTe rule: the work item is the role's, whichever lanes read it); the gate
        # rule refuses that, and any condition.
        body = """
    mov.u32 %r1, %tid.x;
    ld.param.b32 %r2, [k_param_2];
    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd4, [k_param_1];
    mov.u32 %r9, %ctaid.x;
    setp.lt.s32 %p1, PRED, %r2;
    mul.wide.s32 %rd2, %r9, 4;
    add.s64 %rd3, %rd4, %rd2;
    mov.b32 %r4, 0;
    @%p1 ld.global.u32 %r4, [%rd3];
    @!%p1 ld.global.u32 %r4, [%rd4+8];
    mul.wide.s32 %rd5, %r4, 4;
    add.s64 %rd6, %rd1, %rd5;
    setp.eq.s32 %p3, %r1, 0;
    @%p3 st.global.b32 [%rd6], %r2;
    ret;
"""

        def anchored(pred, mode="each"):
            text = body.replace("PRED", pred)
            fn = parse.parse(kernel(text, ("x", "meta"), ("n",))).function("k")
            sites = [{"field": "batch", "kind": "load", "instr": ("k", j, i.line),
                      "detail": {"tensor": "meta"}}
                     for j, i in enumerate(fn.instructions()) if i.opcode.startswith("ld.global")]  # fmt: skip
            work = {"fields": ["batch"], "mode": mode, "sites": sites}
            return run(text, family="cute", ptrs=("x", "meta"), work=work)

        f = anchored("%r9")
        self.assertEqual(kir_errors(f), [])
        meta = {"meta": [5] * 8}
        for pid, batch in ((0, 0), (3, 2)):
            got = evaluate(f, {"n": 1}, metadata=meta, pid=(pid, 0, 0))
            self.assertEqual([t[1] for t in got if t[0] == "work"], [{"batch": batch}])
        f = anchored("%r1")
        self.assertEqual(kir_errors(f), [])
        got = evaluate(f, {"n": 1}, metadata=meta, pid=(3, 0, 0))
        self.assertEqual(
            [t[1] for t in got if t[0] == "work"], [{"batch": 3}, {"batch": 2}]
        )
        f = anchored("%r1", mode="gate")
        self.assertIn("runtime values in each lane", f.refusals[0]["msg"])
        f = anchored("%r9", mode="gate")
        self.assertIn("gate rule", f.refusals[0]["msg"])

    def test_sign_extension_is_not_marked(self):
        # x[sext8(t + 200) + 128]: the shl of the sign extension drops the bits above
        # the narrow value by design; marked, Lean would report its wrap as P0-1.
        f_body = """
    ld.param.u64 %rd1, [k_param_0];
    mov.u32 %r1, %tid.x;
    cvt.u16.u32 %rs1, %r1;
    add.s16 %rs2, %rs1, 200;
    cvt.s16.s8 %rs3, %rs2;
    cvt.s32.s16 %r2, %rs3;
    add.s32 %r3, %r2, 128;
    bfe.s32 %r4, %r1, 4, 1;
    add.s32 %r5, %r3, %r4;
    mul.wide.s32 %rd2, %r5, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r1;
    ret;
"""
        for fam in ("ttgir", "cute"):
            f = run(f_body, family=fam, threads=32)
            self.assertEqual(kir_errors(f), [])
            stores = [t for t in evaluate(f, {"n": 1}) if t[0] != "overflow"]
            offs = sorted(o for t in stores for o in t[2])
            want = sorted(
                ((t + 200) & 255 ^ 128) - 128 + 128 - (t >> 4 & 1) for t in range(32)
            )
            self.assertEqual(offs, want)
            self.assertNotIn("overflow", [t[0] for t in evaluate(f, {"n": 1})])


class ParamTable(unittest.TestCase):
    """param_table's struct pointer fields: named by the tensor their address is in every
    record (a null pointer, a zero-length tensor's, resolves nowhere and is skipped)."""

    @staticmethod
    def records(*fields):
        return [{"params": [{"i": 0, "name": "p0", "kind": "aggregate", "size": 8,
                             "fields": [{"off": 0, "size": 8, "kind": "ptr", "name": "param0_0", **f}]}]}
                for f in fields]  # fmt: skip

    def name(self, *fields):
        return T.param_table(self.records(*fields), "cute")[0]["fields"][0]["name"]

    def test_pointer_field_names_one_tensor_in_every_record(self):
        x = {"tensor": "X", "storage": [64, 192], "match": "exact", "elem_off": 0}
        inside = {**x, "match": "inside", "elem_off": 32}
        y = {"tensor": "Y", "storage": [256, 512], "match": "inside", "elem_off": 4}
        self.assertEqual(self.name({"value": 64, "tensor": [x]}, {"value": 0}), "X")
        # another launch points 32 elements into X, or into Y, or at no named tensor:
        # accesses there are no X accesses at X's offsets
        for other in ([inside], [y], []):
            got = self.name(
                {"value": 64, "tensor": [x]}, {"value": 192, "tensor": other}
            )
            self.assertIsNone(got, other)


if __name__ == "__main__":
    unittest.main()
