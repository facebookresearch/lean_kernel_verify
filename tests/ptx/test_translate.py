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
rule); refusal kinds (an unknown instruction is an error, a cluster
barrier in a Triton kernel a gap,
a control-flow shape a gap, an exception of the core an error at its instruction); work
anchors under both rules (TTGIR gate: one event after the definition; CuTe each: -1 at
the role's start, then the anchored load's index), and an anchored load counting only
where its predicate holds. A switch is a compare chain (a module the caller parsed stays
as parsed: the chain goes into a copy of the entry function). param_table names a struct's pointer field only by the tensor
its address starts in every record that has a non-null pointer. Captures: forwarded
(also bytewise), unknown where the store may not happen (a predicate or a branch on a
uniform runtime condition), runs in only some of the default's threads, differs between
threads or follows the first fork barrier, refused where overwritten after a read. A
dispatch that may run no round (cfg.py, item rounds): each worker's region a KIR `if` on
the default's decision (a parameter, a metadata load), captures of the round side
forwarded under it, refused where the decision is data or differs between the default's
threads.
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
    SRC,
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


def ws_facts(text=WS, **pol):
    params = [{"ptx": "k_param_0", "kind": "ptr", "name": "x", "size": 8, "fields": {}},
              {"ptx": "k_param_1", "kind": "scalar", "name": "n", "size": 4, "fields": {}}]  # fmt: skip
    policy = T.policy(
        "ttgir", src_root="/src", params=params, tensors={"x": {"elem_bytes": 4}}, **pol
    )
    return T.translate(text, "k", [{"block": [256, 1, 1]}], policy)


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

    def test_generic_shared_write_is_refused(self):
        # a generic store may rewrite a worker's state byte, which the dispatch
        # recognizer reads from st.shared only (cfg.py)
        gen = "    cvt.u64.u32 %rd4, %r30;\n    cvta.shared.u64 %rd5, %rd4;\n" \
            "    st.u32 [%rd5+24], 0;\n"  # fmt: skip
        f = ws_facts(
            WS.replace("    barrier.sync 1;\n", gen + "    barrier.sync 1;\n", 1)
        )
        self.assertIn("a generic write of shared memory in a warp-specialized kernel",
                      f.refusals[0]["msg"])  # fmt: skip
        # ... as may a shared atomic, though no generic address
        red = "    red.shared.add.u32 [%r30+24], 1;\n"
        f = ws_facts(
            WS.replace("    barrier.sync 1;\n", red + "    barrier.sync 1;\n", 1)
        )
        self.assertIn("red.shared.add.u32 in a warp-specialized kernel",
                      f.refusals[0]["msg"])  # fmt: skip

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
        f = run(
            "    mov.u32 %r10, smem;\n    mbarrier.complete_tx.shared::cta.b64 [%r10], 16;\n    ret;\n"
        )
        self.assertEqual(f.refusals[0]["kind"], "gap")
        self.assertIn(
            "KIR complete_tx is a shared::cluster copy's", f.refusals[0]["msg"]
        )
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

    def test_float_tensor_loads(self):
        # A load of a float tensor (its contents are never recorded) is Data: refused
        # where it reaches an address, clean as data, and it still anchors a work field.
        body = """
    mov.u32 %r1, %tid.x;
    ld.param.u64 %rd1, [k_param_0];
    ld.param.u64 %rd5, [k_param_1];
    mov.u32 %r2, %ctaid.x;
    mul.wide.u32 %rd4, %r2, 4;
    add.s64 %rd6, %rd5, %rd4;
    ld.global.b32 %r4, [%rd6];
    setp.eq.s32 %p1, %r1, 0;
    USE
    ret;
"""
        data = body.replace("USE", "@%p1 st.global.b32 [%rd1], %r4;")
        fn = parse.parse(kernel(data, ptrs=("x", "seg"), scalars=())).function("k")
        k = next(
            j for j, i in enumerate(fn.instructions()) if i.text.startswith("ld.global")
        )
        site = {"field": "batch", "kind": "load", "instr": ("k", k, fn.instructions()[k].line),
                "detail": {"tensor": "seg"}}  # fmt: skip
        f = run(data, ptrs=("x", "seg"), scalars=(), floats=("seg",),
                work={"fields": ["batch"], "mode": "gate", "sites": [site]})  # fmt: skip
        self.assertEqual(f.refusals, [])
        self.assertEqual(len(events(f, "work")), 1)
        self.assertFalse(f.tensors["seg"]["metadata"])
        addr = body.replace("USE", "mul.wide.s32 %rd2, %r4, 4;\n    add.s64 %rd3, %rd1, %rd2;\n"
                            "    @%p1 st.global.b32 [%rd3], %r1;")  # fmt: skip
        f = run(addr, ptrs=("x", "seg"), scalars=(), floats=("seg",))
        self.assertIn("data value from ld.global.b32 of seg", f.refusals[0]["msg"])

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

    def test_work_field_hoisted_out_of_the_loop(self):
        # head = ctaid.y >> 1 is computed before the work loop (LLVM hoists the
        # loop-invariant value out of the loop TTGIR defines it in), batch in it: the loop
        # body starts with head, so each trip's work event has both. A field with a site
        # inside the loop (head again, a novalue site) is not carried in.
        body = """
    mov.u32 %r1, %tid.x;
    ld.param.u64 %rd1, [k_param_0];
    mov.u32 %r2, %ctaid.y;
    shr.u32 %r3, %r2, 1;
    mov.b32 %r20, 0;
LOOP:
    add.s32 %r21, %r20, 7;
    mul.wide.s32 %rd2, %r21, 4;
    add.s64 %rd3, %rd1, %rd2;
    setp.eq.s32 %p1, %r1, 0;
    @%p1 st.global.b32 [%rd3], %r3;
    add.s32 %r20, %r20, 1;
    setp.lt.s32 %p2, %r20, 3;
    @%p2 bra LOOP;
    ret;
"""
        instrs = (
            parse.parse(kernel(body, ptrs=("x",), scalars=()))
            .function("k")
            .instructions()
        )

        def site(field, kind, prefix, reg=None):
            j = next(j for j, i in enumerate(instrs) if i.text.startswith(prefix))
            return {"field": field, "kind": kind, "instr": ("k", j, instrs[j].line),
                    "detail": {"reg": reg} if reg else {}}  # fmt: skip

        sites = [
            site("head", "assign", "shr", "%r3"),
            site("batch", "assign", "add.s32 %r21", "%r21"),
        ]
        work = {"fields": ["batch", "head"], "mode": "gate", "sites": sites}
        f = run(body, ptrs=("x",), scalars=(), work=work)
        self.assertEqual(f.refusals, [])
        got = [t[1] for t in evaluate(f, {}, pid=(0, 5, 0)) if t[0] == "work"]
        self.assertEqual(got, [{"batch": b, "head": 2} for b in (7, 8, 9)])
        self.assertEqual(kir_errors(f), [])
        work["sites"] = sites + [site("head", "novalue", "add.s32 %r21")]
        f = run(body, ptrs=("x",), scalars=(), work=work)
        got = [t[1] for t in evaluate(f, {}, pid=(0, 5, 0)) if t[0] == "work"]
        self.assertEqual(got, [{"batch": b, "head": -1} for b in (7, 8, 9)])

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

    def test_bit_moves_do_not_wrap(self):
        # bits moved into bit 31 by brev, prmt, bfi, shf, vote.ballot and %lanemask_eq
        # (libdevice ffs is brev + bfind.shiftamt): no arithmetic, so no address wrap
        f_body = """
    ld.param.u64 %rd1, [k_param_0];
    mov.u32 %r1, %tid.x;
    brev.b32 %r2, %r1;
    shr.u32 %r3, %r2, 27;
    or.b32 %r4, %r1, 128;
    prmt.b32 %r5, %r4, 0, 0x0123;
    shr.u32 %r6, %r5, 24;
    add.s32 %r7, %r3, %r6;
    bfi.b32 %r8, %r1, 0, 31, 1;
    shr.u32 %r9, %r8, 31;
    add.s32 %r10, %r7, %r9;
    shf.l.wrap.b32 %r11, %r1, %r1, 31;
    shr.u32 %r12, %r11, 31;
    add.s32 %r13, %r10, %r12;
    setp.eq.u32 %p1, %r1, 31;
    vote.sync.ballot.b32 %r14, %p1, -1;
    shr.u32 %r15, %r14, 31;
    add.s32 %r16, %r13, %r15;
    mov.u32 %r17, %lanemask_eq;
    shr.u32 %r18, %r17, 31;
    add.s32 %r19, %r16, %r18;
    mul.wide.s32 %rd2, %r19, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.b32 [%rd3], %r1;
    ret;
"""
        for fam in ("ttgir", "cute"):
            f = run(f_body, family=fam, threads=32)
            self.assertEqual(kir_errors(f), [])
            trace = evaluate(f, {"n": 1})
            self.assertNotIn("overflow", [t[0] for t in trace])
            offs = sorted(o for t in trace for o in t[2])
            rev5 = [int(f"{t:05b}"[::-1], 2) for t in range(32)]
            want = [
                rev5[t] + (t | 128) + 2 * (t & 1) + 1 + (t == 31) for t in range(32)
            ]
            self.assertEqual(offs, sorted(want))


# The default computes a capture from n (n + 7) and stores it before its first fork
# barrier; partition0 loads it between its first and second fork barrier and waits with
# its low bit as parity (Triton's captures through shared memory).
CAP_STORE = "    add.s32 %r40, %r20, 7;\n    st.shared.b32 [smem+32], %r40;\n"
CAP_LOAD = "    ld.shared.b32 %r41, [smem+32];\n"
CAP = WS.replace("    st.shared.b32 [smem+24], 33554689;\n",
                 "    st.shared.b32 [smem+24], 33554689;\n" + CAP_STORE).replace(
    "$P0:\n    barrier.sync 1;\n", "$P0:\n" + CAP_LOAD + "    barrier.sync 1;\n").replace(
    "    and.b32 %r10, %r9, 1;\n", "    and.b32 %r10, %r41, 1;\n")  # fmt: skip


class Captures(unittest.TestCase):
    def parity(self, f, n):
        (w,) = [
            t for t in evaluate(f, {"n": n, "x.shape.0": 64}, role=1) if t[0] == "wait"
        ]
        return w[3]

    def test_capture_forwarded(self):
        f = ws_facts(CAP)
        self.assertEqual(f.refusals, [])
        self.assertEqual([self.parity(f, n) for n in (0, 1, 2)], [1, 0, 1])
        self.assertEqual(kir_errors(f), [])
        # bytewise: two b8 stores, two b8 loads put back together
        st = "    st.shared.b8 [smem+32], %r40;\n    shr.u32 %r42, %r40, 8;\n" \
             "    st.shared.b8 [smem+33], %r42;\n"  # fmt: skip
        ld = "    ld.shared.b8 %r43, [smem+32];\n    ld.shared.b8 %r44, [smem+33];\n" \
             "    shl.b32 %r45, %r44, 8;\n    or.b32 %r41, %r43, %r45;\n"  # fmt: skip
        g = ws_facts(
            CAP.replace("    st.shared.b32 [smem+32], %r40;\n", st).replace(
                CAP_LOAD, ld
            )
        )
        self.assertEqual(g.refusals, [])
        self.assertEqual([self.parity(g, n) for n in (0, 1, 2)], [1, 0, 1])

    def test_unknown_captures_stay_data(self):
        cases = [
            # a store under a runtime condition may not happen
            CAP.replace("    st.shared.b32 [smem+32]",
                        "    setp.eq.s32 %p9, %r20, 0;\n    @%p9 st.shared.b32 [smem+32]"),
            # a value that differs between the threads
            CAP.replace("    add.s32 %r40, %r20, 7;", "    add.s32 %r40, %r1, 7;"),
            # a byte nobody stored
            CAP.replace(CAP_LOAD, "    ld.shared.b32 %r41, [smem+30];\n"),
            # a store after the default's first fork barrier
            CAP.replace(CAP_STORE, "").replace("    barrier.sync 1;\n    barrier.sync 1;\n    add.s32 %r21",
                        "    barrier.sync 1;\n" + CAP_STORE + "    barrier.sync 1;\n    add.s32 %r21", 1),
            # a store in a branch on a uniform runtime condition (a KIR `if`) may not happen
            CAP.replace(CAP_STORE, "    setp.eq.s32 %p9, %r20, 0;\n    @%p9 bra $SKIP;\n" + CAP_STORE
                        + "$SKIP:\n"),
        ]  # fmt: skip
        for text in cases:
            f = ws_facts(text)
            self.assertIn("reaches", f.refusals[0]["msg"] if f.refusals else "", text)

    def test_store_by_part_of_the_threads_stays_data(self):
        # a store that only some of the default's threads run may land before another
        # thread's earlier store of the bytes, or not happen (the elected lane under a lane
        # condition): thread 0 or warp 0 over every thread's n + 7, and the elected lane
        # of warp 0 if it is lane 0
        lane0 = "    and.b32 %r60, %r1, 31;\n    setp.eq.u32 %p9, %r60, 0;\n" \
                "    and.pred %p9, %p7, %p9;\n"  # fmt: skip
        cases = [
            CAP.replace(CAP_STORE, CAP_STORE + "    setp.eq.s32 %p9, %r1, 0;\n"
                        "    @%p9 st.shared.b32 [smem+32], 5;\n"),
            CAP.replace(CAP_STORE, CAP_STORE + "    @%p5 st.shared.b32 [smem+32], 5;\n"),
            CAP.replace("    st.shared.b32 [smem+32], %r40;\n",
                        lane0 + "    @%p9 st.shared.b32 [smem+32], %r40;\n"),
        ]  # fmt: skip
        for text in cases:
            f = ws_facts(text)
            self.assertIn("reaches", f.refusals[0]["msg"] if f.refusals else "", text)

    def test_overwrite_after_a_read_is_refused(self):
        f = ws_facts(
            CAP.replace(CAP_LOAD, CAP_LOAD + "    st.shared.b32 [smem+32], 0;\n")
        )
        self.assertIn("may overwrite a warp-specialize capture", f.refusals[0]["msg"])


# WS whose default may run no round (cfg.py, item rounds): with n < 1 it stores the exit
# state 3 for every worker before one fork barrier and returns, before the barrier init's
# arrive and the store; partition0 (warp 6) then never waits.
ZERO_IF = "    setp.lt.s32 %p9, %r20, 1;\n"
ZERO = WS.replace(
    "    st.shared.b32 [smem+24], 33554689;\n",
    ZERO_IF + "    @%p9 bra $NONE;\n    st.shared.b32 [smem+24], 33554689;\n",
).replace(
    "    st.shared.b32 [smem+24], 50529027;\n    barrier.sync 1;\n$RET:\n",
    "    st.shared.b32 [smem+24], 50529027;\n    barrier.sync 1;\n    bra.uni $RET;\n$NONE:\n"
    "    st.shared.b32 [smem+24], 50529027;\n    barrier.sync 1;\n$RET:\n",
)


class ZeroRounds(unittest.TestCase):
    def trace(self, f, role, meta=None, **inp):
        return [t[0] for t in evaluate(f, {"x.shape.0": 64, **inp}, meta, role=role)]

    def test_regions_run_under_the_decision(self):
        f = ws_facts(ZERO)
        self.assertEqual(f.refusals, [])
        self.assertEqual(kir_errors(f), [])
        # partition0's region (its wait) is a KIR `if` on the default's decision (n >= 1,
        # the default's round side); the code before it runs in either case
        (top,) = [s for s in f.roles[1]["body"] if s["op"] == "if"]
        self.assertEqual([s["ev"] for s in statements(top["then"]) if s["op"] == "event"],
                         ["wait"])  # fmt: skip
        self.assertEqual(top["else"], [])
        for n, worker, default in ((0, [], ["bar_init"]),
                                   (1, ["wait"], ["bar_init", "gstore", "arrive"]),
                                   (5, ["wait"], ["bar_init", "gstore", "arrive"])):  # fmt: skip
            self.assertEqual(self.trace(f, 1, n=n), worker, n)
            self.assertEqual(self.trace(f, 0, n=n), default, n)

    def test_decision_from_metadata(self):
        # a weak load of x: the metadata `load` both roles evaluate
        f = ws_facts(
            ZERO.replace(
                ZERO_IF,
                "    ld.global.b32 %r50, [%rd1];\n    setp.lt.s32 %p9, %r50, 1;\n",
            )
        )
        self.assertEqual(f.refusals, [])
        self.assertTrue(f.tensors["x"]["metadata"])
        lets = [s["expr"] for s in f.roles[1]["body"] if s["op"] == "let"]
        self.assertIn("'e': 'load'", str(lets))
        self.assertEqual(self.trace(f, 1, {"x": [0] * 64}, n=0), [])
        self.assertEqual(self.trace(f, 1, {"x": [1] * 64}, n=0), ["wait"])

    def test_capture_under_the_decision(self):
        # a capture the default stores on its round side holds where the region runs
        text = ZERO.replace("    st.shared.b32 [smem+24], 33554689;\n",
                            "    st.shared.b32 [smem+24], 33554689;\n" + CAP_STORE).replace(
            "$P0:\n    barrier.sync 1;\n", "$P0:\n" + CAP_LOAD + "    barrier.sync 1;\n").replace(
            "    and.b32 %r10, %r9, 1;\n", "    and.b32 %r10, %r41, 1;\n")  # fmt: skip
        f = ws_facts(text)
        self.assertEqual(f.refusals, [])
        waits = [[t for t in evaluate(f, {"n": n, "x.shape.0": 64}, role=1) if t[0] == "wait"]
                 for n in (0, 1, 2)]  # fmt: skip
        self.assertEqual(
            waits, [[], [("wait", "mbar_16", 0, 0)], [("wait", "mbar_16", 0, 1)]]
        )
        # on the zero-round side it would be no capture
        g = ws_facts(text.replace(CAP_STORE, "").replace(
            "$NONE:\n", "$NONE:\n" + CAP_STORE))  # fmt: skip
        self.assertIn("reaches", g.refusals[0]["msg"] if g.refusals else "")

    def test_refusals(self):
        cases = [
            # a strong load: data the workers cannot evaluate
            ("depends on data", ZERO.replace(ZERO_IF, "    ld.relaxed.gpu.global.b32 %r50, [%rd1];\n"
                                                      "    setp.lt.s32 %p9, %r50, 1;\n")),
            # the threads of the default disagree
            ("differs between the default's threads",
             ZERO.replace(ZERO_IF, "    setp.lt.u32 %p9, %r1, 64;\n")),
        ]  # fmt: skip
        for msg, text in cases:
            with self.subTest(msg):
                f = ws_facts(text)
                self.assertIn(msg, f.refusals[0]["msg"] if f.refusals else "")
                self.assertIn("run a round", f.refusals[0]["msg"])


# A switch on pid % 4 (LLVM's jump table of a select chain): cfg.lower_switches makes it a
# compare chain; each case picks the element the store writes.
SWITCH = """
JT:
    .branchtargets C0, C1, C2, C3;
    mov.u32 %r1, %tid.x;
    ld.param.u64 %rd1, [k_param_0];
    mov.u32 %r2, %ctaid.x;
    and.b32 %r3, %r2, 3;
    brx.idx %r3, JT;
C0:
    mov.b32 %r4, 5;
    bra.uni J;
C1:
    mov.b32 %r4, 9;
    bra.uni J;
C2:
    mov.b32 %r4, 2;
    bra.uni J;
C3:
    mov.b32 %r4, 7;
J:
    setp.eq.s32 %p1, %r1, 0;
    mul.wide.u32 %rd2, %r4, 4;
    add.s64 %rd3, %rd1, %rd2;
    @%p1 st.global.b32 [%rd3], %r1;
    ret;
"""


class Switches(unittest.TestCase):
    def test_compare_chain(self):
        f = run(SWITCH)
        self.assertEqual(f.refusals, [])
        got = [[t[2] for t in evaluate(f, {"x.shape.0": 64}, pid=(c, 0, 0)) if t[0] == "gstore"]
               for c in range(8)]  # fmt: skip
        self.assertEqual(got, [[[5]], [[9]], [[2]], [[7]]] * 2)
        self.assertEqual(kir_errors(f), [])
        g = run(SWITCH, family="cute")
        self.assertEqual(g.refusals, [])

    def test_index_out_of_range_is_refused(self):
        f = run(SWITCH.replace("and.b32 %r3, %r2, 3;", "and.b32 %r3, %r2, 7;"))
        self.assertIn("brx.idx at line", f.refusals[0]["msg"])
        self.assertIn("not provably in [0, 4)", f.refusals[0]["msg"])

    def test_one_target(self):
        # a table of one target checks its index too: a parameter may be out of [0, 1)
        one = SWITCH.replace(".branchtargets C0, C1, C2, C3;", ".branchtargets C0;")
        f = run(one.replace("and.b32 %r3, %r2, 3;", "ld.param.u32 %r3, [k_param_1];"))
        self.assertIn(
            "not provably in [0, 1)", f.refusals[0]["msg"] if f.refusals else ""
        )
        f = run(one.replace("and.b32 %r3, %r2, 3;", "mov.u32 %r3, 0;"))
        self.assertEqual(f.refusals, [])
        got = [t[2] for t in evaluate(f, {"x.shape.0": 64}) if t[0] == "gstore"]
        self.assertEqual(got, [[5]])

    def test_taken_names_are_refused(self):
        # the compare chain's names must be fresh: a kernel's own register of such a name
        body = "    .reg .pred %__swL_0;\n" + SWITCH
        line = 1 + kernel(body).splitlines().index("    brx.idx %r3, JT;")
        f = run(body.replace("%__swL_0", f"%__sw{line}_0"))
        self.assertIn(f"brx.idx at line {line}: the names of its compare chain",
                      f.refusals[0]["msg"] if f.refusals else "")  # fmt: skip
        self.assertEqual(run(body.replace("%__swL_0", "%__sw0_0")).refusals, [])

    def test_dispatch_is_not_lowered(self):
        # WS's brx.idx indexes by the state byte: still the dispatch (two roles)
        self.assertEqual(
            [r["name"] for r in ws_facts().roles], ["default", "partition0"]
        )

    def test_parsed_module_stays_as_parsed(self):
        # a module the caller parsed (to_kir parses each PTX once per task): the compare
        # chain goes into a copy of the entry function, so the module keeps its brx.idx
        # and translating it again gives the Facts of the text
        text = kernel(SWITCH)
        module = parse.parse(text)
        fn = module.function("k")
        parsed = (list(fn.body), dict(fn.regs))
        params = [{"ptx": "k_param_0", "kind": "ptr", "name": "x", "size": 8, "fields": {}},
                  {"ptx": "k_param_1", "kind": "scalar", "name": "n", "size": 4, "fields": {}}]  # fmt: skip
        pol = T.policy(
            "ttgir", src_root=SRC, params=params, tensors={"x": {"elem_bytes": 4}}
        )
        recs = [{"block": [128, 1, 1], "cluster": [1, 1, 1]}]
        got = [T.translate(text, "k", recs, pol, module) for _ in range(2)]
        self.assertEqual((list(fn.body), dict(fn.regs)), parsed)
        want = run(SWITCH)
        self.assertEqual(want.refusals, [])
        for f in got:
            self.assertEqual((f.roles, f.barriers, f.tensors, f.params, f.refusals),
                             (want.roles, want.barriers, want.tensors, want.params, []))  # fmt: skip


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
