# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Pure-Python tests of the CuTe frontend core (no CuTe DSL needed).

    python3 -I tests/cute/test_core.py      (or the buck target :test_core)

Exact BitVec folding and interval decisions, CuTe crd2idx, the generic MLIR reader, and
the translator on hand-written stages: the slice soundness check rejects a float
comparison deciding a global store, an integer guard becomes a per-thread mask, the KIR
of loop-carried and warp-specialized code validates with frontend/kir.py (a loop-carried
value that changes structure is rejected, a carried shuffle reads the previous
iteration's values); every op of the kernel is classified (unknown ops, vector memory
ops and side-effecting inline PTX off the slice are rejected, st.global inline PTX
becomes a gstore), global views built from data and elect.sync under a partial warp are
rejected; to_kir's metadata-write, provenance (metadata and host reads against skipped
launches), DSL-version (Python package and native libraries), validate-before-write and
addr-marking rules.
"""

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

if __package__ in (None, ""):
    # Run as a script: the project root becomes the package `lkv_root`, so the relative
    # imports below resolve as they do in the buck target.
    sys.dont_write_bytecode = True
    _root = ModuleType("lkv_root")
    _root.__path__ = [str(Path(__file__).resolve().parents[2])]
    sys.modules["lkv_root"] = _root
    __package__ = "lkv_root.tests.cute"  # noqa: A001

from ...frontend.cute import (  # noqa: E402, F401
    expr as E,
    layout as L,
    mlir_text,
    ops,
    ops_mem,
    ops_tmem,
    slicer,
    to_kir,
)
from ...frontend.cute.interp import (  # noqa: E402
    Kernel,
    Lay,
    Ptr,
    SplitWarps,
    Translator,
    Unsupported,
    View,
)


def _signed(v, w):
    v &= (1 << w) - 1
    return v - (1 << w) if w > 1 and v >> (w - 1) else v


def _ref(op, a, b, w):
    ua, ub = a & ((1 << w) - 1), b & ((1 << w) - 1)
    q = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1) if b else None
    return {
        "add": a + b, "sub": a - b, "mul": a * b, "sdiv": q, "srem": a - b * q if b else None,
        "udiv": ua // ub if ub else None, "urem": ua % ub if ub else None, "and": a & b,
        "or": a | b, "xor": a ^ b, "smin": min(a, b), "smax": max(a, b), "umin": min(ua, ub),
        "umax": max(ua, ub), "shl": ua << ub if ub < w else 0, "lshr": ua >> ub if ub < w else 0,
        "ashr": a >> min(ub, w - 1),
    }[op]  # fmt: skip


def _eval(e, env):
    """Reference evaluator: exact BitVec semantics, signed readings."""
    k = e[0]
    if k == "const":
        return e[2]
    if k == "var":
        return env[e[2]]
    if k == "bin":
        return _signed(_ref(e[2], _eval(e[3], env), _eval(e[4], env), e[1]), e[1])
    if k == "cmp":
        a, b, w = _eval(e[2], env), _eval(e[3], env), E.width(e[2])
        if e[1][0] == "u":
            a, b = a & ((1 << w) - 1), b & ((1 << w) - 1)
        return int(
            {
                "eq": a == b,
                "ne": a != b,
                "lt": a < b,
                "le": a <= b,
                "gt": a > b,
                "ge": a >= b,
            }[e[1].lstrip("su")]
        )
    if k == "select":
        return _eval(e[3], env) if _eval(e[2], env) else _eval(e[4], env)
    if k == "ext":
        a = _eval(e[3], env)
        return a if e[2] else a & ((1 << E.width(e[3])) - 1)
    if k == "trunc":
        return _signed(_eval(e[2], env), e[1])
    raise AssertionError(k)


class ExprTest(unittest.TestCase):
    def test_folding_matches_bitvec_semantics(self):
        rng = random.Random(7)
        for _ in range(4000):
            w = rng.choice((32, 64))
            a = rng.randrange(-(1 << (w - 1)), 1 << (w - 1))
            b = rng.choice(
                [
                    rng.randrange(-(1 << (w - 1)), 1 << (w - 1)),
                    rng.randrange(-8, 70),
                    0,
                    1,
                    -1,
                ]
            )
            op = rng.choice(E.BIN_OPS)
            want = _ref(op, a, b, w)
            if want is None:
                with self.assertRaises(E.ExprError):
                    E.bin(op, E.const(a, w), E.const(b, w))
                continue
            self.assertIs(
                E.bin(op, E.const(a, w), E.const(b, w)), E.const(want, w), (op, a, b, w)
            )

    def test_floor_and_ceil_division(self):
        for a in range(-9, 10):
            for b in (-4, -3, 3, 4):
                self.assertEqual(E.floordiv(E.const(a, 32), E.const(b, 32))[2], a // b)
                self.assertEqual(
                    E.ceildiv(E.const(a, 32), E.const(b, 32))[2], -(-a // b)
                )

    def test_interval_decisions_agree_with_every_lane(self):
        # Comparisons folded by interval reasoning must hold for every value of LANE.
        rng = random.Random(3)
        leaves = [
            E.LANE,
            E.const(0, 32),
            E.const(31, 32),
            E.const(7, 32),
            E.const(-5, 32),
            E.var("x", 32),
        ]

        def rand_expr(d):
            if d == 0:
                return rng.choice(leaves)
            op = rng.choice(
                ("add", "sub", "mul", "and", "urem", "lshr", "smin", "smax")
            )
            b = (
                E.const(rng.randrange(1, 9), 32)
                if op in ("urem", "lshr")
                else rand_expr(d - 1)
            )
            return E.bin(op, rand_expr(d - 1), b)

        decided = 0
        for _ in range(3000):
            a, b = rand_expr(2), rand_expr(2)
            op = rng.choice(E.CMP_OPS)
            c = E.cmp(op, a, b)
            if not E.is_const(c) or E.free_vars(a) | E.free_vars(b) <= {"%lane"}:
                continue
            decided += 1
            for lane in range(32):
                for x in (-(1 << 31), -1, 0, 5, (1 << 31) - 1):
                    env = {"%lane": lane, "x": x}
                    self.assertEqual(_eval(E.cmp(op, E.subst(a, {"%lane": E.const(lane, 32), "x": E.const(x, 32)}),
                                                 E.subst(b, {"%lane": E.const(lane, 32), "x": E.const(x, 32)})), env),
                                     c[2], (op, a, b, lane, x))  # fmt: skip
        self.assertGreater(decided, 50)

    def test_narrow_integers_lower_exactly(self):
        # i8 / i16 values (multicast masks, fast divmod shifts) reach KIR as zero-extended
        # i32: every lowered expression agrees with the narrow one on random values.
        rng = random.Random(11)
        ops_ = (
            "add",
            "sub",
            "mul",
            "and",
            "or",
            "xor",
            "shl",
            "lshr",
            "ashr",
            "smin",
            "umax",
            "udiv",
        )
        for _ in range(3000):
            w = rng.choice((8, 16))
            x, y, z = E.var("x", w), E.var("y", w), E.var("z", 32)
            inner = E.bin(rng.choice(("add", "xor", "mul")), y, E.trunc(z, w))
            e = E.bin(rng.choice(ops_), x, inner)
            e = rng.choice([E.ext(e, 32, True), E.ext(e, 32, False), E.ext(e, 64, True),
                            E.ext(E.cmp(rng.choice(("slt", "ult", "eq")), e, y), 32, False)])  # fmt: skip
            env = {"x": rng.randrange(-(1 << (w - 1)), 1 << (w - 1)),
                   "y": rng.randrange(-(1 << (w - 1)), 1 << (w - 1)), "z": rng.randrange(-(1 << 31), 1 << 31)}  # fmt: skip
            try:
                want = _eval(e, env)
            except TypeError:  # division by zero
                continue
            low = E.lower_narrow(e)
            self.assertFalse(E.has_narrow(low))
            zext = {
                **env,
                "x": env["x"] & ((1 << w) - 1),
                "y": env["y"] & ((1 << w) - 1),
            }
            self.assertEqual(
                _signed(_eval(low, zext), E.width(e)), _signed(want, E.width(e)), e
            )

    def test_table_and_select_chain(self):
        entries = tuple(E.var(f"v{i}", 32) for i in range(32))
        t = E.table(E.LANE, entries)
        self.assertIs(E.subst(t, {"%lane": E.const(5, 32)}), entries[5])
        j = E.to_json(E.subst(t, {"%lane": E.var("k", 32)}))
        self.assertEqual(j["e"], "select")  # a chain over the 32 entries


class LayoutTest(unittest.TestCase):
    def test_crd2idx_hierarchical_and_dynamic(self):
        shape, stride = L.parse_layout("((8,1),8,1):((1,0),?{i64 div=128},0)")
        stride = L.fill(stride, [E.var("s", 64)])
        idx = L.crd2idx([E.const(5, 32), E.const(3, 32), E.const(0, 32)], shape, stride)
        self.assertEqual(_eval(idx, {"s": 1000}), 5 + 3 * 1000)

    def test_flat_coordinate_splits_colexicographically(self):
        shape, stride = L.parse_layout("((4,2),3):((2,1),8)")
        got = [_eval(L.crd2idx(E.const(i, 32), shape, stride), {}) for i in range(24)]
        self.assertEqual(
            got, [(i % 4) * 2 + ((i // 4) % 2) + (i // 8) * 8 for i in range(24)]
        )

    def test_basis_strides_give_coordinates(self):
        shape, stride = L.parse_layout("((8,1),8,2):((1@1,0),16@0,64@1)")
        r = L.crd2idx(
            [[E.const(3, 32), E.const(0, 32)], E.const(2, 32), E.const(1, 32)],
            shape,
            stride,
        )
        self.assertEqual(
            {k: _eval(v, {}) for k, v in r.items()}, {(1,): 3 + 64, (0,): 32}
        )

    def test_nested_basis_path_is_outermost_first(self):
        _shape, stride = L.parse_layout("(2,4):(1@1@0@1,1@0@1)")
        self.assertEqual([b.path for b in L.leaves(stride)], [(1, 0, 1), (1, 0)])

    def test_incongruent_layout_is_rejected(self):
        with self.assertRaises(L.LayoutError):
            L.parse_layout("(4,2):(1)")


def _kernel_stage(body: str, nargs: int = 2, elem: str = "f32") -> str:
    """A cuda.kernel at the analyzed stage over `nargs` 1-D `elem` tensors of 128 elements."""
    ty = f'!cute.memref<{elem}, gmem, "(128):(1)">'
    args = ", ".join(f"%arg{i}: {ty}" for i in range(nargs))
    return f"""
"builtin.module"() ({{
  "gpu.module"() ({{
    "cuda.kernel"() <{{sym_name = "k"}}> ({{
    ^bb0({args}):
{body}
      "cuda.return"() : () -> () loc(#loc1)
    }}) : () -> () loc(#loc1)
  }}) : () -> () loc(#loc1)
}}) : () -> () loc(#loc1)
#loc1 = loc("/src/k.py":10:1)
#loc2 = loc("x > 0.0"("/src/k.py":11:5))
#loc3 = loc("/src/k.py":12:9)
"""


_TID = '      %0 = "nvvm.read.ptx.sreg.tid.x"() : () -> i32 loc(#loc1)\n'
# Thread t loads x[t]; the condition %4 (COND, defined from %0 = tid or %2 = x[t])
# decides whether it stores y[t] = 0.0.
_STORE = """      %1 = "cute.make_coord"(%0) : (i32) -> !cute.coord<"?"> loc(#loc1)
      %2 = "cute.memref.load"(%arg0, %1) : (!cute.memref<f32, gmem, "(128):(1)">, !cute.coord<"?">) -> f32 loc(#loc1)
      %3 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32 loc(#loc1)
COND
      "scf.if"(%4) ({
        "cute.memref.store"(%arg1, %1, %3) : (!cute.memref<f32, gmem, "(128):(1)">, !cute.coord<"?">, f32) -> () loc(#loc3)
        "scf.yield"() : () -> () loc(#loc3)
      }, {
      }) : (i1) -> () loc(#loc2)
"""
_FLOAT_COND = '      %4 = "arith.cmpf"(%2, %3) <{predicate = 2 : i64}> : (f32, f32) -> i1 loc(#loc2)'
_INT_COND = """      %9 = "arith.constant"() <{value = 100 : i32}> : () -> i32 loc(#loc2)
      %4 = "arith.cmpi"(%0, %9) <{predicate = 2 : i64}> : (i32, i32) -> i1 loc(#loc2)"""


def translate(text: str, nwarps: int = 4, p1: bool = False, cluster: tuple = (1, 1, 1)):
    """Translate every role of the kernel in `text`; returns (kernel state, roles). With
    `p1`, the P1 pass (ops_tmem: TMEM events, tcgen05.wait, named barriers)."""
    m = mlir_text.parse(text)
    kop = next(o for o in mlir_text.walk(m.top) if o.name == "cuda.kernel")
    k = Kernel(m, "/src/", block=(32 * nwarps, 1, 1), cluster=cluster)
    args = []
    for name, ty in kop.regions[0][0].args:
        bits = ops.memref_info(ty)[0]
        k.tensors[name] = {"bits": bits, "elem_bytes": bits // 8, "pack": 1, "shape": [E.const(128, 32)],
                           "strides": [E.const(1, 32)], "uses": set(), "metadata": False}  # fmt: skip
        args.append(
            View(
                Ptr("gmem", name, E.const(0, 64), bits),
                Lay(E.const(128, 32), E.const(1, 32)),
            )
        )
    spec = ops_tmem.spec_p1 if p1 else ops.spec
    effects = ops_tmem.effects_p1() if p1 else None
    k.slice = slicer.Slice(
        kop, ops.effect_operands(Translator(k, spec, p1=p1), effects)
    )
    todo, roles = [(0, nwarps)], []
    while todo:
        w0, nw = todo.pop(0)
        tr = Translator(k, spec, w0, nw, p1=p1)
        try:
            tr.run_block(kop.regions[0][0], args)
        except SplitWarps as s:
            todo = s.groups + todo
            continue
        roles.append(tr)
    return k, roles


def as_kir(k, roles) -> dict:
    """A complete KIR object around translated roles (for frontend/kir.py)."""
    tensors = [{"name": n, "kind": "global", "elem_bytes": t["elem_bytes"], "shape": [E.to_json(E.const(128, 32))],
                "strides": [E.to_json(E.const(1, 32))], "segments": None,
                "role": "output" if t["uses"] & {"gstore"} else "input", "metadata": False}
               for n, t in k.tensors.items()]  # fmt: skip
    one = E.to_json(E.const(1, 32))
    return {
        "kir_version": 1, "kernel": "k", "source": "k.py", "ir": "cute-mlir", "grid": [one, one, one],
        "params": [], "tensors": tensors, "barriers": [],
        "roles": [{"name": f"warps{r.w0}", "warps": r.nw, "threads_per_issue": 1, "body": r.stmts} for r in roles],
        "case": {"name": "t", "inputs": [{"name": "i", "params": {}}]},
    }  # fmt: skip


class ReaderTest(unittest.TestCase):
    def test_ops_regions_and_locations(self):
        m = mlir_text.parse(_kernel_stage(_TID + _STORE.replace("COND", _FLOAT_COND)))
        names = [o.name for o in mlir_text.walk(m.top)]
        self.assertEqual(names.count("scf.if"), 1)
        cmpf = next(o for o in mlir_text.walk(m.top) if o.name == "arith.cmpf")
        self.assertEqual(cmpf.props["predicate"], "2 : i64")
        self.assertIn('"x > 0.0"', m.resolve_loc(cmpf.loc))
        kop = next(o for o in mlir_text.walk(m.top) if o.name == "cuda.kernel")
        self.assertEqual(len(kop.regions[0][0].args), 2)


class TranslateTest(unittest.TestCase):
    def test_data_to_control_dependency_is_rejected(self):
        with self.assertRaises(Unsupported) as cm:
            translate(_kernel_stage(_TID + _STORE.replace("COND", _FLOAT_COND)))
        msg = str(cm.exception)
        self.assertIn("arith.cmpf at k.py:11", msg)
        self.assertIn("slice soundness", msg)

    def test_integer_guard_becomes_a_thread_mask(self):
        k, roles = translate(_kernel_stage(_TID + _STORE.replace("COND", _INT_COND)))
        self.assertEqual(len(roles), 1)
        load, store = [
            s for s in roles[0].stmts if s["op"] == "for"
        ]  # loops over the role's threads
        self.assertEqual([load["ub"], store["ub"]], [E.to_json(E.const(128, 32))] * 2)
        self.assertEqual(load["body"][-1]["ev"], "gload")  # every thread loads x[t]
        guarded = store["body"][-1]
        self.assertEqual(guarded["op"], "if")  # only threads with t < 100 store
        self.assertEqual(guarded["then"][0]["ev"], "gstore")
        self.assertEqual(guarded["then"][0]["loc"], "k.py:12")
        self.assertEqual(to_kir.validate(as_kir(k, roles)), [])

    def test_warp_branch_splits_roles_and_loops_carry_variables(self):
        # Warp 0 stores x[i] for i in 0..n with a loop-carried index; the other warps idle.
        body = (
            _TID
            + """      %1 = "arith.constant"() <{value = 32 : i32}> : () -> i32 loc(#loc1)
      %2 = "arith.divui"(%0, %1) : (i32, i32) -> i32 loc(#loc1)
      %3 = "arith.constant"() <{value = 0 : i32}> : () -> i32 loc(#loc1)
      %4 = "arith.cmpi"(%2, %3) <{predicate = 0 : i64}> : (i32, i32) -> i1 loc(#loc2)
      "scf.if"(%4) ({
        %5 = "arith.constant"() <{value = 1 : i32}> : () -> i32 loc(#loc1)
        %6 = "arith.constant"() <{value = 4 : i32}> : () -> i32 loc(#loc1)
        %7 = "scf.for"(%3, %6, %5, %3) ({
        ^bb0(%i: i32, %acc: i32):
          %8 = "arith.addi"(%acc, %1) : (i32, i32) -> i32 loc(#loc3)
          %9 = "cute.make_coord"(%acc) : (i32) -> !cute.coord<"?"> loc(#loc3)
          %10 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32 loc(#loc3)
          "cute.memref.store"(%arg1, %9, %10) : (!cute.memref<f32, gmem, "(128):(1)">, !cute.coord<"?">, f32) -> () loc(#loc3)
          "scf.yield"(%8) : (i32) -> () loc(#loc3)
        }) : (i32, i32, i32, i32) -> i32 loc(#loc3)
        "scf.yield"() : () -> () loc(#loc1)
      }, {
      }) : (i1) -> () loc(#loc2)
"""
        )
        k, roles = translate(_kernel_stage(body))
        self.assertEqual([(r.w0, r.nw) for r in roles], [(0, 1), (1, 3)])
        loop = next(s for s in roles[0].stmts if s["op"] == "for")
        assigns = [s["var"] for s in loop["body"] if s["op"] == "let"]
        carried = next(
            s["var"]
            for s in roles[0].stmts
            if s["op"] == "let" and s["expr"] == E.to_json(E.const(0, 32))
        )
        self.assertIn(carried, assigns)  # reassigned at the end of each iteration
        self.assertEqual(roles[1].stmts, [])
        self.assertEqual(to_kir.validate(as_kir(k, roles)), [])

    def test_loop_carried_value_changing_structure_is_rejected(self):
        # Each iteration swaps the two views, so the view stored through points into
        # %arg0, then %arg1: the slots carry its offset, not which tensor it points into.
        mt = '!cute.memref<f32, gmem, "(128):(1)">'
        consts = """      %c0 = "arith.constant"() <{value = 0 : i32}> : () -> i32 loc(#loc1)
      %c1 = "arith.constant"() <{value = 1 : i32}> : () -> i32 loc(#loc1)
      %c4 = "arith.constant"() <{value = 4 : i32}> : () -> i32 loc(#loc1)
"""
        store = f"""        %9 = "cute.make_coord"(%0) : (i32) -> !cute.coord<"?"> loc(#loc3)
        %10 = "arith.constant"() <{{value = 0.000000e+00 : f32}}> : () -> f32 loc(#loc3)
        "cute.memref.store"(%a, %9, %10) : ({mt}, !cute.coord<"?">, f32) -> () loc(#loc3)
"""
        loops = {
            "for": f"""      %r:2 = "scf.for"(%c0, %c4, %c1, %arg0, %arg1) ({{
      ^bb0(%i: i32, %a: {mt}, %b: {mt}):
{store}        "scf.yield"(%b, %a) : ({mt}, {mt}) -> () loc(#loc3)
      }}) : (i32, i32, i32, {mt}, {mt}) -> ({mt}, {mt}) loc(#loc3)
""",
            "while": f"""      %r:3 = "scf.while"(%c0, %arg0, %arg1) ({{
      ^bb0(%i: i32, %x: {mt}, %y: {mt}):
        %lt = "arith.cmpi"(%i, %c4) <{{predicate = 2 : i64}}> : (i32, i32) -> i1 loc(#loc3)
        "scf.condition"(%lt, %i, %x, %y) : (i1, i32, {mt}, {mt}) -> () loc(#loc3)
      }}, {{
      ^bb0(%j: i32, %a: {mt}, %b: {mt}):
{store}        %j1 = "arith.addi"(%j, %c1) : (i32, i32) -> i32 loc(#loc3)
        "scf.yield"(%j1, %b, %a) : (i32, {mt}, {mt}) -> () loc(#loc3)
      }}) : (i32, {mt}, {mt}) -> (i32, {mt}, {mt}) loc(#loc3)
""",
        }
        for name, loop in loops.items():
            for p1 in (False, True):
                with self.subTest(loop=name, p1=p1):
                    with self.assertRaisesRegex(Unsupported, "changes structure"):
                        translate(_kernel_stage(_TID + consts + loop), nwarps=1, p1=p1)

    def test_loop_carried_shuffle_reads_the_previous_iteration(self):
        # x = shfl_up(x, 1), or x + shfl_up(x, 1): each lane's new x reads another lane's
        # old x, so no let of the carried family may read another entry of the family
        # (its lets run one after another).
        for nwarps, update in ((1, "%s, %c0"), (2, "%x, %s")):
            body = (
                _TID
                + f"""      %c0 = "arith.constant"() <{{value = 0 : i32}}> : () -> i32 loc(#loc1)
      %c1 = "arith.constant"() <{{value = 1 : i32}}> : () -> i32 loc(#loc1)
      %c2 = "arith.constant"() <{{value = 2 : i32}}> : () -> i32 loc(#loc1)
      %cm = "arith.constant"() <{{value = -1 : i32}}> : () -> i32 loc(#loc1)
      %r = "scf.for"(%c0, %c2, %c1, %0) ({{
      ^bb0(%i: i32, %x: i32):
        %s = "nvvm.shfl.sync"(%cm, %x, %c1, %c0) <{{kind = #nvvm<shfl_kind up>}}> : (i32, i32, i32, i32) -> i32 loc(#loc3)
        %y = "arith.addi"({update}) : (i32, i32) -> i32 loc(#loc3)
        "scf.yield"(%y) : (i32) -> () loc(#loc3)
      }}) : (i32, i32, i32, i32) -> i32 loc(#loc3)
      %9 = "cute.make_coord"(%r) : (i32) -> !cute.coord<"?"> loc(#loc3)
      %10 = "arith.constant"() <{{value = 0.000000e+00 : f32}}> : () -> f32 loc(#loc3)
      "cute.memref.store"(%arg1, %9, %10) : (!cute.memref<f32, gmem, "(128):(1)">, !cute.coord<"?">, f32) -> () loc(#loc3)
"""
            )
            with self.subTest(nwarps=nwarps, update=update):
                k, roles = translate(_kernel_stage(body), nwarps=nwarps)
                stmts = roles[0].stmts
                loop = next(s for s in stmts if s["op"] == "for")
                before = {s["var"] for s in stmts if s["op"] == "let"}
                family = {s["var"] for s in loop["body"] if s["op"] == "let"} & before
                self.assertEqual(len(family), 32 * nwarps)
                for s in loop["body"]:
                    if s["op"] == "let" and s["var"] in family:
                        others = family - {s["var"]}
                        self.assertFalse(_vars(s["expr"]) & others, s["var"])
                self.assertEqual(to_kir.validate(as_kir(k, roles)), [])


def _vars(e) -> set:
    """Variable names a KIR expression reads."""
    if isinstance(e, dict):
        own = {e["name"]} if e.get("e") == "var" else set()
        return own.union(*map(_vars, e.values()))
    if isinstance(e, list):
        return set().union(*map(_vars, e))
    return set()


def _events(stmts) -> list:
    out = []
    for s in stmts:
        if s["op"] == "event":
            out.append(s)
        for k in ("then", "else", "body"):
            out += _events(s.get(k, []))
    return out


def _asm(asm: str, operands: list, types: list, constraints: str) -> str:
    return (f'      "llvm.inline_asm"({", ".join(operands)}) <{{asm_dialect = 0 : i64, asm_string = "{asm}", '
            f'constraints = "{constraints}", has_side_effects}}> : ({", ".join(types)}) -> () loc(#loc3)\n')  # fmt: skip


def _elem_address(elem: str, scale: int) -> str:
    """%4 = the address (cute.ptrtoint) of element scale * tid of %arg1."""
    mt, pt = f'!cute.memref<{elem}, gmem, "(128):(1)">', f"!cute.ptr<{elem}, gmem>"
    return (
        _TID
        + f"""      %c9 = "arith.constant"() <{{value = {scale} : i32}}> : () -> i32 loc(#loc1)
      %8 = "arith.muli"(%0, %c9) : (i32, i32) -> i32 loc(#loc1)
      %1 = "cute.get_iter"(%arg1) : ({mt}) -> {pt} loc(#loc1)
      %2 = "cute.make_int_tuple"(%8) : (i32) -> !cute.int_tuple<"?"> loc(#loc1)
      %3 = "cute.add_offset"(%1, %2) : ({pt}, !cute.int_tuple<"?">) -> {pt} loc(#loc1)
      %4 = "cute.ptrtoint"(%3) : ({pt}) -> i64 loc(#loc1)
"""
    )


# A scale-factor store helper: two bytes packed into one st.global.u16.
_ST_U16 = "{\\0A.reg .u16 packed16;\\0Acvt.u16.u32 packed16, $1;\\0Ast.global.u16 [$0], packed16;\\0A}\\0A"


class ClassificationTest(unittest.TestCase):
    """Every op of the kernel has a table meaning, kept or not, and inline PTX is
    classified statement by statement."""

    LOOP = """      %c0 = "arith.constant"() <{value = 0 : i32}> : () -> i32 loc(#loc1)
      %c4 = "arith.constant"() <{value = 4 : i32}> : () -> i32 loc(#loc1)
      %c1 = "arith.constant"() <{value = 1 : i32}> : () -> i32 loc(#loc1)
      "scf.for"(%c0, %c4, %c1) ({
      ^bb0(%i: i32):
        BODY
        "scf.yield"() : () -> () loc(#loc3)
      }) : (i32, i32, i32) -> () loc(#loc3)
"""

    def assert_rejected(self, text: str, *needles: str):
        with self.assertRaises(Unsupported) as cm:
            translate(text, nwarps=1)
        for n in needles:
            self.assertIn(n, str(cm.exception))

    def test_unknown_op_off_the_slice_is_rejected(self):
        body = self.LOOP.replace(
            "BODY",
            '"nvvm.cp.async.bulk.tensor.shared.cluster.global"(%i) : (i32) -> () loc(#loc3)',
        )
        self.assert_rejected(_kernel_stage(body), "nvvm.cp.async.bulk.tensor.shared.cluster.global at k.py:12",
                             "not in the CuTe op table")  # fmt: skip

    def test_vector_memory_ops_are_not_data(self):
        # Register-vector ops are data; the vector dialect's memory ops are not in the table.
        mt = '!cute.memref<f32, gmem, "(128):(1)">'
        store = (
            _TID
            + f"""      %m = "builtin.unrealized_conversion_cast"(%arg1) : ({mt}) -> memref<128xf32, 1> loc(#loc1)
      %i = "arith.index_cast"(%0) : (i32) -> index loc(#loc1)
      %x = "arith.constant"() <{{value = dense<0.000000e+00> : vector<4xf32>}}> : () -> vector<4xf32> loc(#loc1)
      "vector.store"(%x, %m, %i) : (vector<4xf32>, memref<128xf32, 1>, index) -> () loc(#loc3)
"""
        )
        self.assert_rejected(
            _kernel_stage(store), "vector.store at k.py:12", "not in the CuTe op table"
        )
        extract = (
            _TID
            + """      %x = "arith.constant"() <{value = dense<0.000000e+00> : vector<4xf32>}> : () -> vector<4xf32> loc(#loc1)
      %e = "vector.extract"(%x) <{static_position = array<i64: 0>}> : (vector<4xf32>) -> f32 loc(#loc3)
"""
        )
        _k, roles = translate(_kernel_stage(extract), nwarps=1)
        self.assertEqual(roles[0].stmts, [])

    def test_data_op_off_the_slice_is_dropped(self):
        body = self.LOOP.replace(
            "BODY", '%j = "arith.addi"(%i, %c1) : (i32, i32) -> i32 loc(#loc3)'
        )
        _k, roles = translate(_kernel_stage(body), nwarps=1)
        self.assertEqual(roles[0].stmts, [])

    def test_side_effecting_asm_without_results_is_rejected(self):
        body = _TID + _asm(
            "mbarrier.arrive.shared::cta.b64 _, [$0];", ["%0"], ["i32"], "r"
        )
        self.assert_rejected(
            _kernel_stage(body), "llvm.inline_asm at k.py:12", "mbarrier.arrive"
        )

    def test_data_asm_without_results_is_dropped(self):
        k, roles = translate(
            _kernel_stage(_asm("bar.sync 1, 64;", [], [], "")), nwarps=1
        )
        self.assertEqual(roles[0].stmts, [])

    def test_st_global_u16_is_a_two_byte_gstore(self):
        body = _elem_address("i8", 2) + _asm(
            _ST_U16, ["%4", "%0"], ["i64", "i32"], "l,r"
        )
        k, roles = translate(_kernel_stage(body, elem="i8"), nwarps=1)
        (loop,) = roles[0].stmts  # one store per thread of the warp
        self.assertEqual((loop["op"], loop["ub"]["v"]), ("for", 32))
        (ev,) = _events(roles[0].stmts)
        self.assertEqual(
            (ev["ev"], ev["tensor"], ev["loc"]), ("gstore", "%arg1", "k.py:12")
        )
        self.assertEqual(ev["offsets"]["shape"], [2])  # bytes 2t and 2t + 1
        self.assertEqual(to_kir.validate(as_kir(k, roles)), [])

    def test_st_global_u32_in_braces_on_one_line_is_kept(self):
        body = _elem_address("f32", 1) + _asm(
            "{ st.global.u32 [$0], $1; }", ["%4", "%0"], ["i64", "i32"], "l,r"
        )
        _k, roles = translate(_kernel_stage(body), nwarps=1)
        self.assertEqual(
            [(e["ev"], e["offsets"]["shape"]) for e in _events(roles[0].stmts)],
            [("gstore", [1])],
        )

    def test_st_global_narrower_than_the_storage_element_is_rejected(self):
        body = _elem_address("f32", 1) + _asm(
            _ST_U16, ["%4", "%0"], ["i64", "i32"], "l,r"
        )
        self.assert_rejected(_kernel_stage(body), "sub-element access")

    def test_predicated_st_global_is_rejected(self):
        asm = "{\\0A.reg .pred p;\\0Asetp.ne.u32 p, $1, 0;\\0A@p st.global.u32 [$0], $1;\\0A}"
        body = _elem_address("f32", 1) + _asm(asm, ["%4", "%0"], ["i64", "i32"], "l,r")
        self.assert_rejected(_kernel_stage(body), "under a PTX predicate")

    def test_bulk_reduce_is_a_gatomic(self):
        # A bulk f16 reduce-add helper: 256 bytes = 128 f16 elements from element 0.
        mt, pt = '!cute.memref<f16, gmem, "(128):(1)">', "!cute.ptr<f16, gmem>"
        body = _TID + f"""      %1 = "cute.get_iter"(%arg1) : ({mt}) -> {pt} loc(#loc1)
      %n = "arith.constant"() <{{value = 256 : i32}}> : () -> i32 loc(#loc1)
""" + _asm("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [$0], [$1], $2;",
           ["%1", "%0", "%n"], [pt, "i32", "i32"], "l,r,r")  # fmt: skip
        _k, roles = translate(_kernel_stage(body, elem="f16"), nwarps=1)
        evs = _events(roles[0].stmts)
        self.assertEqual(
            [(e["ev"], e["offsets"]["shape"]) for e in evs], [("gatomic", [128])]
        )

    def test_st_global_through_an_integer_address_is_rejected(self):
        body = (
            _TID
            + '      %4 = "arith.constant"() <{value = 4096 : i64}> : () -> i64 loc(#loc1)\n'
        )
        body += _asm("st.global.u32 [$0], $1;", ["%4", "%0"], ["i64", "i32"], "l,r")
        self.assert_rejected(_kernel_stage(body), "not a kernel tensor pointer")


_STORE_VIEW = """      %c = "cute.make_coord"(%0) : (i32) -> !cute.coord<"?"> loc(#loc1)
      %x = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32 loc(#loc1)
      "cute.memref.store"(%v, %c, %x) : (MT, !cute.coord<"?">, f32) -> () loc(#loc3)
"""


class ViewSpaceTest(unittest.TestCase):
    """Global and generic views computed from data, or of unknown space, are rejected; a
    generic view takes its pointer's space."""

    def run_view(self, define: str, mt: str):
        return translate(
            _kernel_stage(_TID + define + _STORE_VIEW.replace("MT", mt)), nwarps=1
        )

    def test_global_view_from_an_integer_is_rejected(self):
        define = """      %a = "arith.constant"() <{value = 4096 : i64}> : () -> i64 loc(#loc1)
      %p = "cute.inttoptr"(%a) : (i64) -> !cute.ptr<f32, gmem> loc(#loc1)
      %v = "cute.make_view"(%p) : (!cute.ptr<f32, gmem>) -> !cute.memref<f32, gmem, "(4):(1)"> loc(#loc1)
"""
        with self.assertRaises(Unsupported) as cm:
            self.run_view(define, '!cute.memref<f32, gmem, "(4):(1)">')
        self.assertIn("gmem view computed from data", str(cm.exception))
        self.assertIn("slice soundness", str(cm.exception))

    def test_generic_view_of_a_global_pointer_is_a_gstore(self):
        define = """      %p = "cute.get_iter"(%arg1) : (!cute.memref<f32, gmem, "(128):(1)">) -> !cute.ptr<f32, gmem> loc(#loc1)
      %v = "cute.make_view"(%p) : (!cute.ptr<f32, gmem>) -> !cute.memref<f32, generic, "(128):(1)"> loc(#loc1)
"""
        _k, roles = self.run_view(define, '!cute.memref<f32, generic, "(128):(1)">')
        self.assertEqual([e["ev"] for e in _events(roles[0].stmts)], ["gstore"])

    def test_generic_view_of_registers_is_data(self):
        define = """      %r = "cute.memref.alloca"() : () -> !cute.memref<f32, rmem, "(4):(1)"> loc(#loc1)
      %p = "cute.get_iter"(%r) : (!cute.memref<f32, rmem, "(4):(1)">) -> !cute.ptr<f32, rmem> loc(#loc1)
      %v = "cute.make_view"(%p) : (!cute.ptr<f32, rmem>) -> !cute.memref<f32, generic, "(4):(1)"> loc(#loc1)
"""
        _k, roles = self.run_view(define, '!cute.memref<f32, generic, "(4):(1)">')
        self.assertEqual(_events(roles[0].stmts), [])

    def test_generic_view_of_unknown_space_is_rejected(self):
        define = '      %v = "cute.memref.alloca"() : () -> !cute.memref<f32, generic, "(4):(1)"> loc(#loc1)\n'
        with self.assertRaises(Unsupported) as cm:
            self.run_view(define, '!cute.memref<f32, generic, "(4):(1)">')
        self.assertIn("address space is not known", str(cm.exception))


class ElectTest(unittest.TestCase):
    """elect.sync needs every lane of a warp that reaches it."""

    ELECTED_STORE = """      %e = "nvvm.elect.sync"() : () -> i1 loc(#loc3)
      "scf.if"(%e) ({
        "cute.memref.store"(%arg1, %c, %x) : (!cute.memref<f32, gmem, "(128):(1)">, !cute.coord<"?">, f32) -> () loc(#loc3)
        "scf.yield"() : () -> () loc(#loc3)
      }, {
      }) : (i1) -> () loc(#loc3)
"""
    PRELUDE = (
        _TID
        + """      %z = "arith.constant"() <{value = 5 : i32}> : () -> i32 loc(#loc1)
      %c = "cute.make_coord"(%z) : (i32) -> !cute.coord<"?"> loc(#loc1)
      %x = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32 loc(#loc1)
"""
    )

    def test_elect_in_a_full_warp_issues_once(self):
        _k, roles = translate(
            _kernel_stage(self.PRELUDE + self.ELECTED_STORE), nwarps=1
        )
        self.assertEqual(
            [s["ev"] for s in roles[0].stmts], ["gstore"]
        )  # one event, no thread loop

    def test_elect_under_a_partial_lane_mask_is_rejected(self):
        body = (
            self.PRELUDE
            + """      %l = "nvvm.read.ptx.sreg.laneid"() : () -> i32 loc(#loc1)
      %c16 = "arith.constant"() <{value = 16 : i32}> : () -> i32 loc(#loc1)
      %lt = "arith.cmpi"(%l, %c16) <{predicate = 2 : i64}> : (i32, i32) -> i1 loc(#loc2)
      "scf.if"(%lt) ({
"""
            + self.ELECTED_STORE.replace("\n      ", "\n        ")
            + """        "scf.yield"() : () -> () loc(#loc2)
      }, {
      }) : (i1) -> () loc(#loc2)
"""
        )
        with self.assertRaises(Unsupported) as cm:
            translate(_kernel_stage(body), nwarps=1)
        self.assertIn("nvvm.elect.sync at k.py:12", str(cm.exception))
        self.assertIn("part of warp(s) [0]", str(cm.exception))


def _tensor_arg(block: int, offset: int, n: int, dtype: str = "int32") -> dict:
    return {
        "kind": "tensor",
        "dtype": dtype,
        "shape": [n],
        "stride": [1],
        "storage": [block, 1 << 20],
        "byte_offset": offset,
    }


class ToKirRulesTest(unittest.TestCase):
    def variant(self, uses: dict):
        """A translated variant as check_metadata_writes sees it; `cu` is metadata."""
        tensors = {n: {"uses": set(u), "metadata": n == "cu"} for n, u in uses.items()}
        return SimpleNamespace(
            cls="K",
            k=SimpleNamespace(tensors=tensors),
            host_args=[(n, "tensor", None) for n in uses],
        )

    def test_metadata_written_by_the_launch_is_rejected(self):
        v = self.variant({"cu": {"gload", "gstore"}, "y": {"gstore"}})
        with self.assertRaises(Unsupported) as cm:
            to_kir.Variant.check_metadata_writes(
                v, [("i", [_tensor_arg(1, 0, 4), _tensor_arg(2, 0, 8)])]
            )
        self.assertIn("K writes its metadata tensors ['cu']", str(cm.exception))

    def test_metadata_sharing_bytes_with_a_written_tensor_is_rejected(self):
        v = self.variant({"cu": {"gload"}, "y": {"gstore"}})
        with self.assertRaises(
            Unsupported
        ) as cm:  # bytes [0, 16) and [8, 40) of one storage block
            to_kir.Variant.check_metadata_writes(
                v, [("i", [_tensor_arg(1, 0, 4), _tensor_arg(1, 8, 8, "float32")])]
            )
        self.assertIn(
            "input i: metadata tensor cu shares storage with the written tensor y",
            str(cm.exception),
        )
        # Disjoint views of one block, and an unwritten neighbour, are fine.
        to_kir.Variant.check_metadata_writes(
            v, [("i", [_tensor_arg(1, 0, 4), _tensor_arg(1, 16, 8, "float32")])]
        )
        to_kir.Variant.check_metadata_writes(self.variant({"cu": {"gload"}, "y": {"gload"}}),
                                             [("i", [_tensor_arg(1, 0, 4), _tensor_arg(1, 0, 4)])])  # fmt: skip

    def test_dump_of_another_dsl_is_refused(self):
        for version, native in (("4.6.1", ops.DSL_VERSION), (ops.DSL_VERSION, "4.6.1")):
            dsl = {
                "version": version,
                "installed": {"nvidia-cutlass-dsl-libs-base": native},
            }
            with self.assertRaises(Unsupported) as cm:
                to_kir.translate({}, {"dsl": dsl}, {}, [])
            self.assertIn(
                f"made with CuTe DSL {version} (native libraries nvidia-cutlass-dsl-libs-base {native}); "
                f"the op table was checked against {ops.DSL_VERSION}",
                str(cm.exception),
            )

    def provenance(self, sources: list, reads: list, written: set):
        """_check_provenance on two launches of one input: launch 0 (kernel A, translated,
        writes the tensors in `written`) gets storage blocks 1 and 2; launch 1 (kernel B,
        translated, reads metadata `cu` with `sources`) after host reads `reads`.
        Returns launch 1's report entry."""

        def variant(uses, meta):
            tensors = {
                n: {"uses": set(u), "metadata": n in meta} for n, u in uses.items()
            }
            return SimpleNamespace(
                k=SimpleNamespace(tensors=tensors),
                host_args=[(n, "tensor", None) for n in uses],
            )

        a = variant(
            {"x": {"gstore"} if "x" in written else {"gload"}, "y": {"gload"}}, set()
        )
        b = variant({"cu": {"gload"}}, {"cu"})
        dump = {"compiles": [{"index": 0, "digest": "A"}, {"index": 1, "digest": "B"}],
                "inputs": [{"name": "i", "host_reads": reads, "launches": [
                    {"compile": 0, "args": [{"kind": "tensor", "storage": [1, 64]}, {"kind": "tensor", "storage": [2, 64]}]},
                    {"compile": 1, "args": [{"kind": "tensor", "storage": [3, 64], "sources": sources}]}]}]}  # fmt: skip
        by_digest = {"A": [{"callable": "m.A"}], "B": [{"callable": "m.B"}]}
        results = {"A": (a, None, {}), "B": (b, None, {})}
        to_kir._check_provenance(dump, by_digest, results)
        return results["B"][2]

    def test_metadata_computed_from_a_skipped_write_is_refused(self):
        # cu (block 3) was computed on the host from block 1 after launch 0 wrote it.
        entry = self.provenance([[1, 1]], [], {"x"})
        self.assertIn(
            "metadata tensor cu of input i may hold data that launch 0 (m.A) writes",
            entry["error"],
        )
        self.assertNotIn(
            "error", self.provenance([[1, 1]], [], set())
        )  # launch 0 only reads x
        self.assertNotIn(
            "error", self.provenance([[1, 0]], [], {"x"})
        )  # computed before launch 0

    def test_host_read_of_a_skipped_write_before_a_launch_is_refused(self):
        read = {
            "op": "TensorBase.item",
            "loc": "k.py:7",
            "after": 1,
            "sources": [[1, 1]],
        }
        entry = self.provenance([], [read], {"x"})
        self.assertIn("before launch 1 the host reads data (TensorBase.item at k.py:7) that launch 0 (m.A) may write",
                      entry["error"])  # fmt: skip
        self.assertNotIn(
            "error", self.provenance([], [{**read, "after": 2}], {"x"})
        )  # read after launch 1
        self.assertNotIn(
            "error", self.provenance([], [{**read, "sources": [[2, 1]]}], {"x"})
        )  # y is not written

    def test_kir_is_written_only_when_valid(self):
        k, roles = translate(_kernel_stage(_TID + _STORE.replace("COND", _INT_COND)))
        good, bad = as_kir(k, roles), as_kir(k, roles)
        bad["roles"][0]["warps"] = 0
        out = Path(tempfile.mkdtemp(prefix="lkv_cute_test_kir_"))
        entry = {}
        self.assertFalse(to_kir.write_kir(out / "bad.json", bad, entry))
        self.assertFalse((out / "bad.json").exists())
        self.assertIn("invalid KIR (frontend/kir.py): $.roles[0].warps", entry["error"])
        entry = {}
        self.assertTrue(to_kir.write_kir(out / "good.json", good, entry))
        self.assertEqual(json.loads((out / "good.json").read_text()), good)
        self.assertNotIn("error", entry)

    def test_addr_marks_shift_and_trunc_on_address_paths_only(self):
        def c(v):
            return {"e": "const", "w": 64, "v": v}

        a = {"e": "var", "w": 64, "name": "a"}
        offset = {
            "e": "trunc",
            "w": 32,
            "a": {"e": "bin", "w": 64, "o": "shl", "a": a, "b": c(2)},
        }
        flag = {"e": "bin", "w": 64, "o": "shl", "a": a, "b": c(3)}
        body = [
            {"op": "let", "var": "off", "expr": offset},
            {"op": "let", "var": "f", "expr": flag},
            {"op": "if", "cond": {"e": "cmp", "w": 1, "o": "eq", "a": {"e": "var", "w": 64, "name": "f"}, "b": c(0)},
             "then": [], "else": []},
            {"op": "event", "ev": "gload", "tensor": "x", "mask": None, "loc": "k.py:1",
             "offsets": {"t": "splat", "shape": [1], "w": 64,
                         "a": {"e": "ext", "w": 64, "signed": True, "a": {"e": "var", "w": 32, "name": "off"}}}},
        ]  # fmt: skip
        to_kir.mark_addr({"roles": [{"body": body}]})
        self.assertTrue(offset["addr"] and offset["a"]["addr"])
        self.assertNotIn("addr", flag)


# P1 pass (ops_tmem.py): warp 1 copies scale factors into TMEM cols [80, 84) of the
# fake-zero-base allocation (ALLOC columns), warp 0 loads cols [64, 96) of its lane
# quarter and waits for the load. LD_OFF / CP_OFF are the column offsets (i32 values).
_P1_STAGE = """      %0 = "nvvm.read.ptx.sreg.tid.x"() : () -> i32 loc(#loc1)
      %c32 = "arith.constant"() <{value = 32 : i32}> : () -> i32 loc(#loc1)
      %w = "arith.divui"(%0, %c32) : (i32, i32) -> i32 loc(#loc1)
      %c0 = "arith.constant"() <{value = 0 : i32}> : () -> i32 loc(#loc1)
      %c1 = "arith.constant"() <{value = 1 : i32}> : () -> i32 loc(#loc1)
      %cn = "arith.constant"() <{value = ALLOC : i32}> : () -> i32 loc(#loc1)
      %cb = "arith.constant"() <{value = BASE : i32}> : () -> i32 loc(#loc1)
      %hold = "cute_nvgpu.arch.get_dyn_smem"() : () -> !cute.ptr<i32, smem, align<32>> loc(#loc1)
      %a = "cute.assume"(%cb) : (i32) -> !cute.i32<divby 16> loc(#loc1)
      %base = "cute.inttoptr"(%a) : (!cute.i32<divby 16>) -> !cute.ptr<f32, tmem, align<16>> loc(#loc1)
      %is1 = "arith.cmpi"(%w, %c1) <{predicate = 0 : i64}> : (i32, i32) -> i1 loc(#loc2)
      "scf.if"(%is1) ({
        %t80 = CP_OFF
        %p80 = "cute.add_offset"(%base, %t80) : (!cute.ptr<f32, tmem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<f32, tmem, align<16>> loc(#loc3)
        %sf = "cute.recast_iter"(%p80) : (!cute.ptr<f32, tmem, align<16>>) -> !cute.ptr<f8E8M0FNU, tmem, align<16>> loc(#loc3)
        %desc = "cute_nvgpu.make_umma_smem_desc"(%hold) <{layout = #cute.layout<"((32,16)):((16,1))">, major = #cute_nvgpu.major<k>}> : (!cute.ptr<i32, smem, align<32>>) -> !cute_nvgpu.smem_desc loc(#loc3)
        "cute_nvgpu.arch.copy.SM100.copy_s2t"(%desc, %sf) <{bits = 128 : i32, broadcast = #cute_nvgpu.copy_s2t_broadcast_mode<x4>, cta = 1 : i32, dp = 32 : i32}> : (!cute_nvgpu.smem_desc, !cute.ptr<f8E8M0FNU, tmem, align<16>>) -> () loc(#loc3)
        "scf.yield"() : () -> () loc(#loc3)
      }, {
        "cute_nvgpu.arch.sm100.alloc_tmem"(%cn, %hold) : (i32, !cute.ptr<i32, smem, align<32>>) -> () loc(#loc1)
        %t64 = LD_OFF
        %p64 = "cute.add_offset"(%base, %t64) : (!cute.ptr<f32, tmem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<f32, tmem, align<16>> loc(#loc3)
        %v = "cute_nvgpu.arch.copy.SM100.tmem_load"(%p64) <{num_b = 32 : i32, num_dp = 32 : i32, num_rep = 32 : i32}> : (!cute.ptr<f32, tmem, align<16>>) -> vector<32xi32> loc(#loc3)
        "nvvm.tcgen05.wait"() <{kind = #nvvm.tcgen05_wait<load>}> : () -> () loc(#loc3)
        EXTRA
        "scf.yield"() : () -> () loc(#loc3)
      }) : (i1) -> () loc(#loc2)
"""
_OFF = '"cute.make_int_tuple"(%cN) : (i32) -> !cute.int_tuple<"?"> loc(#loc3)'


def _p1_stage(alloc=512, base=0, ld="64", cp="80", extra="", nwarps=2):
    def off(v):
        if v.startswith("%"):
            return _OFF.replace("%cN", v)
        return (
            f'"cute.make_int_tuple"(%k{v}) : (i32) -> !cute.int_tuple<"?"> loc(#loc3)'
        )

    consts = "".join(
        f'      %k{v} = "arith.constant"() <{{value = {v} : i32}}> : () -> i32 loc(#loc1)\n'
        for v in {ld, cp}
        if not v.startswith("%")
    )
    body = (
        _P1_STAGE.replace("ALLOC", str(alloc))
        .replace("BASE", str(base))
        .replace("CP_OFF", off(cp))
        .replace("LD_OFF", off(ld))
        .replace("EXTRA", extra)
    )
    return _kernel_stage(consts + body, nargs=0), nwarps


# P1 pass: one warp issues a block-scaled 128xN f8 MMA (scale vector 32, K = 32) with D at
# column 0 and its SFA / SFB at columns SFA / SFB of the fake-zero-base allocation.
_P1_MMA_STAGE = """      %c0 = "arith.constant"() <{value = 0 : i32}> : () -> i32 loc(#loc1)
      %cA = "arith.constant"() <{value = SFA : i32}> : () -> i32 loc(#loc1)
      %cB = "arith.constant"() <{value = SFB : i32}> : () -> i32 loc(#loc1)
      %id = "arith.constant"() <{value = IDESC : i32}> : () -> i32 loc(#loc1)
      %acc = "arith.constant"() <{value = true}> : () -> i1 loc(#loc1)
      %hold = "cute_nvgpu.arch.get_dyn_smem"() : () -> !cute.ptr<i32, smem, align<32>> loc(#loc1)
      %a = "cute.assume"(%c0) : (i32) -> !cute.i32<divby 16> loc(#loc1)
      %d = "cute.inttoptr"(%a) : (!cute.i32<divby 16>) -> !cute.ptr<f32, tmem, align<16>> loc(#loc1)
      %tA = "cute.make_int_tuple"(%cA) : (i32) -> !cute.int_tuple<"?"> loc(#loc3)
      %tB = "cute.make_int_tuple"(%cB) : (i32) -> !cute.int_tuple<"?"> loc(#loc3)
      %pA = "cute.add_offset"(%d, %tA) : (!cute.ptr<f32, tmem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<f32, tmem, align<16>> loc(#loc3)
      %pB = "cute.add_offset"(%d, %tB) : (!cute.ptr<f32, tmem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<f32, tmem, align<16>> loc(#loc3)
      %sA = "cute.recast_iter"(%pA) : (!cute.ptr<f32, tmem, align<16>>) -> !cute.ptr<f8E8M0FNU, tmem, align<16>> loc(#loc3)
      %sB = "cute.recast_iter"(%pB) : (!cute.ptr<f32, tmem, align<16>>) -> !cute.ptr<f8E8M0FNU, tmem, align<16>> loc(#loc3)
      %m0 = "cute.make_tiled_mma"() : () -> !mma_t loc(#loc3)
      %m1 = "cute_nvgpu.atom.set_value"(%m0, %sA) <{field = #cute_nvgpu.atom_mma_field_sm100_block_scaled<sf_a>}> : (!mma_t, !cute.ptr<f8E8M0FNU, tmem, align<16>>) -> !mma_t loc(#loc3)
      %m2 = "cute_nvgpu.atom.set_value"(%m1, %sB) <{field = #cute_nvgpu.atom_mma_field_sm100_block_scaled<sf_b>}> : (!mma_t, !cute.ptr<f8E8M0FNU, tmem, align<16>>) -> !mma_t loc(#loc3)
      %m3 = "cute_nvgpu.atom.set_value"(%m2, %acc) <{field = #cute_nvgpu.atom_mma_field_sm100_block_scaled<accum_c>}> : (!mma_t, i1) -> !mma_t loc(#loc3)
      %gA = "cute_nvgpu.atom.get_value"(%m3) <{field = #cute_nvgpu.atom_mma_field_sm100_block_scaled<sf_a>}> : (!mma_t) -> !cute.ptr<f8E8M0FNU, tmem, align<16>> loc(#loc3)
      %gB = "cute_nvgpu.atom.get_value"(%m3) <{field = #cute_nvgpu.atom_mma_field_sm100_block_scaled<sf_b>}> : (!mma_t) -> !cute.ptr<f8E8M0FNU, tmem, align<16>> loc(#loc3)
      %gC = "cute_nvgpu.atom.get_value"(%m3) <{field = #cute_nvgpu.atom_mma_field_sm100_block_scaled<accum_c>}> : (!mma_t) -> i1 loc(#loc3)
      %desc = "cute_nvgpu.make_umma_smem_desc"(%hold) <{layout = #cute.layout<"((32,16)):((16,1))">, major = #cute_nvgpu.major<k>}> : (!cute.ptr<i32, smem, align<32>>) -> !cute_nvgpu.smem_desc loc(#loc3)
      "cute_nvgpu.arch.mma.SM100.umma_block_scaled"(%desc, %desc, %d, %gA, %gB, %id, %gC) <{a_type = f8E4M3FN, num_cta = 1 : i32, shape_k = 32 : i32, vec_size = 32 : i32}> : (!cute_nvgpu.smem_desc, !cute_nvgpu.smem_desc, !cute.ptr<f32, tmem, align<16>>, !cute.ptr<f8E8M0FNU, tmem, align<16>>, !cute.ptr<f8E8M0FNU, tmem, align<16>>, i32, i1) -> () loc(#loc3)
"""


def _p1_mma_stage(
    n: int, sfa: int = 256, sfb: int = 264, m: int = 128, ncta: int = 1
) -> str:
    alias = (
        f"!mma_t = !cute.tiled_mma<!cute_nvgpu.sm100.mma_bs<{m}x{n}x32, num_cta = {ncta}, ab_major = (k, k), "
        "elem_type = (f8E4M3FN, f8E4M3FN, f32), sf_type = f8E8M0FNU, frag_kind = ss, vec_size = 32>, "
        'atom_layout_MNK = <"(1,1,1):(0,0,0)">>'
    )
    idesc = ((n >> 3) << 17) | ((m >> 7) << 27)  # N >> 3 at bits 17-22, M >> 7 at 27-28
    body = (
        _P1_MMA_STAGE.replace("SFA", str(sfa))
        .replace("SFB", str(sfb))
        .replace("IDESC", str(idesc))
        .replace("num_cta = 1 : i32", f"num_cta = {ncta} : i32")
    )
    return alias + _kernel_stage(body, nargs=0)


# P1 pass: a plain (dense) umma, D at column 0, the shape only in its instruction descriptor.
_P1_UMMA_STAGE = """      %c0 = "arith.constant"() <{value = 0 : i32}> : () -> i32 loc(#loc1)
      %id = "arith.constant"() <{value = IDESC : i32}> : () -> i32 loc(#loc1)
      %acc = "arith.constant"() <{value = true}> : () -> i1 loc(#loc1)
      %hold = "cute_nvgpu.arch.get_dyn_smem"() : () -> !cute.ptr<i32, smem, align<32>> loc(#loc1)
      %a = "cute.assume"(%c0) : (i32) -> !cute.i32<divby 16> loc(#loc1)
      %d = "cute.inttoptr"(%a) : (!cute.i32<divby 16>) -> !cute.ptr<f32, tmem, align<16>> loc(#loc1)
      %desc = "cute_nvgpu.make_umma_smem_desc"(%hold) <{layout = #cute.layout<"((32,16)):((16,1))">, major = #cute_nvgpu.major<k>}> : (!cute.ptr<i32, smem, align<32>>) -> !cute_nvgpu.smem_desc loc(#loc3)
      "cute_nvgpu.arch.mma.SM100.umma"(%desc, %desc, %d, %id, %acc) <{a_type = bf16, num_cta = NCTA : i32, scale = 0 : i32}> : (!cute_nvgpu.smem_desc, !cute_nvgpu.smem_desc, !cute.ptr<f32, tmem, align<16>>, i32, i1) -> () loc(#loc3)
"""


def _p1_umma_stage(m: int, n: int, ncta: int = 1) -> str:
    # f32 D (bits 4-5 = 1), N >> 3 at bits 17-22, M >> 4 at bits 24-28
    idesc = (1 << 4) | ((n >> 3) << 17) | ((m >> 4) << 24)
    body = _P1_UMMA_STAGE.replace("IDESC", str(idesc)).replace("NCTA", str(ncta))
    return _kernel_stage(body, nargs=0)


# P1 pass, inline PTX: thread 0 copies scale factors into TMEM column 288 (an absolute
# address: cute.ptrtoint of the fake-zero base, + 288); the warp issues two kind::f16 MMAs
# (D at column 256, N from the function-scope register fa_idesc that another inline PTX
# sets) with a blocking wait loop between them, the first one with a TMEM A at 288 + 8.
_P1_ASM_STAGE = """      %0 = "nvvm.read.ptx.sreg.tid.x"() : () -> i32 loc(#loc1)
      %c0 = "arith.constant"() <{value = 0 : i32}> : () -> i32 loc(#loc1)
      %c256 = "arith.constant"() <{value = 256 : i32}> : () -> i32 loc(#loc1)
      %c288 = "arith.constant"() <{value = 288 : i32}> : () -> i32 loc(#loc1)
      %hold = "cute_nvgpu.arch.get_dyn_smem"() : () -> !cute.ptr<i32, smem, align<32>> loc(#loc1)
      %a = "cute.assume"(%c0) : (i32) -> !cute.i32<divby 16> loc(#loc1)
      %base = "cute.inttoptr"(%a) : (!cute.i32<divby 16>) -> !cute.ptr<f32, tmem, align<16>> loc(#loc1)
      %ta0 = "cute.ptrtoint"(%base) : (!cute.ptr<f32, tmem, align<16>>) -> i32 loc(#loc1)
      %ta = "arith.addi"(%ta0, %c288) : (i32, i32) -> i32 loc(#loc1)
      %sa = "cute.ptrtoint"(%hold) : (!cute.ptr<i32, smem, align<32>>) -> i32 loc(#loc1)
      %is0 = "arith.cmpi"(%0, %c0) <{predicate = PRED : i64}> : (i32, i32) -> i1 loc(#loc2)
      "scf.if"(%is0) ({
CP
        "scf.yield"() : () -> () loc(#loc3)
      }, {
      }) : (i1) -> () loc(#loc2)
IDESC
MMA
"""
_CP_ASM = "{\\0A.reg .b64 d;\\0Amov.b64 d, {$1, $2};\\0Atcgen05.cp.cta_group::1.32x128b.warpx4 [$0], d;\\0A}\\0A"
_IDESC_ASM = (
    ".reg .b32 fa_idesc;\\0Amov.b32 fa_idesc, 0x08100010;\\0A"  # M 128, N 64, f32 D
)
_MMA_ASM = (
    "{\\0A.reg .pred leader_thread;\\0A.reg .pred p;\\0A.reg .b32 tmem_acc;\\0A.reg .b32 tmem_a;\\0A"
    ".reg .b64 smem_desc_b;\\0Aelect.sync _|leader_thread, -1;\\0Amov.b32 tmem_acc, $0;\\0Amov.b32 tmem_a, $1;\\0A"
    "mov.b64 smem_desc_b, {$2, $2};\\0A"
    "GUARD tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a + 0x8], smem_desc_b, fa_idesc, 1;\\0A"
    "LAB_WAIT:\\0Ambarrier.try_wait.parity.shared::cta.b64 p, [$2], $3;\\0A@!p bra.uni LAB_WAIT;\\0A"
    "GUARD tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], smem_desc_b, smem_desc_b, fa_idesc, 1;\\0A}\\0A"
)


def _p1_asm_stage(pred: int = 0, guard: str = "@leader_thread") -> str:
    """_P1_ASM_STAGE with the cp under tid PRED 0 (0: eq, 1: ne) and the MMAs under
    `guard`."""
    body = (
        _P1_ASM_STAGE.replace("PRED", str(pred))
        .replace("CP", _asm(_CP_ASM, ["%ta", "%c0", "%c0"], ["i32"] * 3, "r,r,r").rstrip())
        .replace("IDESC", _asm(_IDESC_ASM, [], [], "").rstrip())
        .replace("MMA", _asm(_MMA_ASM.replace("GUARD", guard), ["%c256", "%ta", "%sa", "%c0"], ["i32"] * 4,
                             "r,r,r,r").rstrip())
    )  # fmt: skip
    return _kernel_stage(body, nargs=0)


# P1 pass, mbarrier issue sets: a role of two warps loads TMEM lanes [32 (w mod 4), +32) x
# cols [0, 32) and waits for the loads; WAIT / ARRIVE (an mbarrier try_wait before the load,
# an arrive after the wait::ld) are issued where the lane condition %sel (SEL) holds.
_P1_SYNC_STAGE = """      %0 = "nvvm.read.ptx.sreg.tid.x"() : () -> i32 loc(#loc1)
      %c0 = "arith.constant"() <{value = 0 : i32}> : () -> i32 loc(#loc1)
      %c1 = "arith.constant"() <{value = 1 : i32}> : () -> i32 loc(#loc1)
      %c4 = "arith.constant"() <{value = 4 : i32}> : () -> i32 loc(#loc1)
      %c21 = "arith.constant"() <{value = 21 : i32}> : () -> i32 loc(#loc1)
      %c32 = "arith.constant"() <{value = 32 : i32}> : () -> i32 loc(#loc1)
      %cn = "arith.constant"() <{value = 512 : i32}> : () -> i32 loc(#loc1)
      %hold = "cute_nvgpu.arch.get_dyn_smem"() : () -> !cute.ptr<i32, smem, align<32>> loc(#loc1)
      "cute_nvgpu.arch.sm100.alloc_tmem"(%cn, %hold) : (i32, !cute.ptr<i32, smem, align<32>>) -> () loc(#loc1)
      %a = "cute.assume"(%c0) : (i32) -> !cute.i32<divby 16> loc(#loc1)
      %base = "cute.inttoptr"(%a) : (!cute.i32<divby 16>) -> !cute.ptr<f32, tmem, align<16>> loc(#loc1)
      %w = "arith.divui"(%0, %c32) : (i32, i32) -> i32 loc(#loc1)
      %q = "arith.remui"(%w, %c4) : (i32, i32) -> i32 loc(#loc1)
      %l = "arith.shli"(%q, %c21) : (i32, i32) -> i32 loc(#loc1)
      %t = "cute.make_int_tuple"(%l) : (i32) -> !cute.int_tuple<"?"> loc(#loc3)
      %p = "cute.add_offset"(%base, %t) : (!cute.ptr<f32, tmem, align<16>>, !cute.int_tuple<"?">) -> !cute.ptr<f32, tmem, align<16>> loc(#loc3)
      %bar = "cute.recast_iter"(%hold) : (!cute.ptr<i32, smem, align<32>>) -> !cute.ptr<i64, smem, align<32>> loc(#loc3)
      %bp = "builtin.unrealized_conversion_cast"(%bar) : (!cute.ptr<i64, smem, align<32>>) -> !llvm.ptr<3> loc(#loc3)
      %isz = "arith.cmpi"(%0, %c0) <{predicate = 0 : i64}> : (i32, i32) -> i1 loc(#loc2)
      "scf.if"(%isz) ({
        "nvvm.mbarrier.init.shared"(%bp, %c1) : (!llvm.ptr<3>, i32) -> () loc(#loc3)
        "scf.yield"() : () -> () loc(#loc3)
      }, {
      }) : (i1) -> () loc(#loc2)
SEL
      "scf.if"(%sel) ({
        WAIT
        "scf.yield"() : () -> () loc(#loc3)
      }, {
      }) : (i1) -> () loc(#loc2)
      %v = "cute_nvgpu.arch.copy.SM100.tmem_load"(%p) <{num_b = 32 : i32, num_dp = 32 : i32, num_rep = 32 : i32}> : (!cute.ptr<f32, tmem, align<16>>) -> vector<32xi32> loc(#loc3)
      "nvvm.tcgen05.wait"() <{kind = #nvvm.tcgen05_wait<load>}> : () -> () loc(#loc3)
      "scf.if"(%sel) ({
        ARRIVE
        "scf.yield"() : () -> () loc(#loc3)
      }, {
      }) : (i1) -> () loc(#loc2)
"""
_SEL_TID0 = '      %sel = "arith.cmpi"(%0, %c0) <{predicate = 0 : i64}> : (i32, i32) -> i1 loc(#loc2)'
_SEL_LANE0 = """      %ln = "arith.remui"(%0, %c32) : (i32, i32) -> i32 loc(#loc2)
      %sel = "arith.cmpi"(%ln, %c0) <{predicate = 0 : i64}> : (i32, i32) -> i1 loc(#loc2)"""
# thread t issues iff t < x[0] (a runtime value: metadata)
_SEL_RUNTIME = """      %k0 = "cute.make_coord"(%c0) : (i32) -> !cute.coord<"?"> loc(#loc1)
      %n = "cute.memref.load"(%arg0, %k0) : (!cute.memref<i32, gmem, "(128):(1)">, !cute.coord<"?">) -> i32 loc(#loc1)
      %sel = "arith.cmpi"(%0, %n) <{predicate = 6 : i64}> : (i32, i32) -> i1 loc(#loc2)"""
_WAIT = '"nvvm.mbarrier.try_wait.parity.shared"(%bp, %c0) : (!llvm.ptr<3>, i32) -> () loc(#loc3)'
_ARRIVE = '%ar = "nvvm.mbarrier.arrive.shared"(%bp) : (!llvm.ptr<3>) -> i64 loc(#loc3)'


def _p1_sync_stage(sel: str, wait: bool = False, arrive: bool = False) -> str:
    body = (
        _P1_SYNC_STAGE.replace("SEL", sel)
        .replace("WAIT", _WAIT if wait else "")
        .replace("ARRIVE", _ARRIVE if arrive else "")
    )
    return _kernel_stage(body, nargs=1, elem="i32")


class P1Test(unittest.TestCase):
    """The P1 translation pass (ops_tmem.py, references/ir_ops_cute.md "P1 slice")."""

    def mma_rects(self, n: int) -> list:
        _k, roles = translate(_p1_mma_stage(n), nwarps=1, p1=True)
        (mma,) = [e for e in _events(roles[0].stmts) if e["ev"] == "tmem"]
        self.assertEqual(mma["inst"], "mma")
        return [
            (r["use"], r["write"], r["nlanes"], r["col"]["v"], r["ncols"])
            for r in mma["rects"]
        ]

    def test_block_scaled_mma_scale_factor_footprint(self):
        # CuTe DSL 4.4.2 tmem SF layouts: SFA 4 columns (M = 128); SFB 4 columns per 128 rows
        # of N (N = 192 or 256: 8 columns, not N / 32).
        for n, sfb_cols in ((128, 4), (64, 4), (192, 8), (256, 8)):
            self.assertEqual(
                self.mma_rects(n),
                [
                    ("d", True, 128, 0, n),
                    ("sf", False, 128, 256, 4),
                    ("sf", False, 128, 264, sfb_cols),
                ],
                n,
            )

    def test_block_scaled_mma_with_n_over_256_is_refused(self):
        with self.assertRaisesRegex(
            Unsupported, "R5: block-scaled tcgen05.mma 128x320x32"
        ):
            self.mma_rects(320)

    def test_mma_atom_carried_by_a_loop_is_joined(self):
        # The loop sets accum_c on the carried atom, a field the initial atom lacks: the
        # atoms are joined (accum_c becomes data) before the loop-carried structure check.
        text = (
            _p1_mma_stage(128)
            .replace(
                '      %m3 = "cute_nvgpu.atom.set_value"(%m2, %acc)',
                """      %c1 = "arith.constant"() <{value = 1 : i32}> : () -> i32 loc(#loc1)
      %c4 = "arith.constant"() <{value = 4 : i32}> : () -> i32 loc(#loc1)
      %mr = "scf.for"(%c0, %c4, %c1, %m2) ({
      ^bb0(%i: i32, %mm: !mma_t):
      %m3 = "cute_nvgpu.atom.set_value"(%mm, %acc)""",
            )
            .replace(
                '      "cuda.return"()',
                """      "scf.yield"(%m3) : (!mma_t) -> () loc(#loc3)
      }) : (i32, i32, i32, !mma_t) -> !mma_t loc(#loc3)
      "cuda.return"()""",
            )
        )
        _k, roles = translate(text, nwarps=1, p1=True)
        loop = next(s for s in roles[0].stmts if s["op"] == "for")
        (mma,) = [e for e in _events(loop["body"]) if e["ev"] == "tmem"]
        self.assertEqual(mma["inst"], "mma")

    def p1(self, **kw):
        text, nw = _p1_stage(**kw)
        return translate(text, nwarps=nw, p1=True)

    def test_tmem_events_rectangles_and_base(self):
        k, roles = self.p1()
        self.assertEqual([(r.w0, r.nw) for r in roles], [(0, 1), (1, 1)])
        ld, wait = [
            e for e in _events(roles[0].stmts) if e["ev"] in ("tmem", "tmem_wait")
        ]
        self.assertEqual((ld["inst"], ld["warp"]), ("ld", E.to_json(E.const(0, 32))))
        r = ld["rects"][0]
        self.assertEqual(
            (r["use"], r["write"], r["nlanes"], r["ncols"]), ("data", False, 32, 32)
        )
        self.assertEqual(
            (r["lane"], r["col"]),
            (E.to_json(E.const(0, 32)), E.to_json(E.const(64, 32))),
        )
        self.assertEqual(
            wait, {"op": "event", "ev": "tmem_wait", "ld": True, "loc": "k.py:12"}
        )
        (cp,) = [e for e in _events(roles[1].stmts) if e["ev"] == "tmem"]
        r = cp["rects"][0]
        self.assertEqual(
            (cp["inst"], r["write"], r["nlanes"], r["ncols"], r["col"]),
            ("cp", True, 128, 4, E.to_json(E.const(80, 32))),
        )
        self.assertEqual(ops_tmem.tmem_top(roles), {"cols": 512, "base": "zero"})
        kir = as_kir(k, roles)
        kir["tmem"] = ops_tmem.tmem_top(roles)
        self.assertEqual(to_kir.validate(kir), [])

    def test_p0_pass_has_no_tmem_events(self):
        text, nw = _p1_stage()
        _k, roles = translate(text, nwarps=nw)
        self.assertEqual([e for r in roles for e in _events(r.stmts)], [])

    def test_absolute_pointers_need_a_512_column_allocation(self):
        _k, roles = self.p1(alloc=256)
        with self.assertRaisesRegex(
            Unsupported, "R3: absolute TMEM addresses with a 256-column allocation"
        ):
            ops_tmem.tmem_top(roles)

    def test_inttoptr_of_a_nonzero_value_is_refused(self):
        with self.assertRaisesRegex(Unsupported, "R1: tcgen05"):
            self.p1(base=16)

    def test_lane_varying_ld_address_is_refused(self):
        with self.assertRaisesRegex(
            Unsupported, "R6: tcgen05.ld: an operand varies across the lanes"
        ):
            self.p1(ld="%0")

    def test_cp_in_a_two_warp_role_is_refused(self):
        text, _ = _p1_stage()
        text = text.replace(
            '"arith.cmpi"(%w, %c1) <{predicate = 0 : i64}>',
            '"arith.cmpi"(%w, %c1) <{predicate = 5 : i64}>',
        )
        with self.assertRaisesRegex(Unsupported, "R6: tcgen05.cp in a role of 2 warps"):
            translate(text, nwarps=3, p1=True)

    def test_named_barriers(self):
        bar = _asm("bar.sync $0, $1;", ["%c1", "%c32"], ["i32", "i32"], "r,r")
        _k, roles = self.p1(
            extra=bar.strip()
            + '\n        "nvvm.barrier"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> () loc(#loc3)'
        )
        nb = [e for e in _events(roles[0].stmts) if e["ev"] == "named_bar"]
        self.assertEqual(
            [(e["id"]["v"], e["count"]["v"], e["sync"]) for e in nb],
            [(1, 32, True), (0, 64, True)],
        )

    def test_runtime_named_barrier_id_is_refused(self):
        bar = _asm("bar.sync $0, $1;", ["%0", "%c32"], ["i32", "i32"], "r,r")
        with self.assertRaisesRegex(
            Unsupported, "R7: named barrier id is not a constant"
        ):
            self.p1(extra=bar.strip())

    def test_explicit_stream_launch_argument_is_dropped(self):
        # A launch may pass a cuda.CUstream; the host function's stream argument takes
        # no launch value (to_kir._bind_host_args), so it must not count (W0).
        args = [
            {"kind": "scalar", "type": "int", "value": 3},
            {"kind": "opaque", "type": "CUstream"},
            {
                "kind": "tuple",
                "items": [
                    {"kind": "none"},
                    {"kind": "scalar", "type": "int", "value": 4},
                ],
            },
        ]
        self.assertEqual([a.get("value") for a in to_kir._flatten(args)], [3, 4])

    def sync_roles(self, sel, **kw):
        """(w0, nw, [(event, issuers)]) of every P1 role: its mbarrier events and loads."""
        _k, roles = translate(_p1_sync_stage(sel, **kw), nwarps=2, p1=True)
        out = []
        for r in roles:
            issuers = {id(u.event): u.issuers for u in r.barriers}
            evs = [
                (
                    e["ev"] + (" " + e["inst"] if e["ev"] == "tmem" else ""),
                    issuers.get(id(e)),
                )
                for e in _events(r.stmts)
                if e["ev"] in ("wait", "arrive", "tmem")
            ]
            out.append((r.w0, r.nw, evs))
        return out

    def test_thread0_arrive_splits_the_role_in_p1_only(self):
        # Warp 1's load is released by no arrive: as one 2-warp role the arrive would
        # stand for both warps (review of 6acbcf1f3a04).
        self.assertEqual(
            self.sync_roles(_SEL_TID0, arrive=True),
            [(0, 1, [("tmem ld", None), ("arrive", 1)]), (1, 1, [("tmem ld", None)])],
        )
        _k, roles = translate(_p1_sync_stage(_SEL_TID0, arrive=True), nwarps=2)
        self.assertEqual([(r.w0, r.nw) for r in roles], [(0, 2)])  # P0 roles unchanged

    def test_thread0_wait_splits_the_role(self):
        self.assertEqual(
            self.sync_roles(_SEL_TID0, wait=True),
            [(0, 1, [("wait", None), ("tmem ld", None)]), (1, 1, [("tmem ld", None)])],
        )

    def test_per_warp_arrive_keeps_the_role(self):
        # lane 0 of every warp (the elect-one-per-warp pattern): one event of both warps
        (r,) = self.sync_roles(_SEL_LANE0, wait=True, arrive=True)
        self.assertEqual(r[:2], (0, 2))
        self.assertEqual(
            [e for e in r[2] if e[0] != "tmem ld"], [("wait", None), ("arrive", 2)]
        )

    def test_arrive_under_a_runtime_lane_condition_is_refused(self):
        with self.assertRaisesRegex(
            Unsupported,
            "R7: mbarrier arrive under a lane condition that depends on runtime",
        ):
            self.sync_roles(_SEL_RUNTIME, arrive=True)

    def test_inline_tcgen05_access_is_refused_in_p1_only(self):
        asm = _asm(
            "tcgen05.ld.sync.aligned.32x32b.x1.b32 {$0}, [$1];", ["%c1"], ["i32"], "r"
        )
        text, nw = _p1_stage(extra=asm.strip())
        translate(text, nwarps=nw)  # P0: data
        with self.assertRaisesRegex(Unsupported, "R4: inline PTX with a tcgen05 data"):
            translate(text, nwarps=nw, p1=True)

    def test_plain_umma_shape_comes_from_the_instruction_descriptor(self):
        _k, (role,) = translate(_p1_umma_stage(128, 64), nwarps=1, p1=True)
        (mma,) = [e for e in _events(role.stmts) if e["ev"] == "tmem"]
        self.assertEqual(
            [(r["use"], r["write"], r["nlanes"], r["col"]["v"], r["ncols"]) for r in mma["rects"]],
            [("d", True, 128, 0, 64)],
        )  # fmt: skip
        self.assertNotIn("cta_mask", mma)

    def test_inline_tcgen05_cp_and_mma(self):
        _k, (role,) = translate(_p1_asm_stage(), nwarps=1, p1=True)
        evs = [e for e in _events(role.stmts) if e["ev"] in ("tmem", "wait")]
        self.assertEqual(
            [(e["ev"], e.get("inst")) for e in evs],
            [("tmem", "cp"), ("tmem", "mma"), ("wait", None), ("tmem", "mma")],
        )
        cp, mma1, _w, mma2 = evs

        def rects(e):
            return [
                (r["use"], r["write"], r["nlanes"], r["col"], r["ncols"])
                for r in e["rects"]
            ]

        def col(c):
            return E.to_json(E.const(c, 32))

        self.assertEqual(rects(cp), [("data", True, 128, col(288), 4)])
        self.assertEqual(
            rects(mma1),
            [("d", True, 128, col(256), 64), ("a", False, 128, col(296), 8)],
        )
        self.assertEqual(rects(mma2), [("d", True, 128, col(256), 64)])
        self.assertEqual(role.tmem_bases, {"abs"})

    def test_inline_tcgen05_cp_by_every_thread_is_refused(self):
        # no elect.sync in the cp's PTX: each of the 31 threads with tid != 0 would issue one
        with self.assertRaisesRegex(Unsupported, "R6: tcgen05.cp issued by 31 threads"):
            translate(_p1_asm_stage(pred=1), nwarps=1, p1=True)

    def test_inline_tcgen05_mma_under_another_predicate_is_refused(self):
        with self.assertRaisesRegex(Unsupported, "R6: inline PTX 'tcgen05.mma"):
            translate(_p1_asm_stage(guard="@p"), nwarps=1, p1=True)

    def test_named_barrier_id_per_warp_splits_the_role(self):
        ids = """      %c1i = "arith.constant"() <{value = 1 : i32}> : () -> i32 loc(#loc1)
      %c64 = "arith.constant"() <{value = 64 : i32}> : () -> i32 loc(#loc1)
      %bid = "arith.addi"(%w, %c1i) : (i32, i32) -> i32 loc(#loc1)
"""
        bar = _asm("bar.sync $0, $1;", ["%bid", "%c64"], ["i32", "i32"], "r,r")
        text, _nw = _p1_stage(extra=ids + bar.strip())
        # both warps take the load branch (w == 32 never holds) and sync on barrier 1 + w
        text = text.replace('"arith.cmpi"(%w, %c1)', '"arith.cmpi"(%w, %c32)')
        _k, roles = translate(text, nwarps=2, p1=True)
        self.assertEqual([(r.w0, r.nw) for r in roles], [(0, 1), (1, 1)])
        self.assertEqual(
            [[e["id"]["v"] for e in _events(r.stmts) if e["ev"] == "named_bar"] for r in roles],
            [[1], [2]],
        )  # fmt: skip


# A cluster of 2 (thread-block clusters, kir.md change 11): barrier init, the cluster
# barrier, a remote arrive on the peer's barrier (mapa), a cta_group::2 commit to both CTAs,
# an inline PTX blocking wait loop and a shared::cluster bulk copy into the peer.
_SMEM = "!cute.ptr<i64, smem, align<1024>>"
_CLUSTER_STAGE = f"""      %c0 = "arith.constant"() <{{value = 0 : i32}}> : () -> i32 loc(#loc1)
      %c1 = "arith.constant"() <{{value = 1 : i32}}> : () -> i32 loc(#loc1)
      %c3 = "arith.constant"() <{{value = 3 : i16}}> : () -> i16 loc(#loc1)
      %n = "arith.constant"() <{{value = 512 : i32}}> : () -> i32 loc(#loc1)
      %s = "cute_nvgpu.arch.get_dyn_smem"() : () -> {_SMEM} loc(#loc1)
      %b = "builtin.unrealized_conversion_cast"(%s) : ({_SMEM}) -> !llvm.ptr<3> loc(#loc1)
      "nvvm.mbarrier.init.shared"(%b, %c1) : (!llvm.ptr<3>, i32) -> () loc(#loc3)
      "nvvm.cluster.arrive.relaxed"() : () -> () loc(#loc3)
      "nvvm.cluster.wait"() : () -> () loc(#loc3)
      %r = "nvvm.read.ptx.sreg.cluster.ctarank"() : () -> i32 loc(#loc1)
      %p = "arith.xori"(%r, %c1) : (i32, i32) -> i32 loc(#loc1)
      %m = "nvvm.mapa.shared.cluster"(%b, %p) : (!llvm.ptr<3>, i32) -> !llvm.ptr<3> loc(#loc1)
      "nvvm.mbarrier.txn"(%m, %c1) <{{kind = #nvvm.mbar_txn_kind<arrive>, scope = #nvvm.mbar_scope<cta>, space = #nvvm.mbar_space<cluster>}}> : (!llvm.ptr<3>, i32) -> () loc(#loc3)
      "nvvm.tcgen05.commit.arrive"(%b, %c3) <{{group = #nvvm.tcgen05_group<cta_2>}}> : (!llvm.ptr<3>, i16) -> () loc(#loc3)
      %a = "cute.ptrtoint"(%s) : ({_SMEM}) -> i32 loc(#loc1)
"""
_WAIT_LOOP = "{\\0A.reg .pred p;\\0ALAB_WAIT:\\0Ambarrier.try_wait.parity.shared::cta.b64 p, [$0], $1;\\0A@!p bra.uni LAB_WAIT;\\0A}\\0A"
_S2CLUSTER = ("{\\0A.reg .b32 rd, rb;\\0Amapa.shared::cluster.u32 rd, $0, $4;\\0Amapa.shared::cluster.u32 rb, $2, $4;"
              "\\0Acp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [rd], [$1], $3, [rb];\\0A}\\0A")  # fmt: skip


class ClusterTest(unittest.TestCase):
    """Thread-block clusters in the P0 pass (references/frontend_cute.md, "Clusters")."""

    def test_cluster_events(self):
        body = (
            _CLUSTER_STAGE
            + _asm(_WAIT_LOOP, ["%a", "%c0"], ["i32", "i32"], "r,r")
            + _asm(_S2CLUSTER, ["%a", "%a", "%a", "%n", "%p"], ["i32"] * 5, "r,r,r,r,r")
        )
        k, (role,) = translate(
            _kernel_stage(body, nargs=0), nwarps=1, cluster=(2, 1, 1)
        )
        (let,) = [s for s in role.stmts if s["op"] == "let" and s["expr"].get("o") == "xor"
                  and s["var"] != "cluster_barrier_parity"]  # fmt: skip
        self.assertEqual(  # the peer's rank: %cluster_ctarank = pid.x % 2, xor 1
            let["expr"],
            E.to_json(
                E.bin("xor", E.bin("urem", E.pid(0), E.const(2, 32)), E.const(1, 32))
            ),
        )
        peer = {"e": "var", "w": 32, "name": let["var"]}
        evs = [
            (e["ev"], e.get("bar"), e.get("cta"), e.get("cta_mask"))
            for e in _events(role.stmts)
        ]
        self.assertEqual(
            evs,
            [
                ("bar_init", None, None, None),
                ("arrive", "cluster_barrier", E.to_json(E.const(0, 32)), None),
                ("wait", "cluster_barrier", E.to_json(E.const(0, 32)), None),
                ("arrive", None, peer, None),
                ("commit_arrive", None, None, E.to_json(E.const(3, 32))),
                ("wait", None, None, None),
                ("complete_tx", None, peer, None),
            ],
        )
        self.assertTrue(role.uses_cluster_wait)
        (cx,) = [e for e in _events(role.stmts) if e["ev"] == "complete_tx"]
        self.assertEqual(cx["bytes"], E.to_json(E.const(512, 32)))

    def test_cluster_rank_without_clusters_is_zero(self):
        _k, (role,) = translate(_kernel_stage(_CLUSTER_STAGE, nargs=0), nwarps=1)
        (arrive,) = [
            e for e in _events(role.stmts) if e["ev"] == "arrive" and e["bar"] is None
        ]
        self.assertEqual(arrive["cta"], E.to_json(E.const(1, 32)))  # rank 0, xor 1

    def test_two_cta_mma_accesses_both_ctas_of_the_pair(self):
        # P1: M = 256 is 128 lanes x N in each CTA; M = 128 is 128 lanes x N / 2 (CuTe's 2x2
        # layout); the CTA mask is the issuing CTA's pair, 3 << (rank & ~1).
        for m, ncols in ((256, 128), (128, 64)):
            _k, (role,) = translate(
                _p1_mma_stage(128, m=m, ncta=2), nwarps=1, p1=True, cluster=(2, 1, 1)
            )
            (mma,) = [e for e in _events(role.stmts) if e["ev"] == "tmem"]
            self.assertEqual(mma["rects"][0]["ncols"], ncols, m)
            rank = E.bin("urem", E.pid(0), E.const(2, 32))
            mask = E.bin("shl", E.const(3, 32), E.bin("and", rank, E.const(-2, 32)))
            self.assertEqual(mma["cta_mask"], E.to_json(mask))  # fmt: skip

    def test_cta_group_2_without_a_cluster_is_refused(self):
        with self.assertRaisesRegex(Unsupported, "R5: cta_group::2 without a cluster"):
            translate(_p1_mma_stage(128, m=256, ncta=2), nwarps=1, p1=True)

    def test_inline_try_wait_that_does_not_loop_is_rejected(self):
        asm = "{\\0A.reg .pred p;\\0Ambarrier.try_wait.parity.shared::cta.b64 p, [$0], $1;\\0A}\\0A"
        body = _CLUSTER_STAGE + _asm(asm, ["%a", "%c0"], ["i32", "i32"], "r,r")
        with self.assertRaisesRegex(Unsupported, "not a blocking wait loop"):
            translate(_kernel_stage(body, nargs=0), nwarps=1, cluster=(2, 1, 1))

    def test_multicast_descriptor_box_is_the_slice(self):
        # CuTe make_tma_copy_desc: the box shrinks by num_multicast from the outermost dim.
        for box, n, want in (
            ([512, 1, 1], 2, [256, 1, 1]),
            ([64, 4], 8, [32, 1]),
            ([64, 128], 2, [64, 64]),
        ):
            self.assertEqual(ops_mem._multicast_slice(None, None, box, n), want)

    def test_fast_division_idiom_is_udiv(self):
        k, (role,) = translate(_kernel_stage("", nargs=0), nwarps=1)
        n, d = E.var("n", 32), E.var("D", 32)
        k.fdm = [d]
        m, s1, s2 = E.var("%fdm0_m", 32), E.var("%fdm0_s1", 8), E.var("%fdm0_s2", 8)
        t1 = E.trunc(
            E.bin(
                "lshr",
                E.bin("mul", E.ext(n, 64, False), E.ext(m, 64, False)),
                E.const(32, 64),
            ),
            32,
        )
        q = E.bin("lshr", E.bin("add", t1, E.bin("lshr", E.bin("sub", n, t1), E.ext(s1, 32, False))),
                  E.ext(s2, 32, False))  # fmt: skip
        self.assertIs(ops.fast_division(role, q), E.bin("udiv", n, d))
        self.assertIsNone(
            ops.fast_division(role, E.bin("lshr", t1, E.ext(s2, 32, False)))
        )
        with self.assertRaisesRegex(E.ExprError, "reserved value"):
            E.to_json(t1)

    def test_barrier_runs_split_where_no_dynamic_index_crosses(self):
        # One init site, six slots: a use indexed by (x & 1) + 2 keeps slots 2..3 together.
        x = E.var("x", 32)
        dyn = E.bin(
            "mul",
            E.bin("add", E.bin("and", x, E.const(1, 32)), E.const(2, 32)),
            E.const(8, 32),
        )
        uses = [SimpleNamespace(addr=E.const(0, 64), base=E.const(0, 64)),
                SimpleNamespace(addr=E.bin("mul", E.var("v", 32), E.const(8, 32)), base=dyn)]  # fmt: skip
        self.assertEqual(
            to_kir._split_arrays([[0, 6, "c"]], uses),
            [[0, 1, "c"], [8, 1, "c"], [16, 2, "c"], [32, 1, "c"], [40, 1, "c"]],
        )
        # an index of unknown range keeps the run whole
        uses[1].base = E.bin("mul", E.var("v", 32), E.const(8, 32))
        self.assertEqual(to_kir._split_arrays([[0, 6, "c"]], uses), [[0, 6, "c"]])

    def test_store_through_a_register_pointer_is_not_an_event(self):
        # A float-guarded store to registers (an llvm.store through a cast of an alloca's
        # pointer) needs no branch condition: P0 drops it.
        rmem = '!cute.memref<f32, rmem, "(4):(1)">'
        cond = _FLOAT_COND.replace("%4 =", "%f =")
        body = (
            _TID
            + _STORE.split("COND")[0]
            + cond
            + f"""
      "scf.if"(%f) ({{
        %al = "cute.memref.alloca"() : () -> {rmem} loc(#loc3)
        %it = "cute.get_iter"(%al) : ({rmem}) -> !cute.ptr<f32, rmem> loc(#loc3)
        %lp = "builtin.unrealized_conversion_cast"(%it) : (!cute.ptr<f32, rmem>) -> !llvm.ptr loc(#loc3)
        "llvm.store"(%3, %lp) <{{ordering = 0 : i64}}> : (f32, !llvm.ptr) -> () loc(#loc3)
        "scf.yield"() : () -> () loc(#loc3)
      }}, {{
      }}) : (i1) -> () loc(#loc2)
"""
        )
        _k, roles = translate(_kernel_stage(body), nwarps=1)
        self.assertEqual(
            [e["ev"] for e in _events(roles[0].stmts)], ["gload"]
        )  # x[t] only


if __name__ == "__main__":
    unittest.main()
