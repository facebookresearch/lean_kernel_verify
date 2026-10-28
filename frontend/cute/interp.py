# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Translation of one role of a CuTe kernel (analyzed stage) into KIR statements.

A role is a contiguous warp range [w0, w0 + nw) executing the same code path. The kernel
body is translated once per role with WARP and LANE symbolic; a branch whose condition
is uniform within each warp but differs between warps of the range raises SplitWarps,
and the caller translates each part again (partition refinement).

Values (one per relevant SSA value, slicer.py):
  * integer expressions (expr.py). Thread-uniform ones are bound by `let` (one per IR
    op, with its loc); thread-varying ones stay symbolic over LANE/WARP/ELECT;
  * Data(origin): values the slice drops (floats, vectors, register/shared/tensor memory
    contents, MMA state). A kept use of Data is the slice soundness error;
  * CuTe structure: Tup (IntTuple), Lay, Composed, Ptr, View, ArithIter, Opaque (copy
    atoms, TMA descriptors, divisors).

Control flow. A thread-uniform `scf.if` is a KIR `if`; its results are variables declared
before it and assigned in both branches. A thread-varying condition is a mask: both
branches are translated in place, events carry the mask, results are `select`s, and a
division, remainder or metadata load under a mask has its divisor/index replaced where
the mask is false (the value is unused there), so no spurious verdict error arises.
Loops are KIR `for`/`while`: loop-carried values are variables assigned before the loop
and reassigned (in parallel) at the end of the body; `scf.while` emits its condition
region before the loop and again at the end of the body. A thread-varying value stored
in a variable becomes a family of variables, one per lane (if it does not depend on
WARP) or per thread of the role, read through a `table` expression.

Events. Memory events execute once per issuing thread: a single event when one thread
issues it, else a KIR `for` over the role's threads with the thread's mask as an `if`.
Events follow the op table (ir_ops_cute.md): waits once per role, arrivals counted,
expect_tx/commit once per issuing thread, TMA stores once per distinct operand tuple of
each warp's issuing lanes, TMA loads and reduces once per issuing warp. In the P1
pass a role of several warps whose warps do not all issue an mbarrier wait, arrive or
commit is split by issuing warps (Translator._p1_split_by_issuers).
"""

import itertools
import re
from dataclasses import dataclass, field
from typing import NoReturn

from . import expr as E, layout as L, mlir_text, slicer


class Unsupported(Exception):
    """A construct the frontend cannot translate exactly (a hard error). `kind`: "gap", a
    construct the frontend refuses by design (a known limit, not an error), or "error",
    something the case, the DSL install or LKV must fix (kir/report.json "kind")."""

    def __init__(self, op: str, loc: str, why: str, kind: str = "gap"):
        super().__init__(f"{op} at {loc}: {why}")
        self.op, self.loc, self.why, self.kind = op, loc, why, kind


class SplitWarps(Exception):
    def __init__(self, groups: list):
        super().__init__(f"split into warp groups {groups}")
        self.groups = groups  # [(w0, nw)]


@dataclass(frozen=True)
class Data:
    origin: str


@dataclass(frozen=True)
class Tup:
    t: object


@dataclass(frozen=True)
class Lay:
    shape: object
    stride: object


@dataclass(frozen=True)
class Composed:
    swizzle: tuple
    offset: int
    lay: Lay


@dataclass(frozen=True)
class Ptr:
    space: str  # gmem | smem | rmem | tmem
    base: str  # gmem: kernel tensor name; smem: "dyn"; rmem: alloca name
    off: tuple  # i64 expression, in elements of `bits`
    bits: int


@dataclass(frozen=True)
class Remote:
    """A shared::cluster address (mapa): shared-memory pointer `ptr` in the CTA of rank `cta`
    (an i32 expression) of the cluster."""

    ptr: Ptr
    cta: tuple


@dataclass(frozen=True)
class View:
    ptr: object  # Ptr | ArithIter
    lay: object  # Lay | Composed


@dataclass(frozen=True)
class ArithIter:
    coord: tuple


@dataclass(frozen=True)
class Opaque:
    kind: str
    info: tuple = ()  # sorted (key, value) pairs

    def get(self, key, default=None):
        return dict(self.info).get(key, default)

    def with_(self, **kw):
        return Opaque(self.kind, tuple(sorted({**dict(self.info), **kw}.items())))


def int_width(ty: str):
    ty = ty.strip()
    if ty == "index":
        return 64
    m = re.fullmatch(r"i(\d+)", ty) or re.fullmatch(r"!cute\.i(\d+)<.*>", ty)
    return int(m.group(1)) if m else None


@dataclass
class Kernel:
    """State shared by the roles of one kernel."""

    module: mlir_text.Module  # the stage
    src_root: str
    block: tuple = (1, 1, 1)
    cluster: tuple = (1, 1, 1)
    smem: tuple = E.const(0, 32)  # dynamic shared memory bytes (expression)
    max_iters: int = 4096
    work: dict = field(
        default_factory=dict
    )  # work field -> [{"load": tensor, "where": text}]
    tensors: dict = field(default_factory=dict)  # name -> {"bits", "elem_bytes", ...}
    descs: dict = field(default_factory=dict)  # name -> descriptor spec
    fdm: list = field(
        default_factory=list
    )  # fast divmod divisors whose aux values are read
    slice: slicer.Slice | None = None  # set before any role is translated
    _defs: dict = field(default_factory=dict, repr=False)  # SSA value -> its op (lazy)
    _parents: dict = field(default_factory=dict, repr=False)  # id(op) -> op holding it

    def loc_text(self, op) -> str:
        """op's full location: the statement text the DSL records and every frame."""
        return self.module.resolve_loc(op.loc)

    def def_of(self, value: str):
        """The op defining SSA value `value` (None for a block argument)."""
        self._index()
        return self._defs.get(value)

    def parent_of(self, op):
        """The op whose region holds `op` (None for the top op)."""
        self._index()
        return self._parents.get(id(op))

    def _index(self) -> None:
        if self._parents:
            return
        for o in mlir_text.walk(self.module.top):
            for r in o.results:
                self._defs[r] = o
            for region in o.regions:
                for block in region:
                    for c in block.ops:
                        self._parents[id(c)] = o

    def user_loc(self, op):
        """Innermost `file:line` of op's location under the source root, or None (ops the
        DSL builds inside its own helpers, ops a pass created)."""
        frames = self.user_frames(op)
        return frames[0] if frames else None

    def user_frames(self, op) -> tuple:
        """Every `file:line` of op's location under the source root, innermost first."""
        return tuple(
            f"{path[len(self.src_root) :]}:{line}"
            for path, line in re.findall(r'"([^"]+\.py)":(\d+)', self.loc_text(op))
            if path.startswith(self.src_root)
        )


# A role's barrier event awaiting address resolution: the KIR event dict (filled in
# later), the shared-memory byte address, the issuing thread count of an arrive (None
# when counted per thread inside a thread loop), the loc, and for a bar_init every
# address it initializes (one per iteration of enclosing constant-trip loops) and its
# site: its source frames and the op whose region holds it (the DSL's pipelines initialize
# each barrier array under a guard of its own, at one library-called source line).
# `fixed` names a barrier slot that has no shared-memory address (the cluster barrier);
# `base` is the address with let-bound index variables expanded (Translator.unbind).
@dataclass
class BarrierUse:
    event: dict
    addr: tuple
    issuers: object
    loc: str
    inits: tuple = ()
    fixed: str | None = None
    site: tuple = ()
    base: tuple | None = None


class Translator:
    def __init__(
        self,
        k: Kernel,
        spec,
        w0: int = 0,
        nw: int = 1,
        host: bool = False,
        p1: bool = False,
    ):
        self.k = k
        self.spec = (
            spec  # op name -> handler (ops.spec, or ops_tmem.spec_p1 in the P1 pass)
        )
        self.w0, self.nw = w0, nw
        self.host = host
        # P1 pass (P1-1, ops_tmem.py): atom fields are skeleton leaves (flatten), TMEM
        # pointers have a provenance, and TMEM ops, tcgen05.wait and named barriers are
        # events. The P0 pass (p1 False) is unchanged.
        self.p1 = p1
        self.tmem_allocs: list = []  # P1: (column count, loc) of each alloc_tmem
        self.tmem_abs: dict = {}  # P1: an absolute TMEM address (ptrtoint) -> its pointer
        self.tmem_bases: set = (
            set()
        )  # P1: pointer bases ("abs" / "alloc") of TMEM events
        self.loop_depth = 0
        self.loc_hist: list = []  # recent distinct source locations (P1 `site`)
        self.env: dict = {}
        self.stmts: list = []
        self.mask = E.TRUE
        self.nfresh = 0
        self.barriers: list = []  # BarrierUse
        self.bound: dict = {}  # variable of a `bind` let -> the expression it holds
        self.launches: list = []  # host side: cuda.launch_ex records
        self.cur_loc = "unknown"  # last source location seen (see loc)
        self.loop_values: dict = {}  # iv name -> (values, width) of enclosing constant-trip loops
        self.uses_cluster_wait = False  # the role waits on the cluster barrier
        self._inst: dict = {}

    # ---------------------------------------------------------------- basics
    def loc(self, op) -> str:
        """op's source `file:line`; an op without one (built inside a DSL helper or by a
        pass) gets the location of the closest preceding op that has one."""
        return self.k.user_loc(op) or self.cur_loc

    def fail(self, op, why: str) -> NoReturn:
        raise Unsupported(op.name, self.loc(op), why)

    def fresh(self, hint: str) -> str:
        self.nfresh += 1
        return f"{hint}{self.nfresh}"

    def val(self, op, i: int):
        name = op.operands[i]
        if name not in self.env:
            self.fail(
                op, f"operand {i} ({name}) has no value (op table and slice disagree)"
            )
        return self.env[name]

    def as_int(self, op, v, use: str):
        if isinstance(v, Data):
            self.fail(op, f"data value from {v.origin} reaches {use} (slice soundness)")
        if isinstance(v, Tup):
            ls = L.leaves(v.t)
            if len(ls) == 1 and isinstance(ls[0], tuple):
                return ls[0]
        if isinstance(v, tuple):
            return v
        self.fail(op, f"{use} needs an integer, got {type(v).__name__}")

    def int_of(self, op, i: int, use: str):
        return self.as_int(op, self.val(op, i), use)

    def data(self, op) -> Data:
        return Data(f"{op.name} at {self.loc(op)}")

    def emit(self, op, stmt: dict) -> None:
        if self.host:
            self.fail(op, "a statement in the host function")
        stmt.setdefault("loc", self.loc(op))
        self.stmts.append(stmt)

    def bind(self, e, op, hint: str = "v"):
        """A `let` for a thread-uniform, non-trivial expression; its variable is returned."""
        if (
            self.host
            or e[0] in ("const", "var", "pid", "nprog")
            or E.thread_varying(e)
            or E.reserved(e)
        ):
            return e
        name = self.fresh(hint)
        self.stmts += self.lets(op, [(name, e)])
        self.bound[name] = e
        return E.var(name, E.width(e))

    def unbind(self, e, depth: int = 8):
        """e with the variables of `bind` lets replaced by the expressions they hold (at
        most `depth` levels): the constant part of a barrier address an index variable
        hides (to_kir.py picks the barrier array by it)."""
        for _ in range(depth):
            if E.free_vars(e).isdisjoint(self.bound):
                break
            e = E.subst(e, self.bound)
        return e

    def lets(self, op, pairs: list) -> list:
        """`let name = e` statements for thread-uniform expressions; subexpressions they
        share are bound first, so the JSON stays linear in the expression DAG."""
        shared = _shared_nodes([e for _n, e in pairs], self.fresh("s"))
        loc = self.loc(op)
        out = [
            {"op": "let", "var": n, "expr": E.to_json(x, shared, n), "loc": loc}
            for x, n in shared.items()
        ]
        return out + [
            {"op": "let", "var": n, "expr": E.to_json(e, shared, n), "loc": loc}
            for n, e in pairs
        ]

    def require(self, op, cond, msg: str) -> None:
        """A condition the op's meaning depends on, checked by a KIR assert (per issuing
        thread when it varies across threads or holds under a mask)."""
        u = self.uniform(cond)
        if u is not None and E.is_const(u, 1):
            return
        if u is not None and E.is_const(self.mask, 1):
            self.emit(op, {"op": "assert", "cond": E.to_json(u), "msg": msg})
            return
        self.thread_event(
            op,
            [cond],
            lambda v, j: {
                "op": "assert",
                "cond": j(v[0]),
                "msg": msg,
                "loc": self.loc(op),
            },
        )

    def guard(self, x, safe: int):
        """`x` where the current mask holds, `safe` elsewhere (operands of partial ops)."""
        return (
            x
            if E.is_const(self.mask, 1)
            else E.select(self.mask, x, E.const(safe, E.width(x)))
        )

    # --------------------------------------------------------------- threads
    def threads(self):
        return [
            (w, lane) for w in range(self.w0, self.w0 + self.nw) for lane in range(32)
        ]

    def thread_subst(self, w, lane) -> dict:
        return {
            "%warp": E.const(w, 32),
            "%lane": E.const(lane, 32),
            "%elect": E.const(int(lane == 0), 1),
        }

    def instances(self, e) -> list:
        """e for every thread of the role, in threads() order (folded)."""
        r = self._inst.get(e)
        if r is None:
            fv = E.free_vars(e)
            if fv.isdisjoint(E.THREAD_VARS):
                r = [e] * (32 * self.nw)
            else:
                r = []
                for w in range(self.w0, self.w0 + self.nw):
                    ew = E.subst(e, {"%warp": E.const(w, 32)}) if "%warp" in fv else e
                    for lane in range(32):
                        r.append(
                            E.subst(
                                ew,
                                {
                                    "%lane": E.const(lane, 32),
                                    "%elect": E.const(int(lane == 0), 1),
                                },
                            )
                        )
            self._inst[e] = r
        return r

    def uniform(self, e):
        """The expression if it has one value for every thread of the role, else None."""
        if not E.thread_varying(e):
            return e
        inst = self.instances(e)
        return inst[0] if all(x is inst[0] for x in inst) else None

    def split_per_warp(self, e) -> None:
        """Raise SplitWarps on the runs of warps with one value of `e` if e is uniform
        within each warp but not across the role; else return."""
        pw = self.per_warp(e)
        if pw is None or all(x is pw[0] for x in pw):
            return
        groups, start = [], self.w0
        for i in range(1, self.nw + 1):
            if i == self.nw or pw[i] is not pw[i - 1]:
                groups.append((start, self.w0 + i - start))
                start = self.w0 + i
        raise SplitWarps(groups)

    def per_warp(self, e):
        """[value per warp] if e is uniform within each warp, else None."""
        inst = self.instances(e)
        out = []
        for i in range(self.nw):
            part = inst[32 * i : 32 * i + 32]
            if not all(x is part[0] for x in part):
                return None
            out.append(part[0])
        return out

    def lane_only(self, e) -> bool:
        """e does not depend on WARP (one value per lane, the same in every warp)."""
        inst = self.instances(e)
        return all(inst[i] is inst[i % 32] for i in range(len(inst)))

    def thread_key(self, lane_only: bool):
        if lane_only:
            return E.LANE
        return E.bin(
            "add",
            E.bin("mul", E.bin("sub", E.WARP, E.const(self.w0, 32)), E.const(32, 32)),
            E.LANE,
        )

    def lane_table(self, op, values: list, hint: str):
        """table over lanes (32 values) or threads (32 * nw values); entries bound."""
        bound: dict = {}
        entries = tuple(
            bound[v] if v in bound else bound.setdefault(v, self.bind(v, op, hint))
            for v in values
        )
        return E.table(self.thread_key(len(values) == 32), entries)

    # ---------------------------------------------------------- variables
    # A slot stores one integer leaf of a value in KIR variables: ("u", name, w) for a
    # uniform value, ("f", names, w) for a family indexed by lane (32) or thread.

    def new_slot(self, e, kind: str, hint: str):
        """A slot for values like e; its variables exist once assigned."""
        if kind == "u":
            return ("u", self.fresh(hint), E.width(e))
        base = self.fresh(hint)
        n = 32 if kind == "lane" else 32 * self.nw
        return ("f", tuple(f"{base}_{i}" for i in range(n)), E.width(e))

    def assign(self, op, slot, e) -> list:
        if slot[0] == "u":
            u = self.uniform(e)
            if u is None:
                self.fail(op, "thread-varying value assigned to a uniform variable")
            return self.lets(op, [(slot[1], u)])
        names = slot[1]
        inst = self.instances(e)
        if len(names) == 32 and not self.lane_only(e):
            self.fail(op, "warp-dependent value assigned to a per-lane variable")
        return self.lets(op, list(zip(names, inst)))

    def read(self, slot):
        if slot[0] == "u":
            return E.var(slot[1], slot[2])
        return E.table(
            self.thread_key(len(slot[1]) == 32),
            tuple(E.var(n, slot[2]) for n in slot[1]),
        )

    def kind_of(self, e) -> str:
        if self.uniform(e) is not None:
            return "u"
        return "lane" if self.lane_only(e) else "thread"

    @staticmethod
    def join(a: str, b: str) -> str:
        order = ("u", "lane", "thread")
        return order[max(order.index(a), order.index(b))]

    def slot_kind(self, slot) -> str:
        return "u" if slot[0] == "u" else ("lane" if len(slot[1]) == 32 else "thread")

    # ------------------------------------------------- structured values
    def flatten(self, op, v):
        """(skeleton, integer leaves) of an abstract value; Data -> (None, [])."""
        if isinstance(v, Data):
            return None, []
        if isinstance(v, tuple):
            return ("int", E.width(v)), [v]
        if isinstance(v, Tup):
            return ("tup", _signature(v.t)), _int_leaves(v.t)
        if isinstance(v, Lay):
            return ("lay", _signature(v.shape), _signature(v.stride)), _int_leaves(
                v.shape
            ) + _int_leaves(v.stride)
        if isinstance(v, Composed):
            s, ls = self.flatten(op, v.lay)
            return ("composed", v.swizzle, v.offset, s), ls
        if isinstance(v, Ptr):
            return ("ptr", v.space, v.base, v.bits), [v.off]
        if isinstance(v, Remote):
            return ("remote", v.ptr.space, v.ptr.base, v.ptr.bits), [v.ptr.off, v.cta]
        if isinstance(v, ArithIter):
            return ("iter", len(v.coord)), list(v.coord)
        if isinstance(v, View):
            (s1, l1), (s2, l2) = self.flatten(op, v.ptr), self.flatten(op, v.lay)
            return ("view", s1, s2), l1 + l2
        if isinstance(v, Opaque):
            if not self.p1 or v.kind != "atom":
                return ("opaque", v), []
            # P1: an atom's integer and pointer fields are leaves (loop-carried and merged
            # like any value); other fields are part of the skeleton.
            skel, leaves = [], []
            for key, x in v.info:
                if isinstance(x, (tuple, Ptr)):
                    s, ls = self.flatten(op, x)
                    skel.append((key, s))
                    leaves += ls
                else:
                    skel.append((key, ("value", x)))
            return ("atom", tuple(skel)), leaves
        self.fail(op, f"cannot store a value of kind {type(v).__name__}")

    def unflatten(self, v, leaves: list):
        """v with its integer leaves replaced (inverse of flatten for the same skeleton)."""
        it = iter(leaves)

        def tree(t):
            return L.rebuild(
                t, [next(it) if isinstance(x, tuple) else x for x in L.leaves(t)]
            )

        def go(x):
            if isinstance(x, tuple):
                return next(it)
            if isinstance(x, Tup):
                return Tup(tree(x.t))
            if isinstance(x, Lay):
                return Lay(tree(x.shape), tree(x.stride))
            if isinstance(x, Composed):
                return Composed(x.swizzle, x.offset, go(x.lay))
            if isinstance(x, Ptr):
                return Ptr(x.space, x.base, next(it), x.bits)
            if isinstance(x, Remote):
                off = next(it)
                return Remote(Ptr(x.ptr.space, x.ptr.base, off, x.ptr.bits), next(it))
            if isinstance(x, ArithIter):
                return ArithIter(tuple(next(it) for _ in x.coord))
            if isinstance(x, View):
                return View(go(x.ptr), go(x.lay))
            if isinstance(x, Opaque) and self.p1 and x.kind == "atom":
                return Opaque(
                    x.kind,
                    tuple(
                        (key, go(y) if isinstance(y, (tuple, Ptr)) else y)
                        for key, y in x.info
                    ),
                )
            return x

        return go(v)

    def join_atoms(self, op, values: list) -> list:
        """P1: atoms joined at a merge (scf.if results, loop-carried values) get one
        skeleton: a field missing on one side, or with different non-integer values, or of
        different structure, becomes Data on every side (a kept read of it is then the slice
        soundness error, R8); integer and pointer fields stay leaves."""
        if not self.p1 or not all(
            isinstance(v, Opaque) and v.kind == "atom" for v in values
        ):
            return values
        if len({v.get("type") for v in values}) != 1:
            return values
        keys = sorted({key for v in values for key, _x in v.info})
        bad = set()
        for key in keys:
            xs = [dict(v.info).get(key, Data("")) for v in values]
            if any(isinstance(x, Data) for x in xs):
                bad.add(key)
            elif all(isinstance(x, (tuple, Ptr)) for x in xs):
                if len({self.flatten(op, x)[0] for x in xs}) != 1:
                    bad.add(key)
            elif any(x != xs[0] for x in xs):
                bad.add(key)
        if not bad:
            return values
        d = Data(
            f"atom field merged at {self.loc(op)} with different values or structure"
        )
        return [
            Opaque(
                v.kind,
                tuple(
                    (key, d if key in bad else dict(v.info).get(key)) for key in keys
                ),
            )
            for v in values
        ]

    def merged_skeleton(self, op, values: list, what: str):
        """Common skeleton of values joined at a control-flow merge, or None for Data."""
        values = self.join_atoms(op, values)
        skels = [self.flatten(op, v) for v in values]
        if any(s is None for s, _ in skels):
            return None, None
        if any(s != skels[0][0] for s, _ in skels):
            self.fail(op, f"{what}: values of different structure are merged")
        return skels[0][0], [ls for _, ls in skels]

    # --------------------------------------------------------- blocks and ops
    def run_block(self, block, args: list) -> list:
        """Translate a block; returns the values of its terminator's operands."""
        for (name, _ty), v in zip(block.args, args):
            self.env[name] = v
        for op in block.ops:
            self.cur_loc = self.k.user_loc(op) or self.cur_loc
            if self.p1 and (not self.loc_hist or self.loc_hist[-1] != self.cur_loc):
                self.loc_hist = self.loc_hist[-31:] + [self.cur_loc]
            if op.name in ("scf.yield", "scf.condition"):
                return [
                    self.env.get(x, Data("an operand outside the slice"))
                    for x in op.operands
                ]
            if op.name in ("func.return", "cuda.return"):
                return []
            self.run_op(op)
        return []

    def run_op(self, op) -> None:
        """Translate op if the slice wants it; the slice has already rejected every op
        without a table entry (ops.effect_operands)."""
        assert self.k.slice is not None, "roles are translated after slicing"
        if not self.k.slice.wanted(op):
            return
        try:
            res = self.spec(op)(self, op)
        except (L.LayoutError, E.ExprError, ValueError) as e:
            self.fail(op, str(e))
        if op.results:
            res = res if isinstance(res, list) else [res]
            if len(res) != len(op.results):
                self.fail(
                    op, f"op table gave {len(res)} values for {len(op.results)} results"
                )
            for r, v in zip(op.results, res):
                self.env[r] = self.scalarize(op, v)

    def scalarize(self, op, v):
        """A lane-varying integer computed from per-lane memory reads or shuffles (warp-
        cooperative code) becomes a table of 32 bound per-lane values, so that later uses
        (more shuffles, per-lane variables, thread loops) do not copy its expression."""
        if (
            not isinstance(v, tuple)
            or self.host
            or not E.thread_varying(v)
            or not E.has_data_nodes(v)
        ):
            return v
        if not self.lane_only(v) or "%elect" in E.free_vars(v):
            return v
        return self.lane_table(op, self.instances(v)[:32], "l")

    def snapshot(self):
        return len(self.stmts), len(self.barriers), self.stmts

    def restore(self, snap) -> None:
        n, nb, stmts = snap
        self.stmts = stmts
        del self.stmts[n:]
        del self.barriers[nb:]

    def relevant(self, name: str) -> bool:
        assert self.k.slice is not None, "roles are translated after slicing"
        return name in self.k.slice.relevant

    # ------------------------------------------------------------- scf.if
    def scf_if(self, op):
        if self.host:
            self.fail(op, "control flow in the host function")
        c = self.int_of(op, 0, "branch condition")
        then_b = op.regions[0][0] if op.regions[0] else None
        else_b = op.regions[1][0] if len(op.regions) > 1 and op.regions[1] else None
        cu = self.uniform(c)
        if cu is not None and E.is_const(cu):
            taken = then_b if cu[2] else else_b
            return self.run_block(taken, []) if taken else []
        if cu is None:
            self.split_per_warp(c)
            return self._masked_if(op, c, then_b, else_b)
        return self._uniform_if(op, cu, then_b, else_b)

    def _branch(self, block):
        saved = self.stmts
        self.stmts = []
        ys = self.run_block(block, []) if block else []
        out, self.stmts = self.stmts, saved
        return out, ys

    def _uniform_if(self, op, c, then_b, else_b):
        then_s, ty = self._branch(then_b)
        else_s, ey = self._branch(else_b)
        results = []
        for i, r in enumerate(op.results):
            skel, parts = (
                self.merged_skeleton(op, [ty[i], ey[i]], "scf.if")
                if self.relevant(r)
                else (None, None)
            )
            if skel is None:
                results.append(Data(f"result of scf.if at {self.loc(op)}"))
                continue
            ty_i = self.join_atoms(op, [ty[i], ey[i]])[0]
            leaves = []
            for x, y in zip(*parts):
                if x is y:
                    leaves.append(x)
                    continue
                slot = self.new_slot(
                    x, self.join(self.kind_of(x), self.kind_of(y)), "r"
                )
                self.stmts += self.assign(
                    op, slot, E.const(0, E.width(x))
                )  # declared before the if
                then_s += self.assign(op, slot, x)
                else_s += self.assign(op, slot, y)
                leaves.append(self.read(slot))
            results.append(self.unflatten(ty_i, leaves))
        if then_s or else_s:
            self.emit(
                op, {"op": "if", "cond": E.to_json(c), "then": then_s, "else": else_s}
            )
        return results

    def _masked_if(self, op, c, then_b, else_b):
        saved = self.mask
        self.mask = E.and1(saved, c)
        ty = self.run_block(then_b, []) if then_b else []
        self.mask = E.and1(saved, E.not1(c))
        ey = self.run_block(else_b, []) if else_b else []
        self.mask = saved
        results = []
        for i, r in enumerate(op.results):
            skel, parts = (
                self.merged_skeleton(op, [ty[i], ey[i]], "scf.if")
                if self.relevant(r)
                else (None, None)
            )
            if skel is None:
                results.append(Data(f"result of scf.if at {self.loc(op)}"))
                continue
            results.append(
                self.unflatten(
                    self.join_atoms(op, [ty[i], ey[i]])[0],
                    [E.select(c, x, y) for x, y in zip(*parts)],
                )
            )
        return results

    # ------------------------------------------------------------- loops
    def _slots_for(self, op, values: list, kinds: list):
        """Declare slots for the integer leaves of each value (None for Data); returns
        (slots per value, values read back from the slots)."""
        slots, reads = [], []
        for v, kinds_v in zip(values, kinds):
            skel, ls = self.flatten(op, v)
            if skel is None:
                slots.append(None)
                reads.append(v)
                continue
            s = [
                self.new_slot(x, kinds_v[j] if kinds_v else self.kind_of(x), "c")
                for j, x in enumerate(ls)
            ]
            self.stmts += [a for sl, x in zip(s, ls) for a in self.assign(op, sl, x)]
            slots.append(s)
            reads.append(self.unflatten(v, [self.read(sl) for sl in s]))
        return slots, reads

    def _widen(
        self, op, slots: list, values: list, kinds: list, inits: list | None = None
    ) -> bool:
        """Raise the slot kinds needed by `values`; True if any kind changed. A yield whose
        skeleton differs from its initial value's (e.g. a pointer into another tensor) is
        refused: the slots carry integer leaves only. P1: a loop-carried atom whose yielded
        skeleton differs from its initial one is joined into `inits` (join_atoms) and the
        loop translated again."""
        changed = False
        for k, (s, v) in enumerate(zip(slots, values)):
            if s is None:
                continue
            if (
                self.p1
                and inits is not None
                and isinstance(v, Opaque)
                and isinstance(inits[k], Opaque)
            ):
                if self.flatten(op, v)[0] != self.flatten(op, inits[k])[0]:
                    joined = self.join_atoms(op, [inits[k], v])
                    if self.flatten(op, joined[0])[0] == self.flatten(op, inits[k])[0]:
                        if (
                            self.flatten(op, joined[1])[0]
                            != self.flatten(op, inits[k])[0]
                        ):
                            self.fail(op, "a loop-carried atom changes its type (P1)")
                        values[k] = v = joined[1]  # the yield in the carried skeleton
                    else:
                        inits[k] = joined[0]
                        kinds[k] = None
                        changed = True
                        continue
            skel, ls = self.flatten(op, v)
            if skel is None:
                self.fail(op, "a loop-carried value becomes data")
            if inits is not None and skel != self.flatten(op, inits[k])[0]:
                self.fail(op, "a loop-carried value changes structure")
            need = [
                self.join(self.slot_kind(sl), self.kind_of(x)) for sl, x in zip(s, ls)
            ]
            if need != [self.slot_kind(sl) for sl in s]:
                kinds[k] = need
                changed = True
        return changed

    def _parallel_assign(self, op, slots: list, values: list) -> list:
        """Assignments slot := value for all pairs, as if simultaneous. A family's lets run
        one entry after another, so an entry that reads another entry of its own family
        (e.g. a shuffle) goes through a temporary too."""
        pairs = [
            (sl, x)
            for s, v in zip(slots, values)
            if s is not None
            for sl, x in zip(s, self.flatten(op, v)[1])
        ]
        written = set()
        temps, final = [], []
        for sl, x in pairs:
            names = {sl[1]} if sl[0] == "u" else set(sl[1])
            reads = E.free_vars(x)
            cross = sl[0] == "f" and any(
                (E.free_vars(y) & names) - {n} for n, y in zip(sl[1], self.instances(x))
            )
            if reads & written or cross:
                tmp = self.new_slot(x, self.slot_kind(sl), "tmp")
                temps += self.assign(op, tmp, x)
                x = self.read(tmp)
            final += self.assign(op, sl, x)
            written |= names
        return temps + final

    def scf_for(self, op):
        if self.host:
            self.fail(op, "control flow in the host function")
        if "unsignedCmp" in op.props or "unsignedCmp" in op.attrs:
            self.fail(op, "scf.for with unsigned comparison")
        bounds = []
        for i in range(3):
            b = self.uniform(self.int_of(op, i, "loop bound"))
            if b is None:
                self.fail(op, "loop bounds vary across the threads of the role")
            bounds.append(b)
        lb, ub, step = bounds
        if E.is_const(step) and step[2] <= 0:
            self.fail(op, f"scf.for with non-positive step {step[2]}")
        body = op.regions[0][0]
        n = len(op.operands) - 3
        carried = [
            i
            for i in range(n)
            if self.relevant(body.args[1 + i][0]) or self.relevant(op.results[i])
        ]
        inits = [self.val(op, 3 + i) for i in carried]
        kinds = [None] * len(carried)
        while True:
            snap = self.snapshot()
            slots, reads = self._slots_for(op, inits, kinds)
            iv = self.fresh("i")
            args = [E.var(iv, E.width(lb))] + [
                Data(f"loop-carried value of scf.for at {self.loc(op)}")
            ] * n
            for j, i in enumerate(carried):
                args[1 + i] = reads[j]
            saved = self.stmts
            self.stmts = []
            if (
                all(E.is_const(b) for b in bounds)
                and len(range(lb[2], ub[2], step[2])) <= _MAX_ENUMERATED
            ):
                self.loop_values[iv] = (range(lb[2], ub[2], step[2]), E.width(lb))
            self.loop_depth += 1
            ys = self.run_block(body, args)
            self.loop_depth -= 1
            self.loop_values.pop(iv, None)
            yields = [ys[i] for i in carried]
            if self._widen(op, slots, yields, kinds, inits):
                self.restore(snap)
                continue
            body_stmts = self.stmts + self._parallel_assign(op, slots, yields)
            self.stmts = saved
            break
        if body_stmts:
            self.emit(op, {"op": "for", "var": iv, "lb": E.to_json(lb), "ub": E.to_json(ub),
                           "step": E.to_json(step), "body": body_stmts})  # fmt: skip
        results = [Data(f"result of scf.for at {self.loc(op)}")] * n
        for j, i in enumerate(carried):
            results[i] = reads[j]
        return results

    def scf_while(self, op):
        if self.host:
            self.fail(op, "control flow in the host function")
        before, after = op.regions[0][0], op.regions[1][0]
        n = len(op.operands)
        nf = len(before.ops[-1].operands) - 1
        carried = [i for i in range(n) if self.relevant(before.args[i][0])]
        fwd = [
            j
            for j in range(nf)
            if self.relevant(after.args[j][0]) or self.relevant(op.results[j])
        ]
        inits = [self.val(op, i) for i in carried]
        kinds_c, kinds_f = [None] * len(carried), [None] * len(fwd)
        while True:
            snap = self.snapshot()
            slots_c, reads_c = self._slots_for(op, inits, kinds_c)
            cond, fwd_vals = self._before(op, before, n, carried, reads_c)
            cond_slot = self.new_slot(cond, "u", "cond")
            self.stmts += self.assign(op, cond_slot, cond)
            slots_f, reads_f = self._slots_for(op, [fwd_vals[j] for j in fwd], kinds_f)
            saved = self.stmts
            self.stmts = []
            args = [Data(f"value of scf.while at {self.loc(op)}")] * nf
            for k, j in enumerate(fwd):
                args[j] = reads_f[k]
            self.loop_depth += 1
            ys = self.run_block(after, args)
            self.loop_depth -= 1
            yields = [ys[i] for i in carried]
            if self._widen(op, slots_c, yields, kinds_c, inits):
                self.restore(snap)
                continue
            self.stmts += self._parallel_assign(op, slots_c, yields)
            cond2, fwd2 = self._before(op, before, n, carried, reads_c)
            fwd2 = [fwd2[j] for j in fwd]
            if self._widen(op, slots_f, fwd2, kinds_f):
                self.restore(snap)
                continue
            self.stmts += self.assign(op, cond_slot, cond2)
            self.stmts += self._parallel_assign(op, slots_f, fwd2)
            body_stmts, self.stmts = self.stmts, saved
            break
        self.emit(op, {"op": "while", "cond": E.to_json(self.read(cond_slot)), "body": body_stmts,
                       "max_iters": self.k.max_iters})  # fmt: skip
        results = [Data(f"result of scf.while at {self.loc(op)}")] * nf
        for k, j in enumerate(fwd):
            results[j] = reads_f[k]
        return results

    def _before(self, op, before, n, carried, reads_c):
        args = [Data(f"loop-carried value of scf.while at {self.loc(op)}")] * n
        for k, i in enumerate(carried):
            args[i] = reads_c[k]
        outs = self.run_block(before, args)
        cond = self.uniform(self.as_int(op, outs[0], "while condition"))
        if cond is None:
            self.fail(op, "while condition varies across the threads of the role")
        return cond, outs[1:]

    # ---------------------------------------------------------------- events
    def split_mask(self):
        """(thread-uniform part, thread-varying part) of the current mask."""
        uni, var = E.TRUE, E.TRUE
        for c in _conjuncts(self.mask):
            if E.thread_varying(c):
                var = E.and1(var, c)
            else:
                uni = E.and1(uni, c)
        return uni, var

    def active(self, cond):
        """Indices (threads() order) of the threads where cond holds; None if that depends
        on runtime values."""
        inst = self.instances(cond)
        if not all(E.is_const(x) for x in inst):
            return None
        return [i for i, x in enumerate(inst) if x[2]]

    def _emit_under(self, op, uni, stmt: dict) -> None:
        if not E.is_const(uni, 1):
            stmt = {
                "op": "if",
                "cond": E.to_json(uni),
                "then": [stmt],
                "else": [],
                "loc": self.loc(op),
            }
        self.emit(op, stmt)

    def _thread_loop(self, op, cond, operands: list, build) -> dict:
        """`for t in [0, 32*nw): if cond(t): build(...)` with thread t's operand values.
        Shared subexpressions become `let`s in the loop body."""
        t = self.fresh("t")
        tv = E.var(t, 32)
        lane = E.bin("urem", tv, E.const(32, 32))
        m = {
            "%lane": lane,
            "%warp": E.bin(
                "add", E.bin("udiv", tv, E.const(32, 32)), E.const(self.w0, 32)
            ),
            "%elect": E.cmp("eq", lane, E.const(0, 32)),
        }
        cache: dict = {}
        vals = [E.subst(e, m, cache) for e in operands]
        c = E.subst(cond, m, cache)
        names = _shared_nodes(vals + [c], t)
        loc = self.loc(op)
        body = [
            {"op": "let", "var": n, "expr": E.to_json(e, names, n), "loc": loc}
            for e, n in names.items()
        ]
        inner = build(vals, lambda e: E.to_json(e, names))
        body += (
            [inner]
            if E.is_const(c, 1)
            else [
                {
                    "op": "if",
                    "cond": E.to_json(c, names),
                    "then": [inner],
                    "else": [],
                    "loc": loc,
                }
            ]
        )
        one = E.to_json(E.const(1, 32))
        return {"op": "for", "var": t, "lb": E.to_json(E.const(0, 32)), "ub": E.to_json(E.const(32 * self.nw, 32)),
                "step": one, "body": body, "loc": loc}  # fmt: skip

    def thread_event(self, op, operands: list, build) -> None:
        """Emit an event once per issuing thread of the role. `operands` are the
        (possibly thread-varying) expressions it uses; build(values, to_json) returns the
        KIR statement for one thread's values."""
        uni, var = self.split_mask()
        self._check_elected(op, var, operands)
        act = self.active(var)
        if act == []:
            return
        if act is not None and len(act) == 1:
            self._emit_under(op, uni, self._thread_stmt(act[0], operands, build))
            return
        full = act is not None and len(act) == 32 * self.nw
        self._emit_under(
            op, uni, self._thread_loop(op, E.TRUE if full else var, operands, build)
        )

    def waterfall_event(self, op, key: list, operands: list, build) -> None:
        """Emit an event once per distinct value of the expressions `key` among the issuing
        lanes of each warp. This is the waterfall loop ptxas puts around an instruction
        that reads its operands from uniform registers (TMA store, ir_ops_cute.md): each
        round issues with the operands of one remaining lane and retires every lane that
        holds the same ones. So a lane issues iff no lower issuing lane of its warp holds
        its key. `operands` and `build` are as in thread_event."""
        uni, var = self.split_mask()
        self._check_elected(op, var, operands + key)
        conds = self.instances(var)
        keys = [self.instances(e) for e in key]
        first = []
        for i, c in enumerate(conds):
            for j in range(i - i % 32, i):
                if E.is_const(c, 0):
                    break
                same = conds[j]
                for k in keys:
                    same = E.and1(same, E.cmp("eq", k[j], k[i]))
                c = E.and1(c, E.not1(same))
            first.append(c)
        issuers = [i for i, c in enumerate(first) if not E.is_const(c, 0)]
        if len(issuers) == 1:
            i = issuers[0]
            self._emit_under(
                op, E.and1(uni, first[i]), self._thread_stmt(i, operands, build)
            )
        elif issuers:
            every = all(E.is_const(c, 1) for c in first)
            cond = E.TRUE if every else self.lane_table(op, first, "first")
            self._emit_under(op, uni, self._thread_loop(op, cond, operands, build))

    def _check_elected(self, op, var, operands: list) -> None:
        """Under elect.sync (`var` uses ELECT) the elected lane is unspecified, so the
        operands must not depend on which lane it is."""
        if "%elect" not in E.free_vars(var):
            return
        for e in operands:
            inst = self.instances(e)
            if any(inst[i] is not inst[i - i % 32] for i in range(len(inst))):
                self.fail(
                    op, "an operand under elect.sync depends on which lane is elected"
                )

    def _thread_stmt(self, i: int, operands: list, build) -> dict:
        """build() with the operand values of thread i (threads() order) of the role."""
        m = self.thread_subst(*self.threads()[i])
        return build([E.subst(e, m) for e in operands], E.to_json)

    def barrier_event(self, op, ev: str, ptr, mode: str, **fields) -> None:
        """An mbarrier event; `bar`/`idx` are resolved from the shared-memory address once
        every role is translated (to_kir.py). Modes: "once" (one event if any thread of
        the role issues it: bar_init, wait), "count" (one arrive counting the issuing
        threads), "thread" (once per issuing thread), "warp" (once per warp with an issuing
        thread: TMA loads, whose lowering elects a lane). A `Remote` pointer (mapa) names
        the barrier at that offset in another CTA of the cluster (the event's `cta`)."""
        if isinstance(ptr, Remote):
            ptr, fields = ptr.ptr, {**fields, "cta": self._remote_cta(op, ev, ptr)}
        if not isinstance(ptr, Ptr) or ptr.space != "smem" or ptr.bits % 8:
            self.fail(op, "mbarrier operand is not a shared-memory pointer")
        byte = E.bin("mul", ptr.off, E.const(ptr.bits // 8, 64))
        addr = self.uniform(byte)
        if addr is None:
            # one role per run of warps with one address, if that splits it
            self.split_per_warp(byte)
            self.fail(op, "mbarrier address varies across the threads of the role")
        stmt = {"op": "event", "ev": ev, "bar": None, "idx": None}
        for k, v in fields.items():
            if isinstance(v, tuple):
                u = self.uniform(v)
                if u is None:
                    self.fail(
                        op,
                        f"mbarrier event operand '{k}' varies across the threads of the role",
                    )
                v = E.to_json(u)
            elif isinstance(v, list):
                us = [self.uniform(x) for x in v]
                if None in us:
                    self.fail(
                        op,
                        f"mbarrier event operand '{k}' varies across the threads of the role",
                    )
                v = [E.to_json(u) for u in us]
            stmt[k] = v
        stmt["loc"] = self.loc(op)
        inits = ()
        if ev == "bar_init":
            inits = self._enumerate(addr)
            if inits is None:
                self.fail(
                    op,
                    "mbarrier initialized at an address that is not a constant of the kernel",
                )
        issued = self._issue(op, stmt, mode)
        if issued is not None:
            site = (self.k.user_frames(op), id(self.k.parent_of(op))) if inits else ()
            self.barriers.append(BarrierUse(stmt, addr, issued[2], stmt["loc"], inits, site=site,
                                            base=self.unbind(addr)))  # fmt: skip
            self._emit_under(op, issued[0], issued[1])

    def _remote_cta(self, op, ev: str, ptr: Remote):
        """The CTA rank of a remote barrier: uniform over the role (a role whose warps
        differ in it is split)."""
        if ev in ("bar_init", "wait"):
            self.fail(
                op, f"mbarrier {ev} on another CTA's barrier (shared::cluster address)"
            )
        cta = self.uniform(ptr.cta)
        if cta is None:
            self.split_per_warp(ptr.cta)
            self.fail(
                op,
                "the CTA rank of a remote mbarrier varies across the threads of the role",
            )
        return E.resize(cta, 32)

    def cluster_barrier_event(self, op, arrive: bool) -> None:
        """barrier.cluster.arrive / .wait: an arrive of every executing thread on, or a
        wait for the next phase of, the cluster barrier, a frontend-made barrier slot of
        the cluster's CTA of rank 0 with one arrival per thread of the cluster (to_kir.py).
        The j-th wait of a role waits for phase j: its parity is the role variable
        `cluster_bar_parity`, flipped after each wait."""
        stmt = {"op": "event", "ev": "arrive" if arrive else "wait", "bar": CLUSTER_BAR,
                "idx": E.to_json(E.const(0, 32)), "cta": E.to_json(E.const(0, 32)),
                "loc": self.loc(op)}  # fmt: skip
        if not arrive:
            stmt["parity"] = E.to_json(E.var(CLUSTER_PARITY, 32))
            self.uses_cluster_wait = True
        _uni, var = self.split_mask()
        act = self.active(var)
        if act is None or (act and len(act) != 32 * self.nw):
            self.fail(
                op,
                "barrier.cluster reached by only part of the role (or under a runtime lane condition)",
            )
        issued = self._issue(op, stmt, "count" if arrive else "once")
        if issued is None:
            return
        self.barriers.append(
            BarrierUse(stmt, E.const(0, 64), issued[2], stmt["loc"], (), CLUSTER_BAR)
        )
        flip = {"op": "let", "var": CLUSTER_PARITY, "loc": stmt["loc"],
                "expr": E.to_json(E.bin("xor", E.var(CLUSTER_PARITY, 32), E.const(1, 32)))}  # fmt: skip
        self._emit_under(op, issued[0], issued[1])
        if not arrive:
            self._emit_under(op, issued[0], flip)

    def warp_event(self, op, stmt: dict) -> None:
        """Emit an event with thread-uniform operands once per warp with an issuing thread
        (the issue pattern of a TMA load, barrier_event mode "warp")."""
        issued = self._issue(op, stmt, "warp")
        if issued is not None:
            self._emit_under(op, issued[0], issued[1])

    def _issue(self, op, stmt: dict, mode: str):
        """(uniform condition, statement, issuer count in mode "count") for an event issued
        under the current mask in `mode` (barrier_event), or None if no thread issues it."""
        uni, var = self.split_mask()
        act = self.active(var)
        if act == []:
            return None
        if self.p1 and self.nw > 1 and stmt["ev"] in P1_SYNC_EVENTS:
            self._p1_split_by_issuers(op, stmt["ev"], act)
        issuers = None
        if act is None:
            if mode in ("once", "warp"):
                self.fail(
                    op, "issued under a lane condition that depends on runtime values"
                )
            out = self._thread_loop(op, var, [], lambda _v, _j: stmt)
        elif mode == "once":
            out = stmt
        elif mode == "count":
            out, issuers = stmt, len(act)
        else:
            k = (
                len(act)
                if mode == "thread"
                else self.full_warps(op, act, "TMA load (lowered to elect.sync)")
            )
            out = stmt if k == 1 else {
                "op": "for", "var": self.fresh("issuer"), "lb": E.to_json(E.const(0, 32)),
                "ub": E.to_json(E.const(k, 32)), "step": E.to_json(E.const(1, 32)), "body": [stmt],
                "loc": stmt["loc"]}  # fmt: skip
        return uni, out, issuers

    def _p1_split_by_issuers(self, op, ev: str, act) -> None:
        """P1 pass: an mbarrier arrive or commit releases, and a wait acquires, the TMEM
        accesses of the warps that issue it only, while the KIR takes a role's event as an
        event of every warp of the role. So a role of several warps whose warps do not all
        issue it (a thread-0 arrive or wait) is split into runs of issuing and non-issuing
        warps (SplitWarps; each part is translated again), and one issued under a runtime
        lane condition is refused (R7). Any lane of a warp issues for the warp (warp
        granularity, as the warp-collective tcgen05 ops). The P0 pass keeps its roles."""
        if act is None:
            self.fail(
                op,
                f"R7: mbarrier {ev} under a lane condition that depends on runtime values "
                f"in a role of {self.nw} warps (the warps it orders are unknown)",
            )
        issuing = {i // 32 for i in act}
        if len(issuing) == self.nw:
            return
        groups, start = [], 0
        for i in range(1, self.nw + 1):
            if i == self.nw or (i in issuing) != (i - 1 in issuing):
                groups.append((self.w0 + start, i - start))
                start = i
        raise SplitWarps(groups)

    def full_warps(self, op, act: list, what: str) -> int:
        """The number of warps among the threads `act` (threads() indices), each of which
        must be fully active: `what` runs elect.sync with the full member mask, undefined
        in PTX when a lane of the warp does not execute it."""
        lanes: dict = {}
        for i in act:
            lanes[i // 32] = lanes.get(i // 32, 0) + 1
        partial = sorted(self.w0 + w for w, n in lanes.items() if n != 32)
        if partial:
            self.fail(
                op,
                f"{what} with only part of warp(s) {partial} active (undefined in PTX)",
            )
        return len(lanes)

    def _enumerate(self, e):
        """Every value of e over the enclosing constant-trip loops (e may use only their
        induction variables), or None."""
        names = sorted(E.free_vars(e))
        if any(n not in self.loop_values for n in names):
            return None
        out = []
        for vals in itertools.product(*(self.loop_values[n][0] for n in names)):
            v = E.subst(
                e, {n: E.const(x, self.loop_values[n][1]) for n, x in zip(names, vals)}
            )
            if not E.is_const(v):
                return None
            out.append(v[2])
        return tuple(out)


_MAX_ENUMERATED = 64  # constant-trip loops whose induction values barrier inits may use

# The cluster barrier (barrier.cluster): a barrier array of one slot in the cluster's CTA of
# rank 0, and the role variable holding the parity of a role's next wait on it.
CLUSTER_BAR = "cluster_barrier"
CLUSTER_PARITY = "cluster_barrier_parity"

# mbarrier events that order TMEM accesses in P1-1 (Translator._p1_split_by_issuers).
P1_SYNC_EVENTS = ("arrive", "wait", "commit_arrive")


def _signature(t):
    """Structure of an IntTuple with its integer leaves reduced to their widths."""
    if isinstance(t, list):
        return tuple(_signature(x) for x in t)
    if isinstance(t, tuple):
        return ("i", E.width(t))
    return repr(t)


def _int_leaves(t) -> list:
    return [x for x in L.leaves(t) if isinstance(x, tuple)]


def _conjuncts(c) -> list:
    if c[0] == "bin" and c[2] == "and" and c[1] == 1:
        return _conjuncts(c[3]) + _conjuncts(c[4])
    return [] if E.is_const(c, 1) else [c]


def _shared_nodes(roots: list, prefix: str) -> dict:
    """Non-trivial nodes reachable more than once from `roots`, children first, mapped to
    fresh variable names (local CSE for thread-loop bodies)."""
    count: dict = {}
    order: list = []

    def visit(x):
        if x[0] in ("const", "var", "pid", "nprog"):
            return
        count[x] = count.get(x, 0) + 1
        if count[x] == 1:
            for c in E.children(x):
                visit(c)
            order.append(x)

    for r in roots:
        visit(r)
    return {x: f"{prefix}_{i}" for i, x in enumerate(n for n in order if count[n] > 1)}
