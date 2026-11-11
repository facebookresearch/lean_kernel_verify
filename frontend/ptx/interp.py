# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Translation of one role of a PTX kernel into KIR statements (design §5.4, §5.5).

The IR-agnostic core of frontend/cute/interp.py (thread model, slots and families,
masks, emitters, partition refinement), driven by the region tree of regions.py instead
of scf ops. A role is a contiguous warp range [w0, w0 + nw) running the same code; it is
translated with WARP and LANE symbolic. A branch whose condition is uniform within each
warp but differs between warps of the range raises SplitWarps when the kernel's policy
splits (CuTe: roles warps<a>_<b>), and the caller translates each part again; otherwise
(Triton: the roles are the warp-specialize partitions) such a branch is a mask.

Registers are mutable slots, not SSA values: `env` maps each register to
  * an integer expression (expr.py) at the register's width: thread-varying ones over
    LANE/WARP/elect symbols, addresses over the reserved bases of mem.py;
  * Data(origin): a value the model drops (floats, shared-memory contents, data loads).
    Data reaching a root (an event operand, a branch condition, an address, a predicate,
    a barrier field, a work anchor) is the slice soundness error, named by its origin.
Nothing is emitted for a value until a root needs it (so dead integer code costs
nothing): then each thread-uniform subexpression an instruction computed is bound by a
`let` at that instruction's loc, before the root. Bindings are scoped like KIR
variables: one made inside an `if` branch or a loop body is forgotten after it. A node
whose evaluation can stop the run (a metadata load, a division by a value that may be
zero; expr.unsafe) is never bound ahead of a `select` arm holding it: KIR evaluates only
the chosen arm, and the arm is what a predicate or a divergent branch guarded.

Control flow (regions.py). An If with a thread-uniform condition is a KIR `if`; the
registers live at its join that differ get `select(c, then, else)` or, when a value uses
a name bound inside its branch, a slot assigned in both branches. A thread-varying
condition is a mask: both branches are translated in place, events carry the mask,
registers merge by select, and partial operations and metadata loads under the mask have
their operand replaced where it is false. A predicated instruction is a one-instruction
mask region. A Loop is a KIR `for` if it is an exact signed-i32 counted loop with
constant bounds (_counted), else a `while` on an exit flag with the policy's max_iters:
the registers it writes that are live at its header or exit are carried in slots (kinds
widened, and the trailing zeros of an address offset slot narrowed, to a fixed point;
assigned in parallel at every back edge and exit edge), and an exit edge clears the
flag. A Wait is one `wait` event per role.

Events execute once per issuing thread (frontend/cute/interp.py's rules): a single event
when one thread issues it, else a `for` over the role's threads with the thread's mask
as an `if` (consecutive ones share the loop); waits and inits once per role, arrivals
counted. Under the Triton policy a global access of several threads is one event over
them (mem_event). The holes of the CuTe frontend are closed (design §5.4.1):
  H1 the elect check is on the final event mask: under an elect symbol every other
     conjunct must be the same in every lane of a warp, and the operands must not depend
     on the lane;
  H2 elect symbols stay symbolic through merges (select); a value depending on the
     elected lane is never judged uniform (a branch on it is a mask), and one that would
     be an event operand or need a per-thread slot is refused rather than made concrete
     (lane 0);
  H3 full-mask elect/shfl/vote need every lane of each executing warp active (asserted
     per warp when the mask depends on runtime values).
"""

import re
from dataclasses import dataclass, field
from typing import NoReturn, TYPE_CHECKING

from . import expr as E, ops, parse, regions as R

if TYPE_CHECKING:  # mem.py imports this module
    from .mem import Memory


class Unsupported(Exception):
    """A construct the frontend cannot translate exactly. `kind`: "gap", refused by
    design (a known limit), or "error", something the case, the dump or LKV must fix."""

    def __init__(self, op: str, loc: str, why: str, kind: str = "gap"):
        super().__init__(f"{op} at {loc}: {why}")
        self.op, self.loc, self.why, self.kind = op, loc, why, kind


class SplitWarps(Exception):
    def __init__(self, groups: list):
        super().__init__(f"split into warp groups {groups}")
        self.groups = groups  # [(w0, nw)]


class _Widen(Exception):
    """A loop-carried slot needs a wider kind: translate the loop again."""


@dataclass(frozen=True)
class Data:
    origin: str


@dataclass
class Kernel:
    """State shared by the roles of one kernel (mem.Memory holds the memory model)."""

    fn: parse.Function
    mem: "Memory"
    locs: dict  # id(instr) -> KIR loc "path:line"
    block: tuple = (32, 1, 1)
    cluster: tuple = (1, 1, 1)
    max_iters: int = 1 << 16
    split: bool = True  # SplitWarps (CuTe role naming) or masks (Triton partitions)
    vectorize: bool = False  # memory events over the threads as one TExpr (Triton)
    work: dict = field(default_factory=dict)  # id(instr) -> [anchor site]
    work_fields: tuple = ()
    work_mode: str = "gate"  # "gate" (the TTGIR rule) or "each" (the CuTe rule)
    work_per: tuple = ()  # work fields that name tensor segments (`per`)
    origin: dict = field(
        default_factory=dict
    )  # expression -> (loc, line) of its instruction
    srclocs: dict = field(default_factory=dict)  # id(instr) -> (source path, line)
    elects: dict = field(default_factory=dict)  # id(elect.sync) -> its elect symbol


# An mbarrier event awaiting its array (mem.Memory.barriers): the KIR event (bar and idx
# filled in then), the shared-memory symbol, the constant part `const` of the slot's byte
# offset and the KIR JSON of the rest divided by 8 (None for a constant offset; bound in
# the role's scope at the event), the interval of the offset (CuTe array splitting), the
# issuing thread count of an arrive (None when counted inside a thread loop), and for a
# bar_init its offset, its source line (path, line) and the key its array is split by
# (design §5.6).
@dataclass
class BarrierUse:
    event: dict
    sym: str
    const: int
    rest: dict | None
    rng: tuple | None
    issuers: object
    loc: str
    init: int | None = None
    srcloc: tuple | None = None
    key: tuple = ()


class Translator:
    def __init__(self, k: Kernel, tree, w0: int, nw: int, name: str):
        self.k, self.tree = k, tree
        self.w0, self.nw, self.name = w0, nw, name
        self.env: dict = {}
        self.stmts: list = []
        self.mask = E.TRUE
        self.nfresh = 0
        self.barriers: list = []
        self.created: list = []  # every KIR variable name made so far, in order
        self.names: dict = {}  # expression -> visible let variable
        self.scopes: list = [[]]  # expressions bound in each open scope
        self.loops: list = []  # open loops: [carried, slots, go, kinds, caps, latch If]
        self.regions: list = []  # open If branches: (id(node), side) (barrier keys)
        self.work_scopes: list = [{}]  # work fields defined in each open region
        self.work_events = 0
        self.slot_facts: dict = {}  # slot variable -> known trailing zeros
        self.cur = None  # the instruction being translated
        self._inst: dict = {}
        self._uniform_checked: dict = {}  # (id(statement list), mask, uniform) -> list
        self.pending = None  # events not emitted yet (mem_event, thread_event; flush)

    # ---------------------------------------------------------------- basics
    def loc(self, i=None) -> str:
        i = i if i is not None else self.cur
        return self.k.locs.get(id(i), "unknown") if i is not None else "unknown"

    def fail(self, i, why: str, kind: str = "gap") -> NoReturn:
        op = i.opcode if i is not None else "role"
        line = f" (PTX line {i.line})" if i is not None else ""
        raise Unsupported(op, self.loc(i) + line, why, kind)

    def fresh(self, hint: str) -> str:
        self.nfresh += 1
        name = f"{hint}{self.nfresh}"
        self.created.append(name)
        return name

    def emit(self, stmt: dict) -> None:
        self.flush()
        stmt.setdefault("loc", self.loc())
        self.stmts.append(stmt)

    def as_int(self, v, use: str):
        if isinstance(v, Data):
            self.fail(
                self.cur, f"data value from {v.origin} reaches {use} (slice soundness)"
            )
        if v is None:
            self.fail(self.cur, f"an undefined register reaches {use}")
        return v

    def json(self, e) -> dict:
        """KIR JSON of a thread-uniform expression; its subexpressions computed by
        instructions (and those it uses twice) are bound by `let`s first."""
        e = _lowered(e)
        self._bind_deps([e])
        return E.to_json(e, self.names)

    def _bind_deps(self, roots: list) -> None:
        """Bind the hoistable subexpressions of roots that an instruction computed (or that
        they share), in program order: a metadata load that fails stops the role where
        the kernel's first failing load would."""
        order, shared = _hoistable(roots, self.names)
        keys: dict = {}
        for x in order:  # children first: an unmarked node runs after its last child
            o = self.k.origin.get(x)
            keys[x] = (
                o[1] if o else max((keys.get(c, -1) for c in E.children(x)), default=-1)
            )
        for x in sorted(order, key=keys.__getitem__):
            if (x in self.k.origin or x in shared) and _bindable(x):
                self._let(x, self.k.origin[x][0] if x in self.k.origin else self.loc())

    def _let(self, x, loc: str) -> str:
        if E.unsafe(x):  # it may stop the run: not ahead of a pending event
            self.flush()
        name = self.fresh("v")
        self.stmts.append(
            {
                "op": "let",
                "var": name,
                "expr": E.to_json(x, self.names, name),
                "loc": loc,
            }
        )
        self.names[x] = name
        self.scopes[-1].append(x)
        return name

    def bind(self, e, hint: str = "v"):
        """Bind a thread-uniform, non-trivial expression now; its variable."""
        if (
            e[0] in ("const", "var", "pid", "nprog")
            or not _bindable(e)
            or E.has_narrow(e)
        ):
            return e
        self._bind_deps([e])
        if e not in self.names:
            self._let(e, self.k.origin[e][0] if e in self.k.origin else self.loc())
        return E.var(self.names[e], E.width(e))

    def push_scope(self) -> int:
        self.scopes.append([])
        return len(self.created)

    def pop_scope(self, mark: int) -> set:
        """Forget the bindings of the innermost scope; the names made in it."""
        for x in self.scopes.pop():
            self.names.pop(x, None)
        return set(self.created[mark:])

    def require(self, cond, msg: str) -> None:
        """A condition the instruction's meaning depends on, checked by a KIR assert (per
        issuing thread when it varies across threads or holds under a mask)."""
        u = self.uniform(cond)
        if u is not None and E.is_const(u, 1):
            return
        if u is not None and E.is_const(self.mask, 1):
            self.emit({"op": "assert", "cond": self.json(u), "msg": msg})
            return
        loc = self.loc()
        self.thread_event(
            [cond],
            lambda v, j: {"op": "assert", "cond": j(v[0]), "msg": msg, "loc": loc},
            tag="assert",
        )

    def guard(self, x, safe: int):
        """`x` where the current mask holds, `safe` elsewhere (operands of partial ops)."""
        if E.is_const(self.mask, 1):
            return x
        return E.select(self.mask, x, E.const(safe, E.width(x)))

    # --------------------------------------------------------------- threads
    def threads(self) -> list[tuple[int, int]]:
        return [(w, ln) for w in range(self.w0, self.w0 + self.nw) for ln in range(32)]

    def thread_subst(self, w, lane, fv=()) -> dict:
        m = {"%warp": E.const(w, 32), "%lane": E.const(lane, 32)}
        m.update({v: E.const(int(lane == 0), 1) for v in fv if v.startswith("%elect")})
        return m

    def instances(self, e) -> list:
        """e for every thread of the role, in threads() order (folded). The elected lane
        is instantiated as lane 0: uniform() and per_warp() do not judge a value that
        depends on it, and _check_elected keeps that choice invisible to events."""
        r = self._inst.get(e)
        if r is None:
            fv = E.free_vars(e)
            if not any(map(E.is_thread_var, fv)):
                r = [e] * (32 * self.nw)
            else:
                elects = {v: None for v in fv if v.startswith("%elect")}
                r = []
                for w in range(self.w0, self.w0 + self.nw):
                    ew = E.subst(e, {"%warp": E.const(w, 32)}) if "%warp" in fv else e
                    for lane in range(32):
                        m = {"%lane": E.const(lane, 32)}
                        m.update({v: E.const(int(lane == 0), 1) for v in elects})
                        r.append(E.subst(ew, m))
            self._inst[e] = r
        return r

    def uniform(self, e):
        """The expression if it has one value for every thread of the role, else None."""
        if not E.thread_varying(e):
            return e
        if _has_elect(e):  # its instances assume lane 0 is elected (r5 ? lane : 0)
            return None
        inst = self.instances(e)
        return inst[0] if all(x is inst[0] for x in inst) else None

    def split_per_warp(self, e) -> None:
        """Raise SplitWarps on the runs of warps with one value of `e` if e is uniform
        within each warp but not across the role (and the policy splits); else return."""
        if not self.k.split:
            return
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
        if _has_elect(e):  # as in uniform()
            return None
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

    def lane_table(self, values: list, hint: str):
        """table over lanes (32 values) or threads (32 * nw values); entries bound."""
        bound: dict = {}
        entries = tuple(
            bound[v] if v in bound else bound.setdefault(v, self.bind(v, hint))
            for v in values
        )
        return E.table(self.thread_key(len(values) == 32), entries)

    def scalarize(self, v):
        """A lane-varying integer computed from per-lane memory reads or shuffles (warp-
        cooperative code: a thread-varying load or table in it) becomes a table of 32
        bound per-lane values, so that later uses do not copy its expression. (A uniform
        load in a lane-varying value is just a uniform part, bound where it is used.)
        Under a mask a value that may fail (its load) has the entry select(mask, value,
        0): the bound `let`s run unconditionally, and a lane that does not run the
        instruction must not evaluate its load (KIR evaluates only the chosen arm); such
        an entry is never read."""
        if isinstance(v, Data) or not E.thread_varying(v) or not _varying_data(v):
            return v
        m = v
        if not E.is_const(self.mask, 1) and E.unsafe(v):
            m = E.select(self.mask, v, E.const(0, E.width(v)))
        if _has_elect(m) or not self.lane_only(m):
            return v
        return self.lane_table(self.instances(m)[:32], "l")

    # ------------------------------------------------------------- slots
    # A slot stores one register in KIR variables: ("u", name, w, base) for a uniform
    # value, ("f", names, w, base) for a family indexed by lane (32) or thread. `base` is
    # the address base and constant part kept outside the variables (mem.slot_base; None
    # for a plain integer): the slot holds value - base.

    def new_slot(self, e, kind: str, hint: str, base=None):
        if kind == "u":
            return ("u", self.fresh(hint), E.width(e), base)
        name = self.fresh(hint)
        n = 32 if kind == "lane" else 32 * self.nw
        names = tuple(f"{name}_{i}" for i in range(n))
        self.created.extend(names)
        return ("f", names, E.width(e), base)

    def assign(self, slot, e) -> list:
        if slot[3] is not None:
            e = self.k.mem.rebase(self, e, slot[3])
        if slot[0] == "u":
            u = self.uniform(e)
            if u is None:
                self.fail(
                    self.cur, "thread-varying value assigned to a uniform variable"
                )
            return self._lets([(slot[1], u)])
        names = slot[1]
        if _has_elect(e):
            self.fail(
                self.cur,
                "a value depending on which lane elect.sync picks is carried in a "
                "per-thread variable",
            )
        inst = self.instances(e)
        if len(names) == 32 and not self.lane_only(e):
            self.fail(self.cur, "warp-dependent value assigned to a per-lane variable")
        return self._lets(list(zip(names, inst)))

    def _lets(self, pairs: list) -> list:
        """`let name = x` statements, with the lets of their subexpressions first. The
        caller places them (possibly inside a branch), so those bindings are forgotten
        after."""
        saved, self.stmts = self.stmts, []
        mark = self.push_scope()
        out = []
        for n, x in pairs:
            j = self.json(x)
            out += self.stmts
            self.stmts = []
            out.append({"op": "let", "var": n, "expr": j, "loc": self.loc()})
        self.pop_scope(mark)
        self.stmts = saved
        return out

    def read(self, slot):
        if slot[0] == "u":
            v = E.var(slot[1], slot[2])
        else:
            v = E.table(
                self.thread_key(len(slot[1]) == 32),
                tuple(E.var(n, slot[2]) for n in slot[1]),
            )
        return v if slot[3] is None else E.bin("add", slot[3], v)

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

    def snapshot(self):
        self.flush()
        return (len(self.stmts), len(self.barriers), self.stmts, dict(self.names),
                [list(s) for s in self.scopes], len(self.created), dict(self.env),
                dict(self.slot_facts), self.work_events,
                len(self.work_scopes), len(self.regions))  # fmt: skip

    def restore(self, snap) -> None:
        """Back to a snapshot (a loop translated again): the statement lists, bindings,
        environment and the stacks a translation interrupted by _Widen left open."""
        self.pending = None
        n, nb, stmts, names, scopes, nc, env, facts, works, nw, nr = snap
        del self.work_scopes[nw:]
        del self.regions[nr:]
        self.stmts = stmts
        del self.stmts[n:]
        del self.barriers[nb:]
        self.names, self.scopes, self.env = names, scopes, env
        self.slot_facts, self.work_events = facts, works
        del self.created[nc:]

    # ------------------------------------------------------------ regions
    def run(self, seq: list) -> None:
        """Translate a region sequence of regions.py."""
        self.work_scopes.append({})
        for node in seq:
            if isinstance(node, R.Block):
                for i in node.instrs:
                    self.step(i)
            elif isinstance(node, R.If) and self.loops and node is self.loops[-1][5]:
                self.flush()  # a counted loop's latch test: the `for` does it
                self.end_iteration(False)
            elif isinstance(node, R.If):
                self.flush()
                self.run_if(node)
            elif isinstance(node, R.Loop):
                self.flush()
                self.run_loop(node)
            elif isinstance(node, R.Wait):
                self.cur = node.instr
                self.k.mem.wait(self, node.instr)
            elif node in (R.CONTINUE, R.BREAK):
                self.flush()
                self.end_iteration(node == R.BREAK)
        self.work_scopes.pop()

    def step(self, i) -> None:
        self.cur = i
        try:
            cls, why = ops.classify(i)
        except ops.UnknownInstruction as e:
            self.fail(i, str(e), kind="error")
        if cls == "refuse":
            self.fail(i, why)
        if cls == "control":
            return
        if i.guard is None:
            self.execute(i, cls)
        else:
            self.predicated(i, cls)
        sites = self.k.work.get(id(i))
        if sites:
            self.work(i, sites)

    def predicated(self, i, cls: str) -> None:
        reg, neg = i.guard
        p = self.env.get(reg, Data(f"undefined {reg}"))
        if isinstance(p, Data):
            if cls in ("value", "drop"):  # a data value, used by no root so far
                for r in R.dests(i):
                    self.env[r] = Data(
                        f"{i.opcode} under a data predicate at {self.loc(i)}"
                    )
                return
            self.as_int(p, "the predicate of an event")
        if neg:
            p = E.not1(p)
        u = self.uniform(p)
        if u is not None and E.is_const(u):
            if u[2]:
                self.execute(i, cls)
            return
        old = {r: self.env.get(r) for r in R.dests(i)}
        saved = self.mask
        self.mask = E.and1(saved, p)
        self.execute(i, cls)
        self.mask = saved
        for r, x in old.items():
            self.env[r] = self._select(p, self.env.get(r), x, r)

    def _select(self, c, x, y, reg: str):
        if x is y:
            return x
        if isinstance(x, Data) or isinstance(y, Data) or x is None or y is None:
            return Data(f"{reg} merged at {self.loc()} with a value outside the model")
        return E.select(c, x, y)

    def _cond(self, node: R.If):
        assert node.branch.guard is not None, "regions.py: an If is a guarded bra"
        reg, neg = node.branch.guard
        self.cur = node.branch
        c = self.as_int(
            self.env.get(reg, Data(f"undefined {reg}")), "a branch condition"
        )
        return E.not1(c) if neg else c

    def run_if(self, node: R.If) -> None:
        c = self._cond(node)
        cu = self.uniform(c)
        if cu is not None and E.is_const(cu):
            self.run(node.then if cu[2] else node.else_)
            return
        if cu is None:
            self.split_per_warp(c)
            self._masked_if(node, c)
        else:
            self._uniform_if(node, cu)

    def _live(self, node: R.If):
        """The registers to merge at the join of an If. A join CONTINUE (BREAK) is the
        innermost loop's back (exit) edge, which reads its carried registers."""
        if node.join in (R.CONTINUE, R.BREAK):
            return self.loops[-1][0]
        return self.tree.live_in.get(node.join, frozenset()) if node.join else ()

    def _branch(self, node, side: int, seq: list, env: dict):
        saved, self.stmts = self.stmts, []
        self.env = dict(env)
        mark = self.push_scope()
        self.regions.append((id(node), side))
        self.run(seq)
        self.flush()
        self.regions.pop()
        local = self.pop_scope(mark)
        out, self.stmts = self.stmts, saved
        return out, self.env, local

    def _uniform_if(self, node: R.If, c) -> None:
        loc = self.loc(node.branch)
        cj = self.json(c)  # before the branches, which may reuse its lets
        env0 = self.env
        then_s, env_t, loc_t = self._branch(node, 1, node.then, env0)
        else_s, env_e, loc_e = self._branch(node, 0, node.else_, env0)
        self.env = dict(env0)
        self.cur = node.branch
        pre = []
        for r in sorted(self._live(node)):
            x, y = env_t.get(r), env_e.get(r)
            if x is y:
                self.env[r] = x
                continue
            if isinstance(x, Data) or isinstance(y, Data) or x is None or y is None:
                self.env[r] = Data(
                    f"{r} merged at {loc} with a value outside the model"
                )
                continue
            if E.free_vars(x).isdisjoint(loc_t) and E.free_vars(y).isdisjoint(loc_e):
                self.env[r] = E.select(c, x, y)
                continue
            base = self.k.mem.slot_base(self, x)
            slot = self.new_slot(
                x, self.join(self.kind_of(x), self.kind_of(y)), "r", base
            )
            # Declared before the `if`, where x may read names its branch binds (or a
            # load only the branch may run): any value of the slot's width does.
            pre += self.assign(slot, E.const(0, E.width(x)) if base is None else base)
            then_s += self.assign(slot, x)
            else_s += self.assign(slot, y)
            self.env[r] = self.read(slot)
        self.stmts += pre
        if then_s or else_s:
            self.stmts.append(
                {"op": "if", "cond": cj, "then": then_s, "else": else_s, "loc": loc}
            )

    def _masked_if(self, node: R.If, c) -> None:
        env0, saved = self.env, self.mask
        self.env = dict(env0)
        self.mask = E.and1(saved, c)
        self.regions.append((id(node), 1))
        self.run(node.then)
        env_t = self.env
        self.env = dict(env0)
        self.mask = E.and1(saved, E.not1(c))
        self.regions[-1] = (id(node), 0)
        self.run(node.else_)
        self.regions.pop()
        env_e = self.env
        self.mask = saved
        self.env = dict(env0)
        self.cur = node.branch
        for r in self._live(node):
            self.env[r] = self._select(c, env_t.get(r), env_e.get(r), r)

    # ------------------------------------------------------------- loops
    def run_loop(self, node: R.Loop) -> None:
        """A loop as KIR `for` when it is an exact signed-i32 counted loop (_counted),
        else as `while` on an exit flag with the policy's max_iters."""
        if not E.is_const(self.mask, 1):
            self.fail(
                self.cur, "a loop under a thread-varying condition (divergent loop)"
            )
        head = self.k.mem.first_instr(node.header)
        live = self.tree.live_in.get(node.header, frozenset()) | self.tree.live_in.get(
            node.exit, frozenset()
        )
        carried = sorted(r for r in self.tree.defs[node.header] if r in live)
        counted = self._counted(node, carried)
        if counted is not None:
            carried = [r for r in carried if r != counted[0]]
        kinds = [None] * len(carried)
        caps = [None] * len(carried)  # trailing zeros assumed for offset slots
        while True:
            snap = self.snapshot()
            try:
                body, slots, go = self._loop_once(node, carried, kinds, caps, counted)
                break
            except _Widen:
                self.restore(snap)
        if counted is None:
            cond = self.json(E.cmp("ne", self.read(go), E.const(0, 32)))
            self.stmts.append({"op": "while", "cond": cond, "body": body,
                               "max_iters": self.k.max_iters, "loc": self.loc(head)})  # fmt: skip
        else:
            reg, lb, ub, step, _latch, var = counted
            self.stmts.append({"op": "for", "var": var, "lb": _c32(lb), "ub": _c32(ub),
                               "step": _c32(step), "body": body, "loc": self.loc(head)})  # fmt: skip
            self.env[reg] = E.const(lb + -(-(ub - lb) // step) * step, 32)
        for r, s in zip(carried, slots):
            self.env[r] = (
                self.read(s)
                if s is not None
                else Data(f"{r} after the loop at {self.loc(head)}")
            )

    def _counted(self, node: R.Loop, carried: list):
        """(register, lb, ub, step, latch If, for variable) if the loop is the rotated
        counted loop `do { ... iv += step; } while (iv <s ub)` with every exit at that
        latch test, constant step > 0 and ub, and iv constant on entry with lb < ub and
        no i32 wrap of its last value: then `for iv in [lb, ub) step step` runs the same
        iterations (design §5.3 item 4). Else None (a `while`)."""
        latch = node.body[-1] if node.body else None
        if (
            not isinstance(latch, R.If)
            or (latch.then, latch.else_)
            not in (([R.CONTINUE], [R.BREAK]), ([R.BREAK], [R.CONTINUE]))
            or _exits(node.body[:-1])
        ):
            return None
        assert latch.branch.guard is not None, "regions.py: an If is a guarded bra"
        reg, neg = latch.branch.guard
        cont_is_p = (latch.then == [R.CONTINUE]) != neg
        instrs = self.k.mem.cfg.blocks[latch.block].instrs
        defs = [(k, i) for k, i in enumerate(instrs) if reg in R.dests(i)]
        if not defs or defs[-1][1].guard is not None:
            return None
        k_setp, setp = defs[-1]
        want = "setp.lt.s32" if cont_is_p else "setp.ge.s32"
        if (
            setp.opcode != want
            or setp.operands[1][0] != "reg"
            or setp.operands[2][0] != "imm"
        ):
            return None
        iv, ub = setp.operands[1][1], setp.operands[2][1]
        incs = [
            i
            for b in node.blocks
            for i in self.k.mem.cfg.blocks[b].instrs
            if iv in R.dests(i)
        ]
        if iv not in carried or len(incs) != 1:
            return None
        add = incs[0]
        if (add.opcode not in ("add.s32", "add.u32") or add.guard is not None
                or not any(add is i for i in instrs[:k_setp]) or add.operands[1] != ("reg", iv)
                or add.operands[2][0] != "imm" or add.operands[2][1] <= 0):  # fmt: skip
            return None
        v, step = self.env.get(iv), add.operands[2][1]
        lb = self.uniform(v) if isinstance(v, tuple) else None
        if lb is None or not E.is_const(lb) or not lb[2] < ub:
            return None
        lb = lb[2]
        if lb + -(-(ub - lb) // step) * step > 2**31 - 1:
            return None
        return iv, lb, ub, step, latch, self.fresh("i")

    def _loop_once(self, node, carried: list, kinds: list, caps: list, counted=None):
        slots, env = [], dict(self.env)
        if counted is not None:
            env[counted[0]] = E.var(counted[5], 32)
        for r, kind, cap in zip(carried, kinds, caps):
            v = self.env.get(r)
            if isinstance(v, Data) or v is None:
                slots.append(None)
                env[r] = Data(f"{r} carried by the loop at {self.loc()}")
                continue
            base = self.k.mem.slot_base(self, v)
            s = self.new_slot(v, kind or self.kind_of(v), "c", base)
            self.stmts += self.assign(s, v)
            if s[0] == "u" and base is not None:
                tz = E.tz(self.k.mem.rebase(self, v, base), self.slot_facts)
                self.slot_facts[s[1]] = tz if cap is None else min(tz, cap)
            slots.append(s)
            env[r] = self.read(s)
        go = None
        if counted is None:
            go = self.new_slot(E.const(1, 32), "u", "go")
            self.stmts += self.assign(go, E.const(1, 32))
        saved, self.stmts = self.stmts, []
        self.env = env
        mark = self.push_scope()
        self.loops.append(
            [carried, slots, go, kinds, caps, counted[4] if counted else None]
        )
        try:
            self.run(node.body)
        finally:
            self.loops.pop()
            self.pop_scope(mark)
        body, self.stmts = self.stmts, saved
        return body, slots, go

    def end_iteration(self, brk: bool) -> None:
        """A back edge (or, brk, an exit edge) of the innermost loop: the carried slots
        take the registers' values, in parallel; an exit also clears the loop's flag. A
        slot holding an offset from a base keeps the trailing zeros assumed for it (the
        known-bits fixed point of design §5.3 item 5), else the loop is translated again
        with fewer. Under a thread-varying mask (a masked If's arm, however deeply nested
        in uniform ones) the edge is taken by some threads only: refused."""
        if not E.is_const(self.mask, 1):
            self.fail(
                self.cur,
                "a loop exits or continues under a thread-varying condition "
                "(divergent loop)",
            )
        carried, slots, go, kinds, caps, _latch = self.loops[-1]
        values = [self.env.get(r) for r in carried]
        widened = False
        for k, (s, v) in enumerate(zip(slots, values)):
            if s is None:
                continue
            if isinstance(v, Data) or v is None:
                self.fail(self.cur, f"loop-carried register {carried[k]} becomes data")
            kind = self.join(self.slot_kind(s), self.kind_of(v))
            if kind != self.slot_kind(s):
                kinds[k], widened = kind, True
            if s[0] == "u" and s[3] is not None:
                have = E.tz(self.k.mem.rebase(self, v, s[3]), self.slot_facts)
                if have < self.slot_facts.get(s[1], 0):
                    caps[k], widened = have, True
        if widened:
            raise _Widen()
        self.stmts += self._parallel_assign(slots, values)
        if brk and go is not None:
            self.stmts += self.assign(go, E.const(0, 32))

    def _parallel_assign(self, slots: list, values: list) -> list:
        """slot := value for all pairs, as if simultaneous (temporaries where a value
        reads a slot assigned before it, or another entry of its own family)."""
        pairs = [(s, v) for s, v in zip(slots, values) if s is not None]
        written = set()
        temps, final = [], []
        for sl, x in pairs:
            names = {sl[1]} if sl[0] == "u" else set(sl[1])
            reads = E.free_vars(x)
            cross = sl[0] == "f" and any(
                (E.free_vars(y) & names) - {n} for n, y in zip(sl[1], self.instances(x))
            )
            if reads & written or cross:
                tmp = self.new_slot(x, self.slot_kind(sl), "tmp", sl[3])
                temps += self.assign(tmp, x)
                x = self.read(tmp)
            final += self.assign(sl, x)
            written |= names
        return temps + final

    # ---------------------------------------------------------------- events
    def split_mask(self, mask=None):
        """(thread-uniform part, thread-varying part) of the current mask (or `mask`)."""
        uni, var = E.TRUE, E.TRUE
        for c in _conjuncts(self.mask if mask is None else mask):
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

    def _emit_under(self, uni, stmt: dict) -> None:
        self.flush()
        if not E.is_const(uni, 1):
            stmt = {"op": "if", "cond": self.json(uni), "then": [stmt], "else": [],
                    "loc": self.loc()}  # fmt: skip
        self.stmts.append(stmt)

    def stmts_under(self, uni, stmt: dict) -> None:
        """Append stmt, inside `if uni` unless uni is true (pending events flushed)."""
        if not E.is_const(uni, 1):
            stmt = {"op": "if", "cond": self.json(uni), "then": [stmt], "else": [],
                    "loc": stmt["loc"]}  # fmt: skip
        self.stmts.append(stmt)

    def _thread_loop(self, items: list) -> dict:
        """`for t in [0, 32*nw): ... if cond_i(t): build_i(...) ...` for the per-thread
        events items [(cond, operands, build, loc)] in order, with thread t's values: the
        thread-major order of consecutive memory events (and asserts), which no spec
        sees. Uniform parts are bound before the loop; the subexpressions the body uses
        twice become lets at its top, as frontend/cute/interp.py's thread loops (address
        arithmetic is evaluated on every thread, as it executes on every thread), but a
        node that may fail stays inside the select arm that guards it (_hoistable)."""
        for cond, operands, _b, _l in items:
            self._bind_deps([_lowered(x) for x in operands] + [_lowered(cond)])
        t = self.fresh("t")
        tv = E.var(t, 32)
        lane = E.bin("urem", tv, E.const(32, 32))
        m = {
            "%lane": lane,
            "%warp": E.bin(
                "add", E.bin("udiv", tv, E.const(32, 32)), E.const(self.w0, 32)
            ),
        }
        cache: dict = {}
        parts = []
        for cond, operands, build, loc in items:
            fv = set().union(*map(E.free_vars, operands + [cond]))
            m.update(
                {
                    v: E.cmp("eq", lane, E.const(0, 32))
                    for v in fv
                    if v.startswith("%elect")
                }
            )
            vals = [_lowered(E.subst(_lowered(e), m, cache)) for e in operands]
            parts.append(
                (_lowered(E.subst(_lowered(cond), m, cache)), vals, build, loc)
            )
        names = dict(self.names)
        order, shared = _hoistable(
            [x for c, vals, _b, _l in parts for x in [c, *vals]], names
        )
        body = []
        for x in order:
            if x in shared:
                n = f"{t}_{len(names) - len(self.names)}"
                body.append({"op": "let", "var": n, "expr": E.to_json(x, names, n),
                             "loc": items[0][3]})  # fmt: skip
                names[x] = n
        for c, vals, build, loc in parts:
            inner = build(vals, lambda e: E.to_json(_lowered(e), names))
            if not E.is_const(c, 1):
                inner = {"op": "if", "cond": E.to_json(c, names), "then": [inner], "else": [],
                         "loc": loc}  # fmt: skip
            body.append(inner)
        return {"op": "for", "var": t, "lb": _c32(0), "ub": _c32(32 * self.nw), "step": _c32(1),
                "body": body, "loc": items[0][3]}  # fmt: skip

    def mem_event(self, kind: str, tensor: str, off, n: int) -> None:
        """A gload/gstore/gatomic of n consecutive elements of `tensor` at element offset
        `off` (i64) by every issuing thread. Under the policy `vectorize` (Triton: the
        TTGIR frontend's one event per tensor operation) the issuing threads are an axis
        of one event: thread t's values come from its expressions over %t = 32 WARP +
        LANE (the elected lane of an elect symbol is lane 0, which _check_elected makes
        invisible); and consecutive accesses of one instruction family (same kind,
        tensor, width, uniform condition and loc, nothing emitted in between but safe
        lets) whose offsets and conditions differ only in constants of one arithmetic
        progression are one event over another axis (an unrolled tl.load is many PTX
        loads). Without `vectorize` (CuTe: the CuTe frontend's events per issuing
        thread), or with a value no TExpr can carry (a metadata load or a division that
        may fail on another thread, a table), the event is issued per thread."""
        if not self.k.vectorize:
            self._per_thread_access(kind, tensor, off, n)
            return
        uni, var = self.split_mask()
        self._check_elected(var, [off])
        act = self.active(var)
        if act == []:
            return
        if act is not None and len(act) == 1:
            w, ln = self.threads()[act[0]]
            m = self.thread_subst(w, ln, E.free_vars(off) | E.free_vars(var))
            toff, tvar, threads = E.subst(off, m), E.TRUE, 1
        else:
            tv = E.var("%t", 32)
            lane = E.bin("urem", tv, E.const(32, 32))
            m = {"%lane": lane, "%warp": E.bin("udiv", tv, E.const(32, 32))}
            m.update({v: E.cmp("eq", lane, E.const(0, 32))
                      for v in E.free_vars(off) | E.free_vars(var) if v.startswith("%elect")})  # fmt: skip
            toff, tvar, threads = E.subst(off, m), E.subst(var, m), 32 * self.nw
        toff, tvar = _lowered(toff), _lowered(tvar)
        if threads > 1 and not (_tensor_safe(toff) and _tensor_safe(tvar)):
            self._per_thread_access(kind, tensor, off, n)
            return
        if (
            threads > 1
            and kind == "gload"
            and "%t" not in E.free_vars(toff)
            and act is not None
        ):
            tvar, threads = (
                E.TRUE,
                1,
            )  # every issuing thread reads the same elements: once
        self._bind_deps([toff, tvar])  # in program order, as in thread_event
        key = (kind, tensor, n, uni, self.loc(), threads, self.w0)
        p = self.pending
        if p is not None and not p.get("loop") and p["key"] == key \
                and _antiunify(p["offs"] + [toff]) is not None \
                and _antiunify(p["conds"] + [tvar]) is not None:  # fmt: skip
            p["offs"].append(toff)
            p["conds"].append(tvar)
            return
        self.flush()
        self.pending = {"key": key, "offs": [toff], "conds": [tvar]}

    def _per_thread_access(self, kind: str, tensor: str, off, n: int) -> None:
        loc = self.loc()
        self.thread_event([off], lambda v, j: {
            "op": "event", "ev": kind, "tensor": tensor, "mask": None,
            "offsets": _scalar_offsets(j(v[0]), n), "loc": loc})  # fmt: skip

    def flush(self) -> None:
        """Emit what is pending: memory events (mem_event) as one event, or consecutive
        per-thread events (thread_event) as one thread loop."""
        p, self.pending = self.pending, None
        if p is None:
            return
        if p.get("loop"):
            self.stmts_under(p["key"][0], self._thread_loop(p["items"]))
            return
        kind, tensor, n, uni, loc, threads, _w0 = p["key"]
        sizes = {"k": len(p["offs"]), "t": threads, "j": n}
        want = tuple(a for a in _AXES if sizes[a] > 1)
        off = _vectorize(self, _antiunify(p["offs"]), sizes, want)
        if n > 1:
            j = {"t": "range", "shape": [n], "w": 64, "start": 0, "end": n}
            off = {"t": "bin", "shape": off["shape"], "w": 64, "o": "add", "a": off,
                   "b": _lift(j, ("j",), want, sizes)}  # fmt: skip
        cond = _antiunify(p["conds"])
        mask = None if cond is E.TRUE else _vectorize(self, cond, sizes, want)
        self.pending = None
        stmt = {"op": "event", "ev": kind, "tensor": tensor, "mask": mask, "offsets": off,
                "loc": loc}  # fmt: skip
        if not E.is_const(uni, 1):
            stmt = {
                "op": "if",
                "cond": self.json(uni),
                "then": [stmt],
                "else": [],
                "loc": loc,
            }
        self.stmts.append(stmt)

    def thread_event(self, operands: list, build, tag: str = "event") -> None:
        """Emit an event once per issuing thread of the role. `operands` are the
        (possibly thread-varying) expressions it uses; build(values, to_json) returns the
        KIR statement for one thread's values. Consecutive ones of one `tag` and uniform
        condition share a thread loop (flush)."""
        uni, var = self.split_mask()
        self._check_elected(var, operands)
        act = self.active(var)
        if act == []:
            return
        if act is not None and len(act) == 1:
            self._emit_under(uni, self._thread_stmt(act[0], operands, build))
            return
        full = act is not None and len(act) == 32 * self.nw
        # Bind the uniform parts here, in program order: a part that may fail (a metadata
        # load) then flushes the earlier events first (_let), cutting the thread loop.
        self._bind_deps([_lowered(x) for x in operands] + [_lowered(var)])
        item = (E.TRUE if full else var, operands, build, self.loc())
        p = self.pending
        if p is not None and p.get("loop") and p["key"] == (uni, tag):
            p["items"].append(item)
            return
        self.flush()
        self.pending = {"loop": True, "key": (uni, tag), "items": [item]}

    def _check_elected(self, var, operands: list) -> None:
        """Under elect.sync the elected lane is unspecified (ISA elect.sync), so an event
        must not depend on which lane it is (H1): an elect symbol may appear only as one
        conjunct of the mask, itself or negated; every other conjunct must be the same in
        every lane of a warp, and the operands too. An operand never depends on the
        elected lane, under elect.sync or not (its threads' values would be lane 0's)."""
        if any(map(_has_elect, operands)):
            self.fail(
                self.cur, "an event operand depends on which lane elect.sync picks"
            )
        if not _has_elect(var):
            return
        elect = [c for c in _conjuncts(var) if _has_elect(c)]
        if len(elect) != 1 or not _bare_elect(elect[0]):
            self.fail(
                self.cur,
                "an event under elect.sync is also guarded by another condition on the "
                "elected lane",
            )
        for c in _conjuncts(var):
            if c is not elect[0] and self.per_warp(c) is None:
                self.fail(
                    self.cur,
                    "an event under elect.sync is also guarded by a condition that "
                    "depends on the lane",
                )
        for e in operands:
            if self.per_warp(e) is None:
                self.fail(
                    self.cur,
                    "an operand under elect.sync depends on which lane is elected",
                )

    def _thread_stmt(self, i: int, operands: list, build) -> dict:
        """build() with the operand values of thread i (threads() order) of the role."""
        fv = set().union(*map(E.free_vars, operands)) if operands else set()
        m = self.thread_subst(*self.threads()[i], fv)
        return build([E.subst(e, m) for e in operands], self.json)

    def full_warps(self, act: list, what: str) -> int:
        """The number of warps among the threads `act`, each of which must be fully
        active: `what` runs with the full member mask, undefined in PTX when a lane of the
        warp does not execute it (H3)."""
        lanes: dict = {}
        for i in act:
            lanes[i // 32] = lanes.get(i // 32, 0) + 1
        partial = sorted(self.w0 + w for w, n in lanes.items() if n != 32)
        if partial:
            self.fail(
                self.cur,
                f"{what} with only part of warp(s) {partial} active (undefined in PTX)",
            )
        return len(lanes)

    def collective(self, what: str) -> None:
        """A full-mask warp collective: every executing warp must be fully active (H3).
        Under a lane condition that depends on runtime values, a KIR assert per warp
        checks that its lanes agree on the condition (all active or none)."""
        uni, var = self.split_mask()
        if _has_elect(var):
            self.fail(self.cur, f"{what} under elect.sync (one lane of the warp)")
        act = self.active(var)
        if act is not None:
            self.full_warps(act, what)
            return
        key = (id(self.stmts), var, uni)
        if self._uniform_checked.get(key) is self.stmts:
            return  # an earlier check in this statement list covers it
        self._uniform_checked[key] = self.stmts  # kept alive: its id is not reused
        inst = self.instances(var)
        ok = E.TRUE
        for w in range(self.nw):
            lanes = list(dict.fromkeys(inst[32 * w : 32 * w + 32]))
            ok = E.and1(
                ok, E.bin("or", _fold("and", lanes), E.not1(_fold("or", lanes)))
            )
        if not E.is_const(ok, 1):
            msg = f"{what} with the full member mask: all lanes of a warp or none"
            self._emit_under(uni, {"op": "assert", "cond": self.json(ok), "msg": msg,
                                   "loc": self.loc()})  # fmt: skip

    def aligned(self, what: str) -> None:
        """A .sync.aligned instruction: every thread of a warp executes it or none
        (ISA: .aligned), so the mask must be the same in all lanes of each warp."""
        _uni, var = self.split_mask()
        if not E.is_const(var, 1) and (_has_elect(var) or self.per_warp(var) is None):
            self.fail(
                self.cur,
                f"{what} under a condition that differs between the lanes of a warp",
            )

    def barrier_event(self, ev: str, sym: str, addr, mode: str, **fields) -> None:
        """An mbarrier event at byte offset `addr` of shared symbol `sym`; bar/idx are
        resolved once every role is translated (mem.Memory.barriers): the offset's
        constant part picks the array, the rest divided by 8 (exactly, else refused) is
        bound here. Modes: "once" (one event if any thread issues it: bar_init, wait),
        "count" (one arrive counting the issuing threads), "thread" (once per issuing
        thread)."""
        a = self.uniform(addr)
        if a is None:
            self.split_per_warp(addr)
            self.fail(
                self.cur, "mbarrier address varies across the threads of the role"
            )
        stmt: dict = {"op": "event", "ev": ev, "bar": None, "idx": None}
        for key, v in fields.items():
            u = self.uniform(v)
            if u is None:
                self.split_per_warp(v)
                self.fail(
                    self.cur,
                    f"mbarrier event operand '{key}' varies across the threads of the role",
                )
            stmt[key] = self.json(u)
        stmt["loc"] = self.loc()
        issued = self._issue(stmt, mode)
        if issued is None:
            return
        c, rest = self.k.mem.split_offset(a)
        q = E.div_exact(rest, 8, self.slot_facts)
        if q is None:
            self.fail(
                self.cur,
                "mbarrier offset not provably a multiple of 8 bytes from its array",
            )
        if ev == "bar_init" and not E.is_const(rest, 0):
            self.fail(
                self.cur,
                "mbarrier initialized at an address that is not a constant of the kernel",
            )
        restj = None if E.is_const(q, 0) else self.json(E.resize(q, 32))
        init = c if ev == "bar_init" else None
        self.barriers.append(BarrierUse(
            stmt, sym, c, restj, E.interval(a), issued[2], stmt["loc"], init,
            self.k.srclocs.get(id(self.cur)) if init is not None else None,
            self.k.mem.init_key(self) if init is not None else ()))  # fmt: skip
        self._emit_under(issued[0], issued[1])

    def _issue(self, stmt: dict, mode: str):
        """(uniform condition, statement, issuer count in mode "count") for an event
        issued under the current mask, or None if no thread issues it."""
        uni, var = self.split_mask()
        self._check_elected(var, [])
        act = self.active(var)
        if act == []:
            return None
        issuers = None
        if act is None:
            if mode == "once":
                self.fail(
                    self.cur,
                    "issued under a lane condition that depends on runtime values",
                )
            out = self._thread_loop([(var, [], lambda _v, _j: stmt, stmt["loc"])])
        elif mode == "once":
            out = stmt
        elif mode == "count":
            out, issuers = stmt, len(act)
        else:
            k = len(act)
            out = stmt if k == 1 else {
                "op": "for", "var": self.fresh("issuer"), "lb": _c32(0), "ub": _c32(k),
                "step": _c32(1), "body": [stmt], "loc": stmt["loc"]}  # fmt: skip
        return uni, out, issuers

    # ---------------------------------------------------------------- work
    def work(self, i, sites: list) -> None:
        """Work-item anchors resolved at instruction i (design §5.9, translate.py):
        "gate" emits one `work` event in a region once every declared field is defined
        in it, after the last definition (the TTGIR rule); "each" sets `work_<f>` and
        emits the event at every anchor (the CuTe rule; fields start at -1).

        An anchor counts only where it runs: under the mask and, for a load, the load's
        predicate (an assignment's register already holds the predicated merge). Its
        thread-varying part must hold on a known set of threads (empty: skipped); "each"
        sets the field and emits the event inside `if` the thread-uniform part, "gate"
        refuses one."""
        for site in sites:
            name = site["field"]
            cond = self.mask
            if site["kind"] == "load" and i.guard is not None:
                reg, neg = i.guard  # an int: predicated() refuses data for an event
                cond = E.and1(cond, E.not1(self.env[reg]) if neg else self.env[reg])
            uni, var = self.split_mask(cond)
            self._check_elected(var, [])
            act = self.active(var)
            if E.is_const(uni, 0) or act == []:
                continue  # no thread runs it here
            if act is None or (self.k.work_mode != "each" and not E.is_const(uni, 1)):
                self.fail(
                    i,
                    f"work field '{name}': the anchor runs under a condition "
                    + (
                        "that depends on runtime values in each lane"
                        if act is None
                        else "(gate rule: the work event must follow it unconditionally)"
                    ),
                )
            if site["kind"] == "novalue":  # defined here, no instruction holds it
                if name in self.k.work_per:
                    self.fail(
                        i,
                        f"work field '{name}' names segments but no instruction holds "
                        "its value (anchors.py)",
                        "error",
                    )
                v = E.const(-1, 64)
            else:
                v = self.k.mem.anchor_value(self, i, site)
            if v is None:
                continue  # a load of another tensor on the anchored line
            z = self.uniform(v)
            if z is None:
                self.fail(
                    i,
                    f"work field '{name}': the anchor's value varies across the role's threads",
                    "error",
                )
            if self.k.work_mode == "each":
                fields = {
                    f: {"e": "var", "w": 64, "name": f"work_{f}"}
                    for f in self.k.work_fields
                }
                event = {"op": "event", "ev": "work", "fields": fields}
                if E.is_const(uni, 1):
                    self.emit(
                        {
                            "op": "let",
                            "var": f"work_{name}",
                            "expr": self.json(E.resize(z, 64)),
                        }
                    )
                    self.emit(event)
                else:  # the value is computed inside the `if` too (it may be a load)
                    self.flush()
                    body = self._lets([(f"work_{name}", E.resize(z, 64))])
                    body.append({**event, "loc": self.loc()})
                    self.stmts.append({"op": "if", "cond": self.json(uni), "then": body,
                                       "else": [], "loc": self.loc()})  # fmt: skip
                self.work_events += 1
                continue
            scope = self.work_scopes[-1]
            scope[name] = z
            if set(scope) >= set(self.k.work_fields):
                fields = {f: self.json(scope[f]) for f in self.k.work_fields}
                self.emit({"op": "event", "ev": "work", "fields": fields})
                self.work_events += 1
                scope.clear()

    # ---------------------------------------------------------- instructions
    def execute(self, i, cls: str) -> None:
        mn = ops.mnemonic(i.opcode)
        if cls == "drop":
            for r in R.dests(i):
                self.env[r] = Data(f"{i.opcode} at {self.loc(i)}")
            return
        if cls == "event" or mn in ("ld.param", "cvta.param") or mn.startswith("mapa"):
            self.k.mem.execute(self, i, mn)
            return
        base = mn.split(".")[0]
        if base in ("elect", "shfl", "vote", "redux"):
            self._collective(i, mn)
            return
        dsts = _dest_regs(i)
        values = self._value(i, mn, len(dsts))
        if len(values) != len(dsts):
            self.fail(i, f"{len(values)} results for {len(dsts)} destinations", "error")
        for r, v in zip(dsts, values):
            if r is not None:
                self.write(r, v)

    def write(self, reg: str, v) -> None:
        """Assign a register at its declared width (a narrower or wider value is extended
        or truncated by the callers' types; here only zero-extended or truncated)."""
        if not isinstance(v, Data):
            w = _reg_width(self.k.fn.regs.get(reg, "b32"))
            if E.width(v) != w:
                v = E.resize(v, w, False)
            v = self.scalarize(v)
            if v[0] not in ("const", "var", "pid", "nprog"):
                self.k.origin.setdefault(
                    v, (self.loc(), self.cur.line if self.cur else 0)
                )
        self.env[reg] = v

    def operand(self, i, k: int, w: int | None = None, signed: bool = False):
        """Operand k of i as an integer expression of width w (extended by `signed` or
        truncated), or Data."""
        op = i.operands[k]
        kind = op[0]
        if kind == "imm":
            return E.const(op[1], w or 32)
        if kind in ("f32", "f64"):
            return Data(f"a float literal in {i.opcode} at {self.loc(i)}")
        v = self.operand_value(op, w)
        if isinstance(v, Data) or w is None or E.width(v) == w:
            return v
        return E.resize(v, w, signed)

    def operand_value(self, op, w=None):
        kind = op[0]
        if kind == "reg":
            v = self.env.get(op[1])
            return Data(f"undefined register {op[1]}") if v is None else v
        if kind == "imm":
            return E.const(op[1], w or 32)
        if kind == "sreg":
            return self.sreg(op[1])
        if kind == "sym":
            return self.k.mem.symbol(self, op[1], w or 32)
        if kind == "not":
            v = self.operand_value(op[1])
            return v if isinstance(v, Data) else E.not1(v)
        return Data(f"operand {op[0]} in {self.cur.opcode if self.cur else '?'}")

    def sreg(self, name: str):  # noqa: C901 (one case per special register)
        """Special registers (design §5.4.1): folded from the block and cluster dims."""
        base, _, axis = name.partition(".")
        dims = self.k.cluster
        if name == "%laneid":
            return E.LANE
        if name == "%cluster_nctarank":
            return E.const(dims[0] * dims[1] * dims[2], 32)
        if name == "%cluster_ctarank":
            rank, scale = E.const(0, 32), 1
            for a, n in enumerate(dims):
                c = E.bin("urem", E.pid(a), E.const(n, 32))
                rank = E.bin("add", rank, E.bin("mul", c, E.const(scale, 32)))
                scale *= n
            return rank
        if name == "%is_explicit_cluster":
            return E.const(int(tuple(dims) != (1, 1, 1)), 1)
        if name.startswith("%lanemask_"):
            eq = E.bin("shl", E.const(1, 32), E.LANE)
            lt = E.bin("sub", eq, E.const(1, 32))
            le = E.bin("or", lt, eq)
            ones = E.const(-1, 32)
            return {"eq": eq, "lt": lt, "le": le, "gt": E.bin("xor", le, ones),
                    "ge": E.bin("xor", lt, ones)}[name[len("%lanemask_"):]]  # fmt: skip
        if axis not in ("x", "y", "z"):  # the rest are vectors read one axis at a time
            return Data(f"special register {name}")
        ax = "xyz".index(axis)
        if name == "%tid.x":
            return E.bin("add", E.bin("mul", E.WARP, E.const(32, 32)), E.LANE)
        if base == "%tid":
            return E.const(0, 32)  # the block is (32k, 1, 1) (translate.py)
        if base == "%ntid":
            return E.const(self.k.block[ax], 32)
        if base == "%ctaid":
            return E.pid(ax)
        if base == "%nctaid":
            return E.nprog(ax)
        if base == "%clusterid":
            return E.bin("udiv", E.pid(ax), E.const(dims[ax], 32))
        if base == "%nclusterid":
            return E.bin("udiv", E.nprog(ax), E.const(dims[ax], 32))
        if base == "%cluster_ctaid":
            return E.bin("urem", E.pid(ax), E.const(dims[ax], 32))
        if base == "%cluster_nctaid":
            return E.const(dims[ax], 32)
        return Data(f"special register {name}")

    def _value(self, i, mn: str, ndst: int) -> list:  # noqa: C901 (operand widths per opcode)
        """Results of an integer value instruction (op table class "value"), one per
        destination register."""
        types = _types(i.opcode)
        base = mn.split(".")[0]
        if base == "mov":
            return self._mov(i, ndst)
        if base == "cvt":
            return [self._cvt(i, types)]
        if base == "cvta":
            return [self.operand(i, 1, 64)]
        if not types:
            self.fail(i, "integer instruction without a type", "error")
        kind, w = types[-1]
        signed = kind == "s"
        if base == "setp":
            return self._setp(i, types)
        if base == "selp":
            c = self.operand(i, 3, 1)
            a, b = self.operand(i, 1, w), self.operand(i, 2, w)
            if any(isinstance(x, Data) for x in (a, b, c)):
                return [Data(f"selp of data at {self.loc(i)}")]
            return [E.select(c, a, b)]
        n = len(i.operands)
        if mn in ("mul.wide", "mad.wide"):
            args = [self.operand(i, 1, w, signed), self.operand(i, 2, w, signed)]
            if n == 4:
                args.append(self.operand(i, 3, 2 * w, signed))
        elif base in ("shl", "shr"):
            args = [self.operand(i, 1, w), self.operand(i, 2, 32)]
        elif base in ("bfe", "bfi"):
            args = [self.operand(i, k, w if k <= (2 if base == "bfi" else 1) else 32)
                    for k in range(1, n)]  # fmt: skip
        elif base == "popc" or base == "clz" or base == "brev":
            args = [self.operand(i, 1, w)]
        else:
            args = [self.operand(i, k, w, signed) for k in range(1, n)]
        data = next((a for a in args if isinstance(a, Data)), None)
        if data is not None:
            return [Data(data.origin)] * len(_dest_regs(i))
        return [self._arith(i, mn, base, args, w, signed)]

    def _arith(self, i, mn, base, args, w, signed):  # noqa: C901 (one case per opcode)
        a = args[0]
        b = args[1] if len(args) > 1 else None
        if base in ("add", "sub", "and", "or", "xor") and ".cc" not in i.opcode:
            return E.bin(base, a, b)
        if base == "not":
            return E.bin("xor", a, E.const(-1, w)) if w > 1 else E.not1(a)
        if base == "neg":
            return E.bin("sub", E.const(0, w), a)
        if base == "abs":
            return E.select(
                E.cmp("slt", a, E.const(0, w)), E.bin("sub", E.const(0, w), a), a
            )
        if mn == "mul.lo":
            return E.bin("mul", a, b)
        if mn == "mul.wide":
            return E.mul_wide(a, b, signed)
        if mn == "mul.hi":
            if w != 32:
                self.fail(i, f"mul.hi on {w}-bit operands (only 32-bit is expanded)")
            return E.mul_hi(a, b, signed)
        if mn == "mad.lo":
            return E.bin("add", E.bin("mul", a, b), args[2])
        if mn == "mad.wide":
            return E.bin("add", E.mul_wide(a, b, signed), args[2])
        if mn == "mad.hi" and w == 32:
            return E.bin("add", E.mul_hi(a, b, signed), args[2])
        if base in ("div", "rem"):
            return E.bin(
                ("s" if signed else "u") + base, self.guard(a, 0), self.guard(b, 1)
            )
        if base in ("min", "max") and ".relu" not in i.opcode:
            return E.bin(("s" if signed else "u") + base, a, b)
        if base in ("shl", "shr"):
            amt = _shift_amount(b, w)
            op = "shl" if base == "shl" else ("ashr" if signed else "lshr")
            return E.bin(op, a, amt)
        if base == "popc":
            return E.popc(a)
        if base == "clz":
            return E.clz(a)
        if base == "brev":
            return E.brev(a)
        consts = [x[2] if E.is_const(x) else None for x in args]
        if base == "bfe":
            if None in consts[1:3]:
                self.fail(i, "bfe with a runtime position or length")
            return E.bfe(a, consts[1], consts[2], signed)
        if base == "bfi":
            if None in consts[2:4]:
                self.fail(i, "bfi with a runtime position or length")
            return E.bfi(a, b, consts[2], consts[3])
        if base == "prmt":
            if mn != "prmt" or consts[2] is None:
                self.fail(i, "prmt with a mode or a runtime selector")
            return E.prmt(a, b, consts[2])
        if base == "lop3":
            if len(args) != 4 or consts[3] is None:
                self.fail(i, "lop3 with a predicate output or a runtime truth table")
            return E.lop3(a, b, args[2], consts[3])
        if base == "shf":
            if consts[2] is None:
                self.fail(i, "shf with a runtime shift amount")
            return E.shf(a, b, consts[2], ".l." in i.opcode, ".wrap" in i.opcode)
        self.fail(i, f"no integer rule for {i.opcode}", "error")

    def _mov(self, i, ndst: int) -> list:
        dst, src = i.operands[0], i.operands[1]
        if dst[0] == "vec":  # unpack: mov.b32 {lo, hi}, x
            x = self.operand(i, 1)
            out, pos = [], 0
            for op in dst[1]:
                w = (
                    _reg_width(self.k.fn.regs.get(op[1], "b16"))
                    if op[0] == "reg"
                    else 16
                )
                out.append(x if isinstance(x, Data) else
                           E.trunc(E.bin("lshr", x, E.const(pos, E.width(x))), w))  # fmt: skip
                pos += w
            return out
        if src[0] == "vec":  # pack: mov.b64 x, {lo, hi}
            vals = [self.operand_value(op) for op in src[1]]
            if any(isinstance(v, Data) for v in vals):
                return [Data(f"{i.opcode} of data at {self.loc(i)}")]
            w = sum(E.width(v) for v in vals)
            out, pos = E.const(0, w), 0
            for v in vals:
                out = E.bin(
                    "or", out, E.bin("shl", E.resize(v, w, False), E.const(pos, w))
                )
                pos += E.width(v)
            return [out]
        types = _types(i.opcode)
        return [self.operand(i, 1, types[-1][1] if types else None)]

    def _cvt(self, i, types):
        if len(types) != 2 or any(k not in "bsu" for k, _ in types):
            return Data(f"{i.opcode} at {self.loc(i)}")
        if re.search(r"\.(sat|rn|rz|rm|rp|rni|rzi|rmi|rpi)\b", i.opcode):
            self.fail(i, "integer cvt with saturation or rounding")
        (_dk, dw), (sk, sw) = types
        a = self.operand(i, 1)
        if isinstance(a, Data):
            return a
        return E.cvt_int(a, dw, sw, sk == "s")

    def _setp(self, i, types):
        parts = i.opcode.split(".")
        cmpop = parts[1]
        boolop = (
            parts[2] if len(parts) > 3 and parts[2] in ("and", "or", "xor") else None
        )
        kind, w = types[-1]
        a, b = self.operand(i, 1, w), self.operand(i, 2, w)
        c = self.operand(i, 3, 1) if boolop else None
        ndst = len(_dest_regs(i))
        if any(isinstance(x, Data) for x in (a, b, c)):
            return [Data(f"setp of data at {self.loc(i)}")] * ndst
        op = _CMP.get((cmpop, kind == "s"))
        if op is None or (len(i.operands) > 3) != (boolop is not None):
            self.fail(i, f"setp form {i.opcode}", "error")
        p = E.cmp(op, a, b)
        out = [p, E.not1(p)]
        if boolop:
            out = [E.bin(boolop, x, c) for x in out]
        return out[:ndst]

    # ----------------------------------------------------------- collectives
    def _collective(self, i, mn: str) -> None:
        if mn == "elect.sync":
            self._elect(i)
        elif mn.startswith("shfl.sync"):
            self._shfl(i, mn.split(".")[2])
        elif mn.startswith("vote.sync"):
            self._vote(i, mn.split(".")[2])
        else:
            self.fail(i, f"{mn} (gap: not in M1)")

    def _full_mask(self, i, k: int, what: str) -> None:
        m = self.operand(i, k, 32)
        if isinstance(m, Data) or not E.is_const(m, -1):
            self.fail(i, f"{what} with a member mask other than the full warp")
        self.collective(what)

    def _elect(self, i) -> None:
        """elect.sync d|p, -1: p holds on one lane of each warp, which one the ISA does
        not say (an elect symbol of its own); d, that lane's id, is Data."""
        self._full_mask(i, 1, "elect.sync")
        sym = self.k.elects.setdefault(id(i), E.elect(len(self.k.elects)))
        dst = i.operands[0]
        regs = [dst[1], dst[2]] if dst[0] == "pair" else [dst]
        if regs[0][0] == "reg":
            self.env[regs[0][1]] = Data(f"elect.sync leader lane at {self.loc(i)}")
        if len(regs) > 1 and regs[1][0] == "reg":
            self.env[regs[1][1]] = sym

    def _shfl(self, i, kind: str) -> None:
        """shfl.sync d[|p], a, b, c, -1 on integers: lane l receives a at source lane j(l),
        or its own a when j(l) is outside its segment, p telling which (ISA shfl.sync;
        frontend/cute/ops.py _shfl)."""
        self._full_mask(i, 4, "shfl.sync")
        dst = i.operands[0]
        regs = [dst[1], dst[2]] if dst[0] == "pair" else [dst]
        v = self.operand(i, 1, 32)
        b, c = self.operand(i, 2, 32), self.operand(i, 3, 32)
        b, c = self.as_int(b, "the shfl lane"), self.as_int(c, "the shfl clamp")
        lane, m31 = E.LANE, E.const(31, 32)
        bval = E.bin("and", b, m31)
        segmask = E.bin("and", E.bin("lshr", c, E.const(8, 32)), m31)
        notseg = E.bin("xor", segmask, E.const(-1, 32))
        min_lane = E.bin("and", lane, segmask)
        max_lane = E.bin("or", min_lane, E.bin("and", E.bin("and", c, m31), notseg))
        if kind == "up":
            j = E.bin("sub", lane, bval)
            ok = E.cmp("sge", j, max_lane)
        elif kind == "down":
            j = E.bin("add", lane, bval)
            ok = E.cmp("sle", j, max_lane)
        elif kind == "bfly":
            j = E.bin("xor", lane, bval)
            ok = E.cmp("sle", j, max_lane)
        else:
            j = E.bin("or", min_lane, E.bin("and", bval, notseg))
            ok = E.cmp("sle", j, max_lane)
        if len(regs) > 1 and regs[1][0] == "reg":
            self.write(regs[1][1], ok)
        if regs[0][0] != "reg":
            return
        if isinstance(v, Data) or self.uniform(v) is not None:
            self.write(regs[0][1], v)  # every lane holds the same value (or data)
            return
        if _has_elect(v):
            self.fail(
                i, "shfl.sync of a value that depends on which lane elect.sync picks"
            )
        src = E.select(ok, j, lane)
        out = []
        for w, ln in self.threads():
            s = E.subst(src, self.thread_subst(w, ln))
            out.append(E.subst(E.subst(v, {"%warp": E.const(w, 32)}), {"%lane": s}))
        lane_only = all(out[k] is out[k % 32] for k in range(len(out)))
        self.write(regs[0][1], self.lane_table(out[:32] if lane_only else out, "sh"))

    def _vote(self, i, mode: str) -> None:
        """vote.sync over the full warp: ballot (the lanes' predicate bits), any, all or
        uni, from the predicate at each lane (frontend/cute/ops.py _ballot)."""
        self._full_mask(i, 2, f"vote.{mode}.sync")
        p = self.operand(i, 1, 1)
        if isinstance(p, Data):
            self.write(i.operands[0][1], p)
            return
        if _has_elect(p):
            self.fail(
                i,
                f"vote.{mode}.sync of a value that depends on which lane elect.sync picks",
            )
        bits = [E.subst(p, {"%lane": E.const(k, 32)}) for k in range(32)]
        if mode == "ballot":
            v = E.const(0, 32)
            for k, bit in enumerate(bits):
                v = E.bin("or", v, E.bin("shl", E.ext(bit, 32, False), E.const(k, 32)))
        elif mode == "any":
            v = _fold("or", bits)
        elif mode == "all":
            v = _fold("and", bits)
        else:
            v = E.bin("or", _fold("and", bits), E.not1(_fold("or", bits)))
        self.write(i.operands[0][1], v)


# --------------------------------------------------------------------- helpers


def _exits(seq) -> bool:
    """seq has a CONTINUE or BREAK of its loop (not of an inner one)."""
    for n in seq:
        if n in (R.CONTINUE, R.BREAK):
            return True
        if isinstance(n, R.If) and (_exits(n.then) or _exits(n.else_)):
            return True
    return False


def _fold(op, xs):
    out = xs[0]
    for x in xs[1:]:
        out = E.bin(op, out, x)
    return out


def _has_elect(e) -> bool:
    return any(v.startswith("%elect") for v in E.free_vars(e))


def _bare_elect(c) -> bool:
    """c is an elect symbol or its negation."""
    if E.is_elect(c):
        return True
    return c[0] == "bin" and c[2] == "xor" and E.is_elect(c[3]) and E.is_const(c[4], 1)


def _bindable(x) -> bool:
    return not (E.thread_varying(x) or E.reserved(x) or E.has_narrow(x))


def _hoistable(roots: list, names: dict):
    """(order, shared): the non-trivial nodes of `roots` a `let` may compute ahead of
    them, children first, and those of them reached more than once. Only a node every
    evaluation of the roots evaluates is hoistable: KIR evaluates a select's arm only when
    chosen, so the wraps (addr marks) and failing loads or divisions of the other arm are
    dead (lean/KV/Eval.lean), and a `let` ahead of the select would make them
    unconditional (R1: a predicated instruction's or a branch's value is such an arm)."""
    reached: set = set()
    count: dict = {}
    order: list = []
    stack = [(r, False, False) for r in reversed(roots)]
    while stack:
        x, arm, done = stack.pop()
        if done:
            order.append(x)
            continue
        if x[0] in ("const", "var", "pid", "nprog") or x in names:
            continue
        count[x] = count.get(x, 0) + 1
        if arm or x in reached:
            continue
        reached.add(x)
        stack.append((x, False, True))
        kids = E.children(x)
        if x[0] == "select":
            stack += [
                (kids[2], True, False),
                (kids[1], True, False),
                (kids[0], False, False),
            ]
        else:
            stack += [(c, False, False) for c in reversed(kids)]
    shared = {x for x, n in count.items() if n > 1 and x in reached}
    return order, shared


_CMP = {
    ("eq", True): "eq", ("eq", False): "eq", ("ne", True): "ne", ("ne", False): "ne",
    ("lt", True): "slt", ("le", True): "sle", ("gt", True): "sgt", ("ge", True): "sge",
    ("lt", False): "ult", ("le", False): "ule", ("gt", False): "ugt", ("ge", False): "uge",
    ("lo", False): "ult", ("ls", False): "ule", ("hi", False): "ugt", ("hs", False): "uge",
    ("lo", True): "ult", ("ls", True): "ule", ("hi", True): "ugt", ("hs", True): "uge",
}  # fmt: skip

_TYPE = re.compile(r"([bsu])(8|16|32|64|128)|(pred)")


def _types(opcode: str) -> list:
    """The integer types of an opcode, in order: [(kind "b"|"s"|"u", bits)]; .pred is
    ("b", 1). A float type makes the list end in ("f", bits)."""
    out = []
    for t in opcode.split(".")[1:]:
        m = _TYPE.fullmatch(t)
        if m:
            out.append(("b", 1) if m.group(3) else (m.group(1), int(m.group(2))))
        elif re.fullmatch(r"b?f(16|32|64)(x2)?", t):
            out.append(("f", 32))
    return out


def _reg_width(t: str) -> int:
    if t == "pred":
        return 1
    m = re.fullmatch(r"[bsuf](\d+)", t)
    return int(m.group(1)) if m else 32


def _dest_regs(i) -> list:
    op = i.operands[0] if i.operands else None
    if op is None:
        return []
    if op[0] == "pair":
        return [x[1] if x[0] == "reg" else None for x in (op[1], op[2])]
    if op[0] == "reg":
        return [op[1]]
    if op[0] == "vec":
        return [x[1] if x[0] == "reg" else None for x in op[1]]
    return [None]


def _shift_amount(b, w: int):
    """A PTX shift amount (an unsigned 32-bit value; amounts beyond the width shift
    everything out, ISA shl/shr) at width w."""
    if E.width(b) == w:
        return b
    if E.width(b) < w:
        return E.resize(b, w, False)
    big = E.cmp("ugt", b, E.const(w, E.width(b)))
    return E.trunc(E.select(big, E.const(w, E.width(b)), b), w)


def _c32(v: int) -> dict:
    return {"e": "const", "w": 32, "v": v}


def _lowered(e):
    return E.lower_narrow(e) if E.has_narrow(e) else e


def _conjuncts(c) -> list:
    if c[0] == "bin" and c[2] == "and" and c[1] == 1:
        return _conjuncts(c[3]) + _conjuncts(c[4])
    return [] if E.is_const(c, 1) else [c]


# ------------------------------------------------------------------ vectorizing


def _tensor_safe(e, arm: bool = False) -> bool:
    """e has a TExpr form evaluated on every thread with the same outcome as per thread:
    no table, no thread-varying metadata load or division that may fail, and nothing
    that may fail under a select arm (a TExpr select evaluates both arms)."""
    if not E.unsafe(e) and not _has_table(e):
        return True
    if "%t" not in E.free_vars(e):
        return not arm and not _has_table(e)
    k = e[0]
    if k in ("load", "table") or (
        k == "bin" and e[2] in E.PARTIAL_OPS and "%t" in E.free_vars(e[4])
    ):
        return False
    if k == "select":
        return (
            _tensor_safe(e[2], arm)
            and _tensor_safe(e[3], True)
            and _tensor_safe(e[4], True)
        )
    return all(_tensor_safe(c, arm) for c in E.children(e))


def _varying_data(e) -> bool:
    """e has a load or table that varies across threads."""
    return any(x[0] in ("load", "table") and E.thread_varying(x) for x in _walk(e))


def _has_table(e) -> bool:
    return any(x[0] == "table" for x in _walk(e))


def _walk(e):
    seen, todo = set(), [e]
    while todo:
        x = todo.pop()
        if x not in seen:
            seen.add(x)
            yield x
            todo += E.children(x)


def _scalar_offsets(off: dict, n: int) -> dict:
    """TExpr of n consecutive element offsets from off (i64)."""
    s = {"t": "splat", "shape": [n], "w": 64, "a": off}
    if n == 1:
        return s
    r = {"t": "range", "shape": [n], "w": 64, "start": 0, "end": n}
    return {"t": "bin", "shape": [n], "w": 64, "o": "add", "a": s, "b": r}


def _antiunify(xs: list):
    """The common pattern of expressions xs: an expression where they agree, ("k", c0,
    d, w) where they are the constants c0 + k*d (k = their index), ("node", x0, [child
    patterns]) where they have the same kind; None if they differ otherwise."""
    x0 = xs[0]
    if all(x is x0 for x in xs):
        return x0
    if all(E.is_const(x) for x in xs) and len({E.width(x) for x in xs}) == 1:
        d = xs[1][2] - x0[2]
        if all(x[2] == x0[2] + i * d for i, x in enumerate(xs)):
            return ("k", x0[2], d, E.width(x0))
        return None
    head = _head(x0)
    if head is None or any(_head(x) != head for x in xs):
        return _antiunify_const_term(xs)
    kids = [E.children(x) for x in xs]
    out = []
    for j in range(len(kids[0])):
        pj = _antiunify([c[j] for c in kids])
        if pj is None:
            return None
        out.append(pj)
    return ("node", x0, out)


def _antiunify_const_term(xs: list):
    """The pattern of xs where some are add(y, c) or or(y, c) with a constant c and the
    others are taken as y + 0 (y | 0): the identity an unrolled offset's first copy
    folded away."""
    for op in ("add", "or"):
        tmpl = next(
            (x for x in xs if x[0] == "bin" and x[2] == op and E.is_const(x[4])), None
        )
        if tmpl is None:
            continue
        parts = [(x[3], x[4]) if x[0] == "bin" and x[2] == op and E.is_const(x[4])
                 else (x, E.const(0, E.width(x))) for x in xs]  # fmt: skip
        py = _antiunify([y for y, _c in parts])
        pc = _antiunify([c for _y, c in parts])
        if py is not None and pc is not None:
            return ("node", tmpl, [py, pc])
    return None


def _head(x):
    """What two expressions of one pattern share besides their children."""
    k = x[0]
    if k == "bin":
        return ("bin", x[1], x[2])
    if k == "cmp":
        return ("cmp", x[1])
    if k in ("select", "trunc"):
        return (k, x[1])
    if k == "ext":
        return ("ext", x[1], x[2])
    return None


_AXES = ("k", "t", "j")


def _lift(te: dict, have: tuple, want: tuple, sizes: dict) -> dict:
    """te over the axes `have` broadcast to the axes `want` (a superset, in _AXES order)."""
    cur = [sizes[a] for a in have]
    for pos, a in enumerate(want):
        if a not in have:
            cur = cur[:pos] + [1] + cur[pos:]
            te = {"t": "expand", "shape": list(cur), "w": te["w"], "a": te, "axis": pos}
    for pos, a in enumerate(want):
        if a not in have:
            cur[pos] = sizes[a]
            te = {"t": "bcast", "shape": list(cur), "w": te["w"], "a": te, "axis": pos}
    return te


def _vectorize(tr, pat, sizes: dict, want: tuple) -> dict:  # noqa: C901 (one case per node)
    """TExpr over the axes `want` of a pattern (_antiunify): axis "k" indexes the merged
    instructions, "t" the role's threads (the values of %t). Each node is computed over
    the axes it depends on and broadcast where it meets a node with more (the TTGIR
    frontend's form); a thread-uniform node is a splat."""
    memo: dict = {}

    def scalar(x, axes):
        shape = [sizes[a] for a in axes] or [1]
        if E.is_const(x):
            return {"t": "full", "shape": shape, "w": E.width(x), "v": x[2]}
        return {"t": "splat", "shape": shape, "w": E.width(x), "a": tr.json(x)}

    def materialize(r, axes):
        if r[0] == "s":
            return scalar(r[1], axes)
        return _lift(r[1], r[2], axes, sizes)

    def go(p):
        """("s", expr) for a uniform value, ("t", texpr, axes) for a varying one."""
        key = id(p) if p[0] == "node" else p
        if key in memo:
            return memo[key]
        if p[0] == "k":
            _k, c0, d, w = p
            n = sizes["k"]
            te = {"t": "range", "shape": [n], "w": w, "start": 0, "end": n}
            if d != 1:
                te = {"t": "bin", "shape": [n], "w": w, "o": "mul", "a": te,
                      "b": {"t": "full", "shape": [n], "w": w, "v": d}}  # fmt: skip
            if c0:
                te = {"t": "bin", "shape": [n], "w": w, "o": "add", "a": te,
                      "b": {"t": "full", "shape": [n], "w": w, "v": c0}}  # fmt: skip
            out = ("t", te, ("k",))
        elif p[0] != "node" and "%t" not in E.free_vars(p):
            out = ("s", p)
        elif p[0] == "var":
            lo = 32 * tr.w0
            out = ("t", {"t": "range", "shape": [sizes["t"]], "w": 32, "start": lo,
                         "end": lo + sizes["t"]}, ("t",))  # fmt: skip
        else:
            x, kids = (p[1], [go(c) for c in p[2]]) if p[0] == "node" else (
                p, [go(c) for c in E.children(p)])  # fmt: skip
            axes = tuple(
                a for a in _AXES if any(r[0] == "t" and a in r[2] for r in kids)
            )
            shape = [sizes[a] for a in axes]
            m = [materialize(r, axes) for r in kids]
            k, w = x[0], E.width(x)
            if k == "bin":
                te = {
                    "t": "bin",
                    "shape": shape,
                    "w": w,
                    "o": x[2],
                    "a": m[0],
                    "b": m[1],
                }
            elif k == "cmp":
                te = {
                    "t": "cmp",
                    "shape": shape,
                    "w": 1,
                    "o": x[1],
                    "a": m[0],
                    "b": m[1],
                }
            elif k == "select":
                te = {
                    "t": "select",
                    "shape": shape,
                    "w": w,
                    "c": m[0],
                    "a": m[1],
                    "b": m[2],
                }
            elif k == "ext":
                te = {"t": "ext", "shape": shape, "w": w, "signed": x[2], "a": m[0]}
            elif k == "trunc":
                te = {"t": "trunc", "shape": shape, "w": w, "a": m[0]}
            else:
                raise E.ExprError(f"no tensor form for {k}")
            out = ("t", te, axes)
        memo[key] = out
        return out

    return materialize(go(pat), want)
