# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""P0-6 "named barriers convergent": the barrier pass of the PTX frontend (both families).

    translate(ptx_text, entry, launch_records, policy) -> Facts

It reads the PTX of a variant whatever its P0 translation does (also where that refuses
it, also under `--frontend legacy`) and makes the bodies of a P0-6 KIR (references/kir.md,
change 14): per role, its control flow restricted to what decides how often each named
barrier is reached, and one event per named barrier instruction reached. Lean evaluates
that KIR per input and decides P0-6 (lean/KV/Spec/NamedBarriers.lean); this pass
classifies, it decides no verdict.

Model (PTX ISA, bar / barrier). A CTA has 16 named barriers. `bar.sync a{, b}` is
`barrier.sync.aligned`, `bar.arrive a, b` is `barrier.arrive.aligned`, `bar.red` is
`barrier.red.aligned`: an aligned barrier is executed by all threads of a warp together
(`bar.cta`/`barrier.cta` are the same instructions: ops.mnemonic drops the `.cta`).
A warp's arrival counts its 32 threads; an instance of barrier a completes when b threads
arrived (no b: all threads of the CTA); `sync` and `red` wait for the completion,
`arrive` does not. A named barrier converges when the warps that take part in it reach
it the same number of times with the same id and count, so this pass checks whether every
condition that decides that is uniform over those warps.

Roles. TLX/Triton: the warp-specialize roles of cfg.py (`default`, `partition<k>`),
padding states' roles (Role.padding) included whether or not they emit an event: their
warps run the code before the dispatch, which may hold CTA-wide barriers (the P0
translation drops a padding role that emits no event). The dispatch's own fork barriers
(`barrier.sync 1` without a count, whose protocol cfg.py checks) emit nothing, and any
other use of barrier 1 in such a kernel is an `nb_open`. CuTe and Triton without a dispatch: one role of the whole block. A role is
split into warp ranges (interp.SplitWarps) wherever a condition that decides a barrier
(a branch around it, a loop exit of a loop holding one, its predicate, id or count) is
uniform within each warp but differs between warps of the role: `partition1:w12-12`
(TLX), `warps4_7` (CuTe). So every `named_bar` event is executed by every warp of its
role, and Lean counts arrivals per role.

Values. Integers are interp.py's exact expressions; the rest is data, classified by
NbData: `var`, how it varies over the role's threads: "u" (one value), "l" (a value per
lane, the same in every warp), "w" (a value per warp) or "t" (any); `nondet`: two reads
of it may differ (a strong load, `.volatile`/`.relaxed`/`.acquire`/`.cv`; an atomic's
result; %clock, %globaltimer; an mbarrier test_wait or try_wait result outside the
blocking wait idiom); `srcs`: the strong loads it derives from. Uniform data, the trust
base of a PASS (references/specs.md "P0-6"):
* kernel parameters and constants (exact), ctaid and cluster rank (exact);
* a weak global load, at an address uniform over the role, of a tensor the kernel never
  writes (translate.writes, here ignoring stores of shared memory only and taking a value
  loaded with a type of 32 bits or less as no pointer; by tensor name: that no other
  tensor argument shares the storage of a written one is assumed, the callers check
  storage only for metadata tensors); an integer one of a tensor in the policy's
  `nb_meta` (the family hook's metadata tensors: recorded contents) is an exact metadata
  `load` instead (mem.Memory._load, as in the P0 translation), so loops bounded by it
  are exact KIR loops whose passes Lean counts;
* shared-memory data (a weak ld.shared, ldmatrix, a generic load of a shared address,
  the CLC response; a strong ld.shared is a strong load): a load at an address of class
  u (l, w) has class u (l, w), an ldmatrix at an address of class u or l class l.
  Assumed: the location was written before a synchronization the readers passed, so
  every reader sees the same value;
* published values: a value a spin loop accepted (the loop re-reads it with strong loads
  until its exit test accepts it; every source of the value is such a load, at an address
  uniform over the role), in the register that holds it as loaded (the load's
  destination or a copy of it) at every exit; a value computed from it in the loop is
  data that may differ between the threads, no certain divergence. Assumed: the producer
  publishes the accepted value once (a write-once slot, a tagged word whose tag is
  written once), so every reader accepts the same value. The loop itself repeats a
  number of times that differs between readers;
* a warp collective over lanes: redux/vote of class-u or class-l data is u (of other data
  w); `shfl.sync.idx` from one lane is u (w); other shuffles keep their operand's class.
Every other data is "t". Each class also records `ou`: the value is uniform over the role
before warp splitting (exact values: their thread-index instances over all its warps;
symbols and loads inherit it), and `ol`: it is alike in corresponding lanes of all that
role's warps (a warp collective of `ol` data is `ou`). An `nb_rep` region whose condition is `ou` gets the key
`<role>@<PTX line>`, the same in every split role: they repeat it alike, so Lean may pair
a barrier they share inside one nest of keyed regions.

Repeated regions of split roles. A split role translates only what its own warps run:
an If or predicate whose condition is one constant over its threads takes one arm, and
some values are held as its threads' instances (a KIR `if` or a slot on a value that is
one over its threads, a shuffle's table); `ou` judges such a value, not the path it came
by. So each keyed region of a split role records its path signature (_sign): the
decisions on the paths of the values its condition is computed from (data and control
dependence; the registers a loop writes depend on its exits), a value held as its own
instance counting as a decision no other role shares. Where the signatures of one key
differ between the roles split from one role, the key gets `#<n>`: regions whose
conditions came by different paths never pair.

Control flow (regions.py tree; registers live across it as in interp.py):
* An If whose arms hold no named barrier (nor, inside a loop holding one, an exit of that
  loop) only computes values: both arms run (under the mask for an exact condition) and
  registers merge (a data condition: the classes join); on an exact condition uniform
  over the role around a loop it is interp's KIR `if`, so the loop may be exact.
* An If deciding a barrier: an exact condition uniform over the role is a KIR `if`; one
  uniform within each warp but not over the role splits the role; a class-u data
  condition makes each arm an `nb_rep` region (taken 0 or 1 times, alike in every warp);
  any other condition (lanes of a warp differ, or data of class l, w, t) is divergent:
  both arms are translated and every barrier reached in them is an `nb_open`.
* A loop is first translated exactly (interp.Translator.run_loop: a KIR `for`/`while`),
  which needs every exit condition exact and uniform. Else its classes are iterated to a
  fixed point (the registers it writes start as their entry classes, every back and exit
  edge joins in) and its exit class is the join of its exit conditions' classes: a loop
  holding a barrier is an `nb_rep` region (its body once, between `nb_rep` begin and end
  events) if that class is u, else divergent (its body once, every barrier an `nb_open`).
  A loop without a barrier only computes values.
An `nb_open` is `certain` when the model admits participants that diverge: thread-index
constants that differ (an elect.sync leader), or nondet data (warps may read different
values, and a spin loop's warps leave it in different iterations), for every divergent
condition around it, and it lies in no `nb_rep` region (which may run no pass); else not
certain. Lanes of a warp that diverge at a barrier without `.aligned` are never certain:
each thread arrives on its own, which the model does not count.
A blocking mbarrier wait (regions.Wait), mbarriers, memory, TMA, tcgen05 and CLC emit
nothing here.

Named barrier instructions: id and count must be constants (count absent: the block's
threads); a per-warp value splits the role, any other value is an `nb_open` (kind
`operand`, not certain). Under a lane-varying mask (a predicate, elect.sync) a barrier is
an `nb_open` of kind `lane`.

Control flow outside cfg.py's shape: a switch `brx.idx` (not the dispatch) is
cfg.lower_switches' compare chain, as in the P0 translation, with the same check that its
index is provably in range (interp's _switch_index). A cycle entered other than at its
header (LLVM tail-merges the identical exits of two inlined spin loops) is made reducible
by node splitting: the blocks entered from outside, up to the header, are copied for those
entries. Still refused (Facts.refusals, P0-6 UNDECIDED for the variant): other control
flow outside cfg.py / regions.py (a loop exiting to two blocks, several rets, several
dispatches, a switch index not provably in range), a call of a device function, roles
that are not warp ranges, an instruction without an op table row that may be a barrier or
a branch.
"""

import dataclasses
from dataclasses import dataclass, field

from . import (
    cfg as C,
    cluster,
    expr as E,
    interp,
    mem as M,
    ops,
    parse,
    regions as R,
    translate as T,
)

# Named barrier instructions (op table rows "thread and cluster barriers" and
# "CTA barrier with a reduction"), by normalized mnemonic.
_NAMED = ("bar.sync", "bar.arrive", "barrier.sync", "barrier.sync.aligned",
          "barrier.arrive", "barrier.arrive.aligned")  # fmt: skip
_RED = ("bar.red.", "barrier.red.")
# Instruction bases that are or may be barriers or control flow: one without an op table
# row is an error here too (any other only computes its results, data in this pass).
_CONTROL = ("bar", "barrier", "bra", "brx", "call", "ret", "exit", "trap")
_TIMERS = ("%clock", "%clock64", "%globaltimer", "%globaltimer_lo", "%globaltimer_hi")
_NONE = frozenset()
_U = ("u", False, _NONE)
_UO = ("u", False, _NONE, True, True)  # uniform over the role before warp splitting too
# The class of a register a loop writes but nothing reads at its header or after its exit
# (regions.py liveness): its value never reaches a barrier decision, so it is not
# classified (the classification enumerates the role's threads: the bulk of the pass).
_DEAD = ("t", False, _NONE, False, False)
# Integer comparisons and their negations (the acceptance test of a spin, a guard).
_NEG = {"eq": "ne", "ne": "eq", "lt": "ge", "ge": "lt", "le": "gt", "gt": "le",
        "lo": "hs", "hs": "lo", "ls": "hi", "hi": "ls"}  # fmt: skip
_ROUNDS = 8  # class fixed point of a loop (each class rises at most 3 times)
_SPLIT_BUDGET = 64  # node splitting rounds (each removes one irreducible entry)
_LOCAL = "%local"  # local memory, as one register of the path signatures


def is_named(mn: str) -> bool:
    """mn (a normalized mnemonic) is a named barrier instruction."""
    return mn in _NAMED or mn.startswith(_RED)


@dataclass(frozen=True)
class NbData(interp.Data):
    """Data with its variation over the role's threads (module doc)."""

    var: str = "t"
    nondet: bool = False
    srcs: frozenset = _NONE
    ou: bool = False  # one value over the role before warp splitting (rep keys)
    ol: bool = False  # the same in corresponding lanes of all that role's warps
    pub: frozenset = _NONE  # a published value: the addresses whose spin accepted it
    raw: bool = False  # the value a strong load returned, as loaded (or a copy of it)


@dataclass
class Facts:
    kernel: str
    target: str = ""
    block: tuple = ()
    cluster: tuple = (1, 1, 1)
    roles: list = field(default_factory=list)  # [{name, w0, warps, body}]
    params: dict = field(default_factory=dict)  # KIR param -> width
    sites: list = field(
        default_factory=list
    )  # the classification of every barrier site
    dispatch: bool = False  # a Triton warp-specialize dispatch (fork barriers skipped)
    split: list = field(default_factory=list)  # blocks copied by node splitting
    refusals: list = field(default_factory=list)  # [{"kind": "gap" | "error", "msg"}]
    meta: dict = field(
        default_factory=dict
    )  # metadata tensor -> elem bytes (KIR loads)
    written: list | None = None  # the tensors the kernel may write (None: unknown)


class _Fallback(Exception):
    """A loop is not exact: an exit condition is not an exact uniform value."""


class _Symbolic(_Fallback):
    """A KIR expression would read a symbol of a loop translated by classes."""


def _lanes_of(fv) -> range:
    """The lanes whose substitution can differ for an expression with free vars fv
    (all, or lane 0 alone when %lane is not among them)."""
    return range(32) if "%lane" in fv else range(1)


def _warp_subst(tr, e, fv, w: int):
    """e with %warp := w (memoized in the translator, interp's memo)."""
    if "%warp" not in fv:
        return e
    return E.subst(e, {"%warp": E.const(w, 32)}, tr.memo(("w", w)))


def _lane_subst(tr, e, fv, ln: int):
    """e with %lane := ln (memoized in the translator, interp's memo)."""
    if "%lane" not in fv:
        return e
    lc = E.const(ln, 32)
    return E.subst(e, {"%lane": lc}, tr.memo(("l", lc)))


def _lane_elect(e, lane: int, cache: dict):
    """e with %lane := lane and every %elect var := (lane == 0), memoized in `cache`:
    interp._warp_instances's per-lane substitution (whose mapping names the %elect vars
    of its expression; a node's result depends only on the vars it holds, so one memo per
    lane serves every expression)."""

    def go(x):
        r = cache.get(x)
        if r is None:
            if x[0] == "var":
                name = x[2]
                if name == "%lane":
                    r = E.const(lane, 32)
                elif name.startswith("%elect"):
                    r = E.const(int(lane == 0), 1)
                else:
                    r = x
            elif not any(
                v == "%lane" or v.startswith("%elect") for v in E.free_vars(x)
            ):
                r = x
            else:
                r = E.rebuild(x, [go(c) for c in E.children(x)])
            cache[x] = r
        return r

    return go(e)


def _jvar(a: str, b: str) -> str:
    if a == b or b == "u":
        return a
    return b if a == "u" else "t"


def _join(*cls) -> tuple:
    var, nd, srcs, ou, ol = "u", False, _NONE, True, True
    for c in cls:
        var, nd, srcs = _jvar(var, c[0]), nd or c[1], srcs | c[2]
        ou, ol = ou and len(c) > 3 and c[3], ol and len(c) > 4 and c[4]
    return var, nd, srcs, ou, ol


def _lanes(cls) -> tuple:
    """The class of a collective over the lanes of each warp of a value of class cls: one
    value per warp, the same in every warp where the operand's lanes are alike in every
    warp of the role before splitting."""
    same = len(cls) > 4 and cls[4]
    return ("u" if cls[0] in ("u", "l") else "w", cls[1], cls[2], same, same)


# ------------------------------------------------------------------------- memory


class _Memory(M.Memory):
    """mem.Memory's addresses and parameters, with no event: every load is a value of
    its class, every other memory instruction a no-op (its results data)."""

    def execute(self, tr, i, mn: str) -> None:
        try:
            if mn == "ld.param" or mn.startswith("cvta"):
                return super().execute(tr, i, mn)
            if mn.startswith("mapa"):
                return cluster.mapa(tr, i)
            if mn in ("ld.local", "st.local") or (
                mn in ("ld", "st")
                and self._is_local(tr, i.operands[1 if mn == "ld" else 0])
            ):
                return self._local(tr, i, mn)
        except interp.Unsupported:
            pass  # a value the P0 rules refuse: data below
        if mn == "ld" or mn.startswith("ld.global"):
            return self._nb_load(tr, i)
        if mn.startswith("clusterlaunchcontrol.query_cancel"):
            cls = tr.klass(tr.operand(i, len(i.operands) - 1))
            for r in R.dests(i):
                tr.env[r] = tr._val(f"the CLC response at {tr.where(i)}", cls, (r,))
            return
        nondet = mn.startswith("atom")  # each thread's atomic returns its own value
        for r in R.dests(i):
            tr.env[r] = NbData(f"{i.opcode} at {tr.where(i)}", "t", nondet)

    def _nb_load(self, tr, i) -> None:
        """A global (or generic) load (module doc "Values")."""
        try:
            a = self.address(tr, i.operands[1])
        except interp.Unsupported:
            a = None
        if ops.strong(i.opcode):
            tr.strong_load(i, a)
            return
        p = None if isinstance(a, interp.Data) else M.peel(a)
        base = p[0][2] if p is not None else ""
        written = self.policy.get("written")
        mine = base.startswith(("%s:", "%S:")) or (
            base.startswith("%g:") and written is not None and base[3:] not in written
        )
        if mine and base[3:] in self.policy.get("nb_meta", ()) and self._exact(tr, i):
            # a base mem.py picked by a condition of this role's warps (_pick_bases)
            if tr.track and not tr.olane(a):
                tr.own(R.dests(i), i.line)
            return
        ka = tr.klass(a) if mine else _DEAD  # the address's class matters only then
        cls = (ka[0] if mine else "t", False, _NONE, mine and ka[3], mine and ka[4])
        for r in R.dests(i):
            tr.env[r] = tr._val(f"{i.opcode} at {tr.where(i)}", cls, (r,))

    def _exact(self, tr, i) -> bool:
        """A weak integer load of a metadata tensor (policy `nb_meta`: recorded contents,
        never written by the kernel) as an exact KIR `load` (mem.Memory._load), the way the
        P0 translation reads it; False (nothing written) where that gives no exact value
        for some register."""
        dests = R.dests(i)
        old = {r: tr.env.get(r) for r in dests}
        try:
            self._load(tr, i, ops.mnemonic(i.opcode))
        except interp.Unsupported:
            ok = False
        else:
            ok = not any(isinstance(tr.env.get(r), interp.Data) for r in dests)
        if not ok:
            for r, x in old.items():
                tr.env[r] = x
        return ok

    def wait(self, tr, instr) -> None:
        """The blocking wait pseudo-instruction: no P0-6 event."""


# --------------------------------------------------------------------- translator


class _Nb(interp.Translator):
    """interp.Translator with the P0-6 slice: control flow and named barriers only."""

    def __init__(self, k, tree, w0: int, nw: int, name: str, ctx: dict, orig=None):
        super().__init__(k, tree, w0, nw, name)
        self.ctx = ctx
        self.ow0, self.onw = orig or (w0, nw)  # the role before warp splitting
        self._ou: dict = {}
        self.nb: list = []  # open loops: {"mode": "exact" | "class", ...}
        self.div: list = []  # enclosing divergent regions: (kind, certain, why, by data)
        self.reps = 0  # enclosing nb_rep regions
        self.sites: list = []
        # Path signatures of a split role (module doc "Repeated regions"), recorded over
        # the whole translation: the registers each register's values were computed
        # from, the decisions and own instances on their paths, the stack of the
        # decisions around the current point, the open loops' exits, and the keyed
        # nb_rep regions: [begin event, (registers, decisions) or an open loop's exits].
        self.track = (self.ow0, self.onw) != (w0, nw)
        self._deps: dict = {}
        self._path: dict = {}
        self._ctl: list = [_NONE]
        self._exits: list = []
        self.keyed: list = []

    def where(self, i) -> str:
        """`file.py:line (PTX line n)` of an instruction, for messages (the event's own
        loc keeps the full path)."""
        return f"{self.loc(i).rsplit('/', 1)[-1]} (PTX line {i.line})" if i else "?"

    # ------------------------------------------------------------ no P0 events
    def mem_event(self, kind, tensor, off, n) -> None:
        return

    def thread_event(self, operands, build, tag="event") -> None:
        return

    def barrier_event(self, ev, sym, addr, mode, lanes=False, **fields) -> None:
        return

    def require(self, cond, msg: str, op: str = "assert") -> None:
        return

    def collective(self, what) -> None:
        return

    def work(self, i, sites) -> None:
        return

    def snapshot(self):
        return super().snapshot() + (
            (len(self.sites), self.reps),
        )  # base indexes the rest

    def restore(self, snap) -> None:
        super().restore(snap[:-1])
        n, self.reps = snap[-1]
        del self.sites[n:]

    # ------------------------------------------------------ class-loop symbols
    @staticmethod
    def _symbolic(e) -> bool:
        return any(v.startswith("nb~") for v in E.free_vars(e))

    def _in_class(self) -> bool:
        return any(x["mode"] == "class" for x in self.nb)

    def json(self, e) -> dict:
        if self._symbolic(e):
            raise _Symbolic()
        return super().json(e)

    def _emit_under(self, uni, stmt: dict) -> None:
        if not E.is_const(uni, 1) and self._symbolic(uni):
            self._rep(True, "under a condition uniform over the role, in a loop "
                            "translated by classes")  # fmt: skip
            self.flush()
            self.stmts.append(stmt)
            self._rep(False, "")
            return
        super()._emit_under(uni, stmt)

    # ------------------------------------------------------------- classes
    def klass(self, v) -> tuple:
        """(var, nondet, srcs) of a value over the role's threads (module doc)."""
        if v is None:
            return "t", False, _NONE, False, False
        if isinstance(v, NbData):
            cls = v.var, v.nondet, v.srcs, v.ou, v.ol
        elif isinstance(v, (cluster.Remote, cluster.Peer)):
            cls = self.klass(v.addr)
        elif isinstance(v, interp.WarpData):
            cls = "w", False, _NONE, False, False
        elif isinstance(v, interp.Data):
            cls = "t", False, _NONE, False, False
        elif not E.thread_varying(v) or self.uniform(v) is not None:
            ou = self.ouniform(v)
            cls = _U + (ou, ou)
        elif self.per_warp(v) is not None:
            cls = "w", False, _NONE, False, False
        elif not interp._has_elect(v) and self.lane_only(v):
            cls = "l", False, _NONE, False, self.olane(v)
        else:
            cls = "t", False, _NONE, False, self.olane(v)
        return ("u",) + cls[1:] if cls[0] == "w" and self.nw == 1 else cls

    def _warp_instances(self, e, w: int, lanes: tuple = interp._LANES) -> list:
        """interp's _warp_instances, with the substitution memos shared between
        expressions (interp's memo; ("e", lane) for _lane_elect's): the values of a role
        mostly share their thread-dependent subexpressions (thread index, lane offsets),
        which then are rebuilt once."""
        r = self._inst.get(e)
        if r is not None:
            return [r[32 * (w - self.w0) + ln] for ln in lanes]
        fv = E.free_vars(e)
        if not any(map(E.is_thread_var, fv)):
            return [e] * len(lanes)
        ew = _warp_subst(self, e, fv, w)
        if not any(v == "%lane" or v.startswith("%elect") for v in fv):
            return [ew] * len(lanes)  # every lane's substitution is the identity
        return [_lane_elect(ew, lane, self.memo(("e", lane))) for lane in lanes]

    def ouniform(self, e) -> bool:
        """The exact value e has one value over every thread of the role before warp
        splitting, its symbols too (the condition of a shared nb_rep key)."""
        if any(
            v.startswith("nb~") and v not in self.ctx["sym_ou"] for v in E.free_vars(e)
        ):
            return False
        if not E.thread_varying(e):
            return True
        if (self.ow0, self.onw) == (self.w0, self.nw):
            return self.uniform(e) is not None
        if interp._has_elect(e):
            return False
        if e not in self._ou:
            self._ou[e] = self._same_instances(e, E.free_vars(e))
        return self._ou[e]

    def _same_instances(self, e, fv) -> bool:
        """Whether e has one instance over every thread of the role before warp
        splitting (ouniform), stopping at the first that differs; an ExprError (e.g. a
        table of the split role's threads only) is a difference."""
        first = None
        try:
            for w in range(self.ow0, self.ow0 + self.onw):
                ew = _warp_subst(self, e, fv, w)
                for ln in _lanes_of(fv):
                    x = _lane_subst(self, ew, fv, ln)
                    if first is None:
                        first = x
                    elif x is not first:
                        return False
        except E.ExprError:
            return False
        return True

    def olane(self, e) -> bool:
        """The exact value e is the same in corresponding lanes of every warp of the role
        before warp splitting (its symbols uniform over that role)."""
        fv = E.free_vars(e)
        if any(v.startswith("nb~") and v not in self.ctx["sym_ou"] for v in fv):
            return False
        if "%warp" not in fv:
            return True
        key = ("l", e)
        if key not in self._ou:
            ok = True
            try:
                for ln in _lanes_of(fv):
                    el = _lane_subst(self, e, fv, ln)
                    vals = [_warp_subst(self, el, fv, w)
                            for w in range(self.ow0, self.ow0 + self.onw)]  # fmt: skip
                    if any(x is not vals[0] for x in vals):
                        ok = False
                        break
            except E.ExprError:  # e.g. a table of the split role's threads only
                ok = False
            self._ou[key] = ok
        return self._ou[key]

    def _key(self, line: int, ou: bool) -> str:
        """The key of an nb_rep region: the role before splitting and the PTX line of
        the condition, if that condition is uniform over that role (the sibling roles of
        a split then repeat the region alike), else empty. _sign appends a split role's
        path signature where the sibling roles' signatures of one key differ."""
        return f"{self.name}@{line}" if ou else ""

    # ------------------------------------------------------- path signatures
    def _dep(self, r: str, srcs=(), path=_NONE) -> None:
        if srcs:
            self._deps.setdefault(r, set()).update(srcs)
        if path:
            self._path.setdefault(r, set()).update(path)

    def own(self, regs, line: int) -> None:
        """regs now hold a value in a form only this role's warps share (its instance
        over their threads of a value that differs over the role before splitting): a
        path no sibling role shares."""
        if self.track:
            for r in regs:
                self._dep(r, (), {(line, f"w{self.w0}")})

    def _special(self, v) -> bool:
        """v has one value over this role's threads (one per lane, alike in its warps)
        but not over the role before splitting (not alike in that role's warps): a slot
        of interp.py holds v's instance, which hides that."""
        if not self.track or not isinstance(v, tuple) or not E.thread_varying(v):
            return False
        if self.uniform(v) is not None:
            return not self.ouniform(v)
        return not interp._has_elect(v) and self.lane_only(v) and not self.olane(v)

    def _flow(self, i) -> None:
        """What instruction i's results are computed from: the registers it reads (local
        memory as one register), the decisions around it and a predicate this role
        resolves to a constant."""
        mn = ops.mnemonic(i.opcode)
        srcs, path = R.reads_of(i), self._ctl[-1]
        if i.guard is not None:
            p = self.env.get(i.guard[0])
            if p is not None and not isinstance(p, interp.Data):
                u = self.uniform(E.not1(p) if i.guard[1] else p)
                if u is not None and E.is_const(u):
                    path = path | {(i.line, int(u[2] != 0))}
        if mn in ("st", "st.local"):
            self._dep(_LOCAL, srcs, path)
        if mn in ("ld", "ld.local"):
            srcs = srcs | {_LOCAL}
        for r in R.dests(i):
            self._dep(r, srcs, path)

    def signature(self, spec) -> str:
        """The path signature of a keyed region: every decision on the paths of the values
        its condition is computed from (spec: (registers, decisions), or the exits of
        its loop), in a canonical order."""
        roots, path = (spec["roots"], spec["path"]) if isinstance(spec, dict) else spec
        out, seen, todo = set(path), set(), list(roots)
        while todo:
            r = todo.pop()
            if r not in seen:
                seen.add(r)
                out |= self._path.get(r, _NONE)
                todo.extend(self._deps.get(r, ()))
        return ",".join(
            f"{a}:{b}" for a, b in sorted(out, key=lambda x: (x[0], str(x[1])))
        )

    def strong_load(self, i, a) -> None:
        """A strong load (ops.strong) at address a: data another thread may write while
        the warps read it (nondet), a source of a published value (module doc)."""
        ka = self.klass(a)
        key = None if a is None or isinstance(a, (interp.Data, cluster.Remote,
                                                   cluster.Peer)) else a  # fmt: skip
        self.ctx["loads"][id(i)] = (ka[0], key, ka[3])
        v = NbData(f"{i.opcode} at {self.where(i)}, a strong load", "t", True,
                   frozenset((id(i),)), raw=True)  # fmt: skip
        for r in R.dests(i):
            self.env[r] = v

    def _val(self, origin, cls, regs=()):
        """A value of class cls for the registers regs: class-u data (no strong load
        in it) is a fresh symbol, exact and uniform, so arithmetic on it stays exact
        (addresses keep their base); it never reaches the KIR (json, _Symbolic)."""
        widths = {_width(self.k.fn.regs.get(r, "")) for r in regs}
        if cls[:3] == _U and len(widths) == 1 and 0 not in widths:
            self.ctx["nsym"] += 1
            name = f"nb~{self.ctx['nsym']}:{sorted(regs)[0]}"
            if len(cls) > 3 and cls[3]:
                self.ctx["sym_ou"].add(name)
            return E.var(name, widths.pop())
        return NbData(origin, *cls)

    def _data(self, origin, args):
        cls = _join(*(self.klass(a) for a in args if a is not None))
        regs = R.dests(self.cur) if self.cur is not None else ()
        return self._val(self._root(origin, args), cls, regs)

    @staticmethod
    def _root(origin, args) -> str:
        """The origin of a nondet operand (the root cause of a divergence), else origin."""
        for a in args:
            if isinstance(a, NbData) and a.nondet:
                return a.origin
        return origin

    def _select(self, c, x, y, reg):
        if x is y:
            return x
        if isinstance(c, interp.Data) or any(
            v is None or isinstance(v, interp.Data) for v in (x, y)
        ):
            vals = (c, x, y)
            origin = self._root(f"{reg} merged at {self.where(self.cur)}", vals)
            return self._val(origin, _join(*map(self.klass, vals)), (reg,))
        return E.select(c, x, y)

    def sreg(self, name):
        if name in _TIMERS:
            return NbData(f"special register {name}", "t", True)
        return super().sreg(name)

    def _vote(self, i, mode):
        p = self.operand(i, 1, 1)
        if isinstance(p, interp.Data):
            reg = i.operands[0][1]
            origin = self._root(f"vote.{mode} at {self.where(i)}", [p])
            self.write(reg, self._val(origin, _lanes(self.klass(p)), (reg,)))
            return
        super()._vote(i, mode)

    def _redux(self, i, op):
        a = self.operand(i, 1, 32)
        if isinstance(a, interp.Data):
            reg = i.operands[0][1]
            origin = self._root(f"redux.{op} at {self.where(i)}", [a])
            self.write(reg, self._val(origin, _lanes(self.klass(a)), (reg,)))
            return
        super()._redux(i, op)

    def _shfl(self, i, kind):
        v = self.operand(i, 1, 32)
        super()._shfl(i, kind)
        dst = i.operands[0]
        reg = dst[1] if dst[0] == "reg" else dst[1][1] if dst[0] == "pair" else None
        if (reg is not None and self.track and not isinstance(v, interp.Data)
                and self.uniform(v) is None and not self.olane(v)):  # fmt: skip
            self.own((reg,), i.line)  # a table of this role's threads' values
        if not isinstance(v, interp.Data) or reg is None:
            return
        b, c = self.operand(i, 2, 32), self.operand(i, 3, 32)
        one = (kind == "idx" and not isinstance(b, interp.Data)
               and self.uniform(b) is not None and not isinstance(c, interp.Data)
               and E.is_const(c, 31))  # fmt: skip
        cls = _lanes(self.klass(v)) if one else self.klass(v)
        origin = self._root(f"shfl.sync.{kind} at {self.where(i)}", [v])
        self.env[reg] = self._val(origin, cls, (reg,))

    # --------------------------------------------------------- instructions
    def step(self, i) -> None:
        self.cur = i
        if self.track:
            self._flow(i)
        try:
            cls, why = ops.classify(i)
        except ops.UnknownInstruction as e:
            if i.base in _CONTROL:  # it could be a barrier or move control: fail closed
                self.fail(i, str(e), kind="error")
            for r in R.dests(i):  # no barrier, no control: only its results matter here
                self.env[r] = NbData(f"{i.opcode} at {self.where(i)} (no op table row)")
            return
        if cls == "control":
            return
        if cls == "refuse":
            if i.base == "call":
                self.fail(i, f"P0-6: a call ({why}): named barriers in a called function "
                             "are not analyzed")  # fmt: skip
            mn = ops.mnemonic(i.opcode)
            nondet = mn.startswith(("mbarrier.try_wait", "mbarrier.test_wait"))
            for r in R.dests(i):
                self.env[r] = NbData(f"{i.opcode} at {self.where(i)}", "t", nondet)
            return
        if id(i) in self.k.switches:  # the index of a lowered switch must be in range
            self._switch_index(i, *self.k.switches[id(i)])
        if i.guard is None:
            self.execute(i, cls)
        else:
            self.predicated(i, cls)

    def execute(self, i, cls) -> None:
        mn = ops.mnemonic(i.opcode)
        if is_named(mn):
            self.named_bar(i, mn)
            if mn.startswith(_RED):
                self._barrier_red(i, mn)
            return
        if cls == "drop":
            if mn.startswith(("ld.shared", "ldmatrix")):
                self._shared_load(i, mn)
                return
            for r in R.dests(i):
                self.env[r] = interp.Data(f"{i.opcode} at {self.where(i)}")
            return
        if cls == "event" or mn == "ld.param" or mn.startswith(("mapa", "cvta")):
            self.k.mem.execute(self, i, mn)
            return
        try:
            super().execute(i, cls)
        except interp.Unsupported as e:  # a value the P0 rules refuse: data here
            for r in R.dests(i):
                self.env[r] = NbData(
                    f"{i.opcode} at {self.where(i)} ({e.why})", "t", False
                )

    def write(self, reg: str, v) -> None:
        own = (self.track and isinstance(v, tuple) and E.thread_varying(v)
               and interp._varying_data(v) and not self.olane(v))  # fmt: skip
        super().write(reg, v)
        if own:  # interp.scalarize: a table of the first warp's values
            self.own((reg,), self.cur.line if self.cur else 0)

    def _shared_load(self, i, mn: str) -> None:
        """ld.shared(::cluster), ldmatrix: data by the class of the address (module doc;
        assumed written before a synchronization every reader passed), a strong ld.shared
        as a strong global load."""
        op = i.operands[1]
        try:
            a = cluster._address(self, op) if op[0] == "mem" else None
        except interp.Unsupported:
            a = None
        if ops.strong(i.opcode):
            self.strong_load(i, a)
            return
        ka = self.klass(a)  # an address of unknown provenance (plain Data): t
        var = ka[0]
        if mn.startswith("ldmatrix"):  # each lane receives its own part of the tile
            var = "l" if var in ("u", "l") else "t"
        cls = (var, False, _NONE, var == "u" and ka[3], ka[4])
        for r in R.dests(i):
            self.env[r] = self._val(f"{i.opcode} at {self.where(i)}", cls, (r,))

    def predicated(self, i, cls) -> None:
        reg, neg = i.guard
        p = self.env.get(reg)
        mn = ops.mnemonic(i.opcode)
        if p is None or isinstance(p, interp.Data):
            p = p if p is not None else NbData(f"undefined {reg}", "t", False)
            if is_named(mn):
                pc = self.klass(p)
                if pc[0] == "u":
                    key = self._key(i.line, pc[3])
                    self._rep(True, f"{i.opcode} under the predicate {reg}, data uniform "
                                    f"over the role ({p.origin})", key, ((reg,), _NONE))  # fmt: skip
                    self.execute(i, cls)
                    self._rep(False, "", key)
                    return
                self.div.append(("cond", pc[1], f"under the predicate {reg}, data that may "
                                                f"differ between the participants ({p.origin})",
                                 pc[1]))  # fmt: skip
                try:
                    self.execute(i, cls)
                finally:
                    self.div.pop()
                return
            old = {r: self.env.get(r) for r in R.dests(i)}
            self.execute(i, cls)
            for r, x in old.items():
                self.env[r] = self._select(p, self.env.get(r), x, r)
            return
        if interp._carries(mn):
            self.fail(i, "a carry-flag instruction under a predicate")
        super().predicated(i, cls)

    # ------------------------------------------------------- named barriers
    def named_bar(self, i, mn: str) -> None:
        """A named barrier instruction (module doc): a named_bar event of the whole role
        under the uniform part of the mask, or an nb_open."""
        if id(i) in self.ctx["forks"]:
            return  # the warp-specialize dispatch's fork barrier (cfg.py protocol)
        red = mn.startswith(_RED)
        n = len(i.operands) - (1 if red else 0)
        k0 = 1 if red else 0
        vals, bad = [], None
        for k, what in ((k0, "id"), (k0 + 1, "thread count")):
            if k >= n:
                vals.append(self.k.block[0])
                continue
            e = self.operand(i, k, 32)
            u = None if isinstance(e, interp.Data) else self.uniform(e)
            if u is None and not isinstance(e, interp.Data) and self.per_warp(e):
                self.split_per_warp(e)  # one role per run of warps with one value
            if u is None or not E.is_const(u):
                vals.append(None)
                bad = bad or (what, e)
                continue
            vals.append(u[2])
        aligned = mn.startswith("bar.") or mn.endswith(".aligned")
        if self.div:
            kind, certain, why = self._div_reason(aligned)
            self._open(i, mn, kind, certain, why, vals)
            return
        if bad is not None:
            what, e = bad
            if isinstance(e, interp.Data):
                why = f"its {what} is data ({e.origin})"
            elif self.uniform(e) is None:
                why = f"its {what} differs between the lanes of a warp"
            else:
                why = f"its {what} is not a constant of the kernel"
            self._open(i, mn, "operand", False, why, vals)
            return
        bid, count = vals
        if count <= 0 or count % 32 or count > self.k.block[0]:
            self._open(i, mn, "operand", True,
                       f"count {count} is not a positive multiple of 32 at most the block's "
                       f"{self.k.block[0]} threads: no set of warps completes an instance", vals)  # fmt: skip
            return
        uni, var = self.split_mask()
        if not E.is_const(var, 1):
            if self.per_warp(var) is not None:
                self.split_per_warp(var)
            if self.active(var) == []:
                return
            if aligned:
                self._open(i, mn, "lane", self._lane_certain(var),
                           "reached by only some lanes of a warp (an aligned barrier needs "
                           "every thread of the warp)", vals)  # fmt: skip
            else:
                self._open(i, mn, "lane", False,
                           "reached by only some lanes of a warp (without .aligned each "
                           "thread arrives on its own, which the model does not count)", vals)  # fmt: skip
            return
        if self.ctx["dispatch"] and bid == 1:
            self._open(i, mn, "dispatch", False,
                       "barrier 1 is the warp-specialize dispatch's fork barrier, whose "
                       "protocol cfg.py checks; another use of it is not modeled", vals)  # fmt: skip
            return
        stmt = {"op": "event", "ev": "named_bar", "id": interp._c32(bid),
                "count": interp._c32(count), "sync": _op(mn) != "arrive",
                "loc": self.loc(i)}  # fmt: skip
        if n < k0 + 2:  # no thread count: threads that exit release it (PTX ISA exit)
            stmt["all"] = True
        self._site(i, mn, "counted", bid, count)
        self._emit_under(uni, stmt)

    def _div_reason(self, aligned: bool) -> tuple:
        """(kind, certain, why) of the enclosing divergent regions: certain if every one
        is (a barrier without .aligned: every one by data, as the lanes of a warp may
        reach it at different instructions), with the innermost reason; else the
        innermost one that is not."""
        for d in reversed(self.div):
            if not (d[1] if aligned else d[3]):
                return d[0], False, d[2]
        return self.div[-1][:3]

    def _lane_certain(self, e) -> bool:
        """The lanes of some warp of the role certainly see different values of e: they
        are constants that differ (an elect symbol picks one lane)."""
        if interp._has_elect(e):
            return True
        for w in range(self.w0, self.w0 + self.nw):
            xs = self._warp_instances(e, w)
            if all(E.is_const(x) for x in xs) and len({x[2] for x in xs}) > 1:
                return True
        return False

    def _site(self, i, mn, status, bid, count, kind=None, certain=None, why=None):
        self.sites.append({"loc": self.loc(i), "ptx_line": i.line, "role": self.name,
                           "w0": self.w0, "warps": self.nw, "op": _op(mn), "id": bid,
                           "count": count, "status": status, "kind": kind,
                           "certain": certain, "why": why, "rep": self.reps > 0})  # fmt: skip

    def _open(self, i, mn, kind, certain, why, vals) -> None:
        """An nb_open event: a barrier P0-6 cannot count (module doc); not certain inside
        a repeated region, which may run no pass."""
        certain = bool(certain) and self.reps == 0
        what = f"{i.opcode} {', '.join(str(o[1]) for o in i.operands)}"
        why = f"{what} at PTX line {i.line}: {why}"
        stmt = {"op": "event", "ev": "nb_open", "kind": kind, "certain": certain,
                "why": why, "sync": _op(mn) != "arrive", "loc": self.loc(i),
                "id": None if vals[0] is None else interp._c32(vals[0]),
                "count": None if vals[1] is None else interp._c32(vals[1])}  # fmt: skip
        self._site(i, mn, "open", vals[0], vals[1], kind, certain, why)
        self._emit_under(self.split_mask()[0], stmt)

    def _rep(self, begin: bool, why: str, key: str = "", spec=None) -> None:
        """An nb_rep event; a keyed region of a split role records what its path
        signature reads (spec: see signature; None: a path of this role alone)."""
        self.reps += 1 if begin else -1
        ev = {"op": "event", "ev": "nb_rep", "begin": begin, "why": why, "key": key}
        self.emit(ev)
        if begin and key and self.track:
            self.keyed.append((ev, spec or ((), {(0, f"w{self.w0}")})))

    # ---------------------------------------------------------------- ifs
    def run_if(self, node) -> None:
        """An If (_run_if); in a split role, what its merged registers and its loop's
        trip count are computed from: its condition, or, where this role resolves the
        condition to a constant, that decision (_run_arm), and an own decision where it
        is one value over this role's threads but not over the role before splitting."""
        if not self.track:
            self._run_if(node, None)
            return
        reg, neg = node.branch.guard
        c = self.env.get(reg)
        exact = c is not None and not isinstance(c, interp.Data)
        if exact and neg:
            c = E.not1(c)
        cu = self.uniform(c) if exact else None
        side = int(cu[2] != 0) if cu is not None and E.is_const(cu) else None
        dec = None
        if cu is not None and side is None and not self.ouniform(c):
            dec = (node.branch.line, f"w{self.w0}")
        if (
            self._exits
            and _controls(node)
            and (interp._exits(node.then) or interp._exits(node.else_))
        ):
            ex = self._exits[-1]
            ex["roots"].add(reg)
            ex["path"] |= self._ctl[-1] | {
                d
                for d in (dec, None if side is None else (node.branch.line, side))
                if d
            }
        self._run_if(node, dec)
        if side is None:
            for r in self._arm_defs(node):
                self._dep(r, (reg,), {dec} if dec else _NONE)

    def _run_if(self, node, dec) -> None:  # noqa: C901 (one case per condition class)
        reg, neg = node.branch.guard
        self.cur = node.branch
        c = self.env.get(reg)
        if c is None:
            c = NbData(f"undefined {reg}", "t", False)
        elif neg and not isinstance(c, interp.Data):
            c = E.not1(c)
        top = self.nb[-1] if self.nb else None
        exits = interp._exits(node.then) or interp._exits(node.else_)
        bars = self._bars(node.then) or self._bars(node.else_)
        exact = not isinstance(c, interp.Data)
        cu = self.uniform(c) if exact else None
        if exits and top is not None and top["mode"] == "exact":
            if cu is not None:
                self._exact_if(node, cu, dec)
                return
            if exact and (bars or top["bars"]) and self.per_warp(c):
                self.split_per_warp(c)
            raise _Fallback()
        if exits and top is not None and _controls(node):
            if exact and top["bars"] and cu is None and self.per_warp(c):
                self.split_per_warp(c)
            top["exits"].append((self.klass(c), self._certain(c), self._origin(c)))
        if not bars:
            self._value_if(node, c, dec)
            return
        if (
            cu is not None
            and not E.is_const(cu)
            and (self._in_class() or self._symbolic(cu))
        ):
            self._rep_if(node, cu, dec)  # no KIR `if` on a symbol, nor in a class loop
            return
        if cu is not None:
            self._exact_if(node, cu, dec)
            return
        if exact and self.per_warp(c):
            self.split_per_warp(c)
        if not exact and self.klass(c)[0] == "u":
            self._rep_if(node, c)
            return
        if exact:
            why = (f"inside a branch at PTX line {node.branch.line} on a condition that "
                   "differs between the lanes of a warp")  # fmt: skip
            self._div_if(node, c, ("lane", self._certain(c), why, False))
            return
        why = (f"inside a branch at PTX line {node.branch.line} on data that may differ "
               f"between the participants ({c.origin})")  # fmt: skip
        self._div_if(node, c, ("cond", self._certain(c), why, self._certain(c)))

    def _run_arm(self, node, side: int) -> None:
        """The arm of an If whose condition is one constant over this role's threads; in
        a split role, a decision on the path of every register it writes."""
        if self.track:
            self._ctl.append(self._ctl[-1] | {(node.branch.line, side)})
        try:
            self.run(node.then if side else node.else_)
        finally:
            if self.track:
                self._ctl.pop()

    def _arm_defs(self, node) -> frozenset:
        """The registers an If's arms may write."""
        memo = self.ctx["defs"]
        if id(node) not in memo:
            memo[id(node)] = frozenset(
                r for i in _instrs([node.then, node.else_]) for r in R.dests(i)
            )
            self.ctx["keep"].append(node)  # id(node) stays this If's
        return memo[id(node)]

    def _exact_if(self, node, cu, dec=None) -> None:
        if E.is_const(cu):
            self._run_arm(node, int(cu[2] != 0))
        elif (
            self.nb
            and self.nb[-1]["mode"] == "exact"
            and (interp._exits(node.then) or interp._exits(node.else_))
        ):
            interp.Translator._uniform_if(
                self, node, cu
            )  # a symbol: the loop falls back
        else:
            self._uniform_if(node, cu, dec=dec)

    def _uniform_if(self, node, c, dec=None) -> None:
        """interp's KIR `if`; if its arms leave symbols to merge, the If is translated
        again with each arm an nb_rep region (barriers) or under the mask (values)."""
        if self._symbolic(c):
            raise _Symbolic()
        snap = self.snapshot()
        try:
            super()._uniform_if(node, c)
        except _Symbolic:
            self.restore(snap)
            if self._bars(node.then) or self._bars(node.else_):
                self._rep_if(node, c, dec)
            else:
                self._masked_if(node, c)

    def _branch(self, node, side: int, seq: list, env: dict):
        """interp's arm of a KIR `if`: a merged register may take a slot holding its value
        in this role's threads (_special)."""
        out = super()._branch(node, side, seq, env)
        if self.track:
            self.own([r for r in self._live(node) if self._special(out[1].get(r))],
                     node.branch.line)  # fmt: skip
        return out

    def _value_if(self, node, c, dec=None) -> None:
        """An If that decides no barrier: both arms for their values (under the mask), or,
        on an exact condition uniform over the role around a loop, interp's KIR `if`, so
        the loop may be exact (it may move a pipeline counter that a loop holding
        barriers carries)."""
        if not isinstance(c, interp.Data):
            cu = self.uniform(c)
            if cu is not None and E.is_const(cu):
                self._run_arm(node, int(cu[2] != 0))
                return
            if (cu is not None and not self._in_class() and not self._symbolic(cu)
                    and _has_loop([node.then, node.else_])):  # fmt: skip
                self._uniform_if(node, cu, dec=dec)
                return
            self._masked_if(node, c)
            return
        self._data_if(node, c)

    def _data_if(self, node, c) -> None:
        """Both arms unmasked (a data condition), registers merged by class."""
        env0, arms = self.env, []
        for side, seq in ((1, node.then), (0, node.else_)):
            self.env = dict(env0)
            self.regions.append((id(node), side))
            self.run(seq)
            self.regions.pop()
            arms.append(self.env)
        keys = self._guard(node, env0)
        self.env = dict(env0)
        self.cur = node.branch
        for r in self._live(node):
            x, y = arms[0].get(r), arms[1].get(r)
            if keys and self._accepted(x, keys) and self._accepted(y, keys):
                self.env[r] = self._published(r, keys, f"the spin at PTX line "
                                                       f"{node.branch.line}")  # fmt: skip
                continue
            if keys and self._spun(x, keys) and self._spun(y, keys):
                self.env[r] = _derived(r, f"the spin at PTX line {node.branch.line}",
                                       x.srcs | y.srcs)  # fmt: skip
                continue
            self.env[r] = self._select(c, x, y, r)

    def _guard(self, node, env0) -> frozenset:
        """The addresses of a spin whose loop LLVM rotated: `if accept(v0) skip, else
        loop { v = load } until accept(v)`, the If testing the first load v0 of the same
        addresses with the same acceptance test; empty otherwise."""
        arms = (node.then, node.else_)
        bare = [not any(True for _ in _instrs([a])) for a in arms]
        if bare.count(True) != 1:
            return _NONE
        body = [n for n in arms[bare.index(False)]
                if not (isinstance(n, R.Block) and not n.instrs)]  # fmt: skip
        spin = self.ctx["spins"].get(body[0].header) if (
            len(body) == 1 and isinstance(body[0], R.Loop)) else None  # fmt: skip
        sig = self._accept(node.block, node.branch, bare[0])
        if spin is None or sig is None or sig[:2] != spin["sig"][:2]:
            return _NONE
        if sig[3] != spin["sig"][3] or (
            sig[3][0] == "reg" and sig[3][1] in spin["defs"]
        ):
            return _NONE
        v0 = env0.get(sig[2])
        return spin["keys"] if self._accepted(v0, spin["keys"], loads=True) else _NONE

    def _accepted(self, v, keys, loads=False) -> bool:
        """v is a value published at `keys`, or a strong load of those addresses as
        loaded (a copy of it included); `loads`: any value computed from such loads (the
        value a rotated spin's guard tests)."""
        if isinstance(v, NbData) and v.pub and v.pub <= keys and not loads:
            return True
        if isinstance(v, tuple) and v[0] == "var" and not loads:
            return self.ctx["pub"].get(v[2], {None}) <= keys
        return (isinstance(v, NbData) and v.nondet and (v.raw or loads)
                and self._spun(v, keys))  # fmt: skip

    def _spun(self, v, keys) -> bool:
        """v is data computed from strong loads of the addresses `keys` (and other
        values)."""
        if not (isinstance(v, NbData) and v.srcs):
            return False
        ks = {self.ctx["loads"].get(s, (None, None))[1] for s in v.srcs}
        return None not in ks and ks <= keys

    def _accept(self, block, branch, taken: bool):
        """(comparison, type, register, comparand) under which control leaves on the
        taken (or else the fall-through) side of `@[!]p bra` at the end of `block`, p a
        setp of that block on an integer register; else None."""
        reg, neg = branch.guard
        instrs = self.k.mem.cfg.blocks[block].instrs
        pos = next((k for k, x in enumerate(instrs) if x is branch), None)
        setp: parse.Instr | None = next(
            (x for x in reversed(instrs[: pos or 0]) if reg in R.dests(x)), None
        )
        if setp is None:
            return None
        parts = setp.opcode.split(".")
        if len(parts) != 3 or parts[0] != "setp" or parts[1] not in _NEG:
            return None
        if len(setp.operands) != 3 or setp.operands[1][0] != "reg":
            return None
        when = _NEG[parts[1]] if neg else parts[1]
        cmp = when if taken else _NEG[when]
        typ = "b" + parts[2][1:] if cmp in ("eq", "ne") else parts[2]
        return cmp, typ, setp.operands[1][1], setp.operands[2]

    def _div_if(self, node, c, reason) -> None:
        """A divergent If around barriers: both arms, every barrier in them an nb_open."""
        self.div.append(reason)
        try:
            if isinstance(c, interp.Data):
                self._data_if(node, c)
            else:
                self._masked_if(node, c)
        finally:
            self.div.pop()

    def _rep_if(self, node, c, dec=None) -> None:
        """An If on a condition uniform over the role (class-u data, or a symbol) around
        barriers: each arm an nb_rep region (taken 0 or 1 times, alike in every warp).
        `dec`: run_if's own decision (c is this role's instance of the condition)."""
        env0, arms = self.env, []
        what = (
            c.origin if isinstance(c, interp.Data) else "a value the model does not fix"
        )
        why = (f"a branch at PTX line {node.branch.line} on a condition uniform over the "
               f"role ({what}): taken or not, alike in every warp")  # fmt: skip
        key = self._key(node.branch.line, self.klass(c)[3])
        spec = ((node.branch.guard[0],), {dec} if dec else _NONE)
        for side, seq in ((1, node.then), (0, node.else_)):
            self.env = dict(env0)
            mark = self.push_scope()
            self.regions.append((id(node), side))
            rep = self._bars(seq)
            if rep:
                self._rep(True, why, f"{key}/{side}" if key else "", spec)
            self.run(seq)
            self.flush()
            if rep:
                self._rep(False, why, f"{key}/{side}" if key else "")
            self.regions.pop()
            self.pop_scope(mark)
            arms.append(self.env)
        self.env = dict(env0)
        self.cur = node.branch
        for r in self._live(node):
            self.env[r] = self._select(c, arms[0].get(r), arms[1].get(r), r)

    def _certain(self, c) -> bool:
        """The model admits participants that see different values of the condition c:
        nondet data, or thread-index constants that differ."""
        if isinstance(c, interp.Data):
            return self.klass(c)[1]
        return self.uniform(c) is None and self._lane_certain(c)

    @staticmethod
    def _origin(c) -> str:
        return c.origin if isinstance(c, interp.Data) else "a thread-index condition"

    # -------------------------------------------------------------- loops
    def run_loop(self, node) -> None:
        """A loop (_run_loop); in a split role, the registers it writes are computed from
        its exits (their conditions, the decisions around them), and a carried register
        whose value is one only over this role's threads (_special) takes a slot of this
        role alone."""
        if not self.track:
            self._run_loop(node)
            return
        head = self.k.mem.first_instr(node.header)
        line = head.line if head else 0
        live = self.tree.live_in.get(node.header, frozenset())
        defs = self.tree.defs[node.header]
        self.own(
            [r for r in defs if r in live and self._special(self.env.get(r))], line
        )
        ex = {"roots": set(), "path": set()}
        self._exits.append(ex)
        try:
            self._run_loop(node)
        finally:
            self._exits.pop()
        for r in defs:
            self._dep(r, ex["roots"], ex["path"])

    def _run_loop(self, node) -> None:
        bars = self._bars(node.body)
        if E.is_const(self.mask, 1):
            snap = self.snapshot()
            hoisted = self.hoist[-1][0] if self.hoist else []
            nh = len(hoisted)
            self.nb.append({"mode": "exact", "bars": bars})
            try:
                super().run_loop(node)
                return
            except (_Fallback, interp.Unsupported):
                pass
            finally:
                self.nb.pop()
            self.restore(snap)
            del hoisted[nh:]
        self._class_loop(node, bars)

    def end_iteration(self, brk: bool) -> None:
        top = self.nb[-1] if self.nb else None
        if top is not None and top["mode"] == "class":
            edge = top["outs" if brk else "backs"]
            kinds = top["exitk" if brk else "backk"]  # what _class_loop publishes
            for r in top["carried"]:
                v = self.env.get(r)
                edge[r] = _join(edge.get(r, _UO), self.klass(v))
                kind = (
                    "raw" if _raw(v) else "header" if v is top["start"].get(r) else ""
                )
                kinds[r] = kinds.get(r, frozenset()) | {kind}
            return
        carried, slots, _go, kinds = self.loops[-1][:4]
        self.own([r for r in carried if self._special(self.env.get(r))],
                 self.cur.line if self.cur else 0)  # fmt: skip
        for k, r in enumerate(carried):
            v = self.env.get(r)
            if slots[k] is not None and (v is None or isinstance(v, interp.Data)):
                kinds[k] = "data"  # a register carried as data, read in the loop
                raise interp._Widen()
        super().end_iteration(brk)

    def _class_round(self, node, written, classes, live, why, mode=None, key="",
                     spec=None):  # fmt: skip
        """One pass of a loop body with the written registers at their classes (a
        register uniform over the role is a fresh symbol, so values computed from it in
        the body stay exact); mode: None (statements dropped by the caller), "rep" (key,
        spec: _rep) or a divergence reason."""
        self.ctx["nsym"] += 1
        for r in written:
            w = _width(self.k.fn.regs.get(r, ""))
            if classes[r][:3] == _U and w:
                name = f"nb~{self.ctx['nsym']}:{r}"
                if classes[r][3]:
                    self.ctx["sym_ou"].add(name)
                self.env[r] = E.var(name, w)
            else:
                self.env[r] = NbData(f"{r} carried by {why}", *classes[r])
        carried = [r for r in written if r in live]
        top = {"mode": "class", "bars": self._bars(node.body), "exits": [], "backs": {},
               "outs": {}, "exitk": {}, "backk": {}, "written": written,
               "start": {r: self.env[r] for r in carried}, "carried": carried}  # fmt: skip
        self.nb.append(top)
        self.loops.append([written, [None] * len(written), None, [None] * len(written),
                           [None] * len(written), None, live, [None] * len(written), {}])  # fmt: skip
        if mode not in (None, "rep"):
            self.div.append(mode)
        mark = self.push_scope()
        if mode == "rep":
            self._rep(True, why, key, spec)
        try:
            self.run(node.body, inherit=False)
            self.flush()
        finally:
            self.pop_scope(mark)
            self.loops.pop()
            self.nb.pop()
            if mode not in (None, "rep"):
                self.div.pop()
        if mode == "rep":
            self._rep(False, why, key)
        return top

    def _class_loop(self, node, bars: bool) -> None:
        """A loop that is not exact (module doc): its classes to a fixed point; with a
        barrier inside, an nb_rep region or a divergent region."""
        head = self.k.mem.first_instr(node.header)
        why = f"the loop at {self.where(head)}"
        written = sorted(self.tree.defs[node.header])
        live = self.tree.live_in.get(node.header, frozenset()) | self.tree.live_in.get(
            node.exit, frozenset())  # fmt: skip
        classes = {
            r: self.klass(self.env.get(r)) if r in live else _DEAD for r in written
        }
        # a strong load's value as loaded, or (at the entry) one value over the role: at
        # an exit, either one is what the spin accepted (_spin)
        entry = {r: self.env.get(r) for r in written if r in live}
        snap = self.snapshot()
        hoisted = self.hoist[-1][0] if self.hoist else []
        nh = len(hoisted)
        for _round in range(_ROUNDS):
            top = self._class_round(node, written, classes, live, why)
            self.restore(snap)
            del hoisted[nh:]
            new = {r: _join(classes[r], top["backs"].get(r, _UO)) for r in written}
            if new == classes:
                break
            classes = new
        else:
            self.fail(head, f"P0-6: no class fixed point for {why}", "error")
        exits = [e for e in top["exits"] if e[0][0] != "u"]
        ex = _join(*(e[0] for e in exits)) if exits else _UO
        if bars and not exits:
            ou = all(len(e[0]) > 3 and e[0][3] for e in top["exits"])
            self._class_round(node, written, classes, live,
                              f"{why} repeats a number of times the model does not "
                              "evaluate, the same in every warp", "rep",
                              self._key(head.line if head else 0, ou),
                              self._exits[-1] if self._exits else None)  # fmt: skip
        elif bars:
            certain = any(e[1] for e in exits)
            data = any(e[1] and e[0][1] for e in exits)  # nondet data, not lanes alone
            origin = next((e[2] for e in exits if e[1]), exits[0][2])
            self._class_round(node, written, classes, live, why,
                              ("loop", certain, f"inside {why}, whose exit depends on "
                               f"{origin}: the participants may leave it in different "
                               "iterations", data))  # fmt: skip
        spin = self._spin(node, top["exits"])
        keys = frozenset(self.ctx["loads"][s][1] for s in spin)
        if spin:
            self._record_spin(node, keys)
        for r in written:
            out = _join(classes[r], top["outs"].get(r, _UO))
            raw = self._raw_exits(r, top, entry)
            if spin and out[1] and out[2] and out[2] <= spin:
                # the accepted value itself; a value computed from it in the loop is
                # data (from the accepted value, or from an earlier pass's load)
                self.env[r] = (self._published(r, keys, f"the spin {why}")
                               if raw else _derived(r, f"the spin {why}", out[2]))  # fmt: skip
                continue
            v = self._val(f"{r} after {why}", _join(out, ex), (r,))
            # a strong load's value as loaded at every exit (a rotated spin, _guard)
            self.env[r] = (
                dataclasses.replace(v, raw=True) if raw and isinstance(v, NbData) else v
            )

    def _raw_exits(self, r, top, entry) -> bool:
        """Every exit of a loop's last class round leaves r holding a strong load's value
        as loaded: loaded in that pass, or r's value at the header if every back edge
        leaves it so and its entry value is one too or one value over the role (a spin
        that tests its entry value first, _spin)."""
        exits, backs = (
            top["exitk"].get(r, frozenset()),
            top["backk"].get(r, frozenset()),
        )
        if not exits or "" in exits:
            return False
        if "header" not in exits:
            return True
        v = entry.get(r)
        return backs <= {"raw", "header"} and (
            _raw(v) or self.klass(v)[:2] == ("u", False)
        )

    def _published(self, reg, keys, what: str):
        """The value of reg a spin accepted (module doc "published values")."""
        ou = all(self.ouniform(k) for k in keys)
        v = self._val("", _U + (ou, ou), (reg,))
        if isinstance(v, NbData):
            return NbData(f"{reg}, the value accepted by {what} (a published value)",
                          "u", ou=ou, pub=keys)  # fmt: skip
        self.ctx["pub"][v[2]] = keys
        return v

    def _record_spin(self, node, keys) -> None:
        """Note a spin loop's acceptance test (its one exit If) for _guard."""
        ifs = [n for n in node.body if isinstance(n, R.If) and _controls(n)]
        if len(ifs) != 1:
            return
        out = _outcomes(ifs[0].then)
        if out not in ({"break"}, {"continue"}):
            return
        sig = self._accept(ifs[0].block, ifs[0].branch, out == {"break"})
        if sig is not None:
            self.ctx["spins"][node.header] = {"keys": keys, "sig": sig,
                                              "defs": self.tree.defs[node.header]}  # fmt: skip

    def _spin(self, node, exits) -> frozenset:
        """The strong loads a spin loop accepted: the sources of its exit conditions, if
        one of them is re-read in the loop and all are at addresses uniform over the role
        (module doc "published values"); else empty."""
        srcs = frozenset().union(*(e[0][2] for e in exits)) if exits else _NONE
        if not srcs:
            return _NONE
        mine = self.ctx["loop_instrs"](node)
        if not srcs & mine:
            return _NONE
        loads = self.ctx["loads"]
        if any(
            loads.get(s, ("t", None))[0] != "u" or loads[s][1] is None for s in srcs
        ):
            return _NONE
        return srcs

    # ------------------------------------------------------------- helpers
    def _bars(self, seq) -> bool:
        key = id(seq)
        memo = self.ctx["bars"]
        if key not in memo:
            memo[key] = any(
                is_named(ops.mnemonic(i.opcode)) and id(i) not in self.ctx["forks"]
                for i in _instrs([seq])
            )
            self.ctx["keep"].append(seq)  # id(seq) stays this list's
        return memo[key]


def _op(mn: str) -> str:
    return "red" if mn.startswith(_RED) else "sync" if ".sync" in mn else "arrive"


def _raw(v) -> bool:
    return isinstance(v, NbData) and v.raw


def _derived(reg: str, what: str, srcs) -> NbData:
    """reg after a spin (`what`) where it holds a value computed from the strong loads
    it re-reads, not the accepted value itself: data, but no certain divergence (from the
    accepted value it is the same in every reader under assumption 4 of the module doc,
    from an earlier pass's load it is not)."""
    return NbData(f"{reg} after {what}, computed from the values its loads read", "t",
                  False, frozenset(srcs))  # fmt: skip


def _width(typ: str) -> int:
    """The width of an integer or predicate register type, else 0."""
    if typ == "pred":
        return 1
    if typ[:1] in ("b", "s", "u") and typ[1:].isdigit() and int(typ[1:]) <= 64:
        return int(typ[1:])
    return 0


def _instrs(seqs):
    """The instructions of region sequences, nested Ifs and loops included, in order."""
    for seq in seqs:
        for n in seq:
            if isinstance(n, R.Block):
                yield from n.instrs
            elif isinstance(n, R.If):
                yield from _instrs([n.then, n.else_])
            elif isinstance(n, R.Loop):
                yield from _instrs([n.body])


def _has_loop(seqs) -> bool:
    """Some statement list of seqs holds a loop (at any depth)."""
    for seq in seqs:
        for n in seq:
            if isinstance(n, R.Loop):
                return True
            if isinstance(n, R.If) and _has_loop([n.then, n.else_]):
                return True
    return False


def _outcomes(seq) -> set:
    """How a sequence of a loop body may end: "break", "continue" or "fall" (to the code
    after it)."""
    acc: set = set()
    for n in seq:
        if n == R.CONTINUE:
            return acc | {"continue"}
        if n == R.BREAK:
            return acc | {"break"}
        if isinstance(n, R.If):
            o = _outcomes(n.then) | _outcomes(n.else_)
            acc |= o - {"fall"}
            if "fall" not in o:
                return acc
    return acc | {"fall"}


def _controls(node) -> bool:
    """An If with loop edges in its arms decides the loop's iterations: its arms may end
    differently (or one of them in several ways, decided by a nested condition)."""
    a, b = _outcomes(node.then), _outcomes(node.else_)
    return a != b or len(a) > 1


# ---------------------------------------------------------------------------- CFG


def _cfg(fn, split: list):
    """cfg.build, with cycles entered other than at their header made reducible by node
    splitting (module doc); the copied blocks are appended to `split`."""
    try:
        return C.build(fn)
    except C.CfgError as e:
        if "irreducible control flow" not in str(e):
            raise
    blocks, succ = _blocks(fn)
    entry = next(iter(blocks))
    for _round in range(_SPLIT_BUDGET):
        found = _irreducible(fn, entry, succ)
        if found is not None:
            cyc, entries, preds = found
            _split_entries(blocks, succ, cyc, entries, preds, split)
        elif not _absorb_exits(fn, entry, blocks, succ, split):
            break
    else:
        raise C.CfgError(f"{fn.name}: irreducible control flow beyond the node "
                         "splitting budget")  # fmt: skip
    rpo = C._rpo(fn, entry, succ)
    keep = {n: blocks[n] for n in blocks if n in set(rpo)}
    for n, b in keep.items():
        b.succs, b.preds = succ[n], []
    for n, b in keep.items():
        for s in dict.fromkeys(b.succs):
            keep[s].preds.append(n)
    cfg = C.Cfg(fn, keep, C._idom(keep, rpo), {})
    index = {b: k for k, b in enumerate(rpo)}
    for b in rpo:
        for s in dict.fromkeys(keep[b].succs):
            if index[s] <= index[b]:
                if not cfg.dominates(s, b):
                    raise C.CfgError(f"{fn.name}: irreducible control flow after node "
                                     f"splitting at {cfg.where(b)}")  # fmt: skip
                cfg.loops.setdefault(s, set()).update(C._natural_loop(keep, s, b))
    C._check_shape(cfg)
    return cfg


def _absorb_exits(fn, entry, blocks, succ, split) -> bool:
    """One loop that exits to several blocks, one of them the start of a path back to
    its header (LLVM rotates a loop's test into one path of its body, so that path
    leaves the natural loop and re-enters it through the old test): copy that path, if it
    only reaches the header or the loop's other exits, into the loop. True if a loop
    changed."""
    rpo = C._rpo(fn, entry, succ)
    nodes = {b: C.Block(b, [], succ[b], []) for b in rpo}
    for b in rpo:
        for x in dict.fromkeys(succ[b]):
            nodes[x].preds.append(b)
    idom = C._idom(nodes, rpo)
    index = {b: k for k, b in enumerate(rpo)}
    loops: dict = {}
    for b in rpo:
        for x in dict.fromkeys(succ[b]):
            if index[x] <= index[b] and _dominates(idom, entry, x, b):
                loops.setdefault(x, set()).update(C._natural_loop(nodes, x, b))
    for h, body in loops.items():
        outs = {x for u in body for x in succ[u] if x not in body}
        if len(outs) < 2:
            continue
        for t in sorted(outs, key=index.__getitem__):
            path = _back_path(succ, t, h, body, outs - {t})
            if path is None:
                continue
            k = sum(1 for b in blocks if "~" in b) + 1
            copy = {b: f"{b}~{k}" for b in path}
            for b in path:
                blocks[copy[b]] = C.Block(copy[b], blocks[b].instrs)
                succ[copy[b]] = tuple(copy.get(x, x) for x in succ[b])
                split.append(copy[b])
            for u in body:
                succ[u] = tuple(copy[t] if x == t else x for x in succ[u])
            return True
    return False


def _back_path(succ, t, h, body, others):
    """The blocks reachable from t without passing h or `others`, if all of them stay
    outside the loop body, none is the ret, and there are at most 64 (cycles among them
    become loops nested in the loop), else None."""
    path, todo = [t], [t]
    while todo:
        for x in succ[todo.pop()]:
            if x == h or x in others or x in path:
                continue
            if x is None or x in body or not succ.get(x) or len(path) >= 64:
                return None  # the loop body, ret, or too far: a real exit
            path.append(x)
            todo.append(x)
    return path


def _dominates(idom, entry, a, b) -> bool:
    while b != a:
        if b == entry:
            return False
        b = idom[b]
    return True


def _blocks(fn):
    """The blocks and successors of cfg.build (its first part)."""
    blocks, alias, cur = {}, {}, None
    for x in fn.body:
        if isinstance(x, parse.Label):
            if cur is not None and not cur.instrs and "#" not in x.name:
                alias[x.name] = cur.name
                continue
            cur = blocks[x.name] = C.Block(x.name, [])
        else:
            if cur is None or (cur.instrs and C._ends(cur.instrs[-1])):
                cur = blocks[f"@{x.line}"] = C.Block(f"@{x.line}", [])
            cur.instrs.append(x)
    names = list(blocks)
    succ = {n: C._succs(fn, blocks, alias, n, names[k + 1] if k + 1 < len(names) else None)
            for k, n in enumerate(names)}  # fmt: skip
    for n in names:
        last = blocks[n].instrs[-1] if blocks[n].instrs else None
        if last is not None and last.base == "trap" and last.guard is None:
            preds = [m for m, s in succ.items() if n in s]
            other = set(succ[preds[0]]) - {n} if len(preds) == 1 else set()
            if len(other) != 1:
                raise C.CfgError(f"{fn.name}: trap at line {last.line} is not reached by "
                                 "one conditional branch")  # fmt: skip
            succ[n] = (other.pop(),)
    return blocks, succ


def _irreducible(fn, entry, succ) -> tuple | None:
    """(cycle, its entries in reverse postorder, predecessors) of a cycle entered at
    several blocks (loop nesting forest by recursive strongly connected components:
    a component entered at one block is a loop; its body without that block is searched
    again), or None."""
    rpo = C._rpo(fn, entry, succ)
    index = {b: k for k, b in enumerate(rpo)}
    preds: dict = {b: set() for b in rpo}
    for b in rpo:
        for x in succ[b]:
            preds[x].add(b)

    def visit(nodes) -> tuple | None:
        for comp in _sccs(nodes, succ, index):
            cyc = set(comp)
            if len(comp) == 1 and comp[0] not in succ[comp[0]]:
                continue
            entries = sorted((n for n in cyc if n == entry or preds[n] - cyc),
                             key=index.__getitem__)  # fmt: skip
            if len(entries) > 1:
                return cyc, entries, preds
            inner = visit(cyc - {entries[0]})
            if inner is not None:
                return inner
        return None

    return visit(set(rpo))


def _sccs(nodes: set, succ: dict, index: dict) -> list:
    """The strongly connected components of the subgraph on `nodes` (iterative Tarjan,
    roots in reverse postorder)."""
    num, low, stack, on, out = {}, {}, [], set(), []

    def push(v):
        num[v] = low[v] = len(num)
        stack.append(v)
        on.add(v)
        return v, iter([x for x in succ[v] if x in nodes])

    for root in sorted(nodes, key=index.__getitem__):
        if root in num:
            continue
        work = [push(root)]
        while work:
            v, it = work[-1]
            for w in it:
                if w not in num:
                    work.append(push(w))
                    break
                if w in on:
                    low[v] = min(low[v], num[w])
            else:
                work.pop()
                if work:
                    u = work[-1][0]
                    low[u] = min(low[u], low[v])
                if low[v] == num[v]:
                    comp = []
                    while True:
                        w = stack.pop()
                        on.discard(w)
                        comp.append(w)
                        if w == v:
                            break
                    out.append(comp)
    return out


def _split_entries(blocks, succ, cyc, entries, preds, split) -> None:
    """Make the cycle entered at one block: its first entry in reverse postorder is the
    header; for every other entry e, the blocks of the cycle reachable from e without
    passing the header are copied, and e's predecessors outside the cycle enter the
    copy."""
    head = entries[0]
    for e in entries[1:]:
        part, todo = [e], [e]
        while todo:
            for x in succ[todo.pop()]:
                if x in cyc and x != head and x not in part:
                    part.append(x)
                    todo.append(x)
        k = sum(1 for b in blocks if "~" in b) + 1
        copy = {b: f"{b}~{k}" for b in part}
        for b in part:
            blocks[copy[b]] = C.Block(copy[b], blocks[b].instrs)
            succ[copy[b]] = tuple(copy.get(x, x) for x in succ[b])
            split.append(copy[b])
        for p in preds[e] - cyc:
            succ[p] = tuple(copy[e] if x == e else x for x in succ[p])


# ---------------------------------------------------------------------------- KIR


def document(facts: Facts, kernel: str, source: str, case_name: str, inputs: list,
             compiler: dict | None = None, tensors: list | tuple = (),
             tensor_params: dict | None = None) -> dict:  # fmt: skip
    """The P0-6 KIR (kir.md, change 14) of a variant: the Facts' roles and the params
    they read, one input per launch (`inputs`: {"name", "params" (a value for every param
    of the Facts and of `tensor_params`), "grid", "flags", "metadata"}), the metadata
    tensors the roles load (`tensors`: their KIR declarations, whose shape and stride
    params `tensor_params` names with their widths), and the classification of every
    barrier site under "nbar" (informational, for audit). KeyError names an input without
    a param's value or a metadata tensor's contents."""
    names = {**facts.params, **(tensor_params or {})}
    ins = []
    for x in inputs:
        missing = sorted(set(names) - set(x["params"]))
        missing += sorted(
            f"metadata {t}" for t in facts.meta if t not in x.get("metadata", {})
        )
        if missing:
            raise KeyError(f"input {x['name']} has no value for {missing}")
        ins.append({"name": x["name"], "params": {n: x["params"][n] for n in names},
                    "grid": list(x["grid"]), "flags": dict(x.get("flags", {})),
                    **({"metadata": {t: x["metadata"][t] for t in facts.meta}}
                       if facts.meta else {})})  # fmt: skip
    keys = ("role", "op", "id", "count", "loc", "ptx_line", "status", "kind", "certain",
            "rep", "why")  # fmt: skip
    doc = {
        "kir_version": 1,
        "kernel": kernel,
        "source": source,
        "ir": "ptx",
        "grid": [{"e": "const", "w": 32, "v": g} for g in ins[0]["grid"]],
        "params": [{"name": n, "type": f"i{w}"} for n, w in names.items()],
        "tensors": list(tensors),
        "barriers": [],
        "roles": [
            {
                "name": r["name"],
                "warps": r["warps"],
                "threads_per_issue": 1,
                "body": r["body"],
            }
            for r in facts.roles
        ],  # fmt: skip
        "nbar": {
            "block": list(facts.block),
            "dispatch": facts.dispatch,
            "split_blocks": len(facts.split),
            "sites": [{k: x[k] for k in keys} for x in facts.sites],
        },  # fmt: skip
        "case": {"name": case_name, "inputs": ins},
    }
    if facts.target in ("sm_100", "sm_103"):
        doc["target"] = facts.target
    if compiler:
        doc["compiler"] = compiler
    if tuple(facts.cluster) != (1, 1, 1):
        doc["cluster"] = list(facts.cluster)
    return doc


# ------------------------------------------------------------------------- driver


def translate(ptx_text: str, entry: str, launch_records: list, pol: dict,
              module=None) -> Facts:  # fmt: skip
    """The P0-6 Facts of one kernel (module doc); a refusal stops the variant's P0-6.
    `module`: ptx_text already parsed (never modified: the pass works on a copy of the
    entry function)."""
    facts = Facts(entry)
    try:
        _translate(facts, ptx_text, entry, launch_records, pol, module)
    except interp.Unsupported as e:
        facts.refusals.append({"kind": e.kind, "msg": str(e)})
    except (C.CfgError, R.RegionError) as e:
        facts.refusals.append({"kind": "gap", "msg": str(e)})
    except Exception as e:  # a PtxSyntaxError, bad inputs or a bug: refused, not raised
        facts.refusals.append({"kind": "error", "msg": f"{type(e).__name__}: {e}"})
    if facts.refusals:
        facts.roles, facts.sites, facts.params = [], [], {}
    return facts


def _translate(facts, text, entry, records, pol, module=None) -> None:
    module = module or parse.parse(text)
    fn = module.function(entry)
    # a copy, so that the caller's module stays as it is
    fn = dataclasses.replace(fn, regs=dict(fn.regs))
    switches = C.lower_switches(fn)  # as the P0 translation: a switch's compare chain
    facts.target = module.target.removesuffix("a")
    facts.block, facts.cluster = T._launch_dims(fn, records)
    cfg = _cfg(fn, facts.split)
    roles, dispatch = C.roles(cfg, pol["family"])
    locs, srclocs = T._locs(fn, pol)
    shared = {v.name for v in module.vars if v.space == "shared"}
    align = {v.name: v.align for v in module.vars + fn.vars if v.space == "shared"}
    written = T.writes(text, entry, pol["params"], ignore=_shared_only, narrow=True,
                       module=module)  # fmt: skip
    mem = _Memory(cfg, pol["params"], pol["tensors"],
                  {**pol, "shared_symbols": shared, "records": records, "target": facts.target,
                   "shared_align": align, "dispatch": False,
                   "written": None if written is None else set(written)})  # fmt: skip
    facts.written = written
    k = interp.Kernel(fn, mem, locs, facts.block, facts.cluster, pol["max_iters"],
                      split=True, vectorize=False, srclocs=srclocs,
                      switches=switches)  # fmt: skip
    facts.dispatch = dispatch is not None
    forks = {id(i) for i in fn.instructions() if facts.dispatch and C._is_fork(i)}
    loop_ids: dict = {}

    def loop_instrs(node):
        key = node.header
        if key not in loop_ids:
            loop_ids[key] = frozenset(id(i) for b in node.blocks
                                      for i in cfg.blocks[b].instrs)  # fmt: skip
        return loop_ids[key]

    ctx = {"forks": forks, "dispatch": facts.dispatch, "bars": {}, "defs": {},
           "keep": [], "loads": {}, "spins": {}, "pub": {}, "loop_instrs": loop_instrs,
           "nsym": 0, "sym_ou": set()}  # fmt: skip
    nwarps = facts.block[0] // 32
    for role in roles:
        tree = R.build(cfg, role)
        warps = role.warps or tuple(range(nwarps))
        if list(warps) != list(range(warps[0], warps[0] + len(warps))):
            raise interp.Unsupported(
                "role", role.name, f"warps {warps} are not contiguous"
            )
        for tr in _role(k, tree, role.name, warps[0], len(warps), ctx):
            name = _name(pol["family"], role.name, tr, warps)
            facts.roles.append({"name": name, "w0": tr.w0, "warps": tr.nw,
                                "body": tr.stmts})  # fmt: skip
            facts.sites += [{**s, "role": name} for s in tr.sites]
    facts.params = dict(sorted(mem.used_params.items()))
    loaded = {n["tensor"] for r in facts.roles for n in T._expr_nodes(r["body"])
              if n.get("e") == "load"}  # fmt: skip
    facts.meta = {t: pol["tensors"][t]["elem_bytes"] for t in sorted(loaded)}


def _shared_only(i) -> bool:
    """A store-like instruction of shared memory only (its opcode names a shared state
    space and no global one, e.g. st.async.shared::cluster): it writes no tensor, whether
    or not the op table has a row for it (translate.writes)."""
    parts = i.opcode.split(".")
    return (i.base in ("st", "cp", "red", "atom")
            and any(p.startswith("shared") for p in parts)
            and not any(p.startswith("global") for p in parts))  # fmt: skip


def _name(family: str, role: str, tr, warps) -> str:
    if family == "cute":
        return f"warps{tr.w0}_{tr.w0 + tr.nw - 1}"
    if (tr.w0, tr.nw) == (warps[0], len(warps)):
        return role
    return f"{role}:w{tr.w0}-{tr.w0 + tr.nw - 1}"


def _role(k, tree, name: str, w0: int, nw: int, ctx: dict) -> list:
    """The translators of one role, after splitting it into warp ranges (translate.py
    _role, without the P0 prologues)."""
    out, todo = [], [(w0, nw)]
    while todo:
        a, n = todo.pop(0)
        tr = _Nb(k, tree, a, n, name, ctx, (w0, nw))
        try:
            tr.run(tree.body)
            tr.flush()
        except interp.SplitWarps as s:
            todo = s.groups + todo
            continue
        except interp.Unsupported:
            raise
        except Exception as e:  # a bug of the core (an ExprError, ...): refused there
            tr.fail(tr.cur, f"{type(e).__name__}: {e}", "error")
        out.append(tr)
    _sign(out)
    return out


def _sign(trs: list) -> None:
    """The keys of the keyed nb_rep regions of the roles split from one role (module doc
    "Repeated regions"): where those regions' path signatures for one key differ, each
    key gets `#n`, n the rank of its signature among that key's; the end event takes its
    begin's key."""
    recs, sigs = [], {}
    for tr in trs:
        if not tr.track:
            continue
        live = {id(s) for s in _events(tr.stmts)}
        for ev, spec in tr.keyed:
            if id(ev) in live:
                sig = tr.signature(spec)
                recs.append((ev, sig))
                sigs.setdefault(ev["key"], set()).add(sig)
    rank = {k: {s: n for n, s in enumerate(sorted(v), 1)} for k, v in sigs.items()
            if len(v) > 1}  # fmt: skip
    if not rank:
        return
    for ev, sig in recs:
        if ev["key"] in rank:
            ev["key"] = f"{ev['key']}#{rank[ev['key']][sig]}"
    for tr in trs:
        _pair_keys(tr.stmts)


def _events(stmts):
    """The nb_rep begin events of a statement list, nested statements included."""
    for s in stmts:
        if s.get("ev") == "nb_rep" and s["begin"]:
            yield s
        for k in ("then", "else", "body"):
            yield from _events(s.get(k, ()))


def _pair_keys(stmts) -> None:
    """Every nb_rep end event takes the key of its begin (kir.py: they nest per list)."""
    open_ = []
    for s in stmts:
        if s.get("ev") == "nb_rep":
            if s["begin"]:
                open_.append(s)
            elif open_:
                s["key"] = open_.pop()["key"]
        for k in ("then", "else", "body"):
            _pair_keys(s.get(k, ()))
