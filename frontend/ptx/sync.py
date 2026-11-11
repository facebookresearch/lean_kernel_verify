# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Synchronization rows of the PTX frontend besides plain mbarriers (design §5.6, §5.7):
tcgen05, cluster launch control (CLC), mbarrier invalidation and the cluster-scope
barrier forms.

    PREFIXES                          the mnemonics mem.Memory.execute routes here
    execute(tr, i, mnemonic)          one instruction of a role (interp.Translator)
    finish(translators) -> bool       after every role: the invalidated slots' check, the
                                      CLC role prologue; whether the kernel uses CLC
                                      (Facts.clc)
    use_ranges(translators)           the byte ranges of dynamic mbarrier offsets over
                                      slots (BarrierUse.rng), for the CuTe array split
    top_const(offset) -> int          the constant that picks an mbarrier offset's array

The plain mbarrier rows (init, arrive, expect_tx, the wait pseudo-instruction) and the
named barriers (bar/barrier.sync/arrive: no P0 event, as both legacy frontends drop
them; .aligned forms need a warp-uniform mask) stay in mem.py.

tcgen05 (P0): the P0 specs do not see TMEM, as neither legacy frontend's P0 pass does
(TTGIR drops ttng.tmem_*, CuTe takes the tcgen05 ops as data). So alloc, dealloc,
relinquish_alloc_permit, ld, st, wait::ld/st, cp, shift and mma emit nothing; a
.sync.aligned form needs a mask uniform in each warp (interp.Translator.aligned);
tcgen05.ld's registers are Data. tcgen05.commit[.mbarrier::arrive::one] is a
`commit_arrive` once per issuing thread (ISA: the mbarrier tracks the completion of the
executing thread's prior async tcgen05 ops; the CuTe frontend's rule; the TTGIR
frontend's one per tc_gen5_mma barrier or tc_gen5_commit is the same count where one
thread issues it). The P1 pass adds its tmem events before these rules run (tmem.py,
from interp.Translator.execute); nothing here hides what it needs. cta_group::2 needs a
cluster of CTA pairs, a multicast commit has the commit_arrive's cta_mask (cluster.py).

CLC (the TTGIR frontend's model, frontend/ttgir/to_kir.py `_clc_query`/`_with_clc`;
design §5.6; the Triton family only, CuTe CLC is M5):
  * try_cancel [resp], [mbar] -> `tma_load(__clc_response, [0], mbar)` per issuing
    thread: the 16-byte response lands through the async proxy and completes its
    transaction bytes on the barrier. .multicast::cluster::all is the same with a cluster
    of one CTA (Triton clusters are refused before, translate.py).
  * ld.shared.b128 r, [a] -> r holds a CLC response token (ClcResponse, a Data: any use
    but the queries below is refused as data). A queried token must have been read from
    the response address of a try_cancel of the kernel (any role), after a wait of the
    reading role on that try_cancel's mbarrier (earlier in its code: flow-insensitive);
    finish() refuses any other (the model hands out the schedule entry whatever the
    kernel reads, so a stale or wrong buffer would pass).
  * query_cancel.is_canceled p, r: one query by the whole role (the mask must be true):
    `let clc<k> = load(__clc_schedule, [pid.x, __clc_k])`, then `__clc_k += 1`;
    p = clc<k> != -1, and r's token remembers clc<k> and p.
  * query_cancel.get_first_ctaid.v4 {x, y, z, _}, r of a queried token, predicated by
    its p (ISA: undefined on a response that is not canceled): x = clc<k>, y, z =
    (clc<k> == -1 ? -1 : 0) (the mov defaults before it stay where p is false); the
    per-axis forms get_first_ctaid::x/::y/::z d, r (CuTe DSL 4.6.1) give that one.
  Any other use of a response is refused. finish() gives every role the prologue
  `let __clc_k = 0; if __clc_launched[pid.x] != 0 { body }` (a CTA the schedule never
  launches runs nothing) when some role uses CLC. The family assembly declares
  __clc_response, __clc_schedule and __clc_launched and makes the per-schedule inputs
  (Facts.clc, frontend/ttgir/to_kir.py).

Pipeline stages (CuTe policy): a run of barriers that one guard block and one .loc
initialize (DSL 4.4.2 gives every pipeline's inits the same .loc, and LLVM merges guard
blocks) is split by the slots each dynamic use may take (mem._split, the CuTe frontend's
_split_arrays). A stage index is a loop-carried or merged slot, `i = i + 1 == S ? 0 : i +
1`, whose expression alone has no interval: use_ranges gives every KIR variable of a role
the hull of the values its lets assign anywhere (flow-insensitive, so it holds wherever
the variable is read), to a fixed point, with `select(x == k, a, x)` excluding k from x
in the else arm; a range that keeps growing is unknown (the run stays whole). Only an
index that varies is narrowed: an offset over slots proved invariant (LLVM unrolled the
stages into copies) keeps the unknown interval of its expression, as without the
analysis, so its run stays whole as the CuTe frontend's dynamic index over the stages
keeps it.

mbarrier.inval [a] (KIR has no re-initialization: a slot's phases only count up, so a
slot initialized again after an inval would look like one barrier whose later waits
repeat a parity). Accepted only where it ends a private synchronous cycle, which is
elided: TLX's tlx.async_dot without mBarriers waits for its MMAs on a compiler barrier,
`init(n); tcgen05.commit; wait(parity 0); inval` in each iteration. The cycle is the
role's events on the slot (a constant offset) since its last bar_init: that init of a
constant count n, arrivals (arrive, commit_arrive on the issuing CTA's slot, each of
statically known issuing threads) adding up to exactly n, then one wait of parity 0;
each of them, and the inval (statically known issuing threads, no runtime condition), in
one KIR statement list (no uniform branch or loop between them). So the wait returns
once the arrivals land (a commit's when its MMAs complete), the phase is complete and
nothing is pending when the slot is invalidated: the cycle only orders the role after
its own arrivals, which no P0 spec sees (P0 has no TMEM), and its events are removed.
finish() refuses a kernel in which any other event names that slot (another role's, or
this role's outside such a cycle: an inval does not make a later use valid). The P1
pass refuses mbarrier.inval (its cycles order the tcgen05 ops P1 checks). Anything else
is refused, e.g. a TMA load's cycle (Triton's synchronous descriptor load), whose
transaction bytes P0-4 checks.

Cluster scope: barrier.cluster.*, mbarrier.arrive(.expect_tx).shared::cluster (a remote
arrive) and mbarrier.complete_tx go to cluster.py.
"""

import collections
from dataclasses import dataclass

from . import cluster, expr as E, regions as R
from .interp import Data, Translator, Unsupported

PREFIXES = (
    "tcgen05.", "clusterlaunchcontrol.", "ld.shared", "barrier.cluster",
    "mbarrier.arrive.shared::cluster", "mbarrier.arrive.expect_tx.shared::cluster",
    "mbarrier.complete_tx", "mbarrier.inval",
)  # fmt: skip

CLC_K = "__clc_k"  # the role's query counter (frontend/ttgir/to_kir.py)


@dataclass(frozen=True)
class ClcResponse(Data):
    """The 16-byte CLC response in a register; `query` is the schedule entry an
    is_canceled read for it (a KIR variable) and `pred` the predicate it set, None
    before; `read` the ld.shared it came from: (response address, the mbarriers its
    role waited on before, instruction)."""

    query: object = None
    pred: object = None
    read: tuple = ()


def execute(tr: Translator, i, mn: str) -> None:
    if mn.startswith("tcgen05."):
        _tcgen05(tr, i, mn)
    elif mn.startswith("clusterlaunchcontrol."):
        _clc(tr, i, mn)
    elif mn == "ld.shared":
        _response(tr, i)
    elif mn == "mbarrier.inval.shared":
        _inval(tr, i)
    else:
        cluster.execute(tr, i, mn)


# ------------------------------------------------------------------- tcgen05


def _tcgen05(tr: Translator, i, mn: str) -> None:
    if "cta_group::2" in mn:
        cluster.pair(tr, i, mn)
    if mn.startswith("tcgen05.commit"):
        sym, off = tr.k.mem._smem(tr, i, i.operands[0])
        fields = cluster.commit_fields(tr, i, mn)  # a multicast commit's cta_mask
        tr.barrier_event("commit_arrive", sym, off, "thread", **fields)
        return
    if ".sync.aligned" in mn:
        tr.aligned(mn)
    for r in R.dests(i):  # tcgen05.ld: TMEM contents
        tr.env[r] = Data(f"{i.opcode} at {tr.loc(i)}")


# ----------------------------------------------------------------------- CLC


def _clc(tr: Translator, i, mn: str) -> None:
    if tr.k.mem.policy.get("family") != "ttgir":
        tr.fail(i, f"{mn} (gap: CLC outside the Triton family is M5)")
    if mn.startswith("clusterlaunchcontrol.try_cancel"):
        sym, off = tr.k.mem._smem(tr, i, i.operands[1])
        resp = _address(tr, i)
        zero = E.const(0, 32)
        issued = len(tr.barriers)
        tr.barrier_event(
            "tma_load", sym, off, "thread", desc="__clc_response", coords=[zero]
        )
        if len(tr.barriers) > issued:  # some thread issues it: it writes `resp`
            tr.clc_cancels.append((resp, (sym, tr.uniform(off))))
        tr.uses_clc = True
        return
    reg = i.operands[-1]
    tok = tr.env.get(reg[1]) if reg[0] == "reg" else None
    if not isinstance(tok, ClcResponse):
        tr.fail(i, f"{mn} of a value that is not a CLC response read (ld.shared.b128)")
    if mn == "clusterlaunchcontrol.query_cancel.is_canceled":
        _query(tr, i, reg[1], tok)
        return
    if tok.query is None:
        tr.fail(i, f"{mn} of a CLC response no is_canceled queried")
    g = i.guard
    if g is None or g[1] or tok.pred is None or tr.env.get(g[0]) is not tok.pred:
        tr.fail(i, f"{mn} not predicated by the is_canceled result of its response (ISA: "
                   "undefined for a response that is not canceled)")  # fmt: skip
    x = tok.query
    y = E.select(E.cmp("eq", x, E.const(-1, 32)), E.const(-1, 32), E.const(0, 32))
    dst = i.operands[0]
    regs = dst[1] if dst[0] == "vec" else [dst]
    axis = "xyz".index(mn[-1]) if "::" in mn else 0  # get_first_ctaid::x/::y/::z
    for k, r in enumerate(regs, axis):
        if r[0] == "reg":
            w = Data(f"the 4th component of {i.opcode} at {tr.loc(i)}")
            tr.write(r[1], x if k == 0 else y if k < 3 else w)


def _query(tr: Translator, i, reg: str, tok: ClcResponse) -> None:
    """is_canceled: the role's next entry of its CTA's schedule, read into a variable of
    its own before the counter advances (a later use must not see the new count)."""
    if not E.is_const(tr.mask, 1):
        tr.fail(i, "a CLC query under a condition (the role queries as a whole)")
    name = tr.fresh("clc")
    k = _j("var", name=CLC_K)
    load = _j("load", tensor="__clc_schedule", index=[_j("pid", axis=0), k])
    tr.emit({"op": "let", "var": name, "expr": load})
    step = _j("bin", o="add", a=k, b=_j("const", v=1))
    tr.emit({"op": "let", "var": CLC_K, "expr": step})
    x = E.var(name, 32)
    dst = i.operands[0]
    pred = None
    if dst[0] == "reg":
        tr.write(dst[1], E.cmp("ne", x, E.const(-1, 32)))
        pred = tr.env[dst[1]]
    tr.env[reg] = ClcResponse(tok.origin, x, pred, tok.read)
    tr.clc_reads.append(tok.read)
    tr.uses_clc = True


def _response(tr: Translator, i) -> None:
    """ld.shared: a b128 load is a CLC response token (where it was read from, after which
    waits: finish checks it if queried), any other is data."""
    dst = i.operands[0]
    if not i.opcode.endswith(".b128") or dst[0] != "reg":
        for r in R.dests(i):
            tr.env[r] = Data(f"{i.opcode} at {tr.loc(i)}")
        return
    read = (_address(tr, i), frozenset(tr.waited), i)
    tr.env[dst[1]] = ClcResponse(f"CLC response {i.opcode} at {tr.loc(i)}", read=read)


def _address(tr: Translator, i):
    """(shared symbol, offset) of i's address operand [a] (try_cancel's response, an
    ld.shared's source) if the offset is the same in every thread of the role, else
    None."""
    op = next(x for x in i.operands if x[0] == "mem")
    try:
        sym, off = tr.k.mem._smem(tr, i, op)
    except Unsupported:
        return None
    u = tr.uniform(off)
    return None if u is None else (sym, u)


# -------------------------------------------------------------- invalidation

_ARRIVALS = ("arrive", "commit_arrive")


def _inval(tr: Translator, i) -> None:
    """mbarrier.inval [a]: the end of a private synchronous cycle on a's slot, whose
    events are removed (module doc); else refused."""
    if tr.k.p1 is not None:
        tr.fail(i, "mbarrier.inval in the P1 pass (gap: its cycle orders tcgen05 ops)")
    sym, off = tr.k.mem._smem(tr, i, i.operands[0])
    c = tr.uniform(off)
    uni, var = tr.split_mask()
    if c is None or not E.is_const(c) or not E.is_const(uni, 1) or not tr.active(var):
        tr.fail(i, "mbarrier.inval of a slot that is not a constant offset, or under a "
                   "condition that depends on runtime values (or that no thread meets)")  # fmt: skip
    mine = [u for u in tr.barriers if u.sym == sym and u.const == c[2]]
    start = max(
        (k for k, u in enumerate(mine) if u.event["ev"] == "bar_init"), default=None
    )
    cycle = mine[start:] if start is not None else []
    tops = [_top(tr.stmts, u.event) for u in cycle]
    why = _cycle_fault(cycle, tops)
    if why:
        tr.fail(i, f"mbarrier.inval of {sym}+{c[2]} ends no private synchronous cycle "
                   f"of the role (init, arrivals, one wait of parity 0): {why}")  # fmt: skip
    gone = {id(u) for u in cycle}
    tr.barriers[:] = [u for u in tr.barriers if id(u) not in gone]
    drop = {id(t[0]) for t in tops if t is not None}
    tr.stmts[:] = [s for s in tr.stmts if id(s) not in drop]
    tr.invals.append((sym, c[2], i))


def _top(stmts: list, ev: dict):
    """(the statement of `stmts` holding event ev, its issuing threads): ev itself (1),
    or a `for` over the issuing threads whose body is ev (thread mode, tr._issue: its
    variable `issuer<n>`, which no other loop is named); None if ev is not at the top of
    the list (inside an `if`, a program loop, a thread loop, another list)."""
    for s in stmts:
        if s is ev:
            return s, 1
        if (s["op"] == "for" and s["var"].startswith("issuer") and s["body"] == [ev]
                and s["body"][0] is ev and s["lb"] == {"e": "const", "w": 32, "v": 0}
                and s["step"] == {"e": "const", "w": 32, "v": 1}
                and s["ub"]["e"] == "const"):  # fmt: skip
            return s, s["ub"]["v"]
    return None


def _cycle_fault(cycle: list, tops: list) -> str | None:
    """Why the role's uses `cycle` of a slot (from its last bar_init on) with their
    statements `tops` are no private synchronous cycle, or None."""
    if not cycle:
        return "no bar_init of the slot before it"
    if None in tops:
        return "an event of the cycle in another branch, loop or thread loop"
    if any(u.rest is not None for u in cycle):
        return "a dynamic offset names the slot"
    init, wait, arrivals = cycle[0].event, cycle[-1].event, cycle[1:-1]
    count = init["count"]
    if wait["ev"] != "wait" or wait.get("parity") != {"e": "const", "w": 32, "v": 0}:
        return "it does not end in one wait of parity 0"
    if count["e"] != "const":
        return "the init count is not a constant"
    n = 0
    for u, (_s, k) in zip(arrivals, tops[1:-1]):
        ev = u.event
        if ev["ev"] not in _ARRIVALS or {"cta", "cta_mask", "leader"} & set(ev):
            return f"a {ev['ev']} event on the slot (only arrivals of the issuing CTA)"
        m = ev.get("n", {"e": "const", "v": 1})
        if m["e"] != "const":
            return "an arrive of a count that is not a constant"
        n += (u.issuers if ev["ev"] == "arrive" else k) * m["v"]
    if n != count["v"]:
        return f"{n} arrivals before the wait for a count of {count['v']}"
    return None


def _check_invals(translators: list) -> None:
    """Refuse a kernel in which an event outside an elided cycle names an invalidated
    slot (module doc)."""
    for tr in translators:
        for sym, c, i in tr.invals:
            if any(
                u.sym == sym and u.const == c for t in translators for u in t.barriers
            ):
                tr.fail(i, f"mbarrier.inval of {sym}+{c}, a slot that events outside a "
                           "private synchronous cycle use")  # fmt: skip


def finish(translators: list) -> bool:
    """The CLC prologue of every role if some role uses CLC; whether one does. Refused:
    a queried response read from an address no try_cancel writes, or before its role
    waits on one of that address's try_cancel mbarriers; an invalidated slot used outside
    its elided cycles."""
    _check_invals(translators)
    if not any(tr.uses_clc for tr in translators):
        return False
    bars: dict = {}
    for tr in translators:
        for resp, bar in tr.clc_cancels:
            bars.setdefault(resp, set()).add(bar)
    for tr in translators:
        for resp, waited, i in tr.clc_reads:
            if resp is None or resp not in bars:
                tr.fail(i, "CLC response read from an address no try_cancel writes")
            if not bars[resp] & waited:
                tr.fail(
                    i, "CLC response read before a wait on its try_cancel's mbarrier"
                )
    load = _j("load", tensor="__clc_launched", index=[_j("pid", axis=0)])
    launched = {"e": "cmp", "w": 1, "o": "ne", "a": load, "b": _j("const", v=0)}
    for tr in translators:
        init = {"op": "let", "var": CLC_K, "expr": _j("const", v=0)}
        tr.stmts = [init, {"op": "if", "cond": launched, "then": tr.stmts, "else": []}]
    return True


def _j(e: str, **fields) -> dict:
    """A 32-bit KIR Expr node (the CLC tables are i32, frontend/ttgir/to_kir.py)."""
    return {"e": e, "w": 32, **fields}


# ------------------------------------------------------------ barrier arrays


def top_const(e) -> int:
    """The constant of e's top-level sum (adds and subtracts, not through a scale)."""
    if E.is_const(e):
        return e[2]
    if e[0] == "bin" and e[2] in ("add", "sub"):
        a, b = top_const(e[3]), top_const(e[4])
        return a + b if e[2] == "add" else a - b
    return 0


_BOTTOM = "bottom"  # a variable no let has given a range yet
_ROUNDS = 16  # fixed-point rounds before a growing range becomes unknown


def use_ranges(translators: list) -> None:
    """Set the byte range of each dynamic mbarrier offset the interval of its expression
    alone left unknown, from the ranges of the role's variables, where it spans several
    slots (module docstring)."""
    for tr in translators:
        todo = [u for u in tr.barriers if u.rest is not None and u.rng is None]
        if not todo:
            continue
        env, defs = _var_ranges(tr.stmts)
        for u in todo:
            r = _range(u.rest, env, defs)
            if r not in (None, _BOTTOM) and r[0] < r[1]:
                u.rng = (u.const + 8 * r[0], u.const + 8 * r[1])


def _var_ranges(stmts: list) -> tuple:
    """({variable: (lo, hi) or None (unknown)} over every let of the statements and the
    constant-bound `for` variables, {variable: expr} of those assigned once)."""
    lets, env = [], {}

    def walk(ss):
        for s in ss:
            if s["op"] == "let":
                lets.append((s["var"], s["expr"]))
            if s["op"] == "for":
                lb, ub = s["lb"], s["ub"]
                const = lb["e"] == ub["e"] == "const" and lb["v"] < ub["v"]
                env[s["var"]] = (lb["v"], ub["v"] - 1) if const else None
            for k in ("then", "else", "body"):
                walk(s.get(k, []))

    walk(stmts)
    env.update((v, _BOTTOM) for v, _e in lets)
    once = collections.Counter(v for v, _e in lets)
    defs = {v: e for v, e in lets if once[v] == 1}
    for k in range(_ROUNDS + len(lets)):
        changed = False
        for v, e in lets:
            if env[v] is None:
                continue
            r = _hull(env[v], _range(e, env, defs))
            if k >= _ROUNDS and r != env[v]:
                r = None  # still growing: unknown (only None spreads from here on)
            if r != env[v]:
                env[v], changed = r, True
        if not changed:
            break
    return env, defs


def _hull(a, b):
    if a == _BOTTOM:
        return b
    if b == _BOTTOM:
        return a
    if a is None or b is None:
        return None
    return min(a[0], b[0]), max(a[1], b[1])


def _range(e: dict, env: dict, defs: dict):  # noqa: C901 (one case per KIR node)
    """Signed bounds of a KIR Expr given the variables' ranges, None if unknown, _BOTTOM
    if it reads a variable without one yet; `defs` holds the expressions of variables
    assigned once (a select's condition is often one)."""
    k, w = e["e"], e["w"]
    if k == "const":
        return e["v"], e["v"]
    if k == "var":
        return env.get(e["name"])
    if k == "cmp":
        return 0, 1
    if k == "select":
        return _hull(*_arms(e, env, defs))
    if k in ("load", "pid", "nprog"):
        return None
    a = _range(e["a"], env, defs)
    if a in (None, _BOTTOM):
        return a
    if k == "ext" and e["a"]["w"] == 1:
        return (-1, 0) if e["signed"] else (0, 1)
    if k == "ext" and not e["signed"] and a[0] < 0:
        return None
    if k in ("ext", "trunc"):
        return _fit(a, w)
    if k != "bin":
        return None
    b = _range(e["b"], env, defs)
    if b in (None, _BOTTOM):
        return b
    o, c = e["o"], (b[0] if b[0] == b[1] else None)
    if o == "shl" and c is not None and 0 <= c < w:
        o, b = "mul", (1 << c, 1 << c)
    if o == "add":
        r = a[0] + b[0], a[1] + b[1]
    elif o == "sub":
        r = a[0] - b[1], a[1] - b[0]
    elif o == "mul":
        ps = [x * y for x in a for y in b]
        r = min(ps), max(ps)
    elif a[0] < 0 or b[0] < 0:
        return None
    elif o == "and":
        r = 0, min(a[1], b[1])
    elif o in ("or", "xor"):
        r = 0, (1 << max(a[1], b[1]).bit_length()) - 1
    elif o in ("urem", "srem") and b[0] > 0:
        r = 0, min(a[1], b[1] - 1)
    elif o in ("udiv", "sdiv") and b[0] > 0:
        r = a[0] // b[1], a[1] // b[0]
    elif o in ("lshr", "ashr") and c is not None:
        r = a[0] >> c, a[1] >> c
    elif o in ("smin", "umin", "smax", "umax"):
        f = min if o.endswith("min") else max
        r = f(a[0], b[0]), f(a[1], b[1])
    else:
        return None
    return _fit(r, w)


def _arms(e: dict, env: dict, defs: dict) -> tuple:
    """The ranges of a select's arms, each where the condition chooses it: under
    `x == k` (x != k) the other arm sees x without k, as the stage wraps
    `i == S - 1 ? 0 : i + 1` and `i == S - 2 ? 0 : i + 2` need. Each arm is evaluated
    once (a lane table is a deep select chain)."""
    c = defs.get(e["c"].get("name"), e["c"])
    arms = [e["a"], e["b"]]
    if c["e"] != "cmp" or c["o"] not in ("eq", "ne") or c["b"]["e"] != "const":
        return _range(arms[0], env, defs), _range(arms[1], env, defs)
    x, kv = c["a"], c["b"]["v"]
    other = 1 if c["o"] == "eq" else 0  # the arm taken when x != k
    out = [None, None]
    out[1 - other] = _range(arms[1 - other], env, defs)
    old = env.get(x.get("name")) if x["e"] == "var" else None
    if arms[other] == x and x["e"] != "var":  # x == k ? a : x, x an expression
        r = _range(x, env, defs)
        out[other] = _without(r, kv) if isinstance(r, tuple) else r
    elif isinstance(old, tuple):  # evaluate the arm with x != k
        env[x["name"]] = _without(old, kv)
        try:
            out[other] = _range(arms[other], env, defs)
        finally:
            env[x["name"]] = old
    else:
        out[other] = _range(arms[other], env, defs)
    return out[0], out[1]


def _fit(r, w: int):
    """r if no value in it wraps at width w, else None."""
    if w == 1:
        return r if 0 <= r[0] <= r[1] <= 1 else None
    return r if -(1 << (w - 1)) <= r[0] <= r[1] < (1 << (w - 1)) else None


def _without(r, k: int):
    """The range r without the value k where k is an end of it (_BOTTOM if r is k)."""
    if r == (k, k):
        return _BOTTOM
    if r[0] == k:
        return k + 1, r[1]
    return (r[0], k - 1) if r[1] == k else r
