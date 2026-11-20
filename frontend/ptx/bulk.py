# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""P1-2 rules of the PTX frontend's P1 pass ("async store sources reuse ordered",
references/specs.md "P1-2", references/kir.md change 15): bulk stores, their commit and
wait groups, and the writes of shared memory and of tensormaps that may overlap what a
bulk store reads, read off the PTX instructions.

    State()                          per kernel (tmem.P1.bulk)
    execute(st, tr, i, mn, maybe)    before the P0 rule of every instruction (tmem.P1.execute):
                                     commit/wait groups, non-tensor bulk stores, and every
                                     write of shared memory by its kind (interp.shared_write)
                                     but those P1-2 reads elsewhere (ELSEWHERE): st.shared,
                                     stmatrix, atom/red on shared memory, generic stores to
                                     it, cp.async and bulk loads, tcgen05.alloc, CLC
                                     responses, writes of another CTA's shared memory;
                                     `maybe`: under a predicate that depends on data
    store(st, tr, i, kind, src, nbytes, tmap)
                                     a bulk store (tma.py: cp{.reduce}.async.bulk{.tensor}
                                     .global.shared::cta.bulk_group)
    landing(st, tr, i, dst, nbytes)  a TMA, bulk or cp.async copy into shared memory
                                     (tma.py, mem.py): an asynchronous write
    tmap(st, tr, i, dst)             tensormap.cp_fenceproxy (tma.py): a tensormap write
    fixed(st, tr, a, nbytes, what)   shared bytes written outside the events (tensormap
                                     staging buffers, CLC responses): checked disjoint
    finish(st, translators) -> bool  after every role: the checks below, pruning; whether
                                     the kernel has bulk stores (P1-2 applies)

Events (kir.md change 15). A bulk store is a `bulk_store` of its issuing thread: `warp`,
`lane` (absent: the elected lane of a full-warp elect.sync), the shared-memory source
`sym` + `src`, `bytes` (a TMA store: the hardware box times the element bytes, the box
dense in shared memory; a bulk store: its size) and, through an in-kernel descriptor, the
tensormap it reads (`tmap`: the global buffer, Triton's scratch, and the byte offset of the
128-byte tensormap). One event per issuing thread; a store several lanes of a warp may
execute (ptxas issues it per distinct operands) is refused. cp.async.bulk.commit_group and
cp.async.bulk.wait_group{.read} N are `bulk_commit` / `bulk_wait` events of the role; the
P1-2 model applies them to every bulk store issuer of the role, so each must be executed by
every issuing thread of the role's bulk stores (checked in finish; a wait with
.sync_restrict::proxy::async, which orders only async-proxy accesses, is refused). A write
of shared memory is a `smem_write` per executing thread (`warp`, `lane` as above) of
`bytes` at `sym` + its offset: st.shared and generic stores to shared memory (vector width
times element size), atom/red on shared memory, a stmatrix (`sync`: one warp-synchronizing
instruction of the warp; each address-providing lane writes a 16-byte row), and, with
`async`, the landing of a TMA load, a bulk copy or a cp.async (its box, size or copy size;
a TMA or bulk load's landing names its load's mbarrier slot (`bar`, `idx`; `leader`: the
pair leader's; not when its load computes the slot index only after it), which lets P1-2
order it before a later store through that mbarrier's bytes; otherwise the KIR orders it
only by its issue: a store issued after it may read before it lands, which P1-2 leaves
undecided). A write of another CTA's shared memory
(st.async, red.async, a shared::cluster bulk copy or reduce through a mapa address) is such
an `async` smem_write with `cta`, the destination's rank. tensormap.cp_fenceproxy is a
`tmap_write` of every warp executing it (`.sync.aligned`).

Checks, refusals (the P1-2 pass refuses; P0 is unaffected): every executing thread of a write
must be statically known (no lane condition that depends on runtime values), and so must
the issuers of stores and groups; none of them may depend on which lane elect.sync elects
(interp._check_elected) or run under a negated elect.sync; an address that is no shared
symbol plus an offset; a write under a predicate that depends on data that may overlap a
store source; shared bytes written outside the events (mbarrier objects, tensormap staging
buffers, tcgen05.alloc's address slot, CLC responses) that may overlap a store source
(static intervals over the issuing threads); a global store or atomic into a buffer that
holds the tensormaps of bulk stores (only tensormap.cp_fenceproxy is a tensormap write); a
multicast copy.

Pruning: a `smem_write` whose bytes are statically disjoint from every store source of its
symbol is dropped (no pair can involve it); the static intervals are computed over the
issuing threads (an offset with a loop-carried part keeps its event); the loops and ifs
left empty are dropped.
"""

import weakref
from dataclasses import dataclass, field

from . import cluster, expr as E, interp

_COMMIT = "cp.async.bulk.commit_group"
_WAIT = "cp.async.bulk.wait_group"
# The kinds of shared-memory writes (interp.shared_write) P1-2 reads elsewhere: a TMA
# tensor load's landing and a tensormap staging buffer (tensormap.replace) in tma.py,
# mbarrier objects in finish; mbarrier.inval is refused in a P1 pass (sync.py)
ELSEWHERE = ("tma.load", "tensormap.replace", "mbarrier", "mbarrier.inval")


@dataclass
class _Role:
    # (warp, lane or None: the elected lane) of every bulk store issuer
    issuers: set = field(default_factory=set)
    # per commit/wait: (executing threads {(warp, lane)}, elected warps {warp}, loc)
    groups: list = field(default_factory=list)


class State:
    def __init__(self):
        self._roles: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        # shared bytes written outside the events: (symbol, (lo, hi) or None, what, loc)
        self.fixed: list = []
        # id(instruction) -> the holder of its load's mbarrier use, for its latest landing
        self.holders: dict = {}

    def role(self, tr) -> _Role:
        r = self._roles.get(tr)
        if r is None:
            r = self._roles[tr] = _Role()
        return r


def of(tr) -> State | None:
    """The P1-2 state of tr's kernel (tmem.P1.bulk), None outside a P1-2 pass."""
    return None if tr.k.p1 is None else tr.k.p1.bulk


# ------------------------------------------------------------------ dispatch


def execute(st: State, tr, i, mn: str, maybe: bool = False) -> None:
    if mn == _COMMIT or mn.startswith(_WAIT):
        if maybe:
            tr.fail(i, f"P1-2: {mn} under a predicate that depends on data")
        return _group(st, tr, i, mn)
    if mn.startswith(
        ("cp.async.bulk.global.shared", "cp.reduce.async.bulk.global.shared")
    ):
        kind = "bulk_reduce" if mn.startswith("cp.reduce") else "bulk"
        return store(st, tr, i, kind, i.operands[1], _size(tr, i, 2))
    kind, what = interp.shared_write(mn), mn.split(".")[0]
    if kind in ("st.async", "bulk.cluster"):  # into another CTA, or the issuing one
        nbytes = (
            _size(tr, i, 2) if kind == "bulk.cluster" else interp.write_bytes(i.opcode)
        )
        return _write(st, tr, i, i.operands[0], nbytes, what, async_=True)
    if kind == "bulk.load":
        return landing(st, tr, i, i.operands[0], _size(tr, i, 2))
    if kind == "cp.async":
        n = i.operands[2]
        if n[0] != "imm":
            tr.fail(i, f"{mn} with a non-constant copy size", "error")
        return landing(st, tr, i, i.operands[0], n[1])
    if kind == "clc":
        return fixed(st, tr, tr.k.mem.address(tr, i.operands[0]), 16, "a CLC response")
    if kind == "tcgen05.alloc":
        return fixed(
            st,
            tr,
            tr.k.mem.address(tr, i.operands[0]),
            4,
            "tcgen05.alloc's address slot",
        )
    if kind == "stmatrix":
        return _stmatrix(st, tr, i, mn, maybe)
    if kind in ("st.shared", "atom.shared", "generic"):
        op = i.operands[1 if mn.startswith("atom") else 0]
        if kind == "st.shared":
            what = mn
        elif kind == "generic":  # at a shared address
            p = _peel(tr.k.mem.address(tr, op))
            if p is None or p[0][2][:3] not in ("%s:", "%S:"):
                return None
            what = f"generic {what}"
        return _write(st, tr, i, op, interp.write_bytes(i.opcode), what, maybe=maybe)
    return None


def _size(tr, i, k: int) -> int:
    """The constant byte size operand k of a bulk copy (Translator.copy_size), else
    refused."""
    n = tr.copy_size(i, k)
    if n is None:
        tr.as_int(tr.operand(i, k, 32), "the bulk copy size")  # data: slice soundness
        tr.fail(i, "P1-2: a bulk copy of a size that is not a positive constant")
    return n


def _peel(a):
    from .mem import peel

    return None if isinstance(a, interp.Data) or a is None else peel(a)


def _threads(tr, what: str, operands=()):
    """(threads() indices of the executing threads, issued by the elected lane?). Which
    lane elect.sync elects is unspecified, so nothing may depend on it:
    interp._check_elected (H1) over the mask and `operands`, and no negated elect.sync
    (the other lanes)."""
    _uni, var = tr.split_mask()
    tr._check_elected(var, list(operands))
    cs = interp._conjuncts(var)
    if any(interp._bare_elect(c) and not E.is_elect(c) for c in cs):
        tr.fail(tr.cur, f"P1-2: {what} under the negation of elect.sync (its executing "
                        "lanes depend on which lane is elected)")  # fmt: skip
    act = tr.active(var)
    if act is None:
        tr.fail(
            tr.cur,
            f"P1-2: {what} under a lane condition that depends on runtime values",
        )
    return act, any(map(E.is_elect, cs))


def _shared_offset(tr, i, a, what: str):
    """(symbol, i32 offset) of a shared address (window or generic), else refused; the
    symbol of a .extern variable is its region's (interp._region)."""
    p = _peel(a)
    if p is None or p[0][2][:3] not in ("%s:", "%S:"):
        tr.fail(
            i,
            f"P1-2: {what} at an address that is no shared-memory symbol plus an offset",
        )
    return tr._region(p[0][2][3:]), E.resize(p[1], 32, False)


# ------------------------------------------------------------------ intervals


def _iv(e):
    """Signed bounds of e, or None: expr.interval, else expr's rules over the operands'
    _iv (expr._interval), plus srem by a positive constant (a ring index `i % N` of a
    loop-carried i); for pruning and the disjointness checks only (the P0 translation
    keeps expr.interval; the captures' finer interp.bounds, with masks and LLVM's
    remainders, would prune writes the P1 KIRs keep). As in expr.interval, None where
    the bounds of e or of a subexpression leave its signed width (the value may wrap)."""
    r = E.interval(e)
    if r is not None:
        return r
    if e[0] == "bin" and e[2] == "srem" and E.is_const(e[4]) and e[4][2] > 0:
        ia, k = _iv(e[3]), e[4][2] - 1
        r = (0, k) if ia is not None and ia[0] >= 0 else (-k, k)
    else:
        r = E._interval(e, _iv)
    w = E.width(e)
    if r is not None and w > 1 and not -(1 << (w - 1)) <= r[0] <= r[1] < 1 << (w - 1):
        return None
    return r


def _range(tr, a, nbytes: int):
    """[lo, hi) of the bytes the executing threads write at shared address a, each
    nbytes, or None: their footprint (interp's Translator._footprint, over _iv), None
    where it has no static bounds."""
    fp = tr._footprint(a, nbytes, _iv)
    return (fp[1], fp[2]) if fp is not None and len(fp) == 3 else None


# ------------------------------------------------------------------ events


def _splat(x: dict) -> dict:
    return {"t": "splat", "shape": [1], "w": 32, "a": x}


def _write(st: State, tr, i, op, nbytes: int, what: str, sync=False, async_=False, maybe=False,
           addr=None, cta=None, slot=None) -> None:  # fmt: skip
    """A smem_write per executing thread (module doc); `addr`: the address, else op's;
    `cta`: the rank of the CTA whose shared memory it writes (another CTA's); `slot`: a
    landing's holder of its load's mbarrier use (`landing`)."""
    a = addr if addr is not None else cluster._address(tr, op)
    if isinstance(a, cluster.Remote):
        return _remote_write(st, tr, i, a, nbytes, what)
    sym, off = _shared_offset(tr, i, a, what)
    act, elect = _threads(tr, what, [off])
    if not act:
        return
    iv = _range(tr, a, nbytes)
    lanes = not (elect or sync)
    k = 2 if lanes else 1  # the offset's operand
    ops = (
        [E.WARP]
        + ([E.LANE] if lanes else [])
        + [off]
        + ([cta] if cta is not None else [])
    )
    loc = tr.loc()

    def build(v, j):
        e = {"op": "event", "ev": "smem_write", "sym": sym, "offsets": _splat(j(v[k])),
             "mask": None, "bytes": nbytes, "warp": j(v[0]), "loc": loc,
             "_p1_2": (sym, iv, maybe, what)}  # fmt: skip
        if lanes:
            e["lane"] = j(v[1])
        if cta is not None:
            e["cta"] = j(v[k + 1])
        if sync:
            e["sync"] = True
        if async_:
            e["async"] = True
        if slot is not None and cta is None:
            e["_slot"] = slot
        return e

    tr.thread_event(ops, build, tag="p1_2")


def _remote_write(st: State, tr, i, a, nbytes: int, what: str) -> None:
    """A write of another CTA's shared memory (a Remote, cluster.py): a `smem_write` with
    `cta`, the destination's rank, and `async` (st.async and the shared::cluster copies
    complete through the destination's mbarrier); in a cluster of one CTA the rank is 0,
    the issuing CTA's own (no `cta`)."""
    p = _peel(a.addr)
    if p is None or p[0][2][:3] != "%s:":
        tr.fail(i, f"P1-2: {what} into another CTA at an address that is no shared symbol "
                   "plus an offset")  # fmt: skip
    one = tr.k.ncta <= 1
    _write(
        st,
        tr,
        i,
        None,
        nbytes,
        what,
        async_=True,
        addr=a.addr,
        cta=None if one else a.cta,
    )


def _stmatrix(st: State, tr, i, mn: str, maybe: bool) -> None:
    """stmatrix.sync.aligned.m8n8.x{1,2,4}{.trans}.shared.b16 [p], {...}: lanes 0 .. 8x-1
    each give the address of one 16-byte row (ISA stmatrix)."""
    parts = i.opcode.split(".")  # (the mnemonic drops the .x1/.x2/.x4 count)
    if "m8n8" not in parts or not any(p in ("x1", "x2", "x4") for p in parts):
        tr.fail(i, f"P1-2: {i.opcode} (only the m8n8 .x1/.x2/.x4 shapes are modeled)")
    x = next(int(p[1]) for p in parts if p in ("x1", "x2", "x4"))
    saved = tr.mask
    tr.mask = E.and1(saved, E.cmp("ult", E.LANE, E.const(8 * x, 32)))
    try:
        _write(st, tr, i, i.operands[0], 16, "stmatrix", sync=True, maybe=maybe)
    finally:
        tr.mask = saved


def landing(st: State, tr, i, dst_op, nbytes: int) -> None:
    """An asynchronous copy into the issuing CTA's shared memory (a TMA or bulk load, a
    cp.async): a smem_write with `async` per issuing thread; a load that then emits its
    tma_load (tma.py: `landed_on`) gives it that event's mbarrier slot (`finish`)."""
    st.holders[id(i)] = holder = {}
    _write(st, tr, i, dst_op, nbytes, i.opcode.split(".")[0] + " landing", async_=True,
           slot=holder)  # fmt: skip


def landed_on(st: State, i, use) -> None:
    """The tma_load the translation emitted for the copy at i, after its landing: its
    mbarrier use (interp.BarrierUse), whose slot `finish` gives the landing's events."""
    holder = st.holders.get(id(i))
    if holder is not None:
        holder["use"] = use


def _slot_of(e: dict, holder) -> None:
    """The landing `e`'s `bar`, `idx` and `leader`: those of its load's tma_load, if that is
    one event on an mbarrier of the issuing CTA or (`leader`, a cta_group::2 load) of its
    pair leader (not wrapped by a select between arrays, no `cta_mask` or `cta`)."""
    use = None if holder is None else holder.get("use")
    ev = None if use is None else use.event
    if (ev is not None and ev.get("op") == "event" and ev.get("ev") == "tma_load"
            and ev.get("bar") is not None and not {"cta", "cta_mask"} & ev.keys()):  # fmt: skip
        e["bar"], e["idx"] = ev["bar"], ev["idx"]
        if ev.get("leader"):
            e["leader"] = True


def _tensormap(tr, i, a, what: str) -> tuple:
    """(global buffer, i64 byte offset) of a tensormap's address a, Triton's scratch or a
    tensor plus an offset, else refused."""
    p = _peel(a)
    if p is None or p[0][2][:3] not in ("%x:", "%g:"):
        tr.fail(i, f"P1-2: {what} of unknown provenance")
    return p[0][2][3:], E.resize(p[1], 64)


def tmap(st: State, tr, i, dst) -> None:
    """tensormap.cp_fenceproxy [dst], ...: a tmap_write of every warp executing it."""
    base, off = _tensormap(tr, i, dst, "tensormap.cp_fenceproxy to an address")
    act, _elect = _threads(tr, "tensormap.cp_fenceproxy")
    off = tr.json(off)
    for w in sorted({tr.threads()[t][0] for t in act}):
        tr.emit({"op": "event", "ev": "tmap_write", "base": base, "addr": off,
                 "warp": interp._c32(w), "loc": tr.loc()})  # fmt: skip


def fixed(st: State, tr, a, nbytes: int, what: str) -> None:
    """Shared bytes written outside the events (module doc): their static interval."""
    sym, off = _shared_offset(tr, tr.cur, a, what)
    act, _elect = _threads(tr, what, [off])
    if act:
        st.fixed.append((sym, _range(tr, a, nbytes), what, tr.loc()))


def store(st: State, tr, i, kind: str, src_op, nbytes: int, tmap_addr=None) -> None:
    """A bulk store of `nbytes` from shared memory (module doc)."""
    a = tr.k.mem.address(tr, src_op)
    sym, off = _shared_offset(tr, i, a, "a bulk store source")
    act, elect = _threads(tr, "a bulk store", [off])
    if not act:
        return
    if len({t // 32 for t in act}) != len(act):
        tr.fail(i, "P1-2: a bulk store several lanes of a warp may execute (its issuing threads "
                   "depend on run-time operand values)")  # fmt: skip
    tm = None
    if tmap_addr is not None:
        tm = _tensormap(tr, i, tmap_addr, "a bulk store through a tensormap")
    r = st.role(tr)
    for t in act:
        w, lane = tr.threads()[t]
        r.issuers.add((w, None if elect else lane))
    iv = _range(tr, a, nbytes)
    ops = [E.WARP] + ([] if elect else [E.LANE]) + [off] + ([tm[1]] if tm else [])
    loc = tr.loc()

    def build(v, j):
        k = 1 if elect else 2
        e = {"op": "event", "ev": "bulk_store", "kind": kind, "warp": j(v[0]), "sym": sym,
             "src": j(v[k]), "bytes": nbytes, "loc": loc, "_p1_2": (sym, iv, False, "store")}  # fmt: skip
        if not elect:
            e["lane"] = j(v[1])
        if tm:
            e["tmap"] = {"base": tm[0], "addr": j(v[k + 1])}
        return e

    tr.thread_event(ops, build, tag="p1_2")


def _group(st: State, tr, i, mn: str) -> None:
    """cp.async.bulk.commit_group / wait_group{.read} N: an event of the role, with its
    executing threads recorded for finish's check."""
    if "sync_restrict" in mn:
        tr.fail(i, f"P1-2: {mn} orders only async-proxy accesses (not modeled)")
    uni, _var = tr.split_mask()
    act, elect = _threads(tr, mn)
    if not act:
        return
    threads = tr.threads()
    execs = {threads[t] for t in act}
    warps = {w for w, _l in execs} if elect else set()
    st.role(tr).groups.append(
        (frozenset() if elect else frozenset(execs), frozenset(warps), tr.loc())
    )
    if mn == _COMMIT:
        stmt = {"op": "event", "ev": "bulk_commit", "loc": tr.loc()}
    else:
        n = i.operands[0]
        if n[0] != "imm":
            tr.fail(i, f"{mn} with a non-constant group count", "error")
        stmt = {"op": "event", "ev": "bulk_wait", "n": n[1], "read": mn.endswith(".read"),
                "loc": tr.loc()}  # fmt: skip
    tr._emit_under(uni, stmt)


# ------------------------------------------------------------------ finish


def _covers(group, issuer) -> bool:
    """The commit/wait `group` is executed by the bulk store issuer (warp, lane | None)."""
    execs, elected, _loc = group
    w, lane = issuer
    if lane is None:  # the elected lane: every lane of its warp, or the elected one
        return w in elected or all((w, k) in execs for k in range(32))
    return (w, lane) in execs


def _events(stmts):
    for s in stmts:
        if s["op"] == "event":
            yield s
        for k in ("then", "else", "body"):
            yield from _events(s.get(k, []))


def _prune(stmts: list, keep) -> list:
    """stmts without the events `keep` rejects, and without the loops and ifs left empty
    (a loop left with `let`s stays: they may carry values out of it)."""
    out = []
    for s in stmts:
        if s["op"] == "event":
            if keep(s):
                out.append(s)
            continue
        if s["op"] in ("for", "while"):
            body = _prune(s["body"], keep)
            if body:
                out.append({**s, "body": body})
            continue
        if s["op"] == "if":
            t, e = _prune(s["then"], keep), _prune(s["else"], keep)
            if t or e:
                out.append({**s, "then": t, "else": e})
            continue
        out.append(s)
    return out


def _overlap(a, b) -> bool:
    return a is None or b is None or (a[0] < b[1] and b[0] < a[1])


def finish(st: State, translators: list) -> bool:
    """After every role (module doc): the issuers of each role's bulk stores execute each
    of its commit/wait groups; no global store or atomic writes a buffer holding the
    tensormaps bulk stores read; the shared bytes written outside the events and the
    writes under a data predicate are disjoint from every store source; the writes that
    cannot meet a store source are pruned. Returns whether the kernel has bulk stores."""
    stores: dict = {}  # symbol -> [interval or None]
    bases: set = set()  # the global buffers of the tensormaps bulk stores read
    for tr in translators:
        r = st.role(tr)
        for g in r.groups:
            for iss in r.issuers:
                if not _covers(g, iss):
                    who = f"lane {iss[1]}" if iss[1] is not None else "the elected lane"
                    raise interp.Unsupported(
                        "cp.async.bulk", g[2], f"P1-2: a bulk commit or wait group that the bulk "
                        f"store issuer ({who} of warp {iss[0]}) does not execute")  # fmt: skip
        for s in _events(tr.stmts):
            if s["ev"] == "bulk_store":
                stores.setdefault(s["_p1_2"][0], []).append(s["_p1_2"][1])
                if "tmap" in s:
                    bases.add(s["tmap"]["base"])
    for tr in translators:  # a tensormap write is a tmap_write (cp_fenceproxy) only
        for s in _events(tr.stmts):
            if s["ev"] in ("gstore", "gatomic") and s["tensor"] in bases:
                why = (
                    f"P1-2: a global write into {s['tensor']}, which holds the "
                    "tensormaps of bulk stores (not modeled)"
                )
                raise interp.Unsupported(s["ev"], s["loc"], why)
    objects = {(tr._region(u.sym), u.init, u.loc) for tr in translators for u in tr.barriers
               if u.init is not None}  # fmt: skip
    for sym, off, loc in sorted(objects):  # mbarrier objects (8 bytes; mem.py's arrays)
        st.fixed.append((sym, (off, off + 8), "an mbarrier object", loc))
    for sym, iv, what, loc in st.fixed:
        if any(_overlap(iv, x) for x in stores.get(sym, ())):
            raise interp.Unsupported("shared memory", loc, f"P1-2: {what} may overlap a bulk "
                                     "store's shared-memory source (not modeled)")  # fmt: skip

    def keep(s) -> bool:
        if "_slot" in s:
            _slot_of(s, s.pop("_slot"))
        info = s.pop("_p1_2", None)
        if info is None or s["ev"] != "smem_write":
            return True
        sym, iv, maybe, what = info
        hit = any(_overlap(iv, x) for x in stores.get(sym, ()))
        if hit and maybe:
            raise interp.Unsupported(what, s["loc"], "P1-2: a shared-memory write under a "
                                     "predicate that depends on data may overlap a bulk store source")  # fmt: skip
        return hit

    for tr in translators:
        tr.stmts[:] = _prune(tr.stmts, keep)
        _unbound_slots(tr.stmts, set(tr.k.mem.used_params))
        while _drop_dead_lets(tr.stmts):  # and then the loops and ifs left empty
            tr.stmts[:] = _prune(tr.stmts, lambda s: True)
    return bool(stores)


def _unbound_slots(stmts: list, bound: set) -> None:
    """Drop the mbarrier slot (`bar`, `idx`, `leader`) of a landing whose `idx` reads a
    variable not bound before it (in place; kir.py's scopes: `bound` holds the params and
    the variables of the enclosing `for`s, a `let` binds to the end of its statement list):
    its load computes the index after the landing (`_slot_of`). The landing then names no
    slot, as one whose load names none."""
    bound = set(bound)
    for s in stmts:
        if s["op"] == "let":
            bound.add(s["var"])
        elif s["op"] == "event":
            if s["ev"] == "smem_write" and "idx" in s and not _vars(s["idx"]) <= bound:
                for k in ("bar", "idx", "leader"):
                    s.pop(k, None)
        elif s["op"] == "for":
            _unbound_slots(s["body"], bound | {s["var"]})
        else:
            for k in ("then", "else", "body"):
                if k in s:
                    _unbound_slots(s[k], bound)


def _drop_dead_lets(stmts: list) -> bool:
    """Drop the `let`s of variables nothing reads, transitively (in place); whether it
    dropped one (a loop or if may be left empty: `_prune` drops it). The P1 specs
    read only events, so a dead `let` matters only where it can fail on its own: a load
    (P0-1's concern). A part of a role this pass splits off binds the per-lane values of
    lane-only loads (interp.scalarize), which data uses only, and those loads would make
    their tensors metadata, whose contents the dump records only where P0 needs them."""
    reads: dict = {}  # let variable -> the variables its expressions read
    roots: set = set()

    def walk(ss: list) -> None:
        for s in ss:
            if s["op"] == "let":
                reads.setdefault(s["var"], set()).update(_vars(s["expr"]))
                continue
            for k, v in s.items():
                if k in ("then", "else", "body"):
                    walk(v)
                else:
                    roots.update(_vars(v))

    walk(stmts)
    live, todo = set(), list(roots)
    while todo:
        v = todo.pop()
        if v not in live:
            live.add(v)
            todo.extend(reads.get(v, ()))

    def cut(ss: list) -> list:
        out = []
        for s in ss:
            if s["op"] == "let" and s["var"] not in live:
                continue
            for k in ("then", "else", "body"):
                if k in s:
                    s[k] = cut(s[k])
            out.append(s)
        return out

    stmts[:] = cut(stmts)
    return bool(reads.keys() - live)


def _vars(x) -> set:
    """The variable names a KIR JSON value reads."""
    out, stack = set(), [x]
    while stack:
        y = stack.pop()
        if isinstance(y, dict):
            if y.get("e") == "var":
                out.add(y["name"])
            stack.extend(y.values())
        elif isinstance(y, list):
            stack.extend(y)
    return out
