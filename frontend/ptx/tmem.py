# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""P1 pass of the PTX frontend (design §5.10; CuTe family only, decision D3): the P1-1
"TMEM reuse ordered" events of frontend/cute/ops_tmem.py read off the PTX instructions,
so that the P1 KIR is the P0 KIR plus `tmem`, `tmem_wait` and `named_bar` events and the
top-level `tmem` (references/kir.md change 10).

    P1()                            per-kernel state; translate.py hands it to
                                    interp.Kernel.p1 when the policy asks for P1 ("p1")
      .execute(tr, i, mn)           before the P0 rule of an instruction (interp.py): adds
                                    its P1 event, if any; the P0 rule still runs
      .issued(tr, ev, var, act)     an mbarrier event about to be issued (interp._issue)
      .top(translators) -> dict | None
                                    the KIR `tmem` after every role; None: the kernel
                                    allocates and accesses no TMEM (nothing to check)

Events (the CuTe P1 pass's rules, ops_tmem.py, on PTX):

* tcgen05.ld / tcgen05.st .32x32b.xN: one `tmem` ld/st per issuing warp, every lane of
  the warp active, the address the same in all its lanes (ISA: .sync.aligned, one taddr):
  lanes [L, L+32) x columns [C, C+N). One event per warp, or a `for` over the warps when
  every warp of the role issues it.
* tcgen05.cp (128x256b: 8 columns; 128x128b, 64x128b.warpx2, 32x128b.warpx4: 4; all 128
  lanes, the multicast shapes replicating into every lane quarter) writes; tcgen05.shift
  (never seen in a kernel so far) is a cp writing lanes [0, 128) x [C, C+8), an
  over-approximation of the rows it moves (KIR has no shift; a write conflicts wherever
  a read-write does).
* tcgen05.mma: D (written) 128 lanes x N columns, N / 2 for M = 128 with cta_group::2
  (CuTe's 2x2 data path layout, ops_tmem._d_cols); A from TMEM (read) 8 columns (every
  kind's K per instruction is 256 bits of a row); block-scaled kinds read SFA 4 columns and
  SFB 4 per 128 rows of N, the CuTe DSL's TMEM layouts (wider than the ISA's, design §5.10:
  max of the two). M and N come from the instruction descriptor's known bits (expr._bit:
  N >> 3 at bits 17-22; M >> 4 at bits 24-28, or M >> 7 at bits 27-28 for the block-scaled
  kinds; ISA instruction descriptor). The disable-output-lane mask is ignored (D is taken
  as all 128 lanes, a superset). An SF address keeps the byte the DSL selects in bits
  30-31 (the descriptor's SF id picks it): those bits are dropped, as the CuTe pass floors
  the SF pointer to its column.
* cp, shift and mma are issued by one thread of a one-warp role, the elected lane of an
  elect.sync or one fixed thread (ISA: single-thread issue); cta_group::2 accesses the
  same rectangles in both CTAs of the pair, `cta_mask` 3 << (rank & ~1).
* tcgen05.wait::ld / ::st: a `tmem_wait` of the whole role.
* bar.sync / bar.arrive / barrier.sync / barrier.arrive id[, count]: a `named_bar` of the
  whole role, id and count constants (count absent: the block's threads).
* tcgen05.alloc: the allocation (one per issuing warp); dealloc, relinquish_alloc_permit
  and tcgen05.fence::* are not modeled (P1-1 checks neither lifetime nor fences).
* An mbarrier wait, arrive or commit that only some warps of a role issue splits the role
  into runs of issuing and non-issuing warps (SplitWarps; interp.Translator.
  _p1_split_by_issuers of the CuTe frontend), as P1-1 takes a role's event as every
  warp's.

TMEM addresses are absolute (the CuTe DSL's fake-zero-base idiom: constants and the
integer arithmetic on them), KIR `tmem.base` "zero" with a 512-column allocation; an
address from the allocation's result (relative, TLX) is M5 (design §5.10). An address
built after a loop from a register the loop rematerializes (`mov r, %tid.x` in the body,
r read after the loop only) is warp-uniform only if r keeps that value there: in the P1
pass interp.Translator.run_loop gives such a register the one loop-invariant value every
loop edge leaves it at, instead of its slot (whose placeholder entry value makes it a
per-thread family). The P0 pass keeps its slots, so the P0 KIR does not change.

Refusals (Unsupported, the P1 KIR is not checked; P0 is unaffected), tagged as the CuTe
pass's (references/frontend_cute.md R1-R9; R4 and R8 are gone with the MLIR):
R1 a TMEM address that is data (e.g. read back from shared memory) or a memory address;
R2 not exactly one allocation, one in a loop or under a runtime lane condition, or of a
non-constant or unsupported column count; R3 absolute addresses with an allocation under
512 columns; R5 shapes and kinds outside the above, an unknown or unsupported M/N, a
block-scaled N over 256, cta_group::2 without a cluster; R6 an ld/st with part of a warp
active, a lane-varying address or under a runtime lane condition, a cp/mma in a role of
several warps, issued by several threads or with a lane-varying operand; R7 a named
barrier or tcgen05.wait reached by part of the role, a named barrier id or count that is
not a constant (one uniform per warp splits the role) or a count that is not a positive
multiple of 32, an mbarrier wait, arrive or commit under a runtime lane condition in a
role of several warps; R10 (new, design §5.4.1) a role whose cp/mma and commits are not
all issued alike (elected lane, or one fixed thread): P1-1 takes a commit as covering the
same thread's earlier cp/mma.
"""

import math
import re
import weakref
from dataclasses import dataclass, field

from . import cluster, expr as E, interp

_SYNC_EVENTS = ("arrive", "wait", "commit_arrive")  # order TMEM accesses in P1-1
_NAMED_BARS = ("bar.sync", "bar.arrive", "barrier.sync", "barrier.arrive",
               "barrier.sync.aligned", "barrier.arrive.aligned")  # fmt: skip
_CP_COLS = {"128x256b": 8, "128x128b": 4, "64x128b": 4, "32x128b": 4}
_BLOCK_SCALED = ("mxf8f6f4", "mxf4", "mxf4nvf4")
_ALLOC_COLS = (32, 64, 128, 256, 512)


@dataclass
class _Role:
    """What a role's translator recorded (top() reads the final roles' only: a role that
    split, or a loop translated again, leaves no trace)."""

    # id(tcgen05.alloc) -> (issuing warps, columns, loc): by instruction, so that an arm
    # translated twice (interp._inert's trial, a loop widened) counts once
    allocs: dict = field(default_factory=dict)
    accessed: bool = False  # a tcgen05 ld/st/cp/shift/mma
    issuers: dict = field(default_factory=dict)  # cp/mma/commit issuer -> loc
    async_ops: bool = False  # a cp, shift or mma


class P1:
    def __init__(self):
        self._roles: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._index: dict = {}  # id(instr) -> its index in fn.instructions() (site)

    def _role(self, tr: interp.Translator) -> _Role:
        r = self._roles.get(tr)
        if r is None:
            r = self._roles[tr] = _Role()
        return r

    # ------------------------------------------------------------- dispatch
    def execute(self, tr: interp.Translator, i, mn: str) -> None:
        if mn.startswith(("tcgen05.ld.", "tcgen05.st.")):
            self._ld_st(tr, i, mn)
        elif mn.startswith("tcgen05.wait::"):
            what = "ld" if mn.startswith("tcgen05.wait::ld") else "st"
            _role_event(tr, {"op": "event", "ev": "tmem_wait", "ld": what == "ld",
                             "loc": tr.loc()}, f"tcgen05.wait::{what}")  # fmt: skip
        elif mn.startswith(("tcgen05.cp.", "tcgen05.shift.")):
            self._cp(tr, i, mn)
        elif mn.startswith("tcgen05.mma."):
            self._mma(tr, i, mn)
        elif mn.startswith("tcgen05.alloc."):
            self._alloc(tr, i)
        elif mn in _NAMED_BARS:
            _named_bar(tr, i, mn)

    def issued(self, tr: interp.Translator, ev: str, var, act) -> None:
        """An mbarrier event under the thread-varying mask `var`, issued by the threads
        `act` (threads() indices, None: it depends on runtime values): a commit's issuer
        is recorded (R10); a role of several warps of which only some issue a wait,
        arrive or commit is split (SplitWarps), refused under a runtime condition (R7)."""
        if ev == "commit_arrive":
            self._role(tr).issuers.setdefault(_issuer(var, act), tr.loc())
        if ev not in _SYNC_EVENTS or tr.nw == 1:
            return
        if act is None:
            tr.fail(tr.cur, f"R7: mbarrier {ev} under a lane condition that depends on runtime "
                            f"values in a role of {tr.nw} warps (the warps it orders are unknown)")  # fmt: skip
        issuing = {k // 32 for k in act}
        if len(issuing) == tr.nw:
            return
        groups, start = [], 0
        for w in range(1, tr.nw + 1):
            if w == tr.nw or (w in issuing) != (w - 1 in issuing):
                groups.append((tr.w0 + start, w - start))
                start = w
        raise interp.SplitWarps(groups)

    def top(self, translators: list):
        """The P1 KIR's top-level `tmem` from the final roles' allocations and accesses;
        None if there are neither (ops_tmem.tmem_top)."""
        roles = [self._role(tr) for tr in translators]
        allocs = [
            (c, loc) for r in roles for n, c, loc in r.allocs.values() for _ in range(n)
        ]
        if not allocs and not any(r.accessed for r in roles):
            return None
        if len(allocs) != 1:
            loc = allocs[1][1] if len(allocs) > 1 else "kernel"
            raise interp.Unsupported("tcgen05.alloc", loc,
                                     f"R2: {len(allocs)} TMEM allocations (exactly one is supported)")  # fmt: skip
        cols, loc = allocs[0]
        if cols not in _ALLOC_COLS:
            raise interp.Unsupported("tcgen05.alloc", loc, f"R2: {cols} columns")
        if any(r.accessed for r in roles) and cols != 512:
            raise interp.Unsupported("tcgen05.alloc", loc, f"R3: absolute TMEM addresses with a {cols}-column "
                                     "allocation (the allocation's address is not known)")  # fmt: skip
        for tr, r in zip(translators, roles):
            if r.async_ops and len(r.issuers) > 1:
                raise interp.Unsupported("tcgen05.commit", ", ".join(sorted(set(r.issuers.values()))),
                                         f"R10: role warps{tr.w0}_{tr.w0 + tr.nw - 1} issues its cp/mma and "
                                         "commits by different threads (elected lane, fixed lane)")  # fmt: skip
        return {"cols": cols, "base": "zero"}

    # ------------------------------------------------------------- accesses
    def _ld_st(self, tr: interp.Translator, i, mn: str) -> None:
        inst = mn.split(".")[1]
        m = re.search(r"\.32x32b\.x(\d+)\b", i.opcode)
        if m is None or not 1 <= int(m.group(1)) <= 128:
            tr.fail(i, f"R5: {i.opcode} (only the .32x32b.xN shapes are in the table)")
        n = int(m.group(1))
        lane, col = self._addr(tr, i, _mem(i), f"tcgen05.{inst}")
        loc = tr.loc()
        _per_warp(tr, [lane, col], lambda v, w, j: {
            "op": "event", "ev": "tmem", "inst": inst, "warp": j(w),
            "rects": [_rect("data", inst == "st", v[0], 32, v[1], n, j)], "site": "", "loc": loc,
        }, f"tcgen05.{inst}")  # fmt: skip

    def _cp(self, tr: interp.Translator, i, mn: str) -> None:
        if mn.startswith("tcgen05.shift."):
            shape, ncols, what = None, 8, "tcgen05.shift"
        else:
            shape = mn.split(".")[3]
            ncols, what = _CP_COLS.get(shape), "tcgen05.cp"
            if ncols is None:
                tr.fail(
                    i, f"R5: {i.opcode} (cp shapes in the table: {sorted(_CP_COLS)})"
                )
        lane, col = self._addr(tr, i, _mem(i), what)
        if shape is None:  # shift: every lane, whatever the address's lane
            lane = E.const(0, 32)
        mask = _pair_mask(tr, i, mn)
        loc = tr.loc()
        self._one_thread(tr, i, [lane, col] + ([mask] if mask is not None else []), lambda v, j: {
            "op": "event", "ev": "tmem", "inst": "cp", "warp": j(E.const(tr.w0, 32)),
            "rects": [_rect("data", True, v[0], 128, v[1], ncols, j)],
            **({"cta_mask": j(v[2])} if mask is not None else {}), "site": "", "loc": loc,
        }, what)  # fmt: skip

    def _mma(self, tr: interp.Translator, i, mn: str) -> None:
        kind = re.search(r"kind::(\w+)", mn)
        kind = kind.group(1) if kind else "?"
        bs = kind in _BLOCK_SCALED
        ncta = 2 if "cta_group::2" in mn else 1
        idesc = tr.operand(i, 3, 32)
        n_f = None if isinstance(idesc, interp.Data) else _field(idesc, 17, 6)
        m_f = None if isinstance(idesc, interp.Data) else (
            _field(idesc, 27, 2) if bs else _field(idesc, 24, 5))  # fmt: skip
        if n_f is None or m_f is None:
            tr.fail(
                i,
                "R5: tcgen05.mma instruction descriptor whose shape bits are not known",
            )
        m, n = m_f << (7 if bs else 4), n_f << 3
        if (m, ncta) not in ((128, 1), (128, 2), (256, 2)) or not 8 <= n <= 256:
            tr.fail(i, f"R5: tcgen05.mma.cta_group::{ncta}.kind::{kind} {m}x{n} (only M = 128 with "
                       "cta_group::1, M = 128 or 256 with cta_group::2, N <= 256)")  # fmt: skip
        lane, col = self._addr(tr, i, i.operands[0], "tcgen05.mma D")
        d_cols = n // 2 if (m, ncta) == (128, 2) else n  # ops_tmem._d_cols
        operands, shapes = [lane, col], [("d", True, d_cols)]
        if i.operands[1][0] == "mem":  # A from TMEM
            operands += self._addr(tr, i, i.operands[1], "tcgen05.mma A")
            shapes.append(("a", False, 8))
        if bs:
            for k, ncols in ((4, 4), (5, 4 * -(-n // 128))):
                operands += self._addr(tr, i, i.operands[k], "tcgen05.mma SF", sf=True)
                shapes.append(("sf", False, ncols))
        mask = _pair_mask(tr, i, mn)
        if mask is not None:
            operands.append(mask)
        loc, site = tr.loc(), self._site(tr, i)
        self._one_thread(tr, i, operands, lambda v, j: {
            "op": "event", "ev": "tmem", "inst": "mma", "warp": j(E.const(tr.w0, 32)),
            "rects": [_rect(u, wr, v[2 * k], 128, v[2 * k + 1], nc, j) for k, (u, wr, nc) in enumerate(shapes)],
            **({"cta_mask": j(v[-1])} if mask is not None else {}), "site": site, "loc": loc,
        }, "tcgen05.mma")  # fmt: skip

    def _alloc(self, tr: interp.Translator, i) -> None:
        n = tr.operand(i, 1, 32)
        u = None if isinstance(n, interp.Data) else tr.uniform(n)
        if u is None or not E.is_const(u):
            tr.fail(i, "R2: tcgen05.alloc with a non-constant column count")
        if tr.loops:
            tr.fail(i, "R2: tcgen05.alloc inside a loop")
        _uni, var = tr.split_mask()
        act = tr.active(var)
        if act is None:
            tr.fail(i, "R2: tcgen05.alloc under a runtime lane condition")
        if act:  # warp-collective: one allocation per issuing warp
            n = tr.full_warps(act, "tcgen05.alloc")
            self._role(tr).allocs[id(i)] = (n, u[2], tr.loc())

    def _addr(self, tr: interp.Translator, i, op, what: str, sf: bool = False) -> list:
        """[lane, column] (i32) of the TMEM address operand [a + off] (ISA: lane in bits
        31-16, column in 15-0); an SF address without its byte bits 30-31."""
        a = tr.k.mem.address(tr, op)
        if isinstance(a, interp.Data) or a is None:
            origin = a.origin if a is not None else "an undefined register"
            tr.fail(
                i,
                f"R1: {what}: data value from {origin} used as a TMEM address (slice soundness)",
            )
        if E.reserved(a):
            tr.fail(i, f"R1: {what}: a memory address used as a TMEM address")
        self._role(tr).accessed = True
        a = E.resize(a, 32, False)
        if sf:
            a = E.bin("and", a, E.const(0x3FFFFFFF, 32))
        return [E.bin("lshr", a, E.const(16, 32)), E.bin("and", a, E.const(0xFFFF, 32))]

    def _one_thread(
        self, tr: interp.Translator, i, operands: list, build, what: str
    ) -> None:
        """A cp/shift/mma: issued by one thread of the role's only warp (the elected lane
        or a fixed thread), its operands the same in every lane (ops_tmem.one_warp_event)."""
        if tr.nw != 1:
            tr.fail(
                i,
                f"R6: {what} in a role of {tr.nw} warps (the landing rule needs the issuing thread)",
            )
        _uni, var = tr.split_mask()
        act = tr.active(var)
        if act is None:
            tr.fail(
                i, f"R6: {what} under a lane condition that depends on runtime values"
            )
        if not act:
            return
        if len(act) != 1:
            tr.fail(i, f"R6: {what} issued by {len(act)} threads")
        if any(tr.uniform(e) is None for e in operands):
            tr.fail(i, f"R6: {what}: an operand varies across the lanes of the warp")
        r = self._role(tr)
        r.async_ops = True
        r.issuers.setdefault(_issuer(var, act), tr.loc())
        tr.thread_event(operands, build)

    def _site(self, tr: interp.Translator, i) -> str:
        """The closest preceding location in another file (ops_tmem.site_of over the
        loc history; here the static instruction order): the kernel line that reached the
        DSL helper an MMA is located in ("" under DSL 4.4.2, whose PTX names one file)."""
        instrs = tr.k.fn.instructions()
        if not self._index:
            self._index = {id(x): k for k, x in enumerate(instrs)}
        here = tr.loc(i).rsplit(":", 1)[0]
        for k in range(self._index[id(i)] - 1, -1, -1):
            loc = tr.k.locs.get(id(instrs[k]), "unknown")
            if loc != "unknown" and loc.rsplit(":", 1)[0] != here:
                return loc
        return ""


# ------------------------------------------------------------------ helpers


def _mem(i):
    return next(op for op in i.operands if op[0] == "mem")


def _rect(use, write, lane, nlanes, col, ncols, j) -> dict:
    return {"use": use, "write": write, "lane": j(lane), "nlanes": nlanes, "col": j(col),
            "ncols": ncols}  # fmt: skip


def _field(e, lo: int, n: int):
    """Bits [lo, lo + n) of the i32 e if every one is known (expr._bit), else None."""
    v = 0
    for k in range(n):
        b = E._bit(e, lo + k)
        if b is None or not E.is_const(b):
            return None
        v |= b[2] << k
    return v


def _issuer(var, act):
    """Who issues an event under the thread-varying mask `var`: "elect" (the elected lane
    of an elect.sync over the full warp: the same lane for every elect.sync of the warp,
    ISA elect.sync) or the issuing threads (threads() indices; None if unknown)."""
    if any(interp._bare_elect(c) for c in interp._conjuncts(var)):
        return "elect"
    return tuple(act) if act is not None else None


def _pair_mask(tr: interp.Translator, i, mn: str):
    """The CTA mask of a cta_group::2 access, both CTAs of the issuing CTA's pair
    (cluster.pair_mask); None for cta_group::1."""
    if "cta_group::2" not in mn:
        return None
    if math.prod(tr.k.cluster) % 2:
        tr.fail(
            i, f"R5: {mn.split('.')[1]} cta_group::2 without a cluster of CTA pairs"
        )
    return cluster.pair_mask(tr)


def _per_warp(tr: interp.Translator, operands: list, build, what: str) -> None:
    """One event per issuing warp (ops_tmem.per_warp_event): every lane of an issuing
    warp active, the operands uniform in each warp. build(values, warp, to_json) -> KIR
    statement. A `for` over the role's warps when every warp issues it, else one event per
    issuing warp."""
    uni, var = tr.split_mask()
    act = tr.active(var)
    if act is None:
        tr.fail(
            tr.cur, f"R6: {what} under a lane condition that depends on runtime values"
        )
    if not act:
        return
    warps = sorted({k // 32 for k in act})
    if any(sum(1 for k in act if k // 32 == w) != 32 for w in warps):
        tr.fail(tr.cur, f"R6: {what} with only part of a warp active")
    if any(tr.per_warp(e) is None for e in operands):
        tr.fail(tr.cur, f"R6: {what}: an operand varies across the lanes of a warp")
    loc = tr.loc()
    if len(warps) == 1 or len(warps) != tr.nw:
        for w in warps:
            warp = E.const(tr.w0 + w, 32)
            vals = [
                E.subst(e, {"%warp": warp, "%lane": E.const(0, 32)}) for e in operands
            ]
            tr._emit_under(uni, build(vals, warp, tr.json))
        return
    # Every warp: `for wq in [0, nw)` with the warp wq + w0; the operands' uniform parts
    # are bound before the loop, what the body uses twice at its top (as _thread_loop).
    tr.flush()
    tr._bind_deps([interp._lowered(e) for e in operands])
    t = tr.fresh("wq")
    warp = E.bin("add", E.var(t, 32), E.const(tr.w0, 32))
    m = {"%warp": warp, "%lane": E.const(0, 32)}
    vals = [interp._lowered(E.subst(interp._lowered(e), m)) for e in operands]
    names = dict(tr.names)
    order, shared = interp._hoistable(vals, names)
    body = []
    for x in order:
        if x in shared:
            n = f"{t}_{len(names) - len(tr.names)}"
            body.append(
                {"op": "let", "var": n, "expr": E.to_json(x, names, n), "loc": loc}
            )
            names[x] = n
    body.append(build(vals, warp, lambda e: E.to_json(interp._lowered(e), names)))
    tr._emit_under(uni, {"op": "for", "var": t, "lb": interp._c32(0), "ub": interp._c32(tr.nw),
                         "step": interp._c32(1), "body": body, "loc": loc})  # fmt: skip


def _role_event(tr: interp.Translator, stmt: dict, what: str) -> None:
    """An event of the whole role (tcgen05.wait, named barriers): every thread of the role
    executes it, under thread-uniform conditions only (ops_tmem.role_event)."""
    uni, var = tr.split_mask()
    act = tr.active(var)
    if act is None or (act and len(act) != 32 * tr.nw):
        tr.fail(
            tr.cur,
            f"R7: {what} reached by only part of the role (or under a runtime lane condition)",
        )
    if act:
        tr._emit_under(uni, stmt)


def _named_bar(tr: interp.Translator, i, mn: str) -> None:
    """bar/barrier.sync|arrive id[, count]: a named_bar of every thread of the role; count
    absent, every thread of the block (ops_tmem._named_bar_asm, nvvm.barrier)."""

    def value(k: int, what: str) -> int:
        if k >= len(i.operands):
            return tr.k.block[0]
        e = tr.operand(i, k, 32)
        u = None if isinstance(e, interp.Data) else tr.uniform(e)
        if u is None and not isinstance(e, interp.Data):
            tr.split_per_warp(e)  # one role per run of warps with one value
        if u is None or not E.is_const(u):
            tr.fail(i, f"R7: named barrier {what} is not a constant")
        return u[2]

    bid, count = value(0, "id"), value(1, "thread count")
    if count <= 0 or count % 32:
        tr.fail(
            i,
            f"R7: named barrier thread count {count} is not a positive multiple of 32",
        )
    sync = ".sync" in mn
    _role_event(tr, {"op": "event", "ev": "named_bar", "id": interp._c32(bid),
                     "count": interp._c32(count), "sync": sync, "loc": tr.loc()},
                f"{mn.split('.')[0]}.{'sync' if sync else 'arrive'}")  # fmt: skip
