# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Synchronization through global memory as KIR events (policy `gsync` True; kir.md change 18;
references/frontend_ptx.md, "Global synchronization").

The census of gsync.py finds and classifies the sites; this module turns them into events of
the role being translated (interp.py), with the existing slicing and exact integer
expressions, and refuses every other shape:

* Sync tensors. A tensor is a sync tensor (KIR `sync`) if a spin wait loads it or a claim
  targets it, or the variant reads it as metadata and writes it; the case may add more
  (`sync_tensors`). The set is found by translating again (Restart) until it is stable. Every
  access of a sync tensor is a sync event: never a data event, never a metadata load.
* Tagged words: a spin over 64-bit words (regions.GSpin, the census's `tag_eq` idiom) that exits
  once every word's high half (its tag) equals its target is one `gwait` per word, binding the
  low half (tag_spin); a store to a sync word is a `gsig`; a store of a tagged word whose value
  half is data stores `tag << 32` with `data_lo`. A trap-on-timeout
  path makes `timeout` true (taken to be never taken). Registers the loop writes are Data after
  it.
* Claims (policy `claims` "representative"): a fetch-add whose old value the census classifies as
  a claim (it indexes memory or bounds a loop) is a `gclaim`, its old value a fresh variable;
  the claimers are the CTAs that claim on the word, in linear order (lean/KV/Eval.lean gives the
  k-th claim the value of the chain a fetch-add hands out, `claimsFeasible`).
* Relay channels (policy `relay_channels` [{symbol (regex), start, slots, stride, name?}]): a
  shared array region of `slots` slots of `stride` bytes, written once per slot and CTA (a claim
  ring): st.shared to slot k is `rput`, ld.shared `rget` (Lean: the value the CTA's earlier role
  or this role wrote; trusted: one write per slot and CTA). The slot is the constant slot, or X
  for an offset stride x (X mod slots) (also C's `%` by a power of two,
  `X - ((X + bias) & -slots)`).
* named_bar events (bar.sync, bar.arrive, barrier.sync...) in the P0 KIR.
* Roles: a global sync event issued by only part of the warps of a role splits the role
  (SplitWarps), as the P1 pass does for mbarrier events: a peer warp's writes are ordered before
  an elected lane's signal only by a named barrier between them.
"""

import re

from . import expr as E, gsync as GS, interp, ops, regions as R


class Restart(Exception):
    """The translation found more sync tensors: translate the variant again with them."""

    def __init__(self, names):
        super().__init__(f"sync tensors {sorted(names)}")
        self.names = set(names)


class State:
    """What the roles of one kernel share: the census of the entry, spin loops by header
    line, the sync tensors and the relay channels."""

    def __init__(self, fn, sync: set):
        self.fn = fn
        self.a = GS._Fn(fn, GS.HELPERS)
        self.sites = self.a.sites()
        self.by_line = {}
        for s in self.sites:
            self.by_line.setdefault(s["line"], s)
        self.spins = {
            s["header_line"]: s
            for s in self.sites
            if s["kind"] == "wait" and s.get("header_line")
        }
        self.sync = set(sync)
        # policy claims / relay_channels, set by translate.py
        self.claims = False
        self.channels: list = []
        self.used: set = (
            set()
        )  # "claims": the variant leans on a representative schedule
        self.relay_data: set = set()  # (channel, field) a role stored a data value to

    def need(self, name: str) -> None:
        if name not in self.sync:
            raise Restart({name})


def seed(sites: list, table: list, tensors: dict) -> set:
    """The tensors the census's sync sites (spin waits, fetch-adds)
    address through a kernel parameter or struct field (`base` "<param>+<offset>"), by the
    parameter table's names (a view names its base tensor)."""
    by_ptx = {p["ptx"]: p for p in table}
    out = set()
    for s in sites:
        sync = s["kind"] in ("wait", "fetch")
        b = s.get("base")
        if (
            not sync
            or not b
            or s.get("idiom") in ("timer_delay", "shared_spin", "shared_flag_loop")
        ):
            continue
        name, _, off = b.rpartition("+")
        p = by_ptx.get(name)
        if p is None or not off.isdigit():
            continue
        t = (
            p.get("name")
            if p["kind"] == "ptr"
            else (p["fields"].get(int(off)) or {}).get("name")
        )
        if t:
            out.add(tensors.get(t, {}).get("base", t))
    return {
        t
        for t in out
        if t in tensors or any(v.get("base") == t for v in tensors.values())
    }


def split_partial(tr, act) -> None:
    """A global sync event issued by only part of the warps of a role splits it (module
    doc); `act` the issuing threads (threads() indices; None: runtime dependent)."""
    if tr.nw == 1 or not tr.k.split:
        return
    if act is None:
        tr.fail(tr.cur, "a global synchronization event under a lane condition that depends on runtime "
                        f"values in a role of {tr.nw} warps")  # fmt: skip
    issuing = {k // 32 for k in act}
    if not issuing or len(issuing) == tr.nw:
        return
    groups, start = [], 0
    for w in range(1, tr.nw + 1):
        if w == tr.nw or (w in issuing) != (w - 1 in issuing):
            groups.append((tr.w0 + start, w - start))
            start = w
    raise interp.SplitWarps(groups)


def sync_event(tr, operands: list, build, release_only: bool = False) -> None:
    """A sync event per issuing thread (interp.thread_event), splitting a role whose warps
    issue it only in part, except a signal (`release_only`: only events before it matter)
    right after a named barrier of the whole role with no event of the role in between: the
    barrier already orders every warp's earlier accesses before it (seam item_publish's
    `bar 3; elected red.release`)."""
    if not (release_only and _right_after_bar(tr)):
        _uni, var = tr.split_mask()
        split_partial(tr, tr.active(var))
    tr.gs_bar = None
    tr.thread_event(operands, build, tag="gsync")


def _right_after_bar(tr) -> bool:
    """The role emitted nothing but lets since its last named barrier (mark_bar)."""
    m = getattr(tr, "gs_bar", None)
    if (
        m is None
        or tr.pending is not None
        or m[0] is not tr.stmts
        or len(tr.stmts) < m[1]
    ):
        return False
    return all(s.get("op") == "let" for s in tr.stmts[m[1] :])


def mark_bar(tr) -> None:
    """A named barrier of the whole role was just emitted (mem.py)."""
    tr.flush()
    tr.gs_bar = (tr.stmts, len(tr.stmts))


def _word(tr, i, op, nbytes: int):
    """(tensor, element offset) of a sync access at operand op: one whole element."""
    g = tr.k.mem._global(tr, i, op, nbytes)
    if g is None:
        tr.fail(i, "a sync access to shared memory")
    name, off, n = g
    if n != 1:
        tr.fail(
            i,
            f"a {n}-element access of the sync tensor {name} (one word per sync access)",
        )
    return name, off


def width_of(tr, name: str) -> int:
    return tr.k.mem.tensors[name]["elem_bytes"] * 8


def store(tr, i, name: str, off) -> None:
    """A store to a sync word: `gsig`. A 64-bit tagged word (value, tag) whose value half is
    data (an fp32 partial) and whose tag half is an integer stores `tag << 32` with `data_lo`
    (its takes read the tag; a take's variable has no value)."""
    st = tr.k.gsync
    st.need(name)
    w = width_of(tr, name)
    val = i.operands[1]
    if val[0] not in ("reg", "imm"):
        tr.fail(i, f"a vector store to the sync tensor {name}")
    v = tr.operand(i, 1, w)
    data_lo = False
    if isinstance(v, interp.Data) and val[0] == "reg" and w == 64:
        hi = _packed_tag(tr, i, val[1])
        if hi is not None:
            v, data_lo = E.shl_bits(E.resize(hi, 64, False), 32), True
    v = tr.as_int(v, f"the value stored to the sync word {name}")
    loc = tr.loc()
    extra = {"data_lo": True} if data_lo else {}
    sync_event(tr, [off, v], lambda x, j: {
        "op": "event", "ev": "gsig", "tensor": name, "index": j(x[0]), "value": j(x[1]),
        **extra, "loc": loc}, release_only=True)  # fmt: skip


def _packed_tag(tr, i, reg: str):
    """The integer high half of a 64-bit register that the inline asm block of i packs
    (`mov.b64 reg, {lo, hi}`, the last write of reg before i in the block), else None."""
    if i.asm is None:
        return None
    body = tr.k.fn.instructions()
    try:
        k = next(n for n, x in enumerate(body) if x is i)
    except StopIteration:
        return None
    for x in reversed(body[:k]):
        if x.asm != i.asm:
            return None
        if reg in R.dests(x):
            src = x.operands[1] if len(x.operands) > 1 else None
            if x.base != "mov" or src is None or src[0] != "vec" or len(src[1]) != 2:
                return None
            hi = tr.operand_value(src[1][1])
            if isinstance(hi, interp.Data) or E.width(hi) != 32:
                return None
            return hi
    return None


def atomic(tr, i, mn: str, name: str, off, ret: bool) -> None:
    """red / atom on a sync word: a claim (`gclaim`) under the policy claims, else refused."""
    st = tr.k.gsync
    st.need(name)
    w = width_of(tr, name)
    site = st.by_line.get(i.line) or {}
    idiom = site.get("idiom") if site.get("kind") == "fetch" else None
    mods = i.opcode.split(".")
    if not (ret and idiom == "claim" and st.claims and "add" in mods):
        tr.fail(
            i,
            f"{mn} on the sync tensor {name}: only claims (policy claims) are modeled",
        )
    v = tr.as_int(
        tr.operand(i, 2, w), f"the operand of a claim on the sync word {name}"
    )
    _claim(tr, i, name, off, v, w, tr.loc())


def _claim(tr, i, name: str, off, v, w: int, loc: str) -> None:
    """A claim (module doc): `gclaim` by its one issuing thread, its old value a fresh variable."""
    st = tr.k.gsync
    uni, var = tr.split_mask()
    if not E.is_const(uni, 1):
        tr.fail(i, f"a claim on the sync word {name} under a uniform mask")
    act = tr.active(var)
    if act is None or len(act) != 1:
        tr.fail(i, f"a claim on the sync word {name} issued by {len(act) if act else 'a runtime set of'} "
                   "threads (one issuing thread per claim)")  # fmt: skip
    split_partial(tr, act)
    wl = tr.threads()[act[0]]
    sub = tr.thread_subst(*wl, E.free_vars(off) | E.free_vars(v))
    idx, val = E.subst(off, sub), E.subst(v, sub)
    if E.thread_varying(idx) or E.thread_varying(val):
        tr.fail(
            i,
            f"a claim on the sync word {name} whose word or add depends on the thread",
        )
    bind = tr.fresh("claim")
    tr.emit({"op": "event", "ev": "gclaim", "tensor": name, "index": tr.json(idx),
             "value": tr.json(E.resize(val, w, False)), "var": bind, "w": w, "loc": loc})  # fmt: skip
    st.used.add("claims")
    dst = i.operands[0]
    if dst[0] == "reg":
        tr.write(dst[1], E.var(bind, w))


def spin(tr, node) -> None:
    """A spin wait (regions.GSpin; module doc): a blocking tagged-word spin, else refused."""
    site = node.site
    first = tr.k.mem.cfg.blocks[node.header].instrs[0]
    tr.cur = first
    if site.get("idiom") != "tag_eq" or not site.get("blocking"):
        tr.fail(first, f"a spin wait of shape {site.get('idiom')} ({'blocking' if site.get('blocking') else 'it may return '
                       'without its predicate'}): only blocking spins over tagged words are translated")  # fmt: skip
    tag_spin(tr, node)


def _tag_halves(V):
    """(low, high) 32-bit halves of the 64-bit value V, as interp.py unpacks them."""
    return tuple(E.trunc(E.bin("lshr", V, E.const(pos, 64)), 32) for pos in (0, 32))


def tag_spin(tr, node) -> None:  # noqa: C901 (one refusal per shape condition)
    """A blocking spin over tagged words (64-bit (value, tag) pairs, the tag the high half):
    the loop is one block that loads each word with one strong 64-bit load and exits once every
    word's tag equals its target (tl.inline_asm spins: `ld.relaxed.b64 d; mov.b64 {v, t}, d;
    setp.ne t, T; @p bra`). One `gwait` per word, in load order, with the tag as `target` and
    `var` bound to the low half (Lean gives it the value the word's store of
    that tag carries). Checked exactly: every loaded word reaches the exit
    condition only through one comparison of its high half with a target that no load reads,
    and the condition, as a function of those comparisons, is true exactly when every tag
    matches (all 2^n cases). After the loop the header's registers hold their last trip's
    values: a tag its target, a value the bound variable (others that read a loaded word are
    Data)."""
    st = tr.k.gsync
    site = node.site
    cfg = tr.k.mem.cfg
    hdr = list(cfg.blocks[node.header].instrs)
    first = hdr[0]
    if set(node.blocks) != {node.header}:
        tr.fail(first, "a tagged-word spin over several blocks")
    lines = set(site.get("load_lines") or ())
    loads = [x for x in hdr if x.line in lines and x.base == "ld"]
    if not loads:
        tr.fail(first, "a tagged-word spin without its loads in the loop's block")
    for x in loads:
        if GS_bits(x.opcode) != 64 or x.operands[0][0] != "reg":
            tr.fail(x, "a tagged-word spin load that is not one 64-bit register load")
        if not set(x.opcode.split(".")[1:]) & {"relaxed", "acquire", "volatile"}:
            tr.fail(x, "a tagged-word spin load that is not strong")
    last = hdr[-1]
    if last.base != "bra" or last.guard is None:
        tr.fail(first, "a spin wait whose header block does not end in its exit branch")
    snap = tr.snapshot()
    saved_mask = tr.mask
    words, vs = [], []
    try:
        for x in hdr[:-1]:
            tr.cur = x
            if x in loads:
                name, off = _word(tr, x, x.operands[1], 8)
                st.need(name)
                V = E.var(tr.fresh("__spinv"), 64)
                words.append((x, name, off))
                vs.append(V)
                tr.env[x.operands[0][1]] = V
                continue
            cls, _why = ops.classify(x)
            if cls not in ("value", "control", "drop"):
                tr.fail(x, "an event in a spin wait's header block")
            if x.guard is None:
                tr.execute(x, cls)
            else:
                tr.predicated(x, cls)
        reg, neg = last.guard
        p = tr.env.get(reg)
        if p is None or isinstance(p, interp.Data):
            tr.fail(
                last,
                "a tagged-word spin's exit predicate is not an integer expression of the loaded tags",
            )
        target = last.operands[0][1]
        exits_on_true = target == node.exit or _maps_to(cfg, target, node.exit)
        c = E.not1(p) if neg else p
        c = c if exits_on_true else E.not1(c)
    finally:
        tr.restore(snap)
        tr.mask = saved_mask
    targets = _tag_targets(tr, first, c, vs)
    timeout = site.get("timeout") == "trap"
    binds = []
    for (x, name, off), tgt in zip(words, targets):
        if width_of(tr, name) != 64:
            tr.fail(
                x,
                f"a tagged-word spin on {name}, a sync tensor of {width_of(tr, name)}-bit words",
            )
        loc = tr.k.locs.get(id(x), tr.loc())
        tr.cur = x
        stmt = {
            "op": "event",
            "ev": "gwait",
            "tensor": name,
            "timeout": timeout,
            "loc": loc,
        }
        bind = _take(tr, x, off, tgt, stmt)
        binds.append(bind)
    # the header's last trip: every load reads the word's store of its tag
    for x in hdr[:-1]:
        tr.cur = x
        if x in loads:
            tr.env[x.operands[0][1]] = vs[loads.index(x)]
            continue
        cls, _why = ops.classify(x)
        if x.guard is None:
            tr.execute(x, cls)
        else:
            tr.predicated(x, cls)
    for r in tr.tree.defs.get(node.header, ()):
        e = tr.env.get(r)
        if e is None or isinstance(e, interp.Data):
            continue
        for V, tgt, bind in zip(vs, targets, binds):
            lo, hi = _tag_halves(V)
            e = E.subst_node(e, hi, tgt)
            if bind is not None:
                e = E.subst_node(e, lo, E.var(bind, 32))
        if any(V in _nodes(e) for V in vs):
            tr.env[r] = interp.Data(
                f"a register the tagged-word spin at {tr.loc(first)} writes"
            )
        else:
            tr.env[r] = e


def _take(tr, x, off, tgt, stmt: dict):
    """The take of a tagged-word spin's word: one of the role, its value half bound to a fresh
    variable (returned), when every thread that runs the spin takes the same word and tag;
    else one per thread (sync_event) whose value half is Data (None)."""
    uni, var = tr.split_mask()
    act = tr.active(var)
    if act is None:
        tr.fail(x, "a tagged-word spin run by a runtime set of threads")
    if not act:
        return None
    if E.thread_varying(off) or E.thread_varying(tgt):
        subs = [
            tr.thread_subst(*tr.threads()[t], E.free_vars(off) | E.free_vars(tgt))
            for t in act
        ]
        if (
            len({E.subst(off, m) for m in subs}) != 1
            or len({E.subst(tgt, m) for m in subs}) != 1
        ):
            sync_event(
                tr,
                [off, tgt],
                lambda y, j: {**stmt, "index": j(y[0]), "target": j(y[1])},
            )
            return None
        off, tgt = E.subst(off, subs[0]), E.subst(tgt, subs[0])
    split_partial(tr, act)
    bind = tr.fresh("tagv")
    tr.gs_bar = None
    tr._emit_under(
        uni,
        {**stmt, "index": tr.json(off), "target": tr.json(tgt), "var": bind, "w": 32},
    )
    return bind


def _tag_targets(tr: interp.Translator, first, c, vs) -> list:
    """The tag each word of a tagged-word spin waits for (tag_spin's check)."""
    halves = [_tag_halves(V) for V in vs]
    atoms, targets = [], [None] * len(vs)
    for n in _nodes(c):
        if n[0] != "cmp" or not any(V in _nodes(n) for V in vs):
            continue
        k = next((k for k, (_lo, hi) in enumerate(halves) if hi in (n[2], n[3])), None)
        if k is None or n[1] not in ("eq", "ne"):
            tr.fail(first, "a tagged-word spin whose exit condition reads a loaded word other than "
                           "by comparing its high half (the tag) for equality")  # fmt: skip
        tgt = n[3] if n[2] is halves[k][1] else n[2]
        if any(V in _nodes(tgt) for V in vs):
            tr.fail(
                first, "a tagged-word spin whose tag target depends on a loaded word"
            )
        if targets[k] is not None:
            tr.fail(first, "a tagged-word spin that compares one word's tag twice")
        targets[k] = tgt
        atoms.append((n, n[1] == "eq"))
    if any(t is None for t in targets):
        tr.fail(
            first,
            "a tagged-word spin with a loaded word its exit condition does not test",
        )
    rest = c
    for n, _ in atoms:
        rest = E.subst_node(rest, n, E.const(0, 1))
    if any(V in _nodes(rest) for V in vs):
        tr.fail(first, "a tagged-word spin whose exit condition reads a loaded word outside its "
                       "tag comparisons")  # fmt: skip
    for bits in range(1 << len(atoms)):
        x = c
        for k, (n, _eq) in enumerate(atoms):
            x = E.subst_node(x, n, E.const((bits >> k) & 1, 1))
        if not E.is_const(x):
            tr.fail(first, "a tagged-word spin whose exit condition is not a function of its tag "
                           "comparisons alone")  # fmt: skip
        want = all(((bits >> k) & 1) == int(eq) for k, (_n, eq) in enumerate(atoms))
        if bool(x[2] & 1) != want:
            tr.fail(first, "a tagged-word spin that does not exit exactly when every word carries "
                           "its tag")  # fmt: skip
    return targets


def GS_bits(opcode: str) -> int:
    m = re.search(r"\.[bsu](8|16|32|64)\b", opcode)
    return int(m.group(1)) if m else 32


def _maps_to(cfg, label: str, exit_: str) -> bool:
    """`label` is the exit block."""
    return label == exit_


def _nodes(e):
    out, todo, seen = [], [e], set()
    while todo:
        x = todo.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        out.append(x)
        todo.extend(E.children(x))
    return out


def _sum_terms(e, sign: int = 1) -> tuple:
    """(terms [(sign, expr)], constant) of e's top-level sum of adds and subtracts."""
    if E.is_const(e):
        return [], sign * e[2]
    if e[0] == "bin" and e[2] in ("add", "sub"):
        ta, ca = _sum_terms(e[3], sign)
        tb, cb = _sum_terms(e[4], sign if e[2] == "add" else -sign)
        return ta + tb, ca + cb
    return [(sign, e)], 0


def _scaled(e) -> tuple:
    """(k, X) with e == k x X: constant multiplies and left shifts, through extensions."""
    if e[0] == "ext":
        k, x = _scaled(e[3])
        return k, x
    if e[0] == "bin" and e[2] == "mul":
        for a, b in ((e[3], e[4]), (e[4], e[3])):
            if E.is_const(b):
                k, x = _scaled(a)
                return k * b[2], x
    if e[0] == "bin" and e[2] == "shl" and E.is_const(e[4]):
        k, x = _scaled(e[3])
        return k << e[4][2], x
    return 1, e


def _ring_rem_terms(terms: list, slots: int, stride: int):
    """X if the terms sum to stride x (X srem slots), slots a power of two, the remainder in
    LLVM's form `X - ((X + bias) & -slots)` (bias `X lshr (w - 1)` for 2 slots, else
    `(X ashr (w - 1)) lshr (w - k)`: C's `%` on an int) and the multiple split over several
    terms of it (`8 R - 4 R`); else None. The slot is keyed by X, the unreduced counter."""
    if slots < 2 or slots & (slots - 1):
        return None
    coef: dict = {}
    for sg, t in terms:
        k, x = _scaled(t)
        coef[x] = coef.get(x, 0) + sg * k
    live = [(x, k) for x, k in coef.items() if k]
    if len(live) != 1 or live[0][1] != stride:
        return None
    r = live[0][0]
    if r[0] != "bin" or r[2] != "sub":
        return None
    X, m = r[3], r[4]
    w = E.width(X)
    if m[0] != "bin" or m[2] != "and" or not E.is_const(m[4]) or m[4][2] != -slots:
        return None
    a = m[3]
    if a[0] != "bin" or a[2] != "add" or a[3] is not X:
        return None
    b = a[4]
    k = slots.bit_length() - 1
    if slots == 2:
        ok = b[0] == "bin" and b[2] == "lshr" and b[3] is X and E.is_const(b[4], w - 1)
    else:
        ok = (b[0] == "bin" and b[2] == "lshr" and E.is_const(b[4], w - k) and b[3][0] == "bin"
              and b[3][2] == "ashr" and b[3][3] is X and E.is_const(b[3][4], w - 1))  # fmt: skip
    return X if ok else None


def _channel_at(tr, i, a, nbytes: int):
    """_channel of the address expression a."""
    from .mem import peel  # mem.py imports interp, which imports this module lazily

    pe = peel(a)
    if pe is None or pe[0][2][:3] not in (
        "%s:",
        "%S:",
    ):  # a window or generic shared address
        return None
    sym = pe[0][2][3:]
    terms, c = _sum_terms(pe[1])
    for ch in tr.k.gsync.channels:
        lo, n, stride = ch["start"], ch["slots"], ch["stride"]
        if not re.search(ch["symbol"], sym) or not lo <= c < lo + n * stride:
            continue
        name = ch.get("name") or f"{sym}+{lo}"
        rel = c - lo
        if (rel % stride) % nbytes:
            tr.fail(
                i,
                f"a {nbytes}-byte access of the relay channel {name} off its field grid",
            )
        field = (rel % stride) // nbytes
        if not terms:
            return name, E.const(rel // stride, 32), field
        x = _ring_rem_terms(terms, n, stride) if rel < stride else None
        if x is not None:
            return name, E.resize(x, 32, False), field
        lin = [(sg * k, x) for sg, t in terms for k, x in [_scaled(t)]]
        if all(
            k % stride == 0 for k, _x in lin
        ):  # slot = rel / stride + sum (k / stride) X
            slot = E.const(rel // stride, 32)
            for k, x in lin:
                slot = E.bin(
                    "add",
                    slot,
                    E.bin("mul", E.resize(x, 32, False), E.const(k // stride, 32)),
                )
            return name, slot, field
        tr.fail(i, f"a shared access of the relay channel {name} at an offset that is not a slot of "
                   f"{stride} bytes (stride x (X mod {n}) + field)")  # fmt: skip
    return None


def _access_bytes(i) -> int:
    t = next(
        (x for x in i.opcode.split(".")[1:] if re.fullmatch(r"[usb](8|16|32|64)", x)),
        "u32",
    )
    return int(t[1:]) // 8


def _elements(op) -> list:
    return list(op[1]) if op[0] == "vec" else [op]


def _channel_threads(tr, i, op, nb: int, k: int):
    """Per active thread of a relay channel access at operand op, element k of a vector access:
    [(thread index, substitution, (channel, slot, field))]; [] where no thread is active; None if
    the access is not in a channel. A thread-varying address (each lane its own field, e.g.
    s_buf[4 + lane]) is resolved per thread."""
    st = tr.k.gsync
    if not st.channels or op[0] != "mem" or op[3] is not None:
        return None
    a = tr.k.mem.address(tr, op)
    if isinstance(a, interp.Data):
        return None
    a = E.bin("add", a, E.const(k * nb, E.width(a))) if k else a
    return _channel_threads_at(tr, i, a, nb)


def _channel_threads_at(tr, i, a, nb: int):
    """_channel_threads of the address expression a."""
    uni, var = tr.split_mask()
    act = tr.active(var)
    if (
        act is None
    ):  # a runtime set of threads: not relayed (Data, refused where it reaches a root)
        return None
    if not act:
        return (
            [] if E.thread_varying(a) or _channel_at(tr, i, a, nb) is not None else None
        )
    out = []
    for t in act:
        wl = tr.threads()[t]
        sub = tr.thread_subst(*wl, E.free_vars(a))
        ch = _channel_at(tr, i, E.subst(a, sub), nb)
        if ch is None:
            if out:
                tr.fail(
                    i,
                    "threads access a relay channel and other shared memory in one instruction",
                )
            return None
        out.append((t, sub, ch))
    return out


def channel_store(tr, i) -> bool:
    """st.shared into a relay channel: one `rput` per element and per distinct (slot, field) the
    active threads store (module doc); True if handled."""
    nb = _access_bytes(i)
    vals = _elements(i.operands[1])
    st = tr.k.gsync
    handled = False
    for k, vop in enumerate(vals):
        per = _channel_threads(tr, i, i.operands[0], nb, k)
        if per is None:
            if handled:
                tr.fail(i, "a vector store partly in a relay channel")
            return False
        handled = True
        if not per:
            continue
        v = tr.operand_value(vop, nb * 8)
        if not isinstance(v, interp.Data) and E.width(v) != nb * 8:
            v = E.resize(v, nb * 8, False)
        uni, _var = tr.split_mask()
        done: dict = {}
        for _t, sub, (name, slot, field) in per:
            if isinstance(v, interp.Data):
                st.relay_data.add((name, field))
                continue
            sub = {**sub, **tr.thread_subst(*tr.threads()[_t], E.free_vars(v))}
            val, slot1 = E.subst(v, sub), E.subst(slot, sub)
            if interp._has_elect(val) or interp._has_elect(slot1):
                tr.fail(
                    i,
                    f"a store to the relay channel {name} depends on the elected lane",
                )
            key = (name, str(slot1), field)
            if key in done:
                if done[key] != val:
                    tr.fail(
                        i,
                        f"threads store different values to one slot of the relay channel {name}",
                    )
                continue
            done[key] = val
            tr._emit_under(uni, {"op": "event", "ev": "rput", "chan": name, "ord": tr.json(slot1), "field": field,
                                 "value": tr.json(val), "loc": tr.loc()})  # fmt: skip
    return handled


def channel_load(tr, i) -> bool:
    """ld.shared from a relay channel: one `rget` per element and per distinct (slot, field) the
    active threads read; each register its variable (per thread where the threads read
    different fields: a select over the thread's warp and lane); True if handled."""
    nb = _access_bytes(i)
    dsts = _elements(i.operands[0])
    if any(d[0] != "reg" for d in dsts):
        return False
    st = tr.k.gsync
    uni, _var = tr.split_mask()
    handled = False
    for k, d in enumerate(dsts):
        per = _channel_threads(tr, i, i.operands[1], nb, k)
        if per is None:
            if handled:
                tr.fail(i, "a vector load partly in a relay channel")
            return False
        handled = True
        if not per:
            continue
        if not E.is_const(uni, 1):
            tr.fail(
                i, f"a read of the relay channel {per[0][2][0]} under a uniform mask"
            )
        groups: dict = {}
        for t, sub, (name, slot, field) in per:
            slot1 = E.subst(slot, sub) if sub else slot
            if interp._has_elect(slot1):
                tr.fail(
                    i,
                    f"a read of the relay channel {name} at a slot that depends on the elected lane",
                )
            groups.setdefault((name, str(slot1), field), (name, slot1, field, []))[
                3
            ].append(t)
        w = nb * 8
        value = None
        for name, slot1, field, ts in groups.values():
            if (name, field) in st.relay_data:
                tr.env[d[1]] = interp.Data(
                    f"a data value relayed through {name} at {tr.loc()}"
                )
                value = None
                break
            bind = tr.fresh("relay")
            ev = {"op": "event", "ev": "rget", "chan": name, "ord": tr.json(slot1), "field": field,
                  "var": bind, "w": w, "loc": tr.loc()}  # fmt: skip
            tr.emit(ev)
            x = E.var(bind, w)
            if value is None or len(groups) == 1:
                value = x
                continue
            who = None
            for t in ts:
                wv, ln = tr.threads()[t]
                c = E.and1(E.cmp("eq", E.var("%warp", 32), E.const(wv, 32)),
                           E.cmp("eq", E.var("%lane", 32), E.const(ln, 32)))  # fmt: skip
                who = c if who is None else E.bin("or", who, c)
            value = E.select(who, x, value)
        if value is not None:
            tr.write(d[1], value)
    return handled


def hook(tr, i, mn: str, cls: str) -> bool:
    """interp.Translator.execute's entry (policy gsync): True if the instruction is handled
    here (module doc), False to translate it as before."""
    st = tr.k.gsync
    if st.channels and mn in (
        "st",
        "ld",
        "st.shared",
        "ld.shared",
    ):  # a relay channel's access
        if (channel_store if mn.startswith("st") else channel_load)(tr, i):
            return True
    return False
