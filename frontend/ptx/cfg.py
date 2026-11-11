# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Control flow of the PTX frontend: basic blocks, dominators, natural loops, the shape
check, and the roles of a kernel (the Triton warp-specialize dispatch recognizer).

    build(fn) -> Cfg                    roles(cfg, family) -> ([Role], Dispatch or None)

build: the basic blocks of a parse.Function. A block starts at a label or after a
branch and ends at `bra` (guarded: successors are the target, then the fall-through),
`brx.idx` (the labels of its `.branchtargets`), `ret`/`exit` (none) or an unguarded
`trap`: the kernel aborts, and the run goes on as if the trap were not reached (ops.py),
at the other successor of the block's one predecessor, a conditional branch (Triton's
`call __assertfail; trap;` blocks after the ret, CuTe's `if x: trap()`); else refused.
Labels of inline-asm scopes (`waitLoop#k`) are blocks like any other, so a wait loop is
a natural loop with one exit. A block without a label is named `@<line of its first
instruction>`. Unreachable blocks are dropped. Dominators: Cooper, Harvey and Kennedy,
"A Simple, Fast Dominance Algorithm". Then the shape check admits only what the real
kernels have: reducible, every natural loop's exit edges leaving to one block, exactly
one `ret`. Anything else, like a guarded ret or control falling off the end, is a
CfgError naming the lines: the variant is refused, not guessed.

roles: a kernel with brx.idx, or (Triton family only) with a loop whose header loads the
state byte as below and branches on comparing it (with at most 3 worker states LLVM
lowers the switch to a compare chain, no jump table), is a Triton warp-specialized
kernel, and its dispatch must match the protocol item by item, else a CfgError names the
failing item; any other kernel is one role, "default". The match is on the CFG and on
reaching definitions, never on adjacency: unrelated instructions may sit anywhere in
between. A branch condition is a `setp.cmp p, a, b` whose only result p is the guard: of
`setp p|q` the guard may be q, the complement, and a matcher reading cmp would get it
wrong.

  entry    wid = shfl.sync.idx(%tid.x >> 5, 0, 31, -1), then a branch on wid < D (D
           default warps) of either polarity; one side reaches the worker switch loop,
           the other (the default) never does
  header   the switch loop header: optional setmaxnreg, barrier.sync 1 (no count: the
           whole CTA), ld.shared.b8 s, [smem + wid + K], then either the out-of-range
           bypass `@(s > N) bra` and a block ending in `brx.idx s` over N+1 targets
           (state k: target k), or a chain of `@(s == v) bra T` (or the negation): the
           header's own branch, then each next block holding only pure instructions and
           such a branch, with the previous link as its only predecessor (state v: the T
           of the first link comparing v; the chain ends at a block that is no link, the
           bypass)
  table    immediate st.shared stores of the default (a vector of immediates too: LLVM
           merges adjacent ones, `st.shared.v2.b32 [..], {imm, imm}`) write the state of
           every worker warp w (byte smem + K + w, D <= w < nwarps) before its first fork
           barrier
  exit     ... and the exit state E, the target holding only `ret`, to every worker
           after its second and before its fourth (last) fork barrier; both rounds run
           on every path of the default, no other st.shared to smem + constant touches
           those bytes, and no worker starts in state E
  forks    the count of fork barriers (barrier.sync 1, no count) never depends on the
           path: none before the header, 1 in it, 2 in each used state's region, 4 in
           the default

Each worker runs the switch loop exactly twice: once into its state, once into E. So
state k's role is the pre-dispatch code, the worker-side code up to the header, one pass
of the header and of the brx.idx block (or the chain links up to the one deciding k)
with their branches resolved to target k, the region from target k to its back edges
into the header, then `ret`. The header pass stays because the pure instructions LLVM
puts in the header, the brx.idx block and the links (anything else there is refused)
define registers the regions read; their second pass, into E, only precedes `ret`. A used state whose region only syncs (fork and cluster
barriers, setmaxnreg, unconditional bra) is padding and has no role. Roles: "default" (warps
0..D-1) and "partition<k>" (the warps in state k), as the TTGIR frontend names them.

Ceiling: the table/exit check sees st.shared whose address folds to smem + constant
through mov/add; a computed-address st.shared into the state bytes would go unseen (the
smem address-slot analysis can close this). atom/red.shared and a generic st/atom/red
of shared memory (cvta.shared) in a kernel with a dispatch are refused (mem.py).
"""

import re
from dataclasses import dataclass, field
from itertools import combinations

from . import ops, parse


class CfgError(Exception):
    """The control flow is outside what the frontend translates: refuse the variant."""


@dataclass
class Block:
    name: str
    instrs: list
    succs: tuple = ()
    preds: list = field(default_factory=list)


@dataclass
class Cfg:
    fn: parse.Function
    blocks: dict  # name -> Block: the reachable blocks in text order, the entry first
    idom: dict  # block -> immediate dominator (the entry's is itself)
    loops: dict  # natural loop header -> its blocks

    @property
    def entry(self):
        return next(iter(self.blocks))

    def dominates(self, a, b):
        while b != a:
            if b == self.entry:
                return False
            b = self.idom[b]
        return True

    def where(self, b):
        instrs = self.blocks[b].instrs
        return f"{b} (line {instrs[0].line})" if instrs else b


@dataclass
class Role:
    name: str  # "default" or "partition<k>"
    # None for a kernel without .reqntid: the block size is not in the PTX.
    warps: tuple | None
    entry: str
    # The role's own control flow: its blocks -> successors, with the dispatch branch,
    # the bypass and brx.idx resolved and the back edges into the switch loop header
    # redirected to the ret block. A block with one successor here is unconditional,
    # whatever its last instruction says.
    succs: dict


@dataclass
class Dispatch:
    default_warps: int  # D: warps 0..D-1 run the default
    warps: dict  # worker warp -> state
    exit_state: int
    padding: tuple  # used states whose region only syncs
    dispatch: str  # the block ending in the wid < D branch
    header: str  # the switch loop header: one pass of it in each worker role


# ---------------------------------------------------------------------------- CFG


def build(fn):
    """The shape-checked CFG of a parsed function; CfgError outside the shape."""
    blocks, alias, cur = {}, {}, None
    for x in fn.body:
        if isinstance(x, parse.Label):
            # Consecutive labels name one block, but an inline-asm scope's label (`L#k`)
            # starts its own: a wait loop right at an outer loop's header stays a loop
            # of its own (regions.py's Wait), not part of the outer one.
            if cur is not None and not cur.instrs and "#" not in x.name:
                alias[x.name] = cur.name
                continue
            cur = blocks[x.name] = Block(x.name, [])
        else:
            if cur is None or (cur.instrs and _ends(cur.instrs[-1])):
                cur = blocks[f"@{x.line}"] = Block(f"@{x.line}", [])
            cur.instrs.append(x)
    if not blocks:
        raise CfgError(f"{fn.name}: no instructions")
    names = list(blocks)
    succ = {
        n: _succs(fn, blocks, alias, n, names[k + 1] if k + 1 < len(names) else None)
        for k, n in enumerate(names)
    }
    for n in names:
        last = blocks[n].instrs[-1] if blocks[n].instrs else None
        if last is not None and last.base == "trap" and last.guard is None:
            preds = [m for m, s in succ.items() if n in s]
            other = set(succ[preds[0]]) - {n} if len(preds) == 1 else set()
            if len(other) != 1:
                raise CfgError(f"{fn.name}: trap at line {last.line} is not reached by "
                               "one conditional branch")  # fmt: skip
            succ[n] = (other.pop(),)
    rpo = _rpo(fn, names[0], succ)
    reach = set(rpo)
    blocks = {n: blocks[n] for n in names if n in reach}
    for n, b in blocks.items():
        b.succs = succ[n]
        for s in dict.fromkeys(b.succs):
            blocks[s].preds.append(n)
    cfg = Cfg(fn, blocks, _idom(blocks, rpo), {})
    index = {b: i for i, b in enumerate(rpo)}
    for b in rpo:
        for s in dict.fromkeys(blocks[b].succs):
            if index[s] > index[b]:
                continue  # only a retreating edge (in reverse postorder) closes a cycle
            if not cfg.dominates(s, b):
                raise CfgError(
                    f"{fn.name}: irreducible control flow: {cfg.where(b)} -> "
                    f"{cfg.where(s)} enters a cycle other than through its header"
                )
            cfg.loops.setdefault(s, set()).update(_natural_loop(blocks, s, b))
    _check_shape(cfg)
    return cfg


def _ends(i):
    return i.base in ("bra", "brx", "ret", "exit") or (
        i.base == "trap" and i.guard is None
    )


def _succs(fn, blocks, alias, name, fall):
    """Successors of a block; a fall-through past the last block is None."""
    instrs = blocks[name].instrs
    last = instrs[-1] if instrs else None
    if last is None or not _ends(last):
        return (fall,)
    if last.base in ("ret", "exit", "brx") and last.guard is not None:
        raise CfgError(f"{fn.name}: guarded {last.opcode} at line {last.line}")
    if last.base == "brx":
        table = fn.branchtargets.get(last.operands[1][1])
        if table is None:
            raise CfgError(f"{fn.name}: brx.idx at line {last.line}: no .branchtargets")
        return tuple(alias.get(t, t) for t in table)
    if last.base != "bra":
        return ()
    kind, target = last.operands[0][:2]
    target = alias.get(target, target)
    if kind != "sym" or target not in blocks:
        raise CfgError(f"{fn.name}: bra to {target} at line {last.line}: no such label")
    return (target, fall) if last.guard is not None else (target,)


def _rpo(fn, entry, succ):
    """Reverse postorder of the blocks reachable from entry (iterative DFS)."""
    order, seen, stack = [], {entry}, [(entry, iter(succ[entry]))]
    while stack:
        b, it = stack[-1]
        for s in it:
            if s is None:
                raise CfgError(f"{fn.name}: control falls off the end after block {b}")
            if s not in seen:
                seen.add(s)
                stack.append((s, iter(succ[s])))
                break
        else:
            stack.pop()
            order.append(b)
    return order[::-1]


def _idom(blocks, rpo):
    """Cooper-Harvey-Kennedy: iterate idom over reverse postorder to a fixed point."""
    index = {b: i for i, b in enumerate(rpo)}
    idom = {rpo[0]: rpo[0]}

    def meet(a, b):
        while a != b:
            while index[a] > index[b]:
                a = idom[a]
            while index[b] > index[a]:
                b = idom[b]
        return a

    changed = True
    while changed:
        changed = False
        for b in rpo[1:]:
            done = [p for p in blocks[b].preds if p in idom]
            new = done[0]
            for p in done[1:]:
                new = meet(p, new)
            if idom.get(b) != new:
                idom[b], changed = new, True
    return idom


def _natural_loop(blocks, header, latch):
    body, todo = {header, latch}, [latch] if latch != header else []
    while todo:
        for p in blocks[todo.pop()].preds:
            if p not in body:
                body.add(p)
                todo.append(p)
    return body


def _check_shape(cfg):
    name = cfg.fn.name
    for h, body in cfg.loops.items():
        exits = sorted(
            {(u, s) for u in body for s in cfg.blocks[u].succs if s not in body}
        )
        # Several exit edges are fine when they all leave to one block (`while (a && b)`
        # in TLX backward kernels): the region translator joins them with an exit flag.
        if len({s for _u, s in exits}) != 1:
            edges = ", ".join(f"{cfg.where(u)} -> {s}" for u, s in exits) or "none"
            raise CfgError(
                f"{name}: the loop at {cfg.where(h)} has {len(exits)} exit edges "
                f"({edges}); only exits to one block are supported"
            )
    rets = [
        i.line
        for b in cfg.blocks.values()
        for i in b.instrs
        if i.base in ("ret", "exit")
    ]
    if len(rets) != 1:
        raise CfgError(
            f"{name}: {len(rets)} ret/exit (lines {rets}); exactly one is supported"
        )


# ---------------------------------------------------------------------------- roles


def roles(cfg, family="ttgir"):
    """The roles of the kernel and its warp-specialize Dispatch (None without one; a
    compare-chain dispatch is looked for in the Triton family only)."""
    fn = cfg.fn
    brx = [
        b
        for b, blk in cfg.blocks.items()
        if blk.instrs and blk.instrs[-1].base == "brx"
    ]
    if len(brx) > 1:
        lines = [cfg.blocks[b].instrs[-1].line for b in brx]
        raise CfgError(
            f"{fn.name}: brx.idx at lines {lines}; only one dispatch is supported"
        )
    nwarps = _nwarps(fn)
    if brx:
        return _warp_specialized(cfg, *_jump_table(cfg, brx[0]), nwarps)
    chains = [(h, ld) for h in cfg.loops if family == "ttgir"
              for ld in [_state_load(cfg, h)]
              if ld is not None and _link(cfg, h, (h, ld))]  # fmt: skip
    if len(chains) > 1:
        lines = [cfg.blocks[h].instrs[ld].line for h, ld in chains]
        raise CfgError(
            f"{fn.name}: state byte loads at lines {lines}; only one dispatch is supported"
        )
    if chains:
        h, ld = chains[0]
        return _warp_specialized(cfg, h, ld, _compare_chain(cfg, h, ld), nwarps)
    warps = tuple(range(nwarps)) if nwarps else None
    succs = {b: blk.succs for b, blk in cfg.blocks.items()}
    return [Role("default", warps, cfg.entry, succs)], None


def _nwarps(fn):
    n = fn.directives.get("reqntid")
    if n and n[0] % 32 == 0 and all(v == 1 for v in n[1:]):
        return n[0] // 32
    return None


def _no(cfg, item, msg):
    return CfgError(f"{cfg.fn.name}: warp-specialize dispatch ({item}): {msg}")


_LD_B8 = re.compile(r"ld\.shared(?:::cta)?\.[bu]8")


def _warp_specialized(cfg, h, ld, arms, nwarps):
    """arms: {state: (the blocks from the header h to the one deciding it, its target)}."""
    blocks = cfg.blocks
    db, D, wid, default, worker = _entry(cfg, h, nwarps)
    smem, K = _state_base(cfg, h, ld, wid)
    ret = next(
        b
        for b, blk in blocks.items()
        if blk.instrs and blk.instrs[-1].base in ("ret", "exit")
    )
    pre, side, dflt = _parts(cfg, h, db, default, worker, ret)
    warps, E = _state_map(cfg, smem, K, D, nwarps, arms, default, dflt, ret)
    first = {b: blocks[b].succs for b in dflt}
    out = [
        Role("default", tuple(range(D)), cfg.entry, _flow(cfg, pre, db, default, first))
    ]
    padding = []
    for k in sorted(set(warps.values())):
        path, t = arms[k]
        region = _region(cfg, k, t, h)
        if all(_syncs_only(i) for u in region for i in blocks[u].instrs):
            padding.append(k)
            continue
        rest = {u: blocks[u].succs for u in side}
        rest.update(zip(path, [(u,) for u in path[1:] + (t,)]))
        rest[ret] = ()
        rest.update({u: _to(blocks[u].succs, h, ret) for u in region})
        mine = tuple(w for w, s in warps.items() if s == k)
        succs = _flow(cfg, pre, db, worker, rest)
        out.append(Role(f"partition{k}", mine, cfg.entry, succs))
    return out, Dispatch(D, warps, E, tuple(padding), db, h)


def _parts(cfg, h, db, default, worker, ret):
    """{block: fork barriers before it} of the pre-dispatch code, the worker side up to
    the header and the default; they and the switch loop must not overlap."""
    pre = _forks(cfg, cfg.entry, {default, worker}, "before the dispatch")
    side = _forks(cfg, worker, {h}, "on the worker side") if worker != h else {}
    dflt = _forks(cfg, default, set(), "in the default")
    parts = {
        "pre-dispatch": pre,
        "worker-side": side,
        "default": dflt,
        "switch loop": cfg.loops[h],
    }
    for (a, x), (c, y) in combinations(parts.items(), 2):
        if set(x) & set(y):
            shared = cfg.where(min(set(x) & set(y)))
            raise _no(cfg, "entry", f"the {a} and {c} code share {shared}")
    if db not in pre or not (cfg.dominates(db, default) and cfg.dominates(db, worker)):
        raise _no(cfg, "entry", f"{cfg.where(db)} does not dominate both sides")
    if any(_nforks(cfg, b) for b in [*pre, *side]):
        raise _no(cfg, "forks", "a fork barrier before the switch loop header")
    if dflt.get(ret) != 4:
        n = dflt.get(ret, "no")
        raise _no(
            cfg, "forks", f"the default reaches ret after {n} fork barriers, not 4"
        )
    return pre, side, dflt


def _state_map(cfg, smem, K, D, nwarps, arms, default, dflt, ret):
    """({worker warp: state}, exit state) from the default's two rounds of stores."""
    lo, hi = K + D, K + nwarps
    table, exit_ = _state_stores(cfg, smem, lo, hi, default, dflt, ret)
    for item, got in (("table", table), ("exit", exit_)):
        missing = [a - K for a in range(lo, hi) if a not in got]
        if missing:
            raise _no(cfg, item, f"the state byte of warps {missing} is not written")
    E = exit_[lo]
    if (
        set(exit_.values()) != {E}
        or E not in arms
        or arms[E][1] != ret
        or [i.base for i in cfg.blocks[ret].instrs] != ["ret"]
    ):
        raise _no(
            cfg,
            "exit",
            f"the exit stores write states {sorted(set(exit_.values()))},"
            " not the one state whose target is the block holding only ret",
        )
    warps = {w: table[K + w] for w in range(D, nwarps)}
    bad = {w: s for w, s in warps.items() if s not in arms or s == E}
    if bad:
        raise _no(cfg, "table", f"warps start in states {bad}: the exit state or none")
    return warps, E


def _region(cfg, k, t, h):
    """{block: fork barriers before it} of state k: from its target t to the back edges
    into the header h, with 2 fork barriers on every path."""
    region = _forks(cfg, t, {h}, f"in state {k}") if t != h else {}
    if not region or not set(region) <= cfg.loops[h]:
        raise _no(cfg, "forks", f"state {k} ({t}) leaves the switch loop")
    for u in region:
        n = region[u] + _nforks(cfg, u)
        if h in cfg.blocks[u].succs and n != 2:
            raise _no(
                cfg,
                "forks",
                f"state {k} reaches the header from "
                f"{cfg.where(u)} after {n} fork barriers, not 2",
            )
    return region


def _jump_table(cfg, xb):
    """(header, index of the state load, arms) of the switch loop whose brx.idx ends block
    xb. Besides the protocol's instructions, the header and xb hold only pure ones."""
    blocks = cfg.blocks
    brx = blocks[xb].instrs[-1]
    if len(blocks[xb].preds) != 1:
        raise _no(
            cfg,
            "header",
            f"the block of brx.idx at line {brx.line} has "
            f"{len(blocks[xb].preds)} predecessors",
        )
    h = blocks[xb].preds[0]
    instrs = blocks[h].instrs
    bypass = instrs[-1] if instrs else None
    if (
        xb not in cfg.loops.get(h, ())
        or bypass is None
        or bypass.base != "bra"
        or bypass.guard is None
    ):
        raise _no(
            cfg,
            "header",
            f"{cfg.where(h)} is not a switch loop header ending "
            "in the out-of-range bypass `@p bra`",
        )
    ld = _header_load(cfg, h)
    for i in blocks[xb].instrs[:-1]:
        if not _pure(i):
            raise _no(cfg, "header", f"unexpected `{i.text}` at line {i.line}")
    state, n = instrs[ld].operands[0], len(blocks[xb].succs) - 1
    cpos, cmp = _setp(cfg, h)
    if (
        bypass.guard[1]
        or not cmp
        or not cmp.opcode.startswith("setp.gt.")
        or cmp.operands[1:] != (state, ("imm", n))
        or _def(cfg, *cpos, state[1]) != (h, ld)
        or brx.operands[0] != state
        or _def(cfg, xb, len(blocks[xb].instrs) - 1, state[1]) != (h, ld)
    ):
        raise _no(
            cfg,
            "header",
            f"the bypass at line {bypass.line} is not "
            f"`@(state > {n}) bra` before brx.idx of that state",
        )
    return h, ld, {k: ((h, xb), t) for k, t in enumerate(blocks[xb].succs)}


def _header_load(cfg, h):
    """The index of the state load ld.shared.b8 in the switch loop header h, after an
    optional setmaxnreg and the fork barrier; the header's other instructions but its last
    (its branch) must be pure."""
    fork = ld = None
    instrs = cfg.blocks[h].instrs
    for k, i in enumerate(instrs[:-1]):
        if i.guard is None and _is_fork(i) and fork is None:
            fork = k
        elif i.guard is None and i.opcode.startswith("setmaxnreg") and fork is None:
            pass
        elif (
            i.guard is None
            and _LD_B8.fullmatch(i.opcode)
            and fork is not None
            and ld is None
        ):
            ld = k
        elif not _pure(i):
            raise _no(cfg, "header", f"unexpected `{i.text}` at line {i.line}")
    if ld is None:
        raise _no(
            cfg,
            "header",
            f"{cfg.where(h)} lacks barrier.sync 1 followed by "
            "ld.shared.b8 of the state byte",
        )
    return ld


def _state_load(cfg, h):
    """The index of an unguarded ld.shared.b8 after an unguarded fork barrier in block h,
    or None: what marks a compare-chain dispatch's header (roles)."""
    instrs = cfg.blocks[h].instrs
    fork = next(
        (k for k, i in enumerate(instrs) if i.guard is None and _is_fork(i)), None
    )
    if fork is None:
        return None
    return next((k for k, i in enumerate(instrs) if k > fork and i.guard is None
                 and _LD_B8.fullmatch(i.opcode)), None)  # fmt: skip


def _link(cfg, b, ld):
    """(v, target, other) if block b ends in `@p bra` on p = (s == v) or (s != v), s the
    state the instruction at position ld loads: target is b's successor when s == v;
    else None."""
    blk = cfg.blocks[b]
    br = blk.instrs[-1] if blk.instrs else None
    if br is None or br.base != "bra" or br.guard is None or len(blk.succs) != 2:
        return None
    pp, p = _setp(cfg, b)
    parts = p.opcode.split(".") if p else []
    if not (
        p
        and parts[1] in ("eq", "ne")
        and p.operands[1][0] == "reg"
        and p.operands[2][0] == "imm"
        and _def(cfg, *pp, p.operands[1][1]) == ld
    ):
        return None
    taken, fall = blk.succs
    if (parts[1] == "eq") != br.guard[1]:
        return p.operands[2][1], taken, fall
    return p.operands[2][1], fall, taken


def _compare_chain(cfg, h, ld):
    """{state: (links from the header h to the one deciding it, target)} of the compare
    chain the header h starts (module doc, item header); the first link comparing a value
    decides it."""
    _header_load(cfg, h)
    arms, path, b = {}, (h,), h
    while True:
        v, t, b = _link(cfg, b, (h, ld))
        if not 0 <= v < 256:
            raise _no(cfg, "header", f"the chain compares the state byte with {v}")
        arms.setdefault(v, (path, t))
        blk = cfg.blocks[b]
        if b in path or _link(cfg, b, (h, ld)) is None:
            return arms  # the bypass: no role runs it
        bad = [i for i in blk.instrs[:-1] if not _pure(i)]
        if len(blk.preds) != 1 or bad:
            why = f"`{bad[0].text}`" if bad else f"{len(blk.preds)} predecessors"
            raise _no(cfg, "header", f"the chain link {cfg.where(b)} has {why}")
        path += (b,)


def _pure(i):
    """i only computes a register (op table class "value"): harmless to run again."""
    try:
        return ops.classify(i)[0] == "value"
    except ops.UnknownInstruction:
        return False


def _entry(cfg, h, nwarps):
    """(dispatch block, D, position of the wid shfl, default side, worker side): the
    first conditional branch on the dominator path to the header must be on wid < D."""
    blocks = cfg.blocks
    path, b = [], h
    while b != cfg.entry:
        b = cfg.idom[b]
        path.append(b)
    db = next((b for b in reversed(path) if len(blocks[b].succs) == 2), None)
    if db is None:
        raise _no(cfg, "entry", f"no conditional branch dominates {cfg.where(h)}")
    br = blocks[db].instrs[-1]
    pp, p = _setp(cfg, db)
    parts = p.opcode.split(".") if p else []
    if not (
        p
        and parts[1] in ("lt", "ge")
        and p.operands[1][0] == "reg"
        and p.operands[2][0] == "imm"
    ):
        raise _no(cfg, "entry", f"the branch at line {br.line} is not on wid < D")
    wid = _def(cfg, *pp, p.operands[1][1])
    shfl = _instr(cfg, wid)
    if not (
        shfl
        and ops.mnemonic(shfl.opcode) == "shfl.sync.idx"
        and shfl.operands[2:] == (("imm", 0), ("imm", 31), ("imm", -1))
        and _is_warp_id(cfg, wid, shfl.operands[1])
    ):
        raise _no(
            cfg,
            "entry",
            f"the branch at line {br.line} compares "
            f"{p.operands[1][1]}, not shfl.sync.idx(%tid.x >> 5, 0)",
        )
    D = p.operands[2][1]
    if nwarps is None or not 0 < D < nwarps:
        raise _no(
            cfg,
            "entry",
            f"wid < {D} with .reqntid "
            f"{cfg.fn.directives.get('reqntid')}: no 1-D block with warps beyond D",
        )
    taken, fall = blocks[db].succs
    lt_taken = (parts[1] == "lt") != br.guard[1]
    default, worker = (taken, fall) if lt_taken else (fall, taken)
    if h in _reach(cfg, default, {db}) or h not in _reach(cfg, worker, {db}):
        raise _no(
            cfg,
            "entry",
            f"the side of line {br.line} taken when wid < {D} "
            f"({cfg.where(default)}) is not the one apart from the switch loop",
        )
    return db, D, wid, default, worker


def _is_warp_id(cfg, pos, op):
    """op (an operand at position pos) = %tid.x >> 5, possibly through a mov."""
    spos = _def(cfg, *pos, op[1]) if op[0] == "reg" else None
    shr = _instr(cfg, spos)
    if not (
        shr and ops.mnemonic(shr.opcode) == "shr" and shr.operands[2] == ("imm", 5)
    ):
        return False
    t = shr.operands[1]
    if t[0] == "reg":
        mov = _instr(cfg, _def(cfg, *spos, t[1]))
        t = mov.operands[1] if mov and ops.mnemonic(mov.opcode) == "mov" else t
    return t == ("sreg", "%tid.x")


def _state_base(cfg, h, ld, wid):
    """(smem symbol, K) of the state byte load ld.shared.b8 [smem + wid + K]."""
    load = cfg.blocks[h].instrs[ld]
    addr = _lin(cfg, h, ld, load.operands[1], wid)
    syms = [t for t in addr or () if t not in ("", "%wid")]
    if not addr or addr.get("%wid") != 1 or len(syms) != 1 or addr[syms[0]] != 1:
        raise _no(
            cfg,
            "header",
            f"the state byte load at line {load.line} is not [smem + wid + K]",
        )
    return syms[0], addr[""]


def _state_stores(cfg, smem, lo, hi, default, dflt, ret):
    """({byte: state} stored before the default's first fork barrier, {byte: state}
    stored after its second and before its fourth), from the st.shared to smem +
    constant overlapping [lo, hi); CfgError for any other such store."""
    rounds = ({}, {})
    for b, blk in cfg.blocks.items():
        for k, i in enumerate(blk.instrs):
            if ops.mnemonic(i.opcode) != "st.shared":
                continue
            addr = _lin(cfg, b, k, i.operands[0], None)
            if addr is None or set(addr) != {"", smem} or addr[smem] != 1:
                continue  # not smem + constant (see the module's ceiling)
            vec = re.search(r"\.v(\d)\.", i.opcode)
            size = parse.type_bytes(i.opcode.split(".")[-1]) * int(vec[1] if vec else 1)
            span = range(max(addr[""], lo), min(addr[""] + size, hi))
            if not span:
                continue
            n = (
                dflt[b] + sum(_is_fork(x) for x in blk.instrs[:k])
                if b in dflt
                else None
            )
            item = "exit" if n is not None and n >= 2 else "table"
            why = _store_fault(cfg, i, n, b, default, ret)
            if why:
                msg = f"`{i.text}` at line {i.line} writes the state bytes {why}"
                raise _no(cfg, item, msg)
            got = rounds[n > 0]
            vals = _immediates(i.operands[1])  # not None: _store_fault
            k = size // len(vals)
            data = b"".join((v % (1 << 8 * k)).to_bytes(k, "little") for v in vals)
            for a in span:
                if a in got:
                    raise _no(
                        cfg, item, f"state byte {a} written twice (line {i.line})"
                    )
                got[a] = data[a - addr[""]]
    return rounds


def _store_fault(cfg, i, n, b, default, ret):
    """Why store i (in block b, after n fork barriers of the default; None outside the
    default) to the state bytes breaks the protocol, or None."""
    if n is None:
        return "outside the default"
    if n not in (0, 2, 3):
        return f"after {n} fork barriers"
    if i.guard is not None:
        return "under a guard"
    if _immediates(i.operands[1]) is None:
        return "not as an immediate"
    if b != default and ret in _reach(cfg, default, {b}):
        return "not on every path"
    return None


def _immediates(op):
    """The values of an immediate store operand, or of a vector of immediates (LLVM
    merges adjacent state stores into `st.shared.v2.b32 [..], {imm, imm}`), in memory
    order; None if any is not an immediate."""
    xs = op[1] if op[0] == "vec" else (op,)
    return [x[1] for x in xs] if all(x[0] == "imm" for x in xs) else None


def _flow(cfg, pre, db, first, rest):
    """A role's successors: the pre-dispatch blocks, the dispatch branch resolved to
    `first`, then rest (block -> successors); in text order."""
    succs = {b: cfg.blocks[b].succs for b in pre}
    succs[db] = (first,)
    succs.update(rest)
    return {b: succs[b] for b in cfg.blocks if b in succs}


def _to(succs, frm, to):
    return tuple(to if s == frm else s for s in succs)


def _is_fork(i):
    return i.opcode == "barrier.sync" and i.operands == (("imm", 1),)


def _nforks(cfg, b):
    n = 0
    for i in cfg.blocks[b].instrs:
        if _is_fork(i):
            if i.guard is not None:
                raise _no(cfg, "forks", f"guarded fork barrier at line {i.line}")
            n += 1
    return n


def _syncs_only(i):
    """i only synchronizes: a fork barrier, setmaxnreg, an unconditional bra, or a
    cluster barrier (2-CTA kernels end every worker region with barrier.cluster.arrive,
    padding included)."""
    return (
        _is_fork(i)
        or i.opcode.startswith("setmaxnreg")
        or (i.guard is None and i.opcode.startswith("barrier.cluster."))
        or (i.base == "bra" and i.guard is None)
    )


def _reach(cfg, start, stop):
    seen, todo = {start}, [start]
    while todo:
        for s in cfg.blocks[todo.pop()].succs:
            if s not in seen and s not in stop:
                seen.add(s)
                todo.append(s)
    return seen


def _forks(cfg, start, stop, what):
    """{block: fork barriers executed before it} over the blocks reachable from start
    without entering stop; CfgError if a count depends on the path."""
    count, todo = {start: 0}, [start]
    while todo:
        b = todo.pop()
        n = count[b] + _nforks(cfg, b)
        for s in cfg.blocks[b].succs:
            if s in stop:
                continue
            if s not in count:
                count[s] = n
                todo.append(s)
            elif count[s] != n:
                raise _no(
                    cfg,
                    "forks",
                    f"{what}, the fork barriers executed before "
                    f"{cfg.where(s)} depend on the path ({count[s]} or {n})",
                )
    return count


# ---------------------------------------------------------------------------- dataflow

# Opcodes whose first operand is read, not written (but barrier.red's is its result).
_READS_FIRST = ("brx", "bar.", "barrier", "nanosleep", "tcgen05.dealloc")
_BAR_RED = ("bar.red.", "barrier.red.")


def _dests(i):
    """Registers i may write: those of its first operand. Taking a read for a write
    only makes a match fail, never succeed."""
    if not i.operands or (
        i.opcode.startswith(_READS_FIRST) and not i.opcode.startswith(_BAR_RED)
    ):
        return ()
    op = i.operands[0]
    regs = op[1] if op[0] in ("vec", "list") else op[1:] if op[0] == "pair" else (op,)
    return tuple(o[1] for o in regs if o[0] == "reg")


def _def(cfg, b, k, reg):
    """Position (block, index) of the definition of reg that reaches instruction k of
    block b: the latest one before it in b or, up a chain of single predecessors, in
    theirs; else the only definition in the function if its block strictly dominates b.
    None otherwise, or if that definition is guarded (it may not execute)."""
    use, seen = b, set()
    while b not in seen:
        seen.add(b)
        instrs = cfg.blocks[b].instrs
        for j in range(k - 1, -1, -1):
            if reg in _dests(instrs[j]):
                return (b, j) if instrs[j].guard is None else None
        if len(cfg.blocks[b].preds) != 1:
            break
        b = cfg.blocks[b].preds[0]
        k = len(cfg.blocks[b].instrs)
    defs = [
        (c, j)
        for c, blk in cfg.blocks.items()
        for j, x in enumerate(blk.instrs)
        if reg in _dests(x)
    ]
    if len(defs) != 1 or defs[0][0] == use or not cfg.dominates(defs[0][0], use):
        return None
    return defs[0] if _instr(cfg, defs[0]).guard is None else None


def _instr(cfg, pos):
    return cfg.blocks[pos[0]].instrs[pos[1]] if pos else None


def _setp(cfg, b):
    """(position, instruction) of the `setp.cmp p, a, b` defining the guard of the
    branch ending block b (module doc, roles); (None, None) if it is defined otherwise,
    like by a pair `setp p|q` or a `setp.cmp.bop p, a, b, c` (combining with c)."""
    instrs = cfg.blocks[b].instrs
    pos = _def(cfg, b, len(instrs) - 1, instrs[-1].guard[0])
    p = _instr(cfg, pos)
    if not p or p.base != "setp" or p.operands[0][0] != "reg" or len(p.operands) != 3:
        return None, None
    return pos, p


def _lin(cfg, b, k, op, wid, depth=8):
    """Operand op of instruction k of block b as {symbol: coefficient, "%wid":
    coefficient, "": constant} through mov and add, "%wid" being the result of the
    instruction at position wid; None if it is anything else."""
    kind = op[0]
    if kind == "imm":
        return {"": op[1]}
    if kind == "sym":
        return {op[1]: 1, "": 0}
    if kind == "mem":
        base = _lin(cfg, b, k, op[1], wid, depth)
        return base and {**base, "": base[""] + op[2]}
    pos = _def(cfg, b, k, op[1]) if kind == "reg" and depth else None
    if pos is None:
        return None
    if pos == wid:
        return {"%wid": 1, "": 0}
    x = _instr(cfg, pos)
    m = ops.mnemonic(x.opcode)
    if (m, len(x.operands)) not in (("mov", 2), ("add", 3)):
        return None
    out = {"": 0}
    for o in x.operands[1:]:
        t = _lin(cfg, *pos, o, wid, depth - 1)
        if t is None:
            return None
        for key, v in t.items():
            out[key] = out.get(key, 0) + v
    return out
