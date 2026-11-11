# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Region tree of one role of a PTX kernel (design §5.3): what interp.py walks.

    build(cfg, role) -> Tree

The role's own control flow (cfg.Role.succs: the warp-specialize dispatch already
resolved; a block with one successor there is unconditional, whatever its last
instruction says) becomes a tree of

    Block(name, instrs)     straight-line code of a CFG block, its branch dropped
    If(block, branch, then, else_, join)
                            the two successors of `block` up to their immediate post-
                            dominator `join` (CONTINUE: they meet only at the loop's back
                            edge, which follows the If; None: they never meet again inside
                            the enclosing loop iteration or before ret); `then` starts at the
                            branch target, taken when the guard of `branch` (`@p bra` /
                            `@!p bra`) holds
    Loop(header, body, exit, blocks)
                            a natural loop: `body` is one iteration from the header and
                            ends on every path in CONTINUE (a back edge) or BREAK (an edge
                            to `exit`, the one block all exit edges go to, cfg.py's shape
                            rule; several exit edges join there: interp.py's exit flag)
    Wait(instr, header)     the blocking wait pseudo-instruction (design §5.2.2): a loop
                            that is exactly `L: try_wait.parity p, [a], ph(, hint); @!p bra
                            L` or `L: try_wait...; @p bra D; bra L; D:`, its predicate p
                            read nowhere else (frontend/cute/ops_mem.py _wait_loops, on the
                            CFG: try_wait may return false spuriously, so only the whole
                            loop is a blocking wait, ISA mbarrier.test_wait/try_wait). Any
                            other use of try_wait reaches the op table, which refuses it

Structuring: inner loops are collapsed into their headers, so every scope (the role, a
loop body with back edges as CONTINUE and exit edges as BREAK) is a DAG; an If is
followed by its immediate post-dominator in that DAG. A block reached on two paths that
meet only later is emitted on each path, which is exact for acyclic code.

`Tree.live_in[block]`: the registers live on entry of each block of the role (a guarded
definition does not kill); interp.py merges at a join and carries around a loop only
those. `Tree.defs[header]`: the registers a loop's blocks may write.
"""

from dataclasses import dataclass
from types import SimpleNamespace

from . import cfg as C, ops, parse

EXIT, CONTINUE, BREAK = "<exit>", "<continue>", "<break>"
WAIT = "mbarrier.try_wait.parity.shared"


@dataclass
class Block:
    name: str
    instrs: list


@dataclass
class If:
    block: str
    branch: parse.Instr  # `@p bra target`
    then: list
    else_: list
    join: str | None


@dataclass
class Loop:
    header: str
    body: list
    exit: str
    blocks: frozenset


@dataclass
class Wait:
    instr: parse.Instr  # mbarrier.try_wait.parity p, [addr], parity(, hint)
    header: str


@dataclass
class Tree:
    body: list
    live_in: dict  # block -> frozenset of registers
    defs: dict  # loop header -> frozenset of registers its blocks may write


class RegionError(Exception):
    """Control flow the region builder cannot structure: refuse the role."""


def build(cfg, role):
    """The region tree of `role` (a cfg.Role) of the shape-checked `cfg`."""
    g = _Graph(cfg, role)
    body = _Scope(g, None).seq(role.entry, EXIT)
    defs = {
        h: frozenset(r for b in blk for r in g.writes(b)) for h, blk in g.loops.items()
    }
    return Tree(body, _liveness(g), defs)


class _Graph:
    """The role's blocks, successors, dominators and natural loops."""

    def __init__(self, cfg, role):
        self.cfg, self.fn = cfg, cfg.fn
        succ = {b: tuple(dict.fromkeys(s)) for b, s in role.succs.items()}
        self.entry = role.entry
        rpo = C._rpo(self.fn, self.entry, succ)
        self.succ = {b: succ[b] for b in rpo}
        preds = {b: [] for b in rpo}
        for b in rpo:
            for s in self.succ[b]:
                preds[s].append(b)
        nodes = {b: SimpleNamespace(preds=preds[b]) for b in rpo}
        idom = C._idom(nodes, rpo)
        index = {b: i for i, b in enumerate(rpo)}

        def dominates(a, b):
            while b != a:
                if b == self.entry:
                    return False
                b = idom[b]
            return True

        loops = {}
        for b in rpo:
            for s in self.succ[b]:
                if index[s] <= index[b]:
                    if not dominates(s, b):
                        raise RegionError(
                            f"{self.fn.name}: irreducible flow at {cfg.where(b)}"
                        )
                    loops.setdefault(s, set()).update(C._natural_loop(nodes, s, b))
        self.loops = {h: frozenset(b) for h, b in loops.items()}
        self.exit = {}
        for h, blk in self.loops.items():
            outs = {s for u in blk for s in self.succ[u] if s not in blk}
            if len(outs) != 1:
                raise RegionError(
                    f"{self.fn.name}: the loop at {cfg.where(h)} exits to "
                    f"{sorted(outs) or 'nothing'}; only exits to one block are supported"
                )
            self.exit[h] = outs.pop()

    def instrs(self, b):
        return self.cfg.blocks[b].instrs

    def writes(self, b):
        return {r for i in self.instrs(b) for r in dests(i)}


class _Scope:
    """Structuring of one scope: the role (loop None) or one loop's body."""

    def __init__(self, g, loop):
        self.g, self.loop = g, loop
        self.order = self._postorder(loop if loop is not None else g.entry)
        self.ipdom = self._ipdoms()

    def succ(self, n):
        g = self.g
        if n in (CONTINUE, BREAK):
            return (EXIT,)
        if n == EXIT:
            return ()
        if n in g.loops and n != self.loop:  # an inner loop, collapsed into its header
            return (self._map(g.exit[n]),)
        return tuple(dict.fromkeys(self._map(s) for s in g.succ[n])) or (EXIT,)

    def _map(self, s):
        if self.loop is None:
            return s
        if s == self.loop:
            return CONTINUE
        return s if s in self.g.loops[self.loop] else BREAK

    def _postorder(self, entry):
        order, seen, stack = [], {entry}, [(entry, iter(self.succ(entry)))]
        while stack:
            n, it = stack[-1]
            for s in it:
                if s not in seen:
                    seen.add(s)
                    stack.append((s, iter(self.succ(s))))
                    break
            else:
                stack.pop()
                order.append(n)
        return order

    def _ipdoms(self):
        """Immediate post-dominators in the scope DAG: in postorder every node comes
        after its successors, so after its post-dominators (EXIT, the only sink, first)."""
        index = {n: i for i, n in enumerate(self.order)}
        ipdom = {EXIT: EXIT}

        def meet(a, b):
            while a != b:
                if index[a] > index[b]:
                    a = ipdom[a]
                else:
                    b = ipdom[b]
            return a

        for n in self.order:
            if n == EXIT:
                continue
            ss = self.succ(n)
            p = ss[0]
            for s in ss[1:]:
                p = meet(p, s)
            ipdom[n] = p
        return ipdom

    def seq(self, n, stop):
        g, out = self.g, []
        while n not in (stop, EXIT):
            if n in (CONTINUE, BREAK):
                out.append(n)
                break
            if n in g.loops and n != self.loop:
                out.append(self._loop_node(n))
                n = self._map(g.exit[n])
                continue
            instrs = g.instrs(n)
            last = instrs[-1] if instrs else None
            ends = last is not None and last.base in ("bra", "brx", "ret", "exit")
            out.append(Block(n, instrs[:-1] if ends else list(instrs)))
            ss = self.succ(n)
            if len(ss) == 1:
                n = ss[0]
                continue
            if len(ss) != 2 or last is None or last.base != "bra" or last.guard is None:
                raise RegionError(
                    f"{g.fn.name}: {g.cfg.where(n)} has successors {ss} in its role"
                )
            j = self.ipdom[n]
            then, else_ = (self.seq(s, j) for s in ss)
            out.append(If(n, last, then, else_, None if j == EXIT else j))
            n = j
        return out

    def _loop_node(self, h):
        g = self.g
        wait = _wait(g, h)
        if wait is not None:
            return wait
        return Loop(h, _Scope(g, h).seq(h, EXIT), g.exit[h], g.loops[h])


def _wait(g, h):
    """Wait(instr) if the loop at h is a blocking wait loop, else None."""
    blk, instrs = g.loops[h], g.instrs(h)
    if len(instrs) != 2 or ops.mnemonic(instrs[0].opcode) != WAIT:
        return None
    tw, br = instrs
    pred = tw.operands[0]
    if tw.guard is not None or pred[0] != "reg":
        return None
    if br.base != "bra" or br.guard is None or br.guard[0] != pred[1]:
        return None
    target = br.operands[0][1]
    if blk == {h} and br.guard[1] and target == h:  # L: try_wait; @!p bra L
        pass
    elif len(blk) == 2 and not br.guard[1] and target == g.exit[h]:
        (m,) = blk - {h}  # L: try_wait; @p bra D; bra L; D:
        back = g.instrs(m)
        if not (
            len(back) == 1
            and back[0].base == "bra"
            and back[0].guard is None
            and g.succ[m] == (h,)
            and g.succ[h] == (target, m)
        ):
            return None
    else:
        return None
    reads = [i for i in g.fn.instructions() if i is not br and pred[1] in reads_of(i)]
    if reads:
        raise RegionError(
            f"{g.fn.name}: the result of the try_wait at line {tw.line} is read at line "
            f"{reads[0].line} (only the blocking wait loop is supported)"
        )
    return Wait(tw, h)


# --------------------------------------------------------------------- registers


def dests(i):
    """Registers instruction i writes: the registers of its first operand (cfg.py's
    rule: a store, barrier or branch writes none)."""
    return C._dests(i)


def reads_of(i):
    """Registers instruction i reads: every register operand but the ones it writes,
    its guard, and address bases; a guarded instruction also reads what it writes (the
    old value survives where the guard is false)."""
    out = set()
    ds = set(dests(i))

    def visit(op, top):
        kind = op[0]
        if kind == "reg":
            if not top or op[1] not in ds:
                out.add(op[1])
        elif kind in ("vec", "list"):
            for x in op[1]:
                visit(x, top)
        elif kind == "pair":
            visit(op[1], top)
            visit(op[2], top)
        elif kind == "not":
            visit(op[1], False)
        elif kind == "mem":
            visit(op[1], False)
            if op[3] is not None:
                visit(op[3], False)

    for k, op in enumerate(i.operands):
        visit(op, k == 0)
    if i.guard is not None:
        out.add(i.guard[0])
        out |= ds
    return out


def _liveness(g):
    """Registers live on entry of each block (backward dataflow to a fixed point)."""
    use, kill = {}, {}
    for b in g.succ:
        u, k = set(), set()
        for i in g.instrs(b):
            u |= reads_of(i) - k
            if i.guard is None:
                k |= set(dests(i))
        use[b], kill[b] = u, k
    live = {b: set(use[b]) for b in g.succ}
    changed = True
    while changed:
        changed = False
        for b in reversed(list(g.succ)):
            out = set().union(*(live[s] for s in g.succ[b]))
            new = use[b] | (out - kill[b])
            if new != live[b]:
                live[b], changed = new, True
    return {b: frozenset(v) for b, v in live.items()}
