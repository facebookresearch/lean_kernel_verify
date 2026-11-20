# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Protocol regions: a small atomic protocol on one word that a group of CTAs shares (policy
`protocols`; references/frontend_ptx.md "Protocol regions", kir.md change 18).

    find(fn, cfg, roles, tensors, params) -> [Region]     the protocol regions of an entry
    rewrite(fn, regions) -> parse.Function                each region replaced by its outcome
    outcome(tr, i)                                        interp.py: the synthetic `lkv.outcome`
    agents(region, record, inp) -> [[Agent]] | None       the groups of one launch (Python
                                                          evaluation of the recorded live-ins)
    explore(region, group, start) -> Exploration          every reachable outcome of a group

A team of CTAs that agree through atomics on a word (register, poll, abort on a timeout) decides
by the interleaving of its atomics and by timing which CTAs go on how: no single evaluation of
each CTA can stand for that. Where the case names the protocol's tensor (`protocols`), its
accesses must lie in one region of one role: the blocks from the first access to the last
(and on through the pure blocks that only decide from what the region computed where to go
on), entered only at that first access. The region is cut out of the function and replaced by
the synthetic `lkv.outcome` instruction, which reads the CTA's outcome of the protocol (which
exit it leaves by, and the registers it leaves live) from a frontend-made metadata table
`__proto<k>`, and branches to that exit. The table is input data, like the CLC schedule tables:
the family's assembly makes one KIR input per reachable outcome of the launch's teams.

The outcomes come from an exhaustive exploration (`explore`). The CTAs whose word is the same
(evaluated per CTA from the region's live-in values, which the translation records at the
outcome instruction) form a group; each runs the region's code as an automaton of one warp (32
lanes, exact integer semantics, the role's warp): local steps are deterministic, the visible
steps are its accesses of the word. Every interleaving of the group's visible steps is explored.
That is complete when every access of the word is morally strong with every other one (atomics,
.relaxed/.acquire/.release/.volatile loads and stores, scope gpu or sys, by CTAs of one GPU):
the PTX memory model's SC-per-location axiom makes the accesses of one location an interleaving
consistent with each thread's program order, reads returning the latest write, and aligned
32/64-bit accesses are single-copy atomic. Timing (nanosleep, poll counts, timeouts) is not
modeled at all: the scheduler may delay any CTA by any amount, so a timeout may fire at any
poll, which the interleavings cover. Trusted: the lanes of one warp that execute one strong
load of one address in one instruction read one value (one memory transaction; checked: the
same address in every active lane). Checked: the word is touched only by the region (no other
access of the tensor in the kernel; KIR v1 tensors do not alias), the region holds only integer
computation, warp shuffles, nanosleep and the word's accesses, its branches are uniform over the
warp, an atomic is executed by one lane, a store by lanes that store one value, the region is not
inside a loop and runs at most once per CTA, under a condition the assembly can evaluate.
Anything else refuses the variant (or, at assembly, the input) with the reason.

The exploration's states are the word's value and each agent's position and live registers
(dead registers dropped; agents with the same code and start are interchangeable, so a state is
taken up to their permutation); it stops at `MAX_STATES` (refused). An outcome is a terminal
state (every agent left the region): per agent its exit and live-out values. A reachable state
in which no agent can change anything but each repeats a poll forever is a hang of the protocol
itself (`Exploration.stuck`); a cycle through other states is refused (its fair termination is
not decided).
"""

import itertools
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from . import ops, parse

if TYPE_CHECKING:  # interp.py imports this module
    from .interp import Translator

MAX_STATES = 3_000_000  # explored states per group before the exploration is refused
MAX_LOCAL = 100_000  # instructions of one agent between two accesses of the word
OUTCOME = "lkv.outcome"  # the synthetic instruction (ops.py row)


class ProtocolError(Exception):
    """A protocol the frontend cannot explore exactly: refuse."""


# ------------------------------------------------------------------ the region


@dataclass
class Region:
    k: int  # the protocol's index: its table is __proto<k>
    tensor: str  # the KIR tensor of the word
    role: str  # the role whose code holds it
    entry: str  # the block of its first access
    start: int  # index of the first access in that block
    blocks: dict  # block -> [Instr] (the entry block from `start`)
    succs: dict  # block -> successors (exits included)
    exits: list  # exit target blocks, in order: exit j continues at exits[j]
    live_in: list  # registers the region reads before writing them
    live_out: list  # registers it writes that are live at an exit
    addr: tuple  # (register, byte offset) of the word's address
    width: int  # the word's bits (32 or 64)
    lines: tuple  # (first, last) PTX line of the region
    loc: str = ""  # the source location of its first access
    exit_locs: list = field(default_factory=list)  # first loc of each exit target
    exit_first: list = field(
        default_factory=list
    )  # per exit: its first Instr if unlabeled


_SPACES = {"shared", "local", "param", "const"}


def _space(i) -> str | None:
    """The state space an ld/st/atom/red names (`.param::entry` is .param), else None."""
    for m in i.opcode.split(".")[1:]:
        if m.split("::")[0] in _SPACES:
            return m.split("::")[0]
    return None


def _addr_op(i):
    """The address operand of a memory instruction, or None."""
    for op in i.operands:
        if op[0] == "mem":
            return op
    return None


FROM_MEMORY = "<memory>"  # a 64-bit value loaded from memory: a pointer to anywhere


class _Provenance:
    """The kernel pointer parameters a register's value may derive from: back through every
    definition of the register (mov, cvta, add, sub, bitwise ops, the addend of mad, selp,
    ld.param of a pointer parameter), FROM_MEMORY for a 64-bit value loaded from memory (a
    pointer read from memory may point anywhere); products, shifts and predicates of a
    pointer are no address in its storage. A least fixpoint over every register (loop-carried
    ones included). Over-approximate: it finds every access the parameter may reach."""

    def __init__(self, fn):
        defs: dict = {}
        for i in fn.instructions():
            for r in _dests(i):
                defs.setdefault(r, []).append(i)
        self.memo: dict = {r: frozenset() for r in defs}
        changed = True
        while changed:
            changed = False
            for r, ds in defs.items():
                out = frozenset().union(*(self._of_def(d) for d in ds))
                if out != self.memo[r]:
                    self.memo[r], changed = out, True

    def _of_def(self, d) -> frozenset:
        if d.base == "ld" and _space(d) == "param":
            a = _addr_op(d)
            return (
                frozenset({a[1][1]})
                if a is not None and a[1][0] == "sym"
                else frozenset()
            )
        if d.base in ("ld", "atom", "red"):
            return frozenset({FROM_MEMORY}) if d.opcode.endswith("64") else frozenset()
        if d.base in ("mul", "shl", "shr", "setp"):
            return frozenset()
        out: set = set()
        for op in d.operands[1:]:
            if op[0] == "reg":
                out |= self.memo.get(op[1], frozenset())
            elif op[0] == "sym":
                out.add(op[1])
        return frozenset(out)

    def params(self, reg: str) -> frozenset:
        return self.memo.get(reg, frozenset())

    def of(self, i) -> frozenset:
        a = _addr_op(i)
        if a is None:
            return frozenset()
        if a[1][0] == "reg":
            return self.params(a[1][1])
        if a[1][0] == "sym":
            return frozenset({a[1][1]})
        return frozenset()


def _is_access(i, word_sym: set, prov) -> bool:
    """A global (or generic) memory instruction whose address may come from one of the
    protocol tensor's parameters."""
    if i.base not in ("ld", "st", "atom", "red", "multimem", "cp", "prefetch"):
        return False
    if _space(i) is not None:
        return False
    return bool(prov.of(i) & (word_sym | {FROM_MEMORY}))


def _strong(i) -> bool:
    mods = set(i.opcode.split(".")[1:])
    if i.base in ("atom", "red"):
        return not mods & {"cta", "cluster"}  # default scope gpu
    if i.base in ("ld", "st"):
        return bool(
            mods & {"relaxed", "acquire", "release", "volatile"}
        ) and not mods & {
            "cta",
            "cluster",
        }
    return False


def _width(i) -> int:
    t = i.opcode.split(".")[-1]
    m = re.fullmatch(r"[bsu](32|64)", t)
    if not m:
        raise ProtocolError(
            f"line {i.line}: {i.opcode}: a protocol access of type .{t}"
        )
    return int(m[1])


_ALLOWED_SREGS = re.compile(
    r"%(laneid|tid\.[xyz]|ntid\.[xyz]|ctaid\.[xyz]|nctaid\.[xyz])"
)


def _check_instr(i, is_access) -> None:
    """The region holds integer computation, warp shuffles, nanosleep, branches and the word's
    accesses only (module doc)."""
    for op in i.operands:
        if op[0] == "sreg" and not _ALLOWED_SREGS.fullmatch(op[1]):
            raise ProtocolError(
                f"line {i.line}: {i.opcode} reads {op[1]} in a protocol region"
            )
    if is_access(i):
        if i.base not in ("atom", "ld", "st", "red") or not _strong(i):
            raise ProtocolError(
                f"line {i.line}: {i.opcode}: an access of the protocol word that is not an "
                "atomic or a strong (.relaxed/.acquire/.release/.volatile) load or store of scope "
                "gpu or sys (the accesses must be morally strong for the interleaving model)"
            )
        if re.search(r"\.v\d", i.opcode):
            raise ProtocolError(
                f"line {i.line}: {i.opcode}: a vector access of the protocol word"
            )
        _width(i)
        return
    if i.base in ("bra", "nanosleep"):
        return
    if i.base == "shfl":
        if ".idx" not in i.opcode or not i.opcode.startswith("shfl.sync"):
            raise ProtocolError(
                f"line {i.line}: {i.opcode} in a protocol region (only shfl.sync.idx)"
            )
        return
    try:
        cls, _why = ops.classify(i)
    except ops.UnknownInstruction as e:
        raise ProtocolError(str(e)) from None
    if cls != "value" or i.base in (
        "ld",
        "vote",
        "elect",
        "redux",
        "match",
        "activemask",
    ):
        raise ProtocolError(f"line {i.line}: {i.opcode} in a protocol region")
    if any(re.fullmatch(r"(b?f16|f32|f64)(x2)?", t) for t in i.opcode.split(".")[1:]):
        raise ProtocolError(
            f"line {i.line}: {i.opcode}: floating point in a protocol region"
        )


def _dests(i):
    from .regions import dests

    return dests(i)


def _reads(i):
    from .regions import reads_of

    return reads_of(i)


def _liveness(blocks: dict, succs: dict, live_at_exit: dict) -> dict:
    """block -> registers live at its entry, over the region's blocks (exit targets carry
    their live-in sets)."""
    use, defs = {}, {}
    for n, ins in blocks.items():
        u, d = set(), set()
        for i in ins:
            u |= set(_reads(i)) - d
            if i.guard is None:
                d |= set(_dests(i))
        use[n], defs[n] = u, d
    live = {n: set() for n in blocks}
    changed = True
    while changed:
        changed = False
        for n in reversed(list(blocks)):
            out = set()
            for s in succs[n]:
                out |= live[s] if s in blocks else live_at_exit.get(s, set())
            new = use[n] | (out - defs[n])
            if new != live[n]:
                live[n], changed = new, True
    return live


def _role_live(cfg, succs: dict) -> dict:
    """block -> registers live at its entry in one role's control flow."""
    blocks = {n: cfg.blocks[n].instrs for n in succs}
    return _liveness(blocks, succs, {})


def _pure(ins) -> bool:
    """A block that only computes registers and branches (no memory, no event)."""
    for i in ins:
        if i.base == "bra":
            continue
        try:
            cls, _ = ops.classify(i)
        except ops.UnknownInstruction:
            return False
        if cls != "value" or i.base in (
            "ld",
            "shfl",
            "vote",
            "elect",
            "redux",
            "match",
        ):
            return False
        if any(op[0] == "sreg" for op in i.operands):
            return False
    return True


def find(fn, cfg, roles: list, tensors: list, params: list) -> list:  # noqa: C901 (one check per requirement)
    """The protocol region of each tensor of `tensors` (policy protocols; module doc); a
    ProtocolError names the first requirement that fails."""
    if not tensors:
        return []
    syms = {}
    for p in params:
        if p.get("kind") == "ptr" and p.get("name") in tensors:
            syms.setdefault(p["name"], set()).add(p["ptx"])
    missing = sorted(set(tensors) - set(syms))
    if missing:
        raise ProtocolError(
            f"protocol tensors {missing} are not pointer parameters of {fn.name}"
        )
    prov = _Provenance(fn)
    out = []
    for k, name in enumerate(tensors):
        is_access = lambda i, s=syms[name]: _is_access(i, s, prov)  # noqa: E731
        hits = [
            (n, j)
            for n, b in cfg.blocks.items()
            for j, i in enumerate(b.instrs)
            if is_access(i)
        ]
        if not hits:
            raise ProtocolError(
                f"protocol tensor {name}: the kernel does not access it"
            )
        reach = {r.name: _reachable(r) for r in roles}
        owners = [r for r in roles if any(n in reach[r.name] for n, _ in hits)]
        if len(owners) != 1 or any(n not in reach[owners[0].name] for n, _ in hits):
            raise ProtocolError(
                f"protocol tensor {name}: its accesses are not all in one role "
                f"({sorted({r.name for r in owners})})"
            )
        role = owners[0]
        out.append(_region(k, name, fn, cfg, role, hits, is_access))
    return out


def _reachable(role) -> set:
    """The blocks a role's own control flow reaches from its entry (cfg.Role.succs)."""
    seen, todo = {role.entry}, [role.entry]
    while todo:
        for s in role.succs.get(todo.pop(), ()):
            if s is not None and s in role.succs and s not in seen:
                seen.add(s)
                todo.append(s)
    return seen


def _rpo(entry: str, succs: dict) -> list:
    order, seen, stack = [], {entry}, [(entry, iter(succs[entry]))]
    while stack:
        b, it = stack[-1]
        for s in it:
            if s is not None and s not in seen and s in succs:
                seen.add(s)
                stack.append((s, iter(succs[s])))
                break
        else:
            stack.pop()
            order.append(b)
    return order[::-1]


def _region(k, name, fn, cfg, role, hits, is_access):  # noqa: C901 (one check per requirement)
    reach = _reachable(role)
    succs = {b: tuple(s for s in role.succs[b]) for b in role.succs if b in reach}
    preds: dict = {b: [] for b in succs}
    for b, ss in succs.items():
        for s in ss:
            if s in preds:
                preds[s].append(b)
    order = _rpo(role.entry, succs)
    index = {b: i for i, b in enumerate(order)}
    hit_blocks = {n for n, _ in hits}
    if any(b not in index for b in hit_blocks):
        raise ProtocolError(
            f"protocol tensor {name}: an access in code its role does not reach"
        )
    entry = min(hit_blocks, key=lambda b: index[b])
    start = min(j for n, j in hits if n == entry)
    ins0 = cfg.blocks[entry].instrs
    while (
        start > 0
        and ins0[start].asm is not None
        and ins0[start - 1].asm == ins0[start].asm
    ):
        start -= (
            1  # the inline asm block of the first access belongs to the region whole
        )
    # blocks from which an access is reachable, reachable from the entry
    reach_from_entry, todo = {entry}, [entry]
    while todo:
        for s in succs[todo.pop()]:
            if s in succs and s not in reach_from_entry:
                reach_from_entry.add(s)
                todo.append(s)
    to_hit, todo = set(hit_blocks), list(hit_blocks)
    while todo:
        for p in preds[todo.pop()]:
            if p not in to_hit:
                to_hit.add(p)
                todo.append(p)
    region = reach_from_entry & to_hit
    if not hit_blocks <= region:
        raise ProtocolError(
            f"protocol tensor {name}: an access its first access does not reach"
        )
    role_live = _role_live(cfg, succs)

    def part(b):
        ins = cfg.blocks[b].instrs
        return ins[start:] if b == entry else ins

    defined = {r for b in region for i in part(b) for r in _dests(i)}
    # on through the pure blocks entered only from the region at which a register it
    # computed is live (the decision where to go on)
    grew = True
    while grew:
        grew = False
        for b in list(region):
            for s in succs[b]:
                if s in region or s not in succs:
                    continue
                if not all(p in region for p in preds[s]) or s == entry:
                    continue
                if not (role_live[s] & defined) or not _pure(cfg.blocks[s].instrs):
                    continue
                region.add(s)
                defined |= {r for i in cfg.blocks[s].instrs for r in _dests(i)}
                grew = True
    for b in region:
        if b == entry:
            if any(p in region for p in preds[b]):
                raise ProtocolError(
                    f"protocol tensor {name}: the region's loop goes back to its first block "
                    f"{cfg.where(b)}"
                )
            continue
        outside = [p for p in preds[b] if p not in region]
        if outside:
            raise ProtocolError(
                f"protocol tensor {name}: {cfg.where(b)} is entered from {cfg.where(outside[0])}, "
                "outside the protocol region (one entry: the first access)"
            )
    blocks = {}
    for b in sorted(region, key=lambda x: index[x]):
        ins = cfg.blocks[b].instrs
        blocks[b] = list(ins[start:] if b == entry else ins)
    for ins in blocks.values():
        for i in ins:
            _check_instr(i, is_access)

    def landing(s):
        """An exit target that only jumps on (`bra.uni X`) continues at X."""
        seen = set()
        while s not in blocks and s not in seen and s in succs and len(succs[s]) == 1:
            ins = cfg.blocks[s].instrs
            if len(ins) != 1 or ins[0].base != "bra" or ins[0].guard is not None:
                break
            seen.add(s)
            s = succs[s][0]
        return s

    rsuccs = {
        b: tuple(s if s in blocks else landing(s) for s in succs[b]) for b in blocks
    }
    exits = []
    for b in blocks:
        for s in rsuccs[b]:
            if s not in blocks and s not in exits:
                exits.append(s)
    if not exits:
        raise ProtocolError(f"protocol tensor {name}: the region never ends")
    if any("~trap" in b for b in list(blocks) + exits):
        raise ProtocolError(
            f"protocol tensor {name}: a trap block in or after the region"
        )
    if any(not cfg.blocks[s].instrs for s in exits):
        raise ProtocolError(f"protocol tensor {name}: an empty exit block")
    defined = {r for ins in blocks.values() for i in ins for r in _dests(i)}
    live_out = sorted(set().union(*(role_live.get(s, set()) for s in exits)) & defined)
    for reg in live_out:  # the outcome table holds one 32-bit integer per live-out
        t = fn.regs.get(reg, "b32")
        if t[0] not in "bsu" or t[1:] != "32":
            raise ProtocolError(
                f"protocol tensor {name}: live-out register {reg} is .{t}, not a 32-bit integer"
            )
    # the automaton's live-ins: what the region reads before writing it, and the live-outs
    # some path leaves as they were (registers it never writes pass through untouched)
    live = _liveness(blocks, rsuccs, {s: set(live_out) for s in exits})
    accesses = [i for ins in blocks.values() for i in ins if is_access(i)]
    addrs = set()
    for i in accesses:
        op = i.operands[1] if i.base in ("atom", "ld") else i.operands[0]
        if op[0] != "mem" or op[1][0] != "reg" or op[3] is not None:
            raise ProtocolError(
                f"line {i.line}: {i.opcode}: an address the protocol cannot name"
            )
        addrs.add((op[1][1], op[2]))
    if len(addrs) != 1:
        raise ProtocolError(
            f"protocol tensor {name}: accesses at {len(addrs)} addresses ({sorted(addrs)}): one word only"
        )
    addr = addrs.pop()
    if addr[0] in defined:
        raise ProtocolError(
            f"protocol tensor {name}: the region changes its address register {addr[0]}"
        )
    widths = {_width(i) for i in accesses}
    if len(widths) != 1:
        raise ProtocolError(
            f"protocol tensor {name}: accesses of {sorted(widths)} bits"
        )
    lines = [i.line for ins in blocks.values() for i in ins]
    first = blocks[entry][0]
    return Region(k, name, role.name, entry, start, blocks, rsuccs, exits, sorted(live[entry]),
                  live_out, addr, widths.pop(), (min(lines), max(lines)), _loc_of(first),
                  [_loc_of(cfg.blocks[s].instrs[0]) for s in exits],
                  [cfg.blocks[s].instrs[0] if s.startswith("@") else None for s in exits])  # fmt: skip


def _loc_of(i) -> str:
    """`file:line` of i's innermost frame, then `<-line` per call site it is inlined at."""
    if i.loc is None:
        return f"PTX line {i.line}"
    frames = list(i.loc.chain())
    f = frames[0]
    return f"{f.file.rsplit('/', 1)[-1]}:{f.line}" + "".join(
        f"<-{x.line}" for x in frames[1:]
    )


def rewrite(fn, regions: list):
    """fn with each region cut out: at its first access, `lkv.outcome` reads the CTA's exit
    into a fresh register (and each live-out register its value), then branches to the exit's
    block (labeled if it had no label). The region's other blocks and their labels go."""
    if not regions:
        return fn
    drop: set = set()
    first_ids = {}
    labels: dict = {}  # exit target -> its label
    new_labels: dict = {}  # id(first instruction of an unlabeled exit target) -> label
    dead_labels: set = set()
    for r in regions:
        for b, ins in r.blocks.items():
            drop.update(id(i) for i in ins)
            if b != r.entry and not b.startswith("@"):
                dead_labels.add(b)
        first_ids[id(r.blocks[r.entry][0])] = r
        for j, (s, x) in enumerate(zip(r.exits, r.exit_first)):
            if s in labels:
                continue
            if x is None:
                labels[s] = s
            else:
                labels[s] = f"$__lkv_proto{r.k}_exit{j}"
                new_labels[id(x)] = labels[s]
    body = []
    for x in fn.body:
        if isinstance(x, parse.Label):
            if x.name not in dead_labels:
                body.append(x)
            continue
        if id(x) in new_labels:
            body.append(parse.Label(new_labels[id(x)], x.line))
        if id(x) in first_ids:
            body += _outcome_code(first_ids[id(x)], x, labels)
        if id(x) in drop:
            continue
        body.append(x)
    out = replace(fn, body=body, regs=dict(fn.regs))
    for r in regions:
        out.regs[f"%__po{r.k}"] = "b32"
        out.regs[f"%__pp{r.k}"] = "pred"
    return out


def _outcome_code(r: Region, at, labels: dict) -> list:
    def mk(opcode, operands, guard=None):
        return parse.Instr(
            opcode,
            tuple(operands),
            guard,
            at.loc,
            at.line,
            None,
            f"{opcode} (protocol {r.k})",
        )

    dst = ("reg", f"%__po{r.k}")
    live_in = tuple(("reg", x) for x in r.live_in)
    code = [mk(f"{OUTCOME}.u32", (dst, ("imm", r.k), ("imm", 0)) + live_in)]
    for j, reg in enumerate(r.live_out, 1):
        code.append(mk(f"{OUTCOME}.u32", (("reg", reg), ("imm", r.k), ("imm", j))))
    p = ("reg", f"%__pp{r.k}")
    for j, s in enumerate(r.exits[:-1]):
        code.append(mk("setp.eq.u32", (p, dst, ("imm", j))))
        code.append(mk("bra", (("sym", labels.get(s, s)),), (p[1], False)))
    code.append(mk("bra.uni", (("sym", labels.get(r.exits[-1], r.exits[-1])),)))
    return code


# ------------------------------------------------------------------ the translator side


@dataclass
class Record:
    """What the translation of a region's role recorded at its outcome instruction."""

    region: Region
    live_in: dict = field(
        default_factory=dict
    )  # register -> internal expression (closed)
    conds: list = field(default_factory=list)  # per occurrence: [internal conditions]
    word: tuple | None = None  # (tensor, element offset expression)
    w0: int = 0  # the role's first warp


def table(k: int) -> str:
    return f"__proto{k}"


def outcome(tr: "Translator", i) -> None:
    """interp.py: `lkv.outcome %r, k, j[, live-ins]`: %r = column j of the CTA's row of the
    outcome table __proto<k> (j = 0: the exit; j > 0: live-out j's value). At j = 0 the
    translation records the region's live-in values, the conditions under which it runs and
    the word's footprint (a strong `gatomic` of its element, issued by lane 0: P0-1's bounds,
    P1-3's pairs)."""
    from . import expr as E, interp  # interp.py imports this module lazily
    from .mem import peel

    k, j = i.operands[1][1], i.operands[2][1]
    regions = getattr(tr.k, "protocols", None) or {}
    r = regions.get(k)
    if r is None:
        tr.fail(i, f"{OUTCOME} of protocol {k}, which no region declares", "error")
    rec = tr.k.protocol_records.setdefault(k, Record(r, w0=tr.w0))
    if tr.loops:
        tr.fail(
            i,
            f"the protocol region of {r.tensor} is inside a loop (it must run at most once per CTA)",
        )
    if not E.is_const(tr.mask, 1):
        tr.fail(
            i,
            f"the protocol region of {r.tensor} runs under a thread-varying condition",
        )
    if tr.nw != 1:
        tr.fail(
            i,
            f"the protocol region of {r.tensor} is in a role of {tr.nw} warps (one warp only)",
        )
    if j == 0:
        for op in i.operands[3:]:
            v = tr.env.get(op[1])
            if v is None or isinstance(v, interp.Data):
                tr.fail(
                    i,
                    f"the protocol region of {r.tensor} reads {op[1]}, a data value ({v})",
                )
            if not _closed(v):
                tr.fail(i, f"the protocol region of {r.tensor} reads {op[1]}, a value of the role's "
                           "own variables (the assembly evaluates live-ins per CTA)")  # fmt: skip
            old = rec.live_in.get(op[1])
            if old is not None and old is not v:
                tr.fail(
                    i,
                    f"the protocol region of {r.tensor} is reached with two values of {op[1]}",
                )
            rec.live_in[op[1]] = v
        conds = list(tr.uconds)
        if not all(_closed(c) for c in conds):
            tr.fail(i, f"the protocol region of {r.tensor} runs under a condition over the role's "
                       "own variables")  # fmt: skip
        rec.conds.append(conds)
        a = rec.live_in.get(r.addr[0])
        p = peel(a) if a is not None else None
        if p is None or p[0][2] != f"%g:{r.tensor}":
            tr.fail(
                i,
                f"the protocol word's address {r.addr[0]} is not {r.tensor} plus an offset",
            )
        off = E.bin("add", p[1], E.const(r.addr[1], E.width(p[1])))
        eb = r.width // 8
        q = E.div_exact(off, eb, tr.slot_facts)
        if q is None:
            tr.fail(
                i,
                f"the protocol word's offset into {r.tensor} is not a multiple of {eb} bytes",
            )
        rec.word = (r.tensor, E.resize(q, 64))
        t = tr.k.mem.tensors.get(r.tensor)
        if t is None or t["elem_bytes"] != eb:
            tr.fail(i, f"protocol tensor {r.tensor}: {eb}-byte words of a tensor of "
                       f"{t and t['elem_bytes']}-byte elements")  # fmt: skip
        t["uses"].add("gatomic")
        saved = tr.mask
        tr.mask = E.cmp("eq", E.LANE, E.const(0, 32))
        tr.mem_event("gatomic", r.tensor, E.resize(q, 64), 1)
        tr.mask = saved
    lin = E.pid(0)  # x-fastest linear CTA index
    lin = E.bin("add", lin, E.bin("mul", E.nprog(0), E.bin("add", E.pid(1),
                E.bin("mul", E.nprog(1), E.pid(2)))))  # fmt: skip
    cols = 1 + len(r.live_out)
    idx = E.bin(
        "add", E.bin("mul", E.resize(lin, 64, False), E.const(cols, 64)), E.const(j, 64)
    )
    name = tr.fresh(f"proto{k}_")
    tr.emit(
        {
            "op": "let",
            "var": name,
            "expr": tr.json(E.load(table(k), idx, 32)),
            "loc": tr.loc(i),
        }
    )
    tr.write(i.operands[0][1], E.var(name, 32))


def _closed(e) -> bool:
    """An internal expression over params, CTA indices, constants and metadata loads only
    (no role variable, no thread symbol but the lane, which the agents substitute)."""
    from . import expr as E

    stack, seen = [e], set()
    while stack:
        x = stack.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        if x[0] == "var":
            n = x[2]
            if n.startswith("%g:") or n in ("%lane", "%warp"):
                continue
            if n.startswith("%") or n.startswith("__"):
                return False
            if not _PARAM.match(n):
                return False
        if x[0] == "table":
            return False
        stack.extend(E.children(x))
    return True


_PARAM = re.compile(r"[A-Za-z_][\w.]*$")


# ------------------------------------------------------------------ evaluation


@dataclass(frozen=True)
class Ptr:
    tensor: str
    off: int  # bytes


class EvalError(Exception):
    pass


def evaluate(e, env: dict):  # noqa: C901 (one case per node)
    """Value of an internal expression (expr.py) for one CTA and lane: env has `params`
    {name: int}, `pid`, `nprog` (3-tuples), `lane`, `warp`, `meta` {tensor: [contents]};
    `%g:<tensor>` is the pointer Ptr(tensor, 0). Exact BitVec semantics, read as signed."""
    from . import expr as E

    memo = env.setdefault("_memo", {})
    key = id(e)
    if key in memo:
        return memo[key]
    k = e[0]
    w = E.width(e)

    def sg(v, width=w):
        v &= (1 << width) - 1
        return v - (1 << width) if width > 1 and v >> (width - 1) else v

    if k == "const":
        r = e[2]
    elif k == "var":
        n = e[2]
        if n.startswith("%g:"):
            r = Ptr(n[3:], 0)
        elif n == "%lane":
            r = env["lane"]
        elif n == "%warp":
            r = env["warp"]
        elif n in env["params"]:
            r = sg(int(env["params"][n]))
        else:
            raise EvalError(f"variable {n}")
    elif k == "pid":
        r = env["pid"][e[2]]
    elif k == "nprog":
        r = env["nprog"][e[2]]
    elif k == "load":
        idx = [evaluate(x, env) for x in e[3]]
        data = env["meta"].get(e[2])
        if (
            data is None
            or len(idx) != 1
            or not isinstance(idx[0], int)
            or not 0 <= idx[0] < len(data)
        ):
            raise EvalError(f"load of {e[2]}{idx}")
        r = sg(int(data[idx[0]]))
    elif k == "select":
        r = evaluate(e[3] if evaluate(e[2], env) else e[4], env)
    elif k == "ext":
        a = evaluate(e[3], env)
        if isinstance(a, Ptr):
            raise EvalError("extension of a pointer")
        wa = E.width(e[3])
        r = sg(a if e[2] else a & ((1 << wa) - 1))
    elif k == "trunc":
        a = evaluate(e[2], env)
        if isinstance(a, Ptr):
            raise EvalError("truncation of a pointer")
        r = sg(a)
    elif k == "cmp":
        a, b = evaluate(e[2], env), evaluate(e[3], env)
        o = e[1]
        if isinstance(a, Ptr) or isinstance(b, Ptr):
            if not (
                isinstance(a, Ptr)
                and isinstance(b, Ptr)
                and a.tensor == b.tensor
                and o in ("eq", "ne")
            ):
                raise EvalError("a comparison of pointers")
            r = int((a.off == b.off) == (o == "eq"))
        else:
            wa = E.width(e[2])
            if o[0] == "u":
                a, b = a & ((1 << wa) - 1), b & ((1 << wa) - 1)
            r = int({"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b,
                     "ge": a >= b}[o.lstrip("su")])  # fmt: skip
    elif k == "bin":
        a, b = evaluate(e[3], env), evaluate(e[4], env)
        o = e[2]
        if isinstance(a, Ptr) or isinstance(b, Ptr):
            if o == "add" and isinstance(a, Ptr) and not isinstance(b, Ptr):
                r = Ptr(a.tensor, a.off + b)
            elif o == "add" and isinstance(b, Ptr) and not isinstance(a, Ptr):
                r = Ptr(b.tensor, b.off + a)
            elif o == "sub" and isinstance(a, Ptr) and not isinstance(b, Ptr):
                r = Ptr(a.tensor, a.off - b)
            else:
                raise EvalError(f"{o} of a pointer")
            memo[key] = r
            return r
        ua, ub = a & ((1 << w) - 1), b & ((1 << w) - 1)
        if o in ("sdiv", "srem", "udiv", "urem") and b == 0:
            raise EvalError("division by zero")
        if o in ("sdiv", "srem"):
            q = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)
            r = q if o == "sdiv" else a - b * q
        else:
            r = {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b,
                 "udiv": lambda: ua // ub, "urem": lambda: ua % ub, "and": lambda: a & b,
                 "or": lambda: a | b, "xor": lambda: a ^ b, "smin": lambda: min(a, b),
                 "smax": lambda: max(a, b), "umin": lambda: min(ua, ub), "umax": lambda: max(ua, ub),
                 "shl": lambda: ua << ub if ub < w else 0, "lshr": lambda: ua >> ub if ub < w else 0,
                 "ashr": lambda: a >> min(ub, w - 1)}[o]()  # fmt: skip
    else:
        raise EvalError(f"node {k}")
    if not isinstance(r, Ptr):
        r = sg(r)
    memo[key] = r
    return r


# ------------------------------------------------------------------ agents


@dataclass
class Agent:
    cta: int  # linear CTA index
    pid: tuple
    regs: dict  # live-in register -> per-lane values (tuple of 32) or one value
    word: tuple  # (tensor, element)


def agents(rec: Record, grid: tuple, params: dict, meta: dict) -> list:
    """The groups (lists of Agents, by word) of the CTAs of one launch that run the region:
    the recorded conditions and live-ins evaluated per CTA."""
    gx, gy, gz = grid
    groups: dict = {}
    for ci in range(gx * gy * gz):
        pid = (ci % gx, (ci // gx) % gy, ci // (gx * gy))
        base = {
            "params": params,
            "pid": pid,
            "nprog": grid,
            "meta": meta,
            "lane": 0,
            "warp": rec.w0,
        }
        try:
            runs = any(all(evaluate(c, dict(base)) for c in cs) for cs in rec.conds)
        except EvalError as e:
            raise ProtocolError(
                f"the condition of the protocol region of {rec.region.tensor}: {e}"
            ) from None
        if not runs:
            continue
        regs: dict = {}
        for reg, e in rec.live_in.items():
            vals = []
            for lane in range(32):
                try:
                    vals.append(evaluate(e, {**base, "lane": lane, "_memo": {}}))
                except EvalError as x:
                    raise ProtocolError(
                        f"live-in {reg} of the protocol region of {rec.region.tensor}: {x}"
                    ) from None
            regs[reg] = vals[0] if all(v == vals[0] for v in vals) else tuple(vals)
        if rec.word is None:
            raise ProtocolError(
                f"the protocol region of {rec.region.tensor}: no word was recorded"
            )
        try:
            word = evaluate(rec.word[1], dict(base))
        except EvalError as x:
            raise ProtocolError(
                f"the word of the protocol region of {rec.region.tensor}: {x}"
            ) from None
        a = Agent(ci, pid, regs, (rec.word[0], word))
        groups.setdefault(a.word, []).append(a)
    return list(groups.values())


# ------------------------------------------------------------------ the automaton


def _bits(t: str) -> int:
    if t == "pred":
        return 1
    m = re.fullmatch(r"[usb](8|16|32|64)", t)
    if not m:
        raise ProtocolError(f"type .{t} in a protocol region")
    return int(m[1])


def _sx(v: int, w: int) -> int:
    v &= (1 << w) - 1
    return v - (1 << w) if w > 1 and v >> (w - 1) else v


def _lanes(v):
    return v if isinstance(v, tuple) else (v,) * 32


def _pack(vals):
    first = vals[0]
    return first if all(x == first for x in vals) else tuple(vals)


_CMP = {
    "eq": lambda a, b: a == b, "ne": lambda a, b: a != b, "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b, "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b,
    "lo": lambda a, b: a < b, "ls": lambda a, b: a <= b, "hi": lambda a, b: a > b,
    "hs": lambda a, b: a >= b,
}  # fmt: skip


class _Code:
    """The region's code as a program: positions (block, index), labels, the agent's
    local step function."""

    def __init__(self, r: Region, is_access):
        self.r = r
        self.blocks = r.blocks
        self.is_access = is_access
        self.exits = {s: j for j, s in enumerate(r.exits)}
        self.live = _liveness(r.blocks, r.succs, {s: set(r.live_out) for s in r.exits})
        # registers live right before each access (the states the exploration keeps)
        self.live_at: dict = {}
        for b, ins in r.blocks.items():
            cur = set()
            for s in r.succs[b]:
                cur |= self.live[s] if s in r.blocks else set(r.live_out)
            for k in range(len(ins) - 1, -1, -1):
                i = ins[k]
                if i.guard is None:
                    cur -= set(_dests(i))
                cur |= set(_reads(i))
                if is_access(i):
                    self.live_at[(b, k)] = frozenset(cur)

    def succ_of(self, b: str, taken: bool) -> str:
        ss = self.r.succs[b]
        if len(ss) == 1:  # the role resolved the branch (cfg.Role.succs)
            return ss[0]
        return ss[0] if taken else ss[1]


def _get(regs, op, w, agent, lane=None):
    k = op[0]
    if k == "reg":
        if op[1] not in regs:
            raise ProtocolError(
                f"register {op[1]} read before it is written in a protocol region"
            )
        return regs[op[1]]
    if k == "imm":
        return _sx(op[1], w) if w > 1 else op[1] & 1
    if k == "sreg":
        n = op[1]
        if n == "%laneid":
            return tuple(range(32))
        if n == "%tid.x":
            return tuple(32 * agent["warp"] + ln for ln in range(32))
        m = re.fullmatch(r"%(ctaid|nctaid|ntid|tid)\.([xyz])", n)
        ax = "xyz".index(m[2])
        return {"ctaid": agent["pid"][ax], "nctaid": agent["grid"][ax],
                "ntid": agent["block"][ax], "tid": 0}[m[1]]  # fmt: skip
    if k == "not":
        v = _get(regs, op[1], 1, agent)
        return _pack([1 - x for x in _lanes(v)])
    raise ProtocolError(f"operand {op} in a protocol region")


def _set(regs, op, v, w):
    if op[0] == "sink":
        return
    if op[0] != "reg":
        raise ProtocolError(f"destination {op} in a protocol region")
    if isinstance(v, Ptr):
        regs[op[1]] = v
        return
    regs[op[1]] = _pack([_sx(x, w) if w > 1 else x & 1 for x in _lanes(v)])


def _active(regs, i):
    """Per-lane guard of i (a tuple of 32 bools), or None if unguarded."""
    if i.guard is None:
        return None
    g = _lanes(regs[i.guard[0]])
    return tuple(bool(x) != i.guard[1] for x in g)


def _merge(old, new, act, w):
    if act is None:
        return new
    o, n = _lanes(old if old is not None else 0), _lanes(new)
    return _pack([n[ln] if act[ln] else o[ln] for ln in range(32)])


def _value(regs, i, agent) -> None:  # noqa: C901 (one case per opcode)
    """Execute one integer value instruction for the warp (per lane)."""
    parts = i.opcode.split(".")
    base = parts[0]
    t = parts[-1]
    act = _active(regs, i)
    ops_ = i.operands
    dst = ops_[0]

    def put(v, w):
        old = regs.get(dst[1]) if dst[0] == "reg" else None
        if isinstance(v, Ptr):
            if act is not None:
                raise ProtocolError(
                    f"line {i.line}: a predicated pointer move in a protocol region"
                )
            _set(regs, dst, v, w)
            return
        _set(regs, dst, _merge(old, v, act, w), w)

    if base == "mov":
        w = _bits(t) if t != "pred" else 1
        v = _get(regs, ops_[1], w, agent)
        put(v, w)
        return
    if base == "setp":
        cmp, w = parts[1], _bits(t)
        signed = t.startswith("s")
        a, b = _get(regs, ops_[1], w, agent), _get(regs, ops_[2], w, agent)
        if isinstance(a, Ptr) or isinstance(b, Ptr):
            if not (
                isinstance(a, Ptr)
                and isinstance(b, Ptr)
                and a == b
                or cmp in ("eq", "ne")
            ):
                raise ProtocolError(
                    f"line {i.line}: a pointer comparison in a protocol region"
                )
            r = (a == b) == (cmp == "eq")
            res = [int(r)] * 32
        else:
            la, lb = _lanes(a), _lanes(b)
            f = _CMP.get(cmp)
            if f is None:
                raise ProtocolError(f"line {i.line}: setp.{cmp} in a protocol region")
            m = (1 << w) - 1
            res = [
                int(f(_sx(x, w), _sx(y, w)) if signed else f(x & m, y & m))
                for x, y in zip(la, lb)
            ]
        if len(parts) > 3 and parts[2] in ("and", "or", "xor"):
            c = _lanes(_get(regs, ops_[3], 1, agent))
            res = [
                {"and": x & y, "or": x | y, "xor": x ^ y}[parts[2]]
                for x, y in zip(res, c)
            ]
        if dst[0] == "pair":
            old1, old2 = regs.get(dst[1][1]), regs.get(dst[2][1])
            regs[dst[1][1]] = _merge(old1, _pack(res), act, 1)
            regs[dst[2][1]] = _merge(old2, _pack([1 - x for x in res]), act, 1)
        else:
            put(_pack(res), 1)
        return
    if base == "selp":
        w = _bits(t)
        a, b = (
            _lanes(_get(regs, ops_[1], w, agent)),
            _lanes(_get(regs, ops_[2], w, agent)),
        )
        c = _lanes(_get(regs, ops_[3], 1, agent))
        put(_pack([x if z else y for x, y, z in zip(a, b, c)]), w)
        return
    if base == "not":
        w = _bits(t)
        a = _lanes(_get(regs, ops_[1], w, agent))
        put(_pack([(1 - x) if w == 1 else ~x for x in a]), w)
        return
    if base == "cvt":
        if "sat" in parts:
            raise ProtocolError(
                f"line {i.line}: {i.opcode}: a saturating convert in a protocol region"
            )
        dt, st_ = parts[-2], parts[-1]
        wd, ws = _bits(dt), _bits(st_)
        a = _lanes(_get(regs, ops_[1], ws, agent))
        m = (1 << ws) - 1
        put(_pack([_sx(x, ws) if st_.startswith("s") else x & m for x in a]), wd)
        return
    w = _bits(t)
    m = (1 << w) - 1
    signed = t.startswith("s")
    wide = "wide" in parts
    srcs = [_get(regs, op, w, agent) for op in ops_[1:]]
    if any(isinstance(x, Ptr) for x in srcs):
        a = srcs[0]
        b = srcs[1] if len(srcs) > 1 else None
        if base == "add" and isinstance(a, Ptr) and isinstance(b, int):
            put(Ptr(a.tensor, a.off + b), 64)
            return
        raise ProtocolError(
            f"line {i.line}: {i.opcode} of a pointer in a protocol region"
        )
    L = [_lanes(x) for x in srcs]

    def iv(x):
        return _sx(x, w) if signed else x & m

    out = []
    for ln in range(32):
        a = L[0][ln]
        b = L[1][ln] if len(L) > 1 else 0
        if base == "add":
            r = iv(a) + iv(b)
        elif base == "sub":
            r = iv(a) - iv(b)
        elif base == "mul":
            r = iv(a) * iv(b)
            if "hi" in parts:
                r >>= w
        elif base == "mad":
            c = L[2][ln]
            r = iv(a) * iv(b) + (_sx(c, 2 * w if wide else w) if signed else c)
            if "hi" in parts:
                raise ProtocolError(f"line {i.line}: mad.hi in a protocol region")
        elif base == "shl":
            r = (a & m) << (b & m) if (b & m) < w else 0
        elif base == "shr":
            sh = b & m
            r = iv(a) >> min(sh, w - 1) if signed else ((a & m) >> sh if sh < w else 0)
        elif base in ("and", "or", "xor"):
            r = {"and": a & b, "or": a | b, "xor": a ^ b}[base]
        elif base in ("min", "max"):
            r = (min if base == "min" else max)(iv(a), iv(b))
        elif base == "neg":
            r = -iv(a)
        else:
            raise ProtocolError(f"line {i.line}: {i.opcode} in a protocol region")
        out.append(r)
    put(_pack(out), 2 * w if wide else w)


def _shfl(regs, i, agent) -> None:
    """shfl.sync.idx.b32 d[|p], a, b, c, membermask (ISA: shfl.sync), every lane active."""
    if _active(regs, i) is not None:
        raise ProtocolError(
            f"line {i.line}: a predicated shfl.sync in a protocol region"
        )
    ops_ = i.operands
    a = _lanes(_get(regs, ops_[1], 32, agent))
    b = _lanes(_get(regs, ops_[2], 32, agent))
    c = _lanes(_get(regs, ops_[3], 32, agent))
    mm = _lanes(_get(regs, ops_[4], 32, agent))
    if any(x & 0xFFFFFFFF != 0xFFFFFFFF for x in mm):
        raise ProtocolError(
            f"line {i.line}: shfl.sync with a partial member mask in a protocol region"
        )
    out, valid = [], []
    for ln in range(32):
        bv, cv = b[ln] & 0x1F, c[ln] & 0x1F
        mask = (c[ln] >> 8) & 0x1F
        max_lane = (ln & mask) | (cv & ~mask & 0x1F)
        min_lane = ln & mask
        j = min_lane | (bv & ~mask & 0x1F)
        ok = j <= max_lane
        out.append(a[j if ok else ln])
        valid.append(int(ok))
    d = ops_[0]
    if d[0] == "pair":
        _set(regs, d[1], _pack(out), 32)
        _set(regs, d[2], _pack(valid), 1)
    else:
        _set(regs, d, _pack(out), 32)


class Exploration:
    """Every reachable outcome of one group (module doc)."""

    def __init__(self):
        self.outcomes: set = (
            set()
        )  # tuples over the group's agents of (exit, live-out values)
        self.states = 0
        self.stuck: list = []  # a reachable state in which every agent only repeats a poll
        self.symmetric = False


def explore(r: Region, group: list, start: int, launch: dict) -> Exploration:  # noqa: C901 (one search)
    """Every interleaving of the group's accesses of the word (module doc). `start` is the
    word's value when the launch begins; `launch` gives `grid`, `block` and the role's warp."""
    code = _Code(r, lambda i: _is_proto_access(i, r))
    w = r.width
    m = (1 << w) - 1
    agents_ = []
    for a in group:
        agents_.append({"pid": a.pid, "grid": launch["grid"], "block": launch["block"],
                        "warp": launch["warp"], "regs": dict(a.regs)})  # fmt: skip

    def run_local(ag, regs, b, k):
        """Run from (b, k) to the next access or an exit: ("at", b, k, regs) or ("exit", j, regs)."""
        steps = 0
        while True:
            ins = code.blocks[b]
            if k >= len(ins):
                nxt = code.r.succs[b][0] if len(code.r.succs[b]) == 1 else None
                if nxt is None:
                    raise ProtocolError(
                        f"block {b} falls through to several successors"
                    )
                if nxt in code.exits:
                    return ("exit", code.exits[nxt], regs)
                b, k = nxt, 0
                continue
            i = ins[k]
            steps += 1
            if steps > MAX_LOCAL:
                raise ProtocolError(
                    f"a protocol agent runs {MAX_LOCAL} instructions without an access of the word "
                    f"(line {i.line})"
                )
            if code.is_access(i):
                return ("at", b, k, regs)
            if i.base == "bra":
                act = _active(regs, i)
                if act is None:
                    taken = True
                else:
                    if len(set(act)) != 1:
                        raise ProtocolError(
                            f"line {i.line}: a branch the warp's lanes take differently in a protocol region"
                        )
                    taken = act[0]
                nxt = (
                    code.succ_of(b, taken)
                    if i.guard is not None
                    else code.r.succs[b][0]
                )
                if nxt in code.exits:
                    return ("exit", code.exits[nxt], regs)
                b, k = nxt, 0
                continue
            if i.base == "nanosleep":
                k += 1
                continue
            if i.base == "shfl":
                _shfl(regs, i, ag)
            else:
                _value(regs, i, ag)
            k += 1

    def freeze(ag_idx, res):
        if res[0] == "exit":
            regs = res[2]
            vals = tuple(_frozen(regs.get(x)) for x in r.live_out)
            return ("exit", res[1], vals)
        _tag, b, k, regs = res
        live = code.live_at[(b, k)]
        return (
            "at",
            b,
            k,
            tuple(sorted((x, _frozen(regs[x])) for x in live if x in regs)),
        )

    def thaw(st):
        return dict(st[3])

    def access(ag, st, word):
        """The agent's access at its position: (new word, next result)."""
        _tag, b, k, _ = st
        regs = thaw(st)
        i = code.blocks[b][k]
        act = _active(regs, i)
        lanes = [ln for ln in range(32) if act is None or act[ln]]
        mn = i.opcode.split(".")
        if not lanes:
            return word, run_local(ag, regs, b, k + 1)
        op = i.operands[1] if i.base in ("atom", "ld") else i.operands[0]
        base = regs.get(op[1][1])
        if not isinstance(base, Ptr):
            raise ProtocolError(
                f"line {i.line}: the protocol word's address is not a pointer"
            )
        if i.base == "ld":
            v = _sx(word, w)
            dst = i.operands[0]
            old = regs.get(dst[1])
            _set(regs, dst, _merge(old, v, act, w), w)
            return word, run_local(ag, regs, b, k + 1)
        if i.base == "st":
            vals = {_lanes(_get(regs, i.operands[1], w, ag))[ln] & m for ln in lanes}
            if len(vals) != 1:
                raise ProtocolError(
                    f"line {i.line}: lanes store different values to the protocol word"
                )
            return vals.pop(), run_local(ag, regs, b, k + 1)
        if len(lanes) != 1:
            raise ProtocolError(
                f"line {i.line}: {len(lanes)} lanes execute an atomic of the protocol word"
            )
        ln = lanes[0]
        srcs = [
            (_lanes(_get(regs, x, w, ag))[ln]) & m
            for x in i.operands[(2 if i.base == "atom" else 1) :]
        ]
        old = word
        opn = next((x for x in mn[1:] if x in ("add", "cas", "exch", "and", "or", "xor", "min", "max",
                                                   "inc", "dec")), None)  # fmt: skip
        signed = mn[-1].startswith("s")
        if opn == "add":
            new = (old + srcs[0]) & m
        elif opn == "cas":
            new = srcs[1] if old == srcs[0] else old
        elif opn == "exch":
            new = srcs[0]
        elif opn in ("and", "or", "xor"):
            new = {"and": old & srcs[0], "or": old | srcs[0], "xor": old ^ srcs[0]}[opn]
        elif opn in ("min", "max"):
            x, y = (_sx(old, w), _sx(srcs[0], w)) if signed else (old, srcs[0])
            new = ((min if opn == "min" else max)(x, y)) & m
        elif opn == "inc":
            new = 0 if old >= srcs[0] else old + 1
        elif opn == "dec":
            new = srcs[0] if old == 0 or old > srcs[0] else old - 1
        else:
            raise ProtocolError(f"line {i.line}: {i.opcode} in a protocol region")
        if i.base == "atom":
            dst = i.operands[0]
            prev = _lanes(regs.get(dst[1], 0)) if dst[0] == "reg" else None
            if dst[0] == "reg":
                vals = list(prev)
                vals[ln] = _sx(old, w)
                regs[dst[1]] = _pack(vals)
        return new, run_local(ag, regs, b, k + 1)

    # symmetry: agents with the same start are interchangeable
    first = []
    for ag in agents_:
        res = run_local(ag, dict(ag["regs"]), r.entry, 0)
        first.append(res)
    starts = [freeze(n, res) for n, res in enumerate(first)]
    reads_pid = any(op[0] == "sreg" and op[1].startswith(("%ctaid", "%tid"))
                    for ins in r.blocks.values() for i in ins for op in i.operands)  # fmt: skip
    classes = {}
    for n, s in enumerate(starts):
        key = (s, None if not reads_pid else n)
        classes.setdefault(key, []).append(n)
    cls_of = [0] * len(agents_)
    for c, (_key, members) in enumerate(classes.items()):
        for n in members:
            cls_of[n] = c
    ex = Exploration()
    ex.symmetric = any(len(v) > 1 for v in classes.values())

    def canon(word, sts):
        if not ex.symmetric:
            return (word, tuple(sts))
        out = list(sts)
        for members in classes.values():
            if len(members) > 1:
                vals = sorted((sts[n] for n in members), key=_skey)
                for n, v in zip(members, vals):
                    out[n] = v
        return (word, tuple(out))

    init = canon(start & m, starts)
    seen = {init: None}
    order = [init]
    edges: dict = {}
    cache: dict = {}
    k = 0
    while k < len(order):
        word, sts = order[k]
        k += 1
        succ = []
        for n, st in enumerate(sts):
            if st[0] != "at":
                continue
            key = (cls_of[n], st, word)
            got = cache.get(key)
            if got is None:
                nw, res = access(agents_[n], st, word)
                got = (nw, freeze(n, res))
                cache[key] = got
            nw, ns = got
            nxt = list(sts)
            nxt[n] = ns
            c = canon(nw, nxt)
            succ.append(c)
            if c not in seen:
                seen[c] = None
                order.append(c)
                if len(order) > MAX_STATES:
                    raise ProtocolError(
                        f"the protocol of {r.tensor} has more than {MAX_STATES} states for "
                        f"{len(sts)} CTAs"
                    )
        state = (word, sts)
        edges[state] = succ
        if not succ:
            ex.outcomes |= _expand(sts, classes, ex.symmetric)
        elif all(s == state for s in succ):
            ex.stuck.append(state)
    ex.states = len(order)
    _check_acyclic(edges, r)
    return ex


def _is_proto_access(i, r: Region) -> bool:
    if i.base not in ("ld", "st", "atom", "red"):
        return False
    op = i.operands[1] if i.base in ("atom", "ld") else i.operands[0]
    return op[0] == "mem" and op[1][0] == "reg" and (op[1][1], op[2]) == r.addr


def _frozen(v):
    return v if not isinstance(v, list) else tuple(v)


def _skey(s):
    return (hash(s), repr(s))


def _expand(sts, classes, symmetric) -> set:
    """The outcome vectors of a terminal state: per agent (exit, live-outs); with symmetric
    agents every permutation within a class (they are interchangeable)."""
    vec = tuple((s[1], s[2]) for s in sts)
    if not symmetric:
        return {vec}
    out = {vec}
    for members in classes.values():
        if len(members) < 2:
            continue
        new = set()
        for v in out:
            for perm in set(itertools.permutations([v[n] for n in members])):
                x = list(v)
                for n, y in zip(members, perm):
                    x[n] = y
                new.add(tuple(x))
        out = new
    return out


def _check_acyclic(edges: dict, r: Region) -> None:
    """Refuse a state graph with a cycle other than self-loops (module doc)."""
    color: dict = {}
    for root in edges:
        if root in color:
            continue
        stack = [(root, iter(edges.get(root, ())))]
        color[root] = 1
        while stack:
            s, it = stack[-1]
            for t in it:
                if t == s:
                    continue
                c = color.get(t, 0)
                if c == 1:
                    raise ProtocolError(
                        f"the protocol of {r.tensor} can return to an earlier state (a cycle of "
                        "its states): its fair termination is not decided"
                    )
                if c == 0:
                    color[t] = 1
                    stack.append((t, iter(edges.get(t, ()))))
                    break
            else:
                color[s] = 2
                stack.pop()
