# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Work anchors of the PTX frontend (design §5.9): a case's `work` annotation -> the PTX
instructions that define each work field.

    resolve(work, family, fn, sources, loc_mode) -> (sites, notes)

work      the kernel's `work`: {field: VARNAME} or {field: [{"load": T, "where": TEXT}, ...]}
family    "ttgir" (Triton/TLX cases) or "cute"
fn        the kernel's parse.Function
sources   the export root(s) holding the kernel's Python files
loc_mode  "file_line" (Triton, CuTe DSL 4.6.1) or "line_only" (CuTe DSL 4.4.2, whose
          .file index is unreliable, decision D1): how much of a .loc frame to believe

A static mapping from annotation, sources and PTX to instruction sites. The interpreter
owns everything that needs values: which tensor a load reads and at what index (it drops
candidate loads whose address does not derive from T), and the gating (a work event in
each region that defines every field, after the last definition, as ttgir/to_kir.py
_with_work does).

Site(field, kind, instr, detail): kind "load", "assign" or "novalue"; instr = (function
name, index in fn.instructions(), PTX line). The value: for "load", the element index the
load reads; for "assign", register detail["reg"] right after instr; "novalue" has none (an
assignment that runs but whose value no instruction is known to hold, below): the field
is defined from instr on, which serves the gating, but its value is unknown, so the
interpreter must refuse it for a field the spec reads (a segments `per` field). notes:
findings worth a message that do not stop the kernel (a dropped branch, loads on several
lines). A case error is an AnchorError naming the field and the candidates.

Load anchors (CuTe; also accepted for Triton, decision D2). TEXT matches a source line of a
file under a root, as a substring of the line or of "<relpath>:<line>" (the legacy CuTe
frontend matched an op's statement text and frames, cute/ops_mem.py metadata_value). TEXT
matching no source line is a case error. The candidates are the global loads (ld.global*)
with a .loc frame on a matching line: same file and line in "file_line" mode (frames
outside the export never match), the line alone in "line_only" mode. Every match is a
site, one per (load, anchor entry), as before. A note names an entry without a load and,
in "line_only" mode, the entries whose loads sit on different source lines (per function:
a superset of "a role with matches on different lines"; roles are the interpreter's).
For Triton a field without any site is a case error, as in the TTGIR frontend; for CuTe it
only gets a note, as before.

Assignment anchors (Triton). The kernel is the jit function named like the PTX entry, in
the file of the outermost .loc frames. Each assignment of VARNAME in it is an occurrence:
Assign/AnnAssign (the value) or AugAssign (the whole statement, which is where Triton puts
the operation), to the plain name or to an element of a flat tuple target. A for-loop
target is no occurrence (it is a block argument in TTGIR, which the NameLoc rule never saw
either); any other binding form of VARNAME is refused. An occurrence's region: the
instructions whose outermost .loc frame (the kernel's: inlined helpers are inner frames)
lies in its span, Triton columns being 1-based (ast col_offset + 1), except ld.param (LLVM
schedules parameter loads anywhere, under whatever .loc precedes them). A line-0 .loc is no
location: the instruction keeps the previous one (parse.py). For element j of a tuple target
the region narrows to the instructions computing element j: through tuple displays,
conditional expressions (a constexpr branch has no code), calls of helpers (the next frame
inward is the callee's; element j of each of its returns) and names (their assignments in
their function).

The variable is the result leaving the region: a register a region instruction writes whose
value reaches a read outside the region (followed over the control flow until an
unguarded rewrite). Several writes of one register in one basic block (a default, then a
guarded load into it: Triton's masked load) are one result, at the last write. A result
that flows into another one's instruction inside the region (a subexpression the compiler
shares with code outside, like an index scaled for two loads) gives way to it. Of several
independent results the one at the position of the value's root expression wins: Triton
gives each AST node's op that node's position (in the mxfp8 no-CLC kernel LLVM leaves a
sign extension of the helper's index, used elsewhere too, under the position of the load's
address subexpression). Then exactly one result is the occurrence's site; any other count
is a case error naming the candidates.
An occurrence without instructions (a dropped constexpr branch, or a value the compiler
folded into its uses or computed under another statement's .loc) and a tuple element
without a result (unused there) have no value: if another work-field occurrence in the same
statement list has a site, the statement runs, and it gets a "novalue" site at the last of
those; else a note. Both cases happen in the JFA warp-specialized kernels: LLVM drops the
default partition's unused `off_h`, and in the non-load-balanced path gives `pid =
tile_idx % n_tile_num`'s rem the .loc of `off_h = off_seq_h % H` (whose own rem it sinks
under line 0), so `off_h`'s site holds pid's value. Hence only the fields the spec reads
deserve trust in their values; all `per` fields of the cases are `batch`, which
resolves exactly. A field without any site in the function is a case error. This follows
the TTGIR frontend's gating (a field counted as defined wherever its op survived to TTGIR)
where LLVM dropped the op, not where TTGIR had dropped it already: there the TTGIR
frontend's role has no work item, this one's has the fields that resolve (the dQ-reduce
and MMA partitions of the nvfp4 FULL_TILES variants, whose pid and off_h TTGIR removed,
check their accesses against the batch's segment rather than as "no current work item").
"""

import ast
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import cfg, ops, parse

FAMILIES = ("ttgir", "cute")
LOC_MODES = ("file_line", "line_only")


class AnchorError(Exception):
    """The case's work annotation does not resolve in this kernel: a case error."""


@dataclass(frozen=True)
class Site:
    field: str
    kind: str  # "load", "assign" or "novalue"
    instr: tuple  # (function name, index in Function.instructions(), PTX line)
    detail: dict  # load: tensor, where, frame, source; assign/novalue: var, stmt, element, reg


def resolve(work, family, fn, sources, loc_mode):
    if family not in FAMILIES or loc_mode not in LOC_MODES:
        raise ValueError(f"family {family!r} / loc_mode {loc_mode!r}")
    if not work:
        return [], []
    if isinstance(sources, (str, os.PathLike)):
        sources = [sources]
    roots = tuple(_real(str(s)) for s in sources)
    flow, kern = _Flow(fn), None
    sites, notes = [], []
    for field, anchor in work.items():
        if isinstance(anchor, str) and family == "ttgir":
            if loc_mode != "file_line":
                raise ValueError("assignment anchors need file_line locations")
            kern = kern or _Kernel(flow, roots)
            got = _assign_sites(kern, field, anchor, notes)
        elif isinstance(anchor, list) and anchor and all(map(_is_load, anchor)):
            got = _load_sites(flow, roots, loc_mode, field, anchor, notes)
        else:
            raise AnchorError(
                f"work field {field!r}: {anchor!r} is neither a list of "
                '{"load": tensor, "where": text} nor (Triton only) a variable name'
            )
        sites += got
    if kern:
        sites += _novalue_sites(kern, notes)
    for field, anchor in work.items():
        if family == "ttgir" and not any(s.field == field for s in sites):
            why = [n for n in notes if n.startswith(f"work field {field!r}")]
            raise AnchorError(
                f"work field {field!r}: {anchor!r} resolves to no instruction of {fn.name}"
                + "".join(f"; {n}" for n in why)
            )
    return sites, notes


def _is_load(a):
    return (
        isinstance(a, dict)
        and set(a) == {"load", "where"}
        and all(isinstance(v, str) and v for v in a.values())
    )


# ---------------------------------------------------------------------------- sources


@lru_cache(maxsize=None)
def _real(path):
    return os.path.realpath(path)


def _rel(path, roots):
    """path relative to the root holding it, or None outside the export."""
    for r in roots:
        if path.startswith(r + os.sep):
            return path[len(r) + 1 :]
    return None


@lru_cache(maxsize=None)
def _lines(path):
    return Path(path).read_text(encoding="utf-8", errors="replace").split("\n")


@lru_cache(maxsize=None)
def _tree(path):
    return ast.parse("\n".join(_lines(path)), path)


@lru_cache(maxsize=None)
def _enclosing_def(path, line):
    """The innermost function definition of source file `path` around `line`."""
    out = None
    for n in ast.walk(_tree(path)):
        if isinstance(n, ast.FunctionDef) and n.lineno <= line <= n.end_lineno:
            if out is None or n.lineno > out.lineno:
                out = n
    return out


# ---------------------------------------------------------------------------- dataflow


class _Flow:
    """fn's instructions and what every anchor needs of them: .loc frames (outermost
    first, files resolved), registers written and read, successors, basic blocks."""

    def __init__(self, fn):
        self.fn = fn
        self.instrs = fn.instructions()
        self.frames = [_frames(i) for i in self.instrs]
        self.dests = [cfg._dests(i) for i in self.instrs]
        self.reads = [_reads(i, d) for i, d in zip(self.instrs, self.dests)]
        at, self.block, b = {}, [], 0
        for x in fn.body:
            if isinstance(x, parse.Label):
                at[x.name], b = len(self.block), b + 1
            else:
                self.block.append(b)
                b += x.base in ("bra", "brx", "ret", "exit")
        end = len(self.instrs)
        self.succs = [
            [s for s in _succs(fn, at, k, i) if s < end]
            for k, i in enumerate(self.instrs)
        ]

    def site(self, k):
        return (self.fn.name, k, self.instrs[k].line)

    def describe(self, k):
        return f"line {self.instrs[k].line}: {self.instrs[k].text}"

    def live_out(self, k, reg, region):
        """Whether the value instruction k writes to reg reaches a read outside region."""
        todo, seen = list(self.succs[k]), set()
        while todo:
            n = todo.pop()
            if n in seen:
                continue
            seen.add(n)
            if reg in self.reads[n] and n not in region:
                return True
            if reg not in self.dests[n] or self.instrs[n].guard is not None:
                todo += self.succs[n]
        return False


def _succs(fn, at, k, i):
    """Successor instruction indices of instruction k (at: label -> index of the
    instruction after it)."""
    if i.base in ("ret", "exit"):
        return (k + 1,) if i.guard else ()
    if i.base == "brx":
        return tuple(at[t] for t in fn.branchtargets[i.operands[1][1]])
    if i.base == "bra":
        target = at[i.operands[0][1]]
        return (target, k + 1) if i.guard else (target,)
    return (k + 1,)


def _frames(i):
    if i.loc is None:
        return ()
    return tuple((_real(f.file), f.line, f.col) for f in reversed(tuple(i.loc.chain())))


def _reads(i, dests):
    """Registers instruction i reads: its guard and every register operand but its
    destinations (cfg._dests: the first operand, if it writes)."""
    regs = {r for o in (i.operands[1:] if dests else i.operands) for r in _regs(o)}
    if i.guard:
        regs.add(i.guard[0])
    return regs


def _regs(op):
    kind = op[0]
    if kind == "reg":
        return (op[1],)
    if kind in ("vec", "list"):
        return tuple(r for o in op[1] for r in _regs(o))
    if kind == "pair":
        return _regs(op[1]) + _regs(op[2])
    if kind == "not":
        return _regs(op[1])
    if kind == "mem":
        return _regs(op[1]) + (_regs(op[3]) if op[3] else ())
    return ()


# ---------------------------------------------------------------------------- load anchors


def _load_sites(flow, roots, loc_mode, field, anchors, notes):
    loads = [
        k
        for k, i in enumerate(flow.instrs)
        if ops.mnemonic(i.opcode).startswith("ld.global")
    ]
    out = []
    for a in anchors:
        lines = _matching_lines(a["where"], roots)
        if not lines:
            raise AnchorError(
                f"work field {field!r}: where {a['where']!r} matches no source line "
                f"under {', '.join(roots)}"
            )
        hits = set()
        for k in loads:
            for path, line, _col in flow.frames[k]:
                rel = _rel(path, roots)
                src = [
                    f"{r}:{n}"
                    for r, n in lines
                    if n == line and (loc_mode == "line_only" or r == rel)
                ]
                if src:
                    hits.add(line)
                    detail = {"tensor": a["load"], "where": a["where"],
                              "frame": f"{rel or path}:{line}", "source": src}  # fmt: skip
                    out.append(Site(field, "load", flow.site(k), detail))
                    break
        if not hits:
            notes.append(
                f"work field {field!r}: no global load of {flow.fn.name} on a line "
                f"matching {a['where']!r} ({', '.join(f'{r}:{n}' for r, n in lines)})"
            )
        elif loc_mode == "line_only" and len(hits) > 1:
            notes.append(
                f"work field {field!r}: the loads of {flow.fn.name} matching "
                f"{a['where']!r} sit on different lines {sorted(hits)} (line-only "
                "locations, best effort)"
            )
    return out


@lru_cache(maxsize=None)
def _matching_lines(where, roots):
    """(relpath, line) of every source line under roots matching `where`."""
    out = []
    for root in roots:
        for p in sorted(Path(root).rglob("*.py")):
            rel = str(p.relative_to(root))
            for n, text in enumerate(_lines(_real(str(p))), 1):
                if where in text or where in f"{rel}:{n}":
                    out.append((rel, n))
    return out


# ---------------------------------------------------------------------------- assignment anchors


class _Kernel:
    """The kernel's jit function: its file, definition, and which instructions are
    parameter loads."""

    def __init__(self, flow, roots):
        self.flow, self.roots = flow, roots
        name = flow.fn.name
        first = next((f for f in flow.frames if f), None)
        self.path = first[0][0] if first else None
        self.rel = self.path and _rel(self.path, roots)
        if self.rel is None:
            raise AnchorError(
                f"{name}: its outermost .loc frames ({self.path}) are not in a file "
                f"under {', '.join(roots)}"
            )
        defs = [
            n
            for n in ast.walk(_tree(self.path))
            if isinstance(n, ast.FunctionDef) and n.name == name
        ]
        if len(defs) != 1:
            raise AnchorError(f"{name}: {len(defs)} functions {name} in {self.rel}")
        self.fdef = defs[0]
        self.where = f"{self.rel}:{self.fdef.lineno}"
        self.param = [ops.mnemonic(i.opcode) == "ld.param" for i in flow.instrs]
        self.blocks = _blocks(self.fdef)
        self.live = {}  # statement list -> instruction indices of its assign sites
        self.dead = []  # (field, detail, statement list, note) of occurrences without a value


def _assign_sites(kern, field, var, notes):
    flow, name = kern.flow, kern.flow.fn.name
    occurrences = _bindings(kern.fdef, var)
    if not occurrences:
        raise AnchorError(
            f"work field {field!r}: {var} is not assigned in {name} ({kern.where})"
        )
    out = []
    for node, j, statement in occurrences:
        stmt = f"{kern.rel}:{node.lineno}"
        what = f"work field {field!r} ({var} at {stmt}"
        what += ")" if j is None else f", element {j})"
        detail = {"var": var, "stmt": stmt, "element": j, "reg": None}
        block = kern.blocks[id(statement)]
        region = {
            k
            for k, f in enumerate(flow.frames)
            if f and f[0][0] == kern.path and _within(f[0], node) and not kern.param[k]
        }
        if not region:
            note = f"{what}: no instruction (dropped or folded by the compiler)"
            kern.dead.append((field, detail, block, note))
            continue
        if j is not None:
            region = {
                k for k in region if _part(kern, flow.frames[k], kern.fdef, node, j)
            }
        results = [
            (k, r)
            for k in sorted(region)
            for r in flow.dests[k]
            if flow.live_out(k, r, region)
        ]
        results = _final(flow, results, region)
        if (
            len(results) > 1
        ):  # a tie: the value is the one produced at the root expression
            at_root = [
                (k, r) for k, r in results
                if _part(kern, flow.frames[k], kern.fdef, node, j, _starts)
            ]  # fmt: skip
            results = at_root if len(at_root) == 1 else results
        if len(results) == 1:
            ((k, detail["reg"]),) = results
            out.append(Site(field, "assign", flow.site(k), detail))
            kern.live.setdefault(block, []).append(k)
        elif not results and j is not None:
            note = f"{what}: no value leaves its code (unused here)"
            kern.dead.append((field, detail, block, note))
        else:
            got = [f"{flow.describe(k)} -> {r}" for k, r in results]
            got = got or [flow.describe(k) for k in sorted(region)]
            raise AnchorError(
                f"{what}: {len(results)} results leave its code in {name}, need exactly "
                f"one: {'; '.join(got)}"
            )
    return out


def _novalue_sites(kern, notes):
    """A "novalue" site per occurrence without a value whose statement list gave another
    occurrence a site, at the last of those sites: the statement runs there (a constexpr
    branch the compiler dropped has no code at all), so the field counts as defined, as
    the TTGIR frontend counted it wherever its op survived to TTGIR; but no instruction
    is known to hold its value. Otherwise a note."""
    out = []
    for field, detail, block, note in kern.dead:
        live = kern.live.get(block)
        if live:
            out.append(Site(field, "novalue", kern.flow.site(max(live)), detail))
        else:
            notes.append(note)
    return out


def _blocks(fdef):
    """id(statement) -> (id(parent), field) of the statement list holding it."""
    out = {}
    for n in ast.walk(fdef):
        for name in ("body", "orelse", "finalbody"):
            stmts = getattr(n, name, None)
            if isinstance(stmts, list):
                out.update((id(s), (id(n), name)) for s in stmts)
    return out


def _assignment(n):
    """(targets, the node whose code computes the value) of an assignment statement."""
    if isinstance(n, ast.Assign):
        return n.targets, n.value
    if isinstance(n, ast.AnnAssign) and n.value is not None:
        return [n.target], n.value
    if isinstance(n, ast.AugAssign):  # Triton puts the operation at the statement
        return [n.target], n
    return None


def _bindings(fdef, name):
    """(span node, k, statement) of each assignment to `name` in fdef, in source order: the
    node whose code computes it (the value; for an AugAssign the statement), the index of
    `name` in a flat tuple target (None for the plain name) and the statement."""
    out, handled = [], set()
    for n in ast.walk(fdef):
        if isinstance(n, ast.For):  # deliberately no occurrence (see the docstring)
            handled.update(id(x) for x in ast.walk(n.target))
        assignment = _assignment(n)
        if assignment is None:
            continue
        targets, node = assignment
        for t in targets:
            flat = t.elts if isinstance(t, ast.Tuple) else ()
            if not all(isinstance(e, ast.Name) for e in flat):
                continue  # a nested target: refused below if it binds `name`
            for k, e in enumerate(flat or [t]):
                if isinstance(e, ast.Name) and e.id == name:
                    out.append((node, k if flat else None, n))
                    handled.add(id(e))
    for n in ast.walk(fdef):
        if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store):
            if id(n) not in handled:
                raise AnchorError(
                    f"{name} is bound at line {n.lineno} in a form work anchors do not "
                    "support (only a plain or flat tuple assignment)"
                )
    return sorted(out, key=lambda b: (b[0].lineno, b[0].col_offset))


def _within(frame, node):
    # Triton's columns are 1-based (ast col_offset + 1); end_col_offset is exclusive.
    pos = (frame[1], frame[2])
    return (
        (node.lineno, node.col_offset + 1)
        <= pos
        <= (node.end_lineno, node.end_col_offset)
    )


def _starts(frame, node):
    return (frame[1], frame[2]) == (node.lineno, node.col_offset + 1)


def _part(kern, frames, fdef, node, j, test=_within, depth=0):
    """Whether an instruction with these frames (outermost first, frames[0] in fdef's
    file) computes element j of node's value (all of it for None). With test=_starts:
    whether it sits at the position of that value's root expression, where Triton puts
    the op that produces the value (each AST node's op gets the node's position)."""
    if depth > 16:
        return False
    if isinstance(node, ast.Name):  # no code of its own: follow its assignments
        return any(
            _part(kern, frames, fdef, v, j if k is None else k, test, depth + 1)
            for v, k, _ in _bindings(fdef, node.id)
            if k is None or j is None
        )
    if j is not None and isinstance(node, ast.Tuple):
        e = node.elts[j] if j < len(node.elts) else None
        return e is not None and _part(kern, frames, fdef, e, None, test, depth + 1)
    if j is not None and isinstance(node, ast.IfExp):
        return any(
            _part(kern, frames, fdef, b, j, test, depth + 1)
            for b in (node.body, node.orelse)
        )
    if j is None:
        return test(frames[0], node)
    # A helper inlined at this call: its frame sits inside the call site's, whose
    # position is the call's own (an inner frame from an argument's call starts later).
    if isinstance(node, ast.Call) and len(frames) > 1 and _starts(frames[0], node):
        path, line, _ = frames[1]
        callee = _rel(path, kern.roots) is not None and _enclosing_def(path, line)
        return bool(callee) and any(
            _part(kern, frames[1:], callee, r.value, j, test, depth + 1)
            for r in ast.walk(callee)
            if isinstance(r, ast.Return) and r.value is not None
        )
    return False


def _final(flow, results, region):
    """The results leaving the region that no other one is computed from: several writes
    of one register in one basic block are one result, at the last write; a result that
    flows into another one's instruction inside the region gives way to it."""
    last = {}
    for k, r in results:
        key = (r, flow.block[k])
        last[key] = max(last.get(key, k), k)
    results = sorted((k, r) for (r, _b), k in last.items())
    order = sorted(region)

    def feeds(k, reg, target):
        regs = {reg}
        for n in order:
            if n > k and regs & flow.reads[n]:
                if n == target:
                    return True
                regs.update(flow.dests[n])
        return False

    return [
        (k, r)
        for k, r in results
        if not any(t != k and feeds(k, r, t) for t, _ in results)
    ]
