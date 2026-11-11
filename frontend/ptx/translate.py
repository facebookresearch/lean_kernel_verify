# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PTX core translator (design §5.2-§5.8): one kernel's PTX + its LaunchRecords -> Facts.

    translate(ptx_text, entry, launch_records, policy) -> Facts
    policy(family, **overrides) -> dict            the adapter conventions below
    param_table(records, family, fn, leaves) -> [param]
                                                   Memory's view of LaunchRecords
    mark_addr(kir)                                 frontend/cute/to_kir.py's (per role)
    writes(ptx_text, entry, params) -> [tensor] | None
                                                   the tensors a kernel may write, for
                                                   the stale-data checks of a refused one

The family assembly code (frontend/ttgir/to_kir.py build, frontend/cute/to_kir.py
Variant.kir) turns Facts into its KIR with its own naming, inputs and output files: it
keeps its tensors (shape/stride params, segments, roles, metadata contents), params,
inputs, grid (from the LaunchRecords), CLC tables and report, and takes the roles,
barriers, the tensors touched and written, the scalar params read and the work-event
count from the Facts. KIR `ir` is "ptx".

Where a variant runs: one translate() per compiled kernel (TTGIR spec, CuTe digest),
with the LaunchRecords of every launch of it (they must agree on block and cluster; the
first names the params unless the policy gives them). The Triton family with a cluster
of more than one CTA is a gap in v1 (design §1.3).

Policy (a dict; `policy()` fills the family defaults):
    family          "ttgir" | "cute"
    roles           "partition": the roles of cfg.roles (default / partition<k>, the
                    TTGIR names), no splitting: a branch on the warp is a mask;
                    "warps": partition refinement from the whole block, roles
                    warps<a>_<b> (CuTe)
    src_root        the export root: KIR locs name files below it
    loc_prefix      "fbcode/" (TTGIR) or "" (CuTe)
    loc_mode        "file_line": the innermost .loc frame below src_root (TLX, DSL
                    4.6.1); "line_only": DSL 4.4.2 (decision D1), the one .file is the
                    kernel file and only lines are trusted, a .loc on a blank or comment
                    position (or past the line's end) inherits the previous one;
                    instructions without a loc inherit the previous one
    asm_locs        True (DSL 4.6.1): an instruction inside inline asm takes the loc of
                    the asm block's start (synthetic asm locs)
    barrier_names   "ast" (TLX: the init line's assignment target) | "offset" (mbar_<off>)
    barrier_split   split barrier runs by their dynamic uses (CuTe) or not (TLX)
    max_iters       `while` budget: 65536 (TTGIR WHILE_FUEL) or the case's max_iters
    tpi             "one": threads_per_issue 1, arrive n = issuing threads (TTGIR);
                    "gcd": threads_per_issue = gcd of a role's arrive counts (CuTe)
    vectorize       True (TTGIR): a global access of all issuing threads is one event
                    with TExpr offsets over the threads, like the TTGIR frontend's one
                    event per tensor operation; False (CuTe): per-thread events in a
                    thread loop, like the CuTe frontend's (interp.Translator.mem_event)
    params          param_table() of the LaunchRecords, or the adapter's own
    tensors         {tensor: {"elem_bytes"}} of the tensors the pointer params name
    work            {"fields": [name, ...], "mode": "gate" (TTGIR) | "each" (CuTe),
                    "per": [fields naming tensor segments], "sites": anchor sites}: a
                    site is anchors.Site or a dict {field, kind: "load" | "assign" |
                    "novalue", instr: (function, index in fn.instructions(), line),
                    detail: {tensor (load) | reg (assign)}}; a "novalue" field (it is
                    defined there but no instruction holds its value, anchors.py) is
                    -1, refused if it names segments

Facts (refusals non-empty: the variant is refused, everything else empty):
    kernel, target ("sm_100" / "sm_103"), block, cluster
    roles      [{"name", "w0", "warps", "threads_per_issue", "body"}]: bodies are KIR
               statements (frontend/kir.py), addr marks applied per role (mark_addr,
               frontend/cute/to_kir.py's, plus: the mul.hi and sign-extension
               expansions are not marked)
    barriers   [{"name", "count", "num"}] (KIR)
    tensors    {name: {"uses": [gload|gstore|gatomic], "metadata", "elem_bytes"}}:
               the tensors the bodies access or `load`
    params     {name: width}: KIR scalar params the bodies read (ld.param)
    written    [tensor]: tensors with a gstore or gatomic
    work_events  number of work events emitted
    refusals   [{"kind": "gap" | "error", "msg"}]; an exception of the core is an "error"
               refusal (at the instruction being translated, if any), never raised

CLI (eyeballing the M0 dumps): python3 -I translate.py --dump DUMP.json [--case
CASE.json] [--body] [--kir] prints the Facts of every variant of a TTGIR or CuTe dump;
--kir assembles a minimal KIR (constant shapes from the first launch) and validates it
with frontend/kir.py.
"""

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

if __package__ in (None, ""):  # python3 -I path/to/translate.py
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "ptx"  # noqa: A001

from . import cfg as C, interp, mem as M, ops, parse, regions as R  # noqa: E402

DEFAULTS = {
    "ttgir": {"family": "ttgir", "roles": "partition", "loc_prefix": "fbcode/",
              "loc_mode": "file_line", "asm_locs": False, "barrier_names": "ast",
              "barrier_split": False, "max_iters": 1 << 16, "tpi": "one", "vectorize": True},
    "cute": {"family": "cute", "roles": "warps", "loc_prefix": "", "loc_mode": "line_only",
             "asm_locs": False, "barrier_names": "offset", "barrier_split": True,
             "max_iters": 4096, "tpi": "gcd", "vectorize": False},
}  # fmt: skip


def policy(family: str, **overrides) -> dict:
    return {**DEFAULTS[family], "src_root": "", "params": [], "tensors": {},
            "work": {"fields": [], "mode": "gate" if family == "ttgir" else "each",
                     "sites": []}, **overrides}  # fmt: skip


@dataclass
class Facts:
    kernel: str
    target: str = ""
    block: tuple = ()
    cluster: tuple = (1, 1, 1)
    roles: list = field(default_factory=list)
    barriers: list = field(default_factory=list)
    tensors: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    written: list = field(default_factory=list)
    work_events: int = 0
    refusals: list = field(default_factory=list)


def translate(ptx_text: str, entry: str, launch_records: list, pol: dict) -> Facts:
    facts = Facts(entry)
    try:
        _translate(facts, ptx_text, entry, launch_records, pol)
    except interp.Unsupported as e:
        facts.refusals.append({"kind": e.kind, "msg": str(e)})
    except (C.CfgError, R.RegionError) as e:
        facts.refusals.append({"kind": "gap", "msg": str(e)})
    except Exception as e:  # a PtxSyntaxError, bad inputs or a bug: refused, not raised
        facts.refusals.append({"kind": "error", "msg": f"{type(e).__name__}: {e}"})
    if facts.refusals:
        facts.roles, facts.barriers, facts.tensors, facts.params = [], [], {}, {}
    return facts


def _translate(facts, ptx_text, entry, records, pol) -> None:
    module = parse.parse(ptx_text)
    fn = module.function(entry)
    facts.target = module.target.removesuffix("a")
    facts.block, facts.cluster = _launch_dims(fn, records)
    if pol["family"] == "ttgir" and facts.cluster != (1, 1, 1):
        raise interp.Unsupported("launch", fn.name, f"cluster {facts.cluster}: Triton kernels with "
                            "clusters are not in v1 (design §1.3)")  # fmt: skip
    cfg = C.build(fn)
    roles, _dispatch = C.roles(cfg)
    locs, srclocs = _locs(fn, pol)
    shared = {v.name for v in module.vars if v.space == "shared"}
    mem = M.Memory(
        cfg, pol["params"], pol["tensors"], {**pol, "shared_symbols": shared}
    )
    work = pol.get("work") or {}
    k = interp.Kernel(fn, mem, locs, facts.block, facts.cluster, pol["max_iters"],
                 split=pol["roles"] == "warps", vectorize=bool(pol.get("vectorize")),
                 work=_sites(fn, work.get("sites", ())),
                 work_fields=tuple(work.get("fields", ())), work_mode=work.get("mode", "gate"),
                 work_per=tuple(work.get("per", ())),
                 srclocs=srclocs)  # fmt: skip
    translators = []
    for role in roles:
        tree = R.build(cfg, role)
        warps = role.warps or tuple(range(facts.block[0] // 32))
        if list(warps) != list(range(warps[0], warps[0] + len(warps))):
            raise interp.Unsupported(
                "role", role.name, f"warps {warps} are not contiguous"
            )
        translators += _role(k, tree, role.name, warps[0], len(warps), pol)
    facts.barriers = mem.barriers(translators)
    for tr in translators:
        name = (
            tr.name
            if pol["roles"] == "partition"
            else f"warps{tr.w0}_{tr.w0 + tr.nw - 1}"
        )
        r = {"name": name, "w0": tr.w0, "warps": tr.nw, "threads_per_issue": tr.tpi,
             "body": tr.stmts}  # fmt: skip
        mark_addr({"roles": [r]})
        facts.roles.append(r)
        facts.work_events += tr.work_events
    loaded = {
        n["tensor"]
        for r in facts.roles
        for n in _expr_nodes(r["body"])
        if n.get("e") == "load"
    }
    for name, t in mem.tensors.items():
        if t["uses"] or name in loaded:
            facts.tensors[name] = {"uses": sorted(t["uses"]), "metadata": name in loaded,
                                   "elem_bytes": t["elem_bytes"]}  # fmt: skip
    facts.written = sorted(
        n for n, t in facts.tensors.items() if {"gstore", "gatomic"} & set(t["uses"])
    )
    facts.params = dict(sorted(mem.used_params.items()))


def _role(k, tree, name, w0, nw, pol) -> list:
    """The translators of one role: itself, or (policy "warps") its parts after
    partition refinement (frontend/cute/to_kir.py Variant.roles)."""
    out, todo = [], [(w0, nw)]
    while todo:
        a, n = todo.pop(0)
        tr = interp.Translator(k, tree, a, n, name)
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
        if k.work_mode == "each" and k.work_fields:  # fields are -1 until an anchor
            tr.stmts[:0] = [{"op": "let", "var": f"work_{f}", "expr": {"e": "const", "w": 64, "v": -1}}
                            for f in k.work_fields]  # fmt: skip
        out.append(tr)
    return out


def _launch_dims(fn, records) -> tuple:
    """(block, cluster): the launches' (all equal), checked against .reqntid; the block
    must be (32k, 1, 1) (%tid.y/z fold to 0)."""
    blocks = {tuple(r["block"]) for r in records} or {None}
    clusters = {tuple(r.get("cluster") or (1, 1, 1)) for r in records} or {(1, 1, 1)}
    req = fn.directives.get("reqntid")
    if len(blocks) != 1 or len(clusters) != 1:
        raise interp.Unsupported(
            "launch",
            fn.name,
            f"launches differ in block {blocks} or cluster {clusters}",
            "error",
        )
    block = blocks.pop() or (tuple(req) + (1, 1))[:3]
    if req and tuple(req) + (1,) * (3 - len(req)) != tuple(block):
        raise interp.Unsupported(
            "launch", fn.name, f"block {block} but .reqntid {req}", "error"
        )
    if block[0] % 32 or block[1] != 1 or block[2] != 1:
        raise interp.Unsupported(
            "launch", fn.name, f"block {block} (only (32k, 1, 1) is supported)"
        )
    return tuple(block), clusters.pop()


# ---------------------------------------------------------------------- locations


def _locs(fn, pol) -> tuple:
    """({id(instr): KIR loc}, {id(instr): (source path, line)}) per design §5.2.3."""
    root = str(pol.get("src_root") or "").rstrip("/") + "/"
    prefix, mode = pol.get("loc_prefix", ""), pol.get("loc_mode", "file_line")
    locs, srcs = {}, {}
    cur, src = "unknown", None
    lines: dict = {}
    for i in fn.instructions():
        loc = i.loc
        if pol.get("asm_locs") and i.asm is not None:
            loc = fn.asm_locs[i.asm] or loc
        frame = _frame(loc, root, mode, lines)
        if frame is not None:
            src = frame
            cur = f"{prefix}{frame[0][len(root) :]}:{frame[1]}"
        locs[id(i)], srcs[id(i)] = cur, src
    return locs, srcs


def _frame(loc, root, mode, lines):
    """(path, line) of the frame a KIR loc names, or None (inherit the previous one)."""
    if loc is None:
        return None
    if mode == "line_only":
        if not loc.file.startswith(root) or not _plausible(loc, lines):
            return None
        return loc.file, loc.line
    for f in loc.chain():
        if f.file.startswith(root):
            return f.file, f.line
    return None


def _plausible(loc, lines) -> bool:
    """D1: the kernel file's line has code at the column (not blank, a comment, or past
    the end of the line). The DSL writes Python's 0-based ast columns."""
    if loc.file not in lines:
        try:
            lines[loc.file] = Path(loc.file).read_text().splitlines()
        except OSError:
            lines[loc.file] = []
    text = lines[loc.file]
    if not 1 <= loc.line <= len(text):
        return False
    s = text[loc.line - 1]
    code = s.split("#", 1)[0]
    if not code.strip():
        return False
    return loc.col < len(code) and not code[loc.col].isspace()


def _sites(fn, sites) -> dict:
    """{id(instr): [site dict]} of the work anchors (anchors.Site or dicts)."""
    instrs = fn.instructions()
    out: dict = {}
    for s in sites:
        s = s if isinstance(s, dict) else asdict(s)
        fname, index, line = s["instr"] if not isinstance(s["instr"], dict) else (
            s["instr"]["function"], s["instr"]["index"], s["instr"]["line"])  # fmt: skip
        if fname != fn.name:
            continue
        i = instrs[index]
        if i.line != line:
            raise interp.Unsupported(
                "work",
                f"line {line}",
                f"anchor site {index} is at line {i.line}",
                "error",
            )
        detail = s.get("detail") or {}
        out.setdefault(id(i), []).append(
            {"field": s["field"], "kind": s["kind"], **detail}
        )
    return out


# ------------------------------------------------------------------ addr marking
#
# frontend/cute/to_kir.py mark_addr, ported unchanged but for the design §5.8 exception,
# run per role: the nodes of a mul.hi expansion and of a sign extension (or signed bit
# field) expansion are not marked, their operands are followed. In Exprs and TExprs alike.

_ADDR_OPS = ("add", "sub", "mul", "shl")  # wraps are P0-1 violations on address paths
_MEM_EVENTS = ("gload", "gstore", "gatomic")


def _kind(node):
    return node.get("e") or node.get("t")


def _cval(node):
    """The value of a constant Expr or TExpr node, else None."""
    return node["v"] if _kind(node) in ("const", "full") else None


def _mark(node) -> None:
    kind = _kind(node)
    if kind == "trunc" or kind == "bin" and node["o"] in _ADDR_OPS:
        node["addr"] = True


def _statements(stmts):
    for s in stmts:
        yield s
        for k in ("then", "else", "body"):
            yield from _statements(s.get(k, []))


def _children(node) -> list:
    """Sub-nodes of a KIR Expr or TExpr (operands and load indices)."""
    return [
        node[k] for k in ("a", "b", "c") if isinstance(node.get(k), dict)
    ] + node.get("index", [])


def _nodes(node):
    yield node
    for c in _children(node):
        yield from _nodes(c)


def _mulhi_operands(e):
    """The operands of a mul.hi expansion trunc(lshr(mul(ext a, ext b), w), w)
    (expr.mul_hi; a constant operand is folded, as in a magic-number division), or
    None."""
    if _kind(e) != "trunc":
        return None
    s = e["a"]
    if _kind(s) != "bin" or s["o"] != "lshr" or _cval(s["b"]) != e["w"]:
        return None
    m = s["a"]
    if _kind(m) != "bin" or m["o"] != "mul" or m["w"] != 2 * e["w"]:
        return None
    if any(_kind(x) not in ("ext", "const", "full") for x in (m["a"], m["b"])):
        return None
    return [x["a"] for x in (m["a"], m["b"]) if _kind(x) == "ext"]


def _sext_operand(e):
    """[x] for ashr(shl(x, k), k2) with constants k <= k2: a sign extension from a
    narrow width or a signed bit field (expr._sx, expr.bfe), whose shl drops the bits
    above the field by design (cvt.s32.s8 truncates), so its wrap is no address
    overflow; else None."""
    if _kind(e) != "bin" or e["o"] != "ashr" or _kind(e["a"]) != "bin":
        return None
    s = e["a"]
    k, k2 = _cval(s["b"]), _cval(e["b"])
    if s["o"] != "shl" or k is None or k2 is None or not 0 <= k <= k2:
        return None
    return [s["a"]]


def mark_addr(kir: dict) -> None:
    """Set "addr" on every add/sub/mul/shl and trunc whose value reaches a global-memory
    offset, a TMA coordinate or a metadata load index, through any chain of `let`s (Lean
    reports their wraps and lossy truncations as P0-1 violations)."""
    stmts = [s for r in kir["roles"] for s in _statements(r["body"])]
    lets: dict = {}
    seeds: list = []
    for s in stmts:
        if s["op"] == "let":
            lets.setdefault(s["var"], []).append(s["expr"])
        if s["op"] == "event" and s["ev"] in _MEM_EVENTS:
            seeds.append(s["offsets"])  # the TExpr arithmetic and the Exprs it splats
        if s["op"] == "event" and s["ev"] == "tma_load":
            seeds += s["coords"]
        roots = [v for k, v in s.items() if isinstance(v, dict) and "e" in v] + s.get(
            "coords", []
        )
        seeds += [
            i
            for r in roots
            for n in _nodes(r)
            if n.get("e") == "load"
            for i in n["index"]
        ]
    done: set = set()
    while seeds:
        e = seeds.pop()
        if id(e) in done:
            continue
        done.add(id(e))
        follow = _mulhi_operands(e)
        if follow is None:
            follow = _sext_operand(e)
        if follow is not None:
            seeds += follow
            continue
        _mark(e)
        if e.get("e") == "var":
            seeds += lets.get(e["name"], [])
        seeds += _children(e)


def _expr_nodes(stmts):
    for s in _statements(stmts):
        for v in s.values():
            if isinstance(v, dict) and ("e" in v or "t" in v):
                yield from _nodes(v)
            elif isinstance(v, dict):  # work fields
                for x in v.values():
                    if isinstance(x, dict) and "e" in x:
                        yield from _nodes(x)
        for x in s.get("coords", []):
            yield from _nodes(x)


# ------------------------------------------------------------- launch records


def param_table(
    records: list, family: str, fn=None, leaves: list | None = None
) -> list:
    """Memory's parameter view of a variant's LaunchRecords (mem.py): per PTX parameter
    its kind, KIR name (a scalar param, or the tensor a pointer names) and struct
    fields. TTGIR names are the records' (jit argument names); CuTe top-level scalars are
    param<i>, fields keep the records' param<i>_<off>, and a pointer field names the
    tensor it resolves to exactly in the records that resolve it (a zero-length tensor
    resolves nowhere). Pointers that alias one host tensor passed under several jit names
    (`leaves`, the jit argument order; same storage in every record) are named in that
    order, the k-th such field the k-th name, as pass-through arguments are; any other
    ambiguity leaves the field unnamed (a load of it is Data)."""
    out, taken = [], {}
    for k, p in enumerate(records[0]["params"]):
        ptx = fn.params[k].name if fn is not None else p.get("ptx", p["name"])
        kind = {"aggregate": "aggregate", "bytes": "aggregate"}.get(
            p["kind"], p["kind"]
        )
        name = p["name"] if family == "ttgir" else f"param{p['i']}"
        fields = {}
        for f in p.get("fields") or ():
            fname = f["name"]
            if f["kind"] == "ptr":
                fname = _pointer_name(records, k, f["off"], leaves or [], taken)
            fields[f["off"]] = {"size": f["size"], "kind": f["kind"], "name": fname}
        out.append(
            {
                "ptx": ptx,
                "kind": kind,
                "name": name,
                "size": p["size"],
                "fields": fields,
            }
        )
    return out


def _pointer_name(records, k, off, leaves, taken):
    sets = set()
    for r in records:
        rf = next(x for x in r["params"][k]["fields"] if x["off"] == off)
        exact = [c for c in rf.get("tensor") or () if c["match"] == "exact"]
        if exact:
            if len({tuple(c["storage"]) for c in exact}) != 1:
                return None  # distinct storages at one address: not an alias
            sets.add(frozenset(c["tensor"] for c in exact))
    if len(sets) != 1:
        return None
    (names,) = sets
    if len(names) == 1:
        return next(iter(names))
    order = [n for n in leaves if n in names]
    i = taken.get(names, 0)
    taken[names] = i + 1
    return order[i] if len(order) == len(names) and i < len(order) else None


# ------------------------------------------------------------------- write set
#
# The op table's global-memory writes (ops.py) with an address operand: generic and
# global stores, atomics, bulk copies and reductions to global memory. TMA stores and
# in-kernel tensormaps write through descriptors, whose base this pass does not follow.
_WRITES = ("st.global", "atom.global.", "red.global.", "cp.async.bulk.global.shared",
           "cp.reduce.async.bulk.global.shared")  # fmt: skip
_UNKNOWN = "?"  # a base no named pointer accounts for
_SHARED = "%shared"


def _descriptor_write(mn: str) -> bool:
    return mn.startswith("tensormap.cp_fenceproxy") or (
        mn.startswith(("cp.async.bulk.tensor", "cp.reduce.async.bulk.tensor"))
        and ".global.shared" in mn
    )


def writes(ptx_text: str, entry: str, params: list) -> list | None:
    """The tensors a kernel may write, for the stale-data checks of a variant the
    translation refused (design §5.1: the write set from the PTX, else every pointer
    argument): the named pointers (param_table's `params`) the address of each global
    write derives from (_Bases). None when some write's address has an unknown or no
    base or goes through a descriptor, or an instruction has no op-table row or is
    refused by it (a call of a device function may store anywhere; of the refused rows
    only try_wait/test_wait, which read shared memory, are known to write nothing): then
    any argument may be written."""
    module = parse.parse(ptx_text)
    fn = module.function(entry)
    try:
        for i in fn.instructions():
            if ops.classify(i)[0] == "refuse" and not ops.mnemonic(i.opcode).startswith(
                ("mbarrier.try_wait", "mbarrier.test_wait")
            ):
                return None
    except ops.UnknownInstruction:
        return None
    shared = {v.name for v in module.vars + fn.vars if v.space == "shared"}
    bases = _Bases(fn, {p["ptx"]: p for p in params}, shared)
    out: set = set()
    for i, mn in bases.instrs:
        if _descriptor_write(mn):
            return None
        if not (mn == "st" or mn.startswith(_WRITES)):
            continue
        got = bases.of(next(op for op in i.operands if op[0] == "mem"))
        if mn == "st" and got == {_SHARED}:
            continue  # a generic store into shared memory
        if not got - {_SHARED} or _UNKNOWN in got:
            return None
        out |= got - {_SHARED}
    return sorted(out)


class _Bases:
    """Per register, the pointers its value may derive from: the union of its operands'
    over every instruction writing it (flow-insensitive, so an over-approximation), to a
    fixed point. A pointer parameter or field is its tensor, a shared symbol _SHARED, a
    loaded 64-bit value or any other symbol _UNKNOWN; scalars and narrower loads none."""

    def __init__(self, fn, table: dict, shared: set):
        self.fn, self.table, self.shared = fn, table, shared
        self.regs: dict = {}
        self.instrs = [(i, ops.mnemonic(i.opcode)) for i in fn.instructions()]
        changed = True
        while changed:  # sets only grow
            changed = False
            for i, mn in self.instrs:
                for r, got in self.defined(i, mn):
                    if not got <= self.regs.get(r, set()):
                        self.regs[r] = self.regs.get(r, set()) | got
                        changed = True

    def of(self, op) -> set:
        kind = op[0]
        if kind == "reg":
            return self.regs.get(op[1], set())
        if kind == "sym":
            return {_SHARED} if op[1] in self.shared else {_UNKNOWN}
        if kind in ("mem", "not"):
            return self.of(op[1])
        if kind in ("vec", "list"):
            return set().union(*(self.of(x) for x in op[1]))
        return self.of(op[1]) | self.of(op[2]) if kind == "pair" else set()

    def defined(self, i, mn) -> list:
        """(register, bases) of each register instruction i writes."""
        ds = R.dests(i)
        if mn == "ld.param":
            sym, off = i.operands[1][1], i.operands[1][2]
            p = self.table.get(sym[1]) if sym[0] == "sym" else None
            size = int(re.findall(r"\d+", i.opcode.rsplit(".", 1)[-1])[0]) // 8
            return [(r, _param_bases(p, off + k * size)) for k, r in enumerate(ds)]
        if mn.startswith(("ld", "atom")):  # memory: a 64-bit value may be a pointer
            wide = {r for r in ds if self.fn.regs.get(r, "")[1:] == "64"}
            return [(r, {_UNKNOWN} if r in wide else set()) for r in ds]
        got = set().union(*(self.of(op) for op in i.operands[1:]))
        return [(r, got) for r in ds]


def _param_bases(p, off: int) -> set:
    """The bases of an ld.param result: a pointer's tensor, nothing for a scalar."""
    if p is None:
        return {_UNKNOWN}
    entry = p if p["kind"] in ("ptr", "scalar") and off == 0 else p["fields"].get(off)
    if entry is None or entry["kind"] not in ("ptr", "scalar"):
        return {_UNKNOWN}
    if entry["kind"] == "scalar":
        return set()
    return {entry["name"]} if entry["name"] else {_UNKNOWN}


TORCH_BYTES = {
    "float64": 8, "int64": 8, "float32": 4, "int32": 4, "bfloat16": 2, "float16": 2,
    "int16": 2, "uint8": 1, "int8": 1, "bool": 1, "float8_e4m3fn": 1, "float8_e5m2": 1,
    "float8_e8m0fnu": 1, "float4_e2m1fn_x2": 1,
}  # fmt: skip


def _cute_args(compile_rec: dict, args: list) -> dict:
    """{leaf name: launch argument} of a CuTe launch (tuples flattened, None dropped)."""
    flat = []

    def go(a):
        if a["kind"] == "tuple":
            for x in a["items"]:
                go(x)
        elif a["kind"] != "none":
            flat.append(a)

    for a in args:
        go(a)
    leaves = [
        x for p in compile_rec["params"] if not p["constexpr"] for x in p["leaves"]
    ]
    return dict(zip(leaves, flat))


# ---------------------------------------------------------------------------- CLI


def _variants(dump: dict, where: Path):
    """(label, ptx path, entry, [LaunchRecord], family, tensors {name: elem_bytes},
    first-launch snapshot {name: {shape, strides}}, jit leaf order) of every variant of a
    dump."""
    if "specs" in dump:  # TTGIR
        for key, spec in dump["specs"].items():
            las = [
                la
                for inp in dump["inputs"]
                for la in inp["launches"]
                if la.get("spec") == key
            ]
            recs = [la["launch"] for la in las if "launch" in la]
            snap = las[0]["args"] if las else {}
            tensors = {
                n: a["elem_bytes"] for n, a in snap.items() if a.get("kind") == "tensor"
            }
            shapes = {n: a for n, a in snap.items() if a.get("kind") == "tensor"}
            yield (key, where / spec["ptx"], recs[0]["entry"] if recs else None, recs, "ttgir",
                   tensors, shapes, [])  # fmt: skip
        return
    by_digest: dict = {}
    for c in dump["compiles"]:  # one variant per initial-MLIR digest, as cute/to_kir.py
        by_digest.setdefault(c["digest"], []).append(c)
    for digest, cs in by_digest.items():
        c = cs[0]
        idx = {x["index"] for x in cs}
        las = [
            la
            for inp in dump["inputs"]
            for la in inp["launches"]
            if la["compile"] in idx
        ]
        recs = [r for la in las for r in la.get("launch_records", [])]
        args = _cute_args(c, las[0]["args"]) if las else {}
        tensors = {
            n: TORCH_BYTES[a["dtype"]] for n, a in args.items() if a["kind"] == "tensor"
        }
        shapes = {n: {"shape": a["shape"], "strides": a["stride"]}
                  for n, a in args.items() if a["kind"] == "tensor"}  # fmt: skip
        label = f"{c['callable'].rsplit('.', 1)[-1]}__{digest}"
        leaves = [x for p in c["params"] if not p["constexpr"] for x in p["leaves"]]
        yield (label, c.get("ptx", {}).get("file"), recs[0]["entry"] if recs else None, recs,
               "cute", tensors, shapes, leaves)  # fmt: skip


def _anchor_sites(work: dict, family: str, fn, root: str, case: dict) -> list:
    """Sites of the case's work anchors (frontend/ptx/anchors.py)."""
    if not work:
        return []
    from . import anchors

    mode = "line_only" if family == "cute" else "file_line"
    got = anchors.resolve(work, family, fn, [root], mode)
    return list(got[0] if isinstance(got, tuple) else got)


def _per_fields(case: dict) -> list:
    """Work fields the case's tensor annotations name as segment `per` fields."""
    tensors = dict(case.get("tensors", {}))
    for k in case.get("kernels", {}).values():
        tensors.update(k.get("tensors", {}))
    return sorted({t["segments"]["per"] for t in tensors.values() if t.get("segments")})


def _minimal_kir(
    facts: Facts, label: str, record: dict, shapes: dict, params: list
) -> dict:
    """A KIR around the Facts with the first launch's constant tensor shapes, its scalar
    values and zero metadata contents: enough for frontend/kir.py."""
    tensors, metadata = [], {}
    for name, t in facts.tensors.items():
        s = shapes.get(name, {"shape": [1], "strides": [1]})
        if t["metadata"]:
            n = max(1, 1 + sum((d - 1) * st for d, st in zip(s["shape"], s["strides"])))
            dims, metadata[name] = ([n], [1]), [0] * n
        else:
            dims = (s["shape"], s["strides"])
        tensors.append({"name": name, "kind": "global", "elem_bytes": t["elem_bytes"],
                        "shape": [{"e": "const", "w": 64, "v": v} for v in dims[0]],
                        "strides": [{"e": "const", "w": 64, "v": v} for v in dims[1]],
                        "segments": None, "role": "output" if name in facts.written else "input",
                        "metadata": t["metadata"]})  # fmt: skip
    values = {}
    for p, rp in zip(params, record["params"]):
        if p["kind"] == "scalar":
            values[p["name"]] = rp.get("value") or 0
        for off, f in p["fields"].items():
            if f["kind"] == "scalar":
                values[f["name"]] = (
                    next(
                        (
                            x.get("value")
                            for x in rp.get("fields", [])
                            if x["off"] == off
                        ),
                        0,
                    )
                    or 0
                )
    return {
        "kir_version": 1, "kernel": "k", "source": "ptx", "ir": "ptx", "target": facts.target,
        "grid": [{"e": "const", "w": 32, "v": g} for g in record["grid"]],
        "params": [{"name": n, "type": f"i{w}"} for n, w in facts.params.items()],
        **({"cluster": list(facts.cluster)} if facts.cluster != (1, 1, 1) else {}),
        "tensors": tensors, "barriers": facts.barriers,
        "roles": [{k: r[k] for k in ("name", "warps", "threads_per_issue", "body")} for r in facts.roles],
        "case": {"name": label, "inputs": [{
            "name": "first", "params": {n: values.get(n, 0) for n in facts.params},
            "metadata": metadata, "intended_writes": dict.fromkeys(facts.written, "unchecked")}]},
    }  # fmt: skip


def _validate(obj) -> list:
    spec = importlib.util.spec_from_file_location(
        "lkv_kir_validator", Path(__file__).resolve().parents[1] / "kir.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the KIR validator frontend/kir.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.validate(obj)


def _summary(facts: Facts) -> str:
    out = [
        f"  kernel {facts.kernel[:80]} target {facts.target} block {facts.block} cluster {facts.cluster}"
    ]
    for r in facts.roles:
        evs = {}
        for s in _statements(r["body"]):
            if s["op"] == "event":
                evs[s["ev"]] = evs.get(s["ev"], 0) + 1
        size = len(json.dumps(r["body"]))
        out.append(f"  role {r['name']} warps {r['warps']} tpi {r['threads_per_issue']} "
                   f"{size} bytes events {dict(sorted(evs.items()))}")  # fmt: skip
    out.append(f"  barriers {[(b['name'], b['num']['v']) for b in facts.barriers]}")
    out.append(f"  tensors {facts.tensors}")
    out.append(f"  params {facts.params} work_events {facts.work_events}")
    for f in facts.refusals:
        out.append(f"  REFUSED ({f['kind']}): {f['msg']}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--case", type=Path)
    ap.add_argument("--only", help="variants whose label contains this text")
    ap.add_argument("--body", action="store_true", help="print the role bodies (JSON)")
    ap.add_argument(
        "--kir", action="store_true", help="validate a minimal assembled KIR"
    )
    a = ap.parse_args()
    dump = json.loads(a.dump.read_text())
    case: dict = json.loads(a.case.read_text()) if a.case else {}
    root = dump.get("src_root") or dump.get("source", {}).get("src", "")
    bad = 0
    for label, ptx, entry, recs, family, tensors, shapes, leaves in _variants(
        dump, a.dump.parent
    ):
        if a.only and a.only not in label:
            continue
        print(f"{label} ({family})")
        if not recs or not ptx or entry is None:
            print("  no LaunchRecord or PTX")
            continue
        text = Path(ptx).read_text()
        fn = parse.parse(text).function(entry)
        params = param_table(recs, family, fn, leaves)
        work = case.get("work", {}) if family == "ttgir" else (
            case.get("kernels", {}).get(label.split("__")[0], {}).get("work", {}))  # fmt: skip
        sites = _anchor_sites(work, family, fn, root, case)
        pol = policy(family, src_root=root, params=params,
                     tensors={n: {"elem_bytes": b} for n, b in tensors.items()},
                     max_iters=(1 << 16) if family == "ttgir" else int(case.get("max_iters", 4096)),
                     work={"fields": list(work), "mode": "gate" if family == "ttgir" else "each",
                           "per": _per_fields(case), "sites": sites})  # fmt: skip
        facts = translate(text, entry, recs, pol)
        print(_summary(facts))
        if a.body:
            print(json.dumps(facts.roles, indent=1))
        if a.kir and not facts.refusals:
            errors = _validate(_minimal_kir(facts, label, recs[0], shapes, params))
            print(f"  kir.py: {'OK' if not errors else errors[:5]}")
            bad += bool(errors)
        bad += bool(facts.refusals)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
