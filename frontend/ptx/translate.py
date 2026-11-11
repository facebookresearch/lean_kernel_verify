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
    tma             "desc" (TTGIR): TMA events on host descriptors, no base footprint;
                    "footprint" (CuTe): a tma_load plus its gload footprint on the base,
                    stores and reduces as gstore/gatomic footprints (tma.py)
    desc_names      "param" (TTGIR: the jit name, the kernel's <d>.shape.i params) or
                    "tma" (CuTe: tma<k>_<base>, synthetic <d>.shape.i params; tma.py)
    ptxas           the V-string of the compiler that built the kernel's cubin (CuTe:
                    its .note.nv.tkinfo, tma.tkinfo_version); unset, the LaunchRecords'
                    "ptxas" (TTGIR: `ptxas --version` at dump time). A TMA or bulk copy
                    several lanes of a warp may execute is translated only if it is in
                    tma.ISSUE_PTXAS and the PTX .target in tma.ISSUE_TARGETS, else
                    refused (tma.py, "Issuing threads")
    blocks          {descriptor: block} (TTGIR: a host descriptor's !tt.tensordesc block,
                    KIR order): its KIR box; Triton cuts a block into hardware boxes
                    (the record's), one message each, which tma.py merges into one event
    params          param_table() of the LaunchRecords, or the adapter's own
    tensors         {tensor: {"elem_bytes"}} of the tensors the pointer params name
                    (CuTe: optionally the first launch's "shape" and "strides", which
                    pick the base of a descriptor over an aliased tensor, tma.py)
    p1              True: the P1 pass (CuTe only, tmem.py): the roles also carry the
                    tmem, tmem_wait and named_bar events of P1-1, a role whose warps
                    issue an mbarrier wait, arrive or commit only in part is split, and
                    Facts.tmem is set (refusals tagged R1-R10)
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
    written    [tensor]: tensors with a gstore or gatomic, descriptors with a
               tma_store or tma_reduce
    descs      {name: {kind: "host" | "bulk", base, elem_bytes, box, shape, strides,
               uses}}: the KIR `desc` tensors the bodies name (tma.py), shape/strides KIR
               JSON over `params` (None for a bulk descriptor: the assembly gives it its
               base's extent and stride 1, as the CuTe frontend); role, segments
               (tma.segment_dims for an inherited one) and metadata are the assembly's
    values     [{param: value}] per LaunchRecord: the synthetic params of `descs`
               (CuTe <d>.shape.i, <d>.stride.i, <d>.off; TTGIR's are the kernel's own),
               which the assembly adds to each input
    work_events  number of work events emitted
    clc        the bodies use CLC (sync.py): the assembly declares __clc_response,
               __clc_schedule and __clc_launched and makes the per-schedule inputs, as
               frontend/ttgir/to_kir.py build does for its Translator's uses_clc
    tmem       the P1 pass: the P1 KIR's top-level `tmem` {"cols", "base"}, None if the
               kernel allocates and accesses no TMEM (P1-1 has nothing to check)
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

from . import (  # noqa: E402
    cfg as C,
    cluster,
    interp,
    mem as M,
    ops,
    parse,
    regions as R,
    sync,
    tma,
    tmem,
)

DEFAULTS = {
    "ttgir": {"family": "ttgir", "roles": "partition", "loc_prefix": "fbcode/",
              "loc_mode": "file_line", "asm_locs": False, "barrier_names": "ast",
              "barrier_split": False, "max_iters": 1 << 16, "tpi": "one", "vectorize": True,
              "tma": "desc", "desc_names": "param", "ptxas": None},
    "cute": {"family": "cute", "roles": "warps", "loc_prefix": "", "loc_mode": "line_only",
             "asm_locs": False, "barrier_names": "offset", "barrier_split": True,
             "max_iters": 4096, "tpi": "gcd", "vectorize": False,
             "tma": "footprint", "desc_names": "tma", "ptxas": None},
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
    clc: bool = False
    descs: dict = field(default_factory=dict)
    values: list = field(default_factory=list)
    tmem: dict | None = None
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
        facts.descs, facts.values, facts.tmem = {}, [], None
    return facts


def _translate(facts, ptx_text, entry, records, pol) -> None:
    if pol.get("ptxas") is None:  # TTGIR: dump.py records it in every LaunchRecord
        got = {r.get("ptxas") for r in records}
        pol = {**pol, "ptxas": got.pop() if len(got) == 1 else None}
    module = parse.parse(ptx_text)
    fn = module.function(entry)
    facts.target = module.target.removesuffix("a")
    facts.block, facts.cluster = _launch_dims(fn, records)
    if pol["family"] == "ttgir" and facts.cluster != (1, 1, 1):
        raise interp.Unsupported("launch", fn.name, f"cluster {facts.cluster}: Triton kernels with "
                            "clusters are not in v1 (design §1.3)")  # fmt: skip
    tcgen05 = any(i.opcode.startswith("tcgen05.") for i in fn.instructions())
    p1 = tmem.P1() if pol.get("p1") and tcgen05 else None  # interp.Kernel.p1
    if p1 is not None and pol["family"] != "cute":
        raise interp.Unsupported("P1", fn.name, "P1-1 on Triton kernels is M5 (D3)")
    cfg = C.build(fn)
    roles, dispatch = C.roles(cfg, pol["family"])
    locs, srclocs = _locs(fn, pol)
    shared = {v.name for v in module.vars if v.space == "shared"}
    align = {v.name: v.align for v in module.vars + fn.vars if v.space == "shared"}
    mem = M.Memory(
        cfg,
        pol["params"],
        pol["tensors"],
        {
            **pol,
            "shared_symbols": shared,
            "records": records,
            "target": facts.target,
            "shared_align": align,  # cluster.py: the peer-bit mask
            "dispatch": dispatch is not None,  # mem.py: generic shared writes
        },
    )
    work = pol.get("work") or {}
    k = interp.Kernel(fn, mem, locs, facts.block, facts.cluster, pol["max_iters"],
                 split=pol["roles"] == "warps", vectorize=bool(pol.get("vectorize")),
                 work=_sites(fn, work.get("sites", ())),
                 work_fields=tuple(work.get("fields", ())), work_mode=work.get("mode", "gate"),
                 work_per=tuple(work.get("per", ())),
                 srclocs=srclocs, p1=p1)  # fmt: skip
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
    facts.clc = sync.finish(translators)
    facts.tmem = p1.top(translators) if p1 is not None else None
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
    facts.descs, facts.values = mem.tma.facts()
    facts.written = sorted(
        [n for n, t in facts.tensors.items() if {"gstore", "gatomic"} & set(t["uses"])]
        + [
            n
            for n, d in facts.descs.items()
            if {"tma_store", "tma_reduce"} & set(d["uses"])
        ]
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
        tr.stmts[:0] = cluster.prologue(tr)  # the role's cluster barrier parity
        if out and _idle(out[-1]) and _idle(tr) and out[-1].w0 + out[-1].nw == a:
            out[-1].nw += n  # warps that only park, split by LLVM's branches: one role
            continue
        out.append(tr)
    return out


def _idle(tr) -> bool:
    """The part emits nothing: constant lets only (warps that only park, e.g. the
    setmaxnreg of `if warp in (14, 15)`, which LLVM may test in two branches)."""
    return all(s["op"] == "let" and s["expr"]["e"] == "const" for s in tr.stmts)


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
        if s["op"] == "event" and s["ev"] in ("tma_load", "tma_store", "tma_reduce"):
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
    tensor it resolves to exactly in every record where it is not null (a zero-length
    tensor's; _pointer_name). Pointers that alias one host tensor passed under several jit names
    (`leaves`, the jit argument order; same storage in every record) are named in that
    order, the k-th such field the k-th name, as pass-through arguments are; any other
    ambiguity leaves the field unnamed (a load of it is Data). A tensormap parameter has
    a "base": TTGIR its own name (what a store through it writes), CuTe the tensor its
    address resolves to, and the "aliases" of a base passed under several jit names
    (tma.desc_base); Triton's scratch pointer is "scratch"."""
    out, taken, taken_desc = [], {}, {}
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
        entry = {
            "ptx": ptx,
            "kind": kind,
            "name": name,
            "size": p["size"],
            "fields": fields,
        }
        if kind == "tensormap":
            entry["base"], entry["aliases"] = (name, []) if family == "ttgir" else (
                tma.desc_base(records, k, leaves or [], taken_desc))  # fmt: skip
        if p.get("scratch"):
            entry["scratch"] = True
        out.append(entry)
    return out


def _pointer_name(records, k, off, leaves, taken):
    """The tensor a struct's pointer field names (param_table): the same exact address
    candidates in every record whose pointer is not null (a zero-length tensor's null
    pointer resolves nowhere). A record where it points inside a tensor, or nowhere
    named, leaves the field unnamed: named after the other records' tensor, its accesses
    there would be checked at the wrong offsets."""
    sets = set()
    for r in records:
        rf = next(x for x in r["params"][k]["fields"] if x["off"] == off)
        if rf.get("value") == 0:
            continue
        exact = [c for c in rf.get("tensor") or () if c["match"] == "exact"]
        if not exact or len({tuple(c["storage"]) for c in exact}) != 1:
            return None  # no tensor starts there, or distinct storages: not an alias
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
# global stores, atomics, bulk copies and reductions to global memory. A TMA store or
# reduce writes its descriptor's base (tma.py): a descriptor parameter's param_table
# "base"; an in-kernel descriptor (its address derives from the scratch pointer) any
# tensor a tensormap.replace.global_address may name. tensormap.cp_fenceproxy writes the
# scratch, which is no host tensor.
_WRITES = ("st.global", "atom.global.", "red.global.", "cp.async.bulk.global.shared",
           "cp.reduce.async.bulk.global.shared")  # fmt: skip
_UNKNOWN = "?"  # a base no named pointer accounts for
_SHARED = "%shared"
_DESC = "%desc:"  # the address of a descriptor parameter (+ its PTX name)


def _tma_write(mn: str) -> bool:
    return mn.startswith(("cp.async.bulk.tensor", "cp.reduce.async.bulk.tensor")) and (
        ".global.shared" in mn
    )


def writes(ptx_text: str, entry: str, params: list) -> list | None:
    """The tensors a kernel may write, for the stale-data checks of a variant the
    translation refused (design §5.1: the write set from the PTX, else every pointer
    argument): the named pointers (param_table's `params`) the address of each global
    write derives from (_Bases; through a descriptor: its base). None when some write's
    address has an unknown or no base, or an instruction has no op-table row or is
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
    table = {p["ptx"]: p for p in params}
    bases = _Bases(fn, table, shared)
    scratch = {p["name"] for p in params if p.get("scratch")}
    replaced = set().union(*(bases.of(i.operands[1]) for i, mn in bases.instrs
                             if mn.startswith("tensormap.replace.global_address")))  # fmt: skip
    out: set = set()
    for i, mn in bases.instrs:
        if mn.startswith("tensormap.cp_fenceproxy") or _tma_write(mn):
            got = _through_descriptor(
                bases.of(i.operands[0]), mn, table, scratch, replaced
            )
            if got is None:
                return None
            out |= got
            continue
        if not (mn == "st" or mn.startswith(_WRITES)):
            continue
        got = bases.of(next(op for op in i.operands if op[0] == "mem"))
        if mn == "st" and got == {_SHARED}:
            continue  # a generic store into shared memory
        if not got - {_SHARED} or _UNKNOWN in got:
            return None
        out |= got - {_SHARED}
    return sorted(out - scratch)


def _through_descriptor(got: set, mn: str, table: dict, scratch: set, replaced: set):
    """What a cp_fenceproxy (to the scratch: nothing) or a TMA store or reduce (through a
    descriptor parameter: its base; through the scratch: any replaced global address)
    whose address has the bases `got` writes; None if unknown."""
    if not got:
        return None
    if mn.startswith("tensormap.cp_fenceproxy"):
        return set() if got <= scratch else None
    out = set()
    for b in got:
        base = table[b[len(_DESC) :]].get("base") if b.startswith(_DESC) else None
        if base is None and not (b in scratch and replaced):
            return None
        out |= {base} if base is not None else replaced
    return None if {_UNKNOWN, _SHARED} & out else out


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
            if self.table.get(op[1], {}).get("kind") == "tensormap":
                return {_DESC + op[1]}
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


def _cubin(dump: dict, label: str) -> str | None:
    """The cubin path of a CuTe variant (label: <class>__<digest>)."""
    digest = label.rsplit("__", 1)[-1]
    c = next(c for c in dump["compiles"] if c["digest"] == digest)
    return c.get("ptx", {}).get("cubin")


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
    for name, d in facts.descs.items():  # a bulk descriptor spans its base's storage
        s = shapes.get(d["base"], {"shape": [1], "strides": [1]})
        span = [
            {
                "e": "const",
                "w": 64,
                "v": 1 + sum((n - 1) * st for n, st in zip(s["shape"], s["strides"])),
            }
        ]
        tensors.append({"name": name, "kind": "desc", "elem_bytes": d["elem_bytes"], "box": d["box"],
                        "shape": d["shape"] or span, "strides": d["strides"] or [{"e": "const", "w": 64, "v": 1}],
                        "segments": None, "role": "output" if name in facts.written else "input",
                        "metadata": False})  # fmt: skip
    values = dict(facts.values[0]) if facts.values else {}
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
        cubin = family == "cute" and _cubin(dump, label)
        pol = policy(family, src_root=root, params=params,
                     ptxas=tma.tkinfo_version(Path(cubin).read_bytes()) if cubin else None,
                     tensors={n: {"elem_bytes": b, **shapes.get(n, {})} for n, b in tensors.items()},
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
