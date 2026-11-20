# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""KIR JSON -> Lean file checked against lean/KV.

    python3 -I emit_lean.py KERNEL.kir.json [-o OUT.lean]
    scripts/run_lean_check.sh OUT.lean

The KIR is validated with kir.py first (invalid KIR: errors on stderr, exit 1). The Lean
file imports KV, lifts the heartbeat limit (a KIR's definitions are data whose elaboration
grows with their size, e.g. the 64 MB attention-core KIRs), defines `kernel` (one constructor
per KIR node, see lean/KV/IR.lean; an Expr or TExpr whose text is longer than SHARE_MIN
characters is a definition `sx<n>` of its own, named wherever it recurs) and one `KV.Input`
per case input (each with the case's `elision_flags`), then per input one
`#lkv_check kernel <input> <spec>...` (lean/KV/Command.lean;
the five P0 specs, and P0-7 `gwait` for a KIR with sync tensors, KIR change 18):
one evaluation of the kernel on the input prints the report (verdicts, counterexamples and
notes with IR loc; lean/KV/Check.lean, Report.render) and proves, per spec, the theorem
`KV.Holds .<spec> kernel <input>` or its negation, then prints `THEOREM <input> <spec> PASS|FAIL`.
A P1 KIR (top-level `tmem` or `p1`, file suffix .p1.json; references/kir.md changes 10
and 15) emits `tmem := some {...}` and `cta_threads := some n` in the
kernel and `#lkv_check_p1` with its P1 specs (`tmemReuse`, `storeReuse`) instead; the output
for a P0 KIR is unchanged; a P0-6 KIR (top-level `nbar`, file suffix .nb.json; change 14)
`#lkv_check_nb` with P0-6 (`namedBarriers`). --legacy-theorems writes the earlier form
(`#eval KV.runCase` / `KV.runP1` / `KV.runNb` per input, then one theorem per (spec, input)
closed by `native_decide`: one evaluation of the kernel per spec), for comparisons.
--selfcheck writes `#eval KV.SelfCheck.run kernel <input>` per input instead (the
evaluator's shortcuts against what they replace, lean/KV/SelfCheck.lean), --profile LOG[:N]
`#eval KV.Profile.run ...` (the phases of one evaluation timed into LOG, lean/KV/Profile.lean);
neither proves a verdict. --with-selfcheck writes the self-checks after the checks, in the same
file (tests/lean/run_all.sh: one Lean run per fixture).
"""

import argparse
import json
import os
import re
import sys
import threading

sys.dont_write_bytecode = True  # keep the source tree free of __pycache__
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kir  # noqa: E402

# Lean elaborates long list literals recursively (a few hundred elements exhaust the
# default limits), so longer lists are split into chunk definitions joined by List.flatten.
CHUNK = 32
KEYWORD_CTORS = {"let", "for", "while", "if"}


def lstr(s):
    """Lean string literal."""
    out = []
    for c in s:
        if c in '"\\':
            out.append("\\" + c)
        elif c == "\n":
            out.append("\\n")
        elif c == "\t":
            out.append("\\t")
        elif ord(c) < 0x20:
            out.append(f"\\x{ord(c):02x}")
        else:
            out.append(c)
    return '"' + "".join(out) + '"'


def lint(v):
    return f"({v})" if v < 0 else str(v)


def lbool(b):
    return "true" if b else "false"


def llist(xs):
    return "[" + ", ".join(xs) + "]"


def ctor(name, *args):
    name = f"«{name}»" if name in KEYWORD_CTORS else name
    return "(." + " ".join([name, *args]) + ")"


# Shared terms: an Expr or TExpr whose Lean text is longer than SHARE_MIN characters becomes a
# definition of its own (`sx<n>`), one per distinct text of a kernel's role bodies, emitted before
# its first use: KIRs repeat large subtrees (a lane's select chain in many events), which the
# text would otherwise copy at every use (a DAG of tens of thousands of distinct nodes printed as
# millions); the values, and so every verdict, are the same.
SHARE_MIN = 256


class Shared:
    """The shared-term definitions of one kernel's role bodies (emit)."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.names: dict = {}

    def use(self, text: str, typ: str) -> str:
        if len(text) <= SHARE_MIN:
            return text
        name = self.names.get((typ, text))
        if name is None:
            name = self.names[(typ, text)] = f"sx{len(self.names)}"
            self.chunks.pending.append(f"def {name} : {typ} := {text}\n")
        return name


_SHARED = threading.local()  # .stack: the Shared of the role bodies this thread emits


def _shared(text: str, typ: str) -> str:
    stack = getattr(_SHARED, "stack", None)
    return stack[-1].use(text, typ) if stack else text


def expr(e):
    return _shared(_expr(e), "KV.Expr")


def texpr(t):
    return _shared(_texpr(t), "KV.TExpr")


def _expr(e):
    k, w = e["e"], str(e["w"])
    if k == "const":
        return ctor("const", lint(e["v"]), w)
    if k == "var":
        return ctor("var", lstr(e["name"]), w)
    if k in ("pid", "nprog"):
        return ctor(k, str(e["axis"]), w)
    if k == "bin":
        return ctor(
            "bin",
            "." + e["o"],
            expr(e["a"]),
            expr(e["b"]),
            w,
            lbool(e.get("addr", False)),
        )
    if k == "cmp":
        return ctor("cmp", "." + e["o"], expr(e["a"]), expr(e["b"]), w)
    if k == "select":
        return ctor("select", expr(e["c"]), expr(e["a"]), expr(e["b"]), w)
    if k == "ext":
        return ctor("ext", lbool(e["signed"]), expr(e["a"]), w)
    if k == "trunc":
        return ctor("trunc", expr(e["a"]), w, lbool(e.get("addr", False)))
    if k == "load":
        return ctor("load", lstr(e["tensor"]), llist(map(expr, e["index"])), w)
    raise AssertionError(k)  # unreachable after kir.validate


def _texpr(t):
    k, shape, w = t["t"], llist(map(str, t["shape"])), str(t["w"])
    if k == "range":
        return ctor("range", lint(t["start"]), lint(t["end"]), shape, w)
    if k == "splat":
        return ctor("splat", expr(t["a"]), shape, w)
    if k in ("bcast", "expand"):
        return ctor(k, texpr(t["a"]), str(t["axis"]), shape, w)
    if k == "bin":
        return ctor(
            "bin",
            "." + t["o"],
            texpr(t["a"]),
            texpr(t["b"]),
            shape,
            w,
            lbool(t.get("addr", False)),
        )
    if k == "cmp":
        return ctor("cmp", "." + t["o"], texpr(t["a"]), texpr(t["b"]), shape, w)
    if k == "select":
        return ctor("select", texpr(t["c"]), texpr(t["a"]), texpr(t["b"]), shape, w)
    if k == "full":
        return ctor("full", lint(t["v"]), shape, w)
    if k == "ext":
        return ctor("ext", lbool(t["signed"]), texpr(t["a"]), shape, w)
    if k == "trunc":
        return ctor("trunc", texpr(t["a"]), shape, w, lbool(t.get("addr", False)))
    raise AssertionError(k)


def opt_expr(e):
    """`none` or `(some <Expr>)` for an optional Expr field."""
    return "none" if e is None else f"(some {expr(e)})"


def event(s):
    ev, loc = s["ev"], lstr(s["loc"])
    if ev == "bar_init":
        return ctor(ev, lstr(s["bar"]), expr(s["idx"]), expr(s["count"]), loc)
    if ev == "arrive":
        return ctor(
            ev,
            lstr(s["bar"]),
            expr(s["idx"]),
            str(s.get("n", 1)),
            lbool(s.get("per_warp", False)),
            opt_expr(s.get("cta")),
            loc,
        )
    if ev == "commit_arrive":
        return ctor(
            ev, lstr(s["bar"]), expr(s["idx"]), opt_expr(s.get("cta_mask")), loc
        )
    if ev in ("expect_tx", "complete_tx"):
        return ctor(
            ev,
            lstr(s["bar"]),
            expr(s["idx"]),
            expr(s["bytes"]),
            opt_expr(s.get("cta")),
            loc,
        )
    if ev == "wait":
        return ctor(
            ev,
            lstr(s["bar"]),
            expr(s["idx"]),
            expr(s["parity"]),
            opt_expr(s.get("cta")),
            loc,
        )
    if ev == "tma_load":
        return ctor(
            ev,
            lstr(s["desc"]),
            llist(map(expr, s["coords"])),
            lstr(s["bar"]),
            expr(s["idx"]),
            opt_expr(s.get("cta_mask")),
            lbool(s.get("leader", False)),
            loc,
        )
    if ev in ("tma_store", "tma_reduce"):
        return ctor(ev, lstr(s["desc"]), llist(map(expr, s["coords"])), loc)
    if ev in ("gload", "gstore", "gatomic"):
        mask = "none" if s["mask"] is None else f"(some {texpr(s['mask'])})"
        return ctor(ev, lstr(s["tensor"]), texpr(s["offsets"]), mask, loc)
    if ev == "work":
        return ctor(
            ev, llist(f"({lstr(k)}, {expr(v)})" for k, v in s["fields"].items()), loc
        )
    if ev == "tmem":
        rects = [
            f"({{ use := .{r['use']}, write := {lbool(r['write'])}, lane := {expr(r['lane'])}, "
            f"nlanes := {r['nlanes']}, col := {expr(r['col'])}, ncols := {r['ncols']} }} : KV.TmemRect)"
            for r in s["rects"]
        ]
        return ctor(
            ev,
            "." + s["inst"],
            expr(s["warp"]),
            llist(rects),
            opt_expr(s.get("cta_mask")),
            lstr(s.get("site", "")),
            loc,
        )
    if ev == "tmem_wait":
        return ctor(ev, lbool(s["ld"]), loc)
    if ev == "named_bar":
        return ctor(ev, expr(s["id"]), expr(s["count"]), lbool(s["sync"]),
                    lbool(s.get("all", False)), loc)  # fmt: skip
    if ev == "nb_open":  # its kind, id, count and sync are for audit (IR.lean)
        return ctor(ev, lbool(s["certain"]), lstr(s["why"]), loc)
    if ev == "nb_rep":  # its why is for audit (IR.lean)
        return ctor(ev, lbool(s["begin"]), lstr(s.get("key", "")), loc)
    if ev == "bulk_store":
        t = s.get("tmap")
        tmap = "none" if t is None else f"(some ({lstr(t['base'])}, {expr(t['addr'])}))"
        return ctor(ev, "." + s["kind"], expr(s["warp"]), opt_expr(s.get("lane")),
                    lstr(s["sym"]), expr(s["src"]), str(s["bytes"]), tmap, loc)  # fmt: skip
    if ev == "bulk_commit":
        return ctor(ev, loc)
    if ev == "bulk_wait":  # P1-2 takes `read` and plain waits alike (IR.lean)
        return ctor(ev, str(s["n"]), loc)
    if ev == "smem_write":  # its mask is null (kir.py)
        slot = (
            "none"
            if "bar" not in s
            else f"(some ({lstr(s['bar'])}, {expr(s['idx'])}, {lbool(s.get('leader', False))}))"
        )
        return ctor(ev, lstr(s["sym"]), texpr(s["offsets"]), str(s["bytes"]),
                    expr(s["warp"]), opt_expr(s.get("lane")), lbool(s.get("sync", False)),
                    lbool(s.get("async", False)), opt_expr(s.get("cta")), slot, loc)  # fmt: skip
    if ev == "tmap_write":
        return ctor(ev, lstr(s["base"]), expr(s["addr"]), expr(s["warp"]), loc)
    if ev == "handoff_put":
        return ctor(ev, lstr(s["chan"]), expr(s["value"]), loc)
    if ev == "handoff_get":
        return ctor(ev, lstr(s["chan"]), lstr(s["var"]), loc)
    if ev == "gsig":
        return ctor(ev, lstr(s["tensor"]), expr(s["index"]), expr(s["value"]),
                    lbool(s.get("data_lo", False)), loc)  # fmt: skip
    if ev == "gwait":
        var = f"(some ({lstr(s['var'])}, {s['w']}))" if "var" in s else "none"
        return ctor(
            ev, lstr(s["tensor"]), expr(s["index"]), expr(s["target"]), var, loc
        )
    if ev == "gclaim":
        return ctor(ev, lstr(s["tensor"]), expr(s["index"]), expr(s["value"]),
                    lstr(s["var"]), str(s["w"]), loc)  # fmt: skip
    if ev == "rput":
        return ctor(
            ev, lstr(s["chan"]), expr(s["ord"]), str(s["field"]), expr(s["value"]), loc
        )
    if ev == "rget":
        return ctor(ev, lstr(s["chan"]), expr(s["ord"]), str(s["field"]), lstr(s["var"]),
                    str(s["w"]), loc)  # fmt: skip
    raise AssertionError(ev)


class Chunks:
    """Chunk definitions; `take()` returns the pending ones, to be emitted before their use."""

    def __init__(self):
        self.count = 0
        self.pending = []

    def take(self):
        defs, self.pending = self.pending, []
        return defs

    def list(self, items, elem_type, indent=None):
        """Lean list of `items` (terms of type elem_type), one per line if `indent` is given."""
        if len(items) > CHUNK:
            names = []
            for j in range(0, len(items), CHUNK):
                names.append(f"chunk{self.count}")
                self.count += 1
                body = self.list(
                    items[j : j + CHUNK], elem_type, None if indent is None else 0
                )
                self.pending.append(f"def {names[-1]} : List ({elem_type}) := {body}\n")
            return f"(List.flatten {self.list(names, f'List ({elem_type})')})"
        if indent is None or not items:
            return llist(items)
        pad = " " * (indent + 2)
        return "[\n" + ",\n".join(pad + x for x in items) + "]"


def block(stmts, indent, chunks):
    return chunks.list([stmt(s, indent + 2, chunks) for s in stmts], "KV.Stmt", indent)


def stmt(s, indent, chunks):
    op, loc = s["op"], lstr(s.get("loc", ""))
    if op == "event":
        return ctor("event", event(s))
    if op == "let":
        return ctor("let", lstr(s["var"]), expr(s["expr"]), loc)
    if op == "for":
        body = block(s["body"], indent, chunks)
        return ctor(
            "for",
            lstr(s["var"]),
            expr(s["lb"]),
            expr(s["ub"]),
            expr(s["step"]),
            body,
            loc,
        )
    if op == "while":
        return ctor(
            "while",
            expr(s["cond"]),
            block(s["body"], indent, chunks),
            str(s["max_iters"]),
            loc,
        )
    if op == "if":
        return ctor(
            "if",
            expr(s["cond"]),
            block(s["then"], indent, chunks),
            block(s["else"], indent, chunks),
            loc,
        )
    if op in ("assert", "assume"):
        return ctor(op, expr(s["cond"]), lstr(s["msg"]), loc)
    raise AssertionError(op)


def tensor(t):
    seg = t["segments"]
    seg_l = (
        "none"
        if seg is None
        else (
            f"(some {{ dim := {seg['dim']}, offsets := {lstr(seg['offsets'])}, per := {lstr(seg['per'])} }})"
        )
    )
    return (
        f"{{ name := {lstr(t['name'])}, kind := .{t['kind']}, elem_bytes := {t['elem_bytes']},\n"
        f"      shape := {llist(map(expr, t['shape']))},\n"
        f"      strides := {llist(map(expr, t['strides']))},\n"
        f"      segments := {seg_l}, role := .{t['role']}, metadata := {lbool(t['metadata'])},\n"
        f"      box := {llist(map(str, t.get('box', [])))}"
        + (f", sync := some {t['sync']['width']}" if "sync" in t else "")
        + " }"
    )


def intended(v, chunks):
    if isinstance(v, str):
        return "." + v
    return f"(.rows {chunks.list([f'({lo}, {hi})' for lo, hi in v], 'Nat × Nat')})"


def ident(s):
    """`s` with every character outside [A-Za-z0-9_] replaced by '_'."""
    return re.sub(r"\W", "_", s, flags=re.ASCII)


def roles(k, chunks):
    """The role definitions of kernel `k`, each after the chunk and shared-term definitions it
    uses (one Shared for all its role bodies), and their names."""
    out, names = [], []
    stack = _SHARED.__dict__.setdefault("stack", [])
    stack.append(Shared(chunks))
    try:
        for i, r in enumerate(k["roles"]):
            name = f"role{i}_{ident(r['name'])}"
            names.append(name)
            body = block(r["body"], 2, chunks)
            out += chunks.take() + [
                f"def {name} : KV.Role where",
                f"  name := {lstr(r['name'])}",
                f"  warps := {r['warps']}",
                f"  threads_per_issue := {r['threads_per_issue']}",
                f"  body := {body}",
                *([f"  group := {lstr(r['group'])}"] if "group" in r else []),
                "",
            ]
    finally:
        stack.pop()
    return out, names


def debug_evals(inputs, kind, specs, profile, selfcheck):
    """The per-input `#eval`s that replace the verdicts under --profile (lean/KV/Profile.lean;
    `profile` = (log path, CTA limit), a negative limit: every issue and note of every spec,
    `dumpIssues`) or --selfcheck (lean/KV/SelfCheck.lean); None for neither. `kind` is
    checkAll's report kind (lean/KV/Command.lean: 0 P0, 1 P1, 2 P0-6), `specs` the Lean list
    of the KIR's specs."""
    if profile is not None:
        log, limit = profile
        if limit < 0:
            return [
                f"#eval KV.Profile.dumpIssues kernel {name} {lstr(log)} {specs}"
                for name in inputs
            ]
        return [
            f"#eval KV.Profile.run kernel {name} {lstr(log)} {limit} {kind} {specs}"
            for name in inputs
        ]
    if selfcheck:
        return [f"#eval KV.SelfCheck.run kernel {name} {specs}" for name in inputs]
    return None


def emit(
    k, src_name, legacy=False, profile=None, selfcheck=False, with_selfcheck=False
):
    # IR.lean's `Ir` constructor of each kir.IRS value (`cute-mlir` needs «»).
    ir = {"ttgir": ".ttgir", "cute-mlir": ".«cute-mlir»", "ptx": ".ptx"}[k["ir"]]
    out = [
        f"-- Generated by frontend/emit_lean.py from {src_name}; do not edit.",
        "import KV"
        + (
            "\nimport KV.Profile"
            if profile is not None
            else "\nimport KV.SelfCheck"
            if selfcheck or with_selfcheck
            else ""
        ),
        "",
        "set_option maxHeartbeats 0",
        "",
        f"namespace KIR.«{ident(k['kernel'])}»",
        "",
    ]
    chunks = Chunks()
    defs, role_defs = roles(k, chunks)
    out += defs
    out += [
        "def kernel : KV.Kernel where",
        f"  kir_version := {k['kir_version']}",
        f"  kernel := {lstr(k['kernel'])}",
        f"  source := {lstr(k['source'])}",
        f"  ir := {ir}",
        f"  grid := {llist(map(expr, k['grid']))}",
        "  params := "
        + llist(
            f"{{ name := {lstr(p['name'])}, type := .{p['type']} }}"
            for p in k["params"]
        ),
        "  tensors := [\n    " + ",\n    ".join(map(tensor, k["tensors"])) + "]",
        "  barriers := "
        + llist(
            f"{{ name := {lstr(b['name'])}, count := {expr(b['count'])}, num := {expr(b['num'])} }}"
            for b in k["barriers"]
        ),
        f"  roles := {llist(role_defs)}",
    ]
    if k.get("cluster", [1, 1, 1]) != [1, 1, 1]:
        out.append(f"  cluster := {llist(map(str, k['cluster']))}")
    p1 = "tmem" in k or "p1" in k
    if "tmem" in k:
        t = k["tmem"]
        out.append(
            f"  tmem := some {{ cols := {t['cols']}, zeroBase := {lbool(t['base'] == 'zero')} }}"
        )
    if "cta_threads" in k:
        out.append(f"  cta_threads := some {k['cta_threads']}")
    out.append("")
    inputs = []
    elision = llist(map(lstr, k["case"].get("elision_flags", [])))
    for i, inp in enumerate(k["case"]["inputs"]):
        name = f"input{i}_{ident(inp['name'])}"
        inputs.append(name)
        grid = f"(some {llist(map(str, inp['grid']))})" if "grid" in inp else "none"
        meta = llist(
            f"({lstr(n)}, {chunks.list([lint(x) for x in v], 'Int')}.toArray)"
            for n, v in inp.get("metadata", {}).items()
        )
        iw = llist(
            f"({lstr(n)}, {intended(v, chunks)})"
            for n, v in inp.get("intended_writes", {}).items()
        )
        out += chunks.take() + [
            f"def {name} : KV.Input where",
            f"  name := {lstr(inp['name'])}",
            "  params := "
            + llist(f"({lstr(n)}, {lint(v)})" for n, v in inp["params"].items()),
            f"  grid := {grid}",
            f"  metadata := {meta}",
            "  flags := "
            + llist(
                f"({lstr(n)}, {lbool(v)})" for n, v in inp.get("flags", {}).items()
            ),
            "  shapes := "
            + llist(
                f"({lstr(n)}, {llist(map(str, v))})"
                for n, v in inp.get("shapes", {}).items()
            ),
            f"  intended_writes := {iw}",
            f"  elision_flags := {elision}",
        ]
        if "capacity" in inp:  # KIR change 18
            c = inp["capacity"]
            cl = f"some {c['clusters']}" if "clusters" in c else "none"
            out.append(f"  capacity := some ({c['ctas']}, {cl})")
        if inp.get("sync_groups"):  # KIR change 18
            out.append(
                "  syncGroups := "
                + llist(llist(map(str, g)) for g in inp["sync_groups"])
            )
        out.append("")
    nb = "nbar" in k
    # (Lean name, label) of the KIR's specs (kir.labels), in report order
    specs = [
        (spec, label)
        for label, (spec, _) in kir.SPECS.items()
        if label in kir.labels(k)
    ]
    kind = 1 if p1 else 2 if nb else 0
    spec_list = llist(f".{spec}" for spec, _ in specs)
    debug = debug_evals(inputs, kind, spec_list, profile, selfcheck)
    if debug is not None:
        out += debug
    elif legacy:
        run = "KV.runP1" if p1 else "KV.runNb" if nb else "KV.runCase"
        extra = "" if nb else f" {spec_list}"
        out += [f"#eval {run} kernel {name}{extra}" for name in inputs]
        out.append("")
        for name in inputs:
            for spec, label in specs:
                out.append(
                    f"theorem {label.replace('-', '_')}_{spec}__{name} : KV.Holds .{spec} kernel {name} := by native_decide"
                )
    else:
        cmd = "#lkv_check_p1" if p1 else "#lkv_check_nb" if nb else "#lkv_check"
        names = " ".join(spec for spec, _ in specs)
        out += [f"{cmd} kernel {name} {names}" for name in inputs]
        if with_selfcheck:
            out += debug_evals(inputs, kind, spec_list, None, True)
    out += ["", f"end KIR.«{ident(k['kernel'])}»", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("kir", help="KIR JSON file")
    ap.add_argument("-o", "--output", help="Lean file to write (default: stdout)")
    ap.add_argument(
        "--legacy-theorems",
        action="store_true",
        help="one native_decide theorem per (spec, input), as before #lkv_check",
    )
    ap.add_argument(
        "--profile",
        metavar="LOG[:N]",
        help="time one evaluation per input instead of checking it (lean/KV/Profile.lean), "
        "appending to LOG; N > 0 evaluates only the first N CTAs; N = -1 appends every issue "
        "and note of every spec instead",
    )
    ap.add_argument(
        "--selfcheck",
        action="store_true",
        help="compare the evaluator's shortcuts with what they replace on every input "
        "instead of checking it (lean/KV/SelfCheck.lean)",
    )
    ap.add_argument(
        "--with-selfcheck",
        action="store_true",
        help="the self-checks of --selfcheck after the checks, in the same file",
    )
    args = ap.parse_args()
    profile = None
    if args.profile:
        log, _, n = args.profile.partition(":")
        profile = (os.path.abspath(log), int(n or 0))
    with open(args.kir) as f:
        k = json.load(f)
    errors = kir.validate(k)
    if errors:
        for e in errors:
            print(f"{args.kir}: {e}", file=sys.stderr)
        return 1
    text = emit(
        k,
        os.path.basename(args.kir),
        args.legacy_theorems,
        profile,
        args.selfcheck,
        args.with_selfcheck,
    )
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
