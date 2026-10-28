# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""KIR JSON -> Lean file checked against lean/KV.

    python3 -I emit_lean.py KERNEL.kir.json [-o OUT.lean]
    scripts/run_lean_check.sh OUT.lean

The KIR is validated with kir.py first (invalid KIR: errors on stderr, exit 1). The Lean
file imports KV, defines `kernel` (one constructor per KIR node, see lean/KV/IR.lean) and
one `KV.Input` per case input (each with the case's `elision_flags`), then
  (a) `#eval KV.runCase kernel <input>` for every input: verdicts, counterexamples and
      notes with IR loc, printed before any theorem (lean/KV/Check.lean, Report.render);
  (b) one theorem `KV.Holds .<spec> kernel <input>` per (spec, input), closed by
      `native_decide`, so a failing spec fails the check.
A P1 KIR (top-level `tmem`, file suffix .p1.json; references/kir.md change 10) emits
`tmem := some {...}` in the kernel, `#eval KV.runP1 kernel <input>` and the P1 theorems
(`KV.Holds .tmemReuse`) instead; the output for a P0 KIR is unchanged.
"""

import argparse
import json
import os
import re
import sys

sys.dont_write_bytecode = True  # keep the source tree free of __pycache__
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kir  # noqa: E402

# Lean elaborates long list literals recursively (a few hundred elements exhaust the
# default limits), so longer lists are split into chunk definitions joined by List.flatten.
CHUNK = 32
SPECS = (
    ("bounds", "P0_1"),
    ("tripCount", "P0_2"),
    ("coverage", "P0_3"),
    ("barrier", "P0_4"),
    ("maskElision", "P0_5"),
)
SPECS_P1 = (("tmemReuse", "P1_1"),)
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


def expr(e):
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


def texpr(t):
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
        return ctor(ev, expr(s["id"]), expr(s["count"]), lbool(s["sync"]), loc)
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
    if op == "assert":
        return ctor("assert", expr(s["cond"]), lstr(s["msg"]), loc)
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
        f"      box := {llist(map(str, t.get('box', [])))} }}"
    )


def intended(v, chunks):
    if isinstance(v, str):
        return "." + v
    return f"(.rows {chunks.list([f'({lo}, {hi})' for lo, hi in v], 'Nat × Nat')})"


def ident(s):
    """`s` with every character outside [A-Za-z0-9_] replaced by '_'."""
    return re.sub(r"\W", "_", s, flags=re.ASCII)


def emit(k, src_name):
    ir = ".ttgir" if k["ir"] == "ttgir" else ".«cute-mlir»"
    out = [
        f"-- Generated by frontend/emit_lean.py from {src_name}; do not edit.",
        "import KV",
        "",
        f"namespace KIR.«{ident(k['kernel'])}»",
        "",
    ]
    chunks = Chunks()
    role_defs = []
    for i, r in enumerate(k["roles"]):
        name = f"role{i}_{ident(r['name'])}"
        role_defs.append(name)
        body = block(r["body"], 2, chunks)
        out += chunks.take() + [
            f"def {name} : KV.Role where",
            f"  name := {lstr(r['name'])}",
            f"  warps := {r['warps']}",
            f"  threads_per_issue := {r['threads_per_issue']}",
            f"  body := {body}",
            "",
        ]
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
    p1 = "tmem" in k
    if p1:
        t = k["tmem"]
        out.append(
            f"  tmem := some {{ cols := {t['cols']}, zeroBase := {lbool(t['base'] == 'zero')} }}"
        )
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
            "",
        ]
    run = "KV.runP1" if p1 else "KV.runCase"
    out += [f"#eval {run} kernel {name}" for name in inputs]
    out.append("")
    for name in inputs:
        for spec, label in SPECS_P1 if p1 else SPECS:
            out.append(
                f"theorem {label}_{spec}__{name} : KV.Holds .{spec} kernel {name} := by native_decide"
            )
    out += ["", f"end KIR.«{ident(k['kernel'])}»", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("kir", help="KIR JSON file")
    ap.add_argument("-o", "--output", help="Lean file to write (default: stdout)")
    args = ap.parse_args()
    with open(args.kir) as f:
        k = json.load(f)
    errors = kir.validate(k)
    if errors:
        for e in errors:
            print(f"{args.kir}: {e}", file=sys.stderr)
        return 1
    text = emit(k, os.path.basename(args.kir))
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
