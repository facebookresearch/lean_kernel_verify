# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""KIR validator: enforces references/kir.md on a parsed JSON object.

    validate(obj) -> list[str]       # empty iff obj is valid KIR; dependency-free

CLI: python3 -I kir.py FILE.json...  (prints errors, exit status 1 if any file is invalid)

Rules of kir.md beyond its field tables, which lean/KV relies on:

* An `event` statement is the Event object itself, inline:
  {"op": "event", "ev": "wait", "bar": ..., "idx": ..., "parity": ..., "loc": ...}.
* Every TExpr node carries "w" like an Expr node (`range` is i32 in TTGIR, `cmp` is 1).
  TExpr `ext` / `trunc` are the elementwise scalar casts (same shape as the operand).
* Integer fields that may depend on runtime values are Exprs: grid entries, tensor
  shape/strides, barrier count/num, event idx/count/bytes/parity/coords, loop bounds,
  conditions, work field values. Static attributes are JSON ints: warps,
  threads_per_issue, elem_bytes, box, arrive n, max_iters, axis, range start/end,
  const/full v, segments dim.
* Defaults: arrive "n" = 1, arrive "per_warp" = false, Expr/TExpr "addr" = false
  ("addr" only has an effect on bin add/sub/mul/shl and on trunc). "loc" is required on
  events and optional on the other statements.
* `expect_tx` only raises the expected transaction bytes; it is not an arrival. An
  `mbarrier.arrive.expect_tx` is emitted as `expect_tx` followed by `arrive`, in the role
  that arrives for that phase.
* Host-side Exprs (grid, tensor shape/strides, barrier count/num) may use only params:
  no pid, nprog, load or loop variables.
* Variables: one mutable environment per role. `let` assigns: it creates a variable or
  overwrites a visible one of the same width (loop-carried values, scf.if results). A
  variable is readable after a `let` in its block or an enclosing block (a name first
  assigned inside a branch or loop body is not readable after it); params and `for`
  variables are read-only.
* `coords`, `box` and `load` indices follow `shape` order (dim 0 first). A metadata
  tensor's contents are a flat buffer: element (i0, i1, ...) is
  contents[i0*strides[0] + i1*strides[1] + ...].
* Segments: "offsets" names a 1-D metadata tensor (segment z is rows
  [offsets[z], offsets[z+1]) of dim "dim"), "per" names the `work` field holding z; some
  `work` event must define that field.
* Case: "name" and a non-empty "inputs" list are required; "elision_flags" (optional) lists
  distinct flag names that select a mask-elided fast path (P0-5 checks P0-1 and P0-3 on the
  inputs that set one). An input need not name every elision flag (absent: not set).
* Input: "name" and "params" (every kernel param; signed values, i1 0/1) are required;
  "grid" (3 ints) overrides the grid Exprs, "shapes" overrides tensor shape Exprs,
  "metadata" must give the contents of every metadata tensor, "intended_writes" values
  are "all_rows_of_segments", "all", [[lo, hi], ...] half-open row ranges, or "unchecked"
  (opt-out). An output tensor missing from "intended_writes" is valid KIR but fails P0-3.
* Optional, informational top-level fields (Layer 2 ignores them): "target" (the GPU
  architecture the IR was compiled for) and "compiler" ({"name", "version"} strings).
* A tensor that a gstore, gatomic, tma_store or tma_reduce writes is not "input": an
  output (which P0-3 checks against intended_writes) or workspace.
* P1 KIR (kir.md, change 10; file suffix .p1.json): a top-level "tmem" {"cols", "base"}
  is present iff some `tmem`, `tmem_wait` or `named_bar` event exists (a P0 KIR has
  none of them). cols is 32, 64, 128, 256 or 512; base "zero" (absolute addresses, the
  fake-zero-base idiom) needs cols 512, else "relative". A `tmem` event has a w=32
  `warp` and non-empty `rects` {"use", "write", "lane", "nlanes", "col", "ncols"}
  (lane/col w=32 Exprs; nlanes 16/32/64/128; 1 <= ncols <= 512): ld only data reads,
  st and cp only data writes, mma exactly one d write, at most one a read and zero or
  two sf reads; a role with a cp or mma event has one warp (the landing rule needs the
  issuing thread). `named_bar` id and count are const Exprs, count a positive multiple
  of 32; `tmem_wait` and `named_bar` flags are booleans; "site" is an optional string.
  In a role of several warps every warp issues each mbarrier event (the CuTe P1 pass
  splits roles so); checked here as: an arrive gives at least `warps` arrivals.
* Clusters (kir.md, change 11): an optional top-level "cluster" [x, y, z] of positive ints
  (default [1, 1, 1]). Barrier events may name another CTA of the cluster: "cta" (arrive,
  expect_tx, wait, complete_tx) is a w=32 Expr, its rank in the cluster; "cta_mask"
  (commit_arrive, tma_load, tmem) a w=32 Expr, a set of ranks; tma_load's "leader" (bool,
  default false) completes each destination's bytes on its pair leader's slot. They and the
  `complete_tx` event (bar, idx, bytes) need a cluster of more than one CTA.
"""

import json
import sys

WIDTHS = (1, 32, 64)
INT_TYPES = {"i1": 1, "i32": 32, "i64": 64}
BIN_OPS = (
    "add", "sub", "mul", "sdiv", "udiv", "srem", "urem", "shl", "ashr", "lshr",
    "and", "or", "xor", "smin", "smax", "umin", "umax",
)  # fmt: skip
CMP_OPS = ("eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge")
IRS = ("ttgir", "cute-mlir")
TARGETS = ("sm_100", "sm_103")
TENSOR_KINDS = ("global", "desc")
TENSOR_ROLES = ("input", "output", "workspace")
INTENDED_NAMED = ("all_rows_of_segments", "all", "unchecked")

EXPR_FIELDS = {
    "const": {"v"},
    "var": {"name"},
    "pid": {"axis"},
    "nprog": {"axis"},
    "bin": {"o", "a", "b"},
    "cmp": {"o", "a", "b"},
    "select": {"c", "a", "b"},
    "ext": {"signed", "a"},
    "trunc": {"a"},
    "load": {"tensor", "index"},
}
TEXPR_FIELDS = {
    "range": {"start", "end"},
    "splat": {"a"},
    "bcast": {"a", "axis"},
    "expand": {"a", "axis"},
    "bin": {"o", "a", "b"},
    "cmp": {"o", "a", "b"},
    "select": {"c", "a", "b"},
    "full": {"v"},
    "ext": {"signed", "a"},
    "trunc": {"a"},
}
STMT_FIELDS = {
    "let": {"var", "expr"},
    "for": {"var", "lb", "ub", "step", "body"},
    "while": {"cond", "body", "max_iters"},
    "if": {"cond", "then", "else"},
    "assert": {"cond", "msg"},
}
EVENT_FIELDS = {
    "bar_init": ({"bar", "idx", "count"}, set()),
    "arrive": ({"bar", "idx"}, {"n", "per_warp", "cta"}),
    "commit_arrive": ({"bar", "idx"}, {"cta_mask"}),
    "expect_tx": ({"bar", "idx", "bytes"}, {"cta"}),
    "wait": ({"bar", "idx", "parity"}, {"cta"}),
    "tma_load": ({"desc", "coords", "bar", "idx"}, {"cta_mask", "leader"}),
    "complete_tx": ({"bar", "idx", "bytes", "cta"}, set()),
    "tma_store": ({"desc", "coords"}, set()),
    "tma_reduce": ({"desc", "coords"}, set()),
    "gload": ({"tensor", "offsets", "mask"}, set()),
    "gstore": ({"tensor", "offsets", "mask"}, set()),
    "gatomic": ({"tensor", "offsets", "mask"}, set()),
    "work": ({"fields"}, set()),
    "tmem": ({"inst", "warp", "rects"}, {"site", "cta_mask"}),
    "tmem_wait": ({"ld"}, set()),
    "named_bar": ({"id", "count", "sync"}, set()),
}
P1_EVENTS = ("tmem", "tmem_wait", "named_bar")
TMEM_INSTS = ("ld", "st", "cp", "mma")
RECT_USES = ("data", "d", "a", "sf")
RECT_FIELDS = {"use", "write", "lane", "nlanes", "col", "ncols"}
TMEM_COLS = (32, 64, 128, 256, 512)
WRITE_EVENTS = ("gstore", "gatomic", "tma_store", "tma_reduce")
TOP_FIELDS = {
    "kir_version",
    "kernel",
    "source",
    "ir",
    "grid",
    "params",
    "tensors",
    "barriers",
    "roles",
    "case",
}
TOP_OPTIONAL = {"target", "compiler", "tmem", "cluster"}
CLUSTER_FIELDS = ("cta", "cta_mask")
TENSOR_FIELDS = {
    "name",
    "kind",
    "elem_bytes",
    "shape",
    "strides",
    "segments",
    "role",
    "metadata",
}
INPUT_REQUIRED = {"name", "params"}
INPUT_OPTIONAL = {"grid", "metadata", "flags", "shapes", "intended_writes"}


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def tag(obj, key):
    """obj[key] if obj is an object and that value is a string, else None."""
    v = obj.get(key) if isinstance(obj, dict) else None
    return v if isinstance(v, str) else None


def fits(v, w):
    """Two's-complement literal for width w: signed or unsigned spelling."""
    return -(1 << (w - 1)) <= v < (1 << w)


def fits_signed(v, w):
    """Runtime value of an i1 (0/1) or signed iw kernel argument."""
    return v in (0, 1) if w == 1 else -(1 << (w - 1)) <= v < (1 << (w - 1))


class _Checker:
    def __init__(self, errors):
        self.errors = errors
        self.params = {}  # name -> (width, read-only)
        self.tensors = {}  # name -> tensor object
        self.barriers = set()
        self.work_fields = set()
        self.p1_events = 0  # tmem, tmem_wait and named_bar events seen
        self.role_warps = None  # warps of the role being checked
        self.role_tpi = None  # threads_per_issue of the role being checked
        self.cluster_size = 1  # CTAs per cluster
        self.few_arrivals = []  # (path, arrivals, warps): arrives no P1 role may have

    def err(self, path, msg):
        self.errors.append(f"{path}: {msg}")

    def keys(self, path, obj, required, optional=()):
        if not isinstance(obj, dict):
            self.err(path, f"expected an object, got {type(obj).__name__}")
            return False
        for k in sorted(set(required) - obj.keys()):
            self.err(path, f"missing field '{k}'")
        for k in sorted(obj.keys() - set(required) - set(optional)):
            self.err(path, f"unknown field '{k}'")
        return set(required) <= obj.keys()

    def string(self, path, x):
        if not isinstance(x, str):
            self.err(path, "expected a string")
            return False
        return True

    def int_field(self, path, x, lo=None):
        if not is_int(x):
            self.err(path, "expected an integer")
            return False
        if lo is not None and x < lo:
            self.err(path, f"must be >= {lo}")
            return False
        return True

    def bool_field(self, path, x):
        if not isinstance(x, bool):
            self.err(path, "expected a boolean")
            return False
        return True

    def list_field(self, path, x):
        if not isinstance(x, list):
            self.err(path, "expected a list")
            return False
        return True

    # ---------------------------------------------------------------- expressions

    def expr(self, path, e, scope, device=True):
        """Check a scalar Expr; return its width or None."""
        if not isinstance(e, dict) or tag(e, "e") not in EXPR_FIELDS:
            self.err(path, f"expected an Expr with 'e' in {sorted(EXPR_FIELDS)}")
            return None
        kind = e["e"]
        if not self.keys(path, e, {"e", "w"} | EXPR_FIELDS[kind], {"addr"}):
            return None
        w = e["w"]
        if w not in WIDTHS or isinstance(w, bool):
            self.err(path, f"'w' must be one of {WIDTHS}")
            return None
        if "addr" in e:
            self.bool_field(f"{path}.addr", e["addr"])
        if kind == "const":
            if self.int_field(f"{path}.v", e["v"]) and not fits(e["v"], w):
                self.err(f"{path}.v", f"{e['v']} does not fit in {w} bits")
        elif kind == "var":
            name = e["name"]
            if self.string(f"{path}.name", name):
                if name not in scope:
                    self.err(f"{path}.name", f"unbound variable '{name}'")
                elif scope[name][0] != w:
                    self.err(
                        path,
                        f"variable '{name}' has width {scope[name][0]}, used with w={w}",
                    )
        elif kind in ("pid", "nprog"):
            if not device:
                self.err(path, f"'{kind}' is not allowed in a host-side expression")
            if e["axis"] not in (0, 1, 2) or isinstance(e["axis"], bool):
                self.err(f"{path}.axis", "must be 0, 1 or 2")
        elif kind == "bin":
            if e["o"] not in BIN_OPS:
                self.err(f"{path}.o", f"unknown bin op {e['o']!r}")
            self._same_width(
                path,
                w,
                self.expr(f"{path}.a", e["a"], scope, device),
                self.expr(f"{path}.b", e["b"], scope, device),
            )
        elif kind == "cmp":
            if e["o"] not in CMP_OPS:
                self.err(f"{path}.o", f"unknown cmp op {e['o']!r}")
            if w != 1:
                self.err(path, "cmp result must have w=1")
            wa = self.expr(f"{path}.a", e["a"], scope, device)
            wb = self.expr(f"{path}.b", e["b"], scope, device)
            if wa is not None and wb is not None and wa != wb:
                self.err(path, f"cmp operand widths differ ({wa} vs {wb})")
        elif kind == "select":
            if self.expr(f"{path}.c", e["c"], scope, device) not in (1, None):
                self.err(f"{path}.c", "select condition must have w=1")
            self._same_width(
                path,
                w,
                self.expr(f"{path}.a", e["a"], scope, device),
                self.expr(f"{path}.b", e["b"], scope, device),
            )
        elif kind == "ext":
            self.bool_field(f"{path}.signed", e["signed"])
            wa = self.expr(f"{path}.a", e["a"], scope, device)
            if wa is not None and not wa < w:
                self.err(path, f"ext must widen (operand w={wa}, result w={w})")
        elif kind == "trunc":
            wa = self.expr(f"{path}.a", e["a"], scope, device)
            if wa is not None and not wa > w:
                self.err(path, f"trunc must narrow (operand w={wa}, result w={w})")
        elif kind == "load":
            self._check_load(path, e, scope, device)
        return w

    def _same_width(self, path, w, wa, wb):
        for side, wx in (("a", wa), ("b", wb)):
            if wx is not None and wx != w:
                self.err(path, f"operand {side} has w={wx}, node has w={w}")

    def _check_load(self, path: str, e: dict, scope: dict, device: bool) -> None:
        if not device:
            self.err(path, "'load' is not allowed in a host-side expression")
        t = self.tensors.get(e["tensor"]) if isinstance(e["tensor"], str) else None
        if t is None:
            self.err(f"{path}.tensor", f"unknown tensor {e['tensor']!r}")
        elif t.get("metadata") is not True:
            self.err(f"{path}.tensor", f"'{e['tensor']}' is not a metadata tensor")
        if self.list_field(f"{path}.index", e["index"]):
            if (
                t is not None
                and isinstance(t.get("shape"), list)
                and len(e["index"]) != len(t["shape"])
            ):
                self.err(
                    f"{path}.index",
                    f"rank {len(e['index'])} != tensor rank {len(t['shape'])}",
                )
            for i, x in enumerate(e["index"]):
                self.expr(f"{path}.index[{i}]", x, scope, device)

    def texpr(self, path, t, scope):
        """Check a TExpr; return (shape, w) or None."""
        if not isinstance(t, dict) or tag(t, "t") not in TEXPR_FIELDS:
            self.err(path, f"expected a TExpr with 't' in {sorted(TEXPR_FIELDS)}")
            return None
        kind = t["t"]
        if not self.keys(path, t, {"t", "shape", "w"} | TEXPR_FIELDS[kind], {"addr"}):
            return None
        shape, w = t["shape"], t["w"]
        if not (
            isinstance(shape, list)
            and shape
            and all(is_int(d) and d >= 1 for d in shape)
        ):
            self.err(f"{path}.shape", "expected a non-empty list of positive ints")
            return None
        if w not in WIDTHS or isinstance(w, bool):
            self.err(path, f"'w' must be one of {WIDTHS}")
            return None
        if "addr" in t:
            self.bool_field(f"{path}.addr", t["addr"])

        def sub(field):
            return self.texpr(f"{path}.{field}", t[field], scope)

        def expect(cond, msg):
            if not cond:
                self.err(path, msg)

        if kind == "range":
            if self.int_field(f"{path}.start", t["start"]) and self.int_field(
                f"{path}.end", t["end"]
            ):
                expect(
                    shape == [t["end"] - t["start"]],
                    f"range shape must be [{t['end'] - t['start']}]",
                )
        elif kind == "full":
            if self.int_field(f"{path}.v", t["v"]) and not fits(t["v"], w):
                self.err(f"{path}.v", f"{t['v']} does not fit in {w} bits")
        elif kind == "splat":
            wa = self.expr(f"{path}.a", t["a"], scope)
            expect(wa in (w, None), f"splat operand has w={wa}, node has w={w}")
        elif kind in ("ext", "trunc"):
            a = sub("a")
            if kind == "ext":
                self.bool_field(f"{path}.signed", t["signed"])
            if a is not None:
                expect(
                    a[0] == shape, f"operand a has shape {a[0]}, node has shape {shape}"
                )
                if kind == "ext":
                    expect(a[1] < w, f"ext must widen (operand w={a[1]}, result w={w})")
                else:
                    expect(
                        a[1] > w, f"trunc must narrow (operand w={a[1]}, result w={w})"
                    )
        elif kind in ("bcast", "expand"):
            a = sub("a")
            axis = t["axis"]
            if not self.int_field(f"{path}.axis", axis, 0) or a is None:
                return shape, w
            ash, aw = a
            expect(aw == w, f"{kind} operand has w={aw}, node has w={w}")
            if kind == "expand":
                expect(
                    axis <= len(ash) and shape == ash[:axis] + [1] + ash[axis:],
                    f"expand of {ash} at axis {axis} must have shape {ash[:axis] + [1] + ash[axis:]}",
                )
            else:
                ok = axis < len(ash) and ash[axis] == 1 and len(shape) == len(ash)
                ok = ok and all(
                    shape[i] == ash[i] for i in range(len(ash)) if i != axis
                )
                expect(
                    ok, f"bcast of {ash} along axis {axis} cannot give shape {shape}"
                )
        else:
            parts = {f: sub(f) for f in ("c", "a", "b") if f in t}
            for f, r in parts.items():
                if r is not None:
                    expect(
                        r[0] == shape,
                        f"operand {f} has shape {r[0]}, node has shape {shape}",
                    )
            if kind == "bin":
                expect(t["o"] in BIN_OPS, f"unknown bin op {t['o']!r}")
                for f in ("a", "b"):
                    expect(
                        parts[f] is None or parts[f][1] == w,
                        f"operand {f} width differs from node w={w}",
                    )
            elif kind == "cmp":
                expect(t["o"] in CMP_OPS, f"unknown cmp op {t['o']!r}")
                expect(w == 1, "cmp result must have w=1")
                if parts["a"] and parts["b"]:
                    expect(parts["a"][1] == parts["b"][1], "cmp operand widths differ")
            else:
                expect(
                    parts["c"] is None or parts["c"][1] == 1,
                    "select condition must have w=1",
                )
                for f in ("a", "b"):
                    expect(
                        parts[f] is None or parts[f][1] == w,
                        f"operand {f} width differs from node w={w}",
                    )
        return shape, w

    # ----------------------------------------------------------------- statements

    def block(self, path, stmts, scope):
        if not self.list_field(path, stmts):
            return
        scope = dict(scope)
        for i, s in enumerate(stmts):
            self.stmt(f"{path}[{i}]", s, scope)

    def bind(self, path, name, w, scope, readonly=False):
        if not self.string(path, name):
            return
        if name in scope:
            if scope[name][1]:
                self.err(
                    path, f"'{name}' is a param or for variable and cannot be assigned"
                )
            elif w is not None and scope[name][0] != w:
                self.err(
                    path,
                    f"assignment changes the width of '{name}' from {scope[name][0]} to {w}",
                )
            return
        if w is not None:
            scope[name] = (w, readonly)

    def stmt(self, path, s, scope):
        op = tag(s, "op")
        if op == "event":
            self.event(path, s, scope)
            return
        if op not in STMT_FIELDS:
            self.err(
                path,
                f"expected a statement with 'op' in {sorted(STMT_FIELDS) + ['event']}",
            )
            return
        if not self.keys(path, s, {"op"} | STMT_FIELDS[op], {"loc"}):
            return
        if "loc" in s:
            self.string(f"{path}.loc", s["loc"])
        if op == "let":
            w = self.expr(f"{path}.expr", s["expr"], scope)
            self.bind(f"{path}.var", s["var"], w, scope)
        elif op == "for":
            ws = [self.expr(f"{path}.{f}", s[f], scope) for f in ("lb", "ub", "step")]
            if None not in ws and len(set(ws)) != 1:
                self.err(path, f"lb/ub/step widths differ {ws}")
            inner = dict(scope)
            self.bind(f"{path}.var", s["var"], ws[0], inner, readonly=True)
            self.block(f"{path}.body", s["body"], inner)
        elif op == "while":
            self.cond(f"{path}.cond", s["cond"], scope)
            self.int_field(f"{path}.max_iters", s["max_iters"], 0)
            self.block(f"{path}.body", s["body"], scope)
        elif op == "if":
            self.cond(f"{path}.cond", s["cond"], scope)
            self.block(f"{path}.then", s["then"], scope)
            self.block(f"{path}.else", s["else"], scope)
        elif op == "assert":
            self.cond(f"{path}.cond", s["cond"], scope)
            self.string(f"{path}.msg", s["msg"])

    def cond(self, path, e, scope):
        if self.expr(path, e, scope) not in (1, None):
            self.err(path, "condition must have w=1")

    def _check_tma_coords(self, path: str, s: dict, scope: dict) -> None:
        t = self.tensors.get(s["desc"]) if isinstance(s["desc"], str) else None
        if t is None or t.get("kind") != "desc":
            self.err(f"{path}.desc", f"{s['desc']!r} is not a desc tensor")
        if self.list_field(f"{path}.coords", s["coords"]):
            if (
                t is not None
                and isinstance(t.get("shape"), list)
                and len(s["coords"]) != len(t["shape"])
            ):
                self.err(
                    f"{path}.coords",
                    f"rank {len(s['coords'])} != tensor rank {len(t['shape'])}",
                )
            for i, c in enumerate(s["coords"]):
                self.expr(f"{path}.coords[{i}]", c, scope)

    def event(self, path, s, scope):
        ev = tag(s, "ev")
        if ev not in EVENT_FIELDS:
            self.err(path, f"expected 'ev' in {sorted(EVENT_FIELDS)}")
            return
        required, optional = EVENT_FIELDS[ev]
        if not self.keys(path, s, {"op", "ev", "loc"} | required, optional):
            return
        self.string(f"{path}.loc", s["loc"])
        if "bar" in s and not (isinstance(s["bar"], str) and s["bar"] in self.barriers):
            self.err(f"{path}.bar", f"unknown barrier {s['bar']!r}")
        for f in ("idx", "count", "bytes", "parity"):
            if f in s:
                self.expr(f"{path}.{f}", s[f], scope)
        self._check_cluster_fields(path, ev, s, scope)
        if "n" in s:
            self.int_field(f"{path}.n", s["n"], 1)
        if "per_warp" in s:
            self.bool_field(f"{path}.per_warp", s["per_warp"])
        if ev == "arrive" and is_int(self.role_warps) and self.role_warps > 1:
            n, per = s.get("n", 1), s.get("per_warp", False)
            mult = self.role_warps if per is True else self.role_tpi
            if is_int(n) and is_int(mult) and n * mult < self.role_warps:
                self.few_arrivals.append((path, n * mult, self.role_warps))
        if ev in WRITE_EVENTS:
            name = s.get("desc", s.get("tensor"))
            t = self.tensors.get(name) if isinstance(name, str) else None
            if t is not None and t.get("role") == "input":
                self.err(
                    path,
                    f"{ev} writes '{name}', declared role 'input': declare it 'output' (P0-3 checks it) or 'workspace'",
                )
        if ev in ("tma_load", "tma_store", "tma_reduce"):
            self._check_tma_coords(path, s, scope)
        elif ev in ("gload", "gstore", "gatomic"):
            t = self.tensors.get(s["tensor"]) if isinstance(s["tensor"], str) else None
            if t is None or t.get("kind") != "global":
                self.err(f"{path}.tensor", f"{s['tensor']!r} is not a global tensor")
            off = self.texpr(f"{path}.offsets", s["offsets"], scope)
            if s["mask"] is not None:
                m = self.texpr(f"{path}.mask", s["mask"], scope)
                if m is not None and m[1] != 1:
                    self.err(f"{path}.mask", "mask must have w=1")
                if m is not None and off is not None and m[0] != off[0]:
                    self.err(
                        f"{path}.mask", f"mask shape {m[0]} != offsets shape {off[0]}"
                    )
        elif ev in P1_EVENTS:
            self.p1_event(path, ev, s, scope)
        elif ev == "work":
            if isinstance(s["fields"], dict) and s["fields"]:
                for k, v in s["fields"].items():
                    self.expr(f"{path}.fields.{k}", v, scope)
                    self.work_fields.add(k)
            else:
                self.err(f"{path}.fields", "expected a non-empty object of Exprs")

    def _check_cluster_fields(self, path: str, ev: str, s: dict, scope: dict) -> None:
        """`cta` / `cta_mask` (w=32 Exprs) and tma_load's `leader` (kir.md, change 11)."""
        named = [f for f in CLUSTER_FIELDS if f in s] + (
            ["complete_tx"] if ev == "complete_tx" else []
        )
        if named and self.cluster_size <= 1:
            self.err(path, f"{', '.join(named)} without a cluster of more than one CTA")
        for f in CLUSTER_FIELDS:
            if f in s and self.expr(f"{path}.{f}", s[f], scope) not in (32, None):
                self.err(f"{path}.{f}", "must have w=32")
        if "leader" in s:
            self.bool_field(f"{path}.leader", s["leader"])
            if s["leader"] is True and self.cluster_size % 2:
                self.err(f"{path}.leader", "a pair leader needs an even cluster size")

    def _check_tmem_rects(self, path: str, rects: list, scope: dict) -> list:
        uses = []
        for i, r in enumerate(rects):
            rp = f"{path}.rects[{i}]"
            if not self.keys(rp, r, RECT_FIELDS):
                continue
            if r["use"] not in RECT_USES:
                self.err(f"{rp}.use", f"must be one of {RECT_USES}")
            self.bool_field(f"{rp}.write", r["write"])
            for f in ("lane", "col"):
                if self.expr(f"{rp}.{f}", r[f], scope) not in (32, None):
                    self.err(f"{rp}.{f}", "must have w=32")
            if r["nlanes"] not in (16, 32, 64, 128) or isinstance(r["nlanes"], bool):
                self.err(f"{rp}.nlanes", "must be 16, 32, 64 or 128")
            if not (is_int(r["ncols"]) and 1 <= r["ncols"] <= 512):
                self.err(f"{rp}.ncols", "must be an integer in [1, 512]")
            uses.append((r["use"], r["write"]))
        return uses

    def p1_event(self, path, ev, s, scope):
        """tmem / tmem_wait / named_bar events of a P1 KIR (kir.md, change 10)."""
        self.p1_events += 1
        if ev == "tmem_wait":
            self.bool_field(f"{path}.ld", s["ld"])
            return
        if ev == "named_bar":
            self.bool_field(f"{path}.sync", s["sync"])
            for f in ("id", "count"):
                e = s[f]
                if not (isinstance(e, dict) and e.get("e") == "const"):
                    self.err(
                        f"{path}.{f}", "named barrier id and count must be const Exprs"
                    )
                else:
                    self.expr(f"{path}.{f}", e, scope)
            c = s["count"]
            if (
                isinstance(c, dict)
                and is_int(c.get("v"))
                and (c["v"] <= 0 or c["v"] % 32)
            ):
                self.err(
                    f"{path}.count",
                    "named barrier count must be a positive multiple of 32",
                )
            return
        if "site" in s:
            self.string(f"{path}.site", s["site"])
        inst = s["inst"]
        if inst not in TMEM_INSTS:
            self.err(f"{path}.inst", f"must be one of {TMEM_INSTS}")
            return
        if self.expr(f"{path}.warp", s["warp"], scope) not in (32, None):
            self.err(f"{path}.warp", "must have w=32")
        rects = s["rects"]
        if not (isinstance(rects, list) and rects):
            self.err(f"{path}.rects", "expected a non-empty list")
            return
        uses = self._check_tmem_rects(path, rects, scope)
        if inst == "ld" and any(u != ("data", False) for u in uses):
            self.err(path, "an ld has only data reads")
        if inst in ("st", "cp") and any(u != ("data", True) for u in uses):
            self.err(path, f"a {inst} has only data writes")
        if inst == "mma":
            n = {k: sum(u[0] == k for u in uses) for k in RECT_USES}
            ok = n["data"] == 0 and n["d"] == 1 and n["a"] <= 1 and n["sf"] in (0, 2)
            ok = ok and all(w == (k == "d") for k, w in uses)
            if not ok:
                self.err(
                    path,
                    "an mma has exactly one d write, at most one a read and zero or two sf reads",
                )
        if inst in ("cp", "mma") and self.role_warps not in (1, None):
            self.err(
                path,
                f"a {inst} in a role of {self.role_warps} warps (cp/mma roles must be one warp)",
            )

    def tmem_top(self, k):
        t = k.get("tmem")
        if t is None:
            if self.p1_events:
                self.err(
                    "$",
                    "tmem, tmem_wait or named_bar events without a top-level 'tmem' (P1 KIR)",
                )
            return
        if not self.p1_events:
            self.err(
                "$.tmem",
                "a top-level 'tmem' without tmem, tmem_wait or named_bar events",
            )
        for path, n, warps in self.few_arrivals:
            self.err(
                path,
                f"{n} arrival(s) per arrive in a role of {warps} warps: in a P1 KIR every "
                "warp of a role issues each mbarrier arrive (split the role by issuing warps)",
            )
        if not self.keys("$.tmem", t, {"cols", "base"}):
            return
        if t["cols"] not in TMEM_COLS or isinstance(t["cols"], bool):
            self.err("$.tmem.cols", f"must be one of {TMEM_COLS}")
        if t["base"] not in ("zero", "relative"):
            self.err("$.tmem.base", "must be 'zero' or 'relative'")
        elif t["base"] == "zero" and t["cols"] != 512:
            self.err(
                "$.tmem.base",
                "'zero' (absolute addresses) needs a 512-column allocation",
            )

    # ------------------------------------------------------------------ top level

    def _check_params(self, k: dict) -> None:
        if self.list_field("$.params", k["params"]):
            for i, p in enumerate(k["params"]):
                path = f"$.params[{i}]"
                if self.keys(path, p, {"name", "type"}) and self.string(
                    f"{path}.name", p["name"]
                ):
                    if tag(p, "type") not in INT_TYPES:
                        self.err(f"{path}.type", f"must be one of {sorted(INT_TYPES)}")
                    elif p["name"] in self.params:
                        self.err(f"{path}.name", f"duplicate param '{p['name']}'")
                    else:
                        self.params[p["name"]] = (INT_TYPES[p["type"]], True)

    def _check_header(self, k: dict) -> None:
        if not (is_int(k["kir_version"]) and k["kir_version"] == 1):
            self.err("$.kir_version", "must be 1")
        if "target" in k and k["target"] not in TARGETS:
            self.err("$.target", f"must be one of {TARGETS}")
        if "compiler" in k and self.keys(
            "$.compiler", k["compiler"], {"name", "version"}
        ):
            for f in ("name", "version"):
                self.string(f"$.compiler.{f}", k["compiler"][f])
        for f in ("kernel", "source"):
            self.string(f"$.{f}", k[f])
        if k["ir"] not in IRS:
            self.err("$.ir", f"must be one of {IRS}")

    def _check_cluster(self, k: dict) -> None:
        c = k.get("cluster", [1, 1, 1])
        if not (
            isinstance(c, list) and len(c) == 3 and all(is_int(x) and x >= 1 for x in c)
        ):
            self.err("$.cluster", "expected 3 positive ints")
            return
        self.cluster_size = c[0] * c[1] * c[2]

    def kernel(self, k):
        if not self.keys("$", k, TOP_FIELDS, TOP_OPTIONAL):
            return
        self._check_header(k)
        self._check_cluster(k)
        self._check_params(k)
        host = dict(self.params)
        if self.list_field("$.grid", k["grid"]):
            if len(k["grid"]) != 3:
                self.err("$.grid", "must have 3 entries")
            for i, g in enumerate(k["grid"]):
                self.expr(f"$.grid[{i}]", g, host, device=False)
        if self.list_field("$.tensors", k["tensors"]):
            for i, t in enumerate(k["tensors"]):
                if isinstance(t, dict) and isinstance(t.get("name"), str):
                    if t["name"] in self.tensors:
                        self.err(
                            f"$.tensors[{i}].name", f"duplicate tensor '{t['name']}'"
                        )
                    self.tensors[t["name"]] = t
            for i, t in enumerate(k["tensors"]):
                self.tensor(f"$.tensors[{i}]", t, host)
        if self.list_field("$.barriers", k["barriers"]):
            for i, b in enumerate(k["barriers"]):
                path = f"$.barriers[{i}]"
                if self.keys(path, b, {"name", "count", "num"}) and self.string(
                    f"{path}.name", b["name"]
                ):
                    if b["name"] in self.barriers:
                        self.err(f"{path}.name", f"duplicate barrier '{b['name']}'")
                    self.barriers.add(b["name"])
                    self.expr(f"{path}.count", b["count"], host, device=False)
                    self.expr(f"{path}.num", b["num"], host, device=False)
        if self.list_field("$.roles", k["roles"]):
            if not k["roles"]:
                self.err("$.roles", "must not be empty")
            names = set()
            for i, r in enumerate(k["roles"]):
                path = f"$.roles[{i}]"
                if not self.keys(
                    path, r, {"name", "warps", "threads_per_issue", "body"}
                ):
                    continue
                if self.string(f"{path}.name", r["name"]):
                    if r["name"] in names:
                        self.err(f"{path}.name", f"duplicate role '{r['name']}'")
                    names.add(r["name"])
                self.int_field(f"{path}.warps", r["warps"], 1)
                self.int_field(f"{path}.threads_per_issue", r["threads_per_issue"], 1)
                self.role_warps = r["warps"] if is_int(r["warps"]) else None
                self.role_tpi = r["threads_per_issue"]
                self.block(f"{path}.body", r["body"], self.params)
                self.role_warps = self.role_tpi = None
        for name, t in self.tensors.items():
            seg = t.get("segments")
            if (
                isinstance(seg, dict)
                and isinstance(seg.get("per"), str)
                and seg["per"] not in self.work_fields
            ):
                self.err(
                    f"$.tensors[{name}].segments.per",
                    f"no work event defines field '{seg['per']}'",
                )
        self.tmem_top(k)
        self.case(k["case"])

    def tensor(self, path, t, host):
        is_desc = isinstance(t, dict) and t.get("kind") == "desc"
        if not self.keys(path, t, TENSOR_FIELDS | ({"box"} if is_desc else set())):
            return
        self.string(f"{path}.name", t["name"])
        if t["kind"] not in TENSOR_KINDS:
            self.err(f"{path}.kind", f"must be one of {TENSOR_KINDS}")
        self.int_field(f"{path}.elem_bytes", t["elem_bytes"], 1)
        rank = None
        if self.list_field(f"{path}.shape", t["shape"]) and self.list_field(
            f"{path}.strides", t["strides"]
        ):
            rank = len(t["shape"])
            if rank == 0 or len(t["strides"]) != rank:
                self.err(
                    path, "shape and strides must be non-empty and of equal length"
                )
            for f in ("shape", "strides"):
                for i, x in enumerate(t[f]):
                    self.expr(f"{path}.{f}[{i}]", x, host, device=False)
        if t["role"] not in TENSOR_ROLES:
            self.err(f"{path}.role", f"must be one of {TENSOR_ROLES}")
        self.bool_field(f"{path}.metadata", t["metadata"])
        if t["metadata"] is True and t["kind"] != "global":
            self.err(f"{path}.metadata", "metadata tensors must be global")
        if is_desc:
            box = t["box"]
            if not (
                isinstance(box, list)
                and all(is_int(b) and b >= 1 for b in box)
                and len(box) == rank
            ):
                self.err(f"{path}.box", "expected one positive int per dim")
        seg = t["segments"]
        if seg is not None and self.keys(
            f"{path}.segments", seg, {"dim", "offsets", "per"}
        ):
            if not (is_int(seg["dim"]) and rank is not None and 0 <= seg["dim"] < rank):
                self.err(f"{path}.segments.dim", "must be a dim of the tensor")
            off = (
                self.tensors.get(seg["offsets"])
                if isinstance(seg["offsets"], str)
                else None
            )
            if (
                off is None
                or off.get("metadata") is not True
                or len(off.get("shape", [])) != 1
            ):
                self.err(
                    f"{path}.segments.offsets",
                    f"{seg['offsets']!r} is not a 1-D metadata tensor",
                )
            self.string(f"{path}.segments.per", seg["per"])

    def case(self, c):
        if not self.keys("$.case", c, {"name", "inputs"}, {"elision_flags"}):
            return
        self.string("$.case.name", c["name"])
        flags = c.get("elision_flags", [])
        if not (
            isinstance(flags, list)
            and all(isinstance(f, str) and f for f in flags)
            and len(set(flags)) == len(flags)
        ):
            self.err("$.case.elision_flags", "expected a list of distinct flag names")
        if not self.list_field("$.case.inputs", c["inputs"]) or not c["inputs"]:
            self.err("$.case.inputs", "expected a non-empty list")
            return
        names = set()
        for i, inp in enumerate(c["inputs"]):
            path = f"$.case.inputs[{i}]"
            if not self.keys(path, inp, INPUT_REQUIRED, INPUT_OPTIONAL):
                continue
            if self.string(f"{path}.name", inp["name"]):
                if inp["name"] in names:
                    self.err(f"{path}.name", f"duplicate input '{inp['name']}'")
                names.add(inp["name"])
            self.input(path, inp)

    def input(self, path, inp):
        params = inp["params"]
        if isinstance(params, dict):
            for k in sorted(self.params.keys() - params.keys()):
                self.err(f"{path}.params", f"missing param '{k}'")
            for k, v in params.items():
                if k not in self.params:
                    self.err(f"{path}.params.{k}", "not a kernel param")
                elif self.int_field(f"{path}.params.{k}", v) and not fits_signed(
                    v, self.params[k][0]
                ):
                    self.err(
                        f"{path}.params.{k}",
                        f"{v} does not fit in {self.params[k][0]} bits",
                    )
        else:
            self.err(f"{path}.params", "expected an object")
        if "grid" in inp:
            g = inp["grid"]
            if not (
                isinstance(g, list)
                and len(g) == 3
                and all(is_int(x) and x >= 0 for x in g)
            ):
                self.err(f"{path}.grid", "expected 3 non-negative ints")
        meta_names = {n for n, t in self.tensors.items() if t.get("metadata") is True}
        meta = inp.get("metadata", {})
        if isinstance(meta, dict):
            for k in sorted(meta_names - meta.keys()):
                self.err(
                    f"{path}.metadata", f"missing contents of metadata tensor '{k}'"
                )
            for k, v in meta.items():
                if k not in meta_names:
                    self.err(f"{path}.metadata.{k}", "not a metadata tensor")
                elif not (isinstance(v, list) and all(is_int(x) for x in v)):
                    self.err(f"{path}.metadata.{k}", "expected a list of ints")
        else:
            self.err(f"{path}.metadata", "expected an object")
        flags = inp.get("flags", {})
        if not (
            isinstance(flags, dict) and all(isinstance(v, bool) for v in flags.values())
        ):
            self.err(f"{path}.flags", "expected an object of booleans")
        shapes = inp.get("shapes", {})
        if isinstance(shapes, dict):
            for k, v in shapes.items():
                t = self.tensors.get(k)
                if t is None:
                    self.err(f"{path}.shapes.{k}", "unknown tensor")
                elif not (
                    isinstance(v, list)
                    and all(is_int(x) and x >= 0 for x in v)
                    and len(v) == len(t.get("shape", []))
                ):
                    self.err(
                        f"{path}.shapes.{k}", "expected one non-negative int per dim"
                    )
        else:
            self.err(f"{path}.shapes", "expected an object")
        iw = inp.get("intended_writes", {})
        if not isinstance(iw, dict):
            self.err(f"{path}.intended_writes", "expected an object")
            return
        for k, v in iw.items():
            p = f"{path}.intended_writes.{k}"
            t = self.tensors.get(k)
            if t is None or t.get("role") != "output":
                self.err(p, "not an output tensor")
                continue
            if v == "all_rows_of_segments":
                if t.get("segments") is None:
                    self.err(p, "all_rows_of_segments needs a tensor with segments")
            elif v not in INTENDED_NAMED:
                ok = isinstance(v, list) and all(
                    isinstance(r, list)
                    and len(r) == 2
                    and all(is_int(x) and x >= 0 for x in r)
                    and r[0] <= r[1]
                    for r in v
                )
                if not ok:
                    self.err(
                        p,
                        f"expected one of {INTENDED_NAMED} or a list of [lo, hi] row ranges",
                    )


def validate(obj):
    """Return the list of KIR contract violations in obj (empty iff valid)."""
    errors = []
    _Checker(errors).kernel(obj)
    return errors


def main(paths):
    bad = 0
    for path in paths:
        with open(path) as f:
            errors = validate(json.load(f))
        for e in errors:
            print(f"{path}: {e}")
        bad += bool(errors)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
