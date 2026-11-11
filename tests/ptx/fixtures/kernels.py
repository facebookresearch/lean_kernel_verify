# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Hand-written PTX kernels for the core translator tests (test_interp, test_mem,
test_translate): a kernel builder, translate() with a ready policy, and evaluate(), a
small KIR evaluator of one role (values, select arms, addr wraps, metadata bounds).

A kernel `k` has the parameters `ptrs` (pointer params, the tensors of the same name,
f32 unless `elem`), then `scalars` (.b32 params), and an extern shared array `smem`;
`body` is the PTX between the register declarations and the closing brace. Locations
name the file `/src/k.py` (below the policy's src_root `/src`).
"""

from ....frontend.ptx import translate as T

SRC = "/src"


def kernel(body, ptrs=("x",), scalars=("n",), threads=128, extra="", src=SRC):
    params = [
        f".param .u64 .ptr .global .align 1 k_param_{i}" for i in range(len(ptrs))
    ]
    params += [f".param .b32 k_param_{len(ptrs) + i}" for i in range(len(scalars))]
    return f"""
.version 9.3
.target sm_100a
.address_size 64
.extern .shared .align 16 .b8 smem[];
{extra}
.visible .entry k(
    {", ".join(params)}
)
.reqntid {threads}
{{
    .reg .pred %p<32>;
    .reg .b16 %rs<16>;
    .reg .b32 %r<200>;
    .reg .b64 %rd<64>;
    .reg .f32 %f<16>;
    .loc 1 1 0
{body}
}}
    .file 1 "{src}/k.py"
"""


def run(body, family="ttgir", ptrs=("x",), scalars=("n",), threads=128, elem=None,
        cluster=(1, 1, 1), extra="", src=SRC, **pol):  # fmt: skip
    """Facts of kernel(body, ...) under the family's policy (overrides in pol); the
    source root is `src`."""
    elem = elem or {}
    params = [{"ptx": f"k_param_{i}", "kind": "ptr", "name": p, "size": 8, "fields": {}}
              for i, p in enumerate(ptrs)]  # fmt: skip
    params += [{"ptx": f"k_param_{len(ptrs) + i}", "kind": "scalar", "name": s, "size": 4,
                "fields": {}} for i, s in enumerate(scalars)]  # fmt: skip
    policy = T.policy(family, src_root=src, params=params,
                      tensors={p: {"elem_bytes": elem.get(p, 4)} for p in ptrs}, **pol)  # fmt: skip
    rec = {"block": [threads, 1, 1], "cluster": list(cluster)}
    return T.translate(
        kernel(body, ptrs, scalars, threads, extra, src), "k", [rec], policy
    )


def statements(stmts):
    for s in stmts:
        yield s
        for k in ("then", "else", "body"):
            yield from statements(s.get(k, []))


def events(facts, ev=None, role=None):
    return [s for r in facts.roles if role in (None, r["name"])
            for s in statements(r["body"]) if s["op"] == "event" and ev in (None, s["ev"])]  # fmt: skip


def kir_errors(facts) -> list:
    """frontend/kir.py's errors for a KIR around the Facts (1-D tensors of 4096 elements,
    every param 1, the barriers and roles as they are)."""
    from ....frontend import kir

    tensors = [{"name": n, "kind": "global", "elem_bytes": t["elem_bytes"],
                "shape": [{"e": "const", "w": 64, "v": 4096}], "strides": [{"e": "const", "w": 64, "v": 1}],
                "segments": None, "role": "output" if n in facts.written else "input",
                "metadata": t["metadata"]} for n, t in facts.tensors.items()]  # fmt: skip
    metadata = {n: [0] * 4096 for n, t in facts.tensors.items() if t["metadata"]}
    if facts.clc:  # what frontend/ttgir/to_kir.py declares for CLC (sync.py)
        c = lambda v, w=64: {"e": "const", "w": w, "v": v}  # noqa: E731
        tensors.append({"name": "__clc_response", "kind": "desc", "elem_bytes": 1, "shape": [c(16, 32)],
                        "strides": [c(1)], "box": [16], "segments": None, "role": "input",
                        "metadata": False})  # fmt: skip
        for n, strides in (("__clc_schedule", [4, 1]), ("__clc_launched", [1])):
            tensors.append({"name": n, "kind": "global", "elem_bytes": 4, "shape": [c(4)] * len(strides),
                            "strides": [c(x) for x in strides], "segments": None, "role": "input",
                            "metadata": True})  # fmt: skip
            metadata[n] = [0] * 4 ** len(strides)
    obj = {"kir_version": 1, "kernel": "k", "source": "k.py", "ir": "ptx", "target": "sm_100",
           "grid": [{"e": "const", "w": 32, "v": 1}] * 3,
           "params": [{"name": n, "type": f"i{w}"} for n, w in facts.params.items()],
           "tensors": tensors, "barriers": facts.barriers,
           "roles": [{k: r[k] for k in ("name", "warps", "threads_per_issue", "body")} for r in facts.roles],
           "case": {"name": "t", "inputs": [{"name": "i", "params": dict.fromkeys(facts.params, 1),
                                              "metadata": metadata,
                                              "intended_writes": dict.fromkeys(facts.written, "unchecked")}]}}  # fmt: skip
    return kir.validate(obj)


# ------------------------------------------------------------------ evaluation
#
# A small reading of lean/KV/Eval.lean for the statements and scalar expressions the
# tests' kernels produce (1-D TExprs only): one role of one CTA on concrete params and
# metadata, returning its trace. It checks what a KIR shape alone cannot show: values
# (merges, loop-carried slots), which arm of a select runs, and where Lean would report an
# `addr` wrap or a metadata load outside its tensor.


class _Stop(Exception):
    """A metadata load outside its tensor: the role stops (Eval.lean metaOob)."""


def _wrap(v, w):
    v &= (1 << w) - 1
    return v - (1 << w) if w > 1 and v >> (w - 1) else v


def _bin(o, x, y, w):
    ux, uy = x & ((1 << w) - 1), y & ((1 << w) - 1)
    if o in ("sdiv", "srem"):
        q = abs(x) // abs(y) * (1 if (x < 0) == (y < 0) else -1)
        return q if o == "sdiv" else x - y * q
    return {
        "add": lambda: x + y, "sub": lambda: x - y, "mul": lambda: x * y,
        "udiv": lambda: ux // uy, "urem": lambda: ux % uy,
        "shl": lambda: x << uy if uy < w else 0, "lshr": lambda: ux >> uy if uy < w else 0,
        "ashr": lambda: x >> min(uy, w - 1), "and": lambda: x & y, "or": lambda: x | y,
        "xor": lambda: x ^ y, "smin": lambda: min(x, y), "smax": lambda: max(x, y),
        "umin": lambda: min(ux, uy), "umax": lambda: max(ux, uy),
    }[o]()  # fmt: skip


def _ext(x, w, signed):
    """x (a w-bit value, i1 as 0/1) extended: signed or zero."""
    if w == 1:
        return -x if signed else x
    return x if signed else x & ((1 << w) - 1)


_CMPS = {
    "eq": lambda a, b: a == b, "ne": lambda a, b: a != b, "slt": lambda a, b: a < b,
    "sle": lambda a, b: a <= b, "sgt": lambda a, b: a > b, "sge": lambda a, b: a >= b,
}  # fmt: skip


def evaluate(facts, params, metadata=None, pid=(0, 0, 0), role=0):  # noqa: C901 (one case per KIR node and statement)
    """Trace of role `role` on one CTA: [("gload"|"gstore"|"gatomic", tensor, [offset]),
    ("work", {field: value}), ("overflow", node op), ("meta_oob", tensor, index),
    ("assert", msg), (event, bar, idx[, parity]) for barrier events]. A metadata tensor
    read with several indices has contents {(i0, i1, ...): value}."""
    env, trace, metadata = dict(params), [], metadata or {}

    def ex(e):  # scalar Expr
        k, w = e["e"], e["w"]
        if k == "const":
            return e["v"]
        if k == "var":
            return env[e["name"]]
        if k in ("pid", "nprog"):
            return pid[e["axis"]] if k == "pid" else 1
        if k == "select":
            return ex(e["a"]) if ex(e["c"]) else ex(e["b"])
        if k == "load":
            idx = [ex(x) for x in e["index"]]
            data = metadata[e["tensor"]]
            if len(idx) > 1:
                if tuple(idx) not in data:
                    trace.append(("meta_oob", e["tensor"], tuple(idx)))
                    raise _Stop()
                return _wrap(data[tuple(idx)], w)
            (i,) = idx
            if not 0 <= i < len(data):
                trace.append(("meta_oob", e["tensor"], i))
                raise _Stop()
            return _wrap(data[i], w)
        if k == "ext":
            return _ext(ex(e["a"]), e["a"]["w"], e["signed"])
        if k == "trunc":
            x = ex(e["a"])
            if e.get("addr") and _wrap(x, w) != x:
                trace.append(("overflow", "trunc"))
            return _wrap(x, w)
        x, y = ex(e["a"]), ex(e["b"])
        if k == "cmp":
            if e["o"][0] == "u":
                m = (1 << e["a"]["w"]) - 1
                return int(_CMPS["s" + e["o"][1:]](x & m, y & m))
            return int(_CMPS[e["o"]](x, y))
        r = _bin(e["o"], x, y, w)
        if e.get("addr") and _wrap(r, w) != r:
            trace.append(("overflow", e["o"]))
        return _wrap(r, w)

    def tex(t):  # 1-D TExpr: a list of lane values
        k = t["t"]
        n = t["shape"][0]
        if k == "splat":
            return [ex(t["a"])] * n
        if k == "full":
            return [t["v"]] * n
        if k == "range":
            return list(range(t["start"], t["end"]))
        if k == "ext":
            return [_ext(x, t["a"]["w"], t["signed"]) for x in tex(t["a"])]
        if k == "trunc":
            return [_wrap(x, t["w"]) for x in tex(t["a"])]
        if k == "select":
            return [
                a if c else b for c, a, b in zip(tex(t["c"]), tex(t["a"]), tex(t["b"]))
            ]
        xs, ys = tex(t["a"]), tex(t["b"])
        if k == "cmp":
            return [ex({"e": "cmp", "o": t["o"], "w": 1, "a": {"e": "const", "w": t["a"]["w"], "v": x},
                        "b": {"e": "const", "w": t["a"]["w"], "v": y}}) for x, y in zip(xs, ys)]  # fmt: skip
        out = [_bin(t["o"], x, y, t["w"]) for x, y in zip(xs, ys)]
        if t.get("addr") and any(_wrap(r, t["w"]) != r for r in out):
            trace.append(("overflow", t["o"]))
        return [_wrap(r, t["w"]) for r in out]

    def run_(stmts):
        for s in stmts:
            op = s["op"]
            if op == "let":
                env[s["var"]] = ex(s["expr"])
            elif op == "if":
                run_(s["then"] if ex(s["cond"]) else s["else"])
            elif op == "for":
                for i in range(ex(s["lb"]), ex(s["ub"]), ex(s["step"])):
                    env[s["var"]] = i
                    run_(s["body"])
            elif op == "while":
                for _ in range(s["max_iters"]):
                    if not ex(s["cond"]):
                        break
                    run_(s["body"])
                else:
                    raise AssertionError("while exceeds max_iters")
            elif op == "assert":
                if not ex(s["cond"]):
                    trace.append(("assert", s["msg"]))
            elif s["ev"] in ("gload", "gstore", "gatomic"):
                offs = tex(s["offsets"])
                if s["mask"] is not None:
                    offs = [o for o, m in zip(offs, tex(s["mask"])) if m]
                trace.append((s["ev"], s["tensor"], offs))
            elif s["ev"] == "work":
                trace.append(("work", {f: ex(v) for f, v in s["fields"].items()}))
            else:
                par = (ex(s["parity"]),) if "parity" in s else ()
                trace.append(
                    (s["ev"], s.get("bar"), ex(s["idx"]) if s.get("idx") else None)
                    + par
                )

    try:
        run_(facts.roles[role]["body"])
    except _Stop:
        pass
    return trace
