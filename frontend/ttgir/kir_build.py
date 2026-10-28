# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""KIR node constructors (references/kir.md) and TTGIR type helpers."""

import re

from .mlir_text import find_top, split_top, strip_strings

INT_WIDTHS = {"i1": 1, "i32": 32, "i64": 64}
# Integer ops whose two's-complement wrap on an address path is a P0-1 violation (Layer 2
# checks the nodes marked `addr`); `trunc` nodes on address paths are marked too (lossy).
ADDR_OPS = ("add", "sub", "mul", "shl")


def elem_bits(t: str) -> int | None:
    """Bit width of an MLIR scalar element type (i8, ui8, bf16, f8E4M3FN, ...), else None."""
    m = re.match(r"^(?:[su]?i|f|bf)(\d+)$|^f(\d+)E\d+M\d+\w*$", t.strip())
    return int(m.group(1) or m.group(2)) if m else None


def elem_bytes(t: str) -> int | None:
    """Bytes per element of a global tensor's element type; None for sub-byte types.

    i1 is stored as one byte."""
    bits = elem_bits(t)
    if bits == 1:
        return 1
    return bits // 8 if bits is not None and bits % 8 == 0 else None


# -- scalar expressions ---------------------------------------------------------------
def const(v: int, w: int) -> dict:
    return {"e": "const", "v": int(v), "w": w}


def var(name: str, w: int) -> dict:
    return {"e": "var", "name": name, "w": w}


def pid(axis: int) -> dict:
    return {"e": "pid", "axis": axis, "w": 32}


def nprog(axis: int) -> dict:
    return {"e": "nprog", "axis": axis, "w": 32}


def bin_(o: str, a: dict, b: dict, w: int, addr: bool = False) -> dict:
    node = {"e": "bin", "o": o, "a": a, "b": b, "w": w}
    if addr and o in ADDR_OPS:
        node["addr"] = True
    return node


def cmp(o: str, a: dict, b: dict) -> dict:
    return {"e": "cmp", "o": o, "a": a, "b": b, "w": 1}


def select(c: dict, a: dict, b: dict) -> dict:
    return {"e": "select", "c": c, "a": a, "b": b, "w": a["w"]}


def ext(a: dict, w: int, signed: bool) -> dict:
    return a if a["w"] == w else {"e": "ext", "signed": signed, "a": a, "w": w}


def trunc(a: dict, w: int, addr: bool = False) -> dict:
    if a["w"] == w:
        return a
    return {"e": "trunc", "a": a, "w": w, **({"addr": True} if addr else {})}


def load(tensor: str, index: list, w: int) -> dict:
    return {"e": "load", "tensor": tensor, "index": index, "w": w}


def evaluate(e: dict, env: dict) -> int:
    """Value of a host-side Expr (constants, params, bin/cmp/select/ext/trunc) with the
    params in `env`, in the exact BitVec semantics of kir.md, read as signed. Division by
    zero is a ValueError."""
    w = e["w"]

    def signed(v: int, width: int = w) -> int:
        v &= (1 << width) - 1
        return v - (1 << width) if width > 1 and v >> (width - 1) else v

    def unsigned(v: int, width: int) -> int:
        return v & ((1 << width) - 1)

    k = e["e"]
    if k == "const":
        return signed(e["v"])
    if k == "var":
        return signed(env[e["name"]])
    if k == "select":
        return evaluate(e["a"] if evaluate(e["c"], env) else e["b"], env)
    if k == "ext":
        a = evaluate(e["a"], env)
        return signed(a if e["signed"] else unsigned(a, e["a"]["w"]))
    if k == "trunc":
        return signed(evaluate(e["a"], env))
    a, b = evaluate(e["a"], env), evaluate(e["b"], env)
    wa = e["a"]["w"]
    ua, ub = unsigned(a, wa), unsigned(b, wa)
    if k == "cmp":
        x, y = (ua, ub) if e["o"][0] == "u" else (a, b)
        return int(
            {
                "eq": x == y,
                "ne": x != y,
                "lt": x < y,
                "le": x <= y,
                "gt": x > y,
                "ge": x >= y,
            }[e["o"].lstrip("su")]
        )
    o = e["o"]
    if o in ("sdiv", "srem", "udiv", "urem") and b == 0:
        raise ValueError(f"division by zero in host-side expression {e}")
    if o in ("sdiv", "srem"):
        q = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)
        return signed(q if o == "sdiv" else a - b * q)
    ops = {
        "add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b, "udiv": lambda: ua // ub,
        "urem": lambda: ua % ub, "and": lambda: a & b, "or": lambda: a | b, "xor": lambda: a ^ b,
        "smin": lambda: min(a, b), "smax": lambda: max(a, b), "umin": lambda: min(ua, ub),
        "umax": lambda: max(ua, ub), "shl": lambda: ua << ub if ub < w else 0,
        "lshr": lambda: ua >> ub if ub < w else 0, "ashr": lambda: a >> min(ub, w - 1),
    }  # fmt: skip
    return signed(ops[o]())


def mark_addr(e: dict) -> dict:
    """Copy of a scalar Expr with `addr` set on every ADDR_OPS and trunc node (not below vars)."""
    if not isinstance(e, dict) or "e" not in e:
        return e
    out = {k: (mark_addr(v) if isinstance(v, dict) else v) for k, v in e.items()}
    if (out["e"] == "bin" and out["o"] in ADDR_OPS) or out["e"] == "trunc":
        out["addr"] = True
    return out


# -- tensor expressions ---------------------------------------------------------------
def t_splat(a: dict, shape: list) -> dict:
    return {"t": "splat", "shape": list(shape), "a": a, "w": a["w"]}


def t_range(start: int, end: int) -> dict:
    return {"t": "range", "shape": [end - start], "start": start, "end": end, "w": 32}


def t_full(v: int, shape: list, w: int) -> dict:
    return {"t": "full", "shape": list(shape), "v": int(v), "w": w}


def t_bin(o: str, a: dict, b: dict, w: int, addr: bool = False) -> dict:
    node = {"t": "bin", "shape": a["shape"], "o": o, "a": a, "b": b, "w": w}
    if addr and o in ADDR_OPS:
        node["addr"] = True
    return node


def t_cmp(o: str, a: dict, b: dict) -> dict:
    return {"t": "cmp", "shape": a["shape"], "o": o, "a": a, "b": b, "w": 1}


def t_select(c: dict, a: dict, b: dict) -> dict:
    return {"t": "select", "shape": a["shape"], "c": c, "a": a, "b": b, "w": a["w"]}


def t_ext(a: dict, w: int, signed: bool) -> dict:
    return (
        a
        if a["w"] == w
        else {"t": "ext", "shape": a["shape"], "signed": signed, "a": a, "w": w}
    )


def t_trunc(a: dict, w: int, addr: bool = False) -> dict:
    if a["w"] == w:
        return a
    return {
        "t": "trunc",
        "shape": a["shape"],
        "a": a,
        "w": w,
        **({"addr": True} if addr else {}),
    }


def is_texpr(x: dict) -> bool:
    return "t" in x


def is_const_zero(x: dict) -> bool:
    return x.get("e") == "const" and x["v"] == 0


# -- types ----------------------------------------------------------------------------
TENSOR_RE = re.compile(r"^tensor<((?:\d+x)*)(.+?)(?:, #.*)?>$")


def tensor_type(t: str) -> tuple[list, str] | None:
    """`tensor<128x64xi32, #l>` -> ([128, 64], "i32"); None for non-tensor types."""
    m = TENSOR_RE.match(t.strip())
    if not m:
        return None
    dims = [int(d) for d in m.group(1).split("x") if d]
    return dims, m.group(2).strip()


def elem_and_shape(t: str) -> tuple[list, str]:
    tt = tensor_type(t)
    return tt if tt else ([], t.strip())


def ptr_pointee(t: str) -> str | None:
    m = re.match(r"^!tt\.ptr<([^,>]+)(?:, \d+)?>$", t.strip())
    return m.group(1) if m else None


def is_ptr_type(t: str) -> bool:
    """A pointer or a tensor of pointers."""
    return ptr_pointee(elem_and_shape(t)[1]) is not None


def _shaped(t: str, prefix: str) -> tuple[list, str, list] | None:
    """`<prefix><128x64xf16, #enc, ...>` -> ([128, 64], "f16", ["#enc", ...])."""
    t = t.strip()
    if not (t.startswith(prefix + "<") and t.endswith(">")):
        return None
    parts = split_top(t[len(prefix) + 1 : -1])
    m = re.match(r"^((?:\d+x)+)(\w+)$", parts[0])
    if not m:
        return None
    return [int(d) for d in m.group(1).split("x") if d], m.group(2), parts[1:]


def tensordesc_type(t: str) -> tuple[list, str, list] | None:
    """`!tt.tensordesc<128x16xbf16, #shared1>` -> ([128, 16], "bf16", ["#shared1"])."""
    return _shaped(t, "!tt.tensordesc")


def memdesc_type(t: str) -> tuple[list, str, list] | None:
    """`!ttg.memdesc<2xi64, #s, #smem, mutable>` -> ([2], "i64", ["#s", "#smem", "mutable"])."""
    return _shaped(t, "!ttg.memdesc")


def nvmma_params(enc: str) -> dict | None:
    """Parameters of a `#ttg.nvmma_shared<{...}>` encoding (alias resolved), else None."""
    m = re.match(r"^#ttg\.nvmma_shared<\{(.*)\}>$", enc.strip())
    return dict(re.findall(r"(\w+) = (\w+)", m.group(1))) if m else None


def tma_message_box(shape: list, nvmma: dict) -> list:
    """Box of one TMA message for a block `shape` in an NVMMA shared encoding.

    getTMABlockShapeTiled (lib/Dialect/TritonGPU/IR/Dialect.cpp) without fp4Padded: every
    dim clamped to 256, the contiguous dim (0 if transposed, else the last) set to the
    swizzle width in elements. The lowering issues shape / box messages per dim.
    """
    box = [min(s, 256) for s in shape]
    swizzle = int(nvmma["swizzlingByteWidth"])
    if swizzle:
        contig = 0 if nvmma["transposed"] == "true" else len(box) - 1
        box[contig] = 8 * swizzle // int(nvmma["elementBitWidth"])
    return box


def type_section(text: str) -> str:
    clean = strip_strings(text)
    idx = find_top(clean, " : ")
    return clean[idx + 3 :].strip() if idx >= 0 else ""


def types_after(section: str, sep: str) -> list[str]:
    idx = find_top(section, sep)
    if idx < 0:
        return []
    tail = section[idx + len(sep) :].strip()
    if tail.startswith("(") and tail.endswith(")"):
        tail = tail[1:-1]
    return [t for t in split_top(tail) if t]
