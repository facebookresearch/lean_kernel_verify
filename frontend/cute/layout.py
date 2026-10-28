# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CuTe layout algebra on the values the CuTe dialect types describe.

An IntTuple is a nested Python list whose leaves are integer expressions (expr.py),
`Basis` elements (ArithTuple strides such as `16@0`) or `UNDER` (the `_` of a slicing
coordinate). Static leaves are constant expressions; the dynamic leaves of a type string
(`?`, `?{i64 div=8}`) are `Dyn` placeholders until `fill` supplies the values the
defining op passes, in leaf order.

Implemented: type-string parsing, leaves, crd2idx (hierarchical coordinates, a scalar
coordinate split colexicographically over a tuple mode, basis strides giving
ArithTuple components), size, mode access and elementwise tuple arithmetic. These are
the parts the lowered IR still uses; anything else is an unsupported op for the caller.
"""

import re
from dataclasses import dataclass

from . import expr as E


class LayoutError(Exception):
    pass


@dataclass(frozen=True)
class Basis:
    """`scale@m1@m2...` as printed: scale times the unit vector of a coordinate mode.

    CuTe prints a nested basis E<a,b> innermost first (`1@b@a`), so `path` (outermost
    mode first) is the printed mode list reversed. A rational scale `n/d@m` (TMA
    coordinates in units wider than the element) has den = d."""

    scale: tuple  # expression (the numerator)
    path: tuple
    den: int = 1


class _Under:
    def __repr__(self) -> str:
        return "_"


UNDER = _Under()


@dataclass(frozen=True)
class Dyn:
    """A `?` leaf of a type string: width and known divisibility."""

    w: int
    div: int


_TOKEN = re.compile(r"\s*(\(|\)|,|:|_|\?(?:\{[^}]*\})?|-?\d+(?:/\d+)?(?:@\d+)*)")


def _tuple_at(s: str, pos: int):
    m = _TOKEN.match(s, pos)
    if not m:
        raise LayoutError(f"cannot parse {s!r} at {pos}")
    tok, pos = m.group(1), m.end()
    if tok == "(":
        items = []
        while True:
            item, pos = _tuple_at(s, pos)
            items.append(item)
            m = _TOKEN.match(s, pos)
            if not m or m.group(1) not in (",", ")"):
                raise LayoutError(f"expected ',' or ')' in {s!r} at {pos}")
            pos = m.end()
            if m.group(1) == ")":
                return items, pos
    if tok == "_":
        return UNDER, pos
    if tok.startswith("?"):
        attrs = tok[2:-1]
        div = re.search(r"div=(\d+)", attrs)
        return Dyn(64 if "i64" in attrs else 32, int(div.group(1)) if div else 1), pos
    if "@" in tok:
        scale, *modes = tok.split("@")
        num, _, den = scale.partition("/")
        return Basis(
            E.const(int(num), 32),
            tuple(reversed([int(m) for m in modes])),
            int(den or 1),
        ), pos
    if "/" in tok:
        raise LayoutError(f"rational leaf {tok} outside a basis in {s!r}")
    if tok in (",", ")", ":"):
        raise LayoutError(f"unexpected {tok!r} in {s!r} at {pos}")
    return E.const(int(tok), 32), pos


def parse_tuple(s: str):
    v, pos = _tuple_at(s, 0)
    if s[pos:].strip():
        raise LayoutError(f"trailing text in {s!r}")
    return v


def parse_layout(s: str):
    """`shape:stride` -> (shape, stride)."""
    shape, pos = _tuple_at(s, 0)
    m = _TOKEN.match(s, pos)
    if not m or m.group(1) != ":":
        raise LayoutError(f"expected ':' in layout {s!r}")
    stride, pos = _tuple_at(s, m.end())
    if s[pos:].strip():
        raise LayoutError(f"trailing text in {s!r}")
    if not congruent(shape, stride):
        raise LayoutError(f"shape and stride are not congruent in {s!r}")
    return shape, stride


_COMPOSED = re.compile(r"^S<(\d+),(\d+),(\d+)> o (-?\d+) o (.*)$")


def parse_composed(s: str):
    """`S<b,m,s> o off o shape:stride` -> ((b, m, s), offset, (shape, stride))."""
    m = _COMPOSED.match(s.strip())
    if not m:
        raise LayoutError(f"unsupported composed layout {s!r}")
    b, mm, sh, off = (int(x) for x in m.groups()[:4])
    return (b, mm, sh), off, parse_layout(m.group(5))


def leaves(t) -> list:
    if isinstance(t, list):
        return [x for y in t for x in leaves(y)]
    return [t]


def rebuild(ref, flat: list):
    """`flat` (leaf order) arranged in the structure of `ref`."""
    it = iter(flat)

    def go(r):
        return [go(x) for x in r] if isinstance(r, list) else next(it)

    return go(ref)


def fill(t, values: list):
    """Replace the Dyn leaves of `t`, in order, with `values` (resized to the leaf width)."""
    vals = iter(values)

    def leaf(x):
        if not isinstance(x, Dyn):
            return x
        v = next(vals, None)
        if v is None:
            raise LayoutError("too few dynamic values for the tuple type")
        return E.resize(v, x.w)

    out = rebuild(t, [leaf(x) for x in leaves(t)])
    if next(vals, None) is not None:
        raise LayoutError("too many dynamic values for the tuple type")
    return out


def dyn_count(t) -> int:
    return sum(isinstance(x, Dyn) for x in leaves(t))


def congruent(a, b) -> bool:
    if isinstance(a, list) != isinstance(b, list):
        return False
    return not isinstance(a, list) or (len(a) == len(b) and all(map(congruent, a, b)))


def rank(t) -> int:
    return len(t) if isinstance(t, list) else 1


def mode(t, i: int):
    if isinstance(t, list):
        if i >= len(t):
            raise LayoutError(f"mode {i} of a rank-{len(t)} tuple")
        return t[i]
    if i != 0:
        raise LayoutError(f"mode {i} of a rank-1 tuple")
    return t


def at_path(t, path: tuple):
    for i in path:
        t = mode(t, i)
    return t


def as_int(x, w: int):
    if not isinstance(x, tuple):
        raise LayoutError(f"expected an integer leaf, got {x!r}")
    return E.resize(x, w)


def size(shape):
    """Product of all leaves (i32 unless a leaf is i64)."""
    ls = leaves(shape)
    w = max([E.width(x) for x in ls if isinstance(x, tuple)] + [32])
    out = E.const(1, w)
    for x in ls:
        out = E.bin("mul", out, as_int(x, w))
    return out


def _add(a, b):
    """Sum of index values: integers, or ArithTuple components {path: expr}."""
    if isinstance(a, dict) or isinstance(b, dict):
        out = dict(a) if isinstance(a, dict) else ({} if E.is_const(a, 0) else {(): a})
        for k, v in (
            b.items()
            if isinstance(b, dict)
            else ([] if E.is_const(b, 0) else [((), b)])
        ):
            out[k] = _add(out[k], v) if k in out else v
        return out
    w = max(E.width(a), E.width(b))
    return E.bin("add", as_int(a, w), as_int(b, w))


def _scale(c, s, require):
    """Coordinate (expr) times one stride leaf (expr or Basis)."""
    if isinstance(s, Basis):
        w = max(E.width(c), E.width(s.scale))
        v = E.bin("mul", as_int(c, w), as_int(s.scale, w))
        if s.den != 1:
            den = E.const(s.den, w)
            exact = E.cmp("eq", E.bin("srem", v, den), E.const(0, w))
            if not E.is_const(exact, 1):
                if require is None:
                    raise LayoutError(
                        f"coordinate {to_str(c)} times rational stride {to_str(s)} is not a known integer"
                    )
                require(
                    exact,
                    f"coordinate times rational stride {to_str(s)} is not an integer",
                )
            v = E.bin("sdiv", v, den)
        return {s.path: v}
    w = max(E.width(c), E.width(s))
    return E.bin("mul", as_int(c, w), as_int(s, w))


def crd2idx(coord, shape, stride, require=None):
    """CuTe crd2idx. `_` coordinates contribute 0 (the offset of a slice).

    Returns an expression, or {path: expr} when strides are basis elements (coordinate
    tensors). A scalar coordinate of a tuple mode is split colexicographically (first
    sub-mode fastest), as CuTe does. A coordinate times a rational basis stride must be
    an integer: `require(cond, msg)` receives that condition when it is not constant
    (without `require`, such a coordinate is an error)."""
    if coord is UNDER:
        return E.const(0, 32)
    if isinstance(coord, list):
        if not isinstance(shape, list) or len(coord) != len(shape):
            raise LayoutError(
                f"coordinate of rank {rank(coord)} for shape of rank {rank(shape)}"
            )
        out = E.const(0, 32)
        for c, sh, st in zip(coord, shape, stride):
            out = _add(out, crd2idx(c, sh, st, require))
        return out
    if isinstance(shape, list):
        out, rest = E.const(0, 32), coord
        for i, (sh, st) in enumerate(zip(shape, stride)):
            if i == len(shape) - 1:
                ci = rest
            else:
                n = size(sh)
                w = max(E.width(rest), E.width(n))
                ci = E.bin("srem", as_int(rest, w), as_int(n, w))
                rest = E.bin("sdiv", as_int(rest, w), as_int(n, w))
            out = _add(out, crd2idx(ci, sh, st, require))
        return out
    if isinstance(stride, list):
        raise LayoutError("scalar shape with a tuple stride")
    return _scale(coord, stride, require)


def elementwise(op: str, a, b):
    """Elementwise integer op on congruent tuples, or tuple op scalar."""
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            if op not in ("add", "sub"):
                raise LayoutError(f"tuple {op} of different ranks")
            n = max(len(a), len(b))  # CuTe tuple add/sub pad the shorter tuple with 0
            zero = E.const(0, 32)
            a, b = a + [zero] * (n - len(a)), b + [zero] * (n - len(b))
        return [elementwise(op, x, y) for x, y in zip(a, b)]
    if isinstance(a, list):
        return [elementwise(op, x, b) for x in a]
    if isinstance(b, list):
        return [elementwise(op, a, y) for y in b]
    if not (isinstance(a, tuple) and isinstance(b, tuple)):
        raise LayoutError(f"tuple {op} on non-integer leaves {to_str(a)}, {to_str(b)}")
    w = max(E.width(a), E.width(b))
    a, b = as_int(a, w), as_int(b, w)
    if op == "div":
        return E.bin("sdiv", a, b)
    if op == "ceil_div":
        return E.ceildiv(a, b)
    return E.bin(op, a, b)


def to_str(t) -> str:
    if isinstance(t, list):
        return "(" + ",".join(map(to_str, t)) + ")"
    if t is UNDER:
        return "_"
    if isinstance(t, Basis):
        scale = to_str(t.scale) + (f"/{t.den}" if t.den != 1 else "")
        return scale + "".join(f"@{m}" for m in reversed(t.path))
    if isinstance(t, Dyn):
        return "?"
    return str(t[2]) if E.is_const(t) else "?"
