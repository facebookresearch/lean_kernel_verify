# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Integer expressions of the CuTe frontend (KIR `Expr`, references/kir.md).

Expressions are interned tuples: structurally equal expressions are the same object, so
equality and hashing are O(1) and memo tables are cheap.

    ("const", w, v)              v in signed two's-complement range of w bits (i1: 0/1)
    ("var", w, name)
    ("pid", w, axis) / ("nprog", w, axis)
    ("bin", w, op, a, b)         op in BIN_OPS
    ("cmp", op, a, b)            width 1
    ("select", w, c, a, b)
    ("ext", w, signed, a) / ("trunc", w, a)
    ("load", w, tensor, (index,))
    ("table", w, key, entries)   entries[key]: `key` varies per thread, entries do not

Thread-varying values are expressions over three reserved variables: LANE (0..31),
WARP (absolute warp index) and ELECT (i1, true on the lane `elect.sync` picks in each
warp). They never reach KIR as such: interp.py substitutes the threads of a role.
Folding is exact BitVec arithmetic, so it never changes a value.
"""

BIN_OPS = (
    "add", "sub", "mul", "sdiv", "udiv", "srem", "urem", "shl", "ashr", "lshr",
    "and", "or", "xor", "smin", "smax", "umin", "umax",
)  # fmt: skip
CMP_OPS = ("eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge")
PARTIAL_OPS = ("sdiv", "udiv", "srem", "urem")


class ExprError(Exception):
    """An operation with no exact KIR meaning (e.g. a constant division by zero)."""


class _Node(tuple):
    __hash__ = object.__hash__

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other


_INTERN: dict = {}


def _mk(*parts):
    n = _INTERN.get(parts)
    if n is None:
        n = _INTERN[parts] = _Node(parts)
    return n


def _wrap(v: int, w: int) -> int:
    v &= (1 << w) - 1
    if w > 1 and v >> (w - 1):
        v -= 1 << w
    return v


def _unsigned(v: int, w: int) -> int:
    return v & ((1 << w) - 1)


def const(v: int, w: int):
    return _mk("const", w, _wrap(int(v), w))


def var(name: str, w: int):
    return _mk("var", w, name)


def pid(axis: int):
    return _mk("pid", 32, axis)


def nprog(axis: int):
    return _mk("nprog", 32, axis)


LANE = var("%lane", 32)
WARP = var("%warp", 32)
ELECT = var("%elect", 1)
THREAD_VARS = frozenset({"%lane", "%warp", "%elect"})
TRUE = const(1, 1)
FALSE = const(0, 1)


def width(e) -> int:
    return 1 if e[0] == "cmp" else e[1]


def is_const(e, v=None) -> bool:
    return e[0] == "const" and (v is None or e[2] == _wrap(v, e[1]))


def resize(e, w: int, signed: bool = True):
    """Sign/zero extend or truncate to width w (identity at the same width)."""
    we = width(e)
    return e if we == w else ext(e, w, signed) if we < w else trunc(e, w)


# ------------------------------------------------------------------- folding


def _fold_divrem(op: str, w: int, operands: tuple[int, int]) -> int:
    a, b = operands
    if b == 0:
        raise ExprError(f"constant {op} by zero")
    if op in ("sdiv", "srem"):
        q = abs(a) // abs(b)
        q = q if (a < 0) == (b < 0) else -q
        return q if op == "sdiv" else a - b * q
    ua, ub = _unsigned(a, w), _unsigned(b, w)
    return ua // ub if op == "udiv" else ua % ub


def _fold_bin(op: str, w: int, a: int, b: int) -> int:
    ua, ub = _unsigned(a, w), _unsigned(b, w)
    if op == "add":
        r = a + b
    elif op == "sub":
        r = a - b
    elif op == "mul":
        r = a * b
    elif op in PARTIAL_OPS:
        r = _fold_divrem(op, w, (a, b))
    elif op == "shl":
        r = ua << ub if ub < w else 0
    elif op == "lshr":
        r = ua >> ub if ub < w else 0
    elif op == "ashr":
        r = a >> min(ub, w - 1)
    elif op == "and":
        r = a & b
    elif op == "or":
        r = a | b
    elif op == "xor":
        r = a ^ b
    elif op == "smin":
        r = min(a, b)
    elif op == "smax":
        r = max(a, b)
    elif op == "umin":
        r = min(ua, ub)
    elif op == "umax":
        r = max(ua, ub)
    else:
        raise ExprError(f"unknown binary op {op}")
    return _wrap(r, w)


_COMMUTATIVE = ("add", "mul", "and", "or", "xor", "smin", "smax", "umin", "umax")


def bin(op: str, a, b):
    w = width(a)
    if width(b) != w:
        raise ExprError(f"{op}: width mismatch i{w} vs i{width(b)}")
    if is_const(a) and is_const(b):
        return const(_fold_bin(op, w, a[2], b[2]), w)
    if op in _COMMUTATIVE and is_const(a):
        a, b = b, a
    # Identities that hold in BitVec semantics for every value of the other operand.
    if is_const(b, 0) and op in ("add", "sub", "or", "xor", "shl", "lshr", "ashr"):
        return a
    if is_const(b, 0) and op in ("mul", "and"):
        return b
    if is_const(b, 1) and op in ("mul", "sdiv", "udiv"):
        return a
    if is_const(b, 1) and op in ("srem", "urem"):
        return const(0, w)
    if is_const(b, -1) and op == "and":
        return a
    if op == "sub" and is_const(b):
        return bin("add", a, const(-b[2], w))
    if (
        op == "add"
        and is_const(b)
        and a[0] == "bin"
        and a[2] == "add"
        and is_const(a[4])
    ):
        return bin("add", a[3], const(a[4][2] + b[2], w))
    if op in ("and", "or", "smin", "smax", "umin", "umax") and a is b:
        return a
    if op == "xor" and a is b:
        return const(0, w)
    return _mk("bin", w, op, a, b)


def _cmp_const(op: str, x: int, y: int, w: int) -> bool:
    if op[0] == "u":
        x, y = _unsigned(x, w), _unsigned(y, w)
    return {
        "eq": x == y, "ne": x != y, "slt": x < y, "sle": x <= y, "sgt": x > y,
        "sge": x >= y, "ult": x < y, "ule": x <= y, "ugt": x > y, "uge": x >= y,
    }[op]  # fmt: skip


_RANGE: dict = {}


def interval(e):
    """Signed bounds (lo, hi) that contain every value of e, or None; exact reasoning
    only (used to decide comparisons)."""
    if e in _RANGE:
        return _RANGE[e]
    r = _interval(e)
    if r is not None and not (
        -(1 << (width(e) - 1)) <= r[0] <= r[1] < (1 << (width(e) - 1)) or width(e) == 1
    ):
        r = None
    _RANGE[e] = r
    return r


def _interval(e):
    k = e[0]
    if k == "const":
        return e[2], e[2]
    if k == "cmp" or e is ELECT:
        return 0, 1
    if e is LANE:
        return 0, 31
    if k == "select":
        ia, ib = interval(e[3]), interval(e[4])
        return (min(ia[0], ib[0]), max(ia[1], ib[1])) if ia and ib else None
    if k == "ext":
        ia = interval(e[3])
        if width(e[3]) == 1:
            return (-1, 0) if e[2] else (0, 1)
        return ia if ia and (e[2] or ia[0] >= 0) else None
    if k == "trunc":
        return interval(e[2])
    if k != "bin":
        return None
    op, w = e[2], e[1]
    ia, ib = interval(e[3]), interval(e[4])
    if op == "and" and (ia and ia[0] >= 0 or ib and ib[0] >= 0):
        return 0, min(x[1] for x in (ia, ib) if x and x[0] >= 0)
    if op == "urem" and is_const(e[4]) and e[4][2] > 0:
        return 0, e[4][2] - 1
    if op == "lshr" and is_const(e[4]) and 1 <= e[4][2] < w:
        return 0, (1 << (w - e[4][2])) - 1
    if not (ia and ib):
        return None
    if op == "add":
        return ia[0] + ib[0], ia[1] + ib[1]
    if op == "sub":
        return ia[0] - ib[1], ia[1] - ib[0]
    if op == "mul":
        ps = [x * y for x in ia for y in ib]
        return min(ps), max(ps)
    if op in ("smin", "smax"):
        f = min if op == "smin" else max
        return f(ia[0], ib[0]), f(ia[1], ib[1])
    return None


_DECIDE = {
    "slt": lambda a, b: True if a[1] < b[0] else False if a[0] >= b[1] else None,
    "sle": lambda a, b: True if a[1] <= b[0] else False if a[0] > b[1] else None,
    "sgt": lambda a, b: True if a[0] > b[1] else False if a[1] <= b[0] else None,
    "sge": lambda a, b: True if a[0] >= b[1] else False if a[1] < b[0] else None,
    "eq": lambda a, b: False if a[1] < b[0] or b[1] < a[0] else None,
    "ne": lambda a, b: True if a[1] < b[0] or b[1] < a[0] else None,
}


def cmp(op: str, a, b):
    if op not in CMP_OPS:
        raise ExprError(f"unknown comparison {op}")
    w = width(a)
    if width(b) != w:
        raise ExprError(f"cmp {op}: width mismatch i{w} vs i{width(b)}")
    if is_const(a) and is_const(b):
        return const(int(_cmp_const(op, a[2], b[2], w)), 1)
    if a is b:
        return const(int(op in ("eq", "sle", "sge", "ule", "uge")), 1)
    ia, ib = interval(a), interval(b)
    if ia and ib and w > 1:
        sop = (
            op
            if op[0] != "u"
            else ("s" + op[1:] if ia[0] >= 0 and ib[0] >= 0 else None)
        )
        d = _DECIDE[sop](ia, ib) if sop else None
        if d is not None:
            return const(int(d), 1)
    if w == 1 and op in ("eq", "ne") and is_const(b):
        # i1 == c and i1 != c are the operand itself or its negation.
        return a if (op == "eq") == (b[2] == 1) else not1(a)
    return _mk("cmp", op, a, b)


def select(c, a, b):
    if width(c) != 1:
        raise ExprError("select condition must be i1")
    if width(a) != width(b):
        raise ExprError(f"select: width mismatch i{width(a)} vs i{width(b)}")
    if is_const(c):
        return a if c[2] else b
    if a is b:
        return a
    if width(a) == 1 and is_const(a, 1) and is_const(b, 0):
        return c
    return _mk("select", width(a), c, a, b)


def ext(a, w: int, signed: bool):
    wa = width(a)
    if w <= wa:
        raise ExprError(f"ext from i{wa} to i{w}")
    if is_const(a):
        v = a[2]
        if wa == 1:
            v = -v if signed else v
        elif not signed:
            v = _unsigned(v, wa)
        return const(v, w)
    return _mk("ext", w, bool(signed), a)


def trunc(a, w: int):
    wa = width(a)
    if w >= wa:
        raise ExprError(f"trunc from i{wa} to i{w}")
    if is_const(a):
        return const(a[2], w)
    if a[0] == "ext" and width(a[3]) == w:
        return a[3]
    return _mk("trunc", w, a)


def load(tensor: str, index, w: int):
    return _mk("load", w, tensor, (index,))


def table(key, entries: tuple):
    """entries[key]: `key` is a thread index expression, entries are thread-uniform."""
    if is_const(key):
        if not 0 <= key[2] < len(entries):
            raise ExprError(f"table index {key[2]} outside {len(entries)} entries")
        return entries[key[2]]
    if all(x is entries[0] for x in entries):
        return entries[0]
    return _mk("table", width(entries[0]), key, tuple(entries))


def not1(c):
    return bin("xor", c, TRUE)


def and1(a, b):
    return bin("and", a, b)


def floordiv(a, b):
    """arith.floordivsi: floor(a / b), signed."""
    w = width(a)
    q, r = bin("sdiv", a, b), bin("srem", a, b)
    fix = and1(cmp("ne", r, const(0, w)), cmp("slt", bin("xor", r, b), const(0, w)))
    return select(fix, bin("sub", q, const(1, w)), q)


def ceildiv(a, b):
    """arith.ceildivsi: ceil(a / b), signed."""
    w = width(a)
    q, r = bin("sdiv", a, b), bin("srem", a, b)
    fix = and1(cmp("ne", r, const(0, w)), cmp("sge", bin("xor", r, b), const(0, w)))
    return select(fix, bin("add", q, const(1, w)), q)


def ceildivu(a, b):
    """arith.ceildivui: ceil(a / b), unsigned."""
    w = width(a)
    q, r = bin("udiv", a, b), bin("urem", a, b)
    return select(cmp("ne", r, const(0, w)), bin("add", q, const(1, w)), q)


# -------------------------------------------------------------- traversals


def children(e) -> list:
    k = e[0]
    if k == "bin":
        return [e[3], e[4]]
    if k == "cmp":
        return [e[2], e[3]]
    if k == "select":
        return [e[2], e[3], e[4]]
    if k == "ext":
        return [e[3]]
    if k == "trunc":
        return [e[2]]
    if k == "load":
        return [e[3][0]]
    if k == "table":
        return [e[2], *e[3]]
    return []


def rebuild(e, args: list):
    """A node of e's kind over new children (folded)."""
    k = e[0]
    if k == "bin":
        return bin(e[2], args[0], args[1])
    if k == "cmp":
        return cmp(e[1], args[0], args[1])
    if k == "select":
        return select(*args)
    if k == "ext":
        return ext(args[0], e[1], e[2])
    if k == "trunc":
        return trunc(args[0], e[1])
    if k == "load":
        return load(e[2], args[0], e[1])
    if k == "table":
        return table(args[0], tuple(args[1:]))
    raise ExprError(f"{k} has no children")


_FREE: dict = {}


def free_vars(e) -> frozenset:
    r = _FREE.get(e)
    if r is None:
        r = (
            frozenset((e[2],))
            if e[0] == "var"
            else frozenset().union(*map(free_vars, children(e)))
        )
        _FREE[e] = r
    return r


def thread_varying(e) -> bool:
    return not free_vars(e).isdisjoint(THREAD_VARS)


def reserved(e) -> bool:
    """e uses a reserved variable other than the thread variables (a `%` name the frontend
    resolves before KIR, e.g. a fast divmod auxiliary value)."""
    return any(v.startswith("%") and v not in THREAD_VARS for v in free_vars(e))


def subst(e, mapping: dict, cache: dict | None = None):
    """Replace variables (by name) with expressions. `cache` may be shared between calls
    with the same mapping."""
    cache = {} if cache is None else cache
    names = frozenset(mapping)

    def go(x):
        r = cache.get(x)
        if r is None:
            if x[0] == "var":
                r = mapping.get(x[2], x)
            elif names.isdisjoint(free_vars(x)):
                r = x
            else:
                r = rebuild(x, [go(c) for c in children(x)])
            cache[x] = r
        return r

    return go(e)


_DATA_NODES: dict = {}


def has_data_nodes(e) -> bool:
    """e contains a metadata load or a thread table (per-thread data, not arithmetic)."""
    r = _DATA_NODES.get(e)
    if r is None:
        r = _DATA_NODES[e] = e[0] in ("load", "table") or any(
            map(has_data_nodes, children(e))
        )
    return r


# ---------------------------------------------------------------- KIR JSON


KIR_WIDTHS = (1, 32, 64)
_NARROW_CACHE: dict = {}


def _is_narrow(w: int) -> bool:
    return w not in KIR_WIDTHS


def has_narrow(e) -> bool:
    """e has an integer node of a width KIR lacks (i8, i16: narrower than 32, not 1)."""
    r = _NARROW_CACHE.get(e)
    if r is None:
        r = _is_narrow(width(e)) or any(has_narrow(c) for c in children(e))
        _NARROW_CACHE[e] = r
    return r


def _sx(a, w: int):
    """The i32 sign extension of the w-bit value whose zero extension is the i32 `a`."""
    k = const(32 - w, 32)
    return bin("ashr", bin("shl", a, k), k)


def _mask(a, w: int):
    return bin("and", a, const((1 << w) - 1, 32))


# narrow ops whose result, on zero-extended operands, is the zero-extended result
_ZEXT_CLOSED = ("and", "or", "xor", "udiv", "urem", "umin", "umax", "lshr")
_SIGNED = ("sdiv", "srem", "smin", "smax")


def lower_narrow(e):  # noqa: C901 (one case per expression kind)
    """e with every integer of a width KIR lacks (i8, i16) carried as its zero-extended
    i32 value (a variable of such a width holds that value too), so that only widths 1,
    32 and 64 remain. Exact: each narrow result is reduced modulo 2^w, and signed ops and
    comparisons see the sign extension."""
    memo: dict = {}

    def go(x):
        r = memo.get(x)
        if r is not None:
            return r
        k, w = x[0], width(x)
        if k == "const":
            r = const(_unsigned(x[2], w), 32) if _is_narrow(w) else x
        elif k == "var":
            r = var(x[2], 32) if _is_narrow(w) else x
        elif k == "bin":
            r = _lower_bin(x[2], w, go(x[3]), go(x[4]))
        elif k == "cmp":
            wa = width(x[2])
            a, b = go(x[2]), go(x[3])
            if _is_narrow(wa) and x[1] in ("slt", "sle", "sgt", "sge"):
                a, b = _sx(a, wa), _sx(b, wa)
            r = cmp(x[1], a, b)
        elif k == "select":
            r = select(go(x[2]), go(x[3]), go(x[4]))
        elif k == "ext":
            r = _lower_ext(go(x[3]), width(x[3]), w, x[2])
        elif k == "trunc":
            a = go(x[2])
            if _is_narrow(w):
                r = _mask(trunc(a, 32) if width(a) == 64 else a, w)
            else:
                r = trunc(a, w)
        elif k in ("pid", "nprog", "load", "table"):
            if _is_narrow(w):
                raise ExprError(f"{k} of width {w}")
            r = x
        else:
            raise ExprError(f"unknown expression kind {k}")
        memo[x] = r
        return r

    return go(e)


def _lower_bin(o: str, w: int, a, b):
    if not _is_narrow(w):
        return bin(o, a, b)
    if o in _ZEXT_CLOSED:
        return bin(o, a, b)
    if o in _SIGNED:
        return _mask(bin(o, _sx(a, w), _sx(b, w)), w)
    if o == "ashr":
        return _mask(bin(o, _sx(a, w), b), w)
    return _mask(bin(o, a, b), w)  # add, sub, mul, shl: modulo 2^w


def _lower_ext(a, wa: int, w: int, signed: bool):
    """ext of the (lowered) wa-bit value `a` to w bits."""
    if _is_narrow(wa):  # `a` is the i32 zero extension
        v = _sx(a, wa) if signed else a
    elif _is_narrow(w):  # i1 to a narrow width: its i32 extension, reduced
        return _mask(ext(a, 32, signed), w)
    else:
        return ext(a, w, signed)
    if _is_narrow(w):
        return _mask(v, w)
    return v if w == 32 else ext(v, w, signed)


def to_json(e, names: dict | None = None, defining: str | None = None) -> dict:
    """KIR JSON of a thread-uniform expression. A `table` with a non-constant key (thread
    loops) becomes a select chain over its entries. `names` maps shared nodes to the
    variables that hold them; `defining` is the variable being defined by e itself.
    Integers of widths KIR lacks are carried as zero-extended i32 (lower_narrow)."""
    memo: dict = {}
    names = names or {}
    if has_narrow(e):
        e, names = lower_narrow(e), {}

    def go(x):
        r = memo.get(x)
        if r is not None:
            return r
        n = names.get(x)
        if n is not None and n != defining:
            r = memo[x] = {"e": "var", "w": width(x), "name": n}
            return r
        k = x[0]
        if k == "const":
            r = {"e": "const", "w": x[1], "v": x[2]}
        elif k == "var":
            if x[2] in THREAD_VARS:
                raise ExprError(f"thread variable {x[2]} where KIR needs a value")
            if x[2].startswith("%"):
                raise ExprError(f"reserved value {x[2]} where KIR needs a value")
            r = {"e": "var", "w": x[1], "name": x[2]}
        elif k in ("pid", "nprog"):
            r = {"e": k, "w": x[1], "axis": x[2]}
        elif k == "bin":
            r = {"e": "bin", "w": x[1], "o": x[2], "a": go(x[3]), "b": go(x[4])}
        elif k == "cmp":
            r = {"e": "cmp", "w": 1, "o": x[1], "a": go(x[2]), "b": go(x[3])}
        elif k == "select":
            r = {"e": "select", "w": x[1], "c": go(x[2]), "a": go(x[3]), "b": go(x[4])}
        elif k == "ext":
            r = {"e": "ext", "w": x[1], "signed": x[2], "a": go(x[3])}
        elif k == "trunc":
            r = {"e": "trunc", "w": x[1], "a": go(x[2])}
        elif k == "load":
            r = {"e": "load", "w": x[1], "tensor": x[2], "index": [go(x[3][0])]}
        elif k == "table":
            key, entries = x[2], x[3]
            r = go(entries[-1])
            for i in range(len(entries) - 2, -1, -1):
                c = go(cmp("eq", key, const(i, width(key))))
                r = {"e": "select", "w": x[1], "c": c, "a": go(entries[i]), "b": r}
        else:
            raise ExprError(f"unknown expression kind {k}")
        memo[x] = r
        return r

    return go(e)
