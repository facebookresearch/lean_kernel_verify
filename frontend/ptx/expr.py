# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Integer expressions of the PTX frontend (KIR `Expr`, references/kir.md).

A copy of frontend/cute/expr.py (kept unchanged for the legacy CuTe translation)
plus the exact expansions of PTX integer instructions into KIR nodes (the end of this
file: mul.hi/mul.wide, bfe, bfi, prmt and its modes, lop3, popc, clz, brev, shf, cvt
widths and saturation, add/sub.sat, mul24, sad), known
trailing zeros (tz), exact division by a constant (div_exact), the nodes a `let` must
not hoist out of a `select` arm (unsafe), and a few folds that keep thread offsets
affine (`(t/32)*32 + t%32 = t`, masks and ors that cannot change a value) and predicates
small (`x != 0` for an x with one possible one bit is that bit: LLVM's packed predicate
masks). Every fold is an identity of BitVec arithmetic: none changes a value.

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

Thread-varying values are expressions over reserved variables: LANE (0..31), WARP
(absolute warp index) and the elect symbols elect(k) (i1, true on the lane the k-th
`elect.sync` of a role picks in each warp; the ISA does not say which lane, nor that two
executions pick the same one). They never reach KIR as such: interp.py substitutes the threads of a role.
Other reserved `%` names (the address bases `%g:<tensor>` and `%s:<symbol>` of mem.py)
are resolved before KIR too. Folding is exact BitVec arithmetic, so it never changes a
value.
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


def elect(k: int):
    """The elect symbol of the k-th elect.sync (k = 0 is ELECT)."""
    return var(f"%elect{k}" if k else "%elect", 1)


def is_thread_var(name: str) -> bool:
    return name in ("%lane", "%warp") or name.startswith("%elect")


def is_elect(e) -> bool:
    return e[0] == "var" and e[2].startswith("%elect")


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
    if (
        op == "xor"
        and is_const(b)
        and a[0] == "bin"
        and a[2] == "xor"
        and is_const(a[4])
    ):
        return bin("xor", a[3], const(a[4][2] ^ b[2], w))  # e.g. not(not(p)) = p
    if op == "xor" and a is b:
        return const(0, w)
    folded = _fold_ptx(op, w, a, b)
    if folded is not None:
        return folded
    return _mk("bin", w, op, a, b)


def _fold_ptx(op: str, w: int, a, b):  # noqa: C901 (one case per identity)
    """Identities that keep LLVM's address arithmetic readable and affine (it writes
    tid & 127 for a tid known to be below 128, x + x for 2x, or for a disjoint add);
    each holds for every value of the operands."""
    if op == "add" and a is b and w > 1:
        return bin("mul", a, const(2, w))
    if op == "and" and is_const(b) and b[2] > 0 and b[2] & (b[2] + 1) == 0:
        ia = interval(a)  # a mask of low bits that keeps every bit a can have
        if ia and 0 <= ia[0] and ia[1] <= b[2]:
            return a
    if op == "or" and w > 1:
        for x, y in ((a, b), (b, a)):
            iy = interval(y)
            if iy and iy[0] >= 0 and iy[1] < (1 << min(tz(x), w - 1)):
                return bin("add", x, y)  # disjoint bits: or is add
            if is_const(y) and x[0] == "bin" and x[2] == "add" and is_const(x[4]):
                # x = z + k, k below z's trailing zeros: x's low bits are k's
                top = 1 << min(tz(x[3]), w - 1)
                k, c = _unsigned(x[4][2], w), _unsigned(y[2], w)
                if k < top and c < top and not k & c:
                    return bin("add", x, y)
    if op == "add":
        for x, y in ((a, b), (b, a)):
            # (t udiv c) * c + t urem c = t, for a positive constant c
            if (
                x[0] == "bin" and x[2] == "mul" and is_const(x[4]) and x[4][2] > 0
                and x[3][0] == "bin" and x[3][2] == "udiv" and x[3][4] is x[4]
                and y[0] == "bin" and y[2] == "urem" and y[3] is x[3][3] and y[4] is x[4]
            ):  # fmt: skip
                return y[3]
    if op == "shl" and is_const(b) and 0 < b[2] < w:
        # x << k reads the low w - k bits of x, and LLVM keeps only those of a constant
        # added to x (demanded bits: (n - 2) << 7 becomes (n + 0x1FFFFFE) << 7), which
        # then wraps where the source's does not; the constant closest to 0 with the same
        # low bits gives the same value
        x = _closest_const(a, w - b[2], set())
        if x is not None:
            return bin("shl", x, b)
    return None


def _closest_const(e, bits: int, seen: set):
    """e, an add/sub tree, with its first constant operand whose value is not the one
    closest to 0 among those of the same low `bits` bits replaced by that one (the same
    value modulo 2**bits); None if it has no such constant."""
    if e[0] != "bin" or e[2] not in ("add", "sub") or e in seen:
        return None
    seen.add(e)
    for k in (3, 4):
        x = e[k]
        if is_const(x):
            c = _wrap(x[2], bits)
            y = None if c == x[2] else const(c, width(e))
        else:
            y = _closest_const(x, bits, seen)
        if y is not None:
            return bin(e[2], *((y, e[4]) if k == 3 else (e[3], y)))
    return None


def tz(e, facts: dict | None = None) -> int:
    """Known trailing zero bits of e (0 if nothing is known): every value of e is a
    multiple of 2**tz(e). `facts` gives them for variables (loop-carried slots, mem.py)."""
    k, w = e[0], width(e)
    if k == "const":
        v = _unsigned(e[2], w)
        return w if v == 0 else (v & -v).bit_length() - 1
    if k == "var":
        return min((facts or {}).get(e[2], 0), w)
    if k == "ext":
        return tz(e[3], facts) if width(e[3]) > 1 else 0
    if k == "trunc":
        return min(tz(e[2], facts), w)
    if k == "select":
        return min(tz(e[3], facts), tz(e[4], facts))
    if k == "table":  # one of its entries
        return min(tz(x, facts) for x in e[3])
    if k != "bin":
        return 0
    op, x, y = e[2], e[3], e[4]
    if op in ("add", "sub", "or", "xor", "smin", "smax", "umin", "umax"):
        return min(tz(x, facts), tz(y, facts))
    if op == "and":
        return max(tz(x, facts), tz(y, facts))
    if op == "mul":
        return min(tz(x, facts) + tz(y, facts), w)
    if op == "shl" and is_const(y):
        return min(tz(x, facts) + _unsigned(y[2], w), w)
    return 0


def _fits(e) -> bool:
    """e's value is the same as on unbounded integers (its interval is known)."""
    return interval(e) is not None


def div_exact(e, k: int, facts: dict | None = None):
    """e / k (a positive constant) as an expression, if k provably divides every value of
    e, else None. Distributes over add/sub and cancels against mul/shl constants only
    where the node cannot wrap (interval known), so the quotient is the exact one; else
    it is sdiv(e, k), exact on a multiple of k (known trailing zeros, power-of-two k)."""
    w = width(e)
    if k == 1:
        return e
    if is_const(e):
        return const(e[2] // k, w) if e[2] % k == 0 else None
    if e[0] == "bin" and _fits(e):
        op, a, b = e[2], e[3], e[4]
        if op in ("add", "sub"):
            qa, qb = div_exact(a, k, facts), div_exact(b, k, facts)
            if qa is not None and qb is not None:
                return bin(op, qa, qb)
        if op == "mul" and is_const(b):
            if b[2] % k == 0:
                return bin("mul", a, const(b[2] // k, w))
            qa = div_exact(a, k, facts)
            if qa is not None:
                return bin("mul", qa, b)
        if op == "shl" and is_const(b) and 0 <= b[2] < w and (1 << b[2]) % k == 0:
            return bin("mul", a, const((1 << b[2]) // k, w))
    if k & (k - 1) == 0 and tz(e, facts) >= k.bit_length() - 1:
        return bin("sdiv", e, const(k, w))
    return None


_UNSAFE: dict = {}


def unsafe(e) -> bool:
    """e has a node whose evaluation can stop the run or report a violation on its own: a
    metadata load (out of range: P0-1) or a division by a value that may be zero. A `let`
    of such a node must stay where it is evaluated (not hoisted out of a select arm, which
    KIR evaluates only when chosen, or above the `if` of a thread loop)."""
    r = _UNSAFE.get(e)
    if r is None:
        r = (
            e[0] == "load"
            or (e[0] == "bin" and e[2] in PARTIAL_OPS and not _nonzero_const(e[4]))
            or any(map(unsafe, children(e)))
        )
        _UNSAFE[e] = r
    return r


def _nonzero_const(e) -> bool:
    return is_const(e) and e[2] != 0


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
    if k == "cmp" or is_elect(e):
        return 0, 1
    if e is LANE:
        return 0, 31
    if k == "select":
        ia, ib = interval(e[3]), interval(e[4])
        return (min(ia[0], ib[0]), max(ia[1], ib[1])) if ia and ib else None
    if k == "ext":
        ia, wa = interval(e[3]), width(e[3])
        if wa == 1:
            return (-1, 0) if e[2] else (0, 1)
        if ia is None:  # every value of the source width
            return (
                (-(1 << (wa - 1)), (1 << (wa - 1)) - 1) if e[2] else (0, (1 << wa) - 1)
            )
        return ia if e[2] or ia[0] >= 0 else (0, (1 << wa) - 1)
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
        if ia and ia[0] >= 0:
            return ia[0] >> e[4][2], ia[1] >> e[4][2]
        return 0, (1 << (w - e[4][2])) - 1
    if op == "shl" and is_const(e[4]) and 0 <= e[4][2] < w and ia:
        return ia[0] << e[4][2], ia[1] << e[4][2]  # None below if it leaves the width
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
    if op in ("eq", "ne") and is_const(b, 0):
        nz = _nonzero_bit(a)
        if nz is not None:
            return nz if op == "ne" else not1(nz)
    return _mk("cmp", op, a, b)


# Known-one bits: LLVM packs predicates into an i16 mask (selp of -1/1, shl, and, disjoint
# adds) and tests one bit of it per predicated load; the test is that predicate.
_ONES: dict = {}
_BITS: dict = {}


def _ones(e) -> int:
    """The bits of e that may be 1 (an unsigned mask of its width)."""
    r = _ONES.get(e)
    if r is None:
        w, k = width(e), e[0]
        full = (1 << w) - 1
        r = full
        if k == "const":
            r = _unsigned(e[2], w)
        elif k == "select":
            r = _ones(e[3]) | _ones(e[4])
        elif k == "ext" and not e[2]:
            r = _ones(e[3])
        elif k == "trunc":
            r = _ones(e[2]) & full
        elif k == "bin":
            op, x, y = e[2], e[3], e[4]
            ox, oy = _ones(x), _ones(y)
            if op == "and":
                r = ox & oy
            elif op in ("or", "xor") or op == "add" and not ox & oy:
                r = ox | oy  # disjoint bits: add is or
            elif op in ("shl", "lshr") and is_const(y):
                s = _unsigned(y[2], w)
                r = (ox << s) & full if op == "shl" else ox >> s
        _ONES[e] = r
    return r


def _bit(e, k: int):
    """Bit k of e as an i1 expression through bit operations (_ones' nodes), or None."""
    r = _BITS.get((e, k), _BITS)
    if r is not _BITS:
        return r
    r, kind = None, e[0]
    if not (_ones(e) >> k) & 1:
        r = const(0, 1)
    elif kind == "const":
        r = const(1, 1)
    elif kind == "cmp":
        r = e
    elif kind == "select":
        x, y = _bit(e[3], k), _bit(e[4], k)
        r = None if x is None or y is None else select(e[2], x, y)
    elif kind == "ext" and not e[2]:
        r = _bit(e[3], k)
    elif kind == "trunc":
        r = _bit(e[2], k)
    elif kind == "bin" and e[2] in ("shl", "lshr") and is_const(e[4]):
        s = _unsigned(e[4][2], e[1])
        r = _bit(e[3], k - s if e[2] == "shl" else k + s)
    elif kind == "bin" and e[2] in ("and", "or", "xor", "add"):
        x, y = _bit(e[3], k), _bit(e[4], k)  # add: disjoint bits (_ones), an or
        if x is not None and y is not None:
            r = bin({"add": "or"}.get(e[2], e[2]), x, y)
    _BITS[(e, k)] = r
    return r


def _nonzero_bit(e):
    """e != 0 as an i1 expression when at most one bit of e may be 1 and _bit derives it,
    else None."""
    o = _ones(e)
    if o & (o - 1):
        return None
    return const(0, 1) if o == 0 else _bit(e, o.bit_length() - 1)


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
    return any(map(is_thread_var, free_vars(e)))


def reserved(e) -> bool:
    """e uses a reserved variable other than the thread variables (a `%` name the frontend
    resolves before KIR, e.g. a fast divmod auxiliary value)."""
    return any(v.startswith("%") and not is_thread_var(v) for v in free_vars(e))


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
            if is_thread_var(x[2]):
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
            r = go(_table_select(x[2], x[3]))
        else:
            raise ExprError(f"unknown expression kind {k}")
        memo[x] = r
        return r

    return go(e)


def _table_select(key, entries: tuple):
    """entries[key] (key in [0, len(entries)), a lane or thread index) as selects over
    the distinct entries, the most frequent one the default: each other entry on the keys
    holding it, `(key urem p) == r` when they are every p-th from r, else `key == i` for
    each, or'd (a lane table of one value on every fourth lane is one select, not 31)."""
    n, w = len(entries), width(key)
    keys: dict = {}
    for i, x in enumerate(entries):
        keys.setdefault(x, []).append(i)
    default = max(keys, key=lambda x: (len(keys[x]), keys[x][-1]))
    r = default
    for x, idx in reversed(list(keys.items())):
        if x is default:
            continue
        p = idx[1] - idx[0] if len(idx) > 1 else n
        if p < n and n % p == 0 and idx == list(range(idx[0], n, p)):
            c = cmp("eq", bin("urem", key, const(p, w)), const(idx[0], w))
        else:
            c = cmp("eq", key, const(idx[0], w))
            for i in idx[1:]:
                c = bin("or", c, cmp("eq", key, const(i, w)))
        r = select(c, x, r)
    return r


# ------------------------------------------------------- PTX integer instructions
#
# Exact expansions (PTX ISA, Integer Arithmetic and Logic Instructions) into the KIR
# nodes above. Operands are already at the instruction's width; a constant field
# operand (bit position, length, selector, LUT) is a Python int, and a runtime one is
# refused by the caller (interp.py).


def mul_wide(a, b, signed: bool):
    """mul.wide: the double-width product."""
    w = 2 * width(a)
    return bin("mul", ext(a, w, signed), ext(b, w, signed))


def mul_hi(a, b, signed: bool):
    """mul.hi: the high half of the double-width product. Address arithmetic marks every
    add/sub/mul/shl a value passes through and Lean reports their signed wraps, so each
    width is computed where no such node can wrap, or in the one form translate.mark_addr
    skips:
      32 bits: trunc(lshr(mul(ext a, ext b), 32), 32), the form mark_addr skips (its
               unsigned product may pass 2**63);
      16 bits: the product at 64 bits (|p| < 2**32) shifted by 16, arithmetically when
               signed, so that the truncation never drops bits;
      64 bits: KIR has no i128: x udiv d when that is provably the same (_magic_div:
               LLVM's division by a constant), else the product of 16-bit limbs
               (_mul_hi64), whose tree repeats a 16 times."""
    w = width(a)
    if w == 64:
        x, m = (b, a) if is_const(a) else (a, b)
        q = _magic_div(x, m, signed) if is_const(m) else None
        return q if q is not None else _mul_hi64(a, b, signed)
    if w == 32:
        return trunc(bin("lshr", mul_wide(a, b, signed), const(w, 2 * w)), w)
    p = bin("mul", ext(a, 64, signed), ext(b, 64, signed))
    return trunc(bin("ashr" if signed else "lshr", p, const(w, 64)), w)


def _magic_div(x, m, signed: bool):
    """(x << k) udiv d if that is mul.hi(x, m) (64 bits) for every value of x, else None
    (LLVM divides by a constant c as mul.hi(x >> s, m) >> t, m ~ 2**(64 + t) / (c >> s)).
    x's interval is [0, X] (so a signed mul.hi with m >= 0 is the unsigned one); for the
    least k with X << k below 2**63, d = ceil(2**(64 + k) / m) and e = m d - 2**(64 + k)
    has X e < 2**64. Then x m / 2**64 = (x << k) / d + x e / (d 2**64), and the second
    term is below 1/d: the floors agree (Granlund and Montgomery, Division by Invariant
    Integers using Multiplication, 1994). The shift cannot wrap (address marks)."""
    iv, mu = interval(x), _unsigned(m[2], 64)
    if iv is None or iv[0] < 0 or mu == 0 or (signed and m[2] < 0):
        return None
    for k in range(64 - iv[1].bit_length()):
        d = -(-(1 << (64 + k)) // mu)
        if iv[1] * (mu * d - (1 << (64 + k))) < 1 << 64:
            if d >= 1 << 63:
                return None
            return bin("udiv", bin("shl", x, const(k, 64)), const(d, 64))
    return None


def _mul_hi64(a, b, signed: bool):
    """The high 64 bits of the 128-bit product of a and b, from four 16-bit limbs each
    (the top one signed when `signed`), with no add, mul or shl that wraps as a signed
    i64: every partial product and column sum stays below 2**35 in magnitude, the carry
    is an exact floor shift, and the result is (top digit << 48) + the 48 bits below it,
    the top digit taken as signed (an unsigned one through a sign extension, the
    ashr(shl) form mark_addr skips) so that its shift cannot wrap either."""
    m16, sh = const(0xFFFF, 64), "ashr" if signed else "lshr"

    def limbs(x):
        low = [bin("and", bin("lshr", x, const(16 * k, 64)), m16) for k in range(3)]
        return low + [bin(sh, x, const(48, 64))]

    la, lb = limbs(a), limbs(b)
    digits, carry = [], const(0, 64)
    for k in range(7):  # column k of the product has weight 2**(16k)
        col = carry
        for i in range(max(0, k - 3), min(k, 3) + 1):
            col = bin("add", col, bin("mul", la[i], lb[k - i]))
        digits.append(bin("and", col, m16))
        carry = bin(sh, col, const(16, 64))
    low48 = bin("add", digits[4], bin("add", bin("shl", digits[5], const(16, 64)),
                                      bin("shl", digits[6], const(32, 64))))  # fmt: skip
    top = carry if signed else _sx64(carry, 16)
    return bin("add", bin("shl", top, const(48, 64)), low48)


def _sx64(a, w: int):
    """The i64 sign extension of the w-bit value whose zero extension is the i64 `a`."""
    k = const(64 - w, 64)
    return bin("ashr", bin("shl", a, k), k)


def popc(a):
    """popc: the number of set bits (the result is 32 bits wide)."""
    w = width(a)
    total = const(0, w)
    for k in range(w):
        total = bin("add", total, bin("and", bin("lshr", a, const(k, w)), const(1, w)))
    return resize(total, 32, False)


def clz(a):
    """clz: leading zero bits = the number of j in 1..w with a < 2**(w - j) (unsigned)."""
    w = width(a)
    total = const(0, 32)
    for j in range(1, w + 1):
        total = bin("add", total, ext(cmp("ult", a, const(1 << (w - j), w)), 32, False))
    return total


def brev(a):
    """brev: the bits of a in reverse order."""
    w = width(a)
    out = const(0, w)
    for k in range(w):
        bit = bin("and", bin("lshr", a, const(k, w)), const(1, w))
        out = bin("or", out, shl_bits(bit, w - 1 - k))
    return out


def bfe(a, pos: int, length: int, signed: bool):
    """bfe with constant position and length (PTX: both are taken modulo 256)."""
    w = width(a)
    pos, length = pos & 255, length & 255
    if length == 0:
        return const(0, w)
    if not signed:
        if pos >= w:
            return const(0, w)
        n = min(length, w - pos)
        return bin("and", bin("lshr", a, const(pos, w)), const((1 << n) - 1, w))
    # Sign bit: bit min(pos + len - 1, w - 1) of a; bits above the field copy it.
    top = min(pos + length - 1, w - 1)
    if pos >= w:
        return bin("ashr", a, const(w - 1, w))
    up = w - 1 - top
    return bin("ashr", bin("shl", a, const(up, w)), const(up + pos, w))


def bfi(a, b, pos: int, length: int):
    """bfi: b with bits [pos, pos + len) replaced by the low bits of a (constant pos and
    len, taken modulo 256)."""
    w = width(a)
    pos, length = pos & 255, length & 255
    if length == 0 or pos >= w:
        return b
    n = min(length, w - pos)
    field = ((1 << n) - 1) << pos
    keep = bin("and", b, const(~field, w))
    return bin("or", keep, bin("and", shl_bits(a, pos), const(field, w)))


def prmt(a, b, sel: int):
    """prmt.b32 default mode with a constant selector: result byte i is byte (sel >> 4i)
    & 7 of {b, a} (a is bytes 0-3), replicated from its sign bit when (sel >> 4i) & 8."""
    out = const(0, 32)
    for i in range(4):
        s = (sel >> (4 * i)) & 15
        src = a if (s & 7) < 4 else b
        byte = bin("and", bin("lshr", src, const(8 * (s & 3), 32)), const(255, 32))
        if s & 8:  # the byte's top bit, copied into all 8 bits
            byte = bin("mul", bin("lshr", byte, const(7, 32)), const(255, 32))
        out = bin("or", out, shl_bits(byte, 8 * i))
    return out


def lop3(a, b, c, lut: int):
    """lop3: the bitwise function of (a, b, c) whose truth table is lut (bit index
    4a + 2b + c, per bit position), as an or of its minterms."""
    w = width(a)
    ones = const(-1, w)
    out = const(0, w)
    for m in range(8):
        if not (lut >> m) & 1:
            continue
        term = ones
        for bit, x in ((4, a), (2, b), (1, c)):
            term = bin("and", term, x if m & bit else bin("xor", x, ones))
        out = bin("or", out, term)
    return out


def shf(a, b, n: int, left: bool, wrap: bool):
    """shf.l/shf.r on 32-bit halves (b:a is the 64-bit value) by a constant amount, taken
    modulo 32 (.wrap) or clamped to 32 (.clamp)."""
    n = n & 31 if wrap else min(n & 0xFFFFFFFF, 32)
    if n == 0:
        return b if left else a
    if n == 32:
        return a if left else b
    if left:  # the high word of (b:a) << n
        return bin("or", shl_bits(b, n), bin("lshr", a, const(32 - n, 32)))
    return bin("or", bin("lshr", a, const(n, 32)), shl_bits(b, 32 - n))


def cvt_int(a, dst_bits: int, src_bits: int, src_signed: bool):
    """cvt between integer types: the low src_bits of a, extended by the source type's
    signedness (or truncated) to dst_bits."""
    v = resize(a, src_bits, False)
    return resize(v, dst_bits, src_signed)


def lanes16(a):
    """The two 16-bit lanes of a packed 32-bit value (ISA: .s16x2/.u16x2), low first."""
    return [trunc(a, 16), trunc(bin("lshr", a, const(16, 32)), 16)]


def shl_bits(a, k: int):
    """a << k for a constant 0 <= k < w, a move of bits (the ones shifted out are dropped
    by design: brev, bfi, prmt, shf, packing): unless a's interval shows the shift does
    not wrap, the shift of the sign extension of a's low w - k bits, the same bits, so
    that no node wraps as a signed integer (address arithmetic;
    translate._sext_operand skips the extension)."""
    if k == 0:
        return a
    w, c = width(a), const(k, width(a))
    ia = interval(a)
    if ia is not None and -(1 << (w - 1)) <= ia[0] << k and ia[1] << k < 1 << (w - 1):
        return bin("shl", a, c)
    return bin("shl", bin("ashr", bin("shl", a, c), c), c)


def pack16(lo, hi):
    """Two 16-bit lanes as one 32-bit value. The high lane is shifted from its sign
    extension and the low one added: the same bits as the unsigned (hi << 16) | lo, but
    neither node wraps as a signed i32 (address arithmetic)."""
    return bin("add", bin("shl", ext(hi, 32, True), const(16, 32)), ext(lo, 32, False))


def bfind(a, signed: bool, shiftamt: bool):
    """bfind: the position of the most significant non-sign bit of a (the top 1, or the
    top 0 of a negative signed a), with .shiftamt the left shift that moves it to the top
    bit instead; 0xFFFFFFFF when there is none. The result is 32 bits wide."""
    w = width(a)
    x = (
        select(cmp("slt", a, const(0, w)), bin("xor", a, const(-1, w)), a)
        if signed
        else a
    )
    n = clz(x)
    r = n if shiftamt else bin("sub", const(w - 1, 32), n)
    return select(cmp("eq", x, const(0, w)), const(-1, 32), r)


def dp2a(a, b, c, hi: bool, a_signed: bool, b_signed: bool):
    """dp2a.lo/.hi: c + a.h0 * b.b(k) + a.h1 * b.b(k + 1), k = 2 for .hi else 0: the two
    16-bit halves of a times two bytes of b, each extended by its operand's type (32-bit,
    wrapping). The halves and bytes are bit fields (bfe), whose products cannot wrap."""
    k = 2 if hi else 0
    for j in range(2):
        h = bfe(a, 16 * j, 16, a_signed)
        byte = bfe(b, 8 * (k + j), 8, b_signed)
        c = bin("add", c, bin("mul", h, byte))
    return c


def add_sat(a, b, sub: bool):
    """add.sat/sub.sat (.s32 only): the exact sum or difference at 64 bits clamped to
    the signed range of the width (ISA add, sub: .sat limits the result to MININT..
    MAXINT); no node wraps."""
    w = width(a)
    s = bin("sub" if sub else "add", ext(a, 64, True), ext(b, 64, True))
    s = bin("smin", s, const((1 << (w - 1)) - 1, 64))
    return trunc(bin("smax", s, const(-(1 << (w - 1)), 64)), w)


def mul24(a, b, hi: bool, signed: bool):
    """mul24.lo/.hi (.u32/.s32): bits 0-31 or 16-47 of the 48-bit product of the low 24
    bits of a and b, sign-extended by .s32 (ptxas: SGXT 24, else a byte mask). .lo is
    the 32-bit product of the two fields, whose wrap is mul.lo's; .hi is shifted from
    the product at 64 bits (|p| < 2**48), so only its truncation can lose bits: when the
    unsigned result is 2**31 or more."""
    fa, fb = bfe(a, 0, 24, signed), bfe(b, 0, 24, signed)
    if not hi:
        return bin("mul", fa, fb)
    p = bin("mul", ext(fa, 64, signed), ext(fb, 64, signed))
    return trunc(bin("ashr", p, const(16, 64)), 32)


def sad(a, b, c, signed: bool):
    """sad: c + |a - b| (ISA sad: c + ((a < b) ? b - a : a - b), wrapping), compared by
    the type's signedness. Below 64 bits it is computed at 64, where nothing wraps, and
    truncated once: an address mark reports only a sum that does not fit the width."""
    w = width(a)
    if w < 64:
        a, b, c = (ext(x, 64, signed) for x in (a, b, c))
    lt = cmp("slt" if signed else "ult", a, b)
    r = bin("add", c, select(lt, bin("sub", b, a), bin("sub", a, b)))
    return trunc(r, w) if w < 64 else r


def cvt_sat(a, dst_bits: int, dst_signed: bool, src_bits: int, src_signed: bool):
    """cvt.sat between integer types: the source value (the low src_bits of a, read by
    the source type) clamped to the destination type's range (ISA cvt: .sat limits the
    result to MININT..MAXINT of the destination size, signed or unsigned), at
    dst_bits."""
    v = resize(a, src_bits, False)
    lo = -(1 << (dst_bits - 1)) if dst_signed else 0
    hi = (1 << (dst_bits - 1 if dst_signed else dst_bits)) - 1
    if src_signed:
        if lo > -(1 << (src_bits - 1)):
            v = bin("smax", v, const(lo, src_bits))
        if hi < (1 << (src_bits - 1)) - 1:
            v = bin("smin", v, const(hi, src_bits))
    elif hi < (1 << src_bits) - 1:
        v = bin("umin", v, const(hi, src_bits))
    return resize(v, dst_bits, src_signed)


# prmt's modes (ISA prmt): c[1:0] -> the default-mode selector that picks the same bytes
# (nibble i: the source byte of d.b<i>, so each reads as the ISA's row d.b3 ... d.b0)
_PRMT_MODES = {
    "f4e": (0x3210, 0x4321, 0x5432, 0x6543),
    "b4e": (0x5670, 0x6701, 0x7012, 0x0123),
    "rc8": (0x0000, 0x1111, 0x2222, 0x3333),
    "ecl": (0x3210, 0x3211, 0x3222, 0x3333),
    "ecr": (0x0000, 0x1110, 0x2210, 0x3210),
    "rc16": (0x1010, 0x3232, 0x1010, 0x3232),
}


def prmt_mode(a, b, c, mode: str):
    """prmt.b32.<mode>: the default-mode prmt whose selector the mode and c[1:0] give
    (no sign replication); a runtime c is a select over its four values."""
    sels = _PRMT_MODES[mode]
    if is_const(c):
        return prmt(a, b, sels[c[2] & 3])
    k = bin("and", c, const(3, width(c)))
    out = prmt(a, b, sels[3])
    for v in (2, 1, 0):
        out = select(cmp("eq", k, const(v, width(c))), prmt(a, b, sels[v]), out)
    return out
