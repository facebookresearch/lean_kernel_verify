# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""KIR variable names for the SSA values of a role (references/frontend_ttgir.md, "KIR
produced", Statements).

The translation names the KIR variable of an SSA value by a token (`token(i)` for the
Translator's value i), plus a suffix for the value's temporaries (`.next`, `.in`,
`.shape.0`, ...). `name_role` then replaces the tokens of a finished role body. A value
keeps its printed SSA name unless sharing that name with another value's variable would
not be exact. In that case it is named `<printed name>@<TTGIR line>` (the line that defines
it); no printed name contains `@`, so that name is fresh.

Why names can collide: MLIR's printer numbers values per region, and sibling regions start
from the same counters. The before and after regions of an scf.while, the branches of an
scf.if and consecutive loop bodies all reuse `%N`, `%argN` and `name_N`. The translation
assigns some values in an enclosing block, outside their region: an scf.while before region
and its arguments (before the loop, and again at the end of its body), loop-carried values
(before the loop), and the default partition's values (at the role's top level). So a
variable can still be visible where another value with the same printed name is assigned.

Sharing a variable is exact (the coalescing condition of SSA destruction) if neither
variable is live where the other is assigned (no path from that assignment reads it before
its own next assignment; liveness over every syntactic path of the role body), and,
wherever the two are visible at once, they have the same width and neither is a `for`
variable. Visibility follows kir.md: a `let` creates a variable that stays visible to the
end of its block, a `for` variable is visible in its body, and a `let` may overwrite only a
visible variable of its own width that is not a `for` variable.
"""

from typing import Any

TOKEN = "\x00"
NESTED = ("body", "then", "else")


def token(i: int) -> str:
    """The variable name of value i until `name_role` names it."""
    return f"{TOKEN}{i}{TOKEN}"


def _split(name: str) -> tuple[int, str] | None:
    """(value index, suffix) of a tokenized variable name; None for other names."""
    if not name.startswith(TOKEN):
        return None
    end = name.index(TOKEN, 1)
    return int(name[1:end]), name[end + 1 :]


def _reads(x, out: set) -> None:
    """Names of the `var` Exprs in x (Exprs, TExprs, event fields; not nested statements)."""
    if isinstance(x, dict):
        if x.get("e") == "var":
            out.add(x["name"])
            return
        for k, v in x.items():
            if k not in NESTED:
                _reads(v, out)
    elif isinstance(x, list):
        for v in x:
            _reads(v, out)


class _Vars:
    """The variables of one role body: assignments, scopes (pre-order statement ranges
    where a binding is visible), widths and liveness."""

    def __init__(self, body: list):
        self.n = 0  # statements numbered so far (pre-order)
        self.bit: dict[str, int] = {}
        self.defs: dict[str, list] = {}  # name -> its let/for statements
        self.spans: dict[str, list] = {}  # name -> [(first, end)] where it is visible
        self.width: dict[str, int] = {}
        # name -> pre-order index of its first assignment
        self.first: dict[str, int] = {}
        self.for_vars: set[str] = set()
        # id(statement) -> bit mask of the names it reads
        self.read_mask: dict[int, int] = {}
        # id(let or for) -> bit mask of the names live right after it assigns
        self.live_after: dict[int, int] = {}
        self._scan(body, set())
        self._live(body, 0)

    def _mask(self, names) -> int:
        m = 0
        for x in names:
            m |= 1 << self.bit.setdefault(x, len(self.bit))
        return m

    def _scan(self, stmts: list, visible: set) -> None:
        visible = set(visible)
        created = []
        for s in stmts:
            i, op = self.n, s["op"]
            self.n += 1
            if id(s) in self.read_mask:
                raise ValueError(f"statement object used twice in a role body: {s}")
            names: set = set()
            _reads(s, names)
            self.read_mask[id(s)] = self._mask(names)
            if op in ("let", "for"):
                x = s["var"]
                self._mask([x])
                self.defs.setdefault(x, []).append(s)
                self.width.setdefault(x, (s["expr"] if op == "let" else s["lb"])["w"])
                self.first.setdefault(x, i)
            if op == "let" and s["var"] not in visible:
                visible.add(s["var"])
                created.append((s["var"], i))
            elif op == "for":
                self.for_vars.add(s["var"])
                new = s["var"] not in visible
                self._scan(s["body"], visible | {s["var"]})
                if new:
                    self.spans.setdefault(s["var"], []).append((i, self.n))
            else:
                for k in NESTED:
                    if k in s:
                        self._scan(s[k], visible)
        for x, i in created:
            self.spans.setdefault(x, []).append((i, self.n))

    def _live(self, stmts: list, out: int) -> int:
        """Names live before `stmts` given those live after (bit masks). Records, for every
        let and for, the names live right after it assigns (the last record of a loop body
        is the one of its fixpoint)."""
        for s in reversed(stmts):
            out = self._live_stmt(s, out)
        return out

    def _live_stmt(self, s: dict, out: int) -> int:
        op, read = s["op"], self.read_mask[id(s)]
        if op == "let":
            self.live_after[id(s)] = out
            return read | (out & ~(1 << self.bit[s["var"]]))
        if op == "if":
            return read | self._live(s["then"], out) | self._live(s["else"], out)
        if op == "while":  # head: test the condition, then the body or the exit
            head = read | out
            while True:
                new = read | out | self._live(s["body"], head)
                if new == head:
                    return head
                head = new
        if op == "for":
            # The bounds are read once; each trip assigns the variable, then runs the body.
            keep = ~(1 << self.bit[s["var"]])
            head = out
            while True:
                body = self._live(s["body"], head)
                new = out | (body & keep)
                if new == head:
                    break
                head = new
            self.live_after[id(s)] = body
            return read | head
        return read | out

    def _overlap(self, a: str, b: str) -> bool:
        return any(
            s1 < e2 and s2 < e1
            for s1, e1 in self.spans.get(a, ())
            for s2, e2 in self.spans.get(b, ())
        )

    def _clobbers(self, a: str, b: str) -> bool:
        """Some assignment of `a` happens while `b` is live."""
        bit = 1 << self.bit[b]
        return any(self.live_after[id(s)] & bit for s in self.defs.get(a, ()))

    def can_share(self, a: str, b: str) -> bool:
        """Whether variables `a` and `b` can be one KIR variable (module docstring)."""
        if self._clobbers(a, b) or self._clobbers(b, a):
            return False
        if not self._overlap(a, b):
            return True
        return self.width.get(a) == self.width.get(b) and not {a, b} & self.for_vars


def name_role(body: list, printed: list, lines: list, reserved: set) -> list:
    """`body` with every tokenized variable named (module docstring). printed[i] and
    lines[i] are value i's printed SSA name and TTGIR line; `reserved` (the params) and
    the role's other untokenized names are never given to a value."""
    v = _Vars(body)
    keys: dict[int, list] = {}  # value index -> [(variable, suffix)]
    taken: dict[str, list] = {n: [] for n in reserved}  # name -> variables named so
    for x in sorted(v.bit):
        sp = _split(x)
        if sp is None:
            taken.setdefault(x, [])
        else:
            keys.setdefault(sp[0], []).append((x, sp[1]))
    fixed = set(taken)

    def fits(i: int, base: str) -> bool:
        for x, suffix in keys[i]:
            full = base + suffix
            if full in fixed or not all(v.can_share(x, y) for y in taken.get(full, [])):
                return False
        return True

    final = {}
    for i in sorted(
        keys, key=lambda i: (min(v.first.get(x, v.n) for x, _ in keys[i]), i)
    ):
        base = next(
            (b for b in (printed[i], f"{printed[i]}@{lines[i]}") if fits(i, b)), None
        )
        if base is None:
            raise ValueError(
                f"no KIR variable name for %{printed[i]} (line {lines[i]})"
            )
        final[i] = base
        for x, suffix in keys[i]:
            taken.setdefault(base + suffix, []).append(x)
    return _rename(body, final)


def _rename(x: Any, final: dict) -> Any:
    if isinstance(x, list):
        return [_rename(y, final) for y in x]
    if not isinstance(x, dict):
        return x
    out = {k: _rename(y, final) for k, y in x.items()}
    if out.get("e") == "var":
        out["name"] = _final(out["name"], final)
    elif out.get("op") in ("let", "for"):
        out["var"] = _final(out["var"], final)
    return out


def _final(name: str, final: dict) -> str:
    sp = _split(name)
    return name if sp is None else final[sp[0]] + sp[1]
