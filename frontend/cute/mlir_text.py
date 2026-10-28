# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Reader for MLIR in generic operation form (`get_asm(print_generic_op_form=True)`).

Every op of the analyzed stage has the shape

    %r, %s:2 = "dialect.op"(%a, %b#1) <{properties}> ({regions}) {attributes}
               : (operand types) -> result types loc(...)

Types and attribute values are kept as text (balanced brackets, quoted strings);
`!alias` and `#locN` definitions are resolved on demand.

Why text and not the MLIR Python bindings: the CuTe DSL registers value casters that
wrap every `ir.Value` the bindings hand out in DSL objects, and some casters build new
IR ops when they run (walking a lowered kernel raises "No current Location" from
`cute.deref_arith_tuple_iter`). The printed generic form is the compiler's own
rendering and needs only this small reader. (The custom form does not round-trip:
the DSL prints `nvvm.read.ptx.sreg.*` ranges it cannot parse back.)
"""

import re
from dataclasses import dataclass, field


class MlirSyntaxError(Exception):
    pass


@dataclass
class Block:
    args: list  # [(name, type)]
    ops: list


@dataclass
class Op:
    name: str
    results: list  # value names; "%3#1" for result 1 of a multi-result op
    operands: list
    operand_types: list
    result_types: list
    props: dict
    attrs: dict
    regions: list = field(default_factory=list)  # [[Block]]
    loc: str = ""


_OPEN = {"(": ")", "[": "]", "{": "}", "<": ">"}
_VALUE = re.compile(r"%[A-Za-z0-9_$.\-]+(?:#\d+)?")
_RESULT = re.compile(r"%[A-Za-z0-9_$.\-]+")
_COUNT = re.compile(r":\d+")
_KEY = re.compile(r"[A-Za-z_][\w.$]*")
_LABEL = re.compile(r"\^\w+")
_LOC_SUFFIX = re.compile(r"\s+loc\(.*\)$", re.S)
_ALIAS = re.compile(r"![A-Za-z_][\w.]*")


class _Reader:
    def __init__(self, text: str):
        self.s = text
        self.i = 0
        self.n = len(text)

    def ws(self) -> None:
        s, n = self.s, self.n
        while self.i < n:
            c = s[self.i]
            if c in " \t\r\n":
                self.i += 1
            elif s.startswith("//", self.i):
                j = s.find("\n", self.i)
                self.i = n if j < 0 else j + 1
            else:
                break

    def peek(self, tok: str) -> bool:
        self.ws()
        return self.s.startswith(tok, self.i)

    def eat(self, tok: str) -> bool:
        if self.peek(tok):
            self.i += len(tok)
            return True
        return False

    def expect(self, tok: str) -> None:
        if not self.eat(tok):
            raise MlirSyntaxError(
                f"expected {tok!r} at offset {self.i}: {self.s[self.i : self.i + 80]!r}"
            )

    def match(self, pattern) -> str:
        self.ws()
        m = pattern.match(self.s, self.i)
        if not m:
            raise MlirSyntaxError(
                f"expected {pattern.pattern} at offset {self.i}: {self.s[self.i : self.i + 60]!r}"
            )
        self.i = m.end()
        return m.group(0)

    def string(self) -> str:
        """The double-quoted string at the cursor, quotes included."""
        s, j = self.s, self.i
        if s[j] != '"':
            raise MlirSyntaxError(f"expected a string at offset {j}")
        j += 1
        while s[j] != '"':
            j += 2 if s[j] == "\\" else 1
        out = s[self.i : j + 1]
        self.i = j + 1
        return out

    def balanced(self, stop: str) -> str:
        """Text up to a top-level character in `stop` (or an unmatched closer)."""
        self.ws()
        s, start, depth = self.s, self.i, []
        while self.i < self.n:
            c = s[self.i]
            if c == '"':
                self.string()
                continue
            if not depth and c in stop:
                break
            if c == "-" and s.startswith("->", self.i):
                self.i += 2
                continue
            if c in _OPEN:
                depth.append(_OPEN[c])
            elif depth and c == depth[-1]:
                depth.pop()
            elif c in ")]}>" and not depth:
                break
            self.i += 1
        return s[start : self.i].strip()


def _attr_dict(r: _Reader) -> dict:
    """Body of `{a = v, b}` or `<{...}>`, cursor after the opening brace."""
    out = {}
    while not r.eat("}"):
        key = r.string()[1:-1] if r.peek('"') else r.match(_KEY)
        out[key] = r.balanced(",}") if r.eat("=") else "unit"
        r.eat(",")
    return out


def _type_list(text: str) -> list:
    r = _Reader(text.strip())
    out = []
    while r.i < r.n:
        out.append(r.balanced(","))
        r.eat(",")
        r.ws()
    return out


def _func_type(r: _Reader):
    r.expect("(")
    ins = r.balanced(")")
    r.expect(")")
    r.expect("->")
    if r.eat("("):
        outs = _type_list(r.balanced(")"))
        r.expect(")")
    else:
        outs = [r.balanced(" \n")]
    return _type_list(ins), outs


def _op(r: _Reader) -> Op:
    results = []
    if r.peek("%"):
        while True:
            name = r.match(_RESULT)
            count = int(r.match(_COUNT)[1:]) if _COUNT.match(r.s, r.i) else 1
            results += [name] if count == 1 else [f"{name}#{k}" for k in range(count)]
            if not r.eat(","):
                break
        r.expect("=")
    r.ws()
    name = r.string()[1:-1]
    r.expect("(")
    operands = []
    while not r.eat(")"):
        operands.append(r.match(_VALUE))
        r.eat(",")
    if r.peek("["):
        raise MlirSyntaxError(f"{name}: unstructured control flow (block successors)")
    props = {}
    if r.eat("<{"):
        props = _attr_dict(r)
        r.expect(">")
    regions = []
    if r.eat("("):
        while True:
            regions.append(_region(r))
            if not r.eat(","):
                break
        r.expect(")")
    attrs = _attr_dict(r) if r.eat("{") else {}
    r.expect(":")
    ins, outs = _func_type(r)
    loc = ""
    if r.eat("loc("):
        loc = "loc(" + r.balanced(")") + ")"
        r.expect(")")
    return Op(name, results, operands, ins, outs, props, attrs, regions, loc)


def _region(r: _Reader) -> list:
    r.expect("{")
    blocks, cur = [], None
    while not r.eat("}"):
        if r.peek("^"):
            r.match(_LABEL)
            args = []
            if r.eat("("):
                while not r.eat(")"):
                    nm = r.match(_VALUE)
                    r.expect(":")
                    args.append((nm, _LOC_SUFFIX.sub("", r.balanced(",)"))))
                    r.eat(",")
            r.expect(":")
            cur = Block(args, [])
            blocks.append(cur)
            continue
        if cur is None:
            cur = Block([], [])
            blocks.append(cur)
        cur.ops.append(_op(r))
    return blocks


def _expand_aliases(aliases: dict[str, str], text: str) -> str:
    """text with every type alias it names replaced by the alias's text (one level)."""

    def expand(m: re.Match[str]) -> str:
        return aliases.get(m.group(0), m.group(0))

    return _ALIAS.sub(expand, text)


@dataclass
class Module:
    top: Op
    locs: dict[str, str]  # "#loc12" -> "loc(...)"
    aliases: dict[str, str]  # "!memref_gmem_f32_" -> type text (aliases expanded)

    def resolve_type(self, t: str) -> str:
        return _expand_aliases(self.aliases, t)

    def resolve_loc(self, loc: str, depth: int = 0) -> str:
        """Location text with #locN references expanded."""
        if depth > 16:
            return loc

        def expand(m: re.Match[str]) -> str:
            return self.resolve_loc(self.locs.get(m.group(0), ""), depth + 1)

        return re.sub(r"#loc\d*", expand, loc)


def parse(text: str) -> Module:
    aliases: dict[str, str] = {}
    locs: dict[str, str] = {}
    body: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^(![A-Za-z_][\w.]*) = (.*)$", line)
        if m:
            aliases[m.group(1)] = m.group(2)
            continue
        m = re.match(r"^(#loc\d*) = (loc\(.*\))$", line)
        if m:
            locs[m.group(1)] = m.group(2)
            continue
        if re.match(r"^#[\w.]+ = ", line):
            continue  # other attribute aliases (loop annotations) carry no meaning here
        body.append(line)
    r = _Reader("\n".join(body))
    top = _op(r)
    r.ws()
    if r.i != r.n:
        raise MlirSyntaxError(f"trailing text at offset {r.i}: {r.s[r.i : r.i + 80]!r}")
    for k in aliases:  # aliases may refer to earlier aliases
        for _ in range(8):
            expanded = _expand_aliases(aliases, aliases[k])
            if expanded == aliases[k]:
                break
            aliases[k] = expanded
    return Module(top, locs, aliases)


def walk(op: Op):
    yield op
    for region in op.regions:
        for block in region:
            for o in block.ops:
                yield from walk(o)
