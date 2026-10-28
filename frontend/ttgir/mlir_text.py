# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Structural parser for the TTGIR text Triton prints (`CompiledKernel.asm["ttgir"]`).

Triton's Python bindings expose no generic op printer, attribute accessor or op loc, so
the frontend parses the printed module. The printer emits one op per line (regions open
with a trailing `{` and close with a line starting with `}`) and resolves every loc to an
alias table, which keeps a line-based parser exact. This module only recovers structure:
ops with their result and operand values, regions/blocks with their arguments, the raw
op text (for op-specific operand/attribute parsing in to_kir) and the source loc. Value
scoping follows the printer: a nested region sees the names of enclosing regions, sibling
regions may reuse names, and `ttg.warp_specialize` partition regions are isolated.
"""

import re
from dataclasses import dataclass, field

NAME = r"%[\w$.\-]+"
REF_RE = re.compile(NAME + r"(?:#\d+)?")
RESULTS_RE = re.compile(rf"^((?:{NAME}(?::\d+)?)(?:, {NAME}(?::\d+)?)*) = ")
OPNAME_RE = re.compile(r'^"?([a-z_][\w]*\.[\w.]+)"?')
LOC_ALIAS_RE = re.compile(r"^#(loc\d*) = (loc\(.*\))$")
ATTR_ALIAS_RE = re.compile(r"^(#[\w.$\-]+) = (#.*)$")


class ParseError(Exception):
    pass


@dataclass(eq=False)
class Value:
    name: str  # printed SSA name without '%', e.g. "curr_m#1"
    type: str  # printed type ("" if the printer does not repeat it)
    owner: "Op | Block"
    index: int

    def __repr__(self):
        return f"%{self.name}"


@dataclass(eq=False)
class Block:
    args: list
    ops: list = field(default_factory=list)
    label: str = ""  # header text that introduced the block ("default", "^bb1", ...)
    # Further blocks of the same region, in textual order (unstructured control flow,
    # e.g. the cf.cond_br an early `return` lowers to). Empty for structured regions.
    more: list = field(default_factory=list)


@dataclass(eq=False)
class Op:
    name: str
    text: (
        str  # op text after "results = ", trailing loc removed; region bodies excluded
    )
    loc: str  # "file:line" or "" (unknown)
    line: int  # 1-based line in the TTGIR text
    loc_name: str = (
        ""  # outermost NameLoc name (the Python variable the value was bound to)
    )
    results: list = field(default_factory=list)
    operands: list = field(default_factory=list)  # Values, in textual order
    regions: list = field(
        default_factory=list
    )  # list[Block] (TTGIR regions have one block)
    region_headers: list = field(default_factory=list)  # text that opened each region

    def where(self) -> str:
        return f"`{self.name}` at {self.loc or 'unknown loc'} (ttgir line {self.line})"


@dataclass
class Module:
    attrs: str
    func: Op  # the tt.func; its regions[0] is the kernel body
    args: list  # kernel arguments (Values owned by the body block)
    aliases: dict  # attribute aliases ("#shared1" -> "#ttg.nvmma_shared<{...}>")

    def resolve(self, attr: str) -> str:
        """An attribute with its alias (if any) expanded."""
        return self.aliases.get(attr.strip(), attr.strip())


def strip_strings(text: str) -> str:
    """Replace string literal contents by nothing (keeps the quotes)."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append('""')
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def split_top(text: str, sep: str = ",") -> list[str]:
    """Split on `sep` at bracket depth 0, outside string literals."""
    parts, depth, cur, in_str = [], 0, [], False
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    closers = set(pairs.values())
    i = 0
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\":
                cur.append(text[i : i + 2])
                i += 2
                continue
            in_str = c != '"'
        elif c == '"':
            in_str = True
        elif c in pairs:
            depth += 1
        elif c in closers and not (c == ">" and i > 0 and text[i - 1] == "-"):
            depth -= 1
        if depth == 0 and not in_str and text.startswith(sep, i):
            parts.append("".join(cur))
            cur = []
            i += len(sep)
            continue
        cur.append(c)
        i += 1
    parts.append("".join(cur))
    return [p.strip() for p in parts]


def find_top(text: str, needle: str) -> int:
    """Index of the last occurrence of `needle` at bracket depth 0, or -1."""
    depth, found = 0, -1
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    closers = set(pairs.values())
    for i, c in enumerate(text):
        if depth == 0 and text.startswith(needle, i):
            found = i
        if c in pairs:
            depth += 1
        elif c in closers and not (c == ">" and i > 0 and text[i - 1] == "-"):
            depth -= 1
    return found


# Locations under the source root are reported as REPO_PREFIX + their path below it.
REPO_PREFIX = "fbcode/"


class _Locs:
    """Resolves loc(...) syntax and #locN aliases to "file:line" of the innermost file loc."""

    def __init__(self, aliases: dict[str, str], src_root: str, repo_prefix: str):
        self.aliases = aliases
        self.src_root = src_root.rstrip("/") + "/" if src_root else ""
        self.repo_prefix = repo_prefix
        self.cache: dict[str, str] = {}

    def resolve(self, loc: str) -> str:
        loc = loc.strip()
        if loc.startswith("#"):
            key = loc[1:]
            if key not in self.cache:
                if key not in self.aliases:
                    raise ParseError(f"undefined loc alias #{key}")
                self.cache[key] = ""  # cycle guard
                self.cache[key] = self.resolve(self.aliases[key])
            return self.cache[key]
        if not (loc.startswith("loc(") and loc.endswith(")")):
            raise ParseError(f"malformed loc {loc!r}")
        return self._inner(loc[4:-1].strip())

    def _inner(self, body: str) -> str:
        if body == "unknown":
            return ""
        if body.startswith("#"):
            return self.resolve(body)
        m = re.match(r'^"((?:[^"\\]|\\.)*)":(\d+):\d+', body)
        if m:
            return f"{self._path(m.group(1))}:{m.group(2)}"
        if body.startswith("callsite("):
            # callsite(callee at caller): report the callee line if it is in the kernel's
            # source tree, else the call site (e.g. `tl.cdiv` -> the line calling it).
            callee, caller = split_top(body[len("callsite(") : -1], " at ")
            inner = self._inner(callee)
            if inner.startswith(self.repo_prefix) or not self.src_root:
                return inner
            return self._inner(caller) or inner
        if body.startswith("fused"):
            inner = body[body.index("[") + 1 : body.rindex("]")]
            for part in split_top(inner):
                r = self._inner(part)
                if r:
                    return r
            return ""
        m = re.match(r'^"(?:[^"\\]|\\.)*"\((.*)\)$', body)  # NameLoc("name"(child))
        if m:
            return self._inner(m.group(1).strip())
        if re.match(r'^"(?:[^"\\]|\\.)*"$', body):
            return ""
        raise ParseError(f"unsupported loc syntax {body!r}")

    def name(self, loc: str) -> str:
        """Outermost NameLoc name of a loc, looking through call sites into the callee
        ("" if it has none)."""
        loc = loc.strip()
        if loc.startswith("#"):
            return self.name(self.aliases.get(loc[1:], "loc(unknown)"))
        body = loc[4:-1].strip() if loc.startswith("loc(") else loc
        if body.startswith("#"):
            return self.name(body)
        if body.startswith("callsite("):
            return self.name(split_top(body[len("callsite(") : -1], " at ")[0])
        m = re.match(r'^"((?:[^"\\]|\\.)*)"\(', body)
        return m.group(1) if m else ""

    def _path(self, path: str) -> str:
        if self.src_root and path.startswith(self.src_root):
            return self.repo_prefix + path[len(self.src_root) :]
        if "/triton/" in path:  # the compiler's own library, wherever it is installed
            return path[path.rindex("/triton/") + 1 :]
        return path


def _trailing_loc(text: str) -> tuple[str, str]:
    """Split `... loc(...)` into (text, loc-syntax); loc-syntax is "" if absent."""
    depth, in_str, last, i = 0, False, -1, 0
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        else:
            if (
                depth == 0
                and text.startswith("loc(", i)
                and (i == 0 or text[i - 1] == " ")
            ):
                last = i
            if c in "([{<":
                depth += 1
            elif c in ")]}" or (c == ">" and text[i - 1] != "-"):
                depth -= 1
        i += 1
    if last < 0:
        return text, ""
    return text[:last].rstrip(), text[last:].strip()


class Parser:
    def __init__(self, text: str, src_root: str = "", repo_prefix: str = REPO_PREFIX):
        self.lines = text.splitlines()
        loc_aliases, self.aliases = {}, {}
        for ln in self.lines:
            m = LOC_ALIAS_RE.match(ln.strip())
            if m:
                loc_aliases[m.group(1)] = m.group(2)
            m = ATTR_ALIAS_RE.match(ln.strip())
            if m:
                self.aliases[m.group(1)] = m.group(2)
        self.locs = _Locs(loc_aliases, src_root, repo_prefix)
        self.i = 0

    # -- line cursor -------------------------------------------------------------------
    def _peek(self) -> str:
        return self.lines[self.i].strip()

    def _next(self) -> str:
        ln = self.lines[self.i].strip()
        self.i += 1
        return ln

    # -- module ------------------------------------------------------------------------
    def parse(self) -> Module:
        while not self._peek().startswith("module"):
            self.i += 1
        header = self._next()
        attrs = (
            header[len("module attributes ") :].rstrip("{").strip()
            if "attributes" in header
            else ""
        )
        funcs = []
        while not self._peek().startswith("}"):
            ln = self._peek()
            if not ln.startswith("tt.func"):
                raise ParseError(
                    f"line {self.i + 1}: unexpected module-level text {ln[:80]!r}"
                )
            funcs.append(self._parse_func())
        if len(funcs) != 1:
            raise ParseError(
                f"expected exactly one tt.func after inlining, found {len(funcs)} "
                "(non-inlined tt.call targets are unsupported)"
            )
        func = funcs[0]
        return Module(
            attrs=attrs, func=func, args=func.regions[0].args, aliases=self.aliases
        )

    def _parse_func(self) -> Op:
        line_no = self.i + 1
        ln = self._next()
        if not ln.endswith("{"):
            raise ParseError(f"line {line_no}: tt.func without body")
        open_ = ln.index("(")
        close = self._match(ln, open_)
        scope: dict[str, Value] = {}
        block = Block(args=[], label="func")
        op = Op(name="tt.func", text=ln[: ln.index("(")], loc="", line=line_no)
        for k, arg in enumerate(a for a in split_top(ln[open_ + 1 : close]) if a):
            v = self._block_arg(arg, block, k)
            block.args.append(v)
            scope[v.name] = v
        op.regions.append(block)
        op.region_headers.append(ln)
        block.ops = self._parse_ops(scope, block)
        self._close_line(op)
        return op

    @staticmethod
    def _match(text: str, open_idx: int) -> int:
        """Index of the bracket closing text[open_idx], skipping string literals."""
        depth, in_str, j = 0, False, open_idx
        while j < len(text):
            c = text[j]
            if in_str:
                if c == "\\":
                    j += 1
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
                if depth == 0:
                    return j
            j += 1
        raise ParseError(f"unbalanced brackets in {text[:120]!r}")

    def _block_arg(self, text: str, block: Block, k: int) -> Value:
        m = re.match(rf"^({NAME}): (.*)$", text.strip())
        if not m:
            raise ParseError(f"malformed block argument {text[:80]!r}")
        body, _ = _trailing_loc(m.group(2))
        type_ = re.sub(r"\s*\{[^}]*\}\s*$", "", body.strip())  # drop arg attributes
        return Value(name=m.group(1)[1:], type=type_.strip(), owner=block, index=k)

    def _close_line(self, op: Op) -> str:
        ln = self._next()
        if not ln.startswith("}"):
            raise ParseError(f"line {self.i}: expected '}}' closing {op.where()}")
        return ln[1:].strip()

    # -- ops ---------------------------------------------------------------------------
    def _parse_ops(self, scope: dict, entry: Block) -> list:
        """Ops of `entry` up to the closing `}`; later `^bbN` blocks go to entry.more."""
        ops, current = [], None
        while not self._peek().startswith("}"):
            if self._peek().startswith("^"):
                label = self._next().split("//")[0].strip().rstrip(":")
                current = Block(args=[], label=label.split("(")[0])
                if "(" in label:
                    body = label[label.index("(") + 1 : label.rindex(")")]
                    current.args = [
                        self._block_arg(a, current, k)
                        for k, a in enumerate(split_top(body))
                    ]
                for v in current.args:
                    scope[v.name] = v
                entry.more.append(current)
                continue
            op = self._parse_op(scope)
            (current.ops if current is not None else ops).append(op)
        return ops

    def _parse_op(self, scope: dict) -> Op:
        line_no = self.i + 1
        ln = self._next()
        m = RESULTS_RE.match(ln)
        result_spec = m.group(1) if m else ""
        rest = ln[m.end() :] if m else ln
        nm = OPNAME_RE.match(rest)
        if not nm:
            raise ParseError(f"line {line_no}: cannot parse op from {ln[:100]!r}")
        op = Op(name=nm.group(1), text="", loc="", line=line_no)
        if strip_strings(rest).rstrip().endswith("{"):
            self._parse_region_op(op, rest, rest, scope)
        elif op.name == "ttg.warp_specialize":
            # The op header stands alone; its regions start on the next line ("default {").
            if self._peek() != "default {":
                raise ParseError(
                    f"line {line_no}: ttg.warp_specialize without a default region"
                )
            self._parse_region_op(op, rest, self._next(), scope)
        else:
            body, loc = _trailing_loc(rest)
            op.text = body
            op.loc = self.locs.resolve(loc) if loc else ""
            op.loc_name = self.locs.name(loc) if loc else ""
            op.operands = self._refs(body, scope, op)
        op.results = self._results(result_spec, op, scope)
        if op.regions:
            types = region_result_types(op)
            if len(types) == len(op.results):
                for v, t in zip(op.results, types):
                    v.type = t
        return op

    def _results(self, spec: str, op: Op, scope: dict) -> list:
        results = []
        for part in split_top(spec) if spec else []:
            name, _, count = part[1:].partition(":")
            n = int(count) if count else 1
            for k in range(n):
                full = name if n == 1 else f"{name}#{k}"
                v = Value(name=full, type="", owner=op, index=len(results))
                results.append(v)
                scope[full] = v
            if n > 1:
                scope[name] = results[-n]  # bare name of a multi-result op = result #0
        return results

    def _refs(self, text: str, scope: dict, op: Op) -> list:
        vals = []
        for ref in REF_RE.findall(strip_strings(text)):
            name = ref[1:]
            if name not in scope:
                raise ParseError(f"{op.where()}: use of undefined value %{name}")
            vals.append(scope[name])
        return vals

    def _parse_region_op(
        self, op: Op, op_header: str, region_header: str, scope: dict
    ) -> None:
        """scf.for/if/while, ttg.warp_specialize and generic `({` region ops.

        A generic-form op (`"tt.reduce"(%x) <{..}> ({ ... }) : type loc(..)`) closes its
        region list with the `)` that starts the tail; the loc follows it.
        """
        hdr = region_header
        while True:
            # warp_specialize partitions (every region after "default") are isolated.
            isolated = op.name == "ttg.warp_specialize" and len(op.regions) > 0
            block, inner = self._region_block(op, op_header, hdr, scope, isolated)
            op.regions.append(block)
            op.region_headers.append(hdr)
            block.ops = self._parse_ops(inner, block)
            tail = self._close_line(op)
            if strip_strings(tail).endswith("{"):  # "} else {", "} do {"
                hdr = tail
                continue
            if not tail and self._peek().startswith("partition"):
                hdr = self._next()
                continue
            break
        generic = strip_strings(op_header).rstrip().endswith("({") and tail.startswith(
            ")"
        )
        trailer, loc = _trailing_loc(tail[1:] if generic else tail)
        trailer = (")" + trailer) if generic else trailer
        op.loc = self.locs.resolve(loc) if loc else ""
        op.loc_name = self.locs.name(loc) if loc else ""
        head = op_header.rstrip()
        head = head[:-1].rstrip() if head.endswith("{") else head
        op.text = head + (" " + trailer if trailer else "")
        # Operands: references in the op header and trailer, minus the names the header
        # binds (scf.for induction variable / iter_args, scf.while before-region args).
        bound = (
            {v.name for v in op.regions[0].args}
            if op.name in ("scf.for", "scf.while")
            else set()
        )
        refs = []
        for ref in REF_RE.findall(strip_strings(head) + " " + strip_strings(trailer)):
            if ref[1:] in bound:
                continue
            if ref[1:] not in scope:
                raise ParseError(f"{op.where()}: use of undefined value {ref}")
            refs.append(scope[ref[1:]])
        op.operands = refs

    def _region_block(
        self, op: Op, op_header: str, hdr: str, scope: dict, isolated: bool
    ):
        inner = {} if isolated else scope
        block = Block(args=[], label=hdr)
        clean = strip_strings(op_header)
        first = not op.regions
        if op.name == "scf.for" and first:
            # `unsigned` (unsignedCmp) is kept in the op text; to_kir refuses it.
            m = re.match(rf"^scf\.for (?:unsigned )?({NAME}) = .* : (\w+) \{{$", clean)
            if not m:
                raise ParseError(f"line {op.line}: unrecognized scf.for header")
            args = [(m.group(1), m.group(2))]
            ia = re.search(r"iter_args\((.*?)\) -> (\(.*\)|[^ ]+)\s+:", clean)
            if ia:
                names = [p.split(" = ")[0] for p in split_top(ia.group(1))]
                args += list(zip(names, _type_list(ia.group(2))))
            block.args = [Value(n[1:], t, block, k) for k, (n, t) in enumerate(args)]
        elif op.name == "scf.while" and first:
            m = re.match(r"^scf\.while \((.*?)\) : (\(.*?\)) ->", clean)
            if not m:
                raise ParseError(f"line {op.line}: unrecognized scf.while header")
            names = (
                [p.split(" = ")[0] for p in split_top(m.group(1))] if m.group(1) else []
            )
            block.args = [
                Value(n[1:], t, block, k)
                for k, (n, t) in enumerate(zip(names, _type_list(m.group(2))))
            ]
        elif op.name == "ttg.warp_specialize" and hdr.startswith("partition"):
            body = hdr[hdr.index("(") + 1 : self._match(hdr, hdr.index("("))]
            block.args = [
                self._block_arg(a, block, k) for k, a in enumerate(split_top(body))
            ]
        elif self._peek().startswith("^"):
            label = self._next()
            if "(" in label:
                body = label[label.index("(") + 1 : label.rindex(")")]
                block.args = [
                    self._block_arg(a, block, k) for k, a in enumerate(split_top(body))
                ]
        for v in block.args:
            inner[v.name] = v
        return block, inner


def _type_list(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    return [t for t in split_top(text) if t]


def region_result_types(op: Op) -> list[str]:
    """Result types of a region op, from the `-> (types)` in its header."""
    clean = strip_strings(op.text)
    if op.name == "scf.for":
        m = re.search(r"iter_args\(.*?\) -> (\(.*\)|[^ ]+)\s+:", clean)
        return _type_list(m.group(1)) if m else []
    if op.name in ("scf.if", "scf.while"):
        idx = find_top(clean, " -> ")
        return _type_list(clean[idx + 4 :]) if idx >= 0 else []
    return []


def parse(text: str, src_root: str = "", repo_prefix: str = REPO_PREFIX) -> Module:
    return Parser(text, src_root, repo_prefix).parse()


def walk(block: Block):
    """Pre-order walk over every op nested in `block` (and its region's later blocks)."""
    for b in [block] + block.more:
        for op in b.ops:
            yield op
            for region in op.regions:
                yield from walk(region)
