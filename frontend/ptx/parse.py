# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PTX reader of the PTX frontend: module text -> Module (functions, instructions, locs).

    parse(text, path="<ptx>") -> Module          parse_file(path) -> Module

Reads what LLVM's NVPTX back end (Triton) and NVVM 20/23 (CuTe DSL) emit: the module
directives (.version, .target, .address_size), the `.file` table (wherever it appears;
LLVM puts it after the functions), module and function variables, every `.entry`/`.func`
with its parameters and performance directives, `.branchtargets`, `.pragma`, labels,
guarded instructions and `.loc` with `function_name` and `inlined_at`. DWARF sections are
skipped, except `.debug_str`, which names the `function_name` labels. The trailing NUL
byte of CuTe DSL PTX files is stripped. Anything else is a PtxSyntaxError naming the
line, never a silent skip.

Inline asm is ordinary PTX. A `{ }` scope may declare registers, variables and labels;
those are renamed `name#k` (k numbers the scopes of the function), so the local labels
(`waitLoop`, `LAB_WAIT`, `DONE`) and `.reg .pred P1` of every asm block stay distinct.
Registers an unbraced asm block declares (`.reg .b64 desc_<8>;`) live in the function
scope and keep their names, so later asm blocks read and update them. `Instr.asm` is the
index of the enclosing `// begin inline asm` block; `Function.asm_locs[i]` is the location
current where block i begins (asm-internal `.loc`s are synthetic in some emitters).

Locations: an instruction's `loc` is the nearest preceding `.loc` of its function, with its
inlined_at chain (`Loc.inlined_at`, innermost frame first). `inlined_at f l c` names the
call site; the PTX ISA (`.loc`) requires that location to lexically precede, so the call
site's frame is the latest earlier `.loc f l c` of the function, with its own chain (a
bare frame if there is none). `.loc f 0 c` (line 0) is "no location": the instruction
inherits the previous one. File indices are resolved to the `.file` paths.

Operands are tuples:

    ("reg", name)                  declared register (scope-renamed as above)
    ("sreg", name)                 special register, e.g. "%tid.x"
    ("sym", name)                  label, variable, parameter, branch table or function
    ("imm", value)                 integer literal (Python int as written: -1 stays -1)
    ("f32", bits) / ("f64", bits)  0f/0d literal; a decimal float literal becomes f64 bits
    ("mem", base, offset, coords)  [base+offset]; base is a reg/sym/imm operand, coords the
                                   ("vec", ...) of a tensor access [tmap, {c0, ...}] or None
    ("vec", (operand, ...))        {a, b, ...}
    ("pair", a, b)                 a|b, the two destinations of elect.sync / setp
    ("list", (operand, ...))       (a, b), the return and argument lists of call
    ("not", operand)               !p
    ("sink",)                      _

Opcodes are kept as written (`ld.param::entry.b32`); ops.mnemonic normalizes spellings.
"""

import re
import struct
from dataclasses import dataclass, field, replace
from pathlib import Path


class PtxSyntaxError(Exception):
    pass


class _Err(Exception):
    """A syntax error at a line; parse() adds the path."""

    def __init__(self, line, msg):
        super().__init__(msg)
        self.line = line


@dataclass(frozen=True)
class Loc:
    file: str
    line: int
    col: int
    func: str | None = None  # function_name: the function the line belongs to
    inlined_at: "Loc | None" = None  # the call site this frame is inlined at

    def chain(self):
        """The frames, innermost first."""
        loc = self
        while loc is not None:
            yield loc
            loc = loc.inlined_at


@dataclass(frozen=True)
class Param:
    name: str
    type: str  # "b8", "u64", ...
    array: int | None  # N of `name[N]`
    align: int  # `.align N`, else the natural alignment of `type`
    ptr: str | None  # state space of a `.ptr` parameter ("generic" if none is named)
    decl: str

    @property
    def size(self):
        return type_bytes(self.type) * (self.array or 1)


@dataclass(frozen=True)
class Var:
    name: str
    space: str  # "shared", "global", "const", "local" or "param"
    type: str
    array: int | None  # None: a scalar; 0: unsized `name[]`
    align: int
    extern: bool
    init: tuple | None  # initializer values (ints, or the text of anything else)
    line: int


@dataclass(frozen=True)
class Label:
    name: str
    line: int


@dataclass
class Instr:
    opcode: str  # as written, e.g. "ld.global.v4.b32"
    operands: tuple
    guard: tuple | None  # (register, negated) of `@p` / `@!p`
    loc: Loc | None
    line: int
    asm: int | None  # index of the enclosing inline-asm block (Function.asm_locs)
    text: str  # the statement as written, for messages

    @property
    def base(self):
        return self.opcode.split(".", 1)[0]


@dataclass
class Function:
    name: str
    kind: str  # "entry" or "func"
    linkage: tuple  # e.g. ("visible",)
    params: list
    rets: list  # return parameters of a .func
    directives: (
        dict  # performance directives, e.g. {"reqntid": (384,), "explicitcluster": ()}
    )
    line: int
    body: list | None = None  # Instr and Label in text order; None for a prototype
    regs: dict = field(default_factory=dict)  # register -> type ("b32", "pred", ...)
    vars: list = field(default_factory=list)  # variables declared in the body
    branchtargets: dict = field(default_factory=dict)  # table -> label names
    pragmas: list = field(default_factory=list)  # (line, text)
    asm_locs: list = field(default_factory=list)

    def instructions(self):
        return [x for x in self.body or () if isinstance(x, Instr)]


@dataclass
class Module:
    path: str
    version: str = ""
    target: str = ""  # e.g. "sm_100a"
    target_opts: tuple = ()  # the rest of `.target sm_100a, debug`
    address_size: int = 0
    files: dict = field(default_factory=dict)  # .file index -> path
    vars: list = field(default_factory=list)
    functions: list = field(default_factory=list)  # entries, funcs and prototypes
    pragmas: list = field(default_factory=list)  # (line, text)

    def function(self, name):
        for f in self.functions:
            if f.name == name:
                return f
        raise KeyError(name)

    @property
    def entries(self):
        return [f for f in self.functions if f.kind == "entry"]


def type_bytes(t):
    m = re.fullmatch(r"(?:[bsuf]|bf)(\d+)(?:x(\d+))?", t)
    if not m:
        raise ValueError(f"no size for type .{t}")
    return int(m[1]) // 8 * int(m[2] or 1)


def _align(declared, t, line):
    """The `.align` of a declaration, else the natural alignment of its type."""
    if declared:
        return int(declared)
    try:
        return type_bytes(t)
    except ValueError as e:
        raise _Err(line, str(e)) from None


def parse_file(path):
    return parse(Path(path).read_text(), str(path))


def parse(text, path="<ptx>"):
    try:
        return _module(text.rstrip("\0"), path)
    except _Err as e:
        raise PtxSyntaxError(f"{path}:{e.line}: {e}") from None


# ---------------------------------------------------------------------------- lexing
#
# Items, line always at index 1:
#   ("stmt", line, text, end)    a statement; end is ";" or, for a function header, "{"
#   ("open", line) / ("close", line)    a `{` / `}` scope brace
#   ("label", line, name)
#   ("dir", line, text)          a directive ending at the newline (.loc, .file, ...)
#   ("asm", line, "begin"/"end") an inline-asm marker comment

_NL_DIRECTIVE = re.compile(r"\.(?:version|target|address_size|file|loc)\b")
_LABEL = re.compile(r"([A-Za-z_$%][\w$]*)\s*:(?!:)")
_HEADER = re.compile(r"(?:\.(?:visible|extern|weak)\s+)*\.(?:entry|func)\b")
_SECTION = re.compile(r"\.section\s+([^\s{}]+)(.*)")


def _split_comment(raw):
    i = raw.find("//")
    if i < 0:
        return raw, ""
    if '"' in raw[:i]:  # the // may be inside a string literal
        quoted = False
        for j, ch in enumerate(raw):
            if ch == '"':
                quoted = not quoted
            elif not quoted and raw.startswith("//", j):
                return raw[:j], raw[j + 2 :]
        return raw, ""
    return raw[:i], raw[i + 2 :]


def _find_end(s, chars):
    if '"' not in s:
        hits = [i for i in map(s.find, chars) if i >= 0]
        return min(hits) if hits else -1
    quoted = False
    for i, ch in enumerate(s):
        if ch == '"':
            quoted = not quoted
        elif not quoted and ch in chars:
            return i
    return -1


class _Section:
    """A DWARF `.section NAME { ... }`: skipped, except the bytes of `.debug_str` labels."""

    def __init__(self, name, strings):
        self.name, self.strings = name, strings
        self.depth, self.opened, self.label = 0, False, None

    def feed(self, code):
        """Consume one line; False once the section's braces closed."""
        for tok in re.findall(r"[{}]|[^{}]+", code):
            if tok == "{":
                self.depth, self.opened = self.depth + 1, True
            elif tok == "}":
                self.depth -= 1
            elif self.name == ".debug_str":
                t = tok.strip()
                m = re.match(r"([$\w]+):\s*", t)
                if m:
                    self.label = m[1]
                    self.strings[self.label] = bytearray()
                    t = t[m.end() :]
                if t.startswith(".b8") and self.label is not None:
                    self.strings[self.label] += bytes(
                        _int(v) for v in t[3:].replace(",", " ").split()
                    )
        return not (self.opened and self.depth == 0)


def _lex(text):
    items, strings, buf = [], {}, []
    start = header = section = None
    for lineno, raw in enumerate(text.split("\n"), 1):
        code, comment = _split_comment(raw)
        if section is not None:
            if not section.feed(code):
                section = None
            continue
        s = code.strip()
        if "/*" in s:
            raise _Err(lineno, "block comments are not supported")
        while s:
            if not buf:
                if s[0] == "{":
                    items.append(("open", lineno))
                    s = s[1:].lstrip()
                    continue
                if s[0] == "}":
                    items.append(("close", lineno))
                    s = s[1:].lstrip()
                    continue
                if s.startswith(".section"):
                    m = _SECTION.match(s)
                    if not m:
                        raise _Err(lineno, f"bad .section: {s}")
                    section = _Section(m[1], strings)
                    if not section.feed(m[2]):
                        section = None
                    break
                if _NL_DIRECTIVE.match(s):
                    items.append(("dir", lineno, s))
                    break
                m = _LABEL.match(s)
                if m:
                    items.append(("label", lineno, m[1]))
                    s = s[m.end() :].lstrip()
                    continue
                start, header = lineno, bool(_HEADER.match(s))
            # A header runs to its body's `{` (or `;` of a prototype); its performance
            # directives (.reqntid ...) have no `;`. Elsewhere `{` is a vector brace.
            i = _find_end(s, "{;" if header else ";")
            if i < 0:
                buf.append(s)
                break
            buf.append(s[:i])
            items.append(("stmt", start, " ".join(buf).strip(), s[i]))
            buf = []
            s = s[i + 1 :].lstrip() if s[i] == ";" else s[i:]
        if comment.strip() in ("begin inline asm", "end inline asm"):
            items.append(("asm", lineno, comment.split()[0]))
    if buf:
        raise _Err(start, f"unterminated statement: {' '.join(buf)[:80]}")
    return items, strings


# ---------------------------------------------------------------------------- module

_FILE = re.compile(r'\.file\s+(\d+)\s+"([^"]*)"(?:\s*,\s*\d+\s*,\s*\d+)?')
_HEADER_FULL = re.compile(
    r"((?:\.(?:visible|extern|weak)\s+)*)\.(entry|func)\s*(?:\(([^()]*)\)\s*)?"
    r"([%$\w]+)\s*(?:\(([^()]*)\))?(.*)",
    re.S,
)
_PARAM = re.compile(
    r"\.param\s+(?:\.align\s+(\d+)\s+)?\.(\w+)\s+"
    r"(\.ptr\s+(?:\.(global|shared|const|local)\s+)?(?:\.align\s+\d+\s+)?)?"
    r"([%$\w]+)\s*(?:\[(\d+)\])?"
)
_PERF = {
    "reqntid", "maxntid", "maxnreg", "minnctapersm", "maxnctapersm", "explicitcluster",
    "reqnctapercluster", "maxclusterrank", "noreturn",
}  # fmt: skip
_VAR = re.compile(
    r"((?:\.(?:extern|visible|weak|common)\s+)*)\.(shared|global|const|local|param)\s+"
    r"(?:\.align\s+(\d+)\s+)?\.(\w+)\s+([%$\w]+)\s*(?:\[(\d*)\])?\s*(?:=\s*(.*))?",
    re.S,
)


def _module(text, path):
    items, strings = _lex(text)
    m = Module(path)
    for it in items:
        if it[0] == "dir" and it[2].startswith(".file"):
            f = _FILE.fullmatch(it[2])
            if not f:
                raise _Err(it[1], f"bad .file: {it[2]}")
            m.files[int(f[1])] = f[2]
    i = 0
    while i < len(items):
        it = items[i]
        kind, line = it[0], it[1]
        if kind == "dir":
            word, rest = (it[2].split(None, 1) + [""])[:2]
            if word == ".version":
                m.version = rest
            elif word == ".target":
                m.target, *opts = [t.strip() for t in rest.split(",")]
                m.target_opts = tuple(opts)
            elif word == ".address_size":
                m.address_size = int(rest)
            elif word != ".file":
                raise _Err(line, f"{word} outside a function")
        elif kind == "stmt" and _HEADER.match(it[2]):
            fn = _header(it[2], line)
            if it[3] == "{":
                j = _matching_close(items, i + 1)
                _body(fn, items[i + 2 : j], m.files, strings)
                i = j
            m.functions.append(fn)
        elif kind == "stmt" and it[2].startswith(".pragma"):
            m.pragmas.append((line, _pragma(it[2], line)))
        elif kind == "stmt":
            m.vars.append(_var(it[2], line))
        elif kind != "asm":
            raise _Err(line, f"unexpected {kind} outside a function")
        i += 1
    return m


def _matching_close(items, i):
    depth = 0
    for j in range(i, len(items)):
        if items[j][0] == "open":
            depth += 1
        elif items[j][0] == "close":
            depth -= 1
            if depth == 0:
                return j
    raise _Err(items[i][1], "unterminated function body")


def _header(text, line):
    m = _HEADER_FULL.fullmatch(text)
    if not m:
        raise _Err(line, f"bad function header: {text[:80]}")
    directives = {}
    for d in re.finditer(r"\.(\w+)([\s\d,]*)", m[6]):
        if d[1] not in _PERF:
            raise _Err(line, f"unsupported function directive .{d[1]}")
        directives[d[1]] = tuple(int(v) for v in re.split(r"[\s,]+", d[2]) if v)
    if re.sub(r"\.\w+[\s\d,]*", "", m[6]).strip():
        raise _Err(line, f"unsupported function header text: {m[6].strip()[:80]}")
    return Function(
        name=m[4],
        kind=m[2],
        linkage=tuple(re.findall(r"\.(\w+)", m[1])),
        params=_params(m[5], line),
        rets=_params(m[3], line),
        directives=directives,
        line=line,
    )


def _params(text, line):
    out = []
    for decl in filter(None, (p.strip() for p in (text or "").split(","))):
        m = _PARAM.fullmatch(decl)
        if not m:
            raise _Err(line, f"unsupported parameter: {decl}")
        out.append(
            Param(
                name=m[5],
                type=m[2],
                array=int(m[6]) if m[6] else None,
                align=_align(m[1], m[2], line),
                ptr=(m[4] or "generic") if m[3] else None,
                decl=decl,
            )
        )
    return out


def _var(text, line):
    m = _VAR.fullmatch(text)
    if not m:
        raise _Err(line, f"unsupported declaration: {text[:80]}")
    init = None
    if m[7] is not None:
        values = (v.strip() for v in m[7].strip().strip("{}").split(","))
        init = tuple(v if _int(v) is None else _int(v) for v in values)
    return Var(
        name=m[5],
        space=m[2],
        type=m[4],
        array=None if m[6] is None else int(m[6] or 0),
        align=_align(m[3], m[4], line),
        extern="extern" in m[1],
        init=init,
        line=line,
    )


def _pragma(text, line):
    m = re.fullmatch(r'\.pragma\s+"([^"]*)"', text)
    if not m:
        raise _Err(line, f"bad .pragma: {text}")
    return m[1]


# ---------------------------------------------------------------------------- function


class _Scopes:
    """The `{ }` scopes of a function and what each declares. Scope 0 is the function
    body; a name declared in scope k > 0 becomes `name#k`."""

    def __init__(self):
        self.parent, self.decls, self.stack = {0: None}, {0: {}}, [0]

    @property
    def current(self):
        return self.stack[-1]

    def open(self):
        k = len(self.decls)
        self.parent[k], self.decls[k] = self.current, {}
        self.stack.append(k)

    def close(self):
        self.stack.pop()

    def declare(self, name, kind, line):
        """Declare a "reg" or "sym" (label, variable) in the current scope; its new name."""
        d, k = self.decls[self.current], self.current
        if name in d:
            raise _Err(line, f"{name} declared twice in one scope")
        d[name] = (kind, name if k == 0 else f"{name}#{k}")
        return d[name][1]

    def resolver(self, scope, line):
        """name -> ("reg"|"sym"|"sreg", resolved name), seen from `scope`."""

        def ident(name):
            s = scope
            while s is not None:
                if name in self.decls[s]:
                    return self.decls[s][name]
                s = self.parent[s]
            if name.startswith("%"):
                if _SREG.fullmatch(name):
                    return ("sreg", name)
                raise _Err(line, f"undeclared register {name}")
            if "." in name:
                raise _Err(line, f"bad identifier {name}")
            return ("sym", name)

        return ident


def _body(fn, items, files, strings):
    """Fill fn from the items between its braces. Pass 1 builds the scopes and their
    declarations (labels included, so forward branches resolve); pass 2 parses the
    instructions against them."""
    scopes = _Scopes()
    # raw: ("label", name, scope, line) / ("instr", text, scope, line, loc, asm)
    raw, tables = [], []
    loc, seen, asm = None, {}, None
    for it in items:
        kind, line = it[0], it[1]
        if kind == "open":
            scopes.open()
        elif kind == "close":
            scopes.close()
        elif kind == "label":
            scopes.declare(it[2], "sym", line)
            raw.append(("label", it[2], scopes.current, line))
        elif kind == "asm":
            asm = len(fn.asm_locs) if it[2] == "begin" else None
            if it[2] == "begin":
                fn.asm_locs.append(loc)
        elif kind == "dir":
            if it[2].startswith(".loc"):
                loc = _loc(it[2], line, files, strings, seen) or loc
            elif not it[2].startswith(".file"):
                raise _Err(line, f"unexpected directive in a function: {it[2]}")
        elif it[2].startswith(".branchtargets"):
            if not raw or raw[-1][0] != "label":
                raise _Err(line, ".branchtargets without a label")
            targets = it[2][len(".branchtargets") :]
            tables.append((raw.pop()[1], targets, scopes.current, line))
        elif it[2].startswith("."):
            _declaration(fn, scopes, it[2], line)
        else:
            raw.append(("instr", it[2], scopes.current, line, loc, asm))

    fn.body = []
    for r in raw:
        ident = scopes.resolver(r[2], r[3])
        if r[0] == "label":
            fn.body.append(Label(ident(r[1])[1], r[3]))
        else:
            fn.body.append(_instr(r[1], ident, r[3], r[4], r[5]))
    for name, targets, scope, line in tables:
        ident = scopes.resolver(scope, line)
        labels = tuple(ident(t.strip())[1] for t in targets.split(","))
        fn.branchtargets[ident(name)[1]] = labels


def _declaration(fn, scopes, text, line):
    """A `.reg`, `.pragma` or variable declaration in a function body."""
    if text.startswith(".reg"):
        for name, typ in _regs(text, line):
            fn.regs[scopes.declare(name, "reg", line)] = typ
    elif text.startswith(".pragma"):
        fn.pragmas.append((line, _pragma(text, line)))
    else:
        v = _var(text, line)
        fn.vars.append(replace(v, name=scopes.declare(v.name, "sym", line)))


def _regs(text, line):
    m = re.fullmatch(r"\.reg\s+\.(\w+)\s+(.+)", text, re.S)
    if not m:
        raise _Err(line, f"unsupported register declaration: {text}")
    for decl in m[2].split(","):
        d = re.fullmatch(r"\s*([%$\w]+)(?:<(\d+)>)?\s*", decl)
        if not d:
            raise _Err(line, f"unsupported register declaration: {text}")
        if d[2] is None:
            yield d[1], m[1]
        else:  # %r<N> declares %r0 .. %r(N-1)
            yield from ((f"{d[1]}{k}", m[1]) for k in range(int(d[2])))


_LOC = re.compile(
    r"\.loc\s+(\d+)\s+(\d+)\s+(\d+)"
    r"(?:\s*,\s*function_name\s+([$\w]+)(?:\s*\+\s*(\d+))?)?"
    r"(?:\s*,\s*inlined_at\s+(\d+)\s+(\d+)\s+(\d+))?\s*"
)


def _loc(text, line, files, strings, seen):
    """The Loc of a `.loc`, or None for line 0. `seen` maps (file, line, col) to the
    latest Loc there: the frame an `inlined_at` names."""
    m = _LOC.fullmatch(text)
    if not m:
        raise _Err(line, f"bad .loc: {text}")
    key = (int(m[1]), int(m[2]), int(m[3]))
    if key[1] == 0:
        return None

    def path(idx):
        if idx not in files:
            raise _Err(line, f".loc names undefined .file {idx}")
        return files[idx]

    func = None
    if m[4]:  # a .debug_str label (+offset); kept as written if the section is absent
        b = strings.get(m[4])
        func = (
            m[4] if b is None else bytes(b[int(m[5] or 0) :]).split(b"\0")[0].decode()
        )
    site = None
    if m[6]:
        at = (int(m[6]), int(m[7]), int(m[8]))
        site = seen.get(at) or Loc(path(at[0]), at[1], at[2])
    seen[key] = Loc(path(key[0]), key[1], key[2], func, site)
    return seen[key]


# ---------------------------------------------------------------------------- instruction

_INSTR = re.compile(r"(?:@(!?)\s*([%$\w]+)\s+)?([A-Za-z_][\w.:]*)\s*(.*)", re.S)
_IDENT = re.compile(r"[%$A-Za-z_][\w$]*(?:\.[xyz])?")
_INT = re.compile(r"([-+]?)(0[xX][0-9a-fA-F]+|0[bB][01]+|0[0-7]+|[1-9]\d*|0)U?")
_DEC_FLOAT = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?\d+[eE][-+]?\d+"
)
_ADDR = re.compile(r"\s*([%$\w]+)\s*(?:([+-])\s*([-+]?\w+))?\s*")
_SREG = re.compile(
    r"%(?:(?:tid|ntid|ctaid|nctaid|clusterid|nclusterid|cluster_ctaid|cluster_nctaid)"
    r"(?:\.[xyz])?|laneid|warpid|nwarpid|smid|nsmid|gridid|is_explicit_cluster"
    r"|cluster_ctarank|cluster_nctarank|lanemask_(?:eq|le|lt|ge|gt)|clock|clock_hi"
    r"|clock64|globaltimer(?:_lo|_hi)?|pm[0-7](?:_64)?|envreg\d+|total_smem_size"
    r"|aggr_smem_size|dynamic_smem_size|reserved_smem_offset_\w+|current_graph_exec)"
)
SINK = ("sink",)


def _int(s):
    m = _INT.fullmatch(s)
    if not m:
        return None
    t = m[2]
    v = int(t, 8) if t[0] == "0" and t[1:2].isdigit() else int(t, 0)  # 017 is octal
    return -v if m[1] == "-" else v


def _split(s):
    """Split at the commas outside brackets."""
    if not any(c in s for c in "[{("):
        return s.split(",")
    parts, depth, cur = [], 0, 0
    for i, ch in enumerate(s):
        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(s[cur:i])
            cur = i + 1
    parts.append(s[cur:])
    return parts


def _instr(text, ident, line, loc, asm):
    m = _INSTR.fullmatch(text)
    if not m:
        raise _Err(line, f"bad instruction: {text[:80]}")
    guard = None
    if m[2]:
        g = ident(m[2])
        if g[0] != "reg":
            raise _Err(line, f"guard {m[2]} is not a register")
        guard = (g[1], m[1] == "!")
    try:
        ops = tuple(_operand(x, ident) for x in _split(m[4])) if m[4].strip() else ()
    except ValueError as e:
        raise _Err(line, f"{e} in: {text[:80]}") from None
    return Instr(m[3], ops, guard, loc, line, asm, text)


def _operand(s, ident):
    s = s.strip()
    if s == "_":
        return SINK
    head, tail = s[:1], s[-1:]
    if head == "{" and tail == "}":
        return ("vec", tuple(_operand(x, ident) for x in _split(s[1:-1])))
    if head == "(" and tail == ")":
        inner = s[1:-1]
        return (
            "list",
            tuple(_operand(x, ident) for x in _split(inner)) if inner.strip() else (),
        )
    if head == "[" and tail == "]":
        parts = _split(s[1:-1])
        coords = _operand(parts[1], ident) if len(parts) == 2 else None
        a = _ADDR.fullmatch(parts[0])
        if len(parts) > 2 or not a or (coords and coords[0] != "vec"):
            raise ValueError(f"unsupported address {s}")
        off = _int(a[3]) if a[3] else 0
        if off is None:
            raise ValueError(f"unsupported address {s}")
        return ("mem", _operand(a[1], ident), -off if a[2] == "-" else off, coords)
    if "|" in s:
        a, b = s.split("|")
        return ("pair", _operand(a, ident), _operand(b, ident))
    if head == "!":
        return ("not", _operand(s[1:], ident))
    if re.fullmatch(r"0[fF][0-9a-fA-F]{8}", s):
        return ("f32", int(s[2:], 16))
    if re.fullmatch(r"0[dD][0-9a-fA-F]{16}", s):
        return ("f64", int(s[2:], 16))
    v = _int(s)
    if v is not None:
        return ("imm", v)
    if _DEC_FLOAT.fullmatch(s):
        return ("f64", struct.unpack("<Q", struct.pack("<d", float(s)))[0])
    if _IDENT.fullmatch(s):
        return ident(s)
    raise ValueError(f"unsupported operand {s!r}")
