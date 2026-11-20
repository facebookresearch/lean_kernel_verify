# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Census of synchronization through global memory in PTX (gsev.py's input, policy `gsync`).

    function_census(module, entry, helpers, src_root) -> {"sites": [...], "counts": {...}}

It finds, in the entry and the functions it calls, the instruction sequences gsev.py turns
into events, and classifies each by the idiom it implements:

* Spin waits (`wait`): a natural loop that strong-loads (.acquire, .relaxed, .volatile)
  one or more global words and leaves on a predicate of the loaded values and loop
  invariants. The loop may also hold warp votes and shuffles, nanosleep, globaltimer
  reads and a timeout path that ends in `trap` (a printf before it included); any
  other effect (a global write or atomic, a barrier, an mbarrier op, TMA) makes it no
  spin (`loop_effects`, listed). The success predicate is classified by shape:
  `tag_eq` (the high half of each polled 64-bit word compared with a target: a tagged
  word), `dist_ge`, `udist_ge`, `ge`, `all_ge`, `topbit`, `nonzero`, `lamport`, else
  `other` (only `tag_eq` spins are translated, gsev.py). An exit decided by globaltimer
  alone is a timeout: `trap` if every path from it ends in trap, else `soft` (the wait
  may return without its predicate: no blocking wait). An exit decided by a loop counter
  alone is a bounded poll. A loop whose only exit is a globaltimer comparison is a delay
  (`timer_delay`), no wait.
* Fetch-adds (`fetch`): an atom whose result is used, classified by the use:
  `last_arriver` (old == N - 1, N a constant or a grid size), `last_mod` (old % k ==
  k - 1), `ticket_base` (old - old % n, old / k * k + k), `grid_barrier` (the result
  reaches a spin's predicate otherwise), `claim` (the result indexes memory or bounds a
  loop: a work cursor), `relay` (stored), else `other` (only claims are translated).

Every site carries its function, PTX line, opcode, the innermost source frame and the
first frame outside the helper directories (`site`), and, where its address comes from a
kernel parameter, the parameter and byte offset (`base`). Loops spinning on shared memory
are `shared_spin` (intra-CTA, out of scope).
"""

import re
from collections import Counter
from types import SimpleNamespace

from . import cfg as C, parse

HELPERS = ("/cuda-", "/include/cuda", "/include/crt/")
_STRONG = {"acquire", "relaxed", "volatile"}
_SPACES = ("shared", "local", "param", "const")
_ARITH = {
    "add", "sub", "mul", "mad", "div", "rem", "min", "max", "neg", "abs", "and", "or",
    "xor", "not", "shl", "shr", "shf", "prmt", "bfe", "bfi", "lop3", "popc", "clz",
    "brev", "setp", "selp", "set", "mov", "cvt", "cvta", "slct", "testp", "fma", "rcp",
    "sqrt", "ex2", "lg2",
}  # fmt: skip
_COLLECTIVE = {"vote", "shfl", "activemask", "redux", "match", "elect"}
_LOOP_OK = _ARITH | _COLLECTIVE | {"bra", "nanosleep", "ld", "st", "call", "trap"}


def function_census(module, entry, helpers=HELPERS, src_root=""):
    """The census of `entry` and the functions it calls (translate.py, policy `gsync`;
    `helpers`: the source directories whose frames a site skips, `src_root`: the export
    root sites are relative to)."""
    names, todo, out = set(), [entry], []
    while todo:
        name = todo.pop()
        if name in names:
            continue
        names.add(name)
        try:
            fn = module.function(name)
        except KeyError:
            continue
        if fn.body is None:
            continue
        a = _Fn(fn, helpers, src_root)
        out += a.sites()
        todo += [
            op[1]
            for i in fn.instructions()
            if i.base == "call"
            for op in i.operands
            if op[0] == "sym"
        ]
    counts = Counter(f"{s['kind']}:{s['idiom']}" for s in out)
    return {"sites": out, "counts": dict(sorted(counts.items()))}


# ------------------------------------------------------------------ instructions


def _mods(i):
    return set(i.opcode.split(".")[1:])


def _global(i):
    """A global or generic memory instruction (no other state space named)."""
    mods = _mods(i)
    return not any(s in mods or f"{s}::cta" in mods or f"{s}::cluster" in mods
                   for s in _SPACES)  # fmt: skip


def strong_load(i):
    return i.base == "ld" and bool(_STRONG & _mods(i)) and _global(i)


def shared_strong_load(i):
    m = _mods(i)
    return (
        i.base == "ld" and bool(_STRONG & m) and ("shared" in m or "shared::cta" in m)
    )


def global_load(i):
    return i.base == "ld" and _global(i)


def _timer(i):
    return i.base == "mov" and any(op == ("sreg", "%globaltimer") for op in i.operands)


def _scope(i):
    m = _mods(i)
    return next((s for s in ("cta", "cluster", "gpu", "sys") if s in m), None)


def _sem(i):
    m = _mods(i)
    sem = next(
        (s for s in ("release", "acq_rel", "acquire", "relaxed") if s in m), None
    )
    return "volatile" if "volatile" in m else sem


def _addr(i):
    return next((op for op in i.operands if op[0] == "mem"), None)


def _rmw_op(i):
    m = i.opcode.split(".")
    return next((t for t in m if t in ("add", "max", "min", "or", "and", "xor", "exch",
                                       "cas", "inc", "dec")), None)  # fmt: skip


# ------------------------------------------------------------------ one function


class _Fn:
    def __init__(self, fn, helpers, root=""):
        self.fn, self.helpers, self.root = fn, helpers, root
        self.instrs = fn.instructions()
        self._flow()
        self.defs = {}
        for i in self.instrs:
            for r in C._dests(i):
                self.defs.setdefault(r, []).append(i)
        self.uses = {}
        for i in self.instrs:
            for r in _reads(i):
                self.uses.setdefault(r, []).append(i)
        self._memo = {}

    # -------------------------------------------------------------- flow

    def _flow(self):
        """Tolerant basic blocks (no shape check): a trap or ret ends a path."""
        blocks, alias, cur, order = {}, {}, None, []
        for x in self.fn.body:
            if isinstance(x, parse.Label):
                if cur is not None and not cur.instrs:
                    alias[x.name] = cur.name
                    continue
                cur = blocks[x.name] = SimpleNamespace(name=x.name, instrs=[])
                order.append(x.name)
            else:
                if cur is None or (cur.instrs and _ends(cur.instrs[-1])):
                    n = f"@{x.line}"
                    cur = blocks[n] = SimpleNamespace(name=n, instrs=[])
                    order.append(n)
                cur.instrs.append(x)
        succ = {}
        for k, n in enumerate(order):
            ins = blocks[n].instrs
            last = ins[-1] if ins else None
            fall = order[k + 1] if k + 1 < len(order) else None
            if last is None or not _ends(last):
                s = (fall,)
            elif last.base == "bra":
                t = last.operands[0][1]
                t = alias.get(t, t)
                s = (t, fall) if last.guard is not None else (t,)
            elif last.base == "brx":
                s = tuple(alias.get(t, t) for t in
                          self.fn.branchtargets.get(last.operands[1][1], ()))  # fmt: skip
            elif last.base in ("ret", "exit") and last.guard is not None:
                s = (fall,)
            else:  # ret, exit, unguarded trap: the path ends
                s = ()
            succ[n] = tuple(x for x in dict.fromkeys(s) if x in blocks)
        seen, stack, post = {order[0]}, [(order[0], iter(succ[order[0]]))], []
        while stack:
            b, it = stack[-1]
            for s in it:
                if s not in seen:
                    seen.add(s)
                    stack.append((s, iter(succ[s])))
                    break
            else:
                stack.pop()
                post.append(b)
        rpo = post[::-1]
        self.blocks = {n: blocks[n] for n in rpo}
        self.succ = {n: succ[n] for n in rpo}
        preds = {n: [] for n in rpo}
        for n in rpo:
            for s in self.succ[n]:
                preds[s].append(n)
        self.preds = preds
        self.block_of = {id(i): n for n in rpo for i in self.blocks[n].instrs}
        nodes = {n: SimpleNamespace(preds=preds[n]) for n in rpo}
        idom = C._idom(nodes, rpo)
        self.idom = idom
        index = {b: k for k, b in enumerate(rpo)}

        def dominates(a, b):
            while b != a:
                if b == rpo[0]:
                    return False
                b = idom[b]
            return True

        loops = {}
        for b in rpo:
            for s in self.succ[b]:
                if index[s] <= index[b] and dominates(s, b):
                    loops.setdefault(s, set()).update(C._natural_loop(nodes, s, b))
        self.loops = {h: frozenset(v) for h, v in loops.items()}
        # blocks every path from which ends in a trap (a timeout's or assert's path)
        doomed = {n for n in rpo if self.blocks[n].instrs
                  and self.blocks[n].instrs[-1].base == "trap" and not self.succ[n]}  # fmt: skip
        changed = True
        while changed:
            changed = False
            for n in rpo:
                if (
                    n not in doomed
                    and self.succ[n]
                    and all(s in doomed for s in self.succ[n])
                ):
                    doomed.add(n)
                    changed = True
        self.doomed = doomed

    def loop_of(self, i):
        """The innermost natural loop holding instruction i: (header, blocks) or None."""
        b = self.block_of.get(id(i))
        best = None
        for h, blk in self.loops.items():
            if b in blk and (best is None or len(blk) < len(best[1])):
                best = (h, blk)
        return best

    # -------------------------------------------------------------- values

    def expr(self, op, depth=12):
        """A small expression tree for an operand: ("const", v), ("sreg", s),
        ("param", sym, off), ("load", instr), ("timer",), ("fetch", instr),
        ("phi", reg, [exprs]), ("op", base, opcode, [args]), ("unknown", reg)."""
        if op[0] == "imm":
            return ("const", op[1])
        if op[0] == "sreg":
            return ("sreg", op[1])
        if op[0] == "sym":
            return ("sym", op[1])
        if op[0] == "not":
            return ("op", "not", "not", [self.expr(op[1], depth)])
        if op[0] != "reg":
            return ("unknown", str(op))
        return self._reg(op[1], depth)

    def _reg(self, r, depth):
        key = (r, depth)
        if key in self._memo:
            return self._memo[key]
        self._memo[key] = ("phi", r, [])  # a cycle through loop-carried registers
        ds = self.defs.get(r, [])
        if depth <= 0 or not ds:
            out = ("unknown", r)
        elif len(ds) == 1:
            out = self._def(ds[0], r, depth - 1)
        else:
            out = ("phi", r, [self._def(d, r, depth - 1) for d in ds[:6]])
        self._memo[key] = out
        return out

    def _def(self, i, r, depth):
        if strong_load(i) or global_load(i):
            return ("load", i)
        if _timer(i):
            return ("timer",)
        if i.base == "atom":
            return ("fetch", i)
        if i.base == "ld" and "param" in _mods(i):
            a = _addr(i)
            base = self.expr(a[1], depth) if a else ("unknown", "?")
            off = a[2] if a else 0
            d = i.operands[0] if i.operands else None
            if (
                d is not None and d[0] == "vec"
            ):  # ld.param.v2/v4: element k at off + k * size
                m = re.search(r"\.[bsuf](8|16|32|64)$", i.opcode)
                regs = [x[1] if x[0] == "reg" else None for x in d[1]]
                if m and r in regs:
                    off += regs.index(r) * int(m.group(1)) // 8
            return ("param", base, off)
        args = [self.expr(op, depth) for op in i.operands[1:]
                if op[0] in ("reg", "imm", "sreg", "not", "sym")]  # fmt: skip
        if i.base == "mov" and i.operands and i.operands[0][0] == "vec":  # unpack
            return ("op", "unpack", i.opcode, args)
        if i.base == "mov" and len(args) == 1:
            return args[0]
        return ("op", i.base, i.opcode, args)

    # -------------------------------------------------------------- sites

    def where(self, i):
        frames = list(i.loc.chain()) if i.loc else []
        inner = f"{_rel(frames[0].file, self.root)}:{frames[0].line}" if frames else ""
        site = next((f"{_rel(f.file, self.root)}:{f.line}" for f in frames
                     if not any(h in f.file for h in self.helpers)), inner)  # fmt: skip
        return inner, site

    def _site(self, kind, idiom, i, **kw):
        inner, site = self.where(i)
        return {"kind": kind, "idiom": idiom, "function": self.fn.name, "line": i.line,
                "opcode": i.opcode, "inner": inner, "site": site, **kw}  # fmt: skip

    def sites(self):
        out, in_spin = [], set()
        groups = {}  # innermost loop header -> its blocks, for loops with strong loads
        for i in self.instrs:
            if strong_load(i) or shared_strong_load(i) or _timer(i):
                lp = self.loop_of(i)
                if lp is not None:
                    groups.setdefault(lp[0], lp[1])
        self.spin_blocks = set()
        for h, blk in groups.items():
            s = self._loop(h, blk)
            if s is not None:
                if s["kind"] == "wait":
                    self.spin_blocks |= set(blk)
                in_spin |= set(s.pop("_loads"))
                out.append(s)
        for i in self.instrs:
            if i.base == "atom" and _global(i) and self._used(i):
                out.append(self._fetch(i))
        return out

    def _used(self, i):
        d = C._dests(i)
        return bool(d) and any(self.uses.get(r) for r in d)

    def base(self, i):
        """The kernel parameter (struct field) an address comes from, if any."""
        a = _addr(i)
        if a is None:
            return None
        return _param_of(self.expr(a[1], 16))

    # -------------------------------------------------------------- spin loops

    def _loop(self, h, blk):
        live = [i for b in blk if b not in self.doomed for i in self.blocks[b].instrs]
        loads = [i for i in live if strong_load(i) and self.loop_of(i)[0] == h]
        shared = [i for i in live if shared_strong_load(i) and self.loop_of(i)[0] == h]
        timers = [i for i in live if _timer(i)]
        if not loads and not shared and not timers:
            return None
        first = (loads or shared or timers)[0]
        effects = sorted({i.opcode for i in live if not _loop_ok(i)})
        exits = [(u, s) for u in sorted(blk) for s in self.succ[u] if s not in blk]
        kinds = [self._exit(u, s, blk) for u, s in exits]
        info = {
            "exits": [k[0] for k in kinds],
            "words": len(loads),
            "header_line": self.blocks[h].instrs[0].line
            if self.blocks[h].instrs
            else 0,
        }
        if loads:
            info.update(scope=sorted({_scope(i) or "-" for i in loads}),
                        sem=sorted({_sem(i) for i in loads}), base=self.base(loads[0]),
                        load_lines=[i.line for i in loads])  # fmt: skip
        if effects:
            # the strong loads of a loop with other effects are coherent reads, no spin
            return (
                None
                if loads
                else (
                    None
                    if not shared
                    else {
                        **self._site(
                            "loop",
                            "shared_flag_loop",
                            first,
                            effects=effects[:8],
                            **info,
                        ),
                        "_loads": [],
                    }
                )
            )
        if not loads:
            idiom = "shared_spin" if shared else "timer_delay"
            return {**self._site("loop", idiom, first, **info), "_loads": []}
        success = [k for k in kinds if k[0] == "success"]
        if not success:
            return {**self._site("wait", "no_value_exit", first, **info),
                    "_loads": [id(i) for i in loads]}  # fmt: skip
        shapes = sorted({k[1] for k in success})
        soft = any(
            k[0] in ("timeout_soft", "poll_soft", "other_exit", "weak_value")
            for k in kinds
        )
        trap = any(k[0] in ("timeout_trap", "poll_trap", "trap") for k in kinds) or any(
            b in self.doomed for b in blk
        )
        idiom = shapes[0] if len(shapes) == 1 else "+".join(shapes)
        acq = any(_sem(i) == "acquire" for i in loads) or self._acq_fence_after(h, blk)
        target = next((k[3] for k in success if k[3]), None)
        return {**self._site("wait", idiom, first, blocking=not soft,
                             timeout="trap" if trap else ("soft" if soft else None),
                             acquire=acq, fetch_base=self._fetch_base(success),
                             target=target, **info), "_loads": [id(i) for i in loads]}  # fmt: skip

    def _acq_fence_after(self, h, blk):
        """A fence.acquire/acq_rel/sc in the loop's exit block's first instructions
        (the fence-based acquire of a relaxed poll)."""
        outs = {s for u in blk for s in self.succ[u] if s not in blk}
        for s in outs:
            for i in self.blocks[s].instrs[:6]:
                if re.match(r"^fence\.(acquire|acq_rel|sc)\b", i.opcode):
                    return True
        return False

    def _fetch_base(self, success):
        for k in success:
            for e in _leaves(k[2]):
                if e[0] == "fetch":
                    return e[1].line
        return None

    def _exit(self, u, s, blk):
        """(class, shape, expr, target) of the loop exit edge u -> s."""
        last = self.blocks[u].instrs[-1] if self.blocks[u].instrs else None
        if last is None or last.base != "bra" or last.guard is None:
            return ("other_exit", None, None, None)
        reg = last.guard[0]
        e = self._reg(reg, 14)
        ls = _leaves(e)
        loads = [x for x in ls if x[0] == "load" and self.block_of.get(id(x[1])) in blk]
        if loads and any(strong_load(x[1]) for x in loads):
            sh = _Shape(self, blk)
            return ("success", sh.of(e), e, sh.target)
        doom = s in self.doomed
        if any(x[0] == "timer" for x in ls):
            return ("timeout_trap" if doom else "timeout_soft", None, e, None)
        if any(
            x[0] == "phi"
            and self.block_of.get(id(self.defs.get(x[1], [None])[0])) in blk
            for x in ls
        ):
            return ("poll_trap" if doom else "poll_soft", None, e, None)
        if loads:
            return ("weak_value", None, e, None)
        return ("trap" if doom else "other_exit", None, e, None)

    # -------------------------------------------------------------- signals, fetches

    def _fetch(self, i):
        d = C._dests(i)[0]
        uses = self._forward(d)
        sem, scope, op = _sem(i), _scope(i), _rmw_op(i)
        idiom, detail = self._fetch_idiom(d, uses)
        return self._site("fetch", idiom, i, op=op, sem=sem, scope=scope, detail=detail,
                          base=self.base(i), uses=sorted({u[0] for u in uses})[:8])  # fmt: skip

    def _fetch_idiom(self, d, uses):
        """(idiom, detail) of a fetch-add with destination d from its uses (module doc)."""
        spin = [
            (u, r)
            for k, u, r in uses
            if k == "compare" and self.block_of.get(id(u)) in self.spin_blocks
        ]
        if spin:
            path = {x[1].base for x in uses if x[0] in ("value", "rem", "div")}
            if path & {"xor"}:
                return (
                    "grid_barrier",
                    "the result reaches a spin's predicate through xor",
                )
            return "ticket_base", "the result sets a spin's target (" + ",".join(
                sorted(path)
            ) + ")"
        for k, u, r in uses:
            if k != "compare":
                continue
            cmp = u.opcode.split(".")[1] if "." in u.opcode else ""
            other = [op for op in u.operands[1:3] if op != ("reg", r)]
            oe = self.expr(other[0], 6) if other else None
            invariant = oe is not None and not any(
                x[0] in ("load", "fetch", "unknown") for x in _leaves(oe)
            )
            direct = r == d or self._copy_of(r, d)
            if direct and invariant and cmp in ("eq", "ne"):
                return "last_arriver", f"old {cmp} {_show(oe)}"
            if (
                not direct
                and invariant
                and cmp in ("eq", "ne")
                and self._via(r, d, ("rem", "and"))
            ):
                return "last_mod", f"(old % k) {cmp} {_show(oe)}"
        kinds = {u[0] for u in uses}
        if "address" in kinds or "compare" in kinds or "branch" in kinds:
            return "claim", "the result indexes memory or bounds a loop"
        if "stored" in kinds:
            return "relay", "the result is stored (e.g. to shared memory for the block)"
        return "other", ""

    def _copy_of(self, r, d, depth=4):
        """r is d through moves and conversions only."""
        for _ in range(depth):
            ds = self.defs.get(r, [])
            if len(ds) != 1 or ds[0].base not in ("mov", "cvt", "shfl"):
                return False
            src = [op for op in ds[0].operands[1:] if op[0] == "reg"]
            if not src:
                return False
            r = src[0][1]
            if r == d:
                return True
        return False

    def _via(self, r, d, bases, depth=6):
        """r is computed from d through an instruction whose base is in `bases`."""
        e = self._reg(r, depth)
        return any(_has_op(e, b) for b in bases) and any(
            x[0] == "fetch" and d in C._dests(x[1]) for x in _leaves(e)
        )

    def _forward(self, reg, depth=6):
        """(class, instr, detail) of the uses of reg, following value instructions."""
        out, todo, seen = [], [(reg, depth)], set()
        while todo:
            r, d = todo.pop()
            if (r, d) in seen or d < 0:
                continue
            seen.add((r, d))
            for u in self.uses.get(r, []):
                a = _addr(u)
                if a is not None and a[1] == ("reg", r):
                    out.append(("address", u, None))
                    continue
                if u.base == "setp":
                    out.append(("compare", u, r))
                    for q in C._dests(u):
                        todo.append((q, d - 1))
                    continue
                if u.base == "st" or u.base == "red" or u.base == "atom":
                    out.append(("stored", u, None))
                    continue
                if u.base in ("rem", "div"):
                    out.append((u.base, u, r))
                if u.base == "bra":
                    out.append(("branch", u, None))
                    continue
                if u.base == "call":
                    out.append(("call", u, None))
                    continue
                for q in C._dests(u):
                    out.append(("value", u, q))
                    todo.append((q, d - 1))
        return out

    # -------------------------------------------------------------- PDL


# ------------------------------------------------------------------ predicates


_THRESHOLD = ("dist_ge", "udist_ge", "ge")


class _Shape:
    """The shape of a spin's success predicate (module doc). Loads inside the loop are
    the polled values; everything else, loads before the loop included, is invariant.
    `target`: the invariant side of the comparison, shown."""

    def __init__(self, fn, blk):
        self.fn, self.blk, self.target = fn, blk, None

    def polled(self, e):
        return any(x[0] == "load" and self.fn.block_of.get(id(x[1])) in self.blk
                   for x in _leaves(e))  # fmt: skip

    def unwrap(self, e):
        """Through conversions, bool-to-int selects, low-bit masks and loop-carried
        flags (the polled producer of a phi)."""
        for _ in range(12):
            if e is None:
                return None
            if e[0] == "phi":
                inner = [x for x in e[2] if self.polled(x)]
                if not inner:
                    return e
                e = inner[0]
            elif e[0] == "op" and e[1] == "cvt" and e[3]:
                e = e[3][0]
            elif (
                e[0] == "op"
                and e[1] == "selp"
                and len(e[3]) == 3
                and self.polled(e[3][2])
                and not self.polled(e[3][0])
                and not self.polled(e[3][1])
            ):
                e = e[3][2]
            elif (
                e[0] == "op"
                and e[1] == "and"
                and len(e[3]) == 2
                and e[3][1] in (("const", 1), ("const", 255), ("const", 0xFFFF))
            ):
                e = e[3][0]
            else:
                return e
        return e

    def of(self, e):
        e = self.unwrap(e)
        if e is None or e[0] != "op":
            return "other"
        base, opcode, args = e[1], e[2], e[3]
        if base == "vote" and "all" in opcode and args:
            inner = self.of(args[0])
            return "all_ge" if inner in _THRESHOLD + ("all_ge",) else f"all_{inner}"
        if base == "not" and args:
            return self.of(args[0])
        if base in ("and", "or", "xor") and opcode.endswith("pred"):
            parts = {self.of(x) for x in args if self.polled(x)}
            parts = {
                "lamport" if x in ("lamport", "lamport_zero") else x for x in parts
            }
            if len(parts) == 1:
                return parts.pop()
            return "+".join(sorted(parts)) if parts else "other"
        if base == "setp":
            return self.cmp(e)
        if self.sign(e):
            return "dist_ge"
        return "other"

    def sign(self, x):
        """x is the sign bit of a polled difference: (T - v) >> 31 or 63."""
        if (
            x[0] == "op"
            and x[1] == "shr"
            and len(x[3]) == 2
            and x[3][1][0] == "const"
            and x[3][1][1] in (31, 63)
        ):
            d = self.unwrap(x[3][0])
            if (
                d is not None
                and d[0] == "op"
                and d[1] in ("sub", "add")
                and self.polled(d)
            ):
                self.target = _show(next((y for y in d[3] if not self.polled(y)), None))
                return True
        return False

    def cmp(self, e):
        parts = e[2].split(".")
        cmp, typ = (parts[1] if len(parts) > 2 else ""), parts[-1]
        if len(e[3]) < 2:
            return "other"
        a, b = e[3][0], e[3][1]
        pa, pb = self.polled(a), self.polled(b)
        if pa == pb:
            return "other"
        x, y = (a, b) if pa else (b, a)
        x = self.unwrap(x)
        zero = y in (("const", 0),)
        if x is None:
            return "other"
        if zero and cmp in ("eq", "ne") and self.sign(x):
            return "dist_ge"
        if (
            zero
            and cmp in ("eq", "ne")
            and x[0] == "op"
            and x[1] in ("setp", "vote", "and", "or")
        ):
            return self.of(x)
        if (
            x[0] == "op"
            and x[1] == "unpack"
            and x[2].endswith(".b64")
            and cmp in ("eq", "ne")
            and typ.endswith("32")
            and x[3]
            and x[3][0][0] == "load"
        ):  # a half of a polled 64-bit word against an invariant: a tagged word (gsev.tag_spin)
            self.target = _show(y)
            return "tag_eq"
        if x[0] == "op" and x[1] == "unpack" or typ in ("s16", "u16", "b16"):
            return "lamport_zero" if zero and cmp in ("eq", "ne") else "other"
        if x[0] == "op" and x[1] == "and" and _has_op(x, "xor"):
            return "topbit"
        if (
            x[0] == "op"
            and x[1] == "mad"
            and len(x[3]) == 3
            and self.polled(x[3][2])
            and not (self.polled(x[3][0]) or self.polled(x[3][1]))
        ):
            x = ("op", "add", "add", [x[3][2], ("op", "mul", "mul", x[3][:2])])
        if x[0] == "op" and x[1] in ("sub", "add"):
            inv = next((z for z in x[3] if not self.polled(z)), None)
            self.target = f"{_show(inv)} ~ {_show(y)}"
            return (
                "udist_ge"
                if typ.startswith(("u", "b")) and not zero
                else ("dist_ge" if typ.startswith("s") else "udist_ge")
            )
        if x[0] == "op" and x[1] in ("min", "max"):
            self.target = _show(y)
            return "all_ge"
        if x[0] == "load":
            self.target = _show(y)
            return "nonzero" if zero and cmp in ("eq", "ne") else "ge"
        if self.sign(x):
            return "dist_ge"
        return "other"


# ------------------------------------------------------------------ helpers


def _ends(i):
    return i.base in ("bra", "brx", "ret", "exit") or (
        i.base == "trap" and i.guard is None
    )


def _reads(i):
    out = set()

    def visit(op):
        if op[0] == "reg":
            out.add(op[1])
        elif op[0] in ("vec", "list"):
            for x in op[1]:
                visit(x)
        elif op[0] == "pair":
            visit(op[1])
            visit(op[2])
        elif op[0] == "not":
            visit(op[1])
        elif op[0] == "mem":
            visit(op[1])
            if op[3] is not None:
                visit(op[3])

    ds = set(C._dests(i))
    for k, op in enumerate(i.operands):
        if k == 0 and ds and op[0] in ("reg", "vec", "pair"):
            continue
        visit(op)
    if i.guard is not None:
        out.add(i.guard[0])
    return out


def _loop_ok(i):
    if i.base not in _LOOP_OK and not _timer(i):
        return False
    if i.base == "ld":
        return (
            strong_load(i)
            or "param" in _mods(i)
            or "shared" in _mods(i)
            or "local" in _mods(i)
        )
    if i.base == "st":
        return "local" in _mods(i)
    if i.base == "call":
        return any(op == ("sym", "vprintf") for op in i.operands)
    return True


_LEAF_KINDS = ("const", "sreg", "sym", "load", "timer", "fetch", "unknown", "param")


def _leaves(e, depth=0):
    """The leaves (and phis) of an expression within 40 levels of it, each once, in the order
    a depth-first walk first meets them. The expressions are DAGs (_Fn._reg shares the
    subexpression of a register): a node is walked again only when met at a smaller depth,
    so the walk is linear in the DAG, not in its unfolded tree."""
    out, best, listed = [], {}, set()

    def walk(x, d):
        if d > 40 or x is None or best.get(id(x), 41) <= d:
            return
        best[id(x)] = d
        if x[0] in _LEAF_KINDS or x[0] == "phi":
            if id(x) not in listed:
                listed.add(id(x))
                out.append(x)
            if x[0] != "phi":
                return
        for y in x[2] if x[0] == "phi" else x[3]:
            walk(y, d + 1)

    walk(e, depth)
    return out


def _has_op(e, base, depth=0, memo=None):
    """`base` is an op of e within 20 levels (memoized per node and depth: e is a DAG)."""
    if depth > 20 or e is None or e[0] not in ("op", "phi"):
        return False
    memo = {} if memo is None else memo
    key = (id(e), depth)
    if key not in memo:
        memo[key] = False  # a cycle through a phi finds nothing new
        kids = e[2] if e[0] == "phi" else e[3]
        memo[key] = (e[0] == "op" and e[1] == base) or any(
            _has_op(x, base, depth + 1, memo) for x in kids
        )
    return memo[key]


def _param_of(e, depth=0, memo=None):
    """(param symbol, byte offset) of an address built on a kernel parameter, or a
    description of where a pointer comes from (the first one a depth-first walk finds;
    memoized per node and depth: e is a DAG)."""
    if e is None or depth > 30:
        return None
    memo = {} if memo is None else memo
    key = (id(e), depth)
    if key in memo:
        return memo[key]
    memo[key] = None
    out = None
    if e[0] == "param":
        b = e[1]
        while b[0] == "op" and b[1] in ("cvta", "mov"):
            b = b[3][0]
        if b[0] == "sym":
            out = f"{b[1]}+{e[2]}"
        elif b[0] == "param":
            out = f"*({_param_of(b, depth + 1, memo)})+{e[2]}"
        else:
            out = f"param@{e[2]}"
    elif e[0] == "load":
        out = "loaded pointer (memory)"
    elif e[0] in ("op", "phi"):
        for x in e[3] if e[0] == "op" else e[2]:
            out = _param_of(x, depth + 1, memo)
            if out is not None:
                break
    elif e[0] == "sym":
        out = e[1]
    memo[key] = out
    return out


def _show(e, depth=0):
    if e is None:
        return None
    if depth > 3:
        return "..."
    if e[0] == "const":
        return str(e[1])
    if e[0] in ("sreg", "sym"):
        return e[1]
    if e[0] == "param":
        return f"param({_param_of(e)})"
    if e[0] == "load":
        return f"load@{e[1].line}"
    if e[0] == "fetch":
        return f"fetch@{e[1].line}"
    if e[0] == "phi":
        return f"phi({e[1]})"
    if e[0] == "op":
        return f"{e[1]}(" + ",".join(_show(x, depth + 1) or "?" for x in e[3]) + ")"
    return e[0]


def _rel(path, root=""):
    """`path` below the export root `root`, else its file name."""
    r = root.rstrip("/") + "/" if root else ""
    if r and path.startswith(r):
        return path[len(r) :]
    return path.rsplit("/", 1)[-1]
