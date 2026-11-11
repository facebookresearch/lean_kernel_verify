# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Memory model of the PTX frontend (design §5.6): addresses, global accesses, kernel
parameters and mbarriers.

    Memory(cfg, params, tensors, policy)    one per kernel variant; interp.py calls
        execute(tr, instr, mnemonic)        the op table's event rows, ld.param, cvta
        symbol(tr, name, w)                 a symbol operand (shared-memory address)
        wait(tr, instr)                     the wait pseudo-instruction (regions.Wait)
        anchor_value(tr, instr, site)       the value of a work anchor
        barriers(translators)               the barrier arrays, after every role

Addresses are integer expressions over reserved base variables: `%g:<tensor>` is the
address of a kernel tensor (its data pointer plus the record's element offset), and
`%s:<symbol>` the shared-window address of a `.shared` variable. A pointer parameter (or
a pointer field of a struct parameter, named by the adapter) loads as its tensor's base;
`peel` splits an address into (base, byte offset) exactly, through adds, subtracts, a
select between two offsets of one base, a per-thread table of offsets of one base and,
for shared addresses, a truncation to 32 bits. A global access through a select between
two bases takes the arm its condition picks in the role; a condition uniform in each warp
but not in the role splits the role (CuTe; _pick_bases). An address that is no such sum
is refused, never guessed (design §5.5). tma.py adds
`%d:<param>`, a descriptor parameter's address, and `%x:<param>`, Triton's scratch
pointer: only TMA instructions may use them (any other global access is refused).
`%l:<var>` is the local-window address of a `.local` variable (LLVM's __local_depot);
`%S:<symbol>` and `%L:<var>` are the generic addresses of the same bytes, which only
cvta makes (_cvta: cvta.shared(::cluster) %s: -> %S:, cvta.to.shared(::cluster) %S: ->
%s:, cvta.local %l: -> %L:, cvta.to.local %L: -> %l:, the offset kept; of anything else
Data), so a generic address is never taken for a window one (a shared-memory rule wants
%s:). A generic ld/st/atom/red whose address is %S: (or %s:) accesses shared memory: Data.

Global accesses (ld/st.global, generic ld/st whose address peels to a tensor, atom/red,
cp.async) are events of the issuing threads on the tensor (interp.Translator.mem_event):
element offset = byte offset / element bytes, which must be provably exact
(expr.div_exact), else the access is refused (a sub-element access, as the CuTe frontend
refuses it); a vector access covers its consecutive elements with a `range`. A loaded
value is extended to its register by the load type's signedness (ISA ld). An integer ld.global whose
elements are the register's width also yields `load(T, [index])` as its value: a value
the role uses only for data costs nothing, one that reaches a root makes T a metadata
tensor (translate.py finds the loads in the KIR). Its index is replaced by 0 where the
mask is false, so a lane that does not read never makes the load fail; the predicated
instruction's own select keeps the old value there (KIR evaluates only the chosen arm).
A strong load (ops.strong: .volatile, .relaxed, .acquire; also .cv) reads what another
thread may write while the kernel runs, not the recorded contents: its value is Data
(still a gload, and a load anchor's index). An atom or red is a gatomic (atom.exch a
gstore) of its vector's elements, whatever its ordering and scope (ops.py drops them);
red.async.global is one too (it names no mbarrier: there is no completion to model). An
atom's results are Data. atom/red.shared (ops.py drops) write shared memory: no event,
as a generic one on a shared address (_shared_write).

Local memory (ld/st.local through %l:, generic ld/st through %L:) is the thread's own: no
event. A store must stay inside its variable (beyond it lie the thread's other local data,
e.g. ptxas spills, which the model keeps in registers): a KIR assert, per thread where
the offset varies. A load is exact where the same thread stored the same bytes (offset
and width) at a constant offset earlier in the basic block, unpredicated: those values
live in Translator.env[LOCAL], which interp.run forgets at the start of each basic block
(so no merge or loop slot is needed); any other load is Data, which the slicer refuses
where it reaches a root. A store at a runtime offset forgets everything known.

Kernel parameters: `params[i]` = {"kind": "ptr" | "scalar" | "aggregate" | "tensormap",
"name", "fields": {offset: {"size", "kind", "name"}}} per PTX parameter (translate.py
param_table builds it from a LaunchRecord). A scalar or scalar field loads as the KIR
param `name` at the load's width (extended to the register by the type's signedness); a
pointer or pointer field as `%g:<name>`.

mbarriers: the address peels to (shared symbol, byte offset) and must be the same for
every thread of the role (else the role is split by warps, or refused). Each event waits
for its array (Translator.barrier_event, BarrierUse). The arrays are runs of 8-byte slots
initialized with one count, in one guarding region of the role tree, at one .loc chain,
by one role; under the CuTe policy a run is split further by the slots its dynamic uses
may take (frontend/cute/to_kir.py _barrier_runs / _split_arrays; the ranges of stage
slots: sync.use_ranges), except between slots only constant offsets name and the same
roles use with the same events: the stages of one pipeline whose index LLVM folded in
unrolled copies, kept together as the DSL code's dynamic index keeps them for the CuTe
frontend (one array: P0-2 makes a role that waits on one stage a static waiter of every
stage). A use belongs to the array holding the constant of its offset's
top-level sum (sync.top_const); its index is (offset - start) / 8, exact
by structure or known trailing zeros (a loop-carried offset slot keeps its constant part
outside, design §5.3 item 5), else it is refused: an if-chain over candidate arrays would
make unreachable waits static waiters for P0-2. Names: `mbar_<start>` (CuTe), or the
Python assignment target of the init's source line (TLX `k_fulls = tlx.alloc_barriers(..)`
-> k_fulls, `_n` on a repeat), else `mbar_<start>`.
"""

import ast
import math
import re
from pathlib import Path

from . import cluster, expr as E, ops, sync, tma
from .interp import Data, LOCAL, Translator, Unsupported
from .parse import type_bytes

# Mnemonics of global loads and stores (a generic one by its address).
_GLOBAL_LD = ("ld.global", "ld")
_GLOBAL_ST = ("st.global", "st")
# ... and of the explicitly global accesses (a shared address faults).
_EXPLICIT_GLOBAL = ("ld.global", "st.global", "atom.global", "red.global", "red.async")
# cvta between a window and generic addresses: (operand's base, result's base) prefixes.
_CVTA = {
    "cvta.shared": ("%s:", "%S:"),
    "cvta.shared::cluster": ("%s:", "%S:"),
    "cvta.to.shared": ("%S:", "%s:"),
    "cvta.to.shared::cluster": ("%S:", "%s:"),
    "cvta.local": ("%l:", "%L:"),
    "cvta.to.local": ("%L:", "%l:"),
}


class Memory:
    def __init__(self, cfg, params: list, tensors: dict, policy: dict):
        self.cfg, self.fn = cfg, cfg.fn
        self.policy = policy
        self.params = {p["ptx"]: p for p in params}
        self.tensors: dict[str, dict] = {
            n: {**t, "uses": set(), "metadata": False} for n, t in tensors.items()
        }
        self.used_params: dict = {}  # KIR param name -> width
        self.shared = {v.name for v in self.fn.vars if v.space == "shared"} | set(
            policy.get("shared_symbols", ())
        )
        self.local = {  # .local variable (the __local_depot frame) -> its bytes
            v.name: type_bytes(v.type) * (v.array or 1)
            for v in self.fn.vars
            if v.space == "local"
        }
        self.loads: dict = {}  # id(ld.global) -> (tensor, element index), for anchors
        self.tma = tma.Tma(
            self, policy.get("records") or []
        )  # tma.py: TMA, bulk copies

    # ------------------------------------------------------------- addresses
    def symbol(self, tr: Translator, name: str, w: int):
        if name in self.shared:
            return E.var(f"%s:{name}", w)
        if name in self.local:
            return E.var(f"%l:{name}", w)
        if self.params.get(name, {}).get("kind") == "tensormap":  # tma.py: a descriptor
            return self.tma.symbol(name, w)
        return Data(f"the address of {name} (no shared-memory or tensor provenance)")

    def address(self, tr: Translator, op):
        """The address an ("mem", base, offset, coords) operand names."""
        if op[0] != "mem" or op[3] is not None:
            tr.fail(tr.cur, f"address operand {op[0]} (only [base + offset])", "error")
        base = op[1]
        if base[0] == "imm":
            return E.const(base[1] + op[2], 64)
        v = tr.operand_value(base, 64 if base[0] == "sym" else None)
        if isinstance(v, Data):
            return v
        return E.bin("add", v, E.const(op[2], E.width(v))) if op[2] else v

    def split_offset(self, a):
        """(c, rest): a == c + rest, c the constant of a's top-level sum (sync.py: the
        mbarrier array of `C + (index << 3)`, as LLVM writes array accesses; a constant
        inside the scaled index, as the -1 of the srem idiom of `(n - 1) % 2`, is the
        index's, not the array's)."""
        c = sync.top_const(a)
        return c, E.bin("sub", a, E.const(c, E.width(a)))

    def first_instr(self, block: str):
        """The first instruction run from a block (an empty labeled block falls through),
        for its loc."""
        seen = set()
        while block not in seen:
            seen.add(block)
            blk = self.cfg.blocks[block]
            if blk.instrs or len(blk.succs) != 1:
                break
            block = blk.succs[0]
        blk = self.cfg.blocks[block]
        return blk.instrs[0] if blk.instrs else None

    def rebase(self, tr: Translator, e, base):
        """e - base for a value e with base's address variable (an offset slot)."""
        bv, c0 = _base_parts(base)
        p = peel(e)
        if p is None or p[0] is not bv:
            tr.fail(tr.cur, "a loop-carried or merged address changes its base")
        return E.bin("sub", p[1], E.const(c0, E.width(p[1])))

    def slot_base(self, tr: Translator, v):
        """The part of a value kept outside a slot: None for a plain integer; for an
        address its base variable, plus, for a shared one, the constant part of its
        offset (design §5.3 item 5: the barrier array stays known)."""
        if isinstance(v, Data) or not E.reserved(v):
            return None
        p = peel(v)
        if p is None:
            tr.fail(tr.cur, "an address of unknown provenance is carried in a variable")
        if p[0][2].startswith("%s:"):
            c, _terms = _linear(p[1])
            return E.bin("add", p[0], E.const(c, E.width(p[0])))
        return p[0]

    @staticmethod
    def thread_part(off):
        """The sum of the thread-varying terms of off's linear form (_linear), or None:
        kept outside a loop-carried slot, it leaves the slot uniform where a trip adds
        uniform terms only (LSR's per-thread pointers: base + 4 * row, + 512 per
        trip)."""
        _c, terms = _linear(off)
        tv = [(k, t) for k, t in terms if E.thread_varying(t)]
        return _sum(0, tv, E.width(off)) if tv else None

    @staticmethod
    def minus(e, part):
        """e - part with the linear terms they share cancelled (exact in wrapping
        arithmetic, as _linear)."""
        ce, te = _linear(e)
        cp, tp = _linear(part)
        coef: dict = {}
        for k, t in te + [(-k, t) for k, t in tp]:
            coef[t] = coef.get(t, 0) + k
        return _sum(ce - cp, [(k, t) for t, k in coef.items()], E.width(e))

    # ---------------------------------------------------------------- dispatch
    def execute(self, tr: Translator, i, mn: str) -> None:  # noqa: C901 (one case per op family)
        if mn.startswith(tma.SHARED_WRITES):  # (tma.py: tensormap staging buffers)
            self.tma.close_staging(tr)
        if mn == "ld.param":
            return self._ld_param(tr, i)
        if mn.startswith(sync.PREFIXES):  # sync.py: tcgen05, CLC, cluster scope
            return sync.execute(tr, i, mn)
        if mn == "cvta.param":  # tma.py: generic addressing of a descriptor parameter
            return self.tma.cvta_param(tr, i)
        if mn.startswith("cvta"):
            return self._cvta(tr, i, mn)
        if mn.startswith("mapa"):  # cluster.py: a peer CTA's shared memory
            return cluster.mapa(tr, i)
        if mn.startswith(
            tma.PREFIXES
        ):  # tma.py: TMA, bulk copies, in-kernel tensormaps
            return self.tma.execute(tr, i, mn)
        if mn in ("ld.local", "st.local") or (
            mn in ("ld", "st")
            and self._is_local(tr, i.operands[1 if mn == "ld" else 0])
        ):
            return self._local(tr, i, mn)
        if mn in _GLOBAL_LD or mn.startswith("ld.global"):
            return self._load(tr, i, mn)
        if mn in _GLOBAL_ST or mn.startswith("st.global"):
            return self._store(tr, i)
        if mn.startswith(("atom", "red")):
            return self._atomic(tr, i, mn)
        if mn.startswith(("cp.async.ca", "cp.async.cg")):
            return self._cp_async(tr, i)
        if mn.startswith("mbarrier."):
            return self._mbarrier(tr, i, mn)
        if mn.startswith(("bar.", "barrier.")):  # barrier.cluster: sync.py, above
            bar = mn in ("bar.sync", "bar.arrive") or mn.startswith("bar.red.")
            if bar or mn.endswith(".aligned"):  # bar.*: barrier.*.aligned (ISA)
                tr.aligned(mn)
            return  # no P0 event, as both legacy frontends (named_bar is P1, M3)
        if mn == "call":
            return self._call(tr, i)
        tr.fail(i, f"no memory rule for {mn}", "error")

    def _call(self, tr: Translator, i) -> None:
        cls, _why = ops.classify(i)
        if cls == "event":  # __assertfail: reaching it fails the run
            tr.require(E.FALSE, "device assert (call __assertfail)")
            return
        tr.fail(i, f"call: {_why}")

    # ------------------------------------------------------------ parameters
    def _ld_param(self, tr: Translator, i) -> None:
        addr = i.operands[1]
        if addr[0] != "mem" or addr[1][0] != "sym":
            tr.fail(i, "ld.param from a computed address", "error")
        p = self.params.get(addr[1][1])
        if p is None:
            tr.fail(
                i,
                f"ld.param of {addr[1][1]}, which is not a parameter of the kernel",
                "error",
            )
        dst = i.operands[0]
        regs = list(dst[1]) if dst[0] == "vec" else [dst]
        size = _elem_bits(i.opcode) // 8
        for k, r in enumerate(regs):
            if r[0] != "reg":
                continue
            v = self._param_value(tr, i, p, addr[2] + k * size, size)
            tr.write(r[1], _extend(tr, r[1], v, _signed(i.opcode)))

    def _param_value(self, tr: Translator, i, p, off: int, size: int):
        kind = p["kind"]
        if kind in ("scalar", "ptr"):
            entry = {"kind": kind, "name": p["name"], "size": p.get("size", size)}
            if off != 0:
                tr.fail(
                    i,
                    f"ld.param at offset {off} of the scalar parameter {p['ptx']}",
                    "error",
                )
        elif kind == "aggregate":
            entry = p.get("fields", {}).get(off)
            if entry is None:
                tr.fail(
                    i, f"ld.param of {p['ptx']}+{off}: no recorded field there", "error"
                )
        else:
            tr.fail(
                i,
                f"ld.param of the {kind} parameter {p['ptx']} (gap: not translated by the PTX frontend yet)",
            )
        if entry["size"] != size:
            tr.fail(
                i,
                f"ld.param of {size} bytes from a {entry['size']}-byte field",
                "error",
            )
        if entry["kind"] == "ptr":
            if p.get("scratch"):  # tma.py: the kernel's scratch (in-kernel descriptors)
                return E.var(f"%x:{entry['name']}", 64)
            if entry["name"] not in self.tensors:
                return Data(
                    f"pointer parameter {entry['name']} (no tensor of the launch)"
                )
            return E.var(f"%g:{entry['name']}", 64)
        if entry["kind"] != "scalar":
            tr.fail(i, f"ld.param of a {entry['kind']} field", "error")
        w = 8 * size
        self.used_params[entry["name"]] = 32 if w < 32 else w
        return E.var(entry["name"], w)

    # ---------------------------------------------------------- global memory
    def _global(self, tr: Translator, i, op, nbytes: int):
        """(tensor, element offset, elements) of a global access of nbytes at operand op,
        or None if its address is a shared one (generic ld/st/atom/red; through a global
        one, ld/st/atom/red.global or red.async: an error)."""
        a = self.address(tr, op)
        a = tr.as_int(a, "a global address")
        p = peel(a)
        if p is None:
            p = peel(_pick_bases(tr, a))
        if p is None:
            tr.fail(i, "a global access whose address has no tensor provenance")
        base, off = p
        if base[2][:3] in ("%s:", "%S:"):
            mn = ops.mnemonic(i.opcode)
            if mn.startswith(_EXPLICIT_GLOBAL):
                tr.fail(i, f"{mn} on shared memory through a global address", "error")
            return None
        if not base[2].startswith("%g:"):  # tma.py's descriptor parameters and scratch
            tr.fail(i, f"a global access through {base[2][1:]} (only TMA instructions)")
        name = base[2][3:]
        t = self.tensors[name]
        eb = t["elem_bytes"]
        if nbytes % eb:
            tr.fail(
                i,
                f"{nbytes}-byte access of {eb}-byte elements of {name} (sub-element access)",
            )
        q = E.div_exact(off, eb, tr.slot_facts)  # loop-carried pointers: their facts
        if q is None:
            tr.fail(
                i,
                f"byte offset into {name} not provably a multiple of its {eb}-byte elements",
            )
        if t.get("elem_off"):
            q = E.bin("add", E.const(t["elem_off"], 64), q)
        return name, E.resize(q, 64), nbytes // eb

    def _event(self, tr: Translator, kind: str, name: str, off, n: int) -> None:
        self.tensors[name]["uses"].add(kind)
        tr.mem_event(kind, name, off, n)

    def _load(self, tr: Translator, i, mn: str) -> None:
        dst = i.operands[0]
        regs = list(dst[1]) if dst[0] == "vec" else [dst]
        bits = _elem_bits(i.opcode)
        g = self._global(tr, i, i.operands[1], len(regs) * bits // 8)
        if g is None:  # a generic load of shared memory: data
            for r in regs:
                if r[0] == "reg":
                    tr.env[r[1]] = Data(f"{i.opcode} at {tr.loc(i)}")
            return
        name, off, n = g
        self._event(tr, "gload", name, off, n)
        eb = self.tensors[name]["elem_bytes"]
        per = n // len(regs)  # elements per register
        strong = ops.strong(i.opcode)
        for k, r in enumerate(regs):
            if r[0] != "reg":
                continue
            if per != 1 or eb * 8 != bits or bits < 32 or _reg_type(tr, r[1]) == "f":
                tr.env[r[1]] = Data(f"{i.opcode} of {name} at {tr.loc(i)}")
                continue
            idx = E.bin("add", off, E.const(k, 64))
            if k == 0:
                self.loads[id(i)] = (name, off)
            if strong:  # another thread may write it during the run: no metadata read
                tr.env[r[1]] = Data(f"{i.opcode} of {name} at {tr.loc(i)} (a strong "
                                    "load: another thread may write it)")  # fmt: skip
                continue
            v = E.load(name, tr.guard(idx, 0), bits)
            tr.write(r[1], _extend(tr, r[1], v, _signed(i.opcode)))

    def _store(self, tr: Translator, i) -> None:
        val = i.operands[1]
        n = len(val[1]) if val[0] == "vec" else 1
        nbytes = n * _elem_bits(i.opcode) // 8
        g = self._global(tr, i, i.operands[0], nbytes)
        if g is None:  # a generic store to shared memory: data
            self._shared_write(tr, i, i.operands[0], nbytes)
            return
        self._event(tr, "gstore", *g)

    def _shared_write(self, tr: Translator, i, op, nbytes: int) -> None:
        """A generic st/atom/red of shared memory: it closes the staging buffers it may
        overlap (tma.py); refused in a warp-specialized kernel, whose dispatch recognizer
        sees the state bytes' st.shared writes only (cfg.py)."""
        if self.policy.get("dispatch"):
            what = "a generic write of shared memory"
            what = i.opcode if ".shared" in i.opcode else what
            tr.fail(i, f"{what} in a warp-specialized kernel "
                       "(the dispatch's state bytes are checked against st.shared only)")  # fmt: skip
        self.tma.close_staging(tr, self.address(tr, op), nbytes)

    def _atomic(self, tr: Translator, i, mn: str) -> None:
        """atom/red (.global or generic; red.async.global): a gatomic (atom.exch: gstore)
        of the vector's elements per thread; atom/red.shared (interp.py routes the drop
        here), or a generic one whose address is a shared one, is shared-memory data, as
        a generic ld/st. An atom's results are Data."""
        ret = mn.startswith("atom")
        nbytes = _vec(i.opcode) * _elem_bits(i.opcode) // 8
        op = i.operands[1 if ret else 0]
        g = None if ".shared" in mn else self._global(tr, i, op, nbytes)
        if g is None:
            self._shared_write(tr, i, op, nbytes)
        else:
            self._event(tr, "gstore" if mn.endswith(".exch") else "gatomic", *g)
        if ret:
            dst = i.operands[0]
            for r in dst[1] if dst[0] == "vec" else [dst]:
                if r[0] == "reg":
                    tr.env[r[1]] = Data(f"{i.opcode} result at {tr.loc(i)}")

    def _cp_async(self, tr: Translator, i) -> None:
        """cp.async.{ca,cg}.shared.global [dst], [src], size{, src-size | ignore-src}
        {, cache-policy}: a gload of `size` bytes at src; a src-size of 0 (or a true
        ignore-src) reads nothing, any other src-size must be the size (asserted), as
        the CuTe frontend's rule. The cache-policy operand is present iff the opcode has
        .L2::cache_hint."""
        hint = ".L2::cache_hint" in i.opcode
        ops_ = i.operands[: len(i.operands) - 1] if hint else i.operands
        size = ops_[2]
        if size[0] != "imm" or len(ops_) not in (3, 4):
            tr.fail(
                i, "cp.async with a runtime size or an unknown operand list", "error"
            )
        saved = tr.mask
        if len(ops_) == 4:
            x = ops_[3]
            if (
                x[0] in ("reg", "not")
                and _reg_type(tr, x[1] if x[0] == "reg" else x[1][1]) == "pred"
            ):
                p = tr.as_int(tr.operand_value(x), "the cp.async ignore-src predicate")
                tr.mask = E.and1(saved, E.not1(p))
            else:
                n = tr.as_int(tr.operand(i, 3, 32), "the cp.async src-size")
                zero = E.cmp("eq", n, E.const(0, 32))
                full = E.cmp("eq", n, E.const(size[1], 32))
                tr.require(
                    E.bin("or", zero, full), "cp.async src-size is 0 or the copy size"
                )
                tr.mask = E.and1(saved, E.not1(zero))
        g = self._global(tr, i, ops_[1], size[1])
        if g is None:
            tr.fail(i, "cp.async from a shared address", "error")
        self._event(tr, "gload", *g)
        tr.mask = saved

    # ------------------------------------------------------------ local memory
    def _is_local(self, tr: Translator, op) -> bool:
        """The address operand op of a generic ld/st is a generic local address."""
        a = self.address(tr, op)
        p = None if isinstance(a, Data) else peel(a)
        return p is not None and p[0][2].startswith("%L:")

    def _local(self, tr: Translator, i, mn: str) -> None:
        """ld/st.local through a local-window address, or a generic ld/st through a
        generic local one, of the thread's depot (module doc)."""
        store = mn.startswith("st")
        addr, val = i.operands[:2] if store else i.operands[1::-1]
        regs = list(val[1]) if val[0] == "vec" else [val]
        size = _elem_bits(i.opcode) // 8
        p = peel(tr.as_int(self.address(tr, addr), "a local address"))
        window = "%l:" if mn.endswith(".local") else "%L:"
        if p is None or not p[0][2].startswith(window):
            tr.fail(i, "a local access that is no offset into the thread's local depot")
        depot, off = p[0][2][3:], p[1]
        k = off[2] if E.is_const(off) else None
        known = tr.env.get(LOCAL, {})
        if not store:
            for j, r in enumerate(regs):
                if r[0] != "reg":
                    continue
                got = None if k is None else known.get((depot, k + j * size))
                if got is None or got[0] != size or _reg_type(tr, r[1]) == "f":
                    tr.env[r[1]] = Data(f"{i.opcode} of local memory at {tr.loc(i)}")
                else:
                    tr.write(r[1], _extend(tr, r[1], got[1], _signed(i.opcode)))
            return
        n = size * len(regs)
        if n > self.local[depot]:
            tr.fail(i, f"a {n}-byte store into the {self.local[depot]}-byte {depot}")
        # Beyond the depot lie the thread's other local data (ptxas spills): a KIR
        # assert, per thread where the offset varies.
        lim = E.const(self.local[depot] - n, E.width(off))
        tr.require(E.cmp("ule", off, lim), f"st.local inside {depot}")
        if k is None:  # a runtime offset: any byte may change
            tr.env[LOCAL] = {}
            return
        new = {(d, o): x for (d, o), x in known.items()
               if d != depot or o + x[0] <= k or k + n <= o}  # fmt: skip
        for j, r in enumerate(regs):
            v = tr.operand_value(r, 8 * size)
            if i.guard is None and not isinstance(v, Data):  # predicated: unknown
                new[(depot, k + j * size)] = (size, E.resize(v, 8 * size, False))
        tr.env[LOCAL] = new

    # ------------------------------------------------------- address conversion
    def _cvta(self, tr: Translator, i, mn: str) -> None:
        """cvta.global keeps the address (the global window is the generic one). Between
        a window of shared or local memory and generic addresses (_CVTA) the base changes
        and the offset stays: the same bytes. The issuing CTA's shared memory only: its
        shared::cluster addresses are its shared::cta ones (ISA: Shared Memory window);
        another CTA's (cluster.Remote) is Data, as is a value of another window."""
        v = tr.operand(i, 1, 64)
        if mn != "cvta.global":
            frm, to = _CVTA[mn]
            p = None if isinstance(v, Data) else peel(v)
            if p is None or not p[0][2].startswith(frm):
                v = Data(f"{i.opcode} of no {frm[1]} address at {tr.loc(i)}")
            else:
                v = E.bin("add", E.var(to + p[0][2][3:], 64), E.resize(p[1], 64, False))
        if i.operands[0][0] == "reg":
            tr.write(i.operands[0][1], v)

    # --------------------------------------------------------------- mbarriers
    def _smem(self, tr: Translator, i, op):
        a = tr.as_int(self.address(tr, op), "an mbarrier address")
        p = peel(a)
        if p is None or not p[0][2].startswith("%s:"):
            tr.fail(i, "mbarrier address is not a shared-memory symbol plus an offset")
        return p[0][2][3:], p[1]

    def _mbarrier(self, tr: Translator, i, mn: str) -> None:
        ops_ = i.operands
        if mn == "mbarrier.init.shared":
            sym, off = self._smem(tr, i, ops_[0])
            count = tr.as_int(tr.operand(i, 1, 32), "the mbarrier init count")
            tr.barrier_event("bar_init", sym, off, "once", count=count)
            return
        if mn in ("mbarrier.arrive.shared", "mbarrier.arrive.expect_tx.shared"):
            sym, off = self._smem(tr, i, ops_[1])
            if ops_[0][0] == "reg":
                tr.env[ops_[0][1]] = Data(f"mbarrier state at {tr.loc(i)}")
            n = None
            if len(ops_) > 2:
                n = tr.as_int(tr.operand(i, 2, 32), "the mbarrier count")
            if mn.endswith("expect_tx.shared"):
                tr.barrier_event("expect_tx", sym, off, "thread", bytes=n)
                tr.barrier_event("arrive", sym, off, "count")
            else:
                if n is not None and not E.is_const(n, 1):
                    u = tr.uniform(n)
                    if u is None or not E.is_const(u) or u[2] < 1:
                        tr.fail(
                            i,
                            "mbarrier arrive with a count that is not a positive constant",
                        )
                    tr.barrier_event("arrive", sym, off, "count", n=u)
                    return
                tr.barrier_event("arrive", sym, off, "count")
            return
        if mn == "mbarrier.expect_tx.shared":
            sym, off = self._smem(tr, i, ops_[0])
            n = tr.as_int(tr.operand(i, 1, 32), "the transaction bytes")
            tr.barrier_event("expect_tx", sym, off, "thread", bytes=n)
            return
        tr.fail(i, f"no rule for {mn}", "error")

    def wait(self, tr: Translator, instr) -> None:
        sym, off = self._smem(tr, instr, instr.operands[1])
        parity = tr.as_int(tr.operand(instr, 2, 32), "the wait parity")
        tr.barrier_event("wait", sym, off, "once", parity=parity)

    def init_key(self, tr: Translator) -> tuple:
        """What a bar_init's array is split by besides its count: the guarding region
        (innermost If branch of the role tree), the init's whole .loc chain, the role."""
        region = tr.regions[-1] if tr.regions else None
        loc = tr.cur.loc if tr.cur is not None else None
        return (region, tuple(loc.chain()) if loc is not None else (), tr.name)

    # ------------------------------------------------------------------ anchors
    def anchor_value(self, tr: Translator, i, site):
        """The value a work anchor reads at instruction i: an assignment anchor's
        register, or a load anchor's element index (None if i loads another tensor)."""
        if site["kind"] == "load":
            got = self.loads.get(id(i))
            if got is None or got[0] != site.get("tensor", got[0]):
                return None
            return got[1]
        reg = site.get("reg")
        if reg is None:
            dst = i.operands[0] if i.operands else None
            reg = dst[1] if dst is not None and dst[0] == "reg" else None
        if reg is None:
            tr.fail(
                i,
                f"work field '{site['field']}': the anchor writes no register",
                "error",
            )
        return tr.as_int(tr.env.get(reg), f"work field '{site['field']}'")

    # ------------------------------------------------------------- barrier arrays
    def barriers(self, translators: list) -> list:
        """Barrier arrays from the constant init offsets; resolves every event's bar and
        idx and each role's threads_per_issue. KIR barrier list."""
        uses = [u for tr in translators for u in tr.barriers]
        inits = _inits(uses)
        arrays = _runs(inits)
        if self.policy.get("barrier_split"):
            sync.use_ranges(translators)  # sync.py: offsets over stage slots
            arrays = _split(arrays, uses, _profiles(translators))
        names = self._names(arrays, inits)
        for u in uses:
            if u.event["bar"] is None:  # (cluster.py's cluster barrier has its own)
                _resolve(u, arrays, names)
        for tr in translators:
            arrive = [u for u in tr.barriers if u.event["ev"] == "arrive"]
            # Lean multiplies every arrive of the role by threads_per_issue, so the gcd
            # takes an arrive counted inside a thread loop at its per-thread count too
            # (frontend/cute/to_kir.py: `u.issuers or 1`).
            counts = [_arrivals(u) for u in arrive]
            tr.tpi = (
                math.gcd(*counts) if counts and self.policy.get("tpi") == "gcd" else 1
            )
            for u, c in zip(arrive, counts):
                n = c // tr.tpi
                if n != 1:
                    u.event["n"] = n
                else:
                    u.event.pop("n", None)
        return [
            {"name": names[(s, a)], "count": c, "num": {"e": "const", "w": 32, "v": n}}
            for s, a, n, c, _k in arrays
        ] + cluster.barriers(translators)

    def _names(self, arrays, inits) -> dict:
        out, seen = {}, {}
        for s, a, _n, _c, _k in arrays:
            name = f"mbar_{a}"
            if self.policy.get("barrier_names") == "ast":
                name = _assign_target(inits[(s, a)][1]) or name
            k = seen[name] = seen.get(name, 0) + 1
            out[(s, a)] = name if k == 1 else f"{name}_{k}"
        return out


# ------------------------------------------------------------------ addresses


_BASES = ("%g:", "%s:", "%S:", "%l:", "%L:", "%d:", "%x:")  # (_is_base)


def _is_base(x) -> bool:
    """A tensor (%g:), shared symbol (%s:; generic: %S:), local variable (%l:; generic:
    %L:), descriptor parameter (%d:, tma.py) or scratch (%x:, tma.py) address."""
    return x[0] == "var" and x[2][:3] in _BASES


def peel(e):
    """(base variable, offset) with e == base + offset exactly, or None."""
    if _is_base(e):
        return e, E.const(0, E.width(e))
    k = e[0]
    if k == "bin" and e[2] in ("add", "sub"):
        a, b = e[3], e[4]
        pa, pb = (peel(x) if E.reserved(x) else None for x in (a, b))
        if pa is not None and not E.reserved(b):
            return pa[0], E.bin(e[2], pa[1], b)
        if pb is not None and e[2] == "add" and not E.reserved(a):
            return pb[0], E.bin("add", a, pb[1])
        return None
    if k in ("select", "table"):
        return _peel_arms(e)
    if k == "trunc" and E.width(e) == 32:
        p = peel(e[2])
        if p is not None and p[0][2].startswith("%s:"):
            return E.var(p[0][2], 32), E.trunc(p[1], 32)
    if k == "ext" and not e[2] and E.width(e) == 64 and E.width(e[3]) == 32:
        # A zero-extended 32-bit shared, local or parameter-space address (cvt.u64.u32;
        # tma.py's staging buffers and descriptors; the local depot before cvta.local):
        # its window is below 2^32, base + offset does not wrap there.
        p = peel(e[3])
        if p is not None and p[0][2][:3] in ("%s:", "%l:", "%d:"):
            return E.var(p[0][2], 64), E.resize(p[1], 64, False)
    return None


def _peel_arms(e):
    """A select between two addresses, or a per-thread table of addresses (e.g. of a
    lane table), of one base: that base plus the select (table) of their offsets."""
    arms = e[3:5] if e[0] == "select" else e[3]
    ps = [peel(x) for x in arms]
    if E.reserved(e[2]) or any(p is None or p[0] is not ps[0][0] for p in ps):
        return None
    offs = [p[1] for p in ps if p is not None]
    if e[0] == "select":
        return ps[0][0], E.select(e[2], *offs)
    return ps[0][0], E.table(e[2], tuple(offs))


def _pick_bases(tr: Translator, e):
    """e with each select between two addresses whose condition has one value in the
    whole role replaced by the arm it picks. One whose condition is uniform within each
    warp but differs between warps splits the role (CuTe: SplitWarps, as a branch on it
    would; LLVM merged `if tidx < 128: zero dK else: zero dV` into a selp of the base);
    else it stays (peel refuses it)."""
    if e[0] == "select" and (E.reserved(e[3]) or E.reserved(e[4])):
        c = tr.uniform(e[2])
        if c is None:
            tr.split_per_warp(e[2])
        elif E.is_const(c):
            return _pick_bases(tr, e[3] if c[2] else e[4])
    kids = E.children(e)
    new = [_pick_bases(tr, x) if E.reserved(x) else x for x in kids]
    return e if all(x is y for x, y in zip(new, kids)) else E.rebuild(e, new)


def _base_parts(base):
    """(variable, constant) of a slot base (mem.slot_base)."""
    if _is_base(base):
        return base, 0
    return base[3], base[4][2]


def _linear(e):
    """e = c + sum k_i * t_i (exact in wrapping arithmetic) -> (c, [(k_i, t_i)])
    (frontend/cute/to_kir.py)."""
    if E.is_const(e):
        return e[2], []
    if e[0] == "bin" and e[2] in ("add", "sub"):
        ca, ta = _linear(e[3])
        cb, tb = _linear(e[4])
        if e[2] == "sub":
            cb, tb = -cb, [(-k, t) for k, t in tb]
        return ca + cb, ta + tb
    if e[0] == "bin" and e[2] in ("mul", "shl") and E.is_const(e[4]):
        f = e[4][2] if e[2] == "mul" else 1 << e[4][2]
        c, ts = _linear(e[3])
        return c * f, [(k * f, t) for k, t in ts]
    return 0, [(1, e)]


def _sum(c: int, terms: list, w: int):
    """c + sum k_i * t_i at width w (the inverse of _linear); zero terms dropped."""
    out = E.const(c, w)
    for k, t in terms:
        k %= 1 << w
        if k:
            out = E.bin("add", out, t if k == 1 else E.bin("mul", t, E.const(k, w)))
    return out


def _elem_type(opcode: str) -> str:
    for t in reversed(opcode.split(".")):
        if re.fullmatch(r"[bsuf](8|16|32|64|128)|b?f16x2|bf16", t):
            return t
    raise ValueError(f"no element type in {opcode}")


def _elem_bits(opcode: str) -> int:
    return 8 * type_bytes(_elem_type(opcode))


def _vec(opcode: str) -> int:
    """The vector width of a .v2/.v4/.v8 access, else 1."""
    m = re.search(r"\.v(\d+)\b", opcode)
    return int(m[1]) if m else 1


def _signed(opcode: str) -> bool:
    return _elem_type(opcode)[0] == "s"


def _extend(tr: Translator, reg: str, v, signed: bool):
    """A loaded value v in register reg: a wider register gets it extended by the load
    type's signedness (ISA ld: the destination may be wider than the type)."""
    t = tr.k.fn.regs.get(reg, "b32")
    w = 1 if t == "pred" else int(t[1:]) if t[1:].isdigit() else 32
    if isinstance(v, Data) or E.width(v) >= w:
        return v
    return E.resize(v, w, signed)


def _reg_type(tr: Translator, reg: str) -> str:
    t = tr.k.fn.regs.get(reg, "b32")
    return "pred" if t == "pred" else t[0]


# ------------------------------------------------------------- barrier arrays
#
# A BarrierUse keeps the slot's offset expression; at the event the offset must be the
# same in every thread, so it is (after interp's checks) a thread-uniform expression.


def _inits(uses) -> dict:
    """{(symbol, offset): (count, first init)}: every initialized slot, with one count
    (frontend/cute/to_kir.py _barrier_runs: a slot initialized twice keeps its first
    site)."""
    out = {}
    for u in uses:
        if u.init is None:
            continue
        count = u.event["count"]
        if out.setdefault((u.sym, u.init), (count, u))[0] != count:
            raise _err(u, f"{u.sym}+{u.init} initialized with two counts")
    return out


def _runs(inits: dict) -> list:
    """[symbol, start, num, count, key]: consecutive 8-byte slots initialized with one
    count at one site (BarrierUse.key)."""
    arrays = []
    for s, a in sorted(inits):
        count, u = inits[(s, a)]
        last = arrays[-1] if arrays else None
        if (
            last
            and last[0] == s
            and last[1] + 8 * last[2] == a
            and last[3] == count
            and last[4] == u.key
        ):
            last[2] += 1
        else:
            arrays.append([s, a, 1, count, u.key])
    return arrays


def _profiles(translators: list) -> dict:
    """{(symbol, offset): {(role's first warp, warps, event kind)}} of the slots named by a
    constant offset, from their events other than bar_init."""
    out: dict = {}
    for tr in translators:
        for u in tr.barriers:
            if u.rest is None and u.init is None:
                out.setdefault((u.sym, u.const), set()).add(
                    (tr.w0, tr.nw, u.event["ev"])
                )
    return out


def _split(arrays: list, uses: list, profiles: dict) -> list:
    """Split each run into the arrays its uses index (frontend/cute/to_kir.py
    _split_arrays): a use with a dynamic index keeps together the slots its offset may
    take; a run with a use whose range is unknown, or leaves the run, stays whole. Slots
    no dynamic use may take stay together where their `profiles` (_profiles) are equal:
    one pipeline's stages, LLVM unrolled (module doc)."""
    cuts = [set(range(1, n)) for _s, _a, n, _c, _k in arrays]
    dynamic = set()  # (symbol, slot offset) a dynamic use may take
    for u in uses:
        if u.rest is None:
            continue
        i = next((j for j, (s, a, n, _c, _k) in enumerate(arrays)
                  if s == u.sym and a <= u.const < a + 8 * n), None)  # fmt: skip
        if i is None:
            continue  # reported when the address is resolved
        _s, start, num, _c, _k = arrays[i]
        rng = u.rng
        if rng is None or rng[0] < start or rng[1] >= start + 8 * num:
            cuts[i] = set()
        else:
            first, last = (rng[0] - start) // 8, (rng[1] - start) // 8
            cuts[i] -= set(range(first + 1, last + 1))
            dynamic |= {(u.sym, start + 8 * j) for j in range(first, last + 1)}
    out = []
    for (s, start, num, count, key), cs in zip(arrays, cuts):
        for j in sorted(cs):
            a, b = (s, start + 8 * (j - 1)), (s, start + 8 * j)
            if (
                not {a, b} & dynamic
                and profiles.get(a)
                and profiles.get(a) == profiles.get(b)
            ):
                cs.discard(j)
        bounds = [0, *sorted(cs), num]
        out += [
            [s, start + 8 * a, b - a, count, key] for a, b in zip(bounds, bounds[1:])
        ]
    return out


def _resolve(u, arrays: list, names: dict) -> None:
    """Set the event's bar and idx: the array holding the constant part c of its
    offset, idx = (c - start) / 8 + the rest / 8 (BarrierUse.rest)."""
    c = u.const
    arr = next(
        (x for x in arrays if x[0] == u.sym and x[1] <= c < x[1] + 8 * x[2]), None
    )
    if arr is None:
        raise _err(u, f"{u.sym}+{c} is not an initialized mbarrier")
    if (c - arr[1]) % 8:
        raise _err(u, "mbarrier offset not a multiple of 8 bytes from its array")
    k = {"e": "const", "w": 32, "v": (c - arr[1]) // 8}
    idx = k if u.rest is None else (
        u.rest if k["v"] == 0 else {"e": "bin", "w": 32, "o": "add", "a": k, "b": u.rest})  # fmt: skip
    u.event["bar"] = names[(arr[0], arr[1])]
    u.event["idx"] = idx


def _count(u) -> int:
    n = u.event.get("n")
    return n["v"] if isinstance(n, dict) else 1


def _arrivals(u) -> int:
    """Arrivals per issue: issuing threads times the count (one thread for an arrive
    inside a thread loop, issuers None)."""
    return (u.issuers or 1) * _count(u)


def _assign_target(use) -> str | None:
    """The plain variable a TLX init's source line assigns (`k_fulls = ...`), or None."""
    if not use.srcloc:
        return None
    path, line = use.srcloc
    try:
        text = Path(path).read_text().splitlines()[line - 1].strip()
        node = ast.parse(text).body[0]
    except (OSError, IndexError, SyntaxError, ValueError):
        return None
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        t = node.targets[0]
        return t.id if isinstance(t, ast.Name) else None
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _err(u, why: str):
    return Unsupported(u.event["ev"], u.loc, why)
