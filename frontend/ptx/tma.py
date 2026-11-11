# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""TMA and bulk copies of the PTX frontend (design §5.6, §5.7): tensor-map descriptors,
the copy instructions and their KIR events.

    PREFIXES                          the event mnemonics mem.Memory.execute routes here
    Tma(mem, records)                 one per kernel variant (mem.Memory.tma); mem.py and
        symbol(name, w)               interp.py route here the address of a descriptor
        cvta_param(tr, i)             parameter, cvta.param, every st.shared (a tensormap
        staging_store(tr, i)          staging buffer's zeroing), and the cp.async.bulk*,
        execute(tr, i, mnemonic)      cp.reduce.async.bulk* and tensormap.* rows
        facts() -> (descs, values)    after every role: Facts.descs and Facts.values
    desc_base(records, k, leaves, taken) -> tensor | None
                                      the base tensor of a CuTe descriptor parameter
                                      (translate.param_table)
    segment_dims(desc, tensor) -> [dim | None]
                                      the tensor dim each descriptor dim walks, for the
                                      descriptor's segments (family assembly)
    tkinfo_version(cubin) -> str | None
                                      the compiler V-string of a cubin (the ISSUE_PTXAS
                                      gate; CuTe assembly)

Descriptors.
  * Host: a `.param .align 64 .b8 p[128]` (param table kind "tensormap"). Its address is
    `%d:<p>` (`mov` of the symbol, zero-extended, `cvta.param`: generic addressing of the
    parameter space); only TMA operands may use it (mem.py refuses any other access
    through it; ld.param of the bytes is refused too). The KIR desc by policy
    `desc_names`: "param" (TTGIR): named by its jit parameter, shape/strides the kernel's
    own params `<name>.shape.i` (i32) and `<name>.stride.i` (i64) that Triton passes
    after the tensormap, box and elem_bytes from the LaunchRecord's tensordesc value; "tma"
    (CuTe): `tma<k>_<base>` (k: its index among the kernel's tensormap params; base: the
    tensor its address resolves to, param_table "base"), shape/strides synthetic params
    `<name>.shape.i`/`<name>.stride.i` (i64) valued per launch by launch.decode ->
    launch.kir_desc over the base's elements (KIR order; packed sub-byte dim 0 in bytes,
    its coordinate converted by dim0_ratio with an assert that it is byte aligned), and
    `<name>.off` (i64) when a launch's base element offset is not 0 (a shifted, ragged
    base). The box must be the same in every launch. Zero-length (appendix E item 6): a
    descriptor at address 0 is a zero-length tensor's; the encoder's extent - 1 =
    0xFFFFFFFF decodes to 2^32, taken as extent 0, as in the legacy frontend (whose extents
    are the tensor's shape, 0): it has no base and no footprint (no element is inside).
    But the hardware reads the extents 2^32, so a TMA through it would access memory at
    address 0: each TMA instruction on it is a KIR `assert false` per issuing thread (it
    fails wherever the instruction is issued). A descriptor null in some launches only
    has no base in those: refused (_make_host).
  * In-kernel (Triton `tl.make_tensor_descriptor`; design §5.6): a 128-byte shared staging
    buffer zeroed by one full warp (32 lanes x `st.shared.b32 0`, the exception to the
    st.shared drop rule), filled by `tensormap.replace` (global_address, rank, elemtype,
    box_dim, global_dim, global_stride per dim; element_stride 1, interleave 0; swizzle,
    fill and atomicity have no KIR meaning) and copied to global memory by
    `tensormap.cp_fenceproxy`, flow-sensitively: each cp_fenceproxy takes the fields
    written since its buffer's last zeroing (TLX reuses one buffer for dV and dK in a
    tile). The zeroing, the replaces and the cp_fenceproxy must be in one KIR statement
    list (masked branches only: a uniform branch or loop between them could skip one),
    with at least one issuing thread each. A TMA instruction whose descriptor address is
    the very address expression a cp_fenceproxy of the role wrote, earlier in the same
    loop iteration and in an enclosing branch, uses that descriptor (each tile writing it
    again is fine); anything else is refused. Only stores and reduces use them: a clipped
    gstore/gatomic on the base (frontend/ttgir/to_kir.py _clipped_store; extents
    zero-extended, byte strides divided exactly); a load through one is a gap.
  * Bulk (non-tensor cp.async.bulk global -> shared): the 1-D descriptor `bulk_<T>_<n>`
    (box n elements of T) of the CuTe frontend; its shape (T's extent) and strides ([1])
    are the family assembly's (in its own tensor parameter names).

Events, by policy `tma` (frontend/cute/ops_mem.py, frontend/ttgir/to_kir.py):
  "desc" (TTGIR): `tma_load` (with bar, idx), `tma_store`, `tma_reduce` on a host
    descriptor, no base footprint. With the policy's `blocks`, a descriptor whose block
    is several hardware boxes has the block as its KIR box, and the messages Triton cuts
    a block into (the next ones of the first's opcode and descriptor in its basic block,
    before any other event) are one event at the first's coordinates, as the TTGIR
    frontend's: each later one must be at another box of the block under the same mask
    and mbarrier (_message).
  "footprint" (CuTe): `tma_load` plus a `gload` of its footprint on the base (the box
    elements inside the extents: frontend/cute/ops_mem.py _box_footprint); a store or
    reduce is a `gstore`/`gatomic` of its footprint on the base.
  Bulk copies (both): g2s a `gload` of the n elements and a `tma_load` of the bulk
  descriptor (coordinate: the element offset) on the mbarrier; s2g a `gstore`; reduce a
  `gatomic`. commit_group/wait_group are drops (ops.py).
Issuing threads (design §5.7, D15; measured on sm_100a hardware for every compiler in
ISSUE_PTXAS: tests/ptx/test_gpu_tma_issue.py). An instruction is issued under the PTX's own
predicates and elect.sync; one that at most one lane of each warp executes (statically) is
issued once per executing thread, as before D15. One that several lanes of a warp may
execute follows ptxas's lowering, the same for every validated version:
  * TMA load and bulk g2s: once per executing lane, never merged (an ELECT loop issues
    each lane's copy with that lane's operands, identical ones too): a tma_load (its
    footprint gload, the bulk gload) per lane, with the lane's coordinates or offset; the
    lanes' complete_tx bytes add up (P0-4), each lane's box as below.
  * TMA reduce and bulk reduce: one gatomic (tma_reduce) per executing lane.
  * TMA store: in each warp, once per distinct run-time value of (descriptor address,
    every coordinate, shared source address) among the executing lanes, issued by the
    lowest lane of each group (ptxas's waterfall loop; equal values in lane-varying
    registers collapse). The L2 cache-policy operand is not part of the key.
  * Bulk s2g: the same waterfall keyed on (global destination, shared source); the size
    is a constant every lane shares (_bulk), so it never splits a group.
Such an instruction is refused (gap) unless the variant's compiler (policy `ptxas`: CuTe
the V-string of its cubin's .note.nv.tkinfo, tkinfo_version; TTGIR the `ptxas --version`
one frontend/ttgir/dump.py records in each LaunchRecord) is in ISSUE_PTXAS: another
ptxas may lower it otherwise; and its target (policy `target`, the PTX .target) in
ISSUE_TARGETS: sm_100 only until the GB300 run of the hardware test validates sm_103.
Scope is one warp (the ptxas loops are per warp); the mbarrier of a load must be the
same for every lane, and a load one thread issues may have lane-dependent operands. Not
measured: multicast and cta_group::2 TMA loads and shared::cluster bulk copies, translated
only where one lane of each warp issues them (cluster.py); prefetch is a drop.
Transaction bytes (P0-4): the KIR box x elem_bytes, including out-of-bounds fill; packed
sub-byte formats count their packed global bytes (launch.kir_desc), as the CuTe frontend,
also 16U4_ALIGN16B / 16U6_ALIGN16B whose shared-memory copy is wider, and also into each
CTA of a multicast and (twice) on a cta_group::2 leader's mbarrier (measured on sm_100a:
tests/ptx/test_gpu_tma_subbyte.py). Their footprints are in bytes too; the hardware traps
on a dim-0 coordinate that is not a multiple of 32 (16U4_ALIGN8B) or 128 (the 16-byte
aligned formats) elements, on any reduce and on a 16U4_ALIGN16B store: a KIR assert per
issuing thread (_PACKED). A packed Triton or in-kernel descriptor is a gap.
Coordinates: PTX lists them innermost first, KIR outermost first (all of shape, strides,
box, coords reversed); i32, signed. mark_addr (translate.py) marks them like the offsets.
Clusters (cluster.py): a load's destination is the issuing CTA's shared memory; a TMA
load's multicast mask is the tma_load's cta_mask and its pair leader's mbarrier (peer bit
cleared, .cta_group::2) its leader; a shared::cluster bulk copy into another CTA is a
complete_tx there; a multicast bulk copy is refused, as by the CuTe frontend. im2col and
gather4 are refused by the op table.
"""

import itertools
import re
import struct
from dataclasses import dataclass, field

from . import cluster, expr as E, launch, ops
from .interp import Data, Translator

PREFIXES = ("cp.async.bulk", "cp.reduce.async.bulk", "tensormap.")  # mem.py routes here
# The compilers whose TMA / bulk issuing (module doc) was measured on hardware: CuTe DSL
# 4.4.2's embedded nvPTXCompiler, DSL 4.6.1's, Triton's ptxas-blackwell (CUDA 13.3), CUDA
# 13.0 and 12.8 ptxas (sm_100a; 12.8 cannot target sm_103a).
ISSUE_PTXAS = frozenset({"12.9.83", "13.3.27", "13.3.33", "13.0.88", "12.8.93"})
# ... and the targets it was measured on (GB200); sm_103 once test_gpu_tma_issue.py passes
# on GB300 (design D15, GPU test (c)).
ISSUE_TARGETS = frozenset({"sm_100"})
# The packed sub-byte CUtensorMap formats, as measured on sm_100a (GB200, driver 13000:
# tests/ptx/test_gpu_tma_subbyte.py) -> (the multiple of the format's elements a dim-0 TMA
# coordinate must be, the TMA kinds through such a descriptor). Any other coordinate or
# kind traps (CUDA_ERROR_ILLEGAL_INSTRUCTION): 32 elements are the 16 global bytes every
# format's dim-0 coordinate must be aligned to; the 16-byte-aligned formats need 128 (the
# only box width their encoder accepts); none reduces, and 16U4_ALIGN16B does not store.
# Trust item: a plain format traps too off that 16-byte grid (measured for UINT8 at
# dim-0 coordinates 1, 3 and 17), which is not asserted (nor by either legacy frontend):
# such a copy is modeled as performed, its footprint and bytes counted.
_PACKED = {
    "16U4_ALIGN8B": (32, ("load", "store")),
    "16U4_ALIGN16B": (128, ("load",)),
    "16U6_ALIGN16B": (128, ("load", "store")),
}
# tensormap.replace elemtype codes (ISA tensormap.replace: .u8 .u16 .u32 .s32 .u64 .s64
# .f16 .f32 .f32.ftz .f64 .bf16 .tf32 .tf32.ftz) -> bytes; 13-15 (packed sub-byte) are
# not in the M3 kernels (refused).
_ELEMTYPE_BYTES = {0: 1, 1: 2, 2: 4, 3: 4, 4: 8, 5: 8, 6: 2, 7: 4, 8: 4, 9: 8, 10: 2,
                   11: 4, 12: 4}  # fmt: skip
_REDUCE_BITS = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64, "u32": 32, "s32": 32,
                "u64": 64, "s64": 64, "b32": 32, "b64": 64}  # fmt: skip
_KERNEL_FIELDS = ("global_address", "rank", "elemtype")  # required besides per-dim ones


@dataclass
class Desc:
    """A descriptor as KIR sees it: the `desc` tensor `name` (shape/strides Exprs in KIR
    order, empty for a bulk one) and, for footprints, the base tensor and the element
    offset of the descriptor's address in it."""

    name: str
    kind: str  # "host" | "kernel" | "bulk"
    base: str | None
    elem_bytes: int
    box: list
    shape: list = field(default_factory=list)
    strides: list = field(default_factory=list)
    off: tuple = E.const(0, 64)
    ratio: tuple = (1, 1)  # dim 0 coordinate -> KIR units: x * num / den
    empty: bool = False  # no element inside its extents in any launch
    packed: str | None = None  # its packed sub-byte CUtensorMap format (_PACKED)
    uses: set = field(default_factory=set)
    msg: list | None = None  # the hardware box when `box` is a block of boxes


@dataclass
class _Ctx:
    """Where a role is in its translation: the KIR statement list (its identity: one per
    uniform branch and loop body, shared by masked branches), the open If branches and the
    open loops (their identities)."""

    stmts: list
    regions: tuple
    loops: tuple

    @staticmethod
    def of(tr: Translator) -> "_Ctx":
        return _Ctx(tr.stmts, tuple(tr.regions), tuple(tr.loops))

    def open_at(self, now: "_Ctx") -> bool:
        """Still current at `now`: the same loop iteration, a branch enclosing `now`."""
        return (
            len(now.loops) == len(self.loops)
            and all(a is b for a, b in zip(self.loops, now.loops))
            and now.regions[: len(self.regions)] == self.regions
        )


@dataclass
class _Stage:
    fields: dict  # (field, dim) -> Expr
    ctx: _Ctx


class Tma:
    def __init__(self, mem, records: list):
        self.mem, self.records = mem, records
        self.params = list(mem.params.values())  # in kernel order
        self.index = {p["ptx"]: k for k, p in enumerate(self.params)}
        self.tmaps = [p["ptx"] for p in self.params if p["kind"] == "tensormap"]
        self.host: dict = {}  # ptx param -> Desc
        self.bulk: dict = {}  # name -> Desc
        self.values = [{} for _ in records]  # per launch: synthetic param -> value
        self.dyn: dict = {}  # translator -> {"stage": {(sym, off): _Stage}, "descs": {}}
        self.bases: dict | None = (
            None  # param index -> base, aliases assigned (_assign)
        )
        pol = mem.policy
        self.footprint = pol.get("tma") == "footprint"
        self.names = pol.get("desc_names", "param")
        self.ptxas = pol.get("ptxas")  # the compiler's V-string: the ISSUE_PTXAS gate
        self.target = pol.get("target")  # "sm_100" / "sm_103": the ISSUE_TARGETS gate
        self.blocks = pol.get("blocks") or {}  # TTGIR descriptor -> block (_messages)
        self.groups: dict = {}  # id(first message) -> its block's group (_message)
        self.later: dict = {}  # id(later message) -> id(first message of its block)
        self.pos: dict | None = None  # id(instr) -> (its basic block's instrs, index)

    # -------------------------------------------------------------- addresses
    def symbol(self, name: str, w: int):
        """The parameter-space address of descriptor parameter `name`."""
        return E.var(f"%d:{name}", w)

    def cvta_param(self, tr: Translator, i) -> None:
        """cvta.param d, a: a descriptor's generic address; any other parameter's is Data
        (a generic access through it is then refused as data)."""
        v = tr.operand_value(i.operands[1], 64)
        d = None if isinstance(v, Data) else _peel(v)
        dst = i.operands[0][1]
        if d is not None and d[0][2].startswith("%d:") and E.is_const(d[1], 0):
            tr.write(dst, E.var(d[0][2], 64))
        else:
            tr.env[dst] = Data(f"generic address of a kernel parameter at {tr.loc(i)}")

    def _smem(self, tr: Translator, i, op, what: str):
        a = tr.as_int(self.mem.address(tr, op), what)
        p = _peel(a)
        if p is None or not p[0][2].startswith("%s:"):
            tr.fail(i, f"{what} is not a shared-memory symbol plus an offset")
        return p[0][2][3:], p[1]

    # --------------------------------------------------------------- dispatch
    def execute(self, tr: Translator, i, mn: str) -> None:
        if mn.startswith("tensormap.replace"):
            return self._replace(tr, i, mn)
        if mn.startswith("tensormap.cp_fenceproxy"):
            return self._cp_fenceproxy(tr, i, mn)
        if mn.startswith("cp.async.bulk.shared::cluster.shared"):  # into another CTA
            return cluster.bulk_s2cluster(tr, i)
        if "multicast" in mn and not mn.startswith("cp.async.bulk.tensor"):
            tr.fail(i, f"{mn} (gap: a multicast bulk copy, as in the CuTe frontend)")
        if ".global.mbarrier" in mn:  # a load: into the issuing CTA (cluster.py)
            cluster.own_cta(tr, i, 0, "the copy destination")
        if mn.startswith("cp.async.bulk.tensor") and ".global.mbarrier" in mn:
            return self._tensor_load(tr, i)
        if mn.startswith(("cp.async.bulk.tensor", "cp.reduce.async.bulk.tensor")):
            return self._tensor_store(tr, i, mn.startswith("cp.reduce"))
        if mn.startswith("cp.async.bulk.shared") and ".global.mbarrier" in mn:
            return self._bulk_load(tr, i)
        if mn == "cp.async.bulk.global.shared.bulk_group":
            return self._bulk_out(tr, i, "gstore")
        if mn.startswith("cp.reduce.async.bulk.global.shared"):
            return self._bulk_out(tr, i, "gatomic")
        tr.fail(i, f"{mn} (gap: no TMA or bulk copy rule)")

    # ------------------------------------------------------------------ TMA
    def _tensor_load(self, tr: Translator, i) -> None:
        """cp.async.bulk.tensor.Nd.shared(::cluster).global.tile.mbarrier::complete_tx::bytes
        [dst], [tmap, {coords}], [mbar] {, cache-policy}."""
        d = self._desc(tr, i, i.operands[1])
        if d.kind == "kernel":
            tr.fail(
                i,
                "TMA load through an in-kernel descriptor (gap: not in the M2 kernels)",
            )
        coords = self._coords(tr, i, d, i.operands[1])
        sym, off, extra = cluster.load_slot(tr, i)  # its cta_mask and leader
        lanes = self._lanes(tr, i, "TMA load") or _one_thread(tr)
        if d.msg is not None and not self._message(tr, i, d, d.msg, coords, (sym, off)):
            return  # a later message of a block: its first stands for it
        tr.barrier_event("tma_load", sym, off, "thread", lanes=lanes, desc=d.name,
                         coords=coords, **extra)  # fmt: skip
        d.uses.add("tma_load")
        if self.footprint:
            self._footprint(tr, i, "gload", d, coords, None)

    def _tensor_store(self, tr: Translator, i, reduce: bool) -> None:
        """cp.async.bulk.tensor.Nd.global.shared::cta.bulk_group [tmap, {coords}], [src]
        {, cache-policy}; cp.reduce.async.bulk.tensor.Nd.global.shared::cta.<op>.bulk_group
        likewise."""
        ops_ = i.operands
        d = self._desc(tr, i, ops_[0])
        coords = self._coords(tr, i, d, ops_[0])
        if d.packed and ("reduce" if reduce else "store") not in _PACKED[d.packed][1]:
            tr.require(E.const(0, 1), f"TMA {'reduce' if reduce else 'store'} through "
                       f"{d.name}, a {d.packed} descriptor: the hardware traps")  # fmt: skip
            return
        if d.msg is not None:
            ev = "tma_reduce" if reduce else "tma_store"
            origin = self._threads_block(tr, d, d.msg, coords)
            if origin is not None:  # the issuing threads' messages are one block: once
                d.uses.add(ev)
                uni, _var = tr.split_mask()
                tr.flush()
                stmt = {"op": "event", "ev": ev, "desc": d.name, "loc": tr.loc(),
                        "coords": [tr.json(c) for c in origin]}  # fmt: skip
                tr.stmts_under(uni, stmt)
                return
            if not self._message(tr, i, d, d.msg, coords, None):
                return  # a later message of a block: its first stands for it
        key = None  # a store several lanes of a warp may execute: the waterfall key
        if self._lanes(tr, i, "TMA reduce" if reduce else "TMA store") and not reduce:
            key = [
                tr.operand_value(ops_[0][1], 64),
                *coords,
                self.mem.address(tr, ops_[1]),
            ]
        if d.kind == "kernel" or self.footprint:
            self._footprint(tr, i, "gatomic" if reduce else "gstore", d, coords, key)
            return
        ev = "tma_reduce" if reduce else "tma_store"
        d.uses.add(ev)
        loc = tr.loc()
        self._issue(tr, i, coords, key, lambda v, j: {
            "op": "event", "ev": ev, "desc": d.name, "coords": [j(c) for c in v], "loc": loc})  # fmt: skip

    def _coords(self, tr: Translator, i, d: Desc, op) -> list:
        """The coordinates in KIR order and units (i32): reversed, dim 0 (KIR last) times
        num / den of the descriptor's dim0_ratio, asserted a whole number of KIR elements
        when den > 1; for a packed sub-byte format asserted the multiple of its elements
        the hardware takes (_PACKED; it traps on any other), which also makes it whole."""
        if op[3] is None or op[3][0] != "vec" or op[2]:
            tr.fail(i, "TMA tensor operand is not [descriptor, {coordinates}]", "error")
        cs = [E.resize(tr.as_int(tr.operand_value(c, 32), "a TMA coordinate"), 32)
              for c in op[3][1]][::-1]  # fmt: skip
        if len(cs) != len(d.box):
            tr.fail(
                i,
                f"{len(cs)} coordinates for the rank-{len(d.box)} descriptor {d.name}",
                "error",
            )
        num, den = d.ratio
        if d.packed:  # a multiple of den too (kir_desc's den is 2, 4 or at most 32)
            g = _PACKED[d.packed][0]
            tr.require(E.cmp("eq", E.bin("srem", cs[-1], E.const(g, 32)), E.const(0, 32)),
                       f"packed sub-byte TMA coordinate is not a multiple of {g} "
                       f"{d.packed} elements (the hardware traps)")  # fmt: skip
        if num > 1:
            cs[-1] = E.bin("mul", cs[-1], E.const(num, 32))
        if den > 1:
            k = E.const(den, 32)
            if not d.packed:
                tr.require(E.cmp("eq", E.bin("srem", cs[-1], k), E.const(0, 32)),
                           "packed sub-byte TMA coordinate is not byte aligned")  # fmt: skip
            cs[-1] = E.bin("sdiv", cs[-1], k)
        return cs

    def _message(
        self, tr: Translator, i, d: Desc, msg: list, coords: list, bar
    ) -> bool:
        """Triton cuts each block of a descriptor into hardware boxes, one message each
        (frontend/ttgir/to_kir.py _tma_block), which the TTGIR frontend's events stand
        for as one: True for the first message of a block, whose event (box: the block)
        stands for them all; False for a later one of the group the first takes from its
        basic block (_later), which must run under the same mask and mbarrier at another
        box of the block, else it is refused."""
        g = self.groups.get(self.later.get(id(i)))
        if g is None:
            left = set(itertools.product(*(range(0, b, m) for b, m in zip(d.box, msg))))
            left.discard((0,) * len(d.box))
            rest = self._later(i, len(left))
            if len(rest) != len(left):
                tr.fail(i, f"{len(rest) + 1} TMA messages of descriptor {d.name} before "
                           f"another event, its block {d.box} has {len(left) + 1}")  # fmt: skip
            self.groups[id(i)] = {"tr": tr, "mask": tr.mask, "bar": bar, "coords": coords,
                                  "left": left}  # fmt: skip
            self.later.update((id(j), id(i)) for j in rest)
            return True
        off = [self.mem.minus(c, c0) for c, c0 in zip(coords, g["coords"])]
        at = tuple(o[2] if E.is_const(o) else None for o in off)
        if not (g["tr"] is tr and g["mask"] is tr.mask and _same(bar, g["bar"])
                and at in g["left"]):  # fmt: skip
            tr.fail(i, f"TMA message of descriptor {d.name} at {at} (box {msg}) is no "
                       f"other box of its block {d.box} under the same mask and mbarrier")  # fmt: skip
        g["left"].discard(at)
        return False

    def _threads_block(self, tr: Translator, d: Desc, msg: list, coords: list):
        """The coordinates of the block, if the statically known issuing threads (one lane
        per warp, several warps) of this instruction issue exactly its boxes once each
        (TLX's dQ reduce: warp parity picks the half of the block), else None."""
        _uni, var = tr.split_mask()
        act = tr.active(var)
        if act is None or len(act) < 2 or not _one_lane_per_warp(tr):
            return None
        tr._check_elected(var, coords)
        inst = [tr.instances(c) for c in coords]
        pts = [[x[t] for x in inst] for t in act]
        offs = []
        for p in pts:
            off = [self.mem.minus(a, b) for a, b in zip(p, pts[0])]
            if not all(E.is_const(o) for o in off):
                return None
            offs.append(tuple(o[2] for o in off))
        lo = tuple(map(min, zip(*offs)))
        rel = [tuple(a - b for a, b in zip(o, lo)) for o in offs]
        boxes = itertools.product(*(range(0, b, m) for b, m in zip(d.box, msg)))
        if sorted(rel) != sorted(boxes):
            return None
        return pts[rel.index((0,) * len(lo))]

    def _later(self, i, n: int) -> list:
        """The (up to n) later messages of the block whose first is i: the next ones of i's
        opcode and descriptor operand in its basic block, before any other event
        instruction."""
        if self.pos is None:
            self.pos = {id(x): (b.instrs, k) for b in self.mem.cfg.blocks.values()
                        for k, x in enumerate(b.instrs)}  # fmt: skip
        instrs, k = self.pos[id(i)]
        t = 1 if "mbarrier" in i.opcode else 0  # its [descriptor, {coords}] operand
        out = []
        for j in instrs[k + 1 :]:
            if len(out) == n:
                break
            if j.opcode == i.opcode and j.operands[t][1] == i.operands[t][1]:
                out.append(j)
            elif ops.classify(j)[0] == "event":
                break
        return out

    def _footprint(
        self, tr: Translator, i, kind: str, d: Desc, coords: list, key
    ) -> None:
        """A gload/gstore/gatomic of the box elements inside the extents, on the base."""
        if d.empty:
            return
        if d.base is None or d.base not in self.mem.tensors:
            tr.fail(i, f"the base of descriptor {d.name} is no tensor of the launch")
        self.mem.tensors[d.base]["uses"].add(kind)
        r = len(coords)
        operands = [E.resize(c, 64) for c in coords] + [
            E.resize(x, 64) for x in d.shape + d.strides] + [d.off]  # fmt: skip
        loc = tr.loc()
        self._issue(tr, i, operands, key, lambda v, j: {
            "op": "event", "ev": kind, "tensor": d.base, "loc": loc,
            **_box_footprint(j, d.box, v[:r], v[r : 2 * r], v[2 * r : 3 * r], v[3 * r])})  # fmt: skip

    def _issue(self, tr: Translator, i, operands: list, key, build) -> None:
        """Once per issuing thread, or with a key once per distinct key (_waterfall)."""
        if key is None:
            tr.thread_event(operands, build)
        else:
            self._waterfall(tr, i, key, lambda: tr.thread_event(operands, build))

    def _lanes(self, tr: Translator, i, what: str) -> bool:
        """True if several lanes of a warp may execute instruction i under the current
        mask; such an instruction is refused unless the variant's compiler is one whose
        issuing was measured (module doc)."""
        if _one_lane_per_warp(tr):
            return False
        if self.ptxas not in ISSUE_PTXAS or self.target not in ISSUE_TARGETS:
            tr.fail(i, f"{what} that several lanes of a warp may execute, compiled by ptxas "
                       f"{self.ptxas or '(version unknown)'} for {self.target}: its issuing is "
                       f"measured for ptxas {', '.join(sorted(ISSUE_PTXAS))} on "
                       f"{', '.join(sorted(ISSUE_TARGETS))} only (tma.ISSUE_PTXAS, "
                       "tma.ISSUE_TARGETS)")  # fmt: skip
        return True

    def _waterfall(self, tr: Translator, i, key: list, emit) -> None:
        """emit() (events of the issuing threads under tr.mask) once per distinct value of
        `key` among the executing lanes of each warp, by the lowest lane holding it
        (ptxas's waterfall loop; frontend/cute/interp.py waterfall_event). An address
        compares by its offset from its base (one base per address expression: peel).
        Under elect.sync the key and the mask must not depend on which lane is elected
        (Translator._check_elected, H1), as for a load or reduce."""
        keys = []
        for k in key:
            p = None if isinstance(k, Data) or not E.reserved(k) else _peel(k)
            if isinstance(k, Data) or (p is None and E.reserved(k)):
                tr.fail(i, "a store of several lanes with an operand outside the model")
            keys.append(tr.instances(k if p is None else p[1]))
        uni, var = tr.split_mask()
        # H1 before the lane table below hides the elect symbol from thread_event's check
        tr._check_elected(var, key)
        conds = tr.instances(var)
        first = []
        for t, c in enumerate(conds):
            for s in range(t - t % 32, t):
                if E.is_const(c, 0):
                    break
                same = conds[s]
                for k in keys:
                    same = E.and1(same, E.cmp("eq", k[s], k[t]))
                c = E.and1(c, E.not1(same))
            first.append(c)
        saved = tr.mask
        tr.mask = E.and1(uni, tr.lane_table(first, "first"))
        try:
            emit()
        finally:
            tr.mask = saved

    # --------------------------------------------------------- descriptors
    def _desc(self, tr: Translator, i, op) -> Desc:
        """The descriptor a TMA operand [tmap, ...] names (a zero-length tensor's null
        one asserted not issued: module doc)."""
        v = tr.operand_value(op[1], 64)
        if isinstance(v, Data):
            tr.as_int(v, "a TMA descriptor address")
        p = _peel(v)
        if p is not None and p[0][2].startswith("%d:") and E.is_const(p[1], 0):
            d = self._host(tr, i, p[0][2][3:])
            if d.empty:
                tr.require(E.const(0, 1), f"TMA through {d.name}, a zero-length tensor's "
                           "descriptor (address 0): it accesses address 0")  # fmt: skip
            return d
        got = self._dyn(tr)["descs"].get(v)
        if got is not None and got[1].open_at(_Ctx.of(tr)):
            return got[0]
        return tr.fail(i, "TMA descriptor address of unknown provenance: neither a "
                          "descriptor parameter nor one this role wrote by "
                          "tensormap.cp_fenceproxy earlier in the iteration")  # fmt: skip

    def _host(self, tr: Translator, i, ptx: str) -> Desc:
        d = self.host.get(ptx)
        if d is None:
            try:
                d = self._make_host(ptx)
            except (launch.LaunchError, KeyError, TypeError, StopIteration) as e:
                tr.fail(i, f"descriptor parameter {ptx}: {type(e).__name__}: {e}")
            self.host[ptx] = d
        return d

    def _make_host(self, ptx: str) -> Desc:
        k = self.index[ptx]
        p = self.params[k]
        recs = [r["params"][k] for r in self.records]
        if not recs:
            raise launch.LaunchError("no LaunchRecord")
        # TTGIR: the tensordesc value and the kernel's own shape/stride params
        if self.names == "param":
            vals = [r["value"] for r in recs]
            # e.g. fp4_padded (packed sub-byte host dtypes 13-15)
            if any(v.get("host_dtype") in (13, 14, 15) for v in vals):
                raise launch.LaunchError(
                    "a packed sub-byte Triton descriptor (gap, design §5.1)"
                )
            if len({(tuple(v["box"]), v["elem_size"]) for v in vals}) != 1:
                raise launch.LaunchError("launches differ in the descriptor's box")
            rank = len(vals[0]["box"])
            names = [
                f"{p['name']}.{f}.{j}" for f in ("shape", "stride") for j in range(rank)
            ]
            dims = [self._own_param(n) for n in names]
            box, msg = list(vals[0]["box"]), None
            block = self.blocks.get(p["name"])
            if block is not None and list(block) != box:  # a block of several boxes
                if len(block) != rank or any(b % m for b, m in zip(block, box)):
                    raise launch.LaunchError(
                        f"box {box} does not tile its block {block}"
                    )
                box, msg = list(block), box
            return Desc(p["name"], "host", p["name"], vals[0]["elem_size"], box,
                        dims[:rank], dims[rank:], msg=msg)  # fmt: skip
        decs = [_zero_length(r["tensormap"]) for r in recs]
        if self.bases is None:
            self.bases = self._assign()
        base = self.bases.get(k, p.get("base"))
        empty = all(d["address"] == 0 for d in decs)
        if base is None and not empty:
            raise launch.LaunchError(
                "its address resolves to no single tensor of the launch"
            )
        eb = (
            self.mem.tensors[base]["elem_bytes"]
            if base
            else max(1, decs[0]["elem_bits"] // 8)
        )
        kds = [launch.kir_desc(d, eb) for d in decs]
        fmts = {
            (tuple(kd["box"]), kd["dim0_ratio"], d["dtype"]) for kd, d in zip(kds, decs)
        }
        if len(fmts) != 1:
            raise launch.LaunchError(
                "launches differ in the descriptor's box or format"
            )
        name = f"tma{self.tmaps.index(ptx)}" + (f"_{base}" if base else "")
        rank = len(kds[0]["box"])
        shape = [self._synthetic(f"{name}.shape.{j}", [kd["shape"][j] for kd in kds])
                 for j in range(rank)]  # fmt: skip
        strides = [self._synthetic(f"{name}.stride.{j}", [kd["strides"][j] for kd in kds])
                   for j in range(rank)]  # fmt: skip
        offs = [0 if base is None else _elem_off(r["tensormap"], base) for r in recs]
        off = E.const(0, 64) if not any(offs) else self._synthetic(f"{name}.off", offs)
        return Desc(name, "host", base, eb, kds[0]["box"], shape, strides, off,
                    tuple(kds[0]["dim0_ratio"]), empty,
                    decs[0]["dtype"] if decs[0]["dtype"] in _PACKED else None)  # fmt: skip

    def _assign(self) -> dict:
        """The base of each descriptor whose address aliases several jit names: one
        whose layout (extents and strides, as the policy's tensors give them for the first
        launch) only one free alias has gets it; the others the free ones in jit order
        (param_table's rule). E.g. an accumulator passed as `acc` [2, 49152] and as
        `acc_tma` [768, 128]: a descriptor of extents [768, 128] is over `acc_tma`."""
        groups: dict = {}
        for k, p in enumerate(self.params):
            if p["kind"] == "tensormap" and len(p.get("aliases") or ()) > 1:
                groups.setdefault(tuple(p["aliases"]), []).append(k)
        out = {}
        for names, ks in groups.items():
            fits = {k: [n for n in names if self._fits(k, n)] for k in ks}
            free = list(names)
            for k in ks:
                if len(fits[k]) == 1 and fits[k][0] in free:
                    out[k] = fits[k][0]
                    free.remove(fits[k][0])
            for k in ks:
                if k not in out:
                    out[k] = free.pop(0) if free else None
        return out

    def _fits(self, k: int, name: str) -> bool:
        """Descriptor parameter k walks tensor `name`'s dims in the first launch."""
        t = self.mem.tensors.get(name, {})
        if "shape" not in t or not self.records:
            return False
        try:
            kd = launch.kir_desc(
                _zero_length(self.records[0]["params"][k]["tensormap"]), t["elem_bytes"]
            )
        except launch.LaunchError:
            return False
        return _dims(kd["shape"], kd["strides"]) == _dims(t["shape"], t["strides"])

    def _own_param(self, name: str):
        """A scalar kernel parameter (TTGIR: a descriptor's shape/stride), by KIR name."""
        k = next(
            k
            for k, p in enumerate(self.params)
            if p["name"] == name and p["kind"] == "scalar"
        )
        w = 8 * self.params[k]["size"]
        return self._param(
            name, max(w, 32), [r["params"][k]["value"] for r in self.records]
        )

    def _synthetic(self, name: str, values: list):
        return self._param(name, 64, values)

    def _param(self, name: str, w: int, values: list):
        self.mem.used_params[name] = w
        for vals, v in zip(self.values, values):
            vals[name] = v
        return E.var(name, w)

    # ------------------------------------------------------- in-kernel ones
    def _dyn(self, tr: Translator) -> dict:
        return self.dyn.setdefault(tr, {"stage": {}, "descs": {}})

    def _active(self, tr: Translator, i, what: str) -> list:
        """The issuing threads (threads() indices), statically known and unconditional."""
        uni, var = tr.split_mask()
        act = tr.active(var)
        if not E.is_const(uni, 1) or not act:
            tr.fail(i, f"{what} under a condition that depends on runtime values "
                       "(or that no thread meets)")  # fmt: skip
        return act

    def staging_store(self, tr: Translator, i) -> None:
        """st.shared (a drop): 32 lanes of one warp writing b32 zeros to the 128 bytes
        from a constant offset zero a tensormap staging buffer; any other store that may
        overlap an open staging buffer closes it (a later cp_fenceproxy of it is refused)."""
        stage = self._dyn(tr)["stage"]
        val = tr.operand_value(i.operands[1], 32) if len(i.operands) > 1 else None
        zero = (
            i.opcode.endswith(".b32") and isinstance(val, tuple) and E.is_const(val, 0)
        )
        if not zero and not stage:
            return  # (most stores: nothing to check)
        a = self.mem.address(tr, i.operands[0])
        p = None if isinstance(a, Data) else _peel(a)
        span = None if p is None else _span(tr, p)
        if (
            zero
            and span is not None
            and span[2] == list(range(span[1], span[1] + 128, 4))
        ):
            stage[(span[0], span[1])] = _Stage({}, _Ctx.of(tr))
            return
        for s, a0 in list(stage):  # a store of unknown or overlapping bytes closes it
            if span is None or span[0] == s and span[1] < a0 + 128 and a0 < span[3]:
                del stage[(s, a0)]

    def _stage(self, tr: Translator, i, op, what: str) -> tuple:
        sym, off = self._smem(tr, i, op, f"the {what} staging address")
        if not E.is_const(off):
            tr.fail(i, f"{what} of a staging buffer at a non-constant offset")
        key = (sym, off[2])
        st = self._dyn(tr)["stage"].get(key)
        if st is None:
            tr.fail(i, f"{what} of a tensormap staging buffer that was not zeroed by "
                       "one warp first (32 lanes x st.shared.b32 0)")  # fmt: skip
        now = _Ctx.of(tr)
        if (
            st.ctx.stmts is not tr.stmts
            or now.regions[: len(st.ctx.regions)] != st.ctx.regions
        ):
            tr.fail(i, f"{what} in another branch or loop than its staging buffer's "
                       "zeroing (a uniform branch could skip it)")  # fmt: skip
        return key, st

    def _replace(self, tr: Translator, i, mn: str) -> None:
        """tensormap.replace.tile.<field>.shared::cta.b1024.<type> [a], {dim,} value."""
        what = mn.split(".")[2]
        _key, st = self._stage(tr, i, i.operands[0], f"tensormap.replace.{what}")
        self._active(tr, i, f"tensormap.replace.{what}")
        dim = None
        if len(i.operands) == 3:
            if i.operands[1][0] != "imm":
                tr.fail(i, "tensormap.replace with a runtime dimension", "error")
            dim = i.operands[1][1]
        v = tr.operand_value(i.operands[-1], 64 if i.opcode.endswith(".b64") else 32)
        u = tr.uniform(tr.as_int(v, f"the tensormap {what}"))
        if u is None:
            tr.fail(i, f"tensormap {what} varies across the threads of the role")
        st.fields[(what, dim)] = u

    def _cp_fenceproxy(self, tr: Translator, i, mn: str) -> None:
        """tensormap.cp_fenceproxy.global.shared::cta... [dst], [staging], 128: the
        descriptor of the staging buffer's fields is at dst from now on."""
        tr.aligned(mn)
        key, st = self._stage(tr, i, i.operands[1], "tensormap.cp_fenceproxy")
        if tuple(_Ctx.of(tr).regions) != st.ctx.regions:
            tr.fail(
                i,
                "tensormap.cp_fenceproxy in a branch inside its staging buffer's zeroing",
            )
        if i.operands[2] != ("imm", 128):
            tr.fail(i, "tensormap.cp_fenceproxy of other than 128 bytes", "error")
        self._active(tr, i, "tensormap.cp_fenceproxy")
        dst = tr.uniform(
            tr.as_int(
                self.mem.address(tr, i.operands[0]), "the descriptor's global address"
            )
        )
        if dst is None:
            tr.fail(
                i, "tensormap.cp_fenceproxy to an address that varies across the role"
            )
        name = f"tmap_{key[0]}_{key[1]}_line{i.line}"
        self._dyn(tr)["descs"][dst] = (
            self._kernel_desc(tr, i, name, st.fields),
            _Ctx.of(tr),
        )

    def _kernel_desc(self, tr: Translator, i, name: str, f: dict) -> Desc:
        missing = [x for x in _KERNEL_FIELDS if (x, None) not in f]
        if missing:
            tr.fail(i, f"in-kernel descriptor without {', '.join(missing)}")
        rank = _const(tr, i, f[("rank", None)], "rank") + 1
        need = [(x, k) for k in range(rank) for x in ("box_dim", "global_dim")] + [
            ("global_stride", k) for k in range(rank - 1)]  # fmt: skip
        if any(x not in f for x in need):
            tr.fail(i, f"in-kernel descriptor of rank {rank} without every box_dim, "
                       "global_dim and global_stride")  # fmt: skip
        for (x, _k), v in f.items():
            if (x == "element_stride" and not E.is_const(v, 1)) or (
                x == "interleave_layout" and not E.is_const(v, 0)
            ):
                tr.fail(i, f"in-kernel descriptor with an {x} other than "
                           f"{1 if x == 'element_stride' else 0} (KIR boxes are dense)")  # fmt: skip
        code = _const(tr, i, f[("elemtype", None)], "elemtype")
        eb = _ELEMTYPE_BYTES.get(code)
        if eb is None:
            tr.fail(
                i,
                f"in-kernel descriptor of elemtype {code} (13-15, packed sub-byte: gap)",
            )
        p = _peel(f[("global_address", None)])
        if (
            p is None
            or not p[0][2].startswith("%g:")
            or p[0][2][3:] not in self.mem.tensors
        ):
            tr.fail(i, "in-kernel descriptor whose global address is no kernel tensor")
        base = p[0][2][3:]
        if eb != self.mem.tensors[base]["elem_bytes"]:
            tr.fail(
                i,
                f"in-kernel descriptor elemtype ({eb} bytes) differs from {base}'s elements",
            )
        off = E.div_exact(E.resize(p[1], 64), eb)
        strides = [
            E.div_exact(E.resize(f[("global_stride", k)], 64), eb)
            for k in range(rank - 1)
        ]
        if off is None or None in strides:
            tr.fail(i, "in-kernel descriptor address or stride not provably a multiple of "
                       "its elements")  # fmt: skip
        box = [_const(tr, i, f[("box_dim", k)], "box_dim") for k in range(rank)]
        shape = [E.resize(f[("global_dim", k)], 64, False) for k in range(rank)]
        return Desc(name, "kernel", base, eb, box[::-1], shape[::-1],
                    ([E.const(1, 64)] + strides)[::-1], off)  # fmt: skip

    # ---------------------------------------------------------- bulk copies
    def _bulk(self, tr: Translator, i, k: int, size_k: int) -> tuple:
        """(tensor, element offset, elements, element bytes) of the global operand k of a
        bulk copy of the constant size operand size_k."""
        g = self.mem._global(tr, i, i.operands[k], 0)
        if g is None:
            tr.fail(i, "bulk copy global operand is a shared-memory address", "error")
        name, off, _n = g
        eb = self.mem.tensors[name]["elem_bytes"]
        n = tr.uniform(tr.as_int(tr.operand(i, size_k, 32), "the bulk copy size"))
        if n is None or not E.is_const(n) or n[2] <= 0 or n[2] % eb:
            tr.fail(
                i,
                f"bulk copy of a size that is not a constant number of {eb}-byte elements",
            )
        return name, off, n[2] // eb, eb

    def _bulk_load(self, tr: Translator, i) -> None:
        """cp.async.bulk.shared(::cluster).global.mbarrier::complete_tx::bytes [dst], [src],
        size, [mbar] {, cache-policy}: gload of the elements, tma_load of bulk_<T>_<n>."""
        name, off, n, eb = self._bulk(tr, i, 1, 2)
        lanes = self._lanes(tr, i, "bulk copy") or _one_thread(tr)
        self.mem._event(tr, "gload", name, off, n)
        d = self.bulk.setdefault(
            f"bulk_{name}_{n}", Desc(f"bulk_{name}_{n}", "bulk", name, eb, [n])
        )
        d.uses.add("tma_load")
        sym, boff = self.mem._smem(tr, i, i.operands[3])
        tr.barrier_event("tma_load", sym, boff, "thread", lanes=lanes, desc=d.name,
                         coords=[off])  # fmt: skip

    def _bulk_out(self, tr: Translator, i, kind: str) -> None:
        """cp.async.bulk.global.shared::cta.bulk_group / cp.reduce.async.bulk.global.
        shared::cta.bulk_group.<op>.<type> [dst], [src], size: gstore/gatomic of the
        elements (a reduce's type must be the tensor's element); a store several lanes
        of a warp may execute, once per distinct (dst, src) (_waterfall)."""
        name, off, n, eb = self._bulk(tr, i, 0, 2)
        if kind == "gatomic" and _reduce_bits(tr, i) != 8 * eb:
            tr.fail(
                i,
                f"bulk reduce of {i.opcode.split('.')[-1]} values into {eb}-byte elements",
            )
        if self._lanes(tr, i, "bulk " + ("reduce" if kind == "gatomic" else "store")) \
                and kind == "gstore":  # fmt: skip
            key = [
                self.mem.address(tr, i.operands[0]),
                self.mem.address(tr, i.operands[1]),
            ]
            self._waterfall(tr, i, key, lambda: self.mem._event(tr, kind, name, off, n))
            return
        self.mem._event(tr, kind, name, off, n)

    # ---------------------------------------------------------------- facts
    def facts(self) -> tuple:
        """(descs, values): Facts.descs {name: {kind, base, elem_bytes, box, shape,
        strides, uses}} of the descriptors with KIR events (shape/strides KIR JSON over
        Facts.params; None for a bulk descriptor), and per launch the values of the
        synthetic params."""
        out = {}
        for d in list(self.host.values()) + list(self.bulk.values()):
            if d.uses:
                out[d.name] = {"kind": d.kind, "base": d.base, "elem_bytes": d.elem_bytes,
                               "box": list(d.box), "uses": sorted(d.uses),
                               "shape": [E.to_json(x) for x in d.shape] if d.kind != "bulk" else None,
                               "strides": [E.to_json(x) for x in d.strides] if d.kind != "bulk" else None}  # fmt: skip
        used = set(self.mem.used_params)
        return out, [
            {n: v for n, v in vals.items() if n in used} for vals in self.values
        ]


# ------------------------------------------------------------------- helpers


def _peel(e):
    from .mem import peel  # mem.py imports this module

    return peel(e)


def _span(tr: Translator, p):
    """(symbol, first byte, sorted per-thread byte offsets, end byte) of a shared store's
    address over its issuing threads, or None if that is not static."""
    if not p[0][2].startswith("%s:"):
        return None
    uni, var = tr.split_mask()
    act = tr.active(var) if E.is_const(uni, 1) else None
    inst = tr.instances(p[1]) if act else []
    offs = [inst[t] for t in act or ()]
    if not offs or not all(E.is_const(o) for o in offs):
        return None
    got = sorted(o[2] for o in offs)
    return p[0][2][3:], got[0], got, got[-1] + 16


def _same(a, b) -> bool:
    """Two mbarrier addresses (shared symbol, offset), or two None, are the same."""
    return a is b or (a is not None and b is not None and a[0] == b[0] and a[1] is b[1])


def _const(tr: Translator, i, v, what: str) -> int:
    if not E.is_const(v):
        tr.fail(i, f"in-kernel descriptor {what} is not a constant")
    return v[2]


def _one_lane_per_warp(tr: Translator) -> bool:
    """At most one lane of each warp issues under the current mask (statically)."""
    act = tr.active(tr.split_mask()[1])
    return act is not None and len({t // 32 for t in act}) == len(act)


def _one_thread(tr: Translator) -> bool:
    """One thread of the role issues (statically): its operands may be lane-dependent."""
    act = tr.active(tr.split_mask()[1])
    return act is not None and len(act) == 1


def _reduce_bits(tr: Translator, i) -> int:
    t = i.opcode.split(".")[-1]
    if t not in _REDUCE_BITS:
        tr.fail(i, f"bulk reduce of type .{t}", "error")
    return _REDUCE_BITS[t]


def _zero_length(d: dict) -> dict:
    """A zero-length tensor's descriptor (address 0): extents 2^32 are 0 (module doc)."""
    if d["address"] != 0:
        return d
    return {**d, "dims": [0 if n == 1 << 32 else n for n in d["dims"]]}


def _dims(shape: list, strides: list) -> list:
    return sorted((n, s) for n, s in zip(shape, strides) if n != 1)


def _elem_off(tm: dict, base: str) -> int:
    return next(c["elem_off"] for c in tm["base"] if c["tensor"] == base)


def tkinfo_version(cubin: bytes) -> str | None:
    """The compiler version ptxas (or the PTX compiler library a CuTe DSL embeds) records
    in a cubin's .note.nv.tkinfo, e.g. "12.9.83" of "release 12.9, V12.9.83"; None if the
    ELF has no such note."""
    try:
        (shoff,) = struct.unpack_from("<Q", cubin, 0x28)
        size, num, stridx = struct.unpack_from("<HHH", cubin, 0x3A)
        secs = [struct.unpack_from("<IIQQQQIIQQ", cubin, shoff + k * size)
                for k in range(num)]  # fmt: skip
        names = secs[stridx][4]
    except (struct.error, IndexError):
        return None
    for sec in secs:
        if cubin.startswith(b".note.nv.tkinfo\0", names + sec[0]):
            m = re.search(
                rb"release [\d.]+, V([\d.]+)", cubin[sec[4] : sec[4] + sec[5]]
            )
            return m.group(1).decode() if m else None
    return None


def desc_base(records: list, k: int, leaves: list, taken: dict) -> tuple:
    """(base, aliases): the tensor CuTe descriptor parameter k points into in every
    launch, from the record's address candidates (launch.resolve): the best match class
    ('exact', else 'inside', else 'shifted'). Several candidates of one storage are one
    host tensor passed under several jit names (aliases, in `leaves`, the jit argument
    order): the k-th such descriptor gets the k-th name, as param_table names aliased
    pointer fields (`taken` counts them; Tma refines it by layout). No candidate or
    distinct storages: (None, [])."""
    sets = set()
    for r in records:
        cands = (r["params"][k].get("tensormap") or {}).get("base") or []
        best = [c for c in cands if c["match"] == cands[0]["match"]]
        if len({tuple(c["storage"]) for c in best}) > 1:
            return None, []
        sets.add(frozenset(c["tensor"] for c in best))
    if len(sets) != 1:
        return None, []
    (names,) = sets
    if len(names) <= 1:
        return next(iter(names), None), []
    order = [n for n in leaves if n in names]
    if len(order) != len(names):
        return None, []
    j = taken.get(names, 0)
    taken[names] = j + 1
    return (order[j] if j < len(order) else None), order


def segment_dims(desc: list, tensor: list) -> list:
    """For each descriptor dim, the tensor dim it walks (same element stride and extent),
    or None: `desc` and `tensor` are per launch [(shape, strides)] in KIR order (the
    descriptor's values and the base tensor's); a match must be unique in every launch and
    the same in all (design §5.1: a descriptor inherits its base's segments by it)."""
    out = []
    for j in range(len(desc[0][0])):
        got = set()
        for (dsh, dst), (tsh, tst) in zip(desc, tensor):
            m = [
                t
                for t, (n, s) in enumerate(zip(tsh, tst))
                if (n, s) == (dsh[j], dst[j])
            ]
            got.add(m[0] if len(m) == 1 else None)
        out.append(got.pop() if len(got) == 1 else None)
    return out


def _box_footprint(j, box, coords, shape, strides, off) -> dict:
    """offsets/mask TExprs of the elements c + i (i in the box) inside [0, shape), at
    off + sum (c + i) * stride (frontend/cute/ops_mem.py _box_footprint, which this copies:
    frontend/ptx is pure stdlib). c + i is computed along its own dim, then broadcast, as
    frontend/ttgir/to_kir.py _clipped_store does: Lean evaluates it over box[d] elements,
    not the whole box (three times: offsets and both bounds)."""
    full = [b for b in box if b > 1] or [1]
    axes = {d: k for k, d in enumerate(d for d, b in enumerate(box) if b > 1)}

    def lift(x):
        return {"t": "splat", "shape": full, "w": 64, "a": j(x)}

    def node(t, w, **kw):
        return {"t": t, "shape": full, "w": w, **kw}

    offsets, mask = None, None
    for d, (c, n, st) in enumerate(zip(coords, shape, strides)):
        pos = lift(c) if d not in axes else None
        if d in axes:
            cur = [box[d]]
            r = {"t": "bin", "shape": [box[d]], "w": 64, "o": "add",
                 "a": {"t": "splat", "shape": [box[d]], "w": 64, "a": j(c)},
                 "b": {"t": "range", "shape": [box[d]], "w": 64, "start": 0, "end": box[d]}}  # fmt: skip
            for ax in range(len(full)):
                if ax != axes[d]:
                    cur = cur[:ax] + [1] + cur[ax:]
                    r = {"t": "expand", "shape": list(cur), "w": 64, "a": r, "axis": ax}
            for ax in range(len(full)):
                if ax != axes[d]:
                    cur[ax] = full[ax]
                    r = {"t": "bcast", "shape": list(cur), "w": 64, "a": r, "axis": ax}
            pos = r
        term = node("bin", 64, o="mul", a=pos, b=lift(st))
        offsets = (
            term if offsets is None else node("bin", 64, o="add", a=offsets, b=term)
        )
        inside = node("bin", 1, o="and", a=node("cmp", 1, o="sge", a=pos, b=node("full", 64, v=0)),
                      b=node("cmp", 1, o="slt", a=pos, b=lift(n)))  # fmt: skip
        mask = inside if mask is None else node("bin", 1, o="and", a=mask, b=inside)
    if not E.is_const(off, 0):
        offsets = node("bin", 64, o="add", a=lift(off), b=offsets)
    return {"offsets": offsets, "mask": mask}
