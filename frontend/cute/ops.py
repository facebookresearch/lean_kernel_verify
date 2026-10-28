# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The CuTe op table, part 1: integers, control flow, thread registers and warp
collectives, CuTe tuples, layouts, pointers and views (references/ir_ops_cute.md).
Memory, copies, mbarriers, TMA and the host side are in ops_mem.py.

Each entry is a handler `h(tr, op) -> value | [values]` (tr: interp.Translator). Ops in
DATA (or with a DATA_PREFIXES prefix) have no global-memory or mbarrier effect; their
results are Data, and a kept use of one is the slice soundness error. Any op without an
entry is a hard error wherever it appears (it could hide an effect): the slicer visits
every op of the function through effect_operands, kept or not.
"""

import re

from . import expr as E, layout as L
from .interp import ArithIter, Composed, Data, int_width, Lay, Opaque, Ptr, Tup, View

# The CuTe DSL whose op definitions and lowering (issuing threads, checked in its PTX)
# this table was written against. to_kir.py refuses dumps made with any other version.
DSL_VERSION = "4.4.2"

HANDLERS: dict = {}
EFFECTS: dict = {}  # op name -> f(tr, op) -> kept operand indices, or None (not an event)
# op name -> f(op) raising ValueError when this instance has no meaning in the table
# (inline PTX is classified per statement).
CLASSIFIERS: dict = {}
# Structure consumed by the enclosing op (interp.Translator.run_block) or by to_kir.py.
STRUCTURE = (
    "cuda.kernel",
    "func.func",
    "scf.yield",
    "scf.condition",
    "func.return",
    "cuda.return",
)

DATA = {
    "arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.negf", "arith.remf",
    "arith.maximumf", "arith.minimumf", "arith.maxnumf", "arith.minnumf", "arith.cmpf",
    "arith.extf", "arith.truncf", "arith.sitofp", "arith.uitofp", "arith.fptosi",
    "arith.fptoui", "arith.bitcast", "nvgpu.cvt_fpext", "llvm.extractvalue",
    "llvm.insertvalue", "llvm.mlir.undef", "llvm.mlir.poison", "llvm.bitcast",
    "nvvm.mul.packed.f32x2", "nvvm.fma.packed.f32x2", "nvvm.sub.packed.f32x2",
    "nvvm.add.packed.f32x2", "nvvm.fmax", "nvvm.fmin", "nvvm.rcp.approx.ftz.f",
    # Synchronization outside the P0 specs (named barriers, fences, register budgets):
    "nvvm.barrier", "nvvm.bar.warp.sync", "nvvm.fence.proxy", "nvvm.fence.mbarrier.init",
    "nvvm.setmaxregister",
    # Tensor memory, MMA and bulk-async group bookkeeping:
    "nvvm.tcgen05.wait", "nvvm.cp.async.bulk.commit.group", "nvvm.cp.async.bulk.wait_group",
    "nvvm.cp.async.commit.group", "nvvm.cp.async.wait.group",
    "cute.mma_atom_call", "cute.mma.make_fragment", "cute.make_tiled_mma", "cute.apply_swizzle",
    "cute_nvgpu.arch.sm100.alloc_tmem", "cute_nvgpu.arch.sm100.dealloc_tmem",
    "cute_nvgpu.arch.sm100.retrieve_tmem_ptr", "cute_nvgpu.arch.sm100.relinquish_tmem_alloc_permit",
    "cute_nvgpu.atom.make_tmem_copy", "cute_nvgpu.atom.make_s2t_copy",
    "cute_nvgpu.atom.get_copy_s2t_smem_desc_view", "cute_nvgpu.prefetch_tma_desc",
    "cute_nvgpu.arch.copy.SM100.tmem_load", "cute_nvgpu.arch.copy.SM100.tmem_store",
    "cute_nvgpu.arch.copy.SM100.copy_s2t",
    "cute_nvgpu.arch.mma.SM100.umma", "cute_nvgpu.arch.mma.SM100.umma_block_scaled",
    "cute_nvgpu.make_umma_smem_desc", "llvm.inttoptr",
    # Register vectors (the vector dialect's memory ops, load/store/gather/scatter/
    # transfer_*, are not listed: they may access global memory):
    "vector.broadcast", "vector.extract", "vector.extractelement", "vector.insert", "vector.insertelement",
    "vector.from_elements", "vector.splat", "vector.shape_cast", "vector.shuffle", "vector.transpose",
    "vector.bitcast", "vector.reduction", "vector.multi_reduction", "vector.fma",
}  # fmt: skip
DATA_PREFIXES = ("math.",)


def spec(op):
    h = HANDLERS.get(op.name)
    if h is not None:
        return h
    if op.name in DATA or op.name.startswith(DATA_PREFIXES):
        return _data_op
    return None


def unknown(op):
    """Why op has no meaning in the op table, or None."""
    if op.name in STRUCTURE:
        return None
    if spec(op) is None:
        return "op is not in the CuTe op table (references/ir_ops_cute.md)"
    classify = CLASSIFIERS.get(op.name)
    if classify is not None:
        try:
            classify(op)
        except ValueError as e:
            return str(e)
    return None


def effect_operands(tr, effects=None):
    """The slicer's callback, called on every op of the function in program order: an op
    without a table meaning is a hard error (kept or not); an event op returns the
    operand indices it keeps (`effects`, default EFFECTS), any other op None."""
    effects = EFFECTS if effects is None else effects

    def f(op):
        tr.cur_loc = tr.k.user_loc(op) or tr.cur_loc
        why = unknown(op)
        if why is not None:
            tr.fail(op, why)
        g = effects.get(op.name)
        return g(tr, op) if g else None

    return f


def op(*names):
    def deco(fn):
        for n in names:
            HANDLERS[n] = fn
        return fn

    return deco


def effect(*names):
    def deco(fn):
        for n in names:
            EFFECTS[n] = fn
        return fn

    return deco


def _data_op(tr, op):
    d = tr.data(op)
    return [d] * len(op.results)


# ----------------------------------------------------------------------------- types


def elem_bits(t: str) -> int:
    """Bits of a scalar element type: iN/uiN/siN, fN, bf16, tf32, fNE... (f8E4M3FN, ...)."""
    t = t.strip()
    m = re.fullmatch(r"(?:s|u)?i(\d+)|f(\d+)(?:E\w+)?|bf(16)|tf(32)", t)
    if not m:
        raise ValueError(f"unknown element type {t!r}")
    return int(next(g for g in m.groups() if g))


def quoted(t: str) -> list:
    return re.findall(r'"([^"]*)"', t)


def fields(t: str) -> list:
    """Top-level comma-separated fields inside the outermost <...> of a type."""
    body = t[t.index("<") + 1 : t.rindex(">")]
    out, depth, cur, q = [], 0, [], False
    for ch in body:
        if ch == '"':
            q = not q
        if not q:
            if ch in "<([{":
                depth += 1
            elif ch in ">)]}":
                depth -= 1
            elif ch == "," and depth == 0:
                out.append("".join(cur).strip())
                cur = []
                continue
        cur.append(ch)
    out.append("".join(cur).strip())
    return out


SPACES = ("gmem", "smem", "rmem", "tmem", "generic")


def memref_info(t: str):
    """!cute.memref<elem, space, ..., "layout"> -> (elem bits, space, layout text)."""
    f = fields(t)
    q = quoted(t)
    return elem_bits(f[0]), f[1], q[-1] if q else None


def ptr_info(t: str):
    """!cute.ptr<elem, space, ...> (or !cute.ptr<space, ...>) -> (elem bits, space)."""
    f = fields(t)
    if f[0] in SPACES:
        return 8, f[0]
    return elem_bits(f[0]), f[1]


def static_value(t: str):
    """The value of a fully static CuTe type (cute.static and result structures)."""
    kind = t.split("<", 1)[0]
    if kind in ("!cute.tiled_copy", "!cute.tiled_mma") or kind.startswith(
        "!cute_nvgpu.atom."
    ):
        return Opaque("atom", (("type", t),))
    q = quoted(t)
    if kind in (
        "!cute.int_tuple",
        "!cute.coord",
        "!cute.shape",
        "!cute.stride",
        "!cute.tile",
    ):
        v = L.parse_tuple(q[0])
    elif kind == "!cute.layout":
        v = Lay(*L.parse_layout(q[0]))
    elif kind == "!cute.composed_layout":
        sw, off, (shape, stride) = L.parse_composed(q[0])
        return Composed(sw, off, Lay(shape, stride))
    else:
        raise ValueError(f"not a static CuTe value type: {t[:80]}")
    if L.dyn_count(v.shape if isinstance(v, Lay) else v) or (
        isinstance(v, Lay) and L.dyn_count(v.stride)
    ):
        raise ValueError(f"type has dynamic leaves: {t[:80]}")
    return v if isinstance(v, Lay) else Tup(v)


def int_array(attr: str) -> list:
    """`array<i32: 0, 2, 1>` -> [0, 2, 1]."""
    m = re.search(r"array<\w+:([^>]*)>", attr or "")
    return [int(x) for x in m.group(1).split(",") if x.strip()] if m else []


def rtype(tr, op, i=0):
    return tr.k.module.resolve_type(op.result_types[i])


def otype(tr, op, i):
    return tr.k.module.resolve_type(op.operand_types[i])


# --------------------------------------------------------------------------- helpers


def dyn_values(tr, op, i):
    """The dynamic leaves operand i contributes to a tuple constructor, in order."""
    v = tr.val(op, i)
    t = otype(tr, op, i)
    if int_width(t) is not None:
        return [tr.as_int(op, v, "tuple construction")]
    if isinstance(v, Data):
        tr.fail(
            op,
            f"data value from {v.origin} reaches a tuple construction (slice soundness)",
        )
    if isinstance(v, Tup):
        ref = L.parse_tuple(quoted(t)[0])
        return [x for x, r in zip(L.leaves(v.t), L.leaves(ref)) if isinstance(r, L.Dyn)]
    tr.fail(op, f"operand of kind {type(v).__name__} in a tuple constructor")


def to_tup(tr, op, v, use="layout operand"):
    if isinstance(v, Tup):
        return v.t
    if isinstance(v, tuple):
        return v
    if isinstance(v, Data):
        tr.fail(op, f"data value from {v.origin} reaches {use} (slice soundness)")
    tr.fail(op, f"{use} must be an IntTuple, got {type(v).__name__}")


def lay_of(tr, op, v):
    if isinstance(v, Lay):
        return v
    if isinstance(v, View):
        return lay_of(tr, op, v.lay)
    if isinstance(v, Composed):
        tr.fail(op, "a swizzled (composed) layout reaches a kept use")
    tr.fail(op, f"expected a layout, got {type(v).__name__}")


def i64(e):
    return E.resize(e, 64)


def advance(tr, op, p, off):
    """Pointer or coordinate iterator plus a crd2idx result (expr or {path: expr})."""
    if isinstance(p, Ptr):
        if isinstance(off, dict):
            tr.fail(op, "ArithTuple offset added to a pointer")
        new = E.bin("add", p.off, i64(off))
        # Shared-memory offsets stay symbolic: barrier addresses are resolved from them.
        return Ptr(
            p.space, p.base, new if p.space == "smem" else tr.bind(new, op, "p"), p.bits
        )
    if isinstance(p, ArithIter):
        comps = list(p.coord)
        if isinstance(off, dict):
            for path, e in off.items():
                if len(path) != 1 or path[0] >= len(comps):
                    tr.fail(
                        op, f"ArithTuple component {path} outside rank {len(comps)}"
                    )
                w = max(E.width(comps[path[0]]), E.width(e))
                comps[path[0]] = tr.bind(
                    E.bin("add", E.resize(comps[path[0]], w), E.resize(e, w)), op
                )
        elif not E.is_const(off, 0):
            tr.fail(op, "scalar offset of a coordinate iterator")
        return ArithIter(tuple(comps))
    tr.fail(op, f"cannot offset a {type(p).__name__}")


def requirement(tr, op):
    """The `require` callback of layout.crd2idx: a KIR assert at this op."""
    return lambda cond, msg: tr.require(op, cond, msg)


def bind_tree(tr, op, t):
    if isinstance(t, list):
        return [bind_tree(tr, op, x) for x in t]
    return tr.bind(t, op) if isinstance(t, tuple) else t


# ---------------------------------------------------------------------------- arith


@op("arith.constant", "llvm.mlir.constant")
def _constant(tr, op):
    w = int_width(rtype(tr, op))
    if w is None:
        return tr.data(op)
    v = op.props.get("value") or op.attrs.get("value", "")
    m = re.match(r"\s*(true|false|-?\d+)\b", v)
    if not m:
        tr.fail(op, f"cannot read integer constant {v!r}")
    s = m.group(1)
    return E.const(1 if s == "true" else 0 if s == "false" else int(s), w)


_BIN = {
    "arith.addi": "add", "arith.subi": "sub", "arith.muli": "mul", "arith.divsi": "sdiv",
    "arith.divui": "udiv", "arith.remsi": "srem", "arith.remui": "urem", "arith.andi": "and",
    "arith.ori": "or", "arith.xori": "xor", "arith.shli": "shl", "arith.shrsi": "ashr",
    "arith.shrui": "lshr", "arith.maxsi": "smax", "arith.minsi": "smin", "arith.maxui": "umax",
    "arith.minui": "umin",
}  # fmt: skip
_DIV = {
    "arith.floordivsi": E.floordiv,
    "arith.ceildivsi": E.ceildiv,
    "arith.ceildivui": E.ceildivu,
}


def _int_operands(tr, op):
    """Integer operands, or (None, Data) if an operand is data (the result is data)."""
    vals = [tr.val(op, i) for i in range(len(op.operands))]
    for v in vals:
        if isinstance(v, Data):
            return None, v
    return [tr.as_int(op, v, "integer arithmetic") for v in vals], None


@op(*_BIN, *_DIV)
def _binop(tr, op):
    if int_width(rtype(tr, op)) is None:
        return tr.data(op)
    xs, d = _int_operands(tr, op)
    if d is not None:
        return d
    a, b = xs
    if op.name in _DIV:
        return tr.bind(_DIV[op.name](a, tr.guard(b, 1)), op)
    o = _BIN[op.name]
    e = E.bin(o, a, tr.guard(b, 1) if o in E.PARTIAL_OPS else b)
    q = fast_division(tr, e) if o == "lshr" else None
    return tr.bind(q if q is not None else e, op)


_CMPI = ("eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge")


@op(
    "arith.cmpi", "llvm.icmp"
)  # same predicate numbering (eq ne slt sle sgt sge ult ule ugt uge)
def _cmpi(tr, op):
    if int_width(rtype(tr, op)) is None:
        return tr.data(op)
    xs, d = _int_operands(tr, op)
    if d is not None:
        return d
    pred = int(re.match(r"\s*(\d+)", op.props["predicate"]).group(1))
    return tr.bind(E.cmp(_CMPI[pred], xs[0], xs[1]), op, "c")


@op("arith.select")
def _select(tr, op):
    if int_width(rtype(tr, op)) is None:
        return tr.data(op)
    xs, d = _int_operands(tr, op)
    return d if d is not None else tr.bind(E.select(*xs), op)


@op(
    "arith.extsi",
    "arith.extui",
    "arith.trunci",
    "arith.index_cast",
    "arith.index_castui",
)
def _cast(tr, op):
    w = int_width(rtype(tr, op))
    if w is None:
        return tr.data(op)
    v = tr.val(op, 0)
    if isinstance(v, Data):
        return v
    x = tr.as_int(op, v, "integer cast")
    return E.resize(x, w, op.name in ("arith.extsi", "arith.index_cast"))


@op("nvvm.mul")
def _nvvm_mul(tr, op):
    """nvvm.mul: the low (`lo`) or high (`hi`) half of the double-width product."""
    if int_width(rtype(tr, op)) is None:
        return tr.data(op)
    xs, d = _int_operands(tr, op)
    if d is not None:
        return d
    a, b = xs
    w = E.width(a)
    mode = re.search(r"mul_mode\s*(\w+)", op.props.get("mode", ""))
    mode = mode.group(1) if mode else "lo"
    if mode == "lo":
        return tr.bind(E.bin("mul", a, b), op)
    if mode != "hi" or w != 32:
        tr.fail(op, f"nvvm.mul mode {mode} on i{w}")
    signed = op.props.get("isSigned", "false").strip().startswith("true")
    wide = E.bin("mul", E.ext(a, 64, signed), E.ext(b, 64, signed))
    return tr.bind(E.trunc(E.bin("lshr", wide, E.const(32, 64)), 32), op)


@op("llvm.intr.ctpop")
def _ctpop(tr, op):
    x = tr.int_of(op, 0, "popcount")
    w = E.width(x)
    total = E.const(0, w)
    for k in range(w):
        total = E.bin(
            "add", total, E.bin("and", E.bin("lshr", x, E.const(k, w)), E.const(1, w))
        )
    return tr.bind(total, op)


# ----------------------------------------------------------------- control and asserts


@op("scf.if")
def _scf_if(tr, op):
    return tr.scf_if(op)


@op("scf.for")
def _scf_for(tr, op):
    return tr.scf_for(op)


@op("scf.while")
def _scf_while(tr, op):
    return tr.scf_while(op)


@op("cf.assert")
def _assert(tr, op):
    c = tr.int_of(op, 0, "assert condition")
    msg = (op.props.get("msg") or op.attrs.get("msg") or '"device assert"').strip('"')
    if E.is_const(tr.mask, 1) and tr.uniform(c) is not None:
        tr.emit(op, {"op": "assert", "cond": E.to_json(tr.uniform(c)), "msg": msg})
        return
    tr.thread_event(
        op,
        [c],
        lambda v, j: {"op": "assert", "cond": j(v[0]), "msg": msg, "loc": tr.loc(op)},
    )


@effect("cf.assert")
def _assert_kept(tr, op):
    return [0]


# ----------------------------------------------------------------- thread registers


@op("nvvm.read.ptx.sreg.tid.x")
def _tid_x(tr, op):
    return E.bin("add", E.bin("mul", E.WARP, E.const(32, 32)), E.LANE)


@op("nvvm.read.ptx.sreg.tid.y", "nvvm.read.ptx.sreg.tid.z")
def _tid_yz(tr, op):
    return E.const(0, 32)  # to_kir.py requires blockDim == (32k, 1, 1)


@op("nvvm.read.ptx.sreg.laneid")
def _laneid(tr, op):
    return E.LANE


@op(
    "nvvm.read.ptx.sreg.ctaid.x",
    "nvvm.read.ptx.sreg.ctaid.y",
    "nvvm.read.ptx.sreg.ctaid.z",
)
def _ctaid(tr, op):
    return E.pid("xyz".index(op.name[-1]))


@op(
    "nvvm.read.ptx.sreg.nctaid.x",
    "nvvm.read.ptx.sreg.nctaid.y",
    "nvvm.read.ptx.sreg.nctaid.z",
)
def _nctaid(tr, op):
    return E.nprog("xyz".index(op.name[-1]))


@op(
    "nvvm.read.ptx.sreg.ntid.x",
    "nvvm.read.ptx.sreg.ntid.y",
    "nvvm.read.ptx.sreg.ntid.z",
)
def _ntid(tr, op):
    return E.const(tr.k.block["xyz".index(op.name[-1])], 32)


def cluster_dims(tr, op) -> tuple:
    """The launch's cluster dims (constants: to_kir.py refuses others)."""
    c = tr.k.cluster
    if any(not isinstance(x, int) or x < 1 for x in c):
        tr.fail(op, f"cluster dims {c} are not constants of the launch")
    return c


def cluster_rank(tr, op):
    """%cluster_ctarank: x % cx + (y % cy) * cx + (z % cz) * cx * cy (consecutive CTAs
    form a cluster; PTX ISA, special registers)."""
    cx, cy, cz = cluster_dims(tr, op)
    rank = E.const(0, 32)
    scale = 1
    for axis, n in enumerate((cx, cy, cz)):
        c = E.bin("urem", E.pid(axis), E.const(n, 32))
        rank = E.bin("add", rank, E.bin("mul", c, E.const(scale, 32)))
        scale *= n
    return rank


@op("nvvm.read.ptx.sreg.cluster.ctarank", "nvvm.read.ptx.sreg.cluster.nctarank")
def _cluster_rank(tr, op):
    if op.name.endswith(".nctarank"):
        cx, cy, cz = cluster_dims(tr, op)
        return E.const(cx * cy * cz, 32)
    return cluster_rank(tr, op)


@op(*(f"nvvm.read.ptx.sreg.{r}.{a}" for r in ("clusterid", "nclusterid", "cluster.ctaid", "cluster.nctaid")
      for a in "xyz"))  # fmt: skip
def _cluster_sreg(tr, op):
    """Cluster coordinates of the CTA (clusterid = ctaid / cluster dim, cluster.ctaid =
    ctaid % cluster dim) and the cluster grid (nclusterid = nctaid / cluster dim)."""
    axis = "xyz".index(op.name[-1])
    n = E.const(cluster_dims(tr, op)[axis], 32)
    kind = op.name.removeprefix("nvvm.read.ptx.sreg.").rsplit(".", 1)[0]
    if kind == "clusterid":
        return E.bin("udiv", E.pid(axis), n)
    if kind == "nclusterid":
        return E.bin("udiv", E.nprog(axis), n)
    if kind == "cluster.ctaid":
        return E.bin("urem", E.pid(axis), n)
    return n


def _no_elect(tr, op, e, what: str):
    if "%elect" in E.free_vars(e):
        tr.fail(op, f"{what} of a value that depends on which lane elect.sync picks")


@op("cute_nvgpu.arch.make_warp_uniform")
def _warp_uniform(tr, op):
    # __shfl_sync(-1, x, 0): every lane gets lane 0's value.
    v = tr.val(op, 0)
    if isinstance(v, Data):
        return v
    x = tr.as_int(op, v, "make_warp_uniform")
    _no_elect(tr, op, x, "make_warp_uniform")
    return E.subst(x, {"%lane": E.const(0, 32)})


@op("nvvm.elect.sync")
def _elect(tr, op):
    """elect.sync with the full member mask: every lane of the warp must execute it (PTX:
    undefined otherwise), so each warp reaching it must be fully active under the mask."""
    if op.operands:
        tr.fail(op, "elect.sync with an explicit member mask")
    _uni, var = tr.split_mask()
    act = tr.active(var)
    if act is None:
        tr.fail(op, "elect.sync under a lane condition that depends on runtime values")
    tr.full_warps(op, act, "elect.sync")
    return E.ELECT


@op("nvvm.shfl.sync")
def _shfl(tr, op):
    """PTX shfl.sync on integers: lane l receives the value of source lane j(l), or its
    own value when j(l) is outside its segment."""
    if int_width(rtype(tr, op)) is None:
        return tr.data(op)
    if not E.is_const(tr.int_of(op, 0, "shfl member mask"), -1):
        tr.fail(op, "shfl.sync with a partial member mask")
    v = tr.int_of(op, 1, "shfl value")
    b = tr.int_of(op, 2, "shfl lane/offset")
    c = tr.int_of(op, 3, "shfl clamp")
    _no_elect(tr, op, v, "shfl.sync")
    kind = re.search(r"shfl_kind\s*(\w+)", op.props.get("kind", "")).group(1)
    # PTX ISA shfl.sync, with lane, bval = b[4:0], cval = c[4:0], segmask = c[12:8].
    lane, m31 = E.LANE, E.const(31, 32)
    bval = E.bin("and", b, m31)
    segmask = E.bin("and", E.bin("lshr", c, E.const(8, 32)), m31)
    notseg = E.bin("xor", segmask, E.const(-1, 32))
    min_lane = E.bin("and", lane, segmask)
    max_lane = E.bin("or", min_lane, E.bin("and", E.bin("and", c, m31), notseg))
    if kind == "up":
        j = E.bin("sub", lane, bval)
        ok = E.cmp("sge", j, max_lane)
    elif kind == "down":
        j = E.bin("add", lane, bval)
        ok = E.cmp("sle", j, max_lane)
    elif kind == "bfly":
        j = E.bin("xor", lane, bval)
        ok = E.cmp("sle", j, max_lane)
    elif kind == "idx":
        j = E.bin("or", min_lane, E.bin("and", bval, notseg))
        ok = E.cmp("sle", j, max_lane)
    else:
        tr.fail(op, f"shfl kind {kind}")
    src = E.select(ok, j, lane)
    if tr.uniform(v) is not None:
        return v
    # Lane l's result is v evaluated at source lane src(l) of the same warp.
    out = []
    for w, ln in tr.threads():
        m = tr.thread_subst(w, ln)
        s = E.subst(src, m)
        out.append(E.subst(E.subst(v, {"%warp": E.const(w, 32)}), {"%lane": s}))
    return tr.lane_table(
        op,
        out[:32] if all(out[i] is out[i % 32] for i in range(len(out))) else out,
        "sh",
    )


@op("nvvm.vote.ballot.sync")
def _ballot(tr, op):
    if not E.is_const(tr.int_of(op, 0, "vote member mask"), -1):
        tr.fail(op, "vote.ballot.sync with a partial member mask")
    p = tr.int_of(op, 1, "ballot predicate")
    _no_elect(tr, op, p, "vote.ballot.sync")
    total = E.const(0, 32)
    for k in range(32):
        bit = E.ext(E.subst(p, {"%lane": E.const(k, 32)}), 32, False)
        total = E.bin("or", total, E.bin("shl", bit, E.const(k, 32)))
    return tr.bind(total, op)


# --------------------------------------------------------------------- cute tuples


@op("cute.static")
def _static(tr, op):
    try:
        return static_value(rtype(tr, op))
    except (ValueError, L.LayoutError) as e:
        tr.fail(op, str(e))


@op(
    "cute.make_int_tuple",
    "cute.make_coord",
    "cute.make_shape",
    "cute.make_stride",
    "cute.make_tile",
)
def _make_tuple(tr, op):
    vals = [x for i in range(len(op.operands)) for x in dyn_values(tr, op, i)]
    return Tup(L.fill(L.parse_tuple(quoted(rtype(tr, op))[0]), vals))


@op("cute.assume")
def _assume(tr, op):
    # Divisibility facts are taken as given (references/frontend_cute.md); the value is unchanged.
    v = tr.val(op, 0)
    return v if isinstance(v, (Tup, Data, Opaque)) else tr.as_int(op, v, "assume")


@op("cute.ptrtoint")
def _ptrtoint(tr, op):
    # The address of a pointer is not known; it is kept symbolically for inttoptr.
    p = tr.val(op, 0)
    return Opaque("address", (("ptr", p),)) if isinstance(p, Ptr) else tr.data(op)


@op("cute.inttoptr")
def _inttoptr(tr, op):
    """inttoptr(ptrtoint(p)) is p (the alignment-assumption idiom); any other integer
    has no known provenance."""
    a = tr.val(op, 0)
    if isinstance(a, Data):
        return a
    if not (isinstance(a, Opaque) and a.kind == "address"):
        return Data(f"pointer of unknown provenance from {op.name} at {tr.loc(op)}")
    p = a.get("ptr")
    bits, space = ptr_info(rtype(tr, op))
    if (bits, space) != (p.bits, p.space):
        tr.fail(
            op,
            f"inttoptr changes the pointer type ({p.bits}-bit {p.space} -> {bits}-bit {space})",
        )
    return p


@op("cute.to_int_tuple", "cute.derefine", "builtin.unrealized_conversion_cast")
def _identity(tr, op):
    return tr.val(op, 0)


@op("cute.get_leaves")
def _get_leaves(tr, op):
    v = tr.val(op, 0)
    if isinstance(v, Data):
        return [v] * len(op.results)
    ls = L.leaves(to_tup(tr, op, v))
    if len(ls) != len(op.results):
        tr.fail(op, f"{len(ls)} leaves for {len(op.results)} results")
    return [Tup(x) for x in ls]


@op("cute.get_scalars")
def _get_scalars(tr, op):
    v = tr.val(op, 0)
    if isinstance(v, Data):
        return [v] * len(op.results)
    t = otype(tr, op, 0)
    if isinstance(v, Lay):
        ref = L.parse_layout(quoted(t)[0])
        vals, refs = (
            L.leaves(v.shape) + L.leaves(v.stride),
            L.leaves(ref[0]) + L.leaves(ref[1]),
        )
    elif isinstance(v, Tup):
        vals, refs = L.leaves(v.t), L.leaves(L.parse_tuple(quoted(t)[0]))
    else:
        tr.fail(op, f"get_scalars of a {type(v).__name__}")
    if "only_dynamic" in op.props:
        vals = [x for x, r in zip(vals, refs) if isinstance(r, L.Dyn)]
    if len(vals) != len(op.results):
        tr.fail(op, f"{len(vals)} scalars for {len(op.results)} results")
    out = []
    for x, rt in zip(vals, op.result_types):
        if not isinstance(x, tuple):
            tr.fail(op, f"a scalar of kind {type(x).__name__}")
        out.append(E.resize(x, int_width(tr.k.module.resolve_type(rt))))
    return out


_TUPLE_OPS = {
    "cute.tuple_add": "add", "cute.tuple_sub": "sub", "cute.tuple_mul": "mul",
    "cute.tuple_div": "div", "cute.tuple_mod": "srem", "cute.ceil_div": "ceil_div",
}  # fmt: skip


@op(*_TUPLE_OPS)
def _tuple_arith(tr, op):
    a, b = tr.val(op, 0), tr.val(op, 1)
    for v in (a, b):
        if isinstance(v, Data):
            return v
    ta, tb = to_tup(tr, op, a), to_tup(tr, op, b)
    if _TUPLE_OPS[op.name] in ("div", "srem", "ceil_div"):
        tb = L.rebuild(
            tb, [tr.guard(x, 1) if isinstance(x, tuple) else x for x in L.leaves(tb)]
        )
    return Tup(bind_tree(tr, op, L.elementwise(_TUPLE_OPS[op.name], ta, tb)))


@op("cute.get_flat_coord")
def _get_flat_coord(tr, op):
    """The coordinate of a flat index in a layout's domain, one integer per top-level mode
    (colexicographic: mode i is (idx / size(modes before i)) % size(mode i))."""
    idx = tr.int_of(op, 0, "flat coordinate index")
    v = tr.val(op, 1)
    if isinstance(v, Data):
        return v
    lay = lay_of(tr, op, v)
    modes = lay.shape if isinstance(lay.shape, list) else [lay.shape]
    ref = L.parse_tuple(quoted(rtype(tr, op))[0])
    refs = ref if isinstance(ref, list) else [ref]
    if len(refs) != len(modes):
        tr.fail(op, f"{len(refs)} coordinates for a layout of rank {len(modes)}")
    out, rest = [], idx
    for m, r in zip(modes, refs):
        n = L.size(m)
        w = max(E.width(rest), E.width(n))
        c = E.bin("srem", L.as_int(rest, w), L.as_int(tr.guard(n, 1), w))
        rest = E.bin("sdiv", L.as_int(rest, w), L.as_int(tr.guard(n, 1), w))
        if isinstance(r, L.Dyn):
            out.append(tr.bind(E.resize(c, r.w), op))
            continue
        if not (isinstance(r, tuple) and E.is_const(r)):
            tr.fail(op, f"flat coordinate leaf {L.to_str(r)}")
        r = E.resize(r, E.width(c))
        tr.require(
            op, E.cmp("eq", c, r), "flat coordinate differs from its static type"
        )
        out.append(r)
    return Tup(out if isinstance(ref, list) else out[0])


@op("cute.append_to_rank")
def _append_to_rank(tr, op):
    """A layout padded to `rank` top-level modes with copies of the second layout."""
    v, pad = tr.val(op, 0), tr.val(op, 1)
    for x in (v, pad):
        if isinstance(x, Data):
            return x
    lay, p = lay_of(tr, op, v), lay_of(tr, op, pad)
    r = int(re.match(r"\s*(\d+)", op.props.get("rank", "")).group(1))
    shape = lay.shape if isinstance(lay.shape, list) else [lay.shape]
    stride = lay.stride if isinstance(lay.stride, list) else [lay.stride]
    while len(shape) < r:
        shape, stride = shape + [p.shape], stride + [p.stride]
    ref_shape, ref_stride = L.parse_layout(quoted(rtype(tr, op))[-1])
    return restructure(
        tr, op, ref_shape, ref_stride, list(zip(L.leaves(shape), L.leaves(stride)))
    )


@op("cute.size")
def _size(tr, op):
    v = tr.val(op, 0)
    if isinstance(v, Data):
        return v
    shape = lay_of(tr, op, v).shape if isinstance(v, (Lay, View)) else to_tup(tr, op, v)
    for k in int_array(op.props.get("mode", "")):
        shape = L.mode(shape, k)
    return Tup(tr.bind(L.size(shape), op))


@op("cute.get_shape", "cute.get_stride")
def _get_shape(tr, op):
    v = tr.val(op, 0)
    if isinstance(v, Data):
        return v
    lay = lay_of(tr, op, v)
    return Tup(lay.shape if op.name.endswith("shape") else lay.stride)


@op("cute.get_layout", "cute.get_iter")
def _view_part(tr, op):
    v = tr.val(op, 0)
    if isinstance(v, Data):
        return v
    if not isinstance(v, View):
        tr.fail(op, f"{op.name} of a {type(v).__name__}")
    return v.lay if op.name.endswith("layout") else v.ptr


def compact_colmajor(shape):
    """Strides of the compact column-major layout of `shape`."""
    acc = [E.const(1, 32)]

    def go(s):
        if isinstance(s, list):
            return [go(x) for x in s]
        cur = acc[0]
        w = max(E.width(cur), E.width(s))
        acc[0] = E.bin("mul", E.resize(cur, w), E.resize(s, w))
        return cur

    return go(shape)


@op("cute.make_layout")
def _make_layout(tr, op):
    ref_shape, ref_stride = L.parse_layout(quoted(rtype(tr, op))[0])
    shape = to_tup(tr, op, tr.val(op, 0)) if op.operands else ref_shape
    if len(op.operands) > 1:
        stride = to_tup(tr, op, tr.val(op, 1))
    elif L.dyn_count(ref_stride):
        stride = compact_colmajor(shape)
    else:
        stride = ref_stride
    if L.dyn_count(shape) or L.dyn_count(stride):
        tr.fail(op, "layout with unresolved dynamic leaves")
    return Lay(shape, stride)


@op("cute.make_view")
def _make_view(tr, op):
    p = tr.val(op, 0)
    if isinstance(p, Data):
        return p
    if len(op.operands) > 1:
        lay = tr.val(op, 1)
        if isinstance(lay, Data):
            return lay
    else:
        lay = static_value(f'!cute.layout<"{quoted(rtype(tr, op))[-1]}">')
    return View(p, lay)


@op("cute.crd2idx")
def _crd2idx(tr, op):
    c, lay = tr.val(op, 0), tr.val(op, 1)
    for v in (c, lay):
        if isinstance(v, Data):
            return v
    lay = lay_of(tr, op, lay)
    r = L.crd2idx(
        to_tup(tr, op, c, "coordinate"), lay.shape, lay.stride, requirement(tr, op)
    )
    ref = L.parse_tuple(quoted(rtype(tr, op))[0])
    if isinstance(r, dict):
        n = L.rank(ref)
        comps = [E.const(0, 32)] * n
        for path, e in r.items():
            if len(path) != 1 or path[0] >= n:
                tr.fail(op, f"ArithTuple component {path} outside rank {n}")
            comps[path[0]] = e
        return Tup(
            [tr.bind(x, op) for x in comps]
            if isinstance(ref, list)
            else tr.bind(comps[0], op)
        )
    if isinstance(ref, list):
        tr.fail(op, "scalar crd2idx result for a tuple-typed index")
    return Tup(tr.bind(r, op))


@op("cute.add_offset")
def _add_offset(tr, op):
    p, o = tr.val(op, 0), tr.val(op, 1)
    for v in (p, o):
        if isinstance(v, Data):
            return v
    off = to_tup(tr, op, o, "offset")
    if isinstance(off, list):
        off = {(k,): x for k, x in enumerate(L.leaves(off))}
    return advance(tr, op, p, off)


@op("cute.make_arith_tuple_iter")
def _arith_iter(tr, op):
    vals = [x for i in range(len(op.operands)) for x in dyn_values(tr, op, i)]
    return ArithIter(
        tuple(L.leaves(L.fill(L.parse_tuple(quoted(rtype(tr, op))[0]), vals)))
    )


@op("cute.deref_arith_tuple_iter")
def _deref_iter(tr, op):
    it = tr.val(op, 0)
    return it if isinstance(it, Data) else Tup(list(it.coord))


@op("cute.make_identity_tensor")
def _identity_tensor(tr, op):
    q = quoted(rtype(tr, op))
    base = L.parse_tuple(q[0])
    shape, stride = L.parse_layout(q[1])
    if op.operands:
        shape = L.fill(shape, dyn_values(tr, op, 0))
    if L.dyn_count(base) or L.dyn_count(stride):
        tr.fail(op, "identity tensor with a dynamic start or stride")
    return View(ArithIter(tuple(L.leaves(base))), Lay(shape, stride))


def slice_leaves(shape, stride, coord):
    """CuTe slice: the (shape, stride) leaves of the modes whose coordinate is `_`."""
    if coord is L.UNDER:
        return list(zip(L.leaves(shape), L.leaves(stride)))
    if isinstance(coord, list):
        return [
            p
            for c, sh, st in zip(coord, shape, stride)
            for p in slice_leaves(sh, st, c)
        ]
    return []


def restructure(tr, op, ref_shape, ref_stride, pairs):
    """Leaves arranged in the structure of the result type; static leaves are checked."""
    rs, rd = L.leaves(ref_shape), L.leaves(ref_stride)
    if len(rs) != len(pairs):
        tr.fail(op, f"{len(pairs)} leaves for a result type with {len(rs)}")
    # A size-1 mode never moves; the type gives its stride (CuTe uses 0).
    pairs = [
        (a, b) if isinstance(a, tuple) and E.is_const(a, 1) else p
        for p, a, b in zip(pairs, rs, rd)
    ]
    for (s, d), a, b in zip(pairs, rs, rd):
        for x, ref in ((s, a), (d, b)):
            static = isinstance(ref, tuple) or isinstance(ref, L.Basis)
            if static and not (
                x == ref
                or (
                    isinstance(x, tuple)
                    and isinstance(ref, tuple)
                    and E.is_const(x)
                    and x[2] == ref[2]
                )
            ):
                tr.fail(
                    op,
                    f"computed leaf {L.to_str(x)} disagrees with the type's {L.to_str(ref)}",
                )
    return Lay(
        L.rebuild(ref_shape, [p[0] for p in pairs]),
        L.rebuild(ref_stride, [p[1] for p in pairs]),
    )


@op("cute.slice")
def _slice(tr, op):
    v, c = tr.val(op, 0), tr.val(op, 1)
    for x in (v, c):
        if isinstance(x, Data):
            return x
    coord = to_tup(tr, op, c, "slice coordinate")
    lay = lay_of(tr, op, v)
    ref_shape, ref_stride = L.parse_layout(quoted(rtype(tr, op))[-1])
    kept = restructure(
        tr, op, ref_shape, ref_stride, slice_leaves(lay.shape, lay.stride, coord)
    )
    if isinstance(v, Lay):
        return kept
    return View(
        advance(
            tr, op, v.ptr, L.crd2idx(coord, lay.shape, lay.stride, requirement(tr, op))
        ),
        kept,
    )


@op("cute.filter_zeros")
def _filter_zeros(tr, op):
    """Modes with a static zero stride get extent 1 (CuTe filter_zeros)."""
    v = tr.val(op, 0)
    if isinstance(v, Data):
        return v
    lay = lay_of(tr, op, v)
    shape = L.rebuild(lay.shape, [E.const(1, 32) if isinstance(d, tuple) and E.is_const(d, 0) else n
                                  for n, d in zip(L.leaves(lay.shape), L.leaves(lay.stride))])  # fmt: skip
    ref_shape, ref_stride = L.parse_layout(quoted(rtype(tr, op))[-1])
    out = restructure(
        tr, op, ref_shape, ref_stride, list(zip(L.leaves(shape), L.leaves(lay.stride)))
    )
    return out if isinstance(v, Lay) else View(v.ptr, out)


@op("cute.select")
def _select_modes(tr, op):
    v = tr.val(op, 0)
    if isinstance(v, Data):
        return v
    modes = int_array(op.props.get("mode", ""))
    if isinstance(v, Lay):
        return Lay(
            [L.mode(v.shape, k) for k in modes], [L.mode(v.stride, k) for k in modes]
        )
    if isinstance(v, Tup):
        return Tup([L.mode(v.t, k) for k in modes])
    tr.fail(op, f"select of a {type(v).__name__}")


@op("cute.recast_layout")
def _recast_layout(tr, op):
    v = tr.val(op, 0)
    if isinstance(v, Data):
        return v
    try:
        return static_value(rtype(tr, op))
    except (ValueError, L.LayoutError):
        tr.fail(op, "recast of a layout with dynamic leaves")


@op("cute.recast_iter")
def _recast_iter(tr, op):
    p = tr.val(op, 0)
    if isinstance(p, Data):
        return p
    if not isinstance(p, Ptr):
        tr.fail(op, f"recast of a {type(p).__name__}")
    bits, _space = ptr_info(rtype(tr, op))
    if p.bits == bits:
        return p
    if p.bits > bits:
        return Ptr(
            p.space, p.base, E.bin("mul", p.off, E.const(p.bits // bits, 64)), bits
        )
    if bits % p.bits:
        tr.fail(op, f"recast from {p.bits}-bit to {bits}-bit elements")
    # A wider element type: the offset must be a whole number of wide elements. The
    # DSL's alignment types assume it; the KIR checks it with an assert.
    k = E.const(bits // p.bits, 64)
    tr.require(
        op,
        E.cmp("eq", E.bin("srem", p.off, k), E.const(0, 64)),
        "recast_iter to a wider element at an unaligned offset",
    )
    off = E.bin("sdiv", p.off, k)
    return Ptr(
        p.space, p.base, off if p.space == "smem" else tr.bind(off, op, "p"), bits
    )


@op("cute_nvgpu.arch.get_dyn_smem")
def _dyn_smem(tr, op):
    return Ptr("smem", "dyn", E.const(0, 64), 8)


@op("cute_nvgpu.arch.get_dyn_smem_size")
def _dyn_smem_size(tr, op):
    return E.resize(tr.k.smem, int_width(rtype(tr, op)), False)


@op("cute.make_atom", "cute.make_tiled_copy")
def _make_atom(tr, op):
    return Opaque("atom", (("type", rtype(tr, op)),))


@op("cute.memref.alloca")
def _alloca(tr, op):
    bits, space, ltxt = memref_info(rtype(tr, op))
    return View(
        Ptr(space, tr.fresh("alloca"), E.const(0, 64), bits),
        static_value(f'!cute.layout<"{ltxt}">'),
    )


@op("cute.fast_divmod.create_divisor", "cute.fast_divmod.make_divisor")
def _divisor(tr, op):
    return Opaque("divisor", (("d", tr.int_of(op, 0, "fast divmod divisor")),))


def _divisor_of(tr, op):
    v = tr.val(op, 0)
    if isinstance(v, Data):
        return None, v
    if not (isinstance(v, Opaque) and v.kind == "divisor"):
        tr.fail(op, f"{op.name} of a {type(v).__name__}, not a fast divmod divisor")
    return v.get("d"), None


@op("cute.fast_divmod.get_divisor")
def _divisor_value(tr, op):
    d, data = _divisor_of(tr, op)
    return data if data is not None else d


# The fast unsigned division idiom on a divisor's auxiliary values (multiplier m, shifts
# s1, s2): q = (mulhi(n, m) + ((n - mulhi(n, m)) >> s1)) >> s2 is floor(n / d) for every
# 32-bit unsigned n (the round-up method the DSL's divisor implements). get_aux yields
# reserved variables %fdm<k>_{m,s1,s2} (never in KIR); the shift that completes the idiom
# becomes udiv(n, d) (_binop, fast_division).
FDM = re.compile(r"%fdm(\d+)_(m|s1|s2)$")


@op("cute.fast_divmod.get_aux")
def _divisor_aux(tr, op):
    d, data = _divisor_of(tr, op)
    if data is not None:
        return [data] * 3
    if d not in tr.k.fdm:
        tr.k.fdm.append(d)
    k = tr.k.fdm.index(d)
    widths = [int_width(rtype(tr, op, i)) for i in range(3)]
    return [E.var(f"%fdm{k}_{n}", w) for n, w in zip(("m", "s1", "s2"), widths)]


def _fdm_var(e, part: str):
    """k if e is the reserved variable %fdm<k>_<part> (possibly zero-extended), else None."""
    if e[0] == "ext" and not e[2]:
        e = e[3]
    m = FDM.match(e[2]) if e[0] == "var" else None
    return int(m.group(1)) if m and m.group(2) == part else None


def fast_division(tr, e):
    """udiv(n, d) if e is the fast division idiom on divisor k's aux values, else None."""
    if not (e[0] == "bin" and e[2] == "lshr" and e[3][0] == "bin" and e[3][2] == "add"):
        return None
    k = _fdm_var(e[4], "s2")
    t1, rhs = e[3][3], e[3][4]
    if k is None or not (
        rhs[0] == "bin" and rhs[2] == "lshr" and _fdm_var(rhs[4], "s1") == k
    ):
        return None
    diff = rhs[3]
    if not (diff[0] == "bin" and diff[2] == "sub" and diff[4] is t1):
        return None
    n = diff[3]
    mul_hi = E.trunc(E.bin("lshr", E.bin("mul", E.ext(n, 64, False), E.ext(E.var(f"%fdm{k}_m", 32), 64, False)),
                           E.const(32, 64)), 32)  # fmt: skip
    if t1 is not mul_hi:
        return None
    return E.bin("udiv", n, tr.guard(tr.k.fdm[k], 1))
