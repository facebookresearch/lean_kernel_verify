# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The CuTe op table, part 2: global memory, TMA and bulk copies, mbarriers, inline PTX
and the host-side ops (references/ir_ops_cute.md).

Global accesses become gload/gstore/gatomic events whose offsets are in the KIR tensor's
storage elements: a TExpr over the elements of one access (shape = the extents of the
accessed layout's modes), issued once per thread (interp.thread_event; TMA stores once
per distinct operand tuple of a warp, interp.waterfall_event). mbarrier events
carry the barrier's shared-memory byte address; to_kir.py turns it into (barrier array,
index) once every `mbarrier.init` of the kernel is known.
"""

import math
import re

from . import expr as E, layout as L
from .interp import ArithIter, Data, Lay, Opaque, Ptr, Remote, Translator, Tup, View
from .ops import (
    advance,
    CLASSIFIERS,
    effect,
    elem_bits,
    fields,
    i64,
    int_array,
    lay_of,
    op,
    otype,
    quoted,
    requirement,
    rtype,
    to_tup,
)  # fmt: skip

# ------------------------------------------------------------------ global accesses


def space_of(t: str):
    """Address space of a CuTe memref/pointer, coordinate tensor or LLVM pointer type."""
    if t.startswith(("!cute.memref<", "!cute.ptr<")):
        f = fields(t)
        return f[0] if f[0] in ("gmem", "smem", "rmem", "tmem", "generic") else f[1]
    if t.startswith("!cute.coord_tensor<"):
        return "coord"
    m = re.match(r"!llvm\.ptr(?:<(\d+)>)?", t)
    if m:
        return {"1": "gmem", "3": "smem"}.get(m.group(1), "generic")
    return None


def _offsets_texpr(j, base, dims):
    """TExpr: base + sum_k i_k * stride_k over i in the box `dims` = [(extent, stride)]
    (strides are expressions); all at i64."""
    dims = [(n, s) for n, s in dims if n > 1]
    shape = [n for n, _ in dims] or [1]
    out = {"t": "splat", "shape": shape, "w": 64, "a": j(base)}
    for axis, (n, s) in enumerate(dims):
        r = {"t": "range", "shape": [n], "w": 64, "start": 0, "end": n}
        cur = [n]
        for ax in range(len(shape)):
            if ax != axis:
                cur = cur[:ax] + [1] + cur[ax:]
                r = {"t": "expand", "shape": list(cur), "w": 64, "a": r, "axis": ax}
        for ax in range(len(shape)):
            if ax != axis:
                cur[ax] = shape[ax]
                r = {"t": "bcast", "shape": list(cur), "w": 64, "a": r, "axis": ax}
        st = (
            {"t": "full", "shape": shape, "w": 64, "v": s[2]}
            if E.is_const(s)
            else {"t": "splat", "shape": shape, "w": 64, "a": j(s)}
        )
        out = {"t": "bin", "shape": shape, "w": 64, "o": "add", "a": out,
               "b": {"t": "bin", "shape": shape, "w": 64, "o": "mul", "a": r, "b": st}}  # fmt: skip
    return out


def _storage_dims(tr, op, ptr: Ptr, dims: list, bits: int):
    """(scale, dims) converting an access of `bits`-bit elements at element offsets
    (in ptr.bits units) into the KIR tensor's storage elements."""
    t = tr.k.tensors.get(ptr.base)
    if t is None:
        tr.fail(op, f"global pointer into {ptr.base!r}, which is not a kernel tensor")
    sb = t["elem_bytes"] * 8
    if ptr.bits == sb and bits == sb:
        return 1, dims
    if ptr.bits % sb or bits % sb:
        tr.fail(
            op,
            f"{bits}-bit access through a {ptr.bits}-bit pointer into {sb}-bit storage (sub-element access)",
        )
    k = ptr.bits // sb
    per = bits // sb
    out = [(n, E.bin("mul", i64(s), E.const(k, 64))) for n, s in dims]
    return k, out + ([(per, E.const(1, 64))] if per > 1 else [])


def gaccess(tr, op, kind: str, ptr: Ptr, dims: list, bits: int | None = None):
    """A gload/gstore/gatomic event over the elements ptr + sum_k i_k * stride_k (dims in
    ptr.bits elements), each `bits` wide (default ptr.bits)."""
    if not isinstance(ptr, Ptr) or ptr.space != "gmem":
        tr.fail(
            op, "global access through a pointer that is not a kernel tensor pointer"
        )
    k, sdims = _storage_dims(tr, op, ptr, dims, bits or ptr.bits)
    base = ptr.off if k == 1 else E.bin("mul", ptr.off, E.const(k, 64))
    tr.k.tensors[ptr.base]["uses"].add(kind)
    tr.thread_event(
        op, [base] + [s for _n, s in sdims],
        lambda v, j: {"op": "event", "ev": kind, "tensor": ptr.base, "loc": tr.loc(op), "mask": None,
                      "offsets": _offsets_texpr(j, v[0], [(n, x) for (n, _s), x in zip(sdims, v[1:])])},
    )  # fmt: skip


def _layout_dims(tr, op, lay):
    """Leaves (extent, stride) of a layout with static extents."""
    out = []
    for n, s in zip(L.leaves(lay.shape), L.leaves(lay.stride)):
        if not E.is_const(n):
            tr.fail(op, "access over a layout with a dynamic extent")
        if not isinstance(s, tuple):
            tr.fail(op, f"access over a layout with stride {L.to_str(s)}")
        out.append((n[2], s))
    return out


def metadata_value(tr, op, ptr: Ptr, idx, w: int):
    """The value an integer load reads: element `idx` of a metadata tensor. A load the
    case names as a work-item anchor also sets that work field to the element index."""
    t = tr.k.tensors[ptr.base]
    if t["elem_bytes"] * 8 != ptr.bits or ptr.bits != w:
        tr.fail(op, "integer load whose width differs from the tensor's element")
    t["metadata"] = True
    flat = E.bin("add", ptr.off, i64(idx))
    value = tr.bind(E.load(ptr.base, tr.guard(flat, 0), w), op, "ld")
    where = f"{tr.loc(op)} {tr.k.loc_text(op)}"
    for name, anchors in tr.k.work.items():
        if any(a["load"] == ptr.base and a["where"] in where for a in anchors):
            # The work item is the role's (the index, not which threads read it).
            z = tr.uniform(flat)
            if z is None:
                tr.fail(
                    op,
                    f"work field '{name}': the anchored load's index varies across the role's threads",
                )
            tr.emit(op, {"op": "let", "var": f"work_{name}", "expr": E.to_json(z)})
            fields = {f: {"e": "var", "w": 64, "name": f"work_{f}"} for f in tr.k.work}
            tr.emit(op, {"op": "event", "ev": "work", "fields": fields})
    return value


def view_space(tr, op, v, space: str):
    """The address space a view's accesses go to: the type's space, or for a generic view
    its pointer's. A gmem or generic view computed from data, or a generic one whose
    pointer's space is not known, is a hard error (it may be a global access)."""
    if space not in ("gmem", "generic"):
        return space
    if isinstance(v, Data):
        tr.fail(op, f"{space} view computed from data ({v.origin}) (slice soundness)")
    if not (isinstance(v, View) and isinstance(v.ptr, Ptr)) or v.ptr.space == "generic":
        tr.fail(op, f"{space} access through a view whose address space is not known")
    return v.ptr.space


@op("cute.memref.load", "cute.memref.store")
def _memref_ls(tr, op):
    is_load = op.name.endswith("load")
    v = tr.val(op, 0)
    space = view_space(tr, op, v, space_of(otype(tr, op, 0)))
    if space == "coord" and not isinstance(v, Data):
        lay = lay_of(tr, op, v)
        c = to_tup(tr, op, tr.val(op, 1), "coordinate")
        return Tup(
            list(
                advance(
                    tr,
                    op,
                    v.ptr,
                    L.crd2idx(c, lay.shape, lay.stride, requirement(tr, op)),
                ).coord
            )
        )
    if space != "gmem":
        return tr.data(op) if is_load else None
    lay = lay_of(tr, op, v)
    idx = L.crd2idx(
        to_tup(tr, op, tr.val(op, 1), "coordinate"),
        lay.shape,
        lay.stride,
        requirement(tr, op),
    )
    p = Ptr(v.ptr.space, v.ptr.base, E.bin("add", v.ptr.off, i64(idx)), v.ptr.bits)
    gaccess(tr, op, "gload" if is_load else "gstore", p, [])
    if is_load and re.fullmatch(r"i\d+", rtype(tr, op).strip()):
        return metadata_value(tr, op, v.ptr, idx, int(rtype(tr, op).strip()[1:]))
    return tr.data(op) if is_load else None


@effect("cute.memref.load", "cute.memref.store")
def _memref_ls_kept(tr, op):
    return [0, 1] if space_of(otype(tr, op, 0)) in ("gmem", "generic") else None


@op("cute.memref.load_vec", "cute.memref.store_vec")
def _memref_vec(tr, op):
    is_load = op.name.endswith("load_vec")
    vi = 0 if is_load else 1
    v = tr.val(op, vi)
    if view_space(tr, op, v, space_of(otype(tr, op, vi))) != "gmem":
        return tr.data(op) if is_load else None
    gaccess(
        tr,
        op,
        "gload" if is_load else "gstore",
        v.ptr,
        _layout_dims(tr, op, lay_of(tr, op, v)),
    )
    return tr.data(op) if is_load else None


@effect("cute.memref.load_vec", "cute.memref.store_vec")
def _memref_vec_kept(tr, op):
    vi = 0 if op.name.endswith("load_vec") else 1
    return [vi] if space_of(otype(tr, op, vi)) in ("gmem", "generic") else None


def _value_bits(t: str):
    """(element count, element bits) of an LLVM value type."""
    m = re.fullmatch(r"vector<(\d+)x(\w+)>", t.strip())
    return (int(m.group(1)), elem_bits(m.group(2))) if m else (1, elem_bits(t.strip()))


@op("llvm.load", "llvm.store")
def _llvm_ls(tr, op):
    """Loads/stores through LLVM pointers: global when the pointer is (or derives from)
    a global CuTe pointer; shared/register memory is data."""
    is_load = op.name == "llvm.load"
    pi = 0 if is_load else 1
    if space_of(otype(tr, op, pi)) == "smem":
        return tr.data(op) if is_load else None
    p = tr.val(op, pi)
    if not isinstance(p, Ptr):
        tr.fail(op, "access through a pointer of unknown provenance")
    if p.space != "gmem":
        return tr.data(op) if is_load else None
    n, bits = _value_bits(rtype(tr, op) if is_load else otype(tr, op, 0))
    kind = "gload" if is_load else "gstore"
    if bits % p.bits == 0:  # n elements, each a whole number of pointer elements
        gaccess(tr, op, kind, p, [(n, E.const(bits // p.bits, 64))], bits)
    else:  # elements narrower than the pointer's: one access of n * bits
        gaccess(tr, op, kind, p, [], n * bits)
    return tr.data(op) if is_load else None


def _static_space(tr, value: str):
    """The address space an LLVM pointer surely has from its definition: an alloca, or a
    cast of a register or shared-memory CuTe pointer (through GEPs); else None."""
    for _ in range(32):
        d = tr.k.def_of(value.split("#")[0])
        if d is None:
            return None
        if d.name == "llvm.alloca":
            return "rmem"
        if d.name == "llvm.getelementptr" and d.operands:
            value = d.operands[0]
            continue
        if d.name == "builtin.unrealized_conversion_cast" and d.operand_types:
            sp = space_of(tr.k.module.resolve_type(d.operand_types[0]))
            return sp if sp in ("rmem", "smem") else None
        return None
    return None


@effect("llvm.load", "llvm.store")
def _llvm_ls_kept(tr, op):
    pi = 0 if op.name == "llvm.load" else 1
    if (
        space_of(otype(tr, op, pi)) == "smem"
        or _static_space(tr, op.operands[pi]) is not None
    ):
        return None  # shared or register memory: no event
    return [pi]


@op("llvm.alloca")
def _llvm_alloca(tr, op):
    return Ptr("rmem", tr.fresh("alloca"), E.const(0, 64), 8)


_LLVM_BYTES = {
    "i1": 1,
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "i64": 8,
    "f16": 2,
    "bf16": 2,
    "f32": 4,
    "f64": 8,
}


def _llvm_size(t: str) -> int:
    t = t.strip()
    if t in _LLVM_BYTES:
        return _LLVM_BYTES[t]
    m = re.fullmatch(r"!llvm\.array<(\d+) x (.+)>", t) or re.fullmatch(
        r"vector<(\d+)x(.+)>", t
    )
    if m:
        return int(m.group(1)) * _llvm_size(m.group(2))
    raise ValueError(f"size of LLVM type {t}")


@op("llvm.getelementptr")
def _gep(tr, op):
    """Byte offset sum_k idx_k * sizeof(type_k); kept in the pointer's element units when
    every term is a whole number of elements."""
    base = tr.val(op, 0)
    if not isinstance(base, Ptr):
        return base if isinstance(base, Data) else tr.data(op)
    if base.bits % 8:
        tr.fail(op, "getelementptr on a sub-byte pointer")
    raw = int_array(op.props["rawConstantIndices"])
    dyn = iter(range(1, len(op.operands)))
    t = op.props["elem_type"]
    terms = []
    for k, r in enumerate(raw):
        idx = (
            E.const(r, 64)
            if r != -(1 << 31)
            else i64(tr.int_of(op, next(dyn), "getelementptr index"))
        )
        if k > 0:
            m = re.fullmatch(r"!llvm\.array<\d+ x (.+)>", t.strip()) or re.fullmatch(
                r"vector<\d+x(.+)>", t.strip()
            )
            if not m:
                tr.fail(op, f"getelementptr into {t}")
            t = m.group(1)
        terms.append((idx, _llvm_size(t)))
    eb = base.bits // 8
    unit = eb if all(size % eb == 0 for _i, size in terms) else 1
    off = E.bin("mul", base.off, E.const(eb // unit, 64))
    for idx, size in terms:
        off = E.bin("add", off, E.bin("mul", idx, E.const(size // unit, 64)))
    return Ptr(
        base.space,
        base.base,
        off if base.space == "smem" else tr.bind(off, op, "p"),
        unit * 8,
    )


@op("cute_nvgpu.arch.copy.SM80.cp_async")
def _cp_async(tr, op):
    """cp.async global -> shared of one source element (no mbarrier); with a predicate
    operand, a false predicate reads nothing (src-size 0, zero fill)."""
    src = tr.val(op, 1)
    if space_of(otype(tr, op, 1)) != "gmem" or not isinstance(src, Ptr):
        tr.fail(op, "cp.async source is not a kernel tensor pointer")
    saved = tr.mask
    if len(op.operands) > 2:
        tr.mask = E.and1(saved, tr.int_of(op, 2, "cp.async predicate"))
    gaccess(tr, op, "gload", src, [])
    tr.mask = saved


@effect("cute_nvgpu.arch.copy.SM80.cp_async")
def _cp_async_kept(tr, op):
    return list(range(1, len(op.operands)))


# ------------------------------------------------------------- TMA and bulk copies


@op("cute_nvgpu.atom.make_exec_tma")
def _exec_tma(tr, op):
    return tr.val(op, 0)


@op("cute_nvgpu.atom.set_value")
def _set_value(tr, op):
    a, v = tr.val(op, 0), tr.val(op, 1)
    if isinstance(a, Data):
        return a
    return a.with_(**{_field(op): v})


@op("cute_nvgpu.atom.get_value")
def _get_value(tr, op):
    a = tr.val(op, 0)
    if isinstance(a, Data):
        return a
    return a.get(_field(op), Data(f"field {_field(op)} of a copy atom at {tr.loc(op)}"))


def _field(op) -> str:
    m = re.search(r"<(\w+)>", op.props.get("field", ""))
    if m is None:
        raise ValueError(f"copy atom field {op.props.get('field')!r}")
    return m.group(1)


@op("cute_nvgpu.get_tma_desc_addr")
def _desc_addr(tr, op):
    a = tr.val(op, 0)
    if not isinstance(a, Opaque) or a.get("desc") is None:
        tr.fail(op, "TMA descriptor of an atom that is not a host TMA descriptor")
    return Opaque("tma_desc", (("desc", a.get("desc")),))


def _segments(op) -> list:
    return int_array(op.props["operandSegmentSizes"])


def _segment(op, k: int) -> list:
    sizes = _segments(op)
    start = sum(sizes[:k])
    return list(range(start, start + sizes[k]))


def _tma_coords(tr, op, idxs, desc: str) -> list:
    """Coordinates in descriptor dim order (dim 0 innermost), converted from TMA format
    units to the KIR descriptor's units (_make_tma): dim 0 times the format's width in
    elements, or divided by the packing factor of sub-byte data (which must divide it)."""
    d = tr.k.descs[desc]
    if len(idxs) != len(d["box"]):
        tr.fail(
            op, f"{len(idxs)} TMA coordinates for a rank-{len(d['box'])} descriptor"
        )
    out = [tr.int_of(op, i, "TMA coordinate") for i in idxs]
    if d["scale"] > 1:
        out[0] = E.bin("mul", out[0], E.const(d["scale"], E.width(out[0])))
    if d["pack"] > 1:
        c = out[0]
        k = E.const(d["pack"], E.width(c))
        tr.require(
            op,
            E.cmp("eq", E.bin("srem", c, k), E.const(0, E.width(c))),
            "packed sub-byte TMA coordinate is not byte aligned",
        )
        out[0] = E.bin("sdiv", c, k)
    return out


def _check_attr(tr, op, name: str, want: str):
    got = op.props.get(name, op.attrs.get(name, ""))
    if want not in got:
        tr.fail(op, f"{name} = {got} (only {want} is supported)")


@op("cute_nvgpu.arch.copy.SM100.tma_load")
def _tma_load(tr, op):
    """TMA global -> shared: a tma_load on the descriptor (transaction bytes, P0-4) and a
    gload on its kernel tensor with exactly the copy's footprint (the box elements inside
    the descriptor's extents), so that P0-1 checks the read against the tensor's
    allocation, not only against the extents the descriptor claims."""
    # Operands: descriptor, smem destination, mbarrier, coordinates (descriptor dim
    # order), multicast mask, im2col offsets, cache policy. num_cta = 2 (cta_group::2):
    # each destination CTA's bytes complete on the barrier at the same offset in the even
    # CTA of its pair (the DSL clears the peer bit of the barrier address, as CUTLASS's
    # SM100_TMA_2SM_LOAD); a multicast mask (bit r: rank r) writes the box into every CTA
    # it names and signals each destination's barrier.
    _check_attr(tr, op, "mode", "tiled")
    ncta = op.props.get("num_cta", op.attrs.get("num_cta", ""))
    if not re.match(r"\s*[12] : i32", ncta):
        tr.fail(op, f"num_cta = {ncta} (only cta_group::1 and ::2 are supported)")
    leader = ncta.strip().startswith("2")
    if _segment(op, 5):
        tr.fail(op, "im2col TMA load")
    extra = {}
    if _segment(op, 4):
        mask = tr.uniform(tr.int_of(op, _segment(op, 4)[0], "TMA multicast mask"))
        if mask is None:
            tr.fail(op, "TMA multicast mask varies across the threads of the role")
        extra["cta_mask"] = E.resize(mask, 32, False)
    if leader:
        extra["leader"] = True
    desc = tr.val(op, 0)
    if not isinstance(desc, Opaque) or desc.kind != "tma_desc":
        tr.fail(op, "TMA load through a descriptor that is not a host TMA descriptor")
    name = desc.get("desc")
    d = tr.k.descs[name]
    t = tr.k.tensors[d["base"]]
    if d["elem_bytes"] != t["elem_bytes"]:
        tr.fail(
            op,
            f"TMA load in {d['elem_bytes']}-byte units from a tensor of {t['elem_bytes']}-byte elements",
        )
    coords = _tma_coords(tr, op, _segment(op, 3), name)
    tr.barrier_event(
        op, "tma_load", tr.val(op, 2), "warp", desc=name, coords=coords, **extra
    )
    d["uses"].add("tma_load")
    t["uses"].add("gload")
    r = len(coords)
    v = [E.resize(tr.uniform(c), 64) for c in coords] + [
        E.resize(x, 64) for x in d["shape"] + d["strides"]
    ]
    tr.warp_event(op, {"op": "event", "ev": "gload", "tensor": d["base"], "loc": tr.loc(op),
                       **_box_footprint(E.to_json, d["box"], v[:r], v[r : 2 * r], v[2 * r :])})  # fmt: skip


@effect("cute_nvgpu.arch.copy.SM100.tma_load")
def _tma_load_kept(tr, op):
    return [0, 2] + _segment(op, 3) + _segment(op, 4)


@op("cute_nvgpu.arch.copy.SM100.tma_store", "cute_nvgpu.arch.copy.SM100.tma_reduce")
def _tma_store(tr, op):
    """TMA shared -> global: a gstore (gatomic for reduce-add) on the descriptor's kernel
    tensor with exactly the copy's footprint, the box elements at the coordinates that
    lie inside the descriptor's extents (TMA drops the others). Unlike a tma_store event
    this keeps every write to a tensor on one KIR tensor, whether it goes through TMA or
    through per-thread stores (KIR v1 cannot say that a descriptor aliases a tensor).
    Issuing threads (ir_ops_cute.md): a store is issued once per distinct (coordinates,
    shared-memory source) among the executing lanes of a warp (ptxas's waterfall loop); a
    reduce, which the DSL lowers to a full-mask elect.sync and the reduce, once per warp
    as under an explicit elect_one, each issuing warp fully active (as a TMA load)."""
    # Operands: descriptor, smem source, coordinates (descriptor dim order), cache policy.
    _check_attr(tr, op, "mode", "tiled")
    is_reduce = op.name.endswith("tma_reduce")
    if is_reduce:
        _check_attr(tr, op, "kind", "<ADD>")
    desc = tr.val(op, 0)
    if not isinstance(desc, Opaque) or desc.kind != "tma_desc":
        tr.fail(op, "TMA store through a descriptor that is not a host TMA descriptor")
    d = tr.k.descs[desc.get("desc")]
    t = tr.k.tensors[d["base"]]
    if d["elem_bytes"] != t["elem_bytes"]:
        tr.fail(
            op,
            f"TMA store in {d['elem_bytes']}-byte units into a tensor of {t['elem_bytes']}-byte elements",
        )
    coords = [
        E.resize(c, 64) for c in _tma_coords(tr, op, _segment(op, 2), desc.get("desc"))
    ]
    kind = "gatomic" if is_reduce else "gstore"
    t["uses"].add(kind)
    r = len(coords)
    operands = coords + [E.resize(x, 64) for x in d["shape"] + d["strides"]]

    def build(v, j):
        return {"op": "event", "ev": kind, "tensor": d["base"], "loc": tr.loc(op),
                **_box_footprint(j, d["box"], v[:r], v[r : 2 * r], v[2 * r :])}  # fmt: skip

    if is_reduce:
        _uni, var = tr.split_mask()
        act = tr.active(var)
        if act is None:
            tr.fail(
                op,
                "TMA reduce (lowered to elect.sync) under a lane condition that depends on runtime values",
            )
        tr.full_warps(op, act, "TMA reduce (lowered to elect.sync)")
        saved = tr.mask
        tr.mask = E.and1(saved, E.ELECT)
        tr.thread_event(op, operands, build)
        tr.mask = saved
        return
    src = tr.val(op, 1)
    if not isinstance(src, Ptr) or src.space != "smem":
        tr.fail(op, "TMA store source is not a shared-memory pointer")
    raw = [tr.int_of(op, i, "TMA coordinate") for i in _segment(op, 2)]
    tr.waterfall_event(op, raw + [src.off], operands, build)


def _box_footprint(j, box, coords, shape, strides) -> dict:
    """offsets/mask TExprs of the elements c + i (i in the box) inside [0, shape)."""
    full = [b for b in box if b > 1] or [1]
    axes = {d: k for k, d in enumerate(d for d, b in enumerate(box) if b > 1)}

    def lift(x):
        return {"t": "splat", "shape": full, "w": 64, "a": j(x)}

    def node(t, w, **kw):
        return {"t": t, "shape": full, "w": w, **kw}

    offsets, mask = None, None
    for d, (c, n, st) in enumerate(zip(coords, shape, strides)):
        pos = lift(c)
        if d in axes:
            r = {"t": "range", "shape": [box[d]], "w": 64, "start": 0, "end": box[d]}
            cur = [box[d]]
            for ax in range(len(full)):
                if ax != axes[d]:
                    cur = cur[:ax] + [1] + cur[ax:]
                    r = {"t": "expand", "shape": list(cur), "w": 64, "a": r, "axis": ax}
            for ax in range(len(full)):
                if ax != axes[d]:
                    cur[ax] = full[ax]
                    r = {"t": "bcast", "shape": list(cur), "w": 64, "a": r, "axis": ax}
            pos = node("bin", 64, o="add", a=pos, b=r)
        term = node("bin", 64, o="mul", a=pos, b=lift(st))
        offsets = (
            term if offsets is None else node("bin", 64, o="add", a=offsets, b=term)
        )
        inside = node("bin", 1, o="and", a=node("cmp", 1, o="sge", a=pos, b=node("full", 64, v=0)),
                      b=node("cmp", 1, o="slt", a=pos, b=lift(n)))  # fmt: skip
        mask = inside if mask is None else node("bin", 1, o="and", a=mask, b=inside)
    return {"offsets": offsets, "mask": mask}


@effect("cute_nvgpu.arch.copy.SM100.tma_store", "cute_nvgpu.arch.copy.SM100.tma_reduce")
def _tma_store_kept(tr, op):
    # A store's shared-memory source decides how often a warp issues it (_tma_store).
    return ([0] if op.name.endswith("tma_reduce") else [0, 1]) + _segment(op, 2)


@op("cute_nvgpu.arch.copy.SM90.bulk_copy_g2s")
def _bulk_g2s(tr, op):
    """cp.async.bulk global -> shared of `size` bytes completing on an mbarrier. KIR has
    no bulk-copy event: a gload carries the footprint (P0-1) and a tma_load on a 1-D
    descriptor over the source tensor's storage (box = size) the transaction bytes the
    barrier receives (P0-4)."""
    if _segment(op, 3):
        tr.fail(op, "multicast bulk copy")
    p = tr.val(op, 0)
    if not isinstance(p, Ptr) or p.space != "gmem":
        tr.fail(op, "bulk copy source is not a kernel tensor pointer")
    nbytes = int(re.match(r"\s*(\d+)", op.props["size"]).group(1))
    t = tr.k.tensors[p.base]
    eb = t["elem_bytes"]
    if p.bits != eb * 8 or nbytes % eb:
        tr.fail(op, "bulk copy of a recast element type or of partial elements")
    n = nbytes // eb
    gaccess(tr, op, "gload", p, [(n, E.const(1, 64))])
    name = f"bulk_{p.base}_{n}"
    tr.k.descs.setdefault(
        name, {"bulk": p.base, "box": [n], "scale": 1, "pack": 1, "uses": set()}
    )["uses"].add("tma_load")
    tr.barrier_event(op, "tma_load", tr.val(op, 2), "thread", desc=name, coords=[p.off])


@effect("cute_nvgpu.arch.copy.SM90.bulk_copy_g2s")
def _bulk_kept(tr, op):
    return [0, 2]


# ------------------------------------------------------------------------ mbarriers


@op("nvvm.mbarrier.init.shared")
def _mbar_init(tr, op):
    # Several threads initializing one barrier with the same count (the DSL helpers init
    # with every lane of the warp) leave it in the state of one initialization.
    tr.barrier_event(
        op,
        "bar_init",
        tr.val(op, 0),
        "once",
        count=tr.int_of(op, 1, "mbarrier init count"),
    )


@op("nvvm.mapa.shared.cluster")
def _mapa(tr, op):
    """mapa: the shared-memory address `ptr` in the CTA of rank `cta` of the cluster."""
    p = tr.val(op, 0)
    if isinstance(p, Data):
        return p
    if isinstance(p, Remote):
        p = p.ptr
    if not isinstance(p, Ptr) or p.space != "smem":
        tr.fail(op, "mapa of a pointer that is not a shared-memory pointer")
    return Remote(p, E.resize(tr.int_of(op, 1, "mapa CTA rank"), 32, False))


@op("nvvm.mbarrier.txn")
def _mbar_txn(tr, op):
    # space cluster: the barrier address may name another CTA of the cluster (a mapa
    # result, interp.Remote); a shared::cta address names the issuing CTA.
    for name in ("scope", "space"):
        got = op.props.get(name, op.attrs.get(name, ""))
        if not re.search(r"<(cta|cluster)>", got):
            tr.fail(op, f"{name} = {got} (only cta and cluster are supported)")
    kind = re.search(r"mbar_txn_kind<(\w+)>", op.props.get("kind", "")).group(1)
    p = tr.val(op, 0)
    n = tr.int_of(op, 1, "mbarrier transaction count")
    if kind in ("arrive_expect_tx", "expect_tx"):
        tr.barrier_event(op, "expect_tx", p, "thread", bytes=n)
    if kind in ("arrive_expect_tx", "arrive"):
        if kind == "arrive" and not E.is_const(n, 1):
            tr.fail(op, "mbarrier arrive with a count other than 1")
        tr.barrier_event(op, "arrive", p, "count")
    if kind not in ("arrive_expect_tx", "expect_tx", "arrive"):
        tr.fail(op, f"mbarrier transaction kind {kind}")


@op("nvvm.mbarrier.arrive.shared", "nvvm.mbarrier.arrive")
def _mbar_arrive(tr, op):
    tr.barrier_event(op, "arrive", tr.val(op, 0), "count")
    return [tr.data(op)] * len(op.results)


@op("nvvm.mbarrier.try_wait.parity.shared")
def _mbar_wait(tr, op):
    if op.results:
        tr.fail(
            op, "try_wait whose result is used (only the blocking form is supported)"
        )
    tr.barrier_event(
        op, "wait", tr.val(op, 0), "once", parity=tr.int_of(op, 1, "wait parity")
    )


@op("nvvm.tcgen05.commit.arrive")
def _commit(tr, op):
    """tcgen05.commit (cta_group::1 or ::2): the completion of the issuing thread's prior
    async tcgen05 ops arrives once on the barrier, or with a multicast mask (operand 1,
    bit r: rank r) on the barrier at that offset in every CTA it names."""
    group = op.props.get("group", op.attrs.get("group", ""))
    if not re.search(r"<cta_[12]>", group):
        tr.fail(op, f"group = {group} (only cta_1 and cta_2 are supported)")
    extra = {}
    if len(op.operands) > 1:
        mask = tr.uniform(tr.int_of(op, 1, "tcgen05.commit multicast mask"))
        if mask is None:
            tr.fail(
                op,
                "tcgen05.commit multicast mask varies across the threads of the role",
            )
        extra["cta_mask"] = E.resize(mask, 32, False)
    tr.barrier_event(op, "commit_arrive", tr.val(op, 0), "thread", **extra)


@effect("nvvm.mbarrier.init.shared", "nvvm.mbarrier.txn", "nvvm.mbarrier.arrive.shared",
        "nvvm.mbarrier.arrive", "nvvm.mbarrier.try_wait.parity.shared", "nvvm.tcgen05.commit.arrive")  # fmt: skip
def _mbar_kept(tr, op):
    return list(range(min(len(op.operands), 2)))


@op("nvvm.cluster.arrive", "nvvm.cluster.arrive.relaxed", "nvvm.cluster.wait")
def _cluster_barrier(tr, op):
    """barrier.cluster.arrive (relaxed or not) / barrier.cluster.wait of every executing
    thread: the cluster barrier (interp.Translator.cluster_barrier_event)."""
    if op.operands:
        tr.fail(op, f"{op.name} with operands")
    tr.cluster_barrier_event(op, arrive=not op.name.endswith(".wait"))


@effect("nvvm.cluster.arrive", "nvvm.cluster.arrive.relaxed", "nvvm.cluster.wait")
def _cluster_barrier_kept(tr, op):
    return []


# ------------------------------------------------------------------- inline PTX

# Statements with no global-memory or mbarrier effect. `trap` (a deliberate abort) is not
# modeled: the run is evaluated as if it were not reached (references/frontend_cute.md).
_ASM_DATA = re.compile(
    r"(griddepcontrol\.(wait|launch_dependents)|cvt\.|ex2\.|rcp\.|lg2\.|max\.|min\.|fma\.|"
    r"mul\.|add\.|sub\.|prmt\.|mov\.|and\.|or\.|xor\.|not\.|shl\.|shr\.|bfe\.|bfi\.|abs\.|neg\.|"
    r"selp\.|setp\.|lop3\.|tanh\.|copysign\.|ld\.shared|st\.shared|"
    r"tcgen05\.(ld|st|cp|mma|alloc|dealloc|shift|fence|wait|relinquish_alloc_permit)|"
    r"elect\.sync|fence\.proxy|bar\.(sync|arrive)|barrier\.(sync|arrive)|redux\.sync|shfl\.sync|"
    r"trap$)"
)
_BULK_REDUCE = re.compile(
    r"cp\.reduce\.async\.bulk\.global\.shared::cta\.bulk_group\.add(?:\.noftz)?\.(\w+) \[\$(\d+)\], \[\$\d+\], \$(\d+)$"
)
# st.global of one value or a vector of 2 or 4 (u/s/b/f elements of 8 to 64 bits)
_ST_GLOBAL = re.compile(
    r"st\.global\.(?:v(2|4)\.)?[usbf](8|16|32|64) \[\$(\d+)\], (?:[\w$]+|\{[\w$, ]+\}?)$"
)
# A blocking wait: `L: mbarrier.try_wait.parity.shared::cta.b64 P, [$a], $p[, hint];` then
# `@!P bra L` or `@P bra D; bra L; D:` (the try_wait's predicate only loops back).
_TRY_WAIT = re.compile(
    r"mbarrier\.try_wait\.parity\.shared::cta\.b64 (\w+), \[\$(\d+)\], \$(\d+)(?:, \w+)?$"
)
_BRA = re.compile(r"bra(?:\.uni)?\s+(\w+)$")
_MAPA = re.compile(r"mapa\.shared::cluster\.u32 (\w+), \$(\d+), \$(\d+)$")
_BULK_S2CLUSTER = re.compile(
    r"cp\.async\.bulk\.shared::cluster\.shared::cta\.mbarrier::complete_tx::bytes "
    r"\[(\w+)\], \[\$(\d+)\], \$(\d+), \[(\w+)\]$"
)


def _asm_statements(op) -> list:
    """(guard, statement) per PTX statement of an inline asm; scope braces, `.reg`
    declarations, labels and comments dropped; guard is a leading `@p` / `@!p` or ""."""
    text = (
        op.props.get("asm_string", "")
        .strip('"')
        .replace("\\0A", "\n")
        .replace("\\09", " ")
    )
    out = []
    for s in re.split(r"[;\n]", text):
        s = s.strip(" \t{}")
        if s and not s.startswith((".reg", "//")) and not re.fullmatch(r"\w+:", s):
            m = re.match(r"(@!?\w+)\s+(.*)$", s)
            out.append((m.group(1), m.group(2)) if m else ("", s))
    return out


def _is_branch(s: str) -> bool:
    return bool(re.match(r"bra(\.uni)?\s+\w+$", s))


def _asm_lines(op) -> list:
    """(guard, statement) per PTX statement with labels kept as ("", "L:")."""
    text = (
        op.props.get("asm_string", "")
        .strip('"')
        .replace("\\0A", "\n")
        .replace("\\09", " ")
    )
    out = []
    for s in re.split(r"[;\n]", text):
        s = s.strip(" \t{}")
        if s and not s.startswith((".reg", "//")):
            m = re.match(r"(@!?\w+)\s+(.*)$", s)
            out.append((m.group(1), m.group(2)) if m else ("", s))
    return out


def _bra_target(s: str) -> str | None:
    """The label a `bra` / `bra.uni` statement jumps to, else None."""
    m = _BRA.match(s)
    return m.group(1) if m else None


def _wait_loops(lines: list):
    """[(first index, end index, address placeholder, parity placeholder)] of the blocking
    wait loops in `lines` (_TRY_WAIT), or ValueError for a try_wait that is not one."""
    out = []
    for i, (g, s) in enumerate(lines):
        m = _TRY_WAIT.match(s)
        if not m:
            continue
        pred, addr, parity = m.groups()
        label = (
            lines[i - 1][1][:-1]
            if i and re.fullmatch(r"\w+:", lines[i - 1][1])
            else None
        )
        if g or label is None:
            raise ValueError(
                f"inline PTX try_wait that is not a blocking wait loop: {s[:80]!r}"
            )
        nxt = lines[i + 1 : i + 4]
        to = [_bra_target(x) for _g, x in nxt]
        if nxt and nxt[0][0] == f"@!{pred}" and to[0] == label:  # @!P bra L
            out.append((i - 1, i + 2, int(addr), int(parity)))
        elif (
            len(nxt) == 3 and nxt[0][0] == f"@{pred}" and to[0] is not None
            and not nxt[1][0] and to[1] == label and nxt[2][1] == f"{to[0]}:"
        ):  # @P bra D; bra L; D:  # fmt: skip
            out.append((i - 1, i + 4, int(addr), int(parity)))
        else:
            raise ValueError(
                f"inline PTX try_wait that is not a blocking wait loop: {s[:80]!r}"
            )
    return out


def _classify_wait(op, lines: list):
    """("wait", address operand, parity operand) for data statements plus one blocking wait
    loop, None without a try_wait, else ValueError."""
    waits = _wait_loops(lines)
    if not waits:
        return None
    if len(waits) > 1:
        raise ValueError("inline PTX with several wait loops")
    lo, hi, addr, parity = waits[0]
    for _g, s in lines[:lo] + lines[hi:]:
        if not (_ASM_DATA.match(s) or re.fullmatch(r"\w+:", s)):
            raise ValueError(
                f"inline PTX wait loop next to a statement with a possible effect: {s[:80]!r}"
            )
    return ("wait", _asm_operand(op, addr), _asm_operand(op, parity))


def _bulk_s2cluster(lines: list):
    """("complete_tx", barrier operand, bytes operand, CTA rank operand) for exactly
    `mapa rd, $dst, $cta; mapa rb, $bar, $cta; cp.async.bulk.shared::cluster.shared::cta.
    mbarrier::complete_tx::bytes [rd], [$src], $n, [rb]` (unguarded, in any mapa order),
    else None."""
    if len(lines) != 3 or any(g for g, _s in lines):
        return None
    m0, m1 = (_MAPA.match(s) for _g, s in lines[:2])
    cp = _BULK_S2CLUSTER.match(lines[2][1])
    if m0 is None or m1 is None or cp is None:
        return None
    regs = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in (m0, m1)}
    dst, bar = regs.get(cp.group(1)), regs.get(cp.group(4))
    if dst is None or bar is None or dst[1] != bar[1]:
        return None
    return bar[0], int(cp.group(3)), bar[1]


def _asm_operand(op, placeholder: int, address: bool = False) -> int:
    """Operand index of input placeholder `$k` (outputs come first in the constraints);
    an address must be a 64-bit register operand ("l")."""
    cons = [
        c.strip()
        for c in op.props.get("constraints", "").strip('"').split(",")
        if c.strip()
    ]
    nout = sum(c.startswith("=") for c in cons)
    k = placeholder - nout
    if (
        placeholder >= len(cons)
        or not 0 <= k < len(op.operands)
        or (address and cons[placeholder] != "l")
    ):
        raise ValueError(
            f"inline PTX operand ${placeholder} is not an input{' address (l)' if address else ''}"
        )
    return k


def classify_asm(op):
    """The table meaning of an llvm.inline_asm: ("data",) when no statement touches global
    memory or an mbarrier; ("bulk_reduce", type, address operand, size operand) for one
    cp.reduce.async.bulk; ("st_global", bits, address operand) for data statements plus
    one unguarded st.global [$k] (8 to 64 bits, or a vector of 2 or 4) without branches;
    ("wait", address operand, parity operand) for data statements plus one blocking wait
    loop (_TRY_WAIT); ("complete_tx", barrier operand, bytes operand, CTA rank operand) for a
    shared::cluster bulk copy into another CTA (_bulk_s2cluster). ValueError otherwise."""
    lines = _asm_lines(op)
    s2c = _bulk_s2cluster(lines)
    if s2c is not None:
        return ("complete_tx", *(_asm_operand(op, k) for k in s2c))
    wait = _classify_wait(op, lines)
    if wait is not None:
        return wait
    stmts = _asm_statements(op)
    if len(stmts) == 1 and not stmts[0][0] and _BULK_REDUCE.match(stmts[0][1]):
        m = _BULK_REDUCE.match(stmts[0][1])
        return (
            "bulk_reduce",
            m.group(1),
            _asm_operand(op, int(m.group(2)), True),
            _asm_operand(op, int(m.group(3))),
        )
    stores = []
    for guard, s in stmts:
        if _ST_GLOBAL.match(s):
            if guard:
                raise ValueError(
                    f"inline PTX global store under a PTX predicate: {guard} {s[:60]!r}"
                )
            stores.append(_ST_GLOBAL.match(s))
        elif not (_ASM_DATA.match(s) or _is_branch(s)):
            raise ValueError(
                f"inline PTX with a possible global-memory or mbarrier effect: {s[:80]!r}"
            )
    if not stores:
        return ("data",)
    if len(stores) > 1 or any(_is_branch(s) for _g, s in stmts):
        raise ValueError(
            "inline PTX with a global store next to another store or a branch"
        )
    return (
        "st_global",
        int(stores[0].group(1) or 1) * int(stores[0].group(2)),
        _asm_operand(op, int(stores[0].group(3)), True),
    )


CLASSIFIERS["llvm.inline_asm"] = classify_asm


@op("llvm.inline_asm")
def _inline_asm(tr, op):
    kind = classify_asm(op)
    if kind[0] == "bulk_reduce":
        _bulk_reduce(tr, op, *kind[1:])
    elif kind[0] == "st_global":
        _st_global(tr, op, *kind[1:])
    elif kind[0] == "wait":
        tr.barrier_event(op, "wait", _smem_address(tr, op, kind[1]), "once",
                         parity=tr.int_of(op, kind[2], "wait parity"))  # fmt: skip
    elif kind[0] == "complete_tx":
        bar = Remote(
            _smem_address(tr, op, kind[1]),
            E.resize(tr.int_of(op, kind[3], "mapa CTA rank"), 32, False),
        )
        tr.barrier_event(
            op,
            "complete_tx",
            bar,
            "thread",
            bytes=tr.int_of(op, kind[2], "bulk copy bytes"),
        )
    return [tr.data(op)] * len(op.results)


def _smem_address(tr: Translator, op, i: int) -> Ptr:
    """The shared-memory pointer an inline PTX address operand holds: cute.ptrtoint of a
    shared-memory pointer (exact; anything else fails)."""
    a = tr.val(op, i)
    p = a.get("ptr") if isinstance(a, Opaque) and a.kind == "address" else None
    if not (isinstance(p, Ptr) and p.space == "smem"):
        tr.fail(
            op,
            f"inline PTX barrier address (operand {i}) is not a shared-memory pointer",
        )
    return p


def _bulk_reduce(tr, op, ty: str, pi: int, si: int):
    """cp.reduce.async.bulk shared -> global add of `size` bytes: a gatomic over them."""
    p = tr.val(op, pi)
    if not isinstance(p, Ptr) or p.space != "gmem":
        tr.fail(op, "bulk reduce destination is not a kernel tensor pointer")
    nbytes = tr.uniform(tr.int_of(op, si, "bulk reduce size"))
    bits = elem_bits(ty)
    if (
        nbytes is None
        or not E.is_const(nbytes)
        or nbytes[2] * 8 % bits
        or p.bits != bits
    ):
        tr.fail(op, "bulk reduce with a non-constant size or of another element type")
    gaccess(tr, op, "gatomic", p, [(nbytes[2] * 8 // bits, E.const(1, 64))])


def _st_global(tr: Translator, op, bits: int, ai: int):
    """st.global.u<bits> [$k]: a gstore of bits/8 bytes at the address operand k holds,
    which must be cute.ptrtoint of a kernel tensor pointer (exact; anything else fails)."""
    a = tr.val(op, ai)
    p = a.get("ptr") if isinstance(a, Opaque) and a.kind == "address" else None
    if not (isinstance(p, Ptr) and p.space == "gmem"):
        tr.fail(
            op,
            "inline PTX global store through an address that is not a kernel tensor pointer",
        )
    if bits % p.bits == 0:
        gaccess(tr, op, "gstore", p, [(bits // p.bits, E.const(1, 64))], p.bits)
    else:
        gaccess(tr, op, "gstore", p, [], bits)


@effect("llvm.inline_asm")
def _asm_kept(tr, op):
    kind = classify_asm(op)
    if kind[0] == "bulk_reduce":
        return [kind[2], kind[3]]
    if kind[0] == "st_global":
        return [kind[2]]
    if kind[0] in ("wait", "complete_tx"):
        return list(kind[1:])
    return None


# --------------------------------------------------------------------- host side


def _atom_field(atom_type: str, name: str) -> str:
    m = re.search(rf'{name} = (<"[^"]*">|\w+)', atom_type)
    return m.group(1).strip('<">') if m else ""


def _tma_dims(tr: Translator, op, atom_type: str, lay):
    """Descriptor dims of a TMA atom over a global layout, computed like CuTe's
    fill_tma_gmem_shape_stride: one dim per top-level mode of tma_gbasis; a mode over
    several global modes gets their gcd stride and the extent of the span they cover
    ((s - 1) * ceil(d / g) summed + 1). Extents and strides in global elements."""
    gshape, gstride = L.parse_layout(_atom_field(atom_type, "tma_gbasis"))
    modes = (
        list(zip(gshape, gstride)) if isinstance(gshape, list) else [(gshape, gstride)]
    )
    dims = []
    for msh, mst in modes:
        parts = []
        for b in L.leaves(mst):
            if not (isinstance(b, L.Basis) and E.is_const(b.scale, 1) and b.den == 1):
                tr.fail(
                    op,
                    f"TMA basis leaf {L.to_str(b)}: only unit basis strides are supported",
                )
            try:
                ext, st = L.at_path(lay.shape, b.path), L.at_path(lay.stride, b.path)
            except L.LayoutError as e:
                tr.fail(
                    op,
                    f"TMA basis mode {b.path} of global layout {L.to_str(lay.shape)}: {e}",
                )
            if not (isinstance(ext, tuple) and isinstance(st, tuple)):
                tr.fail(
                    op,
                    f"TMA basis leaf over a hierarchical global mode {L.to_str(ext)}",
                )
            parts.append((ext, st))
        box = 1
        for n in L.leaves(msh):
            box *= n[2]
        if len(parts) == 1:
            ext, st = parts[0]
            dims.append({"box": box, "shape": ext, "stride": st})
            continue
        if not all(E.is_const(st) for _e, st in parts):
            tr.fail(op, "TMA dim over several global modes with dynamic strides")
        shape, g = E.const(1, 64), 0
        for ext, st in parts:
            old, g = g, math.gcd(g, st[2])
            if g:
                shape = E.bin("add", E.bin("add", E.bin("mul", E.bin("sub", shape, E.const(1, 64)), E.const(-(-old // g), 64)),
                                           E.bin("mul", E.bin("sub", E.resize(ext, 64), E.const(1, 64)), E.const(-(-st[2] // g), 64))),
                              E.const(1, 64))  # fmt: skip
            else:
                shape = E.resize(ext, 64)
        dims.append({"box": box, "shape": shape, "stride": E.const(g, 64)})
    return dims


def _format_bits(tr: Translator, op, fmt: str) -> int:
    m = re.match(
        r"[A-Z]+?(\d+)", fmt
    )  # U4, U4_UNPACK_U8, U8, U16, BF16_RN, F16_RN, FP32, ...
    if not m:
        tr.fail(op, f"TMA data format {fmt!r}")
    return int(m.group(1))


@op("cute_nvgpu.atom.make_non_exec_tiled_tma_load", "cute_nvgpu.atom.make_non_exec_tiled_tma_store",
    "cute_nvgpu.atom.make_non_exec_tiled_tma_reduce")  # fmt: skip
def _make_tma(tr, op):
    """A host TMA descriptor over a global view: a KIR `desc` tensor in global elements
    (bytes for packed 4-bit data). The TMA moves units of its data format; dim 0 is the
    contiguous one, so a format r elements wide scales dim 0's box and coordinates by r.
    The descriptor's bytes per copy must equal the atom's copy_bits / 8. A multicast
    descriptor (num_multicast n > 1) covers the CTA tile, and each load moves a 1/n slice
    of it (CuTe make_tma_copy_desc truncates the box by n from the outermost dim; copy_bits
    is the untruncated tile): the KIR box is the slice."""
    g = tr.val(op, 0)
    if not isinstance(g, View) or not isinstance(g.ptr, Ptr) or g.ptr.space != "gmem":
        tr.fail(op, "TMA descriptor over something other than a kernel tensor")
    if not E.is_const(g.ptr.off, 0):
        tr.fail(op, "TMA descriptor over an offset view")
    atom_t = rtype(tr, op, 0)
    lay = lay_of(tr, op, g)
    dims = _tma_dims(tr, op, atom_t, lay)
    ebits, fbits = g.ptr.bits, _format_bits(tr, op, _atom_field(atom_t, "tma_format"))
    if not E.is_const(dims[0]["stride"], 1):
        tr.fail(
            op,
            f"TMA dim 0 has global stride {L.to_str(dims[0]['stride'])} (must be contiguous)",
        )
    shape = [d["shape"] for d in dims]
    strides = [d["stride"] for d in dims]
    box = [d["box"] for d in dims]
    mcast = re.match(r"\s*(\d+)", op.props.get("num_multicast", "1"))
    slice_box = _multicast_slice(tr, op, box, int(mcast.group(1)) if mcast else 1)
    scale, pack = 1, 1
    if ebits >= 8 and fbits % ebits == 0:
        scale, ebytes = fbits // ebits, ebits // 8
        box[0] *= scale
    elif ebits < 8 and fbits == ebits and 8 % ebits == 0:
        # Packed sub-byte data is described in bytes (U4, and U4_UNPACK_U8 whose shared
        # memory copy expands but whose transaction bytes are the packed source bytes).
        pack, ebytes = 8 // ebits, 1
        if box[0] % pack:
            tr.fail(
                op, f"TMA box {box[0]} over {ebits}-bit elements is not byte aligned"
            )
        shape[0] = E.bin("sdiv", shape[0], E.const(pack, E.width(shape[0])))
        strides = [strides[0]] + [
            E.bin("sdiv", x, E.const(pack, E.width(x))) for x in strides[1:]
        ]
        box[0] //= pack
    else:
        tr.fail(op, f"TMA format of {fbits} bits over {ebits}-bit elements")
    copy_bits = int(_atom_field(atom_t, "copy_bits") or 0)
    if math.prod(box) * ebytes * 8 != copy_bits:
        tr.fail(
            op,
            f"descriptor box {box} x {ebytes} bytes disagrees with the atom's copy_bits = {copy_bits}",
        )
    if slice_box != [d["box"] for d in dims]:  # in the same units as `box`
        box = [b * scale if i == 0 else b for i, b in enumerate(slice_box)]
        if pack != 1:
            box[0] = slice_box[0] // pack
    # The kernel-tensor dim each descriptor dim walks: the one with the same extent and
    # stride (the global view may permute or regroup the tensor's modes).
    t = tr.k.tensors[g.ptr.base]
    tdims = [next((i for i, (n, s) in enumerate(zip(t["shape"], t["strides"])) if n is d["shape"] and s is d["stride"]), None)
             for d in dims]  # fmt: skip
    name = f"tma{len(tr.k.descs)}_{g.ptr.base}"
    tr.k.descs[name] = {
        "base": g.ptr.base, "tensor_dims": tdims, "box": box, "shape": shape, "strides": strides,
        "elem_bytes": ebytes, "scale": scale, "pack": pack, "uses": set(),
    }  # fmt: skip
    ref = quoted(rtype(tr, op, 1))
    base = L.parse_tuple(ref[0])
    cshape, cstride = L.parse_layout(ref[1])
    if L.dyn_count(base) or L.dyn_count(cstride):
        tr.fail(op, "TMA coordinate tensor with a dynamic start or stride")
    gl = L.leaves(lay.shape)
    if len(gl) != len(L.leaves(cshape)):
        tr.fail(op, "TMA coordinate tensor and global view differ in structure")
    return [
        Opaque("tma", (("desc", name),)),
        View(ArithIter(tuple(L.leaves(base))), Lay(L.rebuild(cshape, gl), cstride)),
    ]


def _multicast_slice(tr, op, box: list, n: int) -> list:
    """The box one load of an n-CTA multicast descriptor moves (CuTe make_tma_copy_desc:
    from the outermost dim down, each dim's box shrinks by the remaining factor)."""
    box = list(box)
    i = len(box) - 1
    while n > 1:
        if i < 0 or (box[i] % n and n % box[i]):
            tr.fail(op, f"multicast TMA box {box} does not split into {n} slices")
        nxt = -(-n // box[i])
        box[i] = -(-box[i] // n)
        n, i = nxt, i - 1
    return box


@op("cuda.launch_cfg.create")
def _launch_cfg(tr, op):
    vals = [tr.int_of(op, i, "launch configuration") for i in range(7)]
    return Opaque(
        "launch_cfg",
        (("block", tuple(vals[0:3])), ("smem", vals[3]), ("grid", tuple(vals[4:7]))),
    )


@op("cuda.launch_cfg.cluster_dim")
def _cluster_dim(tr, op):
    # Updates the launch configuration in place (no result).
    cfg = tr.val(op, 0)
    tr.env[op.operands[0]] = cfg.with_(
        cluster=tuple(
            tr.int_of(op, i, "cluster dims") for i in range(1, len(op.operands))
        )
    )


@op("cuda.launch_cfg.programmatic_stream_serialization_allowed", "cuda.launch_cfg.cooperative",
    "cuda.cast", "cuda.return_if_error")  # fmt: skip
def _host_noop(tr, op):
    return [tr.data(op)] * len(op.results)


# The host function's slice: the launch and every operand it passes, and the cluster dims
# (cuda.launch_cfg.cluster_dim updates the launch configuration in place).
HOST_EFFECTS = {
    "cuda.launch_ex": lambda tr, op: list(range(len(op.operands))),
    "cuda.launch_cfg.cluster_dim": lambda tr, op: list(range(len(op.operands))),
}


@op("cuda.launch_ex")
def _launch(tr, op):
    cfg = tr.val(op, 0)
    callee = re.search(r"@kernels::@(\w+)", op.props.get("callee", "")).group(1)
    tr.launches.append(
        {
            "kernel": callee,
            "cfg": cfg,
            "args": [tr.val(op, i) for i in range(1, len(op.operands))],
        }
    )
    return [tr.data(op)] * len(op.results)
