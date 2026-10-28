# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The TTGIR op table: which ops are events, values, structure or droppable, and how
each op's operands are laid out in its printed form.

The meaning of every op is documented in references/ir_ops.md (TTGIR section); this file
is its executable form and must stay in sync with it. Operand layouts follow the
`assemblyFormat` of the op definitions in Triton's TritonNvidiaGPUOps.td /
TritonOps.td / TritonGPUOps.td (the TLX Triton tree). Every op in a kernel must fall into exactly
one class; anything else is a hard error naming the op and its loc.
"""

import re

from .kir_build import INT_WIDTHS, ptr_pointee, tensordesc_type
from .mlir_text import find_top, Op, REF_RE, split_top, strip_strings, Value


class OpError(Exception):
    pass


ARITH_BIN = {
    "arith.addi": "add",
    "arith.subi": "sub",
    "arith.muli": "mul",
    "arith.divsi": "sdiv",
    "arith.divui": "udiv",
    "arith.remsi": "srem",
    "arith.remui": "urem",
    "arith.shli": "shl",
    "arith.shrsi": "ashr",
    "arith.shrui": "lshr",
    "arith.andi": "and",
    "arith.ori": "or",
    "arith.xori": "xor",
    "arith.maxsi": "smax",
    "arith.minsi": "smin",
    "arith.maxui": "umax",
    "arith.minui": "umin",
}
CMP_PREDS = {"eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"}

# Ops translated to KIR expressions / references when their results are on the slice.
VALUE_OPS = set(ARITH_BIN) | {
    "arith.constant",
    "arith.cmpi",
    "arith.select",
    "arith.extsi",
    "arith.extui",
    "arith.trunci",
    "arith.index_cast",
    "math.absi",
    "tt.get_program_id",
    "tt.get_num_programs",
    "tt.make_range",
    "tt.splat",
    "tt.broadcast",
    "tt.expand_dims",
    "ttg.convert_layout",
    "tt.addptr",
    "tt.load",
    "ttg.local_alloc",
    "ttg.memdesc_index",
    "ttng.map_to_remote_buffer",
    "ttng.clc_query_cancel",
    "nvg.cluster_id",
    "ttg.global_scratch_alloc",
    "ttng.reinterpret_tensor_descriptor",
}

TMA_OPS = {
    "ttng.async_tma_copy_global_to_local",
    "ttng.async_tma_copy_local_to_global",
    "ttng.async_tma_reduce",
}

# Ops that produce KIR events (some, like tt.load, are also value ops).
EVENT_OPS = {
    "ttng.init_barrier",
    "ttng.arrive_barrier",
    "ttng.barrier_expect",
    "ttng.wait_barrier",
    "ttng.tc_gen5_mma",
    "ttng.tc_gen5_mma_scaled",
    "ttng.tc_gen5_commit",
    "ttng.async_tma_copy_global_to_local",
    "ttng.async_tma_copy_local_to_global",
    "ttng.async_tma_reduce",
    "tt.load",
    "tt.store",
    "tt.atomic_rmw",
    "tt.atomic_cas",
    "ttng.async_clc_try_cancel",
    "ttng.tensormap_create",
    "tt.assert",
}

STRUCT_OPS = {
    "scf.for",
    "scf.if",
    "scf.while",
    "scf.yield",
    "scf.condition",
    "ttg.warp_specialize",
    "ttg.warp_yield",
    "ttg.warp_return",
    "tt.return",
    "cf.cond_br",
    "cf.br",
}

# Ops with no P0 meaning: data movement/math on tile contents, layout changes, fences,
# waits on async stores, register hints. Dropping them is checked by slicing.Slicer: their
# results must never reach the kept slice.
DROP_OPS = {
    "arith.addf",
    "arith.subf",
    "arith.mulf",
    "arith.divf",
    "arith.remf",
    "arith.negf",
    "arith.maximumf",
    "arith.minimumf",
    "arith.maxnumf",
    "arith.minnumf",
    "arith.truncf",
    "arith.extf",
    "arith.sitofp",
    "arith.uitofp",
    "arith.fptosi",
    "arith.fptoui",
    "arith.cmpf",
    "arith.bitcast",
    "math.exp",
    "math.exp2",
    "math.log",
    "math.log2",
    "math.sqrt",
    "math.rsqrt",
    "math.fma",
    "math.absf",
    "math.erf",
    "math.tanh",
    "math.floor",
    "math.ceil",
    "math.sin",
    "math.cos",
    "tt.fp_to_fp",
    "tt.reduce",
    "tt.reduce.return",
    "tt.scan",
    "tt.scan.return",
    "tt.elementwise_inline_asm",
    "tt.precise_divf",
    "tt.precise_sqrt",
    "tt.trans",
    "tt.reshape",
    "tt.split",
    "tt.join",
    "tt.cat",
    "tt.clampf",
    "tt.bitcast",
    "tt.print",
    "ttg.local_load",
    "ttg.local_store",
    "ttg.local_dealloc",
    "ttg.memdesc_trans",
    "ttg.memdesc_reinterpret",
    "ttg.memdesc_subslice",
    "ttg.barrier",
    "ttng.tmem_alloc",
    "ttng.tmem_load",
    "ttng.tmem_store",
    "ttng.tmem_subslice",
    "ttng.tmem_copy",
    "ttng.fence_async_shared",
    "ttng.fence",
    "ttng.async_tma_store_wait",
    "ttng.tensormap_fenceproxy_acquire",
}

KNOWN_OPS = VALUE_OPS | EVENT_OPS | STRUCT_OPS | DROP_OPS


def classify(op: Op) -> None:
    """Raise unless the op is in the table (unknown ops may hide events)."""
    if op.name not in KNOWN_OPS:
        raise OpError(f"unsupported op {op.where()}")
    if op.name == "tt.elementwise_inline_asm" and "pure = true" not in strip_strings(
        op.text
    ):
        raise OpError(f"side-effecting inline asm {op.where()}")


def has_effect(op: Op) -> bool:
    """Events, and `ttng.clc_query_cancel`, which advances the role's query counter even
    when its results are unused (references/ir_ops.md): kept wherever they are."""
    return op.name in EVENT_OPS or op.name == "ttng.clc_query_cancel"


def is_kernel_arg_type(t: str) -> bool:
    return (
        t in INT_WIDTHS or ptr_pointee(t) is not None or tensordesc_type(t) is not None
    )


# -- operand layouts ----------------------------------------------------------------------
def _refs(text: str) -> list[str]:
    return [r[1:] for r in REF_RE.findall(strip_strings(text))]


def _int_after_comma(op: Op) -> int:
    """`opname %x, 3 ...` -> 3 (init_barrier count, arrive count, expect size)."""
    m = re.search(r",\s*(-?\d+)\b", strip_strings(op.text))
    if not m:
        raise OpError(f"cannot read the integer attribute of {op.where()}")
    return int(m.group(1))


def _before_types(op: Op) -> str:
    clean = strip_strings(op.text)
    idx = find_top(clean, " : ")
    return clean[:idx] if idx >= 0 else clean


def attr_dict(op: Op) -> str:
    """The trailing `{...}` attribute dictionary of a custom-form op ("" if none)."""
    body = _before_types(op).rstrip()
    if not body.endswith("}"):
        return ""
    depth = 0
    for i in range(len(body) - 1, -1, -1):
        depth += {"}": 1, "{": -1}.get(body[i], 0)
        if depth == 0:
            return body[i:]
    return ""


def barrier_op(op: Op) -> dict:
    """init_barrier / arrive_barrier / barrier_expect / wait_barrier / tc_gen5_commit."""
    ops = op.operands
    if op.name == "ttng.init_barrier":
        return {"bar": ops[0], "count": _int_after_comma(op)}
    if op.name == "ttng.arrive_barrier":
        return {
            "bar": ops[0],
            "count": _int_after_comma(op),
            "pred": ops[1] if len(ops) > 1 else None,
            "attrs": attr_dict(op),
        }
    if op.name == "ttng.barrier_expect":
        return {"bar": ops[0], "bytes": _int_after_comma(op), "pred": ops[1]}
    if op.name == "ttng.wait_barrier":
        head = _before_types(op).split(" deps ")[0]
        n = len(_refs(head))
        return {"bar": ops[0], "phase": ops[1], "pred": ops[2] if n > 2 else None}
    if op.name == "ttng.tc_gen5_commit":
        head = _before_types(op).split(" descs ")[0]
        n = len(_refs(head))
        if " descs " in _before_types(op):
            raise OpError(
                f"tc_gen5_commit with descs is not in the op table: {op.where()}"
            )
        return {"bar": ops[0], "pred": ops[1] if n > 1 else None}
    raise OpError(f"not a barrier op: {op.where()}")


def tma_op(op: Op) -> dict:
    """TMA copies / reduce: desc, coords (and for loads: barrier, pred)."""
    body = _before_types(op)
    if "[" not in body:
        raise OpError(f"cannot parse TMA coordinates of {op.where()}")
    lb, rb = body.index("["), body.index("]")
    desc_name = _refs(body[:lb])[-1]
    coord_names = _refs(body[lb:rb])
    rest = body[rb + 1 :]
    if "offsets" in rest:
        raise OpError(f"im2col TMA offsets are not in the op table: {op.where()}")
    by_name = {v.name: v for v in op.operands}
    out = {"desc": by_name[desc_name], "coords": [by_name[c] for c in coord_names]}
    if op.name == "ttng.async_tma_copy_global_to_local":
        names = _refs(rest.split("cacheModifier")[0].split("evictionPolicy")[0])
        if len(names) != 3:
            raise OpError(f"multicast TMA load is not in the op table: {op.where()}")
        out["bar"], out["pred"] = by_name[names[1]], by_name[names[2]]
        if "two_cta" in attr_dict(op) and "two_cta = false" not in attr_dict(op):
            raise OpError(f"two_cta TMA load is not in the op table: {op.where()}")
    return out


BAR_PRED_RE = re.compile(r"^%([\w$.\-]+)\[%([\w$.\-]+)\]$")


def mma_barriers(op: Op) -> tuple[Value, list]:
    """tc_gen5_mma[_scaled]: the issue predicate and [(barrier, barrier_pred)].

    Printed as `a, b, d[token]?, ..., pred bar0[p0], bar1[p1], ...`: the completion
    barriers are the trailing `%bar[%pred]` operands, the issue predicate precedes them.
    """
    body = _before_types(op)
    attrs = attr_dict(op)
    if attrs:
        body = body[: body.rindex(attrs)]
    parts = split_top(body.split(" ", 1)[1])
    by_name = {v.name: v for v in op.operands}
    k = next(
        (i for i, p in enumerate(parts) if i >= 3 and BAR_PRED_RE.match(p)), len(parts)
    )
    pred = by_name[_refs(parts[k - 1])[0]]
    bars = []
    for p in parts[k:]:
        m = BAR_PRED_RE.match(p)
        if not m:
            raise OpError(f"cannot parse completion barriers of {op.where()}")
        bars.append((by_name[m.group(1)], by_name[m.group(2)]))
    return pred, bars


def mem_op(op: Op) -> dict:
    """tt.load / tt.store / tt.atomic_rmw / tt.atomic_cas pointer, mask, other."""
    o = op.operands
    if op.name == "tt.load":
        return {
            "ptr": o[0],
            "mask": o[1] if len(o) > 1 else None,
            "other": o[2] if len(o) > 2 else None,
        }
    if op.name == "tt.store":
        return {"ptr": o[0], "mask": o[2] if len(o) > 2 else None}
    if op.name == "tt.atomic_rmw":
        return {"ptr": o[0], "mask": o[2] if len(o) > 2 else None}
    if op.name == "tt.atomic_cas":
        return {"ptr": o[0], "mask": None}
    raise OpError(f"not a memory op: {op.where()}")


def tensormap_op(op: Op) -> dict:
    """ttng.tensormap_create: desc_ptr, global, [box], [gdim], [gstride], [estride]."""
    body = _before_types(op)
    groups = re.findall(r"\[([^\]]*)\]", body)
    if len(groups) != 4:
        raise OpError(f"malformed tensormap_create {op.where()}")
    by_name = {v.name: v for v in op.operands}
    head = _refs(body[: body.index("[")])
    return {
        "desc_ptr": by_name[head[0]],
        "global": by_name[head[1]],
        "box": [by_name[n] for n in _refs(groups[0])],
        "gdim": [by_name[n] for n in _refs(groups[1])],
        "gstride": [by_name[n] for n in _refs(groups[2])],
        "estride": [by_name[n] for n in _refs(groups[3])],
        "attrs": attr_dict(op),
    }


def root_operands(op: Op):
    """(value, feeds_address, role) for the operands an event reads."""
    n = op.name
    if n in (
        "ttng.init_barrier",
        "ttng.arrive_barrier",
        "ttng.barrier_expect",
        "ttng.wait_barrier",
        "ttng.tc_gen5_commit",
    ):
        b = barrier_op(op)
        yield b["bar"], False, "barrier"
        for key in ("phase", "pred"):
            if b.get(key) is not None:
                yield b[key], False, key
    elif n in ("ttng.tc_gen5_mma", "ttng.tc_gen5_mma_scaled"):
        pred, bars = mma_barriers(op)
        yield pred, False, "issue predicate"
        for bar, p in bars:
            yield bar, False, "completion barrier"
            yield p, False, "completion barrier predicate"
    elif n in TMA_OPS:
        t = tma_op(op)
        yield t["desc"], False, "descriptor"
        for c in t["coords"]:
            yield c, True, "TMA coordinate"
        if "bar" in t:
            yield t["bar"], False, "barrier"
            yield t["pred"], False, "pred"
    elif n in ("tt.load", "tt.store", "tt.atomic_rmw", "tt.atomic_cas"):
        m = mem_op(op)
        yield m["ptr"], True, "pointer"
        if m["mask"] is not None:
            yield m["mask"], False, "mask"
    elif n == "ttng.async_clc_try_cancel":
        yield op.operands[0], False, "barrier"
    elif n == "ttng.tensormap_create":
        t = tensormap_op(op)
        yield t["global"], True, "global address"
        for v in t["gdim"] + t["gstride"] + t["box"] + t["estride"]:
            yield v, False, "descriptor extent"
    elif n == "tt.assert":
        yield op.operands[0], False, "condition"
    elif n == "cf.cond_br":
        yield op.operands[0], False, "branch condition"


def value_deps(op: Op, v: Value) -> list:
    """[(operand, carries the value)] a value result depends on (backward closure).

    A comparison's operands, a select condition and a load mask decide which value is
    taken but are not part of it (slicing.Slicer tags address paths through the others).
    """
    n = op.name
    if n in (
        "arith.constant",
        "tt.make_range",
        "tt.get_program_id",
        "tt.get_num_programs",
        "nvg.cluster_id",
        "ttg.global_scratch_alloc",
    ):
        return []
    if n == "ttg.local_alloc":
        if op.operands:
            raise OpError(
                f"initialized local_alloc used as a barrier/reference: {op.where()}"
            )
        return []
    if n == "tt.load":
        m = mem_op(op)
        return [
            (x, carries)
            for x, carries in ((m["ptr"], True), (m["mask"], False), (m["other"], True))
            if x is not None
        ]
    if n == "arith.cmpi":
        return [(x, False) for x in op.operands]
    if n == "arith.select":
        return [(op.operands[0], False)] + [(x, True) for x in op.operands[1:]]
    return [(x, True) for x in op.operands]
