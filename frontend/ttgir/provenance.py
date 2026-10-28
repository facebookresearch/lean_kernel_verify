# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Which pointer / descriptor parameters a kernel may write, read off its TTGIR.

dump skips every launch (there is no GPU), so the contents of a tensor written by a
launch the host made *before* the analyzed launch (of any kernel, the analyzed one
included) are not computed on CPU. If such data is metadata of the analyzed kernel (its
contents steer control flow or addresses), directly or through host computation, or the
host reads it back before the launch, using the CPU run would be a silent lie; to_kir
refuses it. This module provides the conservative write set of a launch used for those
checks: `None` means "any pointer argument".
"""

from . import op_table
from .mlir_text import Block, Op, parse, ParseError, strip_strings, Value, walk

FORWARD_OPS = {
    "tt.addptr",
    "tt.splat",
    "tt.broadcast",
    "tt.expand_dims",
    "ttg.convert_layout",
    "tt.make_tensor_descriptor",
}
WRITE_OPS = {
    "tt.store",
    "tt.atomic_rmw",
    "tt.atomic_cas",
    "tt.descriptor_store",
    "tt.descriptor_reduce",
    "tt.descriptor_scatter",
    "ttng.async_tma_copy_local_to_global",
    "ttng.async_tma_reduce",
    "ttng.async_tma_scatter",
    "ttng.tensormap_create",
}
WRITE_WORDS = ("store", "atomic", "scatter")


class _Unknown(Exception):
    pass


def written_params(ttgir: str) -> set | None:
    try:
        module = parse(ttgir)
    except ParseError:
        return None
    body = module.func.regions[0]
    parents = {
        id(b): (op, k)
        for op in walk(body)
        for k, r in enumerate(op.regions)
        for b in [r] + r.more
    }
    written = set()
    try:
        for op in walk(body):
            if op.name in WRITE_OPS:
                written |= _roots(_target(op), body, parents, set())
            elif op.name not in op_table.KNOWN_OPS and any(
                w in op.name for w in WRITE_WORDS
            ):
                return None
            elif (
                op.name == "tt.elementwise_inline_asm"
                and "pure = true" not in strip_strings(op.text)
            ):
                return None  # side-effecting inline asm may store anywhere
    except _Unknown:
        return None
    return written


def _target(op: Op):
    if op.name in op_table.TMA_OPS:
        return op_table.tma_op(op)["desc"]
    if op.name == "ttng.tensormap_create":
        return op_table.tensormap_op(op)["global"]
    return op.operands[0]


def _roots_from_block(
    v: Value, body: Block, parents: dict[int, tuple[Op, int]], seen: set[int]
) -> set[str]:
    owner = v.owner
    assert isinstance(owner, Block)
    if owner is body:
        return {v.name}
    if id(owner) not in parents:  # argument of a later block of the kernel body
        raise _Unknown()
    op, k = parents[id(owner)]
    if op.name == "ttg.warp_specialize" and k > 0:
        return _roots(op.operands[v.index], body, parents, seen)
    if op.name == "scf.for" and v.index > 0:
        return _roots(op.operands[2 + v.index], body, parents, seen) | _roots(
            owner.ops[-1].operands[v.index - 1], body, parents, seen
        )
    raise _Unknown()


def _roots(v, body: Block, parents: dict, seen: set) -> set:
    if id(v) in seen:
        return set()
    seen.add(id(v))
    owner = v.owner
    if isinstance(owner, Block):
        return _roots_from_block(v, body, parents, seen)
    if owner.name in FORWARD_OPS:
        return _roots(owner.operands[0], body, parents, seen)
    if owner.name == "arith.select":
        return _roots(owner.operands[1], body, parents, seen) | _roots(
            owner.operands[2], body, parents, seen
        )
    if owner.name == "ttng.reinterpret_tensor_descriptor":
        scratch = owner.operands[0]
        for op in walk(body):
            if (
                op.name == "ttng.tensormap_create"
                and op_table.tensormap_op(op)["desc_ptr"] is scratch
            ):
                return _roots(op_table.tensormap_op(op)["global"], body, parents, seen)
    if owner.name in ("scf.if", "scf.for"):
        out = set()
        for region in owner.regions:
            out |= _roots(region.ops[-1].operands[v.index], body, parents, seen)
        if owner.name == "scf.for":
            out |= _roots(owner.operands[3 + v.index], body, parents, seen)
        return out
    raise _Unknown()
