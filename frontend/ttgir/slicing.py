# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Program slicing over parsed TTGIR (design.md, "Program slicing rule").

The kept slice is the backward closure, over SSA def-use and structured control flow, of
the operands that events read (barrier refs, phases, predicates, pointers, masks, TMA
coordinates, descriptor extents) plus the control operands of every structured op that
contains a kept op. Every value on the slice must be defined by an op the op table
translates (op_table.VALUE_OPS) or by a kernel/block argument. A value defined by any
other op (floating-point math, TMEM/SMEM data loads, MMA results, inline asm, unknown
ops) aborts extraction with both ops and their locs: that is the soundness check that
nothing dropped influences what is kept.
"""

from dataclasses import dataclass, field

from . import op_table
from .mlir_text import Block, Op, Value, walk


class SliceError(Exception):
    pass


@dataclass
class Slice:
    needed: dict = field(default_factory=dict)  # Value -> set of tags {"ctl", "addr"}
    live: set = field(default_factory=set)  # structured ops that must be emitted
    why: dict = field(default_factory=dict)  # Value -> consumer description


def terminator(block: Block) -> Op:
    return block.ops[-1]


class Slicer:
    def __init__(self, body: Block):
        self.body = body
        self.s = Slice()
        self.work: list = []
        self.parents = {
            id(b): (op, k)
            for op in walk(body)
            for k, r in enumerate(op.regions)
            for b in [r] + r.more
        }

    def run_with(self, extra_roots: list) -> Slice:
        """Slice from the events' operands plus `extra_roots` ([(value, why)])."""
        for op in walk(self.body):
            for v, addr, role in op_table.root_operands(op):
                self.need(v, f"{role} of {op.where()}", addr)
        for v, why in extra_roots:
            self.need(v, why, False)
        self._drain()
        changed = True
        while changed:
            changed = False
            for op in reversed(list(walk(self.body))):
                if (
                    op.name in ("scf.for", "scf.if", "scf.while")
                    and op not in self.s.live
                ):
                    if self._has_kept(op):
                        self.s.live.add(op)
                        for v in self._control_operands(op):
                            self.need(v, f"control operand of {op.where()}", False)
                        changed = True
            self._drain()
        return self.s

    # -- worklist ------------------------------------------------------------------------
    def need(self, v: Value, why: str, addr: bool) -> None:
        tags = self.s.needed.setdefault(v, set())
        new = {"ctl"} | ({"addr"} if addr else set())
        if new <= tags:
            return
        tags |= new
        self.s.why.setdefault(v, why)
        self.work.append((v, addr))

    def _drain(self) -> None:
        """Close the slice; `addr` follows only the operands that carry the value (not
        comparisons, select/if conditions or masks), so it marks exactly address paths."""
        while self.work:
            v, addr = self.work.pop()
            for dep, carries in self._deps(v):
                self.need(dep, self.s.why[v], addr and carries)

    def _deps(self, v: Value) -> list:
        """[(operand, carries the value)] that `v` depends on."""
        owner = v.owner
        if isinstance(owner, Block):
            return [(d, True) for d in self._block_arg_deps(v, owner)]
        op: Op = owner
        if op.name == "scf.for":
            return [(op.regions[0].args[v.index + 1], True)]
        if op.name == "scf.if":
            return [(op.operands[0], False)] + [
                (terminator(r).operands[v.index], True) for r in op.regions
            ]
        if op.name == "scf.while":
            return [(terminator(op.regions[0]).operands[v.index + 1], True)]
        if op.name not in op_table.VALUE_OPS:
            kind = "dropped" if op.name in op_table.DROP_OPS else "unsupported"
            raise SliceError(
                f"slice error: %{v.name} is defined by {kind} op {op.where()} but reaches "
                f"the kept slice ({self.s.why[v]})"
            )
        return op_table.value_deps(op, v)

    def _block_arg_deps(self, v: Value, block: Block) -> list:
        if block.label.startswith("^"):
            raise SliceError(
                f"slice error: argument %{v.name} of block {block.label} (unstructured "
                f"control flow with block arguments) reaches the kept slice ({self.s.why[v]})"
            )
        parent = self._block_parent(block)
        if parent is None:  # kernel argument
            if not op_table.is_kernel_arg_type(v.type):
                raise SliceError(
                    f"slice error: kernel argument %{v.name} of type {v.type} reaches "
                    f"the kept slice ({self.s.why[v]})"
                )
            return []
        op, ridx = parent
        if op.name == "scf.for":
            if v.index == 0:
                return []
            k = v.index - 1
            return [op.operands[3 + k], terminator(block).operands[k]]
        if op.name == "scf.while" and ridx == 0:
            return [op.operands[v.index], terminator(op.regions[1]).operands[v.index]]
        if op.name == "scf.while" and ridx == 1:
            return [terminator(op.regions[0]).operands[v.index + 1]]
        if op.name == "ttg.warp_specialize" and ridx > 0:
            return [op.operands[v.index]]
        raise SliceError(
            f"slice error: block argument %{v.name} of {op.where()} reaches the kept "
            f"slice ({self.s.why[v]})"
        )

    def _block_parent(self, block: Block):
        return None if block is self.body else self.parents[id(block)]

    # -- liveness ------------------------------------------------------------------------
    def _has_kept(self, op: Op) -> bool:
        if any(r in self.s.needed for r in op.results):
            return True
        for region in op.regions:
            for inner in walk(region):
                if inner in self.s.live or op_table.has_effect(inner):
                    return True
                if any(r in self.s.needed for r in inner.results):
                    return True
                if isinstance(inner, Op) and any(
                    a in self.s.needed for r in inner.regions for a in r.args
                ):
                    return True
            if any(a in self.s.needed for a in region.args):
                return True
        return False

    @staticmethod
    def _control_operands(op: Op) -> list:
        if op.name == "scf.for":
            return op.operands[:3]
        if op.name == "scf.if":
            return op.operands[:1]
        return terminator(op.regions[0]).operands[
            :1
        ]  # scf.while: the scf.condition flag
