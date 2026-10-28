# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Program slice of a kernel: the ops and SSA values the KIR needs.

Kept ops are the op table's events (global memory accesses, TMA, mbarrier operations,
device asserts) and every scf op whose regions contain a kept op. Relevant values are
the operands the op table names for each event (addresses, coordinates, barrier
pointers, parities, byte counts, conditions), the control operands of kept scf ops
(branch conditions, loop bounds, while conditions) and, transitively, every operand of
an op defining a relevant value, including the yields and initial values that flow
into relevant scf results and block arguments. A relevant scf result also keeps its op
(control dependence: the branch condition or trip count decides the value).

Everything else is dropped. Whether a value may be dropped is checked during
translation: a relevant value defined by an op the table classifies as data (float
math, register/shared/tensor memory, MMA) is the slice soundness error, reported with
the defining op and the kept use it reaches.
"""

_SCF = ("scf.if", "scf.for", "scf.while")


class Slice:
    def __init__(self, func, effect_operands):
        """`effect_operands(op)` -> kept operand indices if op is an event, else None."""
        self.kept: set = set()  # id(op)
        self.relevant: set = set()  # value names
        self._defs: dict = {}
        self._parent: dict = {}
        ops = []
        self._index(func, None, ops)
        work = []
        for op in ops:
            idx = effect_operands(op)
            if idx is not None:
                self._keep(op, work)
                work += [op.operands[i] for i in idx]
        self._propagate(work)

    def _index(self, op, parent, ops: list) -> None:
        ops.append(op)
        self._parent[id(op)] = parent
        for i, r in enumerate(op.results):
            self._defs[r] = ("res", op, i)
        for ri, region in enumerate(op.regions):
            for block in region:
                for ai, (name, _ty) in enumerate(block.args):
                    self._defs[name] = ("arg", op, ri, ai)
                for o in block.ops:
                    self._index(o, op, ops)

    def _keep(self, op, work: list) -> None:
        """Keep op and its enclosing scf ops (with their control operands)."""
        while op is not None and id(op) not in self.kept:
            self.kept.add(id(op))
            work += _control_operands(op)
            op = self._parent[id(op)]

    def _propagate(self, work: list) -> None:
        while work:
            v = work.pop()
            if v in self.relevant:
                continue
            self.relevant.add(v)
            d = self._defs.get(v)
            if d is None:
                continue
            if d[0] == "res":
                _, op, i = d
                if op.name in _SCF:
                    self._keep(op, work)
                    work += _result_sources(op, i)
                else:
                    work += op.operands
            else:
                _, op, ri, ai = d
                if op.name in _SCF:
                    self._keep(op, work)
                    work += _arg_sources(op, ri, ai)

    def wanted(self, op) -> bool:
        return id(op) in self.kept or any(r in self.relevant for r in op.results)


def terminator(block):
    return block.ops[-1] if block.ops else None


def _control_operands(op) -> list:
    if op.name == "scf.if":
        return [op.operands[0]]
    if op.name == "scf.for":
        return list(op.operands[:3])
    if op.name == "scf.while":
        return [terminator(op.regions[0][0]).operands[0]]
    return []


def _result_sources(op, i: int) -> list:
    """Values flowing into result i of an scf op."""
    if op.name == "scf.if":
        return [terminator(r[0]).operands[i] for r in op.regions if r and r[0].ops]
    if op.name == "scf.for":
        return [op.operands[3 + i], terminator(op.regions[0][0]).operands[i]]
    return [
        terminator(op.regions[0][0]).operands[1 + i]
    ]  # scf.while: scf.condition args


def _arg_sources(op, region: int, i: int) -> list:
    """Values flowing into block argument i of region `region` of an scf op."""
    if op.name == "scf.for":
        if i == 0:
            return list(op.operands[:3])  # induction variable
        return [op.operands[3 + i - 1], terminator(op.regions[0][0]).operands[i - 1]]
    if region == 0:  # scf.while before region: initial values and after-region yields
        return [op.operands[i], terminator(op.regions[1][0]).operands[i]]
    return [terminator(op.regions[0][0]).operands[1 + i]]
