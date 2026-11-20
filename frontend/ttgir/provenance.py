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

import re

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
# Instructions of an inline asm that write no memory (the mnemonic's first part; ld reads)
ASM_NO_WRITE = {
    "abs", "add", "and", "bfe", "bfi", "bfind", "brev", "clz", "cos", "cvt", "cvta",
    "div", "ex2", "fma", "ld", "lg2", "lop3", "mad", "max", "min", "mov", "mul", "neg",
    "not", "or", "popc", "prmt", "rcp", "rem", "rsqrt", "selp", "set", "setp", "shf",
    "shl", "shr", "sin", "slct", "sqrt", "sub", "tanh", "testp", "xor",
}  # fmt: skip


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
                written |= _asm_writes(op, body, parents)
    except _Unknown:
        return None
    return written


def _asm_writes(op: Op, body: Block, parents: dict) -> set:
    """The parameters a side-effecting inline asm writes: every instruction of its text
    writes no memory (ASM_NO_WRITE) and no input register, or is a st/red/atom (shared
    memory: no parameter) whose address is [$k] or [$k +- n], $k a register of a pointer
    operand; the roots of those operands. The registers are numbered as Triton packs
    them: the outputs first, then each operand in order, a pointer taking
    packed_element registers (constraint "l"), any other operand packed_element /
    (elements per 32-bit register); a pointer's registers are exactly the "l" ones, and
    clobbers come last. Else _Unknown (any pointer argument: side-effecting asm may
    store anywhere)."""
    asm = re.match(r'tt\.elementwise_inline_asm\s+"((?:[^"\\]|\\.)*)"', op.text)
    cons = re.search(r'constraints = "([^"]*)"', op.text)
    packed = re.search(r"packed_element = (\d+)", op.text)
    if asm is None or cons is None or packed is None:
        raise _Unknown()
    types = _asm_operand_types(op)
    n = int(packed[1])
    cs = cons[1].split(",")
    regs = [c for c in cs if not c.startswith("~")]
    outs = sum(1 for c in regs if c.startswith("="))
    if (
        cs[: len(regs)] != regs  # clobbers last
        or any(not re.fullmatch(r"=?[A-Za-z]", c) for c in regs)  # no matching ones
        or regs[:outs] != [c for c in regs if c.startswith("=")]
    ):
        raise _Unknown()
    reg_ptr, k = {}, outs  # input register -> its pointer operand
    for v, t in zip(op.operands, types):
        bits = _elem_bits(t)
        span = n if bits == 64 else n // min(max(32 // bits, 1), n)
        for j in range(k, k + span):
            if j >= len(regs) or (regs[j] == "l") != ("!tt.ptr" in t):
                raise _Unknown()
            if "!tt.ptr" in t:
                reg_ptr[j] = v
        k += span
    if k != len(regs):
        raise _Unknown()
    written = set()
    for ins in re.split(r";|\\0A", asm[1].replace("\\09", " ")):
        ins = re.sub(r"^@!?\w+\s+", "", ins.strip(" {}"))
        if not ins or ins.startswith("."):
            continue  # a directive (.reg)
        mn = ins.split()[0]
        base = mn.split(".")[0]
        if base in ASM_NO_WRITE or (base in ("st", "red", "atom") and ".shared" in mn):
            dst = ins[len(mn) :].split(",")[0]
            if base not in ("st", "red") and any(
                int(x) >= outs for x in re.findall(r"\$(\d+)", dst)
            ):
                raise _Unknown()  # it rewrites an input register
            continue
        a = re.search(r"\[\s*\$(\d+)\s*(?:[+-]\s*\d+\s*)?\]", ins)
        if base not in ("st", "red", "atom") or a is None or int(a[1]) not in reg_ptr:
            raise _Unknown()
        if base == "atom" and any(
            int(x) >= outs for x in re.findall(r"\$(\d+)", ins[len(mn) :].split(",")[0])
        ):
            raise _Unknown()  # its result rewrites an input register
        written |= _roots(reg_ptr[int(a[1])], body, parents, set())
    return written


def _asm_operand_types(op: Op) -> list:
    """The operand types of an elementwise_inline_asm (its signature before `->`)."""
    if not op.operands:
        return []
    sig = strip_strings(op.text).rsplit(" : ", 1)[-1].split(" -> ")[0]
    types, depth, cur = [], 0, ""
    for ch in sig + ",":
        depth += {"<": 1, ">": -1}.get(ch, 0)
        if ch == "," and depth == 0:
            types.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if len(types) != len(op.operands):
        raise _Unknown()
    return types


def _elem_bits(t: str) -> int:
    """Bit width of a (tensor of) scalar or pointer type: 64 for a pointer."""
    if "!tt.ptr" in t:
        return 64
    m = re.search(r"(?:^|[x<\s])(i1|i8|i16|i32|i64|bf16|f16|f32|f64|f8\w*)\b", t)
    if m is None:
        raise _Unknown()
    s = m[1]
    return 8 if s.startswith("f8") else 1 if s == "i1" else int(re.sub(r"\D", "", s))


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
