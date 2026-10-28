# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""TTGIR -> KIR (references/kir.md), Layer 1 for TLX/Triton.

    to_kir --dump /tmp/lkv_out/tlx/<case>/dump.json --case CASE.json

For every specialization in dump.json: parse its TTGIR (mlir_text), slice it (slicing),
translate the kept ops with the op table (op_table, references/ir_ops.md) and write
<spec>.kir.json next to the dump, with `case.inputs` holding every input whose launch
compiled to that specialization. Every unsupported construct on the kept slice and every
dropped op whose result reaches the slice is a hard error naming the op and its loc, of
kind "gap" (refused by design) or "error" (the case or LKV must be fixed; `KirError`).
"""

import argparse
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from .. import kir as kir_contract
from . import kir_build as K, op_table
from .mlir_text import (
    Block,
    Op,
    parse,
    ParseError,
    REPO_PREFIX,
    split_top,
    strip_strings,
    Value,
    walk,
)
from .op_table import OpError
from .slicing import SliceError, Slicer, terminator

# Fuel for `while` loops: the frontend does not prove loop bounds, so `max_iters` is a
# budget; reaching it is a verdict error in Layer 2, never a silent pass.
WHILE_FUEL = 1 << 16
CLC_SCHEDULES = ("none", "round_robin", "greedy")
# Bytes per element of the tensormap `elem_type` values Triton's tensormap_create carries:
# the PTX `tensormap.replace .elemtype` encoding (0 u8, 1 u16, 2 u32, 3 s32, 4 u64, 5 s64,
# 6 f16, 7 f32, 8 f32.ftz, 9 f64, 10 bf16, 11 tf32, 12 tf32.ftz); 13-15 are the packed
# sub-byte formats (b4x16, b4x16_p64, b6x16_p32).
TMA_ELEM_BYTES = {
    0: 1,
    1: 2,
    2: 4,
    3: 4,
    4: 8,
    5: 8,
    6: 2,
    7: 4,
    8: 4,
    9: 8,
    10: 2,
    11: 4,
    12: 4,
}


class KirError(Exception):
    """A construct the frontend cannot translate exactly. `kind`: "gap", a construct it
    refuses by design (a known limit, not an error), or "error", something the case or LKV
    must fix (the `[to_kir] <spec>: FAILED (<kind>): ...` line)."""

    def __init__(self, msg: str, kind: str = "gap"):
        super().__init__(msg)
        self.kind = kind


def failure_kind(e: Exception) -> str:
    """ "gap" or "error" for an exception that stopped a translation (KirError.kind; a
    TTGIR text the parser cannot read is an error, a slice or op-table refusal a gap)."""
    return getattr(e, "kind", "error" if isinstance(e, ParseError) else "gap")


def _mod_int(attrs: str, key: str, default: int | None = None) -> int | None:
    m = re.search(rf'"?{re.escape(key)}"? = (\d+) : i32', attrs)
    return int(m.group(1)) if m else default


class Translator:
    def __init__(self, module, case: dict, work_fields: dict):
        self.m = module
        self.body: Block = module.func.regions[0]
        num_warps = _mod_int(module.attrs, "ttg.num-warps")
        if num_warps is None:
            raise KirError(
                "module attribute ttg.num-warps (warps of the default role) is missing"
            )
        self.num_warps: int = num_warps
        if _mod_int(module.attrs, "ttg.threads-per-warp") != 32:
            raise KirError("module attribute ttg.threads-per-warp must be 32")
        self.cluster = [
            _mod_int(module.attrs, f"ttg.cluster-dim-{a}", 1) for a in "xyz"
        ]
        # Multi-CTA layouts scale barrier counts and paired (2-CTA) MMA modules multicast
        # every tcgen05 commit to the peer CTA: both need cross-CTA events (not in KIR v1).
        if _mod_int(module.attrs, "ttg.num-ctas", 1) != 1:
            raise KirError("ttg.num-ctas > 1 (multi-CTA layouts) is not in KIR v1")
        for flag in ("ttng.two-ctas", "tlx.enable_paired_cta_mma"):
            if re.search(rf'"?{re.escape(flag)}"? = true', module.attrs):
                raise KirError(
                    f"module attribute {flag} (2-CTA paired MMA) is not in KIR v1"
                )
        self.case = case
        self.work_fields = work_fields
        self.tensors: dict[str, dict] = {}
        self.barriers: dict[str, dict] = {}
        self.metadata: set[str] = set()
        self.written: set[str] = set()
        self.uses_clc = False
        self.device_descs: dict[
            int, dict
        ] = {}  # id(scratch op) -> in-kernel descriptor
        self.desc_bases: dict[
            str, str
        ] = {}  # KIR desc tensor of an in-kernel descriptor -> base pointer
        self.param_names = {v.name for v in module.args if v.type in K.INT_WIDTHS}
        self.captures: dict = {}
        self.work_events = 0
        for op in walk(self.body):
            op_table.classify(op)
        self.slice = Slicer(self.body).run_with(self._work_roots())

    # -- roles ---------------------------------------------------------------------------
    def roles(self) -> list[dict]:
        ws_ops = [op for op in walk(self.body) if op.name == "ttg.warp_specialize"]
        if not ws_ops:
            return [
                self._role(
                    "default", self.num_warps, lambda: self._region(self.body), []
                )
            ]
        ws = ws_ops[0]
        if (
            len(ws_ops) > 1
            or ws not in self.body.ops
            or self.body.more
            or ws.regions[0].more
        ):
            raise KirError(
                f"{ws.where()}: only one top-level warp_specialize in a structured "
                f"kernel body is supported"
            )
        k = self.body.ops.index(ws)
        ops = self.body.ops[:k] + ws.regions[0].ops + self.body.ops[k + 1 :]
        roles = [self._role("default", self.num_warps, lambda: self._block(ops), [])]
        for idx, (region, hdr) in enumerate(zip(ws.regions[1:], ws.region_headers[1:])):
            m = re.search(r"num_warps\((\d+)\)", hdr)
            if not m:
                raise KirError(f"partition{idx} of {ws.where()} has no num_warps")
            binds = list(zip(region.args, ws.operands))
            roles.append(
                self._role(
                    f"partition{idx}",
                    int(m.group(1)),
                    lambda r=region: self._region(r),
                    binds,
                )
            )
        return roles

    def _region(self, block: Block) -> list:
        return self._cfg(block) if block.more else self._block(block.ops)

    def _cfg(self, entry: Block) -> list:
        """An acyclic CFG without block arguments (e.g. early `return`) as nested ifs.

        A block reached along several paths is emitted on each path; that is exact for
        acyclic control flow. Loops and block arguments are hard errors.
        """
        blocks = {b.label: b for b in entry.more}

        def go(b: Block, path: tuple) -> list:
            if b.label in path:
                raise KirError(f"unstructured loop through block {b.label}")
            if b is not entry and b.args:
                raise KirError(
                    f"block {b.label} has arguments (unstructured control flow)"
                )
            term = b.ops[-1]
            if term.name not in ("cf.cond_br", "cf.br"):
                return self._block(b.ops)
            succ = re.findall(r"\^(\w+)(\()?", strip_strings(term.text))
            if any(paren for _, paren in succ):
                raise KirError(f"branch with block operands {term.where()}")
            stmts = self._block(b.ops[:-1])
            targets = [blocks["^" + name] for name, _ in succ]
            if term.name == "cf.br":
                return stmts + go(targets[0], path + (b.label,))
            cond = self._int(term.operands[0])["e"]
            then, else_ = (go(t, path + (b.label,)) for t in targets)
            return stmts + [
                self._stmt("if", term, cond=cond, then=then, **{"else": else_})
            ]

        return go(entry, ())

    def _role(self, name: str, warps: int, emit, captures: list) -> dict:
        self.env: dict = {}
        self.role = name
        self.warps = warps
        # Partitions are isolated from above: a captured value is rebuilt from kernel
        # arguments only (inline); integer captures are bound once at role entry.
        self.captures = dict(captures)
        body = []
        for arg, outer in captures:
            if arg in self.slice.needed and arg.type in K.INT_WIDTHS:
                e = self._inline(outer)["e"]
                body.append(self._stmt("let", None, var=arg.name, expr=e))
                self.env[arg] = {"k": "int", "e": K.var(arg.name, e["w"])}
        body += emit()
        if self.uses_clc:
            launched = K.cmp(
                "ne", K.load("__clc_launched", [K.pid(0)], 32), K.const(0, 32)
            )
            body = [
                self._stmt("let", None, var="__clc_k", expr=K.const(0, 32)),
                self._stmt("if", None, cond=launched, then=body, **{"else": []}),
            ]
        return {"name": name, "warps": warps, "threads_per_issue": 1, "body": body}

    # -- statements ----------------------------------------------------------------------
    @staticmethod
    def _stmt(kind: str, op: Op | None, **fields) -> dict:
        s = {"op": kind, **fields}
        if op is not None and op.loc:
            s["loc"] = op.loc
        return s

    @staticmethod
    def _event(op: Op, ev: str, **fields) -> dict:
        """An event statement: the Event object inline (frontend/kir.py), loc required."""
        return {"op": "event", "ev": ev, **fields, "loc": op.loc}

    def _guard(self, op: Op, pred: Value | None, stmts: list) -> list:
        if pred is None:
            return stmts
        c = self._int(pred)["e"]
        if c["e"] == "const":
            return stmts if c["v"] & 1 else []
        return [self._stmt("if", op, cond=c, then=stmts, **{"else": []})]

    def _block(self, ops: list) -> list:
        out = []
        for op in ops:
            out += self._op(op)
        return self._with_work(ops, out)

    def _with_work(self, ops: list, stmts: list) -> list:
        """Insert the case's `work` event after the last definition of its fields.

        Emitted in every block that defines all declared fields (each exactly once);
        roles that never compute some field (e.g. an MMA role that needs no head index)
        get no work event.
        """
        if not self.work_fields:
            return stmts
        defs = {}
        for op in ops:
            for r in op.results:
                for field_, pyvar in self.work_fields.items():
                    if op.loc_name == pyvar and self._is_int(r):
                        defs.setdefault(field_, []).append((op, r))
        if set(defs) != set(self.work_fields):
            return stmts
        if any(len(d) != 1 for d in defs.values()):
            raise KirError(
                f"work fields {self.work_fields}: several integer definitions in one "
                f"block: {[(f, [o.where() for o, _ in d]) for f, d in defs.items()]}",
                kind="error",
            )
        anchor = ops[max(ops.index(d[0][0]) for d in defs.values())]
        fields = {f: self._int(d[0][1])["e"] for f, d in defs.items()}
        self.work_events += 1
        ev = self._event(anchor, "work", fields=fields)
        names = {r.name for r in anchor.results}
        for i, s in enumerate(stmts):
            if s.get("op") == "let" and s.get("var") in names:
                return stmts[: i + 1] + [ev] + stmts[i + 1 :]
        raise KirError(
            f"cannot place the work event after {anchor.where()}", kind="error"
        )

    def _op(self, op: Op) -> list:
        try:
            return self._translate(op)
        except (KeyError, IndexError, AttributeError, ValueError, TypeError) as e:
            raise KirError(
                f"cannot translate {op.where()}: {type(e).__name__}: {e}", kind="error"
            ) from e

    def _translate(self, op: Op) -> list:
        n = op.name
        if n in ("scf.for", "scf.if", "scf.while"):
            return self._structured(op) if op in self.slice.live else []
        if n in (
            "scf.yield",
            "scf.condition",
            "ttg.warp_yield",
            "ttg.warp_return",
            "tt.return",
        ):
            return []
        if n in ("cf.cond_br", "cf.br"):
            raise KirError(
                f"unstructured control flow outside a kernel/partition body {op.where()}"
            )
        if n == "ttg.warp_specialize":
            raise KirError(f"nested warp_specialize {op.where()}")
        stmts = self._events(op) if op_table.has_effect(op) else []
        if n == "ttng.clc_query_cancel":
            return stmts
        for r in op.results:
            if r in self.slice.needed and self._is_int(r) and n != "arith.constant":
                s = self._build(op, r)
                if s["k"] != "int":
                    raise KirError(
                        f"internal: {op.where()} expected an integer", kind="error"
                    )
                expr = K.mark_addr(s["e"]) if "addr" in self.slice.needed[r] else s["e"]
                stmts.append(self._stmt("let", op, var=r.name, expr=expr))
                self.env[r] = {"k": "int", "e": K.var(r.name, expr["w"])}
        return stmts

    # -- structured control flow ---------------------------------------------------------
    def _structured(self, op: Op) -> list:
        need = self.slice.needed
        if op.name == "scf.for":
            if strip_strings(op.text).startswith("scf.for unsigned"):
                raise KirError(
                    f"scf.for with unsigned bound comparison (unsignedCmp) {op.where()}: "
                    "KIR `for` compares signed"
                )
            lb, ub, step = (self._int(v)["e"] for v in op.operands[:3])
            for e in (lb, ub, step):
                if e["w"] != 32:
                    raise KirError(f"non-i32 loop bounds {op.where()}")
            body_blk = op.regions[0]
            iv = body_blk.args[0]
            carried = [
                (a, init, k)
                for k, (a, init) in enumerate(zip(body_blk.args[1:], op.operands[3:]))
                if a in need
            ]
            self._scalar_ints(op, [a for a, _, _ in carried])
            pre = []
            for a, init, _ in carried:
                e = self._int(init)["e"]
                pre.append(self._stmt("let", op, var=a.name, expr=e))
                self.env[a] = {"k": "int", "e": K.var(a.name, e["w"])}
            self.env[iv] = {"k": "int", "e": K.var(iv.name, 32)}
            body = self._block(body_blk.ops)
            body += self._carry(
                op, terminator(body_blk), [(a, k) for a, _, k in carried]
            )
            stmts = pre + [
                self._stmt("for", op, var=iv.name, lb=lb, ub=ub, step=step, body=body)
            ]
            # Results get their own names: the printer numbers the args of sibling regions
            # alike, so a later loop's pre-lets would rebind %argN (as in _while).
            for r in op.results:
                if r in need:
                    e = K.var(body_blk.args[r.index + 1].name, self._width(r))
                    stmts.append(self._stmt("let", op, var=r.name, expr=e))
                    self.env[r] = {"k": "int", "e": K.var(r.name, e["w"])}
            return stmts
        if op.name == "scf.if":
            cond = self._int(op.operands[0])["e"]
            # Results are assigned in both branches; a name first assigned inside a branch
            # is not readable after it, so each result is declared (as 0) before the if.
            self._scalar_ints(op, [r for r in op.results if r in need])
            pre = [
                self._stmt("let", op, var=r.name, expr=K.const(0, self._width(r)))
                for r in op.results
                if r in need
            ]
            branches = []
            for region in op.regions:
                stmts = self._block(region.ops)
                y = terminator(region)
                for r in op.results:
                    if r in need:
                        stmts.append(
                            self._stmt(
                                "let",
                                y,
                                var=r.name,
                                expr=self._int(y.operands[r.index])["e"],
                            )
                        )
                branches.append(stmts)
            if len(branches) == 1:
                branches.append([])
            for r in op.results:
                if r in need:
                    self.env[r] = {"k": "int", "e": K.var(r.name, self._width(r))}
            return pre + [
                self._stmt(
                    "if", op, cond=cond, then=branches[0], **{"else": branches[1]}
                )
            ]
        return self._while(op)

    def _scalar_ints(self, op: Op, values: list) -> None:
        """Values carried by a loop or yielded by an if must be scalar integers (KIR variables)."""
        for v in values:
            if K.is_ptr_type(self._type(v)):
                raise KirError(
                    f"pointer %{v.name} carried by {op.where()} (e.g. `ptrs += BLOCK` in a loop): "
                    "KIR carries integers only; compute the pointer from the loop index"
                )
            if not self._is_int(v):
                raise KirError(
                    f"%{v.name} of type {self._type(v)} carried by {op.where()}: KIR variables are "
                    "scalar i1/i32/i64 integers (for a tensor such as `offs += BLOCK` in a loop, "
                    "compute it from the loop index)"
                )

    def _carry(self, op: Op, yield_op: Op, carried: list) -> list:
        """Parallel assignment of loop-carried values at the end of an iteration."""
        exprs = [(a, self._int(yield_op.operands[k])["e"]) for a, k in carried]
        stmts = [
            self._stmt("let", yield_op, var=f"{a.name}.next", expr=e) for a, e in exprs
        ]
        stmts += [
            self._stmt(
                "let", yield_op, var=a.name, expr=K.var(f"{a.name}.next", e["w"])
            )
            for a, e in exprs
        ]
        return stmts

    def _while(self, op: Op) -> list:
        need = self.slice.needed
        before, after = op.regions
        cond_op = terminator(before)
        carried = [(a, k) for k, a in enumerate(before.args) if a in need]
        self._scalar_ints(
            op, [a for a, _ in carried] + [b for b in after.args if b in need]
        )
        pre = []
        for a, k in carried:
            e = self._int(op.operands[k])["e"]
            pre.append(self._stmt("let", op, var=a.name, expr=e))
            self.env[a] = {"k": "int", "e": K.var(a.name, e["w"])}
        # The before region runs before every test, so it is emitted before the loop and
        # again at the end of each iteration.
        before_stmts = self._block(before.ops)
        cond = self._int(cond_op.operands[0])["e"]
        # The after args are printed with the before args' names, so they are assigned in
        # parallel (through `.in` temporaries): scf.condition may forward them permuted.
        fwd = [
            (b, self._int(cond_op.operands[b.index + 1])["e"])
            for b in after.args
            if b in need
        ]
        body = [self._stmt("let", cond_op, var=f"{b.name}.in", expr=e) for b, e in fwd]
        for b, e in fwd:
            tmp = K.var(f"{b.name}.in", e["w"])
            body.append(self._stmt("let", cond_op, var=b.name, expr=tmp))
            self.env[b] = {"k": "int", "e": K.var(b.name, e["w"])}
        body += self._block(after.ops)
        body += self._carry(op, terminator(after), carried)
        body += self._block(before.ops)
        post = []
        for r in op.results:
            if r in need:
                e = self._int(cond_op.operands[r.index + 1])["e"]
                post.append(self._stmt("let", op, var=r.name, expr=e))
                self.env[r] = {"k": "int", "e": K.var(r.name, e["w"])}
        loop = self._stmt("while", op, cond=cond, body=body, max_iters=WHILE_FUEL)
        return pre + before_stmts + [loop] + post

    # -- events --------------------------------------------------------------------------
    def _events(self, op: Op) -> list:
        n = op.name
        try:
            if n == "ttng.init_barrier":
                b = op_table.barrier_op(op)
                bar = self._bar(b["bar"], op)
                slot = self.barriers.setdefault(
                    bar["name"], {"count": b["count"], "num": bar["num"]}
                )
                if slot["count"] != b["count"]:
                    raise KirError(
                        f"barrier {bar['name']} slots initialized with different counts {op.where()}"
                    )
                return [
                    self._event(
                        op,
                        "bar_init",
                        bar=bar["name"],
                        idx=bar["idx"],
                        count=K.const(b["count"], 32),
                    )
                ]
            if n == "ttng.arrive_barrier":
                b = op_table.barrier_op(op)
                attrs = b["attrs"]
                if "ctaMask" in attrs:
                    raise KirError(
                        f"cross-CTA (ctaMask) arrive is not in KIR v1: {op.where()}"
                    )
                n_arr = b["count"] * (self.warps * 32 if "perThread" in attrs else 1)
                bar = self._bar(b["bar"], op)
                ev = self._event(
                    op,
                    "arrive",
                    bar=bar["name"],
                    idx=bar["idx"],
                    n=n_arr,
                    per_warp=False,
                )
                return self._guard(op, b["pred"], [ev])
            if n == "ttng.barrier_expect":
                b = op_table.barrier_op(op)
                bar = self._bar(b["bar"], op)
                evs = [
                    self._event(
                        op,
                        "expect_tx",
                        bar=bar["name"],
                        idx=bar["idx"],
                        bytes=K.const(b["bytes"], 32),
                    ),
                    self._event(
                        op,
                        "arrive",
                        bar=bar["name"],
                        idx=bar["idx"],
                        n=1,
                        per_warp=False,
                    ),
                ]
                return self._guard(op, b["pred"], evs)
            if n == "ttng.wait_barrier":
                b = op_table.barrier_op(op)
                bar = self._bar(b["bar"], op)
                ev = self._event(
                    op,
                    "wait",
                    bar=bar["name"],
                    idx=bar["idx"],
                    parity=self._int(b["phase"])["e"],
                )
                return self._guard(op, b["pred"], [ev])
            if n in ("ttng.tc_gen5_mma", "ttng.tc_gen5_mma_scaled"):
                attrs = op_table.attr_dict(op)
                if "two_ctas" in attrs or "multicast" in attrs:
                    raise KirError(
                        f"2-CTA / multicast MMA completion is not in KIR v1: {op.where()}"
                    )
                pred, bars = op_table.mma_barriers(op)
                out = []
                for bv, bp in bars:
                    bar = self._bar(bv, op)
                    ev = self._event(
                        op, "commit_arrive", bar=bar["name"], idx=bar["idx"]
                    )
                    out += self._guard(op, bp, [ev])
                return self._guard(op, pred, out)
            if n == "ttng.tc_gen5_commit":
                b = op_table.barrier_op(op)
                bar = self._bar(b["bar"], op)
                return self._guard(
                    op,
                    b["pred"],
                    [self._event(op, "commit_arrive", bar=bar["name"], idx=bar["idx"])],
                )
            if n in op_table.TMA_OPS:
                t = op_table.tma_op(op)
                d = self._sym(t["desc"])
                if d["k"] != "desc":
                    raise KirError(
                        f"{op.where()}: %{t['desc'].name} is not a TMA descriptor"
                    )
                block = self._tma_block(op, d["dev"])
                coords = [K.mark_addr(self._int(c)["e"]) for c in t["coords"]]
                if n == "ttng.async_tma_copy_global_to_local":
                    bar = self._bar(t["bar"], op)
                    ev = self._event(
                        op,
                        "tma_load",
                        desc=self._load_desc(op, d, block),
                        coords=coords,
                        bar=bar["name"],
                        idx=bar["idx"],
                    )
                    return self._guard(op, t["pred"], [ev])
                if d["dev"] is not None:
                    return [self._clipped_store(op, d["dev"], block, coords)]
                self.tensors.setdefault(
                    d["name"], {"name": d["name"], "kind": "desc"}
                )  # build() completes it
                self.written.add(d["name"])
                kind = (
                    "tma_store"
                    if n == "ttng.async_tma_copy_local_to_global"
                    else "tma_reduce"
                )
                return [self._event(op, kind, desc=d["name"], coords=coords)]
            if n in ("tt.load", "tt.store", "tt.atomic_rmw", "tt.atomic_cas"):
                return self._mem_event(op)
            if n == "ttng.async_clc_try_cancel":
                # The 16-byte response lands through the async proxy and completes its
                # transaction bytes on the barrier: a tma_load of the constant 16-byte
                # descriptor __clc_response (what the response says comes from the
                # schedule, see _clc_query).
                self._clc_check(op)
                bar = self._bar(op.operands[0], op)
                return [
                    self._event(
                        op,
                        "tma_load",
                        desc="__clc_response",
                        coords=[K.const(0, 32)],
                        bar=bar["name"],
                        idx=bar["idx"],
                    )
                ]
            if n == "ttng.clc_query_cancel":
                return self._clc_query(op)
            if n == "ttng.tensormap_create":
                return self._tensormap(op)
            if n == "tt.assert":
                c = self._sym(op.operands[0])
                if c["k"] != "int":
                    raise KirError(f"tensor-valued device assert {op.where()}")
                m = re.search(r'"((?:[^"\\]|\\.)*)"', op.text)
                return [
                    self._stmt("assert", op, cond=c["e"], msg=m.group(1) if m else "")
                ]
        except OpError as e:
            raise KirError(str(e)) from e
        raise KirError(f"internal: no event rule for {op.where()}", kind="error")

    def _mem_event(self, op: Op) -> list:
        m = op_table.mem_op(op)
        p = self._sym(m["ptr"])
        if p["k"] != "ptr":
            raise KirError(
                f"pointer of {op.where()} does not derive from a kernel pointer argument"
            )
        offs = self._offsets(p)
        mask = None
        if m["mask"] is not None:
            mk = self._sym(m["mask"])
            mask = mk["t"] if mk["k"] == "tensor" else K.t_splat(mk["e"], offs["shape"])
        kind = {"tt.load": "gload", "tt.store": "gstore"}.get(op.name, "gatomic")
        if kind != "gload":
            self.written.add(p["tensor"])
        return [self._event(op, kind, tensor=p["tensor"], offsets=offs, mask=mask)]

    @staticmethod
    def _scalar_offset(p: dict) -> dict:
        """Element offset of a scalar pointer: the sum of its sign-extended terms (i64)."""
        off = K.const(0, 64)
        for t in p["terms"]:
            e = K.ext(t, 64, True)
            off = e if off == K.const(0, 64) else K.bin_("add", off, e, 64, True)
        return off

    def _offsets(self, p: dict) -> dict:
        """Element offsets TExpr of a (tensor of) pointer(s) for a global access event.

        A single tensor offset over the base argument stays in its own width: offsets are
        read as signed integers, which is the sign extension addptr applies, and i32
        address arithmetic keeps its overflow checks. Otherwise the terms are summed in
        i64, extending i32 tensor terms elementwise (TExpr `ext`).
        """
        scalars = [t for t in p["terms"] if not K.is_texpr(t)]
        tensors = [t for t in p["terms"] if K.is_texpr(t)]
        shape = p["shape"] or [1]
        if len(tensors) == 1 and all(K.is_const_zero(t) for t in scalars):
            return tensors[0]
        off = K.t_splat(self._scalar_offset({"terms": scalars}), shape)
        for t in tensors:
            off = K.t_bin("add", off, K.t_ext(t, 64, True), 64, True)
        return off

    def _tensormap(self, op: Op) -> list:
        """A TMA descriptor created in the kernel (`tl.make_tensor_descriptor`).

        Recorded under the scratch buffer it is written to: base tensor, extents, element
        strides and tensormap box. Extents that are host-side expressions (params and
        constants) are used inline; runtime extents (e.g. a segment end loaded from
        metadata) are bound by `let`s here. Loads through it are `tma_load`s of a `desc`
        tensor (host-side extents only); stores and reduces are global accesses of the
        base tensor with the TMA footprint (_clipped_store).
        """
        t = op_table.tensormap_op(op)
        if "interleave_layout = 0" not in t["attrs"]:
            raise KirError(
                f"interleaved tensormap is not in the op table: {op.where()}"
            )
        for v in t["estride"]:
            if self._int(v)["e"] != K.const(1, 32):
                raise KirError(f"tensormap element strides must be 1: {op.where()}")
        g = self._sym(t["global"])
        if g["k"] != "ptr" or g["terms"]:
            raise KirError(
                f"tensormap global address must be a kernel pointer argument: {op.where()}"
            )
        scratch = t["desc_ptr"].owner
        if not isinstance(scratch, Op) or scratch.name != "ttg.global_scratch_alloc":
            raise KirError(
                f"tensormap target is not a global_scratch_alloc: {op.where()}"
            )
        elem = self.tensors[g["tensor"]]["elem_bytes"]
        m = re.search(r"elem_type = (\d+)", t["attrs"])
        etype = int(m.group(1)) if m else None
        if etype not in TMA_ELEM_BYTES:
            raise KirError(
                f"tensormap elem_type {etype} (packed sub-byte data) is not in the op table: {op.where()}"
            )
        if TMA_ELEM_BYTES[etype] != elem:
            raise KirError(
                f"tensormap elem_type {etype} has {TMA_ELEM_BYTES[etype]}-byte elements but "
                f"%{g['tensor']} has {elem}-byte elements: {op.where()}"
            )
        box = [self._int(v)["e"] for v in t["box"]]
        if any(b["e"] != "const" for b in box):
            raise KirError(f"non-constant TMA box {op.where()}")
        name = t["desc_ptr"].name
        prev = self.device_descs.get(id(scratch))
        if prev is not None and prev["op"] is not op:
            raise KirError(
                f"{op.where()} writes a second TMA descriptor into %{name} (first at "
                f"{prev['op'].where()}): which one a use reads depends on the execution order"
            )
        # tensormap operands are innermost-first; KIR descriptors are outermost-first. Its
        # strides are in bytes for every dim but the innermost (element stride 1).
        rec = {
            "name": name,
            "op": op,
            "base": g["tensor"],
            "elem_bytes": elem,
            "role": self.role,
            "box": [b["v"] for b in reversed(box)],
        }
        self.device_descs[id(scratch)] = rec
        dims = [self._try_inline(v) for v in reversed(t["gdim"])]
        strides = [self._try_inline(v) for v in reversed(t["gstride"])]
        rec["runtime"] = not all(
            e is not None and self._host_side(e) for e in dims + strides
        )
        if not rec["runtime"]:
            rec["shape"] = dims
            # Not runtime: every dim and stride inlined, so no entry is None.
            rec["strides"] = [
                self._elem_stride(e, elem) for e in strides if e is not None
            ] + [K.const(1, 64)]
            return []
        stmts, rec["shape"], rec["strides"] = [], [], []
        for i, v in enumerate(reversed(t["gdim"])):
            e = self._int(v)["e"]
            stmts.append(self._stmt("let", op, var=f"{name}.shape.{i}", expr=e))
            rec["shape"].append(K.var(f"{name}.shape.{i}", e["w"]))
        for i, v in enumerate(reversed(t["gstride"])):
            stmts.append(
                self._stmt(
                    "let",
                    op,
                    var=f"{name}.stride.{i}",
                    expr=self._elem_stride(self._int(v)["e"], elem),
                )
            )
            rec["strides"].append(K.var(f"{name}.stride.{i}", 64))
        rec["strides"].append(K.const(1, 64))
        return stmts

    @staticmethod
    def _elem_stride(byte_stride: dict, elem: int) -> dict:
        return K.bin_("sdiv", K.ext(byte_stride, 64, True), K.const(elem, 64), 64)

    def _tma_block(self, op: Op, dev: dict | None) -> list:
        """The block a TMA op moves: the block shape of its `!tt.tensordesc` type.

        The lowering (LoadStoreOpToLLVM.cpp) cuts the shared-memory tile into messages of
        K.tma_message_box(tile) and the hardware moves one tensormap box per message: the
        host's (getTMABlockShape of the descriptor type) or the in-kernel tensormap's.
        Together they move exactly the block iff the tile is the block (leading unit dims
        may be dropped), both are the same NVMMA layout (not fp4Padded), and the message
        box divides the block and equals the tensormap box. Anything else is refused.
        """
        sec = K.type_section(op.text)
        desc = K.tensordesc_type(split_top(sec)[0])
        smem = K.memdesc_type((K.types_after(sec, " -> ") or split_top(sec))[-1])
        if desc is None or smem is None or not desc[2] or not smem[2]:
            raise KirError(
                f"cannot read the descriptor and shared-memory types of {op.where()}"
            )
        (block, elem, denc), (tile, _, senc) = desc, smem
        lead = len(block) - len(tile)
        if lead < 0 or block[lead:] != tile or any(b != 1 for b in block[:lead]):
            raise KirError(
                f"{op.where()}: shared-memory tile {tile} is not the descriptor block {block}"
            )
        dp, sp = (K.nvmma_params(self.m.resolve(e[0])) for e in (denc, senc))
        if dp is None or sp is None:
            raise KirError(
                f"{op.where()}: TMA with a non-NVMMA shared encoding is not in the op table"
            )
        if "true" in (dp.get("fp4Padded"), sp.get("fp4Padded")):
            raise KirError(
                f"{op.where()}: fp4Padded TMA (the lowering doubles the innermost "
                "coordinate) is not in the op table"
            )
        if any(
            dp.get(k) != sp.get(k)
            for k in ("swizzlingByteWidth", "transposed", "elementBitWidth")
        ):
            raise KirError(
                f"{op.where()}: descriptor encoding {dp} differs from the shared-memory encoding {sp}"
            )
        if dev is not None and K.elem_bits(elem) != 8 * dev["elem_bytes"]:
            raise KirError(
                f"{op.where()}: descriptor element type {elem} differs from %{dev['base']}'s elements"
            )
        msg = [1] * lead + K.tma_message_box(tile, sp)
        hw = dev["box"] if dev is not None else K.tma_message_box(block, dp)
        if hw != msg or any(b % m for b, m in zip(block, msg)):
            raise KirError(
                f"{op.where()}: tensormap box {hw} (lowering messages {msg}) does not tile "
                f"the descriptor block {block}"
            )
        return block

    def _load_desc(self, op: Op, d: dict, block: list) -> str:
        """The KIR `desc` tensor a TMA load reads (host descriptors are completed in build())."""
        name, dev = d["name"], d["dev"]
        if dev is None:
            self.tensors.setdefault(name, {"name": name, "kind": "desc"})
            return name
        if dev["runtime"]:
            raise KirError(
                f"TMA load through a descriptor created in the kernel with runtime extents "
                f"is not in KIR v1: {op.where()}"
            )
        shape = [
            K.ext(e, 64, False) for e in dev["shape"]
        ]  # unsigned extents, as in _clipped_store
        t = self.tensors.setdefault(
            name,
            {
                "name": name,
                "kind": "desc",
                "elem_bytes": dev["elem_bytes"],
                "shape": shape,
                "strides": dev["strides"],
                "box": block,
            },
        )
        self.desc_bases[name] = dev["base"]
        if t["box"] != block:
            raise KirError(
                f"{op.where()}: descriptor %{name} is used with blocks {t['box']} and {block}"
            )
        return name

    def _try_inline(self, v: Value) -> dict | None:
        """`v` as an expression over kernel arguments, or None if it depends on the role."""
        try:
            s = self._inline(v)
        except KirError:
            return None
        return s["e"] if s["k"] == "int" else None

    def _host_side(self, e: dict) -> bool:
        """True if `e` only uses constants and kernel parameters."""
        if e["e"] == "const":
            return True
        if e["e"] == "var":
            return e["name"] in self.param_names
        if e["e"] in ("pid", "nprog", "load"):
            return False
        return all(self._host_side(e[k]) for k in ("a", "b", "c") if k in e)

    def _clipped_store(self, op: Op, d: dict, box: list, coords: list) -> dict:
        """A TMA store/reduce through an in-kernel descriptor as a global access of its base.

        Element `i` of the block lands at row-major coordinate coords + i of the base tensor
        iff 0 <= coords + i < shape in every dim (TMA drops the rest); its element offset
        is sum((coords + i) * strides). Coordinates are signed i32, extents unsigned (the
        tensormap global_dim is a b32 TMA reads as unsigned: -64 is 2^32 - 64, not empty);
        offsets are i64.
        """
        if d["runtime"] and d["role"] != self.role:
            raise KirError(
                f"{op.where()}: descriptor %{d['name']} with runtime extents is created in role "
                f"{d['role']} and used in role {self.role}"
            )
        rank = len(box)
        offsets = mask = None
        for dim in range(rank):
            idx: dict[str, Any] = K.t_bin(
                "add",
                K.t_splat(K.ext(coords[dim], 64, True), [box[dim]]),
                {**K.t_range(0, box[dim]), "w": 64},
                64,
            )
            for axis in range(rank):  # place along `dim`, broadcast over the other dims
                if axis != dim:
                    shape = idx["shape"][:axis] + [1] + idx["shape"][axis:]
                    idx = {
                        "t": "expand",
                        "shape": shape,
                        "a": idx,
                        "axis": axis,
                        "w": 64,
                    }
            for axis in range(rank):
                if axis != dim:
                    shape = list(idx["shape"])
                    shape[axis] = box[axis]
                    idx = {
                        "t": "bcast",
                        "shape": shape,
                        "a": idx,
                        "axis": axis,
                        "w": 64,
                    }
            term = K.t_bin("mul", idx, K.t_splat(d["strides"][dim], box), 64, True)
            offsets = (
                term if offsets is None else K.t_bin("add", offsets, term, 64, True)
            )
            extent = K.t_splat(K.ext(d["shape"][dim], 64, False), box)
            inside = K.t_bin(
                "and",
                K.t_cmp("sge", idx, K.t_full(0, box, 64)),
                K.t_cmp("slt", idx, extent),
                1,
            )
            mask = inside if mask is None else K.t_bin("and", mask, inside, 1)
        self.written.add(d["base"])
        kind = (
            "gstore" if op.name == "ttng.async_tma_copy_local_to_global" else "gatomic"
        )
        return self._event(op, kind, tensor=d["base"], offsets=offsets, mask=mask)

    def _clc_check(self, op: Op) -> None:
        if self.cluster != [1, 1, 1]:
            raise KirError(
                f"CLC with clusters of {self.cluster} CTAs is not in KIR v1: {op.where()}"
            )
        self.uses_clc = True

    def _clc_query(self, op: Op) -> list:
        self._clc_check(op)
        x = K.load("__clc_schedule", [K.pid(0), K.var("__clc_k", 32)], 32)
        stmts = []
        for r in op.results:
            if r in self.slice.needed:
                e = (
                    x
                    if r.index == 0
                    else K.select(
                        K.cmp("eq", x, K.const(-1, 32)), K.const(-1, 32), K.const(0, 32)
                    )
                )
                stmts.append(self._stmt("let", op, var=r.name, expr=e))
                self.env[r] = {"k": "int", "e": K.var(r.name, 32)}
        stmts.append(
            self._stmt(
                "let",
                op,
                var="__clc_k",
                expr=K.bin_("add", K.var("__clc_k", 32), K.const(1, 32), 32),
            )
        )
        return stmts

    # -- values --------------------------------------------------------------------------
    def _is_int(self, v: Value) -> bool:
        t = self._type(v)
        return t in K.INT_WIDTHS

    def _width(self, v: Value) -> int:
        t = self._type(v)
        if t not in K.INT_WIDTHS:
            shape_elem = K.tensor_type(t)
            if shape_elem and shape_elem[1] in K.INT_WIDTHS:
                return K.INT_WIDTHS[shape_elem[1]]
            raise KirError(f"%{v.name} has non-integer type {t!r}")
        return K.INT_WIDTHS[t]

    def _type(self, v: Value) -> str:
        if v.type:
            return v.type
        op = v.owner
        if isinstance(op, Block):
            return v.type
        v.type = result_type(op, v.index)
        return v.type

    def _int(self, v: Value) -> dict:
        s = self._sym(v)
        if s["k"] != "int":
            raise KirError(f"%{v.name} ({s['k']}) used where an integer is required")
        return s

    def _inline(self, v: Value) -> dict:
        """`v` rebuilt from its definitions down to kernel arguments (no role variables)."""
        saved, self.env = self.env, {}
        try:
            return self._sym(v, inline=True)
        finally:
            self.env = saved

    def _sym(self, v: Value, inline: bool = False) -> dict:
        if v in self.env:
            return self.env[v]
        owner = v.owner
        if v in self.captures:
            s = self._inline(self.captures[v])
        elif isinstance(owner, Block):
            if owner is not self.body:
                raise KirError(f"block argument %{v.name} used before it is bound")
            s = self._kernel_arg(v)
        else:
            s = self._build(owner, v, inline)
            if s["k"] == "int" and s["e"]["e"] != "const" and not inline:
                # Non-constant integers are let-bound where they are defined (_op).
                raise KirError(
                    f"internal: %{v.name} used before its definition {owner.where()}"
                )
        self.env[v] = s
        return s

    def _kernel_arg(self, v: Value) -> dict:
        t = v.type
        if t in K.INT_WIDTHS:
            return {"k": "int", "e": K.var(v.name, K.INT_WIDTHS[t])}
        pointee = K.ptr_pointee(t)
        if pointee is not None:
            self._register_ptr(v.name, pointee)
            return {"k": "ptr", "tensor": v.name, "terms": [], "shape": []}
        if K.tensordesc_type(t) is not None:
            return {"k": "desc", "name": v.name, "dev": None}
        raise KirError(f"kernel argument %{v.name} of type {t} on the kept slice")

    def _register_ptr(self, name: str, pointee: str | None) -> None:
        if name in self.tensors:
            return
        elem = K.elem_bytes(pointee) if pointee is not None else None
        if elem is None:
            raise KirError(
                f"pointer argument %{name} has elements of type {pointee} (not whole bytes)"
            )
        self.tensors[name] = {"name": name, "kind": "global", "elem_bytes": elem}

    def _bar(self, v: Value, op: Op) -> dict:
        s = self._sym(v)
        if s["k"] == "barray":
            if s["num"] != 1:
                raise KirError(
                    f"barrier array %{v.name} used without an index {op.where()}"
                )
            return {"name": s["name"], "idx": K.const(0, 32), "num": 1}
        if s["k"] != "bar":
            raise KirError(
                f"{op.where()}: %{v.name} is not a local mbarrier ({s['k']})"
            )
        return s

    def _build(self, op: Op, v: Value, inline: bool = False) -> dict:
        n = op.name

        def sym(x):
            return self._sym(x, inline)

        addr = v in self.slice.needed and "addr" in self.slice.needed[v]
        if n == "arith.constant":
            return self._constant(op)
        if n in op_table.ARITH_BIN or n == "arith.cmpi":
            a, b = (sym(x) for x in op.operands[-2:])
            if n == "arith.cmpi":
                pred = strip_strings(op.text).split()[1].rstrip(",")
                if pred not in op_table.CMP_PREDS:
                    raise KirError(f"unknown cmpi predicate {pred} {op.where()}")
                return self._lift(
                    a,
                    b,
                    lambda x, y: K.cmp(pred, x, y),
                    lambda x, y: K.t_cmp(pred, x, y),
                )
            o = op_table.ARITH_BIN[n]
            w = self._width(v)
            return self._lift(
                a,
                b,
                lambda x, y: K.bin_(o, x, y, w, addr),
                lambda x, y: K.t_bin(o, x, y, w, addr),
            )
        if n == "arith.select":
            c, a, b = (sym(x) for x in op.operands)
            if c["k"] == "int" and a["k"] == "int":
                return {"k": "int", "e": K.select(c["e"], a["e"], b["e"])}
            ta, tb = self._as_t(a), self._as_t(b)
            tc = c["t"] if c["k"] == "tensor" else K.t_splat(c["e"], ta["shape"])
            return {"k": "tensor", "t": K.t_select(tc, ta, tb)}
        if n in ("arith.extsi", "arith.extui", "arith.trunci", "arith.index_cast"):
            a = sym(op.operands[0])
            w = self._width(v)
            signed = n != "arith.extui"
            if a["k"] == "int":
                e = (
                    K.trunc(a["e"], w, addr)
                    if w < a["e"]["w"]
                    else K.ext(a["e"], w, signed)
                )
                return {"k": "int", "e": e}
            t = a["t"]
            return {
                "k": "tensor",
                "t": K.t_trunc(t, w, addr) if w < t["w"] else K.t_ext(t, w, signed),
            }
        if n == "math.absi":
            a = sym(op.operands[0])
            if a["k"] != "int":
                raise KirError(f"tensor math.absi on the kept slice {op.where()}")
            e, w = a["e"], a["e"]["w"]
            neg = K.cmp("slt", e, K.const(0, w))
            return {
                "k": "int",
                "e": K.select(neg, K.bin_("sub", K.const(0, w), e, w), e),
            }
        if n in ("tt.get_program_id", "tt.get_num_programs"):
            axis = "xyz".index(strip_strings(op.text).split()[1])
            return {
                "k": "int",
                "e": (K.pid if n == "tt.get_program_id" else K.nprog)(axis),
            }
        if n == "tt.make_range":
            m = re.search(r"end = (\d+) : i32, start = (\d+) : i32", op.text)
            if not m:
                raise KirError(f"cannot read the range of {op.where()}")
            return {"k": "tensor", "t": K.t_range(int(m.group(2)), int(m.group(1)))}
        if n in ("tt.splat", "tt.broadcast", "tt.expand_dims", "ttg.convert_layout"):
            return self._shape_op(op, v, sym(op.operands[0]))
        if n == "tt.addptr":
            # Each addptr adds its own sign-extended offset in 64-bit address arithmetic;
            # the terms are kept separate (in their own width) until an event uses them.
            p, off = sym(op.operands[0]), sym(op.operands[1])
            if p["k"] != "ptr" or off["k"] not in ("int", "tensor"):
                raise KirError(f"tt.addptr on a non-argument pointer {op.where()}")
            term = off["e"] if off["k"] == "int" else off["t"]
            return {**p, "terms": p["terms"] + [term]}
        if n == "tt.load":
            return self._metadata_load(op, v, sym)
        if n == "ttg.local_alloc":
            md = K.memdesc_type(self._type(v))
            if not md or md[1] != "i64" or len(md[0]) != 1:
                raise KirError(f"{op.where()} is not an mbarrier array")
            return {"k": "barray", "name": v.name, "num": md[0][0]}
        if n == "ttg.memdesc_index":
            src = sym(op.operands[0])
            if src["k"] != "barray":
                raise KirError(
                    f"memdesc_index of a non-barrier buffer on the kept slice {op.where()}"
                )
            idx = sym(op.operands[1])
            return {"k": "bar", "name": src["name"], "idx": idx["e"], "num": src["num"]}
        if n == "ttng.map_to_remote_buffer":
            raise KirError(
                f"remote (cluster peer) shared-memory barrier is not in KIR v1: {op.where()}"
            )
        if n == "nvg.cluster_id":
            if self.cluster != [1, 1, 1]:
                raise KirError(
                    f"cluster CTA rank with cluster {self.cluster} is not in KIR v1: {op.where()}"
                )
            return {"k": "int", "e": K.const(0, 32)}
        if n == "ttg.global_scratch_alloc":
            return {"k": "scratch", "op": op}
        if n == "ttng.reinterpret_tensor_descriptor":
            rec = self.device_descs.get(id(op.operands[0].owner))
            if rec is None:
                raise KirError(
                    f"{op.where()}: descriptor memory not written by tensormap_create first"
                )
            return {"k": "desc", "name": rec["name"], "dev": rec}
        raise KirError(f"no value rule for {op.where()}")

    def _constant(self, op: Op) -> dict:
        text = strip_strings(op.text)
        body = text[len("arith.constant") :].strip()
        if body in ("true", "false"):
            return {"k": "int", "e": K.const(1 if body == "true" else 0, 1)}
        ty = K.type_section(op.text)
        m = re.match(r"^dense<(-?\d+|true|false)>", body)
        if m:
            shape, elem = K.elem_and_shape(ty)
            if elem not in K.INT_WIDTHS:
                raise KirError(
                    f"non-integer dense constant on the kept slice {op.where()}"
                )
            val = {"true": 1, "false": 0}.get(m.group(1))
            return {
                "k": "tensor",
                "t": K.t_full(
                    int(m.group(1)) if val is None else val, shape, K.INT_WIDTHS[elem]
                ),
            }
        m = re.match(r"^(-?\d+) : (i\d+)$", body)
        if not m or m.group(2) not in K.INT_WIDTHS:
            raise KirError(f"non-integer constant on the kept slice {op.where()}")
        return {"k": "int", "e": K.const(int(m.group(1)), K.INT_WIDTHS[m.group(2)])}

    @staticmethod
    def _as_t(s: dict, shape=None) -> dict:
        if s["k"] == "tensor":
            return s["t"]
        if s["k"] == "int":
            return K.t_splat(s["e"], shape or [])
        raise KirError(f"expected an integer tensor, got {s['k']}")

    def _lift(self, a: dict, b: dict, scalar, tensor) -> dict:
        if a["k"] == "int" and b["k"] == "int":
            return {"k": "int", "e": scalar(a["e"], b["e"])}
        if a["k"] not in ("int", "tensor") or b["k"] not in ("int", "tensor"):
            raise KirError(f"integer op on {a['k']}/{b['k']} values")
        shape = (a if a["k"] == "tensor" else b)["t"]["shape"]
        return {"k": "tensor", "t": tensor(self._as_t(a, shape), self._as_t(b, shape))}

    def _shape_op(self, op: Op, v: Value, a: dict) -> dict:
        n = op.name
        out_shape, _ = K.elem_and_shape(self._type(v))
        if a["k"] == "ptr":
            terms = [
                self._reshape_t(op, t, out_shape) if K.is_texpr(t) else t
                for t in a["terms"]
            ]
            return {**a, "terms": terms, "shape": out_shape}
        if a["k"] == "int":
            if n != "tt.splat":
                raise KirError(f"{op.where()} on a scalar")
            return {"k": "tensor", "t": K.t_splat(a["e"], out_shape)}
        if a["k"] != "tensor":
            raise KirError(f"{op.where()} on a {a['k']} value")
        return {"k": "tensor", "t": self._reshape_t(op, a["t"], out_shape)}

    @staticmethod
    def _reshape_t(op: Op, t: dict, out_shape: list) -> dict:
        n = op.name
        if n == "ttg.convert_layout":
            return t
        if n == "tt.expand_dims":
            m = re.search(r"axis = (\d+)", op.text)
            if not m:
                raise KirError(f"cannot read the axis of {op.where()}")
            return {
                "t": "expand",
                "shape": out_shape,
                "a": t,
                "axis": int(m.group(1)),
                "w": t["w"],
            }
        if n == "tt.broadcast":
            cur: dict[str, Any] = t
            for axis, (src, dst) in enumerate(zip(t["shape"], out_shape)):
                if src != dst:
                    if src != 1:
                        raise KirError(f"invalid broadcast {op.where()}")
                    shape = list(cur["shape"])
                    shape[axis] = dst
                    cur = {
                        "t": "bcast",
                        "shape": shape,
                        "a": cur,
                        "axis": axis,
                        "w": t["w"],
                    }
            return cur
        raise KirError(f"no shape rule for {op.where()}")

    def _metadata_load(self, op: Op, v: Value, sym) -> dict:
        m = op_table.mem_op(op)
        p = sym(m["ptr"])
        if p["k"] != "ptr" or p["shape"]:
            raise KirError(
                f"tensor-valued load on the kept slice (data-dependent addresses or "
                f"control from a tensor of loaded values) {op.where()}"
            )
        w = self._width(v)
        self.metadata.add(p["tensor"])
        index = self._scalar_offset(p)
        if m["mask"] is None:
            return {"k": "int", "e": K.load(p["tensor"], [index], w)}
        # Lanes the mask disables read nothing: their index is replaced by 0, so the load
        # is total (an out-of-range index under a false mask is not an evaluation error).
        mask = sym(m["mask"])["e"]
        other = sym(m["other"])["e"] if m["other"] is not None else K.const(0, w)
        e = K.load(p["tensor"], [K.select(mask, index, K.const(0, 64))], w)
        return {"k": "int", "e": K.select(mask, e, other)}

    # -- slicing roots for declared work-item fields -------------------------------------
    def _work_roots(self) -> list:
        roots = []
        if not self.work_fields:
            return roots
        for op in walk(self.body):
            for r in op.results:
                if (
                    op.loc_name in self.work_fields.values()
                    and result_type(op, r.index) in K.INT_WIDTHS
                ):
                    roots.append((r, f"work field `{op.loc_name}` at {op.where()}"))
        return roots


def result_type(op: Op, k: int) -> str:
    """Printed result type of an op result (custom forms of the op table's value ops)."""
    sec = K.type_section(op.text)
    n = op.name
    if n == "arith.constant":
        return sec or "i1"
    if n == "arith.cmpi":
        shape, _ = K.elem_and_shape(sec)
        return f"tensor<{'x'.join(map(str, shape))}xi1>" if shape else "i1"
    if n == "arith.select":
        return split_top(sec)[-1]
    if n in (
        "arith.extsi",
        "arith.extui",
        "arith.trunci",
        "arith.index_cast",
        "arith.fptosi",
        "arith.sitofp",
        "arith.truncf",
        "arith.extf",
        "arith.bitcast",
    ):
        return K.types_after(sec, " to ")[0]
    if n == "tt.addptr":
        return split_top(sec)[0]
    if n == "tt.load":
        shape, ptr = K.elem_and_shape(sec)
        pointee = (
            K.ptr_pointee(ptr) or ""
        )  # "" (unknown, as below) if ptr is not a !tt.ptr type
        return f"tensor<{'x'.join(map(str, shape))}x{pointee}>" if shape else pointee
    if n in (
        "tt.get_program_id",
        "tt.get_num_programs",
        "nvg.cluster_id",
        "ttng.clc_query_cancel",
    ):
        return "i32"
    arrow = K.types_after(sec, " -> ")
    if arrow:
        return arrow[k] if k < len(arrow) else ""
    types = split_top(sec) if sec else []
    return types[k] if k < len(types) else (types[0] if types else "")


# -- KIR document ---------------------------------------------------------------------------
def build(ttgir: str, dump: dict, key: str, case: dict) -> dict:
    module = parse(ttgir, dump["src_root"])
    target = dump["target"]
    m = re.search(r'ttg\.target = "cuda:(\d+)"', module.attrs)
    if not m or int(m.group(1)) != target["arch"]:
        raise KirError(
            f"TTGIR ttg.target is {m and m.group(0)}, but the dump is for {target['name']}",
            kind="error",
        )
    tr = Translator(module, case, case.get("work", {}))
    roles = tr.roles()
    for role in roles:
        for ev in events_of(role["body"]):
            if not ev["loc"].startswith(REPO_PREFIX):
                raise KirError(
                    f"{ev['ev']} in role {role['name']} has location {ev['loc']!r}, not a line of the "
                    "exported sources (did a compiler switch such as USE_IR_LOC rewrite the locations?)",
                    kind="error",
                )
    if tr.work_fields and not tr.work_events:
        raise KirError(
            f"no block defines all work fields {tr.work_fields}", kind="error"
        )
    # Metadata contents are the values at launch time; a launch that also writes them
    # (directly, or through storage they share, checked per input in _input) breaks that.
    rewritten = tr.metadata & tr.written
    if rewritten:
        raise KirError(
            f"metadata tensor(s) {sorted(rewritten)} are written by {dump['kernel']} itself, "
            "so their contents during the launch are not the recorded ones"
        )
    args = {v.name: v for v in module.args}
    launches = [
        (inp, la)
        for inp in dump["inputs"]
        for la in inp["launches"]
        if la["spec"] == key
    ]
    if not launches:
        raise KirError(f"no input launched spec {key}", kind="error")
    spec = dump["specs"][key]
    snap0 = launches[0][1]["args"]

    tensor_args = {
        n for n, v in args.items() if K.ptr_pointee(v.type) or K.tensordesc_type(v.type)
    }
    annotations = _annotations(case, [inp["name"] for inp, _ in launches], key)
    for key_ in ("tensors", "intended_writes"):
        named = set(case.get(key_, {})).union(
            *(i.get(key_, {}).keys() for i in case["inputs"])
        )
        if named - tensor_args:
            raise KirError(
                f"case `{key_}` names {sorted(named - tensor_args)} that are not "
                f"pointer/descriptor arguments of {dump['kernel']}",
                kind="error",
            )
    # Segment offsets (cu_seqlens) are metadata of the specs even where the kernel itself
    # never reads them (e.g. output offsets recomputed in the kernel).
    for ann in annotations.values():
        seg = ann.get("segments")
        if seg:
            tr._register_ptr(seg["offsets"], K.ptr_pointee(args[seg["offsets"]].type))
            tr.metadata.add(seg["offsets"])

    params = [
        {"name": n, "type": v.type} for n, v in args.items() if v.type in K.INT_WIDTHS
    ]
    tensors = []
    for name, t in tr.tensors.items():
        if name in args and t["kind"] == "desc":  # host descriptor
            desc_type = K.tensordesc_type(args[name].type)
            if desc_type is None:
                raise KirError(
                    f"host descriptor {name} has type {args[name].type}, not a !tt.tensordesc"
                )
            block, elem, _ = desc_type
            for inp, launch in launches:
                rec = launch["args"][name]
                if rec["block_shape"] != block or 8 * rec["elem_bytes"] != K.elem_bits(
                    elem
                ):
                    raise KirError(
                        f"input {inp['name']}: host descriptor {name} (block {rec['block_shape']}, "
                        f"{rec['elem_bytes']}-byte elements) does not match its type {args[name].type}"
                    )
            t.update(
                {
                    "elem_bytes": snap0[name]["elem_bytes"],
                    "shape": [
                        K.var(f"{name}.shape.{i}", 32) for i in range(len(block))
                    ],
                    "strides": [
                        K.var(f"{name}.stride.{i}", 64) for i in range(len(block))
                    ],
                    "box": block,
                }
            )
        elif (
            name in args
        ):  # pointer argument (in-kernel descriptors are complete already)
            if name in tr.metadata and len(snap0[name]["shape"]) != 1:
                raise KirError(
                    f"metadata tensor {name} must be 1-D (shape {snap0[name]['shape']})"
                )
            rank = len(snap0[name]["shape"])
            t["shape"] = [K.var(f"{name}.shape.{i}", 64) for i in range(rank)]
            t["strides"] = [K.var(f"{name}.stride.{i}", 64) for i in range(rank)]
            params += [
                {"name": f"{name}.{f}.{i}", "type": "i64"}
                for i in range(rank)
                for f in ("shape", "stride")
            ]
        ann = annotations.get(name, {})
        t["role"] = ann.get("role", "output" if name in tr.written else "input")
        t["segments"] = ann.get("segments")
        t["metadata"] = name in tr.metadata
        tensors.append(t)
    if tr.uses_clc:
        tensors.append(
            {
                "name": "__clc_response",
                "kind": "desc",
                "elem_bytes": 1,
                "shape": [K.const(16, 32)],
                "strides": [K.const(1, 64)],
                "box": [16],
                "segments": None,
                "role": "input",
                "metadata": False,
            }
        )
        for name, rank in (("__clc_schedule", 2), ("__clc_launched", 1)):
            tensors.append(
                {
                    "name": name,
                    "kind": "global",
                    "elem_bytes": 4,
                    "shape": [K.var(f"{name}.shape.{i}", 64) for i in range(rank)],
                    "strides": [K.var(f"{name}.stride.{i}", 64) for i in range(rank)],
                    "segments": None,
                    "role": "input",
                    "metadata": True,
                }
            )
            params += [
                {"name": f"{name}.{f}.{i}", "type": "i64"}
                for i in range(rank)
                for f in ("shape", "stride")
            ]

    inputs = []
    for inp, launch in launches:
        same = [la for la in inp["launches"] if la["spec"] == key]
        name = inp["name"] if len(same) == 1 else f"{inp['name']}#{same.index(launch)}"
        base = _input(
            name, inp, launch, params, tensors, spec, case, tr.written, tr.desc_bases
        )
        if tr.uses_clc:
            resident = dump["num_sms"] * max(
                1, target["max_shared_mem"] // max(1, spec["shared_bytes"])
            )
            inputs += [
                _with_clc(base, sched, resident)
                for sched in case.get("clc_schedules", CLC_SCHEDULES)
            ]
        else:
            inputs.append(base)

    barriers = [
        {"name": b, "count": K.const(c["count"], 32), "num": K.const(c["num"], 32)}
        for b, c in tr.barriers.items()
    ]
    elision = case.get("elision_flags", [])
    unknown = sorted(set(elision) - set(inputs[0]["flags"]))
    if unknown:
        raise KirError(
            f"case `elision_flags` names {unknown}, which are not boolean constexprs of "
            f"{dump['kernel']} (flags {sorted(inputs[0]['flags'])})",
            kind="error",
        )
    return {
        "kir_version": 1,
        "kernel": dump["kernel"],
        "source": case.get("source", ""),
        "ir": "ttgir",
        "target": target["name"],
        "compiler": {
            "name": dump["compiler"]["name"],
            "version": dump["compiler"]["version"],
        },
        # Every input carries its own grid (it overrides these); the first one is shown.
        "grid": [K.const(g, 32) for g in inputs[0]["grid"]],
        "params": params,
        "tensors": tensors,
        "barriers": barriers,
        "roles": roles,
        "case": {
            "name": f"{dump['case']}/{key}",
            "inputs": inputs,
            **({"elision_flags": elision} if elision else {}),
        },
    }


def _annotations(case: dict, input_names: list, key: str) -> dict:
    """Tensor annotations (case-level, overridden per input); one KIR needs one set."""
    by_name = {i["name"]: i for i in case["inputs"]}
    sets = [
        {**case.get("tensors", {}), **by_name[n].get("tensors", {})}
        for n in input_names
    ]
    if any(s != sets[0] for s in sets):
        raise KirError(
            f"inputs {input_names} of spec {key} disagree on tensor annotations",
            kind="error",
        )
    return sets[0]


def _input(
    name: str,
    inp: dict,
    launch: dict,
    params: list,
    tensors: list,
    spec: dict,
    case: dict,
    written: set,
    desc_bases: dict,
) -> dict:
    snap = launch["args"]
    values = {}
    for p in params:
        pname = p["name"]
        if pname.startswith("__clc_"):
            continue
        if pname in snap:
            values[pname] = int(snap[pname]["value"])
            continue
        base, field_, idx = pname.rsplit(".", 2)
        values[pname] = int(
            snap[base]["shape" if field_ == "shape" else "strides"][int(idx)]
        )
    # Every launch is skipped: host code that read data an earlier launch may write may
    # have computed this launch (its grid, arguments, whether it happens) from stale
    # CPU contents.
    for r in inp["stale_reads"]:
        if r["after"] <= launch["index"]:
            w = r["writers"][0]
            raise KirError(
                f"input {name}: before this launch the host reads data ({r['op']} at {r['loc']}) that "
                f"launch {w['launch']} of {w['kernel']} (argument {w['param']}) may write; the CPU run "
                "skipped that launch, so the host may branch or size this launch differently"
            )
    metadata, shapes = {}, {}
    for t in tensors:
        if (
            t["kind"] == "desc" and t["name"] in desc_bases
        ):  # in-kernel descriptor with host-side extents
            base = desc_bases[t["name"]]
            dims = [_host_value(e, values, name) for e in t["shape"] + t["strides"]]
            rank = len(t["shape"])
            _check_extents(name, t["name"], dims[:rank], dims[rank:], base, snap[base])
        rec = snap.get(t["name"])
        if rec is None:  # created in the kernel, or CLC schedule (attached later)
            continue
        shapes[t["name"]] = rec["shape"]
        if rec["kind"] == "desc":
            _check_extents(
                name,
                t["name"],
                rec["shape"],
                rec["strides"],
                "its base tensor",
                rec["base"],
            )
        if t["metadata"]:
            if "values" not in rec:
                raise KirError(
                    f"input {name}: metadata tensor {t['name']} contents were not recorded"
                )
            if rec["storage_offset"] != 0:
                raise KirError(f"input {name}: metadata tensor {t['name']} is a view")
            if rec["shape"][0] > 1 and rec["strides"] != [1]:
                raise KirError(
                    f"input {name}: metadata tensor {t['name']} has strides {rec['strides']}: its "
                    "contents are recorded in index order, but `load` indices are element offsets"
                )
            stale = launch["stale"].get(t["name"])
            if stale:
                w = stale[0]
                raise KirError(
                    f"input {name}: metadata tensor {t['name']} may hold data that launch {w['launch']} "
                    f"of {w['kernel']} (argument {w['param']}) writes before this launch (directly or "
                    "through host computation); the CPU run skipped it, so the recorded contents are "
                    "not what the kernel would read"
                )
            lo, hi = rec["storage"]
            for w in sorted(written):
                wlo, whi = (snap[w]["base"] if snap[w]["kind"] == "desc" else snap[w])[
                    "storage"
                ]
                if lo < whi and wlo < hi:
                    raise KirError(
                        f"input {name}: metadata tensor {t['name']} shares storage with {w}, "
                        "which the analyzed kernel writes"
                    )
            metadata[t["name"]] = rec["values"]
    per_input = next((i for i in case["inputs"] if i["name"] == inp["name"]), {})
    return {
        "name": name,
        "params": values,
        "grid": launch["grid"],
        "metadata": metadata,
        # Boolean constexprs are the host-selected flags; integer constexprs (block sizes,
        # window sizes) are compiled into the KIR itself.
        "flags": {k: v for k, v in spec["constexprs"].items() if isinstance(v, bool)},
        "shapes": shapes,
        "intended_writes": per_input.get(
            "intended_writes", case.get("intended_writes", {})
        ),
    }


def _host_value(e: dict, values: dict, input_name: str) -> int:
    try:
        return K.evaluate(e, values)
    except ValueError as err:
        raise KirError(f"input {input_name}: {err}") from err


# Descriptor rows (outer index combinations) checked one by one at most.
MAX_CHECKED_ROWS = 1 << 20


def _check_extents(
    input_name: str, desc: str, shape: list, strides: list, base_name: str, base: dict
) -> None:
    """Every in-extent element of a TMA descriptor (element offsets sum c_i * strides_i from
    its global address, the base tensor's data pointer) is an element of the base tensor's
    allocation (kir.md: its elements when the strides do not overlap, else its offset
    span); else a KirError. Layer 2 checks TMA accesses against the descriptor's extents
    (TMA clips there), so extents beyond the allocation would hide out-of-bounds accesses."""
    bad = _outside(shape, strides, base["shape"], base["strides"])
    if bad is not None:
        raise KirError(
            f"input {input_name}: TMA descriptor {desc} (shape {shape}, strides {strides}) has in-extent "
            f"elements outside {base_name} (shape {base['shape']}, strides {base['strides']}), e.g. "
            f"element offset {bad}"
        )


def _outside(shape: list, strides: list, bshape: list, bstrides: list) -> int | None:
    """An element offset of the box {sum c_i s_i : 0 <= c_i < n_i} outside the allocation
    of a tensor (bshape, bstrides), or None. The box is a set of runs (its dims of stride
    1, n_0, n_0 n_1, ... merged), each checked against the run of the allocation it starts in."""
    dims = sorted((s, n) for n, s in zip(shape, strides) if n != 1)
    if any(n <= 0 for _, n in dims):
        return None  # no element
    if any(n <= 0 for n in bshape):
        return 0
    lo = sum(min(0, (n - 1) * s) for s, n in dims)
    hi = sum(max(0, (n - 1) * s) for s, n in dims)
    bdims = sorted((s, n) for n, s in zip(bshape, bstrides) if n != 1)
    span = 0
    for s, n in bdims:
        if s <= span:  # overlapping strides: the allocation is the offset span
            blo = sum(min(0, (m - 1) * t) for t, m in bdims)
            bhi = sum(max(0, (m - 1) * t) for t, m in bdims)
            return lo if lo < blo else hi if hi > bhi else None
        span += (n - 1) * s
    run, k = (
        1,
        0,
    )  # length of the allocation's contiguous runs, and its dims beyond them
    while k < len(bdims) and bdims[k][0] == run:
        run *= bdims[k][1]
        k += 1
    outer_b = bdims[k:][::-1]
    length, j = 1, 0
    while j < len(dims) and dims[j][0] == length:
        length *= dims[j][1]
        j += 1
    outer = dims[j:]
    rows = math.prod(n for _, n in outer)
    if rows > MAX_CHECKED_ROWS:
        raise KirError(
            f"TMA descriptor with {rows} rows: too many to check against its base tensor"
        )
    for c in itertools.product(*(range(n) for _, n in outer)):
        a = sum(ci * s for ci, (s, _) in zip(c, outer))
        r = a
        for s, n in outer_b:
            d = r // s
            if not 0 <= d < n:
                return a
            r -= d * s
        if not 0 <= r < run:
            return a
        if r + length > run:
            return a - r + run
    return None


def _with_clc(inp: dict, schedule: str, resident: int) -> dict:
    """Attach a Cluster Launch Control schedule (which CTAs launch, what each query returns).

    Up to `resident` CTAs start at once (SMs x CTAs per SM by shared memory); the others
    are pending and may be cancelled by a running CTA's try_cancel and processed by it:
      none        every try_cancel fails; every CTA launches and runs its own tile
      round_robin pending CTAs are stolen in order, cycling over the resident CTAs
      greedy      CTA 0 steals every pending CTA
    """
    if inp["grid"][1] * inp["grid"][2] != 1:
        raise KirError(
            f"input {inp['name']}: CLC with a multi-dimensional grid {inp['grid']}"
        )
    if resident < 1:
        raise KirError(
            f"input {inp['name']}: CLC needs at least one resident CTA, got {resident}"
            " (check the case's num_sms)",
            kind="error",
        )
    g = inp["grid"][0]
    resident = min(g, resident)
    steals = [[] for _ in range(g)]
    launched = [1] * g
    if schedule != "none":
        for t in range(resident, g):
            thief = 0 if schedule == "greedy" else (t - resident) % resident
            steals[thief].append(t)
            launched[t] = 0
    k = 1 + max((len(s) for s in steals), default=0)
    table = [s + [-1] * (k - len(s)) for s in steals]
    out: dict[str, Any] = {**inp, "name": f"{inp['name']}@clc:{schedule}"}
    out["metadata"] = dict(
        inp["metadata"],
        __clc_schedule=[x for row in table for x in row],
        __clc_launched=launched,
    )
    out["params"] = dict(
        inp["params"],
        **{
            "__clc_schedule.shape.0": g,
            "__clc_schedule.shape.1": k,
            "__clc_schedule.stride.0": k,
            "__clc_schedule.stride.1": 1,
            "__clc_launched.shape.0": g,
            "__clc_launched.stride.0": 1,
        },
    )
    out["shapes"] = dict(inp["shapes"], __clc_schedule=[g, k], __clc_launched=[g])
    return out


def translate(dump_path: Path, case: dict) -> dict:
    """KIR (or the error that stopped extraction) for every specialization of a dump.

    Writes <spec>.kir.json next to dump.json for each specialization that translates.
    """
    dump = json.loads(dump_path.read_text())
    results: dict[str, dict | Exception] = {}
    for key, spec in dump["specs"].items():
        ttgir = (dump_path.parent / spec["ttgir"]).read_text()
        try:
            kir = build(ttgir, dump, key, case)
            violations = kir_contract.validate(kir)
            if violations:
                raise KirError(
                    f"KIR contract (frontend/kir.py): {'; '.join(violations[:5])}",
                    kind="error",
                )
        except (KirError, SliceError, OpError, ParseError) as e:
            results[key] = e
            continue
        (dump_path.parent / f"{key}.kir.json").write_text(json.dumps(kir))
        results[key] = kir
    return results


def events_of(stmts: list):
    """The events of a statement list, nested ones included, in program order."""
    for s in stmts:
        if s["op"] == "event":
            yield s
        for key in ("body", "then", "else"):
            yield from events_of(s.get(key, []))


def summary(kir: dict) -> str:
    counts = {r["name"]: sum(1 for _ in events_of(r["body"])) for r in kir["roles"]}
    roles = ", ".join(
        f"{r['name']}:{r['warps']}w/{counts[r['name']]}ev" for r in kir["roles"]
    )
    return (
        f"roles=[{roles}] static_events={sum(counts.values())} "
        f"tensors={len(kir['tensors'])} barriers={len(kir['barriers'])} "
        f"inputs={len(kir['case']['inputs'])}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", required=True, type=Path)
    ap.add_argument("--case", required=True, type=Path)
    args = ap.parse_args()
    results = translate(args.dump, json.loads(args.case.read_text()))
    for key, r in results.items():
        if isinstance(r, Exception):
            print(f"[to_kir] {key}: FAILED ({failure_kind(r)}): {r}", file=sys.stderr)
        else:
            print(f"[to_kir] {key}: {key}.kir.json ({summary(r)})", file=sys.stderr)
    sys.exit(1 if any(isinstance(r, Exception) for r in results.values()) else 0)


if __name__ == "__main__":
    main()
