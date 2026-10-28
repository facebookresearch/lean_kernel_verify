# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The CuTe op table, P1 slice (P1-1 TMEM reuse ordered; references/ir_ops_cute.md,
"P1 slice"). Used only by the second translation pass of to_kir.py, which writes a
separate `.p1.json` KIR; the P0 pass and its KIR do not change.

In the P1 pass these ops are events (everything else is as in ops.py / ops_mem.py):

* `tmem_load` / `tmem_store` (32x32b.xN only): one `tmem` ld/st per issuing warp, every
  lane of the warp active, address warp-uniform; lanes [L, L+32) x cols [C, C+N).
* `copy_s2t` (32x128b.warpx4 only): one `tmem` cp; lanes [L, L+128) x cols [C, C+4).
* `umma` / `umma_block_scaled` (f32 accumulator; shape from the atom type, cross-checked
  against the known bits of the instruction descriptor, or from those bits alone when no
  atom feeds the op): one `tmem` mma with the D rectangle (write), the A rectangle (ts
  form) and two 4-column SF rectangles (block scaled). cp and mma need a one-warp role
  (the landing rule needs the issuing thread).
* `nvvm.tcgen05.wait`, inline `tcgen05.wait::ld/st`: `tmem_wait`.
* `nvvm.barrier` (no operands), inline `bar.sync/arrive`, `barrier.sync/arrive`:
  `named_bar` (constant id and count, whole role).
* `cute.inttoptr` of the constant 0 to a TMEM pointer (the fake-base idiom): an absolute
  TMEM pointer, and `cute.ptrtoint` of one its absolute address (an integer, so that
  inline PTX can take it); `retrieve_tmem_ptr`: a pointer relative to the allocation;
  `alloc_tmem`: the column count (exactly one, a constant).
* cta_group::2 (a cluster's CTA pair, references/frontend_cute.md "Clusters"): an MMA
  (M = 256, or M = 128 with N / 2 columns per CTA) or copy accesses the same rectangles
  in the TMEM of both CTAs of the pair (the event's `cta_mask`).
* Inline PTX `tcgen05.cp` (shapes 128x256b, 128x128b, 32x128b.warpx4, 64x128b.warpx2) and
  `tcgen05.mma.cta_group::N.kind::f16/tf32` (TMEM addresses from operands, the instruction
  descriptor a constant, here or in a function-scope register another inline PTX sets
  once): `tmem` cp / mma events, issued under an elect.sync predicate or by the single
  active thread; a blocking wait loop between the MMAs is a `wait` in its place.

Refusals (Unsupported, the P1 KIR is "not checked"; P0 is unaffected), tagged R1-R9 as in
references/frontend_cute.md: R1 a TMEM pointer of unknown provenance; R2 not exactly one
constant alloc_tmem outside loops; R3 absolute pointers with an allocation under 512
columns, or absolute and relative pointers mixed; R4 tcgen05 data, MMA or allocation
statements in inline PTX outside the table; R5 shapes and attributes outside the table;
R6 a lane-varying ld/st address, a partial or runtime lane mask, cp/mma in a role of
several warps or issued by several threads; R7 a
named barrier with a non-constant id or count, or reached by part of a role, or an mbarrier
wait/arrive/commit under a runtime lane condition in a role of several warps
(interp.Translator._p1_split_by_issuers, which splits a role whose warps issue one only in
part); R8 a kept read of an atom field that is data (interp.join_atoms).
"""

import re

from . import expr as E, interp, ops, ops_mem
from .interp import Data, Ptr
from .ops import otype, ptr_info, rtype

TMEM_HANDLERS: dict = {}
TMEM_EFFECTS: dict = {}


def spec_p1(op):
    return TMEM_HANDLERS.get(op.name) or ops.spec(op)


def effects_p1() -> dict:
    return {**ops.EFFECTS, **TMEM_EFFECTS}


def handler(*names):
    def deco(fn):
        for n in names:
            TMEM_HANDLERS[n] = fn
        return fn

    return deco


def effect(*names):
    def deco(fn):
        for n in names:
            TMEM_EFFECTS[n] = fn
        return fn

    return deco


# ---------------------------------------------------------------- addresses


def tmem_addr(tr: interp.Translator, op, p, what: str, exact: bool = True):
    """(lane, column) i32 expressions of TMEM pointer p: the 32-bit-column address is
    `off * bits / 32` (exact for data, ld/st/cp, D and A operands; floored for scale-factor
    pointers, whose byte offset selects the SF within the column); lane = addr >> 16,
    column = addr & 0xffff."""
    if isinstance(p, Data):
        tr.fail(
            op,
            f"R1: {what}: data value from {p.origin} used as a TMEM pointer (slice soundness)",
        )
    if not isinstance(p, Ptr) or p.space != "tmem":
        tr.fail(op, f"R1: {what}: not a TMEM pointer of known provenance")
    tr.tmem_bases.add(p.base)
    off = E.resize(p.off, 64)
    if p.bits == 32:
        a = off
    elif p.bits < 32 and 32 % p.bits == 0:
        k = E.const(32 // p.bits, 64)
        if exact:
            tr.require(
                op,
                E.cmp("eq", E.bin("srem", off, k), E.const(0, 64)),
                f"{what}: TMEM address inside a 32-bit column",
            )
        a = E.bin("sdiv", off, k)
    elif p.bits > 32 and p.bits % 32 == 0:
        a = E.bin("mul", off, E.const(p.bits // 32, 64))
    else:
        tr.fail(op, f"R5: {what}: {p.bits}-bit TMEM elements")
    lane = E.trunc(E.bin("lshr", a, E.const(16, 64)), 32)
    col = E.trunc(E.bin("and", a, E.const(0xFFFF, 64)), 32)
    return lane, col


def site_of(tr, loc: str) -> str:
    """The closest preceding source location in another file (MMAs are located in the
    DSL helper; the site names the kernel line that reached it)."""
    f = loc.rsplit(":", 1)[0]
    for x in reversed(tr.loc_hist):
        if x.rsplit(":", 1)[0] != f:
            return x
    return ""


def _rect(use, write, lane, nlanes, col, ncols, j):
    return {
        "use": use,
        "write": write,
        "lane": j(lane),
        "nlanes": nlanes,
        "col": j(col),
        "ncols": ncols,
    }


# ------------------------------------------------------------- issue patterns


def per_warp_event(tr, op, operands: list, build, what: str) -> None:
    """One event per issuing warp: every lane of an issuing warp active, operands uniform
    within each warp. build(values, warp_expr, to_json) -> KIR statement."""
    uni, var = tr.split_mask()
    act = tr.active(var)
    if act is None:
        tr.fail(op, f"R6: {what} under a lane condition that depends on runtime values")
    if not act:
        return
    warps = sorted({i // 32 for i in act})
    if any(sum(1 for i in act if i // 32 == w) != 32 for w in warps):
        tr.fail(op, f"R6: {what} with only part of a warp active")
    for e in operands:
        if tr.per_warp(e) is None:
            tr.fail(op, f"R6: {what}: an operand varies across the lanes of a warp")
    loc = tr.loc(op)
    if len(warps) == 1 or len(warps) != tr.nw:
        for w in warps:
            m = {
                "%warp": E.const(tr.w0 + w, 32),
                "%lane": E.const(0, 32),
                "%elect": E.const(1, 1),
            }
            vals = [E.subst(e, m) for e in operands]
            tr._emit_under(op, uni, build(vals, m["%warp"], E.to_json))
        return
    t = tr.fresh("wq")
    m = {
        "%warp": E.bin("add", E.var(t, 32), E.const(tr.w0, 32)),
        "%lane": E.const(0, 32),
        "%elect": E.const(1, 1),
    }
    cache: dict = {}
    vals = [E.subst(e, m, cache) for e in operands]
    names = interp._shared_nodes(vals, t)
    body = [
        {"op": "let", "var": n, "expr": E.to_json(e, names, n), "loc": loc}
        for e, n in names.items()
    ]
    body.append(build(vals, m["%warp"], lambda e: E.to_json(e, names)))
    tr._emit_under(
        op,
        uni,
        {
            "op": "for",
            "var": t,
            "lb": E.to_json(E.const(0, 32)),
            "ub": E.to_json(E.const(tr.nw, 32)),
            "step": E.to_json(E.const(1, 32)),
            "body": body,
            "loc": loc,
        },
    )


def one_warp_event(
    tr, op, operands: list, build, what: str, elected: bool = True
) -> None:
    """A cp/mma: issued once by the role's only warp, operands uniform: by one lane its
    lowering elects, every lane of the warp active (`elected`), or by the single active
    thread (inline PTX without elect.sync)."""
    if tr.nw != 1:
        tr.fail(
            op,
            f"R6: {what} in a role of {tr.nw} warps (the landing rule needs the issuing thread)",
        )
    uni, var = tr.split_mask()
    act = tr.active(var)
    if act is None:
        tr.fail(op, f"R6: {what} under a lane condition that depends on runtime values")
    if not act:
        return
    if elected and len(act) != 32:
        tr.fail(op, f"R6: {what} with only part of the warp active")
    if not elected and len(act) != 1:
        tr.fail(
            op,
            f"R6: {what} issued by {len(act)} threads (inline PTX without elect.sync)",
        )
    vals = []
    for e in operands:
        u = tr.uniform(e)
        if u is None:
            tr.fail(op, f"R6: {what}: an operand varies across the lanes of the warp")
        vals.append(u)
    tr._emit_under(op, uni, build(vals, E.const(tr.w0, 32), E.to_json))


def role_event(tr, op, stmt: dict, what: str) -> None:
    """An event of the whole role (tcgen05.wait, named barriers): every thread of the role
    executes it, under thread-uniform conditions only."""
    uni, var = tr.split_mask()
    act = tr.active(var)
    if act is None or (act and len(act) != 32 * tr.nw):
        tr.fail(
            op,
            f"R7: {what} reached by only part of the role (or under a runtime lane condition)",
        )
    if act:
        tr._emit_under(op, uni, stmt)


# ---------------------------------------------------------------- ld / st / cp


def _prop_int(op, name):
    m = re.match(r"\s*(-?\d+)", op.props.get(name, op.attrs.get(name, "")))
    return int(m.group(1)) if m else None


@handler(
    "cute_nvgpu.arch.copy.SM100.tmem_load", "cute_nvgpu.arch.copy.SM100.tmem_store"
)
def _tmem_ld_st(tr, op):
    inst = "ld" if op.name.endswith("tmem_load") else "st"
    dp, b, rep = (_prop_int(op, k) for k in ("num_dp", "num_b", "num_rep"))
    extra = set(op.props) - {"num_dp", "num_b", "num_rep"}
    if dp != 32 or b != 32 or rep is None or not 1 <= rep <= 128 or extra:
        tr.fail(
            op,
            f"R5: tcgen05.{inst} shape num_dp={dp} num_b={b} num_rep={rep} {sorted(extra)} "
            "(only 32x32b.xN is in the table)",
        )
    lane, col = tmem_addr(tr, op, tr.val(op, 0), f"tcgen05.{inst}")
    loc, site = tr.loc(op), ""
    per_warp_event(
        tr,
        op,
        [lane, col],
        lambda v, w, j: {
            "op": "event",
            "ev": "tmem",
            "inst": inst,
            "warp": j(w),
            "rects": [_rect("data", inst == "st", v[0], 32, v[1], rep, j)],
            "site": site,
            "loc": loc,
        },
        f"tcgen05.{inst}",
    )
    return [tr.data(op)] * len(op.results)


@effect("cute_nvgpu.arch.copy.SM100.tmem_load", "cute_nvgpu.arch.copy.SM100.tmem_store")
def _ld_st_kept(tr, op):
    return [0]


@handler("cute_nvgpu.arch.copy.SM100.copy_s2t")
def _copy_s2t(tr, op):
    p = op.props
    ok = (
        _prop_int(op, "bits") == 128
        and _prop_int(op, "dp") == 32
        and _prop_int(op, "cta") == 1
        and "x4" in p.get("broadcast", "")
        and set(p) <= {"bits", "dp", "cta", "broadcast"}
    )
    if not ok:
        tr.fail(
            op,
            f"R5: tcgen05.cp shape {p} (only 32x128b.warpx4, cta_group::1 is in the table)",
        )
    lane, col = tmem_addr(tr, op, tr.val(op, 1), "tcgen05.cp")
    loc = tr.loc(op)
    one_warp_event(
        tr,
        op,
        [lane, col],
        lambda v, w, j: {
            "op": "event",
            "ev": "tmem",
            "inst": "cp",
            "warp": j(w),
            "rects": [_rect("data", True, v[0], 128, v[1], 4, j)],
            "site": "",
            "loc": loc,
        },
        "tcgen05.cp",
    )
    return []


@effect("cute_nvgpu.arch.copy.SM100.copy_s2t")
def _s2t_kept(tr, op):
    return [1]


# ---------------------------------------------------------------------- MMA

_MMA_TYPE = re.compile(
    r"!cute_nvgpu\.sm100\.(mma|mma_bs)<(\d+)x(\d+)x(\d+), num_cta = (\d+),.*?"
    r"elem_type = \(([^)]*)\).*?frag_kind = (\w+)"
)


def _defs(tr) -> dict:
    d = getattr(tr.k, "_p1_defs", None)
    if d is None:
        d = {}
        for o in interp.mlir_text.walk(tr.k.module.top):
            for r in o.results:
                d[r] = o
        tr.k._p1_defs = d
    return d


def _mma_atom(tr: interp.Translator, op, bs: bool):
    """(M, N, K, a bits, acc type, frag) from the MMA atom type of the get_value feeding
    the SF (block scaled) or accumulate operand."""
    d = _defs(tr).get(op.operands[3]) if bs else None
    if d is None or d.name != "cute_nvgpu.atom.get_value":
        return _mma_shape_from_idesc(tr, op, bs)
    t = tr.k.module.resolve_type(d.operand_types[0])
    m = _MMA_TYPE.search(t)
    if m is None or (m.group(1) == "mma_bs") != bs:
        tr.fail(op, f"R5: MMA atom type {t[:160]} is not in the table")
    kind, M, N, K, ncta, elems, frag = m.groups()
    et = [x.strip() for x in elems.split(",")]
    return int(M), int(N), int(K), ops.elem_bits(et[0]), et[2], frag, int(ncta)


def _mma_shape_from_idesc(tr, op, bs: bool):
    """The shape of an MMA whose atom is not in its operands (a plain `umma`, or an atom
    the kernel makes and sets per call): M and N from the instruction descriptor (every bit
    of the shape fields known), the A type from the op; block scaled: K from the op, an f32
    accumulator; dense: 256 bits of K per MMA, the D format at bits 4-5 (PTX ISA table 52)."""
    k, v = _known_bits(tr, op.operands[_IDESC[bs]])
    n_f = _field(k, v, 17, 6)
    m_f = _field(k, v, 27, 2) if bs else _field(k, v, 24, 5)
    at = op.props.get("a_type", "").strip()
    abits = ops.elem_bits(at) if at else None
    kk = _prop_int(op, "shape_k") if bs else (256 // abits if abits else None)
    acc = "f32" if bs else {0: "f16", 1: "f32", 2: "i32"}.get(_field(k, v, 4, 2))
    if n_f is None or m_f is None or kk is None or acc is None:
        tr.fail(op, "R5: the MMA atom is not found from its operands, and the instruction descriptor "
                    "does not give its shape")  # fmt: skip
    t0 = otype(tr, op, 0)
    frag = "ts" if t0.startswith("!cute.ptr<") and "tmem" in t0 else "ss"
    m = m_f << 7 if bs else m_f << 4
    return m, n_f << 3, kk, abits, acc, frag, _prop_int(op, "num_cta")


# Operand index of the instruction descriptor: umma (a, b, d, idesc, accumulate),
# umma_block_scaled (a, b, d, sfa, sfb, idesc, accumulate).
_IDESC = {False: 3, True: 5}


def _known_bits(tr, name: str, depth: int = 0):
    """(known mask, value) of an i32 by a syntactic walk of its definition: constants, or,
    and, shl by a constant, zero extension; anything else is unknown."""
    full = 0xFFFFFFFF
    d = _defs(tr).get(name)
    if d is None or depth > 32:
        return 0, 0
    if d.name in ("arith.constant", "llvm.mlir.constant"):
        m = re.match(r"\s*(-?\d+)", d.props.get("value", ""))
        return (full, int(m.group(1)) & full) if m else (0, 0)
    if d.name in ("arith.ori", "arith.andi"):
        (ka, va), (kb, vb) = (_known_bits(tr, x, depth + 1) for x in d.operands[:2])
        one_a, one_b = ka & va, kb & vb
        zero_a, zero_b = ka & ~va & full, kb & ~vb & full
        if d.name == "arith.ori":
            one, zero = one_a | one_b, zero_a & zero_b
        else:
            one, zero = one_a & one_b, zero_a | zero_b
        return one | zero, one
    if d.name == "arith.shli":
        ks, vs = _known_bits(tr, d.operands[1], depth + 1)
        if ks != full or vs >= 32:
            return 0, 0
        ka, va = _known_bits(tr, d.operands[0], depth + 1)
        return ((ka << vs) | ((1 << vs) - 1)) & full, (va << vs) & full
    if d.name == "arith.extui":
        w = interp.int_width(d.operand_types[0])
        if w and w < 32:
            ka, va = _known_bits(tr, d.operands[0], depth + 1)
            low = (1 << w) - 1
            return (full & ~low) | (ka & low), va & low
    return 0, 0


def _field(k, v, lo, n):
    """Bits [lo, lo + n) of (known mask, value), or None if not all known."""
    mask = ((1 << n) - 1) << lo
    return (v >> lo) & ((1 << n) - 1) if k & mask == mask else None


@handler(
    "cute_nvgpu.arch.mma.SM100.umma", "cute_nvgpu.arch.mma.SM100.umma_block_scaled"
)
def _umma(tr, op):
    bs = op.name.endswith("block_scaled")
    M, N, K, abits, acc, frag, ncta = _mma_atom(tr, op, bs)
    if (
        acc != "f32"
        or ncta != _prop_int(op, "num_cta")
        or (M, ncta) not in ((128, 1), (128, 2), (256, 2))
    ):
        tr.fail(
            op,
            f"R5: tcgen05.mma {M}x{N}x{K} with a {acc} accumulator, cta_group::{ncta} "
            "(only M = 128 with cta_group::1, M = 128 or 256 with cta_group::2, f32)",
        )
    if bs and _prop_int(op, "shape_k") not in (None, K):
        tr.fail(
            op,
            f"R5: shape_k {_prop_int(op, 'shape_k')} differs from the atom's K = {K}",
        )
    vec = _prop_int(op, "vec_size")
    if bs and (N > 256 or not vec or K % vec or K // vec > 4):
        tr.fail(
            op,
            f"R5: block-scaled tcgen05.mma 128x{N}x{K} with vec_size {vec} (the SF footprint "
            "rows cover N <= 256 and at most 4 scale factors per row and MMA)",
        )
    # Instruction descriptor: N >> 3 at bits 17-22; M >> 4 at bits 24-28 (kind f16, tf32,
    # f8f6f4, i8) or M >> 7 at bits 27-28 (block-scaled kinds); PTX ISA tables 51-53.
    k, v = _known_bits(tr, op.operands[_IDESC[bs]])
    n_f = _field(k, v, 17, 6)
    m_f = _field(k, v, 27, 2) if bs else _field(k, v, 24, 5)
    if n_f != N >> 3 or m_f != (M >> 7 if bs else M >> 4):
        tr.fail(
            op,
            f"R5: instruction descriptor shape (N>>3 = {n_f}, M field {m_f}) not known to equal the atom's "
            f"{M}x{N}",
        )
    lane_d, col_d = tmem_addr(tr, op, tr.val(op, 2), "tcgen05.mma D")
    operands = [lane_d, col_d]
    shapes = [("d", True, 128, _d_cols(M, N, ncta))]
    if otype(tr, op, 0).startswith("!cute.ptr<") and "tmem" in otype(tr, op, 0):
        if frag != "ts":
            tr.fail(op, f"R5: A from TMEM with frag_kind {frag}")
        la, ca = tmem_addr(tr, op, tr.val(op, 0), "tcgen05.mma A")
        if K * abits % 32:
            tr.fail(op, f"R5: A of {K} x {abits}-bit elements is not whole columns")
        operands += [la, ca]
        shapes.append(("a", False, 128, K * abits // 32))
    if bs:
        # One MMA's scale factors (the CuTe DSL's tmem SF layouts, blockscaled_layout.py):
        # SFA (operand 3) 128 rows in 4 columns x 4 lane quarters, SFB (operand 4) N rows
        # padded to 128-row atoms, 4 columns each (N = 192 or 256: 8); the K/vec <= 4
        # scale factors of a row are bytes of one column at the floored SF address.
        for i, ncols in ((3, 4), (4, 4 * -(-N // 128))):
            ls, cs = tmem_addr(tr, op, tr.val(op, i), "tcgen05.mma SF", exact=False)
            operands += [ls, cs]
            shapes.append(("sf", False, 128, ncols))
    loc = tr.loc(op)
    site = site_of(tr, loc)
    mask = _pair_mask(tr, op, ncta)
    one_warp_event(
        tr,
        op,
        operands,
        lambda vals, w, j: {
            "op": "event",
            "ev": "tmem",
            "inst": "mma",
            "warp": j(w),
            "rects": [
                _rect(u, wr, vals[2 * i], nl, vals[2 * i + 1], nc, j)
                for i, (u, wr, nl, nc) in enumerate(shapes)
            ],
            **({"cta_mask": j(mask)} if mask is not None else {}),
            "site": site,
            "loc": loc,
        },
        "tcgen05.mma",
    )
    return []


def _d_cols(m: int, n: int, ncta: int) -> int:
    """Columns of an MMA's D in each CTA, all 128 lanes: N, but N / 2 for M = 128 with
    cta_group::2 (64 rows per CTA in CuTe's "2x2" data path layout: the second half of N
    in lanes 64-127; tmem_frg_2sm, M_MMA = 64)."""
    return n // 2 if (m, ncta) == (128, 2) else n


def _pair_mask(tr, op, ncta: int):
    """The CTA mask of a cta_group::2 access, both CTAs of the issuing CTA's pair
    (3 << (rank & ~1)); None for cta_group::1."""
    if ncta == 1:
        return None
    if ncta != 2 or ops.cluster_dims(tr, op) == (1, 1, 1):
        tr.fail(op, f"R5: cta_group::{ncta} without a cluster of CTA pairs")
    rank = ops.cluster_rank(tr, op)
    return E.bin("shl", E.const(3, 32), E.bin("and", rank, E.const(-2, 32)))


@effect("cute_nvgpu.arch.mma.SM100.umma", "cute_nvgpu.arch.mma.SM100.umma_block_scaled")
def _umma_kept(tr, op):
    kept = [2]
    t0 = otype(tr, op, 0)
    if t0.startswith("!cute.ptr<") and "tmem" in t0:
        kept.append(0)
    if op.name.endswith("block_scaled"):
        kept += [3, 4]
    return kept


# ------------------------------------------------------- waits, barriers, alloc


@handler("nvvm.tcgen05.wait")
def _tcgen05_wait(tr, op):
    m = re.search(r"tcgen05_wait<(\w+)>", op.props.get("kind", ""))
    if m is None or m.group(1) not in ("load", "store"):
        tr.fail(op, f"R5: tcgen05.wait kind {op.props.get('kind')}")
    role_event(
        tr,
        op,
        {
            "op": "event",
            "ev": "tmem_wait",
            "ld": m.group(1) == "load",
            "loc": tr.loc(op),
        },
        "tcgen05.wait",
    )
    return []


@effect("nvvm.tcgen05.wait")
def _wait_kept(tr, op):
    return []


@handler("nvvm.barrier")
def _cta_barrier(tr, op):
    if op.operands:
        tr.fail(op, "R7: nvvm.barrier with operands")
    n = tr.k.block[0]
    role_event(
        tr,
        op,
        {
            "op": "event",
            "ev": "named_bar",
            "id": E.to_json(E.const(0, 32)),
            "count": E.to_json(E.const(n, 32)),
            "sync": True,
            "loc": tr.loc(op),
        },
        "barrier.sync 0",
    )
    return []


@effect("nvvm.barrier")
def _cta_barrier_kept(tr, op):
    return []


_BAR = re.compile(
    r"(?:bar|barrier)\.(sync|arrive)(?:\.aligned)?\s+(\$\d+|\d+)(?:\s*,\s*(\$\d+|\d+))?$"
)
_NUM = r"(?:0x[0-9a-fA-F]+|\d+)"
_TCP = re.compile(
    rf"tcgen05\.cp\.cta_group::([12])\.(128x256b|128x128b|32x128b\.warpx4|64x128b\.warpx2::(?:01_23|02_13))"
    rf"\s+\[([\w$]+)(?:\s*\+\s*({_NUM}))?\],\s*[\w$]+$"
)
_TMMA = re.compile(
    rf"tcgen05\.mma\.cta_group::([12])\.kind::(f16|tf32)\s+\[([\w$]+)\],\s*"
    rf"(?:\[([\w$]+)(?:\s*\+\s*({_NUM}))?\]|[\w$]+),\s*[\w$]+,\s*([\w$]+)(?:,\s*[\w$]+)?$"
)
_MOV = re.compile(rf"mov\.b32\s+(\w+),\s*(\$\d+|{_NUM})$")
# the destination operand of a PTX statement: a register or a {vector} of them
_DEST = re.compile(r"[a-z][\w.:]*\s+(\{[^}]*\}|[\w%]+)\s*,")
_TWAIT = re.compile(r"tcgen05\.wait::(ld|st)(?:\.sync\.aligned)?$")
_TACCESS = re.compile(r"tcgen05\.(ld|st|cp|mma|shift|alloc|dealloc)\b")


def _dests(s: str) -> list:
    """Registers a PTX statement writes (its first operand, when that is a register or a
    vector of registers)."""
    m = _DEST.match(s)
    return [x.strip() for x in m.group(1).strip("{}").split(",")] if m else []


def _src(regs: dict, op, x: str):
    """("op", operand index) or ("imm", value) of a PTX operand: `$k`, a constant, or a
    register the asm last set from one of them (mov.b32); None for a register set by any
    other statement."""
    if x.startswith("$"):
        return ("op", ops_mem._asm_operand(op, int(x[1:])))
    if re.fullmatch(_NUM, x):
        return ("imm", int(x, 0))
    return regs.get(x)


def _tcgen05_items(op, lines: list):
    """The tcgen05 cp / mma statements of an inline PTX in order, with the blocking wait
    loop between them (ops_mem._wait_loops): ("cp", elected, cta_group, ncols, taddr,
    offset), ("mma", elected, cta_group, idesc, d, a, a offset) with a None for an A in
    shared memory (elected: under the predicate of an elect.sync of the asm), ("wait",
    address operand, parity operand). Sources are _src values (an idesc may also
    be ("ptx", register): a function-scope register set elsewhere). ValueError (R4) for a
    statement outside the table."""
    regs: dict = {}
    elect: set = set()  # predicates elect.sync sets (full member mask)
    waits = {lo: (hi, a, p) for lo, hi, a, p in ops_mem._wait_loops(lines)}
    items, i = [], 0
    while i < len(lines):
        if i in waits:
            hi, a, p = waits[i]
            items.append(
                ("wait", ops_mem._asm_operand(op, a), ops_mem._asm_operand(op, p))
            )
            i = hi
            continue
        g, s = lines[i]
        i += 1
        m = _MOV.match(s)
        if m:
            regs[m.group(1)] = _src(regs, op, m.group(2))
            elect.discard(m.group(1))
            continue
        m = re.fullmatch(r"elect\.sync\s+_\|(\w+),\s*-1", s)
        if m:
            elect.add(m.group(1))
            continue
        for x in _dests(
            s
        ):  # computed here: no longer a known source or elect predicate
            regs[x] = None
            elect.discard(x)
        tcp, tmma = _TCP.match(s), _TMMA.match(s)
        if (tcp or tmma) and g and g[1:] not in elect:
            raise ValueError(f"R6: inline PTX {s[:40]!r} under the predicate {g}")
        if tcp:
            ncols = 8 if tcp.group(2).endswith("256b") else 4
            items.append(("cp", bool(g), int(tcp.group(1)), ncols, _src(regs, op, tcp.group(3)),
                          int(tcp.group(4) or "0", 0)))  # fmt: skip
            continue
        if tmma:
            idesc = _src(regs, op, tmma.group(6)) or ("ptx", tmma.group(6))
            a = _src(regs, op, tmma.group(4)) if tmma.group(4) else None
            items.append(("mma", bool(g), int(tmma.group(1)), idesc, _src(regs, op, tmma.group(3)), a,
                          int(tmma.group(5) or "0", 0)))  # fmt: skip
            continue
        if _TACCESS.match(s):
            raise ValueError(
                f"R4: inline PTX with a tcgen05 data, MMA or allocation statement outside the table: {s[:60]!r}"
            )
    return items


def classify_p1(op):
    """P1 meaning of an inline asm: ("bar", sync, id, count) with id/count ("imm", v) or
    ("op", operand index) (count None: all threads of the block), ("twait", ld),
    ("tcgen05", items) (_tcgen05_items), or None (the P0 meaning applies). ValueError for
    R4 / R7."""
    lines = ops_mem._asm_lines(op)
    if any(_TCP.match(s) or _TMMA.match(s) for _g, s in lines):
        return ("tcgen05", _tcgen05_items(op, lines))
    stmts = ops_mem._asm_statements(op)
    for _g, s in stmts:
        if _TACCESS.match(s):
            raise ValueError(
                f"R4: inline PTX with a tcgen05 data, MMA or allocation statement: {s[:60]!r}"
            )
    evs = [(g, s) for g, s in stmts if _BAR.match(s) or _TWAIT.match(s)]
    if not evs:
        return None
    if len(stmts) != 1 or stmts[0][0]:
        raise ValueError(
            f"R7: inline PTX mixing a named barrier or tcgen05.wait with other statements: {stmts[:3]}"
        )
    s = stmts[0][1]
    m = _TWAIT.match(s)
    if m:
        return ("twait", m.group(1) == "ld")
    m = _BAR.match(s)

    def arg(x):
        if x is None:
            return None
        return (
            ("op", ops_mem._asm_operand(op, int(x[1:])))
            if x.startswith("$")
            else ("imm", int(x))
        )

    return ("bar", m.group(1) == "sync", arg(m.group(2)), arg(m.group(3)))


@handler("llvm.inline_asm")
def _inline_asm_p1(tr, op):
    try:
        kind = classify_p1(op)
    except ValueError as e:
        tr.fail(op, str(e))
    if kind is None:
        return ops.HANDLERS["llvm.inline_asm"](tr, op)
    if kind[0] == "tcgen05":
        for item in kind[1]:
            _TCGEN05_EVENTS[item[0]](tr, op, *item[1:])
    elif kind[0] == "twait":
        role_event(
            tr,
            op,
            {"op": "event", "ev": "tmem_wait", "ld": kind[1], "loc": tr.loc(op)},
            "tcgen05.wait",
        )
    else:
        _named_bar_asm(tr, op, *kind[1:])
    return [tr.data(op)] * len(op.results)


def _named_bar_asm(tr, op, sync: bool, idx, cnt) -> None:
    """bar.sync / bar.arrive id[, count] (classify_p1's sources): a named_bar event."""

    def value(a, what):
        if a is None:
            return tr.k.block[0]
        if a[0] == "imm":
            return a[1]
        e = tr.int_of(op, a[1], f"named barrier {what}")
        u = tr.uniform(e)
        if u is None:  # one role per run of warps with one value, if that splits it
            tr.split_per_warp(e)
        if u is None or not E.is_const(u):
            tr.fail(op, f"R7: named barrier {what} is not a constant")
        return u[2]

    bid, count = value(idx, "id"), value(cnt, "thread count")
    if count <= 0 or count % 32:
        tr.fail(
            op,
            f"R7: named barrier thread count {count} is not a positive multiple of 32",
        )
    role_event(
        tr,
        op,
        {
            "op": "event",
            "ev": "named_bar",
            "id": E.to_json(E.const(bid, 32)),
            "count": E.to_json(E.const(count, 32)),
            "sync": sync,
            "loc": tr.loc(op),
        },
        f"bar.{'sync' if sync else 'arrive'}",
    )


@effect("llvm.inline_asm")
def _asm_kept_p1(tr, op):
    try:
        kind = classify_p1(op)
    except ValueError as e:
        tr.fail(op, str(e))
    if kind is None:
        return ops.EFFECTS["llvm.inline_asm"](tr, op)
    if kind[0] == "tcgen05":
        kept = set()
        for item in kind[1]:
            srcs = item[1:] if item[0] == "wait" else item[3:]
            for x in srcs:
                if isinstance(x, tuple) and x[0] == "op":
                    kept.add(x[1])
                elif isinstance(x, int) and item[0] == "wait":
                    kept.add(x)
        return sorted(kept)
    if kind[0] == "twait":
        return []
    return [a[1] for a in kind[2:] if a is not None and a[0] == "op"]


def _asm_tmem(tr, op, src, off: int, what: str):
    """(lane, column) of an inline PTX TMEM operand: an absolute address (lane << 16 |
    column; _ptrtoint_p1 or a constant), plus `off` columns."""
    if src is None or src[0] not in ("op", "imm"):
        tr.fail(op, f"R4: {what}: a TMEM address the asm computes")
    a = (
        E.const(src[1], 32)
        if src[0] == "imm"
        else E.resize(tr.int_of(op, src[1], f"{what} address"), 32)
    )
    tr.tmem_bases.add("abs")
    lane = E.bin("lshr", a, E.const(16, 32))
    col = E.bin("add", E.bin("and", a, E.const(0xFFFF, 32)), E.const(off, 32))
    return lane, col


def _ptx_const(tr, op, name: str) -> int:
    """The value of function-scope PTX register `name`, which some inline PTX of the kernel
    declares (`.reg .b32 name;` without braces) and sets to one constant, and no inline PTX
    writes otherwise."""
    table = tr.k.__dict__.get("_ptx_consts")
    if table is None:
        table, declared = {}, set()
        for o in interp.mlir_text.walk(tr.k.module.top):
            if o.name != "llvm.inline_asm":
                continue
            text = o.props.get("asm_string", "")
            if "{" not in text:
                declared.update(re.findall(r"\.reg\s+\.b32\s+(\w+)\s*;", text))
            for _g, s in ops_mem._asm_lines(o):
                m = _MOV.match(s)
                if m and re.fullmatch(_NUM, m.group(2)):
                    table.setdefault(m.group(1), set()).add(int(m.group(2), 0))
                else:
                    for x in _dests(s):
                        table.setdefault(x, set()).add(None)
        table = {x: v for x, v in table.items() if x in declared}
        tr.k._ptx_consts = table
    vals = table.get(name, set())
    if len(vals) != 1 or None in vals:
        tr.fail(
            op,
            f"R5: instruction descriptor register {name} is not a function-scope register set to one constant",
        )
    return next(iter(vals))


def _asm_cp(tr, op, elected: bool, ncta: int, ncols: int, taddr, off: int):
    lane, col = _asm_tmem(tr, op, taddr, off, "tcgen05.cp")
    mask = _pair_mask(tr, op, ncta)
    loc = tr.loc(op)
    one_warp_event(
        tr,
        op,
        [lane, col] + ([mask] if mask is not None else []),
        lambda v, w, j: {
            "op": "event", "ev": "tmem", "inst": "cp", "warp": j(w),
            "rects": [_rect("data", True, v[0], 128, v[1], ncols, j)],
            **({"cta_mask": j(v[2])} if mask is not None else {}),
            "site": site_of(tr, loc), "loc": loc,
        },
        "tcgen05.cp",
        elected,
    )  # fmt: skip


def _asm_mma(
    tr: interp.Translator, op, elected: bool, ncta: int, idesc, d, a, a_off: int
):
    """A dense kind::f16 / tf32 MMA: N >> 3 at idesc bits 17-22, M >> 4 at bits 24-28
    (PTX ISA table 52); D is 128 lanes per CTA (_d_cols); a TMEM A of K x 16 (f16) or x 32
    (tf32) bits per instruction is 8 columns."""
    v = (
        idesc[1]
        if idesc[0] == "imm"
        else _ptx_const(tr, op, idesc[1])
        if idesc[0] == "ptx"
        else None
    )
    if v is None:
        tr.fail(op, "R5: tcgen05.mma instruction descriptor is not a constant")
    n, m = ((v >> 17) & 0x3F) << 3, ((v >> 24) & 0x1F) << 4
    if (m, ncta) not in ((128, 1), (128, 2), (256, 2)) or not 8 <= n <= 256:
        tr.fail(
            op,
            f"R5: inline tcgen05.mma.cta_group::{ncta} {m}x{n} (instruction descriptor {v:#x})",
        )
    lane_d, col_d = _asm_tmem(tr, op, d, 0, "tcgen05.mma D")
    operands, shapes = [lane_d, col_d], [("d", True, 128, _d_cols(m, n, ncta))]
    if a is not None:
        la, ca = _asm_tmem(tr, op, a, a_off, "tcgen05.mma A")
        operands += [la, ca]
        shapes.append(("a", False, 128, 8))
    mask = _pair_mask(tr, op, ncta)
    if mask is not None:
        operands.append(mask)
    loc = tr.loc(op)
    one_warp_event(
        tr,
        op,
        operands,
        lambda vals, w, j: {
            "op": "event", "ev": "tmem", "inst": "mma", "warp": j(w),
            "rects": [_rect(u, wr, vals[2 * i], nl, vals[2 * i + 1], nc, j) for i, (u, wr, nl, nc) in enumerate(shapes)],
            **({"cta_mask": j(vals[-1])} if mask is not None else {}),
            "site": site_of(tr, loc), "loc": loc,
        },
        "tcgen05.mma",
        elected,
    )  # fmt: skip


def _asm_wait(tr, op, addr: int, parity: int):
    tr.barrier_event(op, "wait", ops_mem._smem_address(tr, op, addr), "once",
                     parity=tr.int_of(op, parity, "wait parity"))  # fmt: skip


_TCGEN05_EVENTS = {"cp": _asm_cp, "mma": _asm_mma, "wait": _asm_wait}


@handler("cute_nvgpu.arch.sm100.alloc_tmem")
def _alloc_tmem(tr, op):
    n = tr.uniform(tr.int_of(op, 0, "TMEM allocation column count"))
    if n is None or not E.is_const(n):
        tr.fail(op, "R2: alloc_tmem with a non-constant column count")
    if tr.loop_depth:
        tr.fail(op, "R2: alloc_tmem inside a loop")
    uni, var = tr.split_mask()
    act = tr.active(var)
    if act is None:
        tr.fail(op, "R2: alloc_tmem under a runtime lane condition")
    if act:  # tcgen05.alloc is warp-collective: one allocation per issuing warp
        for _w in range(tr.full_warps(op, act, "tcgen05.alloc")):
            tr.tmem_allocs.append((n[2], tr.loc(op)))
    return []


@effect("cute_nvgpu.arch.sm100.alloc_tmem")
def _alloc_kept(tr, op):
    return [0]


@handler("cute.make_tiled_mma")
def _tiled_mma(tr, op):
    # P1: the MMA atom carries its fields (sf_a, sf_b, accum_c, ...) like a copy atom
    return interp.Opaque("atom", (("type", rtype(tr, op)),))


@handler("cute_nvgpu.arch.sm100.retrieve_tmem_ptr")
def _retrieve(tr, op):
    bits, _space = ptr_info(rtype(tr, op))
    return Ptr("tmem", "alloc", E.const(0, 64), bits)


@handler("cute.inttoptr")
def _inttoptr_p1(tr, op):
    bits, space = ptr_info(rtype(tr, op))
    if space != "tmem":
        return ops.HANDLERS["cute.inttoptr"](tr, op)
    a = tr.val(op, 0)
    if isinstance(a, tuple):
        u = tr.uniform(a)
        if u is not None and E.is_const(u, 0):
            return Ptr("tmem", "abs", E.const(0, 64), bits)
        if u is not None and tr.tmem_abs.get(u) is not None:
            return tr.tmem_abs[u]  # inttoptr(ptrtoint(p)): p
    return Data(
        f"R1: TMEM pointer of unknown provenance (cute.inttoptr at {tr.loc(op)})"
    )


@handler("cute.ptrtoint")
def _ptrtoint_p1(tr, op):
    """The address of an absolute TMEM pointer is an integer (lane << 16 | column), which
    inline PTX may take as a TMEM operand; any other pointer as in the P0 pass."""
    p = tr.val(op, 0)
    if not (isinstance(p, Ptr) and p.space == "tmem" and p.base == "abs"):
        return ops.HANDLERS["cute.ptrtoint"](tr, op)
    lane, col = tmem_addr(tr, op, p, "cute.ptrtoint of a TMEM pointer")
    w = interp.int_width(rtype(tr, op)) or 32
    a = E.resize(E.bin("or", E.bin("shl", lane, E.const(16, 32)), col), w)
    tr.tmem_abs[a] = p
    return a


def tmem_top(roles: list) -> dict:
    """The P1 KIR's top-level `tmem` from the roles' allocations and pointer bases."""
    allocs = [a for tr in roles for a in tr.tmem_allocs]
    bases = set().union(*(tr.tmem_bases for tr in roles)) if roles else set()
    if len(allocs) != 1:
        raise interp.Unsupported(
            "cute_nvgpu.arch.sm100.alloc_tmem",
            allocs[1][1] if len(allocs) > 1 else "kernel",
            f"R2: {len(allocs)} TMEM allocations (exactly one is supported)",
        )
    cols, loc = allocs[0]
    if cols not in (32, 64, 128, 256, 512):
        raise interp.Unsupported(
            "cute_nvgpu.arch.sm100.alloc_tmem", loc, f"R2: {cols} columns"
        )
    if bases == {"abs"} and cols != 512:
        raise interp.Unsupported(
            "cute.inttoptr",
            loc,
            f"R3: absolute TMEM addresses with a {cols}-column allocation "
            "(the allocation's address is not known)",
        )
    if len(bases) > 1:
        raise interp.Unsupported(
            "cute.inttoptr",
            loc,
            "R3: absolute and allocation-relative TMEM pointers mixed",
        )
    return {"cols": cols, "base": "relative" if bases == {"alloc"} else "zero"}
