# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Thread-block clusters in the PTX frontend (design §5.4.1, §5.6, §5.7): the cluster
semantics of the CuTe family as the legacy CuTe frontend models them
(references/frontend_cute.md "Clusters", references/kir.md change 11).

    mapa(tr, i)                       mem.py: mapa.shared::cluster d, a, rank
    peer(tr, v) -> Peer | None        interp.py: a shared address with the peer bit cleared
    execute(tr, i, mnemonic)          sync.py: barrier.cluster.*, mbarrier.arrive(.expect_tx)
                                      on shared::cluster, mbarrier.complete_tx
    pair(tr, i, mnemonic)             sync.py, here: a cta_group::2 instruction needs a
                                      cluster of CTA pairs
    commit_fields(tr, i, mnemonic)    sync.py: {cta_mask} of a multicast tcgen05.commit
    load_slot(tr, i) -> (symbol, offset, fields)
                                      tma.py: a TMA load's mbarrier slot and its fields
                                      cta_mask (multicast) and leader (cta_group::2)
    own_cta(tr, i, k, what)           tma.py: memory operand k is the issuing CTA's
    bulk_s2cluster(tr, i)             tma.py: cp.async.bulk.shared::cluster.shared::cta
    pair_mask(tr) -> expression       the CTAs 3 << (rank & ~1) of the issuing CTA's pair
                                      (a cta_group::2 tmem access in P1)
    barriers(translators) -> [barrier]
                                      mem.py: the cluster barrier array, if a role uses it
    prologue(tr) -> [statement]       translate.py: a role's cluster barrier parity

The cluster special registers (%cluster_ctarank, %cluster_nctarank, %clusterid,
%nclusterid, %cluster_ctaid, %cluster_nctaid, %is_explicit_cluster) are folded from the
LaunchRecords' cluster dims by interp.Translator.sreg.

Addresses. `mapa.shared::cluster d, a, r` makes a Remote: the shared address `a` (a
shared symbol plus an offset, or another Remote's) in the CTA of rank r. A Remote is Data
to every rule but the operands below, so any arithmetic on it is refused where it reaches
a root, as a mapa result was before. An mbarrier arrive or arrive.expect_tx through a
Remote gets the event's `cta` (uniform over the role, a role whose warps differ in it is
split), as the legacy frontend's interp.Remote; through a shared::cta address it is the
issuing CTA's slot (no `cta`). In a cluster of one CTA a Remote must name rank 0, and no
`cta` is written (KIR: the field needs a cluster of several CTAs).

The peer bit (design §5.6, a trust item of design §7). `a & 0xFEFFFFF8` of a shared
address `a` clears bit 24 of its shared::cluster window address, which names the even CTA
of the issuing CTA's pair (CUTLASS's Sm100MmaPeerBitMask; the PTX ISA does not document
the bit): a Peer, the slot at offset `a` in the CTA of rank `rank & ~1`. LLVM shrinks the
mask by the bits it knows are zero (a 1024-aligned barrier: 0xFEFFFC00), so a mask is
accepted when it keeps every bit at or above the symbol's .align but bit 24, clears bit
24, and every lower bit it clears is provably zero in the offset (known trailing zeros,
the bits its bit operations and disjoint adds may set, or an interval below that bit);
else the value is Data, as before. A Peer is a TMA load's
mbarrier only, and only with .cta_group::2: KIR `leader` (each destination's bytes
complete on the slot of the even CTA of its pair). A cta_group::2 load whose mbarrier is a
shared::cta address completes on the issuing CTA's own slot (no leader: the ISA lets the
mbarrier be in either CTA of the pair, the address says which); a Peer without
.cta_group::2 is refused (the mbarrier must be in the destination CTA).

Events (the legacy frontend's, references/frontend_cute.md "Clusters"):
  * barrier.cluster.arrive{.relaxed}{.aligned}: an `arrive` of every thread of the role
    (counted: n = threads / threads_per_issue) on `cluster_barrier`, a frontend-made
    barrier of one slot in the cluster's CTA of rank 0 (`cta` 0) whose count is one
    arrival per thread of the cluster; barrier.cluster.wait{.aligned}: a `wait` on it for
    the role's next phase, its parity the role variable `cluster_barrier_parity` (0 at
    the role's start, prologue; flipped after each wait). Reached by part of a role, or
    under a lane condition that depends on runtime values: refused (threads that exit
    early are not modeled; in PTX they count as arrived).
  * tcgen05.commit...multicast::cluster [a], m: the commit_arrive per issuing thread
    (sync.py) with `cta_mask` m (zero-extended, uniform over the role).
  * TMA load .multicast::cluster [dst], [tmap, {...}], [mbar], m: `cta_mask` m; with
    .cta_group::2 and a Peer mbarrier also `leader`. The box is the descriptor's (the
    hardware box: the legacy frontend's slice of a multicast descriptor's tile).
  * cp.async.bulk.shared::cluster.shared::cta [dst], [src], n, [mbar] with dst and mbar
    Remotes of one CTA: a `complete_tx` of n bytes on that CTA's slot, per issuing thread.
  * cta_group::2 tcgen05 instructions emit nothing in P0 (sync.py); they need a cluster
    of an even number of CTAs.
A load (TMA or bulk) must write the issuing CTA's shared memory unless it multicasts:
a destination Remote of another rank is refused (the mbarrier must be in the destination
CTA, and KIR has no remote tma_load). mbarrier.complete_tx and anything else on another
CTA are gaps.

Issuing (design §5.7, D15). A remote arrive and a commit follow the ISA (per executing
thread, as their shared::cta forms). The ptxas lowering of a multicast or cta_group::2 TMA
load and of a shared::cluster bulk copy that several lanes of a warp execute was not
measured on hardware (tests/ptx/test_gpu_tma_issue.py covers the other forms), so such an
instruction is translated only where at most one lane of each warp executes it
(elect.sync, a lane test), once per executing thread, and refused otherwise.

The Triton family: clusters of more than one CTA are refused before translation (v1,
design D3), and its cluster-scope barrier forms stay gaps.
"""

import math
from dataclasses import dataclass

from . import expr as E
from .interp import BarrierUse, Data, Translator

PEER_BIT = 1 << 24  # the CTA-pair bit of a shared::cluster address (module doc)
CLUSTER_BAR = "cluster_barrier"  # the legacy frontend's names (frontend/cute/interp.py)
CLUSTER_PARITY = "cluster_barrier_parity"
_MASK32 = (1 << 32) - 1


@dataclass(frozen=True)
class Remote(Data):
    """A shared::cluster address made by mapa: the shared address `addr` (32-bit, over a
    %s: base) in the CTA of rank `cta` (i32) of the cluster."""

    addr: tuple = ()
    cta: tuple = ()


@dataclass(frozen=True)
class Peer(Data):
    """A shared address `addr` with the peer bit cleared: that slot in the even CTA of the
    issuing CTA's pair (module doc)."""

    addr: tuple = ()


# ------------------------------------------------------------------ addresses


def mapa(tr: Translator, i) -> None:
    """mapa.shared::cluster d, a, r: d is a Remote, or Data if a is no shared address."""
    dst = i.operands[0]
    origin = f"shared::cluster address (mapa) at {tr.loc(i)}"
    a = tr.operand_value(i.operands[1], 32)
    if isinstance(a, Remote):
        a = a.addr
    r = tr.operand(i, 2, 32)
    if dst[0] != "reg":
        return
    if isinstance(a, Data) or isinstance(r, Data) or _shared(a) is None:
        tr.env[dst[1]] = Data(origin)
        return
    tr.env[dst[1]] = Remote(origin, E.resize(a, 32, False), r)


def peer(tr: Translator, v):
    """Peer(addr) if v is `addr & M` for a shared address addr and the peer-bit mask M up
    to bits addr provably has zero (module doc), else None."""
    if v[0] != "bin" or v[2] != "and" or E.width(v) != 32 or not E.is_const(v[4]):
        return None
    p = _shared(v[3])
    if p is None:
        return None
    m = v[4][2] & _MASK32
    s = min(_align_bits(tr, p[0][2][3:]), 24)
    keep = (_MASK32 & -(1 << s)) & ~PEER_BIT  # address bits the symbol does not fix
    if m & PEER_BIT or m & keep != keep:
        return None
    # offset bits known to be zero: trailing zeros (a slot's facts), bit structure
    zero = ~E._ones(p[1]) | ((1 << E.tz(p[1], tr.slot_facts)) - 1)
    rng = E.interval(p[1])
    for k in range(s):
        below = rng is not None and 0 <= rng[0] and rng[1] < 1 << k
        if not m >> k & 1 and not zero >> k & 1 and not below:
            return None  # the mask clears an offset bit that may be 1
    return Peer(f"the pair leader's address (peer bit cleared) at {tr.loc()}", v[3])


def own_cta(tr: Translator, i, k: int, what: str) -> None:
    """Refuse memory operand k of i if it is a Remote of a CTA other than the issuing one."""
    v = _address(tr, i.operands[k])
    if isinstance(v, Remote) and not _is_own(tr, v.cta):
        tr.fail(i, f"{what} is in another CTA of the cluster (a mapa result): only the "
                   "issuing CTA's shared memory is modeled there")  # fmt: skip


def pair(tr: Translator, i, mn: str) -> None:
    """A cta_group::2 instruction runs on a CTA pair: the cluster has an even size."""
    if math.prod(tr.k.cluster) % 2:
        tr.fail(
            i, f"{mn} without a cluster of CTA pairs (cluster {tuple(tr.k.cluster)})"
        )


def pair_mask(tr: Translator):
    """The CTA mask of the issuing CTA's pair, 3 << (rank & ~1) (i32)."""
    rank = tr.sreg("%cluster_ctarank")
    return E.bin("shl", E.const(3, 32), E.bin("and", rank, E.const(-2, 32)))


# --------------------------------------------------------------------- events


def execute(tr: Translator, i, mn: str) -> None:
    if tr.k.mem.policy.get("family") != "cute":
        tr.fail(
            i, f"{mn} (gap: cluster scope in the Triton family is not in v1, design D3)"
        )
    if mn.startswith("barrier.cluster"):
        _cluster_barrier(tr, i, mn)
    elif mn in (
        "mbarrier.arrive.shared::cluster",
        "mbarrier.arrive.expect_tx.shared::cluster",
    ):
        _remote_arrive(tr, i, mn)
    else:
        tr.fail(i, f"{mn} (gap: KIR complete_tx is a shared::cluster copy's, as in the CuTe "
                   "frontend)")  # fmt: skip


def _cluster_barrier(tr: Translator, i, mn: str) -> None:
    arrive = ".arrive" in mn
    if mn.endswith(".aligned"):
        tr.aligned(mn)
    _uni, var = tr.split_mask()
    act = tr.active(var)
    if act is None or (act and len(act) != 32 * tr.nw):
        tr.fail(i, "barrier.cluster reached by only part of the role (or under a lane "
                   "condition that depends on runtime values)")  # fmt: skip
    zero = {"e": "const", "w": 32, "v": 0}
    cta = {k: tr.json(v) for k, v in _cta(tr, i, E.const(0, 32)).items()}
    loc = tr.loc()
    stmt = {"op": "event", "ev": "arrive" if arrive else "wait", "bar": CLUSTER_BAR,
            "idx": zero, **cta, "loc": loc}  # fmt: skip
    parity = {"e": "var", "w": 32, "name": CLUSTER_PARITY}
    if not arrive:
        stmt["parity"] = parity
    issued = tr._issue(stmt, "count" if arrive else "once", [])
    if issued is None:
        return
    tr.barriers.append(BarrierUse(stmt, CLUSTER_BAR, 0, None, None, issued[2], loc))
    tr._emit_under(issued[0], issued[1])
    if not arrive:
        flip = {"e": "bin", "w": 32, "o": "xor", "a": parity, "b": {**zero, "v": 1}}
        tr._emit_under(issued[0], {"op": "let", "var": CLUSTER_PARITY, "expr": flip,
                                   "loc": loc})  # fmt: skip


def _remote_arrive(tr: Translator, i, mn: str) -> None:
    """mbarrier.arrive(.expect_tx).shared::cluster _, [a](, n): mem.py's arrive on the
    slot a names, `cta` when a is a Remote."""
    sym, off, where = _slot(tr, i, i.operands[1])
    if i.operands[0][0] == "reg":
        tr.env[i.operands[0][1]] = Data(f"mbarrier state at {tr.loc(i)}")
    n = (
        tr.as_int(tr.operand(i, 2, 32), "the mbarrier count")
        if len(i.operands) > 2
        else None
    )
    if "expect_tx" in mn:
        tr.barrier_event("expect_tx", sym, off, "thread", bytes=n, **where)
        tr.barrier_event("arrive", sym, off, "count", **where)
        return
    u = E.const(1, 32) if n is None else tr.uniform(n)
    if u is None or not E.is_const(u) or u[2] < 1:
        tr.fail(i, "mbarrier arrive with a count that is not a positive constant")
    tr.barrier_event("arrive", sym, off, "count", **where,
                     **({} if E.is_const(u, 1) else {"n": u}))  # fmt: skip


def commit_fields(tr: Translator, i, mn: str) -> dict:
    """{"cta_mask": m} of tcgen05.commit...multicast::cluster [a], m; {} without it."""
    if "multicast" not in mn:
        return {}
    _several(tr, i, "a multicast tcgen05.commit")
    return {"cta_mask": tr.as_int(tr.operand(i, 1, 32), "the tcgen05.commit CTA mask")}


def load_slot(tr: Translator, i) -> tuple:
    """(symbol, offset, fields) of the mbarrier of a TMA load
    cp.async.bulk.tensor...[.multicast::cluster][.cta_group::2] [dst], [tmap, {..}],
    [mbar](, mask)(, policy): fields {cta_mask, leader} (module doc)."""
    op, group2 = i.opcode, ".cta_group::2" in i.opcode
    if group2:
        pair(tr, i, "TMA load .cta_group::2")
    v = _address(tr, i.operands[2])
    fields: dict = {}
    if isinstance(v, Peer):
        if not group2:
            tr.fail(i, "TMA load completing on the pair leader's mbarrier (address & "
                       "0xFEFFFFF8) without .cta_group::2 (ISA: the mbarrier is in the "
                       "destination CTA)")  # fmt: skip
        fields["leader"], v = True, v.addr
    if ".multicast::cluster" in op:
        _several(tr, i, "a multicast TMA load")
        fields["cta_mask"] = tr.as_int(
            tr.operand(i, 3, 32), "the TMA multicast CTA mask"
        )
    if fields or group2:  # not measured (module doc), on the own mbarrier too
        _one_lane(tr, i, "TMA load" + " .multicast::cluster" * ("cta_mask" in fields)
                  + " .cta_group::2" * group2)  # fmt: skip
    sym, off = _symbol_offset(tr, i, tr.as_int(v, "an mbarrier address"))
    return sym, off, fields


def bulk_s2cluster(tr: Translator, i) -> None:
    """cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [dst], [src],
    n, [mbar]: complete_tx(n) on the slot of mbar's CTA, which must be dst's."""
    _several(tr, i, "a shared::cluster bulk copy")
    dst, bar = _address(tr, i.operands[0]), _address(tr, i.operands[3])
    if not (isinstance(dst, Remote) and isinstance(bar, Remote)):
        tr.fail(i, "shared::cluster bulk copy whose destination and mbarrier are not mapa "
                   "results")  # fmt: skip
    same = tr.uniform(E.cmp("eq", dst.cta, bar.cta))
    if same is None or not E.is_const(same, 1):
        tr.fail(
            i, "shared::cluster bulk copy into one CTA completing on another's mbarrier"
        )
    n = tr.as_int(tr.operand(i, 2, 32), "the bulk copy size")
    _one_lane(tr, i, "shared::cluster bulk copy")
    sym, off = _symbol_offset(tr, i, bar.addr)
    tr.barrier_event("complete_tx", sym, off, "thread", bytes=n, cta=bar.cta)


# ----------------------------------------------------------- after the roles


def barriers(translators: list) -> list:
    """[the KIR barrier `cluster_barrier`] if some role uses it, else []: one slot, one
    arrival per thread of the cluster."""
    if not any(
        u.event.get("bar") == CLUSTER_BAR for tr in translators for u in tr.barriers
    ):
        return []
    k = translators[0].k
    count = k.block[0] * math.prod(k.cluster)
    return [{"name": CLUSTER_BAR, "count": {"e": "const", "w": 32, "v": count},
             "num": {"e": "const", "w": 32, "v": 1}}]  # fmt: skip


def prologue(tr: Translator) -> list:
    """`let cluster_barrier_parity = 0` for a role that waits on the cluster barrier."""
    if not any(u.event["ev"] == "wait" and u.event.get("bar") == CLUSTER_BAR
               for u in tr.barriers):  # fmt: skip
        return []
    return [
        {"op": "let", "var": CLUSTER_PARITY, "expr": {"e": "const", "w": 32, "v": 0}}
    ]


# -------------------------------------------------------------------- helpers


def _peel(e):
    from .mem import peel  # mem.py imports this module (through sync.py)

    return peel(e)


def _shared(a):
    """peel(a) if a is a shared symbol plus an offset, else None."""
    p = None if isinstance(a, Data) else _peel(a)
    return p if p is not None and p[0][2].startswith("%s:") else None


def _address(tr: Translator, op):
    """The value a memory operand [base + offset] names: a Remote (or Peer) base keeps its
    kind, its address moved by the offset; else mem.Memory.address."""
    if op[0] == "mem" and op[1][0] == "reg" and op[3] is None:
        v = tr.env.get(op[1][1])
        if isinstance(v, (Remote, Peer)) and op[2]:
            a = E.bin("add", v.addr, E.const(op[2], 32))
            return (
                Remote(v.origin, a, v.cta)
                if isinstance(v, Remote)
                else Peer(v.origin, a)
            )
        if isinstance(v, (Remote, Peer)):
            return v
    return tr.k.mem.address(tr, op)


def _symbol_offset(tr: Translator, i, a) -> tuple:
    p = _shared(a)
    if p is None:
        tr.fail(i, "mbarrier address is not a shared-memory symbol plus an offset")
    return p[0][2][3:], p[1]


def _slot(tr: Translator, i, op) -> tuple:
    """(symbol, offset, {"cta": rank} or {}) of an mbarrier operand in shared::cluster."""
    v = _address(tr, op)
    where = _cta(tr, i, v.cta) if isinstance(v, Remote) else {}
    a = v.addr if isinstance(v, Remote) else tr.as_int(v, "an mbarrier address")
    return (*_symbol_offset(tr, i, a), where)


def _cta(tr: Translator, i, cta) -> dict:
    """{"cta": cta}, or {} in a cluster of one CTA (cta must be 0 there)."""
    if math.prod(tr.k.cluster) > 1:
        return {"cta": cta}
    if not _is_own(tr, cta):
        tr.fail(
            i,
            "shared::cluster address of a CTA rank other than 0 in a cluster of one CTA",
        )
    return {}


def _is_own(tr: Translator, cta) -> bool:
    """cta is the issuing CTA's rank (in every thread of the role)."""
    if math.prod(tr.k.cluster) == 1:
        return E.is_const(cta, 0)
    same = tr.uniform(E.cmp("eq", cta, tr.sreg("%cluster_ctarank")))
    return same is not None and E.is_const(same, 1)


def _several(tr: Translator, i, what: str) -> None:
    """The KIR forms cta_mask and complete_tx need a cluster of several CTAs."""
    if math.prod(tr.k.cluster) == 1:
        tr.fail(i, f"{what} outside a cluster of several CTAs")


def _one_lane(tr: Translator, i, what: str) -> None:
    """At most one lane of each warp issues i under the current mask (module doc)."""
    act = tr.active(tr.split_mask()[1])
    if act is None or len({t // 32 for t in act}) != len(act):
        tr.fail(i, f"{what} that several lanes of a warp may execute: its issuing is not "
                   "measured (tests/ptx/test_gpu_tma_issue.py covers the other copies)")  # fmt: skip


def _align_bits(tr: Translator, sym: str) -> int:
    """log2 of the declared .align of shared symbol sym (0 if unknown)."""
    align = (tr.k.mem.policy.get("shared_align") or {}).get(sym)
    if align is None:
        align = next((v.align for v in tr.k.fn.vars if v.name == sym), 1)
    return max(align, 1).bit_length() - 1
