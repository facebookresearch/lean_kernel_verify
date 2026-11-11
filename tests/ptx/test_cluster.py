# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of thread-block clusters in the PTX core (frontend/ptx/cluster.py) through
translate() on hand-written PTX.

    python3 -I tests/ptx/test_cluster.py      (or the buck target :test_cluster)

Special registers folded from the cluster dims; mapa and a remote mbarrier arrive /
arrive.expect_tx (`cta`, a role split by a per-warp rank, a lane-varying rank refused, no
`cta` in a cluster of one CTA); barrier.cluster arrive/wait on `cluster_barrier` (count,
parity prologue and flips, part of a role refused, the Triton family a gap); the peer-bit
mask of a CTA pair's mbarrier (`leader` with .cta_group::2, LLVM's shrunken masks, masks
that clear a bit the offset may set or that the symbol's alignment does not cover,
without .cta_group::2, outside a cluster of pairs); multicast TMA loads (`cta_mask`,
outside a cluster, issued by several lanes, as cta_group::2 ones); a load into another
CTA refused; multicast
tcgen05.commit (`cta_mask`) and cta_group::2 tcgen05 ops; shared::cluster bulk copies
(`complete_tx` on the destination CTA, and their refusals); pair_mask; branches whose
arms emit nothing (on data: not a root if the arms only set registers, a warp collective
or a .sync.aligned instruction only under a condition the same in all lanes of a warp, a
vote.sync of data; else, or if the trial refuses an arm, the condition is refused; on
the warp: no role split, a warp branch inside an arm splitting by its own condition).
Every translated kernel's KIR is checked with frontend/kir.py.
"""

import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

if __package__ in (None, ""):
    sys.dont_write_bytecode = True
    _root = ModuleType("lkv_root")
    _root.__path__ = [str(Path(__file__).resolve().parents[2])]
    sys.modules["lkv_root"] = _root
    __package__ = "lkv_root.tests.ptx"  # noqa: A001

from ...frontend import kir  # noqa: E402
from ...frontend.ptx import cluster, expr as E, interp, translate as T  # noqa: E402

SRC = "/src"
PAIR = (2, 1, 1)


def _tmap(address=0x10000):
    """A decoded 2-D descriptor (launch.decode's fields) over the 2-byte tensor x."""
    return {"dtype": "X", "elem_bits": 16, "rank": 2, "address": address, "dims": [128, 256],
            "strides_bytes": [256], "box": [32, 64], "estrides": [1, 1], "interleave": "NONE",
            "base": [{"tensor": "x", "storage": [address, address + (1 << 20)], "elem_off": 0,
                      "match": "exact"}]}  # fmt: skip


def run(body, cluster=PAIR, family="cute", threads=64, tmap=False, align=1024, **pol):
    """Facts of a kernel with an optional descriptor parameter k_param_0 (over x), a .b32
    parameter and an extern shared array `smem` aligned to `align`; `pol`: more policy."""
    params = ([".param .align 64 .b8 k_param_0[128]"] if tmap else []) + [
        f".param .b32 k_param_{int(tmap)}"
    ]
    text = f"""
.version 9.3
.target sm_100a
.address_size 64
.extern .shared .align {align} .b8 smem[];
.visible .entry k({", ".join(params)})
.reqntid {threads}
{{
    .reg .pred %p<32>;
    .reg .b16 %rs<8>;
    .reg .b32 %r<100>;
    .reg .b64 %rd<16>;
    .loc 1 1 0
{body}
    ret;
}}
    .file 1 "{SRC}/k.py"
"""
    ps = [{"i": 0, "ptx": "k_param_0", "name": "t", "kind": "tensormap", "size": 128,
           "tensormap": _tmap()}] if tmap else []  # fmt: skip
    ps.append({"i": len(ps), "ptx": f"k_param_{len(ps)}", "name": "n", "kind": "scalar",
               "size": 4, "value": 7})  # fmt: skip
    rec = {
        "block": [threads, 1, 1],
        "cluster": list(cluster),
        "grid": [2, 1, 1],
        "params": ps,
    }
    pol = T.policy(family, src_root=SRC, params=T.param_table([rec], family, None, []),
                   tensors={"x": {"elem_bytes": 2}}, **pol)  # fmt: skip
    return T.translate(text, "k", [rec], pol)


def statements(stmts):
    for s in stmts:
        yield s
        for k in ("then", "else", "body"):
            yield from statements(s.get(k, []))


def events(f, ev=None, role=None):
    return [s for r in f.roles if role in (None, r["name"]) for s in statements(r["body"])
            if s["op"] == "event" and ev in (None, s["ev"])]  # fmt: skip


def refusal(f) -> str:
    return f.refusals[0]["msg"] if f.refusals else ""


def kir_errors(f) -> list:
    """frontend/kir.py's errors for a KIR around the Facts, with their cluster."""
    one = {"e": "const", "w": 64, "v": 1}
    big = {"e": "const", "w": 64, "v": 1 << 20}
    ts = [{"name": n, "kind": "global", "elem_bytes": t["elem_bytes"], "shape": [big],
           "strides": [one], "segments": None, "role": "input", "metadata": t["metadata"]}
          for n, t in f.tensors.items()]  # fmt: skip
    ts += [{"name": n, "kind": "desc", "elem_bytes": d["elem_bytes"], "box": d["box"],
            "shape": d["shape"], "strides": d["strides"], "segments": None, "role": "input",
            "metadata": False} for n, d in f.descs.items()]  # fmt: skip
    values = {**dict.fromkeys(f.params, 1), **(f.values[0] if f.values else {})}
    obj = {"kir_version": 1, "kernel": "k", "source": "k.py", "ir": "ptx", "target": "sm_100",
           "grid": [{"e": "const", "w": 32, "v": 2 * f.cluster[0]}, {"e": "const", "w": 32, "v": 1},
                    {"e": "const", "w": 32, "v": 1}],
           "params": [{"name": n, "type": f"i{w}"} for n, w in f.params.items()],
           "tensors": ts, "barriers": f.barriers,
           **({"cluster": list(f.cluster)} if f.cluster != (1, 1, 1) else {}),
           "roles": [{k: r[k] for k in ("name", "warps", "threads_per_issue", "body")} for r in f.roles],
           "case": {"name": "t", "inputs": [{"name": "i", "params": values, "metadata": {},
                                              "intended_writes": {}}]}}  # fmt: skip
    return kir.validate(obj)


def const(v, w=32):
    return {"e": "const", "w": w, "v": v}


def resolved(f, e):
    """e with every let variable replaced by its definition (the tests' lets are bound once)."""
    lets = {
        s["var"]: s["expr"]
        for r in f.roles
        for s in statements(r["body"])
        if s["op"] == "let"
    }

    def go(x):
        if x.get("e") == "var" and x["name"] in lets:
            return go(lets[x["name"]])
        return {k: go(v) if isinstance(v, dict) else v for k, v in x.items()}

    return go(e)


PEER = E.to_json(E.bin("xor", E.bin("urem", E.pid(0), E.const(2, 32)), E.const(1, 32)))


# thread id, smem, thread 0, warp, lane 0 of each warp: the elected lane (p2) and the
# elected lane of warp 0 (p4), the CTA rank and its peer
PRE = """
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, smem;
    setp.eq.u32 %p1, %r1, 0;
    shr.u32 %r30, %r1, 5;
    @%p1 mbarrier.init.shared::cta.b64 [%r2], 1;
    @%p1 mbarrier.init.shared::cta.b64 [%r2+8], 64;
    elect.sync %r3|%p2, -1;
    setp.eq.u32 %p3, %r30, 0;
    and.pred %p4, %p2, %p3;
    mov.u32 %r4, %cluster_ctarank;
    xor.b32 %r5, %r4, 1;
"""
# the descriptor's generic address, coordinates and the shared destination
TMA = """
    mov.b32 %r10, k_param_0;
    cvt.u64.u32 %rd1, %r10;
    cvta.param.u64 %rd2, %rd1;
    mov.b32 %r7, 32;
    add.s32 %r6, %r2, 1024;
    mov.u16 %rs1, 3;
    @%p4 mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r2], 4096;
"""
LOAD = "    @%p4 cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes{mods} [%r6], [%rd2, {{%r7, %r7}}], [{bar}]{mask};\n"


def load(bar="%r2", mods="", mask=""):
    return LOAD.format(mods=mods, bar=bar, mask=mask)


class SpecialRegisters(unittest.TestCase):
    def test_folded_from_the_cluster_dims(self):
        f = run(PRE + """
    mov.u32 %r11, %cluster_nctarank;
    sub.s32 %r12, %r11, 1;
    mapa.shared::cluster.u32 %r13, %r2, %r12;
    mov.pred %p5, %is_explicit_cluster;
    @%p5 mbarrier.arrive.release.cta.shared::cluster.b64 _, [%r13+8], 1;
    mov.u32 %r14, %cluster_ctaid.x;
    mapa.shared::cluster.u32 %r15, %r2, %r14;
    mbarrier.arrive.release.cta.shared::cluster.b64 _, [%r15+8], 1;
""", cluster=(2, 1, 1))  # fmt: skip
        self.assertEqual(f.refusals, [])
        a, b = events(f, "arrive")
        self.assertEqual((a["bar"], a["cta"]), ("mbar_8", const(1)))  # nctarank - 1
        self.assertEqual(resolved(f, b["cta"]), {"e": "bin", "w": 32, "o": "urem", "a": {"e": "pid", "w": 32, "axis": 0},
                                    "b": const(2)})  # fmt: skip
        self.assertEqual(kir_errors(f), [])

    def test_pair_mask(self):
        tr = SimpleNamespace(k=SimpleNamespace(cluster=PAIR))
        tr.sreg = lambda name: interp.Translator.sreg(tr, name)
        rank = {
            "e": "bin",
            "w": 32,
            "o": "urem",
            "a": {"e": "pid", "w": 32, "axis": 0},
            "b": const(2),
        }
        self.assertEqual(E.to_json(cluster.pair_mask(tr)), {
            "e": "bin", "w": 32, "o": "shl", "a": const(3),
            "b": {"e": "bin", "w": 32, "o": "and", "a": rank, "b": const(-2)}})  # fmt: skip


class RemoteArrive(unittest.TestCase):
    def test_arrive_and_expect_tx_on_the_peer(self):
        f = run(PRE + """
    mapa.shared::cluster.u32 %r11, %r2, %r5;
    mbarrier.arrive.release.cta.shared::cluster.b64 _, [%r11+8], 1;
    @%p1 mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%r11], 16;
    mbarrier.arrive.shared::cluster.b64 _, [%r2+8];
""")  # fmt: skip
        self.assertEqual(f.refusals, [])
        (role,) = f.roles
        a64, a1, own = events(f, "arrive")
        (x,) = events(f, "expect_tx")
        self.assertEqual((a64["bar"], resolved(f, a64["cta"])), ("mbar_8", PEER))
        self.assertEqual(
            (x["bar"], resolved(f, x["cta"]), x["bytes"]), ("mbar_0", PEER, const(16))
        )
        self.assertEqual((a1["bar"], resolved(f, a1["cta"])), ("mbar_0", PEER))
        self.assertNotIn("cta", own)  # a shared::cta address: the issuing CTA's slot
        # 64 arrivals per issue, one: threads_per_issue 1
        self.assertEqual((role["threads_per_issue"], a64["n"]), (1, 64))
        self.assertEqual(kir_errors(f), [])

    def test_a_rank_per_warp_splits_the_role(self):
        f = run(PRE + """
    mapa.shared::cluster.u32 %r11, %r2, %r30;
    mbarrier.arrive.release.cta.shared::cluster.b64 _, [%r11+8], 1;
""")  # fmt: skip
        self.assertEqual([r["name"] for r in f.roles], ["warps0_0", "warps1_1"])
        self.assertEqual([e["cta"] for e in events(f, "arrive")], [const(0), const(1)])
        self.assertEqual(kir_errors(f), [])

    def test_refusals(self):
        lane = (
            PRE
            + """
    and.b32 %r12, %r1, 1;
    mapa.shared::cluster.u32 %r11, %r2, %r12;
    mbarrier.arrive.release.cta.shared::cluster.b64 _, [%r11+8], 1;
"""
        )
        self.assertIn("operand 'cta' varies", refusal(run(lane)))
        f = run(PRE + """
    mapa.shared::cluster.u32 %r11, %r2, %r5;
    add.s32 %r12, %r11, 8;
    mbarrier.arrive.release.cta.shared::cluster.b64 _, [%r12], 1;
""")  # fmt: skip
        self.assertIn("mapa", refusal(f))  # arithmetic on a mapa result: data
        f = run(PRE + """
    mapa.shared::cluster.u32 %r11, %r2, %r5;
    @%p1 mbarrier.init.shared::cta.b64 [%r11+16], 1;
""")  # fmt: skip
        self.assertIn("mapa", refusal(f))  # init (and wait) only on the CTA's own slot
        f = run(PRE + "    @%p1 mbarrier.complete_tx.shared::cta.b64 [%r2], 16;\n")
        self.assertIn("complete_tx", refusal(f))

    def test_one_cta(self):
        body = (
            PRE
            + """
    mapa.shared::cluster.u32 %r11, %r2, %r4;
    mbarrier.arrive.release.cta.shared::cluster.b64 _, [%r11+8], 1;
"""
        )
        f = run(body, cluster=(1, 1, 1))
        self.assertEqual(f.refusals, [])
        self.assertNotIn("cta", events(f, "arrive")[0])
        self.assertEqual(kir_errors(f), [])
        f = run(body.replace("%r2, %r4;", "%r2, %r5;"), cluster=(1, 1, 1))
        self.assertIn("rank other than 0 in a cluster of one CTA", refusal(f))


CB = """
    barrier.cluster.arrive.relaxed.aligned;
    barrier.cluster.wait.aligned;
    @%p1 mbarrier.arrive.shared::cta.b64 _, [%r2];
    barrier.cluster.arrive;
    barrier.cluster.wait;
"""


class ClusterBarrier(unittest.TestCase):
    def test_arrive_and_wait(self):
        f = run(PRE + CB)
        self.assertEqual(f.refusals, [])
        self.assertEqual(f.barriers[-1], {"name": "cluster_barrier", "count": const(128),
                                          "num": const(1)})  # fmt: skip
        body = f.roles[0]["body"]
        self.assertEqual(
            body[0], {"op": "let", "var": "cluster_barrier_parity", "expr": const(0)}
        )
        cb = [e for e in events(f) if e.get("bar") == "cluster_barrier"]
        parity = {"e": "var", "w": 32, "name": "cluster_barrier_parity"}
        self.assertEqual([(e["ev"], e["cta"], e.get("parity"), e.get("n")) for e in cb],
                         [("arrive", const(0), None, 64), ("wait", const(0), parity, None)] * 2)  # fmt: skip
        flips = [
            s for s in statements(body) if s.get("var") == "cluster_barrier_parity"
        ]
        self.assertEqual(len(flips), 3)  # the prologue and one flip per wait
        self.assertEqual(
            flips[1]["expr"],
            {"e": "bin", "w": 32, "o": "xor", "a": parity, "b": const(1)},
        )
        self.assertEqual(kir_errors(f), [])

    def test_one_cta_and_refusals(self):
        f = run(PRE + CB, cluster=(1, 1, 1))
        self.assertEqual(f.refusals, [])
        self.assertTrue(all("cta" not in e for e in events(f, "wait")))
        self.assertEqual(kir_errors(f), [])
        f = run(PRE + "    @%p1 barrier.cluster.arrive;\n")
        self.assertIn("only part of the role", refusal(f))
        f = run(
            PRE
            + "    setp.lt.u32 %p6, %r1, 16;\n    @%p6 barrier.cluster.arrive.aligned;\n"
        )
        self.assertIn("differs between the lanes of a warp", refusal(f))
        f = run(PRE + CB, family="ttgir", cluster=(1, 1, 1))
        self.assertIn("Triton family is not in v1", refusal(f))
        self.assertEqual(f.refusals[0]["kind"], "gap")


class TmaLoad(unittest.TestCase):
    def test_leader(self):
        for mask, align, off in (("-16777224", 1024, 0), ("-16778240", 1024, 0),
                                 ("-16778232", 1024, 8), ("-16777224", 16, 0)):  # fmt: skip
            body = (
                PRE
                + TMA
                + f"    add.s32 %r8, %r2, {off};\n    and.b32 %r9, %r8, {mask};\n"
            )
            f = run(body + load("%r9", ".cta_group::2"), tmap=True, align=align)
            self.assertEqual(f.refusals, [], mask)
            (ld,) = events(f, "tma_load")
            self.assertEqual(
                (ld["bar"], ld["idx"], ld["leader"]), (f"mbar_{off}", const(0), True)
            )
            self.assertNotIn("cta_mask", ld)
            self.assertEqual(len(events(f, "gload")), 1)  # its footprint, once
            self.assertEqual(kir_errors(f), [])

    def test_cta_group_2_on_the_own_mbarrier(self):
        f = run(PRE + TMA + load("%r2", ".cta_group::2"), tmap=True)
        self.assertEqual(f.refusals, [])
        self.assertNotIn("leader", events(f, "tma_load")[0])

    def test_multicast(self):
        mc = ".multicast::cluster"
        f = run(PRE + TMA + load("%r2", mc, ", %rs1"), tmap=True)
        self.assertEqual(f.refusals, [])
        (ld,) = events(f, "tma_load")
        self.assertEqual((ld["cta_mask"], "leader" in ld), (const(3), False))
        self.assertEqual(kir_errors(f), [])
        body = (
            PRE
            + TMA
            + "    and.b32 %r9, %r2, -16777224;\n"
            + load("%r9", mc + ".cta_group::2", ", %rs1")
        )
        f = run(body, tmap=True)
        (ld,) = events(f, "tma_load")
        self.assertEqual((ld["cta_mask"], ld["leader"]), (const(3), True))
        self.assertEqual(kir_errors(f), [])

    def test_refusals(self):
        mc = ".multicast::cluster"
        cases = [
            # the mask clears bit 3 of offset 8: not the peer bit alone
            ("    add.s32 %r8, %r2, 8;\n    and.b32 %r9, %r8, -16778240;\n" + load("%r9", ".cta_group::2"),
             {}, "transformed by and.b32"),
            # an alignment of 16 does not fix bits 4..9 the mask clears
            ("    and.b32 %r9, %r2, -16778240;\n" + load("%r9", ".cta_group::2"), {"align": 16},
             "transformed by and.b32"),
            ("    and.b32 %r9, %r2, -16777224;\n" + load("%r9"), {}, "without .cta_group::2"),
            (load("%r2", ".cta_group::2"), {"cluster": (1, 1, 1)}, "without a cluster of CTA pairs"),
            (load("%r2", mc, ", %rs1"), {"cluster": (1, 1, 1)}, "outside a cluster of several CTAs"),
            (load("%r2", mc, ", %rs1").replace("@%p4", "@%p3"), {}, "several lanes of a warp"),
            # ... also on the own mbarrier, under a compiler whose plain loads were measured
            (load("%r2", ".cta_group::2").replace("@%p4", "@%p3"), {"ptxas": "12.9.83"},
             "several lanes of a warp"),
            (load("%r2").replace("[%r6]", "[%r11]"), {}, "in another CTA of the cluster"),
        ]  # fmt: skip
        for body, kw, why in cases:
            pre = PRE + TMA + "    mapa.shared::cluster.u32 %r11, %r6, %r5;\n"
            f = run(pre + body, tmap=True, **kw)
            self.assertIn(why, refusal(f), body)
        # into the issuing CTA through mapa: fine
        f = run(PRE + TMA + "    mapa.shared::cluster.u32 %r11, %r6, %r4;\n"
                + load("%r2").replace("[%r6]", "[%r11]"), tmap=True)  # fmt: skip
        self.assertEqual(f.refusals, [])


class Tcgen05(unittest.TestCase):
    def test_multicast_commit_and_pairs(self):
        body = (
            PRE
            + """
    mov.u16 %rs1, 3;
    @%p3 tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%r2+64], 512;
    @%p4 tcgen05.mma.cta_group::2.kind::f16 [%r20], %rd5, %rd6, %r24, %p14;
    @%p4 tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%r2], %rs1;
    @%p3 tcgen05.dealloc.cta_group::2.sync.aligned.b32 %r20, 512;
"""
        )
        f = run(body)
        self.assertEqual(f.refusals, [])
        (c,) = events(f, "commit_arrive")
        self.assertEqual((c["bar"], c["cta_mask"]), ("mbar_0", const(3)))
        self.assertEqual(kir_errors(f), [])
        self.assertIn(
            "without a cluster of CTA pairs", refusal(run(body, cluster=(1, 1, 1)))
        )
        self.assertIn(
            "without a cluster of CTA pairs", refusal(run(body, cluster=(3, 1, 1)))
        )
        f = run(PRE + "    @%p4 tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster"
                ".multicast::cluster.b64 [%r2], 1;\n", cluster=(1, 1, 1))  # fmt: skip
        self.assertIn("multicast tcgen05.commit outside a cluster", refusal(f))


S2C = """
    add.s32 %r6, %r2, 1024;
    mapa.shared::cluster.u32 %r11, %r6, %r5;
    mapa.shared::cluster.u32 %r12, %r2, {bar_rank};
    @%p4 cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%r11], [%r6], 512, [{bar}];
"""


class BulkS2Cluster(unittest.TestCase):
    def test_complete_tx_on_the_destination(self):
        f = run(PRE + S2C.format(bar_rank="%r5", bar="%r12"))
        self.assertEqual(f.refusals, [])
        (c,) = events(f, "complete_tx")
        self.assertEqual(
            (c["bar"], c["bytes"], resolved(f, c["cta"])), ("mbar_0", const(512), PEER)
        )
        self.assertEqual(kir_errors(f), [])

    def test_refusals(self):
        for kw, why, cl in [
            (
                {"bar_rank": "%r4", "bar": "%r12"},
                "completing on another's mbarrier",
                PAIR,
            ),
            ({"bar_rank": "%r5", "bar": "%r2"}, "not mapa results", PAIR),
            (
                {"bar_rank": "%r5", "bar": "%r12"},
                "outside a cluster of several CTAs",
                (1, 1, 1),
            ),
        ]:
            self.assertIn(why, refusal(run(PRE + S2C.format(**kw), cluster=cl)))
        f = run(PRE + S2C.format(bar_rank="%r5", bar="%r12").replace("@%p4", "@%p3"))
        self.assertIn("several lanes of a warp", refusal(f))


class InertBranches(unittest.TestCase):
    def test_on_data(self):
        body = (
            PRE
            + """
    ld.shared.u32 %r20, [smem+64];
    setp.eq.s32 %p5, %r20, 0;
    @%p5 bra OK;
    trap;
OK:
    @%p1 mbarrier.arrive.shared::cta.b64 _, [%r2];
"""
        )
        f = run(body)
        self.assertEqual((f.refusals, len(events(f, "arrive"))), ([], 1))
        # under a vote.sync of data, the same in all lanes of a warp: an arm of tcgen05 ops
        # only (P0 has no TMEM events: a 2-CTA CuTe kernel's rescale skip) or of a warp
        # collective keeps each warp together (.sync.aligned, H3)
        ballot = body.replace(
            "    setp.eq.s32 %p5, %r20, 0;\n",
            "    setp.lt.s32 %p6, %r20, 3;\n    vote.sync.ballot.b32 %r22, %p6, -1;\n"
            "    setp.eq.s32 %p5, %r22, 0;\n",
        )
        tc = "    tcgen05.ld.sync.aligned.32x32b.x1.b32 {%r21}, [%r20];\n"
        shfl = "    shfl.sync.idx.b32 %r23, %r1, 0, 31, -1;\n"
        for arm in (tc, shfl):
            f = run(ballot.replace("    trap;\n", arm))
            self.assertEqual((f.refusals, len(events(f, "arrive"))), ([], 1), arm)
        # not under a lane-varying condition on data (the trial would check them under
        # the enclosing mask), nor events, shared stores (a tensormap staging buffer's
        # zeroing: test_tma) or an arm the trial refuses: the branch condition is refused
        # (an instruction without a model, also inside a loop, whose refusal the trial
        # would report instead)
        inval = "    mbarrier.inval.shared::cta.b64 [%r2+16];\n"
        loop = f"    mov.b32 %r24, 0;\nLOOP:\n{inval}    add.s32 %r24, %r24, 1;\n" \
            "    setp.lt.s32 %p7, %r24, %r20;\n    @%p7 bra LOOP;\n"  # fmt: skip
        for arm in (tc, shfl, "    @%p1 mbarrier.arrive.shared::cta.b64 _, [%r2+8];\n",
                    "    st.shared.b32 [%r2+64], %r1;\n", inval, loop):  # fmt: skip
            f = run(body.replace("    trap;\n", arm))
            self.assertIn(
                "reaches a branch condition (slice soundness)", refusal(f), arm
            )

    def test_on_the_warp(self):
        stores = """
    setp.ne.s32 %p6, %r30, 1;
    @%p6 bra ELSE;
    st.shared.b32 [%r2+64], %r1;
    bra.uni JOIN;
ELSE:
    setp.ne.s32 %p7, %r30, 0;
    @%p7 bra JOIN;
    @%p2 mbarrier.init.shared::cta.b64 [%r2+16], 1;
JOIN:
    mbarrier.arrive.shared::cta.b64 _, [%r2+8];
"""
        # warp 1 only stores: no role of its own; the init splits warp 0 off
        f = run(PRE + stores, threads=128)
        self.assertEqual(f.refusals, [])
        self.assertEqual([r["name"] for r in f.roles], ["warps0_0", "warps1_3"])
        self.assertEqual(kir_errors(f), [])
        # warp 1 arrives: a role of its own
        f = run(PRE + stores.replace("st.shared.b32 [%r2+64], %r1", "mbarrier.arrive.shared::cta.b64 _, [%r2]"),
                threads=128)  # fmt: skip
        self.assertEqual(
            [r["name"] for r in f.roles], ["warps0_0", "warps1_1", "warps2_3"]
        )

    def test_attempt_leaves_nothing(self):
        # in a loop, warp 1's inert arm binds a loop-invariant branch condition, which a let
        # before the loop would hold: the attempt must not leave it there
        f = run(PRE + """
    ld.param.b32 %r40, [k_param_0];
    mov.b32 %r41, 0;
LOOP:
    setp.ne.s32 %p8, %r30, 1;
    @%p8 bra SKIP;
    mul.lo.s32 %r42, %r40, 3;
    setp.gt.s32 %p9, %r42, 5;
    @%p9 bra SKIP;
    st.shared.b32 [%r2+64], %r1;
SKIP:
    add.s32 %r41, %r41, 1;
    setp.lt.s32 %p10, %r41, 4;
    @%p10 bra LOOP;
    @%p1 mbarrier.arrive.shared::cta.b64 _, [%r2];
""")  # fmt: skip
        self.assertEqual(([r["name"] for r in f.roles], f.refusals), (["warps0_1"], []))
        lets = [s for s in statements(f.roles[0]["body"]) if s["op"] == "let"]
        self.assertFalse([s for s in lets if s["expr"].get("o") == "mul"], lets)


if __name__ == "__main__":
    unittest.main()
