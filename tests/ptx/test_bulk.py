# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the PTX frontend's P1-2 pass (frontend/ptx/bulk.py: "async store sources reuse
ordered" events on PTX) through translate() with policy p1 ["P1-2"] on hand-written PTX.

    python3 -I tests/ptx/test_bulk.py      (or the buck target :test_bulk)

Events: a bulk store by the elected lane of warp 0 (no `lane`), its commit and wait
groups (n, .read), named barriers, st.shared per thread (`warp`, `lane`), a write pruned
when it cannot meet a store source, a TMA store through an in-kernel descriptor with its
`tmap` and the cp_fenceproxy's `tmap_write`; the top-level `p1`, none without a bulk store;
the role split by a thread-0 arrive (parts named <partition>_w<a>_<b>); a load's landing
naming its mbarrier (`bar`, `idx`), its tma_load next; a shared::cluster bulk copy or
reduce into the peer CTA (an `async` smem_write with `cta`); the `let`s nothing
reads dropped (a split part's per-lane loads would make their tensors metadata). Refusals: a group
the issuer does not execute, a store several lanes of a warp issue, a write under a data
predicate that may overlap a source, an mbarrier object or tcgen05.alloc's slot inside a
source, an mbarrier wait by part of a warp, a wait or write under a negated elect.sync or
under elect.sync and a lane condition, a global store into a buffer holding the tensormaps
of bulk stores. Pruning keeps a loop whose lets carry values out of it, drops the loops
left empty and bounds a truncated offset as truncated; a landing whose load binds the
slot index after it names no slot. The P0 pass unchanged.
"""

import sys
import unittest
from pathlib import Path
from types import ModuleType

if __package__ in (None, ""):
    sys.dont_write_bytecode = True
    _root = ModuleType("lkv_root")
    _root.__path__ = [str(Path(__file__).resolve().parents[2])]
    sys.modules["lkv_root"] = _root
    __package__ = "lkv_root.tests.ptx"  # noqa: A001

from ...frontend.ptx import bulk, translate as T  # noqa: E402
from .fixtures.kernels import events, kir_errors, run  # noqa: E402

P12 = {"p1": ["P1-2"]}
# thread id, the shared array, warp, warp 0, the pointer x, 16 bytes per thread at smem
# + 16 tid, an elected lane of warp 0 (%p3)
PRE = """
    mov.u32 %r1, %tid.x;
    mov.u32 %r10, smem;
    ld.param.u64 %rd1, [k_param_0];
    shr.u32 %r3, %r1, 5;
    setp.eq.s32 %p11, %r3, 0;
    shl.b32 %r5, %r1, 4;
    add.s32 %r6, %r10, %r5;
    elect.sync %r7|%p2, -1;
    and.pred %p3, %p11, %p2;
"""
# the staging-buffer step: wait for the previous store's reads, sync, write, sync, store
STEP = """
    cp.async.bulk.wait_group.read 0;
    bar.sync 1, 128;
    st.shared.v4.b32 [%r6], {%r1, %r1, %r1, %r1};
    fence.proxy.async.shared::cta;
    bar.sync 1, 128;
    @%p3 cp.async.bulk.global.shared::cta.bulk_group [%rd1], [%r10], 2048;
    cp.async.bulk.commit_group;
"""


def p12(body, threads=128, **kw):
    return run(PRE + body + "\n    ret;\n", threads=threads, **P12, **kw)


def refusal(f) -> str:
    return f.refusals[0]["msg"] if f.refusals else ""


class Events(unittest.TestCase):
    def test_staging_step(self) -> None:
        f = p12(STEP)
        self.assertEqual((f.refusals, f.p1), ([], ["P1-2"]))
        names = [e["ev"] for e in events(f) if e["ev"] != "gstore"]
        self.assertEqual(names, ["bulk_wait", "named_bar", "smem_write", "named_bar",
                                 "bulk_store", "bulk_commit"])  # fmt: skip
        (w,) = events(f, "bulk_wait")
        self.assertEqual((w["n"], w["read"]), (0, True))
        (s,) = events(f, "bulk_store")
        self.assertEqual((s["kind"], s["sym"], s["bytes"], s["src"]["v"], s["warp"]["v"]),
                         ("bulk", "smem", 2048, 0, 0))  # fmt: skip
        self.assertNotIn("lane", s)  # the elected lane
        (x,) = events(f, "smem_write")  # in a loop over the 128 threads
        self.assertEqual((x["sym"], x["bytes"], "lane" in x, x.get("sync", False)),
                         ("smem", 16, True, False))  # fmt: skip
        self.assertEqual(kir_errors(f), [])

    def test_p0_unchanged(self) -> None:
        p0 = run(PRE + STEP + "\n    ret;\n", threads=128)
        self.assertEqual((p0.refusals, p0.p1), ([], []))
        self.assertEqual([e["ev"] for e in events(p0)], ["gstore"])

    def test_pruned_write_and_no_store(self) -> None:
        far = STEP.replace("st.shared.v4.b32 [%r6]", "st.shared.v4.b32 [%r6+4096]")
        f = p12(far)
        self.assertEqual((f.refusals, events(f, "smem_write")), ([], []))
        f = p12("    st.shared.v4.b32 [%r6], {%r1, %r1, %r1, %r1};")
        self.assertEqual(
            (f.refusals, f.p1), ([], [])
        )  # no bulk store: nothing to check

    def test_split_by_thread0_arrive(self) -> None:
        body = ("    setp.eq.s32 %p12, %r1, 0;\n"
                "    @%p12 mbarrier.init.shared::cta.b64 [%r10+4096], 1;\n"
                "    @%p12 mbarrier.arrive.shared::cta.b64 _, [%r10+4096];\n" + STEP)  # fmt: skip
        f = p12(body)
        self.assertEqual(f.refusals, [])
        self.assertEqual([(r["name"], r["warps"]) for r in f.roles],
                         [("default_w0_0", 1), ("default_w1_3", 3)])  # fmt: skip
        self.assertEqual(kir_errors(f), [])
        p0 = run(PRE + body + "\n    ret;\n", threads=128)
        self.assertEqual(
            [r["name"] for r in p0.roles], ["default"]
        )  # P0 keeps its role

    def test_loop_of_lets_kept(self) -> None:
        # the loop's write (smem+8192) is pruned, but its lets carry %r25 (8 n after the
        # loop) out of it to the later write's offset (smem [0, 2048) for every n, the
        # store's source): the loop stays, or the offset would read n
        body = """
    ld.param.u32 %r30, [k_param_1];
    mov.u32 %r20, 0;
    mov.u32 %r25, %r30;
LOOP:
    st.shared.u32 [%r10+8192], %r20;
    mul.lo.s32 %r25, %r25, 2;
    add.s32 %r20, %r20, 1;
    setp.lt.s32 %p20, %r20, 3;
    @%p20 bra LOOP;
    @%p3 cp.async.bulk.global.shared::cta.bulk_group [%rd1], [%r10], 2048;
    cp.async.bulk.commit_group;
    rem.u32 %r21, %r25, 2;
    mul.lo.s32 %r22, %r21, 2048;
    add.s32 %r23, %r10, %r22;
    add.s32 %r24, %r23, %r5;
    st.shared.v4.b32 [%r24], {%r1, %r1, %r1, %r1};
"""
        f = p12(body)
        self.assertEqual(f.refusals, [])
        (r,) = f.roles
        loop = next(s for s in r["body"] if s["op"] == "for")
        self.assertEqual({s["op"] for s in loop["body"]}, {"let"})
        self.assertLess(
            r["body"].index(loop), [s.get("ev") for s in r["body"]].index("bulk_store")
        )
        self.assertEqual(kir_errors(f), [])

    def test_truncated_offset(self) -> None:
        # (n & 1) 2048 + 2^32 in 64 bits, truncated to 32: smem [0, 16) or [2048, 2064), the
        # first inside the store's source (not pruned as the untruncated [2^32, 2^32 + 2064))
        body = """
    ld.param.u32 %r30, [k_param_1];
    @%p3 cp.async.bulk.global.shared::cta.bulk_group [%rd1], [%r10], 2048;
    @%p3 cp.async.bulk.commit_group;
    and.b32 %r31, %r30, 1;
    mul.wide.u32 %rd3, %r31, 2048;
    add.u64 %rd5, %rd3, 4294967296;
    cvt.u32.u64 %r21, %rd5;
    add.u32 %r22, %r10, %r21;
    @%p3 st.shared.v4.b32 [%r22], {%r1, %r1, %r1, %r1};
    @%p3 cp.async.bulk.wait_group.read 0;
"""
        f = p12(body)
        self.assertEqual(f.refusals, [])
        self.assertEqual(len(events(f, "smem_write")), 1)
        self.assertEqual(kir_errors(f), [])


class Stmatrix(unittest.TestCase):
    def test_rows(self) -> None:
        body = STEP.replace("    st.shared.v4.b32 [%r6], {%r1, %r1, %r1, %r1};",
                            "    stmatrix.sync.aligned.m8n8.x1.shared.b16 [%r6], {%r1};")  # fmt: skip
        f = p12(body)
        self.assertEqual(f.refusals, [])
        (x,) = events(f, "smem_write")  # lanes 0..7 of every warp give a 16-byte row
        self.assertEqual((x["bytes"], x.get("sync"), "lane" in x), (16, True, False))
        self.assertEqual(kir_errors(f), [])


class RemoteWrite(unittest.TestCase):
    # a 2-CTA cluster: thread 0 copies 512 bytes into the peer CTA's smem+0 (a DSMEM bulk
    # copy completing on the peer's mbarrier) and stores its own smem [0, 2048) to global
    BODY = """
    mov.u32 %r1, %tid.x;
    mov.u32 %r10, smem;
    ld.param.u64 %rd1, [k_param_0];
    setp.eq.s32 %p12, %r1, 0;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+4096], 1;
    mov.u32 %r30, %cluster_ctarank;
    xor.b32 %r31, %r30, 1;
    mapa.shared::cluster.u32 %r32, %r10, %r31;
    add.s32 %r35, %r10, 4096;
    mapa.shared::cluster.u32 %r33, %r35, %r31;
    @%p12 cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%r32], [%r10+8192], 512, [%r33];
    @%p12 cp.async.bulk.global.shared::cta.bulk_group [%rd1], [%r10], 2048;
    @%p12 cp.async.bulk.commit_group;
    ret;
"""

    def test_into_the_peer(self) -> None:
        f = run(self.BODY, family="cute", threads=32, cluster=(2, 1, 1), **P12)
        self.assertEqual((f.refusals, f.p1), ([], ["P1-2"]))
        (x,) = events(f, "smem_write")
        self.assertEqual((x["bytes"], x.get("async"), "cta" in x), (512, True, True))
        self.assertEqual(kir_errors(f), [])

    def test_reduce_into_the_peer(self) -> None:
        # a bulk reduce into the peer (ISA cp.reduce.async.bulk .shared::cluster) writes
        # its destination bytes as the copy does
        body = self.BODY.replace(
            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes",
            "cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
            ".max.s32",
        )
        f = run(body, family="cute", threads=32, cluster=(2, 1, 1), **P12)
        self.assertEqual((f.refusals, f.p1), ([], ["P1-2"]))
        (c,) = events(f, "complete_tx")
        self.assertEqual(c["bytes"]["v"], 512)
        (x,) = events(f, "smem_write")
        self.assertEqual((x["bytes"], x.get("async"), "cta" in x), (512, True, True))
        self.assertEqual(kir_errors(f), [])


class Refusals(unittest.TestCase):
    def test_group_not_executed_by_the_issuer(self) -> None:
        body = STEP.replace("    cp.async.bulk.wait_group.read 0;",
                            "    setp.eq.s32 %p13, %r3, 1;\n    @%p13 cp.async.bulk.wait_group.read 0;")  # fmt: skip
        self.assertIn("does not execute", refusal(p12(body)))
        # warp 0 alone executes it: the issuer does (Triton's descriptor copy)
        body = STEP.replace("    cp.async.bulk.wait_group.read 0;",
                            "    @%p11 cp.async.bulk.wait_group.read 0;")  # fmt: skip
        self.assertEqual(p12(body).refusals, [])

    def test_store_by_several_lanes(self) -> None:
        body = STEP.replace("@%p3 cp.async.bulk.global", "@%p11 cp.async.bulk.global")
        self.assertIn("several lanes of a warp", refusal(p12(body)))

    def test_elected_lane_unknown(self) -> None:
        # which lane elect.sync elects is unspecified (H1): the lanes of warp 0 but the
        # elected one (%p4) wait or write, or the elected lane waits if it is below 16
        # (%p9: lane 0 would be credited, lane 16 not); the elected lane itself (%p3) is not
        # refused
        pre = """
    not.pred %p5, %p2;
    and.pred %p4, %p11, %p5;
    mov.u32 %r8, %laneid;
    setp.lt.u32 %p8, %r8, 16;
    and.pred %p9, %p3, %p8;
    @%p3 cp.async.bulk.global.shared::cta.bulk_group [%rd1], [%r10], 2048;
    @%p3 cp.async.bulk.commit_group;
"""
        tail = """
    bar.sync 1, 128;
    st.shared.v4.b32 [%r6], {%r1, %r1, %r1, %r1};
"""
        wait = "    @{p} cp.async.bulk.wait_group.read 0;"
        f = p12(pre + wait.format(p="%p4") + tail)
        self.assertIn("P1-2: cp.async.bulk.wait_group.read under the negation of elect.sync",
                      refusal(f))  # fmt: skip
        f = p12(pre + wait.format(p="%p3") + "\n    @%p4 st.shared.u32 [%r10+64], %r1;")
        self.assertIn("P1-2: st.shared under the negation of elect.sync", refusal(f))
        f = p12(pre + wait.format(p="%p9") + tail)
        self.assertIn(
            "also guarded by a condition that depends on the lane", refusal(f)
        )
        f = p12(pre + wait.format(p="%p3") + tail)
        self.assertEqual((f.refusals, f.p1), ([], ["P1-2"]))

    def test_data_predicated_write(self) -> None:
        pred = "    ld.shared.u32 %r20, [%r10+8192];\n    setp.eq.s32 %p14, %r20, 0;\n"  # data
        near = pred + STEP.replace("    st.shared.v4", "    @%p14 st.shared.v4")
        self.assertIn("predicate that depends on data", refusal(p12(near)))
        far = near.replace("st.shared.v4.b32 [%r6]", "st.shared.v4.b32 [%r6+4096]")
        self.assertEqual(p12(far).refusals, [])

    def test_unmodeled_writes_inside_a_source(self) -> None:
        bar = ("    setp.eq.s32 %p12, %r1, 0;\n"
               "    @%p12 mbarrier.init.shared::cta.b64 [%r10+64], 1;\n" + STEP)  # fmt: skip
        self.assertIn("mbarrier object", refusal(p12(bar)))
        alloc = ("    mov.b32 %r9, 512;\n"
                 "    @%p11 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r10+16], %r9;\n"
                 + STEP)  # fmt: skip
        self.assertIn("tcgen05.alloc's address slot", refusal(p12(alloc)))

    def test_wait_by_part_of_a_warp(self) -> None:
        body = """
    setp.eq.s32 %p12, %r1, 0;
    @%p12 mbarrier.init.shared::cta.b64 [%r10+4096], 1;
    @%p12 mbarrier.arrive.shared::cta.b64 _, [%r10+4096];
    @!%p12 bra DONE;
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 %p5, [%r10+4096], 0;
    @!%p5 bra WAIT;
DONE:
""" + STEP  # fmt: skip
        self.assertIn("part of a warp", refusal(p12(body)))


# In-kernel descriptor (tests/ptx/test_tma.py's): warp 0 zeroes the staging buffer
# smem+4096, thread 0 replaces the fields, the warp copies it to this CTA's scratch slot,
# the elected lane stores the box from smem+8192 through it, twice (two work items), the
# second rebuild after the first store with or without a full wait and named barrier first.
_DECL = {
    "ptr": ".param .u64 .ptr .global .align 1 k_param_{i}",
    "b32": ".param .b32 k_param_{i}",
}
KP = [
    ("ptr", "dv", 0x20000),
    ("b32", "rows", 70),
    ("ptr", "__global_scratch", "scratch"),
]
STAGE = """
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, smem;
    ld.param.u64 %rd10, [k_param_2];
    mov.u32 %r30, %ctaid.x;
    mul.wide.u32 %rd11, %r30, 256;
    add.s64 %rd12, %rd10, %rd11;
    ld.param.b32 %r5, [k_param_1];
    add.s32 %r40, %r2, 4096;
    shl.b32 %r41, %r1, 2;
    add.s32 %r42, %r40, %r41;
    mov.b32 %r43, 0;
    setp.eq.u32 %p1, %r1, 0;
    elect.sync %r3|%p2, -1;
"""
ITEM = """
    st.shared.b32 [%r42], %r43;
    @!%p1 bra SKIP{k};
    cvt.u64.u32 %rd15, %r40;
    ld.param.u64 %rd14, [k_param_0];
    tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%rd15], %rd14;
    tensormap.replace.tile.rank.shared::cta.b1024.b32 [%rd15], 0x1;
    tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%rd15], 0x0, 32;
    tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%rd15], 0x1, 64;
    tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%rd15], 0x0, 128;
    tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%rd15], 0x1, %r5;
    tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%rd15], 0x0, 256;
    tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [%rd15], 0x0, 1;
    tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [%rd15], 0xa;
    tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [%rd15], 0x2;
SKIP{k}:
    cvt.u64.u32 %rd16, %r40;
    tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%rd12], [%rd16], 0x80;
    cvta.global.u64 %rd21, %rd12;
    mov.b32 %r50, 0;
    mov.b32 %r51, 0;
    add.s32 %r52, %r2, 8192;
    @%p2 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd21, {{%r50, %r51}}], [%r52];
    cp.async.bulk.commit_group;
"""


def in_kernel(drain: str, kp=KP):
    decl = ", ".join(_DECL[k].format(i=i) for i, (k, _n, _v) in enumerate(kp))
    body = STAGE + ITEM.format(k=0) + drain + ITEM.format(k=1)
    text = f"""
.version 9.3
.target sm_100a
.address_size 64
.extern .shared .align 1024 .b8 smem[];
.visible .entry k({decl})
.reqntid 32
{{
    .reg .pred %p<32>;
    .reg .b32 %r<100>;
    .reg .b64 %rd<64>;
    .loc 1 1 0
{body}
    ret;
}}
    .file 1 "/src/k.py"
"""
    ps = [{"i": i, "ptx": f"k_param_{i}", "name": n, "kind": "ptr" if k == "ptr" else "scalar",
           "size": 8 if k == "ptr" else 4, "value": None if v == "scratch" else v,
           **({"scratch": {"bytes": 1024, "align": 128}} if v == "scratch" else {})}
          for i, (k, n, v) in enumerate(kp)]  # fmt: skip
    recs = [
        {"block": [32, 1, 1], "cluster": [1, 1, 1], "grid": [1, 1, 1], "params": ps}
    ]
    tensors = {n: {"elem_bytes": 2 if n == "dv" else 1}
               for k, n, v in kp if k == "ptr" and v != "scratch"}  # fmt: skip
    pol = T.policy("ttgir", src_root="/src", params=T.param_table(recs, "ttgir", None, []),
                   tensors=tensors, **P12)  # fmt: skip
    return T.translate(text, "k", recs, pol)


class InKernelDescriptor(unittest.TestCase):
    def test_tmap_events(self) -> None:
        f = in_kernel("    cp.async.bulk.wait_group 0;\n    bar.sync 1, 32;\n")
        self.assertEqual((f.refusals, f.p1), ([], ["P1-2"]))
        tw = events(f, "tmap_write")
        self.assertEqual([(e["base"], e["warp"]["v"]) for e in tw],
                         [("__global_scratch", 0)] * 2)  # fmt: skip
        stores = events(f, "bulk_store")
        self.assertEqual([(s["kind"], s["bytes"], s["src"]["v"], s["tmap"]["base"]) for s in stores],
                         [("tensor", 32 * 64 * 2, 8192, "__global_scratch")] * 2)  # fmt: skip
        self.assertEqual([e["n"] for e in events(f, "bulk_wait")], [0])
        # the staging buffer's zeroing (smem+4096) cannot meet the source [8192, 12288)
        self.assertEqual(events(f, "smem_write"), [])

    def test_global_store_into_the_tensormap(self) -> None:
        # the descriptors in a buffer of the kernel's own (`maps`, not the scratch): thread 0
        # overwrites the tensormap item 0's store reads before its wait; P1-2's tensormap
        # writes are the cp_fenceproxy ones only
        kp = KP[:2] + [("ptr", "maps", 0x30000)]
        drain = "    cp.async.bulk.wait_group 0;\n    bar.sync 1, 32;\n"
        f = in_kernel(drain, kp)
        self.assertEqual((f.refusals, f.p1), ([], ["P1-2"]))
        self.assertEqual({s["tmap"]["base"] for s in events(f, "bulk_store")}, {"maps"})
        f = in_kernel("    @%p1 st.global.b32 [%rd12], %r1;\n" + drain, kp)
        self.assertIn("P1-2: a global write into maps", refusal(f))


class Landing(unittest.TestCase):
    # the elected lane of warp 0 loads 2048 bytes into smem [0, 2048) on the mbarrier at
    # smem+4096, the warp waits for it, then the lane stores smem [0, 2048)
    BODY = """
    @%p3 mbarrier.init.shared::cta.b64 [%r10+4096], 1;
    @%p3 mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r10+4096], 2048;
    @%p3 cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%r10], [%rd1], 2048, [%r10+4096];
    mov.b32 %r20, 0;
{
    .reg .pred P1;
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 P1, [%r10+4096], %r20;
    @!P1 bra.uni WAIT;
}
    @%p3 cp.async.bulk.global.shared::cta.bulk_group [%rd1], [%r10], 2048;
    @%p3 cp.async.bulk.commit_group;
"""

    def test_names_its_load_mbarrier(self) -> None:
        f = p12(self.BODY, threads=32)
        self.assertEqual(f.refusals, [])
        (x,) = events(f, "smem_write")
        (t,) = events(f, "tma_load")
        self.assertEqual(
            (x.get("async"), x["bar"], x["idx"]), (True, t["bar"], t["idx"])
        )
        names = [
            e["ev"] for e in events(f) if e["ev"] in ("smem_write", "gload", "tma_load")
        ]
        self.assertEqual(
            names, ["smem_write", "gload", "tma_load"]
        )  # its load follows it

    def test_slot_index_bound_after(self) -> None:
        # the load's mbarrier slot index (i & 1) is bound only after the landing, where
        # the copy's address and mbarrier are translated: the landing names no slot
        # (otherwise the P1 KIR reads that variable before its `let`)
        body = """
    @%p3 mbarrier.init.shared::cta.b64 [%r10+8192], 1;
    @%p3 mbarrier.init.shared::cta.b64 [%r10+8200], 1;
    ld.param.b32 %r40, [k_param_1];
    mov.b32 %r41, 0;
LOOP:
    and.b32 %r42, %r41, 1;
    shl.b32 %r43, %r42, 3;
    add.s32 %r44, %r10, %r43;
    cp.async.bulk.wait_group.read 0;
    @%p3 cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%r10], [%rd1], 2048, [%r44+8192];
    @%p3 mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r44+8192], 2048;
    shr.u32 %r47, %r41, 1;
    and.b32 %r48, %r47, 1;
{
    .reg .pred P1;
WAIT:
    mbarrier.try_wait.parity.shared::cta.b64 P1, [%r44+8192], %r48;
    @!P1 bra.uni WAIT;
}
    @%p3 cp.async.bulk.global.shared::cta.bulk_group [%rd1], [%r10], 2048;
    @%p3 cp.async.bulk.commit_group;
    add.s32 %r41, %r41, 1;
    setp.lt.s32 %p20, %r41, %r40;
    @%p20 bra LOOP;
"""
        f = p12(body, threads=32)
        self.assertEqual(f.refusals, [])
        (x,) = events(f, "smem_write")
        self.assertEqual((x.get("async"), "bar" in x, "idx" in x), (True, False, False))
        # (the fixture declares every tensor global, the bulk load's descriptor too)
        self.assertEqual(
            [e for e in kir_errors(f) if "is not a desc tensor" not in e], []
        )


class DeadLets(unittest.TestCase):
    def test_lets_nothing_reads_are_dropped(self):
        def var(n):
            return {"e": "var", "w": 32, "name": n}

        def load(t):
            return {
                "e": "load",
                "w": 32,
                "tensor": t,
                "index": [{"e": "const", "w": 32, "v": 0}],
            }

        stmts = [
            {"op": "let", "var": "a", "expr": load("T")},  # read by b only
            {"op": "let", "var": "b", "expr": {"e": "bin", "w": 32, "o": "add", "a": var("a"), "b": var("a")}},
            {"op": "let", "var": "c", "expr": load("U")},  # read by the event
            {"op": "let", "var": "d", "expr": {"e": "const", "w": 32, "v": 1}},  # by the if
            {"op": "if", "cond": {"e": "cmp", "w": 1, "o": "eq", "a": var("d"), "b": var("d")},
             "then": [{"op": "let", "var": "e", "expr": load("V")},
                      {"op": "let", "var": "c", "expr": var("c")}], "else": []},
            {"op": "event", "ev": "bulk_wait", "n": var("c")},
        ]  # fmt: skip
        bulk._drop_dead_lets(stmts)
        self.assertEqual([s.get("var") for s in stmts], ["c", "d", None, None])
        self.assertEqual([s["var"] for s in stmts[2]["then"]], ["c"])

    def test_loop_left_empty_dropped(self) -> None:
        # the loop's write (smem+8192+(4i & 15)) is pruned and its address `let` then
        # reads nothing else: the loop, left empty, goes too
        body = """
    mov.u32 %r20, 0;
LOOP:
    shl.b32 %r21, %r20, 2;
    and.b32 %r22, %r21, 15;
    add.s32 %r23, %r10, %r22;
    st.shared.u32 [%r23+8192], %r20;
    add.s32 %r20, %r20, 1;
    setp.lt.s32 %p20, %r20, 4;
    @%p20 bra LOOP;
    @%p3 cp.async.bulk.global.shared::cta.bulk_group [%rd1], [%r10], 2048;
    cp.async.bulk.commit_group;
"""
        f = p12(body)
        self.assertEqual(f.refusals, [])
        (r,) = f.roles
        self.assertEqual([s["op"] for s in r["body"]], ["event"] * 3)
        self.assertEqual(kir_errors(f), [])


if __name__ == "__main__":
    unittest.main()
