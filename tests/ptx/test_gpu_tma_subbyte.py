# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""GPU truth for packed sub-byte TMA: the rules frontend/ptx/tma.py (_PACKED) and
launch.kir_desc apply to the CUtensorMap formats 16U4_ALIGN8B, 16U4_ALIGN16B and
16U6_ALIGN16B.

    python3 -I tests/ptx/test_gpu_tma_subbyte.py   (or the buck target :test_gpu_tma_subbyte)

Needs libcuda.so.1 of a real driver and a visible sm_100+ GPU; skips otherwise. One thread
issues each copy of hand-written PTX, compiled by the driver (who issues is
test_gpu_tma_issue.py's subject); every CUDA call runs in a forked child (a trap ends the
CUDA context it happens in, and none survives a fork), the expectations in this process:
a descriptor of cuTensorMapEncodeTiled over a 2-D tensor of 4- or 6-bit elements, decoded by
launch.decode and mapped by launch.kir_desc onto a byte tensor, as the frontend does.
  transaction bytes  an mbarrier expecting E bytes completes, once the copy landed, after
                     E - tx one-byte complete_tx steps (the next phase with none): tx is the
                     KIR box's bytes, the packed global bytes (also for the formats whose
                     shared-memory copy is wider), for full boxes, boxes clipped in either
                     dim and boxes wholly outside; in each CTA of a 2-CTA multicast; twice on
                     the pair leader's mbarrier with .cta_group::2 (each CTA loading, or a
                     multicast), none on the peer's;
  load footprint     the shared-memory image: a KIR box element inside the extents holds the
                     global byte off + sum (c + i) * stride (the dim-0 coordinate converted
                     by dim0_ratio), one outside zero; 16U4_ALIGN16B / 16U6_ALIGN16B put each
                     8 / 12 packed bytes at the start of 16;
  store footprint    the global bytes a store writes are the KIR box elements inside the
                     extents;
  traps              CUDA_ERROR_ILLEGAL_INSTRUCTION for a dim-0 coordinate that is not a
                     multiple of _PACKED's granularity (half of it here), a store _PACKED does
                     not admit (16U4_ALIGN16B) and a reduce (any); a byte format (UINT8) traps
                     at a dim-0 coordinate of 8 and not of 16: the 16 global bytes that
                     16U4_ALIGN8B's granularity of 32 is.
"""

import ctypes as C
import itertools
import json
import math
import multiprocessing
import random
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

from ...frontend.ptx import launch, tma  # noqa: E402

# host CUtensorMapDataType, element bits
FORMATS = {"16U4_ALIGN8B": (13, 4), "16U4_ALIGN16B": (14, 4), "16U6_ALIGN16B": (15, 6),
           "UINT8": (0, 8)}  # fmt: skip
PACKED = [f for f in FORMATS if f in tma._PACKED]
W, R, BOX = 256, 16, [128, 8]  # tensor: W elements a row, R rows; the box
SM = 4096  # shared bytes of the destination
# the mbarrier's expected bytes: above any count (and the probe's step limit)
E = 8 * math.prod(BOX)
SRC = bytes(1 if b == 0 else 0xED if b == 0xEE else b
            for b in random.Random(20261106).randbytes(1 << 16))  # fmt: skip
ILLEGAL_INSTRUCTION = 715


def granularity(fmt: str) -> int:
    """The multiple of its elements a dim-0 coordinate must be (16 bytes for UINT8)."""
    return tma._PACKED[fmt][0] if fmt in tma._PACKED else 16


def row_bytes(fmt: str) -> int:
    return W * FORMATS[fmt][1] // 8  # a multiple of 32: the stride, dense


# ------------------------------------------------------------------------------- PTX
TL = "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"


def _kernel(name: str, body: str, cluster: bool = False) -> str:
    """Thread 0 issues; out + 16 * cluster rank + 16: the probe (steps, completed, the next
    phase completed with no bytes); out + 256: the shared image (loads)."""
    return f"""
.visible .entry {name}(.param .align 64 .b8 tm[128], .param .u64 p_out, .param .u32 p_e,
    .param .u32 p_c0, .param .u32 p_c1)
.reqntid 32
{".reqnctapercluster 2, 1, 1" if cluster else ""}
{{
    .reg .pred %p<16>;
    .reg .b32 %r<64>;
    .reg .b16 %h<2>;
    .reg .b64 %rd<40>;
    .shared .align 1024 .b8 smem[{SM}];
    .shared .align 8 .b64 mbar;
    mov.u32 %r0, %tid.x;
    ld.param.u64 %rd0, [p_out];
    cvta.to.global.u64 %rd0, %rd0;
    ld.param.u32 %r3, [p_e];
    ld.param.u32 %r1, [p_c0];
    ld.param.u32 %r2, [p_c1];
    mov.u32 %r4, smem;
    mov.u32 %r5, mbar;
    mov.b64 %rd1, tm;
    cvta.param.u64 %rd1, %rd1;
    setp.eq.u32 %p0, %r0, 0;
    mov.u32 %r21, %cluster_ctarank;
    mapa.shared::cluster.u32 %r20, %r5, 0;
    shl.b32 %r6, %r0, 2;
    mov.u32 %r9, 0xEEEEEEEE;
FILL:
    add.u32 %r7, %r4, %r6;
    st.shared.u32 [%r7], %r9;
    add.u32 %r6, %r6, 128;
    setp.lt.u32 %p1, %r6, {SM};
    @%p1 bra FILL;
    @%p0 mbarrier.init.shared::cta.b64 [%r5], 1;
    @%p0 mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%r5], %r3;
    fence.mbarrier_init.release.cluster;
    fence.proxy.async.shared::cta;
    {"barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" if cluster else "bar.sync 0;"}
{body}
    ret;
}}
"""


def _probe() -> str:
    return """
    mul.lo.u32 %r22, %r21, 16;
    cvt.u64.u32 %rd32, %r22;
    add.u64 %rd2, %rd0, %rd32;
    @!%p0 bra NOPROBE;
    mov.u64 %rd25, %clock64;
S1:
    mov.u64 %rd26, %clock64;
    sub.s64 %rd27, %rd26, %rd25;
    setp.lt.s64 %p9, %rd27, 8000000;
    @%p9 bra S1;
    mbarrier.arrive.shared::cta.b64 %rd23, [%r5];
    mov.u32 %r40, 0;
PROBE:
    mbarrier.test_wait.parity.shared::cta.b64 %p5, [%r5], 0;
    @%p5 bra PROBED;
    setp.ge.u32 %p6, %r40, %r3;
    @%p6 bra PROBED;
    mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%r5], 1;
    add.u32 %r40, %r40, 1;
    bra PROBE;
PROBED:
    st.global.u32 [%rd2+16], %r40;
    selp.u32 %r41, 1, 0, %p5;
    st.global.u32 [%rd2+20], %r41;
    mbarrier.arrive.shared::cta.b64 %rd24, [%r5];
    mbarrier.test_wait.parity.shared::cta.b64 %p7, [%r5], 1;
    selp.u32 %r42, 1, 0, %p7;
    st.global.u32 [%rd2+24], %r42;
NOPROBE:
    bar.sync 0;
"""


_COPY = f"""
    shl.b32 %r50, %r0, 2;
COPY:
    add.u32 %r52, %r4, %r50;
    ld.shared.u32 %r53, [%r52];
    cvt.u64.u32 %rd30, %r50;
    add.u64 %rd31, %rd0, %rd30;
    st.global.u32 [%rd31+256], %r53;
    add.u32 %r50, %r50, 128;
    setp.lt.u32 %p8, %r50, {SM};
    @%p8 bra COPY;
"""
_CS = (
    "barrier.cluster.arrive.release.aligned;\n    barrier.cluster.wait.acquire.aligned;"
)
_OUT = "cp.async.bulk.commit_group;\n    @%p0 cp.async.bulk.wait_group 0;"
# kind -> (the copy and what follows it, a 2-CTA cluster); shared memory starts as 0xEE
KERNELS = {
    "ld": (f"    @%p0 {TL} [%r4], [%rd1, {{%r1, %r2}}], [%r5];" + _probe() + _COPY, False),
    "st": ("    @%p0 cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group "
           f"[%rd1, {{%r1, %r2}}], [%r4];\n    @%p0 {_OUT}", False),
    "rd": ("    @%p0 cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group "
           f"[%rd1, {{%r1, %r2}}], [%r4];\n    @%p0 {_OUT}", False),
    "mc": ("    setp.eq.u32 %p2, %r21, 0;\n    and.pred %p3, %p0, %p2;\n    mov.u16 %h0, 3;\n"
           f"    @%p3 {TL}.multicast::cluster [%r4], [%rd1, {{%r1, %r2}}], [%r5], %h0;"
           + _probe() + _CS, True),
    "mccg": ("    setp.eq.u32 %p2, %r21, 0;\n    and.pred %p3, %p0, %p2;\n    mov.u16 %h0, 3;\n"
             f"    @%p3 {TL}.multicast::cluster.cta_group::2 [%r4], [%rd1, {{%r1, %r2}}], [%r20], %h0;"
             + _probe() + _CS, True),
    "cg": (f"    @%p0 {TL}.cta_group::2 [%r4], [%rd1, {{%r1, %r2}}], [%r20];" + _probe() + _CS,
           True),
}  # fmt: skip


def module(target: str) -> str:
    ver = "8.7" if target == "sm_100a" else "8.8"  # 8.8 introduced sm_103a
    ks = [_kernel(kind, body, cluster) for kind, (body, cluster) in KERNELS.items()]
    return f".version {ver}\n.target {target}\n.address_size 64\n" + "".join(ks)


# --------------------------------------------------------------- the child's CUDA
class _Cuda:
    def __init__(self):
        self.cu = cu = C.CDLL("libcuda.so.1")
        dev, ctx, mj, mn = C.c_int(), C.c_void_p(), C.c_int(), C.c_int()
        if (cu.cuInit(0) or cu.cuDeviceGet(C.byref(dev), 0)
                or cu.cuDevicePrimaryCtxRetain(C.byref(ctx), dev) or cu.cuCtxSetCurrent(ctx)):  # fmt: skip
            raise OSError("no visible GPU")
        cu.cuDeviceGetAttribute(C.byref(mj), 75, dev)  # compute capability major
        cu.cuDeviceGetAttribute(C.byref(mn), 76, dev)
        self.cc = mj.value * 10 + mn.value
        U32, U64 = C.c_uint32, C.c_uint64
        cu.cuTensorMapEncodeTiled.argtypes = [
            C.c_void_p, C.c_int, U32, C.c_void_p, C.POINTER(U64), C.POINTER(U64),
            C.POINTER(U32), C.POINTER(U32), C.c_int, C.c_int, C.c_int, C.c_int]  # fmt: skip
        self.mod = C.c_void_p()

    def ok(self, rc):
        if rc:
            raise RuntimeError(rc)

    def load(self):
        src = module(f"sm_{self.cc}a").encode() + b"\0"
        self.ok(self.cu.cuModuleLoadData(C.byref(self.mod), src))

    def alloc(self, data: bytes) -> int:
        d = C.c_uint64()
        self.ok(self.cu.cuMemAlloc_v2(C.byref(d), C.c_size_t(len(data))))
        self.ok(self.cu.cuMemcpyHtoD_v2(d, data, C.c_size_t(len(data))))
        return d.value

    def get(self, d: int, n: int) -> bytes:
        b = (C.c_uint8 * n)()
        self.ok(self.cu.cuMemcpyDtoH_v2(b, C.c_uint64(d), C.c_size_t(n)))
        return bytes(b)

    def run(self, kind: str, fmt: str, c: list) -> dict:
        """One launch: the descriptor's bytes, and the probes (and image) or the written
        bytes (two runs over the fills 0xEE and 0x11: a byte is written if it changed in
        either)."""
        out_ = self.alloc(bytes(256 + SM))
        store = kind in ("st", "rd")
        g = self.alloc(bytes([0xEE]) * len(SRC) if store else SRC)
        buf = (C.c_uint8 * 192)()
        at = C.addressof(buf) + (-C.addressof(buf)) % 64
        U32, U64 = C.c_uint32, C.c_uint64
        self.ok(self.cu.cuTensorMapEncodeTiled(
            C.c_void_p(at), FORMATS[fmt][0], U32(2), C.c_void_p(g), (U64 * 2)(W, R),
            (U64 * 1)(row_bytes(fmt)), (U32 * 2)(*BOX), (U32 * 2)(1, 1), 0, 0, 0, 0))  # fmt: skip
        raw = C.string_at(at, 128)
        f = C.c_void_p()
        self.ok(self.cu.cuModuleGetFunction(C.byref(f), self.mod, kind.encode()))
        args = [(C.c_uint8 * 128).from_buffer_copy(raw), U64(out_), U32(E),
                U32(c[0] & 0xFFFFFFFF), U32(c[1] & 0xFFFFFFFF)]  # fmt: skip
        ptrs = (C.c_void_p * 5)(*[C.cast(C.pointer(x), C.c_void_p) for x in args])
        grid = 2 if KERNELS[kind][1] else 1
        runs = []
        for fill in (0xEE, 0x11) if store else (None,):
            if fill == 0x11:
                self.ok(self.cu.cuMemcpyHtoD_v2(U64(g), bytes([fill]) * len(SRC),
                                                C.c_size_t(len(SRC))))  # fmt: skip
            rc = self.cu.cuLaunchKernel(f, grid, 1, 1, 32, 1, 1, 0, None, ptrs, None)
            rc = rc or self.cu.cuCtxSynchronize()
            if rc:
                return {"raw": raw.hex(), "rc": rc}
            runs.append(self.get(g, len(SRC)))
        o = self.get(out_, 256 + SM)
        w = list(memoryview(o[:64]).cast("I"))
        r = {
            "raw": raw.hex(),
            "rc": 0,
            "probe": [w[4 + 4 * k : 7 + 4 * k] for k in range(grid)],
        }
        if kind == "ld":
            r["image"] = o[256:].hex()
        if store:
            r["written"] = [
                k for k, (a, b) in enumerate(zip(*runs)) if a != 0xEE or b != 0x11
            ]
        return r


def _child(conn, probes):
    try:
        cu = _Cuda()
        if probes is None:
            conn.send({"cc": cu.cc})
            return
        cu.load()
        conn.send([cu.run(*p) for p in probes])
    except (OSError, AttributeError, RuntimeError) as e:
        conn.send({"error": repr(e)})


def in_child(probes):
    """_child(probes) in a forked process: its results, or {"error": ...}."""
    ctx = multiprocessing.get_context("fork")
    r, w = ctx.Pipe(False)
    p = ctx.Process(target=_child, args=(w, probes))
    p.start()
    w.close()
    try:
        return r.recv() if r.poll(600) else {"error": "timeout"}
    except EOFError:
        return {"error": f"child exit status {p.exitcode}"}
    finally:
        p.join(30)


# ------------------------------------------------------------------- the frontend's view
def kir(raw_hex: str) -> tuple:
    """(decoded descriptor, kir_desc over bytes) as the frontend sees the descriptor."""
    d = launch.decode(bytes.fromhex(raw_hex))
    return d, launch.kir_desc(d, 1)


def footprint(k: dict, c: list) -> dict:
    """KIR box element (row i1, byte i0) -> its global byte offset, inside the extents only
    (c in descriptor units, dim 0 innermost; KIR order is outermost first)."""
    num, den = k["dim0_ratio"]
    c0, c1 = c[0] * num // den, c[1]
    (n1, n0), s1, (b1, b0) = k["shape"], k["strides"][0], k["box"]
    return {(i1, i0): (c1 + i1) * s1 + c0 + i0 for i1 in range(b1) for i0 in range(b0)
            if 0 <= c1 + i1 < n1 and 0 <= c0 + i0 < n0}  # fmt: skip


def smem_offset(fmt: str, b0: int, i1: int, i0: int) -> int:
    """Where KIR byte i0 of box row i1 lands in a dense (unswizzled) shared copy: each 8 /
    12 packed bytes of the 16-byte-aligned formats at the start of 16."""
    chunk = {"16U4_ALIGN16B": 8, "16U6_ALIGN16B": 12}.get(fmt)
    if chunk is None:
        return i1 * b0 + i0
    return i1 * (-(-b0 // chunk) * 16) + i0 // chunk * 16 + i0 % chunk


def per_cta(kind: str, tx: int) -> list:
    """The bytes each CTA's mbarrier sees of one copy of tx bytes: a multicast's in every
    destination, a .cta_group::2 copy's (both CTAs') on the pair leader's."""
    return {"ld": [tx], "mc": [tx, tx], "mccg": [2 * tx, 0], "cg": [2 * tx, 0]}.get(
        kind, []
    )


class SubByteTmaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        got = in_child(None)
        if "error" in got:
            raise unittest.SkipTest(
                f"no CUDA driver with a visible GPU ({got['error']})"
            )
        if got["cc"] < 100:
            raise unittest.SkipTest(f"sm_{got['cc']}: needs sm_100 or later")

    def test_bytes_and_footprints(self):
        probes = []
        for fmt in PACKED:
            g = granularity(fmt)
            for c in ([0, 0], [0, R - 4], [0, -4], [W - g, 0], [-g, 0], [W, 0]):
                probes.append(("ld", fmt, c))
            if "store" in tma._PACKED[fmt][1]:
                probes += [("st", fmt, c) for c in ([0, 0], [0, R - 4], [W - g, 0])]
            probes += [(k, fmt, [0, 0]) for k in ("mc", "mccg", "cg")]
            probes.append(("cg", fmt, [0, R - 4]))
        probes.append(("ld", "UINT8", [16, 0]))
        got = in_child(probes)
        self.assertNotIn("error", got)
        for (kind, fmt, c), r in zip(probes, got):
            with self.subTest(kind=kind, fmt=fmt, c=c):
                self.assertEqual(r["rc"], 0)
                d, k = kir(r["raw"])
                self.assertEqual(d["dtype"], fmt)
                foot = footprint(k, c)
                tx = math.prod(k["box"]) * k["elem_bytes"]  # the frontend's bytes
                if fmt in tma._PACKED:
                    self.assertEqual(tx, math.prod(BOX) * FORMATS[fmt][1] // 8)
                for (steps, done, empty), n in zip(r["probe"], per_cta(kind, tx)):
                    self.assertEqual((E - steps, done, empty), (n, 1, 1))
                if kind == "ld":
                    img = bytes.fromhex(r["image"])
                    for i1, i0 in itertools.product(*map(range, k["box"])):
                        got = img[smem_offset(fmt, k["box"][1], i1, i0)]
                        want = SRC[foot[i1, i0]] if (i1, i0) in foot else 0
                        self.assertEqual(got, want, (i1, i0))
                if kind == "st":
                    self.assertEqual(r["written"], sorted(foot.values()))

    def test_traps(self):
        traps = [("ld", fmt, [granularity(fmt) // 2, 0]) for fmt in PACKED]
        traps += [
            ("st", fmt, [0, 0]) for fmt in PACKED if "store" not in tma._PACKED[fmt][1]
        ]
        traps += [("rd", fmt, [0, 0]) for fmt in PACKED] + [("ld", "UINT8", [8, 0])]
        for probe in traps:
            with self.subTest(probe=probe):
                got = in_child([probe])
                self.assertNotIn("error", got)
                self.assertEqual(got[0]["rc"], ILLEGAL_INSTRUCTION, json.dumps(probe))


if __name__ == "__main__":
    unittest.main()
