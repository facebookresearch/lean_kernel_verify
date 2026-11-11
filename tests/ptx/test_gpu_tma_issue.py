# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""GPU truth for the TMA / bulk-copy issuing rule of the PTX op table
(frontend/ptx/tma.py), for each ptxas that builds a family's cubins.

    LKV_PTXAS=/path/a/ptxas:/path/b/ptxas python3 -I tests/ptx/test_gpu_tma_issue.py

Needs libcuda.so.1 of a real driver, a visible sm_100+ GPU and at least one compiler;
skips otherwise. Compilers: each ptxas in LKV_PTXAS (os.pathsep-separated; Triton's
ptxas-blackwell), the CuTe DSL of this process when LKV_TMA_CUTE=1 (its own pipeline
and embedded PTX compiler, as it builds a kernel: cute_cubin), and cubins in
LKV_TMA_CUBINS that such a compiler made of this module (cute_cubin run elsewhere). A
compiler's version is the one its cubin's .note.nv.tkinfo records (tma.tkinfo_version).

One warp runs hand-written PTX in which k of its lanes (k = 1, 2, 32 under a predicate,
and the whole warp unpredicated) execute one copy whose operands are the same on every
lane, equal values read from a lane-indexed table, or differ in coordinates / global
address, shared address, descriptor, bulk size or L2 cache policy.
Counted:
  loads    the transaction bytes the mbarrier saw: it expects 64 tiles; after the copies
           land one thread arrives, then completes a fixed number of bytes at a time until
           the phase completes; the next phase must then complete with no bytes (nothing
           landed late);
  reduces  u32 adds of 1 (1 << lane for a per-lane source) into a zeroed tensor;
  stores   which tiles / descriptors / sizes were written, and the time of 32 KiB copies
           in units of one copy (identical stores leave no trace in memory).
The counts must be the op table's (tma.py): loads and reduces (tensor and bulk) once per
executing lane; stores (tensor and bulk) once per distinct (descriptor or global address,
coordinates, shared address, bulk size) among a warp's executing lanes, the cache policy
not part of it. Each compiler version must also be one the op table names as
validated (tma.ISSUE_PTXAS), so a toolchain bump fails here first, and the GPU's target
one it admits (tma.ISSUE_TARGETS: sm_100 and sm_103, measured on a GB200 and a GB300).
"""

import ctypes as C
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

if __package__ in (None, ""):
    sys.dont_write_bytecode = True
    _root = ModuleType("lkv_root")
    _root.__path__ = [str(Path(__file__).resolve().parents[2])]
    sys.modules["lkv_root"] = _root
    __package__ = "lkv_root.tests.ptx"  # noqa: A001

from ...frontend.ptx import tma  # noqa: E402

PTXAS = [p for p in os.environ.get("LKV_PTXAS", "").split(os.pathsep) if p]
CUBINS = [p for p in os.environ.get("LKV_TMA_CUBINS", "").split(os.pathsep) if p]
CUTE = os.environ.get("LKV_TMA_CUTE") == "1"
TILE, BIG, ROWS = 128, 32768, 256  # tile {8 x u32, 4 rows}; timing tile {64 x u32, 128}
TV = ("same", "same_dyn", "coord", "smem", "desc", "hint")
BV = ("same", "same_dyn", "coord", "smem", "size", "hint")
FORMS = {"tload": TV, "tred": TV, "tstore": TV, "bload": BV, "bred": BV, "bstore": BV}
# Distinct store keys among k lanes: values compared at run time (`same_dyn`: equal values
# in lane-varying registers); `desc` and `size` alternate over even / odd lanes; the
# cache policy (`hint`, alternating too) is one lane's.
STORE_KEYS = {
    "same": 1,
    "same_dyn": 1,
    "hint": 1,
    "coord": 32,
    "smem": 32,
    "desc": 2,
    "size": 2,
}
CASES = [(f, v, k, False) for f, vs in FORMS.items() for v in vs for k in (1, 2, 32)] + [
    (f, v, 32, True) for f, vs in FORMS.items() for v in vs]  # fmt: skip
TIMED = {"tstore": TV, "bstore": BV}


def expected(form: str, var: str, k: int) -> int:
    """The op table's copy count: stores once per distinct key, the rest per lane."""
    return min(k, STORE_KEYS[var]) if form.endswith("store") else k


# ------------------------------------------------------------------------------- PTX
def kernel(form: str, var: str, full: bool = False, big: bool = False) -> tuple:
    """(entry, PTX) of one warp in which the lanes < p_k (all if `full`: unpredicated)
    execute one copy; out[0..2] the load probe, out[4..5] cycles."""
    name = f"{form}_{var}" + ("_full" if full else "") + ("_big" if big else "")
    load, red = form.endswith("load"), form.endswith("red")
    nb = BIG if big else TILE
    step = nb // 2 if var == "size" else nb  # the load probe's bytes per step
    val = "mov.u32 %r15, 0;" if load else (
        ("shl.b32 %r15, 1, %r0;" if var == "smem" else "mov.u32 %r15, 1;") if red
        else "add.u32 %r15, %r0, 1;")  # fmt: skip
    fill = (["shl.b32 %r30, %r0, 2;", "FILL: add.u32 %r32, %r3, %r30;"] if big else
            ["mov.u32 %r30, 0;", "FILL: add.u32 %r32, %r11, %r30;"]) + [
        "st.shared.u32 [%r32], %r15;", f"add.u32 %r30, %r30, {128 if big else 4};",
        f"setp.lt.u32 %p2, %r30, {BIG + 32 * TILE if big else 128};", "@%p2 bra FILL;"]  # fmt: skip
    c1 = {"same_dyn": "%r13", "coord": "%r12"}.get(var, "%r20")
    sm = {"coord": "%r11", "smem": "%r11", "same_dyn": "%r16"}.get(var, "%r3")
    tm = "%rd7" if var == "desc" else "%rd3"
    sz = "%r17" if var == "size" else nb
    hq, ho = (".L2::cache_hint", ", %rd10") if var == "hint" else ("", "")
    off = {
        "coord": f"mul.wide.u32 %rd11, %r0, {nb};",
        "same_dyn": "cvt.u64.u32 %rd11, %r13;",
    }
    crd = f"[{tm}, {{%r20, {c1}}}]"
    op = {
        "tload": f"cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes{hq} [{sm}], {crd}, [%r4]{ho}",
        "tstore": f"cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group{hq} {crd}, [{sm}]{ho}",
        "tred": f"cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group{hq} {crd}, [{sm}]{ho}",
        "bload": f"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes{hq} [{sm}], [%rd12], {sz}, [%r4]{ho}",
        "bstore": f"cp.async.bulk.global.shared::cta.bulk_group{hq} [%rd12], [{sm}], {sz}{ho}",
        "bred": f"cp.reduce.async.bulk.global.shared::cta.bulk_group{hq}.add.u32 [%rd12], [{sm}], {sz}{ho}",
    }[form]  # fmt: skip
    spin = ["mov.u64 %rd25, %clock64;", "{0}: mov.u64 %rd26, %clock64;",
            "sub.s64 %rd27, %rd26, %rd25;", "setp.lt.s64 %p9, %rd27, 4000000;", "@%p9 bra {0};"]  # fmt: skip
    after = [  # load: count the landed bytes (module docstring)
        "bar.sync 0;", "@!%p3 bra NOPROBE;", *[s.format("S1") for s in spin],
        "mbarrier.arrive.shared::cta.b64 %rd23, [%r4];", "mov.u32 %r40, 0;",
        "PROBE: mbarrier.try_wait.parity.shared::cta.b64 %p5, [%r4], 0;",
        "@%p5 bra PROBED;", "setp.ge.u32 %p6, %r40, 200;", "@%p6 bra PROBED;",
        f"mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%r4], {step};",
        "add.u32 %r40, %r40, 1;", "bra PROBE;",
        "PROBED: st.global.u32 [%rd0], %r40;", "selp.u32 %r41, 1, 0, %p5;",
        "st.global.u32 [%rd0+4], %r41;", *[s.format("S2") for s in spin],
        "mbarrier.arrive.shared::cta.b64 %rd24, [%r4];",
        "mbarrier.try_wait.parity.shared::cta.b64 %p7, [%r4], 1;",
        "selp.u32 %r42, 1, 0, %p7;", "st.global.u32 [%rd0+8], %r42;", "NOPROBE: bar.sync 0;",
    ] if load else []  # fmt: skip
    body = [
        "mov.u32 %r0, %tid.x;", "ld.param.u32 %r2, [p_k];", "setp.lt.u32 %p0, %r0, %r2;",
        "ld.param.u64 %rd0, [p_out];", "cvta.to.global.u64 %rd0, %rd0;",
        "ld.param.u64 %rd1, [p_g];", "cvta.to.global.u64 %rd1, %rd1;",
        "ld.param.u64 %rd2, [p_tab];", "cvta.to.global.u64 %rd2, %rd2;",
        "mov.u32 %r3, smem;", "mov.u32 %r4, mbar;",
        "mov.b64 %rd3, tmA;", "cvta.param.u64 %rd3, %rd3;",
        "mov.b64 %rd4, tmB;", "cvta.param.u64 %rd4, %rd4;",
        "shl.b32 %r10, %r0, 7;", "add.u32 %r11, %r3, %r10;", "shl.b32 %r12, %r0, 2;",
        "mul.wide.u32 %rd5, %r0, 4;", "add.u64 %rd6, %rd2, %rd5;",
        "ld.global.u32 %r13, [%rd6];",  # tab[lane] == 0
        "and.b32 %r14, %r0, 1;", "setp.eq.u32 %p1, %r14, 1;", "mov.u32 %r20, 0;",
        val, *fill,
        "add.u32 %r16, %r3, %r13;", "selp.b64 %rd7, %rd4, %rd3, %p1;",
        f"selp.u32 %r17, {nb // 2}, {nb}, %p1;",
        "createpolicy.fractional.L2::evict_last.b64 %rd8, 0f3F800000;",
        "createpolicy.fractional.L2::evict_first.b64 %rd9, 0f3F800000;",
        "selp.b64 %rd10, %rd9, %rd8, %p1;",
        off.get(var, "mov.u64 %rd11, 0;"), "add.u64 %rd12, %rd1, %rd11;",
        "setp.eq.u32 %p3, %r0, 0;", "@%p3 mbarrier.init.shared::cta.b64 [%r4], 1;",
        f"@%p3 mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%r4], {64 * nb};" if load else "",
        "fence.mbarrier_init.release.cluster;", "fence.proxy.async.shared::cta;",
        "bar.sync 0;", "mov.u64 %rd20, %clock64;",
        f"{op};" if full else f"@%p0 {op};",
        "" if load else "cp.async.bulk.commit_group;",
        "" if load else "cp.async.bulk.wait_group 0;",
        "mov.u64 %rd21, %clock64;", "sub.s64 %rd22, %rd21, %rd20;",
        "@%p3 st.global.u64 [%rd0+16], %rd22;", *after, "ret;",
    ]  # fmt: skip
    code = "\n    ".join(s for s in body if s)
    return (
        name,
        f"""
.visible .entry {name}(.param .align 64 .b8 tmA[128], .param .align 64 .b8 tmB[128],
    .param .u64 p_out, .param .u64 p_g, .param .u64 p_tab, .param .u32 p_k)
{{
    .reg .pred %p<16>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<40>;
    .shared .align 1024 .b8 smem[{BIG + 32 * TILE if big else 32 * TILE}];
    .shared .align 8 .b64 mbar;
    {code}
}}
""",
    )


def module(target: str) -> str:
    ks = {kernel(f, v, full) for f, v, _, full in CASES}
    ks |= {kernel(f, v, False, True) for f, vs in TIMED.items() for v in vs}
    ver = "8.7" if target == "sm_100a" else "8.8"  # 8.8 introduced sm_103a
    return f".version {ver}\n.target {target}\n.address_size 64\n" + "".join(
        k[1] for k in sorted(ks)
    )


# ---------------------------------------------------------------------------- driver
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
        self.target = f"sm_{self.cc}a"

    def ok(self, rc):
        if rc:
            raise RuntimeError(f"CUDA error {rc}")

    def alloc(self, data: bytes) -> int:
        d = C.c_uint64()
        self.ok(self.cu.cuMemAlloc_v2(C.byref(d), C.c_size_t(len(data))))
        self.ok(self.cu.cuMemcpyHtoD_v2(d, data, C.c_size_t(len(data))))
        return d.value

    def get(self, d: int, n: int) -> list:
        b = (C.c_uint32 * (n // 4))()
        self.ok(self.cu.cuMemcpyDtoH_v2(b, C.c_uint64(d), C.c_size_t(n)))
        return list(b)

    def tmap(self, addr: int, cols: int, br: int):
        """A 2-D u32 tiled CUtensorMap, box {cols, br}, no swizzle."""
        buf = (C.c_uint8 * 192)()
        at = C.addressof(buf) + (-C.addressof(buf)) % 64
        U64, U32 = C.c_uint64, C.c_uint32
        self.ok(self.cu.cuTensorMapEncodeTiled(
            C.c_void_p(at), 2, U32(2), C.c_void_p(addr), (U64 * 2)(cols, ROWS),
            (U64 * 1)(cols * 4), (U32 * 2)(cols, br), (U32 * 2)(1, 1), 0, 0, 0, 0))  # fmt: skip
        return (C.c_uint8 * 128).from_buffer_copy(C.string_at(at, 128))

    def run(self, mod, form, var, k, full=False, big=False) -> dict:
        nb = BIG if big else TILE
        cols, br = (64, 128) if big else (8, 4)
        size = max(ROWS * cols * 4, 32 * nb)
        load = form.endswith("load")
        A = self.alloc(b"".join((i + 1).to_bytes(4, "little") for i in range(size // 4))
                       if load else bytes(size))  # fmt: skip
        B, out, tab = (
            self.alloc(bytes(size)),
            self.alloc(bytes(64)),
            self.alloc(bytes(128)),
        )
        f = C.c_void_p()
        name = kernel(form, var, full, big)[0].encode()
        self.ok(self.cu.cuModuleGetFunction(C.byref(f), mod, name))
        args = [self.tmap(A, cols, br), self.tmap(B, cols, br), C.c_uint64(out),
                C.c_uint64(A), C.c_uint64(tab), C.c_uint32(k)]  # fmt: skip
        ptrs = (C.c_void_p * 6)(*[C.cast(C.pointer(x), C.c_void_p) for x in args])
        try:
            self.ok(self.cu.cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, None, ptrs, None))
            self.ok(self.cu.cuCtxSynchronize())
            o = self.get(out, 64)
            r = {"cycles": o[4] | o[5] << 32}
            if load:
                step = nb // 2 if var == "size" else nb
                r.update(bytes=64 * nb - o[0] * step, phase0=o[1], no_late=o[2])
            elif not big:
                r["A"], r["B"] = self.get(A, 32 * TILE), self.get(B, 32 * TILE)
            return r
        finally:
            for d in (A, B, out, tab):
                self.cu.cuMemFree_v2(C.c_uint64(d))


def _tiles(ws: list) -> dict:
    """{tile: value} of the nonzero 128-byte tiles ('mixed' if not uniform)."""
    t = [ws[32 * i : 32 * i + 32] for i in range(len(ws) // 32)]
    return {
        i: (x[0] if len(set(x)) == 1 else "mixed") for i, x in enumerate(t) if any(x)
    }


def ptxas_cubin(ptxas: str, src: str, target: str) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "t.ptx").write_text(src)
        subprocess.run(
            [ptxas, f"-arch={target}", "-o", f"{d}/t.cubin", f"{d}/t.ptx"], check=True
        )
        return (Path(d) / "t.cubin").read_bytes()


def cute_cubin(src: str, target: str) -> bytes:
    """The cubin the CuTe DSL of this process makes of `src`'s kernels, through its own
    pass pipeline and embedded PTX compiler: each kernel an llvm.func whose body is the
    kernel's PTX as inline assembly (registers renamed off the operands' %r / %rd), its
    descriptors byval grid-constant parameters (the generic address cvta.param gives)."""
    from cutlass._mlir import ir, passmanager  # noqa: PLC0415

    subst = {"ld.param.u32 %r2, [p_k];": "mov.u32 %r2, $5;",
             "ld.param.u64 %rd0, [p_out];": "mov.u64 %rd0, $0;",
             "ld.param.u64 %rd1, [p_g];": "mov.u64 %rd1, $1;",
             "ld.param.u64 %rd2, [p_tab];": "mov.u64 %rd2, $2;",
             "mov.u32 %r3, smem;": "mov.u32 %r3, $6;", "mov.u32 %r4, mbar;": "mov.u32 %r4, $7;",
             "mov.b64 %rd3, tmA;": "mov.u64 %rd3, $3;", "mov.b64 %rd4, tmB;": "mov.u64 %rd4, $4;",
             "cvta.param.u64 %rd3, %rd3;": "", "cvta.param.u64 %rd4, %rd4;": "", "ret;": ""}  # fmt: skip
    byval = "!llvm.ptr {llvm.byval = !llvm.array<128 x i8>, llvm.align = 64 : i64, nvvm.grid_constant}"
    out = ["module attributes {gpu.container_module} {", "gpu.module @kernels {"]
    for fn in src.split("\n.visible .entry ")[1:]:
        name = fn[: fn.index("(")]
        smem = re.search(r"smem\[(\d+)\]", fn).group(1)
        lines = [subst.get(x.strip(), x.strip()) for x in fn[fn.index("{") + 1 : fn.rindex("}")].splitlines()
                 if x.strip() and not x.strip().startswith(".shared")]  # fmt: skip
        asm = (
            "{\n"
            + "\n".join(re.sub(r"%(rd|r|p)(?=[\d<])", r"%x\1", x) for x in lines if x)
            + "\n}"
        )
        asm = asm.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        out += [
            f"llvm.mlir.global internal @smem_{name}() {{addr_space = 3 : i32, alignment = 1024 : i64}} : !llvm.array<{smem} x i8>",
            f"llvm.mlir.global internal @mbar_{name}() {{addr_space = 3 : i32, alignment = 8 : i64}} : i64",
            f"llvm.func @{name}(%ta: {byval}, %tb: {byval}, %out: i64, %g: i64, %tab: i64, %k: i32) attributes {{nvvm.kernel}} {{",
            f"%s = llvm.mlir.addressof @smem_{name} : !llvm.ptr<3>", "%si = llvm.ptrtoint %s : !llvm.ptr<3> to i32",
            f"%m = llvm.mlir.addressof @mbar_{name} : !llvm.ptr<3>", "%mi = llvm.ptrtoint %m : !llvm.ptr<3> to i32",
            f'llvm.inline_asm has_side_effects asm_dialect = att "{asm}", "l,l,l,l,l,r,r,r,~{{memory}}" '
            "%out, %g, %tab, %ta, %tb, %k, %si, %mi : (i64, i64, i64, !llvm.ptr, !llvm.ptr, i32, i32, i32) -> ()",
            "llvm.return", "}",
        ]  # fmt: skip
    with tempfile.TemporaryDirectory() as d, ir.Context(), ir.Location.unknown():
        m = ir.Module.parse("\n".join(out + ["}", "}"]))
        passmanager.PassManager.parse(
            f"builtin.module(cute-to-nvvm{{cubin-format=bin opt-level=3 cubin-chip={target} "
            f"dump-cubin-path='{d}/t'}})"
        ).run(m.operation)
        return next(Path(d).iterdir()).read_bytes()


class TmaIssueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (PTXAS or CUBINS or CUTE):
            raise unittest.SkipTest(
                "no compiler: LKV_PTXAS, LKV_TMA_CUBINS, LKV_TMA_CUTE"
            )
        try:
            cls.cu = _Cuda()
        except (OSError, AttributeError) as e:
            raise unittest.SkipTest(f"no CUDA driver with a visible GPU ({e})")
        if cls.cu.cc < 100:
            raise unittest.SkipTest(f"{cls.cu.target}: needs sm_100 or later")

    def test_issue_counts_per_compiler(self):
        src = module(self.cu.target)
        builds = [(p, lambda p=p: ptxas_cubin(p, src, self.cu.target)) for p in PTXAS]
        builds += [(p, lambda p=p: Path(p).read_bytes()) for p in CUBINS]
        builds += (
            [("cute-dsl", lambda: cute_cubin(src, self.cu.target))] if CUTE else []
        )
        for what, build in builds:
            cubin = build()
            ver = tma.tkinfo_version(cubin)
            with self.subTest(compiler=what, version=ver):
                self.assertIn(
                    ver, tma.ISSUE_PTXAS, "a compiler the op table does not name"
                )
                mod = C.c_void_p()
                self.cu.ok(self.cu.cu.cuModuleLoadData(C.byref(mod), cubin))
                try:
                    self._counts(mod)
                    self._store_times(mod)
                finally:
                    self.cu.cu.cuModuleUnload(mod)
                self.assertIn(  # measured here: the gate may admit the target
                    self.cu.target.removesuffix("a"), tma.ISSUE_TARGETS,
                    "the rules hold on this target: add it to tma.ISSUE_TARGETS",
                )  # fmt: skip

    def _counts(self, mod):
        for form, var, k, full in CASES:
            with self.subTest(form=form, var=var, k=k, full=full):
                r, n = self.cu.run(mod, form, var, k, full), expected(form, var, k)
                lanes = range(k)
                if form.endswith("load"):  # exact: the transaction bytes
                    want = sum(
                        TILE // 2 if var == "size" and t % 2 else TILE for t in lanes
                    )
                    self.assertEqual(
                        (r["bytes"], r["phase0"], r["no_late"]), (want, 1, 1)
                    )
                    continue
                a, b, w = _tiles(r["A"]), _tiles(r["B"]), r["A"]
                if form.endswith("red"):  # exact: every add lands
                    if var == "size":  # word 0 every copy, word 31 the full-size ones
                        self.assertEqual((w[0], w[31]), (k, (k + 1) // 2))
                    elif var == "smem":
                        self.assertEqual(a, {0: (1 << k) - 1})
                    elif var == "coord":
                        self.assertEqual(a, dict.fromkeys(lanes, 1))
                    elif var == "desc":
                        self.assertEqual(
                            (a, b), ({0: (k + 1) // 2}, {0: k // 2} if k > 1 else {})
                        )
                    else:
                        self.assertEqual(a, {0: n})
                elif var == "coord":  # stores: every lane's tile, with its source
                    self.assertEqual(a, {t: t + 1 for t in lanes})
                elif var == "desc":  # one per descriptor
                    self.assertEqual((a, b), ({0: 1}, {0: 1} if k > 1 else {}))
                elif var == "smem":  # one tile, from one of the k sources (time counts)
                    self.assertEqual(list(a), [0])
                    self.assertIn(a[0], range(1, k + 1))
                else:  # same, same_dyn, hint: lane 0's source; size: the full copy too
                    self.assertEqual(a, {0: 1})

    def _store_times(self, mod):
        """Identical stores leave no trace: count 32 KiB stores by time, in units of one
        copy: (t(k=32) - t(k=1)) / (t(coord, k=2) - t(coord, k=1)) + 1 (each kernel has
        its own fixed overhead)."""

        def t(form, var, k):
            return min(
                self.cu.run(mod, form, var, k, False, True)["cycles"] for _ in range(5)
            )

        for form, vs in TIMED.items():
            copy = t(form, "coord", 2) - t(form, "coord", 1)
            for var in vs:
                units = (t(form, var, 32) - t(form, var, 1)) / copy + 1
                # in 32 KiB copies: the odd lanes' size key is half a copy
                n = 1.5 if var == "size" else expected(form, var, 32)
                with self.subTest(form=form, var=var, units=round(units, 2)):
                    self.assertLess(abs(units - n), max(0.35, 0.2 * n))


if __name__ == "__main__":
    unittest.main()
