# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tiny CuTe DSL kernels for the frontend tests, run through the real dump -> to_kir path.

Each `run_*` function is a case entry (the host path: compile with CuTe tensors, launch
with torch tensors) and each `args_*` function its case builder. dump.py imports this
file from its directory, like an exported kernel package (it is not a buck src).
"""

import os
from os import environ as _environ

import cutlass
import cutlass.cute as cute
import cutlass.utils
import numpy
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint8
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op

TILE = 64


def _cute(t, leading=1):
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=leading)


# ------------------------------------------------ TMA producer / consumer on an mbarrier


def _tma_pipeline(elect_producer: bool, extra_rows: int, copy_lanes: int):
    """Warp 0 loads one TILE x TILE tile per CTA with TMA and signals `full`; warp 1 waits
    and stores one value per row of the tile (rows past the end are masked). Seeded bugs:
    elect_producer=False runs the producer's arrive.expect_tx on all 32 lanes (32 arrivals
    and 32 x the bytes on a count-1 barrier); extra_rows > 0 builds the TMA descriptor on
    a view claiming that many rows past the tensor (TMA does not clip reads there);
    copy_lanes < 32 lets only that many lanes of warp 0 reach the TMA copy, whose lowering
    elects a lane with a full-mask elect.sync."""

    @cute.kernel
    def kernel(
        tma_atom: cute.CopyAtom,
        mA_tma: cute.Tensor,
        mOut: cute.Tensor,
        smem_layout: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        smem = cutlass.utils.SmemAllocator()
        full = smem.allocate_array(cutlass.Int64, 1)
        sA = smem.allocate_tensor(
            cutlass.BFloat16,
            smem_layout.outer,
            byte_alignment=128,
            swizzle=smem_layout.inner,
        )
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(full, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        gA = cute.local_tile(mA_tma, (TILE, TILE), (cute.arch.block_idx()[0], 0))
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        if warp == 0:
            if cutlass.const_expr(elect_producer):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full, TILE * TILE * 2)
            else:
                cute.arch.mbarrier_arrive_and_expect_tx(full, TILE * TILE * 2)
            if cutlass.const_expr(copy_lanes < 32):
                if tidx < copy_lanes:
                    cute.copy(tma_atom, tAgA, tAsA, tma_bar_ptr=full)
            else:
                cute.copy(tma_atom, tAgA, tAsA, tma_bar_ptr=full)
        if warp == 1:
            cute.arch.mbarrier_wait(full, 0)
            for half in cutlass.range(TILE // 32):
                row = cute.arch.block_idx()[0] * TILE + half * 32 + tidx - 32
                if row < mOut.shape[0]:
                    mOut[row] = Float32(sA[half * 32 + tidx - 32, 0])

    @cute.jit
    def host(mA: cute.Tensor, mOut: cute.Tensor):
        smem_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_ordered_layout((TILE, TILE), order=(1, 0)),
        )
        view = mA
        if cutlass.const_expr(extra_rows > 0):
            view = cute.make_tensor(
                mA.iterator,
                cute.make_layout(
                    (mA.shape[0] + extra_rows, mA.shape[1]), stride=mA.stride
                ),
            )
        tma_atom, mA_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), view, smem_layout, (TILE, TILE)
        )
        grid = (cute.ceil_div(mA.shape[0], TILE), 1, 1)
        kernel(tma_atom, mA_tma, mOut, smem_layout).launch(
            grid=grid, block=(64, 1, 1), smem=TILE * TILE * 2 + 1024
        )

    return host


_TMA_HOSTS: dict = {}


def run_tma_pipeline(
    a: torch.Tensor,
    out: torch.Tensor,
    elect_producer: bool,
    extra_rows: int,
    copy_lanes: int,
) -> None:
    key = (elect_producer, extra_rows, copy_lanes)
    if key not in _TMA_HOSTS:
        _TMA_HOSTS[key] = _tma_pipeline(*key)
    cute.compile(_TMA_HOSTS[key], _cute(a), _cute(out, 0))(a, out)


def args_tma_pipeline(inp: dict):
    rows = inp["rows"]
    args = [
        torch.zeros((rows, TILE), dtype=torch.bfloat16),
        torch.zeros((rows,), dtype=torch.float32),
    ]
    kwargs = {
        "elect_producer": inp.get("elect_producer", True),
        "extra_rows": inp.get("extra_rows", 0),
        "copy_lanes": inp.get("copy_lanes", 32),
    }
    return args, kwargs, {}


# -------------------------------------------- per-thread copies bounded by metadata


@cute.kernel
def _segment_copy_kernel(mX: cute.Tensor, mY: cute.Tensor, mCu: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    seg, _, _ = cute.arch.block_idx()
    start = mCu[seg]
    n = mCu[seg + 1] - start
    for i in cutlass.range(cute.ceil_div(n, 128)):
        r = i * 128 + tidx
        if r < n:
            mY[start + r] = mX[start + r] * Float32(2.0)


@cute.jit
def _segment_copy_host(mX: cute.Tensor, mY: cute.Tensor, mCu: cute.Tensor):
    _segment_copy_kernel(mX, mY, mCu).launch(
        grid=(mCu.shape[0] - 1, 1, 1), block=(128, 1, 1)
    )


def run_segment_copy(x, y, cu) -> None:
    cute.compile(_segment_copy_host, _cute(x, 0), _cute(y, 0), _cute(cu, 0))(x, y, cu)


def run_segment_copy_no_empty(x, y, cu) -> None:
    """A host that refuses empty segments before launching (like a one-block attention
    backward that cannot handle them)."""
    if any(b == a for a, b in zip(cu.tolist(), cu.tolist()[1:])):
        raise ValueError("empty segments are not supported")
    run_segment_copy(x, y, cu)


def run_segment_copy_skip_empty(x, y, cu) -> None:
    """A host that returns before launching when there are no rows."""
    if x.numel() == 0:
        return
    run_segment_copy(x, y, cu)


def args_segment_copy(inp: dict):
    off = [0]
    for n in inp["lens"]:
        off.append(off[-1] + n)
    return (
        [
            torch.zeros(off[-1]),
            torch.zeros(off[-1]),
            torch.tensor(off, dtype=torch.int32),
        ],
        {},
        {},
    )


# ---------------------------------- warp-cooperative segment search (shfl, ballot, popc)


@cute.kernel
def _warp_search_kernel(mY: cute.Tensor, mCu: cute.Tensor):
    """CTA b finds the segment holding 32-row tile b with a warp prefix sum over the
    per-segment tile counts (lane l: segment l), then warp 0 writes that tile's rows."""
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    nseg = mCu.shape[0] - 1
    cu = Int32(0)
    if lane <= nseg:
        cu = mCu[lane]
    nxt = cute.arch.shuffle_sync_down(cu, offset=1)
    tiles = Int32(0)
    if lane < nseg:
        tiles = (nxt - cu + 31) // 32
    cum = tiles
    for i in cutlass.range_constexpr(5):
        up = cute.arch.shuffle_sync_up(cum, offset=1 << i)
        if lane >= (1 << i):
            cum = cum + up
    seg = cute.arch.popc(cute.arch.vote_ballot_sync(cum <= bid))
    if seg < nseg:
        first = cute.arch.shuffle_sync(cum, seg) - cute.arch.shuffle_sync(tiles, seg)
        start = cute.arch.shuffle_sync(cu, seg)
        end = cute.arch.shuffle_sync(nxt, seg)
        row = start + (bid - first) * 32 + lane
        if tidx < 32 and row < end:
            mY[row] = Float32(1.0)


@cute.jit
def _warp_search_host(mY: cute.Tensor, mCu: cute.Tensor, num_tiles: Int32):
    _warp_search_kernel(mY, mCu).launch(grid=(num_tiles, 1, 1), block=(64, 1, 1))


def run_warp_search(y, cu, num_tiles: int) -> None:
    cute.compile(_warp_search_host, _cute(y, 0), _cute(cu, 0), Int32(num_tiles))(
        y, cu, Int32(num_tiles)
    )


def args_warp_search(inp: dict):
    off = [0]
    for n in inp["lens"]:
        off.append(off[-1] + n)
    tiles = sum((n + 31) // 32 for n in inp["lens"])
    return [torch.zeros(off[-1]), torch.tensor(off, dtype=torch.int32), tiles], {}, {}


# ---------------------------------------- seeded data -> control dependency (rejected)


@cute.kernel
def _data_branch_kernel(mX: cute.Tensor, mY: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if mX[tidx] > Float32(0.0):  # a float loaded from memory decides whether to store
        mY[tidx] = Float32(1.0)


@cute.jit
def _data_branch_host(mX: cute.Tensor, mY: cute.Tensor):
    _data_branch_kernel(mX, mY).launch(grid=(1, 1, 1), block=(128, 1, 1))


def run_data_branch(x, y) -> None:
    cute.compile(_data_branch_host, _cute(x, 0), _cute(y, 0))(x, y)


def args_data_branch(inp: dict):
    return [torch.zeros(128), torch.zeros(128)], {}, {}


# ------------------------------------------ TMA stores and reduces issued by a warp


def _tma_store(mode: str):
    """Warp 0 of each CTA copies TILE x TILE tiles of shared memory to global memory with
    TMA. Modes: "elect", a store from one elected lane; "warp", a store from every lane
    with the same operands (cute.copy without elect_one); "lane_coords", a store from every
    lane, lanes 16-31 storing the CTA's second tile; "lane_smem", a store from every lane,
    lanes 16-31 storing a second shared-memory buffer to the same tile; "reduce", a TMA
    reduce-add from every lane; "reduce_half", the reduce from lanes 0-15;
    "reduce_elect", the reduce from one elected lane; "reduce_lane_coords", the reduce
    from every lane, lanes 16-31 reducing into the CTA's second tile."""
    tiles = 2 if mode.endswith("lane_coords") else 1
    stages = 2 if mode == "lane_smem" else 1
    copy_op = (
        cpasync.CopyReduceBulkTensorTileS2GOp
        if mode.startswith("reduce")
        else cpasync.CopyBulkTensorTileS2GOp
    )

    @cute.kernel
    def kernel(
        tma_atom: cute.CopyAtom, mC_tma: cute.Tensor, smem_layout: cute.ComposedLayout
    ):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        half = cute.arch.lane_idx() // 16
        smem = cutlass.utils.SmemAllocator()
        sC = smem.allocate_tensor(
            cutlass.BFloat16,
            smem_layout.outer,
            byte_alignment=128,
            swizzle=smem_layout.inner,
        )
        tile = cute.arch.block_idx()[0] * tiles
        if cutlass.const_expr(tiles == 2):
            tile = tile + half
        gC = cute.local_tile(mC_tma, (TILE, TILE), (tile, 0))
        tCsC, tCgC = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(gC, 0, 2),
        )
        stage = 0
        if cutlass.const_expr(mode == "lane_smem"):
            stage = half
        if warp == 0:
            if cutlass.const_expr(mode in ("elect", "reduce_elect")):
                with cute.arch.elect_one():
                    cute.copy(tma_atom, tCsC[None, stage], tCgC)
            elif cutlass.const_expr(mode == "reduce_half"):
                if half == 0:
                    cute.copy(tma_atom, tCsC[None, stage], tCgC)
            else:
                cute.copy(tma_atom, tCsC[None, stage], tCgC)

    @cute.jit
    def host(mC: cute.Tensor):
        swizzle = cute.make_swizzle(3, 4, 3)
        tile_layout = cute.make_composed_layout(
            swizzle, 0, cute.make_ordered_layout((TILE, TILE), order=(1, 0))
        )
        staged = cute.make_composed_layout(
            swizzle, 0, cute.make_ordered_layout((TILE, TILE, stages), order=(1, 0, 2))
        )
        tma_atom, mC_tma = cpasync.make_tiled_tma_atom(
            copy_op(), mC, tile_layout, (TILE, TILE)
        )
        grid = (cute.ceil_div(mC.shape[0], TILE * tiles), 1, 1)
        kernel(tma_atom, mC_tma, staged).launch(
            grid=grid, block=(64, 1, 1), smem=stages * TILE * TILE * 2 + 1024
        )

    return host


_TMA_STORE_HOSTS: dict = {}


def run_tma_store(c: torch.Tensor, mode: str) -> None:
    if mode not in _TMA_STORE_HOSTS:
        _TMA_STORE_HOSTS[mode] = _tma_store(mode)
    cute.compile(_TMA_STORE_HOSTS[mode], _cute(c))(c)


def args_tma_store(inp: dict):
    return (
        [torch.zeros((inp["rows"], TILE), dtype=torch.bfloat16)],
        {"mode": inp["mode"]},
        {},
    )


# ---------------------- per-row bytes stored through inline PTX (scale-factor stores)


@dsl_user_op
def _store_2xu8_global(ptr: Int64, lo: Uint8, hi: Uint8, *, loc=None, ip=None) -> None:
    """A scale-factor store helper: two bytes in one st.global.u16."""
    llvm.inline_asm(
        None,
        [
            Int64(ptr).ir_value(loc=loc, ip=ip),
            Uint32(lo).ir_value(loc=loc, ip=ip),
            Uint32(hi).ir_value(loc=loc, ip=ip),
        ],
        "{\n.reg .u16 packed16;\n.reg .u32 lo, hi, packed32;\nand.b32 lo, $1, 0xff;\nshl.b32 hi, $2, 8;\n"
        "or.b32 packed32, lo, hi;\ncvt.u16.u32 packed16, packed32;\nst.global.u16 [$0], packed16;\n}\n",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _red_global_add(ptr: Int64, v: Uint32, *, loc=None, ip=None) -> None:
    """A global reduction in inline PTX: it has no meaning in the op table."""
    llvm.inline_asm(
        None,
        [Int64(ptr).ir_value(loc=loc, ip=ip), Uint32(v).ir_value(loc=loc, ip=ip)],
        "red.global.add.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _sf_store(use_red: bool):
    """Thread t of CTA b writes the two bytes of row 128 b + t of a [rows, 2] uint8 tensor
    (rows past the end are masked), like scale-factor pair stores; use_red seeds a
    `red.global` instead."""

    @cute.kernel
    def kernel(mSF: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        row = cute.arch.block_idx()[0] * 128 + tidx
        if row < mSF.shape[0]:
            addr = (mSF.iterator + cute.crd2idx((row, 0), mSF.layout)).toint()
            if cutlass.const_expr(use_red):
                _red_global_add(addr, Uint32(1))
            else:
                _store_2xu8_global(addr, Uint8(1), Uint8(2))

    @cute.jit
    def host(mSF: cute.Tensor):
        kernel(mSF).launch(
            grid=(cute.ceil_div(mSF.shape[0], 128), 1, 1), block=(128, 1, 1)
        )

    return host


_SF_HOSTS = {True: _sf_store(True), False: _sf_store(False)}


def run_sf_store(sf: torch.Tensor, use_red: bool) -> None:
    cute.compile(_SF_HOSTS[use_red], _cute(sf, 1))(sf)


def args_sf_store(inp: dict):
    return (
        [torch.zeros((inp["rows"], 2), dtype=torch.uint8)],
        {"use_red": inp.get("use_red", False)},
        {},
    )


# ------------------------------------ a kernel that writes the metadata it reads (refused)


@cute.kernel
def _cu_self_write_kernel(mY: cute.Tensor, mCu: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    seg, _, _ = cute.arch.block_idx()
    start = mCu[seg]
    if tidx < mCu[seg + 1] - start:
        mY[start + tidx] = Float32(1.0)
    if tidx == 0:
        mCu[seg] = start + 1  # the kernel rewrites the offsets it steers by


@cute.jit
def _cu_self_write_host(mY: cute.Tensor, mCu: cute.Tensor):
    _cu_self_write_kernel(mY, mCu).launch(
        grid=(mCu.shape[0] - 1, 1, 1), block=(128, 1, 1)
    )


def run_cu_self_write(y, cu) -> None:
    cute.compile(_cu_self_write_host, _cute(y, 0), _cute(cu, 0))(y, cu)


def args_cu_self_write(inp: dict):
    off = [0]
    for n in inp["lens"]:
        off.append(off[-1] + n)
    return [torch.zeros(off[-1]), torch.tensor(off, dtype=torch.int32)], {}, {}


# --------------------------- metadata read through a strided view (contents by offset)


@cute.kernel
def _first_segment_kernel(mY: cute.Tensor, mCu: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    start = mCu[0]
    if tidx < mCu[1] - start:
        mY[start + tidx] = Float32(1.0)


@cute.jit
def _first_segment_host(mY: cute.Tensor, mCu: cute.Tensor):
    _first_segment_kernel(mY, mCu).launch(grid=(1, 1, 1), block=(64, 1, 1))


def run_first_segment(y, cu) -> None:
    # The offsets keep their (static) layout: a stride-2 view stays (4):(2).
    cute.compile(_first_segment_host, _cute(y, 0), from_dlpack(cu, assumed_align=16))(
        y, cu
    )


def args_first_segment(inp: dict):
    offsets = [0, 10, 30, 60]
    if inp["strided"]:  # every other element of a buffer
        cu = torch.tensor([x for o in offsets for x in (o, 99)], dtype=torch.int32)[::2]
    else:
        cu = torch.tensor(offsets, dtype=torch.int32)
    return [torch.zeros(64), cu], {}, {}


# ------------------------------- host code reading what a skipped launch writes, or not


@cute.kernel
def _count_kernel(mSrc: cute.Tensor, mCnt: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        mCnt[0] = mSrc[0]


@cute.jit
def _count_host(mSrc: cute.Tensor, mCnt: cute.Tensor):
    _count_kernel(mSrc, mCnt).launch(grid=(1, 1, 1), block=(32, 1, 1))


@cute.kernel
def _fill_kernel(mOut: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    mOut[bid * 32 + tidx] = mOut.element_type(1.0)


@cute.jit
def _fill_host(mOut: cute.Tensor, num_tiles: Int32):
    _fill_kernel(mOut).launch(grid=(num_tiles, 1, 1), block=(32, 1, 1))


def run_host_reads(src, cnt, out, read: str, how: str, cute_args: bool) -> None:
    """The count kernel copies src[0] to cnt[0]; the host then sizes the fill grid from
    cnt (written by the skipped launch) or from src (only read by it), with item() or the
    shape of repeat_interleave(). The count launch passes the torch tensors, or the CuTe
    runtime tensors wrapping them."""
    c_src, c_cnt = _cute(src, 0), _cute(cnt, 0)
    count = cute.compile(_count_host, c_src, c_cnt)
    if cute_args:
        count(c_src, c_cnt)
    else:
        count(src, cnt)
    data = cnt if read == "cnt" else src
    tiles = (
        torch.repeat_interleave(torch.arange(1), data).numel()
        if how == "repeat_interleave"
        else int(data[0].item())
    )
    n = Int32(tiles)
    cute.compile(_fill_host, _cute(out, 0), n)(out, n)


def args_host_reads(inp: dict):
    src = torch.tensor([4], dtype=torch.int32)
    kwargs = {
        "read": inp["read"],
        "how": inp.get("how", "item"),
        "cute_args": inp.get("cute_args", False),
    }
    return [src, torch.zeros(1, dtype=torch.int32), torch.zeros(128)], kwargs, {}


_COUNTS: dict = {}


def run_cached(src, cnt, out, cache: str) -> None:
    """The first call (input) counts the tiles, with the count kernel or on the host, and
    caches the count; every call sizes the fill grid with it."""
    if cache not in _COUNTS:
        if cache == "kernel":
            cute.compile(_count_host, _cute(src, 0), _cute(cnt, 0))(src, cnt)
            _COUNTS[cache] = cnt
        else:
            _COUNTS[cache] = src.clone()
    n = Int32(int(_COUNTS[cache][0].item()))
    cute.compile(_fill_host, _cute(out, 0), n)(out, n)


def args_cached(inp: dict):
    out = torch.zeros(128, dtype=getattr(torch, inp["dtype"]))
    return (
        [torch.tensor([4], dtype=torch.int32), torch.zeros(1, dtype=torch.int32), out],
        {"cache": inp["cache"]},
        {},
    )


def run_direct(out) -> None:
    """A direct call of a @cute.jit function compiles and launches it (no cute.compile)."""
    _fill_host(_cute(out, 0), Int32(4))


def run_sync(out) -> None:
    """Host code synchronizing the device before a launch."""
    torch.cuda.synchronize()
    n = Int32(4)
    cute.compile(_fill_host, _cute(out, 0), n)(out, n)


def run_numpy_arg(out) -> None:
    """A launch argument that is a CuTe runtime tensor over numpy memory (no torch tensor)."""
    host = _cute(numpy.zeros(out.shape[0], dtype=numpy.float32), 0)
    n = Int32(4)
    cute.compile(_fill_host, host, n)(host, n)


def args_out(inp: dict):
    return [torch.zeros(128)], {}, {}


class _FillTiles:
    """A jit object with a host-selected boolean switch (one of its `flags`)."""

    def __init__(self, full_tiles: bool):
        self.full_tiles = full_tiles

    @cute.jit
    def __call__(self, mOut: cute.Tensor, num_tiles: Int32):
        _fill_kernel(mOut).launch(grid=(num_tiles, 1, 1), block=(32, 1, 1))


def run_fill_tiles(out) -> None:
    """Seeded host bug: every tile is taken as full (the fill kernel has no mask), and the
    grid covers a partial last tile."""
    n = Int32(-(-out.shape[0] // 32))
    cute.compile(_FillTiles(True), _cute(out, 0), n)(out, n)


def args_fill_tiles(inp: dict):
    return [torch.zeros(inp["n"])], {}, {}


@cute.kernel
def _ptr_table_kernel(mX: cute.Tensor, mTable: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if mX[tidx] > Float32(
        0.0
    ):  # not translatable (slice soundness): its write set is unknown
        mX[tidx] = Float32(0.0)


@cute.jit
def _ptr_table_host(mX: cute.Tensor, mTable: cute.Tensor):
    _ptr_table_kernel(mX, mTable).launch(grid=(1, 1, 1), block=(32, 1, 1))


def run_ptr_table(x, y, cu, other, target: str) -> None:
    """A kernel gets the address of cu (or of another buffer) in an int64 pointer table, as
    a grouped GEMM gets its operands; then cu steers the first-segment kernel."""
    table = torch.tensor(
        [(cu if target == "cu" else other).data_ptr()], dtype=torch.int64
    )
    cute.compile(_ptr_table_host, _cute(x, 0), _cute(table, 0))(x, table)
    cute.compile(_first_segment_host, _cute(y, 0), _cute(cu, 0))(y, cu)


def args_ptr_table(inp: dict):
    cu = torch.tensor([0, 10, 30, 60], dtype=torch.int32)
    return (
        [torch.zeros(32), torch.zeros(64), cu, torch.zeros(4, dtype=torch.int32)],
        {"target": inp["target"]},
        {},
    )


def run_env_fill(out) -> None:
    """The grid comes from an environment variable."""
    n = Int32(int(os.environ.get("LKV_CUTE_TEST_TILES", "2")))
    cute.compile(_fill_host, _cute(out, 0), n)(out, n)


def args_env_fill(inp: dict):
    return [torch.zeros(128)], {}, {}


def args_env_alias(inp: dict):
    """A builder reading the environment through an alias the source scan does not see."""
    return [torch.zeros(int(_environ.get("LKV_CUTE_TEST_ALIAS", "128")))], {}, {}


# ------------------------------------------------ a cluster of 2 CTAs


def _cluster_ready(remote: bool):
    """Clusters of two CTAs: after the cluster barrier, an elected thread of warp 0 of each
    CTA arrives on `ready` (count 2) of the cluster's CTA of rank 0, through mapa for rank 1;
    warp 1 of rank 0 waits for both and stores one value per lane; the cluster barrier
    again before exit. Seeded bug: remote=False makes rank 1 arrive on its own `ready`, and
    rank 0's wait never completes."""

    @cute.kernel
    def kernel(mOut: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        smem = cutlass.utils.SmemAllocator()
        ready = smem.allocate_array(cutlass.Int64, 1)
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(ready, 2)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
        if warp == 0:
            with cute.arch.elect_one():
                if cutlass.const_expr(remote):
                    cute.arch.mbarrier_arrive(ready, Int32(0))
                else:
                    cute.arch.mbarrier_arrive(ready)
        if warp == 1:
            if rank == 0:
                cute.arch.mbarrier_wait(ready, 0)
                mOut[cute.arch.block_idx()[0] // 2 * 32 + tidx - 32] = Float32(1.0)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

    @cute.jit
    def host(mOut: cute.Tensor):
        clusters = cute.ceil_div(mOut.shape[0], 32)
        kernel(mOut).launch(
            grid=(2 * clusters, 1, 1), block=(64, 1, 1), cluster=(2, 1, 1)
        )

    return host


_CLUSTER_HOSTS: dict = {}


def run_cluster_ready(out: torch.Tensor, remote: bool) -> None:
    if remote not in _CLUSTER_HOSTS:
        _CLUSTER_HOSTS[remote] = _cluster_ready(remote)
    cute.compile(_CLUSTER_HOSTS[remote], _cute(out, 0))(out)


def args_cluster_ready(inp: dict):
    return [torch.zeros(inp["rows"])], {"remote": inp.get("remote", True)}, {}
