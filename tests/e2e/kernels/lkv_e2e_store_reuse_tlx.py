# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""TLX kernels of P1-2 "async store sources reuse ordered" (references/specs.md, "P1-2";
cases/store_reuse_staging_tlx.json and cases/store_reuse_rebuild_tlx.json).

`staging`: the default partition (4 warps) writes TILES tiles of its row block through one
shared-memory staging buffer: each tile is a `local_store` into the buffer, a fence, a
named barrier of the partition and a TMA store of the buffer (host descriptor, issued by
one elected thread). DRAIN=1 waits for the previous store to have read the buffer
(`cp.async.bulk.wait_group.read 0`) and syncs the partition before the buffer is written
again; DRAIN=0 writes it right away: the classic staging-buffer race (P1-2 F1).

`rebuild`: the default partition processes ITEMS work items; each item builds an
in-kernel tensor descriptor (`tl.make_tensor_descriptor`: its tensormap is copied into the
CTA's global scratch slot, the same slot in every item) and stores one tile through it.
Before the tile is written the partition waits for the previous store to have read the
buffer and syncs, in every variant, so only the tensormap may race. DRAIN=0 rebuilds the
descriptor before any wait: the previous item's store may still read the tensormap
(P1-2 F2, the shape of the descriptor-rebuild race); DRAIN=1 waits for the previous
stores' completion (`tlx.async_descriptor_store_wait(0)`, `cp.async.bulk.wait_group 0`) and
syncs before the rebuild; DRAIN=2 waits with `.read` only (the PTX ISA:
cp.async.bulk.wait_group.read waits for the reads of the tensormap too).

The second partition (one warp) only waits on an mbarrier the default partition arrives on
at its end. Loaded from this directory like an exported source tree; `run_*` are the
cases' host entries.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _wait_read0(dummy):
    return tl.inline_asm_elementwise(
        "cp.async.bulk.wait_group.read 0;\n\tmov.b32 $0, $1;",
        "=r,r",
        [dummy],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def staging(
    x_desc,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    TILES: tl.constexpr,
    DRAIN: tl.constexpr,
):
    buf = tlx.local_alloc((ROWS, COLS), tl.float16, 1)
    bars = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    done = tlx.local_view(bars, 0)
    with tlx.async_tasks():
        with tlx.async_task("default"):
            row = tl.program_id(0) * ROWS
            for t in tl.range(0, TILES):
                if DRAIN:
                    _wait_read0(t)
                    tlx.named_barrier_wait(9, 128)
                vals = tl.full((ROWS, COLS), 1.0, tl.float16) * t.to(tl.float16)
                tlx.local_store(buf[0], vals)
                tlx.fence_async_shared()
                tlx.named_barrier_wait(9, 128)
                tlx.async_descriptor_store(x_desc, buf[0], [row, t * COLS])
            tlx.async_descriptor_store_wait(0)
            tlx.barrier_arrive(done)
        with tlx.async_task(num_warps=1):
            tlx.barrier_wait(done, 0)


@triton.jit
def rebuild(
    out_ptr,
    n_rows,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    ITEMS: tl.constexpr,
    DRAIN: tl.constexpr,
):
    buf = tlx.local_alloc((ROWS, COLS), tl.float16, 1)
    bars = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    done = tlx.local_view(bars, 0)
    with tlx.async_tasks():
        with tlx.async_task("default"):
            row = tl.program_id(0) * ROWS
            for it in tl.range(0, ITEMS):
                if DRAIN == 1:
                    tlx.async_descriptor_store_wait(0)
                    tlx.named_barrier_wait(9, 128)
                if DRAIN == 2:
                    _wait_read0(it)
                    tlx.named_barrier_wait(9, 128)
                desc = tl.make_tensor_descriptor(
                    out_ptr,
                    shape=[n_rows, COLS * ITEMS],
                    strides=[COLS * ITEMS, 1],
                    block_shape=[ROWS, COLS],
                )
                _wait_read0(it)
                tlx.named_barrier_wait(9, 128)
                vals = tl.full((ROWS, COLS), 1.0, tl.float16) * it.to(tl.float16)
                tlx.local_store(buf[0], vals)
                tlx.fence_async_shared()
                tlx.named_barrier_wait(9, 128)
                tlx.async_descriptor_store(desc, buf[0], [row, it * COLS])
            tlx.async_descriptor_store_wait(0)
            tlx.barrier_arrive(done)
        with tlx.async_task(num_warps=1):
            tlx.barrier_wait(done, 0)


def run_staging(x, drain=1, ROWS=64, COLS=64):
    """x: [blocks * ROWS, TILES * COLS] float16, one program per row block."""
    if x.dim() != 2 or x.shape[0] % ROWS or x.shape[1] % COLS:
        raise ValueError("staging takes a 2-D tensor of whole tiles")
    desc = TensorDescriptor.from_tensor(x, block_shape=[ROWS, COLS])
    staging[(x.shape[0] // ROWS,)](
        desc, ROWS=ROWS, COLS=COLS, TILES=x.shape[1] // COLS, DRAIN=drain
    )


def run_rebuild(out, drain=1, ROWS=64, COLS=64):
    """out: [blocks * ROWS, ITEMS * COLS] float16, one program per row block."""
    if out.dim() != 2 or out.shape[0] % ROWS or out.shape[1] % COLS:
        raise ValueError("rebuild takes a 2-D tensor of whole tiles")

    def alloc_fn(size: int, alignment: int, stream: int | None) -> torch.Tensor:
        return torch.empty(size, device=out.device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    rebuild[(out.shape[0] // ROWS,)](
        out, out.shape[0], ROWS=ROWS, COLS=COLS, ITEMS=out.shape[1] // COLS, DRAIN=drain
    )
