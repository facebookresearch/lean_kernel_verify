# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""TLX probe kernel of the end-to-end test (cases/probe_tlx.json).

The default partition does a masked load/store of one BLOCK and arrives on an mbarrier
that a 1-warp partition waits on. `run` launches cdiv(n, BLOCK) programs; with
`floor_grid=True` it launches n // BLOCK and leaves the tail of a partial input unwritten
(a seeded P0-3 bug). It rejects a tensor that is not 1-D, and with `skip_empty=True` it
returns before the launch on an empty tensor. Loaded from this directory like an exported
source tree.
"""

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def probe(x_ptr, n, BLOCK: tl.constexpr):
    bars = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    b = tlx.local_view(bars, 0)
    with tlx.async_tasks():
        with tlx.async_task("default"):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            v = tl.load(x_ptr + offs, mask=offs < n)
            tl.store(x_ptr + offs, v + 1, mask=offs < n)
            tlx.barrier_arrive(b)
        with tlx.async_task(num_warps=1):
            tlx.barrier_wait(b, 0)


def run(x, floor_grid=False, BLOCK=128, skip_empty=False):
    if x.dim() != 1:
        raise ValueError("probe takes a 1-D tensor")
    if skip_empty and x.numel() == 0:
        return
    programs = x.numel() // BLOCK if floor_grid else triton.cdiv(x.numel(), BLOCK)
    probe[(programs,)](x, x.numel(), BLOCK=BLOCK)
