# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CuTe DSL probe kernel of the end-to-end test (cases/probe_cute.json).

One CTA of 128 threads; thread t adds 1 to element t. On a tensor of fewer than 128
elements the threads past its end access out of bounds (a seeded P0-1 bug). `run` is
the case entry (compile with a CuTe tensor, launch with the torch tensor; it rejects more
than 128 elements) and `args` its builder. Loaded from this directory like an exported
source tree.
"""

import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def add_one(m: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    m[tidx] = m[tidx] + 1.0


@cute.jit
def host(m: cute.Tensor):
    add_one(m).launch(grid=(1, 1, 1), block=(128, 1, 1))


def run(x: torch.Tensor) -> None:
    if x.numel() > 128:
        raise ValueError("probe takes at most 128 elements")
    m = from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    cute.compile(host, m)(x)


def args(inp: dict):
    return [torch.zeros(inp["n"], dtype=torch.float32)], {}, {}
