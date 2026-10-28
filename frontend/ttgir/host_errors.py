# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Which exceptions of a host call are the host refusing its input (both frontends).

A dump calls the host entry point once per input. The host refuses an input when the
exported sources raise: the innermost frame of the exception is theirs (a `raise`, a
failed `assert`, a builtin they call), or a torch check they call fails (TORCH_CHECKS: on
the CPU host they raise at once; on the GPU `torch._assert_async` fails the stream before
the next launch runs). The frontend's torch modes (gpu_free.CudaOnCpu, host_reads.HostReads)
sit between a source line and the torch function it calls, so a check called through them
is recognized by the `func` their `__torch_function__` dispatches. Any other exception (a
failing torch op, a CUDA call on the CPU-only host, an error of the compiler or of the
frontend) is not a decision of the host and fails the dump.
"""

from pathlib import Path

import torch

TORCH_CHECKS = (
    torch._assert_async,
    torch._assert,
    torch._check,
    torch._check_index,
    torch._check_is_size,
    torch._check_not_implemented,
    torch._check_type,
    torch._check_value,
    torch._check_with,
)
_CHECK_CODES = frozenset(f.__code__ for f in TORCH_CHECKS if hasattr(f, "__code__"))


def host_rejection(e: BaseException, root: Path) -> str | None:
    """`Type: message at file:line` (the innermost frame under the export root `root`) if
    the exported sources refused the input by raising `e`, else None."""
    root = Path(root).resolve()
    frames = []
    tb = e.__traceback__
    while tb is not None:
        frames.append((tb.tb_frame, tb.tb_lineno))
        tb = tb.tb_next
    mine = [i for i, (f, _) in enumerate(frames) if _path(f).is_relative_to(root)]
    if not mine:
        return None
    last = mine[-1]
    if last + 1 < len(frames) and not _is_check(frames[last + 1][0]):
        return None
    frame, line = frames[last]
    return f"{type(e).__name__}: {e} at {_path(frame).relative_to(root)}:{line}"


def _path(frame) -> Path:
    return Path(frame.f_code.co_filename).resolve()


def _is_check(frame) -> bool:
    """The callee frame of a source line is a torch check, or a torch mode dispatching one."""
    if frame.f_code in _CHECK_CODES:
        return True
    if frame.f_code.co_name != "__torch_function__":
        return False
    func = frame.f_locals.get("func")
    return any(func is check for check in TORCH_CHECKS)
