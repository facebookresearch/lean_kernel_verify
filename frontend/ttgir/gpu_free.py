# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Run real Triton host code on a CPU-only machine.

Everything this module changes is part of the trust base and is listed, with its
justification, in references/frontend_ttgir.md ("What dump monkeypatches"). Only device
*queries* and device *placement* are replaced; no host computation (tile tables, grids,
flags, descriptor shapes) is touched. Launches are intercepted separately (capture.py).
"""

from types import SimpleNamespace
from typing import cast

import torch
from torch.overrides import TorchFunctionMode
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase
from triton.runtime import driver as triton_driver

# Analyzed targets (32-thread warps): B200 (sm_100) and GB300 (sm_103).
TARGETS = {"sm_100": GPUTarget("cuda", 100, 32), "sm_103": GPUTarget("cuda", 103, 32)}
# CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, the same on both targets (Triton's
# offline table in third_party/nvidia/backend/compiler.py).
MAX_SHARED_MEM = 232448


class CpuOnlyCudaDriver:
    """Stand-in for triton's CudaDriver: answers device/stream/target queries only.

    Any other attribute access raises AttributeError, so host code that needs more of
    the driver fails loudly instead of being silently faked.
    """

    def __init__(self, num_sms: int, target: GPUTarget):
        self.utils = SimpleNamespace(get_device_properties=self._props)
        self._num_sms = num_sms
        self._target = target

    def _props(self, device):
        return {"multiprocessor_count": self._num_sms, "max_shared_mem": MAX_SHARED_MEM}

    def get_current_target(self):
        return self._target

    def get_current_device(self):
        return 0

    def get_current_stream(self, device=None):
        return 0

    def get_active_torch_device(self):
        return torch.device("cpu")


class CudaOnCpu(TorchFunctionMode):
    """Serve every CUDA placement request on CPU; make pin_memory a no-op.

    Values are unchanged: only the `device` of created/moved tensors differs.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        if func in (torch.Tensor.pin_memory, torch.Tensor.cuda):
            return args[0]
        if kwargs.get("device") is not None and _is_cuda(kwargs["device"]):
            kwargs["device"] = "cpu"
        kwargs.pop("pin_memory", None)
        args = tuple("cpu" if _is_cuda(a) else a for a in args)
        return func(*args, **kwargs)


def _is_cuda(x) -> bool:
    if isinstance(x, torch.device):
        return x.type == "cuda"
    return isinstance(x, str) and x.split(":")[0] == "cuda"


def install(num_sms: int, target: str) -> CudaOnCpu:
    """Install the device-query shims for `target` (a TARGETS key); returns the placement
    mode (use as a context).

    Call before importing kernel modules: `@triton.jit` reads the C-cache knob at
    decoration time, and the C launch fast path would bypass JITFunction.run. Every
    compile runs (always_compile): Triton's cache is keyed by source content, not path,
    so a hit would return TTGIR whose locs name another export of the same file.
    """
    gpu = TARGETS[target]
    # The stand-in implements only the queries host code makes, deliberately not the rest
    # of DriverBase (see CpuOnlyCudaDriver), so it is not a DriverBase subclass.
    triton_driver.set_active(cast(DriverBase, CpuOnlyCudaDriver(num_sms, gpu)))
    knobs.nvidia.enable_c_cache = False
    knobs.compilation.always_compile = True
    major, minor = divmod(
        int(gpu.arch), 10
    )  # TARGETS are CUDA targets: an integer arch
    props = SimpleNamespace(
        multi_processor_count=num_sms,
        major=major,
        minor=minor,
        name=f"cpu-only {target} stand-in",
        shared_memory_per_block_optin=MAX_SHARED_MEM,
    )
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 1
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_capability = lambda device=None: (props.major, props.minor)
    torch.cuda.get_device_properties = lambda device=None: props
    return CudaOnCpu()
