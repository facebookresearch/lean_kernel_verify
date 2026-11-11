# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""GPU-only calls replaced so a kernel's real host code path runs on a CPU-only host.

Every replacement is part of the trust base (references/frontend_cute.md, "Host
patches"); nothing else on the host path is changed:

  P1 torch.cuda.is_available / device_count / get_device_properties /
     get_device_capability / current_device: one SM100 device (compute capability 10.0)
     with `num_sms` SMs and 232448 bytes of opt-in shared memory per block; a property
     it lacks fails the dump (host_errors: not a host rejection).
  P2 cutlass.utils.HardwareInfo: `num_sms` SMs, `num_sms // cluster_size` clusters.
  P3 torch.Tensor.is_cuda is True: host validation demands CUDA tensors; the tensors are
     CPU tensors with the same shapes, strides, dtypes and integer contents.
  P4 DLPack capsules made by cute.runtime.from_dlpack report device kDLCUDA, so the DSL
     types those tensors exactly as CUDA tensors (global memory) at compile time. Each
     runtime tensor remembers the torch tensor it wraps (its memory).
  P5 BaseDSL.compile_and_cache records the initial MLIR it is handed (the DSL has
     verified it; with locations) and the pass pipeline the DSL computes for the compile
     (its own _get_pipeline + preprocess_pipeline with the target arch): these are what
     the variant digest and the analyzed stage (stage.py) come from. It then lets the DSL
     compile that module with its own full pipeline, with line info and the PTX and cubin
     kept (compile options set on the compile itself, not CUTE_DSL_* variables: the
     environment would change the initial MLIR and the recorded pipeline), without
     compile caches, and build the JIT engine of P9. With `lineinfo_check` (DSL 4.4.2),
     a copy of the module is first compiled without line info, to check that line info
     does not change the code. A compile that fails here keeps its MLIR (ptx_error).
  P6 cute.compile returns a launcher: each call records the host's launch arguments (a
     runtime tensor as the torch tensor it wraps), with the sources of each tensor
     argument's data and the storages its integer arguments point into (Recording.reads,
     the run's ttgir.host_reads.HostReads, which dump.py activates around each host
     call), then calls the compiled function: the DSL's JIT host code runs natively and
     its kernel launches reach P9 (Launch.records). A failure of that run is recorded on
     the launch (capture_error); the recorded arguments do not depend on it.
  P8 torch.Tensor.pin_memory returns a copy of the tensor (is_pinned is True): page-
     locked host memory needs a driver, and pinning changes how copies to the device
     run, not the data.
  P9 The CUDA runtime under the JIT host code is replaced by callbacks
     (ExecutionEngine.raw_register_runtime), one per runtime function the module
     declares (_HostRuntime): libraries and kernels load as handles, device queries are
     answered for P1's device (_device_attrs), the static shared memory of a kernel from
     its cubin, and _cudaLaunchKernelEx records grid, block, cluster, dynamic shared
     memory, launch attributes and the bytes of every kernel parameter (sized by the
     kernel's PTX .entry, frontend/ptx/launch.py) instead of launching. A declared
     function without a callback stops the engine; an unknown attribute, an unknown
     kernel or any other call the callbacks do not model fails the launch. The DSL's
     runtime library is not loaded (get_shared_libs is empty), its cuInit check is
     skipped and there is one device; the libraries the host code loads are handles that
     are never unloaded (a real driver would crash on them), and the engine is built
     without the module's global destructors (the unload; teardown is not captured). A
     TVM-FFI function is made from that engine, as DSL 4.4.2 does (DSL 4.6.1 re-links
     the module with JITLink against its runtime library).
  P10 TVM-FFI (`--enable-tvm-ffi`): the TVM-FFI tensor of a runtime tensor, and every
     torch tensor a TVM-FFI function is called with, is made from a DLPack capsule marked
     kDLCUDA (P4 for TVM-FFI), so the generated wrapper takes them as device tensors (its
     device checks pass and it finds its environment stream).
  P11 While cute.compile traces the jit function, BaseDSL.generate_execution_arguments is
     observed, not changed: each block argument of the jit function is named by the
     compile parameter and leaf of the argument the DSL made from it (_host_arguments).
     After the trace, a copy of the initial MLIR goes through cute-remove-static-args, the
     first pass of the DSL's pipeline: the operands each kernel launch keeps are the
     kernel's parameters, each traced back to the jit arguments (_kernel_params). Every
     LaunchRecord of the kernel gets that as "jit_params" (_name_params): a parameter that
     passes a jit argument through is named by position, not by resolving its address
     (which cannot tell one tensor passed under two jit names apart, nor see a zero-length
     tensor, whose data pointer is 0), and a derived one (a TMA descriptor, a view) is
     known to come from some jit arguments only.
"""

import ctypes
import dataclasses
import inspect
import os
import re
import shutil
import struct
import tempfile
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cutlass
import torch
from cutlass import Numeric
from cutlass._mlir import ir as _mlir_ir, passmanager as _passmanager
from cutlass.base_dsl.dsl import extract_mlir_values

from ..ptx import launch as ptx_launch


KDLCUDA = 2
METADATA_LIMIT = 1 << 16  # integer tensors up to this many elements keep their contents
# The initial MLIR is kept in DSL 4.4.2's custom form (the text of its CUTE_DSL_KEEP_IR file)
# and in generic form otherwise: DSL 4.6.1's custom form of nvvm.barrier with a location
# does not parse back.
GENERIC_FORM = cutlass.__version__ != "4.4.2"
# CUtensorMap words DSL 4.6.1's software encoder never writes: its lowered host code
# zeroes and fills w0..w7 of a 16-word alloca, so w8..w15 hold stale stack bytes. They are
# decoded as zero and the record says so ("unwritten_words"); DSL 4.4.2 writes all 16.
UNWRITTEN_TMAP_WORDS = [] if cutlass.__version__ == "4.4.2" else list(range(8, 16))
# Shared memory the device reserves per CTA (SM90+), included in a cubin's .nv.shared.*.
RESERVED_SMEM = 1024
# Runtime error code the callbacks return for a call they do not model (cudaErrorUnknown).
CUDA_ERROR = 999
# Functions the JIT host code may take from the process instead of a callback: libc's
# printf and sprintf (host-side diagnostics) and TVM-FFI's own C API (error reporting,
# the environment stream: none is set, so kernels get the default stream).
PROCESS_SYMBOLS = ("printf", "sprintf")
PROCESS_PREFIXES = ("TVMFFI",)
# Every compile's _HostRuntime, for the life of the process: its callbacks must outlive
# the JIT engine, which may call them until it is collected.
_RUNTIMES: list = []
# The MLIR bindings (native, without stubs; stage.py).
ir: Any = _mlir_ir
passmanager: Any = _passmanager


@dataclass
class PtxCapture:
    """What the DSL's own compile of a module produced (P5)."""

    ptx: str | None = None  # line info on
    cubin: bytes | None = None
    ptx_noline: str | None = None  # without line info (lineinfo_check)
    lineinfo_error: str | None = None  # why that compile failed
    error: str | None = None  # why the compile or its JIT engine failed
    tvm_ffi: bool = False  # compiled with --enable-tvm-ffi


class _Captured(Exception):
    """Raised by the patched compile_and_cache with the initial MLIR and pipeline, and
    with what the DSL's own compile produced from them (P5)."""

    def __init__(self, mlir: str, pipeline: str):
        super().__init__("initial MLIR captured before the DSL's pass pipeline")
        self.mlir = mlir
        self.pipeline = pipeline
        self.capture = PtxCapture()
        self.compiled = None  # the compiled function (the DSL's JIT executor)
        self.runtime: _HostRuntime | None = None
        self.kernel_params: dict | str = {}  # _kernel_params (P11)


@dataclass
class Compile:
    index: int
    callable: str  # module.qualname of the jit function or class
    params: list  # [{"name", "constexpr", "leaves"}] per compile argument
    flags: (
        dict  # boolean attributes of the jit object and its config (variant switches)
    )
    mlir: str  # initial MLIR written by the DSL
    pipeline: str  # the DSL's pass pipeline for this compile
    capture: PtxCapture = field(default_factory=PtxCapture)


@dataclass
class Launch:
    compile: int
    args: list
    hinted: list  # storages its integer arguments point into (address hints, ttgir.host_reads)
    # The LaunchRecords of the kernels its compiled host code launched.
    records: list = field(default_factory=list)
    capture_error: str | None = None  # why running the compiled host code failed


@dataclass
class Recording:
    compiles: list = field(default_factory=list)
    launches: list = field(default_factory=list)
    reads: object = None  # the run's HostReads


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLTensorHead(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice)]


_capsule_name = ctypes.pythonapi.PyCapsule_GetName
_capsule_name.restype = ctypes.c_char_p
_capsule_name.argtypes = [ctypes.py_object]
_capsule_ptr = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_ptr.restype = ctypes.c_void_p
_capsule_ptr.argtypes = [ctypes.py_object, ctypes.c_char_p]


def _mark_capsule_cuda(capsule) -> None:
    name = _capsule_name(capsule)
    if name == b"dltensor":
        offset = 0  # DLManagedTensor starts with its DLTensor
    elif name == b"dltensor_versioned":
        offset = 32  # DLManagedTensorVersioned: version, manager_ctx, deleter, flags
    else:
        raise RuntimeError(f"unexpected DLPack capsule {name!r}")
    head = _DLTensorHead.from_address(_capsule_ptr(capsule, name) + offset)
    head.device.device_type = KDLCUDA
    head.device.device_id = 0


def _cuda_ffi_tensor(t: torch.Tensor):
    """A TVM-FFI tensor over t's memory whose device is CUDA 0 (P10)."""
    import tvm_ffi

    capsule = t.__dlpack__()
    _mark_capsule_cuda(capsule)
    return tvm_ffi.from_dlpack(capsule)


def _ffi_args(values):
    """Call arguments of a TVM-FFI function with every torch tensor as _cuda_ffi_tensor."""
    if isinstance(values, (tuple, list)):
        items = [_ffi_args(v) for v in values]
        return (
            type(values)(*items) if hasattr(values, "_fields") else type(values)(items)
        )
    return _cuda_ffi_tensor(values) if isinstance(values, torch.Tensor) else values


def describe(value):
    """JSON description of a launch argument. Integer tensors keep their contents by
    element offset from the data pointer (KIR `load` indices are offsets), the elements
    between those of a strided view included."""
    if value is None:
        return {"kind": "none"}
    if isinstance(value, torch.Tensor):
        d = {
            "kind": "tensor",
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "storage": [
                value.untyped_storage().data_ptr(),
                value.untyped_storage().nbytes(),
            ],
            "byte_offset": value.storage_offset() * value.element_size(),
        }
        span = (
            1 + sum((n - 1) * s for n, s in zip(value.shape, value.stride()))
            if value.numel()
            else 0
        )
        if value.dtype in (torch.int32, torch.int64) and span <= METADATA_LIMIT:
            d["data"] = value.as_strided((span,), (1,)).tolist()
        return d
    if isinstance(
        value, Numeric
    ):  # a DSL scalar (Int32, Float32, ...) passed at launch
        value = value.value
    if isinstance(value, bool):
        return {"kind": "scalar", "type": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "scalar", "type": "int", "value": value}
    if isinstance(value, float):
        return {"kind": "scalar", "type": "float", "value": value}
    if isinstance(value, (tuple, list)):
        return {"kind": "tuple", "items": [describe(v) for v in value]}
    return {"kind": "opaque", "type": type(value).__name__}


def _ints(d: dict) -> list:
    """The values of a described launch argument that may be byte addresses: an integer
    scalar, the recorded contents of an int64 tensor."""
    if d["kind"] == "scalar":
        return [d["value"]] if d["type"] == "int" else []
    return d.get("data", []) if d["kind"] == "tensor" and d["dtype"] == "int64" else []


def _torch_args(values, runtime_tensor: type):
    """Launch arguments with every CuTe runtime tensor (`runtime_tensor`, from_dlpack)
    replaced by the torch tensor whose memory it wraps (P4)."""
    if isinstance(values, (tuple, list)):
        items = [_torch_args(v, runtime_tensor) for v in values]
        return (
            type(values)(*items) if hasattr(values, "_fields") else type(values)(items)
        )
    if isinstance(values, runtime_tensor):
        if getattr(values, "_lkv_source", None) is None:
            raise RuntimeError(
                "a launch argument is a CuTe runtime tensor that wraps no torch tensor: the "
                "dump cannot track its memory"
            )
        return values._lkv_source
    return values


def _flat(values) -> list:
    """Launch arguments and their descriptions in one order (tuples flattened)."""
    out = []
    for v in values:
        if isinstance(v, (tuple, list)):
            out += _flat(v)
        elif isinstance(v, dict) and v.get("kind") == "tuple":
            out += _flat(v["items"])
        else:
            out.append(v)
    return out


def _leaves(name: str, value) -> list:
    """(name, value) of the non-None leaves of a compile or launch argument; tuples and
    namedtuples flatten in order, as the DSL flattens them into host-function
    arguments."""
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        fields = getattr(value, "_fields", None) or [str(i) for i in range(len(value))]
        return [x for f, v in zip(fields, value) for x in _leaves(f"{name}_{f}", v)]
    return [(name, value)]


def _leaf_names(name: str, value) -> list:
    return [n for n, _ in _leaves(name, value)]


def _positional(fn, args, kwargs, compile_call: bool = False) -> list:
    """`args` and `kwargs` of a call of the jit function `fn` (a function or an object
    whose `__call__` is jitted) as one positional list in parameter order, defaults
    applied; the arguments unchanged when there are no keyword arguments for `fn` (a
    kernel may pass compile and launch arguments by keyword). For `cute.compile` (`compile_call`),
    keywords that are not parameters of `fn` are cute.compile's own (`options=...`) and are
    left out. A call that does not fit the signature is a frontend error (RuntimeError
    raised here, not a host rejection)."""
    method = not inspect.isfunction(fn)
    sig = inspect.signature(type(fn).__call__ if method else fn)
    if compile_call:
        kwargs = {
            k: v for k, v in kwargs.items() if k in sig.parameters and k != "self"
        }
    if not kwargs:
        return list(args)
    try:
        bound = sig.bind(*((None,) if method else ()), *args, **kwargs)
    except TypeError as e:
        raise RuntimeError(
            f"recorded call of {_qualname(fn)} does not fit its signature: {e}"
        ) from None
    bound.apply_defaults()
    return list(bound.args[1:] if method else bound.args)


def _params(fn, args) -> list:
    target = fn if inspect.isfunction(fn) else type(fn).__call__
    sig = inspect.signature(target)
    names = [p for p in sig.parameters if p != "self"]
    out = []
    for i, a in enumerate(args):
        name = names[i] if i < len(names) else f"arg{i}"
        ann = sig.parameters[name].annotation if name in sig.parameters else None
        out.append(
            {
                "name": name,
                "constexpr": "Constexpr" in str(ann),
                "leaves": _leaf_names(name, a),
            }
        )
    return out


def _flags(obj, prefix: str = "", depth: int = 0) -> dict:
    """Boolean attributes of a jit object and of its dataclass-valued attributes (two
    levels: `config.elide_full_tile_mask`, `config.optimizations.use_early_k_release`):
    the host-selected variant switches."""
    if inspect.isfunction(obj):
        return {}
    items = (
        vars(obj)
        if hasattr(obj, "__dict__")
        else {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    )
    out = {}
    for k, v in sorted(items.items()):
        if isinstance(v, bool):
            out[prefix + k] = v
        elif dataclasses.is_dataclass(v) and not isinstance(v, type) and depth < 2:
            out.update(_flags(v, f"{prefix}{k}.", depth + 1))
    return out


def _qualname(fn) -> str:
    obj = fn if inspect.isfunction(fn) else type(fn)
    return f"{obj.__module__}.{obj.__qualname__}"


def _device_attrs(num_sms: int) -> dict:
    """cudaDeviceAttr values of P1's device: 16 multiProcessorCount, 75/76 compute
    capability, 97 maxSharedMemoryPerBlockOptin (227 KiB on SM100)."""
    return {16: num_sms, 75: 10, 76: 0, 97: 232448}


def static_smem(cubin: bytes, entry: str) -> int:
    """The static shared memory of kernel `entry` in an ELF cubin, as cuFuncGetAttribute
    reports it (SHARED_SIZE_BYTES): its .nv.shared.<entry> section without the 1 KiB the
    device reserves at the bottom of every CTA's shared memory, which the section starts
    with (cudaDevAttrReservedSharedMemoryPerBlock; measured against the driver on GB300,
    driver 13000); 0 without the section."""
    (shoff,) = struct.unpack_from("<Q", cubin, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", cubin, 0x3A)

    def section(i: int) -> tuple:  # (name offset, type, flags, addr, offset, size, ...)
        return struct.unpack_from("<IIQQQQIIQQ", cubin, shoff + i * shentsize)

    names = section(shstrndx)[4]
    want = f".nv.shared.{entry}\0".encode()
    for i in range(shnum):
        name, size = section(i)[0], section(i)[5]
        if cubin[names + name : names + name + len(want)] == want:
            if size < RESERVED_SMEM:
                raise RuntimeError(f"{entry}: .nv.shared section of {size} bytes")
            return size - RESERVED_SMEM
    return 0


class _LaunchAttr(ctypes.Structure):  # cudaLaunchAttribute
    _fields_ = [
        ("id", ctypes.c_int32),
        ("pad", ctypes.c_uint8 * 4),
        ("val", ctypes.c_uint8 * 64),
    ]


class _LaunchConfig(ctypes.Structure):  # cudaLaunchConfig_t
    _fields_ = [
        ("grid", ctypes.c_uint32 * 3),
        ("block", ctypes.c_uint32 * 3),
        ("smem", ctypes.c_uint64),
        ("stream", ctypes.c_void_p),
        ("attrs", ctypes.POINTER(_LaunchAttr)),
        ("num_attrs", ctypes.c_uint32),
    ]


_P, _I32, _I64 = ctypes.c_void_p, ctypes.c_int32, ctypes.c_int64


class _HostRuntime:
    """The CUDA runtime functions the JIT host code of one compile calls (P9), as ctypes
    callbacks. Kernels are handles named by _cudaLibraryGetKernel; launches append to
    `launches` (raw LaunchRecords), and a call the callbacks cannot answer exactly
    returns CUDA_ERROR with the reason in `errors`."""

    def __init__(self, num_sms: int):
        self.device_attrs = _device_attrs(num_sms)
        self.ptx: str | None = None
        self.cubin: bytes | None = None
        self.kernels: dict = {}  # handle -> PTX .entry name
        self.kernel_attrs: dict = {}  # entry -> {cudaFuncAttribute (str): value} set
        self.libraries = 0
        self.launches: list = []
        self.errors: list = []
        # name: (result type, argument types)
        sigs: dict[str, tuple[Any, list[Any]]] = {
            "_cudaLaunchKernelEx": (_I32, [_P, _P, _P]),
            "_cudaLaunchKernel": (_I32, [_P, *[_I32] * 6, _P, _I64, _P]),
            "_cudaLibraryLoadData": (_I32, [_P, _P, _P, _P, _I32, _P, _P, _I32]),
            "_cudaLibraryGetKernel": (_I32, [_P, _P, ctypes.c_char_p]),
            "_cuKernelGetAttribute": (_I32, [_P, _I32, _P, _I32]),
            "_cudaDeviceGetAttribute": (_I32, [_P, _I32, _I32]),
            "_cudaGetDevice": (_I32, [_P]),
            "_cudaSetDevice": (_I32, [_I32]),
            "_cudaKernelSetAttributeForDevice": (_I32, [_P, _I32, _I32, _I32]),
            "_cudaFuncSetAttribute": (_I32, [_P, _I32, _I32]),
            "_cudaOccupancyMaxActiveBlocksPerMultiprocessor": (
                _I32,
                [_P, _P, _I32, _I64],
            ),
            "cuda_dialect_init_library_once": (_I32, [_P, _P, _P, _P]),
            "cuda_dialect_unload_library_once": (None, [_P]),
            "cuda_dialect_get_error_name": (_P, [_I32]),
        }
        self.callbacks = {
            name: ctypes.CFUNCTYPE(res, *args)(self._guarded(getattr(self, name), res))
            for name, (res, args) in sigs.items()
        }
        self._error_name = ctypes.create_string_buffer(
            b"lkv: CUDA runtime call not modeled on the CPU host (see the dump)"
        )

    def register(self, engine, declared: list) -> None:
        """Register a callback for every runtime function the module declares; one the
        callbacks do not cover is an error, unless the process provides it."""
        unknown = [
            n
            for n in declared
            if n not in self.callbacks
            and n not in PROCESS_SYMBOLS
            and not n.startswith(PROCESS_PREFIXES)
        ]
        if unknown:
            raise RuntimeError(
                f"the JIT host code declares {', '.join(unknown)}, which the CPU host "
                "runtime does not model (host_patches.py, P9)"
            )
        for name in declared:
            if name in self.callbacks:
                engine.raw_register_runtime(
                    name, ctypes.cast(self.callbacks[name], ctypes.c_void_p)
                )

    def _guarded(self, method, restype):
        def call(*args):
            try:
                return method(*args)
            except Exception as e:  # noqa: BLE001 - reported through the launch
                self.errors.append(f"{method.__name__}: {type(e).__name__}: {e}")
                return CUDA_ERROR if restype is _I32 else None

        return call

    def _entry(self, handle) -> str:
        if handle not in self.kernels:
            raise RuntimeError(f"unknown kernel handle {handle:#x}")
        return self.kernels[handle]

    def _record(self, handle, cfg, cluster: list, attrs: dict, args_p) -> int:
        entry = self._entry(handle)
        params = ptx_launch.ptx_params(self.ptx or "", entry)
        argv = (ctypes.c_void_p * len(params)).from_address(args_p) if params else []
        for p, ptr in zip(params, argv):
            if not ptr:
                raise RuntimeError(f"{entry}: no argument for parameter {p['i']}")
            p["bytes"] = ctypes.string_at(ptr, p["size"]).hex()
        self.launches.append(
            {
                "entry": entry,
                "grid": list(cfg.grid),
                "block": list(cfg.block),
                "cluster": cluster,
                "smem_dynamic": cfg.smem,
                "attrs": attrs,
                "kernel_attrs": dict(self.kernel_attrs.get(entry, {})),
                "params": params,
            }
        )
        return 0

    # The callbacks, named as the runtime functions they replace.

    def _cudaLaunchKernelEx(self, cfg_p, handle, args_p) -> int:
        cfg = _LaunchConfig.from_address(cfg_p)
        cluster = [1, 1, 1]
        attrs: dict[str, Any] = {"pdl": False, "cooperative": False}
        for i in range(cfg.num_attrs):
            a = cfg.attrs[i]
            val = bytes(a.val)
            if a.id == 2:  # cudaLaunchAttributeCooperative
                attrs["cooperative"] = bool(struct.unpack_from("<i", val)[0])
            elif a.id == 4:  # cudaLaunchAttributeClusterDimension
                cluster = list(struct.unpack_from("<3I", val))
            elif a.id == 6:  # cudaLaunchAttributeProgrammaticStreamSerialization
                attrs["pdl"] = bool(struct.unpack_from("<i", val)[0])
            elif a.id == 11:  # cudaLaunchAttributePreferredClusterDimension
                attrs["preferred_cluster"] = list(struct.unpack_from("<3I", val))
            else:
                raise RuntimeError(f"launch attribute {a.id} is not modeled")
        return self._record(handle, cfg, cluster, attrs, args_p)

    def _cudaLaunchKernel(self, handle, gx, gy, gz, bx, by, bz, args_p, smem, stream):
        raise RuntimeError(
            "_cudaLaunchKernel is not modeled (the DSL launches with Ex)"
        )

    def _cudaLibraryLoadData(self, out, code, *options) -> int:
        self.libraries += 1
        ctypes.c_void_p.from_address(out).value = 0x1000 + self.libraries
        return 0

    def _cudaLibraryGetKernel(self, out, library, name) -> int:
        handle = 0x100000 + len(self.kernels)
        self.kernels[handle] = name.decode()
        ctypes.c_void_p.from_address(out).value = handle
        return 0

    def _cuKernelGetAttribute(self, out, attr, handle, device) -> int:
        if attr != 1:  # CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
            raise RuntimeError(f"kernel attribute {attr} is not modeled")
        if self.cubin is None:
            raise RuntimeError("no cubin for the kernel's static shared memory")
        ctypes.c_int32.from_address(out).value = static_smem(
            self.cubin, self._entry(handle)
        )
        return 0

    def _cudaDeviceGetAttribute(self, out, attr, device) -> int:
        if device != 0 or attr not in self.device_attrs:
            raise RuntimeError(f"device {device} attribute {attr} is not modeled")
        ctypes.c_int32.from_address(out).value = self.device_attrs[attr]
        return 0

    def _cudaGetDevice(self, out) -> int:
        ctypes.c_int32.from_address(out).value = 0
        return 0

    def _cudaSetDevice(self, device) -> int:
        if device != 0:
            raise RuntimeError(f"device {device} (the CPU host has device 0 only)")
        return 0

    def _cudaKernelSetAttributeForDevice(self, handle, attr, value, device) -> int:
        return self._cudaFuncSetAttribute(handle, attr, value)

    def _cudaFuncSetAttribute(self, handle, attr, value) -> int:
        self.kernel_attrs.setdefault(self._entry(handle), {})[str(attr)] = value
        return 0

    def _cudaOccupancyMaxActiveBlocksPerMultiprocessor(self, out, handle, block, smem):
        raise RuntimeError("occupancy queries are not modeled")

    def cuda_dialect_init_library_once(self, state, init, load_to_device, set_error):
        # TVM-FFI's lazy module load: cuda_init(&library), then cuda_load_to_device for
        # the one device, as CudaDialectJitCompiledFunction._load_cuda_library does.
        flag = ctypes.c_int64.from_address(state)
        if flag.value:
            return 0
        library = ctypes.c_void_p()
        rc = ctypes.CFUNCTYPE(_I32, _P)(init)(ctypes.addressof(library))
        if rc == 0:
            rc = ctypes.CFUNCTYPE(_I32, _P, _I32)(load_to_device)(
                ctypes.addressof(library), 0
            )
        if rc:
            ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_char_p)(set_error)(
                b"RuntimeError", b"lkv: loading the kernels failed (see the dump)"
            )
            return rc
        flag.value = 1
        return 0

    def cuda_dialect_unload_library_once(self, state) -> None:
        return None

    def cuda_dialect_get_error_name(self, code):
        return ctypes.addressof(self._error_name)


def _declared(module) -> list:
    """Names of the functions a lowered (LLVM dialect) module declares but does not
    define: what its JIT engine must resolve."""
    out = []
    for op in module.body.operations:
        o = op.operation
        if o.name == "llvm.func" and not (o.regions and len(o.regions[0].blocks)):
            out.append(o.attributes["sym_name"].value)
    return out


_TENSORMAP = {"kind": "bytes", "size": 128, "align": 64}


def _ld_param_fields(ptx: str, name: str) -> list:
    """(offset, bytes) of every element the kernel reads from parameter `name` with
    ld.param (a vector load is one element per lane)."""
    out = set()
    pattern = (
        r"ld\.param(?:\.v(\d))?\.[a-z]+(\d+)\s[^;\[]*\[\s*"
        + re.escape(name)
        + r"\s*(?:\+\s*(\d+))?\s*\]"
    )
    for m in re.finditer(pattern, ptx):
        n, size, off = int(m[1] or 1), int(m[2]) // 8, int(m[3] or 0)
        out |= {(off + k * size, size) for k in range(n)}
    return sorted(out)


def _value(raw: bytes, off: int, size: int, tensors: dict) -> dict:
    """A parameter field: its value and, if it is an address (8 bytes), the named tensors
    it may point into."""
    value = int.from_bytes(raw[off : off + size], "little")
    hits = ptx_launch.resolve(value, tensors) if size == 8 and value else []
    return {"value": value, "kind": "ptr" if hits else "scalar"} | (
        {"tensor": hits} if hits else {}
    )


def complete(record: dict, ptx: str, tensors: dict) -> dict:
    """A raw LaunchRecord with every parameter interpreted (kind):
      tensormap  a CUtensorMap (`.align 64 .b8[128]`; words the DSL does not write read
                 as zero) decoded, with the named tensors its global address may be in;
                 tensormap_error instead if the decoder refuses a bit;
      aggregate  a struct, as the fields the kernel reads by name with ld.param, named
                 param<i>_<offset> (only CUtensorMaps have their address taken);
      ptr, scalar  the value.
    Every address is resolved against `tensors` (the launch's named leaves,
    frontend/ptx/launch.py resolve): candidates "tensor", all of them."""
    for p in record["params"]:
        raw = bytes.fromhex(p["bytes"])
        if all(p[k] == v for k, v in _TENSORMAP.items()):
            written = bytearray(raw)
            for w in UNWRITTEN_TMAP_WORDS:
                written[8 * w : 8 * w + 8] = bytes(8)
            try:
                d = ptx_launch.decode(bytes(written))
            except ptx_launch.LaunchError as e:
                p["tensormap_error"] = str(e)
                continue
            p["kind"] = "tensormap"
            p["tensormap"] = d | {
                "base": ptx_launch.resolve(d["address"], tensors, d),
                "unwritten_words": UNWRITTEN_TMAP_WORDS,
            }
        elif p["kind"] == "bytes":
            p["kind"] = "aggregate"
            p["fields"] = [
                {"off": off, "size": size, "name": f"param{p['i']}_{off}"}
                | _value(raw, off, size, tensors)
                for off, size in _ld_param_fields(ptx, p["name"])
            ]
        else:
            declared_ptr = p["kind"] == "ptr"
            p |= _value(raw, 0, p["size"], tensors)
            if declared_ptr:
                p["kind"] = "ptr"
    return record


def _tensor_table(params: list, torch_args) -> dict:
    """The launch's named torch tensors (_launch_leaves) in the form
    frontend/ptx/launch.py resolve takes."""
    out = {}
    for name, t in _launch_leaves(params, torch_args):
        if isinstance(t, torch.Tensor):
            s = t.untyped_storage()
            out[name] = {
                "data_ptr": t.data_ptr(),
                "storage": [s.data_ptr(), s.data_ptr() + s.nbytes()],
                "elem_bytes": t.element_size(),
                "shape": list(t.shape),
                "strides": list(t.stride()),
            }
    return out


def _raw(value):
    """The MLIR value of a DSL value: the DSL's value casters wrap the values of some types
    (a memref's in a cute Tensor, which emits a cute.get_iter where it is made)."""
    if hasattr(value, "_CAPIPtr"):
        return value
    (raw,) = extract_mlir_values(value)
    return raw


def _address(obj) -> int:
    """The address of an MLIR value or operation, by which they are compared: the DSL's
    value classes overload `==` (it builds IR)."""
    capsule = obj._CAPIPtr
    return _capsule_ptr(capsule, _capsule_name(capsule))


def _host_arguments(fop, spec, ir_args: list, ir_kwargs: dict) -> dict | str:
    """{"function": the jit function's name, "args": {block argument number: {"arg",
    "leaf": k[, "part"]}}, "leaves": {arg: number of leaves}} of the jit function `fop`:
    each block argument named by the parameter and the index of the leaf (_leaves) of the
    argument the DSL made from it (`ir_args`, `ir_kwargs`, in the order of `spec`, a
    FullArgSpec (DSL 4.4.2) or a Signature; "part" k if the leaf has several values).
    Indices, not names: the DSL may make a tuple argument the NamedTuple its annotation
    names, and leaves are named after the compile arguments (Compile.params). Reads the
    DSL's argument objects, never the block's arguments (the value casters would emit IR
    into the function). A string (why) if some block argument is no leaf's."""
    names = list(spec.parameters) if isinstance(spec, inspect.Signature) else spec.args
    block = fop.regions[0].blocks[0]
    out, counts = {}, {}
    for name, arg in [*zip(names, ir_args), *ir_kwargs.items()]:
        if name == "self":
            continue
        leaves = _leaves(name, arg)
        counts[name] = len(leaves)
        for i, (_, value) in enumerate(leaves):
            values = [ir.BlockArgument(v) for v in extract_mlir_values(value)]
            for k, v in enumerate(values):
                if v.owner != block:
                    return f"{name} is not an argument of the jit function"
                out[v.arg_number] = {"arg": name, "leaf": i} | (
                    {"part": k} if len(values) > 1 else {}
                )
    if len(out) != len(block.arguments):
        return f"{len(block.arguments) - len(out)} jit function arguments are no leaf's"
    return {"function": fop.sym_name.value, "args": out, "leaves": counts}


def _named(host: dict, params: list) -> dict | str:
    """_host_arguments' block arguments with their leaf names: the compile's (`params`,
    Compile.params), leaf by leaf; a string (why) if a parameter's leaves differ in
    number."""
    leaves = {p["name"]: p["leaves"] for p in params}
    for name, n in host["leaves"].items():
        if len(leaves.get(name, ())) != n:
            return f"{name}: {len(leaves.get(name, ()))} leaves compiled, {n} traced"
    return {
        k: a | {"leaf": leaves[a["arg"]][a["leaf"]]} for k, a in host["args"].items()
    }


def _inner(op) -> list:
    """The ops in an op's regions (one level)."""
    return [n.operation for r in op.regions for b in r.blocks for n in b.operations]


def _uses(op, seen: set) -> list | None:
    """The values an op uses, those of the ops in its regions included; each op once
    (`seen`, addresses). None if one of them loads: a value read from host memory may
    come from any value stored there (the DSL's host functions have no loads: layout
    algebra, arithmetic, scf, launches)."""
    out, todo = [], [getattr(op, "operation", op)]  # an OpView or an Operation
    while todo:
        o = todo.pop()
        if _address(o) in seen:
            continue
        seen.add(_address(o))
        if o.name.rsplit(".", 1)[-1].startswith("load"):
            return None
        out += [_raw(v) for v in o.operands]
        todo += _inner(o)
    return out


def _origin(value, entry, args: dict) -> dict:
    """Where a kernel operand of the jit function (entry block `entry`, its arguments
    `args`, _named) comes from: {"arg", "leaf"[, "part"]} if it is an argument
    (passed through), else {"from": [leaves]}, the leaves of the arguments reachable
    backwards through operands, where an op with regions (scf.if, scf.for) uses every
    value used inside it and a region argument counts as its op; every leaf if the value
    may come through memory (_uses). So no source is missed: a value computed in a region
    can only gain sources."""

    def argument(v):
        owner = v.owner
        if isinstance(owner, ir.Block) and owner == entry:
            return args[ir.BlockArgument(v).arg_number]
        return None

    if (a := argument(value)) is not None:
        return dict(a)
    leaves, seen, seen_ops, todo = set(), set(), set(), [value]
    while todo:
        v = todo.pop()
        if _address(v) in seen:
            continue
        seen.add(_address(v))
        if (a := argument(v)) is not None:
            leaves.add(a["leaf"])
            continue
        owner = v.owner
        used = _uses(owner.owner if isinstance(owner, ir.Block) else owner, seen_ops)
        if used is None:
            return {"from": sorted({a["leaf"] for a in args.values()})}
        todo += used
    return {"from": sorted(leaves)}


def _kernel_params(mlir: str, host: dict | str, params: list) -> dict | str:
    """{PTX entry: per kernel parameter, in order, where the launch takes it from
    (_origin)} for every kernel the jit function (`host`, _host_arguments, its leaves
    named after `params`) launches, read from its initial MLIR text after
    cute-remove-static-args, the first pass of the DSL's pipeline (stage.py), which drops
    the operands of a static type (a static layout, a copy atom without dynamic state)
    from kernels and launches: the operands left are the kernel's PTX parameters. Parsed
    in a context of its own, like _compile_copy, so the IR the value casters emit goes
    into the copy. A string (why) if the arguments are not all named or this fails."""
    if isinstance(host, str):
        return host
    args = _named(host, params)
    if isinstance(args, str):
        return args
    out: dict = {}
    try:
        with ir.Context(), ir.Location.unknown():
            module = ir.Module.parse(mlir)
            passmanager.PassManager.parse(
                "builtin.module(cute-remove-static-args)"
            ).run(module.operation)
            (fn,) = [o for o in _inner(module.operation) if o.name == "func.func"
                     and o.attributes["sym_name"].value == host["function"]]  # fmt: skip
            entry, launches, todo = fn.regions[0].blocks[0], [], [fn]
            while todo:
                op = todo.pop()
                todo += _inner(op)
                launches += [op] if op.name == "cuda.launch_ex" else []
            with ir.InsertionPoint.at_block_begin(entry):
                for op in launches:
                    callee = str(op.attributes["callee"])  # @kernels::@<PTX entry>
                    kernel = callee.rsplit("::@", 1)[-1].strip('"')
                    operands = [_raw(v) for v in op.operands][1:]  # after the config
                    out[kernel] = ("launched twice" if kernel in out else
                                   [_origin(v, entry, args) for v in operands])  # fmt: skip
    except Exception as e:  # noqa: BLE001 - recorded, see _name_params
        return f"naming the kernel parameters: {type(e).__name__}: {e}"
    return out


def _name_params(record: dict, traced: dict | str, pointers: dict) -> dict:
    """A completed LaunchRecord with "jit_params" (P11): per parameter the jit argument
    leaf it passes through or the leaves it is computed from (`traced`, PTX entry ->
    _kernel_params), if that is exact: the kernel was traced, its launch operands map
    onto its parameters, and every parameter passing a tensor through starts with that
    tensor's data pointer (`pointers`, _data_pointers; 0 for a zero-length tensor).
    Otherwise "jit_params" is None and "jit_params_error" says why."""
    got = (traced if isinstance(traced, str) else
           traced.get(record["entry"], f"kernel {record['entry']} was not traced"))  # fmt: skip
    params = record["params"]
    if isinstance(got, list) and len(got) != len(params):
        got = f"{len(got)} kernel parameters traced, {len(params)} in the PTX"
    for p, j in zip(params, got if isinstance(got, list) else []):
        want = pointers.get(j.get("leaf")) if "part" not in j else None
        pointer = int.from_bytes(bytes.fromhex(p["bytes"])[:8], "little")
        if want is not None and pointer != want:
            got = f"parameter {p['i']} does not start with the data pointer of {j['leaf']}"
            break
    if isinstance(got, list):
        record["jit_params"] = got
    else:
        record["jit_params"], record["jit_params_error"] = None, got
    return record


def _data_pointers(params: list, torch_args) -> dict:
    """{leaf: data pointer} of a launch's torch tensors (P11, _launch_leaves)."""
    return {name: t.data_ptr() for name, t in _launch_leaves(params, torch_args)
            if isinstance(t, torch.Tensor)}  # fmt: skip


def _launch_leaves(params: list, torch_args) -> list:
    """(leaf name, value) of a launch's arguments: each argument's leaves named after those
    of its compile parameter, leaf by leaf (`params`, Compile.params but those the launch
    passes no argument for: a TVM-FFI function takes no environment stream, wherever it
    sits, so the arguments after one go to the parameters after it); an argument whose
    leaves differ in number from its parameter's is left out."""
    out = []
    for p, a in zip(params, torch_args):
        leaves = _leaves(p["name"], a)
        if len(leaves) == len(p["leaves"]):
            out += [(name, t) for name, (_, t) in zip(p["leaves"], leaves)]
    return out


def _keep(option_cls, stem: str, arch: str, ext: str):
    """A DSL file-dump compile option (KeepPTX, KeepCUBIN) writing <stem>.<arch>.<ext>."""
    opt = option_cls(True)
    opt.dump_path = stem
    setattr(opt, f"full_{ext}_path", f"{stem}.{arch}.{ext}")
    return opt


def _dumped(stem: str, arch: str, ext: str) -> bytes | None:
    """The file a DSL file-dump option wrote, <stem>.<arch>.<ext>, if any."""
    path = Path(f"{stem}.{arch}.{ext}")
    return path.read_bytes() if path.exists() else None


def _ptx(stem: str, arch: str) -> str | None:
    data = _dumped(stem, arch, "ptx")
    return None if data is None else data.replace(b"\0", b"").decode()  # NUL-ended


def _compile_copy(dsl, mlir: str, pipeline, arch: str, stem: str) -> tuple:
    """(PTX, error) of the initial MLIR text compiled by the DSL's pipeline with its
    current options, in a context of its own (the CUDA dialect refuses a second binary of
    a module in one context); no JIT engine, no TVM-FFI wrapper (host code)."""
    from cutlass._mlir import ir as _ir

    ir: Any = _ir  # native bindings without stubs (stage.py)
    hook = dsl.compiler_provider._post_compile_hook
    dsl.compiler_provider._post_compile_hook = None
    try:
        with ir.Context(), ir.Location.unknown():
            dsl.compiler_provider.compile(
                ir.Module.parse(mlir),
                dsl.preprocess_pipeline(dsl._get_pipeline(pipeline), arch),
            )
        return _ptx(stem, arch), None
    except Exception as e:  # noqa: BLE001 - recorded with the compile
        return None, f"{type(e).__name__}: {e}"
    finally:
        dsl.compiler_provider._post_compile_hook = hook


def _patch_devices(patch, num_sms: int) -> None:
    """P1-P4, P8, P10."""
    import cutlass.utils
    from cutlass.cute import runtime as cute_runtime

    props = types.SimpleNamespace(
        multi_processor_count=num_sms,
        major=10,
        minor=0,
        name="lkv-cpu-sm100",
        shared_memory_per_block_optin=232448,  # as _device_attrs (attribute 97)
    )
    patch(torch.cuda, "is_available", lambda: True)
    patch(torch.cuda, "device_count", lambda: 1)
    patch(torch.cuda, "get_device_properties", lambda *a, **k: props)
    patch(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    patch(torch.cuda, "current_device", lambda: 0)

    hw = cutlass.utils.HardwareInfo
    patch(hw, "__init__", lambda self, device_id=0: None)
    patch(hw, "get_device_multiprocessor_count", lambda self: num_sms)
    patch(
        hw,
        "get_max_active_clusters",
        lambda self, cluster_size=1: num_sms // cluster_size,
    )

    patch(torch.Tensor, "is_cuda", property(lambda self: True))
    # P8: page-locked memory changes how copies to the device run, not the data.
    patch(torch.Tensor, "pin_memory", lambda self, *a, **k: self.clone())
    patch(torch.Tensor, "is_pinned", lambda self, *a, **k: True)

    tensor_init = cute_runtime._Tensor.__init__

    def cuda_typed_init(self, *a, **k):
        tensor_init(self, *a, **k)
        _mark_capsule_cuda(self._dlpack_data)
        wrapped = a[0] if a else k.get("tensor")
        self._lkv_source = wrapped if isinstance(wrapped, torch.Tensor) else None
        if self._lkv_source is not None and hasattr(self, "_tvm_ffi_tensor"):
            self._tvm_ffi_tensor = _cuda_ffi_tensor(self._lkv_source)  # P10

    patch(cute_runtime._Tensor, "__init__", cuda_typed_init)


def _patch_host_runtime(patch, capturing: list) -> None:
    """P9: no runtime library, no cuInit, one device; a JIT engine gets the callbacks of
    the compile being captured (capturing[-1]), and a TVM-FFI function is made from it."""
    from cutlass.base_dsl import compiler
    from cutlass.base_dsl.dsl import BaseDSL
    from cutlass.cutlass_dsl import cuda_jit_executor, tvm_ffi_provider

    patch(BaseDSL, "get_shared_libs", lambda self: [])
    if hasattr(compiler.Compiler, "_check_cuda_dependencies_once"):  # DSL 4.4.2
        patch(compiler.Compiler, "_check_cuda_dependencies_once", lambda s, libs: None)
    patch(cuda_jit_executor.CudaDialectJitCompiledFunction, "num_devices", 1)
    real_jit = compiler.Compiler.jit

    def jit_with_host_runtime(self, module, *a, **k):
        if not capturing:
            raise RuntimeError("a JIT engine built outside a captured compile")
        # The callbacks can only be registered once the engine exists, but building the
        # engine registers a module's global destructors (TVM-FFI: the library unload),
        # which on x86-64 links the whole module then, before the callbacks are there
        # ("Symbols not found", and a crash when registering). Teardown is not part of
        # the capture, so the engine is built without them.
        for op in list(module.body.operations):
            if op.operation.name == "llvm.mlir.global_dtors":
                op.operation.erase()
        engine = real_jit(self, module, *a, **k)
        capturing[-1].register(engine, _declared(module))
        return engine

    patch(compiler.Compiler, "jit", jit_with_host_runtime)

    def ffi_function_on_engine(self):
        # As DSL 4.4.2 makes it; DSL 4.6.1 would re-link the module with JITLink against
        # the DSL's runtime library, without the callbacks.
        if self.engine is None:
            return None
        import tvm_ffi

        # Compiled eagerly, as the DSL does.
        for name in ("cuda_init", "cuda_load_to_device"):
            if self.engine.raw_lookup(name) is None:
                raise RuntimeError(f"{name} not found in the JIT engine")
        return tvm_ffi.Function.__from_mlir_packed_safe_call__(
            self.engine.raw_lookup("__tvm_ffi_" + self.function_name),
            keep_alive_object=self.engine,
        )

    patch(
        tvm_ffi_provider.TVMFFIJitCompiledFunctionBase,
        "_create_tvm_ffi_function",
        ffi_function_on_engine,
    )


def _patch_trace(patch, trace: dict) -> None:
    """P11: while a jit function is traced, trace["host"] = its arguments
    (_host_arguments). Observed only: a failure is recorded as the reason."""
    from cutlass.base_dsl.dsl import BaseDSL

    real_arguments = BaseDSL.generate_execution_arguments

    def arguments(self, args, kwargs, fop, spec):
        ir_args, ir_kwargs = real_arguments(self, args, kwargs, fop, spec)
        if "host" not in trace and fop.operation.name == "func.func":
            try:
                trace["host"] = _host_arguments(fop, spec, ir_args, ir_kwargs)
            except Exception as e:  # noqa: BLE001 - recorded, see _name_params
                trace["host"] = (
                    f"naming the jit function's arguments: {type(e).__name__}: {e}"
                )
        return ir_args, ir_kwargs

    patch(BaseDSL, "generate_execution_arguments", arguments)


def _capture_compile(dsl, real, args: tuple, kwargs: dict, stem: str, num_sms: int,
                     lineinfo_check: bool, capturing: list) -> _Captured:  # fmt: skip
    """P5 for one BaseDSL.compile_and_cache call (`real`, with `args` after self): the
    initial MLIR and pipeline, then the DSL's own compile of the module with line info
    and PTX and cubin kept at <stem>.*, no cache, and its JIT engine (P9)."""
    from cutlass.base_dsl import compiler

    module, _hash, _name, pipeline, *rest = args
    # The arch and pipeline exactly as BaseDSL.compile_and_cache computes them.
    arch = dsl.compile_options.gpu_arch or dsl.envar.arch
    captured = _Captured(
        # (+ newline: the text save_ir writes, so 4.4.2 digests are unchanged)
        module.operation.get_asm(
            enable_debug_info=True, print_generic_op_form=GENERIC_FORM
        )
        + "\n",
        dsl.preprocess_pipeline(dsl._get_pipeline(pipeline), arch),
    )
    cap = captured.capture
    cap.tvm_ffi = bool(dsl.compile_options.enable_tvm_ffi)
    captured.runtime = runtime = _HostRuntime(num_sms)
    _RUNTIMES.append(runtime)
    # The options are swapped, not changed in place: cute.compile hands every compile the
    # same CompileOptions object, and line info would reach the next initial MLIR.
    options = dsl.compile_options.options
    saved = dict(options)
    if lineinfo_check and not GENERIC_FORM:
        options[compiler.GenerateLineInfo] = compiler.GenerateLineInfo(False)
        options[compiler.KeepPTX] = _keep(compiler.KeepPTX, stem + "_nl", arch, "ptx")
        cap.ptx_noline, cap.lineinfo_error = _compile_copy(
            dsl, captured.mlir, pipeline, arch, stem + "_nl"
        )
    # Once the pipeline has run, the kernels are a binary in the module; line info in the
    # host code only breaks the TVM-FFI wrapper and the engine (DSL 4.4.2: a call without
    # a debug location in a function with one), so it is stripped first.
    provider = dsl.compiler_provider
    hook = provider._post_compile_hook

    def strip_host_line_info(m):
        provider.passmanager.PassManager.parse(
            "builtin.module(strip-debuginfo)", context=m.context
        ).run(m.operation)
        if hook is not None:
            hook(m)

    try:
        options[compiler.GenerateLineInfo] = compiler.GenerateLineInfo(True)
        options[compiler.KeepPTX] = _keep(compiler.KeepPTX, stem, arch, "ptx")
        options[compiler.KeepCUBIN] = _keep(compiler.KeepCUBIN, stem, arch, "cubin")
        provider._post_compile_hook = strip_host_line_info
        capturing.append(runtime)
        # rest = (args_spec, no_cache, no_jit_engine, [func_type]): no cache, an engine.
        captured.compiled = real(
            dsl, *args[:4], rest[0], True, False, *rest[3:], **kwargs
        )
    except Exception as e:  # noqa: BLE001 - recorded; the MLIR capture stands
        cap.error = f"{type(e).__name__}: {e}"
    finally:
        provider._post_compile_hook = hook
        capturing.clear()
        options.clear()
        options.update(saved)
        cap.ptx = runtime.ptx = _ptx(stem, arch)
        cap.cubin = runtime.cubin = _dumped(stem, arch, "cubin")
    return captured


def _run_host_code(captured: _Captured, call_args: tuple, call_kwargs: dict,
                   tensors: dict, pointers: dict) -> tuple:  # fmt: skip
    """P6: call a captured compile's function as the host did; (its LaunchRecords,
    completed against `tensors` and named (P11, `pointers`); why it failed or None)."""
    compiled, runtime, cap = captured.compiled, captured.runtime, captured.capture
    if compiled is None or runtime is None:
        return [], f"not compiled: {cap.error}"
    if cap.tvm_ffi:
        call_args = _ffi_args(call_args)
        call_kwargs = {k: _ffi_args(v) for k, v in call_kwargs.items()}
    runtime.launches, runtime.errors = [], []
    try:
        compiled(*call_args, **call_kwargs)
    except Exception as e:  # noqa: BLE001 - recorded on the launch
        runtime.errors.append(f"{type(e).__name__}: {e}")
    # The module unloads its libraries (P9 handles) when it is collected, possibly after
    # the patches are gone: it must have none.
    if getattr(compiled, "jit_module", None) is not None:
        compiled.jit_module.cuda_library.clear()
    records = [
        _name_params(
            complete(r, cap.ptx or "", tensors), captured.kernel_params, pointers
        )
        for r in runtime.launches
    ]
    return records, "; ".join(runtime.errors) or None


@contextmanager
def installed(num_sms: int, lineinfo_check: bool = False):
    """Install P1-P10; yields the Recording the patched compile and launch calls fill.
    `lineinfo_check`: also compile every module without line info (P5, DSL 4.4.2)."""
    import cutlass.cute as cute
    from cutlass.base_dsl.dsl import BaseDSL
    from cutlass.cute import runtime as cute_runtime

    rec = Recording()
    saved = []

    def patch(obj, name, value):
        saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    _patch_devices(patch, num_sms)
    capturing: list = []  # the _HostRuntime of the compile being captured
    _patch_host_runtime(patch, capturing)
    trace: dict = {}  # of the compile being traced (P11)
    _patch_trace(patch, trace)
    scratch = tempfile.mkdtemp(prefix="lkv_cute_ptx_")
    real_compile_and_cache = BaseDSL.compile_and_cache

    def compile_and_capture(self, *args, **kwargs):
        stem = os.path.join(scratch, f"{len(rec.compiles):02d}")
        raise _capture_compile(self, real_compile_and_cache, args, kwargs, stem,
                               num_sms, lineinfo_check, capturing)  # fmt: skip

    patch(BaseDSL, "compile_and_cache", compile_and_capture)

    real_compile = cute.compile

    def compile_and_record(fn, *args, **kwargs):
        trace.clear()
        try:
            real_compile(fn, *args, **kwargs)
        except Exception as e:  # noqa: BLE001 - the DSL wraps _Captured in its own errors
            cause = e
            while cause is not None and not isinstance(cause, _Captured):
                cause = cause.__cause__ or cause.__context__
            if cause is None:
                raise
        else:
            raise RuntimeError(
                "cute.compile returned without reaching compile_and_cache"
            )
        positional = _positional(fn, args, kwargs, compile_call=True)
        c = Compile(
            len(rec.compiles),
            _qualname(fn),
            _params(fn, positional),
            _flags(fn),
            cause.mlir,
            cause.pipeline,
            cause.capture,
        )
        rec.compiles.append(c)
        captured = cause
        captured.kernel_params = _kernel_params(
            c.mlir, trace.get("host", "not traced"), c.params
        )
        launched = [p for p, a in zip(c.params, positional)
                    if not getattr(a, "use_tvm_ffi_env_stream", False)]  # fmt: skip

        def launch(*call_args, **call_kwargs):
            positional = tuple(_positional(fn, call_args, call_kwargs))
            with rec.reads.paused():
                torch_args = _torch_args(positional, cute_runtime._Tensor)
                args = [describe(a) for a in torch_args]
                tensors = [a for a in _flat(torch_args) if isinstance(a, torch.Tensor)]
                described = [d for d in _flat(args) if d["kind"] == "tensor"]
                hinted = rec.reads.pointees([v for d in _flat(args) for v in _ints(d)])
                for d, src in zip(described, rec.reads.launch(tensors, hinted)):
                    d["sources"] = src
                la = Launch(c.index, args, hinted)
                la.records, la.capture_error = _run_host_code(
                    captured, call_args, call_kwargs, _tensor_table(launched, torch_args),
                    _data_pointers(launched, torch_args),
                )  # fmt: skip
            rec.launches.append(la)

        return launch

    patch(cute, "compile", compile_and_record)
    try:
        yield rec
    finally:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)
        shutil.rmtree(scratch, ignore_errors=True)
