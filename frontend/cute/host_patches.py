# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""GPU-only calls replaced so a kernel's real host code path runs on a CPU-only host.

Every replacement is part of the trust base (references/frontend_cute.md, "Host
patches"); nothing else on the host path is changed:

  P1 torch.cuda.is_available / device_count / get_device_properties /
     get_device_capability / current_device: one SM100 device (compute capability 10.0)
     with `num_sms` SMs.
  P2 cutlass.utils.HardwareInfo: `num_sms` SMs, `num_sms // cluster_size` clusters.
  P3 torch.Tensor.is_cuda is True: host validation demands CUDA tensors; the tensors are
     CPU tensors with the same shapes, strides, dtypes and integer contents.
  P4 DLPack capsules made by cute.runtime.from_dlpack report device kDLCUDA, so the DSL
     types those tensors exactly as CUDA tensors (global memory) at compile time. Each
     runtime tensor remembers the torch tensor it wraps (its memory).
  P5 BaseDSL.compile_and_cache records the pass pipeline the DSL computes for the
     compile (its own _get_pipeline + preprocess_pipeline with the target arch) and
     stops before running it. The DSL has already written (CUTE_DSL_KEEP_IR) and
     verified the initial MLIR; nothing is lowered, linked or loaded.
  P6 cute.compile returns a recorder: the host's launch calls record their arguments
     instead of running the kernel (a runtime tensor as the torch tensor it wraps), with
     the sources of each tensor argument's data and the storages its integer arguments
     point into (Recording.reads, the run's ttgir.host_reads.HostReads, which
     dump.py activates around each host call).
  P8 torch.Tensor.pin_memory returns a copy of the tensor (is_pinned is True): page-
     locked host memory needs a driver, and pinning changes how copies to the device
     run, not the data.
"""

import ctypes
import dataclasses
import inspect
import types
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from cutlass import Numeric

KDLCUDA = 2
METADATA_LIMIT = 1 << 16  # integer tensors up to this many elements keep their contents


class _Captured(Exception):
    """Raised by the patched compile_and_cache once the initial MLIR is on disk."""

    def __init__(self, path: str, pipeline: str):
        super().__init__(path)
        self.path = path
        self.pipeline = pipeline


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


@dataclass
class Launch:
    compile: int
    args: list
    hinted: list  # storages its integer arguments point into (address hints, ttgir.host_reads)


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


def _leaf_names(name: str, value) -> list:
    """Names of the non-None leaves of a compile argument; tuples and namedtuples
    flatten in order, as the DSL flattens them into host-function arguments."""
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        fields = getattr(value, "_fields", None) or [str(i) for i in range(len(value))]
        return [n for f, v in zip(fields, value) for n in _leaf_names(f"{name}_{f}", v)]
    return [name]


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


@contextmanager
def installed(num_sms: int):
    """Install P1-P6; yields the Recording the patched compile and launch calls fill."""
    import cutlass.cute as cute
    import cutlass.utils
    from cutlass.base_dsl.dsl import BaseDSL
    from cutlass.cute import runtime as cute_runtime

    rec = Recording()
    saved = []

    def patch(obj, name, value):
        saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    props = types.SimpleNamespace(
        multi_processor_count=num_sms, major=10, minor=0, name="lkv-cpu-sm100"
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

    patch(cute_runtime._Tensor, "__init__", cuda_typed_init)

    def stop_before_passes(self, module, module_hash, function_name, pipeline, *a, **k):
        # The arch and pipeline exactly as BaseDSL.compile_and_cache computes them.
        arch = self.compile_options.gpu_arch or self.envar.arch
        raise _Captured(
            str(self.dump_mlir_path),
            self.preprocess_pipeline(self._get_pipeline(pipeline), arch),
        )

    patch(BaseDSL, "compile_and_cache", stop_before_passes)

    real_compile = cute.compile

    def compile_and_record(fn, *args, **kwargs):
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
        with open(cause.path) as f:
            text = f.read()
        c = Compile(
            len(rec.compiles),
            _qualname(fn),
            _params(fn, _positional(fn, args, kwargs, compile_call=True)),
            _flags(fn),
            text,
            cause.pipeline,
        )
        rec.compiles.append(c)

        def launch(*call_args, **call_kwargs):
            call_args = tuple(_positional(fn, call_args, call_kwargs))
            with rec.reads.paused():
                call_args = _torch_args(call_args, cute_runtime._Tensor)
                args = [describe(a) for a in call_args]
                tensors = [a for a in _flat(call_args) if isinstance(a, torch.Tensor)]
                described = [d for d in _flat(args) if d["kind"] == "tensor"]
                hinted = rec.reads.pointees([v for d in _flat(args) for v in _ints(d)])
                for d, src in zip(described, rec.reads.launch(tensors, hinted)):
                    d["sources"] = src
            rec.launches.append(Launch(c.index, args, hinted))

        return launch

    patch(cute, "compile", compile_and_record)
    try:
        yield rec
    finally:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)
