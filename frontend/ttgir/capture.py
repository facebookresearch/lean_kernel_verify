# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Intercept Triton kernel launches made by real host code, without launching.

Two hooks (both documented in references/frontend_ttgir.md):

* `JITFunction._do_compile` is replaced by a recorder. `JITFunction.run` reaches it after
  the real binder and `_pack_args` computed the specialization (signature, constexprs,
  divisibility attrs, backend options), exactly as for a GPU launch. The recorder also
  evaluates the launch grid on the real bound arguments and returns None, which makes
  `run` return before any launch.
* `Autotuner.run` is replaced by a loop over every config the autotuner would benchmark
  for these arguments (`prune_configs`, i.e. after `early_config_prune`/`perf_model`).
  For each config it does what `Autotuner.run` does for the selected config on a cold
  tuning cache: zero the `reset_to_zero` arguments if it benchmarked, set `best_config`,
  run the config `pre_hook` (e.g. TMA descriptor block shapes), then
  `fn.run(..., **config.all_kwargs())`. Heuristics wrappers run unchanged.

`warmup` calls (`kernel.warmup(...)`, preloads) compile without launching on the GPU and
are not recorded.

Arguments are snapshotted at capture time because later pre_hooks mutate the same
TensorDescriptor objects. The specialization (tt.divisibility of pointers and ints, ints
equal to 1 as constexprs) is the binder's, computed from the same values a GPU launch
sees; pointer alignment agrees with the GPU because every storage base is checked to be
16-byte aligned, as CUDA caching-allocator blocks (512-byte aligned) are.

After the compile, `launch_record` gives the launch as the GPU would see it (a
LaunchRecord): grid, block, cluster and dynamic shared memory as CudaLauncher and the C
launcher derive them from the kernel metadata, and the flat device parameter list as Triton
itself builds it: `expand_signature` for the types and `wrap_handle_tensordesc` (Triton's
make_tensordesc_args and make_tensordesc_arg) for the values, constexprs dropped and the
global and profile scratch pointers appended. Nothing is encoded: a host TensorDescriptor's
CUtensorMap parameter is the arguments the launcher passes to `fill_tma_descriptor_tiled`
(gpu_free.recorded_tma_fills), i.e. its base address, shape and strides plus the box,
swizzle and element type of `tensordesc_meta`. The list is checked against the `.entry`
of the PTX (count, kind, size and alignment of every parameter) and `.reqntid` /
`.reqnctapercluster`; any difference fails the dump.

During the capture the dump's HostReads mode (host_reads.py, one for the whole run) tracks
host data: every launch records, per tensor argument, the earlier launches' storages its
data may come from (`Launch.sources`) and the storages its integer arguments point into
(`Launch.hinted`), and host reads of such data are logged (`HostReads.reads`). The grid
callable is host code and runs tracked; the binder and the argument snapshots run paused.
"""

import contextvars
import copy
import math
import re
from dataclasses import dataclass, field
from itertools import zip_longest

import torch
from triton.backends.nvidia.driver import expand_signature, wrap_handle_tensordesc
from triton.compiler import ASTSource, compile as triton_compile
from triton.runtime.autotuner import Autotuner
from triton.runtime.jit import JITFunction
from triton.tools.tensor_descriptor import TensorDescriptor

from ..ptx import launch as ptx_launch
from . import gpu_free
from .host_reads import HostReads

# Integer tensors up to this many elements are recorded with their contents (metadata).
MAX_RECORDED_INTS = 1 << 16

_CALL = contextvars.ContextVar("lkv_jit_call")
_CONFIG = contextvars.ContextVar("lkv_autotune_config", default=None)


@dataclass
class Launch:
    kernel: str
    jit: JITFunction
    signature: dict
    constexprs: dict  # param name -> value
    attrs: dict  # param name -> [[attr, value], ...]
    specialization: dict  # param name -> [type, key] as the binder computed it
    options: dict
    grid: tuple
    args: dict  # param name -> snapshot record
    config: dict | None
    compile_args: (
        tuple  # (signature, constexprs, attrs, options) exactly as JITFunction has them
    )
    sources: dict  # tensor/descriptor param -> [[storage, t], ...] its data may come from (host_reads.py)
    hinted: dict  # integer param -> storages its values point into (address hints, host_reads.py)
    values: tuple  # bound argument values as the launcher reads them (_detached)

    def compile(self, mode, target: str):
        """The compile JITFunction would do for this launch on the target GPU."""
        signature, constexprs, attrs, options = self.compile_args
        with mode:
            return triton_compile(
                ASTSource(self.jit, signature, constexprs, attrs),
                target=gpu_free.TARGETS[target],
                options=options,
            )


@dataclass
class Capture:
    reads: HostReads  # the dump's source tracking (shared by every input of the run)
    launches: list = field(default_factory=list)

    def __enter__(self):
        self._saved = (JITFunction.run, JITFunction._do_compile, Autotuner.run)
        orig_run = JITFunction.run
        capture = self

        def run(jit, *args, grid, warmup, **kwargs):
            if getattr(jit, "c_cache", False):
                raise RuntimeError(
                    f"{jit.fn.__name__}: c_cache launches bypass JITFunction.run"
                )
            token = _CALL.set((args, kwargs, grid))
            try:
                return orig_run(jit, *args, grid=grid, warmup=warmup, **kwargs)
            finally:
                _CALL.reset(token)

        def do_compile(
            jit, key, signature, device, device_key, constexprs, options, attrs, warmup
        ):
            if not warmup:
                capture.launches.append(
                    _record(
                        jit,
                        device_key,
                        signature,
                        constexprs,
                        options,
                        attrs,
                        capture.reads,
                    )
                )
            return None

        JITFunction.run = run
        JITFunction._do_compile = do_compile
        Autotuner.run = _autotuner_run
        self.reads.__enter__()
        return self

    def __exit__(self, *exc):
        self.reads.__exit__(*exc)
        JITFunction.run, JITFunction._do_compile, Autotuner.run = self._saved
        return False


def _autotuner_run(tuner, *args, **kwargs):
    if getattr(tuner, "include_npot", False):
        raise RuntimeError(
            "autotune(include_npot=True) adds configs at tune time; not supported"
        )
    tuner.nargs = dict(zip(tuner.arg_names, args))
    configs = tuner.prune_configs(kwargs) if len(tuner.configs) > 1 else tuner.configs
    benchmarked = len(tuner.configs) > 1 and (
        len(configs) > 1 or tuner.ir_config_prune is not None
    )
    for config in configs:
        full_nargs = {**tuner.nargs, **kwargs, **config.all_kwargs()}
        if benchmarked:
            tuner.pre_hook(full_nargs, reset_only=True)
        tuner.best_config = config
        if config.pre_hook is not None:
            config.pre_hook(full_nargs)
        token = _CONFIG.set({k: _plain(v) for k, v in config.all_kwargs().items()})
        try:
            tuner.fn.run(*args, **kwargs, **config.all_kwargs())
        finally:
            _CONFIG.reset(token)
    tuner.nargs = None


def _record(
    jit, device_key, signature, constexprs, options, attrs, reads: HostReads
) -> Launch:
    args, kwargs, grid = _CALL.get()
    binder = jit.device_caches[device_key][4]
    with reads.paused():
        bound, specialization, _ = binder(*args, **kwargs)
    if callable(grid):  # host code: its reads are tracked like any other
        grid = grid(bound)
    grid = tuple(int(g) for g in grid) + (1,) * (3 - len(grid))
    with reads.paused():
        return _launch(
            jit,
            signature,
            constexprs,
            options,
            attrs,
            reads,
            bound,
            specialization,
            grid,
        )


def _launch(
    jit,
    signature,
    constexprs,
    options,
    attrs,
    reads: HostReads,
    bound: dict,
    specialization,
    grid: tuple,
) -> Launch:
    names = jit.arg_names

    def by_name(paths: dict) -> dict:
        out = {}
        for path, value in paths.items():
            if len(path) != 1:
                raise RuntimeError(
                    f"{jit.fn.__name__}: tuple kernel arguments are not supported"
                )
            out[names[path[0]]] = value
        return out

    data = {
        k: v.base if isinstance(v, TensorDescriptor) else v
        for k, v in bound.items()
        if isinstance(v, (torch.Tensor, TensorDescriptor))
    }
    snapshots = {k: _snapshot(v) for k, v in bound.items()}
    hinted = {
        k: keys for k, rec in snapshots.items() if (keys := reads.pointees(_ints(rec)))
    }
    return Launch(
        kernel=jit.fn.__name__,
        jit=jit,
        signature=dict(signature),
        constexprs={k: _plain(v) for k, v in by_name(constexprs).items()},
        attrs=by_name(attrs),
        specialization={
            name: [_plain(x) for x in spec] for name, spec in zip(names, specialization)
        },
        options={k: _plain(v) for k, v in options.__dict__.items()},
        grid=grid,
        args=snapshots,
        config=_CONFIG.get(),
        compile_args=(signature, constexprs, attrs, options.__dict__),
        sources=dict(
            zip(
                data,
                reads.launch(
                    list(data.values()), [k for ks in hinted.values() for k in ks]
                ),
            )
        ),
        hinted=hinted,
        values=tuple(_detached(v) for v in bound.values()),
    )


class _Ptr(int):
    """A tensor argument reduced to its address: what the launcher reads of it."""

    def data_ptr(self) -> int:
        return int(self)


def _detached(v):
    """A bound argument as the launcher reads it at the (skipped) launch, holding no tensor:
    keeping the tensors alive until the dump's compile would change which storage addresses
    later inputs get. A TensorDescriptor is copied because later pre_hooks mutate it."""
    if isinstance(v, torch.Tensor):
        return _Ptr(v.data_ptr())
    if isinstance(v, TensorDescriptor):
        v = copy.copy(v)
        v.base = _Ptr(v.base.data_ptr())
    return v


# Device parameter of a leaf type of Triton's expanded signature -> (kind, bytes, alignment),
# as the C launcher packs it (driver.c; scalars in their own type, i1 as a byte).
_SCALAR_BYTES = {
    "i1": 1, "u1": 1, "i8": 1, "u8": 1, "i16": 2, "u16": 2, "fp16": 2, "bf16": 2,
    "i32": 4, "u32": 4, "fp32": 4, "i64": 8, "u64": 8, "fp64": 8,
}  # fmt: skip
# ptx_params kind of each LaunchRecord kind.
_PTX_KIND = {"ptr": "ptr", "scalar": "scalar", "tensormap": "bytes"}


def _leaf(ty: str) -> tuple:
    if ty == "nvTmaDesc":
        return "tensormap", 128, 64  # CUtensorMap: 128 bytes, 64-byte aligned
    if ty.startswith("*"):
        return "ptr", 8, 8
    if ty not in _SCALAR_BYTES:
        raise RuntimeError(f"kernel parameter of type {ty}: not supported")
    return "scalar", _SCALAR_BYTES[ty], _SCALAR_BYTES[ty]


def launch_record(launch: Launch, kernel) -> dict:
    """The launch as CudaLauncher makes it, for the compiled `kernel`:
    {entry, grid, block, cluster, smem_dynamic, attrs, params}. Every parameter must agree
    with the PTX .entry (launch.ptx_params), block and cluster with its .reqntid and
    .reqnctapercluster."""
    md = kernel.metadata
    ptx = kernel.asm["ptx"]
    # CudaLauncher: ctas_per_cga (TLX) regroups the grid into clusters through the PTX's
    # .reqnctapercluster; num_ctas (Triton) multiplies grid x and sets the cluster attribute.
    ctas_per_cga = getattr(md, "ctas_per_cga", None)
    num_ctas = 1 if ctas_per_cga is not None else md.num_ctas
    reqn = _directive(ptx, "reqnctapercluster")
    cluster = [num_ctas, 1, 1] if num_ctas > 1 else reqn or [1, 1, 1]
    block = [32 * md.num_warps, 1, 1]
    for what, ours, ptx_value in (
        ("block", block, _directive(ptx, "reqntid")),
        ("cluster", cluster, reqn or cluster),
        ("ctas_per_cga", cluster, _dims3(ctas_per_cga) if ctas_per_cga else cluster),
    ):
        if ours != ptx_value:
            raise RuntimeError(
                f"{launch.kernel}: {what} {ours} disagrees with the PTX / metadata ({ptx_value})"
            )
    params = _device_params(launch, md, math.prod(launch.grid) * num_ctas)
    _check_params(launch.kernel, md.name, params, ptx_launch.ptx_params(ptx, md.name))
    return {
        "entry": md.name,
        "grid": [launch.grid[0] * num_ctas, *launch.grid[1:]],
        "block": block,
        "cluster": cluster,
        "smem_dynamic": md.shared,
        "attrs": {
            "pdl": bool(md.launch_pdl),
            "cooperative": bool(md.launch_cooperative_grid),
            "preferred_cluster": _plain(getattr(md, "preferred_ctas_per_cga", None)),
        },
        "params": params,
    }


def _directive(ptx: str, name: str) -> list | None:
    """The values of a performance directive of the (single) .entry, padded to 3 dims."""
    m = re.search(rf"^\s*\.{name}\s+(\d+(?:\s*,\s*\d+)*)", ptx, re.M)
    if m is None:
        return None
    return _dims3(int(x) for x in m.group(1).split(","))


def _dims3(dims) -> list:
    dims = list(dims)
    return dims + [1] * (3 - len(dims))


def _device_params(launch: Launch, md, programs: int) -> list:
    """The flat device parameter list: [{name, type, kind, size, align, value}]."""
    if getattr(md, "auto_tma_recipes", None):
        raise RuntimeError(
            f"{launch.kernel}: compiler-synthesized TMA descriptors (auto_tma_recipes) "
            "are not supported"
        )
    signature = launch.compile_args[0]
    meta = getattr(md, "tensordesc_meta", None)
    wrapped: list = []
    with gpu_free.recorded_tma_fills():
        wrap_handle_tensordesc(lambda *a: wrapped.extend(a[-1]), signature, meta)(
            launch.values
        )
    values = iter(wrapped)
    params = []
    k = 0  # tensor descriptors so far
    for name, ty in signature.items():
        if not isinstance(ty, str):
            raise RuntimeError(
                f"{launch.kernel}: tuple kernel arguments are not supported"
            )
        if ty.startswith("tensordesc"):
            desc_meta = meta[k] if meta else None
            types = expand_signature([ty], [desc_meta] if meta else None)
            k += 1
            rank = (len(types) - 1) // 2
            names = (
                [name]
                + [f"{name}.shape.{i}" for i in range(rank)]
                + [f"{name}.stride.{i}" for i in range(rank)]
                if desc_meta is not None
                else [f"{name}.{j}" for j in range(len(types))]  # decomposed (no TMA)
            )
        else:
            types, names = [ty], [name]
        for leaf, leaf_ty in zip(names, types):
            value = next(values)
            if leaf_ty == "constexpr":
                continue
            kind, size, align = _leaf(leaf_ty)
            if kind == "ptr":  # a tensor (_Ptr) or a decomposed descriptor's base
                value = 0 if value is None else int(value.data_ptr())
            elif kind == "scalar":
                value = _plain(value)
            params.append(
                {
                    "name": leaf,
                    "type": leaf_ty,
                    "kind": kind,
                    "size": size,
                    "align": align,
                    "value": value,  # a tensormap: the _tiled_fill record
                }
            )
    if next(values, values) is not values:
        raise RuntimeError(
            f"{launch.kernel}: more launcher arguments than signature leaves"
        )
    # CudaLauncher.allocate_scratch: a fresh buffer of size x programs (no host address),
    # NULL when the size is 0.
    for name, size, align in (
        ("__global_scratch", md.global_scratch_size, md.global_scratch_align),
        ("__profile_scratch", md.profile_scratch_size, md.profile_scratch_align),
    ):
        p: dict = {"name": name, "type": "*i8", "kind": "ptr", "size": 8, "align": 8}
        if size:
            p.update(value=None, scratch={"bytes": programs * size, "align": align})
        else:
            p.update(value=0)
        params.append(p)
    return params


def _check_params(kernel: str, entry: str, params: list, entry_params: list) -> None:
    """Every parameter has the PTX parameter's kind, size and alignment (the launcher packs
    exactly this list); the PTX declaration is added to each."""
    ours = [(_PTX_KIND[p["kind"]], p["size"], p["align"]) for p in params]
    theirs = [(d["kind"], d["size"], d["align"]) for d in entry_params]
    if ours != theirs:
        rows = "\n".join(
            f"  {i}: {p['name'] if p else '-'} {o or '-'} | {d['decl'] if d else '-'}"
            for i, (p, o, d) in enumerate(zip_longest(params, ours, entry_params))
        )
        raise RuntimeError(
            f"{kernel}: the launcher's device parameters (name (kind, size, align)) do not "
            f"match .entry {entry} of the PTX:\n{rows}"
        )
    params[:] = [
        {"i": i, **p, "decl": d["decl"]}
        for i, (p, d) in enumerate(zip(params, entry_params))
    ]


def _ints(rec: dict) -> list:
    """The values of a snapshot that may be byte addresses: an integer scalar, the recorded
    contents of an int64 tensor."""
    if rec["kind"] == "scalar":
        v = rec["value"]
        return [v] if isinstance(v, int) and not isinstance(v, bool) else []
    return (
        rec.get("values", [])
        if rec["kind"] == "tensor" and rec["dtype"] == "int64"
        else []
    )


def _plain(v):
    """JSON-friendly constexpr / option value (tl.constexpr and dtypes become plain)."""
    v = getattr(v, "value", v)
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return str(v)


def _snapshot(v) -> dict:
    if isinstance(v, TensorDescriptor):
        return {
            "kind": "desc",
            "dtype": str(v.base.dtype).removeprefix("torch."),
            "elem_bytes": v.base.element_size(),
            "shape": [int(x) for x in v.shape],
            "strides": [int(x) for x in v.strides],
            "block_shape": [int(x) for x in v.block_shape],
            "padding": v.padding,
            "base": _tensor(v.base),
        }
    if isinstance(v, torch.Tensor):
        return _tensor(v)
    return {"kind": "scalar", "value": _plain(v)}


def _tensor(t: torch.Tensor) -> dict:
    base = t.untyped_storage().data_ptr()
    if base % 16:
        raise RuntimeError(
            f"tensor storage at {base:#x} is not 16-byte aligned (CUDA allocations are), "
            "so the CPU specialization (tt.divisibility) could differ from the GPU's"
        )
    rec = {
        "kind": "tensor",
        "dtype": str(t.dtype).removeprefix("torch."),
        "elem_bytes": t.element_size(),
        "shape": list(t.shape),
        "strides": list(t.stride()),
        "storage_offset": t.storage_offset(),
        "storage": [
            base,
            base + t.untyped_storage().nbytes(),
        ],  # byte range of the storage
        "data_ptr": t.data_ptr(),
    }
    if not t.dtype.is_floating_point and t.numel() <= MAX_RECORDED_INTS:
        rec["values"] = t.reshape(-1).tolist()
    return rec
