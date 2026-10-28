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

During the capture the dump's HostReads mode (host_reads.py, one for the whole run) tracks
host data: every launch records, per tensor argument, the earlier launches' storages its
data may come from (`Launch.sources`) and the storages its integer arguments point into
(`Launch.hinted`), and host reads of such data are logged (`HostReads.reads`). The grid
callable is host code and runs tracked; the binder and the argument snapshots run paused.
"""

import contextvars
from dataclasses import dataclass, field

import torch
from triton.compiler import ASTSource, compile as triton_compile
from triton.runtime.autotuner import Autotuner
from triton.runtime.jit import JITFunction
from triton.tools.tensor_descriptor import TensorDescriptor

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
    )


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
