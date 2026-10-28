# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Where host data comes from, relative to the launches a dump skips (both frontends).

A dump runs the host code on CPU and skips every launch, so a tensor that a kernel
writes keeps its CPU contents. Host code that reads such data back (`.item()`,
`.tolist()`, `int(t)`, `bool(t)`, printing, `numpy()`, the shape of `nonzero(t)`, ...)
may branch or size later launches differently than on the GPU, and a tensor the host
computes from it (a copy, a reduction, arithmetic, `.cpu()`) holds other values.
HostReads, a TorchFunctionMode active during the host calls of a dump (one instance for
the whole run: the inputs are consecutive calls of one process, and data a launch of one
input writes may reach a later one through a module-level cache), tracks it:

  * every storage passed to a launch is a possible source of stale data from then on, and
    so is every storage an integer argument of the launch points into (an address hint:
    an integer scalar or an element of an int64 tensor argument whose value is a byte
    address inside a storage some torch call touched; a kernel may write through it,
    e.g. through a grouped GEMM's pointer array);
  * a torch call with an input that may hold such data makes everything it writes (its
    tensor results, which include the tensor of an in-place op, and the target of
    `__setitem__`) derived from the same sources;
  * a call that turns such data into a Python value (any result that is not a tensor,
    except the metadata calls below) is a host read, logged with its sources and the
    number of launches made before it; so is an aten op of DATA_TAGS on such data (the
    shape of its result, or a Python value, depends on the data), which _Aten, a
    TorchDispatchMode entered with HostReads, logs. One call logs one read: a call-level
    read replaces the aten-level reads made inside it;
  * `launch()` returns, per argument, the sources its data may come from.

A move between host and device (`.cpu()`, `.cuda()`, `.to(device)`, `pin_memory()`, ...)
copies on the GPU, unless the data is already there, but returns its input on the CPU
host. Later writes through either tensor would then reach both on the CPU and only one on
the GPU, so a write (an aten op's mutable argument, _Aten) to a storage that such a move
returned is a HostAliasError.

A source is (storage, t): the contents of the storage when t launches had been made. It
is stale if one of the first t launches may write that storage; which launch writes what
is decided afterwards from the kernels' write sets (dump.py and to_kir.py of each
frontend; a launch whose write set is unknown may write its tensor arguments and the
storages its address hints point into). Storages are identified by their data pointer;
every tensor involved is kept alive so that no address is reused during the run. The
frontend's own reads (argument snapshots) run `paused()`.
"""

import bisect
import traceback
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

# Calls whose non-tensor results are metadata, not data: shape, dtype and the other
# property getters (`__get__`), strides, pointers, the DLPack export; `__iter__` returns
# an iterator over tensors that `unbind` (tracked) produced.
METADATA = frozenset({
    "__get__", "size", "stride", "dim", "ndimension", "numel", "nelement", "element_size",
    "data_ptr", "storage_offset", "is_contiguous", "untyped_storage", "__len__", "__hash__",
    "__dlpack__", "__dlpack_device__", "get_device", "is_floating_point", "is_complex",
    "is_signed", "is_pinned", "type", "__iter__",
})  # fmt: skip

# aten ops whose result shape (dynamic_output_shape: nonzero, boolean indexing,
# masked_select, unique, repeat_interleave, bincount, ...) or Python value
# (data_dependent_output: _local_scalar_dense, behind item(), int(), empty(t), arange(t),
# narrow(t)) depends on the data. Shapes are metadata, so these are the host reads.
DATA_TAGS = (torch.Tag.dynamic_output_shape, torch.Tag.data_dependent_output)

# Calls that move data between host and device: they may return their input on the CPU
# host (same storage) where the GPU makes a copy. `to`, `type`, `as_tensor` and `asarray`
# only when given a device (or a tensor or type naming one).
MOVES = frozenset({"cpu", "cuda", "pin_memory", "to", "type", "as_tensor", "asarray"})


class HostAliasError(Exception):
    pass


class HostReads(TorchFunctionMode):
    def __init__(self, root: Path):
        super().__init__()
        self.root = Path(root).resolve()
        self.launches = 0
        self.reads: list = []  # [{"op", "loc", "after", "sources": [[storage, t], ...]}]
        self._launched: set = set()
        self._derived: dict = {}  # storage key -> {(storage, t)}
        self._bytes: dict = {}  # storage key -> its size, for every storage a torch call touched
        self._moved: dict = {}  # storage key -> the source line of a move that returned it
        self._keep: list = []
        self._paused = False
        self._aten = _Aten(self)

    def __enter__(self):
        super().__enter__()
        self._aten.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._aten.__exit__(exc_type, exc_val, exc_tb)
        return super().__exit__(exc_type, exc_val, exc_tb)

    @contextmanager
    def paused(self):
        saved, self._paused = self._paused, True
        try:
            yield
        finally:
            self._paused = saved

    def pointees(self, values: list) -> list:
        """The storages (keys) that contain one of the integer `values` as a byte address,
        among those some torch call touched: the storages a launch's address hints point
        into."""
        starts = sorted(self._bytes)
        out = set()
        for v in values:
            i = bisect.bisect_right(starts, v) - 1
            if i >= 0 and v < starts[i] + self._bytes[starts[i]]:
                out.add(starts[i])
        return sorted(out)

    def launch(self, tensors: list, addressed: Iterable[int] = ()) -> list:
        """Per launch argument, the sources its data may come from ([[storage, t], ...]);
        the arguments' storages and the `addressed` ones (pointees of the launch's address
        hints) are possible sources from now on. Call paused()."""
        out = [[list(s) for s in sorted(self.sources(t))] for t in tensors]
        for t in tensors:
            key = _key(t)
            if isinstance(key, int):  # kernels take tensors with storage
                self._launched.add(key)
                self._keep.append(t)
        self._launched.update(addressed)
        self.launches += 1
        return out

    def sources(self, t: torch.Tensor) -> set:
        key = _key(t)
        if key is None:
            return set()
        out = set(self._derived.get(key, ()))
        if key in self._launched:
            out.add((key, self.launches))
        return out

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        first = len(self.reads)
        result = func(*args, **kwargs)
        if self._paused:
            return result
        inputs = _tensors((args, kwargs))
        for t in inputs + _tensors(result):
            key = _key(t)
            if isinstance(key, int):
                self._bytes[key] = t.untyped_storage().nbytes()
        name = getattr(func, "__name__", "")
        if name in MOVES and _aliasing_move(name, args, kwargs, result):
            self._moved[_key(result)] = self._loc()
            self._keep.append(result)
        src = set().union(*map(self.sources, inputs))
        if not src:
            return result
        if name not in METADATA and any(
            not isinstance(x, torch.Tensor) and x is not None for x in _leaves(result)
        ):
            del self.reads[first:]  # the aten-level reads made inside this call
            self.log(getattr(func, "__qualname__", name), src)
        written = _tensors(result) + (list(args[:1]) if name == "__setitem__" else [])
        for t in written:
            key = _key(t)
            if key is not None:
                self._derived.setdefault(key, set()).update(src)
                self._keep.append(t)
        return result

    def log(self, op: str, src: set) -> None:
        """Log a host read of data with sources `src`."""
        self.reads.append(
            {
                "op": op,
                "loc": self._loc(),
                "after": self.launches,
                "sources": [list(s) for s in sorted(src)],
            }
        )

    def _loc(self) -> str:
        """The innermost frame of the exported sources (the host code that read)."""
        for f in reversed(traceback.extract_stack()):
            path = Path(f.filename).resolve()
            if path.is_relative_to(self.root):
                return f"{path.relative_to(self.root)}:{f.lineno}"
        return "?"


class _Aten(TorchDispatchMode):
    """The aten ops of the host code: logs the ops of DATA_TAGS whose inputs may hold
    launched data as host reads of `reads`, and refuses writes to storages a move between
    host and device returned."""

    def __init__(self, reads: HostReads):
        super().__init__()
        self.reads = reads

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if not self.reads._paused:
            if any(tag in func.tags for tag in DATA_TAGS):
                src = set().union(*map(self.reads.sources, _tensors((args, kwargs))))
                if src:
                    self.reads.log(str(func), src)
            if self.reads._moved:
                for t in _written(func, args, kwargs):
                    if (moved := self.reads._moved.get(_key(t))) is not None:
                        raise HostAliasError(
                            f"{self.reads._loc()} writes ({func}) data that {moved} moved between host and device: on "
                            "the GPU the move copies it (unless it is already there), on the CPU host it returns the "
                            "same tensor, so the dump cannot tell which of the two the write changes"
                        )
        return func(*args, **kwargs)


def _aliasing_move(name: str, args: tuple, kwargs: dict, result) -> bool:
    """A move between host and device (MOVES) that returned its input's storage."""
    key = _key(result) if isinstance(result, torch.Tensor) else None
    if not (
        isinstance(key, int)
        and args
        and isinstance(args[0], torch.Tensor)
        and _key(args[0]) == key
    ):
        return False
    if name in ("cpu", "cuda", "pin_memory"):
        return True
    return kwargs.get("device") is not None or any(
        isinstance(a, (torch.device, str, torch.Tensor, type)) for a in args[1:]
    )


def _written(func, args: tuple, kwargs: dict) -> list:
    """The tensors an aten op writes: its schema's mutable arguments."""
    out = []
    for i, a in enumerate(func._schema.arguments):
        if a.alias_info is not None and a.alias_info.is_write:
            out += _tensors(args[i] if i < len(args) else kwargs.get(a.name))
    return out


def _key(t: torch.Tensor):
    """The storage a tensor's data lives in: its data pointer (the object id for tensor
    subclasses without a storage); None for a tensor without data."""
    try:
        storage = t.untyped_storage()
    except (RuntimeError, TypeError):  # NotImplementedError is a RuntimeError
        return ("tensor", id(t))
    return storage.data_ptr() if storage.nbytes() else None


def _leaves(x) -> list:
    if isinstance(x, (list, tuple)):
        return [leaf for v in x for leaf in _leaves(v)]
    if isinstance(x, dict):
        return [leaf for v in x.values() for leaf in _leaves(v)]
    return [x]


def _tensors(x) -> list:
    return [t for t in _leaves(x) if isinstance(t, torch.Tensor)]
