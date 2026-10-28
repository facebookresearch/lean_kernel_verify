# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Case-file arguments of the host entry point as host values (references/frontend_ttgir.md)."""

import importlib

import torch


def build_arg(spec, ns: dict):
    """Case-file argument -> Python value. Tensors are {"dtype", "shape"} (zero-filled)
    or {"dtype", "values"}; {"ref": name} is a value bound by a setup step or another
    argument; everything else is passed through as is."""
    if not isinstance(spec, dict):
        return spec
    if set(spec) == {"ref"}:
        if spec["ref"] not in ns:
            raise ValueError(f"unbound ref {spec['ref']!r}")
        return ns[spec["ref"]]
    if set(spec) == {"dtype", "shape"}:
        return torch.zeros(spec["shape"], dtype=getattr(torch, spec["dtype"]))
    if set(spec) == {"dtype", "values"}:
        return torch.tensor(spec["values"], dtype=getattr(torch, spec["dtype"]))
    raise ValueError(f"unsupported argument spec {spec}")


def build_call(case: dict, inp: dict, module) -> dict:
    """Resolve an input's entry arguments, running the case's setup steps first.

    Setup steps (`{"call": fn, "args": {...}, "bind": [names]}`) are host helpers the
    real caller runs before the entry point (e.g. tile-table precomputation); `fn` is
    an attribute of the kernel module or a dotted path into the exported tree.
    """
    spec = {**case.get("defaults", {}), **inp["args"]}
    ns = {k: build_arg(v, {}) for k, v in spec.items() if not _has_ref(v)}
    for step in case.get("setup", []) + inp.get("setup", []):
        fn = _resolve(step["call"], module)
        out = fn(**{k: build_arg(v, ns) for k, v in step.get("args", {}).items()})
        names = step["bind"]
        values = out if len(names) > 1 else (out,)
        if len(values) != len(names):
            raise ValueError(
                f"setup {step['call']} returned {len(values)} values for {names}"
            )
        ns.update(zip(names, values))
    return {k: (build_arg(v, ns) if _has_ref(v) else ns[k]) for k, v in spec.items()}


def _has_ref(v) -> bool:
    return isinstance(v, dict) and set(v) == {"ref"}


def _resolve(path: str, module):
    if "." not in path:
        return getattr(module, path)
    mod, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(mod), attr)
