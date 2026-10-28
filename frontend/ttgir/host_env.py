# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The environment of a frontend dump (both frontends), pinned before anything reads it.

Variant selection and compiler passes read environment variables. A dump must not take
values from the shell that the case does not state, so before the compiler and the
kernel sources are imported, `pin` unsets

  * every variable the exported sources read by name: `os.environ.get(K)`,
    `os.getenv(K)`, `os.environ[K]`, `K in os.environ`, `os.environ.pop(K)`,
    `os.environ.setdefault(K)` with K a string literal or a module-level string constant
    (any other key is an error: the variable cannot be pinned), and
  * every compiler switch (is_compiler_switch): TRITON_*, CUTE_DSL_*, MLIR_*, LLVM_* and
    NVPTX_* variables and the triton.knobs variables outside those prefixes,

except those the case's `env` sets, then applies the case's `env` and the frontend's own
settings (which a case may not override). Until `release`, a read of any other variable
by code under the scanned roots (through an alias of `os` or `os.environ`, a computed
key the scan could not see, ...) and any enumeration of the environment from there raise
EnvError: their values would come from the shell. `drop` unsets more switches found once
the compiler is imported (the TTGIR dump: libtriton's cache-invalidating variables).
`record` gives dump.json's "env": what was pinned and the effective values at the end of
the dump.

Stdlib only: the dumps import it before torch, Triton and the CuTe DSL.
"""

import _collections_abc
import ast
import os
import sys
from pathlib import Path

COMPILER_PREFIXES = ("TRITON_", "CUTE_DSL_", "MLIR_", "LLVM_", "NVPTX_")
# triton.knobs variables outside the prefixes (USE_IR_LOC rewrites every TTGIR location to
# the IR file; CC and the AMDGCN_* ones do not reach an NVIDIA TTGIR). tests/ttgir checks
# that every triton.knobs variable is a compiler switch.
COMPILER_NAMES = frozenset(
    {"USE_IR_LOC", "USE_PTX_LOC", "DISABLE_PTXAS_OPT", "PTXAS_OPTIONS", "ENABLE_TILE"}
)
# The guarded methods, and the code files between them and the code that reads the
# environment (os.getenv, Mapping.get, KeysView, ...; frozen or not).
_ENVIRON = (os._Environ.__getitem__, os._Environ.__iter__)
_LIBRARY = frozenset(
    {
        __file__,
        os.__file__,
        _collections_abc.__file__,
        "<frozen os>",
        "<frozen _collections_abc>",
    }
)


def is_compiler_switch(name: str) -> bool:
    return name.startswith(COMPILER_PREFIXES) or name in COMPILER_NAMES


class EnvError(Exception):
    pass


def source_reads(roots: list) -> set:
    """Names of the environment variables the Python files under `roots` (files or
    directories) read; EnvError names every read whose variable cannot be determined."""
    names, unknown = set(), []
    for root in map(Path, roots):
        for path in sorted(root.rglob("*.py")) if root.is_dir() else [root]:
            try:
                tree = ast.parse(path.read_text(errors="replace"), str(path))
            except SyntaxError as e:
                raise EnvError(f"cannot scan {path} for environment reads: {e}") from e
            for key, line in _reads(tree):
                if key is None:
                    unknown.append(f"{path}:{line}")
                else:
                    names.add(key)
    if unknown:
        raise EnvError(
            f"cannot tell which environment variable is read at {', '.join(unknown)} (the key is "
            "not a string literal or module-level string constant), so the dump cannot pin it"
        )
    return names


def _reads(tree: ast.Module):
    """(variable name or None, line) per environment read of a module."""
    consts = {}
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        value = getattr(node, "value", None)
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            consts.update(
                (t.id, value.value) for t in targets if isinstance(t, ast.Name)
            )

    def key(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return consts.get(node.id) if isinstance(node, ast.Name) else None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args:
            f = node.func
            environ_method = (
                isinstance(f, ast.Attribute)
                and f.attr in ("get", "pop", "setdefault")
                and _is_environ(f.value)
            )
            getenv = (
                isinstance(f, ast.Attribute)
                and f.attr == "getenv"
                and _is_name(f.value, "os")
            ) or _is_name(f, "getenv")
            if environ_method or getenv:
                yield key(node.args[0]), node.lineno
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Load)
            and _is_environ(node.value)
        ):
            yield key(node.slice), node.lineno
        elif isinstance(node, ast.Compare) and any(
            isinstance(o, (ast.In, ast.NotIn)) for o in node.ops
        ):
            if any(_is_environ(c) for c in node.comparators):
                yield key(node.left), node.lineno


def _is_name(node, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _is_environ(node) -> bool:
    """`os.environ` or `environ` (from os import environ)."""
    return _is_name(node, "environ") or (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and _is_name(node.value, "os")
    )


def pin(case_env: dict, roots: list, own: dict) -> dict:
    """Pin the environment for a dump of the sources under `roots`; returns the first part
    of the record: {"case", "own", "unset": {variable: dropped shell value}, "read_by_sources"}."""
    clash = sorted(set(case_env) & set(own))
    if clash:
        raise EnvError(f"the case's env sets {clash}, which the frontend sets itself")
    reads = source_reads(roots)
    unset = {}
    for name in sorted(os.environ):
        if name not in case_env and (name in reads or is_compiler_switch(name)):
            unset[name] = os.environ.pop(name)
    case = {k: str(v) for k, v in case_env.items()}
    os.environ.update(case)
    os.environ.update(own)
    _guard([Path(r).resolve() for r in roots], reads | set(case) | set(own))
    return {
        "case": case,
        "own": dict(own),
        "unset": unset,
        "read_by_sources": sorted(reads),
    }


def drop(pinned: dict, names) -> None:
    """Unset the variables among `names` that the case's env and the frontend do not set,
    recording their shell values in the pin record's "unset"."""
    for name in sorted(names):
        if (
            name in os.environ
            and name not in pinned["case"]
            and name not in pinned["own"]
        ):
            pinned["unset"][name] = os.environ.pop(name)


def _guard(roots: list, pinned: set) -> None:
    """Until release(), an environment read by code under `roots` of a variable that is
    neither pinned nor a compiler switch, and any enumeration of the environment from
    there, raise EnvError (the scan did not see the read, so the shell's value would be
    used)."""
    getitem, iterate = _ENVIRON
    under_roots: dict = {}  # code file name -> its resolved path if under one of the roots, else None

    def reader():
        """file:line of the code that called the guarded method, if it is under the roots
        (a library reading on its behalf, e.g. posixpath.expanduser, is not)."""
        f = sys._getframe(2)
        while f is not None and f.f_code.co_filename in _LIBRARY:
            f = f.f_back
        if f is None:
            return None
        name = f.f_code.co_filename
        if name not in under_roots:
            path = Path(name).resolve()
            under_roots[name] = (
                path if any(path.is_relative_to(r) for r in roots) else None
            )
        return f"{under_roots[name]}:{f.f_lineno}" if under_roots[name] else None

    def guarded_getitem(self, key):
        if (
            self is os.environ
            and key not in pinned
            and not is_compiler_switch(str(key))
            and (where := reader())
        ):
            raise EnvError(
                f"{where} reads environment variable {key}, which the source scan did not find "
                "(read through an alias?), so the dump did not pin it"
            )
        return getitem(self, key)

    def guarded_iter(self):
        if self is os.environ and (where := reader()):
            raise EnvError(
                f"{where} enumerates the environment; the dump can pin only variables read by name"
            )
        return iterate(self)

    os._Environ.__getitem__, os._Environ.__iter__ = guarded_getitem, guarded_iter


def release() -> None:
    """End of the dump: environment reads are no longer checked."""
    os._Environ.__getitem__, os._Environ.__iter__ = _ENVIRON


def record(pinned: dict) -> dict:
    """dump.json "env": the pin record plus the effective values, now, of every variable
    the sources read and of every compiler switch (the frontend's own settings and any the
    host code set while it ran)."""
    reads = set(pinned["read_by_sources"])
    effective = {
        k: v
        for k, v in sorted(os.environ.items())
        if k in reads or is_compiler_switch(k)
    }
    return {**pinned, "effective": effective}
