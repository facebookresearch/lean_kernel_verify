# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""dump: run a kernel's real host code path on a CPU-only host and record

  * the initial CuTe MLIR of every kernel the host compiles, plus its analyzed stage
    (stage.py), and the PTX (with line info) and cubin the DSL compiles it to,
  * the arguments of every launch the host performs (tensor dtype/shape/stride/storage,
    integer tensor contents, scalars), and
  * per launch, the LaunchRecord of every kernel launch its compiled host code makes
    (host_patches.py P6/P9: grid, block, cluster, dynamic shared memory, attributes and
    every kernel parameter, CUtensorMaps decoded, addresses resolved to the named
    tensors; frontend/ptx/launch.py), with each parameter's origin in the traced host
    code (P11).

The environment is pinned first (ttgir.host_env, shared with the TTGIR frontend):
variables the exported sources and the case's builder read, and compiler switches
(CUTE_DSL_*, TRITON_*, MLIR_*), come from the case's `env` only, plus the frontend's own
CUTE_DSL_ARCH (and CUTE_DSL_LINEINFO with DSL 4.6.1). The kernel package is then
imported from the exported source directory the case names (`source.src`, or --src),
never from the build graph. The inputs of a case run in order in one process, like
consecutive calls of one training process, so host compile caches behave as in
production. GPU-only calls are replaced by host_patches.py.

Output: <out>/dump.json, <out>/ir/ and <out>/ptx/
  {"case", "source", "num_sms", "dsl", "env",
   "compiles": [{"index", "callable", "params", "flags", "digest", "ir", "pipeline",
                 "stage", "host_stage", "verifier_failures"} | {..., "stage_error"},
                 + "ptx": {"file", "sha256", "target", "cubin", "tvm_ffi",
                           ["lineinfo_check"]} | null,
                 ["ptx_error"]],
   "inputs": [{"name", "launches": [{"compile", "args", "hinted", "launch_records",
                                     ["capture_error"]}], "host_reads", "meta", ["host_error"]}]}

ptx/<NN>_<digest>.ptx and .cubin are compile NN's (the digest is the variant's, of the
initial MLIR: two variants can compile to the same PTX). A launch record is {"kernel",
"entry", "ptx": {"file", "sha256"}, "target", "grid", "block", "cluster",
"smem_dynamic", "attrs", "kernel_attrs", "params", "jit_params", ["jit_params_error"]}
(host_patches.complete, host_patches._name_params). "jit_params" has one entry per
"params" entry, its origin in the traced host code:
  {"arg", "leaf"[, "part"]}  the parameter passes leaf `leaf` of the compile parameter
                             `arg` (the compile's "params" leaves; "part" k: the leaf's
                             k-th value) through unchanged. A tensor's is its data
                             pointer, whatever else points there or if it is empty, so
                             the parameter is that tensor even where its address resolves
                             to several leaves (one tensor under several jit names) or to
                             none (a zero-length tensor: data pointer 0);
  {"from": [leaves]}         a value the host computed (a TMA descriptor, a view, a
                             divisor): the leaves it is computed from, all of them (a
                             superset), so only those of the tensors its address resolves
                             to ("tensor" of a field, "base" of a "tensormap") can be it.
"jit_params" is null, with "jit_params_error" saying why, if the parameters cannot be
named exactly (host_patches P11).

With `lineinfo_check` (DSL 4.4.2) every compile is also compiled without line info
(ptx/<NN>_<digest>.noline.ptx) and "lineinfo_check" says whether the instructions are the
same ({"identical", "lines", "differing_lines"} | {"error"}). A capture that fails is
recorded ("ptx_error", "capture_error") and printed; it changes nothing else.

"dsl" is the CuTe DSL this process loaded (dsl_info); to_kir.py checks its version.
"env" is the pinned environment (host_env.record). One ttgir.host_reads.HostReads
tracks the whole run (data a launch of one input writes may reach a later input): every
tensor argument of a launch carries "sources", "hinted" lists the storages its integer
arguments point into, and "host_reads" lists the input's host reads of data passed to an
earlier launch of the run ("after" and the sources' launch counts are run-wide);
to_kir.py decides from the kernels' write sets which of them are stale. An input the host
refuses (ttgir.host_errors: the exported sources raise, a `raise` or `assert` statement such as a
validation check, or a torch check they call fails) keeps the launches made before the exception and records the
exception and its innermost source line as "host_error"; so does, with "no launch", an
input for which the host launches nothing. The next inputs still run. Any other exception
(a direct call of a @cute.jit function reaching the patched compile, a CUDA call on the
CPU-only host) fails the dump.
"""

import difflib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import platform
import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..ttgir import host_env


def _pin_env(case: dict, roots: list) -> dict:
    """Pin the environment the DSL and the kernel package read at import time (so a
    case's `env` needs a fresh process; the CLI runs one case per process), after
    checking that this process has the case's DSL (`dsl`, default 4.4.2)."""
    version = importlib.metadata.version("nvidia-cutlass-dsl")
    if version != case.get("dsl", "4.4.2"):
        raise SystemExit(
            f"dump: the case is for CuTe DSL {case.get('dsl', '4.4.2')}, this binary has "
            f'{version} (scripts/lkv.py picks the binary from the case\'s "dsl")'
        )
    own = {"CUTE_DSL_ARCH": "sm_100a"}
    if version != "4.4.2":
        # DSL 4.6.1 gives ops source locations only with line info (4.4.2 always does,
        # and line info would change its pass pipeline).
        own["CUTE_DSL_LINEINFO"] = "1"
    return host_env.pin(case.get("env", {}), roots, own)


def _load_builder(case_path: Path, spec: str):
    """`file.py:function`, the file relative to the case file (or absolute)."""
    file_name, func = spec.split(":")
    path = case_path.parent / file_name
    mod_spec = importlib.util.spec_from_file_location(f"lkv_case_{path.stem}", path)
    if mod_spec is None or mod_spec.loader is None:
        raise SystemExit(f"dump: cannot load the builder file {path}")
    module = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(module)
    return getattr(module, func)


def dsl_info() -> dict:
    """The CuTe DSL of this process: `cutlass.__version__`, the files actually imported
    (Python package, native MLIR library), the installed versions of the DSL's
    distributions, of quack and of the CUDA bindings, and the Python version."""
    import cutlass

    files = (
        getattr(m, "__file__", None) or "" for name, m in sys.modules.items()
        if name.startswith("cutlass._mlir._mlir_libs.")
    )  # fmt: skip
    native = sorted(f for f in files if f.endswith(".so"))
    installed = {}
    for d in importlib.metadata.distributions():
        name = d.metadata["Name"].lower().replace("_", "-")
        if name.startswith(("nvidia-cutlass-dsl", "quack-kernels", "cuda-bindings")):
            installed[name] = d.version
    return {
        "version": cutlass.__version__,
        "package": str(Path(inspect.getfile(cutlass)).resolve().parent),
        "native": [str(Path(f).resolve()) for f in native],
        "installed": dict(sorted(installed.items())),
        "python": platform.python_version(),
    }


def _resolve(entry: str) -> Callable[..., Any]:
    module_name, attr = entry.split(":")
    obj: Any = importlib.import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def run(
    case_path: Path,
    out: Path,
    src: str | None = None,
    only: list | None = None,
    lineinfo_check: bool = False,
) -> dict:
    try:
        return _run(case_path, out, src, only, lineinfo_check)
    finally:
        host_env.release()


def _empty_dir(d: Path) -> Path:
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def ptx_code(ptx: str) -> list:
    """The lines of a PTX module without line info: .loc and .file directives and debug
    sections dropped, whitespace normalized."""
    out, skip = [], False
    for line in ptx.splitlines():
        s = line.strip()
        if s.startswith(".section") and ".debug" in s:
            skip = True
        if skip:
            skip = s != "}"
            continue
        if s and not s.startswith((".loc", ".file")):
            out.append(" ".join(s.split()))
    return out


def _lineinfo_check(c) -> dict:
    """Whether a compile's PTX (c, host_patches.PtxCapture) has the same instructions
    with and without line info."""
    if c.ptx is None or c.ptx_noline is None:
        return {"error": c.lineinfo_error or "no PTX"}
    a, b = ptx_code(c.ptx), ptx_code(c.ptx_noline)
    changed = [
        d
        for d in difflib.unified_diff(a, b, lineterm="", n=0)
        if d[:1] in "+-" and not d.startswith(("+++", "---"))
    ]
    return {"identical": a == b, "lines": len(a), "differing_lines": len(changed)}


def _write_ptx(c, stem: Path, lineinfo_check: bool) -> dict:
    """Write a compile's PTX and cubin (c, host_patches.PtxCapture) to <stem>.*; the
    compile's dump.json keys for them ("ptx", "ptx_error")."""
    if c.error:
        print(f"dump: PTX capture of {stem.name} failed: {c.error}")
    error = {"ptx_error": c.error} if c.error else {}
    if c.ptx is None:
        return {"ptx": None} | error
    Path(f"{stem}.ptx").write_text(c.ptx)
    target = re.search(r"^\s*\.target\s+(\w+)", c.ptx, re.M)
    entry: dict = {
        "file": f"{stem}.ptx",
        "sha256": hashlib.sha256(c.ptx.encode()).hexdigest(),
        "target": target[1] if target else None,
        "cubin": None,
        "tvm_ffi": c.tvm_ffi,
    }
    if c.cubin is not None:
        Path(f"{stem}.cubin").write_bytes(c.cubin)
        entry["cubin"] = f"{stem}.cubin"
    if lineinfo_check:
        if c.ptx_noline is not None:
            Path(f"{stem}.noline.ptx").write_text(c.ptx_noline)
        entry["lineinfo_check"] = _lineinfo_check(c)
    return {"ptx": entry} | error


def _launch(la) -> dict:
    """A launch's dump.json entry: its recorded arguments, then its LaunchRecords."""
    d = {"compile": la.compile, "args": la.args, "hinted": la.hinted}
    d["launch_records"] = la.records
    if la.capture_error:
        d["capture_error"] = la.capture_error
    return d


def _head_launch_records(result: dict) -> None:
    """Head every LaunchRecord with its kernel (jit class), PTX file and target, and
    report the launches whose capture failed."""
    for inp in result["inputs"]:
        for la in inp["launches"]:
            c = result["compiles"][la["compile"]]
            ptx = c["ptx"] or {}
            head = {
                "kernel": c["callable"].rsplit(".", 1)[-1],
                "ptx": {"file": ptx.get("file"), "sha256": ptx.get("sha256")},
                "target": ptx.get("target"),
            }
            la["launch_records"] = [head | r for r in la["launch_records"]]
            if "capture_error" in la:
                print(
                    f"dump: input {inp['name']}: capture failed: {la['capture_error']}"
                )


def _run(
    case_path: Path, out: Path, src: str | None, only: list | None, lineinfo_check: bool
) -> dict:
    case = json.loads(case_path.read_text())
    src_root = Path(src or case["source"]["src"]).resolve()
    if not (src_root / case["source"]["package"]).exists():
        raise SystemExit(
            f"dump: {src_root / case['source']['package']} does not exist; export it first"
        )
    builder_file = case_path.parent / case["builder"].split(":")[0]
    pinned = _pin_env(case, [src_root, builder_file])
    ir_dir, ptx_dir = _empty_dir(out / "ir"), _empty_dir(out / "ptx")
    sys.dont_write_bytecode = True  # keep the case and source directories clean
    sys.path.insert(0, str(src_root))

    from ..ttgir.host_errors import (  # noqa: PLC0415 - after the environment is set
        host_rejection,
    )
    from ..ttgir.host_reads import HostReads  # noqa: PLC0415
    from . import host_patches, stage  # noqa: PLC0415

    builder = _load_builder(case_path, case["builder"])
    entry = _resolve(case["entry"])
    entry_source = inspect.getsourcefile(inspect.unwrap(entry))
    if entry_source is None:
        raise SystemExit(f"dump: {case['entry']} has no Python source file")
    entry_file = Path(entry_source).resolve()
    if not entry_file.is_relative_to(src_root):
        raise SystemExit(
            f"dump: {case['entry']} was imported from {entry_file}, not from {src_root}"
        )

    dsl = dsl_info()
    print(
        f"dump: CuTe DSL {dsl['version']} from {dsl['package']} (Python {dsl['python']}; {dsl['installed']})"
    )
    result = {
        "case": case["name"],
        "source": {**case["source"], "src": str(src_root)},
        "num_sms": case["num_sms"],
        "dsl": dsl,
        "compiles": [],
        "inputs": [],
    }
    with host_patches.installed(case["num_sms"], lineinfo_check) as rec:
        rec.reads = HostReads(
            src_root
        )  # one for the run: data may reach a later input (caches)
        for inp in case["inputs"]:
            if only and inp["name"] not in only:
                continue
            first, first_read = len(rec.launches), len(rec.reads.reads)
            record = {"name": inp["name"]}
            with rec.reads:
                args, kwargs, record["meta"] = builder(inp)
                try:
                    entry(*args, **kwargs)
                except Exception as e:  # noqa: BLE001 - the host refusing an input is a result
                    if (rejected := host_rejection(e, src_root)) is None:
                        raise
                    record["host_error"] = rejected
            record["host_reads"] = rec.reads.reads[first_read:]
            record["launches"] = [_launch(la) for la in rec.launches[first:]]
            if not record["launches"] and "host_error" not in record:
                record["host_error"] = "no launch"
            if "host_error" in record:
                print(
                    f"dump: the host rejected input {inp['name']}: {record['host_error']}"
                )
            result["inputs"].append(record)
    result["env"] = host_env.record(pinned)
    stages: dict = {}
    for c in rec.compiles:
        digest = hashlib.sha256(c.mlir.encode()).hexdigest()[:16]
        ir_path = ir_dir / f"{c.index:02d}_{digest}.mlir"
        ir_path.write_text(c.mlir)
        if digest not in stages:
            try:
                kernel_text, host_text, failures = stage.lower(c.mlir, c.pipeline)
                (ir_dir / f"{digest}.stage.mlir").write_text(kernel_text)
                (ir_dir / f"{digest}.host.mlir").write_text(host_text)
                stages[digest] = {
                    "stage": str(ir_dir / f"{digest}.stage.mlir"),
                    "host_stage": str(ir_dir / f"{digest}.host.mlir"),
                    "verifier_failures": failures,
                }
            except Exception as e:  # noqa: BLE001 - the compiler's own diagnostic is the result
                stages[digest] = {"stage_error": f"{type(e).__name__}: {e}"}
        result["compiles"].append(
            {
                "index": c.index,
                "callable": c.callable,
                "params": c.params,
                "flags": c.flags,
                "digest": digest,
                "ir": str(ir_path),
                "pipeline": c.pipeline,
                **stages[digest],
            }
            | _write_ptx(c.capture, ptx_dir / f"{c.index:02d}_{digest}", lineinfo_check)
        )
    _head_launch_records(result)
    (out / "dump.json").write_text(json.dumps(result, indent=1))
    return result
