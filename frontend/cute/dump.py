# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""dump: run a kernel's real host code path on a CPU-only host and record

  * the initial CuTe MLIR of every kernel the host compiles, plus its analyzed stage
    (stage.py), and
  * the arguments of every launch the host performs (tensor dtype/shape/stride/storage,
    integer tensor contents, scalars).

The environment is pinned first (ttgir.host_env, shared with the TTGIR frontend):
variables the exported sources and the case's builder read, and compiler switches
(CUTE_DSL_*, TRITON_*, MLIR_*), come from the case's `env` only, plus the frontend's own
CUTE_DSL_ARCH, CUTE_DSL_KEEP_IR and CUTE_DSL_DUMP_DIR. The kernel package is then
imported from the exported source directory the case names (`source.src`, or --src),
never from the build graph. The inputs of a case run in order in one process, like
consecutive calls of one training process, so host compile caches behave as in
production. GPU-only calls are replaced by host_patches.py.

Output: <out>/dump.json and <out>/ir/
  {"case", "source", "num_sms", "dsl", "env",
   "compiles": [{"index", "callable", "params", "flags", "digest", "ir", "pipeline",
                 "stage", "host_stage", "verifier_failures"} | {..., "stage_error"}],
   "inputs": [{"name", "launches": [{"compile", "args", "hinted"}], "host_reads", "meta", ["host_error"]}]}

"dsl" is the CuTe DSL this process loaded (dsl_info); to_kir.py checks its version.
"env" is the pinned environment (host_env.record). One ttgir.host_reads.HostReads
tracks the whole run (data a launch of one input writes may reach a later input): every
tensor argument of a launch carries "sources", "hinted" lists the storages its integer
arguments point into, and "host_reads" lists the input's host reads of data passed to an
earlier launch of the run ("after" and the sources' launch counts are run-wide);
to_kir.py decides from the kernels' write sets which of them are stale. An input the host
refuses (ttgir.host_errors: the exported sources raise, e.g. a validation check, or a
torch check they call fails) keeps the launches made before the exception and records the
exception and its innermost source line as "host_error"; so does, with "no launch", an
input for which the host launches nothing. The next inputs still run. Any other exception
(a direct call of a @cute.jit function reaching the patched compile, a CUDA call on the
CPU-only host) fails the dump.
"""

import atexit
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import platform
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..ttgir import host_env


def _pin_env(case: dict, roots: list) -> dict:
    """Pin the environment the DSL and the kernel package read at import time (so a
    case's `env` needs a fresh process; the CLI runs one case per process). The DSL
    writes the initial MLIR to a process scratch directory (dump.json keeps copies under
    <out>/ir/)."""
    raw = tempfile.mkdtemp(prefix="lkv_cute_raw_ir_")
    atexit.register(shutil.rmtree, raw, ignore_errors=True)
    own = {
        "CUTE_DSL_ARCH": "sm_100a",
        "CUTE_DSL_KEEP_IR": "1",
        "CUTE_DSL_DUMP_DIR": raw,
    }
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
    case_path: Path, out: Path, src: str | None = None, only: list | None = None
) -> dict:
    try:
        return _run(case_path, out, src, only)
    finally:
        host_env.release()


def _run(case_path: Path, out: Path, src: str | None, only: list | None) -> dict:
    case = json.loads(case_path.read_text())
    src_root = Path(src or case["source"]["src"]).resolve()
    if not (src_root / case["source"]["package"]).exists():
        raise SystemExit(
            f"dump: {src_root / case['source']['package']} does not exist; export it first"
        )
    ir_dir = out / "ir"
    shutil.rmtree(ir_dir, ignore_errors=True)
    ir_dir.mkdir(parents=True)
    builder_file = case_path.parent / case["builder"].split(":")[0]
    pinned = _pin_env(case, [src_root, builder_file])
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
    with host_patches.installed(case["num_sms"]) as rec:
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
            record["launches"] = [
                {"compile": la.compile, "args": la.args, "hinted": la.hinted}
                for la in rec.launches[first:]
            ]
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
        )
    (out / "dump.json").write_text(json.dumps(result, indent=1))
    return result
