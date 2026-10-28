# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Case-driven TTGIR dump: run the real host code on CPU, capture the launch, compile.

    dump --case CASE.json [--out /tmp/lkv_out/tlx/<case>] [--inputs a,b]
         [--src DIR] [--target sm_100|sm_103]

The environment is pinned first (host_env.py): variables the exported sources read and
compiler switches come from the case's `env` only. Then Triton and the kernel module are
imported. For every input of the case the host entry point is called on CPU tensors
(gpu_free.py answers device queries for the target, capture.py intercepts launches).
Every launch is recorded with its bound arguments, constexprs, specialization, options
and grid; each distinct specialization is compiled once with
`triton.compile(ASTSource(fn, signature, constexprs, attrs), target=<target>,
options=...)`, the same call JITFunction makes on a cache miss on that GPU.

An input the host refuses (host_errors.py: the exported sources raise, or a torch check
they call fails) keeps the launches made before the exception and records the exception
and its innermost source line as "host_error"; so does, with "no launch of <kernel>", an
input for which the host makes no launch of the case's kernel. The next inputs still run.
Any other exception raised by the host call fails the dump.

Every launch is skipped, so a launch's write set (provenance.py, from its TTGIR; if it is
unknown, every tensor argument and every storage its address hints point into) says which
storages hold stale CPU contents afterwards. The inputs run in one process like
consecutive calls, so a launch of one input may write what a later input reads (a
module-level cache): sources, launch indices and write sets span the whole run. Per
launch of the case's kernel, dump.json lists the earlier launches of the run (of any
kernel, the case's included) that may have written each tensor argument's data
(`stale`, directly or through host computation), and per input the host reads made so
far in the run of data an earlier launch may have written (`stale_reads`,
host_reads.py). to_kir refuses metadata and inputs that depend on them.

Writes <out>/<spec>.ttgir per specialization of the case's kernel and <out>/dump.json
(see references/frontend_ttgir.md for its format).
"""

import argparse
import atexit
import hashlib
import importlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from . import host_env

DEFAULT_OUT = Path("/tmp/lkv_out/tlx")


def check_exported(root: Path) -> None:
    """Every loaded module of a package the export provides was loaded from the export.

    A module missing from the export (or shadowed by an earlier sys.path entry) would
    otherwise be analyzed from wherever Python finds it."""
    root = root.resolve()
    tops = {p.name for p in root.iterdir() if p.is_dir()} | {
        p.stem for p in root.glob("*.py")
    }
    for name, mod in list(sys.modules.items()):
        if mod is None or name.split(".")[0] not in tops:
            continue
        paths = (
            [mod.__file__]
            if getattr(mod, "__file__", None)
            else list(getattr(mod, "__path__", []))
        )
        for path in paths:
            if not Path(path).resolve().is_relative_to(root):
                raise RuntimeError(
                    f"module {name} was loaded from {path}, not from the export {root}"
                )


def load_entry(case: dict, mode):
    """Import the kernel module from the exported source tree; return (module, entry)."""
    root = Path(case["src_root"])
    if not (root / (case["module"].replace(".", "/") + ".py")).is_file():
        raise RuntimeError(
            f"{case['module']} not found under {root} (export it with export_src.py)"
        )
    sys.path.insert(0, str(root))
    sys.dont_write_bytecode = True  # keep the exported source tree free of __pycache__
    with mode:
        module = importlib.import_module(case["module"])
    check_exported(root)
    for name, value in case.get("module_overrides", {}).items():
        if not hasattr(module, name):
            raise RuntimeError(
                f"module_overrides: {case['module']} has no attribute {name}"
            )
        setattr(module, name, value)
    entry = getattr(module, case["entry"])
    # torch.library.custom_op wraps the Python function; call it directly so the
    # dispatcher does not run fake-tensor/alias checks on the (None) launch result.
    if hasattr(entry, "_init_fn"):
        entry = entry._init_fn
    return module, entry


def spec_key(launch) -> str:
    blob = json.dumps(
        [
            launch.kernel,
            launch.signature,
            launch.constexprs,
            launch.attrs,
            launch.options,
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--inputs", help="comma-separated subset of input names")
    ap.add_argument(
        "--src", type=Path, help="exported source root (overrides the case's src_root)"
    )
    ap.add_argument(
        "--target",
        choices=("sm_100", "sm_103"),
        default="sm_100",
        help="GPU the launch is captured and compiled for (gpu_free.TARGETS; default sm_100)",
    )
    args = ap.parse_args()
    case = json.loads(args.case.read_text())
    if args.src:
        case["src_root"] = str(args.src.resolve())
    out = args.out or DEFAULT_OUT / case["name"]
    run(case, out, set(args.inputs.split(",")) if args.inputs else None, args.target)


def run(
    case: dict, out: Path, wanted: set | None = None, target: str = "sm_100"
) -> dict:
    """Pin the environment, capture every input's launches, compile each specialization,
    write <out>."""
    try:
        return _run(case, out, wanted, target)
    finally:
        host_env.release()


def _run(case: dict, out: Path, wanted: set | None, target: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    root = Path(case["src_root"])
    # Triton writes every compile to its cache directory (always_compile, gpu_free.py,
    # keeps it from serving one): a process scratch directory, the frontend's own setting.
    cache = tempfile.mkdtemp(prefix="lkv_triton_cache_")
    atexit.register(shutil.rmtree, cache, ignore_errors=True)
    pinned = host_env.pin(case.get("env", {}), [root], {"TRITON_CACHE_DIR": cache})
    # Imported only now: Triton reads its environment at import (and load_entry imports
    # the kernel module).
    import triton  # noqa: PLC0415

    # triton._C.libtriton is Triton's native extension module; it ships no type stubs.
    # pyrefly: ignore [missing-import]
    from triton._C import libtriton  # noqa: PLC0415

    from . import capture, case_args, gpu_free, provenance  # noqa: PLC0415
    from .host_errors import host_rejection  # noqa: PLC0415
    from .host_reads import HostReads  # noqa: PLC0415

    # Every variable libtriton's passes read is on its cache-invalidating list; those the
    # shell set and the case does not are dropped too (libtriton reads them at compile time).
    host_env.drop(pinned, libtriton.get_cache_invalidating_env_vars())
    mode = gpu_free.install(case["num_sms"], target)
    module, entry = load_entry(case, mode)

    specs: dict = {}  # key -> first launch of the case's kernel with that specialization
    kernels: dict = {}  # key -> compiled kernel, for every launched specialization
    write_sets: dict = {}  # key -> params it may write (None: any pointer argument)
    inputs = []
    tracker = HostReads(
        root
    )  # one for the run: data may reach a later input (module-level caches)
    writes = []  # per launch of the run: storage -> the params that may write it
    for inp in case["inputs"]:
        if wanted is not None and inp["name"] not in wanted:
            continue
        rejected = None
        with mode, capture.Capture(tracker) as cap:
            call = case_args.build_call(case, inp, module)
            try:
                entry(**call)
            except Exception as e:  # noqa: BLE001 - the host refusing an input is a result
                if (rejected := host_rejection(e, root)) is None:
                    raise
        launches: list[dict] = []
        for launch in cap.launches:
            k = len(writes)
            key = spec_key(launch)
            if key not in kernels:
                t0 = time.time()
                kernels[key] = launch.compile(mode, target)
                write_sets[key] = provenance.written_params(kernels[key].asm["ttgir"])
                print(
                    f"[dump] compiled {launch.kernel} {key} in {time.time() - t0:.1f}s",
                    file=sys.stderr,
                )
            if launch.kernel == case["kernel"]:
                specs.setdefault(key, launch)
                stale = {
                    p: w
                    for p, src in launch.sources.items()
                    if (w := writers(src, writes))
                }
                launches.append(
                    {
                        "spec": key,
                        "index": k,
                        "grid": list(launch.grid),
                        "config": launch.config,
                        "args": launch.args,
                        "stale": stale,
                    }
                )
            writes.append(written_storages(launch, write_sets[key]))
        others = sorted(
            {la.kernel for la in cap.launches if la.kernel != case["kernel"]}
        )
        if rejected is None and not launches:
            rejected = f"no launch of {case['kernel']} (the host launched: {', '.join(others) or 'nothing'})"
        # Stale reads of the run so far: one in an earlier input may have filled a cache.
        reads = [
            {"op": r["op"], "loc": r["loc"], "after": r["after"], "writers": w}
            for r in tracker.reads
            if (w := writers(r["sources"], writes))
        ]
        record = {
            "name": inp["name"],
            "launches": launches,
            "other_launches": others,
            "stale_reads": reads,
        }
        if rejected is not None:
            record["host_error"] = rejected
            print(
                f"[dump] the host rejected input {inp['name']}: {rejected}",
                file=sys.stderr,
            )
        inputs.append(record)
        print(
            f"[dump] {inp['name']}: {len(launches)} launch(es) -> "
            f"{sorted({la['spec'] for la in launches})}",
            file=sys.stderr,
        )
    check_exported(root)  # host code may import lazily

    compiled = {}
    for key, launch in specs.items():
        kernel = kernels[key]
        (out / f"{key}.ttgir").write_text(kernel.asm["ttgir"])
        compiled[key] = {
            "kernel": launch.kernel,
            "ttgir": f"{key}.ttgir",
            "signature": launch.signature,
            "constexprs": launch.constexprs,
            "attrs": launch.attrs,
            "specialization": launch.specialization,
            "options": launch.options,
            "shared_bytes": kernel.metadata.shared,
        }

    dump = {
        "case": case["name"],
        "kernel": case["kernel"],
        "source": case.get("source", module.__file__),
        "src_root": case["src_root"],
        "num_sms": case["num_sms"],
        "target": {
            "name": target,
            "arch": gpu_free.TARGETS[target].arch,
            "max_shared_mem": gpu_free.MAX_SHARED_MEM,
        },
        "compiler": {
            "name": "triton",
            "version": triton.__version__,
            "libtriton_sha256": hashlib.sha256(
                Path(libtriton.__file__).read_bytes()
            ).hexdigest(),
        },
        "env": {
            **host_env.record(pinned),
            "triton_cache_invalidating": libtriton.get_cache_invalidating_env_vars(),
        },
        "specs": compiled,
        "inputs": inputs,
    }
    (out / "dump.json").write_text(json.dumps(dump, indent=1))
    rejected = sum("host_error" in i for i in inputs)
    print(
        f"[dump] wrote {out}/dump.json ({len(inputs)} inputs, {rejected} rejected by the host, "
        f"{len(compiled)} specs; Triton {triton.__version__}; case env {dump['env']['case']}; shell "
        f"values ignored: {sorted(dump['env']['unset'])})",
        file=sys.stderr,
    )
    return dump


def written_storages(launch, written: set | None) -> dict:
    """Storage -> the launch's params that may write it (if the write set is unknown, every
    tensor and descriptor argument, and every integer argument for the storages its values
    point into)."""
    out: dict = {}
    for name, rec in launch.args.items():
        base = rec.get("base") if rec["kind"] == "desc" else rec
        if (
            base is None
            or base["kind"] != "tensor"
            or (written is not None and name not in written)
        ):
            continue
        out.setdefault(base["storage"][0], []).append(
            {"kernel": launch.kernel, "param": name}
        )
    if written is None:
        for name, storages in launch.hinted.items():
            for storage in storages:
                out.setdefault(storage, []).append(
                    {"kernel": launch.kernel, "param": name}
                )
    return out


def writers(sources: list, writes: list) -> list:
    """The earlier launches that may have written data with these sources: (storage, t)
    is stale if one of the first t launches may write the storage."""
    return [
        {"launch": k, **w}
        for storage, t in sources
        for k in range(t)
        for w in writes[k].get(storage, [])
    ]


if __name__ == "__main__":
    main()
