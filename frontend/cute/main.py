# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""lkv_cute: CuTe DSL frontend (Layer 1) of lean-kernel-verify.

  export --rev REV --path fbcode/<pkg dir> --out DIR [--repo DIR]
                                                       export a package at a revision
  dump CASE.json [--src DIR] [--out DIR] [--inputs a,b] host run: MLIR + launch arguments
  to_kir CASE.json [--out DIR]                         dump.json + stages -> KIR JSON
  run CASE.json [--src DIR] [--out DIR] [--inputs a,b]  dump, then to_kir

Default output directory: /tmp/lkv_out/<case dir name>/<case name>/. `export` reads the
repository that contains --repo (default: the current directory).
"""

import argparse
import json
import subprocess
from pathlib import Path


def default_out(case_path: Path) -> Path:
    case = json.loads(case_path.read_text())
    return Path("/tmp/lkv_out") / case_path.parent.name / case["name"]


def export(rev: str, path: str, out: Path, repo: Path | None = None) -> None:
    """Write every file under the repo-relative `path` at `rev` to out/<basename>/
    (read-only `sl root`, `sl files` and `sl cat` in the repository that contains `repo`,
    default: the current directory)."""
    root = subprocess.run(
        ["sl", "root"], capture_output=True, text=True, check=True, cwd=repo
    ).stdout.strip()
    path = path.rstrip("/")
    listing = subprocess.run(
        ["sl", "files", "-r", rev, "--include", f"path:{path}"],
        capture_output=True, text=True, check=True, cwd=root,
    ).stdout.split()  # fmt: skip
    files = [rel for rel in listing if rel.startswith(path + "/")]
    if not files:
        raise SystemExit(f"export: nothing under {path} at {rev}")
    for rel in files:
        data = subprocess.run(
            ["sl", "cat", "-r", rev, rel], capture_output=True, check=True, cwd=root
        ).stdout
        dst = out / Path(path).name / rel[len(path) + 1 :]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    print(f"exported {len(files)} files to {out / Path(path).name}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="lkv_cute", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export")
    p.add_argument("--rev", required=True)
    p.add_argument("--path", required=True, help="fbsource-relative package directory")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--repo",
        type=Path,
        help="a directory of the repository (default: the current directory)",
    )
    for name in ("dump", "to_kir", "run"):
        p = sub.add_parser(name)
        p.add_argument("case", type=Path)
        p.add_argument("--out", type=Path)
        if name != "dump":
            p.add_argument(
                "--no-p1",
                action="store_true",
                help="skip the P1 (.p1.json) translation pass",
            )
        if name != "to_kir":
            p.add_argument(
                "--src", help="exported source root (overrides the case's source.src)"
            )
            p.add_argument(
                "--inputs", help="comma-separated input names (default: all)"
            )
    args = ap.parse_args()

    if args.cmd == "export":
        export(args.rev, args.path, args.out, args.repo)
        return
    case_path = args.case.resolve()
    out = args.out or default_out(case_path)
    if args.cmd in ("dump", "run"):
        from . import dump

        dump.run(
            case_path, out, args.src, args.inputs.split(",") if args.inputs else None
        )
    if args.cmd in ("to_kir", "run"):
        from . import to_kir

        for path in to_kir.run(case_path, out, p1=not args.no_p1):
            print(path)


if __name__ == "__main__":
    main()
