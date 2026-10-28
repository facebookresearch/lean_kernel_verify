# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Export kernel sources at a revision, with the first-party modules they import.

    python3 -I export_src.py --rev REV --out /tmp/lkv_src/base fbcode/<path>/kernel.py ...

Each file is written to <out>/<path relative to fbcode>, so the export directory can be put
on sys.path and the original import names resolve. Imports are followed transitively
(module-level and function-level, absolute and relative); a module is first-party when
`fbcode/<mod>.py` or
`fbcode/<mod>/__init__.py` exists at the revision, and the `__init__.py` of every parent
package is exported with it. Standard-library names and the runtime packages (the compiler
and torch, which always come from the running interpreter) are skipped. Only read-only
`sl cat` is used.
"""

import argparse
import ast
import subprocess
import sys
from pathlib import Path

# Provided by the analysis binary, never by the export: triton is the compiler under trust.
RUNTIME_PACKAGES = {"triton", "torch"}


def sl_cat(rev: str, path: str) -> str | None:
    """File contents at `rev` (path relative to the repo root), or None if absent."""
    r = subprocess.run(
        ["sl", "cat", "-r", rev, f"path:{path}"], capture_output=True, text=True
    )
    return r.stdout if r.returncode == 0 else None


def imported_modules(src: str, package: str) -> set[str]:
    """Every dotted name an import statement may resolve to (module or module.attr).

    Relative imports are resolved against `package`, the importing module's package."""
    names = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package.split(".") if node.level else []
            if node.level > 1:
                base = base[: 1 - node.level]
            mod = ".".join(base + ([node.module] if node.module else []))
            if mod:
                names.add(mod)
                names.update(f"{mod}.{a.name}" for a in node.names)
    return names


def package_of(rel: str) -> str:
    """Package of the module at `rel` (path below fbcode/): its directory, dotted."""
    return rel.rpartition("/")[0].replace("/", ".")


def export(rev: str, out: Path, roots: list[str]) -> list[str]:
    todo = [r.removeprefix("fbcode/") for r in roots]
    seen: set[str] = set()
    written = []
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        src = sl_cat(rev, f"fbcode/{rel}")
        if src is None:
            if rel in (r.removeprefix("fbcode/") for r in roots):
                sys.exit(f"export_src: fbcode/{rel} does not exist at {rev}")
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src)
        written.append(rel)
        for mod in imported_modules(src, package_of(rel)):
            parts = mod.split(".")
            if parts[0] in sys.stdlib_module_names or parts[0] in RUNTIME_PACKAGES:
                continue
            todo += [
                f"{'/'.join(parts[:i])}/__init__.py" for i in range(1, len(parts) + 1)
            ]
            todo.append(f"{'/'.join(parts)}.py")
    return sorted(written)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rev", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("roots", nargs="+", help="fbcode/... source files to export")
    args = ap.parse_args()
    for rel in export(args.rev, args.out, args.roots):
        print(args.out / rel)


if __name__ == "__main__":
    main()
