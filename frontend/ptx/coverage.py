# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Op table coverage of PTX files: parse each file and classify every instruction.

    python3 -I frontend/ptx/coverage.py PTX_FILE... [--json OUT]

Prints, per file and in total, the instruction count, the number of distinct normalized
mnemonics, the count per class, and every parse error and mnemonic without an op table
row (ops.UnknownInstruction); the total also lists each mnemonic with its classes. Exit
status 1 if any file has a parse error or an unknown mnemonic: the coverage gate is
"every mnemonic in the PTX of every case has a row".
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):  # python3 -I path/to/coverage.py
    sys.dont_write_bytecode = True  # keep the source tree free of __pycache__
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "ptx"  # noqa: A001

from . import ops, parse  # noqa: E402


def file_coverage(path):
    """{"path", "instructions", "classes", "mnemonics": {m: {class: n}}, "unknown":
    {m: {"count", "line", "opcode"}}, "error"}."""
    out = {"path": str(path), "instructions": 0, "classes": {}, "mnemonics": {}}
    out.update(unknown={}, error=None)
    try:
        module = parse.parse_file(path)
    except Exception as e:  # any failure to read the file is a finding, not a crash
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    classes, mnemonics = Counter(), {}
    for fn in module.functions:
        for instr in fn.instructions():
            m = ops.mnemonic(instr.opcode)
            try:
                cls = ops.classify(instr)[0]
            except ops.UnknownInstruction:
                u = out["unknown"].setdefault(
                    m, {"count": 0, "line": instr.line, "opcode": instr.opcode}
                )
                u["count"] += 1
                cls = "unknown"
            classes[cls] += 1
            mnemonics.setdefault(m, Counter())[cls] += 1
    out["instructions"] = sum(classes.values())
    out["classes"] = dict(classes)
    out["mnemonics"] = {m: dict(c) for m, c in sorted(mnemonics.items())}
    return out


def _summary(c):
    per_class = ", ".join(f"{k} {c['classes'].get(k, 0)}" for k in ops.CLASSES)
    unknown = c["classes"].get("unknown", 0)
    return (
        f"{c['instructions']} instructions, {len(c['mnemonics'])} mnemonics; {per_class}"
        + (f", unknown {unknown}" if unknown else "")
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ptx", nargs="+", help="PTX files")
    ap.add_argument("--json", help="write the coverage report here")
    args = ap.parse_args(argv)

    files = [file_coverage(p) for p in args.ptx]
    total = {"files": len(files), "instructions": 0, "classes": Counter()}
    total.update(mnemonics={}, errors=0, unknown={})
    for c in files:
        print(f"== {c['path']}: {_summary(c)}")
        if c["error"]:
            print(f"   parse error: {c['error']}")
            total["errors"] += 1
        for m, u in c["unknown"].items():
            print(f"   unknown: {m} x{u['count']} (line {u['line']}: {u['opcode']})")
            total["unknown"].setdefault(m, f"{c['path']}:{u['line']}: {u['opcode']}")
        total["instructions"] += c["instructions"]
        total["classes"].update(c["classes"])
        for m, per in c["mnemonics"].items():
            total["mnemonics"].setdefault(m, Counter()).update(per)
    total["classes"] = dict(total["classes"])
    total["mnemonics"] = {m: dict(c) for m, c in sorted(total["mnemonics"].items())}

    print(f"== total ({len(files)} files): {_summary(total)}")
    print(
        f"   parse errors {total['errors']}, unknown mnemonics {len(total['unknown'])}"
    )
    order = {k: i for i, k in enumerate(ops.CLASSES + ("unknown",))}
    for m, per in sorted(
        total["mnemonics"].items(),
        key=lambda x: (order[min(x[1], key=order.get)], x[0]),
    ):
        print(f"   {' '.join(f'{k} {n}' for k, n in per.items()):>16}  {m}")
    if args.json:
        Path(args.json).write_text(
            json.dumps({"files": files, "total": total}, indent=1)
        )
    return 1 if total["errors"] or total["unknown"] else 0


if __name__ == "__main__":
    sys.exit(main())
