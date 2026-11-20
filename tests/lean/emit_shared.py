# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""emit_lean.py's shared terms (run by run_all.sh): an Expr whose Lean text is longer than
SHARE_MIN is one `def sx<n>`, emitted before its first use and named at every use; emits from
several threads at once (scripts/lkv.py checks KIRs in a thread pool) equal the serial ones.

    python3 -I tests/lean/emit_shared.py      exit 0 iff every check passes
"""

import concurrent.futures
import copy
import json
import os
import re
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(HERE, "..", "..", "frontend"))
import emit_lean  # noqa: E402


def fixture(name):
    with open(os.path.join(HERE, f"{name}.kir.json")) as f:
        return json.load(f)


def big(depth):
    """add(...add(n, 1)..., depth), whose text is longer than SHARE_MIN."""
    e = {"e": "var", "w": 32, "name": "n"}
    for i in range(depth):
        e = {
            "e": "bin",
            "w": 32,
            "o": "add",
            "a": e,
            "b": {"e": "const", "w": 32, "v": i + 1},
        }
    return e


def main():
    bad = []
    k = fixture("eval_semantics")
    role = k["roles"][0]
    e = big(12)
    role["body"] = [
        {"op": "let", "var": "s0", "expr": copy.deepcopy(e), "loc": "sh.cu:1"},
        {"op": "let", "var": "s1", "expr": copy.deepcopy(e), "loc": "sh.cu:2"},
    ] + role["body"]
    text = emit_lean.emit(k, "shared")
    defs = re.findall(r"^def (sx\d+) ", text, re.M)
    if len(emit_lean._expr(e)) <= emit_lean.SHARE_MIN:
        bad.append("the test term is not long enough to be shared")
    for n in set(re.findall(r"\b(sx\d+)\b", text)):
        at = text.find(f"def {n} ")
        if at < 0 or at > text.find(f" {n}", at + 1 + len(n) + 4):
            bad.append(f"{n} is used before its definition or never defined")
    lets = re.findall(r'\(\.«let» "s[01]" (.*) "sh\.cu:\d"\)', text)
    if len(lets) != 2 or lets[0] != lets[1] or not re.search(r"\bsx\d+\b", lets[0]):
        bad.append(f"the two equal lets do not name one shared term: {lets}")
    if not defs:
        bad.append("no shared term emitted")
    kirs = [k]
    for n, d in (("eval_semantics", 14), ("distinct_locs", 16)):  # other shared terms
        x = fixture(n)
        x["roles"][0]["body"] = [
            {"op": "let", "var": f"t{j}", "expr": big(d + j), "loc": "sh.cu:3"}
            for j in range(4)
        ] + x["roles"][0]["body"]
        kirs.append(x)
    serial = [emit_lean.emit(x, "t") for x in kirs]
    # every thread enters its first role body only when all three are inside kernel_defs, so
    # their emits interleave
    gate, block, local = threading.Barrier(3), emit_lean.block, threading.local()

    def gated(*a):
        if not getattr(local, "met", False):
            local.met = True
            gate.wait(timeout=60)
        return block(*a)

    emit_lean.block = gated
    try:
        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            par = list(pool.map(lambda i: emit_lean.emit(kirs[i], "t"), range(3)))
    finally:
        emit_lean.block = block
    if par != serial:
        bad.append("emits from several threads differ from the serial ones")
    for b in bad:
        print(f"emit_shared: {b}", file=sys.stderr)
    print(f"emit_shared: {'FAIL' if bad else 'OK'} ({len(defs)} shared term(s))")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
