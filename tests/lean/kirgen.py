# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""KIR builders shared by the fixture generators tests/lean/p1/make_fixtures.py,
p1_2/make_fixtures.py, nb/make_fixtures.py and gsync/make_fixtures.py: expressions,
statements, events, roles and inputs as references/kir.md writes them; `write` writes a generator's fixtures."""

import json
import sys
from pathlib import Path


def c(v, w=32):
    return {"e": "const", "v": v, "w": w}


def var(n, w=32):
    return {"e": "var", "name": n, "w": w}


def b(o, x, y, w=32):
    return {"e": "bin", "o": o, "a": x, "b": y, "w": w}


def cmp(o, x, y):
    return {"e": "cmp", "o": o, "a": x, "b": y, "w": 1}


def ev(name, loc, **f):
    return {"op": "event", "ev": name, "loc": loc, **f}


def for_(v, lb, ub, body, loc):
    return {
        "op": "for",
        "var": v,
        "lb": lb,
        "ub": ub,
        "step": c(1),
        "body": body,
        "loc": loc,
    }


def if_(cond, then, els, loc):
    return {"op": "if", "cond": cond, "then": then, "else": els, "loc": loc}


def mode_is(*ks):
    out = cmp("eq", var("MODE"), c(ks[0]))
    for k in ks[1:]:
        out = b("or", out, cmp("eq", var("MODE"), c(k)), 1)
    return out


def nbar(i, count, loc, sync=True):
    return ev("named_bar", loc, id=c(i), count=c(count), sync=sync)


def wait(bar, parity, loc):
    return ev("wait", loc, bar=bar, idx=c(0), parity=parity)


def arrive(bar, loc, per_warp=False):
    e = ev("arrive", loc, bar=bar, idx=c(0))
    if per_warp:
        e["per_warp"] = True
    return e


def role(name, warps, body):
    return {"name": name, "warps": warps, "threads_per_issue": 1, "body": body}


def inp(name, grid=None, **params):
    out = {"name": name, "params": params}
    if grid is not None:
        out["grid"] = [grid, 1, 1]
    return out


def write(fixtures):
    """Writes every fixture (name -> KIR) as OUT_DIR/<name>.kir.json, OUT_DIR the command
    line's argument, in place of the KIR files there (tests/lean/run_all.sh: $LKV_TEST_OUT/gen/
    <generator's directory>)."""
    if len(sys.argv) != 2:
        sys.exit(f"usage: python3 -I {sys.argv[0]} OUT_DIR")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.kir.json"):
        old.unlink()
    for name, k in fixtures.items():
        (out / f"{name}.kir.json").write_text(json.dumps(k, indent=1) + "\n")
