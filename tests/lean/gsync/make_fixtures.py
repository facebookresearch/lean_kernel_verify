# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Writes the global-synchronization fixtures of tests/lean/gsync (kir.md change 18): tagged
words (a take of a stored tag, of a tag no CTA stores, of tag 0), co-residency, a take that a
named barrier keeps its store from (a hang), claims (feasible, a CTA that stops early, CTAs
that synchronize), claim turns the tagged words may contradict, relay slots (written,
unwritten), a long take chain and a take of data.

    python3 -I tests/lean/gsync/make_fixtures.py     # rewrites tests/lean/gsync/*.kir.json
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def c(v, w=32):
    return {"e": "const", "v": v, "w": w}


def var(n, w=32):
    return {"e": "var", "name": n, "w": w}


def pid():
    return {"e": "pid", "axis": 0, "w": 32}


def cmp(o, a, b):
    return {"e": "cmp", "o": o, "a": a, "b": b, "w": 1}


def add(a, b):
    return {"e": "bin", "o": "add", "a": a, "b": b, "w": 32}


def iff(cond, then, els=None, loc="k.py:1"):
    return {"op": "if", "cond": cond, "then": then, "else": els or [], "loc": loc}


def ev(name, loc, **f):
    return {"op": "event", "ev": name, "loc": loc, **f}


def gstore(t, off, loc):
    return ev("gstore", loc, tensor=t, offsets={"t": "splat", "a": off, "shape": [1], "w": 32},
              mask=None)  # fmt: skip


def store(t, idx, value, loc, data=False):
    return ev(
        "gsig",
        loc,
        tensor=t,
        index=idx,
        value=c(value, 64),
        **({"data_lo": True} if data else {}),
    )


def take(t, idx, tag, loc, var_=None):
    f = {"var": var_, "w": 32} if var_ else {}
    return ev(
        "gwait",
        loc,
        tensor=t,
        index=idx,
        target=tag if isinstance(tag, dict) else c(tag),
        **f,
    )


def claim(t, var_, loc):
    return ev("gclaim", loc, tensor=t, index=c(0), value=c(1), var=var_, w=32)


def tensor(name, n, sync=None, role="workspace"):
    t = {"name": name, "kind": "global", "elem_bytes": 4, "shape": [c(n)], "strides": [c(1)],
         "segments": None, "role": role, "metadata": False}  # fmt: skip
    if sync:
        t.update(sync={"width": sync}, elem_bytes=sync // 8)
    return t


def role(name, body):
    return {"name": name, "warps": 1, "threads_per_issue": 1, "body": body}


def kernel(name, grid, params, tensors, roles, inputs):
    return {"kir_version": 1, "kernel": name, "source": f"tests/lean/gsync/{name}", "ir": "ptx",
            "grid": [c(grid), c(1), c(1)], "params": [{"name": p, "type": "i32"} for p in params],
            "tensors": tensors, "barriers": [], "roles": roles,
            "case": {"name": name, "inputs": inputs}}  # fmt: skip


def inp(name, params=None, **extra):
    return {"name": name, "params": params or {}, **extra}


def pid_is(i):
    return cmp("eq", pid(), c(i))


FIX = {}

# Tagged words: CTA 0 stores tag 1 with value 2, CTA 1 takes tag TAG and stores to out at the
# value it read (PASS); a tag no CTA stores hangs (P0-7 FAIL); tag 0 is the start value's.
FIX["tagged"] = kernel(
    "tagged", 2, ["TAG"], [tensor("mail", 1, sync=64), tensor("out", 4)],
    [role("p", [iff(pid_is(0), [store("mail", c(0), (1 << 32) | 2, "tg.py:10")])]),
     role("c", [iff(pid_is(1), [take("mail", c(0), var("TAG"), "tg.py:20", "v"),
                                gstore("out", var("v"), "tg.py:21")])])],
    [inp("paired", {"TAG": 1}, capacity={"ctas": 2}),
     inp("unstored_tag", {"TAG": 2}, capacity={"ctas": 2}),
     inp("tag0", {"TAG": 0}, capacity={"ctas": 2})],
)  # fmt: skip

# Co-residency: CTA 0 takes the tags CTAs 1..3 store; 4 resident CTAs PASS, 2 FAIL, no
# capacity is undecided.
FIX["coresident"] = kernel(
    "coresident", 4, [], [tensor("mail", 4, sync=64)],
    [role("p", [iff(pid_is(0), [], [store("mail", pid(), 1 << 32, "cr.py:10")])]),
     role("c", [iff(pid_is(0), [take("mail", c(k), 1, "cr.py:20") for k in (1, 2, 3)])])],
    [inp("capacity4", capacity={"ctas": 4}), inp("capacity2", capacity={"ctas": 2}),
     inp("no_capacity")],
)  # fmt: skip
# A named barrier keeps the store from the take: role a stores after a named-barrier sync that
# needs role b's arrival, which b makes only after its take (ORDER 0: a hang, P0-7 FAIL); b
# arrives first (ORDER 1: PASS).
BAR = ev("named_bar", "nb.py:5", id=c(1), count=c(64), sync=True)
FIX["barrier_cycle"] = kernel(
    "barrier_cycle", 1, ["ORDER"], [tensor("mail", 1, sync=64)],
    [role("a", [dict(BAR, loc="nb.py:10"), store("mail", c(0), 1 << 32, "nb.py:11")]),
     role("b", [iff(cmp("eq", var("ORDER"), c(0)),
                    [take("mail", c(0), 1, "nb.py:20"), dict(BAR, loc="nb.py:21")],
                    [dict(BAR, loc="nb.py:22"), take("mail", c(0), 1, "nb.py:23")])])],
    [inp("cycle", {"ORDER": 0}, capacity={"ctas": 1}),
     inp("ordered", {"ORDER": 1}, capacity={"ctas": 1})],
)  # fmt: skip

# Claims: each CTA claims work items from a cursor until it runs past N, storing out at each
# (every item once: PASS); a CTA that stops after one claim (EARLY) skips values of the chain
# a fetch-add hands out: a limit of the model.
LOOP = {"op": "while", "cond": cmp("ne", var("go"), c(0)), "max_iters": 16, "loc": "cl.py:10",
        "body": [claim("ctr", "k", "cl.py:11"),
                 iff(cmp("slt", var("k"), var("N")),
                     [gstore("out", var("k"), "cl.py:12"),
                      iff(cmp("eq", add(pid(), var("EARLY")), c(2)),
                          [{"op": "let", "var": "go", "expr": c(0), "loc": "cl.py:13"}])],
                     [{"op": "let", "var": "go", "expr": c(0), "loc": "cl.py:14"}])]}  # fmt: skip
FIX["claims"] = kernel(
    "claims", 2, ["N", "EARLY"], [tensor("ctr", 1, sync=32), tensor("out", 4, role="output")],
    [role("w", [{"op": "let", "var": "go", "expr": c(1), "loc": "cl.py:9"}, LOOP])],
    [inp("n4", {"N": 4, "EARLY": 0}, intended_writes={"out": [[0, 4]]}, capacity={"ctas": 2}),
     inp("stop_early", {"N": 4, "EARLY": 1}, intended_writes={"out": [[0, 4]]},
         capacity={"ctas": 2}),
     # the two CTAs synchronize outside the model (a protocol region): their turns may not
     # alternate, a limit
     inp("grouped", {"N": 4, "EARLY": 0}, intended_writes={"out": [[0, 4]]},
         capacity={"ctas": 2}, sync_groups=[[0, 1]])],
)  # fmt: skip
FIX["claims"]["representative"] = ["claims"]

# Claim turns: one CTA claims and then stores tag 1, the other takes it and then claims. CTA 1
# storing (ORDER 0) makes CTA 1's claim come first, against the turns (CTA 0 first): a limit;
# CTA 0 storing (ORDER 1) agrees with them (PASS).
FIX["claim_turns"] = kernel(
    "claim_turns", 2, ["ORDER"], [tensor("ctr", 1, sync=32), tensor("mail", 1, sync=64),
                                  tensor("out", 2, role="output")],
    [role("w", [iff(cmp("eq", pid(), var("ORDER")),
                    [claim("ctr", "k", "ct.py:10"), store("mail", c(0), 1 << 32, "ct.py:11"),
                     gstore("out", var("k"), "ct.py:12")],
                    [take("mail", c(0), 1, "ct.py:20"), claim("ctr", "k", "ct.py:21"),
                     gstore("out", var("k"), "ct.py:22")])])],
    [inp("contradicted", {"ORDER": 1}, intended_writes={"out": "all"}, capacity={"ctas": 2}),
     inp("agreed", {"ORDER": 0}, intended_writes={"out": "all"}, capacity={"ctas": 2})],
)  # fmt: skip
FIX["claim_turns"]["representative"] = ["claims"]

# Relay slots: role p claims 3 items and relays each through slot j of channel ring; role c
# reads slot j + OFF and stores out at the value (OFF 0: PASS; OFF 1: slot 3 is never written,
# a limit of the model).
FIX["relay"] = kernel(
    "relay", 1, ["OFF"], [tensor("ctr", 1, sync=32), tensor("out", 4, role="output")],
    [role("p", [{"op": "for", "var": "j", "lb": c(0), "ub": c(3), "step": c(1), "loc": "rl.py:10",
                 "body": [claim("ctr", "k", "rl.py:11"),
                          ev("rput", "rl.py:12", chan="ring", ord=var("j"), field=0, value=var("k"))]}]),
     role("c", [{"op": "for", "var": "j", "lb": c(0), "ub": c(3), "step": c(1), "loc": "rl.py:20",
                 "body": [ev("rget", "rl.py:21", chan="ring", ord=add(var("j"), var("OFF")), field=0,
                             var="v", w=32),
                          gstore("out", var("v"), "rl.py:22")]}])],
    [inp("written", {"OFF": 0}, intended_writes={"out": [[0, 3]]}, capacity={"ctas": 1}),
     inp("unwritten", {"OFF": 1}, intended_writes={"out": [[0, 3]]}, capacity={"ctas": 1})],
)  # fmt: skip
FIX["relay"]["representative"] = ["claims"]

# A take chain longer than a fixed round bound: two CTAs pass tags 1..100 back and forth
# (PASS: every take returns).
K = 100
FIX["pingpong"] = kernel(
    "pingpong", 2, [], [tensor("a", 1, sync=64), tensor("b", 1, sync=64)],
    [role("w", [iff(pid_is(0),
                    [x for j in range(1, K + 1) for x in (store("a", c(0), j << 32, "pp.py:10"),
                                                         take("b", c(0), j, "pp.py:11"))],
                    [x for j in range(1, K + 1) for x in (take("a", c(0), j, "pp.py:20"),
                                                         store("b", c(0), j << 32, "pp.py:21"))])])],
    [inp("long", capacity={"ctas": 2})],
)  # fmt: skip

# A take whose store carries data unbinds its variable, also one an earlier take bound: the use
# after it is a limit of the model (data), not a second store at the earlier value.
FIX["stale"] = kernel(
    "stale", 2, [], [tensor("mail", 1, sync=64), tensor("out", 4, role="output")],
    [role("w", [iff(pid_is(0),
                    [store("mail", c(0), 1 << 32, "st.py:10"),
                     store("mail", c(0), 2 << 32, "st.py:11", data=True)],
                    [take("mail", c(0), 1, "st.py:20", "x"), gstore("out", var("x"), "st.py:21"),
                     take("mail", c(0), 2, "st.py:22", "x"), gstore("out", var("x"), "st.py:23")])])],
    [inp("data", intended_writes={"out": "unchecked"}, capacity={"ctas": 2})],
)  # fmt: skip


def main():
    for name, k in FIX.items():
        (HERE / f"{name}.kir.json").write_text(json.dumps(k, indent=1) + "\n")


if __name__ == "__main__":
    main()
