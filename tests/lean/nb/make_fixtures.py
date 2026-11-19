# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Write the P0-6 KIR fixtures of tests/lean/nb (lean/KV/Spec/NamedBarriers.lean).

    python3 -I tests/lean/nb/make_fixtures.py      (rewrites nb/*.kir.json)

Small hand-written P0-6 KIRs (top-level `nbar`), one grid CTA unless said otherwise,
each with the verdicts nb/expected.txt lists:
  uniform        one role of 4 warps syncs barrier 1 (count 128) in a loop: PASS
  short          warp 0 alone syncs a 128-thread barrier: FAIL (waits forever)
  per_input      the short barrier under `if n > 0`: PASS for n = 0, FAIL for n = 1
  meta_stop      grid 2: each CTA loads metadata at its pid before a barrier: PASS in
                 range; UNDECIDED where CTA 1's load is out of bounds (its trace stops)
  meta_stop_two_roles  the same load in one of two roles that sync one barrier: PASS in
                 range; UNDECIDED, not FAIL, out of range (CTA 1 is not paired: the other
                 role does not wait forever on a trace that stopped)
  pairing        producers arrive and consumers sync, twice, on one barrier: PASS
  forced_pair    roles of 13 and 1 warps: [sync, arrive] / [sync, sync] (FA4's tile
                 barrier): PASS
  count_mismatch two roles sync one barrier with counts 64 and 96: FAIL
  cycle          roles sync barriers 1, 2 and 2, 1: FAIL (a cycle between barriers)
  open_certain   a reached certain nb_open: FAIL
  open_undecided a reached nb_open not certain: UNDECIDED (refuted, not certain)
  open_unreached an nb_open under a false branch: PASS
  rep_private    a repeated region whose barrier only its role uses: PASS
  rep_shared     the same barrier in a repeated region and in another role: UNDECIDED
  rep_keyed      two roles split from one role use a barrier in the same keyed region: PASS
  rep_keyed_two  the same, but the regions' keys differ: UNDECIDED
  arrive_twice   a role arrives twice, the other role syncs once: UNDECIDED (an instance
                 is left incomplete)
  arrive_double  a 1-warp role arrives and then syncs on one 160-thread instance with a
                 3-warp role: PASS (nobody else can complete it in between)
  arrive_ahead   a producer arrives twice before its consumer's first sync: PASS under the
                 warp-ahead reading (the second arrival waits for the next instance)
  held_cycle     the same, but the consumer first waits for the producer elsewhere:
                 UNDECIDED
  rep_outside_inside  two split roles use a barrier before and inside one keyed region: PASS
  rep_mixed      one role's use outside a region pairs with another's inside it: UNDECIDED
  short_arrive_after  the short barrier, then an arrive that never runs: FAIL (the pairing
                 up to the wait that never returns has no arrive)
  arrive_leftover an arrive no operation completes: UNDECIDED
  two_ctas       grid 2: CTA 1 alone takes the short path (`pid.x == 1`): FAIL
  mix_alone      a 2-warp role syncs a 64-thread barrier once, another twice: UNDECIDED
                 (warps of both roles may complete an instance; the pairing let each role
                 complete its own)
  mix_counts     warp 0 syncs barrier 6 with count 32, warps 1-3 with count 96: UNDECIDED
  mix_groups     two 4-warp roles each sync a 128-thread barrier twice: UNDECIDED
  wait_after_rep warp 1 syncs barrier 2 in a repeated region, then both warps barrier 1:
                 UNDECIDED (the region may run no pass), in either role order
                 (wait_after_rep_swapped)
  rep_open_certain  a certain nb_open in a repeated region: UNDECIDED
  rep_bad_count  a count above the CTA's threads in a repeated region: UNDECIDED
  all_exit       warps 0-1 sync barrier 0 without a thread count, warps 2-3 end: PASS
                 (threads that exit release it); all_counted: the same with count 128: FAIL
  all_exit_two   one role syncs barrier 0 without a count twice, the other once: PASS
  all_mixed      operations with and without a count on one instance, the other role
                 ended: UNDECIDED
  all_cycle      barrier 0 without a count waits for a role blocked on barrier 1: FAIL
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def c32(v):
    return {"e": "const", "w": 32, "v": v}


def bar(i, n, sync=True, line=1, every=False):
    ev = {"op": "event", "ev": "named_bar", "id": c32(i), "count": c32(n), "sync": sync,
          "loc": f"nb.py:{line}"}  # fmt: skip
    if every:  # no thread count: all threads of the CTA
        ev["all"] = True
    return ev


def nb_open(certain, line=1, kind="cond"):
    return {"op": "event", "ev": "nb_open", "kind": kind, "certain": certain,
            "why": f"bar.sync 1, 128 at PTX line {line}: inside a branch on data",
            "id": c32(1), "count": c32(128), "sync": True, "loc": f"nb.py:{line}"}  # fmt: skip


def rep(begin, line=1, key=None):
    ev = {"op": "event", "ev": "nb_rep", "begin": begin, "why": "a loop on uniform data",
          "loc": f"nb.py:{line}"}  # fmt: skip
    if key is not None:
        ev["key"] = key
    return ev


def role(name, warps, body):
    return {"name": name, "warps": warps, "threads_per_issue": 1, "body": body}


def loop(n, body, var="i"):
    return {"op": "for", "var": var, "lb": c32(0), "ub": c32(n), "step": c32(1),
            "body": body, "loc": "nb.py:2"}  # fmt: skip


def kir(name, roles, inputs=None, params=(), grid=(1, 1, 1)):
    inputs = inputs or [{"name": "one", "params": {}}]
    return {
        "kir_version": 1,
        "kernel": name,
        "source": "nb.py",
        "ir": "ptx",
        "grid": [c32(g) for g in grid],
        "params": [{"name": p, "type": "i32"} for p in params],
        "tensors": [],
        "barriers": [],
        "roles": roles,
        "nbar": {"sites": []},
        "case": {"name": f"nb/{name}", "inputs": inputs},
    }


def gt0(var):
    return {"e": "cmp", "w": 1, "o": "sgt", "a": {"e": "var", "w": 32, "name": var},
            "b": c32(0)}  # fmt: skip


# A load of metadata tensor `meta` at the CTA's pid, in bounds on input in_range only.
META_LOAD = {"op": "let", "var": "v", "loc": "nb.py:5",
             "expr": {"e": "load", "tensor": "meta", "w": 32,
                      "index": [{"e": "pid", "axis": 0, "w": 32}]}}  # fmt: skip
META_INPUTS = [
    {"name": "in_range", "params": {"N_META": 2}, "metadata": {"meta": [3, 5]}},
    {"name": "out_of_range", "params": {"N_META": 1}, "metadata": {"meta": [3]}},
]
META_TENSOR = {"name": "meta", "kind": "global", "elem_bytes": 4,
               "shape": [{"e": "var", "name": "N_META", "w": 32}], "strides": [c32(1)],
               "segments": None, "role": "input", "metadata": True}  # fmt: skip

FIXTURES = {
    "uniform": kir("uniform", [role("all", 4, [loop(3, [bar(1, 128)])])]),
    "short": kir("short", [role("w0", 1, [bar(1, 128, line=7)]), role("w1_3", 3, [])]),
    "per_input": kir(
        "per_input",
        [
            role(
                "w0",
                1,
                [
                    {
                        "op": "if",
                        "cond": gt0("n"),
                        "then": [bar(1, 128, line=7)],
                        "else": [],
                        "loc": "nb.py:6",
                    }
                ],
            ),  # fmt: skip
            role("w1_3", 3, []),
        ],  # fmt: skip
        [{"name": "n0", "params": {"n": 0}}, {"name": "n1", "params": {"n": 1}}],
        params=("n",),
    ),
    "meta_stop": dict(
        kir(
            "meta_stop",
            [
                role(
                    "all",
                    4,
                    [
                        {
                            "op": "let",
                            "var": "v",
                            "loc": "nb.py:5",
                            "expr": {
                                "e": "load",
                                "tensor": "meta",
                                "w": 32,
                                "index": [{"e": "pid", "axis": 0, "w": 32}],
                            },
                        },
                        bar(1, 128, line=6),
                    ],
                )
            ],  # fmt: skip
            [
                {
                    "name": "in_range",
                    "params": {"N_META": 2},
                    "metadata": {"meta": [3, 5]},
                },
                {
                    "name": "out_of_range",
                    "params": {"N_META": 1},
                    "metadata": {"meta": [3]},
                },
            ],
            params=("N_META",),
            grid=(2, 1, 1),
        ),  # fmt: skip
        tensors=[
            {
                "name": "meta",
                "kind": "global",
                "elem_bytes": 4,
                "shape": [{"e": "var", "name": "N_META", "w": 32}],
                "strides": [c32(1)],
                "segments": None,
                "role": "input",
                "metadata": True,
            }
        ],  # fmt: skip
    ),
    "meta_stop_two_roles": dict(
        kir(
            "meta_stop_two_roles",
            [
                role("a", 2, [META_LOAD, bar(1, 128, line=6)]),
                role("b", 2, [bar(1, 128, line=7)]),
            ],
            META_INPUTS,
            params=("N_META",),
            grid=(2, 1, 1),
        ),
        tensors=[META_TENSOR],
    ),
    "pairing": kir(
        "pairing",
        [
            role("prod", 2, [bar(1, 128, False), bar(1, 128, False)]),
            role("cons", 2, [bar(1, 128), bar(1, 128)]),
        ],
    ),  # fmt: skip
    "forced_pair": kir(
        "forced_pair",
        [
            role(
                "w0_11", 12, [bar(5, 416), loop(4, [bar(4, 384)]), bar(5, 416, False)]
            ),
            role("w12", 1, [bar(5, 416), bar(5, 416)]),
        ],
    ),  # fmt: skip
    "count_mismatch": kir(
        "count_mismatch",
        [
            role("a", 1, [bar(1, 64)]),
            role("b", 1, [bar(1, 96, line=4)]),
            role("c", 2, []),
        ],
    ),  # fmt: skip
    "cycle": kir(
        "cycle",
        [
            role("a", 2, [bar(1, 128), bar(2, 128)]),
            role("b", 2, [bar(2, 128), bar(1, 128)]),
        ],
    ),  # fmt: skip
    "open_certain": kir("open_certain", [role("all", 4, [nb_open(True, 9)])]),
    "open_undecided": kir("open_undecided", [role("all", 4, [nb_open(False, 9)])]),
    "open_unreached": kir(
        "open_unreached",
        [
            role(
                "all",
                4,
                [
                    {
                        "op": "if",
                        "cond": gt0("n"),
                        "then": [nb_open(True, 9)],
                        "else": [],
                        "loc": "nb.py:8",
                    }
                ],
            )
        ],  # fmt: skip
        [{"name": "n0", "params": {"n": 0}}],
        params=("n",),
    ),
    "rep_private": kir(
        "rep_private",
        [
            role("a", 4, [rep(True), bar(1, 128), rep(False)]),
            role("b", 2, [bar(2, 64)]),
        ],
    ),  # fmt: skip
    "rep_shared": kir(
        "rep_shared",
        [
            role("a", 2, [rep(True), bar(1, 128, line=5), rep(False)]),
            role("b", 2, [bar(1, 128)]),
        ],
    ),  # fmt: skip
    "rep_keyed": kir(
        "rep_keyed",
        [
            role("a", 1, [rep(True, key="r@4"), bar(1, 128), rep(False, key="r@4")]),
            role(
                "b",
                3,
                [rep(True, key="r@4"), bar(1, 128, line=5), rep(False, key="r@4")],
            ),
        ],
    ),  # fmt: skip
    "rep_keyed_two": kir(
        "rep_keyed_two",
        [
            role("a", 1, [rep(True, key="r@4"), bar(1, 128), rep(False, key="r@4")]),
            role(
                "b",
                3,
                [rep(True, key="r@9"), bar(1, 128, line=5), rep(False, key="r@9")],
            ),
        ],
    ),  # fmt: skip
    "arrive_twice": kir(
        "arrive_twice",
        [
            role("a", 2, [bar(1, 128, False), bar(1, 128, False, 6)]),
            role("b", 2, [bar(2, 64), bar(1, 128)]),
        ],
    ),  # fmt: skip
    "arrive_double": kir(
        "arrive_double",
        [
            role("a", 1, [bar(1, 160, False), bar(1, 160, line=3)]),
            role("b", 3, [bar(2, 96), bar(1, 160, line=4)]),
            role("c", 1, []),
        ],
    ),  # fmt: skip
    "arrive_ahead": kir(
        "arrive_ahead",
        [
            role("p", 1, [bar(1, 64, False), bar(1, 64, False, 3)]),
            role("c", 1, [bar(2, 32), bar(1, 64, line=5), bar(1, 64, line=6)]),
        ],
    ),  # fmt: skip
    "held_cycle": kir(
        "held_cycle",
        [
            role("p", 1, [bar(1, 64, False), bar(1, 64, False, 3), bar(2, 64, line=4)]),
            role("c", 1, [bar(2, 64, line=5), bar(1, 64, line=6)]),
        ],
    ),  # fmt: skip
    "rep_outside_inside": kir(
        "rep_outside_inside",
        [
            role(
                "a",
                1,
                [bar(1, 128), rep(True, key="r@4"), bar(1, 128), rep(False, key="r@4")],
            ),
            role(
                "b",
                3,
                [
                    bar(1, 128),
                    rep(True, key="r@4"),
                    bar(1, 128, line=5),
                    rep(False, key="r@4"),
                ],
            ),
        ],
    ),  # fmt: skip
    "rep_mixed": kir(
        "rep_mixed",
        [
            role("a", 1, [bar(1, 128, line=2)]),
            role(
                "b",
                3,
                [rep(True, key="r@4"), bar(1, 128, line=5), rep(False, key="r@4")],
            ),
        ],
    ),  # fmt: skip
    "short_arrive_after": kir(
        "short_arrive_after",
        [
            role("w0", 1, [bar(1, 128, line=7), bar(2, 32, False, 8)]),
            role("w1_3", 3, []),
        ],
    ),  # fmt: skip
    "arrive_leftover": kir(
        "arrive_leftover", [role("a", 2, [bar(1, 128, False, 3)]), role("b", 2, [])]
    ),  # fmt: skip
    "two_ctas": kir(
        "two_ctas",
        [
            role(
                "w0",
                1,
                [
                    {
                        "op": "if",
                        "cond": {
                            "e": "cmp",
                            "w": 1,
                            "o": "eq",
                            "a": {"e": "pid", "w": 32, "axis": 0},
                            "b": c32(1),
                        },
                        "then": [bar(1, 128, line=7)],
                        "else": [bar(1, 32)],
                        "loc": "nb.py:6",
                    }
                ],
            ),
            role("w1_3", 3, []),
        ],  # fmt: skip
        grid=(2, 1, 1),
    ),
    "mix_alone": kir(
        "mix_alone",
        [
            role("w0_1", 2, [bar(5, 64, line=3)]),
            role("w2_3", 2, [bar(5, 64, line=4), bar(5, 64, line=5)]),
        ],
    ),  # fmt: skip
    "mix_counts": kir(
        "mix_counts",
        [role("w0", 1, [bar(6, 32, line=3)]), role("w1_3", 3, [bar(6, 96)])],
    ),  # fmt: skip
    "mix_groups": kir(
        "mix_groups",
        [
            role("g0", 4, [bar(2, 128, line=3), bar(2, 128)]),
            role("g1", 4, [bar(2, 128), bar(2, 128)]),
        ],
    ),  # fmt: skip
    "wait_after_rep": kir(
        "wait_after_rep",
        [
            role("w0", 1, [bar(1, 64, line=3)]),
            role(
                "w1", 1, [rep(True), bar(2, 64, line=4), rep(False), bar(1, 64, line=5)]
            ),
        ],
    ),  # fmt: skip
    "wait_after_rep_swapped": kir(
        "wait_after_rep_swapped",
        [
            role(
                "w1", 1, [rep(True), bar(2, 64, line=4), rep(False), bar(1, 64, line=5)]
            ),
            role("w0", 1, [bar(1, 64, line=3)]),
        ],
    ),  # fmt: skip
    "rep_open_certain": kir(
        "rep_open_certain", [role("all", 4, [rep(True), nb_open(True, 9), rep(False)])]
    ),  # fmt: skip
    "rep_bad_count": kir(
        "rep_bad_count", [role("all", 4, [rep(True), bar(1, 256, line=6), rep(False)])]
    ),  # fmt: skip
    "all_exit": kir(
        "all_exit",
        [role("w0_1", 2, [bar(0, 128, line=4, every=True)]), role("w2_3", 2, [])],
    ),  # fmt: skip
    "all_counted": kir(
        "all_counted", [role("w0_1", 2, [bar(0, 128, line=4)]), role("w2_3", 2, [])]
    ),  # fmt: skip
    "all_exit_two": kir(
        "all_exit_two",
        [
            role("a", 2, [bar(0, 128, every=True), bar(0, 128, line=3, every=True)]),
            role("b", 2, [bar(0, 128, every=True)]),
        ],
    ),  # fmt: skip
    "all_mixed": kir(
        "all_mixed",
        [
            role("a", 2, [bar(0, 128, line=3, every=True)]),
            role("b", 1, [bar(0, 128, line=4)]),
            role("c", 1, []),
        ],
    ),  # fmt: skip
    "all_cycle": kir(
        "all_cycle",
        [
            role("a", 2, [bar(0, 128, line=3, every=True)]),
            role("b", 2, [bar(1, 128, line=4), bar(0, 128, every=True)]),
        ],
    ),  # fmt: skip
}


def main():
    for old in HERE.glob("*.kir.json"):
        old.unlink()
    for name, k in FIXTURES.items():
        (HERE / f"{name}.kir.json").write_text(json.dumps(k, indent=1) + "\n")


if __name__ == "__main__":
    main()
