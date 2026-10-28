# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Writes the P1-1 Lean fixtures tests/lean/p1/*.kir.json (P1 KIR, references/kir.md
change 10). Each fixture is a small hand-made kernel; its inputs select the variant
(param MODE) and the number of m-blocks (param M). Expected verdicts:
tests/lean/p1/expected.txt; run by tests/lean/run_all.sh.

    python3 -I tests/lean/p1/make_fixtures.py      # rewrites the fixtures in place

The fixtures model the TMEM protocol of a block-scaled attention backward: an MMA role
(one warp) writes the S accumulator (cols [0, 128)) with a block-scaled MMA, commits
S_full, and in m-block m >= 1 waits for compute's dS of m-block m-1, copies dQ scale
factors into cols [80, 84) of the S region (tcgen05.cp) and runs the dQ MMA; a compute
role (warps 2..3, lane quarters 2 and 3) waits S_full, loads cols [64, 96) of S, stores P
to cols [0, 8), and arrives dS_full and S_empty.

A role of several warps arrives once per warp (`per_warp`, the elect-one-per-warp
pattern; count = its warps), as the CuTe P1 pass emits it: it splits a role whose warps do
not all issue an mbarrier event (interp.Translator._p1_split_by_issuers; fixture
`partial_issue` has the split form), and frontend/kir.py rejects an arrive with fewer
arrivals than warps.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def c(v, w=32):
    return {"e": "const", "v": v, "w": w}


def var(n, w=32):
    return {"e": "var", "name": n, "w": w}


def b(o, x, y, w=32):
    return {"e": "bin", "o": o, "a": x, "b": y, "w": w}


def cmp(o, x, y):
    return {"e": "cmp", "o": o, "a": x, "b": y, "w": 1}


def ev(kind, loc, **f):
    return {"op": "event", "ev": kind, "loc": loc, **f}


def wait(bar, parity, loc, idx=None):
    return ev("wait", loc, bar=bar, idx=idx or c(0), parity=parity)


def arrive(bar, loc, idx=None, per_warp=False):
    e = ev("arrive", loc, bar=bar, idx=idx or c(0))
    if per_warp:
        e["per_warp"] = True
    return e


def commit(bar, loc, idx=None):
    return ev("commit_arrive", loc, bar=bar, idx=idx or c(0))


def rect(use, write, lane, nlanes, col, ncols):
    return {
        "use": use,
        "write": write,
        "lane": lane,
        "nlanes": nlanes,
        "col": col,
        "ncols": ncols,
    }


def tmem(inst, warp, rects, loc, site=""):
    e = ev("tmem", loc, inst=inst, warp=warp, rects=rects)
    if site:
        e["site"] = site
    return e


def twait(ld, loc):
    return ev("tmem_wait", loc, ld=ld)


def nbar(i, count, sync, loc):
    return ev("named_bar", loc, id=c(i), count=c(count), sync=sync)


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


def mode_is(k):
    return cmp("eq", var("MODE"), c(k))


def parity(e):
    return b("and", e, c(1))


def quarter(w):
    """First lane of warp w's lane quarter: 32 * (w % 4)."""
    return b("mul", b("urem", w, c(4)), c(32))


ALL = c(0)  # lane 0: a 128-lane rectangle starts there


def mma(d_col, loc, sf=None, a=None, ncols=128, site=""):
    rs = [rect("d", True, ALL, 128, c(d_col), ncols)]
    if a is not None:
        rs.append(rect("a", False, ALL, 128, c(a[0]), a[1]))
    if sf is not None:
        rs += [
            rect("sf", False, ALL, 128, c(sf[0]), 4),
            rect("sf", False, ALL, 128, c(sf[1]), 4),
        ]
    return tmem("mma", c(0), rs, loc, site)


def cp(col, loc):
    return tmem(
        "cp",
        c(0),
        [rect("data", True, ALL, 128, c(col) if isinstance(col, int) else col, 4)],
        loc,
    )


def per_warp(inst, col, ncols, loc, write, w="w", lo=2, hi=4, lane=None):
    return for_(
        w,
        c(lo),
        c(hi),
        [
            tmem(
                inst,
                var(w),
                [rect("data", write, lane or quarter(var(w)), 32, c(col), ncols)],
                loc,
            )
        ],
        loc,
    )


def kernel(
    name, roles, barriers, inputs, params=("M", "MODE"), tmem_info=None, tensors=None
):
    return {
        "kir_version": 1,
        "kernel": name,
        "source": f"tests/lean/p1/{name}.kir.json",
        "ir": "cute-mlir",
        "grid": [c(1), c(1), c(1)],
        "params": [{"name": p, "type": "i32"} for p in params],
        "tensors": tensors or [],
        "barriers": [{"name": n, "count": c(k), "num": c(1)} for n, k in barriers],
        "roles": roles,
        "tmem": tmem_info or {"cols": 512, "base": "zero"},
        "case": {"name": name, "inputs": inputs},
    }


def role(name, warps, body):
    return {"name": name, "warps": warps, "threads_per_issue": 1, "body": body}


def inp(name, **params):
    return {"name": name, "params": params}


# ----------------------------------------------------------------------- the bug


def s2t(name, drain_mode):
    """drain_mode: None = the parent (no S_drain); "fix" = compute arrives S_drain after
    the wait::ld and the MMA waits it before the copy; "early" = the arrive precedes the
    wait::ld; "nowait" = the arrive is there but the MMA does not wait."""
    f = f"{name}.py"
    m = var("m")
    mma_body = [
        if_(
            cmp("sge", m, c(1)),
            [wait("S_empty", parity(b("sub", m, c(1))), f"{f}:10")],
            [],
            f"{f}:10",
        ),
        mma(0, f"{f}:11", sf=(400, 404), site=f"{f}:11"),
        commit("S_full", f"{f}:12"),
        if_(
            cmp("sge", m, c(1)),
            [
                wait("dS_full", parity(b("sub", m, c(1))), f"{f}:13"),
                *(
                    [wait("S_drain", parity(m), f"{f}:14")]
                    if drain_mode == "fix" or drain_mode == "early"
                    else []
                ),
                cp(80, f"{f}:15"),
                mma(256, f"{f}:16", sf=(80, 80), site=f"{f}:16"),
                commit("dQ_full", f"{f}:17"),
            ],
            [],
            f"{f}:13",
        ),
    ]
    drain = [arrive("S_drain", f"{f}:24", per_warp=True)] if drain_mode else []
    comp_body = [
        wait("S_full", parity(m), f"{f}:20"),
        per_warp("ld", 64, 32, f"{f}:21", False),
        *(drain if drain_mode == "early" else []),
        twait(True, f"{f}:22"),
        *(drain if drain_mode in ("fix", "nowait") else []),
        per_warp("st", 0, 8, f"{f}:25", True),
        twait(False, f"{f}:26"),
        arrive("dS_full", f"{f}:27", per_warp=True),
        arrive("S_empty", f"{f}:28", per_warp=True),
    ]
    roles = [
        role("mma", 1, [for_("m", c(0), var("M"), mma_body, f"{f}:9")]),
        role("compute", 2, [for_("m", c(0), var("M"), comp_body, f"{f}:19")]),
    ]
    bars = [("S_full", 1), ("S_empty", 2), ("dS_full", 2), ("dQ_full", 1)]
    if drain_mode:
        bars.append(("S_drain", 2))
    ins = [
        inp("one_mblock", M=1, MODE=0),
        inp("two_mblocks", M=2, MODE=0),
        inp("three_mblocks", M=3, MODE=0),
    ]
    return kernel(name, roles, bars, ins)


# ------------------------------------------------------------ completion rules


def handoff():
    """A writer role copies into cols [0, 4) and signals a reader role, which then loads
    cols [0, 32) in lane quarter 0. MODE: 0 commit then wait (PASS); 1 plain arrive after
    the commit (FAIL: a plain arrive does not complete the cp); 2 a slot with both a commit
    and a plain arrive of the writer (FAIL: no landing credit); 3 wait::st after the cp, then
    a plain arrive (FAIL: wait::st does not complete a cp); 4 the reader loads without a
    wait::ld before releasing, and the writer copies again after the release (FAIL: the
    load never completes); 5 as 4 with the wait::ld (PASS)."""
    f = "handoff.py"
    w_body = [
        cp(0, f"{f}:10"),
        if_(mode_is(0), [commit("ready", f"{f}:11")], [], f"{f}:11"),
        if_(
            mode_is(1),
            [commit("other", f"{f}:12"), arrive("ready", f"{f}:12")],
            [],
            f"{f}:12",
        ),
        if_(
            mode_is(2),
            [commit("ready", f"{f}:13"), arrive("ready", f"{f}:13")],
            [],
            f"{f}:13",
        ),
        if_(
            mode_is(3),
            [twait(False, f"{f}:14"), arrive("ready", f"{f}:14")],
            [],
            f"{f}:14",
        ),
        if_(
            cmp("sge", var("MODE"), c(4)),
            [
                commit("ready", f"{f}:15"),
                wait("released", c(0), f"{f}:16"),
                cp(0, f"{f}:17"),
            ],
            [],
            f"{f}:15",
        ),
    ]
    r_body = [
        # MODE 2: the slot completes at 2 arrivals (count 2)
        wait("ready", c(0), f"{f}:20"),
        tmem("ld", c(4), [rect("data", False, c(0), 32, c(0), 32)], f"{f}:21"),
        if_(cmp("ne", var("MODE"), c(4)), [twait(True, f"{f}:22")], [], f"{f}:22"),
        if_(
            cmp("sge", var("MODE"), c(4)),
            [arrive("released", f"{f}:23")],
            [],
            f"{f}:23",
        ),
    ]
    roles = [role("writer", 1, w_body), role("reader", 1, r_body)]
    ins = [
        inp("commit_then_wait", M=0, MODE=0),
        inp("plain_arrive_after_commit", M=0, MODE=1),
        inp("mixed_commit_slot", M=0, MODE=2),
        inp("wait_st_after_cp", M=0, MODE=3),
        inp("ld_without_wait", M=0, MODE=4),
        inp("ld_with_wait", M=0, MODE=5),
    ]
    k = kernel("handoff", roles, [("ready", 1), ("other", 1), ("released", 1)], ins)
    return k


def handoff_mixed():
    """A count-2 slot: the writer commits after its copy and then arrives once more,
    plainly (MODE 0) or by a second commit (MODE 1); the reader waits for the phase and
    loads the copied columns. MODE 0 FAILs: the slot is not commit-only for the writer, so
    the rule gives no landing (conservative: here the commit is one of the two needed
    arrivals); MODE 1 PASSes (the 2nd commit landed)."""
    f = "handoff_mixed.py"
    w_body = [
        cp(0, f"{f}:10"),
        commit("ready", f"{f}:11"),
        if_(
            mode_is(0),
            [arrive("ready", f"{f}:12")],
            [commit("ready", f"{f}:12")],
            f"{f}:12",
        ),
    ]
    r_body = [
        wait("ready", c(0), f"{f}:20"),
        tmem("ld", c(1), [rect("data", False, c(32), 32, c(0), 32)], f"{f}:21"),
        twait(True, f"{f}:22"),
    ]
    roles = [role("writer", 1, w_body), role("reader", 1, r_body)]
    return kernel(
        "handoff_mixed",
        roles,
        [("ready", 2)],
        [inp("commit_and_plain", M=0, MODE=0), inp("two_commits", M=0, MODE=1)],
    )


# --------------------------------------------------------------- happens-before


def split_drain():
    """A split S_drain shape: two compute roles (1 and 3 warps) arrive on one count-4
    slot, one arrival per warp per m-block, after draining their loads of cols [64, 96).
    The MMA warp copies into cols [80, 84) once per m-block m, after waiting for phase m of
    the drain (the loads of m-blocks before m) and before committing `go` (compute's loads
    of m-block m wait for it). Needs from the run totals (4m minus the other role's arrivals
    over the whole run) cannot attribute phase m to the 1-warp role; `needsWith` subtracts
    only the other role's arrivals that may precede the drain wait's return (those before
    its `go` wait of m-block m, which needs a commit after the copy), so sound
    happens-before orders every earlier load before the copy: PASS, no lockstep credit, no
    notes."""
    f = "split_drain.py"
    m = var("m")
    mma_body = [
        if_(
            cmp("sge", m, c(1)),
            [wait("drain", parity(b("sub", m, c(1))), f"{f}:10")],
            [],
            f"{f}:10",
        ),
        cp(80, f"{f}:11"),
        commit("go", f"{f}:12"),
    ]

    def comp(lo, hi, line):
        return [
            for_(
                "m",
                c(0),
                var("M"),
                [
                    wait("go", parity(m), f"{f}:{line}"),
                    per_warp("ld", 64, 32, f"{f}:{line + 1}", False, lo=lo, hi=hi),
                    twait(True, f"{f}:{line + 2}"),
                    arrive("drain", f"{f}:{line + 3}", per_warp=True),
                ],
                f"{f}:{line}",
            )
        ]

    roles = [
        role("mma", 1, [for_("m", c(0), var("M"), mma_body, f"{f}:9")]),
        role("comp_a", 1, comp(4, 5, 20)),
        role("comp_b", 3, comp(5, 8, 25)),
    ]
    # comp_a's warp 4 covers lane quarter 0, comp_b's warps 5..7 quarters 1..3
    return kernel(
        "split_drain",
        roles,
        [("go", 1), ("drain", 4)],
        [inp("two_mblocks", M=2, MODE=0), inp("three_mblocks", M=3, MODE=0)],
    )


def lockstep_only():
    """Pairs that only the lockstep contents order (review of b926ed39298b: they used to
    PASS with a note). `drain` has count 2. The reader (warp 4, lane quarter 0) per m-block
    waits `go`, loads cols [80, 84), waits for the load and arrives on `drain`; the MMA
    warp per m-block m >= 1 waits for phase m of `drain`, then copies into cols [80, 84)
    and commits `go`. MODE 0: a `credit` role arrives on `drain` once per m-block with no
    wait, so on hardware its arrivals can complete phase 1 alone and the copy of m-block 1
    can overwrite the columns while the load of m-block 0 is in flight; the lockstep
    contents (1 + 1 per phase) order the pair, sound happens-before does not: undecided
    (U1). MODE 1: the landing rule under the credit: the writer copies and commits on a
    count-2 slot a helper role arrives on twice; the reader waits for phase 1 and loads the
    copied columns: undecided. MODE 2: as 0, but the credit role waits `go` of the m-block
    before arriving, so the drain wait of m-block m sees at most m of its arrivals: PASS.
    MODE 3: as 0, plus an unordered pair (the MMA copies into cols [200, 204) at the start
    of every m-block, the reader stores there at its end): a certain FAIL (F1), so the
    input is decided and the U1 pair is listed after it without a `not decided` line."""
    f = "lockstep_only.py"
    m = var("m")
    plain = cmp("ne", var("MODE"), c(1))
    mma_body = [
        if_(
            plain,
            [
                if_(
                    cmp("sge", m, c(1)),
                    [wait("drain", parity(b("sub", m, c(1))), f"{f}:10")],
                    [],
                    f"{f}:10",
                ),
                if_(mode_is(3), [cp(200, f"{f}:13")], [], f"{f}:13"),
                cp(80, f"{f}:11"),
                commit("go", f"{f}:12"),
            ],
            [],
            f"{f}:10",
        )
    ]
    credit_body = [
        if_(mode_is(2), [wait("go", parity(m), f"{f}:30")], [], f"{f}:30"),
        if_(plain, [arrive("drain", f"{f}:31")], [], f"{f}:31"),
    ]
    rd_body = [
        if_(
            plain,
            [
                wait("go", parity(m), f"{f}:20"),
                tmem("ld", c(4), [rect("data", False, c(0), 32, c(80), 4)], f"{f}:21"),
                twait(True, f"{f}:22"),
                arrive("drain", f"{f}:23"),
                if_(
                    mode_is(3),
                    [
                        tmem(
                            "st",
                            c(4),
                            [rect("data", True, c(0), 32, c(200), 4)],
                            f"{f}:24",
                        ),
                        twait(False, f"{f}:25"),
                    ],
                    [],
                    f"{f}:24",
                ),
            ],
            [],
            f"{f}:20",
        )
    ]
    w1 = [
        if_(mode_is(1), [cp(0, f"{f}:40"), commit("ready", f"{f}:41")], [], f"{f}:40")
    ]
    h1 = [
        if_(
            mode_is(1),
            [arrive("ready", f"{f}:50"), arrive("ready", f"{f}:51")],
            [],
            f"{f}:50",
        )
    ]
    r1 = [
        if_(
            mode_is(1),
            [
                wait("ready", c(0), f"{f}:60"),
                tmem("ld", c(5), [rect("data", False, c(32), 32, c(0), 4)], f"{f}:61"),
                twait(True, f"{f}:62"),
            ],
            [],
            f"{f}:60",
        )
    ]
    roles = [
        role("mma", 1, [for_("m", c(0), var("M"), mma_body, f"{f}:9")] + w1),
        role("credit", 1, [for_("m", c(0), var("M"), credit_body, f"{f}:29")] + h1),
        role("reader", 1, [for_("m", c(0), var("M"), rd_body, f"{f}:19")] + r1),
    ]
    return kernel(
        "lockstep_only",
        roles,
        [("go", 1), ("drain", 2), ("ready", 2)],
        [
            inp("credit_two_mblocks", M=2, MODE=0),
            inp("credit_three_mblocks", M=3, MODE=0),
            inp("commit_landing", M=1, MODE=1),
            inp("waited_credit", M=3, MODE=2),
            inp("credit_and_race", M=4, MODE=3),
        ],
    )


def named_bars():
    """Two roles share a lane quarter: role a loads cols [0, 32) of quarter 0, role b then
    stores there. MODE 0: both bar.sync id 1 with count 64 (exact, PASS, no note); 1: a
    bar.arrive, b bar.sync (cap rule, PASS); 2: no barrier (FAIL); 3: b syncs on id 1 with
    count 96 (inconsistent counts: undecided)."""
    f = "named_bars.py"
    a_body = [
        tmem("ld", c(0), [rect("data", False, c(0), 32, c(0), 32)], f"{f}:10"),
        twait(True, f"{f}:11"),
        if_(
            cmp("sle", var("MODE"), c(0)), [nbar(1, 64, True, f"{f}:12")], [], f"{f}:12"
        ),
        if_(mode_is(1), [nbar(1, 64, False, f"{f}:13")], [], f"{f}:13"),
        if_(mode_is(3), [nbar(1, 64, True, f"{f}:14")], [], f"{f}:14"),
    ]
    b_body = [
        if_(
            cmp("sle", var("MODE"), c(1)), [nbar(1, 64, True, f"{f}:20")], [], f"{f}:20"
        ),
        if_(mode_is(3), [nbar(1, 96, True, f"{f}:21")], [], f"{f}:21"),
        tmem("st", c(4), [rect("data", True, c(0), 32, c(0), 32)], f"{f}:22"),
        twait(False, f"{f}:23"),
    ]
    roles = [role("a", 1, a_body), role("b", 1, b_body)]
    return kernel(
        "named_bars",
        roles,
        [],
        [
            inp("bar_sync", M=0, MODE=0),
            inp("bar_arrive_sync", M=0, MODE=1),
            inp("no_barrier", M=0, MODE=2),
            inp("inconsistent", M=0, MODE=3),
        ],
    )


def same_role():
    """Same-role hazards of the MMA warp. MODE 0: cp [80, 84) then an MMA reading SF at 80
    and an MMA A read [0, 8) then an MMA D write there (pipelined, no note); 1: cp then cp
    to the same columns (N1); 2: MMA D write [0, 128) then a cp into it (N1); 3: as 2 with a
    commit and a wait on it between (no note)."""
    f = "same_role.py"
    body = [
        if_(
            mode_is(0),
            [
                cp(80, f"{f}:10"),
                mma(256, f"{f}:11", sf=(80, 80), a=(0, 8)),
                mma(0, f"{f}:12", ncols=8),
            ],
            [],
            f"{f}:10",
        ),
        if_(mode_is(1), [cp(80, f"{f}:13"), cp(80, f"{f}:14")], [], f"{f}:13"),
        if_(mode_is(2), [mma(0, f"{f}:15"), cp(80, f"{f}:16")], [], f"{f}:15"),
        if_(
            mode_is(3),
            [
                mma(0, f"{f}:17"),
                commit("acc", f"{f}:18"),
                wait("acc", c(0), f"{f}:19"),
                cp(80, f"{f}:20"),
            ],
            [],
            f"{f}:17",
        ),
    ]
    roles = [role("mma", 1, body)]
    return kernel(
        "same_role",
        roles,
        [("acc", 1)],
        [
            inp("pipelined", M=0, MODE=0),
            inp("cp_cp", M=0, MODE=1),
            inp("mma_d_cp", M=0, MODE=2),
            inp("mma_d_commit_cp", M=0, MODE=3),
        ],
    )


def cross_warp():
    """One role of warps 0 and 4 (both lane quarter 0): warp 0 loads cols [0, 32), then
    warp 4 stores there. MODE 0: no bar.sync between them (PASS with an N2 note); 1: a
    bar.sync of the role (count 64) after the wait::ld (PASS, no note)."""
    f = "cross_warp.py"
    body = [
        tmem("ld", c(0), [rect("data", False, c(0), 32, c(0), 32)], f"{f}:10"),
        twait(True, f"{f}:11"),
        if_(mode_is(1), [nbar(2, 64, True, f"{f}:12")], [], f"{f}:12"),
        tmem("st", c(4), [rect("data", True, c(0), 32, c(0), 32)], f"{f}:13"),
        twait(False, f"{f}:14"),
    ]
    return kernel(
        "cross_warp",
        [role("pair", 2, body)],
        [],
        [inp("no_sync", M=0, MODE=0), inp("bar_sync", M=0, MODE=1)],
    )


def partial_issue():
    """The CuTe P1 pass's split form (interp.Translator._p1_split_by_issuers) of a
    2-warp compute role (warps 2 and 3) in which thread 0 alone issues an mbarrier event:
    the split roles c2 and c3 are one warp each and only c2 has the event (review of
    6acbcf1f3a04: as one 2-warp role these PASSed). The MMA warp copies into cols [64, 68) and commits `go`. MODE 0: both
    warps wait `go`, load cols [64, 96) of their lane quarter and wait for the load; thread
    0 (c2) alone arrives on the count-1 `drain`, which the MMA waits before copying into
    cols [64, 68) again: c3's load is unordered against that copy (FAIL). MODE 1: as 0
    with a bar.sync of both warps (id 1, count 64) between the wait::ld and the arrive
    (PASS). MODE 2: thread 0 (c2) alone waits `go` before both warps load; no second copy:
    c3's load is unordered against the first copy (FAIL)."""
    f = "partial_issue.py"
    upto1 = cmp("sle", var("MODE"), c(1))
    mma_body = [
        cp(64, f"{f}:10"),
        commit("go", f"{f}:11"),
        if_(upto1, [wait("drain", c(0), f"{f}:12"), cp(64, f"{f}:13")], [], f"{f}:12"),
    ]

    def comp(w, line, lead):
        go = [wait("go", c(0), f"{f}:{line}")]
        body = go if lead else [if_(upto1, go, [], f"{f}:{line}")]
        body += [
            tmem("ld", c(w), [rect("data", False, quarter(c(w)), 32, c(64), 32)], f"{f}:{line + 1}"),
            twait(True, f"{f}:{line + 2}"),
            if_(mode_is(1), [nbar(1, 64, True, f"{f}:{line + 3}")], [], f"{f}:{line + 3}"),
        ]  # fmt: skip
        if lead:
            body.append(
                if_(upto1, [arrive("drain", f"{f}:{line + 4}")], [], f"{f}:{line + 4}")
            )
        return body

    roles = [
        role("mma", 1, mma_body),
        role("c2", 1, comp(2, 20, True)),
        role("c3", 1, comp(3, 30, False)),
    ]
    return kernel(
        "partial_issue",
        roles,
        [("go", 1), ("drain", 1)],
        [
            inp("one_thread_arrives", M=0, MODE=0),
            inp("bar_sync_then_one_arrives", M=0, MODE=1),
            inp("one_thread_waits", M=0, MODE=2),
        ],
    )


def partial_event():
    """Partial credit of a multi-warp arrive (review round 2, CE1). A compute role of 2
    warps (warps 2, 3) waits `go`, loads cols [64, 96) of its lane quarter, waits for the
    load and arrives once per warp (2 arrivals). The MMA warp copies into cols [64, 68),
    commits `go`, waits for phase 1 of the drain and copies there again (WAR against the
    loads). MODE 0: `drain` has count 3 and a `credit` role arrives on it twice with no
    wait, so phase 1 can complete with warp 2's arrival while warp 3 is still loading: the
    phase needs only 1 of the event's 2 arrivals, so neither load surely completed before the
    second copy; the lockstep contents (1 + 2) order them: undecided (U1). MODE 1: no
    credit; the count-1 `drain1` completes at either warp's arrival: unordered even under
    the lockstep reading (F1). Before partial credit both PASSed."""
    f = "partial_event.py"
    mma_body = [
        cp(64, f"{f}:10"),
        commit("go", f"{f}:11"),
        if_(
            mode_is(0),
            [wait("drain", c(0), f"{f}:12")],
            [wait("drain1", c(0), f"{f}:12")],
            f"{f}:12",
        ),
        cp(64, f"{f}:13"),
    ]
    credit_body = [
        if_(
            mode_is(0),
            [arrive("drain", f"{f}:30"), arrive("drain", f"{f}:31")],
            [],
            f"{f}:30",
        )
    ]
    comp_body = [
        wait("go", c(0), f"{f}:20"),
        per_warp("ld", 64, 32, f"{f}:21", False),
        twait(True, f"{f}:22"),
        if_(
            mode_is(0),
            [arrive("drain", f"{f}:23", per_warp=True)],
            [arrive("drain1", f"{f}:23", per_warp=True)],
            f"{f}:23",
        ),
    ]
    roles = [
        role("mma", 1, mma_body),
        role("credit", 1, credit_body),
        role("compute", 2, comp_body),
    ]
    return kernel(
        "partial_event",
        roles,
        [("go", 1), ("drain", 3), ("drain1", 1)],
        [inp("credit_two", M=0, MODE=0), inp("count_one", M=0, MODE=1)],
    )


def partial_event_control():
    """`partial_event` with a count-4 `drain`. MODE 0: the credit role arrives twice, so
    phase 1 needs both compute warps' arrivals: every warp's load completed before the
    second copy (PASS, the control). MODE 1: the credit role arrives 3 times, so phase 1
    needs only one of them: undecided (U1)."""
    f = "partial_event_control.py"
    mma_body = [
        cp(64, f"{f}:10"),
        commit("go", f"{f}:11"),
        wait("drain", c(0), f"{f}:12"),
        cp(64, f"{f}:13"),
    ]
    credit_body = [
        arrive("drain", f"{f}:30"),
        arrive("drain", f"{f}:31"),
        if_(mode_is(1), [arrive("drain", f"{f}:32")], [], f"{f}:32"),
    ]
    comp_body = [
        wait("go", c(0), f"{f}:20"),
        per_warp("ld", 64, 32, f"{f}:21", False),
        twait(True, f"{f}:22"),
        arrive("drain", f"{f}:23", per_warp=True),
    ]
    roles = [
        role("mma", 1, mma_body),
        role("credit", 1, credit_body),
        role("compute", 2, comp_body),
    ]
    return kernel(
        "partial_event_control",
        roles,
        [("go", 1), ("drain", 4)],
        [inp("all_warps_needed", M=0, MODE=0), inp("credit_three", M=0, MODE=1)],
    )


def partial_sync():
    """Partial credit with a bar.sync of the role before each of its arrivals (review
    round 3, `ce1b_partial_sync`): no warp of the role runs ahead of the other, yet a phase
    can complete without one of them. Two rounds; compute (warps 2, 3) per round:
    bar.sync(1, 64) of its warps, loads cols [64, 96) of its lane quarter, wait::ld, one
    arrival per warp on the drain; round 2 first waits `go`. The MMA warp waits for phase 1,
    copies into cols [64, 68), commits `go`, waits for phase 2 and copies again. MODE 0:
    `drain` has count 3 and a credit role arrives twice up front: phase 1 needs one compute
    arrival, so the first copy may overwrite the columns while the other warp still loads;
    the lockstep contents (1 + 2) order it: undecided (U1; it used to PASS). MODE 1: the
    credit role arrives once, so phase 2 never completes (a hang): FAIL. MODE 2: no credit,
    count-2 `drain2`: each phase needs both compute arrivals (PASS)."""
    f = "partial_sync.py"
    two = mode_is(2)

    def drain_wait(p, loc):
        return if_(two, [wait("drain2", c(p), loc)], [wait("drain", c(p), loc)], loc)

    def rnd(first):
        pre = [] if first else [wait("go", c(0), f"{f}:20")]
        return pre + [
            nbar(1, 64, True, f"{f}:21"),
            per_warp("ld", 64, 32, f"{f}:22", False),
            twait(True, f"{f}:23"),
            if_(
                two,
                [arrive("drain2", f"{f}:24", per_warp=True)],
                [arrive("drain", f"{f}:24", per_warp=True)],
                f"{f}:24",
            ),
        ]

    credit_body = [
        if_(cmp("sle", var("MODE"), c(1)), [arrive("drain", f"{f}:30")], [], f"{f}:30"),
        if_(mode_is(0), [arrive("drain", f"{f}:31")], [], f"{f}:31"),
    ]
    mma_body = [
        drain_wait(0, f"{f}:10"),
        cp(64, f"{f}:11"),
        commit("go", f"{f}:12"),
        drain_wait(1, f"{f}:13"),
        cp(64, f"{f}:14"),
    ]
    roles = [
        role("mma", 1, mma_body),
        role("credit", 1, credit_body),
        role("compute", 2, rnd(True) + rnd(False)),
    ]
    return kernel(
        "partial_sync",
        roles,
        [("go", 1), ("drain", 3), ("drain2", 2)],
        [
            inp("credit_two", M=0, MODE=0),
            inp("credit_one", M=0, MODE=1),
            inp("no_credit", M=0, MODE=2),
        ],
    )


def partial_split():
    """Partial credit in the split S_drain shape: compute
    split into roles of 1 warp (c1, warp 1) and 3 warps (c3, warps 2..4) on a count-4
    drain; each loads cols [64, 96) of its lane quarters, waits for the loads and arrives
    (c3 once per warp); the MMA warp waits for phase 1 and copies into cols [64, 68).
    MODE 0: c1 arrives twice, so phase 1 needs only 2 of c3's 3 arrivals: c3's loads are
    ordered only by the lockstep contents (1 + 3): undecided (U1; it used to PASS). MODE 1:
    c1 arrives once, so phase 1 needs all of c3's arrivals (PASS)."""
    f = "partial_split.py"
    one = [
        tmem("ld", c(1), [rect("data", False, c(32), 32, c(64), 32)], f"{f}:20"),
        twait(True, f"{f}:21"),
        arrive("drain", f"{f}:22"),
        if_(mode_is(0), [arrive("drain", f"{f}:23")], [], f"{f}:23"),
    ]
    three = [
        per_warp("ld", 64, 32, f"{f}:30", False, lo=2, hi=5),
        twait(True, f"{f}:31"),
        arrive("drain", f"{f}:32", per_warp=True),
    ]
    mma_body = [wait("drain", c(0), f"{f}:10"), cp(64, f"{f}:11")]
    roles = [role("mma", 1, mma_body), role("c1", 1, one), role("c3", 3, three)]
    return kernel(
        "partial_split",
        roles,
        [("drain", 4)],
        [inp("run_ahead_one", M=0, MODE=0), inp("one_each", M=0, MODE=1)],
    )


def partial_named():
    """Partial credit on a named barrier (review round 3, `ce_nbar_partial`; count 128): Y
    (warps 1, 2) loads cols [0, 32) of its lane quarters, waits and bar.sync(1); X (warp 3)
    bar.arrive(1); C (warp 0) bar.sync(1), then copies into cols [0, 4). MODE 0: X arrives
    twice, so phase 1 needs only one Y warp (128 - 64 - 32 = 32 of Y's 64 threads): the
    copy is unordered against the other warp's load (F1; it used to PASS; the lagging warp
    then waits in phase 2, a hang the model does not report for named barriers). MODE 1: X
    arrives once, so phase 1 needs both Y warps (PASS)."""
    f = "partial_named.py"
    Y = [
        per_warp("ld", 0, 32, f"{f}:20", False, lo=1, hi=3),
        twait(True, f"{f}:21"),
        nbar(1, 128, True, f"{f}:22"),
    ]
    X = [
        nbar(1, 128, False, f"{f}:30"),
        if_(mode_is(0), [nbar(1, 128, False, f"{f}:31")], [], f"{f}:31"),
    ]
    C = [nbar(1, 128, True, f"{f}:10"), cp(0, f"{f}:11")]
    roles = [role("C", 1, C), role("Y", 2, Y), role("X", 1, X)]
    return kernel(
        "partial_named",
        roles,
        [],
        [inp("x_twice", M=0, MODE=0), inp("x_once", M=0, MODE=1)],
    )


def stale_handoff():
    """The persistent dQ handoff shape: per tile, the MMA waits dq_empty with parity 1
    (fresh on the first tile) and writes D [256, 384); the reducer waits dq_full, loads
    the accumulator, and releases dq_empty. MODE 0 toggles the MMA's parity per tile
    (PASS); MODE 1 repeats parity 1 on every tile, a stale wait counted at its lower bound,
    so the next tile's D write is unordered against the previous tile's load (FAIL)."""
    f = "stale_handoff.py"
    t = var("t")
    par = b("xor", parity(t), c(1))
    mma_body = [
        for_(
            "t",
            c(0),
            var("M"),
            [
                if_(
                    mode_is(0),
                    [wait("dq_empty", par, f"{f}:10")],
                    [wait("dq_empty", c(1), f"{f}:11")],
                    f"{f}:10",
                ),
                mma(256, f"{f}:12"),
                commit("dq_full", f"{f}:13"),
            ],
            f"{f}:9",
        )
    ]
    red_body = [
        for_(
            "t",
            c(0),
            var("M"),
            [
                wait("dq_full", parity(t), f"{f}:20"),
                per_warp("ld", 256, 32, f"{f}:21", False, lo=0, hi=4),
                twait(True, f"{f}:22"),
                arrive("dq_empty", f"{f}:23", per_warp=True),
            ],
            f"{f}:19",
        )
    ]
    roles = [role("mma", 1, mma_body), role("reduce", 4, red_body)]
    return kernel(
        "stale_handoff",
        roles,
        [("dq_full", 1), ("dq_empty", 4)],
        [
            inp("toggled_two_tiles", M=2, MODE=0),
            inp("stale_two_tiles", M=2, MODE=1),
            inp("stale_one_tile", M=1, MODE=1),
        ],
    )


def ranges():
    """F2 and F3: MODE 0 in range (PASS); 1 a cp at column 510 (cols [510, 514) leave the
    512-column allocation: FAIL); 2 warp 1 loads lanes [0, 32) (outside its quarter [32,
    64): FAIL)."""
    f = "ranges.py"
    body = [
        if_(
            mode_is(0),
            [
                cp(508, f"{f}:10"),
                tmem("ld", c(1), [rect("data", False, c(32), 32, c(0), 8)], f"{f}:11"),
            ],
            [],
            f"{f}:10",
        ),
        if_(mode_is(1), [cp(510, f"{f}:12")], [], f"{f}:12"),
        if_(
            mode_is(2),
            [tmem("ld", c(1), [rect("data", False, c(0), 32, c(0), 8)], f"{f}:13")],
            [],
            f"{f}:13",
        ),
    ]
    k = kernel(
        "ranges",
        [role("w", 1, body)],
        [],
        [
            inp("in_range", M=0, MODE=0),
            inp("col_out_of_range", M=0, MODE=1),
            inp("lane_quarter", M=0, MODE=2),
        ],
    )
    return k


def ranges_relative():
    """F2 with a relative 256-column allocation: a cp at column M (252 in range, 254
    leaves it)."""
    f = "ranges_relative.py"
    body = [cp(var("M"), f"{f}:10")]
    return kernel(
        "ranges_relative",
        [role("w", 1, body)],
        [],
        [inp("in_range", M=252, MODE=0), inp("col_out_of_range", M=254, MODE=0)],
        tmem_info={"cols": 256, "base": "relative"},
    )


def meta_oob():
    """A role stops at an out-of-bounds metadata load before its copy: P1-1 undecided."""
    f = "meta_oob_p1.py"
    body = [
        {
            "op": "let",
            "var": "x",
            "expr": {"e": "load", "tensor": "tbl", "index": [var("M")], "w": 32},
            "loc": f"{f}:9",
        },
        cp(0, f"{f}:10"),
    ]
    r_body = [
        tmem("ld", c(0), [rect("data", False, c(0), 32, c(32), 32)], f"{f}:20"),
        twait(True, f"{f}:21"),
    ]
    tensors = [
        {
            "name": "tbl",
            "kind": "global",
            "elem_bytes": 4,
            "shape": [c(2)],
            "strides": [c(1)],
            "segments": None,
            "role": "input",
            "metadata": True,
        }
    ]
    k = kernel(
        "meta_oob_p1",
        [role("w", 1, body), role("r", 1, r_body)],
        [],
        [
            {
                "name": "in_bounds",
                "params": {"M": 1, "MODE": 0},
                "metadata": {"tbl": [0, 0]},
            },
            {"name": "oob", "params": {"M": 5, "MODE": 0}, "metadata": {"tbl": [0, 0]}},
        ],
        tensors=tensors,
    )
    return k


def hang():
    """A wait that never returns (nobody arrives on `never`) before an unordered copy: FAIL
    with the N3 note that happens-before after the blocked wait is vacuous."""
    f = "hang_vacuous.py"
    w_body = [wait("never", c(0), f"{f}:10"), cp(0, f"{f}:11")]
    r_body = [
        tmem("ld", c(0), [rect("data", False, c(0), 32, c(0), 32)], f"{f}:20"),
        twait(True, f"{f}:21"),
    ]
    return kernel(
        "hang_vacuous",
        [role("w", 1, w_body), role("r", 1, r_body)],
        [("never", 1)],
        [inp("blocked", M=0, MODE=0)],
    )


def cluster_pair():
    """A cluster of two CTAs (kir.md change 11): the leader's MMA role issues a
    cta_group::2 MMA (D cols [64, 128) in the TMEM of both CTAs, `cta_mask` 3) and commits
    `full` in both; a drain role in each CTA (warp 1) waits its `full`, loads its lane
    quarter of D and releases it on the leader's barrier (a remote arrive, `cta` 0), the
    2-CTA dQ drain. MODE 0: the MMA of m-block m >= 1 waits for both drains of m-block
    m - 1 (`empty`, count 2); MODE 1: the last m-block's MMA does not wait; MODE 2: the
    release barrier counts only the leader's drain (`empty1`, count 1), so nothing orders
    the follower's load before the next MMA writes the follower's TMEM."""
    f = "cluster_pair.py"
    rank = b("urem", {"e": "pid", "axis": 0, "w": 32}, c(2))
    m = var("m")

    def release(make):
        return if_(mode_is(2), [make("empty1")], [make("empty")], f"{f}:9")

    mma_ev = tmem("mma", c(0), [rect("d", True, ALL, 128, c(64), 64)], f"{f}:12")
    mma_ev["cta_mask"] = c(3)
    full = commit("full", f"{f}:13")
    full["cta_mask"] = c(3)
    take = release(lambda bar: wait(bar, parity(b("sub", m, c(1))), f"{f}:11"))
    mma_body = [
        if_(
            cmp("eq", rank, c(0)),
            [
                for_(
                    "m",
                    c(0),
                    var("M"),
                    [
                        if_(
                            cmp("sge", m, c(1)),
                            [if_(mode_is(1), [if_(cmp("slt", m, b("sub", var("M"), c(1))), [take], [], f"{f}:10")],
                                 [take], f"{f}:10")],
                            [],
                            f"{f}:10",
                        ),
                        mma_ev,
                        full,
                    ],
                    f"{f}:10",
                )
            ],
            [],
            f"{f}:9",
        )
    ]  # fmt: skip

    def give(bar):
        a = arrive(bar, f"{f}:23")
        a["cta"] = c(0)
        return a if bar == "empty" else if_(cmp("eq", rank, c(0)), [a], [], f"{f}:23")

    drain_body = [
        for_(
            "m",
            c(0),
            var("M"),
            [
                wait("full", parity(m), f"{f}:20"),
                tmem(
                    "ld", c(1), [rect("data", False, c(32), 32, c(64), 32)], f"{f}:21"
                ),
                twait(True, f"{f}:22"),
                release(give),
            ],
            f"{f}:19",
        )
    ]
    k = kernel(
        "cluster_pair",
        [role("mma", 1, mma_body), role("drain", 1, drain_body)],
        [("full", 1), ("empty", 2), ("empty1", 1)],
        [
            inp("ordered", M=3, MODE=0),
            inp("last_unwaited", M=3, MODE=1),
            inp("last_unwaited_one_block", M=1, MODE=1),
            inp("leader_only_release", M=2, MODE=2),
        ],
    )
    k["grid"] = [c(2), c(1), c(1)]
    k["cluster"] = [2, 1, 1]
    return k


FIXTURES = {
    "s2t_race": lambda: s2t("s2t_race", None),
    "s2t_drain": lambda: s2t("s2t_drain", "fix"),
    "drain_before_wait_ld": lambda: s2t("drain_before_wait_ld", "early"),
    "drain_not_waited": lambda: s2t("drain_not_waited", "nowait"),
    "handoff": handoff,
    "handoff_mixed": handoff_mixed,
    "split_drain": split_drain,
    "lockstep_only": lockstep_only,
    "named_bars": named_bars,
    "same_role": same_role,
    "cross_warp": cross_warp,
    "partial_issue": partial_issue,
    "partial_event": partial_event,
    "partial_event_control": partial_event_control,
    "partial_sync": partial_sync,
    "partial_split": partial_split,
    "partial_named": partial_named,
    "stale_handoff": stale_handoff,
    "ranges": ranges,
    "ranges_relative": ranges_relative,
    "meta_oob_p1": meta_oob,
    "hang_vacuous": hang,
    "cluster_pair": cluster_pair,
}


def main():
    for name, make in FIXTURES.items():
        (HERE / f"{name}.kir.json").write_text(json.dumps(make(), indent=1) + "\n")
        print(HERE / f"{name}.kir.json")


if __name__ == "__main__":
    main()
