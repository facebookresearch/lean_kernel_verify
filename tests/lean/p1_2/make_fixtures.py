# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Writes the P1-2 Lean fixtures tests/lean/p1_2/*.kir.json (P1 KIR with "p1": ["P1-2"],
references/kir.md change 15). Each fixture is a small hand-made kernel; its inputs select
the variant (param MODE) and the trip count (T tiles or I work items). Expected verdicts:
tests/lean/p1_2/expected.txt; run by tests/lean/run_all.sh.

    python3 -I tests/lean/p1_2/make_fixtures.py      # rewrites the fixtures in place

The fixtures model the TLX/Triton patterns P1-2 was written for: an epilogue whose 128
threads write a staging buffer in shared memory and whose elected lane of warp 0 stores it
with a TMA store (`staging`, single and double buffered, with and without the `.read` wait
and the named barrier); a partition that rebuilds an in-kernel tensor descriptor in global
scratch at the top of every work item while the previous item's stores through it may still
read it (`tmap_rebuild`); a cross-role handoff through mbarriers or a named barrier
(`handoff`); the issuing thread's own program order (`same_thread`); and a TMA load landing
in a store source, completed through its mbarrier before the store or not (`landing`).
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYM = "smem"
SCR = "scratch"


def c(v, w=32):
    return {"e": "const", "v": v, "w": w}


def var(n, w=32):
    return {"e": "var", "name": n, "w": w}


def b(o, x, y, w=32):
    return {"e": "bin", "o": o, "a": x, "b": y, "w": w}


def cmp(o, x, y):
    return {"e": "cmp", "o": o, "a": x, "b": y, "w": 1}


def ext64(x):
    return {"e": "ext", "signed": False, "a": x, "w": 64}


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


def when(cond, body, loc):
    return if_(cond, body, [], loc)


def nbar(i, count, loc, sync=True):
    return ev("named_bar", loc, id=c(i), count=c(count), sync=sync)


def wait(bar, parity, loc):
    return ev("wait", loc, bar=bar, idx=c(0), parity=parity)


def arrive(bar, loc, per_warp=False):
    e = ev("arrive", loc, bar=bar, idx=c(0))
    if per_warp:
        e["per_warp"] = True
    return e


def store(src, nbytes, loc, warp=0, lane=None, tmap=None, kind="tensor"):
    e = ev("bulk_store", loc, kind=kind, warp=c(warp), sym=SYM, src=src, bytes=nbytes)
    if lane is not None:
        e["lane"] = c(lane)
    if tmap is not None:
        e["tmap"] = {"base": SCR, "addr": tmap}
    return e


def commit(loc):
    return ev("bulk_commit", loc)


def bwait(n, read, loc):
    return ev("bulk_wait", loc, n=n, read=read)


def splat(x):
    return {"t": "splat", "shape": [1], "w": 32, "a": x}


def write1(off, nbytes, loc, warp, lane=None, sync=False, async_=False, bar=None):
    e = ev(
        "smem_write",
        loc,
        sym=SYM,
        offsets=splat(off),
        mask=None,
        bytes=nbytes,
        warp=warp,
    )
    if lane is not None:
        e["lane"] = lane
    if sync:
        e["sync"] = True
    if async_:
        e["async"] = True
    if bar is not None:  # a landing's mbarrier slot (index 0)
        e["bar"], e["idx"] = bar, c(0)
    return e


def write_threads(base, loc, threads=128, w0=0):
    """Every thread t of a role writes 16 bytes at base + 16 t (st.shared.v4)."""
    t = var("t")
    return for_(
        "t",
        c(0),
        c(threads),
        [write1(b("add", base, b("mul", t, c(16))), 16, loc,
                b("add", b("udiv", t, c(32)), c(w0)), b("urem", t, c(32)))],
        loc,
    )  # fmt: skip


def tmapw(addr, warp, loc):
    return ev("tmap_write", loc, base=SCR, addr=addr, warp=c(warp))


def kernel(
    name, roles, inputs, params, barriers=(), grid=1, cta_threads=None, tensors=()
):
    out = {
        "kir_version": 1,
        "kernel": name,
        "source": f"tests/lean/p1_2/{name}.kir.json",
        "ir": "ptx",
        "grid": [c(grid), c(1), c(1)],
        "params": [{"name": p, "type": "i32"} for p in params],
        "tensors": list(tensors),
        "barriers": [{"name": n, "count": c(k), "num": c(1)} for n, k in barriers],
        "roles": roles,
        "p1": ["P1-2"],
        "case": {"name": name, "inputs": inputs},
    }
    if cta_threads is not None:
        out["cta_threads"] = cta_threads
    return out


def role(name, warps, body):
    return {"name": name, "warps": warps, "threads_per_issue": 1, "body": body}


def inp(name, grid=None, **params):
    out = {"name": name, "params": params}
    if grid is not None:
        out["grid"] = [grid, 1, 1]
    return out


# ------------------------------------------------------------------ staging buffer


def staging():
    """A 4-warp epilogue: per tile, the 128 threads write a 2048-byte staging buffer, the
    elected lane of warp 0 stores it (TMA) and commits. MODE: 0 `.read` wait + bar.sync
    before the writes (single buffer); 1 no wait; 2 wait, no bar.sync before the writes;
    3 double buffer, wait_group.read 1; 4 double buffer, wait_group.read 2 (too lax);
    5 full wait; 6 bar.sync before the wait only; 7 no bar.sync between the writes and the
    store; 8 the second bar.sync uses another count on the same id; 9 as 0 with barriers of
    256 threads, the CTA's (`cta_threads` 256: idle warps without a role take part)."""
    f = "staging.py"
    i = var("i")
    double = mode_is(3, 4)
    buf = {
        "e": "select",
        "c": double,
        "a": b("mul", b("urem", i, c(2)), c(2048)),
        "b": c(0),
        "w": 32,
    }
    body = [
        {"op": "let", "var": "buf", "expr": buf, "loc": f"{f}:3"},
        when(mode_is(0, 2, 7, 8), [bwait(0, True, f"{f}:4")], f"{f}:4"),
        when(mode_is(3), [bwait(1, True, f"{f}:5")], f"{f}:5"),
        when(mode_is(4), [bwait(2, True, f"{f}:6")], f"{f}:6"),
        when(mode_is(5), [bwait(0, False, f"{f}:7")], f"{f}:7"),
        when(mode_is(0, 1, 3, 4, 5, 7, 8), [nbar(1, 128, f"{f}:8")], f"{f}:8"),
        when(mode_is(9), [bwait(0, True, f"{f}:4"), nbar(2, 256, f"{f}:8")], f"{f}:8"),
        when(mode_is(6), [nbar(1, 128, f"{f}:9"), bwait(0, True, f"{f}:9")], f"{f}:9"),
        write_threads(var("buf"), f"{f}:10"),
        when(mode_is(0, 1, 2, 3, 4, 5, 6), [nbar(1, 128, f"{f}:11")], f"{f}:11"),
        when(mode_is(8), [nbar(1, 64, f"{f}:12")], f"{f}:12"),
        when(mode_is(9), [nbar(2, 256, f"{f}:11")], f"{f}:11"),
        store(var("buf"), 2048, f"{f}:13"),
        commit(f"{f}:13"),
    ]
    roles = [role("epi", 4, [for_("i", c(0), var("T"), body, f"{f}:2")])]
    inputs = [
        inp("read_bar_1", MODE=0, T=1),
        inp("read_bar_3", MODE=0, T=3),
        inp("no_wait_1", MODE=1, T=1),
        inp("no_wait_2", MODE=1, T=2),
        inp("wait_no_bar_2", MODE=2, T=2),
        inp("double_wait1_3", MODE=3, T=3),
        inp("double_wait2_2", MODE=4, T=2),
        inp("double_wait2_3", MODE=4, T=3),
        inp("full_wait_3", MODE=5, T=3),
        inp("bar_before_wait_2", MODE=6, T=2),
        inp("no_bar_before_store_1", MODE=7, T=1),
        inp("inconsistent_bar_2", MODE=8, T=2),
        inp("cta_wide_bar_3", MODE=9, T=3),
    ]
    return kernel("staging", roles, inputs, ("MODE", "T"), cta_threads=256)


def staging_wide():
    """staging's MODE 9 without `cta_threads`: its 256-thread barriers are not exact (the
    role has 128 threads), so they order nothing in the sound reading (undecided)."""
    k = staging()
    k["kernel"] = k["case"]["name"] = "staging_wide"
    k["source"] = "tests/lean/p1_2/staging_wide.kir.json"
    k["case"]["inputs"] = [
        i for i in k["case"]["inputs"] if i["name"] == "cta_wide_bar_3"
    ]
    del k["cta_threads"]
    return k


# ------------------------------------------------------------- descriptor rebuild


def tmap_rebuild():
    """A 4-warp partition that rebuilds an in-kernel descriptor (tensormap.cp_fenceproxy of
    warp 0 to its CTA's 128-byte scratch slot) at the top of each of I work items, then per
    item stores one tile through it. MODE: 0 unfixed (the only wait, wait_group.read 0 +
    bar.sync, follows the rebuild); 1 fixed (wait_group 0 + bar.sync before it); 2 fixed with
    wait_group.read 0 (the PTX ISA: .read covers the tensormap read); 3 full wait, no
    bar.sync, rebuild by warp 1; 4 full wait, no bar.sync, rebuild by warp 0 (the issuer's
    warp: undecided); 5 the fix, but every CTA uses scratch slot 0 (two CTAs, no clusters)."""
    f = "tmap_rebuild.py"
    slot = {"e": "select", "c": mode_is(5), "a": c(0, 64),
            "b": b("mul", ext64({"e": "pid", "axis": 0, "w": 32}), c(128, 64), 64), "w": 64}  # fmt: skip
    body = [
        when(
            mode_is(1, 5),
            [bwait(0, False, f"{f}:4"), nbar(10, 128, f"{f}:4")],
            f"{f}:4",
        ),
        when(mode_is(2), [bwait(0, True, f"{f}:5"), nbar(10, 128, f"{f}:5")], f"{f}:5"),
        when(mode_is(3, 4), [bwait(0, False, f"{f}:6")], f"{f}:6"),
        if_(
            mode_is(3),
            [tmapw(var("slot", 64), 1, f"{f}:7")],
            [tmapw(var("slot", 64), 0, f"{f}:8")],
            f"{f}:7",
        ),
        bwait(0, True, f"{f}:9"),
        nbar(10, 128, f"{f}:9"),
        write_threads(c(0), f"{f}:10"),
        nbar(10, 128, f"{f}:11"),
        store(c(0), 2048, f"{f}:12", tmap=var("slot", 64)),
        commit(f"{f}:12"),
    ]
    roles = [role("ln", 4, [{"op": "let", "var": "slot", "expr": slot, "loc": f"{f}:2"},
                            for_("it", c(0), var("I"), body, f"{f}:3")])]  # fmt: skip
    inputs = [
        inp("unfixed_1", MODE=0, I=1),
        inp("unfixed_2", MODE=0, I=2),
        inp("fixed_3", MODE=1, I=3),
        inp("fixed_read_3", MODE=2, I=3),
        inp("other_warp_2", MODE=3, I=2),
        inp("issuer_warp_2", MODE=4, I=2),
        inp("shared_slot_2ctas", grid=2, MODE=5, I=1),
        inp("own_slots_2ctas", grid=2, MODE=1, I=2),
    ]
    return kernel("tmap_rebuild", roles, inputs, ("MODE", "I"))


# ---------------------------------------------------------------- cross-role handoff


def handoff():
    """A 1-warp store role and a 4-warp fill role share one staging buffer through a
    full/empty mbarrier pair: fill waits `empty` (producer parity), writes, bar.syncs its
    warps and arrives `full` once per warp; store waits `full`, stores from the elected
    lane, commits, waits wait_group.read 0 and arrives `empty`. MODE: 0 as described (the
    arrive's issuing threads are not in the KIR: undecided); 1 store and fill also meet on
    named barrier 2 (32 + 128 threads) after the store's wait and before the writes; 2 no
    wait_group in the store role."""
    f = "handoff.py"
    i = var("i")
    fill = [
        wait("empty", b("xor", b("and", i, c(1)), c(1)), f"{f}:3"),
        when(mode_is(1), [nbar(2, 160, f"{f}:4")], f"{f}:4"),
        write_threads(c(0), f"{f}:5"),
        nbar(1, 128, f"{f}:6"),
        arrive("full", f"{f}:7", per_warp=True),
    ]
    st = [
        wait("full", b("and", i, c(1)), f"{f}:9"),
        store(c(0), 2048, f"{f}:10"),
        commit(f"{f}:10"),
        when(mode_is(0, 1), [bwait(0, True, f"{f}:11")], f"{f}:11"),
        when(mode_is(1), [nbar(2, 160, f"{f}:12")], f"{f}:12"),
        arrive("empty", f"{f}:13"),
    ]
    roles = [
        role("fill", 4, [for_("i", c(0), var("T"), fill, f"{f}:2")]),
        role("store", 1, [for_("i", c(0), var("T"), st, f"{f}:8")]),
    ]
    inputs = [
        inp("mbarrier_only_2", MODE=0, T=2),
        inp("named_2", MODE=1, T=2),
        inp("named_3", MODE=1, T=3),
        inp("no_wait_1", MODE=2, T=1),
        inp("no_wait_2", MODE=2, T=2),
    ]
    return kernel(
        "handoff", roles, inputs, ("MODE", "T"), barriers=(("full", 4), ("empty", 1))
    )


# --------------------------------------------------------------- same-thread order


def same_thread():
    """A 1-warp role: one bulk store of shared bytes [0, 1024), one 16-byte write at 0.
    MODE: 0 store by lane 0, wait, write by lane 0 (program order); 1 write by lane 5 (another
    lane: a bar.warp.sync, which the KIR drops, may order it: undecided);
    2 store by the elected lane, write by lane 3 (maybe the issuer: undecided); 3 store and
    write both by the elected lane (a TMA load by the same elected lane); 4 a warp-collective
    write (stmatrix) and then the store, no barrier; 5 a write by lane 7 then the store by the
    elected lane, no barrier (undecided); 6 store and write by the elected lane, no wait;
    7 a bulk reduce never waited for and no conflicting write (a note); 8 a TMA load by the
    elected lane landing in the source, then the store (the landing may follow the store's
    read: undecided); 9 the store, its wait, then such a load (PASS); 10 store by lane 0, wait,
    then every lane writes 16 bytes (lane 0's ordered, the others as in 1: undecided)."""
    f = "same_thread.py"
    w0 = c(0)
    body = [
        if_(mode_is(0, 1), [store(c(0), 1024, f"{f}:2", lane=0, kind="bulk")],
            [when(mode_is(2, 3, 6), [store(c(0), 1024, f"{f}:3", kind="bulk")], f"{f}:3")], f"{f}:2"),
        when(mode_is(7), [store(c(0), 1024, f"{f}:4", kind="bulk_reduce")], f"{f}:4"),
        commit(f"{f}:5"),
        when(mode_is(0, 1, 2, 3), [bwait(0, True, f"{f}:6")], f"{f}:6"),
        when(mode_is(0), [write1(c(0), 16, f"{f}:7", w0, c(0))], f"{f}:7"),
        when(mode_is(1), [write1(c(0), 16, f"{f}:8", w0, c(5))], f"{f}:8"),
        when(mode_is(2), [write1(c(0), 16, f"{f}:9", w0, c(3))], f"{f}:9"),
        when(mode_is(3, 6), [write1(c(0), 16, f"{f}:10", w0)], f"{f}:10"),
        when(mode_is(4), [write1(c(0), 16, f"{f}:11", w0, sync=True),
                          store(c(0), 1024, f"{f}:12", kind="bulk"), commit(f"{f}:12")], f"{f}:11"),
        when(mode_is(5), [write1(c(0), 16, f"{f}:13", w0, c(7)),
                          store(c(0), 1024, f"{f}:14", kind="bulk"), commit(f"{f}:14")], f"{f}:13"),
        when(mode_is(7), [write1(c(4096), 16, f"{f}:15", w0, c(0))], f"{f}:15"),
        when(mode_is(8), [write1(c(0), 1024, f"{f}:16", w0, async_=True),
                          store(c(0), 1024, f"{f}:17", kind="bulk"), commit(f"{f}:17")], f"{f}:16"),
        when(mode_is(9), [store(c(0), 1024, f"{f}:18", kind="bulk"), commit(f"{f}:18"),
                          bwait(0, True, f"{f}:19"), write1(c(0), 1024, f"{f}:20", w0, async_=True)],
             f"{f}:18"),
        when(mode_is(10), [store(c(0), 1024, f"{f}:21", lane=0, kind="bulk"), commit(f"{f}:21"),
                           bwait(0, True, f"{f}:22"), write_threads(c(0), f"{f}:23", threads=32)],
             f"{f}:21"),
    ]  # fmt: skip
    roles = [role("w", 1, body)]
    inputs = [
        inp("lane0_lane0", MODE=0),
        inp("lane0_lane5", MODE=1),
        inp("elect_lane3", MODE=2),
        inp("elect_elect", MODE=3),
        inp("collective_before", MODE=4),
        inp("lane7_before_elect", MODE=5),
        inp("elect_elect_no_wait", MODE=6),
        inp("never_completed", MODE=7),
        inp("landing_before", MODE=8),
        inp("landing_after_wait", MODE=9),
        inp("lane0_warp_after_wait", MODE=10),
    ]
    return kernel("same_thread", roles, inputs, ("MODE",))


# ------------------------------------------------------------------ landings


def landing():
    """A 1-warp role, one mbarrier `full` (count 1): per tile i of T, the elected lane
    expects 1024 bytes on `full`, a TMA load of 1024 bytes lands in the store source
    [0, 1024) on `full` (the landing names that slot, its tma_load next), the role waits
    for phase i + 1, the elected lane stores the source and waits `.read` for it. MODE:
    0 as described (the wait's phase needs the bytes of every load issued before it, so the
    landing is complete before the store: PASS); 1 the store before the wait (undecided);
    2 a second load (into [4096, 5120)) on `full` without its own expect_tx, before the wait
    (the phase can complete with either load's bytes: undecided); 3 the landing names no
    slot (undecided)."""
    f = "landing.py"
    i = var("i")
    w0 = c(0)

    def load(loc, off, bar):
        return [write1(c(off), 1024, loc, w0, async_=True, bar=bar),
                ev("tma_load", loc, desc="src", coords=[c(0)], bar="full", idx=c(0))]  # fmt: skip

    wait_ = wait("full", b("urem", i, c(2)), f"{f}:5")
    store_ = [store(c(0), 1024, f"{f}:6", kind="bulk"), commit(f"{f}:6")]
    body = [
        for_("i", c(0), var("T"), [
            ev("expect_tx", f"{f}:2", bar="full", idx=c(0), bytes=c(1024)),
            arrive("full", f"{f}:2"),
            if_(mode_is(3), load(f"{f}:3", 0, None), load(f"{f}:3", 0, "full"), f"{f}:3"),
            when(mode_is(2), load(f"{f}:4", 4096, "full"), f"{f}:4"),
            if_(mode_is(1), store_ + [wait_], [wait_] + store_, f"{f}:5"),
            bwait(0, True, f"{f}:7"),
        ], f"{f}:1"),
    ]  # fmt: skip
    src = {"name": "src", "kind": "desc", "elem_bytes": 4, "shape": [c(256)],
           "strides": [c(1)], "box": [256], "segments": None, "role": "input",
           "metadata": False}  # fmt: skip
    roles = [role("w", 1, body)]
    inputs = [
        inp("resolved_1", MODE=0, T=1),
        inp("resolved_2", MODE=0, T=2),
        inp("store_before_wait_2", MODE=1, T=2),
        inp("load_ahead_1", MODE=2, T=1),
        inp("no_slot_1", MODE=3, T=1),
    ]
    return kernel("landing", roles, inputs, ("MODE", "T"), barriers=(("full", 1),),
                  tensors=[src])  # fmt: skip


# ------------------------------------------------------------ another CTA's write


def cluster_remote():
    """A 2-CTA cluster, one 1-warp role per CTA: lane 0 stores its CTA's shared bytes
    [0, 1024) with a bulk store, then writes 16 bytes into the peer CTA's shared memory (an
    st.async: `cta` the peer's rank, `async`) and waits for its store. MODE 0: at offset 0,
    inside the peer's store source, with no happens-before between the CTAs (FAIL F1);
    1: at offset 4096 (PASS)."""
    f = "cluster_remote.py"
    peer = b("xor", b("urem", {"e": "pid", "axis": 0, "w": 32}, c(2)), c(1))
    off = {"e": "select", "c": mode_is(0), "a": c(0), "b": c(4096), "w": 32}
    w = write1(off, 16, f"{f}:4", c(0), c(0), async_=True)
    w["cta"] = peer
    body = [
        store(c(0), 1024, f"{f}:2", lane=0, kind="bulk"),
        commit(f"{f}:2"),
        w,
        bwait(0, True, f"{f}:5"),
    ]
    inputs = [inp("into_source", MODE=0), inp("elsewhere", MODE=1)]
    k = kernel("cluster_remote", [role("w", 1, body)], inputs, ("MODE",), grid=2)
    k["cluster"] = [2, 1, 1]
    return k


def main():
    for k in (staging(), staging_wide(), tmap_rebuild(), handoff(), same_thread(), landing(),
              cluster_remote()):  # fmt: skip
        (HERE / f"{k['kernel']}.kir.json").write_text(json.dumps(k, indent=1) + "\n")


if __name__ == "__main__":
    main()
