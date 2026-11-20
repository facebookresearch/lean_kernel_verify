# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Negative controls for frontend/kir.py: every mutation of a valid fixture in MUTATIONS
(and P1_MUTATIONS, P1_2_MUTATIONS, NB_MUTATIONS) must be rejected with the expected
message; every mutation in ACCEPTED must stay valid.

    python3 -I kir_negative.py      (exit 1 if some mutation is not handled as expected)
"""

import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True  # keep the source tree free of __pycache__
sys.path.insert(0, os.path.join(HERE, "..", "..", "frontend"))
import kir  # noqa: E402


def c(v, w=32):
    return {"e": "const", "v": v, "w": w}


def fixture(name):
    with open(os.path.join(HERE, f"{name}.kir.json")) as f:
        return json.load(f)


# Global-synchronization fixtures (tests/lean/gsync, references/kir.md change 18)
def gs(name):
    with open(os.path.join(HERE, "gsync", f"{name}.kir.json")) as f:
        return json.load(f)


def gs_ev(k, ev):  # the first `ev` event of any role
    return next(e for r in k["roles"] if (e := first_ev(r["body"], ev)) is not None)


GSYNC_MUTATIONS = [
    (
        "tagged",
        "a store to a data tensor (KIR change 18)",
        lambda k: gs_ev(k, "gsig").update(tensor="out"),
        "'out' is not a sync tensor",
    ),
    (
        "tagged",
        "a data event on a sync tensor",
        lambda k: gs_ev(k, "gstore").update(tensor="mail"),
        "'mail' is a sync tensor: only sync events access it",
    ),
    (
        "tagged",
        "a tag of width 64",
        lambda k: gs_ev(k, "gwait").update(target=c(1, 64)),
        "a tag has w=32",
    ),
    (
        "tagged",
        "a take binding a var without its width",
        lambda k: gs_ev(k, "gwait").pop("w"),
        "a wait binds 'var' and 'w' together",
    ),
    (
        "tagged",
        "a sync tensor of width 16",
        lambda k: k["tensors"][0].update(sync={"width": 16}, elem_bytes=2),
        'expected {"width": 32 | 64}',
    ),
    (
        "tagged",
        "a negative co-resident capacity",
        lambda k: k["case"]["inputs"][0].update(capacity={"ctas": -1}),
        "capacity.ctas",
    ),
    (
        "tagged",
        "sync groups that are not lists of CTA indices",
        lambda k: k["case"]["inputs"][0].update(sync_groups=[0, 1]),
        "expected lists of non-negative ints (CTA indices)",
    ),
    (
        "claims",
        "claims without the representative schedule",
        lambda k: k.pop("representative"),
        "a KIR with gclaim events names 'claims'",
    ),
    (
        "claims",
        "a representative schedule other than claims",
        lambda k: k.update(representative="claims"),
        "expected a list of 'claims'",
    ),
    (
        "claims",
        "a claim bound at another width than its word's",
        lambda k: gs_ev(k, "gclaim").update(w=64),
        "must be the sync word's width 32",
    ),
    (
        "claims",
        "a data value half on a 32-bit word",
        lambda k: k["roles"][0]["body"].append(
            {
                "op": "event",
                "ev": "gsig",
                "loc": "x:1",
                "tensor": "ctr",
                "index": c(0, 32),
                "value": c(0, 32),
                "data_lo": True,
            }
        ),  # fmt: skip
        "only on a 64-bit sync word",
    ),
    (
        "relay",
        "a relay read of width 16",
        lambda k: gs_ev(k, "rget").update(w=16),
        "must be 1, 32 or 64",
    ),
]


# P1 KIR fixtures (tests/lean/p1, references/kir.md change 10)
def p1(name):
    with open(os.path.join(HERE, "p1", f"{name}.kir.json")) as f:
        return json.load(f)


# P0-6 KIR fixtures (tests/lean/nb, references/kir.md change 14)
def nb(name):
    with open(os.path.join(HERE, "nb", f"{name}.kir.json")) as f:
        return json.load(f)


def nb_body(k):
    return k["roles"][0]["body"]


NB_MUTATIONS = [
    (
        "uniform",
        "nb_open outside a P0-6 KIR",
        lambda k: (
            k.pop("nbar"),
            nb_body(k).append(
                {
                    "op": "event",
                    "ev": "nb_open",
                    "kind": "cond",
                    "certain": True,
                    "why": "w",
                    "id": None,
                    "count": None,
                    "sync": True,
                    "loc": "x.py:1",
                }
            ),
        ),
        "outside a P0-6 KIR",
    ),  # fmt: skip
    (
        "uniform",
        "an mbarrier event in a P0-6 KIR",
        lambda k: nb_body(k).append(
            {"op": "event", "ev": "tmem_wait", "ld": True, "loc": "x.py:1"}
        ),
        "in a P0-6 KIR",
    ),  # fmt: skip
    (
        "rep_private",
        "nb_rep begin without its end",
        lambda k: nb_body(k).pop(2),
        "without an end",
    ),
    (
        "open_certain",
        "nb_open kind",
        lambda k: nb_body(k)[0].update(kind="spin"),
        "must be one of",
    ),
    (
        "open_certain",
        "nb_open id not const",
        lambda k: nb_body(k)[0].update(id={"e": "var", "w": 32, "name": "x"}),
        "null or a const Expr",
    ),  # fmt: skip
    (
        "uniform",
        "named_bar count 100",
        lambda k: nb_body(k)[0]["body"][0].update(count=c(100)),
        "positive multiple of 32",
    ),
    (
        "uniform",
        "a declared barrier in a P0-6 KIR",
        lambda k: k["barriers"].append({"name": "b", "count": c(1), "num": c(1)}),
        "declares no barriers",
    ),
    (
        "meta_stop",
        "a non-metadata tensor in a P0-6 KIR",
        lambda k: k["tensors"][0].update(metadata=False),
        "only 1-D metadata tensors",
    ),
    (
        "meta_stop",
        "a 2-D metadata tensor in a P0-6 KIR",
        lambda k: k["tensors"][0].update(shape=[c(2), c(1)], strides=[c(1), c(1)]),
        "only 1-D metadata tensors",
    ),  # fmt: skip
    (
        "meta_stop",
        "a gload in a P0-6 KIR",
        lambda k: nb_body(k).append(
            {
                "op": "event",
                "ev": "gload",
                "tensor": "meta",
                "offsets": c(0),
                "mask": None,
                "loc": "x.py:1",
            }
        ),  # fmt: skip
        "gload in a P0-6 KIR",
    ),
    (
        "meta_stop",
        "a gstore in a P0-6 KIR",
        lambda k: nb_body(k).append(
            {
                "op": "event",
                "ev": "gstore",
                "tensor": "meta",
                "offsets": c(0),
                "mask": None,
                "loc": "x.py:1",
            }
        ),  # fmt: skip
        "gstore in a P0-6 KIR",
    ),
    (
        "meta_stop",
        "a gatomic in a P0-6 KIR",
        lambda k: nb_body(k).append(
            {
                "op": "event",
                "ev": "gatomic",
                "tensor": "meta",
                "offsets": c(0),
                "mask": None,
                "loc": "x.py:1",
            }
        ),  # fmt: skip
        "gatomic in a P0-6 KIR",
    ),
    (
        "uniform",
        "a P0-6 KIR naming P1 specs",
        lambda k: k.update(p1=["P1-2"]),
        "both 'nbar'",
    ),
    (
        "all_exit",
        "named_bar all not a boolean",
        lambda k: nb_body(k)[0].update({"all": 1}),
        "expected a boolean",
    ),
    (
        "all_exit",
        "an arrive without a thread count",
        lambda k: nb_body(k)[0].update(sync=False),
        "always names a thread count",
    ),
]


# P1-2 KIR fixtures (tests/lean/p1_2, references/kir.md change 15)
def p1_2(name):
    with open(os.path.join(HERE, "p1_2", f"{name}.kir.json")) as f:
        return json.load(f)


def drop(e, *fields):  # the event without `fields`
    for f in fields:
        e.pop(f)


def first(
    k, ev, field=None
):  # the first event `ev` (with `field`) of the KIR (depth first)
    def walk(stmts):
        for x in stmts:
            if x["op"] == "event" and x["ev"] == ev and (field is None or field in x):
                return x
            for key in ("then", "else", "body"):
                got = walk(x.get(key, []))
                if got is not None:
                    return got
        return None

    return next(e for r in k["roles"] if (e := walk(r["body"])) is not None)


def p1_event_in_p0(k):  # a P1 event in a P0 KIR (no top-level tmem)
    k["roles"][0]["body"].append(
        {"op": "event", "ev": "tmem_wait", "ld": True, "loc": "x.py:1"}
    )


def drain_arrive(k):  # s2t_drain: the compute role's (2 warps) S_drain arrive
    body = k["roles"][1]["body"][0]["body"]
    return next(s for s in body if s.get("ev") == "arrive" and s["bar"] == "S_drain")


def mma_events(k):  # same_role: the MMA of input mma_d_cp
    return k["roles"][0]["body"][2]["then"][0]


def cp_event(k):  # same_role: the first cp of input cp_cp
    return k["roles"][0]["body"][1]["then"][0]


def producer(k):
    return k["roles"][0]["body"]


def loop_body(k):
    return producer(k)[1]["body"]


def store_event(k):  # floor_tile_tail: the gstore
    return k["roles"][0]["body"][1]


def spill_then(k):  # tma_store_spill: the guarded block
    return k["roles"][0]["body"][4]["then"]


def nest_event(k):
    ev = loop_body(k)[2]
    loop_body(k)[2] = {
        "op": "event",
        "event": {key: v for key, v in ev.items() if key != "op"},
    }


def branch_local_use(k):
    k["roles"][1]["body"] += [
        {
            "op": "if",
            "cond": c(1, 1),
            "then": [{"op": "let", "var": "tmp", "expr": c(1)}],
            "else": [],
        },
        {"op": "let", "var": "use", "expr": {"e": "var", "name": "tmp", "w": 32}},
    ]


def relay_copy(k):  # cluster_dsmem_relay: the relay's complete_tx
    return k["roles"][0]["body"][0]["body"][3]


def remote_arrive(k):  # cluster_remote_arrive: the worker's arrive
    return k["roles"][0]["body"][0]


# (fixture, description, mutation, expected substring of some error)
def assume_stmt(k):
    return next(x for x in k["roles"][0]["body"] if x["op"] == "assume")


def role(k, name):  # the role of that name
    return next(r for r in k["roles"] if r["name"] == name)


def first_ev(stmts, ev):  # the first `ev` event of stmts (nested blocks included)
    for s in stmts:
        if s.get("ev") == ev:
            return s
        for key in ("body", "then", "else"):
            x = first_ev(s.get(key, []), ev)
            if x is not None:
                return x
    return None


def put_ev(k):  # handoff: the putter's first put
    return first_ev(role(k, "putter")["body"], "handoff_put")


def get_ev(k):  # handoff: the reader's first get
    return first_ev(role(k, "reader")["body"], "handoff_get")


def split_putter(k):  # handoff: the putter as two parts of one role group
    b = copy.deepcopy(role(k, "putter"))
    b["name"] = "putter_b"
    k["roles"].append(b)
    for r in k["roles"][2:]:
        r["group"] = "putter"


MUTATIONS = [
    (
        "handoff",
        "a handoff channel put by two roles (KIR change 17)",
        lambda k: role(k, "default")["body"].append(copy.deepcopy(put_ev(k))),
        "handoff channel 'mbox' is put by two roles",
    ),
    (
        "handoff",
        "a handoff channel no role puts",
        lambda k: get_ev(k).update(chan="other"),
        "handoff channel 'other' is put by no role",
    ),
    (
        "handoff",
        "a putter that reads a handoff channel",
        lambda k: role(k, "putter")["body"].append(copy.deepcopy(get_ev(k))),
        "a role that puts a handoff channel reads one",
    ),
    (
        "handoff",
        "a handoff value of width 64",
        lambda k: put_ev(k).update(value=c(1, 64)),
        "must have w=32",
    ),
    (
        "handoff",
        "a role group that is not a string",
        lambda k: role(k, "putter").update(group=3),
        "expected a string",
    ),
    (
        "assume",
        "an assume with an i32 condition (KIR change 16)",
        lambda k: assume_stmt(k).update(cond=c(1)),
        "condition must have w=1",
    ),
    (
        "assume",
        "an assume without a message",
        lambda k: assume_stmt(k).pop("msg"),
        "missing field 'msg'",
    ),
    (
        "assume",
        "an assume with a non-string message",
        lambda k: assume_stmt(k).update(msg=3),
        "expected a string",
    ),
    (
        "cluster_pair_tma",
        "cta_mask without a cluster",
        lambda k: k.pop("cluster"),
        "cta_mask without a cluster of more than one CTA",
    ),
    (
        "cluster_pair_tma",
        "cluster dims that are not positive",
        lambda k: k.update(cluster=[0, 1, 1]),
        "$.cluster: expected 3 positive ints",
    ),
    (
        "cluster_pair_tma",
        "a pair leader in a cluster of 3",
        lambda k: k.update(cluster=[3, 1, 1]),
        "a pair leader needs an even cluster size",
    ),
    (
        "cluster_remote_arrive",
        "an i64 CTA rank",
        lambda k: remote_arrive(k).update(cta=c(0, 64)),
        "must have w=32",
    ),
    (
        "cluster_dsmem_relay",
        "complete_tx without a CTA rank",
        lambda k: relay_copy(k).pop("cta"),
        "missing field 'cta'",
    ),
    (
        "cluster_dsmem_relay",
        "complete_tx in a cluster of one CTA",
        lambda k: k.update(cluster=[1, 1, 1]),
        "complete_tx without a cluster of more than one CTA",
    ),
    (
        "pipeline_ok",
        "unknown top-level field",
        lambda k: k.update(extra=1),
        "unknown field 'extra'",
    ),
    (
        "pipeline_ok",
        "unknown target",
        lambda k: k.update(target="sm_90"),
        "$.target: must be one of ('sm_100', 'sm_103')",
    ),
    (
        "pipeline_ok",
        "compiler without a version",
        lambda k: k.update(compiler={"name": "triton"}),
        "$.compiler: missing field 'version'",
    ),
    (
        "pipeline_ok",
        "compiler version that is not a string",
        lambda k: k.update(compiler={"name": "triton", "version": 3}),
        "$.compiler.version: expected a string",
    ),
    (
        "pipeline_ok",
        "wrong kir_version",
        lambda k: k.update(kir_version=2),
        "$.kir_version: must be 1",
    ),
    (
        "pipeline_ok",
        "unsupported width",
        lambda k: k["grid"][1].update(w=16),
        "'w' must be one of",
    ),
    (
        "pipeline_ok",
        "unknown bin op",
        lambda k: k["grid"][0].update(o="pow"),
        "unknown bin op 'pow'",
    ),
    (
        "pipeline_ok",
        "literal too wide",
        lambda k: k["grid"].__setitem__(1, c(1 << 32)),
        "does not fit in 32 bits",
    ),
    (
        "pipeline_ok",
        "pid in a host expression",
        lambda k: k["grid"].__setitem__(1, {"e": "pid", "axis": 0, "w": 32}),
        "'pid' is not allowed in a host-side expression",
    ),
    (
        "pipeline_ok",
        "unbound variable",
        lambda k: producer(k)[0].update(expr={"e": "var", "name": "nope", "w": 32}),
        "unbound variable 'nope'",
    ),
    (
        "pipeline_ok",
        "operand width mismatch",
        lambda k: producer(k)[0]["expr"].update(b=c(64, 64)),
        "operand b has w=64, node has w=32",
    ),
    (
        "pipeline_ok",
        "ext that does not widen",
        lambda k: producer(k)[0].update(
            expr={"e": "ext", "signed": True, "w": 32, "a": c(1)}
        ),
        "ext must widen",
    ),
    (
        "pipeline_ok",
        "assignment to a param",
        lambda k: producer(k).insert(0, {"op": "let", "var": "N", "expr": c(1)}),
        "'N' is a param or for variable and cannot be assigned",
    ),
    (
        "pipeline_ok",
        "assignment to a for variable",
        lambda k: loop_body(k).append({"op": "let", "var": "k", "expr": c(0)}),
        "'k' is a param or for variable and cannot be assigned",
    ),
    (
        "pipeline_ok",
        "assignment changing the width",
        lambda k: loop_body(k).append({"op": "let", "var": "s", "expr": c(0, 64)}),
        "assignment changes the width of 's' from 32 to 64",
    ),
    (
        "pipeline_ok",
        "use of a branch-local variable",
        branch_local_use,
        "unbound variable 'tmp'",
    ),
    ("pipeline_ok", "nested event object", nest_event, "expected 'ev' in"),
    (
        "pipeline_ok",
        "event without loc",
        lambda k: loop_body(k)[5].pop("loc"),
        "missing field 'loc'",
    ),
    (
        "pipeline_ok",
        "unknown barrier",
        lambda k: loop_body(k)[2].update(bar="nope"),
        "unknown barrier 'nope'",
    ),
    (
        "pipeline_ok",
        "TMA on an unknown descriptor",
        lambda k: loop_body(k)[5].update(desc="nope"),
        "'nope' is not a desc tensor",
    ),
    (
        "pipeline_ok",
        "TMA coordinate rank",
        lambda k: k["roles"][1]["body"][1]["coords"].pop(),
        "rank 1 != tensor rank 2",
    ),
    (
        "pipeline_ok",
        "desc tensor without box",
        lambda k: k["tensors"][0].pop("box"),
        "missing field 'box'",
    ),
    (
        "pipeline_ok",
        "input without a param",
        lambda k: k["case"]["inputs"][0]["params"].pop("K_TILES"),
        "missing param 'K_TILES'",
    ),
    (
        "pipeline_ok",
        "param out of range",
        lambda k: k["case"]["inputs"][0]["params"].update(N=1 << 31),
        "does not fit in 32 bits",
    ),
    (
        "pipeline_ok",
        "intended writes on an input tensor",
        lambda k: k["case"]["inputs"][0].update(intended_writes={"a": "all"}),
        "not an output tensor",
    ),
    (
        "pipeline_ok",
        "all_rows_of_segments without segments",
        lambda k: k["case"]["inputs"][0].update(
            intended_writes={"out": "all_rows_of_segments"}
        ),
        "needs a tensor with segments",
    ),
    (
        "pipeline_ok",
        "unknown intended_writes value (the opt-out is spelled unchecked)",
        lambda k: k["case"]["inputs"][0].update(intended_writes={"out": "none"}),
        "expected one of ('all_rows_of_segments', 'all', 'unchecked')",
    ),
    (
        "trunc_offset_wrap",
        "addr on trunc that is not a boolean",
        lambda k: k["roles"][0]["body"][1]["expr"].update(addr=1),
        "addr: expected a boolean",
    ),
    (
        "floor_tile_tail",
        "mask that is not i1",
        lambda k: store_event(k)["mask"].update(w=32),
        "mask must have w=1",
    ),
    (
        "floor_tile_tail",
        "range shape",
        lambda k: store_event(k)["offsets"]["b"]["a"]["a"].update(end=8),
        "range shape must be [8]",
    ),
    (
        "floor_tile_tail",
        "bcast to a wrong shape",
        lambda k: store_event(k)["offsets"]["b"].update(axis=1),
        "cannot give shape",
    ),
    (
        "tma_store_spill",
        "segment field no work event defines",
        lambda k: spill_then(k).pop(0),
        "no work event defines field 'batch'",
    ),
    (
        "tma_store_spill",
        "missing metadata contents",
        lambda k: k["case"]["inputs"][0].pop("metadata"),
        "missing contents of metadata tensor 'cu_seqlens'",
    ),
    (
        "tma_store_spill",
        "tma_store to a tensor declared input",
        lambda k: next(t for t in k["tensors"] if t["name"] == "out").update(
            role="input"
        ),
        "tma_store writes 'out', declared role 'input'",
    ),
    (
        "floor_tile_tail",
        "gstore to a tensor declared input",
        lambda k: next(t for t in k["tensors"] if t["name"] == "out").update(
            role="input"
        ),
        "gstore writes 'out', declared role 'input'",
    ),
    (
        "tma_store_spill",
        "load from a non-metadata tensor",
        lambda k: k["roles"][0]["body"][1]["expr"].update(tensor="out"),
        "'out' is not a metadata tensor",
    ),
    (
        "pipeline_ok",
        "region event (removed: P0-5 takes the case's elision_flags)",
        lambda k: loop_body(k).append(
            {"op": "event", "ev": "region", "name": "FULL_TILES", "loc": "x.py:1"}
        ),
        "expected 'ev' in",
    ),
    (
        "pipeline_ok",
        "elision_flags that is not a list",
        lambda k: k["case"].update(elision_flags="FULL_TILES"),
        "$.case.elision_flags: expected a list of distinct flag names",
    ),
    (
        "pipeline_ok",
        "elision flag that is not a string",
        lambda k: k["case"].update(elision_flags=[1]),
        "$.case.elision_flags: expected a list of distinct flag names",
    ),
    (
        "pipeline_ok",
        "repeated elision flag",
        lambda k: k["case"].update(elision_flags=["FULL_TILES", "FULL_TILES"]),
        "$.case.elision_flags: expected a list of distinct flag names",
    ),
    (
        "pipeline_ok",
        "elision_flags on an input (a case-level key)",
        lambda k: k["case"]["inputs"][0].update(elision_flags=["FULL_TILES"]),
        "unknown field 'elision_flags'",
    ),
]

# (fixture, description, mutation): valid KIR after the mutation
ACCEPTED = [
    ("handoff", "the putters of a channel: the parts of one role group", split_putter),
    ("pipeline_ok", "target sm_103", lambda k: k.update(target="sm_103")),
    (
        "floor_tile_tail",
        "gstore to a workspace tensor",
        lambda k: (
            next(t for t in k["tensors"] if t["name"] == "out").update(
                role="workspace"
            ),
            k["case"]["inputs"][0].pop("intended_writes", None),
            [i.pop("intended_writes", None) for i in k["case"]["inputs"]],
        ),
    ),
    (
        "pipeline_ok",
        "compiler",
        lambda k: k.update(compiler={"name": "nvidia-cutlass-dsl", "version": "4.4.2"}),
    ),
    (
        "pipeline_ok",
        "elision flags no input names",
        lambda k: k["case"].update(elision_flags=["FULL_TILES", "EVEN_N"]),
    ),
]


P1_MUTATIONS = [
    (
        "s2t_race",
        "zero base with 256 columns",
        lambda k: k["tmem"].update(cols=256),
        "'zero' (absolute addresses) needs",
    ),
    (
        "s2t_race",
        "tmem without P1 events",
        lambda k: k.update(roles=[{**k["roles"][0], "body": []}]),
        "a top-level 'tmem' without",
    ),
    (
        "s2t_race",
        "cp in a 2-warp role",
        lambda k: k["roles"][0].update(warps=2),
        "cp/mma roles must be one warp",
    ),
    (
        "s2t_drain",
        "one arrival per arrive of a 2-warp role (a thread-0 arrive)",
        lambda k: drain_arrive(k).pop("per_warp"),
        "1 arrival(s) per arrive in a role of 2 warps",
    ),
    (
        "same_role",
        "unknown rect use",
        lambda k: cp_event(k)["rects"][0].update(use="meta"),
        "must be one of ('data'",
    ),
    (
        "named_bars",
        "non-constant named_bar id",
        lambda k: k["roles"][0]["body"][2]["then"][0].update(
            id={"e": "var", "name": "MODE", "w": 32}
        ),
        "named barrier id and count must be const Exprs",
    ),
    (
        "same_role",
        "mma without a d rect",
        lambda k: mma_events(k).update(
            rects=mma_events(k)["rects"][:0]
            + [{**mma_events(k)["rects"][0], "use": "a", "write": False}]
        ),
        "an mma has exactly one d write",
    ),
    (
        "same_role",
        "ld writing",
        lambda k: cp_event(k).update(inst="ld"),
        "an ld has only data reads",
    ),
    (
        "named_bars",
        "named_bar count 48",
        lambda k: k["roles"][0]["body"][2]["then"][0].update(count=c(48)),
        "positive multiple of 32",
    ),
    (
        "named_bars",
        "named_bar all in a P1 KIR",
        lambda k: k["roles"][0]["body"][2]["then"][0].update({"all": True}),
        "only in a P0-6 KIR",
    ),
]


P1_2_MUTATIONS = [
    (
        "same_thread",
        "P1-2 events without p1",
        lambda k: k.pop("p1"),
        "P1-2 events in a KIR whose 'p1' does not name P1-2",
    ),
    (
        "same_thread",
        "p1 naming P1-1 without tmem",
        lambda k: k.update(p1=["P1-1", "P1-2"]),
        "names P1-1 iff the KIR has a top-level 'tmem'",
    ),
    (
        "same_thread",
        "unknown p1 label",
        lambda k: k.update(p1=["P1-3"]),
        "expected a non-empty list of distinct labels",
    ),
    (
        "same_thread",
        "bulk_store of an unknown kind",
        lambda k: first(k, "bulk_store").update(kind="gather"),
        "must be one of ('tensor'",
    ),
    (
        "same_thread",
        "a warp-collective write with a lane",
        lambda k: first(k, "smem_write").update(sync=True, lane=c(3)),
        "a warp-collective write (sync) has no lane",
    ),
    (
        "same_thread",
        "bulk_wait with a negative count",
        lambda k: first(k, "bulk_wait").update(n=-1),
        ".n: must be >= 0",
    ),
    (
        "landing",
        "a landing's mbarrier slot on a write that is not async",
        lambda k: first(k, "smem_write", "bar").pop("async"),
        "bar and idx name the mbarrier of an async landing",
    ),
    (
        "landing",
        "a landing's mbarrier without its index",
        lambda k: first(k, "smem_write", "bar").pop("idx"),
        "bar and idx name the mbarrier of an async landing",
    ),
    (
        "pair_chain",
        "a landing's leader without its mbarrier slot",
        lambda k: drop(first(k, "smem_write", "leader"), "bar", "idx"),
        "bar and idx name the mbarrier of an async landing",
    ),
    (
        "pair_chain",
        "a landing's leader that is no boolean",
        lambda k: first(k, "smem_write", "leader").update(leader=1),
        "leader",
    ),
    (
        "staging",
        "cta_threads not a multiple of 32",
        lambda k: k.update(cta_threads=100),
        "positive multiple of 32",
    ),
    (
        "staging",
        "cta_threads below the roles' threads",
        lambda k: k.update(cta_threads=64),
        "at least the roles' threads",
    ),
]


def main():
    failed = 0
    for name, what, mutate, expect in P1_2_MUTATIONS:
        k = copy.deepcopy(p1_2(name))
        assert not kir.validate(k), f"p1_2/{name} must be valid before mutation"
        mutate(k)
        errors = kir.validate(k)
        if not any(expect in e for e in errors):
            failed += 1
            print(
                f"kir_negative: NOT REJECTED as expected: p1_2/{name}: {what}: want {expect!r}, got {errors}"
            )
    k = copy.deepcopy(fixture("pipeline_ok"))
    p1_event_in_p0(k)
    if not any("without a top-level 'tmem'" in e for e in kir.validate(k)):
        failed += 1
        print(
            "kir_negative: NOT REJECTED as expected: pipeline_ok: a P1 event in a P0 KIR"
        )
    for name, what, mutate, expect in P1_MUTATIONS:
        k = copy.deepcopy(p1(name))
        assert not kir.validate(k), f"p1/{name} must be valid before mutation"
        mutate(k)
        errors = kir.validate(k)
        if not any(expect in e for e in errors):
            failed += 1
            print(
                f"kir_negative: NOT REJECTED as expected: p1/{name}: {what}: want {expect!r}, got {errors}"
            )
    for name, what, mutate, expect in NB_MUTATIONS:
        k = copy.deepcopy(nb(name))
        assert not kir.validate(k), f"nb/{name} must be valid before mutation"
        mutate(k)
        errors = kir.validate(k)
        if not any(expect in e for e in errors):
            failed += 1
            print(
                f"kir_negative: NOT REJECTED as expected: nb/{name}: {what}: want {expect!r}, got {errors}"
            )
    for name, what, mutate, expect in GSYNC_MUTATIONS:
        k = copy.deepcopy(gs(name))
        assert not kir.validate(k), f"gsync/{name} must be valid before mutation"
        mutate(k)
        errors = kir.validate(k)
        if not any(expect in e for e in errors):
            failed += 1
            print(
                f"kir_negative: NOT REJECTED as expected: gsync/{name}: {what}: want {expect!r}, got {errors}"
            )
    for name, what, mutate in ACCEPTED:
        k = copy.deepcopy(fixture(name))
        mutate(k)
        errors = kir.validate(k)
        if errors:
            failed += 1
            print(f"kir_negative: REJECTED but valid: {name}: {what}: {errors}")
    for name, what, mutate, expect in MUTATIONS:
        k = copy.deepcopy(fixture(name))
        assert not kir.validate(k), f"{name} must be valid before mutation"
        mutate(k)
        errors = kir.validate(k)
        if not any(expect in e for e in errors):
            failed += 1
            print(
                f"kir_negative: NOT REJECTED as expected: {name}: {what}: want {expect!r}, got {errors}"
            )
    nmut = (
        len(MUTATIONS) + len(P1_MUTATIONS) + len(P1_2_MUTATIONS) + len(NB_MUTATIONS)
        + len(GSYNC_MUTATIONS) + 1
    )  # fmt: skip
    print(
        f"kir_negative: {nmut + len(ACCEPTED) - failed}/{nmut + len(ACCEPTED)} as expected"
        f" ({nmut} invalid KIR variants rejected with the expected error, {len(ACCEPTED)} valid ones accepted)"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
