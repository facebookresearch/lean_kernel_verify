# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PTX op table of the PTX frontend: every normalized mnemonic -> one class.

    mnemonic(opcode) -> str              classify(instr) -> (class, reason)

Classes:

    event    becomes a KIR event or feeds an event model: global memory, mbarrier, TMA
             and bulk copies, tcgen05, CLC, thread and cluster barriers, in-kernel
             tensormaps, device assert
    value    defines a register the translator evaluates exactly: integer arithmetic,
             comparison, selection, moves, integer conversion, parameter loads, warp
             collectives
    control  ends a basic block: bra, brx.idx, ret
    drop     affects no spec: floating-point data, shared-memory data movement, fences,
             setmaxnreg, group bookkeeping. A dropped result is Data; the slicer rejects
             it if it ever reaches a root, so a wrong `drop` cannot hide an event
    refuse   never translated: the variant stops with the reason

So far the table only classifies; the KIR mapping and issue rules of each row come with
the translator. Rows follow the PTX ISA only (reasons name ISA sections, not table
numbers). A mnemonic without a row raises UnknownInstruction: the table grows by
deciding a row, never by a default.

Normalization (the spellings of LLVM, NVVM 20 and NVVM 23 meet on one mnemonic):
  - integer and bit types (.b32, .u64, .s16, .pred, .b128, .b1024), vector widths (.v4),
    repetition counts (.x16), cache hints (.L2::cache_hint) and spelled-out defaults
    (bra.uni's .uni, the .tile load mode, setp's comparison) are dropped;
  - an arithmetic, comparison, move or conversion op with any floating-point type
    collapses to `<op>.f` (all of them are drop);
  - .shared::cta is .shared and .param::entry is .param; mbarrier's default .sem.scope
    (.release.cta for arrive, .acquire.cta for test/try_wait, .relaxed.cta for
    expect_tx/complete_tx) is dropped; tcgen05.mma's .scale_vec::1X/2X/4X is
    .block32/.block32/.block16; tcgen05.cp's .warpx* follows the shape.
E.g. `mbarrier.arrive.release.cta.shared::cta.b64` (NVVM 20), `mbarrier.arrive.shared.b64`
(NVVM 23) and `mbarrier.arrive.shared::cta.b64` (LLVM) are all `mbarrier.arrive.shared`.
"""

import re

CLASSES = ("event", "value", "control", "drop", "refuse")


class UnknownInstruction(Exception):
    pass


_INT_TYPE = re.compile(r"[bsu](?:8|16|32|64|128)|b1024|pred")
_FLOAT_TYPE = re.compile(
    r"(?:b?f16|f32|f64|tf32)(?:x2)?|(?:e4m3|e5m2|e2m1|e2m3|e3m2|ue8m0)x2"
)
_DROPPED = re.compile(r"[vx]\d+|L[12]::\w+|tile")
_CMP = {
    "eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs",
    "equ", "neu", "ltu", "leu", "gtu", "geu", "num", "nan",
}  # fmt: skip
_ALIAS = {
    "shared::cta": "shared",
    "param::entry": "param",
    "scale_vec::1X": "block32",
    "scale_vec::2X": "block32",
    "scale_vec::4X": "block16",
}
_MBAR_DEFAULT = {
    "arrive": "release.cta",
    "test_wait": "acquire.cta",
    "try_wait": "acquire.cta",
    "expect_tx": "relaxed.cta",
    "complete_tx": "relaxed.cta",
}
_UNI = ("bra", "brx", "call", "ret")  # .uni: the same transfer, asserted non-divergent
# Ops whose class depends on the type: integer is exact value, float is Data.
FLOAT_OPS = (
    "add", "sub", "mul", "mad", "fma", "div", "rem", "abs", "neg", "min", "max", "setp",
    "set", "selp", "slct", "mov", "cvt", "redux", "testp", "copysign", "rcp", "sqrt",
    "rsqrt", "ex2", "lg2", "sin", "cos", "tanh",
)  # fmt: skip


def mnemonic(opcode):
    base, *mods = opcode.split(".")
    if base in FLOAT_OPS and any(_FLOAT_TYPE.fullmatch(t) for t in mods):
        return base + ".f"
    out = [base]
    for t in mods:
        t = _ALIAS.get(t, t)
        if _INT_TYPE.fullmatch(t) or _FLOAT_TYPE.fullmatch(t) or _DROPPED.fullmatch(t):
            continue
        if (base in ("setp", "set") and t in _CMP) or (base in _UNI and t == "uni"):
            continue
        out.append(t)
    if base == "mbarrier" and len(out) > 1 and out[1] in _MBAR_DEFAULT:
        default = _MBAR_DEFAULT[out[1]].split(".")
        for i in range(2, len(out) - 1):
            if out[i : i + 2] == default:
                del out[i : i + 2]
                break
    # tcgen05.cp: LLVM writes .warpx4.32x128b, the ISA (and NVVM) .32x128b.warpx4.
    if out[:2] == ["tcgen05", "cp"]:
        warps = [t for t in out if t.startswith("warpx")]
        out = [t for t in out if not t.startswith("warpx")]
        at = next(
            (i + 1 for i, t in enumerate(out) if re.fullmatch(r"\d+x\d+b", t)), len(out)
        )
        out[at:at] = warps
    return ".".join(out)


_RED_OPS = ("add", "min", "max", "inc", "dec", "and", "or", "xor")
_CTA = ("cta_group::1", "cta_group::2")
_TMA_LOAD_DST = ("shared", "shared::cluster")
_TMA_LOAD_OPT = (
    "",
    ".multicast::cluster",
    ".cta_group::2",
    ".multicast::cluster.cta_group::2",
)

# (class, reason, mnemonics): one row per family.
_ROWS = [
    (
        "value",
        "integer arithmetic, logic and shift (ISA: Integer Arithmetic, Logic and Shift "
        "Instructions): exact KIR expressions",
        "add sub mul.lo mul.hi mul.wide mad.lo mad.wide div rem min max neg and or xor not "
        "shl shr popc clz brev bfe bfi prmt lop3 shf.l.wrap shf.r.wrap shf.l.clamp "
        "shf.r.clamp".split(),
    ),
    (
        "value",
        "comparison, selection, moves and integer conversion (ISA: Comparison and "
        "Selection, Data Movement and Conversion Instructions)",
        ["setp", "selp", "mov", "cvt", "cvta.param", "cvta.global"],
    ),
    (
        "value",
        "kernel parameter read (ISA: ld, .param state space): a host value",
        ["ld.param"],
    ),
    (
        "value",
        "warp collectives (ISA: elect.sync, shfl.sync, vote.sync, redux.sync), evaluated "
        "per thread by the thread model",
        ["elect.sync"]
        + [f"shfl.sync.{m}" for m in ("idx", "up", "down", "bfly")]
        + [f"vote.sync.{m}" for m in ("ballot", "any", "all", "uni")]
        + [f"redux.sync.{op}" for op in ("add", "min", "max", "and", "or", "xor")],
    ),
    (
        "value",
        "address of a peer CTA's shared memory (ISA: mapa): the event's cta",
        ["mapa.shared::cluster"],
    ),
    (
        "drop",
        "floating-point data (ISA: Floating-Point, Half Precision and Mixed Precision "
        "Instructions; cvt to or from a float type): the result is Data",
        [f"{op}.f" for op in FLOAT_OPS],
    ),
    (
        "event",
        "global memory access per thread (ISA: ld, st): gload/gstore or a metadata load",
        ["ld.global", "ld", "st.global", "st"],
    ),
    (
        "event",
        "global atomics per thread (ISA: atom, red): gatomic (atom.exch: gstore)",
        [f"atom.global.{op}" for op in _RED_OPS + ("exch", "cas")]
        + [f"red.global.{op}" for op in _RED_OPS],
    ),
    (
        "drop",
        "shared memory data (ISA: ld, st on .shared): the values are Data (the tensormap "
        "staging and CLC response reads are handled by the memory model)",
        ["ld.shared", "st.shared"],
    ),
    (
        "event",
        "mbarrier (ISA: mbarrier.init, mbarrier.arrive, mbarrier.expect_tx, "
        "mbarrier.complete_tx): per-thread counts aggregate to one event",
        [
            "mbarrier.init.shared",
            "mbarrier.arrive.shared",
            "mbarrier.arrive.shared::cluster",
            "mbarrier.arrive.expect_tx.shared",
            "mbarrier.arrive.expect_tx.shared::cluster",
            "mbarrier.expect_tx.shared",
            "mbarrier.complete_tx.shared",
        ],
    ),
    (
        "refuse",
        "try_wait/test_wait outside a recognized wait loop (ISA: mbarrier.test_wait/"
        "mbarrier.try_wait): it may return false spuriously; only the whole loop waits",
        [
            f"mbarrier.{op}{parity}.shared"
            for op in ("try_wait", "test_wait")
            for parity in ("", ".parity")
        ],
    ),
    (
        "event",
        "TMA tensor copy (ISA: cp.async.bulk.tensor, cp.reduce.async.bulk.tensor): "
        "load/reduce per issuing thread, store deduplicated per warp",
        [
            f"cp.async.bulk.tensor.{d}d.{dst}.global.mbarrier::complete_tx::bytes{opt}"
            for d in range(1, 6)
            for dst in _TMA_LOAD_DST
            for opt in _TMA_LOAD_OPT
        ]
        + [f"cp.async.bulk.tensor.{d}d.global.shared.bulk_group" for d in range(1, 6)]
        + [
            f"cp.reduce.async.bulk.tensor.{d}d.global.shared.{op}.bulk_group"
            for d in range(1, 6)
            for op in _RED_OPS
        ],
    ),
    (
        "event",
        "bulk copy (ISA: cp.async.bulk, cp.reduce.async.bulk) per issuing thread",
        [
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes",
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
            ".multicast::cluster",
            "cp.async.bulk.shared.global.mbarrier::complete_tx::bytes",
            "cp.async.bulk.shared::cluster.shared.mbarrier::complete_tx::bytes",
            "cp.async.bulk.global.shared.bulk_group",
        ]
        + [f"cp.reduce.async.bulk.global.shared.bulk_group.{op}" for op in _RED_OPS],
    ),
    (
        "drop",
        "async-group bookkeeping (ISA: cp.async.bulk.commit_group/wait_group, "
        "cp.async.commit_group/wait_group): completion is not a spec event",
        [
            "cp.async.bulk.commit_group",
            "cp.async.bulk.wait_group",
            "cp.async.bulk.wait_group.read",
            "cp.async.commit_group",
            "cp.async.wait_group",
            "cp.async.wait_all",
        ],
    ),
    (
        "event",
        "non-bulk async copy global -> shared per thread (ISA: cp.async): gload",
        ["cp.async.ca.shared.global", "cp.async.cg.shared.global"],
    ),
    (
        "refuse",
        "cp.async.mbarrier.arrive.noinc (ISA: cp.async.mbarrier.arrive): the pending "
        "count is not incremented, so the arrival count depends on the copies",
        ["cp.async.mbarrier.arrive.noinc.shared"],
    ),
    (
        "event",
        "tcgen05 (ISA: TensorCore 5th Generation Family Instructions): TMEM allocation, "
        "ld/st/wait, cp, mma and commit are P1 events; commit arrives on an mbarrier",
        [f"tcgen05.alloc.{c}.sync.aligned.shared" for c in _CTA]
        + [f"tcgen05.dealloc.{c}.sync.aligned" for c in _CTA]
        + [f"tcgen05.relinquish_alloc_permit.{c}.sync.aligned" for c in _CTA]
        + [
            f"tcgen05.{op}.sync.aligned.{shape}"
            for op in ("ld", "st")
            for shape in ("32x32b", "16x64b", "16x128b", "16x256b", "16x32bx2")
        ]
        + ["tcgen05.wait::ld.sync.aligned", "tcgen05.wait::st.sync.aligned"]
        + [
            f"tcgen05.cp.{c}.{shape}"
            for c in _CTA
            for shape in (
                "128x256b",
                "4x256b",
                "128x128b",
                "64x128b.warpx2::02_13",
                "64x128b.warpx2::01_23",
                "32x128b.warpx4",
            )  # fmt: skip
        ]
        + [
            f"tcgen05.mma.{c}.kind::{kind}"
            for c in _CTA
            for kind in (
                "f16",
                "tf32",
                "f8f6f4",
                "i8",
                "mxf8f6f4.block_scale.block32",
                "mxf4.block_scale.block32",
                "mxf4nvf4.block_scale.block32",
                "mxf4nvf4.block_scale.block16",
            )  # fmt: skip
        ]
        + [
            f"tcgen05.commit.{c}.mbarrier::arrive::one.shared::cluster{mc}"
            for c in _CTA
            for mc in ("", ".multicast::cluster")
        ],
    ),
    (
        "event",
        "cluster launch control (ISA: clusterlaunchcontrol.try_cancel, "
        "clusterlaunchcontrol.query_cancel): the CLC schedule model",
        [
            "clusterlaunchcontrol.try_cancel.async.shared.mbarrier::complete_tx::bytes",
            "clusterlaunchcontrol.try_cancel.async.shared.mbarrier::complete_tx::bytes"
            ".multicast::cluster::all",
            "clusterlaunchcontrol.query_cancel.is_canceled",
            "clusterlaunchcontrol.query_cancel.get_first_ctaid",
        ],
    ),
    (
        "event",
        "thread and cluster barriers (ISA: bar/barrier, barrier.cluster): executed by "
        "whole roles; named barriers in P1, the cluster barrier model",
        [
            "bar.sync",
            "bar.arrive",
            "barrier.sync",
            "barrier.sync.aligned",
            "barrier.arrive",
            "barrier.arrive.aligned",
        ]
        + [
            f"barrier.cluster.{op}{a}"
            for op in ("arrive", "arrive.relaxed", "wait")
            for a in ("", ".aligned")
        ],
    ),
    (
        "drop",
        "warp convergence only (ISA: bar.warp.sync): no ordering across warps",
        ["bar.warp.sync"],
    ),
    (
        "event",
        "in-kernel tensormap (ISA: tensormap.replace, tensormap.cp_fenceproxy): fields "
        "collected per cp_fenceproxy into a descriptor",
        [
            f"tensormap.replace.{field}.shared"
            for field in (
                "global_address",
                "rank",
                "box_dim",
                "global_dim",
                "global_stride",
                "element_stride",
                "elemtype",
                "interleave_layout",
                "swizzle_mode",
                "swizzle_atomicity",
                "fill_mode",
            )  # fmt: skip
        ]
        + [
            "tensormap.cp_fenceproxy.global.shared.tensormap::generic.release.gpu.sync.aligned"
        ],
    ),
    (
        "control",
        "control flow (ISA: Control Flow Instructions: bra, brx.idx, ret); brx.idx only "
        "in the warp-specialize dispatch",
        ["bra", "brx.idx", "ret"],
    ),
    (
        "drop",
        "argument staging of a call (ISA: st, .param state space): only calls of "
        "__assertfail, vprintf and libdevice are accepted, their arguments are Data",
        ["st.param"],
    ),
    (
        "drop",
        "deliberate abort (ISA: trap): not modeled, the run is evaluated as if it were "
        "not reached, as the CuTe frontend does (references/frontend_cute.md)",
        ["trap"],
    ),
]

# (class, reason, prefixes): families whose modifiers never change the class.
_FAMILIES = [
    (
        "drop",
        "memory ordering only (ISA: membar/fence, tcgen05.fence): the model has no weak "
        "memory",
        ["fence", "membar", "tcgen05.fence"],
    ),
    (
        "drop",
        "no effect on any spec (ISA: griddepcontrol, setmaxnreg, prefetch)",
        ["griddepcontrol", "setmaxnreg", "prefetch"],
    ),
    (
        "drop",
        "shared memory data (ISA: ldmatrix, stmatrix): the values are Data",
        ["ldmatrix", "stmatrix"],
    ),
]

TABLE = {k: (cls, reason) for cls, reason, keys in _ROWS for k in keys}

_LOAD_MODE = (
    "refuse",
    "TMA im2col/gather4/scatter4 modes are not modeled (ISA: cp.async.bulk.tensor)",
)
_WARPID = (
    "refuse",
    "%warpid may change during execution (ISA: Special Registers: %warpid)",
)


def classify(instr):
    """(class, reason) of a parse.Instr; UnknownInstruction if no row decides it."""
    if any(m in instr.opcode for m in ("im2col", "gather4", "scatter4")):
        return _LOAD_MODE
    if ("sreg", "%warpid") in instr.operands:
        return _WARPID
    key = mnemonic(instr.opcode)
    if key == "call":
        return _call(instr)
    if key in TABLE:
        return TABLE[key]
    for cls, reason, prefixes in _FAMILIES:
        for p in prefixes:
            if key.startswith(p) and key[len(p) : len(p) + 1] in ("", ".", ":"):
                return cls, reason
    raise UnknownInstruction(
        f"line {instr.line}: no op table row for {key} ({instr.opcode})"
    )


def _call(instr):
    callee = next((op[1] for op in instr.operands if op[0] == "sym"), None)
    if callee == "__assertfail":
        return "event", "device assert (call __assertfail): KIR assert"
    if callee == "vprintf":
        return "drop", "device printf (call vprintf): affects no spec"
    if callee and callee.startswith("__nv_"):
        return "drop", "libdevice call: the result is Data"
    return (
        "refuse",
        f"call of {callee or 'a register'}: only __assertfail, vprintf and libdevice",
    )
