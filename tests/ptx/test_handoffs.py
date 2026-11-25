# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests of the handoffs through shared memory (frontend/ptx/interp.py, translate.py,
frontend_ptx.md "Handoffs", KIR change 17), through translate().

    python3 -I tests/ptx/test_handoffs.py      (or the buck target :test_handoffs)

A Triton direct dispatch (8 warps: the default 0-3, partition0 = warp 6, partition1 =
warp 7, padding 4-5). partition1 puts n, n + 1, then -1 into a 32-bit word of shared
memory, arriving on `full` after each and waiting on `empty` before the next; partition0
waits on `full`, reads the word, stops at -1 and otherwise stores to x[word], arriving on
`empty`. The first translation refuses partition0's address (data); the second, with the
word a handoff channel, puts and gets it. A word stored by two roles, or with a value that
differs between threads, is no channel (the first refusal stays); another write that may
touch a channel refuses the second translation. The same through stmatrix / ldmatrix and a
warp reduction of the rows. A role that puts a channel gets none; a word no other role loads
after a wait is no channel; an ld.shared.s32 into a 64-bit register sign-extends the value
got; a write without static bounds assumes it misses the channel; a write at the zero extension
of a value that may be negative may overwrite a channel; a channel over an mbarrier array, or
an error in the second translation, refuses the variant.
"""

import sys
import unittest
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import ModuleType
from unittest import mock

if __package__ in (None, ""):
    sys.dont_write_bytecode = True
    _root = ModuleType("lkv_root")
    _root.__path__ = [str(Path(__file__).resolve().parents[2])]
    sys.modules["lkv_root"] = _root
    __package__ = "lkv_root.tests.ptx"  # noqa: A001

from ...frontend import kir  # noqa: E402
from ...frontend.ptx import interp, translate as T  # noqa: E402

SRC = "/src"
BODY = """
    mov.u32 %r1, %tid.x;
    shr.u32 %r2, %r1, 5;
    shfl.sync.idx.b32 %r3, %r2, 0, 31, -1;
    ld.param.u64 %rd1, [k_param_0];
    ld.param.b32 %r20, [k_param_1];
    mov.u32 %r30, global_smem;
    and.b32 %r70, %r1, 7;
    shl.b32 %r71, %r70, 4;
    add.s32 %r72, %r30, %r71;
    setp.lt.u32 %p1, %r3, 4;
    @!%p1 bra $HDR;
    setp.eq.u32 %p5, %r1, 0;
    @%p5 mbarrier.init.shared::cta.b64 [%r30+72], 32;
    @%p5 mbarrier.init.shared::cta.b64 [%r30+80], 32;
    barrier.sync 1;
    barrier.sync 1;
{default}    bra.uni $RET;
$HDR:
    barrier.sync 1;
    setp.gt.u32 %p2, %r3, 6;
    @!%p2 bra $L2;
    setmaxnreg.dec.sync.aligned.u32 24;
    barrier.sync 1;
    mov.u32 %r50, 0;
$PL:
    add.s32 %r51, %r20, %r50;
{put}    mbarrier.arrive.shared::cta.b64 %rd30, [global_smem+72];
    and.b32 %r52, %r50, 1;
$PW:
    mbarrier.try_wait.parity.shared::cta.b64 %p20, [global_smem+80], %r52;
    @!%p20 bra $PW;
    add.s32 %r50, %r50, 1;
    setp.lt.u32 %p21, %r50, 2;
    @%p21 bra $PL;
    mov.u32 %r51, -1;
{put}    mbarrier.arrive.shared::cta.b64 %rd31, [global_smem+72];
    bra.uni $RET;
$L2:
    setp.eq.b32 %p3, %r3, 6;
    @!%p3 bra $PAD;
    setmaxnreg.dec.sync.aligned.u32 24;
    barrier.sync 1;
    mov.u32 %r60, 0;
$RL:
    and.b32 %r61, %r60, 1;
$RW:
    mbarrier.try_wait.parity.shared::cta.b64 %p22, [global_smem+72], %r61;
    @!%p22 bra $RW;
{get}    setp.eq.s32 %p23, %r62, -1;
    @%p23 bra $RET;
    mul.wide.s32 %rd20, %r62, 4;
    add.s64 %rd21, %rd1, %rd20;
    setp.eq.u32 %p24, %r1, 192;
    @%p24 st.global.b32 [%rd21], %r3;
    mbarrier.arrive.shared::cta.b64 %rd32, [global_smem+80];
    add.s32 %r60, %r60, 1;
    bra.uni $RL;
$PAD:
    setmaxnreg.dec.sync.aligned.u32 24;
    barrier.sync 1;
$RET:
    ret;
"""
WORD = {"put": "    st.shared.b32 [global_smem+128], %r51;\n",
        "get": "    ld.shared.b32 %r62, [global_smem+128];\n"}  # fmt: skip
MATRIX = {"put": "    stmatrix.sync.aligned.m8n8.x1.shared.b16 [%r72+128], {%r51};\n",
          "get": "    ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%r63}, [%r72+128];\n"
                 "    redux.sync.max.s32 %r62, %r63, -1;\n"}  # fmt: skip


def run(default: str = "", **hooks: str) -> "T.Facts":
    body = BODY.format(default=default, **{**WORD, **hooks})
    text = f"""
.version 9.3
.target sm_100a
.address_size 64
.extern .shared .align 1024 .b8 global_smem[];
.visible .entry k(.param .u64 .ptr .global .align 1 k_param_0, .param .b32 k_param_1)
.reqntid 256
{{
    .reg .pred %p<32>;
    .reg .b32 %r<100>;
    .reg .b64 %rd<64>;
    .loc 1 1 0
{body}
}}
    .file 1 "{SRC}/k.py"
"""
    params = [{"i": 0, "ptx": "k_param_0", "name": "x", "kind": "ptr", "size": 8, "value": 0x20000},
              {"i": 1, "ptx": "k_param_1", "name": "n", "kind": "scalar", "size": 4, "value": 3}]  # fmt: skip
    recs = [
        {
            "block": [256, 1, 1],
            "cluster": [1, 1, 1],
            "grid": [1, 1, 1],
            "params": params,
        }
    ]
    table = T.param_table(recs, "ttgir", None, [])
    pol = T.policy(
        "ttgir", src_root=SRC, params=table, tensors={"x": {"elem_bytes": 4}}
    )
    return T.translate(text, "k", recs, pol)


def events(f: "T.Facts", ev: str) -> dict:
    """role name -> the `ev` events of its body, in order."""

    def walk(stmts: Iterable[dict]) -> Iterator[dict]:
        for s in stmts:
            if s.get("ev") == ev:
                yield s
            for k in ("then", "else", "body"):
                yield from walk(s.get(k, []))

    return {r["name"]: list(walk(r["body"])) for r in f.roles}


def refusal(f: "T.Facts") -> str:
    return f.refusals[0]["msg"] if f.refusals else ""


def kir_of(f: "T.Facts") -> dict:
    """A KIR of translation f's roles (for kir.validate)."""
    return {"kir_version": 1, "kernel": "k", "source": "k.py", "ir": "ptx",
            "grid": [{"e": "const", "w": 32, "v": 1}] * 3,
            "params": [{"name": n, "type": f"i{w}"} for n, w in f.params.items()],
            "tensors": [{"name": "x", "kind": "global", "elem_bytes": 4,
                         "shape": [{"e": "const", "w": 32, "v": 64}],
                         "strides": [{"e": "const", "w": 32, "v": 1}],
                         "segments": None, "role": "workspace", "metadata": False}],
            "barriers": f.barriers,
            "roles": [{k: r[k] for k in ("name", "warps", "threads_per_issue", "body", "group")
                       if k in r} for r in f.roles],
            "case": {"name": "k", "inputs": [{"name": "a", "params": {"n": 3, "x": 0}}]}}  # fmt: skip


def _exprs(x: object) -> Iterator[dict]:
    """Every dict nested in x."""
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from _exprs(v)
    elif isinstance(x, list):
        for v in x:
            yield from _exprs(v)


class Handoffs(unittest.TestCase):
    def check_channel(self, f: "T.Facts") -> None:
        self.assertEqual(f.refusals, [])
        puts, gets = events(f, "handoff_put"), events(f, "handoff_get")
        self.assertEqual([len(puts[r]) for r in ("partition0", "partition1")], [0, 2])
        self.assertEqual([len(gets[r]) for r in ("partition0", "partition1")], [1, 0])
        self.assertEqual({e["chan"] for e in puts["partition1"] + gets["partition0"]},
                         {"%extern+128"})  # fmt: skip
        (store,) = events(f, "gstore")["partition0"]
        self.assertIn(gets["partition0"][0]["var"], str(store["offsets"]))

    def test_word(self) -> None:
        self.check_channel(run())

    def test_matrix_rows(self) -> None:
        self.check_channel(run(**MATRIX))

    def test_kir_valid(self) -> None:
        self.assertEqual([e for e in kir.validate(kir_of(run())) if "handoff" in e], [])

    def test_putter_gets_no_channel(self) -> None:
        # partition0 also puts a word partition1 reads after its first wait: partition1,
        # a putter, gets nothing, so the KIR stays valid
        f = run(get=WORD["get"] + "    st.shared.b32 [global_smem+132], %r60;\n",
                put=WORD["put"] + "    ld.shared.b32 %r55, [global_smem+132];\n")  # fmt: skip
        gets = events(f, "handoff_get") if not f.refusals else {}
        self.assertFalse(any(gets.get(r) for r in ("partition1",)))
        if f.refusals:
            self.assertIn("translated again with the handoff channel(s)", refusal(f))
        else:
            self.assertEqual([e for e in kir.validate(kir_of(f)) if "handoff" in e], [])

    def test_signed_load(self) -> None:
        # the 64-bit test of -1 stays (zero-extended, it would fold to false)
        f = run(get="    ld.shared.s32 %rd40, [global_smem+128];\n"
                    "    setp.eq.s64 %p25, %rd40, -1;\n    @%p25 bra $RET;\n"
                    "    cvt.u32.u64 %r62, %rd40;\n")  # fmt: skip
        self.assertEqual(f.refusals, [])
        (role,) = [r for r in f.roles if r["name"] == "partition0"]
        self.assertTrue(
            any(d.get("e") == "ext" and d.get("signed") for d in _exprs(role["body"]))
        )
        self.assertIn(
            {"e": "const", "w": 64, "v": -1},
            [d["b"] for d in _exprs(role["body"]) if d.get("e") == "cmp"],
        )

    def test_unread_word_is_no_channel(self) -> None:
        # partition0 also stores a word no role loads: it puts no channel, so it gets
        f = run(get="    st.shared.b32 [global_smem+136], %r20;\n" + WORD["get"])
        self.check_channel(f)

    def test_unbounded_write_assumes_it_misses_the_channel(self) -> None:
        f = run(default="    shl.b32 %r91, %r20, 2;\n    add.s32 %r92, %r30, %r91;\n"
                        "    st.shared.b32 [%r92], %r1;\n")  # fmt: skip
        self.assertEqual(f.refusals, [])
        (role,) = [r for r in f.roles if r["name"] == "default"]
        self.assertIn("a shared write misses the handoff channel %extern+128",
                      [s.get("msg") for s in _exprs(role["body"]) if s.get("op") == "assume"])  # fmt: skip

    def test_write_through_a_zero_extension_may_overwrite(self) -> None:
        # the default stores at the zero extension of x = (n & 7) - 1 in [-1, 6] at 16
        # bits: 0..6 or 65535, so its 4 bytes may cover the channel at 65536 (bounds of
        # the extension, not of x: interp.bounds)
        f = run(default="    and.b32 %r80, %r20, 7;\n    add.s32 %r81, %r80, -1;\n"
                        "    cvt.u32.u16 %r82, %r81;\n    add.s32 %r84, %r30, %r82;\n"
                        "    st.shared.b32 [%r84], %r1;\n",
                put="    st.shared.b32 [global_smem+65536], %r51;\n",
                get="    ld.shared.b32 %r62, [global_smem+65536];\n")  # fmt: skip
        self.assertIn("may overwrite the handoff channel %extern+65536", refusal(f))

    def test_channel_over_an_mbarrier_array_is_refused(self) -> None:
        f = run(put="    st.shared.b32 [global_smem+84], %r51;\n",
                get="    ld.shared.b32 %r62, [global_smem+84];\n")  # fmt: skip
        self.assertIn("overlaps an mbarrier array", refusal(f))

    def test_error_in_the_second_translation_is_an_error(self) -> None:
        load = interp.Translator._handoff_load

        def boom(tr, i, mn):  # a bug in the translation with channels
            if tr.k.handoffs:
                raise KeyError("a bug")
            return load(tr, i, mn)

        with mock.patch.object(interp.Translator, "_handoff_load", boom):
            f = run()
        self.assertEqual(f.refusals[0]["kind"], "error")

    def test_matrix_operand_count_no_channel(self) -> None:
        f = run(put="    stmatrix.sync.aligned.m8n8.x1.shared.b16 [%r72+128], {%r51, %r51};\n",
                get=MATRIX["get"])  # fmt: skip
        self.assertIn("reaches", refusal(f))
        self.assertNotIn("handoff", refusal(f))

    def test_two_writers_no_channel(self) -> None:
        f = run(default="    st.shared.b32 [global_smem+128], %r20;\n")
        self.assertIn("reaches", refusal(f))
        self.assertNotIn("handoff", refusal(f))

    def test_varying_value_no_channel(self) -> None:
        f = run(put="    st.shared.b32 [global_smem+128], %r1;\n")
        self.assertIn("reaches", refusal(f))
        self.assertNotIn("handoff", refusal(f))

    def test_other_write_refused(self) -> None:
        f = run(default="    st.shared.b32 [global_smem+128], %r1;\n")
        self.assertIn("reaches", refusal(f))
        self.assertIn("may overwrite the handoff channel %extern+128", refusal(f))


if __name__ == "__main__":
    unittest.main()
