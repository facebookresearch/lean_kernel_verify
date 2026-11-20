# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end tests of scripts/lkv.py: frontends, KIR contract, Lean checks, reports.

    python3 -I tests/e2e/test_e2e.py [-k NAME]      (about 2 minutes; buck2 + Lean, no GPU)

* Unit (no buck2, no Lean, a second): lkv.py's reading of Lean output, its report and
  its command line. Notes are never counterexamples and never change a verdict; a PASS
  needs a proved theorem (Lean's exit status, its errors and the refuted theorems must
  agree); every note count is accounted for; notes are grouped by source line; a refuted
  spec is FAIL unless Lean marks it `UNDECIDED` (no certain counterexample: a role trace
  stopped at an out-of-bounds metadata load, P1-1 U1 pairs, an evaluation error that is
  not a kernel error); an evaluation error's kind decides its specs (kernel error: FAIL,
  model limit: UNDECIDED, unexpected: ERROR), and an INCONSISTENT verdict is an ERROR; a
  variant without verdicts is UNDECIDED if the frontend refuses it by design, else an
  ERROR (a refused all-preferred cluster geometry `<spec>@cluster<dims>` for the inputs
  of its spec whose grid the dims divide); a spec, or its all-preferred geometry, for
  which to_kir wrote neither a P0-6 KIR nor a P0-6 failure is an ERROR too; the outcome
  and exit status are FAIL (1) on
  any FAIL, else ERROR (2) on any failure, else UNDECIDED (0) on anything a known limit
  leaves open, else PASS (0); the P1 specs' ERRORs always count, their FAIL and
  UNDECIDED only with --p1-strict; a preferred geometry's P1-2 KIR
  (`<spec>@cluster<dims>.p1.json`) is a P1 variant of its own; inputs the host rejects
  count as none; an export's revision marker names the analyzed revision;
  --target reaches the TTGIR dump and is refused for CuTe cases (exit status 2, never 1);
  `run` translates with --frontend, else $LKV_FRONTEND, else the family's default (ptx for
  TLX/Triton, legacy for CuTe);
  the report renders the frontend's environment record, and there marks the source lines
  of the CuTe DSL 4.4.2 PTX translation (LKV_FRONTEND=ptx) best effort (report.json
  `loc_quality`; a legacy report has no such key).
* lean_fixtures: `lkv.py check` on tests/lean/*.kir.json gives the verdicts of
  tests/lean/expected.txt, and lkv files every text of tests/lean/expected_messages.txt
  under the right kind: a `note: ` text among the notes and never among the
  counterexamples, any other text among the counterexamples or the evaluation error.
  Every evaluation error has the kind tests/lean/eval_errors and segmented_gload
  bad_layout pin, and every spec of its input that kind's outcome; the other UNDECIDED
  outcomes are exactly P0-1..P0-5 of the inputs of `assume` and `assume_reader` whose
  assumption does not hold (KIR change 16; the certain counterexamples of a CTA where no
  trace stopped stay FAILs), P0-2..P0-5 of the four inputs with an out-of-bounds
  metadata load (meta_oob_load out_of_range, eval_errors meta_oob, meta_masked_load unmasked_empty,
  meta_oob_elided out_of_range) and P0-2/P0-4 of ledger_cycle's inputs and ledger_confirm
  at_budget (a barrier ledger that does not converge) where nothing but the `not decided`
  line is listed, and P0-4 of cluster_pair_prologue peer_unordered, cluster_multicast_pipeline
  local_release and cluster_lander_shares' inputs (another CTA's bytes that balance only if
  a landing completes a later phase than its issue order shows).
* probe_tlx / probe_cute: the probe kernels of kernels/ through the real TTGIR and CuTe
  frontends (`lkv.py run`), with degenerate inputs (an empty tensor, fewer elements than
  one block). The seeded bugs (a floor grid; threads past the end) must be the only FAILs.
  An input the host rejects (a `raise` in its validation) and, for TLX, one it makes no
  launch for are listed as rejected with the reason; the other inputs still get verdicts.
  The shell sets a TRITON_* and a CUTE_DSL_* variable: the frontend must drop both and
  the report must list them as ignored shell values. Each probe runs twice, by default
  and in the other translation: probe_tlx in the PTX translation (KIR `ir` "ptx") and with
  `--frontend legacy` (KIR `ir` "ttgir"), probe_cute in the legacy translation (KIR `ir`
  "cute-mlir") and with `--frontend ptx` (KIR `ir` "ptx"), with the same verdicts; the
  report names the translation.

Outputs: $LKV_E2E_OUT (default ${TMPDIR:-/tmp}/lkv_e2e_$USER), one directory per test.
"""

from __future__ import annotations

import getpass
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True  # keep the skill tree free of __pycache__
HERE = Path(__file__).resolve().parent
SKILL = HERE.parents[1]
LKV = SKILL / "scripts" / "lkv.py"
OUT = Path(
    os.environ.get(
        "LKV_E2E_OUT", f"{tempfile.gettempdir()}/lkv_e2e_{getpass.getuser()}"
    )
)
SPECS = [f"P0-{n}" for n in range(1, 6)]

_spec = importlib.util.spec_from_file_location("lkv", LKV)
lkv_py = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lkv_py)


def lkv(cmd: str, name: str, *args: str, env: dict | None = None) -> dict:
    """Run `lkv.py cmd args` into OUT/name, with `env` added to the environment (less a
    shell LKV_FRONTEND: `run` takes the default translation unless args say otherwise);
    return its report.json, whose result gives the exit status (0, 1 and 2 all write one)."""
    out = OUT / name
    shell = {k: v for k, v in os.environ.items() if k != "LKV_FRONTEND"}
    r = subprocess.run(
        [sys.executable, "-I", str(LKV), cmd, *args, "--out", str(out)],
        env={**shell, **(env or {})},
    )
    if r.returncode not in (0, 1, 2) or not (out / "report.json").exists():
        raise AssertionError(f"lkv.py {cmd} exited {r.returncode}; see {out}")
    report = json.loads((out / "report.json").read_text())
    assert report["result"]["exit_status"] == r.returncode, (report["result"], r)
    return report


def runs(report: dict) -> dict:
    """{(variant, input): {spec: verdict}} over every checked variant; no unchecked ones."""
    assert not report["frontend_failures"], report["frontend_failures"]
    out = {}
    for v in report["variants"]:
        assert v["status"] == "checked", (v["variant"], v["status"], v.get("error"))
        for i in v["inputs"]:
            out[(v["variant"], i["name"])] = {
                s: i["specs"][s]["verdict"] for s in SPECS
            }
    return out


def failing(verdicts: dict) -> set:
    return {s for s, v in verdicts.items() if v != "PASS"}


def kir_irs(report: dict) -> set:
    """The `ir` of every KIR a report's variants were checked from."""
    return {json.loads(Path(v["kir"]).read_text())["ir"] for v in report["variants"]}


def runs_of_report(report: dict) -> list:
    """(variant, input) pairs of a report's checked variants."""
    return [(v, i) for v in report["variants"] for i in v.get("inputs", [])]


# The first line Lean lists for a spec an input does not decide (lean/KV/Check.lean).
NOT_DECIDED_HEAD = "not decided on this input"
# The kind of every evaluation error of tests/lean (lean/KV/Eval.lean `ErrKind`).
EVAL_ERROR_KINDS = {
    ("eval_errors", "assert_fail"): "kernel",
    ("eval_errors", "barrier_index"): "kernel",
    ("eval_errors", "for_step_zero"): "kernel",
    ("eval_errors", "negative_grid"): "kernel",
    ("eval_errors", "div_zero"): "limit",
    ("eval_errors", "tensor_div_zero_masked"): "limit",
    ("eval_errors", "while_bound"): "unexpected",
    ("eval_errors", "meta_missing_value"): "unexpected",
    ("eval_errors", "meta_negative_offset"): "unexpected",
    ("eval_errors", "negative_barrier_num"): "unexpected",
    ("segmented_gload", "bad_layout"): "unexpected",
    ("cluster_launch", "grid3"): "kernel",
    ("cluster_remote_arrive", "rank_outside_cluster"): "kernel",
    ("cluster_dsmem_relay", "negative_bytes"): "unexpected",
    ("handoff", "overwritten"): "limit",
    ("handoff", "other_arrival"): "limit",
    ("handoff", "read_before_wait"): "limit",
    ("handoff", "no_value"): "limit",
    ("handoff", "parity_order"): "limit",
    ("handoff_group", "parts_differ"): "limit",
}


# P0-4 UNDECIDED through another CTA's landings (Ledger.lean, "Landers").
LANDER_UNDECIDED = {
    ("cluster_pair_prologue", "peer_unordered", "P0-4"),
    ("cluster_multicast_pipeline", "local_release", "P0-4"),
    ("cluster_lander_shares", "ordered", "P0-4"),
    ("cluster_lander_shares", "unordered", "P0-4"),
    ("cluster_clc_response", "unordered", "P0-4"),
    ("cluster_pair_two_release", "early_release", "P0-4"),
}


def render(f: dict) -> str:
    return f"{f['loc']}: {f['context']}: {f['msg']}"


def env_section(md: str) -> str:
    """The 'Environment of the frontend run' section of a report.md."""
    return md.split("## Environment of the frontend run")[1].split("\n## ")[0]


# Compiler switches in the shell of a probe run: the frontend must not use them.
SHELL_ENV = {"TRITON_LKV_E2E_SHELL": "ignored", "CUTE_DSL_LKV_E2E_SHELL": "ignored"}


# Lean output as lean/KV/Check.lean prints it, for input "a" of a one-input KIR: P0-1
# passes with notes (5 notes, 2 printed), P0-3 fails (its theorem is refuted) and has a
# note.
P0_3_FAIL = """\
P0-3 output coverage exactly once: FAIL
  k.py:9: cta (0,0,0) role r: gstore y: 1 element(s) outside the intended region (rows [7, 8))
  ... 2 more
"""
REFUTATION = """\
/tmp/k.lean:40:1: error: Tactic `native_decide` evaluated that the proposition
  KV.Holds KV.SpecId.coverage kernel input0_a
is false
"""
LEAN_OUT = (
    """\
== k / input a
grid (1,1,1): 1 CTA(s) x 1 role(s), 3 events
P0-1 global accesses in bounds: PASS
  note: k.py:5: cta (0,0,0) role r: gload x: read spill: rows [8, 9) outside segment batch=0 is rows [0, 8)
  note: k.py:7: cta (0,0,0) role r: gload x: read spill: rows [8, 9) outside segment batch=0 is rows [0, 8)
  note: ... 3 more
P0-2 role trip-count agreement: PASS
"""
    + P0_3_FAIL
    + """\
  note: ?: tensor z: intended_writes is "unchecked": P0-3 checks nothing for this tensor
P0-4 mbarrier protocol: PASS
P0-5 mask-elision soundness: PASS
VERDICT a P0-1 PASS
VERDICT a P0-2 PASS
VERDICT a P0-3 FAIL
VERDICT a P0-4 PASS
VERDICT a P0-5 PASS
NOTES a P0-1 5
NOTES a P0-3 1
"""
    + REFUTATION
)
ONE_INPUT = {"case": {"inputs": [{"name": "a"}]}}

# Input "a" stops at an out-of-bounds metadata load (tests/lean/meta_oob_load): P0-1
# fails with the load, P0-3 fails with a certain double write, and P0-2, P0-4 and P0-5,
# with nothing certain, are undecided.
NOT_DECIDED = (
    "oob.py:5: cta (1,0,0) role default: not decided on this input: 1 role trace(s) stop "
    "at an out-of-bounds metadata load (P0-1), so their later events are unknown; the "
    "lines of this spec that are not `not decided` lines are certain counterexamples"
)
UNDECIDED_OUT = (
    f"""\
== k / input a
grid (2,1,1): 2 CTA(s) x 1 role(s), 3 events; 1 role trace(s) stop at an out-of-bounds metadata load, so P0-2..P0-5 are not decided
P0-1 global accesses in bounds: FAIL
  oob.py:5: cta (1,0,0) role default: load meta[1] outside the metadata tensor (shape [1]): an out-of-bounds global read
P0-2 role trip-count agreement: FAIL
  {NOT_DECIDED}
P0-3 output coverage exactly once: FAIL
  {NOT_DECIDED}
  ?: tensor out: 1 element(s) written more than once by plain stores (rows [0, 1))
P0-4 mbarrier protocol: FAIL
  {NOT_DECIDED}
P0-5 mask-elision soundness: FAIL
  {NOT_DECIDED}
"""
    + "".join(f"VERDICT a {label} FAIL\n" for label in SPECS)
    + "".join(f"UNDECIDED a {label}\n" for label in ("P0-2", "P0-4", "P0-5"))
    + "".join(
        REFUTATION.replace("coverage", spec)
        for spec in ("bounds", "tripCount", "coverage", "barrier", "maskElision")
    )
)


def eval_error_out(kind: str, msg: str, consequence: str) -> str:
    """Lean's output for input "a" stopped by an evaluation error of `kind`: every theorem
    is refuted; only a kernel error is certain (no `UNDECIDED` lines)."""
    return (
        f"== k / input a\nevaluation error ({kind}): k.py:6: cta (0,0,0) role r: {msg} "
        f"({consequence})\n"
        + "".join(f"{label} {lkv_py.TITLES[label]}: FAIL\n" for label in SPECS)
        + "".join(f"VERDICT a {label} FAIL\n" for label in SPECS)
        + (
            ""
            if kind == "kernel error"
            else "".join(f"UNDECIDED a {label}\n" for label in SPECS)
        )
        + "".join(
            REFUTATION.replace("coverage", spec)
            for spec in ("bounds", "tripCount", "coverage", "barrier", "maskElision")
        )
    )


EVAL_ERRORS = {
    "kernel error": eval_error_out(
        "kernel error",
        "device assert failed: X must not be 7",
        "a certain error of the kernel on this input: every spec FAILs",
    ),
    "model limit": eval_error_out(
        "model limit",
        "division by zero: sdiv(100, 0)",
        "a limit of the model: no spec is decided on this input; not an error",
    ),
    "unexpected": eval_error_out(
        "unexpected",
        "while loop exceeds max_iters = 10",
        "the case or LKV needs a fix: no spec is decided on this input",
    ),
}
# LEAN_OUT as `#lkv_check` prints it (lean/KV/Command.lean): no refuted theorem, one THEOREM
# line per spec (the negation of P0-3's `KV.Holds` is proved), exit status 0.
THEOREMS = "".join(
    f"THEOREM a {label} {'FAIL' if label == 'P0-3' else 'PASS'}\n" for label in SPECS
)
LKV_CHECK_OUT = LEAN_OUT.replace(REFUTATION, "") + THEOREMS
# LEAN_OUT with every spec proved.
PASSING = (
    LEAN_OUT.replace(REFUTATION, "")
    .replace(P0_3_FAIL, "P0-3 output coverage exactly once: PASS\n")
    .replace("VERDICT a P0-3 FAIL", "VERDICT a P0-3 PASS")
)
# dump.json "env" of a TTGIR dump (frontend/ttgir/host_env.py).
ENV = {
    "case": {"FLASH_ATTENTION_ARCH": "sm_100"},
    "own": {"TRITON_CACHE_DIR": "/tmp/c"},
    "unset": {"TRITON_X": "1"},
    "read_by_sources": ["FA_CLC", "FLASH_ATTENTION_ARCH"],
    "effective": {"FLASH_ATTENTION_ARCH": "sm_100", "TRITON_CACHE_DIR": "/tmp/c"},
    "triton_cache_invalidating": {},
}


class Unit(unittest.TestCase):
    def test_notes_are_not_counterexamples(self):
        (inp,) = lkv_py.parse_lean(LEAN_OUT, ONE_INPUT, 1)
        bounds, coverage = inp["specs"]["P0-1"], inp["specs"]["P0-3"]
        self.assertEqual(bounds["verdict"], "PASS")
        self.assertEqual(bounds["counterexamples"], [])
        self.assertEqual([n["loc"] for n in bounds["notes"]], ["k.py:5", "k.py:7"])
        self.assertEqual((bounds["notes_more"], bounds["note_count"]), (3, 5))
        self.assertTrue(bounds["notes"][0]["msg"].startswith("gload x: read spill"))
        self.assertEqual(coverage["verdict"], "FAIL")
        self.assertEqual([c["loc"] for c in coverage["counterexamples"]], ["k.py:9"])
        self.assertEqual(coverage["more"], 2)
        self.assertEqual(
            [(n["loc"], n["context"]) for n in coverage["notes"]], [("?", "tensor z")]
        )
        others = {s["verdict"] for label, s in inp["specs"].items() if label != "P0-3"}
        self.assertEqual(others, {"PASS"})

    def test_every_note_is_accounted_for(self):
        text = LEAN_OUT.replace("NOTES a P0-1 5", "NOTES a P0-1 6")
        with self.assertRaisesRegex(RuntimeError, "NOTES gives 6 notes"):
            lkv_py.parse_lean(text, ONE_INPUT, 1)
        text = LEAN_OUT.replace("NOTES a P0-3 1\n", "")
        with self.assertRaisesRegex(RuntimeError, "NOTES gives 0 notes"):
            lkv_py.parse_lean(text, ONE_INPUT, 1)

    def test_pass_needs_a_proved_theorem(self):
        # Lean failed or was killed after printing the #eval report: no verdicts at all.
        unproved = LEAN_OUT.replace(REFUTATION, "")
        for status in (1, -9):
            with self.assertRaisesRegex(RuntimeError, f"exited with status {status}"):
                lkv_py.parse_lean(unproved, ONE_INPUT, status)
        with self.assertRaisesRegex(RuntimeError, "exited with status 0"):
            lkv_py.parse_lean(LEAN_OUT, ONE_INPUT, 0)
        other_error = LEAN_OUT + "/tmp/k.lean:9:0: error: unknown identifier\n"
        with self.assertRaisesRegex(RuntimeError, "1 error"):
            lkv_py.parse_lean(other_error, ONE_INPUT, 1)
        # The #eval report says FAIL but the theorem was proved: not a PASS either.
        (inp,) = lkv_py.parse_lean(unproved, ONE_INPUT, 0)
        self.assertEqual(inp["specs"]["P0-3"]["verdict"], "INCONSISTENT")
        self.assertEqual(inp["specs"]["P0-3"]["outcome"], "ERROR")
        self.assertEqual(inp["specs"]["P0-1"]["verdict"], "PASS")
        self.assertEqual(inp["specs"]["P0-1"]["outcome"], "PASS")
        # Lean never marks a proved theorem undecided.
        with self.assertRaisesRegex(RuntimeError, "UNDECIDED, but proved"):
            lkv_py.parse_lean(LEAN_OUT + "UNDECIDED a P0-1\n", ONE_INPUT, 1)

    def test_representative_schedule_pass_is_undecided(self):
        # kir.md change 18: on the representative claim schedule every PASS is undecided (a
        # spec may depend on which work a CTA claims); a FAIL stays, noted.
        k = {**ONE_INPUT, "representative": ["claims"]}
        (inp,) = lkv_py.parse_lean(LEAN_OUT, k, 1)
        outcomes = {label: s["outcome"] for label, s in inp["specs"].items()}
        self.assertEqual(outcomes, {"P0-1": "UNDECIDED", "P0-2": "UNDECIDED", "P0-3": "FAIL",
                                    "P0-4": "UNDECIDED", "P0-5": "UNDECIDED"})  # fmt: skip
        self.assertIn("PASS on one feasible schedule only",
                      inp["specs"]["P0-1"]["counterexamples"][-1]["msg"])  # fmt: skip
        self.assertIn(
            "witnessed by a feasible schedule", inp["specs"]["P0-3"]["notes"][-1]["msg"]
        )

    def test_lkv_check_theorems_are_the_verdicts(self):
        (inp,) = lkv_py.parse_lean(LKV_CHECK_OUT, ONE_INPUT, 0)
        (legacy,) = lkv_py.parse_lean(LEAN_OUT, ONE_INPUT, 1)
        self.assertEqual(inp, legacy)
        self.assertEqual(inp["specs"]["P0-3"]["verdict"], "FAIL")
        self.assertEqual(inp["specs"]["P0-1"]["verdict"], "PASS")
        # Any Lean error or a non-zero exit status: no verdicts.
        with self.assertRaisesRegex(RuntimeError, "exited with status 1"):
            lkv_py.parse_lean(LKV_CHECK_OUT, ONE_INPUT, 1)
        other_error = LKV_CHECK_OUT + "/tmp/k.lean:9:0: error: unknown identifier\n"
        with self.assertRaisesRegex(RuntimeError, "1 error"):
            lkv_py.parse_lean(other_error, ONE_INPUT, 1)
        # Every (input, spec) needs its theorem.
        missing = LKV_CHECK_OUT.replace("THEOREM a P0-5 PASS\n", "")
        with self.assertRaisesRegex(RuntimeError, "no theorem for 1"):
            lkv_py.parse_lean(missing, ONE_INPUT, 0)
        # The report says FAIL but `KV.Holds` was proved: INCONSISTENT, an ERROR.
        proved = LKV_CHECK_OUT.replace("THEOREM a P0-3 FAIL", "THEOREM a P0-3 PASS")
        (inp,) = lkv_py.parse_lean(proved, ONE_INPUT, 0)
        self.assertEqual(inp["specs"]["P0-3"]["verdict"], "INCONSISTENT")
        self.assertEqual(inp["specs"]["P0-3"]["outcome"], "ERROR")
        with self.assertRaisesRegex(RuntimeError, "UNDECIDED, but proved"):
            lkv_py.parse_lean(LKV_CHECK_OUT + "UNDECIDED a P0-1\n", ONE_INPUT, 0)

    def test_lkv_check_p0_7_theorem(self):
        # kir.md change 18: a KIR with sync tensors also decides P0-7 (`gwait`) in the input's
        # one evaluation; its THEOREM line is required like every other spec's.
        k = {**ONE_INPUT, "tensors": [{"name": "w", "sync": {"width": 64}}]}
        text = (
            LKV_CHECK_OUT.replace(
                "VERDICT a P0-1 PASS",
                "P0-7 global waits reached, co-residency: PASS\nVERDICT a P0-1 PASS",
            ).replace("VERDICT a P0-5 PASS", "VERDICT a P0-5 PASS\nVERDICT a P0-7 PASS")
            + "THEOREM a P0-7 PASS\n"
        )
        (inp,) = lkv_py.parse_lean(text, k, 0)
        self.assertEqual(inp["specs"]["P0-7"]["verdict"], "PASS")
        self.assertEqual(inp["specs"]["P0-7"]["outcome"], "PASS")
        with self.assertRaisesRegex(RuntimeError, r"no theorem for 1 .*'P0-7'"):
            lkv_py.parse_lean(text.replace("THEOREM a P0-7 PASS\n", ""), k, 0)

    def report(self, text, status, **extra):
        """A report of two variants that both printed `text`."""
        inputs = lkv_py.parse_lean(text, ONE_INPUT, status)
        for s in inputs[0]["specs"].values():
            for f in s["counterexamples"] + s["notes"]:
                f["source"], f["line_text"] = None, None
        variants = [
            {"variant": v, "kernel": "k", "status": "checked", "inputs": inputs}
            for v in ("k__1", "k__2")
        ]
        report = {
            "case": "unit",
            "host_rejected": [],
            "frontend_failures": [],
            "variants": variants,
            **extra,
        }
        report["totals"] = lkv_py.totals(variants)
        return report

    def test_report_lists_notes_by_source_line(self):
        report = self.report(LEAN_OUT, 1)
        md = lkv_py.markdown(report)
        self.assertIn(
            "| k | `a` | PASS; notes (2/2) | PASS | **FAIL** (2/2); notes (2/2) | PASS | PASS |",
            md,
        )
        fails, notes = md.split("## Notes by spec and source line")
        self.assertIn("- `k.py:9`: 2 run(s)", fails)
        self.assertNotIn("k.py:5", fails)
        for where in ("`k.py:5`", "`k.py:7`", "(no IR location) tensor z"):
            self.assertIn(f"- {where}: 2 run(s)", notes)
        self.assertEqual(
            report["totals"]["P0-1"],
            {
                "PASS": 2,
                "FAIL": 0,
                "ERROR": 0,
                "UNDECIDED": 0,
                "INCONSISTENT": 0,
                "with_notes": 2,
            },
        )
        self.assertEqual(lkv_py.exit_status(report), 1)
        self.assertIn("## Result: FAIL (exit status 1)", md)
        self.assertIn("- FAIL: 2 certain violation(s) (P0-3: 2)", md)
        self.assertNotIn("## ERROR", md)
        self.assertNotIn("## UNDECIDED", md)

    def test_refuted_specs_are_fail_or_undecided(self):
        (inp,) = lkv_py.parse_lean(UNDECIDED_OUT, ONE_INPUT, 1)
        self.assertEqual(
            {label: (s["verdict"], s["outcome"]) for label, s in inp["specs"].items()},
            {
                "P0-1": ("FAIL", "FAIL"),
                "P0-2": ("FAIL", "UNDECIDED"),
                "P0-3": ("FAIL", "FAIL"),  # the double write is certain
                "P0-4": ("FAIL", "UNDECIDED"),
                "P0-5": ("FAIL", "UNDECIDED"),
            },
        )
        locs = [c["loc"] for c in inp["specs"]["P0-3"]["counterexamples"]]
        self.assertEqual(locs, ["oob.py:5", "?"])
        # A FAIL without an UNDECIDED line is certain.
        (decided,) = lkv_py.parse_lean(LEAN_OUT, ONE_INPUT, 1)
        self.assertEqual(decided["specs"]["P0-3"]["outcome"], "FAIL")
        report = self.report(UNDECIDED_OUT, 1)
        self.assertEqual(
            {
                label: (t["FAIL"], t["UNDECIDED"])
                for label, t in report["totals"].items()
            },
            {
                "P0-1": (2, 0),
                "P0-2": (0, 2),
                "P0-3": (2, 0),
                "P0-4": (0, 2),
                "P0-5": (0, 2),
            },
        )
        # Certain violations decide the outcome, whatever else is undecided.
        self.assertEqual(lkv_py.exit_status(report), 1)
        md = lkv_py.markdown(report)
        self.assertIn("## Result: FAIL (exit status 1)", md)
        self.assertIn("- FAIL: 4 certain violation(s) (P0-1: 2, P0-3: 2)", md)
        self.assertIn(
            "- not decided (known limit): 6 spec(s) refuted without a certain "
            "counterexample (P0-2: 2, P0-4: 2, P0-5: 2)",
            md,
        )
        self.assertIn("| P0-2 role trip-count agreement | 0 | 0 | 0 | 2 | 0 |", md)
        self.assertIn(
            "| k | `a` | **FAIL** (2/2) | UNDECIDED (2/2) | **FAIL** (2/2) | UNDECIDED (2/2) "
            "| UNDECIDED (2/2) |",
            md,
        )
        # Each spec sits in the section of its outcome.
        fails = md.split("## FAIL\n")[1].split("\n## ")[0]
        undecided = md.split("## UNDECIDED\n")[1].split("\n## ")[0]
        self.assertIn("#### P0-3 output coverage exactly once", fails)
        self.assertNotIn("P0-2", fails)
        self.assertIn("#### P0-2 role trip-count agreement", undecided)
        self.assertNotIn("P0-3", undecided)
        self.assertNotIn("## ERROR", md)

    def test_evaluation_errors_by_kind(self):
        # A kernel error FAILs every spec of the input, a model limit leaves them UNDECIDED
        # (not an error), an unexpected error makes them ERRORs.
        want = {
            "kernel error": ("FAIL", 1, "**FAIL** (2/2)"),
            "model limit": ("UNDECIDED", 0, "UNDECIDED (2/2)"),
            "unexpected": ("ERROR", 2, "**ERROR** (2/2)"),
        }
        for kind, (outcome, status, cell) in want.items():
            (inp,) = lkv_py.parse_lean(EVAL_ERRORS[kind], ONE_INPUT, 1)
            self.assertEqual(inp["eval_kind"], lkv_py.EVAL_KINDS[kind])
            self.assertTrue(
                inp["eval_error"].startswith("k.py:6: cta (0,0,0) role r: ")
            )
            self.assertNotIn("this input", inp["eval_error"])  # the heading says it
            self.assertEqual({s["outcome"] for s in inp["specs"].values()}, {outcome})
            report = self.report(EVAL_ERRORS[kind], 1)
            self.assertEqual(lkv_py.exit_status(report), status, kind)
            md = lkv_py.markdown(report)
            self.assertIn(f"## Result: {outcome} (exit status {status})", md)
            self.assertIn("| k | `a` | " + " | ".join([cell] * 5) + " |", md)
            section = md.split(f"## {outcome}\n")[1].split("\n## ")[0]
            self.assertIn(
                "- k `a` (2 variant(s)): k.py:6: cta (0,0,0) role r: ", section
            )
            for other in {"FAIL", "ERROR", "UNDECIDED"} - {outcome}:
                self.assertNotIn(f"## {other}\n", md)
        self.assertEqual(
            lkv_py.result(self.report(EVAL_ERRORS["kernel error"], 1))["fail"],
            ["2 (variant, input) run(s) stopped at a certain error of the kernel"],
        )
        self.assertEqual(
            lkv_py.result(self.report(EVAL_ERRORS["model limit"], 1))["undecided"],
            [
                "2 (variant, input) run(s) whose evaluation stopped at a limit of the "
                "model"
            ],
        )
        self.assertEqual(
            lkv_py.result(self.report(EVAL_ERRORS["unexpected"], 1))["error"],
            [
                "2 (variant, input) run(s) whose evaluation stopped unexpectedly (the "
                "case or LKV needs a fix)"
            ],
        )

    def test_report_shows_the_frontend_environment(self):
        env = env_section(lkv_py.markdown(self.report(LEAN_OUT, 1, env=ENV)))
        for line in (
            "- case `env`: `FLASH_ATTENTION_ARCH=sm_100`",
            "- shell values ignored (unset before the compiler and kernel imports): `TRITON_X=1`",
            "- read by the exported sources (unset unless effective): `FA_CLC`, `FLASH_ATTENTION_ARCH`",
            "- effective (variables the exported sources read, compiler switches): "
            "`FLASH_ATTENTION_ARCH=sm_100`, `TRITON_CACHE_DIR=/tmp/c`",
            "- Triton cache-invalidating variables: none",
        ):
            self.assertIn(line, env)
        cute = {k: v for k, v in ENV.items() if k != "triton_cache_invalidating"}
        self.assertNotIn(
            "Triton", env_section(lkv_py.markdown(self.report(LEAN_OUT, 1, env=cute)))
        )
        # `check` runs no frontend: no environment section.
        self.assertNotIn("## Environment", lkv_py.markdown(self.report(LEAN_OUT, 1)))

    def test_cute_ptx_locations_are_best_effort(self):
        # The CuTe binary stands in as its to_kir alone, on the recorded DSL 4.4.2
        # probe_cute dump of tests/ptx (the legacy translation finds no stage there).
        fixtures, kernels = SKILL / "tests/ptx/fixtures", HERE / "kernels"
        if sys.version_info < (3, 10) or not fixtures.is_dir():
            self.skipTest(
                "needs the frontends (Python 3.10+, not in the CI's Lean tree)"
            )

        def buck_run(target, args, log_path):
            to_kir = SKILL / "frontend/cute/to_kir.py"
            out = args[args.index("--out") + 1]
            cmd = [sys.executable, "-I", str(to_kir), str(args[1]), "--out", str(out)]
            with open(log_path, "w") as f:
                return subprocess.run(cmd, stdout=f, stderr=f).returncode

        real, lkv_py.buck_run = lkv_py.buck_run, buck_run
        self.addCleanup(setattr, lkv_py, "buck_run", real)
        self.addCleanup(lkv_py.P1_NOT_CHECKED.clear)
        self.addCleanup(lkv_py.LOC_QUALITY.clear)
        line = "- source locations are best effort in `host__02200b3bc0acee10`: "
        with tempfile.TemporaryDirectory() as d:
            fe = Path(d)
            text = (fixtures / "probe_cute.ptx").read_text()
            (fe / "probe_cute.ptx").write_text(text.replace("@SRC@", str(kernels)))
            text = (fixtures / "probe_cute.dump.json").read_text()
            dump = json.loads(text.replace("@SRC@", str(kernels)))
            for c in dump["compiles"]:
                c["ptx"]["file"] = str(fe / "probe_cute.ptx")
            (fe / "dump.json").write_text(json.dumps(dump))
            for mode, want in (
                ("legacy", {}),
                ("ptx", {"host__02200b3bc0acee10": "line_only"}),
            ):
                lkv_py.LOC_QUALITY.clear()
                with mock.patch.dict(os.environ, {"LKV_FRONTEND": mode}):
                    case = HERE / "cases/probe_cute.json"
                    lkv_py.run_cute(case, {}, kernels, fe, fe, [])
                self.assertEqual(lkv_py.LOC_QUALITY, want)
                with mock.patch.object(lkv_py, "log"):  # quiet
                    lkv_py.write_report(self.report(PASSING, 0, env=ENV), fe)
                report = json.loads((fe / "report.json").read_text())
                md = env_section((fe / "report.md").read_text())
                if want:
                    self.assertEqual(report["loc_quality"], want)
                    self.assertIn(line, md)
                else:  # the legacy report: no key, no line
                    self.assertNotIn("loc_quality", report)
                    self.assertNotIn("best effort", md)

    def test_notes_and_host_rejected_inputs_keep_exit_status_0(self):
        rejected = [{"input": "b", "error": "the host rejected input b: ValueError"}]
        report = self.report(PASSING, 0, host_rejected=rejected)
        self.assertEqual(lkv_py.exit_status(report), 0)
        md = lkv_py.markdown(report)
        self.assertIn("## Result: PASS (exit status 0)", md)
        self.assertIn("## Inputs the host code rejected", md)

    def test_ttgir_p1_per_cluster_geometry(self):
        # the P1-2 KIR of a preferred cluster geometry (<spec>@cluster<dims>.p1.json) is
        # checked as its own variant, and a refused one is not checked, with its inputs
        self.addCleanup(lkv_py.P1_NOT_CHECKED.clear)
        la = lambda spec: {"spec": spec, "launch": {"grid": [8, 1, 1]}}  # noqa: E731
        dump = {"kernel": "k", "specs": {"s": {}, "t": {}},
                "inputs": [{"name": "i", "launches": [la("s")]},
                           {"name": "j", "launches": [la("t")]}]}  # fmt: skip
        with tempfile.TemporaryDirectory() as d:
            fe, logs = Path(d) / "fe", Path(d) / "logs"
            fe.mkdir()
            logs.mkdir()
            for name in ("s", "s@cluster4x1x1", "t"):
                (fe / f"{name}.p1.json").write_text("{}")
            (logs / "to_kir.log").write_text(
                "[to_kir] LKV_FRONTEND=ptx: translating the PTX\n"
                "[to_kir] t@cluster4x1x1: P1 refused: a gap\n"
            )
            kirs = lkv_py.ttgir_p1(dump, fe, logs)
        self.assertEqual(
            [k.name for k in kirs], ["s.p1.json", "t.p1.json", "s@cluster4x1x1.p1.json"]
        )
        self.assertEqual(
            [(x["variant"], x["status"], x["inputs"]) for x in lkv_py.P1_NOT_CHECKED],
            [("t@cluster4x1x1", "refused", ["j"])],
        )

    def test_ttgir_p1_geometry_inputs(self):
        # a refused all-preferred P1-2 KIR counts for the inputs whose grid its dims
        # divide (as a failed P0 KIR of that geometry, variant_inputs); one refused because
        # the launches disagree on the preferred dims (`<spec>@cluster`) for every input
        self.addCleanup(lkv_py.P1_NOT_CHECKED.clear)
        la = lambda grid: {"spec": "s", "launch": {"grid": grid}}  # noqa: E731
        dump = {"kernel": "k", "specs": {"s": {}},
                "inputs": [{"name": "a", "launches": [la([8, 1, 1])]},
                           {"name": "b", "launches": [la([6, 1, 1])]}]}  # fmt: skip
        with tempfile.TemporaryDirectory() as d:
            fe, logs = Path(d) / "fe", Path(d) / "logs"
            fe.mkdir()
            logs.mkdir()
            (fe / "s.p1.json").write_text("{}")
            (logs / "to_kir.log").write_text(
                "[to_kir] LKV_FRONTEND=ptx: translating the PTX\n"
                "[to_kir] s@cluster4x1x1: P1 refused: a gap\n"
                "[to_kir] s@cluster: P1 refused: launches differ\n"
            )
            kirs = lkv_py.ttgir_p1(dump, fe, logs)
        self.assertEqual([k.name for k in kirs], ["s.p1.json"])
        self.assertEqual(
            [(x["variant"], x["status"], x["inputs"]) for x in lkv_py.P1_NOT_CHECKED],
            [("s@cluster", "refused", ["a", "b"]),
             ("s@cluster4x1x1", "refused", ["a"])],
        )  # fmt: skip

    def test_variants_without_verdicts_by_kind(self):
        refused = {
            "variant": "k__3",
            "kernel": "k",
            "inputs": ["a", "c"],
            "kind": "gap",
            "error": "refused",
        }
        report = self.report(PASSING, 0, frontend_failures=[refused])
        self.assertEqual(lkv_py.exit_status(report), 0)  # a known limit, not an error
        report["totals"] = lkv_py.totals(
            report["variants"], lkv_py.LABELS, lkv_py.unchecked(report)
        )
        self.assertEqual(
            report["totals"]["P0-1"],
            {
                "PASS": 2,
                "FAIL": 0,
                "ERROR": 0,
                "UNDECIDED": 2,
                "INCONSISTENT": 0,
                "with_notes": 2,
            },
        )
        md = lkv_py.markdown(report)
        self.assertIn("## Result: UNDECIDED (exit status 0)", md)
        self.assertIn(
            "- not decided (known limit): 1 variant(s) the frontend refuses by design: "
            "`k__3`",
            md,
        )
        self.assertIn(
            "- `k__3` (inputs `a`, `c`): frontend refused: refused",
            md.split("### Variants the frontend refuses by design")[1],
        )
        # Its inputs count as UNDECIDED for every spec, beside the variants that ran them.
        self.assertIn(
            "| k | `a` | UNDECIDED (1/3); notes (2/3) | UNDECIDED (1/3) "
            "| UNDECIDED (1/3); notes (2/3) | UNDECIDED (1/3) | UNDECIDED (1/3) |",
            md,
        )
        self.assertIn("| k | `c` | " + " | ".join(["UNDECIDED (1/1)"] * 5) + " |", md)
        # A frontend failure, or a KIR whose check did not finish, is an ERROR.
        failed = {
            **refused,
            "kind": "error",
            "error": "invalid KIR (frontend/kir.py): x",
        }
        report = self.report(PASSING, 0, frontend_failures=[failed])
        self.assertEqual(lkv_py.exit_status(report), 2)
        md = lkv_py.markdown(report)
        self.assertIn("## Result: ERROR (exit status 2)", md)
        self.assertIn(
            "- ERROR: 1 variant(s) that failed to translate or check: `k__3`", md
        )
        self.assertIn(
            "- `k__3` (inputs `a`, `c`): frontend error: invalid KIR (frontend/kir.py): x",
            md.split("### Variants that failed to translate or check")[1],
        )
        self.assertIn("| k | `c` | " + " | ".join(["**ERROR** (1/1)"] * 5) + " |", md)
        report = self.report(PASSING, 0)
        report["variants"].append(
            {
                "variant": "k__4",
                "kernel": "k",
                "status": "lean_error",
                "error": "timeout",
            }
        )
        self.assertEqual(lkv_py.exit_status(report), 2)
        # A certain violation decides the outcome, whatever else failed.
        report = self.report(LEAN_OUT, 1, frontend_failures=[refused, failed])
        self.assertEqual(lkv_py.exit_status(report), 1)
        # A pipeline error is an ERROR; so is a run without any KIR. One whose every
        # variant the frontend refuses by design is not.
        report = self.report(PASSING, 0, error="CuTe frontend failed")
        self.assertEqual(lkv_py.exit_status(report), 2)
        report = self.report(PASSING, 0)
        report["variants"] = []
        self.assertEqual(
            lkv_py.result(report)["error"],
            [
                "no variant reached the Lean checks: the frontend made no KIR (see the "
                "inputs the host rejected)"
            ],
        )
        report["frontend_failures"] = [refused]
        self.assertEqual(lkv_py.exit_status(report), 0)
        self.assertIn(
            "no variant reached the Lean checks: the frontend refuses every variant by "
            "design",
            lkv_py.result(report)["undecided"],
        )

    def test_p1_verdicts_are_reported_apart_and_keep_the_exit_status(self):
        text = (
            "== k / input a\ngrid (1,1,1): 1 CTA(s) x 2 role(s), 9 events; P1: 3 TMEM accesses\n"
            "P1-1 TMEM reuse ordered: FAIL\n"
            "  x.py:15: cta (0,0,0) roles mma / compute: tcgen05.cp at x.py:15 ...: unordered.\n"
            "VERDICT a P1-1 FAIL\n"
            "k.lean:9:9: error: Tactic `native_decide` evaluated that the proposition\n"
            "  KV.Holds KV.SpecId.tmemReuse kernel input0_a\nis false\n"
        )
        p1_kir = {**ONE_INPUT, "tmem": {"cols": 512, "base": "zero"}}
        inputs = lkv_py.parse_lean(text, p1_kir, 1)
        self.assertEqual(list(inputs[0]["specs"]), ["P1-1"])
        self.assertEqual(inputs[0]["specs"]["P1-1"]["verdict"], "FAIL")
        self.assertEqual(inputs[0]["specs"]["P1-1"]["outcome"], "FAIL")

        def with_p1(p1_inputs):
            report = self.report(PASSING, 0)
            for s in p1_inputs[0]["specs"].values():
                for f in s["counterexamples"]:
                    f["source"], f["line_text"] = None, None
            report["variants_p1"] = [
                {
                    "variant": "k__1.p1",
                    "kernel": "k",
                    "status": "checked",
                    "inputs": p1_inputs,
                }
            ]
            report["totals_p1"] = lkv_py.totals(report["variants_p1"], lkv_py.LABELS_P1)
            return report

        report = with_p1(inputs)
        self.assertEqual(lkv_py.exit_status(report), 0)  # P1-1 is not part of it
        md = lkv_py.markdown(report)
        self.assertIn(
            "## P1-1 TMEM reuse ordered (P1; CuTe only; its FAIL and UNDECIDED are not "
            "part of the outcome, its ERRORs are)",
            md,
        )
        self.assertIn("`x.py:15`: 1 run(s)", md)
        self.assertEqual(md.count("| kernel | input | P0-1 |"), 1)
        report["p1_strict"] = True
        self.assertEqual(lkv_py.exit_status(report), 1)
        self.assertEqual(
            lkv_py.result(report)["fail"], ["P1-1: 1 certain violation(s) (P1-1: 1)"]
        )
        # U1: a pair ordered only under the lockstep contents leaves the input undecided.
        undecided = text.replace(
            "  x.py:15: cta (0,0,0) roles mma / compute: tcgen05.cp at x.py:15 ...: unordered.\n",
            "  x.py:15: cta (0,0,0) roles mma / compute: not decided on this input: 1 conflicting"
            " pair(s) of TMEM accesses are ordered only under the ledger's lockstep contents\n"
            "  x.py:15: cta (0,0,0) roles mma / compute: tcgen05.cp at x.py:15 ...: ordered only"
            " under the lockstep contents of the ledger, an assumption\n",
        ).replace("VERDICT a P1-1 FAIL\n", "VERDICT a P1-1 FAIL\nUNDECIDED a P1-1\n")
        (u,) = lkv_py.parse_lean(undecided, p1_kir, 1)
        self.assertEqual(u["specs"]["P1-1"]["outcome"], "UNDECIDED")
        t = lkv_py.totals([{"variant": "k__1.p1", "inputs": [u]}], lkv_py.LABELS_P1)
        self.assertEqual((t["P1-1"]["FAIL"], t["P1-1"]["UNDECIDED"]), (0, 1))
        report = with_p1([u])
        self.assertEqual(lkv_py.result(report)["undecided"], [])
        report["p1_strict"] = True
        self.assertEqual(lkv_py.result(report)["outcome"], "UNDECIDED")
        self.assertEqual(lkv_py.exit_status(report), 0)
        md = lkv_py.markdown(report)
        p1 = md.split("## P1-1 TMEM reuse ordered")[1]
        self.assertIn("| k | `a` | UNDECIDED (1/1) |", p1)
        self.assertIn("### P1-1 UNDECIDED", p1)
        self.assertNotIn("### P1-1 FAIL", p1)
        # With --p1-strict a refused P1 translation is not decided either.
        report = self.report(PASSING, 0, p1_strict=True)
        report["p1_not_checked"] = [
            {"variant": "k__1", "status": "refused", "error": "R4: inline PTX"},
            {"variant": "k__2", "status": "not_applicable", "error": "no TMEM"},
        ]
        self.assertEqual(lkv_py.result(report)["outcome"], "UNDECIDED")
        self.assertEqual(
            lkv_py.result(report)["undecided"],
            ["P1-1: 1 variant(s) the frontend refuses by design: `k__1`"],
        )
        # A failed P1 translation is an ERROR, with or without --p1-strict.
        report = self.report(PASSING, 0)
        report["p1_not_checked"] = [
            {"variant": "k__1", "status": "error", "error": "P1 pass internal error: X"}
        ]
        self.assertEqual(lkv_py.exit_status(report), 2)
        self.assertEqual(
            lkv_py.result(report)["error"],
            ["P1-1: 1 variant(s) that failed to translate or check: `k__1`"],
        )

    def test_p1_2_verdicts_are_labelled_per_spec(self):
        text = (
            "== k / input a\ngrid (1,1,1): 1 CTA(s) x 1 role(s), 9 events; P1: P1-2: 1 bulk store(s)\n"
            "P1-2 async store sources reuse ordered: FAIL\n"
            "  x.py:7: cta (0,0,0) roles w / w: F1: the write at x.py:7 ...: unordered.\n"
            "VERDICT a P1-2 FAIL\n"
            "k.lean:9:9: error: Tactic `native_decide` evaluated that the proposition\n"
            "  KV.Holds KV.SpecId.storeReuse kernel input0_a\nis false\n"
        )
        kir = {**ONE_INPUT, "p1": ["P1-2"]}
        (i,) = lkv_py.parse_lean(text, kir, 1)
        self.assertEqual(list(i["specs"]), ["P1-2"])
        self.assertEqual(i["specs"]["P1-2"]["outcome"], "FAIL")
        for f in i["specs"]["P1-2"]["counterexamples"]:
            f["source"], f["line_text"] = None, None
        report = self.report(PASSING, 0, p1_strict=True)
        report["variants_p1"] = [{"variant": "s.p1", "kernel": "k", "status": "checked",
                                  "p1_specs": ["P1-2"], "inputs": [i]}]  # fmt: skip
        report["p1_not_checked"] = [{"variant": "s", "status": "not_checked", "error": "x",
                                     "specs": ["P1-1"]}]  # fmt: skip
        report["totals_p1"] = lkv_py.totals(
            report["variants_p1"], lkv_py.LABELS_P1, lkv_py.unchecked(report, p1=True)
        )
        self.assertEqual(
            lkv_py.result(report)["fail"], ["P1-2: 1 certain violation(s) (P1-2: 1)"]
        )
        self.assertEqual(report["totals_p1"]["P1-1"]["FAIL"], 0)
        md = lkv_py.markdown(report)
        self.assertIn(
            "## P1-2 async store sources reuse ordered (P1; PTX translation;", md
        )

    def test_export_marker_names_the_revision(self):
        case = {"source": "k.py", "src_root": "src", "rev": "aaaaaaaaaaaa"}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(lkv_py.source_rev(case, root, None), "aaaaaaaaaaaa")
            self.assertIsNone(lkv_py.source_rev(case, root, d))
            mark = root / lkv_py.EXPORT_MARK
            mark.write_text('{"rev": "bbbbbbbbbbbb", "paths": []}')
            self.assertEqual(lkv_py.source_rev(case, root, d), "bbbbbbbbbbbb")

    def test_check_refuses_two_kir_files_with_one_name(self):
        # Both would be checked as out/lean/k.lean: one file's verdicts for both.
        with tempfile.TemporaryDirectory() as d:
            kirs = [Path(d) / s / "k.kir.json" for s in ("base", "fix")]
            with self.assertRaisesRegex(RuntimeError, "same name: k "):
                lkv_py.layer2(kirs, Path(d) / "out", None, 1, 1)
            self.assertFalse((Path(d) / "out").exists())

    def test_variant_inputs_of_a_geometry(self):
        # a to_kir failure line names `<spec>` or `<spec>@cluster<dims>`: the inputs of
        # the spec, those whose grid the dims divide (a bare `@cluster`: every one)
        la = lambda spec, grid: {"spec": spec, "launch": {"grid": grid}}  # noqa: E731
        dump = {"inputs": [{"name": "a", "launches": [la("s", [4, 1, 1])]},
                           {"name": "b", "launches": [la("s", [6, 1, 1])]},
                           {"name": "c", "launches": [la("t", [4, 1, 1])]}]}  # fmt: skip
        self.assertEqual(lkv_py.variant_inputs(dump, "s"), ["a", "b"])
        self.assertEqual(lkv_py.variant_inputs(dump, "s@cluster4x1x1"), ["a"])
        self.assertEqual(lkv_py.variant_inputs(dump, "s@cluster2x1x1"), ["a", "b"])
        self.assertEqual(lkv_py.variant_inputs(dump, "s@cluster"), ["a", "b"])

    def test_target_reaches_the_ttgir_dump(self):
        calls = []

        def buck_run(target, args, log_path):
            calls.append((target, list(map(str, args))))
            Path(log_path).write_text("stub")
            return 1

        real, lkv_py.buck_run = lkv_py.buck_run, buck_run
        try:
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaisesRegex(RuntimeError, "TTGIR dump failed"):
                    lkv_py.run_frontend(
                        Path(d) / "c.json", {}, Path(d), Path(d), None, "sm_103"
                    )
        finally:
            lkv_py.buck_run = real
        ((target, args),) = calls
        self.assertEqual(target, "frontend/ttgir:dump")
        self.assertEqual(args[args.index("--target") + 1], "sm_103")

    def test_geometry_failure_counts_for_its_spec(self):
        # a to_kir failure of another cluster geometry of a variant (`<spec>@cluster`:
        # launches that disagree on the preferred dims) counts for the inputs of <spec>,
        # next to the regular KIR
        def buck_run(target, args, log_path):
            args = list(map(str, args))
            if target == "frontend/ttgir:dump":
                la = {"spec": "s", "launch": {"grid": [4, 1, 1]}}
                dump = {"kernel": "k", "specs": {"s": {}},
                        "inputs": [{"name": "a", "launches": [la]}, {"name": "b", "launches": [la]}]}  # fmt: skip
                (Path(args[args.index("--out") + 1]) / "dump.json").write_text(
                    json.dumps(dump)
                )
                Path(log_path).write_text("")
                return 0
            (Path(args[args.index("--dump") + 1]).parent / "s.kir.json").write_text(
                "{}"
            )
            Path(log_path).write_text("[to_kir] s@cluster: FAILED (gap): launches differ in "
                                      "their (preferred) cluster dims\n")  # fmt: skip
            return 1

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(lkv_py, "buck_run", buck_run), \
                mock.patch.object(lkv_py, "NB_NOT_CHECKED", []), \
                mock.patch.object(lkv_py, "P1_NOT_CHECKED", []):  # fmt: skip
            kirs, failures = lkv_py.run_ttgir(Path(d) / "c.json", Path(d), Path(d),
                                              Path(d), [], "sm_100")  # fmt: skip
        self.assertEqual([k.name for k in kirs], ["s.kir.json"])
        self.assertEqual([(f["variant"], f["inputs"], f["kind"]) for f in failures],
                         [("s@cluster", ["a", "b"], "gap")])  # fmt: skip

    def test_lost_p0_6_kir_is_an_error(self):
        # to_kir stopped after its P0 lines (a P0 failure keeps lkv.py going): a spec
        # without a P0-6 KIR or `P0-6 FAILED` line, and a spec's all-preferred geometry
        # without one, get an ERROR entry instead of no verdict
        def buck_run(target, args, log_path):
            args = list(map(str, args))
            if target == "frontend/ttgir:dump":
                la = lambda spec, attrs: {"spec": spec, "launch": {  # noqa: E731
                    "grid": [4, 1, 1], "cluster": [1, 1, 1], "attrs": attrs}}  # fmt: skip
                launches = [
                    la("s", {}),
                    la("t", {}),
                    la("u", {"preferred_cluster": [2, 1, 1]}),
                ]
                dump = {"kernel": "k", "specs": {"s": {}, "t": {}, "u": {}},
                        "inputs": [{"name": "a", "launches": launches}]}  # fmt: skip
                (Path(args[args.index("--out") + 1]) / "dump.json").write_text(
                    json.dumps(dump)
                )
                Path(log_path).write_text("")
                return 0
            fe = Path(args[args.index("--dump") + 1]).parent
            for name in ("s.kir.json", "s.nb.json", "u.kir.json", "u.nb.json"):
                (fe / name).write_text("{}")
            Path(log_path).write_text(
                "[to_kir] t: FAILED (gap): x\nTraceback (most recent call last):\n"
            )
            return 1

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(lkv_py, "buck_run", buck_run), \
                mock.patch.object(lkv_py, "NB_NOT_CHECKED", []) as nb, \
                mock.patch.object(lkv_py, "P1_NOT_CHECKED", []):  # fmt: skip
            kirs, failures = lkv_py.run_ttgir(Path(d) / "c.json", Path(d), Path(d),
                                              Path(d), [], "sm_100")  # fmt: skip
        self.assertEqual(
            [k.name for k in kirs],
            ["s.kir.json", "u.kir.json", "s.nb.json", "u.nb.json"],
        )
        self.assertEqual([f["variant"] for f in failures], ["t"])
        self.assertEqual([(x["variant"], x["inputs"], x["kind"]) for x in nb],
                         [("t.nb", ["a"], "error"), ("u@cluster.nb", ["a"], "error")])  # fmt: skip
        report = {"variants_nb": [], "nb_not_checked": nb}
        self.assertEqual(
            [x["outcome"] for x in lkv_py.unchecked_nb(report)], ["ERROR", "ERROR"]
        )

    def test_run_outside_a_checkout_is_an_error(self):
        # A copy of the skill outside an fbsource checkout cannot build the frontend.
        with tempfile.TemporaryDirectory() as d:
            copy = Path(d) / "lean-kernel-verify"
            shutil.copytree(
                SKILL, copy, ignore=shutil.ignore_patterns("tests", "__pycache__")
            )
            r = subprocess.run(
                [sys.executable, "-I", str(copy / "scripts/lkv.py"), "run"]
                + [str(HERE / "cases/probe_cute.json"), "--out", str(Path(d) / "out")],
                capture_output=True,
                text=True,
            )
        self.assertEqual(r.returncode, 2, r.stderr)  # an ERROR; 1 would mean a FAIL
        self.assertIn("must be inside an fbsource checkout", r.stderr)

    def test_default_translation_per_family(self):
        # --frontend, else $LKV_FRONTEND, else ptx for TLX/Triton and legacy for CuTe
        ttgir, cute = {"src_root": "src"}, {"builder": "b.py"}
        with mock.patch.dict(os.environ):
            os.environ.pop("LKV_FRONTEND", None)
            self.assertEqual(lkv_py.translation_of(ttgir), "ptx")
            self.assertEqual(lkv_py.translation_of(cute), "legacy")
            self.assertEqual(lkv_py.translation_of(ttgir, "legacy"), "legacy")
            self.assertEqual(lkv_py.translation_of(cute, "ptx"), "ptx")
            os.environ["LKV_FRONTEND"] = "legacy"
            self.assertEqual(lkv_py.translation_of(ttgir), "legacy")
            self.assertEqual(lkv_py.translation_of(ttgir, "ptx"), "ptx")
            os.environ["LKV_FRONTEND"] = ""  # empty: unset
            self.assertEqual(lkv_py.translation_of(ttgir), "ptx")

    def test_cute_cases_refuse_sm103(self):
        case = HERE / "cases/probe_cute.json"
        r = subprocess.run(
            [sys.executable, "-I", str(LKV), "run", str(case), "--target", "sm_103"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 2)  # an ERROR (usage); 1 would mean a FAIL
        self.assertIn("the CuTe frontend compiles for sm_100 only", r.stderr)


class EndToEnd(unittest.TestCase):
    def test_lean_fixtures(self):
        fixtures = sorted(str(p) for p in (SKILL / "tests/lean").glob("*.kir.json"))
        report = lkv("check", "lean_fixtures", *fixtures, "--name", "lean_fixtures")
        want = {}
        for line in (SKILL / "tests/lean/expected.txt").read_text().splitlines():
            f = line.split()
            if f and not f[0].startswith("#"):
                want[(f[0], f[1])] = dict(zip(SPECS, f[2:7]))
        self.assertEqual(runs(report), want)
        self.assert_outcomes(report)
        self.assertEqual(report["result"]["outcome"], "FAIL")
        found = {}  # fixture -> {"notes": [...], "fails": [...]}, rendered findings
        for v in report["variants"]:
            kinds = found.setdefault(v["variant"], {"notes": [], "fails": []})
            for i in v["inputs"]:
                kinds["fails"] += [i["eval_error"]] if i["eval_error"] else []
                for s in i["specs"].values():
                    kinds["notes"] += map(render, s["notes"])
                    kinds["fails"] += map(render, s["counterexamples"])
        messages = SKILL / "tests/lean/expected_messages.txt"
        for line in messages.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            fixture, text = line.split(" ", 1)
            kinds = found[fixture]
            if text.startswith("note: "):
                text = text.removeprefix("note: ")
                self.assertTrue(any(text in n for n in kinds["notes"]), (fixture, text))
                self.assertFalse(
                    any(text in c for c in kinds["fails"]), (fixture, text)
                )
            else:
                self.assertTrue(any(text in c for c in kinds["fails"]), (fixture, text))

    def assert_outcomes(self, report):
        """Every evaluation error has the kind EVAL_ERROR_KINDS pins, and every spec of its
        input that kind's outcome; the other UNDECIDED outcomes are the specs of the inputs
        with an assumption that does not hold, an out-of-bounds metadata load (but P0-1) or
        a barrier ledger that does not converge that list nothing but the `not decided`
        line (its other lines are certain counterexamples), and P0-4 of the inputs whose
        peer CTA's bytes balance only if a landing completes a later phase than its issue
        order shows (LANDER_UNDECIDED), which list nothing but the `not decided` line;
        there is no other ERROR."""
        outcomes, evals, only_marker = {}, {}, set()
        for v, i in runs_of_report(report):
            if i["eval_error"]:
                evals[(v["variant"], i["name"])] = i["eval_kind"]
            for label, s in i["specs"].items():
                run = (v["variant"], i["name"], label)
                outcomes.setdefault(s["outcome"], set()).add(run)
                msgs = [c["msg"] for c in s["counterexamples"]]
                if (
                    msgs
                    and not s["more"]
                    and all(m.startswith(NOT_DECIDED_HEAD) for m in msgs)
                ):
                    only_marker.add(run)
        self.assertEqual(evals, EVAL_ERROR_KINDS)

        def of_kind(kind):
            return {
                (*x, label)
                for x, k in EVAL_ERROR_KINDS.items()
                if k == kind
                for label in SPECS
            }

        stopped = {
            ("meta_oob_load", "out_of_range"),
            ("eval_errors", "meta_oob"),
            ("meta_masked_load", "unmasked_empty"),
            ("meta_oob_elided", "out_of_range"),
        }
        assumed = {  # an assumption that does not hold (KIR change 16): P0-1 too
            ("assume", "fails"),
            ("assume", "oob_before"),
            ("assume", "dup_before"),
            ("assume_reader", "fails"),
            ("assume_reader", "fails_both"),
        }
        unconverged = {
            ("ledger_cycle", i) for i in ("extra0", "extra1", "hang_elsewhere")
        } | {("ledger_confirm", "at_budget")}
        self.assertTrue(of_kind("kernel") <= outcomes["FAIL"])
        self.assertEqual(outcomes["ERROR"], of_kind("unexpected"))
        self.assertEqual(
            outcomes["UNDECIDED"],
            of_kind("limit")
            | {
                (*x, label)
                for x in stopped | unconverged | assumed
                for label in (SPECS if x in assumed else SPECS[1:])
                if (*x, label) in only_marker
            }
            | LANDER_UNDECIDED,
        )
        self.assertLessEqual(LANDER_UNDECIDED, only_marker)
        # A stopped input FAILs a spec it has a certain counterexample for.
        self.assertIn(("meta_oob_load", "out_of_range", "P0-3"), outcomes["FAIL"])
        self.assertIn(("assume", "oob_before", "P0-1"), outcomes["FAIL"])
        self.assertIn(("assume", "dup_before", "P0-3"), outcomes["FAIL"])
        self.assertIn(("assume_reader", "fails_both", "P0-1"), outcomes["FAIL"])
        # An assumption that does not hold is no P0-1 finding (an assert would FAIL all),
        # and leaves P0-1 undecided, also for the other roles of its CTA.
        self.assertIn(("assume", "fails", "P0-1"), outcomes["UNDECIDED"])
        self.assertIn(("assume_reader", "fails", "P0-1"), outcomes["UNDECIDED"])
        self.assertIn(("meta_oob_elided", "out_of_range", "P0-2"), outcomes["FAIL"])
        # So does a CTA's certain hang beside a CTA whose ledger does not converge.
        self.assertIn(("ledger_cycle", "hang_elsewhere", "P0-2"), outcomes["FAIL"])
        self.assertIn(("ledger_cycle", "extra1", "P0-2"), outcomes["UNDECIDED"])

    def assert_shell_env_ignored(self, name, report):
        """The frontend dropped SHELL_ENV and the report lists the dropped values."""
        env = report["env"]
        self.assertEqual(env["case"], {})
        md = env_section((OUT / name / "report.md").read_text())
        for k, v in SHELL_ENV.items():
            self.assertEqual(env["unset"].get(k), v)
            self.assertNotIn(k, env["effective"])
            self.assertIn(f"`{k}={v}`", md)

    def assert_rejected(self, name, report, want):
        """report.json and report.md list exactly the inputs of `want` as rejected by the
        host, each with an error matching its pattern."""
        got = {r["input"]: r["error"] for r in report["host_rejected"]}
        self.assertEqual(set(got), set(want))
        md = (OUT / name / "report.md").read_text()
        rejected = md.split("## Inputs the host code rejected")[1].split("\n## ")[0]
        for inp, pattern in want.items():
            self.assertRegex(got[inp], pattern)
            self.assertIn(f"- `{inp}`: {got[inp]}", rejected)

    def test_probe_tlx(self):
        report = self.probe_tlx("probe_tlx")
        self.assertEqual(report["translation"], "ptx")  # the TLX/Triton default
        self.assertEqual(kir_irs(report), {"ptx"})

    def test_probe_tlx_legacy(self):
        report = self.probe_tlx("probe_tlx_legacy", "--frontend", "legacy")
        self.assertEqual(report["translation"], "legacy")
        self.assertEqual(kir_irs(report), {"ttgir"})
        md = (OUT / "probe_tlx_legacy/report.md").read_text()
        self.assertIn("- translation: `legacy`", md)

    def probe_tlx(self, out, *args):
        """`lkv.py run` of probe_tlx into OUT/out (args: lkv.py options) and its checks;
        returns the report."""
        case = str(HERE / "cases/probe_tlx.json")
        report = lkv("run", out, case, *args, env=SHELL_ENV)
        self.assert_shell_env_ignored(out, report)
        self.assertEqual(report["result"]["outcome"], "FAIL")  # the seeded bug
        got = runs(report)
        self.assertEqual(
            sorted(i for _, i in got),
            [
                "aligned",
                "below_block",
                "below_block_floor",
                "empty",
                "partial",
                "partial_floor",
            ],
        )
        for (_, name), verdicts in got.items():
            # A floor grid leaves the tail unwritten: P0-3 holes.
            want = {"P0-3"} if name.endswith("_floor") else set()
            self.assertEqual(failing(verdicts), want, name)
        self.assert_rejected(
            out,
            report,
            {
                "two_dim": r"^ValueError: probe takes a 1-D tensor at lkv_e2e_probe_tlx\.py:\d+$",
                "empty_skipped": r"^no launch of probe \(the host launched: nothing\)$",
            },
        )
        return report

    def test_probe_cute(self):
        report = self.probe_cute("probe_cute")
        self.assertEqual(report["translation"], "legacy")  # the default
        self.assertEqual(kir_irs(report), {"cute-mlir"})

    def test_probe_cute_ptx(self):
        report = self.probe_cute("probe_cute_ptx", "--frontend", "ptx")
        self.assertEqual(report["translation"], "ptx")
        self.assertEqual(kir_irs(report), {"ptx"})
        md = (OUT / "probe_cute_ptx/report.md").read_text()
        self.assertIn("- translation: `ptx`", md)

    def probe_cute(self, out, *args):
        """`lkv.py run` of probe_cute into OUT/out (args: lkv.py options) and its checks;
        returns the report."""
        case = str(HERE / "cases/probe_cute.json")
        report = lkv("run", out, case, *args, env=SHELL_ENV)
        self.assert_shell_env_ignored(out, report)
        self.assertEqual(report["result"]["outcome"], "FAIL")  # the seeded bug
        got = runs(report)
        self.assertEqual(sorted(i for _, i in got), ["n0", "n100", "n128"])
        for (_, name), verdicts in got.items():
            # n100 / n0: 28 / 128 threads read and write past the end (P0-1), and those
            # writes leave the intended region (P0-3).
            want = {"P0-1", "P0-3"} if name != "n128" else set()
            self.assertEqual(failing(verdicts), want, name)
        self.assert_rejected(
            out,
            report,
            {
                "n200": r"^ValueError: probe takes at most 128 elements at lkv_e2e_probe_cute\.py:\d+$"
            },
        )
        # The probe has no tensor memory: P1-1 does not apply, and nothing else changes.
        self.assertEqual(report["variants_p1"], [])
        self.assertEqual(
            {x["status"] for x in report["p1_not_checked"]}, {"not_applicable"}
        )
        return report


if __name__ == "__main__":
    unittest.main()
