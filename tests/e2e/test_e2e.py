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
  ERROR; the outcome and exit status are FAIL (1) on any FAIL, else ERROR (2) on any
  failure, else UNDECIDED (0) on anything a known limit leaves open, else PASS (0); P1-1
  ERRORs always count, its FAIL and UNDECIDED only with --p1-strict; inputs the host
  rejects count as none; an export's revision marker names the analyzed revision;
  --target reaches the TTGIR dump and is refused for CuTe cases (exit status 2, never 1);
  the report renders the frontend's environment record.
* lean_fixtures: `lkv.py check` on tests/lean/*.kir.json gives the verdicts of
  tests/lean/expected.txt, and lkv files every text of tests/lean/expected_messages.txt
  under the right kind: a `note: ` text among the notes and never among the
  counterexamples, any other text among the counterexamples or the evaluation error.
  Every evaluation error has the kind tests/lean/eval_errors and segmented_gload
  bad_layout pin, and every spec of its input that kind's outcome; the other UNDECIDED
  outcomes are exactly P0-2..P0-5 of the four inputs with an out-of-bounds metadata load
  (meta_oob_load out_of_range, eval_errors meta_oob, meta_masked_load unmasked_empty,
  meta_oob_elided out_of_range) where nothing but the `not decided` line is listed.
* probe_tlx / probe_cute: the probe kernels of kernels/ through the real TTGIR and CuTe
  frontends (`lkv.py run`), with degenerate inputs (an empty tensor, fewer elements than
  one block). The seeded bugs (a floor grid; threads past the end) must be the only FAILs.
  An input the host rejects (a `raise` in its validation) and, for TLX, one it makes no
  launch for are listed as rejected with the reason; the other inputs still get verdicts.
  The shell sets a TRITON_* and a CUTE_DSL_* variable: the frontend must drop both and
  the report must list them as ignored shell values.

Outputs: $LKV_E2E_OUT (default /tmp/lkv_e2e_$USER), one directory per test.
"""

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

sys.dont_write_bytecode = True  # keep the skill tree free of __pycache__
HERE = Path(__file__).resolve().parent
SKILL = HERE.parents[1]
LKV = SKILL / "scripts" / "lkv.py"
OUT = Path(os.environ.get("LKV_E2E_OUT", f"/tmp/lkv_e2e_{getpass.getuser()}"))
SPECS = [f"P0-{n}" for n in range(1, 6)]

_spec = importlib.util.spec_from_file_location("lkv", LKV)
lkv_py = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lkv_py)


def lkv(cmd: str, name: str, *args: str, env: dict | None = None) -> dict:
    """Run `lkv.py cmd args` into OUT/name, with `env` added to the environment; return
    its report.json, whose result gives the exit status (0, 1 and 2 all write one)."""
    out = OUT / name
    r = subprocess.run(
        [sys.executable, "-I", str(LKV), cmd, *args, "--out", str(out)],
        env={**os.environ, **(env or {})},
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
    "other lines of this spec are certain counterexamples"
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

    def test_notes_and_host_rejected_inputs_keep_exit_status_0(self):
        rejected = [{"input": "b", "error": "the host rejected input b: ValueError"}]
        report = self.report(PASSING, 0, host_rejected=rejected)
        self.assertEqual(lkv_py.exit_status(report), 0)
        md = lkv_py.markdown(report)
        self.assertIn("## Result: PASS (exit status 0)", md)
        self.assertIn("## Inputs the host code rejected", md)

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

    def test_export_marker_names_the_revision(self):
        case = {"source": "k.py", "src_root": "src", "rev": "aaaaaaaaaaaa"}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(lkv_py.source_rev(case, root, None), "aaaaaaaaaaaa")
            self.assertIsNone(lkv_py.source_rev(case, root, d))
            mark = root / lkv_py.EXPORT_MARK
            mark.write_text('{"rev": "bbbbbbbbbbbb", "paths": []}')
            self.assertEqual(lkv_py.source_rev(case, root, d), "bbbbbbbbbbbb")

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
        with an out-of-bounds metadata load that list nothing but the `not decided` line
        (its other lines are certain counterexamples), and there is no other ERROR."""
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
        self.assertTrue(of_kind("kernel") <= outcomes["FAIL"])
        self.assertEqual(outcomes["ERROR"], of_kind("unexpected"))
        self.assertEqual(
            outcomes["UNDECIDED"],
            of_kind("limit")
            | {
                (*x, label)
                for x in stopped
                for label in SPECS[1:]
                if (*x, label) in only_marker
            },
        )
        # A stopped input FAILs a spec it has a certain counterexample for.
        self.assertIn(("meta_oob_load", "out_of_range", "P0-3"), outcomes["FAIL"])
        self.assertIn(("meta_oob_elided", "out_of_range", "P0-2"), outcomes["FAIL"])

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
        case = str(HERE / "cases/probe_tlx.json")
        report = lkv("run", "probe_tlx", case, env=SHELL_ENV)
        self.assert_shell_env_ignored("probe_tlx", report)
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
            "probe_tlx",
            report,
            {
                "two_dim": r"^ValueError: probe takes a 1-D tensor at lkv_e2e_probe_tlx\.py:\d+$",
                "empty_skipped": r"^no launch of probe \(the host launched: nothing\)$",
            },
        )

    def test_probe_cute(self):
        case = str(HERE / "cases/probe_cute.json")
        report = lkv("run", "probe_cute", case, env=SHELL_ENV)
        self.assert_shell_env_ignored("probe_cute", report)
        self.assertEqual(report["result"]["outcome"], "FAIL")  # the seeded bug
        got = runs(report)
        self.assertEqual(sorted(i for _, i in got), ["n0", "n100", "n128"])
        for (_, name), verdicts in got.items():
            # n100 / n0: 28 / 128 threads read and write past the end (P0-1), and those
            # writes leave the intended region (P0-3).
            want = {"P0-1", "P0-3"} if name != "n128" else set()
            self.assertEqual(failing(verdicts), want, name)
        self.assert_rejected(
            "probe_cute",
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


if __name__ == "__main__":
    unittest.main()
