#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# run_all.sh: Layer-2 test suite (KIR validator, Lean evaluator, P0 specs).
#
# Usage: tests/lean/run_all.sh     exit 0 iff every check below passes
#
# 1. kir_negative.py: invalid KIR variants are rejected by frontend/kir.py.
# 2. For every tests/lean/*.kir.json fixture: kir.py accepts it, emit_lean.py turns it into
#    Lean, scripts/run_lean_check.sh checks that file, and
#    * the VERDICT lines printed by KV.runCase equal expected.txt (every input, every spec);
#    * the theorems native_decide refutes are exactly the expected FAILs, there are no
#      other Lean errors, and lean exits non-zero iff some FAIL is expected;
#    * the report contains the counterexample and note text listed in
#      expected_messages.txt (notes are printed as "note: ..." lines and never change a
#      verdict, so a PASS row with an expected note checks that a finding stays a note);
#    * the report contains every line of report_checks.txt for the fixture, and none of its
#      `!`-prefixed ones (texts that must be absent, and lines other than counterexamples
#      and notes, e.g. summaries and NOTES counts).
# 3. The same for every tests/lean/p1/*.kir.json fixture (P1 KIR, P1-1 TMEM reuse ordered;
#    written by p1/make_fixtures.py) against p1/expected.txt (one P1-1 column),
#    p1/expected_messages.txt and p1/report_checks.txt.
#
# Generated Lean files and outputs are kept in $LKV_TEST_OUT (default /tmp/lkv_lean_tests_$USER).
set -uo pipefail

T="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="$(cd "$T/../.." && pwd)"
OUT="${LKV_TEST_OUT:-/tmp/lkv_lean_tests_$(id -un)}"
mkdir -p "$OUT"

failures=0
bad() { echo "run_all: FAIL: $*"; failures=$((failures + 1)); }

python3 -I "$T/kir_negative.py" || bad "kir_negative.py"
"$S/scripts/build_lean.sh" >/dev/null || { bad "scripts/build_lean.sh"; exit 1; }

nfix=0
nverdicts=0
for f in "$T"/*.kir.json; do
  n="$(basename "$f" .kir.json)"
  nfix=$((nfix + 1))
  python3 -I "$S/frontend/kir.py" "$f" || { bad "$n: invalid KIR"; continue; }
  python3 -I "$S/frontend/emit_lean.py" "$f" -o "$OUT/$n.lean" || { bad "$n: emit_lean.py failed"; continue; }
  "$S/scripts/run_lean_check.sh" "$OUT/$n.lean" >"$OUT/$n.out" 2>&1
  status=$?

  want="$(awk -v f="$n" '$1 == f { for (i = 3; i <= 7; i++) printf "VERDICT %s P0-%d %s\n", $2, i - 2, $i }' \
    "$T/expected.txt" | sort)"
  got="$(grep '^VERDICT ' "$OUT/$n.out" | sort)"
  if [[ -z "$want" ]]; then
    bad "$n: no rows in expected.txt"
    continue
  fi
  if [[ "$want" != "$got" ]]; then
    bad "$n: verdicts differ from expected.txt (< expected, > got; output in $OUT/$n.out)"
    diff <(echo "$want") <(echo "$got") | grep '^[<>]' | sed 's/^/    /'
  fi
  nverdicts=$((nverdicts + $(grep -c . <<<"$want")))

  want_thm="$(awk -v f="$n" 'BEGIN { split("bounds tripCount coverage barrier maskElision", s, " ") }
    $1 == f { for (i = 3; i <= 7; i++) if ($i == "FAIL") printf "%s %s\n", s[i - 2], $2 }' "$T/expected.txt" | sort)"
  got_thm="$(sed -n 's/^ *KV\.Holds KV\.SpecId\.\([A-Za-z]*\) kernel input[0-9]*_\(.*\)$/\1 \2/p' "$OUT/$n.out" | sort)"
  nerr="$(grep -c ': error' "$OUT/$n.out")"
  if [[ "$want_thm" != "$got_thm" ]] || (( nerr != $(grep -c . <<<"$got_thm") )); then
    bad "$n: refuted theorems differ from the expected FAILs, or other Lean errors ($nerr errors; see $OUT/$n.out)"
  fi
  if (( (status != 0) != ($(grep -c . <<<"$want_thm") > 0) )); then
    bad "$n: lean exit status $status does not match the expected verdicts"
  fi

  while IFS=' ' read -r fx msg; do
    [[ "$fx" == "$n" ]] || continue
    grep -qF -- "$msg" "$OUT/$n.out" || bad "$n: report lacks: $msg"
  done < <(grep -v '^#' "$T/expected_messages.txt")

  while IFS=' ' read -r fx line; do
    [[ "$fx" == "$n" ]] || continue
    if [[ "$line" == '!'* ]]; then
      if grep -qF -- "${line:1}" "$OUT/$n.out"; then bad "$n: report contains: ${line:1}"; fi
    else
      grep -qF -- "$line" "$OUT/$n.out" || bad "$n: report lacks: $line"
    fi
  done < <(grep -v '^#' "$T/report_checks.txt")
done

# P1 fixtures: one P1-1 verdict per input (KV.runP1, theorem KV.Holds .tmemReuse).
P1="$T/p1"
for f in "$P1"/*.kir.json; do
  n="$(basename "$f" .kir.json)"
  nfix=$((nfix + 1))
  python3 -I "$S/frontend/kir.py" "$f" || { bad "p1/$n: invalid KIR"; continue; }
  python3 -I "$S/frontend/emit_lean.py" "$f" -o "$OUT/p1_$n.lean" || { bad "p1/$n: emit_lean.py failed"; continue; }
  "$S/scripts/run_lean_check.sh" "$OUT/p1_$n.lean" >"$OUT/p1_$n.out" 2>&1
  status=$?
  want="$(awk -v f="$n" '$1 == f { printf "VERDICT %s P1-1 %s\n", $2, $3 }' "$P1/expected.txt" | sort)"
  got="$(grep '^VERDICT ' "$OUT/p1_$n.out" | sort)"
  if [[ -z "$want" ]]; then
    bad "p1/$n: no rows in p1/expected.txt"
    continue
  fi
  if [[ "$want" != "$got" ]]; then
    bad "p1/$n: verdicts differ from p1/expected.txt (< expected, > got; output in $OUT/p1_$n.out)"
    diff <(echo "$want") <(echo "$got") | grep '^[<>]' | sed 's/^/    /'
  fi
  nverdicts=$((nverdicts + $(grep -c . <<<"$want")))
  want_thm="$(awk -v f="$n" '$1 == f && $3 == "FAIL" { printf "tmemReuse %s\n", $2 }' "$P1/expected.txt" | sort)"
  got_thm="$(sed -n 's/^ *KV\.Holds KV\.SpecId\.\([A-Za-z]*\) kernel input[0-9]*_\(.*\)$/\1 \2/p' "$OUT/p1_$n.out" | sort)"
  nerr="$(grep -c ': error' "$OUT/p1_$n.out")"
  if [[ "$want_thm" != "$got_thm" ]] || (( nerr != $(grep -c . <<<"$got_thm") )); then
    bad "p1/$n: refuted theorems differ from the expected FAILs, or other Lean errors ($nerr errors; see $OUT/p1_$n.out)"
  fi
  if (( (status != 0) != ($(grep -c . <<<"$want_thm") > 0) )); then
    bad "p1/$n: lean exit status $status does not match the expected verdicts"
  fi
  while IFS=' ' read -r fx msg; do
    [[ "$fx" == "$n" ]] || continue
    grep -qF -- "$msg" "$OUT/p1_$n.out" || bad "p1/$n: report lacks: $msg"
  done < <(grep -v '^#' "$P1/expected_messages.txt")
  while IFS=' ' read -r fx line; do
    [[ "$fx" == "$n" ]] || continue
    if [[ "$line" == '!'* ]]; then
      if grep -qF -- "${line:1}" "$OUT/p1_$n.out"; then bad "p1/$n: report contains: ${line:1}"; fi
    else
      grep -qF -- "$line" "$OUT/p1_$n.out" || bad "p1/$n: report lacks: $line"
    fi
  done < <(grep -v '^#' "$P1/report_checks.txt")
done

if (( failures )); then
  echo "run_all: $failures check(s) failed ($nfix fixtures, outputs in $OUT)"
  exit 1
fi
echo "run_all: OK: $nfix fixtures, $nverdicts (input, spec) verdicts as expected, refuted theorems match, counterexample, note and report-check texts as expected"
