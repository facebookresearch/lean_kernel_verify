#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# run_all.sh: Layer-2 test suite (KIR validator, Lean evaluator, P0 and P1 specs).
#
# Usage: tests/lean/run_all.sh     exit 0 iff every check below passes
#
# 1. kir_negative.py: invalid KIR variants are rejected by frontend/kir.py.
# 2. For every tests/lean/*.kir.json fixture: kir.py accepts it, emit_lean.py turns it into
#    Lean, scripts/run_lean_check.sh checks that file, and
#    * the VERDICT lines printed by KV.runCase equal expected.txt (every input, every spec);
#    * the THEOREM lines of #lkv_check (lean/KV/Command.lean: `KV.Holds` proved for a PASS,
#      its negation for a FAIL) equal the same verdicts, there are no Lean errors, and lean
#      exits 0;
#    * the report contains the counterexample and note text listed in
#      expected_messages.txt (notes are printed as "note: ..." lines and never change a
#      verdict, so a PASS row with an expected note checks that a finding stays a note);
#    * the report contains every line of report_checks.txt for the fixture, and none of its
#      `!`-prefixed ones (texts that must be absent, and lines other than counterexamples
#      and notes, e.g. summaries and NOTES counts).
# 3. The same for every tests/lean/p1/*.kir.json fixture (P1 KIR, P1-1 TMEM reuse ordered;
#    written by p1/make_fixtures.py; its P1-1 lines, the KIR may name P1-2 too) against
#    p1/expected.txt (one P1-1 column),
#    p1/expected_messages.txt and p1/report_checks.txt, and for every
#    tests/lean/p1_2/*.kir.json fixture (P1-2 async store sources reuse ordered; written by
#    p1_2/make_fixtures.py) against p1_2/expected.txt (one P1-2 column) and its message files;
#    and p1_2/pairs_eq.py: on every input of those fixtures, StoreReuse.pairs (the cursor
#    search) finds exactly the pairs of its definition.
# 4. The same for every tests/lean/nb/*.kir.json fixture (P0-6 KIR, named barriers
#    convergent; written by nb/make_fixtures.py) against nb/expected.txt (one P0-6 column),
#    nb/expected_messages.txt and nb/report_checks.txt.
# 5. The same for every tests/lean/gsync/*.kir.json fixture (global synchronization, kir.md
#    change 18; written by gsync/make_fixtures.py) against gsync/expected.txt (P0-1..P0-5 and
#    P0-7) and gsync/expected_messages.txt.
#
# Generated Lean files and outputs are kept in $LKV_TEST_OUT (default ${TMPDIR:-/tmp}/lkv_lean_tests_$USER).
set -uo pipefail

T="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="$(cd "$T/../.." && pwd)"
OUT="${LKV_TEST_OUT:-${TMPDIR:-/tmp}/lkv_lean_tests_$(id -un)}"
mkdir -p "$OUT"

failures=0
bad() { echo "run_all: FAIL: $*"; failures=$((failures + 1)); }

# check_theorems NAME OUT WANT STATUS [LABEL]: the THEOREM lines of OUT (with LABEL, those of
# that spec) are the VERDICT lines WANT (one theorem per input and spec, the negation for a
# FAIL), no Lean error, exit status 0.
check_theorems() {
  local want_thm got_thm nerr
  # shellcheck disable=SC2001  # a per-line anchor
  want_thm="$(sed 's/^VERDICT /THEOREM /' <<<"$3" | sort)"
  got_thm="$(grep "^THEOREM [^ ]* ${5:-[^ ]*} " "$2" | sort)"
  nerr="$(grep -c ': error' "$2")"
  if [[ "$want_thm" != "$got_thm" ]] || (( nerr != 0 )); then
    bad "$1: theorems differ from the expected verdicts, or Lean errors ($nerr errors; see $2)"
    diff <(echo "$want_thm") <(echo "$got_thm") | grep '^[<>]' | sed 's/^/    /'
  fi
  if (( $4 != 0 )); then
    bad "$1: lean exit status $4"
  fi
}

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

  check_theorems "$n" "$OUT/$n.out" "$want" "$status"

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
  got="$(grep '^VERDICT [^ ]* P1-1 ' "$OUT/p1_$n.out" | sort)"
  if [[ -z "$want" ]]; then
    bad "p1/$n: no rows in p1/expected.txt"
    continue
  fi
  if [[ "$want" != "$got" ]]; then
    bad "p1/$n: verdicts differ from p1/expected.txt (< expected, > got; output in $OUT/p1_$n.out)"
    diff <(echo "$want") <(echo "$got") | grep '^[<>]' | sed 's/^/    /'
  fi
  nverdicts=$((nverdicts + $(grep -c . <<<"$want")))
  check_theorems "p1/$n" "$OUT/p1_$n.out" "$want" "$status" P1-1
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

# P1-2 fixtures: one P1-2 verdict per input (#lkv_check_p1, theorem KV.Holds .storeReuse).
P12="$T/p1_2"
for f in "$P12"/*.kir.json; do
  n="$(basename "$f" .kir.json)"
  nfix=$((nfix + 1))
  python3 -I "$S/frontend/kir.py" "$f" || { bad "p1_2/$n: invalid KIR"; continue; }
  python3 -I "$S/frontend/emit_lean.py" "$f" -o "$OUT/p1_2_$n.lean" || { bad "p1_2/$n: emit_lean.py failed"; continue; }
  "$S/scripts/run_lean_check.sh" "$OUT/p1_2_$n.lean" >"$OUT/p1_2_$n.out" 2>&1
  status=$?
  want="$(awk -v f="$n" '$1 == f { printf "VERDICT %s P1-2 %s\n", $2, $3 }' "$P12/expected.txt" | sort)"
  got="$(grep '^VERDICT ' "$OUT/p1_2_$n.out" | sort)"
  if [[ -z "$want" ]]; then
    bad "p1_2/$n: no rows in p1_2/expected.txt"
    continue
  fi
  if [[ "$want" != "$got" ]]; then
    bad "p1_2/$n: verdicts differ from p1_2/expected.txt (< expected, > got; output in $OUT/p1_2_$n.out)"
    diff <(echo "$want") <(echo "$got") | grep '^[<>]' | sed 's/^/    /'
  fi
  nverdicts=$((nverdicts + $(grep -c . <<<"$want")))
  check_theorems "p1_2/$n" "$OUT/p1_2_$n.out" "$want" "$status" P1-2
  while IFS=' ' read -r fx msg; do
    [[ "$fx" == "$n" ]] || continue
    grep -qF -- "$msg" "$OUT/p1_2_$n.out" || bad "p1_2/$n: report lacks: $msg"
  done < <(grep -v '^#' "$P12/expected_messages.txt")
  while IFS=' ' read -r fx line; do
    [[ "$fx" == "$n" ]] || continue
    if [[ "$line" == '!'* ]]; then
      if grep -qF -- "${line:1}" "$OUT/p1_2_$n.out"; then bad "p1_2/$n: report contains: ${line:1}"; fi
    else
      grep -qF -- "$line" "$OUT/p1_2_$n.out" || bad "p1_2/$n: report lacks: $line"
    fi
  done < <(grep -v '^#' "$P12/report_checks.txt")
done
python3 -I "$P12/pairs_eq.py" "$OUT/p1_2_pairs" "$P12"/*.kir.json ||
  bad "p1_2: StoreReuse.pairs differs from its definition (pairs_eq.py)"

# P0-6 fixtures: one P0-6 verdict per input (#lkv_check_nb, theorem KV.Holds .namedBarriers).
NB="$T/nb"
for f in "$NB"/*.kir.json; do
  n="$(basename "$f" .kir.json)"
  nfix=$((nfix + 1))
  python3 -I "$S/frontend/kir.py" "$f" || { bad "nb/$n: invalid KIR"; continue; }
  python3 -I "$S/frontend/emit_lean.py" "$f" -o "$OUT/nb_$n.lean" || { bad "nb/$n: emit_lean.py failed"; continue; }
  "$S/scripts/run_lean_check.sh" "$OUT/nb_$n.lean" >"$OUT/nb_$n.out" 2>&1
  status=$?
  want="$(awk -v f="$n" '$1 == f { printf "VERDICT %s P0-6 %s\n", $2, $3 }' "$NB/expected.txt" | sort)"
  got="$(grep '^VERDICT ' "$OUT/nb_$n.out" | sort)"
  if [[ -z "$want" ]]; then
    bad "nb/$n: no rows in nb/expected.txt"
    continue
  fi
  if [[ "$want" != "$got" ]]; then
    bad "nb/$n: verdicts differ from nb/expected.txt (< expected, > got; output in $OUT/nb_$n.out)"
    diff <(echo "$want") <(echo "$got") | grep '^[<>]' | sed 's/^/    /'
  fi
  nverdicts=$((nverdicts + $(grep -c . <<<"$want")))
  check_theorems "nb/$n" "$OUT/nb_$n.out" "$want" "$status" P0-6
  while IFS=' ' read -r fx msg; do
    [[ "$fx" == "$n" ]] || continue
    grep -qF -- "$msg" "$OUT/nb_$n.out" || bad "nb/$n: report lacks: $msg"
  done < <(grep -v '^#' "$NB/expected_messages.txt")
  while IFS=' ' read -r fx line; do
    [[ "$fx" == "$n" ]] || continue
    if [[ "$line" == '!'* ]]; then
      if grep -qF -- "${line:1}" "$OUT/nb_$n.out"; then bad "nb/$n: report contains: ${line:1}"; fi
    else
      grep -qF -- "$line" "$OUT/nb_$n.out" || bad "nb/$n: report lacks: $line"
    fi
  done < <(grep -v '^#' "$NB/report_checks.txt")
done

# Global-synchronization fixtures (kir.md change 18; written by gsync/make_fixtures.py): the five
# P0 verdicts, then P0-7, per input against gsync/expected.txt.
GS="$T/gsync"
for f in "$GS"/*.kir.json; do
  n="$(basename "$f" .kir.json)"
  nfix=$((nfix + 1))
  python3 -I "$S/frontend/kir.py" "$f" || { bad "gsync/$n: invalid KIR"; continue; }
  python3 -I "$S/frontend/emit_lean.py" "$f" -o "$OUT/gsync_$n.lean" || { bad "gsync/$n: emit_lean.py failed"; continue; }
  "$S/scripts/run_lean_check.sh" "$OUT/gsync_$n.lean" >"$OUT/gsync_$n.out" 2>&1
  status=$?
  want="$(awk -v f="$n" 'BEGIN { split("1 2 3 4 5 7", l, " ") }
    $1 == f { for (i = 3; i <= 8; i++) printf "VERDICT %s P0-%s %s\n", $2, l[i - 2], $i }' "$GS/expected.txt" | sort)"
  got="$(grep '^VERDICT ' "$OUT/gsync_$n.out" | sort)"
  if [[ -z "$want" ]]; then
    bad "gsync/$n: no rows in gsync/expected.txt"
    continue
  fi
  if [[ "$want" != "$got" ]]; then
    bad "gsync/$n: verdicts differ from gsync/expected.txt (< expected, > got; output in $OUT/gsync_$n.out)"
    diff <(echo "$want") <(echo "$got") | grep '^[<>]' | sed 's/^/    /'
  fi
  nverdicts=$((nverdicts + $(grep -c . <<<"$want")))
  check_theorems "gsync/$n" "$OUT/gsync_$n.out" "$want" "$status"
  while IFS=' ' read -r fx msg; do
    [[ "$fx" == "$n" ]] || continue
    grep -qF -- "$msg" "$OUT/gsync_$n.out" || bad "gsync/$n: report lacks: $msg"
  done < <(grep -v '^#' "$GS/expected_messages.txt")
done

if (( failures )); then
  echo "run_all: $failures check(s) failed ($nfix fixtures, outputs in $OUT)"
  exit 1
fi
echo "run_all: OK: $nfix fixtures, $nverdicts (input, spec) verdicts as expected, theorems match, counterexample, note and report-check texts as expected"
