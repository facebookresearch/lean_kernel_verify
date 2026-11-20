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
# 2. For every tests/lean/*.kir.json fixture: emit_lean.py validates it (frontend/kir.py) and
#    turns it into Lean, scripts/run_lean_check.sh checks that file, and
#    * the VERDICT lines printed by KV.runCase equal expected.txt (every input, every spec;
#      a FAIL for which Lean also prints `UNDECIDED <input> <spec>`, a refuted spec without a
#      certain counterexample, is UNDECIDED there);
#    * the THEOREM lines of #lkv_check (lean/KV/Command.lean: `KV.Holds` proved for a PASS,
#      its negation for a FAIL or UNDECIDED) equal the same verdicts, there are no Lean
#      errors, and lean exits 0;
#    * the report contains the counterexample and note text listed in
#      expected_messages.txt (notes are printed as "note: ..." lines and never change a
#      verdict, so a PASS row with an expected note checks that a finding stays a note);
#    * the report contains every line of report_checks.txt for the fixture, and none of its
#      `!`-prefixed ones (texts that must be absent, and lines other than counterexamples
#      and notes, e.g. summaries and NOTES counts).
# 3. The same for every tests/lean/p1 fixture (P1 KIR, P1-1 TMEM reuse ordered;
#    written by p1/make_fixtures.py into $LKV_TEST_OUT/gen/p1; its P1-1 lines, the KIR may
#    name P1-2 too) against p1/expected.txt (one P1-1 column),
#    p1/expected_messages.txt and p1/report_checks.txt, and for every
#    tests/lean/p1_2 fixture (P1-2 async store sources reuse ordered; written by
#    p1_2/make_fixtures.py into $LKV_TEST_OUT/gen/p1_2) against p1_2/expected.txt (one P1-2
#    column) and its message files;
#    and p1_2/pairs_eq.py, in each of those fixtures' own Lean run: on every input,
#    StoreReuse.pairs (the cursor search) finds exactly the pairs of its definition.
# 4. The same for every tests/lean/nb fixture (P0-6 KIR, named barriers convergent; written
#    by nb/make_fixtures.py into $LKV_TEST_OUT/gen/nb) against nb/expected.txt (one P0-6
#    column), nb/expected_messages.txt and nb/report_checks.txt.
# 5. The same for every tests/lean/gsync fixture (global synchronization, kir.md change 18;
#    written by gsync/make_fixtures.py into $LKV_TEST_OUT/gen/gsync) against gsync/expected.txt
#    (P0-1..P0-5 and P0-7) and gsync/expected_messages.txt.
# 6. fast_paths.lean: the evaluator's lane loops give what the zips and lane-state loops they
#    replace give, on synthetic operands. Self-checks (lean/KV/SelfCheck.lean, written after the
#    checks by emit_lean.py --with-selfcheck, so in the same Lean run) on every fixture of
#    2.-5.: on every input, the evaluation (CTAs in parallel
#    tasks, the fast lane loops) gives the same run, event by event, as the sequential one with
#    the reference loops, and every spec's result in the report (from the shared analyses) is
#    the one computed from scratch; one `SELFCHECK <input> OK` line per input. emit_shared.py: the
#    emitted Lean's shared terms (`sx<n>`) are defined before their uses and emits from
#    several threads equal the serial ones.
#
# Generated fixtures, Lean files and outputs are kept in $LKV_TEST_OUT (default ${TMPDIR:-/tmp}/lkv_lean_tests_$USER).
set -uo pipefail

T="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="$(cd "$T/../.." && pwd)"
OUT="${LKV_TEST_OUT:-${TMPDIR:-/tmp}/lkv_lean_tests_$(id -un)}"
mkdir -p "$OUT"

failures=0
bad() { echo "run_all: FAIL: $*"; failures=$((failures + 1)); }

# check_theorems NAME OUT WANT STATUS LABELS: the THEOREM lines of OUT of the specs LABELS
# (an extended regular expression) are the VERDICT lines WANT (one theorem per input and spec,
# the negation for a FAIL or UNDECIDED), no Lean error, exit status 0.
check_theorems() {
  local want_thm got_thm nerr
  # shellcheck disable=SC2001  # a per-line anchor
  want_thm="$(sed 's/^VERDICT /THEOREM /; s/ UNDECIDED$/ FAIL/' <<<"$3" | sort)"
  got_thm="$(grep -E "^THEOREM [^ ]* $5 " "$2" | sort)"
  nerr="$(grep -c ': error' "$2")"
  if [[ "$want_thm" != "$got_thm" ]] || (( nerr != 0 )); then
    bad "$1: theorems differ from the expected verdicts, or Lean errors ($nerr errors; see $2)"
    diff <(echo "$want_thm") <(echo "$got_thm") | grep '^[<>]' | sed 's/^/    /'
  fi
  if (( $4 != 0 )); then
    bad "$1: lean exit status $4"
  fi
}

# check_self NAME LEAN OUT: one `SELFCHECK <input> OK` line in OUT per self-check of LEAN
# (emit_lean.py --with-selfcheck; p1_2/pairs_eq.py's `PairsEq.check`), and no DIFF line.
nself=0
npairs=0
check_self() {
  local want pairs got
  want="$(grep -c '^#eval KV.SelfCheck.run' "$2")"
  pairs="$(grep -c '^#eval PairsEq.check' "$2")"
  got="$(grep -c '^SELFCHECK .* OK' "$3")"
  if (( want + pairs != got )) || grep -q '^SELFCHECK .* DIFF' "$3"; then
    bad "$1: self-check: $got of $((want + pairs)) OK (output in $3)"
    grep '^SELFCHECK .* DIFF' "$3" | sed 's/^/    /'
  fi
  nself=$((nself + want))
  npairs=$((npairs + pairs))
}

python3 -I "$T/kir_negative.py" || bad "kir_negative.py"
python3 -I "$T/emit_shared.py" || bad "emit_shared.py"
"$S/scripts/build_lean.sh" >/dev/null || { bad "scripts/build_lean.sh"; exit 1; }

# check_fixtures SUB KIRS LABEL...: every fixture KIRS/<fixture>.kir.json against the tables of
# tests/lean/SUB (tests/lean itself for an empty SUB): its VERDICT and THEOREM lines of the specs
# LABEL... against expected.txt (one column per LABEL), its report against expected_messages.txt
# and report_checks.txt (if SUB has one), its self-checks; Lean file and output
# $OUT/[SUB_]<fixture>.lean, .out.
nfix=0
nverdicts=0
check_fixtures() {
  local sub="$1" kirs="$2"
  shift 2
  local dir="$T${sub:+/$sub}" labels="$*" re f n name lean out status want got fx line
  re="(${labels// /|})"
  local checks=("$dir/expected_messages.txt")
  [[ -f "$dir/report_checks.txt" ]] && checks+=("$dir/report_checks.txt")
  for f in "$kirs"/*.kir.json; do
    n="$(basename "$f" .kir.json)"
    name="${sub:+$sub/}$n"
    lean="$OUT/${sub:+${sub}_}$n.lean"
    out="$OUT/${sub:+${sub}_}$n.out"
    nfix=$((nfix + 1))
    python3 -I "$S/frontend/emit_lean.py" "$f" --with-selfcheck -o "$lean" ||
      { bad "$name: invalid KIR or emit_lean.py failed (its errors above)"; continue; }
    if [[ "$sub" == p1_2 ]]; then
      python3 -I "$T/p1_2/pairs_eq.py" "$lean" || { bad "$name: pairs_eq.py failed"; continue; }
    fi
    "$S/scripts/run_lean_check.sh" "$lean" >"$out" 2>&1
    status=$?

    want="$(awk -v f="$n" -v labels="$labels" 'BEGIN { k = split(labels, L, " ") }
      $1 == f { for (i = 1; i <= k; i++) printf "VERDICT %s %s %s\n", $2, L[i], $(i + 2) }' \
      "$dir/expected.txt" | sort)"
    # The VERDICT lines of the specs LABEL..., UNDECIDED where Lean printed an UNDECIDED line.
    got="$(awk -v re="^$re\$" '$1 == "UNDECIDED" { u[$2 " " $3] = 1 }
      $1 == "VERDICT" && $3 ~ re { k[++n] = $2 " " $3; v[n] = $4 }
      END { for (i = 1; i <= n; i++) print "VERDICT " k[i] " " ((k[i] in u) ? "UNDECIDED" : v[i]) }' \
      "$out" | sort)"
    if [[ -z "$want" ]]; then
      bad "$name: no rows in ${sub:+$sub/}expected.txt"
      continue
    fi
    if [[ "$want" != "$got" ]]; then
      bad "$name: verdicts differ from ${sub:+$sub/}expected.txt (< expected, > got; output in $out)"
      diff <(echo "$want") <(echo "$got") | grep '^[<>]' | sed 's/^/    /'
    fi
    nverdicts=$((nverdicts + $(grep -c . <<<"$want")))

    check_theorems "$name" "$out" "$want" "$status" "$re"
    check_self "$name" "$lean" "$out"

    while IFS=' ' read -r fx line; do
      [[ "$fx" == "$n" ]] || continue
      if [[ "$line" == '!'* ]]; then
        if grep -qF -- "${line:1}" "$out"; then bad "$name: report contains: ${line:1}"; fi
      else
        grep -qF -- "$line" "$out" || bad "$name: report lacks: $line"
      fi
    done < <(grep -hv '^#' "${checks[@]}")
  done
}

for sub in p1 p1_2 nb gsync; do
  python3 -I "$T/$sub/make_fixtures.py" "$OUT/gen/$sub" || bad "$sub/make_fixtures.py"
done
check_fixtures "" "$T" P0-1 P0-2 P0-3 P0-4 P0-5
# P1 fixtures: one P1-1 verdict per input (KV.runP1, theorem KV.Holds .tmemReuse).
check_fixtures p1 "$OUT/gen/p1" P1-1
# P1-2 fixtures: one P1-2 verdict per input (#lkv_check_p1, theorem KV.Holds .storeReuse);
# check_fixtures adds p1_2/pairs_eq.py's check to each fixture's Lean file.
check_fixtures p1_2 "$OUT/gen/p1_2" P1-2
# P0-6 fixtures: one P0-6 verdict per input (#lkv_check_nb, theorem KV.Holds .namedBarriers).
check_fixtures nb "$OUT/gen/nb" P0-6

# Global-synchronization fixtures (kir.md change 18; written by gsync/make_fixtures.py): the five
# P0 verdicts, then P0-7, per input against gsync/expected.txt.
check_fixtures gsync "$OUT/gen/gsync" P0-1 P0-2 P0-3 P0-4 P0-5 P0-7

# 6. self-checks
"$S/scripts/run_lean_check.sh" "$T/fast_paths.lean" >"$OUT/fast_paths.out" 2>&1 || bad "fast_paths.lean (output in $OUT/fast_paths.out)"
grep -q "error" "$OUT/fast_paths.out" && bad "fast_paths.lean: errors (output in $OUT/fast_paths.out)"

if (( failures )); then
  echo "run_all: $failures check(s) failed ($nfix fixtures, outputs in $OUT)"
  exit 1
fi
echo "run_all: OK: $nfix fixtures, $nverdicts (input, spec) verdicts as expected, theorems match, counterexample, note and report-check texts as expected; $nself self-checked inputs, StoreReuse.pairs on $npairs"
