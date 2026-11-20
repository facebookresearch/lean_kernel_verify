#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# run_lean_check.sh: check a Lean file that imports KV (e.g. one written by
# frontend/emit_lean.py) against the compiled lean/KV library.
#
# Usage: run_lean_check.sh FILE.lean [lean args...]
#
# Builds lean/KV first if needed (scripts/build_lean.sh), then runs lean with that
# LEAN_PATH and the native libKV.so loaded. Exit status is lean's: non-zero iff the file has
# errors (#lkv_check proves each spec or its negation; in a file emit_lean.py --legacy-theorems
# wrote, a refuted theorem, one per spec and input, is an error too). The evaluator computes
# independent parts (CTAs, cluster ledgers, P0-1's per-CTA checks) in parallel tasks on
# lean's thread pool: LKV_LEAN_THREADS threads (default min(8, CPUs); scripts/lkv.py sets it
# to its CPUs divided by the Lean files it runs at a time), unless the args give -j. Without
# -j, lean would start one worker thread per CPU in every process.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: run_lean_check.sh FILE.lean [lean args...]" >&2
  exit 2
fi

S="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_OUT="$("$S/scripts/build_lean.sh")"
get() { sed -n "s/^$1=//p" <<<"$ENV_OUT"; }
LEAN="$(get LEAN)"
DYNLIB="$(get KV_DYNLIB)"
CPUS="$(nproc 2>/dev/null || echo 1)"
THREADS=(-j "${LKV_LEAN_THREADS:-$((CPUS < 8 ? CPUS : 8))}")
for a in "$@"; do
  case "$a" in -j*|--threads*) THREADS=() ;; esac
done
# -DmaxErrors=0: every refuted theorem of a --legacy-theorems file is reported (Lean stops after
# 100 errors by default; a KIR with many inputs can refute more); #lkv_check refutes none
LEAN_PATH="$(get LEAN_PATH)" exec "$LEAN" -DmaxErrors=0 --load-dynlib="$DYNLIB" "${THREADS[@]}" "$@"
