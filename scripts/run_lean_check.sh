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
# LEAN_PATH and the native libKV.so loaded. Exit status is lean's: non-zero iff some
# theorem (one per spec and input) failed or the file has errors.
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
LEAN_PATH="$(get LEAN_PATH)" exec "$LEAN" --load-dynlib="$DYNLIB" "$@"
