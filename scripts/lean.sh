#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# lean.sh: find a Lean 4 toolchain and run it, without lake.
#
# The skill needs core Lean + Std only, both part of the toolchain: scripts/build_lean.sh
# runs `lean.sh --print-prefix` to find bin/lean, bin/leanc and lib/lean, and compiles
# lean/KV with them.
#
# Usage:
#   lean.sh File.lean [lean args...]   check a file that imports only core Lean and Std
#                                      (exit code = lean's)
#   lean.sh --print-prefix             print the toolchain prefix (passed to lean)
#   lean.sh --version                  print the Lean version (passed to lean)
#   lean.sh --env                      print the resolved toolchain and LEAN_PATH
#
# The toolchain (Lean 4.34.0) is, in this order:
#   LEAN_TOOLCHAIN_DIR   a toolchain directory (contains bin/lean and lib/lean)
#   facebook/lean_toolchain.sh, if present: prints the toolchain directory
#   the `lean` on PATH (e.g. installed with elan): its --print-prefix
set -euo pipefail

S="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die_missing() {
  {
    echo "lean.sh: $1"
    echo "lean.sh: install Lean 4.34.0 (e.g. with elan) and put lean on PATH,"
    echo "lean.sh: or set LEAN_TOOLCHAIN_DIR to a toolchain directory"
  } >&2
  exit 127
}

if [[ -n "${LEAN_TOOLCHAIN_DIR:-}" ]]; then
  T="$LEAN_TOOLCHAIN_DIR"
elif [[ -x "$S/facebook/lean_toolchain.sh" ]]; then
  T="$("$S/facebook/lean_toolchain.sh")"
elif command -v lean >/dev/null 2>&1; then
  T="$(lean --print-prefix)" || die_missing "lean --print-prefix failed"
else
  die_missing "no Lean toolchain found"
fi
[[ -x "$T/bin/lean" ]] || die_missing "no executable $T/bin/lean"
export LEAN_PATH="$T/lib/lean"

case "${1:-}" in
  "")
    echo "usage: lean.sh File.lean [lean args...] | --print-prefix | --version | --env" >&2
    exit 2
    ;;
  --env)
    echo "toolchain: $T"
    echo "LEAN_PATH=$LEAN_PATH"
    exit 0
    ;;
esac

exec "$T/bin/lean" "$@"
