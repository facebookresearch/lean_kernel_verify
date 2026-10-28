#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# build_lean.sh: compile the Lean library lean/KV (core Lean + Std only) if anything changed,
# then print the environment to use it, one VAR=value per line on stdout:
#
#   LEAN=<toolchain>/bin/lean
#   LEAN_PATH=<toolchain>/lib/lean:<build dir>
#   KV_DYNLIB=<build dir>/libKV.so
#
# Usage: build_lean.sh      (build logs go to stderr)
#
# Modules are compiled in dependency order (from their `import KV...` lines) with
# `lean -o <olean> -c <c>`; a module is rebuilt when its source or an imported .olean is
# newer than its .olean. The generated C is compiled with the toolchain's leanc into
# libKV.so. Passing it as `lean --load-dynlib` (run_lean_check.sh does) makes #eval and
# native_decide run the evaluator as native code instead of in the IR interpreter
# (~20x faster). The toolchain is the one scripts/lean.sh selects.
#
# Env: LKV_LEAN_BUILD  build directory (default /tmp/lkv_lean_build_$USER; see below)
set -euo pipefail

S="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$S/lean"
PREFIX="$("$S/scripts/lean.sh" --print-prefix)"
if [[ -n "${LKV_LEAN_BUILD:-}" ]]; then
  OUT="$LKV_LEAN_BUILD"
else
  # One build directory per user, named after the login name. In a network-isolated buck
  # test sandbox the uid has no name: it is an unmapped user namespace where every user is
  # uid 65534, so a directory named after the uid would be shared by all users (`id -un`
  # fails there, though it still prints 65534). $USER still names the user there. With
  # neither, build in a new private directory (not reused).
  BUILD_USER="$(id -un 2>/dev/null)" || BUILD_USER="${USER:-}"
  if [[ -n "$BUILD_USER" ]]; then
    OUT="/tmp/lkv_lean_build_$BUILD_USER"
  else
    OUT="$(mktemp -d "${TMPDIR:-/tmp}/lkv_lean_build.XXXXXX")"
  fi
fi
LEAN="$PREFIX/bin/lean"
LEANC="$PREFIX/bin/leanc"
export LEAN_PATH="$PREFIX/lib/lean:$OUT"

mkdir -p "$OUT"
exec 9>"$OUT/.lock"
flock 9

declare -A DEPS PLACED
MODS=()
while IFS= read -r f; do
  m="${f#"$SRC"/}"
  m="${m%.lean}"
  m="${m//\//.}"
  MODS+=("$m")
  DEPS[$m]="$(sed -n 's/^import \(KV[A-Za-z0-9_.]*\).*/\1/p' "$f" | tr '\n' ' ')"
done < <(find "$SRC" -name '*.lean' | sort)

ORDER=()
while (( ${#ORDER[@]} < ${#MODS[@]} )); do
  progress=0
  for m in "${MODS[@]}"; do
    [[ -n "${PLACED[$m]:-}" ]] && continue
    ready=1
    for d in ${DEPS[$m]}; do
      [[ -n "${PLACED[$d]:-}" ]] || { ready=0; break; }
    done
    if (( ready )); then ORDER+=("$m"); PLACED[$m]=1; progress=1; fi
  done
  if (( !progress )); then
    echo "build_lean.sh: import cycle or missing KV module among: ${MODS[*]}" >&2
    exit 1
  fi
done

OBJS=()
for m in "${ORDER[@]}"; do
  rel="${m//.//}"
  src="$SRC/$rel.lean"
  olean="$OUT/$rel.olean"
  stale=0
  [[ -f "$olean" && "$olean" -nt "$src" ]] || stale=1
  for d in ${DEPS[$m]}; do
    [[ "$OUT/${d//.//}.olean" -nt "$olean" ]] && stale=1
  done
  if (( stale )); then
    echo "build_lean.sh: lean $m" >&2
    mkdir -p "$(dirname "$olean")"
    "$LEAN" -R "$SRC" -o "$olean" -i "$OUT/$rel.ilean" -c "$OUT/$rel.c" "$src" >&2
  fi
  if [[ ! -f "$OUT/$rel.o" || "$OUT/$rel.c" -nt "$OUT/$rel.o" ]]; then
    echo "build_lean.sh: leanc $m" >&2
    "$LEANC" -c -O2 -fPIC -DLEAN_EXPORTING -o "$OUT/$rel.o" "$OUT/$rel.c" >&2
  fi
  OBJS+=("$OUT/$rel.o")
done

LIB="$OUT/libKV.so"
relink=0
[[ -f "$LIB" ]] || relink=1
for o in "${OBJS[@]}"; do
  [[ "$o" -nt "$LIB" ]] && relink=1
done
if (( relink )); then
  echo "build_lean.sh: link $LIB" >&2
  "$LEANC" -shared -o "$LIB" "${OBJS[@]}" >&2
fi

echo "LEAN=$LEAN"
echo "LEAN_PATH=$LEAN_PATH"
echo "KV_DYNLIB=$LIB"
