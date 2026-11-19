# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Checks that `KV.StoreReuse.pairs`, the cursor search, finds exactly the pairs of the
definition its docstring states: every (store, write) pair with
`conflicts s x && !ordered C C.sound s x`, by store then write, with its `pairKind`, in
every cluster of every input of the given P1 KIRs. Run by tests/lean/run_all.sh on the
tests/lean/p1_2 fixtures.

    python3 -I tests/lean/p1_2/pairs_eq.py OUTDIR K1.kir.json [K2.kir.json ...]

Writes OUTDIR/<fixture>.pairs.lean (emit_lean.py's file with this check in place of the
theorems) and runs it with scripts/run_lean_check.sh. Exit status 0 iff every input of
every KIR compares equal.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

S = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(S / "frontend"))
import emit_lean  # noqa: E402

CHECK = """
namespace PairsEq
open KV

/-- `StoreReuse.pairs` by its definition: every pair tried, by store then write. -/
def slow (C : StoreReuse.Ctx) : Array (Nat × Nat × StoreReuse.PairKind) := Id.run do
  let mut out := #[]
  for (s, i) in C.stores.zipIdx do
    for (x, j) in C.writes.zipIdx do
      if StoreReuse.conflicts s x && !StoreReuse.ordered C C.sound s x then
        out := out.push (i, j, StoreReuse.pairKind C s x)
  return out

def check (k : Kernel) (i : Input) : String :=
  match eval k i with
  | .error _ => s!"PAIRS {i.name} EVAL-ERROR"
  | .ok r =>
    let Cs := StoreReuse.ctxs r r.ledgersP1
    let n := Cs.foldl (fun n C => n + (slow C).size) 0
    let same := Cs.all fun C => StoreReuse.pairs C == slow C
    s!"PAIRS {i.name} {if same then "EQUAL" else "DIFFER"} ({Cs.size} cluster(s), {n} pair(s))"

end PairsEq
"""


def lean_file(path: Path) -> tuple[str, list]:
    """emit_lean.py's Lean for the KIR at `path`, its theorems and evals replaced by one
    PairsEq.check per input; and the inputs."""
    text = emit_lean.emit(json.loads(path.read_text()), path.name)
    inputs = re.findall(r"^def (input\d+_\S*) : KV.Input", text, re.M)
    body = "\n".join(
        ln for ln in text.splitlines() if not ln.startswith(("theorem ", "#eval "))
    )
    end = body.rindex("\nend KIR.")
    evals = "".join(f"\n#eval PairsEq.check kernel {x}" for x in inputs)
    return body[:end] + "\n" + CHECK + evals + "\n" + body[end:], inputs


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    bad = 0
    for path in map(Path, sys.argv[2:]):
        name = path.name.removesuffix(".kir.json")
        text, inputs = lean_file(path)
        f = out / f"{name}.pairs.lean"
        f.write_text(text)
        r = subprocess.run([str(S / "scripts/run_lean_check.sh"), str(f)],
                           capture_output=True, text=True)  # fmt: skip
        got = re.findall(r"PAIRS (\S+) (EQUAL|DIFFER|EVAL-ERROR)", r.stdout)
        equal = [x for x, v in got if v == "EQUAL"]
        ok = r.returncode == 0 and len(got) == len(inputs) == len(equal)
        bad += not ok
        print(f"pairs_eq: {name}: {len(equal)}/{len(inputs)} input(s) equal"
              + ("" if ok else f" FAILED (lean exit {r.returncode}; see {f})"))  # fmt: skip
        if not ok:
            print(
                "\n".join(
                    "    " + ln for ln in (r.stdout + r.stderr).splitlines()[-20:]
                )
            )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
