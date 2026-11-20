# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Checks that `KV.StoreReuse.pairs`, the cursor search, finds exactly the pairs of the
definition its docstring states: every (store, write) pair with
`conflicts s x && !ordered C C.sound s x`, by store then write, with its `pairKind`, in
every cluster of every input of a P1 KIR. Run by tests/lean/run_all.sh on the
tests/lean/p1_2 fixtures.

    python3 -I tests/lean/p1_2/pairs_eq.py FIXTURE.lean

Adds the check to FIXTURE.lean, the file emit_lean.py wrote for a P1 KIR, before its
`end KIR.`: one `#eval PairsEq.check kernel <input>` per input, which prints
`SELFCHECK <input> OK StoreReuse.pairs ...` if every cluster compares equal, else
`SELFCHECK <input> DIFF StoreReuse.pairs ...`; run_all.sh counts these lines with the
self-checks of lean/KV/SelfCheck.lean in the same Lean output.
"""

import re
import sys
from pathlib import Path

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

def check (k : Kernel) (i : Input) : IO Unit :=
  IO.println <| match eval k i with
  | .error _ => s!"SELFCHECK {i.name} DIFF StoreReuse.pairs: evaluation error"
  | .ok r =>
    let Cs := StoreReuse.ctxs r r.ledgersP1
    let n := Cs.foldl (fun n C => n + (slow C).size) 0
    let same := Cs.all fun C => StoreReuse.pairs C == slow C
    s!"SELFCHECK {i.name} {if same then "OK" else "DIFF"} StoreReuse.pairs ({Cs.size} cluster(s), {n} pair(s))"

end PairsEq
"""


def main() -> int:
    path = Path(sys.argv[1])
    text = path.read_text()
    inputs = re.findall(r"^def (input\d+_\S*) : KV.Input", text, re.M)
    end = text.rindex("\nend KIR.")
    evals = "".join(f"\n#eval PairsEq.check kernel {x}" for x in inputs)
    path.write_text(text[:end] + "\n" + CHECK + evals + "\n" + text[end:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
