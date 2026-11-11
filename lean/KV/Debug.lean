/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
Debug only, for frontend parity tools: the global-memory footprint of one evaluation. No
theorem, verdict or report uses it, and KV.lean does not import it (scripts/build_lean.sh
still compiles it into libKV.so, as every module under lean/).

`footprint k i` evaluates kernel `k` on input `i` (`KV.eval`) and prints one line per
(tensor, access kind) that touches at least one element:

  FOOTPRINT <input> <tensor> <kind> <distinct element offsets> <min row> <max row>

`FOOTPRINT_NONE <input>` if no event touches global memory (so an input without accesses
is not mistaken for a missing one), or, if the evaluation stops,
`FOOTPRINT_ERROR <input> <ErrKind.label>`. Kinds:
`read` (gload), `write` (gstore, tma_store), `atomic` (gatomic, tma_reduce), `tma_load`.
A pointer access counts its masked-in lanes; a TMA access the in-extent elements of its
box (`TensorInfo.tmaElems`, as P0-1 and P0-3 read it). Rows are `TensorInfo.rowOf` (TMA:
the box's row coordinates). A role trace that stops at an out-of-bounds metadata load
counts its events up to the load. Two frontends that give the same verdicts but different
lines lost or moved an access that the specs did not catch.
-/
import Std.Data.HashSet
import KV.Spec.Common

namespace KV.Debug

/-- Access kinds, in output order. -/
def kindNames : Array String := #["read", "write", "atomic", "tma_load"]

/-- Tensor, access kind (index into `kindNames`) and `(offset, row)` elements of a global
memory event; `none` for the other events. -/
def access (h : Host) : Ev → Option (Nat × Nat × Array (Int × Int))
  | .gload t offs => some (t, 0, offs.map fun o => (o, h.tensors[t]!.rowOf o))
  | .gstore t offs => some (t, 1, offs.map fun o => (o, h.tensors[t]!.rowOf o))
  | .gatomic t offs => some (t, 2, offs.map fun o => (o, h.tensors[t]!.rowOf o))
  | .tmaStore t cs => some (t, 1, h.tensors[t]!.tmaElems cs)
  | .tmaReduce t cs => some (t, 2, h.tensors[t]!.tmaElems cs)
  | .tmaLoad t cs _ _ => some (t, 3, h.tensors[t]!.tmaElems cs)
  | _ => none

/-- Distinct element offsets and row range of one (tensor, kind). A set, not a list: a
kernel touches the same element from many CTAs and iterations. -/
structure Foot where
  offs : Std.HashSet Int
  rows : Option (Int × Int)

def Foot.empty : Foot := { offs := {}, rows := none }

instance : Inhabited Foot := ⟨Foot.empty⟩

def Foot.add (f : Foot) (e : Int × Int) : Foot :=
  let offs := f.offs.insert e.1
  match f.rows with
  | some (lo, hi) => { offs, rows := some (min lo e.2, max hi e.2) }
  | none => { offs, rows := some (e.2, e.2) }

/-- The lines of `k` on `i`, by tensor (KIR order), then kind. -/
def lines (k : Kernel) (i : Input) : Array String :=
  match eval k i with
  | .error e => #[s!"FOOTPRINT_ERROR {i.name} {e.kind.label}"]
  | .ok r => Id.run do
    let mut acc : Std.HashMap (Nat × Nat) Foot := {}
    for c in r.ctas do
      for t in c.roles do
        for e in t.events do
          if let some (ti, ki, elems) := access r.host e.ev then
            acc := acc.insert (ti, ki) (elems.foldl Foot.add (acc.getD (ti, ki) Foot.empty))
    let sorted := acc.toArray.qsort fun (a, _) (b, _) => a.1 < b.1 || (a.1 == b.1 && a.2 < b.2)
    let out := sorted.filterMap fun ((ti, ki), f) => f.rows.map fun (lo, hi) =>
      s!"FOOTPRINT {i.name} {r.host.tensorName ti} {kindNames[ki]!} {f.offs.size} {lo} {hi}"
    return if out.isEmpty then #[s!"FOOTPRINT_NONE {i.name}"] else out

/-- `#eval KV.Debug.footprint kernel input0_x` prints `lines` as plain text (an `#eval` of
a `String` would print it quoted). -/
def footprint (k : Kernel) (i : Input) : IO Unit :=
  (lines k i).forM IO.println

end KV.Debug
