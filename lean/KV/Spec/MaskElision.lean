/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
P0-5: mask-elision soundness (references/specs.md). The frontends compile each
host-selected flag set separately, so a mask-elided fast path (e.g. FULL_TILES: no masks,
no tail handling) is a variant of its own, run on the inputs whose flags select it. The
case names those flags (`elision_flags`); an input that sets one (`Input.elisions`) must
satisfy on its whole run what P0-1 and P0-3 require:

* every event satisfies `Bounds.EventOk` (accesses inside the allocation, plain writes
  inside their work item's segment, no wrap on an address path);
* the coverage conditions `Coverage.Ok` (every output tensor named in `intended_writes`,
  every intended element written exactly once by plain stores, no plain store outside the
  intended region).

P0-1, P0-3 and P0-5 partition these violations by input: on an input that sets an elision
flag they and the notes of P0-1 and P0-3 are reported here only, so a FAIL says that the
host selected the elided variant for an input it does not handle. A metadata load outside
its tensor stays P0-1's on every input (Bounds.lean). Inputs that set no elision flag
satisfy P0-5 vacuously.
-/
import KV.Spec.Bounds
import KV.Spec.Coverage

namespace KV.MaskElision

/-- P0-5 on one run. -/
def Holds (r : Run) : Prop := r.elided = true → Bounds.EventsOk r ∧ Coverage.Ok r

instance (r : Run) : Decidable (Holds r) := by unfold Holds; infer_instance

/-- Counterexamples of P0-5. -/
def issues (r : Run) : Array Issue := if r.elided then Bounds.eventsIssues r ++ Coverage.okIssues r else #[]

/-- Notes of P0-5. -/
def notes (r : Run) : Array Issue := if r.elided then Bounds.eventsNotes r ++ Coverage.okNotes r else #[]

end KV.MaskElision
