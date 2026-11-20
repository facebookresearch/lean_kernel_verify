/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
P0-1: global accesses in bounds (references/specs.md). A violation is certain in the
extracted model, whatever the data values:

* gload/gstore/gatomic: a masked-in lane outside the tensor's allocation (`inAlloc`: no
  element of the tensor when its strides do not overlap, else outside the allocation span);
* plain writes (gstore lanes, tma_store in-extent rows) outside the segment owned by the
  current work item; a work item that does not name the segment field owns no segment.
  TMA clips at the descriptor extents (zero fill on load, dropped on store), so a TMA box
  never leaves the descriptor and only its in-extent rows count;
* `overflow` events: an `addr` add/sub/mul/shl wrapped or an `addr` trunc lost bits;
* `metaOob` events: a metadata `load` outside the tensor's extents (its role stops there,
  Eval.lean).

Notes (no effect on the verdict): reads (gload lanes inside the allocation, tma_load
rows) outside the work item's segment ("read spill"), and gatomic/tma_reduce rows outside
it ("reduce spill"). They are harmless only if the values read are masked before use, or
the reduction leaves the values there unchanged (e.g. it adds zero); that is value-level
and not checked by P0.

P0-1 covers the inputs that set no elision flag (`Input.elisions`); on the others the same
violations and notes are P0-5's (MaskElision), except `metaOob` events, which are P0-1's on
every input: a scalar metadata read is not a masked access, and it leaves the other specs
undecided.
-/
import KV.Spec.Common

namespace KV.Bounds

def LanesInAlloc (t : TensorInfo) (offs : Array Int) : Prop := ∀ o ∈ offs, t.inAlloc o = true

def EventOk (h : Host) (e : TEvent) : Prop :=
  match e.ev with
  | .overflow .. => False
  | .gload t offs | .gatomic t offs => LanesInAlloc h.tensors[t]! offs
  | .gstore t offs =>
    LanesInAlloc h.tensors[t]! offs ∧ ∀ o ∈ offs, RowInSegment (h.segOf t e.work) (h.tensors[t]!.rowOf o)
  | .tmaStore t cs => ∀ r ∈ h.tensors[t]!.tmaRows cs, RowInSegment (h.segOf t e.work) r
  | _ => True

instance (h : Host) (e : TEvent) : Decidable (EventOk h e) := by
  unfold EventOk LanesInAlloc; split <;> infer_instance

/-- Every event satisfies `EventOk`: P0-1's condition, or P0-5's on an input that sets an
elision flag. -/
def EventsOk (r : Run) : Prop := ∀ c ∈ r.ctas, ∀ t ∈ c.roles, ∀ e ∈ t.events, EventOk r.host e

instance (r : Run) : Decidable (EventsOk r) := by unfold EventsOk; infer_instance

/-- No metadata load outside its tensor. -/
def NoMetaOob (r : Run) : Prop := ∀ c ∈ r.ctas, ∀ t ∈ c.roles, ∀ e ∈ t.events, e.ev.isMetaOob = false

instance (r : Run) : Decidable (NoMetaOob r) := by unfold NoMetaOob; infer_instance

/-- P0-1 on one run. -/
def Holds (r : Run) : Prop := (r.elided = false → EventsOk r) ∧ NoMetaOob r

instance (r : Run) : Decidable (Holds r) := by unfold Holds; infer_instance

/-- `decide (LanesInAlloc t offs)`, the tensor's span computed once (`lanesInAllocB_iff`). -/
def lanesInAllocB (t : TensorInfo) (offs : Array Int) : Bool :=
  let sp := t.span
  offs.all (t.inAllocWith sp)

theorem lanesInAllocB_iff (t : TensorInfo) (offs : Array Int) :
    lanesInAllocB t offs = true ↔ LanesInAlloc t offs := by
  simp only [lanesInAllocB, LanesInAlloc, Array.all_eq_true', TensorInfo.inAlloc_eq_with]

/-- `decide (EventOk h e)`, with the tensor's span and the segment of the event's work item
computed once per event instead of once per lane (`eventOkB_eq`). -/
def eventOkB (h : Host) (e : TEvent) : Bool :=
  match e.ev with
  | .overflow .. => false
  | .gload t offs | .gatomic t offs => lanesInAllocB h.tensors[t]! offs
  | .gstore t offs =>
    let ti := h.tensors[t]!
    let c := h.segOf t e.work
    lanesInAllocB ti offs && offs.all fun o => decide (RowInSegment c (ti.rowOf o))
  | .tmaStore t cs =>
    let c := h.segOf t e.work
    (h.tensors[t]!.tmaRows cs).all fun r => decide (RowInSegment c r)
  | _ => true

theorem eventOkB_iff (h : Host) (e : TEvent) : eventOkB h e = true ↔ EventOk h e := by
  cases hev : e.ev <;>
    simp only [eventOkB, EventOk, hev, lanesInAllocB_iff, Array.all_eq_true', decide_eq_true_eq,
      Bool.and_eq_true] <;> simp

theorem eventOkB_eq (h : Host) (e : TEvent) : eventOkB h e = decide (EventOk h e) := by
  rw [Bool.eq_iff_iff, decide_eq_true_iff]; exact eventOkB_iff h e

/-- `decide (EventsOk r)`, the CTAs checked in parallel, each event by `eventOkB`
(`eventsOkPar_eq`). -/
def eventsOkPar (r : Run) : Bool :=
  (parMap (fun c => c.roles.all fun t => t.events.all (eventOkB r.host)) r.ctas).all id

theorem eventsOkPar_eq (r : Run) : eventsOkPar r = decide (EventsOk r) := by
  rw [Bool.eq_iff_iff, decide_eq_true_iff]
  unfold eventsOkPar EventsOk
  simp only [parMap_eq, Array.all_map, Array.all_eq_true', Function.comp_def, id, eventOkB_iff]

/-- `decide (NoMetaOob r)`, the CTAs checked in parallel (`noMetaOobPar_eq`). -/
def noMetaOobPar (r : Run) : Bool :=
  (parMap (fun c => c.roles.all fun t => t.events.all fun e => !e.ev.isMetaOob) r.ctas).all id

theorem noMetaOobPar_eq (r : Run) : noMetaOobPar r = decide (NoMetaOob r) := by
  rw [Bool.eq_iff_iff, decide_eq_true_iff]
  unfold noMetaOobPar NoMetaOob
  simp only [parMap_eq, Array.all_map, Array.all_eq_true', Function.comp_def, id, Bool.not_eq_true']

/-- `decide (Holds r)`, its two parts computed per CTA in parallel (`holdsPar_eq`). -/
def holdsPar (r : Run) : Bool := (r.elided || eventsOkPar r) && noMetaOobPar r

theorem holdsPar_eq (r : Run) : holdsPar r = decide (Holds r) := by
  rw [holdsPar, eventsOkPar_eq, noMetaOobPar_eq, Bool.eq_iff_iff, decide_eq_true_iff]
  unfold Holds
  by_cases h : r.elided = true <;> simp [h]

def what (h : Host) (e : TEvent) (t : Nat) : String := s!"{e.ev.opName} {h.tensorName t}"

def boxWhat (h : Host) (e : TEvent) (t : Nat) (cs : Array Int) : String :=
  s!"{what h e t} box at {cs.toList} (box {h.tensors[t]!.decl.box})"

/-- Lanes of a pointer access outside the allocation, as messages. -/
def allocIssues (h : Host) (e : TEvent) (t : Nat) (offs : Array Int) : Array String :=
  let info := h.tensors[t]!
  let sp := info.span
  let outside := offs.filter (!info.inAllocWith sp ·)
  match sp with
  | none => if outside.isEmpty then #[] else #[s!"{what h e t}: {outside.size} lane(s) on an empty tensor"]
  | some (lo, hi) =>
    let beyond := outside.filter fun o => o < lo || o > hi
    let holes := outside.filter fun o => lo ≤ o && o ≤ hi
    (if beyond.isEmpty then #[]
     else #[s!"{what h e t}: {beyond.size} lane(s) outside allocation [{lo}, {hi}]: offsets {runs beyond}"]) ++
    (if holes.isEmpty then #[]
     else #[s!"{what h e t}: {holes.size} lane(s) inside [{lo}, {hi}] but at no element of shape {info.shape.toList} strides {info.strides.toList} (a coordinate outside its extent): offsets {runs holes}"])

/-- Why `EventOk h e` fails (empty iff it holds). -/
def eventIssues (h : Host) (e : TEvent) : Array String :=
  match e.ev with
  | .overflow what r w => #[s!"i{w} wrap in address arithmetic: {what} = {r}"]
  | .gload t offs | .gatomic t offs => allocIssues h e t offs
  | .gstore t offs =>
    let info := h.tensors[t]!
    let sp := info.span
    let rows := (offs.filter (info.inAllocWith sp)).map info.rowOf
    allocIssues h e t offs ++ ((segIssue (h.segOf t e.work) rows).map (s!"{what h e t}: " ++ ·)).toArray
  | .tmaStore t cs =>
    ((segIssue (h.segOf t e.work) (h.tensors[t]!.tmaRows cs)).map (s!"{boxWhat h e t cs}: " ++ ·)).toArray
  | _ => #[]

/-- Segment spills of reads and reductions (notes; they never fail P0-1). -/
def eventNotes (h : Host) (e : TEvent) : Array String :=
  let spill (pre : String) (t : Nat) (rows : Array Int) (read : Bool) :=
    let (kind, why) := if read then ("read spill", "harmless only if the values are masked before use (value-level, not checked by P0)")
      else ("reduce spill", "harmless only if the reduction leaves the values there unchanged (e.g. it adds zero)")
    ((segIssue (h.segOf t e.work) rows).map fun m => s!"{pre}: {kind}: {m}; {why}").toArray
  let laneRows (t : Nat) (offs : Array Int) :=
    let ti := h.tensors[t]!
    let sp := ti.span
    (offs.filter (ti.inAllocWith sp)).map ti.rowOf
  match e.ev with
  | .gload t offs => spill (what h e t) t (laneRows t offs) true
  | .gatomic t offs => spill (what h e t) t (laneRows t offs) false
  | .tmaLoad t cs _ _ => spill (boxWhat h e t cs) t (h.tensors[t]!.tmaRows cs) true
  | .tmaReduce t cs => spill (boxWhat h e t cs) t (h.tensors[t]!.tmaRows cs) false
  | _ => #[]

/-- Why `e.ev.isMetaOob = false` fails (empty iff it holds). -/
def oobIssues (h : Host) (e : TEvent) : Array String :=
  match e.ev with
  | .metaOob t idx shape =>
    #[s!"load {h.tensorName t}{idx.toList} outside the metadata tensor (shape {shape.toList}): an out-of-bounds global read; the role's later events depend on the value read, so its evaluation stops here and the other specs are not decided on this input"]
  | .syncOob t idx =>
    #[s!"sync access {h.tensorName t}[{idx}] outside the sync tensor (shape {h.tensors[t]!.shape.toList}): an out-of-bounds global access; the role's evaluation stops here and the other specs are not decided on this input"]
  | _ => #[]

/-- Why `EventsOk r` fails (empty iff it holds). -/
def eventsIssues (r : Run) : Array Issue := r.collect (eventIssues r.host)

/-- Segment spills of the run's reads and reductions. -/
def eventsNotes (r : Run) : Array Issue := r.collect (eventNotes r.host)

/-- Counterexamples of P0-1. -/
def issues (r : Run) : Array Issue :=
  (if r.elided then #[] else eventsIssues r) ++ r.collect (oobIssues r.host)

/-- Notes of P0-1. -/
def notes (r : Run) : Array Issue := if r.elided then #[] else eventsNotes r

end KV.Bounds
