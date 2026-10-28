/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
Shared vocabulary of the P0 specs: spec ids, counterexample and note lines, and the
footprint of an access (allocation, rows, TMA boxes, segment of the current work item).
-/
import KV.Eval

namespace KV

inductive SpecId where
  | bounds | tripCount | coverage | barrier | maskElision | tmemReuse

/-- The five P0 specs (every KIR). -/
def SpecId.all : List SpecId := [.bounds, .tripCount, .coverage, .barrier, .maskElision]

/-- The P1 specs (decided on a P1 KIR only, references/specs.md "P1-1"). -/
def SpecId.p1 : List SpecId := [.tmemReuse]

def SpecId.label : SpecId → String
  | .bounds => "P0-1" | .tripCount => "P0-2" | .coverage => "P0-3" | .barrier => "P0-4"
  | .maskElision => "P0-5" | .tmemReuse => "P1-1"

def SpecId.title : SpecId → String
  | .bounds => "global accesses in bounds"
  | .tripCount => "role trip-count agreement"
  | .coverage => "output coverage exactly once"
  | .barrier => "mbarrier protocol"
  | .maskElision => "mask-elision soundness"
  | .tmemReuse => "TMEM reuse ordered"

/-- One counterexample or note line: IR location, where (CTA/role/slot), what. -/
structure Issue where
  loc : Loc
  context : String
  msg : String

def Issue.render (i : Issue) : String :=
  s!"{if i.loc.isEmpty then "?" else i.loc}: {i.context}: {i.msg}"

/-! ## Names for messages -/

def Host.tensorName (h : Host) (t : Nat) : String := h.tensors[t]!.decl.name

def Host.roleName (h : Host) (r : Nat) : String := (h.kernel.roles[r]?.map (·.name)).getD s!"#{r}"

/-- Program id of the CTA of rank `rank` in cluster `cid` (`cid` without clusters). -/
def Host.pidOf (h : Host) (cid : Nat × Nat × Nat) (rank : Nat) : Nat × Nat × Nat :=
  let (cx, cy, cz) := h.cluster
  (cid.1 * cx + rank % cx, cid.2.1 * cy + (rank / cx) % cy, cid.2.2 * cz + rank / (cx * cy))

/-- Name of role `g` of a cluster ledger (Ledger.lean, `ClusterTrace`): the kernel role, and
without clusters nothing else; with clusters, the CTA it runs in. -/
def Host.gRoleName (h : Host) (cid : Nat × Nat × Nat) (g : Nat) : String :=
  let n := max 1 h.kernel.roles.length
  if h.csize ≤ 1 then h.roleName g else s!"{h.roleName (g % n)} of cta {pidStr (h.pidOf cid (g / n))}"

def Host.slotName (h : Host) (s : Slot) : String := s!"{h.barriers[s.bar]!.decl.name}[{s.idx}]"

/-- "cta (x,y,z) slot mbar_0[1]": the slot of a cluster ledger with the CTA that holds it. -/
def Host.slotCtx (h : Host) (cid : Nat × Nat × Nat) (s : Slot) : String :=
  s!"cta {pidStr (h.pidOf cid s.cta)} slot {h.slotName s}"

/-- KIR name of a memory event. -/
def Ev.opName : Ev → String
  | .gload .. => "gload" | .gstore .. => "gstore" | .gatomic .. => "gatomic"
  | .tmaLoad .. => "tma_load" | .tmaStore .. => "tma_store" | .tmaReduce .. => "tma_reduce"
  | _ => "event"

def Ev.isMetaOob : Ev → Bool
  | .metaOob .. => true
  | _ => false

def Host.roleCtx (h : Host) (pid : Nat × Nat × Nat) (role : Nat) : String :=
  s!"cta {pidStr pid} role {h.roleName role}"

/-- The distinct elements of `xs`, ascending. -/
def dedup (xs : Array Int) : Array Int :=
  (xs.qsort (· < ·)).foldl (init := #[]) fun acc x => if acc.back? == some x then acc else acc.push x

/-- Sorted integers as half-open runs, e.g. `[0, 4) [9, 10)`; at most `max` runs shown. -/
def runs (xs : Array Int) (max : Nat := 4) : String := Id.run do
  let xs := xs.qsort (· < ·)
  let mut out : Array (Int × Int) := #[]
  for x in xs do
    match out.back? with
    | some (lo, hi) =>
      if x == hi then out := out.pop.push (lo, hi + 1)
      else if x > hi then out := out.push (x, x + 1)
    | none => out := out.push (x, x + 1)
  let shown := (out.toList.take max).map fun (lo, hi) => s!"[{lo}, {hi})"
  return " ".intercalate shown ++ (if out.size > max then s!" ... ({out.size} runs)" else "")

/-! ## Footprints -/

def intRange (lo hi : Int) : Array Int := (Array.range (hi - lo).toNat).map fun (i : Nat) => lo + i

/-- The dim whose coordinates are "rows": the segment dim, else dim 0. -/
def TensorInfo.rowDim (t : TensorInfo) : Nat := (t.decl.segments.map (·.dim)).getD 0

/-- Element offset inside the tensor's allocation: an element (every coordinate inside its
extent) when the strides do not overlap, else inside the allocation span; false for an
empty tensor. -/
def TensorInfo.inAlloc (t : TensorInfo) (o : Int) : Bool :=
  match t.span with
  | some (lo, hi) => if t.exact then t.isElem o else lo ≤ o && o ≤ hi
  | none => false

/-- Row of an element offset of a global tensor: its coordinate along the row dim, after
removing the dims of larger stride greedily (largest first). Exact for segmented tensors,
whose strides are checked not to overlap (`setupHost`). -/
def TensorInfo.rowOf (t : TensorInfo) (o : Int) : Int :=
  if t.shape[t.rowDim]! ≤ 1 then 0 else t.rowOuter.foldl (· % ·) o / t.strides[t.rowDim]!

/-- All offsets `Σ c_i * strides[i]` over per-dim coordinate lists, row-major, each with
its row coordinate. -/
def TensorInfo.cartesian (t : TensorInfo) (coords : Array (Array Int)) : Array (Int × Int) :=
  coords.zipIdx.foldl (init := #[(0, 0)]) fun acc (cs, d) =>
    let s := t.strides[d]!
    acc.flatMap fun (off, row) => cs.map fun c => (off + c * s, if d = t.rowDim then c else row)

/-- Per-dim coordinates of a TMA box that lie inside the descriptor extents (TMA clips the
rest: zero-filled on load, dropped on store). -/
def TensorInfo.boxCoords (t : TensorInfo) (cs : Array Int) : Array (Array Int) :=
  (cs.zip (t.shape.zip t.decl.box.toArray)).map fun (c, n, b) => intRange (max 0 c) (min n (c + b))

/-- In-extent rows a TMA box touches (none if the box is clipped away entirely). -/
def TensorInfo.tmaRows (t : TensorInfo) (cs : Array Int) : Array Int :=
  let bc := t.boxCoords cs
  if bc.any (·.isEmpty) then #[] else bc[t.rowDim]!

/-- In-extent elements of a TMA box: (offset, row). -/
def TensorInfo.tmaElems (t : TensorInfo) (cs : Array Int) : Array (Int × Int) :=
  t.cartesian (t.boxCoords cs)

/-! ## Segments -/

/-- What the current work item says about the segment of an access to a tensor. -/
inductive SegCtx where
  | unsegmented
  | noWork (field : String)
  | badIndex (field : String) (z : Int) (n : Nat)
  | seg (field : String) (z lo hi : Int)

def Host.segOf (h : Host) (t : Nat) (work : Array (String × Int)) : SegCtx :=
  match h.tensors[t]!.decl.segments with
  | none => .unsegmented
  | some sg =>
    match ((work.find? (·.1 == sg.per)).map (·.2) : Option Int) with
    | none => .noWork sg.per
    | some z =>
      let offs := (h.tensorIdx[sg.offsets]?.map (h.tensors[·]!.contents)).getD #[]
      if 0 ≤ z && z + 1 < (offs.size : Int) then .seg sg.per z offs[z.toNat]! offs[z.toNat + 1]!
      else .badIndex sg.per z (offs.size - 1)

/-- Row `r` lies in the segment of the current work item (vacuous when unsegmented). -/
def RowInSegment (c : SegCtx) (r : Int) : Prop :=
  match c with
  | .unsegmented => True
  | .seg _ _ lo hi => lo ≤ r ∧ r < hi
  | _ => False

instance (c : SegCtx) (r : Int) : Decidable (RowInSegment c r) := by
  unfold RowInSegment; split <;> infer_instance

def SegCtx.describe : SegCtx → String
  | .unsegmented => "unsegmented"
  | .noWork f => s!"no current work item defines '{f}'"
  | .badIndex f z n => s!"work {f}={z} names no segment ({n} segments)"
  | .seg f z lo hi => s!"segment {f}={z} is rows [{lo}, {hi})"

/-- Rows outside the segment of the current work item, as a message (none if all inside). -/
def segIssue (c : SegCtx) (rows : Array Int) : Option String :=
  let bad := rows.filter fun r => !decide (RowInSegment c r)
  if bad.isEmpty then none
  else match c with
    | .seg .. => some s!"rows {runs bad} outside {c.describe}"
    | _ => some c.describe

/-- The input sets an elision flag: P0-1 and P0-3 violations of the run are P0-5's. -/
def Run.elided (r : Run) : Bool := !r.host.input.elisions.isEmpty

/-- Every event of every role of every CTA, with its CTA and role. -/
def Run.forEvents {α : Type} (r : Run) (init : α) (f : α → Nat × Nat × Nat → Nat → TEvent → α) : α :=
  r.ctas.foldl (init := init) fun acc c =>
    c.roles.foldl (init := acc) fun acc t => t.events.foldl (init := acc) fun acc e => f acc c.pid t.role e

/-- `f`'s messages about every event, as lines located at the event. -/
def Run.collect (r : Run) (f : TEvent → Array String) : Array Issue :=
  r.forEvents #[] fun acc pid role e =>
    acc ++ (f e).map fun m => { loc := e.loc, context := r.host.roleCtx pid role, msg := m }

end KV
