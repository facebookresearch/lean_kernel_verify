/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
P0-3: output coverage exactly once (references/specs.md). A violation is certain in the
extracted model:

* an output tensor that the input's `intended_writes` does not name (`unchecked` is the
  explicit opt-out);
* for every output tensor with an intended region: an intended element that no plain
  store (gstore lane, in-extent tma_store element) writes, or that plain stores write more
  than once;
* a plain store to an element outside the intended region.

Notes (no effect on the verdict): gatomic/tma_reduce elements outside the intended region
(harmless only if the reduced values are zero there) and output tensors opted out with
`unchecked`. Plain writes outside the segment of their work item are P0-1.

P0-3 covers the inputs that set no elision flag (`Input.elisions`); on the others the same
violations and notes are P0-5's (MaskElision).

Elements are identified by their element offset (sum of coordinate x stride), so two
writes collide exactly when they hit the same memory.

On an input with an out-of-bounds metadata load (P0-1) some role traces stop early; P0-3
is then undecided (Check.lean) and only its certain counterexamples are listed (no holes:
a stopped role might still write them).
-/
import Std.Data.HashSet
import KV.Spec.Common

namespace KV.Coverage

/-- The intended region of one output tensor: (offset, row) of every element, and the set
of offsets. -/
structure Target where
  t : Nat
  elems : Array (Int × Int)
  set : Std.HashSet Int
  deriving Inhabited

/-- Rows of the intended region (`none` for the `unchecked` opt-out). -/
def intendedRows (h : Host) (info : TensorInfo) : Intended → Option (Array Int)
  | .all => some (intRange 0 info.shape[info.rowDim]!)
  | .all_rows_of_segments =>
    let offs := (info.decl.segments.bind fun sg => h.tensorIdx[sg.offsets]?.map (h.tensors[·]!.contents)).getD #[]
    some ((offs.zip (offs.extract 1)).flatMap fun (lo, hi) => intRange lo hi)
  | .rows rs => some (rs.toArray.flatMap fun (lo, hi) => intRange lo hi)
  | .unchecked => none

/-- Targets of every output tensor with an intended region. -/
def targets (h : Host) : Array Target :=
  h.input.intended_writes.toArray.filterMap fun (name, iw) => do
    let t ← h.tensorIdx[name]?
    let info := h.tensors[t]!
    let rows := dedup (← intendedRows h info iw)
    let coords := (Array.range info.rank).map fun d =>
      if d = info.rowDim then rows else intRange 0 info.shape[d]!
    let elems := info.cartesian coords
    pure { t, elems, set := Std.HashSet.ofArray (elems.map (·.1)) }

/-- The input names output tensor `t` in `intended_writes` (with a region or `unchecked`). -/
def HasIntent (h : Host) (t : TensorInfo) : Prop :=
  t.decl.role = .output → h.input.intended_writes.any (·.1 == t.decl.name) = true

/-- Elements one event writes: (tensor, plain store?, (offset, row) pairs). -/
def writes (h : Host) (e : TEvent) : Option (Nat × Bool × Array (Int × Int)) :=
  let lanes (t : Nat) (offs : Array Int) := offs.map fun o => (o, h.tensors[t]!.rowOf o)
  match e.ev with
  | .gstore t offs => some (t, true, lanes t offs)
  | .tmaStore t cs => some (t, true, h.tensors[t]!.tmaElems cs)
  | .gatomic t offs => some (t, false, lanes t offs)
  | .tmaReduce t cs => some (t, false, h.tensors[t]!.tmaElems cs)
  | _ => none

/-- A plain store writes only intended elements of its tensor (if that has a target). -/
def WriteOk (h : Host) (tgs : Array Target) (e : TEvent) : Prop :=
  match writes h e with
  | some (t, true, xs) =>
    match tgs.find? (·.t == t) with
    | some g => ∀ x ∈ xs, x.1 ∈ g.set
    | none => True
  | _ => True

instance (h : Host) (tgs : Array Target) (e : TEvent) : Decidable (WriteOk h tgs e) := by
  unfold WriteOk
  split
  · split <;> infer_instance
  · infer_instance

/-- Plain-store count of one intended element. -/
structure Count where
  plain : Nat
  /-- The first two writers, for messages. -/
  writers : Array String

structure Ledger where
  target : Target
  counts : Std.HashMap Int Count

def Ledger.plain (L : Ledger) (x : Int) : Nat := (L.counts[x]?.map (·.plain)).getD 0

/-- Plain-store counts of the intended elements of every target, over all CTAs and roles. -/
def ledgers (r : Run) (tgs : Array Target) : Array Ledger :=
  let counts := r.forEvents (tgs.map fun _ => ({} : Std.HashMap Int Count)) fun acc pid role e =>
    match writes r.host e with
    | some (t, true, xs) =>
      match tgs.findIdx? (·.t == t) with
      | some i =>
        let g := tgs[i]!
        let who := s!"{if e.loc.isEmpty then "?" else e.loc} ({r.host.roleCtx pid role})"
        acc.modify i fun m => xs.foldl (init := m) fun m (o, _) =>
          if !g.set.contains o then m
          else
            let c := m.getD o { plain := 0, writers := #[] }
            m.insert o { plain := c.plain + 1, writers := if c.writers.size < 2 then c.writers.push who else c.writers }
      | none => acc
    | _ => acc
  tgs.zipWith (fun target counts => { target, counts }) counts

def OkWith (tgs : Array Target) (Ls : Array Ledger) (r : Run) : Prop :=
  (∀ t ∈ r.host.tensors, HasIntent r.host t) ∧
  (∀ L ∈ Ls, ∀ x ∈ L.target.elems, L.plain x.1 = 1) ∧
  ∀ c ∈ r.ctas, ∀ t ∈ c.roles, ∀ e ∈ t.events, WriteOk r.host tgs e

instance (tgs : Array Target) (Ls : Array Ledger) (r : Run) : Decidable (OkWith tgs Ls r) := by
  unfold OkWith HasIntent; infer_instance

/-- The coverage conditions on one run: P0-3's, or P0-5's on an input that sets an elision
flag. -/
def Ok (r : Run) : Prop :=
  let tgs := targets r.host
  OkWith tgs (ledgers r tgs) r

/-- Decides `Ok` computing the targets once. -/
def decideWith (tgs : Array Target) (r : Run) : Decidable (OkWith tgs (ledgers r tgs) r) := inferInstance

instance (r : Run) : Decidable (Ok r) := decideWith (targets r.host) r

/-- P0-3 on one run. -/
def Holds (r : Run) : Prop := r.elided = false → Ok r

instance (r : Run) : Decidable (Holds r) := by unfold Holds; infer_instance

/-! ## Counterexamples and notes -/

/-- Output tensors the input's `intended_writes` does not name. -/
def intentIssues (h : Host) : Array Issue :=
  (h.tensors.filter fun t => t.decl.role = .output && !h.input.intended_writes.any (·.1 == t.decl.name)).map fun t =>
    { loc := "", context := s!"tensor {t.decl.name}",
      msg := "output tensor without an intended_writes entry in this input: give its intended region, or \"unchecked\" to opt out of the coverage check" }

/-- Intended elements no plain store writes. -/
def holeIssues (h : Host) (Ls : Array Ledger) : Array Issue :=
  Ls.filterMap fun L =>
    let miss := L.target.elems.filter fun (o, _) => L.plain o == 0
    if miss.isEmpty then none
    else some { loc := "", context := s!"tensor {h.tensorName L.target.t}",
                msg := s!"{miss.size} intended element(s) never written by a plain store: rows {runs (dedup (miss.map (·.2)))}" }

/-- Intended elements written more than once. -/
def dupIssues (h : Host) (Ls : Array Ledger) : Array Issue :=
  Ls.filterMap fun L =>
    let dups := L.target.elems.filter fun (o, _) => L.plain o ≥ 2
    match dups[0]? with
    | none => none
    | some (o, row) =>
      let c := L.counts.getD o { plain := 0, writers := #[] }
      some { loc := "", context := s!"tensor {h.tensorName L.target.t}",
             msg := s!"{dups.size} element(s) written more than once by plain stores (rows {runs (dedup (dups.map (·.2)))}); e.g. offset {o} (row {row}) {c.plain} times, by {" and ".intercalate c.writers.toList}" }

/-- Elements of the write event `e` outside the intended region of its tensor: plain stores
(`plain = true`, violations) or reductions (`plain = false`, notes). -/
def outsideWrites (h : Host) (tgs : Array Target) (plain : Bool) (e : TEvent) : Array String :=
  match writes h e with
  | some (t, p, xs) =>
    match tgs.find? (·.t == t) with
    | some g =>
      let outside := xs.filter fun (o, _) => !g.set.contains o
      if p != plain || outside.isEmpty then #[]
      else
        let msg := s!"{e.ev.opName} {h.tensorName t}: {outside.size} element(s) outside the intended region (rows {runs (dedup (outside.map (·.2)))})"
        #[if plain then msg else msg ++ "; harmless only if the reduction leaves the values there unchanged (e.g. it adds zero)"]
    | none => #[]
  | none => #[]

/-- Why `Ok r` fails (holes only if every role trace is complete). -/
def okIssues (r : Run) : Array Issue :=
  let tgs := targets r.host
  let Ls := ledgers r tgs
  intentIssues r.host ++ (if r.complete then holeIssues r.host Ls else #[]) ++ dupIssues r.host Ls ++
    r.collect (outsideWrites r.host tgs true)

/-- Tensors opted out with `unchecked`, and reductions that leave the intended region. -/
def okNotes (r : Run) : Array Issue :=
  let unchecked := r.host.input.intended_writes.toArray.filterMap fun (name, iw) =>
    if let .unchecked := iw then
      some { loc := "", context := s!"tensor {name}", msg := "intended_writes is \"unchecked\": its coverage is not checked" }
    else none
  unchecked ++ r.collect (outsideWrites r.host (targets r.host) false)

/-- Counterexamples of P0-3. -/
def issues (r : Run) : Array Issue := if r.elided then #[] else okIssues r

/-- Notes of P0-3. -/
def notes (r : Run) : Array Issue := if r.elided then #[] else okNotes r

end KV.Coverage
