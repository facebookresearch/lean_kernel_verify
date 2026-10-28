/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
P0-2: role trip-count agreement (references/specs.md), per CTA and mbarrier slot, with C
the declared count, A the arrivals (arrive: n x threads_per_issue, or n x warps when
per_warp; commit_arrive: 1) and K = A / C the completions. A violation is certain in the
extracted model:

* a malformed barrier: C <= 0, or a `bar_init` of the slot with another count;
* a hang: a role whose waits need R > K completions (R: the `assumedPhase` of its last
  wait, from the phase classification of Ledger.lean: a stale wait may return at once, so
  it counts at its lower bound; a presumed skip counts at the phase it skips to, because if
  it does not skip it is stale), or a wait that never returns in the optimistic progress
  pass (`deadlocks`: every arrival as early as the roles' waits allow, so a hang or a
  cross-role wait cycle in the lockstep model);
* arrivals of the waited phases without lockstep contents: over the complete phases some
  role waits for (up to the highest phase a role's waits need, stale waits and presumed
  skips counted at the phase meant), every arriving role contributes the same share to each
  phase, reached exactly at one of its arrive events, and the shares sum to C; or an
  initialization arrival forms phase 1 by itself and the shares cover the phases after it
  (`SlotLedger.contents`). A role with arrivals takes no share only if they surely follow
  the waited phases. Otherwise a waited phase completes early (inside an arrive event or
  before some role's share arrived): the count does not match the arrivals per phase, the
  arriving roles disagree on their per-phase share, or a role's arrivals may land in a
  waited phase it has no share of;
* a missing wait on a data-ready slot: a role whose code waits on the barrier array has
  no wait, arrival or transaction event on a slot that completes and receives TMA bytes
  (a `bar_init` does not count). A role with an arrival or transaction bytes on the slot
  is one of its producers and is not expected to wait; this matters where one array holds
  several logical barriers, as the CuTe frontend's per-pipeline arrays of full and empty
  barriers do.

P0-2 is decided on the CTAs whose roles all ran to their end; an input with an
out-of-bounds metadata load (P0-1) leaves it undecided (Check.lean).

Notes (no effect on the verdict): phases that complete after a role's last wait (R < K,
e.g. CUTLASS producer_tail's trailing arrive_and_expect_tx, or an empty barrier's final
releases), a trailing partial phase nobody waits for (A not a multiple of C), and a role
whose code waits on the barrier array but has no event on a completing slot without TMA
bytes.
-/
import KV.Spec.Ledger

namespace KV.TripCount

def RoleOk (L : SlotLedger) (rs : RoleSlot) : Prop :=
  rs.needed ≤ L.completions ∧ ∀ w ∈ rs.waits, w.blocked = none

/-- Static waiter `r` has no event on the slot, which completes and receives TMA bytes. -/
def MissedDataWait (L : SlotLedger) (r : Nat) : Prop :=
  L.touches r = false ∧ 0 < L.completions ∧ 0 < L.tmaBytes

def SlotOk (L : SlotLedger) : Prop :=
  0 < L.count ∧ (∀ i ∈ L.inits, i.1 = L.count) ∧ (∀ rs ∈ L.roles, RoleOk L rs) ∧
  (0 < L.lockstep → L.contents.isSome) ∧ ∀ r ∈ L.staticWaiters, ¬ MissedDataWait L r

instance (L : SlotLedger) : Decidable (SlotOk L) := by
  unfold SlotOk RoleOk MissedDataWait; infer_instance

def HoldsWith (Ls : Ledgers) : Prop := ∀ c ∈ Ls, ∀ L ∈ c.2, SlotOk L

instance (Ls : Ledgers) : Decidable (HoldsWith Ls) := by unfold HoldsWith; infer_instance

/-- P0-2 on one run (its complete CTAs). -/
def Holds (r : Run) : Prop := HoldsWith r.ledgers

instance (r : Run) : Decidable (Holds r) := by unfold Holds; infer_instance

/-- "role R (consumer convention, n waits)". -/
def who (h : Host) (cid : Nat × Nat × Nat) (rs : RoleSlot) : String :=
  let conv := if rs.firstParity = 1 then "producer" else "consumer"
  s!"role {h.gRoleName cid rs.role} ({conv} convention, {rs.waits.size} waits)"

def lastLoc (L : SlotLedger) (rs : RoleSlot) : Loc := (rs.waits.back?.map (·.loc)).getD L.firstLoc

def completes (L : SlotLedger) : String :=
  s!"the slot completes {L.completions} times ({L.arrivals} arrivals / count {L.count})"

/-- Why wait `w` of `rs` needs `w.assumedPhase` rather than its lower bound: the role's first
presumed skip before it (empty if none). -/
def presumedWhy (rs : RoleSlot) (w : WaitRec) : String :=
  match (rs.waits.filter (·.pos ≤ w.pos)).find? (·.presumed) with
  | some p =>
    if w.assumedPhase > w.minPhase then
      s!" (counting the repeated parity at {p.loc} as the phase skip the lockstep contents imply; if it is no skip, that wait is stale)"
    else ""
  | none => ""

/-- "phase 3 completes" or "phases 3..8 complete". -/
def phases (lo hi : Int) : String := if lo = hi then s!"phase {lo} completes" else s!"phases {lo}..{hi} complete"

def missedWait (h : Host) (cid : Nat × Nat × Nat) (L : SlotLedger) (r : Nat) : String :=
  s!"role {h.gRoleName cid r} waits on {h.barriers[L.slot.bar]!.decl.name} in its code but has no wait, arrival or transaction event on {h.slotName L.slot} in this run (a bar_init does not count), while {completes L}"

def slotIssues (h : Host) (pid : Nat × Nat × Nat) (L : SlotLedger) : Array Issue := Id.run do
  let issue (loc : Loc) (msg : String) : Issue := { loc, context := h.slotCtx pid L.slot, msg }
  if L.count ≤ 0 then
    return #[issue L.firstLoc s!"declared arrival count {L.count} is not positive"]
  let mut out := #[]
  for (n, loc) in L.inits do
    if n != L.count then out := out.push (issue loc s!"bar_init count {n} != declared count {L.count}")
  for rs in L.roles do
    if rs.needed > L.completions then
      let why := (rs.waits.back?.map (presumedWhy rs)).getD ""
      out := out.push (issue (lastLoc L rs) s!"{who h pid rs} needs {rs.needed} completions but {completes L}: the wait hangs{why}")
    else if let some (w, got, need) := rs.waits.findSome? fun w => w.blocked.map fun (g, n) => (w, g, n) then
      out := out.push (issue w.loc s!"{who h pid rs}: this wait never returns: even with every arrival as early as the roles' waits allow, {got} arrival(s) reach the slot before it and its phase needs {need} (a hang or a cross-role wait cycle){presumedWhy rs w}")
  if 0 < L.lockstep && L.contents.isNone then
    let arrivers := (L.roles.filter (0 < ·.total)).toList.map fun rs =>
      s!"role {h.gRoleName pid rs.role}: {rs.total} arrivals, reaching {(rs.arrivals.toList.take 4).map (·.2)}{if rs.arrivals.size > 4 then " ..." else ""}"
    out := out.push (issue L.firstLoc
      s!"the arrivals of the {L.lockstep} waited phase(s) do not split into per-role shares that sum to the count {L.count} and end at arrive events ({"; ".intercalate arrivers}), and no initialization arrival forms phase 1 by itself: a waited phase completes early, so the count does not match the arrivals per phase, the arriving roles disagree on their per-phase share, or a role's arrivals may land in a waited phase it has no share of")
  for r in L.staticWaiters do
    if !L.touches r && 0 < L.completions && 0 < L.tmaBytes then
      out := out.push (issue L.firstLoc s!"{missedWait h pid L r} and receives {L.tmaBytes} TMA bytes (a data-ready barrier)")
  return out

def slotNotes (h : Host) (pid : Nat × Nat × Nat) (L : SlotLedger) : Array Issue := Id.run do
  let note (loc : Loc) (msg : String) : Issue := { loc, context := h.slotCtx pid L.slot, msg }
  if L.count ≤ 0 then return #[]
  let k := L.completions
  let mut out := #[]
  for rs in L.roles do
    if 0 < rs.waits.size && rs.required < k then
      out := out.push (note (lastLoc L rs)
        s!"{who h pid rs} waits for phases up to {rs.required} but {completes L}: {phases (rs.required + 1) k} after its last wait")
  if L.arrivals % L.count != 0 && L.waited ≤ k then
    out := out.push (note L.firstLoc
      s!"{L.arrivals} arrivals is not a multiple of the count {L.count}: phase {k + 1} gets {L.arrivals % L.count} arrival(s) and never completes; nobody waits for it")
  for r in L.staticWaiters do
    if !L.touches r && 0 < k && L.tmaBytes = 0 then
      out := out.push (note L.firstLoc (missedWait h pid L r))
  return out

/-- Counterexamples of P0-2 on the ledgers `Ls` of a run on host `h`. -/
def issues (h : Host) (Ls : Ledgers) : Array Issue := forSlots Ls (slotIssues h)

/-- Notes of P0-2 on the ledgers `Ls` of a run on host `h`. -/
def notes (h : Host) (Ls : Ledgers) : Array Issue := forSlots Ls (slotNotes h)

end KV.TripCount
