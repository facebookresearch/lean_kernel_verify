/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
P0-4: mbarrier protocol (references/specs.md), per CTA and mbarrier slot. A violation is
certain in the extracted model:

* a wait parity other than 0 or 1;
* a stale wait: a repeated parity that the phase classification (Ledger.lean) justifies
  neither as a re-check nor as a skip, not even as a skip presumed under the lockstep
  credit, so it can return at once or wait for a later phase depending on timing (a role that changes parity on every wait alternates p0, 1-p0, ...
  from its start convention, p0 = 0 consumer, 1 producer);
* on a slot that receives TMA bytes, a role whose first wait synchronizes with phase 0
  (parity 1 on the fresh barrier): it passes before any byte lands;
* in a phase some role waits for (at most the highest phase a role's waits need):
  expect_tx bytes != tma_load bytes landing in it;
* TMA bytes landing in a phase nobody waits for, or on a slot nobody waits on: no wait
  ever synchronizes with that load (phase K + 1 is the last phase, so every such load is
  in a phase up to K + 1);
* a tma_load not covered by earlier expect_tx bytes of its role in its phase (or by no
  expect_tx of its role at all), or transaction bytes whose phase is unknown (Ledger.lean,
  lockstep phase contents).

The transaction clauses (the last three, `SlotLedger.TxOk`) fail only if they fail under
every share assignment that fits the lockstep contents (Ledger.lean; the counterexamples
show the attribution under the representative one).

Notes (no effect on the verdict): laps (a wait after the role's first one on the slot that
synchronizes with phase k+2 or later instead of the phase k after its previous wait,
because phase k+1 surely completed before it: intended if the role skips phases on
purpose, a wrong phase if not); expect_tx bytes with no load in a phase nobody waits for
(transaction bytes pending after the last wait, e.g. CUTLASS producer_tail's trailing
arrive_and_expect_tx), and expect_tx bytes on a slot nobody waits on.

P0-4 is decided on the CTAs whose roles all ran to their end; an input with an
out-of-bounds metadata load (P0-1) leaves it undecided (Check.lean).
-/
import KV.Spec.Ledger

namespace KV.Barrier

def WaitOk (w : WaitRec) : Prop := (w.parity = 0 ∨ w.parity = 1) ∧ w.stale = none

/-- Phase the role's first wait on the slot synchronizes with. -/
def firstPhase (rs : RoleSlot) : Option Int := rs.waits[0]?.map (·.phase)

def RoleOk (L : SlotLedger) (rs : RoleSlot) : Prop :=
  (∀ w ∈ rs.waits, WaitOk w) ∧ (0 < L.tmaBytes → firstPhase rs ≠ some 0)

/-- The waits are well formed and the transaction bytes, attributed under the first fitting
share assignment that explains them (Ledger.lean), satisfy the transaction clauses. -/
def SlotOk (L : SlotLedger) : Prop := (∀ rs ∈ L.roles, RoleOk L rs) ∧ L.TxOk

instance (L : SlotLedger) : Decidable (SlotOk L) := by unfold SlotOk RoleOk WaitOk; infer_instance

def HoldsWith (Ls : Ledgers) : Prop := ∀ c ∈ Ls, ∀ L ∈ c.2, SlotOk L

instance (Ls : Ledgers) : Decidable (HoldsWith Ls) := by unfold HoldsWith; infer_instance

/-- P0-4 on one run (its complete CTAs). -/
def Holds (r : Run) : Prop := HoldsWith r.ledgers

instance (r : Run) : Decidable (Holds r) := by unfold Holds; infer_instance

def ctx (h : Host) (pid : Nat × Nat × Nat) (L : SlotLedger) (role : Option Nat) : String :=
  h.slotCtx pid L.slot ++ ((role.map fun r => s!" role {h.gRoleName pid r}").getD "")

def bytesMsg (p : TxPhase) : String := s!"phase {p.phase}: expect_tx {p.expect} bytes != tma_load {p.landed} bytes"

def uncoveredMsg (u : Uncovered) : String :=
  match u.phase with
  | some p => s!"tma_load not covered by an earlier expect_tx of this role in phase {p} ({u.bytes} bytes uncovered)"
  | none => "tma_load not covered by an earlier expect_tx of this role (it issued none on the slot before it)"

/-- How a phase surely completed before a wait. -/
def evidenceMsg (h : Host) (cid : Nat × Nat × Nat) (e : Evidence) : String :=
  if e.wait then s!"role {h.gRoleName cid e.role}'s wait at {e.loc} needs it and surely returned before"
  else s!"the arrivals that surely precede it reach the phase at role {h.gRoleName cid e.role}'s arrive at {e.loc}"

def slotIssues (h : Host) (pid : Nat × Nat × Nat) (L : SlotLedger) : Array Issue := Id.run do
  let mut out := #[]
  for rs in L.roles do
    let ps := (rs.waits.map (·.parity)).toList.take 8
    let conv := if rs.firstParity = 1 then "producer" else "consumer"
    for w in rs.waits do
      if w.parity != 0 && w.parity != 1 then
        out := out.push { loc := w.loc, context := ctx h pid L rs.role, msg := s!"wait parity {w.parity} (only 0 and 1 exist)" }
    if let some (i, w, before, need) := (rs.waits.zipIdx.findSome? fun (w, i) => w.stale.map fun (b, n) => (i, w, b, n)) then
      out := out.push { loc := w.loc, context := ctx h pid L rs.role,
                        msg := s!"wait #{i} has parity {w.parity}, expected {1 - w.parity} ({conv} convention, sequence {ps}): it repeats the previous parity, but {before} arrival(s) may precede it and {need} complete the next phase, which is not sure to complete first either, so it can return at once or wait for a later phase" }
    if 0 < L.tmaBytes && firstPhase rs = some 0 then
      if let some w := rs.waits[0]? then
        out := out.push { loc := w.loc, context := ctx h pid L rs.role,
                          msg := s!"first wait has parity 1 on a slot that receives TMA bytes: it passes on the fresh barrier, before any byte lands (a consumer of the data starts at parity 0)" }
  for p in L.txPhases do
    if p.phase ≤ L.waited && p.expect != p.landed then
      out := out.push { loc := p.loc, context := ctx h pid L none, msg := s!"{bytesMsg p} landing in it" }
    if L.waited < p.phase && p.landed != 0 then
      out := out.push { loc := p.loc, context := ctx h pid L none,
                        msg := s!"phase {p.phase}: {p.landed} TMA bytes land in it, but the waits observe phases up to {L.waited}: no wait synchronizes with the load (its data is read unsynchronized or never, and it may still write shared memory after the CTA exits; e.g. an arrive before its expect_tx or an extra completion shifts the waits against the data)" }
  if L.waited ≤ 0 && L.tmaBytes != 0 then
    out := out.push { loc := L.firstLoc, context := ctx h pid L none,
                      msg := s!"{L.tmaBytes} TMA bytes land on the slot, but nobody waits on it: no wait synchronizes with the loads" }
  for u in L.uncovered do
    out := out.push { loc := u.loc, context := ctx h pid L u.role, msg := uncoveredMsg u }
  if let some (r, loc) := L.unattributed[0]? then
    let lander := L.roles.any fun rs => rs.role == r && rs.lander
    let why := if lander then "the role lands transaction bytes on the slot without arriving or expecting bytes there (a TMA load or copy from another CTA of the cluster), and its bytes do not split into one fixed share per waited phase"
      else if L.contents.isNone then "the arriving roles have no lockstep contents (P0-2)"
      else "the role does not arrive on the slot in every waited phase (an expect_tx and the arrive of its phase must be in the same role)"
    out := out.push { loc, context := ctx h pid L (some r),
                      msg := s!"{L.unattributed.size} transaction event(s) cannot be attributed to a phase: {why}" }
  return out

def slotNotes (h : Host) (pid : Nat × Nat × Nat) (L : SlotLedger) : Array Issue := Id.run do
  let mut out := #[]
  for rs in L.roles do
    let conv := if rs.firstParity = 1 then "producer" else "consumer"
    for (w, i) in rs.waits.zipIdx do
      if let some (k, e) := w.lapped then
        if 0 < i then
          out := out.push { loc := w.loc, context := ctx h pid L rs.role,
                            msg := s!"wait #{i} (parity {w.parity}, {conv} convention) synchronizes with phase {w.minPhase}, not phase {k} after its previous wait: phase {k + 1} surely completed before it ({evidenceMsg h pid e}); intended if the role skips phases (e.g. it waits once per tile on a slot that completes an odd number of phases per tile), else it observes the wrong phase" }
  if L.waited ≤ 0 && L.tmaBytes = 0 && L.expectBytes != 0 then
    out := out.push { loc := L.firstLoc, context := ctx h pid L none,
                      msg := s!"expect_tx {L.expectBytes} bytes != tma_load 0 bytes, and nobody waits on the slot" }
  for p in L.txPhases do
    if L.waited < p.phase && p.landed = 0 && p.expect != 0 then
      out := out.push { loc := p.loc, context := ctx h pid L none,
                        msg := s!"{bytesMsg p}, and nobody waits for this phase (transaction bytes pending after the last wait)" }
  return out

/-- Counterexamples of P0-4 on the ledgers `Ls` of a run on host `h`. -/
def issues (h : Host) (Ls : Ledgers) : Array Issue := forSlots Ls (slotIssues h)

/-- Notes of P0-4 on the ledgers `Ls` of a run on host `h`. -/
def notes (h : Host) (Ls : Ledgers) : Array Issue := forSlots Ls (slotNotes h)

end KV.Barrier
