/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
Entry points.

* `runCase k i : Report` evaluates kernel `k` on input `i` and decides the five P0 specs;
  `#eval runCase k i` prints the verdicts, counterexamples and notes (with IR loc, every
  distinct loc at least once), then `VERDICT <input> <P0-n> PASS|FAIL` lines,
  `NOTES <input> <P0-n> <count>` lines (specs with notes only) and
  `UNDECIDED <input> <P0-n>` lines for scripts. Notes never affect a verdict.
* `Holds s k i : Prop` is spec `s` for `k` on `i` (false if evaluation fails), decidable,
  so `theorem _ : Holds .bounds k i := by native_decide` is the verdict as a theorem
  (trusts the Lean compiler).
* A refuted spec is a certain violation only if it has a certain counterexample
  (`SpecResult.certain`). Without one it is undecided, and an `UNDECIDED` line says so:
  on an evaluation error of kind `limit` or `unexpected` (every spec; a `kernel` error is
  certain), after a role trace stopped (below), for P0-2 and P0-4 on a cluster
  whose barrier ledger does not converge (Ledger.lean, `unconverged`), for P0-4 on a slot
  whose transaction bytes break it only through another CTA's landings that may complete
  a later phase than their issue order shows (Ledger.lean, "Landers"), for P0-6 on an
  `nb_open` not marked certain or a pairing it cannot decide (NamedBarriers.lean), or for
  P1-1 and P1-2 pairs that only the lockstep reading, named barriers used with two counts
  or another round of a ledger that does not converge might order, and P1-2's other
  undecided pairs (TmemReuse.lean, StoreReuse.lean). The evaluation-error line names the
  kind (Eval.lean `ErrKind`).
* A role that loads metadata outside the tensor stops there (Eval.lean): P0-1 fails with
  that load, and P0-2..P0-6, P1-1 and P1-2 hold only on runs where every role trace is
  complete. On other runs they are false, listed with the counterexamples that are
  certain anyway: P0-2/P0-4 (P1-1/P1-2) on the clusters whose roles all completed,
  P0-3/P0-5 on the events evaluated (no coverage holes), P0-6 on the CTAs whose roles all
  completed and the single events that ran. A spec without such a counterexample is
  undecided. A role whose `assume` does not hold (KIR change 16) stops the same way,
  without a P0-1 finding, and P0-1 then holds only on complete runs too. The other roles
  of its cluster (its CTA without clusters) may have used a value the broken assumption
  made wrong (a warp-specialize capture), so P0-1, P0-3 and P0-5 list only the
  counterexamples of the clusters where no trace stopped at an `assume`
  (`Run.certainPart`); the `not decided` line and the summary name the assumption.
* On an input that sets an elision flag of the case (`Input.elisions`), the P0-1 and P0-3
  checks are P0-5's (MaskElision.lean); the summary line says so.
-/
import KV.Spec.Bounds
import KV.Spec.TripCount
import KV.Spec.Coverage
import KV.Spec.Barrier
import KV.Spec.MaskElision
import KV.Spec.TmemReuse
import KV.Spec.NamedBarriers
import KV.Spec.StoreReuse
import KV.Spec.Global

namespace KV

/-- The specs of a P0 KIR: the five P0 specs, then P0-7 if it has sync tensors (KIR change
18). -/
def specsOf (k : Kernel) : List SpecId :=
  if k.tensors.any (·.sync.isSome) then SpecId.all ++ SpecId.global else SpecId.all

def SpecId.Holds : SpecId → Run → Prop
  | .bounds, r => r.complete ∧ Bounds.Holds r
  | .tripCount, r => r.complete ∧ TripCount.Holds r
  | .coverage, r => r.complete ∧ Coverage.Holds r
  | .barrier, r => r.complete ∧ Barrier.Holds r
  | .maskElision, r => r.complete ∧ MaskElision.Holds r
  | .namedBarriers, r => r.complete ∧ NamedBarriers.Holds r
  | .tmemReuse, r => r.complete ∧ TmemReuse.Holds r
  | .storeReuse, r => r.complete ∧ StoreReuse.Holds r
  | .gwait, r => Global.Holds r = true

/-- The decision of `s.Holds r`, by an `if` chain: only the asked spec runs (a `match` on the
enum compiles to code that evaluates every arm before the switch, so each spec's theorem ran
every spec). -/
def SpecId.holdsB (s : SpecId) (r : Run) : Bool :=
  if s matches .bounds then r.complete && decide (Bounds.Holds r)
  else if s matches .tripCount then r.complete && decide (TripCount.Holds r)
  else if s matches .coverage then r.complete && decide (Coverage.Holds r)
  else if s matches .barrier then r.complete && decide (Barrier.Holds r)
  else if s matches .maskElision then r.complete && decide (MaskElision.Holds r)
  else if s matches .namedBarriers then r.complete && decide (NamedBarriers.Holds r)
  else if s matches .tmemReuse then r.complete && decide (TmemReuse.Holds r)
  else if s matches .storeReuse then r.complete && decide (StoreReuse.Holds r)
  else Global.Holds r

instance (s : SpecId) (r : Run) : Decidable (s.Holds r) :=
  decidable_of_iff (s.holdsB r = true) (by cases s <;> simp [SpecId.holdsB, SpecId.Holds])

def RunHolds (s : SpecId) : Except EvalError Run → Prop
  | .ok r => s.Holds r
  | .error _ => False

instance (s : SpecId) : (x : Except EvalError Run) → Decidable (RunHolds s x)
  | .ok r => inferInstanceAs (Decidable (s.Holds r))
  | .error _ => isFalse id

/-- Spec `s` holds for kernel `k` on input `i`. -/
def Holds (s : SpecId) (k : Kernel) (i : Input) : Prop := RunHolds s (eval k i)

instance (s : SpecId) (k : Kernel) (i : Input) : Decidable (Holds s k i) :=
  inferInstanceAs (Decidable (RunHolds s (eval k i)))

/-- The analyses of one run that its report and its decisions share (`checkAll`, Command.lean):
each is computed when first read, at most once (before, the report and every decision computed
their own: three barrier-ledger computations per P0 input, two P1-ledger computations per P1
spec, P0-1's, P0-3's and P0-5's decisions twice and P0-3's targets three times, P0-6's analysis,
P1-1's and P1-2's pair searches twice, and a P0 input with sync tensors its barrier ledgers once
more and P0-7's analysis twice). -/
structure Shared where
  ledgers : Thunk Ledgers
  ledgersP1 : Thunk Ledgers
  /-- `decide (Bounds.EventsOk r)` and `decide (Bounds.NoMetaOob r)` (P0-1, P0-5). -/
  eventsOk : Thunk Bool
  noMetaOob : Thunk Bool
  /-- P0-3's targets and their plain-store ledgers, and `decide (Coverage.Ok r)` (P0-3, P0-5). -/
  covTargets : Thunk (Array Coverage.Target)
  covLedgers : Thunk (Array Coverage.Ledger)
  covOk : Thunk Bool
  named : Thunk NamedBarriers.Out
  /-- The P1 CTAs (P1-1) and contexts (P1-2) with their conflicting pairs. -/
  tmem : Thunk (Array (TmemReuse.P1Cta × Array (Nat × Nat × TmemReuse.PairKind)))
  store : Thunk (Array (StoreReuse.Ctx × Array (Nat × Nat × StoreReuse.PairKind)))
  /-- P0-7's analysis (`Global.analyze` with the barrier ledgers). -/
  global : Thunk Global.Outcome

/-- The shared analyses of `r` with its barrier ledgers `Ls` (`r.ledgers`) and P1 ledgers `L1`
(`r.ledgersP1`). -/
def Shared.ofLedgers (r : Run) (Ls : Thunk Ledgers) (L1 : Thunk Ledgers := Thunk.mk fun _ => r.ledgersP1Par) :
    Shared :=
  let tg : Thunk (Array Coverage.Target) := Thunk.mk fun _ => Coverage.targets r.host
  let cl : Thunk (Array Coverage.Ledger) := Thunk.mk fun _ => Coverage.ledgers r tg.get
  { ledgers := Ls, ledgersP1 := L1,
    eventsOk := Thunk.mk fun _ => Bounds.eventsOkPar r,
    noMetaOob := Thunk.mk fun _ => Bounds.noMetaOobPar r,
    covTargets := tg, covLedgers := cl,
    covOk := Thunk.mk fun _ => Coverage.okWithB tg.get cl.get r,
    named := Thunk.mk fun _ => NamedBarriers.analyze r,
    tmem := Thunk.mk fun _ => TmemReuse.withPairs (TmemReuse.cols r) (TmemReuse.p1Ctas r L1.get),
    store := Thunk.mk fun _ => StoreReuse.withPairs (StoreReuse.ctxs r L1.get),
    global := Thunk.mk fun _ => Global.analyze r Ls.get }

def Shared.of (r : Run) : Shared := Shared.ofLedgers r (Thunk.mk fun _ => r.ledgersPar)

/-- P0-1's decision from the shared analyses (`Bounds.holdsPar`). -/
def Shared.bounds (sh : Shared) (r : Run) : Bool := (r.elided || sh.eventsOk.get) && sh.noMetaOob.get

/-- P0-3's decision from the shared analyses (`Coverage.decide_holds`). -/
def Shared.coverage (sh : Shared) (r : Run) : Bool := r.elided || sh.covOk.get

/-- P0-5's decision from the shared analyses (`MaskElision.decide_holds`). -/
def Shared.maskElision (sh : Shared) (r : Run) : Bool := !r.elided || (sh.eventsOk.get && sh.covOk.get)

/-- `s.holdsB r` from the shared analyses `sh = Shared.of r` (`holdsWith_of`). -/
def SpecId.holdsWith (s : SpecId) (r : Run) (sh : Shared) : Bool :=
  if s matches .bounds then r.complete && sh.bounds r
  else if s matches .tripCount then r.complete && decide (TripCount.HoldsWith sh.ledgers.get)
  else if s matches .coverage then r.complete && sh.coverage r
  else if s matches .barrier then r.complete && decide (Barrier.HoldsWith sh.ledgers.get)
  else if s matches .maskElision then r.complete && sh.maskElision r
  else if s matches .namedBarriers then r.complete && sh.named.get.ok
  else if s matches .tmemReuse then r.complete && TmemReuse.holdsB (TmemReuse.cols r) sh.tmem.get
  else if s matches .storeReuse then r.complete && StoreReuse.holdsOnB sh.store.get
  else r.complete && sh.global.get.ok

theorem NamedBarriers.decide_holds (r : Run) :
    decide (NamedBarriers.Holds r) = (NamedBarriers.analyze r).ok := by
  rw [Bool.eq_iff_iff, decide_eq_true_iff]; rfl

theorem holdsWith_of (s : SpecId) (r : Run) : s.holdsWith r (Shared.of r) = s.holdsB r := by
  have hb : (Shared.of r).bounds r = decide (Bounds.Holds r) := by
    rw [← Bounds.holdsPar_eq]; rfl
  have hc : (Shared.of r).coverage r = decide (Coverage.Holds r) := by
    rw [Coverage.decide_holds, ← Coverage.okB_eq]; rfl
  have hm : (Shared.of r).maskElision r = decide (MaskElision.Holds r) := by
    rw [MaskElision.decide_holds, ← Coverage.okB_eq, ← Bounds.eventsOkPar_eq]; rfl
  have ht : TmemReuse.holdsB (TmemReuse.cols r) (Shared.of r).tmem.get = decide (TmemReuse.Holds r) := by
    show TmemReuse.holdsB (TmemReuse.cols r)
      (TmemReuse.withPairs (TmemReuse.cols r) (TmemReuse.p1Ctas r r.ledgersP1Par)) = _
    rw [Run.ledgersP1Par_eq]; exact TmemReuse.holdsB_eq _ _
  have hs : StoreReuse.holdsOnB (Shared.of r).store.get = decide (StoreReuse.Holds r) := by
    show StoreReuse.holdsOnB (StoreReuse.withPairs (StoreReuse.ctxs r r.ledgersP1Par)) = _
    rw [Run.ledgersP1Par_eq]; exact StoreReuse.holdsOnB_eq _
  have hl : (Shared.of r).ledgers.get = r.ledgers := Run.ledgersPar_eq r
  have hn : (Shared.of r).named.get = NamedBarriers.analyze r := rfl
  have hg : (Shared.of r).global.get = Global.analyze r r.ledgers := by
    show Global.analyze r r.ledgersPar = _
    rw [Run.ledgersPar_eq]
  cases s <;> cases h : r.complete <;>
    simp [SpecId.holdsWith, SpecId.holdsB, h, hb, hc, hm, ht, hs, hl, hn, hg, TripCount.Holds,
      Barrier.Holds, NamedBarriers.decide_holds, Global.Holds]

theorem decide_holds (s : SpecId) (r : Run) : decide (s.Holds r) = s.holdsB r := by
  have e : s.Holds r ↔ s.holdsB r = true := by cases s <;> simp [SpecId.holdsB, SpecId.Holds]
  rw [Bool.eq_iff_iff, decide_eq_true_iff]
  exact e

structure SpecResult where
  spec : SpecId
  holds : Bool
  issues : Array Issue
  /-- Findings that do not affect the verdict (references/specs.md). -/
  notes : Array Issue
  /-- Some issue is a certain counterexample (a `not decided` line is not one). A spec that
  does not hold without one is undecided on the input; on an input whose evaluation fails
  every spec is certain iff the error is a `kernel` error. -/
  certain : Bool

/-- Why P0-2..P0-6 (P0-1 too after an `assume` stop) are undecided on `r`: the first role
trace that stops (an out-of-bounds metadata load, an assumption that does not hold; none
if every trace is complete), with every reason a trace stopped for. -/
def undecided (r : Run) : Array Issue :=
  let stops := r.ctas.flatMap fun c => (c.roles.filter (·.stopped)).map fun t => (c.pid, t)
  match stops[0]? with
  | none => #[]
  | some (pid, t) =>
    #[{ loc := t.stopLoc, context := r.host.roleCtx pid t.role,
        msg := s!"not decided on this input: {stops.size} role trace(s) stop at {stopReasons (stops.map (·.2))}, so their later events are unknown; the lines of this spec that are not `not decided` lines are certain counterexamples" }]

/-- The stopped role traces of `r`. -/
def Run.stops (r : Run) : Array RoleTrace :=
  r.ctas.flatMap fun c => c.roles.filter (·.stopped)

/-- The trace stopped at an `assume` that does not hold. -/
def RoleTrace.assumeStop (t : RoleTrace) : Bool :=
  t.stopped && match t.why with
    | .assume .. => true
    | .metaOob | .blocked .. => false

/-- Some role trace of `r` stopped at an `assume` that does not hold. -/
def Run.assumeStopped (r : Run) : Bool := r.ctas.any (·.roles.any (·.assumeStop))

/-- `r` with every role trace of each cluster (each CTA without clusters) in which some
trace stopped at an `assume` emptied and marked stopped: the events there may rest on a
value the broken assumption made wrong (a warp-specialize capture another role read), so
none of them is a certain counterexample. -/
def Run.certainPart (r : Run) : Run :=
  let bad := (r.ctas.filter (·.roles.any (·.assumeStop))).map (r.host.clusterOf ·.pid)
  { r with ctas := r.ctas.map fun c =>
      if bad.contains (r.host.clusterOf c.pid) then
        { c with roles := c.roles.map fun t => { t with events := #[], stopped := true } }
      else c }

/-- Without a trace stopped at an `assume`, `r.certainPart` is `r` (the report then takes
P0-3's and P0-5's lines from the shared ledgers, `SpecId.resultWith`). -/
theorem Run.certainPart_eq_self (r : Run) (h : r.assumeStopped = false) : r.certainPart = r := by
  have hbad : (r.ctas.filter (·.roles.any (·.assumeStop))) = #[] := by
    simp only [Run.assumeStopped, Array.any_eq_false] at h
    apply Array.filter_eq_empty_iff.mpr
    intro a ha
    obtain ⟨i, hi, rfl⟩ := Array.getElem_of_mem ha
    exact h i hi
  unfold Run.certainPart
  simp [hbad]

/-- Verdict, counterexamples and notes of spec `s` on run `r` from the shared analyses `sh`
(`Shared.of r`); the same decision as `s.Holds r`. -/
def SpecId.resultWith (s : SpecId) (r : Run) (sh : Shared) : SpecResult :=
  let un := undecided r
  -- `un` and `undec` (P0-2, P0-4: clusters whose ledger did not converge; P0-4: slots whose
  -- landers' bytes balance only if a landing completes a later phase than its issue order
  -- shows) are the only issues of a P0 spec that are not certain counterexamples.
  let p0 (holds : Bool) (own nts : Array Issue) (undec : Array Issue := #[]) : SpecResult :=
    { spec := s, holds := holds, issues := un ++ undec ++ own, notes := nts, certain := !own.isEmpty }
  match s with
  | .bounds =>
    let own := Bounds.issues r.certainPart
    { spec := s, holds := r.complete && sh.bounds r,
      issues := (if r.assumeStopped || r.hangs then un else #[]) ++ own, notes := Bounds.notes r,
      certain := !own.isEmpty }
  | .tripCount =>
    let Ls := sh.ledgers.get
    p0 (r.complete && decide (TripCount.HoldsWith Ls)) (TripCount.issues r.host Ls) (TripCount.notes r.host Ls) (unconverged r.host Ls)
  | .coverage =>
    -- `Coverage.issues r.certainPart` and `Coverage.notes r` from the shared targets and
    -- ledgers: `r.certainPart` is `r` unless a trace stopped at an `assume`
    -- (`Run.certainPart_eq_self`); then its ledgers differ and are computed apart
    let issues := if r.elided then #[]
      else if r.assumeStopped then Coverage.okIssues r.certainPart
      else Coverage.okIssuesWith sh.covTargets.get sh.covLedgers.get r
    let notes := if r.elided then #[] else Coverage.okNotesWith sh.covTargets.get r
    p0 (r.complete && sh.coverage r) issues notes
  | .barrier =>
    let Ls := sh.ledgers.get
    p0 (r.complete && decide (Barrier.HoldsWith Ls)) (Barrier.issues r.host Ls) (Barrier.notes r.host Ls)
      (unconverged r.host Ls ++ Barrier.undecided r.host Ls)
  | .maskElision =>
    -- `MaskElision.issues r.certainPart` and `MaskElision.notes r` from the shared targets and
    -- ledgers, as for P0-3
    let issues := if !r.elided then #[]
      else if r.assumeStopped then MaskElision.issues r.certainPart
      else Bounds.eventsIssues r ++ Coverage.okIssuesWith sh.covTargets.get sh.covLedgers.get r
    let notes := if !r.elided then #[] else Bounds.eventsNotes r ++ Coverage.okNotesWith sh.covTargets.get r
    p0 (r.complete && sh.maskElision r) issues notes
  | .namedBarriers =>
    let o := sh.named.get
    { spec := s, holds := r.complete && o.ok, issues := un ++ o.certain ++ o.undecided,
      notes := #[], certain := !o.certain.isEmpty }
  | .tmemReuse =>
    let o := TmemReuse.analyze.analyzeWith r sh.tmem.get
    { spec := s, holds := r.complete && o.ok, issues := o.issues, notes := o.notes, certain := o.certain }
  | .storeReuse =>
    let o := StoreReuse.analyze.analyzeWith r sh.store.get
    { spec := s, holds := r.complete && o.ok, issues := o.issues, notes := o.notes, certain := o.certain }
  | .gwait =>
    let o := sh.global.get
    { spec := s, holds := r.complete && o.ok, issues := o.issues, notes := #[], certain := o.certain }

/-- `s.resultWith` with the barrier ledgers `Ls = r.ledgers`. -/
def SpecId.result (s : SpecId) (r : Run) (Ls : Ledgers) : SpecResult :=
  s.resultWith r (Shared.ofLedgers r (Thunk.pure Ls))

structure Report where
  kernel : String
  input : Input
  error : Option EvalError
  summary : String
  results : List SpecResult

/-- " in clusters (2,1,1)" for a cluster launch, else "". -/
def clusterStr (h : Host) : String := if h.csize ≤ 1 then "" else s!" in clusters {pidStr h.cluster}"

/-- The report of kernel `k` on input `i` from its evaluation `x` (`eval k i`; `checkAll` shares
one evaluation and its analyses between the report and the theorems, Command.lean). -/
def reportOf (k : Kernel) (i : Input) (x : Except EvalError Run) : Report :=
  match x with
  | .error e =>
    { kernel := k.kernel, input := i, error := some e, summary := "",
      results := (specsOf k).map fun s =>
        { spec := s, holds := false, issues := #[], notes := #[], certain := e.kind == .kernel } }
  | .ok r => reportRun k i r (Shared.of r)
where
  /-- The report of a run from the shared analyses `sh` (`Shared.of r`). -/
  reportRun (k : Kernel) (i : Input) (r : Run) (sh : Shared) : Report :=
    let events := r.ctas.foldl (fun n c => c.roles.foldl (fun n t => n + t.events.size) n) 0
    let stops := r.stops
    { kernel := k.kernel, input := i, error := none,
      summary := s!"grid {pidStr r.host.grid}{clusterStr r.host}: {r.ctas.size} CTA(s) x {k.roles.length} role(s), {events} events" ++
        (if stops.isEmpty then ""
         else if r.hangs then s!"; {stops.size} role trace(s) are blocked forever at a take of a tagged word, so P0-1..P0-5 are not decided"
         else s!"; {stops.size} role trace(s) stop at {stopReasons stops (detail := false)}, so {if r.assumeStopped then "P0-1" else "P0-2"}..P0-5 are not decided") ++
        (match i.elisions with
         | [] => ""
         | fs => s!"; elision flag(s) {", ".intercalate fs} set: the P0-1 and P0-3 checks of this input are reported under P0-5") ++
        (if r.host.hasSync then s!"; global sync: {Global.summary r}" else ""),
      results := (specsOf k).map (·.resultWith r sh) }

def runCase (k : Kernel) (i : Input) : Report := reportOf k i (eval k i)

/-- The P1 specs a P1 KIR is decided for: those its `p1` names, else P1-1 iff it has a TMEM
allocation (KIR change 10). -/
def Kernel.p1Specs (k : Kernel) : List SpecId :=
  if k.p1.isEmpty then (if k.tmem.isSome then [.tmemReuse] else [])
  else SpecId.p1.filter fun s => k.p1.contains s.label

/-- The specs `checkAll` decides for a KIR of kind `kind` (0 P0, 1 P1, 2 P0-6; KV/Profile.lean,
KV/SelfCheck.lean). -/
def Kernel.specsOf (k : Kernel) (kind : Nat) : List SpecId :=
  if kind = 1 then k.p1Specs else if kind = 2 then SpecId.nb else KV.specsOf k

/-- P1 specs of a P1 KIR (references/specs.md, "P1-1", "P1-2") on input `i` from its
evaluation `x`: the P1 summary line (P1-1: accesses, conflicting pairs, named barriers,
blocked waits; P1-2: bulk stores, writes, unordered pairs), then each P1 spec's verdict,
counterexamples and notes. -/
def reportP1Of (k : Kernel) (i : Input) (x : Except EvalError Run) : Report :=
  match x with
  | .error e =>
    { kernel := k.kernel, input := i, error := some e, summary := "",
      results := k.p1Specs.map fun s =>
        { spec := s, holds := false, issues := #[], notes := #[], certain := e.kind == .kernel } }
  | .ok r => reportRun k i r (Shared.of r)
where
  /-- The P1 report of a run from the shared analyses `sh` (`Shared.of r`). -/
  reportRun (k : Kernel) (i : Input) (r : Run) (sh : Shared) : Report := Id.run do
    let specs := k.p1Specs
    let events : Nat := r.ctas.foldl (fun n c => c.roles.foldl (fun n t => n + t.events.size) n) 0
    let stops := r.stops
    let mut summary := s!"grid {pidStr r.host.grid}{clusterStr r.host}: {r.ctas.size} CTA(s) x {k.roles.length} role(s), {events} events; P1: "
    let mut res : List SpecResult := []
    let mut stats : List String := []
    if specs.contains .tmemReuse then
      let o := TmemReuse.analyze.analyzeWith r sh.tmem.get
      res := res ++ [{ spec := .tmemReuse, holds := r.complete && o.ok, issues := o.issues, notes := o.notes, certain := o.certain }]
      stats := stats ++ [s!"{o.accesses} TMEM accesses, {o.pairs} conflicting pair(s) classified ({o.lockOnly} ordered only under the lockstep contents), {o.named} named-barrier slot(s)" ++
        (if o.inconsistent.isEmpty then "" else s!" (inconsistent: ids {o.inconsistent.toList})") ++
        s!", {o.blocked} blocked wait(s)"]
    if specs.contains .storeReuse then
      let o := StoreReuse.analyze.analyzeWith r sh.store.get
      res := res ++ [{ spec := .storeReuse, holds := r.complete && o.ok, issues := o.issues, notes := o.notes, certain := o.certain }]
      stats := stats ++ [s!"P1-2: {o.stores} bulk store(s), {o.writes} write(s), {o.pairs} unordered conflicting pair(s) ({o.undecided} undecided), {o.named} named-barrier slot(s)" ++
        (if o.inconsistent.isEmpty then "" else s!" (inconsistent: ids {o.inconsistent.toList})") ++
        s!", {o.blocked} blocked wait(s)"]
    summary := summary ++ "; ".intercalate stats ++
      (if stops.isEmpty then "" else s!"; {stops.size} role trace(s) stop at {stopReasons stops (detail := false)}, so {", ".intercalate (specs.map (·.label))} {if specs.length = 1 then "is" else "are"} not decided")
    return { kernel := k.kernel, input := i, error := none, summary, results := res }

/-- P0-6 on a P0-6 KIR (references/specs.md, "P0-6") on input `i` from its evaluation `x`:
the summary line (named barrier operations, `nb_open` events, `nb_rep` regions reached), then
the P0-6 verdict and its counterexamples. -/
def reportNbOf (k : Kernel) (i : Input) (x : Except EvalError Run) : Report :=
  match x with
  | .error e =>
    { kernel := k.kernel, input := i, error := some e, summary := "",
      results := SpecId.nb.map fun s =>
        { spec := s, holds := false, issues := #[], notes := #[], certain := e.kind == .kernel } }
  | .ok r => reportRun k i r (Shared.of r)
where
  /-- The P0-6 report of a run from the shared analyses `sh` (`Shared.of r`). -/
  reportRun (k : Kernel) (i : Input) (r : Run) (sh : Shared) : Report :=
    let (ops, opens, reps) := NamedBarriers.counts r
    let held := NamedBarriers.heldArrivals r
    let heldStr := if held == 0 then "" else s!", {held} arrival(s) held under the warp-ahead reading"
    { kernel := k.kernel, input := i, error := none,
      summary := s!"grid {pidStr r.host.grid}{clusterStr r.host}: {r.ctas.size} CTA(s) x {k.roles.length} role(s); P0-6: {ops} named barrier operation(s), {opens} not counted (nb_open), {reps} repeated region(s) reached{heldStr}",
      results := SpecId.nb.map (·.resultWith r sh) }

def runNb (k : Kernel) (i : Input) : Report := reportNbOf k i (eval k i)

def runP1 (k : Kernel) (i : Input) : Report := reportP1Of k i (eval k i)

/-- Counterexample lines, and note lines, printed per spec beyond one per loc. -/
def maxShown : Nat := 12

/-- Lines `pre ++ line`: the first issue at every distinct loc (so a report names every
source line involved), then the others in order up to `maxShown` lines in all, then a
count of the rest. -/
def shown (pre : String) (xs : Array Issue) : Array String := Id.run do
  let mut locs : Array Loc := #[]
  let mut firsts := #[]
  let mut rest := #[]
  for x in xs do
    if locs.contains x.loc then
      rest := rest.push x
    else
      locs := locs.push x.loc
      firsts := firsts.push x
  let pick := firsts ++ rest.extract 0 (maxShown - firsts.size)
  return pick.map (s!"  {pre}" ++ ·.render) ++
    (if xs.size > pick.size then #[s!"  {pre}... {xs.size - pick.size} more"] else #[])

def Report.render (rep : Report) : String := Id.run do
  let flags := rep.input.flags.map fun (k, v) => s!"{k}={v}"
  let mut out := #[s!"== {rep.kernel} / input {rep.input.name}" ++
    (if flags.isEmpty then "" else s!" (flags {", ".intercalate flags})")]
  match rep.error with
  | some e =>
    out := out.push s!"evaluation error ({e.kind.label}): {if e.loc.isEmpty then "?" else e.loc}: {e.context}: {e.msg} ({e.kind.consequence})"
  | none => out := out.push rep.summary
  let mut verdicts := #[]
  let mut noteCounts := #[]
  let mut undecided := #[]
  for res in rep.results do
    -- The verdict comes from the decision procedure; the finder only explains it.
    let consistent := rep.error.isSome || res.holds == res.issues.isEmpty
    let v := if !consistent then "INCONSISTENT" else if res.holds then "PASS" else "FAIL"
    out := out.push s!"{res.spec.label} {res.spec.title}: {v}"
    out := out ++ shown "" res.issues
    if !consistent then
      out := out.push "  internal error: the decision procedure and the counterexample finder disagree"
    out := out ++ shown "note: " res.notes
    verdicts := verdicts.push s!"VERDICT {rep.input.name} {res.spec.label} {v}"
    if !res.notes.isEmpty then
      noteCounts := noteCounts.push s!"NOTES {rep.input.name} {res.spec.label} {res.notes.size}"
    -- A refuted spec without a certain counterexample: neither proved nor violated.
    if consistent && !res.holds && !res.certain then
      undecided := undecided.push s!"UNDECIDED {rep.input.name} {res.spec.label}"
  return "\n".intercalate (out ++ verdicts ++ noteCounts ++ undecided).toList

instance : ToString Report := ⟨Report.render⟩

end KV
