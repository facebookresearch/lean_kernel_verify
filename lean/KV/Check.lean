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
  certain), after an out-of-bounds metadata load (below), or for P1-1 pairs that only
  the lockstep contents or named barriers used with two counts might order
  (TmemReuse.lean). The evaluation-error line names the kind (Eval.lean `ErrKind`).
* A role that loads metadata outside the tensor stops there (Eval.lean): P0-1 fails with
  that load, and P0-2..P0-5 hold only on runs where every role trace is complete. On
  other runs they are false, listed with the counterexamples that are certain anyway:
  P0-2/P0-4 on the CTAs whose roles all completed, P0-3/P0-5 on the events evaluated (no
  coverage holes). A spec without such a counterexample is undecided.
* On an input that sets an elision flag of the case (`Input.elisions`), the P0-1 and P0-3
  checks are P0-5's (MaskElision.lean); the summary line says so.
-/
import KV.Spec.Bounds
import KV.Spec.TripCount
import KV.Spec.Coverage
import KV.Spec.Barrier
import KV.Spec.MaskElision
import KV.Spec.TmemReuse

namespace KV

def SpecId.Holds : SpecId → Run → Prop
  | .bounds, r => Bounds.Holds r
  | .tripCount, r => r.complete ∧ TripCount.Holds r
  | .coverage, r => r.complete ∧ Coverage.Holds r
  | .barrier, r => r.complete ∧ Barrier.Holds r
  | .maskElision, r => r.complete ∧ MaskElision.Holds r
  | .tmemReuse, r => r.complete ∧ TmemReuse.Holds r

instance : (s : SpecId) → (r : Run) → Decidable (s.Holds r)
  | .bounds, r => inferInstanceAs (Decidable (Bounds.Holds r))
  | .tripCount, r => inferInstanceAs (Decidable (r.complete ∧ TripCount.Holds r))
  | .coverage, r => inferInstanceAs (Decidable (r.complete ∧ Coverage.Holds r))
  | .barrier, r => inferInstanceAs (Decidable (r.complete ∧ Barrier.Holds r))
  | .maskElision, r => inferInstanceAs (Decidable (r.complete ∧ MaskElision.Holds r))
  | .tmemReuse, r => inferInstanceAs (Decidable (r.complete ∧ TmemReuse.Holds r))

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

/-- Why P0-2..P0-5 are undecided on `r`: the first role trace that stops at an
out-of-bounds metadata load (none if every trace is complete). -/
def undecided (r : Run) : Array Issue :=
  let stops := r.ctas.flatMap fun c => (c.roles.filter (·.stopped)).map fun t => (c.pid, t)
  match stops[0]? with
  | none => #[]
  | some (pid, t) =>
    #[{ loc := (t.events.back?.map (·.loc)).getD "", context := r.host.roleCtx pid t.role,
        msg := s!"not decided on this input: {stops.size} role trace(s) stop at an out-of-bounds metadata load (P0-1), so their later events are unknown; the other lines of this spec are certain counterexamples" }]

/-- Verdict, counterexamples and notes of spec `s` on run `r` with its barrier ledgers
`Ls = r.ledgers` (computed once for P0-2 and P0-4); the same decision as `s.Holds r`. -/
def SpecId.result (s : SpecId) (r : Run) (Ls : Ledgers) : SpecResult :=
  let un := undecided r
  -- `un` is the only issue of a P0 spec that is not a certain counterexample.
  let p0 (holds : Bool) (own nts : Array Issue) : SpecResult :=
    { spec := s, holds := holds, issues := un ++ own, notes := nts, certain := !own.isEmpty }
  match s with
  | .bounds =>
    let own := Bounds.issues r
    { spec := s, holds := decide (Bounds.Holds r), issues := own, notes := Bounds.notes r, certain := !own.isEmpty }
  | .tripCount => p0 (r.complete && decide (TripCount.HoldsWith Ls)) (TripCount.issues r.host Ls) (TripCount.notes r.host Ls)
  | .coverage => p0 (r.complete && decide (Coverage.Holds r)) (Coverage.issues r) (Coverage.notes r)
  | .barrier => p0 (r.complete && decide (Barrier.HoldsWith Ls)) (Barrier.issues r.host Ls) (Barrier.notes r.host Ls)
  | .maskElision => p0 (r.complete && decide (MaskElision.Holds r)) (MaskElision.issues r) (MaskElision.notes r)
  | .tmemReuse =>
    let o := TmemReuse.analyze r r.ledgersP1
    { spec := s, holds := r.complete && o.ok, issues := o.issues, notes := o.notes, certain := o.certain }

structure Report where
  kernel : String
  input : Input
  error : Option EvalError
  summary : String
  results : List SpecResult

/-- " in clusters (2,1,1)" for a cluster launch, else "". -/
def clusterStr (h : Host) : String := if h.csize ≤ 1 then "" else s!" in clusters {pidStr h.cluster}"

def runCase (k : Kernel) (i : Input) : Report :=
  match eval k i with
  | .error e =>
    { kernel := k.kernel, input := i, error := some e, summary := "",
      results := SpecId.all.map fun s =>
        { spec := s, holds := false, issues := #[], notes := #[], certain := e.kind == .kernel } }
  | .ok r =>
    let events := r.ctas.foldl (fun n c => c.roles.foldl (fun n t => n + t.events.size) n) 0
    let stops := r.ctas.foldl (fun n c => n + (c.roles.filter (·.stopped)).size) 0
    let Ls := r.ledgers
    { kernel := k.kernel, input := i, error := none,
      summary := s!"grid {pidStr r.host.grid}{clusterStr r.host}: {r.ctas.size} CTA(s) x {k.roles.length} role(s), {events} events" ++
        (if stops = 0 then "" else s!"; {stops} role trace(s) stop at an out-of-bounds metadata load, so P0-2..P0-5 are not decided") ++
        (match i.elisions with
         | [] => ""
         | fs => s!"; elision flag(s) {", ".intercalate fs} set: the P0-1 and P0-3 checks of this input are reported under P0-5"),
      results := SpecId.all.map (·.result r Ls) }

/-- P1 specs of a P1 KIR (references/specs.md, "P1-1") on input `i`: the P1 summary line
(accesses, conflicting pairs, named barriers, blocked waits), then the P1-1 verdict,
counterexamples and notes. -/
def runP1 (k : Kernel) (i : Input) : Report :=
  match eval k i with
  | .error e =>
    { kernel := k.kernel, input := i, error := some e, summary := "",
      results := SpecId.p1.map fun s =>
        { spec := s, holds := false, issues := #[], notes := #[], certain := e.kind == .kernel } }
  | .ok r =>
    let events := r.ctas.foldl (fun n c => c.roles.foldl (fun n t => n + t.events.size) n) 0
    let stops := r.ctas.foldl (fun n c => n + (c.roles.filter (·.stopped)).size) 0
    let o := TmemReuse.analyze r r.ledgersP1
    let res := [{ spec := .tmemReuse, holds := r.complete && o.ok, issues := o.issues, notes := o.notes, certain := o.certain }]
    { kernel := k.kernel, input := i, error := none,
      summary := s!"grid {pidStr r.host.grid}{clusterStr r.host}: {r.ctas.size} CTA(s) x {k.roles.length} role(s), {events} events; P1: " ++
        s!"{o.accesses} TMEM accesses, {o.pairs} conflicting pair(s) classified ({o.lockOnly} ordered only under the lockstep contents), {o.named} named-barrier slot(s)" ++
        (if o.inconsistent.isEmpty then "" else s!" (inconsistent: ids {o.inconsistent.toList})") ++
        s!", {o.blocked} blocked wait(s)" ++
        (if stops = 0 then "" else s!"; {stops} role trace(s) stop at an out-of-bounds metadata load, so P1-1 is not decided"),
      results := res }

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
