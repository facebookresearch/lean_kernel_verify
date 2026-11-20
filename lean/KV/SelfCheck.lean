/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
Self-checks of the evaluator's shortcuts (tests/lean/run_all.sh). No theorem, verdict or report
uses this module, and KV.lean does not import it (scripts/build_lean.sh compiles it into
libKV.so, as every module under lean/). `frontend/emit_lean.py --selfcheck` writes, instead of
`#lkv_check`,

  #eval KV.SelfCheck.run kernel input SPECS

(SPECS: the specs `#lkv_check` decides for the KIR, e.g. `[.storeReuse]`), which checks on that
input:

1. the evaluation, twice, the two runs compared event by event:
   * `KV.eval`: the CTAs in parallel tasks (`evalAll`, `parMap`), the lane loops of tensor
     expressions and masks without per-lane state (`mapBin`, `cmpLanes`, `bcastData`,
     `maskedVals`);
   * `evalRef`: the CTAs one after another in a loop that stops at the first error, as `eval`
     did before (`evalAllRef`: each CTA's roles in the plan's order, its handoff records and relay
     slots carried from role to role), and the reference lane loops those replace (`Ctx.ref`:
     `mapBinRef`, the zipped comparison, `bcastDataRef`, the zipped filter); in every pass of a
     run with sync tensors, with the same handoff and claim checks (`evalWith`, which `eval` is
     with `evalAll`), so the passes, their stores and claimers (`Host.puts`, `Host.claimOrder`)
     are compared too;
2. every spec's result in the report (verdict, certainty, every counterexample and note line)
   as `SpecId.resultWith` computes it from the shared analyses (`Shared`: one P0-3 ledger for
   the decision and the lines, the pair searches of P1-1 and P1-2 for the decision and the
   report, P0-1's per-event checks, P0-7's analysis with the shared ledgers) against
   `resultRef`, which computes each from scratch with the decisions `decide (s.Holds r)`, as the
   report did before;

then prints `SELFCHECK <input> OK <counts>`, or `SELFCHECK <input> DIFF <what>` at the first
difference. tests/lean/fast_paths.lean checks the same lane loops on synthetic operands (every
operation and width, edge values).

The `BitVec` operations (KV/BV.lean), `binBV`, `wraps`, `cmpBV`, `select3`, the parallel maps
(`parMap_eq`, `Run.ledgersPar_eq`, `Run.ledgersP1Par_eq`, `Run.collect_eq`), the fast decisions
(`Bounds.eventOkB_eq`, `Bounds.holdsPar_eq`, `Coverage.writeOkB_iff`, `Coverage.okB_eq`,
`TmemReuse.holdsB_eq`, `StoreReuse.holdsOnB_eq`) and the decisions from the shared analyses
(`holdsWith_of`, `checkAll_snd`) are proved equal to what they replace; 2. checks the report
lines, which no theorem covers.
-/
import KV.Check

namespace KV.SelfCheck

/-- `evalAll` as it was: every CTA in x-fastest order, one after another, each CTA's roles in
`order` with its handoff records and relay slots carried from role to role, each role run with
the reference lane loops (`Ctx.ref`); the first error stops the loop. -/
def evalAllRef (h : Host) (order : List Nat) : Except EvalError (Array CtaTrace) := do
  let roles := h.kernel.roles.toArray
  let (gx, gy, gz) := h.grid
  let mut ctas := Array.emptyWithCapacity (gx * gy * gz)
  for z in [0:gz] do
    for y in [0:gy] do
      for x in [0:gx] do
        let rank := h.rankOf (x, y, z)
        let cta := x + gx * (y + gy * z)
        let mut traces := Array.replicate roles.size ({ role := 0, events := #[], stopped := false } : RoleTrace)
        let mut records : Handoffs := {}
        let mut relay : Relay := {}
        for ri in order do
          if let some role := roles[ri]? then
            let rr := runRoleRun h (x, y, z) ri role rank records cta relay (ref := true)
            let t ← rr.trace
            traces := traces.set! ri t
            records := rr.records
            -- a putter blocked at a take never writes its later slots
            relay := if t.blocked then (relayChans false role.body).foldl (fun m c => m.insert (blockedKey c) 1) rr.relay
              else rr.relay
        ctas := ctas.push { pid := (x, y, z), roles := traces, rank }
  pure ctas

/-- `eval` as it was: `evalWith` (the role order, the passes of a run with sync tensors, the
handoff and claim checks) with `evalAllRef`. -/
def evalRef (k : Kernel) (inp : Input) : Except EvalError Run := evalWith evalAllRef k inp

/-- The parts of two role traces a spec (or a pass of `evalSyncPasses`) reads. -/
def sameTrace (a b : RoleTrace) : Bool :=
  a.role == b.role && a.stopped == b.stopped && a.why == b.why && a.events.size == b.events.size &&
  (a.events.zip b.events).all (fun (x, y) => x.ev == y.ev && x.loc == y.loc && x.work == y.work) &&
  a.blocked == b.blocked && a.claimed == b.claimed && a.puts == b.puts && a.claimWords == b.claimWords

def sameCta (a b : CtaTrace) : Bool :=
  a.pid == b.pid && a.rank == b.rank && a.roles.size == b.roles.size &&
  (a.roles.zip b.roles).all fun (x, y) => sameTrace x y

def sameError (x y : EvalError) : Bool :=
  x.msg == y.msg && x.loc == y.loc && x.context == y.context && x.kind == y.kind &&
  x.stop == y.stop && x.why == y.why && x.blocked == y.blocked

/-- Spec `s`'s result on `r` as the report computed it before the shared analyses: every
analysis from scratch, the verdict by `decide (s.Holds r)`. -/
def resultRef (s : SpecId) (r : Run) : SpecResult :=
  let Ls := r.ledgers
  let un := undecided r
  let p0 (holds : Bool) (own nts : Array Issue) (undec : Array Issue := #[]) : SpecResult :=
    { spec := s, holds := holds, issues := un ++ undec ++ own, notes := nts, certain := !own.isEmpty }
  match s with
  | .bounds =>
    let own := Bounds.issues r.certainPart
    { spec := s, holds := r.complete && decide (Bounds.Holds r),
      issues := (if r.assumeStopped || r.hangs then un else #[]) ++ own, notes := Bounds.notes r,
      certain := !own.isEmpty }
  | .tripCount => p0 (r.complete && decide (TripCount.HoldsWith Ls)) (TripCount.issues r.host Ls) (TripCount.notes r.host Ls) (unconverged r.host Ls)
  | .coverage => p0 (r.complete && decide (Coverage.Holds r)) (Coverage.issues r.certainPart) (Coverage.notes r)
  | .barrier => p0 (r.complete && decide (Barrier.HoldsWith Ls)) (Barrier.issues r.host Ls) (Barrier.notes r.host Ls)
      (unconverged r.host Ls ++ Barrier.undecided r.host Ls)
  | .maskElision => p0 (r.complete && decide (MaskElision.Holds r)) (MaskElision.issues r.certainPart)
      (MaskElision.notes r)
  | .namedBarriers =>
    let o := NamedBarriers.analyze r
    { spec := s, holds := r.complete && decide (NamedBarriers.Holds r), issues := un ++ o.certain ++ o.undecided,
      notes := #[], certain := !o.certain.isEmpty }
  | .tmemReuse =>
    let o := TmemReuse.analyze r r.ledgersP1
    { spec := s, holds := r.complete && decide (TmemReuse.Holds r), issues := o.issues, notes := o.notes,
      certain := o.certain }
  | .storeReuse =>
    let o := StoreReuse.analyze r r.ledgersP1
    { spec := s, holds := r.complete && decide (StoreReuse.Holds r), issues := o.issues, notes := o.notes,
      certain := o.certain }
  | .gwait =>
    let o := Global.analyze r Ls
    { spec := s, holds := r.complete && decide (s.Holds r), issues := o.issues, notes := #[],
      certain := o.certain }

def sameResult (a b : SpecResult) : Bool :=
  a.holds == b.holds && a.certain == b.certain &&
  a.issues.map (·.render) == b.issues.map (·.render) && a.notes.map (·.render) == b.notes.map (·.render)

def run (k : Kernel) (i : Input) (ss : List SpecId := SpecId.all) : IO Unit := do
  let say (s : String) : IO Unit := IO.println s!"SELFCHECK {i.name} {s}"
  match eval k i, evalRef k i with
  | .error e, .error e' =>
    if sameError e e' then say s!"OK (evaluation error: {e.msg.take 80})"
    else say s!"DIFF evaluation errors: {e.msg.take 80} / {e'.msg.take 80}"
  | .ok _, .error e' => say s!"DIFF only the reference evaluation fails: {e'.msg.take 80}"
  | .error e, .ok _ => say s!"DIFF only the evaluation fails: {e.msg.take 80}"
  | .ok r, .ok r' =>
    if r.ctas.size != r'.ctas.size then
      say s!"DIFF {r.ctas.size} CTAs, {r'.ctas.size} in the reference evaluation"
      return
    for (c, c') in r.ctas.zip r'.ctas do
      if !sameCta c c' then
        say s!"DIFF CTA {pidStr c.pid}"
        return
    if !(samePuts r.host.puts r'.host.puts && sameOrder r.host.claimOrder r'.host.claimOrder) then
      say "DIFF the stores or claimers of the last pass"
      return
    let sh := Shared.of r
    let mut lines := 0
    for s in ss do
      let a := s.resultWith r sh
      if !sameResult a (resultRef s r) then
        say s!"DIFF {s.label} result from the shared analyses"
        return
      lines := lines + a.issues.size + a.notes.size
    say s!"OK ({r.ctas.size} CTAs, {r.events} events, {lines} report lines)"

end KV.SelfCheck
