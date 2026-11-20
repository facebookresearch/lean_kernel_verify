/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
Debug only: where one evaluation spends its time. No theorem, verdict or report uses it, and
KV.lean does not import it (scripts/build_lean.sh compiles it into libKV.so, as every module
under lean/). `frontend/emit_lean.py --profile LOG[:N]` writes, instead of `#lkv_check`,

  #eval KV.Profile.run kernel input "LOG" N KIND

(KIND: 0 a P0 KIR, 1 a P1 KIR, 2 a P0-6 KIR, as `checkAll`), which evaluates `kernel` on
`input` as `KV.eval` does, but one CTA after another (`evalCtaWith`, each role run timed; a run
with sync tensors: its first pass), and appends to LOG, as it goes:

  RUN cta=<ci> role=<name> ms=<t> nodes=<n> ...           every role run of the first 2 CTAs
  CTA <ci> <ms> roles=<n>                                 per CTA
  ROLE <name> runs=<r> ms=<t> nodes=<n> ns/node=<x>       per role, totals
  LOC <loc> execs=<e> nodes=<n>                           the statement locs with most nodes
  PHASE <name> <ms>                                       setup, the role plan, the CTA loop,
                                                          the passes of a run with sync tensors
                                                          (`evalSyncPasses`, the first again),
                                                          the run's handoff and claim checks,
                                                          the ledgers or P0-6's analysis, P0-7's
                                                          analysis, each spec's result and
                                                          decision

`N > 0` evaluates only the first N CTAs (no phase after the CTA loop). Nodes are expression nodes
evaluated (`Ctx.prof`). `N = -1` (`dumpIssues`) appends every issue and note of every spec
instead, one line each.
-/
import KV.Check

namespace KV.Profile

/-- `f ()`, evaluated inside the action, and its wall time in ns. -/
@[noinline] def timed {α : Type} (f : Unit → α) : IO (α × Nat) := do
  let t0 ← IO.monoNanosNow
  let x ← IO.lazyPure f
  let t1 ← IO.monoNanosNow
  return (x, t1 - t0)

structure RoleAcc where
  runs : Nat := 0
  ns : Nat := 0
  nodes : Nat := 0

def ms (ns : Nat) : String :=
  let t := ns / 1000
  s!"{t / 1000}.{toString (t % 1000 + 1000) |>.drop 1}"

def run (k : Kernel) (i : Input) (log : String) (maxCtas : Nat := 0) (kind : Nat := 0)
    (detail : Nat := 2) : IO Unit := do
  let hd ← IO.FS.Handle.mk log .append
  let say (s : String) : IO Unit := do hd.putStrLn s; hd.flush
  let tAll ← IO.monoNanosNow
  say s!"INPUT {i.name}"
  let (h?, t) ← timed fun _ => setupHost k i
  say s!"PHASE setup {ms t}"
  let h ← match h? with
    | .ok h => pure h
    | .error e => do say s!"ERROR setup {e.msg}"; return
  let (plan, t) ← timed fun _ => (do
    let (putters, order) ← handoffPlan k
    return (putters, ← roleOrder k putters order) : Except EvalError (Std.HashMap String (List Nat) × List Nat))
  say s!"PHASE plan {ms t}"
  let (putters, order) ← match plan with
    | .ok x => pure x
    | .error e => do say s!"ERROR plan {e.msg}"; return
  let (gx, gy, gz) := h.grid
  let n := gx * gy * gz
  let lim := if maxCtas == 0 then n else min n maxCtas
  let acc ← IO.mkRef ({} : Std.HashMap String RoleAcc)
  let locs ← IO.mkRef ({} : Std.HashMap String (Nat × Nat))
  let mut ctas : Array CtaTrace := #[]
  let mut err : Option EvalError := none
  let tLoop ← IO.monoNanosNow
  for ci in [0:lim] do
    let pid := h.pidAt ci
    let rank := h.rankOf pid
    let timedRun := fun (ri : Nat) (r : Role) (records : Handoffs) (relay : Relay) => do
      let (rr, t) ← timed fun _ => runRoleRun h pid ri r rank records ci relay (prof := true)
      if ci < detail then
        let st := match rr.trace with
          | .ok tr => s!"ok events={tr.events.size} stopped={tr.stopped}"
          | .error e => s!"error {e.loc}: {e.msg.take 120}"
        say s!"RUN cta={ci} role={r.name} ms={ms t} nodes={rr.evals} {st}"
      acc.modify fun m =>
        let a := m.getD r.name {}
        m.insert r.name { runs := a.runs + 1, ns := a.ns + t, nodes := a.nodes + rr.evals }
      locs.modify fun m => rr.profLoc.fold (init := m) fun m key (e, nd) =>
        let (e0, n0) := m.getD key (0, 0)
        m.insert key (e0 + e, n0 + nd)
      pure rr
    let t0 ← IO.monoNanosNow
    let x ← (evalCtaWith (m := IO) h order ci timedRun).run
    let t1 ← IO.monoNanosNow
    say s!"CTA {ci} {ms (t1 - t0)} roles={k.roles.length}"
    match x with
    | .ok c => ctas := ctas.push c
    | .error e => do
      say s!"ERROR cta {ci}: {e.context}: {e.loc}: {e.msg}"
      err := some e
      break
  let tEnd ← IO.monoNanosNow
  say s!"PHASE ctas {ms (tEnd - tLoop)} ({ctas.size} of {n})"
  for (name, a) in (← acc.get).toList do
    say s!"ROLE {name} runs={a.runs} ms={ms a.ns} nodes={a.nodes} ns/node={if a.nodes == 0 then 0 else a.ns / a.nodes}"
  let top := ((← locs.get).toList.toArray.qsort fun a b => a.2.2 > b.2.2).extract 0 40
  for (key, (e, nd)) in top do
    say s!"LOC {key} execs={e} nodes={nd}"
  if err.isSome || lim < n then
    say s!"PHASE total {ms ((← IO.monoNanosNow) - tAll)}"
    return
  -- the rest of `eval` (`evalWith`): the passes of a run with sync tensors, the run's checks
  let rx ← if h.hasSync then do
      let (x, t) ← timed fun _ => evalSyncPassesWith (evalAll · order) h
      if let .ok (_, passes) := x then say s!"PHASE passes {ms t} ({passes} pass(es))"
      pure ((·.1) <$> x)
    else pure (.ok ({ host := h, ctas } : Run) : Except EvalError Run)
  let r ← match rx with
    | .ok r => pure r
    | .error e => do
      say s!"ERROR passes {e.msg}"
      say s!"PHASE total {ms ((← IO.monoNanosNow) - tAll)}"
      return
  let (c, t) ← timed fun _ => checkRun k r putters
  say s!"PHASE checks {ms t}"
  if let .error e := c then
    say s!"ERROR checks {e.msg}"
    say s!"PHASE total {ms ((← IO.monoNanosNow) - tAll)}"
    return
  -- the rest of `checkAll`, phase by phase, from shared analyses computed here first
  let sh0 := Shared.of r
  let sh ← if kind = 1 then do
      let (Ls, t) ← timed fun _ => r.ledgersP1Par
      say s!"PHASE ledgersP1 {ms t} ({Ls.size} clusters)"
      pure (Shared.ofLedgers r (Thunk.mk fun _ => r.ledgersPar) (Thunk.pure Ls))
    else if kind = 2 then do
      let (o, t) ← timed fun _ => NamedBarriers.analyze r
      say s!"PHASE named {ms t} ok={o.ok}"
      pure { sh0 with named := Thunk.pure o }
    else do
      let (Ls, t) ← timed fun _ => r.ledgersPar
      say s!"PHASE ledgers {ms t} ({Ls.size} clusters)"
      let sh := Shared.ofLedgers r (Thunk.pure Ls)
      if !r.host.hasSync then pure sh else
      let (o, t) ← timed fun _ => Global.analyze r Ls
      say s!"PHASE global {ms t} ok={o.ok} issues={o.issues.size}"
      pure { sh with global := Thunk.pure o }
  for s in k.specsOf kind do
    let (res, t) ← timed fun _ => s.resultWith r sh
    say s!"PHASE result.{s.label} {ms t} holds={res.holds} issues={res.issues.size} notes={res.notes.size}"
  for s in k.specsOf kind do
    let (b, t) ← timed fun _ => s.holdsWith r sh
    say s!"PHASE decide.{s.label} {ms t} {b}"
  say s!"PHASE total {ms ((← IO.monoNanosNow) - tAll)}"

/-- Every issue and note of every spec of a KIR of kind `kind` (the report shows a few per spec),
as `checkAll` computes them (`eval`, then `Shared.of`), one line per issue appended to `path`:
`<spec>\t<issue|note>\t<loc>\t<context>\t<msg>`, then a `PHASE` line per step. For triage. -/
def dumpIssues (k : Kernel) (i : Input) (path : String) (kind : Nat := 0) : IO Unit := do
  let hd ← IO.FS.Handle.mk path .append
  let t0 ← IO.monoNanosNow
  match eval k i with
  | .error e => hd.putStrLn s!"ERROR\t{e.kind.label}\t{e.loc}\t{e.context}\t{e.msg}"
  | .ok r =>
    let t1 ← IO.monoNanosNow
    hd.putStrLn s!"PHASE eval {ms (t1 - t0)}"
    let sh := Shared.of r
    for s in k.specsOf kind do
      let ta ← IO.monoNanosNow
      let res := s.resultWith r sh
      for x in res.issues do
        hd.putStrLn s!"{s.label}\tissue\t{x.loc}\t{x.context}\t{x.msg}"
      for x in res.notes do
        hd.putStrLn s!"{s.label}\tnote\t{x.loc}\t{x.context}\t{x.msg}"
      let tb ← IO.monoNanosNow
      hd.putStrLn s!"PHASE result.{s.label} {ms (tb - ta)} holds={res.holds} issues={res.issues.size} notes={res.notes.size}"
      hd.flush
  hd.flush

end KV.Profile
