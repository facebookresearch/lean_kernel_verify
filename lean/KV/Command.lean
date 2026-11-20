/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
One evaluation per input. `#lkv_check kernel input s₁ … sₙ` (frontend/emit_lean.py writes one per
input) evaluates `kernel` on `input` once and decides every spec `sᵢ` from that evaluation:

1. it compiles and runs `checkAll [s₁, …, sₙ] 0 kernel input` (one `eval`, then the report and
   the decision `decide (Holds sᵢ kernel input)` of every spec from the same run, computing the
   analyses they share, e.g. the run's barrier ledgers, once for all of them: `Shared`,
   `checkAll_snd`) and prints the report (`Report.render`, as `#eval KV.runCase` prints it);
2. it adds the axiom `(checkAll [s₁, …, sₙ] 0 kernel input).2 = [b₁, …, bₙ]`, the decisions
   as the compiled evaluator computed them. This is the trust of `native_decide`, which in Lean
   4.34 evaluates its decision with `evalConst` and adds the axiom `e = true` (Lean.Meta.Native):
   here the evaluated term is a list of decisions instead of one;
3. per spec it adds a theorem, `KV.Holds sᵢ kernel input` if `bᵢ`, else
   `¬ KV.Holds sᵢ kernel input`, proved from the axiom by `holds_of_verdicts` /
   `not_holds_of_verdicts` (the kernel checks it; nothing is evaluated again), named as the
   per-spec `native_decide` theorems were (`P0_1_bounds__input0_…`);
4. it prints `THEOREM <input> <label> PASS` (resp. `FAIL`) per spec once its theorem
   (resp. its negation) is in the environment: scripts/lkv.py takes the verdict from these lines
   and requires the report's `VERDICT` lines to agree.

`#lkv_check_p1` is the same with the P1 report (`runP1`), `#lkv_check_nb` with the P0-6 report
(`runNb`). Before, every spec had its own
`native_decide` theorem, each evaluating the kernel again (1 + number of specs evaluations per
input, the report's included); now an input costs one evaluation.
-/
import Lean.Elab.Command
import Lean.Compiler.Options
import KV.Check

namespace KV

/-- The decisions of the specs `ss` for kernel `k` on input `i`. -/
def verdicts (ss : List SpecId) (k : Kernel) (i : Input) : List Bool :=
  ss.map fun s => decide (Holds s k i)

/-- One evaluation of `k` on `i`: the report of the specs `ss` (`kind` 0 the P0 report, 1 the P1
report, 2 the P0-6 report) and their decisions, both from the same run and its shared analyses
(`Shared`: barrier ledgers, P1 ledgers, P0-1's, P0-3's and P0-5's decisions, P0-6's and P0-7's
analyses, the P1 pair searches). -/
def checkAll (ss : List SpecId) (kind : Nat) (k : Kernel) (i : Input) : Report × List Bool :=
  match eval k i with
  | .error e => (report kind ss k i (.error e), ss.map fun _ => false)
  | .ok r =>
    let sh := Shared.of r
    (report.reportRun kind ss k i r sh, ss.map (·.holdsWith r sh))

theorem decide_runHolds (s : SpecId) (x : Except EvalError Run) :
    decide (RunHolds s x) = (match x with | .ok r => s.holdsB r | .error _ => false) := by
  cases x with
  | error _ => rfl
  | ok r => exact decide_holds s r

/-- The decisions `checkAll` computes from the shared analyses are the specs' decisions
(`holdsWith_of`). -/
theorem checkAll_snd (ss : List SpecId) (kind : Nat) (k : Kernel) (i : Input) :
    (checkAll ss kind k i).2 = verdicts ss k i := by
  have e : verdicts ss k i = ss.map fun s => match eval k i with
      | .ok r => s.holdsB r | .error _ => false := by
    unfold verdicts; exact List.map_congr_left fun s _ => decide_runHolds s (eval k i)
  rw [e]; unfold checkAll
  cases eval k i with
  | error _ => rfl
  | ok r => exact List.map_congr_left fun s _ => holdsWith_of s r

theorem holds_of_verdicts {ss : List SpecId} {k : Kernel} {i : Input} {bs : List Bool}
    (h : verdicts ss k i = bs) (j : Nat) (s : SpecId) (hs : ss[j]? = some s)
    (hb : bs[j]? = some true) : Holds s k i := by
  subst h
  rw [verdicts, List.getElem?_map, hs] at hb
  exact of_decide_eq_true (Option.some.inj hb)

theorem not_holds_of_verdicts {ss : List SpecId} {k : Kernel} {i : Input} {bs : List Bool}
    (h : verdicts ss k i = bs) (j : Nat) (s : SpecId) (hs : ss[j]? = some s)
    (hb : bs[j]? = some false) : ¬ Holds s k i := by
  subst h
  rw [verdicts, List.getElem?_map, hs] at hb
  exact of_decide_eq_false (Option.some.inj hb)

def SpecId.ctorName : SpecId → String
  | .bounds => "bounds" | .tripCount => "tripCount" | .coverage => "coverage"
  | .barrier => "barrier" | .maskElision => "maskElision" | .namedBarriers => "namedBarriers"
  | .tmemReuse => "tmemReuse" | .storeReuse => "storeReuse" | .gwait => "gwait"

def SpecId.every : List SpecId :=
  [.bounds, .tripCount, .coverage, .barrier, .maskElision, .namedBarriers, .tmemReuse, .storeReuse,
   .gwait]

open Lean Elab Command Meta

def SpecId.expr (s : SpecId) : Lean.Expr := mkConst (`KV.SpecId ++ Name.mkSimple s.ctorName)

def specListExpr (ss : List SpecId) : Lean.Expr :=
  ss.foldr (fun s acc => mkApp3 (mkConst ``List.cons [Level.zero]) (mkConst ``SpecId) s.expr acc)
    (mkApp (mkConst ``List.nil [Level.zero]) (mkConst ``SpecId))

unsafe def evalCheckUnsafe (n : Name) : CoreM (Report × List Bool) :=
  evalConst (Report × List Bool) n

@[implemented_by evalCheckUnsafe]
opaque evalCheck (n : Name) : CoreM (Report × List Bool)

/-- Steps 1-4 of the module doc. -/
def lkvCheck (kind : Nat) (kId iId : Syntax) (specIds : Array Syntax) : CommandElabM Unit := do
  let ss ← specIds.toList.mapM fun s =>
    match SpecId.every.find? (·.ctorName == s.getId.toString) with
    | some x => pure x
    | none => throwErrorAt s "#lkv_check: unknown spec {s.getId}"
  let lines ← liftTermElabM do
    let kName ← realizeGlobalConstNoOverloadWithInfo kId
    let iName ← realizeGlobalConstNoOverloadWithInfo iId
    let kE := mkConst kName
    let iE := mkConst iName
    let ssE := specListExpr ss
    let p1E := mkNatLit kind
    let reportE := mkConst ``Report
    let boolsE := mkApp (mkConst ``List [Level.zero]) (mkConst ``Bool)
    let e := mkApp4 (mkConst ``checkAll) ssE p1E kE iE
    let noAsync (o : Options) : Options := Elab.async.set o false
    -- 1. one evaluation, as `native_decide` runs its decision (Lean.Meta.Native.nativeEqTrue)
    let (rep, bs) ← withoutModifyingEnv do
      let aux ← mkAuxDeclName `_lkv_check
      let decl := Declaration.defnDecl {
        name := aux, levelParams := [], type := mkApp2 (mkConst ``Prod [Level.zero, Level.zero]) reportE boolsE,
        value := e, hints := .abbrev, safety := .safe }
      modifyEnv (markMeta · aux)
      withOptions noAsync do
      withOptions (Compiler.compiler.postponeCompile.set · false) do
      withOptions (Compiler.compiler.relaxedMetaCheck.set · true) do
        addAndCompile decl
      evalCheck aux
    logInfo m!"{rep.render}"
    if bs.length != ss.length then
      throwError "#lkv_check: {bs.length} decisions for {ss.length} specs"
    -- 2. the axiom: the decisions the compiled evaluator computed
    let bsE := toExpr bs
    let axName := iName ++ `_lkv_native_ax
    let sndE := mkApp3 (mkConst ``Prod.snd [Level.zero, Level.zero]) reportE boolsE e
    withOptions noAsync <| addDecl (.axiomDecl {
      name := axName, levelParams := [], isUnsafe := false,
      type := mkApp3 (mkConst ``Eq [Level.one]) boolsE sndE bsE })
    -- 3. the theorems
    let vE := mkApp3 (mkConst ``verdicts) ssE kE iE
    let h0Name := iName ++ `_lkv_verdicts
    let h0 := mkAppN (mkConst ``Eq.trans [Level.one]) #[boolsE, vE, sndE, bsE,
      mkAppN (mkConst ``Eq.symm [Level.one]) #[boolsE, sndE, vE, mkAppN (mkConst ``checkAll_snd) #[ssE, p1E, kE, iE]],
      mkConst axName]
    withOptions noAsync <| addDecl (.thmDecl {
      name := h0Name, levelParams := [], type := mkApp3 (mkConst ``Eq [Level.one]) boolsE vE bsE, value := h0 })
    let ns ← getCurrNamespace
    let optSpec := mkApp (mkConst ``Option [Level.zero]) (mkConst ``SpecId)
    let optBool := mkApp (mkConst ``Option [Level.zero]) (mkConst ``Bool)
    let mut lines := #[]
    for (s, j) in ss.zipIdx do
      let b := bs[j]!
      let label := s.label
      let name := ns ++ Name.mkSimple s!"{label.replace "-" "_"}_{s.ctorName}__{iId.getId.toString}"
      let holds := mkApp3 (mkConst ``Holds) s.expr kE iE
      let someS := mkApp2 (mkConst ``Option.some [Level.zero]) (mkConst ``SpecId) s.expr
      let someB := mkApp2 (mkConst ``Option.some [Level.zero]) (mkConst ``Bool) (toExpr b)
      let lemma := if b then ``holds_of_verdicts else ``not_holds_of_verdicts
      let value := mkAppN (mkConst lemma) #[ssE, kE, iE, bsE, mkConst h0Name, mkNatLit j, s.expr,
        mkApp2 (mkConst ``Eq.refl [Level.one]) optSpec someS, mkApp2 (mkConst ``Eq.refl [Level.one]) optBool someB]
      withOptions noAsync <| addDecl (.thmDecl {
        name, levelParams := [], type := if b then holds else mkNot holds, value })
      -- 4.
      lines := lines.push s!"THEOREM {rep.input.name} {label} {if b then "PASS" else "FAIL"}"
    return lines
  logInfo m!"{"\n".intercalate lines.toList}"

syntax (name := lkvCheckCmd) "#lkv_check " ident ident ident* : command
syntax (name := lkvCheckP1Cmd) "#lkv_check_p1 " ident ident ident* : command
syntax (name := lkvCheckNbCmd) "#lkv_check_nb " ident ident ident* : command

@[command_elab lkvCheckCmd] def elabLkvCheck : CommandElab := fun stx =>
  lkvCheck 0 stx[1] stx[2] stx[3].getArgs

@[command_elab lkvCheckP1Cmd] def elabLkvCheckP1 : CommandElab := fun stx =>
  lkvCheck 1 stx[1] stx[2] stx[3].getArgs

@[command_elab lkvCheckNbCmd] def elabLkvCheckNb : CommandElab := fun stx =>
  lkvCheck 2 stx[1] stx[2] stx[3].getArgs

end KV
