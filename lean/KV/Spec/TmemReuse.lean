/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
P1-1: TMEM reuse ordered (references/specs.md, "P1-1"), decided on a P1 KIR per cluster (a
CTA without clusters) whose role traces are all complete.

Model. A cell is (CTA, lane, 32-bit column) of a CTA's tensor-memory allocation. Every
tcgen05.ld/st/cp/mma is one access with exact lane x column rectangles, each read or
written, in the TMEM of one or more CTAs of the cluster (`Ev.tmem`; a cta_group::2 cp/mma
issued by one CTA accesses the TMEM of both CTAs of its pair). Two accesses of different roles
conflict when they access a common CTA's TMEM and some pair of their rectangles overlaps in
lanes and in columns and one of the two writes. Roles are the cluster's (Ledger.lean,
`ClusterTrace`); named barriers are per CTA.

Completion (PTX ISA, tcgen05 memory consistency): an ld/st completes at its role's next
`tcgen05.wait::ld/st` of the matching kind (never, if none follows); a cp/mma completes when
a later `tcgen05.commit` of the same thread has landed. A landing is seen only through a
wait that surely returned and needed it: if a wait needs m arrivals of role r on an
mbarrier slot on which every arrival of r is a commit, then the m-th commit event of r, or
a later one, has landed (m landed arrivals come from m distinct commit events), so every
cp/mma of r before the m-th commit event has completed. Roles that issue cp/mma must be
one warp (the frontend refuses others); a role with more warps gets no landing credit.

Happens-before is the ledger's (Ledger.lean, `ledger` with `precise`, `Run.ledgersP1`),
plus named barriers, over role traces. The frontend gives a role of several warps only
mbarrier events all its warps issue (it splits roles otherwise; specs.md, "P1-1"), but its
warps need not be at one point. Partial credit (`creditArrivals`, Ledger.lean): if a wait
returned and its phase needs m arrivals of a role, some warp of the role issued the arrive
event that holds the m-th arrival (`Hb.done`: that warp passed the role's waits before it,
whose phases therefore completed), and every warp issued it only if the phase needs more
of its arrivals than all warps but one give (`Hb.every`; else every warp issued the
role's previous arrive event). Each warp gives one arrival of an mbarrier arrive in
`sound`, an equal part of it in `lock`, and 32 threads of a named-barrier arrival. An
ld/st completes only within `every`. A warp of a role running so far ahead of another
that its arrivals on a slot stand in for the other's is not modeled. Two closures are
computed before every access:
* `sound`: the waits at their sound phase lower bounds (`WaitRec.minPhase`); a wait that
  returned for phase k needs, of each arriving role, `k x count` minus the other roles'
  arrivals that may precede its return (`SlotLedger.needsWith`, `mayBeforeReturn`). No
  lockstep contents. A PASS rests on this closure only.
* `lock`, the lockstep reading at its most generous: the waits at `WaitRec.lockPhase`
  (presumed skips at c+2, laps the lockstep credit shows) and, on top of `sound`'s needs,
  each role's largest fitting lockstep share of every completed phase
  (`SlotLedger.lockNeeds`). A pair it cannot order is unordered under every fitting share
  assignment.
Named barriers: the j-th bar.sync of a role on id b waits for phase j of b, and phase k
needs `k x count - sum over the other roles o of cap_o(k)` arrivals of a role, with
`cap_o(k) = min(total_o, k x threads_o)` if o only uses bar.sync on b (a thread blocked in
bar.sync arrives at most once per phase) and `total_o` otherwise. A named barrier used with
two counts orders nothing.

Decision. A conflicting cross-role pair (a, b) is ordered under a closure when a completes
before b is issued or b before a. P1-1 holds iff every such pair is ordered under `sound`
and every access is in range. FAIL: a pair not ordered even under `lock` (F1); an access
outside lanes [0, 128) or the allocation's columns (F2); an ld/st outside the lane quarter
of its warp (F3). Undecided (the theorem is refuted, the report says `not decided`): U1 a
pair ordered under `lock` but not under `sound`, when no certain counterexample exists.
Notes (no effect on the verdict): N1 a same-role pair involving a cp/mma that conflicts, is
not a PTX pipelined pair and is not separated by a completion; N2 an intra-role cross-warp
ld/st pair with no bar.sync of the role between the first one's completion and the second;
N3 a wait the ledger finds blocked (happens-before after it is vacuous).
-/
import KV.Spec.Ledger

namespace KV.TmemReuse

structure Rect where
  use : RectUse
  write : Bool
  l0 : Int
  l1 : Int
  c0 : Int
  c1 : Int
  deriving Inhabited, BEq

def Rect.overlaps (a b : Rect) : Bool :=
  max a.l0 b.l0 < min a.l1 b.l1 && max a.c0 b.c0 < min a.c1 b.c1

def Rect.conflicts (a b : Rect) : Bool := (a.write || b.write) && a.overlaps b

def Rect.show (r : Rect) : String :=
  s!"lanes [{r.l0},{r.l1}) x cols [{r.c0},{r.c1})"

def useName : RectUse → String
  | .data => "" | .d => "D " | .a => "A " | .sf => "SF "

structure Access where
  role : Nat
  warp : Int
  pos : Nat
  inst : TmemInst
  /-- Ranks of the CTAs whose TMEM it accesses. -/
  ctas : Array Nat
  rects : Array Rect
  loc : Loc
  site : Loc
  /-- ld/st: position of the role's next tcgen05.wait of the matching kind (the trace size
  if none: it never completes); cp/mma: unused. -/
  done : Nat
  deriving Inhabited

def Access.async (a : Access) : Bool := a.inst == .cp || a.inst == .mma

def instName : TmemInst → String
  | .ld => "tcgen05.ld" | .st => "tcgen05.st" | .cp => "tcgen05.cp" | .mma => "tcgen05.mma"

def Access.where_ (a : Access) : String :=
  if a.site.isEmpty || a.site == a.loc then a.loc else s!"{a.loc} (near {a.site})"

/-- The two accesses use the TMEM of a common CTA. -/
def Access.shareCta (a b : Access) : Bool := a.ctas.any (b.ctas.contains ·)

def Access.describe (h : Host) (cid : Nat × Nat × Nat) (a : Access) : String :=
  let rs := a.rects.toList.map fun r => s!"{if r.write then "writes" else "reads"} {useName r.use}{r.show}"
  let tm := if h.csize ≤ 1 then "" else
    s!" in the TMEM of {", ".intercalate (a.ctas.toList.map fun c => s!"cta {pidStr (h.pidOf cid c)}")}"
  s!"{instName a.inst} at {a.where_} (role {h.gRoleName cid a.role}, warp {a.warp}){tm} {", ".intercalate rs}"

/-- One role's events on a named barrier: (trace position, arrivals so far). -/
structure NamedRole where
  role : Nat
  arrivals : Array (Nat × Int)
  threads : Int
  syncOnly : Bool
  deriving Inhabited

def NamedRole.total (nr : NamedRole) : Int := (nr.arrivals.back?.map (·.2)).getD 0

/-- Arrivals of `o` that can count toward phases 1..k: a role that only uses bar.sync
arrives at most once per phase with its threads. -/
def NamedRole.cap (o : NamedRole) (k : Int) : Int :=
  if o.syncOnly then min o.total (k * o.threads) else o.total

structure NamedSlot where
  id : Int
  count : Int
  /-- Every event on the id uses one count. -/
  consistent : Bool
  roles : Array NamedRole
  deriving Inhabited

/-- Arrivals of `roles[j]` that phase k of the named barrier needs. -/
def NamedSlot.needs (s : NamedSlot) (j : Nat) (k : Int) : Int :=
  k * s.count - s.roles.zipIdx.foldl (fun acc (o, i) => if i == j then acc else acc + o.cap k) 0

/-- A wait in its role's program order: (trace position, named barrier?, slot index in the
ledger or in the named slots, phase lower bound, per entry of the ledger slot's `roles` the
arrivals that may precede its return (`mayBeforeReturn`; empty: none known)). -/
abbrev PWait := Nat × Bool × Nat × Int × Array Int

/-- Per role: the trace prefix that surely happened in some warp of the role (`done`: the
waits there returned), the one that surely happened in every warp (`every`: an ld/st there
completed if its tcgen05.wait is in it too; module doc, "Partial credit") and the position
before which every cp/mma surely completed (`landed`). -/
structure Hb where
  done : Array Nat
  every : Array Nat
  landed : Array Nat
  deriving Inhabited

structure P1Cta where
  /-- The cluster's coordinates (the CTA's program id without clusters). -/
  pid : Nat × Nat × Nat
  /-- The cluster's roles (`ClusterTrace.roles`). -/
  roles : Array RoleTrace
  ledger : Array SlotLedger
  named : Array NamedSlot
  /-- Per ledger slot and entry of its `roles`: every arrival of that role is a commit. -/
  commitOnly : Array (Array Bool)
  /-- Per role, its waits at their sound phase lower bounds (`sound`). -/
  waits : Array (Array PWait)
  warps : Array Nat
  sizes : Array Nat
  /-- Per role, the positions of its bar.sync events. -/
  syncs : Array (Array Nat)
  /-- Role-major, ascending positions. -/
  accesses : Array Access
  sound : Array Hb
  /-- The `lock` closures (waits at `lockPhase`), computed only if some conflicting pair is
  unordered under `sound`. -/
  lock : Thunk (Array Hb)
  /-- Waits the ledger finds blocked in the optimistic progress pass: (role, loc). -/
  blocked : Array (Nat × Loc)
  deriving Inhabited

/-- Grow `h` to a fixpoint (Ledger.saturate with named barriers and commit landing) over
the waits `waits` (`C.waits`, or those at `lockPhase`): a wait inside the prefix some warp
of its role surely passed (`Hb.done`) returned, so each arriving role issued the arrivals
its phase needs (`SlotLedger.needsWith`; with `lockstep` at least `SlotLedger.lockNeeds`),
credited by `creditArrivals` (module doc, "Partial credit"), and on a commit-only slot of a
one-warp role the needed commit landed. `applied` counts per role the waits already
applied. -/
def saturateP1 (C : P1Cta) (waits : Array (Array PWait)) (lockstep : Bool) (h : Hb) (applied : Array Nat) :
    Hb × Array Nat := Id.run do
  let mut done := h.done
  let mut every := h.every
  let mut landed := h.landed
  let mut applied := applied
  for _ in [0:waits.foldl (· + ·.size) 1] do
    let mut changed := false
    for r in [0:waits.size] do
      let ws := waits[r]!
      for i in [applied[r]!:ws.size] do
        let (p, named, s, k, may) := ws[i]!
        if p ≥ done[r]! then break
        applied := applied.modify r (· + 1)
        changed := true
        if named then
          let ns := C.named[s]!
          if ns.consistent then
            for (nr, j) in ns.roles.zipIdx do
              -- every thread of the role arrives: 32 per warp
              let w := C.warps[nr.role]!
              let (one, all) := creditArrivals nr.arrivals w (· / ((max w 1 : Nat) : Int)) (ns.needs j k)
              done := done.modify nr.role (max · one)
              every := every.modify nr.role (max · all)
        else
          let l := C.ledger[s]!
          for (rs, j) in l.roles.zipIdx do
            let m := l.needsWith j k false may
            let m := if lockstep then max m (l.lockNeeds j k) else m
            let w := C.warps[rs.role]!
            let (one, all) := creditArrivals rs.arrivals w (warpShare w lockstep) m
            done := done.modify rs.role (max · one)
            every := every.modify rs.role (max · all)
            -- the needed commit event (one warp) is at position `one - 1`
            if 0 < one && (C.commitOnly[s]!.getD j false) && w == 1 then
              landed := landed.modify rs.role (max · (one - 1))
    if !changed then break
  return ({ done, every, landed }, applied)

/-- The closure before every access (indexed like `C.accesses`), incrementally per role:
the start state before an access at position q of role r is `done[r] = q` (its warp passed
the role's earlier events). -/
def closures (C : P1Cta) (waits : Array (Array PWait)) (lockstep : Bool) : Array Hb := Id.run do
  let n := C.sizes.size
  let mut out : Array Hb := Array.replicate C.accesses.size default
  for r in [0:n] do
    let mut h : Hb := { done := Array.replicate n 0, every := Array.replicate n 0, landed := Array.replicate n 0 }
    let mut applied := Array.replicate n 0
    for (a, i) in C.accesses.zipIdx do
      if a.role == r then
        (h, applied) := saturateP1 C waits lockstep { h with done := h.done.modify r (max · a.pos) } applied
        out := out.set! i h
  return out

/-- `a` surely completed within `h` (an ld/st: in every warp of its role, since a phase
may complete without the issuing warp's arrival). -/
def completes (a : Access) (h : Hb) : Bool :=
  if a.async then a.pos < h.landed[a.role]! else a.done < h.every[a.role]!

/-- The P1 view of cluster `cl` with its barrier ledger `L` (Ledger.lean); `roleWarps` per
kernel role. -/
def build (cl : ClusterTrace) (L : Array SlotLedger) (roleWarps : Array Nat) : P1Cta := Id.run do
  let nr := max 1 roleWarps.size
  let croles := cl.roles nr
  let n := croles.size
  let mut named : Array NamedSlot := #[]
  -- the (CTA rank, id) of each entry of `named`: named barriers are per CTA
  let mut namedKey : Array (Nat × Int) := #[]
  let mut accesses : Array Access := #[]
  let mut nwaits : Array (Array PWait) := Array.replicate n #[]
  let mut plain : Array (Array (Nat × Nat × Nat)) := Array.replicate n #[]
  let mut syncs : Array (Array Nat) := Array.replicate n #[]
  for rt in croles do
    let r := rt.role
    let evs := rt.events
    -- the role's next tcgen05.wait::ld / ::st after each access
    let mut nextLd := evs.size
    let mut nextSt := evs.size
    let mut doneAt : Array Nat := Array.replicate evs.size evs.size
    for k in [0:evs.size] do
      let i := evs.size - 1 - k
      match evs[i]?.map (·.ev) with
      | some (Ev.tmemWait ld) => if ld then nextLd := i else nextSt := i
      | some (Ev.tmem inst ..) => doneAt := doneAt.set! i (if inst == .ld then nextLd else nextSt)
      | _ => pure ()
    let mut syncCount : Array (Int × Nat) := #[]
    for (e, pos) in evs.zipIdx do
      match e.ev with
      | .tmem inst w ctas rects site =>
        let rs := rects.map fun (use, write, l0, l1, c0, c1) => ({ use, write, l0, l1, c0, c1 } : Rect)
        accesses := accesses.push { role := r, warp := w, pos, inst, ctas, rects := rs, loc := e.loc, site,
                                    done := doneAt[pos]! }
      | .namedBar id count sync threads =>
        let key := (r / nr, id)
        let si := (namedKey.findIdx? (· == key)).getD named.size
        if si == named.size then
          named := named.push { id, count, consistent := true, roles := #[] }
          namedKey := namedKey.push key
        named := named.modify si fun ns =>
          let ns := if ns.count != count then { ns with consistent := false } else ns
          match ns.roles.findIdx? (·.role == r) with
          | some j => { ns with roles := ns.roles.modify j fun nr =>
              { nr with arrivals := nr.arrivals.push (pos, nr.total + threads), syncOnly := nr.syncOnly && sync } }
          | none => { ns with roles := ns.roles.push { role := r, arrivals := #[(pos, threads)], threads, syncOnly := sync } }
        if sync then
          syncs := syncs.modify r (·.push pos)
          let j := ((syncCount.find? (·.1 == id)).map (·.2)).getD 0 + 1
          syncCount := (syncCount.filter (·.1 != id)).push (id, j)
          nwaits := nwaits.modify r (·.push (pos, true, si, (j : Int), #[]))
      | .arrive s _ false =>
        if !(plain[r]!.contains (s.cta, s.bar, s.idx)) then plain := plain.modify r (·.push (s.cta, s.bar, s.idx))
      | _ => pure ()
  let sizes := croles.map (·.events.size)
  -- per role, its mbarrier waits at the phase bounds `bound` with the arrivals that may
  -- precede their returns, merged with its named-barrier waits
  let merged (bound : WaitRec → Int) : Array (Array PWait) :=
    let mbar := waitBounds L n bound
    let may := mayBeforeReturn L mbar sizes
    (Array.range n).map fun r =>
      let mine : Array (Array Int) := may[r]!
      ((mbar[r]!.zipIdx.map fun (xi : WaitPos × Nat) =>
          let ((p, s, k), i) := xi
          ((p, false, s, k, mine[i]!) : PWait)) ++ nwaits[r]!).qsort (·.1 < ·.1)
  let commitOnly := L.map fun l => l.roles.map fun rs =>
    !rs.arrivals.isEmpty && !(plain[rs.role]!.contains (l.slot.cta, l.slot.bar, l.slot.idx))
  let blocked := L.foldl (init := #[]) fun acc l => l.roles.foldl (init := acc) fun acc rs =>
    rs.waits.foldl (init := acc) fun acc w => if w.blocked.isSome then acc.push (rs.role, w.loc) else acc
  let warps := (Array.range n).map fun g => roleWarps.getD (g % nr) 1
  let C : P1Cta := { pid := cl.cid, roles := croles, ledger := L, named, commitOnly, waits := merged (·.minPhase),
                     warps, sizes, syncs, accesses, sound := #[], lock := Thunk.pure #[],
                     blocked }
  let C := { C with sound := closures C C.waits false }
  return { C with lock := Thunk.mk fun _ => closures C (merged (·.lockPhase)) true }

/-! ## Pairs -/

/-- PTX pipelined pairs (same warp, `x` issued before `y`): the ISA orders them without a
wait (mma A/SF reads -> mma D, mma A/SF reads -> cp, mma D -> mma D at the same address and
shape, cp -> mma A/SF reads at the same address, and ld/st -> ld/st). -/
def pipelined (a : Access) (x : Rect) (b : Access) (y : Rect) : Bool :=
  match a.inst, b.inst with
  | .mma, .mma => ((x.use == .a || x.use == .sf) && y.use == .d) || (x.use == .d && y.use == .d && x == y)
  | .mma, .cp => x.use == .a || x.use == .sf
  | .cp, .mma => (y.use == .a || y.use == .sf) && x.c0 == y.c0 && x.l0 == y.l0
  | .ld, .ld | .ld, .st | .st, .ld | .st, .st => true
  | _, _ => false

inductive PairKind where
  /-- F1: a cross-role pair not ordered even under the lockstep reading (`lock`). -/
  | fail
  /-- U1: a cross-role pair ordered under `lock` but not under `sound`. -/
  | lockOnly
  /-- N1: a same-role hazard involving a cp/mma. -/
  | sameRole
  /-- N2: an intra-role cross-warp ld/st pair not separated by a bar.sync of the role. -/
  | crossWarp
  deriving BEq, Inhabited

/-- Column group of a column (groups of 32, clamped to `[0, nb)`). -/
def group (nb : Nat) (c : Int) : Nat := min (nb - 1) (if c < 0 then 0 else (c / 32).toNat)

/-- The column group of the first conflicting rectangle pair of `a` and `b` (or of a
non-pipelined one, `samePipe`), where the pair is tested exactly once. -/
def firstConflict (nb : Nat) (a b : Access) (skip : Rect → Rect → Bool) : Option Nat := Id.run do
  let mut best : Option Nat := none
  for x in a.rects do
    for y in b.rects do
      if x.conflicts y && !skip x y then
        let g := group nb (max x.c0 y.c0)
        best := some (match best with | some g' => min g g' | none => g)
  return best

/-- A bar.sync of the role lies strictly between positions `lo` and `hi`. -/
def syncBetween (C : P1Cta) (r : Nat) (lo hi : Nat) : Bool :=
  C.syncs[r]!.any fun p => lo < p && p < hi

/-- The classified conflicting pairs of the CTA (indices into `C.accesses`, first issued
first within a role), each once. -/
def pairs (C : P1Cta) (cols : Nat) : Array (Nat × Nat × PairKind) := Id.run do
  let nb := max 1 ((cols + 31) / 32)
  let mut buckets : Array (Array Nat) := Array.replicate nb #[]
  for (a, i) in C.accesses.zipIdx do
    let mut gs : Array Nat := #[]
    for x in a.rects do
      for g in [group nb x.c0 : group nb (x.c1 - 1) + 1] do
        if !gs.contains g then gs := gs.push g
    for g in gs do
      buckets := buckets.modify g (·.push i)
  let mut out := #[]
  for g in [0:nb] do
    let bs := buckets[g]!
    for u in [0:bs.size] do
      for v in [u + 1:bs.size] do
        let i := bs[u]!
        let j := bs[v]!
        let a := C.accesses[i]!
        let b := C.accesses[j]!
        if !a.shareCta b then
          pure ()
        else if a.role != b.role then
          if firstConflict nb a b (fun _ _ => false) == some g then
            let ord (H : Array Hb) := completes a H[j]! || completes b H[i]!
            if !ord C.sound then out := out.push (i, j, if ord C.lock.get then .lockOnly else .fail)
        else
          -- same role: i is issued before j (role-major, ascending positions)
          let (i, j, a, b) := if a.pos ≤ b.pos then (i, j, a, b) else (j, i, b, a)
          if a.warp != b.warp && !a.async && !b.async then
            if firstConflict nb a b (fun _ _ => false) == some g
                && !(a.done < b.pos && syncBetween C a.role a.done b.pos) then
              out := out.push (i, j, .crossWarp)
          else if a.async || b.async then
            if firstConflict nb a b (pipelined a · b ·) == some g && !completes a C.sound[j]! then
              out := out.push (i, j, .sameRole)
  return out

/-! ## Decision -/

def allocCols (t : Option TmemInfo) : Nat := (t.map (·.cols)).getD 512

/-- F2: every rectangle inside lanes [0, 128) and the allocation's columns. -/
def Access.inRange (cols : Nat) (a : Access) : Bool :=
  a.rects.all fun r => 0 ≤ r.l0 && r.l1 ≤ 128 && r.l0 < r.l1 && 0 ≤ r.c0 && r.c1 ≤ cols && r.c0 < r.c1

/-- F3: an ld/st accesses only the lane quarter of its warp. -/
def Access.inQuarter (a : Access) : Bool :=
  a.async || a.rects.all fun r => 32 * (a.warp % 4) ≤ r.l0 && r.l1 ≤ 32 * (a.warp % 4) + 32

/-- Every access is in range and no pair of `ps` (`pairs C cols`) is F1 or U1. -/
def ctaOkWith (cols : Nat) (C : P1Cta) (ps : Array (Nat × Nat × PairKind)) : Bool :=
  C.accesses.all (fun a => a.inRange cols && a.inQuarter) && ps.all fun (_, _, k) => k != .fail && k != .lockOnly

/-- Every access is in range and every conflicting cross-role pair is ordered under `sound`. -/
def ctaOk (cols : Nat) (C : P1Cta) : Bool := ctaOkWith cols C (pairs C cols)

def p1Ctas (r : Run) (Ls : Ledgers) : Array P1Cta :=
  let warps := r.host.kernel.roles.toArray.map (·.warps)
  (r.clusters.zip Ls).map fun (c, (_, L)) => build c L warps

def cols (r : Run) : Nat := allocCols r.host.kernel.tmem

/-- P1-1 on one run: in every complete CTA every access is in range and every conflicting
cross-role pair is ordered under sound happens-before (over the P1 ledgers). -/
def Holds (r : Run) : Prop := ∀ c ∈ p1Ctas r r.ledgersP1, ctaOk (cols r) c = true

instance (r : Run) : Decidable (Holds r) := by unfold Holds; infer_instance

/-! ## Counterexamples and notes -/

def ctx (h : Host) (C : P1Cta) (a b : Access) : String :=
  if h.csize ≤ 1 then s!"cta {pidStr C.pid} roles {h.roleName a.role} / {h.roleName b.role}"
  else s!"cluster {pidStr C.pid} roles {h.gRoleName C.pid a.role} / {h.gRoleName C.pid b.role}"

def locAt (C : P1Cta) (r p : Nat) : Loc := ((C.roles[r]?.bind (·.events[p]?)).map (·.loc)).getD "?"

/-- Why `a` does not complete before `b` is issued under `H` (the closure before b). -/
def whyNot (h : Host) (C : P1Cta) (a b : Access) (H : Hb) : String :=
  let rn := h.gRoleName C.pid a.role
  if a.async then
    let l := H.landed[a.role]!
    s!"{instName a.inst} completes only when a later commit of {rn} lands, and the latest commit whose landing the waits before {instName b.inst} need is " ++
      (if l == 0 then "none" else s!"at {locAt C a.role l}")
  else
    let d := H.every[a.role]!
    let one := H.done[a.role]!
    (if a.done ≥ (C.roles[a.role]?.map (·.events.size)).getD 0 then s!"{instName a.inst} never completes (no tcgen05.wait::{if a.inst == .ld then "ld" else "st"} follows it in {rn})"
     else s!"{instName a.inst} completes at the tcgen05.wait at {locAt C a.role a.done}") ++
    s!", and the latest {rn} event surely before {instName b.inst} is " ++
      (if d == 0 then "none" else s!"at {locAt C a.role (d - 1)}") ++
      (if one ≤ d then "" else
        s!" in every warp (some warp of {rn} surely reached {locAt C a.role (one - 1)}, but the waits need only part of the arrivals there)")

def pairIssue (h : Host) (C : P1Cta) (i j : Nat) (k : PairKind) (extra : Nat) : Issue :=
  let a := C.accesses[i]!
  let b := C.accesses[j]!
  -- the line is the non-MMA access that writes (else the other one)
  let swap := if a.inst == .mma && b.inst != .mma then true
    else if b.inst == .mma && a.inst != .mma then false
    else !(a.rects.any (·.write) && !b.rects.any (·.write))
  let (pi, qi) := if swap then (j, i) else (i, j)
  let p := C.accesses[pi]!
  let q := C.accesses[qi]!
  let more := if extra = 0 then "" else s!" ({extra} more pair(s) at these two lines in this CTA)"
  let d (x : Access) := x.describe h C.pid
  let msg := match k with
    | .fail => s!"{d p}; {d q}: unordered, even under the lockstep reading. {whyNot h C q p C.lock.get[pi]!}. {whyNot h C p q C.lock.get[qi]!}"
    | .lockOnly => s!"{d p}; {d q}: ordered only under the lockstep contents of the ledger, an assumption (no sound happens-before edge: {whyNot h C q p C.sound[pi]!}; {whyNot h C p q C.sound[qi]!})"
    | .sameRole => s!"{d a}, then {d b}: same-role hazard (not a PTX pipelined pair, and no completion of the first lies between them)"
    | .crossWarp => s!"{d a}, then {d b}: intra-role cross-warp pair ordered only by the lockstep reading of the role (no bar.sync of the role after the first one completes)"
  { loc := p.loc, context := ctx h C a b, msg := msg ++ more }

/-- Pairs of kind `k`, one line per distinct (line, partner line) with a count. -/
def pairIssues (h : Host) (C : P1Cta) (ps : Array (Nat × Nat × PairKind)) (k : PairKind) : Array Issue := Id.run do
  let mut keys : Array (Loc × Loc × Nat × Nat × Nat) := #[]  -- (loc, loc, first pair i j, extra)
  for (i, j, k') in ps do
    if k' == k then
      let a := C.accesses[i]!
      let b := C.accesses[j]!
      let (x, y) := if a.loc ≤ b.loc then (a.loc, b.loc) else (b.loc, a.loc)
      match keys.findIdx? fun (x', y', _, _, _) => x' == x && y' == y with
      | some t => keys := keys.modify t fun (x', y', i', j', e) => (x', y', i', j', e + 1)
      | none => keys := keys.push (x, y, i, j, 0)
  return keys.map fun (_, _, i, j, e) => pairIssue h C i j k e

def rangeIssues (h : Host) (cols : Nat) (C : P1Cta) : Array Issue :=
  C.accesses.filterMap fun a =>
    let cx := if h.csize ≤ 1 then s!"cta {pidStr C.pid} role {h.roleName a.role}"
      else s!"cluster {pidStr C.pid} role {h.gRoleName C.pid a.role}"
    if !a.inRange cols then some { loc := a.loc, context := cx, msg := s!"{a.describe h C.pid}: outside lanes [0,128) x cols [0,{cols}) of the allocation" }
    else if !a.inQuarter then some { loc := a.loc, context := cx, msg := s!"{a.describe h C.pid}: outside lanes [{(32 : Int) * (a.warp % 4)},{(32 : Int) * (a.warp % 4) + 32}), the lane quarter of warp {a.warp}" }
    else none

structure P1Out where
  /-- Every complete CTA passes `ctaOk` (the decision of `Holds` on the same CTAs). -/
  ok : Bool
  issues : Array Issue
  notes : Array Issue
  /-- Some issue is a certain counterexample: F2, F3, or F1 in a CTA whose named barriers
  each have one count. A refuted P1-1 without one is undecided on the input. -/
  certain : Bool
  accesses : Nat
  pairs : Nat
  /-- U1 pairs (ordered only under the lockstep reading). -/
  lockOnly : Nat
  named : Nat
  inconsistent : Array Int
  blocked : Nat

/-- P1-1's verdict lines and notes on run `r` with its P1 ledgers `Ls = r.ledgersP1`. Issue
order: a stopped-trace line, then per CTA the certain counterexamples (F2, F3, and F1 where
every named barrier has one count) or, with an inconsistent named barrier, a `not decided`
line and its F1 lines; then the U1 lines, after a `not decided` line if nothing above is
certain. -/
def analyze (r : Run) (Ls : Ledgers) : P1Out := Id.run do
  let h := r.host
  let mut ok := true
  let mut issues : Array Issue := #[]
  let mut certain := false
  let mut lockLines : Array Issue := #[]
  let mut nlock := 0
  let mut notes : Array Issue := #[]
  let mut nacc := 0
  let mut npairs := 0
  let mut nnamed := 0
  let mut incons : Array Int := #[]
  let mut nblocked := 0
  for C in p1Ctas r Ls do
    let ps := pairs C (cols r)
    let bad := C.named.filter (!·.consistent)
    let fails := pairIssues h C ps .fail
    let ranges := rangeIssues h (cols r) C
    let und : Array Issue := if bad.isEmpty || fails.isEmpty then #[] else
      #[{ loc := "", context := s!"{if h.csize ≤ 1 then "cta" else "cluster"} {pidStr C.pid}",
          msg := s!"not decided on this input: named barrier(s) {(bad.map (·.id)).toList} are used with two counts, so they order nothing; the unordered pairs below may be ordered by them" }]
    let hang : Array Issue := match C.blocked[0]? with
      | some (role, loc) =>
        #[{ loc := loc, context := s!"{if h.csize ≤ 1 then "cta" else "cluster"} {pidStr C.pid} role {h.gRoleName C.pid role}",
            msg := s!"the P1 ledger finds {C.blocked.size} wait(s) that never return in the lockstep model (P0-2): happens-before after them is vacuous, so other P1-1 lines may be their consequences" }]
      | none => #[]
    ok := ok && ctaOkWith (cols r) C ps
    certain := certain || !ranges.isEmpty || (bad.isEmpty && !fails.isEmpty)
    issues := issues ++ ranges ++ und ++ fails
    lockLines := lockLines ++ pairIssues h C ps .lockOnly
    nlock := nlock + (ps.filter (·.2.2 == .lockOnly)).size
    notes := notes ++ pairIssues h C ps .sameRole ++ pairIssues h C ps .crossWarp ++ hang
    nacc := nacc + C.accesses.size
    npairs := npairs + ps.size
    nnamed := nnamed + C.named.size
    incons := incons ++ bad.map (·.id)
    nblocked := nblocked + C.blocked.size
  -- U1: not certain; the input is undecided unless a certain counterexample exists
  if !lockLines.isEmpty then
    let first := lockLines[0]?
    let head : Array Issue := if certain then #[] else
      #[{ loc := (first.map (·.loc)).getD "", context := (first.map (·.context)).getD "",
          msg := s!"not decided on this input: {nlock} conflicting pair(s) of TMEM accesses are ordered only under the ledger's lockstep contents (an assumption), not by sound happens-before; P1-1 is proved only if sound happens-before orders every pair" }]
    issues := issues ++ head ++ lockLines
  -- a stopped role trace leaves P1-1 undecided (as P0-2..P0-5)
  let stops := r.ctas.flatMap fun c => (c.roles.filter (·.stopped)).map fun t => (c.pid, t)
  match stops[0]? with
  | some (pid, t) =>
    let line : Issue :=
      { loc := (t.events.back?.map (·.loc)).getD "", context := h.roleCtx pid t.role,
        msg := s!"not decided on this input: {stops.size} role trace(s) stop at an out-of-bounds metadata load (P0-1), so their later events are unknown; the other lines of this spec are certain counterexamples" }
    issues := #[line] ++ issues
  | none => pure ()
  return { ok, issues, notes, certain, accesses := nacc, pairs := npairs, lockOnly := nlock, named := nnamed,
           inconsistent := incons, blocked := nblocked }

end KV.TmemReuse
