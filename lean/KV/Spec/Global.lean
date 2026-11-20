/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
P0-7, global waits reached and co-residency (references/specs.md, "P0-7"; KIR change 18),
decided per input on one launch of a kernel with sync tensors. Roles are not interleaved: each
role instance (CTA, role) is evaluated on its own (Eval.lean), and the argument across instances
is made here.

A take (`gwait`: a spin until a 64-bit sync word carries a tag, its high half) returns once a
store of its tag to its word is reached; the word starts at 0, so a take of tag 0 needs none.
Optimistic progress (`G.progress`): every role instance runs as far as its waits allow when
every store and arrival takes effect when its event is reached: an mbarrier wait passes once the
arrivals reached on its slot cover its phase (`WaitRec.assumedPhase`, the cluster ledger of
Ledger.lean), a named barrier's j-th sync once `j x count` arrivals are reached, a take once a
store of its tag is. FAIL: a take still blocked when no instance can move (a hang), a take that
no store of the launch satisfies, a take that needs a store of another CTA of the launch on a
grid larger than its co-resident capacity (CUDA guarantees no dispatch order; the CTAs of a
cluster are co-scheduled). UNDECIDED: a take not reached because an mbarrier or named-barrier
wait blocks first (P0-2), cross-CTA takes without a capacity. The verdict is relative to this
algorithm, as P0-2 and P0-4 are relative to the barrier ledger.
-/
import KV.Spec.Ledger

namespace KV.Global

/-- A store of instance `g` (CTA linear index x roles + role) at trace position `pos`. -/
structure GSig where
  g : Nat
  pos : Nat
  w : Word
  v : Int
  deriving Inhabited

/-- A take of tag `tag` of word `w`. -/
structure GWait where
  g : Nat
  pos : Nat
  w : Word
  tag : Int
  deriving Inhabited

/-- One role's events on a named barrier of its CTA: (position, arrivals so far). -/
structure NamedArr where
  g : Nat
  arrivals : Array (Nat × Int)
  deriving Inhabited

structure NamedBar where
  id : Int
  count : Int
  roles : Array NamedArr
  deriving Inhabited

/-- A cluster's CTAs (linear indices by rank), ledger and waits at the phase bounds optimistic
progress uses (`assumedPhase`), by cluster-local role. -/
structure ClusterInfo where
  ctas : Array Nat
  L : Array SlotLedger
  assumed : Array (Array WaitPos)
  deriving Inhabited

/-- The analysis context of one run (`build`). -/
structure G where
  run : Run
  nr : Nat
  n : Nat
  sizes : Array Nat
  sigs : Array GSig
  waits : Array GWait
  /-- Named barriers per (CTA, id) and per instance its syncs: (position, barrier, phase). -/
  named : Array NamedBar
  instNamed : Array (Array (Nat × Nat × Int))
  clusters : Array ClusterInfo
  /-- Per CTA linear index, its cluster in `clusters` (none: its cluster has no ledger). -/
  clusterOf : Array (Option Nat)

def G.ctaOf (G : G) (g : Nat) : Nat := g / G.nr

def G.instName (G : G) (g : Nat) : String :=
  match G.run.ctas[G.ctaOf g]? with
  | some c => G.run.host.roleCtx c.pid (g % G.nr)
  | none => "?"

def G.locAt (G : G) (g p : Nat) : Loc :=
  match G.run.ctas[G.ctaOf g]? with
  | some c => ((c.roles[g % G.nr]?.bind (·.events[p]?)).map (·.loc)).getD ""
  | none => ""

def G.wordName (G : G) (w : Word) : String := s!"{G.run.host.tensorName w.t}[{w.e}]"

/-- The tag a store of `v` to a 64-bit word carries (its high half). -/
def tagOf (v : Int) : Int := ((BitVec.ofInt 64 v).toNat : Int) / 2 ^ 32

/-- The stores that satisfy take `W` (indices into `G.sigs`). -/
def G.satisfiers (G : G) (W : GWait) : Array Nat :=
  (Array.range G.sigs.size).filter fun i => G.sigs[i]!.w == W.w && tagOf G.sigs[i]!.v == W.tag

/-- The analysis context of run `r` with its barrier ledgers `Ls` (`Run.ledgers`). -/
def build (r : Run) (Ls : Ledgers) : G := Id.run do
  let h := r.host
  let nr := max 1 h.kernel.roles.length
  let n := r.ctas.size * nr
  let mut sizes := Array.replicate n 0
  let mut sigs : Array GSig := #[]
  let mut waits : Array GWait := #[]
  let mut named : Array NamedBar := #[]
  let mut namedKey : Std.HashMap (Nat × Int) Nat := {}
  let mut instNamed := Array.replicate n #[]
  for (c, ci) in r.ctas.zipIdx do
    for t in c.roles do
      let g := ci * nr + t.role
      sizes := sizes.set! g t.events.size
      let mut syncCount : Std.HashMap Int Nat := {}
      for (e, pos) in t.events.zipIdx do
        match e.ev with
        | .gsig w v => sigs := sigs.push { g, pos, w, v }
        | .gwait w tag => waits := waits.push { g, pos, w, tag }
        | .namedBar id count sync threads =>
          let bi := namedKey.getD (ci, id) named.size
          if bi == named.size then
            named := named.push { id, count, roles := #[] }
            namedKey := namedKey.insert (ci, id) bi
          let b := named[bi]!
          named := named.set! bi <| match b.roles.findIdx? (·.g == g) with
            | some j => { b with roles := b.roles.modify j fun a =>
                { a with arrivals := a.arrivals.push (pos, ((a.arrivals.back?.map (·.2)).getD 0) + threads) } }
            | none => { b with roles := b.roles.push { g, arrivals := #[(pos, threads)] } }
          if sync then
            let j := syncCount.getD id 0 + 1
            syncCount := syncCount.insert id j
            instNamed := instNamed.modify g (·.push (pos, bi, (j : Int)))
        | _ => pure ()
  let mut clusters : Array ClusterInfo := #[]
  let mut clusterOf : Array (Option Nat) := Array.replicate r.ctas.size none
  let pidIdx : Std.HashMap (Nat × Nat × Nat) Nat := r.ctas.zipIdx.foldl (init := {}) fun m (c, i) => m.insert c.pid i
  for (cid, L) in Ls do
    let members := (Array.range h.csize).filterMap fun rk => pidIdx[h.pidOf cid rk]?
    for ci in members do clusterOf := clusterOf.set! ci (some clusters.size)
    clusters := clusters.push { ctas := members, L, assumed := waitBounds L (members.size * nr) (·.assumedPhase) }
  return { run := r, nr, n, sizes, sigs, waits, named, instNamed, clusters, clusterOf }

/-- A blocking event of an instance for optimistic progress: (position, kind, resource, need).
kind 0: mbarrier wait (cluster, ledger slot, arrivals); 1: named-barrier sync (barrier, phase
arrivals); 2: take (`a`: its index in `G.waits`). -/
structure Block where
  pos : Nat
  kind : Nat
  a : Nat
  b : Nat
  need : Int
  deriving Inhabited

structure Progress where
  /-- Per instance, its position when nothing can move. -/
  cur : Array Nat
  /-- Per instance, the blocking event it stopped at (none: it ran to its end). -/
  stuck : Array (Option Block)
  /-- Nothing could move any more (false: the round bound ran out first; never with the bound
  below, which every round that moves lowers). -/
  fixpoint : Bool := true

/-- Optimistic progress (module doc). -/
def G.progress (G : G) : Progress := Id.run do
  let n := G.n
  let mut blocks : Array (Array Block) := Array.replicate n #[]
  for (c, ci) in G.clusters.zipIdx do
    for (ws, lr) in c.assumed.zipIdx do
      let g := c.ctas[lr / G.nr]! * G.nr + lr % G.nr
      for (p, s, k) in ws do
        blocks := blocks.modify g (·.push { pos := p, kind := 0, a := ci, b := s, need := k * c.L[s]!.count })
  for g in [0:n] do
    for (p, bi, k) in G.instNamed[g]! do
      blocks := blocks.modify g (·.push { pos := p, kind := 1, a := bi, b := 0, need := k * G.named[bi]!.count })
  for (W, wi) in G.waits.zipIdx do
    blocks := blocks.modify W.g (·.push { pos := W.pos, kind := 2, a := wi, b := 0, need := 0 })
  blocks := blocks.map fun bs => bs.qsort (·.pos < ·.pos)
  -- arrivals per instance: (position, kind 0 mbarrier slot (cluster, ledger slot) / 1 named
  -- barrier, resources, amount)
  let slotOf : Array (Std.HashMap (Nat × Nat × Nat) Nat) := G.clusters.map fun c =>
    c.L.zipIdx.foldl (init := {}) fun m (l, i) => m.insert (l.slot.cta, l.slot.bar, l.slot.idx) i
  let mut arrivals : Array (Array (Nat × Nat × Nat × Nat × Int)) := Array.replicate n #[]
  for (c, ci) in G.run.ctas.zipIdx do
    for t in c.roles do
      let g := ci * G.nr + t.role
      for (e, pos) in t.events.zipIdx do
        if let .arrive s k _ := e.ev then
          if let some cl := G.clusterOf[ci]! then
            if let some si := slotOf[cl]![(s.cta, s.bar, s.idx)]? then
              arrivals := arrivals.modify g (·.push (pos, 0, cl, si, k))
  for (b, bi) in G.named.zipIdx do
    for a in b.roles do
      let mut prev : Int := 0
      for (p, tot) in a.arrivals do
        arrivals := arrivals.modify a.g (·.push (p, 1, bi, 0, tot - prev))
        prev := tot
  arrivals := arrivals.map fun xs => xs.qsort (·.1 < ·.1)
  let sat := G.waits.map G.satisfiers
  let mut cur : Array Nat := Array.replicate n 0
  let mut nextBlock : Array Nat := Array.replicate n 0
  let mut nextArr : Array Nat := Array.replicate n 0
  let mut slotArr : Array (Array Int) := G.clusters.map fun c => Array.replicate c.L.size 0
  let mut namedArr : Array Int := Array.replicate G.named.size 0
  -- a round that moves passes a block, applies an arrival or moves `cur` to a new limit (one
  -- per block and one past the last per instance): at most this many rounds move
  let bound := blocks.foldl (· + ·.size) 0 + arrivals.foldl (· + ·.size) 0 + 2 * n + 2
  let mut fixpoint := false
  for _ in [0:bound] do
    let mut moved := false
    for g in [0:n] do
      let mut go := true
      while go do
        let bs := blocks[g]!
        let limit := if nextBlock[g]! < bs.size then bs[nextBlock[g]!]!.pos else G.sizes[g]!
        -- apply the arrivals before `limit` (a named sync arrives before it waits)
        let xs := arrivals[g]!
        let selfFirst := nextBlock[g]! < bs.size && bs[nextBlock[g]!]!.kind == 1
        let mut ai := nextArr[g]!
        while ai < xs.size && (xs[ai]!.1 < limit || (xs[ai]!.1 == limit && selfFirst)) do
          let (_, k, r1, r2, amt) := xs[ai]!
          if k == 0 then slotArr := slotArr.modify r1 (·.modify r2 (· + amt))
          else namedArr := namedArr.modify r1 (· + amt)
          ai := ai + 1
          moved := true
        nextArr := nextArr.set! g ai
        if cur[g]! < limit then
          cur := cur.set! g limit
          moved := true
        if nextBlock[g]! ≥ bs.size then
          go := false
        else
          let b := bs[nextBlock[g]!]!
          let ok : Bool := match b.kind with
            | 0 => slotArr[b.a]![b.b]! ≥ b.need
            | 1 => namedArr[b.a]! ≥ b.need
            | _ => G.waits[b.a]!.tag == 0 || sat[b.a]!.any fun i => let x := G.sigs[i]!; cur[x.g]! > x.pos
          if ok then
            nextBlock := nextBlock.modify g (· + 1)
            cur := cur.set! g (b.pos + 1)
            moved := true
          else go := false
    if !moved then
      fixpoint := true
      break
  let stuck : Array (Option Block) := (Array.range n).map fun (g : Nat) =>
    if nextBlock[g]! < blocks[g]!.size then some blocks[g]![nextBlock[g]!]! else none
  return { cur, stuck, fixpoint }

structure Outcome where
  ok : Bool
  issues : Array Issue
  certain : Bool
  deriving Inhabited

/-- An outcome from its certain counterexamples and undecided lines. -/
def Outcome.of (fails undec : Array Issue) : Outcome :=
  { ok := fails.isEmpty && undec.isEmpty, issues := undec ++ fails, certain := !fails.isEmpty }

/-- `o` not decided (`why`): its counterexamples become undecided lines. -/
def Outcome.undecided (o : Outcome) (why : String) : Outcome :=
  { ok := false, certain := false,
    issues := o.issues.map (fun i => if i.msg.startsWith "not decided" then i else { i with msg := s!"not decided ({why}): {i.msg}" })
      |>.push { loc := "", context := "run", msg := s!"not decided on this input: {why}" } }

def issueAt (G : G) (g p : Nat) (msg : String) : Issue :=
  { loc := G.locAt g p, context := G.instName g, msg }

/-- G1 (module doc). -/
def g1 (G : G) (P : Progress) : Outcome := Id.run do
  let h := G.run.host
  let mut fails := #[]
  let mut undec := #[]
  let mut anyTake := false
  for g in [0:G.n] do
    if let some b := P.stuck[g]! then
      if b.kind == 2 then
        anyTake := true
        let W := G.waits[b.a]!
        let n := (G.satisfiers W).size
        fails := fails.push (issueAt G g b.pos <|
          s!"global wait never returns (hang): tag {W.tag} on the tagged word {G.wordName W.w}: " ++
          (if n == 0 then "no store of the launch carries it"
           else s!"{n} store(s) of the launch carry it, none reached when no role instance can move"))
  -- takes not reached because an mbarrier or named-barrier wait blocks first
  if !anyTake then
    for g in [0:G.n] do
      if let some b := P.stuck[g]! then
        if G.waits.any fun W => W.g == g && W.pos ≥ b.pos then
          undec := undec.push (issueAt G g b.pos s!"not decided: global wait(s) of this role instance are not reached: it blocks first at a{if b.kind == 0 then "n mbarrier" else " named-barrier"} wait (P0-2)")
  -- co-residency: a take whose stores are all made by other CTAs (outside its cluster) needs
  -- those CTAs resident while it spins
  let coSched (g g' : Nat) : Bool :=
    G.ctaOf g == G.ctaOf g' ||
      (h.csize > 1 && (G.clusterOf[G.ctaOf g]?).join.isSome &&
        (G.clusterOf[G.ctaOf g]?).join == (G.clusterOf[G.ctaOf g']?).join)
  let cross := G.waits.filter fun W => W.tag != 0 && !((G.satisfiers W).any fun i => coSched G.sigs[i]!.g W.g)
  if let some W := cross[0]? then
    let ctas := G.run.ctas.size
    let clusters := ctas / max 1 h.csize
    match h.input.capacity with
    | none => undec := undec.push (issueAt G W.g W.pos s!"not decided: {cross.size} global wait(s) need stores of other CTAs of the launch (tag {W.tag} on {G.wordName W.w}), and the input gives no co-resident capacity")
    | some (cap, ccap) =>
      if ctas > cap then
        fails := fails.push (issueAt G W.g W.pos s!"co-residency: {cross.size} global wait(s) need stores of other CTAs of the launch (tag {W.tag} on {G.wordName W.w}), but the grid has {ctas} CTAs and only {cap} can be resident at once: waiting CTAs may hold every slot while a CTA they wait for is never scheduled (CUDA guarantees no dispatch order)")
      else if h.csize > 1 then
        match ccap with
        | some k =>
          if clusters > k then
            fails := fails.push (issueAt G W.g W.pos s!"co-residency: {cross.size} global wait(s) need stores of other CTAs of the launch, but the grid has {clusters} clusters of {h.csize} and only {k} can be resident at once")
        | none => undec := undec.push (issueAt G W.g W.pos s!"not decided: cross-CTA waits in a cluster launch, and the input gives no co-resident cluster count")
  return Outcome.of fails undec

/-- P0-7 on run `r` with its barrier ledgers `Ls` (module doc). On a run that hangs at a take
(`Run.hangs`) its certain hangs stand: it never holds there (the blocked roles' later events
are not evaluated), so without one it is not decided; a run with another stopped trace decides
nothing. -/
def analyze (r : Run) (Ls : Ledgers) : Outcome :=
  if !r.hangs && !r.complete then
    { ok := false, certain := false, issues := #[{ loc := "", context := "run", msg := "not decided on this input: some role trace stops (P0-1)" }] }
  else
    let G := build r Ls
    let P := G.progress
    let o := g1 G P
    -- the progress of mbarrier waits rests on the converged ledger (`assumedPhase`)
    let o := if !Ls.all (·.2.all (·.converged)) then o.undecided "a barrier ledger of the run does not converge (P0-2)"
      else if !P.fixpoint then o.undecided "the progress analysis reached no fixpoint"
      else o
    if !r.hangs then o
    else if o.certain then { o with ok := false }
    else { o with ok := false, issues := o.issues.push { loc := "", context := "run", msg := "not decided on this input: a role is blocked at a take of a tagged word, but no global wait is found stuck" } }

/-- The report's summary of the run's global synchronization. -/
def summary (r : Run) : String := Id.run do
  let mut stores := 0
  let mut takes := 0
  let mut claims := 0
  for c in r.ctas do
    for t in c.roles do
      for e in t.events do
        match e.ev with
        | .gsig .. => stores := stores + 1
        | .gwait .. => takes := takes + 1
        | .gclaim .. => claims := claims + 1
        | _ => pure ()
  return s!"{stores} store(s), {takes} take(s), {claims} claim(s)"

end KV.Global

namespace KV

/-- P0-7 holds on `r` (`Global.analyze`). Not inlined: `r.ledgers` stays inside this spec's
decision. -/
@[noinline] def Global.Holds (r : Run) : Bool := r.complete && (Global.analyze r r.ledgers).ok

end KV
