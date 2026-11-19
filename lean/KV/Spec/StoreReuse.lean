/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
P1-2: async store sources reuse ordered (references/specs.md, "P1-2"), decided on a P1 KIR
whose `p1` names it, per cluster (a CTA without clusters) whose role traces are all complete.

Model (PTX ISA, "Completion Mechanisms for Asynchronous Copy Operations",
"cp.async.bulk.commit_group", "cp.async.bulk.wait_group"). Once a bulk asynchronous copy is
initiated, modifying its source or its tensor descriptor before it completes is undefined
behavior. A bulk store S (a TMA tensor store or reduce, or a non-tensor bulk store or reduce
from shared memory) is issued by one thread t and joins t's open bulk async-group, which t's
next cp.async.bulk.commit_group closes. t's cp.async.bulk.wait_group{.read} N returns once all
but the N most recent of t's committed groups completed (a thread's groups complete in commit
order); with .read, once their operations completed reading the tensormap and their source
locations, which is what P1-2 needs. So S completes at the first wait W of t after S that
follows at least N + 1 of t's commits after S (`Store.done`), or never. The frontend checks
that every issuing thread of a role's bulk stores executes each commit and wait event of the
role, so t's are the role's.

Conflicts. A write X conflicts with S if it writes shared memory of S's CTA overlapping S's
source bytes (F1), or the 128-byte tensormap S reads (F2). The pair is ordered if S completes
before X (W happens before X in every thread executing X) or X happens before S is issued.

Happens-before over the cluster's role traces (`Hb`): per role the trace prefix some thread
surely passed (`done`) and the one every thread surely passed (`all`). The frontend gives each
mbarrier wait and named barrier of a P1-2 role to every thread of the role, so a thread that
passed a position passed the waits before it, and they returned. A wait that returned for
phase k (the P1 ledger's sound phase bounds, `Run.ledgersP1`) needs, of each arriving role,
`k x count` minus the other roles' arrivals that may precede its return
(`SlotLedger.needsWith`); some thread of that role issued the arrive event holding the last
needed arrival (no thread issued more than the events before it otherwise), which raises its
`done`. A named barrier whose participants all use bar.sync with one count, their threads
summing to it or its count the CTA's thread count (KIR `cta_threads`: every thread of the CTA
takes part, idle warps the KIR has no role for included) (`exact`), completes its k-th phase
with exactly the k-th bar.sync of every
participating thread (a thread blocks until the phase of its arrival completes), so a bar.sync
that returned raises every participant's `all` to its k-th bar.sync. Another named barrier
orders nothing in the sound reading: threads outside the roles may take part (the idle warps
of a Triton warp-specialized kernel join its start and end barrier), so what a phase needs of
a role is unknown. An mbarrier arrival never raises `all` (the KIR does not say which threads
issue an arrive). Within a role,
t's program order orders W before X if X is t's own (`Who` equal: the same lane, or both the
elected lane of the warp), and X before S if X is t's own or a warp-synchronizing instruction
(`.sync.aligned`) of t's warp, which t leaves only after every lane executed it.

Lock reading (FAIL vs undecided): waits at `lockPhase` and needs of at least the largest
fitting lockstep share (`SlotLedger.lockNeeds`), as P1-1, a named barrier that is not exact
needing of each role what the other roles' arrivals leave of its count (`NamedSlot.needs`),
bytes carry order (a wait that returned for phase k saw the bytes of phases 1..k land, so the
loads that brought them were issued before it returned: each role's loads on the slot up to
its lockstep share of those bytes, its load bytes spread evenly over the slot's completed
phases, `loadsCoverLock`), and every credited arrival or load taken as every thread's of its
role (`all`). A pair it cannot order is unordered under every fitting share assignment,
whatever threads issue the arrives, with each role's loads bringing its share of each
phase's bytes.

Decision. P1-2 holds iff every conflicting pair is ordered in the sound reading. FAIL (F1, F2):
a pair unordered even in the lock reading that no same-thread order the KIR cannot pin down
might order: not when X is a fixed lane of t's warp and t its elected lane (or the reverse) and
t's program order would order them, nor when X is another fixed lane of t's warp that would be
ordered were it t's own (a `bar.warp.sync`, which the KIR drops, orders the memory accesses of
the lanes taking part), nor when X is a warp-synchronizing write of t's warp after W (U2); F2
also a tensormap that a store of one cluster reads and another cluster writes (no
happens-before across clusters). Undecided: U1 a pair ordered only in the lock reading; U2;
U3 an asynchronous write (a load's landing) issued before the store in the sound reading, or
in the lock reading (its landing may then complete before the store reads), whose landing is
not known to precede the store's issue (below); U4 a pair unordered even in the lock reading
(also a tensormap pair of two clusters) whose store or write follows a wait the P1 ledger
finds blocked in its role (`firstBlocked`: in the lockstep model that event never runs, so the
pair may never run); an F pair in a
cluster with a named barrier used with two counts or whose ledger does not converge; a
stopped role trace.

Landings (`landed`). A load's landing X issued before S is ordered before S if the bytes of
its mbarrier slot (in X's CTA, or for a cta_group::2 load in its pair leader's) show it
complete before S is issued: X's load t (the role's next
transaction event on X's slot, only global accesses between) is among the loads that may
land before the return of a wait W on that slot that surely returned before S is issued
(`returnCuts`: positions from which every event surely follows W's return), and those loads
bring no more bytes than the expect_tx bytes phases 1..k surely hold, k W's phase bound
(`bytesIn`: each role's expect_tx before the last of its arrivals those phases surely hold,
`k x count` minus what the other roles can contribute, a role contributing no arrival after
its own wait for phase k or later). Phases 1..k completed when W returned, each with as
many bytes landed as expected, so at least that many bytes landed before W returned, all
of them from those loads: every byte of them landed, X's included (however a copy splits
its complete_tx). Otherwise the landing may follow the store's read (a load running ahead
of its phase, bytes the ledger cannot place). Notes: N1 a bulk store no wait of its issuer completes before
its role ends; N2 waits the P1 ledger finds blocked (happens-before after them is vacuous).
-/
import KV.Spec.TmemReuse

namespace KV.StoreReuse

open TmemReuse (NamedRole NamedSlot PWait)

structure Store where
  role : Nat
  pos : Nat
  who : Who
  /-- Rank of the issuing CTA in its cluster (whose shared memory it reads). -/
  cta : Nat
  kind : BulkKind
  sym : String
  lo : Int
  hi : Int
  tmap : Option (String × Int)
  loc : Loc
  /-- Trace position of the wait of its issuer that completes it (the trace size: none). -/
  done : Nat
  deriving Inhabited

structure Write where
  role : Nat
  pos : Nat
  who : Who
  /-- Every lane of `who.warp` in one warp-synchronizing instruction. -/
  sync : Bool
  /-- The landing of an asynchronous copy issued at `pos`: it may land later. -/
  async : Bool := false
  /-- The mbarrier slot the landing's copy completes its bytes on, if known. -/
  slot : Option Slot := none
  /-- A tensormap write (F2: `sym` is the global buffer), else a shared-memory one (F1). -/
  tmap : Bool
  /-- Rank of the CTA whose shared memory it writes (for a tensormap: the issuing CTA). -/
  cta : Nat
  sym : String
  ranges : Array (Int × Int)
  loc : Loc
  deriving Inhabited

def kindName : BulkKind → String
  | .tensor => "TMA store" | .tensor_reduce => "TMA reduce" | .bulk => "bulk store"
  | .bulk_reduce => "bulk reduce"

def whoName (w : Who) : String :=
  match w.lane with
  | some l => s!"lane {l} of warp {w.warp}"
  | none => s!"the elected lane of warp {w.warp}"

/-- Per role, the trace prefixes that surely happened before a point: in some thread of the
role (`done`), in every thread (`all`). -/
structure Hb where
  done : Array Nat
  all : Array Nat
  deriving Inhabited

structure Ctx where
  pid : Nat × Nat × Nat
  roles : Array RoleTrace
  ledger : Array SlotLedger
  named : Array NamedSlot
  /-- Per named slot, `NamedSlot.exact`: bar.sync only, one count, the participating
  threads summing to it or the count the CTA's (`cta_threads`). -/
  exact : Array Bool
  /-- Per role, its waits at their sound phase lower bounds, merged with its bar.syncs. -/
  waits : Array (Array PWait)
  /-- Per role, its mbarrier waits by ledger slot at their sound phase lower bounds. -/
  ws : WaitsBySlot
  sizes : Array Nat
  stores : Array Store
  writes : Array Write
  /-- Waits the ledger finds blocked: (role, loc). -/
  blocked : Array (Nat × Loc)
  /-- Per role, the trace position of its first wait the ledger finds blocked (its trace
  size if none): the role's later events do not run in the lockstep model. -/
  firstBlocked : Array Nat
  /-- Per role, the closure before its point after k of its waits, k = 0, 1, ...
  (sound reading; computed only for roles with stores or writes). -/
  sound : Array (Array Hb)
  /-- The same in the lock reading (computed only if some pair is unordered soundly). -/
  lock : Thunk (Array (Array Hb))
  deriving Inhabited

/-- Arrivals of `rs` that can count toward phases 1..k of its slot: those before its first
wait on the slot whose sound phase bound is at least k (it returns only after phase k
completed, and so do the role's events after it), else all. -/
def contrib (rs : RoleSlot) (k : Int) : Int :=
  match rs.waits.find? (·.minPhase ≥ k) with
  | some w => rs.arrivedBefore w.pos
  | none => rs.total

/-- Expect_tx bytes that phases 1..k of the slot surely hold: each role's expect_tx issued
before the last of its arrivals those phases surely hold, which are `k x count` minus what
the other roles can contribute to them (`contrib`), or `SlotLedger.needs` if larger (an
expect_tx applies to the phase current at its issue, at most its role's next arrival's). -/
def bytesIn (L : SlotLedger) (k : Int) : Int :=
  L.roles.zipIdx.foldl (init := 0) fun acc (rs, i) =>
    let others := L.roles.zipIdx.foldl (init := 0) fun n (o, j) => if j == i then n else n + contrib o k
    let need := max (k * L.count - others) (L.needs i k false)
    rs.tx.foldl (init := acc) fun acc x => if !x.load && x.arrivedBefore < need then acc + x.bytes else acc

/-- Lock reading, bytes carry order (module doc): the trace position after the load of `rs`
on slot `L` that completes its lockstep share of the bytes of phases 1..k, its load bytes
spread evenly over the slot's completed phases (after its last load if the share needs them
all; 0 if it has no load or no phase completes). -/
def loadsCoverLock (L : SlotLedger) (rs : RoleSlot) (k : Int) : Nat := Id.run do
  let K := L.completions
  if K ≤ 0 || k ≤ 0 then return 0
  let total : Int := rs.tx.foldl (fun n x => if x.load then n + x.bytes else n) 0
  if total ≤ 0 then return 0
  let kk := min k K
  let mut acc : Int := 0
  let mut last := 0
  for x in rs.tx do
    if x.load then
      acc := acc + x.bytes
      last := x.pos + 1
      if acc * K ≥ kk * total then return last
  return last

/-- Grow `h` to a fixpoint over the waits inside its `done` prefixes (module doc). -/
def saturate (named : Array NamedSlot) (exact : Array Bool) (ledger : Array SlotLedger)
    (waits : Array (Array PWait)) (lockstep : Bool) (h : Hb) (applied : Array Nat) : Hb × Array Nat := Id.run do
  let mut done := h.done
  let mut all := h.all
  let mut applied := applied
  for _ in [0:waits.foldl (· + ·.size) 1] do
    let mut changed := false
    for r in [0:waits.size] do
      let ws := waits[r]!
      for i in [applied[r]!:ws.size] do
        let (p, isNamed, s, k, may) := ws[i]!
        if p ≥ done[r]! then break
        applied := applied.modify r (· + 1)
        changed := true
        if isNamed then
          let ns := named[s]!
          if ns.consistent then
            for (nr, j) in ns.roles.zipIdx do
              if exact[s]! then
                -- phase k is the k-th bar.sync of every participating thread
                match nr.arrivals[(k - 1).toNat]? with
                | some (q, _) =>
                  done := done.modify nr.role (max · (q + 1))
                  all := all.modify nr.role (max · (q + 1))
                | none => pure ()
              else if lockstep then
                -- not exact: threads outside the roles may take part (the idle warps of a
                -- Triton warp-specialized kernel join its start and end barrier), so the
                -- arrivals a phase needs from a role are unknown: the lock reading only
                let (one, _) := creditArrivals nr.arrivals 1 (fun _ => 1) (ns.needs j k)
                done := done.modify nr.role (max · one)
                all := all.modify nr.role (max · one)
        else
          let l := ledger[s]!
          for (rs, j) in l.roles.zipIdx do
            let m := l.needsWith j k false may
            let m := if lockstep then max m (l.lockNeeds j k) else m
            let (one, _) := creditArrivals rs.arrivals 1 (fun _ => 1) m
            done := done.modify rs.role (max · one)
            if lockstep then all := all.modify rs.role (max · one)
          if lockstep then
            -- bytes carry order (module doc, "Lock reading"): the loads that brought the
            -- bytes of phases 1..k were issued before the wait returned
            for rs in l.roles do
              let q := loadsCoverLock l rs k
              if q > 0 then
                done := done.modify rs.role (max · q)
                all := all.modify rs.role (max · q)
    if !changed then break
  return ({ done, all }, applied)

/-- Per role in `wanted`, the closures before a point of the role after k of its waits,
k = 0 .. its waits (incrementally: each starts from the one before). -/
def closures (named : Array NamedSlot) (exact : Array Bool) (ledger : Array SlotLedger)
    (waits : Array (Array PWait)) (lockstep : Bool) (wanted : Array Bool) : Array (Array Hb) := Id.run do
  let n := waits.size
  let mut out : Array (Array Hb) := Array.replicate n #[]
  for r in [0:n] do
    if !wanted.getD r false then continue
    let mut h : Hb := { done := Array.replicate n 0, all := Array.replicate n 0 }
    let mut applied := Array.replicate n 0
    let mut hs : Array Hb := #[]
    for k in [0:waits[r]!.size + 1] do
      let start := if k = 0 then 0 else waits[r]![k - 1]!.1 + 1
      (h, applied) := saturate named exact ledger waits lockstep { h with done := h.done.modify r (max · start) } applied
      hs := hs.push h
    out := out.set! r hs
  return out

/-- The closure before position `q` of role `r`: after the role's waits before `q`. -/
def before (C : Ctx) (H : Array (Array Hb)) (r q : Nat) : Hb :=
  let k := searchFirst C.waits[r]! (·.1 ≥ q)
  (H[r]!)[k]!

/-- Named-barrier slots of a cluster's roles (`nr` kernel roles; named barriers are per CTA),
with each role's bar.sync waits (trace position, slot, phase) (TmemReuse.build's). -/
def namedOf (croles : Array RoleTrace) (nr : Nat) : Array NamedSlot × Array (Array PWait) := Id.run do
  let mut named : Array NamedSlot := #[]
  let mut key : Array (Nat × Int) := #[]
  let mut nwaits : Array (Array PWait) := Array.replicate croles.size #[]
  for rt in croles do
    let r := rt.role
    let mut syncCount : Array (Int × Nat) := #[]
    for (e, pos) in rt.events.zipIdx do
      match e.ev with
      | .namedBar id count sync threads =>
        let k := (r / nr, id)
        let si := (key.findIdx? (· == k)).getD named.size
        if si == named.size then
          named := named.push { id, count, consistent := true, roles := #[] }
          key := key.push k
        named := named.modify si fun ns =>
          let ns := if ns.count != count then { ns with consistent := false } else ns
          match ns.roles.findIdx? (·.role == r) with
          | some j => { ns with roles := ns.roles.modify j fun x =>
              { x with arrivals := x.arrivals.push (pos, x.total + threads), syncOnly := x.syncOnly && sync } }
          | none => { ns with roles := ns.roles.push { role := r, arrivals := #[(pos, threads)], threads, syncOnly := sync } }
        if sync then
          let j := ((syncCount.find? (·.1 == id)).map (·.2)).getD 0 + 1
          syncCount := (syncCount.filter (·.1 != id)).push (id, j)
          nwaits := nwaits.modify r (·.push (pos, true, si, (j : Int), #[]))
      | _ => pure ()
  return (named, nwaits)

/-- The stores (with their completing waits) and writes of a cluster's roles; role `g` runs in
the CTA of rank `g / nr`. -/
def accessesOf (croles : Array RoleTrace) (nr : Nat) : Array Store × Array Write := Id.run do
  let mut stores : Array Store := #[]
  let mut writes : Array Write := #[]
  for rt in croles do
    let r := rt.role
    let rank := r / nr
    -- this role's stores not completed yet: (index in `stores`, commits before them)
    let mut pending : Array (Nat × Nat) := #[]
    let mut commits : Nat := 0
    for (e, pos) in rt.events.zipIdx do
      match e.ev with
      | .bulkStore kind who sym lo hi tmap =>
        pending := pending.push (stores.size, commits)
        stores := stores.push { role := r, pos, who, cta := rank, kind, sym, lo, hi, tmap, loc := e.loc,
                                done := rt.events.size }
      | .bulkCommit => commits := commits + 1
      | .bulkWait n _ =>
        -- the groups of all but the n most recent commits are complete
        let keep := pending.filter fun (_, c) => (commits : Int) - c < n + 1
        for (i, c) in pending do
          if (commits : Int) - c ≥ n + 1 then stores := stores.modify i ({ · with done := pos })
        pending := keep
      | .smemWrite who sync async cta sym ranges slot =>
        writes := writes.push { role := r, pos, who, sync, async, slot, tmap := false, cta, sym, ranges,
                                loc := e.loc }
      | .tmapWrite warp base lo =>
        writes := writes.push { role := r, pos, who := { warp, lane := none }, sync := true, tmap := true,
                                cta := rank, sym := base, ranges := #[(lo, lo + 128)], loc := e.loc }
      | _ => pure ()
  return (stores, writes)

/-- The P1-2 view of cluster `cl` with its P1 ledger `L`. -/
def build (cl : ClusterTrace) (L : Array SlotLedger) (nkr : Nat) (ctaThreads : Option Nat) : Ctx :=
    Id.run do
  let nr := max 1 nkr
  let croles := cl.roles nr
  let n := croles.size
  let (named, nwaits) := namedOf croles nr
  let exact := named.map (·.exact ctaThreads)
  let sizes := croles.map (·.events.size)
  let merged (bound : WaitRec → Int) : Array (Array PWait) :=
    let mbar := waitBounds L n bound
    let may := mayBeforeReturn L mbar sizes
    (Array.range n).map fun r =>
      let mine : Array (Array Int) := may[r]!
      ((mbar[r]!.zipIdx.map fun (xi : WaitPos × Nat) =>
          let ((p, s, k), i) := xi
          ((p, false, s, k, mine[i]!) : PWait)) ++ nwaits[r]!).qsort (·.1 < ·.1)
  let (stores, writes) := accessesOf croles nr
  let wanted := (Array.range n).map fun r => stores.any (·.role == r) || writes.any (·.role == r)
  let blocked := L.foldl (init := #[]) fun acc l => l.roles.foldl (init := acc) fun acc rs =>
    rs.waits.foldl (init := acc) fun acc w => if w.blocked.isSome then acc.push (rs.role, w.loc) else acc
  let firstBlocked := L.foldl (init := sizes) fun acc l => l.roles.foldl (init := acc) fun acc rs =>
    rs.waits.foldl (init := acc) fun acc w => if w.blocked.isSome then acc.modify rs.role (min · w.pos) else acc
  let waits := merged (·.minPhase)
  let ws := WaitsBySlot.of (waitBounds L n (·.minPhase))
  { pid := cl.cid, roles := croles, ledger := L, named, exact, waits, ws, sizes, stores, writes, blocked, firstBlocked,
    sound := closures named exact L waits false wanted,
    lock := Thunk.mk fun _ => closures named exact L (merged (·.lockPhase)) true wanted }

/-! ## Pairs -/

/-- Whether write `x` is surely, maybe or surely not executed by the thread that issued `s`
(a warp-synchronizing write of `s`'s warp: `lanes`, every lane, the issuer's included;
another fixed lane of `s`'s warp: `otherLane`). -/
inductive Same where
  | sure | maybe | no | lanes | otherLane
  deriving BEq, Inhabited

def same (s : Store) (x : Write) : Same :=
  if s.role != x.role || s.who.warp != x.who.warp then .no
  else if x.sync then .lanes
  else match s.who.lane, x.who.lane with
    | some a, some b => if a == b then .sure else .otherLane
    | none, none => .sure
    | _, _ => .maybe

def overlaps (lo hi : Int) (rs : Array (Int × Int)) : Bool :=
  rs.any fun (a, b) => max lo a < min hi b

/-- `x` conflicts with `s`: a write of `s`'s shared-memory source bytes in `s`'s CTA, or of the
tensormap `s` reads. -/
def conflicts (s : Store) (x : Write) : Bool :=
  if x.tmap then
    match s.tmap with
    | some (base, a) => base == x.sym && overlaps a (a + 128) x.ranges
    | none => false
  else x.cta == s.cta && x.sym == s.sym && overlaps s.lo s.hi x.ranges

/-- `s` completes before `x`: its completing wait happened in every thread executing `x`
(`bx`: the closure before `x`; `sm`: `same s x`). -/
def completesBefore (C : Ctx) (bx : Hb) (sm : Same) (s : Store) (x : Write) : Bool :=
  s.done < C.sizes[s.role]! && (bx.all[s.role]! > s.done || (sm == .sure && s.done < x.pos))

/-- `x` is issued before `s` (`bs`: the closure before `s`; `sm`: `same s x`), counting the
issuer's own program order as `Same` allows. -/
def issuedBeforeWith (bs : Hb) (sm : Same) (s : Store) (x : Write) : Bool :=
  bs.all[x.role]! > x.pos || ((sm == .sure || sm == .lanes) && x.pos < s.pos)

/-- `x` is issued before `s` under the closures `H`, counting the issuer's own program order
as `Same` allows. -/
def issuedBefore (C : Ctx) (H : Array (Array Hb)) (s : Store) (x : Write) : Bool :=
  issuedBeforeWith (before C H s.role s.pos) (same s x) s x

/-- The ledger slot of the landing `x` and the position of its load: the first transaction
event of `x`'s role on `x`'s slot after it, with only global accesses between (module doc,
"Landings"); none if `x` names no slot or no such load. -/
def landingLoad (C : Ctx) (x : Write) : Option (Nat × Nat) := Id.run do
  let some sl := x.slot | return none
  let some si := C.ledger.findIdx? fun l => l.slot.bar == sl.bar && l.slot.idx == sl.idx && l.slot.cta == sl.cta
    | return none
  let L := C.ledger[si]!
  let some rs := L.roles.find? (·.role == x.role) | return none
  let some t := rs.tx.find? fun t => t.load && t.pos > x.pos | return none
  let evs := (C.roles[x.role]?.map (·.events)).getD #[]
  let between := (List.range (t.pos - x.pos - 1)).all fun d =>
    match evs[x.pos + 1 + d]? with
    | some e => match e.ev with
      | Ev.gload _ _ => true
      | _ => false
    | none => false
  return if between then some (si, t.pos) else none

/-- The wait at position `p` of role `r` on ledger slot `si`, with phase bound `k`, shows the
loads that may land before its return complete (module doc, "Landings"): `returnCuts` of its
return, and whether the loads before those cuts bring no more bytes than phases 1..k surely
hold. -/
def landingWait (C : Ctx) (si r p : Nat) (k : Int) : Array Nat × Bool :=
  let L := C.ledger[si]!
  let cuts := returnCuts C.ledger C.ws C.sizes r p
  (cuts, 0 < k && L.landedBefore cuts ≤ bytesIn L k)

/-- The waits on ledger slot `si` that the landing test of a store tries (`hb`: the closure
before the store): per role, its last wait on the slot that surely returned before the store
is issued, as (role, position, phase bound). -/
def landingWaits (C : Ctx) (si : Nat) (hb : Hb) : Array (Nat × Nat × Int) := Id.run do
  let mut out := #[]
  for r in [0:C.ws.size] do
    for (s', xs) in C.ws[r]! do
      if s' != si then continue
      if let some (p, k) := (xs.filter fun (p, _) => p < hb.done[r]!).back? then
        out := out.push (r, p, k)
  return out

/-- The landing `x` completed before `s` is issued under the closures `H` (module doc,
"Landings"): its load (`landingLoad`) may land before the return of a wait on that slot that
surely returned before `s` is issued (the last such wait of each role is tried), and the
loads that may land before that return bring no more bytes than its phase bound surely
needs (`landingWait`). -/
def landed (C : Ctx) (H : Array (Array Hb)) (s : Store) (x : Write) : Bool := Id.run do
  let some (si, t) := landingLoad C x | return false
  for (r, p, k) in landingWaits C si (before C H s.role s.pos) do
    let (cuts, bytes) := landingWait C si r p k
    if t < cuts[x.role]! && bytes then
      return true
  return false

/-- `s` and `x` are ordered under the closures `H` (module doc), counting the issuer's own
program order as `Same` allows. -/
def ordered (C : Ctx) (H : Array (Array Hb)) (s : Store) (x : Write) : Bool :=
  -- s completes before x, or x happens before s is issued (an asynchronous write: its
  -- landing, `landed`)
  completesBefore C (before C H x.role x.pos) (same s x) s x ||
    (!x.async && issuedBefore C H s x) || (x.async && landed C H s x)

/-- The issuer's own program order might order the pair, which the KIR cannot pin down (U2):
the write may be the issuer's (`maybe`), or it is another lane's of the issuer's warp, which a
`bar.warp.sync` of the warp (dropped from the KIR) orders as the issuer's program order
would (`otherLane`). -/
def maybeOrdered (C : Ctx) (s : Store) (x : Write) : Bool :=
  match same s x with
  | .maybe | .otherLane => (s.done < C.sizes[s.role]! && s.done < x.pos) || x.pos < s.pos
  | .lanes => s.done < C.sizes[s.role]! && s.done < x.pos
  | _ => false

inductive PairKind where
  /-- F1 / F2: unordered even in the lock reading. -/
  | fail
  /-- U1: ordered only in the lock reading. -/
  | lockOnly
  /-- U2: unordered in the lock reading, but maybe ordered by the issuer's own program order
  (for another lane's write: with a `bar.warp.sync` of the warp). -/
  | maybeSame
  /-- U3: an asynchronous write issued before the store (sound reading) that may land after
  the store reads: its landing is not known to precede the store's issue (`landed`). -/
  | landing
  /-- U4: unordered in the lock reading, but the store or the write follows a wait the
  ledger finds blocked in its role (`Ctx.firstBlocked`): in the lockstep model it never runs. -/
  | blocked
  deriving BEq, Inhabited

/-- The kind of an unordered conflicting pair (`PairKind`). -/
def pairKind (C : Ctx) (s : Store) (x : Write) : PairKind :=
  if x.async && issuedBefore C C.sound s x then .landing
  else if ordered C C.lock.get s x then .lockOnly
  else if x.async && issuedBefore C C.lock.get s x then .landing
  else if maybeOrdered C s x then .maybeSame
  else if s.pos > C.firstBlocked[s.role]! || x.pos > C.firstBlocked[x.role]! then .blocked
  else .fail

/-- Per symbol, the union `[lo, hi)` of the byte ranges `xs` (symbol, lo, hi). -/
def unionBySym (xs : Array (String × Int × Int)) : Array (String × Int × Int) :=
  xs.foldl (init := #[]) fun acc (sym, lo, hi) =>
    match acc.findIdx? (·.1 == sym) with
    | some j => acc.modify j fun (s', a, b) => (s', min a lo, max b hi)
    | none => acc.push (sym, lo, hi)

def Same.index : Same → Nat
  | .sure => 0 | .maybe => 1 | .no => 2 | .lanes => 3 | .otherLane => 4

/-- The conflicting pairs not ordered in the sound reading, (store, write, kind), by store
then write: the pairs with `conflicts s x && !ordered C C.sound s x`, found without trying
every pair. The stores of one role and issuing thread form a group, in program order, on
which `same s x` is constant for a write `x`. On a group, the stores that complete before `x`
(`completesBefore`) are a prefix: `Store.done` does not decrease in program order (a wait
that completes a store's group completes every earlier group of its thread). For a
non-asynchronous `x`, those `x` is issued before (`issuedBeforeWith`) are a suffix: the
closure before a point grows with the point (`closures`). So only the stores between the two
bounds are tried. The writes come role by role in program order, and both bounds only grow
with the write's position (a later write has a larger closure before it, and fewer stores are
issued after it), so they are cursors per group and `same` value, reset at each role; a
group none of whose stores can conflict with `x` (its CTA, symbols and byte ranges, or its
tensormaps) is skipped. A landing tries every store after the prefix, its `landed` test
sharing the cuts of each wait it tries (`landingWait`) over all pairs. -/
def pairs (C : Ctx) : Array (Nat × Nat × PairKind) := Id.run do
  let H := C.sound
  let mut keys : Array (Nat × Who) := #[]
  let mut groups : Array (Array Nat) := #[]
  for (s, i) in C.stores.zipIdx do
    match keys.findIdx? (· == (s.role, s.who)) with
    | some g => groups := groups.modify g (·.push i)
    | none =>
      keys := keys.push (s.role, s.who)
      groups := groups.push #[i]
  let bs : Array Hb := C.stores.map fun s => before C H s.role s.pos
  -- per group: its CTA, the union of its sources per symbol, of its tensormaps per buffer
  let gcta : Array Nat := groups.map fun g => C.stores[g[0]!]!.cta
  let gsrc : Array (Array (String × Int × Int)) :=
    groups.map fun g => unionBySym (g.map fun i => let s := C.stores[i]!; (s.sym, s.lo, s.hi))
  let gtm : Array (Array (String × Int × Int)) :=
    groups.map fun g => unionBySym (g.filterMap fun i => (C.stores[i]!.tmap).map fun (b, a) => (b, a, a + 128))
  let may (gi : Nat) (x : Write) : Bool :=
    if x.tmap then gtm[gi]!.any fun (b, lo, hi) => b == x.sym && overlaps lo hi x.ranges
    else gcta[gi]! == x.cta && gsrc[gi]!.any fun (sym, lo, hi) => sym == x.sym && overlaps lo hi x.ranges
  let fresh : Array (Nat × Nat) := Array.replicate (5 * groups.size) (0, 0)
  let mut cur := fresh
  let mut role := C.roles.size
  let mut wk := 0
  let mut memo : Std.HashMap (Nat × Nat) (Array Nat × Bool) := {}
  let mut out : Array (Nat × Nat × PairKind) := #[]
  for (x, j) in C.writes.zipIdx do
    if x.role != role then
      role := x.role
      cur := fresh
      wk := 0
    -- the closure before x (`before`): after the role's waits before it
    let ws := C.waits[x.role]!
    while wk < ws.size && ws[wk]!.1 < x.pos do
      wk := wk + 1
    let bx := H[x.role]![wk]!
    let load := if x.async then landingLoad C x else none
    for gi in [0:groups.size] do
      if !may gi x then continue
      let g := groups[gi]!
      let sm := same C.stores[g[0]!]! x
      let ci := 5 * gi + sm.index
      let (lo0, hi0) := cur[ci]!
      let mut lo := lo0
      while lo < g.size && completesBefore C bx sm C.stores[g[lo]!]! x do
        lo := lo + 1
      let mut hi := hi0
      while hi < g.size && !issuedBeforeWith bs[g[hi]!]! sm C.stores[g[hi]!]! x do
        hi := hi + 1
      cur := cur.set! ci (lo, hi)
      let last := if x.async then g.size else hi
      for k in [lo:last] do
        let i := g[k]!
        let s := C.stores[i]!
        if !conflicts s x then continue
        -- `ordered C H s x`: the prefix and suffix are out of [lo, last); a landing's test
        let mut ord := false
        if let some (si, t) := load then
          for (r, p, kb) in landingWaits C si bs[i]! do
            let w ← match memo.get? (r, p) with
              | some w => pure w
              | none =>
                let w := landingWait C si r p kb
                memo := memo.insert (r, p) w
                pure w
            if t < w.1[x.role]! && w.2 then
              ord := true
              break
        if !ord then
          out := out.push (i, j, pairKind C s x)
  return out.qsort fun a b => a.1 < b.1 || (a.1 == b.1 && a.2.1 < b.2.1)

/-- The tensormap pairs of two clusters (no happens-before across clusters): per cluster of
`Cs` its stores' tensormaps and its tensormap writes. -/
def crossPairs (Cs : Array Ctx) : Array (Nat × Nat × Nat × Nat) := Id.run do
  -- per cluster, the indices of its tensormap writes (few), in order
  let tw : Array (Array Nat) := Cs.map fun D => (D.writes.zipIdx.filter (·.1.tmap)).map (·.2)
  let mut out := #[]
  for (C, a) in Cs.zipIdx do
    for (s, i) in C.stores.zipIdx do
      if s.tmap.isSome then
        for (D, b) in Cs.zipIdx do
          if a != b then
            for j in tw[b]! do
              if conflicts s D.writes[j]! then out := out.push (a, i, b, j)
  return out

/-! ## Decision -/

def ctxs (r : Run) (Ls : Ledgers) : Array Ctx :=
  let nkr := r.host.kernel.roles.length
  (r.clusters.zip Ls).map fun (c, (_, L)) => build c L nkr r.host.kernel.cta_threads

/-- In every cluster of `Cs` every conflicting pair is ordered in the sound reading, and no
tensormap a store of one cluster reads is written by another cluster. -/
def HoldsOn (Cs : Array Ctx) : Prop := (∀ c ∈ Cs, (pairs c).isEmpty = true) ∧ (crossPairs Cs).isEmpty = true

instance (Cs : Array Ctx) : Decidable (HoldsOn Cs) := by unfold HoldsOn; infer_instance

/-- P1-2 on one run, over its complete clusters and P1 ledgers. -/
def Holds (r : Run) : Prop := HoldsOn (ctxs r r.ledgersP1)

instance (r : Run) : Decidable (Holds r) := by unfold Holds; infer_instance

/-! ## Counterexamples and notes -/

def ctxName (h : Host) (C : Ctx) : String :=
  if h.csize ≤ 1 then s!"cta {pidStr C.pid}" else s!"cluster {pidStr C.pid}"

def locAt (C : Ctx) (r p : Nat) : Loc := ((C.roles[r]?.bind (·.events[p]?)).map (·.loc)).getD "?"

def Store.describe (h : Host) (C : Ctx) (s : Store) : String :=
  let tm := match s.tmap with
    | some (b, a) => s!" and the tensormap at {b}+{a}"
    | none => ""
  s!"{kindName s.kind} at {s.loc} (role {h.gRoleName C.pid s.role}, {whoName s.who}) reading {s.sym}[{s.lo}, {s.hi}){tm}"

def Write.describe (h : Host) (C : Ctx) (x : Write) : String :=
  let who := if x.sync then s!"every lane of warp {x.who.warp}" else whoName x.who
  let rs := (x.ranges.toList.take 2).map fun (a, b) => s!"[{a}, {b})"
  let what := if x.tmap then s!"writes the tensormap at {x.sym}+{(x.ranges[0]?.map (·.1)).getD 0}"
    else if x.async then s!"lands in {x.sym} {" ".intercalate rs}{if x.ranges.size > 2 then s!" ... ({x.ranges.size} ranges)" else ""}"
    else s!"writes {x.sym} {" ".intercalate rs}{if x.ranges.size > 2 then s!" ... ({x.ranges.size} ranges)" else ""}"
  s!"the write at {x.loc} (role {h.gRoleName C.pid x.role}, {who}) {what}"

/-- Why neither direction is ordered under `H`. -/
def whyNot (h : Host) (C : Ctx) (H : Array (Array Hb)) (s : Store) (x : Write) : String :=
  let rs := h.gRoleName C.pid s.role
  let comp := if s.done < C.sizes[s.role]! then
      s!"it completes at the wait at {locAt C s.role s.done}, which is not surely before the write in every thread executing it"
    else s!"no cp.async.bulk.wait_group of {rs} completes it"
  let a := (before C H s.role s.pos).all[x.role]!
  let prior := if a = 0 then "none" else s!"at {locAt C x.role (a - 1)}"
  s!"{comp}; the latest event of role {h.gRoleName C.pid x.role} surely before the store in every thread is {prior}"

def pairIssue (h : Host) (C : Ctx) (i j : Nat) (k : PairKind) (extra : Nat) : Issue :=
  let s := C.stores[i]!
  let x := C.writes[j]!
  let spec := if x.tmap then "F2" else "F1"
  let more := if extra = 0 then "" else s!" ({extra} more pair(s) at these two lines in this {if h.csize ≤ 1 then "CTA" else "cluster"})"
  let msg := match k with
    | .fail => s!"{spec}: {x.describe h C} while the {s.describe h C} may still read it: unordered, even under the lockstep reading ({whyNot h C C.lock.get s x})"
    | .lockOnly => s!"{x.describe h C} and the {s.describe h C}: ordered only under the lockstep reading (the ledger's lockstep contents, or mbarrier arrivals taken as every thread's), not by sound happens-before ({whyNot h C C.sound s x})"
    | .maybeSame => if same s x == .otherLane then
        s!"{x.describe h C} and the {s.describe h C}: ordered only if a bar.warp.sync of their warp orders the write with the store's issuing thread (another lane of its warp), which the KIR does not keep (bar.warp.sync orders the memory accesses of the lanes taking part)"
      else s!"{x.describe h C} and the {s.describe h C}: ordered only if the store's issuing thread itself executes the write in program order, which the KIR does not pin down (an elected lane and a fixed lane of one warp, or a warp-collective write after the completing wait)"
    | .landing => s!"{x.describe h C} and the {s.describe h C}: the asynchronous copy is issued before the store, but may land after the store reads (its mbarrier's bytes do not show it complete before the store is issued)"
    | .blocked => s!"{x.describe h C} and the {s.describe h C}: unordered even under the lockstep reading, but {if s.pos > C.firstBlocked[s.role]! then "the store" else "the write"} follows a wait the P1 ledger finds blocked (it never returns in the lockstep model: P0-2), so the pair may never run"
  -- an undecided pair is no counterexample, also beside certain ones (a FAIL's lines)
  let tag := if k == .fail then "" else "not decided on this input: "
  { loc := x.loc, context := s!"{ctxName h C} roles {h.gRoleName C.pid s.role} / {h.gRoleName C.pid x.role}", msg := tag ++ msg ++ more }

/-- Pairs of kind `k`, one line per distinct (write line, store line) with a count. -/
def pairIssues (h : Host) (C : Ctx) (ps : Array (Nat × Nat × PairKind)) (k : PairKind) : Array Issue := Id.run do
  let mut keys : Array (Loc × Loc × Nat × Nat × Nat) := #[]
  for (i, j, k') in ps do
    if k' == k then
      let a := C.writes[j]!.loc
      let b := C.stores[i]!.loc
      match keys.findIdx? fun (a', b', _, _, _) => a' == a && b' == b with
      | some t => keys := keys.modify t fun (a', b', i', j', e) => (a', b', i', j', e + 1)
      | none => keys := keys.push (a, b, i, j, 0)
  return keys.map fun (_, _, i, j, e) => pairIssue h C i j k e

structure P2Out where
  ok : Bool
  issues : Array Issue
  notes : Array Issue
  /-- Some issue is a certain counterexample. -/
  certain : Bool
  stores : Nat
  writes : Nat
  pairs : Nat
  undecided : Nat
  named : Nat
  inconsistent : Array Int
  blocked : Nat

/-- P1-2's verdict lines and notes on run `r` with its P1 ledgers `Ls` (module doc): a
stopped-trace line, then per cluster its certain F1/F2 lines (or, with a named barrier used
with two counts or a ledger that does not converge, a `not decided` line and them), the
cross-cluster F2 lines, then the U1-U4 lines (each a `not decided` line), after a
`not decided` head line if nothing is certain. -/
def analyze (r : Run) (Ls : Ledgers) : P2Out := Id.run do
  let h := r.host
  let Cs := ctxs r Ls
  let mut ok := true
  let mut issues : Array Issue := #[]
  let mut certain := false
  let mut undec : Array Issue := #[]
  let mut nund := 0
  let mut notes : Array Issue := #[]
  let mut nst := 0
  let mut nwr := 0
  let mut npairs := 0
  let mut nnamed := 0
  let mut incons : Array Int := #[]
  let mut nblocked := 0
  for C in Cs do
    let ps := pairs C
    let bad := C.named.filter (!·.consistent)
    let fails := pairIssues h C ps .fail
    let why : Option String :=
      if !bad.isEmpty then
        some s!"named barrier(s) {(bad.map (·.id)).toList} are used with two counts, so they order nothing; the unordered pairs below may be ordered by them"
      else if !C.ledger.all (·.converged) then
        some s!"the barrier ledger of this {if h.csize ≤ 1 then "CTA" else "cluster"} does not converge, so the lockstep reading the pairs below are judged under is one round's, not a fixpoint's"
      else none
    let und : Array Issue := match why with
      | some w => if fails.isEmpty then #[] else #[{ loc := "", context := ctxName h C, msg := s!"not decided on this input: {w}" }]
      | none => #[]
    ok := ok && ps.isEmpty
    certain := certain || (why.isNone && !fails.isEmpty)
    issues := issues ++ und ++ fails
    let u := pairIssues h C ps .lockOnly ++ pairIssues h C ps .maybeSame ++ pairIssues h C ps .landing ++
      pairIssues h C ps .blocked
    undec := undec ++ u
    nund := nund + (ps.filter (·.2.2 != .fail)).size
    -- N1: stores never completed; N2: blocked waits
    for s in C.stores do
      if s.done ≥ C.sizes[s.role]! then
        notes := notes.push { loc := s.loc, context := s!"{ctxName h C} role {h.gRoleName C.pid s.role}",
                              msg := s!"the {s.describe h C} is never completed by a cp.async.bulk.wait_group of its issuing thread before its role ends" }
    match C.blocked[0]? with
    | some (role, loc) =>
      notes := notes.push { loc, context := s!"{ctxName h C} role {h.gRoleName C.pid role}",
                            msg := s!"the P1 ledger finds {C.blocked.size} wait(s) that never return in the lockstep model (P0-2): happens-before after them is vacuous, so other P1-2 lines may be their consequences" }
    | none => pure ()
    nst := nst + C.stores.size
    nwr := nwr + C.writes.size
    npairs := npairs + ps.size
    nnamed := nnamed + C.named.size
    incons := incons ++ bad.map (·.id)
    nblocked := nblocked + C.blocked.size
  -- tensormaps shared by two clusters: no happens-before at all (certain, but U4 after a
  -- blocked wait)
  for (a, i, b, j) in crossPairs Cs do
    let C := Cs[a]!
    let D := Cs[b]!
    let s := C.stores[i]!
    let x := D.writes[j]!
    ok := false
    npairs := npairs + 1
    let context := s!"{ctxName h C} / {ctxName h D}"
    let pair := s!"{x.describe h D} (in {ctxName h D}) while the {s.describe h C} (in {ctxName h C}) may still read it: no happens-before across clusters"
    if s.pos > C.firstBlocked[s.role]! || x.pos > D.firstBlocked[x.role]! then
      nund := nund + 1
      undec := undec.push { loc := x.loc, context,
                            msg := s!"not decided on this input: {pair}, but {if s.pos > C.firstBlocked[s.role]! then "the store" else "the write"} follows a wait the P1 ledger finds blocked (it never returns in the lockstep model: P0-2), so the pair may never run" }
    else
      certain := true
      issues := issues.push { loc := x.loc, context, msg := s!"F2: {pair}" }
  if !undec.isEmpty then
    let first := undec[0]?
    let head : Array Issue := if certain then #[] else
      #[{ loc := (first.map (·.loc)).getD "", context := (first.map (·.context)).getD "",
          msg := s!"not decided on this input: {nund} conflicting pair(s) of a bulk store and a write are ordered only under the lockstep reading or by a same-thread program order the KIR does not pin down; P1-2 is proved only if sound happens-before orders every pair" }]
    issues := issues ++ head ++ undec
  let stops := r.ctas.flatMap fun c => (c.roles.filter (·.stopped)).map fun t => (c.pid, t)
  match stops[0]? with
  | some (pid, t) =>
    issues := #[{ loc := t.stopLoc, context := h.roleCtx pid t.role,
                  msg := s!"not decided on this input: {stops.size} role trace(s) stop at {stopReasons (stops.map (·.2))}, so their later events are unknown; the lines of this spec that are not `not decided` lines are certain counterexamples" }] ++ issues
  | none => pure ()
  return { ok, issues, notes, certain, stores := nst, writes := nwr, pairs := npairs, undecided := nund,
           named := nnamed, inconsistent := incons, blocked := nblocked }

end KV.StoreReuse
