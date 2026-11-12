/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
Per-CTA barrier ledger: for every mbarrier slot touched in a CTA, what P0-2 and P0-4 are
stated over: arrivals, per role its waits with the phase each one synchronizes with, the
lockstep phase contents, and the transaction bytes per phase.

Phases. Phase k of a slot completes when its arrivals reach k x count and, on a slot with
transaction bytes, the bytes of the phase have landed. A wait with parity p returns once
the number of completed phases has parity 1-p. A role's first wait on a slot synchronizes
with phase 1 (parity 0, consumer convention) or phase 0 (parity 1, producer convention: it
passes on the fresh barrier). A wait whose parity differs from the role's previous wait on
the slot moves to the next phase. A wait that repeats the previous parity (previous phase
c) is
* a re-check of phase c if phase c+1 cannot complete before the wait runs (too few
  arrivals may precede it);
* a skip to phase c+2 if phase c+1 surely completed before the wait runs;
* a presumed skip if phase c+1 surely completed before it only with the lockstep credit
  (below): it counts as phase c+2 with lower bound c, is no P0-4 violation, and the hang
  checks of P0-2 count it at c+2 (`assumedPhase`);
* stale otherwise: it can return at once or wait for phase c+2, depending on timing. It
  counts as phase c+1, the one the role presumably meant, with lower bound c.
A first wait, a wait with a new parity or a skip that would synchronize with phase k while
phase k+1 surely completed before it returns at phase k+2 or later, so it synchronizes
with k+2 (repeated while that holds). After the role's first wait this is a lap
(`WaitRec.lapped`, a P0-4 note): parity does not say which phase a role means, so it can
be intended (a role that waits once per tile on a slot completing an odd number of phases
per tile) or not.

Happens-before. "Cannot complete before" (`arrivalsMayPrecede`) and "surely completed
before" (`saturate`, `completedIn`) are arguments over the CTA's role traces that treat
each role as one thread of control (its threads run in lockstep). Phase k surely completed
before an event if a wait on the slot for phase k or later surely returned before it or,
on a slot whose phases complete by arrivals alone (no transaction bytes, no commit
arrivals), k x count arrivals surely precede it. Arrivals are necessary for a phase to
complete, not sufficient: the phase may still wait for bytes, and a commit arrives when its
MMAs complete, some time after the event that issues it. An event surely precedes another
if it precedes it in its role, or precedes an arrival that a wait which surely returned
before the other needs: phase k needs k x count minus all other roles' arrivals over the
run (`SlotLedger.needs`). Every wait is credited with a lower bound of its phase
(re-checks, stale waits and presumed skips at c, skips at c+2). Classifying the repeated
parities can raise these bounds and change the contents, so classification is repeated
until a round changes neither the bounds, the contents nor any wait's classification, for
at most (the CTA's waits + 2) rounds. Classification and contents justify each other only
at that fixpoint: a CTA that does not reach it (also one whose round brings back the state
of two rounds before, a cycle that never ends) leaves P0-2 and P0-4 not decided
(`SlotLedger.converged`, `unconverged`). The soundness of these arguments against mbarrier
semantics is argued here, not proved, so P0-2 and P0-4 verdicts are relative to this
algorithm (references/design.md, trust base).

Lockstep credit. A second closure also credits phase k with each arriving role's least
share of the lockstep contents (below) over phases up to k. The contents are an
assumption (fixed shares per phase), not a certainty: a slot whose real phases are not
uniform can still fit uniform shares. So the credited closure is used for one decision
only: a repeated parity that the sound closure calls stale is a presumed skip if phase
c+1 surely completed before it with the credit. Its phase lower bound stays c, so
happens-before never uses the credit; the hang checks of P0-2 count it at c+2
(`WaitRec.assumedPhase`): if the contents are right the role needs that phase, and if not
the wait is stale, so a hang found that way is a violation either way. The credit thus
replaces a stale FAIL by the verdict under the contents, and never fails an input that
passes without it.

Lockstep phase contents. Each arriving role contributes the same share of arrivals to
every waited complete phase, made of whole arrive events (`lockstepShares`); failing that,
an initialization arrival forms phase 1 by itself and the shares cover the phases after it
(`lockstepContents`). A role with arrivals takes share 0 only if they all surely follow the
last waited phase. An initialization arrival is the first count arrivals of one role,
reached at one of its arrive events, such that nothing else can land in phase 1: every
other role's first arrival on the slot surely follows phase 1's completion, or the
initialization arrivals are plain (no commit among them) and surely precede every other
role's first arrive event (a commit lands after its event, so it is later still; a wait
before them just blocks until they land). This evidence uses happens-before credit only
from slots whose contents need none (every arriving role has a positive least share, no
initialization arrival) and never from the slot itself, so it does not justify itself.
Transaction bytes are attributed through the issuing role's own arrivals: an expect_tx
belongs to the phase its role's next arrival falls in, a tma_load to the phase of its
role's latest expect_tx. Phase K + 1 (K the completions) is the last phase: it never
completes, so bytes issued after phase K land in it. Several share assignments can fit the
arrivals; bytes are attributed under each of them (the representative first, at most
`maxShareVectors` in all) until one satisfies P0-4's transaction clauses
(`SlotLedger.TxOk`), so P0-4 fails only if none does.

Precise needs (P1-1: `ledger` with `precise`, `Run.ledgersP1`; P0-4 only for the landers'
lower bounds, below; P0-2 and the classification of P0-4 use the default). Subtracting the other roles' arrivals over the whole run says nothing about a
role whose co-arrivers have later arrivals: with compute warps split into roles of 1 and
3 warps on one count-4 slot, phase k needs `4k - 3K` of the 1-warp role (K the run's
phases), at most 0 for k < K. A wait that returned for phase k saw `k x count` arrivals
that precede its return, and an arrival of role o can precede it only if o issues it
before `returnCuts`' cut of o: the first position from which o's events surely follow the
return (o's events after a wait of o that cannot return first, because the arrivals that
may precede the return cannot complete its phase bound; the fixpoint of
`arrivalsMayPrecede`'s cuts, which only move down, each move sound given the cuts before
it). So role r issued at least `k x count` minus the other roles' arrivals before their
cuts (`SlotLedger.needsWith`, `mayBeforeReturn`): in a pipeline, exactly its share. These
needs are as sound as the run-total ones and at least as large; the classification uses
them too (more laps and skips proved). Each wait also gets `lockPhase`, its phase lower
bound under the lockstep reading: `assumedPhase`, at least the previous wait's
`lockPhase` (+1 after a parity change), raised by the laps the lockstep credit shows. P1-1
uses it, with each role's largest fitting share (`SlotLedger.lockNeeds`), only to tell a
certain race from an undecided pair, never to prove ordering.

Partial credit (P1-1 only, `precise`). The warps of a P1 role all issue its mbarrier
events (the CuTe P1 pass splits roles so) but need not be at one point. A need of m
arrivals of a role of several warps shows that some warp issued the arrive event holding
the m-th arrival, and that every warp issued it only if m takes more of its arrivals than
all warps but one give (each warp at least one; an equal part under the lockstep credit);
otherwise every warp issued the role's previous arrive event (`creditArrivals`). The
closures keep both prefixes (`saturate` with `warps`): a wait in the first returned (its
phase completed), arrivals count toward a phase only in the second, and a wait's own role
is in the second up to it only if it is one warp. A warp running so far ahead of another
that its arrivals stand in for the other's is not modeled.

Optimistic progress (`deadlocks`). Every role runs as far as its waits allow when each
arrival takes effect as soon as its event is reached and transaction bytes are ignored; a
wait passes once the arrivals on its slot reach its `assumedPhase` x count. A wait still
blocked when nothing can move never returns in the lockstep model (unless a presumed skip
it depends on is stale): a hang or a cross-role wait cycle.

Clusters. The ledger is per cluster (one CTA without clusters): its roles are the roles of
every CTA of the cluster (`ClusterTrace`, role `g` is role `g % n` of the CTA of rank `g / n`)
and its slots the barrier slots of every CTA, so an arrival, commit or transaction event a
role issues on another CTA's slot (a remote arrive, a multicast commit, a TMA load completing
on the pair leader's barrier, a bulk copy into another CTA) counts on that slot like a local
one, and happens-before runs over all of them.

Landers. A role of another CTA whose only events on a slot are transaction landings (a TMA
load or `complete_tx`, no arrival, no expect_tx) lands bytes a role of the slot's CTA
expects: the other CTA of a pair loading into the leader's barrier, or a peer copying into
this CTA. (Within a CTA, transaction bytes keep the rule above: the role that expects them
arrives.) Its bytes are attributed by its issue order: a landing belongs to phase f + 1 (at
most K + 1), f the highest phase of the slot that surely completed before the landing is
issued (sound happens-before with the precise needs above, `TxRec.floor`; e.g. the other CTA
of a pair loads iteration k after its wait on its own empty slot, which the leader's MMA
releases only after it waited for phase k - 1; in a multicast pipeline that empty slot has
an arriving role per CTA, which the run totals never order). That is a lower bound: the
landing cannot complete an earlier phase, but it may land in a later one if the phase it is
put in completes without it. The bytes per phase need not be uniform (a prologue that also
loads K into the first phase), but they are not negative (a negative transaction count is an
evaluation error, Eval.lean `txBytes`).
* `TxOk` under it proves the attribution: by induction over the phases, phase g can only
  receive the landings put in g (later ones are issued after g completed, earlier ones
  landed in their own phases), and needs all of them, so each lands where it was put.
* A failure is certain if it does not depend on the lower bounds: a landing put in a phase
  nobody waits for, a clause of the slot's own roles, or a first unbalanced waited phase
  that receives fewer bytes than it expects (no later landing can make up for it), or
  waited phases whose bytes differ from their expect_tx bytes in all (the excess lands in
  an unwaited phase, or a phase lacks bytes).
* Otherwise the first unbalanced phase receives more bytes than it expects while the
  totals match: a landing may complete a later phase instead, depending on timing the
  model does not order (a named barrier, an order carried only by transaction bytes, a
  happens-before chain the closure does not find). P0-4 is then not decided on the slot
  (`SlotLedger.txUndecided`), never a PASS or a certain FAIL; with several fitting share
  assignments, it is not decided as soon as one of them fails only so (none being `TxOk`),
  and the slot's lines are that assignment's.
-/
import KV.Spec.Common

namespace KV

/-- First index of `xs` satisfying the monotone predicate `p` (`xs.size` if none). -/
def searchFirst {α : Type} [Inhabited α] (xs : Array α) (p : α → Bool) : Nat := Id.run do
  let mut lo := 0
  let mut hi := xs.size
  for _ in [0:xs.size + 1] do
    if lo ≥ hi then break
    let mid := (lo + hi) / 2
    if p xs[mid]! then hi := mid else lo := mid + 1
  return lo

/-- What a need of `m` arrivals of a role of `warps` warps shows about its arrive events on
a slot or named barrier (`arrivals`: (trace position, arrivals so far)), as trace prefixes
(an arrive event's position + 1; 0: none): `(some warp, every warp)` of the role surely
issued it (P1-1, module doc "Partial credit"). The `m`-th arrival is part of event `a`,
the first reaching `m`, so some warp issued `a`. Every warp did if the need takes more of
`a`'s `n` arrivals than all warps but one can give, each giving at least `share n`:
`m - before > n - share n`. Otherwise every warp issued the event before `a`. With one warp
(`warps ≤ 1`) both are `a`: the credit P0-2 and P0-4 use. -/
def creditArrivals (arrivals : Array (Nat × Int)) (warps : Nat) (share : Int → Int) (m : Int) : Nat × Nat :=
  let a := searchFirst arrivals (·.2 ≥ m)
  if m ≤ 0 || a ≥ arrivals.size then (0, 0) else
  let before := if a = 0 then 0 else arrivals[a - 1]!.2
  let n := arrivals[a]!.2 - before
  let reached := arrivals[a]!.1 + 1
  if warps ≤ 1 || m - before > n - share n then (reached, reached)
  else (reached, if a = 0 then 0 else arrivals[a - 1]!.1 + 1)

/-- The arrivals of an mbarrier arrive event bringing `n` that `creditArrivals` counts on
from each warp of a role of `warps` warps: one in sound happens-before (every warp of a P1
role issues every arrive, with at least one thread); an equal part under the `lockstep`
reading (the most each warp can surely give). -/
def warpShare (warps : Nat) (lockstep : Bool) (n : Int) : Int :=
  if lockstep then n / ((max warps 1 : Nat) : Int) else 1

/-- The event that shows a phase completed before another event: a wait for that phase or
a later one that surely returned before it (`wait = true`), or the arrive event that brings
the arrivals that surely precede it to the phase. -/
structure Evidence where
  role : Nat
  loc : Loc
  wait : Bool
  deriving BEq, Inhabited

structure WaitRec where
  parity : Int
  loc : Loc
  /-- Position of the wait in its role's trace. -/
  pos : Nat
  /-- Lower bound of the phase the wait synchronizes with (module doc). -/
  minPhase : Int
  /-- The same lower bound with every presumed skip counted at c+2: the phase the wait
  needs at least if the lockstep contents are right (if not, a presumed skip is stale). -/
  assumedPhase : Int
  /-- Phase the wait synchronizes with (stale waits and presumed skips: the one presumably
  meant). -/
  phase : Int
  /-- A presumed skip (module doc). -/
  presumed : Bool
  /-- A stale repeated parity: (arrivals that may precede the wait, arrivals that complete
  the next phase). -/
  stale : Option (Int × Int)
  /-- A lap (module doc): the phase k the wait would synchronize with after its role's
  previous wait, and the event showing that phase k+1 surely completed before it. -/
  lapped : Option (Int × Evidence)
  /-- A wait that never returns in the optimistic progress pass: (arrivals that reach the
  slot before it, arrivals its `assumedPhase` needs). -/
  blocked : Option (Int × Int)
  /-- The phase lower bound under the lockstep reading (P1-1 only, `ledger` with `precise`;
  else `assumedPhase`): `assumedPhase`, at least the role's previous `lockPhase` on the slot
  (+1 after a parity change), raised by the laps that the lockstep credit shows. -/
  lockPhase : Int := 0
  deriving BEq, Inhabited

/-- An expect_tx (`load = false`) or a tma_load of one role on one slot. -/
structure TxRec where
  load : Bool
  bytes : Int
  loc : Loc
  /-- The role's arrivals on the slot before this event. -/
  arrivedBefore : Int
  /-- Position of the event in its role's trace. -/
  pos : Nat
  /-- A lander's landing (module doc, "Landers"): the highest phase of the slot that surely
  completed before it is issued (`ledger`, sound happens-before with the precise needs). -/
  floor : Int := 0
  deriving Inhabited

structure RoleSlot where
  role : Nat
  /-- This role's waits on the slot, in program order. -/
  waits : Array WaitRec
  /-- This role's arrivals on the slot: (position in the role trace, arrivals so far). -/
  arrivals : Array (Nat × Int)
  /-- This role's expect_tx and tma_load events on the slot, in program order. -/
  tx : Array TxRec
  /-- Index in `arrivals` of this role's first `commit_arrive` on the slot. -/
  firstCommit : Option Nat
  /-- The role runs in another CTA of the cluster than the one holding the slot. -/
  remote : Bool := false
  deriving Inhabited

def RoleSlot.firstParity (rs : RoleSlot) : Int := (rs.waits[0]?.map (·.parity)).getD 0

/-- This role's arrivals on the slot over the whole run. -/
def RoleSlot.total (rs : RoleSlot) : Int := (rs.arrivals.back?.map (·.2)).getD 0

/-- Completions this role's waits need: the phase of its last wait (stale waits and
presumed skips counted at the phase the role presumably meant). -/
def RoleSlot.required (rs : RoleSlot) : Int := (rs.waits.back?.map (·.phase)).getD 0

/-- Completions this role's waits need unless one of its presumed skips is stale (a parity
bug then): the `assumedPhase` of its last wait (the largest). -/
def RoleSlot.needed (rs : RoleSlot) : Int := (rs.waits.back?.map (·.assumedPhase)).getD 0

/-- A lander (module doc): a role of another CTA whose only events on the slot are
transaction landings. -/
def RoleSlot.lander (rs : RoleSlot) : Bool :=
  rs.remote && rs.arrivals.isEmpty && !rs.tx.isEmpty && rs.tx.all (·.load)

/-- Index of the first entry of `rs.arrivals` satisfying the monotone predicate `p`
(`rs.arrivals.size` if none). -/
def RoleSlot.firstArrival (rs : RoleSlot) (p : Nat × Int → Bool) : Nat := searchFirst rs.arrivals p

/-- This role's arrivals on the slot at trace positions before `cut`. -/
def RoleSlot.arrivedBefore (rs : RoleSlot) (cut : Nat) : Int :=
  match rs.firstArrival (·.1 ≥ cut) with
  | 0 => 0
  | i => rs.arrivals[i - 1]!.2

/-- This role's arrivals on the slot reach exactly `n` at one of its arrive events. -/
def RoleSlot.reaches (rs : RoleSlot) (n : Int) : Bool :=
  let i := rs.firstArrival (·.2 ≥ n)
  i < rs.arrivals.size && rs.arrivals[i]!.2 == n

/-- Bytes of one phase of a slot, over all roles. -/
structure TxPhase where
  phase : Int
  expect : Int
  landed : Int
  /-- First transaction event of the phase. -/
  loc : Loc
  deriving Inhabited

/-- A tma_load not covered by earlier expect_tx bytes of its role in its phase; `phase =
none`: its role issued no expect_tx on the slot before it. -/
structure Uncovered where
  role : Nat
  phase : Option Int
  bytes : Int
  loc : Loc
  deriving Inhabited

/-- Lockstep contents of the waited complete phases (module doc): per entry of the slot's
`roles`, an arrival share per phase; with `init = some i`, phase 1 is the first `count`
arrivals of `roles[i]` and the shares cover the phases after it. Several share assignments
can fit the arrivals (e.g. 2+2+4 and 1+1+6 for count 8): `shares` is a representative one
(the most participating roles), `feasible` each role's shares over all of them whose zero
shares are evidenced: the assignments under which bytes may be attributed, and the least
share the lockstep credit counts. -/
structure Contents where
  shares : Array Int
  feasible : Array (Array Int)
  init : Option Nat
  deriving BEq, Inhabited

/-- Least share of `roles[i]` over the fitting assignments. -/
def Contents.least (c : Contents) (i : Nat) : Int := c.feasible[i]!.foldl min (c.feasible[i]!.getD 0 0)

structure SlotLedger where
  slot : Slot
  /-- Declared arrival count of the barrier. -/
  count : Int
  used : Bool
  firstLoc : Loc
  inits : Array (Int × Loc)
  arrivals : Int
  /-- Some arrival is a `commit_arrive` (asynchronous). -/
  async : Bool
  expectBytes : Int
  tmaBytes : Int
  /-- One entry per role that touched the slot, in role order. -/
  roles : Array RoleSlot
  /-- Roles whose code waits on this barrier array. -/
  staticWaiters : Array Nat
  /-- Some role's code initializes this barrier array: the slot needs exactly one
  `bar_init` in its CTA (an array no `bar_init` names, as in hand-written fixtures, is taken
  as initialized once before the roles start). -/
  initDeclared : Bool
  /-- Lockstep contents of the `lockstep` waited complete phases; `none` if none exist or no
  complete phase is waited for. -/
  contents : Option Contents
  /-- Transaction bytes per phase, sorted by phase (`attribute`). -/
  txPhases : Array TxPhase
  uncovered : Array Uncovered
  /-- Transaction events (role, loc) whose phase is unknown. -/
  unattributed : Array (Nat × Loc)
  /-- The cluster's classification reached a fixpoint (`ledger`); if not, P0-2 and P0-4 are
  not decided on the cluster (`unconverged`). -/
  converged : Bool := true
  /-- `TxOk` fails, but only because a lander's landings may complete later phases than
  their lower bounds (module doc, "Landers"): P0-4 is not decided on the slot. -/
  txUndecided : Bool := false
  deriving Inhabited

def SlotLedger.completions (L : SlotLedger) : Int := L.arrivals / L.count

/-- Highest phase some role waits for (0 if none). -/
def SlotLedger.waited (L : SlotLedger) : Int := L.roles.foldl (fun m rs => max m rs.required) 0

/-- The complete phases some role waits for, whose arrivals must have lockstep contents. -/
def SlotLedger.lockstep (L : SlotLedger) : Int := min L.waited L.completions

/-- Phases complete by arrivals alone: no transaction bytes and no commit arrivals. -/
def SlotLedger.plain (L : SlotLedger) : Bool := !L.async && L.expectBytes == 0 && L.tmaBytes == 0

/-- `role` waits, arrives or issues transaction bytes on the slot (a `bar_init` does not
count; a role that only arrives or issues bytes on it is one of its producers). -/
def SlotLedger.touches (L : SlotLedger) (role : Nat) : Bool :=
  L.roles.any fun rs => rs.role == role && !(rs.waits.isEmpty && rs.arrivals.isEmpty && rs.tx.isEmpty)

/-- Arrivals on the slot at trace positions before `cuts[role]`, over all roles. -/
def SlotLedger.arrivedBefore (L : SlotLedger) (cuts : Array Nat) : Int :=
  L.roles.foldl (fun n rs => n + rs.arrivedBefore cuts[rs.role]!) 0

/-- Arrivals of `roles[i]` that the completion of phase `k` needs: `k x count` minus all
other roles' arrivals over the run and, with `lockstep` (the lockstep credit, module doc),
at least its least share of the phases up to min(k, lockstep) of the lockstep contents. -/
def SlotLedger.needs (L : SlotLedger) (i : Nat) (k : Int) (lockstep : Bool) : Int :=
  let anyway := k * L.count - (L.arrivals - L.roles[i]!.total)
  match L.contents, lockstep with
  | some c, true =>
    let k := min k L.lockstep
    let credit := if k ≤ 0 then 0 else match c.init with
      | some j => (if j = i then L.count else 0) + (k - 1) * c.least i
      | none => k * c.least i
    max anyway credit
  | _, _ => anyway

/-- `needs`, raised by `may` (P1-1, `precise`): per entry of `roles`, its arrivals that may
precede the return of the wait (`mayBeforeReturn`). The completion of phase `k` before the
wait returned needs `k x count` arrivals that precede the return, so `roles[i]` issued at
least `k x count` minus the other roles' arrivals that may precede it. `may` empty: `needs`. -/
def SlotLedger.needsWith (L : SlotLedger) (i : Nat) (k : Int) (lockstep : Bool) (may : Array Int) : Int :=
  let n := L.needs i k lockstep
  if may.size != L.roles.size then n else max n (k * L.count - (may.foldl (· + ·) 0 - may[i]!))

/-- Largest share of `roles[i]` over the fitting assignments. -/
def Contents.most (c : Contents) (i : Nat) : Int := c.feasible[i]!.foldl max (c.feasible[i]!.getD 0 0)

/-- The arrivals of `roles[i]` that phase `k` contains under the lockstep reading at its most
generous (P1-1, F1): its largest fitting share of every phase up to `k`, capped at the
completed phases, after the initialization arrival if any. Mixing the largest shares of
several assignments over-approximates every one of them, so a pair this cannot order is
unordered under each fitting assignment. -/
def SlotLedger.lockNeeds (L : SlotLedger) (i : Nat) (k : Int) : Int :=
  match L.contents with
  | none => 0
  | some c =>
    let k := min k L.completions
    if k ≤ 0 then 0 else match c.init with
      | some j => (if j = i then L.count else 0) + (k - 1) * c.most i
      | none => k * c.most i

/-- Phase in which the next arrival of `roles[i]` falls after its first `a` arrivals, under
the lockstep contents with the share assignment `shares` (at most K + 1); every byte
belongs to phase 1 if no phase completes. -/
def SlotLedger.phaseOf (L : SlotLedger) (shares : Array Int) (i : Nat) (a : Int) : Option Int :=
  if L.completions ≤ 0 then some 1
  else match L.contents with
    | none => none
    | some c =>
      let (a, first) := match c.init with
        | some j => (if j = i then a - L.count else a, 2)
        | none => (a, 1)
      let sh := shares[i]!
      if a < 0 then some 1
      else if sh ≤ 0 then none
      else some (min (a / sh + first) (L.completions + 1))

/-- Apply `f` to the entry of `role` (created on the role's first event on the slot), then
`g` to the ledger. Each structure is consumed once, so array pushes stay in place. -/
def SlotLedger.record (L : SlotLedger) (loc : Loc) (role : Nat) (f : RoleSlot → RoleSlot)
    (g : SlotLedger → SlotLedger) : SlotLedger :=
  let L := if L.used then L else { L with used := true, firstLoc := loc }
  let fresh : RoleSlot := { role, waits := #[], arrivals := #[], tx := #[], firstCommit := none }
  let mine := match L.roles.back? with
    | some rs => rs.role == role
    | none => false
  g (if mine then { L with roles := L.roles.modify (L.roles.size - 1) f }
     else { L with roles := L.roles.push (f fresh) })

/-- `record` on slot number `i` of the ledger array. -/
def bump (L : Array SlotLedger) (i : Nat) (loc : Loc) (role : Nat) (f : RoleSlot → RoleSlot)
    (g : SlotLedger → SlotLedger) : Array SlotLedger :=
  L.modify i (·.record loc role f g)

/-! ## Happens-before -/

/-- A wait in its role's program order: (trace position, ledger slot, phase lower bound). -/
abbrev WaitPos := Nat × Nat × Int

/-- Arrivals on ledger slot `slot` that may happen before the event at position `pos` of
role `role`, or some number below `need` once that is established. Per role, `cuts` is
the first position from which every event is definitely after that event: initially the
role's own suffix; a wait joins (with the rest of its role) when the arrivals on its slot
before the cuts cannot complete its phase. An arrival counts at the event that issues it,
which is not later than the arrival itself. Each round lowers some cut to a wait position,
so `waits + 1` rounds reach the fixpoint. -/
def arrivalsMayPrecede (L : Array SlotLedger) (waits : Array (Array WaitPos)) (sizes : Array Nat)
    (role pos slot : Nat) (need : Int) : Int := Id.run do
  let mut cuts := sizes.modify role fun _ => pos
  for _ in [0:waits.foldl (· + ·.size) 1] do
    if L[slot]!.arrivedBefore cuts < need then break
    let mut changed := false
    for r in [0:waits.size] do
      for (p, s, k) in waits[r]! do
        if p ≥ cuts[r]! then break
        if 0 < k && L[s]!.arrivedBefore cuts < k * L[s]!.count then
          cuts := cuts.modify r fun _ => p
          changed := true
          break
    if !changed then break
  return L[slot]!.arrivedBefore cuts

/-- Per role, its waits grouped by ledger slot: (slot, (trace position, phase bound) in
program order). -/
abbrev WaitsBySlot := Array (Array (Nat × Array (Nat × Int)))

def WaitsBySlot.of (waits : Array (Array WaitPos)) : WaitsBySlot :=
  waits.map fun ws => ws.foldl (init := #[]) fun acc (p, s, k) =>
    match acc.findIdx? (·.1 == s) with
    | some j => acc.modify j fun (s, xs) => (s, xs.push (p, k))
    | none => acc.push (s, #[(p, k)])

/-- `arrivalsMayPrecede`'s cuts at their fixpoint for the return of the wait at position
`pos` of `role` (P1-1, `precise`): per role, the first trace position from which every
event surely follows that return. Initially `role`'s own suffix from the wait; a wait of a
role joins (with the rest of its role) when the arrivals on its slot before the cuts cannot
complete its phase bound, since it cannot return before the wait does. The phase bounds of
a role's waits on one slot do not decrease in program order, so the first such wait per
(role, slot) is found by binary search. Each round but the last lowers some cut to an
earlier wait, so `waits + 1` rounds reach the fixpoint (stopping earlier would only leave
cuts higher, which is still sound). -/
def returnCuts (L : Array SlotLedger) (ws : WaitsBySlot) (sizes : Array Nat) (role pos : Nat) : Array Nat := Id.run do
  let mut cuts := sizes.modify role fun _ => pos
  for _ in [0:ws.foldl (fun n r => r.foldl (fun n x => n + x.2.size) n) 1] do
    let mut changed := false
    for r in [0:ws.size] do
      for (s, xs) in ws[r]! do
        let l := L[s]!
        let before := l.arrivedBefore cuts
        let i := searchFirst xs fun (_, k) => before < k * l.count
        match xs[i]? with
        | some (p, _) =>
          if p < cuts[r]! then
            cuts := cuts.modify r fun _ => p
            changed := true
        | none => pure ()
    if !changed then break
  return cuts

/-- Per wait of `waits` (indexed alike), the arrivals of each entry of its slot's `roles`
that may precede its return (`returnCuts`), for `SlotLedger.needsWith`; empty on a slot
with at most one arriving role, where they add nothing to `needs`. -/
def mayBeforeReturn (L : Array SlotLedger) (waits : Array (Array WaitPos)) (sizes : Array Nat) :
    Array (Array (Array Int)) :=
  let ws := WaitsBySlot.of waits
  waits.mapIdx fun r xs => xs.map fun (p, s, _) =>
    let l := L[s]!
    if (l.roles.filter (!·.arrivals.isEmpty)).size ≤ 1 then #[] else
    let cuts := returnCuts L ws sizes r p
    l.roles.map fun rs => rs.arrivedBefore cuts[rs.role]!

/-- Grow per-role trace prefixes `done` that surely happened before some event to a
fixpoint: a wait inside them returned, so its slot completed its phase bound k, and every
arriving role issued the arrivals phase k needs (`SlotLedger.needs`, with the lockstep
credit if `lockstep`; `needsWith` and the wait's entry of `may` if given), which puts its
prefix up to the last of them in `done`. Returns `(done, applied, every)`. With `warps`
(P1-1, `precise`: per role its warps), `done` is the prefix some warp of the role surely
issued (its waits there returned) and `every` the prefix every warp of it surely issued
(its arrivals there happened; `creditArrivals`, module doc "Partial credit"), grown from
the given `every`. Without `warps` (P0-2, P0-4), a role's arrive event counts as all its
warps' and `every` is `done`. `applied` counts, per role, the waits already applied (a
prefix of its waits). Each round applies at least one new wait, so `waits + 1` rounds
suffice. -/
def saturate (L : Array SlotLedger) (waits : Array (Array WaitPos)) (lockstep : Bool) (done applied : Array Nat)
    (may : Array (Array (Array Int)) := #[]) (warps : Array Nat := #[]) (every : Array Nat := #[]) :
    Array Nat × Array Nat × Array Nat := Id.run do
  let mut done := done
  let mut every := if every.isEmpty then Array.replicate done.size 0 else every
  let mut applied := applied
  for _ in [0:waits.foldl (· + ·.size) 1] do
    let mut changed := false
    for r in [0:waits.size] do
      for i in [applied[r]!:waits[r]!.size] do
        let (p, s, k) := waits[r]![i]!
        if p ≥ done[r]! then break
        applied := applied.modify r (· + 1)
        changed := true
        let l := L[s]!
        let mw := (may[r]?.bind (·[i]?)).getD #[]
        for (rs, j) in l.roles.zipIdx do
          let w := warps.getD rs.role 1
          let (one, all) := creditArrivals rs.arrivals w (warpShare w lockstep) (l.needsWith j k lockstep mw)
          done := done.modify rs.role (max · one)
          every := every.modify rs.role (max · all)
    if !changed then break
  return (done, applied, if warps.isEmpty then done else every)

/-- Per role, the length of its trace prefix that surely happened before the event at
position `pos` of role `role`, with the lockstep credit: the evidence for zero shares and
initialization arrivals (`decompose`), which can only make lockstep contents exist. -/
def surelyBefore (L : Array SlotLedger) (waits : Array (Array WaitPos)) (role pos : Nat) : Array Nat :=
  (saturate L waits true ((Array.replicate waits.size 0).modify role fun _ => pos) (Array.replicate waits.size 0)).1

/-- `saturate` before every position of `ps` (ascending) of role `role`, incrementally:
the prefixes before a later event of the role contain those before an earlier one. Per
position `(done, every)` (`saturate`); the event's own role is in `every` up to it only if
it is one warp. -/
def surelyBeforeAll (L : Array SlotLedger) (waits : Array (Array WaitPos)) (lockstep : Bool) (role : Nat)
    (ps : Array Nat) (may : Array (Array (Array Int)) := #[]) (warps : Array Nat := #[]) :
    Array (Array Nat × Array Nat) := Id.run do
  let mut done := Array.replicate waits.size 0
  let mut every := Array.replicate waits.size 0
  let mut applied := Array.replicate waits.size 0
  let mut out := Array.emptyWithCapacity ps.size
  for p in ps do
    let own := if warps.getD role 1 ≤ 1 then every.modify role (max · p) else every
    let (d, a, e) := saturate L waits lockstep (done.modify role (max · p)) applied may warps own
    done := d
    applied := a
    every := e
    out := out.push (done, every)
  return out

/-- Phase `phase` of ledger slot `slot` completed within the prefixes `done` (which surely
happened before some event): a wait on the slot in them has a phase bound of at least
`phase`, or, if `byArrivals` (the slot's phases complete by arrivals alone), the prefixes
`every` (`done` if empty; `saturate`) hold `phase x count` arrivals. Returns the event
that shows it. -/
def completedIn (L : Array SlotLedger) (waits : Array (Array WaitPos)) (locAt : Nat → Nat → Loc)
    (done : Array Nat) (slot : Nat) (phase : Int) (byArrivals : Bool) (every : Array Nat := #[]) :
    Option Evidence := Id.run do
  for r in [0:waits.size] do
    for (p, s, k) in waits[r]! do
      if p ≥ done[r]! then break
      if s == slot && k ≥ phase then return some { role := r, loc := locAt r p, wait := true }
  let l := L[slot]!
  let need := phase * l.count
  let every := if every.isEmpty then done else every
  if !byArrivals || need ≤ 0 || l.arrivedBefore every < need then return none
  -- the arrive event, in the role with the most arrivals in `every`, at which they reach `need`
  let ins : Array Int := l.roles.map fun rs => rs.arrivedBefore every[rs.role]!
  let all := ins.foldl (· + ·) 0
  let best := (Array.range ins.size).foldl (fun (b j : Nat) => if ins[j]! > ins[b]! then j else b) 0
  let rs := l.roles[best]!
  let a := rs.firstArrival (·.2 ≥ need - (all - ins[best]!))
  return rs.arrivals[a]?.map fun (p, _) => { role := rs.role, loc := locAt rs.role p, wait := false }

/-- The highest phase bound of a wait on ledger slot `slot` within the prefixes `done` (0 if
none): on a slot with transaction bytes, the phases that surely completed before the event
the prefixes precede (`completedIn`). -/
def completedUpTo (waits : Array (Array WaitPos)) (done : Array Nat) (slot : Nat) : Int := Id.run do
  let mut k : Int := 0
  for r in [0:waits.size] do
    for (p, s, b) in waits[r]! do
      if p ≥ done[r]! then break
      if s == slot then k := max k b
  return k

/-! ## Phase classification -/

/-- One classification round (module doc): the phase of every wait from the phase lower
bounds `bounds` of the previous round and the happens-before prefixes before each wait,
computed over them: `hb role pos` without and `hbCredited role pos` with the lockstep
credit, each `(done, every)` (`saturate`). -/
def classify (L : Array SlotLedger) (bounds : Array (Array WaitPos)) (sizes : Array Nat)
    (hb hbCredited : Nat → Nat → Array Nat × Array Nat) (locAt : Nat → Nat → Loc) (precise : Bool := false) :
    Array SlotLedger :=
  L.mapIdx fun i l => if !l.used then l else
    { l with roles := l.roles.map fun rs => Id.run do
      let surely (w : WaitRec) (k : Int) : Option Evidence :=
        let (d, e) := hb rs.role w.pos
        completedIn L bounds locAt d i k l.plain e
      -- with the lockstep credit: only turns a stale wait into a presumed skip
      let presumably (w : WaitRec) (k : Int) : Bool :=
        let (d, e) := hbCredited rs.role w.pos
        (completedIn L bounds locAt d i k l.plain e).isSome
      -- (k', lap): the first of k, k+2, ... whose next phase is not sure to complete before
      -- the wait, which therefore returns at k' or later; lap: (k, why phase k+1 did)
      let settle (w : WaitRec) (k : Int) : Int × Option (Int × Evidence) := Id.run do
        let mut k' := k
        let mut lap := none
        for _ in [0:l.completions.toNat / 2 + 1] do
          match surely w (k' + 1) with
          | some e =>
            k' := k' + 2
            lap := lap.orElse fun _ => some (k, e)
          | none => break
        return (k', lap)
      -- P1-1 (`precise`): the phase lower bound under the lockstep reading, from the role's
      -- previous one on the slot (+1 after a parity change), with the laps the credit shows
      let lock (w : WaitRec) (prev : Option WaitRec) : Int := Id.run do
        if !precise then return w.assumedPhase
        let mut k := match prev with
          | some p => max w.assumedPhase (p.lockPhase + (if p.parity != w.parity then 1 else 0))
          | none => w.assumedPhase
        for _ in [0:l.completions.toNat / 2 + 1] do
          if presumably w (k + 1) then k := k + 2 else break
        return k
      let mut ws : Array WaitRec := #[]
      for w in rs.waits do
        let w : WaitRec := { w with presumed := false, stale := none, lapped := none, blocked := none }
        let w' : WaitRec := match ws.back? with
        | none =>
          let (k, _) := settle w (if w.parity = 1 then 0 else 1)
          { w with minPhase := k, assumedPhase := k, phase := k }
        | some prev =>
          if w.parity != prev.parity then
            let (k, why) := settle w (prev.minPhase + 1)
            { w with minPhase := k, assumedPhase := max (prev.assumedPhase + 1) k,
                     phase := max (prev.phase + 1) k, lapped := why }
          else
            let need := (prev.minPhase + 1) * l.count
            let may := arrivalsMayPrecede L bounds sizes rs.role w.pos i need
            if may < need then
              { w with minPhase := prev.minPhase, assumedPhase := prev.assumedPhase, phase := prev.phase }
            else if (surely w (prev.minPhase + 1)).isSome then
              let (k, why) := settle w (prev.minPhase + 2)
              { w with minPhase := k, assumedPhase := max (prev.assumedPhase + 2) k, phase := max (prev.phase + 2) k,
                       lapped := why }
            else if presumably w (prev.minPhase + 1) then
              { w with minPhase := prev.minPhase, assumedPhase := prev.assumedPhase + 2, phase := prev.phase + 2,
                       presumed := true }
            else
              { w with minPhase := prev.minPhase, assumedPhase := prev.assumedPhase, phase := prev.phase + 1,
                       stale := some (may, need) }
        ws := ws.push { w' with lockPhase := lock w' ws.back? }
      return { rs with waits := ws } }

/-- Phase bounds `bound` (`minPhase` or `assumedPhase`) of every role's waits, in program
order. -/
def waitBounds (L : Array SlotLedger) (roles : Nat) (bound : WaitRec → Int) : Array (Array WaitPos) := Id.run do
  let mut out := Array.replicate roles #[]
  for (l, i) in L.zipIdx do
    for rs in l.roles do
      for w in rs.waits do
        out := out.modify rs.role (·.push (w.pos, i, bound w))
  return out.map (·.qsort (·.1 < ·.1))

/-! ## Lockstep contents and transaction bytes per phase -/

/-- Share candidates of a role over `W ≥ 1` waited phases: arrival counts `v ≤ C` that the
role's arrivals reach exactly at its arrive events after `v, 2v, ..., W*v`; and 0. -/
def RoleSlot.shareCandidates (rs : RoleSlot) (C W : Int) : Array Int :=
  let fits (v : Int) := v ≤ C && W * v ≤ rs.total && (List.range W.toNat).all fun j => rs.reaches ((j + 1) * v)
  ((rs.arrivals.map (·.2)).filter fits).push 0

/-- `sums[i]`: the distinct share sums (at most `C`, ascending) of roles i..n-1 over their
admissible shares `cands`. -/
def suffixSums (cands : Array (Array Int)) (C : Int) : Array (Array Int) := Id.run do
  let n := cands.size
  let mut sums := Array.replicate (n + 1) #[(0 : Int)]
  for j in [0:n] do
    let i := n - 1 - j
    sums := sums.set! i (dedup ((sums[i + 1]!.flatMap fun t => cands[i]!.map (· + t)).filter (· ≤ C)))
  return sums

/-- One share per role from `cands` (its admissible shares, all in `[0, C]`) that sum to
`C`, among several one with the most participating roles. -/
def chooseShares (cands : Array (Array Int)) (C : Int) : Option (Array Int) := Id.run do
  -- reachable share sum -> (participating roles, shares so far)
  let mut reach : Array (Int × Nat × Array Int) := #[(0, 0, #[])]
  for cs in cands do
    let mut next : Array (Int × Nat × Array Int) := #[]
    for (sum, n, sh) in reach do
      for v in cs do
        let s := sum + v
        let m := if v > 0 then n + 1 else n
        if s ≤ C then
          match next.findIdx? (·.1 == s) with
          | some j => if m > next[j]!.2.1 then next := next.set! j (s, m, sh.push v)
          | none => next := next.push (s, m, sh.push v)
    reach := next
  return (reach.find? (·.1 == C)).map (·.2.2)

/-- Per role, its shares from `cands` that take part in some choice of one share per role
summing to `C`. -/
def feasibleShares (cands : Array (Array Int)) (C : Int) : Array (Array Int) := Id.run do
  let rest := suffixSums cands C
  -- the share sums of the roles before the current one
  let mut before : Array Int := #[0]
  let mut out := Array.emptyWithCapacity cands.size
  for (cs, i) in cands.zipIdx do
    out := out.push (cs.filter fun v => before.any fun s => rest[i + 1]!.contains (C - s - v))
    before := dedup ((before.flatMap fun s => cs.map (s + ·)).filter (· ≤ C))
  return out

/-- At most this many share assignments are tried to attribute a slot's transaction bytes
(`SlotLedger.attribute`). -/
def maxShareVectors : Nat := 64

/-- The choices of one share per role from `feasible` that sum to `C`, distinct on the roles
marked in `keep` (the other roles' entries are 0), at most `max` of them. -/
def shareVectors (feasible : Array (Array Int)) (C : Int) (keep : Array Bool) (max : Nat) :
    Array (Array Int) := Id.run do
  let rest := suffixSums feasible C
  -- (share sum of the roles so far, their kept shares), each completing to `C`
  let mut states : Array (Int × Array Int) := #[(0, #[])]
  for (cs, i) in feasible.zipIdx do
    let mut next : Array (Int × Array Int) := #[]
    let mut kept : Array (Array Int) := #[]
    for (s, sh) in states do
      for v in cs do
        let sh := sh.push (if keep[i]! then v else 0)
        if rest[i + 1]!.contains (C - s - v) && !next.contains (s + v, sh) && (kept.contains sh || kept.size < max) then
          kept := if kept.contains sh then kept else kept.push sh
          next := next.push (s + v, sh)
    states := next
  return states.map (·.2)

/-- Lockstep shares over `W ≥ 1` waited phases: a representative choice (`chooseShares`) and
each role's feasible shares (`feasibleShares`). Roles without arrivals take share 0; a role
with arrivals takes share 0 only if `zeroOk` holds for it, in the representative only if no
choice without such zeros exists. -/
def lockstepShares (roles : Array RoleSlot) (C W : Int) (zeroOk : RoleSlot → Bool) :
    Option (Array Int × Array (Array Int)) := do
  let cands (zero : RoleSlot → Bool) := roles.map fun rs => (rs.shareCandidates C W).filter (· != 0 || zero rs)
  let evidenced := cands fun rs => rs.arrivals.isEmpty || zeroOk rs
  let shares ← (chooseShares (cands (·.arrivals.isEmpty)) C).orElse fun _ => chooseShares evidenced C
  return (shares, feasibleShares evidenced C)

/-- Lockstep contents of `l` over `W ≥ 1` waited phases (module doc). `surely role pos k`:
phase `k` of the slot surely completed before the event at `pos` of `role`; `before a b`:
the event `a` (role, position) surely precedes the event `b`. The initialization arrival is
tried only if no shares fit all waited phases. -/
def lockstepContents (l : SlotLedger) (W : Int) (surely : Nat → Nat → Int → Bool)
    (before : Nat × Nat → Nat × Nat → Bool) : Option Contents := Id.run do
  let zeroOk (rs : RoleSlot) : Bool := match rs.arrivals[0]? with
    | some (p, _) => surely rs.role p W
    | none => true
  if let some (shares, feasible) := lockstepShares l.roles l.count W zeroOk then
    return some { shares, feasible, init := none }
  for (rs, i) in l.roles.zipIdx do
    let a := rs.firstArrival (·.2 ≥ l.count)
    if a < rs.arrivals.size && rs.arrivals[a]!.2 == l.count then
      let e := (rs.role, rs.arrivals[a]!.1)
      let plain := rs.firstCommit.all (a < ·)
      let alone := l.roles.all fun o => o.role == rs.role || match o.arrivals[0]? with
        | some (p, _) => surely o.role p 1 || (plain && before e (o.role, p))
        | none => true
      if alone then
        let rest := l.roles.modify i fun rs =>
          { rs with arrivals := (rs.arrivals.extract (a + 1) rs.arrivals.size).map fun (p, n) => (p, n - l.count) }
        let sh := if W = 1 then some (rest.map fun _ => 0, rest.map fun _ => #[0])
          else lockstepShares rest l.count (W - 1) zeroOk
        if let some (shares, feasible) := sh then return some { shares, feasible, init := some i }
  return none

/-- Lockstep contents of every used slot from the waited phases of `L`'s classification.
Their evidence is taken over `bounds`, with credit only from slots whose contents need none
and never from the slot itself (module doc). -/
def decompose (L : Array SlotLedger) (bounds : Array (Array WaitPos)) (locAt : Nat → Nat → Loc) :
    Array SlotLedger :=
  let assumed := L.map fun l => match l.contents with
    | some c =>
      if c.init.isNone && l.roles.zipIdx.all (fun (rs, j) => 0 < c.least j || rs.arrivals.isEmpty) then l
      else { l with contents := none }
    | none => l
  L.mapIdx fun i l =>
    let W := l.lockstep
    if !l.used || W ≤ 0 then { l with contents := none } else
    let E := assumed.modify i ({ · with contents := none })
    let surely (role pos : Nat) (k : Int) : Bool :=
      (completedIn E bounds locAt (surelyBefore E bounds role pos) i k l.plain).isSome
    let before (a b : Nat × Nat) : Bool := (surelyBefore E bounds b.1 b.2)[a.1]! > a.2
    { l with contents := lockstepContents l W surely before }

/-- Add bytes to phase `p` of a per-phase table. -/
def addTx (ps : Array TxPhase) (p expect landed : Int) (loc : Loc) : Array TxPhase :=
  match ps.findIdx? (·.phase == p) with
  | some j => ps.modify j fun x => { x with expect := x.expect + expect, landed := x.landed + landed }
  | none => ps.push { phase := p, expect, landed, loc }

/-- Attribute the slot's transaction bytes to phases under the share assignment `shares`
(module doc, `SlotLedger.phaseOf`) and the landers' lower bounds (module doc, "Landers");
nothing is attributed on a slot nobody waits on. -/
def SlotLedger.attributeWith (L : SlotLedger) (shares : Array Int) : SlotLedger := Id.run do
  if L.waited ≤ 0 then return L
  let mut phases : Array TxPhase := #[]
  let mut uncovered : Array Uncovered := #[]
  let mut unattributed : Array (Nat × Loc) := #[]
  for (rs, i) in L.roles.zipIdx do
    if rs.lander then
      -- a lander: the first phase its landing can complete (module doc)
      for x in rs.tx do
        phases := addTx phases (min (x.floor + 1) (L.completions + 1)) 0 x.bytes x.loc
      continue
    -- phase of the role's latest expect_tx (`some none`: unattributed)
    let mut cur : Option (Option Int) := none
    -- per phase: this role's expect_tx bytes minus its tma_load bytes so far
    let mut bal : Array (Int × Int) := #[]
    for x in rs.tx do
      let ph := if x.load then cur else some (L.phaseOf shares i x.arrivedBefore)
      match ph with
      | none => uncovered := uncovered.push { role := rs.role, phase := none, bytes := x.bytes, loc := x.loc }
      | some none => unattributed := unattributed.push (rs.role, x.loc)
      | some (some p) =>
        let b := ((bal.find? (·.1 == p)).map (·.2)).getD 0 + (if x.load then -x.bytes else x.bytes)
        bal := (bal.filter (·.1 != p)).push (p, b)
        if x.load then
          phases := addTx phases p 0 x.bytes x.loc
          if b < 0 then uncovered := uncovered.push { role := rs.role, phase := some p, bytes := -b, loc := x.loc }
        else
          phases := addTx phases p x.bytes 0 x.loc
      if !x.load then cur := ph
  return { L with txPhases := phases.qsort (·.phase < ·.phase), uncovered, unattributed }

/-- P0-4's transaction clauses (Barrier.lean) on the attributed bytes: in every waited phase
the expect_tx bytes equal the TMA bytes landing in it, no TMA bytes land in a phase nobody
waits for or on a slot nobody waits on, every load is covered by an earlier expect_tx of
its role and every transaction event has a phase. -/
def SlotLedger.TxOk (L : SlotLedger) : Prop :=
  (∀ p ∈ L.txPhases, p.phase ≤ L.waited → p.expect = p.landed) ∧
  (∀ p ∈ L.txPhases, L.waited < p.phase → p.landed = 0) ∧
  (L.waited ≤ 0 → L.tmaBytes = 0) ∧ L.uncovered.size = 0 ∧ L.unattributed.size = 0

instance (L : SlotLedger) : Decidable L.TxOk := by unfold SlotLedger.TxOk; infer_instance

/-- `TxOk` fails under this attribution only through landers' bytes, whose phases are lower
bounds (module doc, "Landers"): no clause of the slot's own roles fails, no byte is put in a
phase nobody waits for, the waited phases receive the bytes they expect in all, and the
first unbalanced one receives more. -/
def SlotLedger.landerUnsure (L : SlotLedger) : Bool :=
  let w := L.txPhases.filter (·.phase ≤ L.waited)
  let total (f : TxPhase → Int) := w.foldl (· + f ·) 0
  L.roles.any (·.lander) && L.uncovered.isEmpty && L.unattributed.isEmpty &&
    L.txPhases.all (fun p => p.phase ≤ L.waited || p.landed == 0) && total (·.expect) == total (·.landed) &&
    (w.find? fun p => p.expect != p.landed).any fun p => p.landed > p.expect

/-- Attribute the slot's transaction bytes under the representative shares or, if that
violates `TxOk`, under the first other fitting assignment that does not (module doc). If none
does, the slot is not decided (`txUndecided`) under the first assignment that fails only
through landers' lower bounds (`landerUnsure`), so it lists none of the transaction lines
that some assignment does not certainly fail; failing that, the bytes stay attributed under
the representative one (a certain failure). -/
def SlotLedger.attribute (L : SlotLedger) : SlotLedger := Id.run do
  let arrivals : Array (Array Int) := match L.contents with
    | none => #[#[]]
    | some c =>
      #[c.shares] ++ (shareVectors c.feasible L.count (L.roles.map (!·.tx.isEmpty)) maxShareVectors)
  let first := L.attributeWith (arrivals.getD 0 #[])
  if decide first.TxOk then return first
  let mut unsure : Option SlotLedger := if first.landerUnsure then some first else none
  for sh in arrivals.extract 1 arrivals.size do
    let x := L.attributeWith sh
    if decide x.TxOk then return x
    if unsure.isNone && x.landerUnsure then unsure := some x
  return match unsure with
    | some u => { u with txUndecided := true }
    | none => first

/-! ## Optimistic progress -/

/-- Waits that never return (module doc) for the per-role phase bounds `bounds`: per role
still blocked when nothing can move, (role, wait position, arrivals that reached its slot,
arrivals its phase bound needs). Each round moves some role past a wait, so `waits + 1` rounds suffice. -/
def deadlocks (L : Array SlotLedger) (bounds : Array (Array WaitPos)) (sizes : Array Nat) :
    Array (Nat × Nat × Int × Int) := Id.run do
  let n := bounds.size
  let cuts (next : Array Nat) : Array Nat :=
    (Array.range n).map fun r => (bounds[r]![next[r]!]?.map (·.1)).getD sizes[r]!
  let mut next := Array.replicate n 0
  for _ in [0:bounds.foldl (· + ·.size) 1] do
    let mut moved := false
    for r in [0:n] do
      for _ in [0:bounds[r]!.size] do
        match bounds[r]![next[r]!]? with
        | some (_, s, k) =>
          if L[s]!.arrivedBefore (cuts next) < k * L[s]!.count then break
          next := next.modify r (· + 1)
          moved := true
        | none => break
    if !moved then break
  let c := cuts next
  return (Array.range n).filterMap fun r =>
    bounds[r]![next[r]!]?.map fun (p, s, k) => (r, p, L[s]!.arrivedBefore c, k * L[s]!.count)

/-! ## The ledger -/

mutual
/-- Names of the barriers a statement waits on, or with `init` initializes (in any branch or
loop body). -/
def Stmt.bars (init : Bool) : Stmt → List String
  | .event (.wait bar _ _ _ _) => if init then [] else [bar]
  | .event (.bar_init bar _ _ _) => if init then [bar] else []
  | .«for» _ _ _ _ body _ => barsOf init body
  | .«while» _ body _ _ => barsOf init body
  | .«if» _ t e _ => barsOf init t ++ barsOf init e
  | _ => []

def barsOf (init : Bool) : List Stmt → List String
  | [] => []
  | s :: ss => s.bars init ++ barsOf init ss
end

/-- Per barrier array of the host, the roles whose code waits on it. -/
def staticWaiters (h : Host) : Array (Array Nat) :=
  let bars := h.kernel.roles.toArray.map (barsOf false ·.body)
  h.barriers.map fun b => (Array.range bars.size).filter fun r => bars[r]!.contains b.decl.name

/-- Per barrier array of the host, whether some role's code initializes it. -/
def initDeclared (h : Host) : Array Bool :=
  let bars := h.kernel.roles.flatMap (barsOf true ·.body)
  h.barriers.map (bars.contains ·.decl.name)

/-- The CTAs of one cluster, by rank, as one trace (module doc, "Clusters"): role `g` of the
cluster is role `g % n` (n: the kernel's roles) of the CTA of rank `g / n`. -/
structure ClusterTrace where
  /-- Cluster coordinates (the CTA's program id without clusters). -/
  cid : Nat × Nat × Nat
  /-- The member CTAs, `ctas[r].rank = r`. -/
  ctas : Array CtaTrace

/-- The roles of every CTA of the cluster, renumbered `rank x n + role`. -/
def ClusterTrace.roles (c : ClusterTrace) (n : Nat) : Array RoleTrace :=
  c.ctas.flatMap fun ct => ct.roles.map fun rt => { rt with role := ct.rank * n + rt.role }

/-- Ledger of every slot touched in cluster `c`, in (CTA rank, barrier, index) order. With
`precise` (P1-1 only; P0-2 and P0-4 use the default), happens-before also uses `needsWith`
(the other roles' arrivals that may precede a wait's return instead of their totals over the
run) and every wait gets its `lockPhase`. The landers' lower bounds (`TxRec.floor`) use
`needsWith` either way (module doc, "Landers"). -/
def ledger (h : Host) (c : ClusterTrace) (precise : Bool := false) : Array SlotLedger := Id.run do
  let nr := h.kernel.roles.length
  let nb := h.barriers.size
  let roles := c.roles nr
  let mut base : Array Nat := #[]
  let mut L : Array SlotLedger := #[]
  for rank in [0:c.ctas.size] do
    for (b, bi) in h.barriers.zipIdx do
      base := base.push L.size
      for i in [0:b.num] do
        L := L.push { slot := { bar := bi, idx := i, cta := rank }, count := b.count, used := false,
                      firstLoc := "", inits := #[], arrivals := 0, async := false, expectBytes := 0,
                      tmaBytes := 0, roles := #[], staticWaiters := #[], initDeclared := false,
                      contents := none, txPhases := #[], uncovered := #[], unattributed := #[] }
  let slot (s : Slot) : Nat := base[s.cta * nb + s.bar]! + s.idx
  let tx (load : Bool) (n : Int) (loc : Loc) (pos : Nat) (rs : RoleSlot) : RoleSlot :=
    { rs with tx := rs.tx.push { load, bytes := n, loc, arrivedBefore := rs.total, pos } }
  for rt in roles do
    let r := rt.role
    for (e, pos) in rt.events.zipIdx do
      match e.ev with
      | .barInit s n => L := bump L (slot s) e.loc r id fun l => { l with inits := l.inits.push (n, e.loc) }
      | .arrive s n async =>
        L := bump L (slot s) e.loc r
          (fun rs => { rs with arrivals := rs.arrivals.push (pos, rs.total + n),
                               firstCommit := rs.firstCommit.orElse fun _ => if async then some rs.arrivals.size else none })
          fun l => { l with arrivals := l.arrivals + n, async := l.async || async }
      | .expectTx s n =>
        L := bump L (slot s) e.loc r (tx false n e.loc pos) fun l => { l with expectBytes := l.expectBytes + n }
      | .wait s p =>
        let w : WaitRec := { parity := p, loc := e.loc, pos, minPhase := 0, assumedPhase := 0, phase := 0,
                             presumed := false, stale := none, lapped := none, blocked := none }
        L := bump L (slot s) e.loc r (fun rs => { rs with waits := rs.waits.push w }) id
      | .tmaLoad _ _ s n | .landTx s n =>
        L := bump L (slot s) e.loc r (tx true n e.loc pos) fun l => { l with tmaBytes := l.tmaBytes + n }
      | _ => pure ()
  -- Round 0 knows no other waits and no contents, so only each role's own events order
  -- anything; its bounds are sound, and every later round uses the bounds and contents of
  -- the round before.
  let n := roles.size
  let sizes := roles.map (·.events.size)
  let locAt (r p : Nat) : Loc := ((roles[r]?.bind (·.events[p]?)).map (·.loc)).getD ""
  -- P1-1: partial credit of a multi-warp role's arrive events (module doc)
  let warps := if precise then (Array.range n).map fun g => (h.kernel.roles[g % nr]?.map (·.warps)).getD 1 else #[]
  let positions := (waitBounds L n (·.minPhase)).map (·.map (·.1))
  -- what a round reads of the round before: the bounds, every slot's contents and waits
  let state (b : Array (Array WaitPos)) (L : Array SlotLedger) := (b, L.map fun l => (l.contents, l.roles.map (·.waits)))
  let mut bounds : Array (Array WaitPos) := Array.replicate n #[]
  let mut cur := L
  let mut prev := state bounds cur
  let mut converged := false
  for _ in [0:positions.foldl (· + ·.size) 2] do
    -- per role and wait position, the prefixes surely before it, without and with the credit
    let may := if precise then mayBeforeReturn cur bounds sizes else #[]
    let closure (lockstep : Bool) :=
      (Array.range n).map fun r => surelyBeforeAll cur bounds lockstep r positions[r]! may warps
    let (sound, credited) := (closure false, closure true)
    let before (hbs : Array (Array (Array Nat × Array Nat))) (r p : Nat) : Array Nat × Array Nat :=
      hbs[r]![searchFirst positions[r]! (· ≥ p)]!
    let classified := classify cur bounds sizes (before sound) (before credited) locAt precise
    let next := waitBounds classified n (·.minPhase)
    let decomposed := decompose classified next locAt
    let (here, after) := (state bounds cur, state next decomposed)
    converged := after == here
    -- back to the state of two rounds before: a 2-cycle, which never converges
    let cycle := after == prev
    prev := here
    bounds := next
    cur := decomposed
    if converged || cycle then break
  let blocked := deadlocks cur (waitBounds cur n (·.assumedPhase)) sizes
  let sw := staticWaiters h
  let inited := initDeclared h
  let mark (rs : RoleSlot) : RoleSlot := { rs with waits := rs.waits.map fun w =>
    { w with blocked := (blocked.find? fun (r, p, _) => r == rs.role && p == w.pos).map (·.2.2) } }
  let fin := cur.map fun l => { l with roles := l.roles.map fun rs => { mark rs with remote := rs.role / nr != l.slot.cta } }
  -- Landers (module doc): per role the positions of its landings, ascending, and the
  -- prefixes that surely happened before each (sound happens-before, precise needs)
  let landings := fin.foldl (init := Array.replicate n #[]) fun acc l =>
    l.roles.foldl (init := acc) fun acc rs => if rs.lander then acc.modify rs.role (· ++ rs.tx.map (·.pos)) else acc
  let landings := landings.map fun ps => ps.qsort (· < ·)
  let mayF := if landings.all (·.isEmpty) then #[] else mayBeforeReturn fin bounds sizes
  let hb := (Array.range n).map fun r => if landings[r]!.isEmpty then #[] else surelyBeforeAll fin bounds false r landings[r]! mayF
  let floor (i r p : Nat) : Int := completedUpTo bounds (hb[r]![searchFirst landings[r]! (· ≥ p)]!).1 i
  return (fin.zipIdx.filter (·.1.used)).map fun (l, i) =>
    SlotLedger.attribute { l with staticWaiters := sw[l.slot.bar]!.map (· + l.slot.cta * nr),
                                  initDeclared := inited[l.slot.bar]!, converged,
                                  roles := l.roles.map fun rs => if !rs.lander then rs else
                                    { rs with tx := rs.tx.map fun x => { x with floor := floor i rs.role x.pos } } }

/-- The clusters of the run whose roles all ran to their end (a trace cut short at an
out-of-bounds metadata load has no meaningful ledger), in the order of their first CTA, each
with its CTAs by rank. -/
def Run.clusters (r : Run) : Array ClusterTrace := Id.run do
  let n := r.host.csize
  let mut idx : Std.HashMap (Nat × Nat × Nat) Nat := {}
  let mut out : Array ClusterTrace := #[]
  for c in r.ctas do
    let cid := r.host.clusterOf c.pid
    match idx[cid]? with
    | some i => out := out.modify i fun cl => { cl with ctas := cl.ctas.push c }
    | none =>
      idx := idx.insert cid out.size
      out := out.push { cid, ctas := #[c] }
  return (out.map fun cl => { cl with ctas := cl.ctas.qsort (·.rank < ·.rank) }).filter fun cl =>
    cl.ctas.size == n && cl.ctas.all (·.complete) && (cl.ctas.zipIdx.all fun (c, i) => c.rank == i)

/-- Barrier ledgers of the complete clusters (`Run.clusters`), with the cluster's coordinates
(the CTA's program id without clusters). -/
abbrev Ledgers := Array ((Nat × Nat × Nat) × Array SlotLedger)

def Run.ledgers (r : Run) : Ledgers := r.clusters.map fun c => (c.cid, ledger r.host c)

/-- The ledgers P1-1 uses (`ledger` with `precise`). -/
def Run.ledgersP1 (r : Run) : Ledgers := r.clusters.map fun c => (c.cid, ledger r.host c true)

/-- `f`'s lines about every slot ledger of every cluster in `Ls` whose ledger converged (the
others get an `unconverged` line instead). -/
def forSlots (Ls : Ledgers) (f : Nat × Nat × Nat → SlotLedger → Array Issue) : Array Issue :=
  Ls.foldl (init := #[]) fun acc (pid, ls) => ls.foldl (init := acc) fun acc L =>
    if L.converged then acc ++ f pid L else acc

/-- The `not decided` line of P0-2 and P0-4 for each cluster in `Ls` whose ledger did not
converge. -/
def unconverged (h : Host) (Ls : Ledgers) : Array Issue :=
  Ls.filterMap fun (pid, ls) => if ls.all (·.converged) then none else
    some { loc := "", context := s!"{if h.csize ≤ 1 then "cta" else "cluster"} {pidStr pid}",
           msg := s!"not decided on this input: the barrier ledger of this {if h.csize ≤ 1 then "CTA" else "cluster"} does not converge (its last round still changes the phase classification or the lockstep contents), so its {ls.size} slot(s) are not checked; the lines of this spec that are not `not decided` lines are certain counterexamples" }

end KV
