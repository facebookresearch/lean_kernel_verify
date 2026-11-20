/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
P0-6: named barriers convergent (references/specs.md), decided on the P0-6 KIR of a
variant (top-level `nbar`, file suffix .nb.json; frontend/ptx/nbar.py), per CTA. Every
event is executed by every warp of its role (the frontend splits roles so):
`named_bar` (an aligned barrier instruction with constant id and count: each warp arrives
with its 32 threads; with `all`, one that names no thread count: a barrier for all threads
of the CTA, which threads that exit release, PTX ISA `exit`), `nb_open` (one the frontend
cannot count: reached under a condition its participants may not share) and `nb_rep` (the
begin and end of a region repeated a number of times the model does not fix, the same in
every warp of the role).

A violation is certain in the extracted model:

* a reached `nb_open` marked certain, outside every repeated region: the model admits
  participants that diverge (thread-index constants that differ; warps that read
  different values of a strong load or an atomic, or leave a spin loop in different
  iterations);
* a count no set of warps meets, outside every repeated region: not a positive multiple of
  32, or above the CTA's threads;
* while only `sync` operations have run (each warp blocks in each one, so every instance
  holds exactly one operation of each warp that completes it), an instance whose
  operations name different counts, or roles that wait forever in the pairing below, each
  outside every repeated region (a count above the threads that arrive, a different number
  of syncs per warp, or a cycle between two barriers).

Pairing: one operation of each role in turn (lockstep); an instance of barrier b opens
with the count of its first operation, collects the arriving roles' threads and completes
at its count, releasing the roles that wait on it (`sync`, `red`); `arrive` does not wait.
An instance whose operations all name no thread count also completes when every role
outside it has ended (its threads exited). The warps of a role arrive independently, so an
instance that roles complete while a role outside it still uses the barrier could have
been completed by other warps: the pairing stops there undecided. A region between
`nb_rep` events is paired once. A barrier that several roles use may lie in such a region
only if every region around each use has a key (a key names a region of the role before
warp splitting whose condition is uniform over that role, so the roles split from it
repeat it alike), and every instance must be formed by operations inside the same nest of
regions (or all outside): then pairing one pass pairs every pass.

A role that arrives again on an open instance it already arrived on (an `arrive`, then
another operation on the same barrier): if the other roles cannot complete the instance
without it (their threads that may still arrive before they block on it, each role up to
its first `sync` on the barrier, do not reach the count), its arrival joins the instance,
whatever the timing. Otherwise it is held until the instance completes and then arrives
on the next one: the warp-ahead reading, an assumption (references/specs.md "P0-6"): no
role runs a whole instance ahead of the others on a named barrier (the kernel's other
synchronization keeps it from arriving twice on one instance). The lockstep order may
complete such an instance before the role's next operation, so the pairing counts every
`arrive` after which its role uses the barrier again while the others could complete the
instance without it: the arrivals the reading holds or may hold.

P0-6 is not decided (UNDECIDED) on an `nb_open` not marked certain or inside a repeated
region, on a bad count inside a repeated region, on a barrier several roles use inside a
region without a key, on an instance mixing operations of different region nests, on more
arrivals than the count, on an instance completed while a role outside it uses the barrier
later, on an instance an `arrive` leaves incomplete when every role has ended, and on a
count mismatch or a wait that never returns after an `arrive` ran, inside a repeated
region (it may run no pass; a wait also while another blocked role waits inside one), with
a role held by the warp-ahead reading, or with an instance whose operations partly name no
thread count and whose other roles all ended. A CTA with a stopped role trace (an
out-of-bounds metadata load, an `assume` that does not hold) is not paired (its later
events are unknown; Check.lean's `not decided` line covers it): only the single-event
checks above apply to the events that ran.
-/
import KV.Spec.Common

namespace KV.NamedBarriers

/-- One named barrier operation of a role's P0-6 trace, with the keys of the repeated
regions around it (innermost first; "" a region without a key); `all`: it names no thread
count (`count` is the CTA's threads). -/
structure Op where
  id : Int
  count : Int
  sync : Bool
  all : Bool
  loc : Loc
  nest : List String
  deriving Inhabited

/-- An open instance of a named barrier in the pairing: its roles so far, the roles that
wait for it, where it opened, in which nest of regions, and whether every (`all`) or some
(`anyAll`) operation in it names no thread count. -/
structure Inst where
  id : Int
  count : Int
  arrived : Int
  roles : Array Nat
  waiting : Array Nat
  loc : Loc
  nest : List String
  all : Bool
  anyAll : Bool

/-- Findings of one CTA (or a run): certain violations, and the findings that leave P0-6
undecided; and the arrivals the warp-ahead reading holds or may hold in the pairing of the CTAs
whose traces are complete (`pair`; the summary line). -/
structure Out where
  certain : Array Issue := #[]
  undecided : Array Issue := #[]
  held : Nat := 0

def Out.ok (o : Out) : Bool := o.certain.isEmpty && o.undecided.isEmpty

def Out.add (o : Out) (cert : Bool) (i : Issue) : Out :=
  if cert then { o with certain := o.certain.push i } else { o with undecided := o.undecided.push i }

def Out.append (a b : Out) : Out :=
  { certain := a.certain ++ b.certain, undecided := a.undecided ++ b.undecided, held := a.held + b.held }

def items (t : RoleTrace) : Array Op := Id.run do
  let mut out : Array Op := #[]
  let mut nest : List String := []
  for e in t.events do
    match e.ev with
    | .namedBar id count sync all _ => out := out.push { id, count, sync, all, loc := e.loc, nest }
    | .nbRep b k => nest := if b then k :: nest else nest.drop 1
    | _ => pure ()
  return out

def undecidedMsg (m : String) : String := s!"not decided on this input: {m}"

/-- The threads of kernel role `r`: 32 per warp. -/
def roleThreads (h : Host) (r : Nat) : Int := 32 * (((h.kernel.roles[r]?).map (·.warps)).getD 0 : Nat)

/-- The barriers several roles use inside a repeated region without a key (`ops`: the `items`
of each role of CTA `c`): (role, barrier, loc, other role), once per (role, barrier). -/
def sharedRepeated (c : CtaTrace) (ops : Array (Array Op)) : Array (Nat × Int × Loc × Nat) := Id.run do
  let ops := (c.roles.map (·.role)).zip ops
  let mut out : Array (Nat × Int × Loc × Nat) := #[]
  for (r, os) in ops do
    for o in os do
      if o.nest.any (· == "") && !(out.any fun (r', i, _, _) => r' == r && i == o.id) then
        if let some (s, _) := ops.find? fun (s, os') => s != r && os'.any (·.id == o.id) then
          out := out.push (r, o.id, o.loc, s)
  return out

/-- The threads role `j` may still add to an open instance of barrier `id` before it
blocks on it (module doc): 0 if it waits on the instance, else its threads for each of its
remaining operations on the barrier up to and including the first `sync`. -/
def potential (h : Host) (roleOf : Array Nat) (ops : Array (Array Op)) (pos : Array Nat)
    (inst : Inst) (j : Nat) : Int := Id.run do
  if inst.waiting.contains j then return 0
  let mut n : Int := 0
  for o in (ops[j]!).extract pos[j]! (ops[j]!).size do
    if o.id == inst.id then
      n := n + 1
      if o.sync then break
  return n * roleThreads h roleOf[j]!

/-- Role `j` has an operation on barrier `id` at or after its position. -/
def usesLater (ops : Array (Array Op)) (pos : Array Nat) (j : Nat) (id : Int) : Bool :=
  ((ops[j]!).extract pos[j]! (ops[j]!).size).any (·.id == id)

/-- The roles other than `k` could complete `inst` without it: its arrivals and the threads
they may still add before they block on it (`potential`) reach its count. -/
def othersComplete (h : Host) (roleOf : Array Nat) (ops : Array (Array Op)) (pos : Array Nat) (inst : Inst)
    (k : Nat) : Bool :=
  inst.arrived + ((List.range ops.size).filter (· != k)).foldl (fun a j => a + potential h roleOf ops pos inst j) 0 ≥ inst.count

/-- The pairing (module doc) of the roles of CTA `c` (`ops`: the `items` of each): the finding
that stops it, if any, with whether it is certain, and how many arrivals the warp-ahead reading
holds or may hold. -/
def pair (h : Host) (c : CtaTrace) (ops : Array (Array Op)) : Option (Bool × Issue) × Nat := Id.run do
  let roleOf : Array Nat := c.roles.map (·.role)
  let n := ops.size
  let ctx (k : Nat) : String := h.roleCtx c.pid roleOf[k]!
  let name (k : Nat) : String := h.roleName roleOf[k]!
  -- role `j` has ended: neither blocked nor held, every operation paired
  let ended (blocked held : Array (Option Int)) (pos : Array Nat) (j : Nat) : Bool :=
    blocked[j]!.isNone && held[j]!.isNone && pos[j]! ≥ (ops[j]!).size
  let mut pos : Array Nat := Array.replicate n 0
  let mut blocked : Array (Option Int) := Array.replicate n none
  let mut held : Array (Option Int) := Array.replicate n none
  let mut waitLoc : Array Loc := Array.replicate n ""
  let mut waitNest : Array (List String) := Array.replicate n []
  let mut insts : Array Inst := #[]
  let mut nheld : Nat := 0
  let mut arrives := false  -- an `arrive` ran: the pairing so far may depend on timing
  let fuel := 2 * ops.foldl (fun a o => a + o.size) 1
  for _ in [0:fuel] do
    let mut progress := false
    for k in [0:n] do
      if blocked[k]!.isSome || held[k]!.isSome then continue
      let p := pos[k]!
      match ops[k]![p]? with
      | none => pure ()
      | some o =>
        let w := roleThreads h roleOf[k]!
        let arr := arrives || !o.sync
        let certain (nests : List (List String)) : Bool := !arr && nests.all (·.isEmpty)
        -- the open instance of the barrier, else a new one (no roles yet)
        let inst := (insts.find? (·.id == o.id)).getD
          { id := o.id, count := o.count, arrived := 0, roles := #[], waiting := #[], loc := o.loc, nest := o.nest,
            all := true, anyAll := false }
        if inst.nest != o.nest then
          return (some (false, { loc := o.loc, context := ctx k, msg := undecidedMsg s!"barrier {o.id}: role {name k} joins an instance opened at {inst.loc} from another nest of repeated regions ({o.nest} vs {inst.nest}): how the instances pair over the regions' passes is not known" }), nheld)
        if inst.count != o.count then
          let m := s!"barrier {o.id}: role {name k} arrives with count {o.count} on an instance that roles {inst.roles.toList.map name} opened with count {inst.count} at {inst.loc}: the counts differ between warps that meet"
          let cert := certain [o.nest, inst.nest]
          return (some (cert, { loc := o.loc, context := ctx k, msg := if cert then m else undecidedMsg m }), nheld)
        if inst.roles.contains k && othersComplete h roleOf ops pos inst k then
          held := held.set! k (some o.id)  -- the warp-ahead reading (module doc)
          continue
        pos := pos.set! k (p + 1)
        progress := true
        arrives := arr
        let a := inst.arrived + w
        if a > o.count then
          let m := if inst.roles.isEmpty then s!"barrier {o.id}: role {name k} arrives with {w} threads on an instance of count {o.count}: which warps complete each instance depends on timing"
            else s!"barrier {o.id}: {a} threads arrive on an instance of count {o.count} (opened at {inst.loc}): which warps complete it depends on timing"
          return (some (false, { loc := o.loc, context := ctx k, msg := undecidedMsg m }), nheld)
        if a == o.count then
          let roles := inst.roles.push k
          if let some j := (List.range n).find? fun j => !roles.contains j && usesLater ops pos j o.id then
            let m := if inst.roles.isEmpty then s!"barrier {o.id}: role {name k} completes an instance of count {o.count} alone, and role {name j} uses the barrier too: which warps complete each instance depends on timing"
              else s!"barrier {o.id}: roles {(roles.toList.map name).eraseDups} complete an instance of count {o.count} (opened at {inst.loc}), and role {name j} uses the barrier too: which warps complete each instance depends on timing"
            return (some (false, { loc := o.loc, context := ctx k, msg := undecidedMsg m }), nheld)
          for x in inst.waiting do
            blocked := blocked.set! x none
          held := held.map fun hb => if hb == some o.id then none else hb
          insts := insts.filter (·.id != o.id)
        else
          let inst := { inst with arrived := a, roles := inst.roles.push k,
                                  waiting := if o.sync then inst.waiting.push k else inst.waiting,
                                  all := inst.all && o.all, anyAll := inst.anyAll || o.all }
          insts := match insts.findIdx? (·.id == o.id) with
            | some t => insts.set! t inst
            | none => insts.push inst
          if o.sync then
            blocked := blocked.set! k (some o.id)
            waitLoc := waitLoc.set! k o.loc
            waitNest := waitNest.set! k o.nest
          else if usesLater ops pos k o.id && othersComplete h roleOf ops pos inst k then
            nheld := nheld + 1  -- the warp-ahead reading decides where its next operation lands
    if !progress then
      -- threads that exit release an instance whose operations all name no thread count
      match insts.find? fun i => i.all && (List.range n).all fun j => i.roles.contains j || ended blocked held pos j with
      | some i =>
        for x in i.waiting do
          blocked := blocked.set! x none
        insts := insts.filter (·.id != i.id)
      | none => break
  let anyHeld := held.any (·.isSome)
  if let some k := (List.range n).find? fun j => blocked[j]!.isNone && held[j]!.isNone && !ended blocked held pos j then
    return (some (false, { loc := "", context := ctx k, msg := undecidedMsg "the pairing did not finish" }), nheld)
  -- every blocked role waits outside the repeated regions, and no instance that names no
  -- thread count in some operations (threads that exit may release it) waits only for roles that ended
  let outside := (List.range n).all fun j => blocked[j]!.isNone || waitNest[j]!.isEmpty
  let exitMay := insts.any fun i => i.anyAll && (List.range n).all fun j => i.roles.contains j || ended blocked held pos j
  for k in [0:n] do
    if let some id := blocked[k]! then
      let inst := insts.find? (·.id == id)
      let got := (inst.map (·.arrived)).getD 0
      let cnt := (inst.map (·.count)).getD 0
      let who := (inst.map (·.roles.toList.map name)).getD []
      let m := s!"barrier {id}: role {name k} waits forever: {got} of the {cnt} threads arrive on its instance (roles {who}) and every other role has ended or waits too (a count above the threads that reach it, a different number of operations per warp, or a cycle between barriers)"
      let cert := !arrives && !anyHeld && outside && !exitMay
      return (some (cert, { loc := waitLoc[k]!, context := ctx k, msg := if cert then m else undecidedMsg m }), nheld)
  for k in [0:n] do
    if let some id := held[k]! then
      let o := ops[k]![pos[k]!]!
      return (some (false, { loc := o.loc, context := ctx k, msg := undecidedMsg s!"barrier {id}: role {name k} arrives again on an open instance that the others could complete, so under the warp-ahead reading it waits for the next one, which never opens" }), nheld)
  if let some inst := insts[0]? then
    return (some (false, { loc := inst.loc, context := ctx inst.roles[0]!,
                           msg := undecidedMsg s!"barrier {inst.id}: an instance of count {inst.count} ends the CTA with {inst.arrived} threads arrived (an arrive no operation completes)" }), nheld)
  return (none, nheld)

/-- P0-6 findings of one CTA. A finding inside a repeated region (`nb_rep`; it may run no
pass) is not certain. In a CTA with a role trace cut short (an out-of-bounds
metadata load, an `assume` that does not hold; `CtaTrace.complete` false) only the checks
of single events that ran (certain whatever comes later): the repeated regions and the
pairing need every role's whole trace, so they are not evaluated there (Check.lean's
`not decided` line covers the CTA), as the ledger specs skip such clusters
(`Run.clusters`). A complete CTA is paired, its held arrivals counted (`Out.held`), even when
a finding before the pairing decides it. -/
def ctaOut (h : Host) (c : CtaTrace) : Out := Id.run do
  let mut out : Out := {}
  let threads : Int := c.roles.foldl (fun a t => a + roleThreads h t.role) 0
  for t in c.roles do
    let ctx := h.roleCtx c.pid t.role
    let mut seen : Array (Loc × String) := #[]  -- each nb_open once per role (loops repeat it)
    let mut depth : Nat := 0  -- the repeated regions around the event
    for e in t.events do
      match e.ev with
      | .nbRep b _ => depth := if b then depth + 1 else depth - 1
      | .nbOpen cert why =>
        unless seen.contains (e.loc, why) do
          seen := seen.push (e.loc, why)
          let cert := cert && depth == 0
          out := out.add cert { loc := e.loc, context := ctx, msg := if cert then why else undecidedMsg why }
      | .namedBar id count _ _ _ =>
        -- frontend/kir.py rejects a count that is not a positive multiple of 32 (the barrier
        -- pass makes such a site an `nb_open`), so on a valid KIR only the second check fires;
        -- the decision keeps both
        let bad := if count ≤ 0 || count % 32 != 0 then
            some s!"barrier {id}: count {count} is not a positive multiple of 32: no set of warps meets it"
          else if count > threads then
            some s!"barrier {id}: count {count} exceeds the CTA's {threads} threads: no instance completes"
          else none
        if let some m := bad then
          out := out.add (depth == 0) { loc := e.loc, context := ctx, msg := if depth == 0 then m else undecidedMsg m }
      | _ => pure ()
  if !c.complete then return out
  let ops := c.roles.map items
  let (found, held) := pair h c ops
  out := { out with held }
  if !out.ok then return out
  for (r, id, loc, s) in sharedRepeated c ops do
    out := out.add false { loc, context := h.roleCtx c.pid r,
                           msg := undecidedMsg s!"barrier {id}: role {h.roleName r} uses it in a region repeated a number of times the model does not fix, and role {h.roleName s} uses it too: how many instances each takes part in is not known" }
  if !out.ok then return out
  match found with
  | some (cert, i) => return out.add cert i
  | none => return out

/-- P0-6 findings of a run. -/
def analyze (r : Run) : Out := r.ctas.foldl (fun o c => o.append (ctaOut r.host c)) {}

/-- P0-6 on one run. -/
def Holds (r : Run) : Prop := (analyze r).ok = true

instance (r : Run) : Decidable (Holds r) := by unfold Holds; infer_instance

/-- Operations, open and repeated-region events of a run (the summary line). -/
def counts (r : Run) : Nat × Nat × Nat :=
  r.forEvents (0, 0, 0) fun (a, b, c) _ _ e => match e.ev with
    | .namedBar .. => (a + 1, b, c)
    | .nbOpen .. => (a, b + 1, c)
    | .nbRep true _ => (a, b, c + 1)
    | _ => (a, b, c)

end KV.NamedBarriers
