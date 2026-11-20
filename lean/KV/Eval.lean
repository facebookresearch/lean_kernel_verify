/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
KIR evaluator: runs every role of every program instance (CTA) on one concrete `Input`
and returns one event trace per (CTA, role).

* Integer semantics are exact `BitVec` semantics at the node width. A node marked `addr`
  whose result differs from the result on unbounded integers records an `overflow` event:
  a `bin` add/sub/mul/shl that wraps (signed reading) or a lossy `trunc`; for tensor
  expressions one event per node evaluation (first offending lane).
* A metadata `load` whose index lies outside the tensor's extents records a `metaOob`
  event (a P0-1 violation) and stops its role: its later events depend on the unknown value
  read. The other roles and CTAs run to completion (roles share no values). An `assume`
  whose condition is false (KIR change 16: a model assumption the frontend could not prove
  statically, false on this input) stops its role the same way, without an event; the
  trace records why it stopped (`StopWhy`).
* Verdict errors (evaluation stops: every spec's theorem is refuted), each of a kind
  (`ErrKind`, references/specs.md "Evaluation errors"). `kernel`, a certain error of the
  kernel or its host on the input, so every spec FAILs: a failed device `assert`, a barrier
  index out of range, a `for` that starts below its upper bound with a step that is not
  positive (it never ends; with no trips it runs nothing), a negative grid dimension, a
  grid the cluster dims do not divide, an event naming a CTA outside the cluster.
  `limit`, a known limit of the model, so no spec is decided and it is not an error:
  division or remainder by zero (in a tensor expression on any lane, masked or not; the
  GPU computes an unspecified value, which the kernel may never use). `unexpected`, the
  case or LKV needs a fix, so no spec is decided and it is an ERROR: `while` exceeding
  `max_iters` (a hang of the kernel, or a budget too small for the input), an in-extent
  metadata `load` at a negative offset or one the input gives no value for, a negative
  `expect_tx` or `complete_tx` byte count (PTX transaction counts are unsigned); in the host
  setup a negative barrier `num` and a segmented global tensor whose strides overlap
  (rows of offsets undefined); and IR inconsistencies (width mismatch, unbound variable,
  unknown tensor/barrier) that `frontend/kir.py` should already have rejected.
* Variables: one environment per role, params pre-bound; `let` assigns. Host-side Exprs
  (grid, extents, strides, barrier sizes) see only params and record no events (an `addr`
  wrap in them is dropped: `hostInts` keeps only the values).
* Integer readings of event fields, offsets and coordinates are signed for wide values
  and 0/1 for i1 values.
* `work` is dynamic: the fields of the role's last executed `work` event.
* A scalar `select` evaluates its condition, then only the operand it chooses: wraps and
  metadata loads in the other one are dead. The frontends lower a masked scalar load as
  `select(mask, load(T, [select(mask, off, 0)]), other)`, so a disabled load reads nothing;
  the access itself is also a `gload` event, which P0-1 checks. A tensor `select`
  evaluates both operands on every lane.
* Roles run independently; specs only use per-role order and per-CTA (per-cluster) totals,
  so no interleaving is explored.
* Clusters (`Kernel.cluster`): consecutive CTAs form a cluster, a CTA's rank in it is
  `x % cx + (y % cy) * cx + (z % cz) * cx * cy`. A barrier event names a slot of a CTA of its
  cluster by rank (`Slot.cta`); a rank outside the cluster is a kernel error (PTX: undefined),
  and so is a grid that the cluster dims do not divide (the launch fails).
-/
import Std.Data.HashMap
import KV.IR

namespace KV

/-! ## Values -/

/-- A KIR scalar: a bit vector together with its width. -/
structure Val where
  w : Nat
  bv : BitVec w

/-- Integer reading: signed for `w ≥ 2`, 0/1 for i1. -/
def bvInt {w : Nat} (b : BitVec w) : Int := if w = 1 then b.toNat else b.toInt

def Val.int (v : Val) : Int := bvInt v.bv

/-- A tensor value: row-major data of shape `shape`. -/
structure TVal where
  w : Nat
  shape : List Nat
  data : Array (BitVec w)

def numel (shape : List Nat) : Nat := shape.foldl (· * ·) 1

/-- `o` with `BitVec` semantics; `none` on division/remainder by zero. Shift amounts are
clamped to `w`, which leaves the `BitVec` result unchanged and keeps it cheap. -/
def binBV {w : Nat} (o : BinOp) (a b : BitVec w) : Option (BitVec w) :=
  let sh := min b.toNat w
  match o with
  | .add => some (a + b)
  | .sub => some (a - b)
  | .mul => some (a * b)
  | .sdiv => if b = 0 then none else some (a.sdiv b)
  | .udiv => if b = 0 then none else some (a / b)
  | .srem => if b = 0 then none else some (a.srem b)
  | .urem => if b = 0 then none else some (a % b)
  | .shl => some (a <<< sh)
  | .ashr => some (a.sshiftRight sh)
  | .lshr => some (a >>> sh)
  | .and => some (a &&& b)
  | .or => some (a ||| b)
  | .xor => some (a ^^^ b)
  | .smin => some (if a.slt b then a else b)
  | .smax => some (if a.slt b then b else a)
  | .umin => some (if a.ult b then a else b)
  | .umax => some (if a.ult b then b else a)

/-- Signed wrap: the `w`-bit result `r` of add/sub/mul/shl differs from the integer result
(a shift by `w` or more bits wraps unless `a = 0`). -/
def wraps {w : Nat} (o : BinOp) (a b r : BitVec w) : Bool :=
  match o with
  | .add => r.toInt != a.toInt + b.toInt
  | .sub => r.toInt != a.toInt - b.toInt
  | .mul => r.toInt != a.toInt * b.toInt
  | .shl => r.toInt != a.toInt * 2 ^ min b.toNat w
  | _ => false

/-- Lossy truncation: the `w`-bit value `r = trunc a` reads as a different integer. -/
def truncLoses {v w : Nat} (a : BitVec v) (r : BitVec w) : Bool := bvInt r != bvInt a

def cmpBV {w : Nat} (o : CmpOp) (a b : BitVec w) : Bool :=
  match o with
  | .eq => a == b
  | .ne => a != b
  | .slt => a.slt b
  | .sle => a.sle b
  | .sgt => b.slt a
  | .sge => b.sle a
  | .ult => a.ult b
  | .ule => a.ule b
  | .ugt => b.ult a
  | .uge => b.ule a

def ofBool (c : Bool) : BitVec 1 := if c then 1 else 0

/-! ## Per-input host state -/

structure TensorInfo where
  decl : Tensor
  /-- Effective extents: `Input.shapes` if given, else the evaluated `shape` Exprs. -/
  shape : Array Int
  strides : Array Int
  /-- Contents of a metadata tensor (flat buffer indexed by `strides`), else `#[]`. -/
  contents : Array Int
  /-- The strides of dims with more than one coordinate that are larger than the row dim's
  (segment dim, else dim 0), largest first; see `TensorInfo.rowOf`. -/
  rowOuter : Array Int
  /-- Dims with more than one coordinate, largest stride first. -/
  order : Array Nat
  /-- The strides do not overlap: every dim of `order` has a positive stride larger than
  the span of the other dims with a smaller or equal stride, so an element offset has
  unique coordinates (`TensorInfo.isElem`). -/
  exact : Bool
  deriving Inhabited

/-- Lowest and highest element offset of the tensor; `none` if it has no elements. -/
def TensorInfo.span (t : TensorInfo) : Option (Int × Int) :=
  if t.shape.any (· ≤ 0) then none
  else some ((t.shape.zip t.strides).foldl (fun (lo, hi) (n, s) =>
    (lo + min 0 ((n - 1) * s), hi + max 0 ((n - 1) * s))) (0, 0))

/-- `o` is `Σ c_i * strides[i]` for coordinates `0 ≤ c_i < shape[i]`, read off greedily
from the largest stride down; exact when `t.exact`. -/
def TensorInfo.isElem (t : TensorInfo) (o : Int) : Bool := Id.run do
  let mut rem := o
  for d in t.order do
    let s := t.strides[d]!
    let c := rem / s
    if c < 0 || c ≥ t.shape[d]! then return false
    rem := rem - c * s
  return rem == 0

def TensorInfo.rank (t : TensorInfo) : Nat := t.shape.size

structure BarrierInfo where
  decl : Barrier
  count : Int
  num : Nat
  deriving Inhabited

/-- A sync word: element offset `e` of sync tensor number `t` (KIR change 18). -/
structure Word where
  t : Nat
  e : Int
  deriving BEq, Hashable, Inhabited, Repr

structure Host where
  kernel : Kernel
  input : Input
  grid : Nat × Nat × Nat
  /-- Cluster dims (x, y, z); (1, 1, 1) without clusters. -/
  cluster : Nat × Nat × Nat := (1, 1, 1)
  params : Std.HashMap String Val
  tensors : Array TensorInfo
  tensorIdx : Std.HashMap String Nat
  barriers : Array BarrierInfo
  barIdx : Std.HashMap String Nat
  /-- KIR change 18: the tagged words' stores of the previous pass (`evalSyncPasses`): (word,
  tag) -> the value half (`none`: data). -/
  puts : Std.HashMap (Word × Int) (Option Int) := {}
  /-- KIR change 18: per claimed word, the CTAs that claimed on it in the previous pass (linear
  index -> claimer index) and their count. -/
  claimOrder : Std.HashMap Word (Std.HashMap Nat Nat × Nat) := {}

/-- The kernel has sync tensors (global synchronization events, KIR change 18). -/
def Host.hasSync (h : Host) : Bool := h.tensors.any (·.decl.sync.isSome)

/-- Bit width of sync tensor `t`'s words (32 if it is not one). -/
def Host.syncWidth (h : Host) (t : Nat) : Nat := ((h.tensors[t]?.bind (·.decl.sync)).getD 32)

/-! ## Traces -/

/-- One mbarrier: barrier array index into `Host.barriers`, element index, and the rank in the
cluster of the CTA whose shared memory holds it (0 without clusters). -/
structure Slot where
  bar : Nat
  idx : Nat
  cta : Nat := 0
  deriving Inhabited

/-- A slot as a map key: (barrier array, index, rank of its CTA). -/
def Slot.key (s : Slot) : Nat × Nat × Nat := (s.bar, s.idx, s.cta)

/-- The thread that executes a P1-2 event, in its CTA: lane `lane` of warp `warp`, or with
`lane = none` the elected lane of a full-warp elect.sync in that warp (one lane, the same
for every such election: PTX ISA elect.sync elects deterministically). -/
structure Who where
  warp : Int
  lane : Option Int
  deriving BEq, Inhabited

/-- An evaluated event. Tensors are indices into `Host.tensors`; arrivals are already
multiplied by the issuing thread count; memory offsets keep masked-in lanes only. -/
inductive Ev where
  | barInit (s : Slot) (count : Int)
  /-- `async`: the arrival happens when the issuing role's prior async MMAs complete
  (`commit_arrive`), at some time after the event. -/
  | arrive (s : Slot) (arrivals : Int) (async : Bool)
  | expectTx (s : Slot) (bytes : Int)
  | wait (s : Slot) (parity : Int)
  | tmaLoad (t : Nat) (coords : Array Int) (s : Slot) (bytes : Int)
  /-- Transaction bytes completing on a slot with no global access of their own: the copies
  of a multicast TMA load beyond the first destination, `complete_tx`. -/
  | landTx (s : Slot) (bytes : Int)
  | tmaStore (t : Nat) (coords : Array Int)
  | tmaReduce (t : Nat) (coords : Array Int)
  | gload (t : Nat) (offsets : Array Int)
  | gstore (t : Nat) (offsets : Array Int)
  | gatomic (t : Nat) (offsets : Array Int)
  | work (fields : Array (String × Int))
  /-- An `addr` node `what` (e.g. `mul(2048, 1048576)`) wrapped to `r` at width `w`. -/
  | overflow (what : String) (r : Int) (w : Nat)
  /-- A `load` of metadata tensor `t` at `index`, outside its extents `shape`; the role
  stops here. -/
  | metaOob (t : Nat) (index : Array Int) (shape : Array Int)
  /-- A TMEM access (P1-1): instruction, issuing warp, the ranks of the CTAs whose TMEM it
  accesses, rectangles `(use, write, l0, l1, c0, c1)` with half-open lanes `[l0, l1)` and
  columns `[c0, c1)`, and the informational site. -/
  | tmem (inst : TmemInst) (warp : Int) (ctas : Array Nat) (rects : Array (RectUse × Bool × Int × Int × Int × Int)) (site : Loc)
  /-- tcgen05.wait::ld (`ld`) or wait::st of the role. -/
  | tmemWait (ld : Bool)
  /-- bar.sync (`sync`) or bar.arrive on named barrier `id` with `count` threads, by all
  `threads` threads of the role. -/
  | namedBar (id count : Int) (sync : Bool) (threads : Int)
  /-- bar.sync or bar.red without a thread count (P0-6 KIR `named_bar` with `all`): named
  barrier `id` for all `count` threads of the CTA, which threads that exit release, by all
  `threads` threads of the role. -/
  | namedBarAll (id count : Int) (sync : Bool) (threads : Int)
  /-- A named barrier P0-6 cannot count, reached by the role (P0-6 KIR `nb_open`). -/
  | nbOpen (kind : String) (certain : Bool) (why : String) (id count : Option Int) (sync : Bool)
  /-- The begin or end of a region that runs a number of times the model does not fix
  (P0-6 KIR `nb_rep`; `key`: its identity across the roles of one split role, or empty). -/
  | nbRep (isBegin : Bool) (key : String)
  /-- A bulk asynchronous store (P1-2): its issuing thread, the shared-memory bytes
  `[lo, hi)` of symbol `sym` it reads (in the issuing CTA), and the tensormap it reads
  (global buffer, first byte; 128 bytes) if any. -/
  | bulkStore (kind : BulkKind) (who : Who) (sym : String) (lo hi : Int) (tmap : Option (String × Int))
  /-- cp.async.bulk.commit_group of every thread of the role. -/
  | bulkCommit
  /-- cp.async.bulk.wait_group{.read} `n` of every thread of the role. -/
  | bulkWait (n : Int) (read : Bool)
  /-- A shared-memory write (P1-2) by `who` (with `sync`: every lane of `who.warp`, one
  warp-synchronizing instruction) of the byte ranges `[lo, hi)` of symbol `sym` in the
  CTA of rank `cta` of the cluster; `async`: an asynchronous copy's landing, `slot` the
  mbarrier slot the copy completes its bytes on, if known. -/
  | smemWrite (who : Who) (sync : Bool) (async : Bool) (cta : Nat) (sym : String)
      (ranges : Array (Int × Int)) (slot : Option Slot)
  /-- A tensormap write (P1-2) by every lane of warp `warp` (tensormap.cp_fenceproxy): the
  128 bytes from byte offset `lo` of global buffer `base`. -/
  | tmapWrite (warp : Int) (base : String) (lo : Int)
  /-- The role's value of handoff channel `chan` changed (KIR change 17, `handoff_put`). -/
  | handoffPut (chan : String)
  /-- A `handoff_get` of channel `chan`: the value the putter recorded at its `k`-th arrive on
  slot `s`, the getter's last wait being its `k`-th on `s` (counting from 0). -/
  | handoff (chan : String) (s : Slot) (k : Nat)
  /-- A sync event on element `index` of sync tensor `t`, outside the tensor (a P0-1
  violation); the role stops here. -/
  | syncOob (t : Nat) (index : Int)
  /-- KIR change 18: a store of `v` to sync word `w` (on a 64-bit word its high half is a tag),
  a take of tag `tag` of `w`, a claim adding `add` to `w`. -/
  | gsig (w : Word) (v : Int)
  | gwait (w : Word) (tag : Int)
  | gclaim (w : Word) (add : Int)

structure TEvent where
  ev : Ev
  loc : Loc
  /-- Current work item. -/
  work : Array (String × Int)

/-- Why a role trace stopped before its end: an out-of-bounds metadata `load` (its
`metaOob` event, the trace's last, a P0-1 violation) or an `assume` whose condition is
false on the input (KIR change 16; no event), with the assumption and its loc. -/
inductive StopWhy where
  | metaOob
  | assume (msg : String) (loc : Loc)
  /-- A take of a tagged word at `loc` whose tag no store of the previous pass carries (KIR
  change 18): blocked there, forever unless a later pass knows the store (`Run.hangs`). -/
  | blocked (loc : Loc)
  deriving Inhabited, BEq

/-- The reason, as the summary lines print it. -/
def StopWhy.text : StopWhy → String
  | .metaOob => "an out-of-bounds metadata load"
  | .assume msg _ => s!"a model assumption that does not hold ({msg})"
  | .blocked _ => "a take of a tagged word that no store carries (blocked)"

/-- The reason, as the `not decided` lines print it. -/
def StopWhy.detail : StopWhy → String
  | .metaOob => "an out-of-bounds metadata load (P0-1)"
  | w => w.text

structure RoleTrace where
  role : Nat
  events : Array TEvent
  /-- The role stopped before its end (`why`); later events are unknown. -/
  stopped : Bool
  /-- Why it stopped (meaningful only if `stopped`). -/
  why : StopWhy := .metaOob
  /-- KIR change 18: the role's claims (word, the old value read, the add), its stores of
  tagged words (word, tag, value half; `none`: data) and the words it claimed on; `blocked`:
  it stopped at a take (`stopped` too). -/
  claimed : Array (Word × Int × Int) := #[]
  puts : Array (Word × Int × Option Int) := #[]
  claimWords : Array Word := #[]
  blocked : Bool := false

/-- Where a stopped trace stopped: the `assume`'s or the blocked take's loc, else its last
event's (the `metaOob` event). -/
def RoleTrace.stopLoc (t : RoleTrace) : Loc :=
  match t.why with
  | .assume _ loc | .blocked loc => loc
  | .metaOob => (t.events.back?.map (·.loc)).getD ""

/-- The distinct reasons the stopped traces `ts` stopped, in order (`detail`: the `not
decided` lines' form, else the summaries'). -/
def stopReasons (ts : Array RoleTrace) (detail : Bool := true) : String :=
  let xs := ts.foldl (init := (#[] : Array String)) fun acc t =>
    let s := if detail then t.why.detail else t.why.text
    if acc.contains s then acc else acc.push s
  " and ".intercalate xs.toList

structure CtaTrace where
  pid : Nat × Nat × Nat
  roles : Array RoleTrace
  /-- Rank of the CTA in its cluster (0 without clusters). -/
  rank : Nat := 0

/-- Every role of the CTA ran to its end. -/
def CtaTrace.complete (c : CtaTrace) : Bool := c.roles.all (!·.stopped)

structure Run where
  host : Host
  ctas : Array CtaTrace

/-- Every role of every CTA ran to its end (no trace stopped: an out-of-bounds metadata
load, an assumption that does not hold). -/
def Run.complete (r : Run) : Bool := r.ctas.all (·.complete)

/-- KIR change 18: some role is blocked at a take of a tag that no store of the run carries,
and every role that stopped is blocked so: the run hangs there, which P0-7 decides; the other
specs are not decided. -/
def Run.hangs (r : Run) : Bool :=
  !r.complete && r.ctas.all (·.roles.all fun t => !t.stopped || t.blocked)

/-- Who must act on an evaluation error (references/specs.md, "Evaluation errors"):
`kernel`, a certain error of the kernel or its host on the input (every spec FAILs);
`limit`, a known limit of the model (no spec is decided; not an error); `unexpected`, the
case or LKV needs a fix (no spec is decided; an ERROR). -/
inductive ErrKind where
  | kernel | limit | unexpected
  deriving BEq, Repr, Inhabited

def ErrKind.label : ErrKind → String
  | .kernel => "kernel error" | .limit => "model limit" | .unexpected => "unexpected"

/-- What an evaluation error of the kind means for the input's specs. -/
def ErrKind.consequence : ErrKind → String
  | .kernel => "a certain error of the kernel on this input: every spec FAILs"
  | .limit => "a limit of the model: no spec is decided on this input; not an error"
  | .unexpected => "the case or LKV needs a fix: no spec is decided on this input"

structure EvalError where
  msg : String
  loc : Loc
  /-- "cta (x,y,z) role r", or "host" for input setup. -/
  context : String
  /-- Raised after a `metaOob` event or by an `assume` that does not hold: ends the role,
  not the evaluation, for the reason `why`. -/
  stop : Bool := false
  why : StopWhy := .metaOob
  kind : ErrKind := .unexpected
  /-- Raised by a take whose store is not known (KIR change 18; `stop` too). -/
  blocked : Bool := false

/-! ## Evaluation monad -/

structure Ctx where
  host : Host
  pid : Nat × Nat × Nat
  role : Role
  /-- Rank of the CTA in its cluster. -/
  rank : Nat := 0
  /-- `false` for host-side Exprs (grid, shapes, strides, barrier sizes). -/
  device : Bool
  /-- Linear index of the CTA (x fastest). -/
  cta : Nat := 0

/-- The handoff records of a CTA (KIR change 17): (channel, slot key, k) -> the channel's
value at its putter role's k-th `arrive` on the slot (counting from 0). -/
abbrev Handoffs := Std.HashMap (String × (Nat × Nat × Nat) × Nat) Int

structure St where
  env : Std.HashMap String Val
  trace : Array TEvent
  work : Array (String × Int)
  /-- KIR change 17: the role's values of the handoff channels it puts, its `arrive` and
  `wait` events so far per slot, the slot of its last wait, and the CTA's handoff records
  (carried from role to role, `eval`). -/
  chans : Std.HashMap String Int := {}
  arrived : Std.HashMap (Nat × Nat × Nat) Nat := {}
  waited : Std.HashMap (Nat × Nat × Nat) Nat := {}
  lastWait : Option Slot := none
  records : Handoffs := {}
  /-- KIR change 18: claims made so far per word (the role instance's claim number), the
  role's claims, tagged stores and claimed words (`RoleTrace`), and the CTA's relay slots
  (carried from role to role, `evalCta`). -/
  claims : Std.HashMap Word Nat := {}
  claimed : Array (Word × Int × Int) := #[]
  puts : Array (Word × Int × Option Int) := #[]
  claimWords : Array Word := #[]
  relay : Std.HashMap (String × Int × Nat) Int := {}
  /-- Variables of `rget`s of slots no role wrote, or of takes whose store is data: reading one
  is a model limit, with its message. -/
  poison : Std.HashMap String String := {}

abbrev M := ReaderT Ctx (EStateM EvalError St)

/-- KIR change 18: the relay map's marker that a role putting channel `chan` is blocked at a
take (`evalCta`); no real slot has field 2^20. -/
def blockedKey (chan : String) : String × Int × Nat := (chan, -1, 1048576)

/-- The poison of a relay value its blocked putter never writes. -/
def blockedPrefix : String := "relay (blocked putter) "

def throwAt {α : Type} (loc : Loc) (msg : String) : M α :=
  throw { msg, loc, context := "" }

/-- An evaluation error of a given kind (`throwAt`: `unexpected`). -/
def throwK {α : Type} (loc : Loc) (kind : ErrKind) (msg : String) : M α :=
  throw { msg, loc, context := "", kind }

def emit (loc : Loc) (ev : Ev) : M Unit :=
  modify fun s => { s with trace := s.trace.push { ev, loc, work := s.work } }

def asW (loc : Loc) (v : Val) (w : Nat) : M (BitVec w) :=
  if h : v.w = w then pure (h ▸ v.bv) else throwAt loc s!"width mismatch: i{v.w} used as i{w}"

def dataW (loc : Loc) (x : TVal) (w : Nat) : M (Array (BitVec w)) :=
  if h : x.w = w then pure (h ▸ x.data) else throwAt loc s!"width mismatch: tensor of i{x.w} used as i{w}"

def axisOf (t : Nat × Nat × Nat) : Nat → Nat
  | 0 => t.1 | 1 => t.2.1 | _ => t.2.2

def tensorOf (loc : Loc) (name : String) : M (Nat × TensorInfo) := do
  let h := (← read).host
  match h.tensorIdx[name]? with
  | some i => match h.tensors[i]? with
    | some t => pure (i, t)
    | none => throwAt loc s!"unknown tensor {name}"
  | none => throwAt loc s!"unknown tensor {name}"

/-- Element `idx` of a metadata tensor. An index outside the extents records a `metaOob`
event and stops the role (`EvalError.stop`). -/
def loadMeta (loc : Loc) (name : String) (idx : List Int) : M Int := do
  unless (← read).device do throwAt loc s!"load of {name} in a host-side expression"
  let (ti, t) ← tensorOf loc name
  unless t.decl.metadata do throwAt loc s!"load from non-metadata tensor {name}"
  let idx := idx.toArray
  unless idx.size = t.rank do throwAt loc s!"load {name}: {idx.size} indices for rank {t.rank}"
  unless (idx.zip t.shape).all fun (i, n) => 0 ≤ i && i < n do
    emit loc (.metaOob ti idx t.shape)
    throw { msg := s!"load {name}{idx.toList} outside the tensor", loc, context := "", stop := true }
  let off := (idx.zip t.strides).foldl (fun (acc : Int) (i, s) => acc + i * s) 0
  match t.contents[off.toNat]? with
  | some v => if off ≥ 0 then pure v else throwAt loc s!"load {name}{idx.toList}: negative offset {off}"
  | none => throwAt loc s!"load {name}{idx.toList}: offset {off}, but the input gives {t.contents.size} values"

/-- The sync word at element offset `off` of sync tensor `name`. An offset outside the tensor
records a `syncOob` event and stops the role (`EvalError.stop`). -/
def wordOf (loc : Loc) (name : String) (off : Int) : M Word := do
  let (ti, t) ← tensorOf loc name
  unless t.decl.sync.isSome do throwAt loc s!"sync access to {name}, which is not a sync tensor"
  let inside := match t.span with
    | some (lo, hi) => if t.exact then t.isElem off else lo ≤ off && off ≤ hi
    | none => false
  unless inside do
    emit loc (.syncOob ti off)
    throw { msg := s!"sync access {name}[{off}] outside the tensor", loc, context := "", stop := true }
  pure { t := ti, e := off }

/-! ## Expressions -/

mutual
def evalExpr (loc : Loc) : Expr → M Val
  | .const v w => pure ⟨w, BitVec.ofInt w v⟩
  | .var x w => do
    let st ← get
    match st.env[x]? with
    | some v => if v.w = w then pure v else throwAt loc s!"variable {x} is i{v.w}, used as i{w}"
    | none =>
      match st.poison[x]? with
      | some why =>
        if why.startsWith blockedPrefix then  -- its putter is blocked: so is this role
          throw { msg := why, loc, context := "", stop := true, why := .blocked loc, blocked := true,
                  kind := .limit }
        throwK loc .limit why
      | none => throwAt loc s!"unbound variable {x}"
  | .pid a w => do
    let c ← read
    unless c.device do throwAt loc "pid in a host-side expression"
    pure ⟨w, BitVec.ofNat w (axisOf c.pid a)⟩
  | .nprog a w => do
    let c ← read
    unless c.device do throwAt loc "nprog in a host-side expression"
    pure ⟨w, BitVec.ofNat w (axisOf c.host.grid a)⟩
  | .bin o a b w addr => do
    let x ← asW loc (← evalExpr loc a) w
    let y ← asW loc (← evalExpr loc b) w
    match binBV o x y with
    | none => throwK loc .limit s!"division by zero: {o.name}({x.toInt}, 0)"
    | some r =>
      if addr && wraps o x y r then emit loc (.overflow s!"{o.name}({x.toInt}, {y.toInt})" r.toInt w)
      pure ⟨w, r⟩
  | .cmp o a b w => do
    unless w = 1 do throwAt loc "cmp result must be i1"
    let x ← evalExpr loc a
    let y ← asW loc (← evalExpr loc b) x.w
    pure ⟨1, ofBool (cmpBV o x.bv y)⟩
  | .select c a b w => do
    -- only the chosen operand runs (a masked scalar load reads nothing where its mask is false)
    let x ← if (← asW loc (← evalExpr loc c) 1) = 1 then evalExpr loc a else evalExpr loc b
    pure ⟨w, ← asW loc x w⟩
  | .ext signed a w => do
    let x ← evalExpr loc a
    unless x.w < w do throwAt loc s!"ext from i{x.w} to i{w}"
    pure ⟨w, if signed then x.bv.signExtend w else x.bv.setWidth w⟩
  | .trunc a w addr => do
    let x ← evalExpr loc a
    unless w < x.w do throwAt loc s!"trunc from i{x.w} to i{w}"
    let r := x.bv.setWidth w
    if addr && truncLoses x.bv r then emit loc (.overflow s!"trunc({x.int})" (bvInt r) w)
    pure ⟨w, r⟩
  | .load t index w => do
    let is ← evalExprs loc index
    pure ⟨w, BitVec.ofInt w (← loadMeta loc t (is.map Val.int))⟩

def evalExprs (loc : Loc) : List Expr → M (List Val)
  | [] => pure []
  | e :: es => return (← evalExpr loc e) :: (← evalExprs loc es)
end

def evalInt (loc : Loc) (e : Expr) : M Int := return (← evalExpr loc e).int

def evalCond (loc : Loc) (e : Expr) : M Bool := return (← asW loc (← evalExpr loc e) 1) = 1

/-- Elementwise `o`; also returns operands and result of the first lane that wraps (if `addr`). -/
def mapBin {w : Nat} (o : BinOp) (addr : Bool) (xs ys : Array (BitVec w)) :
    Option (Array (BitVec w) × Option (BitVec w × BitVec w × BitVec w)) := Id.run do
  let mut out := Array.emptyWithCapacity xs.size
  let mut wrap := none
  for i in [0:xs.size] do
    let x := xs[i]!
    let y := ys[i]!
    match binBV o x y with
    | none => return none
    | some r =>
      if addr && wrap.isNone && wraps o x y r then wrap := some (x, y, r)
      out := out.push r
  return some (out, wrap)

/-- Broadcast `xs` (extent 1 along `axis`) to `shape`. -/
def bcastData {α : Type} [Inhabited α] (xs : Array α) (axis : Nat) (shape : List Nat) : Array α :=
  let outer := numel (shape.take axis)
  let n := shape.getD axis 1
  let inner := numel (shape.drop (axis + 1))
  Id.run do
    let mut out := Array.emptyWithCapacity (outer * n * inner)
    for o in [0:outer] do
      for _ in [0:n] do
        for i in [0:inner] do
          out := out.push xs[o * inner + i]!
    return out

def evalTExpr (loc : Loc) : TExpr → M TVal
  | .range s e shape w =>
    pure ⟨w, shape, (Array.range (e - s).toNat).map fun i => BitVec.ofInt w (s + Int.ofNat i)⟩
  | .splat a shape w => do
    let b ← asW loc (← evalExpr loc a) w
    pure ⟨w, shape, Array.replicate (numel shape) b⟩
  | .bcast a axis shape w => do
    let x ← evalTExpr loc a
    let d ← dataW loc x w
    unless x.shape.getD axis 0 = 1 && x.shape.length = shape.length do
      throwAt loc s!"bcast of shape {x.shape} along axis {axis} to {shape}"
    pure ⟨w, shape, bcastData d axis shape⟩
  | .expand a _ shape w => do
    let x ← evalTExpr loc a
    let d ← dataW loc x w
    unless numel x.shape = numel shape do throwAt loc s!"expand of shape {x.shape} to {shape}"
    pure ⟨w, shape, d⟩
  | .bin o a b shape w addr => do
    let x ← dataW loc (← evalTExpr loc a) w
    let y ← dataW loc (← evalTExpr loc b) w
    unless x.size = numel shape && y.size = numel shape do throwAt loc s!"bin operand shapes differ from {shape}"
    match mapBin o addr x y with
    | none => throwK loc .limit s!"division by zero in tensor {o.name}"
    | some (r, wrap) =>
      if let some (p, q, z) := wrap then emit loc (.overflow s!"{o.name}({p.toInt}, {q.toInt})" z.toInt w)
      pure ⟨w, shape, r⟩
  | .cmp o a b shape w => do
    unless w = 1 do throwAt loc "cmp result must be i1"
    let x ← evalTExpr loc a
    let y ← dataW loc (← evalTExpr loc b) x.w
    unless x.data.size = numel shape && y.size = numel shape do throwAt loc s!"cmp operand shapes differ from {shape}"
    pure ⟨1, shape, (x.data.zip y).map fun (p, q) => ofBool (cmpBV o p q)⟩
  | .select c a b shape w => do
    let cs ← dataW loc (← evalTExpr loc c) 1
    let x ← dataW loc (← evalTExpr loc a) w
    let y ← dataW loc (← evalTExpr loc b) w
    let n := numel shape
    unless cs.size = n && x.size = n && y.size = n do throwAt loc s!"select operand shapes differ from {shape}"
    pure ⟨w, shape, (cs.zip (x.zip y)).map fun (k, p, q) => if k = 1 then p else q⟩
  | .full v shape w => pure ⟨w, shape, Array.replicate (numel shape) (BitVec.ofInt w v)⟩
  | .ext signed a shape w => do
    let x ← evalTExpr loc a
    unless x.w < w && x.data.size = numel shape do throwAt loc s!"ext of i{x.w} {x.shape} to i{w} {shape}"
    pure ⟨w, shape, x.data.map fun b => if signed then b.signExtend w else b.setWidth w⟩
  | .trunc a shape w addr => do
    let x ← evalTExpr loc a
    unless w < x.w && x.data.size = numel shape do throwAt loc s!"trunc of i{x.w} {x.shape} to i{w} {shape}"
    let r := x.data.map (·.setWidth w)
    if addr then
      if let some i := (Array.range r.size).find? fun i => truncLoses x.data[i]! r[i]! then
        emit loc (.overflow s!"trunc({bvInt x.data[i]!})" (bvInt r[i]!) w)
    pure ⟨w, shape, r⟩

/-! ## Events and statements -/

/-- Number of CTAs in a cluster. -/
def Host.csize (h : Host) : Nat := h.cluster.1 * h.cluster.2.1 * h.cluster.2.2

/-- A cluster rank as a slot owner: `none` is the issuing CTA; a rank outside the cluster is a
kernel error (a remote shared-memory address outside the cluster is undefined in PTX). -/
def ctaOf (loc : Loc) (what : String) (cta : Option Expr) : M Nat := do
  let c ← read
  match cta with
  | none => pure c.rank
  | some e =>
    let r ← evalInt loc e
    if 0 ≤ r && r < c.host.csize then pure r.toNat
    else throwK loc .kernel s!"{what} names CTA rank {r}, outside the cluster of {c.host.csize} CTA(s)"

def slotOf (loc : Loc) (bar : String) (idx : Expr) (cta : Nat) : M Slot := do
  let h := (← read).host
  let i ← evalInt loc idx
  match h.barIdx[bar]? with
  | none => throwAt loc s!"unknown barrier {bar}"
  | some b =>
    let num := (h.barriers[b]?.map (·.num)).getD 0
    if 0 ≤ i && i < num then pure { bar := b, idx := i.toNat, cta }
    else throwK loc .kernel s!"barrier index {bar}[{i}] out of range (num {num})"

/-- The ranks of a CTA mask (`none`: the issuing CTA), ascending. A bit at or above the
cluster size is a kernel error. -/
def ctasOf (loc : Loc) (what : String) (mask : Option Expr) : M (Array Nat) := do
  let c ← read
  match mask with
  | none => pure #[c.rank]
  | some e =>
    let m ← evalInt loc e
    let n := c.host.csize
    if m < 0 || m ≥ 2 ^ n then
      throwK loc .kernel s!"{what}: CTA mask {m} names a CTA outside the cluster of {n} CTA(s)"
    pure <| (Array.range n).filter fun r => (m.toNat >>> r) % 2 == 1

/-- Masked-in lanes of an offset tensor. -/
def activeOffsets (loc : Loc) (offs : TVal) (mask : Option TVal) : M (Array Int) := do
  let vals := offs.data.map bvInt
  match mask with
  | none => pure vals
  | some m =>
    let bits ← dataW loc m 1
    unless bits.size = vals.size do throwAt loc s!"mask shape {m.shape} differs from offsets shape {offs.shape}"
    pure <| (vals.zip bits).filterMap fun (v, k) => if k = 1 then some v else none

def globalAccess (loc : Loc) (tensor : String) (offsets : TExpr) (mask : Option TExpr) :
    M (Nat × Array Int) := do
  let (t, info) ← tensorOf loc tensor
  unless info.decl.kind = .global do throwAt loc s!"pointer access to non-global tensor {tensor}"
  let offs ← evalTExpr loc offsets
  let m ← match mask with
    | none => pure none
    | some e => some <$> evalTExpr loc e
  return (t, ← activeOffsets loc offs m)

def tmaAccess (loc : Loc) (desc : String) (coords : List Expr) : M (Nat × TensorInfo × Array Int) := do
  let (t, info) ← tensorOf loc desc
  unless info.decl.kind = .desc do throwAt loc s!"TMA access to non-descriptor tensor {desc}"
  let cs := ((← evalExprs loc coords).map Val.int).toArray
  unless cs.size = info.rank && info.decl.box.length = info.rank do
    throwAt loc s!"TMA {desc}: {cs.size} coords, box {info.decl.box}, rank {info.rank}"
  return (t, info, cs)

/-- The byte count of an expect_tx or complete_tx. PTX transaction counts are unsigned, so a
negative one does not say what the hardware sees (`unexpected`); the ledger's attribution of
another CTA's bytes also needs them non-negative (Ledger.lean, "Landers"). -/
def txBytes (loc : Loc) (what : String) (e : Expr) : M Int := do
  let n ← evalInt loc e
  if n < 0 then throwAt loc s!"{what}: negative byte count {n} (transaction counts are unsigned)"
  return n

def execEvent : Event → M Unit
  | .bar_init bar idx count loc => do emit loc (.barInit (← slotOf loc bar idx (← read).rank) (← evalInt loc count))
  | .arrive bar idx n perWarp cta loc => do
    let s ← slotOf loc bar idx (← ctaOf loc "arrive" cta)
    let r := (← read).role
    emit loc (.arrive s (n * (if perWarp then r.warps else r.threads_per_issue)) false)
    let st ← get
    let k := st.arrived.getD s.key 0
    for (c, v) in st.chans.toList do
      if st.records[(c, s.key, k)]?.any (· != v) then
        throwK loc .limit s!"handoff {c}: its putter roles recorded different values at their arrive {k + 1} on {bar}[{s.idx}] (KIR change 17)"
    set { st with arrived := st.arrived.insert s.key (k + 1),
                  records := st.chans.fold (init := st.records) fun m c v => m.insert (c, s.key, k) v }
  | .commit_arrive bar idx mask loc => do
    for c in ← ctasOf loc "commit_arrive" mask do
      emit loc (.arrive (← slotOf loc bar idx c) 1 true)
  | .expect_tx bar idx bytes cta loc => do
    emit loc (.expectTx (← slotOf loc bar idx (← ctaOf loc "expect_tx" cta)) (← txBytes loc "expect_tx" bytes))
  | .wait bar idx parity cta loc => do
    let s ← slotOf loc bar idx (← ctaOf loc "wait" cta)
    emit loc (.wait s (← evalInt loc parity))
    modify fun st => { st with waited := st.waited.insert s.key (st.waited.getD s.key 0 + 1),
                               lastWait := some s }
  | .tma_load desc coords bar idx mask leader loc => do
    let (t, info, cs) ← tmaAccess loc desc coords
    let bytes := numel info.decl.box * info.decl.elem_bytes
    -- one global read; every destination's bytes complete on its (pair leader's) slot
    let dests ← ctasOf loc "tma_load" mask
    for (d, k) in dests.zipIdx do
      let s ← slotOf loc bar idx (if leader then d - d % 2 else d)
      emit loc (if k == 0 then .tmaLoad t cs s bytes else .landTx s bytes)
  | .complete_tx bar idx bytes cta loc => do
    emit loc (.landTx (← slotOf loc bar idx (← ctaOf loc "complete_tx" cta)) (← txBytes loc "complete_tx" bytes))
  | .tma_store desc coords loc => do
    let (t, _, cs) ← tmaAccess loc desc coords
    emit loc (.tmaStore t cs)
  | .tma_reduce desc coords loc => do
    let (t, _, cs) ← tmaAccess loc desc coords
    emit loc (.tmaReduce t cs)
  | .gload tensor offsets mask loc => do
    let (t, offs) ← globalAccess loc tensor offsets mask
    emit loc (.gload t offs)
  | .gstore tensor offsets mask loc => do
    let (t, offs) ← globalAccess loc tensor offsets mask
    emit loc (.gstore t offs)
  | .gatomic tensor offsets mask loc => do
    let (t, offs) ← globalAccess loc tensor offsets mask
    emit loc (.gatomic t offs)
  | .work fields loc => do
    let fs ← fields.toArray.mapM fun (k, e) => return (k, ← evalInt loc e)
    modify fun s => { s with work := fs }
    emit loc (.work fs)
  | .tmem inst warp rects mask site loc => do
    let w ← evalInt loc warp
    let rs ← rects.toArray.mapM fun r => do
      let l ← evalInt loc r.lane
      let c ← evalInt loc r.col
      pure (r.use, r.write, l, l + r.nlanes, c, c + r.ncols)
    emit loc (.tmem inst w (← ctasOf loc "tmem" mask) rs site)
  | .tmem_wait ld loc => emit loc (.tmemWait ld)
  | .named_bar id count sync all loc => do
    let r := (← read).role
    let i ← evalInt loc id
    let c ← evalInt loc count
    emit loc (if all then .namedBarAll i c sync (r.warps * 32) else .namedBar i c sync (r.warps * 32))
  | .nb_open kind certain why id count sync loc => do
    let i ← match id with
      | some e => some <$> evalInt loc e
      | none => pure none
    let c ← match count with
      | some e => some <$> evalInt loc e
      | none => pure none
    emit loc (.nbOpen kind certain why i c sync)
  | .nb_rep b _ key loc => emit loc (.nbRep b key)
  | .bulk_store kind warp lane sym src bytes tmap loc => do
    let who : Who := { warp := ← evalInt loc warp, lane := ← lane.mapM (evalInt loc) }
    let lo ← evalInt loc src
    let tm ← tmap.mapM fun (base, a) => return (base, ← evalInt loc a)
    emit loc (.bulkStore kind who sym lo (lo + bytes) tm)
  | .bulk_commit loc => emit loc .bulkCommit
  | .bulk_wait n read loc => emit loc (.bulkWait n read)
  | .smem_write sym offsets mask bytes warp lane sync async cta slot loc => do
    let who : Who := { warp := ← evalInt loc warp, lane := ← lane.mapM (evalInt loc) }
    let offs ← evalTExpr loc offsets
    let m ← match mask with
      | none => pure none
      | some e => some <$> evalTExpr loc e
    let os ← activeOffsets loc offs m
    let rank := (← read).rank
    let sl ← slot.mapM fun (b, i, leader) => slotOf loc b i (if leader then rank - rank % 2 else rank)
    emit loc (.smemWrite who sync async (← ctaOf loc "smem_write" cta) sym
      (os.map fun o => (o, o + bytes)) sl)
  | .tmap_write base addr warp loc => do
    emit loc (.tmapWrite (← evalInt loc warp) base (← evalInt loc addr))
  | .handoff_put chan value loc => do
    let v ← evalInt loc value
    modify fun st => { st with chans := st.chans.insert chan v }
    emit loc (.handoffPut chan)
  | .handoff_get chan var loc => do
    let st ← get
    let some s := st.lastWait
      | throwK loc .limit s!"handoff {chan}: a read before any wait of the role (KIR change 17)"
    let k := st.waited.getD s.key 0 - 1
    match st.records[(chan, s.key, k)]? with
    | some v =>
      set { st with env := st.env.insert var ⟨32, BitVec.ofInt 32 v⟩ }
      emit loc (.handoff chan s k)
    | none =>
      let bar := (((← read).host.barriers[s.bar]?).map (·.decl.name)).getD "?"
      throwK loc .limit s!"handoff {chan}: its putter role recorded no value at its arrive {k + 1} on {bar}[{s.idx}], which this role's wait {k + 1} on that slot pairs with (KIR change 17)"
  | .gsig tensor idx v data loc => do
    let w ← wordOf loc tensor (← evalInt loc idx)
    let x ← evalInt loc v
    emit loc (.gsig w x)
    -- a store of a 64-bit word is a tagged store (tag: its high half)
    if (← read).host.syncWidth w.t == 64 then
      let u := ((BitVec.ofInt 64 x).toNat : Int)
      modify fun st => { st with puts := st.puts.push (w, u / 2 ^ 32, if data then none else some (u % 2 ^ 32)) }
  | .gwait tensor idx tag var loc => do
    let w ← wordOf loc tensor (← evalInt loc idx)
    let t := ((BitVec.ofInt 32 (← evalInt loc tag)).toNat : Int)
    emit loc (.gwait w t)
    -- the word starts at 0, whose tag is 0: a take of tag 0 reads it, unless a store of the
    -- tag with another value half exists (which of the two it reads depends on the schedule)
    let start : Option (Option Int) := if t == 0 then some (some 0) else none
    let known := match (← read).host.puts[(w, t)]?, start with
      | some p, some s0 => if p == s0 then some p else none
      | some p, none => some p
      | none, s0 => s0
    if ((← read).host.puts[(w, t)]?).isSome && start.isSome && known.isNone then
      throwK loc .limit s!"tagged word {tensor}[{w.e}]: its start value and a store both carry tag {t}, with different values"
    match known, var with
    | some (some v), some (x, wd) => modify fun st => { st with env := st.env.insert x ⟨wd, BitVec.ofInt wd v⟩ }
    | some none, some (x, _) =>
      let why := s!"tagged word {tensor}[{w.e}]: the store of tag {t} carries data, not the integer this take's value is used as"
      modify fun st => { st with env := st.env.erase x, poison := st.poison.insert x why }
    | some _, none => pure ()
    | none, _ =>
      throw { msg := s!"take of tag {t} of {tensor}[{w.e}]: no store of it is known (blocked)", loc, context := "",
              stop := true, why := .blocked loc, blocked := true, kind := .limit }
  | .gclaim tensor idx v var wd loc => do
    let w ← wordOf loc tensor (← evalInt loc idx)
    let add ← evalInt loc v
    -- the claimers are the CTAs that claimed on the word in the previous pass, in linear order
    -- (0 of 1 before a pass knows this CTA's turn)
    let c ← read
    let (i, g) : Int × Int := match c.host.claimOrder[w]? with
      | some (m, n) => match m[c.cta]? with
        | some k => (k, n)
        | none => (0, 1)
      | none => (0, 1)
    let st ← get
    let k := st.claims.getD w 0
    let old := ((BitVec.ofInt wd (((k : Int) * g + i) * add)).toNat : Int)
    set { st with claims := st.claims.insert w (k + 1), claimed := st.claimed.push (w, old, add),
                  claimWords := if st.claimWords.contains w then st.claimWords else st.claimWords.push w,
                  env := st.env.insert var ⟨wd, BitVec.ofInt wd old⟩ }
    emit loc (.gclaim w add)
  | .rput chan ord field v loc => do
    let o ← evalInt loc ord
    let x ← evalInt loc v
    let st ← get
    match st.relay[(chan, o, field)]? with
    | some y =>
      if y != x then
        throwK loc .limit s!"relay {chan}: slot {o} field {field} written twice with different values ({y}, {x}): not one write per slot"
    | none => set { st with relay := st.relay.insert (chan, o, field) x }
  | .rget chan ord field var wd loc => do
    let o ← evalInt loc ord
    let st ← get
    match st.relay[(chan, o, field)]? with
    | some x => set { st with env := st.env.insert var ⟨wd, BitVec.ofInt wd x⟩, poison := st.poison.erase var }
    | none =>
      -- unwritten: an error only where the value is used (a speculated load's is not); a
      -- putter of the channel blocked at a take never writes it: the use blocks this role too
      let why := if st.relay.contains (blockedKey chan) then
          s!"{blockedPrefix}{chan}: slot {o} field {field}, read at {loc}, is one its putter role, blocked at a take, never writes"
        else s!"relay {chan}: the value of slot {o} field {field}, read at {loc} before any write of it by a role of the CTA that puts relay slots, or earlier in this role, is used"
      set { st with env := st.env.erase var, poison := st.poison.insert var why }

def setVar (x : String) (v : Val) : M Unit :=
  modify fun s => { s with env := s.env.insert x v }

mutual
def execStmt : Stmt → M Unit
  | .«let» x e loc => do setVar x (← evalExpr loc e)
  | .«for» x lb ub step body loc => do
    let l ← evalExpr loc lb
    let u ← asW loc (← evalExpr loc ub) l.w
    let s ← asW loc (← evalExpr loc step) l.w
    let (li, ui, si) := (l.bv.toInt, u.toInt, s.toInt)
    -- A loop with no trips runs nothing whatever its step; one that starts with a step that
    -- is not positive never reaches its upper bound.
    if si ≤ 0 && li < ui then
      throwK loc .kernel s!"for {x}: non-positive step {si} from {li} below {ui}: the loop never ends"
    let trips := if li < ui then ((ui - li + si - 1) / si).toNat else 0
    for k in [0:trips] do
      setVar x ⟨l.w, BitVec.ofInt l.w (li + k * si)⟩
      execBlock body
  | .«while» c body maxIters loc => do
    let mut exited := false
    for _ in [0:maxIters] do
      if !(← evalCond loc c) then
        exited := true
        break
      execBlock body
    if !exited && (← evalCond loc c) then
      throwAt loc s!"while loop exceeds max_iters = {maxIters} (a hang of the kernel, or a budget too small for the input)"
  | .«if» c t e loc => do
    if ← evalCond loc c then execBlock t else execBlock e
  | .event ev => execEvent ev
  | .assert c msg loc => do
    unless ← evalCond loc c do throwK loc .kernel s!"device assert failed: {msg}"
  | .assume c msg loc => do
    unless ← evalCond loc c do
      throw { msg := s!"assumption does not hold: {msg}", loc, context := "", stop := true,
              why := .assume msg loc }

def execBlock : List Stmt → M Unit
  | [] => pure ()
  | s :: ss => do execStmt s; execBlock ss
end

/-! ## Host setup and the whole run -/

def pidStr (p : Nat × Nat × Nat) : String := s!"({p.1},{p.2.1},{p.2.2})"

/-- The relay slots of a CTA (KIR change 18): (channel, slot, field) -> value. -/
abbrev Relay := Std.HashMap (String × Int × Nat) Int

def runM {α : Type} (h : Host) (pid : Nat × Nat × Nat) (role : Role) (device : Bool) (act : M α)
    (rank : Nat := 0) (records : Handoffs := {}) (cta : Nat := 0) (relay : Relay := {}) :
    EStateM.Result EvalError St α :=
  (act.run { host := h, pid, role, device, rank, cta }).run
    { env := h.params, trace := #[], work := #[], records, relay }

/-- Values of host-side Exprs (params only). -/
def hostInts (h : Host) (es : List Expr) : Except EvalError (Array Int) :=
  let noRole : Role := { name := "host", warps := 1, threads_per_issue := 1, body := [] }
  match runM h (0, 0, 0) noRole false (evalExprs "" es) with
  | .ok vs _ => .ok (vs.map Val.int).toArray
  | .error e _ => .error { e with context := "host" }

/-- Trace of role `ri` of CTA `pid` (linear index `cta`), run with the CTA's handoff records
`records` and relay slots `relay`, and those after it; a role that stops (a `metaOob` event,
an `assume` that does not hold, a blocked take) keeps the events up to it and the reason. -/
def runRole (h : Host) (pid : Nat × Nat × Nat) (ri : Nat) (role : Role) (rank : Nat := 0)
    (records : Handoffs := {}) (cta : Nat := 0) (relay : Relay := {}) :
    Except EvalError (RoleTrace × Handoffs × Relay) :=
  let trace (s : St) (stopped : Bool) (why : StopWhy) (blocked : Bool) : RoleTrace :=
    { role := ri, events := s.trace, stopped, why, claimed := s.claimed, puts := s.puts,
      claimWords := s.claimWords, blocked }
  match runM h pid role true (execBlock role.body) rank records cta relay with
  | .ok () s => .ok (trace s false .metaOob false, s.records, s.relay)
  | .error e s =>
    if e.stop then .ok (trace s true e.why e.blocked, s.records, s.relay)
    else .error { e with context := s!"cta {pidStr pid} role {role.name}" }

def lookup {α : Type} (xs : List (String × α)) (k : String) : Option α :=
  (xs.find? (·.1 == k)).map (·.2)

def hostErr {α : Type} (msg : String) (kind : ErrKind := .unexpected) : Except EvalError α :=
  .error { msg, loc := "", context := "host", kind }

/-- Evaluate params, grid, tensor extents and barrier sizes for one input. -/
def setupHost (k : Kernel) (inp : Input) : Except EvalError Host := do
  let mut params : Std.HashMap String Val := {}
  for p in k.params do
    match lookup inp.params p.name with
    | some v => params := params.insert p.name ⟨p.type.width, BitVec.ofInt p.type.width v⟩
    | none => hostErr s!"input {inp.name} has no value for param {p.name}"
  let h0 : Host := { kernel := k, input := inp, grid := (0, 0, 0), params, tensors := #[],
                     tensorIdx := {}, barriers := #[], barIdx := {} }
  let host := hostInts h0
  let grid ← match inp.grid with
    | some [x, y, z] => pure (x, y, z)
    | some g => hostErr s!"input grid {g} must have 3 entries"
    | none => do
      let g ← host k.grid
      match g.toList with
      | [x, y, z] =>
        if x < 0 || y < 0 || z < 0 then hostErr s!"negative grid {g.toList}" .kernel else pure (x.toNat, y.toNat, z.toNat)
      | _ => hostErr s!"kernel grid {g.toList} must have 3 entries"
  let mut tensors := #[]
  let mut tensorIdx : Std.HashMap String Nat := {}
  for t in k.tensors do
    let shape ← match lookup inp.shapes t.name with
      | some s => pure (s.map Int.ofNat).toArray
      | none => host t.shape
    let strides ← host t.strides
    unless shape.size = strides.size do hostErr s!"tensor {t.name}: shape and strides differ in rank"
    let contents := if t.metadata then (lookup inp.metadata t.name).getD #[] else #[]
    let rowDim := (t.segments.map (·.dim)).getD 0
    let rowStride := strides.getD rowDim 0
    -- Dims with a single coordinate contribute nothing to an offset and are ignored.
    let big := (shape.zip strides).toList.zipIdx.filter fun ((n, _), _) => n > 1
    let rowOuter := ((big.map (·.1.2)).filter (· > rowStride)).toArray.qsort (· > ·)
    let order := (big.toArray.qsort fun ((_, a), _) ((_, b), _) => a > b).map (·.2)
    let spanBelow (j : Nat) (s : Int) : Int :=
      big.foldl (fun acc ((n, si), i) => if i != j && si ≤ s then acc + (n - 1) * si else acc) 0
    let exact := big.all fun ((_, s), j) => 0 < s && spanBelow j s < s
    let info : TensorInfo := { decl := t, shape, strides, contents, rowOuter, order, exact }
    if t.segments.isSome && t.kind == .global && !shape.any (· ≤ 0) && shape.getD rowDim 0 > 1 then
      -- The row of an element offset is read off greedily, dividing by the strides from
      -- the largest down to the row dim's. That is exact only if the row dim's stride and
      -- every larger one exceed the span of all other dims with a smaller or equal stride
      -- (a non-overlapping layout; checked per input, never assumed).
      let rowExact := big.all fun ((_, s), j) =>
        0 ≤ s && ((j != rowDim && s ≤ rowStride) || spanBelow j s < s)
      unless rowExact do
        hostErr s!"segmented tensor {t.name}: strides {strides.toList} for shape {shape.toList} give no unique dim {rowDim} coordinate per offset; rows of offsets are undefined"
    tensorIdx := tensorIdx.insert t.name tensors.size
    tensors := tensors.push info
  let mut barriers := #[]
  let mut barIdx : Std.HashMap String Nat := {}
  for b in k.barriers do
    let v ← host [b.count, b.num]
    let (count, num) := (v.getD 0 0, v.getD 1 0)
    if num < 0 then hostErr s!"barrier {b.name}: negative num {num}"
    barIdx := barIdx.insert b.name barriers.size
    barriers := barriers.push { decl := b, count, num := num.toNat }
  let cluster ← match k.cluster with
    | [x, y, z] =>
      if x = 0 || y = 0 || z = 0 then hostErr s!"cluster dims {k.cluster} must be positive" else pure (x, y, z)
    | c => hostErr s!"cluster dims {c} must have 3 entries"
  let (gx, gy, gz) := grid
  if gx % cluster.1 != 0 || gy % cluster.2.1 != 0 || gz % cluster.2.2 != 0 then
    hostErr s!"grid {pidStr grid} is not a multiple of the cluster dims {pidStr cluster}: the launch fails" .kernel
  pure { h0 with grid, cluster, tensors, tensorIdx, barriers, barIdx }

/-- Rank of CTA `pid` in its cluster. -/
def Host.rankOf (h : Host) (pid : Nat × Nat × Nat) : Nat :=
  let (cx, cy, cz) := h.cluster
  pid.1 % cx + (pid.2.1 % cy) * cx + (pid.2.2 % cz) * cx * cy

/-- The cluster of CTA `pid` (cluster coordinates). -/
def Host.clusterOf (h : Host) (pid : Nat × Nat × Nat) : Nat × Nat × Nat :=
  (pid.1 / h.cluster.1, pid.2.1 / h.cluster.2.1, pid.2.2 / h.cluster.2.2)

/-! ## Handoffs through shared memory (KIR change 17) -/

mutual
/-- The handoff channels a statement puts (`true`) and gets (`false`). -/
def Stmt.handoffs : Stmt → List (Bool × String)
  | .event (.handoff_put c _ _) => [(true, c)]
  | .event (.handoff_get c _ _) => [(false, c)]
  | .«for» _ _ _ _ b _ => stmtsHandoffs b
  | .«while» _ b _ _ => stmtsHandoffs b
  | .«if» _ t e _ => stmtsHandoffs t ++ stmtsHandoffs e
  | _ => []

def stmtsHandoffs : List Stmt → List (Bool × String)
  | [] => []
  | s :: ss => s.handoffs ++ stmtsHandoffs ss
end

/-- The role group of role `ri` (`Role.group`, else its name). -/
def Kernel.groupOf (k : Kernel) (ri : Nat) : String :=
  match k.roles[ri]? with
  | some r => if r.group.isEmpty then r.name else r.group
  | none => ""

/-- The putter roles of each handoff channel and the order the roles of a CTA run in: the
putters first, so their records exist when another role reads them. Putters of a channel
that are not one role group, a channel no role puts, and a putter that reads a channel are
malformed KIR. -/
def handoffPlan (k : Kernel) : Except EvalError (Std.HashMap String (List Nat) × List Nat) := do
  let mut putters : Std.HashMap String (List Nat) := {}
  let mut gets : Array (Nat × String) := #[]
  for (r, ri) in k.roles.zipIdx do
    for (put, c) in stmtsHandoffs r.body do
      if !put then
        gets := gets.push (ri, c)
      else
        let ps := putters.getD c []
        if ps.any (k.groupOf · != k.groupOf ri) then
          hostErr s!"handoff channel {c} is put by roles of two groups"
        unless ps.contains ri do putters := putters.insert c (ps ++ [ri])
  let isPutter (ri : Nat) : Bool := putters.toList.any (·.2.contains ri)
  for (ri, c) in gets do
    if !putters.contains c then hostErr s!"handoff channel {c} is read by role {ri} and put by none"
    if isPutter ri then hostErr s!"role {ri} puts a handoff channel and reads channel {c}"
  let all := List.range k.roles.length
  pure (putters, all.filter isPutter ++ all.filter (!isPutter ·))

/-- What the role traces of a cluster do to the slots of one of its CTAs, per slot key: the
plain `arrive` events of that CTA's roles (role, trace index, arrivals), whether anything
else completes a phase or expects bytes on the slot (an async arrival, transaction bytes, an
arrival from another CTA), and its initial count. -/
structure SlotUse where
  arrives : Array (Nat × Nat × Int) := #[]
  other : Bool := false
  init : Option Int := none

def slotUses (r : Run) (c : CtaTrace) : Std.HashMap (Nat × Nat × Nat) SlotUse := Id.run do
  let cl := r.host.clusterOf c.pid
  let mut m : Std.HashMap (Nat × Nat × Nat) SlotUse := {}
  for c' in r.ctas do
    if r.host.clusterOf c'.pid != cl then continue
    for t in c'.roles do
      for (e, p) in t.events.zipIdx do
        let (s, f) : Option Slot × (SlotUse → SlotUse) := match e.ev with
          | .arrive s n false =>
            (some s, if c'.pid == c.pid then fun u => { u with arrives := u.arrives.push (t.role, p, n) }
                     else fun u => { u with other := true })
          | .barInit s n => (some s, fun u => { u with init := some n, other := u.other || u.init.any (· != n) })
          | .arrive s _ true | .expectTx s _ | .landTx s _ | .tmaLoad _ _ s _ =>
            (some s, fun u => { u with other := true })
          | _ => (none, id)
        if let some s := s then
          if s.cta == c.rank then m := m.insert s.key (f (m.getD s.key {}))
  return m

/-- The phases of a slot (use `u`) and the waits of trace `tw` on it, if every arrival on the
slot is a plain `arrive` of the roles of one group `g` of the CTA, each role arriving with one
count per event and those counts summing to the slot's count, and the waits of `tw` on the
slot wait for parities 0, 1, 0, ... in order. The roles of a group are taken in lockstep, as
P0 takes the threads of one role (references/specs.md): then phase k completes with the k-th
arrive of each role of `g`, and the k-th wait of `tw` returns after that (PTX ISA, mbarrier
phases). Returns `g` and, per wait k of `tw` (while every role of `g` has a k-th arrive), its
trace index and the trace index of the k-th arrive of each role of `g`. -/
def pairing (k : Kernel) (u : SlotUse) (s : Slot) (tw : RoleTrace) :
    Option (String × Array (Nat × Array (Nat × Nat))) := do
  guard !u.other
  let cnt ← u.init
  let (r0, _, _) ← u.arrives[0]?
  let g := k.groupOf r0
  guard (u.arrives.all fun (r, _, _) => k.groupOf r == g)
  let per := (u.arrives.map (·.1)).toList.eraseDups.map fun r => (r, u.arrives.filter (·.1 == r))
  let ns := per.map fun (_, xs) => (xs[0]?.map (·.2.2)).getD 0
  guard ((per.zip ns).all fun ((_, xs), n) => xs.all (·.2.2 == n))
  guard (ns.foldl (· + ·) 0 == cnt)
  let waits := tw.events.zipIdx.filterMap fun (e, p) => match e.ev with
    | .wait s' par => if s'.key == s.key then some (p, par) else none
    | _ => none
  guard (waits.zipIdx.all fun ((_, par), j) => par == ((j % 2 : Nat) : Int))
  let m := (per.map (·.2.size)).foldl min (waits.size)
  return (g, (waits.extract 0 m).zipIdx.map fun ((p, _), j) =>
    (p, per.toArray.map fun (r, xs) => (r, (xs[j]?.map (·.2.1)).getD 0)))

/-- The handoff checks of run `r` (references/kir.md, change 17): each `handoff` read of a
role, its k-th wait on slot s, (1) pairs with the k-th arrive on s of every putter role of the
channel (`pairing`, the putters' group arriving on s), at which the value read was recorded,
so those arrives happened before the read; (2) is not overwritten before it happens: each
putter's next put of the channel after that arrive follows a wait of the putter that pairs
with an arrive of the reading role after the read (`pairing` on that wait's slot, the reading
role's group arriving on it). A read either is not shown for is a model limit (an evaluation
error). The pairing of each (role, slot) and the puts of each putter are computed once per
CTA. -/
def checkHandoffs (k : Kernel) (r : Run) (putters : Std.HashMap String (List Nat)) :
    Except EvalError Unit := do
  for c in r.ctas do
    let reads := c.roles.flatMap fun t => t.events.zipIdx.filterMap fun (e, p) => match e.ev with
      | .handoff ch s j => some (t, p, ch, s, j)
      | _ => none
    if reads.isEmpty then continue
    let uses := slotUses r c
    -- (putter role, channel) -> the trace indices of its puts, ascending
    let mut puts : Std.HashMap (Nat × String) (Array Nat) := {}
    for t in c.roles do
      for (e, i) in t.events.zipIdx do
        if let .handoffPut ch := e.ev then
          puts := puts.insert (t.role, ch) ((puts.getD (t.role, ch) #[]).push i)
    let mut memo : Std.HashMap (Nat × (Nat × Nat × Nat)) (Option (String × Array (Nat × Array (Nat × Nat)))) := {}
    for (t, p, ch, s, j) in reads do
      let fail (why : String) : Except EvalError Unit :=
        .error { msg := s!"handoff {ch}: {why} (KIR change 17)", loc := (t.events[p]?.map (·.loc)).getD "",
                 context := s!"cta {pidStr c.pid}", kind := .limit }
      let ps := putters.getD ch []
      let pr ← match memo[(t.role, s.key)]? with
        | some x => pure x
        | none => do
          let x := pairing k (uses.getD s.key {}) s t
          memo := memo.insert (t.role, s.key) x
          pure x
      let some (g, pairs) := pr
        | fail "the waits of the reading role on the slot of its last wait do not pair with the arrives on it (another arrival, transaction bytes, counts that do not sum to the slot's, or waits out of parity order)"
      let some (_, arrs) := pairs[j]?
        | fail "its wait has no arrive of every role of the arriving group to pair with"
      unless ps.all (k.groupOf · == g) && ps.all (fun q => arrs.any (·.1 == q)) do
        fail "the arrives its wait pairs with are not those of every putter role"
      for q in ps do
        let some P := c.roles.find? (·.role == q) | fail "no trace of a putter role"
        let a := ((arrs.find? (·.1 == q)).map (·.2)).getD 0
        let some n := (puts.getD (q, ch) #[]).find? (· > a) | continue
        let mut ordered := false
        for w in [a + 1:n] do
          if let some (.wait d _) := (P.events[w]?.map (·.ev) : Option Ev) then
            let pw ← match memo[(q, d.key)]? with
              | some x => pure x
              | none => do
                let x := pairing k (uses.getD d.key {}) d P
                memo := memo.insert (q, d.key) x
                pure x
            if let some (g', ps') := pw then
              if g' == k.groupOf t.role && ps'.any (fun (w', as') =>
                  w' == w && as'.any fun (q', a') => q' == t.role && a' > p) then
                ordered := true
                break
        unless ordered do
          fail s!"a putter's next put of the channel ({(P.events[n]?.map (·.loc)).getD ""}) follows no wait that pairs with an arrive of the reading role after this read"

/-! ## Relay slots, tagged words and claims (KIR change 18)

The roles of a CTA run in order (`roleOrder`), the CTA's relay slots carried from role to role:
a slot is written once per CTA (trusted), so the order only decides that every get sees it. A
take (`Event.gwait`) reads the value half of the store of its tag, which another CTA makes:
`evalSyncPasses` evaluates the launch in passes, each from the stores of the previous one
(`Host.puts`; none in the first); a take whose store is not known stops its role, blocked. A
claim reads its turn among the CTAs that claimed on the word in the previous pass
(`Host.claimOrder`). The passes end when a pass knows the stores and claimers it started from
(a fixed point): then every take of a role that ran on read the store of its tag, and a role
still blocked waits for a tag no CTA of the launch stores in this run (P0-7's hang,
`Run.hangs`). Trusted, as the relay slots: a tagged word's store of a tag is the one store of
that tag, read by every take of the tag. -/

/-- The relay channels whose slots statements `ss` get (`get`) or put. -/
partial def relayChans (get : Bool) (ss : List Stmt) : List String :=
  ss.foldl (fun acc s => acc ++ match s with
    | .event (.rput chan ..) => if get then [] else [chan]
    | .event (.rget chan ..) => if get then [chan] else []
    | .«for» _ _ _ _ body _ => relayChans get body
    | .«while» _ body _ _ => relayChans get body
    | .«if» _ t e _ => relayChans get t ++ relayChans get e
    | _ => []) []

/-- The order the roles of a CTA run in: the handoff plan's (`handoffPlan`: handoff putters
first), the roles that put relay slots moved before the others; malformed if that breaks the
plan. -/
def roleOrder (k : Kernel) (putters : Std.HashMap String (List Nat)) (order : List Nat) :
    Except EvalError (List Nat) := do
  let relayPut (ri : Nat) : Bool := !(relayChans false ((k.roles[ri]?.map (·.body)).getD [])).isEmpty
  let out := order.filter relayPut ++ order.filter (!relayPut ·)
  let isPutter (ri : Nat) : Bool := putters.toList.any (·.2.contains ri)
  if out.dropWhile isPutter |>.any isPutter then
    hostErr "a role that puts handoff channels after a role that puts relay slots"
  pure out

/-- Every CTA of the grid (x-fastest order), each CTA's roles in `order`. -/
def evalAll (h : Host) (order : List Nat) : Except EvalError (Array CtaTrace) := do
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
            let (t, rs, rl) ← runRole h (x, y, z) ri role rank records cta relay
            traces := traces.set! ri t
            records := rs
            -- a putter blocked at a take never writes its later slots
            relay := if t.blocked then (relayChans false role.body).foldl (fun m c => m.insert (blockedKey c) 1) rl
              else rl
        ctas := ctas.push { pid := (x, y, z), roles := traces, rank }
  pure ctas

/-- The tagged stores of a run: (word, tag) -> value half; two stores of one tag with
different values are a model limit. -/
def putsOf (h : Host) (ctas : Array CtaTrace) : Except EvalError (Std.HashMap (Word × Int) (Option Int)) := do
  let mut m : Std.HashMap (Word × Int) (Option Int) := {}
  for c in ctas do
    for t in c.roles do
      for (w, tag, v) in t.puts do
        match m[(w, tag)]? with
        | some v0 =>
          if v0 != v then
            let name := (h.tensors[w.t]?.map (·.decl.name)).getD "?"
            throw { msg := s!"tagged word {name}[{w.e}]: tag {tag} stored with two values ({v0}, {v})",
                    loc := "", context := "run", kind := .limit }
        | none => m := m.insert (w, tag) v
  return m

/-- The claimers of a run: per word, the CTAs that claimed on it, in linear order. -/
def claimOrderOf (ctas : Array CtaTrace) : Std.HashMap Word (Std.HashMap Nat Nat × Nat) := Id.run do
  let mut by_ : Std.HashMap Word (Array Nat) := {}
  for (c, ci) in ctas.zipIdx do
    for t in c.roles do
      for w in t.claimWords do
        let xs := by_.getD w #[]
        if !xs.contains ci then by_ := by_.insert w (xs.push ci)
  let mut out := {}
  for (w, xs) in by_.toList do
    out := out.insert w (xs.zipIdx.foldl (init := {}) (fun m (ci, k) => m.insert ci k), xs.size)
  return out

def samePuts (a b : Std.HashMap (Word × Int) (Option Int)) : Bool :=
  a.size == b.size && a.toList.all fun (k, v) => b[k]? == some v

def sameOrder (a b : Std.HashMap Word (Std.HashMap Nat Nat × Nat)) : Bool :=
  a.size == b.size && a.toList.all fun (w, (m, n)) => match b[w]? with
    | some (m', n') => n == n' && m.size == m'.size && m.toList.all fun (k, v) => m'[k]? == some v
    | none => false

/-- The passes of a run with sync tensors (module section above). -/
def evalSyncPasses (h : Host) (order : List Nat) (maxPasses : Nat := 256) : Except EvalError Run := do
  let mut h := h
  for _ in [0:maxPasses] do
    let ctas ← evalAll h order
    let puts ← putsOf h ctas
    let claims := claimOrderOf ctas
    if samePuts puts h.puts && sameOrder claims h.claimOrder then return { host := h, ctas }
    h := { h with puts, claimOrder := claims }
  throw { msg := s!"tagged words and claims: no pass of {maxPasses} starts from the stores and claimers it ends with",
          loc := "", context := "run", kind := .limit }

/-- The claim assignment is an outcome of the fetch-adds iff, per word, the old values its
claims read are exactly `0, v, …, (T - 1) v` (T claims of `v`): every claiming CTA claims until
its turn runs past the work, so no value is skipped. Why not, else `none`. -/
def claimsFeasible (r : Run) : Option String := Id.run do
  let mut byWord : Std.HashMap Word (Array (Int × Int)) := {}
  let mut order : Array Word := #[]
  for c in r.ctas do
    for t in c.roles do
      for (w, old, add) in t.claimed do
        if !byWord.contains w then order := order.push w
        byWord := byWord.insert w ((byWord.getD w #[]).push (old, add))
  for w in order do
    let xs := byWord.getD w #[]
    let name := s!"{(r.host.tensors[w.t]?.map (·.decl.name)).getD "?"}[{w.e}]"
    let v := (xs[0]?.map (·.2)).getD 1
    if v == 0 || xs.any (·.2 != v) then return some s!"the claims on {name} add different or zero amounts"
    let m : Int := 2 ^ r.host.syncWidth w.t
    let ds := xs.map fun (o, _) => o % m
    if ds.any (· % v != 0) then return some s!"a claim on {name} reads a value off the chain j x {v}"
    let ks := (ds.map (· / v)).qsort (· < ·)
    for (k, j) in ks.zipIdx do
      if k != (j : Int) then
        return some s!"the {xs.size} claims on {name} read j x {v} for j = {(ks.toList.take 6)}…, not 0..{xs.size - 1} (a CTA stops claiming before its turn runs past the work)"
  return none

/-- The CTAs each CTA of `r` reaches (KIR change 18): t takes a tag a store of s carries,
through any chain of CTAs; the CTAs of one cluster, and of one `Input.syncGroups` group, reach
each other. -/
def reaches (r : Run) : Array (Array Bool) := Id.run do
  let n := r.ctas.size
  let mut storers : Std.HashMap (Word × Int) (Array Nat) := {}
  for (c, ci) in r.ctas.zipIdx do
    for t in c.roles do
      for (w, tag, _) in t.puts do
        let xs := storers.getD (w, tag) #[]
        unless xs.contains ci do storers := storers.insert (w, tag) (xs.push ci)
  let mut succ : Array (Array Nat) := Array.replicate n #[]
  for (c, ci) in r.ctas.zipIdx do
    for t in c.roles do
      for e in t.events do
        if let .gwait w tag := e.ev then
          for s in storers.getD (w, tag) #[] do
            unless s == ci || succ[s]!.contains ci do succ := succ.modify s (·.push ci)
  let mut groups : List (List Nat) := r.host.input.syncGroups
  if r.host.csize > 1 then
    let mut byCluster : Std.HashMap (Nat × Nat × Nat) (List Nat) := {}
    for (c, ci) in r.ctas.zipIdx do
      let k := r.host.clusterOf c.pid
      byCluster := byCluster.insert k (ci :: byCluster.getD k [])
    groups := groups ++ byCluster.toList.map (·.2)
  for g in groups do
    for a in g do
      for b in g do
        if a != b && a < n && b < n && !succ[a]!.contains b then succ := succ.modify a (·.push b)
  let mut out := #[]
  for s in [0:n] do
    let mut seen := Array.replicate n false
    let mut stack := succ[s]!
    while !stack.isEmpty do
      let x := stack.back!
      stack := stack.pop
      unless seen[x]! do
        seen := seen.set! x true
        stack := stack ++ succ[x]!
    out := out.push seen
  return out

/-- The representative claim order (`Event.gclaim`: in each round the claiming CTAs claim in
linear turn order) is consistent with the order the run's synchronization imposes between
CTAs, conservatively: where CTA s reaches CTA t (`reaches`), every claim of s on a word comes
before every claim of t on it, i.e. s claims on it once, in an earlier turn. Why not, else
`none`. -/
def claimTurnsFeasible (r : Run) : Option String := Id.run do
  if r.host.claimOrder.isEmpty then return none
  let mut count : Std.HashMap (Word × Nat) Nat := {}
  for (c, ci) in r.ctas.zipIdx do
    for t in c.roles do
      for (w, _, _) in t.claimed do
        count := count.insert (w, ci) (count.getD (w, ci) 0 + 1)
  let reach := reaches r
  for (w, (turn, _)) in r.host.claimOrder.toList do
    let name := s!"{(r.host.tensors[w.t]?.map (·.decl.name)).getD "?"}[{w.e}]"
    for (s, ts) in turn.toList do
      for (t, tt) in turn.toList do
        if s != t && (reach[s]?.bind (·[t]?)).getD false then
          let ks := count.getD (w, s) 0
          if ks > 1 || ts ≥ tt then
            return some s!"CTA {s} reaches CTA {t} (tagged words, cluster or protocol region), but claims on {name} {ks} time(s) in turn {ts} against turn {tt}: the turns may contradict that order"
  return none

/-- Evaluate every role of every CTA of `k` on `inp` (`evalAll`; a kernel with sync tensors in
passes, `evalSyncPasses`). -/
def eval (k : Kernel) (inp : Input) : Except EvalError Run := do
  let h ← setupHost k inp
  let (putters, order) ← handoffPlan k
  let order ← roleOrder k putters order
  let r : Run ← if h.hasSync then evalSyncPasses h order else pure { host := h, ctas := ← evalAll h order }
  unless putters.isEmpty do checkHandoffs k r putters
  if let some why := claimsFeasible r then
    throw { msg := s!"the claim assignment is not an outcome of the fetch-adds: {why}",
            loc := "", context := "run", kind := .limit }
  if let some why := claimTurnsFeasible r then
    throw { msg := s!"the representative claim order: {why}", loc := "", context := "run", kind := .limit }
  pure r

end KV
