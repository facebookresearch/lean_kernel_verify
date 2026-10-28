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
  read. The other roles and CTAs run to completion (roles share no values).
* Verdict errors (evaluation stops: every spec's theorem is refuted), each of a kind
  (`ErrKind`, references/specs.md "Evaluation errors"). `kernel`, a certain error of the
  kernel or its host on the input, so every spec FAILs: a failed device `assert`, a barrier
  index out of range, a `for` that starts below its upper bound with a step that is not
  positive (it never ends; with no trips it runs nothing), a negative grid dimension.
  `limit`, a known limit of the model, so no spec is decided and it is not an error:
  division or remainder by zero (in a tensor expression on any lane, masked or not; the
  GPU computes an unspecified value, which the kernel may never use). `unexpected`, the
  case or LKV needs a fix, so no spec is decided and it is an ERROR: `while` exceeding
  `max_iters` (a hang of the kernel, or a budget too small for the input), an in-extent
  metadata `load` at a negative offset or one the input gives no value for; in the host
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

/-! ## Traces -/

/-- One mbarrier: barrier array index into `Host.barriers`, element index, and the rank in the
cluster of the CTA whose shared memory holds it (0 without clusters). -/
structure Slot where
  bar : Nat
  idx : Nat
  cta : Nat := 0
  deriving Inhabited

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

structure TEvent where
  ev : Ev
  loc : Loc
  /-- Current work item. -/
  work : Array (String × Int)

structure RoleTrace where
  role : Nat
  events : Array TEvent
  /-- The role stopped at a `metaOob` event (its last); later events are unknown. -/
  stopped : Bool

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

/-- Every role of every CTA ran to its end (no out-of-bounds metadata load). -/
def Run.complete (r : Run) : Bool := r.ctas.all (·.complete)

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
  /-- Raised after a `metaOob` event: ends the role, not the evaluation. -/
  stop : Bool := false
  kind : ErrKind := .unexpected

/-! ## Evaluation monad -/

structure Ctx where
  host : Host
  pid : Nat × Nat × Nat
  role : Role
  /-- Rank of the CTA in its cluster. -/
  rank : Nat := 0
  /-- `false` for host-side Exprs (grid, shapes, strides, barrier sizes). -/
  device : Bool

structure St where
  env : Std.HashMap String Val
  trace : Array TEvent
  work : Array (String × Int)

abbrev M := ReaderT Ctx (EStateM EvalError St)

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

/-! ## Expressions -/

mutual
def evalExpr (loc : Loc) : Expr → M Val
  | .const v w => pure ⟨w, BitVec.ofInt w v⟩
  | .var x w => do
    match (← get).env[x]? with
    | some v => if v.w = w then pure v else throwAt loc s!"variable {x} is i{v.w}, used as i{w}"
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

def execEvent : Event → M Unit
  | .bar_init bar idx count loc => do emit loc (.barInit (← slotOf loc bar idx (← read).rank) (← evalInt loc count))
  | .arrive bar idx n perWarp cta loc => do
    let s ← slotOf loc bar idx (← ctaOf loc "arrive" cta)
    let r := (← read).role
    emit loc (.arrive s (n * (if perWarp then r.warps else r.threads_per_issue)) false)
  | .commit_arrive bar idx mask loc => do
    for c in ← ctasOf loc "commit_arrive" mask do
      emit loc (.arrive (← slotOf loc bar idx c) 1 true)
  | .expect_tx bar idx bytes cta loc => do
    emit loc (.expectTx (← slotOf loc bar idx (← ctaOf loc "expect_tx" cta)) (← evalInt loc bytes))
  | .wait bar idx parity cta loc => do
    emit loc (.wait (← slotOf loc bar idx (← ctaOf loc "wait" cta)) (← evalInt loc parity))
  | .tma_load desc coords bar idx mask leader loc => do
    let (t, info, cs) ← tmaAccess loc desc coords
    let bytes := numel info.decl.box * info.decl.elem_bytes
    -- one global read; every destination's bytes complete on its (pair leader's) slot
    let dests ← ctasOf loc "tma_load" mask
    for (d, k) in dests.zipIdx do
      let s ← slotOf loc bar idx (if leader then d - d % 2 else d)
      emit loc (if k == 0 then .tmaLoad t cs s bytes else .landTx s bytes)
  | .complete_tx bar idx bytes cta loc => do
    emit loc (.landTx (← slotOf loc bar idx (← ctaOf loc "complete_tx" cta)) (← evalInt loc bytes))
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
  | .named_bar id count sync loc => do
    let r := (← read).role
    emit loc (.namedBar (← evalInt loc id) (← evalInt loc count) sync (r.warps * 32))

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

def execBlock : List Stmt → M Unit
  | [] => pure ()
  | s :: ss => do execStmt s; execBlock ss
end

/-! ## Host setup and the whole run -/

def pidStr (p : Nat × Nat × Nat) : String := s!"({p.1},{p.2.1},{p.2.2})"

def runM {α : Type} (h : Host) (pid : Nat × Nat × Nat) (role : Role) (device : Bool) (act : M α)
    (rank : Nat := 0) : EStateM.Result EvalError St α :=
  (act.run { host := h, pid, role, device, rank }).run { env := h.params, trace := #[], work := #[] }

/-- Values of host-side Exprs (params only). -/
def hostInts (h : Host) (es : List Expr) : Except EvalError (Array Int) :=
  let noRole : Role := { name := "host", warps := 1, threads_per_issue := 1, body := [] }
  match runM h (0, 0, 0) noRole false (evalExprs "" es) with
  | .ok vs _ => .ok (vs.map Val.int).toArray
  | .error e _ => .error { e with context := "host" }

/-- Trace of role `ri` of CTA `pid`; a role that stops at a `metaOob` event keeps the
events up to it. -/
def runRole (h : Host) (pid : Nat × Nat × Nat) (ri : Nat) (role : Role) (rank : Nat := 0) :
    Except EvalError RoleTrace :=
  match runM h pid role true (execBlock role.body) rank with
  | .ok () s => .ok { role := ri, events := s.trace, stopped := false }
  | .error e s =>
    if e.stop then .ok { role := ri, events := s.trace, stopped := true }
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

/-- Evaluate every role of every CTA of `k` on `inp` (CTAs in x-fastest order). -/
def eval (k : Kernel) (inp : Input) : Except EvalError Run := do
  let h ← setupHost k inp
  let (gx, gy, gz) := h.grid
  let mut ctas := Array.emptyWithCapacity (gx * gy * gz)
  for z in [0:gz] do
    for y in [0:gy] do
      for x in [0:gx] do
        let rank := h.rankOf (x, y, z)
        let mut roles := #[]
        for (r, ri) in k.roles.zipIdx do
          roles := roles.push (← runRole h (x, y, z) ri r rank)
        ctas := ctas.push { pid := (x, y, z), roles, rank }
  pure { host := h, ctas }

end KV
