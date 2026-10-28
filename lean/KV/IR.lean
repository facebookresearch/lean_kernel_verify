/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
KIR abstract syntax: one constructor per KIR node of `references/kir.md`, same names
and field order (table fields first, then `w`, then `addr`, then `loc`).
`frontend/emit_lean.py` produces values of these types from KIR JSON; nothing here is
kernel specific. JSON details fixed by `frontend/kir.py` (inline event statements,
`w` on tensor expressions, defaults) are mirrored here. The checker sees one input at a
time, so the case-level `elision_flags` are a field of every `Input`.
-/
namespace KV

/-- IR location `file:line` ("" when the IR has none). -/
abbrev Loc := String

inductive BinOp where
  | add | sub | mul | sdiv | udiv | srem | urem | shl | ashr | lshr
  | and | or | xor | smin | smax | umin | umax
  deriving Repr

/-- KIR spelling of the op. -/
def BinOp.name : BinOp → String
  | .add => "add" | .sub => "sub" | .mul => "mul" | .sdiv => "sdiv" | .udiv => "udiv"
  | .srem => "srem" | .urem => "urem" | .shl => "shl" | .ashr => "ashr" | .lshr => "lshr"
  | .and => "and" | .or => "or" | .xor => "xor" | .smin => "smin" | .smax => "smax"
  | .umin => "umin" | .umax => "umax"

inductive CmpOp where
  | eq | ne | slt | sle | sgt | sge | ult | ule | ugt | uge
  deriving Repr

/-- Scalar integer expression; `w` is the bit width (1, 32 or 64). `addr` is kept only on
`bin` and `trunc`, the nodes where it has a meaning (overflow events for add/sub/mul/shl
and lossy truncation). -/
inductive Expr where
  | const (v : Int) (w : Nat)
  | var (name : String) (w : Nat)
  | pid (axis : Nat) (w : Nat)
  | nprog (axis : Nat) (w : Nat)
  | bin (o : BinOp) (a b : Expr) (w : Nat) (addr : Bool)
  | cmp (o : CmpOp) (a b : Expr) (w : Nat)
  | select (c a b : Expr) (w : Nat)
  | ext (signed : Bool) (a : Expr) (w : Nat)
  | trunc (a : Expr) (w : Nat) (addr : Bool)
  | load (tensor : String) (index : List Expr) (w : Nat)
  deriving Repr, Inhabited

/-- Tensor-valued index expression (TTGIR), evaluated per coordinate, row-major. -/
inductive TExpr where
  | range (start «end» : Int) (shape : List Nat) (w : Nat)
  | splat (a : Expr) (shape : List Nat) (w : Nat)
  | bcast (a : TExpr) (axis : Nat) (shape : List Nat) (w : Nat)
  | expand (a : TExpr) (axis : Nat) (shape : List Nat) (w : Nat)
  | bin (o : BinOp) (a b : TExpr) (shape : List Nat) (w : Nat) (addr : Bool)
  | cmp (o : CmpOp) (a b : TExpr) (shape : List Nat) (w : Nat)
  | select (c a b : TExpr) (shape : List Nat) (w : Nat)
  | full (v : Int) (shape : List Nat) (w : Nat)
  | ext (signed : Bool) (a : TExpr) (shape : List Nat) (w : Nat)
  | trunc (a : TExpr) (shape : List Nat) (w : Nat) (addr : Bool)
  deriving Repr, Inhabited

/-- A tcgen05 instruction that accesses tensor memory (P1 KIR only, P1-1). -/
inductive TmemInst where
  | ld | st | cp | mma
  deriving Repr, BEq, DecidableEq, Inhabited

/-- What a TMEM rectangle is to its instruction: `data` (ld/st/cp), the MMA accumulator
`d`, the MMA A operand from TMEM `a`, or MMA scale factors `sf`. -/
inductive RectUse where
  | data | d | a | sf
  deriving Repr, BEq, DecidableEq, Inhabited

/-- One TMEM rectangle of an access: lanes `[lane, lane + nlanes)` x 32-bit columns
`[col, col + ncols)`, read or written. -/
structure TmemRect where
  use : RectUse
  write : Bool
  lane : Expr
  nlanes : Nat
  col : Expr
  ncols : Nat
  deriving Repr, Inhabited

/-- The tensor-memory allocation of a P1 KIR: `cols` columns; `zeroBase`: addresses are
absolute (the fake-zero-base idiom, only with a 512-column allocation), else relative to
the allocation. -/
structure TmemInfo where
  cols : Nat
  zeroBase : Bool
  deriving Repr, Inhabited

/-- Barrier events name a slot of the barrier array `bar` in a CTA of the cluster: `cta` is
its rank in the cluster (`none`: the issuing CTA), `cta_mask` a set of ranks (bit r: rank r;
`none`: the issuing CTA). -/
inductive Event where
  | bar_init (bar : String) (idx count : Expr) (loc : Loc)
  | arrive (bar : String) (idx : Expr) (n : Nat) (per_warp : Bool) (cta : Option Expr) (loc : Loc)
  /-- The completion of the role's prior async MMAs arrives once on the slot of every CTA in
  `cta_mask` (tcgen05.commit, multicast with a mask). -/
  | commit_arrive (bar : String) (idx : Expr) (cta_mask : Option Expr) (loc : Loc)
  | expect_tx (bar : String) (idx bytes : Expr) (cta : Option Expr) (loc : Loc)
  | wait (bar : String) (idx parity : Expr) (cta : Option Expr) (loc : Loc)
  /-- A TMA load whose box lands in every CTA of `cta_mask` (multicast). Each destination
  CTA's bytes complete on its slot, or with `leader` (cta_group::2) on the slot of the even
  CTA of its pair (rank with bit 0 cleared). -/
  | tma_load (desc : String) (coords : List Expr) (bar : String) (idx : Expr) (cta_mask : Option Expr)
      (leader : Bool) (loc : Loc)
  /-- `bytes` transaction bytes completing on the slot of CTA `cta`, asynchronously, from a copy
  without a global source (a shared::cluster bulk copy or st.async into another CTA). -/
  | complete_tx (bar : String) (idx bytes : Expr) (cta : Option Expr) (loc : Loc)
  | tma_store (desc : String) (coords : List Expr) (loc : Loc)
  | tma_reduce (desc : String) (coords : List Expr) (loc : Loc)
  | gload (tensor : String) (offsets : TExpr) (mask : Option TExpr) (loc : Loc)
  | gstore (tensor : String) (offsets : TExpr) (mask : Option TExpr) (loc : Loc)
  | gatomic (tensor : String) (offsets : TExpr) (mask : Option TExpr) (loc : Loc)
  | work (fields : List (String × Expr)) (loc : Loc)
  /-- A tcgen05.ld/st/cp/mma (P1 KIR): `warp` is the issuing warp (its lane quarter); the
  rectangles are in the TMEM of every CTA in `cta_mask` (a cta_group::2 cp/mma: both CTAs of
  the pair); `site` names the nearest enclosing kernel source line (informational). -/
  | tmem (inst : TmemInst) (warp : Expr) (rects : List TmemRect) (cta_mask : Option Expr) (site : Loc) (loc : Loc)
  /-- tcgen05.wait::ld (`ld = true`) or wait::st of the role (P1 KIR). -/
  | tmem_wait (ld : Bool) (loc : Loc)
  /-- bar.sync (`sync = true`) or bar.arrive of the whole role on named barrier `id` with
  `count` threads (P1 KIR). -/
  | named_bar (id count : Expr) (sync : Bool) (loc : Loc)
  deriving Repr, Inhabited

inductive Stmt where
  | «let» (var : String) (expr : Expr) (loc : Loc)
  | «for» (var : String) (lb ub step : Expr) (body : List Stmt) (loc : Loc)
  | «while» (cond : Expr) (body : List Stmt) (max_iters : Nat) (loc : Loc)
  | «if» (cond : Expr) («then» «else» : List Stmt) (loc : Loc)
  | event (ev : Event)
  | assert (cond : Expr) (msg : String) (loc : Loc)
  deriving Repr, Inhabited

inductive TensorKind where
  | global | desc
  deriving Repr, BEq, DecidableEq, Inhabited

inductive TensorRole where
  | input | output | workspace
  deriving Repr, DecidableEq, Inhabited

/-- Rows of dim `dim` split into segments `[offsets[z], offsets[z+1])`; the segment of an
access is the value of work field `per` at the time of the access. -/
structure Segments where
  dim : Nat
  offsets : String
  per : String
  deriving Repr, Inhabited

structure Tensor where
  name : String
  kind : TensorKind
  elem_bytes : Nat
  shape : List Expr
  strides : List Expr
  segments : Option Segments
  role : TensorRole
  metadata : Bool
  /-- TMA box extents (`desc` only; `[]` for `global`). -/
  box : List Nat
  deriving Repr, Inhabited

structure Barrier where
  name : String
  count : Expr
  num : Expr
  deriving Repr, Inhabited

structure Role where
  name : String
  warps : Nat
  threads_per_issue : Nat
  body : List Stmt
  deriving Repr

inductive IntType where
  | i1 | i32 | i64
  deriving Repr

def IntType.width : IntType → Nat
  | .i1 => 1 | .i32 => 32 | .i64 => 64

structure Param where
  name : String
  type : IntType
  deriving Repr

inductive Ir where
  | ttgir | «cute-mlir»
  deriving Repr

/-- The KIR top level without `case` (inputs are passed to the checker separately). -/
structure Kernel where
  kir_version : Nat
  kernel : String
  source : String
  ir : Ir
  grid : List Expr
  params : List Param
  tensors : List Tensor
  barriers : List Barrier
  roles : List Role
  /-- The TMEM allocation of a P1 KIR (`none` for a P0 KIR). -/
  tmem : Option TmemInfo := none
  /-- Thread-block cluster dims (x, y, z): consecutive CTAs form a cluster; `[1, 1, 1]`
  without clusters. -/
  cluster : List Nat := [1, 1, 1]
  deriving Repr

/-- P0-3 target region of one output tensor: rows of its segment dim (dim 0 without
segments), all other coordinates. `rows` ranges are half-open `[lo, hi)`. `unchecked` is
the explicit opt-out: P0-3 checks nothing for the tensor and reports a note. -/
inductive Intended where
  | all_rows_of_segments
  | all
  | rows (ranges : List (Nat × Nat))
  | unchecked
  deriving Repr

structure Input where
  name : String
  params : List (String × Int)
  grid : Option (List Nat)
  metadata : List (String × Array Int)
  flags : List (String × Bool)
  shapes : List (String × List Nat)
  intended_writes : List (String × Intended)
  /-- The case's `elision_flags`: host-selected flags that select a mask-elided fast path. -/
  elision_flags : List String
  deriving Repr

/-- The elision flags this input sets, e.g. `["FULL_TILES"]`: if any, the compiled variant
is the mask-elided fast path, and its P0-1/P0-3 violations are P0-5's (references/specs.md). -/
def Input.elisions (i : Input) : List String := i.elision_flags.filter fun f => i.flags.contains (f, true)

end KV
