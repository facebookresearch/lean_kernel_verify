/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
KIR abstract syntax: one constructor per KIR node of `references/kir.md`, same names
and field order (table fields first, then `w`, then `addr`, then `loc`; the few exceptions
are listed at the top of references/kir.md).
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

/-- What a bulk asynchronous store reads and writes (P1 KIR, P1-2): a TMA tensor store or
reduce (`cp{.reduce}.async.bulk.tensor`, which also reads its tensormap), or a non-tensor
bulk store or reduce (`cp{.reduce}.async.bulk.global.shared`). -/
inductive BulkKind where
  | tensor | tensor_reduce | bulk | bulk_reduce
  deriving Repr, BEq, DecidableEq, Inhabited

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
  without a global source (a shared::cluster bulk copy or reduce, st.async or red.async into
  another CTA). -/
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
  /-- bar.sync or bar.red (`sync = true`) or bar.arrive of the whole role on named barrier
  `id` with `count` threads (P1 KIR, P0-6 KIR); `all` (P0-6 KIR only): the instruction names
  no thread count, a barrier for all threads of the CTA (`count`), which threads that exit
  release (PTX ISA `exit`). -/
  | named_bar (id count : Expr) (sync : Bool) (all : Bool) (loc : Loc)
  /-- A named barrier instruction P0-6 cannot count (P0-6 KIR): reached under a condition
  its participants may not share (`kind`; `certain`: the model admits participants that
  certainly diverge; `why`), with its id and count where constant. -/
  | nb_open (kind : String) (certain : Bool) (why : String) (id count : Option Expr)
      (sync : Bool) (loc : Loc)
  /-- The begin or end of a region of a P0-6 KIR that runs a number of times the model does
  not fix, the same in every warp of the role; `key` (or empty): the region of the role before
  warp splitting whose condition is uniform over that role, the same in its split roles. -/
  | nb_rep (isBegin : Bool) (why : String) (key : String) (loc : Loc)
  /-- A bulk asynchronous store (P1 KIR, P1-2) issued by one thread: lane `lane` of warp
  `warp` (`lane = none`: the elected lane of a full-warp elect.sync), reading `bytes` bytes
  of shared memory symbol `sym` from byte offset `src`, and with `tmap = some (base, addr)`
  the 128-byte tensormap at byte offset `addr` of global buffer `base`. It joins its
  thread's open bulk async-group. -/
  | bulk_store (kind : BulkKind) (warp : Expr) (lane : Option Expr) (sym : String) (src : Expr)
      (bytes : Nat) (tmap : Option (String × Expr)) (loc : Loc)
  /-- cp.async.bulk.commit_group of every thread of the role (P1-2). -/
  | bulk_commit (loc : Loc)
  /-- cp.async.bulk.wait_group{.read} `n` of every thread of the role (P1-2). -/
  | bulk_wait (n : Nat) (read : Bool) (loc : Loc)
  /-- A write of shared memory (P1-2) by one thread (lane `lane` of warp `warp`, `none`: the
  elected lane), or with `sync` by every lane of warp `warp` in one warp-synchronizing
  instruction (`.sync.aligned`): `bytes` bytes at each masked-in byte offset of `offsets` in
  symbol `sym`, in the shared memory of CTA `cta` of the cluster (`none`: the issuing CTA);
  `async`: the landing of an asynchronous copy issued here (it may land later); `slot`: the
  mbarrier (array, index, leader) that the copy completes its bytes on, in the issuing CTA or
  with `leader` in its pair leader's (the even rank of its pair, a cta_group::2 load), its load
  the role's next transaction event on that slot (only global accesses between). -/
  | smem_write (sym : String) (offsets : TExpr) (mask : Option TExpr) (bytes : Nat) (warp : Expr)
      (lane : Option Expr) (sync : Bool) (async : Bool) (cta : Option Expr)
      (slot : Option (String × Expr × Bool)) (loc : Loc)
  /-- A write of the 128-byte tensormap at byte offset `addr` of global buffer `base` by
  every lane of warp `warp` in one warp-synchronizing instruction (`tensormap.cp_fenceproxy`,
  P1-2). -/
  | tmap_write (base : String) (addr : Expr) (warp : Expr) (loc : Loc)
  /-- The role's value of handoff channel `chan` (a shared-memory word another role reads) is
  now `value` (w=32); each later `arrive` of the role records it for that arrive (KIR change
  17). -/
  | handoff_put (chan : String) (value : Expr) (loc : Loc)
  /-- `var` (w=32) := the value of handoff channel `chan` at the putter role's k-th `arrive`
  on slot s, where s is the slot of this role's last `wait` and that wait its k-th on s (KIR
  change 17). -/
  | handoff_get (chan : String) (var : String) (loc : Loc)
  /-- A store of `value` (the word's width) to element `index` of sync tensor `tensor` (KIR
  change 18). On a 64-bit word it carries a tag, its high half; `data`: its low half is data,
  not the integer `value`'s (a take of the tag binds nothing). -/
  | gsig (tensor : String) (index : Expr) (value : Expr) (data : Bool) (loc : Loc)
  /-- A take (KIR change 18): a blocking spin until the 64-bit sync word at `index` carries tag
  `tag` (w=32, its high half); `var` (with its width) binds the low half of the word's store of
  that tag (Eval.lean `Host.puts`: a take whose store is not known stops its role, blocked). -/
  | gwait (tensor : String) (index : Expr) (tag : Expr) (var : Option (String × Nat)) (loc : Loc)
  /-- A claim (KIR change 18): a fetch-add of `value` on a work cursor whose old value the role
  uses as a work index, bound to `var` of width `w`. The CTAs that claim on the word claim in
  turn, in linear order: the k-th claim of the j-th of n claiming CTAs reads `(k x n + j) x
  value` (Eval.lean `claimsFeasible` checks that the values form the chain a fetch-add hands
  out). -/
  | gclaim (tensor : String) (index value : Expr) (var : String) (w : Nat) (loc : Loc)
  /-- A shared-memory relay (KIR change 18): the role stores `value` into field `field` of slot
  `ord` of channel `chan` (a ring of slots in one shared array; each slot of a CTA is written
  once). -/
  | rput (chan : String) (ord : Expr) (field : Nat) (value : Expr) (loc : Loc)
  /-- The role reads field `field` of slot `ord` of channel `chan`, which an earlier role of the
  CTA (or this role, earlier) wrote, into variable `var` of width `w` (trusted: the ring's
  handshake delivers slot `ord`'s write to this read). -/
  | rget (chan : String) (ord : Expr) (field : Nat) (var : String) (w : Nat) (loc : Loc)
  deriving Repr, Inhabited

inductive Stmt where
  | «let» (var : String) (expr : Expr) (loc : Loc)
  | «for» (var : String) (lb ub step : Expr) (body : List Stmt) (loc : Loc)
  | «while» (cond : Expr) (body : List Stmt) (max_iters : Nat) (loc : Loc)
  | «if» (cond : Expr) («then» «else» : List Stmt) (loc : Loc)
  | event (ev : Event)
  | assert (cond : Expr) (msg : String) (loc : Loc)
  /-- A model assumption the frontend could not prove statically (KIR change 16): false on
  an input, the role stops there as at an out-of-bounds metadata load (Eval.lean). -/
  | assume (cond : Expr) (msg : String) (loc : Loc)
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
  /-- A sync tensor's word width in bits (32 or 64; KIR change 18): every element is a sync
  word, accessed only by sync events. `none`: a data tensor. -/
  sync : Option Nat := none
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
  /-- KIR change 17: the role this one is a part of (the P1 pass splits a role by warps), or
  empty. P0 reads the threads of a role as running in lockstep (references/specs.md, the
  ledger); the handoff checks read the parts of one role so. -/
  group : String := ""
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
  | ttgir | «cute-mlir» | ptx
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
  /-- The P1 specs a P1 KIR is decided for, by label (KIR `p1`; `[]`: P1-1 iff `tmem`). -/
  p1 : List String := []
  /-- P1-2 KIRs: the CTA's thread count, idle warps included (KIR change 15). -/
  cta_threads : Option Nat := none
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
  /-- Co-resident CTAs of the launch and, for a cluster launch, co-resident clusters (KIR
  change 18, the occupancy model of the dump); `none`: unknown. -/
  capacity : Option (Nat × Option Nat) := none
  /-- Groups of CTAs (linear indices) that synchronize outside the model, e.g. through a
  protocol region (KIR change 18): each reaches the others (`claimTurnsFeasible`). -/
  syncGroups : List (List Nat) := []
  deriving Repr

/-- The elision flags this input sets, e.g. `["FULL_TILES"]`: if any, the compiled variant
is the mask-elided fast path, and its P0-1/P0-3 violations are P0-5's (references/specs.md). -/
def Input.elisions (i : Input) : List String := i.elision_flags.filter fun f => i.flags.contains (f, true)

end KV
