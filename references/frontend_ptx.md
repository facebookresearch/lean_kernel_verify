# PTX translation (Layer 1, both families; the TLX/Triton default)

The PTX translation turns the kernel's PTX, the code its compiler hands to ptxas, into KIR
(`kir.md`) instead of the compiler IR, with one core shared by both frontends and no
hand-written model of the kernel. It is the default for TLX/Triton kernels; `--frontend
legacy` translates their TTGIR instead (`frontend_ttgir.md`). CuTe DSL kernels keep the
CuTe DSL MLIR translation (`frontend_cute.md`) by default, and `--frontend ptx` selects
this one for them. It still refuses some kernels and constructs the legacy translations
handle and has known gaps ("Refusals", "Limitations"). `run` passes the choice to `to_kir`
as `LKV_FRONTEND` (`legacy` or `ptx`; unset or empty is the family's default, `ptx` for
TLX/Triton and `legacy` for CuTe; any other value an error), which also selects it when
set in the environment of `run` (the default of `--frontend`) or of a frontend's `to_kir`
run by hand.

Everything around the translation stays: the case files, the dump (it records the PTX and
the launch records below whichever translation runs), the variant names,
`kir/report.json`, the family's checks and the report. The KIR has `ir` `"ptx"`, the
frontend's log shows `[to_kir] LKV_FRONTEND=ptx: translating the PTX` (TLX:
`logs/to_kir.log`, CuTe: `logs/frontend.log`), and report.json `translation` (report.md:
the `translation` line at the top) names the translation of the run. The PTX translated
is the one the dump writes: TLX `<spec>.ptx` next to each `<spec>.ttgir`, CuTe
`ptx/NN_<digest>.ptx` per variant.

The host adapter of each family (its dump) runs the real host code on a CPU and records
the PTX of every compiled kernel and the facts of every launch (TLX/Triton:
`frontend_ttgir.md`; CuTe DSL: `frontend_cute.md`, which also give the case formats).
The core (`frontend/ptx/`, pure Python stdlib) turns one kernel's PTX and launches into
Facts: roles with their KIR bodies, barriers, the tensors touched and written, the scalar
params read, the TMA descriptors. The family's `to_kir` assembles them into KIR with its
own names, inputs and checks. The op table is `ir_ops_ptx.md`. Tests: `tests/ptx/`
(stdlib, also the GPU tests, which skip without a GPU).

```
case file + exported sources
  | dump (host adapter): host code on CPU, launches intercepted, the real compiler down to PTX
  v
dump.json: per compiled kernel its PTX; per launch a LaunchRecord (grid, block, cluster, parameters)
  | frontend/ptx/translate.py
  |   parse.py     PTX text -> functions, instructions, .loc chains, inline-asm scopes
  |   cfg.py       basic blocks, the shape check, roles (the Triton warp-specialize dispatch)
  |   regions.py   region tree: ifs, loops, the blocking mbarrier wait
  |   interp.py    thread model, slicing, partition refinement: KIR statements per role
  |   ops.py       op table; event rules in mem.py, tma.py, sync.py, cluster.py, tmem.py (P1)
  |   anchors.py   the case's work anchors
  v
Facts (or the refusal that stopped the variant)
  | frontend/ttgir/to_kir.py build_ptx | frontend/cute/to_kir.py PtxVariant.assemble
  v
KIR, `ir` "ptx", written only if frontend/kir.py accepts it; CuTe: a P1 KIR from a second pass
```

## Launch records

Both dumps add to dump.json, per launch, a LaunchRecord (`frontend/ptx/launch.py`): the
PTX entry, grid, block, cluster, dynamic shared memory, launch attributes and every kernel
parameter (kind `ptr`, `scalar`, `tensormap` or `aggregate`, size, alignment, name,
value; struct fields; a decoded CUtensorMap and the tensor its address lies in). CuTe
records also name the kernel, its PTX file and the target, which the TLX dump keeps with
the specialization. The parameter list is checked against the PTX `.entry` (count, kind,
size, alignment of every parameter); any difference fails the dump. The launch attributes
are `pdl`, `cooperative` and `preferred_cluster`; a TLX record also has
`func_attributes` when the host set function attributes on the launched kernel before
the launch: `{"required_cluster": [x, y, z]}` (frontend_ttgir.md, dump.json and patch
13). A launch whose `required_cluster` differs from its `cluster` is refused (a gap:
which of the two the hardware uses is not modeled, so the geometry LKV would check may
not be the one that runs; `to_kir.check_required_cluster`, both TTGIR translations); an
absent attribute is the launch's cluster.

Parameters are named by position, never by guessing from addresses: TLX by the jit
argument names in Triton's own parameter packing (a host descriptor expands to its
tensormap and its shape and stride arguments; two scratch pointers come last); CuTe by the
trace (`jit_params`): a parameter that passes a jit argument through names that tensor,
also when it is empty (data pointer 0) or passed under two names, and a computed one (a
descriptor, a view, a struct of pointers) names the tensor its address resolves to among
the arguments it is computed from. A pointer no tensor can be named for exactly is
unnamed: a load through it is Data.

CuTe host code encodes its TMA descriptors in software; the decoder reads the tiled
CUtensorMap layout (checked against the driver's `cuTensorMapEncodeTiled`,
`tests/ptx/test_gpu_tmap.py`) and refuses any unknown bit. CuTe DSL 4.4.2's encoder
writes 2 where the driver writes 0 in `w1[0:4]` (no KIR meaning; the test checks the two
encodings otherwise agree); DSL 4.6.1's writes only `w0`-`w7`, so the adapter zeroes
`w8`-`w15` and records them as `unwritten_words`. Element strides other than 1,
interleave and im2col are refused. TLX descriptors are not decoded: the record carries
what Triton's launcher would encode (base, shape, strides) and the box, swizzle and
element type of the kernel's `tensordesc_meta`.

## Locations

KIR `loc` is `path:line` of the innermost `.loc` frame (along the `inlined_at` chain) in
the exported sources; an instruction without a `.loc`, or with line 0, keeps the previous
one.

* TLX: `fbcode/<path below src_root>:line` (the dump keeps the PTX's source paths with
  `TRITON_PRESERVE_PTX_SOURCE_PATHS=1`); an event without such a line is an error (a
  compiler switch rewrote the locations).
* CuTe DSL 4.6.1: `<path below the source root>:line`; the instructions of an inline asm
  block take the location where the block starts (the DSL gives them synthetic ones).
* CuTe DSL 4.4.2: its PTX names the wrong source files (one `.file`, the kernel's, for
  lines of DSL library modules too), so only line numbers are used, read in the kernel's
  own source file; a `.loc` on a blank or comment position, or past the line's end, is
  skipped. A line can still belong to another file, so these locations are **best
  effort**: the variant's `kir/report.json` entry has `"loc_quality": "line_only"`,
  report.json `loc_quality` lists it, report.md's environment section says so, and a
  `where` work anchor matches by line.

P1 `tmem` events of an MMA carry `site`: the closest preceding line in another file than
the DSL helper the MMA is located in (a heuristic, for the report only).

## Control flow

`cfg.py` splits the PTX into basic blocks and admits only the shapes real kernels have:
reducible flow, every exit edge of a natural loop leaving to one block, exactly one
`ret`/`exit`. A `brx.idx` is the Triton dispatch (below; its index is the state byte, an
8-bit `ld.shared`) or a switch (LLVM's jump table of a select chain or `switch`), which
`cfg.lower_switches` rewrites into the compare chain `@(i == k) bra T_k` (the last target
when no case matched); the interpreter requires the index to be provably in range (its
interval; out of range is undefined in PTX). Anything else (a guarded `ret`, a loop with
exits to two blocks) is refused naming the lines. An unguarded `trap`
ends its block, which goes on at the other successor of its one predecessor, a
conditional branch, as if the trap were not reached: Triton places a
`tl.device_assert`'s failing path after the `ret` (`call __assertfail; trap;`, a KIR
`assert false` under the failing threads' mask), CuTe writes `if x: trap()`. A trap
reached otherwise is refused.

`regions.py` builds a tree per role: an if runs from its branch to the immediate
post-dominator; a loop has a header, one iteration as its body and one exit block (several
exit edges to it join through an exit flag); a block reached on two paths that meet only
later is emitted on each path. `interp.py` translates it:

* An if with a condition uniform over the role is a KIR `if`; registers that differ at the
  join become `select`s (or variables assigned in both arms). A thread-varying condition
  is a mask: both arms are translated in place, events carry the mask, registers merge by
  `select`, and a divisor or metadata-load index under the mask is replaced where the mask
  is false, so a path not taken never stops the evaluation. A predicated instruction is a
  one-instruction masked region. A branch whose arms only set registers controls no event,
  so its condition may be data (e.g. `if tmem_addr != 0: trap()`).
* A loop is a KIR `for` only if it is an exact signed i32 counted loop with constant
  bounds; any other loop is a `while` on its exit flag with a budget, `max_iters` (TLX
  65536, CuTe the case's `max_iters`, default 4096). Exceeding the budget is an
  evaluation error (ERROR), never a pass. The registers a loop carries are variables
  assigned in parallel on every back and exit edge, their kinds widened to a fixed point.
  The specs read traces, not loop shapes: a `for` is only cheaper to evaluate.
* After a loop, a register that every back and exit edge leaves at one loop-invariant
  value is that value (LLVM rematerializes `mov r, 0` or `mov r, %tid.x` in a loop body
  and reads it after the loop).
* The carry flag of `add.cc`/`addc` and `sub.cc`/`subc` lives within a basic block
  (`ir_ops_ptx.md`, "Values").
* The blocking wait loop around `mbarrier.try_wait.parity` is one `wait`
  (`ir_ops_ptx.md`, "mbarriers").

## Threads and roles

Within a role, a thread-varying value is an integer expression over LANE, WARP and the
elect symbols; a value is uniform if substituting every thread of the role gives one
result (exact BitVec folding plus exact intervals). Shuffles, ballots and warp
reductions substitute lanes, and a value computed from per-lane loads and shuffles is a table of 32 per-lane
values, so warp-cooperative code (a varlen tile scheduler's per-lane `cu_seqlens` loads,
prefix sums and ballots) is exact. The block must be (32k, 1, 1), and every launch of a
variant must have the same block and cluster.

* **TLX/Triton.** A kernel without `brx.idx` is one role, `default`. A warp-specialized
  kernel's dispatch must match Triton's protocol item by item, on the CFG and reaching
  definitions (not on adjacent instructions): `wid = shfl.sync.idx(%tid.x >> 5, 0)` and a
  branch on `wid < D` of either polarity; the worker switch loop (optional `setmaxnreg`,
  `barrier.sync 1`, `ld.shared.b8 s, [smem + wid + K]`, then the out-of-range bypass and
  `brx.idx s`, or, with at most 3 worker states, the chain of `s == v` branches LLVM
  lowers the switch to); the default role's immediate stores (a vector of immediates
  too) of every worker's state before its first fork barrier and of the exit state
  before its last; a fork-barrier count that no path changes. Each branch condition must
  be the only result of its `setp`. A generic write of shared memory (through
  `cvta.shared`) or an `atom`/`red.shared` in such a kernel is refused: the state bytes
  are checked against `st.shared` only. Each worker runs the switch loop twice, into its
  state and into exit, so state k's role `partition<k>` is the code before the dispatch,
  one pass of the loop header and the region from target k back to the header, then
  `ret`; `default` is warps 0..D-1. A state whose region only synchronizes (padding)
  has a role only if its translation emits an event (a cluster kernel's padding warps
  arrive on the cluster barrier, "Clusters, tcgen05, CLC"). A failing item refuses the
  kernel and names the item. A branch on the warp inside a role is a mask.
  The direct dispatch of Triton's single-warp-specialize lowering
  (`tlx.async_tasks(exclusive=True)`: one warp_specialize in the module, no state bytes,
  no switch loop) is matched too: the entry as above; the worker side's first block with
  a fork barrier (`barrier.sync 1`, optional `setmaxnreg`, pure instructions) ends a
  chain of branches whose conditions compare wid with an immediate (any comparison and
  type; each later link pure, with the previous one as its only predecessor); each
  worker warp follows the comparisons, evaluated at its warp id, to its partition's
  first block. Every partition region has one fork barrier on every path to `ret`, the
  default two, and the regions share no block with the default, the dispatch or the code
  before it but the `ret` block. Roles: `default`, then `partition<k>` numbered by first
  warp, the partitions that do more than synchronize first, padding last (Triton gives
  larger partitions the lower warps, so k is the TTGIR partition index where the warp
  counts do not grow in TTGIR order).
  Captures: Triton passes the values a partition captures through shared memory: the
  default stores them before its first fork barrier, each worker loads them between its
  first and second one (both protocols). A worker's `ld.shared` there of bytes the
  default stored gets the stored value (byte by byte, so a capture split into bytes is
  put back together) if the default's store runs unconditionally (statically known
  issuing threads, no runtime condition) and its value is one value in every thread that
  depends on nothing but the launch (parameters, program id, shared symbols, metadata);
  otherwise the load is Data as any shared load. Every other write of shared memory
  reports the bytes it may touch, over the role's executing threads (the bounds of its
  offsets plus its size): `st.shared` (an unknown address, a condition, a value of a
  role, a worker's, the default's after its first fork barrier), `stmatrix` (a 16-byte
  row per address-giving lane), `atom`/`red.shared`, generic `st`/`atom`/`red` through
  `cvta.shared` (a direct dispatch does not refuse them) or a data address (any byte),
  the landings of TMA loads (the
  hardware box; any byte through a packed sub-byte descriptor, whose shared-memory copy
  may be wider than its packed bytes), bulk copies and `cp.async`, a CLC response (16 bytes),
  `st.async`/`red.async`, `tcgen05.alloc`'s address slot, `tensormap.replace` of a
  staging buffer and `mbarrier.inval` (the op-table rows that write shared memory are
  listed in interp.CAPTURE_WRITERS; tests/ptx/test_captures.py checks the table against
  them). The bounds come from interval arithmetic, including masks and the ring-buffer
  index `x % k` as LLVM writes it for any x (`x - ((x + b) & -k)`, `x - q k` with q a
  multiply-high quotient): bounds on the arithmetic, no claim about the quotient. Such a
  write before the workers' loads makes the captured bytes it may touch unknown (all of
  them if its address is no shared symbol plus an offset); roles are translated one
  after the other, the default first, so such a write of a byte another role already
  read refuses the kernel. A write after its role's second fork barrier follows every
  worker's loads, unless it lands in another CTA's shared memory (`st.async`,
  `red.async`, a shared::cluster bulk copy, a multicast TMA, bulk copy or CLC
  response): no barrier of the CTA orders that one, so it counts wherever it is issued.
  An asynchronous write (a landing, a CLC response) may land after a later store: the
  default records no capture over bytes an earlier asynchronous or remote write may
  touch. A write whose offsets have no static bounds (an offset LLVM carries
  strength-reduced through a loop, unrelated to the counter it was derived from) keeps
  the captures under KIR `assume`s (kir.md change 16; `a shared write misses the
  warp-specialize captures [lo, hi)`, per executing thread) that its bytes miss each
  run of known captures, evaluated on every input. Such a write over a capture is no
  certain violation (it may land after the workers read it), so where the assumption
  does not hold the role stops there and the input is undecided (P0-2..P0-6, P1), never
  a FAIL; the report names the assumption. An mbarrier object changes at all its events but waits (some
  asynchronous, some from another CTA, at an index that may be loop-carried): after
  every role, each mbarrier array's bytes (a PASS needs every index in range) must miss
  every capture a worker read, else the kernel is refused.
* **CuTe DSL.** Partition refinement: a role starts as every warp of the block; a branch
  condition, mbarrier address, remote CTA rank, named-barrier id or count, or a select
  between two tensors that is uniform within each warp but differs between warps of the
  role splits it into warp ranges, each translated again. Roles are `warps<a>_<b>`; code
  before the dispatch belongs to every role that runs it.

## Values and slicing

PTX is not SSA: registers are mutable slots, a `mov` on a loop latch is the loop-carried
copy. A register holds an exact integer expression or Data: a value the model drops
(floating point, shared- and tensor-memory contents, data loads, an address transformed
by bit operations). Nothing is emitted for a value until a root needs it: an event
operand, a branch condition, an address, a predicate, a barrier field, a work anchor.
Then each thread-uniform subexpression is bound by a `let` at its instruction's location.
Data that reaches a root refuses the variant (`data value from <origin> reaches <use>
(slice soundness)`): the frontend never guesses a dropped value. A node that can stop
the evaluation (a metadata load, a division) is never bound ahead of the `select` arm
that guards it.

`addr` (P0-1's wrap check): after translation every add, sub, mul, shl and trunc whose
value reaches a global offset, a TMA coordinate or a metadata-load index through any
chain of `let`s (loop-carried induction variables included) is marked. Not marked: the
nodes of a `mul.hi` expansion (magic-number division) and of a sign extension or signed
bit field. The other expansions are built so that no marked node wraps where the
instruction does not: bits that `brev`, `bfi`, `prmt`, `shf`, `vote.ballot`,
`%lanemask_*` or a `mov` packing registers move into the sign bit come from shifts of
sign-extended fields (or logical shifts), and 64-bit `mul.hi`, `.sat`, `sad` and the
carry flag are computed where nothing wraps. In `(x + c) << k` LLVM keeps only the low
bits of `c` that survive the shift (`(n - 2) << 7` may become `(n + 0x1FFFFFE) << 7`),
so the intermediate wrap is judged on the constant closest to 0 with those low bits: the
rewrite reports no wrap the source does not have (and a hand-written such address with a
huge `c` is not reported either).

## Family conventions

The core takes a policy per family, so that each family's KIR keeps its names and output
contract:

| | TLX/Triton | CuTe DSL |
|---|---|---|
| roles | `default`, `partition<k>` | `warps<a>_<b>` |
| a global access by several threads | one event whose offsets range over the issuing threads (closed forms over the thread index) | one event per issuing thread, in a KIR `for` over the role's threads |
| `threads_per_issue` | 1; an arrive's `n` is the issuing threads times its count | the gcd of the role's arrive counts |
| scalar params | the jit argument names | `param<i>`, struct fields `param<i>_<off>` |
| tensor shape params (i64) | `T.shape.i`, `T.stride.i` | `T_shape<i>`, `T_stride<i>` |
| barrier arrays | one per init run (a TLX `alloc_barriers`) | init runs split by the slots their dynamic uses take |
| barrier names | the assignment target of the init's source line (`k_fulls`), else `mbar_<offset>` | `mbar_<offset>` |
| TMA | `tma_load` / `tma_store` / `tma_reduce` on the host descriptor | `tma_load` plus its footprint `gload` on the base tensor; stores and reduces as footprint `gstore` / `gatomic` on the base |
| descriptor dims | outermost first | innermost first |
| locations | `fbcode/<path>:line` | `<path>:line` (DSL 4.4.2: best effort) |
| `while` budget | 65536 | the case's `max_iters` |
| work events | gated (below) | at every anchor |
| also | CLC | clusters, the P1 pass |

## Memory

Addresses are expressions over reserved bases: a kernel tensor's address (its pointer
parameter, or the pointer field of a struct parameter), a shared symbol's and a local
variable's (LLVM's `__local_depot`), and the generic addresses of the same shared and
local bytes, which only `cvta` makes (so a generic address is never taken for a window
one). A global access (`ld`/`st.global`, generic `ld`/`st`/`atom`/`red` on a global
address, `atom`/`red.global`, `red.async`, `cp.async`) peels its address into (tensor,
byte offset) through adds, subtracts and selects; a select between two tensors takes the arm the role's condition picks (CuTe: a
condition uniform per warp splits the role), and one whose condition is one value in the
role but not known (a program id or parameter test) becomes one access per tensor under
the condition and its negation (a generic access through a select between a shared and a
global address: refused); the bases may come from `ld.param` through a select of
two parameter addresses. The element offset is the byte offset over
the element size, provably exact, else the access is refused. An integer global load of
the register's width whose value reaches a root is a metadata read,
KIR `load(T, [index])`, and T is a metadata tensor whose recorded contents the input
carries; not a strong load (`.volatile`, `.relaxed`, `.acquire`, also `.cv`), whose
location another thread may write while the kernel runs: its value is Data. The family
assembly checks that the contents are those the kernel reads (`frontend_ttgir.md`,
`frontend_cute.md`). A generic access of shared memory is data, as `ld`/`st.shared` and
`atom`/`red.shared` (an explicitly global one there is an error); local memory is the
thread's own: no event, a store kept inside its variable by a KIR assert, a load exact
only where the same thread stored those bytes at a constant offset earlier in the basic
block (`ir_ops_ptx.md`, "Global memory").

**mbarriers.** An mbarrier address is a shared symbol plus a byte offset, the same for
every thread of the role. Arrays are runs of adjacent 8-byte slots initialized with one
count, in one guarding region, at one `.loc` chain, by one role. TLX keeps each run whole
(one `alloc_barriers` is one array); CuTe splits a run by the slots each dynamic use may
take (a stage index `i = i + 1 == S ? 0 : i + 1` gets the range its assignments allow),
except slots that only constant offsets name and the same roles use alike (the unrolled
stages of one pipeline stay one array, as P0-2's static-waiter rule expects). A use's
array is the one holding the constant part of its offset (a loop-carried address keeps
that constant outside the carried slot), its index `(offset - start) / 8`, exact by
structure or known trailing zeros; if either is unknown the use is refused (an if-chain
over candidate arrays would make unreachable waits static waiters). Two forms LLVM makes
of a static choice are resolved when the offset's own constant is no mbarrier:
* an induction variable: a uniform integer loop slot to which every edge of its loop
  adds a constant (strength reduction of `start + 8 * i` into a counter that starts at
  `start + 8 * i0` and steps by 8; checked at every edge, else the loop is translated
  again without it) keeps the constant part of its entry value as the array's start: the
  array is the one holding the offset's constant plus those entry constants, the index
  the rest minus them over 8, exact as above. A step that is not a constant, or a slot
  that is not a plain integer, is refused as before;
* a select on a thread-uniform condition in the offset's top-level sum whose two arms
  each hold an mbarrier (`if c: wait(A[i]) else: wait(B[j])` if-converted into one wait
  on a selected address): the event becomes `if c { event on A } else { event on B }`,
  the TTGIR's own branch; any other select is refused.

Counts must be host values. A private synchronous cycle that `mbarrier.inval` ends (TLX's `tlx.async_dot`
without mBarriers) is removed from the KIR; any other inval is refused (`ir_ops_ptx.md`,
"mbarriers").

## TMA descriptors

* **TLX host descriptor**: the jit argument; its shape and strides are the kernel's own
  params `<d>.shape.i` (i32) and `<d>.stride.i` (i64), valued from the host descriptor;
  its box is the descriptor's block, the same in every launch. Triton cuts a block into
  hardware boxes, one copy each; the copies of one block (the next ones of the first's
  opcode and descriptor, at the block's other boxes, under the same mask and mbarrier) are
  one event at the first's coordinates.
* **CuTe host descriptor**: `tma<k>_<base>` (k: its index among the kernel's tensormap
  parameters; base: the tensor its address lies in), with synthetic params
  `<d>.shape.i`, `<d>.stride.i` valued per launch from the decoded CUtensorMap in the
  base's elements, and `<d>.off` when the address is offset from the base's data pointer
  (a ragged TMA tensor shifted back by 2^30 rows). The box must be the same in every
  launch. It inherits the base's segments on the dim that walks the segmented tensor dim
  in every launch (same extent and byte stride), else none. A zero-length tensor's
  descriptor (address 0) has extent 0 and no footprint; every TMA instruction through it
  is a KIR `assert false` (the hardware reads its extents as 2^32 at address 0).
* **In-kernel descriptor** (`tl.make_tensor_descriptor`): a 128-byte shared staging
  buffer zeroed by a full warp (`st.shared.b32 0`, also of a register known to be 0),
  filled by `tensormap.replace` and copied to global memory by
  `tensormap.cp_fenceproxy`. Each cp_fenceproxy takes the fields written since its
  buffer's last zeroing (one buffer may serve two descriptors in a tile); a rank never
  replaced keeps the zeroed 0, rank 1 (Triton writes none for a 1-D descriptor). Any
  other write of shared memory by the role that may overlap an open buffer closes it (a
  later replace or copy of it is refused): a `st.shared` or an `atom`/`red.shared` (also
  under a data predicate), a generic `st`/`atom`/`red` on those bytes, and `stmatrix`,
  `cp.async`, TMA and bulk copies into shared memory, mbarrier and
  `tcgen05.alloc`/`commit` and CLC writes whatever their bytes. The zeroing, the
  replaces and the copy must be in one statement list; a `tensormap.replace` of a
  descriptor in global memory is refused. A TMA instruction uses the descriptor a
  cp_fenceproxy of its role wrote at the same address earlier in the same loop
  iteration, on an enclosing path, or in that iteration before loops it entered since
  when none of those loops holds a `tensormap.cp_fenceproxy` (a work item writes its
  descriptors, its tile loop uses them); anything else is refused. Stores and reduces
  through it are `gstore` / `gatomic` of its base pointer with exactly the TMA footprint
  (extents read unsigned: the tensormap's `global_dim` is a b32), so `intended_writes`
  names the base pointer. A load through it is the `gload` of that footprint on the base
  plus, for its transaction bytes, a `tma_load` on its mbarrier of the box descriptor
  `box_<eb>B_<d0>x<d1>...` at coordinates 0: a KIR desc whose extents are the box
  (constants, role input), so its bytes are the box times the element bytes, as the
  hardware's (out-of-bounds fill included). Assumed, not checked: no other role
  rewrites the descriptor's slot, and its role does not rewrite it while a load through
  it is in flight (P1-2 checks that for stores).
* **Bulk copies** (non-tensor `cp.async.bulk` global -> shared): the 1-D descriptor
  `bulk_<T>_<n>` over T's storage (box n elements), for the transaction bytes, plus the
  footprint `gload`.

Coordinates, footprints, transaction bytes, issuing threads, sub-byte formats:
`ir_ops_ptx.md`, "TMA and bulk copies".

## Clusters, tcgen05, CLC

* **Clusters (CuTe and TLX).** The cluster dims come from the LaunchRecords and are the
  KIR `cluster` (`kir.md` change 11); remote arrives (`cta`), multicast TMA loads and
  commits (`cta_mask`), cta_group::2 loads completing on the pair leader's mbarrier
  (`leader`: CuTe's peer-bit mask, TLX's `mapa` of `rank & 254`), `st.async` /
  `red.async` into another CTA (`complete_tx` per executing thread), shared::cluster bulk
  copies and reduces (`complete_tx`) and the cluster barrier follow `ir_ops_ptx.md`, "Thread barriers
  and clusters". P0-2 and P0-4 decide one barrier ledger per cluster. Which CTA's data a
  copy carries is not checked. TLX 2-CTA kernels: the cluster barrier is reached by every
  role, the worker regions arrive on it at their end (the padding warps too, which
  therefore keep a role, "Threads and roles"); `fence.mbarrier_init.release.cluster`
  emits nothing.
* **Preferred cluster geometry.** A launch with a preferred cluster (Triton's
  `preferred_ctas_per_cga`, the launch attribute
  `cudaLaunchAttributePreferredClusterDimension`; LaunchRecord `attrs.preferred_cluster`,
  a multiple of the regular dims, `cluster`) may get clusters of either size: the device
  groups CTAs into preferred clusters where it can and into regular ones elsewhere, a
  preferred cluster replacing the regular clusters of an aligned group of CTAs, and each
  kind numbered as if the grid held only its kind (CUDA runtime API, the attribute's
  documentation). So every CTA of such a launch runs as in one of two geometries: all
  clusters regular (the variant's KIR, `<spec>`) or all preferred (a second KIR,
  `<spec>@cluster<dims>`, with the inputs whose grid the preferred dims divide), and the
  report names the geometry of each verdict. The P0-6 barrier pass writes both
  geometries too (`<spec>.nb.json`, `<spec>@cluster<dims>.nb.json`, whatever the
  translation): the rank and the cluster special registers may decide which warps reach a
  barrier (a `rank == 0` branch), and they differ by geometry. A mixed launch's clusters
  are each a cluster of one of them, with the same CTAs, ranks and special registers, so
  P0-1, P0-2, P0-4, P0-5 and P0-6 (per access, per cluster ledger, per CTA) hold for every
  mix if they hold in both;
  P0-3 counts writes over the whole grid and holds for a mix only if every aligned group
  of CTAs writes the same elements in both geometries, which is not checked (Limits).
  P1-2 has a KIR per geometry too (`<spec>@cluster<dims>.p1.json`); it decides each
  cluster on its own except a tensormap that one cluster's stores read and another
  cluster writes (F2 across clusters), so it holds for every mix if it holds in both,
  except that cross-cluster check, which pairs only clusters of one geometry (Limits).
  CLC with a preferred geometry is refused. If the launches disagree on the preferred
  dims, the all-preferred geometry is the refused variant `<spec>@cluster` (a gap for
  every input of the specialization) and the regular KIR is still checked. The P0-6 and
  P1-2 KIRs are made from the regular geometry only. A CuTe launch with a preferred
  cluster other than its cluster is refused (gap: the CuTe translations check one
  geometry; `cute/to_kir.preferred_cluster_gap`).
* **tcgen05 in P0.** No event except `tcgen05.commit`, an asynchronous `commit_arrive` on
  its mbarrier; the P0 specs do not see tensor memory.
* **CLC (TLX).** `try_cancel` and its queries follow the schedule model of
  `frontend_ttgir.md`, "Cluster Launch Control": every role becomes
  `let __clc_k = 0; if __clc_launched[pid.x] != 0 { ... }` and its k-th query reads
  `__clc_schedule[pid.x, k]`. With clusters a try_cancel must multicast its response
  (`.multicast::cluster::all`: a `tma_load` with `cta_mask` over the cluster) and the
  schedule steals whole clusters. CLC in a CuTe kernel is refused.

## Work anchors

A case's `work` names, per work field, where its value comes from
(`frontend/ptx/anchors.py`):

* **Assignment anchor** (TLX: `{"batch": "off_z"}`): each assignment of the variable in
  the kernel function's Python source is an occurrence; its instructions are those whose
  kernel-file `.loc` frame lies in the assignment's span, and the variable is the one
  result that leaves that span (a register read outside it). An element of a tuple target
  narrows to the instructions computing that element. No result, or several, is a case
  error naming the candidates. A field the compiler folded away (its assignment runs, no
  register holds its value) is defined from there on with no value: -1, refused in a
  role that accesses a tensor whose segments `per` names it (only those accesses read
  it).
* **Load anchor** (`[{"load": "T", "where": "text"}]`; CuTe, and TLX in this translation
  only): `where` matches a source line as a substring of the line or of
  `<relpath>:<line>`; every global load of T located on a matching line is an anchor (DSL
  4.4.2: by line number only), and the field becomes the load's element index. A `where`
  that matches no line is a case error.

Gating: TLX emits a `work` event in each region that defines every field, after the last
definition; a role that never defines some field gets none, and no region defining all of
them is a case error. A loop body starts with the fields the enclosing region defined
that have no anchor inside the loop (LLVM hoists a loop-invariant field, e.g. a TLX
megakernel's `off_h = cluster_cta_rank % H`, out of the work loop TTGIR defines it in;
every iteration sees that value); a field with an anchor in the loop starts undefined
there, as TTGIR's blocks. CuTe emits one at every anchor; fields are role variables, -1 until
set, and under a lane condition that depends on runtime values only the part of the
condition that is uniform per warp sets them (the work item is the role's).

## P1 pass (CuTe)

The CuTe frontend translates every variant's PTX a second time with the P1 rules
(`frontend/ptx/tmem.py`, `ir_ops_ptx.md`, "tcgen05") into the P1 KIR
`<Kernel>__<digest>.p1.json`: the P0 events plus `tmem`, `tmem_wait` and `named_bar`
events and the top-level `tmem` (`kir.md` change 10). A role whose warps issue an
mbarrier wait, arrive or commit only in part (a thread-0 arrive or wait) is split into
runs of issuing and non-issuing warps, because P1-1 takes a role's event as every
warp's; the P0 KIR keeps its roles and does not change. TMEM addresses are absolute (the
DSL's fake-zero-base idiom: constants and integer arithmetic on them), `tmem.base`
`zero` with a 512-column allocation. A kernel without tcgen05 instructions is
`not_applicable`. A refusal leaves the variant's P1-1 "not checked" (`kir/report.json`
`p1`: `refused`, or `error` for an internal failure); P0 is unaffected. The tags are the
legacy CuTe P1 pass's where the reason is the same (`frontend_cute.md`); R10 is new:

| # | Refused |
|---|---|
| R1 | a TMEM address that is data (e.g. read back from shared memory, where `tcgen05.alloc` writes it: an address relative to the allocation), or a memory address |
| R2 | not exactly one allocation, one in a loop or under a runtime lane condition, or with a non-constant or unsupported column count |
| R3 | absolute addresses with an allocation under 512 columns |
| R5 | shapes and kinds outside `ir_ops_ptx.md`, an unknown M or N, a block-scaled N over 256, cta_group::2 without a cluster |
| R6 | an ld/st with part of a warp active, a lane-varying address or under a runtime lane condition; a cp/mma in a role of several warps, issued by several threads or with a lane-varying operand |
| R7 | a named barrier or `tcgen05.wait` reached by part of the role, a named-barrier id or count that is not constant (one uniform per warp splits the role), a count that is not a positive multiple of 32; an mbarrier wait, arrive or commit under a runtime lane condition in a role of several warps |
| R10 | a role whose cp/mma and commits are not all issued alike (an elected lane, or one fixed thread): P1-1 takes a commit to cover the same thread's earlier cp/mma |

TLX/Triton kernels have no P1-1 pass: report.json lists their variants' P1-1 as
`not_checked`.

## P1-2 pass (TLX/Triton)

In the PTX translation (LKV_FRONTEND=ptx, the TLX/Triton default) the TTGIR frontend
translates every specialization's PTX a second time with policy `p1` `["P1-2"]`
(`frontend/ptx/bulk.py`, `ir_ops_ptx.md`, "P1-2") into the P1 KIR `<spec>.p1.json`: the P0
events plus the P1-2 events and `named_bar` (`kir.md` change 15: top-level
`p1: ["P1-2"]` and `cta_threads`, the block's threads); with a preferred cluster
geometry also `<spec>@cluster<dims>.p1.json`, all clusters preferred ("Clusters, tcgen05,
CLC").
A specialization without a bulk store is `not_applicable`; a refusal or failure is
logged as `P1 refused|error: <why>` and leaves its P1-2 "not checked"; the P0 KIR and
its verdicts never change (`LKV_P1=0` skips the pass). The roles are split as the CuTe
pass splits them (a role whose warps issue an mbarrier wait, arrive or commit only in
part; a TTGIR partition's parts are named `<partition>_w<first>_<last>`), since the P1
ledger takes a role's mbarrier event as every warp's. The `let`s nothing reads are
dropped from the P1 KIR: a part of one warp binds per-lane values of its lane-only loads
(the thread model's tables) that only data reads, and those loads would make their
tensors metadata, whose contents the dump records only where P0 needs them; the P1
specs read only events, so a dead `let` matters only where it fails on its own (P0-1).

Events. A bulk store (TMA or non-tensor, store or reduce) is a `bulk_store` per
issuing thread: its warp, its lane (absent for the elected lane of a full-warp
`elect.sync`), the shared source (symbol, byte offset, bytes: a TMA store's hardware box
times the element bytes, a bulk store's size) and, through an in-kernel descriptor, the
128-byte tensormap it reads (`tmap`: Triton's scratch and the slot's byte offset).
`cp.async.bulk.commit_group` and `cp.async.bulk.wait_group{.read} N` are `bulk_commit`
and `bulk_wait` events of the role. Writes of shared memory are `smem_write` events per
executing thread (in a thread loop): st.shared and generic st to a shared address
(vector width times element size), atom and red on shared memory, stmatrix (m8n8
.x1/.x2/.x4: each address-providing lane writes a 16-byte row; `sync`: one
warp-synchronizing instruction), and with `async` the landings of TMA loads (the
hardware box), bulk g2s copies (their size) and cp.async (its copy size), and of
writes into another CTA (st.async, red.async, shared::cluster bulk copies and reduces
through a mapa address: `async` with `cta`, the destination's rank). The landing of a
TMA or bulk load names its load's mbarrier (`bar`, `idx`: the slot of the one
`tma_load` the translation emits for it, right after it in the role's trace with only
its footprint `gload` between), which lets P1-2 order the landing before a later store
through that mbarrier's bytes (`specs.md` P1-2); a cta_group::2 load completing on its
pair leader's mbarrier names that slot with `leader`, as its `tma_load` does; a later
message of a Triton block (no `tma_load` of its own), a multicast load or one whose
mbarrier is a select between arrays names none.
`tensormap.cp_fenceproxy` is a `tmap_write` of each executing warp. Shared symbols:
every `.extern .shared` variable starts at the dynamic shared memory, so all of them are
the first one.

Pruning and disjointness. The byte interval each write, store and fixed object covers
is bounded statically over its executing threads (`expr.interval`, plus urem and srem
by a positive constant, the ring index of a loop-carried counter; P0 keeps
`expr.interval`). A `smem_write` that cannot meet any store source of its symbol is
dropped. Shared bytes written outside the events must not meet a store source: the
mbarrier objects (8 bytes per initialized slot), tensormap staging buffers
(`tensormap.replace`, 128 bytes), tcgen05.alloc's address slot (4 bytes) and CLC
responses (16 bytes); a write under a predicate that depends on data must not meet one
either. (The pruning compares offsets across CTAs: every CTA has the same shared-memory
layout.) An
interval that cannot be bounded counts as meeting everything.

Refusals (`P1-2: ...`; P0 is unaffected): a store, write or group under a lane
condition that depends on runtime values; a store several lanes of a warp may execute
(its issuers are run-time values); a commit or wait that a bulk store issuer of its role
does not execute (warp 0 alone executing one covers its elected lane, as Triton emits
around a descriptor copy); `wait_group.sync_restrict::proxy::async`; an mbarrier wait
executed by part of a warp; an address that is no shared symbol plus an offset, a
tensormap address of unknown provenance; a packed sub-byte descriptor; a multicast TMA
load; stmatrix shapes other than m8n8; the fixed objects and data-predicated writes
above meeting a store source; `mbarrier.inval` (as P1-1's pass).

## The P0-6 barrier pass (both families)

`frontend/ptx/nbar.py` translates every variant's PTX once more, for P0-6 "named
barriers convergent" (`specs.md` "P0-6"), into the P0-6 KIR (`<spec>.nb.json` next to
the TTGIR dump, `<Kernel>__<digest>.nb.json` in the CuTe `kir/`; `kir.md` change 14):
in both families, whatever `LKV_FRONTEND` selects (also `legacy`), and also for variants
whose P0 translation is refused, but for a cluster geometry LKV does not check ("Launch
records", "Clusters, tcgen05, CLC"). It reuses this frontend's parser, CFG, regions, roles
and exact interpreter (`interp.Translator` subclassed), and emits no P0 event: its slice
is the control flow, the named barriers and the values that decide them.

- **Roles.** The roles above, a padding state's role included whether or not it emits
  an event (its warps run the code before the dispatch too); split into warp ranges (SplitWarps) at every
  barrier-deciding condition that is uniform within each warp but differs over the role
  (`partition1:w12-12`, `warps4_7`). The dispatch's fork barriers (`barrier.sync 1`
  without a count) are skipped: cfg.py checks their protocol.
- **Values.** Exact expressions as above; data carries a class over the role's threads,
  `u` (one value), `l` (per lane, the same in every warp), `w` (per warp), `t` (any),
  `nondet` (strong loads, atomics, timers, test/try_wait outside the wait idiom) and the
  strong loads it comes from. Class-`u` data without a strong load becomes a fresh
  uniform symbol, so arithmetic on it stays exact (an address keeps its base tensor, a
  condition on it its comparison); a KIR expression never reads one (a branch on it is
  an `nb_rep` region). Loads by address class: shared memory (`ld.shared`, generic,
  `::cluster`, `ldmatrix`: per lane), weak loads of tensors the kernel never writes;
  collectives (`redux`, `vote`, `shfl.idx` from one lane) of per-lane data are uniform.
  A weak integer load of a 1-D tensor whose contents the dump recorded and the kernel
  never writes is an exact metadata `load` (mem.py, as in the P0 translation): the TTGIR
  hook declares such tensors with their `<t>.shape.0`/`<t>.stride.0` params and
  `_input`'s checks (recorded, no view, contiguous, not stale, no storage shared with a
  written tensor; a tensor that fails them is translated again as uniform data), the
  CuTe hook as flat buffers of `<t>.nb_len` elements (every launch recorded the contents,
  1-D contiguous, no storage shared with a written tensor).
  A spin loop's accepted value (every source a strong load of a uniform address its exit
  test accepted; also LLVM's rotated `if accept(v0) skip; else spin`) is published
  (`u`). An instruction without an op table row computes `t` data unless it may be a
  barrier or a branch (error).
- **Control flow.** An If deciding no barrier runs both arms for their values; one
  deciding a barrier is a KIR `if` (exact uniform), a split (per warp), two `nb_rep`
  regions (class-u data or a symbol), or divergent: both arms translated with every
  barrier in them an `nb_open` whose reason is the innermost certain enclosing
  condition. A loop is exact (KIR `for`/`while`) if interp's `run_loop` takes it, else
  iterated to a class fixed point (at most 8 rounds) and translated once more: an
  `nb_rep` region if its exit is uniform, divergent (`nb_open` kind `loop`) if not. Each
  class also records whether the value is uniform over the role before warp splitting
  (thread-index instances over all its warps; symbols and loads inherit it); an `nb_rep`
  region whose condition is gets the key `<role>@<PTX line>`, the same in every split
  role, so Lean may pair a barrier the split roles share in it (`specs.md` "P0-6").
  Classes also record whether a value is alike in corresponding lanes of every warp of
  that role, so a warp collective (`vote`, `redux`, `shfl.idx` from one lane) of such
  per-lane data (e.g. a varlen tile scheduler's warp scan of sequence lengths) keys too.
  A barrier-free If on an exact condition uniform over the role around a loop is
  interp's KIR `if` (else its arms run under the mask), so the loop can be exact.
- **Shape.** cfg.build's refusal of irreducible flow is answered by node splitting (the
  blocks entered from outside a cycle other than at its header are copied for those
  entries, up to the header) and a loop with exits to several blocks by copying a path
  that only returns to its header (a test LLVM rotated into one path) into the loop; the
  copies are counted in the KIR's `nbar.split_blocks`. The P0 translation keeps
  cfg.build's refusals.
- **Refusals** (P0-6 UNDECIDED for the variant, `nb_not_checked`): what cfg.py and
  regions.py still refuse (several dispatches, a loop with exits to two blocks, several
  rets; a switch `brx.idx` is `cfg.lower_switches`' compare chain as in the P0
  translation, refused only if its index is not provably in range), a call, roles that
  are not warp ranges, a parameter the launches do not give; failures are ERRORs (`kind`
  `error`).
- **Audit.** The KIR's `nbar.sites` (copied into report.json `variants_nb[].nb_sites`)
  lists every barrier instruction reached with its role, PTX line, loc, status
  (`counted` or `open`), kind, certainty and reason.
- **Cost.** Classifying a value enumerates its thread-index instances, so the pass
  classifies only what can decide a barrier: a class fixed point classifies the registers
  live at the loop's header or after its exit (regions.py liveness; a register the loop
  writes and nothing reads later reaches no decision) and a weak load's address only if
  the loaded value's class depends on it (shared memory, a tensor the kernel never
  writes). The substitutions are memoized per translator across
  expressions (interp's `memo`; the values of a role share their thread-dependent
  subexpressions). The TTGIR hook translates the specializations in up to `--jobs`
  forked processes (`to_kir --jobs`, which `lkv.py run` passes; the results and files are
  the same in any order).

## Refusals

A construct the frontend cannot translate exactly stops its variant with a message naming
the instruction, its source line, its PTX line and the reason:

```
[to_kir] 946e2014bc4d: FAILED (gap): bra at fbcode/k.py:118 (PTX line 9861): data value from setp of data at fbcode/k.py:117 reaches a branch condition (slice soundness)
```

TLX writes these lines to `logs/to_kir.log`; CuTe records them in `kir/report.json`
(`"error"` prefixed `PTX at <file>.ptx: `, and `"kind"`). Both reach report.md and
report.json `frontend_failures`. `gap` is a construct refused by design: every spec of the
variant's inputs is UNDECIDED, not an error. `error` is something the case, the dump or
LKV must fix (an unknown instruction, a case annotation error, invalid KIR, a PTX the
parser cannot read, an internal error): ERROR. Common reasons:

| Message | Meaning |
|---|---|
| `data value from ... reaches ... (slice soundness)` | a dropped value (float, shared or tensor memory contents, a data load) decides an event, a branch, an address or a predicate; the origin names the instruction and line that produced it |
| `irreducible control flow`, `the loop at ... exit edges`, `N ret/exit`, `guarded ret`, `trap at line N is not reached by one conditional branch` | a control-flow shape outside the supported ones |
| `warp-specialize dispatch (<item>): ...` | a TLX dispatch that does not match the protocol (a Triton change of its lowering) |
| `a generic write of shared memory in a warp-specialized kernel`, `atom.shared... in a warp-specialized kernel` | a TLX dispatch kernel that writes shared memory through `cvta.shared` or a shared atomic |
| `mbarrier.inval of ... ends no private synchronous cycle ...` | an inval whose slot's events since its init are not exact arrivals and one wait of parity 0 in one statement list |
| `no op table row for ...` (error) | an instruction without a row: `ir_ops_ptx.md` and `ops.py` need one |
| `... that several lanes of a warp may execute, compiled by ptxas X for sm_N: its issuing is measured for ...` | a multi-lane TMA or bulk copy from a compiler or target whose issuing was not measured |
| `a global access whose address has no tensor provenance` | an address from a loaded pointer, an integer or an unnamed parameter |
| `... (sub-element access)`, `byte offset ... not provably a multiple ...` | an access that is not whole elements of its tensor |
| `mbarrier address is not a shared-memory symbol plus an offset`, an unresolved array or index | a barrier the arrays cannot place |
| `TMA load completing on an mbarrier of another CTA (a mapa result) ...`, `... in a cluster of N CTAs without .multicast::cluster::all` | a cluster idiom outside the supported ones (a load completing on neither the issuing CTA's nor its pair leader's barrier; a CLC response only the asking CTA receives) |
| `... with only part of warp(s) [...] active (undefined in PTX)` | a full-mask collective or aligned instruction under a partial warp |
| ``case `work`: ...``, metadata and stale-data messages | the case or the host run (`frontend_ttgir.md`, `frontend_cute.md`) |

To try a new case, run one input (`lkv.py run --inputs`), read the refusal, and open the
source line it names; the PTX it quotes is in the run's `frontend/` directory.

## Cost

Lean dominates a run (`specs.md`, "Limits"). The KIR size it scales with comes from the
events per access site (CuTe: a thread loop per site; TLX: one event over the threads),
lane tables of warp-cooperative code (32 values per step in every role that runs it),
roles split per warp range, `while` loops (only constant-bound counted loops are
`for`s), and exact expansions whose thread-dependent operands cannot be bound by a `let`
(a 64-bit `mul.hi` that is no division by a constant: a tree of 16-bit limbs). The translator keeps it down: only the slice is emitted, subexpressions are
shared through `let`s, thread offsets fold into closed forms over the thread index (TLX),
adjacent accesses of one source line merge into one multi-lane event, and the CuTe
assembly rerolls LLVM's unrolled blocks into `for` loops (the same trace).

## Limitations

* Known gaps: the translation refuses some kernels and constructs that the legacy
  translations handle ("Refusals"); `--frontend legacy` runs the TTGIR translation of a
  TLX/Triton kernel, and a CuTe kernel stays on the CuTe DSL MLIR unless `--frontend ptx`.
* A verdict covers the case inputs, the compiled variants they select and `max_iters`.
* Below PTX nothing is checked (ptxas, SASS), except the TMA issuing rules measured on
  hardware (`ir_ops_ptx.md`).
* Shared memory, TMEM, MMA math, named barriers and fences are outside the P0 slice; TMEM
  accesses, `tcgen05.wait` and named barriers are P1 events (CuTe), fences and TMEM
  lifetime are not checked.
* Preferred cluster geometries (TLX/Triton) are checked all-preferred and all-regular (P0-6
  and P1-2 too); P0-3 of a launch that mixes them is covered only where both geometries
  write the same elements per preferred-size group of CTAs, which is not checked, and
  P1-2's tensormap check across clusters only between clusters of one geometry ("Clusters,
  tcgen05, CLC"). A CuTe launch with a preferred cluster is refused (gap), as is a TLX
  launch whose function's required cluster (cuFuncSetAttribute) differs from its cluster
  dims.
* Refused by design: CuTe CLC, P1-1 on TLX, TMEM addresses relative
  to the allocation, multicast bulk copies, multicast or cta_group::2 TMA loads and
  shared::cluster bulk copies and reduces that several lanes of a warp may execute (issuing not
  measured, `ir_ops_ptx.md`), im2col TMA, packed sub-byte TLX descriptors, descriptors
  edited in global memory, `mbarrier.inval` outside
  a private synchronous cycle, carry-flag instructions under a predicate, generic writes
  of shared memory and shared atomics in TLX warp-specialized kernels, the control-flow
  shapes above, and anything whose value would have to come from Data.
* A value one role stores to shared memory and another role reads after an mbarrier
  handoff (a mailbox) is Data, so a branch, loop exit or address on it is refused. A TLX
  layer-norm backward with an affine epilogue hands its reducer warps a 32-word header
  per tile (`red_hdr`: the work item's sequence index on the item's last Q tile, -1
  otherwise, -2 at shutdown) and they branch on its maximum (exit at -2) and address
  global stores with it. A sound rule needs a cross-role value: a put event in the writer,
  a get in the reader that returns the k-th put of that location, an evaluation that runs
  the writer before the reader (none in KIR: roles are evaluated independently), and a
  check, as the barrier ledger does for phases, that the k-th put happens before the k-th
  get, the k-th get before the (k+1)-th put, and that only those puts write the location
  (every word: the reader reduces over 32 lanes). A value the same role stores and reads
  back after a barrier of its own threads would need the last two only.
* Memory ordering is not modeled: `.volatile`, `.relaxed`/`.acquire` and scopes change
  no event. A weak integer global load reads the input's recorded contents; a strong one
  (`.volatile`, `.relaxed`, `.acquire`) or a `.cv` one is Data, as another thread may
  write it.
* The elected lane of `elect.sync` is unspecified: operands under it must not depend on
  it, and a value that does is refused where it matters.
