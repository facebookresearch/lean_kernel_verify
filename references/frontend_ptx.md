# PTX translation (Layer 1, both families, opt-in)

`lkv.py run --frontend ptx` makes both frontends translate the kernel's PTX, the code its
compiler hands to ptxas, into KIR (`kir.md`) instead of their compiler IR, with one shared
core and no hand-written model of the kernel. The default, `--frontend legacy`, translates
the TTGIR (`frontend_ttgir.md`) or the CuTe DSL MLIR (`frontend_cute.md`). `run` passes
the choice to `to_kir` as `LKV_FRONTEND` (`legacy` or `ptx`; unset or empty is `legacy`,
any other value an error), which also selects it when set in the environment of `run`
(the default of `--frontend`) or of a frontend's `to_kir` run by hand.

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
size, alignment of every parameter); any difference fails the dump.

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
`ret`/`exit`, `brx.idx` only in the Triton dispatch (below). Anything else (a guarded
`ret`, a loop with exits to two blocks) is refused naming the lines.

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
* The blocking wait loop around `mbarrier.try_wait.parity` is one `wait`
  (`ir_ops_ptx.md`, "mbarriers").

## Threads and roles

Within a role, a thread-varying value is an integer expression over LANE, WARP and the
elect symbols; a value is uniform if substituting every thread of the role gives one
result (exact BitVec folding plus exact intervals). Shuffles and ballots substitute
lanes, and a value computed from per-lane loads and shuffles is a table of 32 per-lane
values, so warp-cooperative code (a varlen tile scheduler's per-lane `cu_seqlens` loads,
prefix sums and ballots) is exact. The block must be (32k, 1, 1), and every launch of a
variant must have the same block and cluster.

* **TLX/Triton.** A kernel without `brx.idx` is one role, `default`. A warp-specialized
  kernel's dispatch must match Triton's protocol item by item, on the CFG and reaching
  definitions (not on adjacent instructions): `wid = shfl.sync.idx(%tid.x >> 5, 0)` and a
  branch on `wid < D` of either polarity; the worker switch loop (optional `setmaxnreg`,
  `barrier.sync 1`, `ld.shared.b8 s, [smem + wid + K]`, the out-of-range bypass,
  `brx.idx s`); the default role's immediate stores of every worker's state before its
  first fork barrier and of the exit state before its last; a fork-barrier count that no
  path changes. Each worker runs the switch loop twice, into its state and into exit, so
  state k's role `partition<k>` is the code before the dispatch, one pass of the loop
  header and the region from target k back to the header, then `ret`; `default` is warps
  0..D-1. A state whose region only synchronizes (padding) has no role. A failing item
  refuses the kernel and names the item. A branch on the warp inside a role is a mask.
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
bit field. In `(x + c) << k` LLVM keeps only the low bits of `c` that survive the shift
(`(n - 2) << 7` may become `(n + 0x1FFFFFE) << 7`), so the intermediate wrap is judged on
the constant closest to 0 with those low bits: the rewrite reports no wrap the source does
not have (and a hand-written such address with a huge `c` is not reported either).

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
parameter, or the pointer field of a struct parameter) and a shared symbol's. A global
access (`ld`/`st.global`, generic `ld`/`st` on a global address, `atom`, `red`,
`cp.async`) peels its address into (tensor, byte offset) through adds, subtracts and
selects; a select between two tensors takes the arm the role's condition picks (CuTe: a
condition uniform per warp splits the role). The element offset is the byte offset over
the element size, provably exact, else the access is refused. An integer global load of
the register's width whose value reaches a root is a metadata read, KIR
`load(T, [index])`, and T is a metadata tensor whose recorded contents the input
carries. The family assembly checks that the contents are those the kernel reads
(`frontend_ttgir.md`, `frontend_cute.md`).

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
over candidate arrays would make unreachable waits static waiters). Counts must be host
values.

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
  buffer zeroed by a full warp (`st.shared.b32 0`), filled by `tensormap.replace` and
  copied to global memory by `tensormap.cp_fenceproxy`. Each cp_fenceproxy takes the
  fields written since its buffer's last zeroing (one buffer may serve two descriptors in
  a tile); the zeroing, the replaces and the copy must be in one statement list. A TMA
  instruction uses the descriptor a cp_fenceproxy of its role wrote at the same address
  earlier in the same loop iteration, on an enclosing path; anything else is refused.
  Stores and reduces through it are `gstore` / `gatomic` of its base pointer with exactly
  the TMA footprint (extents read unsigned: the tensormap's `global_dim` is a b32), so
  `intended_writes` names the base pointer; a load through one is refused.
* **Bulk copies** (non-tensor `cp.async.bulk` global -> shared): the 1-D descriptor
  `bulk_<T>_<n>` over T's storage (box n elements), for the transaction bytes, plus the
  footprint `gload`.

Coordinates, footprints, transaction bytes, issuing threads, sub-byte formats:
`ir_ops_ptx.md`, "TMA and bulk copies".

## Clusters, tcgen05, CLC

* **Clusters (CuTe).** The cluster dims come from the LaunchRecords and are the KIR
  `cluster` (`kir.md` change 11); remote arrives (`cta`), multicast TMA loads and commits
  (`cta_mask`), cta_group::2 loads completing on the pair leader's mbarrier (`leader`),
  shared::cluster bulk copies (`complete_tx`) and the cluster barrier follow
  `ir_ops_ptx.md`, "Thread barriers and clusters". P0-2 and P0-4 decide one barrier
  ledger per cluster. Which CTA's data a copy carries is not checked. A TLX kernel
  launched with clusters (2-CTA) is refused.
* **tcgen05 in P0.** No event except `tcgen05.commit`, an asynchronous `commit_arrive` on
  its mbarrier; the P0 specs do not see tensor memory.
* **CLC (TLX).** `try_cancel` and its queries follow the schedule model of
  `frontend_ttgir.md`, "Cluster Launch Control": every role becomes
  `let __clc_k = 0; if __clc_launched[pid.x] != 0 { ... }` and its k-th query reads
  `__clc_schedule[pid.x, k]`. CLC in a CuTe kernel is refused.

## Work anchors

A case's `work` names, per work field, where its value comes from
(`frontend/ptx/anchors.py`):

* **Assignment anchor** (TLX: `{"batch": "off_z"}`): each assignment of the variable in
  the kernel function's Python source is an occurrence; its instructions are those whose
  kernel-file `.loc` frame lies in the assignment's span, and the variable is the one
  result that leaves that span (a register read outside it). An element of a tuple target
  narrows to the instructions computing that element. No result, or several, is a case
  error naming the candidates. A field the compiler folded away (its assignment runs, no
  register holds its value) is defined from there on with no value: -1, refused if a
  segments `per` names it.
* **Load anchor** (`[{"load": "T", "where": "text"}]`; CuTe, and TLX in this translation
  only): `where` matches a source line as a substring of the line or of
  `<relpath>:<line>`; every global load of T located on a matching line is an anchor (DSL
  4.4.2: by line number only), and the field becomes the load's element index. A `where`
  that matches no line is a case error.

Gating: TLX emits a `work` event in each region that defines every field, after the last
definition; a role that never defines some field gets none, and no region defining all of
them is a case error. CuTe emits one at every anchor; fields are role variables, -1 until
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

TLX/Triton kernels have no P1 pass: report.json lists their variants as `not_checked`.

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
| `irreducible control flow`, `the loop at ... exit edges`, `N ret/exit`, `guarded ret` | a control-flow shape outside the supported ones |
| `warp-specialize dispatch (<item>): ...` | a TLX dispatch that does not match the protocol (a Triton change of its lowering) |
| `no op table row for ...` (error) | an instruction without a row: `ir_ops_ptx.md` and `ops.py` need one |
| `... that several lanes of a warp may execute, compiled by ptxas X for sm_N: its issuing is measured for ...` | a multi-lane TMA or bulk copy from a compiler or target whose issuing was not measured |
| `a global access whose address has no tensor provenance` | an address from a loaded pointer, an integer or an unnamed parameter |
| `... (sub-element access)`, `byte offset ... not provably a multiple ...` | an access that is not whole elements of its tensor |
| `mbarrier address is not a shared-memory symbol plus an offset`, an unresolved array or index | a barrier the arrays cannot place |
| `cluster (2, 1, 1): Triton kernels with clusters ...` | a TLX 2-CTA kernel |
| `... with only part of warp(s) [...] active (undefined in PTX)` | a full-mask collective or aligned instruction under a partial warp |
| ``case `work`: ...``, metadata and stale-data messages | the case or the host run (`frontend_ttgir.md`, `frontend_cute.md`) |

To try a new case, run one input (`lkv.py run --inputs`), read the refusal, and open the
source line it names; the PTX it quotes is in the run's `frontend/` directory.

## Cost

Lean dominates a run (`specs.md`, "Limits"). The KIR size it scales with comes from the
events per access site (CuTe: a thread loop per site; TLX: one event over the threads),
lane tables of warp-cooperative code (32 values per step in every role that runs it),
roles split per warp range, and `while` loops (only constant-bound counted loops are
`for`s). The translator keeps it down: only the slice is emitted, subexpressions are
shared through `let`s, thread offsets fold into closed forms over the thread index (TLX),
adjacent accesses of one source line merge into one multi-lane event, and the CuTe
assembly rerolls LLVM's unrolled blocks into `for` loops (the same trace).

## Limitations

* A verdict covers the case inputs, the compiled variants they select and `max_iters`.
* Below PTX nothing is checked (ptxas, SASS), except the TMA issuing rules measured on
  hardware (`ir_ops_ptx.md`).
* Shared memory, TMEM, MMA math, named barriers and fences are outside the P0 slice; TMEM
  accesses, `tcgen05.wait` and named barriers are P1 events (CuTe), fences and TMEM
  lifetime are not checked.
* Refused by design: TLX clusters (2-CTA), CuTe CLC, P1-1 on TLX, TMEM addresses relative
  to the allocation, multicast bulk copies, multicast or cta_group::2 TMA loads and
  shared::cluster bulk copies that several lanes of a warp may execute (issuing not
  measured, `ir_ops_ptx.md`), im2col TMA, packed sub-byte TLX descriptors, loads through
  in-kernel descriptors, the control-flow shapes above, and anything whose value would
  have to come from Data.
* The elected lane of `elect.sync` is unspecified: operands under it must not depend on
  it, and a value that does is refused where it matters.
