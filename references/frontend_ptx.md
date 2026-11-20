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
the launch records below whichever translation runs), the variant names (plus
`<spec>@cluster<dims>` for a TLX launch's all-preferred cluster geometry),
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
  |                and bulk.py (P1-2)
  |   anchors.py   the case's work anchors
  v
Facts (or the refusal that stopped the variant)
  | frontend/ttgir/to_kir.py build_ptx | frontend/cute/to_kir.py PtxVariant.assemble
  v
KIR, `ir` "ptx", written only if frontend/kir.py accepts it; a P1 KIR from a second pass
(CuTe P1-1, TLX/Triton P1-2); the P0-6 KIR from a pass of its own (nbar.py, both
families, either translation); each again for a TLX launch's all-preferred geometry
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
not be the one that runs; `to_kir.check_required_cluster`, both TTGIR translations, the
P0-6 and the P1-2 pass); an absent attribute is the launch's cluster.

Parameters are named by position, never by guessing from addresses: TLX by the jit
argument names in Triton's own parameter packing (a host descriptor expands to its
tensormap and its shape and stride arguments; two scratch pointers come last); CuTe by the
trace (`jit_params`): a parameter that passes a jit argument through names that tensor,
also when it is empty (data pointer 0) or passed under two names, and a computed one (a
descriptor, a view, a struct of pointers) names the tensor its address resolves to among
the arguments it is computed from. A pointer no tensor can be named for exactly is
unnamed: its value is Data, so an access through it is refused (slice soundness).

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
interval; out of range is undefined in PTX), for a table of one target too, and a kernel
that already has a register or label named like the chain's (`%__sw<line>_<k>`,
`$__sw<line>_<k>`) is refused. Anything else (a guarded `ret`, a loop with
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

* **TLX/Triton.** A kernel without a warp-specialize dispatch (the switch loop's `brx.idx`
  or compare chain, or the direct dispatch, below; a switch `brx.idx`, "Control flow", is
  none) is one role, `default`. A warp-specialized
  kernel's dispatch must match Triton's protocol item by item, on the CFG and reaching
  definitions (not on adjacent instructions): `wid = shfl.sync.idx(%tid.x >> 5, 0)` and a
  branch on `wid < D` of either polarity; the worker switch loop (optional `setmaxnreg`,
  `barrier.sync 1`, `ld.shared.b8 s, [smem + wid + K]`, then the out-of-range bypass and
  `brx.idx s`, or, with at most 3 worker states, the chain of `s == v` branches LLVM
  lowers the switch to); the default role's immediate stores (a vector of immediates
  too) of every worker's state before its first fork barrier and of the exit state
  before its last; a fork-barrier count that no path changes (a fork barrier: barrier 1
  without a thread count, as `barrier.sync`, `barrier.sync.aligned` or `bar.sync`). Each
  branch condition must be the only result of its `setp`. A generic write of shared
  memory (through `cvta.shared`) or an `atom`/`red.shared` in such a kernel is refused:
  the state bytes are checked against `st.shared` only (the direct dispatch below has no
  state bytes and refuses neither). Each worker runs the switch loop twice, into its
  state and into exit, so state k's role `partition<k>` is the code before the dispatch,
  one pass of the loop header and the region from target k back to the header, then
  `ret`; `default` is warps 0..D-1. A state whose region only synchronizes (padding)
  has a role only if its translation emits an event (a cluster kernel's padding warps
  arrive on the cluster barrier, "Clusters, tcgen05, CLC"). A failing item refuses the
  kernel and names the item. A branch on the warp inside a role is a mask.
  A dispatch may run no round (item rounds, cfg.py): the default decides once, at a
  branch outside every loop, on every path and before any fork barrier, whose two sides
  meet only at `ret`; one side runs the protocol above, the other stores the exit state
  for every worker before its one fork barrier and returns (a TLX kernel with an early
  exit when no work is left). Its workers then pass the header
  once, read the exit state and return. The default's branch condition is the decision:
  it must be one value of the launch (parameters, program id, metadata: a weak integer
  load of a tensor), else the variant is refused (data, or a value that differs between
  the default's threads: the workers learn it from the state bytes only). Each worker's
  region is a KIR `if` on it (the code before, the header's first pass included, runs in
  either case), so on an input where the default takes the zero-round side no worker
  event runs; the P0-6 pass does the same (and refuses a decision that is data).
  The direct dispatch of Triton's single-warp-specialize lowering
  (`tlx.async_tasks(exclusive=True)`: one warp_specialize in the module, no state bytes,
  no switch loop) is matched too: the entry as above; the worker side's first block with
  a fork barrier (outside every loop; `barrier.sync 1`, optional `setmaxnreg`, pure
  instructions) ends in the first link of a chain of branches whose conditions compare
  wid with an immediate (any comparison and type; each later link pure, with the previous
  one as its only predecessor); each
  worker warp follows the comparisons, evaluated at its warp id, to its partition's
  first block (warps that reach one block through different links are refused, unless it
  only synchronizes: that padding block has a role per path). Every partition region has
  one fork barrier on every path to `ret`, the default two, and the regions share no
  block with the default, the dispatch or the code before it but the `ret` block. Roles:
  `default`, then `partition<k>` numbered by first warp, the partitions that do more
  than synchronize first, padding last (Triton gives larger partitions the lower warps,
  so k is the TTGIR partition index where the warp counts do not grow in TTGIR order).
  Captures: Triton passes the values a partition captures through shared memory: the
  default stores them before its first fork barrier, each worker loads them between its
  first and second one (both protocols). A worker's `ld.shared` there of bytes the
  default stored gets the stored value (byte by byte, so a capture split into bytes is
  put back together) if every thread of the default runs the store (no predicate or
  thread condition, branch on a uniform condition or loop around it, but the decision of
  a dispatch that may run no round, on its round side, under which the workers load it:
  a store of only some threads may land before another thread's earlier store of the
  bytes, or not happen) and its value is one value in every thread that
  depends on nothing but the launch (parameters, program id, shared symbols, metadata);
  otherwise the load is Data as any shared load (so is a worker's load before its region
  in a dispatch that may run no round). Every other write of shared memory
  reports the bytes it may touch, over the threads that may execute it, whichever lane
  `elect.sync` elects (the bounds of its offsets plus its size; any byte where an offset
  depends on the elected lane): `st.shared` (an unknown address, a condition, a value of a
  role, a worker's, the default's after its first fork barrier), `stmatrix` (a 16-byte
  row per address-giving lane), `atom`/`red.shared`, generic `st`/`atom`/`red` through
  `cvta.shared` (a direct dispatch does not refuse them) or a data address (any byte),
  the landings of TMA loads (the
  hardware box; any byte through a packed sub-byte descriptor, whose shared-memory copy
  may be wider than its packed bytes), bulk copies and `cp.async`, a CLC response (16 bytes),
  `st.async`/`red.async`, `tcgen05.alloc`'s address slot, `tensormap.replace` of a
  staging buffer and `mbarrier.inval` (the op-table rows that write shared memory are
  interp.SHARED_WRITERS, by kind, the table every model of those writes dispatches on:
  the captures, the P1-2 pass and the tensormap staging buffers; tests/ptx/test_captures.py
  checks the table against them, and that each reaches a capture hook and a P1-2 rule). The bounds come from interval arithmetic, including masks and the ring-buffer
  index `x % k` as LLVM writes it for any x (`x - ((x + b) & -k)`, `x - q k` with q a
  multiply-high quotient, its multiplier a 32-bit one): bounds on the arithmetic, no
  claim about the quotient, and none where the arithmetic may wrap at its width. Such a
  write before the workers' loads makes the captured bytes it may touch unknown (all of
  them if its address is no shared symbol plus an offset); roles are translated one
  after the other, the default first, so such a write of a byte another role already
  read refuses the kernel. A write after its role's second fork barrier (counted on its
  path: each arm of an `if` from the count before it) follows every worker's loads,
  unless it lands in another CTA's shared memory (`st.async`,
  `red.async`, a shared::cluster bulk copy or reduce, a multicast TMA, bulk copy or CLC
  response): no barrier of the CTA orders that one, so it counts wherever it is issued.
  An asynchronous write (a landing, a CLC response) may land after a later store: the
  default records no capture over bytes an earlier asynchronous or remote write may
  touch. A write whose offsets have no static bounds (an offset LLVM carries
  strength-reduced through a loop, unrelated to the counter it was derived from) keeps
  the captures under KIR `assume`s (kir.md change 16; `a shared write misses the
  warp-specialize captures [lo, hi)`, per executing thread) that its bytes miss each
  run of known captures, evaluated on every input. Such a write over a capture is no
  certain violation (it may land after the workers read it), so where the assumption
  does not hold the role stops there: the input is undecided, not a FAIL (P0-1..P0-5,
  P1-2: `specs.md`, "Undecided specs"; the P0-6 pass reads no captures), and the report
  names the assumption. An mbarrier object changes at all its events but waits (some
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
(slice soundness)`): the frontend never guesses a dropped value (a value handed off
through shared memory: "Handoffs"). A node that can stop
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

## Handoffs

A value one role stores in shared memory and another reads after an mbarrier wait (a
mailbox: a work header) is Data to the reader, so a branch on it
refuses the variant. When the translation is refused that way, the stores of every role
are looked at (`Kernel.handoff_cands`): a 32-bit store of one value, the same in every
thread of the role, under no mask, whose bytes are the same in every warp of the role
(`st.shared.{b,u,s}32` at an address the same in every thread; `stmatrix
.m8n8.x{1,2,4}.shared.b16`, not `.trans`, whose lanes 0 .. 8x-1 address 16-byte rows that
are one run of bytes). Bytes stored that way by one role only, overlapping no other such
bytes, are a channel, named `<region>+<offset>` (`%extern+219392`: the region of the
`.extern` variables, which all start at the dynamic shared memory, is `%extern`), and the
variant is translated again with them (translate.py):

* such a store of exactly a channel's bytes by its putter is a `handoff_put` of the stored
  value (kir.md change 17). Any other write that may touch a channel (the shared writes
  the warp-specialize captures follow, `st.shared` and `stmatrix` included, and the
  mbarrier arrays) refuses the variant; one whose offsets have no static bounds gets a
  KIR `assume` that its bytes miss the channel;
* a 32-bit load under no mask of words of a channel by another role after an mbarrier
  wait of that role (`ld.shared.{b,u,s}32` at an address the same in every thread,
  `ldmatrix` by the `stmatrix` rule) is a `handoff_get`: its destinations hold the value
  it binds.

A variant the second translation refuses too keeps the first refusal, with the second's
reason. Whether the value was handed off is decided per input in Lean (kir.md change 17):
the get reads the putter's value at the arrive its last wait pairs with, and a run where
that pairing, or the putter's next put following the get, is not shown is a limit of the
model. In the P1-2 pass a partition split by warps names its parts' group (KIR `group`),
whose arrivals Lean takes in lockstep, as the P0 ledger takes the threads of one role.

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
| TMA | `tma_load` / `tma_store` / `tma_reduce` on the host descriptor; through an in-kernel descriptor footprint `gload` / `gstore` / `gatomic` on its base, a load also a `tma_load` of its box descriptor | `tma_load` plus its footprint `gload` on the base tensor; stores and reduces as footprint `gstore` / `gatomic` on the base |
| descriptor dims | outermost first | innermost first |
| locations | `fbcode/<path>:line` | `<path>:line` (DSL 4.4.2: best effort) |
| `while` budget | 65536 | the case's `max_iters` |
| work events | gated (below) | at every anchor |
| also | CLC (with clusters too), clusters, the all-preferred cluster geometry, the P1-2 pass | clusters, the P1-1 pass |

## Memory

Addresses are expressions over reserved bases: a kernel tensor's address (its pointer
parameter, or the pointer field of a struct parameter), a shared symbol's and a local
variable's (LLVM's `__local_depot`), and the generic addresses of the same shared and
local bytes, which only `cvta` makes (so a generic address is never taken for a window
one). A global access (`ld`/`st.global`, generic `ld`/`st`/`atom`/`red` on a global
address, `atom`/`red.global`, `red.async`, `cp.async`) peels its address into (tensor,
byte offset) through adds, subtracts and selects; a select between two tensors takes the
arm the role's condition picks (CuTe: a condition uniform per warp splits the role), and
one whose condition is one value in the role but not known (e.g. a program id or
parameter test) becomes one access per tensor under the condition and its negation (not
for a `cp.async` or bulk copy, nor a generic access through a select between a shared
and a global address: refused); the bases may come from `ld.param` through a select of
two parameter addresses. The element offset is the byte offset over
the element size, provably exact, else the access is refused. An integer global load of
the register's width whose value reaches a root is a metadata read,
KIR `load(T, [index])`, and T is a metadata tensor whose recorded contents the input
carries; not a strong load (`.volatile`, `.relaxed`, `.acquire`, also `.cv`), whose
location another thread may write while the kernel runs, nor a load of a floating-point
tensor (TTGIR: its recorded dtype), whose contents are never recorded: its value is Data. The family
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
  `cp.async`, TMA and bulk copies or reduces into shared memory, `st.async`/`red.async`,
  mbarrier and `tcgen05.alloc`/`commit` and CLC writes whatever their bytes. The zeroing,
  the replaces and the copy must be in one statement list; a `tensormap.replace` of a
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
  `<spec>@cluster<dims>`, with every input of the specialization), and the
  report names the geometry of each verdict. The P0-6 barrier pass writes both
  geometries too (`<spec>.nb.json`, `<spec>@cluster<dims>.nb.json`, whatever the
  translation): the rank and the cluster special registers may decide which warps reach a
  barrier (a `rank == 0` branch), and they differ by geometry. A mixed launch's clusters
  are each a cluster of one of them, with the same CTAs, ranks and special registers, so
  P0-1, P0-2, P0-4, P0-5 and P0-6 (per access, per cluster ledger, per CTA) hold for every
  mix if they hold in both;
  P0-3 counts writes over the whole grid and holds for a mix only if every aligned group
  of CTAs writes the same elements in both geometries, which is not checked
  ("Limitations"). P1-2 has a KIR per geometry too (`<spec>@cluster<dims>.p1.json`); it
  decides each cluster on its own except a tensormap that one cluster's stores read and
  another cluster writes (F2 across clusters), so it holds for every mix if it holds in
  both, except that cross-cluster check, which pairs only clusters of one geometry
  ("Limitations"). With CLC the all-preferred P0 and P1-2 KIRs are refused (`CLC with a
  preferred cluster geometry`, a gap: which clusters a try_cancel may cancel when the
  sizes mix is not modeled); the regular ones and both P0-6 KIRs are still checked. If
  the launches disagree on the preferred dims, or break a rule CUDA states for them
  (`preferred_geometry`: every launch has a regular cluster, i.e. cluster dims other than
  1 x 1 x 1 or the function's REQUIRED_CLUSTER, as a bare `.reqnctapercluster 1, 1, 1` is
  not in the LaunchRecord; the preferred x and y are multiples of the regular ones and z
  is the regular z; the preferred dims divide every launch's grid), the P0, P0-6 and
  P1-2 passes report the failure `<spec>@cluster` for the all-preferred geometry (a gap
  for every input of the specialization: what the device runs for such a launch is not
  modeled); their regular KIRs are still checked. The three passes check the same
  geometries (`to_kir.geometries`), decided before anything is translated: an
  all-preferred KIR is translated whether or not the regular one translates, and a
  launch whose function's required cluster differs from its cluster dims refuses each
  geometry of its specialization (`check_required_cluster`, "Launch records"; a refused
  `<spec>@cluster` keeps its own reason). A CuTe launch with a preferred
  cluster other than its cluster is refused, its P0-6 KIR too (gap: the CuTe frontend
  checks one geometry; `cute/to_kir.preferred_cluster_gap`).
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
A specialization without a bulk store is `not_applicable`, seen in its parsed PTX
before anything is translated (also where the translation would be refused); a refusal
or failure is logged as `P1 refused|error: <why>` and leaves its P1-2 "not checked"; the
P0 KIR and its verdicts never change (`LKV_P1=0` skips the pass: `lkv.py run --no-p1`
sets it, and `LKV_P1=1` otherwise). The roles are split as the CuTe
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
and `bulk_wait` events of the role. Writes of shared memory (the rows of
interp.SHARED_WRITERS, as for the captures, "Threads and roles") are `smem_write` events per
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
message of a Triton block (no `tma_load` of its own), a multicast load, one whose
mbarrier is a select between arrays or one whose load computes the slot index only after
the landing names none.
`tensormap.cp_fenceproxy` is a `tmap_write` of each executing warp. Shared symbols:
every `.extern .shared` variable starts at the dynamic shared memory, so all of them are
the first one.

Pruning and disjointness. The byte interval each write, store and fixed object covers
is bounded statically over its executing threads (`expr.interval`, plus urem and srem
by a positive constant, the ring index of a loop-carried counter, with no bound where a
value may wrap its width, as in `expr.interval`; P0 keeps `expr.interval`). A
`smem_write` that cannot meet any store source of its symbol is dropped, and so are the
loops and ifs it leaves empty (a loop left with `let`s stays: they may carry values out
of it). Shared bytes written outside the events must not meet a store source: the
mbarrier objects (8 bytes per initialized slot), tensormap staging buffers
(`tensormap.replace`, 128 bytes), tcgen05.alloc's address slot (4 bytes) and CLC
responses (16 bytes); a write under a predicate that depends on data must not meet one
either. (The pruning compares offsets across CTAs: every CTA has the same shared-memory
layout.) An
interval that cannot be bounded counts as meeting everything.

Refusals (`P1-2: ...`; P0 is unaffected): a store, write or group under a lane condition
that depends on runtime values, or that depends on which lane `elect.sync` elects (as
every event: a lane condition beside `elect.sync`, an operand that differs between the
lanes of a warp) or runs under a negated `elect.sync`; a commit or wait under a
predicate that depends on data; a store several lanes of a warp may execute (its issuers
are run-time values); a global store or atomic into a buffer that holds the tensormaps
of bulk stores (only `tensormap.cp_fenceproxy` is a tensormap write); a commit or wait
that a bulk store issuer of its role does not execute (warp 0 alone executing one covers
its elected lane, as Triton emits around a descriptor copy);
`wait_group.sync_restrict::proxy::async`; an mbarrier wait executed by part of a warp;
an address that is no shared symbol plus an offset, a tensormap address of unknown
provenance; a bulk copy whose size is no positive constant; a packed sub-byte
descriptor; a multicast TMA load; stmatrix shapes other than m8n8; the fixed objects and
data-predicated writes above meeting a store source; `mbarrier.inval` (as P1-1's pass).
Tagged `R7:` as in the P1-1 pass (the `named_bar` events, the role split): a named
barrier reached by part of the role, a named-barrier id or count that is not constant
(one uniform per warp splits the role), a count that is not a positive multiple of 32,
an mbarrier wait, arrive or commit under a runtime lane condition in a role of several
warps.

## The P0-6 barrier pass (both families)

`frontend/ptx/nbar.py` translates every variant's PTX once more, for P0-6 "named
barriers convergent" (`specs.md` "P0-6"), into the P0-6 KIR (`<spec>.nb.json` next to
the TTGIR dump, also `<spec>@cluster<dims>.nb.json` for a preferred cluster geometry,
`<Kernel>__<digest>.nb.json` in the CuTe `kir/`; `kir.md` change 14):
in both families, whatever `LKV_FRONTEND` selects (also `legacy`), and also for variants
whose P0 translation is refused, but for a cluster geometry LKV does not check ("Launch
records", "Clusters, tcgen05, CLC"). It reuses this frontend's parser, CFG, regions, roles,
driver (translate.py: refusals, a role translated per run of warps) and exact interpreter
(`interp.Translator` subclassed), and emits no P0 event: its slice is the control flow,
the named barriers and the values that decide them.

- **Roles.** The roles above, a padding state's role included whether or not it emits
  an event (its warps run the code before the dispatch too); split into warp ranges (SplitWarps) at every
  barrier-deciding condition that is uniform within each warp but differs over the role
  (`partition1:w12-12`, `warps4_7`). The dispatch's fork barriers (`barrier.sync 1`
  without a count, in any spelling) are skipped: cfg.py checks their protocol.
- **Values.** Exact expressions as above; data carries a class over the role's threads,
  `u` (one value), `l` (per lane, the same in every warp), `w` (per warp), `t` (any),
  `nondet` (strong loads, atomics, timers, test/try_wait outside the wait idiom) and the
  strong loads it comes from. Class-`u` data without a strong load becomes a fresh
  uniform symbol, so arithmetic on it stays exact (an address keeps its base tensor, a
  condition on it its comparison); a KIR expression never reads one (a branch on it is
  an `nb_rep` region). Loads by address class: shared memory (a weak `ld.shared`,
  generic, `ldmatrix`: per lane; `ld.shared::cluster` has no op table row, so `t`; a
  strong `ld.shared` is a strong load), weak loads of tensors the kernel never writes
  (by name);
  collectives (`redux`, `vote`, `shfl.idx` from one lane) of per-lane data are uniform.
  A weak integer load of a 1-D tensor whose contents the dump recorded and the kernel
  never writes is an exact metadata `load` (mem.py, as in the P0 translation): the TTGIR
  hook declares such tensors with their `<t>.shape.0`/`<t>.stride.0` params and
  `_input`'s checks (`_metadata_problem`: recorded, no view, contiguous, not stale, no
  storage shared with a written tensor; none for an input whose contents the case gives,
  `metadata_contents`), the CuTe hook as flat buffers of `<t>.nb_len`
  elements (every launch recorded the contents, 1-D contiguous, no storage shared with a
  written tensor). Both make those checks before the pass, with its write set
  (`nbar.writes`, policy `nb_written`), so a tensor that fails them is uniform data and
  the pass runs once, on the PTX parsed once.
  A spin loop's accepted value (every source a strong load of a uniform address its exit
  test accepted; also LLVM's rotated `if accept(v0) skip; else spin`) is published
  (`u`) in the register that holds it as loaded (or a copy) at every exit; a value
  computed from it in the loop is `t`, not nondet. An instruction without an op table
  row computes `t` data unless it may be a barrier or a branch (error). An instruction
  under a data predicate leaves its destinations, and the carry flag a `.cc` instruction
  writes, merged with their earlier values (the classes joined).
- **Control flow.** An If deciding no barrier runs both arms for their values; one
  deciding a barrier is a KIR `if` (exact uniform), a split (per warp), two `nb_rep`
  regions (class-u data or a symbol), or divergent: both arms translated with every
  barrier in them an `nb_open`, certain if every enclosing divergent condition is (for
  a barrier without `.aligned`, none of them by lanes alone) and it lies in no `nb_rep`
  region, with the innermost reason, else with the innermost condition that is not
  certain. A loop is exact (KIR `for`/`while`) if interp's `run_loop` takes it, else
  iterated to a class fixed point (at most 8 rounds) and translated once more: an
  `nb_rep` region if its exit is uniform, divergent (`nb_open` kind `loop`) if not. Each
  class also records whether the value is uniform over the role before warp splitting
  (thread-index instances over all its warps; symbols and loads inherit it); an `nb_rep`
  region whose condition is gets the key `<role>@<PTX line>`, the same in every split
  role, so Lean may pair a barrier the split roles share in it (`specs.md` "P0-6"),
  unless the regions' path signatures differ: a split role records what each register
  is computed from and under which decisions it resolved to one arm (or held as its own
  warps' instance), and `_sign` appends `#<n>` to a key whose split roles' signatures
  differ.
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
  are not warp ranges; failures are ERRORs (`kind` `error`), among them a parameter the
  launches do not give (`P0-6 pass: ...`).
- **Audit.** The KIR's `nbar.sites` (copied into report.json `variants_nb[].nb_sites`)
  lists every barrier instruction reached with its role, PTX line, loc, status
  (`counted` or `open`), kind, certainty and reason.
- **Cost.** Classifying a value enumerates its thread-index instances, so the pass
  classifies only what can decide a barrier: a class fixed point classifies the registers
  live at the loop's header or after its exit (regions.py liveness; a register the loop
  writes and nothing reads later reaches no decision) and a weak load's address only if
  the loaded value's class depends on it (shared memory, a tensor the kernel never
  writes). The substitutions are memoized per translator across
  expressions, as in every pass (interp's `memo`; the values of a role share their
  thread-dependent subexpressions). Inside an enclosing loop's fixed-point round, whose
  statements are dropped, a loop's final pass (its `nb_rep` or divergent region) is
  skipped: discarded work leaves no trace (interp's `restore` forgets it, the variable
  names it made included), so the KIR does not change. The TTGIR hook makes its P0-6
  KIRs (one per specialization and cluster geometry) as tasks of `to_kir`'s fork pool, with its P0 and P1-2 KIRs: up to `--jobs`
  forked processes,
  one KIR each (`to_kir --jobs`, which `lkv.py run` passes; the
  results and files are
  the same in any order).

## Global synchronization (policy `gsync`, gsev.py)

With the policy key `gsync` set to `True` (the TTGIR case key `gsync: true`), synchronization
through global memory becomes KIR events (kir.md change 18) instead of Data at a root. A
variant with no sync construct translates as without it, and so does every variant when the
key is off.

- *Sync tensors.* Seeded from gsync.py's census of the entry (spin waits, fetch-adds whose old
  value is used) through the parameter table, then grown by translating again
  (`gsev.Restart`) when a sync site or a tensor the variant reads as metadata and writes
  appears; the case may add `sync_tensors`. Every access of a sync tensor is a sync event; such
  a tensor is never metadata, and any other load of it (outside a spin) is refused.
- *Tagged words.* A spin loop of one block whose strong 64-bit loads of sync words feed only
  comparisons of their high halves (`mov.b64 {v, t}, d; setp.ne t, T`; tl.inline_asm spins
  such as `ld.relaxed.gpu.global.b64` / `@p bra`) for equality with targets that no load reads,
  and whose exit condition is true exactly when every loaded word carries its target (checked
  on all 2^n truth assignments of those comparisons), is a take of each word (gsev.py
  `tag_spin`): one `gwait` per word, binding the value half to a variable when every thread
  that runs the spin takes the same word and tag (else one per thread, the value half Data).
  After the loop a register the header computes from a loaded word holds its value on the last
  trip (the tag the target, the value half the variable; anything else that reads the word is
  Data). A trap-on-timeout exit sets `timeout` (taken to be never taken); other spin shapes
  are refused. A store to a sync word is a `gsig`; one whose value is Data but whose inline asm
  block packs it from an integer tag (`mov.b64 d, {data, tag}`) is a `gsig` of `tag << 32`
  with `data_lo` (fp32 partials).
- *Claims* (case `claims: "representative"`): a fetch-add whose old value the census
  classifies as a claim (it indexes memory or bounds a loop) is a `gclaim`, its old value a
  fresh variable; the KIR's top level then says `representative: ["claims"]`. Any other
  atomic on a sync word is refused, and so is a claim without the case key.
- *Relay channels* (case `relay_channels: [{symbol, start, slots, stride, name?}]`): a shared
  array region (`symbol` a regex of the shared symbol, from byte `start`) of `slots` slots of
  `stride` bytes, written once per slot and CTA (a claim ring): a st.shared to slot k is an
  `rput`, an ld.shared an `rget`. The slot is the constant one, or X for an offset
  `stride x (X mod slots)` (also C's `%` by a power of two, `X - ((X + bias) & -slots)`).
- *Other events*: `named_bar` for bar.sync / bar.arrive / barrier.sync in the P0 KIR (P0-7's
  progress).
- *Roles*: a sync event issued by part of a role's warps splits the role (as the P1 pass
  does), except a store right after a named barrier of the whole role with nothing but lets
  emitted since: the barrier already orders the other warps' earlier accesses.

### Protocol regions

Case `protocols: [{"tensor": T, "start": s?}]` (policy `protocols`, frontend/ptx/proto.py): T's
word is a small atomic protocol of a fixed group of CTAs (a team join: registrations by fetch-
add, polls, a timeout that sets an abort bit by CAS). Before the CFG shape check (proto.py works
on `cfg.build(check_exits=False)`), the accesses of T (any global access whose address may come
from T's pointer parameter) must lie in one region of one role: the blocks from the first access
(the start of its inline asm block) to the last, plus the pure blocks entered only from them at
which a register the region computed is live (the decision where to go on); entered only at its
start; holding only integer computation, `shfl.sync.idx`, nanosleep, branches and the word's
accesses, each an atomic or a strong (.relaxed/.acquire/.release/.volatile) load or store of
scope gpu or sys, at one address register the region does not write. Exits: the region's edges
out (through trampoline blocks). The region is cut out: `lkv.outcome %r, k, 0, <live-ins>`
(a synthetic instruction, ops.py) reads the CTA's exit from the metadata table `__proto<k>`
(row: the CTA's linear index; column j > 0: live-out j) and branches to that exit; the rewritten
function passes the shape check. The translation records at it the live-in registers' values
and the conditions of the uniform branches around it (closed expressions: params, CTA index,
metadata loads), refuses it inside a loop, under a thread-varying condition, or in a role of
several warps, and issues the word's footprint (a `gatomic` by lane 0: P0-1). T is then no
sync tensor.

Per input, the family's assembly evaluates the record per CTA (proto.py `agents`: the CTAs that
reach the region, grouped by their word) and explores each group (`explore`): every CTA runs the
region as an automaton of one warp (32 lanes; exact integer semantics; branches must be uniform
over the warp; an atomic is executed by one lane, a store by lanes storing one value, a load by
lanes of one address read one value, trusted: one transaction), and every interleaving of the
group's accesses of the word is explored from its start value (`start`, default 0: the value the
launch begins with, which the case asserts). Sound and complete for the word: the accesses are
morally strong with each other (same GPU, scope gpu or wider), so the PTX memory model's
SC-per-location axiom makes them an interleaving consistent with program order, each read
returning the latest write (aligned 32/64-bit accesses are single-copy atomic); timing is not
modeled (a nanosleep or a poll count is just computation): any CTA may be delayed by any amount,
so every timeout outcome is among the interleavings; nothing else reads or writes the word
(checked: one region, KIR v1 tensors do not alias). States are the word and each agent's
position and live registers, up to the permutation of agents with the same start (symmetry); an
outcome is a terminal state: per agent its exit and live-out values, expanded over the
permutations. Refused (the input is not built): more than `MAX_STATES` (3e6) states, an agent
that runs 1e5 instructions without an access, a cycle through distinct states (fair termination
is not decided), a state in which every agent only repeats an access that changes nothing (the
protocol hangs by itself). The inputs are the product over groups and protocols of the reachable
outcome vectors, named `<input>@proto:<tensor>:<vector>`, the vector each group's exits (digits
in linear CTA order, groups by first CTA, separated by `-`): the report shows which outcome
fails.

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
| `a generic write of shared memory in a warp-specialized kernel`, `atom.shared... in a warp-specialized kernel` | a TLX kernel with a switch-loop dispatch (state bytes) that writes shared memory through `cvta.shared` or a shared atomic (a direct dispatch refuses neither) |
| `mbarrier.inval of ... ends no private synchronous cycle ...` | an inval whose slot's events since its init are not exact arrivals and one wait of parity 0 in one statement list |
| `no op table row for ...` (error) | an instruction without a row: `ir_ops_ptx.md` and `ops.py` need one |
| `... that several lanes of a warp may execute, compiled by ptxas X for sm_N: its issuing is measured for ...` | a multi-lane TMA or bulk copy from a compiler or target whose issuing was not measured |
| `a global access whose address has no tensor provenance` | an address from a loaded pointer or an integer (through an unnamed pointer parameter, whose value is Data: the slice soundness message; through a parameter's own address: `a global access through the address of the parameter ... (only ld.param reads it)`; through a descriptor's or the scratch's: `a global access through ... (only TMA instructions)`) |
| `... (sub-element access)`, `byte offset ... not provably a multiple ...` | an access that is not whole elements of its tensor |
| `mbarrier address is not a shared-memory symbol plus an offset`, an unresolved array or index | a barrier the arrays cannot place |
| `TMA load completing on an mbarrier of another CTA (a mapa result) ...`, `... in a cluster of N CTAs without .multicast::cluster::all` | a cluster idiom outside the supported ones (a load completing on neither the issuing CTA's nor its pair leader's barrier; a CLC response only the asking CTA receives) |
| `... with only part of warp(s) [...] active (undefined in PTX)`, `... under a condition that differs between the lanes of a warp` | a full-mask collective under a partial warp; a `.aligned` instruction under a mask that differs between the lanes of a warp |
| `brx.idx at line N over K targets: its index is not provably in [0, K) ...`, `brx.idx at line N: the names of its compare chain ... are taken` | a switch whose index the interpreter cannot bound, or whose compare chain's names the kernel already uses ("Control flow") |
| `a shared write that may overwrite a warp-specialize capture another role read ...`, `the warp-specialize capture at byte ... lies in an mbarrier array ...` | a shared write that may touch a known capture after a worker read one; a capture a worker read inside an mbarrier array ("Threads and roles") |
| `... that several lanes of a warp may execute: its issuing is not measured ...` | a multicast or cta_group::2 TMA load or a shared::cluster bulk copy or reduce that more than one lane of a warp may issue |
| `CLC with a preferred cluster geometry ...`, `... required cluster (cuFuncSetAttribute REQUIRED_CLUSTER) is ...`, `cuda.launch_ex: a preferred cluster ...`, `launches differ in their (preferred) cluster dims ...`, `preferred cluster dims ... without a regular cluster ...`, `... are no multiple of the regular ...`, `... do not divide the grid of input(s) ...` | a cluster geometry LKV does not check ("Launch records", "Clusters, tcgen05, CLC") |
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

Each variant gives Lean more than one KIR, each checked on its own: the P0 KIR, the P0-6
barrier pass's (both families, whatever the translation), the P1 pass's (CuTe P1-1,
TLX/Triton P1-2) and, for a TLX launch with a preferred cluster, each of them again in
the all-preferred geometry.

## Limitations

* Known gaps: the translation refuses some kernels and constructs that the legacy
  translations handle ("Refusals"); `--frontend legacy` runs the TTGIR translation of a
  TLX/Triton kernel, and a CuTe kernel stays on the CuTe DSL MLIR unless `--frontend ptx`.
* A verdict covers the case inputs, the compiled variants they select and `max_iters`.
* Below PTX nothing is checked (ptxas, SASS), except the TMA issuing rules measured on
  hardware (`ir_ops_ptx.md`).
* Shared memory, TMEM, MMA math, named barriers and fences are outside the P0 slice:
  named barriers are P0-6's (its own pass, both families) and `named_bar` events of both
  P1 passes; TMEM accesses and `tcgen05.wait` are P1-1 events (CuTe); shared-memory
  writes and bulk stores are P1-2 events (TLX/Triton); fences and TMEM lifetime are not
  checked.
* Preferred cluster geometries (TLX/Triton) are checked all-preferred and all-regular (P0-6
  and P1-2 too); P0-3 of a launch that mixes them is covered only where both geometries
  write the same elements per preferred-size group of CTAs, which is not checked, and
  P1-2's tensormap check across clusters only between clusters of one geometry ("Clusters,
  tcgen05, CLC"). A CuTe launch with a preferred cluster is refused (gap), as is a TLX
  launch whose function's required cluster (cuFuncSetAttribute) differs from its cluster
  dims, their P0-6 KIRs too. The all-preferred KIRs of TLX launches that break a rule
  CUDA states for the preferred dims (no regular cluster, a preferred x or y that is no
  multiple of the regular one or another z, dims that do not divide the grid) are
  refused (gap): what the device runs then is not modeled.
* Refused by design: CuTe CLC, P1-1 on TLX, TMEM addresses relative
  to the allocation, multicast bulk copies, multicast or cta_group::2 TMA loads and
  shared::cluster bulk copies and reduces that several lanes of a warp may execute (issuing not
  measured, `ir_ops_ptx.md`), im2col TMA, packed sub-byte TLX descriptors, descriptors
  edited in global memory, `mbarrier.inval` outside
  a private synchronous cycle, carry-flag instructions under a predicate, generic writes
  of shared memory and shared atomics in TLX kernels with a switch-loop dispatch, CLC
  with a preferred cluster geometry (the all-preferred KIRs), the control-flow shapes
  above, and anything whose value would have to come from Data.
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
