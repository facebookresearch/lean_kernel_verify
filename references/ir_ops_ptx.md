# PTX op table (the PTX translation)

One meaning per PTX instruction, identical for every kernel and for both families
(TLX/Triton and CuTe DSL). It is the op table of the opt-in PTX translation
(`lkv.py run --frontend ptx`, `frontend_ptx.md`), where it takes the place of the IR op
tables `ir_ops.md` and `ir_ops_cute.md` (trust base item 2, `design.md`). The rows follow the PTX ISA (sections named, not table
numbers); the only compiler behavior they encode is how ptxas issues a TMA or bulk copy
that several lanes of a warp execute ("Issuing threads", measured on hardware, gated by
compiler version and target). Two more trust items are not ISA facts: the Triton
warp-specialize dispatch protocol (`frontend_ptx.md`, "Threads and roles") and the
cta_group::2 peer-bit address pattern ("Thread barriers and clusters" below). How the
translator around the table works (control flow, roles, slicing, assembly per family):
`frontend_ptx.md`.

Executable form: `frontend/ptx/ops.py` (normalization and classes), `interp.py` (values,
collectives, special registers), `mem.py` (global memory, parameters, mbarriers),
`tma.py` (TMA, bulk copies, tensormaps), `sync.py` (tcgen05 in P0, CLC), `cluster.py`
(cluster scope) and `tmem.py` (the P1 pass); keep them in sync with this file.
`python3 -I frontend/ptx/coverage.py PTX...` classifies every instruction of PTX files and
lists the mnemonics without a row.

## Normalization

The spellings of LLVM (Triton), NVVM 20 (CuTe DSL 4.4.2) and NVVM 23 (4.6.1) meet on one
mnemonic: integer and bit types (`.b32`, `.u64`, `.pred`, `.b128`), vector widths
(`.v4`), repetition counts (`.x16`), cache hints (`.L2::cache_hint`) and spelled-out
defaults (`bra.uni`, the `.tile` load mode, setp's comparison) are dropped; an
arithmetic, comparison, move or conversion op on any floating-point type is `<op>.f`;
`.shared::cta` is `.shared`, `.param::entry` is `.param`; mbarrier's default
`.sem.scope` is dropped (`.release.cta` for arrive, `.acquire.cta` for
try_wait/test_wait, `.relaxed.cta` for expect_tx/complete_tx); tcgen05.mma's
`.scale_vec::1X/2X/4X` is `.block32/.block32/.block16`. So
`mbarrier.arrive.release.cta.shared::cta.b64`, `mbarrier.arrive.shared.b64` and
`mbarrier.arrive.shared::cta.b64` are all `mbarrier.arrive.shared`.

## Classes

Every normalized mnemonic has exactly one class. A mnemonic without a row is an `error`
(`no op table row for ...`: the table grows by deciding a row, never by a default):

* **event**: becomes KIR events or feeds an event model (below);
* **value**: an integer value the translator evaluates exactly;
* **control**: `bra` (guarded: the branch of an if or loop), `brx.idx` (only in the
  Triton warp-specialize dispatch), `ret` (exactly one per kernel);
* **drop**: affects no spec; its results are Data, and Data that reaches a root (an event
  operand, a branch condition, an address, a predicate, a barrier field, a work anchor)
  is the slice soundness refusal naming its origin, so a wrong drop cannot hide an event;
* **refuse**: the variant is refused (a gap) with the row's reason.

Notation. `mask` is the per-thread condition of the enclosing thread-varying branches
and of the instruction's own `@p` guard; its thread-uniform part becomes a KIR `if`
around the event. An instruction's issuing threads are the threads of the role with
`mask` true. LANE, WARP and the elect symbols (true on the lane an `elect.sync` picks)
are the thread model (`frontend_ptx.md`, "Threads and roles"). Per issuing thread means:
one event if one thread issues it; else the TLX family emits one event whose offsets
range over the issuing threads, and the CuTe family a KIR `for` over the role's threads
with the thread's mask as an `if` (`frontend_ptx.md`, "Family conventions").

## Values

| Instruction | Value |
|---|---|
| `add sub mul.lo div rem min max neg and or xor not shl shr` | KIR `bin` (div/rem `sdiv/udiv/srem/urem` by type, shr `ashr/lshr`); a divisor under a mask is replaced where the mask is false |
| `mad.lo`, `mul.wide`, `mad.wide`, `mul.hi` | mul + add; ext + mul (+ add); 32-bit mul.hi: 64-bit product, `lshr 32`, trunc (64-bit operands: refused) |
| `popc clz brev bfe bfi prmt lop3 shf.{l,r}.{wrap,clamp}` | exact expansions; bfe/bfi positions, prmt selector, lop3 truth table and shf amount must be constants (else refused) |
| `setp` (with `.and/.or/.xor`), `selp`, `mov`, `cvt` (integer), `cvta.global` | `cmp` (and the combine), `select`, copy, `ext`/`trunc` (saturating or rounding integer cvt: refused), identity |
| 8- and 16-bit integers | their zero-extended i32 (signed ops on the sign extension) |
| `ld.param` | a kernel parameter: a scalar is the KIR param of the parameter (or struct field) at its width, extended by the load type; a pointer is its tensor's base address (`frontend_ptx.md`, "Memory"); the bytes of a descriptor parameter: refused |
| `cvta.param` | the address of a descriptor parameter (only TMA instructions may use it) |
| `%tid.x`, `%tid.y/z`, `%ntid` | `32*WARP + LANE`, 0 (the block must be (32k, 1, 1)), the launch's block |
| `%laneid`, `%lanemask_{eq,lt,le,gt,ge}` | LANE and its masks |
| `%ctaid`, `%nctaid` | `pid`, `nprog` |
| `%cluster_ctarank`, `%cluster_nctarank`, `%clusterid`, `%nclusterid`, `%cluster_ctaid`, `%cluster_nctaid`, `%is_explicit_cluster` | from the LaunchRecord's cluster dims c: `x % cx + (y % cy) cx + (z % cz) cx cy` over `pid`, its product, `pid / c`, `nprog / c`, `pid % c`, `c`, `c != (1, 1, 1)` |
| `%warpid` | refused (ISA: it may change during execution) |
| other special registers (`%clock`, ...) | Data |
| `elect.sync d\|p, -1` | p: a fresh elect symbol (one lane of each warp, which one the ISA does not say); d: Data |
| `shfl.sync.{idx,up,down,bfly} d\|p, a, b, c, -1` | lane l gets `a` at source lane j(l) with the ISA's rule (`bval = b[4:0]`, `segmask = c[12:8]`, minLane/maxLane), its own `a` where j(l) is out of range, p telling which: a table over the lanes |
| `vote.sync.{ballot,any,all,uni} d, p, -1` | from p at each lane; on Data, a value uniform per warp (a branch on it keeps each warp together) |
| `redux.sync` (integer) | refused (not translated yet); on floats: drop |
| `mapa.shared::cluster d, a, r` | the shared address `a` in the CTA of rank r ("Thread barriers and clusters") |

The full-mask collectives (`elect/shfl/vote.sync` with member mask -1; another mask is
refused) need every lane of each executing warp active (undefined in PTX otherwise):
refused when that is statically false; asserted per warp in the KIR (a FAIL on the input
where it does not hold) when it depends on runtime values. An event under an elect symbol
must have lane-independent operands and no other lane-dependent conjunct in its mask; a
value that depends on which lane `elect.sync` picks is never taken as uniform, and is
refused where an event or a per-thread slot needs it.

## Global memory

| Instruction | KIR | Rule |
|---|---|---|
| `ld.global`, generic `ld` on a global address | `gload(T, offsets, mask)` per issuing thread | `.v2/.v4`: consecutive elements (a `range`). An integer load whose element is the register's width (32 or 64 bits) also yields the value `load(T, [index])`: T becomes metadata if the value reaches a root; where the mask is false the index is 0 and the register keeps its old value |
| `st.global`, generic `st` on a global address | `gstore` per issuing thread | the stored value is data |
| `atom.global.{add,min,max,inc,dec,and,or,xor,cas}`, `red.global.*` | `gatomic` per issuing thread | `atom.exch` is a `gstore`; atom results are Data |
| `cp.async.{ca,cg}.shared.global [d], [s], n{, src-size \| ignore-src}` | `gload` of n bytes at s per issuing thread | constant n; a src-size of 0 (or a true ignore-src) reads nothing, any other src-size must be n (a KIR assert); commit/wait groups are drops |
| generic `ld`/`st` on a shared address | none | Data |

An address is a tensor's base (a pointer parameter, or a pointer field of a struct
parameter, named by the host adapter) plus a byte offset, through adds, subtracts, a
select of two offsets of one base and per-lane tables of one base; a select between two
tensors takes the arm the role's condition picks (CuTe: a condition uniform per warp
splits the role). Any other address is refused. The element offset is the byte offset
divided by the element size, which must be provably exact (known trailing zeros,
factorization), else the access is refused (a sub-element access).

## mbarriers

| Instruction | KIR | Issuing |
|---|---|---|
| `mbarrier.init.shared [a], n` | `bar_init(bar(a), n)` | once per role; n a host value |
| `mbarrier.arrive.shared [a]{, n}` | `arrive(bar(a))` | per issuing thread, counted (below); n a positive constant |
| `mbarrier.arrive.expect_tx.shared [a], b` | `expect_tx(bar(a), b)` + `arrive(bar(a))` | as arrive |
| `mbarrier.expect_tx.shared [a], b` | `expect_tx(bar(a), b)` | per issuing thread |
| the wait pseudo-instruction | `wait(bar(a), parity)` | once per role |
| `mbarrier.arrive(.expect_tx).shared::cluster`, `mbarrier.complete_tx` | "Thread barriers and clusters" | |
| `mbarrier.try_wait`, `mbarrier.test_wait` (any other use) | refused | they may return false spuriously; only the whole loop waits |
| `mbarrier.arrive.relaxed`, `mbarrier.inval`, `cp.async.mbarrier.arrive.noinc` | refused | not modeled |

`bar(a)`: `a` must peel to a shared symbol plus a byte offset that is the same for every
thread of the role (CuTe: one uniform per warp splits the role); its array and index:
`frontend_ptx.md`, "mbarriers". The wait pseudo-instruction is a loop that is exactly
`L: mbarrier.try_wait.parity p, [a], ph{, hint}; @!p bra L` or
`L: ...; @p bra D; bra L; D:` (inline asm, also inside an MMA asm block), with p read
nowhere else: try_wait may return false spuriously, so only the whole loop is a blocking
wait (ISA mbarrier.test_wait / mbarrier.try_wait). Arrivals: TLX `threads_per_issue` is
1 and an arrive's `n` is the issuing threads times the count; CuTe `threads_per_issue`
is the gcd of the role's arrive counts and `n` the count over it.

## TMA and bulk copies

| Instruction | KIR (TLX / CuTe) |
|---|---|
| `cp.async.bulk.tensor.{1-5}d.shared{::cluster}.global.mbarrier::complete_tx::bytes{.multicast::cluster}{.cta_group::2} [dst], [tmap, {c}], [mbar]` | `tma_load(desc, coords, bar)` / plus the footprint `gload` on the base tensor |
| `cp.async.bulk.tensor.{1-5}d.global.shared.bulk_group [tmap, {c}], [src]` | `tma_store` / footprint `gstore` |
| `cp.reduce.async.bulk.tensor.{1-5}d.global.shared.<op>.bulk_group` | `tma_reduce` / footprint `gatomic` |
| `cp.async.bulk.shared{::cluster}.global.mbarrier::complete_tx::bytes [dst], [g], n, [mbar]` | `gload` of the n bytes + `tma_load` of the 1-D descriptor `bulk_<T>_<n>` on mbar |
| `cp.async.bulk.global.shared.bulk_group [g], [src], n` | `gstore` of the n bytes |
| `cp.reduce.async.bulk.global.shared.bulk_group.<op> [g], [src], n` | `gatomic` of the n bytes |
| `cp.async.bulk.shared::cluster.shared.mbarrier::complete_tx::bytes` | `complete_tx` in another CTA ("Thread barriers and clusters") |
| multicast bulk g2s, im2col / gather4 / scatter4 modes | refused |
| `tensormap.replace.*`, `tensormap.cp_fenceproxy` | an in-kernel descriptor (`frontend_ptx.md`, "TMA descriptors") |
| `prefetch.tensormap`, commit/wait groups | drop |

Descriptors, coordinates and footprints: `frontend_ptx.md`, "TMA descriptors". PTX lists
coordinates innermost first. A footprint is exactly the elements the copy moves: element
`off + sum (c + i) * stride` for each box index i, masked to `0 <= c + i < extent` in
every dim (TMA clips there). Transaction bytes: the box times the element bytes,
out-of-bounds fill included, completing on the mbarrier.

**Issuing threads.** A copy executes on the threads the PTX's own predicates, branches and
`elect.sync` admit. One that at most one lane of each warp executes (statically: under
`elect.sync`, a lane test) is issued once per executing thread. One that several lanes of
a warp may execute is issued the way ptxas lowers it, measured on sm_100a and sm_103a
hardware (a GB200, a GB300) for every compiler the families use
(`tests/ptx/test_gpu_tma_issue.py`: 12.8.93, sm_100a only, 12.9.83 = CuTe DSL 4.4.2's
embedded compiler, 13.0.88, 13.3.27 = DSL 4.6.1's, 13.3.33 = Triton's ptxas), which all
issue them alike on both:

* TMA load and bulk g2s: once per executing lane, with that lane's operands, also where
  the lanes' operands are equal (an elect loop issues each lane's copy); the lanes'
  transaction bytes add up.
* TMA reduce and bulk reduce: once per executing lane (every add lands once per lane).
* TMA store: in each warp, once per distinct run-time value of (descriptor address,
  coordinates, shared source address) among the executing lanes, by the lowest lane
  holding it (ptxas's waterfall loop: one `UTMASTG` for warp-uniform operands).
* Bulk s2g: the same, keyed on (global destination, shared source).

The L2 cache-policy operand is never part of a key. Gates: such a copy is translated only
if the variant's compiler is in `tma.ISSUE_PTXAS` (CuTe: the version its cubin's
`.note.nv.tkinfo` records; TLX: the `ptxas --version` the dump records in each
LaunchRecord, refused for a dump without it) and the PTX target in `tma.ISSUE_TARGETS`
(sm_100 and sm_103, measured on a GB200 and a GB300), else it is refused: another
compiler, or the same one on another target, may lower it otherwise. A new compiler
release needs that test run and its version added. The mbarrier of a multi-lane load must be the same for every lane. Not measured,
so translated only where at most one lane of each warp executes them: multicast and
cta_group::2 TMA loads and shared::cluster bulk copies.

**Packed sub-byte formats** (CUtensorMap `16U4_ALIGN8B`, `16U4_ALIGN16B`,
`16U6_ALIGN16B`; measured on GB200 and GB300, `tests/ptx/test_gpu_tma_subbyte.py`): the
KIR descriptor counts dim 0 in bytes (its extent, box and stride; a coordinate is converted,
with an assert that it is byte aligned), and the transaction bytes are the packed global
bytes, also for the two 16-byte-aligned formats whose shared-memory copy is wider, into
each CTA of a multicast and twice on the pair leader's mbarrier with cta_group::2. The
hardware traps on a dim-0 coordinate that is not a multiple of 32 elements
(16U4_ALIGN8B) or 128 (the 16-byte-aligned formats), on any reduce and on a
16U4_ALIGN16B store: each is a KIR assert per issuing thread (a FAIL where it is
reached). A packed TLX host or in-kernel descriptor is refused. A byte format traps too
at a dim-0 coordinate off its 16-byte grid (measured); that is not asserted: such a copy
is modeled as performed.

## Thread barriers and clusters

| Instruction | P0 | P1 |
|---|---|---|
| `bar.sync`, `bar.arrive`, `barrier.sync{.aligned}`, `barrier.arrive{.aligned}` `id{, count}` | no event; the aligned forms need a mask uniform per warp | `named_bar` of the whole role: id and count constants (one uniform per warp but not per role splits the role), count a positive multiple of 32, absent: the block's threads |
| `bar.warp.sync` | drop | drop |
| `barrier.cluster.arrive{.relaxed}{.aligned}` / `.wait{.aligned}` (CuTe) | `arrive` of every thread of the role / `wait` on `cluster_barrier` | as P0 |

Clusters (CuTe; a TLX kernel launched with clusters is refused, `frontend_ptx.md`):

* `mapa.shared::cluster d, a, r` makes a remote address (a in the CTA of rank r); any
  other arithmetic on it makes it Data. An `mbarrier.arrive{.expect_tx}.shared::cluster`
  through one gets the event's `cta` (uniform over the role; one uniform per warp splits
  the role); through a local address it is the issuing CTA's slot.
* The peer bit: `a & 0xFEFFFFF8` (or a mask LLVM narrowed by bits it knows are zero) of a
  shared address clears bit 24 of its shared::cluster window address and names the even
  CTA of the issuing CTA's pair (CUTLASS's peer-bit mask; the ISA does not document the
  bit: a trust item). It is accepted only as the mbarrier of a `.cta_group::2` TMA load,
  which then has KIR `leader`; a cta_group::2 load on a local mbarrier completes on the
  issuing CTA's own slot.
* A multicast TMA load's mask operand is the `tma_load`'s `cta_mask` (its box lands in
  every CTA of the mask); a multicast `tcgen05.commit`'s is the `commit_arrive`'s.
* `cp.async.bulk.shared::cluster.shared...complete_tx::bytes [dst], [src], n, [mbar]`
  with dst and mbar mapa'd into one CTA: `complete_tx(bar, n, cta)` per issuing thread
  (the data is not modeled). `mbarrier.complete_tx` and any other access to another CTA:
  refused.
* `barrier.cluster`: an arrive of every thread on `cluster_barrier`, a frontend-made
  barrier of one slot in the CTA of rank 0 counting one arrival per thread of the cluster;
  the role's j-th wait waits for phase j (`cluster_barrier_parity`, flipped after each
  wait). Reached by part of a role, or under a lane condition that depends on runtime
  values: refused (threads that exit early count as arrived in PTX; not modeled).
* A load must write the issuing CTA's shared memory unless it multicasts; cta_group::2
  instructions need a cluster of an even number of CTAs, `cta`, `cta_mask` and
  `complete_tx` a cluster of several CTAs.

## tcgen05

P0: `tcgen05.alloc`, `dealloc`, `relinquish_alloc_permit`, `ld`, `st`, `wait::ld/st`,
`cp`, `shift`, `mma` emit nothing (the P0 specs do not see TMEM); their `.sync.aligned`
forms need a mask uniform per warp, and tcgen05.ld results are Data. `tcgen05.commit`
(`.cta_group::{1,2}.mbarrier::arrive::one`, optionally `.multicast::cluster` with a mask
m) `[a]` is a `commit_arrive(bar(a))` per issuing thread (it arrives once the issuing
thread's prior async tcgen05 operations complete), with `cta_mask` m when multicast.
`tcgen05.fence::*` is a drop.

P1 (CuTe only; `frontend_ptx.md`, "P1 pass"): the same instructions also give these
events.

| Instruction | P1 event | Issuing |
|---|---|---|
| `tcgen05.ld/st.sync.aligned.32x32b.xN [taddr]` | `tmem` ld/st, `data` lanes [L, L+32) x columns [C, C+N) | one per issuing warp: every lane active, taddr the same in all lanes; other shapes (`16x64b`, `16x128b`, `16x256b`, `16x32bx2`): refused (R5) |
| `tcgen05.cp.cta_group::{1,2}.{128x256b, 128x128b, 64x128b.warpx2::01_23/02_13, 32x128b.warpx4} [taddr], desc` | `tmem` cp, `data` write, lanes [0, 128) x 8 columns (256b) or 4 | one thread of a one-warp role (an elected lane or one fixed thread) |
| `tcgen05.shift.cta_group::{1,2}.down [taddr]` | as a cp writing lanes [0, 128) x [C, C+8) (an over-approximation) | as cp |
| `tcgen05.mma.cta_group::{1,2}.kind::* [d], a, b, idesc, ...` | `tmem` mma: `d` write 128 lanes x N columns (N / 2 for M = 128 with cta_group::2); `a` read 8 columns when A is in TMEM; block-scaled kinds: `sf` reads SFA 4 columns, SFB 4 per 128 rows of N | as cp; M and N from the known bits of idesc (N >> 3 at bits 17-22; M >> 4 at bits 24-28, block-scaled M >> 7 at bits 27-28) |
| `tcgen05.wait::ld` / `::st` | `tmem_wait` of the whole role | every thread of the role |
| `tcgen05.alloc` | the KIR `tmem` allocation | one per issuing warp, once per kernel |
| `tcgen05.dealloc`, `relinquish_alloc_permit`, `fence::*` | none (lifetime and fences are not checked) | |

A cta_group::2 cp or mma accesses the same rectangles in both CTAs of the pair
(`cta_mask` `3 << (rank & ~1)`). The SF widths are those of the CuTe DSL's TMEM SF
layouts, which are wider than the ISA's (the max of the two is taken). An SF address's
bits 30-31 (the byte the descriptor's SF id selects) are dropped: the column is what
counts.

## Cluster launch control (TLX only)

| Instruction | KIR |
|---|---|
| `clusterlaunchcontrol.try_cancel.async.shared...complete_tx::bytes{.multicast::cluster::all} [resp], [mbar]` | `tma_load(__clc_response, [0], bar(mbar))` per issuing thread: the 16-byte response completes 16 transaction bytes |
| `ld.shared.b128 r, [resp]` | r is the response token: read from a try_cancel's response address, after a wait of the reading role on that try_cancel's mbarrier |
| `clusterlaunchcontrol.query_cancel.is_canceled p, r` | one query of the whole role: `let clc<k> = load(__clc_schedule, [pid.x, __clc_k])`, `__clc_k += 1`; p is `clc<k> != -1` |
| `clusterlaunchcontrol.query_cancel.get_first_ctaid.v4 {x, y, z, _}, r` (predicated by p) | x = `clc<k>`, y, z = `clc<k> == -1 ? -1 : 0` |

Any other use of a response is refused; so is CLC in a CuTe kernel. The schedules the
tables hold: `frontend_ttgir.md`, "Cluster Launch Control".

## Drop and refuse

Drop (no spec effect; results are Data): floating-point arithmetic, comparison, moves and
conversions (`<op>.f`), `ld.shared`/`st.shared` (except the tensormap staging buffer's
zeroing and the CLC response read), `ldmatrix`, `stmatrix`, `fence.*`, `membar`,
`tcgen05.fence`, `griddepcontrol`, `setmaxnreg` (no consistency check across a warpgroup:
real kernels run it at different points per warp), `prefetch`, `bar.warp.sync`, the
commit/wait groups of `cp.async` and `cp.async.bulk`, `st.param` (call staging), `trap`
(the run is evaluated as if it were not reached), `call vprintf` and calls of libdevice
(`__nv_*`).

Refuse (a gap naming the instruction): `try_wait`/`test_wait` outside a wait loop, relaxed
arrives, `mbarrier.inval`, `cp.async.mbarrier.arrive.noinc`, TMA im2col/gather4/scatter4,
`%warpid`, and a `call` of anything but `__assertfail` (a KIR `assert false` where it is
reached), `vprintf` and libdevice. Instructions without a row (e.g. `ld.local`) are
errors, as above.
