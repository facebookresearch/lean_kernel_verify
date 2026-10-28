# Which GPU-kernel bugs to verify with Lean, and why

This is the design rationale for the skill's spec set. It answers three questions: which
bug classes Lean should own, which ones tests and profilers already cover, and which
classes are worth formalizing first. Section 2 backs the ranking with public bug reports
from open-source GPU kernel libraries and with the PTX ISA's correctness rules.

## 1. Division of labor

| Dimension | Typical bugs | Best tool | Why |
|---|---|---|---|
| Synchronization protocol | hang, missed wait, wrong parity, unbalanced arrivals, early phase completion | Lean over the extracted barrier program | Fails only for some configs, counts or timings; tests pass on the common path |
| Address and index safety | OOB read/write, i32 wraparound, cross-batch spill, dropped tail | Lean over footprints with exact `BitVec` integer semantics | Triggers at shapes tests rarely use (unaligned, empty, > 2^31 elements) |
| Output coverage | element never written, written twice, written by two CTAs | Lean over store footprints | `torch.empty` hides holes; cosine metrics hide small regions |
| Numerical accuracy | rounding, approximation error | Tests against an FP64 reference | Cheap and direct |
| Performance | bank conflicts, occupancy, uncoalesced access | Profiler (NCU) | Needs hardware counters |
| Loud config failures | compile error, smem over budget, launch failure | CI compile sweep over every host-accepted config | Fails immediately; enumeration is cheaper than proof |

Two caveats change the naive split:

* Reasoning over unbounded integers (e.g. `omega`) misses overflow bugs. Integer
  semantics must be the exact fixed-width (`BitVec 32`/`BitVec 64`) ones; this skill
  evaluates them exactly on concrete inputs.
* Cosine/SQNR checks are scale-invariant. A scale applied twice, or a wrong-stage read
  that only shifts magnitudes, passes them, so scale bookkeeping is not fully "test-owned".

## 2. Evidence

Public reports of bugs in open-source GPU kernel libraries: examples for each P0 spec, and
a few for classes the specs do not cover (status as of 2026-10-26). The Spec column names
the LKV spec for the bug's class. LKV itself checks TLX/Triton and CuTe DSL kernels, so it
would not run on the CUDA C++ kernels among these (FlashAttention 2, CUTLASS C++); the
rows show the classes.

| # | Report | What went wrong | How it showed | Spec |
|---|---|---|---|---|
| E1 | PyTorch [#190857](https://github.com/pytorch/pytorch/issues/190857) (closed) | FlexAttention's backward Triton kernel computes the K/V pointer offsets in int32 | illegal memory access once the flattened K/V address exceeds `INT32_MAX` | P0-1 |
| E2 | FlashAttention [#2755](https://github.com/Dao-AILab/flash-attention/pull/2755) (merged) | sparse-MLA backward on SM100: the dK/dV scatter-reduce epilogues add atomically at rows read from the top-k index tensor with no validity check, while `-1` marks an invalid slot | atomic adds at row -1 | P0-1 |
| E3 | FlashAttention [2865a89](https://github.com/Dao-AILab/flash-attention/commit/2865a891a90c01f21d9d1390c063b4fbf2331fd1) (fix) | sparse-MLA backward preprocess on SM100: `lse_log2` is stored for every row of the tile-rounded sequence; without padded offsets the tail lands on the next sequence's first rows, or past the end of the buffer for the last sequence | intermittent wrong gradients (the CTA that stores last wins), only with 64 Q heads and an odd token count | P0-1, P0-3 |
| E4 | FlashAttention [#2842](https://github.com/Dao-AILab/flash-attention/issues/2842) (open; fix proposed in [#3004](https://github.com/Dao-AILab/flash-attention/pull/3004)) | FlashAttention 2 split-KV kernel, paged KV and varlen inputs: a query that sees no key takes an early exit that writes its LSE at the dense `(b, h, seqlen_q)` offset | the query's own LSE slot is never written, and the infinity lands in another sequence's slot or past the end of `softmax_lse`; `out` is right and no error is raised | P0-1, P0-3 |
| E5 | FlashAttention [#2760](https://github.com/Dao-AILab/flash-attention/issues/2760), fixed by [#2761](https://github.com/Dao-AILab/flash-attention/pull/2761) | SM100 forward with varlen + block-sparse + SplitKV: the load and MMA warps pass the packed split index to the block-sparse helpers (an empty block range), the softmax and correction warps the unpacked one (a non-empty range), so the warps disagree on whether a tile has work | every such forward hangs | P0-2 |
| E6 | FlashAttention [#2705](https://github.com/Dao-AILab/flash-attention/pull/2705) (merged) | rebuilding a tile scheduler from its MLIR values resets its first-tile flag, so the first tile becomes valid again and is repeated | the producer/consumer pipeline can deadlock | P0-2 |
| E7 | CUTLASS [#3782](https://github.com/NVIDIA/cutlass/issues/3782) (open; fix proposed in [#3781](https://github.com/NVIDIA/cutlass/pull/3781)) | SM100 mixed TMA/cp.async GEMM: the scheduler pipeline's consumer arrival count leaves out the epilogue-load warp, which also consumes every scheduler response when C is loaded, so each stage gets more arrivals per phase than it expects | hangs when C is loaded (10 of 10 runs with beta = 1) | P0-2 |
| E8 | CUTLASS [#3783](https://github.com/NVIDIA/cutlass/issues/3783) (open) | TMA epilogue with a null `beta_ptr`, a nonzero `dBeta` and a nonzero scalar beta, which `can_implement` accepts: the load warp decides from the pointer that C is not needed, the epilogue decides after the scalar update that it is and waits for it | hang (3 of 3 runs) | P0-2 |
| E9 | CUTLASS [4.3.0 changelog](https://docs.nvidia.com/cutlass/latest/CHANGELOG.html) (fixed) | "a dead hang issue caused by early return warp" | hang | P0-2 |
| E10 | CUTLASS [#2879](https://github.com/NVIDIA/cutlass/issues/2879), [#3391](https://github.com/NVIDIA/cutlass/issues/3391) | the CuTe DSL elects the issuing lane of a TMA copy inside `cute.copy`, so a copy called from one thread hung (#2879); DSL 4.6.0 moved the election of bulk copies into `cute.copy` too, and kernels that still elected around them hung (#3391) until 4.6.2 reverted the change | hangs that depend on the DSL release | issue counts (trust base item 2); P0-2, P0-4 |
| E11 | FlashAttention [#3075](https://github.com/Dao-AILab/flash-attention/issues/3075) (open) | FlashAttention 2 KV-cache attention with `cache_leftpad` and no `cache_seqlens`: the host selects the `Is_even_MN` kernel, which does not mask the last key block, from `seqlen_k` and `seqlen_q`, but a sequence's own length `seqlen_k - cache_leftpad[b]` need not be a multiple of the key block | keys past each sequence's end (the next sequence's rows, or past the allocation for the last sequence) enter its softmax | P0-5 |
| E12 | FlashAttention [1c41d2b](https://github.com/Dao-AILab/flash-attention/commit/1c41d2b0e5021907374e9509250ad7a22e5693bd) (fix, 2023) | FlashAttention 2 backward: "Fix race condition in bwd (overwriting sK)", a shared-memory K tile overwritten in a race | not stated | not covered: buffer reuse (P1) |
| E13 | CUTLASS [#2760](https://github.com/NVIDIA/cutlass/issues/2760), fixed by [#2719](https://github.com/NVIDIA/cutlass/pull/2719) | SM90 array and grouped GEMMs never called `wait_on_dependent_grids()`, so under programmatic dependent launch (PDL) they could read inputs before the previous kernel's writes were visible | rare wrong results | not covered: PDL ordering (P2) |
| E14 | FlashAttention [#2906](https://github.com/Dao-AILab/flash-attention/pull/2906) (merged) | SM100 forward: a block-sparse Q tile with no live KV block never writes its O accumulator in TMEM (`tcgen05.alloc` does not initialize it), and the epilogue zeroed the leftover by scaling it by 0.0 | zeros most of the time, NaN when an earlier kernel left an infinity in TMEM | not covered: uninitialized reads, NaN propagation (P2) |

The specs also encode the correctness rules of the
[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) for mbarrier (phases,
arrival counts, transaction bytes), bulk copies and TMA, and `tcgen05` (`tcgen05.wait`,
`tcgen05.commit`).

Why ordinary tests miss bugs like these:

* the trigger is an input or configuration tests rarely use: a feature combination (E5), a
  head count with an odd token count (E3), host arguments that `can_implement` accepts but
  no example passes (E8), more than 2^31 addressable elements (E1), sentinel indices (E2),
  a query that sees no key (E4), sequences that end inside a block (E11); test shapes are
  typically aligned, non-empty, or one tile per CTA, so persistent phase drift never shows
  up;
* the failure is intermittent or depends on the environment: a race between CTAs (E3) or
  between kernels (E13), the DSL release (E10), leftover memory contents (E14);
* the output looks plausible: `out` is right and no error is raised (E4), the tile is zero
  most of the time (E14); outputs allocated with `torch.empty` keep plausible old values
  where a write is missing, and tolerance or cosine checks hide small wrong regions;
* the autotuner picks a config without the bug, and gradients scaled below `atol` pass.

## 3. Spec priorities

Selection criteria: (1) a decision procedure closes it, so the proof is reliable;
(2) ordinary tests and profilers miss it; (3) it can be extracted mechanically from
compiler IR, with no hand-written model. Every spec is required to hold on degenerate
inputs: length 0, a CTA with no work, unaligned tiles, more tiles than SMs.

### P0 (implemented by this skill)

| P0 | Spec | Public examples (section 2) | Proof method |
|---|---|---|---|
| 1 | **Global accesses in bounds**, including jagged segment bounds (writes outside the work item's segment; reads and reductions there are notes), TMA box footprint, exact i32/i64 integer semantics, metadata (tile table, offsets) reads inside their tensor | E1-E4 | footprints vs allocation and segment; exact `BitVec` evaluation records every wrap on an address path |
| 2 | **Role trip-count agreement**: no wait needs a completion the kernel never produces or waits in a cross-role cycle, the arrivals of every waited phase split into one fixed share per arriving role that sums to the count (or follow one initialization arrival), a role waiting on a TMA barrier array has a wait, arrival or transaction event on every completing slot that gets data (a `bar_init` does not count), including zero-trip | E5-E9 | count checks and an optimistic progress pass over the extracted per-role event sequences (phase ledger) |
| 3 | **Output coverage exactly once + inter-CTA write disjointness** | E3, E4 | interval cover/disjointness over all store footprints |
| 4 | **mbarrier protocol**: wait parity matches the intended completion (stale repeats, a data consumer starting at parity 1), expect_tx bytes equal the bytes landing in each waited phase, no TMA bytes land in a phase nobody waits for, expect_tx issued before the loads it accounts for | E10 (issue counts), the PTX ISA's mbarrier rules | per-barrier phase ledger over the event sequences |
| 5 | **Mask-elision soundness**: an input for which the host selects a mask-elided variant (a flag the case lists in `elision_flags`, e.g. FULL_TILES) satisfies P0-1 and P0-3 on that variant | E11 | the P0-1/P0-3 checks of those inputs, reported under P0-5 |

### P1

* **Implemented: P1-1 TMEM reuse ordered** (CuTe, `specs.md` "P1-1"): cross-role TMEM
  accesses to a common cell, one writing, must be ordered by sound mbarrier/named-barrier
  happens-before with PTX completion (wait::ld/st, landed commits); pairs only the
  ledger's lockstep contents order leave the input undecided. Basis: the PTX ISA's
  tcgen05 completion rules (a load or store completes at the issuing warp's
  `tcgen05.wait::ld`/`::st`, a copy or MMA at a landed `tcgen05.commit`).
* Buffer reuse WAR/RAW across stages and persistent tiles in shared and global memory,
  and TMEM in TTGIR/TLX kernels (E12; needs interleavings).
* Global deadlock freedom: wait cycles that depend on timing, minimum ring depth (cycles
  certain in the lockstep model are P0-2).
* Completion/wait work-item correspondence: the k-th completion of a slot and a role's
  k-th wait on it belong to the same work item. It catches a mid-run extra completion
  that the waits absorb in order (e.g. an MMA commit for a zero-length tile the
  consumer skips), which phase counting cannot tell
  from a trailing completion (two such commits on one CTA are a P0-2 hang). It needs a
  `work` event in every role, including those that never address memory.
* Whole-kernel IR scans: one `cta_group` per kernel, single-thread issue of
  `tcgen05.mma`/`commit`/`cp.async.bulk`/`expect_tx`, warp-uniform collective operands,
  `try_wait` result looped, no `%warpid`/`%smid` in logic.
* Cluster/2-CTA beyond the mbarrier protocol and TMEM reuse (CuTe: remote arrives,
  multicast commits, cta_group::2 and multicast loads, DSMEM bulk copies and the cluster
  barrier are P0 events, cta_group::2 MMAs and copies P1-1 accesses of both CTAs of the
  pair): cross-CTA data flow, TTGIR 2-CTA; TMEM
  rules beyond ordering (allocation lifetime, relinquish, fences) and the allocator
  protocol; cross-CTA flags and CLC schedulers.
* TMA descriptor legality (box <= 256, swizzle span, 16 B alignment, box == smem tile).
* Writer/reader index-map agreement (smem view stages, inter-kernel SF layouts).
* Async completion (incl. thread affinity), proxy fences (both directions, tensormap proxy).
* mbarrier/named-barrier lifecycle; scale-factor algebra; pointer/stride provenance;
  tcgen05 ordering; inline-asm bit tricks.

### P2

Host-kernel contract and autotune preconditions, PDL ordering (E13), NaN/Inf propagation
(E14), floating-point range invariants (exp argument <= 0, FP8 saturation, FP16
accumulation), uninitialized reads (E14), warp convergence, index-frame confusion, host
orchestration (autotune hooks, relaunch idempotence), float-atomic determinism, RNG offset
injectivity, `setmaxnreg`.

### Out of scope

Numerical accuracy bounds (only overflow ranges), performance, compiler lowering below the
extraction IR, multi-GPU signalling.

## 4. Trust base

A P0 result depends on exactly four things:

1. the IR the compiler produced for the analyzed config (nothing below it is checked);
2. the Lean semantics of each IR op the compiler front end extracts (one table per
   dialect, written from the PTX ISA and TLX/CuTe docs, the same for every kernel);
3. the Lean compiler (`native_decide`, T1), which decides every verdict;
4. for P0-2 and P0-4, the phase ledger algorithm (references/design.md, trust base),
   whose soundness against mbarrier semantics is argued informally.

Nothing in the chain is hand-modeled per kernel.

## 5. Cheap test-side complements

These close most of the "why tests missed it" gaps above without any proof:

* NaN-fill (poison) every output and workspace before the kernel runs;
* more tiles than SMs, unaligned and zero-length segments;
* magnitude checks next to cosine/SQNR;
* run every autotune config, not only the selected one;
* bitwise grid-invariance tests (same result for different grid sizes).
