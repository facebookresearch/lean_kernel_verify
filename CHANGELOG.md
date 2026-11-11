# Changelog

All notable changes to this project are documented in this file.

## Unreleased

First public release.

- Five P0 safety specs, decided in Lean 4.34.0 by `native_decide` on the concrete inputs
  of a case file: global accesses in bounds (P0-1), role trip-count agreement (P0-2),
  output coverage exactly once (P0-3), mbarrier protocol (P0-4) and mask-elision
  soundness (P0-5). For CuTe DSL kernels, also TMEM reuse ordered (P1-1).
- Frontends for TLX/Triton (TTGIR) and the CuTe DSL (`nvidia-cutlass-dsl` 4.4.2 and
  4.6.1, the release a case names) that run the kernel's real host code and compiler
  on a CPU and translate the IR to KIR with a fixed op table.
- `scripts/lkv.py`: `run` (case file to report), `check` (the Lean layer on KIR files)
  and `export`; `run` and `export` still need the monorepo LKV is developed in (see the
  README's Status section).
- Tests: the Lean layer's KIR fixtures (`tests/lean`), the frontends' tests, the PTX
  translation's tests (`tests/ptx`) and the end-to-end tests of `lkv.py`.
- KIR `ir` accepts `"ptx"` (KIR translated from PTX; references/kir.md change 12).
- `lkv.py run --frontend ptx` (experimental and opt-in, with known gaps and bugs; the
  default, `legacy`, is the translation above; `$LKV_FRONTEND` sets the default) makes
  both frontends translate the kernel's PTX instead of their IR, with one shared core
  (`frontend/ptx/`) and one PTX op table (references/frontend_ptx.md, ir_ops_ptx.md), into
  KIR with `ir` `"ptx"`: global and metadata accesses, mbarriers, TMA and bulk copies
  (packed sub-byte formats too), tcgen05 commits, CLC (TLX), clusters and 2-CTA kernels
  and the P1 pass (CuTe). Refusals name the PTX instruction, its source line and its PTX
  line. report.json `translation` and a line of report.md name the translation of a run.
  With CuTe DSL 4.4.2 the source locations are best effort, as report.json `loc_quality`
  and report.md say.
- PTX translation: a TMA or bulk copy that several lanes of a warp may execute follows
  the issuing rules measured on Blackwell hardware for each PTX compiler the frontends
  use (`tests/ptx/test_gpu_tma_issue.py`): loads and reduces once per lane with that
  lane's operands, TMA stores once per distinct (descriptor, coordinates, shared source)
  and bulk stores once per distinct (global address, shared source) in each warp, the
  cache policy not part of the key. It is translated only if the kernel's compiler (CuTe:
  its cubin's `.note.nv.tkinfo`; TTGIR: the `ptxas` the dump now records in every
  LaunchRecord) is one of `frontend/ptx/tma.py` `ISSUE_PTXAS` and the kernel targets
  sm_100 or sm_103 (`ISSUE_TARGETS`, measured on a GB200 and a GB300), else refused.
  A store whose copies would depend on the lane `elect.sync` picks is refused, as a load
  or reduce is.
- PTX translation, coverage: `ld`/`st`/`atom`/`red` with any memory ordering, scope,
  `.volatile` or cache operator (the model has no weak memory, but the value of a strong
  load, `.volatile`, `.relaxed` or `.acquire`, or of a `.cv` one is Data, never a
  metadata read: another thread may write it), generic and vector atomics, shared-memory
  atomics (`atom`/`red.shared`: no event, Data results), `red.async.global`; `cvta`
  between the shared or local windows and generic addresses; thread-local memory
  (`__local_depot`); integer `abs`, `mad.hi`, `.relu`, `bfind`, `dp2a`, `add`/`sub.sat`,
  `mul24`, `sad`, packed `.s16x2`/`.u16x2`, the `prmt` modes, saturating `cvt`, the
  carry and borrow chains (`add.cc`, `addc`, `sub.cc`, `subc`, `mad*.cc`) and 16- and
  64-bit `mul.hi` (a division by a constant as `udiv`); `redux.sync`, `barrier.red` and
  `bar.red`; `.add.noftz` bulk reduces; CLC `get_first_ctaid::{x,y,z}`; `createpolicy`
  and `mma.sync` (drops); Triton's `tl.device_assert` (`st.param::func`, a `trap` ending
  its block); the Triton warp-specialize dispatch as a compare chain (at most 3 worker
  states) and with a state table of vector immediates; in-kernel descriptors without a
  rank (1-D) or zeroed through a register known to be 0; `mbarrier.inval` that ends a
  private synchronous cycle (the cycle is removed). An integer `cvt` into a wider
  register now extends a signed type by its sign, and bits that `brev`, `bfi`, `prmt`,
  `shf`, ballots, lane masks or register packing move into the sign bit are no longer
  reported as address wraps (P0-1). Refused: a shared write that may overlap an open
  tensormap staging buffer before its copy (also under a data predicate), a
  `tensormap.replace` in global memory, a generic shared write or a shared atomic in a
  TLX warp-specialized kernel, a carry-flag instruction under a predicate, an
  `mbarrier.inval` outside such a cycle; an explicit `.global` access through a shared
  address is an error.
- CuTe dump: the LaunchRecord address candidates of a launch whose TVM-FFI environment
  stream is not the jit function's last parameter are no longer shifted by one
  parameter.
- CuTe frontend: `addr` marks follow the `let`s of one role only (each role numbers its
  variables independently), so a value that bounds a loop in one role is no longer
  marked as an address because another role's variable of the same name reaches one
  (false P0-1 wraps). The CPU device stand-in reports `shared_memory_per_block_optin`.
- Both frontends: an input counts as rejected by the host only if the exception comes
  from a `raise` or `assert` statement of the kernel's sources or from a torch check they
  call; any other exception at a source line (a failing builtin, a device property the
  CPU stand-in lacks, also in an assert's condition or a raise's message) fails the dump
  instead of silently dropping the input.
- `lkv.py check` refuses two KIR files of one name (one would overwrite the other's Lean
  file and verdicts). `scripts/build_lean.sh` builds into one directory per `lean/`
  sources and toolchain (its staleness check compares modification times only).
- Docs: CuTe DSL 4.4.2 compiles with the PTX compiler it embeds (12.9.83), not CUDA
  13.0's ptxas; a TMA reduce is once per warp only behind the DSL's `elect.sync`.
