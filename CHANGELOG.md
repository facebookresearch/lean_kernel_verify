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
- Tests: the Lean layer's KIR fixtures (`tests/lean`), the frontends' tests and the
  end-to-end tests of `lkv.py`.
- KIR `ir` accepts `"ptx"` (KIR translated from PTX; references/kir.md change 12).
- Preview: `LKV_FRONTEND=ptx` makes both frontends translate the kernel's PTX
  (`frontend/ptx/`) instead of their IR, into KIR with `ir` `"ptx"`, for kernels whose
  global accesses are plain loads and stores (no TMA, tcgen05, clusters or CLC yet, no
  P1 pass); the default, `legacy`, is the translation above. With CuTe DSL 4.4.2 its
  source locations are best effort, as report.json `loc_quality` and report.md say
  (references/frontend_ttgir.md and frontend_cute.md, "PTX translation").
- PTX translation: a TMA or bulk copy that several lanes of a warp may execute follows
  the issuing rules measured on Blackwell hardware for each PTX compiler the frontends
  use (`tests/ptx/test_gpu_tma_issue.py`): loads and reduces once per lane with that
  lane's operands, TMA stores once per distinct (descriptor, coordinates, shared source)
  and bulk stores once per distinct (global address, shared source) in each warp, the
  cache policy not part of the key. It is translated only if the kernel's compiler (CuTe:
  its cubin's `.note.nv.tkinfo`; TTGIR: the `ptxas` the dump now records in every
  LaunchRecord) is one of `frontend/ptx/tma.py` `ISSUE_PTXAS` and the kernel targets
  sm_100 (`ISSUE_TARGETS`: sm_103 once the hardware test passes on GB300), else refused.
  A store whose copies would depend on the lane `elect.sync` picks is refused, as a load
  or reduce is.
- CuTe dump: the LaunchRecord address candidates of a launch whose TVM-FFI environment
  stream is not the jit function's last parameter are no longer shifted by one
  parameter.
- Docs: CuTe DSL 4.4.2 compiles with the PTX compiler it embeds (12.9.83), not CUDA
  13.0's ptxas; a TMA reduce is once per warp only behind the DSL's `elect.sync`.
