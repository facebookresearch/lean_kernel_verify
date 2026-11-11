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
- `lkv.py run --frontend ptx` (opt-in; the default, `legacy`, is the translation above;
  `$LKV_FRONTEND` sets the default) makes both frontends translate the kernel's PTX
  instead of their IR, with one shared core (`frontend/ptx/`) and one PTX op table
  (references/frontend_ptx.md, ir_ops_ptx.md), into KIR with `ir` `"ptx"`: global and
  metadata accesses, mbarriers, TMA and bulk copies (packed sub-byte formats too),
  tcgen05 commits, CLC (TLX), clusters and 2-CTA kernels and the P1 pass (CuTe).
  Refusals name the PTX instruction, its source line and its PTX line. report.json
  `translation` and a line of report.md name the translation of a run. With CuTe DSL
  4.4.2 the source locations are best effort, as report.json `loc_quality` and report.md
  say.
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
- CuTe dump: the LaunchRecord address candidates of a launch whose TVM-FFI environment
  stream is not the jit function's last parameter are no longer shifted by one
  parameter.
- Docs: CuTe DSL 4.4.2 compiles with the PTX compiler it embeds (12.9.83), not CUDA
  13.0's ptxas; a TMA reduce is once per warp only behind the DSL's `elect.sync`.
