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
