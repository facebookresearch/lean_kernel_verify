# lean_kernel_verify (LKV)

LKV checks Blackwell GPU kernels written in TLX/Triton or the CuTe DSL for hangs,
out-of-bounds or wrapped addresses, missing or duplicate output writes, mbarrier
protocol errors and, for CuTe, unordered reuse of tensor memory (TMEM). Every check is
decided in Lean 4, with no GPU and no hand-written model of the kernel: the kernel's
real host code and compiler produce its IR on a CPU-only machine, a mechanical
frontend turns the IR into a small kernel IR (KIR), and Lean evaluates every role of
every CTA on the concrete inputs of a case file and proves or refutes each spec by
`native_decide`.

Verdicts cover the inputs of the case file, not every shape. LKV does not check
numerics, performance, CUDA C++ kernels, TLX/Triton 2-CTA kernels, or race and
deadlock freedom in general ([Limits](#limits)).

## Status

- Runs anywhere (Linux, Lean 4.34.0, Python 3.10 or newer): `scripts/lkv.py check`,
  the Lean layer on KIR files, and the tests listed under [Tests](#tests).
- `scripts/lkv.py run`, from a case file to a report, still builds the frontends with
  Buck2 inside the monorepo LKV is developed in, and `scripts/lkv.py export` reads
  kernel sources from that monorepo; from a clone of this repository both stop with an
  error. A runner for plain Python environments is planned.

## What it checks

| Spec                                | FAIL: a violation certain in the extracted model                                                                                                                           |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P0-1 global accesses in bounds      | a masked-in access outside its allocation, a plain write outside its work item's segment, an i32/i64 wrap on an address path, or a metadata load outside its tensor        |
| P0-2 role trip-count agreement      | a wait for a barrier completion that never comes or a wait cycle (a hang), or arrivals that do not split into one fixed share per arriving role                            |
| P0-3 output coverage exactly once   | an intended output element that plain stores write zero times or more than once, or a plain store outside the intended region                                             |
| P0-4 mbarrier protocol              | a parity other than 0 or 1 or a stale one, expect_tx bytes that differ from the TMA bytes landing in a waited phase, TMA bytes landing where nobody waits                  |
| P0-5 mask-elision soundness         | an input on a host-selected mask-free fast path that violates P0-1 or P0-3                                                                                                 |
| P1-1 TMEM reuse ordered (CuTe only) | two tcgen05 ld/st/cp/mma accesses of different roles to a common TMEM cell, one of them a write, that the kernel's barriers and waits do not order                         |

Each (compiled variant, input, spec) gets one of four verdicts: **PASS** (the theorem
is proved), **FAIL** (a certain counterexample, reported at the kernel's source line),
**ERROR** (LKV, the case or the environment failed and must be fixed) or **UNDECIDED**
(a known limit, such as a construct the frontend refuses by design). Findings that can
be harmless are notes and never change a verdict. A run exits with 1 if a verdict
FAILs, else 2 on any ERROR, else 0 (P1-1's FAILs count only with `--p1-strict`). The
exact definitions are in [references/specs.md](references/specs.md).

## How it works

```
kernel sources + case file (public host entry point, config, small concrete inputs)
  |  frontend/*/dump: the real host code and compiler on CPU (device queries patched,
  |  launches intercepted)
  v
compiler IR: TTGIR (TLX/Triton, sm_100 or sm_103) or CuTe DSL MLIR (sm_100)
  |  frontend/*/to_kir: a fixed op table and program slicing; unknown ops are errors
  v
KIR (JSON): roles, structured control flow, exact-width integer expressions, events
  |  frontend/kir.py checks the contract; frontend/emit_lean.py writes Lean
  v
lean/KV: every role of every CTA evaluated per input into event traces, then per spec
  |  theorem : KV.Holds .<spec> kernel input := by native_decide
  v
report.md / report.json: a verdict per (variant, input, spec), findings at source file:line
```

The frontends keep only the ops that define control flow, addresses, masks, barrier
operands or events, and stop with an error if a dropped op feeds one of them. A verdict
rests on the compiler IR (nothing below it is checked), the op tables,
`native_decide`, and for P0-2, P0-4 and P1-1 a phase-ledger algorithm that is argued
sound but not proved. The architecture and the trust base are in
[references/design.md](references/design.md).

## Requirements

- Linux.
- Lean 4.34.0, core Lean and Std only (no Mathlib, no Lake): set `LEAN_TOOLCHAIN_DIR`
  to a toolchain directory with `bin/lean` and `lib/lean` (an unpacked Lean release),
  or put its `lean` on `PATH` (e.g. `elan default leanprover/lean4:v4.34.0`).
- Python 3.10 or newer. `lkv.py check` and the Lean layer use the standard library
  only.
- For the frontends, the kernel's own Python stack: a Triton with TLX and PyTorch for
  TLX/Triton kernels, or `nvidia-cutlass-dsl` 4.4.2 exactly and PyTorch for CuTe DSL
  kernels. No GPU.

## Quick start

```bash
git clone https://github.com/facebookresearch/lean_kernel_verify
cd lean_kernel_verify
export LEAN_TOOLCHAIN_DIR=/path/to/lean-4.34.0-linux   # or Lean 4.34.0's lean on PATH

python3 -I scripts/lkv.py check tests/lean/pipeline_ok.kir.json \
  tests/lean/floor_tile_tail.kir.json --out /tmp/lkv_check
```

The first run builds `lean/KV` and its native library `libKV.so` in
`/tmp/lkv_lean_build_$USER` (`LKV_LEAN_BUILD` overrides the directory). `pipeline_ok`
passes every spec; `floor_tile_tail` launches `N // 64` programs of 64 rows, so for
N = 100 and N = 40 the rows after the last full tile are never written:

```
[lkv] lean 1/2 pipeline_ok: 0 FAIL, 0 ERROR, 0 UNDECIDED, 15 notes (0.4 s)
[lkv] lean 2/2 floor_tile_tail: 2 FAIL, 0 ERROR, 0 UNDECIDED, 0 notes (0.5 s)
...
[lkv] result: FAIL (exit status 1); FAIL: 2 certain violation(s) (P0-3: 2)
```

`/tmp/lkv_check/report.md` gives the verdict per input and spec and the counterexample
(`tensor out: 576 intended element(s) never written by a plain store: rows [64, 100)`);
`report.json` has the same data for scripts.

Where the frontends run (see [Status](#status)), a case runs end to end with
`python3 -I scripts/lkv.py run CASE.json --src DIR`, `DIR` holding the kernel sources at
the revision to check. [SKILL.md](SKILL.md), "Run a case", has the options and how to
check a fix against its base.

## Write a case

A case file names the kernel's source, its public host entry point, small concrete
inputs and the annotations the specs need: segments, work items, intended output
regions and mask-elision flags. The annotations name values; they never model behavior.
Because verdicts cover only the case's inputs, degenerate inputs are mandatory: empty
segments, partial last tiles, more work items than SMs with odd and even tile counts,
host-argument extremes and every host-selected variant. Minimal cases:
[tests/e2e/cases/probe_tlx.json](tests/e2e/cases/probe_tlx.json) (TLX/Triton) and
[tests/e2e/cases/probe_cute.json](tests/e2e/cases/probe_cute.json) (CuTe DSL). How to
write one: [references/cases.md](references/cases.md).

## Tests

```bash
tests/lean/run_all.sh                    # Lean layer: verdicts, counterexamples and notes of every KIR fixture
python3 -I tests/cute/test_core.py       # CuTe translation, pure Python
python3 -I tests/e2e/test_e2e.py Unit    # lkv.py's reading of Lean's output, its report and command line
```

The other tests need the frontends: `tests/ttgir/test_translate.py` and
`tests/ttgir/test_ttgir.py` (Triton with TLX), `tests/cute/test_frontend.py` (CuTe
DSL), `tests/e2e/test_e2e.py EndToEnd` (`lkv.py run`) and
`tests/ttgir/test_gpu_parity.py` (CPU-dumped TTGIR equals the TTGIR of a real launch;
needs a Blackwell GPU).

## Repository layout

```
SKILL.md       the same workflow as an Agent Skill, for coding agents
frontend/      Layer 1: ttgir/ (TLX/Triton) and cute/ (CuTe DSL) frontends,
               kir.py (the KIR contract), emit_lean.py (KIR to Lean)
lean/KV/       Layer 2: the KIR AST, the evaluator and the specs (Spec/)
scripts/       lkv.py (run, check, export) and the Lean build and check scripts
references/    the documentation below
tests/         lean/ (Lean layer), cute/ and ttgir/ (frontends), e2e/ (lkv.py end to end)
```

## Documentation

| Document                                                                                                       | Covers                                                               |
| -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| [SKILL.md](SKILL.md)                                                                                           | running LKV and reporting its verdicts, written for coding agents    |
| [references/design.md](references/design.md)                                                                   | the architecture and the trust base                                  |
| [references/specs.md](references/specs.md)                                                                     | each spec's definition, notes and verdicts, `report.json`, limits    |
| [references/kir.md](references/kir.md)                                                                         | KIR, the JSON contract between the layers                            |
| [references/frontend_ttgir.md](references/frontend_ttgir.md), [frontend_cute.md](references/frontend_cute.md) | the frontends: case formats, host patches, translation, refusals     |
| [references/ir_ops.md](references/ir_ops.md), [ir_ops_cute.md](references/ir_ops_cute.md)                     | the op tables: which IR op becomes which KIR event or value          |
| [references/cases.md](references/cases.md)                                                                     | writing a case file                                                  |
| [references/bug_taxonomy.md](references/bug_taxonomy.md)                                                       | which bug classes are checked, and why                               |

## Limits

- Verdicts cover only the case's inputs and the compiled variants they select.
- Clusters: CuTe DSL 2-CTA kernels get the P0 specs (remote barriers, multicast and
  `cta_group::2` TMA loads and commits, distributed shared-memory bulk copies, the
  cluster barrier) and P1-1 (a `cta_group::2` MMA or copy accesses the TMEM of both
  CTAs of its pair).
- Refused with an error naming the construct: TLX/Triton 2-CTA kernels and cross-CTA
  constructs (remote barriers, distributed shared memory), im2col TMA, TTGIR
  loop-carried pointers and side-effecting inline asm. CuTe DSL kernels are supported
  on sm_100 with `nvidia-cutlass-dsl` 4.4.2 only.
- Not checked: anything below the compiler IR, numerics, tcgen05 fences, TMEM lifetime
  and P1-1 on TLX/Triton. A P0-2 or P0-4 PASS means the phase ledger found no
  violation, not that the kernel is free of deadlocks or races.

The full list is in [SKILL.md](SKILL.md) ("Limits") and
[references/specs.md](references/specs.md) ("Limits").

## Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License

lean_kernel_verify is MIT licensed, as found in the [LICENSE](LICENSE) file.
