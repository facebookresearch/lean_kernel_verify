# lean_kernel_verify (LKV)

LKV checks Blackwell GPU kernels for hangs, out-of-bounds or wrapped addresses, missing
or duplicate output writes, mbarrier protocol errors, named barriers that not all their
warps reach alike, and unordered reuse of tensor memory (TMEM) or of the shared memory
and tensormaps a TMA store may still read. Every check is decided in Lean 4, with no GPU
and no hand-written model of the kernel: the kernel's real host code and compiler run on
a CPU-only machine and produce its PTX and its launches, a mechanical translator turns
the PTX into a small kernel IR (KIR), and Lean evaluates every role of every CTA on the
concrete inputs of a case file and proves each spec or its negation from one native
evaluation per input (the trust of `native_decide`).

LKV reads PTX, which Triton, the CuTe DSL and nvcc all hand to `ptxas`, so one translator
and one op table serve every kernel language; only the host adapter, which runs a
kernel's host code and records its PTX and launches, is per language. LKV has adapters
for TLX/Triton and the CuTe DSL; CUDA C++ kernels compile to PTX as well but need an
adapter LKV does not have yet.

Verdicts cover the inputs of the case file, not every shape. LKV does not check
numerics, performance, or race and deadlock freedom in general ([Limits](#limits)).

## Status

- Runs anywhere (Linux, Lean 4.34.0, Python 3.10 or newer): `scripts/lkv.py check`,
  the Lean layer on KIR files, and the tests listed under [Tests](#tests).
- `scripts/lkv.py run`, from a case file to a report, still builds its host adapters and
  translator with Buck2 inside the monorepo LKV is developed in, and
  `scripts/lkv.py export` reads kernel sources from that monorepo; from a clone of this
  repository both stop with an error. A runner for plain Python environments is planned.

## What it checks

| Spec                               | FAIL: a violation certain in the extracted model                                                                                                                          |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P0-1 global accesses in bounds     | a masked-in access outside its allocation, a plain write outside its work item's segment, an i32/i64 wrap on an address path, or a metadata load outside its tensor       |
| P0-2 role trip-count agreement     | a wait for a barrier completion that never comes or a wait cycle (a hang), or arrivals that do not split into one fixed share per arriving role                           |
| P0-3 output coverage exactly once  | an intended output element that plain stores write zero times or more than once, or a plain store outside the intended region                                             |
| P0-4 mbarrier protocol             | a parity other than 0 or 1 or a stale one, expect_tx bytes that differ from the TMA bytes landing in a waited phase, TMA bytes landing where nobody waits                 |
| P0-5 mask-elision soundness        | an input on a host-selected mask-free fast path that violates P0-1 or P0-3                                                                                                |
| P0-6 named barriers convergent     | a named-barrier instance (`bar.sync`, `bar.arrive`) that the warps its count names do not all reach with one count, or a wait on one that never ends                      |
| P0-7 global waits reached (opt-in) | a take of a tagged global word that waits forever, or a take of another CTA's store on a grid over the co-resident capacity (CTAs that synchronize through global memory) |
| P1-1 TMEM reuse ordered            | two tcgen05 ld/st/cp/mma accesses of different roles to a common TMEM cell, one of them a write, that the kernel's barriers and waits do not order                        |
| P1-2 async store sources reuse ordered | a write to the shared memory or the tensormap a TMA or bulk store may still read (before a `cp.async.bulk.wait_group{.read}` of its issuing thread completed its group: `.read` covers both) that the kernel's barriers and waits do not order after that read |

Each (compiled variant, input, spec) gets one of four verdicts: **PASS** (the theorem
is proved), **FAIL** (a certain counterexample, reported at the kernel's source line),
**ERROR** (LKV, the case or the environment failed and must be fixed) or **UNDECIDED**
(a known limit, such as a construct the translator refuses by design). Findings that can
be harmless are notes and never change a verdict. A run exits with 1 if a verdict
FAILs, else 2 on any ERROR, else 0 (the FAILs of P1-1 and P1-2 count only with
`--p1-strict`). The exact definitions are in [references/specs.md](references/specs.md).

## How it works

```
kernel sources + case file (public host entry point, config, small concrete inputs)
  |  host adapter (frontend/ttgir/: TLX/Triton, frontend/cute/: CuTe DSL): the real
  |  host code and compiler on a CPU, device queries patched, launches recorded
  v
the PTX of every compiled kernel + every launch (grid, block, cluster, parameters,
decoded TMA descriptors)
  |  frontend/ptx/: one PTX op table and program slicing for every kernel language; an
  |  instruction it has no row for, or cannot translate exactly, stops the variant
  v
KIR (JSON): roles, structured control flow, exact-width integer expressions, events
  |  frontend/kir.py checks the contract; frontend/emit_lean.py writes Lean
  v
lean/KV: every role of every CTA evaluated per input into event traces, then per spec
  |  the theorem KV.Holds .<spec> kernel input or its negation
  v
report.md / report.json: a verdict per (variant, input, spec), findings at source file:line
```

The translator keeps only the instructions that define control flow, addresses, masks,
barrier operands or events, and refuses a variant, naming the PTX instruction and its
source line, if a value it drops (floating-point data, shared or tensor memory contents,
a data load) reaches one of them. A verdict rests on the PTX it translates (nothing below
it is checked; the op table's TMA issuing rules are measured on Blackwell hardware for
each `ptxas` version), the host adapter's patches, the op table, `native_decide`, for
P0-2, P0-4, P1-1 and P1-2 a phase-ledger algorithm that is argued sound but not proved,
and for P0-6 the barrier pass's uniformity rules and, for arrive-based protocols, the
warp-ahead reading. The translation is in
[references/frontend_ptx.md](references/frontend_ptx.md), the architecture and the trust
base in [references/design.md](references/design.md).

`lkv.py run --frontend legacy` translates the compiler IR (TTGIR for TLX/Triton, MLIR for
the CuTe DSL) instead: an older translator, kept for constructs the PTX translator still
refuses; it refuses others (e.g. TLX/Triton cluster kernels) and does not check P1-2.

## Requirements

- Linux.
- Lean 4.34.0, core Lean and Std only (no Mathlib, no Lake): set `LEAN_TOOLCHAIN_DIR`
  to a toolchain directory with `bin/lean` and `lib/lean` (an unpacked Lean release),
  or put its `lean` on `PATH` (e.g. `elan default leanprover/lean4:v4.34.0`).
- Python 3.10 or newer. `lkv.py check`, the Lean layer and the PTX translator use the
  standard library only.
- For `lkv.py run`, the kernel's own Python stack: a Triton with TLX and PyTorch for
  TLX/Triton kernels, or `nvidia-cutlass-dsl` 4.4.2 or 4.6.1 exactly (the release the
  case names) and PyTorch for CuTe DSL kernels. No GPU.

## Quick start

```bash
git clone https://github.com/facebookresearch/lean_kernel_verify
cd lean_kernel_verify
export LEAN_TOOLCHAIN_DIR=/path/to/lean-4.34.0-linux   # or Lean 4.34.0's lean on PATH

python3 -I scripts/lkv.py check tests/lean/pipeline_ok.kir.json \
  tests/lean/floor_tile_tail.kir.json --out /tmp/lkv_check
```

The first run builds `lean/KV` and its native library `libKV.so` in
`/tmp/lkv_lean_build_$USER/<hash>`, one directory per `lean/` sources and toolchain
(`LKV_LEAN_BUILD` overrides the directory). Lean evaluates independent parts of an input
(CTAs, cluster ledgers) in parallel tasks on `LKV_LEAN_THREADS` threads (`lkv.py`: the
CPUs divided by `--jobs`, at least 2). `pipeline_ok`
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

Where `lkv.py run` works (see [Status](#status)), a case runs end to end with
`python3 -I scripts/lkv.py run CASE.json --src DIR`, `DIR` holding the kernel sources at
the revision to check. [SKILL.md](SKILL.md), "Run a case", has the options and how to
check a fix against its base.

## Write a case

A case file names the kernel's source, its public host entry point, small concrete
inputs and the annotations the specs need: segments, work items, intended output
regions and mask-elision flags. The annotations name values; they never model behavior.
Because verdicts cover only the case's inputs, choose the shapes and conditions the
kernel's callers produce: partial last tiles, more work items than SMs with odd and even
tile counts, and every host-selected variant they reach. Leave out very unlikely inputs
(an empty query, a window of 2^31 - 1, misuse the docs rule out) unless they are asked
for. Minimal cases:
[tests/e2e/cases/probe_tlx.json](tests/e2e/cases/probe_tlx.json) (TLX/Triton) and
[tests/e2e/cases/probe_cute.json](tests/e2e/cases/probe_cute.json) (CuTe DSL). How to
write one: [references/cases.md](references/cases.md).

## Tests

```bash
tests/lean/run_all.sh                    # Lean layer: verdicts, counterexamples and notes of every KIR fixture
python3 -I tests/ptx/test_translate.py   # PTX translator, pure Python (also the other tests/ptx/test_*.py)
python3 -I tests/e2e/test_e2e.py Unit    # lkv.py's reading of Lean's output, its report and command line
```

The host adapters' tests need the kernel's stack: `tests/ttgir/test_ptx_capture.py`
(Triton with TLX) and `tests/cute/test_ptx_capture.py` (CuTe DSL) check the PTX and the
launch records they capture, and `tests/e2e/test_e2e.py EndToEnd` runs `lkv.py run`. On a
Blackwell GPU, `tests/ptx/test_gpu_*.py` check the TMA issuing rules, packed sub-byte TMA
and the TMA descriptor decoder against the hardware, and `tests/ttgir/test_gpu_parity.py`
and `tests/cute/test_gpu_parity.py` that the CPU-only run compiles what a real launch
compiles. The other tests in `tests/ttgir/` and `tests/cute/` mostly cover the legacy IR
translators.

## Repository layout

```
SKILL.md       the same workflow as an Agent Skill, for coding agents
frontend/      Layer 1: ptx/ (the PTX translator, one for every kernel language),
               ttgir/ (TLX/Triton) and cute/ (CuTe DSL): the host adapters and their KIR
               assembly, with the legacy IR translators; kir.py (the KIR contract),
               emit_lean.py (KIR to Lean)
lean/KV/       Layer 2: the KIR AST, the evaluator and the specs (Spec/)
scripts/       lkv.py (run, check, export) and the Lean build and check scripts
references/    the documentation below
tests/         lean/ (Lean layer), ptx/ (PTX translator), ttgir/ and cute/ (host adapters),
               e2e/ (lkv.py end to end)
```

## Documentation

| Document                                                                                                      | Covers                                                               |
| ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| [SKILL.md](SKILL.md)                                                                                          | running LKV and reporting its verdicts, written for coding agents    |
| [references/design.md](references/design.md)                                                                  | the architecture and the trust base                                  |
| [references/specs.md](references/specs.md)                                                                    | each spec's definition, notes and verdicts, `report.json`, limits    |
| [references/frontend_ptx.md](references/frontend_ptx.md), [ir_ops_ptx.md](references/ir_ops_ptx.md)           | Layer 1: how the PTX becomes KIR, refusals; the PTX op table         |
| [references/kir.md](references/kir.md)                                                                        | KIR, the JSON contract between the layers                            |
| [references/frontend_ttgir.md](references/frontend_ttgir.md), [frontend_cute.md](references/frontend_cute.md) | the host adapters (TLX/Triton, CuTe DSL): case formats, host patches |
| [references/cases.md](references/cases.md)                                                                    | writing a case file                                                  |
| [references/bug_taxonomy.md](references/bug_taxonomy.md)                                                      | which bug classes are checked, and why                               |

## Limits

- Verdicts cover only the case's inputs and the compiled variants they select.
- Kernel languages: TLX/Triton (sm_100 and sm_103) and the CuTe DSL (sm_100,
  `nvidia-cutlass-dsl` 4.4.2 and 4.6.1); not yet CUDA C++.
- Clusters: 2-CTA and other cluster kernels get the P0 specs (remote barriers, multicast
  and `cta_group::2` TMA loads and commits, distributed shared-memory bulk copies, the
  cluster barrier, cluster launch control), and P1-1 and P1-2 run per cluster (a
  `cta_group::2` MMA or copy accesses the TMEM of both CTAs of its pair). A launch with a
  preferred cluster is checked with every cluster regular and with every cluster
  preferred.
- Not yet the same for every language: P1-1 runs on CuTe DSL kernels only and P1-2 on
  TLX/Triton kernels only, and CuTe DSL kernels with cluster launch control or launched
  with a preferred cluster are refused.
- Refused, with a message naming the PTX instruction and its source line: constructs the
  translator cannot translate exactly, among them im2col TMA
  ([references/frontend_ptx.md](references/frontend_ptx.md), "Refusals").
- Not checked: anything below the PTX (`ptxas`, SASS), numerics, tcgen05 fences, TMEM
  lifetime. A P0-2 or P0-4 PASS means the phase ledger found no violation, not that the
  kernel is free of deadlocks or races.

The full list is in [SKILL.md](SKILL.md) ("Limits") and
[references/specs.md](references/specs.md) ("Limits").

## Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License

lean_kernel_verify is MIT licensed, as found in the [LICENSE](LICENSE) file.
