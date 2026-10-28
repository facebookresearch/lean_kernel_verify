---
name: lean-kernel-verify
description:
  'Mechanical P0 safety checks of Blackwell GPU kernels (TLX/Triton, CuTe DSL)
  in Lean 4 with no hand-written kernel model: on a CPU-only machine the real
  host code and compiler produce the IR (TTGIR, CuTe MLIR), a sliced op-table
  frontend turns it into KIR, and Lean decides per case input, by native_decide,
  P0-1 bounds and address wrap, P0-2 role trip counts (hangs), P0-3 output
  coverage exactly once, P0-4 mbarrier protocol (parity, expect_tx bytes) and
  P0-5 mask elision on host-selected fast paths, plus (CuTe only, reported
  apart) P1-1 TMEM reuse ordering (cross-warp tcgen05 ld/st/cp/mma races). For
  checking a kernel revision or a fix for hangs, out-of-bounds or wrapped
  addresses, missing or duplicate writes, barrier bugs and TMEM reuse races on
  degenerate shapes, or writing a case file for a new kernel; verdicts cover the
  case inputs only.'
---

# lean-kernel-verify: compiler IR to Lean verdicts for five P0 specs and P1-1

The skill checks a TLX/Triton or CuTe DSL kernel at a given revision against
five P0 safety specs, on the concrete inputs of a case file, with no model of
the kernel: the real host code and compiler produce the IR on a CPU-only
machine, a mechanical frontend turns it into KIR, and Lean proves or refutes
each verdict as a `native_decide` theorem.

**Use it** to formally check a Blackwell kernel, or a fix against its base, for
hangs, out-of-bounds or wrapped addresses, missing or duplicate output writes,
mbarrier parity or expect_tx bugs and (CuTe) unordered TMEM reuse across warps
(P1-1, e.g. a scale-factor copy into a TMEM region that another warp is still
loading), especially on empty, partial and multi-wave shapes, including CuTe
2-CTA (cluster) kernels for the P0 specs; or to add a case for a new kernel.
**Do not use it** for numerics, performance, CUDA C++ kernels, TTGIR/TLX 2-CTA
kernels, or general race/deadlock freedom (see Limits).

```bash
S=/path/to/lean_kernel_verify   # the directory of this file
```

## Pipeline

```
kernel sources exported at REV + case file (host entry point, config, small concrete inputs)
  | Layer 1: CPU only, real code (frontend/)
  |   the real host code runs on CPU tensors; device queries are patched; launch intercepted
  |   TLX/Triton: triton.compile -> TTGIR (sm_100 or sm_103)        frontend/ttgir/dump.py
  |   CuTe DSL:   cute.compile -> initial MLIR -> analyzed stage     frontend/cute/main.py
  |   op table + program slicing (refuses on anything unknown)      to_kir
  v
KIR JSON per compiled flag set: roles, structured control flow, exact-width integer exprs,
  |   events, inputs; frontend/kir.py (contract check) -> frontend/emit_lean.py
  |   CuTe: a second pass with the P1 slice writes <kernel>__<digest>.p1.json (+ TMEM,
  |   tcgen05.wait and named-barrier events); the P0 KIR does not change
  v
Layer 2: lean/KV evaluates every role of every CTA per input -> event traces
  |   theorem : KV.Holds .<spec> kernel input := by native_decide   (5 specs x inputs;
  |             P1 KIR: .tmemReuse x inputs)
  v
report.md / report.json: verdict per (variant, input, spec), findings at source file:line;
  P1-1 in a section of its own
```

## Setup (no GPU needed)

- **Lean 4.34.0**, core Lean + Std only (no Mathlib; do not use `lake`).
  `scripts/lean.sh` takes the toolchain from `LEAN_TOOLCHAIN_DIR` (a directory with
  `bin/lean` and `lib/lean`), else the `lean` on `PATH` (e.g. from elan);
  `scripts/build_lean.sh` compiles `lean/KV` and the native `libKV.so` into
  `/tmp/lkv_lean_build_$USER` (`LKV_LEAN_BUILD`), and `lkv.py` runs it when needed.
- **The frontends run the kernel's real host code**, so they need its Python stack:
  a Triton with TLX and torch (TTGIR frontend, `frontend/ttgir/`), or
  nvidia-cutlass-dsl 4.4.2 exactly (the release the CuTe op table was checked
  against) and torch (CuTe frontend, `frontend/cute/`), plus whatever the kernel
  package imports. At this revision `lkv.py run` builds and runs the frontends with
  Buck2 inside the monorepo that holds the skill; `lkv.py check` (Layer 2 on KIR
  files) runs anywhere.

## Run a case: scripts/lkv.py

```bash
# 1. SRC: a directory holding the kernel sources at the revision to check, from which the
#    case's `module` (TTGIR) or `source.package` (CuTe) imports.
# 2. Run: dump -> to_kir -> kir.py -> emit_lean -> Lean -> report.
python3 -I $S/scripts/lkv.py run CASE.json --src SRC
# 3. Read /tmp/lkv_runs/<case dir>/<case name>/report.md (or --out DIR).
python3 -I $S/scripts/lkv.py check K.kir.json... --out DIR   # Layer 2 alone on KIR files
```

- The frontend is CuTe if the case has a `builder`, else TTGIR. `run` options:
  `--inputs a,b`, `--target sm_103` (TTGIR only), `--jobs N` (default min(8,
  CPUs)), `--timeout S` (per Lean file, default 3600), `--src DIR` (default: the
  case's `src_root` / `source.src`).
- **Check a fix:** put base and fix in two source directories, run the same cases
  with each `--src`, and compare FAIL locations. Done when every confirmed
  root's FAIL is gone and no new FAIL appears; inputs the fixed host now rejects
  are listed apart and the others still run. A fix that moves writes between a
  TMA descriptor and a pointer over the same output needs `intended_writes` per
  revision: KIR v1 cannot alias the two tensors.
- Every verdict, per (variant, input, spec), is one of four: **PASS** (its
  theorem is proved); **FAIL** (a violation certain in the extracted model: a
  certain counterexample, or a certain error of the kernel that stops the
  evaluation: a failed device assert, a barrier index out of range, a loop that
  never ends, a negative grid); **ERROR** (LKV, the case or the environment
  failed and a human must fix it: a frontend error, invalid KIR, a DSL version
  mismatch, a Lean error or timeout, an `INCONSISTENT` verdict, an unexpected
  evaluation error such as a `while` over `max_iters` or a case annotation
  error); **UNDECIDED** (not decided where LKV has a known limit, not an error:
  a construct the frontend refuses by design, a role trace stopped at an
  out-of-bounds metadata load (itself a P0-1 FAIL) when nothing certain is found
  for that spec, a division by zero, P1-1 pairs only the lockstep contents
  order). A variant without verdicts gives every spec of each of its inputs its
  outcome (UNDECIDED if refused by design, else ERROR).
- Outcome and exit status of a run (P0 and P1-1's ERRORs; P1-1's FAIL and
  UNDECIDED only with `--p1-strict`; `--no-p1` skips it): **1 FAIL** = some
  verdict FAILs, whatever else failed; **2 ERROR** = no FAIL, but something
  failed (also a pipeline error, no KIR at all, and any error of lkv.py itself,
  so 1 always means a certain violation); **0 UNDECIDED** = no FAIL and no
  ERROR, but a known limit leaves something open; **0 PASS** = every spec is
  proved on every input of every variant. Host-rejected inputs count as none:
  the exported sources raised for the input (a `raise` or `assert`, a failed
  torch check such as `torch._assert_async`) or launched nothing.
- `report.md`: header (rev, src, target), the result (outcome, exit status and
  why), the frontend's environment, PASS / FAIL / ERROR / UNDECIDED totals per
  spec, verdicts per input (`FAIL (k/n)` = k of the n compiled variants that ran
  it have a certain violation, `ERROR (e/n)` = e of them failed, `UNDECIDED
  (u/n)` = u of them hit a known limit), a FAIL section (counterexamples by spec
  and source line, kernel errors), an ERROR section (variants that failed to
  translate or check, unexpected evaluation errors, INCONSISTENT verdicts), an
  UNDECIDED section (variants refused by design, model limits, specs refuted
  without a certain counterexample), notes, host-rejected inputs, the variants
  table (flags, Lean time). Artifacts: `frontend/`, `lean/`, `logs/`.
- `report.json`: the same data, for scripts. CI reads `result.outcome` (or the exit
  status); every key and count: [references/specs.md](references/specs.md),
  "report.json".

## Cases

A case file names the kernel source, its public host entry point, small concrete
inputs and the annotations the specs need; its `description` says what the inputs
trigger. Cases for public kernels go in `cases/`; `tests/e2e/cases/` has a small
probe case per frontend. [references/cases.md](references/cases.md) says how to
write a new case (source and public host entry point, inputs, annotations).
**Degenerate inputs are mandatory**, because verdicts
cover only the case inputs: a zero-length segment on each side, partial last
tiles, more work items than SMs with odd and even tile counts, host-argument
extremes, every host-selected flag variant, and one aligned control input.

## The five P0 specs

| P0  | Spec                         | FAIL (violation certain in the extracted model)                                                                                                                                                                                                                                                                                                                                                                                    | NOTE (may be harmless; never a FAIL)                                                                                                 |
| --- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | Global accesses in bounds    | a masked-in access outside its allocation (TMA boxes clipped per dim); a plain write outside its work item's segment; an i32/i64 wrap on an address path (add/sub/mul/shl, lossy trunc); a metadata load outside its tensor (on every input)                                                                                                                                                                                       | a read or reduction inside the allocation but outside the segment                                                                    |
| 2   | Role trip-count agreement    | a malformed barrier; a wait that needs a completion that never comes, or a wait cycle in the lockstep model (hang); waited phases whose arrivals do not split into one fixed share per arriving role summing to the count (or follow one initialization arrival); a static waiter of a barrier array with no event on a slot that completes and receives TMA bytes                                                                 | completions after the last wait (`producer_tail`); a trailing partial phase; a missing waiter on a completing slot without TMA bytes |
| 3   | Output coverage exactly once | an output without `intended_writes`; an intended element that plain stores write 0 times or 2+ times (over all CTAs); a plain store outside the intended region                                                                                                                                                                                                                                                                    | reductions outside the region; `unchecked` opt-outs                                                                                  |
| 4   | mbarrier protocol            | a parity other than 0/1; a stale repeated parity; a first wait that synchronizes with phase 0 on a TMA slot (parity 1 on the fresh barrier, unless a late join moves it); in a waited phase, expect_tx bytes != landing bytes; TMA bytes landing in a phase or on a slot nobody waits on; a load not covered by an earlier expect_tx of its role (in its phase, or at all); transaction bytes that cannot be attributed to a phase | laps; expect_tx bytes pending in a phase nobody waits for, or on a slot nobody waits on                                              |
| 5   | Mask-elision soundness       | an elided input (it sets one of the case's `elision_flags`) violates P0-1 or P0-3 on the variant the host selected; those findings are reported under P0-5 instead                                                                                                                                                                                                                                                                 | the P0-1/P0-3 notes of an elided input                                                                                               |

### P1-1 TMEM reuse ordered (CuTe only; `references/specs.md`, "P1-1")

| P1  | Spec                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | FAIL / undecided                                                                                                                                                                                                                                                                     | NOTE                                                                                                                          |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| 1   | TMEM reuse ordered: every pair of tcgen05 ld/st/cp/mma accesses of different roles that touch a common TMEM cell (lane x 32-bit column), one writing, is ordered: one completes (ld/st at the role's next `tcgen05.wait::ld/st`, cp/mma at a landed later commit of the same warp) before the other is issued, under sound mbarrier and named-barrier happens-before between roles (trust item 6, with partial credit of a multi-warp role's arrivals; no lockstep contents) | F1 a conflicting cross-role pair unordered even under the most generous lockstep reading; F2 an access outside the allocation; F3 an ld/st outside its warp's lane quarter; undecided (U1): a pair ordered only under the ledger's lockstep contents, with no certain counterexample | N1 same-role hazards that are not PTX pipelined pairs; N2 intra-role cross-warp pairs no `bar.sync` orders; N3 a blocked wait |

Decided on the P1 KIR, reported in its own report section and `report.json`
`variants_p1`; a refused P1 translation (R1-R9, `frontend_cute.md`) is "not
checked", TTGIR variants always. A PASS rests on sound happens-before between
roles (trust item 6): a wait that returned for phase k needs, of each arriving
role, k x count minus the other roles' arrivals that may precede its return (P1
ledger, `SlotLedger.needsWith`), and those show every warp of a role issued an
arrive only if they take more of its arrivals than all warps but one give
(partial credit). The P1 pass splits a role whose warps issue an mbarrier wait,
arrive or commit only in part (a thread-0 arrive or wait), so each such event of
a role is issued by all its warps.

**How a verdict is decided.** Each host flag set compiles to its own KIR
variant, paired with the inputs that select it. For each (variant, input, spec),
lkv proves or refutes `KV.Holds .<spec> kernel input` by `native_decide`, over
an exact `BitVec` evaluation of the KIR: PASS iff the theorem is proved; a
refuted theorem is a FAIL if Lean lists a certain counterexample for it (or the
evaluation stopped at a certain kernel error), an ERROR if LKV failed to decide
it, else UNDECIDED (a known limit). Roles are not interleaved: P0-2 and P0-4 are stated over the per-CTA
phase ledger (`lean/KV/Spec/Ledger.lean`).

## Trust base and reporting

A verdict depends on exactly these items (`design.md`, "Trust base"):

1. **The compiler IR** for the analyzed config: TTGIR for
   `GPUTarget("cuda", 100|103)`, or the CuTe `cute-to-nvvm` prefix stage at
   `sm_100a` with nvidia-cutlass-dsl 4.4.2, produced by the real host code at
   REV under the frontend's host patches (device queries, device placement,
   compile/launch interception, the case's `env`, tabled in the frontend docs).
   Nothing below the IR is checked.
2. **The op tables** (`ir_ops.md`, `ir_ops_cute.md`): one meaning per op, the
   same for every kernel. Unknown ops are hard errors, and so are data values
   that reach an address, a mask or a barrier operand (slice soundness). Issue
   counts follow what ptxas emits: a CuTe TMA store executed by a whole warp is
   one write per distinct (coordinates, shared-memory source) among its lanes
   (ptxas 13.0 issues it in a waterfall loop); a TMA load or reduce is one per
   warp (the DSL emits it behind `elect.sync`).
3. **`native_decide` (T1)**: the Lean compiler and the native `libKV.so`.
4. **The ledger algorithm** for P0-2 and P0-4, argued sound in Ledger.lean but
   not proved: a PASS there means "the ledger found no violation".
5. **P1-1 only:** the P1 op-table rows (TMEM footprints, the scale-factor
   footprints of the CuTe DSL's TMEM SF layouts, cp/mma and their commit issued
   by one elected lane) and the PTX completion rules it encodes.
6. **P1-1 only:** the P1 ledger (the ledger of item 4 with per-wait needs from
   the arrivals that may precede a wait's return, argued sound in Ledger.lean,
   not proved), named-barrier semantics, and the reading of a role of several
   warps on which every PASS rests. The P1 pass makes every warp of a role issue
   each of its mbarrier waits, arrives and commits (a role whose warps issue one
   only in part, e.g. a thread-0 arrive or wait, is split by issuing warps; one
   under a runtime lane condition is refused, R7; any lane issues for its warp),
   but its warps need not be at one point. Partial credit: a phase that needs m
   arrivals of such a role shows that some warp issued the arrive holding the
   m-th one (so the role's waits before it returned), and that every warp issued
   it only if m takes more of its arrivals than all warps but one give (each
   warp gives at least one mbarrier arrival, 32 threads of a named-barrier
   arrival); otherwise every warp issued only the role's previous arrive, and an
   ld/st completes only once every warp passed its wait. What stays assumed: no
   warp of a role runs so far ahead that its own arrivals on a slot stand in for
   another warp's earlier ones (a phase completes on a count). Intra-role
   cross-warp pairs are notes. The ledger's lockstep contents never decide a
   PASS: they only separate F1 from undecided. tcgen05 fences and TMEM
   allocation lifetime are not checked (the kernels checked so far have no tcgen05
   fences).

The case file decides what is checked: its inputs, and annotations that name
segments, work items, intended regions and elision flags. Review them like code.

When you report:

- Write "P0-n holds for case C, inputs I, at REV, target T". Never write "the
  kernel is verified": verdicts cover only the case inputs.
- A FAIL is a certain violation in the extracted model on an input the real host
  accepted. Before calling it a bug, confirm the root in the source (line,
  input, host acceptance) and check whether the config runs in production. The
  model's symptom can differ from the GPU's (an evaluation error where the GPU
  hangs).
- Notes are hazards, not bugs; an UNDECIDED verdict is neither a pass nor a bug:
  say what is not decided and why (the report's UNDECIDED section); an ERROR is
  a failure of LKV, the case or the environment: fix it (the report's ERROR
  section) before reading the run as a pass;
  P0-2/P0-4 PASS is not deadlock or race freedom. A P1-1 FAIL is an unordered
  pair in the model: confirm the missing edge in the source; a P1-1 UNDECIDED
  input names pairs only the lockstep contents order (check whether a role's
  arrivals can run ahead, e.g. credits arriving without a wait); a P1-1 PASS
  covers TMEM ordering only, on the case inputs, and assumes no warp of a role
  runs so far ahead that its arrivals stand in for another warp's (trust item
  6). Without a GPU run, say "not reproduced on hardware".

## Limits

Clusters (CuTe): remote mbarrier arrives, multicast commits, cta_group::2 and
multicast TMA loads, DSMEM bulk copies and the cluster barrier are checked with one
barrier ledger per cluster, and P1-1 runs per cluster (a cta_group::2 MMA or copy
accesses the TMEM of both CTAs of its pair); which CTA's data a copy carries is not
checked. Refused: TTGIR 2-CTA kernels (remote
barriers, DSMEM, multi-CTA layouts), CLC, im2col TMA, TTGIR loop-carried pointers (CuTe: a
loop-carried value changing structure, e.g. a pointer into another tensor),
side-effecting inline asm, a value that reaches an event through register
memory (CuTe) and a per-CTA tile count the partitions capture from an `scf.if`
(TTGIR). Not covered: completion/wait
work-item correspondence, P1 bug classes other than P1-1 (and P1-1 on TTGIR/TLX,
TMEM lifetime, tcgen05 fences, a warp of one role running ahead so that its
arrivals stand in for another's), inputs and variants no case selects. CuTe
supports sm_100 with DSL 4.4.2 only. Details and the spec-model limits:
[references/specs.md](references/specs.md), "Limits".

## References

The references detail the pipeline above, one piece each
([references/design.md](references/design.md) is the architecture they detail; the
table says when to read each):

- case file: [cases.md](references/cases.md) (the shipped cases, how to write one), with
  its fields per frontend in [frontend_ttgir.md](references/frontend_ttgir.md) "Case
  file" and [frontend_cute.md](references/frontend_cute.md) "Case files";
- Layer 1, IR to KIR: [frontend_ttgir.md](references/frontend_ttgir.md) and
  [frontend_cute.md](references/frontend_cute.md) (the dump, host patches, translation,
  refusals), with the op tables [ir_ops.md](references/ir_ops.md) (TTGIR) and
  [ir_ops_cute.md](references/ir_ops_cute.md) (CuTe): which IR op becomes which KIR event
  or value;
- KIR: [kir.md](references/kir.md), the JSON contract between the layers;
- Layer 2 and the report: [specs.md](references/specs.md) (each spec's Prop, notes,
  outcomes, evaluation errors, Lean's report lines, report.json, limits);
- why these specs: [bug_taxonomy.md](references/bug_taxonomy.md) (evidence, priorities,
  P1/P2, test-side complements).

| File                                                                                      | Read when                                                                                                                     |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [references/design.md](references/design.md)                                              | changing either layer, or explaining the architecture and trust base                                                          |
| [references/specs.md](references/specs.md)                                                | interpreting a FAIL, a note or an undecided spec, reading report.json, or before reporting: exact Props, shared notions, evaluation errors, report.json keys, limits |
| [references/kir.md](references/kir.md)                                                    | changing a frontend, writing a `tests/lean` fixture, or debugging `invalid_kir`                                               |
| [references/frontend_ttgir.md](references/frontend_ttgir.md)                              | writing a TLX/Triton case, or the TTGIR frontend refuses (case format, environment pinning, host patches, CLC)                |
| [references/frontend_cute.md](references/frontend_cute.md)                                | writing a CuTe case or builder, or the CuTe frontend refuses (work anchors, analyzed stage, DSL pin)                          |
| [references/ir_ops.md](references/ir_ops.md), [ir_ops_cute.md](references/ir_ops_cute.md) | an op is unknown or refused, or you need the events an op produces and which threads issue them                               |
| [references/cases.md](references/cases.md)                                                | choosing a case for a kernel, or writing a new case (fields, annotations, mandatory degenerate inputs)                        |
| [references/bug_taxonomy.md](references/bug_taxonomy.md)                                  | someone asks what is covered and why, or which spec to add next (evidence, P1/P2, test-side complements)                      |

## Tests

```bash
$S/tests/lean/run_all.sh                    # Layer 2: positive/negative controls per spec (tests/lean/p1: P1-1)
python3 -I $S/tests/e2e/test_e2e.py         # lkv.py end to end, ~2 min (frontends + Lean, no GPU)
python3 -I $S/tests/cute/test_core.py       # CuTe translation, pure Python
# Frontend tests: tests/ttgir/test_translate.py, test_ttgir.py (Triton) and tests/cute/test_frontend.py
# (CuTe DSL); tests/ttgir/test_gpu_parity.py: CPU-dumped TTGIR == real launch (needs a Blackwell GPU)
```
