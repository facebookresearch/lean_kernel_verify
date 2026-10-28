# Design: kernel IR -> Lean program -> P0 specs (+ P1-1)

Two layers, no hand-written per-kernel model anywhere. This document is the
architecture; the other references detail its pieces, and SKILL.md "References" maps
each piece to its document.

```
kernel source (TLX/Triton or CuTe DSL) + case file (config, small concrete inputs)
   |  frontend/*/dump.py     host code + compile on CPU with the real compiler (no GPU)
   v
compiler IR (TTGIR for TLX/Triton, CuTe DSL MLIR)
   |  frontend/*/to_kir.py   Layer 1: op table + program slicing -> KIR (JSON)
   v
KIR program (roles, structured control flow, integer expressions, events)
   |  frontend/emit_lean.py  KIR -> Lean source
   v
Lean: KV.Kernel value + KV.Inputs value
   |  lean/KV                Layer 2: evaluator -> per-role event traces -> P0 specs
   v
verdict per spec (theorem or counterexample with IR loc -> source file:line)
```

`scripts/lkv.py run <case.json>` drives every step for one case (the frontend,
`frontend/kir.py`, `emit_lean`, `scripts/run_lean_check.sh`) and
writes `report.md` / `report.json`: the run's outcome and exit status (FAIL, 1,
if some verdict FAILs; else ERROR, 2, if something failed; else UNDECIDED, 0, if
a known limit leaves something open; else PASS, 0); the environment the frontend
ran with; per compiled variant, input and spec the verdict (PASS when its
theorem is proved; FAIL when it is refuted with a certain counterexample or the
evaluation stopped at a certain kernel error; ERROR when LKV, the case or the
environment failed: a frontend error, invalid KIR, a Lean error, an
INCONSISTENT verdict, an unexpected evaluation error; UNDECIDED at a known
limit: a construct the frontend refuses by design, a model limit such as a
division by zero, an out-of-bounds metadata load before anything certain);
then the FAIL, ERROR and UNDECIDED findings and the notes, grouped by spec and
source line. `lkv.py export` exports sources at a
revision and records it in the tree, so `run --src` reports the analyzed
revision; `lkv.py check` runs Layer 2 on existing KIR. End-to-end tests:
`tests/e2e/test_e2e.py`.

## Layer 1: the Lean AST compiler

### Frontends

- **TLX / Triton**:
  `triton.compile(ASTSource(fn, signature, constexprs), target=GPUTarget("cuda", 100, 32))`
  (103 with `--target sm_103`) runs on a CPU-only machine and returns
  `asm["ttgir"]` after warp specialization. Roles are the `default` and
  `partitionN` regions of `ttg.warp_specialize`; barriers are `ttng.*_barrier`
  ops; every op carries a `loc` that resolves to a Python source line.
- **CuTe DSL**: the kernel package's real host code (the case's entry point)
  runs on a CPU-only machine on real CPU tensors that the case's builder
  makes. Only device queries, device typing and the compile/launch calls are
  patched (`frontend/cute/host_patches.py`): the host sees one SM100 device with
  the case's SM count, the DLPack capsules `cute.runtime.from_dlpack` makes
  report a CUDA device (so the DSL types the tensors as global memory, as for
  CUDA tensors), `cute.compile` keeps the initial MLIR (arch sm_100a,
  `CUTE_DSL_KEEP_IR`) and stops before lowering, and calling the compiled kernel
  records the launch arguments instead of launching. The analyzed stage is the
  prefix of the DSL's own `cute-to-nvvm` pipeline in which every Python
  construct (`const_expr`, helper inlining, pipeline classes) is resolved,
  layouts are folded into index arithmetic and copy atoms are expanded, while
  `scf` loops are still structured. Roles are warp ranges, split where a
  warp-uniform branch separates warps (`references/frontend_cute.md`).

Kernel sources are loaded from a path given in the case file, so any revision
can be analyzed: export it at the revision first. The frontend never
imports the kernel from the build graph.

### KIR (contract between layers)

JSON, specified in `kir.md` (the contract `frontend/kir.py` checks) and emitted
one to one as the constructors of `lean/KV/IR.lean`. In short:

- Top level: `kir_version`, `kernel`, `source`, `ir`, optional `target` and
  `compiler`, `grid`, `params` (scalar kernel arguments), `tensors` (global
  tensors and TMA descriptors, with segments, role and metadata flag),
  `barriers`, `roles`, and `case` (the concrete inputs and the `elision_flags`).
- `Role = {name, warps, threads_per_issue, body}`; statements are `let`, `for`,
  `while` (with a `max_iters` budget), `if`, `assert` and `event`.
- Expressions are integer and boolean ops of explicit width (1, 32 or 64 bits)
  with exact `BitVec` semantics, program ids, grid sizes and `load` of integer
  metadata (seqlens, tile tables); tensor-valued index expressions (`TExpr`) are
  evaluated per coordinate.
- Events: `bar_init`, `arrive`, `commit_arrive` (the asynchronous completion of
  the role's prior MMAs), `expect_tx`, `wait`, `tma_load`, `tma_store` /
  `tma_reduce`, `gload` / `gstore` / `gatomic`, and `work` (the work item a role
  is processing).
- Every statement and event keeps the IR `loc` for reporting.
- A P1 KIR (`.p1.json`, CuTe only) adds the TMEM events `tmem` (ld/st/cp/mma
  with exact lane x column rectangles), `tmem_wait` and `named_bar`, and the
  top-level `tmem` allocation; the CuTe frontend writes it in a second
  translation pass, so P0 KIRs do not change (`kir.md` change 10).
- Host-selected flags (constexprs such as FULL_TILES) are resolved at compile
  time: each flag set is its own KIR, paired with the inputs that select it. The
  case names the flags that select a mask-elided fast path
  (`case.elision_flags`); P0-5 checks those inputs.

### Program slicing rule

Only ops that define control flow, addresses, masks, barrier operands or events
are kept; floating-point data ops, MMA math and layout conversions are dropped.
Before dropping an op the frontend checks that none of its results reaches a
kept op. If one does, extraction fails with the op and its `loc`. It never
guesses. Integer values loaded from memory become `Load` on concrete input
tensors supplied by the case file.

Unknown ops anywhere in the kernel (they could hide events) and unsupported
constructs (e.g. lane-divergent barrier ops, indirect calls) are hard errors.

## Layer 2: Lean evaluator and P0 specs

`lean/KV` (core Lean + Std only, no Mathlib):

- `IR.lean`: KIR AST.
- `Eval.lean`: evaluates each role of each CTA on concrete inputs and produces
  an event trace. Integer semantics use `BitVec`; a wrap on an address path (an
  `addr` add/sub/mul/shl that wraps or `addr` trunc that loses bits) is itself
  recorded as an event, and so is a metadata `load` outside its tensor, after
  which that role stops (later events depend on the unknown value; P0-1 fails,
  P0-2..P0-5 are undecided and list only their certain counterexamples). A
  scalar `select` evaluates only the operand it chooses, so a masked scalar load
  reads nothing where its mask is false. With thread-block clusters (KIR
  `cluster`) a barrier slot belongs to one CTA, named by its rank in the
  cluster: a remote arrive, a multicast commit or TMA load, a cta_group::2 load
  completing on the pair leader's barrier and a DSMEM copy (`complete_tx`) are
  events on another CTA's slot, and P0-2/P0-4 build one barrier ledger per
  cluster (the roles of all its CTAs; transaction bytes another CTA lands are
  attributed by lockstep byte shares).
- `Spec/*.lean`: the five P0 specs as `Prop`s over traces, each with a
  `Decidable` instance, plus a counterexample extractor and a note extractor for
  reporting.
- Verdicts are theorems closed by `native_decide` (T1: trusts the Lean compiler)
  for each concrete input in the case file. There are no symbolic obligations:
  integer semantics, wraparound included, are evaluated exactly on the concrete
  inputs.
- A spec FAILs only on a violation that is certain in the extracted model,
  whatever the data values. Findings that can be harmless depending on data or
  on how the kernel uses the result (a read past the work item's segment that is
  masked later, completions nobody waits for, transaction bytes pending after
  the last wait) are notes: reported, outside the theorems.
- A refuted theorem without a certain counterexample is not a FAIL
  (`SpecResult.certain`; Lean prints `UNDECIDED <input> <spec>`): every spec of an
  input whose evaluation stops with an error that is not a kernel error, a spec
  of an input with an out-of-bounds metadata load that finds nothing certain on
  the evaluated events, and P1-1 with only pairs that the lockstep contents or
  named barriers used with two counts might order. `lkv.py` reports it as
  UNDECIDED (a known limit), or as an ERROR when the evaluation error is
  unexpected (`Eval.lean` `ErrKind`: the case or LKV needs a fix).

### The five P0 specs

| P0  | Spec                      | FAIL                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Note                                                                                                                                                                                                        |
| --- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Global accesses in bounds | on an input that is not elided: an access outside its allocation (TMA boxes clipped per dim); a plain write outside the segment owned by its work item; an i32/i64 wrap on an address path (add, sub, mul, shl, lossy trunc); on every input: a metadata load outside its tensor                                                                                                                                                                                                                                                                                                                                                               | a read or reduction inside the allocation but outside the work item's segment                                                                                                                               |
| 2   | Role trip-count agreement | a malformed barrier (count <= 0, bar_init count != declared); a wait that needs a completion the kernel never produces (a stale wait counted at its lower bound), or that never returns when every arrival comes as early as the roles' waits allow (a wait cycle in the lockstep model); waited phases without lockstep contents (no fixed share per arriving role summing to the count, nor an initialization arrival forming phase 1; a role with no share whose arrivals may land in a waited phase); a role whose code waits on a barrier array having no wait, arrival or transaction event on a completing slot that receives TMA bytes | completions after a role's last wait (R < K, e.g. producer_tail, an empty barrier's final releases); a trailing partial phase nobody waits for; a static waiter missing a completing slot without TMA bytes |
| 3   | Output coverage           | on an input that is not elided: an output tensor without an `intended_writes` entry; an intended element written zero times or twice by plain stores; a plain store outside the intended region                                                                                                                                                                                                                                                                                                                                                                                                                                                | a reduction outside the intended region; an `unchecked` opt-out                                                                                                                                             |
| 4   | mbarrier protocol         | a parity other than 0/1; a stale parity (a repeated parity that neither provably re-checks the previous phase nor provably skips one, nor skips one under the lockstep credit); a first wait at phase 0 (parity 1 on the fresh barrier) on a slot that receives TMA bytes; under every fitting share assignment: in a waited phase, expect_tx bytes != bytes landing in it, TMA bytes landing in a phase nobody waits for or on a slot nobody waits on, a load not covered by an earlier expect_tx of its role in its phase (or by none at all), or transaction bytes that cannot be attributed to a phase                                     | a lap (a wait synchronizing with a later phase than the one after its previous wait); expect_tx bytes pending in a phase nobody waits for, or on a slot nobody waits on                                     |
| 5   | Mask-elision soundness    | an elided input (it sets one of the case's `elision_flags`, e.g. FULL_TILES) violates P0-1 or P0-3: its P0-1/P0-3 violations are reported here, not there                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | the P0-1/P0-3 notes of an elided input                                                                                                                                                                      |

**P1-1 TMEM reuse ordered** (`lean/KV/Spec/TmemReuse.lean`, P1 KIR only): holds
iff every pair of TMEM accesses of different roles that touch a common cell, one
writing, has one completing before the other is issued under sound mbarrier and
named-barrier happens-before between roles (the P1 ledger's phase lower bounds
and per-wait needs from the arrivals that may precede the wait's return, partial
credit of a multi-warp role's arrivals, plus commit landing; trust item 6), and
every access is inside the allocation and its warp's lane quarter. FAIL (F1) on
a pair unordered even under the most generous lockstep reading; undecided (U1)
on a pair only that reading orders; notes for same-role hazards, intra-role
cross-warp pairs and blocked waits (`specs.md`, "P1-1"). Reported apart; its
FAIL and UNDECIDED are part of the outcome and exit status only with
`--p1-strict`, its ERRORs (a failed P1 translation, a Lean error) always.

All specs are checked on every input of the case file. Each case file must
include degenerate inputs: a zero-length segment, partial last tiles, more work
items than SMs (small SM count), and every host-accepted flag combination the
case targets, and name in `elision_flags` the flags that select a mask-elided
fast path.

## Trust base

1. The compiler IR for the analyzed config (lowering below it is not checked),
   produced by the real host code under the frontend's host patches
   (`frontend_ttgir.md`, `frontend_cute.md`).
2. The op table: one Lean-facing meaning per IR op, written from the PTX ISA and
   the TLX/CuTe docs (`references/ir_ops.md`, `references/ir_ops_cute.md`),
   identical for every kernel. Its issuing-thread rows for CuTe TMA stores follow
   the SASS that ptxas 13.0 emits, and those for TMA loads and reduces the
   `elect.sync` the DSL emits (`ir_ops_cute.md`, "Issuing threads").
3. `native_decide` (Lean compiler) for whole-trace verdicts.
4. The phase ledger (`lean/KV/Spec/Ledger.lean`) for P0-2 and P0-4: its
   classification of waits (happens-before over the role traces, each role taken
   as one thread in lockstep), its lockstep phase contents (fixed per-role
   shares or an initialization arrival; an assumption, so the lockstep credit,
   each role's least share over all fitting assignments, only turns stale
   repeated parities into presumed skips, judged under the contents, and never
   raises a phase lower bound), its optimistic progress pass and its attribution
   of transaction bytes to phases (under every fitting share assignment until
   one satisfies P0-4). The Props are stated over the ledger's output, and the
   ledger's soundness against mbarrier semantics is argued informally in its
   module doc, not proved. A P0-2/P0-4 verdict therefore says the ledger found
   no violation.
5. For P1-1 only: the P1 op-table rows (TMEM footprints, the SF footprints of
   the CuTe DSL's TMEM SF layouts: SFA 4 columns, SFB 4 per 128 rows of N; the
   issuing threads; the cp/mma and the commits that complete them are issued by
   one elected lane) and the PTX completion rules it encodes (ld/st complete at
   `tcgen05.wait`, cp/mma at a landed later commit of the same thread).
6. For P1-1 only: the P1 ledger (`Run.ledgersP1`: item 4's ledger whose
   happens-before also uses per-wait needs, `k x count` minus the other roles'
   arrivals that may precede the wait's return, `SlotLedger.needsWith`; argued
   sound in Ledger.lean, not proved), named-barrier semantics (a thread blocked
   in `bar.sync` arrives once per phase) and the reading of a role of several
   warps on which every P1-1 PASS rests. The CuTe P1 pass makes every warp of a
   role issue each of its mbarrier waits, arrives and commits (it splits a role
   whose warps issue one only in part, e.g. a thread-0 arrive or wait, and
   refuses one under a runtime lane condition, R7). Its warps need not be at one
   point: an arrive counts as every warp's only if the phase needs more of its
   arrivals than all warps but one give, else only the role's previous arrive
   does (partial credit, `creditArrivals`), and the role's waits before the
   arrive holding the needed arrival returned. A warp of a role running so far
   ahead of another that its arrivals stand in for the other's is not modeled.
   Intra-role cross-warp pairs are notes. A P1-1 PASS uses no lockstep contents;
   they only tell F1 from undecided. tcgen05 fences and TMEM lifetime are not
   checked.

A verdict covers the inputs in the case file, not all shapes. The report says
so.
