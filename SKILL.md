---
name: lean-kernel-verify
description:
  'Mechanical safety checks of Blackwell GPU kernels (TLX/Triton, CuTe DSL)
  in Lean 4 with no hand-written kernel model: on a CPU-only machine the real
  host code and compiler produce the PTX and IR, a sliced op-table frontend
  turns one of them into KIR (TLX/Triton: the PTX by default), and Lean decides
  per case input, by native_decide,
  P0-1 bounds and address wrap, P0-2 role trip counts (hangs), P0-3 output
  coverage exactly once, P0-4 mbarrier protocol (parity, expect_tx bytes),
  P0-5 mask elision on host-selected fast paths and P0-6 named barriers
  convergent, plus, reported apart, P1-1 TMEM reuse ordering (CuTe) and P1-2
  async store sources reuse ordering (TLX/Triton: a TMA store source or
  tensormap rewritten too early). For
  checking a kernel revision or a fix for hangs, out-of-bounds or wrapped
  addresses, missing or duplicate writes, barrier bugs and reuse races on the
  shapes callers use (partial tiles, multi-wave grids), or writing a case file for a
  new kernel; verdicts cover the case inputs only.'
---

# lean-kernel-verify: compiler output to Lean verdicts for six P0 specs, P1-1 and P1-2

The skill checks a TLX/Triton or CuTe DSL kernel at a given revision against
six P0 safety specs (and, reported apart, two P1 specs), on the concrete inputs
of a case file, with no model of the kernel: the real host code and compiler
produce the PTX and IR on a CPU-only machine, a mechanical frontend turns one of
them into KIR (TLX/Triton: the PTX by default; CuTe: the IR), and Lean proves or
refutes each verdict as a `native_decide` theorem.

**Use it** to formally check a Blackwell kernel, or a fix against its base, for
hangs, out-of-bounds or wrapped addresses, missing or duplicate output writes,
mbarrier parity or expect_tx bugs, named barriers not all their warps reach alike
(P0-6), (CuTe) unordered TMEM reuse across warps (P1-1, e.g. a scale-factor copy
into a TMEM region that another warp is still loading) and (TLX/Triton) shared
memory or a tensormap rewritten while a TMA store may still read it (P1-2),
especially on partial and multi-wave shapes, including 2-CTA (cluster) kernels
for the P0 specs (CuTe; TLX/Triton in the PTX translation, also with cluster
launch control); or to add a case for a new kernel.
**Do not use it** for numerics, performance, CUDA C++ kernels, or general
race/deadlock freedom (see Limits).

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
  |   the PTX translation (frontend/ptx/) instead of the IR: the TLX/Triton default,
  |   CuTe with --frontend ptx; --frontend legacy translates the TLX/Triton TTGIR
  v
KIR JSON per compiled flag set: roles, structured control flow, exact-width integer exprs,
  |   events, inputs; frontend/kir.py (contract check) -> frontend/emit_lean.py
  |   every variant, whatever the translation: the barrier pass (frontend/ptx/nbar.py)
  |   writes the P0-6 KIR <spec>.nb.json / <kernel>__<digest>.nb.json from its PTX
  |   TLX/Triton, a launch with a preferred cluster: also <spec>@cluster<dims> KIRs, every
  |   cluster of the preferred size (P0-6; P0 and P1-2 in the PTX translation)
  |   CuTe: a second pass with the P1 slice writes <kernel>__<digest>.p1.json (+ TMEM,
  |   tcgen05.wait and named-barrier events); the P0 KIR does not change
  |   TLX/Triton, PTX translation: a second pass writes <spec>.p1.json (+ bulk store,
  |   group, shared-memory and tensormap write and named-barrier events; P1-2)
  v
Layer 2: lean/KV evaluates every role of every CTA per input -> event traces
  |   theorem : KV.Holds .<spec> kernel input := by native_decide   (P0 KIR: 5 specs
  |             x inputs; P0-6 KIR: .namedBarriers; P1 KIR: .tmemReuse / .storeReuse)
  v
report.md / report.json: verdict per (variant, input, spec), findings at source file:line;
  P0-6 with the P0 specs; P1-1 and P1-2 each in a section of its own
```

## Setup (no GPU needed)

- **Lean 4.34.0**, core Lean + Std only (no Mathlib; do not use `lake`).
  `scripts/lean.sh` takes the toolchain from `LEAN_TOOLCHAIN_DIR` (a directory with
  `bin/lean` and `lib/lean`), else the `lean` on `PATH` (e.g. from elan);
  `scripts/build_lean.sh` compiles `lean/KV` and the native `libKV.so` into
  `/tmp/lkv_lean_build_$USER/<hash>`, one directory per `lean/` sources and toolchain
  (`LKV_LEAN_BUILD` overrides it), and `lkv.py` runs it when needed.
- **The frontends run the kernel's real host code**, so they need its Python stack:
  a Triton with TLX and torch (TTGIR frontend, `frontend/ttgir/`), or
  nvidia-cutlass-dsl 4.4.2 or 4.6.1 exactly (the releases the CuTe op table was
  checked against; a CuTe case names its release in `dsl`, default 4.4.2) and torch
  (CuTe frontend, `frontend/cute/`), plus whatever the kernel package imports. At
  this revision `lkv.py run` builds and runs the frontends with Buck2 inside the
  monorepo that holds the skill; `lkv.py check` (Layer 2 on KIR files) runs anywhere.

## Run a case: scripts/lkv.py

```bash
# 1. SRC: a directory holding the kernel sources at the revision to check, from which the
#    case's `module` (TTGIR) or `source.package` (CuTe) imports.
# 2. Run: dump -> to_kir -> kir.py -> emit_lean -> Lean -> report.
python3 -I $S/scripts/lkv.py run CASE.json --src SRC
# 3. Read /tmp/lkv_runs/<case dir>/<case name>/report.md (or --out DIR).
python3 -I $S/scripts/lkv.py check K.kir.json... --out DIR   # Layer 2 alone on KIR files (distinct names)
```

- The frontend is CuTe if the case has a `builder`, else TTGIR. `run` options:
  `--inputs a,b`, `--target sm_103` (TTGIR only), `--jobs N` (Lean files at a time and
  the TTGIR frontend's P0-6 processes; default min(8, CPUs)),
  `--timeout S` (per Lean file, default 3600), `--src DIR` (default: the case's
  `src_root` / `source.src`), `--frontend ptx|legacy` (the Layer-1 translation: `ptx`
  translates the kernel's PTX with one core shared by both frontends, `frontend_ptx.md`;
  `legacy` its TTGIR / CuTe DSL MLIR; default `$LKV_FRONTEND` when set, else `ptx` for
  TLX/Triton cases and `legacy` for CuTe ones; report.json `translation` says which ran).
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
  out-of-bounds metadata load (itself a P0-1 FAIL) or at a model assumption that
  does not hold (KIR `assume`) when nothing certain is found for that spec (for an
  assumption, in the clusters where no role trace stopped at one), a
  division by zero, a barrier ledger that does not converge (P0-2, P0-4), another
  CTA's TMA bytes that balance only if a landing completes a later phase than its
  issue order shows (P0-4), a named barrier P0-6 cannot count or pair for certain
  (an `nb_open` not marked certain, a pairing that may depend on timing), P1-1
  pairs only the lockstep contents order, P1-2's undecided pairs U1-U4). A variant
  without verdicts gives every spec of each of its inputs its outcome (UNDECIDED if
  refused by design, else ERROR).
- Outcome and exit status of a run (P0-1..P0-6 and the P1 specs' ERRORs; the FAIL
  and UNDECIDED of P1-1 and P1-2 only with `--p1-strict`; `--no-p1` skips both):
  **1 FAIL** = some verdict FAILs, whatever else failed; **2 ERROR** = no FAIL, but
  something failed (also a pipeline error, no KIR at all, and any error of lkv.py
  itself, so 1 always means a certain violation); **0 UNDECIDED** = no FAIL and no
  ERROR, but a known limit leaves something open; **0 PASS** = every spec is
  proved on every input of every variant. Host-rejected inputs count as none:
  the exported sources raised for the input (a `raise` or `assert`, a failed
  torch check such as `torch._assert_async`) or launched nothing.
- `report.md`: header (case file, frontend, translation, target, rev, src, out),
  the result (outcome, exit status and why), the frontend's environment, PASS /
  FAIL / ERROR / UNDECIDED totals per spec (P0-6 with the P0 specs), verdicts per
  input (`FAIL (k/n)` = k of the n compiled variants that ran it have a certain
  violation, `ERROR (e/n)` = e of them failed, `UNDECIDED (u/n)` = u of them hit
  a known limit), a FAIL section (counterexamples by spec and source line, kernel
  errors), an ERROR section (variants that failed to translate or check,
  unexpected evaluation errors, INCONSISTENT verdicts), an UNDECIDED section
  (variants refused by design, model limits, specs refuted without a certain
  counterexample), notes, host-rejected inputs, the variants table (flags, Lean
  time), the P0-6 variants that rest on the warp-ahead reading and the barrier
  sites the barrier pass could not count, then a section per P1 spec (totals,
  verdicts per input, FAIL, ERROR, UNDECIDED, notes), the P1 variants and the
  variants the P1 specs do not check. Artifacts: `frontend/`, `lean/`, `logs/`.
- `report.json`: the same data, for scripts. CI reads `result.outcome` (or the exit
  status); every key and count: [references/specs.md](references/specs.md),
  "report.json".

## Cases

A case file names the kernel source, its public host entry point, small concrete
inputs and the annotations the specs need; its `description` says what the inputs
trigger. Cases for public kernels go in `cases/`; `tests/e2e/cases/` has a small
probe case per frontend and two TLX P1-2 cases (`store_reuse_staging_tlx`,
`store_reuse_rebuild_tlx`). [references/cases.md](references/cases.md) says how to
write a new case (source and public host entry point, inputs, annotations).
**Choose common inputs**, because verdicts cover only the case inputs: the shapes and
conditions the kernel's callers produce (partial last tiles, more work items than SMs
with odd and even tile counts, the callers' head counts and dtypes, the head dims the
host accepts, every host-selected variant the callers reach) and one aligned control
input. **Leave out very unlikely inputs unless the user asks for them**: an empty Q
(e.g. `q_len = 0` in flash attention) or another empty segment, a window of 2^31 - 1 or
a negative one, head counts the callers rule out, misuse that the docs or the public
wrapper rule out, and opt-in modes nobody uses.

## The P0 specs

| P0  | Spec                         | FAIL (violation certain in the extracted model)                                                                                                                                                                                                                                                                                                                                                                                    | NOTE (may be harmless; never a FAIL)                                                                                                 |
| --- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | Global accesses in bounds    | a masked-in access outside its allocation (TMA boxes clipped per dim); a plain write outside its work item's segment; an i32/i64 wrap on an address path (add/sub/mul/shl, lossy trunc); a metadata load outside its tensor (on every input)                                                                                                                                                                                       | a read or reduction inside the allocation but outside the segment                                                                    |
| 2   | Role trip-count agreement    | a malformed barrier; a wait that needs a completion that never comes, or a wait cycle in the lockstep model (hang); waited phases whose arrivals do not split into one fixed share per arriving role summing to the count (or follow one initialization arrival); a static waiter of a barrier array with no event on a slot that completes and receives TMA bytes                                                                 | completions after the last wait (`producer_tail`); a trailing partial phase; a missing waiter on a completing slot without TMA bytes |
| 3   | Output coverage exactly once | an output without `intended_writes`; an intended element that plain stores write 0 times or 2+ times (over all CTAs); a plain store outside the intended region                                                                                                                                                                                                                                                                    | reductions outside the region; `unchecked` opt-outs                                                                                  |
| 4   | mbarrier protocol            | a parity other than 0/1; a stale repeated parity; a first wait that synchronizes with phase 0 on a TMA slot (parity 1 on the fresh barrier, unless a late join moves it); in a waited phase, expect_tx bytes != landing bytes; TMA bytes landing in a phase or on a slot nobody waits on; a load not covered by an earlier expect_tx of its role (in its phase, or at all); transaction bytes that cannot be attributed to a phase | laps; expect_tx bytes pending in a phase nobody waits for, or on a slot nobody waits on                                              |
| 5   | Mask-elision soundness       | an elided input (it sets one of the case's `elision_flags`) violates P0-1 or P0-3 on the variant the host selected; those findings are reported under P0-5 instead                                                                                                                                                                                                                                                                 | the P0-1/P0-3 notes of an elided input                                                                                               |
| 6   | Named barriers convergent    | (decided on the P0-6 KIR, `.nb.json`, made from every variant's PTX whatever the translation; `references/specs.md` "P0-6") a named barrier (`bar.sync`/`bar.arrive`/`bar.red`) reached under a condition its participants may not share: thread-index values that differ (F1), a loop whose exit differs, e.g. a spin loop on a strong load (F2); a count no warps meet, ids or counts that differ between the warps that meet, a sync that waits forever in the pairing (F3); UNDECIDED where uniformity or the pairing is not decided (arrive-based pairings rest on the warp-ahead reading) | none                                                                                                                                 |

### P1-1 TMEM reuse ordered (CuTe only; `references/specs.md`, "P1-1")

| P1  | Spec                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | FAIL / undecided                                                                                                                                                                                                                                                                     | NOTE                                                                                                                          |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| 1   | TMEM reuse ordered: every pair of tcgen05 ld/st/cp/mma accesses of different roles that touch a common TMEM cell (lane x 32-bit column), one writing, is ordered: one completes (ld/st at the role's next `tcgen05.wait::ld/st`, cp/mma at a landed later commit of the same warp) before the other is issued, under sound mbarrier and named-barrier happens-before between roles (trust item 6, with partial credit of a multi-warp role's arrivals; no lockstep contents) | F1 a conflicting cross-role pair unordered even under the most generous lockstep reading; F2 an access outside the allocation; F3 an ld/st outside its warp's lane quarter; undecided (U1): a pair ordered only under the ledger's lockstep contents, with no certain counterexample | N1 same-role hazards that are not PTX pipelined pairs; N2 intra-role cross-warp pairs no `bar.sync` orders; N3 a blocked wait |

Decided on the P1 KIR, reported in its own report section and `report.json`
`variants_p1`; a refused P1 translation (R1-R9, `frontend_cute.md`; R10 in the PTX
translation, `frontend_ptx.md`) is UNDECIDED on every input of its variant (a failed
one an ERROR), and P1-1 of TTGIR variants is "not checked". A PASS rests on sound
happens-before between roles (trust item 6): a wait that returned for phase k
needs, of each arriving role, k x count minus the other roles' arrivals that may
precede its return (P1 ledger, `SlotLedger.needsWith`), and those show every warp
of a role issued an arrive only if they take more of its arrivals than all warps
but one give (partial credit). The P1 pass splits a role whose warps issue an
mbarrier wait, arrive or commit only in part (a thread-0 arrive or wait), so each
such event of a role is issued by all its warps.

### P1-2 async store sources reuse ordered (PTX translation; `references/specs.md`, "P1-2")

| P1  | Spec | FAIL / undecided | NOTE |
| --- | ---- | ---------------- | ---- |
| 2   | Async store sources reuse ordered: every write of shared memory overlapping the source of a bulk async store (`cp.async.bulk.tensor` / `cp.reduce.async.bulk.tensor` / `cp{.reduce}.async.bulk` global <- shared::cta, bulk_group), and every write of the tensormap of an in-kernel descriptor it reads (`tensormap.cp_fenceproxy`), is ordered: issued after a `cp.async.bulk.wait_group{.read}` of the store's issuing thread completed its group (`.read` covers the tensormap read: PTX ISA, `wait_group`) and that wait happens before it (mbarrier and exact named-barrier happens-before, the issuer's program order), or happens before the store is issued | F1 a shared-memory write, F2 a tensormap write (also one by another cluster), unordered even in the lockstep reading; undecided: U1 ordered only in the lockstep reading, U2 by an issuer program order the KIR does not pin down (for another lane of a fixed-lane issuer's warp: with a `bar.warp.sync`, which the KIR drops), U3 a load's landing issued before the store that its mbarrier's bytes do not show landed before the store is issued (trust item 8), U4 a store or write after a wait the ledger finds blocked | N1 a store its issuer never waits for; N2 a blocked wait |

Decided on the P1 KIR `<spec>.p1.json` the TTGIR frontend writes in the PTX
translation (the TLX/Triton default) for a specialization with a bulk store, and
for a launch with a preferred cluster also `<spec>@cluster<dims>.p1.json` (all
clusters preferred) (`frontend_ptx.md`, "P1-2 pass"; top-level `p1: ["P1-2"]`),
reported in a section of its own after P1-1's (FAIL and UNDECIDED count only with
`--p1-strict`). A refused P1-2 translation is UNDECIDED on every input of its
specialization (of an all-preferred geometry: those whose grid its dims divide), with
its reason (a failed one an ERROR); a specialization without a bulk store is "not
applicable", and the legacy TTGIR translation and the CuTe frontend do not check P1-2
("not checked").

**How a verdict is decided.** Each host flag set compiles to its own KIR
variant, paired with the inputs that select it. For each (variant, input, spec),
lkv proves or refutes `KV.Holds .<spec> kernel input` by `native_decide`, over
an exact `BitVec` evaluation of the KIR: PASS iff the theorem is proved; a
refuted theorem is a FAIL if Lean lists a certain counterexample for it (or the
evaluation stopped at a certain kernel error), an ERROR if LKV failed to decide
it, else UNDECIDED (a known limit). Roles are not interleaved: P0-2 and P0-4 are
stated over the phase ledger of each cluster (of each CTA without clusters;
`lean/KV/Spec/Ledger.lean`).

## Trust base and reporting

A verdict depends on exactly these items (`design.md`, "Trust base"):

1. **The compiler output** for the analyzed config, produced by the real host code at
   REV under the frontend's host patches (device queries, device placement,
   compile/launch interception, the case's `env`, tabled in the frontend docs): the
   kernel's PTX in the PTX translation (the TLX/Triton default; CuTe with
   `--frontend ptx`), else its IR: TTGIR for `GPUTarget("cuda", 100|103)` (TLX/Triton
   with `--frontend legacy`) or the CuTe `cute-to-nvvm` prefix stage at `sm_100a` with
   the case's nvidia-cutlass-dsl (4.4.2 or 4.6.1). Nothing below it is checked.
2. **The op tables**: `ir_ops_ptx.md` in the PTX translation (its TMA and bulk-copy
   issuing rules are measured on hardware per compiler version), else `ir_ops.md` /
   `ir_ops_cute.md`: one meaning per op, the same for every kernel. Unknown ops are
   hard errors, and so are data values that reach an address, a mask or a barrier
   operand (slice soundness). Issue counts follow what ptxas emits: a CuTe TMA store
   executed by a whole warp is one write per distinct (coordinates, shared-memory
   source) among its lanes (ptxas issues it in a waterfall loop; DSL 4.4.2 compiles
   with its embedded ptxas 12.9.83); a TMA load or reduce is one per executing lane,
   which is one per warp because the DSL emits it behind `elect.sync`. A TMA load
   through an in-kernel descriptor (PTX translation) reads the descriptor its role's
   `tensormap.cp_fenceproxy` wrote: assumed, not checked, that no other role rewrites
   that slot and that its role does not while a load through it is in flight (P1-2
   checks that for stores).
3. **`native_decide` (T1)**: the Lean compiler and the native `libKV.so`.
4. **The ledger algorithm** for P0-2 and P0-4, argued sound in Ledger.lean but
   not proved: a PASS there means "the ledger found no violation".
5. **P1-1 only:** the P1 op-table rows (TMEM footprints, the scale-factor
   footprints of the CuTe DSL's TMEM SF layouts, cp/mma issued by one elected lane,
   checked in the DSL's PTX, and their commit by the same lane, which the kernel's
   `elect_one` or a whole-warp commit gives: not checked) and the PTX completion rules
   it encodes.
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
7. **P0-6 only:** the barrier pass's uniformity rules (`specs.md` "P0-6": exact
   thread-index instances; weak shared-memory loads at a uniform address written
   before a synchronization every reader passed; the value a spin loop accepted, as
   loaded, published once by its producer; weak loads of tensors the kernel never
   writes, told apart by name: no other tensor argument shares a written one's
   storage), its node
   splitting (copies of PTX blocks), the recorded contents of metadata tensors, and
   for arrive-based protocols the warp-ahead reading (no role runs a whole instance
   of a named barrier ahead of the others; the report counts the arrivals it holds or
   may hold), and the PTX ISA's rule that threads that exit release a barrier without a
   thread count (`bar.sync a`).
8. **P1-2 only:** the P1 ledger and the role splitting of item 6; the PTX ISA
   reading that `cp.async.bulk.wait_group.read` also covers the store's tensormap
   read; `elect.sync` electing the same lane for the same membermask (a store and a
   write of one elected lane are one thread's); the frontend's write set (every PTX
   instruction that writes shared memory is a `smem_write`, refused, or provably
   disjoint from every store source, and every global write of a buffer that holds a
   store's tensormap a `tmap_write` or refused, `frontend_ptx.md` "P1-2 pass"); and the
   participants of a named barrier being its roles' threads, all of the CTA's for a
   barrier whose count is the CTA's (idle warps included). A load's landing issued
   before the store is ordered only through its mbarrier's bytes (a phase completes
   only with its expected bytes landed, an `expect_tx` applies to the phase current at
   its issue: `StoreReuse.landed`), else undecided (U3).
9. **Handoffs only** (kir.md change 17): the parts of one role (a role split by warps)
   arriving in lockstep, as P0 takes the threads of one role; the frontend's write set
   for a channel (every other shared write that may touch its bytes is refused, or
   assumed to miss them where its offsets have no static bounds: a KIR `assume`).

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
  6). A P0-6 FAIL names the barrier and the condition, loop or count its warps do
  not share: confirm it in the source; a P0-6 PASS of a variant report.md lists
  under "P0-6 under the warp-ahead reading" assumes no role runs a whole instance
  of a named barrier ahead of the others (trust item 7). A P1-2 FAIL is a write of
  a bulk store's source (F1) or tensormap (F2) that no
  `cp.async.bulk.wait_group{.read}` of the store's thread orders in the model:
  confirm the missing wait or barrier in the source; a P1-2 PASS covers
  store-source reuse only (trust item 8). Without a GPU run, say "not reproduced
  on hardware".

## Limits

Clusters (CuTe, and TLX/Triton in the PTX translation, also with cluster launch control):
remote mbarrier arrives, multicast commits, cta_group::2 and multicast TMA loads, DSMEM
bulk copies, `st.async`/`red.async` into another CTA and the cluster barrier are
checked with one barrier ledger per cluster, and P1-1 and P1-2 run per cluster (CuTe: a
cta_group::2 MMA or copy accesses the TMEM of both CTAs of its pair); which CTA's data a
copy carries is not checked. A TLX/Triton launch with a preferred cluster is checked in
both geometries (all regular, all preferred: P0, P0-6 and P1-2 KIRs each; the legacy
TTGIR translation refuses its P0 KIR, checking one geometry); not covered: P0-3 of a
launch that mixes them and P1-2's tensormap check across clusters of the two
geometries. Refused: the all-preferred P0 and P1-2 KIRs of a kernel with CLC, the
all-preferred KIRs of launches whose preferred cluster breaks a rule CUDA states for it
(no regular cluster, a preferred x or y that is no multiple of the regular one or another
z, dims that do not divide the grid), a CuTe launch with a preferred cluster and a TLX
launch whose function's required cluster differs from its cluster (both with their P0-6
KIRs). A
warp-specialize capture (a value passed to the partitions through shared memory) is
forwarded only where every thread of the default stores it and no shared write may
overwrite it before the partitions read it
(a strength-reduced offset: assumed per input by a KIR `assume`, which leaves the input
undecided where it does not hold). A TMA load through an in-kernel descriptor assumes
that no other role rewrites the descriptor's slot and that its role does not while the
load is in flight (trust item 2). The PTX translation refuses some kernels and
constructs the legacy translations handle (`frontend_ptx.md`, "Refusals", "Limitations");
`--frontend legacy` runs the TTGIR translation of a TLX/Triton kernel.
Refused in every translation: im2col TMA, CuTe CLC and CLC with a multi-dimensional grid;
in the legacy TTGIR translation: 2-CTA kernels (remote barriers, DSMEM, multi-CTA
layouts), launches with a preferred cluster, loop-carried pointers, side-effecting
inline asm and a per-CTA tile count the partitions capture from an `scf.if`; in the
legacy CuTe translation: a loop-carried value changing structure (e.g. a pointer into
another tensor; the PTX translation: a loop-carried or merged address that changes its
base) and a value that reaches an event through register memory.
Not covered: completion/wait
work-item correspondence, P1 bug classes other than P1-1 and P1-2 (P1-1 on TTGIR/TLX,
P1-2 on CuTe and on the legacy TTGIR translation, the source of a DSMEM copy into
another CTA,
TMEM lifetime, tcgen05 fences, a warp of one role running ahead so that its
arrivals stand in for another's), inputs and variants no case selects. CuTe
supports sm_100 with DSL 4.4.2 and 4.6.1. Details and the spec-model limits:
[references/specs.md](references/specs.md), "Limits".

## References

The references detail the pipeline above, one piece each
([references/design.md](references/design.md) is the architecture they detail; the
table says when to read each):

- case file: [cases.md](references/cases.md) (how to write one; examples in
  `tests/e2e/cases/`), with
  its fields per frontend in [frontend_ttgir.md](references/frontend_ttgir.md) "Case
  file" and [frontend_cute.md](references/frontend_cute.md) "Case files";
- Layer 1, IR to KIR: [frontend_ttgir.md](references/frontend_ttgir.md) and
  [frontend_cute.md](references/frontend_cute.md) (the dump, host patches, translation,
  refusals), with the op tables [ir_ops.md](references/ir_ops.md) (TTGIR) and
  [ir_ops_cute.md](references/ir_ops_cute.md) (CuTe): which IR op becomes which KIR event
  or value; the shared PTX translation (the TLX/Triton default, CuTe with
  `--frontend ptx`) [frontend_ptx.md](references/frontend_ptx.md) and its op table
  [ir_ops_ptx.md](references/ir_ops_ptx.md);
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
| [references/frontend_ptx.md](references/frontend_ptx.md)                                  | a TLX/Triton run (PTX translation by default) or CuTe with `--frontend ptx`: how the PTX becomes roles and events, work anchors, how to read a refusal |
| [references/ir_ops_ptx.md](references/ir_ops_ptx.md)                                      | in the PTX translation: an instruction is unknown or refused, its events and issuing threads (TMA, sub-byte)                  |
| [references/cases.md](references/cases.md)                                                | choosing a case for a kernel, or writing a new case (fields, annotations, which inputs to choose)                             |
| [references/bug_taxonomy.md](references/bug_taxonomy.md)                                  | someone asks what is covered and why, or which spec to add next (evidence, P1/P2, test-side complements)                      |

## Tests

```bash
$S/tests/lean/run_all.sh                    # Layer 2: positive/negative controls per spec (tests/lean/p1: P1-1, p1_2: P1-2, nb: P0-6)
python3 -I $S/tests/e2e/test_e2e.py         # lkv.py end to end, ~2 min (frontends + Lean, no GPU)
python3 -I $S/tests/cute/test_core.py       # CuTe translation, pure Python
python3 -I $S/tests/ptx/test_translate.py   # PTX translation, pure Python (and tests/ptx/test_*.py)
# Frontend tests: tests/ttgir/test_translate.py, test_ttgir.py, test_ptx_capture.py (Triton)
# and tests/cute/test_frontend.py (CuTe DSL, run once per supported DSL release),
# test_ptx_capture.py; tests/ttgir/test_gpu_parity.py and
# tests/cute/test_gpu_parity.py: CPU-dumped TTGIR / CuTe MLIR == real launch (need a Blackwell GPU)
```
