# Changelog

All notable changes to this project are documented in this file.

## Unreleased

First public release.

- Six P0 safety specs, decided in Lean 4.34.0 by `native_decide` on the concrete inputs
  of a case file: global accesses in bounds (P0-1), role trip-count agreement (P0-2),
  output coverage exactly once (P0-3), mbarrier protocol (P0-4), mask-elision
  soundness (P0-5) and named barriers convergent (P0-6, decided on the kernel's PTX in
  both families). For CuTe DSL kernels, also TMEM reuse ordered (P1-1); for TLX/Triton
  kernels in the PTX translation, async store sources reuse ordered (P1-2).
- Frontends for TLX/Triton (TTGIR) and the CuTe DSL (`nvidia-cutlass-dsl` 4.4.2 and
  4.6.1, the release a case names) that run the kernel's real host code and compiler
  on a CPU and translate the IR to KIR with a fixed op table.
- `scripts/lkv.py`: `run` (case file to report), `check` (the Lean layer on KIR files)
  and `export`; `run` and `export` still need the monorepo LKV is developed in (see the
  README's Status section).
- Tests: the Lean layer's KIR fixtures (`tests/lean`), the frontends' tests, the PTX
  translation's tests (`tests/ptx`) and the end-to-end tests of `lkv.py`.
- KIR `ir` accepts `"ptx"` (KIR translated from PTX; references/kir.md change 12).
- KIR changes 14 (the P0-6 KIR: top-level `nbar`, events `named_bar`, `nb_open`,
  `nb_rep`), 15 (P1-2: top-level `p1` and `cta_threads`, events `bulk_store`,
  `bulk_commit`, `bulk_wait`, `smem_write`, `tmap_write`) and 16 (the statement
  `assume`, a model assumption: false on an input, its role stops there and the input
  is undecided). `lkv.py` reports P0-6 with the P0 specs (report.json `variants_nb`,
  `nb_not_checked`; report.md "P0-6 under the warp-ahead reading" and "P0-6 barrier
  sites not counted") and each P1 spec in a section of its own; `--p1-strict` and
  `--no-p1` apply to both P1 specs.
- The PTX translation (the default for TLX/Triton kernels; `--frontend ptx` for CuTe
  kernels, which keep their IR translation by default; `--frontend legacy` translates
  the TTGIR; `$LKV_FRONTEND` sets the translation of every run without `--frontend`)
  makes both frontends
  translate the kernel's PTX instead of their IR, with one shared core (`frontend/ptx/`)
  and one PTX op table (references/frontend_ptx.md, ir_ops_ptx.md), into KIR with `ir`
  `"ptx"`: global and metadata accesses, mbarriers, TMA and bulk copies (packed sub-byte
  formats too), tcgen05 commits, CLC (TLX, also in clusters), clusters and 2-CTA kernels
  of both families (a preferred cluster geometry as a second, all-preferred KIR, for
  P0-6 and P1-2 too; its P0 and P1-2 KIRs are refused with CLC, and all three where the
  launches break a rule CUDA states for the preferred dims) and the P1 passes (CuTe:
  P1-1; TLX/Triton: P1-2). Refusals name the PTX instruction, its source line and its
  PTX line.
  report.json `translation` and a line of report.md name the translation of a run. With
  CuTe DSL 4.4.2 the source locations are best effort, as report.json `loc_quality` and
  report.md say.
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
- PTX translation, more forms translated (each was refused): Triton's direct
  warp-specialize dispatch (`tlx.async_tasks(exclusive=True)`: no state bytes, roles
  `default` and `partition<k>` by first warp); a `brx.idx` jump table that is no
  dispatch, as a compare chain whose index must be provably in range (in the P0-6
  barrier pass too); values the default passes to the partitions through shared memory
  (warp-specialize captures, forwarded only where every thread of the default stores
  them and no shared write may overwrite them);
  `ld.param` through a select of parameter addresses, and global accesses through a
  select of two tensors; TMA loads through TLX in-kernel descriptors; `st.async`,
  `red.async` and the `shared::cluster` bulk reduce into another CTA of the cluster;
  mbarrier addresses through an induction variable or a select between two arrays;
  work fields LLVM hoists out of the work loop.
- A CuTe launch with a preferred cluster (both CuTe translations) and a TLX/Triton
  launch whose function's required cluster (`cuFuncSetAttribute`) differs from its
  cluster dims (both TTGIR translations) are refused: the geometry LKV would check may
  not be the one the hardware runs (their P0-6 KIRs too). The legacy TTGIR translation
  refuses a launch with a preferred cluster: it checks one geometry.
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
- P0-2/P0-4: the phase ledger must reach a fixpoint: a round that changes neither the
  phase bounds, any wait's classification nor the lockstep contents. A cluster whose
  ledger does not converge (also a 2-cycle) is no longer judged by its last round, whose
  verdict could depend on the parity of an unrelated role's trip count: P0-2 and P0-4
  list one `not decided` line for it and are UNDECIDED unless another cluster has a
  certain counterexample; P1-1's F1 pairs there are not certain either
  (`tests/lean/ledger_cycle`, `ledger_confirm`, `p1/ledger_cycle`).
- P0-4: transaction bytes that another CTA of the cluster lands on a slot (the other CTA
  of a cta_group::2 pair loading into the leader's barrier, a multicast destination, a
  DSMEM copy) are attributed by that role's issue order: each landing goes to the first
  phase it can complete (one after the highest phase of the slot that surely completed
  before it is issued), no longer to a fixed byte share per phase. A 2-CTA pipeline whose
  first phase also carries a prologue load (per-phase peer bytes 49152, 16384, 16384, ...)
  was a certain FAIL with 3 or more phases on a slot; it now PASSes when every phase
  balances. A failure is certain when a phase lacks bytes, the totals differ or a landing
  can only land in a phase nobody waits for; when only the lower bounds fail (a phase
  receives more bytes than it expects while the totals match), P0-4 is UNDECIDED with a
  `not decided` line (`tests/lean/cluster_pair_prologue`). So are copies ordered after the
  phases before them only by a named barrier, which P0 does not model (e.g. a relay warp
  exchanging DSMEM copies with its peer CTA), or only by transaction bytes (a multicast
  pipeline over more than 2 CTAs), which the byte share used to PASS. "Surely completed"
  uses the precise needs of P1-1, so a multicast pipeline over 2 CTAs, whose empty slots
  two consumers release, PASSes (`tests/lean/cluster_multicast_pipeline`). With several
  fitting arrival share assignments, a slot is not decided as soon as one of them fails
  only through the lower bounds, and lists no transaction line as certain
  (`tests/lean/cluster_lander_shares`).
- P0-4, another CTA's landings: a wait for phase k returns only after the loads and
  landings that bring the expect_tx bytes surely preceding phase k's completion were
  issued, counting only loads the wait's own return does not surely precede
  (`lean/KV/Spec/Ledger.lean`, "Bytes carry order"; the landers' floors only). TLX CLC
  in 2-CTA clusters (each try_cancel response multicast into the other CTA) and
  cta_group::2 pipelines whose empty slot a second role also releases now PASS where
  P0-4 was UNDECIDED (`tests/lean/cluster_clc_response`, `cluster_pair_two_release`).
- A negative `expect_tx` or `complete_tx` byte count is an evaluation error (`unexpected`):
  PTX transaction counts are unsigned, and P0-4's attribution by issue order needs them
  non-negative (`tests/lean/cluster_dsmem_relay` `negative_bytes`).
- P0-2: on a barrier array that some role's code initializes, each slot a CTA uses needs
  exactly one executed `bar_init` there; a slot used without one or initialized twice is
  a malformed barrier (references/kir.md change 13; `tests/lean/bar_init_once`).
- CuTe frontend: `shfl.sync`, `vote.sync` and `make_warp_uniform` on the slice need every
  lane of each warp that reaches them, as `elect.sync` does (a full-mask collective
  executed by part of a warp is undefined in PTX).
- TTGIR frontend: two mbarrier arrays, or two in-kernel descriptors, printed with one SSA
  name in sibling regions are refused instead of merged into one KIR name.
- TTGIR frontend: `tt.approx_divf` (a float division newer Triton builds emit) is float
  math like `tt.precise_divf`, dropped and kept off the slice.
- TTGIR frontend: SSA values that print with one name (MLIR's printer numbers sibling
  regions alike: an `scf.while`'s before and after regions, the branches of an `scf.if`,
  consecutive loop bodies) share a KIR variable only where that is exact
  (`frontend/ttgir/var_names.py`); otherwise the later one is named `<name>@<TTGIR line>`.
  Such a pair of different widths was a KIR contract error ("assignment changes the width
  of ..."). KIR with no such pair is unchanged.
- TTGIR frontend: an `scf.while` that carries no values (printed `scf.while : () -> () {`,
  e.g. a spin loop on a flag) is parsed; it was a parse error.
- TTGIR dump: `torch.cuda.current_stream()` answers device 0's default stream (handle 0),
  the stream of every launch of the CPU run; any other use of the stream API still fails
  the dump.
- TTGIR dump: `with torch.cuda.device(device):` is a no-op for device 0 (and its CPU
  stand-in); another device fails the dump.
- TTGIR dump: `kernel.warmup(...)` returns the compiled kernel (with a stand-in function
  handle) instead of None, still without counting as a launch, and host code may set the
  required cluster dims of that kernel through `cuFuncSetAttribute` of
  `ctypes.CDLL("libcuda.so.1")` or `ctypes.cdll.LoadLibrary("libcuda.so.1")` (other
  driver calls and attributes fail the dump): later launches of the kernel record them in
  their LaunchRecord, `attrs.func_attributes` (`{"required_cluster": [x, y, z]}`). A
  launch of a kernel whose handle the host loaded without setting an attribute that way
  fails the dump.
- PTX translation: a TLX work field the compiler folded away in a role (no register
  holds its value there) is refused only in a role that accesses a tensor whose segments
  it names, the only accesses whose check reads it; elsewhere it is -1.
- Docs: CuTe DSL 4.4.2 compiles with the PTX compiler it embeds (12.9.83), not CUDA
  13.0's ptxas; a TMA reduce is once per warp only behind the DSL's `elect.sync`.
- PTX translation: a TLX/Triton warp-specialize dispatch whose default may run no round
  (it decides once whether to store the state table or only the exit state before one
  fork barrier, e.g. an early exit when no work is left) is translated: each worker's
  region runs under the default's decision, which must be a value of the launch
  (frontend_ptx.md). It was refused ("the fork barriers ... depend on the path").
- PTX translation: a warp-specialize capture the default stores in a branch on a
  uniform runtime condition or in a loop is no longer forwarded to the workers as if it
  always ran (it may not); the workers' loads of it are data.
- Lean: one evaluation per input. emit_lean.py writes one `#lkv_check kernel <input>
  <spec>...` per input (lean/KV/Command.lean): the compiled evaluator runs the input once,
  prints the report and proves, per spec, `KV.Holds` or its negation from that one run,
  with the trust of `native_decide` (which evaluates its decision the same way); before,
  each spec's `native_decide` theorem evaluated the input again. `emit_lean.py
  --legacy-theorems` keeps the earlier form and lkv.py reads both (THEOREM lines, or
  refuted theorems). A spec's decision runs only that spec (`SpecId.holdsB`), the signed
  overflow test of `wraps` uses the sign rule instead of big integers, and the heartbeat
  limit is lifted for large KIRs. Verdicts are unchanged.
- Lean: faster evaluation, same verdicts and reports. `BitVec` operations without a
  big-number power per operation (`lean/KV/BV.lean`, each proved equal to the core one),
  lane loops without per-lane state, one computation of the analyses the report and the
  decisions share (barrier and P1 ledgers, P0-1's, P0-3's and P0-5's decisions, P0-6's
  and P0-7's analyses; `checkAll_snd` is a proof), and parallel Lean tasks for independent
  CTAs (with sync tensors, the CTAs of each pass), cluster ledgers and P0-1's per-CTA checks
  (`LKV_LEAN_THREADS`; lkv.py: its CPUs divided by `--jobs`, at least 2).
  `tests/lean/run_all.sh` checks the lane loops against the loops they replace
  (`tests/lean/fast_paths.lean`) and, on every fixture input, the evaluation against the
  sequential one with those loops (`lean/KV/SelfCheck.lean`, `emit_lean.py --selfcheck`);
  `lean/KV/Profile.lean` (`emit_lean.py --profile`) times the phases of an evaluation.
- emit_lean.py writes an Expr or TExpr whose Lean text is longer than 256 characters once,
  as a definition `sx<n>` used at every place it recurs (a large repeated subterm is
  elaborated once; the values, and so the verdicts, are the same).
- Lean: P0-1's decision computes a tensor's span and the work item's segment once per
  access instead of once per lane (`Bounds.eventOkB_eq`); P0-3's decision reads only the
  offsets a store writes (`Coverage.writeOkB_iff`, `Coverage.okB_eq`), and its targets and
  plain-store ledgers serve the decision and the report lines alike; P1-1's and P1-2's pair
  searches run per CTA or cluster in parallel and serve the decision and the report alike
  (`TmemReuse.holdsB_eq`, `StoreReuse.holdsOnB_eq`). The self-check also compares every
  spec's report lines with the ones computed from scratch. Same verdicts and reports.
- TTGIR to_kir: the P0, P0-6 and P1-2 translations of every specialization and cluster
  geometry are tasks of one fork pool, up to `--jobs` processes (one KIR each; `lkv.py
  run --jobs`), so the specializations of a kernel translate in parallel; each process
  sends back only its log line's contents. The three passes check the same geometries,
  decided before anything is translated (`to_kir.geometries`): the P0 pass also
  translates the all-preferred geometry where the regular P0 KIR fails (its own line,
  `<spec>@cluster<dims>`, or the refused `<spec>@cluster`), and a launch whose function's
  required cluster differs from its cluster dims refuses each geometry's P0 KIR, as it
  did the P0-6 and P1-2 ones. Each translation parses its PTX once. KIRs and verdicts
  are unchanged; the log lines come in task order.
- CuTe to_kir: the P0, P0-6 and P1 translations of the variants run in up to `--jobs`
  forked processes (`lkv_cute to_kir --jobs`, `lkv.py run --jobs`), and `lkv.py run` runs
  the CuTe dump and to_kir as two steps, as for TLX/Triton (`logs/dump.log`,
  `logs/to_kir.log`; `lkv_cute run` still does both in one process, its passes one after
  another). The PTX assembly rerolls LLVM's unrolled blocks from per-statement keys
  computed once. KIRs and verdicts are unchanged.
- TTGIR to_kir: the P1-2 pass of a kernel without a bulk store is `not_applicable` as
  soon as its PTX is parsed, also where its translation would fail (that was a P1-2
  `refused` or `error`). `lkv.py run --no-p1` makes the frontends skip their P1 pass
  (`LKV_P1=0` for the TTGIR to_kir, `--no-p1` for the CuTe binary) and sets `LKV_P1=1`
  otherwise: an `LKV_P1=0` in the shell made every TLX/Triton P1-2 an ERROR.
- P0-6 (both frontends): the metadata tensors of the barrier pass are chosen before it
  runs, with its write set (`nbar.writes`), so it runs once per P0-6 KIR, on the PTX
  parsed once; it ran again for each tensor the input checks rejected. KIRs are
  unchanged.
- TTGIR to_kir writes `report.json` next to dump.json: one entry per variant (also each
  cluster geometry) with the inputs of its specialization and its P0, P0-6 and P1-2
  results, as the CuTe frontend's `kir/report.json`, and `lkv.py` reads both frontends'
  alike instead of to_kir's log lines (which stay, for humans). It is written before the
  first task, every result still to come an error, and again with each line, so a to_kir
  that stops leaves the results it logged (a stop failed the whole run, unless a P0 KIR
  had failed, and could leave a geometry's P1-2 neither checked nor listed); no
  `report.json` is a pipeline error.
- PTX translation: `tcgen05.mma`'s `.collector::a::{fill,use,lastuse,discard}` (where the
  tensor core keeps A between MMAs) has the row of the MMA without it; today's Triton
  writes `fill`/`lastuse` pairs, which the op table refused (every PTX variant of such a
  kernel an ERROR). `use` and `lastuse` read the copy of A an earlier `fill` kept: P1-1
  takes them as reads of A's operand (more reads than the hardware makes).
