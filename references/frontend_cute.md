# CuTe DSL frontend (Layer 1)

`frontend/cute/` turns a CuTe DSL kernel plus a case file into KIR (`kir.md`) with no
hand-written model of the kernel: the real host code runs on a CPU-only machine, the
real CuTe compiler produces the IR, and a fixed op table (`ir_ops_cute.md`) translates it.
Tests: `tests/cute/`; a minimal case: `tests/e2e/cases/probe_cute.json`.

```
exported sources + case file
  | dump     pinned environment, host code on CPU (host patches below); every compile: initial MLIR +
  |          the DSL's own pass pipeline; every launch: tensor dtype/shape/stride/storage,
  |          integer tensor contents, scalars
  v
analyzed stage   prefix of the DSL's cute-to-nvvm pipeline (stage.py), generic MLIR text
  | to_kir   slice (slicer.py) + translation per role (interp.py, ops.py, ops_mem.py)
  v
KIR JSON         validated with frontend/kir.py, written only if valid, one file per
                 compiled kernel variant
  | P1 pass  the same translation again with the P1 slice (ops_tmem.py): TMEM, tcgen05.wait
  |          and named-barrier events
  v
P1 KIR JSON      <Kernel>__<digest>.p1.json (P1-1 TMEM reuse ordered); the P0 KIR is unchanged
```

## Running

```bash
# frontend, contract check, Lean checks and report (outputs in /tmp/lkv_runs/<case dir>/<case>/)
python3 -I scripts/lkv.py run CASE.json --src DIR [--inputs a,b]
# translation only (pure Python, no DSL): re-translate an existing dump
python3 -I frontend/cute/to_kir.py CASE.json [--out DIR]
# tests: tests/cute/test_core.py (pure Python: python3 -I tests/cute/test_core.py) and
# tests/cute/test_frontend.py (tiny CuTe kernels through dump + to_kir; needs the DSL)
```

`scripts/lkv.py run` runs the frontend (`frontend/cute/main.py run`: dump + translate),
the contract check and the Lean checks and writes the report.

`--src` analyzes another exported revision with the same case file (the case's
`source.src` is the default). Outputs: `dump.json` (with `"dsl"`: the DSL version, the
Python package and native MLIR library the process loaded, the installed DSL and quack
distributions, the Python version; also printed; `"env"`: the pinned environment, below;
per input `"host_reads"`; per launch `"hinted"`, the storages its integer arguments point
into, and `"sources"` on every tensor argument); `ir/NN_<digest>.mlir` (initial DSL
MLIR of compile NN), `ir/<digest>.stage.mlir`, `ir/<digest>.host.mlir` (analyzed
stages); `kir/<Kernel>__<digest>.json`; `kir/<Kernel>__<digest>.p1.json` (the P1 KIR);
`kir/report.json` (per variant: the inputs that launched it, KIR size and roles, or the
construct or KIR contract violation that stopped it with its `"kind"`: `gap`, a construct
the frontend refuses by design (UNDECIDED, not an error), or `error`, a failure the case,
the DSL install or LKV must fix (an ERROR: invalid KIR, a DSL version mismatch, an
unaccepted verifier failure, `elision_flags` that are not flags of the variant, no
analyzed stage); and `"p1"`: `checked` with the P1 KIR, `refused` with the refusal,
`error` for a failed P1 translation (an internal error of the P1 pass, invalid P1 KIR),
or `not_applicable` for a kernel without TMEM).
`to_kir --no-p1` (and `lkv_cute run/to_kir --no-p1`) skips the P1 pass. One case per process: the DSL and the kernel package read their environment at
import.

Lean: `python3 -I frontend/emit_lean.py KIR.json -o K.lean && scripts/run_lean_check.sh K.lean`.

## Case files

```json
{"name": "varlen_bwd", "description": "...",
 "source": {"package": "my_kernels", "rev": "<commit>", "path": "<repo path>", "src": "/tmp/lkv_src/base"},
 "env": {"MY_KERNELS_ARCH": "sm_100"},
 "entry": "my_kernels.interface:attention_backward",
 "builder": "my_args.py:varlen",
 "num_sms": 2, "max_iters": 4096,
 "kernels": {"AttentionBackwardSm100": {
    "work": {"batch": [{"load": "mCuSeqlensK", "where": "offset_k = 0 if ..."}]},
    "tensors": {"mK": {"segments": {"dim": 0, "offsets": "mCuSeqlensK", "per": "batch"}, "role": "input"}},
    "intended_writes": {"mdK": "all_rows_of_segments"},
    "elision_flags": ["config.full_tiles"]}},
 "inputs": [{"name": "tail_empty_multiwave", "...builder fields...": 0, "why": "..."}]}
```

* `entry` is called once per input with `builder(input) -> (args, kwargs, meta)` (CPU
  tensors; only shapes, dtypes and integer contents matter). Inputs run in order in one
  process, like consecutive calls of one training process, so host compile caches
  behave as in production. If the host refuses an input (the rule both frontends share,
  `frontend/ttgir/host_errors.py`: the innermost frame of the exception is under the
  source root, e.g. a validation `raise`, or a torch check the sources call fails, e.g.
  `torch._assert_async`), `dump.json` records the exception and its innermost source line
  as `host_error`, `kir/report.json` lists it as a `(host)` entry, and the next inputs
  still run; launches made before the exception are kept. An input for which the host
  launches nothing is recorded the same way, with `host_error` "no launch". `lkv.py`
  lists both under the inputs the host rejected. Any other exception fails the dump: a
  direct call of a `@cute.jit` function (no `cute.compile`) reaching the patched compile
  (P5), a CUDA call on the CPU-only host (`torch.cuda.synchronize()`), a failing torch
  op, an error of the DSL or the frontend.
* `num_sms` is the device the host sees (P1, P2); grids and persistent tile tables
  derive from it. `max_iters` bounds every KIR `while` (exceeding it is a verdict error,
  never a pass).
* `kernels` annotates kernel variants by jit class: KIR tensor names are the jit
  parameter names (tuples flatten: `blocksparse_tensors_full_block_cnt`).
  * `tensors`: `role` and `segments` per kernel tensor; a TMA descriptor over a segmented
    tensor gets the segments on the descriptor dim that walks that tensor dim (same
    extent and stride; the global view may permute the tensor's modes). A segments
    `offsets` tensor is metadata even if the kernel never reads it.
  * `work`: per KIR work field, the anchors that set it: at every thread-uniform integer
    load of tensor `load` whose source location (file:line or the statement text the
    DSL records) contains `where`, the field becomes the load's element index and a
    `work` event with every field follows. Fields are role-level variables, -1 until
    set. This is the only per-kernel input that names source code; it identifies a
    value, it does not model behavior.
  * `intended_writes`: copied to every input of the variant (P0-3). Every tensor the
    kernel writes must be named: a region (`all_rows_of_segments`, `all`, row ranges) or
    `unchecked` with the reason in `why` (e.g. padded statistics layouts, reduce-only
    accumulators, or an output split between two launches); an
    output missing from `intended_writes` fails P0-3 (`kir.md`), a named tensor the
    variant does not write makes its KIR invalid.
  * `elision_flags`: the variant switches (names of its `flags`, below) that select a
    mask-elided fast path, e.g. `config.full_tiles`. Each must be a flag of the
    variant (checked); the list becomes the KIR `case.elision_flags`, and on the inputs
    whose variant sets one of them the P0-1 and P0-3 checks are P0-5's (`kir.md`).
  * `why`: free text, the reason for the annotations (not read by the frontend).

## What dump runs, and what it patches (trust base)

Only device queries, device typing of tensors, and the compile/launch calls are
replaced (`host_patches.py`, installed only during `dump`). Every host computation (tile
tables, grids, flag and variant selection, descriptor construction, validation) is the
real code at the analyzed revision.

| # | Patch | Why it is faithful |
|---|---|---|
| P1 | `torch.cuda.is_available / device_count / get_device_properties / get_device_capability / current_device`: one SM100 device (cc 10.0) with `num_sms` SMs | the host picks the SM100 path and sizes grids and tile tables from these |
| P2 | `cutlass.utils.HardwareInfo`: `num_sms` SMs, `num_sms // cluster` clusters | persistent schedulers bake the SM count into the grid |
| P3 | `torch.Tensor.is_cuda` is True | host validation demands CUDA tensors; the CPU tensors have the same shapes, strides, dtypes and integer contents |
| P4 | DLPack capsules made by `cute.runtime.from_dlpack` report device kDLCUDA; each runtime tensor remembers the torch tensor it wraps | the DSL's C++ DLTensorWrapper derives the address space from the device: the tensors are typed as global memory exactly as CUDA tensors are |
| P5 | `BaseDSL.compile_and_cache` computes the pass pipeline as the original does (`preprocess_pipeline(_get_pipeline(pipeline), arch)`), records it and stops | the DSL has already written (`CUTE_DSL_KEEP_IR`) and verified the initial MLIR; nothing is lowered, linked or loaded (unpatched, `cute.compile` runs the whole pipeline and then fails in `cuInit`) |
| P6 | `cute.compile` returns a recorder; calling it records the launch arguments (DSL scalars unwrapped; a runtime tensor as the torch tensor it wraps, the same memory; one that wraps none fails the dump), the sources of each tensor argument's data and the storages its integer arguments point into (P7). Keyword arguments of the compile and of the launch are bound to the jit function's signature (defaults applied; cute.compile's own keywords such as `options` are left out) and recorded positionally; a call that does not fit the signature is a frontend error, not a host rejection. An explicit `cuda.CUstream` launch argument is dropped (the host function's stream argument takes no launch value) | that call is the launch; nothing runs |
| P7 | a `HostReads` TorchFunctionMode, with a TorchDispatchMode for aten ops (`frontend/ttgir/host_reads.py`, shared with the TTGIR frontend), around each input's host call; one for the whole run | it calls every torch function unchanged and only tracks which data comes from launch arguments (below) |
| P8 | `torch.Tensor.pin_memory` returns a copy of the tensor (`is_pinned` is True) | page-locked memory changes how copies to the device run, not the data (a host builds a tile table on pinned memory before copying it to the device) |

**Environment** (`frontend/ttgir/host_env.py`, shared with the TTGIR frontend; not a
patch). Variant selection reads environment variables (scheduler, sparsity and
architecture switches of the kernel package), and so does the DSL. Before the DSL and
the kernel package are imported, dump unsets every variable the
exported sources and the case's builder read by name (`os.environ.get(K)`,
`os.getenv(K)`, `os.environ[K]`, `K in os.environ`, `os.environ.pop/setdefault(K)`, K a
literal or a module-level string constant; any other read is an error) and every
compiler switch (`CUTE_DSL_*`, `TRITON_*`, `MLIR_*`, `LLVM_*`, `NVPTX_*` and the
`triton.knobs` variables outside those prefixes), except those the case's `env` sets
(e.g. a kernel package's architecture override for CPU compilation); then it sets
the case's `env` and its own `CUTE_DSL_ARCH=sm_100a`,
`CUTE_DSL_KEEP_IR=1` and `CUTE_DSL_DUMP_DIR` (a process scratch directory), which a case
may not set. So a kernel package's on-disk compile cache is off unless a case turns it
on. Until the dump ends, a read by the exported sources or
the builder of any other variable (through an alias of `os` or `os.environ`, a key the
scan cannot see) and any enumeration of the environment from them raise `EnvError` with
the source line. dump.json `"env"` records the case's `env`, the shell
values dropped (`unset`), the variables the sources read and the effective values at
the end of the run; `lkv.py run` copies it into its report.

Checked, not assumed:

* the entry point is imported from the exported source directory (never the build graph);
* every launch's arguments match the host function's arguments one to one (count, rank,
  static extents and strides);
* metadata provenance: a metadata tensor (one the kernel reads to steer control or
  addresses, or a segments `offsets` tensor) must hold the recorded contents when the
  kernel reads it. So it must not be written by the analyzed kernel itself, nor share
  storage bytes with a tensor the kernel writes in the same launch, nor hold data an
  earlier launch of the run may write, directly or through host computation (every
  launch is skipped, such data keeps its CPU contents; the inputs run in one process, so
  a launch of an earlier input counts, e.g. through a module-level cache; write sets come
  from the earlier kernel's KIR, or, when it was not translated, all its tensors and the
  storages its address hints point into). Each is a hard error for the variant. Contents
  are recorded by element offset from the data pointer (the elements between those of a
  strided view included), as `load` indices are.
* host reads: P7 tracks where host data comes from, over the whole run. Every storage
  passed to a launch is a possible source from then on, and so is every storage an
  integer argument of the launch points into (an address hint: an integer scalar, or an
  element of the recorded contents of an int64 tensor, inside a storage some torch call
  touched; a kernel may write through it, e.g. through a grouped GEMM's pointer table); a
  torch call with such an input makes everything it writes derived from the same sources
  (copies, reductions, arithmetic, in-place ops, `__setitem__`); a call that turns such
  data into a Python value (`item`, `tolist`, `int()`, `bool()`, `numpy`, printing, ...;
  not shapes, strides, pointers or the DLPack export) is a host read, and so is an aten op
  whose result shape or Python value depends on such data (`nonzero`, boolean indexing,
  `repeat_interleave`, `unique`, `bincount`, `torch.empty(t)`, ...), one read per call;
  each is recorded with its sources, its source line and the number of launches of the
  run before it. A host read of data an earlier launch may write (same write sets),
  before a launch of a variant, in the same input or an earlier one, refuses that variant:
  the host may branch on it or size the launch with it. Whether its control flow really
  depends on the read is not decided.
* moves between host and device: `.cpu()`, `.to(device)` and the like return their input
  on the CPU host where the GPU makes a copy, so a later write to a storage such a move
  returned fails the dump (`HostAliasError`, naming both lines).
* the case: `elision_flags` are flags of the variant.
* the DSL: `to_kir` refuses a dump made with a DSL other than `ops.DSL_VERSION`, or whose
  installed native libraries (`nvidia-cutlass-dsl-libs-base`) are another release.

**DSL version.** The frontend loads exactly one CuTe DSL release, nvidia-cutlass-dsl
4.4.2 (the Python package and `_cutlass_ir` from nvidia-cutlass-dsl-libs-base 4.4.2),
the release the op table was checked against. `dump.py` records and prints the loaded
DSL (`dump.json` `"dsl"`); `to_kir.py` accepts only `ops.DSL_VERSION` (the op table's
issuing threads were checked in the PTX of that release) and records it per variant in
`kir/report.json` (`"dsl"`) and in the KIR header (`"compiler": {"name":
"nvidia-cutlass-dsl", "version": ...}`, next to `"target": "sm_100"`).

## The analyzed stage (stage.py)

The DSL compiles a module with one pipeline, `cute-to-nvvm{...}`. Its expansion starts
with `cute-remove-static-args, materialize-tma-multicast, software-pipeline,
cute-desugar, cute-expand-ops, cute-fold-static, insert-range-for-threadids,
canonicalize, cse`; the analyzed stage is exactly that prefix, options included (no pass
list is hard-coded). Layout functions (local_tile, partitions, composition) are folded
into crd2idx/add_offset arithmetic and copy atoms expanded into per-thread
loads/stores and `cute_nvgpu.arch.*` ops with explicit TMA coordinates, while `scf`
control flow is still structured. Later passes are not registered as standalone passes.

* **Verifier.** With cutlass-dsl 4.4.2 the pass verifier rejects some kernels after
  `cute-expand-ops`: operand 0 of `cute_nvgpu.arch.copy.SM90.bulk_copy_g2s` (e.g. the
  LSE/dPsum row copies of an attention backward) loses its alignment attribute. The
  prefix runs with the verifier off, every op is verified afterwards, and to_kir accepts
  only this documented failure (the op table's meaning does not use the alignment).
* **Host stage.** The host function is read from a second lowering without
  `cute-expand-ops`: there a TMA descriptor is one `make_non_exec_tiled_tma_*` op over the
  global view (expanded, it becomes bit-packing of descriptor words).
* **Text, not bindings.** Stages are printed in generic op form with debug locations and
  read by `mlir_text.py`. The MLIR Python bindings are used to parse the initial MLIR and
  run the passes, but not to walk the result: the DSL registers value casters on every
  `ir.Value` the bindings return, and some build new ops when invoked (walking a lowered
  kernel raises in `cute.deref_arith_tuple_iter`); the custom op form does not round-trip
  (`nvvm.read.ptx.sreg.*` range attributes).

## Translation

**Slice** (`slicer.py`). Kept: the op table's events and every scf op containing one.
Relevant: the operands each event names (addresses, coordinates, barrier pointers,
parities, byte counts, conditions), the control operands of kept scf ops, and
transitively every operand of an op defining a relevant value (through yields,
iteration arguments and branch results; a relevant scf result also keeps its op, so the
branch condition or trip count that decides it is relevant). Everything else is not
translated. Soundness check: a relevant value defined by a data op (float math,
register/shared/tensor memory, MMA) is a hard error naming the op and the kept use
(`data value from arith.cmpf at k.py:11 reaches branch condition (slice soundness)`).
Classification: the slicer visits every op of the kernel (and of the host function), and
an op the table does not classify, or an inline PTX statement with no table meaning, is
a hard error there, kept or not (an op inside an `scf.for` without events is checked
too).

**Values and threads.** Thread-uniform integers are bound by `let` (one per IR op, with
its loc; subexpressions shared within a let are bound first, so JSON stays linear in the
expression DAG). Thread-varying values stay symbolic over LANE/WARP/ELECT; uniformity is
decided by substituting every thread of the role and comparing the folded results
(folding is exact BitVec arithmetic plus exact interval reasoning, e.g. `x & 31 <= 31`).
Shuffles and ballots are lane substitutions; a lane-varying value computed from
per-lane loads or shuffles becomes a table of 32 bound per-lane values, so warp-
cooperative code (e.g. a varlen tile scheduler: per-lane `cu_seqlens` loads, prefix sums,
ballot + popc) is exact and linear in size.

**Control flow.** A thread-uniform `scf.if` is a KIR `if`; its results are variables
declared before it and assigned in both branches. A thread-varying condition is a mask:
both branches are translated in place, events carry the mask (the uniform part as an
`if`, the rest per thread), results are `select`s, and a divisor or metadata-load index
under a mask is replaced where the mask is false (the value is unused there), so a path
not taken never raises a verdict error. `scf.for` is a KIR `for` (thread-uniform
bounds, positive constant step checked); `scf.while` is a KIR `while` with the case's
`max_iters`, its condition region emitted before the loop and again at the end of the
body. Loop-carried values are variables assigned before the loop and reassigned in
parallel at the end of the body (temporaries when an assignment reads an earlier one, or
a family entry another entry of its family, as a carried shuffle does); the variables
carry integer leaves only, so a value whose structure changes between iterations (e.g. a
pointer into another tensor) is refused. A thread-varying value stored in a variable
(loop-carried, branch result) is a family of variables, one per lane (if it does not
depend on WARP) or per thread, read through a table; families are widened to a fixpoint
before the loop is emitted.

**Roles.** Partition refinement over the CTA's warps: a role is a contiguous warp range
translated with WARP symbolic; a branch whose condition is uniform within each warp but
differs between its warps splits the range and each part is translated again. Roles are
named `warps<first>_<last>`. Code before the warp dispatch (barrier inits by warp 0,
shared prologue) belongs to the roles that execute it.

**Events.** See `ir_ops_cute.md` ("Issuing threads"). Memory events are one event when a
single thread issues them, else a KIR `for` over the role's threads with the thread's
mask as an `if` (shared per-thread subexpressions bound inside the loop). A TMA store is
issued once per distinct (coordinates, shared-memory source) among each warp's lanes, by
the lowest lane holding it (ptxas's waterfall loop): one event when its operands are
warp-uniform; a TMA reduce, like a TMA load, is issued once per warp (the DSL's
full-mask `elect.sync`; each issuing warp must be fully active). TMA stores and
reduces are `gstore`/`gatomic` on the descriptor's kernel tensor with exactly the copy's
footprint (KIR cannot say a descriptor aliases a tensor, and a kernel may write one output
through both TMA and per-thread stores); a TMA load is a `tma_load` on the descriptor (its
transaction bytes) plus a `gload` of its footprint on the kernel tensor, so P0-1 checks
it against the tensor's allocation and not only against the descriptor's extents. Bulk global->shared copies are a `gload` (footprint)
plus a `tma_load` on a 1-D descriptor `bulk_<T>_<n>` over T's storage (transaction
bytes).

**Barriers.** Events carry the barrier's shared-memory byte address; after every role is
translated, init addresses (constants, or every value over enclosing constant-trip
loops) form runs: maximal runs of consecutive 8-byte barriers with one count, one init
source line and one enclosing op (the DSL's pipelines initialize each barrier array
under a guard of its own, all at one library-called line). A run splits into the arrays
its uses index (`to_kir._split_arrays`): an event whose address has a dynamic part keeps
together the slots its address may take (the interval of the address with let-bound
index variables expanded, `Translator.unbind`), slots only constant addresses name are
arrays of one, and a run with an index of unknown range stays whole. Arrays are named
`mbar_<start byte>`. An event's address `C + sum k_i t_i` is resolved to `(array
containing C', (C - start)/8 + sum (k_i/8) t_i)`, C' the constant part of the expanded
address (`v59 = (c & 1) + 29` hides constant 29 from the address `8 * v59`).
`threads_per_issue` is the gcd of the role's arrive issuer counts (1 when some arrive is
counted per thread); each arrive gets `n = issuers / gcd`.

**Tensors, descriptors, params, inputs.** Host tensor arguments are KIR tensors named
after their jit parameter; params are their dynamic shape/stride leaves
(`<name>_shape<i>`, `<name>_stride<i>`, in the kernel's element units) and the i32/i64
scalar arguments, valued per input from the recorded launch. Tensors are in storage
elements (`elem_bytes` of the launch dtype; a packed `float4` storage element holds 2 DSL
elements). Metadata tensors are flat (`[extent]`, stride 1; `load` indices are element
offsets) with their recorded contents. Descriptors: `tma<k>_<tensor>` (`ir_ops_cute.md`,
host side). Identical IR from several compiles is one KIR with every input that
launched it; `flags` are the boolean attributes of the jit object and of its config
dataclasses (the host-selected variant switches, e.g. `use_tma_store`,
`config.elide_full_tile_mask`, `config.full_k_tiles_only`); constexpr branches are
resolved at trace time, so a variant a flag selects is its own KIR, checked on the inputs
that select it. The kernel's `elision_flags` annotation becomes the KIR
`case.elision_flags` (P0-5).

**Clusters.** The cluster dims come from the launch (`cuda.launch_cfg.cluster_dim` is
on the host slice) and must be constants; the KIR gets `cluster` (kir.md change 11).
`%cluster_ctarank` is `x % cx + (y % cy) * cx + (z % cz) * cx * cy` over the CTA's
`pid`, `%clusterid` / `%nclusterid` / `%cluster_ctaid` / `%cluster_nctaid` likewise.
`nvvm.mapa.shared.cluster` makes a remote pointer (`interp.Remote`: a shared-memory
pointer and a CTA rank); an mbarrier arrive or `arrive.expect_tx` through one (space
`cluster`) gets the event's `cta`, which must be uniform over the role (a role whose
warps differ in it is split). `tcgen05.commit` (cta_group::1 or ::2) with a multicast
mask gets `cta_mask`. A TMA load with `num_cta = 2` is `leader` (the DSL clears the peer
bit of the barrier address, as CUTLASS's `SM100_TMA_2SM_LOAD`: each CTA's bytes complete
on the even CTA's barrier), and its multicast mask operand is `cta_mask`. A multicast
descriptor (`num_multicast` n > 1) covers the CTA tile and each load moves a slice: the
KIR box is the descriptor's box truncated by n from the outermost dim, as CuTe's
`make_tma_copy_desc` does (the atom's `copy_bits` is the untruncated tile; the load
coordinates already select the slice). `barrier.cluster.arrive{.relaxed}` / `.wait` are
an arrive of every thread of the role and a wait on `cluster_barrier`, a frontend-made
barrier of one slot in the cluster's CTA of rank 0 with one arrival per thread of the
cluster; the j-th wait of a role waits for phase j (the role variable
`cluster_barrier_parity`, flipped after each wait); threads that exit early are not
modeled (they count as arrived in PTX). Inline PTX: a blocking wait loop (`L:
mbarrier.try_wait.parity.shared::cta.b64 P, [$a], $p[, hint]; @!P bra L` or `@P bra D;
bra L; D:`, next to data statements, e.g. inside an MMA helper) is a `wait` on the
shared-memory pointer `$a` holds (`cute.ptrtoint`); `mapa rd, $dst, $r; mapa rb, $bar,
$r; cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [rd], [$src],
$n, [rb]` is a `complete_tx` of `$n` bytes on the barrier `$bar` of CTA `$r` (once per
issuing thread); `trap` is not modeled (the run is evaluated as if it were not
reached). In the P1 pass a cta_group::2 MMA or copy (the pair's leader issues it) is a
`tmem` event with `cta_mask` `3 << (rank & ~1)`: the same rectangles in the TMEM of both
CTAs of the pair (M = 256: 128 lanes x N in each; M = 128: 128 lanes x N / 2, CuTe's
"2x2" data path layout, `tmem_frg_2sm` with 64 rows per CTA).

**Integers narrower than 32 bits.** i8 and i16 values (multicast masks, fast divmod
shifts) are carried as their zero-extended i32 in KIR (`expr.lower_narrow`: each narrow
result reduced modulo 2^w, signed ops and comparisons on the sign extension).
`cute.fast_divmod.get_aux` yields reserved values that only the unsigned division idiom
`(mulhi(n, m) + ((n - mulhi(n, m)) >> s1)) >> s2` may consume: the shift that completes it
is `udiv(n, d)` (`ops.fast_division`; the DSL's divisor computes m, s1, s2 so that the
idiom is exact for every 32-bit n), any other use of them is an error.

**addr.** Set on every add/sub/mul/shl and every trunc (scalar or TExpr) whose value
reaches a global offset, a TMA coordinate or a metadata load index through any chain of
`let`s; Lean reports their wraps and lossy truncations (P0-1).

**P1 pass (P1-1).** `to_kir` translates every variant a second time with a fresh
translator in P1 mode: the slice is the P0 slice plus `ops_tmem.TMEM_EFFECTS` and the
handlers are `ops_tmem.spec_p1` (`ir_ops_cute.md`, "P1 slice"), host function included
(the tiled MMA atom is an argument of the kernel). The P1 pass may split roles
differently (new kept branches), and splits a role of several warps whose warps do not
all issue an mbarrier wait, arrive or commit (`Translator._p1_split_by_issuers`: e.g. a
thread-0 arrive or wait makes runs of issuing and non-issuing warps roles of their own,
because P1-1 takes a role's event as every warp's and an arrive releases, a wait
acquires, only the issuing warps' TMEM accesses; any lane issues for its warp). It has its
own KIR, ledger and verdicts, and P0 verdicts come only from the P0 KIR (the P0 pass keeps
its roles). The P1 KIR's top-level `tmem` comes from the single
`alloc_tmem` and the pointer bases (`zero` for the fake-zero-base idiom with 512 columns,
`relative` for `retrieve_tmem_ptr`). Refusals, recorded as `report.json` `p1.error` and
reported as "P1-1 not checked" (the P0 entry is unaffected; so is any other error of the
P1 pass):

| # | Construct |
|---|---|
| R1 | a TMEM pointer of unknown provenance on the P1 slice (inttoptr of a non-constant or nonzero value) |
| R2 | zero or several `alloc_tmem` (one per issuing warp), one in a loop or under a runtime lane condition, a non-constant column count |
| R3 | absolute pointers with an allocation under 512 columns, or absolute and relative pointers mixed |
| R4 | tcgen05 data, MMA or allocation statements in inline PTX outside the table (`ir_ops_cute.md`, "P1 slice": `tcgen05.cp` and dense `tcgen05.mma` with operand-set addresses are in it) |
| R5 | shapes or attributes outside the table: other ld/st/cp shapes, an MMA other than M = 128 with cta_group::1 or M = 128 / 256 with cta_group::2, a non-f32 accumulator, cta_group::2 without a cluster, unknown or disagreeing instruction-descriptor shape bits, a block-scaled MMA with N > 256 or more than 4 scale factors per row (K / vec_size) |
| R6 | a lane-varying ld/st address, a partial or runtime lane mask on ld/st/cp/mma, cp/mma in a role of several warps, inline PTX cp/mma issued by several threads (no elect.sync) or under a predicate other than an elect.sync's |
| R7 | a named barrier with a non-constant id or count, or reached by part of a role (an id or count uniform per warp but not per role splits the role instead); tcgen05.wait reached by part of a role; an mbarrier wait, arrive or commit under a runtime lane condition in a role of several warps (issued by part of its warps under a known condition, the role is split instead) |
| R8 | a kept read of an atom field that is data after a merge (atom fields as leaves) |
| R9 | inherited P0 refusals |

MMA events carry `site`, the closest preceding source line in another file than the
DSL helper they are located in (`blackwell_helpers.py:869`).

**Locations.** `loc` is the innermost `file:line` under the source root. Ops the DSL
builds inside its own helpers (operator overloads in `cutlass/base_dsl/typing.py`,
pipeline helpers) and ops a pass created carry no source frame; they get the location of
the closest preceding op in program order that has one.

## Validation of the extraction

`tests/cute`: tiny CuTe kernels through dump + to_kir, and hand-written stages in
`test_core`:

* The TMA producer/consumer passes all five P0 specs in Lean; the seeded bug (producer
  `arrive_expect_tx` on all 32 lanes) fails P0-2 and P0-4. A TMA load through a view
  claiming 64 rows past its tensor fails P0-1 (the footprint `gload`); the correct view
  passes.
* A warp-wide TMA store with warp-uniform operands is one event, as with `elect_one`
  (P0-3 PASS both); one whose lanes 16-31 store the next tile is two stores (PASS), one
  whose lanes 16-31 store a second shared buffer to the same tile writes each element
  twice (P0-3 FAIL); a warp-wide TMA reduce is one event. A `st.global.u16`
  scale-factor helper is a 2-byte `gstore` per thread (all specs PASS).
* Refused: a float compare deciding a store (slicing), `red.global` inline PTX, an
  unknown op in a loop without events, a global view from an integer, `elect.sync` under
  a partial warp, a TMA load or TMA reduce reached by 16 lanes of a warp, a TMA reduce
  inside `elect_one` or with coordinates that depend on the lane, a `vector.store`, two
  global views swapped by a loop (a loop-carried value that changes structure), and a
  kernel writing its own metadata.
* Host data: a host read (`cnt[0].item()`, or the shape of `repeat_interleave()`) of what a
  skipped launch writes before the next launch, metadata computed on the host from such
  data, a count launch passing CuTe runtime tensors (they are the torch tensors they
  wrap), a count a skipped launch of an earlier input wrote and a later input reads from
  a module-level cache, and offsets written by an untranslated kernel that gets their
  address in an int64 pointer table are refused; reading a tensor the skipped launch only
  reads, a host-computed cache and a pointer to another buffer are not. Offsets passed as
  a stride-2 view are recorded by element offset (P0-3 PASS).
* Host runs: an input the host rejects, or makes no launch for, is reported and the next
  input still runs; a direct `@cute.jit` call, `torch.cuda.synchronize()` and a launch
  argument wrapping numpy memory fail the dump instead. A shell value of a variable the
  sources read, and of a `CUTE_DSL_*` variable, is ignored and recorded, and the case's
  `env` applies; a builder reading the environment through an alias raises `EnvError`.
* A case naming the variant's `full_tiles` switch in `elision_flags` moves an
  out-of-bounds write of its seeded unmasked fill from P0-1 to P0-5 in Lean, and a
  misspelled flag is refused.

## Limitations

* A verdict covers the case inputs, the analyzed variants and `max_iters`.
* Below the analyzed stage nothing is checked (later lowering, PTX, SASS).
* `cute.assume` divisibility facts are taken as given; the elected lane of `elect.sync`
  is unspecified (operands under it must not depend on it).
* Shared memory, TMEM, MMA ordering, named barriers and fences are outside the P0 slice
  (data); TMEM accesses, tcgen05.wait and named barriers are events of the P1 slice only
  (P1-1), fences and TMEM lifetime are not checked; so are values that travel through register memory (`rmem` fragments): a
  predicate or index read back from registers that decides an event is a slice error
  (e.g. an epilogue's per-thread predicate fragments).
* Non-constant cluster dims, multicast bulk copies (non-tensor), im2col TMA,
  result-returning `try_wait`, barrier inits at runtime addresses,
  `wait`/`bar_init`/TMA loads under a runtime-dependent lane mask, `elect.sync` or a
  TMA load under a partial or runtime-dependent lane mask, loops with thread-varying
  bounds, a loop-carried value that changes structure, sub-element global accesses and
  inline PTX other than data statements, one blocking wait loop, one bulk reduce, one
  shared::cluster bulk copy or one unguarded `st.global` (8 to 64 bits, or a vector of
  2 or 4) are hard errors.
* Host reads are tracked through torch calls and aten ops only (`ctypes` on a data
  pointer and `numpy.from_dlpack` escape); address hints come only from integer scalars
  and recorded int64 contents; sources are whole storages, and any host read of stale
  data before a launch, in its input or an earlier one, refuses the variant
  (conservative). A write to a tensor a move between host and device returned fails the
  dump even where the GPU would not have copied.
* Barrier arrays are formed by adjacency, count, init line and enclosing op, then split
  by the ranges of their dynamic uses; barriers of several logical arrays that one use
  of unknown index range may address stay one array (P0-2's static-waiter rule then
  spans them).
* Across the CTAs of a cluster only synchronization is checked: which CTA's data a copy
  carries is not (a copy sent to the wrong CTA keeps the byte counts).
* The op table is checked against one DSL release (`ops.DSL_VERSION`, 4.4.2), and its
  TMA store/reduce rows against ptxas 13.0; moving to another release needs the
  issuing-thread checks redone before `DSL_VERSION` changes. A TMA store's cache-policy
  operand is taken to be lane-uniform (a lane-dependent one would add waterfall rounds).
* KIR size grows with warp-cooperative code: the non-broadcast varlen scheduler is
  evaluated per lane (32 per-lane values per step) in every role that runs it.
