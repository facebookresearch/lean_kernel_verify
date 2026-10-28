# TTGIR frontend (TLX / Triton): Layer 1

The frontend turns a TLX/Triton kernel plus a case file into KIR (`kir.md`) with no
hand-written model of the kernel. The real host code runs on CPU, the real compiler
produces TTGIR for the target GPU (sm_100 by default, or sm_103), and a fixed op table
(`ir_ops.md`) translates the TTGIR.
Code: `frontend/ttgir/`. Tests: `tests/ttgir/`; a minimal case: `tests/e2e/cases/probe_tlx.json`.

```
export_src.py   the kernel + the first-party modules it imports (also relative imports), at a revision
dump            pin the environment (host_env), host code on CPU, launch capture, triton.compile
                -> <spec>.ttgir, dump.json
to_kir          TTGIR -> parse (mlir_text) -> slice (slicing) -> KIR (op_table) -> <spec>.kir.json
```

`host_env.py` (environment pinning), `host_reads.py` (host-read tracking) and
`host_errors.py` (which exceptions of a host call reject the input) form the host
library shared with the CuTe frontend.

## Running

```bash
# dump (frontend/ttgir/dump.py --case CASE [--inputs a,b] [--out DIR] [--src DIR] [--target ...]):
#   host run + capture + compile; writes dump.json and one <spec>.ttgir per specialization
# to_kir (frontend/ttgir/to_kir.py --dump DIR/dump.json --case CASE): one KIR per
#   compiled specialization, next to dump.json
python3 -I scripts/lkv.py run CASE.json --src DIR [--inputs a,b] [--target sm_100|sm_103]
# tests (CPU): tests/ttgir/test_translate.py (translation of handwritten TTGIR) and
# tests/ttgir/test_ttgir.py (dump + to_kir on small TLX kernels); tests/ttgir/test_gpu_parity.py:
# CPU-dumped TTGIR == TTGIR of a real launch (needs a Blackwell GPU)
```

`scripts/lkv.py run <case>` runs both steps, then the Lean checks, and writes the
report. `--src` analyzes another export with the same case file; `--target` is passed to
`dump`. `to_kir` exits non-zero if any specialization fails; the message
(`[to_kir] <spec>: FAILED (<kind>): ...`) names the op, its source `file:line` and its
TTGIR line, and its kind: `gap` for a construct the frontend refuses by design (slice and
op-table refusals and the other unsupported constructs: UNDECIDED, not an error), `error`
for what the case or LKV must fix (a TTGIR text the parser cannot read, a KIR contract
violation, an internal error, a case annotation error, a target or location mismatch: an
ERROR). Every KIR it writes has passed `frontend/kir.py`.

The inputs run in order in one process. If the host refuses an input (`host_errors.py`:
the innermost frame of the exception is under `src_root`, e.g. a validation `raise` or
`assert`, or a torch check the sources call fails, e.g. `torch._assert_async`), dump.json
records the exception and its innermost source line as the input's `host_error`, keeps the
launches made before it, and the next inputs still run. An input for which the host makes
no launch of `kernel` gets `host_error` "no launch of <kernel> (the host launched: ...)".
`lkv.py run` lists both under the inputs the host code rejected, so a fix whose host now
refuses some trigger inputs is checked on the others in the same run. Any other exception
of the host call fails the dump: a failing torch op, a device call the patches below do
not answer, an error of Triton or of the frontend.

## Case file

```json
{
  "name": "attn_bwd_1cta",
  "description": "free text",
  "rev": "<commit>",                             // revision the export was made from
  "src_root": "/tmp/lkv_src/base",               // export root; module must resolve here
  "source": "kernels/attn.py",                    // repo path, copied to KIR "source"
  "module": "kernels.attn",
  "entry": "attention_backward",                         // host function (custom ops ok)
  "kernel": "_attn_bwd_ws",                      // launches of this JITFunction are analyzed
  "num_sms": 2,                                  // SM count the host and CLC schedules see
  "env": {"VAR": "value"},                       // optional: the run's environment (below)
  "module_overrides": {"USE_2CTA": true},        // optional host A/B switches
  "defaults": {"sm_scale": 0.088},               // args shared by every input
  "setup": [{"call": "fn", "args": {...}, "bind": ["a", "b"]}],  // optional host helpers
  "tensors": {"desc_k": {"segments": {"dim": 0, "offsets": "K_offsets", "per": "batch"},
                         "role": "output"}},     // optional KIR annotations per kernel arg
  "intended_writes": {"dK": "all_rows_of_segments", "desc_dq": "unchecked"},
  "work": {"batch": "off_z", "head": "off_h", "block": "pid"},   // work-item fields
  "elision_flags": ["FULL_TILES"],               // optional: constexprs that elide masks
  "clc_schedules": ["none", "round_robin", "greedy"],             // optional, default all
  "inputs": [{"name": "partial_lb", "args": {...}, "setup": [...],  // per-input setup too
              "tensors": {...}, "intended_writes": {...}}]       // override the case's
}
```

* Arguments: JSON scalars pass through; `{"dtype", "shape"}` is a zero-filled CPU tensor,
  `{"dtype", "values"}` a tensor with those values, `{"ref": name}` a value bound by a
  setup step. Tensor *values* only matter for integer metadata; floating-point data is
  never read by the analysis.
* `setup` steps run real host helpers (e.g. tile-table precomputation) before the entry
  point; `call` is an attribute of the kernel module or a dotted path into the export.
* `tensors` and `intended_writes` must name pointer/descriptor arguments of the kernel
  (checked). `segments.offsets` tensors become KIR metadata (their contents are recorded
  per input) even when the kernel itself never reads them.
* `intended_writes` must cover every tensor the kernel writes (a missing output fails
  P0-3): a region (`all_rows_of_segments`, `all`, `[[lo, hi], ...]` rows) or `unchecked`
  for outputs with no exactly-once contract (e.g. dQ accumulated by TMA reduce-add).
  Stores through an in-kernel descriptor are writes of its base pointer argument, so
  that argument is the one named (e.g. `dK` for a descriptor `desc_dk` created over it).
* `work` names, per KIR work field, the Python variable (TTGIR NameLoc) holding it. A
  `work` event is emitted in every block that defines all fields, right after the last
  definition; roles that never compute some field get none.
* `elision_flags` names the boolean constexprs that select a mask-elided fast path (e.g.
  `FULL_TILES`). Each must be a boolean constexpr of every specialization
  (checked); the list becomes the KIR `case.elision_flags`, and on the inputs whose
  specialization sets one of them the P0-1 and P0-3 checks are P0-5's (`kir.md`,
  `specs.md`).

## Environment

Variant selection (e.g. a kernel's load-balancing or autotune-set switch) and
compiler passes (e.g. `TRITON_PROMOTE_MBAR_TO_NAMED_BARRIER`) read environment variables.
Before Triton and the kernel module are imported, dump pins the environment
(`host_env.py`): it unsets every variable the exported sources read by name
(`os.environ.get(K)`, `os.getenv(K)`, `os.environ[K]`, `K in os.environ`,
`os.environ.pop/setdefault(K)`, K a string literal or a module-level string constant; a
read whose variable cannot be determined is an error) and every compiler switch: the
`TRITON_*`, `CUTE_DSL_*`, `MLIR_*`, `LLVM_*` and `NVPTX_*` variables and the
`triton.knobs` variables outside those prefixes (`USE_IR_LOC`, which points every TTGIR
location at the IR file, `USE_PTX_LOC`, `DISABLE_PTXAS_OPT`, `PTXAS_OPTIONS`,
`ENABLE_TILE`; `tests/ttgir` checks that every knob is covered), except those the case's
`env` sets; then it applies the case's `env` and the frontend's own `TRITON_CACHE_DIR` (a
process scratch directory: every compile runs, the cache is never read). Once Triton is
imported, the variables of libtriton's cache-invalidating list that the shell set (the
ones its passes read, e.g. `DISABLE_MMA_V5`, `ALLOW_LHS_TMEM_LAYOUT_CONVERSION`) are
dropped too. Until the dump ends, a read by the exported sources of any other variable
(through an alias of `os` or `os.environ`, a key the scan cannot see) and any enumeration
of the environment from them (`dict(os.environ)`) raise `EnvError` with the source line:
their values would come from the shell. dump.json `"env"` records the case's `env`, the
shell values it dropped (`unset`), the variables the sources read, the effective values
at the end of the run (variables the sources read and compiler switches, including the
knobs dump sets, e.g. `TRITON_ALWAYS_COMPILE`) and `triton_cache_invalidating`
(`libtriton.get_cache_invalidating_env_vars()`, the variables Triton's own cache key
includes; `{}` unless the case sets one). `lkv.py run` copies the record into its report.

## What dump runs, and what it monkeypatches (trust base)

Only device *queries*, device *placement* and the launch itself are replaced. Every host
computation (tile tables, grids, flag selection, descriptor shapes, autotune pruning and
pre-hooks, heuristics) is the real code at the analyzed revision.

| # | Patch (gpu_free.py, capture.py, dump.py) | Why it is faithful |
|---|---|---|
| 1 | `triton.runtime.driver.active` = `CpuOnlyCudaDriver`: device 0, stream 0, target `GPUTarget("cuda", 100 or 103, 32)` (`--target`), `get_active_torch_device()` = cpu, `utils.get_device_properties()` = {multiprocessor_count: num_sms, max_shared_mem: 232448}; any other attribute raises | the launch-time target JITFunction compiles for; 232448 is the opt-in shared memory of sm_100 and sm_103 (Triton's own offline table) |
| 2 | `torch.cuda.is_available` -> True, `device_count` -> 1, `current_device` -> 0, `get_device_capability` -> (10, 0) or (10, 3), `get_device_properties` -> {multi_processor_count: num_sms, major 10, minor 0 or 3, name "cpu-only sm_10x stand-in", shared_memory_per_block_optin: 232448} | host code sizes grids and schedules from these; `num_sms` is the case's |
| 3 | `CudaOnCpu` TorchFunctionMode during import and the host call: any `device="cuda"`/`torch.device("cuda")` argument becomes cpu; `Tensor.pin_memory()` and `Tensor.cuda()` return the tensor; `pin_memory=` kwargs are dropped | values are unchanged; only placement differs. Where the GPU would copy (a move between host and device) the CPU run returns the same tensor, so a later write to it is refused (`HostReads`, below) |
| 4 | `triton.knobs.nvidia.enable_c_cache = False` (before the kernel module is imported) | the C launch fast path would bypass `JITFunction.run`; no effect on the IR |
| 5 | `triton.knobs.compilation.always_compile = True` | the compile cache is keyed by content, not path: a hit would return TTGIR whose locs name another export |
| 6 | `JITFunction.run` wrapped to record `(args, kwargs, grid)`; `JITFunction._do_compile` replaced by a recorder that returns None (`warmup` compiles, e.g. `kernel.warmup(...)`, are not recorded: they do not launch) | `run` reaches `_do_compile` only after the real binder and `_pack_args` computed signature, constexprs, divisibility attrs and options; returning None makes `run` return before launching. The grid is evaluated on the binder's bound arguments, as `run` would; a grid callable is host code, and its reads are tracked like any other |
| 7 | `Autotuner.run` replaced: for every config of `prune_configs()` (after `early_config_prune` / `perf_model`): if the tuner would benchmark (several configs, or `ir_config_prune`), zero the `reset_to_zero` arguments (`pre_hook(..., reset_only=True)`); set `best_config` to the config; call the config's `pre_hook`; then `fn.run(..., **config.all_kwargs())` | that is what `Autotuner.run` does for the config it selects on a cold tuning cache; analyzing every candidate covers whichever one benchmarking would pick. Host code that runs after the autotuned launch sees `best_config` = the last candidate. `include_npot` autotuners are refused |
| 8 | a `torch.library.custom_op` entry point is called through its `_init_fn` | the Python body is the op; the dispatcher would only add fake-tensor checks |
| 9 | the case's `env` (pinned, above) and `module_overrides` | explicit host configuration of the analyzed run; reported with the case |
| 10 | `HostReads` TorchFunctionMode (with a TorchDispatchMode for aten ops) during the host calls, one for the whole run (`host_reads.py`) | observes torch calls (it calls each one unchanged) to track which data comes from launch arguments; see below |

Consequences that are checked, not assumed:

* Every launch is skipped, so data a launch writes keeps its CPU contents afterwards.
  dump compiles every launched specialization, the analyzed kernel's included, and reads
  its TTGIR write set (`provenance.py`; if a store target is not traceable, e.g. an
  address loaded from a pointer array, or the kernel has side-effecting inline asm, the
  write set is unknown: every pointer argument, and every storage its address hints point
  into). The inputs run in one process like consecutive calls, so a launch of one input
  may write what a later input reads (a module-level cache): one `HostReads` tracks the
  whole run, and launch indices, sources and write sets are run-wide. It tracks where
  host data comes from: every storage passed to a launch is a possible source from then
  on, and so is every storage an integer argument of the launch points into (an address
  hint: an integer scalar, or an element of the recorded contents of an int64 tensor,
  inside a storage some torch call touched); a torch call with such an input makes
  everything it writes derived from the same sources (copies, reductions, arithmetic,
  `.cpu()`, in-place ops, `__setitem__`); a call that turns such data into a Python value
  (`item`, `tolist`, `int()`, `bool()`, `numpy`, printing, ...; not shapes, strides or
  pointers) is a host read, and so is an aten op whose result shape or Python value
  depends on such data (`nonzero`, boolean indexing, `masked_select`, `unique`,
  `repeat_interleave`, `bincount`, `torch.empty(t)`, `arange(t)`: the
  `dynamic_output_shape` and `data_dependent_output` tags), one read per call; the grid
  callable's reads count. A source is stale if an earlier launch's write set covers its
  storage. dump.json lists per launch of the analyzed kernel the earlier writers of each
  tensor argument's data (`stale`) and per input the host reads of stale data made so
  far in the run (`stale_reads`: one in an earlier input may have filled a cache).
  to_kir refuses a KIR whose metadata tensor has a stale writer (e.g. a later launch of
  the same kernel reading what an earlier one wrote, or a host copy of it, or rows a
  kernel wrote into a pointer array's target), and an input with a stale host read before
  the analyzed launch: the host may branch on it or size the launch with it (e.g.
  `fill[(int(cnt.item()),)]` after a kernel computed `cnt`). Any such read refuses the
  input; whether the host's control flow really depends on it is not decided.
* A move between host and device (`.cpu()`, `.cuda()`, `.to(device)`, `pin_memory()`,
  `as_tensor(device=)`) copies on the GPU unless the data is already there, but on the
  CPU host it returns its input. A later write (any aten op's mutable argument) to a
  storage such a move returned would change both tensors on the CPU and one on the GPU,
  so it fails the dump with `HostAliasError` naming both lines (e.g. `shifted =
  meta.cpu(); shifted -= 2` before a launch reading `meta`).
* Arguments are snapshotted at capture time (later pre-hooks mutate descriptors).
* The specialization compiled is the one the launch would compile on the GPU: same
  `ASTSource` signature, constexprs and attrs, same options, same target. Triton derives
  `tt.divisibility = 16` from `data_ptr % 16` of pointers and from int values (and turns
  ints equal to 1 into constexprs); the values are the GPU's, and so are the pointer
  residues because every tensor's storage base is checked to be 16-byte aligned (CUDA
  caching-allocator blocks are 512-byte aligned; a misaligned CPU storage is an error).
  dump.json records the binder's specialization per parameter. `test_gpu_parity`
  compares the CPU-dumped TTGIR with a real launch on a GB300 (or B200) host.
* Every loaded module of a package present in the export was loaded from the export
  (checked after the import and after the host calls); a missing module cannot silently
  come from elsewhere.

## dump.json

```json
{"case": "...", "kernel": "...", "source": "...", "src_root": "...", "num_sms": 2,
 "target": {"name": "sm_100", "arch": 100, "max_shared_mem": 232448},
 "compiler": {"name": "triton", "version": "...", "libtriton_sha256": "..."},
 "env": {"case", "own", "unset", "read_by_sources", "effective", "triton_cache_invalidating"},
 "specs": {"<key>": {"kernel", "ttgir": "<key>.ttgir", "signature", "constexprs",
                     "attrs", "specialization": {param: [type, key]}, "options",
                     "shared_bytes"}},
 "inputs": [{"name", "other_launches": [kernel names],
             "stale_reads": [{"op", "loc", "after": launches before it, "writers": [writer]}],
             "launches": [{"spec": "<key>", "index": position among all launches of the run,
                           "grid": [x, y, z], "config": {...} | null,
                           "args": {param: snapshot}, "stale": {param: [writer]}}],
             "host_error": "..."}]}       // only on an input the host rejected (above)
```

A writer is `{"launch": k, "kernel", "param"}`: launch k of the run may write the storage
the data comes from (`param`: the argument it writes through, or the integer argument
whose address hint points into the storage).

`specialization` is the binder's per-parameter result, e.g. `["*fp16", "D"]` (pointer
16-byte divisible), `["i32", ""]`, `["constexpr", 128]`. A snapshot is
`{"kind": "scalar", "value"}`, `{"kind": "tensor", dtype, elem_bytes, shape, strides,
storage_offset, storage: [lo, hi) bytes of the whole storage, data_ptr, values (integer
tensors)}` or `{"kind": "desc", dtype, elem_bytes, shape, strides, block_shape, padding,
base}`.
`<key>` hashes kernel, signature, constexprs, attrs and options.

## KIR produced

* One KIR per specialization; `case.inputs` are the launches that compiled to it. The
  KIR `target` is the dump's (`sm_100` / `sm_103`), `compiler` its Triton version.
  Integer specialization attrs (divisibility by 16) split specializations, so inputs
  with different lengths can land in different KIR files.
* `params`: the TTGIR integer kernel arguments, including the per-descriptor arguments
  the host expands (`desc.shape.i` i32, `desc.stride.i` i64), plus `T.shape.i` /
  `T.stride.i` (i64) for every pointer tensor `T` the KIR declares; all valued from the
  host tensors. `inputs[].shapes` repeats the shapes.
* `grid`: each input carries its own grid; the top-level grid is the first input's.
* `flags`: the boolean constexprs of the specialization (host-selected flags such as
  `ENABLE_LB`, `BROADCAST_Q`, `FULL_TILES`). Integer constexprs are compiled in.
  `case.elision_flags`: the case's `elision_flags`, if any.
* Tensors: pointer arguments the kernel accesses are `global`; host
  `TensorDescriptor` arguments are `desc` with `box` = the block of the `!tt.tensordesc`
  type (TMA block, `ir_ops.md`); metadata tensors are 1-D (a `load` index is the flat
  element offset). Role is `output` iff the kernel writes it
  (gstore/gatomic/tma_store/tma_reduce), unless the case says otherwise. Every output
  needs an `intended_writes` entry in every input (or the opt-out `unchecked`), so P0-3
  never passes by omission.
* Roles: `default` (the kernel's `num-warps`) and `partitionN` (`num_warps(N)`) of
  `ttg.warp_specialize`; a kernel without it has one `default` role. The code before
  the warp_specialize (barrier inits, shared values) runs in `default`; partition
  captures are rebuilt from kernel arguments and bound with `let` at role entry.
  `threads_per_issue` is 1: every TTGIR barrier/TMA/MMA op is a single-thread (leader)
  operation; a `perThread` arrive is emitted with `n = count x warps x 32`.
* Statements: `let` assigns (kir.py's mutable role environment). Loop-carried values are
  assigned before the loop and reassigned through `<name>.next` temporaries at the end
  of each iteration (parallel assignment); `scf.if` results are declared (0) before the
  if and assigned in both branches; an `scf.while` before-region is emitted before the
  loop and again at the end of the body; `max_iters` is a fuel of 65536 (the frontend
  does not prove loop bounds; exceeding it is a verdict error, never a pass). An early
  `return` (acyclic `cf` blocks without arguments) becomes nested `if`s.
* `addr` is set on every add/sub/mul/shl and trunc on an address path: the value flows
  into a pointer offset, a TMA coordinate, a descriptor's global address or a metadata
  load index through operands that carry it (not through comparisons, select/if
  conditions or masks).
* `loc` is `fbcode/<path relative to src_root>:line` of the innermost location in the
  kernel's source tree (the call site, for lines inside the Triton library). An event
  whose location is not such a line is an error (a compiler switch such as `USE_IR_LOC`
  rewrote the locations).
* Constexpr branches are resolved at compile time, so the mask-free variant a flag
  selects is its own KIR, checked on the inputs that select it; the case's
  `elision_flags` say which flags make it one (P0-5).

### Cluster Launch Control (CLC)

`clc_try_cancel` fetches, asynchronously, the id of a pending CTA the running CTA may
take over (or failure). Which CTAs launch and what each query returns is a property of
the hardware schedule, so it is input data: per input, to_kir emits one KIR input per
schedule, `<input>@clc:<schedule>`, with metadata tensors `__clc_launched[cta]` (1 if the
CTA starts) and `__clc_schedule[cta, k]` (the k-th query result, -1 = no more work).
`resident = min(grid, num_sms x CTAs-per-SM by shared memory)` CTAs start at once:

* `none`: every try_cancel fails; every CTA launches and runs only its own tile;
* `round_robin`: pending CTAs are stolen in order, cycling over the resident CTAs;
* `greedy`: CTA 0 steals every pending CTA.

Every role body is `let __clc_k = 0; if __clc_launched[pid.x] != 0 { ... }`, and the
k-th `clc_query_cancel` of a role reads `__clc_schedule[pid.x, k]`. That every role reads
the same response at its k-th query relies on the response barrier protocol, which P0-4
checks on the same trace.

### Encodings

* TExpr `ext` / `trunc` where `tt.addptr` sign-extends an i32 tensor offset that is added
  to another offset term (an i64 scalar base, or a second tensor addptr): the hardware
  sums the terms in 64 bits, so the i32 part keeps its own width (and its wrap check)
  and is extended per element. A single tensor term over the base pointer is emitted in
  its own width (offsets are read as signed). Used e.g. by row-statistics (`M`/`D`)
  loads and partial-tile dK/dV stores.

* A descriptor created in the kernel (`tl.make_tensor_descriptor` ->
  `ttng.tensormap_create`): every store/reduce through it is a `gstore`/`gatomic` of its
  base pointer argument with exactly the TMA footprint: element offsets
  `sum((coords + i) * strides)` (i64) and mask `0 <= coords + i < shape` in every dim,
  `i` ranging over the block. So P0-1 checks the stores against the base allocation and
  `intended_writes` names the base pointer (e.g. `c_ptr` of the tutorial GEMM). Loads
  through it are `tma_load`s of a `desc` tensor named after the scratch buffer, whose
  shape/strides are the extents, if those are host-side expressions; with runtime
  extents (dK/dV bounded by the segment end `end_k`) the extents are bound by `let`
  where the descriptor is created (stores must be in that role) and loads are refused.
  Extents are unsigned: the tensormap `global_dim` is a b32 that TMA reads unsigned (an
  extent of -64 is 2^32 - 64 elements, not an empty descriptor), so the KIR uses the i32
  extent zero-extended to i64, in the footprint mask and as a `desc` shape.
* The CLC response transaction is a `tma_load` of the constant 16-byte descriptor
  `__clc_response` on the response barrier.

## Soundness checks (every failure is a hard error with op and loc)

* Every op is classified by the op table; unknown ops anywhere in the kernel are errors
  (they could hide events), not only those on the kept slice.
* Slicing: the backward closure of event operands and control operands of live
  structured ops must consist of value ops; a dropped or unsupported defining op aborts
  with both locs (`slice error: %x is defined by dropped op arith.mulf at ... but
  reaches the kept slice (mask of tt.load at ...)`).
* Metadata provenance and host reads against skipped launches (above).
* Metadata written in the analyzed launch: a metadata tensor the kernel itself writes,
  or (per input) one whose storage overlaps a written tensor's or a written
  descriptor's base, is refused: its recorded contents are not what the kernel reads.
* Metadata layout: contents are recorded in index order and `load` indices are element
  offsets, so a metadata tensor must be contiguous (stride 1).
* TMA descriptor extents: Layer 2 checks TMA accesses against the descriptor's extents
  (TMA clips there), so per input every in-extent element of a host descriptor, and of
  an in-kernel descriptor with host-side extents that a TMA load reads, must be an
  element of the base tensor's allocation (`kir.md`; the descriptor's element offsets
  `sum c_i x strides_i` from the base's data pointer, checked run by run).
* TMA: the block-tiling conditions of `ir_ops.md` (shared-memory tile = block, same
  NVMMA encoding, message box = tensormap box dividing the block), no `fp4Padded`,
  tensormap `elem_type` byte-sized and of the base pointer's element size, host
  descriptors with the block shape and element size of their type.
* Module: `ttg.num-warps` present, `ttg.threads-per-warp` 32, `ttg.target` = the dump's
  target.
* Unsupported constructs: cross-CTA barriers/arrives (`map_to_remote_buffer`,
  `ctaMask`, 2-CTA/multicast MMA, paired-MMA modules, `num-ctas > 1`), CLC with
  clusters or multi-dimensional grids, multicast/two_cta/im2col TMA, tensor-valued
  integer loads on the slice, unstructured loops or block arguments, non-i32 or
  unsigned (`unsignedCmp`) `scf.for` bounds, pointers or tensors carried by loops or
  yielded by `scf.if` (the `ptrs += BLOCK` / `offs += BLOCK` idioms: compute them from
  the loop index instead), pointers to sub-byte element types, side-effecting inline
  asm, two `tensormap_create` into one scratch buffer.
* Malformed op text the translation does not expect is a `KirError` naming the op and
  its loc, never a crash.
* Event locations: every event's `loc` is a line of the exported sources.
* Case: `elision_flags` are boolean constexprs of the kernel.

## Limitations

* A verdict covers the case inputs and the analyzed specializations only. Variants the
  host selects only for huge inputs (e.g. `USE_I64_IDX` for >= 2^31 elements) or through
  non-default environment switches are checked only where a case selects them (`env`).
* Below TTGIR nothing is checked (layouts, register allocation, PTX).
* A pointer tensor and a host descriptor over the same allocation (e.g. dK pointer
  stores and `desc_dk` TMA stores) are separate KIR tensors; KIR v1 has no way
  to say they alias.
* A descriptor whose extents exceed its base tensor refuses the specialization even if
  no access reaches the excess (accesses are checked against the extents).
* Host reads are tracked through torch calls only (`__torch_function__` and aten ops):
  reading a storage object or a data pointer's memory directly (`ctypes`,
  `numpy.from_dlpack`) escapes it. Address hints come only from integer scalars and the
  recorded contents of int64 tensors (not from pointers split into int32 halves, nor from
  tensors too large to record). Sources are whole storages, and any host read of stale
  data before the analyzed launch, in the input or an earlier one, refuses the input,
  so the checks are conservative (a later `zero_()` does not make data fresh).
* A write to a tensor a move between host and device returned fails the dump even where
  the GPU would not have copied (the data already on the target device): the CPU run
  does not know where the data would live.
* Environment variables read by name in the exported sources are pinned, as are
  compiler switches; variables read by libraries outside those prefixes (torch,
  torchao) are not.
* `test_gpu_parity` needs a GB300 (or B200) host. Without a GPU it skips.
* CLC is analyzed under three schedules, not every schedule the hardware may produce.
* 2-CTA (cluster) kernels need cross-CTA events (target CTA of an arrive, DSMEM copies)
  that KIR v1 does not have.
