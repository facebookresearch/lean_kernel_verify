# Case files: how to write one

A case file names the kernel source at a revision, its public host entry point, small
concrete inputs and the annotations the specs need. Verdicts cover only its inputs.
Cases for public kernels go in `cases/`. `tests/e2e/cases/probe_tlx.json` (TLX/Triton)
and `tests/e2e/cases/probe_cute.json` (CuTe DSL, builder in `tests/e2e/kernels/`) are
minimal examples; each `description` says what its inputs trigger.

## Write a new case

Start from the closest case. Formats: TLX/Triton `frontend_ttgir.md`, "Case file"; CuTe
DSL `frontend_cute.md`, "Case files" (a CuTe case also needs a builder).

1. **Source:** the `rev`, the export directory, the repo path, the module and the
   **public host entry point**, so that the real validation, tile tables, grids and flag
   selection run. Name the kernel with `kernel` (TTGIR) or annotate each launched jit
   class under `kernels` (CuTe). Set `num_sms` small (2) so that work items outnumber
   SMs. Non-default variant switches go in `env` (or TTGIR `module_overrides`). A CuTe
   case names the DSL release its sources target in `dsl` (4.4.2, the default, or 4.6.1).
2. **Inputs** (which ones: "Choose the inputs" below): only shapes, dtypes and integer
   metadata matter; data tensors are zeros. TTGIR arguments are `{"dtype","shape"}`,
   `{"dtype","values"}` or `{"ref"}`, and `setup` steps call real host helpers. CuTe
   inputs are the fields the builder reads. Say what each input exercises (`why`, or the
   case `description`).
3. **Annotations** are the only per-kernel input. They name values; they never model
   behavior. `intended_writes` must cover **every** output (`all_rows_of_segments`,
   `all`, `[[lo, hi], ...]` rows, or `unchecked` with a reason for reduce-only or padded
   outputs); an unnamed output fails P0-3. `tensors.<arg>.segments = {dim, offsets, per}`
   marks jagged tensors (P0-1 segments, P0-3 ownership). `work` maps batch/head/block
   to a variable (TTGIR) or a `{"load", "where"}` anchor (CuTe). `elision_flags` (TTGIR:
   top level; CuTe: per jit class) names the flags that select a mask-elided fast path;
   they must be flags of the variant (checked), and P0-5 judges the inputs that set them.
   Optional: `clc_schedules` (TTGIR, default all three).

## Choose the inputs

Verdicts cover only the case inputs, so spend them on the shapes and conditions the
kernel's callers produce: read the callers, the kernel's tests and the docs of its public
entry point. A FAIL on an input no caller passes costs a review and a fix for nothing.

**Include common shapes and conditions:**

* **partial last tiles**: lengths that are not a multiple of the block, and one shorter
  than a block (jagged and varlen lengths are arbitrary), with Q and K lengths that
  differ where the kernel takes both;
* **more work items than SMs** (persistent and CLC schedulers wrap around, as at
  production sizes), with **odd and even** tile counts, since parity bugs show only on
  odd counts;
* **the callers' configurations**: their head counts, head dims and dtypes, and the host
  helpers that build their metadata (tile tables, scale-factor offsets); also the head
  dims the host accepts that no caller uses yet (e.g. one that pads the tile), since a
  model can pick any of them;
* **every host-selected variant the callers reach**: full-tile or mask-elided paths,
  broadcast Q, load balancing on and off, sliding windows of the sizes callers use,
  including lengths for which a window leaves a tile with no query or key in range (all
  autotune configs are analyzed anyway);
* one aligned control input that should pass everything.

**Leave out very unlikely inputs**, unless the user asks for them:

* **empty inputs**: an empty Q (in flash attention, `q_len = 0` or a zero-length query
  segment), a zero-length key segment or sequence, a batch whose segments are all empty;
* **integer-limit and out-of-range values** no caller passes: a window of 2^31 - 1 (as
  "infinite") or a negative window, a head count that the callers' validation rules out
  (e.g. more than 128 heads for a module that takes 2 or 3);
* **misuse**: arguments that the docs or the public wrapper's validation rule out, or
  that contradict the data (a hand-made tile table, a full-tiles flag for partial tiles,
  a dummy for a required tensor);
* **opt-in modes nobody uses**: environment switches, kwargs or explicit configs that no
  caller sets.

A mistake callers plausibly make stays in, e.g. a `max_seq_len` below the longest segment
where callers pass a configured maximum. If the callers do not settle whether a
condition happens, ask the user. LKV's own tests (`tests/`) keep empty and out-of-range
inputs: they check LKV, not a kernel.

Give sibling copies of a kernel the same inputs. Try a new case on one input first
(`--inputs`); a frontend refusal names the op, its IR line and its source `file:line`
(`<out>/logs/`).
