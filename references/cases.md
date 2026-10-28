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
   SMs. Non-default variant switches go in `env` (or TTGIR `module_overrides`).
2. **Inputs:** only shapes, dtypes and integer metadata matter; data tensors are zeros.
   TTGIR arguments are `{"dtype","shape"}`, `{"dtype","values"}` or `{"ref"}`, and
   `setup` steps call real host helpers. CuTe inputs are the fields the builder reads.
   Say what each input exercises (`why`, or the case `description`).
3. **Annotations** are the only per-kernel input. They name values; they never model
   behavior. `intended_writes` must cover **every** output (`all_rows_of_segments`,
   `all`, `[[lo, hi], ...]` rows, or `unchecked` with a reason for reduce-only or padded
   outputs); an unnamed output fails P0-3. `tensors.<arg>.segments = {dim, offsets, per}`
   marks jagged tensors (P0-1 segments, P0-3 ownership). `work` maps batch/head/block
   to a variable (TTGIR) or a `{"load", "where"}` anchor (CuTe). `elision_flags` (TTGIR:
   top level; CuTe: per jit class) names the flags that select a mask-elided fast path;
   they must be flags of the variant (checked), and P0-5 judges the inputs that set them.
   Optional: `clc_schedules` (TTGIR, default all three).

**Degenerate inputs are mandatory**, because verdicts cover only the case inputs and the
bugs show up on these shapes:

* a **zero-length segment**, on each side that can be empty independently (Q and K);
* **partial last tiles**: lengths that are not a multiple of the block, and one shorter
  than a block;
* **more work items than SMs** (persistent and CLC schedulers wrap around), with **odd and
  even** tile counts, since parity bugs show only on odd counts;
* **host-argument extremes** the host accepts: `max_seq_len` shorter than the longest
  segment, window 2^31-1 or negative, more than 128 heads;
* **every host-selected flag variant** the case targets: full-tile or mask-elided paths,
  broadcast Q, load balancing on and off, sliding window (all autotune configs are
  analyzed anyway);
* one aligned control input that should pass everything.

Give sibling copies of a kernel the same trigger inputs. Try a new case on one input
first (`--inputs`); a frontend refusal names the op, its IR line and its source
`file:line` (`<out>/logs/`).
