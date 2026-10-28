# KIR: the kernel IR shared by the frontends and the Lean library

KIR is the only interface between Layer 1 (frontends) and Layer 2 (Lean).
Frontends emit it as JSON; `frontend/kir.py` checks it against this contract
(also inside the frontends, before a KIR file is written) and
`frontend/emit_lean.py` turns it into Lean constructors of `lean/KV/IR.lean` one
to one (same names, same field order: table fields, then `w`, then `addr`, then
`loc`).

## Changes

Changes since the first KIR v1 contract (`kir_version` stays 1). Frontends must
follow items 1 to 3 and 8.

1. **`addr` on `shl` and `trunc`.** `addr` now also has an effect on `bin` `shl`
   and on `trunc`, scalar and tensor (IR.lean's `trunc` constructors gained the
   field): an `addr` shl that wraps, or an `addr` trunc that loses bits, is an
   overflow (P0-1 FAIL). Set `addr` on every shl and trunc whose value reaches
   an address, a TMA coordinate or a metadata index, as on add/sub/mul. Old KIR
   stays valid, but such wraps then go unnoticed.
2. **Every output tensor needs an `intended_writes` entry (incompatible).** An
   input that does not name a `role: "output"` tensor fails P0-3. The new value
   `"unchecked"` opts a tensor out explicitly (P0-3 then reports a note). The
   KIR stays valid; only the verdict changes.
3. **`expect_tx` in the arriving role.** Transaction bytes are matched to phases
   through the arrivals of the role that issues them: an `expect_tx` must be
   issued by the role that arrives for its phase (`mbarrier.arrive.expect_tx` is
   `expect_tx` then `arrive` in one role), and a `tma_load` by a role that
   issued an `expect_tx` on the slot before it. Both frontends already emit
   this; other shapes fail P0-4.
4. **Optional `target` and `compiler`.** Informational top-level fields (below);
   the TTGIR frontend writes `target`, the CuTe frontend both.
5. **Lean only, no JSON change.** `commit_arrive` arrives asynchronously, when
   the MMAs complete; `threads_per_issue` is the number of threads that run each
   event (warps x 32 when every thread of the role runs it); a global tensor's
   allocation is its elements (coordinates inside the extents) when its strides
   do not overlap, so row padding is outside it.
6. **Written tensors are not inputs (incompatible).** `frontend/kir.py` rejects
   a `gstore`, `gatomic`, `tma_store` or `tma_reduce` on a tensor with
   `role: "input"`: declare it `output` (P0-3 then needs an `intended_writes`
   entry) or `workspace`. Before, P0-3 skipped such a tensor silently.
7. **Lean only, no JSON change.** A metadata `load` whose index lies outside the
   tensor's extents is a P0-1 violation, not an evaluation error: its role stops
   there and P0-2..P0-5 are undecided on that input (references/specs.md). A
   `bar_init` does not count as an event of its role on the slot for P0-2's
   static-waiter rule.
8. **`case.elision_flags` replaces the `region` event (incompatible).** The
   frontends compile each host-selected flag set separately, so no IR has a
   branch on such a flag and no frontend emitted `region`; it is removed
   (`frontend/kir.py` rejects it). P0-5 is now per input: `case.elision_flags`
   lists the flags that select a mask-elided fast path, and an input that sets
   one has its P0-1 and P0-3 checks reported under P0-5 (references/specs.md).
   The frontends fill it from the case file's annotations (`frontend_ttgir.md`,
   `frontend_cute.md`).
9. **Lean only, no JSON change.** A scalar `select` evaluates only the operand
   it chooses (a masked scalar load reads nothing where its mask is false;
   ir_ops.md). P0-2's hang rules count a stale wait at its phase lower bound (it
   may return at once), the lockstep contents only turn stale repeated parities
   into presumed skips, and P0-4's transaction clauses are judged under every
   fitting share assignment (references/specs.md).
10. **P1 KIR (P1-1 TMEM reuse ordered; CuTe only).** A P1 KIR (file suffix
    `.p1.json`, written next to the P0 KIR by the CuTe frontend's second
    translation pass) is a full KIR plus a top-level `tmem` and the `tmem`,
    `tmem_wait` and `named_bar` events below. A P0 KIR contains none of them, so
    P0 KIRs and verdicts do not change. `emit_lean.py` emits
    `tmem := some {...}`, `#eval KV.runP1` and the P1-1 theorems
    (`KV.Holds .tmemReuse`) for a P1 KIR, and the P0 output for a P0 KIR byte
    for byte. In a P1 KIR every `wait`, `arrive` and `commit_arrive` of a role
    of several warps is issued by all its warps (P1-1 takes such an event as
    every warp's and credits arrivals per warp; the CuTe P1 pass splits a role
    whose warps issue one only in part); `frontend/kir.py` checks that an
    `arrive` of such a role gives at least `warps` arrivals.
11. **Thread-block clusters.** An optional top-level `cluster` (default
    `[1, 1, 1]`) and optional fields naming another CTA of the cluster on the
    barrier events: `cta` (the CTA's rank in the cluster) on `arrive`,
    `expect_tx` and `wait`, `cta_mask` (a set of ranks) on `commit_arrive`,
    `tma_load` and `tmem`, and `leader` on `tma_load` (cta_group::2), plus the
    `complete_tx` event (transaction bytes of a copy into another CTA's shared
    memory). Without them nothing changes: IR.lean's constructors gained the
    fields and `emit_lean.py` emits `none` / `false` for them. P0-2 and P0-4
    decide one ledger per cluster (references/specs.md, shared notions and P0-4).

## Top level

```json
{
  "kir_version": 1,
  "kernel": "name",
  "source": "path/to/kernel.py",
  "ir": "ttgir | cute-mlir",
  "target": "sm_100 | sm_103",
  "compiler": {"name": "nvidia-cutlass-dsl", "version": "4.4.2"},
  "grid": [Expr, Expr, Expr],
  "cluster": [2, 1, 1],
  "params": [{"name": "N_CTX", "type": "i32"}],
  "tensors": [Tensor],
  "barriers": [Barrier],
  "roles": [Role],
  "case": {"name": "...", "inputs": [Input], "elision_flags": ["FULL_TILES"]}
}
```

- `source`: the kernel's source file, informational (the TTGIR frontend copies
  the case's `source`, the CuTe frontend names the jit class's module; a
  hand-written `tests/lean` fixture names its own file).
- `target` (optional): the GPU architecture the IR was compiled for (TTGIR
  `ttg.target`; the CuTe frontend compiles for sm_100). `compiler` (optional):
  the compiler release that produced the IR, two strings (CuTe: the
  `nvidia-cutlass-dsl` distribution the dump loaded). Both are informational:
  Layer 2 ignores them and `emit_lean` does not emit them.
- `params`: scalar kernel arguments (runtime ints). Values come from
  `case.inputs`.
- `cluster` (optional, change 11): the thread-block cluster dims, three
  positive ints (default `[1, 1, 1]`, no clusters). Consecutive CTAs form a
  cluster: CTA `(x, y, z)` is in cluster `(x / cx, y / cy, z / cz)` with rank
  `x % cx + (y % cy) * cx + (z % cz) * cx * cy`. A grid the cluster dims do not
  divide does not launch (a kernel error: every spec FAILs on that input).
  Barrier slots are per CTA; an event's `cta` / `cta_mask` names the CTA(s)
  whose slot it uses, by rank (absent: the issuing CTA). A rank outside the
  cluster, or a mask bit at or above the cluster size, is a kernel error. The
  `cta`, `cta_mask` and `complete_tx` forms need a cluster of more than one CTA,
  `leader` an even cluster size.
- Host-side Exprs (`grid`, tensor `shape`/`strides`, barrier `count`/`num`) may
  use only params and constants (no `pid`, `nprog`, `load` or loop variables).
  They wrap like role Exprs but record no events: an `addr` wrap in them is not
  reported.
- `Tensor = {"name", "kind": "global" | "desc", "elem_bytes", "shape": [Expr], "strides": [Expr], "segments": Segments | null, "role": "input" | "output" | "workspace", "metadata": bool}`.
  A tensor that a write or reduction event names is `output` or `workspace`.
  - `kind = "desc"`: a TMA descriptor; `shape`/`strides` are the descriptor's
    global extents (the bounds TMA clips at), plus `"box": [int]` and
    `"elem_bytes"`.
  - `kind = "global"`: P0-1 takes its elements as the allocation (every
    coordinate inside its extent) when the strides do not overlap, else the
    offset span. Padding between rows is therefore outside the tensor.
  - `metadata = true`: an integer tensor whose contents the kernel reads to
    steer control flow or addresses (cu_seqlens, tile tables). Its contents are
    given per input.
  - `segments`: how the row dimension `dim` splits into logical segments, e.g.
    `{"dim": 0, "offsets": "cu_seqlens_k", "per": "batch"}`: segment z is rows
    `[offsets[z], offsets[z+1])` of the 1-D metadata tensor `offsets`, and z is
    the current value of the `work` field `per` (some `work` event must define
    it). P0-1 jagged bounds and P0-3 ownership use it. Any dim may be the row
    dim (e.g. dim 1 of an `[H, tokens]` tensor); for pointer accesses the
    strides must not overlap, so that an element offset has a unique row
    (checked per input).
- `Barrier = {"name", "count": Expr, "num": Expr}` (an array of `num`
  mbarriers).
- `tmem` (P1 KIR only, change 10):
  `{"cols": 32 | 64 | 128 | 256 | 512, "base": "zero" | "relative"}`, the CTA's
  TMEM allocation. `zero`: addresses are absolute (the fake-zero-base idiom,
  `cute.inttoptr(0)`), allowed only with 512 columns; `relative`: relative to
  the allocation (`retrieve_tmem_ptr`). Present iff some `tmem`, `tmem_wait` or
  `named_bar` event exists.

## Roles and statements

```json
{"name": "load", "warps": 1, "threads_per_issue": 1, "body": [Stmt]}
```

`threads_per_issue`: how many threads of the role execute each event (1 for
`elect_one`/lane-0 code, warps x 32 when every thread of the role runs it).
Arrival counts are multiplied by it unless the event says `"per_warp": true`
(then by `warps`).

`Stmt` (tagged by `"op"`):

| op       | fields                                                                                                        |
| -------- | ------------------------------------------------------------------------------------------------------------- |
| `let`    | `var`, `expr`                                                                                                 |
| `for`    | `var`, `lb`, `ub`, `step`, `body` (signed i32 semantics; `ub` exclusive)                                      |
| `while`  | `cond`, `body`, `max_iters` (frontend-proven bound; exceeding it is a verdict error)                          |
| `if`     | `cond`, `then`, `else`                                                                                        |
| `event`  | the Event's fields inline: `{"op": "event", "ev": "wait", "bar": ..., "idx": ..., "parity": ..., "loc": ...}` |
| `assert` | `cond`, `msg` (device-side assert in the IR)                                                                  |

Every Stmt carries `"loc": "file:line"` when the IR has one; events always do.

Variables: one mutable environment per role, with the params pre-bound. `let`
assigns: it creates a variable or overwrites a visible one of the same width
(loop-carried values, `scf.if` results). A variable is readable after a `let` in
its block or an enclosing block (a name first assigned inside a branch or loop
body is not readable after it). Params and `for` variables are read-only.
Conditions are `w`=1 Exprs.

## Expressions

`Expr` is a JSON object tagged by `"e"`; every node has `"w"` = 1, 32 or 64 (bit
width).

| e        | fields            | semantics                                                                                                                                                            |
| -------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `const`  | `v`               | integer literal (two's complement in `w` bits)                                                                                                                       |
| `var`    | `name`            | let/loop variable or scalar param                                                                                                                                    |
| `pid`    | `axis`            | program id                                                                                                                                                           |
| `nprog`  | `axis`            | grid size                                                                                                                                                            |
| `bin`    | `o`, `a`, `b`     | `o` in add, sub, mul, sdiv, udiv, srem, urem, shl, ashr, lshr, and, or, xor, smin, smax, umin, umax (wrapping BitVec semantics; division by zero is a verdict error) |
| `cmp`    | `o`, `a`, `b`     | `o` in eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge; result `w`=1                                                                                                  |
| `select` | `c`, `a`, `b`     |                                                                                                                                                                      |
| `ext`    | `signed`, `a`     | sign/zero extend to `w`                                                                                                                                              |
| `trunc`  | `a`               | truncate to `w` (may carry `addr`)                                                                                                                                   |
| `load`   | `tensor`, `index` | element of a metadata tensor (`index` is a list of Exprs); an index outside the extents is a P0-1 violation and stops the role                                       |

Wrapping is exact. The evaluator also records an overflow at the node's loc
whenever a node marked `"addr": true` differs from the same operation on
unbounded integers: a `bin` add/sub/mul/shl whose signed result wraps, or a
`trunc` that changes the value. Frontends set `addr` on every
add/sub/mul/shl/trunc that feeds an address, a TMA coordinate or a metadata
index.

### Tensor-valued expressions (TTGIR)

`TExpr` = `{"t": ..., "shape": [int], "w": ..., ...}` (every node carries `w`
like an Expr, and `bin` and `trunc` may carry `addr`):

| t                        | fields                                                                                                                                          |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `range`                  | `start`, `end` (`tt.make_range`, 1-D)                                                                                                           |
| `splat`                  | `a` (scalar Expr)                                                                                                                               |
| `bcast` / `expand`       | `a`, `axis`                                                                                                                                     |
| `bin` / `cmp` / `select` | as for scalars, elementwise                                                                                                                     |
| `ext` / `trunc`          | as for scalars, elementwise (same shape as `a`), e.g. the i32 -> i64 extension `tt.addptr` applies to a tensor offset; `trunc` may carry `addr` |
| `full`                   | `v`                                                                                                                                             |

A footprint is evaluated per coordinate. Evaluators may use interval/affine
shortcuts only when they are exact.

## Events

Tagged by `"ev"`; all carry `loc`.

| ev                             | fields                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bar_init`                     | `bar`, `idx`, `count`                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `arrive`                       | `bar`, `idx`, `n` (arrivals per issuing thread, default 1), `per_warp` (default false), `cta` (optional w=32 Expr: the slot's CTA, by rank; a remote arrive)                                                                                                                                                                                                                                                                                                                      |
| `commit_arrive`                | `bar`, `idx` (completion of the issuing role's prior async MMAs arrives once, asynchronously: some time after the event), `cta_mask` (optional w=32 Expr: arrives once on the slot of every CTA in the mask, bit r = rank r)                                                                                                                                                                                                                                                       |
| `expect_tx`                    | `bar`, `idx`, `bytes` (raises the expected transaction bytes; not an arrival: `mbarrier.arrive.expect_tx` is `expect_tx` then `arrive`, in the role that arrives for that phase), `cta` (optional)                                                                                                                                                                                                                                                                                |
| `wait`                         | `bar`, `idx`, `parity`, `cta` (optional: the frontend-made cluster barrier)                                                                                                                                                                                                                                                                                                                                                                                                       |
| `tma_load`                     | `desc`, `coords` [Expr], `bar`, `idx` (bytes = box product x elem_bytes), `cta_mask` (optional: a multicast load, its box lands in every CTA in the mask, each destination's bytes completing on its slot), `leader` (optional bool, default false: cta_group::2, each destination's bytes complete on the slot of the even CTA of its pair, rank with bit 0 cleared)                                                                                                                |
| `complete_tx`                  | `bar`, `idx`, `bytes`, `cta` (transaction bytes completing on the slot of CTA `cta`, asynchronously: a shared::cluster bulk copy or st.async into another CTA; reads and writes no global memory)                                                                                                                                                                                                                                                                                |
| `tma_store` / `tma_reduce`     | `desc`, `coords` [Expr]                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `gload` / `gstore` / `gatomic` | `tensor`, `offsets` TExpr (element offsets), `mask` TExpr or null                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `work`                         | `fields`: {name: Expr} (the work item the role is processing, e.g. batch, head, n-block; used for ownership in P0-1/P0-3)                                                                                                                                                                                                                                                                                                                                                         |
| `tmem` (P1)                    | `inst` (`ld`, `st`, `cp`, `mma`), `warp` (w=32 Expr: the issuing warp, whose lane quarter an ld/st must stay in), `rects` [{`use` (`data`, `d`, `a`, `sf`), `write` bool, `lane` w=32 Expr, `nlanes` (16, 32, 64 or 128), `col` w=32 Expr, `ncols` (1..512)}], optional `site` (the kernel line that reached a DSL helper), optional `cta_mask` (the CTAs whose TMEM it accesses; a cta_group::2 cp or mma). ld: data reads only; st, cp: data writes only; mma: one `d` write, at most one `a` read, zero or two `sf` reads. A role with a cp or mma has one warp |
| `tmem_wait` (P1)               | `ld` bool: `tcgen05.wait::ld` (true) or `::st` of the whole role; completes the role's earlier ld (st)                                                                                                                                                                                                                                                                                                                                                                            |
| `named_bar` (P1)               | `id`, `count` (const Exprs; count a positive multiple of 32), `sync` bool: `bar.sync` (true) or `bar.arrive` of every thread of the role on named barrier `id` (`nvvm.barrier`: id 0, count = the block's threads)                                                                                                                                                                                                                                                                |

## Inputs

```json
{
  "name": "partial_tiles",
  "params": {"N_CTX": 300},
  "grid": [4, 1, 1],
  "metadata": {"cu_seqlens_k": [0, 128, 300]},
  "flags": {"FULL_TILES": false},
  "shapes": {"k": [300, 2, 128]},
  "intended_writes": {"dk": "all_rows_of_segments"}
}
```

- Each input is one concrete run. `name` and `params` (every kernel param,
  signed values, i1 as 0/1) are required. `grid` overrides the grid Exprs,
  `shapes` the tensor shape Exprs; `metadata` gives the contents of every
  metadata tensor as a flat buffer (element (i0, i1, ...) is
  `contents[i0*strides[0] + i1*strides[1] + ...]`).
- `flags` are the host-selected constexprs for that input (the frontend compiles
  one KIR per distinct flag set and pairs it with the inputs that select it).
- `case.elision_flags` (optional, distinct names): the flags that select a
  mask-elided fast path, e.g. `FULL_TILES` or `EVEN_N` (a variant without masks
  or tail handling). An input is elided if its `flags` set one of them to `true`
  (a flag an input does not name is not set); P0-5 then requires P0-1 and P0-3
  of it and reports their violations and notes in their place. The case file
  names them (the frontends check that they are flags of the variant); nothing
  infers them from the IR.
- `intended_writes` names, per output tensor, the region P0-3 expects to be
  covered exactly once: `all_rows_of_segments` (every row of every segment, all
  columns), `all`, an explicit list of row ranges, or `unchecked` (explicit
  opt-out: P0-3 reports a note and checks nothing for the tensor). Every output
  tensor must be named; a missing one fails P0-3. Rows are not clipped to the
  tensor's row extent (`frontend/kir.py` checks a listed range only for
  `0 <= lo <= hi`): rows past it name elements outside the tensor, which P0-3
  reports as never written.
