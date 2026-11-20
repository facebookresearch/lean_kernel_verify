# KIR: the kernel IR shared by the frontends and the Lean library

KIR is the only interface between Layer 1 (frontends) and Layer 2 (Lean).
Frontends emit it as JSON; `frontend/kir.py` checks it against this contract
(also inside the frontends, before a KIR file is written) and
`frontend/emit_lean.py` turns it into Lean constructors of `lean/KV/IR.lean` one
to one (same names, same field order: table fields, then `w`, then `addr`, then
`loc`; except a TExpr's `shape` before its `w`, `tmem`'s `cta_mask` before
`site`, `nb_rep`'s `begin` as `isBegin`, `smem_write`'s `bar`, `idx` and `leader`
as one `slot` triple, `bulk_store`'s `tmap` object as a pair, and the top-level
`tmem.base` as `zeroBase`).

## Changes

Changes since the first KIR v1 contract (`kir_version` stays 1). Frontends must
follow items 1 to 3, 6 and 8.

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
   both frontends write both.
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
   there and the other specs are undecided on that input (references/specs.md). A
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
12. **`ir: "ptx"`.** `ir` gains the value `"ptx"`, for KIR translated from the
    kernel's PTX instead of TTGIR or CuTe DSL MLIR (the PTX translation:
    frontend_ptx.md); IR.lean's `Ir`
    gained the constructor `ptx`. Layer 2 does not read `ir`, so nothing else
    changes.
13. **Lean only, no JSON change.** A barrier array that some `bar_init` names
    needs exactly one executed `bar_init` on each slot that its CTA uses: a slot
    used without one, or initialized twice, is a malformed barrier (P0-2 FAIL),
    so an init under a runtime guard or a loop over fewer slots than `num` is
    checked per CTA. An array no `bar_init` names (hand-written fixtures, a
    frontend-made slot such as the cluster barrier) is taken as initialized once
    before the roles start.
14. **P0-6 KIR (named barriers convergent; both families).** A P0-6 KIR (file suffix
    `.nb.json`, written next to the P0 KIR for every variant, whatever the
    translation, by the PTX frontend's barrier pass `frontend/ptx/nbar.py`; for a
    TTGIR variant whose launches prefer other cluster dims also for that all-preferred
    geometry, `<spec>@cluster<dims>.nb.json`, change 11) has a
    top-level `nbar` (the pass's classification of every barrier site, for audit;
    Layer 2 ignores it), no barriers, as tensors only metadata tensors (1-D flat
    buffers the conditions read with `load`; their contents per input), and only the
    `named_bar`, `nb_open` and `nb_rep` events below (`named_bar` without `tmem`). A
    KIR without `nbar` has no `nb_open` or `nb_rep`, so P0 and P1 KIRs and verdicts
    do not change. `emit_lean.py` emits `#eval KV.runNb` and the P0-6 theorems
    (`KV.Holds .namedBarriers`); `frontend/kir.py` rejects a P0-6 KIR with a
    barrier, a tensor that is not a 1-D metadata tensor or any other event (memory
    events such as `gload`, `gstore`, `gatomic` and TMA included), and checks that
    `nb_rep` events nest in each statement list. Every event of a P0-6 KIR is
    executed by every warp of its role (the pass splits roles so). Amended: a P0-6
    KIR's `named_bar` may say `all` (true only with `sync`): the instruction names no
    thread count, a barrier for all threads of the CTA (`count` is their number), which
    threads that exit release (PTX ISA `exit`); IR.lean's `named_bar` gained the flag
    (false in every other KIR).
15. **P1-2 (async store sources reuse ordered).** A P1 KIR may name the P1 specs
    it is decided for in a top-level `p1` (`["P1-2"]`, as the TTGIR frontend's
    PTX translation writes; absent: `["P1-1"]`, change 10, so the CuTe P1 KIRs do
    not change): `tmem` is present iff `p1` names P1-1, the `tmem` and
    `tmem_wait` events need P1-1 and `named_bar` either. P1-2 adds the events
    `bulk_store`, `bulk_commit`, `bulk_wait`, `smem_write` and `tmap_write`
    (below) and an optional top-level `cta_threads`, the CTA's thread count (idle
    warps without a role included; a named barrier of that count has every
    thread of the CTA). Change 10's rule holds for a P1-2 KIR too (every `wait`,
    `arrive` and `commit_arrive` of a role of several warps is issued by all its
    warps; the PTX translation splits roles so, a split TTGIR partition's parts
    named `<partition>_w<first>_<last>`). `emit_lean.py` emits `p1 := [...]`,
    `cta_threads := some n` and the theorems of the named specs
    (`KV.Holds .storeReuse` for P1-2); a P0 KIR and a P1 KIR without `p1` emit as
    before. Amended: a landing `smem_write` may name the mbarrier slot of its pair
    leader with `leader` (a cta_group::2 load; IR.lean's `slot` gained the flag);
    without it nothing changes.
16. **`assume` (a model assumption).** A new statement `assume` {`cond`, `msg`}: a
    condition the frontend relies on but could not prove statically (e.g. that a
    write misses the bytes of a warp-specialize capture, frontend_ptx.md). Where it
    is false on an input, that is not a certain violation of the kernel: the role
    stops there as at an out-of-bounds metadata load (change 7), its later events
    unknown, so every spec, P0-1 included, is undecided on that input and only the
    findings that are certain anyway are listed: none of the stopped trace's cluster
    (CTA without clusters), whose other roles may have used a value the broken
    assumption made wrong (P0-1, P0-3 and P0-5 on the other clusters' events, P0-2/P0-4
    on complete clusters); the `not decided` lines and the summary name the assumption
    (`a model assumption that does not hold (msg)`).
    It records no event and changes nothing where it holds; IR.lean's `Stmt` gained
    the constructor `assume`. A device `assert` (a certain error where false: every
    spec FAILs) is unchanged.
17. **Handoffs through shared memory.** Two events pass a 32-bit value one role
    stores in shared memory to another role that reads it after an mbarrier wait
    (frontend_ptx.md, "Handoffs"): `handoff_put` {`chan`, `value`}, the putter's new
    value of channel `chan`, and `handoff_get` {`chan`, `var`}, which binds `var`
    (w=32, like a `let`) to the value the channel had at the putter's k-th `arrive`
    on slot s, s the slot of the getter's last `wait` and that wait its k-th on s
    (counting from 0). Lean runs the putters of a CTA before its other roles; the
    roles that put a channel are one role or parts of one, and a role that puts a
    channel reads none (another part of its role may). An
    optional role field `group` names the role a part belongs to (the P1 pass
    splits a role by warps). The value is only taken where the run shows it was
    handed off, else the input is a limit of the model (an evaluation error, every
    spec undecided): every arrival on s is a plain `arrive` of one role group, each
    of its roles arriving with one count per event and those counts summing to the
    slot's, the getter's waits on s alternate parities 0, 1, 0, ..., and each
    putter's next put of the channel after its k-th arrive follows a wait of it that
    pairs, the same way, with an arrive of the getter after the get (the roles of a
    group taken in lockstep, as the P0 ledger takes the threads of a role). A get
    with no value recorded, or putters that recorded different values, are limits
    too. IR.lean's `Event` gained `handoff_put` and `handoff_get`, `Role` the field
    `group`.
18. **Global synchronization.** A `global` tensor with `"sync": {"width": 32 | 64}` is a
    *sync tensor* (`elem_bytes` = width / 8, role `workspace`, not metadata): its elements
    are sync words, which only the sync events name. `gsig` stores a value to a word; on a
    64-bit word the value's high half is a *tag* (`data_lo`: its low half is data). `gwait`
    is a *take*: a spin until a 64-bit word carries the tag `target`, binding its low half
    to `var`. `gclaim` is a claim of a work item from a cursor word (a fetch-add whose old
    value the role uses). `rput` / `rget` write and read field `field` of slot `ord` of a
    shared-memory relay channel `chan` (one write per slot and CTA). A KIR with sync
    tensors may carry `named_bar` events without a top-level `tmem`; inputs may give
    `capacity` and `sync_groups`, the top level `representative`. Decided by P0-7
    (specs.md); IR.lean's `Event` gained the five events, `Tensor` the field `sync`,
    `Input` the fields `capacity` and `syncGroups`.

## Top level

```json
{
  "kir_version": 1,
  "kernel": "name",
  "source": "path/to/kernel.py",
  "ir": "ttgir | cute-mlir | ptx",
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
  `ttg.target`; the PTX translation: the PTX `.target` without its `a`; the CuTe
  frontend compiles for sm_100). `compiler` (optional): the compiler release that
  produced the IR, two strings (TTGIR: `triton` and its version; CuTe: the
  `nvidia-cutlass-dsl` distribution the dump loaded). Both are informational:
  Layer 2 ignores them and `emit_lean` does not emit them.
- `params`: scalar kernel arguments (runtime ints). Values come from
  `case.inputs`.
- `representative` (optional, change 18): `["claims"]` when the KIR's claims take the
  representative schedule (the claiming CTAs claim in turn, specs.md "P0-7"); required
  with `gclaim` events. `scripts/lkv.py` reports every PASS of such a KIR UNDECIDED.
- `cluster` (optional, change 11): the thread-block cluster dims, three
  positive ints (default `[1, 1, 1]`, no clusters). Consecutive CTAs form a
  cluster: CTA `(x, y, z)` is in cluster `(x / cx, y / cy, z / cz)` with rank
  `x % cx + (y % cy) * cx + (z % cz) * cx * cy`. A grid the cluster dims do not
  divide does not launch (a kernel error: every spec FAILs on that input).
  Barrier slots are per CTA; an event's `cta` / `cta_mask` names the CTA(s)
  whose slot it uses (`tmem`: whose TMEM it accesses; `smem_write`: whose shared
  memory it writes), by rank (absent: the issuing CTA). A rank outside the
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
  - `sync` (optional, change 18): `{"width": 32 | 64}`, a sync tensor: every
    element is a sync word that starts at 0 and only the sync events access.
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
  `named_bar` event exists (in a KIR without `p1` or `nbar`; with `p1`, iff it
  names P1-1; never in a P0-6 KIR, whose `named_bar` events need none).
- `p1` (P1 KIR only, change 15): the P1 specs the KIR is decided for, a
  non-empty list of distinct labels `"P1-1"`, `"P1-2"`; absent: `["P1-1"]`.
- `cta_threads` (P1-2 KIR only, change 15): the CTA's thread count, a positive
  multiple of 32 at least the roles' threads (warps no role has, such as the
  idle warps of a Triton warp-specialized kernel, included).
- `nbar` (P0-6 KIR only, change 14): an object, the barrier pass's record for
  audit, which Layer 2 ignores and `emit_lean` does not emit: `block` (the CTA's
  thread dims), `dispatch` (whether the kernel has a warp-specialize dispatch),
  `split_blocks` (blocks copied by node splitting) and `sites` (every named
  barrier instruction reached,
  `{role, op, id, count, loc, ptx_line, status, kind, certain, rep, why}`,
  `status` `counted` or `open`; report.json `nb_sites`). Not in a KIR with
  `tmem` or `p1`.

## Roles and statements

```json
{"name": "load", "warps": 1, "threads_per_issue": 1, "body": [Stmt]}
```

Optional `group` (change 17): the role this one is a part of, when the P1 pass split
a role by warps.

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
| `assume` | `cond`, `msg` (a model assumption, change 16: false on an input, the role stops there, undecided)             |

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
| `bin`    | `o`, `a`, `b`     | `o` in add, sub, mul, sdiv, udiv, srem, urem, shl, ashr, lshr, and, or, xor, smin, smax, umin, umax (wrapping BitVec semantics; division by zero is a `model limit`) |
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
| `expect_tx`                    | `bar`, `idx`, `bytes` (raises the expected transaction bytes; a negative count is an evaluation error; not an arrival: `mbarrier.arrive.expect_tx` is `expect_tx` then `arrive`, in the role that arrives for that phase), `cta` (optional)                                                                                                                                                                                                                                       |
| `wait`                         | `bar`, `idx`, `parity`, `cta` (optional: the frontend-made cluster barrier)                                                                                                                                                                                                                                                                                                                                                                                                       |
| `tma_load`                     | `desc`, `coords` [Expr], `bar`, `idx` (bytes = box product x elem_bytes), `cta_mask` (optional: a multicast load, its box lands in every CTA in the mask, each destination's bytes completing on its slot), `leader` (optional bool, default false: cta_group::2, each destination's bytes complete on the slot of the even CTA of its pair, rank with bit 0 cleared)                                                                                                                |
| `complete_tx`                  | `bar`, `idx`, `bytes`, `cta` (transaction bytes completing on the slot of CTA `cta`, asynchronously: a shared::cluster bulk copy or reduce, st.async or red.async into another CTA; reads and writes no global memory; a negative count is an evaluation error)                                                                                                                                                                                                                  |
| `tma_store` / `tma_reduce`     | `desc`, `coords` [Expr]                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `gload` / `gstore` / `gatomic` | `tensor`, `offsets` TExpr (element offsets), `mask` TExpr or null                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `work`                         | `fields`: {name: Expr} (the work item the role is processing, e.g. batch, head, n-block; used for ownership in P0-1/P0-3)                                                                                                                                                                                                                                                                                                                                                         |
| `tmem` (P1)                    | `inst` (`ld`, `st`, `cp`, `mma`), `warp` (w=32 Expr: the issuing warp, whose lane quarter an ld/st must stay in), `rects` [{`use` (`data`, `d`, `a`, `sf`), `write` bool, `lane` w=32 Expr, `nlanes` (16, 32, 64 or 128), `col` w=32 Expr, `ncols` (1..512)}], optional `site` (the kernel line that reached a DSL helper), optional `cta_mask` (the CTAs whose TMEM it accesses; a cta_group::2 cp or mma). ld: data reads only; st, cp: data writes only; mma: one `d` write, at most one `a` read, zero or two `sf` reads. A role with a cp or mma has one warp |
| `tmem_wait` (P1)               | `ld` bool: `tcgen05.wait::ld` (true) or `::st` of the whole role; completes the role's earlier ld (st)                                                                                                                                                                                                                                                                                                                                                                            |
| `named_bar` (P1, P0-6)         | `id`, `count` (const Exprs; count a positive multiple of 32), `sync` bool: `bar.sync` or `bar.red` (true) or `bar.arrive` (false) of every thread of the role, each warp arriving with its 32 threads, on named barrier `id` (CuTe `nvvm.barrier`: id 0, count = the block's threads); `all` (optional bool, P0-6 KIR only, true only with `sync`): the instruction names no thread count, so threads that exit release it (PTX ISA `exit`) |
| `nb_open` (P0-6)               | `kind` (`lane`, `cond`, `loop`, `operand`, `dispatch`), `certain` bool, `why` string, `id`, `count` (const Exprs or null), `sync` bool: a named barrier instruction the barrier pass cannot count (references/specs.md "P0-6"): reached by the role under a condition its participants may not share (`lane`, `cond`, `loop`), with an id or count that is not a constant or that no set of warps meets (`operand`), or barrier 1 of a warp-specialize dispatch used otherwise (`dispatch`); reached and certain: a certain P0-6 violation, else P0-6 is undecided |
| `nb_rep` (P0-6)                | `begin` bool, `why` string, `key` (optional string, default empty): the begin (true) or end of a region that runs a number of times the model does not fix, the same in every warp of the role; begins and ends nest within one statement list; a non-empty key names the region of the role before warp splitting whose condition is uniform over that whole role (its split roles repeat it alike); Lean reads the key of a begin only (the P0-6 pass leaves an end's key without the `#<n>` its begin may get) |
| `bulk_store` (P1-2)            | `kind` (`tensor`, `tensor_reduce`, `bulk`, `bulk_reduce`), `warp` (w=32 Expr) and optional `lane` (w=32 Expr; absent: the elected lane of an `elect.sync` of the full warp): the issuing thread; `sym` (shared symbol) and `src` (Expr: byte offset in it) with `bytes` (positive int): the shared-memory bytes the store reads; optional `tmap` {`base` (global buffer), `addr` (Expr: byte offset)}: the 128-byte tensormap of an in-kernel descriptor it reads. Joins the issuing thread's open bulk async-group |
| `bulk_commit` (P1-2)           | (no field) `cp.async.bulk.commit_group` of the role (every issuing thread of the role's bulk stores executes it)                                                                                                                                                                                                                                                                                                                                                                   |
| `bulk_wait` (P1-2)             | `n` (non-negative int), `read` bool: `cp.async.bulk.wait_group{.read} n` of the role: completes the groups of all but the `n` most recent commits                                                                                                                                                                                                                                                                                                                                 |
| `smem_write` (P1-2)            | `sym`, `offsets` TExpr (byte offsets), `mask` TExpr or null, `bytes` (positive int: bytes at each offset), `warp`, optional `lane` (as `bulk_store`'s), optional `sync` bool (a warp-synchronizing write of every lane of `warp`, no `lane`), optional `async` bool (the landing of an asynchronous copy issued here), optional `cta` (w=32 Expr: another CTA of the cluster), optional `bar` and `idx` (both or neither, only with `async` and without `cta`: the mbarrier slot the copy completes its bytes on, in the issuing CTA or, with the optional bool `leader`, in its pair leader's (a cta_group::2 load, as `tma_load`'s `leader`); its load is the role's next transaction event on that slot, only `gload`s between)                                                                                              |
| `tmap_write` (P1-2)            | `base` (global buffer), `addr` (Expr: byte offset), `warp` (w=32 Expr): the 128-byte tensormap written by every lane of `warp` in one warp-synchronizing instruction (`tensormap.cp_fenceproxy`)                                                                                                                                                                                                                                                                                 |
| `handoff_put`                  | `chan` (string), `value` (w=32 Expr): the role's new value of handoff channel `chan` (change 17) |
| `handoff_get`                  | `chan` (string), `var` (string, bound at w=32): the value of `chan` at the putter's arrive that this role's last wait pairs with (change 17) |
| `gsig`                         | `tensor` (a sync tensor), `index` (w=32/64 Expr: element offset), `value` (the word's width), `data_lo` (optional bool, 64-bit words): a store of `value` to the word (change 18) |
| `gwait`                        | `tensor` (a 64-bit sync tensor), `index`, `target` (w=32), optional `var` and `w` (32) together: a take, which returns once the word's tag is `target` and binds `var` to the value half of that tag's store (change 18) |
| `gclaim`                       | `tensor`, `index`, `value` (the add), `var` and `w` (the word's width): a claim, `var` its old value (change 18) |
| `rput`                         | `chan` (string), `ord` (w=32/64 Expr: the slot), `field` (int >= 0), `value` (Expr): the role writes field `field` of slot `ord` of relay channel `chan` (change 18) |
| `rget`                         | `chan`, `ord`, `field`, `var`, `w` (1, 32 or 64): `var` := that field of the slot, written by an earlier role of the CTA or earlier in this role (change 18) |

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
- `capacity` (optional, change 18): `{"ctas": int, "clusters": int (optional)}`, how
  many CTAs (and clusters) of the launch can be resident at once (the occupancy of the
  variant on the target). P0-7 needs it when a take needs a store of another CTA.
- `sync_groups` (optional, change 18): lists of linear CTA indices that synchronize outside
  the model (the teams of a protocol region): the claims' turns must agree with it
  (specs.md "P0-7", `claimTurnsFeasible`).
- `intended_writes` names, per output tensor, the region P0-3 expects to be
  covered exactly once: `all_rows_of_segments` (every row of every segment, all
  columns), `all`, an explicit list of row ranges, or `unchecked` (explicit
  opt-out: P0-3 reports a note and checks nothing for the tensor). Every output
  tensor must be named; a missing one fails P0-3. Rows are not clipped to the
  tensor's row extent (`frontend/kir.py` checks a listed range only for
  `0 <= lo <= hi`): rows past it name elements outside the tensor, which P0-3
  reports as never written.
