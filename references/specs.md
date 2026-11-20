# Specs: definitions, violations, notes, proof method, limits

Layer 2 (`lean/KV`, core Lean + Std) evaluates a KIR program
(`references/kir.md`) on the concrete inputs of its case file and decides five
P0 specs per input, on the P0-6 KIR of every variant (both families) the sixth P0
spec P0-6 "named barriers convergent" (section "P0-6" below), and on a P1 KIR the
P1 specs it names: P1-1 "TMEM reuse ordered" (CuTe; section "P1-1" below) and P1-2
"async store sources reuse ordered" (the PTX translation of TLX/Triton kernels;
section "P1-2" below). On a P0 KIR with sync tensors (kir.md change 18) it also decides
P0-7 "global waits reached, co-residency" (section "P0-7" below). This file is the
reference for what each verdict means. The Lean definitions are the authority; each section names them.

## What a verdict covers

- One kernel (one KIR of one compiled flag set: its P0, P0-6 or P1 KIR, for a launch
  with a preferred cluster also per cluster geometry) on one concrete `Input`. No
  other shapes, no symbolic parameters.
- KIR semantics as implemented by `KV.eval` (`lean/KV/Eval.lean`): exact
  `BitVec` arithmetic at each node's width; every role of every CTA runs to
  completion independently (a role stops early only at an out-of-bounds metadata
  load, which leaves every spec but P0-1 (its FAIL) undecided unless it finds
  something certain on the evaluated events, or at an `assume` that does not hold,
  which leaves every spec undecided unless it finds something certain in a cluster
  where no role stopped at one); one event trace per (CTA, role).
- **FAIL means a violation that is certain in the extracted model**, whatever
  the data values. Findings that can be harmless depending on data or on how the
  kernel uses the result are **notes**: printed with the report, never a FAIL.
  The theorems prove exactly the FAIL-free Props, so notes do not affect them.
- **Four outcomes.** PASS: the theorem is proved. FAIL: its negation is proved and
  the report lists a certain counterexample (`SpecResult.certain`), or the input's
  evaluation stopped at a certain error of the kernel ("Evaluation errors",
  kind `kernel error`). Refuted without either, a spec is neither proved nor
  violated (its negation holds of the model, with nothing certain behind it), and Lean prints `UNDECIDED <input> <spec>`: every spec of an input
  whose evaluation stops with a `model limit` or an `unexpected` error, a spec
  that finds nothing certain on an input with an out-of-bounds metadata load or
  an `assume` that does not hold ("Undecided specs"), P0-2 and P0-4 with nothing
  certain outside a cluster whose barrier ledger does not converge, P0-4 with
  nothing certain outside a slot whose bytes from another CTA balance only if a
  landing completes a later phase than its issue order shows (P0-4), P0-6 with
  only `nb_open`s not marked certain or a pairing it cannot decide (one that
  depends on timing, or on repeated regions; P0-6), P1-1 with only U1 pairs or
  F1 pairs that a named barrier used with two counts, or another round of a
  ledger that does not converge, might order, and P1-2 with only U1-U4 pairs or
  F pairs such a named barrier or ledger might order. `lkv.py` reports such a spec as
  ERROR after an `unexpected` evaluation error (the case or LKV must be fixed),
  else as UNDECIDED (a known limit, not an error); it also reports an
  `INCONSISTENT` verdict as an ERROR, and every spec of each input of a variant
  without verdicts as UNDECIDED if the frontend refuses the variant by design,
  else as an ERROR. A run's outcome and exit status ("Reports") is FAIL (1) on
  any FAIL, else ERROR (2) on any failure, else UNDECIDED (0) on anything a
  known limit leaves open, else PASS (0).
- Roles are not interleaved. P0-2 and P0-4 are stated over each role's own event
  order, per-cluster totals (per CTA without clusters) and the phase ledger of
  `lean/KV/Spec/Ledger.lean`; they are not a race check, and they catch only the
  deadlocks certain in the lockstep model (the others are P1 bugs that no spec
  checks: Limits).
- **P0-2 and P0-4 are relative to the ledger algorithm.** Their Props quantify
  over the ledger's classification of waits (re-check, skip, presumed skip,
  stale, late join, lap), its lockstep phase contents, its optimistic progress
  pass and its attribution of transaction bytes to phases. The soundness of that
  algorithm against mbarrier semantics is argued in Ledger.lean, not proved in
  Lean: a P0-2/P0-4 theorem says "the ledger finds no violation", not "no
  interleaving violates the protocol" (references/design.md, trust base).
- Proof: `#lkv_check kernel input <specs>` (lean/KV/Command.lean) evaluates the
  kernel on the input once, compiled and run as `native_decide` runs its decision,
  adds the axiom that the evaluation's decisions are what it computed (Lean 4.34's
  `native_decide` adds `e = true` the same way) and proves, per spec, the theorem
  `KV.Holds .<spec> kernel input` (PASS) or its negation (FAIL, UNDECIDED) from it:
  one evaluation per input whatever the number of specs, its analyses (barrier and P1
  ledgers, P0-1's, P0-3's and P0-5's decisions, P0-3's targets and ledgers, P0-6's
  analysis, P1-1's and P1-2's pair searches, P0-7's analysis) computed once for the report and every
  decision (`Check.lean` `Shared`; `checkAll_snd` proves the decisions computed from them
  are the specs' decisions) (the form with one
  `native_decide` theorem per spec, which evaluates the input once per spec, is
  `emit_lean.py --legacy-theorems`). This trusts the Lean compiler and the native
  `libKV.so` built from the same sources (T1). The evaluator is total (no `partial`), so the theorem
  statement has its intended meaning inside Lean's logic.

## Shared notions

| Notion                                    | Definition                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CTA                                       | one program instance `(x, y, z)` of the effective grid (`Input.grid`, else the grid Exprs). With `cluster` dims `(cx, cy, cz)` consecutive CTAs form a cluster; the CTA's rank in it is `x % cx + (y % cy) * cx + (z % cz) * cx * cy`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| slot                                      | one mbarrier of one CTA: barrier array name and element index `idx` (`0 <= idx < num`); an event's `cta` / `cta_mask` names the CTA(s) of its cluster, by rank (default: the issuing CTA)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| arrivals                                  | `arrive`: `n x threads_per_issue` (`n x warps` if `per_warp`); `commit_arrive`: 1 per CTA of its mask, asynchronous (it arrives when the MMAs complete, some time after the event). `expect_tx`, `tma_load` and `complete_tx` are not arrivals                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| completions K                             | arrivals / count (integer division), per CTA and slot                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| first parity `p0`                         | parity of a role's first wait on a slot; `p0 = 0` consumer convention, `p0 = 1` producer convention (passes on the fresh barrier)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| phase of a wait                           | phase k of a slot completes when its arrivals reach k x count and, on a slot with transaction bytes, its bytes have landed; a wait with parity p returns once the number of completed phases has parity 1-p. A role's first wait on a slot synchronizes with phase 1 (`p0 = 0`) or 0 (`p0 = 1`); a wait whose parity differs from the role's previous wait on the slot, with the next phase. A wait that repeats the previous parity (previous phase c) re-checks phase c if phase c+1 cannot complete before it, skips to phase c+2 if phase c+1 surely completed before it, is a presumed skip if phase c+1 surely completed before it only with the lockstep credit (it counts as phase c+2 with lower bound c; it is no P0-4 violation, and P0-2's hang checks count it at c+2), and is stale otherwise (it counts as phase c+1 with lower bound c). A first wait, a wait with a new parity or a skip that would synchronize with phase k while phase k+1 surely completed before it returns at k+2 or later, so it synchronizes with k+2 (repeated while that holds): a first wait so moved joins late; a later one is a lap (P0-4 note)                                                                                                                                                                                                                   |
| cannot complete / surely completed before | happens-before over the role traces of the cluster's CTAs (one CTA without clusters) (`KV.arrivalsMayPrecede`, `KV.saturate`, `KV.completedIn`), each role taken as one thread of control. An event is definitely after wait w if it follows w in its role, or follows a wait (of any role) that needs more arrivals than those not definitely after w. Phase k surely completed before an event if a wait on the slot for phase k or later surely returned before it, or, on a slot whose phases complete by arrivals alone (no expect_tx/tma_load bytes, no commit arrivals), k x count arrivals surely precede it: arrivals are necessary for a phase to complete, not sufficient. An event surely precedes another if it precedes it in its role, or precedes an arrival that a wait which surely returned before the other needs: phase k needs k x count minus all other roles' arrivals over the run. Waits are credited with lower bounds of their phases (re-checks, stale waits and presumed skips at c, skips at c+2), refined until a round changes neither them, the classification nor the lockstep contents, for at most (the cluster's waits + 2) rounds (`KV.ledger`); a cluster without that fixpoint leaves P0-2 and P0-4 undecided ("Undecided specs")                                                                                                                                                                                    |
| lockstep credit                           | a second happens-before closure (`saturate` with `lockstep = true`) in which phase k also needs each arriving role's least share, over all fitting assignments, of the phases up to min(k, lockstep) of the lockstep contents. The contents are an assumption (one fixed share per role and phase), not a certainty, so this closure decides only whether a repeated parity is a presumed skip instead of stale, and serves as evidence for zero shares and initialization arrivals (it can only make contents exist). Phase lower bounds (and with them happens-before) never use it. A presumed skip counts at c+2 for the hang checks of P0-2 (its `assumedPhase`): if the contents are right the role needs that phase, and if not the wait is stale, so a hang found that way is a violation either way. The credit thus replaces a stale FAIL by the verdict under the contents, and never fails an input that passes without it                                                                                                                                                                                                                                                                                                                                                                                                                          |
| waited phases                             | phases up to R* = the highest phase some role's waits need on the slot                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| lockstep contents                         | over the waited complete phases (the first min(R*, K)): each arriving role contributes the same number of arrivals s_r to every one of them, ending exactly at one of its arrive events after each, and the s_r sum to the count (`KV.lockstepShares`); or, if no such shares exist, an initialization arrival forms phase 1 by itself and shares cover the phases after it (`KV.lockstepContents`). A role with arrivals takes s_r = 0 only if its first arrival surely follows the last waited phase. An initialization arrival is the first `count` arrivals of one role, reached at one of its arrive events, such that nothing else lands in phase 1: every other role's first arrival on the slot surely follows phase 1's completion, or the initialization arrivals contain no commit and surely precede every other role's first arrive event (a commit lands after its event). E.g. FA4's dQ ring: warp 4 pre-arrives once at init, the store warp makes every later release; the releases may be commits. Evidence for a zero share or an initialization arrival uses happens-before credit only from slots whose contents need none, never from the slot itself. Several share assignments can fit (count 8 as 2+2+4 or 1+1+6): the representative one has the most participating roles; the fitting ones are those whose zero shares are evidenced |
| bytes of a phase                          | an `expect_tx` belongs to the phase its role's next arrival falls in under a fitting share assignment; a `tma_load` to the phase of its role's latest `expect_tx` on the slot. A role of another CTA whose only events on the slot are transaction landings (`tma_load`, `complete_tx`; a lander: the other CTA of a cta_group::2 pair, a multicast destination, a DSMEM copy) is attributed by its issue order: a landing belongs to phase f + 1 (at most K + 1), f the highest phase of the slot that surely completed before it is issued (sound happens-before with the precise needs of P1-1, whose cuts here also use the bytes a phase needs: P0-4, `Ledger.lean` "Bytes carry order"; `TxRec.floor`), its first possible phase; the bytes per phase need not be equal, but none is negative (a negative `expect_tx` or `complete_tx` count is an evaluation error). Phase K + 1 is the last one: it never completes, so bytes issued after phase K land in it (with K = 0 every byte belongs to phase 1). Bytes of a role with share 0 cannot be attributed. The bytes are attributed under the representative assignment and, if that breaks a transaction clause of P0-4, under the other fitting ones (distinct on the roles with transaction events, at most 64 in all, `KV.maxShareVectors`) until one keeps them all; if none does, P0-4 shows the first attribution that breaks them only through a lander's lower bounds as not decided (P0-4), else the representative one |
| optimistic progress                       | every role runs as far as its waits allow when every arrival takes effect as soon as its event is reached and transaction bytes are ignored; a wait passes once the arrivals on its slot reach its phase lower bound x count, presumed skips counted at c+2 (`KV.deadlocks`, `assumedPhase`). A wait still blocked when nothing can move never returns in the lockstep model (or a presumed skip it depends on is stale)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| static waiter                             | a role whose code (any branch or loop body) waits on the barrier array. It is expected to observe every slot of the array on which it has no wait, arrival or transaction event (a `bar_init` does not count: TTGIR emits every one in the `default` role); a role with an arrival or transaction bytes on a slot is one of the slot's producers (one array can hold several logical barriers: the CuTe frontend emits each pipeline's full and empty barriers as one array)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| complete trace                            | a role's event trace that ran to its end; a role whose metadata `load` index lies outside the tensor stops there (its later events depend on the value read), as does one whose `assume` does not hold. P0-2, P0-4, P1-1 and P1-2 use the clusters (CTAs without clusters) whose role traces are all complete; P0-6 pairs only such CTAs; after an `assume` stop, P0-1, P0-3 and P0-5 use only the clusters where no trace stopped at one                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| elided input                              | an input that sets (`true` in its `flags`) one of the case's `elision_flags`: the host selected a mask-elided fast path, compiled as a variant of its own (`Input.elisions`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| work item                                 | fields of the role's last executed `work` event (dynamic)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| segment `z`                               | rows `[offsets[z], offsets[z+1])` of `segments.dim`; `z` is the current work item's field `segments.per`. A work item that does not name the field, or names no segment, owns no segment                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| allocation                                | of a global tensor: its elements (offsets `sum c_i x s_i` with every coordinate `0 <= c_i < n_i`) when the strides do not overlap (every dim with more than one coordinate has a positive stride larger than the span of the dims with a smaller or equal stride), else the span `[sum min(0,(n_i-1)s_i), sum max(0,(n_i-1)s_i)]`; empty if some `n_i <= 0`. Extents `n_i` are `Input.shapes`, else the shape Exprs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| row of an offset                          | for a segmented global tensor, the coordinate along the segment dim: remove the dims of larger stride greedily (largest first), then divide by the row stride. The evaluator first checks, per input, that this is exact: the row stride and every larger one exceed the span of all other dims with a smaller or equal stride (dims with one coordinate are ignored; else evaluation error)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| TMA box                                   | coordinates `coords[i] + [0, box[i])` per dim; TMA clips each dim at the descriptor extents (zero fill on load, dropped on store), so only in-extent coordinates are accessed and a box never leaves the descriptor                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| element                                   | an element offset (`sum coord_i x stride_i`); two writes collide iff they hit the same offset                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| plain write                               | `gstore` (masked-in lanes) or `tma_store` (in-extent box elements)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |

## P0-1 global accesses in bounds (`KV.Bounds.Holds`)

FAIL, on an input that is not elided (on an elided input these are P0-5's), for
any event (`Bounds.EventOk`):

- `gload` / `gstore` / `gatomic`: a masked-in lane outside the tensor's
  allocation;
- plain writes: a written row (gstore lane, in-extent tma_store row) outside the
  segment owned by the issuing work item;
- an `overflow` event: an `addr` add/sub/mul/shl whose signed result wraps, or
  an `addr` trunc that changes the value, at the node's width (any width; the
  frontends set `addr` only on address paths, which are i32 or i64), including
  lanes a later mask disables, but not an operand that a scalar `select` does
  not choose (it is not evaluated). Host-side Exprs record no events, so a wrap
  in a grid, extent, stride or barrier Expr is not reported (Limits).

FAIL, on every input (elided or not): a metadata `load` whose index lies outside
the tensor's extents (an out-of-bounds global read), with its loc and index. The
role stops there; the other roles and CTAs are still evaluated, and P0-1 lists
their violations as well. The other specs are undecided on such an input
(below).

Notes (`Bounds.notes`; on an elided input P0-5's):

- read spill: `gload` lanes inside the allocation, or in-extent `tma_load` rows,
  outside the work item's segment. Harmless only if the values are masked before
  use; that is value-level and not checked by P0;
- reduce spill: `gatomic` lanes / in-extent `tma_reduce` rows outside the
  segment. Harmless only if the reduction leaves the values there unchanged
  (e.g. it adds zero; KIR `gatomic` does not record the operation, so an
  exchange or compare-and-swap there overwrites).

## P0-2 role trip-count agreement (`KV.TripCount.Holds`)

Per cluster (a CTA without clusters; with complete role traces) and slot touched by
any event (`TripCount.SlotOk`), with C the declared count and R the phase lower bound of a
waiting role's last wait, with presumed skips counted at the phase they skip to
(`RoleSlot.needed`). FAIL:

1. a malformed barrier: C <= 0, a `bar_init` of the slot with another count, or,
   on an array that some role's code initializes (`SlotLedger.initDeclared`), a
   slot without exactly one `bar_init` in its CTA (used uninitialized, or
   re-initialized: undefined in PTX);
2. a hang: some role needs R > K completions, or a wait never returns in the
   optimistic progress pass (a cross-role wait cycle in the lockstep model, e.g.
   a producer waiting on its empty barrier with the consumer's parity);
3. arrivals of the waited complete phases without lockstep contents: a waited
   phase completes early, inside an arrive event (the count does not match the
   arrivals per phase), without some role's share (the arriving roles disagree
   on their per-phase share), or with arrivals of a role that has no share and
   no evidence that they follow the waited phases (e.g. count 1 where two roles
   each arrive once per tile);
4. a missing wait on a data-ready slot: a static waiter with no wait, arrival or
   transaction event on a slot that completes and receives TMA bytes.

Notes (`TripCount.notes`), with the slot and loc: phases that complete after the
phase a role's last wait needs (e.g. CUTLASS `producer_tail`'s trailing
`arrive_and_expect_tx`, an empty barrier's final releases, a role that consumes
fewer tiles than produced), a trailing partial phase nobody waits for (arrivals
not a multiple of C), and a static waiter with no wait, arrival or transaction
event on a completing slot without TMA bytes.

Rule 2 rationale: a first wait with parity 1 passes on the fresh barrier, so it
needs no completion. A repeated parity that provably re-checks the phase it
already passed (e.g. two users of one shared buffer guarded by the same wait)
needs no new completion, and one that provably skips a phase (a role that waits
only inside its loop body, ordered by other waits) needs two. A stale wait may
return at once (a re-check) or wait for a later phase, depending on timing, so
it counts at its lower bound c: a hang after it is a FAIL only if certain, and
its parity bug is P0-4's certain FAIL. A presumed skip counts at c+2: that is
the phase it waits for if the lockstep contents are right, and if they are not
it is stale, so the hang is a violation either way; the message then names the
presumed skip. For the waited phases and the notes (R < K, lockstep contents,
bytes of a phase) both count at the phase the role presumably meant (c+1 or
c+2). A wait whose next phase surely completed before it needs the phase after
that: a role joining late with parity 0 after phase 2 surely completed needs
phase 3, and a lap past the last completion hangs. The optimistic progress pass
uses the same bounds, so every wait it reports blocked needs arrivals that no
execution of the lockstep model can provide before it (or a presumed skip it
depends on is stale).

## P0-3 output coverage exactly once (`KV.Coverage.Holds`)

FAIL:

- an output tensor (`role: "output"`) that the input's `intended_writes` does
  not name;
- for every output tensor with an intended region (rows of every segment / all
  rows / the listed row ranges, all other coordinates): an intended element no
  plain write writes, or that plain writes write more than once;
- a plain write to an element outside the intended region.

These are checked on inputs that are not elided (`Coverage.Ok`; on an elided
input they are P0-5's). Counts are over all CTAs and roles, so two CTAs writing
one element is a violation. Writes outside the segment of their work item are
P0-1.

Intended rows are taken as given, not clipped to the tensor's row extent
(`Coverage.intendedRows`; `frontend/kir.py` checks a listed range only for
`0 <= lo <= hi`). A range or segment that reaches past the extent names elements
outside the tensor: P0-3 reports those no store writes as never written (a TMA
store is clipped at the extent), and a pointer store to one is a P0-1 FAIL.

Notes (`Coverage.notes`; on an elided input P0-5's): `gatomic`/`tma_reduce`
elements outside the intended region (harmless only if the reduction leaves the
values there unchanged), and tensors opted out with
`intended_writes: "unchecked"`.

## P0-4 mbarrier protocol (`KV.Barrier.Holds`)

Per cluster (a CTA without clusters; with complete role traces) and slot (`Barrier.SlotOk`). FAIL:

- a wait parity other than 0 or 1;
- a stale wait: a repeated parity that is neither a provable re-check nor a
  provable or presumed skip (shared notions); a role that changes parity on
  every wait alternates `p0, 1-p0, p0, ...`;
- on a slot that receives TMA bytes, a role whose first wait synchronizes with
  phase 0 (parity 1 on the fresh barrier): it passes before any byte lands. A
  first wait with parity 1 that joins late (phase 1 surely completed before it)
  synchronizes with phase 2 or later and is not one;
- in a waited phase: `expect_tx` bytes != the bytes of the `tma_load`s landing
  in it (`box product x elem_bytes`, clipped or not);
- TMA bytes landing in a phase nobody waits for (beyond the highest waited phase
  R*), or on a slot nobody waits on: no wait synchronizes with that load, so its
  data is read unsynchronized or never, and it may still write shared memory
  after the CTA exits. Typical causes: an arrive before its `expect_tx` (every
  wait returns one phase before its data), an extra completion that shifts the
  waits against the data, a consumer that waits for fewer tiles than are loaded;
- a `tma_load` not covered by earlier `expect_tx` bytes of its role in its
  phase, or whose role issued no `expect_tx` on the slot before it (stricter
  than PTX, which lets the transaction count go below zero while a phase is
  pending; the frontends emit `arrive.expect_tx` before the copies), or bytes
  that cannot be attributed to a phase. A lander (a role of another CTA of the
  cluster, shared notions) issues no `expect_tx`: its bytes need not be covered.

The last three clauses (`SlotLedger.TxOk`) are judged on the bytes of a phase
(shared notions): they fail only if they fail under every fitting share
assignment tried, because the lockstep model admits each of them.

A lander's phase is a lower bound: the landing cannot complete an earlier phase,
but it can complete a later one if the phase it is put in completes without it.
So (`Ledger.lean`, "Landers"):

- the clauses holding under that attribution prove it: by induction over the
  phases, phase g can only receive the landings put in g (later ones are issued
  after g completed, earlier ones landed in their own phases) and needs all of
  them, so every landing lands where it was put (PASS);
- a failure is certain if it does not rest on the lower bounds: a clause of the
  slot's own roles, a landing whose first possible phase nobody waits for, a
  first unbalanced waited phase that receives fewer bytes than it expects (no
  later landing can reach it), or waited phases whose bytes differ from their
  `expect_tx` bytes in all;
- otherwise (the first unbalanced phase receives more bytes than it expects
  while the totals match) a landing may complete a later phase, depending on
  timing the ledger does not order: P0-4 lists one `not decided on this input`
  line for the slot (`SlotLedger.txUndecided`, `Barrier.undecided`) instead of
  its transaction lines, and is UNDECIDED unless something else is certain. E.g.
  a peer that issues its loads without waiting for the phases before
  (`tests/lean/cluster_pair_prologue` `peer_unordered`). With several fitting
  share assignments the slot is not decided as soon as one of them fails only
  so (`tests/lean/cluster_lander_shares`): a transaction clause fails only if it
  fails certainly under every assignment.

"Surely completed before" uses the precise needs (`Ledger.lean`, the arrivals
of other roles that may precede a wait's return instead of their run totals):
in a multicast pipeline every CTA's consumer arrives on every CTA's empty slot,
and the run totals of two such arriving roles order nothing
(`tests/lean/cluster_multicast_pipeline`). Their cuts also use transaction bytes
(`Ledger.lean`, "Bytes carry order"): a phase completes only when its
transaction count is back to 0, so a wait for phase k cannot return before
loads and landings bringing the `expect_tx` bytes that surely precede phase k's
completion were issued (those of each role issued before the arrivals of it that
phase k needs, `SlotLedger.bytesNeeded`). E.g. TLX's cluster launch control:
the rank-0 CTA multicasts each try_cancel response into the other CTA, whose
roles' waits for it are ordered after the try_cancel only by its bytes
(`tests/lean/cluster_clc_response`; before, P0-4 was not decided there). On a
slot with a lander only loads that the wait's own return does not surely
precede count (a load issued after the wait returned lands after its phase
completed): in a cta_group::2 pipeline whose empty slot also counts a second
role's release, the leader's own producer may load the next phase meanwhile,
and those bytes must not stand in for the peer's
(`tests/lean/cluster_pair_two_release`). The
induction needs every landing's bytes to be non-negative: PTX transaction counts are unsigned, and a negative
`expect_tx` or `complete_tx` count stops the evaluation (`unexpected`).

The attribution follows the kernel's issue order whatever the bytes per phase: a
cta_group::2 backward whose prologue also loads K into the first phase (peer
bytes 49152, 16384, 16384, ... on the leader's barrier) balances phase by phase.
Before, a lander had to land the same bytes in every waited phase; no share fit
3 or more such phases, and P0-4 reported a certain FAIL.

The unwaited-phase rule has no bound below K + 1 because none is needed: phases
beyond K + 1 cannot start (phase K + 1 never completes), so every byte lands in
a phase up to K + 1; a load in a completing phase above R* (<= K) and one in the
never-completing phase K + 1 are both loads no wait observes. Only loads count:
`expect_tx` bytes with no load in an unwaited phase are CUTLASS
`producer_tail`'s pending transaction and stay a note.

Notes (`Barrier.notes`): laps, `expect_tx` bytes with no load in a phase nobody
waits for (transaction bytes pending after the last wait, e.g. CUTLASS
`producer_tail`), and `expect_tx` bytes on a slot nobody waits on. A lap is a
wait after the role's first one on the slot that synchronizes with phase k+2 or
later instead of the phase k after its previous wait, because phase k+1 surely
completed before it; the note names the event that shows it (a wait for that
phase, or the arrive that completes it). It is not a FAIL: parity does not say
which phase a role means. E.g. an MMA warp that waits once per tile on a drain
barrier's empty phase while the compute roles complete three phases per tile
means phases 2, 5, 8 (its parity toggles because three is odd): a lap FAIL would
reject that correct protocol. P0-2 sees the moved phase, so a lap
that waits past the last completion fails there.

## P0-5 mask-elision soundness (`KV.MaskElision.Holds`)

The frontends compile each host-selected flag set separately (constexpr branches
are resolved at compile time), so a mask-elided fast path, e.g. the variant
without masks and tail handling that the host picks for `FULL_TILES=True`, is a
KIR of its own, paired with the inputs whose flags select it. The case names the
flags that select such a path (`case.elision_flags`, `references/kir.md`); an
input that sets one is elided. P0-5: an elided input must satisfy P0-1 and P0-3
(`Bounds.EventOk` on every event and `Coverage.Ok`): the variant the host
selected for it handles it. Inputs that are not elided pass vacuously.

P0-1, P0-3 and P0-5 partition the bounds and coverage violations by input: on an
elided input they, and the P0-1/P0-3 notes, are reported under P0-5 only (the
report's summary line names the elision flags set), so a P0-5 FAIL says the host
selected the elided variant for an input it does not handle (e.g. a partial last
tile: an access past the end, or rows nobody writes), while P0-1 and P0-3
describe the inputs on general variants. An out-of-bounds metadata load is
P0-1's on every input (a scalar read, not an elided mask; it leaves P0-2..P0-5
undecided).

## P0-6 named barriers convergent (`KV.NamedBarriers.Holds`)

A CTA-level named barrier (`bar.sync`/`barrier.sync[.aligned] a{, b}`,
`bar.arrive`/`barrier.arrive a, b`, `bar.red`, also spelled `bar.cta`/`barrier.cta`; PTX
ISA "bar, barrier") completes an instance when `b` threads arrived (no `b`: every thread
of the CTA, and threads that exit
release it, PTX ISA `exit`); `sync` and `red` wait for the completion, `arrive` does not,
and an aligned
barrier is executed by all threads of a warp together, so a warp arrives
with its 32
threads (without `.aligned` each thread arrives on its own: P0-6 counts a warp whose
lanes all reach it alike as one arrival of 32 threads, and lanes that diverge at it are
undecided). P0-6: every named barrier is reached by the warps that take part in
it the same number of times, with the same id and count, every condition deciding that (a
branch around it, a loop exit of a loop holding it, its predicate, id or count) uniform
over them. Violations:

- **F1** a barrier reached under a condition that differs between its participants
  (warps, or the lanes of a warp). E.g. a TLX partition barrier (`bar.sync 3, 128`
  before an async MMA in a 4-warp partition) inside `if (tid & 127) < 32`: one warp
  arrives, the instance never completes.
- **F2** a barrier inside a loop whose trip count may differ between its participants,
  e.g. a spin loop re-reading a strong load: the warps leave it in different iterations
  and pair instances across iterations.
- **F3** a count no participants meet: not a positive multiple of 32, above the CTA's
  threads, or other than 32 x the warps that meet; or ids or counts that differ between
  the warps that meet.

**Pipeline.** The P0-6 KIR (`<spec>.nb.json`, `<kernel>__<digest>.nb.json`; kir.md
change 14) is made from the variant's PTX by the barrier pass of the PTX frontend
(`frontend/ptx/nbar.py`, frontend_ptx.md "The P0-6 barrier pass"), for every variant of
both families, whatever the Layer-1 translation (`--frontend legacy` too) and also where
the P0 translation refuses the variant (e.g. the global-memory synchronization of a
Triton kernel across CTAs), but not for a cluster geometry LKV does not check (below).
Its roles are the warp-specialize roles (TLX/Triton with a
warp-specialize dispatch: `default`, `partition<k>`, padding states' roles included,
in a dispatch that may run no round each worker's region under the default's decision,
`frontend_ptx.md`; CuTe and Triton without one: one role of the whole block), split into
warp ranges wherever a deciding condition is uniform within each warp but not over the
role, so every event is executed by every warp of its role:

- `named_bar` (id, count, sync): a barrier instruction with constant id and count, under
  conditions uniform over its role (exact KIR `if`/`for`/`while` where the frontend has
  exact values); `all` when it names no thread count (`count` is the block's threads);
- `nb_open` (kind, certain, why): one the pass cannot count, reached under a condition
  its participants may not share: `lane` (thread-index values that differ between the
  lanes of a warp), `cond` (a branch or predicate on data that may differ), `loop` (a
  loop whose exit may differ), `operand` (an id or count that is not a constant, or a
  count no warps meet), `dispatch` (barrier 1, the warp-specialize dispatch's fork
  barrier, used otherwise). `certain` when the model admits participants that diverge:
  thread-index constants that differ (an elect.sync leader), or nondet data (a strong
  load, an atomic's result, %clock: warps may read different values, and a spin loop's
  warps leave it in different iterations), for every divergent condition around it, and
  it lies in no repeated region (which may run no pass); lanes of a warp that diverge at
  a barrier without `.aligned` are never certain;
- `nb_rep` (begin/end, key): a region repeated a number of times the model does not fix,
  the same in every warp of the role (a loop, or an arm of a branch, on uniform data the
  KIR does not carry); its key (`<role>@<PTX line>`) is non-empty when the condition is
  uniform over the role before warp splitting too. A role split from one role translates
  only its own warps' path (each branch or predicate whose condition is one constant
  over them takes one arm; some values are held as their instances), so each keyed
  region also has a path signature: the decisions on the paths of the values its
  condition is computed from (data and control dependence, a loop's exits for the
  registers it writes; a value held as the role's own instance is a decision no other
  role shares). Where the signatures of one key differ between the roles split from one
  role, the key (of the begin event; an end event's key is not read) gets `#<n>`, and
  regions whose conditions came by different paths never pair.

**Decision** (per input and CTA; `KV.runNb`, theorem `KV.Holds .namedBarriers`):

- a reached `nb_open` marked certain is a certain violation (F1/F2), one not marked
  certain leaves P0-6 UNDECIDED, and so does one inside a repeated region (it may run no
  pass);
- a count that is not a positive multiple of 32 or exceeds the CTA's threads is a
  certain violation (F3; the frontend reports those as certain `nb_open`s, and
  `frontend/kir.py` rejects a `named_bar` whose count is not a positive multiple of 32, so
  on a valid KIR only the second check can fire; the decision keeps both), UNDECIDED
  inside a repeated region;
- the pairing: one operation of each role in turn (lockstep); an instance of barrier b
  opens with the count of its first operation, collects the arriving roles' threads and
  completes at its count, releasing the roles that wait on it; an instance whose
  operations all name no thread count also completes once every role outside it has ended
  (its threads exited). The warps of a role arrive independently, so an instance that
  roles complete (also a role alone at its count) while a role outside it still uses the
  barrier could have been completed by other warps: P0-6 is UNDECIDED there. While only
  `sync` operations have run, each warp blocks in each one, so every instance holds
  exactly one operation of each warp that completes it; with that stop the pairing is
  forced: a count mismatch (F3) or roles that wait forever (a count above the threads
  that reach it, a different number of syncs per warp, a cycle between two barriers)
  found before any `arrive` ran is a certain violation (an `arrive` the CTA would reach
  only later changes nothing: `tests/lean/nb` `short_arrive_after`), unless a blocked
  role waits inside a repeated region or on an instance whose operations only partly name
  no thread count and whose other roles all ended (UNDECIDED).
  Producer `arrive`s pair with consumer `sync`s across roles (CuTe/CUTLASS warp-group
  barriers). A role that arrives again on an open instance it already arrived on (an
  `arrive`, then another operation on the same barrier) joins that instance when the
  other roles cannot complete it without it: their threads that may still arrive before
  they block on it (each role up to its first `sync` on the barrier; none for a role that
  waits on it) do not reach the count, so its arrival lands there whatever the timing
  (e.g. a warp that arrives and then syncs on one 160-thread instance with three other
  warps, `tests/lean/nb` `arrive_double`). Otherwise it is held until the instance
  completes and arrives on the next one: the **warp-ahead reading**, an assumption
  (below). The summary line counts the arrivals the reading holds or may hold: every
  `arrive` after which its role uses the barrier again while the other roles could
  complete the instance without it (the lockstep order may complete the instance before
  that next operation and so hide the reliance). A pairing that may still depend
  on timing leaves P0-6 UNDECIDED: more arrivals than the count, an instance an arrive
  leaves incomplete when every role has ended, a role held for an instance that never
  completes, or a mismatch or wait forever after an `arrive` ran (a wait forever also
  while an arrival is held);
- an `nb_rep` region is paired once (it may run any number of passes, also none). A
  barrier that only one role uses may lie in one (repeating it repeats complete instances
  of that role alone). A barrier several roles use may lie in one only if every region
  around each use has a key (a region of the role before warp splitting whose condition
  is uniform over that whole role, so the roles split from it repeat it alike), and every
  instance must be formed by operations of one nest of regions, or all outside every
  region: then one pass pairs every pass. Otherwise P0-6 is UNDECIDED. A mismatch or a
  wait forever inside a repeated region is UNDECIDED, not FAIL (the region may run no
  pass), and so is a wait forever while any blocked role waits inside one;
- a CTA with a stopped role trace (an out-of-bounds metadata load, an `assume` that does
  not hold) is not paired, and its repeated regions are not judged (both need every
  role's whole trace): only the checks of single events that ran (a reached certain
  `nb_open`, a count) apply there (the "Undecided specs" sections).

P0-6 is the same on every input of a variant unless input-dependent control (exact KIR
`if`/loops on params, CTA ids and recorded metadata contents) decides reach. A TLX/Triton
launch with a preferred cluster geometry is checked in both geometries (`<spec>` and
`<spec>@cluster<dims>`, whatever the translation): the cluster rank can decide reach. A
CuTe launch with a preferred cluster and a TLX/Triton launch whose function's required
cluster differs from its cluster dims are refused for P0-6 too, as for P0 and P1. It
counts toward the outcome and the exit status like the other P0 specs.

**Uniformity: what a PASS rests on** (trust base; frontend_ptx.md "The P0-6 barrier
pass" has the rules). A value is uniform over a role when it is equal in all its threads:
an exact expression whose thread-index instances fold to one value over the role, or
data the pass classifies as uniform:

1. kernel parameters, constants, ctaid, cluster rank (exact);
2. a weak global load, at an address uniform over the role, of a tensor the kernel never
   writes (`translate.writes`, ignoring stores of shared memory only; a value loaded with a
   type of 32 bits or less is **assumed** to be no pointer, so an index read from memory
   does not make every tensor a possible store target; tensors are told apart by name:
   that no other tensor argument shares the storage of a written tensor is **assumed**,
   only metadata tensors are checked for it). Where the dump recorded the
   tensor's contents (a 1-D contiguous integer tensor that is no view, holds no stale
   data and shares no storage with a written tensor) the load is exact, a metadata `load`
   Lean evaluates on each input's contents, as in the P0 translation (the same trust:
   the recorded contents are what the kernel reads); then loops bounded by such data are
   exact and Lean counts their passes;
3. shared-memory data at a uniform address (a weak `ld.shared`, a generic load of a
   shared address; `ldmatrix` at a uniform address is the same in every warp, per lane),
   and the CLC response: **assumed** written before a synchronization every reader
   passed, so every reader sees the same value;
4. **published values**: the value a spin loop accepted (the loop re-reads a location
   with strong loads at an address uniform over the role until its exit test accepts
   the value; a loop LLVM rotated, `if accept(v0) skip; else spin`, counts when the If's
   test is the loop's acceptance test on a strong load of the same address): **assumed**
   published once by its producer (a write-once slot, a tagged word whose tag is written
   once), so every reader accepts the same value. Published is the register that holds
   the accepted value as loaded (or a copy of it) at every exit of the loop; a value
   computed from it inside the loop (e.g. combined with the lane index) is data that
   decides no certain divergence. The spin loop itself repeats a number of times that
   differs between readers (F2 if it holds a barrier). A producer that changes an
   accepted value (e.g. a counter compared with `>=` and then used) breaks this
   assumption;
5. a warp collective over lanes: `redux`/`vote` of data the same in every warp per lane
   (or uniform) is uniform; `shfl.sync.idx` from one lane is uniform over the warp (each
   class also records whether the value is alike in corresponding lanes of every warp of
   the role before warp splitting, so such a collective keys the split roles' regions);
6. values in a loop translated by classes: a register uniform at every entry and back
   edge is a fresh uniform symbol (its exact uses stay exact; no KIR expression reads
   it).

Not uniform: strong loads (`.volatile`, `.relaxed`, `.acquire`, `.cv`) of global or
shared memory, atomics' results, `%clock`/`%globaltimer`, `mbarrier.test_wait`/`try_wait`
results outside the blocking wait idiom, vote/shfl of values that differ between warps,
thread-index values that differ over the role, and any other data (UNDECIDED, never
PASS).

**What a PASS means.** Every reached named barrier of every CTA pairs into complete
instances under the lockstep pairing, every deciding condition uniform over its
participants under the rules above. A PASS rests on assumptions 2 (recorded metadata) to
4, and, where the summary line counts held arrivals, on the **warp-ahead reading**: no
role runs a whole instance ahead of the others on a named barrier, i.e. a role that has
arrived on an open instance does not arrive on that barrier again before the instance
completes unless no other role could complete it first. This is the analogue of P1-1's
reading of a role of several warps: P0-6 has no mbarriers, and in the kernels checked the
mbarrier waits between a producer's arrives (it waits for its consumers to release a
buffer) are what keeps it from running ahead; a kernel without such an order would pair
instances depending on timing, which P0-6 then does not see. Where an instance of a
barrier without a thread count completes because the roles outside it ended, a PASS
rests on the PTX ISA's rule that exited threads release such a barrier (and on those
roles reaching their end: P0-6 sees no other waits). A PASS says nothing about
mbarriers (P0-2, P0-4), memory ordering or which data the barriers protect.

**Evidence.** The fixtures in `tests/lean/nb` (`tests/lean/nb/expected.txt`) cover each
rule: uniform and per-input reach (one or two CTAs), the short barrier and the cycle
(FAIL), count mismatches, certain and uncertain `nb_open`s, private, shared and keyed
repeated regions, producer arrives paired with consumer syncs (`pairing`, `forced_pair`,
`arrive_double`), the warp-ahead reading (`arrive_ahead`, `held_cycle`; `pairing` counts
one arrival), a wait forever before any `arrive` ran (`short_arrive_after`: FAIL),
stopped traces (`meta_stop`, `meta_stop_two_roles`: UNDECIDED), instances a role outside
them could have joined (`mix_alone`, `mix_counts`, `mix_groups`: UNDECIDED), a wait
forever while another role waits in a repeated region, in either role order
(`wait_after_rep`, `wait_after_rep_swapped`: UNDECIDED), findings inside repeated regions
(`rep_open_certain`, `rep_bad_count`: UNDECIDED) and barriers without a thread count
(`all_exit`, `all_exit_two`: PASS; `all_counted`, `all_cycle`: FAIL; `all_mixed`:
UNDECIDED).

**Fails closed** (P0-6 UNDECIDED for the variant, `nb_not_checked`, kind `gap`): control
flow outside cfg.py/regions.py (a loop exiting to two blocks, several rets, several
dispatches; a switch `brx.idx` is lowered to its compare chain as in the P0 translation,
refused only if its index is not provably in range), a call of a device function, roles
that are not warp ranges. Irreducible flow (LLVM tail-merges the identical exits of two
inlined spin loops) is made reducible by node splitting, and a loop whose rotated test
leaves the natural loop on one path back to its header gets that path copied into the
loop; the copies are counted in `nbar.split_blocks`. An instruction without an op table
row that may be a barrier or a branch is an ERROR (any other only computes data).

## P1-1 TMEM reuse ordered (`KV.TmemReuse.Holds`)

Decided on a P1 KIR (`<kernel>__<digest>.p1.json`, written by the CuTe
frontend's second translation pass; `kir.md` change 10) per input, over the
clusters (a CTA without clusters) whose role traces are all complete.
`#lkv_check_p1 kernel input tmemReuse` prints the report (`KV.runP1 kernel input
[.tmemReuse]`); the theorem is `KV.Holds .tmemReuse kernel input`. P1-1 never changes a
P0 verdict, is reported in its own section of report.md, and does not affect
`lkv.py`'s outcome and exit status with its FAIL and UNDECIDED verdicts unless
`--p1-strict` (then a P1 translation refused by design counts as UNDECIDED); its
ERRORs always count (a failed P1 translation: an internal error of the P1 pass
or invalid P1 KIR; a Lean error; an INCONSISTENT verdict). TTGIR variants are
"not checked" (no TMEM events); a refused P1 translation is UNDECIDED with the
reason (frontend_cute.md, refusals R1-R9), counted in `totals_p1`.

**Model.** A cell is (lane, 32-bit column), lane in [0, 128), column in [0,
cols) of the CTA's single allocation (KIR `tmem.cols`; with `base = zero` the
fake-zero-base idiom, addresses are absolute and the allocation must be all 512
columns). Every tcgen05.ld/st/cp/mma is one access with exact rectangles, each
read or written (`ir_ops_cute.md`, "P1 slice"): ld/st 32x32b.xN: lanes [L, L+32)
x [C, C+N) of the issuing warp; cp 32x128b.warpx4: lanes [L, L+128) x [C, C+4)
(write); mma (M = 128, f32; cta_group::2: in each CTA of the pair, M = 256 or, with
N / 2 columns, M = 128): D lanes [0, 128) x [C, C+N) (always a write), A
from TMEM [C, C + K x bits/32) (read), and the scale factors (read, all 128
lanes, from the floored SF column): SFA 4 columns, SFB 4 columns per 128 rows of
N (8 for N = 192 or 256), the CuTe DSL's TMEM SF layouts (`ir_ops_cute.md`).

**Completion** (PTX ISA, tcgen05 memory consistency). An ld/st completes at its
role's next `tcgen05.wait::ld/st` of the matching kind (never if none follows; a
register dependency does not prevent a WAR hazard). A cp/mma completes when a
later `tcgen05.commit` of the same thread has landed; a landing is seen only
through a wait that surely returned and needed it: if a wait needs m arrivals of
role r on an mbarrier slot on which every arrival of r is a commit, the m-th
commit event of r or a later one landed, so every cp/mma of r before the m-th
commit event completed (m landed arrivals come from m distinct commit events). A
plain arrive after a commit, a slot on which r also arrives plainly, and
`tcgen05.wait::st` after a cp complete nothing. cp/mma roles are one warp (trust
item: the commit is issued by the lane the cp/mma lowering elects).

**Happens-before** uses the P1 ledger (`Run.ledgersP1`: Ledger.lean's `ledger`
with `precise`; P0-2 and P0-4 keep the default) and named barriers, over role
traces. Every mbarrier wait, arrive and commit of a role of several warps is
issued by every warp of the role. The CuTe P1 pass makes that so
(`interp.Translator._p1_split_by_issuers`): a role whose warps issue such an
event only in part (a thread-0 arrive or wait) is split into runs of issuing and
non-issuing warps, each translated again as a role of its own, and one issued
under a runtime lane condition is refused (R7). Any lane issues for its warp
(warp granularity, like the warp-collective tcgen05 ops); frontend/kir.py
rejects a P1 KIR whose arrive gives fewer arrivals than its role has warps.

The warps of a role need not be at one point, so a role's arrive event is
credited per warp (**partial credit**, `KV.creditArrivals`). If a phase needs m
arrivals of a role of several warps, the m-th one belongs to some arrive event e
of the role: some warp issued e, so the role's waits before e returned and their
phases completed. Every warp issued e only if m takes more of e's n arrivals
than all warps but one can give, `m - (arrivals before e) > n - s`, with s the
least arrivals each warp gives: 1 for an mbarrier arrive in `sound` (every warp
issues it), n / warps in `lock`, and 32 for a named-barrier arrival (every
thread arrives). Otherwise every warp issued only the role's previous arrive
event. An ld/st completes only when every warp of its role passed its wait
(`Hb.every`); the cp/mma landing rule is for one-warp roles anyway. The P1
ledger's own classification (skips, laps) credits arrivals the same way, and
counts a waiting role's own earlier arrivals as surely before its wait only if
the role is one warp. A `bar.sync` of the role before each arrival does not
change this: the phase may still complete without one warp's arrival. E.g. one
8-warp compute role arriving 8 per event on a count-8 slot, and compute roles split
into 1 + 3 warps on a count-4 slot (each needed for its exact share by
`needsWith`), are credited whole. Two closures are computed before
every access, incrementally per role (`TmemReuse.saturateP1`):

- `sound` decides PASS. Waits count at their sound phase lower bounds
  (`minPhase`: re-checks, stale waits and presumed skips at c, skips at c+2,
  laps and late joins at the moved phase). A wait that returned for phase k
  needs, of each arriving role r, `k x count` minus the other roles' arrivals
  that may precede its return (`SlotLedger.needsWith`, `mayBeforeReturn`), not
  minus their arrivals over the whole run (`needs`, which P0 uses): an arrival
  of role o may precede the return only if o issues it before `returnCuts`' cut
  of o, the first event of o that surely follows the return (o's own events
  after a wait of o that cannot return first, because the arrivals that may
  precede the return cannot complete its phase bound; the fixpoint of
  `arrivalsMayPrecede`). The P1 ledger classifies waits with this closure too,
  so it finds more laps and skips than the P0 ledger. Where several roles arrive
  on one slot in a pipeline (e.g. compute warps split into roles of 1 and 3
  warps), this gives each role's exact share without assuming one.
- `lock`, the lockstep reading at its most generous, separates F1 from
  undecided. Waits count at `lockPhase` (presumed skips at c+2; at least the
  previous wait's `lockPhase`, +1 after a parity change; laps the lockstep
  credit shows), and a role's need is at least its largest fitting lockstep
  share of every completed phase up to k (`SlotLedger.lockNeeds`). Taking each
  role's largest share mixes the fitting assignments, so a pair `lock` cannot
  order is unordered under every one of them. Partial credit counts each warp's
  equal part of an arrive (the most it surely gives). A stale wait keeps its
  lower bound c (it may return at once).

Named barriers: the j-th `bar.sync` of a role on id b waits for phase j of b,
and phase k needs `k x count - sum over the other roles o of cap_o(k)` arrivals
of a role, with `cap_o(k) = min(total_o, k x threads_o)` if o only uses bar.sync
on b (a thread blocked in bar.sync arrives at most once per phase) and `total_o`
if o uses bar.arrive. That count assumes every arriving thread is a role's. A
P1 KIR with `cta_threads` above its roles' threads (a TLX/Triton kernel's idle
padding warps, which join a CTA-wide barrier) uses P1-2's rule: a named barrier
whose participants all use bar.sync with one count, their threads summing to it
or the count `cta_threads`, completes phase k with exactly the k-th bar.sync of
every participant (`NamedSlot.exact`: for a CuTe KIR, which has no
`cta_threads`, the same credit as the count above); another one orders only
under `lock`. A named barrier id used with two counts orders nothing.

**Decision.** Two accesses of different roles conflict when some pair of their
rectangles overlaps in lanes and columns and one of them writes. A conflicting
pair is ordered under a closure when one completes before the other is issued
(completion before issue: warps never pipeline each other's tcgen05 ops).

P1-1 holds (the theorem) iff every access is in range and every conflicting
cross-role pair is ordered under `sound`.

| Kind      | Condition                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FAIL F1   | a conflicting cross-role pair not ordered even under `lock`: a timing-dependent TMEM race in the model under every fitting reading of the lockstep contents                                                                                                                                                                                                                                                                                          |
| FAIL F2   | a rectangle outside lanes [0, 128) or columns [0, cols)                                                                                                                                                                                                                                                                                                                                                                                              |
| FAIL F3   | an ld/st outside lanes [32 (w mod 4), +32) of its warp w                                                                                                                                                                                                                                                                                                                                                                                             |
| undecided | a role trace stopped (an out-of-bounds metadata load, an assumption that does not hold); F1 pairs while a named barrier id is used with two counts, or in a cluster whose ledger does not converge; or (U1) cross-role pairs ordered under `lock` but not under `sound` when no certain counterexample (F1 with consistent named barriers, F2, F3) exists. First line `not decided on this input`; U1 pairs are listed as counterexamples (`ordered only under the lockstep contents of the ledger`) also when the input FAILs anyway |
| NOTE N1   | a same-role pair involving a cp/mma that conflicts, is not a PTX pipelined pair (mma A/SF read -> mma D or cp; mma D -> mma D at the same address; cp -> mma A/SF read at the same address; ld/st -> ld/st) and is not separated by a completion of the first                                                                                                                                                                                        |
| NOTE N2   | an intra-role cross-warp ld/st pair with no bar.sync of the role between the first one's completion and the second (only the lockstep reading of a role orders it)                                                                                                                                                                                                                                                                                   |
| NOTE N3   | the P1 ledger finds a blocked wait: happens-before after it is vacuous                                                                                                                                                                                                                                                                                                                                                                               |

Lines are located at the non-MMA access that writes (else the other), with the
partner, both rectangles and why neither direction is ordered; one line per
(line, partner line) and cluster (a CTA without clusters) with a count of the other
pairs there. MMA accesses are located in the DSL helper (`blackwell_helpers.py:869`);
their `site` is the closest kernel line before them.

**What a PASS means**: every pair of TMEM accesses of different roles in the
extracted model that touch a common cell, one writing, is ordered under mbarrier
and named-barrier happens-before over the role traces, with the P1 ledger's
sound phase lower bounds and needs (never its lockstep contents). Each mbarrier
wait, arrive and commit of a role is issued by all its warps (the P1 pass splits
roles so, above), and an arrive counts as every warp's only when the phase needs
more of its arrivals than all warps but one give (partial credit, above). What
stays assumed is that no warp of a role runs so far ahead that its own later
arrivals on a slot stand in for another warp's earlier ones (a phase completes
on a count, whoever arrives); it is not checked. Intra-role cross-warp TMEM
pairs are notes (N2). It does not check tcgen05 fences (the kernels checked so far have none;
the ISA's canonical cross-thread pattern needs `fence::before_thread_sync` /
`after_thread_sync`), TMEM allocation lifetime (access before the alloc or after
the dealloc), same-role hazards (notes), or shared-memory reuse. Write "P1-1
holds for case C, inputs I, at REV".

**Evidence.** The fixtures in `tests/lean/p1` model the pattern P1-1 was written for, a
block-scaled attention backward in which the MMA warp's scale-factor copy into a TMEM
region races the compute warps' load of that region: `s2t_race` (no barrier orders
them: FAIL from the second m-block on), `s2t_drain` (a drain barrier orders them:
PASS), and the persistent-handoff and split-role variants around them
(`tests/lean/p1/expected.txt`).

## P1-2 async store sources reuse ordered (`KV.StoreReuse.Holds`)

Decided on a P1 KIR whose top-level `p1` names P1-2 (`kir.md` change 15; the
TTGIR frontend writes `<spec>.p1.json` in the PTX translation, its default, for every
specialization with a bulk store, and `<spec>@cluster<dims>.p1.json` for a preferred
cluster geometry, `frontend_ptx.md` "P1-2 pass") per input, over
the clusters (a CTA without clusters) whose role traces are all complete.
`#lkv_check_p1 kernel input storeReuse` prints the report (`KV.runP1 kernel input
[.storeReuse]`); the theorem is `KV.Holds .storeReuse kernel input`. Like P1-1 it never changes a P0 verdict, is
reported in its own section, and counts toward `lkv.py`'s outcome only with
`--p1-strict` (its ERRORs always count). The legacy TTGIR translation does not check it
(its specializations are `not_checked` for P1-2), nor does the CuTe frontend, which has
no P1-2 pass (its runs report nothing for P1-2).

**Model** (PTX ISA 8.7 to 9.5, identical wording; sections "Completion Mechanisms
for Asynchronous Copy Operations", "Bulk async-group completion mechanism",
`cp.async.bulk.commit_group`, `cp.async.bulk.wait_group`). Once a bulk
asynchronous copy is initiated, modifying its source or its tensor descriptor
before it completes is undefined behavior. A bulk store S is a
`cp.async.bulk.tensor.Nd.global.shared::cta.bulk_group` or
`cp.reduce.async.bulk.tensor` (TMA store or reduce) or a non-tensor
`cp{.reduce}.async.bulk.global.shared::cta.bulk_group`, issued by one thread t
(`bulk_store`: t's warp and lane, or the elected lane of an `elect.sync`). S
joins t's open bulk async-group, which t's next `cp.async.bulk.commit_group`
closes (groups are per thread and complete in commit order). t's
`cp.async.bulk.wait_group{.read} N` returns once all but the N most recent of t's
committed groups completed; with `.read`, once their operations "completed
reading from their source locations" and the ISA's `.read` text also lists
"reading from the tensormap" (`wait_group` section). So both the source and the
tensormap of S are free at the first wait W of t after S that follows at least
N + 1 of t's commits after S (`Store.done`, `.read` or not), or never. The
frontend checks that every issuing thread of a role's bulk stores executes each
commit and wait event of the role (a wait or commit of warp 0 only covers its
elected lane, as Triton emits around a descriptor copy), so t's events are the
role's; `wait_group.sync_restrict::proxy::async` (PTX 9.5: orders only
async-proxy accesses) is refused.

**Conflicts.** A write X conflicts with S if it writes shared memory of S's CTA that
overlaps S's source bytes (F1: `smem_write`, the bytes of st.shared, stmatrix,
atom/red on shared memory, generic stores to it, the landing of a TMA, bulk or
cp.async load, and st.async, red.async or a shared::cluster bulk copy or reduce from
another CTA of the cluster), or writes the 128-byte tensormap S reads (F2:
`tmap_write`, the warp-collective `tensormap.cp_fenceproxy` into the global slot of
an in-kernel descriptor; a host descriptor lives in param space, which the kernel
cannot write; the frontend refuses a global store or atomic into a buffer that holds
the tensormaps of bulk stores). The pair is ordered if S completes before X (W
happens before X in every thread executing X) or X happens before S is issued. An
asynchronous write X (a load's landing) happens before S is issued only if the bytes
of its mbarrier show it complete (`StoreReuse.landed`): X's load t (the role's next
transaction event on the slot X names, `smem_write`'s `bar`/`idx`, in X's CTA or
with `leader` in its pair leader's, only gload footprints between) may land before
the return of a wait W on that slot that surely returned before S is issued
(`returnCuts`: positions from which every event surely follows that return), and the
loads that may land before that return bring no more bytes than the `expect_tx`
bytes phases 1..k surely hold, k W's phase bound (`bytesIn`: each role's `expect_tx`
before the last of its arrivals those phases surely hold, `k x count` minus what the
other roles can contribute, a role contributing no arrival after its own wait for
phase k or later). A phase completes only with as many bytes landed as expected (PTX
ISA, mbarrier tx-count), so every byte of those loads landed before W returned, X's
included, however the copy splits its complete-tx. Otherwise X may land after S
reads (U3).

**Happens-before** over the cluster's role traces, per role the trace prefix some
thread surely passed (`done`) and the one every thread surely passed (`all`),
saturated from the waits before a point (`StoreReuse.saturate`, the P1 ledger
`Run.ledgersP1` as P1-1's). Every mbarrier wait and named barrier of a P1-2 role is
executed by every thread of it (the frontend splits a role whose warps issue an
mbarrier event in part, and refuses a wait of part of a warp), so a thread that
passed a position passed the waits before it and they returned. A wait that returned
for phase k (sound lower bound) needs of each arriving role `k x count` minus the
other roles' arrivals that may precede its return (`needsWith`); the arrive event
holding the last needed arrival was issued by some thread: it raises that role's
`done`, never its `all` (the KIR does not say which threads issue an arrive). A
named barrier is exact if every participant uses bar.sync with one count and their
threads sum to it, or its count is the CTA's thread count (`cta_threads`: every
thread takes part, idle warps without a role included); its k-th phase is then
exactly the k-th bar.sync of every participating thread, which raises every
participant's `all`. Another named barrier orders nothing in the sound reading
(threads outside the roles may take part, so what a phase needs of a role is
unknown). The issuing thread's own program order orders W before X if X is t's own
write (the same lane, or both the elected lane of the warp: `elect.sync` elects the
same leader for the same membermask), and X before S if X is t's own or a
warp-synchronizing write of t's warp (stmatrix, the warp-collective cp_fenceproxy: t
leaves it only after every lane executed it). A `bar.warp.sync` of t's warp orders
the memory accesses of the lanes taking part (PTX ISA) as t's program order does,
but the KIR drops it, so a write of another fixed lane of t's warp that t's program
order would order is undecided (U2), not a FAIL.

The lock reading separates FAIL from undecided as P1-1's: waits at `lockPhase`,
needs of at least the largest fitting lockstep share, named barriers that are
not exact needing what the other roles' arrivals leave of the count, and every
credited arrival taken as every thread's of its role; also bytes carry order: a
wait that returned for phase k saw the bytes of phases 1..k land, so the loads
that brought them were issued before it returned, each role's loads on the slot
up to its lockstep share of those bytes (its load bytes spread evenly over the
slot's completed phases, `StoreReuse.loadsCoverLock`), taken as every thread's.
A landing issued before the store in the lock reading is undecided (U3), not a
FAIL: it may complete before the store reads.

**Decision.** P1-2 holds (the theorem) iff every conflicting pair is ordered in
the sound reading and no tensormap a store of one cluster reads is written by
another cluster.

| Kind      | Condition |
| --------- | --------- |
| FAIL F1   | a shared-memory write overlapping a store's source, unordered even in the lock reading and not undecided by U2-U4 below (no same-thread order the KIR cannot pin down might order it, it is no landing issued before the store in the lock reading, and neither event follows a blocked wait) |
| FAIL F2   | the same for a write of the tensormap a store reads; also a tensormap that a store of one cluster reads and another cluster writes (no happens-before across clusters), unless the store or the write follows a blocked wait (U4) |
| undecided | U1 a pair ordered only in the lock reading; U2 a pair the issuing thread's own program order might order, which the KIR cannot pin down (a fixed lane of the issuer's warp and the elected issuer, a warp-synchronizing write of its warp after the completing wait, or another fixed lane of a fixed-lane issuer's warp, which a `bar.warp.sync` the KIR drops may order); U3 a load's landing issued before the store (in the sound or the lock reading) that its mbarrier's bytes do not show complete before the store is issued; U4 a pair unordered even in the lock reading (a tensormap pair of two clusters too) whose store or write follows a wait the P1 ledger finds blocked in its role (in the lockstep model it never runs: the pair may never run; P0-2 reports the hang); F pairs while a named barrier id is used with two counts or in a cluster whose ledger does not converge; a stopped role trace. First line `not decided on this input` |
| NOTE N1   | a bulk store no wait of its issuing thread completes before its role ends |
| NOTE N2   | the P1 ledger finds a blocked wait: happens-before after it is vacuous |

Lines are located at the write, with the store, its source bytes (and
tensormap), the issuing thread and why neither direction is ordered; one line
per (write line, store line) and cluster (a CTA without clusters) with a count of
the other pairs there.

**What a PASS means**: in the extracted model every write of shared memory that
overlaps the source of a bulk store, and every write of the tensormap it reads, is
issued after a `cp.async.bulk.wait_group{.read}` of the store's issuing thread
completed its group and that wait happens before the write (mbarrier and exact
named-barrier happens-before, the issuing thread's program order), or happens before
the store is issued. Trust items: the `.read` reading above (a reading of the ISA
text, `wait_group` section: if a `.read` wait did not cover the tensormap read, a
fix that waits with `.read` before a descriptor rebuild would PASS wrongly);
`elect.sync` electing the same lane for the same membermask (ISA `elect.sync`); the
frontend's write set (`frontend_ptx.md`, "P1-2 pass": every instruction that writes
shared memory is an event, refused, or provably disjoint from every store source,
and every global write of a buffer that holds a store's tensormap is a
`tensormap.cp_fenceproxy` or refused); the named-barrier participants being the KIR
roles' threads, idle warps only in a CTA-wide barrier. Not checked: the source of a
`cp{.reduce}.async.bulk.shared::cluster.shared` copy or reduce into another CTA
(completed through the destination's mbarrier, not a bulk group), the tensormap a
TMA load through an in-kernel descriptor reads (a load completes through its
mbarrier, not a bulk group), multicast loads into a store source (refused), the
data-ready (RAW) side of an asynchronous write. Write "P1-2 holds for case C, inputs
I, at REV".

**Evidence.** The fixtures in `tests/lean/p1_2` model the patterns P1-2 was written
for: `staging` (an epilogue writing a staging buffer and storing it with TMA, single
and double buffered, with and without the `.read` wait and the named barrier;
`staging_wide`: its CTA-wide barrier without `cta_threads`, undecided),
`tmap_rebuild` (a descriptor rebuilt in its scratch slot per work item: unfixed FAIL
F2, fixed with `wait_group 0` + bar.sync PASS, a shared slot across CTAs FAIL),
`handoff` (cross-role through mbarriers: undecided; through a named barrier: PASS),
`same_thread` (the issuing thread's program order; writes of the other lanes of a
fixed-lane issuer's warp: undecided, U2), `landing` (a TMA load landing in a store
source: complete before the store by its mbarrier's bytes PASS; a store before the
wait, a second load without its own `expect_tx`, or a landing without its slot:
undecided, U3), `pair_chain` (the same in a 2-CTA pair whose loads complete on the
pair leader's mbarrier: PASS with the `leader` slot, undecided without a slot, FAIL
F1 for a store before the last wait), `blocked` (a write after a wait nothing
arrives on: undecided, U4; `blocked_cross`: a tensormap write in another CTA after
such a wait, likewise), `cluster_remote` (an st.async into the other CTA's store
source: FAIL F1) and `assume_stop` (an `assume` that does not hold: undecided)
(`tests/lean/p1_2/expected.txt`). The end-to-end TLX cases
`tests/e2e/cases/store_reuse_{staging,rebuild}_tlx.json` run the same races through
the real frontend.

## P0-7 global waits reached, co-residency (`KV.Global.Holds`, kir.md change 18)

Decided on a P0 KIR with sync tensors only (the PTX core's policy `gsync`, frontend_ptx.md
"Global synchronization"), one theorem per input after P0-1..P0-5, from the same evaluation
of the input (`#lkv_check ... gwait`). The run is one launch of every CTA of the grid; every
sync word starts at 0.

Evaluation (lean/KV/Eval.lean). The roles of a CTA run in order, the roles that put relay
slots first; a relay slot is written once per CTA (trusted, as the ring's handshake delivers
it), so the order only decides that every `rget` sees the write: a read of a slot no role
wrote is a limit of the model where its value is used. A take (`gwait`) reads the value half
of the store of its tag, which another CTA may make: the launch is evaluated in passes
(`evalSyncPasses`), each from the stores of the previous one (none in the first); a take
whose store is not known stops its role, blocked, and so does a role that uses a relay slot
its blocked putter never writes. A claim (`gclaim`) reads its turn among the CTAs that
claimed on the word in the previous pass: the k-th claim of the j-th of n claiming CTAs (in
linear order) reads `(k x n + j) x value`, the representative schedule in which the
claiming CTAs claim in turn. The passes end at a fixed point (at most 256). Then
`claimsFeasible` checks that the claims of each word read exactly `0, v, ..., (T - 1) v`
(every claiming CTA claims until its turn runs past the work), and `claimTurnsFeasible` that
the turns agree with the order the run imposes between CTAs, conservatively: where CTA s
reaches CTA t (t takes a tag s stores, through any chain; the CTAs of a cluster or of an
input's `sync_groups`, e.g. a protocol region's teams, reach each other), s claims on the word
once, in an earlier turn; else the input is a limit of the model. Every spec of a KIR with
claims may depend on which work a CTA claims, so `scripts/lkv.py` (`SCHEDULE_SPECS`) reports
each PASS on it UNDECIDED (`PASS on one feasible schedule only`) and keeps a FAIL, which that
schedule witnesses. Trusted: a tagged word's store of a tag is the one store of that tag, read by
every take of the tag (two stores of one tag with different values are a limit); a take
of tag 0 reads the start value.

Statement (lean/KV/Spec/Global.lean). Optimistic progress: every role instance (CTA, role)
runs as far as its waits allow when every store and arrival takes effect when its event is
reached (an mbarrier wait passes once the arrivals reached on its slot cover its phase,
the cluster ledger's `assumedPhase`; a named barrier's j-th sync once `j x count`
arrivals are reached; a take once a store of its tag is), to the fixed point where no
instance can move. P0-7 holds iff no take is
blocked when no instance can move, and a take that needs a store of another CTA (outside
its cluster) runs in a grid that fits the input's `capacity` (`ctas`, and `clusters` for a
cluster launch): CUDA guarantees no dispatch order, so waiting CTAs could hold every slot
while a CTA they wait for is never scheduled.
FAIL: a take blocked forever (`global wait never returns (hang)`: no store of the launch
carries its tag, or the stores that do are never reached), a cross-CTA take in a grid over
the capacity. A run in which some roles are blocked and every stopped role is (`Run.hangs`)
leaves P0-1..P0-5 undecided (`role trace(s) are blocked forever at a take of a tagged
word`). UNDECIDED: a take not reached because an mbarrier or named-barrier wait blocks first
(P0-2), a cross-CTA take without a `capacity`, and every verdict on a run whose cluster barrier
ledger does not converge (the mbarrier waits' progress rests on it). Not checked: timeouts (a trap-on-timeout
path is taken to be never taken), fairness between resident CTAs, writers outside the launch.

Protocol outcomes (frontend_ptx.md, "Protocol regions"): an input `<input>@proto:<tensor>:<v>`
is the launch on one reachable outcome vector v of the protocol's teams; the frontend explores
every interleaving of the teams' accesses of the protocol word, so the inputs over all vectors
cover every execution of the launch (a verdict of the launch is the worst over its inputs), and
a FAIL on one of them is a certain violation for the executions with that outcome.

## Undecided specs: out-of-bounds metadata loads

A role whose metadata `load` index lies outside the tensor stops at that load:
the value the GPU reads there is unknown, and later control flow and addresses
depend on it, so no value can stand in for it soundly. P0-1 FAILs with the load
(loc, tensor, index, extents). P0-2..P0-6, P1-1 and P1-2 hold only on inputs
where every role trace is complete, so their theorems are refuted; their reports
start with a `not decided on this input` line at the load and then list only the
counterexamples that are certain anyway: P0-2/P0-4 on the clusters (CTAs without
clusters) whose role traces are all complete, P0-3 double writes and stray writes
(no holes: a stopped role may still have written them), P0-5 on the evaluated
events, P0-6 the pairing of the CTAs whose role traces are all complete and the
single-event checks on the evaluated events (a role waiting on a barrier that a
stopped role would reach is not a counterexample), P1-1 and P1-2 on the clusters
whose role traces are all complete. A spec with such a counterexample FAILs; one
without is UNDECIDED (`UNDECIDED <input> <spec>`). P0-1 is decided on the P0 KIR
only: in a P0-6 or P1 KIR the load shows only as that line.
Other CTAs and roles are evaluated as usual (roles share no values).

## Undecided specs: assumptions that do not hold

A KIR `assume` (kir.md change 16) is a condition a frontend relies on but could not
prove statically (the PTX translation: a write whose offsets have no static bounds
misses a warp-specialize capture). Where it is false on an input, that is no
certain violation of the kernel, only of LKV's model: the role stops there as at an
out-of-bounds metadata load, without a P0-1 finding. Every spec, P0-1 included, is
then refuted on that input, reports a `not decided on this input` line at the
`assume` naming it (`... role trace(s) stop at a model assumption that does not hold
(<msg>) ...`; the summary line names it too, "P0-1..P0-5 are not decided") and lists
only what is certain anyway. Nothing of the cluster (the CTA without clusters) of the
stopped trace is: its other roles may have used a value the broken assumption made
wrong (a capture the write overwrote before they read it), so P0-1, P0-3 and P0-5 list
the counterexamples of the other clusters (`Run.certainPart`), P0-2/P0-4 those of the
clusters whose role traces completed, and P1-2 and P1-1 decide no pair of a cluster
with a stopped trace: UNDECIDED unless such a counterexample exists. Where it holds it
changes nothing (`tests/lean/assume`, `assume_reader`, `p1/assume_p1`,
`p1_2/assume_stop`). A device `assert` stays a certain error (every spec FAILs).

## Undecided specs: a barrier ledger that does not converge

The phase classification and the lockstep contents of a cluster's ledger (one
CTA without clusters) justify each other only at a fixpoint: a round that
changes neither the phase bounds, any wait's classification nor the contents
(`KV.ledger`). Presumed skips and contents, or the contents of two slots through
each other's evidence, can instead alternate forever. A cluster that reaches no
fixpoint in (its waits + 2) rounds, or whose round brings back the state of two
rounds before (a 2-cycle, stopped at once), is not checked
(`SlotLedger.converged`): P0-2 and P0-4 list one `not decided on this input`
line for it (no IR location) and none of its slot lines or notes. Their certain
counterexamples on other clusters still FAIL the spec; without one it is
UNDECIDED (`tests/lean/ledger_cycle`: before, its P0-2 verdict flipped with the
parity of an unrelated role's trip count). The P1 specs' precise ledger
(`Run.ledgersP1`) runs the same loop: on such a cluster their F pairs (P1-1 F1, P1-2
F1/F2 within the cluster) are not certain (they rest on one round's lockstep contents)
and follow a `not decided` line; their PASS, which uses sound happens-before only, is
unaffected.

## Evaluation errors

`KV.eval` stops with an error and every spec's theorem is refuted on that input;
no counterexample is listed. The error's kind (`Eval.lean` `ErrKind`, printed as
`evaluation error (<kind>): ...`) says what that means:

- `kernel error`: the kernel or its host certainly fails on the input, so every
  spec FAILs: a failed device `assert`, a barrier index out of range, a `for`
  that starts below its upper bound with a step that is not positive (it never
  ends; a `for` with no trips runs nothing whatever its step), a negative grid
  dimension from the grid Exprs, a grid the cluster dims do not divide (the launch
  fails), an event naming a CTA outside the cluster (a `cta` rank, or a `cta_mask`
  bit, at or above the cluster size).
- `model limit`: the model cannot evaluate the input, so no spec is decided and
  it is not an error (UNDECIDED): division or remainder by zero (in a tensor
  expression on any lane, including lanes a mask disables: tensor expressions
  evaluate every lane; the GPU computes an unspecified value, which the kernel
  may never use); a handoff through shared memory the run does not show (kir.md
  change 17: no value recorded at the arrive a get's wait pairs with, putters that
  recorded different values, waits that do not pair with the putters' arrives, or a
  putter's next put not ordered after the get).
- `unexpected`: the case or LKV must be fixed, so no spec is decided and it is
  an ERROR: a `while` exceeding `max_iters` (a hang of the kernel, or a budget
  too small for the input), an in-extent metadata `load` at a negative offset
  (negative strides) or at one the input gives no value for, a negative barrier
  `num`, a negative `expect_tx` or `complete_tx` byte count (PTX transaction
  counts are unsigned), a segmented global tensor whose strides give no unique row per offset
  (shared notions), and IR and input inconsistencies that `frontend/kir.py`
  rejects (width mismatch, unbound variable, unknown tensor/barrier, a param
  without a value).

## Reports

`#eval KV.runCase kernel input` prints a summary line (grid, CTAs, roles,
events; role traces stopped at an out-of-bounds metadata load or an assumption
that does not hold, and why; the elision flags
the input sets), then, per spec, PASS/FAIL from the decision procedure,
counterexample lines `loc: context: message` (on an undecided spec, first the
`not decided on this input` line) and note lines `note: loc: context: message`:
the first one at every distinct loc, then others up to 12 lines in all (each
list followed by `... N more` if longer; `note: ... N more` for notes). Then
come `VERDICT <input> <P0-n> PASS|FAIL` lines, for specs with notes
`NOTES <input> <P0-n> <count>` lines, and for refuted specs without a certain
counterexample `UNDECIDED <input> <P0-n>` lines. The counterexample finders
(`<Spec>.issues`) evaluate the same per-object conditions as the Props; if a
finder and the decision procedure ever disagree the verdict is printed as
`INCONSISTENT`. `#lkv_check` prints the report of an input, then, once the input's
theorems are in the environment, one `THEOREM <input> <label> PASS|FAIL` line per
spec (`KV.Holds` proved, or its negation). A P0-6 KIR's file runs
`#lkv_check_nb` instead (summary: named barrier operations, `nb_open` events and
repeated regions reached, the arrivals the warp-ahead reading holds or may hold), a
P1 KIR's `#lkv_check_p1` (a summary part per P1 spec it names); their lines have the
same form, with the labels `P0-6`, `P1-1`, `P1-2`.

`scripts/lkv.py` turns these into one outcome per (variant, input, spec), taking the
verdict from the THEOREM line (a variant gets verdicts only if Lean reports no error,
exits 0 and every (input, spec) has its THEOREM line; the report's `VERDICT` line must
agree, else INCONSISTENT): PASS;
FAIL (negation proved, no `UNDECIDED` line: a certain counterexample or a kernel error);
ERROR (an `INCONSISTENT` verdict, an `UNDECIDED` line after an `unexpected`
evaluation error, or a variant that failed to translate or check: a frontend error
such as invalid KIR, a DSL version mismatch or a case annotation error, a Lean
error or timeout); UNDECIDED (any other `UNDECIDED` line, or a variant the
frontend refuses by design). A run is FAIL (exit status 1) if any verdict
FAILs, else ERROR (2) if any is an ERROR, the pipeline failed or the frontend
made no KIR, else UNDECIDED (0) if any is undecided (also when the frontend
refuses every variant by design), else PASS (0). The P1 specs' (P1-1, P1-2) FAIL
and UNDECIDED count only with `--p1-strict`, their ERRORs always; lkv.py's own
errors exit 2. report.md opens with that result and lists the FAIL, ERROR and
UNDECIDED findings in sections of their own (P0-6's with the P0 specs', each P1
spec's in its own section).

```bash
python3 -I frontend/kir.py K.kir.json                    # contract check
python3 -I frontend/emit_lean.py K.kir.json -o K.lean     # KIR -> Lean
scripts/run_lean_check.sh K.lean                          # builds lean/KV if needed; THEOREM lines per spec, exit != 0 iff Lean fails
tests/lean/run_all.sh                                     # positive/negative controls for every spec
```

### report.json

`lkv.py run` and `check` write `<out>/report.json`; `report.md` is rendered from it. A
script that needs only pass or fail (CI) reads `result.outcome` or lkv.py's exit status;
the other keys say why.

| key | value |
|---|---|
| `result` | `{outcome, exit_status, fail, error, undecided}`: the run's outcome (`PASS`, `FAIL`, `ERROR`, `UNDECIDED`), the exit status (1 FAIL, 2 ERROR, else 0), and per class the reasons report.md's "Result" lists |
| `error` | only if the pipeline failed (e.g. the frontend crashed): the message; the run is an ERROR |
| `case`, `out`, `src`, `p1_strict` | the case name (`check`: `--name`, default `kir`), the output directory, the exported source root (null for `check` without `--src`), whether `--p1-strict` was given |
| `case_file`, `frontend`, `translation`, `target`, `rev`, `env` | `run` only: the case file, `cute` or `ttgir`, the Layer-1 translation (`legacy` or `ptx`, `--frontend`: frontend_ptx.md), the target, the analyzed revision (null if unknown), the frontend's environment record (dump.json `"env"`) |
| `loc_quality` | `run` only, and only if some KIR's source locations are best effort: `{variant: "line_only"}`, the variants the CuTe frontend translated from DSL 4.4.2 PTX (`--frontend ptx`), whose line numbers are read in the kernel's own source file (frontend_ptx.md, "Locations"); report.md's environment section names them |
| `host_rejected` | `[{input, error}]`: the inputs the host code rejected; they count toward no outcome |
| `frontend_failures` | `[{variant, kernel, inputs, kind, error}]`: the P0 variants the frontend made no KIR for, with the inputs that launched them (a failed all-preferred cluster geometry, `<spec>@cluster<dims>` or `<spec>@cluster`: every input of the specialization); `kind` is `gap` (refused by design: UNDECIDED) or `error` (ERROR; also a TTGIR variant whose P0 translation to_kir did not reach because it stopped) |
| `variants`, `variants_p1` | the P0 and the P1 KIRs Lean checked or tried to (below) |
| `variants_nb` | the P0-6 KIRs Lean checked or tried to (the frontends' variant names end in `.nb`); each also has `nb_sites`, the barrier pass's classification of every site (`{role, op, id, count, loc, ptx_line, status (counted, open), kind, certain, rep, why}`, the KIR's `nbar.sites`), for audit, and `nb_held`, the most arrivals the warp-ahead reading holds or may hold on one input (an `arrive` after which its role uses the barrier again while the others could complete the instance without it; 0: the verdicts do not rest on it; report.md lists the others in "P0-6 under the warp-ahead reading") |
| `nb_not_checked` | `[{variant, kernel, inputs, kind, error}]` (`inputs` as in `frontend_failures`): the variants without a P0-6 KIR; `kind` is `gap` (the barrier pass refuses the variant by design: UNDECIDED) or `error` (ERROR; also a TTGIR variant whose barrier pass to_kir did not reach because it stopped) |
| `p1_not_checked` | `[{variant, status, error, specs}]` (CuTe and P1-2 add `kernel`, `inputs`, as in `frontend_failures`; `specs`: the P1 specs it concerns, `["P1-1"]` if absent): the variants without a P1 KIR; `status` is `refused` (UNDECIDED; the outcome counts it only with `--p1-strict`), `error` (ERROR), `not_applicable` (no TMEM; for P1-2 no bulk store) or `not_checked` (TTGIR for P1-1; the legacy TTGIR translation for P1-2) |
| `totals`, `totals_p1` | per spec label, counts over (variant, input) runs (below); `totals` has `P0-6` (from `variants_nb` and `nb_not_checked`) |

A variant is `{variant, kir, kernel, flags, p1, nb, input_names, status, ...}` (`p1`,
`nb`: a P1 or a P0-6 KIR, which puts it in `variants_p1` or `variants_nb`; a P1 variant
also has `p1_specs`, the KIR's `p1`, else `["P1-1"]`): `variant` is the KIR file name
without `.json` (TTGIR: `.kir.json`; the frontends' P1 KIRs end in `.p1`, their P0-6
KIRs in `.nb`, and an all-preferred cluster geometry of a TTGIR variant adds
`@cluster<x>x<y>x<z>` to the spec), `flags` the variant's flag values,
`input_names` the inputs of its KIR. `status` is `checked`
(with `inputs`, `lean`, `lean_output`, `seconds`), `invalid_kir`, or `lean_error` (Lean
failed or timed out, or its output does not agree with itself); the last two carry
`error`, and every spec of each of `input_names` is an ERROR. An entry of `inputs` is
`{name, eval_error, eval_kind, specs}`: `eval_error` is null or the error that stopped
the input's evaluation, `eval_kind` its kind (`kernel`, `limit` or `unexpected`;
"Evaluation errors"). `specs` maps each label (`P0-1`..`P0-5`; `P0-6` in `variants_nb`;
in `variants_p1` the labels of the KIR's `p1_specs`, `P1-1` and/or `P1-2`)
to:

- `verdict`: `PASS` or `FAIL` as the theorem or its negation is proved, or `INCONSISTENT` if
  Lean's printed report does not say the same (a Layer 2 bug);
- `outcome`: `PASS`, `FAIL`, `ERROR` or `UNDECIDED` ("Reports" above). Read this, not
  `verdict`: a proved negation can be any of the last three;
- `counterexamples`, `notes`: the lines Lean printed (on a spec it does not decide, the
  `not decided on this input` line comes first), each `{loc, context, msg, source,
  line_text}`, where `source` and `line_text` are the exported source line of `loc`
  (null if not found); `more`, `notes_more`: how many it did not print; `note_count`:
  all notes.

`totals[label]` is `{PASS, FAIL, ERROR, UNDECIDED, INCONSISTENT, with_notes}`. The first
four partition the (variant, input) runs; each input of a variant without verdicts
counts under that variant's outcome (a frontend failure by its `kind`, a KIR that was
not checked as ERROR; in `totals_p1` a refused P1 translation as UNDECIDED and a failed
one as ERROR, with or without `--p1-strict`). The last two are sub-counts, not outcomes:
`INCONSISTENT` counts the ERROR runs whose verdict is INCONSISTENT, `with_notes` the runs
with at least one note, whatever their outcome (a PASS can have notes).

## Limits

### Scope: refused kernels and constructs, unchecked bug classes

- **Clusters: CuTe, and TLX/Triton in the PTX translation.** CuTe cluster kernels
  (2-CTA, cta_group::2), and TLX/Triton ones in the PTX translation, are checked
  with one barrier ledger per cluster (kir.md change 11): remote mbarrier arrives,
  multicast commits, cta_group::2 and multicast TMA loads, shared::cluster bulk
  copies and reduces, st.async and red.async into another CTA, the cluster barrier
  and, in TLX/Triton, CLC with clusters (each try_cancel response multicast to the
  cluster); P1-1 runs per cluster, a cta_group::2 MMA or copy accessing the TMEM
  of both CTAs of its pair (`Ev.tmem` CTA set), and so does P1-2 (a write into
  another CTA's shared memory is an `smem_write` with `cta`; a tensormap two
  clusters share is F2). What P0 does not see across CTAs: which CTA's data a copy
  carries (a copy sent to the wrong CTA keeps the byte counts), DSMEM loads and
  stores outside bulk copies and st.async/red.async (of those it sees the bytes,
  not the data), the arrival of exited threads on the cluster barrier, and an
  order between another CTA's landings and a slot's phases that the ledger's
  happens-before does not find: one that only a named barrier gives (P0-2 and P0-4
  use no named barriers, P0-6 checks only their convergence: e.g. a relay warp
  whose next copy waits on a `bar.sync` with compute warps that wait for the
  peer's phase), or one carried by transaction bytes beyond the `expect_tx` bytes
  `SlotLedger.bytesNeeded` counts (those that precede, in their roles, arrivals
  the phase needs); its landings go to their first possible phase, and P0-4 is
  UNDECIDED if they then overfill one.
  Refused: in the legacy TTGIR translation, 2-CTA kernels (`ttng.two-ctas`, remote
  mbarriers, DSMEM), multi-CTA layouts (`ttg.num-ctas` > 1), CLC with clusters,
  launches with a preferred cluster, pointers carried by loops and side-effecting
  inline asm; CLC in CuTe kernels, CLC with a multi-dimensional grid or clusters
  other than `(c, 1, 1)`; im2col TMA; in the legacy CuTe translation a loop-carried
  value that changes structure (e.g. a pointer into another tensor).
- **Preferred cluster geometry (TLX/Triton).** A launch with a preferred cluster
  (`preferred_ctas_per_cga`) is checked in two geometries: all clusters regular (the
  variant's KIRs) and all preferred (`<spec>@cluster<x>x<y>x<z>` KIRs, every input;
  frontend_ptx.md "Clusters, tcgen05, CLC"): P0-6 in every translation, P0 and P1-2 in
  the PTX translation (the legacy TTGIR translation, which checks one geometry, refuses
  the P0 KIR). A launch that
  mixes them is covered where both hold, except P0-3 (it holds for a mix only if
  both geometries write the same elements per preferred-size group of CTAs, which is
  not checked) and P1-2's tensormap check across clusters (only between clusters of
  one geometry). Refused (gap): the all-preferred P0 and P1-2 KIRs of a launch with
  CLC, the all-preferred KIRs of launches that disagree on the preferred dims or break
  a rule CUDA states for them (no regular cluster, i.e. cluster dims 1 x 1 x 1 without
  REQUIRED_CLUSTER; a preferred x or y that is no multiple of the regular one, or
  another z; preferred dims that do not divide every grid), a CuTe launch with a
  preferred cluster other than its cluster dims, and a TLX/Triton launch whose
  function's required cluster differs from its cluster dims (the last two with their
  P0-6 KIRs).
- **In-kernel TMA descriptors (PTX translation)**: a TMA load through one reads the
  descriptor its role's `tensormap.cp_fenceproxy` wrote. That no other role rewrites
  the descriptor's slot, and that its role does not rewrite it while a load through it
  is in flight, is assumed, not checked (P1-2 checks it for bulk stores;
  `frontend_ptx.md`, "TMA descriptors").
- **Values that pass through register memory are refused** (CuTe): e.g. an
  epilogue's store predicate kept in register-memory fragments (slice error).
  Also refused by the legacy TTGIR translation (`--frontend legacy`): a per-CTA
  tile count that warp-specialized partitions capture from an `scf.if` result
  (`no value rule for scf.if`) and `ttng.vote_ballot_sync`.
- **Completion/wait work-item correspondence is not modeled**: a mid-run extra
  completion that the waits absorb in order (an MMA commit for a zero-length Q
  tile the consumer skips) looks like a trailing completion, a note. Only two
  such completions on one CTA give a P0-2 hang.
- **P1 classes other than P1-1 and P1-2 are not covered**: shared- and
  global-memory buffer reuse WAR/RAW beyond bulk store sources (e.g. a TMA load
  overwriting a tile other warps still read), deadlocks that depend on timing or
  ring depth, TMEM allocation lifetime and tcgen05 fences, proxy fences, TMA
  descriptor legality, writer/reader layout agreement, numerics and scale factors
  (`bug_taxonomy.md`). P1-1 (TMEM reuse ordering) is CuTe only, P1-2 (bulk store
  sources) the PTX translation of TLX/Triton kernels only. Named barriers and CTA
  syncs order nothing in P0-2/P0-4 (a P0 KIR has none): P0-6 checks their
  convergence on the P0-6 KIR, and P1-1 and P1-2 use them in their happens-before.
- Only concrete inputs are checked. CLC is analyzed under 3 schedules. Variants
  selected only for huge inputs (i64 indexing) or by non-default environment
  switches are checked only where a case selects them. Ping-pong consumers that
  split one slot's phases fail P0-2 unless a uniform share assignment also fits
  (lockstep ledger). CuTe supports sm_100 with DSL 4.4.2 and 4.6.1.

### Spec model

- Concrete inputs only: a verdict says nothing about a shape no input has. The
  case file picks the shapes and conditions the kernel's callers produce
  (partial tiles, zero-trip loops where a window or block sparsity leaves a
  tile without work, more tiles than CTAs, every flag combination they reach;
  very unlikely inputs such as an empty Q are left out unless asked for,
  `cases.md`), and its `elision_flags` must name every flag that selects a
  mask-elided path. A flag missing there loses nothing: the violations of its
  inputs are reported under P0-1/P0-3.
- No interleavings are enumerated. Wait cycles that exist in the lockstep model
  (every arrival as early as the roles' waits allow) FAIL P0-2; deadlocks that
  depend on timing, ring depth or bytes that never land are P1. The cross-role
  reasoning (happens-before, optimistic progress) uses mbarrier waits only:
  named barriers and CTA syncs are not in a P0 KIR, which can only make it more
  conservative (e.g. an initialization arrival ordered by a CTA sync alone is
  not one). P1-1 and P1-2 add them to their own happens-before (P0-6 checks their
  convergence, not the order they give).
- Lockstep: the happens-before arguments treat each role as one thread of
  control (all its threads at the same event), and phase contents assume every
  arriving role takes part in every waited phase of a slot with a fixed share
  (or that one initialization arrival forms phase 1). That assumption is not a
  certainty: a slot whose real phases are not uniform can still fit uniform
  shares (`nonuniform_contents`). It therefore never raises a phase lower bound:
  the lockstep credit only turns a stale repeated parity into a presumed skip,
  judged under the contents (a hang under them still FAILs,
  `zero_share_credit`), so where the assumption is wrong the cost is a missed
  stale FAIL, never a false hang. Roles that split the phases of one slot
  between them (ping-pong consumers) are outside the model: they fail P0-2 (no
  lockstep contents) unless some uniform assignment also fits, in which case
  their phases are judged under it. Early completion by a role that runs ahead
  of another inside a phase is not detected. P1-1 has every warp of a role issue
  its mbarrier events (the CuTe P1 pass splits roles so) and credits a role's
  arrivals per warp (partial credit), but does not detect a warp of a role
  running so far ahead of another that its arrivals on a slot stand in for the
  other's (section "P1-1", "What a PASS means"). Bytes are attributed under
  every fitting share assignment, at most 64 (distinct on the roles with
  transaction events): beyond that many, P0-4 could fail on bytes some untried
  assignment explains.
- Parity carries no intent, so a wait that observes a later phase than the one
  after its previous wait (a lap) is a note: the ledger synchronizes it with the
  phase it surely observes. A mid-run extra completion that the waits absorb in
  order is invisible: e.g. an MMA role that commits a dq_fulls barrier once for
  a zero-length Q tile that the consumer skips, so
  completion j carries the previous tile's dQ while the consumer's j-th wait
  belongs to its j-th tile. Every wait still observes the completion with its
  own index, and completion j+1 always follows wait j: the MMA issues it only
  after waiting on dq_empties for the consumer's j-th release, which follows
  wait j. The ledger cannot even see which completion is extra (by counts, the
  last one looks like a trailing producer_tail completion: an R < K note).
  Catching it needs a work-item correspondence: the work item of the event that
  issues the k-th completion against that of the k-th wait. The MMA partition
  carries no `work` event (it never computes off_z/off_h), and roles may
  legitimately run on different items (prefetching producers), so that is not a
  P0 FAIL today. Only one such completion per CTA is absorbed: with two
  zero-length Q tiles on one CTA the MMA's dq_empties wait needs one release
  more than the consumer makes, a P0-2 hang.
- Happens-before treats every wait inside a prefix that surely ran as returned.
  On an input where some wait never returns (a P0-2 FAIL) the conclusions drawn
  after it are vacuous, so the other notes and counterexamples of that input
  (laps, blocked waits downstream of the hang) may be consequences of the hang.
  On the real kernels checked so far, this happens only on inputs that already
  FAIL P0-2.
- Arrivals alone decide K; a phase whose bytes never land is reported by P0-4
  (bytes of a waited phase), not as a P0-2 hang.
- A static waiter of a barrier array is expected on every completing slot with
  TMA bytes that it does not produce, so two roles that split the slots of one
  TMA barrier array between them fail P0-2. A consumer that waits on some slots
  of the array but skips others it should consume is caught only where it has no
  event on the skipped slot.
- Expect-before-load is checked within the issuing role; cross-role transaction
  patterns fail P0-4.
- Segments split one dim; rows of pointer offsets need non-overlapping strides.
  For a contiguous tensor, a column overrun into the next row hits an element of
  that row and is not a P0-1 violation (on an output it shows up as a P0-3
  double write).
- Overflow is detected only on `addr`-marked add/sub/mul/shl/trunc, as signed
  wrap or a changed value at the node's width (any width; the frontends mark
  only address paths), including lanes that a later mask disables, and only in
  role code: host-side Exprs (grid, extents, strides, barrier count and num)
  wrap the same way but record no events (`KV.hostInts`), so a wrap there is not
  reported and the wrapped value is used. An `addr` trunc reads its result as
  signed, so truncating a value in [2^31, 2^32) to i32 FAILs even if the result
  is zero-extended again before use. The operand a
  scalar `select` does not choose is not evaluated (its wraps and metadata loads
  are dead; a masked scalar load,
  `select(mask, load(T, [select(mask, off, 0)]), other)`, reads nothing where
  the mask is false), while a tensor `select` evaluates both operands on every
  lane.
- Elements are memory offsets: views that alias the same memory count as one
  element.
- P1-1 cost: the ledger dominates; an attention backward's P1 KIR (2 CTAs x 7
  roles, up to 68k events, 3150 TMEM accesses, 27k conflicting pairs per input)
  takes 2-3 min for 9 inputs on an x86 host, report and theorem included (its P0
  file: 8-10 min).
- Cost (native libKV.so): evaluation is cheap (about 0.6 s for 63k events with
  655k tensor-expression lanes, about 1 s to decide P0-3 over 1.3M intended
  elements); the phase ledger dominates on large role counts. A block-scaled
  attention backward kernel (2 CTAs x 11 roles, 27.7k events) takes about 105 s for the
  report and 515 s with the five theorems, each of which re-evaluates the kernel
  (and P0-2/P0-4 their ledgers). Emitted files with 20k AST nodes elaborate in
  about 6 s.
