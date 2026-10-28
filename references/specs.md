# Specs: definitions, violations, notes, proof method, limits

Layer 2 (`lean/KV`, core Lean + Std) evaluates a KIR program
(`references/kir.md`) on the concrete inputs of its case file and decides five
P0 specs per input, and on a P1 KIR (CuTe only) the P1 spec P1-1 "TMEM reuse
ordered" (section "P1-1" below). This file is the reference for what each
verdict means. The Lean definitions are the authority; each section names them.

## What a verdict covers

- One kernel (one KIR, i.e. one compiled flag set) on one concrete `Input`. No
  other shapes, no symbolic parameters.
- KIR semantics as implemented by `KV.eval` (`lean/KV/Eval.lean`): exact
  `BitVec` arithmetic at each node's width; every role of every CTA runs to
  completion independently (a role stops early only at an out-of-bounds metadata
  load, which leaves P0-2..P0-5 undecided unless they find something certain on
  the evaluated events); one event trace per (CTA, role).
- **FAIL means a violation that is certain in the extracted model**, whatever
  the data values. Findings that can be harmless depending on data or on how the
  kernel uses the result are **notes**: printed with the report, never a FAIL.
  The theorems prove exactly the FAIL-free Props, so notes do not affect them.
- **Four outcomes.** PASS: the theorem is proved. FAIL: it is refuted and the
  report lists a certain counterexample (`SpecResult.certain`), or the input's
  evaluation stopped at a certain error of the kernel ("Evaluation errors",
  kind `kernel error`). Refuted without either, a spec is neither proved nor
  violated, and Lean prints `UNDECIDED <input> <spec>`: every spec of an input
  whose evaluation stops with a `model limit` or an `unexpected` error, a spec
  that finds nothing certain on an input with an out-of-bounds metadata load
  ("Undecided specs"), and P1-1 with only U1 pairs or pairs that a named
  barrier used with two counts might order. `lkv.py` reports such a spec as
  ERROR after an `unexpected` evaluation error (the case or LKV must be fixed),
  else as UNDECIDED (a known limit, not an error); it also reports an
  `INCONSISTENT` verdict as an ERROR, and every spec of each input of a variant
  without verdicts as UNDECIDED if the frontend refuses the variant by design,
  else as an ERROR. A run's outcome and exit status ("Reports") is FAIL (1) on
  any FAIL, else ERROR (2) on any failure, else UNDECIDED (0) on anything a
  known limit leaves open, else PASS (0).
- Roles are not interleaved. P0-2 and P0-4 are stated over each role's own event
  order, per-CTA totals and the phase ledger of `lean/KV/Spec/Ledger.lean`; they
  are not a race check, and they catch only the deadlocks certain in the
  lockstep model (P1 covers the rest).
- **P0-2 and P0-4 are relative to the ledger algorithm.** Their Props quantify
  over the ledger's classification of waits (re-check, skip, presumed skip,
  stale, late join, lap), its lockstep phase contents, its optimistic progress
  pass and its attribution of transaction bytes to phases. The soundness of that
  algorithm against mbarrier semantics is argued in Ledger.lean, not proved in
  Lean: a P0-2/P0-4 theorem says "the ledger finds no violation", not "no
  interleaving violates the protocol" (references/design.md, trust base).
- Proof: `theorem _ : KV.Holds .<spec> kernel input := by native_decide`.
  `native_decide` trusts the Lean compiler and the native `libKV.so` built from
  the same sources (T1). The evaluator is total (no `partial`), so the theorem
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
| cannot complete / surely completed before | happens-before over the role traces of the cluster's CTAs (one CTA without clusters) (`KV.arrivalsMayPrecede`, `KV.saturate`, `KV.completedIn`), each role taken as one thread of control. An event is definitely after wait w if it follows w in its role, or follows a wait (of any role) that needs more arrivals than those not definitely after w. Phase k surely completed before an event if a wait on the slot for phase k or later surely returned before it, or, on a slot whose phases complete by arrivals alone (no expect_tx/tma_load bytes, no commit arrivals), k x count arrivals surely precede it: arrivals are necessary for a phase to complete, not sufficient. An event surely precedes another if it precedes it in its role, or precedes an arrival that a wait which surely returned before the other needs: phase k needs k x count minus all other roles' arrivals over the run. Waits are credited with lower bounds of their phases (re-checks, stale waits and presumed skips at c, skips at c+2), refined until they and the lockstep contents are stable, for at most (the CTA's waits + 2) rounds (`KV.ledger`): if they still change then, the last round's are used, without a note                                                                                                                                                                                    |
| lockstep credit                           | a second happens-before closure (`saturate` with `lockstep = true`) in which phase k also needs each arriving role's least share, over all fitting assignments, of the phases up to min(k, lockstep) of the lockstep contents. The contents are an assumption (one fixed share per role and phase), not a certainty, so this closure decides only whether a repeated parity is a presumed skip instead of stale, and serves as evidence for zero shares and initialization arrivals (it can only make contents exist). Phase lower bounds (and with them happens-before) never use it. A presumed skip counts at c+2 for the hang checks of P0-2 (its `assumedPhase`): if the contents are right the role needs that phase, and if not the wait is stale, so a hang found that way is a violation either way. The credit thus replaces a stale FAIL by the verdict under the contents, and never fails an input that passes without it                                                                                                                                                                                                                                                                                                                                                                                                                          |
| waited phases                             | phases up to R* = the highest phase some role's waits need on the slot                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| lockstep contents                         | over the waited complete phases (the first min(R*, K)): each arriving role contributes the same number of arrivals s_r to every one of them, ending exactly at one of its arrive events after each, and the s_r sum to the count (`KV.lockstepShares`); or, if no such shares exist, an initialization arrival forms phase 1 by itself and shares cover the phases after it (`KV.lockstepContents`). A role with arrivals takes s_r = 0 only if its first arrival surely follows the last waited phase. An initialization arrival is the first `count` arrivals of one role, reached at one of its arrive events, such that nothing else lands in phase 1: every other role's first arrival on the slot surely follows phase 1's completion, or the initialization arrivals contain no commit and surely precede every other role's first arrive event (a commit lands after its event). E.g. FA4's dQ ring: warp 4 pre-arrives once at init, the store warp makes every later release; the releases may be commits. Evidence for a zero share or an initialization arrival uses happens-before credit only from slots whose contents need none, never from the slot itself. Several share assignments can fit (count 8 as 2+2+4 or 1+1+6): the representative one has the most participating roles; the fitting ones are those whose zero shares are evidenced |
| bytes of a phase                          | an `expect_tx` belongs to the phase its role's next arrival falls in under a fitting share assignment; a `tma_load` to the phase of its role's latest `expect_tx` on the slot. A role of another CTA whose only events on the slot are transaction landings (`tma_load`, `complete_tx`; a lander: the other CTA of a cta_group::2 pair, a multicast destination, a DSMEM copy) lands the same bytes v in every waited phase in whole landings: a landing belongs to phase `bytes before it / v + 1` (at most K + 1), for each byte share v its landings reach exactly at v, 2v, ... (`RoleSlot.byteShares`); with no fitting share its bytes cannot be attributed. Phase K + 1 is the last one: it never completes, so bytes issued after phase K land in it (with K = 0 every byte belongs to phase 1). Bytes of a role with share 0 cannot be attributed. The bytes are attributed under the representative assignment and, if that breaks a transaction clause of P0-4, under the other fitting ones (distinct on the roles with transaction events, at most 64 in all, `KV.maxShareVectors`) with each landers' byte share choice (at most 256 combinations) until one keeps them all; P0-4 shows the first combination's attribution if none does                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| optimistic progress                       | every role runs as far as its waits allow when every arrival takes effect as soon as its event is reached and transaction bytes are ignored; a wait passes once the arrivals on its slot reach its phase lower bound x count, presumed skips counted at c+2 (`KV.deadlocks`, `assumedPhase`). A wait still blocked when nothing can move never returns in the lockstep model (or a presumed skip it depends on is stale)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| static waiter                             | a role whose code (any branch or loop body) waits on the barrier array. It is expected to observe every slot of the array on which it has no wait, arrival or transaction event (a `bar_init` does not count: TTGIR emits every one in the `default` role); a role with an arrival or transaction bytes on a slot is one of the slot's producers (one array can hold several logical barriers: the CuTe frontend emits each pipeline's full and empty barriers as one array)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| complete trace                            | a role's event trace that ran to its end; a role whose metadata `load` index lies outside the tensor stops there (its later events depend on the value read). P0-2 and P0-4 use the clusters (CTAs without clusters) whose role traces are all complete                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
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

Per CTA (with complete role traces) and slot touched by any event
(`TripCount.SlotOk`), with C the declared count and R the phase lower bound of a
waiting role's last wait, with presumed skips counted at the phase they skip to
(`RoleSlot.needed`). FAIL:

1. a malformed barrier: C <= 0, or a `bar_init` of the slot with another count;
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
  cluster, shared notions) issues no `expect_tx`: its bytes need only a byte
  share.

The last three clauses (`SlotLedger.TxOk`) are judged on the bytes of a phase
(shared notions): they fail only if they fail under every fitting share
assignment tried, because the lockstep model admits each of them.

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

## P1-1 TMEM reuse ordered (`KV.TmemReuse.Holds`)

Decided on a P1 KIR (`<kernel>__<digest>.p1.json`, written by the CuTe
frontend's second translation pass; `kir.md` change 10) per input, over the CTAs
whose role traces are all complete. `#eval KV.runP1 kernel input` prints the
report; the theorem is `KV.Holds .tmemReuse kernel input`. P1-1 never changes a
P0 verdict, is reported in its own section of report.md, and does not affect
`lkv.py`'s outcome and exit status with its FAIL and UNDECIDED verdicts unless
`--p1-strict` (then a P1 translation refused by design counts as UNDECIDED); its
ERRORs always count (a failed P1 translation: an internal error of the P1 pass
or invalid P1 KIR; a Lean error; an INCONSISTENT verdict). TTGIR variants are
"not checked" (no TMEM events); a refused P1 translation is "not checked" with
the reason (frontend_cute.md, refusals R1-R9).

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
if o uses bar.arrive. With every participant using bar.sync and their threads
summing to the count this is exact; a named barrier id used with two counts
orders nothing.

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
| undecided | a role trace stopped at an out-of-bounds metadata load; F1 pairs while a named barrier id is used with two counts; or (U1) cross-role pairs ordered under `lock` but not under `sound` when no certain counterexample (F1 with consistent named barriers, F2, F3) exists. First line `not decided on this input`; U1 pairs are listed as counterexamples (`ordered only under the lockstep contents of the ledger`) also when the input FAILs anyway |
| NOTE N1   | a same-role pair involving a cp/mma that conflicts, is not a PTX pipelined pair (mma A/SF read -> mma D or cp; mma D -> mma D at the same address; cp -> mma A/SF read at the same address; ld/st -> ld/st) and is not separated by a completion of the first                                                                                                                                                                                        |
| NOTE N2   | an intra-role cross-warp ld/st pair with no bar.sync of the role between the first one's completion and the second (only the lockstep reading of a role orders it)                                                                                                                                                                                                                                                                                   |
| NOTE N3   | the P1 ledger finds a blocked wait: happens-before after it is vacuous                                                                                                                                                                                                                                                                                                                                                                               |

Lines are located at the non-MMA access that writes (else the other), with the
partner, both rectangles and why neither direction is ordered; one line per
(line, partner line) and CTA with a count of the other pairs there. MMA accesses
are located in the DSL helper (`blackwell_helpers.py:869`); their `site` is the
closest kernel line before them.

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

## Undecided specs: out-of-bounds metadata loads

A role whose metadata `load` index lies outside the tensor stops at that load:
the value the GPU reads there is unknown, and later control flow and addresses
depend on it, so no value can stand in for it soundly. P0-1 FAILs with the load
(loc, tensor, index, extents). P0-2..P0-5 hold only on inputs where every role
trace is complete, so their theorems are refuted; their reports start with a
`not decided on this input` line at the load and then list only the
counterexamples that are certain anyway: P0-2/P0-4 on the CTAs whose role traces
are all complete, P0-3 double writes and stray writes (no holes: a stopped role
may still have written them), P0-5 on the evaluated events. A spec with such a
counterexample FAILs; one without is UNDECIDED (`UNDECIDED <input> <P0-n>`).
Other CTAs and roles are evaluated as usual (roles share no values).

## Evaluation errors

`KV.eval` stops with an error and every spec's theorem is refuted on that input;
no counterexample is listed. The error's kind (`Eval.lean` `ErrKind`, printed as
`evaluation error (<kind>): ...`) says what that means:

- `kernel error`: the kernel or its host certainly fails on the input, so every
  spec FAILs: a failed device `assert`, a barrier index out of range, a `for`
  that starts below its upper bound with a step that is not positive (it never
  ends; a `for` with no trips runs nothing whatever its step), a negative grid
  dimension from the grid Exprs.
- `model limit`: the model cannot evaluate the input, so no spec is decided and
  it is not an error (UNDECIDED): division or remainder by zero (in a tensor
  expression on any lane, including lanes a mask disables: tensor expressions
  evaluate every lane; the GPU computes an unspecified value, which the kernel
  may never use).
- `unexpected`: the case or LKV must be fixed, so no spec is decided and it is
  an ERROR: a `while` exceeding `max_iters` (a hang of the kernel, or a budget
  too small for the input), an in-extent metadata `load` at a negative offset
  (negative strides) or at one the input gives no value for, a negative barrier
  `num`, a segmented global tensor whose strides give no unique row per offset
  (shared notions), and IR and input inconsistencies that `frontend/kir.py`
  rejects (width mismatch, unbound variable, unknown tensor/barrier, a param
  without a value).

## Reports

`#eval KV.runCase kernel input` prints a summary line (grid, CTAs, roles,
events; role traces stopped at an out-of-bounds metadata load; the elision flags
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
`INCONSISTENT`. `frontend/emit_lean.py` puts every report before the theorems,
so a failing theorem's counterexample is printed first.

`scripts/lkv.py` turns these into one outcome per (variant, input, spec): PASS;
FAIL (refuted, no `UNDECIDED` line: a certain counterexample or a kernel error);
ERROR (an `INCONSISTENT` verdict, an `UNDECIDED` line after an `unexpected`
evaluation error, or a variant that failed to translate or check: a frontend error
such as invalid KIR, a DSL version mismatch or a case annotation error, a Lean
error or timeout); UNDECIDED (any other `UNDECIDED` line, or a variant the
frontend refuses by design). A run is FAIL (exit status 1) if any verdict
FAILs, else ERROR (2) if any is an ERROR, the pipeline failed or the frontend
made no KIR, else UNDECIDED (0) if any is undecided (also when the frontend
refuses every variant by design), else PASS (0). P1-1's FAIL and UNDECIDED count
only with `--p1-strict`, its ERRORs always; lkv.py's own errors exit 2.
report.md opens with that result and lists the FAIL, ERROR and UNDECIDED
findings in sections of their own.

```bash
python3 -I frontend/kir.py K.kir.json                    # contract check
python3 -I frontend/emit_lean.py K.kir.json -o K.lean     # KIR -> Lean
scripts/run_lean_check.sh K.lean                          # builds lean/KV if needed; exit != 0 iff a spec fails
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
| `case_file`, `frontend`, `target`, `rev`, `env` | `run` only: the case file, `cute` or `ttgir`, the target, the analyzed revision (null if unknown), the frontend's environment record (dump.json `"env"`) |
| `host_rejected` | `[{input, error}]`: the inputs the host code rejected; they count toward no outcome |
| `frontend_failures` | `[{variant, kernel, inputs, kind, error}]`: the P0 variants the frontend made no KIR for, with the inputs that launched them; `kind` is `gap` (refused by design: UNDECIDED) or `error` (ERROR) |
| `variants`, `variants_p1` | the P0 and the P1 KIRs Lean checked or tried to (below) |
| `p1_not_checked` | `[{variant, status, error}]` (CuTe adds `kernel`, `inputs`): the variants without a P1 KIR; `status` is `refused` (UNDECIDED; the outcome counts it only with `--p1-strict`), `error` (ERROR), `not_applicable` (no TMEM) or `not_checked` (TTGIR) |
| `totals`, `totals_p1` | per spec label, counts over (variant, input) runs (below) |

A variant is `{variant, kir, kernel, flags, p1, input_names, status, ...}`: `variant` is
the KIR file name without `.json` (TTGIR: `.kir.json`; P1 KIRs end in `.p1`), `flags`
the variant's flag values, `input_names` the inputs of its KIR. `status` is `checked`
(with `inputs`, `lean`, `lean_output`, `seconds`), `invalid_kir`, or `lean_error` (Lean
failed or timed out, or its output does not agree with itself); the last two carry
`error`, and every spec of each of `input_names` is an ERROR. An entry of `inputs` is
`{name, eval_error, eval_kind, specs}`: `eval_error` is null or the error that stopped
the input's evaluation, `eval_kind` its kind (`kernel`, `limit` or `unexpected`;
"Evaluation errors"). `specs` maps each label (`P0-1`..`P0-5`; `P1-1` in `variants_p1`)
to:

- `verdict`: `PASS` or `FAIL` as the theorem is proved or refuted, or `INCONSISTENT` if
  Lean's printed report does not say the same (a Layer 2 bug);
- `outcome`: `PASS`, `FAIL`, `ERROR` or `UNDECIDED` ("Reports" above). Read this, not
  `verdict`: a refuted theorem can be any of the last three;
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

- **Clusters: CuTe only.** CuTe cluster kernels (2-CTA, cta_group::2) are
  checked with one barrier ledger per cluster (kir.md change 11): remote
  mbarrier arrives, multicast commits, cta_group::2 and multicast TMA loads,
  shared::cluster bulk copies (inline PTX) and the cluster barrier; P1-1 runs
  per cluster, a cta_group::2 MMA or copy accessing the TMEM of both CTAs of its
  pair (`Ev.tmem` CTA set). What P0 does not
  see across CTAs: which CTA's data a copy carries (a copy sent to the wrong CTA
  keeps the byte counts), DSMEM loads and stores outside bulk copies, and the
  arrival of exited threads on the cluster barrier. Refused: 2-CTA TTGIR kernels
  (`ttng.two-ctas`, remote mbarriers, DSMEM), TTGIR multi-CTA layouts
  (`ttg.num-ctas` > 1), CLC, im2col TMA, pointers carried by loops (TTGIR; CuTe:
  a loop-carried value that changes structure, e.g. a pointer into another
  tensor) and side-effecting inline asm.
- **Values that pass through register memory are refused** (CuTe): e.g. an
  epilogue's store predicate kept in register-memory fragments (slice error).
  Also refused (TTGIR): a per-CTA tile count that warp-specialized partitions
  capture from an `scf.if` result (`no value rule for scf.if`) and
  `ttng.vote_ballot_sync`.
- **Completion/wait work-item correspondence is not modeled**: a mid-run extra
  completion that the waits absorb in order (an MMA commit for a zero-length Q
  tile the consumer skips) looks like a trailing completion, a note. Only two
  such completions on one CTA give a P0-2 hang (`zero_q_two_k_tiles*` inputs).
- **P1 classes other than P1-1 are not covered**: shared- and global-memory
  buffer reuse WAR/RAW, deadlocks that depend on timing or ring depth, TMEM
  allocation lifetime and tcgen05 fences, proxy fences, TMA descriptor legality,
  writer/reader layout agreement, numerics and scale factors
  (`bug_taxonomy.md`). P1-1 (TMEM reuse ordering) is CuTe only; named barriers
  and CTA syncs are modeled for P1-1 only.
- Only concrete inputs are checked. CLC is analyzed under 3 schedules. Variants
  selected only for huge inputs (i64 indexing) or by non-default environment
  switches are checked only where a case selects them. Ping-pong consumers that
  split one slot's phases fail P0-2 unless a uniform share assignment also fits
  (lockstep ledger). CuTe supports sm_100 with DSL 4.4.2 only.

### Spec model

- Concrete inputs only; the case file must include the degenerate inputs (empty
  segment, partial tiles, zero-trip loops, more tiles than CTAs, every flag
  combination), and its `elision_flags` must name every flag that selects a
  mask-elided path. A flag missing there loses nothing: the violations of its
  inputs are reported under P0-1/P0-3.
- No interleavings are enumerated. Wait cycles that exist in the lockstep model
  (every arrival as early as the roles' waits allow) FAIL P0-2; deadlocks that
  depend on timing, ring depth or bytes that never land are P1. The cross-role
  reasoning (happens-before, optimistic progress) uses mbarrier waits only:
  named barriers and CTA syncs are not in a P0 KIR, which can only make it more
  conservative (e.g. an initialization arrival ordered by a CTA sync alone is
  not one). P1-1 adds them to its own happens-before.
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
