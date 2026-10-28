# IR op tables (trust base item 2)

One meaning per IR op, identical for every kernel. The CuTe DSL table is in
`ir_ops_cute.md`. Executable form of this TTGIR table: `frontend/ttgir/op_table.py`
(classification and operand layouts) and `frontend/ttgir/to_kir.py` (translation); keep
them in sync with this file.

## TTGIR (Triton with TLX, target cuda:100)

Sources for the semantics: the op definitions (`include/triton/Dialect/*/IR/*Ops.td`,
operand order = printed `assemblyFormat`) and their NVIDIA lowering
(`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/{BarrierOpToLLVM,TMAToLLVM,DotOpToLLVM/
MMAv5,LoadStoreOpToLLVM}.cpp`), which fix how many threads execute each op; PTX ISA for
mbarrier, cp.async.bulk.tensor, tcgen05.commit and clusterlaunchcontrol.

Every op of a kernel falls in exactly one class; any other op is a hard error wherever
it appears (it could hide an event).

* **event**: produces KIR events (always kept);
* **value**: translated to an Expr / TExpr / reference when its result is on the slice;
* **structure**: control flow and roles;
* **drop**: no P0 meaning; slicing proves none of its results reaches a kept op.

Module attributes: `ttg.num-warps` (warps of the `default` role) is required and
`ttg.threads-per-warp` must be 32; `ttg.target` must be the target the dump compiled for
(`cuda:100` = sm_100, `cuda:103` = sm_103), recorded as the KIR `target`.

Notation: `[p]` = the event is emitted under `if p` (omitted when `p` is the constant
true, the event is removed when it is the constant false). `bar(x)` resolves a memdesc
to `(name, idx)`: `ttg.memdesc_index %a[%i]` of a `ttg.local_alloc` of type
`!ttg.memdesc<N x i64>` gives `(%a, i)`; a captured partition argument resolves through
its warp_specialize operand. All TTGIR barrier/TMA/MMA ops below are executed by one
thread of the issuing role (leader pattern), so roles have `threads_per_issue = 1`.

### Events

| Op (printed operands) | KIR | Semantics and checks |
|---|---|---|
| `ttng.init_barrier %b, N` | `bar_init(bar(%b), N)`; Barrier `{name, count N, num}` | `mbarrier.init` with count N (elected thread, before the warp_specialize: emitted in the `default` role). All slots of one alloc must get the same N. |
| `ttng.arrive_barrier %b, N [, %p] {attrs}` | `[p] arrive(bar(%b), n)` | Thread 0 of the role arrives with count N: `n = N`. With `perThread` every thread arrives: `n = N x warps x 32`. `ctaMask` (cluster multicast) or a barrier in `shared_cluster` memory: error. |
| `ttng.barrier_expect %b, B, %p` | `[p] expect_tx(bar(%b), B); arrive(bar(%b), 1)` | `mbarrier.arrive.expect_tx` by thread 0: raises the expected transaction bytes by B and arrives once. |
| `ttng.wait_barrier %b, %phase [, %p] [deps ...]` | `[p] wait(bar(%b), parity = phase)` | `mbarrier.try_wait.parity` loop on the raw phase operand (P0-4 checks it is 0/1). |
| `ttng.async_tma_copy_global_to_local %d[%c...] %dst, %b, %p` | `[p] tma_load(desc(%d), coords, bar(%b))` | `cp.async.bulk.tensor` global->shared of the block (TMA block, below); completes block x elem_bytes transaction bytes on the barrier (full block even when clipped). Coordinates are in descriptor dim order. `offsets` (im2col), a multicast target or `two_cta` attribute: error. Through an in-kernel descriptor with runtime extents: error. The descriptor's extents are checked against its base tensor (below). |
| `ttng.async_tma_copy_local_to_global %d[%c...] %src` | `tma_store(desc(%d), coords)` | Bulk tensor store of the block, clipped at the descriptor extents (checked against its base tensor, below). Through an in-kernel descriptor: `gstore` of its base pointer with the TMA footprint (below). |
| `ttng.async_tma_reduce kind, %d[%c...] %src` | `tma_reduce(desc(%d), coords)` | Bulk tensor reduce-into-global (any `kind`), clipped like a store. Through an in-kernel descriptor: `gatomic` of its base pointer with the TMA footprint. |
| `ttng.tc_gen5_mma %a, %b, %d, %useD, %p %bar0[%q0], ...` / `ttng.tc_gen5_mma_scaled` | `[p] { [q_i] commit_arrive(bar(%bar_i)) }` | Asynchronous tcgen05 MMA issued by one thread; each listed barrier receives one arrive (`tcgen05.commit ... mbarrier::arrive::one`) when the MMA and the issuer's earlier tcgen05 ops complete, predicated `p && q_i`. Operands A/B/D are data (dropped). `two_ctas` / `multicast`: error. |
| `ttng.tc_gen5_commit %b [, %p]` | `[p] commit_arrive(bar(%b))` | One arrive when all prior async tcgen05 ops of the issuing thread complete. `descs` (multicast): error. |
| `tt.load %ptr [, %mask [, %other]]` | `gload(T, offsets, mask)` | Pointer-based global read. `T` = kernel pointer argument the pointer derives from; offsets = element offsets (below). A scalar access has shape `[1]`. If the (scalar, integer) result is on the slice it is also the value `load(T, [offset])`; masked: `select(mask, load(T, [select(mask, offset, 0)]), other)` (a disabled lane reads nothing, so its index is never out of range). T becomes metadata; the analyzed kernel must not write it nor any tensor sharing its storage (error). |
| `tt.store %ptr, %val [, %mask]` | `gstore(T, offsets, mask)` | Pointer-based global write (the stored value is data). |
| `tt.atomic_rmw op, sem, scope, %ptr, %val [, %mask]` / `tt.atomic_cas` | `gatomic(T, offsets, mask)` | Read-modify-write; footprint only. |
| `ttng.tensormap_create %desc_ptr, %g, [box], [gdim], [gstride], [estride] {elem_type}` | none, or `let`s | Writes a TMA descriptor into `%desc_ptr` (a `ttg.global_scratch_alloc`) for base pointer `%g` (must be a kernel pointer argument, offset 0). Operand lists are innermost-first; KIR shape/box are outermost-first; `gstride` is in bytes (dims > 0); `gdim` is unsigned (the tensormap `global_dim` is a b32 that TMA reads unsigned: -64 is 2^32 - 64 elements), so KIR extents are the i32 operands zero-extended to i64. If every extent is a host-side expression (params, constants) loads through it read a `desc` tensor (named by `%desc_ptr`) with those shape/strides; otherwise `let <d>.shape.i` / `<d>.stride.i` are emitted here and loads through it are an error. Stores and reduces through it always use the TMA footprint on `%g`. Element strides must be 1, `interleave_layout` 0, `elem_type` one of the byte-sized types 0-12 of the PTX `tensormap.replace .elemtype` encoding with the size of `%g`'s elements (13-15, packed sub-byte data: error). A second `tensormap_create` into the same scratch buffer: error (which one a use reads would depend on the execution order). |
| `ttng.async_clc_try_cancel %b, %resp` | `tma_load(desc __clc_response, [0], bar(%b))` | Cluster Launch Control request: its 16-byte response is written through the async proxy and completes 16 transaction bytes on `%b`. `__clc_response` is a constant 16 x 1-byte descriptor. Clusters > 1 CTA: error. |
| `ttng.clc_query_cancel %resp` (x, y, z) | `x := load(__clc_schedule, [pid.x, __clc_k]); __clc_k := __clc_k + 1`; y, z := `x == -1 ? -1 : 0` | Decodes the response: id of the cancelled CTA, or -1. The k-th query of a role returns the input's CLC schedule entry k for its CTA (`frontend_ttgir.md`, CLC). Advances the role's query counter even if unused, so it is kept like an event (also in a branch with no other event). Multi-dimensional grids: error. |
| `tt.assert %cond, "msg"` | `assert(cond, msg)` | Scalar condition only (tensor condition: error). |

**TMA block.** Every TMA op moves the block of its `!tt.tensordesc<BxT, #enc>` type,
which is the KIR `box` of the descriptor (and of the footprint below). The lowering
(`LoadStoreOpToLLVM.cpp`) cuts the shared-memory tile into messages of
`getTMABlockShape(tile)` (every dim clamped to 256, the contiguous dim set to the swizzle
width in elements), and the hardware moves one tensormap box per message: for a host
descriptor the driver's `getTMABlockShape(block)`, for an in-kernel one the
`tensormap_create` box. They move exactly the block iff (all checked, else error): the
shared-memory tile is the block (leading unit dims may be dropped), both types have the
same NVMMA encoding (swizzle, transposition, element width) and it is not `fp4Padded`
(the lowering would double the innermost coordinate), the message box equals the
tensormap box and divides the block in every dim. A host `TensorDescriptor` must also
have that block shape and element size in every launch.

**Descriptor extents.** Layer 2 checks a `tma_load` / `tma_store` / `tma_reduce` against
the extents of its `desc` tensor, where TMA clips. So the extents must lie inside the
tensor behind the descriptor: per input, every in-extent element of a host
`TensorDescriptor` (its `shape` and `strides` over its `base` tensor) and of an in-kernel
descriptor with host-side extents that a TMA load reads (extents evaluated on the
input's params, over the base pointer argument) must be an element of the base tensor's
allocation (`kir.md`: its elements when the strides do not overlap, else its offset
span; element offsets from the base's data pointer). Otherwise the specialization is
refused.

**TMA footprint** of a store/reduce at `coords` through an in-kernel descriptor with
extents `shape`, strides `strides` (elements) and block `box` over base tensor `T`:
`gstore`/`gatomic(T, offsets, mask)` with, for every block index `i`,
`offsets[i] = sum_d (sext64(coords_d) + i_d) * strides_d` and
`mask[i] = AND_d 0 <= sext64(coords_d) + i_d < zext64(shape_d)`. This is the exact set of
elements TMA writes; P0-1 checks them against `T`'s allocation and P0-3 states coverage
on `T`.

**Pointer offsets.** A pointer is a kernel pointer argument plus the offsets added by its
`tt.addptr` chain; each `tt.addptr` sign-extends its offset (scalar or tensor, i32 or i64)
and adds it in 64-bit address arithmetic. `tt.splat` / `tt.broadcast` /
`tt.expand_dims` / `ttg.convert_layout` reshape the tensor terms. An event's offsets are
the single tensor term itself (in its own width; read signed) when it is the only
non-zero term, else the i64 sum of all terms (`splat` of the scalar sum, plus `ext` of
i32 tensor terms).

### Values

| Op | KIR |
|---|---|
| `arith.constant` (int / i1 scalar; `dense<v>` int tensor) | `const` / TExpr `full` |
| `arith.addi subi muli divsi divui remsi remui shli shrsi shrui andi ori xori maxsi minsi maxui minui` | `bin` add sub mul sdiv udiv srem urem shl ashr lshr and or xor smax smin umax umin (TExpr `bin` on tensors; a scalar operand of a tensor op is splatted); `addr` on add, sub, mul and shl on an address path |
| `arith.cmpi pred` | `cmp pred` (TExpr `cmp`) |
| `arith.select` | `select` (TExpr `select`; a scalar condition is splatted) |
| `arith.extsi` / `extui` / `trunci` / `index_cast` | `ext signed` / `ext unsigned` / `trunc` / signed `ext` or `trunc` (TExpr `ext` / `trunc` on tensors); `addr` on a `trunc` on an address path |
| `math.absi` (scalar) | `select(a < 0, 0 - a, a)` |
| `tt.get_program_id x/y/z`, `tt.get_num_programs` | `pid axis`, `nprog axis` |
| `tt.make_range {start, end}` | TExpr `range` (w 32) |
| `tt.splat`, `tt.broadcast`, `tt.expand_dims {axis}` | TExpr `splat`, one `bcast` per broadcast dim, `expand` |
| `ttg.convert_layout` | identity (layouts do not change values) |
| `tt.addptr` | pointer offset term (above) |
| address path | a value is on one iff it flows into a pointer offset, a TMA coordinate, a descriptor's global address or a metadata load index through operands that carry it: not through `cmpi` operands, `select`/`scf.if` conditions or load masks (those choose a value, they are not part of it) |
| `ttg.local_alloc : !ttg.memdesc<N x i64>` (no initializer) | mbarrier array of N slots, named by its SSA name |
| `ttg.memdesc_index %a[%i]` | barrier slot `(a, i)`; on any other buffer it never reaches the slice |
| `nvg.cluster_id` | `0` when the cluster is 1x1x1, else error |
| `ttg.global_scratch_alloc`, `ttng.reinterpret_tensor_descriptor` | descriptor memory and the descriptor `tensormap_create` wrote into it |
| `ttng.map_to_remote_buffer` | error when it reaches an event (peer-CTA shared memory) |
| kernel arguments | `iN` -> param `var`; `!tt.ptr<T>` -> global tensor; `!tt.tensordesc<...>` -> desc tensor (its expanded `desc.shape.i` / `desc.stride.i` arguments are params); floats on the slice: error |

### Structure

| Op | KIR |
|---|---|
| `ttg.warp_specialize(captures) default {..} partitionN(args) num_warps(W) {..}` | roles `default` (code before + default region + code after) and `partitionN` (W warps); partition arguments bound from the captures at role entry (`let` for integers) |
| `scf.for %iv = lb to ub step s iter_args(...)` | `let` carried values; `for iv in [lb, ub) step s` (i32 bounds only; `scf.for unsigned`, i.e. unsignedCmp, is an error since KIR `for` compares signed); carried values reassigned at the end of the body through `.next` temporaries; results bound by a `let` of their carried variable after the loop (sibling loops reuse the `%argN` names) |
| `scf.if` | results declared `let r := 0` before, assigned in both branches |
| pointers carried by `scf.for` / `scf.while` or yielded by `scf.if` (the `ptrs += BLOCK` idiom) | error: KIR variables are integers |
| `scf.while (args) {before; scf.condition(c) fwd} do {after; scf.yield}` | before-region emitted before the loop and at the end of the body; `while c` with `max_iters` = 65536 fuel; after-arguments assigned from `fwd` in parallel (through `.in` temporaries: they reuse the before-arguments' names) |
| `cf.cond_br` / `cf.br` (early `return`) in a kernel or partition body | nested `if`s over the acyclic block graph; loops or block arguments: error |
| `scf.yield`, `scf.condition`, `ttg.warp_yield`, `ttg.warp_return`, `tt.return` | consumed by their parent |

### Drop (no P0 meaning; results must not reach the slice)

Floating-point arithmetic and math (`arith.addf subf mulf divf remf negf maximumf
minimumf maxnumf minnumf truncf extf sitofp uitofp fptosi fptoui cmpf bitcast`,
`math.exp exp2 log log2 sqrt rsqrt fma absf erf tanh floor ceil sin cos`,
`tt.fp_to_fp precise_divf precise_sqrt clampf`), tile data movement and reshaping
(`ttg.local_load local_store local_dealloc memdesc_trans memdesc_reinterpret
memdesc_subslice`, `ttng.tmem_alloc tmem_load tmem_store tmem_subslice tmem_copy`,
`tt.trans reshape split join cat bitcast reduce reduce.return scan scan.return`),
pure `tt.elementwise_inline_asm` (side-effecting asm is an error), fences and waits that
order memory but are not P0 events (`ttng.fence_async_shared`, `ttng.fence`,
`ttng.async_tma_store_wait`, `ttng.tensormap_fenceproxy_acquire`, `ttg.barrier`), and
`tt.print`.
