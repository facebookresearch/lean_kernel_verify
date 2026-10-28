# CuTe DSL op table (trust base item 2)

One meaning per op of the analyzed CuTe stage (`frontend_cute.md`), identical for every
kernel. Executable form: `frontend/cute/ops.py` (integers, control flow, thread
registers, warp collectives, CuTe tuples/layouts/pointers) and `frontend/cute/ops_mem.py`
(global memory, TMA, mbarriers, inline PTX, host side); keep them in sync with this file.

Sources: the op definitions and printed types of nvidia-cutlass-dsl 4.4.2
(`cutlass/_mlir/dialects/_cute_ops_gen.py`, `_cute_nvgpu_ops_gen.py`; operand segments
of the TMA and bulk-copy ops are read from `operandSegmentSizes`), the CuTe layout algebra
(crd2idx, slicing, colexicographic coordinate splitting, ScaledBasis strides, CuTe's
`fill_tma_gmem_shape_stride`), the PTX ISA (`shfl.sync`, `vote.sync`, `elect.sync`,
`mbarrier.*`, `cp.async.bulk(.tensor)`, `cp.reduce.async.bulk`, `cp.async`, `st.global`,
`tcgen05.commit`), and the PTX the DSL emits for the copy and barrier atoms and the SASS
ptxas makes of the TMA stores and reduces (checked on small kernels and on a full
attention backward's dQ reduce: see "Issuing threads").

The table is valid for that DSL release only (`ops.DSL_VERSION`): the issuing threads
of an op are decided by its lowering below the analyzed stage (for TMA stores by ptxas,
checked with ptxas 13.0; for TMA loads and reduces by the `elect.sync` the DSL emits).
`dump.py` records the DSL the process loaded and `to_kir.py` refuses a dump made with
another release (`frontend_cute.md`, "DSL version").

Every op falls in exactly one class; an op of no class is a hard error wherever it
appears, even off the slice (it could hide an event). The slicer visits every op of the
kernel and of the host function and rejects it there (`ops.effect_operands`), kept or
not; `llvm.inline_asm` is classified statement by statement (below):

* **event**: emits KIR events whenever a thread of the role executes it;
* **value**: translated only when one of its results is on the slice (`slicer.py`);
* **structure**: `scf.if/for/while/yield/condition`, `cuda.kernel`, `func.func`, returns;
* **data**: no global-memory or mbarrier effect; results are Data. A Data value
  reaching a kept use is the slice soundness error, reported with both locs.

## Notation

Abstract values (`interp.py`): `e` integer expression (exact BitVec semantics);
`Tup` IntTuple (nested; leaves are expressions, basis elements `k@m` or `n/d@m`, or `_`);
`Lay(shape, stride)`; `Ptr(space, base, off, bits)` (offset in `bits`-wide elements;
`base` is the kernel tensor of a global pointer); `View(ptr, lay)`; `Iter(coords)`
(ArithTuple iterator); `Opaque` (atoms, descriptors, addresses, divisors).
`LANE`, `WARP` are the executing thread (`tid.x = 32*WARP + LANE`); `ELECT` is true on the
lane `elect.sync` picks. Types are read from the op's printed types: in `"..."` layout
strings `?` leaves are dynamic and filled, in order, from the op's operands.

`mask` is the per-thread condition of the enclosing thread-varying branches; its
thread-uniform conjuncts become a KIR `if` around the event.

## Issuing threads

| Op family | Executed by | KIR |
|---|---|---|
| global loads/stores, `cp.async`, bulk reduce, bulk copy, `expect_tx`, `tcgen05.commit` | every thread of the role with `mask` true | one event per issuing thread (one event if one thread issues it, else a `for` over the role's threads with `mask` as an `if`, or a `for` over the issuers) |
| TMA store (`tma_store`) | in each warp, once per distinct (coordinates, shared-memory source) among its lanes with `mask` true: ptxas issues the instruction in a waterfall loop over the distinct operand values | one `gstore` per distinct tuple, issued by the lowest such lane holding it: one event when one lane issues (e.g. warp-uniform operands), else a `for` over the role's threads with that condition as an `if` |
| `mbarrier.arrive`, arrive half of `arrive_expect_tx` | every issuing thread | one `arrive` whose arrivals count the issuers (`threads_per_issue` = gcd over the role's arrives, `n` = issuers / gcd) |
| TMA load (`tma_load`) | one lane per warp with an issuing thread: the DSL lowers it to `elect.sync` + `cp.async.bulk.tensor`; that `elect.sync` has the full member mask, so each such warp must be fully active (else error, as for an explicit `elect.sync`) | one `tma_load` and one footprint `gload` per such warp |
| TMA reduce (`tma_reduce`) | as a TMA load: one lane per warp with an issuing thread, the DSL lowering it to a full-mask `elect.sync` + `cp.reduce.async.bulk.tensor`, so each such warp must be fully active (else error; also inside `elect_one`, where one lane reaches it) | one footprint `gatomic` per such warp, translated as under an explicit `elect.sync` (its operands must not depend on LANE) |
| `mbarrier.init` | every issuing thread; with one address and count the state is one initialization | one `bar_init` |
| `mbarrier.try_wait.parity` (blocking form) | every issuing thread spins until the phase completes | one `wait` for the role |

Checked in the PTX of test kernels (nvidia-cutlass-dsl 4.4.2, sm_100a): `mbarrier_init`
and `mbarrier_arrive_and_expect_tx` outside `elect_one` run on every lane; `cute.copy`
with a TMA load or TMA reduce atom emits `elect.sync` with member mask -1 before
`cp.async.bulk.tensor` / `cp.reduce.async.bulk.tensor` (a TMA reduce inside `elect_one`
gets a second `elect.sync`, nested in the first); TMA stores and `cp.async.bulk`
global->shared reach PTX on every executing thread (the PTX of
`if warp == 0: cute.copy(tma_store_atom, sC, gC)`, `CUTE_DSL_KEEP_PTX=1`, has
`cp.async.bulk.tensor.2d.global.shared::cta` under the warp test only, no `elect.sync`,
with 4.4.2 and with 4.6.1); `mbarrier_wait` is an inline-PTX `try_wait.parity` loop. The
TMA reduce rows come from the PTX and cubin that the DSL's own `cute-to-nvvm` pipeline
makes of the module the frontend captures (`dump-ptx-path`, `dump-cubin-path`):
`tests/cute` `_tma_store` in modes `reduce`, `reduce_half`, `reduce_elect` and
`reduce_lane_coords`, and a block-scaled attention backward kernel whose
dQ ring issues `cute.copy(tma_atom_dQ, ...)` from a whole warp without `elect_one`.

Checked in the SASS (`cuobjdump -sass` of CUDA 13.0; the store rows on PTX compiled by
ptxas 13.0, `-arch=sm_100a`, with operands edited in the PTX). The TMA instructions take
their operands from uniform registers.
* Warp-wide store, operands the same on every lane: one `UTMASTG` and no loop, as with
  `elect_one` (`ELECT P0; @!P0 EXIT; UTMASTG`). The same holds when `commit_group` and
  `wait_group.read 0` follow it (an attention backward's dK/dV epilogue), and when every lane loads the
  coordinate from one shared or global address.
* A lane-dependent coordinate (`tid & 31`) or shared-memory source (`+ (tid & 1) * 8192`)
  gives a waterfall loop: `R2UR P0, UR, R` takes one remaining lane's value and marks the
  lanes holding it, `UTMASTG`, then `BRA.U.ANY` back while lanes remain. That is one
  `UTMASTG` per distinct operand tuple.
* TMA reduce: one `UTMAREDG` per warp, with no loop. In the test kernel the DSL's
  `elect.sync` stays as `ELECT P0; @!P0 EXIT` before it; in that attention kernel's
  dQ ring (two reduces) ptxas drops it, as the uniform-datapath `UTMAREDG` issues once per execution
  by the warp. A lane-dependent coordinate is the elected lane's (`R2UR`). Only a reduce
  without that `elect.sync` (hand-edited PTX) gets an elect loop with one `UTMAREDG` per
  executing lane (`ELECT P1`, `UTMAREDG`, `BRA.U.ANY`); the DSL does not emit one.

The descriptor operand is the atom's host descriptor (one per atom), so the store's key
is its coordinates and shared-memory source; its cache-policy operand (an atom field) is
taken to be lane-uniform. The frontend side is `tests/cute`
`test_warp_wide_tma_store_is_issued_once_per_distinct_operands`,
`test_warp_wide_tma_reduce_is_issued_once_per_warp` and
`test_tma_reduce_needs_a_full_warp_and_lane_uniform_operands`.

`elect.sync` picks one active lane per warp; which one is unspecified, so an event under
ELECT is counted once per warp and its operands must not depend on LANE (hard error).
With its full member mask every lane of the warp must execute it (PTX: undefined
otherwise), so it is a hard error when a warp reaching it has only some lanes active
under the current lane mask, or when that depends on runtime values.

Barrier ops need a thread-uniform address and operands; `bar_init`, `wait` and TMA loads
need a statically known set of issuing threads (hard error otherwise). An arrive or
expect_tx under a runtime-dependent lane mask is a per-thread loop.

## Events

| Op | KIR | Meaning and checks |
|---|---|---|
| `cute.memref.load %v, %c` / `cute.memref.store %v, %c, %x` (gmem view, or generic view of a gmem pointer) | `gload` / `gstore` | One element at `v.ptr + crd2idx(c, v.layout)`. An integer load on the slice is also the value `load(T, off)` (T becomes a metadata tensor; under a mask the index is 0 where the mask is false). A generic view takes its pointer's space. Other address spaces: data. A gmem or generic view computed from data (e.g. `inttoptr` of an integer), or a generic view whose pointer's space is unknown: error. |
| `cute.memref.load_vec %v` / `cute.memref.store_vec %x, %v` (gmem, or generic as above) | `gload` / `gstore` over the view | Offsets `ptr + sum_k i_k * stride_k` over the layout's leaves (static extents; a dynamic extent is an error). |
| `llvm.load %p` / `llvm.store %x, %p` | `gload` / `gstore` of the value's elements | `!llvm.ptr<1>`, or a generic pointer derived from a global CuTe pointer. Shared/register pointers: data, and no event at all when the pointer surely is one by its definition (an alloca, or a cast of a register or shared-memory CuTe pointer, through GEPs: no branch condition is then needed). A pointer of unknown provenance: error. |
| `cute_nvgpu.arch.copy.SM80.cp_async %dst, %src [, %pred]` | `gload` of one source element | Global -> shared async copy; with `%pred`, a false predicate reads nothing (src-size 0). No mbarrier. |
| `cute_nvgpu.arch.copy.SM100.tma_load %desc, %smem, %bar, %c..., [mcast], [offsets], %cache` | `tma_load(desc, coords, bar)` + `gload` on the descriptor's kernel tensor | `mode = tiled`, `num_cta = 1` or `2` (cta_group::2: `leader`), no im2col offsets (else error); a multicast mask (thread-uniform) is `cta_mask`. Coordinates in descriptor dim order, converted to KIR descriptor units (below). Completes `box x elem_bytes` transaction bytes on `%bar` (the `tma_load`, P0-4). The `gload` is exactly the copy's footprint, elements `c + i`, `i` in the box, masked to `0 <= c + i < extent` per dim, so P0-1 checks the read against the tensor's allocation and not only against the extents the descriptor claims. Descriptor units other than the tensor's storage elements: error. |
| `cute_nvgpu.arch.copy.SM100.tma_store` / `tma_reduce` (`kind = ADD`) `%desc, %smem, %c..., %cache` | `gstore` / `gatomic` on the descriptor's kernel tensor | Exactly the copy's footprint: elements `c + i`, `i` in the box, masked to `0 <= c + i < extent` per dim (TMA drops the rest). Kept on the base tensor so that TMA and per-thread writes of one tensor meet on one KIR tensor. A store's `%smem` must be a shared-memory pointer: with the coordinates it decides how often a warp issues the store (above). |
| `cute_nvgpu.arch.copy.SM90.bulk_copy_g2s %src, %smem, %bar, [mcast], %cache {size = B}` | `gload(T, [p, p + B))` + `tma_load(bulk_T_n, [p], bar)` | `cp.async.bulk` of B bytes completing B transaction bytes. KIR has no bulk-copy event: the gload is the footprint (P0-1), the tma_load on a 1-D descriptor over T's storage with box B / elem_bytes carries the bytes (P0-4). Multicast: error. With cutlass-dsl 4.4.2 this op fails the stage verifier (its global operand loses its alignment attribute); the meaning does not use alignment. |
| `llvm.inline_asm "cp.reduce.async.bulk.global.shared::cta.bulk_group.add[.noftz].<t> [$a], [$b], $c;"` (the only statement) | `gatomic(T, [p, p + $c/elem))` | `$a` a 64-bit input holding a kernel tensor pointer `p`; `$c` must be a constant byte count of whole `<t>` elements of T. |
| `llvm.inline_asm` of data statements plus one unguarded `st.global.<t> [$k], r;` or `st.global.v2/v4.<t> [$k], {...};` (`<t>` u/s/b/f of 8 to 64 bits), no branch | `gstore(T, [p, p + N/8 bytes))` | `$k` a 64-bit input (`l`) whose operand is `cute.ptrtoint` of a kernel tensor pointer `p` (any other address, e.g. an integer: error). N/8 bytes from `p`, in T's storage elements (narrower than one storage element: error). A `@p`-guarded store, a second store or a branch next to it: error. |
| `nvvm.mbarrier.init.shared %p, %n` | `bar_init(bar(p), n)` | `p` a kernel constant, or an affine function of the induction variables of enclosing constant-trip loops (each value is an initialized address). |
| `nvvm.mbarrier.txn %p, %n {kind}` (`scope`, `space` = `cta` or `cluster`) | `arrive_expect_tx`: `expect_tx(bar, n)` + `arrive(bar)`; `expect_tx`: `expect_tx`; `arrive`: `arrive` (`n` must be 1) | `expect_tx` only raises the expected bytes (frontend/kir.py). A remote `p` (mapa) gives the events its `cta`. |
| `nvvm.mbarrier.arrive(.shared) %p` | `arrive(bar(p))` | Result (state token): data. A remote `p`: `cta`. |
| `nvvm.mbarrier.try_wait.parity.shared %p, %parity, %ticks` without result | `wait(bar(p), parity)` | The blocking form. A result-returning try_wait: error. |
| `nvvm.tcgen05.commit.arrive %p [, %mask]` (`group = cta_1` or `cta_2`) | `commit_arrive(bar(p))` | A multicast mask (thread-uniform) is `cta_mask`. |
| `nvvm.cluster.arrive`, `nvvm.cluster.arrive.relaxed` / `nvvm.cluster.wait` | `arrive` / `wait` on `cluster_barrier[0]`, `cta` 0 | Every thread of the role (else error); the barrier has one arrival per thread of the cluster; the wait's parity is the role variable `cluster_barrier_parity`, flipped after it (frontend_cute.md, "Clusters"). |
| `llvm.inline_asm` of data statements plus one blocking wait loop (`L: mbarrier.try_wait.parity.shared::cta.b64 P, [$a], $p[, hint];` then `@!P bra L`, or `@P bra D; bra L; D:`) | `wait(bar(p), $p)` | `$a` holds `cute.ptrtoint` of the shared-memory pointer `p`. A try_wait that does not loop back: error. |
| `llvm.inline_asm` of exactly `mapa rd, $dst, $r; mapa rb, $bar, $r; cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [rd], [$src], $n, [rb]` | `complete_tx(bar(p), $n, cta = $r)` per issuing thread | `$bar` holds `cute.ptrtoint` of `p`; the data written into CTA `$r` is not modeled. |
| `cf.assert %c, "msg"` | `assert(c, msg)` | Per issuing thread when `c` varies across threads. |

`bar(p)`: `p` is a thread-uniform shared-memory pointer; its byte address
`C + sum k_i * t_i` (exact affine form) names the barrier array containing `C` and index
`(C - start)/8 + sum (k_i/8) * t_i` (every `k_i` and `C - start` a multiple of 8, else
error). Arrays: frontend_cute.md, "Barriers". A remote pointer (`nvvm.mapa.shared.cluster
%p, %r`) is `p` in the CTA of rank `r` (thread-uniform): the event's `cta`.

## Values

| Op | Value |
|---|---|
| `arith.constant`, `llvm.mlir.constant` (integer) | `const` |
| `arith.addi subi muli divsi divui remsi remui andi ori xori shli shrsi shrui maxsi minsi maxui minui` | `bin` (wrapping); a divisor under a mask is 1 where the mask is false |
| `arith.floordivsi`, `arith.ceildivsi`, `arith.ceildivui` | exact expansions with `sdiv/srem/udiv/urem/select` |
| `arith.cmpi`, `llvm.icmp` (predicates 0..9 = eq ne slt sle sgt sge ult ule ugt uge), `arith.select` | `cmp`, `select` |
| `arith.extsi/extui/trunci/index_cast/index_castui` | `ext` (signed/unsigned) / `trunc` (`index` is i64) |
| `llvm.intr.ctpop` | sum of the bits |
| `nvvm.read.ptx.sreg.tid.x` / `.tid.y`, `.tid.z` | `32*WARP + LANE` / 0 (to_kir.py requires blockDim = (32k, 1, 1)) |
| `nvvm.read.ptx.sreg.laneid`, `.ctaid.*`, `.nctaid.*`, `.ntid.*` | `LANE`, `pid`, `nprog`, the launch's block size |
| `nvvm.read.ptx.sreg.cluster.ctarank / nctarank` | `x % cx + (y % cy) * cx + (z % cz) * cx * cy` over `pid` / `cx * cy * cz` (the launch's constant cluster dims) |
| `nvvm.read.ptx.sreg.clusterid.* / nclusterid.* / cluster.ctaid.* / cluster.nctaid.*` | `pid / c`, `nprog / c`, `pid % c`, `c` (c: the cluster dim of the axis) |
| `nvvm.mul {mode}` | `lo`: `mul`; `hi`: the high half of the double-width product (`isSigned`: sign extension) |
| `cute.fast_divmod.get_divisor` / `get_aux` | the divisor's value / reserved values that only the fast division idiom may consume: the `shrui` completing it is `udiv(n, d)` (frontend_cute.md) |
| `cute.get_flat_coord %i, %lay` | one coordinate per top-level mode: `(i / size(modes before)) % size(mode)` (static leaves of the result type checked) |
| `cute.append_to_rank %lay, %pad {rank}` | the layout padded with copies of `%pad` to `rank` modes |
| `nvvm.elect.sync` (no member-mask operand) | `ELECT`; every warp reaching it fully active (above) |
| `cute_nvgpu.arch.make_warp_uniform %x` | `x[LANE := 0]` (`__shfl_sync(-1, x, 0)`) |
| `nvvm.shfl.sync %mask, %v, %b, %c {kind}` (integer) | lane `l` gets `v` at source lane `j(l)` of its warp, with the PTX rule (`bval = b[4:0]`, `segmask = c[12:8]`, `maxLane`/`minLane`; `up: j = l - bval, ok = j >= maxLane`; `down: j = l + bval, ok = j <= maxLane`; `bfly: j = l ^ bval, ok = j <= maxLane`; `idx: j = minLane \| (bval & ~segmask), ok = j <= maxLane`), its own value when not ok. A table over lanes (or threads) of bound per-lane values. Partial member mask: error. |
| `nvvm.vote.ballot.sync %mask, %p` | `OR_k (zext(p[LANE := k]) << k)`. Partial member mask: error. |
| `cute.static` | the value of its (fully static) type: tuple, layout, composed layout, atom |
| `cute.make_int_tuple / make_coord / make_shape / make_stride / make_tile` | the result type's tuple with its `?` leaves filled by the operands' dynamic leaves |
| `cute.get_leaves`, `cute.get_scalars [only_dynamic]`, `cute.to_int_tuple`, `cute.derefine`, `builtin.unrealized_conversion_cast` | leaves / scalars (dynamic positions from the operand type) / identity |
| `cute.assume` | identity (divisibility facts are taken as given) |
| `cute.make_layout %shape [, %stride]` | `Lay`; without a stride: the type's static stride, or compact column-major |
| `cute.get_layout / get_iter / get_shape / get_stride`, `cute.size [mode]` | parts of a view or layout; product of the shape leaves |
| `cute.make_view %ptr [, %lay]` | `View` (layout from the type when omitted) |
| `cute.crd2idx %c, %lay` | CuTe crd2idx: hierarchical coordinates; a scalar coordinate of a tuple mode is split colexicographically; `_` contributes 0; basis strides give ArithTuple components; a coordinate times a rational basis stride `n/d@m` must be divisible by `d` (KIR assert when not constant) |
| `cute.add_offset %p, %o` | pointer offset + `o` (sign-extended to i64) / iterator componentwise add |
| `cute.slice %lay_or_view, %c` | the leaves of the `_` modes in the result type's structure (static leaves checked; a size-1 mode takes the type's stride); a view also advances by `crd2idx(c)` |
| `cute.select {mode}` | mode selection |
| `cute.filter_zeros` | modes with a static zero stride get extent 1 (result type checked) |
| `cute.tuple_add/sub/mul/div/mod`, `cute.ceil_div` | elementwise (`div` = `sdiv`; add/sub pad the shorter tuple with 0) |
| `cute.recast_iter` | the same address in the new element width: narrower = offset scaled; wider = offset / k with a KIR assert that k divides it |
| `cute.recast_layout` | the static result type (a dynamic one: error) |
| `cute.ptrtoint %p` / `cute.inttoptr %a` | the address of `p` (opaque) / `p` again when `a` is that address with the same pointer type (the alignment-assumption idiom); any other integer has no provenance (data) |
| `llvm.getelementptr %p, [idx] {elem_type, rawConstantIndices}` | offset `sum_k idx_k * sizeof(type_k)`, in the pointer's element units when every term is whole elements, else bytes |
| `cute_nvgpu.arch.get_dyn_smem` / `get_dyn_smem_size` | the shared-memory base pointer / the launch's dynamic shared memory bytes |
| `cute.memref.alloca`, `llvm.alloca` | register-memory pointer (its contents are data) |
| `cute.make_atom`, `cute.make_tiled_copy`, `cute_nvgpu.atom.make_exec_tma`, `.set_value`, `.get_value`, `cute_nvgpu.get_tma_desc_addr` | atoms carry their fields: the host descriptor and, after `set_value`, the `tma_bar` pointer; other fields are data |
| `cute.fast_divmod.create_divisor / make_divisor` | the divisor |

## Host side (host stage)

| Op | Meaning |
|---|---|
| host-function arguments | gmem `!cute.memref`: `View(Ptr(gmem, <name>), Lay)` whose `?` leaves are KIR params `<name>_shape<i>` / `<name>_stride<i>`; i32/i64 scalars: params `<name>`; other scalars: data; streams: ignored |
| `cute_nvgpu.atom.make_non_exec_tiled_tma_load / _store / _reduce %g, %smem_layout, %tiler` | a KIR `desc` tensor over `%g`'s kernel tensor (`%g` at offset 0). One descriptor dim per top-level mode of the atom's `tma_gbasis`: a mode on one global mode takes its extent and stride; a mode over several global modes takes their gcd stride and the extent `1 + sum (s_j - 1) * ceil(d_j / g)` (CuTe `fill_tma_gmem_shape_stride`; their strides must be static). Box dim = product of the mode's extents. Dim 0 must have stride 1. Units: global elements, with dim 0's box and coordinates multiplied by the format width in elements (`U16` over 8-bit scale factors: 2); packed 4-bit data (`U4`, `U4_UNPACK_U8`) in bytes (dim 0 extent, box and coordinate / 2). `box x elem_bytes x 8` must equal the atom's `copy_bits` (for `U4_UNPACK_U8` the packed source bytes, which are also the transaction bytes: the shared-memory copy expands, the count does not). With `num_multicast` n > 1 the KIR box is the slice one load moves: the box truncated by n from the outermost dim (CuTe `make_tma_copy_desc`; `copy_bits` is checked against the untruncated box). Result 1: the coordinate tensor with the type's (possibly rational) basis strides over `%g`'s shape. |
| `cuda.launch_cfg.create` | block (constants), dynamic shared memory, grid (KIR `grid`) |
| `cuda.launch_cfg.cluster_dim` | the cluster dims (KIR `cluster`; on the host slice, constants or error) |
| `cuda.launch_ex @kernel(args)` | binds the kernel's block arguments; exactly one launch per host function |
| `cuda.launch_cfg.programmatic_stream_serialization_allowed`, `.cooperative`, `cuda.cast`, `cuda.return_if_error` | no meaning |

## P1 slice (P1-1; `frontend/cute/ops_tmem.py`)

The CuTe frontend translates every variant a second time with this slice and these
handlers (on top of everything above) into the P1 KIR (`kir.md` change 10). The P0 slice
and KIR do not change: in it these ops stay data (below).

| Op | P1 event | Kept operands | Issue pattern and checks (refusal) |
|---|---|---|---|
| `cute_nvgpu.arch.copy.SM100.tmem_load` / `tmem_store` `{num_dp = 32, num_b = 32, num_rep = N}` | `tmem` ld / st, `data` read / write, lanes [L, L+32) x cols [C, C+N) | [0] | one event per issuing warp (a KIR `for` over the role's warps when all issue), every lane of the warp active and the address warp-uniform (R6); other shapes or attributes (R5). The lane-quarter rule is Lean's F3 |
| `cute_nvgpu.arch.copy.SM100.copy_s2t` `{bits = 128, dp = 32, broadcast = x4, cta = 1}` | `tmem` cp, `data` write, lanes [L, L+128) x [C, C+4) | [1] | issued once by the role's single warp, fully active, operands uniform (R6: several warps, a partial or runtime lane mask); other shapes (R5) |
| `cute_nvgpu.arch.mma.SM100.umma` / `umma_block_scaled` | `tmem` mma: `d` write lanes [0, 128) x [C_D, C_D+N) (N / 2 for M = 128 with `num_cta = 2`); `a` read (A operand a TMEM pointer, ts) [C_A, C_A + K x abits/32); `sf` reads, lanes [0, 128) from the floored SF column: SFA (operand 3) 4 columns, SFB (operand 4) 4 x ceil(N / 128) columns; `num_cta = 2`: `cta_mask` `3 << (rank & ~1)` (the TMEM of both CTAs of the pair, in a cluster; R5 without one) | [2]; [0] if a TMEM pointer; [3], [4] (block scaled) | as copy_s2t. M, N, K, the accumulator type and frag kind come from the atom type of the `atom.get_value` feeding operand 3 (block scaled); without one (a plain `umma`, operands a, b, d, idesc, accumulate; or an atom the kernel makes per call) M and N come from the instruction descriptor (every shape bit known), K from `shape_k` (block scaled) or 256 bits of `a_type` (dense), the accumulator from the D format bits 4-5 (dense; block scaled: f32). M = 128 with `num_cta = 1`, M = 128 or 256 with `num_cta = 2`, f32 (R5). Block scaled: N <= 256 and K / `vec_size` <= 4 (R5). The SF widths are the CuTe DSL 4.4.2 TMEM SF layouts (`blockscaled_layout.make_tmem_layout_sfa/sfb`) of one MMA k-block, checked for mxf8 (vec 32) and nvf4 (vec 16) at N = 64, 128, 192, 256: SFA 128 rows as 32 lanes x 4 columns (x4 lane quarters), SFB N rows padded to 128-row atoms (N = 192 and 256 span 8 columns, not N / 32); a row's K / vec <= 4 scale factors are bytes of one column. A syntactic known-bits walk of the instruction descriptor (constants, `ori`, `andi`, `shli` by a constant, `extui`) must give N >> 3 at bits 17-22 and M >> 7 at bits 27-28 (block-scaled kinds) or M >> 4 at bits 24-28 (PTX ISA tables 51-53), else R5. `idesc` and `accum` are not kept |
| `nvvm.tcgen05.wait {kind = load / store}`, inline `tcgen05.wait::ld/st` (only statement) | `tmem_wait` | none | every thread of the role (R7 otherwise) |
| `nvvm.mbarrier.try_wait.parity`, `.arrive` (and the arrive half of `txn`), `nvvm.tcgen05.commit.arrive` | as in "Events" and "Issuing threads" (`wait`, `arrive`, `commit_arrive`) | as there | in a role of several warps, every warp must have an issuing lane (any lane issues for its warp): otherwise the role is split into runs of issuing and non-issuing warps (`interp.Translator._p1_split_by_issuers`), so a thread-0 arrive or wait becomes an event of a one-warp role; under a runtime lane condition: R7 |
| `nvvm.barrier` (no operands) | `named_bar` id 0, count = the block's threads, sync | none | every thread of the role (R7) |
| inline `bar.sync` / `bar.arrive` / `barrier.sync[.aligned]` / `barrier.arrive` `a[, b]` (only statement) | `named_bar` id a, count b (omitted: the block's threads) | the `$k` operands | constant id and count (uniform per warp but not per role, e.g. a barrier per warp pair: the role is split per run of warps), count a positive multiple of 32, every thread of the role (R7) |
| inline `tcgen05.cp.cta_group::{1,2}.{128x256b, 128x128b, 32x128b.warpx4, 64x128b.warpx2::01_23/02_13} [a(+k)], desc` | `tmem` cp, `data` write, lanes [L, L+128) x [C+k, C+k+n) (n = 8 for 256b, else 4); `cta_mask` for cta_group::2 | the address's `$k` | the address `a` is `$k` or a register set from one (`mov.b32`), an absolute TMEM address (lane << 16 \| column: `cute.ptrtoint` of an absolute TMEM pointer, or a constant). Issued once: under an `elect.sync _\|P, -1` predicate of the asm with the whole warp active, else by the single active thread (R6) |
| inline `tcgen05.mma.cta_group::{1,2}.kind::{f16,tf32} [d], a, b_desc, idesc, ...` | `tmem` mma: `d` write lanes [0, 128) x [C_D, C_D+N) (N / 2 for M = 128 with cta_group::2), `a` read [C_A+k, C_A+k+8) if `a` is `[reg(+k)]` (TMEM A: 256 bits of K per MMA); `cta_mask` for cta_group::2 | the addresses' `$k` | addresses as for cp; `idesc` a constant set in the asm, or a function-scope register (`.reg .b32 r; mov.b32 r, C;` without braces) some inline PTX of the kernel sets once to one constant; M = 128 (cta_group::1) or 128 / 256 (::2), N <= 256 from its bits 24-28 / 17-22 (R5); issued as cp. One event per MMA statement; a blocking wait loop between them is a `wait` in its place (frontend_cute.md, "Clusters") |
| inline asm with other `tcgen05.{ld,st,cp,mma,shift,alloc,dealloc}` statements | none | | refused (R4): an untracked TMEM access. `tcgen05.fence`, `relinquish_alloc_permit` stay data (fences are not checked) |
| `cute_nvgpu.arch.sm100.alloc_tmem %n, %holding` | none: records the column count | [0] | `%n` a uniform constant, outside loops, one allocation per issuing warp; exactly one per kernel, 32..512 columns (R2) |
| `cute.inttoptr %a` to a TMEM pointer | `Ptr("tmem", "abs", 0, bits)` | | `%a` a uniform constant 0 (through `cute.assume`), or `cute.ptrtoint` of an absolute TMEM pointer (that pointer); any other integer is data, so a kept use is a slice soundness error (R1) |
| `cute.ptrtoint %p` of an absolute TMEM pointer | its absolute address, the i32 `lane << 16 \| column` (for inline PTX operands) | | other pointers as in the P0 pass |
| `cute_nvgpu.arch.sm100.retrieve_tmem_ptr` | `Ptr("tmem", "alloc", 0, bits)` | | relative addresses; absolute and relative pointers mixed, or absolute ones with fewer than 512 columns: R3 |
| `cute.make_tiled_mma` (host) | an atom (`Opaque`) carrying its fields | | P0: data |

Addresses: a TMEM pointer's 32-bit-column address is `off x bits / 32` (exact, with a KIR
assert when not static, for ld/st/cp/D/A; floored for SF pointers, whose byte offset is the
SF ID within the column); lane = address >> 16, column = address & 0xffff.

**Atom fields (P1).** `atom.set_value` / `get_value` carry every field; in the P1 pass an
atom's integer and pointer fields are skeleton leaves, so loop-carried atoms (the k-phase
loop sets `sf_a` / `sf_b` / `accum_c`) carry them through loop variables and `scf.if`
results merge them; a field missing on one side or with different non-integer values
becomes data, and a kept read of it is the slice soundness error (R8). The P0 pass keeps
the old flatten (atoms have no leaves), so its KIR is unchanged.

**Issuing threads (P1).** `tcgen05.ld/st` are `.sync.aligned`: every lane of the warp,
one access per warp. `copy_s2t` and the UMMA ops are executed by the whole warp at the
analyzed stage and lowered to one elected lane (`elect.sync`, full member mask); the
`tcgen05.commit` of the same warp is under `elect.sync` with the same mask, which elects
the same lane ("the same leader thread is elected for the same membermask every time"):
the landing rule takes the commit and the cp/mma to be issued by one thread (trust item
5 of references/design.md). Not checked in PTX/SASS for this release yet (future work).

## Data (dropped in the P0 slice)

Floating point (`arith.*f`, `arith.cmpf`, conversions, `math.*`,
`nvvm.*.packed.f32x2`, `nvgpu.cvt_fpext`), register vectors (`vector.broadcast/extract/
extractelement/insert/insertelement/from_elements/splat/shape_cast/shuffle/transpose/
bitcast/reduction/multi_reduction/fma`; the vector dialect's memory ops, `load/store/
maskedload/maskedstore/gather/scatter/transfer_*`, are not in the table: they may access
global memory), `llvm.extractvalue/insertvalue/undef/poison/
bitcast`, `llvm.inttoptr`; register, shared and tensor memory traffic (`cute.memref.*`
and `llvm.load/store` outside global memory, `cute_nvgpu.arch.copy.SM100.tmem_load/
tmem_store/copy_s2t` (events in the P1 slice), `cute.apply_swizzle`); MMA (`cute_nvgpu.arch.mma.SM100.umma`, `.umma_block_scaled`,
`cute_nvgpu.make_umma_smem_desc`, `cute.mma_atom_call`, `cute.mma.make_fragment`,
`cute.make_tiled_mma`); TMEM allocation (`cute_nvgpu.arch.sm100.*`,
`cute_nvgpu.atom.make_tmem_copy/make_s2t_copy/get_copy_s2t_smem_desc_view`);
`nvvm.fmax/fmin`, `nvvm.rcp.approx.ftz.f`; synchronization outside P0 (`nvvm.barrier`, `nvvm.bar.warp.sync`, `nvvm.fence.proxy`,
`nvvm.fence.mbarrier.init`, `nvvm.setmaxregister`, `nvvm.tcgen05.wait`, `nvvm.cp.async.bulk.commit.group/wait_group`,
`nvvm.cp.async.commit.group/wait.group`); `cute_nvgpu.prefetch_tma_desc`.

`llvm.inline_asm` is classified for every op, with or without results: every PTX
statement (scope braces, `.reg` declarations, labels and comments aside; a leading
`@p` guard kept) must be data (`griddepcontrol.*`, arithmetic, conversion, move,
predicate and select instructions, `ld/st.shared`,
`tcgen05.{ld,st,cp,mma,alloc,dealloc,shift,fence,wait,relinquish_alloc_permit}`,
`elect.sync`, `fence.proxy`, `bar.*`/`barrier.*`, `redux.sync`, `shfl.sync`, branches,
`trap`, which is not modeled), or the asm is one of the events above (bulk reduce,
`st.global`, a blocking wait loop, a shared::cluster bulk copy); anything else
(`red`/`atom`/`ld.global`, other stores, `mbarrier.*`, `cp.async*`, ...) is a hard error
naming the statement. A data asm is dropped; its results are Data.
