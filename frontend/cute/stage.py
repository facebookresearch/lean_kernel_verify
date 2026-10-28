# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The analyzed IR stage: a prefix of the DSL's own `cute-to-nvvm` pipeline.

The DSL compiles a module with one pass pipeline, `cute-to-nvvm{...}` (target arch among
its options; recorded per compile by host_patches.py). MLIR expands it to

    cute-remove-static-args, materialize-tma-multicast, software-pipeline, cute-desugar,
    cute-expand-ops, cute-fold-static, insert-range-for-threadids, canonicalize, cse,
    loop-invariant-code-motion, ..., convert-scf-to-cf, ..., cuda-to-binary

The analyzed stage is that exact prefix up to and including the first `cse`: every
layout function (local_tile, partitions, composition) has been folded into explicit
crd2idx / add_offset arithmetic and every copy atom expanded into per-thread
loads/stores or `cute_nvgpu.arch.*` ops with explicit TMA coordinates, while `scf`
control flow is still structured. This file contains no pass list of its own.

The kernel stage runs with the pass verifier off and verifies every op afterwards; the
failures are returned and to_kir.py accepts only those its op table documents (with
cutlass-dsl 4.4.2 the global operand of `cute_nvgpu.arch.copy.SM90.bulk_copy_g2s` loses
its alignment attribute in `cute-expand-ops`).

The host function is read from a second lowering without `cute-expand-ops`, where a TMA
descriptor is still one `make_non_exec_tiled_tma_load/store` op over the global view
(expanded, it becomes bit-packing of descriptor words).

Stages are printed in generic op form with debug locations (mlir_text.py).
"""

from typing import Any

from cutlass._mlir import ir as _ir, passmanager as _passmanager

# The DSL's MLIR bindings are native extension modules that cutlass._mlir re-exports with
# `import *`; the DSL version the type checker resolves ships no stubs for them, so their
# names (Context, Module, PassManager, ...) are typed Any here.
ir: Any = _ir
passmanager: Any = _passmanager

_LAST = "cse"


def _split_top_level(s: str) -> list:
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "{(":
            depth += 1
        elif ch in "})":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def _prefix(dsl_pipeline: str) -> list:
    """The passes of the DSL pipeline's expansion up to and including the first cse."""
    with ir.Context():
        full = str(passmanager.PassManager.parse(dsl_pipeline))
    head = "builtin.module("
    if not (full.startswith(head) and full.endswith(")")):
        raise RuntimeError(f"unexpected expansion of the DSL pipeline: {full[:200]}")
    passes = _split_top_level(full[len(head) : -1])
    if _LAST not in passes:
        raise RuntimeError("the DSL pipeline no longer contains a top-level cse pass")
    return passes[: passes.index(_LAST) + 1]


def _run(mlir_text: str, passes: list, verify: bool):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(mlir_text)
        pm = passmanager.PassManager.parse("builtin.module(" + ",".join(passes) + ")")
        pm.enable_verifier(verify)
        pm.run(module.operation)
        failures = []

        def visit(op):
            o = op.operation
            if not o.regions:
                try:
                    o.verify()
                except ir.MLIRError as e:
                    failures.append(
                        {"op": o.name, "loc": str(o.location), "error": str(e)}
                    )
            return ir.WalkResult.ADVANCE

        if not verify:
            module.operation.walk(visit)
        text = module.operation.get_asm(
            enable_debug_info=True, print_generic_op_form=True, assume_verified=True
        )
    return text, failures


def lower(mlir_text: str, dsl_pipeline: str):
    """Initial DSL MLIR -> (kernel stage, host stage, kernel-stage verifier failures).

    Failures are [{"op", "loc", "error"}] for every op of the kernel stage that does not
    verify."""
    passes = _prefix(dsl_pipeline)
    kernel_text, failures = _run(mlir_text, passes, verify=False)
    host_text, _ = _run(
        mlir_text,
        [p for p in passes if p.split("{")[0] != "cute-expand-ops"],
        verify=True,
    )
    return kernel_text, host_text, failures
