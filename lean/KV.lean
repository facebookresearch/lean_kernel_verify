/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
Layer 2 of lean-kernel-verify: KIR AST, evaluator, the P0 specs P0-1..P0-6 and the P1
specs P1-1 and P1-2 (core Lean + Std).
See references/design.md, references/kir.md and references/specs.md.
-/
import KV.IR
import KV.Eval
import KV.Check
