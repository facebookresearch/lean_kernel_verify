/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
Layer 2 of lean-kernel-verify: KIR AST, evaluator and the five P0 specs (core Lean + Std).
See references/design.md, references/kir.md and references/specs.md.
-/
import KV.IR
import KV.Eval
import KV.Check
