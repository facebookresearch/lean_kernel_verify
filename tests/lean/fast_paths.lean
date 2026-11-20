/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
The evaluator's loops that replace a zip or a lane-state loop give what they replace
(tests/lean/run_all.sh, step 6): `mapBin` / `mapBinRef` (every operation, every KIR width, edge
values, shift amounts around the width), `cmpLanes` / the zipped comparison, `bcastData` /
`bcastDataRef`, `maskedVals` / the zipped filter. Lean stops with an error at the first `#guard` that fails.
-/
import KV

open KV

namespace KV.FastPaths

/-- A deterministic pseudo-random stream (a 64-bit LCG). -/
def next (s : Nat) : Nat := (s * 6364136223846793005 + 1442695040888963407) % 2 ^ 64

/-- `n` values of width `w`: edge values, small numbers (shift amounts around `w`) and
pseudo-random ones. -/
def values (w n seed : Nat) : Array (BitVec w) := Id.run do
  let edge : List Nat := [0, 1, 2, 3, 2 ^ w - 1, 2 ^ w - 2, 2 ^ (w - 1), 2 ^ (w - 1) - 1, 2 ^ (w - 1) + 1,
    w - 1, w, w + 1, 2 ^ w / 3]
  let mut out : Array (BitVec w) := edge.toArray.map (BitVec.ofNat w ·)
  let mut s := seed + 1
  while out.size < n do
    s := next s
    out := out.push (BitVec.ofNat w (if s % 3 == 0 then s / 2 ^ 40 % (w + 3) else s / 7))
  return out

def allOps : List BinOp :=
  [.add, .sub, .mul, .sdiv, .udiv, .srem, .urem, .shl, .ashr, .lshr, .and, .or, .xor, .smin, .smax, .umin, .umax]

/-- `mapBin` and `mapBinRef` agree on `xs` against a rotation of `xs` (every pair of edge
values meets once over the rotations). -/
def binAgrees (w : Nat) (o : BinOp) (addr : Bool) (seed : Nat) : Bool :=
  let xs := values w 64 seed
  (List.range 13).all fun r =>
    let ys := (xs.extract r xs.size) ++ xs.extract 0 r
    mapBin o addr xs ys == mapBinRef o addr xs ys

#guard allOps.all fun o => [1, 8, 16, 32, 64].all fun w => [true, false].all fun addr =>
  (List.range 3).all fun seed => binAgrees w o addr seed

-- operands of different sizes take `mapBinRef`
#guard mapBin .add true (values 32 5 0) (values 32 4 1) == mapBinRef .add true (values 32 5 0) (values 32 4 1)

def allCmps : List CmpOp := [.eq, .ne, .slt, .sle, .sgt, .sge, .ult, .ule, .ugt, .uge]

/-- `cmpLanes` and the zipped map agree on `xs` against its rotations. -/
def cmpAgrees (w : Nat) (o : CmpOp) (seed : Nat) : Bool :=
  let xs := values w 64 seed
  (List.range 13).all fun r =>
    let ys := (xs.extract r xs.size) ++ xs.extract 0 r
    cmpLanes o xs ys == Array.zipWith (fun p q => ofBool (cmpBV o p q)) xs ys

#guard allCmps.all fun o => [1, 8, 16, 32, 64].all fun w => (List.range 3).all fun seed => cmpAgrees w o seed

/-- `bcastData` and `bcastDataRef` agree for `shape` along `axis`. -/
def bcastAgrees (shape : List Nat) (axis : Nat) : Bool :=
  let src := shape.set axis 1
  let xs : Array Nat := Array.range (numel src)
  bcastData xs axis shape == bcastDataRef xs axis shape

#guard bcastAgrees [3, 5, 4] 1 && bcastAgrees [6, 7] 0 && bcastAgrees [5, 3] 1 && bcastAgrees [2, 3, 4, 5] 2 &&
  bcastAgrees [1, 1] 0 && bcastAgrees [4] 0 && bcastAgrees [32, 2048] 0 && bcastAgrees [32, 2048] 1

/-- `maskedVals` and the zipped filter agree. -/
def maskAgrees (n seed : Nat) : Bool :=
  let vals : Array Int := (values 64 n seed).map (·.toInt)
  let bits : Array (BitVec 1) := (values 64 n (seed + 7)).map fun b => BitVec.ofNat 1 (b.toNat % 2)
  maskedVals vals bits == (vals.zip bits).filterMap fun (v, k) => if k = 1 then some v else none

#guard (List.range 5).all fun seed => maskAgrees (20 + 37 * seed) seed

end KV.FastPaths
