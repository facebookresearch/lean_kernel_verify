/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-/

/-
`BitVec` operations of the evaluator without computing `2 ^ w` per operation. The core
definitions of `+`, `-`, `*`, `<<<`, `toInt`, `ofInt`, `ofNat`, `setWidth` (truncation),
`signExtend` and `sshiftRight` compute the power `2 ^ w` at every call; `2 ^ 64` is a big number,
so every 64-bit operation allocated (the evaluation of 64-bit address vectors spent most of its
time there). Here the power of the KIR widths is a constant (`pow2`), and each operation is
proved equal to the core one (`add_eq`, ...): the evaluator keeps exact `BitVec` semantics.
-/

namespace KV.BV

/-- `2 ^ w`, a constant for the KIR widths 1, 8, 16, 32 and 64. -/
@[inline] def pow2 (w : Nat) : Nat :=
  if w = 64 then 18446744073709551616
  else if w = 32 then 4294967296
  else if w = 16 then 65536
  else if w = 8 then 256
  else if w = 1 then 2
  else 2 ^ w

theorem pow2_eq (w : Nat) : pow2 w = 2 ^ w := by
  unfold pow2
  split
  · subst_vars; rfl
  split
  · subst_vars; rfl
  split
  · subst_vars; rfl
  split
  · subst_vars; rfl
  split
  · subst_vars; rfl
  rfl

/-- `2 ^ (w - 1)`, a constant for the KIR widths. -/
@[inline] def half (w : Nat) : Nat :=
  if w = 64 then 9223372036854775808
  else if w = 32 then 2147483648
  else if w = 16 then 32768
  else if w = 8 then 128
  else if w = 1 then 1
  else 2 ^ (w - 1)

theorem half_eq (w : Nat) : half w = 2 ^ (w - 1) := by
  unfold half
  split
  · subst_vars; rfl
  split
  · subst_vars; rfl
  split
  · subst_vars; rfl
  split
  · subst_vars; rfl
  split
  · subst_vars; rfl
  rfl

/-- `(2 ^ w : Int)`, a constant for the KIR widths. -/
@[inline] def pow2i (w : Nat) : Int :=
  if w = 64 then 18446744073709551616 else (pow2 w : Int)

theorem pow2i_eq (w : Nat) : pow2i w = ((2 ^ w : Nat) : Int) := by
  unfold pow2i
  split
  · subst_vars; rfl
  rw [pow2_eq]

/-- `n % 2 ^ w` (no division when `n < 2 ^ w`). -/
@[inline] def mod (w n : Nat) : Nat := if n < pow2 w then n else n % pow2 w

theorem mod_eq (w n : Nat) : mod w n = n % 2 ^ w := by
  unfold mod
  rw [pow2_eq]
  split
  · exact (Nat.mod_eq_of_lt ‹_›).symm
  · rfl

theorem mod_lt (w n : Nat) : mod w n < 2 ^ w := by
  rw [mod_eq]; exact Nat.mod_lt _ (Nat.two_pow_pos w)

@[inline] def ofNat (w n : Nat) : BitVec w := .ofNatLT (mod w n) (mod_lt w n)

theorem ofNat_eq (w n : Nat) : ofNat w n = BitVec.ofNat w n := by
  apply BitVec.eq_of_toNat_eq
  simp [ofNat, mod_eq]

@[inline] def add {w : Nat} (x y : BitVec w) : BitVec w := ofNat w (x.toNat + y.toNat)

theorem add_eq {w : Nat} (x y : BitVec w) : add x y = x + y := by
  rw [add, ofNat_eq]; rfl

theorem sub_lt {w : Nat} (x y : BitVec w) (h : ¬ y.toNat ≤ x.toNat) :
    pow2 w - (y.toNat - x.toNat) < 2 ^ w := by
  rw [pow2_eq]; have := y.isLt; omega

/-- `x - y`: `x - y` when `y ≤ x`, else `2 ^ w - (y - x)` (a number of `2 ^ w`'s size only when
the result is one). -/
@[inline] def sub {w : Nat} (x y : BitVec w) : BitVec w :=
  if h : y.toNat ≤ x.toNat then
    .ofNatLT (x.toNat - y.toNat) (Nat.lt_of_le_of_lt (Nat.sub_le _ _) x.isLt)
  else
    .ofNatLT (pow2 w - (y.toNat - x.toNat)) (sub_lt x y h)

theorem sub_eq {w : Nat} (x y : BitVec w) : sub x y = x - y := by
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_sub]
  unfold sub
  have hx := x.isLt
  have hy := y.isLt
  split
  · simp only [BitVec.toNat_ofNatLT]
    generalize 2 ^ w = P at hx hy ⊢
    have : P - y.toNat + x.toNat = (x.toNat - y.toNat) + P := by omega
    rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  · simp only [BitVec.toNat_ofNatLT, pow2_eq]
    generalize 2 ^ w = P at hx hy ⊢
    rw [Nat.mod_eq_of_lt (by omega)]
    omega

@[inline] def mul {w : Nat} (x y : BitVec w) : BitVec w := ofNat w (x.toNat * y.toNat)

theorem mul_eq {w : Nat} (x y : BitVec w) : mul x y = x * y := by
  rw [mul, ofNat_eq]; rfl

@[inline] def shl {w : Nat} (x : BitVec w) (s : Nat) : BitVec w := ofNat w (x.toNat <<< s)

theorem shl_eq {w : Nat} (x : BitVec w) (s : Nat) : shl x s = x <<< s := by
  rw [shl, ofNat_eq]; rfl

@[inline] def toInt {w : Nat} (x : BitVec w) : Int :=
  if 2 * x.toNat < pow2 w then x.toNat else (x.toNat : Int) - pow2i w

theorem toInt_eq {w : Nat} (x : BitVec w) : toInt x = x.toInt := by
  rw [toInt, BitVec.toInt_eq_toNat_cond, pow2_eq, pow2i_eq]

theorem ofInt_lt {w : Nat} (i : Int) (h : 0 ≤ i ∧ i < pow2i w) : i.toNat < 2 ^ w := by
  rw [pow2i_eq] at h; omega

theorem ofInt_neg_lt {w : Nat} (i : Int) (h' : -pow2i w ≤ i ∧ i < 0) :
    (pow2i w + i).toNat < 2 ^ w := by
  rw [pow2i_eq] at h' ⊢; omega

/-- `BitVec.ofInt w i`: `i` itself for `0 ≤ i < 2 ^ w`, `2 ^ w + i` for `-2 ^ w ≤ i < 0`. -/
@[inline] def ofInt (w : Nat) (i : Int) : BitVec w :=
  if h : 0 ≤ i ∧ i < pow2i w then .ofNatLT i.toNat (ofInt_lt i h)
  else if h' : -pow2i w ≤ i ∧ i < 0 then .ofNatLT (pow2i w + i).toNat (ofInt_neg_lt i h')
  else BitVec.ofInt w i

theorem ofInt_eq (w : Nat) (i : Int) : ofInt w i = BitVec.ofInt w i := by
  unfold ofInt
  split
  · rename_i h
    apply BitVec.eq_of_toNat_eq
    rw [BitVec.toNat_ofNatLT, BitVec.toNat_ofInt]
    rw [pow2i_eq] at h
    rw [Int.emod_eq_of_lt h.1 h.2]
  split
  · rename_i h h'
    apply BitVec.eq_of_toNat_eq
    rw [BitVec.toNat_ofNatLT, BitVec.toNat_ofInt]
    rw [pow2i_eq] at h h' ⊢
    have e : i % ((2 ^ w : Nat) : Int) = ((2 ^ w : Nat) : Int) + i := by
      rw [← Int.add_emod_right i ((2 ^ w : Nat) : Int), Int.add_comm i]
      exact Int.emod_eq_of_lt (by omega) (by omega)
    rw [e]
  rfl

/-- Truncation to `v` bits (`setWidth v` with `v < w`). -/
@[inline] def trunc {w : Nat} (v : Nat) (x : BitVec w) : BitVec v := ofNat v x.toNat

theorem trunc_eq {w : Nat} (v : Nat) (x : BitVec w) (h : v < w) : trunc v x = x.setWidth v := by
  rw [trunc, ofNat_eq]
  unfold BitVec.setWidth
  simp [show ¬ w ≤ v by omega]

@[inline] def signExtend {w : Nat} (v : Nat) (x : BitVec w) : BitVec v := ofInt v (toInt x)

theorem signExtend_eq {w : Nat} (v : Nat) (x : BitVec w) : signExtend v x = x.signExtend v := by
  rw [signExtend, ofInt_eq, toInt_eq]; rfl

@[inline] def sshiftRight {w : Nat} (x : BitVec w) (s : Nat) : BitVec w := ofInt w (toInt x >>> s)

theorem sshiftRight_eq {w : Nat} (x : BitVec w) (s : Nat) : sshiftRight x s = x.sshiftRight s := by
  rw [sshiftRight, ofInt_eq, toInt_eq]; rfl

@[inline] def msb {w : Nat} (x : BitVec w) : Bool := decide (half w ≤ x.toNat)

theorem msb_eq {w : Nat} (x : BitVec w) : msb x = x.msb := by
  rw [msb, half_eq, BitVec.msb_eq_decide]

@[inline] def slt {w : Nat} (x y : BitVec w) : Bool := toInt x < toInt y

theorem slt_eq {w : Nat} (x y : BitVec w) : slt x y = x.slt y := by
  simp [slt, toInt_eq, BitVec.slt]

@[inline] def sle {w : Nat} (x y : BitVec w) : Bool := toInt x ≤ toInt y

theorem sle_eq {w : Nat} (x y : BitVec w) : sle x y = x.sle y := by
  simp [sle, toInt_eq, BitVec.sle]

end KV.BV
