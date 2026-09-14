import Mathlib

/-!
# Clamped open-uniform knots and true Greville abscissae

This file formalizes the knot vector and node set of Section 3 of `paper/main.tex`
(lines 485-530) for general degree `p` and general basis count `N`, and proves the three
node placement statements the manuscript records: the two endpoints sit exactly on the
walls, and the interior Greville spacing is the uniform mesh width `1 / (N - p)`.

For degree `p` and `N` basis functions the clamped open-uniform knot vector is
`t_0 = ... = t_p = 0`, `t_N = ... = t_{N+p} = 1`, `t_{p+j} = j / (N - p)` for
`j = 1, ..., N - p - 1`, and the true Greville abscissae are
`x_i = (1/p) * sum_{m=1}^{p} t_{i+m}` for `i = 0, ..., N - 1`.

Nothing analytic enters: these are finite rational sums, and the proofs are finite-sum
algebra valid for every `p >= 1` and every `N >= p + 1`.
-/

namespace BsplineBoundary

/-- The clamped open-uniform degree-`p` knot vector on `[0,1]` with `N` basis functions
(`paper/main.tex`, equations at lines 492-500). Indices `k = 0, ..., N + p`; the value
outside that range is irrelevant to every statement below. -/
def knot (p N k : ℕ) : ℚ :=
  if k ≤ p then 0 else if N ≤ k then 1 else ((k : ℚ) - p) / ((N : ℚ) - p)

/-- The true Greville abscissae of the clamped knot vector
(`paper/main.tex`, equation at lines 517-522): `x_i = (1/p) * sum_{m=1}^{p} t_{i+m}`. -/
def greville (p N i : ℕ) : ℚ :=
  (1 / (p : ℚ)) * ∑ m ∈ Finset.Icc 1 p, knot p N (i + m)

/-! ### Knot values -/

theorem knot_of_le {p N k : ℕ} (h : k ≤ p) : knot p N k = 0 := by
  simp [knot, h]

theorem knot_of_ge {p N k : ℕ} (hpN : p < N) (h : N ≤ k) : knot p N k = 1 := by
  have hk : ¬ k ≤ p := by omega
  simp [knot, hk, h]

/-- On the whole clamped range `p ≤ k ≤ N` the knot value is the single affine formula
`(k - p) / (N - p)`; the two clamped ends `k = p` and `k = N` agree with it. -/
theorem knot_eq {p N k : ℕ} (hpN : p < N) (h1 : p ≤ k) (h2 : k ≤ N) :
    knot p N k = ((k : ℚ) - p) / ((N : ℚ) - p) := by
  have hne : (N : ℚ) - p ≠ 0 := by
    have : (p : ℚ) < N := by exact_mod_cast hpN
    linarith
  unfold knot
  split_ifs with hkp hkN
  · have : k = p := le_antisymm hkp h1
    subst this; simp
  · have : k = N := le_antisymm h2 hkN
    subst this; field_simp
  · rfl

/-! ### The Greville sum over `Finset.range` -/

/-- Reindex a sum over `Finset.Icc 1 p` as a sum over `Finset.range p`. -/
theorem sum_Icc_one_shift (p : ℕ) (f : ℕ → ℚ) :
    ∑ m ∈ Finset.Icc 1 p, f m = ∑ m ∈ Finset.range p, f (m + 1) := by
  induction p with
  | zero => simp
  | succ n ih =>
      rw [Finset.sum_Icc_succ_top (by omega), ih, Finset.sum_range_succ]

/-- The Greville abscissa rewritten over `Finset.range p`, the form used for telescoping. -/
theorem greville_eq_range (p N i : ℕ) :
    greville p N i = (1 / (p : ℚ)) * ∑ m ∈ Finset.range p, knot p N (i + m + 1) := by
  unfold greville
  rw [sum_Icc_one_shift p (fun m => knot p N (i + m))]
  simp only [Nat.add_assoc]

/-! ### The three node placement statements -/

/-- `x_0 = 0`: the clamped knot repetition puts the first Greville node exactly on the
left wall (`paper/main.tex`, line 524). -/
theorem greville_zero (p N : ℕ) : greville p N 0 = 0 := by
  rw [greville_eq_range]
  have : ∀ m ∈ Finset.range p, knot p N (0 + m + 1) = 0 := by
    intro m hm
    exact knot_of_le (by simp at hm ⊢; omega)
  rw [Finset.sum_congr rfl this]
  simp

/-- `x_{N-1} = 1`: the last Greville node sits exactly on the right wall
(`paper/main.tex`, line 524). -/
theorem greville_last {p N : ℕ} (hp : 0 < p) (hpN : p < N) :
    greville p N (N - 1) = 1 := by
  rw [greville_eq_range]
  have h : ∀ m ∈ Finset.range p, knot p N (N - 1 + m + 1) = 1 := by
    intro m _
    exact knot_of_ge hpN (by omega)
  rw [Finset.sum_congr rfl h, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one]
  have : (p : ℚ) ≠ 0 := Nat.cast_ne_zero.mpr hp.ne'
  field_simp

/-- Uniform interior Greville spacing: for `p - 1 ≤ i ≤ N - p - 1` (written without
natural subtraction as `p ≤ i + 1` and `i + p + 1 ≤ N`) consecutive Greville abscissae
differ by exactly the mesh width `h = 1 / (N - p)`.

The proof is the telescoping `x_{i+1} - x_i = (t_{i+p+1} - t_{i+1}) / p` followed by the
affine knot formula at both ends. -/
theorem greville_succ_sub {p N i : ℕ} (hp : 0 < p) (hpN : p < N)
    (h1 : p ≤ i + 1) (h2 : i + p + 1 ≤ N) :
    greville p N (i + 1) - greville p N i = 1 / ((N : ℚ) - p) := by
  have hpQ : (p : ℚ) ≠ 0 := Nat.cast_ne_zero.mpr hp.ne'
  have hne : (N : ℚ) - p ≠ 0 := by
    have : (p : ℚ) < N := by exact_mod_cast hpN
    linarith
  set g : ℕ → ℚ := fun m => knot p N (i + m + 1) with hg
  have htel : (∑ m ∈ Finset.range p, g (m + 1)) - ∑ m ∈ Finset.range p, g m
      = g p - g 0 := by
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_range_sub g p
  have hstep : greville p N (i + 1) - greville p N i
      = (1 / (p : ℚ)) * (g p - g 0) := by
    rw [greville_eq_range, greville_eq_range, ← mul_sub, ← htel]
    congr 2
    exact Finset.sum_congr rfl fun m _ => by rw [hg]; congr 2; omega
  have hgp : g p = ((i : ℚ) + 1) / ((N : ℚ) - p) := by
    show knot p N (i + p + 1) = _
    rw [knot_eq hpN (by omega) (by omega)]
    congr 1
    push_cast
    ring
  have hg0 : g 0 = ((i : ℚ) + 1 - p) / ((N : ℚ) - p) := by
    show knot p N (i + 0 + 1) = _
    rw [knot_eq hpN (by omega) (by omega)]
    congr 1
    push_cast
    ring
  rw [hstep, hgp, hg0, div_sub_div_same]
  field_simp
  ring

end BsplineBoundary
