import Mathlib

/-!
# Generic certificate lemmas for exact rational matrices

The finite cases of Section 3 are certified by exhibiting exact rational witnesses and
letting Lean verify a product identity. This file holds the two reusable bridges from a
verified product to the mathematical conclusion:

* `posDef_of_ldl`: a symmetric matrix with an exact `L D Lᵀ` factorization, unit-triangular
  `L` certified nonsingular by an explicit exact inverse and strictly positive pivots `d`,
  is positive definite;
* `det_ne_zero_of_mul_eq_one`: an exact right inverse certifies a nonzero determinant.

Both are proved for `ℚ` with `Fin n` indices, the setting of the case files.
-/

namespace BsplineBoundary

open Matrix

/-- An exact right inverse certifies nonsingularity. -/
theorem det_ne_zero_of_mul_eq_one {n : ℕ} {A Ainv : Matrix (Fin n) (Fin n) ℚ}
    (h : A * Ainv = 1) : A.det ≠ 0 := by
  intro hdet
  have : A.det * Ainv.det = 1 := by rw [← Matrix.det_mul, h, Matrix.det_one]
  rw [hdet, zero_mul] at this
  exact zero_ne_one this

/-- An exact right inverse certifies that the determinant is a unit. -/
theorem isUnit_det_of_mul_eq_one {n : ℕ} {A Ainv : Matrix (Fin n) (Fin n) ℚ}
    (h : A * Ainv = 1) : IsUnit A.det :=
  isUnit_iff_ne_zero.mpr (det_ne_zero_of_mul_eq_one h)

/-- A matrix with an exact `L D Lᵀ` factorization is symmetric. -/
theorem transpose_eq_of_ldl {n : ℕ} {M L : Matrix (Fin n) (Fin n) ℚ} {d : Fin n → ℚ}
    (hM : L * Matrix.diagonal d * Lᵀ = M) : Mᵀ = M := by
  rw [← hM, Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_transpose,
    Matrix.diagonal_transpose, Matrix.mul_assoc]

/-- **Exact LDL certificate of positive definiteness.** If `L * diagonal d * Lᵀ = M` with
every pivot `d i` strictly positive and `L` nonsingular (certified by an exact right
inverse `Linv`), then `M` is positive definite. -/
theorem posDef_of_ldl {n : ℕ} {M L Linv : Matrix (Fin n) (Fin n) ℚ} {d : Fin n → ℚ}
    (hM : L * Matrix.diagonal d * Lᵀ = M) (hd : ∀ i, 0 < d i) (hinv : L * Linv = 1) :
    M.PosDef := by
  have hT : Mᵀ = M := transpose_eq_of_ldl hM
  have hherm : M.IsHermitian := by
    ext i j
    rw [Matrix.conjTranspose_apply, star_trivial]
    exact congrFun (congrFun hT i) j
  rw [Matrix.posDef_iff_dotProduct_mulVec]
  refine ⟨hherm, ?_⟩
  intro x hx
  -- `Linvᵀ` is a left inverse of `Lᵀ`, so `y = Lᵀ *ᵥ x` is nonzero
  have hTinv : Linvᵀ * Lᵀ = 1 := by
    rw [← Matrix.transpose_mul, hinv, Matrix.transpose_one]
  have hy : Lᵀ *ᵥ x ≠ 0 := by
    intro h0
    apply hx
    calc x = (1 : Matrix (Fin n) (Fin n) ℚ) *ᵥ x := by rw [Matrix.one_mulVec]
    _ = (Linvᵀ * Lᵀ) *ᵥ x := by rw [hTinv]
    _ = Linvᵀ *ᵥ (Lᵀ *ᵥ x) := by rw [Matrix.mulVec_mulVec]
    _ = 0 := by rw [h0, Matrix.mulVec_zero]
  -- expand the quadratic form in the factored coordinates
  have hexp : star x ⬝ᵥ (M *ᵥ x) = ∑ i, d i * (Lᵀ *ᵥ x) i ^ 2 := by
    rw [← hM, Matrix.mul_assoc]
    rw [← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec,
      ← Matrix.mulVec_transpose]
    simp only [star_trivial, dotProduct, Matrix.mulVec_diagonal]
    exact Finset.sum_congr rfl fun i _ => by ring
  rw [hexp]
  obtain ⟨k, hk⟩ : ∃ k, (Lᵀ *ᵥ x) k ≠ 0 := by
    by_contra hcon
    exact hy (funext fun i => not_not.mp fun h => hcon ⟨i, h⟩)
  refine Finset.sum_pos' (fun i _ => ?_) ⟨k, Finset.mem_univ k, ?_⟩
  · exact mul_nonneg (hd i).le (sq_nonneg _)
  · exact mul_pos (hd k) (by positivity)

end BsplineBoundary
