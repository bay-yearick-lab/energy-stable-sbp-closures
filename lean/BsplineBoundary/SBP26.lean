import Mathlib
import BsplineBoundary.Cert
import BsplineBoundary.Knots

/-!
# Exact rational certificates at degree 2 with 6 basis functions

This file certifies, as exact identities between explicit rational matrices, the finite
content of Section 3 of `paper/main.tex` at `p = 2`, `N = 6` on the unit interval with mesh
width `h = 1/4`.

All literals are produced by `gen_sbp.py` from the manuscript's own constructions in
`fractions.Fraction`: the clamped open-uniform knot vector and the Cox-de Boor recursion in
exact rational piecewise-monomial form, the true Greville abscissae, the sampled basis
matrix `B i j = B_j(x_i)`, the exact integrals `M i j = int B_i B_j` and
`K i j = int B_i B_j'`, the signed endpoint matrix
`Bd = -e_0 e_0^T + e_{N-1} e_{N-1}^T`, an exact `L D Lᵀ` factorization of `M`, exact
inverses of `B` and `L`, and the Marsden coefficient vectors of the monomials `x^k`,
`k = 0, ..., 2`. No floating point enters at any stage, and Lean re-verifies every product
from the literals.
-/

namespace BsplineBoundary
namespace P2N6

open Matrix

/-! ## The exact objects -/

/-- True Greville abscissae `x_i` at `p = 2`, `N = 6` (`paper/main.tex` 517-522). -/
def grev : Fin 6 → ℚ := ![(0 : ℚ), (1/8 : ℚ), (3/8 : ℚ), (5/8 : ℚ), (7/8 : ℚ), (1 : ℚ)]

/-- Sampled basis matrix `B i j = B_j(x_i)` (`paper/main.tex` 565-570). -/
def B : Matrix (Fin 6) (Fin 6) ℚ :=
  !![(1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (1/4 : ℚ), (5/8 : ℚ), (1/8 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (1/8 : ℚ), (3/4 : ℚ), (1/8 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (1/8 : ℚ), (3/4 : ℚ), (1/8 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (1/8 : ℚ), (5/8 : ℚ), (1/4 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (1 : ℚ)]

/-- Exact inverse of the sampled basis matrix. -/
def Binv : Matrix (Fin 6) (Fin 6) ℚ :=
  !![(1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (-169/408 : ℚ), (169/102 : ℚ), (-29/102 : ℚ), (5/102 : ℚ), (-1/102 : ℚ), (1/408 : ℚ);
    (29/408 : ℚ), (-29/102 : ℚ), (145/102 : ℚ), (-25/102 : ℚ), (5/102 : ℚ), (-5/408 : ℚ);
    (-5/408 : ℚ), (5/102 : ℚ), (-25/102 : ℚ), (145/102 : ℚ), (-29/102 : ℚ), (29/408 : ℚ);
    (1/408 : ℚ), (-1/102 : ℚ), (5/102 : ℚ), (-29/102 : ℚ), (169/102 : ℚ), (-169/408 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (1 : ℚ)]

/-- Exact spline mass form `M i j = int_0^1 B_i B_j dx` (`paper/main.tex` 578). -/
def M : Matrix (Fin 6) (Fin 6) ℚ :=
  !![(1/20 : ℚ), (7/240 : ℚ), (1/240 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (7/240 : ℚ), (1/12 : ℚ), (5/96 : ℚ), (1/480 : ℚ), (0 : ℚ), (0 : ℚ);
    (1/240 : ℚ), (5/96 : ℚ), (11/80 : ℚ), (13/240 : ℚ), (1/480 : ℚ), (0 : ℚ);
    (0 : ℚ), (1/480 : ℚ), (13/240 : ℚ), (11/80 : ℚ), (5/96 : ℚ), (1/240 : ℚ);
    (0 : ℚ), (0 : ℚ), (1/480 : ℚ), (5/96 : ℚ), (1/12 : ℚ), (7/240 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (1/240 : ℚ), (7/240 : ℚ), (1/20 : ℚ)]

/-- Exact coefficient advection form `K i j = int_0^1 B_i B_j' dx` (`paper/main.tex` 579). -/
def K : Matrix (Fin 6) (Fin 6) ℚ :=
  !![(-1/2 : ℚ), (5/12 : ℚ), (1/12 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (-5/12 : ℚ), (0 : ℚ), (3/8 : ℚ), (1/24 : ℚ), (0 : ℚ), (0 : ℚ);
    (-1/12 : ℚ), (-3/8 : ℚ), (0 : ℚ), (5/12 : ℚ), (1/24 : ℚ), (0 : ℚ);
    (0 : ℚ), (-1/24 : ℚ), (-5/12 : ℚ), (0 : ℚ), (3/8 : ℚ), (1/12 : ℚ);
    (0 : ℚ), (0 : ℚ), (-1/24 : ℚ), (-3/8 : ℚ), (0 : ℚ), (5/12 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (-1/12 : ℚ), (-5/12 : ℚ), (1/2 : ℚ)]

/-- Signed wall-trace endpoint matrix `-e_0 e_0^T + e_{N-1} e_{N-1}^T` (`paper/main.tex` 592). -/
def Bd : Matrix (Fin 6) (Fin 6) ℚ :=
  !![(-1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (1 : ℚ)]

/-- Unit lower-triangular factor of the exact `L D Lᵀ` factorization of `M`. -/
def L : Matrix (Fin 6) (Fin 6) ℚ :=
  !![(1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (7/12 : ℚ), (1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (1/12 : ℚ), (143/191 : ℚ), (1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (6/191 : ℚ), (4823/9166 : ℚ), (1 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (191/9166 : ℚ), (224327/482881 : ℚ), (1 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (18332/482881 : ℚ), (4931/10793 : ℚ), (1 : ℚ)]

/-- Exact inverse of `L`. -/
def Linv : Matrix (Fin 6) (Fin 6) ℚ :=
  !![(1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (-7/12 : ℚ), (1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (135/382 : ℚ), (-143/191 : ℚ), (1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (-3073/18332 : ℚ), (3323/9166 : ℚ), (-4823/9166 : ℚ), (1 : ℚ), (0 : ℚ), (0 : ℚ);
    (4864/68983 : ℚ), (-73793/482881 : ℚ), (15425/68983 : ℚ), (-224327/482881 : ℚ), (1 : ℚ), (0 : ℚ);
    (-279/10793 : ℚ), (605/10793 : ℚ), (-887/10793 : ℚ), (1881/10793 : ℚ), (-4931/10793 : ℚ), (1 : ℚ)]

/-- Exact pivots of the `L D Lᵀ` factorization of `M`. -/
def D : Fin 6 → ℚ := ![(1/20 : ℚ), (191/2880 : ℚ), (4583/45840 : ℚ), (482881/4399680 : ℚ), (86344/1448643 : ℚ), (1211/32379 : ℚ)]

/-- The exact product `L * diagonal D`. -/
def LD : Matrix (Fin 6) (Fin 6) ℚ :=
  !![(1/20 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (7/240 : ℚ), (191/2880 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (1/240 : ℚ), (143/2880 : ℚ), (4583/45840 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (1/480 : ℚ), (4823/91680 : ℚ), (482881/4399680 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (1/480 : ℚ), (224327/4399680 : ℚ), (86344/1448643 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (1/240 : ℚ), (39448/1448643 : ℚ), (1211/32379 : ℚ)]

/-- Marsden coefficients of `x^0`: `x^0 = sum_j alpha0_j B_j(x)`. -/
def alpha0 : Fin 6 → ℚ := ![(1 : ℚ), (1 : ℚ), (1 : ℚ), (1 : ℚ), (1 : ℚ), (1 : ℚ)]

/-- Marsden coefficients of `x^1`: `x^1 = sum_j alpha1_j B_j(x)`. -/
def alpha1 : Fin 6 → ℚ := ![(0 : ℚ), (1/8 : ℚ), (3/8 : ℚ), (5/8 : ℚ), (7/8 : ℚ), (1 : ℚ)]

/-- Marsden coefficients of `x^2`: `x^2 = sum_j alpha2_j B_j(x)`. -/
def alpha2 : Fin 6 → ℚ := ![(0 : ℚ), (0 : ℚ), (1/8 : ℚ), (3/8 : ℚ), (3/4 : ℚ), (1 : ℚ)]


/-! ## The Greville node set

The literal `grev` agrees with the general definition `BsplineBoundary.greville` of
`Knots.lean`, so the node placement theorems proved there for general `p` and `N` apply to
the matrices of this file.
-/

theorem grev_eq_greville (i : Fin 6) : grev i = greville 2 6 (i : ℕ) := by
  fin_cases i <;>
    norm_num [grev, greville_eq_range, knot, Finset.sum_range_succ]


/-! ## Endpoint traces of the clamped basis

The first and last rows of the sampled basis matrix are the standard basis covectors,
which is the numerical instance at `p = 2`, `N = 6` of the rank-two endpoint trace
`B_0(0) = 1`, `B_j(0) = 0` for `j > 0`, `B_{N-1}(1) = 1`, `B_j(1) = 0` for `j < N-1`
(`paper/main.tex` 507-516), using `x_0 = 0` and `x_{N-1} = 1`.
-/

/-- `B_0(x_0) = 1` and `B_j(x_0) = 0` for `j > 0`, with `x_0 = 0`. -/
theorem B_wall_row_left (j : Fin 6) : B 0 j = if j = 0 then 1 else 0 := by
  fin_cases j <;> simp [B]

/-- `B_{N-1}(x_{N-1}) = 1` and `B_j(x_{N-1}) = 0` for `j < N-1`, with `x_{N-1} = 1`. -/
theorem B_wall_row_right (j : Fin 6) : B 5 j = if j = 5 then 1 else 0 := by
  fin_cases j <;> simp [B]

set_option maxHeartbeats 4000000 in
/-- Partition of unity at the Greville nodes: `B 1 = 1`. -/
theorem B_mulVec_one : B *ᵥ (fun _ => (1 : ℚ)) = (fun _ => (1 : ℚ)) := by
  funext i
  fin_cases i <;>
    norm_num [B, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]


/-! ## The coefficient-space Green identity

`paper/main.tex` equation `eq:spline-construction-coeff-green` (lines 588-597) and
Proposition `prop:bmf-coeff-d1` state
`K + K^T = -e_0 e_0^T + e_{N-1} e_{N-1}^T = B^T Bd B` together with `K 1 = 0`.
All three parts are certified below at `p = 2`, `N = 6`. This is the identity that
Theorem `thm:bmg-first-derivative` (lines 1400-1430) turns by congruence into the BMG
natural-metric SBP identity `H D + D^T H = Bd`.
-/

set_option maxHeartbeats 4000000 in
/-- `K + K^T = -e_0 e_0^T + e_{N-1} e_{N-1}^T`, integration by parts in coefficient space. -/
theorem K_add_transpose : K + Kᵀ = Bd := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [K, Bd, Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_succ]

set_option maxHeartbeats 4000000 in
theorem Bd_mul_B : Bd * B = Bd := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [B, Bd, Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_succ]

set_option maxHeartbeats 4000000 in
theorem Bt_mul_Bd : Bᵀ * Bd = Bd := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [B, Bd, Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_succ]

/-- The congruence form of the endpoint matrix: `B^T Bd B = Bd`, a consequence of the
two wall rows of `B` being the standard basis covectors. -/
theorem sample_congruence : Bᵀ * Bd * B = Bd := by
  rw [Matrix.mul_assoc, Bd_mul_B, Bt_mul_Bd]

/-- **The coefficient-space Green identity in the printed congruence form**
(`paper/main.tex`, `eq:spline-construction-coeff-green`). -/
theorem green_identity : K + Kᵀ = Bᵀ * Bd * B := by
  rw [sample_congruence, K_add_transpose]

set_option maxHeartbeats 4000000 in
/-- `K 1 = 0`: the advection form annihilates constants. -/
theorem K_mulVec_one : K *ᵥ (fun _ => (1 : ℚ)) = (fun _ => (0 : ℚ)) := by
  funext i
  fin_cases i <;>
    norm_num [K, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]


/-! ## The mass form is symmetric positive definite

The exact `L D Lᵀ` factorization with strictly positive pivots, together with the exact
inverse of `L`, certifies positive definiteness through `BsplineBoundary.posDef_of_ldl`.
-/

set_option maxHeartbeats 4000000 in
theorem M_transpose : Mᵀ = M := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [M, Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_succ]

set_option maxHeartbeats 4000000 in
/-- `L * diagonal D` as an explicit rational literal. -/
theorem L_mul_D : L * Matrix.diagonal D = LD := by
  ext i j
  rw [Matrix.mul_diagonal]
  fin_cases i <;> fin_cases j <;> norm_num [L, D, LD]

set_option maxHeartbeats 4000000 in
/-- **The exact `L D Lᵀ` factorization of the spline mass form.** -/
theorem M_ldl : L * Matrix.diagonal D * Lᵀ = M := by
  rw [L_mul_D]
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [LD, L, M, Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_succ]

set_option maxHeartbeats 4000000 in
theorem L_mul_Linv : L * Linv = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [L, Linv, Matrix.one_apply, Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_succ]

theorem D_pos : ∀ i, 0 < D i := by
  intro i
  fin_cases i <;> norm_num [D]

/-- **The spline mass form is positive definite** at `p = 2`, `N = 6`. -/
theorem M_posDef : M.PosDef :=
  posDef_of_ldl M_ldl D_pos L_mul_Linv


/-! ## The sampled basis matrix is nonsingular

`paper/main.tex` lines 530-545 records that the Greville sample matrix is nonsingular by
total positivity and Schoenberg-Whitney theory. At this fixed case the fact is certified
directly by an exact rational inverse.
-/

set_option maxHeartbeats 4000000 in
theorem B_mul_Binv : B * Binv = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [B, Binv, Matrix.one_apply, Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_succ]

/-- `det B != 0`: the Greville sample matrix is nonsingular. -/
theorem B_det_ne_zero : B.det ≠ 0 :=
  det_ne_zero_of_mul_eq_one B_mul_Binv

theorem B_isUnit_det : IsUnit B.det :=
  isUnit_det_of_mul_eq_one B_mul_Binv


/-! ## Polynomial reproduction through the Marsden coefficients

For `k = 0, ..., 2` the Marsden coefficient vector `alpha^(k)_j = e_k(t_{j+1},...,t_{j+p}) / C(p,k)`
expands the monomial `x^k` in the clamped basis, so `B alpha^(k)` is the vector of Greville
samples of `x^k`. This is the finite instance of the `P_p` exactness used in
Theorem `thm:bmg-first-derivative` (`eq:bmg-polynomial-exactness`). At `k = 0` it is the
partition of unity and at `k = 1` the coefficient vector is the Greville node vector itself.
-/

set_option maxHeartbeats 4000000 in
theorem poly_reproduction_0 : B *ᵥ alpha0 = (fun i => grev i ^ 0) := by
  funext i
  fin_cases i <;>
    norm_num [B, alpha0, grev, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

set_option maxHeartbeats 4000000 in
theorem poly_reproduction_1 : B *ᵥ alpha1 = (fun i => grev i ^ 1) := by
  funext i
  fin_cases i <;>
    norm_num [B, alpha1, grev, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

set_option maxHeartbeats 4000000 in
theorem poly_reproduction_2 : B *ᵥ alpha2 = (fun i => grev i ^ 2) := by
  funext i
  fin_cases i <;>
    norm_num [B, alpha2, grev, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

/-- At `k = 1` the Marsden coefficients are exactly the Greville abscissae
(`paper/main.tex` 517-522). -/
theorem alpha1_eq_grev : alpha1 = grev := by
  funext i
  fin_cases i <;> norm_num [alpha1, grev]


end P2N6
end BsplineBoundary
