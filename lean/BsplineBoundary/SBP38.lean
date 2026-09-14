import Mathlib
import BsplineBoundary.Cert
import BsplineBoundary.Knots

/-!
# Exact rational certificates at degree 3 with 8 basis functions

This file certifies, as exact identities between explicit rational matrices, the finite
content of Section 3 of `paper/main.tex` at `p = 3`, `N = 8` on the unit interval with mesh
width `h = 1/5`.

All literals are produced by `gen_sbp.py` from the manuscript's own constructions in
`fractions.Fraction`: the clamped open-uniform knot vector and the Cox-de Boor recursion in
exact rational piecewise-monomial form, the true Greville abscissae, the sampled basis
matrix `B i j = B_j(x_i)`, the exact integrals `M i j = int B_i B_j` and
`K i j = int B_i B_j'`, the signed endpoint matrix
`Bd = -e_0 e_0^T + e_{N-1} e_{N-1}^T`, an exact `L D Lᵀ` factorization of `M`, exact
inverses of `B` and `L`, and the Marsden coefficient vectors of the monomials `x^k`,
`k = 0, ..., 3`. No floating point enters at any stage, and Lean re-verifies every product
from the literals.
-/

namespace BsplineBoundary
namespace P3N8

open Matrix

/-! ## The exact objects -/

/-- True Greville abscissae `x_i` at `p = 3`, `N = 8` (`paper/main.tex` 517-522). -/
def grev : Fin 8 → ℚ := ![(0 : ℚ), (1/15 : ℚ), (1/5 : ℚ), (2/5 : ℚ), (3/5 : ℚ), (4/5 : ℚ), (14/15 : ℚ), (1 : ℚ)]

/-- Sampled basis matrix `B i j = B_j(x_i)` (`paper/main.tex` 565-570). -/
def B : Matrix (Fin 8) (Fin 8) ℚ :=
  !![(1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (8/27 : ℚ), (61/108 : ℚ), (43/324 : ℚ), (1/162 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (1/4 : ℚ), (7/12 : ℚ), (1/6 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (1/6 : ℚ), (2/3 : ℚ), (1/6 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (1/6 : ℚ), (2/3 : ℚ), (1/6 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (1/6 : ℚ), (7/12 : ℚ), (1/4 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (1/162 : ℚ), (43/324 : ℚ), (61/108 : ℚ), (8/27 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (1 : ℚ)]

/-- Exact inverse of the sampled basis matrix. -/
def Binv : Matrix (Fin 8) (Fin 8) ℚ :=
  !![(1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (-632/1075 : ℚ), (2133/1075 : ℚ), (-519/1075 : ℚ), (118/1075 : ℚ), (-32/1075 : ℚ), (61/6450 : ℚ), (-9/2150 : ℚ), (4/3225 : ℚ);
    (176/645 : ℚ), (-198/215 : ℚ), (1342/645 : ℚ), (-118/215 : ℚ), (32/215 : ℚ), (-61/1290 : ℚ), (9/430 : ℚ), (-4/645 : ℚ);
    (-236/3225 : ℚ), (531/2150 : ℚ), (-3599/6450 : ℚ), (1888/1075 : ℚ), (-512/1075 : ℚ), (488/3225 : ℚ), (-72/1075 : ℚ), (64/3225 : ℚ);
    (64/3225 : ℚ), (-72/1075 : ℚ), (488/3225 : ℚ), (-512/1075 : ℚ), (1888/1075 : ℚ), (-3599/6450 : ℚ), (531/2150 : ℚ), (-236/3225 : ℚ);
    (-4/645 : ℚ), (9/430 : ℚ), (-61/1290 : ℚ), (32/215 : ℚ), (-118/215 : ℚ), (1342/645 : ℚ), (-198/215 : ℚ), (176/645 : ℚ);
    (4/3225 : ℚ), (-9/2150 : ℚ), (61/6450 : ℚ), (-32/1075 : ℚ), (118/1075 : ℚ), (-519/1075 : ℚ), (2133/1075 : ℚ), (-632/1075 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (1 : ℚ)]

/-- Exact spline mass form `M i j = int_0^1 B_i B_j dx` (`paper/main.tex` 578). -/
def M : Matrix (Fin 8) (Fin 8) ℚ :=
  !![(1/35 : ℚ), (7/400 : ℚ), (31/8400 : ℚ), (1/4200 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (7/400 : ℚ), (31/700 : ℚ), (1/32 : ℚ), (29/4200 : ℚ), (1/16800 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (31/8400 : ℚ), (1/32 : ℚ), (183/2800 : ℚ), (283/6300 : ℚ), (239/50400 : ℚ), (1/25200 : ℚ), (0 : ℚ), (0 : ℚ);
    (1/4200 : ℚ), (29/4200 : ℚ), (283/6300 : ℚ), (151/1575 : ℚ), (397/8400 : ℚ), (239/50400 : ℚ), (1/16800 : ℚ), (0 : ℚ);
    (0 : ℚ), (1/16800 : ℚ), (239/50400 : ℚ), (397/8400 : ℚ), (151/1575 : ℚ), (283/6300 : ℚ), (29/4200 : ℚ), (1/4200 : ℚ);
    (0 : ℚ), (0 : ℚ), (1/25200 : ℚ), (239/50400 : ℚ), (283/6300 : ℚ), (183/2800 : ℚ), (1/32 : ℚ), (31/8400 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (1/16800 : ℚ), (29/4200 : ℚ), (1/32 : ℚ), (31/700 : ℚ), (7/400 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (1/4200 : ℚ), (31/8400 : ℚ), (7/400 : ℚ), (1/35 : ℚ)]

/-- Exact coefficient advection form `K i j = int_0^1 B_i B_j' dx` (`paper/main.tex` 579). -/
def K : Matrix (Fin 8) (Fin 8) ℚ :=
  !![(-1/2 : ℚ), (31/80 : ℚ), (5/48 : ℚ), (1/120 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (-31/80 : ℚ), (0 : ℚ), (133/480 : ℚ), (13/120 : ℚ), (1/480 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (-5/48 : ℚ), (-133/480 : ℚ), (0 : ℚ), (109/360 : ℚ), (37/480 : ℚ), (1/720 : ℚ), (0 : ℚ), (0 : ℚ);
    (-1/120 : ℚ), (-13/120 : ℚ), (-109/360 : ℚ), (0 : ℚ), (49/144 : ℚ), (37/480 : ℚ), (1/480 : ℚ), (0 : ℚ);
    (0 : ℚ), (-1/480 : ℚ), (-37/480 : ℚ), (-49/144 : ℚ), (0 : ℚ), (109/360 : ℚ), (13/120 : ℚ), (1/120 : ℚ);
    (0 : ℚ), (0 : ℚ), (-1/720 : ℚ), (-37/480 : ℚ), (-109/360 : ℚ), (0 : ℚ), (133/480 : ℚ), (5/48 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (-1/480 : ℚ), (-13/120 : ℚ), (-133/480 : ℚ), (0 : ℚ), (31/80 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (-1/120 : ℚ), (-5/48 : ℚ), (-31/80 : ℚ), (1/2 : ℚ)]

/-- Signed wall-trace endpoint matrix `-e_0 e_0^T + e_{N-1} e_{N-1}^T` (`paper/main.tex` 592). -/
def Bd : Matrix (Fin 8) (Fin 8) ℚ :=
  !![(-1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (1 : ℚ)]

/-- Unit lower-triangular factor of the exact `L D Lᵀ` factorization of `M`. -/
def L : Matrix (Fin 8) (Fin 8) ℚ :=
  !![(1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (49/80 : ℚ), (1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (31/240 : ℚ), (19481/22557 : ℚ), (1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (1/120 : ℚ), (1514/7519 : ℚ), (7399647/7549604 : ℚ), (1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (40/22557 : ℚ), (222195/1887401 : ℚ), (8114626188/10698355585 : ℚ), (1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (7519/7549604 : ℚ), (894778031/10698355585 : ℚ), (2786883667747/4244097105281 : ℚ), (1 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (11324406/10698355585 : ℚ), (462335483127/4244097105281 : ℚ), (15131994783533/21397144255303 : ℚ), (1 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (32095066755/8488194210562 : ℚ), (5999678234677/64191432765909 : ℚ), (64083317926099/105391541257626 : ℚ), (1 : ℚ)]

/-- Exact inverse of `L`. -/
def Linv : Matrix (Fin 8) (Fin 8) ℚ :=
  !![(1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (-49/80 : ℚ), (1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (18037/45114 : ℚ), (-19481/22557 : ℚ), (1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (-12541573/45297624 : ℚ), (4870427/7549604 : ℚ), (-7399647/7549604 : ℚ), (1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (10528637341/64190133510 : ℚ), (-12498562582/32095066755 : ℚ), (6693978234/10698355585 : ℚ), (-8114626188/10698355585 : ℚ), (1 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (-1442092743545/16976388421124 : ℚ), (156351329721/771654019142 : ℚ), (-2800143308987/8488194210562 : ℚ), (1758867911285/4244097105281 : ℚ), (-2786883667747/4244097105281 : ℚ), (1 : ℚ), (0 : ℚ), (0 : ℚ);
    (1818727918645/42794288510606 : ℚ), (-2172917674213/21397144255303 : ℚ), (3555583286587/21397144255303 : ℚ), (-4525767006143/21397144255303 : ℚ), (7605492828070/21397144255303 : ℚ), (-15131994783533/21397144255303 : ℚ), (1 : ℚ), (0 : ℚ);
    (-3904186375373/210783082515252 : ℚ), (179502832376/4053520817601 : ℚ), (-3824294404807/52695770628813 : ℚ), (9774396400861/105391541257626 : ℚ), (-16708249754039/105391541257626 : ℚ), (35469062823511/105391541257626 : ℚ), (-64083317926099/105391541257626 : ℚ), (1 : ℚ)]

/-- Exact pivots of the `L D Lᵀ` factorization of `M`. -/
def D : Fin 8 → ℚ := ![(1/35 : ℚ), (7519/224000 : ℚ), (1887401/47369700 : ℚ), (2139671117/38050004160 : ℚ), (4244097105281/67399640185500 : ℚ), (64191432765909/1697638842112400 : ℚ), (52695770628813/2139714425530300 : ℚ), (16133187252109/843132330061008 : ℚ)]

/-- The exact product `L * diagonal D`. -/
def LD : Matrix (Fin 8) (Fin 8) ℚ :=
  !![(1/35 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (7/400 : ℚ), (7519/224000 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (31/8400 : ℚ), (2783/96000 : ℚ), (1887401/47369700 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (1/4200 : ℚ), (757/112000 : ℚ), (822183/21053200 : ℚ), (2139671117/38050004160 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (1/16800 : ℚ), (14813/3157980 : ℚ), (225406283/5284722800 : ℚ), (4244097105281/67399640185500 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (1/25200 : ℚ), (127825433/27178574400 : ℚ), (2786883667747/67399640185500 : ℚ), (64191432765909/1697638842112400 : ℚ), (0 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (1/16800 : ℚ), (22015975387/3209506675500 : ℚ), (45395984350599/1697638842112400 : ℚ), (52695770628813/2139714425530300 : ℚ), (0 : ℚ);
    (0 : ℚ), (0 : ℚ), (0 : ℚ), (0 : ℚ), (1/4200 : ℚ), (5999678234677/1697638842112400 : ℚ), (64083317926099/4279428851060600 : ℚ), (16133187252109/843132330061008 : ℚ)]

/-- Marsden coefficients of `x^0`: `x^0 = sum_j alpha0_j B_j(x)`. -/
def alpha0 : Fin 8 → ℚ := ![(1 : ℚ), (1 : ℚ), (1 : ℚ), (1 : ℚ), (1 : ℚ), (1 : ℚ), (1 : ℚ), (1 : ℚ)]

/-- Marsden coefficients of `x^1`: `x^1 = sum_j alpha1_j B_j(x)`. -/
def alpha1 : Fin 8 → ℚ := ![(0 : ℚ), (1/15 : ℚ), (1/5 : ℚ), (2/5 : ℚ), (3/5 : ℚ), (4/5 : ℚ), (14/15 : ℚ), (1 : ℚ)]

/-- Marsden coefficients of `x^2`: `x^2 = sum_j alpha2_j B_j(x)`. -/
def alpha2 : Fin 8 → ℚ := ![(0 : ℚ), (0 : ℚ), (2/75 : ℚ), (11/75 : ℚ), (26/75 : ℚ), (47/75 : ℚ), (13/15 : ℚ), (1 : ℚ)]

/-- Marsden coefficients of `x^3`: `x^3 = sum_j alpha3_j B_j(x)`. -/
def alpha3 : Fin 8 → ℚ := ![(0 : ℚ), (0 : ℚ), (0 : ℚ), (6/125 : ℚ), (24/125 : ℚ), (12/25 : ℚ), (4/5 : ℚ), (1 : ℚ)]


/-! ## The Greville node set

The literal `grev` agrees with the general definition `BsplineBoundary.greville` of
`Knots.lean`, so the node placement theorems proved there for general `p` and `N` apply to
the matrices of this file.
-/

theorem grev_eq_greville (i : Fin 8) : grev i = greville 3 8 (i : ℕ) := by
  fin_cases i <;>
    norm_num [grev, greville_eq_range, knot, Finset.sum_range_succ]


/-! ## Endpoint traces of the clamped basis

The first and last rows of the sampled basis matrix are the standard basis covectors,
which is the numerical instance at `p = 3`, `N = 8` of the rank-two endpoint trace
`B_0(0) = 1`, `B_j(0) = 0` for `j > 0`, `B_{N-1}(1) = 1`, `B_j(1) = 0` for `j < N-1`
(`paper/main.tex` 507-516), using `x_0 = 0` and `x_{N-1} = 1`.
-/

/-- `B_0(x_0) = 1` and `B_j(x_0) = 0` for `j > 0`, with `x_0 = 0`. -/
theorem B_wall_row_left (j : Fin 8) : B 0 j = if j = 0 then 1 else 0 := by
  fin_cases j <;> simp [B]

/-- `B_{N-1}(x_{N-1}) = 1` and `B_j(x_{N-1}) = 0` for `j < N-1`, with `x_{N-1} = 1`. -/
theorem B_wall_row_right (j : Fin 8) : B 7 j = if j = 7 then 1 else 0 := by
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
All three parts are certified below at `p = 3`, `N = 8`. This is the identity that
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

/-- **The spline mass form is positive definite** at `p = 3`, `N = 8`. -/
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

For `k = 0, ..., 3` the Marsden coefficient vector `alpha^(k)_j = e_k(t_{j+1},...,t_{j+p}) / C(p,k)`
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

set_option maxHeartbeats 4000000 in
theorem poly_reproduction_3 : B *ᵥ alpha3 = (fun i => grev i ^ 3) := by
  funext i
  fin_cases i <;>
    norm_num [B, alpha3, grev, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]

/-- At `k = 1` the Marsden coefficients are exactly the Greville abscissae
(`paper/main.tex` 517-522). -/
theorem alpha1_eq_grev : alpha1 = grev := by
  funext i
  fin_cases i <;> norm_num [alpha1, grev]


end P3N8
end BsplineBoundary
