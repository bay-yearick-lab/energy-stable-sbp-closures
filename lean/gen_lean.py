"""Emit the exact rational Lean certificate files from `gen_sbp.py`.

Run from `lean/`:  python3 gen_lean.py
Writes BsplineBoundary/SBP26.lean (p = 2, N = 6) and BsplineBoundary/SBP38.lean (p = 3, N = 8).
Every matrix written here has already been checked by `gen_sbp.verify` in exact rational
arithmetic; Lean re-verifies each product identity from the literals.
"""
from fractions import Fraction as F
import gen_sbp as g

Q = "ℚ"

def q(x):
    return f"({x.numerator} : {Q})" if x.denominator == 1 \
        else f"({x.numerator}/{x.denominator} : {Q})"

def mat(name, n, M, doc):
    body = ";\n    ".join(", ".join(q(x) for x in row) for row in M)
    return f"/-- {doc} -/\ndef {name} : Matrix (Fin {n}) (Fin {n}) {Q} :=\n  !![{body}]\n"

def vec(name, n, v, doc):
    body = ", ".join(q(x) for x in v)
    return f"/-- {doc} -/\ndef {name} : Fin {n} → {Q} := ![{body}]\n"

MATSIMP = "Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_succ"
VECSIMP = "Matrix.mulVec, dotProduct, Fin.sum_univ_succ"

def matprf(stmt, name, extra):
    return (f"set_option maxHeartbeats 4000000 in\n"
            f"theorem {name} : {stmt} := by\n"
            f"  ext i j\n"
            f"  fin_cases i <;> fin_cases j <;>\n"
            f"    norm_num [{extra}, {MATSIMP}]\n")

def vecprf(stmt, name, extra):
    return (f"set_option maxHeartbeats 4000000 in\n"
            f"theorem {name} : {stmt} := by\n"
            f"  funext i\n"
            f"  fin_cases i <;>\n"
            f"    norm_num [{extra}, {VECSIMP}]\n")

def emit(p, N, path):
    d = g.verify(p, N)
    B, M, K, Bd, L, D, Binv = d["B"], d["M"], d["K"], d["Bd"], d["L"], d["D"], d["Binv"]
    Linv = g.inverse(L)
    assert g.matmul(L, Linv) == g.identity(N)
    xs = d["x"]
    ns = f"P{p}N{N}"
    h = f"1/{N - p}"
    out = []
    A = out.append
    A(f'''import Mathlib
import BsplineBoundary.Cert
import BsplineBoundary.Knots

/-!
# Exact rational certificates at degree {p} with {N} basis functions

This file certifies, as exact identities between explicit rational matrices, the finite
content of Section 3 of `paper/main.tex` at `p = {p}`, `N = {N}` on the unit interval with mesh
width `h = {h}`.

All literals are produced by `gen_sbp.py` from the manuscript's own constructions in
`fractions.Fraction`: the clamped open-uniform knot vector and the Cox-de Boor recursion in
exact rational piecewise-monomial form, the true Greville abscissae, the sampled basis
matrix `B i j = B_j(x_i)`, the exact integrals `M i j = int B_i B_j` and
`K i j = int B_i B_j'`, the signed endpoint matrix
`Bd = -e_0 e_0^T + e_{{N-1}} e_{{N-1}}^T`, an exact `L D Lᵀ` factorization of `M`, exact
inverses of `B` and `L`, and the Marsden coefficient vectors of the monomials `x^k`,
`k = 0, ..., {p}`. No floating point enters at any stage, and Lean re-verifies every product
from the literals.
-/

namespace BsplineBoundary
namespace {ns}

open Matrix

/-! ## The exact objects -/
''')
    A(vec("grev", N, xs, f"True Greville abscissae `x_i` at `p = {p}`, `N = {N}` (`paper/main.tex` 517-522)."))
    A(mat("B", N, B, "Sampled basis matrix `B i j = B_j(x_i)` (`paper/main.tex` 565-570)."))
    A(mat("Binv", N, Binv, "Exact inverse of the sampled basis matrix."))
    A(mat("M", N, M, "Exact spline mass form `M i j = int_0^1 B_i B_j dx` (`paper/main.tex` 578)."))
    A(mat("K", N, K, "Exact coefficient advection form `K i j = int_0^1 B_i B_j' dx` (`paper/main.tex` 579)."))
    A(mat("Bd", N, Bd, "Signed wall-trace endpoint matrix `-e_0 e_0^T + e_{N-1} e_{N-1}^T` (`paper/main.tex` 592)."))
    A(mat("L", N, L, "Unit lower-triangular factor of the exact `L D Lᵀ` factorization of `M`."))
    A(mat("Linv", N, Linv, "Exact inverse of `L`."))
    A(vec("D", N, D, "Exact pivots of the `L D Lᵀ` factorization of `M`."))
    LD = g.matmul(L, [[D[i] if i == j else F(0) for j in range(N)] for i in range(N)])
    A(mat("LD", N, LD, "The exact product `L * diagonal D`."))
    for k in range(p + 1):
        A(vec(f"alpha{k}", N, g.marsden_alpha(p, N, k),
              f"Marsden coefficients of `x^{k}`: `x^{k} = sum_j alpha{k}_j B_j(x)`."))

    A(f'''
/-! ## The Greville node set

The literal `grev` agrees with the general definition `BsplineBoundary.greville` of
`Knots.lean`, so the node placement theorems proved there for general `p` and `N` apply to
the matrices of this file.
-/
''')
    A(f'''theorem grev_eq_greville (i : Fin {N}) : grev i = greville {p} {N} (i : ℕ) := by
  fin_cases i <;>
    norm_num [grev, greville_eq_range, knot, Finset.sum_range_succ]
''')

    A(f'''
/-! ## Endpoint traces of the clamped basis

The first and last rows of the sampled basis matrix are the standard basis covectors,
which is the numerical instance at `p = {p}`, `N = {N}` of the rank-two endpoint trace
`B_0(0) = 1`, `B_j(0) = 0` for `j > 0`, `B_{{N-1}}(1) = 1`, `B_j(1) = 0` for `j < N-1`
(`paper/main.tex` 507-516), using `x_0 = 0` and `x_{{N-1}} = 1`.
-/
''')
    A(f'''/-- `B_0(x_0) = 1` and `B_j(x_0) = 0` for `j > 0`, with `x_0 = 0`. -/
theorem B_wall_row_left (j : Fin {N}) : B 0 j = if j = 0 then 1 else 0 := by
  fin_cases j <;> simp [B]
''')
    A(f'''/-- `B_{{N-1}}(x_{{N-1}}) = 1` and `B_j(x_{{N-1}}) = 0` for `j < N-1`, with `x_{{N-1}} = 1`. -/
theorem B_wall_row_right (j : Fin {N}) : B {N-1} j = if j = {N-1} then 1 else 0 := by
  fin_cases j <;> simp [B]
''')
    A(vecprf(f"B *ᵥ (fun _ => (1 : {Q})) = (fun _ => (1 : {Q}))", "B_mulVec_one", "B") .replace(
        "theorem B_mulVec_one", "/-- Partition of unity at the Greville nodes: `B 1 = 1`. -/\ntheorem B_mulVec_one"))

    A(f'''
/-! ## The coefficient-space Green identity

`paper/main.tex` equation `eq:spline-construction-coeff-green` (lines 588-597) and
Proposition `prop:bmf-coeff-d1` state
`K + K^T = -e_0 e_0^T + e_{{N-1}} e_{{N-1}}^T = B^T Bd B` together with `K 1 = 0`.
All three parts are certified below at `p = {p}`, `N = {N}`. This is the identity that
Theorem `thm:bmg-first-derivative` (lines 1400-1430) turns by congruence into the BMG
natural-metric SBP identity `H D + D^T H = Bd`.
-/
''')
    A(matprf("K + Kᵀ = Bd", "K_add_transpose", "K, Bd").replace(
        "theorem K_add_transpose",
        "/-- `K + K^T = -e_0 e_0^T + e_{N-1} e_{N-1}^T`, integration by parts in coefficient space. -/\ntheorem K_add_transpose"))
    A(matprf("Bd * B = Bd", "Bd_mul_B", "B, Bd"))
    A(matprf("Bᵀ * Bd = Bd", "Bt_mul_Bd", "B, Bd"))
    A(f'''/-- The congruence form of the endpoint matrix: `B^T Bd B = Bd`, a consequence of the
two wall rows of `B` being the standard basis covectors. -/
theorem sample_congruence : Bᵀ * Bd * B = Bd := by
  rw [Matrix.mul_assoc, Bd_mul_B, Bt_mul_Bd]
''')
    A(f'''/-- **The coefficient-space Green identity in the printed congruence form**
(`paper/main.tex`, `eq:spline-construction-coeff-green`). -/
theorem green_identity : K + Kᵀ = Bᵀ * Bd * B := by
  rw [sample_congruence, K_add_transpose]
''')
    A(vecprf(f"K *ᵥ (fun _ => (1 : {Q})) = (fun _ => (0 : {Q}))", "K_mulVec_one", "K").replace(
        "theorem K_mulVec_one",
        "/-- `K 1 = 0`: the advection form annihilates constants. -/\ntheorem K_mulVec_one"))

    A(f'''
/-! ## The mass form is symmetric positive definite

The exact `L D Lᵀ` factorization with strictly positive pivots, together with the exact
inverse of `L`, certifies positive definiteness through `BsplineBoundary.posDef_of_ldl`.
-/
''')
    A(matprf("Mᵀ = M", "M_transpose", "M"))
    A("set_option maxHeartbeats 4000000 in\n"
      "/-- `L * diagonal D` as an explicit rational literal. -/\n"
      "theorem L_mul_D : L * Matrix.diagonal D = LD := by\n"
      "  ext i j\n"
      "  rw [Matrix.mul_diagonal]\n"
      "  fin_cases i <;> fin_cases j <;> norm_num [L, D, LD]\n")
    A("set_option maxHeartbeats 4000000 in\n"
      "/-- **The exact `L D Lᵀ` factorization of the spline mass form.** -/\n"
      "theorem M_ldl : L * Matrix.diagonal D * Lᵀ = M := by\n"
      "  rw [L_mul_D]\n"
      "  ext i j\n"
      "  fin_cases i <;> fin_cases j <;>\n"
      f"    norm_num [LD, L, M, {MATSIMP}]\n")
    A(matprf("L * Linv = 1", "L_mul_Linv", "L, Linv, Matrix.one_apply"))
    A(f'''theorem D_pos : ∀ i, 0 < D i := by
  intro i
  fin_cases i <;> norm_num [D]
''')
    A(f'''/-- **The spline mass form is positive definite** at `p = {p}`, `N = {N}`. -/
theorem M_posDef : M.PosDef :=
  posDef_of_ldl M_ldl D_pos L_mul_Linv
''')

    A(f'''
/-! ## The sampled basis matrix is nonsingular

`paper/main.tex` lines 530-545 records that the Greville sample matrix is nonsingular by
total positivity and Schoenberg-Whitney theory. At this fixed case the fact is certified
directly by an exact rational inverse.
-/
''')
    A(matprf("B * Binv = 1", "B_mul_Binv", "B, Binv, Matrix.one_apply"))
    A(f'''/-- `det B != 0`: the Greville sample matrix is nonsingular. -/
theorem B_det_ne_zero : B.det ≠ 0 :=
  det_ne_zero_of_mul_eq_one B_mul_Binv

theorem B_isUnit_det : IsUnit B.det :=
  isUnit_det_of_mul_eq_one B_mul_Binv
''')

    A(f'''
/-! ## Polynomial reproduction through the Marsden coefficients

For `k = 0, ..., {p}` the Marsden coefficient vector `alpha^(k)_j = e_k(t_{{j+1}},...,t_{{j+p}}) / C(p,k)`
expands the monomial `x^k` in the clamped basis, so `B alpha^(k)` is the vector of Greville
samples of `x^k`. This is the finite instance of the `P_p` exactness used in
Theorem `thm:bmg-first-derivative` (`eq:bmg-polynomial-exactness`). At `k = 0` it is the
partition of unity and at `k = 1` the coefficient vector is the Greville node vector itself.
-/
''')
    for k in range(p + 1):
        A(vecprf(f"B *ᵥ alpha{k} = (fun i => grev i ^ {k})", f"poly_reproduction_{k}",
                 f"B, alpha{k}, grev"))
    A(f'''/-- At `k = 1` the Marsden coefficients are exactly the Greville abscissae
(`paper/main.tex` 517-522). -/
theorem alpha1_eq_grev : alpha1 = grev := by
  funext i
  fin_cases i <;> norm_num [alpha1, grev]
''')
    A(f'''
end {ns}
end BsplineBoundary
''')
    open(path, "w").write("\n".join(out))
    print(f"wrote {path}  (p = {p}, N = {N})")

if __name__ == "__main__":
    emit(2, 6, "BsplineBoundary/SBP26.lean")
    emit(3, 8, "BsplineBoundary/SBP38.lean")
