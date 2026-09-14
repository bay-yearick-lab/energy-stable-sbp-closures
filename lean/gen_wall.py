"""Exact rank certificate for the BMG wall-transfer channel count 8p-6 at p = 2.

The objects are re-derived here from the repository's own constructions
(`code/universal_wall_transfer.py` and `code/all_degree_wall_rank.py`) in exact
`fractions.Fraction` arithmetic:

  npts = 4p + 9                             certificate grid of the wall-rank producer
  unit-cell clamped open-uniform knots      [0]*(p+1) + [1..cells-1] + [cells]*(p+1)
  mass[i][j]      = int B_i B_j             dimensionless, cell width 1
  advection[i][j] = int B_i B_j'
  core            = Toeplitz extension of the cardinal interior row (bandwidth p)
  BMG acoustic pencil (metric Lp, generator Gp) on the pressure-velocity pair,
  velocity restricted to the interior indices 1..npts-2
  correction      = [Lp - Lp0, Gp - Gp0] on the rows where it is nonzero

At p = 2 (npts = 17, pencil size 32, correction 32 x 64) exactly ten rows are nonzero,
namely 0,1,2,14,15,16,17,18,30,31, and the rank of the correction is 10 = 8p - 6.

The Lean file certifies the rank of the ten-row active block by an exact right inverse.
"""
from fractions import Fraction as F
import gen_sbp as g

def unit_knots(p, npts):
    cells = npts - p
    return [F(0)] * (p + 1) + [F(j) for j in range(1, cells)] + [F(cells)] * (p + 1)

def basis_pieces(p, npts):
    t = unit_knots(p, npts)
    sp = [s for s in range(len(t) - 1) if t[s] < t[s + 1]]
    cur = [{s: [F(1)] for s in sp if s == i} for i in range(npts + p)]
    for k in range(1, p + 1):
        nxt = []
        for i in range(npts + p - k):
            piece = {}
            d1 = t[i + k] - t[i]
            if d1 != 0:
                for s, poly in cur[i].items():
                    piece[s] = g.padd(piece.get(s, [F(0)]),
                                      g.pscale_linear(poly, -t[i] / d1, F(1) / d1))
            d2 = t[i + k + 1] - t[i + 1]
            if d2 != 0:
                for s, poly in cur[i + 1].items():
                    piece[s] = g.padd(piece.get(s, [F(0)]),
                                      g.pscale_linear(poly, t[i + k + 1] / d2, F(-1) / d2))
            nxt.append(piece)
        cur = nxt
    return t, sp, cur[:npts]

def coeff_matrices(p, npts):
    t, sp, pc = basis_pieces(p, npts)
    mass = [[F(0)] * npts for _ in range(npts)]
    adv = [[F(0)] * npts for _ in range(npts)]
    for i in range(npts):
        for j in range(max(0, i - p), min(npts, i + p + 1)):
            for s in sp:
                a, b = pc[i].get(s), pc[j].get(s)
                if a and b:
                    mass[i][j] += g.pintegral(g.pmul(a, b), t[s], t[s + 1])
                    adv[i][j] += g.pintegral(g.pmul(a, g.pderiv(b)), t[s], t[s + 1])
    return mass, adv

def toeplitz_core(M, bw):
    n = len(M)
    anchor = n // 2
    v = {o: M[anchor][anchor + o] for o in range(-bw, bw + 1)}
    return [[v.get(j - i, F(0)) for j in range(n)] for i in range(n)]

def bmg_pencil(mass, adv):
    n = len(mass)
    interior = list(range(1, n - 1))
    size = n + len(interior)
    metric = [[F(0)] * size for _ in range(size)]
    for i in range(n):
        for j in range(n):
            metric[i][j] = mass[i][j]
    for a, i in enumerate(interior):
        for b, j in enumerate(interior):
            metric[n + a][n + b] = mass[i][j]
    gen = [[F(0)] * size for _ in range(size)]
    for i in range(n):
        for lj, j in enumerate(interior):
            gen[i][n + lj] = -adv[i][j]
            gen[n + lj][i] = -adv[j][i]
    return metric, gen

def sub(A, B):
    return [[a - b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]

def rref_pivots(M):
    W = [r[:] for r in M]
    rows, cols, r, piv = len(W), len(W[0]), 0, []
    for c in range(cols):
        pr = next((k for k in range(r, rows) if W[k][c] != 0), None)
        if pr is None:
            continue
        W[r], W[pr] = W[pr], W[r]
        pv = W[r][c]
        W[r] = [x / pv for x in W[r]]
        for k in range(rows):
            if k != r and W[k][c] != 0:
                f = W[k][c]
                W[k] = [x - f * y for x, y in zip(W[k], W[r])]
        piv.append(c)
        r += 1
        if r == rows:
            break
    return r, piv

def build(p):
    npts = 4 * p + 9
    mass, adv = coeff_matrices(p, npts)
    Lp, Gp = bmg_pencil(mass, adv)
    Lp0, Gp0 = bmg_pencil(toeplitz_core(mass, p), toeplitz_core(adv, p))
    dL, dG = sub(Lp, Lp0), sub(Gp, Gp0)
    rows = sorted(set(i for i, r in enumerate(dL) if any(r))
                  | set(i for i, r in enumerate(dG) if any(r)))
    A = [dL[i] + dG[i] for i in rows]
    rank, piv = rref_pivots(A)
    assert rank == 8 * p - 6, (rank, 8 * p - 6)
    assert len(rows) == rank
    S = [[A[i][c] for c in piv] for i in range(len(rows))]
    Sinv = g.inverse(S)
    assert g.matmul(S, Sinv) == g.identity(rank)
    width = len(A[0])
    R = [[F(0)] * rank for _ in range(width)]
    for k, c in enumerate(piv):
        R[c] = Sinv[k][:]
    assert g.matmul(A, R) == g.identity(rank)
    return dict(npts=npts, size=len(Lp), A=A, R=R, rows=rows, piv=piv,
                rank=rank, width=width)

Q = "ℚ"

def q(x):
    return f"({x.numerator} : {Q})" if x.denominator == 1 \
        else f"({x.numerator}/{x.denominator} : {Q})"

def mat(name, r, c, M, doc):
    body = ";\n    ".join(", ".join(q(x) for x in row) for row in M)
    return f"/-- {doc} -/\ndef {name} : Matrix (Fin {r}) (Fin {c}) {Q} :=\n  !![{body}]\n"

def emit(p, path):
    d = build(p)
    n, w, k = d["rank"], d["width"], d["rank"]
    out = [f'''import Mathlib
import BsplineBoundary.Cert

/-!
# The wall-transfer channel count `8p - 6` at degree two

The wall-transfer producers of this repository (`code/universal_wall_transfer.py` and `code/all_degree_wall_rank.py`) report the boundary-channel count `8p - 6` for the BMG
acoustic generalized pencil. This file certifies the value at `p = 2`, where `8p - 6 = 10`,
as an exact rational rank identity.

The objects are re-derived in `gen_wall.py` from the same constructions in
`fractions.Fraction`: the certificate grid `npts = 4p + 9 = {d["npts"]}`, the dimensionless
clamped open-uniform mass and advection forms on unit cells, the Toeplitz core obtained by
extending the cardinal interior row of bandwidth `p`, and the BMG acoustic pencil
`(metric, generator)` of size `{d["size"]}` on the pressure-velocity pair with the velocity block
restricted to the interior indices. The correction `[metric - core metric,
generator - core generator]` is a `{d["size"]} x {w}` matrix whose nonzero rows are exactly the
`{n}` indices `{d["rows"]}`.

`wallCorrection` below is that active `{n} x {w}` block. The rank is pinned from both sides:
at most `{n}` because the block has `{n}` rows, and at least `{n}` because `wallRightInverse`
is an exact right inverse, built from the inverse of the nonsingular `{n} x {n}` minor on the
pivot columns `{d["piv"]}`.

**Scope.** Lean certifies the rank of the active block. That the remaining
`{d["size"] - n}` rows of the full `{d["size"]} x {w}` correction vanish is checked in exact rational
arithmetic by the generator and is not re-verified inside Lean; the full correction matrix
is not carried into this file. Only `p = 2` is certified here: the general formula `8p - 6`
is not proved.
-/

namespace BsplineBoundary
namespace WallRankP2

open Matrix

''']
    A = out.append
    A(mat("wallCorrection", n, w, d["A"],
          f"The active {n} x {w} block of the BMG acoustic pencil correction at `p = 2`."))
    A(mat("wallRightInverse", w, n, d["R"],
          "An exact right inverse of the active block, from the inverse of the pivot minor."))
    A(f'''set_option maxHeartbeats 4000000 in
/-- The exact right inverse identity. -/
theorem wallCorrection_mul_rightInverse : wallCorrection * wallRightInverse = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [wallCorrection, wallRightInverse, Matrix.one_apply, Matrix.mul_apply,
      Fin.sum_univ_succ]
''')
    A(f'''/-- **The wall-transfer channel count at degree two.** The BMG acoustic pencil
correction has rank `{n} = 8 * 2 - 6`. -/
theorem wallCorrection_rank : wallCorrection.rank = {n} := by
  refine le_antisymm ?_ ?_
  · simpa using wallCorrection.rank_le_card_height
  · have h : (1 : Matrix (Fin {n}) (Fin {n}) {Q}).rank
        ≤ wallCorrection.rank := by
      rw [← wallCorrection_mul_rightInverse]
      exact Matrix.rank_mul_le_left wallCorrection wallRightInverse
    simpa [Matrix.rank_one] using h

/-- The certified value is the degree-two instance of the reported formula `8p - 6`. -/
theorem channel_count_formula : 8 * 2 - 6 = {n} := by norm_num
''')
    A(f'''
end WallRankP2
end BsplineBoundary
''')
    open(path, "w").write("\n".join(out))
    print(f"wrote {path}: rank {n}, active block {n} x {w}, pivots {d['piv']}")

if __name__ == "__main__":
    emit(2, "BsplineBoundary/WallRank.lean")
