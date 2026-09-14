# Lean 4 certificates for Section 3 of the boundary manuscript

This directory is a Lake project, `BsplineBoundary`, that certifies the finite, exactly
computable content of Section 3 of `paper/main.tex` (the clamped open-uniform spline space,
the true Greville node set, the coefficient bilinear forms `M` and `K`, and the
coefficient-space Green identity that carries the BMG summation-by-parts theorem) in Lean 4
with Mathlib. It also certifies the degree-two instance of the wall-transfer channel count
reported by the repository's wall-rank producers.

Every statement listed below is proved with no `sorry`. The axiom audit of all 76 theorems
reports only `propext`, `Classical.choice` and `Quot.sound`, the standard axioms of Mathlib.
The analytic theorems of the manuscript are not formalized, and the scope table says so
statement by statement.

One object enters by a classical closed form rather than by its defining construction: the
identification of the clamped degree-`p` B-splines on the first knot span with the Bernstein
polynomials of that span, which is the classical consequence of the `(p+1)`-fold knot
repetition at the wall. It appears as the definition `clampedFirstSpan` in `Endpoint.lean`
and is not re-derived from the Cox-de Boor recursion, exactly as the periodic project takes
the cardinal B-spline by its truncated-power closed form. Everything below that definition is
proved. The finite cases in `SBP26.lean` and `SBP38.lean` do not rest on that identification:
there the sampled basis matrices are produced from the Cox-de Boor recursion in exact
rational arithmetic, and the endpoint traces are read off the certified matrices themselves.

## Toolchain

| component | version |
| --- | --- |
| elan | 4.2.4 (227caca13 2026-08-25) |
| Lean | 4.33.1 (`leanprover/lean4:v4.33.1`, commit 819816b2e0a3bf405af45ae5c7af2491d8f5bee6) |
| Lake | 5.0.0-src+819816b |
| Mathlib | tag `v4.33.1`, commit `0df444a360eaa60ab8c11dca51a86af692955474` |
| batteries / aesop / Qq / proofwidgets / importGraph / LeanSearchClient / plausible / Cli | pinned in `lake-manifest.json` |
| Python (generators) | 3, standard library only (`fractions`, `itertools`, `math`) |

## Build

```
curl -sSf https://elan.lean-lang.org/elan-init.sh | sh -s -- -y --default-toolchain none
export PATH="$HOME/.elan/bin:$PATH"
cd lean
lake exe cache get      # fetches the prebuilt Mathlib oleans (about 7.5 GB under lean/.lake)
lake build
```

A clean build of the six modules of `BsplineBoundary/` takes 2 min 16 s once the Mathlib
cache is in place. `WallRank.lean` accounts for 124 s of that, `SBP38.lean` for 30 s and
`SBP26.lean` for 17 s. The build of 2026-09-13 ended with

```
✔ [8706/8713] Built BsplineBoundary.Endpoint (3.7s)
✔ [8707/8713] Built BsplineBoundary.Cert (3.8s)
✔ [8708/8713] Built BsplineBoundary.Knots (3.9s)
✔ [8709/8713] Built BsplineBoundary.SBP26 (17s)
✔ [8710/8713] Built BsplineBoundary.SBP38 (30s)
✔ [8711/8713] Built BsplineBoundary.WallRank (124s)
✔ [8712/8713] Built BsplineBoundary (5.5s)
Build completed successfully (8713 jobs).
```

To audit the axioms of a theorem, run `lake env lean` on a file containing
`import BsplineBoundary` and `#print axioms BsplineBoundary.<name>`.

## Layout

| file | contents |
| --- | --- |
| `BsplineBoundary/Knots.lean` | clamped open-uniform knot vector and true Greville abscissae, general `p` and `N` |
| `BsplineBoundary/Endpoint.lean` | Bernstein basis and the endpoint traces of the clamped basis on the first knot span |
| `BsplineBoundary/Cert.lean` | reusable bridges from an exact product identity to nonsingularity and positive definiteness |
| `BsplineBoundary/SBP26.lean` | exact rational certificates at `p = 2`, `N = 6` |
| `BsplineBoundary/SBP38.lean` | exact rational certificates at `p = 3`, `N = 8` |
| `BsplineBoundary/WallRank.lean` | exact rank certificate for the wall-transfer channel count at `p = 2` |
| `gen_sbp.py` | exact rational generator: Cox-de Boor basis, knots, Greville nodes, `M`, `K`, `B`, LDL, inverses, Marsden coefficients |
| `gen_lean.py` | emits `SBP26.lean` and `SBP38.lean` from `gen_sbp.py` |
| `gen_wall.py` | emits `WallRank.lean`: BMG acoustic pencil, Toeplitz core, correction block, right inverse |

The generators use `fractions.Fraction` throughout. No floating point enters any literal in
this directory. Each generator re-checks every identity in Python before emitting it, and
Lean then re-verifies the identity from the literals alone.

## Scope table

### Clamped knots and true Greville nodes, general degree and general basis count

Knot vector `t_0 = ... = t_p = 0`, `t_N = ... = t_{N+p} = 1`, `t_{p+j} = j / (N - p)`;
Greville abscissae `x_i = (1/p) * sum_{m=1}^{p} t_{i+m}`. These hold for every `p >= 1` and
every `N >= p + 1`, not only at the certified finite cases.

| theorem | certifies | paper label | status |
| --- | --- | --- | --- |
| `knot_of_le` | `t_k = 0` for `k <= p` | main.tex 492-496 | proved, general `p`, `N` |
| `knot_of_ge` | `t_k = 1` for `k >= N` | main.tex 492-496 | proved, general `p`, `N` |
| `knot_eq` | single affine formula `(k - p)/(N - p)` on `p <= k <= N`, both clamped ends included | main.tex 497-500 | proved, general `p`, `N` |
| `sum_Icc_one_shift` | reindexing of the Greville sum onto `Finset.range` | supporting lemma | proved |
| `greville_eq_range` | Greville abscissa in telescoping form | supporting lemma | proved, general `p`, `N` |
| `greville_zero` | `x_0 = 0`, the left wall is a node | main.tex 524 | proved, general `p`, `N` |
| `greville_last` | `x_{N-1} = 1`, the right wall is a node | main.tex 524 | proved, general `p`, `N` |
| `greville_succ_sub` | uniform interior spacing `x_{i+1} - x_i = 1/(N - p)` for `p - 1 <= i <= N - p - 1` | main.tex 517-530 | proved, general `p`, `N` |

### Endpoint traces on the first knot span

| theorem | certifies | paper label | status |
| --- | --- | --- | --- |
| `bern_zero` | `b_{p,0}(x) = (1 - x)^p` | main.tex 510-513 | proved |
| `bern_top` | `b_{p,p}(x) = x^p` | main.tex 510-513 | proved |
| `bern_at_zero_zero`, `bern_at_zero_pos` | `b_{p,0}(0) = 1`, `b_{p,j}(0) = 0` for `j > 0` | main.tex 514-516 | proved |
| `bern_at_one_top`, `bern_at_one_lt` | `b_{p,p}(1) = 1`, `b_{p,j}(1) = 0` for `j < p` | main.tex 514-516 | proved |
| `bern_sum` | Bernstein partition of unity, from the binomial theorem | main.tex 505-516 | proved |
| `clampedFirstSpan_zero` | `B_0` on `[0, h]` equals `(1 - x/h)^p` | main.tex 510-513 | proved from the classical Bernstein identification |
| `clampedFirstSpan_at_wall_zero` | `B_0(0) = 1` | main.tex 514-516 | proved from the classical Bernstein identification |
| `clampedFirstSpan_at_wall_pos` | `B_j(0) = 0` for `j > 0` | main.tex 514-516 | proved from the classical Bernstein identification |
| `clampedFirstSpan_at_span_end_top`, `_lt` | the mirror statement at the far end of the first span | main.tex 514-516 | proved from the classical Bernstein identification |
| `clampedFirstSpan_sum` | partition of unity on the first span | main.tex 505-516 | proved |
| `clampedFirstSpan_wall_row` | the wall row of the sampled basis is the first standard basis covector | main.tex 507-516 | proved from the classical Bernstein identification |

### Reusable certificate bridges

| theorem | certifies | paper label | status |
| --- | --- | --- | --- |
| `det_ne_zero_of_mul_eq_one` | an exact right inverse forces a nonzero determinant | supporting lemma | proved |
| `isUnit_det_of_mul_eq_one` | the same in unit form | supporting lemma | proved |
| `transpose_eq_of_ldl` | a matrix with an exact `L D Lᵀ` factorization is symmetric | supporting lemma | proved |
| `posDef_of_ldl` | exact `L D Lᵀ` with positive pivots and nonsingular `L` gives `Matrix.PosDef` | supporting lemma | proved |

### Exact rational certificates at `p = 2`, `N = 6` and `p = 3`, `N = 8`

Namespaces `P2N6` and `P3N8`. Both cases carry the same statements; the tables are
identical except that polynomial reproduction runs to `k = p`. Objects are the sampled basis
matrix `B i j = B_j(x_i)`, the exact integrals `M i j = int_0^1 B_i B_j` and
`K i j = int_0^1 B_i B_j'`, the signed endpoint matrix
`Bd = -e_0 e_0^T + e_{N-1} e_{N-1}^T`, an exact `L D Lᵀ` factorization of `M`, exact inverses
of `B` and `L`, and the Marsden coefficient vectors.

| theorem | certifies | paper label | status |
| --- | --- | --- | --- |
| `grev_eq_greville` | the literal node vector agrees with the general definition of `Knots.lean` | main.tex 517-522 | certified at both cases |
| `B_wall_row_left` | `B_0(x_0) = 1` and `B_j(x_0) = 0` for `j > 0`, with `x_0 = 0` | main.tex 507-516 | certified at both cases |
| `B_wall_row_right` | `B_{N-1}(x_{N-1}) = 1` and `B_j(x_{N-1}) = 0` for `j < N-1`, with `x_{N-1} = 1` | main.tex 507-516 | certified at both cases |
| `B_mulVec_one` | partition of unity at the Greville nodes, `B 1 = 1` | main.tex 530-545 | certified at both cases |
| `K_add_transpose` | `K + K^T = -e_0 e_0^T + e_{N-1} e_{N-1}^T`, integration by parts in coefficient space | `eq:spline-construction-coeff-green`, `prop:bmf-coeff-d1` | certified at both cases |
| `Bd_mul_B`, `Bt_mul_Bd` | the two one-sided endpoint products | supporting steps | certified at both cases |
| `sample_congruence` | `B^T Bd B = Bd`, the congruence form of the endpoint matrix | `eq:spline-construction-coeff-green` | certified at both cases |
| `green_identity` | `K + K^T = B^T Bd B`, the printed coefficient-space Green identity | `eq:spline-construction-coeff-green`, `prop:bmf-coeff-d1` | certified at both cases |
| `K_mulVec_one` | `K 1 = 0`, the advection form annihilates constants | `prop:bmf-coeff-d1` | certified at both cases |
| `M_transpose` | `M^T = M` | main.tex 577-582 | certified at both cases |
| `L_mul_D` | `L * diagonal D` as an explicit literal | supporting step | certified at both cases |
| `M_ldl` | the exact `L D Lᵀ` factorization of the mass form | main.tex 577-582 | certified at both cases |
| `L_mul_Linv` | `L` is nonsingular, by an exact inverse | supporting step | certified at both cases |
| `D_pos` | every pivot of the factorization is strictly positive | main.tex 577-582 | certified at both cases |
| `M_posDef` | the spline mass form is symmetric positive definite | main.tex 577-582, used by `thm:bmg-first-derivative` | certified at both cases |
| `B_mul_Binv` | an exact rational inverse of the Greville sample matrix | main.tex 530-545 | certified at both cases |
| `B_det_ne_zero`, `B_isUnit_det` | `det B != 0`, the Schoenberg-Whitney nonsingularity at these cases | main.tex 530-545 | certified at both cases |
| `poly_reproduction_0` ... `poly_reproduction_p` | `B alpha^(k)` is the vector of Greville samples of `x^k`, for `k = 0, ..., p` | `eq:bmg-polynomial-exactness`, `prop:bmf-poly-sector` | certified at both cases |
| `alpha1_eq_grev` | the Marsden coefficients at `k = 1` are the Greville abscissae | main.tex 517-522 | certified at both cases |

The manuscript states the coefficient Green identity in two forms,
`K + K^T = -e_0 e_0^T + e_{N-1} e_{N-1}^T` and `K + K^T = B^T Bd B`. Both are certified.
They agree because the first and last rows of the Greville sample matrix are the standard
basis covectors, which is the content of `B_wall_row_left` and `B_wall_row_right` together
with `x_0 = 0` and `x_{N-1} = 1`. Theorem `thm:bmg-first-derivative` turns this identity by
congruence into the BMG natural-metric identity `H D + D^T H = Bd`; the congruence step
itself is a two-line algebraic manipulation in the manuscript and is not formalized here.

### Wall-transfer channel count at degree two

| theorem | certifies | paper label | status |
| --- | --- | --- | --- |
| `wallCorrection_mul_rightInverse` | an exact right inverse of the active correction block | `code/universal_wall_transfer.py` `bmg_acoustic_common_left_rank` | certified at `p = 2` |
| `wallCorrection_rank` | the BMG acoustic pencil correction has rank 10 | reported formula `8p-6` | certified at `p = 2` only |
| `channel_count_formula` | `8 * 2 - 6 = 10` | reported formula `8p-6` | proved |

The certificate grid is `npts = 4p + 9 = 17`, the BMG acoustic generalized pencil has size
32, and the correction `[metric - core metric, generator - core generator]` is a 32 by 64
matrix whose nonzero rows are exactly the ten indices 0, 1, 2, 14, 15, 16, 17, 18, 30, 31.
`wallCorrection` is that ten by 64 active block, and `wallCorrection_rank` pins its rank from
both sides: at most ten because the block has ten rows, and at least ten through the exact
right inverse built from the inverse of the nonsingular ten by ten minor on the pivot columns
0, 1, 2, 14, 15, 16, 17, 18, 30, 31. Two limits apply. That the remaining 22 rows of the full
32 by 64 correction vanish is checked in exact rational arithmetic by `gen_wall.py` and is not
re-verified inside Lean, because the full correction literal was too large for the kernel in
the available time. The general formula `8p - 6` is not proved; only its degree-two instance
is certified.

## Axiom audit

Running `#print axioms` on all 76 theorems of the project gives 75 lines reading
`[propext, Classical.choice, Quot.sound]` and one reading `[propext]`. There is no
`sorryAx` and no additional axiom anywhere. The single `[propext]` line is

```
'BsplineBoundary.WallRankP2.channel_count_formula' depends on axioms: [propext]
```

The remaining 75 theorems, listed in dependency order by module, all report
`[propext, Classical.choice, Quot.sound]`:

```
BsplineBoundary.knot_of_le
BsplineBoundary.knot_of_ge
BsplineBoundary.knot_eq
BsplineBoundary.sum_Icc_one_shift
BsplineBoundary.greville_eq_range
BsplineBoundary.greville_zero
BsplineBoundary.greville_last
BsplineBoundary.greville_succ_sub
BsplineBoundary.det_ne_zero_of_mul_eq_one
BsplineBoundary.isUnit_det_of_mul_eq_one
BsplineBoundary.transpose_eq_of_ldl
BsplineBoundary.posDef_of_ldl
BsplineBoundary.bern_zero
BsplineBoundary.bern_top
BsplineBoundary.bern_at_zero_zero
BsplineBoundary.bern_at_zero_pos
BsplineBoundary.bern_at_one_top
BsplineBoundary.bern_at_one_lt
BsplineBoundary.bern_sum
BsplineBoundary.clampedFirstSpan_zero
BsplineBoundary.clampedFirstSpan_at_wall_zero
BsplineBoundary.clampedFirstSpan_at_wall_pos
BsplineBoundary.clampedFirstSpan_at_span_end_top
BsplineBoundary.clampedFirstSpan_at_span_end_lt
BsplineBoundary.clampedFirstSpan_sum
BsplineBoundary.clampedFirstSpan_wall_row
BsplineBoundary.P2N6.grev_eq_greville
BsplineBoundary.P2N6.B_wall_row_left
BsplineBoundary.P2N6.B_wall_row_right
BsplineBoundary.P2N6.B_mulVec_one
BsplineBoundary.P2N6.K_add_transpose
BsplineBoundary.P2N6.Bd_mul_B
BsplineBoundary.P2N6.Bt_mul_Bd
BsplineBoundary.P2N6.sample_congruence
BsplineBoundary.P2N6.green_identity
BsplineBoundary.P2N6.K_mulVec_one
BsplineBoundary.P2N6.M_transpose
BsplineBoundary.P2N6.L_mul_D
BsplineBoundary.P2N6.M_ldl
BsplineBoundary.P2N6.L_mul_Linv
BsplineBoundary.P2N6.D_pos
BsplineBoundary.P2N6.M_posDef
BsplineBoundary.P2N6.B_mul_Binv
BsplineBoundary.P2N6.B_det_ne_zero
BsplineBoundary.P2N6.B_isUnit_det
BsplineBoundary.P2N6.poly_reproduction_0
BsplineBoundary.P2N6.poly_reproduction_1
BsplineBoundary.P2N6.poly_reproduction_2
BsplineBoundary.P2N6.alpha1_eq_grev
BsplineBoundary.P3N8.grev_eq_greville
BsplineBoundary.P3N8.B_wall_row_left
BsplineBoundary.P3N8.B_wall_row_right
BsplineBoundary.P3N8.B_mulVec_one
BsplineBoundary.P3N8.K_add_transpose
BsplineBoundary.P3N8.Bd_mul_B
BsplineBoundary.P3N8.Bt_mul_Bd
BsplineBoundary.P3N8.sample_congruence
BsplineBoundary.P3N8.green_identity
BsplineBoundary.P3N8.K_mulVec_one
BsplineBoundary.P3N8.M_transpose
BsplineBoundary.P3N8.L_mul_D
BsplineBoundary.P3N8.M_ldl
BsplineBoundary.P3N8.L_mul_Linv
BsplineBoundary.P3N8.D_pos
BsplineBoundary.P3N8.M_posDef
BsplineBoundary.P3N8.B_mul_Binv
BsplineBoundary.P3N8.B_det_ne_zero
BsplineBoundary.P3N8.B_isUnit_det
BsplineBoundary.P3N8.poly_reproduction_0
BsplineBoundary.P3N8.poly_reproduction_1
BsplineBoundary.P3N8.poly_reproduction_2
BsplineBoundary.P3N8.poly_reproduction_3
BsplineBoundary.P3N8.alpha1_eq_grev
BsplineBoundary.WallRankP2.wallCorrection_mul_rightInverse
BsplineBoundary.WallRankP2.wallCorrection_rank
```

## What rests on human proofs

Everything analytic in the manuscript rests on human proofs and is absent from this
directory. The certificates above are finite identities between exact rational numbers at
fixed degree and fixed basis count, together with the finite-sum algebra of the knot vector
and the node set, which does hold for general `p` and `N`. Nothing here bounds any quantity
uniformly in `N`. In particular the following remain outside the formalization: the uniform
inverse and inverse-type bounds on the clamped spline space; the wall order of accuracy and
every claimed convergence rate; the stability of the mass-metric projection and the
projection constants that enter the divergence-conforming pressure argument; every energy
estimate stated as an inequality for all `N`, including the semidiscrete kinetic-energy
theorem and the constrained boundary-condition ledgers; the conditioning claims, which the
manuscript itself reports as measured rather than asserted; the total-positivity and
Schoenberg-Whitney theory behind nonsingularity of the sampled basis matrix at general `p`
and `N`, of which only the two fixed cases are certified here by exact inverses; the
congruence step from the coefficient Green identity to the BMG natural-metric
summation-by-parts identity and the corresponding energy corollary; the general-degree
wall-transfer count `8p - 6`, of which only the degree-two value is certified; and the
identification of the clamped B-splines on the first knot span with the Bernstein basis,
which enters `Endpoint.lean` as a definition rather than as a theorem.
