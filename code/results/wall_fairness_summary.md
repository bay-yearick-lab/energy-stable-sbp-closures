# Comparator fairness at the wall

Contract: `code/results/contracts/K_wall_fairness.md`.
Artifact: `code/results/wall_fairness_summary.json`.
Producer: `code/s6_wall_fairness.py`. Unit checks: `code/test_wall_fairness.py`.
Constructors: `code/boundary_spline_analysis.py`
(`compact_fd_matrices_compact_baseline`, `sbp_sat_acoustic_generator`,
`sbp_narrow_second_derivative_matrix`).

Closed box, unit sound speed, standing mode `m = 6`, RK4 at `dt = 0.012 h`,
`t_end = 6` (six wall transits), trimmed interior strip of six nodes and the
untrimmed norm, `N = 41` for the march and `N in {41, 81}` for the spectral
abscissa. Wall-spectrum thresholds at `N = 129` by the code path of
`code/s4_staggered_wallmode.py`.

## Predeclared predictions

| prediction | verdict | measurement |
| --- | --- | --- |
| K1 (60 percent) | CONFIRMED | C6 with the published Section 4.1 wall row keeps a positive closed-box abscissa: `+4.562` at `N = 41` and `+7.391` at `N = 81` for `alpha = 3`; `+1.365` and `+1.812` for `alpha = 2`. The incumbent closure gives `+12.155` and `+24.296`. |
| K2 (80 percent) | CONFIRMED | SAT-enforced SBP6 abscissa `-3.45e-15` (`tau = -1`) and `-4.05e-16` (`tau = -2`) at `N = 41`, `5.96e-15` and `-1.07e-14` at `N = 81`, all at or below `1e-12`. Six-transit trimmed error `3.333e-2` and `3.363e-2` against `5.767e-2` for the projection-enforced SBP6, ratios `0.578` and `0.583`, inside the factor of three. |
| K3 (85 percent) | PARTIAL | The narrow half holds exactly: the enforced Dirichlet narrow `D2` has `max abs(lambda) = 6.031/h^2` (`9.649e3` at `N = 41`, `3.866e4` at `N = 81`), which is `O(N^2)`, and its six smallest eigenvalues reproduce `-(k pi)^2` to six digits with no spurious value. The wide half misses as stated: the wide composition has `min abs(lambda) = pi^2` as well, so no eigenvalue is near zero. What separates the two is reported below: the wide operator carries a grid-independent spurious eigenvalue near `-49.5` whose eigenvector is `0.88` to `0.89` odd-even, and its interior symbol at the odd-even wavenumber is exactly zero, while the narrow operator damps the same mode at `6.044/h^2`. |
| K4 (75 percent) | CONFIRMED | `CSB-T6` at `1.308e-3` trimmed (`1.377e-3` untrimmed) remains the best stable nodal six-transit error at order six. Every SAT variant lands at `3.33e-2` and `3.36e-2`, above `1e-3`. Every compact-baseline variant is energy-unstable and therefore does not enter the stable set. The spline margin is `8.199e-9` for BMF6 against `1.308e-3`, a factor of `1.6e5`, that is five orders. |
| K5 (60 percent) | MISSED | At tolerance `1e-2` the marched wall threshold is `32.0` points per wavelength for the projection-enforced SBP6 and `11.64` for the SAT-enforced SBP6 at both penalty values, a relative shift of `63.6` percent, far outside the predicted twenty percent. At tolerances `1e-4` and `1e-6` neither enforcement reaches the tolerance anywhere in the resolved band at `N = 129`. |

## Derived closure coefficients

The wall row `f'_1 + alpha f'_2 = (a f_1 + b f_2 + c f_3 + d f_4)/h` is
obtained by solving the square system that makes the row exact on
`1, s, s^2, s^3` at `s = 0, 1, 2, 3`, then testing `s^4` separately.

| alpha | a | b | c | d | residual on `s^4` | formal order |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | `-17/6` | `3/2` | `3/2` | `-1/6` | `-1.8e-15` | 4 |
| 2 | `-5/2` | `2` | `1/2` | `0` | `2.0` | 3 |

The derived values agree with the classical ones to `4.4e-16` at `alpha = 3`
and exactly at `alpha = 2`. The near-wall ladder is derived the same way:
the second node carries the symmetric row at `alpha = 1/4` with right side
`(-3/4, 0, 3/4)`, exact through degree four, and the third node, needed only
when the interior right side reaches three neighbors at `q = 8`, carries the
symmetric row at `alpha = 1/3` with right side
`(-1/36, -7/9, 0, 7/9, 1/36)`, exact through degree six. The assembled left
side is tridiagonal at every row, including the wall row, so no identity row
survives.

## Spectral abscissa of the closed-box acoustic generator

| operator | `N = 41` | `N = 81` | energy stable |
| --- | --- | --- | --- |
| `C4` incumbent closure | `3.095e+0` | `4.613e+0` | no |
| `C6` incumbent closure | `1.216e+1` | `2.430e+1` | no |
| `C8` incumbent closure | `2.080e+1` | `4.157e+1` | no |
| `C4-CB3` | `4.196e+0` | `7.667e+0` | no |
| `C6-CB3` | `4.562e+0` | `7.391e+0` | no |
| `C8-CB3` | `4.105e+0` | `6.633e+0` | no |
| `C4-CB2` | `1.458e+0` | `1.536e+0` | no |
| `C6-CB2` | `1.365e+0` | `1.812e+0` | no |
| `C8-CB2` | `1.485e+0` | `2.143e+0` | no |
| `C6F` incumbent closure | `1.216e+1` | `2.430e+1` | no |
| `C6F-CB3` | `4.562e+0` | `7.391e+0` | no |
| `SBP4` projection | `7.06e-15` | `1.30e-14` | yes |
| `SBP6` projection | `7.23e-15` | `9.66e-15` | yes |
| `SBP8` projection | `4.55e-13` | `1.36e-12` | yes |
| `SBP6-SAT(tau=-1)` | `-3.45e-15` | `5.96e-15` | yes |
| `SBP6-SAT(tau=-2)` | `-4.05e-16` | `-1.07e-14` | yes |
| `SBP6-BO` | `4.74e-15` | `3.21e-14` | yes |
| `CSB-E6` | `5.91e-2` | `1.46e-1` | no |
| `CSB-T6` | `3.35e-14` | `5.50e-13` | yes |
| `BMF6` | `1.55e-14` | `5.03e-14` | yes |
| `BMG6` | `6.39e-14` | `6.04e-14` | yes |

The compact-baseline closures reduce the injection rate of the compact family at order
six by a factor of `2.66` at `alpha = 3` and `8.91` at `alpha = 2`, and they
do not remove it. Filtering does not
change the semidiscrete abscissa, since the filter is a separate map applied
once per completed step.

## Six-transit closed-box error at `N = 41`

| operator | order | trimmed | untrimmed | energy stable |
| --- | --- | --- | --- | --- |
| `BMF4` | 4 | `4.768e-7` | `5.304e-7` | yes |
| `BMG4` | 4 | `5.576e-7` | `1.041e-6` | yes |
| `SBP4` | 4 | `2.022e-1` | `2.106e-1` | yes |
| `C4-CB2` | 4 | `2.127e+0` | `2.524e+0` | no |
| `C4-CB3` | 4 | `7.802e+6` | `2.089e+7` | no |
| `C4` | 4 | `1.840e+4` | `2.468e+4` | no |
| `BMF6` | 6 | `8.199e-9` | `3.593e-8` | yes |
| `BMG6` | 6 | `2.539e-8` | `3.309e-8` | yes |
| `CSB-T6` | 6 | `1.308e-3` | `1.377e-3` | yes |
| `SBP6-BO` | 6 | `8.840e-3` | `8.995e-3` | yes |
| `SBP6-SAT(tau=-1)` | 6 | `3.333e-2` | `3.591e-2` | yes |
| `SBP6-SAT(tau=-2)` | 6 | `3.363e-2` | `3.602e-2` | yes |
| `SBP6` projection | 6 | `5.767e-2` | `5.474e-2` | yes |
| `CSB-E6` | 6 | `6.415e-2` | `6.826e-2` | no |
| `C6-CB2` | 6 | `2.000e+0` | `2.159e+0` | no |
| `C6-CB3` | 6 | `6.210e+7` | `1.465e+8` | no |
| `C6F-CB3` | 6 | `2.857e+23` | `2.399e+24` | no |
| `C6` | 6 | `4.321e+26` | `2.724e+27` | no |
| `BMF8` | 8 | `4.019e-10` | `3.063e-10` | yes |
| `BMG8` | 8 | `4.109e-10` | `7.187e-10` | yes |
| `SBP8` | 8 | `1.032e-2` | `1.028e-2` | yes |
| `C8-CB2` | 8 | `5.706e+0` | `7.162e+0` | no |
| `C8-CB3` | 8 | `3.685e+5` | `6.905e+5` | no |

The `CSB-T6` untrimmed value agrees with the archived
`strong_competitor_suite.json` entry `1.3768774e-3` to seven digits, which
binds the march protocol here to the one already recorded.

## Closed-box across-order row, with a receipt

| `q` | best spline | trimmed | best stable competitor | trimmed | spline leads |
| --- | --- | --- | --- | --- | --- |
| 4 | `BMF4` | `4.8e-7` | `SBP4` | `2.0e-1` | yes |
| 6 | `BMF6` | `8.2e-9` | `CSB-T6` | `1.3e-3` | yes |
| 8 | `BMF8` | `4.0e-10` | `SBP8` | `1.0e-2` | yes |

The stable competitor pools are `{SBP4}` at `q = 4`,
`{CSB-T6, SBP6, SBP6-BO, SBP6-SAT(tau=-1), SBP6-SAT(tau=-2)}` at `q = 6`, and
`{SBP8}` at `q = 8`. No newly fielded stable closure beats the spline in any
cell, so the sentence that the spline route wins every cell of the
across-order table survives. The row as printed in the manuscript quotes
`1.1e-7 / 1.0e-1`, `6.2e-9 / 1.4e-3`, `3.5e-10 / 6.3e-3`; the measured row
under the stated Section 6.1 protocol is `4.8e-7 / 2.0e-1`, `8.2e-9 / 1.3e-3`,
`4.0e-10 / 1.0e-2`.

## Marched wall-spectrum thresholds at `N = 129`, order six

| enforcement | `tol = 1e-2` | `tol = 1e-4` | `tol = 1e-6` |
| --- | --- | --- | --- |
| metric-orthogonal projection | `32.00` ppw | not reached | not reached |
| SAT, `tau = -1` | `11.64` ppw | not reached | not reached |
| SAT, `tau = -2` | `11.64` ppw | not reached | not reached |

The relative shift at the one attainable tolerance is `0.636` for both
penalty values. SAT enforcement is the better of the two wall treatments on
this diagnostic by a factor of `2.75` in points per wavelength, and the
difference is far larger than the twenty percent predicted.

## Second-derivative spectra

`D2 = H^{-1}(-A + B S)` with `A` symmetric positive semidefinite. The narrow
operator built here is a compatible narrow operator with the published order
and structure of the narrow compatible SBP second-derivative construction, derived from the accuracy
conditions in this repository, not their published coefficient set. The
accuracy conditions leave a one-parameter family with relative residual
`1.8e-15`; the fielded member minimizes the spectral radius of the
Dirichlet-enforced operator over the positive semidefinite set and, among
parameters within one part in a thousand of that minimum, maximizes the first
nonzero eigenvalue of `A`, so the null space of `A` stays exactly the
constants.

| quantity | narrow, `N = 41` | narrow, `N = 81` | wide `D1 D1`, `N = 41` | wide `D1 D1`, `N = 81` |
| --- | --- | --- | --- | --- |
| `max abs(lambda)` | `9.649e3` | `3.866e4` | `1.012e4` | `4.205e4` |
| `max abs(lambda) h^2` | `6.031` | `6.041` | `6.326` | `6.571` |
| `min abs(lambda)` | `9.8696` | `9.8696` | `9.8696` | `9.8696` |
| six smallest eigenvalues | `-9.870, -39.478, -88.826, -157.913, -246.739, -355.320` | same to six digits | `-9.870, -39.478, -51.234, -88.826, -157.911, -201.857` | `-9.870, -39.478, -49.485, -88.826, -157.914, -197.235` |
| spurious odd-even eigenvalue | none | none | `-51.234`, overlap `0.880` | `-49.485`, overlap `0.890` |
| odd-even damping rate | `9.437e3` | `3.822e4` | `9.018e2` | `1.849e3` |
| interior symbol at `k = pi` | `-6.044/h^2` | `-6.044/h^2` | `0` | `0` |
| `A` symmetry residual | `0.0` | `0.0` | not applicable | not applicable |
| `A` smallest eigenvalue | `4.4e-15` | `-4.3e-14` | not applicable | not applicable |

The incumbent second derivative attached to `SBP6-BO` equals `D1 D1`
identically: the difference is `1.4e-16` relative at `N = 41` and
`1.3e-16` at `N = 81`. The identity is algebraic. Substituting
`D^T H = B - H D` into `H^{-1}(-D^T H D - e_0 s_0 + e_N s_N)` cancels the
boundary terms and leaves `D D` exactly, so the construction cannot produce a
narrow operator from a wide one.

The separation between the two is not a near-zero eigenvalue. It is the
damping applied to the odd-even mode: the narrow operator damps it at
`9.44e3` at `N = 41` and `3.82e4` at `N = 81`, growing like `N^2`, while the
wide composition damps it at `9.02e2` and `1.85e3`, growing like `N`, and its
interior symbol at that wavenumber is exactly zero. The wide composition also
inserts a spurious eigenvalue near `-49.5` between the second and third
physical Dirichlet eigenvalues, with an eigenvector that is `0.88` to `0.89`
odd-even and whose position does not move with the grid.

## Energy method for the simultaneous approximation terms

With `H D + D^T H = diag(-1, 0, ..., 0, 1)` and the state `(p, u)` carrying
every node of both fields,

    p_t = -D u + H^{-1}(theta_0 e_0 u_0 + theta_N e_N u_N),
    u_t = -D p + H^{-1}(tau_0 e_0 u_0 + tau_N e_N u_N),

the discrete energy `E = p^T H p + u^T H u` obeys

    dE/dt = 2 (1 + theta_0) p_0 u_0 + 2 (theta_N - 1) p_N u_N
            + 2 tau_0 u_0^2 + 2 tau_N u_N^2 .

The boundary pressure traces are unconstrained, so a rate that is nonpositive
for every state forces `theta_0 = -1` and `theta_N = +1`; the remaining
freedom is the admissible range `tau_0, tau_N <= 0`. The endpoint `tau = 0`
is energy neutral and leaves the boundary residual undamped. The smallest
strictly damping penalty scaled by the incoming characteristic speed
`abs(lambda) = 1` is `tau = -1`, at which the penalty direction is exactly the
incoming characteristic eigenvector; the doubled value `tau = -2` remains
admissible. The identity is verified on random data to `1.8e-15` at
`tau = -1` and `5.4e-15` at `tau = -2`. Applying the same penalty magnitude to
the velocity equation alone, with no pressure-equation term, leaves the
indefinite cross term and produces an abscissa of `+2.669` at `N = 41`, which
is why the direction is part of the derivation and not a convention.

## Ready-to-paste sentences

Section 6.1, after the closed-box abscissa discussion:

```latex
The compact family was also closed by the one-sided compact rows that were
published in his Section~4.1, with the wall row
$f'_1+\alpha f'_2=(af_1+bf_2+cf_3+df_4)/h$ derived from exactness on
$1,x,x^2,x^3,x^4$ rather than transcribed, reproducing
$(a,b,c,d)=(-17/6,3/2,3/2,-1/6)$ at $\alpha=3$ and $(-5/2,2,1/2,0)$ at
$\alpha=2$. The closed-box abscissa falls from $12.15$ to $4.56$ at
$\alpha=3$ and to $1.36$ at $\alpha=2$ on $N=41$, and stays positive on
$N=81$, so the energy injection of the compact route at the wall is a
property of the family rather than of the incumbent one-sided rows.
```

```latex
A matched simultaneous-approximation-term comparison completes the
enforcement axis. For the closed acoustic pair with $u=0$ at both walls, the
energy method applied to $E=p^{\top}Hp+u^{\top}Hu$ forces the penalty
directions $\theta_0=-1$ and $\theta_N=+1$ and admits every $\tau\le 0$ on
the velocity equation, with $\tau=-1$ the characteristic-scaled value. The
resulting SBP6 operator has measured abscissa $-3.4\times10^{-15}$ and a
six-transit error of $3.3\times10^{-2}$ against $5.8\times10^{-2}$ for the
same operator under metric-orthogonal projection, so weak and strong
enforcement place the diagonal-norm closure in the same accuracy band.
```

```latex
The closed-box row of Table~\ref{tab:across-order} is measured under the
protocol of this section at $N=41$ and reads
$4.8\times10^{-7}/2.0\times10^{-1}$, $8.2\times10^{-9}/1.3\times10^{-3}$, and
$4.0\times10^{-10}/1.0\times10^{-2}$, with the competitor column carried by
$\mathrm{SBP}_4$, the CSB compact scheme, and $\mathrm{SBP}_8$
(\texttt{results/wall\_fairness\_summary.json}).
```

```latex
The second derivative attached to the boundary-optimized entrant equals the
wide composition $D_1D_1$ identically, since substituting
$D^{\top}H=B-HD$ into $H^{-1}(-D^{\top}HD-e_0s_0+e_Ns_N)$ cancels the
boundary terms. A narrow compatible operator of the same order, with the
standard seven-point interior and a symmetric positive semidefinite $A$,
damps the odd-even mode at $6.04/h^2$, whereas the wide composition has zero
interior symbol at that wavenumber and carries a spurious grid-independent
eigenvalue near $-49.5$ between the second and third Dirichlet eigenvalues.
```

Conclusion, open items:

```latex
Two enforcement questions remain open. The marched wall threshold of the
diagonal-norm closure depends strongly on how the wall condition is imposed:
at tolerance $10^{-2}$ it is $32.0$ points per wavelength under
metric-orthogonal projection and $11.6$ under simultaneous approximation
terms, so the wall-spectrum diagnostic is a statement about the enforced
operator and not about the difference stencil alone.
```

```latex
The narrow second derivative fielded here follows the order and structure of
the narrow compatible SBP second-derivative construction with coefficients rederived from the accuracy
conditions; a comparison against their published coefficient set, and against
the corresponding spline compatible closure on a wall-resolved diffusion
problem, is left for later work.
```

## Commands

```
python code/s6_wall_fairness.py --out code/results/wall_fairness_summary.json
python code/s6_wall_fairness.py --verify code/results/wall_fairness_summary.json
python -m unittest test_wall_fairness            # from code/
python code/s6_wall_longtime_cliff.py            # regenerates Figure 8
python code/make_packet_figure.py
python code/s6_heat_cavity_convergence.py --from-summary
```
