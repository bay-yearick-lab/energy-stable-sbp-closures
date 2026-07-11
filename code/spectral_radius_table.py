#!/usr/bin/env python3
"""Spectral-radius / max-stable-RK4-step table for the six matched-order families.

The recurring-work proxy F_D P^{d+1} used for the
mission-design atlases (Figure 41, `code/s6_wall_mission_design.py`) counts
flops per point and points per wavelength but never counts time steps. The
clamped Greville grid used by the two spline families (BMF, BMG) clusters
nodes near each wall at spacing ~h/p, so an explicit RK4 integrator could in
principle pay for that clustering through the CFL restriction. This script
measures the actual price on the operators the paper actually marches.

WHAT IS MEASURED:

(a) Hyperbolic missions: the COUPLED pressure-velocity wave pair that
    `boundary_spline_analysis.simulate_reflected_acoustics` actually
    advances, not the bare first-derivative matrix. For the same-grid
    families (E, C, SBP, DRP, BMG, SBP6-BO) the marched system is
        dp/dt = -D u,   du/dt = -D p,
    with pressure advanced at ALL N nodes and the velocity trace pinned at
    both walls (u_0 = u_{N-1} = boundary data; homogeneous for the
    eigenvalue test). The semidiscrete block generator is therefore
        A = [[0, -D[:, 1:-1]], [-D[1:-1, :], 0]],
    of size (2N-2). For BMF the marched system is the staggered pair of
    `_simulate_reflected_acoustics_bmf_staggered`: pressure on the primal
    Greville grid (n_p nodes), velocity on the derivative Greville grid
    (n_v nodes) with its two end values pinned,
        A = [[0, -D01[:, 1:-1]], [-G10[1:-1, :], 0]],
    of size (n_p + n_v - 2), where G10 = ops.d10 (BMF gradient) and
    D01 = ops.d01 (BMF divergence).

(b) Parabolic missions: the 2D tensor-product Laplacian the heat-cavity
    driver actually advances, Lap U = D2 @ U + U @ D2^T on the interior
    (walls pinned to zero by `enforce_zero_walls` after every stage). The
    spectrum of the Kronecker sum D2 (+) D2 is exactly the set of pairwise
    sums {lambda_i + lambda_j} of the interior-restricted 1D eigenvalues
    (Schur-triangularization argument, exact, no diagonalizability needed),
    so the 2D stability limit is computed on that pairwise-sum set. For a
    real spectrum this halves the 1D dt_max; the ratio between families is
    essentially unchanged, but the absolute numbers quoted anywhere must be
    the 2D ones because that is the operator the driver steps. The 1D
    interior operators are `heat_cavity_core.diffusion_second_derivative`
    (native second_derivative_matrix() for SBP/BMF/BMG/SBP6-BO, composed
    D@D for E/C/DRP), imported directly so exactly one place in the
    repository encodes "how does family X form a Laplacian".

RK4 STABILITY TEST: exact bisection on
    max_i |R(dt * lambda_i)| <= 1 + tol,     R(z)=1+z+z^2/2+z^3/6+z^4/24,
over the FULL eigenvalue set from scipy.linalg.eigvals (dense, exact). The
tolerance (1e-3) exists because several finite-difference wave pairs carry a
genuinely positive spectral abscissa from their one-sided closure rows (the
long-time cliff subsection of the manuscript documents the same abscissas);
for those families no dt satisfies |R| <= 1 strictly, and dt_max is then the
largest step keeping growth below 0.1% per step. Families whose wave pair is
exactly skew-symmetric (all SBP-identity carriers: SBP, SBP6-BO, BMF, BMG;
abscissa ~1e-14) are stable in the strict |R| <= 1 sense and their dt_max
sits at the imaginary-axis RK4 bound 2*sqrt(2)/rho. The JSON records the
abscissa and a `tolerance_limited` flag so the two regimes cannot be
confused downstream.

Both N=65 and N=129 are tabulated as a scaling check; the dt_max ratios vs
E at matched order are grid-independent to the digits reported.

Output: code/results/spectral_radius_table.json
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.linalg import eigvals

import boundary_spline_analysis as ba
from heat_cavity_core import diffusion_second_derivative, diffusion_second_derivative_extended
from project_paths import RESULTSDIR, ensure_generated_dirs

ensure_generated_dirs()

ORDER_BLOCKS = (4, 6, 8)
GRID_SIZES = (65, 129)
FAMILY_ORDER = ba.family_order()  # ('E', 'C', 'SBP', 'DRP', 'BMF', 'BMG')
RK4_TOL = 1.0e-3

# RK4 stability polynomial R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24 (highest degree first).
_RK4_COEFFS = np.array([1.0 / 24.0, 1.0 / 6.0, 1.0 / 2.0, 1.0, 1.0])


def rk4_amplification(z: np.ndarray) -> np.ndarray:
    return np.polyval(_RK4_COEFFS, z)


def rk4_max_stable_dt(eigenvalues: np.ndarray, *, dt_hi_seed: float, tol: float = RK4_TOL, iters: int = 60) -> float:
    """Largest dt of the CONNECTED stable interval [0, dt_max].

    dt_max is the first up-crossing of max_i |R(dt*lambda_i)| above 1+tol.
    For operators with a positive spectral abscissa the RK4-stable set in dt
    can be disconnected (an eigenvalue with positive real part and large
    imaginary part re-enters the stability region at larger dt through the
    imaginary-axis damping dip); a naive bisection between 0 and a large
    seed can then converge to the boundary of a disconnected island rather
    than the boundary of the interval containing 0. CFL accounting wants
    the connected boundary (the largest dt such that every smaller dt is
    also stable), so the search is: log-spaced scan to bracket the FIRST
    unstable dt, then bisection inside that bracket.
    """
    def stable(dt: float) -> bool:
        return bool(np.all(np.abs(rk4_amplification(dt * eigenvalues)) <= 1.0 + tol))

    dt_hi = float(dt_hi_seed) if dt_hi_seed > 0.0 else 1.0e-6
    growth = 0
    while stable(dt_hi) and growth < 80:
        dt_hi *= 1.6
        growth += 1
    if stable(dt_hi):
        return dt_hi

    # Bracket the first up-crossing on a dense log grid spanning 8 decades
    # below the known-unstable dt_hi.
    grid = np.geomspace(dt_hi * 1.0e-8, dt_hi, 4001)
    dt_lo = 0.0
    bracket_hi = dt_hi
    for dt in grid:
        if stable(float(dt)):
            dt_lo = float(dt)
        else:
            bracket_hi = float(dt)
            break

    for _ in range(iters):
        dt_mid = 0.5 * (dt_lo + bracket_hi)
        if stable(dt_mid):
            dt_lo = dt_mid
        else:
            bracket_hi = dt_mid
    return dt_lo


def build_1d_operator(label: str, npts: int):
    if label == 'SBP6-BO':
        return ba.build_operator_extended('SBP6-BO', 6, npts)
    return ba.build_operator(ba.scheme_kind(label), ba.scheme_order(label), npts)


def coupled_wave_pair_matrix(label: str, npts: int) -> tuple[np.ndarray, dict[str, object]]:
    """Assemble the coupled acoustic generator each family actually marches.

    Mirrors `simulate_reflected_acoustics` exactly: pressure free at every
    node, velocity trace pinned at both walls. For BMF, mirrors
    `_simulate_reflected_acoustics_bmf_staggered`: primal pressure free,
    derivative-grid velocity pinned at its two end nodes.
    """
    op = build_1d_operator(label, npts)
    fam = 'SBP' if label == 'SBP6-BO' else ba.scheme_family(label)

    def horth_reduced(Afull, metric_v, nv, offset):
        # metric-orthogonal enforcement (Definition def:strong-enforcement):
        # generator of the projected flow restricted to the constraint
        # manifold {v_0 = v_end = 0}, via an SVD null basis of the trace rows
        ntot = Afull.shape[0]
        L, XH = ba._horth_wave_lifting(metric_v, nv, offset, ntot)
        Aenf = (np.eye(ntot) - XH @ L) @ Afull
        _u, _s, Vt = np.linalg.svd(L)
        Zb = Vt[2:, :].T
        return Zb.T @ Aenf @ Zb

    if fam == 'BMF':
        ops = op.staggered_operators()
        n_p = ops.x.size
        n_v = ops.eta.size
        Afull = np.zeros((n_p + n_v, n_p + n_v))
        Afull[:n_p, n_p:] = -ops.d01
        Afull[n_p:, :n_p] = -ops.d10
        A = horth_reduced(Afull, ops.h1, n_v, n_p)
        h_min = min(float(np.min(np.diff(ops.x))), float(np.min(np.diff(ops.eta))))
        h_max = max(float(np.max(np.diff(ops.x))), float(np.max(np.diff(ops.eta))))
        meta = {'route': 'staggered BMF pair (d01/d10), metric-orthogonal enforcement',
                'h_min': h_min, 'h_max': h_max}
        return A, meta
    D = op.matrix
    n = D.shape[0]
    dx = np.diff(op.x)
    if fam == 'BMG':
        metric = ba.bmg_nodal_operator(npts, ba.scheme_order(label))[2]
        Afull = np.zeros((2 * n, 2 * n))
        Afull[:n, n:] = -D
        Afull[n:, :n] = -D
        A = horth_reduced(Afull, metric, n, n)
        meta = {'route': 'same-grid pair (D), metric-orthogonal enforcement',
                'h_min': float(np.min(dx)), 'h_max': float(np.max(dx))}
        return A, meta
    A = np.zeros((2 * n - 2, 2 * n - 2))
    A[:n, n:] = -D[:, 1:-1]
    A[n:, :n] = -D[1:-1, :]
    meta = {'route': 'same-grid pair (D)', 'h_min': float(np.min(dx)), 'h_max': float(np.max(dx))}
    return A, meta


def diffusion_interior_matrix(label: str, npts: int) -> np.ndarray:
    op = build_1d_operator(label, npts)
    if label == 'SBP6-BO':
        D2 = diffusion_second_derivative_extended(op, 'SBP6-BO')
    else:
        D2 = diffusion_second_derivative(op, label)
    return D2[1:-1, 1:-1]


def analyze_wave_pair(label: str, npts: int) -> dict[str, object]:
    A, meta = coupled_wave_pair_matrix(label, npts)
    lam = eigvals(A)
    rho = float(np.max(np.abs(lam)))
    abscissa = float(np.max(lam.real))
    dt_max = rk4_max_stable_dt(lam, dt_hi_seed=4.0 * 2.83 / rho)
    return {
        'operator': meta['route'],
        'state_size': int(A.shape[0]),
        'h_min': meta['h_min'],
        'h_max': meta['h_max'],
        'spectral_radius': rho,
        'spectral_abscissa': abscissa,
        'dt_max_rk4': dt_max,
        'tolerance_limited': bool(abscissa > 1.0e-8),
    }


def analyze_diffusion_2d(label: str, npts: int) -> dict[str, object]:
    D2i = diffusion_interior_matrix(label, npts)
    lam1 = eigvals(D2i)
    # Kronecker-sum spectrum of the 2D tensor Laplacian: all pairwise sums.
    lam2 = (lam1[:, None] + lam1[None, :]).ravel()
    rho1 = float(np.max(np.abs(lam1)))
    rho2 = float(np.max(np.abs(lam2)))
    abscissa2 = float(np.max(lam2.real))
    dt_max = rk4_max_stable_dt(lam2, dt_hi_seed=4.0 * 2.785 / rho2)
    return {
        'operator': '2D tensor Laplacian D2(+)D2, interior-restricted (pairwise-sum spectrum)',
        'spectral_radius_1d': rho1,
        'spectral_radius_2d': rho2,
        'spectral_abscissa_2d': abscissa2,
        'dt_max_rk4': dt_max,
        'tolerance_limited': bool(abscissa2 > 1.0e-8),
    }


def build_table() -> dict[str, object]:
    t_start = time.time()
    out: dict[str, object] = {
        'method': {
            'hyperbolic_operator': "coupled acoustic wave pair as marched by simulate_reflected_acoustics: "
                                    "A=[[0,-D[:,1:-1]],[-D[1:-1,:],0]] (pressure free at all nodes, velocity "
                                    "trace pinned); BMF uses the staggered d01/d10 pair of "
                                    "_simulate_reflected_acoustics_bmf_staggered with derivative-grid velocity "
                                    "end nodes pinned",
            'parabolic_operator': "2D tensor Laplacian D2@U+U@D2^T on the interior (enforce_zero_walls "
                                   "convention); spectrum computed exactly as pairwise sums of the "
                                   "interior-restricted 1D eigenvalues (Kronecker-sum spectrum); 1D D2 from "
                                   "heat_cavity_core.diffusion_second_derivative[_extended]",
            'rk4_stability_test': "exact bisection on max_i |R(dt*lambda_i)| <= 1+tol over the full eigvals "
                                   "set, R(z)=1+z+z^2/2+z^3/6+z^4/24; tol=1e-3 (needed only by the "
                                   "positive-abscissa finite-difference wave pairs, where dt_max means "
                                   "growth <= 0.1% per step; SBP/BMF/BMG wave pairs are exactly skew and "
                                   "stable in the strict sense, see tolerance_limited flags)",
            'tol': RK4_TOL,
            'grids': list(GRID_SIZES),
            'order_blocks': list(ORDER_BLOCKS),
        },
        'families': {},
    }

    baseline_dt: dict[tuple[int, int, str], float] = {}
    all_labels_by_q: dict[int, list[str]] = {q: [f'{fam}{q}' for fam in FAMILY_ORDER] for q in ORDER_BLOCKS}
    all_labels_by_q[6].append('SBP6-BO')

    for q in ORDER_BLOCKS:
        for label in all_labels_by_q[q]:
            for npts in GRID_SIZES:
                adv = analyze_wave_pair(label, npts)
                diff = analyze_diffusion_2d(label, npts)
                entry = {
                    'family': 'SBP' if label == 'SBP6-BO' else ba.scheme_family(label),
                    'q_or_p': q,
                    'npts': npts,
                    'h_min': adv['h_min'],
                    'h_max': adv['h_max'],
                    'clustering_ratio_h_max_over_h_min': adv['h_max'] / adv['h_min'],
                    'advection': adv,
                    'diffusion': diff,
                }
                out['families'].setdefault(label, {})[str(npts)] = entry
                if label == f'E{q}':
                    baseline_dt[(q, npts, 'advection')] = adv['dt_max_rk4']
                    baseline_dt[(q, npts, 'diffusion')] = diff['dt_max_rk4']
                print(f'{label:>9s}  q={q}  N={npts:4d}  '
                      f'wave rho={adv["spectral_radius"]:.4e} absc={adv["spectral_abscissa"]:+.2e} dt={adv["dt_max_rk4"]:.4e}  '
                      f'diff2d rho={diff["spectral_radius_2d"]:.4e} dt={diff["dt_max_rk4"]:.4e}',
                      flush=True)

    for q in ORDER_BLOCKS:
        for label in all_labels_by_q[q]:
            for npts in GRID_SIZES:
                entry = out['families'][label][str(npts)]
                for mode in ('advection', 'diffusion'):
                    base = baseline_dt.get((q, npts, mode))
                    entry[mode]['dt_max_ratio_vs_E'] = (entry[mode]['dt_max_rk4'] / base) if base else float('nan')

    out['elapsed_seconds'] = time.time() - t_start
    return out


def main() -> None:
    table = build_table()
    out_path = RESULTSDIR / 'spectral_radius_table.json'
    out_path.write_text(json.dumps(table, indent=2))
    print(f'wrote {out_path}')

    print()
    print('Summary: dt_max ratio vs E at matched order, N=65')
    for q in ORDER_BLOCKS:
        print(f'  q={q}:')
        labels = [f'{fam}{q}' for fam in FAMILY_ORDER]
        if q == 6:
            labels.append('SBP6-BO')
        for label in labels:
            e = table['families'][label]['65']
            skew = 'skew' if not e['advection']['tolerance_limited'] else 'absc>0'
            print(f'    {label:>9s}  wave_dt_ratio={e["advection"]["dt_max_ratio_vs_E"]:9.3f} ({skew})  '
                  f'diff2d_dt_ratio={e["diffusion"]["dt_max_ratio_vs_E"]:8.5f}  '
                  f'clustering={e["clustering_ratio_h_max_over_h_min"]:.3f}')


if __name__ == '__main__':
    main()
