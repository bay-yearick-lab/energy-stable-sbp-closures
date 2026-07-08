#!/usr/bin/env python3
"""Driver-level verification of the diffusion dt_max claims in
code/results/spectral_radius_table.json.

For each verified family, the ACTUAL heat-cavity stepper
(`heat_cavity_core.rk4_step`, which advances Lap U = D2 @ U + U @ D2^T and
pins the walls to zero after every stage) is run for 10,000 steps at
0.999 * dt_max_claimed (expected: bounded, non-growing solution) and at
1.05 * dt_max_claimed (expected: blow-up). Initial data is seeded random
interior noise so every grid mode, including the stiffest boundary-cluster
modes, is excited. The spectral cross-check max_i |R(dt*lambda_i)| over the
exact pairwise-sum (Kronecker) spectrum is reported alongside so the
empirical result can be compared against the eigenvalue prediction.

viscosity nu = 1.0 so the driver dt equals the operator dt directly.

Output: code/results/spectral_radius_verify.json plus stdout table.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.linalg import eigvals

import boundary_spline_analysis as ba
from heat_cavity_core import diffusion_second_derivative, rk4_step
from project_paths import RESULTSDIR, ensure_generated_dirs

ensure_generated_dirs()

NPTS = 65
NSTEPS = 10_000
BLOWUP_CUTOFF = 1.0e12
LABELS = ('E6', 'SBP6', 'BMF6', 'BMG6')

_RK4_COEFFS = np.array([1.0 / 24.0, 1.0 / 6.0, 1.0 / 2.0, 1.0, 1.0])

TABLE = json.loads((RESULTSDIR / 'spectral_radius_table.json').read_text())


def run_case(label: str, dt: float, rng: np.random.Generator) -> dict[str, float | int | bool]:
    op = ba.build_operator(ba.scheme_kind(label), ba.scheme_order(label), NPTS)
    D2 = diffusion_second_derivative(op, label)
    U = rng.standard_normal((NPTS, NPTS))
    U[0, :] = U[-1, :] = 0.0
    U[:, 0] = U[:, -1] = 0.0
    U /= np.max(np.abs(U))
    peak = 1.0
    steps_done = 0
    for step in range(NSTEPS):
        U = rk4_step(D2, U, dt, 1.0)
        steps_done = step + 1
        m = float(np.max(np.abs(U)))
        peak = max(peak, m)
        if not np.isfinite(m) or m > BLOWUP_CUTOFF:
            break
    final = float(np.max(np.abs(U)))
    return {
        'dt': dt,
        'steps_completed': steps_done,
        'final_max_abs': final if np.isfinite(final) else float('inf'),
        'peak_max_abs': peak if np.isfinite(peak) else float('inf'),
        'blew_up': bool(not np.isfinite(final) or final > BLOWUP_CUTOFF),
    }


def spectral_check(label: str, dt: float) -> float:
    op = ba.build_operator(ba.scheme_kind(label), ba.scheme_order(label), NPTS)
    D2i = diffusion_second_derivative(op, label)[1:-1, 1:-1]
    lam1 = eigvals(D2i)
    lam2 = (lam1[:, None] + lam1[None, :]).ravel()
    return float(np.max(np.abs(np.polyval(_RK4_COEFFS, dt * lam2))))


def main() -> None:
    rng = np.random.default_rng(0)
    out: dict[str, object] = {'npts': NPTS, 'nsteps': NSTEPS, 'viscosity': 1.0, 'cases': {}}
    print(f'{"label":>6s} {"factor":>7s} {"dt":>12s} {"max|R|":>10s} {"steps":>6s} {"final max|U|":>14s} {"verdict":>8s}')
    for label in LABELS:
        dt_max = float(TABLE['families'][label][str(NPTS)]['diffusion']['dt_max_rk4'])
        for factor, expect_blowup in ((0.999, False), (1.05, True)):
            dt = factor * dt_max
            amp = spectral_check(label, dt)
            res = run_case(label, dt, rng)
            verdict = 'BLOWUP' if res['blew_up'] else 'stable'
            ok = res['blew_up'] == expect_blowup
            out['cases'][f'{label}@{factor}'] = {
                **res,
                'factor': factor,
                'dt_max_claimed': dt_max,
                'max_rk4_amplification': amp,
                'expected_blowup': expect_blowup,
                'matches_expectation': ok,
            }
            flag = '' if ok else '  <-- UNEXPECTED'
            print(f'{label:>6s} {factor:7.3f} {dt:12.5e} {amp:10.6f} {res["steps_completed"]:6d} '
                  f'{res["final_max_abs"]:14.4e} {verdict:>8s}{flag}')
    out_path = RESULTSDIR / 'spectral_radius_verify.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
