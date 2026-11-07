#!/usr/bin/env python3
"""One-command verification of the headline numbers.

Recomputes, from the operator constructors in this directory:

1. Summation-by-parts residuals. For BMG (same-grid Galerkin derivative D in
   the spline mass metric H) the identity is H D + D^T H = B with
   B = diag(-1, 0, ..., 0, 1); for the staggered BMF pair it is
   H0 D01 + D10^T H1 = B. Both are printed as max-abs residuals, p = 2..8.
2. The discrete inf-sup constant beta_h of the divergence-conforming BMF
   pair on the unit square, p = 2..8, 4/8/16 elements per direction, with
   |beta_h/pi - 1| and the pressure null count (the constant only).
3. Convergence orders on the wall-bounded heat cavity (the configuration of
   s6_heat_cavity_convergence.py), q = 4, 6, 8, for BMF, BMG and classical
   SBP-SAT. The slope is fitted on the finest half of the N = 33..129 ladder,
   which is N = 81, 97, 129, so only those grids are marched.

Usage:  python code/verify.py [--quick]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import boundary_spline_analysis as bsa  # noqa: E402
from bmf_hdiv_projection import BMFHDivProjection2D  # noqa: E402


def sbp_residuals(degree: int, npts: int) -> tuple[float, float]:
    """Return max-abs SBP residuals (BMG, staggered BMF) at one degree and grid."""
    _x, dmat, hmat = bsa.bmg_nodal_operator(npts, degree)
    boundary = np.zeros((npts, npts))
    boundary[0, 0], boundary[-1, -1] = -1.0, 1.0
    bmg = float(np.abs(hmat @ dmat + dmat.T @ hmat - boundary).max())
    bmf = float(np.abs(bsa.bmf_staggered_sbp_residual(npts, degree)).max())
    return bmg, bmf


def infsup(degree: int, elements: int) -> tuple[float, int]:
    """Return beta_h and the pressure null count for the BMF pair."""
    core = BMFHDivProjection2D(elements + degree, degree)
    return core.beta(), core.kernel_certificate().nullity


def heat_slope(label: str, grids: tuple[int, ...]) -> tuple[float, float]:
    """Return the fitted convergence order and the finest-grid relative L2 error."""
    import s6_heat_cavity_convergence as hc

    runs = [hc._run(label, n, 0.0) for n in grids]
    h = np.array([r["h"] for r in runs])
    err = np.array([r["rel_l2_final"] for r in runs])
    slope = float(np.polyfit(np.log(h), np.log(err), 1)[0])
    return slope, float(err[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="skip the q = 8 heat-cavity ladder")
    args = parser.parse_args()
    t0 = time.perf_counter()
    ok = True

    print("1. Summation-by-parts identity, max |H D + D^T H - B|")
    print(f"   {'p':>2} {'N':>4} {'BMG':>10} {'BMF':>10}")
    worst = 0.0
    for p in range(2, 9):
        npts = 4 * p + 9
        bmg, bmf = sbp_residuals(p, npts)
        worst = max(worst, bmg, bmf)
        print(f"   {p:>2} {npts:>4} {bmg:10.1e} {bmf:10.1e}")
    ok &= worst < 1e-12

    print("\n2. Inf-sup constant of the divergence-conforming BMF pair (limit pi)")
    print(f"   {'p':>2} {'elements':>8} {'beta_h':>18} {'|beta_h/pi-1|':>14} {'nulls':>5}")
    for p in range(2, 9):
        for el in (4, 8, 16):
            beta, nulls = infsup(p, el)
            rel = abs(beta / np.pi - 1.0)
            ok &= nulls == 1 and beta >= np.pi * (1.0 - 1e-13)
            print(f"   {p:>2} {el:>8} {beta:18.15f} {rel:14.2e} {nulls:>5}")

    print("\n3. Heat-cavity convergence order (fit on N = 81, 97, 129)")
    print(f"   {'q':>2} {'family':>6} {'order':>6} {'rel L2 at N=129':>16}")
    orders = (4, 6) if args.quick else (4, 6, 8)
    for q in orders:
        for fam in ("BMF", "BMG", "SBP"):
            slope, err = heat_slope(f"{fam}{q}", (81, 97, 129))
            if fam != "SBP":
                ok &= slope > q - 0.6
            print(f"   {q:>2} {fam:>6} {slope:6.2f} {err:16.2e}", flush=True)

    print(f"\n{'PASS' if ok else 'FAIL'} in {time.perf_counter() - t0:.0f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
