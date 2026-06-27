"""Discrete inf-sup stability of the boundary B-spline mimetic Galerkin wall pairing.

The Galerkin velocity-pressure pairing on a clamped (walled) box is inf-sup stable:
its discrete inf-sup constant in the natural spline metric is bounded below
independently of the mesh, and equals the continuous-domain constant pi, with the
physical constant pressure as its only null mode. Equal-order maximally-smooth
B-splines therefore avoid the spurious checkerboard that forces C0 equal-order
(Q1-Q1) pairs to use a mixed-degree (Taylor-Hood) pressure or a stabilization.

The inf-sup constant is

    beta_h = inf_{q, mean-zero} sup_{v} (q, div v) / ( ||v||_{M_v} ||q||_{M_p} ),

measured in the natural spline mass metrics M_v, M_p (the metrics the Galerkin
scheme actually uses). Writing the per-direction pressure-divergence couplings
C = <q, v'> and the cross mass Mxz = <q, v>, the 2D Schur complement is

    A = (C M_v^{-1} C^T) (x) (Mxz M_v^{-1} Mxz^T) + (Mxz M_v^{-1} Mxz^T) (x) (C M_v^{-1} C^T),

and beta_h^2 is the smallest non-constant generalized eigenvalue of A against
M_p (x) M_p. This reproduces the solver's tested mms_infsup result.

A second, metric-free check reports the kernel of the unconstrained pressure
gradient: it is exactly the constants for every degree and grid on both branches,
because the gradient is the exact spline derivative. The collocated same-point
reconstruction derivative B'B^{-1} is not used for the pressure solve; that nodal
route carries the Nyquist/checkerboard null, which is why the conservative routes
are the Galerkin natural-metric solve and the staggered BMF flux complex.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline
from scipy.linalg import eigh

from boundary_spline_analysis import bmf_staggered_operators, bmg_nodal_operator
from project_paths import RESULTSDIR


def _open_uniform_knots(n_elem: int, degree: int, length: float = 1.0) -> np.ndarray:
    inner = np.linspace(0.0, length, n_elem + 1)
    return np.concatenate([np.full(degree, 0.0), inner, np.full(degree, length)])


def _basis(knots: np.ndarray, degree: int):
    nb = knots.size - degree - 1
    out = []
    for i in range(nb):
        c = np.zeros(nb)
        c[i] = 1.0
        out.append(BSpline(knots, c, degree, extrapolate=False))
    return out, nb


def _direction_matrices(n_elem: int, pv: int, pq: int, quad: int | None = None):
    """Velocity mass M_v, pressure mass M_p, coupling C=<q,v'>, cross mass Mxz=<q,v>."""
    quad = quad or (2 * pv + 2)
    tp = _open_uniform_knots(n_elem, pv)
    tq = _open_uniform_knots(n_elem, pq)
    bv, nv = _basis(tp, pv)
    bp, npr = _basis(tq, pq)
    uk = np.unique(np.concatenate([tp, tq]))
    xi, wi = leggauss(quad)
    mv = np.zeros((nv, nv))
    mp = np.zeros((npr, npr))
    cmat = np.zeros((npr, nv))
    mxz = np.zeros((npr, nv))
    for a, b in zip(uk[:-1], uk[1:]):
        if b <= a + 1.0e-13:
            continue
        x = 0.5 * (b - a) * xi + 0.5 * (a + b)
        w = 0.5 * (b - a) * wi
        vv = np.nan_to_num(np.stack([f(x) for f in bv]))
        vd = np.nan_to_num(np.stack([f.derivative(1)(x) for f in bv]))
        pp = np.nan_to_num(np.stack([f(x) for f in bp]))
        mv += vv @ np.diag(w) @ vv.T
        mp += pp @ np.diag(w) @ pp.T
        cmat += pp @ np.diag(w) @ vd.T
        mxz += pp @ np.diag(w) @ vv.T
    return mv, mp, cmat, mxz


def infsup_beta(n_elem: int, pv: int, pq: int) -> tuple[float, int]:
    """Natural-metric Galerkin inf-sup constant and the number of near-null modes."""
    mv, mp, cmat, mxz = _direction_matrices(n_elem, pv, pq)
    mvi = np.linalg.inv(mv)
    p = cmat @ mvi @ cmat.T
    q = mxz @ mvi @ mxz.T
    a = np.kron(p, q) + np.kron(q, p)
    mp2 = np.kron(mp, mp)
    ev = np.sort(np.clip(eigh(0.5 * (a + a.T), 0.5 * (mp2 + mp2.T), eigvals_only=True), 0.0, None))
    scale = ev.max() if ev.max() > 0 else 1.0
    nnull = int((ev <= 1.0e-9 * scale).sum())
    nz = ev[ev > 1.0e-9 * scale]
    beta = float(np.sqrt(nz.min())) if nz.size else 0.0
    return beta, nnull


def grad_kernel_full(branch: str, npts: int, degree: int, tol: float = 1.0e-9) -> int:
    eye = np.eye(npts)
    if branch == "bmg":
        _x, dmat, _h = bmg_nodal_operator(npts, degree)
        g = np.vstack([np.kron(dmat, eye), np.kron(eye, dmat)])
    else:
        op = bmf_staggered_operators(npts, degree)
        g = np.vstack([np.kron(op.d10, eye), np.kron(eye, op.d10)])
    s = np.linalg.svd(g, compute_uv=False)
    return g.shape[1] - int((s > tol * s[0]).sum())


def main() -> None:
    out = {"infsup_beta": [], "grad_kernel_full": []}
    print("Natural-metric Galerkin inf-sup constant beta_h on a clamped box (2D):")
    print(f"{'pv':>3} {'pq':>3} {'n_elem':>7} {'beta_h':>10} {'near_null':>10}")
    for pv in (2, 3, 4):
        for pq in (pv, pv - 1):
            for ne in (6, 8, 10, 12, 16, 24):
                beta, nn = infsup_beta(ne, pv, pq)
                out["infsup_beta"].append({"pv": pv, "pq": pq, "n_elem": ne, "beta": beta, "near_null": nn})
                tag = "equal" if pq == pv else "reduced"
                print(f"{pv:>3} {pq:>3} {ne:>7} {beta:>10.4f} {nn:>10d}   [{tag}]")
    print(f"\npi = {np.pi:.6f}  (the continuous-domain inf-sup constant)")

    print("\nUnconstrained pressure-gradient kernel (1 = constant only, no spurious modes):")
    print(f"{'p':>3} {'N':>4} {'BMF':>5} {'BMG':>5}")
    for p in (2, 3, 4):
        for n in (9, 13, 17, 25):
            kf = grad_kernel_full("bmf", n, p)
            kg = grad_kernel_full("bmg", n, p)
            out["grad_kernel_full"].append({"degree": p, "npts": n, "bmf": kf, "bmg": kg})
            print(f"{p:>3} {n:>4} {kf:>5} {kg:>5}")

    dest = Path(RESULTSDIR) / "infsup_gradient_kernel.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
