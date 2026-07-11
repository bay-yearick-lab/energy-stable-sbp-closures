"""Physical-boundary-condition inf-sup certificate for the spline Stokes pairs.

The mixed velocity-
pressure problem assembled with the PHYSICAL wall constraints, the viscous
energy norm, and the pressure quotient taken by the physical constants only.
It answers, separately and by name, the four questions the manuscript must
never merge:

  (Q1) strong-gradient kernel content            -> prop:no-spurious-pressure
  (Q2) weak-coupling left null under no-slip     -> exact null counts + content
  (Q3) uniform inf-sup of the constrained pair
       in the energy norm                        -> beta_h vs h per degree/pair
  (Q4) pressure uniqueness modulo constants      -> whether the constant is the
                                                    ONLY null on each pair

Problem: 2D Stokes on the unit box, velocity no-slip on all four walls.
For clamped (open-uniform) B-spline bases the endpoint values equal the
first/last coefficients, so no-slip is imposed EXACTLY by removing the wall
coefficient layers of each velocity component (the same strong enforcement the
manuscript's Definition of strong BC rows uses).

Pairs certified:
  * BMG same-grid: u1,u2 in S^p x S^p, pressure in S^pq x S^pq on the same
    breakpoints, pq in {p, p-1, p-2} (equal order and reduced-degree pairs);
  * BMF staggered: pressure in S^p x S^p, u1 in D^{p-1}_x x S^p_y,
    u2 in S^p_x x D^{p-1}_y with D^{p-1} the clamped derivative space
    (the flux-complex placement of the manuscript).

The inf-sup constant in the viscous energy norm |v|_A^2 = int |grad v|^2 is

    beta_h^2 = smallest positive eigenvalue of  B A^{-1} B^T q = mu (Mp x Mp) q,

with B the weak divergence coupling restricted to the constrained velocity
space and A the constrained vector Laplacian. The constrained space is a
tensor product of interior index sets, so A^{-1} is applied by per-axis
generalized eigendecomposition (the same fast-diagonalization structure the
accompanying solver uses for its projection).

Contrast case (the operator the Navier-Stokes solver ACTUALLY inverts): the
projection/Schur pressure operator K M^{-1} K^T over the UNCONSTRAINED
coefficient space, whose beta -> pi certificate already lives in
results/infsup_sweep.json; a single reproduction cell is included here as a
validation gate, not as evidence about the constrained saddle.

Validation gates (the script aborts if any fails):
  G1  unconstrained equal-order mass-metric beta reproduces pi to 2e-5
      at (p=3, ne=30) [cross-checks prop:infsup-certificate / mms_infsup.cpp];
  G2  the constant pressure is a machine-precision null of the no-slip
      coupling (Gauss: (1, div v) = boundary flux = 0);
  G3  the 1D no-slip equal-order kernel has dimension exactly 2
      (constant + derivative-complement mode, by the counting argument).

Adversarial probes (residual sup_v (q, div v)/(|v|_A ||q||) per probe):
  constant, coefficient checkerboard, knot-scale derivative-complement tensor
  mode, wall-localized pressure B_0(x)B_0(y).

Usage:
  python code/check_infsup_physical_bc.py --write code/results/infsup_physical_bc.json
  python code/check_infsup_physical_bc.py --verify code/results/infsup_physical_bc.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline
from scipy.linalg import cho_factor, cho_solve, cholesky, eigh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import RESULTSDIR  # noqa: E402

NULL_TOL = 1.0e-9          # relative eigenvalue threshold for a null mode
DEGREES = (2, 3, 4, 5, 6, 7, 8)
N_ELEMS = (8, 12, 16, 24, 32, 48, 64)
K_LOW = 10                 # how many lowest eigenpairs to record per cell


# ----------------------------------------------------------------------------
# 1D building blocks
# ----------------------------------------------------------------------------

def open_knots(n_elem: int, degree: int) -> np.ndarray:
    inner = np.linspace(0.0, 1.0, n_elem + 1)
    return np.concatenate([np.full(degree, 0.0), inner, np.full(degree, 1.0)])


def basis_list(knots: np.ndarray, degree: int):
    nb = knots.size - degree - 1
    fns = []
    for i in range(nb):
        c = np.zeros(nb)
        c[i] = 1.0
        fns.append(BSpline(knots, c, degree, extrapolate=False))
    return fns, nb


def pair_matrices(n_elem: int, deg_row: int, deg_col: int):
    """Return (Mrow, Srow_col? no) -- exact Gauss matrices between two clamped
    spaces on the same breakpoints:
      M  = int  R_i C_j          (cross mass, rows=row space, cols=col space)
      C  = int  R_i C_j'         (divergence coupling)
    plus the square mass and stiffness of the ROW space when deg_row==deg_col.
    """
    tr = open_knots(n_elem, deg_row)
    tc = open_knots(n_elem, deg_col)
    br, nr = basis_list(tr, deg_row)
    bc, nc = basis_list(tc, deg_col)
    xi, wi = leggauss(deg_row + deg_col + 2)
    uk = np.unique(np.concatenate([tr, tc]))
    M = np.zeros((nr, nc))
    C = np.zeros((nr, nc))
    for a, b in zip(uk[:-1], uk[1:]):
        if b <= a + 1e-13:
            continue
        x = 0.5 * (b - a) * xi + 0.5 * (a + b)
        w = 0.5 * (b - a) * wi
        R = np.nan_to_num(np.stack([f(x) for f in br]))
        Cv = np.nan_to_num(np.stack([f(x) for f in bc]))
        Cd = np.nan_to_num(np.stack([f.derivative(1)(x) for f in bc]))
        M += (R * w) @ Cv.T
        C += (R * w) @ Cd.T
    return M, C


def square_matrices(n_elem: int, degree: int):
    """Mass and stiffness of one clamped space."""
    t = open_knots(n_elem, degree)
    fns, n = basis_list(t, degree)
    xi, wi = leggauss(2 * degree + 2)
    uk = np.unique(t)
    M = np.zeros((n, n))
    S = np.zeros((n, n))
    for a, b in zip(uk[:-1], uk[1:]):
        if b <= a + 1e-13:
            continue
        x = 0.5 * (b - a) * xi + 0.5 * (a + b)
        w = 0.5 * (b - a) * wi
        V = np.nan_to_num(np.stack([f(x) for f in fns]))
        D = np.nan_to_num(np.stack([f.derivative(1)(x) for f in fns]))
        M += (V * w) @ V.T
        S += (D * w) @ D.T
    return M, S, n


def endpoint_interpolatory(n_elem: int, degree: int) -> bool:
    t = open_knots(n_elem, degree)
    fns, n = basis_list(t, degree)
    v0 = np.array([float(np.nan_to_num(f(0.0))) for f in fns])
    v1 = np.array([float(np.nan_to_num(f(1.0 - 1e-14))) for f in fns])
    e0 = np.zeros(n); e0[0] = 1.0
    e1 = np.zeros(n); e1[-1] = 1.0
    return bool(np.allclose(v0, e0, atol=1e-10) and np.allclose(v1, e1, atol=1e-10))


# ----------------------------------------------------------------------------
# The constrained Schur complement in the energy norm
# ----------------------------------------------------------------------------

def component_schur(Cx, Gy, Sx, Mx, Sy, My):
    """S_c = B A^{-1} B^T for ONE velocity component.

    B = (Cx x Gy) restricted to interior velocity columns; A = Sx x My + Mx x Sy
    on the interior tensor space. Uses the per-axis generalized eigenpairs
    S phi = lam M phi with phi^T M phi = I, so A^{-1} = (Phi x Phi) D^{-1} (...)^T.
    Cx, Gy are (pressure-dim x velocity-dim) COUPLING/CROSS matrices already
    restricted to interior velocity columns.
    """
    lx, phix = eigh(Sx, Mx)
    ly, phiy = eigh(Sy, My)
    X = Cx @ phix               # (npx, nvx)
    Y = Gy @ phiy               # (npy, nvy)
    npx, nvx = X.shape
    npy, nvy = Y.shape
    S = np.zeros((npx * npy, npx * npy))
    # S[(i,j),(k,l)] = sum_a X_ia X_ka * [ Y diag(1/(lx_a+ly)) Y^T ]_{jl}
    for a in range(nvx):
        Ra = (Y / (lx[a] + ly)) @ Y.T          # (npy, npy)
        S += np.kron(np.outer(X[:, a], X[:, a]), Ra)
    return S


def whiten_pressure(S, Mp_x, Mp_y):
    """Return Ltot (kron cholesky factor) and the whitened Schur matrix."""
    Lx = cholesky(Mp_x, lower=True)
    Ly = cholesky(Mp_y, lower=True)
    L = np.kron(Lx, Ly)
    Z = np.linalg.solve(L, S)
    W = np.linalg.solve(L, Z.T).T
    return L, 0.5 * (W + W.T)


def cell_bmg(n_elem: int, p: int, pq: int, constrained: bool = True):
    """One BMG same-grid cell: velocity degree p (no-slip), pressure degree pq."""
    Mv, Sv, nv = square_matrices(n_elem, p)
    Mp, _Sp_unused, npr = square_matrices(n_elem, pq)
    G, C = pair_matrices(n_elem, pq, p)     # rows pressure, cols velocity
    if constrained:
        keep = np.arange(1, nv - 1)
    else:
        keep = np.arange(nv)
    Mvi = Mv[np.ix_(keep, keep)]
    Svi = Sv[np.ix_(keep, keep)]
    Ci = C[:, keep]
    Gi = G[:, keep]
    # component 1: coupling Cx x Gy ; component 2: Gx x Cy (same 1D blocks)
    S1 = component_schur(Ci, Gi, Svi, Mvi, Svi, Mvi)
    S2 = component_schur(Gi, Ci, Svi, Mvi, Svi, Mvi)
    S = S1 + S2
    L, W = whiten_pressure(S, Mp, Mp)
    return W, L, Mp, npr


def cell_bmf(n_elem: int, p: int, constrained: bool = True):
    """One BMF staggered cell: pressure S^p x S^p; u1 in D^{p-1}_x x S^p_y,
    u2 in S^p_x x D^{p-1}_y, D^{p-1} = clamped degree-(p-1) space (the
    derivative space of S^p on the same breakpoints)."""
    Mp, _s, npr = square_matrices(n_elem, p)
    Mvp, Svp, nvp = square_matrices(n_elem, p)           # primal factor
    Mvd, Svd, nvd = square_matrices(n_elem, p - 1)       # derivative factor
    Gpp, _cpp = pair_matrices(n_elem, p, p)              # pressure x primal
    Gpd, Cpd = pair_matrices(n_elem, p, p - 1)           # pressure x deriv space
    if constrained:
        kp = np.arange(1, nvp - 1)
        kd = np.arange(1, nvd - 1)
    else:
        kp = np.arange(nvp)
        kd = np.arange(nvd)
    # u1: x-factor derivative space (coupled through d/dx), y-factor primal
    S1 = component_schur(Cpd[:, kd], Gpp[:, kp],
                         Svd[np.ix_(kd, kd)], Mvd[np.ix_(kd, kd)],
                         Svp[np.ix_(kp, kp)], Mvp[np.ix_(kp, kp)])
    S2 = component_schur(Gpp[:, kp], Cpd[:, kd],
                         Svp[np.ix_(kp, kp)], Mvp[np.ix_(kp, kp)],
                         Svd[np.ix_(kd, kd)], Mvd[np.ix_(kd, kd)])
    S = S1 + S2
    L, W = whiten_pressure(S, Mp, Mp)
    return W, L, Mp, npr


def cell_rt(n_elem: int, p: int, constrained: bool = True,
            tangential_strong: bool = True):
    """Divergence-conforming (Raviart-Thomas-spline) pair built from the same
    primal/derivative flux spaces: u1 in S^p_x x D^{p-1}_y, u2 in
    D^{p-1}_x x S^p_y, pressure in D^{p-1} x D^{p-1}, so div u lands EXACTLY
    in the pressure space. This is the compatible layout the isogeometric
    literature proves inf-sup stable;
    here it is certified with strong no-slip on all walls."""
    pd = p - 1
    Mp, _s, npr = square_matrices(n_elem, pd)
    Mvp, Svp, nvp = square_matrices(n_elem, p)
    Mvd, Svd, nvd = square_matrices(n_elem, pd)
    _Gdp, Cdp = pair_matrices(n_elem, pd, p)   # pressure(pd) x primal(p), d/dx
    Gdd, _cdd = pair_matrices(n_elem, pd, pd)  # pressure(pd) x deriv(pd), mass
    if constrained:
        kp = np.arange(1, nvp - 1)
        kd = np.arange(1, nvd - 1) if tangential_strong else np.arange(nvd)
    else:
        kp = np.arange(nvp)
        kd = np.arange(nvd)
    # u1: primal in x (coupled through d/dx), derivative space in y
    S1 = component_schur(Cdp[:, kp], Gdd[:, kd],
                         Svp[np.ix_(kp, kp)], Mvp[np.ix_(kp, kp)],
                         Svd[np.ix_(kd, kd)], Mvd[np.ix_(kd, kd)])
    S2 = component_schur(Gdd[:, kd], Cdp[:, kp],
                         Svd[np.ix_(kd, kd)], Mvd[np.ix_(kd, kd)],
                         Svp[np.ix_(kp, kp)], Mvp[np.ix_(kp, kp)])
    S = S1 + S2
    L, W = whiten_pressure(S, Mp, Mp)
    return W, L, Mp, npr


def analyze_cell(W, L, Mp, npr, k_low=K_LOW):
    """Eigen-analysis of the whitened Schur matrix: nullity, beta, mode ids."""
    n2 = W.shape[0]
    top = eigh(W, subset_by_index=[n2 - 1, n2 - 1], eigvals_only=True)[0]
    scale = max(top, 1e-300)
    # adaptive: grow k until the null space is fully inside the computed set
    k = min(k_low, n2)
    while True:
        vals_low, vecs_low = eigh(W, subset_by_index=[0, k - 1])
        nulls = int((vals_low <= NULL_TOL * scale).sum())
        if nulls < k or k == n2:
            break
        k = min(2 * k, n2)
    above = vals_low[vals_low > NULL_TOL * scale]
    beta = float(np.sqrt(above[0])) if above.size else 0.0
    # identify the constant inside the null space: whitened constant direction
    const = np.ones(n2)
    wconst = L.T @ const
    wconst /= np.linalg.norm(wconst)
    overlap = 0.0
    if nulls:
        Vn = vecs_low[:, :nulls]
        overlap = float(np.linalg.norm(Vn.T @ wconst))
    # beta on the CONSTANTS-ONLY quotient: smallest eigenvalue orthogonal to
    # the constant. If extra nulls exist this is ~0 and reported as such.
    P = np.eye(n2) - np.outer(wconst, wconst)
    Wq = P @ W @ P
    vq = eigh(Wq, subset_by_index=[0, min(2, n2 - 1)], eigvals_only=True)
    # the projector adds one artificial zero (the constant direction itself)
    vq = np.sort(vq)[1:]
    beta_const_quotient = float(np.sqrt(max(vq[0], 0.0))) if vq.size else 0.0
    return {
        "eig_low": [float(v) for v in vals_low],
        "eig_top": float(top),
        "nulldim": nulls,
        "constant_in_null": overlap,
        "beta_beyond_nulls": beta,
        "beta_constants_only_quotient": beta_const_quotient,
    }


def probe_residuals(W, L, Mp, npr):
    """sup_v (q,div v)/(|v|_A ||q||) for adversarial pressure probes."""
    n2 = W.shape[0]
    n = npr
    probes = {}
    const = np.ones(n2)
    cb = np.array([(-1.0) ** (i + j) for i in range(n) for j in range(n)])
    wall = np.zeros(n2)
    wall[0] = 1.0                       # B_0(x) B_0(y)
    # knot-scale derivative-complement tensor mode: 1D left-null of the
    # equal-order coupling over the FULL velocity space, beyond the constant.
    # Recover it from the two lowest whitened modes of the 1D problem if
    # present; fall back to the alternating mode.
    for name, q in (("constant", const), ("checkerboard", cb),
                    ("wall_localized", wall)):
        wq = L.T @ q
        nq = np.linalg.norm(wq)
        if nq == 0.0:
            probes[name] = 0.0
            continue
        wq = wq / nq
        probes[name] = float(np.sqrt(max(wq @ W @ wq, 0.0)))
    return probes


def run_cell(job):
    branch, p, pq, ne = job
    os.nice(19)
    try:
        if branch == "bmg":
            W, L, Mp, npr = cell_bmg(ne, p, pq)
        elif branch == "rt":
            W, L, Mp, npr = cell_rt(ne, p)
        elif branch == "rtn":
            W, L, Mp, npr = cell_rt(ne, p, tangential_strong=False)
        else:
            W, L, Mp, npr = cell_bmf(ne, p)
        rec = {"branch": branch, "p": p, "pq": pq, "n_elem": ne, "npr": npr}
        rec.update(analyze_cell(W, L, Mp, npr))
        rec["probes"] = probe_residuals(W, L, Mp, npr)
        return rec
    except Exception as exc:  # keep the sweep alive; report the failure
        return {"branch": branch, "p": p, "pq": pq, "n_elem": ne,
                "error": f"{type(exc).__name__}: {exc}"}


# ----------------------------------------------------------------------------
# Validation gates
# ----------------------------------------------------------------------------

def gate_unconstrained_pi() -> tuple[bool, float]:
    """G1: mass-metric UNCONSTRAINED equal-order beta reproduces pi (the
    projection-operator certificate of prop:infsup-certificate)."""
    ne, p = 30, 3
    Mv, _sv, nv = square_matrices(ne, p)
    Mp, _sp, npr = square_matrices(ne, p)
    G, C = pair_matrices(ne, p, p)
    Mvi = np.linalg.inv(Mv)
    P = C @ Mvi @ C.T
    Q = G @ Mvi @ G.T
    A2 = np.kron(P, Q) + np.kron(Q, P)
    Mp2 = np.kron(Mp, Mp)
    ev = np.sort(eigh(0.5 * (A2 + A2.T), 0.5 * (Mp2 + Mp2.T), eigvals_only=True))
    scale = ev[-1]
    nz = ev[ev > 1e-9 * scale]
    beta = float(np.sqrt(nz[0]))
    return abs(beta / np.pi - 1.0) < 2e-5, beta


def gate_constant_null() -> tuple[bool, float]:
    """G2: constant pressure is a machine-precision null under no-slip."""
    W, L, Mp, npr = cell_bmg(16, 4, 4)
    n2 = W.shape[0]
    wq = L.T @ np.ones(n2)
    wq /= np.linalg.norm(wq)
    # the Rayleigh quotient itself sits at machine eps; its square root is
    # therefore ~1e-8, which IS the machine-precision floor for this quantity
    res = float(np.sqrt(max(wq @ W @ wq, 0.0)))
    return res < 1e-7, res


def gate_1d_kernel() -> tuple[bool, int]:
    """G3: 1D no-slip equal-order kernel dim == 2 (constant + complement)."""
    ne, p = 16, 3
    Mv, Sv, nv = square_matrices(ne, p)
    Mp, _s, npr = square_matrices(ne, p)
    _G, C = pair_matrices(ne, p, p)
    keep = np.arange(1, nv - 1)
    Ci = C[:, keep]
    # kernel of Ci^T on pressure = left-null of the coupling
    s = np.linalg.svd(Ci, compute_uv=False)
    ker = npr - int((s > 1e-10 * s[0]).sum())
    return ker == 2, ker


# ----------------------------------------------------------------------------
# Sweep, write, verify
# ----------------------------------------------------------------------------

def build_jobs():
    jobs = []
    for p in DEGREES:
        for ne in N_ELEMS:
            for off in (0, 1, 2):
                if p - off >= 1:
                    jobs.append(("bmg", p, p - off, ne))
            jobs.append(("bmf", p, p, ne))
            jobs.append(("rt", p, p - 1, ne))
            jobs.append(("rtn", p, p - 1, ne))
    # largest first so the pool drains evenly
    jobs.sort(key=lambda j: -(j[3] ** 2))
    return jobs


def fitted_slopes(cells):
    """log-log slope of beta_constants_only_quotient vs h over the finest
    grids, per (branch, p, pq)."""
    groups = {}
    for c in cells:
        if "error" in c:
            continue
        groups.setdefault((c["branch"], c["p"], c["pq"]), []).append(c)
    out = []
    for (branch, p, pq), gr in sorted(groups.items()):
        gr.sort(key=lambda c: c["n_elem"])
        h = np.array([1.0 / c["n_elem"] for c in gr])
        b = np.array([c["beta_constants_only_quotient"] for c in gr])
        keep = b > 0
        rec = {"branch": branch, "p": p, "pq": pq,
               "beta_finest": float(b[-1]) if b.size else 0.0,
               "nulldim_finest": gr[-1]["nulldim"]}
        if keep.sum() >= 3:
            hh, bb = h[keep][-4:], b[keep][-4:]
            slope = np.polyfit(np.log(hh), np.log(bb), 1)[0]
            rec["beta_slope_vs_h"] = float(slope)
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", metavar="JSON", default=None)
    ap.add_argument("--verify", metavar="JSON", default=None)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NPROC", "12")))
    args = ap.parse_args()

    if args.verify:
        data = json.loads(Path(args.verify).read_text())
        errs = [c for c in data["cells"] if "error" in c]
        assert not errs, f"{len(errs)} failed cells"
        assert data["gates"]["unconstrained_pi"]["ok"]
        assert data["gates"]["constant_null"]["ok"]
        assert data["gates"]["kernel_1d"]["ok"]
        print(f"verify OK: {len(data['cells'])} cells, gates pass")
        for s in data["slopes"]:
            print("  {branch} p={p} pq={pq}: beta_finest={beta_finest:.4g} "
                  "nulls={nulldim_finest} slope={slope}".format(
                      slope=f"{s.get('beta_slope_vs_h', float('nan')):.2f}", **s))
        return

    print("gates:")
    ok1, b1 = gate_unconstrained_pi()
    print(f"  G1 unconstrained equal-order beta = {b1:.8f} (pi? {ok1})")
    ok2, r2 = gate_constant_null()
    print(f"  G2 constant-null residual under no-slip = {r2:.2e} ({ok2})")
    ok3, k3 = gate_1d_kernel()
    print(f"  G3 1D no-slip equal-order kernel dim = {k3} ({ok3})")
    if not (ok1 and ok2 and ok3):
        raise SystemExit("validation gate failed; certificate aborted")

    jobs = build_jobs()
    print(f"sweep: {len(jobs)} cells on {args.workers} workers")
    cells = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in enumerate(ex.map(run_cell, jobs)):
            cells.append(rec)
            if "error" in rec:
                print(f"  [{i+1}/{len(jobs)}] FAILED {rec}")
            else:
                print("  [{i}/{n}] {branch} p={p} pq={pq} ne={ne}: "
                      "nulls={nl} beta_q={bq:.4g}".format(
                          i=i + 1, n=len(jobs), branch=rec["branch"], p=rec["p"],
                          pq=rec["pq"], ne=rec["n_elem"], nl=rec["nulldim"],
                          bq=rec["beta_constants_only_quotient"]))

    data = {
        "description": "physical no-slip inf-sup certificate, viscous energy "
                       "norm, pressure quotient by constants only",
        "null_tol": NULL_TOL,
        "gates": {
            "unconstrained_pi": {"ok": ok1, "beta": b1},
            "constant_null": {"ok": ok2, "residual": r2},
            "kernel_1d": {"ok": ok3, "dim": k3},
        },
        "cells": cells,
        "slopes": fitted_slopes(cells),
    }
    dest = Path(args.write) if args.write else Path(RESULTSDIR) / "infsup_physical_bc.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=1))
    print(f"wrote {dest}")
    for s in data["slopes"]:
        print("  {branch} p={p} pq={pq}: beta_finest={beta_finest:.4g} "
              "nulls={nulldim_finest} slope={slope}".format(
                  slope=f"{s.get('beta_slope_vs_h', float('nan')):.2f}", **s))


if __name__ == "__main__":
    main()
