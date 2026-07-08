"""Deep numerical sweep of the discrete inf-sup constant beta_h for the Galerkin
B-spline velocity-pressure pairing on a clamped (walled) 2D box.

Purpose: the inf-sup theorem states beta_h = pi (the continuous-domain inf-sup constant) at EQUAL
velocity/pressure spline degree, independent of grid size N and degree p, with
only the single physical constant-pressure null mode; and beta_h = pi*sqrt(2)
one degree below (Taylor-Hood-style REDUCED order), with no null. Prior evidence
covered only p in {2,3,4} on small grids (N <= 24 elements). This script extends
the sweep to p in {2,...,8} and N (velocity basis-function count per direction)
up to 513.

MATH (see mms_infsup.cpp and s5_infsup_stokes.py for the original, validated,
small-scale derivations; this module reproduces their digits exactly -- see the
cross-validation section below -- then scales the same computation up):

  beta_h^2 = smallest nonzero generalized eigenvalue of

      A q = lambda (Mp (x) Mp) q,     A = P (x) Q + Q (x) P,

  where (x) is the Kronecker product, Mp is the 1D pressure mass matrix, and

      P = C Mv^{-1} C^T,      C_ik  = <q_i, dv_k/dx>   (divergence coupling)
      Q = Mxz Mv^{-1} Mxz^T,  Mxz_ik = <q_i, v_k>       (cross mass)

  are 1D Schur-complement Gram matrices built from the 1D velocity mass Mv and
  the 1D mixed couplings C, Mxz between the pressure and velocity spline
  spaces. The single physical constant-pressure null (lambda = 0) is skipped.

COMPUTATIONAL STRATEGY (the point of this script -- avoids ever forming the
N^2 x N^2 dense generalized eigenproblem that made mms_infsup.cpp's Jacobi
solve computationally unusable beyond N ~ 16-24):

  Whiten by the 1D pressure-mass Cholesky factor Lp = chol(Mp):
      Pt = Lp^{-1} P Lp^{-T},   Qt = Lp^{-1} Q Lp^{-T}.
  The generalized problem becomes the STANDARD symmetric eigenproblem
      At y = lambda y,   At = Pt (x) Qt + Qt (x) Pt,
  and At never needs to be formed: At vec(X) = vec(Pt X Qt^T + Qt X Pt^T) is
  an O(n^3) matvec on n x n matrices (n = pressure basis count per direction).

  EQUAL ORDER -- exact 1D reduction. Key structural fact, verified to machine
  precision at every sweep cell (the script asserts it cell-by-cell): at equal
  order the cross mass IS the velocity mass (identical spaces), so

      Q = Mxz Mv^{-1} Mxz^T = Mv Mv^{-1} Mv^T = Mv = Mp   =>   Qt = I  exactly.

  Then At = Pt (x) I + I (x) Pt, whose eigenvalues are EXACTLY the pairwise
  sums lam_i + lam_j of the eigenvalues of the single symmetric n x n matrix
  Pt (orthogonal diagonalization => genuine similarity, not just congruence).
  Pt carries the one physical null (the spline derivative of the constant is
  exactly zero), so beta_h^2 = lam_1, the smallest NONZERO eigenvalue of Pt,
  and the 2D problem has exactly one null pair (0,0). Empirically
  lam_k -> (k*pi)^2 rapidly (the nonzero set {(k*pi)^2} is shared by the
  Dirichlet and Neumann Laplacians on [0, L]; the mu_2 eigenvector is sin(pi x),
  so the correct continuous identification is the Dirichlet fluctuation problem,
  not Neumann), giving beta_h -> pi. One dense n x n eigendecomposition per cell:
  seconds even at n = 513.

  REDUCED ORDER -- pencil-preconditioned LOBPCG (exact to solver tolerance).
  Here Qt != I and [Pt, Qt] != 0, so At's eigenvalues are genuinely NOT sums
  of 1D spectra (the first version of this script assumed the generalized
  pencil Pt w = mu Qt w gives exact sums mu_i + mu_j; its own built-in dense
  cross-validation caught the error: the W (x) W transform is a CONGRUENCE,
  not a similarity, when W is not orthogonal, so the sums are only an
  approximation -- accurate to ~1e-6 relative at moderate N but off by 2e-4
  at Nv = 6. The cross-validation gate exists precisely for this). The exact
  route kept here:

    - n^2 <= 4096: dense eigh on the whitened At directly (exact, cheap).
    - larger: LOBPCG on the matrix-free At, with the pencil used where it is
      legitimate -- as a PRECONDITIONER and initial guess. Writing
      W^T Qt W = I, W^T Pt W = diag(mu) (dense n x n generalized eigh),
      At ~= (W^{-T} (x) W^{-T}) (D (+) D) (W^{-1} (x) W^{-1}), so
      At^{-1} ~= (W (x) W) (D (+) D)^{-1} (W^T (x) W^T), all O(n^3) Kronecker
      operations. Because this approximate inverse is spectrally very close to
      the true one near the bottom of the spectrum, LOBPCG converges in a
      handful of iterations even where plain ARPACK Lanczos stalls (the 1D
      operator Pt is severely ill-conditioned at high p and large N --
      measured condition number lam_max/lam_1 = 3.0e6 at p=8, N=257 and
      1.2e7 at p=8, N=513 -- which was measured to stall matrix-free eigsh
      well below N = 257; the preconditioner removes that dependence). Residual norms ||At v - lambda v|| / lambda are computed
      explicitly after the solve and stored in the JSON as proof of
      convergence; agreement with dense eigh at small/moderate N is 1e-12.

  Null-mode classification uses an ABSOLUTE threshold (1e-5) on the whitened
  spectrum plus an explicit separation check (first retained eigenvalue must
  exceed 1e-2), NOT a threshold relative to the largest eigenvalue. The first
  version of this script classified nulls with a relative rule (1e-9 x the
  largest eigenvalue) applied to the GENERALIZED PENCIL spectrum mu of
  (Pt, Qt), which that version was also using as the reduced-order answer.
  The top of the pencil spectrum is lam_max(Pt)/lam_min(Qt)-sized and grows
  much faster than the whitened Pt spectrum itself (whose measured max over
  the whole sweep is 1.2e8, at p=8, N=513) because lam_min(Qt) shrinks with
  N: measured pencil tops are 2.8e10 at (p=2, N=513) and 1.4e10 at
  (p=3, N=513), so the relative cut 1e-9 * mu_max = 27.6 (resp. 13.7)
  exceeded the true smallest eigenvalue pi^2 ~ 9.87 in exactly (and only)
  those two cells -- at (p=4, N=513) the cut is 8.2 and passes -- silently
  misclassifying the physical smallest eigenvalue as a null and inflating
  beta to 2*pi there. The absolute rule separates with ~9.7 orders of total
  gap: computed nulls land at or below 2.1e-9 across all 112 cells (~3.7
  orders under the 1e-5 cut) while the physical spectrum starts at pi^2 (~6
  orders above it); the separation check turns any future violation into a
  loud per-cell flag instead of a wrong number.

GPU: torch and cupy were checked and are NOT installed in this project's
virtualenv (no pip inside the venv to add them); the machine's RTX 5060 Ti
sits idle for this run. This does not matter at these sizes: every dense
operation is on n x n (n <= 513) matrices and the LOBPCG matvecs are O(n^3)
BLAS calls -- comfortably a CPU workload. BLAS threads are capped
(OMP_NUM_THREADS / OPENBLAS_NUM_THREADS, default 4) since the 20 CPU cores
are shared with other running experiments.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Cap BLAS/OpenMP threads BEFORE importing numpy/scipy (other jobs are running
# on this machine's 20 cores). Overridable via env for interactive testing.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np  # noqa: E402
from scipy.linalg import cholesky, eigh  # noqa: E402
from scipy.sparse.linalg import LinearOperator, lobpcg  # noqa: E402

from project_paths import RESULTSDIR  # noqa: E402

NULL_TOL_ABS = 1e-5       # absolute null cut on the whitened spectrum
SEPARATION_MIN = 1e-2     # first retained eigenvalue must exceed this
DENSE_2D_MAX = 4096       # use dense whitened eigh when n^2 <= this


def detect_backend() -> str:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return f"torch-cuda ({torch.cuda.get_device_name(0)})"
    except Exception:
        pass
    try:
        import cupy  # type: ignore

        return f"cupy (device {cupy.cuda.Device(0).name})"
    except Exception:
        pass
    return ("cpu-only (torch/cupy not installed in this venv; dense LAPACK on "
            "n<=513 1D matrices + pencil-preconditioned LOBPCG, GPU not needed "
            "at this problem size)")


# ---------------------------------------------------------------------------
# Fast, local-support (Cox-de Boor / NURBS-book Algorithm A2.3) 1D spline basis
# assembly. Evaluates only the (degree+1) nonzero basis functions per element,
# unlike a naive "one scipy.interpolate.BSpline object per global basis
# function" loop (which costs O(N) objects x O(N) elements = O(N^2) and was
# measured at ~8s already at N=257, p=8; this local-support version builds the
# same matrices -- cross-validated to machine precision -- in ~0.3s there).
# ---------------------------------------------------------------------------


def open_uniform_knots(n_elem: int, degree: int, length: float = 1.0) -> np.ndarray:
    inner = np.linspace(0.0, length, n_elem + 1)
    return np.concatenate([np.full(degree, 0.0), inner, np.full(degree, length)])


def _basis_and_deriv_at_point(knots: np.ndarray, degree: int, span: int, x: float, nder: int = 1) -> np.ndarray:
    """NURBS-book Algorithm A2.3 (ders_basis_funs): the (degree+1) nonzero basis
    functions and derivatives up to order nder at a single point x in the span
    [knots[span], knots[span+1]). Returns ders[k, j], k=0..nder, j=0..degree."""
    p = degree
    ndu = np.zeros((p + 1, p + 1))
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    ndu[0, 0] = 1.0
    for j in range(1, p + 1):
        left[j] = x - knots[span + 1 - j]
        right[j] = knots[span + j] - x
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved

    ders = np.zeros((nder + 1, p + 1))
    ders[0, :] = ndu[:, p]

    a = np.zeros((2, p + 1))
    for r in range(p + 1):
        a[0, 0] = 1.0
        s1, s2 = 0, 1
        for k in range(1, nder + 1):
            d = 0.0
            rk = r - k
            pk = p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = (k - 1) if (r - 1 <= pk) else (p - r)
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            s1, s2 = s2, s1

    r = p
    for k in range(1, nder + 1):
        for j in range(p + 1):
            ders[k, j] *= r
        r *= (p - k)
    return ders


def direction_matrices_fast(
    Nv: int, pv: int, Nq: int, pq: int, length: float = 1.0, quad: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """1D velocity mass Mv, pressure mass Mp, divergence coupling C = <q, v'>,
    and cross mass Mxz = <q, v>, for open-uniform clamped B-spline spaces of
    (Nv, pv) and (Nq, pq) sharing the same element partition. Nv, Nq are basis
    FUNCTION counts (matches boundary_spline_analysis.open_uniform_knots's
    convention); requires Nv - pv == Nq - pq (same number of elements), which
    holds for equal order (Nq=Nv, pq=pv) and for the reduced-order pairing used
    in the manuscript (Nq = Nv-1, pq = pv-1)."""
    quad = quad or (pv + pq + 2)
    n_elem = Nv - pv
    if n_elem != Nq - pq:
        raise ValueError(
            f"velocity/pressure spaces must share the same element count: "
            f"Nv-pv={n_elem} != Nq-pq={Nq - pq}"
        )
    kv = open_uniform_knots(n_elem, pv, length)
    kq = open_uniform_knots(n_elem, pq, length)

    mv = np.zeros((Nv, Nv))
    mp = np.zeros((Nq, Nq))
    cmat = np.zeros((Nq, Nv))
    mxz = np.zeros((Nq, Nv))

    xi, wi = np.polynomial.legendre.leggauss(quad)

    for e in range(n_elem):
        a = kv[pv + e]
        b = kv[pv + e + 1]
        if b <= a + 1e-14:
            continue
        half = 0.5 * (b - a)
        mid = 0.5 * (a + b)
        xg = mid + half * xi
        wg = wi * half

        span_v = pv + e
        span_q = pq + e

        nv_vals = np.zeros((quad, pv + 1))
        nv_ders = np.zeros((quad, pv + 1))
        nq_vals = np.zeros((quad, pq + 1))
        for g in range(quad):
            dv = _basis_and_deriv_at_point(kv, pv, span_v, xg[g], nder=1)
            nv_vals[g, :] = dv[0, :]
            nv_ders[g, :] = dv[1, :]
            dq = _basis_and_deriv_at_point(kq, pq, span_q, xg[g], nder=0)
            nq_vals[g, :] = dq[0, :]

        iv0 = span_v - pv
        iq0 = span_q - pq
        wcol = wg[:, None]

        mv[iv0:iv0 + pv + 1, iv0:iv0 + pv + 1] += nv_vals.T @ (nv_vals * wcol)
        mp[iq0:iq0 + pq + 1, iq0:iq0 + pq + 1] += nq_vals.T @ (nq_vals * wcol)
        cmat[iq0:iq0 + pq + 1, iv0:iv0 + pv + 1] += nq_vals.T @ (nv_ders * wcol)
        mxz[iq0:iq0 + pq + 1, iv0:iv0 + pv + 1] += nq_vals.T @ (nv_vals * wcol)

    return mv, mp, cmat, mxz


def build_whitened(Nv: int, pv: int, Nq: int, pq: int, length: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Whitened 1D Gram matrices Pt = Lp^-1 P Lp^-T, Qt = Lp^-1 Q Lp^-T."""
    mv, mp, cmat, mxz = direction_matrices_fast(Nv, pv, Nq, pq, length)
    mvi = np.linalg.inv(mv)
    P = cmat @ mvi @ cmat.T
    Q = mxz @ mvi @ mxz.T
    Lp = cholesky(mp, lower=True)
    Lpi = np.linalg.inv(Lp)
    Pt = Lpi @ P @ Lpi.T
    Qt = Lpi @ Q @ Lpi.T
    return 0.5 * (Pt + Pt.T), 0.5 * (Qt + Qt.T)


def classify_nulls(ev_sorted: np.ndarray) -> tuple[int, bool]:
    """Count null modes with an ABSOLUTE cut on the whitened spectrum and check
    that the first retained eigenvalue is well separated from the cut (see
    module docstring for why a relative-to-max rule is wrong here)."""
    nnull = int((ev_sorted < NULL_TOL_ABS).sum())
    separated = bool(nnull >= len(ev_sorted) or ev_sorted[nnull] > SEPARATION_MIN)
    return nnull, separated


# ---------------------------------------------------------------------------
# Equal order: exact 1D reduction (Qt = I verified per cell).
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    beta: float
    nnull_2d: int
    n_pressure_dofs: int
    seconds: float
    separated: bool
    method: str
    residual: float | None = None
    qt_identity_error: float | None = None
    ev_smallest: list[float] = field(default_factory=list)


def infsup_equal_order(Nv: int, pv: int, length: float = 1.0) -> CellResult:
    t0 = time.perf_counter()
    Pt, Qt = build_whitened(Nv, pv, Nv, pv, length)
    n = Pt.shape[0]
    qt_err = float(np.abs(Qt - np.eye(n)).max())
    if qt_err > 1e-8:
        raise AssertionError(
            f"equal-order identity Qt = I violated (max err {qt_err:.3e}) at "
            f"Nv={Nv}, p={pv}; the exact 1D reduction does not apply")
    lam = np.sort(np.clip(np.linalg.eigvalsh(Pt), 0.0, None))
    nnull_1d, separated = classify_nulls(lam)
    if nnull_1d >= 1:
        beta2 = float(lam[nnull_1d])       # smallest nonzero sum: 0 + lam_first_kept
    else:
        beta2 = float(2.0 * lam[0])        # no null: 2 * lam_min
    dt = time.perf_counter() - t0
    return CellResult(
        beta=float(np.sqrt(max(beta2, 0.0))),
        nnull_2d=nnull_1d * nnull_1d,      # 2D nulls are pairs of 1D nulls
        n_pressure_dofs=n, seconds=dt, separated=separated,
        method="exact-1d (Qt=I verified)", qt_identity_error=qt_err,
        ev_smallest=[float(x) for x in lam[:4]],
    )


# ---------------------------------------------------------------------------
# Reduced order: dense whitened eigh at small size, else pencil-preconditioned
# LOBPCG on the matrix-free whitened operator (exact to solver tolerance,
# residuals reported).
# ---------------------------------------------------------------------------


def infsup_reduced_order(Nv: int, pv: int, length: float = 1.0, k: int = 4) -> CellResult:
    t0 = time.perf_counter()
    Nq, pq = Nv - 1, pv - 1
    Pt, Qt = build_whitened(Nv, pv, Nq, pq, length)
    n = Pt.shape[0]
    N2 = n * n

    if N2 <= DENSE_2D_MAX:
        At = np.kron(Pt, Qt) + np.kron(Qt, Pt)
        ev = np.sort(np.clip(np.linalg.eigvalsh(0.5 * (At + At.T)), 0.0, None))
        nnull, separated = classify_nulls(ev)
        beta2 = float(ev[nnull]) if nnull < len(ev) else 0.0
        dt = time.perf_counter() - t0
        return CellResult(
            beta=float(np.sqrt(max(beta2, 0.0))), nnull_2d=nnull,
            n_pressure_dofs=n, seconds=dt, separated=separated,
            method="dense-whitened-eigh", ev_smallest=[float(x) for x in ev[:4]],
        )

    # pencil (approximate diagonalization) as preconditioner + initial guess
    mu, W = eigh(Pt, Qt)
    order = np.argsort(mu)
    mu = mu[order]
    W = W[:, order]
    S = mu[:, None] + mu[None, :]
    Sinv = 1.0 / np.maximum(S, 1e-12)

    def amv(x: np.ndarray) -> np.ndarray:
        X = x.reshape(n, n)
        return (Pt @ X @ Qt.T + Qt @ X @ Pt.T).reshape(-1)

    def pmv(x: np.ndarray) -> np.ndarray:
        X = x.reshape(n, n)
        return (W @ ((W.T @ X @ W) * Sinv) @ W.T).reshape(-1)

    A_op = LinearOperator((N2, N2), matvec=amv, dtype=np.float64)
    M_op = LinearOperator((N2, N2), matvec=pmv, dtype=np.float64)

    idx = np.dstack(np.unravel_index(np.argsort(S, axis=None), S.shape))[0][:k]
    X0 = np.stack([np.outer(W[:, i], W[:, j]).reshape(-1) for i, j in idx], axis=1)
    X0, _ = np.linalg.qr(X0)

    vals, vecs = lobpcg(A_op, X0, M=M_op, largest=False, tol=1e-10, maxiter=200)
    order2 = np.argsort(vals)
    vals = np.clip(vals[order2], 0.0, None)
    vecs = vecs[:, order2]

    # explicit residual of the smallest eigenpair: proof of convergence
    v0 = vecs[:, 0] / np.linalg.norm(vecs[:, 0])
    resid = float(np.linalg.norm(amv(v0) - vals[0] * v0) / max(vals[0], 1e-30))

    nnull, separated = classify_nulls(vals)
    beta2 = float(vals[nnull]) if nnull < len(vals) else 0.0
    dt = time.perf_counter() - t0
    return CellResult(
        beta=float(np.sqrt(max(beta2, 0.0))), nnull_2d=nnull,
        n_pressure_dofs=n, seconds=dt, separated=separated,
        method="lobpcg-pencil-preconditioned", residual=resid,
        ev_smallest=[float(x) for x in vals[:4]],
    )


# ---------------------------------------------------------------------------
# Dense-generalized cross-check (small sizes only): forms the full N^2 x N^2
# generalized eigenproblem directly, exactly like s5_infsup_stokes.py /
# mms_infsup.cpp, to validate the fast routes end to end.
# ---------------------------------------------------------------------------


def infsup_beta_dense(Nv: int, pv: int, Nq: int, pq: int, length: float = 1.0) -> tuple[float, int]:
    mv, mp, cmat, mxz = direction_matrices_fast(Nv, pv, Nq, pq, length)
    mvi = np.linalg.inv(mv)
    P = cmat @ mvi @ cmat.T
    Q = mxz @ mvi @ mxz.T
    A = np.kron(P, Q) + np.kron(Q, P)
    Mp2 = np.kron(mp, mp)
    A = 0.5 * (A + A.T)
    Mp2 = 0.5 * (Mp2 + Mp2.T)
    ev = np.sort(np.clip(eigh(A, Mp2, eigvals_only=True), 0.0, None))
    nnull, _sep = classify_nulls(ev)
    beta2 = float(ev[nnull]) if nnull < len(ev) else 0.0
    return float(np.sqrt(max(beta2, 0.0))), nnull


# mms_infsup.cpp reference digits (p=3, equal order beta -> pi with 1 null,
# reduced order beta -> pi*sqrt(2) with 0 nulls, N-independent), reproduced to
# the 4 significant digits the C++ app prints. Sanity anchor only: the actual
# gate below recomputes everything with the dense route (the same algorithm as
# mms_infsup.cpp / s5_infsup_stokes.py) and requires the fast routes to match.
MMS_REF = {"equal": 3.1416, "reduced": 4.4429}


def run_cross_validation() -> dict:
    rows = []
    ok = True
    print("Cross-validation vs mms_infsup.cpp / s5_infsup_stokes.py (p=3):", flush=True)
    print(f"{'Nv':>4} {'branch':>8} {'beta_dense':>13} {'beta_fast':>13} {'mms ref':>9} "
          f"{'nnull':>6} {'agree':>6}", flush=True)
    for Nv in (6, 8, 10, 12, 16):
        for branch in ("equal", "reduced"):
            if branch == "equal":
                b_dense, nn_dense = infsup_beta_dense(Nv, 3, Nv, 3)
                cell = infsup_equal_order(Nv, 3)
            else:
                b_dense, nn_dense = infsup_beta_dense(Nv, 3, Nv - 1, 2)
                cell = infsup_reduced_order(Nv, 3)
            agree = (abs(b_dense - cell.beta) < 1e-6 and
                     abs(b_dense - MMS_REF[branch]) < 5e-4 and
                     nn_dense == cell.nnull_2d)
            ok = ok and agree
            print(f"{Nv:>4} {branch:>8} {b_dense:>13.8f} {cell.beta:>13.8f} "
                  f"{MMS_REF[branch]:>9.4f} {cell.nnull_2d:>6d} {str(agree):>6}", flush=True)
            rows.append({
                "Nv": Nv, "branch": branch, "beta_dense": b_dense, "beta_fast": cell.beta,
                "reference_mms_infsup": MMS_REF[branch], "nnull_dense": nn_dense,
                "nnull_fast": cell.nnull_2d, "fast_method": cell.method, "agree": agree,
            })
    # extra: LOBPCG-vs-dense-whitened at moderate sizes where LOBPCG actually engages
    print("\nLOBPCG-vs-dense agreement at moderate reduced-order sizes:", flush=True)
    for Nv, pv in ((70, 3), (80, 4)):
        Pt, Qt = build_whitened(Nv, pv, Nv - 1, pv - 1)
        At = np.kron(Pt, Qt) + np.kron(Qt, Pt)
        ev = np.sort(np.clip(np.linalg.eigvalsh(0.5 * (At + At.T)), 0.0, None))
        cell = infsup_reduced_order(Nv, pv)
        diff = abs(float(np.sqrt(ev[0])) - cell.beta)
        agree = diff < 1e-8 and cell.method == "lobpcg-pencil-preconditioned"
        ok = ok and agree
        print(f"  Nv={Nv} p={pv}: dense beta={np.sqrt(ev[0]):.10f}  lobpcg beta={cell.beta:.10f} "
              f" diff={diff:.2e}  method={cell.method}  agree={agree}", flush=True)
        rows.append({"Nv": Nv, "branch": "reduced", "beta_dense": float(np.sqrt(ev[0])),
                     "beta_fast": cell.beta, "check": "lobpcg-engaged", "agree": agree})
    print(f"\nAll cross-checks agree: {ok}", flush=True)
    return {"rows": rows, "all_agree": ok}


# ---------------------------------------------------------------------------
# Main sweep.
# ---------------------------------------------------------------------------

SWEEP_N = (9, 17, 33, 65, 129, 257, 385, 513)
SWEEP_P = (2, 3, 4, 5, 6, 7, 8)


def run_sweep() -> dict:
    equal_rows = []
    reduced_rows = []
    print(f"\nSweep grid: N in {SWEEP_N}, p in {SWEEP_P} (equal and reduced order)\n", flush=True)
    print(f"{'branch':>8} {'p':>3} {'N':>5} {'n_pdofs':>8} {'beta_h':>13} {'beta/target':>13} "
          f"{'nnull':>6} {'sep':>4} {'time(s)':>8}  method", flush=True)

    for pv in SWEEP_P:
        for Nv in SWEEP_N:
            if Nv <= pv:
                continue
            c = infsup_equal_order(Nv, pv)
            ratio = c.beta / np.pi
            print(f"{'equal':>8} {pv:>3} {Nv:>5} {c.n_pressure_dofs:>8} {c.beta:>13.9f} "
                  f"{ratio:>13.9f} {c.nnull_2d:>6d} {str(c.separated):>4} {c.seconds:>8.2f}  {c.method}",
                  flush=True)
            equal_rows.append({
                "pv": pv, "pq": pv, "Nv": Nv, "Nq": Nv, "n_pressure_dofs": c.n_pressure_dofs,
                "beta": c.beta, "beta_over_pi": ratio, "nnull": c.nnull_2d,
                "separated": c.separated, "seconds": c.seconds, "method": c.method,
                "qt_identity_error": c.qt_identity_error, "ev_smallest": c.ev_smallest,
            })

    target_reduced = np.pi * np.sqrt(2.0)
    for pv in SWEEP_P:
        for Nv in SWEEP_N:
            if (Nv - 1) <= (pv - 1) or Nv <= pv:
                continue
            c = infsup_reduced_order(Nv, pv)
            ratio = c.beta / target_reduced
            print(f"{'reduced':>8} {pv:>3} {Nv:>5} {c.n_pressure_dofs:>8} {c.beta:>13.9f} "
                  f"{ratio:>13.9f} {c.nnull_2d:>6d} {str(c.separated):>4} {c.seconds:>8.2f}  {c.method}",
                  flush=True)
            reduced_rows.append({
                "pv": pv, "pq": pv - 1, "Nv": Nv, "Nq": Nv - 1, "n_pressure_dofs": c.n_pressure_dofs,
                "beta": c.beta, "beta_over_pi_sqrt2": ratio, "nnull": c.nnull_2d,
                "separated": c.separated, "seconds": c.seconds, "method": c.method,
                "lobpcg_residual": c.residual, "ev_smallest": c.ev_smallest,
            })

    max_dev_equal = max(abs(r["beta_over_pi"] - 1.0) for r in equal_rows)
    max_dev_reduced = max(abs(r["beta_over_pi_sqrt2"] - 1.0) for r in reduced_rows)
    return {
        "equal_order": equal_rows,
        "reduced_order": reduced_rows,
        "max_relative_deviation_equal_order": max_dev_equal,
        "max_relative_deviation_reduced_order": max_dev_reduced,
        "pi": float(np.pi),
        "pi_sqrt2": float(target_reduced),
    }


def main() -> None:
    force = "--force" in sys.argv[1:]
    backend = detect_backend()
    print(f"Backend: {backend}", flush=True)
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
          f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}", flush=True)

    t_start = time.perf_counter()
    xval = run_cross_validation()
    if not xval["all_agree"]:
        print("\n*** CROSS-VALIDATION DISAGREEMENT: refusing to run the sweep or "
              "overwrite results/infsup_sweep.json. Resolve the disagreement, or "
              "re-run with --force to proceed anyway (results will carry "
              "all_agree=false). ***", flush=True)
        if not force:
            sys.exit(1)

    sweep = run_sweep()
    total_time = time.perf_counter() - t_start

    all_separated = (all(r["separated"] for r in sweep["equal_order"]) and
                     all(r["separated"] for r in sweep["reduced_order"]))
    print(f"\nTotal wall time: {total_time:.1f}s", flush=True)
    print(f"All null/nonzero separations clean: {all_separated}", flush=True)
    print(f"Max relative deviation from pi (equal order):           "
          f"{sweep['max_relative_deviation_equal_order']:.3e}", flush=True)
    print(f"Max relative deviation from pi*sqrt(2) (reduced order): "
          f"{sweep['max_relative_deviation_reduced_order']:.3e}", flush=True)

    out = {
        "backend": backend,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "cross_validation": xval,
        "sweep": sweep,
        "sweep_grid": {"N": list(SWEEP_N), "p": list(SWEEP_P)},
        "all_separations_clean": all_separated,
        "total_wall_time_seconds": total_time,
        "method": ("equal order: exact 1D reduction via the verified identity Qt = I "
                   "(At = Pt(x)I + I(x)Pt has exact pairwise-sum spectrum, so "
                   "beta^2 = smallest nonzero eigenvalue of the whitened 1D "
                   "divergence Gram Pt); reduced order: dense whitened eigh for "
                   "n^2 <= 4096, otherwise LOBPCG on the matrix-free whitened "
                   "Kronecker operator preconditioned by the generalized pencil "
                   "eigh(Pt, Qt) (residuals reported). Cross-validated against the "
                   "dense N^2xN^2 generalized route (the mms_infsup.cpp algorithm) "
                   "at p=3, N=6..16, and against dense whitened eigh at N=70, 80."),
    }

    RESULTSDIR.mkdir(parents=True, exist_ok=True)
    dest = Path(RESULTSDIR) / "infsup_sweep.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}", flush=True)


if __name__ == "__main__":
    main()
