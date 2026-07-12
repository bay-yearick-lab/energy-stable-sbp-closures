"""1D C0 Gauss-Lobatto-Legendre spectral-element operators on [0, 1].

The textbook SEM discretization (textbook form): degree-P
Legendre elements on a uniform macro-mesh, GLL collocation/quadrature nodes,
diagonal (GLL-consistent) mass matrix, assembled Galerkin stiffness, and the
weak first-derivative form for transport. This is the strongest standard
variational competitor for the paper's PDE head-to-heads; it enters every
experiment through the same protocol as the other families (matched total
DOFs, tier-B time step from its own spectral radius, shared diagnostics).

Self-gates (run via --verify or verify()):
  G1  GLL weights integrate polynomials to degree 2P-1 exactly.
  G2  interior rows of the diffusion operator reproduce (x^k)'' for k <= P.
  G3  smallest Dirichlet eigenvalue of -D2 converges to pi^2.
  G4  weak first-derivative matrix satisfies the SBP identity
      C + C^T = diag(-1, 0, ..., 0, 1) (boundary evaluation) to roundoff.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import legendre as npleg

Array = np.ndarray


def gll_nodes_weights(P: int) -> tuple[Array, Array]:
    """Gauss-Lobatto-Legendre nodes and weights of degree P (P+1 points) on [-1,1]."""
    if P < 1:
        raise ValueError("degree must be >= 1")
    # interior nodes: roots of L_P'
    cP = np.zeros(P + 1)
    cP[P] = 1.0
    dcP = npleg.legder(cP)
    inner = npleg.legroots(dcP)
    nodes = np.concatenate([[-1.0], np.sort(inner.real), [1.0]])
    LP = npleg.legval(nodes, cP)
    weights = 2.0 / (P * (P + 1) * LP ** 2)
    return nodes, weights


def lagrange_diff_matrix(nodes: Array) -> Array:
    """Differentiation matrix of the Lagrange basis on the given nodes."""
    n = nodes.size
    wb = np.ones(n)
    for j in range(n):
        wb[j] = 1.0 / np.prod(nodes[j] - np.delete(nodes, j))
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = (wb[j] / wb[i]) / (nodes[i] - nodes[j])
        D[i, i] = -np.sum(D[i, :])
    return D


class SEMOperator1D:
    """Degree-P GLL spectral elements on [0, L] with K uniform elements.

    Exposes:
      x           global node vector (npts = K*P + 1)
      mass        diagonal global mass (vector, length npts)
      D2          diffusion operator  M^{-1} K_stiff  with  (K_stiff)_ij =
                  -int N_i' N_j'  (so D2 ~ d^2/dx^2, negative semidefinite)
      D1          weak transport operator  M^{-1} C  with  C_ij = int N_i N_j'
    """

    def __init__(self, degree: int, nelem: int, length: float = 1.0):
        self.p = int(degree)
        self.nelem = int(nelem)
        self.length = float(length)
        P, K = self.p, self.nelem
        self.npts = K * P + 1
        gl, gw = gll_nodes_weights(P)
        Dref = lagrange_diff_matrix(gl)
        h = length / K
        jac = h / 2.0

        x = np.empty(self.npts)
        mass = np.zeros(self.npts)
        stiff = np.zeros((self.npts, self.npts))   # int N_i' N_j'
        weakd = np.zeros((self.npts, self.npts))   # int N_i  N_j'
        # element stiffness: S_ij = sum_q w_q D_qi D_qj / jac
        Sel = (Dref.T * gw) @ Dref / jac
        # element weak derivative: C_ij = sum_q w_q delta_qi D_qj = diag(w) D / 1
        Cel = (np.diag(gw) @ Dref)
        for e in range(K):
            sl = slice(e * P, e * P + P + 1)
            x[sl] = e * h + jac * (gl + 1.0)
            mass[sl.start:sl.stop] += gw * jac
            stiff[sl, sl] += Sel
            weakd[sl, sl] += Cel
        self.x = x
        self.mass = mass
        self._stiff = stiff
        self._weakd = weakd
        minv = 1.0 / mass
        self.D2 = -(minv[:, None] * stiff)
        self.D1 = minv[:, None] * weakd

    # -- verification gates ------------------------------------------------
    def verify(self, tol: float = 1e-9) -> dict:
        P = self.p
        gl, gw = gll_nodes_weights(P)
        report = {}
        # G1: quadrature exactness to degree 2P-1
        worst = 0.0
        for k in range(2 * P):
            exact = (1.0 - (-1.0) ** (k + 1)) / (k + 1)
            worst = max(worst, abs(float(np.sum(gw * gl ** k)) - exact))
        report["G1_quadrature"] = worst
        assert worst < 1e-12, ("G1 quadrature", worst)
        # G2: interior rows of D2 reproduce (x^k)'' for k <= P
        worst = 0.0
        interior = slice(1, self.npts - 1)
        for k in range(P + 1):
            u = self.x ** k
            exact = k * (k - 1) * self.x ** max(k - 2, 0) if k >= 2 else np.zeros_like(self.x)
            worst = max(worst, float(np.max(np.abs((self.D2 @ u)[interior] - exact[interior]))))
        report["G2_d2_poly"] = worst
        assert worst < tol * max(1.0, self.npts ** 2), ("G2 poly exactness", worst)
        # G3: smallest Dirichlet eigenvalue of -D2 near (pi/L)^2
        A = -self.D2[1:-1, 1:-1]
        lam = np.sort(np.linalg.eigvals(A).real)
        lam1 = float(lam[0])
        target = (np.pi / self.length) ** 2
        report["G3_lam1_relerr"] = abs(lam1 - target) / target
        assert report["G3_lam1_relerr"] < 1e-4, ("G3 eigenvalue", lam1, target)
        # G4: SBP identity for the weak derivative: C + C^T = boundary
        B = self._weakd + self._weakd.T
        Bex = np.zeros_like(B)
        Bex[0, 0] = -1.0
        Bex[-1, -1] = 1.0
        report["G4_sbp_identity"] = float(np.max(np.abs(B - Bex)))
        assert report["G4_sbp_identity"] < 1e-12, ("G4 SBP", report["G4_sbp_identity"])
        return report


def sem_grid_for(npts: int, degree: int = 6) -> SEMOperator1D:
    """SEM operator whose global DOF count matches npts (requires degree | npts-1)."""
    if (npts - 1) % degree != 0:
        raise ValueError(f"npts-1={npts-1} not divisible by degree={degree}")
    return SEMOperator1D(degree, (npts - 1) // degree)


if __name__ == "__main__":
    import sys
    for P in (4, 6, 8):
        for K in (3, 8):
            op = SEMOperator1D(P, K)
            rep = op.verify()
            print(f"P={P} K={K} npts={op.npts}: " +
                  " ".join(f"{k}={v:.2e}" for k, v in rep.items()))
    # convergence sanity: Dirichlet sine decay under RK4, error should fall
    # fast with K at fixed P=6
    errs = []
    for K in (2, 3, 4):
        op = SEMOperator1D(6, K)
        nu = 0.05
        u = np.sin(np.pi * op.x)
        rho = float(np.max(np.abs(np.linalg.eigvals(op.D2).real)))
        dt = 0.9 * 2.785 / (nu * rho)
        t_end = 1.0
        nsteps = int(np.ceil(t_end / dt))
        dt = t_end / nsteps
        for _ in range(nsteps):
            def rhs(v):
                r = nu * (op.D2 @ v)
                r[0] = r[-1] = 0.0
                return r
            k1 = rhs(u); k2 = rhs(u + 0.5 * dt * k1)
            k3 = rhs(u + 0.5 * dt * k2); k4 = rhs(u + dt * k3)
            u = u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            u[0] = u[-1] = 0.0
        exact = np.exp(-nu * np.pi ** 2 * t_end) * np.sin(np.pi * op.x)
        errs.append(float(np.max(np.abs(u - exact))))
        print(f"P=6 K={K} npts={op.npts} heat-MMS max err {errs[-1]:.3e}")
    assert errs[-1] < errs[0] * 1e-2, ("convergence gate", errs)
    print("SEM operator gates: ALL PASS")
    sys.exit(0)
