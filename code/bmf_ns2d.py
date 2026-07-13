"""BMF staggered primitive-variable incompressible Navier-Stokes solver (2D).

Exercises, end to end, the fully conservative staggered BMF Navier-Stokes
formulation of the paper (Theorem thm:ns-conservation; technical edition
Section "Fully conservative incompressible Navier-Stokes formulation"):
velocity components on the compatible face spaces, pressure/divergence on the
primal tensor space, metric-adjoint skew-symmetric product-map advection, compatible
flux-divergence diffusion, and a constrained metric-orthogonal projection that
enforces the discrete divergence constraint and the strong Dirichlet traces
simultaneously (the ConstrainedProjector pattern of cavity/apps/cavity.cpp).

Layout (unit square, N points per direction, degree p, BMF = clamped B-spline
staggered complex from boundary_spline_analysis.build_operator('B', p, N)):

  u  on (x-derivative space) x (z-primal space): nodal shape (N-1, N)
  w  on (x-primal space) x (z-derivative space): nodal shape (N, N-1)
  p  on (x-primal) x (z-primal):                 nodal shape (N, N)

1D staggered blocks per axis (ClampedBSpline.staggered_operators()):
  d10 (exact gradient, primal -> derivative), d01 (weak divergence,
  derivative -> primal), metrics h0/h1, boundary matrix B with the SBP
  identity h0 d01 + d10^T h1 = B (verified at build).

Divergence  D(u,w) = d01_x u + w d01_z^T  (primal x primal).
Gradient    G p    = (d10_x p, p d10_z^T).
Metrics     Hu = h1_x (x) h0_z, Hw = h0_x (x) h1_z, Hp = h0_x (x) h0_z.

ADVECTION -- exact operator definitions used here
-------------------------------------------------
The convection is the paper's metric-adjoint skew-symmetric split (technical
edition eq. one-direction-skew-split / bm-skew-convection),

    C_i(a) u_i = 1/2 sum_r [ D_r^{(i)} A_r(a_r) u_i + A_r(a_r)^dagger G_r^{(i)} u_i ],
    A_r(a_r) v := Pi_{W_r}(a_{r,h} v_h)   (exact-quadrature L2 projection),
    A^dagger  := H_i^{-1} A^T H_{W_r},

with, for the u-component (state space V1 = S^{p-1}_x (x) S^p_z, metric Hu):

  x-direction: the TRANSPOSED staggered pair, flux space W_x = the pressure
     space S^p_x (x) S^p_z:  G_x = d01_x (x) I,  D_x = d10_x (x) I,
     H_Wx = h0_x (x) h0_z. SBP: h1_x d10_x + d01_x^T h0_x = B_x^T (verified).
  z-direction: the standard pair into the corner space S^{p-1}_x (x) S^{p-1}_z:
     G_z = I (x) d10_z,  D_z = I (x) d01_z,  H_Wz = h1_x (x) h1_z.

  Advecting representations (u-equation):
     a_x := L_x u   in the pressure space,
     a_z := E_x w   in the corner space,
  where E is the EXACT spline evaluation-restriction (primal spline evaluated
  at the derivative-space Greville nodes) and L is the intertwined lift

     L := X E d01 + 1 (h1-mean),   X := h0^{-1} d10^T (d10 h0^{-1} d10^T)^{-1},

  i.e. the h0-minimal-norm exact antiderivative of the weak derivative, mean-
  restored. By construction d10 L = E d01 EXACTLY (residual ~1e-13, verified),
  and L is order-p consistent on smooth data (measured at build). Consequence:
  the component-complex divergence of the advecting field is an exact
  restriction of the pressure-complex divergence,
     D_x a_x + D_z a_z = E_x (D(u,w))       (u-equation)
     D_x a_x' + D_z a_z' = (D(u,w)) E_z^T   (w-equation),
  which the projection zeroes; hence the momentum identity of Theorem
  thm:nonlinear-ke ("the defect is the discrete divergence of a") closes at
  roundoff, together with the KE identity, on the constraint manifold.

  The w-equation mirrors this: a_z' := w L_z^T (pressure space, transposed
  z-pair G_z = I (x) d01_z, D_z = I (x) d10_z), a_x' := u E_z^T (corner space,
  standard x-pair G_x = d10_x (x) I, D_x = d01_x (x) I).

DEVIATION from the task sketch (documented, justified): the plain pointwise
Galerkin skew form 1/2[<phi,(U.grad)U_i> + <phi, div(U U_i)>] satisfies the KE
identity unconditionally but leaves a global-momentum defect
-1/2 int u_h (div U)_pointwise ~ 1e-2 on random discretely-divergence-free
states (measured; printed by --verify), because the pointwise spline divergence
of a weakly-projected field is not zero. The paper's own product-map split with
the intertwined advecting representations closes BOTH identities at roundoff,
so that is what is implemented. This is more faithful to the manuscript, whose
convection operator is defined through the product maps A_r (eq.
bm-skew-convection), not through the pointwise weak form.

QUADRATURE: ceil((3p+1)/2) Gauss-Legendre points per knot span (7 for p=4);
exact for every assembled integrand here (max polynomial degree 3p per
direction). The task's p+1 points are NOT exact for the degree-3p products, so
the stricter count is used (accuracy raised, never reduced).

VISCOUS OPERATOR -- exact definition used
-----------------------------------------
Per direction, the compatible flux-divergence second derivative L = D G of the
component's own space:
  * along the component's primal axis: op.second_derivative_matrix() of
    ClampedBSpline(N, p)  (= d01 d10 of the (N,p) staggered complex);
  * along the component's derivative axis: op.second_derivative_matrix() of
    ClampedBSpline(N-1, p-1) -- the derivative space of the (N,p) complex IS
    the primal space of the (N-1,p-1) complex (identical knot vectors and
    h1(N,p) == h0(N-1,p-1), both verified to be exact at build), so this is the
    same D G composition acting inside the component's own space.
So  L_u = D2^{(N-1,p-1)}_x (x) I + I (x) D2^{(N,p)}_z  and mirrored for w.
The dissipation reported is nu * sum_r ||G_r u_i||^2_{H_r} with the same
sub-complex gradients (the interior term of eq. ns-ke-balance).

PROJECTION: constrained metric-orthogonal projection (Definition 3 pattern):
  min ||v - v*||_H  s.t.  D v = 0,  trace(v) = g,
via the div-free projector P = I - H^{-1} D^T Lp^+ D with the Schur complement
Lp = D H^{-1} D^T fast-diagonalized per axis (tensor generalized eigenproblem),
then the boundary Schur block S_b = C P H^{-1} C^T solved by eigendecomposition
pseudo-inverse with relative cutoff 1e-8 (cavity.cpp ConstrainedProjector).
Stage rates are NOT selector-zeroed: the projection owns the boundary manifold
(CPROJ_FREE_RATES pattern).

Time integration: SSP-RK3, dt = 0.4 * 0.1 / N, constrained projection every
stage. Lid: regularized u = 16 x^2 (1-x)^2 on the top wall, no-slip elsewhere.

VERIFICATION GATES (run: --verify, or --smoke for N=17 sizes):
  G1 divergence of projected states < 1e-13 relative
  G2 advective KE production < 1e-12 * KE on 20 random div-free homogeneous states
  G3 momentum: 1^T H C_i equals the SBP boundary flux only; interior residual
     < 1e-12 relative on the same states (boundary fluxes reported)
  G4 heat MMS on each face space: viscous operator decays sin(pi x) sin(pi z)
     at 2 pi^2 nu to 1e-3 relative (N=33 in --verify)
  G5 regularized-lid run N=24, Re=400, T=5: stable, energy bounded, divergence
     at floor

Driver:  python bmf_ns2d.py run N Re T outprefix
writes code/cavity/data/bmfns_<outprefix>_run_summary.json (same fields as the
cavity app) plus dense-spline centerline CSVs.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh, lu_factor, lu_solve

from boundary_spline_analysis import (
    ClampedBSpline,
    bspline_evaluation_matrix,
    bspline_mass_matrix_from_knots,
    build_operator,
    derivative_space_knots,
    scheme_kind,
)
from project_paths import CODEDIR, RESULTSDIR, ensure_generated_dirs

CAVITY_DATA = CODEDIR / "cavity" / "data"


# ---------------------------------------------------------------------------
# per-axis operator bundle
# ---------------------------------------------------------------------------
@dataclass
class Axis:
    n: int
    p: int
    x: np.ndarray = field(init=False)
    eta: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n, p = self.n, self.p
        op = build_operator(scheme_kind(f"BMF{p}"), p, n)   # ClampedBSpline
        assert isinstance(op, ClampedBSpline)
        s = op.staggered_operators()
        self.op = op
        self.x = s.x
        self.eta = s.eta
        self.d10 = s.d10          # (n-1, n) exact gradient
        self.d01 = s.d01          # (n, n-1) weak divergence
        self.h0 = s.h0            # (n, n)  primal metric
        self.h1 = s.h1            # (n-1, n-1) derivative metric
        self.B = s.boundary       # (n, n-1) SBP boundary matrix
        self.bmat = s.bmat
        self.bhat = s.bhat_eta
        self.knots = op.knots
        self.hknots = derivative_space_knots(op.knots)
        self.binv = np.linalg.inv(self.bmat)
        self.bhatinv = np.linalg.inv(self.bhat)
        self.h0inv = np.linalg.inv(self.h0)
        self.h1inv = np.linalg.inv(self.h1)
        self.sbp_residual = float(np.max(np.abs(self.h0 @ self.d01 + self.d10.T @ self.h1 - self.B)))

        # sub-complex: the derivative space as the primal space of (n-1, p-1)
        sub = ClampedBSpline(n - 1, p - 1)
        ssub = sub.staggered_operators()
        self.subknots_residual = float(np.max(np.abs(sub.knots - self.hknots)))
        self.submetric_residual = float(np.max(np.abs(ssub.h0 - self.h1)))
        if self.subknots_residual > 1e-12 or self.submetric_residual > 1e-10:
            raise RuntimeError("derivative space != primal space of (n-1, p-1); "
                               f"knots {self.subknots_residual:.2e}, metric {self.submetric_residual:.2e}")
        self.D2p = op.second_derivative_matrix()    # (n, n)   d01 d10 on primal
        self.D2d = sub.second_derivative_matrix()   # (n-1, n-1) d01' d10' on derivative space
        self.sub_d10 = ssub.d10                     # (n-2, n-1) gradient inside derivative space
        self.sub_h1 = ssub.h1                       # (n-2, n-2)

        # exact evaluation restriction E: primal nodal -> derivative-grid values
        self.E = bspline_evaluation_matrix(self.knots, p, self.eta) @ self.binv

        # intertwined lift L (h0-minimal antiderivative of the weak derivative,
        # h1-mean restored): d10 L = E d01 exactly
        G = self.d10
        X = self.h0inv @ G.T @ np.linalg.inv(G @ self.h0inv @ G.T)
        one_e = np.ones(n - 1)
        mean_row = (self.h1 @ one_e) / float(one_e @ self.h1 @ one_e)
        self.L = X @ self.E @ self.d01 + np.outer(np.ones(n), mean_row)
        self.intertwine_residual = float(np.max(np.abs(self.d10 @ self.L - self.E @ self.d01)))
        # consistency of the lift and the restriction on smooth data
        f_eta = np.sin(2.3 * np.pi * self.eta) + 0.3 * self.eta**2
        f_x = np.sin(2.3 * np.pi * self.x) + 0.3 * self.x**2
        self.lift_consistency = float(np.max(np.abs(self.L @ f_eta - f_x)))
        self.restrict_consistency = float(np.max(np.abs(self.E @ f_x - f_eta)))

        # quadrature (exact through degree 3p per direction) and basis tables
        q1 = math.ceil((3 * p + 1) / 2)
        uk = np.unique(self.knots)
        xi, wi = leggauss(q1)
        pts, wts = [], []
        for a, b in zip(uk[:-1], uk[1:]):
            if b <= a + 1e-14:
                continue
            pts.append(0.5 * (b - a) * xi + 0.5 * (a + b))
            wts.append(0.5 * (b - a) * wi)
        self.xq = np.concatenate(pts)
        self.wq = np.concatenate(wts)
        P0 = bspline_evaluation_matrix(self.knots, p, self.xq)
        F0 = bspline_evaluation_matrix(self.hknots, p - 1, self.xq)
        M0 = bspline_mass_matrix_from_knots(self.knots, p)
        M1 = bspline_mass_matrix_from_knots(self.hknots, p - 1)
        self.Vp = P0 @ self.binv                          # primal nodal -> quad values
        self.Vd = F0 @ self.bhatinv                       # derivative nodal -> quad values
        self.Wp = self.bmat @ lu_solve(lu_factor(M0), P0.T)   # quad moments -> primal nodal L2 proj
        self.Wd = self.bhat @ lu_solve(lu_factor(M1), F0.T)   # quad moments -> derivative nodal L2 proj


# ---------------------------------------------------------------------------
# 2D solver
# ---------------------------------------------------------------------------
class BMFNS2D:
    def __init__(self, N: int, p: int = 4, Re: float = 400.0) -> None:
        self.N, self.p = int(N), int(p)
        self.Re = float(Re)
        self.nu = 1.0 / float(Re)
        self.X = Axis(N, p)
        self.Z = Axis(N, p)
        X, Z = self.X, self.Z
        self.WQ = np.outer(X.wq, Z.wq)
        self.shape_u = (N - 1, N)
        self.shape_w = (N, N - 1)
        self.nu_dof = (N - 1) * N
        self.nw_dof = N * (N - 1)
        self.u = np.zeros(self.shape_u)
        self.w = np.zeros(self.shape_w)
        self._last_phi = np.zeros((N, N))
        self.projector: ConstrainedProjector | None = None

    # ---- metric pairings ----------------------------------------------------
    def qform_u(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * (self.X.h1 @ b @ self.Z.h0)))

    def qform_w(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * (self.X.h0 @ b @ self.Z.h1)))

    def kinetic_energy(self) -> float:
        return 0.5 * (self.qform_u(self.u, self.u) + self.qform_w(self.w, self.w))

    # ---- divergence ----------------------------------------------------------
    def divergence(self, u: np.ndarray | None = None, w: np.ndarray | None = None) -> np.ndarray:
        u = self.u if u is None else u
        w = self.w if w is None else w
        return self.X.d01 @ u + w @ self.Z.d01.T

    def div_relative(self) -> float:
        dv = self.divergence()
        den = np.linalg.norm(self.X.d01 @ self.u) + np.linalg.norm(self.w @ self.Z.d01.T) + 1e-300
        return float(np.linalg.norm(dv) / den)

    # ---- advection: metric-adjoint skew-symmetric product-map split ---------------
    def advection(self, u: np.ndarray, w: np.ndarray):
        X, Z, WQ = self.X, self.Z, self.WQ

        def qU(f):   # V1 field at quad points
            return X.Vd @ f @ Z.Vp.T

        def qW(f):
            return X.Vp @ f @ Z.Vd.T

        def qP(f):   # pressure-space field at quad points
            return X.Vp @ f @ Z.Vp.T

        def qC(f):   # corner-space field at quad points
            return X.Vd @ f @ Z.Vd.T

        def asmP(v):     # exact L2 projection into the pressure space, nodal
            return X.Wp @ (WQ * v) @ Z.Wp.T

        def asmC(v):     # ... into the corner space
            return X.Wd @ (WQ * v) @ Z.Wd.T

        def asmU(v):     # V1 Galerkin (moments + mass solve + nodal)
            return X.Wd @ (WQ * v) @ Z.Wp.T

        def asmW(v):
            return X.Wp @ (WQ * v) @ Z.Wd.T

        # ---- u-component ----
        a_x = X.L @ u              # advecting x-velocity in the pressure space
        a_z = X.E @ w              # advecting z-velocity in the corner space
        gx = X.d01 @ u             # G_x u (transposed pair, pressure space)
        gz = u @ Z.d10.T           # G_z u (corner space)
        Uq = qU(u)
        s_x = asmP(qP(a_x) * Uq)   # A_x(a_x) u
        t_z = asmC(qC(a_z) * Uq)   # A_z(a_z) u
        T1x = X.d10 @ s_x                       # D_x A_x u
        T2x = asmU(qP(a_x) * qP(gx))            # A_x^dagger G_x u
        T1z = t_z @ Z.d01.T                     # D_z A_z u
        T2z = asmU(qC(a_z) * qC(gz))            # A_z^dagger G_z u
        Cu = 0.5 * (T1x + T2x + T1z + T2z)

        # ---- w-component (mirrored) ----
        b_z = w @ Z.L.T
        b_x = u @ Z.E.T
        gzz = w @ Z.d01.T
        gxx = X.d10 @ w
        Wq = qW(w)
        s_z = asmP(qP(b_z) * Wq)
        t_x = asmC(qC(b_x) * Wq)
        S1z = s_z @ Z.d10.T
        S2z = asmW(qP(b_z) * qP(gzz))
        S1x = X.d01 @ t_x
        S2x = asmW(qC(b_x) * qC(gxx))
        Cw = 0.5 * (S1z + S2z + S1x + S2x)

        aux = dict(a_x=a_x, a_z=a_z, b_z=b_z, b_x=b_x, s_x=s_x, t_z=t_z, s_z=s_z, t_x=t_x,
                   T=(T1x, T2x, T1z, T2z), S=(S1z, S2z, S1x, S2x))
        return Cu, Cw, aux

    def ke_production(self, u=None, w=None):
        u = self.u if u is None else u
        w = self.w if w is None else w
        Cu, Cw, aux = self.advection(u, w)
        prod = self.qform_u(u, Cu) + self.qform_w(w, Cw)
        scale = (sum(abs(self.qform_u(u, t)) for t in aux["T"])
                 + sum(abs(self.qform_w(w, t)) for t in aux["S"]) + 1e-300)
        return prod, scale

    def ke_balance(self, u=None, w=None):
        """Advective KE production and the SBP boundary work it must equal.

        Pairing the split form with the state and applying the SBP identities
        leaves only  F_KE = 1/2 sum_r u_i^T Btilde_r (A_r u_i): the advective
        boundary work supplied by the SBP boundary matrices (zero on
        homogeneous states, the lid work on the driven cavity)."""
        u = self.u if u is None else u
        w = self.w if w is None else w
        X, Z = self.X, self.Z
        Cu, Cw, aux = self.advection(u, w)
        prod = self.qform_u(u, Cu) + self.qform_w(w, Cw)
        f_u = 0.5 * (float(np.sum((X.B @ u) * (aux["s_x"] @ Z.h0)))
                     + float(np.sum((X.h1 @ u) * (aux["t_z"] @ Z.B.T))))
        f_w = 0.5 * (float(np.sum((X.h0 @ w) * (aux["s_z"] @ Z.B)))
                     + float(np.sum(w * (X.B @ aux["t_x"] @ Z.h1))))
        scale = (sum(abs(self.qform_u(u, t)) for t in aux["T"])
                 + sum(abs(self.qform_w(w, t)) for t in aux["S"]) + 1e-300)
        return dict(prod=prod, F=f_u + f_w, residual=abs(prod - (f_u + f_w)), scale=scale)

    def momentum_balance(self, u=None, w=None):
        """1^T H_i C_i and the SBP boundary fluxes F_i; Theorem: they coincide.

        u-component boundary flux (from the SBP boundary matrices):
          1^T (B_x^T (x) h0_z) A_x u   +  a_x^T (B_x (x) h0_z) u          (x-pair)
        + 1^T (h1_x (x) B_z)   A_z u   +  a_z^T (h1_x (x) B_z^T) u        (z-pair)
        all over 2; the a^T-terms vanish exactly on zero-trace states.
        """
        u = self.u if u is None else u
        w = self.w if w is None else w
        X, Z = self.X, self.Z
        Cu, Cw, aux = self.advection(u, w)
        ou = np.ones(self.shape_u)
        ow = np.ones(self.shape_w)
        mu = self.qform_u(ou, Cu)
        mw = self.qform_w(ow, Cw)
        h0z1 = Z.h0 @ np.ones(self.N)
        h0x1 = X.h0 @ np.ones(self.N)
        h1x1 = X.h1 @ np.ones(self.N - 1)
        h1z1 = Z.h1 @ np.ones(self.N - 1)
        bx1 = X.B @ np.ones(self.N - 1)
        bz1 = Z.B @ np.ones(self.N - 1)
        # u-component
        t1x = float(bx1 @ aux["s_x"] @ h0z1)
        t2x = float(np.sum(aux["a_x"] * ((X.B @ u) @ Z.h0)))
        t1z = float(h1x1 @ aux["t_z"] @ (Z.B.T @ np.ones(self.N)))
        t2z = float(np.sum(aux["a_z"] * (X.h1 @ u @ Z.B)))
        Fu = 0.5 * (t1x + t2x + t1z + t2z)
        # w-component
        s1z = float(bz1 @ aux["s_z"].T @ h0x1)
        s2z = float(np.sum(aux["b_z"] * (X.h0 @ w @ Z.B.T)))
        s1x = float((X.B.T @ np.ones(self.N)) @ aux["t_x"] @ h1z1)
        s2x = float(np.sum(aux["b_x"] * (X.B.T @ w @ Z.h1)))
        Fw = 0.5 * (s1z + s2z + s1x + s2x)
        scale = (sum(abs(self.qform_u(ou, t)) for t in aux["T"])
                 + sum(abs(self.qform_w(ow, t)) for t in aux["S"]) + 1e-300)
        # advecting-field component-complex divergence (should be a restriction
        # of the pressure divergence: zero on the constraint manifold)
        dv = self.divergence(u, w)
        delta1 = X.d10 @ aux["a_x"] + aux["a_z"] @ Z.d01.T
        delta2 = X.d01 @ aux["b_x"] + aux["b_z"] @ Z.d10.T
        rest1 = float(np.max(np.abs(delta1 - X.E @ dv)))
        rest2 = float(np.max(np.abs(delta2 - dv @ Z.E.T)))
        return dict(mu=mu, mw=mw, Fu=Fu, Fw=Fw, scale=scale,
                    res_u=abs(mu - Fu), res_w=abs(mw - Fw),
                    delta_restriction=max(rest1, rest2),
                    delta_norm=max(float(np.max(np.abs(delta1))), float(np.max(np.abs(delta2)))))

    # ---- viscous -------------------------------------------------------------
    def viscous(self, u: np.ndarray, w: np.ndarray):
        X, Z = self.X, self.Z
        Lu = X.D2d @ u + u @ Z.D2p.T
        Lw = X.D2p @ w + w @ Z.D2d.T
        return Lu, Lw

    def dissipation(self) -> float:
        X, Z = self.X, self.Z
        gux = X.sub_d10 @ self.u          # in h1'_x (x) h0_z
        guz = self.u @ Z.d10.T            # in h1_x  (x) h1_z
        gwx = X.d10 @ self.w              # in h1_x  (x) h1_z
        gwz = self.w @ Z.sub_d10.T        # in h0_x  (x) h1'_z
        d = (float(np.sum(gux * (X.sub_h1 @ gux @ Z.h0)))
             + float(np.sum(guz * (X.h1 @ guz @ Z.h1)))
             + float(np.sum(gwx * (X.h1 @ gwx @ Z.h1)))
             + float(np.sum(gwz * (X.h0 @ gwz @ Z.sub_h1))))
        return self.nu * d

    # ---- semidiscrete rate (projection handles the pressure) -----------------
    def rate(self, u: np.ndarray, w: np.ndarray):
        Cu, Cw, _ = self.advection(u, w)
        Lu, Lw = self.viscous(u, w)
        return -Cu + self.nu * Lu, -Cw + self.nu * Lw

    def step(self, dt: float) -> None:
        proj = self.projector
        u0, w0 = self.u.copy(), self.w.copy()
        ku, kw = self.rate(self.u, self.w)
        self.u, self.w = proj.apply(u0 + dt * ku, w0 + dt * kw)
        ku, kw = self.rate(self.u, self.w)
        self.u, self.w = proj.apply(0.75 * u0 + 0.25 * (self.u + dt * ku),
                                    0.75 * w0 + 0.25 * (self.w + dt * kw))
        ku, kw = self.rate(self.u, self.w)
        self.u, self.w = proj.apply(u0 / 3.0 + (2.0 / 3.0) * (self.u + dt * ku),
                                    w0 / 3.0 + (2.0 / 3.0) * (self.w + dt * kw))
        self._last_phi = proj.last_phi

    # ---- dense spline evaluation ---------------------------------------------
    def eval_u(self, xs: np.ndarray, zs: np.ndarray) -> np.ndarray:
        X, Z = self.X, self.Z
        cu = X.bhatinv @ self.u @ Z.binv.T
        Ph = bspline_evaluation_matrix(X.hknots, self.p - 1, np.atleast_1d(xs))
        Pz = bspline_evaluation_matrix(Z.knots, self.p, np.atleast_1d(zs))
        return Ph @ cu @ Pz.T

    def eval_w(self, xs: np.ndarray, zs: np.ndarray) -> np.ndarray:
        X, Z = self.X, self.Z
        cw = X.binv @ self.w @ Z.bhatinv.T
        Px = bspline_evaluation_matrix(X.knots, self.p, np.atleast_1d(xs))
        Pz = bspline_evaluation_matrix(Z.hknots, self.p - 1, np.atleast_1d(zs))
        return Px @ cw @ Pz.T


# ---------------------------------------------------------------------------
# constrained metric-orthogonal projection (cavity.cpp ConstrainedProjector)
# ---------------------------------------------------------------------------
class ConstrainedProjector:
    """min ||v - v*||_H  s.t.  D v = 0 and boundary trace = g.

    P = I - H^{-1} D^T Lp^+ D with Lp = D H^{-1} D^T fast-diagonalized:
    Lp = (d01x h1x^{-1} d01x^T) (x) h0z^{-1} + h0x^{-1} (x) (d01z h1z^{-1} d01z^T),
    solved in the tensor generalized eigenbasis (per-axis pencils (A, h0^{-1})).
    Boundary Schur block S_b = C P H^{-1} C^T, eigendecomposition pseudo-inverse
    with relative cutoff 1e-8 (near-null = trace combinations unreachable by
    weak-div-free fields).
    """

    def __init__(self, S: BMFNS2D, g_u: np.ndarray, g_w: np.ndarray, relcut: float = 1e-8,
                 quiet: bool = False) -> None:
        self.S = S
        X, Z, N = S.X, S.Z, S.N
        Ax = X.d01 @ X.h1inv @ X.d01.T
        Az = Z.d01 @ Z.h1inv @ Z.d01.T
        lx, Vx = eigh(0.5 * (Ax + Ax.T), 0.5 * (X.h0inv + X.h0inv.T))
        lz, Vz = eigh(0.5 * (Az + Az.T), 0.5 * (Z.h0inv + Z.h0inv.T))
        self.Vx, self.Vz = Vx, Vz
        den = lx[:, None] + lz[None, :]
        tol = 1e-12 * float(np.max(den))
        self.inv_den = np.where(den > tol, 1.0 / np.maximum(den, tol), 0.0)
        self.null_modes = int(np.sum(den <= tol))
        self.last_phi = np.zeros((N, N))

        # boundary index sets (nodal entries on the four walls of each space)
        bu = np.zeros(S.shape_u, dtype=bool)
        bu[[0, -1], :] = True
        bu[:, [0, -1]] = True
        bw = np.zeros(S.shape_w, dtype=bool)
        bw[[0, -1], :] = True
        bw[:, [0, -1]] = True
        self.bu, self.bw = bu, bw
        self.m1 = int(bu.sum())
        self.m = self.m1 + int(bw.sum())
        self.g = np.concatenate([g_u[bu], g_w[bw]])

        # S_b = C P H^{-1} C^T, column by column
        Sb = np.zeros((self.m, self.m))
        for c in range(self.m):
            u = np.zeros(S.shape_u)
            w = np.zeros(S.shape_w)
            if c < self.m1:
                u[np.unravel_index(np.flatnonzero(bu)[c], S.shape_u)] = 1.0
            else:
                w[np.unravel_index(np.flatnonzero(bw)[c - self.m1], S.shape_w)] = 1.0
            u = X.h1inv @ u @ Z.h0inv
            w = X.h0inv @ w @ Z.h1inv
            u, w = self.pfree(u, w)
            Sb[:, c] = np.concatenate([u[bu], w[bw]])
        Sb = 0.5 * (Sb + Sb.T)
        lam, V = np.linalg.eigh(Sb)
        cutoff = relcut * float(lam.max())
        self.kept = lam > cutoff
        self.lam, self.V = lam, V
        if not quiet:
            print(f" constrained projector: m={self.m}, eig range [{lam.min():.2e}, {lam.max():.2e}],"
                  f" kept {int(self.kept.sum())}/{self.m} modes, Lp null modes {self.null_modes}")

    def _sb_solve(self, b: np.ndarray) -> np.ndarray:
        y = self.V.T @ b
        y = np.where(self.kept, y / np.where(self.kept, self.lam, 1.0), 0.0)
        return self.V @ y

    def pfree(self, u: np.ndarray, w: np.ndarray):
        X, Z = self.S.X, self.S.Z
        dv = X.d01 @ u + w @ Z.d01.T
        phi = self.Vx @ ((self.Vx.T @ dv @ self.Vz) * self.inv_den) @ self.Vz.T
        self.last_phi = phi
        u = u - X.h1inv @ (X.d01.T @ phi) @ Z.h0inv
        w = w - X.h0inv @ (phi @ Z.d01) @ Z.h1inv
        return u, w

    def apply(self, u: np.ndarray, w: np.ndarray):
        X, Z = self.S.X, self.S.Z
        u, w = self.pfree(u, w)
        phi_div = self.last_phi
        rhs = np.concatenate([u[self.bu], w[self.bw]]) - self.g
        lam = self._sb_solve(rhs)
        cu = np.zeros(self.S.shape_u)
        cw = np.zeros(self.S.shape_w)
        cu[self.bu] = lam[:self.m1]
        cw[self.bw] = lam[self.m1:]
        cu = X.h1inv @ cu @ Z.h0inv
        cw = X.h0inv @ cw @ Z.h1inv
        cu, cw = self.pfree(cu, cw)
        self.last_phi = phi_div
        return u - cu, w - cw

    def trace_residual(self, u: np.ndarray, w: np.ndarray) -> float:
        return float(np.max(np.abs(np.concatenate([u[self.bu], w[self.bw]]) - self.g)))


class HomogeneousInteriorProjector:
    """Exact zero-trace + div-free projector (trace constraints eliminated).

    Used to manufacture gate states: boundary rows are hard-zeroed (traces
    exact) and the interior is H-orthogonally projected onto ker D. The
    divergence of the result is exactly zero up to eigensolve roundoff because
    D v lies in range(D_int) = null(L_int)^perp.
    """

    def __init__(self, S: BMFNS2D) -> None:
        self.S = S
        N = S.N
        iu = np.zeros(S.shape_u, dtype=bool)
        iu[1:-1, 1:-1] = True
        iw = np.zeros(S.shape_w, dtype=bool)
        iw[1:-1, 1:-1] = True
        self.iu, self.iw = iu, iw
        idx = np.concatenate([np.flatnonzero(iu.ravel()),
                              S.nu_dof + np.flatnonzero(iw.ravel())])
        self.idx = idx
        Hu = np.kron(S.X.h1, S.Z.h0)
        Hw = np.kron(S.X.h0, S.Z.h1)
        H = np.zeros((S.nu_dof + S.nw_dof, S.nu_dof + S.nw_dof))
        H[:S.nu_dof, :S.nu_dof] = Hu
        H[S.nu_dof:, S.nu_dof:] = Hw
        Hii = H[np.ix_(idx, idx)]
        cols = []
        for k in idx:
            v = np.zeros(S.nu_dof + S.nw_dof)
            v[k] = 1.0
            cols.append(S.divergence(v[:S.nu_dof].reshape(S.shape_u),
                                     v[S.nu_dof:].reshape(S.shape_w)).ravel())
        Dint = np.array(cols).T
        self.HiDT = np.linalg.solve(Hii, Dint.T)
        Lint = Dint @ self.HiDT
        lam, V = np.linalg.eigh(0.5 * (Lint + Lint.T))
        self.keep = lam > 1e-12 * float(lam.max())
        self.lam, self.V = lam, V
        self.Dint = Dint

    def apply(self, u: np.ndarray, w: np.ndarray):
        S = self.S
        u = u.copy()
        w = w.copy()
        u[~self.iu] = 0.0
        w[~self.iw] = 0.0
        vi = np.concatenate([u.ravel(), w.ravel()])[self.idx]
        r = self.Dint @ vi
        y = self.V[:, self.keep] @ ((self.V[:, self.keep].T @ r) / self.lam[self.keep])
        vi = vi - self.HiDT @ y
        out = np.zeros(S.nu_dof + S.nw_dof)
        out[self.idx] = vi
        return out[:S.nu_dof].reshape(S.shape_u), out[S.nu_dof:].reshape(S.shape_w)


# ---------------------------------------------------------------------------
# lid data
# ---------------------------------------------------------------------------
def lid_targets(S: BMFNS2D):
    """Regularized lid u = 16 x^2 (1-x)^2 interpolated on the eta grid; the top
    row of u's boundary carries it (z-primal index N-1); everything else 0."""
    g_u = np.zeros(S.shape_u)
    eta = S.X.eta
    g_u[:, -1] = 16.0 * eta**2 * (1.0 - eta) ** 2
    g_w = np.zeros(S.shape_w)
    return g_u, g_w


def tg_ic(S: BMFNS2D):
    """Taylor-Green-like no-slip decay: u = dpsi/dz, w = -dpsi/dx of
    psi = sin^2(pi x) sin^2(pi z) / pi, sampled on the face grids. Both
    components vanish identically on every wall, so the field is exactly
    no-slip; the projection only removes its small discrete divergence."""
    ex, xz = S.X.eta, S.Z.x
    u0 = (np.sin(np.pi * ex)[:, None] ** 2) * np.sin(2.0 * np.pi * xz)[None, :]
    xx, ez = S.X.x, S.Z.eta
    w0 = -np.sin(2.0 * np.pi * xx)[:, None] * (np.sin(np.pi * ez)[None, :] ** 2)
    return u0, w0


# ---------------------------------------------------------------------------
# time marching driver
# ---------------------------------------------------------------------------
def march(S: BMFNS2D, T: float, report_every: int | None = None, verbose: bool = True,
          ckpt_path=None, ckpt_every: int = 4000, tg: bool = False):
    dt = float(os.environ.get("BMFNS_DT_SCALE", "0.4")) * 0.1 / S.N
    nsteps = max(1, int(round(T / dt)))
    if report_every is None:
        report_every = max(1, nsteps // 50)
    if tg:
        g_u, g_w = np.zeros(S.shape_u), np.zeros(S.shape_w)   # no-slip everywhere
        u_init, w_init = tg_ic(S)
    else:
        g_u, g_w = lid_targets(S)
        u_init, w_init = np.zeros(S.shape_u), np.zeros(S.shape_w)
    S.projector = ConstrainedProjector(S, g_u, g_w, quiet=not verbose)
    S.u, S.w = S.projector.apply(u_init, w_init)
    n_start = 0
    if ckpt_path is not None and Path(ckpt_path).exists():
        z = np.load(ckpt_path)
        if int(z["N"]) == S.N and float(z["Re"]) == float(S.Re):
            S.u[:] = z["u"]
            S.w[:] = z["w"]
            n_start = int(z["n"])
            if verbose:
                print(f" ** resumed from {ckpt_path} at step {n_start}, "
                      f"t={n_start * dt:.4f} **")
    hist = []
    u_prev, w_prev = S.u.copy(), S.w.copy()
    max_dudt = 1.0
    steady_step = -1
    t = n_start * dt
    t0 = time.time()
    cpu0 = time.process_time()
    for n in range(n_start, nsteps + 1):
        if ckpt_path is not None and n > n_start and n % ckpt_every == 0:
            tmp = str(ckpt_path) + ".tmp.npz"   # np.savez keeps an .npz name as-is
            np.savez(tmp, u=S.u, w=S.w, n=n, N=S.N, Re=float(S.Re))
            os.replace(tmp, ckpt_path)          # atomic swap; no torn checkpoint
        if n % report_every == 0 or n == nsteps:
            ke = S.kinetic_energy()
            prod, pscale = S.ke_production()
            divrel = S.div_relative()
            hist.append(dict(step=n, t=t, KE=ke, ke_production=prod, div_rel=divrel,
                             max_dudt=max_dudt))
            if verbose:
                print(f"  n={n:6d} t={t:8.4f} KE={ke:.6e} KEprod={prod:+.3e} "
                      f"div_rel={divrel:.2e} max|du/dt|={max_dudt:.3e}")
            if not np.isfinite(ke):
                raise RuntimeError("non-finite state during march")
        if n < nsteps:
            S.step(dt)
            t += dt
            max_dudt = max(float(np.max(np.abs(S.u - u_prev))),
                           float(np.max(np.abs(S.w - w_prev)))) / dt
            u_prev[:] = S.u
            w_prev[:] = S.w
            if steady_step < 0 and max_dudt < 1e-9:
                steady_step = n + 1
                if verbose:
                    print(f" ** steady at step {n + 1}, t={t:.4f} **")
                break
    wall = time.time() - t0
    cpu = time.process_time() - cpu0
    return dict(dt=dt, nsteps=nsteps, t=t, hist=hist, wall=wall, cpu=cpu,
                steady_step=steady_step, max_dudt=max_dudt)


def run_driver(N: int, Re: float, T: float, outprefix: str, tg: bool = False) -> None:
    p = 4
    S = BMFNS2D(N, p, Re)
    print("=" * 65)
    print(" BMF staggered primitive-variable NS (Theorem thm:ns-conservation)")
    print(f" N={N}^2 p={p} Re={Re:.1f} nu={S.nu:.6e} "
          f"{'Taylor-Green no-slip decay' if tg else 'regularized lid'}, cproj free rates")
    print("=" * 65)
    CAVITY_DATA.mkdir(parents=True, exist_ok=True)
    res = march(S, T, ckpt_path=CAVITY_DATA / f"bmfns_{outprefix}_ckpt.npz", tg=tg)
    dt = res["dt"]

    # final diagnostics
    ke = S.kinetic_energy()
    kb = S.ke_balance()
    prod = kb["prod"]
    diss = S.dissipation()
    dv = S.divergence()
    dvHp = S.X.h0 @ dv @ S.Z.h0    # Hp moments of the divergence
    pressure_work = float(np.sum(S._last_phi * dvHp)) / dt
    trace_res = S.projector.trace_residual(S.u, S.w)

    # centerline extraction by dense exact spline evaluation
    nd = 2001
    zs = np.linspace(0.0, 1.0, nd)
    xs = np.linspace(0.0, 1.0, nd)
    uc = S.eval_u(np.array([0.5]), zs).ravel()
    wc = S.eval_w(xs, np.array([0.5])).ravel()
    margin = 0.04
    umin_i = int(np.argmin(uc))
    inner = (xs > margin) & (xs < 1.0 - margin)
    wmin_i = int(np.flatnonzero(inner)[np.argmin(wc[inner])])
    wmax_i = int(np.flatnonzero(inner)[np.argmax(wc[inner])])

    CAVITY_DATA.mkdir(parents=True, exist_ok=True)
    with open(CAVITY_DATA / f"bmfns_{outprefix}_centerline_u.csv", "w") as f:
        f.write("z,u\n")
        for z, v in zip(zs, uc):
            f.write(f"{z:.10e},{v:.10e}\n")
    with open(CAVITY_DATA / f"bmfns_{outprefix}_centerline_w.csv", "w") as f:
        f.write("x,w\n")
        for x, v in zip(xs, wc):
            f.write(f"{x:.10e},{v:.10e}\n")

    summary = {
        "N": N, "p": p, "Re": Re, "nu": S.nu,
        "advection": "bmf_metric_adjoint_skew_product_map_split",
        "dt": dt, "dt_scale": 0.4, "nsteps": res["nsteps"], "T": res["t"], "t_start": 0.0,
        "lid_profile": "regularized_16x2_1mx2",
        "steps_done": res["nsteps"] if res["steady_step"] < 0 else res["steady_step"],
        "wall_seconds": res["wall"], "cpu_seconds": res["cpu"],
        "cpu_seconds_per_step": res["cpu"] / max(1, res["nsteps"]),
        "final_KE": ke,
        "final_ke_production": prod,
        "final_ke_boundary_work": kb["F"],
        "final_ke_balance_residual_rel": kb["residual"] / kb["scale"],
        "final_pressure_work": pressure_work,
        "final_dissipation": diss,
        "steady_step": res["steady_step"], "steady_tol": 1e-9, "final_max_dudt": res["max_dudt"],
        "final_constraint_l2_div": float(np.linalg.norm(dv)),
        "final_constraint_max_div": float(np.max(np.abs(dv))),
        "final_constraint_rel_div": S.div_relative(),
        "final_trace_residual": trace_res,
        "u_centerline_min": float(uc[umin_i]), "u_centerline_min_z": float(zs[umin_i]),
        "w_centerline_min": float(wc[wmin_i]), "w_centerline_min_x": float(xs[wmin_i]),
        "w_centerline_max": float(wc[wmax_i]), "w_centerline_max_x": float(xs[wmax_i]),
    }
    out = CAVITY_DATA / f"bmfns_{outprefix}_run_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f" centerline u_min={uc[umin_i]:.6f} at z={zs[umin_i]:.4f} ; "
          f"w_min={wc[wmin_i]:.6f} at x={xs[wmin_i]:.4f} ; w_max={wc[wmax_i]:.6f} at x={xs[wmax_i]:.4f}")
    print(f" final: KE={ke:.6e} KEprod={prod:.3e} diss={diss:.6e} pwork={pressure_work:.3e} "
          f"div_rel={S.div_relative():.2e} trace_res={trace_res:.2e}")
    print(f" wrote {out}")


# ---------------------------------------------------------------------------
# verification gates
# ---------------------------------------------------------------------------
def gate_g1_g2_g3(N: int, nstates: int = 20, seed: int = 12345):
    """G1 (divergence), G2 (KE production), G3 (momentum = boundary flux only)
    on random discretely divergence-free homogeneous states."""
    S = BMFNS2D(N, 4, 400.0)
    hp = HomogeneousInteriorProjector(S)
    rng = np.random.default_rng(seed)
    g1_max = 0.0
    g2_max = 0.0
    g3_max = 0.0
    flux_max = 0.0
    delta_max = 0.0
    for _ in range(nstates):
        u0 = rng.standard_normal(S.shape_u)
        w0 = rng.standard_normal(S.shape_w)
        u, w = hp.apply(u0, w0)
        ke = 0.5 * (S.qform_u(u, u) + S.qform_w(w, w))
        s = 1.0 / math.sqrt(2.0 * ke)
        u *= s
        w *= s
        # G1
        dv = S.divergence(u, w)
        den = np.linalg.norm(S.X.d01 @ u) + np.linalg.norm(w @ S.Z.d01.T) + 1e-300
        g1_max = max(g1_max, float(np.linalg.norm(dv) / den))
        # G2
        S.u, S.w = u, w
        prod, _ = S.ke_production(u, w)
        g2_max = max(g2_max, abs(prod) / 0.5)      # KE normalized to 1/2
        # G3
        mb = S.momentum_balance(u, w)
        g3_max = max(g3_max, max(mb["res_u"], mb["res_w"]) / mb["scale"])
        flux_max = max(flux_max, max(abs(mb["Fu"]), abs(mb["Fw"])))
        delta_max = max(delta_max, mb["delta_norm"])
    # also exercise the full constrained projector on homogeneous data (G1 path
    # used by the solver itself)
    proj = ConstrainedProjector(S, np.zeros(S.shape_u), np.zeros(S.shape_w), quiet=True)
    u, w = proj.apply(rng.standard_normal(S.shape_u), rng.standard_normal(S.shape_w))
    dv = S.divergence(u, w)
    den = np.linalg.norm(S.X.d01 @ u) + np.linalg.norm(w @ S.Z.d01.T) + 1e-300
    g1_full = float(np.linalg.norm(dv) / den)
    g1_max = max(g1_max, g1_full)
    trace_full = proj.trace_residual(u, w)
    return dict(g1=g1_max, g2=g2_max, g3=g3_max, boundary_flux_max=flux_max,
                advecting_delta_max=delta_max, g1_fullprojector=g1_full,
                trace_residual_fullprojector=trace_full, N=N, nstates=nstates)


def gate_g4(N: int):
    """Heat-equation MMS on each face space: du/dt = nu L u with the zero
    Dirichlet trace enforced by the METRIC-ORTHOGONAL trace projection (the
    paper's strong enforcement; hard row-pinning is not H-orthogonal and is
    unstable for the non-diagonal spline metric), decaying sin(pi x) sin(pi z)
    at the analytic rate 2 pi^2 nu."""
    S = BMFNS2D(N, 4, 1.0)   # nu = 1
    nu = 1.0
    results = {}
    for comp in ("u", "w"):
        if comp == "u":
            xs, zs = S.X.eta, S.Z.x
            A2a, A2b = S.X.D2d, S.Z.D2p
            Ha, Hb = S.X.h1, S.Z.h0
            shape = S.shape_u
        else:
            xs, zs = S.X.x, S.Z.eta
            A2a, A2b = S.X.D2p, S.Z.D2d
            Ha, Hb = S.X.h0, S.Z.h1
            shape = S.shape_w
        # H-orthogonal projector onto zero traces: v <- v - H^{-1}C^T (C H^{-1} C^T)^{-1} C v
        bmask = np.zeros(shape, dtype=bool)
        bmask[[0, -1], :] = True
        bmask[:, [0, -1]] = True
        m = int(bmask.sum())
        Hainv = np.linalg.inv(Ha)
        Hbinv = np.linalg.inv(Hb)
        cols = np.zeros((shape[0] * shape[1], m))
        for c, k in enumerate(np.flatnonzero(bmask.ravel())):
            e = np.zeros(shape)
            e[np.unravel_index(k, shape)] = 1.0
            cols[:, c] = (Hainv @ e @ Hbinv).ravel()
        CHC = cols.reshape(-1, m)[np.flatnonzero(bmask.ravel()), :]
        T = cols @ np.linalg.inv(CHC)

        def proj(v):
            return v - (T @ v[bmask]).reshape(shape)

        def rhs(v):
            return nu * (A2a @ v + v @ A2b.T)

        lam_max = float(np.max(np.abs(np.linalg.eigvals(A2a))) + np.max(np.abs(np.linalg.eigvals(A2b))))
        dt = 1.2 / (nu * lam_max)
        Ttot = 0.02
        nsteps = max(1, int(round(Ttot / dt)))
        dt = Ttot / nsteps
        v = proj(np.outer(np.sin(np.pi * xs), np.sin(np.pi * zs)))
        e0 = math.sqrt(float(np.sum(v * (Ha @ v @ Hb))))
        for _ in range(nsteps):
            v1 = proj(v + dt * rhs(v))
            v2 = proj(0.75 * v + 0.25 * (v1 + dt * rhs(v1)))
            v = proj(v / 3.0 + (2.0 / 3.0) * (v2 + dt * rhs(v2)))
        e1 = math.sqrt(float(np.sum(v * (Ha @ v @ Hb))))
        rate = math.log(e0 / e1) / Ttot
        exact = 2.0 * math.pi**2 * nu
        results[comp] = dict(rate=rate, exact=exact, rel_err=abs(rate - exact) / exact,
                             nsteps=nsteps, dt=dt)
    return results


def gate_g5(N: int = 24, Re: float = 400.0, T: float = 5.0):
    S = BMFNS2D(N, 4, Re)
    res = march(S, T, verbose=False)
    kes = [h["KE"] for h in res["hist"]]
    divs = [h["div_rel"] for h in res["hist"]]
    kb = S.ke_balance()   # KE production must equal the SBP boundary (lid) work
    return dict(N=N, Re=Re, T=res["t"], KE_final=kes[-1], KE_max=max(kes),
                finite=bool(np.all(np.isfinite(kes))),
                ke_production_final=kb["prod"], ke_boundary_work=kb["F"],
                ke_balance_residual_rel=kb["residual"] / kb["scale"],
                div_rel_max=max(divs), dissipation=S.dissipation(),
                trace_residual=S.projector.trace_residual(S.u, S.w),
                wall_seconds=res["wall"])


def direct_galerkin_momentum_defect(N: int, seed: int = 5) -> float:
    """The measured global-momentum defect of the plain pointwise Galerkin skew
    form on a random discretely-div-free homogeneous state (the reason the
    product-map split is used instead; printed for the record)."""
    S = BMFNS2D(N, 4, 400.0)
    X, Z = S.X, S.Z
    hp = HomogeneousInteriorProjector(S)
    rng = np.random.default_rng(seed)
    u, w = hp.apply(rng.standard_normal(S.shape_u), rng.standard_normal(S.shape_w))
    ke = 0.5 * (S.qform_u(u, u) + S.qform_w(w, w))
    s = 1.0 / math.sqrt(2 * ke)
    u *= s
    w *= s
    cu = X.bhatinv @ u @ Z.binv.T
    cw = X.binv @ w @ Z.bhatinv.T
    dP0x = bspline_evaluation_matrix(X.knots, S.p, X.xq, 1)
    dF0x = bspline_evaluation_matrix(X.hknots, S.p - 1, X.xq, 1)
    dP0z = bspline_evaluation_matrix(Z.knots, S.p, Z.xq, 1)
    dF0z = bspline_evaluation_matrix(Z.hknots, S.p - 1, Z.xq, 1)
    P0x = bspline_evaluation_matrix(X.knots, S.p, X.xq)
    F0x = bspline_evaluation_matrix(X.hknots, S.p - 1, X.xq)
    P0z = bspline_evaluation_matrix(Z.knots, S.p, Z.xq)
    F0z = bspline_evaluation_matrix(Z.hknots, S.p - 1, Z.xq)
    Uq = F0x @ cu @ P0z.T
    Uxq = dF0x @ cu @ P0z.T
    Uzq = F0x @ cu @ dP0z.T
    Wq = P0x @ cw @ F0z.T
    Wzq = P0x @ cw @ dF0z.T
    # N_u = 1/2[(U.grad)u + div(U u)] = 3/2 u u_x + w u_z + 1/2 w_z u
    Nu = 1.5 * Uq * Uxq + Wq * Uzq + 0.5 * Wzq * Uq
    mom = float(np.sum(S.WQ * Nu))   # <1, N_u> = 1^T H C for the weak form
    return mom


def run_gates(smoke: bool) -> None:
    ensure_generated_dirs()
    N123 = 17 if smoke else 24
    N4 = 17 if smoke else 33
    N5, T5 = (17, 1.0) if smoke else (24, 5.0)
    label = "SMOKE" if smoke else "VERIFY"
    S = BMFNS2D(N123, 4, 400.0)
    print("=" * 72)
    print(f" BMF NS2D verification gates ({label})")
    print("=" * 72)
    print(" operator construction checks:")
    print(f"   1D SBP residual |h0 d01 + d10^T h1 - B|: x={S.X.sbp_residual:.2e}")
    print(f"   derivative space == primal(N-1,p-1): knots {S.X.subknots_residual:.2e}, "
          f"metric {S.X.submetric_residual:.2e}")
    print(f"   intertwining |d10 L - E d01|: {S.X.intertwine_residual:.2e}")
    print(f"   lift/restriction consistency on smooth data: L {S.X.lift_consistency:.2e}, "
          f"E {S.X.restrict_consistency:.2e}")
    defect = direct_galerkin_momentum_defect(N123)
    print(f"   [context] plain Galerkin skew-form momentum defect on a div-free state: "
          f"{defect:.3e}  (why the product-map split is used)")
    print("-" * 72)

    g = gate_g1_g2_g3(N123)
    ok1 = g["g1"] < 1e-13
    ok2 = g["g2"] < 1e-12
    ok3 = g["g3"] < 1e-12
    print(f" G1 divergence of projected states (N={N123}, {g['nstates']} states + full projector):")
    print(f"    max relative divergence = {g['g1']:.3e}  (full projector path {g['g1_fullprojector']:.3e})"
          f"  -> {'PASS' if ok1 else 'FAIL'} (< 1e-13)")
    print(f" G2 advective KE production on div-free homogeneous states:")
    print(f"    max |u^T Hu Cu + w^T Hw Cw| / KE = {g['g2']:.3e}  -> {'PASS' if ok2 else 'FAIL'} (< 1e-12)")
    print(f" G3 momentum: 1^T H C = SBP boundary flux only:")
    print(f"    max |1^T H C - F_boundary| / scale = {g['g3']:.3e}  -> {'PASS' if ok3 else 'FAIL'} (< 1e-12)")
    print(f"    (SBP boundary flux magnitude up to {g['boundary_flux_max']:.3e}; advecting-field")
    print(f"     component-complex divergence max {g['advecting_delta_max']:.3e}; full-projector")
    print(f"     trace residual {g['trace_residual_fullprojector']:.3e})")

    g4 = gate_g4(N4)
    ok4 = all(r["rel_err"] < 1e-3 for r in g4.values())
    print(f" G4 heat MMS on each face space (N={N4}):")
    for comp, r in g4.items():
        print(f"    {comp}-space: rate {r['rate']:.6f} vs 2 pi^2 = {r['exact']:.6f}, "
              f"rel err {r['rel_err']:.3e} ({r['nsteps']} steps)")
    print(f"    -> {'PASS' if ok4 else 'FAIL'} (< 1e-3)")

    g5 = gate_g5(N5, 400.0, T5)
    ok5 = (g5["finite"] and g5["KE_max"] < 10.0 and g5["div_rel_max"] < 1e-12
           and g5["ke_balance_residual_rel"] < 1e-12)
    print(f" G5 regularized-lid run N={g5['N']}, Re={g5['Re']:.0f}, T={g5['T']:.1f}:")
    print(f"    finite={g5['finite']}, KE_final={g5['KE_final']:.6e}, KE_max={g5['KE_max']:.6e}")
    print(f"    div_rel_max={g5['div_rel_max']:.3e}")
    print(f"    KE production {g5['ke_production_final']:+.6e} vs SBP boundary (lid) work "
          f"{g5['ke_boundary_work']:+.6e}: balance residual rel={g5['ke_balance_residual_rel']:.3e}")
    print(f"    dissipation={g5['dissipation']:.6e}, trace residual={g5['trace_residual']:.3e}, "
          f"wall={g5['wall_seconds']:.1f}s")
    print(f"    -> {'PASS' if ok5 else 'FAIL'} (stable, KE bounded, div at floor, KE balance closed)")

    allok = ok1 and ok2 and ok3 and ok4 and ok5
    print("-" * 72)
    print(f" GATES: {'ALL PASS' if allok else 'FAILURES PRESENT'}")
    summary = dict(mode=label.lower(), G1=g, G4=g4, G5=g5,
                   direct_galerkin_momentum_defect=defect,
                   passed=dict(G1=ok1, G2=ok2, G3=ok3, G4=ok4, G5=ok5, all=allok))
    out = RESULTSDIR / ("bmf_ns2d_smoke_summary.json" if smoke else "bmf_ns2d_verify_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f" wrote {out}")
    if not allok:
        sys.exit(1)


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__.split("\n\n")[0])
        print("usage: bmf_ns2d.py --smoke | --verify | run N Re T outprefix")
        return
    if args[0] == "--smoke":
        run_gates(smoke=True)
    elif args[0] == "--verify":
        run_gates(smoke=False)
    elif args[0] == "run":
        N = int(args[1]) if len(args) > 1 else 24
        Re = float(args[2]) if len(args) > 2 else 400.0
        T = float(args[3]) if len(args) > 3 else 5.0
        outprefix = args[4] if len(args) > 4 else f"N{N}_Re{int(Re)}"
        tg = len(args) > 5 and args[5] == "tg"
        run_driver(N, Re, T, outprefix, tg=tg)
    else:
        raise SystemExit(f"unknown mode {args[0]!r}")


if __name__ == "__main__":
    main()
