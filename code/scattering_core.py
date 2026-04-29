r"""Three-dimensional transient acoustic cavity core.

This module is the reusable numerical kernel for Section 6.10 of the
manuscript: a bounded cubic domain with acoustically soft walls, an optional
interior sphere surface used either as an immersed boundary or as a passive
observation surface, and a fully consistent off-grid evaluation pipeline that
delivers pressure, normal velocity, surface traction, integrated force, and
energy diagnostics on that exact curved surface.

Governing system.
The linearized acoustic equations on the closed box :math:`\\Omega = [0, L]^3`
are discretized in the first-order form

.. math::
   p_t + \\rho_0 c^2 \\, \\nabla\\!\\cdot\\!\\vec u = 0,
   \\qquad
   \\vec u_t + \\frac{1}{\\rho_0}\\nabla p = 0,

with acoustically-soft Dirichlet walls :math:`p = 0` on every face of the
box, which makes the boundary-work integral vanish identically and leaves
the discrete energy estimate closable for every family in the comparison
without needing a family-specific SAT correction. Acoustic waves still
reflect off the box walls, and the reflection carries the standard
soft-wall sign flip. Background medium parameters :math:`\\rho_0, c` are
fixed; setting them to unity is the default in this module.

Discretization.
A single one-dimensional derivative operator :math:`D` from
``boundary_spline_analysis.build_operator`` is tensor-producted into three
axes so that the discrete gradient and divergence inherit the family's
own boundary closure. Because the same :math:`D` is used on every axis,
the full 3D scheme inherits exactly the formal order, wall response, and
structural identities of the underlying 1D operator.

Time integration.
A classical four-stage, fourth-order Runge--Kutta integrator advances the
state :math:`(p, u, v, w)`. The time step is chosen per scheme to sit well
inside the stability envelope; for the verification sweeps below, the
time-integration error is driven far below spatial error by an additional
safety factor.

Immersed body.
When the solver is configured with an immersed sphere, two immersed
conditions are available. A soft sphere enforces :math:`p = 0` through a
per-stage surface-pressure penalty. A rigid sphere enforces
:math:`u_n = \vec u\cdot\vec n = 0` through a per-stage normal-velocity
penalty applied to the momentum equations. In both cases the interpolation
stencil that moves grid data to the smooth surface quadrature points is
the scheme's own one-dimensional reconstruction tensored along three axes;
the adjoint of that stencil is used to spread the penalty back to the
grid. The surface quadrature nodes follow a uniform Fibonacci-spiral
pattern, which makes the surface weight constant and the spherical
integrals unbiased.

Surface observables.
At every recording cadence the module evaluates, on the exact analytic
sphere, the surface pressure, the normal velocity, the traction :math:`-p`,
the net integrated force :math:`\\vec F(t) = -\\!\\oint p\\,\\vec n\\,dS`,
and the cumulative normal work. These are the same quantities the
bounded-interval geometry-transfer and immersed-traction subsections of
the manuscript evaluate in two dimensions; the machinery here is their
three-dimensional generalization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

import boundary_spline_analysis as ba


# ---------------------------------------------------------------------------
# Tensor-product 3D operator.
# ---------------------------------------------------------------------------


class Operator3D:
    """Apply a family's one-dimensional derivative along each Cartesian axis."""

    def __init__(self, label: str, npts: int, length: float = 1.0) -> None:
        self.label = label
        self.family = ba.scheme_family(label)
        self.npts = int(npts)
        self.length = float(length)
        self._op1d = ba.build_operator(
            ba.scheme_kind(label), ba.scheme_order(label), self.npts, self.length
        )
        self.x = np.asarray(self._op1d.x, dtype=float)
        self.h_nominal = self.length / (self.npts - 1)
        self.h = float(np.min(np.diff(self.x)))
        self._D = np.asarray(self._op1d.matrix, dtype=float)

    # Derivatives along each axis via dense tensor contraction.
    def dx(self, U: np.ndarray) -> np.ndarray:
        return np.einsum("ia,ajk->ijk", self._D, U, optimize=True)

    def dy(self, U: np.ndarray) -> np.ndarray:
        return np.einsum("ja,iak->ijk", self._D, U, optimize=True)

    def dz(self, U: np.ndarray) -> np.ndarray:
        return np.einsum("ka,ija->ijk", self._D, U, optimize=True)

    def grad(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.dx(P), self.dy(P), self.dz(P)

    def div(self, Ux: np.ndarray, Uy: np.ndarray, Uz: np.ndarray) -> np.ndarray:
        return self.dx(Ux) + self.dy(Uy) + self.dz(Uz)

# One-dimensional metric consistent with the scheme family.
    @property
    def metric_1d(self) -> np.ndarray:
        fam = self.family
        if fam == "SBP":
            metric = getattr(self._op1d, "metric", None)
            if metric is not None:
                m = np.asarray(metric, dtype=float)
                if m.ndim == 2:
                    return np.diag(m)
                return m
        if fam == "BMG":
            metric = getattr(self._op1d, "metric", None)
            if metric is not None:
                return np.asarray(metric, dtype=float)
        if fam == "BMF":
            _, H = ba.bmf_full_metric(self.npts, ba.scheme_order(self.label), self.length)
            return np.asarray(H, dtype=float)
        return ba.trapezoidal_weights(self.npts, self.length)

    @property
    def diag_metric_1d(self) -> np.ndarray:
        metric = np.asarray(self.metric_1d, dtype=float)
        if metric.ndim == 2:
            return np.diag(metric)
        return metric


@dataclass
class BMFStaggeredAcousticState:
    """BMF acoustic state on the staggered spline complex.

    Pressure is stored on the primal Greville tensor grid. The velocity
    components are stored on derivative Greville grids in their own
    coordinate direction and on primal grids in the transverse directions.
    """

    p: np.ndarray          # (N, N, N)
    u: np.ndarray          # (N-1, N, N)
    v: np.ndarray          # (N, N-1, N)
    w: np.ndarray          # (N, N, N-1)


class BMFStaggeredOperator3D:
    """Tensor-product BMF staggered gradient-divergence complex."""

    def __init__(self, label: str, npts: int, length: float = 1.0) -> None:
        if ba.scheme_family(label) != "BMF":
            raise ValueError("BMFStaggeredOperator3D requires a BMF label")
        self.label = label
        self.family = "BMF"
        self.npts = int(npts)
        self.length = float(length)
        self.degree = ba.scheme_order(label)
        self._op1d = ba.build_operator("B", self.degree, self.npts, self.length)
        self.ops = ba.bmf_staggered_operators(self.npts, self.degree, self.length)
        self.x = np.asarray(self.ops.x, dtype=float)
        self.eta = np.asarray(self.ops.eta, dtype=float)
        self.h_nominal = self.length / (self.npts - 1)
        self.h = float(min(np.min(np.diff(self.x)), np.min(np.diff(self.eta))))

    def grad(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d10 = self.ops.d10
        gx = np.einsum("ia,ajk->ijk", d10, P, optimize=True)
        gy = np.einsum("ja,iak->ijk", d10, P, optimize=True)
        gz = np.einsum("ka,ija->ijk", d10, P, optimize=True)
        return gx, gy, gz

    def div(self, Ux: np.ndarray, Uy: np.ndarray, Uz: np.ndarray) -> np.ndarray:
        d01 = self.ops.d01
        dx = np.einsum("ia,ajk->ijk", d01, Ux, optimize=True)
        dy = np.einsum("ja,iak->ijk", d01, Uy, optimize=True)
        dz = np.einsum("ka,ija->ijk", d01, Uz, optimize=True)
        return dx + dy + dz


# ---------------------------------------------------------------------------
# Sphere surface quadrature + off-grid reconstruction.
# ---------------------------------------------------------------------------


def fibonacci_sphere_points(n: int, center: tuple[float, float, float], radius: float
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (points, normals, weights) for the unit-weight Fibonacci spiral.

    The weight :math:`w_m = 4\\pi R^2 / n` is constant by construction; the
    points are uniformly distributed with asymptotic :math:`O(1/n)` angular
    equidistribution.
    """
    idx = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * idx
    nx = np.sin(phi) * np.cos(theta)
    ny = np.sin(phi) * np.sin(theta)
    nz = np.cos(phi)
    normals = np.stack([nx, ny, nz], axis=1)
    cx, cy, cz = center
    points = np.stack([cx + radius * nx, cy + radius * ny, cz + radius * nz], axis=1)
    area = 4.0 * np.pi * radius * radius
    weights = np.full(n, area / float(n), dtype=float)
    return points, normals, weights


def _axis_eval_matrix(grid: np.ndarray, coords: np.ndarray, family: str,
                      degree: int, *, op1d=None) -> np.ndarray:
    """Return a (M, N) linear map ``coords -> off-grid value`` along one axis.

    For spline families (BMF, BMG) the map is :math:`W = \\Phi A^{-1}`, where
    :math:`\\Phi` is the clamped B-spline basis evaluated at ``coords`` and
    :math:`A` is the 1D nodal-to-control Greville sample matrix. For nodal
    families (E, C, SBP, DRP) the map is the barycentric Lagrange interpolant
    on the family's natural stencil size :math:`q+1`.
    """
    coords = np.asarray(coords, dtype=float)
    n = grid.size
    if family in ("BMF", "BMG"):
        if op1d is None:
            raise ValueError("spline axis map needs the 1D operator for A^{-1}")
        _, phi, _ = ba.spline_eval_matrices(coords, grid, degree)
        bmat = np.asarray(op1d._bmat, dtype=float)
        Ainv = np.linalg.inv(bmat)
        return phi @ Ainv
    stencil_size = degree + 1
    return ba.local_interpolation_matrix(grid, coords, stencil_size)


@dataclass
class SphereSurface:
    """Precomputed off-grid sphere-surface interpolation pipeline.

    Holds the surface quadrature nodes and the three axis-aligned row
    matrices :math:`W_x, W_y, W_z` so that any surface sample of a 3D grid
    field is a single tensor contraction. The same pipeline is reused for
    pressure, each velocity component, gradient components (via a pre-applied
    grid derivative), and the adjoint spreading used in the surface-pressure
    penalty.
    """

    points: np.ndarray         # (M, 3) Cartesian surface nodes.
    normals: np.ndarray        # (M, 3) outward unit normals.
    weights: np.ndarray        # (M,)  quadrature weights, sum = 4 pi R^2.
    center: np.ndarray         # (3,)  sphere centre in world coords.
    radius: float
    Wx: np.ndarray             # (M, N) evaluation rows along x-axis.
    Wy: np.ndarray
    Wz: np.ndarray

    def interpolate(self, P: np.ndarray) -> np.ndarray:
        """Evaluate :math:`P` at every surface quadrature node."""
        tmp = np.einsum("mi,ijk->mjk", self.Wx, P, optimize=True)
        tmp = np.einsum("mj,mjk->mk", self.Wy, tmp, optimize=True)
        return np.einsum("mk,mk->m", self.Wz, tmp, optimize=True)

    def spread(self, surface_values: np.ndarray) -> np.ndarray:
        """Spread scalar surface values back to the grid via the adjoint stencil."""
        tmp = np.einsum("mk,m->mk", self.Wz, surface_values, optimize=True)
        tmp = np.einsum("mj,mk->mjk", self.Wy, tmp, optimize=True)
        return np.einsum("mi,mjk->ijk", self.Wx, tmp, optimize=True)


def build_sphere_surface(operator: Operator3D, center: tuple[float, float, float],
                         radius: float, n_quad: int) -> SphereSurface:
    family = operator.family
    degree = ba.scheme_order(operator.label)
    grid = operator.x
    points, normals, weights = fibonacci_sphere_points(n_quad, center, radius)

    Wx = _axis_eval_matrix(grid, points[:, 0], family, degree, op1d=operator._op1d)
    Wy = _axis_eval_matrix(grid, points[:, 1], family, degree, op1d=operator._op1d)
    Wz = _axis_eval_matrix(grid, points[:, 2], family, degree, op1d=operator._op1d)

    return SphereSurface(
        points=points, normals=normals, weights=weights,
        center=np.asarray(center, dtype=float), radius=float(radius),
        Wx=Wx, Wy=Wy, Wz=Wz,
    )


# ---------------------------------------------------------------------------
# State + solver.
# ---------------------------------------------------------------------------


@dataclass
class AcousticState:
    p: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray


def _apply_box_bc(state: AcousticState) -> None:
    """Enforce acoustically-soft box walls by pinning :math:`p = 0` on every face.

    Soft (Dirichlet) walls are adopted so the first-order discrete system is
    norm-compatible for every family: the boundary work integral
    :math:`\\oint p\\,u_n\\,dS` vanishes because :math:`p = 0`, and the semi-
    discrete energy estimate closes without requiring the SBP-SAT correction
    that hard-wall (:math:`u_n = 0`) enforcement would need. Acoustic waves
    still reflect off the box, and the reflection carries the standard
    soft-wall sign flip.
    """
    state.p[0, :, :] = 0.0
    state.p[-1, :, :] = 0.0
    state.p[:, 0, :] = 0.0
    state.p[:, -1, :] = 0.0
    state.p[:, :, 0] = 0.0
    state.p[:, :, -1] = 0.0


def apply_bmf_staggered_box_bc(state: BMFStaggeredAcousticState) -> None:
    """Enforce the soft-wall pressure condition on the BMF primal grid."""
    state.p[0, :, :] = 0.0
    state.p[-1, :, :] = 0.0
    state.p[:, 0, :] = 0.0
    state.p[:, -1, :] = 0.0
    state.p[:, :, 0] = 0.0
    state.p[:, :, -1] = 0.0


def bmf_staggered_exact_state(
    operator: BMFStaggeredOperator3D,
    exact,
    t: float,
) -> BMFStaggeredAcousticState:
    """Sample a box-eigenmode exact state on the BMF staggered grids."""
    x = operator.x
    eta = operator.eta
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    P = exact.pressure(X, Y, Z, t)

    Xu, Yu, Zu = np.meshgrid(eta, x, x, indexing="ij")
    U, _Vu, _Wu = exact.velocity(Xu, Yu, Zu, t)

    Xv, Yv, Zv = np.meshgrid(x, eta, x, indexing="ij")
    _Uv, V, _Wv = exact.velocity(Xv, Yv, Zv, t)

    Xw, Yw, Zw = np.meshgrid(x, x, eta, indexing="ij")
    _Uw, _Vw, W = exact.velocity(Xw, Yw, Zw, t)

    state = BMFStaggeredAcousticState(
        p=P.copy(),
        u=U.copy(),
        v=V.copy(),
        w=W.copy(),
    )
    apply_bmf_staggered_box_bc(state)
    return state


def _bmf_staggered_rhs(
    state: BMFStaggeredAcousticState,
    operator: BMFStaggeredOperator3D,
    rho0: float,
    c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    div_u = operator.div(state.u, state.v, state.w)
    dp_dx, dp_dy, dp_dz = operator.grad(state.p)
    rhs_p = -rho0 * c * c * div_u
    rhs_u = -(1.0 / rho0) * dp_dx
    rhs_v = -(1.0 / rho0) * dp_dy
    rhs_w = -(1.0 / rho0) * dp_dz
    return rhs_p, rhs_u, rhs_v, rhs_w


def bmf_staggered_rk4_step(
    state: BMFStaggeredAcousticState,
    dt: float,
    operator: BMFStaggeredOperator3D,
    rho0: float,
    c: float,
) -> None:
    """Advance the BMF staggered acoustic system by one RK4 step."""
    k1p, k1u, k1v, k1w = _bmf_staggered_rhs(state, operator, rho0, c)
    s2 = BMFStaggeredAcousticState(
        p=state.p + 0.5 * dt * k1p,
        u=state.u + 0.5 * dt * k1u,
        v=state.v + 0.5 * dt * k1v,
        w=state.w + 0.5 * dt * k1w,
    )
    apply_bmf_staggered_box_bc(s2)

    k2p, k2u, k2v, k2w = _bmf_staggered_rhs(s2, operator, rho0, c)
    s3 = BMFStaggeredAcousticState(
        p=state.p + 0.5 * dt * k2p,
        u=state.u + 0.5 * dt * k2u,
        v=state.v + 0.5 * dt * k2v,
        w=state.w + 0.5 * dt * k2w,
    )
    apply_bmf_staggered_box_bc(s3)

    k3p, k3u, k3v, k3w = _bmf_staggered_rhs(s3, operator, rho0, c)
    s4 = BMFStaggeredAcousticState(
        p=state.p + dt * k3p,
        u=state.u + dt * k3u,
        v=state.v + dt * k3v,
        w=state.w + dt * k3w,
    )
    apply_bmf_staggered_box_bc(s4)

    k4p, k4u, k4v, k4w = _bmf_staggered_rhs(s4, operator, rho0, c)
    state.p += (dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
    state.u += (dt / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
    state.v += (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    state.w += (dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)
    apply_bmf_staggered_box_bc(state)


@dataclass
class AcousticRun:
    label: str
    times: np.ndarray
    force_history: np.ndarray
    receiver_trace: np.ndarray
    energy_history: np.ndarray
    cum_boundary_work: np.ndarray
    energy_defect: np.ndarray
    surface_pressure_l2_history: np.ndarray
    surface_normal_velocity_l2_history: np.ndarray
    surface_work_rate_history: np.ndarray
    snapshots: dict[str, np.ndarray]
    snapshot_times: np.ndarray
    surface_pressure_snapshot: np.ndarray
    surface_normal_velocity_snapshot: np.ndarray
    final_state: AcousticState
    metadata: dict


def _acoustic_rhs(state: AcousticState, op: Operator3D, rho0: float, c: float,
                  sphere: Optional[SphereSurface], penalty_sigma: float,
                  penalty_mode: str,
                  forcing_p: Optional[Callable[[float], np.ndarray]],
                  t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    div_u = op.div(state.u, state.v, state.w)
    dp_dx, dp_dy, dp_dz = op.grad(state.p)
    rhs_p = -rho0 * c * c * div_u
    rhs_u = -(1.0 / rho0) * dp_dx
    rhs_v = -(1.0 / rho0) * dp_dy
    rhs_w = -(1.0 / rho0) * dp_dz

    if forcing_p is not None:
        rhs_p = rhs_p + forcing_p(t)

    if sphere is not None and penalty_sigma != 0.0:
        if penalty_mode == "soft_pressure":
            p_surf = sphere.interpolate(state.p)
            surf_force = -penalty_sigma * p_surf * sphere.weights
            rhs_p = rhs_p + sphere.spread(surf_force)
        elif penalty_mode == "rigid_normal_velocity":
            u_surf = sphere.interpolate(state.u)
            v_surf = sphere.interpolate(state.v)
            w_surf = sphere.interpolate(state.w)
            un_surf = (
                u_surf * sphere.normals[:, 0]
                + v_surf * sphere.normals[:, 1]
                + w_surf * sphere.normals[:, 2]
            )
            surf_force = -penalty_sigma * un_surf * sphere.weights
            rhs_u = rhs_u + sphere.spread(surf_force * sphere.normals[:, 0])
            rhs_v = rhs_v + sphere.spread(surf_force * sphere.normals[:, 1])
            rhs_w = rhs_w + sphere.spread(surf_force * sphere.normals[:, 2])
        else:
            raise ValueError(f"unknown penalty_mode {penalty_mode!r}")

    return rhs_p, rhs_u, rhs_v, rhs_w


def rk4_step(state: AcousticState, dt: float, op: Operator3D, rho0: float, c: float,
             sphere: Optional[SphereSurface], penalty_sigma: float,
             penalty_mode: str,
             forcing_p: Optional[Callable[[float], np.ndarray]], t: float) -> None:
    k1p, k1u, k1v, k1w = _acoustic_rhs(
        state, op, rho0, c, sphere, penalty_sigma, penalty_mode, forcing_p, t
    )
    s2 = AcousticState(
        p=state.p + 0.5 * dt * k1p,
        u=state.u + 0.5 * dt * k1u,
        v=state.v + 0.5 * dt * k1v,
        w=state.w + 0.5 * dt * k1w,
    )
    _apply_box_bc(s2)
    k2p, k2u, k2v, k2w = _acoustic_rhs(
        s2, op, rho0, c, sphere, penalty_sigma, penalty_mode, forcing_p, t + 0.5 * dt
    )
    s3 = AcousticState(
        p=state.p + 0.5 * dt * k2p,
        u=state.u + 0.5 * dt * k2u,
        v=state.v + 0.5 * dt * k2v,
        w=state.w + 0.5 * dt * k2w,
    )
    _apply_box_bc(s3)
    k3p, k3u, k3v, k3w = _acoustic_rhs(
        s3, op, rho0, c, sphere, penalty_sigma, penalty_mode, forcing_p, t + 0.5 * dt
    )
    s4 = AcousticState(
        p=state.p + dt * k3p,
        u=state.u + dt * k3u,
        v=state.v + dt * k3v,
        w=state.w + dt * k3w,
    )
    _apply_box_bc(s4)
    k4p, k4u, k4v, k4w = _acoustic_rhs(
        s4, op, rho0, c, sphere, penalty_sigma, penalty_mode, forcing_p, t + dt
    )

    state.p += (dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
    state.u += (dt / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
    state.v += (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    state.w += (dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)
    _apply_box_bc(state)


def integrated_force(sphere: SphereSurface, p_surf: np.ndarray) -> np.ndarray:
    """Return net force :math:`\\vec F = -\\oint p \\vec n dS` as (3,)."""
    return -np.einsum("m,mj,m->j", p_surf, sphere.normals, sphere.weights, optimize=True)


def _tensor_quadratic_form(field: np.ndarray, metric1d: np.ndarray) -> float:
    """Return <field, field> under the tensor-product metric induced by metric1d."""
    metric = np.asarray(metric1d, dtype=float)
    if metric.ndim == 1:
        weights = np.einsum("i,j,k->ijk", metric, metric, metric, optimize=True)
        return float(np.sum(weights * field * field))
    if metric.ndim != 2:
        raise ValueError("metric1d must be a vector or square matrix")
    tmp = np.einsum("ad,dbc->abc", metric, field, optimize=True)
    tmp = np.einsum("be,aec->abc", metric, tmp, optimize=True)
    tmp = np.einsum("cf,abf->abc", metric, tmp, optimize=True)
    return float(np.einsum("abc,abc->", field, tmp, optimize=True))


def _tensor_mixed_quadratic_form(
    field: np.ndarray,
    metric_x: np.ndarray,
    metric_y: np.ndarray,
    metric_z: np.ndarray,
) -> float:
    """Return a tensor-product quadratic form with axis-specific metrics."""
    mx = np.asarray(metric_x, dtype=float)
    my = np.asarray(metric_y, dtype=float)
    mz = np.asarray(metric_z, dtype=float)
    tmp = np.einsum("ad,dbc->abc", mx, field, optimize=True)
    tmp = np.einsum("be,aec->abc", my, tmp, optimize=True)
    tmp = np.einsum("cf,abf->abc", mz, tmp, optimize=True)
    return float(np.einsum("abc,abc->", field, tmp, optimize=True))


def energy(state: AcousticState, metric1d: np.ndarray, rho0: float, c: float) -> float:
    """Return total acoustic energy in the family's natural 3D metric."""
    e_p = 0.5 * _tensor_quadratic_form(state.p, metric1d) / (rho0 * c * c)
    e_k = 0.5 * rho0 * (
        _tensor_quadratic_form(state.u, metric1d)
        + _tensor_quadratic_form(state.v, metric1d)
        + _tensor_quadratic_form(state.w, metric1d)
    )
    return float(e_p + e_k)


def bmf_staggered_energy(
    state: BMFStaggeredAcousticState,
    operator: BMFStaggeredOperator3D,
    rho0: float,
    c: float,
) -> float:
    """Return acoustic energy in the tensor-product BMF staggered metric."""
    h0 = operator.ops.h0
    h1 = operator.ops.h1
    e_p = 0.5 * _tensor_mixed_quadratic_form(state.p, h0, h0, h0) / (rho0 * c * c)
    e_k = 0.5 * rho0 * (
        _tensor_mixed_quadratic_form(state.u, h1, h0, h0)
        + _tensor_mixed_quadratic_form(state.v, h0, h1, h0)
        + _tensor_mixed_quadratic_form(state.w, h0, h0, h1)
    )
    return float(e_p + e_k)


# ---------------------------------------------------------------------------
# Convenience drivers.
# ---------------------------------------------------------------------------


def run_acoustic_pulse(
    label: str, *,
    npts: int,
    t_end: float,
    cfl: float,
    initial_pressure: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    initial_velocity: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
    sphere_center: Optional[tuple[float, float, float]] = None,
    sphere_radius: float = 0.0,
    n_surface_nodes: int = 0,
    penalty_sigma: float = 0.0,
    penalty_mode: str = "soft_pressure",
    receiver: Optional[tuple[int, int, int]] = None,
    snapshot_times: Optional[np.ndarray] = None,
    record_cadence: int = 4,
    rho0: float = 1.0,
    c: float = 1.0,
    forcing_p: Optional[Callable[[float], np.ndarray]] = None,
    exact_state: Optional[Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None,
) -> AcousticRun:
    op = Operator3D(label, npts)
    x = op.x
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    P0 = initial_pressure(X, Y, Z)
    if initial_velocity is not None:
        U0, V0, W0 = initial_velocity(X, Y, Z)
    else:
        U0 = np.zeros_like(P0)
        V0 = np.zeros_like(P0)
        W0 = np.zeros_like(P0)
    state = AcousticState(p=P0.copy(), u=U0.copy(), v=V0.copy(), w=W0.copy())
    _apply_box_bc(state)

    dt = cfl * op.h / c
    nsteps = int(np.ceil(t_end / dt))
    dt = t_end / nsteps

    sphere = None
    if sphere_center is not None and sphere_radius > 0.0 and n_surface_nodes > 0:
        sphere = build_sphere_surface(op, sphere_center, sphere_radius, n_surface_nodes)

    metric1d = op.metric_1d

    times = []
    force_history = []
    receiver_trace = []
    energy_history = []
    surface_p_l2_history = []
    surface_un_l2_history = []
    surface_work_rate_history = []
    cum_work = 0.0
    cum_work_history = []
    energy_defect_history = []

    snapshots: dict[str, np.ndarray] = {}
    snapshot_times_out: list[float] = []
    surface_p_snapshots: list[np.ndarray] = []
    surface_un_snapshots: list[np.ndarray] = []

    snap_target_times = np.asarray(snapshot_times, dtype=float) if snapshot_times is not None else np.array([])
    snap_taken_mask = np.zeros(snap_target_times.size, dtype=bool)

    E0 = energy(state, metric1d, rho0, c)

    def record(step: int, t: float, prev_t: float, prev_work_rate: float) -> tuple[float, float]:
        nonlocal cum_work
        times.append(t)
        if sphere is not None:
            p_surf = sphere.interpolate(state.p)
            F = integrated_force(sphere, p_surf)
            force_history.append(F)
            # normal velocity u . n on surface
            u_surf = sphere.interpolate(state.u)
            v_surf = sphere.interpolate(state.v)
            w_surf = sphere.interpolate(state.w)
            un_surf = (u_surf * sphere.normals[:, 0]
                       + v_surf * sphere.normals[:, 1]
                       + w_surf * sphere.normals[:, 2])
            # instantaneous boundary-work rate: oint p u_n dS on the sphere (the walls contribute 0).
            work_rate = float(np.sum(p_surf * un_surf * sphere.weights))
            surface_p_l2_history.append(float(np.sqrt(np.sum(p_surf * p_surf * sphere.weights))))
            surface_un_l2_history.append(float(np.sqrt(np.sum(un_surf * un_surf * sphere.weights))))
            surface_work_rate_history.append(work_rate)
        else:
            force_history.append(np.zeros(3))
            work_rate = 0.0
            surface_p_l2_history.append(0.0)
            surface_un_l2_history.append(0.0)
            surface_work_rate_history.append(0.0)
        if receiver is not None:
            ri, rj, rk = receiver
            receiver_trace.append(float(state.p[ri, rj, rk]))
        else:
            receiver_trace.append(0.0)
        E = energy(state, metric1d, rho0, c)
        energy_history.append(E)
        dt_bucket = t - prev_t
        if dt_bucket > 0.0:
            cum_work += 0.5 * (work_rate + prev_work_rate) * dt_bucket
        cum_work_history.append(cum_work)
        energy_defect_history.append(E - E0 + cum_work)
        # Snapshot if requested.
        for si, st in enumerate(snap_target_times):
            if not snap_taken_mask[si] and t + 1.0e-14 >= st:
                snapshots[f"p_t{si:02d}"] = state.p.copy()
                snapshots[f"u_t{si:02d}"] = state.u.copy()
                snapshots[f"v_t{si:02d}"] = state.v.copy()
                snapshots[f"w_t{si:02d}"] = state.w.copy()
                snapshot_times_out.append(t)
                if sphere is not None:
                    surface_p_snapshots.append(sphere.interpolate(state.p).copy())
                    u_surf = sphere.interpolate(state.u)
                    v_surf = sphere.interpolate(state.v)
                    w_surf = sphere.interpolate(state.w)
                    surface_un_snapshots.append(
                        (u_surf * sphere.normals[:, 0]
                         + v_surf * sphere.normals[:, 1]
                         + w_surf * sphere.normals[:, 2]).copy()
                    )
                snap_taken_mask[si] = True
        return t, work_rate

    prev_t, prev_rate = record(0, 0.0, 0.0, 0.0)

    for step in range(nsteps):
        rk4_step(state, dt, op, rho0, c, sphere, penalty_sigma, penalty_mode, forcing_p, step * dt)
        if ((step + 1) % record_cadence == 0) or (step + 1 == nsteps):
            prev_t, prev_rate = record(step + 1, (step + 1) * dt, prev_t, prev_rate)

    return AcousticRun(
        label=label,
        times=np.asarray(times, dtype=float),
        force_history=np.asarray(force_history, dtype=float),
        receiver_trace=np.asarray(receiver_trace, dtype=float),
        energy_history=np.asarray(energy_history, dtype=float),
        cum_boundary_work=np.asarray(cum_work_history, dtype=float),
        energy_defect=np.asarray(energy_defect_history, dtype=float),
        surface_pressure_l2_history=np.asarray(surface_p_l2_history, dtype=float),
        surface_normal_velocity_l2_history=np.asarray(surface_un_l2_history, dtype=float),
        surface_work_rate_history=np.asarray(surface_work_rate_history, dtype=float),
        snapshots=snapshots,
        snapshot_times=np.asarray(snapshot_times_out, dtype=float),
        surface_pressure_snapshot=(np.stack(surface_p_snapshots, axis=0) if surface_p_snapshots
                                   else np.zeros((0, max(n_surface_nodes, 1)), dtype=float)),
        surface_normal_velocity_snapshot=(np.stack(surface_un_snapshots, axis=0) if surface_un_snapshots
                                          else np.zeros((0, max(n_surface_nodes, 1)), dtype=float)),
        final_state=state,
        metadata={
            "h": op.h,
            "h_min": op.h,
            "h_nominal": op.h_nominal,
            "dt": dt,
            "nsteps": nsteps,
            "cfl": cfl,
            "t_end": t_end,
            "rho0": rho0, "c": c, "n_surface_nodes": n_surface_nodes,
            "sphere_center": None if sphere_center is None else list(sphere_center),
            "sphere_radius": float(sphere_radius), "penalty_sigma": penalty_sigma,
            "penalty_mode": penalty_mode,
            "family": op.family,
        },
    )


# ---------------------------------------------------------------------------
# Manufactured-solution helpers.
# ---------------------------------------------------------------------------


@dataclass
class BoxEigenmode:
    """Superposition of soft-wall sine eigenmodes of the closed cube.

    Each mode :math:`(n_x, n_y, n_z)` has pressure

    .. math::
       p_n = A_n \\sin(n_x \\pi x / L)\\sin(n_y \\pi y / L)\\sin(n_z \\pi z / L)\\cos(\\omega_n t)

    and an accompanying velocity field obtained from the acoustic momentum
    equation. The pressure satisfies :math:`p = 0` on every face of the
    cube, the velocity has nonzero normal components on the faces, and the
    superposition is an exact solution of the linearized acoustic equations
    with soft-wall Dirichlet boundary conditions.
    """

    modes: list[tuple[int, int, int, float]]
    c: float = 1.0
    rho0: float = 1.0
    length: float = 1.0

    def _omega(self, nx: int, ny: int, nz: int) -> float:
        return self.c * np.pi * float(np.sqrt(nx * nx + ny * ny + nz * nz)) / self.length

    def pressure(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, t: float) -> np.ndarray:
        out = np.zeros_like(X)
        for nx, ny, nz, A in self.modes:
            k = np.pi / self.length
            out = out + A * np.sin(nx * k * X) * np.sin(ny * k * Y) * np.sin(nz * k * Z) * np.cos(self._omega(nx, ny, nz) * t)
        return out

    def velocity(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, t: float
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        U = np.zeros_like(X)
        V = np.zeros_like(X)
        W = np.zeros_like(X)
        for nx, ny, nz, A in self.modes:
            k = np.pi / self.length
            om = self._omega(nx, ny, nz)
            # u = -(1/rho0) int p_x dt = -(A n_x k / (rho0 omega)) cos(nx k x) sin(ny k y) sin(nz k z) sin(omega t)
            U = U - (A * nx * k / (self.rho0 * om)) * (
                np.cos(nx * k * X) * np.sin(ny * k * Y) * np.sin(nz * k * Z) * np.sin(om * t))
            V = V - (A * ny * k / (self.rho0 * om)) * (
                np.sin(nx * k * X) * np.cos(ny * k * Y) * np.sin(nz * k * Z) * np.sin(om * t))
            W = W - (A * nz * k / (self.rho0 * om)) * (
                np.sin(nx * k * X) * np.sin(ny * k * Y) * np.cos(nz * k * Z) * np.sin(om * t))
        return U, V, W

    def pressure_on_sphere(self, points: np.ndarray, t: float) -> np.ndarray:
        out = np.zeros(points.shape[0], dtype=float)
        for nx, ny, nz, A in self.modes:
            k = np.pi / self.length
            out = out + A * np.sin(nx * k * points[:, 0]) * np.sin(ny * k * points[:, 1]) * np.sin(nz * k * points[:, 2]) * np.cos(self._omega(nx, ny, nz) * t)
        return out

    def grad_pressure_on_sphere(self, points: np.ndarray, t: float) -> np.ndarray:
        out = np.zeros_like(points)
        for nx, ny, nz, A in self.modes:
            k = np.pi / self.length
            om = self._omega(nx, ny, nz)
            cx = np.cos(nx * k * points[:, 0]); sx = np.sin(nx * k * points[:, 0])
            cy = np.cos(ny * k * points[:, 1]); sy = np.sin(ny * k * points[:, 1])
            cz = np.cos(nz * k * points[:, 2]); sz = np.sin(nz * k * points[:, 2])
            ct = np.cos(om * t)
            out[:, 0] += A * nx * k * cx * sy * sz * ct
            out[:, 1] += A * ny * k * sx * cy * sz * ct
            out[:, 2] += A * nz * k * sx * sy * cz * ct
        return out

    def integrated_force_on_sphere(self, sphere: SphereSurface, t: float) -> np.ndarray:
        """Closed-form net force on the sphere due to the box eigenmode."""
        p_surf = self.pressure_on_sphere(sphere.points, t)
        return -np.einsum("m,mj,m->j", p_surf, sphere.normals, sphere.weights, optimize=True)
