#!/usr/bin/env python3
"""Small exact-quadrature integration lane for route-C BMF flow operators.

This module deliberately stops short of a production Navier--Stokes solver.  It
assembles the normal-essential, tangential-Nitsche viscous form; audits pressure
work in the natural tensor metrics; evaluates a conservative/skew convective
form; and solves a small manufactured Stokes saddle problem.
"""
from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve

from bmf_hdiv_projection import BMFHDivProjection2D
from boundary_spline_analysis import (
    bspline_evaluation_matrix,
    derivative_greville_nodes,
    derivative_space_knots,
    greville_from_knots,
    open_uniform_knots,
)


Array = np.ndarray
ScalarField = Callable[[Array, Array], Array]


def _require_finite(name: str, value: Array | float) -> None:
    if not np.all(np.isfinite(np.asarray(value))):
        raise ValueError(f"non-finite values in {name}")


@dataclass(frozen=True)
class _AxisTables:
    q: Array
    w: Array
    x0: Array
    x1: Array
    v0: Array
    v1: Array
    d0: Array
    d1: Array
    e1: Array
    n1: Array
    k0: Array
    k1: Array


def _quadrature(knots: Array, order: int) -> tuple[Array, Array]:
    nodes, weights = leggauss(int(order))
    points: list[Array] = []
    scaled: list[Array] = []
    unique = np.unique(knots)
    for left, right in zip(unique[:-1], unique[1:]):
        if right <= left:
            continue
        points.append(0.5 * (right - left) * nodes + 0.5 * (right + left))
        scaled.append(0.5 * (right - left) * weights)
    return np.concatenate(points), np.concatenate(scaled)


def _right_solve_spd(matrix: Array, rhs: Array) -> Array:
    return solve(matrix, rhs.T, assume_a="pos", check_finite=False).T


class BMFHDivNS2D:
    """Finite route-C integration gates on a square domain."""

    def __init__(
        self,
        npts: int,
        degree: int,
        length: float = 1.0,
        penalty_gamma: float | None = None,
    ) -> None:
        self.core = BMFHDivProjection2D(npts=npts, degree=degree, length=length)
        self.npts = self.core.npts
        self.degree = self.core.degree
        self.length = self.core.length
        gamma = 4.0 * self.degree**2 if penalty_gamma is None else float(penalty_gamma)
        if not np.isfinite(gamma) or gamma < 4.0 * self.degree**2:
            raise ValueError("penalty_gamma must be finite and at least 4 p^2")
        self.penalty_gamma = gamma
        self.mesh_size = self.length / (self.npts - self.degree)
        self.penalty = self.penalty_gamma / self.mesh_size
        self.quadrature_points_per_span = max(2 * self.degree + 6, 12)
        self.maximum_galerkin_integrand_degree = 3 * self.degree - 1
        self.gauss_exact_polynomial_degree = 2 * self.quadrature_points_per_span - 1
        self.axis = self._build_axis_tables()

        self.ux_active = np.ones(self.core.ux_shape, dtype=bool)
        self.ux_active[[0, -1], :] = False
        self.uz_active = np.ones(self.core.uz_shape, dtype=bool)
        self.uz_active[:, [0, -1]] = False
        self.velocity_size_full = int(
            np.prod(self.core.ux_shape) + np.prod(self.core.uz_shape)
        )
        offset = int(np.prod(self.core.ux_shape))
        self.active_indices = np.concatenate(
            [
                np.flatnonzero(self.ux_active.ravel()),
                offset + np.flatnonzero(self.uz_active.ravel()),
            ]
        )
        self.divergence_matrix_full = self._assemble_divergence_matrix()
        self.divergence_matrix = self.divergence_matrix_full[:, self.active_indices]
        ux_metric = np.kron(self.core.h0, self.core.h1)
        uz_metric = np.kron(self.core.h1, self.core.h0)
        self.velocity_mass_full = np.zeros(
            (self.velocity_size_full, self.velocity_size_full)
        )
        split = int(np.prod(self.core.ux_shape))
        self.velocity_mass_full[:split, :split] = ux_metric
        self.velocity_mass_full[split:, split:] = uz_metric
        self.velocity_mass = self.velocity_mass_full[
            np.ix_(self.active_indices, self.active_indices)
        ]
        self.viscous_matrix_full = self._assemble_viscous_matrix_full()
        self.viscous_matrix = self.viscous_matrix_full[
            np.ix_(self.active_indices, self.active_indices)
        ]
        _require_finite("viscous matrix", self.viscous_matrix)
        _require_finite("divergence matrix", self.divergence_matrix)

    def _build_axis_tables(self) -> _AxisTables:
        p = self.degree
        knots0 = open_uniform_knots(self.npts, p, self.length)
        knots1 = derivative_space_knots(knots0)
        x0 = greville_from_knots(knots0, p, self.npts)
        x1 = derivative_greville_nodes(knots1, p - 1)
        b0 = bspline_evaluation_matrix(knots0, p, x0)
        b1 = bspline_evaluation_matrix(knots1, p - 1, x1)
        b0_inverse = np.linalg.inv(b0)
        b1_inverse = np.linalg.inv(b1)
        # This count is exact for all degree-(3p-1) Galerkin convective
        # integrands and is intentionally more accurate for the
        # non-polynomial Stokes datum.
        q, w = _quadrature(knots0, self.quadrature_points_per_span)
        v0 = bspline_evaluation_matrix(knots0, p, q) @ b0_inverse
        v1 = bspline_evaluation_matrix(knots1, p - 1, q) @ b1_inverse
        d0 = bspline_evaluation_matrix(knots0, p, q, derivative=1) @ b0_inverse
        d1 = bspline_evaluation_matrix(knots1, p - 1, q, derivative=1) @ b1_inverse
        endpoints = np.array([0.0, self.length])
        e1 = bspline_evaluation_matrix(knots1, p - 1, endpoints) @ b1_inverse
        derivative = (
            bspline_evaluation_matrix(knots1, p - 1, endpoints, derivative=1)
            @ b1_inverse
        )
        n1 = np.vstack([-derivative[0], derivative[1]])
        k0 = d0.T @ (w[:, None] * d0)
        k1 = d1.T @ (w[:, None] * d1)
        for name, value in (
            ("quadrature points", q),
            ("quadrature weights", w),
            ("primal evaluation", v0),
            ("derivative evaluation", v1),
            ("endpoint traces", e1),
            ("endpoint normal derivatives", n1),
        ):
            _require_finite(name, value)
        return _AxisTables(q, w, x0, x1, v0, v1, d0, d1, e1, n1, k0, k1)

    def _assemble_divergence_matrix(self) -> Array:
        derivative_size = self.core.derivative_size
        identity = np.eye(derivative_size)
        return np.column_stack(
            [
                np.kron(self.core.d10, identity),
                np.kron(identity, self.core.d10),
            ]
        )

    def _assemble_viscous_matrix_full(self) -> Array:
        axis = self.axis
        boundary = -axis.n1.T @ axis.e1 - axis.e1.T @ axis.n1
        boundary += self.penalty * axis.e1.T @ axis.e1
        ux = np.kron(axis.k0, self.core.h1) + np.kron(
            self.core.h0, axis.k1 + boundary
        )
        uz = np.kron(axis.k1 + boundary, self.core.h0) + np.kron(
            self.core.h1, axis.k0
        )
        matrix = np.zeros((self.velocity_size_full, self.velocity_size_full))
        split = int(np.prod(self.core.ux_shape))
        matrix[:split, :split] = ux
        matrix[split:, split:] = uz
        return 0.5 * (matrix + matrix.T)

    def _pack_active(self, ux: Array, uz: Array) -> Array:
        x, z = self.core._velocity_arrays(ux, uz)
        return np.concatenate([x.ravel(), z.ravel()])[self.active_indices]

    def _unpack_active(self, values: Array) -> tuple[Array, Array]:
        vector = np.asarray(values, dtype=float)
        if vector.shape != (self.active_indices.size,):
            raise ValueError("active velocity vector has the wrong shape")
        _require_finite("active velocity vector", vector)
        full = np.zeros(self.velocity_size_full)
        full[self.active_indices] = vector
        split = int(np.prod(self.core.ux_shape))
        return (
            full[:split].reshape(self.core.ux_shape),
            full[split:].reshape(self.core.uz_shape),
        )

    def divergence_rank_certificate(self) -> dict[str, int | float]:
        weak_rank = int(np.linalg.matrix_rank(self.divergence_matrix))
        strong_ux = np.zeros(self.core.ux_shape, dtype=bool)
        strong_ux[1:-1, 1:-1] = True
        strong_uz = np.zeros(self.core.uz_shape, dtype=bool)
        strong_uz[1:-1, 1:-1] = True
        split = int(np.prod(self.core.ux_shape))
        strong_indices = np.concatenate(
            [
                np.flatnonzero(strong_ux.ravel()),
                split + np.flatnonzero(strong_uz.ravel()),
            ]
        )
        strong_rank = int(
            np.linalg.matrix_rank(self.divergence_matrix_full[:, strong_indices])
        )
        reference = np.column_stack(
            [
                np.kron(self.core.g0, np.eye(self.core.derivative_size)),
                np.kron(np.eye(self.core.derivative_size), self.core.g0),
            ]
        )
        change = float(np.max(np.abs(self.divergence_matrix - reference)))
        return {
            "weak_tangential_rank": weak_rank,
            "weak_tangential_nullity": self.core.pressure_size - weak_rank,
            "strong_tangential_rank": strong_rank,
            "strong_tangential_nullity": self.core.pressure_size - strong_rank,
            "weak_matrix_change_from_core": change,
        }

    def _assemble_viscous_rhs(
        self,
        force_ux: ScalarField,
        force_uz: ScalarField,
        trace_ux: ScalarField,
        trace_uz: ScalarField,
    ) -> Array:
        axis = self.axis
        q = axis.q
        w = axis.w
        weight2 = np.outer(w, w)
        xq, zq = np.meshgrid(q, q, indexing="ij")
        fux = np.asarray(force_ux(xq, zq), dtype=float)
        fuz = np.asarray(force_uz(xq, zq), dtype=float)
        _require_finite("ux body force", fux)
        _require_finite("uz body force", fuz)
        ux_moments = axis.v0.T @ (weight2 * fux) @ axis.v1
        uz_moments = axis.v1.T @ (weight2 * fuz) @ axis.v0

        for side, coordinate in enumerate((0.0, self.length)):
            gx = np.asarray(trace_ux(q, np.full_like(q, coordinate)), dtype=float)
            gz = np.asarray(trace_uz(np.full_like(q, coordinate), q), dtype=float)
            ux_xmom = axis.v0.T @ (w * gx)
            uz_zmom = axis.v0.T @ (w * gz)
            boundary_row = -axis.n1[side] + self.penalty * axis.e1[side]
            ux_moments += np.outer(ux_xmom, boundary_row)
            uz_moments += np.outer(boundary_row, uz_zmom)

        full = np.concatenate([ux_moments.ravel(), uz_moments.ravel()])
        _require_finite("viscous right-hand side", full)
        return full[self.active_indices]

    def polynomial_nitsche_consistency(self) -> dict[str, float]:
        ux_exact = lambda x, z: x * (self.length - x) * (1.0 + z)
        uz_exact = lambda x, z: z * (self.length - z) * (1.0 + x)
        force_ux = lambda x, z: 2.0 * (1.0 + z)
        force_uz = lambda x, z: 2.0 * (1.0 + x)
        x0, x1 = self.axis.x0, self.axis.x1
        ux = ux_exact(x0[:, None], x1[None, :])
        uz = uz_exact(x1[:, None], x0[None, :])
        load = self._assemble_viscous_rhs(force_ux, force_uz, ux_exact, uz_exact)
        residual = self.viscous_matrix @ self._pack_active(ux, uz) - load
        trace_norm = self._tangential_trace_norm_squared(ux_exact, uz_exact)
        return {
            "residual_l2_squared": float(residual @ residual),
            "load_l2_squared": float(load @ load),
            "tangential_trace_l2_squared": trace_norm,
        }

    def _tangential_trace_norm_squared(
        self,
        trace_ux: ScalarField,
        trace_uz: ScalarField,
    ) -> float:
        q, w = self.axis.q, self.axis.w
        total = 0.0
        for coordinate in (0.0, self.length):
            gx = np.asarray(trace_ux(q, np.full_like(q, coordinate)), dtype=float)
            gz = np.asarray(trace_uz(np.full_like(q, coordinate), q), dtype=float)
            total += float(w @ (gx * gx) + w @ (gz * gz))
        _require_finite("tangential trace norm", total)
        return total

    def tangential_trace_l2_squared(self, ux: Array, uz: Array) -> float:
        """Return the exact quadrature norm of the discrete tangential wall trace."""
        x, z = self.core._velocity_arrays(ux, uz)
        axis = self.axis
        total = 0.0
        for side in (0, 1):
            trace_x = axis.v0 @ x @ axis.e1[side]
            trace_z = axis.e1[side] @ z @ axis.v0.T
            total += float(axis.w @ (trace_x * trace_x + trace_z * trace_z))
        _require_finite("discrete tangential trace norm", total)
        return total

    def pressure_work_ledger(
        self,
        ux: Array,
        uz: Array,
        pressure: Array,
    ) -> dict[str, float]:
        x, z = self.core._velocity_arrays(ux, uz)
        p = self.core._pressure_array(pressure)
        gx, gz = self.core.adjoint_gradient(p)
        lhs = self.core.velocity_inner((x, z), (gx, gz))
        rhs = self.core.pressure_inner(self.core.divergence(x, z), p)
        scale = self.core.velocity_norm((x, z)) * self.core.velocity_norm((gx, gz))
        scale = max(scale, 1.0e-300)
        return {
            "velocity_gradient_work": lhs,
            "divergence_pressure_work": rhs,
            "adjoint_residual": abs(lhs - rhs),
            "raw_scale": scale,
            "relative_adjoint_residual": abs(lhs - rhs) / scale,
            "relative_pressure_work": abs(lhs) / scale,
        }

    def skew_convection(
        self,
        ux: Array,
        uz: Array,
    ) -> tuple[Array, Array, dict[str, float]]:
        x, z = self.core._velocity_arrays(ux, uz)
        axis = self.axis
        weight = np.outer(axis.w, axis.w)
        uxq = axis.v0 @ x @ axis.v1.T
        uzq = axis.v1 @ z @ axis.v0.T
        ux_x = axis.d0 @ x @ axis.v1.T
        ux_z = axis.v0 @ x @ axis.d1.T
        uz_x = axis.d1 @ z @ axis.v0.T
        uz_z = axis.v1 @ z @ axis.d0.T
        divergence = ux_x + uz_z

        adv_ux = uxq * ux_x + uzq * ux_z
        adv_uz = uxq * uz_x + uzq * uz_z
        cons_ux = 2.0 * uxq * ux_x + uxq * uz_z + uzq * ux_z
        cons_uz = uxq * uz_x + uzq * ux_x + 2.0 * uzq * uz_z
        cux_q = 0.5 * (adv_ux + cons_ux)
        cuz_q = 0.5 * (adv_uz + cons_uz)
        identity_ux = cons_ux - adv_ux - uxq * divergence
        identity_uz = cons_uz - adv_uz - uzq * divergence

        ux_moments = 0.5 * (
            axis.v0.T @ (weight * adv_ux) @ axis.v1
            - axis.d0.T @ (weight * (uxq * uxq)) @ axis.v1
            - axis.v0.T @ (weight * (uzq * uxq)) @ axis.d1
        )
        uz_moments = 0.5 * (
            axis.v1.T @ (weight * adv_uz) @ axis.v0
            - axis.d1.T @ (weight * (uxq * uzq)) @ axis.v0
            - axis.v1.T @ (weight * (uzq * uzq)) @ axis.d0
        )
        cux = solve(
            self.core.h0,
            ux_moments,
            assume_a="pos",
            check_finite=False,
        )
        cux = _right_solve_spd(self.core.h1, cux)
        cuz = solve(
            self.core.h1,
            uz_moments,
            assume_a="pos",
            check_finite=False,
        )
        cuz = _right_solve_spd(self.core.h0, cuz)
        _require_finite("convective ux residual", cux)
        _require_finite("convective uz residual", cuz)

        div_squared = float(np.sum(weight * divergence * divergence))
        div_scale = float(np.sum(weight * (ux_x * ux_x + uz_z * uz_z)))
        kinetic = float(np.sum(x * ux_moments) + np.sum(z * uz_moments))
        kinetic_scale = math.sqrt(
            max(float(np.sum(weight * (uxq * uxq + uzq * uzq))), 0.0)
            * max(float(np.sum(weight * (cux_q * cux_q + cuz_q * cuz_q))), 0.0)
        )
        momentum_x = float(np.sum(ux_moments))
        momentum_z = float(np.sum(uz_moments))
        momentum_defect = math.hypot(momentum_x, momentum_z)
        momentum_scale = float(np.sum(weight * (np.abs(cux_q) + np.abs(cuz_q))))
        identity_squared = float(
            np.sum(weight * (identity_ux * identity_ux + identity_uz * identity_uz))
        )
        identity_scale = float(
            np.sum(
                weight
                * (
                    adv_ux * adv_ux
                    + adv_uz * adv_uz
                    + cons_ux * cons_ux
                    + cons_uz * cons_uz
                    + (uxq * divergence) ** 2
                    + (uzq * divergence) ** 2
                )
            )
        )
        return cux, cuz, {
            "pointwise_divergence_l2_squared": div_squared,
            "pointwise_divergence_scale_squared": div_scale,
            "relative_pointwise_divergence": math.sqrt(div_squared / max(div_scale, 1.0e-300)),
            "kinetic_energy_production": kinetic,
            "kinetic_energy_scale": kinetic_scale,
            "relative_kinetic_energy_production": abs(kinetic) / max(kinetic_scale, 1.0e-300),
            "normal_trace_max": self.core.normal_trace_max(x, z),
            "conservative_identity_residual_l2_squared": identity_squared,
            "conservative_identity_scale_l2_squared": identity_scale,
            "relative_conservative_identity_residual": math.sqrt(
                identity_squared / max(identity_scale, 1.0e-300)
            ),
            "quadrature_points_per_span": self.quadrature_points_per_span,
            "maximum_galerkin_integrand_degree": self.maximum_galerkin_integrand_degree,
            "gauss_exact_polynomial_degree": self.gauss_exact_polynomial_degree,
            "convective_momentum_flux_x_defect": momentum_x,
            "convective_momentum_flux_z_defect": momentum_z,
            "convective_momentum_flux_defect": momentum_defect,
            "convective_momentum_flux_scale": momentum_scale,
            "relative_convective_momentum_flux_defect": momentum_defect
            / max(momentum_scale, 1.0e-300),
        }

    def momentum_reaction_ledger(
        self,
        ux: Array,
        uz: Array,
        pressure: Array,
        viscosity: float = 0.7,
    ) -> dict[str, float]:
        """Audit total momentum with the reaction of strong normal traces.

        The reduced equation is solved only on active velocity coefficients.
        Lifting its acceleration to the full component space leaves a residual
        supported on the eliminated normal-trace coefficients.  That residual
        is the algebraic wall reaction.  Testing the full-space identity with
        the exact constant spline gives the total-momentum ledger without
        pretending that constants belong to the reduced test space.
        """
        x, z = self.core._velocity_arrays(ux, uz)
        p = np.asarray(pressure, dtype=float)
        if p.shape != self.core.pressure_shape:
            raise ValueError("pressure has the wrong shape")
        _require_finite("momentum-ledger pressure", p)
        nu = float(viscosity)
        if not np.isfinite(nu) or nu < 0.0:
            raise ValueError("viscosity must be finite and nonnegative")
        if self.core.normal_trace_max(x, z) > 2.0e-12:
            raise ValueError("momentum reaction ledger requires strong no penetration")

        cux, cuz, convection = self.skew_convection(x, z)
        split = int(np.prod(self.core.ux_shape))
        velocity = np.concatenate([x.ravel(), z.ravel()])
        convective = np.concatenate([cux.ravel(), cuz.ravel()])
        pressure_metric = np.kron(self.core.h1, self.core.h1)
        dual_pressure = pressure_metric @ p.ravel()
        convective_dual = self.velocity_mass_full @ convective
        pressure_dual = self.divergence_matrix_full.T @ dual_pressure
        viscous_dual = nu * (self.viscous_matrix_full @ velocity)
        spatial_dual = convective_dual + pressure_dual + viscous_dual

        acceleration_active = solve(
            self.velocity_mass,
            -spatial_dual[self.active_indices],
            assume_a="pos",
            check_finite=False,
        )
        acceleration = np.zeros(self.velocity_size_full)
        acceleration[self.active_indices] = acceleration_active
        inertial_dual = self.velocity_mass_full @ acceleration
        residual = inertial_dual + spatial_dual

        active_residual = residual[self.active_indices]
        constrained = np.ones(self.velocity_size_full, dtype=bool)
        constrained[self.active_indices] = False
        constrained_support_defect = float(np.linalg.norm(residual[~constrained]))
        residual_scale = float(np.linalg.norm(inertial_dual) + np.linalg.norm(spatial_dual))
        reaction_scale = float(np.linalg.norm(residual[constrained]))

        ones_x = np.ones(split)
        ones_z = np.ones(self.velocity_size_full - split)

        def moment(vector: Array) -> tuple[float, float]:
            return float(ones_x @ vector[:split]), float(ones_z @ vector[split:])

        rate_x, rate_z = moment(inertial_dual)
        conv_x, conv_z = moment(convective_dual)
        pressure_x, pressure_z = moment(pressure_dual)
        viscous_x, viscous_z = moment(viscous_dual)
        reaction_x, reaction_z = moment(residual)
        defect_x = rate_x + conv_x + pressure_x + viscous_x - reaction_x
        defect_z = rate_z + conv_z + pressure_z + viscous_z - reaction_z
        balance_scale = (
            abs(rate_x) + abs(rate_z) + abs(conv_x) + abs(conv_z)
            + abs(pressure_x) + abs(pressure_z) + abs(viscous_x) + abs(viscous_z)
            + abs(reaction_x) + abs(reaction_z)
        )
        projected_convective_defect = math.hypot(
            conv_x - convection["convective_momentum_flux_x_defect"],
            conv_z - convection["convective_momentum_flux_z_defect"],
        )
        projected_convective_scale = max(
            convection["convective_momentum_flux_scale"], 1.0e-300
        )
        _require_finite("momentum-ledger residual", residual)
        return {
            "momentum_rate_x": rate_x,
            "momentum_rate_z": rate_z,
            "convective_moment_x": conv_x,
            "convective_moment_z": conv_z,
            "pressure_moment_x": pressure_x,
            "pressure_moment_z": pressure_z,
            "viscous_moment_x": viscous_x,
            "viscous_moment_z": viscous_z,
            "constraint_reaction_x": reaction_x,
            "constraint_reaction_z": reaction_z,
            "constraint_reaction_scale": reaction_scale,
            "momentum_balance_x_defect": defect_x,
            "momentum_balance_z_defect": defect_z,
            "momentum_balance_defect": math.hypot(defect_x, defect_z),
            "momentum_balance_scale": balance_scale,
            "relative_momentum_balance_defect": math.hypot(defect_x, defect_z)
            / max(balance_scale, 1.0e-300),
            "active_equation_residual": float(np.linalg.norm(active_residual)),
            "active_equation_scale": residual_scale,
            "relative_active_equation_residual": float(np.linalg.norm(active_residual))
            / max(residual_scale, 1.0e-300),
            "constraint_support_defect": constrained_support_defect,
            "relative_constraint_support_defect": constrained_support_defect
            / max(reaction_scale, 1.0e-300),
            "convective_projection_moment_defect": projected_convective_defect,
            "relative_convective_projection_moment_defect": projected_convective_defect
            / projected_convective_scale,
        }

    def solve_manufactured_stokes(
        self,
        viscosity: float = 0.7,
    ) -> dict[str, float | int | str]:
        """Solve ``-nu*Delta(u) - grad(p) = f`` with exact nonzero pressure.

        The saddle variable is the dual pressure ``H_Q p`` because the momentum
        block is ``B.T``.  The returned pressure error first reconstructs the
        primal pressure with ``H_Q^{-1}`` and then removes its best L2 constant.
        """
        nu = float(viscosity)
        if not np.isfinite(nu) or nu <= 0.0:
            raise ValueError("viscosity must be finite and positive")
        pi = np.pi
        ux_exact = lambda x, z: pi * np.sin(pi * x) * np.cos(pi * z)
        uz_exact = lambda x, z: -pi * np.cos(pi * x) * np.sin(pi * z)
        pressure_exact = lambda x, z: np.cos(pi * x) * np.cos(pi * z)
        pressure_x = lambda x, z: -pi * np.sin(pi * x) * np.cos(pi * z)
        pressure_z = lambda x, z: -pi * np.cos(pi * x) * np.sin(pi * z)
        force_ux = lambda x, z: 2.0 * nu * pi**2 * ux_exact(x, z) - pressure_x(x, z)
        force_uz = lambda x, z: 2.0 * nu * pi**2 * uz_exact(x, z) - pressure_z(x, z)
        zero = lambda x, z: np.zeros(np.broadcast_shapes(np.shape(x), np.shape(z)))
        body_load = self._assemble_viscous_rhs(
            force_ux,
            force_uz,
            zero,
            zero,
        )
        boundary_load = self._assemble_viscous_rhs(
            zero,
            zero,
            ux_exact,
            uz_exact,
        )
        load = body_load + nu * boundary_load
        bmat = self.divergence_matrix
        nq = self.core.pressure_size
        nv = self.active_indices.size
        one_dual = self.core.h1 @ np.ones(self.core.derivative_size)
        constant_dual = np.outer(one_dual, one_dual).ravel()
        constant_dual /= np.linalg.norm(constant_dual)
        system = np.block(
            [
                [nu * self.viscous_matrix, bmat.T, np.zeros((nv, 1))],
                [bmat, np.zeros((nq, nq)), constant_dual[:, None]],
                [np.zeros((1, nv)), constant_dual[None, :], np.zeros((1, 1))],
            ]
        )
        rhs = np.concatenate([load, np.zeros(nq + 1)])
        answer = np.linalg.solve(system, rhs)
        ux, uz = self._unpack_active(answer[:nv])
        dual_pressure = answer[nv : nv + nq]
        pressure_metric = np.kron(self.core.h1, self.core.h1)
        pressure = solve(
            pressure_metric,
            dual_pressure,
            assume_a="pos",
            check_finite=False,
        ).reshape(self.core.pressure_shape)
        residual = system @ answer - rhs
        _require_finite("manufactured Stokes solution", answer)
        _require_finite("manufactured primal pressure", pressure)

        axis = self.axis
        q, w = axis.q, axis.w
        weight = np.outer(w, w)
        xq, zq = np.meshgrid(q, q, indexing="ij")
        uxq = axis.v0 @ ux @ axis.v1.T
        uzq = axis.v1 @ uz @ axis.v0.T
        exact_ux = ux_exact(xq, zq)
        exact_uz = uz_exact(xq, zq)
        pressure_q = axis.v1 @ pressure @ axis.v1.T
        exact_pressure = pressure_exact(xq, zq)
        pressure_difference = pressure_q - exact_pressure
        pressure_constant_shift = float(np.sum(weight * pressure_difference)) / self.length**2
        pressure_difference -= pressure_constant_shift
        error_squared = float(
            np.sum(weight * ((uxq - exact_ux) ** 2 + (uzq - exact_uz) ** 2))
        )
        exact_squared = float(np.sum(weight * (exact_ux**2 + exact_uz**2)))
        pressure_error_squared = float(
            np.sum(weight * pressure_difference * pressure_difference)
        )
        pressure_exact_squared = float(np.sum(weight * exact_pressure * exact_pressure))
        divergence = self.core.divergence(ux, uz)
        divergence_squared = self.core.pressure_inner(divergence, divergence)
        return {
            "npts": self.npts,
            "degree": self.degree,
            "elements": self.npts - self.degree,
            "mesh_size": self.mesh_size,
            "velocity_error_l2_squared": error_squared,
            "velocity_exact_l2_squared": exact_squared,
            "relative_velocity_error": math.sqrt(error_squared / exact_squared),
            "pressure_unknown": "dual pressure H_Q p",
            "pressure_mod_constant_error_l2_squared": pressure_error_squared,
            "pressure_exact_l2_squared": pressure_exact_squared,
            "relative_pressure_mod_constant_error": math.sqrt(
                pressure_error_squared / pressure_exact_squared
            ),
            "pressure_mod_constant_shift": pressure_constant_shift,
            "divergence_l2_squared": divergence_squared,
            "normal_trace_max": self.core.normal_trace_max(ux, uz),
            "tangential_data_l2_squared": self._tangential_trace_norm_squared(
                ux_exact, uz_exact
            ),
            "linear_residual_l2_squared": float(residual @ residual),
            "linear_rhs_l2_squared": float(rhs @ rhs),
        }
