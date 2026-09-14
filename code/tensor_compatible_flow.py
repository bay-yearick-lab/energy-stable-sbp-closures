#!/usr/bin/env python3
"""Coefficient-space tensor operators for divergence-conforming BMF flow.

The production representation keeps the clamped spline coefficients. Mass,
stiffness, incidence, quadrature evaluation, and wall traces are stored as
banded or compact-support one-dimensional factors. Multidimensional residuals
are evaluated by tensor line application without assembled Kronecker matrices.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, minres, splu

from boundary_spline_analysis import (
    derivative_space_knots,
    greville_from_knots,
    open_uniform_knots,
)


Array = np.ndarray
Sparse = csr_matrix
ScalarField = Callable[[Array, Array], Array]


def _require_finite(name: str, value: Array | float) -> None:
    if not np.all(np.isfinite(np.asarray(value))):
        raise ValueError(f"non-finite values in {name}")


def _quadrature(knots: Array, order: int) -> tuple[Array, Array]:
    nodes, weights = leggauss(int(order))
    points: list[Array] = []
    scaled_weights: list[Array] = []
    unique = np.unique(np.asarray(knots, dtype=float))
    for left, right in zip(unique[:-1], unique[1:]):
        if right <= left:
            continue
        points.append(0.5 * (right - left) * nodes + 0.5 * (right + left))
        scaled_weights.append(0.5 * (right - left) * weights)
    return np.concatenate(points), np.concatenate(scaled_weights)


def _design(points: Array, knots: Array, degree: int) -> Sparse:
    matrix = BSpline.design_matrix(
        np.asarray(points, dtype=float),
        np.asarray(knots, dtype=float),
        int(degree),
        extrapolate=False,
    )
    result = csr_matrix(matrix)
    result.eliminate_zeros()
    return result


def _incidence(knots: Array, degree: int) -> Sparse:
    p = int(degree)
    n = np.asarray(knots).size - p - 1
    rows = np.repeat(np.arange(n - 1), 2)
    columns = np.column_stack([np.arange(n - 1), np.arange(1, n)]).ravel()
    values = np.empty(2 * (n - 1))
    for index in range(n - 1):
        scale = p / (knots[index + p + 1] - knots[index + 1])
        values[2 * index] = -scale
        values[2 * index + 1] = scale
    return csr_matrix((values, (rows, columns)), shape=(n - 1, n))


def _weighted_gram(evaluation: Sparse, weights: Array) -> Sparse:
    result = evaluation.T @ evaluation.multiply(np.asarray(weights)[:, None])
    result = csr_matrix(0.5 * (result + result.T))
    result.eliminate_zeros()
    return result


def _tensor_apply(left: Sparse, field: Array, right: Sparse) -> Array:
    """Apply ``left @ field @ right.T`` by two one-dimensional line passes."""
    first = left @ np.asarray(field, dtype=float)
    return np.asarray(right @ first.T).T


def _tensor_moments(left: Sparse, values: Array, right: Sparse) -> Array:
    """Apply ``left.T @ values @ right`` by two one-dimensional line passes."""
    first = left.T @ np.asarray(values, dtype=float)
    return np.asarray(right.T @ first.T).T


def _tensor_apply_nd(factors: tuple[Sparse, ...], field: Array) -> Array:
    """Apply one factor along each tensor axis without a Kronecker matrix."""
    result = np.asarray(field, dtype=float)
    if result.ndim != len(factors):
        raise ValueError("factor count must match tensor rank")
    for axis_index, factor in enumerate(factors):
        moved = np.moveaxis(result, axis_index, 0)
        transformed = factor @ moved.reshape(moved.shape[0], -1)
        moved_shape = (factor.shape[0],) + moved.shape[1:]
        result = np.moveaxis(np.asarray(transformed).reshape(moved_shape), 0, axis_index)
    return result


def _tensor_apply_sparse_3d(
    factors: tuple[Sparse, Sparse, Sparse],
    field: Array,
) -> Array:
    """Apply sparse line factors with plane-batched interior-axis work."""
    first_factor, middle_factor, last_factor = factors
    source = np.asarray(field, dtype=float)
    if source.ndim != 3:
        raise ValueError("three-dimensional line application requires rank three")
    first = first_factor @ source.reshape(source.shape[0], -1)
    first = np.asarray(first).reshape(
        first_factor.shape[0], source.shape[1], source.shape[2]
    )
    last = last_factor @ first.reshape(-1, source.shape[2]).T
    last = np.asarray(last).T.reshape(
        first_factor.shape[0], source.shape[1], last_factor.shape[0]
    )
    result = np.empty(
        (first_factor.shape[0], middle_factor.shape[0], last_factor.shape[0]),
        dtype=float,
    )
    for first_index in range(first_factor.shape[0]):
        result[first_index] = middle_factor @ last[first_index]
    return result


def _tensor_moments_nd(factors: tuple[Sparse, ...], values: Array) -> Array:
    """Apply transposed factors along every tensor axis."""
    return _tensor_apply_nd(tuple(factor.T.tocsr() for factor in factors), values)


def _localized_rows(factor: Sparse, rows: slice) -> tuple[Array, Sparse]:
    """Restrict a compact factor to one quadrature slab and its active columns."""
    selected = factor[rows].tocsr()
    columns = np.unique(selected.indices)
    return columns, selected[:, columns].tocsr()


def _sparse_bandwidth(matrix: Sparse, tolerance: float = 1.0e-14) -> int:
    coo = matrix.tocoo()
    keep = np.abs(coo.data) > tolerance
    if not np.any(keep):
        return 0
    return int(np.max(np.abs(coo.row[keep] - coo.col[keep])))


@dataclass(frozen=True)
class CoefficientAxis:
    """Compact one-dimensional factors for a primal and derivative spline pair."""

    npts: int
    degree: int
    length: float
    knots0: Array
    knots1: Array
    x0: Array
    x1: Array
    q: Array
    w: Array
    b0: Sparse
    b1: Sparse
    d0: Sparse
    d1: Sparse
    mass0: Sparse
    mass1: Sparse
    stiffness0: Sparse
    stiffness1: Sparse
    incidence: Sparse
    trace0: Array
    trace1: Array
    normal0: Array
    normal1: Array
    greville0: Sparse
    greville1: Sparse

    @property
    def derivative_size(self) -> int:
        return self.npts - 1

    @property
    def interior_size(self) -> int:
        return self.npts - 2

    @property
    def incidence_interior(self) -> Sparse:
        return self.incidence[:, 1:-1]

    @property
    def mass0_interior(self) -> Sparse:
        return self.mass0[1:-1, 1:-1]

    @property
    def stiffness0_interior(self) -> Sparse:
        return self.stiffness0[1:-1, 1:-1]

    def storage_record(self) -> dict[str, int]:
        factors = (
            self.b0,
            self.b1,
            self.d0,
            self.d1,
            self.mass0,
            self.mass1,
            self.stiffness0,
            self.stiffness1,
            self.incidence,
            self.greville0,
            self.greville1,
        )
        return {
            "stored_factor_entries": int(sum(value.nnz for value in factors)),
            "maximum_factor_bandwidth": max(
                _sparse_bandwidth(self.mass0),
                _sparse_bandwidth(self.mass1),
                _sparse_bandwidth(self.stiffness0),
                _sparse_bandwidth(self.stiffness1),
                _sparse_bandwidth(self.incidence),
            ),
        }


def coefficient_axis(
    npts: int,
    degree: int,
    length: float = 1.0,
    quadrature_points_per_span: int | None = None,
) -> CoefficientAxis:
    """Assemble compact-support factors for one clamped exact sequence."""
    n = int(npts)
    p = int(degree)
    domain_length = float(length)
    if p < 2 or n <= p:
        raise ValueError("coefficient flow requires 2 <= degree < npts")
    if not np.isfinite(domain_length) or domain_length <= 0.0:
        raise ValueError("length must be finite and positive")
    knots0 = open_uniform_knots(n, p, domain_length)
    knots1 = derivative_space_knots(knots0)
    knots2 = derivative_space_knots(knots1)
    incidence0 = _incidence(knots0, p)
    incidence1 = _incidence(knots1, p - 1)
    order = max((3 * p + 1) // 2, 3) if quadrature_points_per_span is None else int(
        quadrature_points_per_span
    )
    q, w = _quadrature(knots0, order)
    b0 = _design(q, knots0, p)
    b1 = _design(q, knots1, p - 1)
    b2 = _design(q, knots2, p - 2)
    d0 = csr_matrix(b1 @ incidence0)
    d1 = csr_matrix(b2 @ incidence1)
    mass0 = _weighted_gram(b0, w)
    mass1 = _weighted_gram(b1, w)
    stiffness0 = _weighted_gram(d0, w)
    stiffness1 = _weighted_gram(d1, w)

    endpoints = np.array([0.0, domain_length])
    trace0 = _design(endpoints, knots0, p).toarray()
    trace1 = _design(endpoints, knots1, p - 1).toarray()
    normal0 = (_design(endpoints, knots1, p - 1) @ incidence0).toarray()
    normal1 = (_design(endpoints, knots2, p - 2) @ incidence1).toarray()
    normal0[0] *= -1.0
    normal1[0] *= -1.0

    x0 = greville_from_knots(knots0, p, n)
    if p - 1 == 0:
        x1 = 0.5 * (knots1[: n - 1] + knots1[1:n])
    else:
        x1 = greville_from_knots(knots1, p - 1, n - 1)
    greville0 = _design(x0, knots0, p)
    greville1 = _design(x1, knots1, p - 1)

    for name, value in (
        ("quadrature points", q),
        ("quadrature weights", w),
        ("primal trace", trace0),
        ("derivative trace", trace1),
        ("primal normal derivative", normal0),
        ("derivative normal derivative", normal1),
    ):
        _require_finite(name, value)
    return CoefficientAxis(
        npts=n,
        degree=p,
        length=domain_length,
        knots0=knots0,
        knots1=knots1,
        x0=x0,
        x1=x1,
        q=q,
        w=w,
        b0=b0,
        b1=b1,
        d0=d0,
        d1=d1,
        mass0=mass0,
        mass1=mass1,
        stiffness0=stiffness0,
        stiffness1=stiffness1,
        incidence=incidence0,
        trace0=trace0,
        trace1=trace1,
        normal0=normal0,
        normal1=normal1,
        greville0=greville0,
        greville1=greville1,
    )


@dataclass(frozen=True)
class VelocityPair:
    ux: Array
    uz: Array


@dataclass(frozen=True)
class ResidualChannels:
    mass: VelocityPair
    viscous: VelocityPair
    pressure: VelocityPair
    convection: VelocityPair
    wall: VelocityPair
    divergence: Array


@dataclass(frozen=True)
class VelocityTriple:
    ux: Array
    uy: Array
    uz: Array


@dataclass(frozen=True)
class ResidualChannels3D:
    mass: VelocityTriple
    viscous: VelocityTriple
    pressure: VelocityTriple
    convection: VelocityTriple
    divergence: Array


@dataclass(frozen=True)
class SaddleSolution2D:
    ux: Array
    uz: Array
    dual_pressure: Array
    gauge_multiplier: float
    iterations: int
    relative_residual: float
    divergence_relative_residual: float
    pressure_gauge_relative_residual: float


class TensorCompatibleFlow2D:
    """Matrix-free coefficient residual for the route-C BMF flow complex."""

    def __init__(
        self,
        npts: int,
        degree: int,
        length: float = 1.0,
        penalty_gamma: float | None = None,
        coefficient_factors: CoefficientAxis | None = None,
        mesh_size: float | None = None,
    ) -> None:
        self.axis = (
            coefficient_axis(npts, degree, length)
            if coefficient_factors is None
            else coefficient_factors
        )
        if (
            self.axis.npts != int(npts)
            or self.axis.degree != int(degree)
            or not np.isclose(self.axis.length, float(length))
        ):
            raise ValueError("coefficient factors do not match the requested grid")
        self.npts = self.axis.npts
        self.degree = self.axis.degree
        self.length = self.axis.length
        self.mesh_size = (
            self.length / (self.npts - self.degree)
            if mesh_size is None
            else float(mesh_size)
        )
        if not np.isfinite(self.mesh_size) or self.mesh_size <= 0.0:
            raise ValueError("mesh_size must be finite and positive")
        gamma = 4.0 * self.degree**2 if penalty_gamma is None else float(penalty_gamma)
        if not np.isfinite(gamma) or gamma < 4.0 * self.degree**2:
            raise ValueError("penalty_gamma must be finite and at least 4 p^2")
        self.penalty_gamma = gamma
        self.penalty = gamma / self.mesh_size
        self.ux_shape = (self.npts, self.npts - 1)
        self.uz_shape = (self.npts - 1, self.npts)
        self.pressure_shape = (self.npts - 1, self.npts - 1)
        self.tangential_factor = self._tangential_factor()

    def _tangential_factor(self) -> Sparse:
        axis = self.axis
        boundary = -axis.normal1.T @ axis.trace1 - axis.trace1.T @ axis.normal1
        boundary += self.penalty * axis.trace1.T @ axis.trace1
        result = csr_matrix(axis.stiffness1 + csr_matrix(boundary))
        result.eliminate_zeros()
        return result

    def _velocity(self, ux: Array, uz: Array) -> tuple[Array, Array]:
        x = np.asarray(ux, dtype=float)
        z = np.asarray(uz, dtype=float)
        if x.shape != self.ux_shape or z.shape != self.uz_shape:
            raise ValueError(
                f"velocity shapes must be {self.ux_shape} and {self.uz_shape}"
            )
        _require_finite("ux coefficients", x)
        _require_finite("uz coefficients", z)
        return x, z

    def _pressure(self, pressure: Array) -> Array:
        value = np.asarray(pressure, dtype=float)
        if value.shape != self.pressure_shape:
            raise ValueError(f"pressure must have shape {self.pressure_shape}")
        _require_finite("pressure coefficients", value)
        return value

    def require_no_penetration(self, ux: Array, uz: Array) -> None:
        x, z = self._velocity(ux, uz)
        trace = max(
            float(np.max(np.abs(x[[0, -1], :]))),
            float(np.max(np.abs(z[:, [0, -1]]))),
        )
        if trace > 2.0e-13:
            raise ValueError("normal velocity coefficients must vanish at the walls")

    def mass_action(self, ux: Array, uz: Array) -> VelocityPair:
        x, z = self._velocity(ux, uz)
        axis = self.axis
        return VelocityPair(
            ux=_tensor_apply(axis.mass0, x, axis.mass1),
            uz=_tensor_apply(axis.mass1, z, axis.mass0),
        )

    def viscous_action(self, ux: Array, uz: Array) -> VelocityPair:
        x, z = self._velocity(ux, uz)
        axis = self.axis
        ux_value = _tensor_apply(axis.stiffness0, x, axis.mass1)
        ux_value += _tensor_apply(axis.mass0, x, self.tangential_factor)
        uz_value = _tensor_apply(self.tangential_factor, z, axis.mass0)
        uz_value += _tensor_apply(axis.mass1, z, axis.stiffness0)
        return VelocityPair(ux=ux_value, uz=uz_value)

    def divergence(self, ux: Array, uz: Array) -> Array:
        x, z = self._velocity(ux, uz)
        result = self.axis.incidence @ x + np.asarray(self.axis.incidence @ z.T).T
        _require_finite("coefficient divergence", result)
        return result

    def pressure_action(self, dual_pressure: Array) -> VelocityPair:
        value = self._pressure(dual_pressure)
        incidence = self.axis.incidence
        return VelocityPair(
            ux=np.asarray(incidence.T @ value),
            uz=np.asarray(incidence.T @ value.T).T,
        )

    def pressure_gauge_vector(self) -> Array:
        constant_axis = np.asarray(self.axis.mass1 @ np.ones(self.npts - 1)).ravel()
        return np.outer(constant_axis, constant_axis)

    def pressure_gauge_residual(self, dual_pressure: Array) -> float:
        value = self._pressure(dual_pressure)
        gauge = self.pressure_gauge_vector()
        return abs(float(np.sum(gauge * value))) / max(
            float(np.linalg.norm(gauge) * np.linalg.norm(value)), 1.0e-300
        )

    def convection_dual(self, ux: Array, uz: Array) -> VelocityPair:
        """Return the exact-quadrature conservative/skew dual moments."""
        x, z = self._velocity(ux, uz)
        axis = self.axis
        weight = np.outer(axis.w, axis.w)
        uxq = _tensor_apply(axis.b0, x, axis.b1)
        uzq = _tensor_apply(axis.b1, z, axis.b0)
        ux_x = _tensor_apply(axis.d0, x, axis.b1)
        ux_z = _tensor_apply(axis.b0, x, axis.d1)
        uz_x = _tensor_apply(axis.d1, z, axis.b0)
        uz_z = _tensor_apply(axis.b1, z, axis.d0)
        adv_ux = uxq * ux_x + uzq * ux_z
        adv_uz = uxq * uz_x + uzq * uz_z
        ux_moments = 0.5 * (
            _tensor_moments(axis.b0, weight * adv_ux, axis.b1)
            - _tensor_moments(axis.d0, weight * (uxq * uxq), axis.b1)
            - _tensor_moments(axis.b0, weight * (uzq * uxq), axis.d1)
        )
        uz_moments = 0.5 * (
            _tensor_moments(axis.b1, weight * adv_uz, axis.b0)
            - _tensor_moments(axis.d1, weight * (uxq * uzq), axis.b0)
            - _tensor_moments(axis.b1, weight * (uzq * uzq), axis.d0)
        )
        _require_finite("ux convective moments", ux_moments)
        _require_finite("uz convective moments", uz_moments)
        return VelocityPair(ux=ux_moments, uz=uz_moments)

    def wall_load(
        self,
        trace_ux: ScalarField,
        trace_uz: ScalarField,
    ) -> VelocityPair:
        """Assemble the tangential Nitsche load in coefficient dual coordinates."""
        axis = self.axis
        ux_moments = np.zeros(self.ux_shape)
        uz_moments = np.zeros(self.uz_shape)
        for side, coordinate in enumerate((0.0, self.length)):
            gx = np.asarray(trace_ux(axis.q, np.full_like(axis.q, coordinate)))
            gz = np.asarray(trace_uz(np.full_like(axis.q, coordinate), axis.q))
            ux_axis = np.asarray(axis.b0.T @ (axis.w * gx)).ravel()
            uz_axis = np.asarray(axis.b0.T @ (axis.w * gz)).ravel()
            boundary_row = -axis.normal1[side] + self.penalty * axis.trace1[side]
            ux_moments += np.outer(ux_axis, boundary_row)
            uz_moments += np.outer(boundary_row, uz_axis)
        _require_finite("ux wall load", ux_moments)
        _require_finite("uz wall load", uz_moments)
        return VelocityPair(ux=ux_moments, uz=uz_moments)

    def body_load(
        self,
        force_ux: ScalarField,
        force_uz: ScalarField,
    ) -> VelocityPair:
        """Assemble volume-force moments in velocity dual coordinates."""
        axis = self.axis
        x = axis.q[:, None]
        z = axis.q[None, :]
        weight = axis.w[:, None] * axis.w[None, :]
        ux_values = np.broadcast_to(np.asarray(force_ux(x, z), dtype=float), weight.shape)
        uz_values = np.broadcast_to(np.asarray(force_uz(x, z), dtype=float), weight.shape)
        ux_moments = _tensor_moments(axis.b0, weight * ux_values, axis.b1)
        uz_moments = _tensor_moments(axis.b1, weight * uz_values, axis.b0)
        _require_finite("ux body load", ux_moments)
        _require_finite("uz body load", uz_moments)
        return VelocityPair(ux=ux_moments, uz=uz_moments)

    def quadrature_velocity(self, ux: Array, uz: Array) -> VelocityPair:
        """Evaluate both velocity components on the tensor quadrature grid."""
        x, z = self._velocity(ux, uz)
        return VelocityPair(
            ux=_tensor_apply(self.axis.b0, x, self.axis.b1),
            uz=_tensor_apply(self.axis.b1, z, self.axis.b0),
        )

    def quadrature_pressure(self, pressure: Array) -> Array:
        """Evaluate primal pressure coefficients on the tensor quadrature grid."""
        value = self._pressure(pressure)
        return _tensor_apply(self.axis.b1, value, self.axis.b1)

    def evaluation_factors(self, points: Array) -> tuple[Sparse, Sparse]:
        """Return primal and derivative-space factors at physical points."""
        coordinates = np.asarray(points, dtype=float)
        if coordinates.ndim != 1:
            raise ValueError("evaluation points must be one-dimensional")
        if np.any(coordinates < 0.0) or np.any(coordinates > self.length):
            raise ValueError("evaluation points must lie in the domain")
        return (
            _design(coordinates, self.axis.knots0, self.degree),
            _design(coordinates, self.axis.knots1, self.degree - 1),
        )

    def evaluate_velocity(
        self,
        ux: Array,
        uz: Array,
        x_points: Array,
        z_points: Array,
    ) -> VelocityPair:
        """Evaluate velocity on a requested tensor-product point grid."""
        x, z = self._velocity(ux, uz)
        x0, x1 = self.evaluation_factors(x_points)
        z0, z1 = self.evaluation_factors(z_points)
        return VelocityPair(
            ux=_tensor_apply(x0, x, z1),
            uz=_tensor_apply(x1, z, z0),
        )

    def evaluate_pressure(
        self,
        pressure: Array,
        x_points: Array,
        z_points: Array,
    ) -> Array:
        """Evaluate primal pressure on a requested tensor-product point grid."""
        value = self._pressure(pressure)
        _x0, x1 = self.evaluation_factors(x_points)
        _z0, z1 = self.evaluation_factors(z_points)
        return _tensor_apply(x1, value, z1)

    def channel_residuals(
        self,
        ux: Array,
        uz: Array,
        dual_pressure: Array,
        trace_ux: ScalarField,
        trace_uz: ScalarField,
    ) -> ResidualChannels:
        self.require_no_penetration(ux, uz)
        return ResidualChannels(
            mass=self.mass_action(ux, uz),
            viscous=self.viscous_action(ux, uz),
            pressure=self.pressure_action(dual_pressure),
            convection=self.convection_dual(ux, uz),
            wall=self.wall_load(trace_ux, trace_uz),
            divergence=self.divergence(ux, uz),
        )

    def kinetic_energy(self, ux: Array, uz: Array) -> float:
        x, z = self._velocity(ux, uz)
        mass = self.mass_action(x, z)
        return 0.5 * float(np.sum(x * mass.ux) + np.sum(z * mass.uz))

    def convective_energy_production(self, ux: Array, uz: Array) -> float:
        x, z = self._velocity(ux, uz)
        convection = self.convection_dual(x, z)
        return float(np.sum(x * convection.ux) + np.sum(z * convection.uz))

    def nodal_velocity(self, ux: Array, uz: Array) -> VelocityPair:
        x, z = self._velocity(ux, uz)
        return VelocityPair(
            ux=_tensor_apply(self.axis.greville0, x, self.axis.greville1),
            uz=_tensor_apply(self.axis.greville1, z, self.axis.greville0),
        )

    def nodal_dual_pressure(self, dual_pressure: Array) -> Array:
        """Transform a coefficient dual pressure into derivative-grid dual coordinates."""
        value = self._pressure(dual_pressure)
        basis = self.axis.greville1.toarray()
        left = np.linalg.solve(basis.T, value)
        result = np.linalg.solve(basis.T, left.T).T
        _require_finite("nodal dual pressure", result)
        return result

    def coefficient_dual_from_nodal(
        self,
        ux_dual: Array,
        uz_dual: Array,
    ) -> VelocityPair:
        """Apply the transpose basis maps to nodal active dual residuals."""
        x = np.asarray(ux_dual, dtype=float)
        z = np.asarray(uz_dual, dtype=float)
        expected_x = (self.npts - 2, self.npts - 1)
        expected_z = (self.npts - 1, self.npts - 2)
        if x.shape != expected_x or z.shape != expected_z:
            raise ValueError(f"active nodal dual shapes must be {expected_x} and {expected_z}")
        primal_interior = self.axis.greville0[1:-1, 1:-1]
        return VelocityPair(
            ux=_tensor_moments(primal_interior, x, self.axis.greville1),
            uz=_tensor_moments(self.axis.greville1, z, primal_interior),
        )

    def wall_traction(
        self,
        ux: Array,
        uz: Array,
        pressure: Array,
        viscosity: float,
    ) -> dict[str, dict[str, Array]]:
        """Evaluate velocity, pressure, flux, and traction from the same state."""
        x, z = self._velocity(ux, uz)
        p = self._pressure(pressure)
        nu = float(viscosity)
        if not np.isfinite(nu) or nu < 0.0:
            raise ValueError("viscosity must be finite and nonnegative")
        axis = self.axis
        result: dict[str, dict[str, Array]] = {}

        for side, name, sign in ((0, "x0", -1.0), (1, "x1", 1.0)):
            b0 = csr_matrix(axis.trace0[[side]])
            b1 = csr_matrix(axis.trace1[[side]])
            d0 = csr_matrix(axis.normal0[[side]] * sign)
            d1 = csr_matrix(axis.normal1[[side]] * sign)
            pressure_wall = _tensor_apply(b1, p, axis.b1).ravel()
            ux_value = _tensor_apply(b0, x, axis.b1).ravel()
            uz_value = _tensor_apply(b1, z, axis.b0).ravel()
            ux_x = _tensor_apply(d0, x, axis.b1).ravel()
            ux_z = _tensor_apply(b0, x, axis.d1).ravel()
            uz_x = _tensor_apply(d1, z, axis.b0).ravel()
            normal_flux = sign * ux_value
            result[name] = {
                "coordinate": axis.q.copy(),
                "normal_flux": normal_flux,
                "traction_x": sign * (-pressure_wall + 2.0 * nu * ux_x),
                "traction_z": sign * nu * (uz_x + ux_z),
            }

        for side, name, sign in ((0, "z0", -1.0), (1, "z1", 1.0)):
            b0 = csr_matrix(axis.trace0[[side]])
            b1 = csr_matrix(axis.trace1[[side]])
            d0 = csr_matrix(axis.normal0[[side]] * sign)
            d1 = csr_matrix(axis.normal1[[side]] * sign)
            pressure_wall = _tensor_apply(axis.b1, p, b1).ravel()
            ux_value = _tensor_apply(axis.b0, x, b1).ravel()
            uz_value = _tensor_apply(axis.b1, z, b0).ravel()
            ux_z = _tensor_apply(axis.b0, x, d1).ravel()
            uz_x = _tensor_apply(axis.d1, z, b0).ravel()
            uz_z = _tensor_apply(axis.b1, z, d0).ravel()
            normal_flux = sign * uz_value
            result[name] = {
                "coordinate": axis.q.copy(),
                "normal_flux": normal_flux,
                "traction_x": sign * nu * (ux_z + uz_x),
                "traction_z": sign * (-pressure_wall + 2.0 * nu * uz_z),
            }
        for wall in result.values():
            for key, value in wall.items():
                if key != "coordinate":
                    _require_finite(f"wall {key}", value)
        return result

    def storage_record(self) -> dict[str, int]:
        record = self.axis.storage_record()
        record.update(
            {
                "tangential_factor_entries": int(self.tangential_factor.nnz),
                "velocity_states": int(
                    np.prod(self.ux_shape) + np.prod(self.uz_shape)
                ),
                "pressure_states": int(np.prod(self.pressure_shape)),
            }
        )
        return record


class TensorCompatibleSaddle2D:
    """Matrix-free pressure-coupled solve with reusable one-dimensional factors."""

    def __init__(
        self,
        flow: TensorCompatibleFlow2D,
        alpha: float,
        viscosity: float,
        relative_tolerance: float = 1.0e-11,
        maximum_iterations: int = 2000,
        pressure_preconditioner: str = "product_schur",
    ) -> None:
        self.flow = flow
        self.alpha = float(alpha)
        self.viscosity = float(viscosity)
        self.relative_tolerance = float(relative_tolerance)
        self.maximum_iterations = int(maximum_iterations)
        self.pressure_preconditioner = str(pressure_preconditioner)
        if not np.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha must be finite and positive")
        if not np.isfinite(self.viscosity) or self.viscosity < 0.0:
            raise ValueError("viscosity must be finite and nonnegative")
        if not np.isfinite(self.relative_tolerance) or self.relative_tolerance <= 0.0:
            raise ValueError("relative_tolerance must be finite and positive")
        if self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive")
        if self.pressure_preconditioner not in {
            "diagonal",
            "mass_schur",
            "product_schur",
        }:
            raise ValueError(
                "pressure_preconditioner must be diagonal, mass_schur, or product_schur"
            )

        axis = flow.axis
        self.m0 = axis.mass0_interior.tocsr()
        self.m1 = axis.mass1.tocsr()
        self.k0 = axis.stiffness0_interior.tocsr()
        self.k1 = flow.tangential_factor.tocsr()
        self.incidence = axis.incidence_interior.tocsr()
        self.ux_active_shape = (axis.interior_size, axis.derivative_size)
        self.uz_active_shape = (axis.derivative_size, axis.interior_size)
        self.nux = int(np.prod(self.ux_active_shape))
        self.nuz = int(np.prod(self.uz_active_shape))
        self.nv = self.nux + self.nuz
        self.nq = int(np.prod(flow.pressure_shape))
        self.size = self.nv + self.nq + 1

        tau = self.viscosity / self.alpha
        p0 = (self.m0 + tau * self.k0).tocsc()
        p1 = (self.m1 + tau * self.k1).tocsc()
        self.factor0 = splu(
            p0,
            permc_spec="NATURAL",
            diag_pivot_thresh=0.0,
            options={"SymmetricMode": True},
        )
        self.factor1 = splu(
            p1,
            permc_spec="NATURAL",
            diag_pivot_thresh=0.0,
            options={"SymmetricMode": True},
        )
        self.pressure_mass_factor = splu(
            self.m1.tocsc(),
            permc_spec="NATURAL",
            diag_pivot_thresh=0.0,
            options={"SymmetricMode": True},
        )
        constant_axis = np.asarray(axis.mass1 @ np.ones(axis.derivative_size)).ravel()
        self.gauge = np.outer(constant_axis, constant_axis).ravel()
        self.gauge /= np.linalg.norm(self.gauge)
        self.pressure_inverse_diagonal = self._pressure_inverse_diagonal()
        self.pressure_cholesky: Array | None = None
        self.pressure_schur_vectors: Array | None = None
        self.pressure_schur_denominator: Array | None = None
        if self.pressure_preconditioner == "mass_schur":
            self._build_separable_schur_preconditioner(self.m0, self.m1)
        elif self.pressure_preconditioner == "product_schur":
            self._build_separable_schur_preconditioner(p0.tocsr(), p1.tocsr())
        self.operator = LinearOperator(
            (self.size, self.size), matvec=self._matvec, dtype=float
        )
        self.preconditioner = LinearOperator(
            (self.size, self.size), matvec=self._precondition, dtype=float
        )

    def _active_dual(self, ux: Array, uz: Array) -> tuple[Array, Array]:
        x, z = self.flow._velocity(ux, uz)
        return x[1:-1, :], z[:, 1:-1]

    def _full_velocity(self, ux_active: Array, uz_active: Array) -> VelocityPair:
        ux = np.zeros(self.flow.ux_shape)
        uz = np.zeros(self.flow.uz_shape)
        ux[1:-1, :] = np.asarray(ux_active, dtype=float)
        uz[:, 1:-1] = np.asarray(uz_active, dtype=float)
        return VelocityPair(ux=ux, uz=uz)

    def _pack_velocity(self, ux_active: Array, uz_active: Array) -> Array:
        return np.concatenate(
            [np.asarray(ux_active, dtype=float).ravel(), np.asarray(uz_active, dtype=float).ravel()]
        )

    def _unpack_velocity(self, vector: Array) -> tuple[Array, Array]:
        value = np.asarray(vector, dtype=float)
        if value.shape != (self.nv,):
            raise ValueError("active velocity vector has the wrong shape")
        return (
            value[: self.nux].reshape(self.ux_active_shape),
            value[self.nux :].reshape(self.uz_active_shape),
        )

    def _velocity_action(self, ux: Array, uz: Array) -> VelocityPair:
        mass_ux = _tensor_apply(self.m0, ux, self.m1)
        mass_uz = _tensor_apply(self.m1, uz, self.m0)
        viscous_ux = _tensor_apply(self.k0, ux, self.m1)
        viscous_ux += _tensor_apply(self.m0, ux, self.k1)
        viscous_uz = _tensor_apply(self.k1, uz, self.m0)
        viscous_uz += _tensor_apply(self.m1, uz, self.k0)
        return VelocityPair(
            ux=self.alpha * mass_ux + self.viscosity * viscous_ux,
            uz=self.alpha * mass_uz + self.viscosity * viscous_uz,
        )

    def _divergence_active(self, ux: Array, uz: Array) -> Array:
        return self.incidence @ ux + np.asarray(self.incidence @ uz.T).T

    def _gradient_active(self, pressure: Array) -> VelocityPair:
        value = np.asarray(pressure, dtype=float).reshape(self.flow.pressure_shape)
        return VelocityPair(
            ux=np.asarray(self.incidence.T @ value),
            uz=np.asarray(self.incidence.T @ value.T).T,
        )

    def _matvec(self, vector: Array) -> Array:
        value = np.asarray(vector, dtype=float)
        ux, uz = self._unpack_velocity(value[: self.nv])
        pressure = value[self.nv : self.nv + self.nq].reshape(self.flow.pressure_shape)
        gauge_multiplier = float(value[-1])
        velocity = self._velocity_action(ux, uz)
        gradient = self._gradient_active(pressure)
        velocity_result = self._pack_velocity(
            velocity.ux + gradient.ux,
            velocity.uz + gradient.uz,
        )
        divergence = self._divergence_active(ux, uz).ravel()
        divergence += gauge_multiplier * self.gauge
        gauge_result = float(self.gauge @ pressure.ravel())
        return np.concatenate([velocity_result, divergence, np.array([gauge_result])])

    def _product_solve(self, ux: Array, uz: Array) -> VelocityPair:
        ux_first = self.factor0.solve(np.asarray(ux, dtype=float))
        ux_result = self.factor1.solve(ux_first.T).T / self.alpha
        uz_first = self.factor1.solve(np.asarray(uz, dtype=float))
        uz_result = self.factor0.solve(uz_first.T).T / self.alpha
        return VelocityPair(ux=ux_result, uz=uz_result)

    def _pressure_inverse_diagonal(self) -> Array:
        m0 = self.m0.diagonal()
        m1 = self.m1.diagonal()
        k0 = self.k0.diagonal()
        k1 = self.k1.diagonal()
        ax = self.alpha * m0[:, None] * m1[None, :]
        ax += self.viscosity * (
            k0[:, None] * m1[None, :] + m0[:, None] * k1[None, :]
        )
        az = self.alpha * m1[:, None] * m0[None, :]
        az += self.viscosity * (
            k1[:, None] * m0[None, :] + m1[:, None] * k0[None, :]
        )
        incidence_squared = self.incidence.multiply(self.incidence)
        first = incidence_squared @ (1.0 / ax)
        second = np.asarray(incidence_squared @ (1.0 / az).T).T
        diagonal = np.asarray(first + second)
        floor = max(float(np.max(diagonal)) * 1.0e-14, np.finfo(float).tiny)
        return 1.0 / np.maximum(diagonal, floor)

    def _build_separable_schur_preconditioner(
        self,
        metric0: Sparse,
        metric1: Sparse,
    ) -> None:
        """Factor the Schur operator induced by separable velocity metrics."""
        mass0 = metric0.toarray()
        mass1 = metric1.toarray()
        incidence = self.incidence.toarray()
        pressure_cholesky = np.linalg.cholesky(mass1)
        one_dimensional_schur = incidence @ np.linalg.solve(mass0, incidence.T)
        transformed = pressure_cholesky.T @ one_dimensional_schur @ pressure_cholesky
        transformed = 0.5 * (transformed + transformed.T)
        eigenvalues, eigenvectors = np.linalg.eigh(transformed)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        denominator = eigenvalues[:, None] + eigenvalues[None, :]
        tolerance = max(float(np.max(denominator)) * 1.0e-13, 1.0e-300)
        denominator = np.where(denominator > tolerance, denominator, 1.0)
        self.pressure_cholesky = pressure_cholesky
        self.pressure_schur_vectors = eigenvectors
        self.pressure_schur_denominator = denominator

    def _separable_schur_inverse(self, pressure: Array) -> Array:
        if (
            self.pressure_cholesky is None
            or self.pressure_schur_vectors is None
            or self.pressure_schur_denominator is None
        ):
            raise RuntimeError("mass Schur preconditioner is not available")
        cholesky = self.pressure_cholesky
        vectors = self.pressure_schur_vectors
        transformed = cholesky.T @ pressure @ cholesky
        transformed = vectors.T @ transformed @ vectors
        transformed = transformed / self.pressure_schur_denominator
        result = cholesky @ (vectors @ transformed @ vectors.T) @ cholesky.T
        return self.alpha * result

    def _precondition(self, vector: Array) -> Array:
        value = np.asarray(vector, dtype=float)
        ux, uz = self._unpack_velocity(value[: self.nv])
        velocity = self._product_solve(ux, uz)
        pressure = value[self.nv : self.nv + self.nq].reshape(self.flow.pressure_shape)
        if self.pressure_preconditioner in {"mass_schur", "product_schur"}:
            pressure_result = self._separable_schur_inverse(pressure)
        else:
            pressure_result = self.pressure_inverse_diagonal * pressure
        return np.concatenate(
            [
                self._pack_velocity(velocity.ux, velocity.uz),
                pressure_result.ravel(),
                value[-1:],
            ]
        )

    def solve(self, rhs: VelocityPair) -> SaddleSolution2D:
        ux_rhs, uz_rhs = self._active_dual(rhs.ux, rhs.uz)
        vector_rhs = np.concatenate(
            [self._pack_velocity(ux_rhs, uz_rhs), np.zeros(self.nq + 1)]
        )
        iterations = 0

        def count(_iterate: Array) -> None:
            nonlocal iterations
            iterations += 1

        answer = np.zeros(self.size)
        rhs_norm = max(float(np.linalg.norm(vector_rhs)), 1.0e-300)
        residual = vector_rhs.copy()
        relative_residual = 1.0
        divergence_relative_residual = math.inf
        pressure_gauge_relative_residual = math.inf
        internal_tolerance = max(
            5.0 * np.finfo(float).eps,
            self.relative_tolerance * 1.0e-4,
        )
        minimum_corrections = (
            2
            if self.pressure_preconditioner == "product_schur"
            and self.relative_tolerance <= 1.0e-11
            else 1
        )
        for correction_index in range(3):
            correction, info = minres(
                self.operator,
                residual,
                M=self.preconditioner,
                rtol=internal_tolerance,
                maxiter=self.maximum_iterations,
                callback=count,
                show=False,
                check=False,
            )
            if info != 0:
                raise RuntimeError(
                    f"matrix-free saddle solve did not converge: info={info}"
                )
            answer += correction
            residual = vector_rhs - self._matvec(answer)
            relative_residual = float(np.linalg.norm(residual)) / rhs_norm
            candidate_ux, candidate_uz = self._unpack_velocity(answer[: self.nv])
            candidate_divergence = self._divergence_active(
                candidate_ux, candidate_uz
            )
            candidate_divergence_scale = max(
                float(np.linalg.norm(self.incidence @ candidate_ux))
                + float(
                    np.linalg.norm(np.asarray(self.incidence @ candidate_uz.T).T)
                ),
                1.0e-300,
            )
            divergence_relative_residual = float(
                np.linalg.norm(candidate_divergence)
            ) / candidate_divergence_scale
            candidate_pressure = answer[
                self.nv : self.nv + self.nq
            ]
            pressure_gauge_relative_residual = abs(
                float(self.gauge @ candidate_pressure)
            ) / max(float(np.linalg.norm(candidate_pressure)), 1.0e-300)
            if (
                relative_residual <= self.relative_tolerance
                and divergence_relative_residual <= 2.0e-11
                and pressure_gauge_relative_residual <= 2.0e-11
                and correction_index + 1 >= minimum_corrections
            ):
                break
        if (
            relative_residual > self.relative_tolerance
            or divergence_relative_residual > 2.0e-11
            or pressure_gauge_relative_residual > 2.0e-11
        ):
            raise RuntimeError(
                "matrix-free saddle refinement missed a residual or invariant tolerance"
            )
        ux_active, uz_active = self._unpack_velocity(answer[: self.nv])
        velocity = self._full_velocity(ux_active, uz_active)
        pressure = answer[self.nv : self.nv + self.nq].reshape(self.flow.pressure_shape)
        divergence = self._divergence_active(ux_active, uz_active)
        divergence_scale = max(
            float(np.linalg.norm(self.incidence @ ux_active))
            + float(np.linalg.norm(np.asarray(self.incidence @ uz_active.T).T)),
            1.0e-300,
        )
        pressure_scale = max(float(np.linalg.norm(pressure)), 1.0e-300)
        return SaddleSolution2D(
            ux=velocity.ux,
            uz=velocity.uz,
            dual_pressure=pressure,
            gauge_multiplier=float(answer[-1]),
            iterations=iterations,
            relative_residual=relative_residual,
            divergence_relative_residual=float(np.linalg.norm(divergence))
            / divergence_scale,
            pressure_gauge_relative_residual=abs(float(self.gauge @ pressure.ravel()))
            / pressure_scale,
        )

    def primal_pressure(self, dual_pressure: Array) -> Array:
        """Convert pressure dual moments to primal spline coefficients."""
        value = np.asarray(dual_pressure, dtype=float)
        if value.shape != self.flow.pressure_shape:
            raise ValueError(f"dual pressure must have shape {self.flow.pressure_shape}")
        left = self.pressure_mass_factor.solve(value)
        result = self.pressure_mass_factor.solve(left.T).T
        _require_finite("primal pressure coefficients", result)
        return result

    def dense_oracle(self, rhs: VelocityPair) -> SaddleSolution2D:
        """Solve the corrected coefficient saddle system for small verification cells."""
        ux_rhs, uz_rhs = self._active_dual(rhs.ux, rhs.uz)
        m0 = self.m0.toarray()
        m1 = self.m1.toarray()
        k0 = self.k0.toarray()
        k1 = self.k1.toarray()
        incidence = self.incidence.toarray()
        ax = self.alpha * np.kron(m0, m1) + self.viscosity * (
            np.kron(k0, m1) + np.kron(m0, k1)
        )
        az = self.alpha * np.kron(m1, m0) + self.viscosity * (
            np.kron(k1, m0) + np.kron(m1, k0)
        )
        velocity_matrix = np.zeros((self.nv, self.nv))
        velocity_matrix[: self.nux, : self.nux] = ax
        velocity_matrix[self.nux :, self.nux :] = az
        identity = np.eye(self.flow.axis.derivative_size)
        divergence = np.column_stack(
            [np.kron(incidence, identity), np.kron(identity, incidence)]
        )
        system = np.block(
            [
                [velocity_matrix, divergence.T, np.zeros((self.nv, 1))],
                [divergence, np.zeros((self.nq, self.nq)), self.gauge[:, None]],
                [np.zeros((1, self.nv)), self.gauge[None, :], np.zeros((1, 1))],
            ]
        )
        vector_rhs = np.concatenate(
            [self._pack_velocity(ux_rhs, uz_rhs), np.zeros(self.nq + 1)]
        )
        answer = np.linalg.solve(system, vector_rhs)
        residual = system @ answer - vector_rhs
        ux_active, uz_active = self._unpack_velocity(answer[: self.nv])
        velocity = self._full_velocity(ux_active, uz_active)
        pressure = answer[self.nv : self.nv + self.nq].reshape(self.flow.pressure_shape)
        divergence_value = self._divergence_active(ux_active, uz_active)
        divergence_scale = max(
            float(np.linalg.norm(self.incidence @ ux_active))
            + float(np.linalg.norm(np.asarray(self.incidence @ uz_active.T).T)),
            1.0e-300,
        )
        pressure_scale = max(float(np.linalg.norm(pressure)), 1.0e-300)
        return SaddleSolution2D(
            ux=velocity.ux,
            uz=velocity.uz,
            dual_pressure=pressure,
            gauge_multiplier=float(answer[-1]),
            iterations=1,
            relative_residual=float(np.linalg.norm(residual))
            / max(float(np.linalg.norm(vector_rhs)), 1.0e-300),
            divergence_relative_residual=float(np.linalg.norm(divergence_value))
            / divergence_scale,
            pressure_gauge_relative_residual=abs(float(self.gauge @ pressure.ravel()))
            / pressure_scale,
        )

    def storage_record(self) -> dict[str, int]:
        factor_entries = int(
            self.factor0.L.nnz
            + self.factor0.U.nnz
            + self.factor1.L.nnz
            + self.factor1.U.nnz
            + self.pressure_mass_factor.L.nnz
            + self.pressure_mass_factor.U.nnz
        )
        pressure_entries = int(self.pressure_inverse_diagonal.size)
        if self.pressure_preconditioner in {"mass_schur", "product_schur"}:
            pressure_entries += int(
                self.pressure_cholesky.size
                + self.pressure_schur_vectors.size
                + self.pressure_schur_denominator.size
            )
        return {
            "one_dimensional_factor_entries": factor_entries,
            "pressure_preconditioner_entries": pressure_entries,
            "multidimensional_dense_operator_entries": 0,
            "saddle_states": self.size,
        }


class TensorCompatibleFlow3D:
    """Three-dimensional matrix-free residual for the same exact sequence."""

    def __init__(
        self,
        npts: int,
        degree: int,
        length: float = 1.0,
        penalty_gamma: float | None = None,
    ) -> None:
        self.axis = coefficient_axis(npts, degree, length)
        self.npts = self.axis.npts
        self.degree = self.axis.degree
        self.length = self.axis.length
        self.mesh_size = self.length / (self.npts - self.degree)
        gamma = 4.0 * self.degree**2 if penalty_gamma is None else float(penalty_gamma)
        if not np.isfinite(gamma) or gamma < 4.0 * self.degree**2:
            raise ValueError("penalty_gamma must be finite and at least 4 p^2")
        self.penalty_gamma = gamma
        self.penalty = gamma / self.mesh_size
        axis = self.axis
        boundary = -axis.normal1.T @ axis.trace1 - axis.trace1.T @ axis.normal1
        boundary += self.penalty * axis.trace1.T @ axis.trace1
        self.tangential_factor = csr_matrix(axis.stiffness1 + csr_matrix(boundary))
        self.tangential_factor.eliminate_zeros()
        n = self.npts
        m = n - 1
        self.ux_shape = (n, m, m)
        self.uy_shape = (m, n, m)
        self.uz_shape = (m, m, n)
        self.pressure_shape = (m, m, m)
        elements = self.npts - self.degree
        if axis.q.size % elements != 0:
            raise RuntimeError("quadrature points do not partition into knot spans")
        points_per_span = axis.q.size // elements
        self.convection_slabs = tuple(
            slice(index * points_per_span, (index + 1) * points_per_span)
            for index in range(elements)
        )

    def _velocity(
        self,
        ux: Array,
        uy: Array,
        uz: Array,
    ) -> tuple[Array, Array, Array]:
        x = np.asarray(ux, dtype=float)
        y = np.asarray(uy, dtype=float)
        z = np.asarray(uz, dtype=float)
        if x.shape != self.ux_shape or y.shape != self.uy_shape or z.shape != self.uz_shape:
            raise ValueError(
                f"velocity shapes must be {self.ux_shape}, {self.uy_shape}, and {self.uz_shape}"
            )
        _require_finite("ux coefficients", x)
        _require_finite("uy coefficients", y)
        _require_finite("uz coefficients", z)
        return x, y, z

    def _pressure(self, pressure: Array) -> Array:
        value = np.asarray(pressure, dtype=float)
        if value.shape != self.pressure_shape:
            raise ValueError(f"pressure must have shape {self.pressure_shape}")
        _require_finite("pressure coefficients", value)
        return value

    def require_no_penetration(self, ux: Array, uy: Array, uz: Array) -> None:
        x, y, z = self._velocity(ux, uy, uz)
        trace = max(
            float(np.max(np.abs(x[[0, -1], :, :]))),
            float(np.max(np.abs(y[:, [0, -1], :]))),
            float(np.max(np.abs(z[:, :, [0, -1]]))),
        )
        if trace > 2.0e-13:
            raise ValueError("normal velocity coefficients must vanish at the walls")

    def mass_action(self, ux: Array, uy: Array, uz: Array) -> VelocityTriple:
        x, y, z = self._velocity(ux, uy, uz)
        m0 = self.axis.mass0
        m1 = self.axis.mass1
        return VelocityTriple(
            ux=_tensor_apply_sparse_3d((m0, m1, m1), x),
            uy=_tensor_apply_sparse_3d((m1, m0, m1), y),
            uz=_tensor_apply_sparse_3d((m1, m1, m0), z),
        )

    def viscous_action(self, ux: Array, uy: Array, uz: Array) -> VelocityTriple:
        x, y, z = self._velocity(ux, uy, uz)
        axis = self.axis
        t1 = self.tangential_factor
        ux_value = _tensor_apply_sparse_3d(
            (axis.stiffness0, axis.mass1, axis.mass1), x
        )
        ux_value += _tensor_apply_sparse_3d((axis.mass0, t1, axis.mass1), x)
        ux_value += _tensor_apply_sparse_3d((axis.mass0, axis.mass1, t1), x)
        uy_value = _tensor_apply_sparse_3d((t1, axis.mass0, axis.mass1), y)
        uy_value += _tensor_apply_sparse_3d(
            (axis.mass1, axis.stiffness0, axis.mass1), y
        )
        uy_value += _tensor_apply_sparse_3d((axis.mass1, axis.mass0, t1), y)
        uz_value = _tensor_apply_sparse_3d((t1, axis.mass1, axis.mass0), z)
        uz_value += _tensor_apply_sparse_3d((axis.mass1, t1, axis.mass0), z)
        uz_value += _tensor_apply_sparse_3d(
            (axis.mass1, axis.mass1, axis.stiffness0), z
        )
        return VelocityTriple(ux=ux_value, uy=uy_value, uz=uz_value)

    def divergence(self, ux: Array, uy: Array, uz: Array) -> Array:
        x, y, z = self._velocity(ux, uy, uz)
        incidence = self.axis.incidence
        identity = csr_matrix(np.eye(self.npts - 1))
        result = _tensor_apply_sparse_3d((incidence, identity, identity), x)
        result += _tensor_apply_sparse_3d((identity, incidence, identity), y)
        result += _tensor_apply_sparse_3d((identity, identity, incidence), z)
        _require_finite("three-dimensional divergence", result)
        return result

    def pressure_action(self, dual_pressure: Array) -> VelocityTriple:
        value = self._pressure(dual_pressure)
        incidence_t = self.axis.incidence.T.tocsr()
        identity = csr_matrix(np.eye(self.npts - 1))
        return VelocityTriple(
            ux=_tensor_apply_sparse_3d((incidence_t, identity, identity), value),
            uy=_tensor_apply_sparse_3d((identity, incidence_t, identity), value),
            uz=_tensor_apply_sparse_3d((identity, identity, incidence_t), value),
        )

    def convection_dual(self, ux: Array, uy: Array, uz: Array) -> VelocityTriple:
        """Return direct weak moments by element-local quadrature slabs."""
        x, y, z = self._velocity(ux, uy, uz)
        axis = self.axis
        basis = (
            (axis.b0, axis.b1, axis.b1),
            (axis.b1, axis.b0, axis.b1),
            (axis.b1, axis.b1, axis.b0),
        )
        derivative = (
            (axis.d0, axis.d1, axis.d1),
            (axis.d1, axis.d0, axis.d1),
            (axis.d1, axis.d1, axis.d0),
        )
        velocity = (x, y, z)
        moments = [np.zeros_like(field) for field in velocity]
        for slab in self.convection_slabs:
            localized = {
                id(factor): _localized_rows(factor, slab)
                for factor in (axis.b0, axis.b1, axis.d0, axis.d1)
            }

            def evaluate(factors: tuple[Sparse, ...], field: Array) -> Array:
                columns, first = localized[id(factors[0])]
                return _tensor_apply_nd((first, factors[1], factors[2]), field[columns])

            def accumulate(
                target: Array,
                factors: tuple[Sparse, ...],
                values: Array,
                scale: float,
            ) -> None:
                columns, first = localized[id(factors[0])]
                local_moments = _tensor_moments_nd(
                    (first, factors[1], factors[2]), values
                )
                target[columns] += scale * local_moments

            values = tuple(
                evaluate(basis[index], velocity[index]) for index in range(3)
            )
            weight = (
                axis.w[slab, None, None]
                * axis.w[None, :, None]
                * axis.w[None, None, :]
            )
            for component in range(3):
                advective = np.zeros_like(values[component])
                for direction in range(3):
                    trial_factors = list(basis[component])
                    trial_factors[direction] = derivative[component][direction]
                    advective += values[direction] * evaluate(
                        tuple(trial_factors), velocity[component]
                    )
                    test_factors = list(basis[component])
                    test_factors[direction] = derivative[component][direction]
                    accumulate(
                        moments[component],
                        tuple(test_factors),
                        weight * values[direction] * values[component],
                        -0.5,
                    )
                accumulate(
                    moments[component],
                    basis[component],
                    weight * advective,
                    0.5,
                )
        for index, moment in enumerate(moments):
            _require_finite(f"three-dimensional convection component {index}", moment)
        return VelocityTriple(ux=moments[0], uy=moments[1], uz=moments[2])

    def convection_workspace_record(self) -> dict[str, int]:
        """Report the largest scalar quadrature slab and global volume sizes."""
        slab_points = max(slab.stop - slab.start for slab in self.convection_slabs)
        scalar_slab_entries = slab_points * self.axis.q.size**2
        return {
            "quadrature_slabs": len(self.convection_slabs),
            "maximum_scalar_slab_entries": int(scalar_slab_entries),
            "global_scalar_volume_entries": int(self.axis.q.size**3),
        }

    def channel_residuals(
        self,
        ux: Array,
        uy: Array,
        uz: Array,
        dual_pressure: Array,
    ) -> ResidualChannels3D:
        self.require_no_penetration(ux, uy, uz)
        return ResidualChannels3D(
            mass=self.mass_action(ux, uy, uz),
            viscous=self.viscous_action(ux, uy, uz),
            pressure=self.pressure_action(dual_pressure),
            convection=self.convection_dual(ux, uy, uz),
            divergence=self.divergence(ux, uy, uz),
        )

    def convective_energy_production(self, ux: Array, uy: Array, uz: Array) -> float:
        x, y, z = self._velocity(ux, uy, uz)
        convection = self.convection_dual(x, y, z)
        return float(
            np.sum(x * convection.ux)
            + np.sum(y * convection.uy)
            + np.sum(z * convection.uz)
        )

    def storage_record(self) -> dict[str, int]:
        record = self.axis.storage_record()
        record.update(
            {
                "tangential_factor_entries": int(self.tangential_factor.nnz),
                "velocity_states": int(
                    np.prod(self.ux_shape)
                    + np.prod(self.uy_shape)
                    + np.prod(self.uz_shape)
                ),
                "pressure_states": int(np.prod(self.pressure_shape)),
            }
        )
        return record


def relative_pair_error(left: VelocityPair, right: VelocityPair) -> float:
    numerator = math.sqrt(
        float(np.sum((left.ux - right.ux) ** 2) + np.sum((left.uz - right.uz) ** 2))
    )
    denominator = math.sqrt(
        float(np.sum(right.ux**2) + np.sum(right.uz**2))
    )
    return numerator / max(denominator, 1.0e-300)
