#!/usr/bin/env python3
"""Divergence-conforming two-dimensional BMF pressure projection.

The tensor spaces are

    u_x : S^p x dS^p,   u_z : dS^p x S^p,   q : dS^p x dS^p.

Only the normal velocity trace is essential.  The pressure solve is posed on
the quotient by the physical constants.  Its all-constant modal coefficient is
removed by construction; no numerical pseudoinverse cutoff defines the method.
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

import numpy as np
from scipy.linalg import eigh, solve

from boundary_spline_analysis import bmf_staggered_operators


Array = np.ndarray


def _require_finite(name: str, value: Array | float) -> None:
    if not np.all(np.isfinite(np.asarray(value))):
        raise ValueError(f"non-finite values in {name}")


@dataclass(frozen=True)
class ProjectionResult:
    ux: Array
    uz: Array
    pressure: Array
    dual_pressure: Array


@dataclass(frozen=True)
class KernelCertificate:
    nullity: int
    constant_overlap: float
    smallest_positive_eigenvalue: float


class BMFHDivProjection2D:
    """Natural-metric projection onto the normal-zero exact-divergence space."""

    def __init__(self, npts: int, degree: int, length: float = 1.0):
        try:
            npts_value = float(npts)
            degree_value = float(degree)
            length_value = float(length)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("constructor scalars must be finite numbers") from exc
        if not np.all(np.isfinite([npts_value, degree_value, length_value])):
            raise ValueError("constructor scalars must be finite numbers")
        if not npts_value.is_integer() or not degree_value.is_integer():
            raise ValueError("npts and degree must be finite integers")
        self.npts = int(npts_value)
        self.degree = int(degree_value)
        self.length = length_value
        if self.length <= 0.0:
            raise ValueError("length must be finite and positive")
        if self.npts < 4:
            raise ValueError("need at least four primal basis functions")
        if self.degree < 1 or self.npts <= self.degree:
            raise ValueError("need 1 <= degree < npts")

        ops = bmf_staggered_operators(self.npts, self.degree, self.length)
        self.d10 = np.asarray(ops.d10, dtype=float).copy()
        self.h0 = np.asarray(ops.h0, dtype=float).copy()
        self.h1 = np.asarray(ops.h1, dtype=float).copy()
        _require_finite("d10", self.d10)
        _require_finite("h0", self.h0)
        _require_finite("h1", self.h1)
        self.g0 = self.d10[:, 1:-1].copy()
        self.h0_interior = self.h0[1:-1, 1:-1].copy()

        singular_values = np.linalg.svd(self.g0, compute_uv=False)
        rank_tolerance = (
            256.0
            * np.finfo(float).eps
            * max(self.g0.shape)
            * max(float(singular_values[0]), 1.0)
        )
        if (
            singular_values.size != self.g0.shape[1]
            or singular_values[-1] <= rank_tolerance
        ):
            raise ValueError(
                "endpoint-restricted derivative must have full column rank"
            )

        derivative_size = self.npts - 1
        constant_dual = self.h1 @ np.ones(derivative_size)
        constant_scale = max(
            float(np.linalg.norm(self.g0, 2) * np.linalg.norm(constant_dual)),
            np.finfo(float).tiny,
        )
        constant_residual = (
            float(np.linalg.norm(self.g0.T @ constant_dual)) / constant_scale
        )
        constant_tolerance = 512.0 * np.finfo(float).eps * max(self.g0.shape)
        if constant_residual > constant_tolerance:
            raise ValueError(
                "endpoint-restricted derivative cokernel is not the physical constant"
            )

        self.derivative_size = derivative_size
        self.ux_shape = (self.npts, self.derivative_size)
        self.uz_shape = (self.derivative_size, self.npts)
        self.pressure_shape = (self.derivative_size, self.derivative_size)
        self.pressure_size = self.derivative_size**2

        eye1 = np.eye(self.derivative_size)
        self.h1_inverse = solve(self.h1, eye1, assume_a="pos", check_finite=False)
        _require_finite("h1 inverse", self.h1_inverse)
        g_solve = solve(
            self.h0_interior,
            self.g0.T,
            assume_a="pos",
            check_finite=False,
        )
        self.axis_schur = 0.5 * (
            self.g0 @ g_solve + (self.g0 @ g_solve).T
        )
        _require_finite("axis Schur matrix", self.axis_schur)
        raw_values, vectors = eigh(
            self.axis_schur,
            self.h1_inverse,
            check_finite=False,
            driver="gvd",
        )
        self.axis_eigenvalues_raw = np.asarray(raw_values, dtype=float)
        self.axis_eigenvectors = np.asarray(vectors, dtype=float)
        _require_finite("axis eigenvalues", self.axis_eigenvalues_raw)
        _require_finite("axis eigenvectors", self.axis_eigenvectors)

        eigenvalue_scale = max(float(np.max(np.abs(self.axis_eigenvalues_raw))), 1.0)
        null_tolerance = (
            2048.0
            * np.finfo(float).eps
            * self.derivative_size
            * eigenvalue_scale
        )
        if abs(float(self.axis_eigenvalues_raw[0])) > null_tolerance:
            raise ValueError("axis Schur eigensystem has no roundoff-level raw null")
        if float(self.axis_eigenvalues_raw[1]) <= null_tolerance:
            raise ValueError("axis Schur eigensystem has no positive spectral gap")

        selected_dual = self.axis_eigenvectors[:, 0]
        selected_pressure = self.h1_inverse @ selected_dual
        one = np.ones(self.derivative_size)
        overlap_numerator = abs(float(selected_pressure @ self.h1 @ one))
        overlap_denominator = np.sqrt(
            float(selected_pressure @ self.h1 @ selected_pressure)
            * float(one @ self.h1 @ one)
        )
        constant_overlap = overlap_numerator / overlap_denominator
        if constant_overlap < 1.0 - 1.0e-10:
            raise ValueError(
                "axis Schur raw null does not reconstruct the physical constant"
            )

        self.axis_eigenvalues = self.axis_eigenvalues_raw.copy()
        # All structural checks above identify the analytic constant cokernel.
        # Only that verified mode is now set to exact zero for the quotient solve.
        self.axis_eigenvalues[0] = 0.0

    def _velocity_arrays(self, ux: Array, uz: Array) -> tuple[Array, Array]:
        x = np.asarray(ux, dtype=float)
        z = np.asarray(uz, dtype=float)
        if x.shape != self.ux_shape:
            raise ValueError(f"ux must have shape {self.ux_shape}")
        if z.shape != self.uz_shape:
            raise ValueError(f"uz must have shape {self.uz_shape}")
        _require_finite("ux", x)
        _require_finite("uz", z)
        return x, z

    def _pressure_array(self, pressure: Array) -> Array:
        value = np.asarray(pressure, dtype=float)
        if value.shape != self.pressure_shape:
            raise ValueError(f"pressure field must have shape {self.pressure_shape}")
        _require_finite("pressure field", value)
        return value

    def divergence(self, ux: Array, uz: Array) -> Array:
        x, z = self._velocity_arrays(ux, uz)
        result = self.d10 @ x + z @ self.d10.T
        _require_finite("divergence result", result)
        return result

    def normal_trace_max(self, ux: Array, uz: Array) -> float:
        x, z = self._velocity_arrays(ux, uz)
        result = float(
            max(
                np.max(np.abs(x[[0, -1], :])),
                np.max(np.abs(z[:, [0, -1]])),
            )
        )
        _require_finite("normal trace result", result)
        return result

    def pressure_inner(self, left: Array, right: Array) -> float:
        a = self._pressure_array(left)
        b = self._pressure_array(right)
        result = float(np.sum(a * (self.h1 @ b @ self.h1)))
        _require_finite("pressure inner product", result)
        return result

    def pressure_norm(self, pressure: Array) -> float:
        value = self._pressure_array(pressure)
        return float(np.sqrt(max(self.pressure_inner(value, value), 0.0)))

    def velocity_inner(
        self,
        left: tuple[Array, Array],
        right: tuple[Array, Array],
    ) -> float:
        ax, az = self._velocity_arrays(*left)
        bx, bz = self._velocity_arrays(*right)
        first = np.sum(ax * (self.h0 @ bx @ self.h1))
        second = np.sum(az * (self.h1 @ bz @ self.h0))
        result = float(first + second)
        _require_finite("velocity inner product", result)
        return result

    def velocity_norm(self, velocity: tuple[Array, Array]) -> float:
        value = self.velocity_inner(velocity, velocity)
        return float(np.sqrt(max(value, 0.0)))

    def remove_pressure_mean(self, pressure: Array) -> Array:
        value = self._pressure_array(pressure)
        one = np.ones(self.pressure_shape)
        mean = self.pressure_inner(value, one) / self.pressure_inner(one, one)
        return value - mean

    def adjoint_gradient(self, pressure: Array) -> tuple[Array, Array]:
        """Return the velocity-metric adjoint of the exact divergence."""
        p = self._pressure_array(pressure)
        gx_free = solve(
            self.h0_interior,
            self.g0.T @ self.h1 @ p,
            assume_a="pos",
            check_finite=False,
        )
        gz_free = solve(
            self.h0_interior,
            self.g0.T @ self.h1 @ p.T,
            assume_a="pos",
            check_finite=False,
        ).T
        gx = np.zeros(self.ux_shape)
        gz = np.zeros(self.uz_shape)
        gx[1:-1, :] = gx_free
        gz[:, 1:-1] = gz_free
        _require_finite("adjoint gradient x component", gx)
        _require_finite("adjoint gradient z component", gz)
        return gx, gz

    def _dual_from_pressure(self, pressure: Array) -> Array:
        p = self._pressure_array(pressure)
        result = self.h1 @ p @ self.h1
        _require_finite("dual pressure", result)
        return result

    def _pressure_from_dual(self, dual_pressure: Array) -> Array:
        dual = self._pressure_array(dual_pressure)
        left = solve(self.h1, dual, assume_a="pos", check_finite=False)
        result = solve(
            self.h1,
            left.T,
            assume_a="pos",
            check_finite=False,
        ).T
        _require_finite("reconstructed pressure", result)
        return result

    def _correction_from_dual(self, dual_pressure: Array) -> tuple[Array, Array]:
        dual = self._pressure_array(dual_pressure)
        x_left = solve(
            self.h0_interior,
            self.g0.T @ dual,
            assume_a="pos",
            check_finite=False,
        )
        x_free = solve(
            self.h1,
            x_left.T,
            assume_a="pos",
            check_finite=False,
        ).T
        z_left = solve(
            self.h1,
            dual @ self.g0,
            assume_a="pos",
            check_finite=False,
        )
        z_free = solve(
            self.h0_interior,
            z_left.T,
            assume_a="pos",
            check_finite=False,
        ).T
        x = np.zeros(self.ux_shape)
        z = np.zeros(self.uz_shape)
        x[1:-1, :] = x_free
        z[:, 1:-1] = z_free
        _require_finite("pressure correction x component", x)
        _require_finite("pressure correction z component", z)
        return x, z

    def _compatibility_residual(self, rhs: Array) -> float:
        value = self._pressure_array(rhs)
        return abs(self.pressure_inner(value, np.ones(self.pressure_shape)))

    def _require_compatible_rhs(self, rhs: Array) -> Array:
        value = self._pressure_array(rhs)
        scale = max(self.pressure_norm(value), 1.0)
        if self._compatibility_residual(value) > 2.0e-11 * scale:
            raise ValueError("pressure right-hand side must be mean-zero")
        return value

    def _solve_dual_tensor(self, rhs: Array) -> Array:
        value = self._require_compatible_rhs(rhs)
        vectors = self.axis_eigenvectors
        values = self.axis_eigenvalues
        transformed = vectors.T @ value @ vectors
        coefficients = np.zeros_like(transformed)
        coefficients[0, 1:] = transformed[0, 1:] / values[1:]
        coefficients[1:, 0] = transformed[1:, 0] / values[1:]
        denominators = values[1:, None] + values[None, 1:]
        coefficients[1:, 1:] = transformed[1:, 1:] / denominators
        # coefficients[0,0] is the explicitly removed constant gauge.
        result = vectors @ coefficients @ vectors.T
        _require_finite("tensor dual-pressure solution", result)
        return result

    def _dense_schur(self) -> Array:
        return np.kron(self.axis_schur, self.h1_inverse) + np.kron(
            self.h1_inverse,
            self.axis_schur,
        )

    def _solve_dual_dense(self, rhs: Array) -> Array:
        value = self._require_compatible_rhs(rhs)
        schur = self._dense_schur()
        one_axis_dual = self.h1 @ np.ones(self.derivative_size)
        constant_dual = np.outer(one_axis_dual, one_axis_dual).ravel()
        gauge_row = np.ones(self.pressure_size)
        augmented = np.block(
            [
                [schur, constant_dual[:, None]],
                [gauge_row[None, :], np.zeros((1, 1))],
            ]
        )
        answer = np.linalg.solve(
            augmented,
            np.concatenate([value.ravel(), np.zeros(1)]),
        )
        result = answer[:-1].reshape(self.pressure_shape)
        _require_finite("dense dual-pressure solution", result)
        return result

    def solve_pressure(self, rhs: Array, method: str = "tensor") -> Array:
        if method == "tensor":
            dual = self._solve_dual_tensor(rhs)
        elif method == "dense":
            dual = self._solve_dual_dense(rhs)
        else:
            raise ValueError("method must be 'tensor' or 'dense'")
        result = self.remove_pressure_mean(self._pressure_from_dual(dual))
        _require_finite("pressure solution", result)
        return result

    def project(self, ux: Array, uz: Array, method: str = "tensor") -> ProjectionResult:
        x, z = self._velocity_arrays(ux, uz)
        if self.normal_trace_max(x, z) > 1.0e-13:
            raise ValueError("projection requires a zero normal trace")
        rhs = self.divergence(x, z)
        pressure = self.solve_pressure(rhs, method=method)
        dual = self._dual_from_pressure(pressure)
        cx, cz = self._correction_from_dual(dual)
        result = ProjectionResult(
            ux=x - cx,
            uz=z - cz,
            pressure=pressure,
            dual_pressure=dual,
        )
        _require_finite("projected ux", result.ux)
        _require_finite("projected uz", result.uz)
        _require_finite("projected pressure", result.pressure)
        _require_finite("projected dual pressure", result.dual_pressure)
        return result

    def divergence_rank(self) -> int:
        identity = np.eye(self.derivative_size)
        bmat = np.column_stack(
            [
                np.kron(self.g0, identity),
                np.kron(identity, self.g0),
            ]
        )
        return int(np.linalg.matrix_rank(bmat))

    def kernel_certificate(self) -> KernelCertificate:
        schur = self._dense_schur()
        values, vectors = eigh(schur, check_finite=False)
        scale = max(float(values[-1]), 1.0)
        nullity = int(np.count_nonzero(values <= 1.0e-10 * scale))
        pressure = self._pressure_from_dual(vectors[:, 0].reshape(self.pressure_shape))
        one = np.ones(self.pressure_shape)
        overlap = abs(self.pressure_inner(pressure, one)) / (
            self.pressure_norm(pressure) * self.pressure_norm(one)
        )
        positive = values[nullity] if nullity < values.size else np.nan
        return KernelCertificate(
            nullity=nullity,
            constant_overlap=float(overlap),
            smallest_positive_eigenvalue=float(positive),
        )

    def beta(self) -> float:
        return float(np.sqrt(self.axis_eigenvalues[1]))
