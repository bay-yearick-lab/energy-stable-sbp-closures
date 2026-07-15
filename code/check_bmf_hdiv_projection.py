#!/usr/bin/env python3
"""Independent finite certificates for the divergence-conforming BMF core.

The direct path in this file assembles B-spline bases, mass matrices,
incidence matrices, tensor divergence, and the quotient solve from SciPy's
``BSpline`` alone.  It intentionally does not call the production spline
operator constructors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline
from scipy.linalg import block_diag, cholesky, eigh, solve, solve_triangular

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bmf_hdiv_projection import BMFHDivProjection2D


Array = np.ndarray
DEFAULT_OUTPUT = HERE / "results" / "bmf_hdiv_projection_certificate.json"

EXPECTED_FORMULATION = {
    "domain": "unit square [0,1]^2",
    "ux": "S_p(x) tensor dS_p(z)",
    "uz": "dS_p(x) tensor S_p(z)",
    "pressure": "dS_p(x) tensor dS_p(z)",
    "essential_trace": "normal velocity only",
    "velocity_metric": "natural L2 tensor metrics H0 tensor H1 and H1 tensor H0",
    "pressure_metric": "natural L2 tensor metric H1 tensor H1",
    "continuum_comparator": "Neumann pressure Laplacian on the unit square modulo constants; beta=pi",
    "gauge": "physical constants removed as the analytic all-zero tensor mode",
    "pseudoinverse_threshold_in_definition": False,
}
EXPECTED_GRID_KEYS = {
    "degree",
    "element_count",
    "npts",
    "rank",
    "expected_rank",
    "nullity",
    "constant_overlap",
    "beta",
    "beta_relative_error_to_pi",
    "production_beta_difference",
}
EXPECTED_GRID_CELLS = {(degree, elements) for degree in range(2, 9) for elements in (4, 8, 16)}
EXPECTED_ADVERSARIES = {"smooth", "checkerboard", "wall_localized", "old_nu"}
EXPECTED_LANE_KEYS = {
    "post_divergence_metric_squared",
    "initial_divergence_metric_squared",
    "postprojection_relative_divergence",
    "post_divergence_euclidean_squared",
    "initial_divergence_euclidean_squared",
    "postprojection_euclidean_relative_divergence",
    "idempotence_metric_squared",
    "input_velocity_metric_squared",
    "idempotence_relative",
    "idempotence_euclidean_squared",
    "input_velocity_euclidean_squared",
    "idempotence_euclidean_relative",
    "metric_vs_euclidean_initial_divergence_ratio",
    "metric_vs_euclidean_input_velocity_ratio",
    "normal_trace_max",
}
EXPECTED_COMPARISON_KEYS = {
    "velocity_metric_difference_squared",
    "velocity_metric_reference_squared",
    "velocity_relative_difference",
    "pressure_metric_difference_squared",
    "pressure_metric_reference_squared",
    "pressure_mod_constant_relative_difference",
}
EXPECTED_INDEPENDENT_COMPARISON_KEYS = EXPECTED_COMPARISON_KEYS | {
    "direct_post_divergence_metric_squared",
    "initial_divergence_metric_squared",
    "direct_postprojection_relative_divergence",
    "production_post_divergence_metric_squared",
    "production_postprojection_relative_divergence",
}
EXPECTED_OPERATOR_KEYS = {
    "degree",
    "npts",
    "idempotence_metric_squared",
    "projector_metric_squared",
    "idempotence_relative",
    "idempotence_euclidean_squared",
    "projector_euclidean_squared",
    "idempotence_euclidean_relative",
}
EXPECTED_TOP_KEYS = {
    "schema_version",
    "formulation",
    "grid_cells",
    "projection_adversaries",
    "dense_tensor",
    "independent_projection",
    "metric_projection_operator",
    "proof_obligations_beyond_finite_certificate",
}


def _require_finite_array(name: str, value: Array) -> Array:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"non-finite values in independent {name}")
    return array


def adversary_input_sha256(name: str, value: Array) -> str:
    """Return a platform-stable digest of a canonical adversary input."""
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    header = json.dumps(
        {
            "dtype": "float64-little-endian",
            "name": name,
            "shape": list(array.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(header + b"\n" + array.tobytes(order="C")).hexdigest()


def adversary_content_sha256(record: dict[str, object]) -> str:
    """Bind a named canonical input to both independently computed lanes."""
    payload = {
        key: record[key]
        for key in ("name", "input_sha256", "production", "independent")
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _open_uniform_knots(npts: int, degree: int, length: float = 1.0) -> Array:
    interior_count = int(npts) - int(degree) - 1
    if interior_count < 0:
        raise ValueError("npts must exceed degree")
    interior = np.linspace(0.0, length, interior_count + 2)[1:-1]
    return np.concatenate(
        [
            np.zeros(degree + 1),
            interior,
            np.full(degree + 1, length),
        ]
    )


def _greville(knots: Array, degree: int) -> Array:
    nbas = knots.size - degree - 1
    if degree == 0:
        return 0.5 * (knots[:nbas] + knots[1 : nbas + 1])
    return np.array(
        [np.mean(knots[i + 1 : i + degree + 1]) for i in range(nbas)],
        dtype=float,
    )


def _basis(knots: Array, degree: int) -> list[BSpline]:
    nbas = knots.size - degree - 1
    answer: list[BSpline] = []
    for index in range(nbas):
        coefficients = np.zeros(nbas)
        coefficients[index] = 1.0
        answer.append(BSpline(knots, coefficients, degree, extrapolate=False))
    return answer


def _evaluation(basis: list[BSpline], nodes: Array) -> Array:
    return np.nan_to_num(np.stack([function(nodes) for function in basis], axis=1))


def _mass_and_advection(knots: Array, degree: int) -> tuple[Array, Array]:
    functions = _basis(knots, degree)
    nbas = len(functions)
    mass = np.zeros((nbas, nbas))
    advection = np.zeros((nbas, nbas))
    xi, wi = leggauss(degree + 2)
    unique = np.unique(knots)
    for left, right in zip(unique[:-1], unique[1:]):
        if right <= left:
            continue
        points = 0.5 * (right - left) * xi + 0.5 * (left + right)
        weights = 0.5 * (right - left) * wi
        values = np.nan_to_num(np.stack([function(points) for function in functions]))
        derivatives = np.nan_to_num(
            np.stack([function.derivative()(points) for function in functions])
        )
        mass += (values * weights) @ values.T
        advection += (values * weights) @ derivatives.T
    return mass, advection


def _incidence(knots: Array, degree: int) -> Array:
    npts = knots.size - degree - 1
    result = np.zeros((npts - 1, npts))
    for index in range(npts - 1):
        denominator = knots[index + degree + 1] - knots[index + 1]
        scale = degree / denominator
        result[index, index] = -scale
        result[index, index + 1] = scale
    return result


@dataclass(frozen=True)
class IndependentAssembly:
    npts: int
    degree: int
    knots0: Array
    knots1: Array
    nodes0: Array
    nodes1: Array
    evaluation0: Array
    evaluation1: Array
    evaluation0_on_nodes1: Array
    mass0: Array
    mass1: Array
    advection0: Array
    delta: Array
    g0: Array
    divergence_matrix: Array
    velocity_metric: Array
    pressure_metric: Array
    schur: Array

    @property
    def derivative_size(self) -> int:
        return self.npts - 1

    @property
    def pressure_size(self) -> int:
        return self.derivative_size**2

    def divergence_rank(self) -> int:
        return int(np.linalg.matrix_rank(self.divergence_matrix))

    def kernel_certificate(self) -> dict[str, float | int]:
        values, vectors = eigh(self.schur, check_finite=False)
        scale = max(float(values[-1]), 1.0)
        nullity = int(np.count_nonzero(values <= 1.0e-10 * scale))
        dual = vectors[:, 0]
        pressure = solve(
            self.pressure_metric,
            dual,
            assume_a="pos",
            check_finite=False,
        )
        one = np.ones(self.pressure_size)
        numerator = abs(float(pressure @ self.pressure_metric @ one))
        denominator = np.sqrt(
            float(pressure @ self.pressure_metric @ pressure)
            * float(one @ self.pressure_metric @ one)
        )
        return {
            "nullity": nullity,
            "constant_overlap": numerator / denominator,
        }

    def beta(self) -> float:
        mass0_interior = self.mass0[1:-1, 1:-1]
        axis_schur = self.g0 @ solve(
            mass0_interior,
            self.g0.T,
            assume_a="pos",
            check_finite=False,
        )
        inverse_metric = solve(
            self.mass1,
            np.eye(self.derivative_size),
            assume_a="pos",
            check_finite=False,
        )
        values = eigh(
            0.5 * (axis_schur + axis_schur.T),
            inverse_metric,
            subset_by_index=[0, 1],
            eigvals_only=True,
            check_finite=False,
        )
        return float(np.sqrt(max(values[1], 0.0)))

    def _constant_dual(self) -> Array:
        return self.pressure_metric @ np.ones(self.pressure_size)

    def _augmented_schur(self) -> Array:
        constant_dual = self._constant_dual()
        result = np.block(
            [
                [self.schur, constant_dual[:, None]],
                [np.ones((1, self.pressure_size)), np.zeros((1, 1))],
            ]
        )
        return _require_finite_array("augmented Schur matrix", result)

    def project_free_vector(self, velocity: Array) -> tuple[Array, Array]:
        value = np.asarray(velocity, dtype=float)
        if value.shape != (self.velocity_metric.shape[0],):
            raise ValueError("free velocity vector has the wrong shape")
        _require_finite_array("free velocity", value)
        _require_finite_array("divergence matrix", self.divergence_matrix)
        _require_finite_array("velocity metric", self.velocity_metric)
        _require_finite_array("pressure metric", self.pressure_metric)
        rhs = _require_finite_array(
            "divergence right-hand side",
            self.divergence_matrix @ value,
        )
        augmented_rhs = np.concatenate([rhs, np.zeros(1)])
        _require_finite_array("augmented right-hand side", augmented_rhs)
        dual = _require_finite_array(
            "dual-pressure solution",
            np.linalg.solve(self._augmented_schur(), augmented_rhs)[:-1],
        )
        correction = _require_finite_array(
            "velocity correction",
            solve(
                self.velocity_metric,
                self.divergence_matrix.T @ dual,
                assume_a="pos",
                check_finite=False,
            ),
        )
        projected = _require_finite_array("projected velocity", value - correction)
        pressure = _require_finite_array(
            "pressure solution",
            solve(
                self.pressure_metric,
                dual,
                assume_a="pos",
                check_finite=False,
            ),
        )
        return projected, pressure

    def velocity_with_divergence(self, pressure_rhs: Array) -> Array:
        rhs = np.asarray(pressure_rhs, dtype=float)
        if rhs.shape != (self.pressure_size,) or not np.all(np.isfinite(rhs)):
            raise ValueError("pressure right-hand side must be a finite pressure vector")
        augmented_rhs = np.concatenate([rhs, np.zeros(1)])
        _require_finite_array("augmented right-hand side", augmented_rhs)
        _require_finite_array("velocity metric", self.velocity_metric)
        _require_finite_array("divergence matrix", self.divergence_matrix)
        dual = _require_finite_array(
            "dual-pressure solution",
            np.linalg.solve(self._augmented_schur(), augmented_rhs)[:-1],
        )
        velocity = _require_finite_array(
            "right-inverse velocity",
            solve(
                self.velocity_metric,
                self.divergence_matrix.T @ dual,
                assume_a="pos",
                check_finite=False,
            ),
        )
        return velocity

    def metric_projection_record(self) -> dict[str, float]:
        nv = self.velocity_metric.shape[0]
        right = np.vstack([self.divergence_matrix, np.zeros((1, nv))])
        dual_map = np.linalg.solve(self._augmented_schur(), right)[:-1, :]
        correction = solve(
            self.velocity_metric,
            self.divergence_matrix.T @ dual_map,
            assume_a="pos",
            check_finite=False,
        )
        projector = np.eye(nv) - correction
        factor = cholesky(self.velocity_metric, lower=True, check_finite=False)
        inverse_factor_transpose = solve_triangular(
            factor.T,
            np.eye(nv),
            lower=False,
            check_finite=False,
        )
        whitened = factor.T @ projector @ inverse_factor_transpose
        metric_numerator = float(np.linalg.norm(whitened @ whitened - whitened, 2) ** 2)
        metric_denominator = float(np.linalg.norm(whitened, 2) ** 2)
        euclidean_numerator = float(np.linalg.norm(projector @ projector - projector, 2) ** 2)
        euclidean_denominator = float(np.linalg.norm(projector, 2) ** 2)
        return {
            "degree": self.degree,
            "npts": self.npts,
            "idempotence_metric_squared": metric_numerator,
            "projector_metric_squared": metric_denominator,
            "idempotence_relative": math.sqrt(metric_numerator / metric_denominator),
            "idempotence_euclidean_squared": euclidean_numerator,
            "projector_euclidean_squared": euclidean_denominator,
            "idempotence_euclidean_relative": math.sqrt(
                euclidean_numerator / euclidean_denominator
            ),
        }

    def metric_projection_idempotence(self) -> float:
        return self.metric_projection_record()["idempotence_relative"]


def assemble_independent(npts: int, degree: int) -> IndependentAssembly:
    npts = int(npts)
    degree = int(degree)
    knots0 = _open_uniform_knots(npts, degree)
    knots1 = knots0[1:-1]
    nodes0 = _greville(knots0, degree)
    nodes1 = _greville(knots1, degree - 1)
    basis0 = _basis(knots0, degree)
    basis1 = _basis(knots1, degree - 1)
    evaluation0 = _evaluation(basis0, nodes0)
    evaluation1 = _evaluation(basis1, nodes1)
    evaluation0_on_nodes1 = _evaluation(basis0, nodes1)
    mass0, advection0 = _mass_and_advection(knots0, degree)
    mass1, _unused = _mass_and_advection(knots1, degree - 1)
    delta = _incidence(knots0, degree)
    g0 = delta[:, 1:-1]
    derivative_size = npts - 1
    identity = np.eye(derivative_size)
    divergence = np.column_stack([np.kron(g0, identity), np.kron(identity, g0)])
    mass0_interior = mass0[1:-1, 1:-1]
    velocity_metric = block_diag(
        np.kron(mass0_interior, mass1),
        np.kron(mass1, mass0_interior),
    )
    pressure_metric = np.kron(mass1, mass1)
    schur = divergence @ solve(
        velocity_metric,
        divergence.T,
        assume_a="pos",
        check_finite=False,
    )
    schur = 0.5 * (schur + schur.T)
    return IndependentAssembly(
        npts=npts,
        degree=degree,
        knots0=knots0,
        knots1=knots1,
        nodes0=nodes0,
        nodes1=nodes1,
        evaluation0=evaluation0,
        evaluation1=evaluation1,
        evaluation0_on_nodes1=evaluation0_on_nodes1,
        mass0=mass0,
        mass1=mass1,
        advection0=advection0,
        delta=delta,
        g0=g0,
        divergence_matrix=divergence,
        velocity_metric=velocity_metric,
        pressure_metric=pressure_metric,
        schur=schur,
    )


def _free_vector_from_coefficients(ux: Array, uz: Array) -> Array:
    return np.concatenate([ux[1:-1, :].ravel(), uz[:, 1:-1].ravel()])


def _coefficient_pressure_mean_zero(assembly: IndependentAssembly, pressure: Array) -> Array:
    vector = pressure.ravel()
    one = np.ones(assembly.pressure_size)
    mean = float(one @ assembly.pressure_metric @ vector) / float(
        one @ assembly.pressure_metric @ one
    )
    return pressure - mean


def compare_independent_projection(npts: int, degree: int, seed: int = 17) -> dict[str, float]:
    direct = assemble_independent(npts, degree)
    production = BMFHDivProjection2D(npts=npts, degree=degree)
    rng = np.random.default_rng(seed)
    ux_coeff = rng.standard_normal((npts, npts - 1))
    uz_coeff = rng.standard_normal((npts - 1, npts))
    ux_coeff[[0, -1], :] = 0.0
    uz_coeff[:, [0, -1]] = 0.0

    initial_free = _free_vector_from_coefficients(ux_coeff, uz_coeff)
    projected_free, pressure_direct_vector = direct.project_free_vector(initial_free)
    pressure_direct = pressure_direct_vector.reshape(npts - 1, npts - 1)

    ux_nodal = direct.evaluation0 @ ux_coeff @ direct.evaluation1.T
    uz_nodal = direct.evaluation1 @ uz_coeff @ direct.evaluation0.T
    projected = production.project(ux_nodal, uz_nodal, method="tensor")
    ux_production = solve(direct.evaluation0, projected.ux, check_finite=False)
    ux_production = solve(
        direct.evaluation1,
        ux_production.T,
        check_finite=False,
    ).T
    uz_production = solve(direct.evaluation1, projected.uz, check_finite=False)
    uz_production = solve(
        direct.evaluation0,
        uz_production.T,
        check_finite=False,
    ).T
    pressure_production = solve(
        direct.evaluation1,
        projected.pressure,
        check_finite=False,
    )
    pressure_production = solve(
        direct.evaluation1,
        pressure_production.T,
        check_finite=False,
    ).T

    production_free = _free_vector_from_coefficients(ux_production, uz_production)
    velocity_difference = production_free - projected_free
    pressure_difference = _coefficient_pressure_mean_zero(
        direct,
        pressure_production - pressure_direct,
    ).ravel()
    pressure_scale = _coefficient_pressure_mean_zero(direct, pressure_direct).ravel()
    velocity_difference_squared = float(
        velocity_difference @ direct.velocity_metric @ velocity_difference
    )
    velocity_reference_squared = float(
        projected_free @ direct.velocity_metric @ projected_free
    )
    pressure_difference_squared = float(
        pressure_difference @ direct.pressure_metric @ pressure_difference
    )
    pressure_reference_squared = float(
        pressure_scale @ direct.pressure_metric @ pressure_scale
    )
    pressure_relative = np.sqrt(pressure_difference_squared / pressure_reference_squared)
    initial_divergence = direct.divergence_matrix @ initial_free
    direct_divergence = direct.divergence_matrix @ projected_free
    production_divergence = direct.divergence_matrix @ production_free
    divergence_scale = np.sqrt(
        float(initial_divergence @ direct.pressure_metric @ initial_divergence)
    )
    initial_divergence_squared = divergence_scale**2
    direct_post_squared = float(
        direct_divergence @ direct.pressure_metric @ direct_divergence
    )
    production_post_squared = float(
        production_divergence @ direct.pressure_metric @ production_divergence
    )
    return {
        "velocity_metric_difference_squared": velocity_difference_squared,
        "velocity_metric_reference_squared": velocity_reference_squared,
        "velocity_relative_difference": float(
            np.sqrt(velocity_difference_squared / velocity_reference_squared)
        ),
        "pressure_metric_difference_squared": pressure_difference_squared,
        "pressure_metric_reference_squared": pressure_reference_squared,
        "pressure_mod_constant_relative_difference": float(pressure_relative),
        "direct_post_divergence_metric_squared": direct_post_squared,
        "initial_divergence_metric_squared": initial_divergence_squared,
        "direct_postprojection_relative_divergence": float(
            np.sqrt(direct_post_squared / initial_divergence_squared)
        ),
        "production_post_divergence_metric_squared": production_post_squared,
        "production_postprojection_relative_divergence": float(
            np.sqrt(production_post_squared / initial_divergence_squared)
        ),
    }


def _old_derivative_complement_rhs(assembly: IndependentAssembly) -> Array:
    _u, _s, vh = np.linalg.svd(assembly.advection0.T)
    old_nu_coefficients = vh[-1].copy()
    pivot = int(np.argmax(np.abs(old_nu_coefficients)))
    if old_nu_coefficients[pivot] < 0.0:
        old_nu_coefficients *= -1.0
    old_nu_values = assembly.evaluation0_on_nodes1 @ old_nu_coefficients
    smooth = np.cos(np.pi * assembly.nodes1)
    return np.outer(old_nu_values, smooth)


def _canonical_adversarial_raw_fields(
    core: BMFHDivProjection2D,
    independent: IndependentAssembly,
) -> dict[str, Array]:
    n = core.derivative_size
    coordinate = np.linspace(0.0, 1.0, n)
    checker = (-1.0) ** np.add.outer(np.arange(n), np.arange(n))
    wall = np.zeros((n, n))
    wall[0, 0] = 1.0
    return {
        "smooth": np.outer(np.cos(np.pi * coordinate), np.ones(n)),
        "checkerboard": checker,
        "wall_localized": wall,
        "old_nu": _old_derivative_complement_rhs(independent),
    }


def _lane_record(
    *,
    initial_divergence: Array,
    post_divergence: Array,
    idempotence_difference: Array,
    input_velocity: Array,
    pressure_metric: Array,
    velocity_metric: Array,
    normal_trace_max: float,
) -> dict[str, float]:
    initial_metric = float(
        initial_divergence @ pressure_metric @ initial_divergence
    )
    post_metric = float(post_divergence @ pressure_metric @ post_divergence)
    initial_euclidean = float(initial_divergence @ initial_divergence)
    post_euclidean = float(post_divergence @ post_divergence)
    input_metric = float(input_velocity @ velocity_metric @ input_velocity)
    idempotence_metric = float(
        idempotence_difference @ velocity_metric @ idempotence_difference
    )
    input_euclidean = float(input_velocity @ input_velocity)
    idempotence_euclidean = float(idempotence_difference @ idempotence_difference)
    return {
        "post_divergence_metric_squared": post_metric,
        "initial_divergence_metric_squared": initial_metric,
        "postprojection_relative_divergence": math.sqrt(post_metric / initial_metric),
        "post_divergence_euclidean_squared": post_euclidean,
        "initial_divergence_euclidean_squared": initial_euclidean,
        "postprojection_euclidean_relative_divergence": math.sqrt(
            post_euclidean / initial_euclidean
        ),
        "idempotence_metric_squared": idempotence_metric,
        "input_velocity_metric_squared": input_metric,
        "idempotence_relative": math.sqrt(idempotence_metric / input_metric),
        "idempotence_euclidean_squared": idempotence_euclidean,
        "input_velocity_euclidean_squared": input_euclidean,
        "idempotence_euclidean_relative": math.sqrt(
            idempotence_euclidean / input_euclidean
        ),
        "metric_vs_euclidean_initial_divergence_ratio": initial_metric
        / initial_euclidean,
        "metric_vs_euclidean_input_velocity_ratio": input_metric / input_euclidean,
        "normal_trace_max": float(normal_trace_max),
    }


def _adversarial_projection_records(core: BMFHDivProjection2D) -> list[dict[str, object]]:
    independent = assemble_independent(core.npts, core.degree)
    raw_fields = _canonical_adversarial_raw_fields(core, independent)
    records: list[dict[str, object]] = []
    for name, raw in raw_fields.items():
        rhs = core.remove_pressure_mean(raw)
        pressure = core.solve_pressure(rhs, method="tensor")
        ux, uz = core.adjoint_gradient(pressure)
        initial_divergence = core.divergence(ux, uz)
        first = core.project(ux, uz, method="tensor")
        second = core.project(first.ux, first.uz, method="tensor")
        post_divergence = core.divergence(first.ux, first.uz)
        input_velocity = np.concatenate([ux.ravel(), uz.ravel()])
        idempotence = np.concatenate(
            [(second.ux - first.ux).ravel(), (second.uz - first.uz).ravel()]
        )
        velocity_metric = block_diag(
            np.kron(core.h0, core.h1),
            np.kron(core.h1, core.h0),
        )
        pressure_metric = np.kron(core.h1, core.h1)
        production_record = _lane_record(
            initial_divergence=initial_divergence.ravel(),
            post_divergence=post_divergence.ravel(),
            idempotence_difference=idempotence,
            input_velocity=input_velocity,
            pressure_metric=pressure_metric,
            velocity_metric=velocity_metric,
            normal_trace_max=core.normal_trace_max(first.ux, first.uz),
        )

        raw_coefficients = solve(
            independent.evaluation1,
            raw,
            check_finite=False,
        )
        raw_coefficients = solve(
            independent.evaluation1,
            raw_coefficients.T,
            check_finite=False,
        ).T
        rhs_coefficients = _coefficient_pressure_mean_zero(
            independent,
            raw_coefficients,
        ).ravel()
        independent_velocity = independent.velocity_with_divergence(rhs_coefficients)
        independent_first, _pressure = independent.project_free_vector(
            independent_velocity
        )
        independent_second, _pressure_second = independent.project_free_vector(
            independent_first
        )
        independent_record = _lane_record(
            initial_divergence=independent.divergence_matrix @ independent_velocity,
            post_divergence=independent.divergence_matrix @ independent_first,
            idempotence_difference=independent_second - independent_first,
            input_velocity=independent_velocity,
            pressure_metric=independent.pressure_metric,
            velocity_metric=independent.velocity_metric,
            normal_trace_max=0.0,
        )
        record: dict[str, object] = {
            "name": name,
            "input_sha256": adversary_input_sha256(name, raw),
            "production": production_record,
            "independent": independent_record,
        }
        record["content_sha256"] = adversary_content_sha256(record)
        records.append(record)
    return records


def _dense_tensor_record(core: BMFHDivProjection2D, seed: int = 31) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    ux = rng.standard_normal(core.ux_shape)
    uz = rng.standard_normal(core.uz_shape)
    ux[[0, -1], :] = 0.0
    uz[:, [0, -1]] = 0.0
    tensor = core.project(ux, uz, method="tensor")
    dense = core.project(ux, uz, method="dense")
    velocity_difference_squared = core.velocity_inner(
        (tensor.ux - dense.ux, tensor.uz - dense.uz),
        (tensor.ux - dense.ux, tensor.uz - dense.uz),
    )
    velocity_reference_squared = core.velocity_inner(
        (dense.ux, dense.uz),
        (dense.ux, dense.uz),
    )
    pressure_difference = core.remove_pressure_mean(tensor.pressure - dense.pressure)
    pressure_difference_squared = core.pressure_inner(
        pressure_difference,
        pressure_difference,
    )
    pressure_reference_squared = core.pressure_inner(
        dense.pressure,
        dense.pressure,
    )
    return {
        "velocity_metric_difference_squared": velocity_difference_squared,
        "velocity_metric_reference_squared": velocity_reference_squared,
        "velocity_relative_difference": math.sqrt(
            velocity_difference_squared / velocity_reference_squared
        ),
        "pressure_metric_difference_squared": pressure_difference_squared,
        "pressure_metric_reference_squared": pressure_reference_squared,
        "pressure_mod_constant_relative_difference": math.sqrt(
            pressure_difference_squared / pressure_reference_squared
        ),
    }


def build_certificate() -> dict[str, object]:
    cells: list[dict[str, float | int]] = []
    for degree in range(2, 9):
        for element_count in (4, 8, 16):
            npts = degree + element_count
            independent = assemble_independent(npts=npts, degree=degree)
            production = BMFHDivProjection2D(npts=npts, degree=degree)
            kernel = independent.kernel_certificate()
            beta = independent.beta()
            cells.append(
                {
                    "degree": degree,
                    "element_count": element_count,
                    "npts": npts,
                    "rank": independent.divergence_rank(),
                    "expected_rank": independent.pressure_size - 1,
                    "nullity": int(kernel["nullity"]),
                    "constant_overlap": float(kernel["constant_overlap"]),
                    "beta": beta,
                    "beta_relative_error_to_pi": abs(beta / np.pi - 1.0),
                    "production_beta_difference": abs(beta - production.beta()),
                }
            )

    representative = BMFHDivProjection2D(npts=12, degree=4)
    operator_assembly = assemble_independent(npts=9, degree=4)
    certificate: dict[str, object] = {
        "schema_version": 2,
        "formulation": EXPECTED_FORMULATION.copy(),
        "grid_cells": cells,
        "projection_adversaries": _adversarial_projection_records(representative),
        "dense_tensor": _dense_tensor_record(representative),
        "independent_projection": compare_independent_projection(
            npts=10,
            degree=4,
            seed=17,
        ),
        "metric_projection_operator": operator_assembly.metric_projection_record(),
        "proof_obligations_beyond_finite_certificate": {
            "C1": "This finite rank census does not prove the all-grid range; the article proposition on the exact divergence range supplies the analytic exact-sequence proof.",
            "C2": "This finite one-null census does not prove the quotient kernel; the article proposition and tensor-reduction lemma supply that proof.",
            "C3": "This standalone projection certificate excludes tangential weak walls and marched PDE states; the separate integration certificate and physical-cavity artifact cover those gates.",
            "C4": "This finite beta sequence does not prove the limit; the article theorem supplies the antiderivative-Poincare lower bound and spline-approximation upper sequence.",
        },
    }
    return certificate


def _require_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _require_exact_keys(value: dict[str, object], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{context} schema keys mismatch; missing={missing}, extra={extra}")


def _require_nonnegative_number(value: object, context: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{context} must be a finite nonnegative scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if result < 0.0 or (positive and result <= 0.0):
        raise ValueError(f"{context} must be {'positive' if positive else 'nonnegative'}")
    return result


def _require_integer(value: object, context: str) -> int:
    result = _require_nonnegative_number(value, context)
    if not result.is_integer():
        raise ValueError(f"{context} must be an integer")
    return int(result)


def _walk_finite_nonnegative(value: object, context: str = "certificate") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _walk_finite_nonnegative(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_finite_nonnegative(child, f"{context}[{index}]")
    elif isinstance(value, bool) or isinstance(value, str) or value is None:
        return
    elif isinstance(value, (int, float, np.integer, np.floating)):
        _require_nonnegative_number(value, context)
    else:
        raise ValueError(f"{context} contains an unsupported value type")


def _require_derived_sqrt_ratio(
    record: dict[str, object],
    numerator_key: str,
    denominator_key: str,
    ratio_key: str,
    context: str,
) -> None:
    numerator = _require_nonnegative_number(record[numerator_key], f"{context}.{numerator_key}")
    denominator = _require_nonnegative_number(
        record[denominator_key],
        f"{context}.{denominator_key}",
        positive=True,
    )
    stored = _require_nonnegative_number(record[ratio_key], f"{context}.{ratio_key}")
    expected = math.sqrt(numerator / denominator)
    if not math.isclose(stored, expected, rel_tol=5.0e-13, abs_tol=1.0e-15):
        raise ValueError(f"{context} derived ratio {ratio_key} is inconsistent")


def _verify_metric_lane(record: object, context: str) -> None:
    lane = _require_mapping(record, context)
    _require_exact_keys(lane, EXPECTED_LANE_KEYS, context)
    _require_derived_sqrt_ratio(
        lane,
        "post_divergence_metric_squared",
        "initial_divergence_metric_squared",
        "postprojection_relative_divergence",
        context,
    )
    _require_derived_sqrt_ratio(
        lane,
        "post_divergence_euclidean_squared",
        "initial_divergence_euclidean_squared",
        "postprojection_euclidean_relative_divergence",
        context,
    )
    _require_derived_sqrt_ratio(
        lane,
        "idempotence_metric_squared",
        "input_velocity_metric_squared",
        "idempotence_relative",
        context,
    )
    _require_derived_sqrt_ratio(
        lane,
        "idempotence_euclidean_squared",
        "input_velocity_euclidean_squared",
        "idempotence_euclidean_relative",
        context,
    )
    for metric_key, euclidean_key, signature_key in (
        (
            "initial_divergence_metric_squared",
            "initial_divergence_euclidean_squared",
            "metric_vs_euclidean_initial_divergence_ratio",
        ),
        (
            "input_velocity_metric_squared",
            "input_velocity_euclidean_squared",
            "metric_vs_euclidean_input_velocity_ratio",
        ),
    ):
        expected = float(lane[metric_key]) / float(lane[euclidean_key])
        stored = float(lane[signature_key])
        if not math.isclose(stored, expected, rel_tol=5.0e-13, abs_tol=1.0e-15):
            raise ValueError(f"{context} derived ratio {signature_key} is inconsistent")
        if math.isclose(stored, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError(f"{context} natural metric data are indistinguishable from Euclidean data")
    if float(lane["postprojection_relative_divergence"]) >= 1.0e-12:
        raise ValueError(f"{context} postprojection divergence gate failed")
    if float(lane["normal_trace_max"]) >= 1.0e-13:
        raise ValueError(f"{context} normal trace gate failed")
    if float(lane["idempotence_relative"]) >= 1.0e-12:
        raise ValueError(f"{context} idempotence gate failed")


def _verify_comparison(record: object, context: str, expected_keys: set[str]) -> dict[str, object]:
    comparison = _require_mapping(record, context)
    _require_exact_keys(comparison, expected_keys, context)
    _require_derived_sqrt_ratio(
        comparison,
        "velocity_metric_difference_squared",
        "velocity_metric_reference_squared",
        "velocity_relative_difference",
        context,
    )
    _require_derived_sqrt_ratio(
        comparison,
        "pressure_metric_difference_squared",
        "pressure_metric_reference_squared",
        "pressure_mod_constant_relative_difference",
        context,
    )
    if float(comparison["velocity_relative_difference"]) >= 1.0e-12:
        raise ValueError(f"{context} velocity agreement gate failed")
    if float(comparison["pressure_mod_constant_relative_difference"]) >= 1.0e-11:
        raise ValueError(f"{context} pressure agreement gate failed")
    return comparison


def _first_deterministic_mismatch(
    recorded: object,
    rebuilt: object,
    context: str = "certificate",
) -> str | None:
    if isinstance(rebuilt, dict):
        if not isinstance(recorded, dict) or set(recorded) != set(rebuilt):
            return context
        for key in rebuilt:
            mismatch = _first_deterministic_mismatch(
                recorded[key],
                rebuilt[key],
                f"{context}.{key}",
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(rebuilt, list):
        if not isinstance(recorded, list) or len(recorded) != len(rebuilt):
            return context
        for index, (recorded_item, rebuilt_item) in enumerate(zip(recorded, rebuilt)):
            mismatch = _first_deterministic_mismatch(
                recorded_item,
                rebuilt_item,
                f"{context}[{index}]",
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(rebuilt, (int, float, np.integer, np.floating)) and not isinstance(
        rebuilt,
        bool,
    ):
        if isinstance(recorded, bool) or not isinstance(
            recorded,
            (int, float, np.integer, np.floating),
        ):
            return context
        return None if float(recorded) == float(rebuilt) else context
    return None if recorded == rebuilt else context


def verify_certificate(certificate: dict[str, object]) -> None:
    document = _require_mapping(certificate, "certificate")
    _walk_finite_nonnegative(document)
    _require_exact_keys(document, EXPECTED_TOP_KEYS, "certificate")
    if _require_integer(document["schema_version"], "schema_version") != 2:
        raise ValueError("unsupported certificate schema version")

    formulation = _require_mapping(document["formulation"], "formulation metadata")
    if formulation != EXPECTED_FORMULATION:
        raise ValueError("formulation metadata do not match the route-C certificate contract")

    cells = document["grid_cells"]
    if not isinstance(cells, list):
        raise ValueError("grid_cells must be a list")
    actual_grid: set[tuple[int, int]] = set()
    by_degree: dict[int, list[dict[str, object]]] = {}
    for index, raw_cell in enumerate(cells):
        cell = _require_mapping(raw_cell, f"grid cell {index}")
        _require_exact_keys(cell, EXPECTED_GRID_KEYS, f"grid cell {index}")
        degree = _require_integer(cell["degree"], f"grid cell {index}.degree")
        elements = _require_integer(
            cell["element_count"],
            f"grid cell {index}.element_count",
        )
        npts = _require_integer(cell["npts"], f"grid cell {index}.npts")
        key = (degree, elements)
        if key in actual_grid:
            raise ValueError(f"duplicate grid cell {key}")
        actual_grid.add(key)
        if npts != degree + elements:
            raise ValueError(f"grid cell {key} has inconsistent npts")
        rank = _require_integer(cell["rank"], f"grid cell {key}.rank")
        expected_rank = _require_integer(
            cell["expected_rank"],
            f"grid cell {key}.expected_rank",
        )
        if expected_rank != (npts - 1) ** 2 - 1 or rank != expected_rank:
            raise ValueError(f"grid cell {key} divergence rank gate failed")
        if _require_integer(cell["nullity"], f"grid cell {key}.nullity") != 1:
            raise ValueError(f"grid cell {key} pressure nullity gate failed")
        if float(cell["constant_overlap"]) < 1.0 - 1.0e-12:
            raise ValueError(f"grid cell {key} physical constant overlap gate failed")
        beta = float(cell["beta"])
        beta_error = float(cell["beta_relative_error_to_pi"])
        if not math.isclose(
            beta_error,
            abs(beta / math.pi - 1.0),
            rel_tol=5.0e-13,
            abs_tol=1.0e-15,
        ):
            raise ValueError(f"grid cell {key} beta derived ratio is inconsistent")
        by_degree.setdefault(degree, []).append(cell)
    if actual_grid != EXPECTED_GRID_CELLS:
        missing = sorted(EXPECTED_GRID_CELLS - actual_grid)
        extra = sorted(actual_grid - EXPECTED_GRID_CELLS)
        raise ValueError(f"exact grid set mismatch; missing={missing}, extra={extra}")

    for degree, sequence in by_degree.items():
        ordered = sorted(sequence, key=lambda item: int(item["element_count"]))
        beta_errors = [float(item["beta_relative_error_to_pi"]) for item in ordered]
        if beta_errors[-1] >= 2.0e-6:
            raise ValueError(f"degree {degree} beta-to-pi gate failed")
        if max(beta_errors) >= 1.0e-10 and beta_errors[-1] > beta_errors[0] + 1.0e-13:
            raise ValueError(f"degree {degree} beta convergence gate failed")
        if max(float(item["production_beta_difference"]) for item in ordered) >= 5.0e-10:
            raise ValueError(f"degree {degree} independent beta agreement gate failed")

    rebuilt = build_certificate()
    expected_input_hashes = {
        str(record["name"]): str(record["input_sha256"])
        for record in rebuilt["projection_adversaries"]
    }

    adversaries = document["projection_adversaries"]
    if not isinstance(adversaries, list):
        raise ValueError("projection adversaries must be a list")
    names: set[str] = set()
    for index, raw_record in enumerate(adversaries):
        record = _require_mapping(raw_record, f"adversary {index}")
        _require_exact_keys(
            record,
            {
                "name",
                "input_sha256",
                "content_sha256",
                "production",
                "independent",
            },
            f"adversary {index}",
        )
        name = record["name"]
        if not isinstance(name, str) or name in names:
            raise ValueError("adversary names must be unique strings")
        names.add(name)
        _verify_metric_lane(record["production"], f"adversary {name}.production")
        _verify_metric_lane(record["independent"], f"adversary {name}.independent")
        input_digest = record["input_sha256"]
        if (
            not isinstance(input_digest, str)
            or input_digest != expected_input_hashes.get(name)
        ):
            raise ValueError(f"adversary {name} canonical input digest is invalid")
        content_digest = record["content_sha256"]
        if (
            not isinstance(content_digest, str)
            or content_digest != adversary_content_sha256(record)
        ):
            raise ValueError(f"adversary {name} content digest is invalid")
    if names != EXPECTED_ADVERSARIES:
        raise ValueError(
            f"exact adversary set mismatch; missing={sorted(EXPECTED_ADVERSARIES - names)}, "
            f"extra={sorted(names - EXPECTED_ADVERSARIES)}"
        )

    _verify_comparison(document["dense_tensor"], "dense_tensor", EXPECTED_COMPARISON_KEYS)
    independent = _verify_comparison(
        document["independent_projection"],
        "independent_projection",
        EXPECTED_INDEPENDENT_COMPARISON_KEYS,
    )
    for numerator_key, ratio_key in (
        ("direct_post_divergence_metric_squared", "direct_postprojection_relative_divergence"),
        (
            "production_post_divergence_metric_squared",
            "production_postprojection_relative_divergence",
        ),
    ):
        _require_derived_sqrt_ratio(
            independent,
            numerator_key,
            "initial_divergence_metric_squared",
            ratio_key,
            "independent_projection",
        )
        if float(independent[ratio_key]) >= 1.0e-12:
            raise ValueError(f"independent_projection {ratio_key} gate failed")

    operator = _require_mapping(
        document["metric_projection_operator"],
        "metric_projection_operator",
    )
    _require_exact_keys(operator, EXPECTED_OPERATOR_KEYS, "metric_projection_operator")
    if _require_integer(operator["degree"], "metric_projection_operator.degree") != 4:
        raise ValueError("metric projection operator degree metadata failed")
    if _require_integer(operator["npts"], "metric_projection_operator.npts") != 9:
        raise ValueError("metric projection operator npts metadata failed")
    _require_derived_sqrt_ratio(
        operator,
        "idempotence_metric_squared",
        "projector_metric_squared",
        "idempotence_relative",
        "metric_projection_operator",
    )
    _require_derived_sqrt_ratio(
        operator,
        "idempotence_euclidean_squared",
        "projector_euclidean_squared",
        "idempotence_euclidean_relative",
        "metric_projection_operator",
    )
    if float(operator["idempotence_relative"]) >= 1.0e-12:
        raise ValueError("metric projection idempotence gate failed")

    obligations = _require_mapping(
        document["proof_obligations_beyond_finite_certificate"],
        "proof obligations",
    )
    _require_exact_keys(obligations, {"C1", "C2", "C3", "C4"}, "proof obligations")
    if any(not isinstance(value, str) or not value.strip() for value in obligations.values()):
        raise ValueError("proof obligations must be nonempty strings")

    mismatch = _first_deterministic_mismatch(document, rebuilt)
    if mismatch is not None:
        raise ValueError(f"deterministic rebuild mismatch at {mismatch}")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def load_json_strict(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(), parse_constant=_reject_json_constant)
    return _require_mapping(loaded, "certificate")


def write_json_strict(path: Path, certificate: dict[str, object]) -> None:
    try:
        payload = json.dumps(
            certificate,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except ValueError as exc:
        raise ValueError("non-finite JSON values are forbidden") from exc
    path.write_text(payload + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None)
    parser.add_argument("--verify", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify is not None:
        certificate = load_json_strict(args.verify)
        verify_certificate(certificate)
        print(f"PASS verified {args.verify}")
        return 0
    certificate = build_certificate()
    verify_certificate(certificate)
    output = args.write if args.write is not None else DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_strict(output, certificate)
    print(f"PASS wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
