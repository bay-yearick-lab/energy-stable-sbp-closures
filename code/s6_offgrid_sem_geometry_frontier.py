"""Optimized hp spectral-element frontier for native off-grid geometry.

The frozen matrix compares the degree-six clamped spline reconstruction with
every continuous GLL spectral-element pair ``(P, K)`` satisfying
``2 <= P <= 12`` and ``K P + 1 <= N`` at ``N in {19, 37, 61}``. All entrants
evaluate native values and first derivatives. The archived analytic fields,
physical markers, outward normals, viscosity, pressure gauge, and traction
diagnostic are shared exactly.

Outputs:
  code/results/offgrid_sem_geometry_frontier_raw.json
  code/results/offgrid_sem_geometry_frontier_summary.json
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import lu_factor, lu_solve

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))

from project_paths import RESULTSDIR, ensure_generated_dirs
from s6_curved_offgrid import NFIELDS, NU, Q, SEED, random_field, surfaces, traction
from sem_operator import gll_nodes_weights, lagrange_diff_matrix
from verify_cavity_blind import open_uniform_knots

ROOT = _CODE.parent
SPEC_PATH = ROOT / "code/data/offgrid_sem_geometry_frontier_matrix.json"
RAW_PATH = RESULTSDIR / "offgrid_sem_geometry_frontier_raw.json"
SUMMARY_PATH = RESULTSDIR / "offgrid_sem_geometry_frontier_summary.json"
TIMING_REPETITIONS = 7
FLOAT_BYTES = np.dtype(np.float64).itemsize


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _basis_matrix(n: int, degree: int, points: np.ndarray, derivative: int = 0) -> np.ndarray:
    knots = open_uniform_knots(n, degree)
    matrix = np.zeros((points.size, n))
    coefficient = np.zeros(n)
    for index in range(n):
        coefficient[:] = 0.0
        coefficient[index] = 1.0
        basis = BSpline(knots, coefficient.copy(), degree, extrapolate=False)
        if derivative:
            basis = basis.derivative(derivative)
        matrix[:, index] = np.nan_to_num(basis(points))
    return matrix


def _greville_grid(n: int, degree: int) -> np.ndarray:
    knots = open_uniform_knots(n, degree)
    return np.array([knots[index + 1:index + degree + 1].mean() for index in range(n)])


def _barycentric_rows(
    points: np.ndarray,
    degree: int,
    elements: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Global value and physical-derivative rows for native GLL evaluation."""
    reference_nodes, _ = gll_nodes_weights(degree)
    reference_derivative = lagrange_diff_matrix(reference_nodes)
    barycentric = np.ones(degree + 1)
    for index in range(degree + 1):
        barycentric[index] = 1.0 / np.prod(
            reference_nodes[index] - np.delete(reference_nodes, index)
        )
    n = elements * degree + 1
    values = np.zeros((points.size, n))
    derivatives = np.zeros((points.size, n))
    for marker, coordinate in enumerate(points):
        element = min(int(np.floor(float(coordinate) * elements)), elements - 1)
        left = element / elements
        reference = 2.0 * elements * (float(coordinate) - left) - 1.0
        distance = reference - reference_nodes
        local = slice(element * degree, element * degree + degree + 1)
        if np.any(np.abs(distance) < 1.0e-12):
            node = int(np.argmin(np.abs(distance)))
            values[marker, local.start + node] = 1.0
            local_derivative = reference_derivative[node, :].copy() * (2.0 * elements)
            local_derivative[np.argmax(np.abs(local_derivative))] -= np.sum(local_derivative)
            derivatives[marker, local] = local_derivative
            continue
        numerator = barycentric / distance
        numerator_derivative = -barycentric / distance**2
        denominator = np.sum(numerator)
        denominator_derivative = np.sum(numerator_derivative)
        values[marker, local] = numerator / denominator
        local_derivative = (
            numerator_derivative * denominator
            - numerator * denominator_derivative
        ) / denominator**2 * (2.0 * elements)
        local_derivative[np.argmax(np.abs(local_derivative))] -= np.sum(local_derivative)
        derivatives[marker, local] = local_derivative
    return values, derivatives


def _sem_grid(degree: int, elements: int) -> np.ndarray:
    reference_nodes, _ = gll_nodes_weights(degree)
    n = elements * degree + 1
    grid = np.empty(n)
    for element in range(elements):
        local = slice(element * degree, element * degree + degree + 1)
        grid[local] = (element + 0.5 * (reference_nodes + 1.0)) / elements
    return grid


@dataclass(frozen=True)
class TensorEvaluationPlan:
    x: np.ndarray
    y: np.ndarray
    value_x: np.ndarray
    derivative_x: np.ndarray
    value_y: np.ndarray
    derivative_y: np.ndarray

    @property
    def memory_bytes(self) -> int:
        return int(sum(array.nbytes for array in (
            self.value_x, self.derivative_x, self.value_y, self.derivative_y
        )))

    def evaluate(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for name in ("u", "v", "p"):
            values = state[name]
            result[name] = np.einsum(
                "mi,ij,mj->m", self.value_x, values, self.value_y, optimize=True
            )
            result[name + "x"] = np.einsum(
                "mi,ij,mj->m", self.derivative_x, values, self.value_y, optimize=True
            )
            result[name + "y"] = np.einsum(
                "mi,ij,mj->m", self.value_x, values, self.derivative_y, optimize=True
            )
        return result


class SplineReconstruction:
    """Degree-six clamped coefficient reconstruction and native evaluation."""

    def __init__(self, n: int, degree: int = Q):
        self.n = n
        self.degree = degree
        self.grid = _greville_grid(n, degree)
        interpolation = _basis_matrix(n, degree, self.grid)
        interpolation[0, :] = 0.0
        interpolation[0, 0] = 1.0
        interpolation[-1, :] = 0.0
        interpolation[-1, -1] = 1.0
        self._lu = lu_factor(interpolation)
        self.operator_memory_bytes = int(interpolation.nbytes + sum(a.nbytes for a in self._lu if hasattr(a, "nbytes")))

    def prepare(self, sampled: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        coefficients = {}
        for name in ("u", "v", "p"):
            first = lu_solve(self._lu, sampled[name])
            coefficients[name] = lu_solve(self._lu, first.T).T
        return coefficients

    def plan(self, x: np.ndarray, y: np.ndarray) -> TensorEvaluationPlan:
        return TensorEvaluationPlan(
            x=np.asarray(x, float),
            y=np.asarray(y, float),
            value_x=_basis_matrix(self.n, self.degree, np.asarray(x, float)),
            derivative_x=_basis_matrix(self.n, self.degree, np.asarray(x, float), 1),
            value_y=_basis_matrix(self.n, self.degree, np.asarray(y, float)),
            derivative_y=_basis_matrix(self.n, self.degree, np.asarray(y, float), 1),
        )


class SpectralElementReconstruction:
    """Continuous native-barycentric GLL reconstruction."""

    def __init__(self, degree: int, elements: int):
        self.degree = degree
        self.elements = elements
        self.n = degree * elements + 1
        self.grid = _sem_grid(degree, elements)
        self.operator_memory_bytes = 0

    @staticmethod
    def prepare(sampled: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return sampled

    def plan(self, x: np.ndarray, y: np.ndarray) -> TensorEvaluationPlan:
        value_x, derivative_x = _barycentric_rows(np.asarray(x, float), self.degree, self.elements)
        value_y, derivative_y = _barycentric_rows(np.asarray(y, float), self.degree, self.elements)
        return TensorEvaluationPlan(
            x=np.asarray(x, float), y=np.asarray(y, float),
            value_x=value_x, derivative_x=derivative_x,
            value_y=value_y, derivative_y=derivative_y,
        )


def _sample_state(field: dict[str, Callable], grid: np.ndarray) -> dict[str, np.ndarray]:
    x, y = np.meshgrid(grid, grid, indexing="ij")
    return {name: np.asarray(field[name](x, y), float) for name in ("u", "v", "p")}


def _score(
    field: dict[str, Callable],
    surface: dict[str, np.ndarray],
    reconstructed: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    x, y = surface["x"], surface["y"]
    nx, ny = surface["nx"], surface["ny"]
    exact_u = field["u"](x, y)
    exact_v = field["v"](x, y)
    velocity_scale = max(float(np.max(np.hypot(exact_u, exact_v))), 1.0e-30)
    velocity_error = np.hypot(
        reconstructed["u"] - exact_u,
        reconstructed["v"] - exact_v,
    ) / velocity_scale

    exact_un = field["ux"](x, y) * nx + field["uy"](x, y) * ny
    exact_vn = field["vx"](x, y) * nx + field["vy"](x, y) * ny
    normal_scale = max(float(np.max(np.hypot(exact_un, exact_vn))), 1.0e-30)
    reconstructed_un = reconstructed["ux"] * nx + reconstructed["uy"] * ny
    reconstructed_vn = reconstructed["vx"] * nx + reconstructed["vy"] * ny
    normal_error = np.hypot(
        reconstructed_un - exact_un,
        reconstructed_vn - exact_vn,
    ) / normal_scale

    exact_tx, exact_ty = traction(field, x, y, nx, ny)
    sxx = -reconstructed["p"] + 2.0 * NU * reconstructed["ux"]
    syy = -reconstructed["p"] + 2.0 * NU * reconstructed["vy"]
    sxy = NU * (reconstructed["uy"] + reconstructed["vx"])
    reconstructed_tx = sxx * nx + sxy * ny
    reconstructed_ty = sxy * nx + syy * ny
    traction_scale = max(float(np.max(np.hypot(exact_tx, exact_ty))), 1.0e-30)
    traction_error = np.hypot(
        reconstructed_tx - exact_tx,
        reconstructed_ty - exact_ty,
    ) / traction_scale
    return {
        "velocity": velocity_error,
        "normal_derivative": normal_error,
        "traction": traction_error,
    }


def _statistics(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, float)
    return {
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "worst": float(np.max(array)),
        "count": int(array.size),
    }


def _median_process_time(function: Callable[[], object], repetitions: int = TIMING_REPETITIONS) -> float:
    samples = []
    for _ in range(repetitions):
        start = time.process_time_ns()
        function()
        samples.append((time.process_time_ns() - start) * 1.0e-9)
    return float(np.median(samples))


def _field_and_surface_provenance(
    fields: list[dict[str, Callable]],
    archived_surfaces: list[dict[str, np.ndarray]],
) -> dict:
    probe = np.linspace(0.03125, 0.96875, 17)
    x, y = np.meshgrid(probe, probe, indexing="ij")
    field_hashes = []
    for field in fields:
        field_hashes.append(_array_hash(*(
            field[name](x, y)
            for name in ("u", "v", "p", "ux", "uy", "vx", "vy")
        )))
    surface_hashes = {
        surface["name"]: _array_hash(
            surface["x"], surface["y"], surface["nx"], surface["ny"]
        )
        for surface in archived_surfaces
    }
    return {"field_hashes": field_hashes, "surface_hashes": surface_hashes}


def _analytic_gates(
    fields: list[dict[str, Callable]],
    archived_surfaces: list[dict[str, np.ndarray]],
) -> dict[str, float]:
    normal_unit = 0.0
    outward_minimum = np.inf
    derivative_fd = 0.0
    divergence = 0.0
    epsilon = 1.0e-6
    centers = {
        "circle_small": (0.512, 0.487),
        "circle_large": (0.503, 0.508),
        "star": (0.501, 0.499),
    }
    for surface in archived_surfaces:
        nx, ny = surface["nx"], surface["ny"]
        normal_unit = max(normal_unit, float(np.max(np.abs(np.hypot(nx, ny) - 1.0))))
        cx, cy = centers[surface["name"]]
        radial_x = surface["x"] - cx
        radial_y = surface["y"] - cy
        radial_norm = np.hypot(radial_x, radial_y)
        outward_minimum = min(
            outward_minimum,
            float(np.min((nx * radial_x + ny * radial_y) / radial_norm)),
        )
        x, y = surface["x"], surface["y"]
        for field in fields:
            ux_fd = (field["u"](x + epsilon, y) - field["u"](x - epsilon, y)) / (2.0 * epsilon)
            uy_fd = (field["u"](x, y + epsilon) - field["u"](x, y - epsilon)) / (2.0 * epsilon)
            vx_fd = (field["v"](x + epsilon, y) - field["v"](x - epsilon, y)) / (2.0 * epsilon)
            vy_fd = (field["v"](x, y + epsilon) - field["v"](x, y - epsilon)) / (2.0 * epsilon)
            derivative_fd = max(
                derivative_fd,
                float(np.max(np.abs(ux_fd - field["ux"](x, y)))),
                float(np.max(np.abs(uy_fd - field["uy"](x, y)))),
                float(np.max(np.abs(vx_fd - field["vx"](x, y)))),
                float(np.max(np.abs(vy_fd - field["vy"](x, y)))),
            )
            divergence = max(
                divergence,
                float(np.max(np.abs(field["ux"](x, y) + field["vy"](x, y)))),
            )
    return {
        "normal_unit_max_abs": normal_unit,
        "normal_outward_dot_min": float(outward_minimum),
        "analytic_derivative_fd_max_abs": derivative_fd,
        "analytic_divergence_max_abs": divergence,
    }


def _polynomial_gate(reconstruction, archived_surfaces: list[dict[str, np.ndarray]]) -> dict[str, float]:
    degree = reconstruction.degree
    x = np.concatenate([surface["x"] for surface in archived_surfaces])
    y = np.concatenate([surface["y"] for surface in archived_surfaces])
    plan = reconstruction.plan(x, y)
    grid_x, grid_y = np.meshgrid(reconstruction.grid, reconstruction.grid, indexing="ij")
    sampled = {
        "u": grid_x**degree * grid_y**degree,
        "v": grid_x**degree,
        "p": grid_y**degree,
    }
    state = reconstruction.prepare(sampled)
    result = plan.evaluate(state)
    exact = {
        "u": x**degree * y**degree,
        "ux": degree * x ** (degree - 1) * y**degree,
        "uy": degree * x**degree * y ** (degree - 1),
        "v": x**degree,
        "vx": degree * x ** (degree - 1),
        "vy": np.zeros_like(x),
        "p": y**degree,
        "px": np.zeros_like(x),
        "py": degree * y ** (degree - 1),
    }
    value_error = max(float(np.max(np.abs(result[name] - exact[name]))) for name in ("u", "v", "p"))
    derivative_error = max(
        float(np.max(np.abs(result[name] - exact[name])))
        for name in ("ux", "uy", "vx", "vy", "px", "py")
    )
    return {"value_max_abs": value_error, "derivative_max_abs": derivative_error}


def _timing_and_memory(
    reconstruction,
    plan: TensorEvaluationPlan,
    sampled: dict[str, np.ndarray],
    operator_factory: Callable[[], object],
) -> dict[str, float | int]:
    prepared = reconstruction.prepare(sampled)
    operator_setup = _median_process_time(operator_factory)
    plan_setup = _median_process_time(lambda: reconstruction.plan(plan.x, plan.y))
    field_setup = _median_process_time(lambda: reconstruction.prepare(sampled))
    recurring = _median_process_time(lambda: plan.evaluate(prepared))
    state_bytes = int(sum(array.nbytes for array in sampled.values()))
    reconstruction_bytes = int(
        sum(array.nbytes for array in prepared.values())
        if prepared is not sampled else 0
    )
    return {
        "state_size_1d": reconstruction.n,
        "tensor_scalar_state": 3 * reconstruction.n**2,
        "sampled_state_bytes": state_bytes,
        "reconstruction_bytes": reconstruction_bytes,
        "operator_bytes": reconstruction.operator_memory_bytes,
        "evaluation_plan_bytes": plan.memory_bytes,
        "operator_setup_process_seconds_median": operator_setup,
        "evaluation_plan_setup_process_seconds_median": plan_setup,
        "field_setup_process_seconds_median": field_setup,
        "total_setup_process_seconds_median": operator_setup + plan_setup + field_setup,
        "recurring_evaluation_process_seconds_median": recurring,
        "timing_repetitions": TIMING_REPETITIONS,
    }


def _case(
    entrant: str,
    budget: int,
    reconstruction,
    field_samples: list[dict[str, np.ndarray]],
    fields: list[dict[str, Callable]],
    surface: dict[str, np.ndarray],
    operator_factory: Callable[[], object],
) -> dict:
    plan = reconstruction.plan(surface["x"], surface["y"])
    errors = {metric: [] for metric in ("velocity", "normal_derivative", "traction")}
    for field, sampled in zip(fields, field_samples, strict=True):
        reconstructed = plan.evaluate(reconstruction.prepare(sampled))
        scored = _score(field, surface, reconstructed)
        for metric, values in scored.items():
            errors[metric].extend(float(value) for value in values)
    cost = _timing_and_memory(
        reconstruction, plan, field_samples[0], operator_factory
    )
    return {
        "entrant": entrant,
        "budget": budget,
        "degree": reconstruction.degree,
        "elements": getattr(reconstruction, "elements", None),
        "state_size_1d": reconstruction.n,
        "surface": surface["name"],
        "errors": errors,
        "cost": cost,
    }


def _admissible_sem_configurations(budget: int) -> list[tuple[int, int]]:
    return [
        (degree, elements)
        for degree in range(2, 13)
        for elements in range(1, (budget - 1) // degree + 1)
    ]


def run_matrix() -> dict:
    spec_bytes = SPEC_PATH.read_bytes()
    specification = json.loads(spec_bytes)
    rng = np.random.default_rng(SEED)
    random_field(rng)
    fields = [random_field(rng) for _ in range(NFIELDS)]
    archived_surfaces = surfaces()
    provenance = _field_and_surface_provenance(fields, archived_surfaces)
    analytic_gates = _analytic_gates(fields, archived_surfaces)

    raw = {
        "protocol_id": specification["protocol_id"],
        "spec_sha256": _sha256_bytes(spec_bytes),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "process_time_clock": {
                "implementation": time.get_clock_info("process_time").implementation,
                "monotonic": time.get_clock_info("process_time").monotonic,
                "adjustable": time.get_clock_info("process_time").adjustable,
                "resolution": time.get_clock_info("process_time").resolution,
            },
            "thread_environment": {
                name: os.environ.get(name, "unset")
                for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            },
        },
        "route_identity": {
            "reported_entrant": "BMF6/BMG6 native reconstruction",
            "common_object": "degree-six clamped coefficient reconstruction with B'B^-1 same-point evaluation derivatives",
            "excluded_distinction": "The staggered BMF conservative pair and natural-metric BMG derivative are not used by static off-grid reconstruction.",
        },
        "provenance": provenance,
        "analytic_gates": analytic_gates,
        "polynomial_gates": [],
        "cases": [],
    }

    for budget in specification["state_budgets_1d"]:
        spline = SplineReconstruction(budget, Q)
        spline_samples = [_sample_state(field, spline.grid) for field in fields]
        raw["polynomial_gates"].append({
            "entrant": "spline_native_BMF6_BMG6",
            "budget": budget,
            "degree": Q,
            "elements": None,
            "state_size_1d": budget,
            **_polynomial_gate(spline, archived_surfaces),
        })
        for surface in archived_surfaces:
            raw["cases"].append(_case(
                "spline_native_BMF6_BMG6", budget, spline,
                spline_samples, fields, surface,
                lambda budget=budget: SplineReconstruction(budget, Q),
            ))

        for degree, elements in _admissible_sem_configurations(budget):
            sem = SpectralElementReconstruction(degree, elements)
            sem_samples = [_sample_state(field, sem.grid) for field in fields]
            raw["polynomial_gates"].append({
                "entrant": "sem",
                "budget": budget,
                "degree": degree,
                "elements": elements,
                "state_size_1d": sem.n,
                **_polynomial_gate(sem, archived_surfaces),
            })
            for surface in archived_surfaces:
                raw["cases"].append(_case(
                    "sem", budget, sem, sem_samples, fields, surface,
                    lambda degree=degree, elements=elements: SpectralElementReconstruction(
                        degree, elements
                    ),
                ))
            print(
                f"budget={budget} P={degree} K={elements} n={sem.n} complete",
                flush=True,
            )
    return raw


def summarize(raw: dict) -> dict:
    rows = []
    for case in raw["cases"]:
        for metric, values in case["errors"].items():
            rows.append({
                "entrant": case["entrant"],
                "budget": case["budget"],
                "degree": case["degree"],
                "elements": case["elements"],
                "state_size_1d": case["state_size_1d"],
                "surface": case["surface"],
                "metric": metric,
                **_statistics(values),
                "cost": case["cost"],
            })

    statistics = ("median", "p95", "worst")
    metrics = ("velocity", "normal_derivative", "traction")
    frontiers = []
    for budget in (19, 37, 61):
        for surface in ("circle_small", "circle_large", "star"):
            for metric in metrics:
                sem_rows = [
                    row for row in rows
                    if row["entrant"] == "sem"
                    and row["budget"] == budget
                    and row["surface"] == surface
                    and row["metric"] == metric
                ]
                spline_row = next(
                    row for row in rows
                    if row["entrant"] == "spline_native_BMF6_BMG6"
                    and row["budget"] == budget
                    and row["surface"] == surface
                    and row["metric"] == metric
                )
                frontier = {
                    "budget": budget,
                    "surface": surface,
                    "metric": metric,
                    "spline": {name: spline_row[name] for name in statistics},
                    "sem_envelope": {},
                    "sem_to_spline_error_factor": {},
                }
                for statistic in statistics:
                    best = min(sem_rows, key=lambda row: row[statistic])
                    frontier["sem_envelope"][statistic] = {
                        "error": best[statistic],
                        "degree": best["degree"],
                        "elements": best["elements"],
                        "state_size_1d": best["state_size_1d"],
                    }
                    frontier["sem_to_spline_error_factor"][statistic] = (
                        best[statistic] / spline_row[statistic]
                    )
                frontiers.append(frontier)

    traction_frontiers = [row for row in frontiers if row["metric"] == "traction"]
    dominance_cells = []
    factor_two_cells = []
    for row in traction_frontiers:
        factors = row["sem_to_spline_error_factor"]
        if all(factors[name] > 1.0 for name in statistics):
            dominance_cells.append({"budget": row["budget"], "surface": row["surface"]})
        if all(factors[name] >= 2.0 for name in statistics):
            factor_two_cells.append({"budget": row["budget"], "surface": row["surface"]})

    state_reduction_cells = []
    spline_traction_rows = [
        row for row in rows
        if row["entrant"] == "spline_native_BMF6_BMG6" and row["metric"] == "traction"
    ]
    sem_traction_rows = [
        row for row in rows if row["entrant"] == "sem" and row["metric"] == "traction"
    ]
    for spline_row in spline_traction_rows:
        admissible = [
            row for row in sem_traction_rows
            if row["surface"] == spline_row["surface"]
            and all(row[name] <= spline_row[name] for name in statistics)
        ]
        if not admissible:
            continue
        smallest_sem = min(admissible, key=lambda row: row["state_size_1d"])
        reduction = 1.0 - spline_row["state_size_1d"] / smallest_sem["state_size_1d"]
        if reduction >= 0.25:
            state_reduction_cells.append({
                "surface": spline_row["surface"],
                "spline_budget": spline_row["budget"],
                "spline_state_size_1d": spline_row["state_size_1d"],
                "sem_state_size_1d": smallest_sem["state_size_1d"],
                "sem_degree": smallest_sem["degree"],
                "sem_elements": smallest_sem["elements"],
                "state_reduction_fraction": reduction,
            })

    all_traction_cells = len(traction_frontiers)
    promotion = {
        "traction_dominance_pass": len(dominance_cells) == all_traction_cells,
        "traction_dominance_cells": dominance_cells,
        "traction_cell_count": all_traction_cells,
        "factor_two_pass": bool(factor_two_cells),
        "factor_two_cells": factor_two_cells,
        "state_reduction_pass": bool(state_reduction_cells),
        "state_reduction_cells": state_reduction_cells,
    }
    promotion["full_promotion_pass"] = bool(
        promotion["traction_dominance_pass"]
        and (promotion["factor_two_pass"] or promotion["state_reduction_pass"])
    )

    return {
        "protocol_id": raw["protocol_id"],
        "spec_sha256": raw["spec_sha256"],
        "route_identity": raw["route_identity"],
        "provenance": raw["provenance"],
        "analytic_gates": raw["analytic_gates"],
        "polynomial_gates": raw["polynomial_gates"],
        "rows": rows,
        "frontiers": frontiers,
        "promotion": promotion,
    }


def main() -> None:
    ensure_generated_dirs()
    if "--summarize" in sys.argv:
        raw = json.loads(RAW_PATH.read_text())
    else:
        raw = run_matrix()
        RAW_PATH.write_text(json.dumps(raw, separators=(",", ":")))
        print(f"wrote {RAW_PATH}")
    summary = summarize(raw)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=1))
    print(f"wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
