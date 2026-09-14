"""Symmetric hp frontier for volume fields and native wall observables.

The design stage evaluates every frozen spline and spectral-element
configuration, selects configurations without held-out access, and writes an
immutable selector record. The held-out stage evaluates only those selected
configurations on the disjoint field split.

Outputs:
  code/results/symmetric_hp_wall_frontier_design.json
  code/results/symmetric_hp_wall_frontier_selectors.json
  code/results/symmetric_hp_wall_frontier_raw.json
  code/results/symmetric_hp_wall_frontier_summary.json
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

_CODE = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(_CODE))

from project_paths import RESULTSDIR, ensure_generated_dirs
from s6_offgrid_sem_geometry_frontier import (
    SplineReconstruction,
    SpectralElementReconstruction,
)

ROOT = _CODE.parent
SPEC_PATH = ROOT / "code/data/symmetric_hp_wall_frontier_matrix.json"
DESIGN_PATH = RESULTSDIR / "symmetric_hp_wall_frontier_design.json"
SELECTOR_PATH = RESULTSDIR / "symmetric_hp_wall_frontier_selectors.json"
RAW_PATH = RESULTSDIR / "symmetric_hp_wall_frontier_raw.json"
SUMMARY_PATH = RESULTSDIR / "symmetric_hp_wall_frontier_summary.json"


Array = np.ndarray
ScalarFunction = Callable[[Array, Array], Array]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


@dataclass(frozen=True)
class FieldCase:
    """Analytic or finite-regularity velocity-pressure field."""

    name: str
    field_class: str
    parameters: tuple[float, ...]
    regularity: str
    provenance: str
    viscosity: float
    diffusivity: float
    u: ScalarFunction
    v: ScalarFunction
    p: ScalarFunction
    ux_closed: ScalarFunction | None = None
    uy_closed: ScalarFunction | None = None
    vx_closed: ScalarFunction | None = None
    vy_closed: ScalarFunction | None = None
    px_closed: ScalarFunction | None = None
    py_closed: ScalarFunction | None = None

    def evaluate(self, x: Array, y: Array) -> dict[str, Array]:
        """Return values and Cartesian first derivatives."""

        def derivative(
            function: ScalarFunction,
            closed: ScalarFunction | None,
            axis: int,
        ) -> Array:
            if closed is not None:
                return np.asarray(closed(x, y), float)
            epsilon = 1.0e-30
            if axis == 0:
                value = function(np.asarray(x, complex) + 1j * epsilon, y)
            else:
                value = function(x, np.asarray(y, complex) + 1j * epsilon)
            return np.asarray(np.imag(value) / epsilon, float)

        return {
            "u": np.asarray(self.u(x, y), float),
            "v": np.asarray(self.v(x, y), float),
            "p": np.asarray(self.p(x, y), float),
            "ux": derivative(self.u, self.ux_closed, 0),
            "uy": derivative(self.u, self.uy_closed, 1),
            "vx": derivative(self.v, self.vx_closed, 0),
            "vy": derivative(self.v, self.vy_closed, 1),
            "px": derivative(self.p, self.px_closed, 0),
            "py": derivative(self.p, self.py_closed, 1),
        }


def _analytic_global(name: str, a: float, b: float, phase: float) -> FieldCase:
    pi = np.pi
    u = lambda x, y: np.sin(a * pi * x + phase) * np.cos(b * pi * y - phase) \
        + 0.2 * np.cos((a + b) * pi * x * y + phase)
    v = lambda x, y: np.cos(b * pi * x - phase) * np.sin(a * pi * y + phase) \
        + 0.15 * np.sin(2.0 * pi * x) * np.cos(3.0 * pi * y)
    p = lambda x, y: 0.3 * np.cos(a * pi * x) * np.cos(b * pi * y) \
        + 0.1 * np.sin(pi * (x + y) + phase)
    return FieldCase(name, "analytic_global", (a, b, phase), "analytic",
                     "global trigonometric manufactured field", 0.01, 0.7,
                     u, v, p)


def _anisotropic(name: str, a: float, b: float, phase: float) -> FieldCase:
    pi = np.pi
    u = lambda x, y: np.sin(a * pi * x + phase) * np.cos(b * pi * y) \
        + 0.08 * np.cos((b + 1.0) * pi * y + phase)
    v = lambda x, y: 0.4 * np.cos(b * pi * x) * np.sin(a * pi * y + phase) \
        + 0.05 * np.sin((b + 2.0) * pi * x)
    p = lambda x, y: 0.2 * np.cos((a + 1.0) * pi * x) * np.sin(b * pi * y + phase)
    return FieldCase(name, "anisotropic", (a, b, phase), "analytic",
                     "anisotropic trigonometric manufactured field", 0.01, 0.8,
                     u, v, p)


def _wall_layer(name: str, delta: float, mode: float, phase: float) -> FieldCase:
    pi = np.pi
    u = lambda x, y: np.exp(-x / delta) * (1.0 + 0.2 * np.cos(mode * pi * y + phase)) \
        + 0.1 * np.sin(pi * x) * np.sin(2.0 * pi * y)
    v = lambda x, y: delta * np.exp(-x / delta) * np.sin(mode * pi * y + phase) \
        + 0.06 * np.cos(2.0 * pi * x) * np.sin(pi * y)
    p = lambda x, y: 0.3 * np.exp(-x / (2.0 * delta)) * np.cos((mode + 1.0) * pi * y - phase)
    return FieldCase(name, "wall_layer", (delta, mode, phase), "analytic",
                     "exponential wall-layer manufactured field", 0.015, 0.6,
                     u, v, p)


def _finite_regularity(
    name: str,
    alpha: float,
    center: float,
    mode: float,
) -> FieldCase:
    pi = np.pi

    def power(z: Array) -> Array:
        return np.abs(z) ** alpha

    def power_derivative(z: Array) -> Array:
        return alpha * np.sign(z) * np.abs(z) ** (alpha - 1.0)

    u = lambda x, y: power(x - center) * (1.0 + 0.2 * np.cos(mode * pi * y)) + 0.1 * y
    v = lambda x, y: power(y - (1.0 - center)) * (1.0 + 0.2 * np.sin(mode * pi * x))
    p = lambda x, y: 0.3 * power(x - center) * power(y - (1.0 - center))
    ux = lambda x, y: power_derivative(x - center) * (1.0 + 0.2 * np.cos(mode * pi * y))
    uy = lambda x, y: -0.2 * mode * pi * power(x - center) * np.sin(mode * pi * y) + 0.1
    vx = lambda x, y: 0.2 * mode * pi * power(y - (1.0 - center)) * np.cos(mode * pi * x)
    vy = lambda x, y: power_derivative(y - (1.0 - center)) * (1.0 + 0.2 * np.sin(mode * pi * x))
    px = lambda x, y: 0.3 * power_derivative(x - center) * power(y - (1.0 - center))
    py = lambda x, y: 0.3 * power(x - center) * power_derivative(y - (1.0 - center))
    return FieldCase(name, "finite_regularity", (alpha, center, mode),
                     f"H^s below s={alpha + 0.5:g} across an interior line",
                     "controlled fractional-power finite-regularity field",
                     0.02, 0.5, u, v, p, ux, uy, vx, vy, px, py)


def _poisson(name: str, a: float, b: float, phase: float) -> FieldCase:
    pi = np.pi
    u = lambda x, y: np.sin(a * pi * x) * np.sin(b * pi * y + phase) + 0.2 * x + 0.1 * y
    v = lambda x, y: 0.35 * np.cos(b * pi * x + phase) * np.sin(a * pi * y) + 0.05 * x * y
    p = lambda x, y: 0.2 * np.cos(a * pi * x + phase) * np.cos(b * pi * y)
    return FieldCase(name, "poisson", (a, b, phase), "analytic",
                     "exact Poisson solution with analytic source and boundary data",
                     0.01, 0.9, u, v, p)


def _heat(name: str, a: float, b: float, time_value: float) -> FieldCase:
    pi = np.pi
    kappa = 0.015
    decay = np.exp(-kappa * pi**2 * (a**2 + b**2) * time_value)
    decay_second = np.exp(-kappa * pi**2 * ((a + 1.0)**2 + (b + 1.0)**2) * time_value)
    u = lambda x, y: decay * np.sin(a * pi * x) * np.sin(b * pi * y)
    v = lambda x, y: 0.3 * decay_second * np.sin((a + 1.0) * pi * x) * np.sin((b + 1.0) * pi * y)
    p = lambda x, y: 0.0 * x + 0.0 * y
    return FieldCase(name, "heat", (a, b, time_value), "analytic",
                     "homogeneous heat-equation snapshot", 0.01, kappa, u, v, p)


def _stokes(name: str, a: float, b: float, phase: float) -> FieldCase:
    pi = np.pi
    u = lambda x, y: 2.0 * b * pi * np.sin(a * pi * x) ** 2 \
        * np.sin(b * pi * y) * np.cos(b * pi * y)
    v = lambda x, y: -2.0 * a * pi * np.sin(a * pi * x) * np.cos(a * pi * x) \
        * np.sin(b * pi * y) ** 2
    p = lambda x, y: np.cos(pi * x + phase) * np.sin(pi * y - phase)
    return FieldCase(name, "stokes", (a, b, phase), "analytic divergence-free",
                     "divergence-free Stokes manufactured solution with analytic forcing",
                     0.025, 0.7, u, v, p)


def _analytic_wake(name: str, reynolds: float, phase: float) -> FieldCase:
    pi = np.pi
    lam = reynolds / 2.0 - np.sqrt(reynolds**2 / 4.0 + 4.0 * pi**2)
    y_shift = lambda y: y + phase / (2.0 * pi)
    u = lambda x, y: 1.0 - np.exp(lam * x) * np.cos(2.0 * pi * y_shift(y))
    v = lambda x, y: lam / (2.0 * pi) * np.exp(lam * x) * np.sin(2.0 * pi * y_shift(y))
    p = lambda x, y: 0.5 * (1.0 - np.exp(2.0 * lam * x))
    return FieldCase(name, "analytic_wake", (reynolds, phase), "analytic divergence-free",
                     "exact steady analytic Navier-Stokes wake snapshot",
                     1.0 / reynolds, 0.6, u, v, p)


FIELD_BUILDERS = {
    "analytic_global": _analytic_global,
    "anisotropic": _anisotropic,
    "wall_layer": _wall_layer,
    "finite_regularity": _finite_regularity,
    "poisson": _poisson,
    "heat": _heat,
    "stokes": _stokes,
    "analytic_wake": _analytic_wake,
}


def build_fields(specification: dict, split: str) -> list[FieldCase]:
    """Build the frozen design or held-out field list."""
    parameters = specification["split"][f"{split}_parameters"]
    fields = []
    for field_class in specification["split"]["classes"]:
        builder = FIELD_BUILDERS[field_class]
        for index, values in enumerate(parameters[field_class]):
            fields.append(builder(f"{split}_{field_class}_{index}", *values))
    return fields


def _gauss_interval(count: int) -> tuple[Array, Array]:
    nodes, weights = np.polynomial.legendre.leggauss(count)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _wall_points(count: int) -> tuple[Array, Array, Array, Array, Array]:
    s, weights = _gauss_interval(count)
    x = np.concatenate([np.zeros(count), np.ones(count), s, s])
    y = np.concatenate([s, s, np.zeros(count), np.ones(count)])
    nx = np.concatenate([-np.ones(count), np.ones(count), np.zeros(count), np.zeros(count)])
    ny = np.concatenate([np.zeros(count), np.zeros(count), -np.ones(count), np.ones(count)])
    return x, y, nx, ny, np.tile(weights, 4)


def _path_points(count: int) -> tuple[Array, Array, list[str]]:
    t = np.linspace(0.0, 1.0, count)
    oblique = (0.08 + 0.84 * t, 0.16 + 0.68 * t)
    angle = np.pi * (0.15 + 0.70 * t)
    arc = (0.5 + 0.32 * np.cos(angle), 0.5 + 0.32 * np.sin(angle))
    moving = (0.08 + 0.84 * t, 0.5 + 0.22 * np.sin(2.0 * np.pi * t + 0.31))
    half = count // 2
    normal = (np.linspace(0.0, 0.35, half), np.full(half, 0.23))
    tangent = (np.zeros(count - half), np.linspace(0.05, 0.95, count - half))
    cuts = (np.concatenate([normal[0], tangent[0]]), np.concatenate([normal[1], tangent[1]]))
    pairs = (oblique, arc, moving, cuts)
    names = ["oblique_line", "smooth_arc", "moving_probe_curve", "wall_normal_and_tangential_cuts"]
    return np.concatenate([pair[0] for pair in pairs]), np.concatenate([pair[1] for pair in pairs]), names


def _stack_exact(fields: list[FieldCase], x: Array, y: Array) -> dict[str, Array]:
    values = [field.evaluate(x, y) for field in fields]
    return {name: np.stack([value[name] for value in values]) for name in values[0]}


def sampling_record(specification: dict, fields: list[FieldCase]) -> dict:
    """Build exact values on the frozen volume, wall, and path points."""
    volume_count = specification["sampling"]["volume_gauss_points_per_direction"]
    wall_count = specification["sampling"]["wall_gauss_points_per_edge"]
    path_count = specification["sampling"]["path_points"]
    volume_nodes, volume_weights = _gauss_interval(volume_count)
    xv, yv = np.meshgrid(volume_nodes, volume_nodes, indexing="ij")
    xw, yw, nx, ny, wall_weights = _wall_points(wall_count)
    xp, yp, path_names = _path_points(path_count)
    return {
        "volume_nodes": volume_nodes,
        "volume_weights": volume_weights,
        "wall_x": xw,
        "wall_y": yw,
        "wall_nx": nx,
        "wall_ny": ny,
        "wall_weights": wall_weights,
        "path_x": xp,
        "path_y": yp,
        "path_names": path_names,
        "volume": _stack_exact(fields, xv, yv),
        "wall": _stack_exact(fields, xw, yw),
        "path": _stack_exact(fields, xp, yp),
    }


def configurations(specification: dict) -> list[dict]:
    """Enumerate every frozen degree and element-count pair."""
    maximum = max(specification["state_caps_1d"])
    rows = []
    for degree in specification["degrees"]:
        for spans in range(1, maximum - degree + 1):
            n = degree + spans
            rows.append({
                "key": f"spline:p{degree}:k{spans}", "family": "spline",
                "degree": degree, "elements": spans, "n": n,
            })
        for elements in range(1, (maximum - 1) // degree + 1):
            n = degree * elements + 1
            rows.append({
                "key": f"sem:p{degree}:e{elements}", "family": "sem",
                "degree": degree, "elements": elements, "n": n,
            })
    return rows


def _reconstruction(config: dict):
    if config["family"] == "spline":
        return SplineReconstruction(config["n"], config["degree"])
    return SpectralElementReconstruction(config["degree"], config["elements"])


def _axis_rows(reconstruction, points: Array) -> tuple[Array, Array]:
    plan = reconstruction.plan(np.asarray(points), np.asarray(points))
    return plan.value_x, plan.derivative_x


def _prepared_state(reconstruction, fields: list[FieldCase]) -> Array:
    grid = reconstruction.grid
    x, y = np.meshgrid(grid, grid, indexing="ij")
    sampled = np.stack([
        np.stack([field.u(x, y), field.v(x, y), field.p(x, y)])
        for field in fields
    ]).astype(float)
    if isinstance(reconstruction, SpectralElementReconstruction):
        return sampled
    prepared = np.empty_like(sampled)
    for field_index in range(len(fields)):
        state = reconstruction.prepare({
            "u": sampled[field_index, 0],
            "v": sampled[field_index, 1],
            "p": sampled[field_index, 2],
        })
        for component, name in enumerate(("u", "v", "p")):
            prepared[field_index, component] = state[name]
    return prepared


def _grid_evaluate(state: Array, value: Array, derivative: Array) -> dict[str, Array]:
    return {
        "value": np.einsum("ai,fcij,bj->fcab", value, state, value, optimize=True),
        "dx": np.einsum("ai,fcij,bj->fcab", derivative, state, value, optimize=True),
        "dy": np.einsum("ai,fcij,bj->fcab", value, state, derivative, optimize=True),
    }


def _paired_evaluate(
    state: Array,
    value_x: Array,
    derivative_x: Array,
    value_y: Array,
    derivative_y: Array,
) -> dict[str, Array]:
    return {
        "value": np.einsum("mi,fcij,mj->fcm", value_x, state, value_y, optimize=True),
        "dx": np.einsum("mi,fcij,mj->fcm", derivative_x, state, value_y, optimize=True),
        "dy": np.einsum("mi,fcij,mj->fcm", value_x, state, derivative_y, optimize=True),
    }


def _relative_l2(error_squared: Array, exact_squared: Array, weights: Array, floor: float) -> float:
    numerator = float(np.sqrt(np.sum(weights * error_squared)))
    denominator = max(float(np.sqrt(np.sum(weights * exact_squared))), floor)
    return numerator / denominator


def _traction(values: dict[str, Array], viscosity: float, nx: Array, ny: Array) -> tuple[Array, Array]:
    pressure = values["value"][2]
    ux, uy = values["dx"][0], values["dy"][0]
    vx, vy = values["dx"][1], values["dy"][1]
    sxx = -pressure + 2.0 * viscosity * ux
    syy = -pressure + 2.0 * viscosity * vy
    sxy = viscosity * (uy + vx)
    return sxx * nx + sxy * ny, sxy * nx + syy * ny


def _exact_triplet(exact: dict[str, Array], index: int) -> dict[str, Array]:
    return {
        "value": np.stack([exact["u"][index], exact["v"][index], exact["p"][index]]),
        "dx": np.stack([exact["ux"][index], exact["vx"][index], exact["px"][index]]),
        "dy": np.stack([exact["uy"][index], exact["vy"][index], exact["py"][index]]),
    }


def _field_metrics(
    field: FieldCase,
    field_index: int,
    samples: dict,
    approximate: dict[str, dict[str, Array]],
) -> dict[str, float]:
    exact_volume = _exact_triplet(samples["volume"], field_index)
    exact_wall = _exact_triplet(samples["wall"], field_index)
    exact_path = _exact_triplet(samples["path"], field_index)
    approx_volume = {name: value[field_index] for name, value in approximate["volume"].items()}
    approx_wall = {name: value[field_index] for name, value in approximate["wall"].items()}
    approx_path = {name: value[field_index] for name, value in approximate["path"].items()}

    volume_weight = np.outer(samples["volume_weights"], samples["volume_weights"])
    value_exact_squared = np.sum(exact_volume["value"] ** 2, axis=0)
    value_error_squared = np.sum((approx_volume["value"] - exact_volume["value"]) ** 2, axis=0)
    field_reference = max(float(np.sqrt(np.sum(volume_weight * value_exact_squared))), 1.0e-14)
    volume_l2 = _relative_l2(value_error_squared, value_exact_squared, volume_weight, 0.1 * field_reference)

    h1_exact_squared = value_exact_squared + np.sum(exact_volume["dx"] ** 2 + exact_volume["dy"] ** 2, axis=0)
    h1_error_squared = value_error_squared + np.sum(
        (approx_volume["dx"] - exact_volume["dx"]) ** 2
        + (approx_volume["dy"] - exact_volume["dy"]) ** 2,
        axis=0,
    )
    h1 = _relative_l2(h1_error_squared, h1_exact_squared, volume_weight, 0.1 * field_reference)

    wall_weight = samples["wall_weights"]
    wall_value_exact_squared = np.sum(exact_wall["value"][:2] ** 2, axis=0)
    wall_value_error_squared = np.sum((approx_wall["value"][:2] - exact_wall["value"][:2]) ** 2, axis=0)
    wall_trace_l2 = _relative_l2(
        wall_value_error_squared, wall_value_exact_squared, wall_weight, 0.1 * field_reference,
    )

    nx, ny = samples["wall_nx"], samples["wall_ny"]
    normal_exact = exact_wall["dx"][:2] * nx + exact_wall["dy"][:2] * ny
    normal_approx = approx_wall["dx"][:2] * nx + approx_wall["dy"][:2] * ny
    normal_error = normal_approx - normal_exact
    normal_scale = max(float(np.sqrt(np.sum(wall_weight * np.sum(normal_exact**2, axis=0)))), 0.1 * field_reference)
    normal_derivative_l2 = float(np.sqrt(np.sum(wall_weight * np.sum(normal_error**2, axis=0)))) / normal_scale

    flux_exact = -field.diffusivity * normal_exact[0]
    flux_approx = -field.diffusivity * normal_approx[0]
    flux_scale = max(float(np.sqrt(np.sum(wall_weight * flux_exact**2))), 0.1 * field.diffusivity * field_reference)
    flux_error = flux_approx - flux_exact
    flux_l2 = float(np.sqrt(np.sum(wall_weight * flux_error**2))) / flux_scale

    exact_tx, exact_ty = _traction(exact_wall, field.viscosity, nx, ny)
    approx_tx, approx_ty = _traction(approx_wall, field.viscosity, nx, ny)
    traction_error = np.hypot(approx_tx - exact_tx, approx_ty - exact_ty)
    traction_exact = np.hypot(exact_tx, exact_ty)
    traction_scale = max(float(np.sqrt(np.sum(wall_weight * traction_exact**2))), 0.1 * field_reference)
    traction_l2 = float(np.sqrt(np.sum(wall_weight * traction_error**2))) / traction_scale

    field_error_points = np.concatenate([
        np.sqrt(np.sum((approx_wall["value"] - exact_wall["value"]) ** 2, axis=0)),
        np.sqrt(np.sum((approx_path["value"] - exact_path["value"]) ** 2, axis=0)),
    ])
    field_exact_points = np.concatenate([
        np.sqrt(np.sum(exact_wall["value"] ** 2, axis=0)),
        np.sqrt(np.sum(exact_path["value"] ** 2, axis=0)),
    ])
    field_scale = max(float(np.max(field_exact_points)), 0.1 * field_reference)
    gradient_error_wall = np.sqrt(np.sum(
        (approx_wall["dx"] - exact_wall["dx"]) ** 2
        + (approx_wall["dy"] - exact_wall["dy"]) ** 2,
        axis=0,
    ))
    gradient_error_path = np.sqrt(np.sum(
        (approx_path["dx"] - exact_path["dx"]) ** 2
        + (approx_path["dy"] - exact_path["dy"]) ** 2,
        axis=0,
    ))
    gradient_exact_points = np.concatenate([
        np.sqrt(np.sum(exact_wall["dx"] ** 2 + exact_wall["dy"] ** 2, axis=0)),
        np.sqrt(np.sum(exact_path["dx"] ** 2 + exact_path["dy"] ** 2, axis=0)),
    ])
    gradient_error_points = np.concatenate([gradient_error_wall, gradient_error_path])
    gradient_scale = max(float(np.max(gradient_exact_points)), 0.1 * field_reference)
    normal_point_scale = max(float(np.max(np.sqrt(np.sum(normal_exact**2, axis=0)))), 0.1 * field_reference)
    flux_point_scale = max(float(np.max(np.abs(flux_exact))), 0.1 * field.diffusivity * field_reference)
    traction_point_scale = max(float(np.max(traction_exact)), 0.1 * field_reference)

    path_field_error = np.sqrt(np.sum((approx_path["value"] - exact_path["value"]) ** 2, axis=0))
    path_gradient_error = gradient_error_path
    return {
        "volume_l2": volume_l2,
        "h1": h1,
        "wall_trace_l2": wall_trace_l2,
        "normal_derivative_l2": normal_derivative_l2,
        "flux_l2": flux_l2,
        "traction_l2": traction_l2,
        "field_max": float(np.max(field_error_points)) / field_scale,
        "field_p95": float(np.quantile(field_error_points, 0.95)) / field_scale,
        "gradient_max": float(np.max(gradient_error_points)) / gradient_scale,
        "gradient_p95": float(np.quantile(gradient_error_points, 0.95)) / gradient_scale,
        "normal_derivative_max": float(np.max(np.sqrt(np.sum(normal_error**2, axis=0)))) / normal_point_scale,
        "normal_derivative_p95": float(np.quantile(np.sqrt(np.sum(normal_error**2, axis=0)), 0.95)) / normal_point_scale,
        "flux_max": float(np.max(np.abs(flux_error))) / flux_point_scale,
        "flux_p95": float(np.quantile(np.abs(flux_error), 0.95)) / flux_point_scale,
        "traction_max": float(np.max(traction_error)) / traction_point_scale,
        "traction_p95": float(np.quantile(traction_error, 0.95)) / traction_point_scale,
        "path_field_max": float(np.max(path_field_error)) / field_scale,
        "path_gradient_max": float(np.max(path_gradient_error)) / gradient_scale,
    }


def configuration_cost(config: dict, specification: dict) -> dict[str, int]:
    """Return deterministic storage, setup, line-work, and query counts."""
    degree, elements, n = config["degree"], config["elements"], config["n"]
    markers = specification["work"]["query_markers_per_snapshot"]
    support = degree + 1
    state_scalars = 3 * n * n
    if config["family"] == "spline":
        operator_nonzeros = 4 * (2 * degree + 1) * n
        factor_nonzeros = (2 * degree + 1) * n
        factorization_work = 2 * n * degree * degree
        state_prepare_work = 3 * 8 * degree * n * n
        line_work = n * (6 * degree + 3)
    else:
        operator_nonzeros = support * support + 3 * n
        factor_nonzeros = 0
        factorization_work = support**3
        state_prepare_work = 0
        line_work = elements * (2 * support * support - support)
    plan_nonzeros = 4 * markers * support
    plan_setup_work = 8 * markers * support
    point_query_work = 3 * 3 * markers * (2 * support * support - 1)
    operator_step_work = 12 * n * line_work
    setup_work = factorization_work + plan_setup_work
    observation_work = state_prepare_work + point_query_work
    storage_bytes = 8 * (state_scalars + operator_nonzeros + factor_nonzeros + plan_nonzeros)
    return {
        "state_1d": n,
        "scalar_state": state_scalars,
        "state_bytes": 8 * state_scalars,
        "operator_nonzeros": operator_nonzeros,
        "factor_nonzeros": factor_nonzeros,
        "evaluation_plan_nonzeros": plan_nonzeros,
        "persistent_storage_bytes": storage_bytes,
        "setup_work": setup_work,
        "state_prepare_work": state_prepare_work,
        "marginal_query_work": observation_work,
        "operator_step_work": operator_step_work,
    }


def _polynomial_gate(reconstruction, degree: int) -> dict[str, float]:
    points = np.linspace(0.013, 0.987, 41)
    value, derivative = _axis_rows(reconstruction, points)
    sampled = reconstruction.grid**degree
    if isinstance(reconstruction, SplineReconstruction):
        prepared = reconstruction.prepare({"u": sampled[:, None] * np.ones((1, reconstruction.n)),
                                           "v": np.zeros((reconstruction.n, reconstruction.n)),
                                           "p": np.zeros((reconstruction.n, reconstruction.n))})["u"][:, 0]
    else:
        prepared = sampled
    exact_value = points**degree
    exact_derivative = degree * points ** (degree - 1)
    return {
        "value": float(np.max(np.abs(value @ prepared - exact_value))),
        "derivative": float(np.max(np.abs(derivative @ prepared - exact_derivative))),
        "partition": float(np.max(np.abs(np.sum(value, axis=1) - 1.0))),
        "linear_derivative": float(np.max(np.abs(derivative @ reconstruction.grid - 1.0))),
    }


def evaluate_configuration(
    config: dict,
    fields: list[FieldCase],
    samples: dict,
    specification: dict,
) -> tuple[list[dict], dict]:
    """Evaluate one configuration on one complete field split."""
    reconstruction = _reconstruction(config)
    state = _prepared_state(reconstruction, fields)
    volume_value, volume_derivative = _axis_rows(reconstruction, samples["volume_nodes"])
    wall_value_x, wall_derivative_x = _axis_rows(reconstruction, samples["wall_x"])
    wall_value_y, wall_derivative_y = _axis_rows(reconstruction, samples["wall_y"])
    path_value_x, path_derivative_x = _axis_rows(reconstruction, samples["path_x"])
    path_value_y, path_derivative_y = _axis_rows(reconstruction, samples["path_y"])
    approximate = {
        "volume": _grid_evaluate(state, volume_value, volume_derivative),
        "wall": _paired_evaluate(
            state, wall_value_x, wall_derivative_x, wall_value_y, wall_derivative_y,
        ),
        "path": _paired_evaluate(
            state, path_value_x, path_derivative_x, path_value_y, path_derivative_y,
        ),
    }
    rows = []
    for index, field in enumerate(fields):
        rows.append({
            "config": config["key"],
            "case": field.name,
            "field_class": field.field_class,
            "parameters": list(field.parameters),
            "regularity": field.regularity,
            "provenance": field.provenance,
            "metrics": _field_metrics(field, index, samples, approximate),
        })
    gate = _polynomial_gate(reconstruction, config["degree"])
    return rows, gate


def _derivative_gate(fields: list[FieldCase]) -> float:
    points = np.linspace(0.071, 0.929, 13)
    x, y = np.meshgrid(points, points[::-1], indexing="ij")
    epsilon = 2.0e-6
    worst = 0.0
    for field in fields:
        exact = field.evaluate(x, y)
        for name, function in (("u", field.u), ("v", field.v), ("p", field.p)):
            x_fd = (function(x + epsilon, y) - function(x - epsilon, y)) / (2.0 * epsilon)
            y_fd = (function(x, y + epsilon) - function(x, y - epsilon)) / (2.0 * epsilon)
            worst = max(
                worst,
                float(np.max(np.abs(x_fd - exact[name + "x"]))),
                float(np.max(np.abs(y_fd - exact[name + "y"]))),
            )
    return worst


def _design_scores(design_rows: list[dict], specification: dict) -> dict[str, dict[str, float]]:
    normalizers = specification["selector"]["normalizers"]
    objectives = specification["selector"]["objectives"]
    grouped: dict[str, list[dict]] = {}
    for row in design_rows:
        grouped.setdefault(row["config"], []).append(row)
    scores = {}
    for config, rows in grouped.items():
        scores[config] = {}
        for objective, metrics in objectives.items():
            scores[config][objective] = max(
                row["metrics"][metric] / normalizers[metric]
                for row in rows for metric in metrics
            )
    return scores


def _resource_entries(config: dict, cost: dict, specification: dict) -> list[dict]:
    entries = [
        {"kind": "state", "cap": cap, "query_count": None, "value": config["n"]}
        for cap in specification["state_caps_1d"]
    ]
    entries.extend(
        {"kind": "operator_work", "cap": cap, "query_count": None,
         "value": cost["operator_step_work"]}
        for cap in specification["work"]["operator_work_caps"]
    )
    steps = specification["work"]["evolution_steps"]
    for queries in specification["work"]["query_counts"]:
        mission = cost["setup_work"] + steps * cost["operator_step_work"] \
            + queries * cost["marginal_query_work"]
        entries.extend(
            {"kind": "mission", "cap": cap, "query_count": queries, "value": mission}
            for cap in specification["work"]["mission_work_caps"]
        )
    return entries


def select_configurations(
    configs: list[dict],
    costs: dict[str, dict],
    scores: dict[str, dict[str, float]],
    specification: dict,
) -> list[dict]:
    """Apply the frozen lexicographic design selector."""
    candidates = []
    for config in configs:
        for resource in _resource_entries(config, costs[config["key"]], specification):
            if resource["value"] <= resource["cap"]:
                candidates.append((config, resource))
    selectors = []
    for objective in specification["selector"]["objectives"]:
        for family in ("spline", "sem"):
            resources = sorted({
                (resource["kind"], resource["cap"], resource["query_count"])
                for config, resource in candidates if config["family"] == family
            }, key=lambda item: (item[0], item[2] or 0, item[1]))
            for kind, cap, query_count in resources:
                admissible = [
                    (config, resource) for config, resource in candidates
                    if config["family"] == family
                    and resource["kind"] == kind
                    and resource["cap"] == cap
                    and resource["query_count"] == query_count
                ]
                if not admissible:
                    continue
                config, resource = min(
                    admissible,
                    key=lambda item: (
                        scores[item[0]["key"]][objective], item[1]["value"],
                        costs[item[0]["key"]]["scalar_state"], item[0]["degree"],
                        item[0]["elements"],
                    ),
                )
                selectors.append({
                    "objective": objective,
                    "family": family,
                    "resource_kind": kind,
                    "resource_cap": cap,
                    "query_count": query_count,
                    "resource_value": resource["value"],
                    "design_score": scores[config["key"]][objective],
                    "config": config,
                    "cost": costs[config["key"]],
                })
    return selectors


def run_design() -> tuple[dict, dict]:
    """Evaluate the complete design matrix and freeze selectors."""
    spec_bytes = SPEC_PATH.read_bytes()
    specification = json.loads(spec_bytes)
    fields = build_fields(specification, "design")
    samples = sampling_record(specification, fields)
    configs = configurations(specification)
    design_rows = []
    gates = []
    costs = {}
    for index, config in enumerate(configs, 1):
        rows, gate = evaluate_configuration(config, fields, samples, specification)
        design_rows.extend(rows)
        gates.append({"config": config["key"], **gate})
        costs[config["key"]] = configuration_cost(config, specification)
        if index % 25 == 0 or index == len(configs):
            print(f"design configurations {index}/{len(configs)}", flush=True)
    scores = _design_scores(design_rows, specification)
    design = {
        "protocol_id": specification["protocol_id"],
        "spec_sha256": _sha256_bytes(spec_bytes),
        "split": "design",
        "field_derivative_fd_max_abs": _derivative_gate(fields),
        "configurations": configs,
        "costs": costs,
        "polynomial_gates": gates,
        "rows": design_rows,
        "scores": scores,
    }
    DESIGN_PATH.write_text(json.dumps(design, separators=(",", ":")))
    selectors = {
        "protocol_id": specification["protocol_id"],
        "spec_sha256": design["spec_sha256"],
        "design_sha256": _sha256_path(DESIGN_PATH),
        "selectors": select_configurations(configs, costs, scores, specification),
    }
    SELECTOR_PATH.write_text(json.dumps(selectors, indent=1))
    return design, selectors


def _aggregate(rows: list[dict], metrics: list[str]) -> dict[str, dict[str, float | int]]:
    out = {}
    for metric in metrics:
        values = np.asarray([row["metrics"][metric] for row in rows], float)
        out[metric] = {
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "worst": float(np.max(values)),
            "count": int(values.size),
        }
    return out


def summarize(raw: dict, selectors: dict, specification: dict) -> dict:
    """Aggregate held-out rows at each frozen selector cell."""
    all_metrics = specification["metrics"]["primary"] \
        + specification["metrics"]["tail"] + specification["metrics"]["path"]
    by_config: dict[str, list[dict]] = {}
    by_config_class: dict[tuple[str, str], list[dict]] = {}
    for row in raw["rows"]:
        by_config.setdefault(row["config"], []).append(row)
        by_config_class.setdefault((row["config"], row["field_class"]), []).append(row)
    frontiers = []
    for selector in selectors["selectors"]:
        item = dict(selector)
        item["heldout"] = _aggregate(by_config[selector["config"]["key"]], all_metrics)
        frontiers.append(item)
    paired = []
    index = {
        (row["objective"], row["resource_kind"], row["resource_cap"], row["query_count"], row["family"]): row
        for row in frontiers
    }
    for key in sorted({item[:-1] for item in index}, key=str):
        spline = index.get((*key, "spline"))
        sem = index.get((*key, "sem"))
        if spline is None or sem is None:
            continue
        ratios = {}
        for metric in all_metrics:
            ratios[metric] = {
                statistic: sem["heldout"][metric][statistic] / spline["heldout"][metric][statistic]
                for statistic in ("median", "p95", "worst")
            }
        classwise = {}
        classes = sorted({field_class for config, field_class in by_config_class if config in {
            spline["config"]["key"], sem["config"]["key"],
        }})
        for field_class in classes:
            spline_class = _aggregate(
                by_config_class[(spline["config"]["key"], field_class)], all_metrics,
            )
            sem_class = _aggregate(
                by_config_class[(sem["config"]["key"], field_class)], all_metrics,
            )
            classwise[field_class] = {
                metric: {
                    statistic: sem_class[metric][statistic] / spline_class[metric][statistic]
                    for statistic in ("median", "p95", "worst")
                }
                for metric in all_metrics
            }
        paired.append({
            "objective": key[0], "resource_kind": key[1], "resource_cap": key[2],
            "query_count": key[3],
            "spline_config": spline["config"], "sem_config": sem["config"],
            "spline_cost": spline["cost"], "sem_cost": sem["cost"],
            "sem_to_spline_error_factor": ratios,
            "classwise_sem_to_spline_error_factor": classwise,
            "spline_state_reduction_fraction": 1.0 - spline["cost"]["scalar_state"] / sem["cost"]["scalar_state"],
            "spline_storage_reduction_fraction": 1.0 - spline["cost"]["persistent_storage_bytes"] / sem["cost"]["persistent_storage_bytes"],
        })
    primary = specification["metrics"]["primary"]
    tie_floor = 1.0 / (1.0 + specification["decision_tolerance"]["tie_relative"])
    decisive_factor = specification["decision_tolerance"]["decisive_error_factor"]
    decisive_state = specification["decision_tolerance"]["decisive_state_reduction"]
    full_uniform_cells = []
    analytic_uniform_cells = []
    for row in paired:
        if row["objective"] != "joint":
            continue
        class_minimum = {
            field_class: min(
                row["classwise_sem_to_spline_error_factor"][field_class][metric]["worst"]
                for metric in primary
            )
            for field_class in row["classwise_sem_to_spline_error_factor"]
        }
        class_maximum = max(
            row["classwise_sem_to_spline_error_factor"][field_class][metric]["worst"]
            for field_class in row["classwise_sem_to_spline_error_factor"]
            for metric in primary
        )
        decisive = class_maximum >= decisive_factor \
            or row["spline_state_reduction_fraction"] >= decisive_state
        cell = {
            "resource_kind": row["resource_kind"],
            "resource_cap": row["resource_cap"],
            "query_count": row["query_count"],
            "spline_config": row["spline_config"]["key"],
            "sem_config": row["sem_config"]["key"],
            "class_minimum_factors": class_minimum,
            "decisive": decisive,
        }
        if decisive and min(class_minimum.values()) >= tie_floor:
            full_uniform_cells.append(cell)
        analytic_minimum = min(
            value for field_class, value in class_minimum.items()
            if field_class != "finite_regularity"
        )
        if decisive and analytic_minimum >= tie_floor:
            analytic_uniform_cells.append({**cell, "analytic_minimum_factor": analytic_minimum})
    return {
        "protocol_id": raw["protocol_id"],
        "spec_sha256": raw["spec_sha256"],
        "design_sha256": raw["design_sha256"],
        "selector_sha256": raw["selector_sha256"],
        "frontiers": frontiers,
        "paired": paired,
        "promotion": {
            "full_declared_class_uniform_pareto_pass": bool(full_uniform_cells),
            "full_declared_class_uniform_cells": full_uniform_cells,
            "analytic_subclass_uniform_pareto_pass": bool(analytic_uniform_cells),
            "analytic_subclass_uniform_cells": analytic_uniform_cells,
        },
    }


def run_heldout() -> tuple[dict, dict]:
    """Evaluate the frozen selectors on disjoint held-out fields."""
    if not DESIGN_PATH.is_file() or not SELECTOR_PATH.is_file():
        raise FileNotFoundError("run the design stage before the held-out stage")
    spec_bytes = SPEC_PATH.read_bytes()
    specification = json.loads(spec_bytes)
    design = json.loads(DESIGN_PATH.read_text())
    selectors = json.loads(SELECTOR_PATH.read_text())
    if selectors["design_sha256"] != _sha256_path(DESIGN_PATH):
        raise ValueError("selector record does not match the frozen design artifact")
    if design["spec_sha256"] != _sha256_bytes(spec_bytes):
        raise ValueError("design artifact does not match the frozen specification")
    fields = build_fields(specification, "heldout")
    samples = sampling_record(specification, fields)
    selected = {row["config"]["key"]: row["config"] for row in selectors["selectors"]}
    rows = []
    gates = []
    for index, config in enumerate(selected.values(), 1):
        config_rows, gate = evaluate_configuration(config, fields, samples, specification)
        rows.extend(config_rows)
        gates.append({"config": config["key"], **gate})
        print(f"held-out configurations {index}/{len(selected)}", flush=True)
    raw = {
        "protocol_id": specification["protocol_id"],
        "spec_sha256": _sha256_bytes(spec_bytes),
        "design_sha256": _sha256_path(DESIGN_PATH),
        "selector_sha256": _sha256_path(SELECTOR_PATH),
        "split": "heldout",
        "field_derivative_fd_max_abs": _derivative_gate(fields),
        "polynomial_gates": gates,
        "rows": rows,
    }
    RAW_PATH.write_text(json.dumps(raw, separators=(",", ":")))
    summary = summarize(raw, selectors, specification)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=1))
    return raw, summary


def verify() -> None:
    """Reconstruct selectors and held-out summaries from saved artifacts."""
    specification = json.loads(SPEC_PATH.read_text())
    design = json.loads(DESIGN_PATH.read_text())
    selectors = json.loads(SELECTOR_PATH.read_text())
    raw = json.loads(RAW_PATH.read_text())
    summary = json.loads(SUMMARY_PATH.read_text())
    spec_hash = _sha256_path(SPEC_PATH)
    assert design["spec_sha256"] == selectors["spec_sha256"] == raw["spec_sha256"] == summary["spec_sha256"] == spec_hash
    assert selectors["design_sha256"] == raw["design_sha256"] == summary["design_sha256"] == _sha256_path(DESIGN_PATH)
    assert raw["selector_sha256"] == summary["selector_sha256"] == _sha256_path(SELECTOR_PATH)
    expected_configs = configurations(specification)
    assert design["configurations"] == expected_configs
    expected_rows = len(expected_configs) * len(build_fields(specification, "design"))
    assert len(design["rows"]) == expected_rows
    tolerance = specification["decision_tolerance"]["algebraic_absolute"]
    for gate in design["polynomial_gates"] + raw["polynomial_gates"]:
        assert gate["value"] < tolerance
        assert gate["derivative"] < tolerance
        assert gate["partition"] < tolerance
        assert gate["linear_derivative"] < tolerance
    assert design["field_derivative_fd_max_abs"] < 2.0e-6
    assert raw["field_derivative_fd_max_abs"] < 2.0e-6
    rebuilt_scores = _design_scores(design["rows"], specification)
    assert rebuilt_scores == design["scores"]
    rebuilt_selectors = select_configurations(
        expected_configs, design["costs"], rebuilt_scores, specification,
    )
    assert rebuilt_selectors == selectors["selectors"]
    selected_keys = {row["config"]["key"] for row in selectors["selectors"]}
    heldout_fields = build_fields(specification, "heldout")
    assert len(raw["rows"]) == len(selected_keys) * len(heldout_fields)
    assert {row["config"] for row in raw["rows"]} == selected_keys
    for row in design["rows"] + raw["rows"]:
        values = np.asarray(list(row["metrics"].values()), float)
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)
    rebuilt_summary = summarize(raw, selectors, specification)
    assert rebuilt_summary == summary
    print(
        "symmetric hp wall frontier verify: ALL PASS; "
        f"design_configs={len(expected_configs)} selected_configs={len(selected_keys)} "
        f"heldout_rows={len(raw['rows'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", action="store_true")
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    ensure_generated_dirs()
    if args.verify:
        verify()
        return
    if args.design or args.all or not (args.design or args.heldout):
        run_design()
        print(f"wrote {DESIGN_PATH} and {SELECTOR_PATH}")
    if args.heldout or args.all or not (args.design or args.heldout):
        run_heldout()
        print(f"wrote {RAW_PATH} and {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
