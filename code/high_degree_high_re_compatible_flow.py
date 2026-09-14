#!/usr/bin/env python3
"""Execute the frozen arbitrary-degree and high-Reynolds compatible-flow study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np

from compatible_sem_flow import CompatibleSEMFlow2D
from manufactured_compatible_comparison import _row as manufactured_row
from project_paths import CODEDIR, RESULTSDIR, ensure_generated_dirs
from tensor_compatible_cavity import TensorCompatibleCavityStepper2D
from tensor_compatible_flow import TensorCompatibleFlow2D, VelocityPair


SCHEMA = "high-degree-high-re-compatible-flow-v1"
ALGEBRA_OUTPUT = RESULTSDIR / "high_degree_compatible_algebra.json"
MANUFACTURED_OUTPUT = RESULTSDIR / "high_degree_manufactured_comparison.json"
TEMPORAL_OUTPUT = RESULTSDIR / "high_re_temporal_control.json"
FLOW_OUTPUT = RESULTSDIR / "high_re_smooth_cavity_comparison.json"
TIMING_OUTPUT = RESULTSDIR / "high_re_matched_accuracy_timing.json"

ALGEBRA_DEGREES = tuple(range(2, 13))
MANUFACTURED_DEGREES = (4, 6, 8, 10, 12)
MANUFACTURED_INDICES = (1, 2, 4)
FLOW_DEGREES = (8, 10, 12)
FLOW_INDICES = (2, 3)
REYNOLDS_NUMBERS = (1000.0, 3200.0)
REFERENCE_DEGREE = 12
REFERENCE_INDEX = 4
FALLBACK_REFERENCE_INDEX = 5
TEMPORAL_END_TIME = 0.5
FLOW_END_TIME = 4.0
STATISTICS_START_TIME = 3.0
PRODUCTION_DT = 0.005
REFERENCE_DT = 0.0025
FALLBACK_PRODUCTION_DT = 0.0025
FALLBACK_REFERENCE_DT = 0.00125
RELATIVE_TOLERANCE = 1.0e-11
MAXIMUM_ITERATIONS = 20000
ALGEBRA_TOLERANCE = 5.0e-12
TEMPORAL_TOLERANCE = 0.02
REFERENCE_TOLERANCE = 0.05
MATCHED_TARGETS = (0.05, 0.02)
FIELD_POINTS = np.linspace(0.0, 1.0, 49)
WALL_POINTS = np.linspace(0.0, 1.0, 99)[1:-1]
SOURCE_FILES = (
    CODEDIR / "tensor_compatible_flow.py",
    CODEDIR / "tensor_compatible_cavity.py",
    CODEDIR / "compatible_sem_flow.py",
    CODEDIR / "manufactured_compatible_comparison.py",
    Path(__file__).resolve(),
)


def _sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a source or result file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    """Bind every result to the complete implementation source set."""
    return {path.name: _sha256(path) for path in SOURCE_FILES}


def _flow(method: str, degree: int, index: int) -> TensorCompatibleFlow2D:
    """Assemble one equal-index BMF or mixed compatible SEM flow space."""
    npts = degree * index + 1
    if method == "bmf":
        return TensorCompatibleFlow2D(npts, degree)
    if method == "compatible_sem":
        return CompatibleSEMFlow2D(degree, index)
    raise ValueError(f"unknown compatible-flow method {method}")


def _smooth_lid(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate the quartic corner-compatible moving-lid trace."""
    shape = np.broadcast_shapes(np.shape(x), np.shape(z))
    coordinate = np.broadcast_to(np.asarray(x, dtype=float), shape)
    top = np.broadcast_to(np.isclose(np.asarray(z, dtype=float), 1.0), shape)
    return top * 16.0 * coordinate**2 * (1.0 - coordinate) ** 2


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    """Return a scale-aware Euclidean relative difference."""
    left_value = np.asarray(left, dtype=float)
    right_value = np.asarray(right, dtype=float)
    return float(np.linalg.norm(left_value - right_value)) / max(
        float(np.linalg.norm(right_value)), 1.0e-300
    )


def _symmetric_relative(left: float, right: float) -> float:
    """Return a symmetric scalar difference for signed wall quantities."""
    return abs(float(left) - float(right)) / max(
        abs(float(left)), abs(float(right)), 1.0e-300
    )


def _pressure_gradient_matrix(flow: TensorCompatibleFlow2D) -> np.ndarray:
    """Assemble the active pressure action for a finite rank census."""
    pressure_size = int(np.prod(flow.pressure_shape))
    velocity_size = (flow.npts - 2) * (flow.npts - 1) * 2
    matrix = np.empty((velocity_size, pressure_size))
    for column in range(pressure_size):
        pressure = np.zeros(flow.pressure_shape)
        pressure.flat[column] = 1.0
        action = flow.pressure_action(pressure)
        matrix[:, column] = np.concatenate(
            [action.ux[1:-1, :].ravel(), action.uz[:, 1:-1].ravel()]
        )
    return matrix


def _algebra_row(method: str, degree: int) -> dict[str, Any]:
    """Measure pressure range, adjoint work, energy skewness, and conditioning."""
    flow = _flow(method, degree, 2)
    axis = flow.axis
    rng = np.random.default_rng(8100 + 37 * degree + (method != "bmf"))
    ux = rng.standard_normal(flow.ux_shape)
    uz = rng.standard_normal(flow.uz_shape)
    ux[[0, -1], :] = 0.0
    uz[:, [0, -1]] = 0.0
    pressure = rng.standard_normal(flow.pressure_shape)
    divergence = flow.divergence(ux, uz)
    pressure_action = flow.pressure_action(pressure)
    pressure_work = float(
        np.sum(ux * pressure_action.ux) + np.sum(uz * pressure_action.uz)
    )
    divergence_work = float(np.sum(divergence * pressure))
    adjoint_residual = abs(pressure_work - divergence_work) / max(
        abs(pressure_work), abs(divergence_work), 1.0e-300
    )
    convection = flow.convection_dual(ux, uz)
    convective_work = float(
        np.sum(ux * convection.ux) + np.sum(uz * convection.uz)
    )
    convective_scale = max(
        float(np.linalg.norm(ux) * np.linalg.norm(convection.ux))
        + float(np.linalg.norm(uz) * np.linalg.norm(convection.uz)),
        1.0e-300,
    )
    gradient = _pressure_gradient_matrix(flow)
    singular_values = np.linalg.svd(gradient, compute_uv=False)
    rank_tolerance = singular_values[0] * max(gradient.shape) * np.finfo(float).eps
    rank = int(np.sum(singular_values > rank_tolerance))
    positive = singular_values[singular_values > rank_tolerance]
    exact_sequence_residual = float(
        np.linalg.norm((axis.d0 - axis.b1 @ axis.incidence).toarray())
    ) / max(float(np.linalg.norm(axis.d0.toarray())), 1.0e-300)
    stepper = TensorCompatibleCavityStepper2D(
        npts=flow.npts,
        degree=degree,
        reynolds=3200.0,
        dt=PRODUCTION_DT,
        relative_tolerance=RELATIVE_TOLERANCE,
        maximum_iterations=MAXIMUM_ITERATIONS,
        flow=flow,
        lid_velocity=_smooth_lid,
    )
    first_step = stepper.step(stepper.initial_state())
    wall = stepper.nitsche_wall_observable(first_step.state)
    flow_storage = flow.storage_record()
    storage = stepper.storage_record()
    storage["stored_factor_entries"] = int(
        storage["flow_factor_entries"]
        + storage["time_factor_entries"]
        + storage["pressure_preconditioner_entries"]
    )
    return {
        "method": method,
        "degree": degree,
        "index": 2,
        "npts": flow.npts,
        "exact_sequence_residual": exact_sequence_residual,
        "pressure_dimension": int(np.prod(flow.pressure_shape)),
        "pressure_action_rank": rank,
        "pressure_nullity": int(np.prod(flow.pressure_shape)) - rank,
        "smallest_positive_pressure_singular_value": float(positive[-1]),
        "pressure_action_condition_number": float(positive[0] / positive[-1]),
        "pressure_work_adjoint_residual": adjoint_residual,
        "relative_convective_energy_production": abs(convective_work)
        / convective_scale,
        "mass0_condition_number": float(np.linalg.cond(axis.mass0.toarray())),
        "mass1_condition_number": float(np.linalg.cond(axis.mass1.toarray())),
        "first_step_relative_residual": first_step.saddle.relative_residual,
        "first_step_divergence_relative_residual": (
            first_step.saddle.divergence_relative_residual
        ),
        "first_step_pressure_gauge_relative_residual": (
            first_step.saddle.pressure_gauge_relative_residual
        ),
        "first_step_energy_balance_residual": first_step.energy[
            "relative_balance_residual"
        ],
        "first_step_maximum_normal_flux": wall["maximum_normal_flux"],
        "first_step_iterations": first_step.saddle.iterations,
        "operator_storage": storage,
    }


def build_algebra_record() -> dict[str, Any]:
    """Execute the smallest arbitrary-degree structural realization audit."""
    rows = [
        _algebra_row(method, degree)
        for degree in ALGEBRA_DEGREES
        for method in ("bmf", "compatible_sem")
    ]
    return {
        "schema": SCHEMA,
        "study": "arbitrary_degree_algebra",
        "case": {
            "degrees": list(ALGEBRA_DEGREES),
            "index": 2,
            "reynolds": 3200.0,
            "dt": PRODUCTION_DT,
            "relative_tolerance": RELATIVE_TOLERANCE,
            "algebra_tolerance": ALGEBRA_TOLERANCE,
        },
        "source_sha256": _source_hashes(),
        "rows": rows,
    }


def verify_algebra_record(record: dict[str, Any]) -> None:
    """Verify every frozen arbitrary-degree algebra condition."""
    if record.get("schema") != SCHEMA or record.get("study") != "arbitrary_degree_algebra":
        raise ValueError("arbitrary-degree algebra schema mismatch")
    if record.get("source_sha256") != _source_hashes():
        raise ValueError("arbitrary-degree algebra source mismatch")
    expected = [
        (method, degree)
        for degree in ALGEBRA_DEGREES
        for method in ("bmf", "compatible_sem")
    ]
    actual = [(row["method"], row["degree"]) for row in record["rows"]]
    if actual != expected:
        raise ValueError("arbitrary-degree algebra matrix changed")
    for row in record["rows"]:
        if int(row["pressure_nullity"]) != 1:
            raise AssertionError("pressure range has a nonphysical nullity")
        for key in (
            "exact_sequence_residual",
            "pressure_work_adjoint_residual",
            "relative_convective_energy_production",
        ):
            if float(row[key]) > ALGEBRA_TOLERANCE:
                raise AssertionError(f"arbitrary-degree algebra failed for {key}")
        if float(row["first_step_relative_residual"]) > RELATIVE_TOLERANCE:
            raise AssertionError("arbitrary-degree saddle residual failed")
        if float(row["first_step_divergence_relative_residual"]) > 2.0e-11:
            raise AssertionError("arbitrary-degree divergence failed")
        if float(row["first_step_pressure_gauge_relative_residual"]) > 2.0e-11:
            raise AssertionError("arbitrary-degree pressure gauge failed")
        if float(row["first_step_energy_balance_residual"]) > 5.0e-9:
            raise AssertionError("arbitrary-degree energy ledger failed")
        if float(row["first_step_maximum_normal_flux"]) > 5.0e-13:
            raise AssertionError("arbitrary-degree no-penetration failed")
        for key in ("mass0_condition_number", "mass1_condition_number"):
            if not math.isfinite(float(row[key])):
                raise AssertionError("arbitrary-degree mass condition is not finite")
        if int(row["operator_storage"]["multidimensional_dense_operator_entries"]) != 0:
            raise AssertionError("arbitrary-degree path stored a dense tensor operator")


def _manufactured_selections(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the fastest admissible high-degree manufactured cell symmetrically."""
    targets = (1.0e-5, 1.0e-8, 1.0e-10)
    selections: dict[str, Any] = {}
    for method in ("bmf", "compatible_sem"):
        selections[method] = {}
        for target in targets:
            admissible = [
                row
                for row in rows
                if row["method"] == method
                and max(
                    row["relative_velocity_l2_error"],
                    row["relative_pressure_l2_error"],
                    row["relative_wall_traction_l2_error"],
                )
                <= target
            ]
            selected = min(admissible, key=lambda row: row["total_seconds"], default=None)
            selections[method][f"{target:.0e}"] = None if selected is None else {
                "degree": selected["degree"],
                "index": selected["elements"],
                "npts": selected["npts"],
                "joint_error": max(
                    selected["relative_velocity_l2_error"],
                    selected["relative_pressure_l2_error"],
                    selected["relative_wall_traction_l2_error"],
                ),
                "state_storage_bytes": selected["state_storage_bytes"],
                "operator_storage_bytes": selected["operator_storage_bytes"],
                "total_seconds": selected["total_seconds"],
            }
    return selections


def _single_element_tie_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Scale equal-space output ties by the measured mass-conditioning floor."""
    algebra = _load(ALGEBRA_OUTPUT)
    algebra_rows = {
        (row["method"], row["degree"]): row for row in algebra["rows"]
    }
    audit: dict[str, Any] = {}
    for degree in MANUFACTURED_DEGREES:
        pair = [
            row for row in rows if row["degree"] == degree and row["elements"] == 1
        ]
        maximum_condition = max(
            float(algebra_rows[(method, degree)][key])
            for method in ("bmf", "compatible_sem")
            for key in ("mass0_condition_number", "mass1_condition_number")
        )
        allowance = max(
            1.0e-12,
            50.0 * np.finfo(float).eps * math.sqrt(maximum_condition),
        )
        differences = {
            key: abs(float(pair[0][key]) - float(pair[1][key]))
            for key in (
                "relative_velocity_l2_error",
                "relative_pressure_l2_error",
                "relative_wall_traction_l2_error",
            )
        }
        audit[str(degree)] = {
            "maximum_mass_condition_number": maximum_condition,
            "absolute_allowance": allowance,
            "absolute_differences": differences,
        }
    return audit


def build_manufactured_record() -> dict[str, Any]:
    """Execute the symmetric manufactured degree and resolution ladder."""
    rows = [
        manufactured_row(method, degree, index)
        for degree in MANUFACTURED_DEGREES
        for index in MANUFACTURED_INDICES
        for method in ("bmf", "compatible_sem")
    ]
    return {
        "schema": SCHEMA,
        "study": "high_degree_manufactured",
        "case": {
            "degrees": list(MANUFACTURED_DEGREES),
            "indices": list(MANUFACTURED_INDICES),
            "relative_tolerance": RELATIVE_TOLERANCE,
            "pressure_preconditioner": "product_schur",
        },
        "source_sha256": _source_hashes(),
        "algebra_result_sha256": _sha256(ALGEBRA_OUTPUT),
        "rows": rows,
        "selections": _manufactured_selections(rows),
        "single_element_tie_audit": _single_element_tie_audit(rows),
    }


def verify_manufactured_record(record: dict[str, Any]) -> None:
    """Verify comparator parity and high-degree manufactured invariants."""
    if record.get("schema") != SCHEMA or record.get("study") != "high_degree_manufactured":
        raise ValueError("high-degree manufactured schema mismatch")
    if record.get("source_sha256") != _source_hashes():
        raise ValueError("high-degree manufactured source mismatch")
    if record.get("algebra_result_sha256") != _sha256(ALGEBRA_OUTPUT):
        raise ValueError("high-degree manufactured algebra lineage mismatch")
    expected = [
        (method, degree, index)
        for degree in MANUFACTURED_DEGREES
        for index in MANUFACTURED_INDICES
        for method in ("bmf", "compatible_sem")
    ]
    actual = [(row["method"], row["degree"], row["elements"]) for row in record["rows"]]
    if actual != expected:
        raise ValueError("high-degree manufactured matrix changed")
    rows = record["rows"]
    for row in rows:
        if float(row["relative_saddle_residual"]) > RELATIVE_TOLERANCE:
            raise AssertionError("high-degree manufactured saddle residual failed")
        if float(row["divergence_relative_residual"]) > 2.0e-11:
            raise AssertionError("high-degree manufactured divergence failed")
        if float(row["pressure_gauge_relative_residual"]) > 2.0e-11:
            raise AssertionError("high-degree manufactured pressure gauge failed")
        if float(row["maximum_normal_flux"]) > 5.0e-13:
            raise AssertionError("high-degree manufactured no-penetration failed")
    tie_audit = _single_element_tie_audit(rows)
    if record.get("single_element_tie_audit") != tie_audit:
        raise ValueError("single-element tie audit changed")
    for degree_record in tie_audit.values():
        allowance = float(degree_record["absolute_allowance"])
        if any(
            float(difference) > allowance
            for difference in degree_record["absolute_differences"].values()
        ):
            raise AssertionError("single-element compatible spaces do not agree")
    if record.get("selections") != _manufactured_selections(rows):
        raise ValueError("high-degree manufactured selections changed")


def _sample_traction(traction: dict[str, Any]) -> dict[str, Any]:
    """Interpolate every wall traction component to common wall points."""
    return {
        wall_name: {
            component: np.interp(
                WALL_POINTS, wall["coordinate"], wall[component]
            ).tolist()
            for component in ("traction_x", "traction_z", "normal_flux")
        }
        for wall_name, wall in traction.items()
    }


def _sample_state(
    stepper: TensorCompatibleCavityStepper2D,
    state: Any,
) -> dict[str, Any]:
    """Evaluate state and wall functionals on the frozen common grids."""
    pressure_coefficients = stepper.bdf2.primal_pressure(state.dual_pressure)
    velocity = stepper.flow.evaluate_velocity(
        state.ux, state.uz, FIELD_POINTS, FIELD_POINTS
    )
    pressure = stepper.flow.evaluate_pressure(
        pressure_coefficients, FIELD_POINTS, FIELD_POINTS
    )
    pressure -= float(np.mean(pressure))
    return {
        "ux": velocity.ux.tolist(),
        "uz": velocity.uz.tolist(),
        "pressure": pressure.tolist(),
        "strong_traction": _sample_traction(stepper.strong_wall_traction(state)),
        "native_traction": _sample_traction(
            stepper.nitsche_consistent_wall_traction(state)
        ),
    }


def _online_statistics(values: list[float]) -> dict[str, float]:
    """Return mean and root-mean-square fluctuation of a sampled scalar."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("statistics window contains no samples")
    return {
        "mean": float(np.mean(array)),
        "rms_fluctuation": float(np.sqrt(np.mean((array - np.mean(array)) ** 2))),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _run_flow_case(
    method: str,
    degree: int,
    index: int,
    reynolds: float,
    dt: float,
    end_time: float,
    statistics_start: float,
    role: str,
) -> dict[str, Any]:
    """Advance one pressure-coupled smooth-lid transient with full accounting."""
    setup_start = time.process_time()
    flow = _flow(method, degree, index)
    stepper = TensorCompatibleCavityStepper2D(
        npts=flow.npts,
        degree=degree,
        reynolds=reynolds,
        dt=dt,
        relative_tolerance=RELATIVE_TOLERANCE,
        maximum_iterations=MAXIMUM_ITERATIONS,
        flow=flow,
        lid_velocity=_smooth_lid,
    )
    setup_seconds = time.process_time() - setup_start
    steps = int(round(end_time / dt))
    if not math.isclose(steps * dt, end_time, rel_tol=0.0, abs_tol=1.0e-14):
        raise ValueError("time horizon must contain an integer number of steps")
    state = stepper.initial_state()
    integrated_native_wall_work = 0.0
    integrated_net_wall_work = 0.0
    integrated_reaction_work = 0.0
    maximum_residual = 0.0
    maximum_divergence = 0.0
    maximum_gauge = 0.0
    maximum_energy = 0.0
    maximum_normal_flux = 0.0
    iterations: list[int] = []
    reaction_window: list[float] = []
    native_power_window: list[float] = []
    energy_window: list[float] = []
    recurring_start = time.process_time()
    for _step in range(steps):
        previous = state
        result = stepper.step(state)
        state = result.state
        native = stepper.nitsche_wall_observable(state)
        strong = stepper.native_wall_observable(state)
        integrated_native_wall_work += float(native["lid_power"]) * dt
        integrated_net_wall_work += float(result.energy["net_wall_power"]) * dt
        integrated_reaction_work += float(strong["lid_power"]) * dt
        iterations.append(result.saddle.iterations)
        maximum_residual = max(maximum_residual, result.saddle.relative_residual)
        maximum_divergence = max(
            maximum_divergence, result.saddle.divergence_relative_residual
        )
        maximum_gauge = max(
            maximum_gauge, result.saddle.pressure_gauge_relative_residual
        )
        maximum_energy = max(
            maximum_energy, result.energy["relative_balance_residual"]
        )
        maximum_normal_flux = max(
            maximum_normal_flux, float(native["maximum_normal_flux"])
        )
        if state.time + 1.0e-14 >= statistics_start:
            reaction_window.append(float(native["lid_reaction"]))
            native_power_window.append(float(native["lid_power"]))
            energy_window.append(float(result.energy["kinetic_energy"]))
    recurring_seconds = time.process_time() - recurring_start
    flow_storage = flow.storage_record()
    storage = stepper.storage_record()
    storage["stored_factor_entries"] = int(
        storage["flow_factor_entries"]
        + storage["time_factor_entries"]
        + storage["pressure_preconditioner_entries"]
    )
    observable_times = []
    for _repeat in range(25):
        observable_start = time.process_time()
        stepper.nitsche_wall_observable(state)
        observable_times.append(time.process_time() - observable_start)
    return {
        "method": method,
        "degree": degree,
        "index": index,
        "npts": flow.npts,
        "role": role,
        "reynolds": reynolds,
        "dt": dt,
        "steps": steps,
        "final_time": state.time,
        "statistics_start_time": statistics_start,
        "setup_seconds": setup_seconds,
        "recurring_seconds": recurring_seconds,
        "total_seconds": setup_seconds + recurring_seconds,
        "median_wall_observable_seconds": float(np.median(observable_times)),
        "median_iterations": float(np.median(iterations)),
        "maximum_iterations": int(max(iterations)),
        "total_iterations": int(sum(iterations)),
        "final_velocity_rate": stepper.velocity_rate(state, previous),
        "maximum_relative_residual": maximum_residual,
        "maximum_divergence_relative_residual": maximum_divergence,
        "maximum_pressure_gauge_relative_residual": maximum_gauge,
        "maximum_energy_balance_residual": maximum_energy,
        "maximum_normal_flux": maximum_normal_flux,
        "integrated_native_wall_work": integrated_native_wall_work,
        "integrated_net_wall_work": integrated_net_wall_work,
        "integrated_reaction_work": integrated_reaction_work,
        "native_lid_reaction_statistics": _online_statistics(reaction_window),
        "native_lid_power_statistics": _online_statistics(native_power_window),
        "kinetic_energy_statistics": _online_statistics(energy_window),
        "state_storage_bytes": int(
            8
            * (
                flow_storage["velocity_states"]
                + flow_storage["pressure_states"]
            )
        ),
        "operator_storage": storage,
        "samples": _sample_state(stepper, state),
    }


def _traction_vector(row: dict[str, Any], family: str, component: str) -> np.ndarray:
    """Flatten one physical wall-traction component set."""
    traction = row["samples"][family]
    if component == "tangential":
        fields = (
            ("x0", "traction_z"),
            ("x1", "traction_z"),
            ("z0", "traction_x"),
            ("z1", "traction_x"),
        )
    elif component == "normal":
        fields = (
            ("x0", "traction_x"),
            ("x1", "traction_x"),
            ("z0", "traction_z"),
            ("z1", "traction_z"),
        )
    else:
        raise ValueError("traction component must be tangential or normal")
    values: list[float] = []
    for wall, field in fields:
        values.extend(traction[wall][field])
    return np.asarray(values)


def _comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    """Compute the frozen joint state and native wall-functional difference."""
    velocity = math.sqrt(
        _relative(np.asarray(left["samples"]["ux"]), np.asarray(right["samples"]["ux"])) ** 2
        + _relative(np.asarray(left["samples"]["uz"]), np.asarray(right["samples"]["uz"])) ** 2
    )
    pressure = _relative(
        np.asarray(left["samples"]["pressure"]),
        np.asarray(right["samples"]["pressure"]),
    )
    tangential = _relative(
        _traction_vector(left, "native_traction", "tangential"),
        _traction_vector(right, "native_traction", "tangential"),
    )
    normal = _relative(
        _traction_vector(left, "strong_traction", "normal"),
        _traction_vector(right, "strong_traction", "normal"),
    )
    reaction = _symmetric_relative(
        left["native_lid_reaction_statistics"]["mean"],
        right["native_lid_reaction_statistics"]["mean"],
    )
    wall_work = _symmetric_relative(
        left["integrated_native_wall_work"], right["integrated_native_wall_work"]
    )
    joint = max(velocity, pressure, tangential, reaction, wall_work)
    return {
        "relative_velocity": velocity,
        "relative_pressure": pressure,
        "relative_native_tangential_traction": tangential,
        "relative_strong_normal_traction_diagnostic": normal,
        "relative_native_lid_reaction": reaction,
        "relative_integrated_native_wall_work": wall_work,
        "native_wall_closure_joint": joint,
    }


def _mutual_comparison(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, float]:
    """Return the larger directional difference for each reference diagnostic."""
    forward = _comparison(left, right)
    reverse = _comparison(right, left)
    return {key: max(forward[key], reverse[key]) for key in forward}


def _verify_flow_row(row: dict[str, Any]) -> None:
    """Verify exact invariants and resource representation for one flow row."""
    if float(row["maximum_relative_residual"]) > RELATIVE_TOLERANCE:
        raise AssertionError("high-Reynolds saddle residual failed")
    if float(row["maximum_divergence_relative_residual"]) > 2.0e-11:
        raise AssertionError("high-Reynolds divergence failed")
    if float(row["maximum_pressure_gauge_relative_residual"]) > 2.0e-11:
        raise AssertionError("high-Reynolds pressure gauge failed")
    if float(row["maximum_energy_balance_residual"]) > 5.0e-9:
        raise AssertionError("high-Reynolds energy ledger failed")
    if float(row["maximum_normal_flux"]) > 5.0e-13:
        raise AssertionError("high-Reynolds no-penetration failed")
    if int(row["operator_storage"]["multidimensional_dense_operator_entries"]) != 0:
        raise AssertionError("high-Reynolds path stored a dense tensor operator")


def build_temporal_record() -> dict[str, Any]:
    """Run the predeclared shared high-Reynolds timestep decision."""
    rows = [
        _run_flow_case(
            method,
            12,
            2,
            3200.0,
            dt,
            TEMPORAL_END_TIME,
            0.0,
            "temporal_control",
        )
        for method in ("bmf", "compatible_sem")
        for dt in (PRODUCTION_DT, REFERENCE_DT)
    ]
    comparisons = {}
    needs_fallback = False
    for method in ("bmf", "compatible_sem"):
        coarse, fine = [row for row in rows if row["method"] == method]
        comparison = _comparison(coarse, fine)
        comparisons[method] = comparison
        needs_fallback = needs_fallback or comparison["native_wall_closure_joint"] > TEMPORAL_TOLERANCE
    selected_production_dt = PRODUCTION_DT
    selected_reference_dt = REFERENCE_DT
    if needs_fallback:
        selected_production_dt = FALLBACK_PRODUCTION_DT
        selected_reference_dt = FALLBACK_REFERENCE_DT
        for method in ("bmf", "compatible_sem"):
            rows.append(
                _run_flow_case(
                    method,
                    12,
                    2,
                    3200.0,
                    FALLBACK_REFERENCE_DT,
                    TEMPORAL_END_TIME,
                    0.0,
                    "temporal_control_fallback",
                )
            )
            production = next(
                row
                for row in rows
                if row["method"] == method and row["dt"] == FALLBACK_PRODUCTION_DT
            )
            reference = rows[-1]
            comparisons[method] = _comparison(production, reference)
    return {
        "schema": SCHEMA,
        "study": "high_re_temporal_control",
        "case": {
            "reynolds": 3200.0,
            "degree": 12,
            "index": 2,
            "end_time": TEMPORAL_END_TIME,
            "temporal_tolerance": TEMPORAL_TOLERANCE,
        },
        "source_sha256": _source_hashes(),
        "rows": rows,
        "comparisons": comparisons,
        "selected_production_dt": selected_production_dt,
        "selected_reference_dt": selected_reference_dt,
    }


def verify_temporal_record(record: dict[str, Any]) -> None:
    """Verify the shared timestep decision and every temporal pilot invariant."""
    if record.get("schema") != SCHEMA or record.get("study") != "high_re_temporal_control":
        raise ValueError("high-Reynolds temporal schema mismatch")
    if record.get("source_sha256") != _source_hashes():
        raise ValueError("high-Reynolds temporal source mismatch")
    for row in record["rows"]:
        _verify_flow_row(row)
    for comparison in record["comparisons"].values():
        if float(comparison["native_wall_closure_joint"]) > TEMPORAL_TOLERANCE:
            raise AssertionError("high-Reynolds temporal control failed")
    selected = (
        float(record["selected_production_dt"]),
        float(record["selected_reference_dt"]),
    )
    if selected not in (
        (PRODUCTION_DT, REFERENCE_DT),
        (FALLBACK_PRODUCTION_DT, FALLBACK_REFERENCE_DT),
    ):
        raise ValueError("high-Reynolds timestep decision changed")


def _candidate_errors(
    row: dict[str, Any], references: tuple[dict[str, Any], dict[str, Any]]
) -> dict[str, Any]:
    """Score a candidate against both equal-state reference families."""
    comparisons = {
        reference["method"]: _comparison(row, reference) for reference in references
    }
    return {
        "against_references": comparisons,
        "robust_native_wall_closure_joint": max(
            value["native_wall_closure_joint"] for value in comparisons.values()
        ),
    }


def _flow_selections(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select complete-mission cells at each matched joint target."""
    result: dict[str, Any] = {}
    for reynolds in REYNOLDS_NUMBERS:
        result[f"{reynolds:.0f}"] = {}
        references = tuple(
            row
            for row in rows
            if row["reynolds"] == reynolds and row["role"] == "reference"
        )
        if len(references) != 2:
            raise ValueError("each Reynolds number requires two references")
        mutual = _mutual_comparison(references[0], references[1])
        result[f"{reynolds:.0f}"]["reference_disagreement"] = mutual
        for method in ("bmf", "compatible_sem"):
            candidates = [
                row
                for row in rows
                if row["reynolds"] == reynolds
                and row["method"] == method
                and row["role"] == "candidate"
            ]
            for row in candidates:
                row["errors"] = _candidate_errors(row, references)
            result[f"{reynolds:.0f}"][method] = {}
            for target in MATCHED_TARGETS:
                admissible = [
                    row
                    for row in candidates
                    if row["errors"]["robust_native_wall_closure_joint"] <= target
                ]
                if not admissible:
                    result[f"{reynolds:.0f}"][method][f"{target:.2f}"] = None
                    continue
                fastest = min(row["total_seconds"] for row in admissible)
                near_fastest = [
                    row for row in admissible if row["total_seconds"] <= 1.05 * fastest
                ]
                selected = min(near_fastest, key=lambda row: row["state_storage_bytes"])
                result[f"{reynolds:.0f}"][method][f"{target:.2f}"] = {
                    "degree": selected["degree"],
                    "index": selected["index"],
                    "npts": selected["npts"],
                    "joint_error": selected["errors"]["robust_native_wall_closure_joint"],
                    "state_storage_bytes": selected["state_storage_bytes"],
                    "stored_factor_entries": selected["operator_storage"]["stored_factor_entries"],
                    "total_iterations": selected["total_iterations"],
                    "total_seconds": selected["total_seconds"],
                }
    return result


def build_flow_record(temporal_record: dict[str, Any]) -> dict[str, Any]:
    """Execute the symmetric high-Reynolds candidate and reference matrix."""
    production_dt = float(temporal_record["selected_production_dt"])
    reference_dt = float(temporal_record["selected_reference_dt"])
    rows: list[dict[str, Any]] = []
    for reynolds_index, reynolds in enumerate(REYNOLDS_NUMBERS):
        for degree in FLOW_DEGREES:
            for index in FLOW_INDICES:
                methods = (
                    ("bmf", "compatible_sem")
                    if (reynolds_index + degree + index) % 2 == 0
                    else ("compatible_sem", "bmf")
                )
                for method in methods:
                    rows.append(
                        _run_flow_case(
                            method,
                            degree,
                            index,
                            reynolds,
                            production_dt,
                            FLOW_END_TIME,
                            STATISTICS_START_TIME,
                            "candidate",
                        )
                    )
        reference_methods = (
            ("bmf", "compatible_sem")
            if reynolds_index % 2 == 0
            else ("compatible_sem", "bmf")
        )
        for method in reference_methods:
            rows.append(
                _run_flow_case(
                    method,
                    REFERENCE_DEGREE,
                    REFERENCE_INDEX,
                    reynolds,
                    reference_dt,
                    FLOW_END_TIME,
                    STATISTICS_START_TIME,
                    "reference",
                )
            )
        active_references = tuple(
            row
            for row in rows
            if row["reynolds"] == reynolds and row["role"] == "reference"
        )
        if _mutual_comparison(*active_references)["native_wall_closure_joint"] > REFERENCE_TOLERANCE:
            for row in active_references:
                row["role"] = "reference_coarse"
            fallback_methods = tuple(reversed(reference_methods))
            for method in fallback_methods:
                rows.append(
                    _run_flow_case(
                        method,
                        REFERENCE_DEGREE,
                        FALLBACK_REFERENCE_INDEX,
                        reynolds,
                        reference_dt,
                        FLOW_END_TIME,
                        STATISTICS_START_TIME,
                        "reference",
                    )
                )
    selections = _flow_selections(rows)
    return {
        "schema": SCHEMA,
        "study": "high_re_smooth_cavity",
        "case": {
            "reynolds_numbers": list(REYNOLDS_NUMBERS),
            "candidate_degrees": list(FLOW_DEGREES),
            "candidate_indices": list(FLOW_INDICES),
            "reference_degree": REFERENCE_DEGREE,
            "reference_index": REFERENCE_INDEX,
            "fallback_reference_index": FALLBACK_REFERENCE_INDEX,
            "production_dt": production_dt,
            "reference_dt": reference_dt,
            "end_time": FLOW_END_TIME,
            "statistics_start_time": STATISTICS_START_TIME,
            "matched_targets": list(MATCHED_TARGETS),
            "reference_tolerance": REFERENCE_TOLERANCE,
            "field_points": FIELD_POINTS.tolist(),
            "wall_points": WALL_POINTS.tolist(),
        },
        "source_sha256": _source_hashes(),
        "temporal_result_sha256": _sha256(TEMPORAL_OUTPUT),
        "rows": rows,
        "selections": selections,
    }


def verify_flow_record(record: dict[str, Any]) -> None:
    """Verify the frozen high-Reynolds matrix, references, and selections."""
    if record.get("schema") != SCHEMA or record.get("study") != "high_re_smooth_cavity":
        raise ValueError("high-Reynolds flow schema mismatch")
    if record.get("source_sha256") != _source_hashes():
        raise ValueError("high-Reynolds flow source mismatch")
    if record.get("temporal_result_sha256") != _sha256(TEMPORAL_OUTPUT):
        raise ValueError("high-Reynolds temporal lineage mismatch")
    for row in record["rows"]:
        _verify_flow_row(row)
        if not math.isclose(float(row["final_time"]), FLOW_END_TIME, abs_tol=1.0e-12):
            raise AssertionError("high-Reynolds final time changed")
    selections = _flow_selections(record["rows"])
    if record.get("selections") != selections:
        raise ValueError("high-Reynolds selections changed")
    for reynolds in REYNOLDS_NUMBERS:
        disagreement = selections[f"{reynolds:.0f}"]["reference_disagreement"]
        if disagreement["native_wall_closure_joint"] > REFERENCE_TOLERANCE:
            raise AssertionError("high-Reynolds references do not agree closely enough")


def _selected_configurations(flow_record: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract unique matched-accuracy selections from the frozen flow record."""
    configurations: dict[tuple[float, str, int, int], dict[str, Any]] = {}
    for reynolds in REYNOLDS_NUMBERS:
        group = flow_record["selections"][f"{reynolds:.0f}"]
        for method in ("bmf", "compatible_sem"):
            for target in MATCHED_TARGETS:
                selected = group[method][f"{target:.2f}"]
                if selected is None:
                    continue
                key = (
                    reynolds,
                    method,
                    int(selected["degree"]),
                    int(selected["index"]),
                )
                configurations[key] = {
                    "reynolds": reynolds,
                    "method": method,
                    "degree": int(selected["degree"]),
                    "index": int(selected["index"]),
                }
    return [configurations[key] for key in sorted(configurations)]


def _timing_factors(
    flow_record: dict[str, Any], timing_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compute matched-target resource factors from three timing blocks."""
    medians: dict[tuple[float, str, int, int], float] = {}
    for configuration in _selected_configurations(flow_record):
        key = (
            configuration["reynolds"],
            configuration["method"],
            configuration["degree"],
            configuration["index"],
        )
        samples = [
            row["total_seconds"]
            for row in timing_rows
            if (
                row["reynolds"],
                row["method"],
                row["degree"],
                row["index"],
            )
            == key
        ]
        if len(samples) != 3:
            raise ValueError("selected timing cell requires three blocks")
        medians[key] = float(np.median(samples))
    factors: dict[str, Any] = {}
    for reynolds in REYNOLDS_NUMBERS:
        factors[f"{reynolds:.0f}"] = {}
        group = flow_record["selections"][f"{reynolds:.0f}"]
        for target in MATCHED_TARGETS:
            bmf = group["bmf"][f"{target:.2f}"]
            sem = group["compatible_sem"][f"{target:.2f}"]
            if bmf is None or sem is None:
                factors[f"{reynolds:.0f}"][f"{target:.2f}"] = None
                continue
            bmf_key = (reynolds, "bmf", int(bmf["degree"]), int(bmf["index"]))
            sem_key = (
                reynolds,
                "compatible_sem",
                int(sem["degree"]),
                int(sem["index"]),
            )
            factors[f"{reynolds:.0f}"][f"{target:.2f}"] = {
                "bmf": {**bmf, "median_total_seconds": medians[bmf_key]},
                "compatible_sem": {
                    **sem,
                    "median_total_seconds": medians[sem_key],
                },
                "comparator_to_bmf_state_factor": (
                    sem["state_storage_bytes"] / bmf["state_storage_bytes"]
                ),
                "comparator_to_bmf_operator_factor": (
                    sem["stored_factor_entries"] / bmf["stored_factor_entries"]
                ),
                "comparator_to_bmf_iteration_factor": (
                    sem["total_iterations"] / bmf["total_iterations"]
                ),
                "comparator_to_bmf_time_factor": (
                    medians[sem_key] / medians[bmf_key]
                ),
            }
    return factors


def build_timing_record(flow_record: dict[str, Any]) -> dict[str, Any]:
    """Rerun selected matched-accuracy cells in three alternating timing blocks."""
    production_dt = float(flow_record["case"]["production_dt"])
    configurations = _selected_configurations(flow_record)
    rows: list[dict[str, Any]] = []
    for block in range(3):
        ordered = configurations if block % 2 == 0 else list(reversed(configurations))
        for configuration in ordered:
            row = _run_flow_case(
                configuration["method"],
                configuration["degree"],
                configuration["index"],
                configuration["reynolds"],
                production_dt,
                FLOW_END_TIME,
                STATISTICS_START_TIME,
                "selected_timing",
            )
            row["timing_block"] = block
            rows.append(row)
    return {
        "schema": SCHEMA,
        "study": "high_re_matched_accuracy_timing",
        "source_sha256": _source_hashes(),
        "flow_result_sha256": _sha256(FLOW_OUTPUT),
        "configurations": configurations,
        "rows": rows,
        "resource_factors": _timing_factors(flow_record, rows),
    }


def verify_timing_record(
    record: dict[str, Any], flow_record: dict[str, Any]
) -> None:
    """Verify selected timing blocks and matched-accuracy resource factors."""
    if record.get("schema") != SCHEMA or record.get("study") != "high_re_matched_accuracy_timing":
        raise ValueError("matched-accuracy timing schema mismatch")
    if record.get("source_sha256") != _source_hashes():
        raise ValueError("matched-accuracy timing source mismatch")
    if record.get("flow_result_sha256") != _sha256(FLOW_OUTPUT):
        raise ValueError("matched-accuracy timing lineage mismatch")
    if record.get("configurations") != _selected_configurations(flow_record):
        raise ValueError("matched-accuracy timing configurations changed")
    for row in record["rows"]:
        _verify_flow_row(row)
    if record.get("resource_factors") != _timing_factors(flow_record, record["rows"]):
        raise ValueError("matched-accuracy resource factors changed")


def _write(path: Path, record: dict[str, Any]) -> None:
    """Write one finite experiment record without nonstandard JSON values."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def _load(path: Path) -> dict[str, Any]:
    """Load one source-bound experiment record."""
    return json.loads(path.read_text())


def main() -> int:
    """Run or verify one frozen phase of the compatible-flow study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("algebra", "manufactured", "temporal", "flow", "timing", "verify-all"),
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    ensure_generated_dirs()
    if args.phase == "algebra":
        if args.run:
            _write(ALGEBRA_OUTPUT, build_algebra_record())
        verify_algebra_record(_load(ALGEBRA_OUTPUT))
    elif args.phase == "manufactured":
        if args.run:
            _write(MANUFACTURED_OUTPUT, build_manufactured_record())
        verify_manufactured_record(_load(MANUFACTURED_OUTPUT))
    elif args.phase == "temporal":
        if args.run:
            _write(TEMPORAL_OUTPUT, build_temporal_record())
        verify_temporal_record(_load(TEMPORAL_OUTPUT))
    elif args.phase == "flow":
        temporal = _load(TEMPORAL_OUTPUT)
        verify_temporal_record(temporal)
        if args.run:
            _write(FLOW_OUTPUT, build_flow_record(temporal))
        verify_flow_record(_load(FLOW_OUTPUT))
    elif args.phase == "timing":
        flow = _load(FLOW_OUTPUT)
        verify_flow_record(flow)
        if args.run:
            _write(TIMING_OUTPUT, build_timing_record(flow))
        verify_timing_record(_load(TIMING_OUTPUT), flow)
    else:
        verify_algebra_record(_load(ALGEBRA_OUTPUT))
        verify_manufactured_record(_load(MANUFACTURED_OUTPUT))
        verify_temporal_record(_load(TEMPORAL_OUTPUT))
        flow = _load(FLOW_OUTPUT)
        verify_flow_record(flow)
        verify_timing_record(_load(TIMING_OUTPUT), flow)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
