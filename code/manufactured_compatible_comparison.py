#!/usr/bin/env python3
"""Compare BMF and mixed compatible SEM on an exact wall-bounded flow."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np

from compatible_sem_flow import CompatibleSEMFlow2D
from project_paths import CODEDIR, RESULTSDIR, ensure_generated_dirs
from tensor_compatible_flow import (
    TensorCompatibleFlow2D,
    TensorCompatibleSaddle2D,
    VelocityPair,
)


SCHEMA = "manufactured-compatible-comparison-v2"
OUTPUT = RESULTSDIR / "manufactured_compatible_comparison.json"
MASS_SCHUR_REFERENCE = (
    RESULTSDIR / "manufactured_compatible_comparison_mass_schur.json"
)
DEGREES = (2, 4, 6, 8)
ELEMENTS = (1, 2, 4)
ALPHA = 1.0
VISCOSITY = 0.1
RELATIVE_TOLERANCE = 1.0e-11
MAXIMUM_ITERATIONS = 40000
PRESSURE_PRECONDITIONER = "product_schur"
TARGETS = (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-8)
SOURCE_FILES = (
    CODEDIR / "tensor_compatible_flow.py",
    CODEDIR / "compatible_sem_flow.py",
    Path(__file__).resolve(),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {path.name: _sha256(path) for path in SOURCE_FILES}


def _ux(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.pi * np.sin(np.pi * x) * np.cos(np.pi * z)


def _uz(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    return -np.pi * np.cos(np.pi * x) * np.sin(np.pi * z)


def _pressure(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.cos(np.pi * x) * np.cos(np.pi * z)


def _force_ux(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    pressure_x = -np.pi * np.sin(np.pi * x) * np.cos(np.pi * z)
    return (ALPHA + 2.0 * VISCOSITY * np.pi**2) * _ux(x, z) - pressure_x


def _force_uz(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    pressure_z = -np.pi * np.cos(np.pi * x) * np.sin(np.pi * z)
    return (ALPHA + 2.0 * VISCOSITY * np.pi**2) * _uz(x, z) - pressure_z


def _sparse_bytes(matrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _operator_bytes(flow, solver) -> int:
    axis = flow.axis
    factors = (
        axis.b0,
        axis.b1,
        axis.d0,
        axis.d1,
        axis.mass0,
        axis.mass1,
        axis.stiffness0,
        axis.stiffness1,
        axis.incidence,
        flow.tangential_factor,
    )
    lu_factors = (
        solver.factor0.L,
        solver.factor0.U,
        solver.factor1.L,
        solver.factor1.U,
        solver.pressure_mass_factor.L,
        solver.pressure_mass_factor.U,
    )
    arrays = (
        axis.q,
        axis.w,
        axis.trace0,
        axis.trace1,
        axis.normal0,
        axis.normal1,
        solver.pressure_inverse_diagonal,
        solver.pressure_cholesky,
        solver.pressure_schur_vectors,
        solver.pressure_schur_denominator,
    )
    return int(
        sum(_sparse_bytes(matrix) for matrix in factors + lu_factors)
        + sum(array.nbytes for array in arrays if array is not None)
    )


def _traction_error(flow, ux, uz, pressure) -> tuple[float, float]:
    walls = flow.wall_traction(ux, uz, pressure, VISCOSITY)
    q = flow.axis.q
    weights = flow.axis.w
    error_squared = 0.0
    exact_squared = 0.0
    maximum_flux = 0.0
    for name, coordinate, sign, normal_axis in (
        ("x0", 0.0, -1.0, "x"),
        ("x1", 1.0, 1.0, "x"),
        ("z0", 0.0, -1.0, "z"),
        ("z1", 1.0, 1.0, "z"),
    ):
        if normal_axis == "x":
            x = np.full_like(q, coordinate)
            z = q
            pressure_exact = _pressure(x, z)
            ux_x = np.pi**2 * np.cos(np.pi * x) * np.cos(np.pi * z)
            cross = np.zeros_like(q)
            exact_x = sign * (-pressure_exact + 2.0 * VISCOSITY * ux_x)
            exact_z = sign * VISCOSITY * cross
        else:
            x = q
            z = np.full_like(q, coordinate)
            pressure_exact = _pressure(x, z)
            uz_z = -np.pi**2 * np.cos(np.pi * x) * np.cos(np.pi * z)
            cross = np.zeros_like(q)
            exact_x = sign * VISCOSITY * cross
            exact_z = sign * (-pressure_exact + 2.0 * VISCOSITY * uz_z)
        difference_x = walls[name]["traction_x"] - exact_x
        difference_z = walls[name]["traction_z"] - exact_z
        error_squared += float(
            np.sum(weights * (difference_x**2 + difference_z**2))
        )
        exact_squared += float(np.sum(weights * (exact_x**2 + exact_z**2)))
        maximum_flux = max(
            maximum_flux,
            float(np.max(np.abs(walls[name]["normal_flux"]))),
        )
    return math.sqrt(error_squared / exact_squared), maximum_flux


def _row(method: str, degree: int, elements: int) -> dict:
    npts = degree * elements + 1
    setup_start = time.process_time()
    if method == "bmf":
        flow = TensorCompatibleFlow2D(npts, degree)
        exact_sequence_residual = 0.0
    elif method == "compatible_sem":
        flow = CompatibleSEMFlow2D(degree, elements)
        exact_sequence_residual = flow.exact_sequence_residual()
    else:
        raise ValueError(f"unknown method {method}")
    body = flow.body_load(_force_ux, _force_uz)
    wall = flow.wall_load(_ux, _uz)
    rhs = VelocityPair(
        ux=body.ux + VISCOSITY * wall.ux,
        uz=body.uz + VISCOSITY * wall.uz,
    )
    solver = TensorCompatibleSaddle2D(
        flow,
        alpha=ALPHA,
        viscosity=VISCOSITY,
        relative_tolerance=RELATIVE_TOLERANCE,
        maximum_iterations=MAXIMUM_ITERATIONS,
        pressure_preconditioner=PRESSURE_PRECONDITIONER,
    )
    setup_seconds = time.process_time() - setup_start
    solve_start = time.process_time()
    solution = solver.solve(rhs)
    solve_seconds = time.process_time() - solve_start

    velocity = flow.quadrature_velocity(solution.ux, solution.uz)
    primal_pressure = solver.primal_pressure(solution.dual_pressure)
    pressure = flow.quadrature_pressure(primal_pressure)
    x = flow.axis.q[:, None]
    z = flow.axis.q[None, :]
    weight = flow.axis.w[:, None] * flow.axis.w[None, :]
    ux_exact = _ux(x, z)
    uz_exact = _uz(x, z)
    pressure_exact = _pressure(x, z)
    velocity_error_squared = float(
        np.sum(weight * ((velocity.ux - ux_exact) ** 2 + (velocity.uz - uz_exact) ** 2))
    )
    velocity_exact_squared = float(np.sum(weight * (ux_exact**2 + uz_exact**2)))
    pressure_difference = pressure - pressure_exact
    pressure_difference -= float(np.sum(weight * pressure_difference)) / float(
        np.sum(weight)
    )
    pressure_error_squared = float(np.sum(weight * pressure_difference**2))
    pressure_exact_squared = float(np.sum(weight * pressure_exact**2))
    traction_error, maximum_flux = _traction_error(
        flow,
        solution.ux,
        solution.uz,
        primal_pressure,
    )
    flow_storage = flow.storage_record()
    state_entries = flow_storage["velocity_states"] + flow_storage["pressure_states"]
    return {
        "method": method,
        "degree": degree,
        "elements": elements,
        "npts": npts,
        "velocity_states": flow_storage["velocity_states"],
        "pressure_states": flow_storage["pressure_states"],
        "state_storage_bytes": int(8 * state_entries),
        "operator_storage_bytes": _operator_bytes(flow, solver),
        "setup_seconds": setup_seconds,
        "solve_seconds": solve_seconds,
        "total_seconds": setup_seconds + solve_seconds,
        "iterations": solution.iterations,
        "pressure_preconditioner": PRESSURE_PRECONDITIONER,
        "relative_saddle_residual": solution.relative_residual,
        "divergence_relative_residual": solution.divergence_relative_residual,
        "pressure_gauge_relative_residual": solution.pressure_gauge_relative_residual,
        "maximum_normal_flux": maximum_flux,
        "exact_sequence_residual": exact_sequence_residual,
        "relative_velocity_l2_error": math.sqrt(
            velocity_error_squared / velocity_exact_squared
        ),
        "relative_pressure_l2_error": math.sqrt(
            pressure_error_squared / pressure_exact_squared
        ),
        "relative_wall_traction_l2_error": traction_error,
    }


def _selections(rows: list[dict]) -> dict[str, dict[str, dict | None]]:
    result = {}
    for method in ("bmf", "compatible_sem"):
        method_rows = [row for row in rows if row["method"] == method]
        result[method] = {}
        for target in TARGETS:
            admissible = [
                row
                for row in method_rows
                if max(
                    row["relative_velocity_l2_error"],
                    row["relative_pressure_l2_error"],
                    row["relative_wall_traction_l2_error"],
                )
                <= target
            ]
            if not admissible:
                result[method][f"{target:.0e}"] = None
                continue
            selected = min(admissible, key=lambda row: row["total_seconds"])
            result[method][f"{target:.0e}"] = {
                "degree": selected["degree"],
                "elements": selected["elements"],
                "npts": selected["npts"],
                "total_seconds": selected["total_seconds"],
                "state_storage_bytes": selected["state_storage_bytes"],
                "joint_error": max(
                    selected["relative_velocity_l2_error"],
                    selected["relative_pressure_l2_error"],
                    selected["relative_wall_traction_l2_error"],
                ),
            }
    return result


def build_record() -> dict:
    rows = [
        _row(method, degree, elements)
        for degree in DEGREES
        for elements in ELEMENTS
        for method in ("bmf", "compatible_sem")
    ]
    return {
        "schema": SCHEMA,
        "case": {
            "alpha": ALPHA,
            "viscosity": VISCOSITY,
            "degrees": list(DEGREES),
            "elements": list(ELEMENTS),
            "relative_tolerance": RELATIVE_TOLERANCE,
            "maximum_iterations": MAXIMUM_ITERATIONS,
            "pressure_preconditioner": PRESSURE_PRECONDITIONER,
            "joint_targets": list(TARGETS),
            "momentum_convention": "alpha u - nu Delta u - grad p = f",
        },
        "rows": rows,
        "design_selections": _selections(rows),
        "mass_schur_reference_sha256": _sha256(MASS_SCHUR_REFERENCE),
        "source_sha256": _source_hashes(),
    }


def verify_record(record: dict) -> None:
    if record.get("schema") != SCHEMA:
        raise ValueError("manufactured comparison schema mismatch")
    if record.get("source_sha256") != _source_hashes():
        raise ValueError("manufactured comparison source hash mismatch")
    if record.get("mass_schur_reference_sha256") != _sha256(
        MASS_SCHUR_REFERENCE
    ):
        raise ValueError("manufactured mass-Schur reference hash mismatch")
    rows = record.get("rows", [])
    expected_case = {
        "alpha": ALPHA,
        "viscosity": VISCOSITY,
        "degrees": list(DEGREES),
        "elements": list(ELEMENTS),
        "relative_tolerance": RELATIVE_TOLERANCE,
        "maximum_iterations": MAXIMUM_ITERATIONS,
        "pressure_preconditioner": PRESSURE_PRECONDITIONER,
        "joint_targets": list(TARGETS),
        "momentum_convention": "alpha u - nu Delta u - grad p = f",
    }
    if record.get("case") != expected_case:
        raise ValueError("manufactured comparison case definition changed")
    expected = [
        (method, degree, elements)
        for degree in DEGREES
        for elements in ELEMENTS
        for method in ("bmf", "compatible_sem")
    ]
    actual = [(row["method"], row["degree"], row["elements"]) for row in rows]
    if actual != expected:
        raise ValueError("manufactured comparison run matrix changed")
    for row in rows:
        if row.get("pressure_preconditioner") != PRESSURE_PRECONDITIONER:
            raise ValueError("manufactured pressure preconditioner changed")
        if float(row["relative_saddle_residual"]) > RELATIVE_TOLERANCE:
            raise AssertionError("manufactured saddle tolerance failed")
        if float(row["divergence_relative_residual"]) > 2.0e-11:
            raise AssertionError("manufactured divergence tolerance failed")
        if float(row["pressure_gauge_relative_residual"]) > 2.0e-11:
            raise AssertionError("manufactured pressure gauge tolerance failed")
        if float(row["maximum_normal_flux"]) > 2.0e-13:
            raise AssertionError("manufactured no-penetration tolerance failed")
        if row["method"] == "compatible_sem" and float(
            row["exact_sequence_residual"]
        ) > 3.0e-14:
            raise AssertionError("compatible SEM exact sequence failed")
        for key in (
            "relative_velocity_l2_error",
            "relative_pressure_l2_error",
            "relative_wall_traction_l2_error",
            "setup_seconds",
            "solve_seconds",
        ):
            if not math.isfinite(float(row[key])) or float(row[key]) < 0.0:
                raise ValueError(f"non-finite manufactured metric {key}")
    mass_record = json.loads(MASS_SCHUR_REFERENCE.read_text())
    mass_rows = {
        (row["method"], row["degree"], row["elements"]): row
        for row in mass_record["rows"]
    }
    error_keys = (
        "relative_velocity_l2_error",
        "relative_pressure_l2_error",
        "relative_wall_traction_l2_error",
    )
    for row in rows:
        reference = mass_rows[(row["method"], row["degree"], row["elements"])]
        for key in error_keys:
            allowed = max(0.01 * abs(float(reference[key])), 1.0e-10)
            if abs(float(row[key]) - float(reference[key])) > allowed:
                raise AssertionError(
                    f"manufactured discretization error changed for {key}"
                )
    expected_selections = _selections(rows)
    if record.get("design_selections") != expected_selections:
        raise ValueError("manufactured design selection mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    ensure_generated_dirs()
    if args.run:
        record = build_record()
        OUTPUT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    else:
        record = json.loads(OUTPUT.read_text())
    if args.verify or args.run:
        verify_record(record)
    print(json.dumps(record["design_selections"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
