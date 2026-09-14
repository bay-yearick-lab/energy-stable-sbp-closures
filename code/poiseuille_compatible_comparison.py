#!/usr/bin/env python3
"""Compare BMF and compatible SEM on analytic planar Poiseuille flow."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
from scipy.sparse.linalg import splu

from compatible_sem_flow import CompatibleSEMFlow2D
from project_paths import CODEDIR, RESULTSDIR, ensure_generated_dirs
from tensor_compatible_flow import TensorCompatibleFlow2D


SCHEMA = "poiseuille-compatible-comparison-v1"
OUTPUT = RESULTSDIR / "poiseuille_compatible_comparison.json"
DEGREES = (2, 4, 6, 8)
RESOLUTION_INDICES = (1, 2, 4)
VISCOSITY = 0.1
PRESSURE_GRADIENT_MAGNITUDE = 1.0
FIELD_TOLERANCE = 1.0e-11
OBSERVABLE_BLOCKS = 5
OBSERVABLE_REPETITIONS = 2000
TARGETS = (1.0e-2, 1.0e-4, 1.0e-8, 1.0e-11)
SOURCE_FILES = (
    CODEDIR / "tensor_compatible_flow.py",
    CODEDIR / "compatible_sem_flow.py",
    Path(__file__).resolve(),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {path.name: _sha256(path) for path in SOURCE_FILES}


def _sparse_bytes(matrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _exact_velocity(z: np.ndarray) -> np.ndarray:
    return PRESSURE_GRADIENT_MAGNITUDE * z * (1.0 - z) / (2.0 * VISCOSITY)


def _native_observables(flow, coefficients: np.ndarray) -> tuple[float, np.ndarray]:
    axis = flow.axis
    velocity = np.asarray(axis.b1 @ coefficients).ravel()
    flux = float(np.sum(axis.w * velocity))
    traction = VISCOSITY * np.asarray(axis.normal1 @ coefficients).ravel()
    return flux, traction


def _observable_timing(flow, coefficients: np.ndarray) -> tuple[list[float], float]:
    samples: list[float] = []
    checksum = 0.0
    for _block in range(OBSERVABLE_BLOCKS):
        start = time.process_time()
        for _repeat in range(OBSERVABLE_REPETITIONS):
            flux, traction = _native_observables(flow, coefficients)
            checksum += flux + float(np.sum(traction))
        samples.append(
            (time.process_time() - start) / OBSERVABLE_REPETITIONS
        )
    return samples, checksum


def _row(method: str, degree: int, resolution_index: int) -> dict:
    npts = degree * resolution_index + 1
    setup_start = time.process_time()
    if method == "bmf":
        flow = TensorCompatibleFlow2D(npts, degree)
        partition_count = npts - degree
    elif method == "compatible_sem":
        flow = CompatibleSEMFlow2D(degree, resolution_index)
        partition_count = resolution_index
    else:
        raise ValueError(f"unknown method {method}")
    axis = flow.axis
    operator = (VISCOSITY * flow.tangential_factor).tocsc()
    factor = splu(operator)
    forcing = PRESSURE_GRADIENT_MAGNITUDE * np.asarray(
        axis.b1.T @ axis.w
    ).ravel()
    setup_seconds = time.process_time() - setup_start

    solve_start = time.process_time()
    coefficients = factor.solve(forcing)
    solve_seconds = time.process_time() - solve_start
    residual = np.asarray(operator @ coefficients).ravel() - forcing
    relative_residual = float(
        np.linalg.norm(residual) / max(np.linalg.norm(forcing), 1.0e-300)
    )

    velocity = np.asarray(axis.b1 @ coefficients).ravel()
    exact_velocity = _exact_velocity(axis.q)
    relative_velocity_error = math.sqrt(
        float(np.sum(axis.w * (velocity - exact_velocity) ** 2))
        / float(np.sum(axis.w * exact_velocity**2))
    )
    flux, traction = _native_observables(flow, coefficients)
    exact_flux = PRESSURE_GRADIENT_MAGNITUDE / (12.0 * VISCOSITY)
    exact_traction = np.full(2, -0.5 * PRESSURE_GRADIENT_MAGNITUDE)
    relative_flux_error = abs(flux - exact_flux) / abs(exact_flux)
    relative_traction_error = float(
        np.linalg.norm(traction - exact_traction)
        / np.linalg.norm(exact_traction)
    )
    observable_samples, checksum = _observable_timing(flow, coefficients)
    stored_factor_entries = int(
        axis.b1.nnz
        + axis.d1.nnz
        + axis.mass1.nnz
        + axis.stiffness1.nnz
        + flow.tangential_factor.nnz
    )
    factor_bytes = sum(
        _sparse_bytes(matrix)
        for matrix in (
            axis.b1,
            axis.d1,
            axis.mass1,
            axis.stiffness1,
            flow.tangential_factor,
            factor.L,
            factor.U,
        )
    )
    joint_error = max(
        relative_velocity_error,
        relative_flux_error,
        relative_traction_error,
    )
    return {
        "method": method,
        "degree": degree,
        "resolution_index": resolution_index,
        "partition_count": partition_count,
        "state_count": int(coefficients.size),
        "state_bytes": int(coefficients.nbytes),
        "stored_factor_entries": stored_factor_entries,
        "operator_and_factor_bytes": int(factor_bytes),
        "setup_seconds": setup_seconds,
        "solve_seconds": solve_seconds,
        "relative_weak_residual": relative_residual,
        "relative_velocity_l2_error": relative_velocity_error,
        "relative_flux_error": relative_flux_error,
        "relative_wall_traction_error": relative_traction_error,
        "joint_error": joint_error,
        "computed_flux": flux,
        "computed_wall_traction": traction.tolist(),
        "pressure_gradient_error": 0.0,
        "observable_seconds_per_query_samples": observable_samples,
        "observable_seconds_per_query_median": statistics.median(
            observable_samples
        ),
        "observable_checksum": checksum,
    }


def _selections(rows: list[dict]) -> dict[str, dict[str, dict | None]]:
    selections: dict[str, dict[str, dict | None]] = {}
    for target in TARGETS:
        by_method: dict[str, dict | None] = {}
        for method in ("bmf", "compatible_sem"):
            feasible = [
                row
                for row in rows
                if row["method"] == method and row["joint_error"] <= target
            ]
            if not feasible:
                by_method[method] = None
                continue
            selected = min(
                feasible,
                key=lambda row: (
                    row["state_count"],
                    row["setup_seconds"] + row["solve_seconds"],
                    row["joint_error"],
                ),
            )
            by_method[method] = {
                key: selected[key]
                for key in (
                    "degree",
                    "resolution_index",
                    "partition_count",
                    "state_count",
                    "state_bytes",
                    "joint_error",
                    "setup_seconds",
                    "solve_seconds",
                )
            }
        selections[f"{target:.0e}"] = by_method
    return selections


def build_record() -> dict:
    rows = [
        _row(method, degree, resolution_index)
        for degree in DEGREES
        for resolution_index in RESOLUTION_INDICES
        for method in ("bmf", "compatible_sem")
    ]
    return {
        "schema": SCHEMA,
        "case": {
            "domain": [0.0, 1.0],
            "viscosity": VISCOSITY,
            "pressure_gradient_magnitude": PRESSURE_GRADIENT_MAGNITUDE,
            "exact_velocity": "G z (1-z) / (2 nu)",
            "exact_pressure_gradient": "dp/dx = -G",
            "exact_flux": PRESSURE_GRADIENT_MAGNITUDE / (12.0 * VISCOSITY),
            "exact_outward_wall_traction": [
                -0.5 * PRESSURE_GRADIENT_MAGNITUDE,
                -0.5 * PRESSURE_GRADIENT_MAGNITUDE,
            ],
            "degrees": list(DEGREES),
            "resolution_indices": list(RESOLUTION_INDICES),
            "field_tolerance": FIELD_TOLERANCE,
            "joint_targets": list(TARGETS),
            "observable_blocks": OBSERVABLE_BLOCKS,
            "observable_repetitions_per_block": OBSERVABLE_REPETITIONS,
        },
        "rows": rows,
        "design_selections": _selections(rows),
        "source_sha256": _source_hashes(),
    }


def verify_record(record: dict) -> None:
    if record.get("schema") != SCHEMA:
        raise ValueError("Poiseuille comparison schema mismatch")
    if record.get("source_sha256") != _source_hashes():
        raise ValueError("Poiseuille comparison source hash mismatch")
    expected_case = {
        "domain": [0.0, 1.0],
        "viscosity": VISCOSITY,
        "pressure_gradient_magnitude": PRESSURE_GRADIENT_MAGNITUDE,
        "exact_velocity": "G z (1-z) / (2 nu)",
        "exact_pressure_gradient": "dp/dx = -G",
        "exact_flux": PRESSURE_GRADIENT_MAGNITUDE / (12.0 * VISCOSITY),
        "exact_outward_wall_traction": [
            -0.5 * PRESSURE_GRADIENT_MAGNITUDE,
            -0.5 * PRESSURE_GRADIENT_MAGNITUDE,
        ],
        "degrees": list(DEGREES),
        "resolution_indices": list(RESOLUTION_INDICES),
        "field_tolerance": FIELD_TOLERANCE,
        "joint_targets": list(TARGETS),
        "observable_blocks": OBSERVABLE_BLOCKS,
        "observable_repetitions_per_block": OBSERVABLE_REPETITIONS,
    }
    if record.get("case") != expected_case:
        raise ValueError("Poiseuille case definition changed")
    rows = record.get("rows", [])
    expected = [
        (method, degree, resolution_index)
        for degree in DEGREES
        for resolution_index in RESOLUTION_INDICES
        for method in ("bmf", "compatible_sem")
    ]
    actual = [
        (row["method"], row["degree"], row["resolution_index"])
        for row in rows
    ]
    if actual != expected:
        raise ValueError("Poiseuille run matrix changed")
    for row in rows:
        expected_states = row["degree"] * row["resolution_index"]
        if row["state_count"] != expected_states:
            raise AssertionError("Poiseuille state matching failed")
        if row["state_bytes"] != 8 * expected_states:
            raise AssertionError("Poiseuille state storage changed")
        if row["relative_weak_residual"] > FIELD_TOLERANCE:
            raise AssertionError("Poiseuille weak residual tolerance failed")
        if row["pressure_gradient_error"] != 0.0:
            raise AssertionError("Poiseuille pressure gradient changed")
        for key in (
            "relative_velocity_l2_error",
            "relative_flux_error",
            "relative_wall_traction_error",
            "joint_error",
            "setup_seconds",
            "solve_seconds",
            "observable_seconds_per_query_median",
        ):
            value = float(row[key])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"non-finite Poiseuille metric {key}")
        samples = row["observable_seconds_per_query_samples"]
        if len(samples) != OBSERVABLE_BLOCKS:
            raise ValueError("Poiseuille observable block count changed")
        if any(not math.isfinite(float(value)) or value <= 0.0 for value in samples):
            raise ValueError("invalid Poiseuille observable timing")
        if row["degree"] >= 4 and row["joint_error"] > FIELD_TOLERANCE:
            raise AssertionError("Poiseuille polynomial reproduction failed")
    indexed = {
        (row["method"], row["degree"], row["resolution_index"]): row
        for row in rows
    }
    for degree in DEGREES:
        bmf = indexed[("bmf", degree, 1)]
        sem = indexed[("compatible_sem", degree, 1)]
        if abs(bmf["joint_error"] - sem["joint_error"]) > FIELD_TOLERANCE:
            raise AssertionError("Poiseuille single-partition equivalence failed")
    if record.get("design_selections") != _selections(rows):
        raise ValueError("Poiseuille design selection mismatch")


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
    if args.run or args.verify:
        verify_record(record)
    print(json.dumps(record["design_selections"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
