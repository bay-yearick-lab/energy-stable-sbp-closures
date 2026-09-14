#!/usr/bin/env python3
"""Run held-out transient cavities with BMF and compatible SEM."""
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
from tensor_compatible_cavity import TensorCompatibleCavityStepper2D
from tensor_compatible_flow import TensorCompatibleFlow2D


SCHEMA = "heldout-cavity-compatible-comparison-v1"
OUTPUT = RESULTSDIR / "heldout_cavity_compatible_comparison.json"
REYNOLDS_NUMBERS = (100.0, 400.0)
T_END = 4.0
RELATIVE_TOLERANCE = 1.0e-11
MAXIMUM_ITERATIONS = 20000
FIELD_POINTS = np.linspace(0.0, 1.0, 33)
WALL_POINTS = np.linspace(0.0, 1.0, 67)[1:-1]
CONFIGURATIONS = (
    {
        "name": "bmf_p6_k2",
        "method": "bmf",
        "degree": 6,
        "elements": 2,
        "npts": 13,
        "dt": 0.02,
        "reference": False,
    },
    {
        "name": "bmf_p8_k2",
        "method": "bmf",
        "degree": 8,
        "elements": 2,
        "npts": 17,
        "dt": 0.02,
        "reference": False,
    },
    {
        "name": "sem_p8_e2",
        "method": "compatible_sem",
        "degree": 8,
        "elements": 2,
        "npts": 17,
        "dt": 0.02,
        "reference": False,
    },
    {
        "name": "bmf_p8_k3_reference",
        "method": "bmf",
        "degree": 8,
        "elements": 3,
        "npts": 25,
        "dt": 0.01,
        "reference": True,
    },
)
SOURCE_FILES = (
    CODEDIR / "tensor_compatible_flow.py",
    CODEDIR / "tensor_compatible_cavity.py",
    CODEDIR / "compatible_sem_flow.py",
    Path(__file__).resolve(),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {path.name: _sha256(path) for path in SOURCE_FILES}


def _flow(configuration: dict) -> TensorCompatibleFlow2D:
    if configuration["method"] == "bmf":
        return TensorCompatibleFlow2D(
            configuration["npts"], configuration["degree"]
        )
    return CompatibleSEMFlow2D(
        configuration["degree"], configuration["elements"]
    )


def _smooth_lid(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    shape = np.broadcast_shapes(np.shape(x), np.shape(z))
    coordinate = np.broadcast_to(np.asarray(x, dtype=float), shape)
    top = np.broadcast_to(np.isclose(np.asarray(z, dtype=float), 1.0), shape)
    return top * 16.0 * coordinate**2 * (1.0 - coordinate) ** 2


def _sample_traction(traction: dict) -> dict:
    sampled = {}
    for name, wall in traction.items():
        sampled[name] = {
            component: np.interp(WALL_POINTS, wall["coordinate"], wall[component]).tolist()
            for component in ("traction_x", "traction_z", "normal_flux")
        }
    return sampled


def _sample(stepper: TensorCompatibleCavityStepper2D, state) -> dict:
    solver = stepper.bdf2
    pressure_coefficients = solver.primal_pressure(state.dual_pressure)
    velocity = stepper.flow.evaluate_velocity(
        state.ux,
        state.uz,
        FIELD_POINTS,
        FIELD_POINTS,
    )
    pressure = stepper.flow.evaluate_pressure(
        pressure_coefficients,
        FIELD_POINTS,
        FIELD_POINTS,
    )
    pressure -= float(np.mean(pressure))
    traction = stepper.strong_wall_traction(state)
    numerical_traction = stepper.nitsche_consistent_wall_traction(state)
    return {
        "ux": velocity.ux.tolist(),
        "uz": velocity.uz.tolist(),
        "pressure": pressure.tolist(),
        "traction": _sample_traction(traction),
        "nitsche_consistent_traction": _sample_traction(numerical_traction),
    }


def _run_case(reynolds: float, configuration: dict) -> dict:
    setup_start = time.process_time()
    flow = _flow(configuration)
    lid_velocity = (
        _smooth_lid if configuration.get("lid_profile") == "smooth_quartic" else None
    )
    stepper = TensorCompatibleCavityStepper2D(
        npts=configuration["npts"],
        degree=configuration["degree"],
        reynolds=reynolds,
        dt=configuration["dt"],
        relative_tolerance=RELATIVE_TOLERANCE,
        maximum_iterations=MAXIMUM_ITERATIONS,
        flow=flow,
        lid_velocity=lid_velocity,
    )
    setup_seconds = time.process_time() - setup_start
    steps = int(round(T_END / configuration["dt"]))
    state = stepper.initial_state()
    maximum_residual = 0.0
    maximum_divergence = 0.0
    maximum_gauge = 0.0
    maximum_energy_balance = 0.0
    maximum_normal_flux = 0.0
    integrated_nitsche_wall_work = 0.0
    integrated_net_wall_work = 0.0
    integrated_lid_reaction_work = 0.0
    integrated_numerical_traction_work = 0.0
    iterations = []
    recurring_start = time.process_time()
    for _ in range(steps):
        old_state = state
        result = stepper.step(state)
        state = result.state
        iterations.append(result.saddle.iterations)
        maximum_residual = max(maximum_residual, result.saddle.relative_residual)
        maximum_divergence = max(
            maximum_divergence, result.saddle.divergence_relative_residual
        )
        maximum_gauge = max(
            maximum_gauge, result.saddle.pressure_gauge_relative_residual
        )
        maximum_energy_balance = max(
            maximum_energy_balance, result.energy["relative_balance_residual"]
        )
        integrated_nitsche_wall_work += (
            result.energy["nitsche_wall_power"] * configuration["dt"]
        )
        integrated_net_wall_work += (
            result.energy["net_wall_power"] * configuration["dt"]
        )
        wall_observable = stepper.native_wall_observable(state)
        integrated_lid_reaction_work += (
            wall_observable["lid_power"] * configuration["dt"]
        )
        numerical_observable = stepper.nitsche_wall_observable(state)
        integrated_numerical_traction_work += (
            numerical_observable["lid_power"] * configuration["dt"]
        )
        maximum_normal_flux = max(
            maximum_normal_flux,
            wall_observable["maximum_normal_flux"],
        )
    recurring_seconds = time.process_time() - recurring_start
    observable_samples = []
    for _ in range(50):
        start = time.process_time()
        observable = stepper.native_wall_observable(state)
        observable_samples.append(time.process_time() - start)
    flow_storage = flow.storage_record()
    return {
        **configuration,
        "reynolds": reynolds,
        "steps": steps,
        "final_time": state.time,
        "setup_seconds": setup_seconds,
        "recurring_seconds": recurring_seconds,
        "total_seconds": setup_seconds + recurring_seconds,
        "median_iterations": float(np.median(iterations)),
        "maximum_iterations": int(max(iterations)),
        "total_iterations": int(sum(iterations)),
        "final_velocity_rate": stepper.velocity_rate(state, old_state),
        "maximum_relative_residual": maximum_residual,
        "maximum_divergence_relative_residual": maximum_divergence,
        "maximum_pressure_gauge_relative_residual": maximum_gauge,
        "maximum_energy_balance_residual": maximum_energy_balance,
        "maximum_normal_flux": maximum_normal_flux,
        "integrated_nitsche_wall_work": integrated_nitsche_wall_work,
        "integrated_net_wall_work": integrated_net_wall_work,
        "integrated_lid_reaction_work": integrated_lid_reaction_work,
        "integrated_numerical_traction_work": integrated_numerical_traction_work,
        "lid_reaction": observable["lid_reaction"],
        "numerical_lid_reaction": numerical_observable["lid_reaction"],
        "median_wall_observable_seconds": float(np.median(observable_samples)),
        "state_storage_bytes": int(
            8 * (flow_storage["velocity_states"] + flow_storage["pressure_states"])
        ),
        "operator_storage": stepper.storage_record(),
        "samples": _sample(stepper, state),
    }


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right)) / max(
        float(np.linalg.norm(right)), 1.0e-300
    )


def _errors(row: dict, reference: dict) -> dict[str, float]:
    samples = row["samples"]
    reference_samples = reference["samples"]
    velocity_error = math.sqrt(
        _relative(np.asarray(samples["ux"]), np.asarray(reference_samples["ux"])) ** 2
        + _relative(np.asarray(samples["uz"]), np.asarray(reference_samples["uz"])) ** 2
    )
    pressure_error = _relative(
        np.asarray(samples["pressure"]), np.asarray(reference_samples["pressure"])
    )
    traction_left = []
    traction_right = []
    for wall in ("x0", "x1", "z0", "z1"):
        for component in ("traction_x", "traction_z"):
            traction_left.extend(samples["traction"][wall][component])
            traction_right.extend(reference_samples["traction"][wall][component])
    traction_error = _relative(
        np.asarray(traction_left), np.asarray(traction_right)
    )
    lid_reaction_error = abs(
        float(row["lid_reaction"]) - float(reference["lid_reaction"])
    ) / max(abs(float(reference["lid_reaction"])), 1.0e-300)
    return {
        "relative_velocity_error": velocity_error,
        "relative_pressure_error": pressure_error,
        "relative_full_wall_traction_diagnostic": traction_error,
        "relative_lid_reaction_error": lid_reaction_error,
        "joint_error": max(velocity_error, pressure_error, lid_reaction_error),
    }


def build_record() -> dict:
    rows = [
        _run_case(reynolds, configuration)
        for reynolds in REYNOLDS_NUMBERS
        for configuration in CONFIGURATIONS
    ]
    for reynolds in REYNOLDS_NUMBERS:
        reference = next(
            row for row in rows if row["reynolds"] == reynolds and row["reference"]
        )
        reference["errors"] = {
            "relative_velocity_error": 0.0,
            "relative_pressure_error": 0.0,
            "relative_full_wall_traction_diagnostic": 0.0,
            "relative_lid_reaction_error": 0.0,
            "joint_error": 0.0,
        }
        for row in rows:
            if row["reynolds"] == reynolds and not row["reference"]:
                row["errors"] = _errors(row, reference)
    return {
        "schema": SCHEMA,
        "case": {
            "reynolds_numbers": list(REYNOLDS_NUMBERS),
            "t_end": T_END,
            "relative_tolerance": RELATIVE_TOLERANCE,
            "maximum_iterations": MAXIMUM_ITERATIONS,
            "field_points": FIELD_POINTS.tolist(),
            "wall_points": WALL_POINTS.tolist(),
            "configurations": list(CONFIGURATIONS),
        },
        "source_sha256": _source_hashes(),
        "rows": rows,
    }


def verify_record(record: dict) -> None:
    if record.get("schema") != SCHEMA:
        raise ValueError("held-out cavity schema mismatch")
    if record.get("source_sha256") != _source_hashes():
        raise ValueError("held-out cavity source hash mismatch")
    rows = record.get("rows", [])
    expected = [
        (reynolds, configuration["name"])
        for reynolds in REYNOLDS_NUMBERS
        for configuration in CONFIGURATIONS
    ]
    if [(row["reynolds"], row["name"]) for row in rows] != expected:
        raise ValueError("held-out cavity run matrix changed")
    for row in rows:
        if float(row["maximum_relative_residual"]) > RELATIVE_TOLERANCE:
            raise AssertionError("held-out cavity saddle residual failed")
        if float(row["maximum_divergence_relative_residual"]) > 2.0e-11:
            raise AssertionError("held-out cavity divergence failed")
        if float(row["maximum_pressure_gauge_relative_residual"]) > 2.0e-11:
            raise AssertionError("held-out cavity gauge failed")
        if float(row["maximum_energy_balance_residual"]) > 5.0e-9:
            raise AssertionError("held-out cavity energy ledger failed")
        if float(row["maximum_normal_flux"]) > 2.0e-13:
            raise AssertionError("held-out cavity no-penetration failed")
        if int(row["operator_storage"]["multidimensional_dense_operator_entries"]) != 0:
            raise AssertionError("held-out cavity stored a dense operator")
        if not math.isclose(float(row["final_time"]), T_END):
            raise AssertionError("held-out cavity final time changed")
        if not row["reference"]:
            expected_errors = _errors(
                row,
                next(
                    reference
                    for reference in rows
                    if reference["reynolds"] == row["reynolds"]
                    and reference["reference"]
                ),
            )
            for key, value in expected_errors.items():
                if not np.isclose(float(row["errors"][key]), value):
                    raise ValueError("held-out cavity error summary mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    ensure_generated_dirs()
    if args.run:
        if OUTPUT.exists():
            raise FileExistsError(f"refusing to overwrite {OUTPUT}")
        record = build_record()
        OUTPUT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    else:
        record = json.loads(OUTPUT.read_text())
    if args.run or args.verify:
        verify_record(record)
    summary = [
        {
            "reynolds": row["reynolds"],
            "name": row["name"],
            "time": row["total_seconds"],
            "storage": row["state_storage_bytes"],
            "joint_error": row["errors"]["joint_error"],
        }
        for row in record["rows"]
        if not row["reference"]
    ]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
