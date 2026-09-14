#!/usr/bin/env python3
"""Profile the complete pressure-coupled tensor-compatible cavity step."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import sys
import time
from typing import Callable

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np

from project_paths import CODEDIR, RESULTSDIR, ensure_generated_dirs
from tensor_compatible_cavity import TensorCompatibleCavityStepper2D


SCHEMA = "tensor-compatible-pressure-step-v2"
OUTPUT = RESULTSDIR / "tensor_compatible_pressure_step.json"
GRIDS = (9, 17, 33, 65)
DEGREE = 4
REYNOLDS = 100.0
DT = 0.02
RELATIVE_TOLERANCE = 1.0e-10
MAXIMUM_ITERATIONS = 4000
SOURCE_FILES = (
    CODEDIR / "tensor_compatible_flow.py",
    CODEDIR / "tensor_compatible_cavity.py",
    Path(__file__).resolve(),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {path.name: _sha256(path) for path in SOURCE_FILES}


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def _timing_stamp() -> dict:
    return {
        "metric": "process_time_seconds",
        "cpu": platform.processor() or _cpu_model(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "thread_env": {
            name: os.environ.get(name, "unset")
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def _median_time(function: Callable[[], object], repeats: int) -> float:
    function()
    samples = []
    for _ in range(int(repeats)):
        start = time.process_time()
        function()
        samples.append(time.process_time() - start)
    return float(np.median(samples))


def _profile_row(npts: int, pressure_preconditioner: str) -> dict:
    setup_start = time.process_time()
    stepper = TensorCompatibleCavityStepper2D(
        npts=npts,
        degree=DEGREE,
        reynolds=REYNOLDS,
        dt=DT,
        relative_tolerance=RELATIVE_TOLERANCE,
        maximum_iterations=MAXIMUM_ITERATIONS,
        pressure_preconditioner=pressure_preconditioner,
    )
    setup_seconds = time.process_time() - setup_start
    state = stepper.initial_state()
    try:
        for _ in range(2):
            state = stepper.step(state).state
    except RuntimeError as error:
        return {
            "npts": npts,
            "pressure_preconditioner": pressure_preconditioner,
            "status": "iteration_limit",
            "setup_seconds": setup_seconds,
            "error": str(error),
            "maximum_iterations": MAXIMUM_ITERATIONS,
            "storage": stepper.storage_record(),
        }

    convection = stepper.convection(state)
    rhs = stepper.assemble_rhs(state, convection)
    saddle = stepper.solve_rhs(state, rhs)
    complete = stepper.step(state)
    observable = stepper.native_wall_observable(complete.state)
    repeats = 3
    nonlinear_seconds = _median_time(lambda: stepper.convection(state), repeats)
    rhs_seconds = _median_time(
        lambda: stepper.assemble_rhs(state, convection), repeats
    )
    saddle_seconds = _median_time(lambda: stepper.solve_rhs(state, rhs), repeats)
    complete_seconds = _median_time(lambda: stepper.step(state), repeats)
    wall_seconds = _median_time(
        lambda: stepper.native_wall_observable(complete.state), repeats
    )
    component_sum = nonlinear_seconds + rhs_seconds + saddle_seconds
    return {
        "npts": npts,
        "pressure_preconditioner": pressure_preconditioner,
        "status": "complete",
        "setup_seconds": setup_seconds,
        "nonlinear_seconds": nonlinear_seconds,
        "rhs_assembly_seconds": rhs_seconds,
        "saddle_seconds": saddle_seconds,
        "complete_step_seconds": complete_seconds,
        "wall_observable_seconds": wall_seconds,
        "component_sum_seconds": component_sum,
        "saddle_fraction_of_component_sum": saddle_seconds / component_sum,
        "complete_to_component_sum_ratio": complete_seconds / component_sum,
        "iterations": saddle.iterations,
        "relative_residual": saddle.relative_residual,
        "divergence_relative_residual": saddle.divergence_relative_residual,
        "pressure_gauge_relative_residual": saddle.pressure_gauge_relative_residual,
        "energy_relative_balance_residual": complete.energy[
            "relative_balance_residual"
        ],
        "maximum_normal_flux": observable["maximum_normal_flux"],
        "lid_reaction": observable["lid_reaction"],
        "storage": stepper.storage_record(),
    }


def _fit_exponent(rows: list[dict], key: str) -> float:
    selected = [row for row in rows if row["status"] == "complete"][-3:]
    if len(selected) < 3:
        return math.nan
    x = np.log(np.asarray([row["npts"] for row in selected], dtype=float))
    y = np.log(np.asarray([row[key] for row in selected], dtype=float))
    return float(np.polyfit(x, y, 1)[0])


def run() -> dict:
    diagonal = []
    for npts in GRIDS:
        row = _profile_row(npts, "diagonal")
        diagonal.append(row)
        if row["status"] != "complete":
            break
    mass_schur = [_profile_row(npts, "mass_schur") for npts in GRIDS]
    product_schur = [_profile_row(npts, "product_schur") for npts in GRIDS]
    largest_mass = mass_schur[-1]
    largest_product = product_schur[-1]
    result = {
        "schema": SCHEMA,
        "case": {
            "grids": list(GRIDS),
            "degree": DEGREE,
            "reynolds": REYNOLDS,
            "dt": DT,
            "relative_tolerance": RELATIVE_TOLERANCE,
            "maximum_iterations": MAXIMUM_ITERATIONS,
            "startup_steps": 2,
            "timed_repeats": 3,
        },
        "timing": _timing_stamp(),
        "diagonal": diagonal,
        "mass_schur": mass_schur,
        "product_schur": product_schur,
        "summary": {
            "mass_schur_saddle_exponent": _fit_exponent(
                mass_schur, "saddle_seconds"
            ),
            "mass_schur_complete_step_exponent": _fit_exponent(
                mass_schur, "complete_step_seconds"
            ),
            "mass_schur_iteration_exponent": _fit_exponent(
                mass_schur, "iterations"
            ),
            "product_schur_saddle_exponent": _fit_exponent(
                product_schur, "saddle_seconds"
            ),
            "product_schur_complete_step_exponent": _fit_exponent(
                product_schur, "complete_step_seconds"
            ),
            "product_schur_iteration_exponent": _fit_exponent(
                product_schur, "iterations"
            ),
            "largest_mass_schur_saddle_fraction": largest_mass[
                "saddle_fraction_of_component_sum"
            ],
            "largest_product_schur_saddle_fraction": largest_product[
                "saddle_fraction_of_component_sum"
            ],
            "largest_iteration_reduction": largest_mass["iterations"]
            / largest_product["iterations"],
            "largest_saddle_speedup": largest_mass["saddle_seconds"]
            / largest_product["saddle_seconds"],
            "diagonal_terminal_status": diagonal[-1]["status"],
        },
        "source_sha256": _source_hashes(),
    }
    ensure_generated_dirs()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def verify(result: dict) -> None:
    if result.get("schema") != SCHEMA:
        raise RuntimeError("pressure-step schema mismatch")
    if result.get("source_sha256") != _source_hashes():
        raise RuntimeError("pressure-step source hash mismatch")
    expected_case = {
        "grids": list(GRIDS),
        "degree": DEGREE,
        "reynolds": REYNOLDS,
        "dt": DT,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "maximum_iterations": MAXIMUM_ITERATIONS,
        "startup_steps": 2,
        "timed_repeats": 3,
    }
    if result.get("case") != expected_case:
        raise RuntimeError("pressure-step case definition changed")
    preconditioned_rows = {}
    for name in ("mass_schur", "product_schur"):
        rows = result.get(name, [])
        preconditioned_rows[name] = rows
        if [row.get("npts") for row in rows] != list(GRIDS):
            raise RuntimeError(f"{name} run matrix is incomplete")
        for row in rows:
            if row.get("status") != "complete":
                raise RuntimeError(f"{name} pressure step did not complete")
            if float(row["relative_residual"]) > RELATIVE_TOLERANCE:
                raise RuntimeError("pressure-step residual tolerance failed")
            if float(row["divergence_relative_residual"]) > 2.0e-11:
                raise RuntimeError("pressure-step divergence tolerance failed")
            if float(row["pressure_gauge_relative_residual"]) > 2.0e-11:
                raise RuntimeError("pressure-step gauge tolerance failed")
            if float(row["energy_relative_balance_residual"]) > 5.0e-9:
                raise RuntimeError("pressure-step energy tolerance failed")
            if float(row["maximum_normal_flux"]) > 2.0e-13:
                raise RuntimeError("pressure-step no-penetration tolerance failed")
            if int(row["storage"]["multidimensional_dense_operator_entries"]) != 0:
                raise RuntimeError("pressure-step route stored a dense operator")
            ratio = float(row["complete_to_component_sum_ratio"])
            if not 0.5 <= ratio <= 2.0:
                raise RuntimeError("complete-step time contradicts component profile")
    mass_rows = preconditioned_rows["mass_schur"]
    product_rows = preconditioned_rows["product_schur"]
    if any(
        int(product["iterations"]) >= int(mass["iterations"])
        for mass, product in zip(mass_rows, product_rows)
    ):
        raise RuntimeError("product-Schur iteration prediction failed")
    if float(product_rows[-1]["saddle_seconds"]) >= float(
        mass_rows[-1]["saddle_seconds"]
    ):
        raise RuntimeError("product-Schur largest-cell timing prediction failed")
    summary = result.get("summary", {})
    for name, rows in preconditioned_rows.items():
        for suffix, metric in (
            ("saddle_exponent", "saddle_seconds"),
            ("complete_step_exponent", "complete_step_seconds"),
            ("iteration_exponent", "iterations"),
        ):
            key = f"{name}_{suffix}"
            expected = _fit_exponent(rows, metric)
            if not np.isclose(
                float(summary[key]), expected, rtol=1.0e-12, atol=1.0e-12
            ):
                raise RuntimeError(f"pressure-step summary mismatch for {key}")
    largest_mass = mass_rows[-1]
    largest_product = product_rows[-1]
    summary_checks = {
        "largest_mass_schur_saddle_fraction": largest_mass[
            "saddle_fraction_of_component_sum"
        ],
        "largest_product_schur_saddle_fraction": largest_product[
            "saddle_fraction_of_component_sum"
        ],
        "largest_iteration_reduction": largest_mass["iterations"]
        / largest_product["iterations"],
        "largest_saddle_speedup": largest_mass["saddle_seconds"]
        / largest_product["saddle_seconds"],
    }
    for key, expected in summary_checks.items():
        if not np.isclose(
            float(summary[key]), float(expected), rtol=1.0e-12, atol=1.0e-12
        ):
            raise RuntimeError(f"pressure-step summary mismatch for {key}")
    diagonal = result.get("diagonal", [])
    if not diagonal or diagonal[-1].get("status") != "iteration_limit":
        raise RuntimeError("diagonal pressure trigger was not preserved")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.run:
        result = run()
    else:
        result = json.loads(OUTPUT.read_text())
    if args.verify or not args.run:
        verify(result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
