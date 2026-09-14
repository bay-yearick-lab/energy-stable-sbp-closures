#!/usr/bin/env python3
"""Single-thread process-time certificate for the full wall propagator."""
from __future__ import annotations

import os

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm

import s6_wall_spectrum_full_propagator as predictor
import s6_wall_spectrum_heldout as base
import s6_wall_spectrum_heldout_runner as runner


OUTPUT = base.RESULTSDIR / "wall_spectrum_full_propagator_cost_certificate.json"
REPETITIONS = 7
APPLICATION_LOOPS = 2000
TIME_VALUE = 7.0


def build_propagator(decomposition: dict, time_value: float) -> np.ndarray:
    """Assemble one full-state propagator from an existing Schur factorization."""
    exponential = expm(decomposition["schur_form"] * time_value)
    reduced = (
        decomposition["inverse_square_root"]
        @ decomposition["schur_vectors"]
        @ exponential
        @ decomposition["schur_vectors"].T
        @ decomposition["square_root"]
    )
    return (
        decomposition["injection"]
        @ reduced
        @ decomposition["restriction"]
    )


def median(values: list[float]) -> float:
    """Return the median of seven recorded process times."""
    return float(np.median(np.asarray(values, dtype=float)))


def certify_method(method: str) -> dict:
    """Measure factorization, new-time construction, and packet application."""
    predictor.factor(method)
    factor_times = []
    for _repeat in range(REPETITIONS):
        start = time.process_time()
        decomposition = predictor.factor(method)
        factor_times.append(time.process_time() - start)

    data = decomposition["data"]
    packet_specs = [(sigma, carrier) for sigma in predictor.SIGMAS for carrier in predictor.CARRIERS]
    states = runner.initial_states(data, packet_specs)
    build_propagator(decomposition, TIME_VALUE)
    build_times = []
    propagator = None
    for _repeat in range(REPETITIONS):
        start = time.process_time()
        propagator = build_propagator(decomposition, TIME_VALUE)
        build_times.append(time.process_time() - start)

    assert propagator is not None
    for _warmup in range(100):
        propagator @ states
    application_times = []
    for _repeat in range(REPETITIONS):
        start = time.process_time()
        for _loop in range(APPLICATION_LOOPS):
            propagator @ states
        elapsed = time.process_time() - start
        application_times.append(elapsed / (APPLICATION_LOOPS * states.shape[1]))

    return {
        "method": method,
        "full_state_dimension": int(data["matrix"].shape[0]),
        "factored_state_dimension": int(decomposition["matrix"].shape[0]),
        "packet_batch_size": int(states.shape[1]),
        "factorization_process_seconds": factor_times,
        "factorization_median_process_seconds": median(factor_times),
        "per_time_build_process_seconds": build_times,
        "per_time_build_median_process_seconds": median(build_times),
        "per_packet_application_process_seconds": application_times,
        "per_packet_application_median_process_seconds": median(application_times),
    }


def write_certificate(output: Path) -> dict:
    """Write the complete fixed-protocol timing certificate."""
    if output.exists():
        raise FileExistsError(output)
    methods = [certify_method(method) for method in base.METHODS]
    record = {
        "schema": "wall-spectrum-full-propagator-cost-v1",
        "status": "post_result_performance_certificate",
        "scientific_acceptance_unchanged": True,
        "environment": {
            "thread_variables": {
                variable: os.environ[variable]
                for variable in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
            "clock": "time.process_time",
            "warmup": "one complete factorization, one propagator build, and 100 batched applications per method",
            "repetitions": REPETITIONS,
            "application_loops_per_repetition": APPLICATION_LOOPS,
            "propagator_time": TIME_VALUE,
        },
        "asymptotic_cost": {
            "factorization": "O(n^3) once per enforced operator",
            "new_time_propagator": "O(n^3) per requested time",
            "packet_application": "O(n^2) per packet",
            "storage": "O(n^2)",
        },
        "methods": methods,
        "median_ranges": {
            "factorization_process_seconds": [
                min(row["factorization_median_process_seconds"] for row in methods),
                max(row["factorization_median_process_seconds"] for row in methods),
            ],
            "per_time_build_process_seconds": [
                min(row["per_time_build_median_process_seconds"] for row in methods),
                max(row["per_time_build_median_process_seconds"] for row in methods),
            ],
            "per_packet_application_process_seconds": [
                min(row["per_packet_application_median_process_seconds"] for row in methods),
                max(row["per_packet_application_median_process_seconds"] for row in methods),
            ],
        },
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def verify(path: Path) -> None:
    """Verify the fixed timing protocol and all recorded medians."""
    record = json.loads(path.read_text())
    assert record["scientific_acceptance_unchanged"] is True
    assert record["environment"]["repetitions"] == REPETITIONS
    assert all(value == "1" for value in record["environment"]["thread_variables"].values())
    assert {row["method"] for row in record["methods"]} == set(base.METHODS)
    for row in record["methods"]:
        for field in (
            "factorization_process_seconds",
            "per_time_build_process_seconds",
            "per_packet_application_process_seconds",
        ):
            values = row[field]
            assert len(values) == REPETITIONS
            assert all(np.isfinite(value) and value > 0.0 for value in values)
        assert row["factorization_median_process_seconds"] == median(row["factorization_process_seconds"])
        assert row["per_time_build_median_process_seconds"] == median(row["per_time_build_process_seconds"])
        assert row["per_packet_application_median_process_seconds"] == median(row["per_packet_application_process_seconds"])
    print("wall_spectrum_full_propagator_cost verify: ALL PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path)
    parser.add_argument("--verify", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify is not None:
        verify(args.verify)
        return 0
    output = args.write or OUTPUT
    record = write_certificate(output)
    print(json.dumps(record["median_ranges"], indent=2))
    verify(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
