#!/usr/bin/env python3
"""Measure robust tensor-compatible residual, storage, and wall scaling."""
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

from project_paths import RESULTSDIR, ensure_generated_dirs
from tensor_compatible_flow import TensorCompatibleFlow2D, TensorCompatibleFlow3D


SCHEMA = "tensor-compatible-scaling-v2"
OUTPUT = RESULTSDIR / "tensor_compatible_scaling.json"
DEGREE = 4
GRIDS_2D = (9, 17, 33, 65, 129)
GRIDS_3D_LINEAR = (9, 17, 33, 65)
GRIDS_3D_NONLINEAR = (9, 13, 17, 25)
BLOCKS = 5
RAW_SAMPLES = 5
MINIMUM_SAMPLE_SECONDS = 0.02
CUBIC_BAND = (2.75, 3.25)
MAXIMUM_BLOCK_INTERVAL = 0.50
THREAD_ENV_NAMES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        "thread_env": {name: os.environ.get(name, "unset") for name in THREAD_ENV_NAMES},
    }


def _timed_samples(function: Callable[[], object]) -> dict:
    function()
    samples = []
    for _ in range(RAW_SAMPLES):
        calls = 1
        attempts = []
        while True:
            start = time.process_time()
            for _call in range(calls):
                function()
            elapsed = time.process_time() - start
            attempts.append({"calls": calls, "elapsed_seconds": elapsed})
            if elapsed >= MINIMUM_SAMPLE_SECONDS:
                break
            scale = max(MINIMUM_SAMPLE_SECONDS / max(elapsed, 1.0e-12), 1.25)
            calls = max(calls + 1, int(math.ceil(1.05 * calls * scale)))
        samples.append(
            {
                "attempts": attempts,
                "calls": calls,
                "elapsed_seconds": elapsed,
                "seconds_per_call": elapsed / calls,
            }
        )
    return {
        "samples": samples,
        "median_seconds_per_call": float(
            np.median([sample["seconds_per_call"] for sample in samples])
        ),
    }


def _sparse_bytes(matrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _operator_bytes(operator) -> int:
    axis = operator.axis
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
        axis.greville0,
        axis.greville1,
        operator.tangential_factor,
    )
    trace_bytes = sum(
        value.nbytes
        for value in (axis.q, axis.w, axis.trace0, axis.trace1, axis.normal0, axis.normal1)
    )
    return int(sum(_sparse_bytes(matrix) for matrix in factors) + trace_bytes)


def _two_dimensional_row(npts: int, block_index: int) -> dict:
    setup_start = time.process_time()
    operator = TensorCompatibleFlow2D(npts, DEGREE)
    setup_time = time.process_time() - setup_start
    rng = np.random.default_rng(2000 + npts + 10000 * block_index)
    ux = rng.standard_normal(operator.ux_shape)
    uz = rng.standard_normal(operator.uz_shape)
    ux[[0, -1], :] = 0.0
    uz[:, [0, -1]] = 0.0
    dual_pressure = rng.standard_normal(operator.pressure_shape)
    pressure = rng.standard_normal(operator.pressure_shape)

    def linear() -> None:
        operator.mass_action(ux, uz)
        operator.viscous_action(ux, uz)
        operator.divergence(ux, uz)
        operator.pressure_action(dual_pressure)

    storage = operator.storage_record()
    state_bytes = int((storage["velocity_states"] + storage["pressure_states"]) * 8)
    saddle_size = storage["velocity_states"] + storage["pressure_states"] + 1
    return {
        "dimension": 2,
        "npts": npts,
        "setup_seconds": setup_time,
        "linear_residual": _timed_samples(linear),
        "nonlinear_residual": _timed_samples(lambda: operator.convection_dual(ux, uz)),
        "wall_traction": _timed_samples(
            lambda: operator.wall_traction(ux, uz, pressure, viscosity=0.01)
        ),
        "operator_storage_bytes": _operator_bytes(operator),
        "state_storage_bytes": state_bytes,
        "stored_factor_entries": storage["stored_factor_entries"],
        "velocity_states": storage["velocity_states"],
        "pressure_states": storage["pressure_states"],
        "dense_saddle_projected_bytes": int(8 * saddle_size**2),
    }


def _three_dimensional_row(
    npts: int,
    include_nonlinear: bool,
    block_index: int,
) -> dict:
    setup_start = time.process_time()
    operator = TensorCompatibleFlow3D(npts, DEGREE)
    setup_time = time.process_time() - setup_start
    rng = np.random.default_rng(3000 + npts + 10000 * block_index)
    ux = rng.standard_normal(operator.ux_shape)
    uy = rng.standard_normal(operator.uy_shape)
    uz = rng.standard_normal(operator.uz_shape)
    ux[[0, -1], :, :] = 0.0
    uy[:, [0, -1], :] = 0.0
    uz[:, :, [0, -1]] = 0.0
    dual_pressure = rng.standard_normal(operator.pressure_shape)

    def linear() -> None:
        operator.mass_action(ux, uy, uz)
        operator.viscous_action(ux, uy, uz)
        operator.divergence(ux, uy, uz)
        operator.pressure_action(dual_pressure)

    storage = operator.storage_record()
    state_bytes = int((storage["velocity_states"] + storage["pressure_states"]) * 8)
    saddle_size = storage["velocity_states"] + storage["pressure_states"] + 1
    row = {
        "dimension": 3,
        "npts": npts,
        "setup_seconds": setup_time,
        "linear_residual": _timed_samples(linear),
        "operator_storage_bytes": _operator_bytes(operator),
        "state_storage_bytes": state_bytes,
        "stored_factor_entries": storage["stored_factor_entries"],
        "velocity_states": storage["velocity_states"],
        "pressure_states": storage["pressure_states"],
        "dense_saddle_projected_bytes": int(8 * saddle_size**2),
    }
    if include_nonlinear:
        row["nonlinear_residual"] = _timed_samples(
            lambda: operator.convection_dual(ux, uy, uz)
        )
        row["nonlinear_workspace"] = operator.convection_workspace_record()
    return row


def _fit_exponent(rows: list[dict], key: str) -> float:
    selected = [row for row in rows if key in row][-3:]
    if len(selected) < 3:
        return math.nan
    x = np.log(np.asarray([row["npts"] for row in selected], dtype=float))
    if key == "stored_factor_entries":
        values = [row[key] for row in selected]
    else:
        values = [row[key]["median_seconds_per_call"] for row in selected]
    return float(np.polyfit(x, np.log(np.asarray(values, dtype=float)), 1)[0])


def _build_exponents(rows_2d: list[dict], rows_3d: list[dict]) -> dict:
    linear_rows_3d = [
        row for row in rows_3d if row["npts"] in GRIDS_3D_LINEAR
    ]
    nonlinear_rows_3d = [
        row for row in rows_3d if row["npts"] in GRIDS_3D_NONLINEAR
    ]
    return {
        "two_dimensional_factor_entries": _fit_exponent(rows_2d, "stored_factor_entries"),
        "two_dimensional_linear_residual": _fit_exponent(rows_2d, "linear_residual"),
        "two_dimensional_nonlinear_residual": _fit_exponent(rows_2d, "nonlinear_residual"),
        "two_dimensional_wall_traction": _fit_exponent(rows_2d, "wall_traction"),
        "three_dimensional_factor_entries": _fit_exponent(
            linear_rows_3d, "stored_factor_entries"
        ),
        "three_dimensional_linear_residual": _fit_exponent(
            linear_rows_3d, "linear_residual"
        ),
        "three_dimensional_nonlinear_residual": _fit_exponent(
            nonlinear_rows_3d, "nonlinear_residual"
        ),
    }


def _build_block(block_index: int) -> dict:
    ascending = block_index % 2 == 0
    order_2d = list(GRIDS_2D if ascending else tuple(reversed(GRIDS_2D)))
    nonlinear_set = set(GRIDS_3D_NONLINEAR)
    all_3d = sorted(set(GRIDS_3D_LINEAR) | nonlinear_set)
    order_3d = all_3d if ascending else list(reversed(all_3d))
    rows_2d_by_n = {
        npts: _two_dimensional_row(npts, block_index) for npts in order_2d
    }
    rows_3d_by_n = {
        npts: _three_dimensional_row(npts, npts in nonlinear_set, block_index)
        for npts in order_3d
    }
    rows_2d = [rows_2d_by_n[npts] for npts in GRIDS_2D]
    rows_3d = [rows_3d_by_n[npts] for npts in all_3d]
    return {
        "block_index": block_index,
        "execution_order_2d": order_2d,
        "execution_order_3d": order_3d,
        "two_dimensional": rows_2d,
        "three_dimensional": rows_3d,
        "fitted_exponents_largest_three": _build_exponents(rows_2d, rows_3d),
    }


def _exponent_summary(blocks: list[dict], key: str) -> dict:
    values = [float(block["fitted_exponents_largest_three"][key]) for block in blocks]
    lower, upper = CUBIC_BAND
    in_band = sum(lower <= value <= upper for value in values)
    interval = max(values) - min(values)
    median = float(np.median(values))
    return {
        "block_exponents": values,
        "median": median,
        "minimum": min(values),
        "maximum": max(values),
        "in_cubic_band_count": in_band,
        "interval_width": interval,
        "stable_cubic": bool(
            lower <= median <= upper
            and in_band >= BLOCKS - 1
            and interval <= MAXIMUM_BLOCK_INTERVAL
        ),
    }


def build_record() -> dict:
    blocks = [_build_block(block_index) for block_index in range(BLOCKS)]
    summaries = {
        key: _exponent_summary(blocks, key)
        for key in blocks[0]["fitted_exponents_largest_three"]
    }
    nonlinear_cubic = summaries["three_dimensional_nonlinear_residual"][
        "stable_cubic"
    ]
    linear_cubic = summaries["three_dimensional_linear_residual"]["stable_cubic"]
    source = Path(__file__).resolve()
    return {
        "schema": SCHEMA,
        "source_sha256": _sha256(source),
        "operator_source_sha256": _sha256(source.with_name("tensor_compatible_flow.py")),
        "config": {
            "degree": DEGREE,
            "two_dimensional_grids": list(GRIDS_2D),
            "three_dimensional_linear_grids": list(GRIDS_3D_LINEAR),
            "three_dimensional_nonlinear_grids": list(GRIDS_3D_NONLINEAR),
            "dense_oracle_memory_stop_bytes": 2 * 1024**3,
            "blocks": BLOCKS,
            "raw_samples_per_metric": RAW_SAMPLES,
            "minimum_sample_seconds": MINIMUM_SAMPLE_SECONDS,
            "cubic_band": list(CUBIC_BAND),
            "maximum_block_interval": MAXIMUM_BLOCK_INTERVAL,
        },
        "timing": _timing_stamp(),
        "blocks": blocks,
        "exponent_summaries": summaries,
        "classification": {
            "three_dimensional_nonlinear": (
                "stable_cubic" if nonlinear_cubic else "timing_unstable_or_noncubic"
            ),
            "three_dimensional_linear": (
                "stable_cubic" if linear_cubic else "delimited_not_cubic"
            ),
            "general_three_dimensional_operator_scaling": (
                "promoted" if nonlinear_cubic and linear_cubic else "not_promoted"
            ),
        },
        "provenance": [
            "tensor_compatible_scaling_pre_slab_summary.json",
            "tensor_compatible_scaling_post_slab_trial_1_failure.json",
            "tensor_compatible_scaling_post_slab_trial_2.json",
        ],
    }


def _verify_timing_record(record: dict) -> None:
    samples = record.get("samples", [])
    if len(samples) != RAW_SAMPLES:
        raise ValueError("raw timing sample count mismatch")
    values = []
    for sample in samples:
        attempts = sample.get("attempts", [])
        if not attempts:
            raise ValueError("timing calibration attempts are missing")
        terminal = attempts[-1]
        if int(sample["calls"]) != int(terminal["calls"]):
            raise ValueError("timing call count mismatch")
        if not np.isclose(
            float(sample["elapsed_seconds"]),
            float(terminal["elapsed_seconds"]),
            rtol=1.0e-14,
            atol=1.0e-14,
        ):
            raise ValueError("timing elapsed value mismatch")
        if float(sample["elapsed_seconds"]) < MINIMUM_SAMPLE_SECONDS:
            raise ValueError("timing sample missed the minimum duration")
        expected = float(sample["elapsed_seconds"]) / int(sample["calls"])
        if not np.isclose(float(sample["seconds_per_call"]), expected):
            raise ValueError("per-call timing mismatch")
        values.append(expected)
    if not np.isclose(float(record["median_seconds_per_call"]), np.median(values)):
        raise ValueError("timing median mismatch")


def verify_record(record: dict) -> None:
    source = Path(__file__).resolve()
    if record.get("schema") != SCHEMA:
        raise ValueError("tensor-compatible scaling schema mismatch")
    if record.get("source_sha256") != _sha256(source):
        raise ValueError("tensor-compatible scaling source hash mismatch")
    if record.get("operator_source_sha256") != _sha256(
        source.with_name("tensor_compatible_flow.py")
    ):
        raise ValueError("tensor-compatible operator hash mismatch")
    expected_config = {
        "degree": DEGREE,
        "two_dimensional_grids": list(GRIDS_2D),
        "three_dimensional_linear_grids": list(GRIDS_3D_LINEAR),
        "three_dimensional_nonlinear_grids": list(GRIDS_3D_NONLINEAR),
        "dense_oracle_memory_stop_bytes": 2 * 1024**3,
        "blocks": BLOCKS,
        "raw_samples_per_metric": RAW_SAMPLES,
        "minimum_sample_seconds": MINIMUM_SAMPLE_SECONDS,
        "cubic_band": list(CUBIC_BAND),
        "maximum_block_interval": MAXIMUM_BLOCK_INTERVAL,
    }
    if record.get("config") != expected_config:
        raise ValueError("tensor-compatible scaling configuration changed")
    timing = record.get("timing", {})
    if timing.get("metric") != "process_time_seconds":
        raise ValueError("tensor-compatible timing metric mismatch")
    if any(timing.get("thread_env", {}).get(name) != "1" for name in THREAD_ENV_NAMES):
        raise ValueError("tensor-compatible scaling was not single-thread pinned")
    blocks = record.get("blocks", [])
    if len(blocks) != BLOCKS:
        raise ValueError("timing block count mismatch")
    all_3d = sorted(set(GRIDS_3D_LINEAR) | set(GRIDS_3D_NONLINEAR))
    for block_index, block in enumerate(blocks):
        if int(block.get("block_index", -1)) != block_index:
            raise ValueError("timing block index mismatch")
        expected_order_2d = list(
            GRIDS_2D if block_index % 2 == 0 else tuple(reversed(GRIDS_2D))
        )
        expected_order_3d = all_3d if block_index % 2 == 0 else list(reversed(all_3d))
        if block.get("execution_order_2d") != expected_order_2d:
            raise ValueError("two-dimensional execution order mismatch")
        if block.get("execution_order_3d") != expected_order_3d:
            raise ValueError("three-dimensional execution order mismatch")
        rows_2d = block.get("two_dimensional", [])
        rows_3d = block.get("three_dimensional", [])
        if [row.get("npts") for row in rows_2d] != list(GRIDS_2D):
            raise ValueError("two-dimensional rows are incomplete")
        if [row.get("npts") for row in rows_3d] != all_3d:
            raise ValueError("three-dimensional rows are incomplete")
        for row in rows_2d:
            for key in ("linear_residual", "nonlinear_residual", "wall_traction"):
                _verify_timing_record(row[key])
        for row in rows_3d:
            _verify_timing_record(row["linear_residual"])
            if row["npts"] in GRIDS_3D_NONLINEAR:
                _verify_timing_record(row["nonlinear_residual"])
                workspace = row.get("nonlinear_workspace", {})
                if int(workspace.get("quadrature_slabs", 0)) <= 1:
                    raise ValueError("three-dimensional convection is not slab batched")
                if int(workspace["maximum_scalar_slab_entries"]) >= int(
                    workspace["global_scalar_volume_entries"]
                ):
                    raise ValueError("slab workspace did not reduce the global volume")
        expected_exponents = _build_exponents(rows_2d, rows_3d)
        for key, expected in expected_exponents.items():
            if not np.isclose(
                float(block["fitted_exponents_largest_three"][key]),
                expected,
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                raise ValueError(f"block exponent mismatch for {key}")
        for row in rows_2d + rows_3d:
            if int(row["operator_storage_bytes"]) >= int(row["dense_saddle_projected_bytes"]):
                raise AssertionError("matrix-free storage did not beat dense storage")
    summaries = record.get("exponent_summaries", {})
    for key in blocks[0]["fitted_exponents_largest_three"]:
        expected = _exponent_summary(blocks, key)
        actual = summaries.get(key, {})
        for field, value in expected.items():
            if isinstance(value, list):
                if not np.allclose(actual.get(field, []), value):
                    raise ValueError(f"exponent summary mismatch for {key}")
            elif isinstance(value, bool):
                if actual.get(field) is not value:
                    raise ValueError(f"exponent stability mismatch for {key}")
            elif not np.isclose(float(actual.get(field, math.nan)), float(value)):
                raise ValueError(f"exponent summary mismatch for {key}")
    nonlinear = summaries["three_dimensional_nonlinear_residual"]["stable_cubic"]
    linear = summaries["three_dimensional_linear_residual"]["stable_cubic"]
    expected_general = "promoted" if nonlinear and linear else "not_promoted"
    if record.get("classification", {}).get(
        "general_three_dimensional_operator_scaling"
    ) != expected_general:
        raise ValueError("three-dimensional classification mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    ensure_generated_dirs()
    if args.run:
        if OUTPUT.exists() and not args.replace:
            raise FileExistsError(f"refusing to overwrite {OUTPUT}")
        record = build_record()
        OUTPUT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        verify_record(record)
        print(json.dumps(record["classification"], indent=2, sort_keys=True))
        return 0
    if args.verify:
        record = json.loads(OUTPUT.read_text())
        verify_record(record)
        print(json.dumps(record["classification"], indent=2, sort_keys=True))
        return 0
    parser.error("select --run or --verify")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
