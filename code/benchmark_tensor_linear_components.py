#!/usr/bin/env python3
"""Profile the matrix-free three-dimensional linear residual components."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
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
from tensor_compatible_flow import TensorCompatibleFlow3D


SCHEMA = "tensor-compatible-linear-components-v1"
OUTPUT = RESULTSDIR / "tensor_compatible_linear_components.json"
DEGREE = 4
GRIDS = (9, 17, 33, 65)
BLOCKS = 3
RAW_SAMPLES = 5
MINIMUM_SAMPLE_SECONDS = 0.02
SOURCE_FILES = (CODEDIR / "tensor_compatible_flow.py", Path(__file__).resolve())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {path.name: _sha256(path) for path in SOURCE_FILES}


def _timed(function: Callable[[], object]) -> dict:
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


def _row(npts: int, block_index: int) -> dict:
    flow = TensorCompatibleFlow3D(npts, DEGREE)
    rng = np.random.default_rng(7000 + npts + 10000 * block_index)
    ux = rng.standard_normal(flow.ux_shape)
    uy = rng.standard_normal(flow.uy_shape)
    uz = rng.standard_normal(flow.uz_shape)
    ux[[0, -1], :, :] = 0.0
    uy[:, [0, -1], :] = 0.0
    uz[:, :, [0, -1]] = 0.0
    pressure = rng.standard_normal(flow.pressure_shape)

    def aggregate() -> None:
        flow.mass_action(ux, uy, uz)
        flow.viscous_action(ux, uy, uz)
        flow.divergence(ux, uy, uz)
        flow.pressure_action(pressure)

    records = {
        "mass": _timed(lambda: flow.mass_action(ux, uy, uz)),
        "viscous": _timed(lambda: flow.viscous_action(ux, uy, uz)),
        "divergence": _timed(lambda: flow.divergence(ux, uy, uz)),
        "pressure": _timed(lambda: flow.pressure_action(pressure)),
        "aggregate": _timed(aggregate),
    }
    component_sum = sum(
        records[component]["median_seconds_per_call"]
        for component in ("mass", "viscous", "divergence", "pressure")
    )
    return {
        "npts": npts,
        "components": records,
        "component_sum_seconds": component_sum,
        "viscous_fraction": records["viscous"]["median_seconds_per_call"]
        / component_sum,
        "aggregate_to_component_sum_ratio": records["aggregate"][
            "median_seconds_per_call"
        ]
        / component_sum,
    }


def _fit(rows: list[dict], component: str) -> float:
    selected = rows[-3:]
    x = np.log(np.asarray([row["npts"] for row in selected], dtype=float))
    if component == "component_sum":
        values = [row["component_sum_seconds"] for row in selected]
    else:
        values = [
            row["components"][component]["median_seconds_per_call"]
            for row in selected
        ]
    return float(np.polyfit(x, np.log(np.asarray(values)), 1)[0])


def build_record() -> dict:
    blocks = []
    for block_index in range(BLOCKS):
        order = GRIDS if block_index % 2 == 0 else tuple(reversed(GRIDS))
        rows_by_grid = {npts: _row(npts, block_index) for npts in order}
        rows = [rows_by_grid[npts] for npts in GRIDS]
        blocks.append(
            {
                "block_index": block_index,
                "execution_order": list(order),
                "rows": rows,
                "exponents": {
                    component: _fit(rows, component)
                    for component in (
                        "mass",
                        "viscous",
                        "divergence",
                        "pressure",
                        "aggregate",
                        "component_sum",
                    )
                },
            }
        )
    return {
        "schema": SCHEMA,
        "case": {
            "degree": DEGREE,
            "grids": list(GRIDS),
            "blocks": BLOCKS,
            "raw_samples": RAW_SAMPLES,
            "minimum_sample_seconds": MINIMUM_SAMPLE_SECONDS,
            "timing_metric": "process_time_seconds",
        },
        "source_sha256": _source_hashes(),
        "blocks": blocks,
        "summary": {
            "median_exponents": {
                component: float(
                    np.median([block["exponents"][component] for block in blocks])
                )
                for component in (
                    "mass",
                    "viscous",
                    "divergence",
                    "pressure",
                    "aggregate",
                    "component_sum",
                )
            },
            "largest_viscous_fraction": float(
                np.median([block["rows"][-1]["viscous_fraction"] for block in blocks])
            ),
            "largest_aggregate_to_component_sum_ratio": float(
                np.median(
                    [
                        block["rows"][-1]["aggregate_to_component_sum_ratio"]
                        for block in blocks
                    ]
                )
            ),
        },
    }


def verify_record(record: dict) -> None:
    if record.get("schema") != SCHEMA:
        raise ValueError("linear component schema mismatch")
    if record.get("source_sha256") != _source_hashes():
        raise ValueError("linear component source hash mismatch")
    if record.get("case") != {
        "degree": DEGREE,
        "grids": list(GRIDS),
        "blocks": BLOCKS,
        "raw_samples": RAW_SAMPLES,
        "minimum_sample_seconds": MINIMUM_SAMPLE_SECONDS,
        "timing_metric": "process_time_seconds",
    }:
        raise ValueError("linear component case changed")
    blocks = record.get("blocks", [])
    if len(blocks) != BLOCKS:
        raise ValueError("linear component blocks are incomplete")
    for block_index, block in enumerate(blocks):
        expected_order = list(
            GRIDS if block_index % 2 == 0 else tuple(reversed(GRIDS))
        )
        if block.get("execution_order") != expected_order:
            raise ValueError("linear component execution order changed")
        rows = block.get("rows", [])
        if [row.get("npts") for row in rows] != list(GRIDS):
            raise ValueError("linear component grids are incomplete")
        for row in rows:
            for component in (
                "mass",
                "viscous",
                "divergence",
                "pressure",
                "aggregate",
            ):
                samples = row["components"][component]["samples"]
                if len(samples) != RAW_SAMPLES:
                    raise ValueError("linear component raw sample count changed")
                for sample in samples:
                    if float(sample["elapsed_seconds"]) < MINIMUM_SAMPLE_SECONDS:
                        raise ValueError("linear component sample is too short")
        for component, exponent in block["exponents"].items():
            if not np.isclose(float(exponent), _fit(rows, component)):
                raise ValueError("linear component exponent mismatch")


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
    else:
        record = json.loads(OUTPUT.read_text())
    if args.run or args.verify:
        verify_record(record)
    print(json.dumps(record["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
