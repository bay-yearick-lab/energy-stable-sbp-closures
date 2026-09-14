#!/usr/bin/env python3
"""Audit temporal error in the held-out compatible cavity comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from heldout_cavity_compatible_comparison import (
    OUTPUT as COARSE_OUTPUT,
    REYNOLDS_NUMBERS,
    _errors,
    _run_case,
)
from project_paths import CODEDIR, RESULTSDIR, ensure_generated_dirs


SCHEMA = "heldout-cavity-temporal-audit-v1"
OUTPUT = RESULTSDIR / "heldout_cavity_temporal_audit.json"
CONFIGURATIONS = (
    {
        "name": "bmf_p8_k2_dt01",
        "method": "bmf",
        "degree": 8,
        "elements": 2,
        "npts": 17,
        "dt": 0.01,
        "reference": False,
        "coarse_name": "bmf_p8_k2",
    },
    {
        "name": "sem_p8_e2_dt01",
        "method": "compatible_sem",
        "degree": 8,
        "elements": 2,
        "npts": 17,
        "dt": 0.01,
        "reference": False,
        "coarse_name": "sem_p8_e2",
    },
)
SOURCE_FILES = (
    CODEDIR / "tensor_compatible_flow.py",
    CODEDIR / "tensor_compatible_cavity.py",
    CODEDIR / "compatible_sem_flow.py",
    CODEDIR / "heldout_cavity_compatible_comparison.py",
    Path(__file__).resolve(),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {path.name: _sha256(path) for path in SOURCE_FILES}


def build_record() -> dict:
    coarse_record = json.loads(COARSE_OUTPUT.read_text())
    rows = []
    for reynolds in REYNOLDS_NUMBERS:
        global_reference = next(
            row
            for row in coarse_record["rows"]
            if row["reynolds"] == reynolds and row["reference"]
        )
        for configuration in CONFIGURATIONS:
            coarse = next(
                row
                for row in coarse_record["rows"]
                if row["reynolds"] == reynolds
                and row["name"] == configuration["coarse_name"]
            )
            run_configuration = {
                key: value
                for key, value in configuration.items()
                if key != "coarse_name"
            }
            row = _run_case(reynolds, run_configuration)
            temporal = _errors(coarse, row)
            refined = _errors(row, global_reference)
            coarse_joint = float(coarse["errors"]["joint_error"])
            row["coarse_name"] = configuration["coarse_name"]
            row["temporal_errors"] = temporal
            row["refined_reference_errors"] = refined
            row["temporal_fraction_of_coarse_reference_error"] = (
                float(temporal["joint_error"]) / coarse_joint
            )
            rows.append(row)
    return {
        "schema": SCHEMA,
        "source_sha256": _source_hashes(),
        "coarse_artifact_sha256": _sha256(COARSE_OUTPUT),
        "configurations": list(CONFIGURATIONS),
        "rows": rows,
    }


def verify_record(record: dict) -> None:
    if record.get("schema") != SCHEMA:
        raise ValueError("held-out temporal schema mismatch")
    if record.get("source_sha256") != _source_hashes():
        raise ValueError("held-out temporal source hash mismatch")
    if record.get("coarse_artifact_sha256") != _sha256(COARSE_OUTPUT):
        raise ValueError("held-out temporal coarse artifact hash mismatch")
    expected = [
        (reynolds, configuration["name"])
        for reynolds in REYNOLDS_NUMBERS
        for configuration in CONFIGURATIONS
    ]
    rows = record.get("rows", [])
    if [(row["reynolds"], row["name"]) for row in rows] != expected:
        raise ValueError("held-out temporal run matrix changed")
    for row in rows:
        if float(row["maximum_relative_residual"]) > 1.0e-11:
            raise AssertionError("held-out temporal residual failed")
        if float(row["maximum_divergence_relative_residual"]) > 2.0e-11:
            raise AssertionError("held-out temporal divergence failed")
        if float(row["maximum_pressure_gauge_relative_residual"]) > 2.0e-11:
            raise AssertionError("held-out temporal gauge failed")
        if float(row["maximum_energy_balance_residual"]) > 5.0e-9:
            raise AssertionError("held-out temporal energy ledger failed")
        if float(row["maximum_normal_flux"]) > 2.0e-13:
            raise AssertionError("held-out temporal no-penetration failed")


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
    print(
        json.dumps(
            [
                {
                    "reynolds": row["reynolds"],
                    "name": row["name"],
                    "temporal_fraction": row[
                        "temporal_fraction_of_coarse_reference_error"
                    ],
                    "refined_joint_error": row["refined_reference_errors"][
                        "joint_error"
                    ],
                }
                for row in record["rows"]
            ],
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
