#!/usr/bin/env python3
"""Run the terminal smooth-cavity reference level at n=57."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from audit_smooth_cavity_neutral_reference import OUTPUT as LEVEL25_OUTPUT
from audit_smooth_cavity_neutral_reference import _smooth_errors, _wall_disagreement
from audit_smooth_cavity_reference_ladder import _score
from audit_smooth_cavity_reference_n41 import OUTPUT as LEVEL41_OUTPUT
from heldout_cavity_compatible_comparison import REYNOLDS_NUMBERS, _run_case
from project_paths import CODEDIR, RESULTSDIR, ensure_generated_dirs


SCHEMA = "smooth-cavity-reference-n57-v1"
OUTPUT = RESULTSDIR / "smooth_cavity_reference_n57.json"
CONFIGURATIONS = (
    {
        "name": "smooth_bmf_p8_n57_reference",
        "method": "bmf",
        "degree": 8,
        "elements": 7,
        "npts": 57,
        "dt": 0.01,
        "reference": True,
        "lid_profile": "smooth_quartic",
        "level41_name": "smooth_bmf_p8_n41_reference",
    },
    {
        "name": "smooth_sem_p8_e7_reference",
        "method": "compatible_sem",
        "degree": 8,
        "elements": 7,
        "npts": 57,
        "dt": 0.01,
        "reference": True,
        "lid_profile": "smooth_quartic",
        "level41_name": "smooth_sem_p8_e5_reference",
    },
)
SOURCE_FILES = (
    CODEDIR / "tensor_compatible_flow.py",
    CODEDIR / "tensor_compatible_cavity.py",
    CODEDIR / "compatible_sem_flow.py",
    CODEDIR / "heldout_cavity_compatible_comparison.py",
    CODEDIR / "audit_smooth_cavity_neutral_reference.py",
    CODEDIR / "audit_smooth_cavity_reference_ladder.py",
    CODEDIR / "audit_smooth_cavity_reference_n41.py",
    Path(__file__).resolve(),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {path.name: _sha256(path) for path in SOURCE_FILES}


def build_record() -> dict:
    level25 = json.loads(LEVEL25_OUTPUT.read_text())
    level41 = json.loads(LEVEL41_OUTPUT.read_text())
    rows = []
    for reynolds in REYNOLDS_NUMBERS:
        for configuration in CONFIGURATIONS:
            run_configuration = {
                key: value
                for key, value in configuration.items()
                if key != "level41_name"
            }
            row = _run_case(reynolds, run_configuration)
            previous = next(
                value
                for value in level41["rows"]
                if value["reynolds"] == reynolds
                and value["name"] == configuration["level41_name"]
            )
            row["level41_name"] = configuration["level41_name"]
            row["family_refinement_errors"] = _smooth_errors(previous, row)
            row["family_wall_refinement"] = _wall_disagreement(previous, row)
            rows.append(row)
    return {
        "schema": SCHEMA,
        "source_sha256": _source_hashes(),
        "level25_artifact_sha256": _sha256(LEVEL25_OUTPUT),
        "level41_artifact_sha256": _sha256(LEVEL41_OUTPUT),
        "configurations": list(CONFIGURATIONS),
        "rows": rows,
        "scores": [
            _score(level25["rows"], rows, reynolds)
            for reynolds in REYNOLDS_NUMBERS
        ],
    }


def verify_record(record: dict) -> None:
    if record.get("schema") != SCHEMA:
        raise ValueError("smooth n57 reference schema mismatch")
    if record.get("source_sha256") != _source_hashes():
        raise ValueError("smooth n57 source hash mismatch")
    if record.get("level25_artifact_sha256") != _sha256(LEVEL25_OUTPUT):
        raise ValueError("smooth n57 level-25 hash mismatch")
    if record.get("level41_artifact_sha256") != _sha256(LEVEL41_OUTPUT):
        raise ValueError("smooth n57 level-41 hash mismatch")
    expected = [
        (reynolds, configuration["name"])
        for reynolds in REYNOLDS_NUMBERS
        for configuration in CONFIGURATIONS
    ]
    rows = record.get("rows", [])
    if [(row["reynolds"], row["name"]) for row in rows] != expected:
        raise ValueError("smooth n57 run matrix changed")
    for row in rows:
        if float(row["maximum_relative_residual"]) > 1.0e-11:
            raise AssertionError("smooth n57 residual failed")
        if float(row["maximum_divergence_relative_residual"]) > 2.0e-11:
            raise AssertionError("smooth n57 divergence failed")
        if float(row["maximum_pressure_gauge_relative_residual"]) > 2.0e-11:
            raise AssertionError("smooth n57 gauge failed")
        if float(row["maximum_energy_balance_residual"]) > 5.0e-9:
            raise AssertionError("smooth n57 energy failed")
        if float(row["maximum_normal_flux"]) > 2.0e-13:
            raise AssertionError("smooth n57 flux failed")
        for value in row["family_wall_refinement"].values():
            if not math.isfinite(float(value)):
                raise ValueError("smooth n57 wall metric is not finite")
    level25 = json.loads(LEVEL25_OUTPUT.read_text())
    expected_scores = [
        _score(level25["rows"], rows, reynolds)
        for reynolds in REYNOLDS_NUMBERS
    ]
    if record.get("scores") != expected_scores:
        raise ValueError("smooth n57 score mismatch")


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
    print(json.dumps(record["scores"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
