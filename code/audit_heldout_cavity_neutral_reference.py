#!/usr/bin/env python3
"""Audit held-out cavity ordering against BMF and compatible SEM references."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from heldout_cavity_compatible_comparison import (
    REYNOLDS_NUMBERS,
    _errors,
    _run_case,
)
from project_paths import CODEDIR, RESULTSDIR, ensure_generated_dirs


SCHEMA = "heldout-cavity-neutral-reference-v1"
OUTPUT = RESULTSDIR / "heldout_cavity_neutral_reference_audit.json"
CANDIDATES = (
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
)
REFERENCES = (
    {
        "name": "bmf_p8_n25_reference",
        "method": "bmf",
        "degree": 8,
        "elements": 3,
        "npts": 25,
        "dt": 0.01,
        "reference": True,
    },
    {
        "name": "sem_p8_e3_reference",
        "method": "compatible_sem",
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
    CODEDIR / "heldout_cavity_compatible_comparison.py",
    Path(__file__).resolve(),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {path.name: _sha256(path) for path in SOURCE_FILES}


def _symmetric_scalar_error(left: float, right: float) -> float:
    return 2.0 * abs(left - right) / max(abs(left) + abs(right), 1.0e-300)


def _score(rows: list[dict], reynolds: float) -> dict:
    selected = [row for row in rows if row["reynolds"] == reynolds]
    bmf_reference = next(row for row in selected if row["name"] == REFERENCES[0]["name"])
    sem_reference = next(row for row in selected if row["name"] == REFERENCES[1]["name"])
    bmf_to_sem = _errors(bmf_reference, sem_reference)
    sem_to_bmf = _errors(sem_reference, bmf_reference)
    reference_disagreement = max(
        float(bmf_to_sem["joint_error"]), float(sem_to_bmf["joint_error"])
    )
    candidate_scores = {}
    for candidate in CANDIDATES:
        row = next(value for value in selected if value["name"] == candidate["name"])
        candidate_scores[candidate["name"]] = {
            "against_bmf_reference": _errors(row, bmf_reference),
            "against_sem_reference": _errors(row, sem_reference),
        }
    orderings = {}
    for name, left_name in (
        ("state_efficient_bmf_vs_sem", "bmf_p6_k2"),
        ("equal_state_bmf_vs_sem", "bmf_p8_k2"),
    ):
        margins = []
        for reference_name in ("against_bmf_reference", "against_sem_reference"):
            bmf_error = candidate_scores[left_name][reference_name]["joint_error"]
            sem_error = candidate_scores["sem_p8_e2"][reference_name]["joint_error"]
            margins.append(float(sem_error) - float(bmf_error))
        orderings[name] = {
            "margins": margins,
            "minimum_margin": min(margins),
            "reference_disagreement": reference_disagreement,
            "eligible": bool(
                min(margins) > 0.0 and min(margins) > reference_disagreement
            ),
        }
    return {
        "reynolds": reynolds,
        "reference_disagreement": {
            "bmf_against_sem": bmf_to_sem,
            "sem_against_bmf": sem_to_bmf,
            "maximum_joint": reference_disagreement,
            "integrated_nitsche_wall_work": _symmetric_scalar_error(
                bmf_reference["integrated_nitsche_wall_work"],
                sem_reference["integrated_nitsche_wall_work"],
            ),
            "integrated_lid_reaction_work": _symmetric_scalar_error(
                bmf_reference["integrated_lid_reaction_work"],
                sem_reference["integrated_lid_reaction_work"],
            ),
            "final_lid_reaction": _symmetric_scalar_error(
                bmf_reference["lid_reaction"], sem_reference["lid_reaction"]
            ),
        },
        "candidate_scores": candidate_scores,
        "orderings": orderings,
    }


def build_record() -> dict:
    configurations = CANDIDATES + REFERENCES
    rows = [
        _run_case(reynolds, configuration)
        for reynolds in REYNOLDS_NUMBERS
        for configuration in configurations
    ]
    return {
        "schema": SCHEMA,
        "source_sha256": _source_hashes(),
        "case": {
            "reynolds_numbers": list(REYNOLDS_NUMBERS),
            "candidates": list(CANDIDATES),
            "references": list(REFERENCES),
        },
        "rows": rows,
        "scores": [_score(rows, reynolds) for reynolds in REYNOLDS_NUMBERS],
    }


def verify_record(record: dict) -> None:
    if record.get("schema") != SCHEMA:
        raise ValueError("neutral reference schema mismatch")
    if record.get("source_sha256") != _source_hashes():
        raise ValueError("neutral reference source hash mismatch")
    configurations = CANDIDATES + REFERENCES
    expected = [
        (reynolds, configuration["name"])
        for reynolds in REYNOLDS_NUMBERS
        for configuration in configurations
    ]
    rows = record.get("rows", [])
    if [(row["reynolds"], row["name"]) for row in rows] != expected:
        raise ValueError("neutral reference run matrix changed")
    for row in rows:
        if float(row["maximum_relative_residual"]) > 1.0e-11:
            raise AssertionError("neutral reference residual failed")
        if float(row["maximum_divergence_relative_residual"]) > 2.0e-11:
            raise AssertionError("neutral reference divergence failed")
        if float(row["maximum_pressure_gauge_relative_residual"]) > 2.0e-11:
            raise AssertionError("neutral reference gauge failed")
        if float(row["maximum_energy_balance_residual"]) > 5.0e-9:
            raise AssertionError("neutral reference energy ledger failed")
        if float(row["maximum_normal_flux"]) > 2.0e-13:
            raise AssertionError("neutral reference no-penetration failed")
        for key in (
            "integrated_nitsche_wall_work",
            "integrated_lid_reaction_work",
            "lid_reaction",
        ):
            if not math.isfinite(float(row[key])):
                raise ValueError("neutral reference wall metric is not finite")
    expected_scores = [_score(rows, reynolds) for reynolds in REYNOLDS_NUMBERS]
    if record.get("scores") != expected_scores:
        raise ValueError("neutral reference score mismatch")


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
