#!/usr/bin/env python3
"""Audit smooth-lid cavity candidates against two compatible references."""
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


SCHEMA = "smooth-cavity-neutral-reference-v1"
OUTPUT = RESULTSDIR / "smooth_cavity_neutral_reference_audit.json"
CONSENSUS_TOLERANCE = 0.10
CANDIDATES = (
    {
        "name": "smooth_bmf_p6_k2",
        "method": "bmf",
        "degree": 6,
        "elements": 2,
        "npts": 13,
        "dt": 0.02,
        "reference": False,
        "lid_profile": "smooth_quartic",
    },
    {
        "name": "smooth_bmf_p8_k2",
        "method": "bmf",
        "degree": 8,
        "elements": 2,
        "npts": 17,
        "dt": 0.02,
        "reference": False,
        "lid_profile": "smooth_quartic",
    },
    {
        "name": "smooth_sem_p8_e2",
        "method": "compatible_sem",
        "degree": 8,
        "elements": 2,
        "npts": 17,
        "dt": 0.02,
        "reference": False,
        "lid_profile": "smooth_quartic",
    },
)
REFERENCES = (
    {
        "name": "smooth_bmf_p8_n25_reference",
        "method": "bmf",
        "degree": 8,
        "elements": 3,
        "npts": 25,
        "dt": 0.01,
        "reference": True,
        "lid_profile": "smooth_quartic",
    },
    {
        "name": "smooth_sem_p8_e3_reference",
        "method": "compatible_sem",
        "degree": 8,
        "elements": 3,
        "npts": 25,
        "dt": 0.01,
        "reference": True,
        "lid_profile": "smooth_quartic",
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


def _smooth_errors(row: dict, reference: dict) -> dict[str, float]:
    values = _errors(row, reference)
    values["joint_error"] = max(
        values["relative_velocity_error"],
        values["relative_pressure_error"],
        values["relative_full_wall_traction_diagnostic"],
        values["relative_lid_reaction_error"],
    )
    return values


def _wall_disagreement(left: dict, right: dict) -> dict[str, float]:
    return {
        "final_lid_reaction": _symmetric_scalar_error(
            left["lid_reaction"], right["lid_reaction"]
        ),
        "integrated_lid_reaction_work": _symmetric_scalar_error(
            left["integrated_lid_reaction_work"],
            right["integrated_lid_reaction_work"],
        ),
        "integrated_net_wall_work": _symmetric_scalar_error(
            left["integrated_net_wall_work"],
            right["integrated_net_wall_work"],
        ),
        "raw_nitsche_load_duality": _symmetric_scalar_error(
            left["integrated_nitsche_wall_work"],
            right["integrated_nitsche_wall_work"],
        ),
    }


def _score(rows: list[dict], reynolds: float) -> dict:
    selected = [row for row in rows if row["reynolds"] == reynolds]
    bmf_reference = next(
        row for row in selected if row["name"] == REFERENCES[0]["name"]
    )
    sem_reference = next(
        row for row in selected if row["name"] == REFERENCES[1]["name"]
    )
    bmf_to_sem = _smooth_errors(bmf_reference, sem_reference)
    sem_to_bmf = _smooth_errors(sem_reference, bmf_reference)
    reference_disagreement = max(
        bmf_to_sem["joint_error"], sem_to_bmf["joint_error"]
    )
    wall_disagreement = _wall_disagreement(bmf_reference, sem_reference)
    consensus_eligible = bool(
        reference_disagreement < CONSENSUS_TOLERANCE
        and max(
            wall_disagreement[key]
            for key in (
                "final_lid_reaction",
                "integrated_lid_reaction_work",
                "integrated_net_wall_work",
            )
        )
        < CONSENSUS_TOLERANCE
    )
    candidate_scores = {}
    for candidate in CANDIDATES:
        row = next(value for value in selected if value["name"] == candidate["name"])
        candidate_scores[candidate["name"]] = {
            "against_bmf_reference": _smooth_errors(row, bmf_reference),
            "against_sem_reference": _smooth_errors(row, sem_reference),
        }
    orderings = {}
    for name, bmf_name in (
        ("state_efficient_bmf_vs_sem", "smooth_bmf_p6_k2"),
        ("equal_state_bmf_vs_sem", "smooth_bmf_p8_k2"),
    ):
        margins = []
        for reference_name in ("against_bmf_reference", "against_sem_reference"):
            bmf_error = candidate_scores[bmf_name][reference_name]["joint_error"]
            sem_error = candidate_scores["smooth_sem_p8_e2"][reference_name][
                "joint_error"
            ]
            margins.append(float(sem_error) - float(bmf_error))
        bmf_row = next(row for row in selected if row["name"] == bmf_name)
        sem_row = next(
            row for row in selected if row["name"] == "smooth_sem_p8_e2"
        )
        orderings[name] = {
            "margins": margins,
            "minimum_margin": min(margins),
            "reference_disagreement": reference_disagreement,
            "accuracy_eligible": bool(
                consensus_eligible
                and min(margins) > reference_disagreement
            ),
            "bmf_state_bytes": bmf_row["state_storage_bytes"],
            "sem_state_bytes": sem_row["state_storage_bytes"],
            "bmf_total_seconds": bmf_row["total_seconds"],
            "sem_total_seconds": sem_row["total_seconds"],
            "time_and_storage_eligible": bool(
                bmf_row["state_storage_bytes"] <= sem_row["state_storage_bytes"]
                and bmf_row["total_seconds"] < sem_row["total_seconds"]
            ),
        }
    return {
        "reynolds": reynolds,
        "reference_disagreement": {
            "bmf_against_sem": bmf_to_sem,
            "sem_against_bmf": sem_to_bmf,
            "maximum_joint": reference_disagreement,
            **wall_disagreement,
        },
        "consensus_eligible": consensus_eligible,
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
            "lid_velocity": "16 x^2 (1-x)^2 on z=1; zero on other walls",
            "reynolds_numbers": list(REYNOLDS_NUMBERS),
            "consensus_tolerance": CONSENSUS_TOLERANCE,
            "candidates": list(CANDIDATES),
            "references": list(REFERENCES),
        },
        "rows": rows,
        "scores": [_score(rows, reynolds) for reynolds in REYNOLDS_NUMBERS],
    }


def verify_record(record: dict) -> None:
    if record.get("schema") != SCHEMA:
        raise ValueError("smooth cavity schema mismatch")
    if record.get("source_sha256") != _source_hashes():
        raise ValueError("smooth cavity source hash mismatch")
    expected_case = {
        "lid_velocity": "16 x^2 (1-x)^2 on z=1; zero on other walls",
        "reynolds_numbers": list(REYNOLDS_NUMBERS),
        "consensus_tolerance": CONSENSUS_TOLERANCE,
        "candidates": list(CANDIDATES),
        "references": list(REFERENCES),
    }
    if record.get("case") != expected_case:
        raise ValueError("smooth cavity case definition changed")
    configurations = CANDIDATES + REFERENCES
    expected = [
        (reynolds, configuration["name"])
        for reynolds in REYNOLDS_NUMBERS
        for configuration in configurations
    ]
    rows = record.get("rows", [])
    if [(row["reynolds"], row["name"]) for row in rows] != expected:
        raise ValueError("smooth cavity run matrix changed")
    for row in rows:
        if row.get("lid_profile") != "smooth_quartic":
            raise ValueError("smooth cavity lid profile changed")
        if float(row["maximum_relative_residual"]) > 1.0e-11:
            raise AssertionError("smooth cavity residual failed")
        if float(row["maximum_divergence_relative_residual"]) > 2.0e-11:
            raise AssertionError("smooth cavity divergence failed")
        if float(row["maximum_pressure_gauge_relative_residual"]) > 2.0e-11:
            raise AssertionError("smooth cavity gauge failed")
        if float(row["maximum_energy_balance_residual"]) > 5.0e-9:
            raise AssertionError("smooth cavity energy ledger failed")
        if float(row["maximum_normal_flux"]) > 2.0e-13:
            raise AssertionError("smooth cavity no-penetration failed")
        if int(row["operator_storage"]["multidimensional_dense_operator_entries"]) != 0:
            raise AssertionError("smooth cavity stored a dense operator")
        for key in (
            "integrated_nitsche_wall_work",
            "integrated_net_wall_work",
            "integrated_lid_reaction_work",
            "lid_reaction",
            "total_seconds",
        ):
            if not math.isfinite(float(row[key])):
                raise ValueError("smooth cavity metric is not finite")
    expected_scores = [_score(rows, reynolds) for reynolds in REYNOLDS_NUMBERS]
    if record.get("scores") != expected_scores:
        raise ValueError("smooth cavity score mismatch")


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
