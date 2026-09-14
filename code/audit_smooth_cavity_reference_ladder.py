#!/usr/bin/env python3
"""Continue the smooth-cavity reference ladder and rescore frozen candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from audit_smooth_cavity_neutral_reference import (
    CANDIDATES,
    CONSENSUS_TOLERANCE,
    OUTPUT as LEVEL25_OUTPUT,
    _smooth_errors,
    _wall_disagreement,
)
from heldout_cavity_compatible_comparison import REYNOLDS_NUMBERS, _run_case
from project_paths import CODEDIR, RESULTSDIR, ensure_generated_dirs


SCHEMA = "smooth-cavity-reference-ladder-v1"
OUTPUT = RESULTSDIR / "smooth_cavity_reference_ladder.json"
CONFIGURATIONS = (
    {
        "name": "smooth_bmf_p8_n33_reference",
        "method": "bmf",
        "degree": 8,
        "elements": 4,
        "npts": 33,
        "dt": 0.01,
        "reference": True,
        "lid_profile": "smooth_quartic",
        "level25_name": "smooth_bmf_p8_n25_reference",
    },
    {
        "name": "smooth_sem_p8_e4_reference",
        "method": "compatible_sem",
        "degree": 8,
        "elements": 4,
        "npts": 33,
        "dt": 0.01,
        "reference": True,
        "lid_profile": "smooth_quartic",
        "level25_name": "smooth_sem_p8_e3_reference",
    },
)
SOURCE_FILES = (
    CODEDIR / "tensor_compatible_flow.py",
    CODEDIR / "tensor_compatible_cavity.py",
    CODEDIR / "compatible_sem_flow.py",
    CODEDIR / "heldout_cavity_compatible_comparison.py",
    CODEDIR / "audit_smooth_cavity_neutral_reference.py",
    Path(__file__).resolve(),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {path.name: _sha256(path) for path in SOURCE_FILES}


def _score(level25_rows: list[dict], rows: list[dict], reynolds: float) -> dict:
    references = [row for row in rows if row["reynolds"] == reynolds]
    bmf_reference = next(row for row in references if row["method"] == "bmf")
    sem_reference = next(
        row for row in references if row["method"] == "compatible_sem"
    )
    bmf_to_sem = _smooth_errors(bmf_reference, sem_reference)
    sem_to_bmf = _smooth_errors(sem_reference, bmf_reference)
    reference_disagreement = max(
        bmf_to_sem["joint_error"], sem_to_bmf["joint_error"]
    )
    wall = _wall_disagreement(bmf_reference, sem_reference)
    physical_wall_maximum = max(
        wall[key]
        for key in (
            "final_lid_reaction",
            "integrated_lid_reaction_work",
            "integrated_net_wall_work",
        )
    )
    candidate_scores = {}
    for candidate in CANDIDATES:
        row = next(
            value
            for value in level25_rows
            if value["reynolds"] == reynolds
            and value["name"] == candidate["name"]
        )
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
        orderings[name] = {
            "margins": margins,
            "minimum_margin": min(margins),
            "reference_disagreement": reference_disagreement,
            "eligible": bool(
                reference_disagreement < CONSENSUS_TOLERANCE
                and physical_wall_maximum < CONSENSUS_TOLERANCE
                and min(margins) > reference_disagreement
            ),
        }
    return {
        "reynolds": reynolds,
        "reference_disagreement": {
            "bmf_against_sem": bmf_to_sem,
            "sem_against_bmf": sem_to_bmf,
            "maximum_joint": reference_disagreement,
            **wall,
        },
        "consensus_eligible": bool(
            reference_disagreement < CONSENSUS_TOLERANCE
            and physical_wall_maximum < CONSENSUS_TOLERANCE
        ),
        "candidate_scores": candidate_scores,
        "orderings": orderings,
    }


def build_record() -> dict:
    level25 = json.loads(LEVEL25_OUTPUT.read_text())
    rows = []
    for reynolds in REYNOLDS_NUMBERS:
        for configuration in CONFIGURATIONS:
            run_configuration = {
                key: value
                for key, value in configuration.items()
                if key != "level25_name"
            }
            row = _run_case(reynolds, run_configuration)
            previous = next(
                value
                for value in level25["rows"]
                if value["reynolds"] == reynolds
                and value["name"] == configuration["level25_name"]
            )
            row["level25_name"] = configuration["level25_name"]
            row["family_refinement_errors"] = _smooth_errors(previous, row)
            row["family_wall_refinement"] = _wall_disagreement(previous, row)
            rows.append(row)
    return {
        "schema": SCHEMA,
        "source_sha256": _source_hashes(),
        "level25_artifact_sha256": _sha256(LEVEL25_OUTPUT),
        "configurations": list(CONFIGURATIONS),
        "rows": rows,
        "scores": [
            _score(level25["rows"], rows, reynolds)
            for reynolds in REYNOLDS_NUMBERS
        ],
    }


def verify_record(record: dict) -> None:
    if record.get("schema") != SCHEMA:
        raise ValueError("smooth reference ladder schema mismatch")
    if record.get("source_sha256") != _source_hashes():
        raise ValueError("smooth reference ladder source hash mismatch")
    if record.get("level25_artifact_sha256") != _sha256(LEVEL25_OUTPUT):
        raise ValueError("smooth reference ladder level-25 hash mismatch")
    expected = [
        (reynolds, configuration["name"])
        for reynolds in REYNOLDS_NUMBERS
        for configuration in CONFIGURATIONS
    ]
    rows = record.get("rows", [])
    if [(row["reynolds"], row["name"]) for row in rows] != expected:
        raise ValueError("smooth reference ladder matrix changed")
    for row in rows:
        if float(row["maximum_relative_residual"]) > 1.0e-11:
            raise AssertionError("smooth reference ladder residual failed")
        if float(row["maximum_divergence_relative_residual"]) > 2.0e-11:
            raise AssertionError("smooth reference ladder divergence failed")
        if float(row["maximum_pressure_gauge_relative_residual"]) > 2.0e-11:
            raise AssertionError("smooth reference ladder gauge failed")
        if float(row["maximum_energy_balance_residual"]) > 5.0e-9:
            raise AssertionError("smooth reference ladder energy failed")
        if float(row["maximum_normal_flux"]) > 2.0e-13:
            raise AssertionError("smooth reference ladder flux failed")
        for value in row["family_wall_refinement"].values():
            if not math.isfinite(float(value)):
                raise ValueError("smooth reference ladder wall metric is not finite")
    level25 = json.loads(LEVEL25_OUTPUT.read_text())
    expected_scores = [
        _score(level25["rows"], rows, reynolds)
        for reynolds in REYNOLDS_NUMBERS
    ]
    if record.get("scores") != expected_scores:
        raise ValueError("smooth reference ladder score mismatch")


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
