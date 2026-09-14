#!/usr/bin/env python3
"""Continue the equal-state cavity reference ladder for both method families."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from audit_heldout_cavity_neutral_reference import (
    OUTPUT as LEVEL25_OUTPUT,
    _symmetric_scalar_error,
)
from heldout_cavity_compatible_comparison import REYNOLDS_NUMBERS, _errors, _run_case
from project_paths import CODEDIR, RESULTSDIR, ensure_generated_dirs


SCHEMA = "heldout-cavity-reference-ladder-v1"
OUTPUT = RESULTSDIR / "heldout_cavity_reference_ladder.json"
CONFIGURATIONS = (
    {
        "name": "bmf_p8_n33_reference",
        "method": "bmf",
        "degree": 8,
        "elements": 4,
        "npts": 33,
        "dt": 0.01,
        "reference": True,
        "level25_name": "bmf_p8_n25_reference",
    },
    {
        "name": "sem_p8_e4_reference",
        "method": "compatible_sem",
        "degree": 8,
        "elements": 4,
        "npts": 33,
        "dt": 0.01,
        "reference": True,
        "level25_name": "sem_p8_e3_reference",
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


def _wall_disagreement(left: dict, right: dict) -> dict[str, float]:
    return {
        "final_lid_reaction": _symmetric_scalar_error(
            left["lid_reaction"], right["lid_reaction"]
        ),
        "integrated_lid_reaction_work": _symmetric_scalar_error(
            left["integrated_lid_reaction_work"],
            right["integrated_lid_reaction_work"],
        ),
        "integrated_nitsche_wall_work": _symmetric_scalar_error(
            left["integrated_nitsche_wall_work"],
            right["integrated_nitsche_wall_work"],
        ),
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
            row["family_refinement_errors"] = _errors(previous, row)
            row["family_wall_refinement"] = _wall_disagreement(previous, row)
            rows.append(row)
    comparisons = []
    for reynolds in REYNOLDS_NUMBERS:
        selected = [row for row in rows if row["reynolds"] == reynolds]
        bmf = next(row for row in selected if row["method"] == "bmf")
        sem = next(row for row in selected if row["method"] == "compatible_sem")
        bmf_to_sem = _errors(bmf, sem)
        sem_to_bmf = _errors(sem, bmf)
        wall = _wall_disagreement(bmf, sem)
        maximum_joint = max(
            bmf_to_sem["joint_error"], sem_to_bmf["joint_error"]
        )
        comparisons.append(
            {
                "reynolds": reynolds,
                "bmf_against_sem": bmf_to_sem,
                "sem_against_bmf": sem_to_bmf,
                "wall_disagreement": wall,
                "maximum_joint": maximum_joint,
                "consensus_eligible": bool(
                    maximum_joint < 0.10
                    and wall["final_lid_reaction"] < 0.10
                    and wall["integrated_lid_reaction_work"] < 0.10
                    and wall["integrated_nitsche_wall_work"] < 0.10
                ),
            }
        )
    return {
        "schema": SCHEMA,
        "source_sha256": _source_hashes(),
        "level25_artifact_sha256": _sha256(LEVEL25_OUTPUT),
        "configurations": list(CONFIGURATIONS),
        "rows": rows,
        "comparisons": comparisons,
    }


def verify_record(record: dict) -> None:
    if record.get("schema") != SCHEMA:
        raise ValueError("reference ladder schema mismatch")
    if record.get("source_sha256") != _source_hashes():
        raise ValueError("reference ladder source hash mismatch")
    if record.get("level25_artifact_sha256") != _sha256(LEVEL25_OUTPUT):
        raise ValueError("reference ladder level-25 hash mismatch")
    expected = [
        (reynolds, configuration["name"])
        for reynolds in REYNOLDS_NUMBERS
        for configuration in CONFIGURATIONS
    ]
    rows = record.get("rows", [])
    if [(row["reynolds"], row["name"]) for row in rows] != expected:
        raise ValueError("reference ladder matrix changed")
    for row in rows:
        if float(row["maximum_relative_residual"]) > 1.0e-11:
            raise AssertionError("reference ladder residual failed")
        if float(row["maximum_divergence_relative_residual"]) > 2.0e-11:
            raise AssertionError("reference ladder divergence failed")
        if float(row["maximum_pressure_gauge_relative_residual"]) > 2.0e-11:
            raise AssertionError("reference ladder gauge failed")
        if float(row["maximum_energy_balance_residual"]) > 5.0e-9:
            raise AssertionError("reference ladder energy failed")
        if float(row["maximum_normal_flux"]) > 2.0e-13:
            raise AssertionError("reference ladder flux failed")
        for value in row["family_wall_refinement"].values():
            if not math.isfinite(float(value)):
                raise ValueError("reference ladder wall value is not finite")


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
    print(json.dumps(record["comparisons"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
