#!/usr/bin/env python3
"""Compare strong and residual-consistent traction on the smooth cavity ladder."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import splu

from compatible_sem_flow import CompatibleSEMFlow2D
from heldout_cavity_compatible_comparison import REYNOLDS_NUMBERS, _errors, _run_case
from project_paths import CODEDIR, RESULTSDIR, ensure_generated_dirs
from tensor_compatible_flow import TensorCompatibleFlow2D


SCHEMA = "smooth-cavity-nitsche-traction-v1"
OUTPUT = RESULTSDIR / "smooth_cavity_nitsche_traction.json"
FULL_VECTOR_TRIAL = (
    RESULTSDIR / "smooth_cavity_nitsche_traction_trial_1_full_vector.json"
)
CONSENSUS_TOLERANCE = 0.025
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
REFERENCE_LEVELS = (25, 33, 41, 57)
REFERENCES = tuple(
    configuration
    for level in REFERENCE_LEVELS
    for configuration in (
        {
            "name": f"smooth_bmf_p8_n{level}_reference",
            "method": "bmf",
            "degree": 8,
            "elements": (level - 1) // 8,
            "npts": level,
            "dt": 0.01,
            "reference": True,
            "lid_profile": "smooth_quartic",
            "reference_level": level,
        },
        {
            "name": f"smooth_sem_p8_e{(level - 1) // 8}_reference",
            "method": "compatible_sem",
            "degree": 8,
            "elements": (level - 1) // 8,
            "npts": level,
            "dt": 0.01,
            "reference": True,
            "lid_profile": "smooth_quartic",
            "reference_level": level,
        },
    )
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


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right)) / max(
        float(np.linalg.norm(right)), 1.0e-300
    )


def _traction_vector(row: dict, key: str, component: str) -> np.ndarray:
    values = []
    traction = row["samples"][key]
    if component == "full":
        fields = (
            ("x0", "traction_x"),
            ("x0", "traction_z"),
            ("x1", "traction_x"),
            ("x1", "traction_z"),
            ("z0", "traction_x"),
            ("z0", "traction_z"),
            ("z1", "traction_x"),
            ("z1", "traction_z"),
        )
    elif component == "tangential":
        fields = (
            ("x0", "traction_z"),
            ("x1", "traction_z"),
            ("z0", "traction_x"),
            ("z1", "traction_x"),
        )
    elif component == "normal":
        fields = (
            ("x0", "traction_x"),
            ("x1", "traction_x"),
            ("z0", "traction_z"),
            ("z1", "traction_z"),
        )
    else:
        raise ValueError(f"unknown traction component {component}")
    for wall, field in fields:
        values.extend(traction[wall][field])
    return np.asarray(values)


def _estimator_errors(row: dict, reference: dict, estimator: str) -> dict[str, float]:
    base = _errors(row, reference)
    if estimator == "strong":
        traction_key = "traction"
        reaction_key = "lid_reaction"
    elif estimator == "nitsche_consistent":
        traction_key = "nitsche_consistent_traction"
        reaction_key = "numerical_lid_reaction"
    else:
        raise ValueError(f"unknown traction estimator {estimator}")
    traction_error = _relative(
        _traction_vector(row, traction_key, "full"),
        _traction_vector(reference, traction_key, "full"),
    )
    tangential_error = _relative(
        _traction_vector(row, traction_key, "tangential"),
        _traction_vector(reference, traction_key, "tangential"),
    )
    normal_error = _relative(
        _traction_vector(row, traction_key, "normal"),
        _traction_vector(reference, traction_key, "normal"),
    )
    reaction_error = abs(
        float(row[reaction_key]) - float(reference[reaction_key])
    ) / max(abs(float(reference[reaction_key])), 1.0e-300)
    return {
        "relative_velocity_error": base["relative_velocity_error"],
        "relative_pressure_error": base["relative_pressure_error"],
        "relative_wall_traction_error": traction_error,
        "relative_tangential_traction_error": tangential_error,
        "relative_normal_traction_error": normal_error,
        "relative_lid_reaction_error": reaction_error,
        "joint_error": max(
            base["relative_velocity_error"],
            base["relative_pressure_error"],
            traction_error,
            reaction_error,
        ),
        "wall_closure_joint_error": max(
            base["relative_velocity_error"],
            base["relative_pressure_error"],
            tangential_error,
            reaction_error,
        ),
    }


def _symmetric_scalar_error(left: float, right: float) -> float:
    return 2.0 * abs(left - right) / max(abs(left) + abs(right), 1.0e-300)


def _poiseuille_check() -> list[dict]:
    records = []
    viscosity = 0.1
    for method in ("bmf", "compatible_sem"):
        flow = (
            TensorCompatibleFlow2D(3, 2)
            if method == "bmf"
            else CompatibleSEMFlow2D(2, 1)
        )
        axis = flow.axis
        forcing = np.asarray(axis.b1.T @ axis.w).ravel()
        coefficients = splu(
            (viscosity * flow.tangential_factor).tocsc()
        ).solve(forcing)
        strong = viscosity * np.asarray(axis.normal1 @ coefficients).ravel()
        mismatch = np.asarray(axis.trace1 @ coefficients).ravel()
        numerical = strong - viscosity * flow.penalty * mismatch
        records.append(
            {
                "method": method,
                "strong_traction": strong.tolist(),
                "nitsche_consistent_traction": numerical.tolist(),
                "maximum_force_balance_error": float(
                    np.max(np.abs(numerical + 0.5))
                ),
            }
        )
    return records


def _reference_scores(rows: list[dict], reynolds: float) -> list[dict]:
    scores = []
    for level in REFERENCE_LEVELS:
        selected = [
            row
            for row in rows
            if row["reynolds"] == reynolds
            and row.get("reference_level") == level
        ]
        bmf = next(row for row in selected if row["method"] == "bmf")
        sem = next(row for row in selected if row["method"] == "compatible_sem")
        estimator_scores = {}
        for estimator in ("strong", "nitsche_consistent"):
            bmf_to_sem = _estimator_errors(bmf, sem, estimator)
            sem_to_bmf = _estimator_errors(sem, bmf, estimator)
            estimator_scores[estimator] = {
                "bmf_against_sem": bmf_to_sem,
                "sem_against_bmf": sem_to_bmf,
                "maximum_joint": max(
                    bmf_to_sem["joint_error"], sem_to_bmf["joint_error"]
                ),
                "maximum_wall_closure_joint": max(
                    bmf_to_sem["wall_closure_joint_error"],
                    sem_to_bmf["wall_closure_joint_error"],
                ),
            }
        estimator_scores["nitsche_consistent"][
            "integrated_traction_work_disagreement"
        ] = _symmetric_scalar_error(
            bmf["integrated_numerical_traction_work"],
            sem["integrated_numerical_traction_work"],
        )
        scores.append(
            {
                "reynolds": reynolds,
                "reference_level": level,
                "estimators": estimator_scores,
            }
        )
    return scores


def _candidate_score(rows: list[dict], reynolds: float) -> dict:
    selected = [row for row in rows if row["reynolds"] == reynolds]
    references = [
        row for row in selected if row.get("reference_level") == REFERENCE_LEVELS[-1]
    ]
    bmf_reference = next(row for row in references if row["method"] == "bmf")
    sem_reference = next(
        row for row in references if row["method"] == "compatible_sem"
    )
    reference_errors = (
        _estimator_errors(bmf_reference, sem_reference, "nitsche_consistent"),
        _estimator_errors(sem_reference, bmf_reference, "nitsche_consistent"),
    )
    full_reference_disagreement = max(
        value["joint_error"] for value in reference_errors
    )
    reference_disagreement = max(
        value["wall_closure_joint_error"] for value in reference_errors
    )
    candidate_scores = {}
    for candidate in CANDIDATES:
        row = next(value for value in selected if value["name"] == candidate["name"])
        candidate_scores[candidate["name"]] = {
            "against_bmf_reference": _estimator_errors(
                row, bmf_reference, "nitsche_consistent"
            ),
            "against_sem_reference": _estimator_errors(
                row, sem_reference, "nitsche_consistent"
            ),
        }
    orderings = {}
    full_vector_orderings = {}
    for name, bmf_name in (
        ("state_efficient_bmf_vs_sem", "smooth_bmf_p6_k2"),
        ("equal_state_bmf_vs_sem", "smooth_bmf_p8_k2"),
    ):
        margins = []
        full_margins = []
        for reference_name in ("against_bmf_reference", "against_sem_reference"):
            margins.append(
                candidate_scores["smooth_sem_p8_e2"][reference_name][
                    "wall_closure_joint_error"
                ]
                - candidate_scores[bmf_name][reference_name][
                    "wall_closure_joint_error"
                ]
            )
            full_margins.append(
                candidate_scores["smooth_sem_p8_e2"][reference_name]["joint_error"]
                - candidate_scores[bmf_name][reference_name]["joint_error"]
            )
        orderings[name] = {
            "margins": margins,
            "minimum_margin": min(margins),
            "reference_disagreement": reference_disagreement,
            "eligible": bool(
                reference_disagreement < CONSENSUS_TOLERANCE
                and min(margins) > reference_disagreement
            ),
        }
        full_vector_orderings[name] = {
            "margins": full_margins,
            "minimum_margin": min(full_margins),
            "reference_disagreement": full_reference_disagreement,
            "eligible": bool(
                full_reference_disagreement < CONSENSUS_TOLERANCE
                and min(full_margins) > full_reference_disagreement
            ),
        }
    return {
        "reynolds": reynolds,
        "reference_disagreement": reference_disagreement,
        "full_vector_reference_disagreement": full_reference_disagreement,
        "candidate_scores": candidate_scores,
        "orderings": orderings,
        "full_vector_orderings": full_vector_orderings,
    }


def build_record(rows: list[dict] | None = None) -> dict:
    configurations = CANDIDATES + REFERENCES
    if rows is None:
        rows = [
            _run_case(reynolds, configuration)
            for reynolds in REYNOLDS_NUMBERS
            for configuration in configurations
        ]
    return {
        "schema": SCHEMA,
        "source_sha256": _source_hashes(),
        "case": {
            "traction_estimators": ["strong", "nitsche_consistent"],
            "consensus_tolerance": CONSENSUS_TOLERANCE,
            "reynolds_numbers": list(REYNOLDS_NUMBERS),
            "candidates": list(CANDIDATES),
            "references": list(REFERENCES),
        },
        "poiseuille_force_balance": _poiseuille_check(),
        "rows": rows,
        "reference_scores": [
            score
            for reynolds in REYNOLDS_NUMBERS
            for score in _reference_scores(rows, reynolds)
        ],
        "candidate_scores": [
            _candidate_score(rows, reynolds) for reynolds in REYNOLDS_NUMBERS
        ],
    }


def verify_record(record: dict) -> None:
    if record.get("schema") != SCHEMA:
        raise ValueError("Nitsche traction schema mismatch")
    if record.get("source_sha256") != _source_hashes():
        raise ValueError("Nitsche traction source hash mismatch")
    configurations = CANDIDATES + REFERENCES
    expected = [
        (reynolds, configuration["name"])
        for reynolds in REYNOLDS_NUMBERS
        for configuration in configurations
    ]
    rows = record.get("rows", [])
    if [(row["reynolds"], row["name"]) for row in rows] != expected:
        raise ValueError("Nitsche traction run matrix changed")
    for check in record.get("poiseuille_force_balance", []):
        if float(check["maximum_force_balance_error"]) > 1.0e-12:
            raise AssertionError("Nitsche traction force balance failed")
    for row in rows:
        if float(row["maximum_relative_residual"]) > 1.0e-11:
            raise AssertionError("Nitsche traction residual failed")
        if float(row["maximum_divergence_relative_residual"]) > 2.0e-11:
            raise AssertionError("Nitsche traction divergence failed")
        if float(row["maximum_pressure_gauge_relative_residual"]) > 2.0e-11:
            raise AssertionError("Nitsche traction gauge failed")
        if float(row["maximum_energy_balance_residual"]) > 5.0e-9:
            raise AssertionError("Nitsche traction energy failed")
        if float(row["maximum_normal_flux"]) > 2.0e-13:
            raise AssertionError("Nitsche traction flux failed")
        for key in (
            "numerical_lid_reaction",
            "integrated_numerical_traction_work",
        ):
            if not math.isfinite(float(row[key])):
                raise ValueError("Nitsche traction wall metric is not finite")
    expected_reference_scores = [
        score
        for reynolds in REYNOLDS_NUMBERS
        for score in _reference_scores(rows, reynolds)
    ]
    if record.get("reference_scores") != expected_reference_scores:
        raise ValueError("Nitsche reference score mismatch")
    expected_candidate_scores = [
        _candidate_score(rows, reynolds) for reynolds in REYNOLDS_NUMBERS
    ]
    if record.get("candidate_scores") != expected_candidate_scores:
        raise ValueError("Nitsche candidate score mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    ensure_generated_dirs()
    if args.run:
        if OUTPUT.exists():
            raise FileExistsError(f"refusing to overwrite {OUTPUT}")
        record = build_record()
        OUTPUT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    elif args.rescore:
        trial = json.loads(FULL_VECTOR_TRIAL.read_text())
        record = build_record(trial["rows"])
        OUTPUT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    else:
        record = json.loads(OUTPUT.read_text())
    if args.run or args.verify:
        verify_record(record)
    print(json.dumps(record["reference_scores"], indent=2, sort_keys=True))
    print(json.dumps(record["candidate_scores"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
