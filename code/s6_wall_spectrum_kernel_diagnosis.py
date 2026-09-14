#!/usr/bin/env python3
"""Quantify why the phase-resolved packet kernel fails to close in time."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import s6_wall_spectrum_heldout as base
import s6_wall_spectrum_heldout_runner as runner
import s6_wall_spectrum_kernel_revision as revision


OUTPUT = base.RESULTSDIR / "wall_spectrum_kernel_revision_diagnosis.json"


def strict_order(values: dict[str, float]) -> list[str]:
    """Return method labels from the smallest value to the largest."""
    return [method for method, _value in sorted(values.items(), key=lambda item: item[1])]


def pairwise_fraction(predicted: dict[str, float], actual: dict[str, float]) -> float:
    """Return the fraction of method pairs ordered in the same direction."""
    methods = sorted(predicted)
    matches = 0
    total = 0
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1:]:
            predicted_sign = np.sign(predicted[left] - predicted[right])
            actual_sign = np.sign(actual[left] - actual[right])
            matches += int(predicted_sign == actual_sign)
            total += 1
    return matches / total


def subspace_records() -> list[dict[str, float | str]]:
    """Measure initial omission and two-way generator coupling by packet band."""
    rows = []
    for method in base.METHODS:
        data = runner.generator(method, base.NPTS)
        matrix = np.asarray(data["matrix"], dtype=float)
        identity = np.eye(matrix.shape[0])
        for sigma in revision.SIGMAS:
            for carrier in revision.CARRIERS:
                mode_max = revision.retained_mode_max(sigma, carrier)
                basis = revision.modal_basis(method, data, mode_max)
                complement = identity - basis @ basis.T
                outward = complement @ matrix @ basis
                backward = basis.T @ matrix @ complement
                initial = runner.initial_states(data, [(sigma, carrier)])[:, 0]
                rows.append({
                    "method": method,
                    "sigma": sigma,
                    "k0_over_pi": carrier,
                    "retained_mode_max": mode_max,
                    "reduced_dimension": int(basis.shape[1]),
                    "full_dimension": int(matrix.shape[0]),
                    "initial_complement_fraction": float(
                        np.linalg.norm(complement @ initial) / np.linalg.norm(initial)
                    ),
                    "outward_coupling_2norm": float(np.linalg.norm(outward, 2)),
                    "return_coupling_2norm": float(np.linalg.norm(backward, 2)),
                    "relative_outward_coupling": float(
                        np.linalg.norm(outward, 2)
                        / max(np.linalg.norm(matrix @ basis, 2), 1.0e-300)
                    ),
                })
    return rows


def heldout_ranker(summary_path: Path) -> dict[str, object]:
    """Score the scalar response and contamination ordering on one holdout."""
    summary = json.loads(summary_path.read_text())
    packet_keys = sorted({
        (row["sigma"], row["k0_over_pi"], row["encounters"])
        for row in summary["cases"]
    })
    checks = []
    for sigma, carrier, encounters in packet_keys:
        group = [
            row for row in summary["cases"]
            if (row["sigma"], row["k0_over_pi"], row["encounters"])
            == (sigma, carrier, encounters)
        ]
        prediction_rows = {
            item["case_id"]: item
            for item in json.loads(
                (revision.PREDICTION_PATH if summary_path == revision.SUMMARY_PATH else base.PREDICTION_PATH).read_text()
            )["predictions"]
        }
        response = {
            row["method"]: abs(prediction_rows[row["case_id"]]["features"]["mean_response_bias"])
            for row in group
        }
        contamination = {
            row["method"]: prediction_rows[row["case_id"]]["features"]["contamination_rms"]
            for row in group
        }
        for metric, ranker in (
            ("phase_shift", response),
            ("waveform_distortion", contamination),
            ("leakage", contamination),
        ):
            actual = {row["method"]: row["actual"][metric] for row in group}
            checks.append({
                "sigma": sigma,
                "k0_over_pi": carrier,
                "encounters": encounters,
                "metric": metric,
                "strict_order_pass": strict_order(ranker) == strict_order(actual),
                "pairwise_order_fraction": pairwise_fraction(ranker, actual),
            })
    return {
        "strict_order_pass_count": int(sum(row["strict_order_pass"] for row in checks)),
        "strict_order_total": len(checks),
        "median_pairwise_order_fraction": float(
            np.median([row["pairwise_order_fraction"] for row in checks])
        ),
        "checks": checks,
    }


def reported_packet_ranker() -> dict[str, object]:
    """Preserve the established BMF, BMG, and SBP ordering on the reported packet."""
    packet = json.loads((base.RESULTSDIR / "packet_reflection_summary.json").read_text())
    methods = ("BMF6", "BMG6", "SBP6")
    checks = []
    for encounters, cycle in ((2, "cycle1"), (6, "cycle3")):
        features = {
            method: base.spectrum_features(
                method, 97, base.reported.SIGMA, base.reported.K0, encounters
            )
            for method in methods
        }
        metrics = {
            method: packet["runs"][f"{method}_N97"][cycle]
            for method in methods
        }
        for metric, feature in (
            ("phase_shift", "mean_response_bias"),
            ("waveform_distortion", "contamination_rms"),
            ("leakage", "contamination_rms"),
        ):
            predicted = {
                method: abs(features[method][feature]) for method in methods
            }
            actual = {method: abs(metrics[method][metric]) for method in methods}
            checks.append({
                "encounters": encounters,
                "metric": metric,
                "predicted_order": strict_order(predicted),
                "actual_order": strict_order(actual),
                "strict_order_pass": strict_order(predicted) == strict_order(actual),
                "pairwise_order_fraction": pairwise_fraction(predicted, actual),
            })
    return {
        "scope": "reported packet, q6, N97, BMF6/BMG6/classical SBP6",
        "strict_order_pass_count": int(sum(row["strict_order_pass"] for row in checks)),
        "strict_order_total": len(checks),
        "median_pairwise_order_fraction": float(
            np.median([row["pairwise_order_fraction"] for row in checks])
        ),
        "checks": checks,
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    subspaces = subspace_records()
    record = {
        "classification": "model_reduction_defect",
        "quantitative_predictor_status": "scalar and truncated-kernel predictors rejected; full operator-valued response unresolved",
        "scalar_spectrum_status": "mechanism-aligned diagnostic and reported-packet ranker only",
        "dimensional_defect": "the first phase driver incorrectly divided spatial shift by carrier mode",
        "corrected_phase_relation": "absolute spatial shift equals absolute encounters times packet-weighted signed response bias at leading order",
        "residual_theory_question": "Can a finite-time Duhamel or pseudospectral bound using outward and return coupling blocks predict phase and distortion without advancing the full enforced generator?",
        "subspace_coupling": {
            "records": subspaces,
            "initial_complement_fraction_median": float(np.median([row["initial_complement_fraction"] for row in subspaces])),
            "initial_complement_fraction_worst": float(np.max([row["initial_complement_fraction"] for row in subspaces])),
            "relative_outward_coupling_median": float(np.median([row["relative_outward_coupling"] for row in subspaces])),
            "relative_outward_coupling_worst": float(np.max([row["relative_outward_coupling"] for row in subspaces])),
        },
        "first_holdout_scalar_ranker": heldout_ranker(base.SUMMARY_PATH),
        "successor_holdout_scalar_ranker": heldout_ranker(revision.SUMMARY_PATH),
        "strongest_preserved_ranker_result": reported_packet_ranker(),
        "fallback": "retain rho and eta curves for mechanism diagnosis and the reported-packet ranking; do not make a quantitative held-out predictor claim",
    }
    OUTPUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "subspace_coupling": record["subspace_coupling"],
        "first_ranker": {key: value for key, value in record["first_holdout_scalar_ranker"].items() if key != "checks"},
        "successor_ranker": {key: value for key, value in record["successor_holdout_scalar_ranker"].items() if key != "checks"},
        "reported_ranker": {key: value for key, value in record["strongest_preserved_ranker_result"].items() if key != "checks"},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
