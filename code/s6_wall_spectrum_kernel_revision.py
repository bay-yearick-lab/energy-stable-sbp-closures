#!/usr/bin/env python3
"""Freeze and test the single phase-resolved response-kernel revision.

The first holdout identifies signed cross-mode coupling as the missing state.
This revision projects the enforced acoustic generator onto a predeclared
cosine-sine packet band and advances that reduced generator through the same
RK4 stability polynomial.  A fresh packet family remains unseen until the
revision prediction artifact is committed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import s6_wall_spectrum_heldout as base
import s6_wall_spectrum_heldout_runner as runner


SCHEMA = "wall-spectrum-kernel-revision-v1"
SIGMAS = (0.065, 0.100)
CARRIERS = (8.0, 28.0, 44.0)
ENCOUNTERS = (6, 10)
PREDICTION_PATH = base.RESULTSDIR / "wall_spectrum_kernel_revision_predictions.json"
RAW_PATH = base.RESULTSDIR / "wall_spectrum_kernel_revision_raw.json"
SUMMARY_PATH = base.RESULTSDIR / "wall_spectrum_kernel_revision_summary.json"


def retained_mode_max(sigma: float, carrier: float) -> int:
    """Retain the carrier band through four Gaussian spectral widths."""
    bandwidth_modes = 4.0 / (np.pi * sigma)
    return min(base.NPTS - 3, int(math.ceil(carrier + bandwidth_modes)))


def modal_basis(method: str, data: dict[str, Any], mode_max: int) -> np.ndarray:
    """Build the pressure-cosine and velocity-sine response subspace."""
    pressure_x = np.asarray(data["x"], dtype=float)
    velocity_x = np.asarray(data["velocity_x"], dtype=float)
    pressure_size = int(data["pressure_size"])
    state_size = data["matrix"].shape[0]
    columns = []
    constant = np.zeros(state_size)
    constant[:pressure_size] = 1.0
    columns.append(constant)
    for mode in range(1, mode_max + 1):
        pressure_column = np.zeros(state_size)
        pressure_column[:pressure_size] = np.cos(mode * np.pi * pressure_x)
        velocity_column = np.zeros(state_size)
        velocity_column[pressure_size:] = np.sin(mode * np.pi * velocity_x)
        columns.extend((pressure_column, velocity_column))
    basis = np.column_stack(columns)
    orthogonal, _upper = np.linalg.qr(basis, mode="reduced")
    return orthogonal


def kernel_prediction(
    method: str, sigma: float, carrier: float, encounters: int
) -> dict[str, Any]:
    """Predict one packet through the truncated signed response kernel."""
    data = runner.generator(method, base.NPTS)
    mode_max = retained_mode_max(sigma, carrier)
    basis = modal_basis(method, data, mode_max)
    reduced_generator = basis.T @ data["matrix"] @ basis
    exact_pressure, exact_velocity = base.exact_pair(sigma, carrier * np.pi)
    initial = runner.initial_states(data, [(sigma, carrier)])[:, 0]
    reduced_initial = basis.T @ initial
    nsteps = int(np.ceil(encounters / (base.CFL * data["h"])))
    step = encounters / nsteps
    reduced_step = runner.rk4_step_map(reduced_generator, step)
    reduced_final = np.linalg.matrix_power(reduced_step, nsteps) @ reduced_initial
    predicted_state = basis @ reduced_final
    pressure, velocity = runner.unpack_state(method, data, predicted_state)
    metrics = base.score_fields(
        np.asarray(data["x"], dtype=float),
        pressure,
        velocity,
        exact_pressure,
        exact_velocity,
        float(encounters),
        carrier * np.pi,
        sigma,
    )
    features = base.spectrum_features(
        method, base.NPTS, sigma, carrier * np.pi, encounters
    )
    corrected_phase_driver = abs(encounters * features["mean_response_bias"])
    corrected_phase_spread = encounters * features["response_bias_spread"]
    corrected_contamination = encounters * features["contamination_rms"]
    return {
        "metrics": metrics,
        "features": features,
        "corrected_features": {
            "signed_mean_response_bias": features["mean_response_bias"],
            "spatial_phase_shift_driver": corrected_phase_driver,
            "signed_bias_spread_driver": corrected_phase_spread,
            "contamination_driver": corrected_contamination,
            "combined_distortion_driver": math.hypot(
                2.0 * np.pi * corrected_phase_spread,
                corrected_contamination,
            ),
        },
        "mode_max": mode_max,
        "reduced_dimension": int(basis.shape[1]),
        "full_dimension": int(data["matrix"].shape[0]),
        "rk4_step_count": nsteps,
    }


def freeze(output_path: Path) -> dict[str, Any]:
    """Write all second-holdout predictions before numerical fields exist."""
    if output_path.exists():
        raise FileExistsError(output_path)
    predictions = []
    for sigma in SIGMAS:
        for carrier in CARRIERS:
            for encounters in ENCOUNTERS:
                for method in base.METHODS:
                    result = kernel_prediction(method, sigma, carrier, encounters)
                    metrics = result["metrics"]
                    features = result["features"]
                    predictions.append({
                        "case_id": base.case_id(sigma, carrier, encounters, method),
                        "method": method,
                        "sigma": sigma,
                        "k0_over_pi": carrier,
                        "encounters": encounters,
                        "npts": base.NPTS,
                        "features": features,
                        "corrected_features": result["corrected_features"],
                        "kernel": {
                            "retained_mode_max": result["mode_max"],
                            "reduced_dimension": result["reduced_dimension"],
                            "full_dimension": result["full_dimension"],
                            "rk4_step_count": result["rk4_step_count"],
                        },
                        "predicted": {
                            "phase_shift": metrics["phase_shift"],
                            "waveform_distortion": metrics["waveform_distortion"],
                            "leakage_excess": metrics["leakage_excess"],
                            "leakage": metrics["leakage"],
                        },
                        "predicted_threshold_side": {
                            "response": features["response_bias_rms"] <= base.PRIMARY_LIMITS["response_bias"],
                            "contamination": features["contamination_rms"] <= base.PRIMARY_LIMITS["orthogonal_contamination"],
                            "phase": carrier * np.pi * metrics["phase_shift"] / encounters <= base.PRIMARY_LIMITS["phase_normalized"],
                            "distortion": metrics["waveform_distortion"] <= base.PRIMARY_LIMITS["waveform_distortion"],
                            "leakage": metrics["leakage_excess"] <= base.PRIMARY_LIMITS["leakage_excess"],
                        },
                    })
    record = {
        "schema": SCHEMA,
        "artifact_state": "immutable_pre_result_predictions",
        "revision_count": 1,
        "diagnosis_inherited": "scalar modal RMS compression loses signed cross-mode coupling",
        "dimensional_diagnosis": {
            "defect": "the first frozen phase driver divided encounter-scaled response bias by carrier mode",
            "derivation": "omega equals k rho, temporal phase error equals k T (rho minus 1), and spatial shift equals temporal phase error divided by k, giving T (rho minus 1)",
            "threshold_reconciliation": "k0 times shift divided by encounters estimates carrier-scaled response bias and contains no inverse carrier factor",
        },
        "model_family": "phase-resolved truncated cosine-sine response kernel",
        "information_boundary": {
            "available": [
                "reported packet calibration",
                "first held-out failure diagnosis",
                "enforced operator response matrices",
            ],
            "forbidden": "second-held-out numerical fields and metrics",
        },
        "primary_matrix": {
            "methods": list(base.METHODS),
            "sigmas": list(SIGMAS),
            "k0_over_pi": list(CARRIERS),
            "encounters": list(ENCOUNTERS),
            "npts": base.NPTS,
            "case_count": len(predictions),
        },
        "decision_contract": {
            "relative_tolerance": base.RELATIVE_TOLERANCE,
            "absolute_floors": base.ABSOLUTE_FLOORS,
            "thresholds": base.PRIMARY_LIMITS,
            "promotion": "every second-holdout case must pass quantitative metrics, threshold sides, and strict family ordering",
            "scalar_spectrum_status": "diagnostic_ranker_only regardless of enriched-kernel outcome",
        },
        "truncation_contract": "cosine-sine modes through carrier plus four divided by pi sigma, capped at N minus 3",
        "semidiscrete_model": {
            "signed_bias": "packet-weighted signed response bias accumulates as encounters times mean bias in spatial shift",
            "spread": "signed modal bias variation accumulates across the packet band and distorts the waveform",
            "contamination": "the phase-resolved off-target response is retained through the projected generator rather than reduced to one unsigned RMS scalar",
            "evolution": "the reduced enforced generator is advanced by the same classical RK4 stability polynomial and declared step count",
        },
        "predictions": predictions,
        "source_sha256": {
            "revision_driver": base.sha256(Path(__file__)),
            "frozen_scalar_predictions": base.sha256(base.PREDICTION_PATH),
            "first_holdout_raw": base.sha256(base.RAW_PATH),
            "first_holdout_summary": base.sha256(base.SUMMARY_PATH),
        },
    }
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def run(prediction_path: Path, output_path: Path) -> dict[str, Any]:
    """Execute the fresh second holdout once through the full RK4 maps."""
    if output_path.exists():
        raise FileExistsError(output_path)
    prediction = json.loads(prediction_path.read_text())
    packet_specs = [(sigma, carrier) for sigma in SIGMAS for carrier in CARRIERS]
    by_id = {}
    for method in base.METHODS:
        for encounters in ENCOUNTERS:
            metrics_rows = runner.march_group(
                method, base.NPTS, packet_specs, encounters
            )
            for (sigma, carrier), metrics in zip(packet_specs, metrics_rows, strict=True):
                by_id[base.case_id(sigma, carrier, encounters, method)] = metrics
            print(f"{method} encounters={encounters}: second holdout complete", flush=True)
    raw = {
        "schema": SCHEMA,
        "artifact_state": "heldout_raw_once",
        "prediction_sha256": base.sha256(prediction_path),
        "runner_sha256": base.sha256(Path(__file__)),
        "results": [
            {"case_id": item["case_id"], "metrics": by_id[item["case_id"]]}
            for item in prediction["predictions"]
        ],
    }
    output_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    return raw


def summarize(prediction_path: Path, raw_path: Path, output_path: Path) -> dict[str, Any]:
    """Apply the unchanged acceptance contract to the second holdout."""
    summary = base.summarize(prediction_path, raw_path, output_path)
    summary["schema"] = SCHEMA
    summary["revision_count"] = 1
    summary["scalar_spectrum_status"] = "diagnostic_ranker_only"
    if summary["decision"]["promotion_earned"]:
        summary["decision"]["claim_status"] = "accepted_enriched_kernel_predictor"
        summary["failure_mechanism"] = None
        summary["classification"] = None
    else:
        summary["decision"]["claim_status"] = "diagnostic_ranker_only"
        summary["failure_mechanism"] = (
            "The truncated signed response kernel still omits material packet-band transfer or evaluator-scale structure on the fresh family."
        )
        summary["classification"] = "model_reduction_defect"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def verify(prediction_path: Path, raw_path: Path, summary_path: Path) -> None:
    """Verify second-holdout completeness, binding, and revision count."""
    prediction = json.loads(prediction_path.read_text())
    raw = json.loads(raw_path.read_text())
    summary = json.loads(summary_path.read_text())
    expected = {
        base.case_id(sigma, carrier, encounters, method)
        for sigma in SIGMAS
        for carrier in CARRIERS
        for encounters in ENCOUNTERS
        for method in base.METHODS
    }
    assert prediction["revision_count"] == summary["revision_count"] == 1
    assert {row["case_id"] for row in prediction["predictions"]} == expected
    assert {row["case_id"] for row in raw["results"]} == expected
    assert {row["case_id"] for row in summary["cases"]} == expected
    assert raw["prediction_sha256"] == base.sha256(prediction_path)
    assert summary["prediction_sha256"] == base.sha256(prediction_path)
    assert summary["raw_sha256"] == base.sha256(raw_path)
    assert summary["scalar_spectrum_status"] == "diagnostic_ranker_only"
    print("wall_spectrum_kernel_revision verify: ALL PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--output", type=Path, default=PREDICTION_PATH)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--prediction", type=Path, default=PREDICTION_PATH)
    run_parser.add_argument("--output", type=Path, default=RAW_PATH)
    summarize_parser = commands.add_parser("summarize")
    summarize_parser.add_argument("--prediction", type=Path, default=PREDICTION_PATH)
    summarize_parser.add_argument("--raw", type=Path, default=RAW_PATH)
    summarize_parser.add_argument("--output", type=Path, default=SUMMARY_PATH)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--prediction", type=Path, default=PREDICTION_PATH)
    verify_parser.add_argument("--raw", type=Path, default=RAW_PATH)
    verify_parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "freeze":
        record = freeze(args.output)
        print(f"wrote {args.output} with {len(record['predictions'])} predictions")
    elif args.command == "run":
        record = run(args.prediction, args.output)
        print(f"wrote {args.output} with {len(record['results'])} results")
    elif args.command == "summarize":
        summary = summarize(args.prediction, args.raw, args.output)
        print(json.dumps(summary["decision"], indent=2))
    elif args.command == "verify":
        verify(args.prediction, args.raw, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
