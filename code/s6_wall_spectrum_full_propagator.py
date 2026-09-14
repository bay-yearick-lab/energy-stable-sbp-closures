#!/usr/bin/env python3
"""Full metric-aware wall propagator and final untouched packet test.

The scalar wall spectrum records target response and contamination norms.  The
full completion retains the enforced generator itself, expressed in an energy
metric and factored once by a real Schur decomposition.  Matrix exponentials of
the Schur form predict arbitrary bounce times without RK4 marching, modal
truncation, or loss of signed wall coupling.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import block_diag, expm, schur

import s6_wall_spectrum_heldout as base
import s6_wall_spectrum_heldout_runner as runner


SCHEMA = "wall-spectrum-full-propagator-v1"
SIGMAS = (0.075,)
CARRIERS = (12.0, 36.0)
ENCOUNTERS = (5, 9)
PREDICTION_PATH = base.RESULTSDIR / "wall_spectrum_full_propagator_predictions.json"
RAW_PATH = base.RESULTSDIR / "wall_spectrum_full_propagator_raw.json"
SUMMARY_PATH = base.RESULTSDIR / "wall_spectrum_full_propagator_summary.json"


def spectral_symmetrizer(matrix: np.ndarray) -> np.ndarray:
    """Construct a positive symmetrizer for a stable diagonalizable wave pair."""
    _values, vectors = np.linalg.eig(matrix)
    inverse = np.linalg.inv(vectors)
    metric = np.real(inverse.conj().T @ inverse)
    return 0.5 * (metric + metric.T)


def state_metric(method: str, data: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Return the positive state metric used for the Schur similarity map."""
    npts = int(data["pressure_size"])
    if method == "BMF6":
        operator = base.ba.build_operator(
            base.ba.scheme_kind(method), base.ORDER, base.NPTS
        )
        staggered = operator.staggered_operators()
        return block_diag(staggered.h0, staggered.h1), "native staggered h0/h1 energy metric"
    if method == "BMG6":
        metric = base.ba.bmg_nodal_operator(base.NPTS, base.ORDER)[2]
        return block_diag(metric, metric), "native BMG spline mass metric"
    if method == "SBP6":
        operator = base.ba.build_operator(
            base.ba.scheme_kind(method), base.ORDER, base.NPTS
        )
        metric = np.asarray(operator.metric, dtype=float)
        return block_diag(metric, metric[1:-1, 1:-1]), "classical SBP norm"
    if method == "SEM6":
        sem = base.sem_grid_for(base.NPTS, base.ORDER)
        metric = np.diag(sem.mass)
        return block_diag(metric, metric[1:-1, 1:-1]), "assembled GLL mass metric"
    if method == "CSB-T6":
        return spectral_symmetrizer(data["matrix"]), "positive wave-pair symmetrizer derived from the stable eliminated generator"
    raise ValueError(method)


def factor(method: str) -> dict[str, Any]:
    """Factor one full enforced generator in metric coordinates."""
    start = time.perf_counter()
    data = runner.generator(method, base.NPTS)
    full_metric, metric_source = state_metric(method, data)
    full_dimension = data["matrix"].shape[0]
    if method in ("BMF6", "BMG6"):
        pressure_size = int(data["pressure_size"])
        retained = np.ones(full_dimension, dtype=bool)
        retained[[pressure_size, full_dimension - 1]] = False
        injection = np.eye(full_dimension)[:, retained]
    else:
        injection = np.eye(full_dimension)
    restriction = injection.T
    matrix = restriction @ data["matrix"] @ injection
    metric = injection.T @ full_metric @ injection
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    if float(np.min(eigenvalues)) <= 0.0:
        raise ValueError(f"nonpositive state metric for {method}")
    square_root = np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse_square_root = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues))
    transformed = square_root @ matrix @ inverse_square_root
    schur_form, schur_vectors = schur(transformed, output="real")
    factor_seconds = time.perf_counter() - start
    similarity_residual = np.linalg.norm(
        transformed - schur_vectors @ schur_form @ schur_vectors.T
    ) / max(np.linalg.norm(transformed), 1.0e-300)
    metric_skew_residual = np.linalg.norm(
        matrix.T @ metric + metric @ matrix
    ) / max(np.linalg.norm(metric @ matrix), 1.0e-300)
    return {
        "method": method,
        "data": data,
        "metric": metric,
        "matrix": matrix,
        "injection": injection,
        "restriction": restriction,
        "metric_source": metric_source,
        "square_root": square_root,
        "inverse_square_root": inverse_square_root,
        "schur_form": schur_form,
        "schur_vectors": schur_vectors,
        "factor_seconds": factor_seconds,
        "similarity_residual": float(similarity_residual),
        "metric_skew_residual": float(metric_skew_residual),
        "metric_condition": float(np.max(eigenvalues) / np.min(eigenvalues)),
    }


def propagate(
    factorization: dict[str, Any], states: np.ndarray, time_value: float
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply the full arbitrary-time Schur propagator to packet states."""
    build_start = time.perf_counter()
    exponential = expm(factorization["schur_form"] * time_value)
    left = factorization["inverse_square_root"] @ factorization["schur_vectors"]
    right = factorization["schur_vectors"].T @ factorization["square_root"]
    reduced_propagator = left @ exponential @ right
    propagator = (
        factorization["injection"]
        @ reduced_propagator
        @ factorization["restriction"]
    )
    build_seconds = time.perf_counter() - build_start
    evaluate_start = time.perf_counter()
    final = propagator @ states
    evaluate_seconds = time.perf_counter() - evaluate_start
    return final, {
        "propagator_build_seconds": build_seconds,
        "packet_batch_evaluation_seconds": evaluate_seconds,
        "packet_count": int(states.shape[1]),
        "per_packet_evaluation_seconds": evaluate_seconds / states.shape[1],
        "storage_entries": int(propagator.size),
    }


def freeze(output_path: Path) -> dict[str, Any]:
    """Freeze full-propagator predictions for the third untouched family."""
    if output_path.exists():
        raise FileExistsError(output_path)
    packet_specs = [(sigma, carrier) for sigma in SIGMAS for carrier in CARRIERS]
    predictions_by_id = {}
    costs = []
    factorization_records = []
    for method in base.METHODS:
        decomposition = factor(method)
        data = decomposition["data"]
        initial = runner.initial_states(data, packet_specs)
        factorization_records.append({
            "method": method,
            "state_dimension": int(decomposition["matrix"].shape[0]),
            "full_state_dimension": int(data["matrix"].shape[0]),
            "metric_source": decomposition["metric_source"],
            "metric_condition": decomposition["metric_condition"],
            "metric_skew_residual": decomposition["metric_skew_residual"],
            "real_schur_similarity_residual": decomposition["similarity_residual"],
            "factor_seconds": decomposition["factor_seconds"],
        })
        for encounters in ENCOUNTERS:
            final, cost = propagate(decomposition, initial, float(encounters))
            costs.append({"method": method, "encounters": encounters, **cost})
            for column, (sigma, carrier) in enumerate(packet_specs):
                pressure, velocity = runner.unpack_state(
                    method, data, final[:, column]
                )
                exact_pressure, exact_velocity = base.exact_pair(
                    sigma, carrier * np.pi
                )
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
                identifier = base.case_id(sigma, carrier, encounters, method)
                predictions_by_id[identifier] = {
                    "case_id": identifier,
                    "method": method,
                    "sigma": sigma,
                    "k0_over_pi": carrier,
                    "encounters": encounters,
                    "npts": base.NPTS,
                    "features": features,
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
                }
    predictions = [
        predictions_by_id[base.case_id(sigma, carrier, encounters, method)]
        for sigma in SIGMAS
        for carrier in CARRIERS
        for encounters in ENCOUNTERS
        for method in base.METHODS
    ]
    record = {
        "schema": SCHEMA,
        "artifact_state": "immutable_pre_result_predictions",
        "predictor_scope": "full metric-aware semidiscrete wall propagator",
        "scalar_spectrum_status": "diagnostic_ranker_only",
        "truncated_kernel_status": "rejected_model_reduction",
        "operator_completion": "rho and eta are scalar projections of the enforced generator response; the full propagator retains every signed target, orthogonal, and cross-mode component through arbitrary time",
        "information_boundary": "third-family RK4 fields and metrics forbidden before this artifact is committed",
        "primary_matrix": {
            "methods": list(base.METHODS),
            "sigmas": list(SIGMAS),
            "k0_over_pi": list(CARRIERS),
            "encounters": list(ENCOUNTERS),
            "case_count": len(predictions),
            "justification": "one untouched width, one resolved and one threshold-crossing carrier, and two untouched nonreturn bounce times",
        },
        "decision_contract": {
            "relative_tolerance": base.RELATIVE_TOLERANCE,
            "absolute_floors": base.ABSOLUTE_FLOORS,
            "thresholds": base.PRIMARY_LIMITS,
            "promotion": "all 20 quantitative cases, threshold sides, and strict family orders must pass",
            "failure": "scientific_rejection of the full propagator predictor after implementation faithfulness is established",
        },
        "cost_model": {
            "one_time_factorization": "metric square root plus real Schur factorization, cubic in state dimension",
            "per_time_setup": "matrix exponential of the real Schur form and propagator assembly, cubic in state dimension",
            "per_packet_evaluation": "dense propagator times one state, quadratic in state dimension",
            "measured_factorizations": factorization_records,
            "measured_evaluations": costs,
        },
        "predictions": predictions,
        "source_sha256": {
            "driver": base.sha256(Path(__file__)),
            "verified_rk4_runner": base.sha256(Path(runner.__file__)),
            "scalar_failure": base.sha256(base.SUMMARY_PATH),
            "truncated_kernel_failure": base.sha256(
                base.RESULTSDIR / "wall_spectrum_kernel_revision_summary.json"
            ),
        },
    }
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def run(prediction_path: Path, output_path: Path) -> dict[str, Any]:
    """March the third untouched family once with the verified RK4 realization."""
    if output_path.exists():
        raise FileExistsError(output_path)
    prediction = json.loads(prediction_path.read_text())
    packet_specs = [(sigma, carrier) for sigma in SIGMAS for carrier in CARRIERS]
    by_id = {}
    for method in base.METHODS:
        for encounters in ENCOUNTERS:
            rows = runner.march_group(method, base.NPTS, packet_specs, encounters)
            for (sigma, carrier), metrics in zip(packet_specs, rows, strict=True):
                by_id[base.case_id(sigma, carrier, encounters, method)] = metrics
            print(f"{method} encounters={encounters}: third holdout complete", flush=True)
    raw = {
        "schema": SCHEMA,
        "artifact_state": "heldout_raw_once",
        "prediction_sha256": base.sha256(prediction_path),
        "runner_sha256": base.sha256(Path(runner.__file__)),
        "results": [
            {"case_id": item["case_id"], "metrics": by_id[item["case_id"]]}
            for item in prediction["predictions"]
        ],
    }
    output_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    return raw


def summarize(prediction_path: Path, raw_path: Path, output_path: Path) -> dict[str, Any]:
    """Apply the unchanged quantitative and ordering rules to the third family."""
    summary = base.summarize(prediction_path, raw_path, output_path)
    summary["schema"] = SCHEMA
    summary["scalar_spectrum_status"] = "diagnostic_ranker_only"
    summary["truncated_kernel_status"] = "rejected_model_reduction"
    if summary["decision"]["promotion_earned"]:
        summary["decision"]["claim_status"] = "accepted_full_propagator_predictor"
        summary["classification"] = None
        summary["failure_mechanism"] = None
    else:
        summary["decision"]["claim_status"] = "full_propagator_rejected"
        summary["classification"] = "scientific_rejection"
        summary["failure_mechanism"] = (
            "The full metric-aware enforced-generator propagator does not predict the verified RK4 packet outcomes within the frozen rules."
        )
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def verify(prediction_path: Path, raw_path: Path, summary_path: Path) -> None:
    """Verify third-family completeness, binding, factorization, and decision."""
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
    assert {row["case_id"] for row in prediction["predictions"]} == expected
    assert {row["case_id"] for row in raw["results"]} == expected
    assert {row["case_id"] for row in summary["cases"]} == expected
    assert len(expected) == 20
    assert raw["prediction_sha256"] == base.sha256(prediction_path)
    assert summary["prediction_sha256"] == base.sha256(prediction_path)
    assert summary["raw_sha256"] == base.sha256(raw_path)
    for record in prediction["cost_model"]["measured_factorizations"]:
        assert record["real_schur_similarity_residual"] < 1.0e-12
        assert record["metric_skew_residual"] < 1.0e-10
        assert record["factor_seconds"] >= 0.0
    print("wall_spectrum_full_propagator verify: ALL PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--output", type=Path, default=PREDICTION_PATH)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--prediction", type=Path, default=PREDICTION_PATH)
    run_parser.add_argument("--output", type=Path, default=RAW_PATH)
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("--prediction", type=Path, default=PREDICTION_PATH)
    summary_parser.add_argument("--raw", type=Path, default=RAW_PATH)
    summary_parser.add_argument("--output", type=Path, default=SUMMARY_PATH)
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
