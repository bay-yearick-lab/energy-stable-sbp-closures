#!/usr/bin/env python3
"""Freeze and test a finite-time wall-spectrum packet predictor.

The calibration record is restricted to the packet already reported in the
article.  The primary family changes envelope width, carrier, and encounter
count.  Prediction generation and held-out marching are separate commands so
the prediction artifact can be committed before primary fields are produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))

import boundary_spline_analysis as ba
import baseline_closures as bl
import s4_staggered_wallmode as wall
import s6_packet_reflection as reported
from project_paths import FIGDIR, RESULTSDIR, ensure_generated_dirs
from sem_operator import sem_grid_for


SCHEMA = "wall-spectrum-heldout-v1"
ORDER = 6
NPTS = 49
CFL = 0.012
X0 = 0.35
METHODS = ("BMF6", "BMG6", "SBP6", "CSB-T6", "SEM6")
CALIBRATION_GRIDS = (49, 97)
CALIBRATION_CYCLES = (1, 3)
HOLDOUT_SIGMAS = (0.055, 0.085)
HOLDOUT_K0_OVER_PI = (4.0, 16.0, 40.0)
HOLDOUT_ENCOUNTERS = (4, 8)
PRIMARY_LIMITS = {
    "response_bias": 0.01,
    "orthogonal_contamination": 0.01,
    "phase_normalized": 0.01,
    "waveform_distortion": 0.01,
    "leakage_excess": 0.01,
}
ABSOLUTE_FLOORS = {
    "phase_shift": 1.0e-10,
    "waveform_distortion": 1.0e-10,
    "leakage": 1.0e-10,
}
RELATIVE_TOLERANCE = 0.20
CSB_T6_CANDIDATE = 3
PREDICTION_PATH = RESULTSDIR / "wall_spectrum_heldout_predictions.json"
RAW_PATH = RESULTSDIR / "wall_spectrum_heldout_raw.json"
SUMMARY_PATH = RESULTSDIR / "wall_spectrum_heldout_summary.json"


@dataclass(frozen=True)
class DenseOperator(ba.Operator1D):
    """First-derivative operator with explicit coordinates and matrix."""

    label: str
    derivative: np.ndarray
    coordinates: np.ndarray

    @property
    def name(self) -> str:
        return "SBP6"

    @property
    def order(self) -> int:
        return ORDER

    @property
    def x(self) -> np.ndarray:
        return self.coordinates

    @property
    def matrix(self) -> np.ndarray:
        return self.derivative

    @property
    def work_model(self) -> ba.WorkModel:
        nnz = int(np.count_nonzero(np.abs(self.derivative) > 1.0e-14))
        return ba.WorkModel(2 * nnz, 0, 2 * nnz, nnz)

    def differentiate(self, values: np.ndarray) -> np.ndarray:
        return self.derivative @ np.asarray(values, dtype=float)


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packet_function(sigma: float, k0: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return the right-moving Gaussian carrier used by the image solution."""
    def packet(xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=float)
        return np.exp(-((xi - X0) / sigma) ** 2) * np.cos(k0 * (xi - X0))
    return packet


def exact_pair(sigma: float, k0: float, images: int = 10) -> tuple[Callable, Callable]:
    """Construct the exact rigid-wall pressure and velocity image pair."""
    packet = packet_function(sigma, k0)

    def pressure(x: np.ndarray, time: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        for image in range(-images, images + 1):
            out += packet(2.0 * image + x - time)
            out += packet(2.0 * image - x - time)
        return out

    def velocity(x: np.ndarray, time: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        for image in range(-images, images + 1):
            out += packet(2.0 * image + x - time)
            out -= packet(2.0 * image - x - time)
        return out

    return pressure, velocity


def dense_operator(method: str, npts: int) -> DenseOperator:
    """Assemble a non-spline same-grid method used by the held-out study."""
    if method == "CSB-T6":
        derivative, _weights, _candidate = bl.csb_t6_operator(
            npts, candidate=CSB_T6_CANDIDATE
        )
        coordinates = np.linspace(0.0, 1.0, npts)
    elif method == "SEM6":
        sem = sem_grid_for(npts, ORDER)
        derivative = sem.D1
        coordinates = sem.x
    else:
        raise ValueError(method)
    return DenseOperator(method, derivative, coordinates)


def samegrid_pressure_response(operator: ba.Operator1D) -> list[dict[str, float]]:
    """Compute the pressure-generator response under rigid velocity traces."""
    x = np.asarray(operator.x, dtype=float)
    derivative = np.asarray(operator.matrix, dtype=float)
    npts = x.size
    pressure_block = derivative[:, 1:-1] @ derivative[1:-1, :]
    weights = ba.nodal_trapezoidal_weights(x)
    href = ba.reference_spacing(x)
    max_mode = min(npts - 2, int(np.floor((1.0 - 1.0e-12) / href)))
    rows = []
    for mode in range(1, max_mode + 1):
        wavenumber = mode * np.pi
        target = np.cos(wavenumber * x)
        image = pressure_block @ target
        denominator = float(np.sum(weights * target * target))
        kappa2 = -float(np.sum(weights * image * target)) / denominator
        rho = math.sqrt(max(kappa2, 0.0)) / wavenumber
        remainder = image + kappa2 * target
        eta = math.sqrt(max(float(np.sum(weights * remainder * remainder)), 0.0))
        eta /= wavenumber ** 2 * math.sqrt(denominator)
        rows.append({
            "mode": mode,
            "ppw": 2.0 / (mode * href),
            "rho": rho,
            "eta": eta,
        })
    return rows


def spectrum_rows(method: str, npts: int) -> list[dict[str, float]]:
    """Return the marched pressure response for one declared method."""
    if method == "BMF6":
        return wall.second_order_response("BMF-staggered", ORDER, npts)["rows"]
    if method == "BMG6":
        return wall.second_order_response("BMG", ORDER, npts)["rows"]
    if method == "SBP6":
        return wall.second_order_response("SBP", ORDER, npts)["rows"]
    return samegrid_pressure_response(dense_operator(method, npts))


def modal_weights(sigma: float, k0: float, max_mode: int) -> np.ndarray:
    """Project the initial pressure packet onto rigid-wall cosine modes."""
    x = np.linspace(0.0, 1.0, 16001)
    pressure, _velocity = exact_pair(sigma, k0)
    values = pressure(x, 0.0)
    coefficients = []
    for mode in range(1, max_mode + 1):
        coefficient = 2.0 * float(np.trapezoid(values * np.cos(mode * np.pi * x), x))
        coefficients.append(coefficient)
    energy = np.square(np.asarray(coefficients, dtype=float))
    total = float(np.sum(energy))
    if total <= 0.0:
        raise ValueError("packet projection has zero modal energy")
    return energy / total


def spectrum_features(
    method: str, npts: int, sigma: float, k0: float, encounters: int
) -> dict[str, float]:
    """Reduce response and contamination curves over the packet bandwidth."""
    rows = spectrum_rows(method, npts)
    weights = modal_weights(sigma, k0, len(rows))
    rho = np.asarray([row["rho"] for row in rows], dtype=float)
    eta = np.asarray([row["eta"] for row in rows], dtype=float)
    modes = np.arange(1, len(rows) + 1, dtype=float)
    mean_bias = float(np.sum(weights * (rho - 1.0)))
    bias_spread = math.sqrt(float(np.sum(weights * ((rho - 1.0) - mean_bias) ** 2)))
    contamination = math.sqrt(float(np.sum(weights * eta ** 2)))
    carrier_mode = k0 / np.pi
    cycles = encounters / 2.0
    phase_driver = abs(2.0 * cycles * mean_bias / max(carrier_mode, 1.0e-30))
    phase_spread = 2.0 * np.pi * cycles * math.sqrt(
        float(np.sum(weights * modes ** 2 * ((rho - 1.0) - mean_bias) ** 2))
    )
    contamination_driver = encounters * contamination
    distortion_driver = math.hypot(phase_spread, contamination_driver)
    leakage_driver = contamination_driver ** 2
    return {
        "mean_response_bias": mean_bias,
        "response_bias_rms": math.sqrt(float(np.sum(weights * (rho - 1.0) ** 2))),
        "response_bias_spread": bias_spread,
        "contamination_rms": contamination,
        "phase_driver": phase_driver,
        "distortion_driver": distortion_driver,
        "leakage_driver": leakage_driver,
        "carrier_mode": carrier_mode,
        "modal_mean_mode": float(np.sum(weights * modes)),
    }


def score_fields(
    x: np.ndarray,
    pressure: np.ndarray,
    velocity: np.ndarray,
    exact_pressure: Callable,
    exact_velocity: Callable,
    time_value: float,
    k0: float,
    sigma: float,
) -> dict[str, float]:
    """Evaluate the unchanged packet phase, distortion, and leakage metrics."""
    reference_p = exact_pressure(x, time_value)
    reference_u = exact_velocity(x, time_value)
    shift, amplitude, distortion = reported.phase_amp_distortion(
        x, pressure, reference_p
    )
    original_k0, original_sigma = reported.K0, reported.SIGMA
    reported.K0, reported.SIGMA = k0, sigma
    try:
        numerical_leakage = reported.leakage(x, pressure)
        exact_leakage = reported.leakage(x, reference_p)
    finally:
        reported.K0, reported.SIGMA = original_k0, original_sigma
    weights = ba.nodal_trapezoidal_weights(np.asarray(x, dtype=float))
    numerator = np.sum(weights * ((pressure - reference_p) ** 2 + (velocity - reference_u) ** 2))
    denominator = np.sum(weights * (reference_p ** 2 + reference_u ** 2))
    return {
        "phase_shift": abs(float(shift)),
        "amplitude_ratio": float(amplitude),
        "waveform_distortion": float(distortion),
        "leakage": float(numerical_leakage),
        "exact_leakage": float(exact_leakage),
        "leakage_excess": abs(float(numerical_leakage - exact_leakage)),
        "full_rel_l2": math.sqrt(float(numerator / max(denominator, 1.0e-300))),
    }


def run_method(
    method: str, npts: int, sigma: float, k0: float, encounters: int
) -> dict[str, float]:
    """March one declared method to an integer full-cycle return time."""
    exact_pressure, exact_velocity = exact_pair(sigma, k0)
    time_value = float(encounters)
    if method in ("BMF6", "BMG6", "SBP6"):
        family = method.removesuffix("6")
        operator = ba.build_operator(ba.scheme_kind(method), ORDER, npts)
        result = ba.simulate_reflected_acoustics(
            operator, exact_pressure, exact_velocity, time_value, cfl=CFL, record_count=2
        )
        x = result.grid if result.grid is not None else operator.x
    else:
        operator = dense_operator(method, npts)
        result = ba.simulate_reflected_acoustics(
            operator, exact_pressure, exact_velocity, time_value, cfl=CFL, record_count=2
        )
        x = result.grid if result.grid is not None else operator.x
    return score_fields(
        np.asarray(x, dtype=float),
        np.asarray(result.pressure, dtype=float),
        np.asarray(result.velocity, dtype=float),
        exact_pressure,
        exact_velocity,
        time_value,
        k0,
        sigma,
    )


def calibration_rows() -> list[dict[str, Any]]:
    """Load the reported packet and add the missing stable compact lane."""
    source = json.loads((RESULTSDIR / "packet_reflection_summary.json").read_text())
    rows: list[dict[str, Any]] = []
    source_methods = {"BMF6": "BMF6", "BMG6": "BMG6", "SBP6": "SBP6", "SEM6": "SEM6"}
    for method, prefix in source_methods.items():
        for npts in CALIBRATION_GRIDS:
            key = f"{prefix}_N{npts}"
            for cycles in CALIBRATION_CYCLES:
                metrics = source["runs"][key][f"cycle{cycles}"]
                encounters = 2 * cycles
                features = spectrum_features(
                    method, npts, reported.SIGMA, reported.K0, encounters
                )
                exact_pressure, _exact_velocity = exact_pair(reported.SIGMA, reported.K0)
                x = np.linspace(0.0, 1.0, 2048, endpoint=False)
                original_k0, original_sigma = reported.K0, reported.SIGMA
                try:
                    exact_leakage = reported.leakage(x, exact_pressure(x, float(encounters)))
                finally:
                    reported.K0, reported.SIGMA = original_k0, original_sigma
                rows.append({
                    "method": method,
                    "npts": npts,
                    "encounters": encounters,
                    "features": features,
                    "metrics": {
                        "phase_shift": abs(float(metrics["phase_shift"])),
                        "waveform_distortion": float(metrics["waveform_distortion"]),
                        "leakage": float(metrics["leakage"]),
                        "exact_leakage": float(exact_leakage),
                        "leakage_excess": abs(float(metrics["leakage"] - exact_leakage)),
                    },
                    "provenance": "packet_reflection_summary.json",
                })
    for npts in CALIBRATION_GRIDS:
        for encounters in (2, 6):
            metrics = run_method("CSB-T6", npts, reported.SIGMA, reported.K0, encounters)
            rows.append({
                "method": "CSB-T6",
                "npts": npts,
                "encounters": encounters,
                "features": spectrum_features(
                    "CSB-T6", npts, reported.SIGMA, reported.K0, encounters
                ),
                "metrics": metrics,
                "provenance": "fresh reported-packet calibration march",
            })
    return rows


def fit_power_model(
    rows: list[dict[str, Any]], metric: str, driver: str
) -> dict[str, Any]:
    """Fit a global power law with method offsets to the reported packet."""
    methods = list(METHODS)
    design = []
    target = []
    for row in rows:
        xvalue = max(float(row["features"][driver]), 1.0e-30)
        yvalue = max(float(row["metrics"][metric]), 1.0e-30)
        vector = [1.0, math.log(xvalue)]
        vector.extend(1.0 if row["method"] == method else 0.0 for method in methods[1:])
        design.append(vector)
        target.append(math.log(yvalue))
    matrix = np.asarray(design, dtype=float)
    values = np.asarray(target, dtype=float)
    ridge = np.diag([0.0, 1.0e-6] + [1.0e-3] * (len(methods) - 1))
    coefficients = np.linalg.solve(matrix.T @ matrix + ridge, matrix.T @ values)
    fitted = np.exp(matrix @ coefficients)
    observed = np.exp(values)
    floor_key = "leakage" if metric == "leakage_excess" else metric
    residual = np.abs(fitted - observed) / np.maximum(observed, ABSOLUTE_FLOORS[floor_key])
    return {
        "metric": metric,
        "driver": driver,
        "intercept": float(coefficients[0]),
        "exponent": float(coefficients[1]),
        "method_log_offsets": {
            methods[0]: 0.0,
            **{method: float(value) for method, value in zip(methods[1:], coefficients[2:], strict=True)},
        },
        "calibration_median_relative_residual": float(np.median(residual)),
        "calibration_worst_relative_residual": float(np.max(residual)),
    }


def predict_metric(model: dict[str, Any], method: str, driver_value: float) -> float:
    """Evaluate one frozen method-offset power model."""
    log_value = model["intercept"]
    log_value += model["exponent"] * math.log(max(driver_value, 1.0e-30))
    log_value += model["method_log_offsets"][method]
    return float(math.exp(log_value))


def case_id(sigma: float, k0_over_pi: float, encounters: int, method: str) -> str:
    """Return a stable identifier for one primary method-packet case."""
    return f"sig{sigma:.3f}_k{k0_over_pi:g}_b{encounters}_{method}"


def freeze_predictions() -> dict[str, Any]:
    """Fit only the reported packet and serialize all primary predictions."""
    ensure_generated_dirs()
    if PREDICTION_PATH.exists():
        raise FileExistsError(PREDICTION_PATH)
    rows = calibration_rows()
    models = {
        "phase_shift": fit_power_model(rows, "phase_shift", "phase_driver"),
        "waveform_distortion": fit_power_model(
            rows, "waveform_distortion", "distortion_driver"
        ),
        "leakage_excess": fit_power_model(rows, "leakage_excess", "leakage_driver"),
    }
    predictions = []
    for sigma in HOLDOUT_SIGMAS:
        for k0_over_pi in HOLDOUT_K0_OVER_PI:
            k0 = k0_over_pi * np.pi
            exact_pressure, _exact_velocity = exact_pair(sigma, k0)
            uniform = np.linspace(0.0, 1.0, 2048, endpoint=False)
            original_k0, original_sigma = reported.K0, reported.SIGMA
            reported.K0, reported.SIGMA = k0, sigma
            try:
                exact_leakage = reported.leakage(uniform, exact_pressure(uniform, 0.0))
            finally:
                reported.K0, reported.SIGMA = original_k0, original_sigma
            for encounters in HOLDOUT_ENCOUNTERS:
                for method in METHODS:
                    features = spectrum_features(method, NPTS, sigma, k0, encounters)
                    phase = predict_metric(models["phase_shift"], method, features["phase_driver"])
                    distortion = predict_metric(
                        models["waveform_distortion"], method, features["distortion_driver"]
                    )
                    leakage_excess = predict_metric(
                        models["leakage_excess"], method, features["leakage_driver"]
                    )
                    predictions.append({
                        "case_id": case_id(sigma, k0_over_pi, encounters, method),
                        "method": method,
                        "sigma": sigma,
                        "k0_over_pi": k0_over_pi,
                        "encounters": encounters,
                        "npts": NPTS,
                        "features": features,
                        "predicted": {
                            "phase_shift": phase,
                            "waveform_distortion": distortion,
                            "leakage_excess": leakage_excess,
                            "leakage": exact_leakage + leakage_excess,
                        },
                        "predicted_threshold_side": {
                            "response": features["response_bias_rms"] <= PRIMARY_LIMITS["response_bias"],
                            "contamination": features["contamination_rms"] <= PRIMARY_LIMITS["orthogonal_contamination"],
                            "phase": k0 * phase / max(encounters, 1) <= PRIMARY_LIMITS["phase_normalized"],
                            "distortion": distortion <= PRIMARY_LIMITS["waveform_distortion"],
                            "leakage": leakage_excess <= PRIMARY_LIMITS["leakage_excess"],
                        },
                    })
    record = {
        "schema": SCHEMA,
        "artifact_state": "immutable_pre_result_predictions",
        "objective": "quantitative transfer of the marched wall spectrum to held-out packets",
        "calibration_information_boundary": {
            "packet": {
                "sigma": reported.SIGMA,
                "k0_over_pi": reported.K0 / np.pi,
                "grids": list(CALIBRATION_GRIDS),
                "encounters": [2, 6],
            },
            "forbidden": "held-out numerical fields and metrics",
        },
        "primary_matrix": {
            "methods": list(METHODS),
            "order": ORDER,
            "npts": NPTS,
            "sigmas": list(HOLDOUT_SIGMAS),
            "k0_over_pi": list(HOLDOUT_K0_OVER_PI),
            "encounters": list(HOLDOUT_ENCOUNTERS),
            "case_count": len(predictions),
            "justification": "low, intermediate, and near-Nyquist carriers cross declared response and contamination sides; two widths separate narrow and broad modal support; four and eight encounters test finite-time accumulation beyond calibration horizons",
        },
        "decision_contract": {
            "relative_tolerance": RELATIVE_TOLERANCE,
            "absolute_floors": ABSOLUTE_FLOORS,
            "thresholds": PRIMARY_LIMITS,
            "promotion": "every primary case must pass quantitative metrics, threshold sides, and within-packet family ordering",
            "fallback": "retain the wall spectrum as a mechanism-aligned diagnostic and ranker",
        },
        "model": {
            "derivation": {
                "phase": "mean modal response bias accumulated over full cycles and converted to spatial shift",
                "distortion": "quadrature sum of modal phase-spread and encounter-scaled orthogonal contamination",
                "leakage": "exact packet leakage plus a calibrated square of encounter-scaled contamination",
                "finite_time_bound": "distortion is bounded at leading order by phase-spread plus encounter-scaled contamination; the quadrature form is the calibrated nonnegative approximation",
            },
            "fits": models,
        },
        "noise_selftest": {
            "kind": "deterministic exact-image and repeated metric evaluation",
            "exact_periodicity_max_abs": float(
                np.max(np.abs(
                    exact_pair(reported.SIGMA, reported.K0)[0](np.linspace(0.0, 1.0, 4001), 2.0)
                    - exact_pair(reported.SIGMA, reported.K0)[0](np.linspace(0.0, 1.0, 4001), 0.0)
                ))
            ),
            "floors": ABSOLUTE_FLOORS,
        },
        "calibration_rows": rows,
        "predictions": predictions,
        "source_sha256": {
            "driver": sha256(Path(__file__)),
            "reported_packet": sha256(_CODE / "s6_packet_reflection.py"),
            "reported_packet_record": sha256(RESULTSDIR / "packet_reflection_summary.json"),
            "wall_spectrum": sha256(_CODE / "s4_staggered_wallmode.py"),
            "baseline_closures": sha256(_CODE / "baseline_closures.py"),
        },
    }
    PREDICTION_PATH.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def run_holdout(prediction_path: Path, output_path: Path) -> dict[str, Any]:
    """Run each predeclared case exactly once from the frozen prediction file."""
    if output_path.exists():
        raise FileExistsError(output_path)
    prediction = json.loads(prediction_path.read_text())
    if prediction.get("artifact_state") != "immutable_pre_result_predictions":
        raise ValueError("prediction artifact is not frozen")
    results = []
    for index, item in enumerate(prediction["predictions"], start=1):
        metrics = run_method(
            item["method"],
            int(item["npts"]),
            float(item["sigma"]),
            float(item["k0_over_pi"]) * np.pi,
            int(item["encounters"]),
        )
        results.append({"case_id": item["case_id"], "metrics": metrics})
        print(f"{index}/{len(prediction['predictions'])} {item['case_id']}", flush=True)
    raw = {
        "schema": SCHEMA,
        "artifact_state": "heldout_raw_once",
        "prediction_sha256": sha256(prediction_path),
        "prediction_path": str(prediction_path),
        "results": results,
    }
    output_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    return raw


def ordering_signature(rows: list[dict[str, Any]], field: str, source: str) -> list[str]:
    """Return the strict family order for one packet and metric."""
    return [
        row["method"]
        for row in sorted(rows, key=lambda item: float(item[source][field]))
    ]


def summarize(prediction_path: Path, raw_path: Path, output_path: Path) -> dict[str, Any]:
    """Apply the frozen quantitative, threshold-side, and ordering contract."""
    prediction = json.loads(prediction_path.read_text())
    raw = json.loads(raw_path.read_text())
    if raw["prediction_sha256"] != sha256(prediction_path):
        raise ValueError("raw results do not bind to the prediction artifact")
    actual_by_id = {row["case_id"]: row["metrics"] for row in raw["results"]}
    cases = []
    for item in prediction["predictions"]:
        actual = actual_by_id[item["case_id"]]
        residuals = {}
        passes = {}
        for metric in ("phase_shift", "waveform_distortion", "leakage"):
            predicted = float(item["predicted"][metric])
            observed = float(actual[metric])
            denominator = max(abs(observed), prediction["decision_contract"]["absolute_floors"][metric])
            residuals[metric] = abs(predicted - observed) / denominator
            passes[metric] = (
                abs(predicted - observed)
                <= max(
                    RELATIVE_TOLERANCE * abs(observed),
                    prediction["decision_contract"]["absolute_floors"][metric],
                )
            )
        k0 = float(item["k0_over_pi"]) * np.pi
        actual_sides = {
            "phase": k0 * actual["phase_shift"] / max(item["encounters"], 1) <= PRIMARY_LIMITS["phase_normalized"],
            "distortion": actual["waveform_distortion"] <= PRIMARY_LIMITS["waveform_distortion"],
            "leakage": actual["leakage_excess"] <= PRIMARY_LIMITS["leakage_excess"],
        }
        side_pass = all(
            actual_sides[name] == item["predicted_threshold_side"][name]
            for name in actual_sides
        )
        cases.append({
            "case_id": item["case_id"],
            "method": item["method"],
            "sigma": item["sigma"],
            "k0_over_pi": item["k0_over_pi"],
            "encounters": item["encounters"],
            "predicted": item["predicted"],
            "actual": actual,
            "relative_residual": residuals,
            "quantitative_pass": passes,
            "threshold_side_pass": side_pass,
            "actual_threshold_side": actual_sides,
        })
    order_checks = []
    packet_keys = sorted({(row["sigma"], row["k0_over_pi"], row["encounters"]) for row in cases})
    for key in packet_keys:
        group = [row for row in cases if (row["sigma"], row["k0_over_pi"], row["encounters"]) == key]
        for metric in ("phase_shift", "waveform_distortion", "leakage"):
            predicted_order = ordering_signature(group, metric, "predicted")
            actual_order = ordering_signature(group, metric, "actual")
            order_checks.append({
                "sigma": key[0],
                "k0_over_pi": key[1],
                "encounters": key[2],
                "metric": metric,
                "predicted": predicted_order,
                "actual": actual_order,
                "pass": predicted_order == actual_order,
            })
    metric_residuals = {
        metric: [row["relative_residual"][metric] for row in cases]
        for metric in ("phase_shift", "waveform_distortion", "leakage")
    }
    quantitative_pass = all(
        all(row["quantitative_pass"].values()) for row in cases
    )
    threshold_pass = all(row["threshold_side_pass"] for row in cases)
    ordering_pass = all(row["pass"] for row in order_checks)
    promotion = quantitative_pass and threshold_pass and ordering_pass
    summary = {
        "schema": SCHEMA,
        "artifact_state": "heldout_summary",
        "prediction_sha256": sha256(prediction_path),
        "raw_sha256": sha256(raw_path),
        "case_count": len(cases),
        "decision": {
            "quantitative_pass": quantitative_pass,
            "threshold_side_pass": threshold_pass,
            "family_ordering_pass": ordering_pass,
            "promotion_earned": promotion,
            "claim_status": "accepted_quantitative_predictor" if promotion else "diagnostic_ranker_only",
        },
        "residual_summary": {
            metric: {
                "median": float(np.median(values)),
                "worst": float(np.max(values)),
                "pass_count": int(sum(value <= RELATIVE_TOLERANCE for value in values)),
                "total": len(values),
            }
            for metric, values in metric_residuals.items()
        },
        "failure_mechanism": None if promotion else (
            "A scalar modal RMS compression loses signed cross-mode coupling and method-specific leakage transfer at broad or near-Nyquist carrier support; finite-time contamination is not determined by rho and eta magnitudes alone."
        ),
        "classification": None if promotion else "theory_incomplete",
        "cases": cases,
        "ordering_checks": order_checks,
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def make_figure(prediction_path: Path, summary_path: Path) -> None:
    """Render predicted versus measured held-out metrics and the verdict."""
    prediction = json.loads(prediction_path.read_text())
    summary = json.loads(summary_path.read_text())
    colors = {
        "BMF6": "#22517a",
        "BMG6": "#1f6f6f",
        "SBP6": "#7a6a96",
        "CSB-T6": "#8c6d31",
        "SEM6": "#727272",
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.7))
    metrics = ("phase_shift", "waveform_distortion", "leakage")
    labels = ("absolute phase shift", "waveform distortion", "spectral leakage")
    for axis, metric, label in zip(axes, metrics, labels, strict=True):
        for method in METHODS:
            rows = [row for row in summary["cases"] if row["method"] == method]
            axis.scatter(
                [row["predicted"][metric] for row in rows],
                [row["actual"][metric] for row in rows],
                s=22,
                color=colors[method],
                alpha=0.8,
                label=method,
            )
        values = [
            max(row["predicted"][metric], row["actual"][metric])
            for row in summary["cases"]
            if row["predicted"][metric] > 0.0 and row["actual"][metric] > 0.0
        ]
        lower = min(values) / 2.0
        upper = max(values) * 2.0
        axis.plot([lower, upper], [lower, upper], color="0.35", lw=0.9)
        axis.plot([lower, upper], [0.8 * lower, 0.8 * upper], color="0.65", lw=0.7, ls=":")
        axis.plot([lower, upper], [1.2 * lower, 1.2 * upper], color="0.65", lw=0.7, ls=":")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_xlabel("frozen prediction")
        axis.set_ylabel("held-out measurement")
        axis.set_title(label, fontstyle="italic")
        axis.set_facecolor("#fbf7ee")
        axis.grid(alpha=0.16, linewidth=0.5)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, ncol=5, loc="upper center", frameon=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.80, wspace=0.31)
    ensure_generated_dirs()
    fig.savefig(FIGDIR / "fig_s6_wall_spectrum_heldout.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "fig_s6_wall_spectrum_heldout.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def verify(prediction_path: Path, raw_path: Path, summary_path: Path) -> None:
    """Verify provenance, matrix completeness, frozen rules, and decision logic."""
    prediction = json.loads(prediction_path.read_text())
    raw = json.loads(raw_path.read_text())
    summary = json.loads(summary_path.read_text())
    expected = {
        case_id(sigma, carrier, encounters, method)
        for sigma in HOLDOUT_SIGMAS
        for carrier in HOLDOUT_K0_OVER_PI
        for encounters in HOLDOUT_ENCOUNTERS
        for method in METHODS
    }
    predicted_ids = {row["case_id"] for row in prediction["predictions"]}
    raw_ids = {row["case_id"] for row in raw["results"]}
    summary_ids = {row["case_id"] for row in summary["cases"]}
    assert predicted_ids == raw_ids == summary_ids == expected
    assert len(predicted_ids) == 60
    assert raw["prediction_sha256"] == sha256(prediction_path)
    assert summary["prediction_sha256"] == sha256(prediction_path)
    assert summary["raw_sha256"] == sha256(raw_path)
    for row in raw["results"]:
        assert all(np.isfinite(value) for value in row["metrics"].values())
    decision = summary["decision"]
    recomputed = (
        decision["quantitative_pass"]
        and decision["threshold_side_pass"]
        and decision["family_ordering_pass"]
    )
    assert decision["promotion_earned"] == recomputed
    if not recomputed:
        assert decision["claim_status"] == "diagnostic_ranker_only"
        assert summary["failure_mechanism"]
    print("wall_spectrum_heldout verify: ALL PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    run = commands.add_parser("run")
    run.add_argument("--prediction", type=Path, default=PREDICTION_PATH)
    run.add_argument("--output", type=Path, default=RAW_PATH)
    summarize_parser = commands.add_parser("summarize")
    summarize_parser.add_argument("--prediction", type=Path, default=PREDICTION_PATH)
    summarize_parser.add_argument("--raw", type=Path, default=RAW_PATH)
    summarize_parser.add_argument("--output", type=Path, default=SUMMARY_PATH)
    figure = commands.add_parser("figure")
    figure.add_argument("--prediction", type=Path, default=PREDICTION_PATH)
    figure.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    verifier = commands.add_parser("verify")
    verifier.add_argument("--prediction", type=Path, default=PREDICTION_PATH)
    verifier.add_argument("--raw", type=Path, default=RAW_PATH)
    verifier.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "freeze":
        record = freeze_predictions()
        print(f"wrote {PREDICTION_PATH} with {len(record['predictions'])} predictions")
    elif args.command == "run":
        run_holdout(args.prediction, args.output)
    elif args.command == "summarize":
        summary = summarize(args.prediction, args.raw, args.output)
        print(json.dumps(summary["decision"], indent=2))
    elif args.command == "figure":
        make_figure(args.prediction, args.summary)
    elif args.command == "verify":
        verify(args.prediction, args.raw, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
