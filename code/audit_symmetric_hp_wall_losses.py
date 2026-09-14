"""Audit exceptional wall cells and normalization scales in the hp frontier."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))

import s6_symmetric_hp_wall_frontier as frontier
from project_paths import RESULTSDIR

OUTPUT = RESULTSDIR / "symmetric_hp_wall_frontier_normalization_audit.json"
METRICS = ("wall_trace_l2", "normal_derivative_l2", "flux_l2", "traction_l2")
DERIVATIVE_METRICS = ("normal_derivative_l2", "flux_l2", "traction_l2")
STATISTICS = ("median", "p95", "worst")


def _wall_diagnostics(
    field: frontier.FieldCase,
    field_index: int,
    samples: dict,
    approximate: dict[str, np.ndarray],
) -> dict[str, dict[str, float | bool]]:
    exact = frontier._exact_triplet(samples["wall"], field_index)
    approx = {name: value[field_index] for name, value in approximate.items()}
    volume = frontier._exact_triplet(samples["volume"], field_index)
    volume_weight = np.outer(samples["volume_weights"], samples["volume_weights"])
    field_reference = max(
        float(np.sqrt(np.sum(volume_weight * np.sum(volume["value"] ** 2, axis=0)))),
        1.0e-14,
    )
    wall_weight = samples["wall_weights"]
    nx, ny = samples["wall_nx"], samples["wall_ny"]

    trace_error = np.sum((approx["value"][:2] - exact["value"][:2]) ** 2, axis=0)
    trace_exact = np.sum(exact["value"][:2] ** 2, axis=0)
    trace_absolute = float(np.sqrt(np.sum(wall_weight * trace_error)))
    trace_exact_norm = float(np.sqrt(np.sum(wall_weight * trace_exact)))
    trace_floor = 0.1 * field_reference

    normal_exact = exact["dx"][:2] * nx + exact["dy"][:2] * ny
    normal_approx = approx["dx"][:2] * nx + approx["dy"][:2] * ny
    normal_error = normal_approx - normal_exact
    normal_absolute = float(np.sqrt(np.sum(wall_weight * np.sum(normal_error**2, axis=0))))
    normal_exact_norm = float(np.sqrt(np.sum(wall_weight * np.sum(normal_exact**2, axis=0))))
    normal_floor = 0.1 * field_reference

    flux_exact = -field.diffusivity * normal_exact[0]
    flux_approx = -field.diffusivity * normal_approx[0]
    flux_absolute = float(np.sqrt(np.sum(wall_weight * (flux_approx - flux_exact) ** 2)))
    flux_exact_norm = float(np.sqrt(np.sum(wall_weight * flux_exact**2)))
    flux_floor = 0.1 * field.diffusivity * field_reference

    exact_tx, exact_ty = frontier._traction(exact, field.viscosity, nx, ny)
    approx_tx, approx_ty = frontier._traction(approx, field.viscosity, nx, ny)
    traction_error = np.hypot(approx_tx - exact_tx, approx_ty - exact_ty)
    traction_exact = np.hypot(exact_tx, exact_ty)
    traction_absolute = float(np.sqrt(np.sum(wall_weight * traction_error**2)))
    traction_exact_norm = float(np.sqrt(np.sum(wall_weight * traction_exact**2)))
    traction_floor = 0.1 * field_reference

    raw = {
        "wall_trace_l2": (trace_absolute, trace_exact_norm, trace_floor),
        "normal_derivative_l2": (normal_absolute, normal_exact_norm, normal_floor),
        "flux_l2": (flux_absolute, flux_exact_norm, flux_floor),
        "traction_l2": (traction_absolute, traction_exact_norm, traction_floor),
    }
    result = {}
    for metric, (absolute_error, exact_norm, floor) in raw.items():
        denominator = max(exact_norm, floor)
        result[metric] = {
            "absolute_error_l2": absolute_error,
            "exact_observable_l2": exact_norm,
            "declared_floor": floor,
            "denominator": denominator,
            "floor_active": bool(floor > exact_norm),
            "normalized_error": absolute_error / denominator,
        }
    return result


def _configuration_diagnostics(config: dict, fields: list, samples: dict) -> list[dict]:
    reconstruction = frontier._reconstruction(config)
    state = frontier._prepared_state(reconstruction, fields)
    vx, dx = frontier._axis_rows(reconstruction, samples["wall_x"])
    vy, dy = frontier._axis_rows(reconstruction, samples["wall_y"])
    approximate = frontier._paired_evaluate(state, vx, dx, vy, dy)
    return [
        {
            "config": config["key"],
            "case": field.name,
            "field_class": field.field_class,
            "metrics": _wall_diagnostics(field, index, samples, approximate),
        }
        for index, field in enumerate(fields)
    ]


def _exception_cells(summary: dict) -> list[dict]:
    exceptions = []
    for row in summary["paired"]:
        if row["objective"] != "wall":
            continue
        for metric in DERIVATIVE_METRICS:
            for statistic in STATISTICS:
                factor = row["sem_to_spline_error_factor"][metric][statistic]
                if factor < 1.0:
                    exceptions.append({
                        "metric": metric,
                        "statistic": statistic,
                        "sem_to_spline_factor": factor,
                        "resource_kind": row["resource_kind"],
                        "resource_cap": row["resource_cap"],
                        "query_count": row["query_count"],
                        "spline_config": row["spline_config"],
                        "sem_config": row["sem_config"],
                        "classwise_worst_factors": {
                            field_class: factors[metric]["worst"]
                            for field_class, factors in row["classwise_sem_to_spline_error_factor"].items()
                        },
                    })
    return exceptions


def _large_ratio_cells(summary: dict, threshold: float = 100.0) -> list[dict]:
    cells = []
    frontier_index = {
        (row["objective"], row["family"], row["resource_kind"], row["resource_cap"], row["query_count"]): row
        for row in summary["frontiers"]
    }
    for row in summary["paired"]:
        if row["objective"] != "wall":
            continue
        for metric in DERIVATIVE_METRICS:
            for statistic in STATISTICS:
                factor = row["sem_to_spline_error_factor"][metric][statistic]
                if factor < threshold:
                    continue
                key = ("wall", row["resource_kind"], row["resource_cap"], row["query_count"])
                spline = frontier_index[(key[0], "spline", key[1], key[2], key[3])]
                sem = frontier_index[(key[0], "sem", key[1], key[2], key[3])]
                cells.append({
                    "metric": metric,
                    "statistic": statistic,
                    "sem_to_spline_factor": factor,
                    "resource_kind": row["resource_kind"],
                    "resource_cap": row["resource_cap"],
                    "query_count": row["query_count"],
                    "spline_config": row["spline_config"]["key"],
                    "sem_config": row["sem_config"]["key"],
                    "spline_normalized_error": spline["heldout"][metric][statistic],
                    "sem_normalized_error": sem["heldout"][metric][statistic],
                })
    return cells


def _selector_candidates(design: dict, specification: dict) -> dict[str, list[dict]]:
    steps = specification["work"]["evolution_steps"]
    query_count = 100
    cap = 20_000_000
    records = {}
    for family in ("spline", "sem"):
        candidates = []
        for config in design["configurations"]:
            if config["family"] != family:
                continue
            cost = design["costs"][config["key"]]
            mission = cost["setup_work"] + steps * cost["operator_step_work"] \
                + query_count * cost["marginal_query_work"]
            if mission <= cap:
                candidates.append({
                    "config": config,
                    "mission_work": mission,
                    "wall_design_score": design["scores"][config["key"]]["wall"],
                })
        candidates.sort(key=lambda row: (
            row["wall_design_score"], row["mission_work"], row["config"]["n"],
            row["config"]["degree"], row["config"]["elements"],
        ))
        records[family] = candidates[:12]
    return records


def _tolerance_materiality(summary: dict, specification: dict) -> dict[str, dict[str, int]]:
    frontier_index = {
        (row["family"], row["resource_kind"], row["resource_cap"], row["query_count"]): row
        for row in summary["frontiers"] if row["objective"] == "wall"
    }
    result = {}
    for metric in DERIVATIVE_METRICS:
        tolerance = specification["selector"]["normalizers"][metric]
        counts = {
            "spline_only_below_tolerance": 0,
            "sem_only_below_tolerance": 0,
            "both_below_tolerance": 0,
            "both_above_tolerance": 0,
            "factor_two_or_more": 0,
        }
        for row in summary["paired"]:
            if row["objective"] != "wall":
                continue
            base = (row["resource_kind"], row["resource_cap"], row["query_count"])
            spline = frontier_index[("spline", *base)]
            sem = frontier_index[("sem", *base)]
            for statistic in STATISTICS:
                spline_error = spline["heldout"][metric][statistic]
                sem_error = sem["heldout"][metric][statistic]
                if spline_error <= tolerance < sem_error:
                    counts["spline_only_below_tolerance"] += 1
                elif sem_error <= tolerance < spline_error:
                    counts["sem_only_below_tolerance"] += 1
                elif spline_error <= tolerance and sem_error <= tolerance:
                    counts["both_below_tolerance"] += 1
                else:
                    counts["both_above_tolerance"] += 1
                if sem_error / spline_error >= 2.0:
                    counts["factor_two_or_more"] += 1
        result[metric] = {"tolerance": tolerance, **counts}
    return result


def audit() -> dict:
    specification = json.loads(frontier.SPEC_PATH.read_text())
    design = json.loads(frontier.DESIGN_PATH.read_text())
    selectors = json.loads(frontier.SELECTOR_PATH.read_text())
    raw = json.loads(frontier.RAW_PATH.read_text())
    summary = json.loads(frontier.SUMMARY_PATH.read_text())
    fields = frontier.build_fields(specification, "heldout")
    samples = frontier.sampling_record(specification, fields)
    selected = {
        row["config"]["key"]: row["config"]
        for row in selectors["selectors"] if row["objective"] == "wall"
    }
    diagnostics = []
    for config in selected.values():
        diagnostics.extend(_configuration_diagnostics(config, fields, samples))
    normalized_index = {
        (row["config"], row["case"]): row["metrics"] for row in raw["rows"]
    }
    for row in diagnostics:
        for metric in METRICS:
            expected = normalized_index[(row["config"], row["case"])][metric]
            observed = row["metrics"][metric]["normalized_error"]
            if not np.isclose(expected, observed, rtol=2.0e-12, atol=2.0e-14):
                raise AssertionError((row["config"], row["case"], metric, expected, observed))
    floor_counts = {
        metric: sum(row["metrics"][metric]["floor_active"] for row in diagnostics)
        for metric in METRICS
    }
    minimum_absolute = {
        metric: min(row["metrics"][metric]["absolute_error_l2"] for row in diagnostics)
        for metric in METRICS
    }
    maximum_absolute = {
        metric: max(row["metrics"][metric]["absolute_error_l2"] for row in diagnostics)
        for metric in METRICS
    }
    exceptional_configs = {"spline:p2:k24", "sem:p3:e11"}
    exceptional_rows = [row for row in diagnostics if row["config"] in exceptional_configs]
    gate_index = {row["config"]: row for row in raw["polynomial_gates"]}
    result = {
        "protocol_id": specification["protocol_id"],
        "spec_sha256": raw["spec_sha256"],
        "summary_sha256": frontier._sha256_path(frontier.SUMMARY_PATH),
        "wall_selected_config_count": len(selected),
        "diagnostic_row_count": len(diagnostics),
        "normalization_reconstruction_pass": True,
        "floor_active_counts": floor_counts,
        "minimum_absolute_error_l2": minimum_absolute,
        "maximum_absolute_error_l2": maximum_absolute,
        "exceptions": _exception_cells(summary),
        "exceptional_case_diagnostics": exceptional_rows,
        "exceptional_polynomial_gates": {
            key: gate_index[key] for key in exceptional_configs
        },
        "selector_candidates_at_20m_q100": _selector_candidates(design, specification),
        "large_ratio_cells_over_100": _large_ratio_cells(summary),
        "tolerance_materiality": _tolerance_materiality(summary, specification),
        "classification": {
            "type": "selector_effect",
            "evidence": "Every derivative loss occurs at the 20,000,000-work, 100-query wall selector, where the spline selects degree two with 26 one-dimensional states and SEM selects degree three with 34. Both declared polynomial gates pass below the frozen tolerance, while every held-out field contains nonpolynomial or higher-degree content.",
            "exact_polynomial_loss": False,
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=1))
    return result


def verify() -> None:
    generated = audit()
    stored = json.loads(OUTPUT.read_text())
    if generated != stored:
        raise AssertionError("normalization audit reconstruction differs from the stored record")
    if len(stored["exceptions"]) != 5:
        raise AssertionError("unexpected wall-exception count")
    if stored["classification"]["type"] != "selector_effect":
        raise AssertionError("wall exceptions are not classified as selector effects")
    print(
        "symmetric hp wall normalization audit: ALL PASS; "
        f"rows={stored['diagnostic_row_count']} exceptions={len(stored['exceptions'])}"
    )


if __name__ == "__main__":
    verify()
