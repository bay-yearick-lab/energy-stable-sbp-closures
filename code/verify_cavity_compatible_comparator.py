#!/usr/bin/env python3
"""Verify raw cavity evidence and apply the frozen finite-ladder decision rule."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from cavity_compatible_comparator import (
    BASE_STEADY_TOLERANCE,
    DIVERGENCE_TOLERANCE,
    GATE_OUTPUT,
    PHYSICAL_TOLERANCE_PP,
    PRESSURE_GAUGE_TOLERANCE,
    RAW_OUTPUT,
    SENSITIVITY_STEADY_TOLERANCE,
    SENSITIVITY_TOLERANCE_PP,
    SUMMARY_OUTPUT,
    centerline_score,
    frozen_config,
)

SCHEMA = "cavity-compatible-comparator-matched-accuracy-v2"
FROZEN_PRIMARY_SOURCE_SHA256 = "b138cf1790fe1f7a52b0c60577991c6e035f6ab1432ee5dfc902aa4fb469add0"


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _analysis_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def verify_primary(raw: dict[str, Any], gate: dict[str, Any]) -> None:
    if raw.get("source_sha256") != FROZEN_PRIMARY_SOURCE_SHA256:
        raise ValueError("primary source provenance mismatch")
    if gate.get("source_sha256") != FROZEN_PRIMARY_SOURCE_SHA256:
        raise ValueError("gate and primary source provenance differ")
    if raw.get("config") != frozen_config():
        raise ValueError("frozen comparator configuration mismatch")
    if raw.get("config_sha256") != _hash(frozen_config()):
        raise ValueError("frozen comparator configuration hash mismatch")
    runs = raw.get("runs", [])
    if len(runs) != 12:
        raise ValueError("primary matrix must contain twelve base and sensitivity runs")
    for run in runs:
        for key in (
            "centerline_discrepancy_percentage_points", "divergence_l2",
            "pressure_gauge_residual", "total_cpu_seconds", "peak_rss_mib",
        ):
            if not math.isfinite(float(run.get(key, math.nan))):
                raise ValueError(f"non-finite primary field: {key}")
        if run.get("converged") is not True:
            raise AssertionError("steady convergence gate failed")
        if run.get("pressure_nullity") != 1:
            raise AssertionError("pressure nullity gate failed")
        if run["divergence_l2"] >= DIVERGENCE_TOLERANCE:
            raise AssertionError("field divergence gate failed")
        if run["pressure_gauge_residual"] >= PRESSURE_GAUGE_TOLERANCE:
            raise AssertionError("pressure gauge gate failed")
        score = centerline_score(
            np.asarray(run["u_centerline"]), np.asarray(run["v_centerline"])
        )
        if score != run["centerline"]:
            raise ValueError("shared reference score does not reproduce")


def base_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    lookup = {
        (run["method"], run["grid"], run["steady_tolerance"]): run
        for run in raw["runs"]
    }
    rows = []
    for method in ("BMF4", "MAC2"):
        grids = sorted({run["grid"] for run in raw["runs"] if run["method"] == method})
        for grid in grids:
            base = lookup[(method, grid, BASE_STEADY_TOLERANCE)]
            sensitivity = lookup[(method, grid, SENSITIVITY_STEADY_TOLERANCE)]
            change = abs(
                base["centerline_discrepancy_percentage_points"]
                - sensitivity["centerline_discrepancy_percentage_points"]
            )
            rows.append({
                "method": method,
                "grid": grid,
                "velocity_states": base["velocity_states"],
                "pressure_states": base["pressure_states"],
                "centerline_discrepancy_percentage_points": base[
                    "centerline_discrepancy_percentage_points"
                ],
                "sensitivity_change_percentage_points": change,
                "physical_accuracy_pass": base[
                    "centerline_discrepancy_percentage_points"
                ] <= PHYSICAL_TOLERANCE_PP,
                "sensitivity_pass": change < SENSITIVITY_TOLERANCE_PP,
                "invariants_pass": (
                    base["converged"]
                    and base["pressure_nullity"] == 1
                    and base["divergence_l2"] < DIVERGENCE_TOLERANCE
                    and base["pressure_gauge_residual"] < PRESSURE_GAUGE_TOLERANCE
                ),
                "divergence_l2": base["divergence_l2"],
                "pressure_nullity": base["pressure_nullity"],
                "pressure_gauge_residual": base["pressure_gauge_residual"],
                "relative_energy_ledger_residual": base[
                    "relative_energy_ledger_residual"
                ],
                "relative_momentum_ledger_residual": base[
                    "relative_momentum_ledger_residual"
                ],
                "wall_work": base["wall_work"],
                "wall_reaction": base["wall_reaction"],
                "setup_cpu_seconds": base["setup_cpu_seconds"],
                "march_cpu_seconds": base["march_cpu_seconds"],
                "total_cpu_seconds": base["total_cpu_seconds"],
                "peak_rss_mib": base["peak_rss_mib"],
                "steps": base["steps"],
                "final_time": base["final_time"],
            })
    return rows


def matched_accuracy_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enumerate finite-ladder pairs with BMF no less accurate than MAC."""
    bmf_rows = [row for row in rows if row["method"] == "BMF4"]
    mac_rows = [row for row in rows if row["method"] == "MAC2"]
    pairs = []
    for bmf in bmf_rows:
        for mac in mac_rows:
            eligible = all(
                row[gate]
                for row in (bmf, mac)
                for gate in ("physical_accuracy_pass", "sensitivity_pass", "invariants_pass")
            )
            no_less_accurate = (
                bmf["centerline_discrepancy_percentage_points"]
                <= mac["centerline_discrepancy_percentage_points"]
            )
            if eligible and no_less_accurate:
                pairs.append({
                    "bmf_grid": bmf["grid"],
                    "mac_grid": mac["grid"],
                    "bmf_velocity_states": bmf["velocity_states"],
                    "mac_velocity_states": mac["velocity_states"],
                    "bmf_discrepancy_percentage_points": bmf[
                        "centerline_discrepancy_percentage_points"
                    ],
                    "mac_discrepancy_percentage_points": mac[
                        "centerline_discrepancy_percentage_points"
                    ],
                    "bmf_velocity_state_reduction_fraction": 1.0
                    - bmf["velocity_states"] / mac["velocity_states"],
                })
    return sorted(
        pairs,
        key=lambda pair: (
            -pair["bmf_velocity_state_reduction_fraction"],
            pair["bmf_grid"],
            pair["mac_grid"],
        ),
    )


def equal_state_cpu_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for states in sorted({row["velocity_states"] for row in rows}):
        bmf = next(row for row in rows if row["method"] == "BMF4" and row["velocity_states"] == states)
        mac = next(row for row in rows if row["method"] == "MAC2" and row["velocity_states"] == states)
        pairs.append({
            "velocity_states": states,
            "bmf_grid": bmf["grid"],
            "mac_grid": mac["grid"],
            "bmf_cpu_seconds": bmf["total_cpu_seconds"],
            "mac_cpu_seconds": mac["total_cpu_seconds"],
            "bmf_to_mac_cpu_ratio": bmf["total_cpu_seconds"] / mac["total_cpu_seconds"],
        })
    return pairs


def build_summary(raw: dict[str, Any]) -> dict[str, Any]:
    rows = base_rows(raw)
    accuracy_pairs = matched_accuracy_pairs(rows)
    cpu_pairs = equal_state_cpu_pairs(rows)
    strongest = accuracy_pairs[0] if accuracy_pairs else None
    promoted = strongest is not None and strongest[
        "bmf_velocity_state_reduction_fraction"
    ] >= 0.25
    return {
        "schema": SCHEMA,
        "primary_source_sha256": raw["source_sha256"],
        "primary_config_sha256": raw["config_sha256"],
        "analysis_source_sha256": _analysis_hash(),
        "matched_accuracy_rule": (
            "BMF discrepancy is no greater than MAC discrepancy; both cells pass "
            "the frozen physical, sensitivity, divergence, nullity, gauge, and steady gates"
        ),
        "rows": rows,
        "eligible_matched_accuracy_pairs": accuracy_pairs,
        "strongest_matched_accuracy_pair": strongest,
        "equal_state_cpu_pairs": cpu_pairs,
        "claim_status": (
            "comparative_promotion_state_efficiency_re100"
            if promoted else "validation_only_fallback"
        ),
        "claim_scope": (
            "Re=100 Cartesian cavity state efficiency at matched reference centerline accuracy; "
            "no runtime superiority and no universal superiority"
        ),
    }


def verify_summary(summary: dict[str, Any], raw: dict[str, Any]) -> None:
    rebuilt = build_summary(raw)
    if summary != rebuilt:
        raise ValueError("matched-accuracy summary does not reproduce from raw evidence")
    pairs = summary["eligible_matched_accuracy_pairs"]
    if not pairs:
        raise AssertionError("no eligible matched-accuracy pair was enumerated")
    for pair in pairs:
        if pair["bmf_discrepancy_percentage_points"] > pair[
            "mac_discrepancy_percentage_points"
        ]:
            raise AssertionError("matched-accuracy direction is reversed")
    for pair in summary["equal_state_cpu_pairs"]:
        if pair["bmf_to_mac_cpu_ratio"] <= 1.0:
            raise AssertionError("equal-state CPU disadvantage was not preserved")


def main() -> int:
    gate = json.loads(GATE_OUTPUT.read_text())
    raw = json.loads(RAW_OUTPUT.read_text())
    verify_primary(raw, gate)
    summary = build_summary(raw)
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    verify_summary(summary, raw)
    print("cavity matched-accuracy verifier: ALL PASS")
    print(json.dumps(summary["strongest_matched_accuracy_pair"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
