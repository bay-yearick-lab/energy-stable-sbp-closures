"""Independent artifact verifier for the off-grid hp-SEM geometry frontier."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))

from project_paths import RESULTSDIR
from s6_curved_offgrid import NU, random_field, sem_entrant, surfaces, traction

ROOT = _CODE.parent
SPEC_PATH = ROOT / "code/data/offgrid_sem_geometry_frontier_matrix.json"
RAW_PATH = RESULTSDIR / "offgrid_sem_geometry_frontier_raw.json"
SUMMARY_PATH = RESULTSDIR / "offgrid_sem_geometry_frontier_summary.json"
STATISTICS = ("median", "p95", "worst")
METRICS = ("velocity", "normal_derivative", "traction")
SURFACES = ("circle_small", "circle_large", "star")
BUDGETS = (19, 37, 61)


def _statistics(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, float)
    return {
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "worst": float(np.max(array)),
        "count": int(array.size),
    }


def _expected_sem_keys() -> set[tuple[int, int, int]]:
    return {
        (budget, degree, elements)
        for budget in BUDGETS
        for degree in range(2, 13)
        for elements in range(1, (budget - 1) // degree + 1)
    }


def _score_independent(field, surface, reconstructed) -> dict[str, np.ndarray]:
    x, y = surface["x"], surface["y"]
    nx, ny = surface["nx"], surface["ny"]
    exact_u = field["u"](x, y)
    exact_v = field["v"](x, y)
    velocity = np.hypot(reconstructed["u"] - exact_u, reconstructed["v"] - exact_v)
    velocity /= max(float(np.max(np.hypot(exact_u, exact_v))), 1.0e-30)

    exact_un = field["ux"](x, y) * nx + field["uy"](x, y) * ny
    exact_vn = field["vx"](x, y) * nx + field["vy"](x, y) * ny
    approx_un = reconstructed["ux"] * nx + reconstructed["uy"] * ny
    approx_vn = reconstructed["vx"] * nx + reconstructed["vy"] * ny
    normal = np.hypot(approx_un - exact_un, approx_vn - exact_vn)
    normal /= max(float(np.max(np.hypot(exact_un, exact_vn))), 1.0e-30)

    exact_tx, exact_ty = traction(field, x, y, nx, ny)
    sxx = -reconstructed["p"] + 2.0 * NU * reconstructed["ux"]
    syy = -reconstructed["p"] + 2.0 * NU * reconstructed["vy"]
    sxy = NU * (reconstructed["uy"] + reconstructed["vx"])
    approx_tx = sxx * nx + sxy * ny
    approx_ty = sxy * nx + syy * ny
    traction_error = np.hypot(approx_tx - exact_tx, approx_ty - exact_ty)
    traction_error /= max(float(np.max(np.hypot(exact_tx, exact_ty))), 1.0e-30)
    return {"velocity": velocity, "normal_derivative": normal, "traction": traction_error}


def verify() -> None:
    specification_bytes = SPEC_PATH.read_bytes()
    specification = json.loads(specification_bytes)
    raw = json.loads(RAW_PATH.read_text())
    summary = json.loads(SUMMARY_PATH.read_text())
    expected_hash = hashlib.sha256(specification_bytes).hexdigest()
    assert raw["protocol_id"] == specification["protocol_id"]
    assert summary["protocol_id"] == specification["protocol_id"]
    assert raw["spec_sha256"] == expected_hash
    assert summary["spec_sha256"] == expected_hash

    analytic = raw["analytic_gates"]
    gates = specification["gates"]
    assert analytic["normal_unit_max_abs"] < gates["normal_unit_absolute_tolerance"]
    assert analytic["normal_outward_dot_min"] > gates["normal_outward_dot_minimum"]
    assert analytic["analytic_derivative_fd_max_abs"] < gates["analytic_stress_fd_absolute_tolerance"]
    assert analytic["analytic_divergence_max_abs"] < gates["analytic_divergence_absolute_tolerance"]

    expected_sem = _expected_sem_keys()
    polynomial_sem = {
        (row["budget"], row["degree"], row["elements"])
        for row in raw["polynomial_gates"] if row["entrant"] == "sem"
    }
    assert polynomial_sem == expected_sem
    polynomial_spline = {
        row["budget"] for row in raw["polynomial_gates"]
        if row["entrant"] == "spline_native_BMF6_BMG6"
    }
    assert polynomial_spline == set(BUDGETS)
    for row in raw["polynomial_gates"]:
        assert row["value_max_abs"] < gates["polynomial_reproduction_absolute_tolerance"]
        assert row["derivative_max_abs"] < gates["polynomial_reproduction_absolute_tolerance"]

    expected_case_keys = {
        ("sem", budget, degree, elements, surface)
        for budget, degree, elements in expected_sem
        for surface in SURFACES
    } | {
        ("spline_native_BMF6_BMG6", budget, 6, None, surface)
        for budget in BUDGETS for surface in SURFACES
    }
    cases = {}
    for case in raw["cases"]:
        key = (
            case["entrant"], case["budget"], case["degree"],
            case["elements"], case["surface"],
        )
        assert key not in cases
        assert set(case["errors"]) == set(METRICS)
        for values in case["errors"].values():
            array = np.asarray(values, float)
            assert array.size == 12 * 48
            assert np.all(np.isfinite(array))
            assert np.all(array >= 0.0)
        cost = case["cost"]
        assert cost["state_size_1d"] == case["state_size_1d"]
        assert cost["tensor_scalar_state"] == 3 * case["state_size_1d"] ** 2
        assert cost["sampled_state_bytes"] == 3 * case["state_size_1d"] ** 2 * 8
        assert cost["operator_setup_process_seconds_median"] > 0.0
        assert cost["evaluation_plan_setup_process_seconds_median"] > 0.0
        assert cost["field_setup_process_seconds_median"] > 0.0
        assert cost["total_setup_process_seconds_median"] == (
            cost["operator_setup_process_seconds_median"]
            + cost["evaluation_plan_setup_process_seconds_median"]
            + cost["field_setup_process_seconds_median"]
        )
        assert cost["recurring_evaluation_process_seconds_median"] > 0.0
        cases[key] = case
    assert set(cases) == expected_case_keys

    rows = {}
    for row in summary["rows"]:
        key = (
            row["entrant"], row["budget"], row["degree"], row["elements"],
            row["surface"], row["metric"],
        )
        assert key not in rows
        case_key = key[:-1]
        expected = _statistics(cases[case_key]["errors"][row["metric"]])
        for statistic in (*STATISTICS, "count"):
            assert row[statistic] == expected[statistic]
        assert row["median"] <= row["p95"] <= row["worst"]
        rows[key] = row
    assert len(rows) == len(expected_case_keys) * len(METRICS)

    frontier_rows = {
        (row["budget"], row["surface"], row["metric"]): row
        for row in summary["frontiers"]
    }
    assert len(frontier_rows) == len(BUDGETS) * len(SURFACES) * len(METRICS)
    dominance_count = 0
    factor_two_count = 0
    for budget in BUDGETS:
        for surface in SURFACES:
            for metric in METRICS:
                frontier = frontier_rows[(budget, surface, metric)]
                spline = rows[("spline_native_BMF6_BMG6", budget, 6, None, surface, metric)]
                sem_rows = [
                    row for key, row in rows.items()
                    if key[0] == "sem" and key[1] == budget
                    and key[4] == surface and key[5] == metric
                ]
                factors = []
                for statistic in STATISTICS:
                    best = min(sem_rows, key=lambda row: row[statistic])
                    recorded = frontier["sem_envelope"][statistic]
                    assert recorded["error"] == best[statistic]
                    assert recorded["degree"] == best["degree"]
                    assert recorded["elements"] == best["elements"]
                    assert recorded["state_size_1d"] == best["state_size_1d"]
                    assert frontier["spline"][statistic] == spline[statistic]
                    factor = best[statistic] / spline[statistic]
                    assert frontier["sem_to_spline_error_factor"][statistic] == factor
                    factors.append(factor)
                if metric == "traction":
                    dominance_count += int(all(factor > 1.0 for factor in factors))
                    factor_two_count += int(all(factor >= 2.0 for factor in factors))

    promotion = summary["promotion"]
    assert promotion["traction_cell_count"] == 9
    assert promotion["traction_dominance_pass"] == (dominance_count == 9)
    assert promotion["factor_two_pass"] == (factor_two_count > 0)
    assert promotion["full_promotion_pass"] == (
        promotion["traction_dominance_pass"]
        and (promotion["factor_two_pass"] or promotion["state_reduction_pass"])
    )

    rng = np.random.default_rng(20260712)
    random_field(rng)
    field = random_field(rng)
    surface = surfaces()[0]
    archived = sem_entrant(19, field, surface)
    independent = _score_independent(field, surface, archived)
    recorded = cases[("sem", 19, 6, 3, "circle_small")]
    for metric in METRICS:
        assert np.allclose(
            np.asarray(recorded["errors"][metric][:48]),
            independent[metric], rtol=5.0e-11, atol=5.0e-13,
        )

    print(
        "offgrid SEM geometry frontier verify: ALL PASS; "
        f"cases={len(cases)} rows={len(rows)} traction_dominance={dominance_count}/9"
    )


if __name__ == "__main__":
    verify()
