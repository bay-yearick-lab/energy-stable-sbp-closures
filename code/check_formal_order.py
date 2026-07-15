#!/usr/bin/env python3
"""Generate condition-aware formal-order certificates on the actual grids."""
from __future__ import annotations

import csv
import io
import json
from typing import Iterable

import numpy as np

import boundary_spline_analysis as ba
from project_paths import RESULTSDIR, ensure_generated_dirs
import wall_order_certificates as wc


DEFAULT_NS_MAP = {
    4: (17, 25, 33, 49, 65),
    6: (17, 25, 33, 49, 65),
    8: (17, 25, 33, 49, 65),
}


def _spline_polynomial_row(
    family: str, degree: int, *, include_exact_endpoint: bool
) -> dict[str, object]:
    npts = max(33, degree + 17)
    floating = wc.floating_moment_certificate(family, degree, npts)
    through_p = [
        moment
        for row in floating["rows"]
        for moment in row["moments"]
        if moment["ell"] <= degree
    ]
    first_free = [
        row["moments"][degree + 1] for row in floating["rows"]
    ]
    row: dict[str, object] = {
        "degree": int(degree),
        "exact_polynomial_degree": int(degree),
        "first_nonzero_moment_degree": int(degree + 1),
        "output_grid": floating["output_grid"],
        "h_definition": floating["h_definition"],
        "precision": floating["precision"],
        "tolerance_rationale": floating["tolerance_rationale"],
        "all_rows_exact_through_p_within_backward_tolerance": bool(
            all(moment["within_backward_tolerance"] for moment in through_p)
        ),
        "max_raw_moment_through_p": float(
            max(abs(moment["raw_moment"]) for moment in through_p)
        ),
        "max_raw_first_free_moment": float(
            max(abs(moment["raw_moment"]) for moment in first_free)
        ),
        # Legacy names retained for downstream readers.  Their classification
        # now comes from scaled/condition-aware certificates.
        "error_on_x_p1": float(max(abs(moment["raw_moment"]) for moment in first_free)),
        "error_on_x_p2": None,
        "boundary_moment_on_x_p1_nonzero": None,
        "finite_grid_exact_endpoint_certificate": False,
    }
    if include_exact_endpoint:
        exact = wc.multiprecision_endpoint_certificate(
            family, degree, degree + 9, decimal_digits=90
        )
        row["boundary_moment_on_x_p1_nonzero"] = bool(all(
            endpoint["interval_excludes_zero"]
            for endpoint in exact["endpoints"].values()
        ))
        row["finite_grid_exact_endpoint_certificate"] = True
        row["exact_endpoint_certificate"] = exact
    return row


def _non_spline_convergence(label: str, kind: str, order: int, ns: tuple[int, ...]) -> dict[str, object]:
    errors: list[float] = []
    for npts in ns:
        op = ba.build_operator(kind, order, npts)
        x = op.x
        errors.append(float(np.linalg.norm(
            op.differentiate(wc.manufactured_u(x)) - wc.manufactured_ux(x),
            ord=np.inf,
        )))
    hs = [1.0 / (npts - 1) for npts in ns]
    observed = [
        float(np.log(errors[i - 1] / errors[i]) / np.log(hs[i - 1] / hs[i]))
        for i in range(1, len(ns))
    ]
    op = ba.build_operator(kind, order, ns[-1])
    return {
        "degree_or_order": int(order),
        "N": list(ns),
        "h": hs,
        "h_definition": "1/(N-1)",
        "output_grid": "family grid",
        "errors_inf": errors,
        "observed_orders": observed,
        "final_retained_pair_order": float(observed[-1]),
        "roundoff_floor_guard_applied": False,
        "work_per_point_nodal": float(op.work_model.total_flops / op.x.size),
        "storage_per_point": float(op.work_model.storage_nnz / op.x.size),
    }


def _spline_convergence(family: str, kind: str, degree: int, ns: tuple[int, ...]) -> dict[str, object]:
    convergence = wc.actual_grid_convergence(family, degree, ns)
    op = ba.build_operator(kind, degree, ns[-1])
    row: dict[str, object] = {
        "degree_or_order": int(degree),
        "N": convergence["N"],
        "h": convergence["h"],
        "h_definition": convergence["h_definition"],
        "output_grid": convergence["output_grid"],
        "errors_inf": convergence["errors_inf"],
        "endpoint_errors": convergence["endpoint_errors"],
        "roundoff_floors": convergence["roundoff_floors"],
        "fit_mask": convergence["fit_mask"],
        "observed_orders": convergence["pairwise_orders"],
        "pre_roundoff_fit_order": convergence["pre_roundoff_fit_order"],
        "fit_N": convergence["fit_N"],
        "fit_N_count": convergence["fit_N_count"],
        "fit_N_min": convergence["fit_N_min"],
        "fit_N_max": convergence["fit_N_max"],
        "final_retained_pair_order": convergence["final_retained_pair_order"],
        "endpoint_pre_roundoff_fit_order": convergence[
            "endpoint_pre_roundoff_fit_order"
        ],
        "roundoff_floor_guard_applied": True,
        "tolerance_rationale": convergence["tolerance_rationale"],
        "work_per_point_nodal": float(op.work_model.total_flops / op.x.size),
        "storage_per_point": float(op.work_model.storage_nnz / op.x.size),
    }
    if kind == "I":
        row["work_per_point_coefficient"] = float(
            op.work_model_coefficient.total_flops / op.x.size
        )
    return row


def build_formal_order_summary(
    *,
    degrees: Iterable[int] = (4, 6, 8),
    convergence_degrees: Iterable[int] = (4, 6, 8),
    ns_map: dict[int, tuple[int, ...]] | None = None,
    include_exact_endpoints: bool = True,
) -> dict[str, object]:
    ns_map = DEFAULT_NS_MAP if ns_map is None else ns_map
    degrees = tuple(int(p) for p in degrees)
    convergence_degrees = tuple(int(p) for p in convergence_degrees)
    summary: dict[str, object] = {
        "schema_version": 2,
        "claim_boundary": (
            "Finite-grid formal-order evidence only. Uniform Peano-kernel and "
            "endpoint-localization statements require separate analytic proofs."
        ),
        "polynomial_exactness": {},
        "quartet_calibration": {},
        "matched_order_calibration": {},
    }

    for family in ("BMF", "BMG"):
        for degree in degrees:
            use_exact = include_exact_endpoints and degree in (4, 6, 8)
            summary["polynomial_exactness"][f"{family}{degree}"] = _spline_polynomial_row(
                family, degree, include_exact_endpoint=use_exact
            )

    for q in convergence_degrees:
        ns = tuple(int(n) for n in ns_map[q])
        block: dict[str, object] = {}
        for label, kind, degree in ba.comparison_block(q):
            family = ba.scheme_family(label)
            if family in ("BMF", "BMG"):
                block[label] = _spline_convergence(family, kind, degree, ns)
            else:
                block[label] = _non_spline_convergence(label, kind, degree, ns)
        summary["quartet_calibration"][f"q{q}"] = block
        summary["matched_order_calibration"][f"q{q}"] = block
    return summary


def render_formal_order_csv(summary: dict[str, object]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow([
        "block", "scheme", "degree_or_order", "pre_roundoff_fit_order",
        "final_retained_pair_order", "fit_N_min", "fit_N_max", "fit_N_count",
        "work_per_point_nodal", "work_per_point_coefficient",
        "storage_per_point", "output_grid", "h_definition",
    ])
    for block_name, block in summary["matched_order_calibration"].items():
        for label, row in block.items():
            writer.writerow([
                block_name, label, row["degree_or_order"],
                row.get("pre_roundoff_fit_order", ""),
                row["final_retained_pair_order"],
                row.get("fit_N_min", ""), row.get("fit_N_max", ""),
                row.get("fit_N_count", ""),
                row["work_per_point_nodal"], row.get("work_per_point_coefficient", ""),
                row["storage_per_point"], row["output_grid"], row["h_definition"],
            ])
    return stream.getvalue()


def write_summary(summary: dict[str, object]) -> None:
    ensure_generated_dirs()
    (RESULTSDIR / "formal_order_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (RESULTSDIR / "formal_order_table.csv").write_text(
        render_formal_order_csv(summary), encoding="utf-8"
    )


def main() -> None:
    summary = build_formal_order_summary()
    write_summary(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
