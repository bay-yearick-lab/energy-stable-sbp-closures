#!/usr/bin/env python3
"""Finite-grid certificates for bounded BMF and BMG first derivatives.

The module deliberately separates three claims:

* double-precision operator identities and condition-aware moment residuals;
* convergence of the operators on the grids actually marched by each route;
* exact-rational, finite-grid endpoint moments for nonvanishing checks.

None of these finite-grid artifacts proves a grid-uniform Peano-kernel bound or
an all-refinement endpoint limit.  Those are separate analytic obligations.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import boundary_spline_analysis as ba
from project_paths import RESULTSDIR, ensure_generated_dirs


FLOAT_BITS = int(np.finfo(float).nmant + 1)
TOLERANCE_RATIONALE = (
    "64 eps times the rowwise absolute matrix-product scale plus the exact "
    "derivative scale; this is a floating-point backward-error test, not a "
    "fixed absolute threshold"
)
CONVERGENCE_FLOOR_RATIONALE = (
    "32 eps times the full absolute matrix-product and exact-derivative scale; "
    "points at or below this condition-aware floor, and every finer point, "
    "are excluded from the slope fit"
)
MATRIX_IDENTITY_TOLERANCE_RATIONALE = (
    "128 eps times the larger assembled/factorized infinity norm for a full "
    "operator-matrix comparison over every coordinate vector"
)


@dataclass(frozen=True)
class ActualDerivativeMap:
    family: str
    degree: int
    npts: int
    input_nodes: np.ndarray
    output_nodes: np.ndarray
    matrix: np.ndarray
    output_grid: str


def knot_span_width(npts: int, degree: int) -> float:
    if npts <= degree:
        raise ValueError("npts must exceed the spline degree")
    return 1.0 / float(npts - degree)


def actual_derivative_map(family: str, degree: int, npts: int) -> ActualDerivativeMap:
    family = family.upper()
    if family == "BMF":
        ops = ba.bmf_staggered_operators(int(npts), int(degree))
        return ActualDerivativeMap(
            family=family,
            degree=int(degree),
            npts=int(npts),
            input_nodes=ops.x.copy(),
            output_nodes=ops.eta.copy(),
            matrix=ops.d10.copy(),
            output_grid="derivative Greville",
        )
    if family == "BMG":
        op = ba.build_operator("I", int(degree), int(npts))
        return ActualDerivativeMap(
            family=family,
            degree=int(degree),
            npts=int(npts),
            input_nodes=op.x.copy(),
            output_nodes=op.x.copy(),
            matrix=op.matrix.copy(),
            output_grid="primal Greville",
        )
    raise ValueError(f"actual derivative map is defined only for BMF/BMG, got {family!r}")


def backward_error_tolerance(
    matrix: np.ndarray,
    values: np.ndarray,
    exact: np.ndarray | float,
    *,
    precision_bits: int = FLOAT_BITS,
    safety: float = 64.0,
) -> float:
    """Return an infinity-norm absolute backward-error tolerance."""
    matrix = np.asarray(matrix, dtype=float)
    values = np.asarray(values, dtype=float)
    exact = np.asarray(exact, dtype=float)
    eps = 2.0 ** (1 - int(precision_bits))
    product_scale = float(np.max(np.abs(matrix) @ np.abs(values)))
    exact_scale = float(np.max(np.abs(exact))) if exact.size else 0.0
    return float(safety * eps * max(product_scale + exact_scale, np.finfo(float).tiny))


def _moment_record(
    row: np.ndarray,
    input_nodes: np.ndarray,
    output_node: float,
    ell: int,
    h: float,
    structural_exact_degree: int,
) -> dict[str, Any]:
    shifted = np.asarray(input_nodes - output_node, dtype=float) ** int(ell)
    target = 0.0 if ell != 1 else 1.0
    raw = float(row @ shifted - target)
    tol = backward_error_tolerance(
        np.asarray(row, dtype=float)[None, :], shifted, np.array([target]),
    )
    scale = h ** (1 - int(ell))
    within_backward_tolerance = bool(abs(raw) <= tol)
    return {
        "ell": int(ell),
        "raw_moment": raw,
        "scaled_moment": float(scale * raw),
        "raw_tolerance": float(tol),
        "scaled_tolerance": float(scale * tol),
        "within_backward_tolerance": within_backward_tolerance,
        "structural_zero": bool(int(ell) <= int(structural_exact_degree)),
        "monomial": "(x-output_node)^ell",
    }


def floating_moment_certificate(family: str, degree: int, npts: int) -> dict[str, Any]:
    derivative = actual_derivative_map(family, degree, npts)
    h = knot_span_width(npts, degree)
    rows = []
    for row_index, (row, z) in enumerate(zip(derivative.matrix, derivative.output_nodes)):
        rows.append({
            "row": int(row_index),
            "output_node": float(z),
            "moments": [
                _moment_record(
                    row, derivative.input_nodes, float(z), ell, h, int(degree)
                )
                for ell in range(int(degree) + 2)
            ],
        })
    return {
        "schema_version": 2,
        "family": derivative.family,
        "degree": int(degree),
        "N": int(npts),
        "h": float(h),
        "h_definition": "1/(N-p)",
        "input_grid": "primal Greville",
        "output_grid": derivative.output_grid,
        "precision": {"arithmetic": "IEEE-754 binary64", "bits": FLOAT_BITS},
        "tolerance_rationale": TOLERANCE_RATIONALE,
        "rows": rows,
    }


def matrix_identity_metrics(
    assembled: np.ndarray, factorized: np.ndarray, *, safety: float = 128.0
) -> dict[str, Any]:
    assembled = np.asarray(assembled, dtype=float)
    factorized = np.asarray(factorized, dtype=float)
    if assembled.shape != factorized.shape or assembled.ndim != 2:
        raise ValueError("assembled and factorized operators must be same-shape matrices")
    residual = float(np.linalg.norm(assembled - factorized, np.inf))
    scale = max(
        float(np.linalg.norm(assembled, np.inf)),
        float(np.linalg.norm(factorized, np.inf)),
        np.finfo(float).tiny,
    )
    tolerance = float(safety * np.finfo(float).eps * scale)
    return {
        "residual_inf": residual,
        "tolerance": tolerance,
        "passes": bool(residual <= tolerance),
    }


def operator_identity_certificate(family: str, degree: int, npts: int) -> dict[str, Any]:
    """Compare the full assembled matrix with the factorized application path."""
    derivative = actual_derivative_map(family, degree, npts)
    if derivative.family == "BMF":
        op = ba.build_operator("B", int(degree), int(npts))
    else:
        op = ba.build_operator("I", int(degree), int(npts))
    coordinate_vectors = np.eye(int(npts), dtype=float)
    factorized_matrix = op.field_and_derivative(
        derivative.output_nodes, coordinate_vectors
    )[1]
    metrics = matrix_identity_metrics(derivative.matrix, factorized_matrix)
    out: dict[str, Any] = {
        "family": derivative.family,
        "degree": int(degree),
        "N": int(npts),
        "output_grid": derivative.output_grid,
        "precision": {"arithmetic": "IEEE-754 binary64", "bits": FLOAT_BITS},
        "tolerance_rationale": MATRIX_IDENTITY_TOLERANCE_RATIONALE,
        "comparison": "full operator matrix",
        "coordinate_vectors_compared": int(npts),
        "identity": (
            "G R u = (I_h u)'(eta)" if derivative.family == "BMF"
            else "D_G R u = [Pi_h (I_h u)'](x)"
        ),
        "identity_residual_inf": metrics["residual_inf"],
        "identity_tolerance": metrics["tolerance"],
        "identity_passes": metrics["passes"],
    }
    if derivative.family == "BMF":
        same_point = ba.build_operator("B", int(degree), int(npts)).matrix
        endpoint_residual = float(np.max(np.abs(
            derivative.matrix[[0, -1]] - same_point[[0, -1]]
        )))
        endpoint_tolerance = float(
            64.0 * np.finfo(float).eps * max(
                np.linalg.norm(derivative.matrix[[0, -1]], np.inf),
                np.linalg.norm(same_point[[0, -1]], np.inf),
            )
        )
        out.update({
            "endpoint_row_residual_inf": endpoint_residual,
            "endpoint_row_tolerance": endpoint_tolerance,
            "endpoint_rows_match_same_point": bool(endpoint_residual <= endpoint_tolerance),
        })
    return out


def manufactured_u(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.exp(x) + 0.31 * np.sin(7.3 * x) + 0.2 * np.cos(5.7 * x) + 0.07 * x**11


def manufactured_ux(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.exp(x) + 0.31 * 7.3 * np.cos(7.3 * x) - 0.2 * 5.7 * np.sin(5.7 * x) + 0.77 * x**10


def fit_pre_roundoff_orders(
    *, hs: np.ndarray, errors: np.ndarray, floors: np.ndarray
) -> dict[str, Any]:
    hs = np.asarray(hs, dtype=float)
    errors = np.asarray(errors, dtype=float)
    floors = np.asarray(floors, dtype=float)
    if hs.shape != errors.shape or hs.shape != floors.shape:
        raise ValueError("hs, errors, and floors must have the same shape")
    if hs.size < 2 or np.any(hs <= 0.0) or np.any(errors <= 0.0):
        raise ValueError("at least two positive spacings and errors are required")

    fit_mask = errors > floors
    # A roundoff plateau invalidates that point and all finer points.  Keeping a
    # contiguous pre-floor prefix prevents selective slope fitting.
    invalid = np.flatnonzero(~fit_mask)
    if invalid.size:
        fit_mask[invalid[0]:] = False
    indices = np.flatnonzero(fit_mask)
    if indices.size < 2:
        raise ValueError("fewer than two pre-roundoff points remain")
    slope, intercept = np.polyfit(np.log(hs[indices]), np.log(errors[indices]), 1)
    pairwise = []
    for left, right in zip(indices[:-1], indices[1:]):
        pairwise.append(float(
            np.log(errors[left] / errors[right]) / np.log(hs[left] / hs[right])
        ))
    return {
        "fit_mask": [bool(v) for v in fit_mask],
        "fit_point_count": int(indices.size),
        "pre_roundoff_fit_order": float(slope),
        "log_intercept": float(intercept),
        "pairwise_orders": pairwise,
    }


def actual_grid_convergence(
    family: str, degree: int, npts_sequence: Iterable[int]
) -> dict[str, Any]:
    npts_values = [int(n) for n in npts_sequence]
    hs: list[float] = []
    errors: list[float] = []
    endpoint_errors: list[float] = []
    floors: list[float] = []
    output_grid = ""
    for npts in npts_values:
        derivative = actual_derivative_map(family, degree, npts)
        values = manufactured_u(derivative.input_nodes)
        exact = manufactured_ux(derivative.output_nodes)
        error = np.abs(derivative.matrix @ values - exact)
        hs.append(knot_span_width(npts, degree))
        errors.append(float(np.max(error)))
        endpoint_errors.append(float(max(error[0], error[-1])))
        floors.append(backward_error_tolerance(
            derivative.matrix, values, exact, precision_bits=FLOAT_BITS, safety=32.0
        ))
        output_grid = derivative.output_grid
    fit = fit_pre_roundoff_orders(
        hs=np.asarray(hs), errors=np.asarray(errors), floors=np.asarray(floors)
    )
    endpoint_fit = fit_pre_roundoff_orders(
        hs=np.asarray(hs), errors=np.asarray(endpoint_errors), floors=np.asarray(floors)
    )
    fit_n = [npts for npts, used in zip(npts_values, fit["fit_mask"]) if used]
    endpoint_fit_n = [
        npts for npts, used in zip(npts_values, endpoint_fit["fit_mask"]) if used
    ]
    return {
        "family": family.upper(),
        "degree": int(degree),
        "N": npts_values,
        "h": hs,
        "h_definition": "1/(N-p)",
        "output_grid": output_grid,
        "errors_inf": errors,
        "endpoint_errors": endpoint_errors,
        "roundoff_floors": floors,
        "roundoff_floor_guard_applied": True,
        **fit,
        "fit_N": fit_n,
        "fit_N_count": len(fit_n),
        "fit_N_min": min(fit_n),
        "fit_N_max": max(fit_n),
        "final_retained_pair_order": fit["pairwise_orders"][-1],
        "endpoint_pre_roundoff_fit_order": endpoint_fit["pre_roundoff_fit_order"],
        "endpoint_fit_mask": endpoint_fit["fit_mask"],
        "endpoint_fit_N": endpoint_fit_n,
        "endpoint_fit_N_count": len(endpoint_fit_n),
        "endpoint_final_retained_pair_order": endpoint_fit["pairwise_orders"][-1],
        "tolerance_rationale": CONVERGENCE_FLOOR_RATIONALE,
    }


def _open_uniform_knots_fraction(npts: int, degree: int) -> list[Fraction]:
    spans = npts - degree
    return (
        [Fraction(0)] * (degree + 1)
        + [Fraction(j, spans) for j in range(1, spans)]
        + [Fraction(1)] * (degree + 1)
    )


def _bspline_values_fraction(
    knots: list[Fraction], degree: int, x: Fraction
) -> list[Fraction]:
    vals = [Fraction(0) for _ in range(len(knots) - 1)]
    if x == knots[-1]:
        positive = [i for i in range(len(knots) - 1) if knots[i] < knots[i + 1]]
        vals[positive[-1]] = Fraction(1)
    else:
        for i in range(len(vals)):
            if knots[i] <= x < knots[i + 1]:
                vals[i] = Fraction(1)
                break
    for order in range(1, degree + 1):
        nxt: list[Fraction] = []
        for i in range(len(vals) - 1):
            left_denom = knots[i + order] - knots[i]
            right_denom = knots[i + order + 1] - knots[i + 1]
            left = Fraction(0) if left_denom == 0 else (x - knots[i]) * vals[i] / left_denom
            right = Fraction(0) if right_denom == 0 else (knots[i + order + 1] - x) * vals[i + 1] / right_denom
            nxt.append(left + right)
        vals = nxt
    return vals


def _greville_fraction(knots: list[Fraction], degree: int) -> list[Fraction]:
    nbasis = len(knots) - degree - 1
    if degree == 0:
        return [(knots[i] + knots[i + 1]) / 2 for i in range(nbasis)]
    return [sum(knots[i + 1:i + degree + 1], Fraction(0)) / degree for i in range(nbasis)]


def _fraction_solve(
    matrix: list[list[Fraction]], rhs: list[list[Fraction]]
) -> list[list[Fraction]]:
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be nonempty and square")
    nrhs = len(rhs[0])
    aug = [list(matrix[i]) + list(rhs[i]) for i in range(n)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if pivot is None:
            raise ValueError("singular exact-rational matrix")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(n):
            if row == col or aug[row][col] == 0:
                continue
            factor = aug[row][col]
            aug[row] = [
                aug[row][j] - factor * aug[col][j]
                for j in range(n + nrhs)
            ]
    return [row[n:] for row in aug]


def _fraction_matvec(matrix: list[list[Fraction]], vector: list[Fraction]) -> list[Fraction]:
    return [sum((a * b for a, b in zip(row, vector)), Fraction(0)) for row in matrix]


def _local_basis_polynomials_fraction(
    knots: list[Fraction], degree: int
) -> list[tuple[Fraction, list[list[Fraction]]]]:
    """Return (span width, coefficient[basis][power]) on every positive span."""
    nbasis = len(knots) - degree - 1
    samples = [Fraction(j + 1, degree + 2) for j in range(degree + 1)]
    vand = [[s**power for power in range(degree + 1)] for s in samples]
    out: list[tuple[Fraction, list[list[Fraction]]]] = []
    for left, right in zip(knots[:-1], knots[1:]):
        if right <= left:
            continue
        width = right - left
        sampled_rows = [
            _bspline_values_fraction(knots, degree, left + width * s)
            for s in samples
        ]
        coeff_by_power = _fraction_solve(vand, sampled_rows)
        coeff_by_basis = [
            [coeff_by_power[power][basis] for power in range(degree + 1)]
            for basis in range(nbasis)
        ]
        out.append((width, coeff_by_basis))
    return out


def _mass_advection_fraction(
    knots: list[Fraction], degree: int
) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    nbasis = len(knots) - degree - 1
    mass = [[Fraction(0) for _ in range(nbasis)] for _ in range(nbasis)]
    adv = [[Fraction(0) for _ in range(nbasis)] for _ in range(nbasis)]
    for width, coeffs in _local_basis_polynomials_fraction(knots, degree):
        for i in range(nbasis):
            ci = coeffs[i]
            if not any(ci):
                continue
            for j in range(nbasis):
                cj = coeffs[j]
                if not any(cj):
                    continue
                mass[i][j] += width * sum(
                    (ci[a] * cj[b] / Fraction(a + b + 1)
                     for a in range(degree + 1) for b in range(degree + 1)),
                    Fraction(0),
                )
                adv[i][j] += sum(
                    (ci[a] * b * cj[b] / Fraction(a + b)
                     for a in range(degree + 1) for b in range(1, degree + 1)),
                    Fraction(0),
                )
    return mass, adv


def _exact_endpoint_moment(
    family: str, degree: int, npts: int, side: str
) -> Fraction:
    knots = _open_uniform_knots_fraction(npts, degree)
    x = _greville_fraction(knots, degree)
    wall = Fraction(0) if side == "left" else Fraction(1)
    values = [(node - wall) ** (degree + 1) for node in x]
    bmat = [_bspline_values_fraction(knots, degree, node) for node in x]
    coeffs = [row[0] for row in _fraction_solve(bmat, [[v] for v in values])]

    if family == "BMF":
        derivative_knots = knots[1:-1]
        derivative_degree = degree - 1
        derivative_values = _bspline_values_fraction(
            derivative_knots, derivative_degree, wall
        )
        delta = [[Fraction(0) for _ in range(npts)] for _ in range(npts - 1)]
        for i in range(npts - 1):
            denom = knots[i + degree + 1] - knots[i + 1]
            scale = Fraction(degree, 1) / denom
            delta[i][i] = -scale
            delta[i][i + 1] = scale
        derivative_coeffs = _fraction_matvec(delta, coeffs)
        raw = sum((a * b for a, b in zip(derivative_values, derivative_coeffs)), Fraction(0))
    elif family == "BMG":
        mass, adv = _mass_advection_fraction(knots, degree)
        rhs = _fraction_matvec(adv, coeffs)
        projected = [row[0] for row in _fraction_solve(mass, [[v] for v in rhs])]
        wall_basis = _bspline_values_fraction(knots, degree, wall)
        raw = sum((a * b for a, b in zip(wall_basis, projected)), Fraction(0))
    else:
        raise ValueError(f"unsupported family {family!r}")

    h = Fraction(1, npts - degree)
    return raw / (h**degree)


def _fraction_decimal(value: Fraction, decimal_digits: int) -> str:
    with localcontext() as ctx:
        ctx.prec = int(decimal_digits)
        return format(Decimal(value.numerator) / Decimal(value.denominator), "g")


def _fraction_float_enclosure(value: Fraction) -> list[float]:
    center = float(value)
    return [
        float(np.nextafter(center, -np.inf)),
        float(np.nextafter(center, np.inf)),
    ]


def multiprecision_endpoint_certificate(
    family: str, degree: int, npts: int, *, decimal_digits: int = 90
) -> dict[str, Any]:
    """Return exact-rational finite-grid endpoint constants and enclosures."""
    if decimal_digits < 80:
        raise ValueError("endpoint certificates require at least 80 decimal digits")
    family = family.upper()
    endpoints: dict[str, Any] = {}
    output_size = npts - 1 if family == "BMF" else npts
    for side, row in (("left", 0), ("right", output_size - 1)):
        exact = _exact_endpoint_moment(family, int(degree), int(npts), side)
        interval = _fraction_float_enclosure(exact)
        exact_rational_nonzero = bool(exact != 0)
        interval_excludes_zero = bool(interval[1] < 0.0 or interval[0] > 0.0)
        endpoints[side] = {
            "row": int(row),
            "scaled_moment_decimal": _fraction_decimal(exact, decimal_digits),
            "scaled_moment_interval": interval,
            "exact_rational_numerator": str(exact.numerator),
            "exact_rational_denominator": str(exact.denominator),
            "exact_rational_nonzero": exact_rational_nonzero,
            "interval_excludes_zero": interval_excludes_zero,
        }
    return {
        "schema_version": 2,
        "family": family,
        "degree": int(degree),
        "N": int(npts),
        "output_size": int(output_size),
        "output_grid": "derivative Greville" if family == "BMF" else "primal Greville",
        "h": knot_span_width(npts, degree),
        "h_definition": "1/(N-p)",
        "precision": {
            "arithmetic": "exact rational; decimal serialization only",
            "decimal_digits": int(decimal_digits),
        },
        "exact_polynomial_degree": int(degree),
        "first_nonzero_moment_degree": int(degree + 1),
        "endpoints": endpoints,
        "claim_boundary": (
            "Finite-grid exact-rational nonvanishing only; no all-N limit, "
            "uniform Peano bound, or endpoint-localization proof is inferred."
        ),
    }


def multiprecision_endpoint_sequence(
    family: str,
    degree: int,
    npts_sequence: Iterable[int],
    *,
    decimal_digits: int = 90,
) -> dict[str, Any]:
    certificates = [
        multiprecision_endpoint_certificate(
            family, degree, int(npts), decimal_digits=decimal_digits
        )
        for npts in npts_sequence
    ]
    all_nonzero = all(
        endpoint["interval_excludes_zero"]
        for certificate in certificates
        for endpoint in certificate["endpoints"].values()
    )
    return {
        "schema_version": 2,
        "family": family.upper(),
        "degree": int(degree),
        "N": [certificate["N"] for certificate in certificates],
        "certificate_count": len(certificates),
        "precision": certificates[0]["precision"] if certificates else {
            "arithmetic": "exact rational; decimal serialization only",
            "decimal_digits": int(decimal_digits),
        },
        "all_intervals_exclude_zero": bool(all_nonzero),
        "proves_all_N_limit": False,
        "claim_boundary": (
            "Exact-rational nonvanishing on the listed finite grids. The "
            "sequence neither encloses an all-N limit nor supplies the W3 tail proof."
        ),
        "certificates": certificates,
    }


def build_summary() -> dict[str, Any]:
    ns_map = {
        4: (17, 25, 33, 49, 65),
        6: (17, 25, 33, 49, 65),
        8: (17, 25, 33, 49, 65),
    }
    return {
        "schema_version": 2,
        "claim_boundary": (
            "Finite-grid operator, moment, and convergence certificates. "
            "W3 all-N, W4 uniform Peano, and W5 localization remain analytic obligations."
        ),
        "operator_identities": {
            f"{family}{degree}_N{npts}": operator_identity_certificate(
                family, degree, npts
            )
            for family in ("BMF", "BMG")
            for degree in (4, 6, 8)
            for npts in (degree + 9, degree + 13, degree + 17)
        },
        "floating_moments": {
            f"{family}{degree}_N{degree + 17}": floating_moment_certificate(
                family, degree, degree + 17
            )
            for family in ("BMF", "BMG") for degree in (4, 6, 8)
        },
        "exact_endpoint_moments": {
            f"{family}{degree}": multiprecision_endpoint_sequence(
                family, degree, (degree + 9, degree + 17, degree + 25)
            )
            for family in ("BMF", "BMG") for degree in (4, 6, 8)
        },
        "actual_grid_convergence": {
            f"{family}{degree}": actual_grid_convergence(family, degree, ns_map[degree])
            for family in ("BMF", "BMG") for degree in (4, 6, 8)
        },
    }


def main() -> None:
    ensure_generated_dirs()
    summary = build_summary()
    path = Path(RESULTSDIR) / "wall_order_certificate_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
