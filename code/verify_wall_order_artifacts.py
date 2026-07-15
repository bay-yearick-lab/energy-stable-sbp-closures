#!/usr/bin/env python3
"""Independently recompute and verify persisted wall-order certificates."""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, localcontext
from fractions import Fraction
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True
for _thread_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_var] = "1"

import numpy as np

import check_formal_order as formal
from project_paths import RESULTSDIR
import s6_closure_moment_ladder as closure
import wall_order_certificates as wall


ARTIFACT_NAMES = (
    "wall_order_certificate_summary.json",
    "formal_order_summary.json",
    "formal_order_table.csv",
    "closure_moment_ladder_summary.json",
)

_CANONICAL_CACHE: dict[str, bytes] | None = None


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2).encode("utf-8")


def _closure_summary() -> dict[str, Any]:
    cfg = closure.EXPERIMENT
    return {
        "configuration": cfg,
        "rows": [
            closure.closure_moments(family, int(q), int(cfg["npts"]), int(cfg["ell_max"]))
            for q in cfg["qs"]
            for family in cfg["families"]
        ],
    }


def canonical_artifact_bytes() -> dict[str, bytes]:
    """Recompute all four canonical persisted artifacts without reading them."""
    global _CANONICAL_CACHE
    if _CANONICAL_CACHE is None:
        wall_summary = wall.build_summary()
        formal_summary = formal.build_formal_order_summary()
        closure_summary = _closure_summary()
        _CANONICAL_CACHE = {
            "wall_order_certificate_summary.json": _json_bytes(wall_summary),
            "formal_order_summary.json": _json_bytes(formal_summary),
            "formal_order_table.csv": formal.render_formal_order_csv(formal_summary).encode("utf-8"),
            "closure_moment_ladder_summary.json": _json_bytes(closure_summary),
        }
    return dict(_CANONICAL_CACHE)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _load_json(payload: bytes, name: str, errors: list[str]) -> Any | None:
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"{name}: invalid JSON: {exc}")
        return None


def _check_finite(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            errors.append(f"{path}: non-finite floating value")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite(item, f"{path}[{index}]", errors)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _check_finite(item, f"{path}.{key}", errors)
        return
    errors.append(f"{path}: unsupported persisted type {type(value).__name__}")


def _check_exact_endpoint_certificate(
    certificate: dict[str, Any], path: str, errors: list[str]
) -> None:
    precision = certificate.get("precision", {})
    digits = precision.get("decimal_digits")
    if not isinstance(digits, int) or digits < 80:
        errors.append(f"{path}: exact endpoint precision must be at least 80 decimal digits")
        return
    endpoints = certificate.get("endpoints")
    if not isinstance(endpoints, dict) or set(endpoints) != {"left", "right"}:
        errors.append(f"{path}: exact endpoint schema/coverage mismatch")
        return
    for side in ("left", "right"):
        endpoint = endpoints[side]
        endpoint_path = f"{path}.endpoints.{side}"
        try:
            numerator = int(endpoint["exact_rational_numerator"])
            denominator = int(endpoint["exact_rational_denominator"])
            fraction = Fraction(numerator, denominator)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append(f"{endpoint_path}: invalid exact fraction: {exc}")
            continue
        exact_nonzero = fraction != 0
        if endpoint.get("exact_rational_nonzero") is not exact_nonzero:
            errors.append(f"{endpoint_path}: exact_rational_nonzero disagrees with fraction")

        with localcontext() as context:
            context.prec = digits
            expected_decimal = Decimal(fraction.numerator) / Decimal(fraction.denominator)
        try:
            stored_decimal = Decimal(endpoint["scaled_moment_decimal"])
        except Exception as exc:
            errors.append(f"{endpoint_path}: invalid decimal serialization: {exc}")
        else:
            if stored_decimal != expected_decimal:
                errors.append(f"{endpoint_path}: decimal serialization disagrees with exact fraction")

        interval = endpoint.get("scaled_moment_interval")
        if not isinstance(interval, list) or len(interval) != 2:
            errors.append(f"{endpoint_path}: invalid ULP interval schema")
            continue
        lower, upper = interval
        if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
            errors.append(f"{endpoint_path}: ULP interval bounds are not numeric")
            continue
        center = float(fraction)
        expected_interval = [
            float(np.nextafter(center, -np.inf)),
            float(np.nextafter(center, np.inf)),
        ]
        if interval != expected_interval:
            errors.append(f"{endpoint_path}: ULP interval is stale or inconsistent")
        if Fraction.from_float(float(lower)) > fraction or Fraction.from_float(float(upper)) < fraction:
            errors.append(f"{endpoint_path}: ULP interval does not enclose exact fraction")
        interval_excludes_zero = bool(upper < 0.0 or lower > 0.0)
        if endpoint.get("interval_excludes_zero") is not interval_excludes_zero:
            errors.append(f"{endpoint_path}: interval_excludes_zero disagrees with stored interval")


def _check_moment(
    moment: dict[str, Any], structural_exact_degree: int, path: str, errors: list[str]
) -> None:
    required = {
        "ell", "raw_moment", "scaled_moment", "raw_tolerance",
        "scaled_tolerance", "within_backward_tolerance", "structural_zero",
    }
    if not required.issubset(moment):
        errors.append(f"{path}: moment schema missing {sorted(required - set(moment))}")
        return
    raw = moment["raw_moment"]
    tolerance = moment["raw_tolerance"]
    within = bool(abs(raw) <= tolerance)
    if moment["within_backward_tolerance"] is not within:
        errors.append(f"{path}: within_backward_tolerance inequality is inconsistent")
    structural = bool(moment["ell"] <= structural_exact_degree)
    if moment["structural_zero"] is not structural:
        errors.append(f"{path}: structural_zero disagrees with exact algebraic degree")


def _check_wall_summary(data: dict[str, Any], expected: dict[str, Any], errors: list[str]) -> None:
    required_top = {
        "schema_version", "claim_boundary", "operator_identities", "floating_moments",
        "exact_endpoint_moments", "actual_grid_convergence",
    }
    if set(data) != required_top:
        errors.append("wall JSON: exact top-level schema mismatch")
    identity_keys = {
        f"{family}{degree}_N{npts}"
        for family in ("BMF", "BMG")
        for degree in (4, 6, 8)
        for npts in (degree + 9, degree + 13, degree + 17)
    }
    six_family_degree_keys = {
        f"{family}{degree}" for family in ("BMF", "BMG") for degree in (4, 6, 8)
    }
    coverage = {
        "operator_identities": identity_keys,
        "floating_moments": {
            f"{family}{degree}_N{degree + 17}"
            for family in ("BMF", "BMG") for degree in (4, 6, 8)
        },
        "exact_endpoint_moments": six_family_degree_keys,
        "actual_grid_convergence": six_family_degree_keys,
    }
    for section, required_keys in coverage.items():
        if set(data.get(section, {})) != required_keys:
            errors.append(f"wall JSON: {section} coverage mismatch")

    for key, identity in data.get("operator_identities", {}).items():
        path = f"wall.operator_identities.{key}"
        residual = identity.get("identity_residual_inf")
        tolerance = identity.get("identity_tolerance")
        if not isinstance(residual, (int, float)) or not isinstance(tolerance, (int, float)):
            errors.append(f"{path}: identity residual/tolerance schema invalid")
            continue
        passes = bool(residual <= tolerance)
        if identity.get("identity_passes") is not passes:
            errors.append(f"{path}: identity_passes inequality is inconsistent")
        if identity.get("comparison") != "full operator matrix":
            errors.append(f"{path}: identity does not cover the full operator matrix")
        if identity.get("coordinate_vectors_compared") != identity.get("N"):
            errors.append(f"{path}: coordinate-vector coverage is incomplete")

    for key, certificate in data.get("floating_moments", {}).items():
        degree = certificate.get("degree")
        for row_index, row in enumerate(certificate.get("rows", [])):
            for moment_index, moment in enumerate(row.get("moments", [])):
                _check_moment(
                    moment, degree,
                    f"wall.floating_moments.{key}.rows[{row_index}].moments[{moment_index}]",
                    errors,
                )

    for key, sequence in data.get("exact_endpoint_moments", {}).items():
        sequence_path = f"wall.exact_endpoint_moments.{key}"
        certificates = sequence.get("certificates", [])
        all_exclude = True
        for index, certificate in enumerate(certificates):
            _check_exact_endpoint_certificate(
                certificate, f"{sequence_path}.certificates[{index}]", errors
            )
            for endpoint in certificate.get("endpoints", {}).values():
                all_exclude = all_exclude and bool(endpoint.get("interval_excludes_zero"))
        if sequence.get("all_intervals_exclude_zero") is not all_exclude:
            errors.append(f"{sequence_path}: all_intervals_exclude_zero is inconsistent")

    for key, convergence in data.get("actual_grid_convergence", {}).items():
        path = f"wall.actual_grid_convergence.{key}"
        if "tail_order" in convergence or "observed_order" in convergence:
            errors.append(f"{path}: obsolete fitted-order field is present")
        fit_n = convergence.get("fit_N", [])
        if convergence.get("fit_N_count") != len(fit_n):
            errors.append(f"{path}: fit_N_count is inconsistent")
        if fit_n and (
            convergence.get("fit_N_min") != min(fit_n)
            or convergence.get("fit_N_max") != max(fit_n)
        ):
            errors.append(f"{path}: fit N range is inconsistent")


def _check_formal_summary(data: dict[str, Any], expected: dict[str, Any], errors: list[str]) -> None:
    required_top = {
        "schema_version", "claim_boundary", "polynomial_exactness",
        "quartet_calibration", "matched_order_calibration",
    }
    if set(data) != required_top:
        errors.append("formal JSON: exact top-level schema mismatch")
    spline_keys = {
        f"{family}{degree}" for family in ("BMF", "BMG") for degree in (4, 6, 8)
    }
    if set(data.get("polynomial_exactness", {})) != spline_keys:
        errors.append("formal JSON: polynomial certificate coverage mismatch")
    expected_blocks = {"q4", "q6", "q8"}
    for section in ("quartet_calibration", "matched_order_calibration"):
        blocks = data.get(section, {})
        if set(blocks) != expected_blocks:
            errors.append(f"formal JSON: {section} block coverage mismatch")
            continue
        for q in (4, 6, 8):
            labels = {f"{family}{q}" for family in ("E", "C", "SBP", "DRP", "BMF", "BMG")}
            if set(blocks.get(f"q{q}", {})) != labels:
                errors.append(f"formal JSON: {section}.q{q} family coverage mismatch")
    for key, row in data.get("polynomial_exactness", {}).items():
        certificate = row.get("exact_endpoint_certificate")
        if certificate is not None:
            _check_exact_endpoint_certificate(
                certificate, f"formal.polynomial_exactness.{key}.exact_endpoint_certificate", errors
            )
    for block_name, block in data.get("matched_order_calibration", {}).items():
        for label, row in block.items():
            path = f"formal.matched_order_calibration.{block_name}.{label}"
            if "tail_order" in row:
                errors.append(f"{path}: obsolete tail_order field is present")


def _check_closure_summary(data: dict[str, Any], expected: dict[str, Any], errors: list[str]) -> None:
    if set(data) != {"configuration", "rows"}:
        errors.append("closure JSON: exact top-level schema mismatch")
    rows = data.get("rows", [])
    row_keys = {(row.get("family"), row.get("q")) for row in rows}
    required_row_keys = {
        (family, q)
        for q in (4, 6, 8)
        for family in ("E", "C", "SBP", "DRP", "BMF", "BMG")
    }
    if len(rows) != 18 or row_keys != required_row_keys:
        errors.append("closure JSON: family/order coverage mismatch")
    for index, row in enumerate(rows):
        path = f"closure.rows[{index}]"
        degree = row.get("exact_polynomial_degree")
        for moment_index, moment in enumerate(row.get("moments", [])):
            _check_moment(moment, degree, f"{path}.moments[{moment_index}]", errors)
        reference = row.get("reference_exact_endpoint_certificate")
        if reference is not None:
            _check_exact_endpoint_certificate(
                reference, f"{path}.reference_exact_endpoint_certificate", errors
            )
            if row.get("reference_exact_endpoint_N") != reference.get("N"):
                errors.append(f"{path}: reference exact endpoint N is inconsistent")
        if "finite_grid_exact_endpoint_certificate" in row:
            errors.append(f"{path}: smaller-grid exact reference is mislabeled as same-grid")
        floating = row.get("floating_first_free_moment_at_N", {})
        if floating.get("N") != row.get("N"):
            errors.append(f"{path}: N-grid floating first-free moment is inconsistent")


def _check_csv(payload: bytes, expected_payload: bytes, errors: list[str]) -> None:
    try:
        text = payload.decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(text)))
        expected_rows = list(csv.DictReader(io.StringIO(expected_payload.decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        errors.append(f"formal CSV: invalid encoding or CSV: {exc}")
        return
    header = text.splitlines()[0].split(",") if text.splitlines() else []
    if "pre_roundoff_fit_order" not in header or "tail_order" in header:
        errors.append("formal CSV: fitted-order header schema is incorrect")
    if len(rows) != 18:
        errors.append("formal CSV: expected exactly 18 family/order rows")
    if rows != expected_rows:
        errors.append("formal CSV: canonical row values or coverage are stale")


def verify_artifact_directory(artifact_dir: Path) -> list[str]:
    artifact_dir = Path(artifact_dir)
    expected_bytes = canonical_artifact_bytes()
    errors: list[str] = []
    payloads: dict[str, bytes] = {}
    for name in ARTIFACT_NAMES:
        path = artifact_dir / name
        if not path.is_file():
            errors.append(f"{name}: missing persisted artifact")
            continue
        payload = path.read_bytes()
        payloads[name] = payload
        if payload != expected_bytes[name]:
            errors.append(f"{name}: byte/staleness mismatch against canonical recomputation")

    parsed: dict[str, Any] = {}
    for name in ARTIFACT_NAMES:
        if not name.endswith(".json") or name not in payloads:
            continue
        value = _load_json(payloads[name], name, errors)
        if value is not None:
            parsed[name] = value
            _check_finite(value, name, errors)

    expected_parsed = {
        name: json.loads(payload.decode("utf-8"))
        for name, payload in expected_bytes.items() if name.endswith(".json")
    }
    wall_name = "wall_order_certificate_summary.json"
    formal_name = "formal_order_summary.json"
    closure_name = "closure_moment_ladder_summary.json"
    if wall_name in parsed:
        _check_wall_summary(parsed[wall_name], expected_parsed[wall_name], errors)
    if formal_name in parsed:
        _check_formal_summary(parsed[formal_name], expected_parsed[formal_name], errors)
    if closure_name in parsed:
        _check_closure_summary(parsed[closure_name], expected_parsed[closure_name], errors)
    csv_name = "formal_order_table.csv"
    if csv_name in payloads:
        _check_csv(payloads[csv_name], expected_bytes[csv_name], errors)
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path(RESULTSDIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = verify_artifact_directory(args.artifact_dir)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"wall-order artifacts verified: {len(ARTIFACT_NAMES)} canonical files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
