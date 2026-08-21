#!/usr/bin/env python3
"""Build and verify the finite C5--C8 route-C integration certificate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from bmf_hdiv_ns2d import BMFHDivNS2D
from project_paths import RESULTSDIR, ensure_generated_dirs


SCHEMA = "bmf-hdiv-ns2d-c5-c8-v4"
FORMULATION = {
    "domain": "unit square [0,1]^2",
    "velocity_space": "ux in S^p tensor dS^p; uz in dS^p tensor S^p",
    "pressure_space": "dS^p tensor dS^p modulo constants",
    "normal_trace": "strong",
    "tangential_trace": "symmetric Nitsche",
    "penalty": "gamma/h with gamma=4p^2",
    "metrics": "exact natural L2 tensor metrics",
    "saddle_pressure_unknown": "dual pressure H_Q p; primal pressure reconstructed by H_Q^{-1}",
    "convection": "direct exact-quadrature weak conservative/skew Galerkin form",
    "momentum": "full-space constant test with the reaction of eliminated normal-trace coefficients",
}
SCOPE = {
    "established": "finite two-dimensional operator, wall-reaction momentum ledger, and smooth velocity-pressure Stokes gates C5-C8",
    "not_established": "by this finite certificate: production Navier-Stokes time integration, high-Re flow, or three-dimensional extension",
}
PROOF_DEBTS = [
    "production nonlinear time integration with inhomogeneous tangential wall work",
    "a standard physical wall-flow benchmark using the adopted route-C pair",
    "three-dimensional route-C viscous and convective integration",
]
TOP_KEYS = {
    "schema",
    "formulation",
    "nitsche",
    "nitsche_coercivity_sweep",
    "divergence_rank",
    "random_projected_states",
    "adversarial_conservation_sweep",
    "stokes_refinement",
    "stokes_observed_orders",
    "stokes_pressure_observed_orders",
    "scope",
    "proof_debts",
}
NITSCHE_KEYS = {
    "npts",
    "degree",
    "penalty_gamma",
    "penalty",
    "input_sha256",
    "content_sha256",
    "matrix_asymmetry_l2_squared",
    "matrix_l2_squared",
    "minimum_eigenvalue",
    "maximum_eigenvalue",
    "polynomial_consistency",
}
CONSISTENCY_KEYS = {
    "residual_l2_squared",
    "load_l2_squared",
    "tangential_trace_l2_squared",
}
SWEEP_KEYS = {
    "degree",
    "elements",
    "npts",
    "penalty_gamma",
    "input_sha256",
    "content_sha256",
    "matrix_asymmetry_l2_squared",
    "matrix_l2_squared",
    "minimum_eigenvalue",
    "maximum_eigenvalue",
}
RANK_KEYS = {
    "weak_tangential_rank",
    "weak_tangential_nullity",
    "strong_tangential_rank",
    "strong_tangential_nullity",
    "weak_matrix_change_from_core",
}
STATE_KEYS = {
    "seed",
    "input_sha256",
    "content_sha256",
    "pressure_adjoint_unprojected",
    "pressure_cancellation_projected",
    "convection_projected",
    "momentum_reaction",
}
ADVERSARIAL_KEYS = {
    "degree",
    "elements",
    "npts",
    "seed",
    "input_sha256",
    "content_sha256",
    "tangential_trace_l2_squared",
    "pressure_cancellation",
    "convection",
    "momentum_reaction",
}
PRESSURE_KEYS = {
    "velocity_gradient_work",
    "divergence_pressure_work",
    "adjoint_residual",
    "raw_scale",
    "relative_adjoint_residual",
    "relative_pressure_work",
}
CONVECTION_KEYS = {
    "pointwise_divergence_l2_squared",
    "pointwise_divergence_scale_squared",
    "relative_pointwise_divergence",
    "kinetic_energy_production",
    "kinetic_energy_scale",
    "relative_kinetic_energy_production",
    "normal_trace_max",
    "conservative_identity_residual_l2_squared",
    "conservative_identity_scale_l2_squared",
    "relative_conservative_identity_residual",
    "quadrature_points_per_span",
    "maximum_galerkin_integrand_degree",
    "gauss_exact_polynomial_degree",
    "convective_momentum_flux_x_defect",
    "convective_momentum_flux_z_defect",
    "convective_momentum_flux_defect",
    "convective_momentum_flux_scale",
    "relative_convective_momentum_flux_defect",
}
MOMENTUM_REACTION_KEYS = {
    "momentum_rate_x",
    "momentum_rate_z",
    "convective_moment_x",
    "convective_moment_z",
    "pressure_moment_x",
    "pressure_moment_z",
    "viscous_moment_x",
    "viscous_moment_z",
    "constraint_reaction_x",
    "constraint_reaction_z",
    "constraint_reaction_scale",
    "momentum_balance_x_defect",
    "momentum_balance_z_defect",
    "momentum_balance_defect",
    "momentum_balance_scale",
    "relative_momentum_balance_defect",
    "active_equation_residual",
    "active_equation_scale",
    "relative_active_equation_residual",
    "constraint_support_defect",
    "relative_constraint_support_defect",
    "convective_projection_moment_defect",
    "relative_convective_projection_moment_defect",
}
STOKES_KEYS = {
    "npts",
    "degree",
    "elements",
    "mesh_size",
    "input_sha256",
    "content_sha256",
    "velocity_error_l2_squared",
    "velocity_exact_l2_squared",
    "relative_velocity_error",
    "pressure_unknown",
    "pressure_mod_constant_error_l2_squared",
    "pressure_exact_l2_squared",
    "relative_pressure_mod_constant_error",
    "pressure_mod_constant_shift",
    "divergence_l2_squared",
    "normal_trace_max",
    "tangential_data_l2_squared",
    "linear_residual_l2_squared",
    "linear_rhs_l2_squared",
}


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_array_input_sha256(metadata: dict[str, Any], *arrays: np.ndarray) -> str:
    normalized = [np.ascontiguousarray(np.asarray(array, dtype="<f8")) for array in arrays]
    header = {
        "metadata": metadata,
        "arrays": [
            {"dtype": "float64-little-endian", "shape": list(array.shape)}
            for array in normalized
        ],
    }
    digest = hashlib.sha256(
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    for array in normalized:
        digest.update(b"\n")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def record_content_sha256(record: dict[str, Any]) -> str:
    return canonical_json_sha256(
        {key: value for key, value in record.items() if key != "content_sha256"}
    )


def _normal_zero_state(solver: BMFHDivNS2D, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ux = rng.standard_normal(solver.core.ux_shape)
    uz = rng.standard_normal(solver.core.uz_shape)
    ux[[0, -1], :] = 0.0
    uz[:, [0, -1]] = 0.0
    return ux, uz


def build_certificate() -> dict[str, Any]:
    solver = BMFHDivNS2D(npts=8, degree=3)
    coercivity_sweep: list[dict[str, float | int]] = []
    for degree in (2, 3, 4, 6):
        for elements in (3, 5):
            cell = BMFHDivNS2D(npts=degree + elements, degree=degree)
            eigenvalues_cell = np.linalg.eigvalsh(cell.viscous_matrix)
            asymmetry = cell.viscous_matrix - cell.viscous_matrix.T
            record: dict[str, Any] = {
                "degree": degree,
                "elements": elements,
                "npts": degree + elements,
                "penalty_gamma": cell.penalty_gamma,
                "input_sha256": canonical_json_sha256(
                    {
                        "degree": degree,
                        "elements": elements,
                        "penalty_gamma": cell.penalty_gamma,
                    }
                ),
                "matrix_asymmetry_l2_squared": float(np.sum(asymmetry**2)),
                "matrix_l2_squared": float(np.sum(cell.viscous_matrix**2)),
                "minimum_eigenvalue": float(eigenvalues_cell[0]),
                "maximum_eigenvalue": float(eigenvalues_cell[-1]),
            }
            record["content_sha256"] = record_content_sha256(record)
            coercivity_sweep.append(record)
    random_states: list[dict[str, Any]] = []
    for seed in (41, 43, 47):
        ux, uz = _normal_zero_state(solver, seed=seed)
        rng = np.random.default_rng(seed + 1)
        pressure = solver.core.remove_pressure_mean(
            rng.standard_normal(solver.core.pressure_shape)
        )
        unprojected_pressure = solver.pressure_work_ledger(ux, uz, pressure)
        projected = solver.core.project(ux, uz)
        projected_pressure = solver.pressure_work_ledger(
            projected.ux,
            projected.uz,
            pressure,
        )
        _cux, _cuz, convection = solver.skew_convection(
            projected.ux,
            projected.uz,
        )
        momentum_reaction = solver.momentum_reaction_ledger(
            projected.ux,
            projected.uz,
            pressure,
        )
        record = {
            "seed": seed,
            "input_sha256": canonical_array_input_sha256(
                {"seed": seed, "kind": "normal-zero velocity and mean-zero pressure"},
                ux,
                uz,
                pressure,
            ),
            "pressure_adjoint_unprojected": unprojected_pressure,
            "pressure_cancellation_projected": projected_pressure,
            "convection_projected": convection,
            "momentum_reaction": momentum_reaction,
        }
        record["content_sha256"] = record_content_sha256(record)
        random_states.append(record)
    adversarial_sweep: list[dict[str, Any]] = []
    for degree in (2, 3, 4, 6):
        for elements in (3, 5):
            cell = BMFHDivNS2D(npts=degree + elements, degree=degree)
            seed = 1000 + 100 * degree + elements
            ux, uz = _normal_zero_state(cell, seed=seed)
            rng = np.random.default_rng(seed + 1)
            pressure = cell.core.remove_pressure_mean(
                rng.standard_normal(cell.core.pressure_shape)
            )
            projected = cell.core.project(ux, uz)
            _cux, _cuz, convection = cell.skew_convection(
                projected.ux,
                projected.uz,
            )
            record = {
                "degree": degree,
                "elements": elements,
                "npts": degree + elements,
                "seed": seed,
                "input_sha256": canonical_array_input_sha256(
                    {
                        "degree": degree,
                        "elements": elements,
                        "seed": seed,
                        "kind": "zero-normal projected velocity with nonzero tangential wall trace",
                    },
                    ux,
                    uz,
                    pressure,
                ),
                "tangential_trace_l2_squared": cell.tangential_trace_l2_squared(
                    projected.ux,
                    projected.uz,
                ),
                "pressure_cancellation": cell.pressure_work_ledger(
                    projected.ux,
                    projected.uz,
                    pressure,
                ),
                "convection": convection,
                "momentum_reaction": cell.momentum_reaction_ledger(
                    projected.ux,
                    projected.uz,
                    pressure,
                ),
            }
            record["content_sha256"] = record_content_sha256(record)
            adversarial_sweep.append(record)
    stokes: list[dict[str, Any]] = []
    for npts in (6, 8, 11):
        record = BMFHDivNS2D(npts=npts, degree=2).solve_manufactured_stokes()
        record["input_sha256"] = canonical_json_sha256(
            {
                "degree": 2,
                "npts": npts,
                "viscosity": 0.7,
                "velocity": "curl(sin(pi*x)*sin(pi*z))",
                "pressure": "cos(pi*x)*cos(pi*z)",
                "pde": "-nu*laplacian(u)-grad(p)=f",
            }
        )
        record["content_sha256"] = record_content_sha256(record)
        stokes.append(record)
    errors = [math.sqrt(float(record["velocity_error_l2_squared"])) for record in stokes]
    orders = [
        math.log(errors[index - 1] / errors[index])
        / math.log(
            float(stokes[index - 1]["mesh_size"])
            / float(stokes[index]["mesh_size"])
        )
        for index in (1, 2)
    ]
    pressure_errors = [
        math.sqrt(float(record["pressure_mod_constant_error_l2_squared"]))
        for record in stokes
    ]
    pressure_orders = [
        math.log(pressure_errors[index - 1] / pressure_errors[index])
        / math.log(
            float(stokes[index - 1]["mesh_size"])
            / float(stokes[index]["mesh_size"])
        )
        for index in (1, 2)
    ]
    eigenvalues = np.linalg.eigvalsh(solver.viscous_matrix)
    asymmetry_squared = float(
        np.sum((solver.viscous_matrix - solver.viscous_matrix.T) ** 2)
    )
    matrix_squared = float(np.sum(solver.viscous_matrix**2))
    nitsche: dict[str, Any] = {
        "npts": solver.npts,
        "degree": solver.degree,
        "penalty_gamma": solver.penalty_gamma,
        "penalty": solver.penalty,
        "input_sha256": canonical_json_sha256(
            {
                "npts": solver.npts,
                "degree": solver.degree,
                "penalty_gamma": solver.penalty_gamma,
            }
        ),
        "matrix_asymmetry_l2_squared": asymmetry_squared,
        "matrix_l2_squared": matrix_squared,
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "polynomial_consistency": solver.polynomial_nitsche_consistency(),
    }
    nitsche["content_sha256"] = record_content_sha256(nitsche)
    return {
        "schema": SCHEMA,
        "formulation": dict(FORMULATION),
        "nitsche": nitsche,
        "nitsche_coercivity_sweep": coercivity_sweep,
        "divergence_rank": solver.divergence_rank_certificate(),
        "random_projected_states": random_states,
        "adversarial_conservation_sweep": adversarial_sweep,
        "stokes_refinement": stokes,
        "stokes_observed_orders": orders,
        "stokes_pressure_observed_orders": pressure_orders,
        "scope": dict(SCOPE),
        "proof_debts": list(PROOF_DEBTS),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_finite_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite number at {path}")


def _require_exact_keys(record: Any, expected: set[str], context: str) -> dict[str, Any]:
    _require(isinstance(record, dict), f"{context} must be an object")
    actual = set(record)
    _require(
        actual == expected,
        f"{context} schema mismatch; missing={sorted(expected - actual)}, "
        f"extra={sorted(actual - expected)}",
    )
    return record


def _number(record: dict[str, Any], key: str, context: str) -> float:
    value = record[key]
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{context}.{key} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{context}.{key} must be finite")
    return result


def _nonnegative(
    record: dict[str, Any],
    key: str,
    context: str,
    *,
    positive: bool = False,
) -> float:
    value = _number(record, key, context)
    _require(
        value > 0.0 if positive else value >= 0.0,
        f"{context}.{key} must be {'positive' if positive else 'nonnegative'}",
    )
    return value


def _integer(record: dict[str, Any], key: str, context: str) -> int:
    value = _nonnegative(record, key, context)
    _require(value.is_integer(), f"{context}.{key} must be an integer")
    return int(value)


def _require_digest(value: Any, context: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{context} must be a lowercase SHA-256 digest",
    )
    return value


def _verify_record_digest(record: dict[str, Any], context: str) -> None:
    _require_digest(record["input_sha256"], f"{context}.input_sha256")
    content = _require_digest(record["content_sha256"], f"{context}.content_sha256")
    _require(
        content == record_content_sha256(record),
        f"{context} content digest mismatch",
    )


def _close(
    actual: float,
    expected: float,
    *,
    rtol: float = 2.0e-12,
    atol: float = 5.0e-15,
) -> bool:
    return abs(actual - expected) <= atol + rtol * max(abs(actual), abs(expected))


def _verify_pressure_lane(
    lane: Any,
    context: str,
    *,
    require_nontrivial: bool,
) -> None:
    pressure = _require_exact_keys(lane, PRESSURE_KEYS, context)
    left = _number(pressure, "velocity_gradient_work", context)
    right = _number(pressure, "divergence_pressure_work", context)
    residual = _nonnegative(pressure, "adjoint_residual", context)
    scale = _nonnegative(pressure, "raw_scale", context, positive=True)
    relative_residual = _nonnegative(
        pressure,
        "relative_adjoint_residual",
        context,
    )
    relative_work = _nonnegative(pressure, "relative_pressure_work", context)
    _require(_close(residual, abs(left - right)), f"{context} adjoint residual mismatch")
    _require(
        _close(relative_residual, residual / scale),
        f"{context} relative adjoint residual mismatch",
    )
    _require(
        _close(relative_work, abs(left) / scale),
        f"{context} relative pressure work mismatch",
    )
    _require(scale > 1.0e-8, f"{context} pressure-work denominator is vacuous")
    _require(relative_residual < 2.0e-13, f"{context} adjoint identity failed")
    if require_nontrivial:
        _require(relative_work > 1.0e-3, f"{context} pressure work is vacuous")
    else:
        _require(relative_work < 2.0e-12, f"{context} pressure cancellation failed")


def _verify_convection(lane: Any, context: str) -> None:
    convection = _require_exact_keys(lane, CONVECTION_KEYS, context)
    divergence_num = _nonnegative(
        convection,
        "pointwise_divergence_l2_squared",
        context,
    )
    divergence_den = _nonnegative(
        convection,
        "pointwise_divergence_scale_squared",
        context,
        positive=True,
    )
    divergence_relative = _nonnegative(
        convection,
        "relative_pointwise_divergence",
        context,
    )
    _require(
        _close(divergence_relative, math.sqrt(divergence_num / divergence_den)),
        f"{context} divergence ratio mismatch",
    )
    kinetic = _number(convection, "kinetic_energy_production", context)
    kinetic_scale = _nonnegative(
        convection,
        "kinetic_energy_scale",
        context,
        positive=True,
    )
    kinetic_relative = _nonnegative(
        convection,
        "relative_kinetic_energy_production",
        context,
    )
    _require(
        _close(kinetic_relative, abs(kinetic) / kinetic_scale),
        f"{context} kinetic-energy ratio mismatch",
    )
    identity_num = _nonnegative(
        convection,
        "conservative_identity_residual_l2_squared",
        context,
    )
    identity_den = _nonnegative(
        convection,
        "conservative_identity_scale_l2_squared",
        context,
        positive=True,
    )
    identity_relative = _nonnegative(
        convection,
        "relative_conservative_identity_residual",
        context,
    )
    _require(
        _close(identity_relative, math.sqrt(identity_num / identity_den)),
        f"{context} conservative identity ratio mismatch",
    )
    mx = _number(convection, "convective_momentum_flux_x_defect", context)
    mz = _number(convection, "convective_momentum_flux_z_defect", context)
    momentum = _nonnegative(
        convection,
        "convective_momentum_flux_defect",
        context,
    )
    momentum_scale = _nonnegative(
        convection,
        "convective_momentum_flux_scale",
        context,
        positive=True,
    )
    momentum_relative = _nonnegative(
        convection,
        "relative_convective_momentum_flux_defect",
        context,
    )
    _require(_close(momentum, math.hypot(mx, mz)), f"{context} flux norm mismatch")
    _require(
        _close(momentum_relative, momentum / momentum_scale),
        f"{context} relative flux defect mismatch",
    )
    normal_trace = _nonnegative(convection, "normal_trace_max", context)
    quadrature = _integer(convection, "quadrature_points_per_span", context)
    maximum_degree = _integer(
        convection,
        "maximum_galerkin_integrand_degree",
        context,
    )
    exact_degree = _integer(convection, "gauss_exact_polynomial_degree", context)
    _require(exact_degree == 2 * quadrature - 1, f"{context} Gauss degree mismatch")
    _require(exact_degree >= maximum_degree, f"{context} quadrature is not exact")
    _require(divergence_relative < 3.0e-12, f"{context} divergence gate failed")
    _require(kinetic_relative < 3.0e-12, f"{context} kinetic-energy gate failed")
    _require(momentum_relative < 3.0e-12, f"{context} momentum-flux gate failed")
    _require(identity_relative < 3.0e-13, f"{context} conservative identity failed")
    _require(normal_trace < 2.0e-13, f"{context} normal-trace hypothesis failed")


def _verify_momentum_reaction(lane: Any, context: str) -> None:
    record = _require_exact_keys(lane, MOMENTUM_REACTION_KEYS, context)
    for key in (
        "momentum_rate_x", "momentum_rate_z", "convective_moment_x",
        "convective_moment_z", "pressure_moment_x", "pressure_moment_z",
        "viscous_moment_x", "viscous_moment_z", "constraint_reaction_x",
        "constraint_reaction_z", "momentum_balance_x_defect",
        "momentum_balance_z_defect",
    ):
        _number(record, key, context)
    reaction_scale = _nonnegative(record, "constraint_reaction_scale", context, positive=True)
    defect_x = _number(record, "momentum_balance_x_defect", context)
    defect_z = _number(record, "momentum_balance_z_defect", context)
    defect = _nonnegative(record, "momentum_balance_defect", context)
    scale = _nonnegative(record, "momentum_balance_scale", context, positive=True)
    relative = _nonnegative(record, "relative_momentum_balance_defect", context)
    _require(_close(defect, math.hypot(defect_x, defect_z)), f"{context} balance norm mismatch")
    _require(_close(relative, defect / scale), f"{context} relative balance mismatch")
    active = _nonnegative(record, "active_equation_residual", context)
    active_scale = _nonnegative(record, "active_equation_scale", context, positive=True)
    active_relative = _nonnegative(record, "relative_active_equation_residual", context)
    _require(_close(active_relative, active / active_scale), f"{context} active residual ratio mismatch")
    support = _nonnegative(record, "constraint_support_defect", context)
    support_relative = _nonnegative(record, "relative_constraint_support_defect", context)
    _require(_close(support_relative, support / reaction_scale), f"{context} support ratio mismatch")
    projection = _nonnegative(record, "convective_projection_moment_defect", context)
    projection_relative = _nonnegative(
        record, "relative_convective_projection_moment_defect", context
    )
    _require(
        projection_relative >= 0.0 and projection >= 0.0,
        f"{context} projection residual invalid",
    )
    _require(relative < 3.0e-12, f"{context} momentum balance gate failed")
    _require(active_relative < 3.0e-12, f"{context} reduced equation gate failed")
    _require(support_relative < 3.0e-12, f"{context} reaction support gate failed")
    _require(projection_relative < 3.0e-12, f"{context} projection moment gate failed")


def _first_mismatch(recorded: Any, rebuilt: Any, path: str = "root") -> str | None:
    if isinstance(rebuilt, dict):
        if not isinstance(recorded, dict) or set(recorded) != set(rebuilt):
            return path
        for key in rebuilt:
            # The persisted digest is verified exactly against the persisted
            # record above.  A rebuild can differ by a few LAPACK roundoff bits,
            # so its digest is not expected to be byte-identical.
            if key == "content_sha256":
                continue
            mismatch = _first_mismatch(recorded[key], rebuilt[key], f"{path}.{key}")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(rebuilt, list):
        if not isinstance(recorded, list) or len(recorded) != len(rebuilt):
            return path
        for index, (actual, expected) in enumerate(zip(recorded, rebuilt)):
            mismatch = _first_mismatch(actual, expected, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(rebuilt, int) and not isinstance(rebuilt, bool):
        if isinstance(recorded, bool) or not isinstance(recorded, int):
            return path
        return None if recorded == rebuilt else path
    if isinstance(rebuilt, float):
        if isinstance(recorded, bool) or not isinstance(recorded, (int, float)):
            return path
        return None if _close(float(recorded), rebuilt) else path
    return None if recorded == rebuilt else path


def verify_certificate(data: dict[str, Any]) -> None:
    document = _require_exact_keys(data, TOP_KEYS, "certificate")
    _require_finite_tree(data)
    _require(document["schema"] == SCHEMA, "schema metadata mismatch")
    _require(document["formulation"] == FORMULATION, "formulation metadata mismatch")

    nitsche = _require_exact_keys(document["nitsche"], NITSCHE_KEYS, "nitsche")
    _verify_record_digest(nitsche, "nitsche")
    _require(_integer(nitsche, "npts", "nitsche") == 8, "Nitsche grid mismatch")
    _require(_integer(nitsche, "degree", "nitsche") == 3, "Nitsche degree mismatch")
    _require(_close(_number(nitsche, "penalty_gamma", "nitsche"), 36.0), "Nitsche penalty mismatch")
    _nonnegative(nitsche, "penalty", "nitsche", positive=True)
    asym = _nonnegative(nitsche, "matrix_asymmetry_l2_squared", "nitsche")
    matrix = _nonnegative(nitsche, "matrix_l2_squared", "nitsche", positive=True)
    _require(math.sqrt(asym / matrix) < 2.0e-13, "Nitsche symmetry gate failed")
    minimum = _nonnegative(nitsche, "minimum_eigenvalue", "nitsche", positive=True)
    maximum = _nonnegative(nitsche, "maximum_eigenvalue", "nitsche", positive=True)
    _require(minimum > 1.0e-3 and maximum > minimum, "Nitsche coercivity gate failed")
    consistency = _require_exact_keys(
        nitsche["polynomial_consistency"],
        CONSISTENCY_KEYS,
        "nitsche.polynomial_consistency",
    )
    residual = _nonnegative(consistency, "residual_l2_squared", "consistency")
    load = _nonnegative(consistency, "load_l2_squared", "consistency", positive=True)
    trace = _nonnegative(consistency, "tangential_trace_l2_squared", "consistency", positive=True)
    _require(load > 0.0 and trace > 1.0e-2, "Nitsche raw consistency norms invalid")
    _require(residual / load < 2.0e-24, "Nitsche consistency gate failed")

    sweep = document["nitsche_coercivity_sweep"]
    _require(isinstance(sweep, list) and len(sweep) == 8, "Nitsche coercivity grid set mismatch")
    expected_cells = [(degree, elements) for degree in (2, 3, 4, 6) for elements in (3, 5)]
    actual_cells: list[tuple[int, int]] = []
    for index, raw_cell in enumerate(sweep):
        cell = _require_exact_keys(raw_cell, SWEEP_KEYS, f"sweep[{index}]")
        _verify_record_digest(cell, f"sweep[{index}]")
        degree = _integer(cell, "degree", f"sweep[{index}]")
        elements = _integer(cell, "elements", f"sweep[{index}]")
        actual_cells.append((degree, elements))
        _require(_integer(cell, "npts", f"sweep[{index}]") == degree + elements, "Nitsche npts metadata mismatch")
        _require(
            _close(_number(cell, "penalty_gamma", f"sweep[{index}]"), 4.0 * degree**2),
            "Nitsche sweep penalty metadata mismatch",
        )
        cell_matrix = _nonnegative(cell, "matrix_l2_squared", f"sweep[{index}]", positive=True)
        cell_asymmetry = _nonnegative(cell, "matrix_asymmetry_l2_squared", f"sweep[{index}]")
        _require(math.sqrt(cell_asymmetry / cell_matrix) < 2.0e-13, "Nitsche sweep symmetry gate failed")
        cell_minimum = _nonnegative(cell, "minimum_eigenvalue", f"sweep[{index}]", positive=True)
        cell_maximum = _nonnegative(cell, "maximum_eigenvalue", f"sweep[{index}]", positive=True)
        _require(cell_minimum > 0.1, "Nitsche sweep coercivity gate failed")
        _require(cell_maximum > cell_minimum, "Nitsche sweep spectrum invalid")
    _require(actual_cells == expected_cells, "Nitsche coercivity grid metadata mismatch")

    rank = _require_exact_keys(document["divergence_rank"], RANK_KEYS, "divergence_rank")
    _require(rank.get("weak_tangential_rank") == 48, "weak tangential rank mismatch")
    _require(rank.get("weak_tangential_nullity") == 1, "weak tangential nullity mismatch")
    _require(rank.get("strong_tangential_nullity") == 5, "strong tangential nullity mismatch")
    _require(float(rank.get("weak_matrix_change_from_core", math.nan)) == 0.0, "weak B matrix changed")

    random_states = document["random_projected_states"]
    _require(isinstance(random_states, list) and len(random_states) == 3, "random-state gate set mismatch")
    actual_seeds: list[int] = []
    for index, raw_state in enumerate(random_states):
        state = _require_exact_keys(raw_state, STATE_KEYS, f"state[{index}]")
        actual_seeds.append(_integer(state, "seed", f"state[{index}]"))
        _verify_record_digest(state, f"state[{index}]")
        _verify_pressure_lane(
            state["pressure_adjoint_unprojected"],
            f"state[{index}].pressure_adjoint_unprojected",
            require_nontrivial=True,
        )
        _verify_pressure_lane(
            state["pressure_cancellation_projected"],
            f"state[{index}].pressure_cancellation_projected",
            require_nontrivial=False,
        )
        _verify_convection(state["convection_projected"], f"state[{index}].convection")
        _verify_momentum_reaction(
            state["momentum_reaction"], f"state[{index}].momentum_reaction"
        )
    _require(actual_seeds == [41, 43, 47], "random-state seed metadata mismatch")

    adversarial = document["adversarial_conservation_sweep"]
    _require(
        isinstance(adversarial, list) and len(adversarial) == 8,
        "adversarial conservation grid set mismatch",
    )
    actual_adversarial_cells: list[tuple[int, int]] = []
    for index, raw_cell in enumerate(adversarial):
        context = f"adversarial[{index}]"
        cell = _require_exact_keys(raw_cell, ADVERSARIAL_KEYS, context)
        _verify_record_digest(cell, context)
        degree = _integer(cell, "degree", context)
        elements = _integer(cell, "elements", context)
        actual_adversarial_cells.append((degree, elements))
        _require(_integer(cell, "npts", context) == degree + elements, f"{context} npts mismatch")
        _require(
            _integer(cell, "seed", context) == 1000 + 100 * degree + elements,
            f"{context} seed mismatch",
        )
        trace = _nonnegative(cell, "tangential_trace_l2_squared", context, positive=True)
        _require(trace > 1.0e-4, f"{context} tangential wall state is vacuous")
        _verify_pressure_lane(
            cell["pressure_cancellation"],
            f"{context}.pressure_cancellation",
            require_nontrivial=False,
        )
        _verify_convection(cell["convection"], f"{context}.convection")
        _require(
            _number(
                cell["convection"],
                "relative_pointwise_divergence",
                f"{context}.convection",
            ) < 1.0e-12,
            f"{context} exact-divergence acceptance threshold failed",
        )
        _require(
            _number(
                cell["convection"],
                "relative_kinetic_energy_production",
                f"{context}.convection",
            ) < 1.0e-12,
            f"{context} kinetic-energy acceptance threshold failed",
        )
        _verify_momentum_reaction(cell["momentum_reaction"], f"{context}.momentum_reaction")
        _require(
            _number(
                cell["momentum_reaction"],
                "relative_momentum_balance_defect",
                f"{context}.momentum_reaction",
            ) < 1.0e-12,
            f"{context} momentum acceptance threshold failed",
        )
    _require(
        actual_adversarial_cells == expected_cells,
        "adversarial conservation grid metadata mismatch",
    )

    stokes = document["stokes_refinement"]
    _require(isinstance(stokes, list) and len(stokes) == 3, "Stokes grid set mismatch")
    errors: list[float] = []
    pressure_errors: list[float] = []
    stokes_npts: list[int] = []
    for index, raw_cell in enumerate(stokes):
        cell = _require_exact_keys(raw_cell, STOKES_KEYS, f"stokes[{index}]")
        stokes_npts.append(_integer(cell, "npts", f"stokes[{index}]"))
        _require(_integer(cell, "degree", f"stokes[{index}]") == 2, "Stokes degree metadata mismatch")
        _verify_record_digest(cell, f"stokes[{index}]")
        numerator = _nonnegative(cell, "velocity_error_l2_squared", f"stokes[{index}]", positive=True)
        denominator = _nonnegative(cell, "velocity_exact_l2_squared", f"stokes[{index}]", positive=True)
        errors.append(math.sqrt(numerator))
        _require(
            _close(_nonnegative(cell, "relative_velocity_error", f"stokes[{index}]"), math.sqrt(numerator / denominator)),
            "Stokes derived relative error mismatch",
        )
        _require(cell["pressure_unknown"] == "dual pressure H_Q p", "Stokes pressure representation mismatch")
        pressure_num = _nonnegative(cell, "pressure_mod_constant_error_l2_squared", f"stokes[{index}]", positive=True)
        pressure_den = _nonnegative(cell, "pressure_exact_l2_squared", f"stokes[{index}]", positive=True)
        pressure_errors.append(math.sqrt(pressure_num))
        _require(
            _close(
                _nonnegative(cell, "relative_pressure_mod_constant_error", f"stokes[{index}]"),
                math.sqrt(pressure_num / pressure_den),
            ),
            "Stokes pressure relative error mismatch",
        )
        _number(cell, "pressure_mod_constant_shift", f"stokes[{index}]")
        _require(_nonnegative(cell, "divergence_l2_squared", f"stokes[{index}]") < 2.0e-20, "Stokes divergence gate failed")
        _require(_nonnegative(cell, "normal_trace_max", f"stokes[{index}]") < 2.0e-13, "Stokes normal trace gate failed")
        _require(_nonnegative(cell, "tangential_data_l2_squared", f"stokes[{index}]") > 1.0, "Stokes tangential data missing")
        linear_den = _nonnegative(cell, "linear_rhs_l2_squared", f"stokes[{index}]", positive=True)
        linear_num = _nonnegative(cell, "linear_residual_l2_squared", f"stokes[{index}]")
        _require(linear_num / linear_den < 2.0e-24, "Stokes linear solve gate failed")
    _require(stokes_npts == [6, 8, 11], "Stokes grid metadata mismatch")
    _require(errors[0] > errors[1] > errors[2], "Stokes errors do not decrease")
    expected_orders = [
        math.log(errors[index - 1] / errors[index])
        / math.log(float(stokes[index - 1]["mesh_size"]) / float(stokes[index]["mesh_size"]))
        for index in (1, 2)
    ]
    orders = document["stokes_observed_orders"]
    _require(isinstance(orders, list) and len(orders) == 2, "missing Stokes derived orders")
    _require(all(_close(float(a), b) for a, b in zip(orders, expected_orders)), "Stokes derived order mismatch")
    _require(min(float(value) for value in orders) > 1.7, "Stokes convergence gate failed")
    expected_pressure_orders = [
        math.log(pressure_errors[index - 1] / pressure_errors[index])
        / math.log(float(stokes[index - 1]["mesh_size"]) / float(stokes[index]["mesh_size"]))
        for index in (1, 2)
    ]
    pressure_orders = document["stokes_pressure_observed_orders"]
    _require(isinstance(pressure_orders, list) and len(pressure_orders) == 2, "missing pressure orders")
    _require(
        all(_close(float(actual), expected) for actual, expected in zip(pressure_orders, expected_pressure_orders)),
        "Stokes pressure derived order mismatch",
    )
    _require(min(float(value) for value in pressure_orders) > 1.7, "Stokes pressure convergence gate failed")

    _require(document["scope"] == SCOPE, "exact scope metadata mismatch")
    _require(document["proof_debts"] == PROOF_DEBTS, "exact proof-debt metadata mismatch")

    rebuilt = build_certificate()
    mismatch = _first_mismatch(document, rebuilt)
    _require(mismatch is None, f"deterministic rebuild mismatch at {mismatch}")


def load_json_strict(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    with path.open() as stream:
        data = json.load(stream, parse_constant=reject_constant)
    if not isinstance(data, dict):
        raise ValueError("certificate must contain a JSON object")
    return data


def write_json_strict(path: Path, data: dict[str, Any]) -> None:
    _require_finite_tree(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTSDIR / "bmf_hdiv_ns2d_certificate.json",
    )
    args = parser.parse_args()
    if args.verify is not None:
        verify_certificate(load_json_strict(args.verify))
        print(f"verified {args.verify}")
        return
    ensure_generated_dirs()
    certificate = build_certificate()
    verify_certificate(certificate)
    write_json_strict(args.output, certificate)
    print(f"wrote and verified {args.output}")


if __name__ == "__main__":
    main()
