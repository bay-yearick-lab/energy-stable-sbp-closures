#!/usr/bin/env python3
"""Certify coefficient residual equivalence and native wall traction."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from bmf_hdiv_ns2d import BMFHDivNS2D
from project_paths import RESULTSDIR, ensure_generated_dirs
from tensor_compatible_flow import (
    TensorCompatibleFlow2D,
    TensorCompatibleFlow3D,
    VelocityPair,
    relative_pair_error,
)


SCHEMA = "tensor-compatible-flow-equivalence-v1"
OUTPUT = RESULTSDIR / "tensor_compatible_flow_equivalence.json"
DEGREES = (2, 4, 6, 8)
RESIDUAL_TOLERANCE = 2.0e-12
TRACTION_TOLERANCE = 2.0e-11


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _active_pair(pair: VelocityPair) -> VelocityPair:
    return VelocityPair(ux=pair.ux[1:-1, :], uz=pair.uz[:, 1:-1])


def _legacy_active_pair(stepper: BMFHDivNS2D, vector: np.ndarray) -> VelocityPair:
    split = (stepper.npts - 2) * (stepper.npts - 1)
    return VelocityPair(
        ux=np.asarray(vector[:split]).reshape(stepper.npts - 2, stepper.npts - 1),
        uz=np.asarray(vector[split:]).reshape(stepper.npts - 1, stepper.npts - 2),
    )


def _pair_norm(pair: VelocityPair) -> float:
    return math.sqrt(float(np.sum(pair.ux**2) + np.sum(pair.uz**2)))


def _coefficient_from_values(
    left: np.ndarray,
    values: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    first = np.linalg.solve(left, values)
    return np.linalg.solve(right, first.T).T


def _polynomial_traction_record() -> dict[str, float]:
    operator = TensorCompatibleFlow2D(11, 4)
    axis = operator.axis
    b0 = axis.greville0.toarray()
    b1 = axis.greville1.toarray()

    def f(value: np.ndarray) -> np.ndarray:
        return value**2 * (1.0 - value) ** 2

    def fp(value: np.ndarray) -> np.ndarray:
        return 2.0 * value - 6.0 * value**2 + 4.0 * value**3

    def fpp(value: np.ndarray) -> np.ndarray:
        return 2.0 - 12.0 * value + 12.0 * value**2

    ux_values = f(axis.x0[:, None]) * fp(axis.x1[None, :])
    uz_values = -fp(axis.x1[:, None]) * f(axis.x0[None, :])
    pressure_values = axis.x1[:, None] + 2.0 * axis.x1[None, :]
    ux = _coefficient_from_values(b0, ux_values, b1)
    uz = _coefficient_from_values(b1, uz_values, b0)
    pressure = _coefficient_from_values(b1, pressure_values, b1)
    walls = operator.wall_traction(ux, uz, pressure, viscosity=0.3)

    maximum_traction_error = 0.0
    maximum_flux = 0.0
    q = axis.q
    for name, coordinate, sign, normal_axis in (
        ("x0", 0.0, -1.0, "x"),
        ("x1", 1.0, 1.0, "x"),
        ("z0", 0.0, -1.0, "z"),
        ("z1", 1.0, 1.0, "z"),
    ):
        if normal_axis == "x":
            x = np.full_like(q, coordinate)
            z = q
            pressure_exact = x + 2.0 * z
            ux_x = fp(x) * fp(z)
            ux_z = f(x) * fpp(z)
            uz_x = -fpp(x) * f(z)
            tx = sign * (-pressure_exact + 0.6 * ux_x)
            tz = sign * 0.3 * (uz_x + ux_z)
        else:
            x = q
            z = np.full_like(q, coordinate)
            pressure_exact = x + 2.0 * z
            ux_z = f(x) * fpp(z)
            uz_x = -fpp(x) * f(z)
            uz_z = -fp(x) * fp(z)
            tx = sign * 0.3 * (ux_z + uz_x)
            tz = sign * (-pressure_exact + 0.6 * uz_z)
        maximum_traction_error = max(
            maximum_traction_error,
            float(np.max(np.abs(walls[name]["traction_x"] - tx))),
            float(np.max(np.abs(walls[name]["traction_z"] - tz))),
        )
        maximum_flux = max(
            maximum_flux, float(np.max(np.abs(walls[name]["normal_flux"])))
        )
    return {
        "maximum_absolute_traction_error": maximum_traction_error,
        "maximum_absolute_normal_flux": maximum_flux,
    }


def _degree_record(degree: int) -> dict[str, float | int]:
    npts = degree + 5
    tensor = TensorCompatibleFlow2D(
        npts=npts,
        degree=degree,
        penalty_gamma=4.0 * degree**2,
    )
    dense = BMFHDivNS2D(
        npts=npts,
        degree=degree,
        penalty_gamma=4.0 * degree**2,
    )
    rng = np.random.default_rng(800 + degree)
    ux = rng.standard_normal(tensor.ux_shape)
    uz = rng.standard_normal(tensor.uz_shape)
    ux[[0, -1], :] = 0.0
    uz[:, [0, -1]] = 0.0
    dual_pressure = rng.standard_normal(tensor.pressure_shape)
    zero = lambda x, z: np.zeros(np.broadcast_shapes(np.shape(x), np.shape(z)))
    lid = lambda x, z: np.broadcast_to(
        np.isclose(np.asarray(z, dtype=float), 1.0).astype(float),
        np.broadcast_shapes(np.shape(x), np.shape(z)),
    )
    channels = tensor.channel_residuals(ux, uz, dual_pressure, lid, zero)
    nodal = tensor.nodal_velocity(ux, uz)
    active_nodal = dense._pack_active(nodal.ux, nodal.uz)

    mass_active = _legacy_active_pair(dense, dense.velocity_mass @ active_nodal)
    mass_oracle = tensor.coefficient_dual_from_nodal(
        mass_active.ux, mass_active.uz
    )
    viscous_active = _legacy_active_pair(dense, dense.viscous_matrix @ active_nodal)
    viscous_oracle = tensor.coefficient_dual_from_nodal(
        viscous_active.ux, viscous_active.uz
    )
    nodal_dual_pressure = tensor.nodal_dual_pressure(dual_pressure)
    pressure_active = _legacy_active_pair(
        dense, dense.divergence_matrix.T @ nodal_dual_pressure.ravel()
    )
    pressure_oracle = tensor.coefficient_dual_from_nodal(
        pressure_active.ux, pressure_active.uz
    )
    wall_active = _legacy_active_pair(
        dense, dense._assemble_viscous_rhs(zero, zero, lid, zero)
    )
    wall_oracle = tensor.coefficient_dual_from_nodal(
        wall_active.ux, wall_active.uz
    )

    cux, cuz, _ledger = dense.skew_convection(nodal.ux, nodal.uz)
    full_convection = np.concatenate([cux.ravel(), cuz.ravel()])
    full_dual = dense.velocity_mass_full @ full_convection
    full_split = int(np.prod(dense.core.ux_shape))
    full_ux_dual = full_dual[:full_split].reshape(dense.core.ux_shape)
    full_uz_dual = full_dual[full_split:].reshape(dense.core.uz_shape)
    convection_oracle = VelocityPair(
        ux=np.asarray(tensor.axis.greville0.T @ full_ux_dual @ tensor.axis.greville1),
        uz=np.asarray(tensor.axis.greville1.T @ full_uz_dual @ tensor.axis.greville0),
    )

    historical_active = _legacy_active_pair(
        dense, dense.velocity_mass @ dense._pack_active(cux, cuz)
    )
    historical_coefficient = tensor.coefficient_dual_from_nodal(
        historical_active.ux, historical_active.uz
    )
    divergence_nodal = dense.core.divergence(nodal.ux, nodal.uz)
    divergence_tensor = np.asarray(
        tensor.axis.greville1 @ channels.divergence @ tensor.axis.greville1.T
    )
    divergence_error = np.linalg.norm(divergence_nodal - divergence_tensor) / max(
        np.linalg.norm(divergence_nodal), 1.0e-300
    )
    energy = tensor.convective_energy_production(ux, uz)
    energy_scale = np.linalg.norm(ux) * np.linalg.norm(
        channels.convection.ux
    ) + np.linalg.norm(uz) * np.linalg.norm(channels.convection.uz)
    storage = tensor.storage_record()
    return {
        "degree": degree,
        "npts": npts,
        "mass_relative_error": relative_pair_error(
            _active_pair(channels.mass), mass_oracle
        ),
        "viscous_relative_error": relative_pair_error(
            _active_pair(channels.viscous), viscous_oracle
        ),
        "pressure_relative_error": relative_pair_error(
            _active_pair(channels.pressure), pressure_oracle
        ),
        "convection_relative_error": relative_pair_error(
            channels.convection, convection_oracle
        ),
        "wall_load_relative_error": relative_pair_error(
            _active_pair(channels.wall), wall_oracle
        ),
        "divergence_relative_error": float(divergence_error),
        "convective_energy_relative_residual": abs(energy)
        / max(energy_scale, 1.0e-300),
        "historical_reduced_convection_relative_difference": relative_pair_error(
            _active_pair(channels.convection), historical_coefficient
        ),
        "stored_factor_entries": storage["stored_factor_entries"],
        "maximum_factor_bandwidth": storage["maximum_factor_bandwidth"],
        "velocity_states": storage["velocity_states"],
        "pressure_states": storage["pressure_states"],
    }


def _three_dimensional_record() -> dict[str, float | int]:
    operator = TensorCompatibleFlow3D(7, 4)
    axis = operator.axis
    rng = np.random.default_rng(417)
    ux = rng.standard_normal(operator.ux_shape)
    uy = rng.standard_normal(operator.uy_shape)
    uz = rng.standard_normal(operator.uz_shape)
    ux[[0, -1], :, :] = 0.0
    uy[:, [0, -1], :] = 0.0
    uz[:, :, [0, -1]] = 0.0
    pressure = rng.standard_normal(operator.pressure_shape)
    channels = operator.channel_residuals(ux, uy, uz, pressure)

    def apply(factors, field):
        arrays = tuple(
            factor.toarray() if hasattr(factor, "toarray") else np.asarray(factor)
            for factor in factors
        )
        return np.einsum(
            "ia,jb,kc,abc->ijk", *arrays, field, optimize=True
        )

    def moments(factors, values):
        arrays = tuple(
            factor.toarray() if hasattr(factor, "toarray") else np.asarray(factor)
            for factor in factors
        )
        return np.einsum(
            "ia,jb,kc,ijk->abc", *arrays, values, optimize=True
        )

    basis = (
        (axis.b0, axis.b1, axis.b1),
        (axis.b1, axis.b0, axis.b1),
        (axis.b1, axis.b1, axis.b0),
    )
    derivative = (
        (axis.d0, axis.d1, axis.d1),
        (axis.d1, axis.d0, axis.d1),
        (axis.d1, axis.d1, axis.d0),
    )
    fields = (ux, uy, uz)
    values = tuple(apply(basis[index], fields[index]) for index in range(3))
    weight = axis.w[:, None, None] * axis.w[None, :, None] * axis.w[None, None, :]
    convection_oracle = []
    for component in range(3):
        advective = np.zeros_like(values[component])
        conservative = np.zeros_like(fields[component])
        for direction in range(3):
            trial = list(basis[component])
            trial[direction] = derivative[component][direction]
            advective += values[direction] * apply(tuple(trial), fields[component])
            test = list(basis[component])
            test[direction] = derivative[component][direction]
            conservative += moments(
                tuple(test), weight * values[direction] * values[component]
            )
        convection_oracle.append(
            0.5 * (moments(basis[component], weight * advective) - conservative)
        )

    mass_oracle = (
        apply((axis.mass0, axis.mass1, axis.mass1), ux),
        apply((axis.mass1, axis.mass0, axis.mass1), uy),
        apply((axis.mass1, axis.mass1, axis.mass0), uz),
    )
    viscous_oracle = (
        apply((axis.stiffness0, axis.mass1, axis.mass1), ux)
        + apply((axis.mass0, operator.tangential_factor, axis.mass1), ux)
        + apply((axis.mass0, axis.mass1, operator.tangential_factor), ux),
        apply((operator.tangential_factor, axis.mass0, axis.mass1), uy)
        + apply((axis.mass1, axis.stiffness0, axis.mass1), uy)
        + apply((axis.mass1, axis.mass0, operator.tangential_factor), uy),
        apply((operator.tangential_factor, axis.mass1, axis.mass0), uz)
        + apply((axis.mass1, operator.tangential_factor, axis.mass0), uz)
        + apply((axis.mass1, axis.mass1, axis.stiffness0), uz),
    )

    def relative(actual, expected):
        return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1.0e-300))

    mass_error = max(
        relative(actual, expected)
        for actual, expected in zip(
            (channels.mass.ux, channels.mass.uy, channels.mass.uz), mass_oracle
        )
    )
    viscous_error = max(
        relative(actual, expected)
        for actual, expected in zip(
            (channels.viscous.ux, channels.viscous.uy, channels.viscous.uz),
            viscous_oracle,
        )
    )
    convection_error = max(
        relative(actual, expected)
        for actual, expected in zip(
            (
                channels.convection.ux,
                channels.convection.uy,
                channels.convection.uz,
            ),
            convection_oracle,
        )
    )
    identity = np.eye(axis.derivative_size)
    divergence_oracle = apply((axis.incidence, identity, identity), ux)
    divergence_oracle += apply((identity, axis.incidence, identity), uy)
    divergence_oracle += apply((identity, identity, axis.incidence), uz)
    divergence_error = relative(channels.divergence, divergence_oracle)
    energy = operator.convective_energy_production(ux, uy, uz)
    energy_scale = sum(
        np.linalg.norm(field) * np.linalg.norm(moment)
        for field, moment in zip(
            fields,
            (
                channels.convection.ux,
                channels.convection.uy,
                channels.convection.uz,
            ),
        )
    )
    storage = operator.storage_record()
    workspace = operator.convection_workspace_record()
    return {
        "degree": 4,
        "npts": 7,
        "mass_relative_error": mass_error,
        "viscous_relative_error": viscous_error,
        "convection_relative_error": convection_error,
        "divergence_relative_error": divergence_error,
        "convective_energy_relative_residual": abs(energy)
        / max(energy_scale, 1.0e-300),
        "stored_factor_entries": storage["stored_factor_entries"],
        "maximum_factor_bandwidth": storage["maximum_factor_bandwidth"],
        "velocity_states": storage["velocity_states"],
        "pressure_states": storage["pressure_states"],
        "quadrature_slabs": workspace["quadrature_slabs"],
        "maximum_scalar_slab_entries": workspace["maximum_scalar_slab_entries"],
        "global_scalar_volume_entries": workspace["global_scalar_volume_entries"],
    }


def build_record() -> dict:
    source = Path(__file__).resolve()
    operator_source = source.with_name("tensor_compatible_flow.py")
    legacy_source = source.with_name("bmf_hdiv_ns2d.py")
    return {
        "schema": SCHEMA,
        "source_sha256": _sha256(source),
        "operator_source_sha256": _sha256(operator_source),
        "legacy_oracle_source_sha256": _sha256(legacy_source),
        "residual_tolerance": RESIDUAL_TOLERANCE,
        "traction_tolerance": TRACTION_TOLERANCE,
        "degree_records": [_degree_record(degree) for degree in DEGREES],
        "three_dimensional_smoke": _three_dimensional_record(),
        "polynomial_wall_traction": _polynomial_traction_record(),
        "classification": {
            "linear_and_direct_weak_equivalence": "verified_current",
            "historical_cavity_convection": "implementation_defect",
            "repair": "accumulate direct weak dual moments before normal-trace restriction",
        },
    }


def verify_record(record: dict) -> None:
    source = Path(__file__).resolve()
    if record.get("schema") != SCHEMA:
        raise ValueError("tensor-compatible equivalence schema mismatch")
    if record.get("source_sha256") != _sha256(source):
        raise ValueError("tensor-compatible verifier hash mismatch")
    if record.get("operator_source_sha256") != _sha256(
        source.with_name("tensor_compatible_flow.py")
    ):
        raise ValueError("tensor-compatible operator hash mismatch")
    if record.get("legacy_oracle_source_sha256") != _sha256(
        source.with_name("bmf_hdiv_ns2d.py")
    ):
        raise ValueError("legacy oracle hash mismatch")
    rows = record.get("degree_records")
    if not isinstance(rows, list) or [row.get("degree") for row in rows] != list(
        DEGREES
    ):
        raise ValueError("tensor-compatible degree coverage mismatch")
    keys = (
        "mass_relative_error",
        "viscous_relative_error",
        "pressure_relative_error",
        "convection_relative_error",
        "wall_load_relative_error",
        "divergence_relative_error",
        "convective_energy_relative_residual",
    )
    for row in rows:
        for key in keys:
            value = float(row.get(key, math.nan))
            if not math.isfinite(value) or value >= RESIDUAL_TOLERANCE:
                raise AssertionError(f"tensor-compatible residual gate failed: {key}")
        if float(row["historical_reduced_convection_relative_difference"]) <= 1.0e-6:
            raise AssertionError("historical convection defect was not detected")
        if int(row["maximum_factor_bandwidth"]) > int(row["degree"]):
            raise AssertionError("one-dimensional factor bandwidth gate failed")
    smoke = record.get("three_dimensional_smoke", {})
    for key in (
        "mass_relative_error",
        "viscous_relative_error",
        "convection_relative_error",
        "divergence_relative_error",
        "convective_energy_relative_residual",
    ):
        value = float(smoke.get(key, math.nan))
        if not math.isfinite(value) or value >= RESIDUAL_TOLERANCE:
            raise AssertionError(f"three-dimensional residual gate failed: {key}")
    if int(smoke.get("maximum_factor_bandwidth", -1)) > int(smoke.get("degree", -2)):
        raise AssertionError("three-dimensional factor bandwidth gate failed")
    if int(smoke.get("quadrature_slabs", 0)) <= 1:
        raise AssertionError("three-dimensional convection is not slab batched")
    if int(smoke.get("maximum_scalar_slab_entries", 0)) >= int(
        smoke.get("global_scalar_volume_entries", 0)
    ):
        raise AssertionError("three-dimensional slab workspace gate failed")
    traction = record.get("polynomial_wall_traction", {})
    if float(traction.get("maximum_absolute_traction_error", math.nan)) >= TRACTION_TOLERANCE:
        raise AssertionError("native polynomial traction gate failed")
    if float(traction.get("maximum_absolute_normal_flux", math.nan)) >= 2.0e-13:
        raise AssertionError("native no-penetration flux gate failed")
    classification = record.get("classification", {})
    if classification.get("historical_cavity_convection") != "implementation_defect":
        raise ValueError("historical convection classification mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    ensure_generated_dirs()
    if args.run:
        if OUTPUT.exists() and not args.replace:
            raise FileExistsError(f"refusing to overwrite {OUTPUT}")
        record = build_record()
        verify_record(record)
        OUTPUT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        print(f"wrote and verified {OUTPUT}")
        return 0
    if args.verify:
        record = json.loads(OUTPUT.read_text())
        verify_record(record)
        print("tensor-compatible flow equivalence: ALL PASS")
        return 0
    parser.error("select --run or --verify")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
