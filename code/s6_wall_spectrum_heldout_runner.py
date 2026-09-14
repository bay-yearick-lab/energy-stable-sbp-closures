#!/usr/bin/env python3
"""Execute the frozen held-out matrix through exact RK4 step maps.

Each acoustic generator is linear and autonomous because the rigid-wall
velocity data vanish identically.  One classical RK4 step is therefore the
fourth-degree stability polynomial in the enforced generator.  Integer powers
of that step map reproduce the declared RK4 march while evaluating all six
packet shapes for a method and horizon together.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import boundary_spline_analysis as ba
import s6_wall_spectrum_heldout as heldout


def generator(method: str, npts: int) -> dict[str, Any]:
    """Assemble the enforced linear generator and scoring reconstruction."""
    if method == "BMF6":
        operator = ba.build_operator(ba.scheme_kind(method), heldout.ORDER, npts)
        staggered = operator.staggered_operators()
        npres = staggered.x.size
        nvel = staggered.eta.size
        zero_p = np.zeros((npres, npres))
        zero_v = np.zeros((nvel, nvel))
        full = np.block([[zero_p, -staggered.d01], [-staggered.d10, zero_v]])
        constraint, lifting = ba._horth_wave_lifting(
            staggered.h1, nvel, npres, npres + nvel
        )
        enforced = (np.eye(npres + nvel) - lifting @ constraint) @ full
        eta_to_x = staggered.bhat_x @ np.linalg.inv(staggered.bhat_eta)
        return {
            "matrix": enforced,
            "projection": np.eye(npres + nvel) - lifting @ constraint,
            "x": staggered.x,
            "velocity_x": staggered.eta,
            "pressure_size": npres,
            "velocity_reconstruction": eta_to_x,
            "h": min(float(np.min(np.diff(staggered.x))), float(np.min(np.diff(staggered.eta)))),
        }
    if method == "BMG6":
        operator = ba.build_operator(ba.scheme_kind(method), heldout.ORDER, npts)
        derivative = operator.matrix
        zero = np.zeros((npts, npts))
        full = np.block([[zero, -derivative], [-derivative, zero]])
        metric = ba.bmg_nodal_operator(npts, heldout.ORDER)[2]
        constraint, lifting = ba._horth_wave_lifting(metric, npts, npts, 2 * npts)
        enforced = (np.eye(2 * npts) - lifting @ constraint) @ full
        return {
            "matrix": enforced,
            "projection": np.eye(2 * npts) - lifting @ constraint,
            "x": operator.x,
            "velocity_x": operator.x,
            "pressure_size": npts,
            "velocity_reconstruction": np.eye(npts),
            "h": float(operator.x[1] - operator.x[0]),
        }
    if method == "SBP6":
        operator = ba.build_operator(ba.scheme_kind(method), heldout.ORDER, npts)
    else:
        operator = heldout.dense_operator(method, npts)
    derivative = np.asarray(operator.matrix, dtype=float)
    matrix = np.block([
        [np.zeros((npts, npts)), -derivative[:, 1:-1]],
        [-derivative[1:-1, :], np.zeros((npts - 2, npts - 2))],
    ])
    return {
        "matrix": matrix,
        "projection": np.eye(2 * npts - 2),
        "x": np.asarray(operator.x, dtype=float),
        "velocity_x": np.asarray(operator.x, dtype=float)[1:-1],
        "pressure_size": npts,
        "velocity_reconstruction": None,
        "h": float(operator.x[1] - operator.x[0]),
    }


def rk4_step_map(matrix: np.ndarray, step: float) -> np.ndarray:
    """Return the classical RK4 stability polynomial of one linear step."""
    identity = np.eye(matrix.shape[0])
    scaled = step * matrix
    squared = scaled @ scaled
    return identity + scaled + 0.5 * squared + (squared @ scaled) / 6.0 + (squared @ squared) / 24.0


def initial_states(
    method_data: dict[str, Any], packet_specs: list[tuple[float, float]]
) -> np.ndarray:
    """Stack initial pressure and velocity states for all packet shapes."""
    columns = []
    pressure_x = method_data["x"]
    velocity_x = method_data["velocity_x"]
    for sigma, carrier in packet_specs:
        pressure, velocity = heldout.exact_pair(sigma, carrier * np.pi)
        columns.append(np.concatenate([pressure(pressure_x, 0.0), velocity(velocity_x, 0.0)]))
    return np.column_stack(columns)


def unpack_state(
    method: str, method_data: dict[str, Any], state: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return pressure and velocity on the pressure scoring grid."""
    npts = method_data["pressure_size"]
    pressure = state[:npts]
    if method == "BMF6":
        velocity = method_data["velocity_reconstruction"] @ state[npts:]
    elif method == "BMG6":
        velocity = state[npts:]
    else:
        velocity = np.zeros(npts)
        velocity[1:-1] = state[npts:]
    return pressure, velocity


def march_group(
    method: str,
    npts: int,
    packet_specs: list[tuple[float, float]],
    encounters: int,
) -> list[dict[str, float]]:
    """March all packet shapes for one method and horizon through RK4 maps."""
    method_data = generator(method, npts)
    matrix = method_data["matrix"]
    initial = initial_states(method_data, packet_specs)
    nsteps = int(np.ceil(float(encounters) / (heldout.CFL * method_data["h"])))
    step = float(encounters) / nsteps
    one_step = rk4_step_map(matrix, step)
    combined_step = method_data["projection"] @ one_step
    final = np.linalg.matrix_power(combined_step, nsteps) @ initial
    rows = []
    for column, (sigma, carrier) in enumerate(packet_specs):
        pressure, velocity = unpack_state(method, method_data, final[:, column])
        exact_pressure, exact_velocity = heldout.exact_pair(sigma, carrier * np.pi)
        rows.append(heldout.score_fields(
            np.asarray(method_data["x"], dtype=float),
            pressure,
            velocity,
            exact_pressure,
            exact_velocity,
            float(encounters),
            carrier * np.pi,
            sigma,
        ))
    return rows


def matrix_power_state(
    method: str,
    npts: int,
    sigma: float,
    carrier: float,
    time_value: float,
) -> tuple[dict[str, Any], np.ndarray, int]:
    """Return one RK4 matrix-power state for implementation-faithfulness tests."""
    data = generator(method, npts)
    initial = initial_states(data, [(sigma, carrier)])[:, 0]
    nsteps = int(np.ceil(time_value / (heldout.CFL * data["h"])))
    step = time_value / nsteps
    one_step = rk4_step_map(data["matrix"], step)
    combined_step = data["projection"] @ one_step
    final = np.linalg.matrix_power(combined_step, nsteps) @ initial
    return data, final, nsteps


def run(prediction_path: Path, output_path: Path) -> dict[str, Any]:
    """Execute the complete frozen matrix and bind raw fields to predictions."""
    if output_path.exists():
        raise FileExistsError(output_path)
    prediction = json.loads(prediction_path.read_text())
    packet_specs = [
        (sigma, carrier)
        for sigma in heldout.HOLDOUT_SIGMAS
        for carrier in heldout.HOLDOUT_K0_OVER_PI
    ]
    by_id: dict[str, dict[str, float]] = {}
    numerical_contract = []
    for method in heldout.METHODS:
        data = generator(method, heldout.NPTS)
        for encounters in heldout.HOLDOUT_ENCOUNTERS:
            rows = march_group(method, heldout.NPTS, packet_specs, encounters)
            for (sigma, carrier), metrics in zip(packet_specs, rows, strict=True):
                identifier = heldout.case_id(sigma, carrier, encounters, method)
                by_id[identifier] = metrics
            numerical_contract.append({
                "method": method,
                "encounters": encounters,
                "rk4_step_count": int(np.ceil(encounters / (heldout.CFL * data["h"]))),
                "step_rule": "t_end / ceil(t_end / (cfl * h_route))",
            })
            print(f"{method} encounters={encounters}: six packet shapes complete", flush=True)
    ordered = [
        {"case_id": item["case_id"], "metrics": by_id[item["case_id"]]}
        for item in prediction["predictions"]
    ]
    raw = {
        "schema": heldout.SCHEMA,
        "artifact_state": "heldout_raw_once",
        "prediction_sha256": heldout.sha256(prediction_path),
        "prediction_path": str(prediction_path),
        "runner_sha256": heldout.sha256(Path(__file__)),
        "realization": "classical RK4 stability-polynomial step map raised to the declared step count",
        "numerical_contract": numerical_contract,
        "results": ordered,
    }
    output_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, default=heldout.PREDICTION_PATH)
    parser.add_argument("--output", type=Path, default=heldout.RAW_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record = run(args.prediction, args.output)
    print(f"wrote {args.output} with {len(record['results'])} primary results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
