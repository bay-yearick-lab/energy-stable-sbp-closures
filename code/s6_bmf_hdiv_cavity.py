#!/usr/bin/env python3
"""Physical route-C benchmark: the Re=100 lid-driven square cavity.

The adopted divergence-conforming BMF velocity-pressure pair is advanced with
normal velocity imposed strongly, tangential velocity imposed by the same
symmetric Nitsche form used in the route-C certificate, exact-quadrature
skew/conservative convection, and a BDF2/AB2 saddle solve.  Centerline values
are scored against Tables I and II of the cavity reference tables, JCP 48 (1982),
387--411, DOI 10.1016/0021-9991(82)90058-4.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import lu_factor, lu_solve

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))

from bmf_hdiv_ns2d import BMFHDivNS2D
from boundary_spline_analysis import (
    bspline_evaluation_matrix,
    derivative_space_knots,
    open_uniform_knots,
)
from project_paths import RESULTSDIR, ensure_generated_dirs


SCHEMA = "bmf-hdiv-route-c-cavity-v1"
DEFAULT_OUTPUT = RESULTSDIR / "bmf_hdiv_cavity_summary.json"
THREAD_ENV_NAMES = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
)

# Original Re=100 columns, Tables I and II, stored in ascending coordinate
# order. Table II's physical component is v even though the scan's heading
# repeats "u-Velocity".
CAVITY_REFERENCE100_U_CENTERLINE = (
    (0.0000, 0.00000), (0.0547, -0.03717), (0.0625, -0.04192),
    (0.0703, -0.04775), (0.1016, -0.06434), (0.1719, -0.10150),
    (0.2813, -0.15662), (0.4531, -0.21090), (0.5000, -0.20581),
    (0.6172, -0.13641), (0.7344, 0.00332), (0.8516, 0.23151),
    (0.9531, 0.68717), (0.9609, 0.73722), (0.9688, 0.78871),
    (0.9766, 0.84123), (1.0000, 1.00000),
)
CAVITY_REFERENCE100_V_CENTERLINE = (
    (0.0000, 0.00000), (0.0625, 0.09233), (0.0703, 0.10091),
    (0.0781, 0.10890), (0.0938, 0.12317), (0.1563, 0.16077),
    (0.2266, 0.17507), (0.2344, 0.17527), (0.5000, 0.05454),
    (0.8047, -0.24533), (0.8594, -0.22445), (0.9063, -0.16914),
    (0.9453, -0.10313), (0.9531, -0.08864), (0.9609, -0.07391),
    (0.9688, -0.05906), (1.0000, 0.00000),
)


def production_config() -> dict[str, Any]:
    return {
        "reynolds": 100.0,
        "degree": 4,
        "grids": [11, 15, 19, 23],
        "dt": 0.02,
        "t_end": 60.0,
        "minimum_time": 10.0,
        "steady_relative_rate_tolerance": 2.0e-7,
        "penalty_gamma": 64.0,
        "time_integrator": "backward-Euler/explicit first step; BDF2/AB2 thereafter",
        "boundary_conditions": "strong no penetration; symmetric-Nitsche tangential lid u=1 at z=1 and zero elsewhere",
        "convection": "exact-quadrature conservative/skew Galerkin form",
    }


def smoke_config() -> dict[str, Any]:
    config = production_config()
    config.update({
        "grids": [7], "t_end": 0.04, "minimum_time": 1.0,
        "steady_relative_rate_tolerance": 0.0,
    })
    return config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def timing_stamp() -> dict[str, Any]:
    cpu = platform.processor()
    if not cpu:
        try:
            cpu = next(
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            )
        except (OSError, StopIteration):
            cpu = "unknown"
    return {
        "metric": "process_time_seconds",
        "cpu": cpu,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "thread_env": {name: os.environ.get(name, "unset") for name in THREAD_ENV_NAMES},
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
    }


def _trap_weights(nodes: np.ndarray) -> np.ndarray:
    x = np.asarray(nodes, dtype=float)
    weights = np.empty_like(x)
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    return weights


def centerline_score(u_values: np.ndarray, v_values: np.ndarray) -> dict[str, float]:
    u = np.asarray(u_values, dtype=float)
    v = np.asarray(v_values, dtype=float)
    uref = np.asarray([value for _coordinate, value in CAVITY_REFERENCE100_U_CENTERLINE])
    vref = np.asarray([value for _coordinate, value in CAVITY_REFERENCE100_V_CENTERLINE])
    if u.shape != uref.shape or v.shape != vref.shape:
        raise ValueError("centerline arrays do not match the reference tables")
    yu = np.asarray([coordinate for coordinate, _value in CAVITY_REFERENCE100_U_CENTERLINE])
    xv = np.asarray([coordinate for coordinate, _value in CAVITY_REFERENCE100_V_CENTERLINE])
    wu, wv = _trap_weights(yu), _trap_weights(xv)

    def score(mask_u: np.ndarray, mask_v: np.ndarray) -> float:
        numerator = float(
            np.sum(wu[mask_u] * (u[mask_u] - uref[mask_u]) ** 2)
            + np.sum(wv[mask_v] * (v[mask_v] - vref[mask_v]) ** 2)
        )
        denominator = float(
            np.sum(wu[mask_u] * uref[mask_u] ** 2)
            + np.sum(wv[mask_v] * vref[mask_v] ** 2)
        )
        return math.sqrt(numerator / max(denominator, 1.0e-300))

    all_u = np.ones(u.size, dtype=bool)
    all_v = np.ones(v.size, dtype=bool)
    interior_u = (yu > 0.0) & (yu < 1.0)
    interior_v = (xv > 0.0) & (xv < 1.0)
    return {
        "combined_relative_l2": score(all_u, all_v),
        "interior_combined_relative_l2": score(interior_u, interior_v),
        "maximum_absolute_error": float(
            max(np.max(np.abs(u - uref)), np.max(np.abs(v - vref)))
        ),
    }


@dataclass(frozen=True)
class CavityState:
    velocity: np.ndarray
    previous_velocity: np.ndarray
    previous_convection: np.ndarray
    dual_pressure: np.ndarray
    step_index: int
    time: float


class RouteCCavityStepper:
    """Prefactored BDF2/AB2 solver for the adopted route-C pair."""

    def __init__(self, npts: int, degree: int, reynolds: float, dt: float,
                 penalty_gamma: float | None = None) -> None:
        self.formulation = BMFHDivNS2D(
            npts=npts, degree=degree, penalty_gamma=penalty_gamma
        )
        self.reynolds = float(reynolds)
        self.viscosity = 1.0 / self.reynolds
        self.dt = float(dt)
        if not np.isfinite(self.reynolds) or self.reynolds <= 0.0:
            raise ValueError("reynolds must be finite and positive")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.mass = self.formulation.velocity_mass
        self.viscous = self.formulation.viscous_matrix
        self.divergence = self.formulation.divergence_matrix
        self.nv = self.mass.shape[0]
        self.nq = self.divergence.shape[0]
        one_dual = self.formulation.core.h1 @ np.ones(
            self.formulation.core.derivative_size
        )
        self.constant_dual = np.outer(one_dual, one_dual).ravel()
        self.constant_dual /= np.linalg.norm(self.constant_dual)
        zero = lambda x, z: np.zeros(np.broadcast_shapes(np.shape(x), np.shape(z)))
        lid = lambda x, z: np.broadcast_to(
            np.isclose(np.asarray(z, dtype=float), self.formulation.length).astype(float),
            np.broadcast_shapes(np.shape(x), np.shape(z)),
        )
        self.boundary_load = self.formulation._assemble_viscous_rhs(
            zero, zero, lid, zero
        )
        self._be_factor = lu_factor(self._saddle_matrix(1.0 / self.dt), check_finite=False)
        self._bdf_factor = lu_factor(self._saddle_matrix(1.5 / self.dt), check_finite=False)

    def _saddle_matrix(self, alpha: float) -> np.ndarray:
        return np.block([
            [alpha * self.mass + self.viscosity * self.viscous,
             self.divergence.T, np.zeros((self.nv, 1))],
            [self.divergence, np.zeros((self.nq, self.nq)),
             self.constant_dual[:, None]],
            [np.zeros((1, self.nv)), self.constant_dual[None, :],
             np.zeros((1, 1))],
        ])

    def _convection_active(self, velocity: np.ndarray) -> np.ndarray:
        ux, uz = self.formulation._unpack_active(velocity)
        cux, cuz, _ledger = self.formulation.skew_convection(ux, uz)
        return self.formulation._pack_active(cux, cuz)

    def initial_state(self) -> CavityState:
        velocity = np.zeros(self.nv)
        return CavityState(
            velocity=velocity,
            previous_velocity=velocity.copy(),
            previous_convection=np.zeros(self.nv),
            dual_pressure=np.zeros(self.nq),
            step_index=0,
            time=0.0,
        )

    def step(self, state: CavityState) -> CavityState:
        velocity = np.asarray(state.velocity, dtype=float)
        convection = self._convection_active(velocity)
        if state.step_index == 0:
            history = velocity / self.dt - convection
            factor = self._be_factor
        else:
            history = (
                (2.0 * velocity - 0.5 * state.previous_velocity) / self.dt
                - (2.0 * convection - state.previous_convection)
            )
            factor = self._bdf_factor
        rhs_velocity = self.mass @ history + self.viscosity * self.boundary_load
        rhs = np.concatenate([rhs_velocity, np.zeros(self.nq + 1)])
        answer = lu_solve(factor, rhs, check_finite=False)
        if not np.all(np.isfinite(answer)):
            raise FloatingPointError("non-finite route-C cavity state")
        return CavityState(
            velocity=answer[:self.nv],
            previous_velocity=velocity.copy(),
            previous_convection=convection,
            dual_pressure=answer[self.nv:self.nv + self.nq],
            step_index=state.step_index + 1,
            time=state.time + self.dt,
        )

    def velocity_rate(self, new: CavityState, old: CavityState) -> float:
        delta = new.velocity - old.velocity
        numerator = math.sqrt(max(float(delta @ self.mass @ delta), 0.0)) / self.dt
        denominator = max(
            math.sqrt(max(float(new.velocity @ self.mass @ new.velocity), 0.0)),
            1.0,
        )
        return numerator / denominator

    def centerlines(self, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ux, uz = self.formulation._unpack_active(velocity)
        p = self.formulation.degree
        n = self.formulation.npts
        knots0 = open_uniform_knots(n, p, self.formulation.length)
        knots1 = derivative_space_knots(knots0)
        b0 = bspline_evaluation_matrix(knots0, p, self.formulation.axis.x0)
        b1 = bspline_evaluation_matrix(knots1, p - 1, self.formulation.axis.x1)
        inv0, inv1 = np.linalg.inv(b0), np.linalg.inv(b1)
        yu = np.asarray([coordinate for coordinate, _value in CAVITY_REFERENCE100_U_CENTERLINE])
        xv = np.asarray([coordinate for coordinate, _value in CAVITY_REFERENCE100_V_CENTERLINE])
        e0_mid = bspline_evaluation_matrix(knots0, p, np.array([0.5])) @ inv0
        e1_y = bspline_evaluation_matrix(knots1, p - 1, yu) @ inv1
        e1_x = bspline_evaluation_matrix(knots1, p - 1, xv) @ inv1
        e0_mid_y = bspline_evaluation_matrix(knots0, p, np.array([0.5])) @ inv0
        u_values = (e0_mid @ ux @ e1_y.T).ravel()
        v_values = (e1_x @ uz @ e0_mid_y.T).ravel()
        return u_values, v_values

    def diagnostics(self, state: CavityState, steady_rate: float) -> dict[str, Any]:
        ux, uz = self.formulation._unpack_active(state.velocity)
        divergence = self.formulation.core.divergence(ux, uz)
        divergence_l2 = self.formulation.core.pressure_norm(divergence)
        cux, cuz, convection_ledger = self.formulation.skew_convection(ux, uz)
        convection = self.formulation._pack_active(cux, cuz)
        convective_dual = self.mass @ convection
        viscous_dual = self.viscosity * (self.viscous @ state.velocity)
        pressure_dual = self.divergence.T @ state.dual_pressure
        wall_dual = self.viscosity * self.boundary_load
        spatial_dual = convective_dual + viscous_dual + pressure_dual - wall_dual
        energy_terms = [
            float(state.velocity @ term)
            for term in (convective_dual, viscous_dual, pressure_dual, -wall_dual)
        ]
        energy_residual = abs(sum(energy_terms))
        energy_scale = max(sum(abs(value) for value in energy_terms), 1.0e-300)
        u_values, v_values = self.centerlines(state.velocity)
        score = centerline_score(u_values, v_values)
        return {
            "npts": self.formulation.npts,
            "elements": self.formulation.npts - self.formulation.degree,
            "degree": self.formulation.degree,
            "steps": state.step_index,
            "final_time": state.time,
            "steady_relative_velocity_rate": steady_rate,
            "divergence_l2": divergence_l2,
            "normal_trace_max": self.formulation.core.normal_trace_max(ux, uz),
            "kinetic_energy": 0.5 * float(state.velocity @ self.mass @ state.velocity),
            "relative_steady_energy_residual": energy_residual / energy_scale,
            "relative_convective_energy_production": convection_ledger[
                "relative_kinetic_energy_production"
            ],
            "dual_pressure_gauge_residual": abs(
                float(self.constant_dual @ state.dual_pressure)
            ),
            "centerline": score,
            "u_centerline": [float(value) for value in u_values],
            "v_centerline": [float(value) for value in v_values],
        }


def run_case(config: dict[str, Any], npts: int) -> dict[str, Any]:
    stepper = RouteCCavityStepper(
        npts=npts,
        degree=int(config["degree"]),
        reynolds=float(config["reynolds"]),
        dt=float(config["dt"]),
        penalty_gamma=float(config["penalty_gamma"]),
    )
    state = stepper.initial_state()
    maximum_steps = int(math.ceil(float(config["t_end"]) / stepper.dt))
    minimum_steps = int(math.ceil(float(config["minimum_time"]) / stepper.dt))
    tolerance = float(config["steady_relative_rate_tolerance"])
    steady_rate = math.inf
    cpu_start = time.process_time()
    converged = False
    for _ in range(maximum_steps):
        previous = state
        state = stepper.step(previous)
        steady_rate = stepper.velocity_rate(state, previous)
        if state.step_index >= minimum_steps and tolerance > 0.0 and steady_rate <= tolerance:
            converged = True
            break
    cpu_seconds = time.process_time() - cpu_start
    record = stepper.diagnostics(state, steady_rate)
    record.update({"converged": converged, "cpu_seconds": cpu_seconds})
    return record


def _worker(payload: tuple[dict[str, Any], int]) -> dict[str, Any]:
    config, npts = payload
    record = run_case(config, npts)
    print(
        f"route-C cavity N={npts}: t={record['final_time']:.2f}, "
        f"rate={record['steady_relative_velocity_rate']:.3e}, "
        f"reference={record['centerline']['interior_combined_relative_l2']:.3e}",
        flush=True,
    )
    return record


def _validate_timing(stamp: dict[str, Any], *, production: bool) -> None:
    required = {"metric", "cpu", "platform", "python", "numpy", "thread_env", "nice"}
    if not isinstance(stamp, dict) or set(stamp) != required:
        raise ValueError("route-C cavity timing stamp is incomplete")
    if stamp["metric"] != "process_time_seconds":
        raise ValueError("route-C cavity timing metric mismatch")
    if production:
        if stamp["nice"] != 19:
            raise ValueError("route-C cavity production run was not nice 19")
        if any(stamp["thread_env"].get(name) != "1" for name in THREAD_ENV_NAMES):
            raise ValueError("route-C cavity production run was not single-thread pinned")


def build_record(config: dict[str, Any], runs: list[dict[str, Any]],
                 stamp: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "smoke": smoke,
        "config": config,
        "reference": {
            "authors": "multigrid reference solution",
            "title": "High-Re Solutions for Incompressible Flow Using the Navier-Stokes Equations and a Multigrid Method",
            "journal": "Journal of Computational Physics 48 (1982) 387-411",
            "doi": "10.1016/0021-9991(82)90058-4",
            "tables": "Tables I and II, Re=100 columns",
            "u_centerline": [list(value) for value in CAVITY_REFERENCE100_U_CENTERLINE],
            "v_centerline": [list(value) for value in CAVITY_REFERENCE100_V_CENTERLINE],
        },
        "provenance": {
            "source": "code/s6_bmf_hdiv_cavity.py",
            "source_sha256": _sha256(Path(__file__).resolve()),
            "config_sha256": _canonical_sha256(config),
        },
        "timing": stamp,
        "runs": sorted(runs, key=lambda run: run["npts"]),
    }


def smoke_record() -> dict[str, Any]:
    config = smoke_config()
    runs = [run_case(config, config["grids"][0])]
    return build_record(config, runs, timing_stamp(), smoke=True)


def verify_record(record: dict[str, Any], *, require_production: bool = True) -> None:
    if record.get("schema") != SCHEMA:
        raise ValueError("route-C cavity schema mismatch")
    smoke = record.get("smoke") is True
    if require_production and smoke:
        raise ValueError("route-C cavity smoke record is not production evidence")
    expected = smoke_config() if smoke else production_config()
    if record.get("config") != expected:
        raise ValueError("route-C cavity configuration mismatch")
    provenance = record.get("provenance", {})
    if provenance.get("source_sha256") != _sha256(Path(__file__).resolve()):
        raise ValueError("route-C cavity source hash mismatch")
    if provenance.get("config_sha256") != _canonical_sha256(expected):
        raise ValueError("route-C cavity configuration hash mismatch")
    _validate_timing(record.get("timing"), production=not smoke)
    reference = record.get("reference", {})
    if reference.get("doi") != "10.1016/0021-9991(82)90058-4":
        raise ValueError("route-C cavity reference DOI mismatch")
    if reference.get("u_centerline") != [list(value) for value in CAVITY_REFERENCE100_U_CENTERLINE]:
        raise ValueError("route-C cavity u reference mismatch")
    if reference.get("v_centerline") != [list(value) for value in CAVITY_REFERENCE100_V_CENTERLINE]:
        raise ValueError("route-C cavity v reference mismatch")
    runs = record.get("runs")
    if not isinstance(runs, list) or [run.get("npts") for run in runs] != expected["grids"]:
        raise ValueError("route-C cavity grid coverage mismatch")
    for run in runs:
        for key in (
            "steady_relative_velocity_rate", "divergence_l2", "normal_trace_max",
            "kinetic_energy", "relative_steady_energy_residual",
            "relative_convective_energy_production",
            "dual_pressure_gauge_residual", "cpu_seconds",
        ):
            value = run.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"route-C cavity non-finite {key}")
        if run["divergence_l2"] >= 2.0e-10:
            raise AssertionError("route-C cavity divergence gate failed")
        if run["normal_trace_max"] >= 2.0e-13:
            raise AssertionError("route-C cavity normal-trace gate failed")
        if run["relative_convective_energy_production"] >= 2.0e-12:
            raise AssertionError("route-C cavity convective-energy gate failed")
        if run["dual_pressure_gauge_residual"] >= 2.0e-10:
            raise AssertionError("route-C cavity pressure-gauge gate failed")
        if len(run.get("u_centerline", [])) != 17 or len(run.get("v_centerline", [])) != 17:
            raise ValueError("route-C cavity centerline coverage mismatch")
        recomputed = centerline_score(run["u_centerline"], run["v_centerline"])
        if run.get("centerline") != recomputed:
            raise ValueError("route-C cavity centerline score arithmetic mismatch")
    if not smoke:
        if not all(run.get("converged") is True for run in runs):
            raise AssertionError("route-C cavity steady-convergence gate failed")
        errors = [run["centerline"]["interior_combined_relative_l2"] for run in runs]
        if errors[-1] >= errors[0] or errors[-1] >= 0.12:
            raise AssertionError("route-C cavity reference-accuracy gate failed")


def _load_checkpoint(path: Path | None, config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    source_hash = _sha256(Path(__file__).resolve())
    config_hash = _canonical_sha256(config)
    completed: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if row.get("source_sha256") != source_hash or row.get("config_sha256") != config_hash:
            raise ValueError(f"route-C cavity checkpoint provenance mismatch on line {line_number}")
        run = row.get("run")
        if not isinstance(run, dict) or run.get("npts") not in config["grids"]:
            raise ValueError(f"route-C cavity checkpoint case mismatch on line {line_number}")
        npts = int(run["npts"])
        if npts in completed:
            raise ValueError(f"duplicate route-C cavity checkpoint case N={npts}")
        completed[npts] = run
    return completed


def _append_checkpoint(path: Path | None, config: dict[str, Any], run: dict[str, Any]) -> None:
    if path is None:
        return
    row = {
        "source_sha256": _sha256(Path(__file__).resolve()),
        "config_sha256": _canonical_sha256(config),
        "run": run,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_all(config: dict[str, Any], workers: int,
            checkpoint: Path | None = None) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    completed = _load_checkpoint(checkpoint, config)
    payloads = [
        (config, npts) for npts in config["grids"] if npts not in completed
    ]
    if workers == 1:
        for payload in payloads:
            run = _worker(payload)
            _append_checkpoint(checkpoint, config, run)
            completed[int(run["npts"])] = run
    elif payloads:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as executor:
            futures = {executor.submit(_worker, payload): payload[1] for payload in payloads}
            for future in as_completed(futures):
                run = future.result()
                _append_checkpoint(checkpoint, config, run)
                completed[int(run["npts"])] = run
    return [completed[npts] for npts in config["grids"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    ensure_generated_dirs()
    if args.verify:
        record = json.loads(args.output.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        verify_record(record)
        print("route-C cavity verify: ALL PASS")
        return 0
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite route-C cavity record: {args.output}")
    config = smoke_config() if args.smoke else production_config()
    stamp = timing_stamp()
    _validate_timing(stamp, production=not args.smoke)
    checkpoint = args.checkpoint or Path(str(args.output) + ".partial.jsonl")
    runs = run_all(config, args.workers, checkpoint)
    record = build_record(config, runs, stamp, smoke=args.smoke)
    verify_record(record, require_production=not args.smoke)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(f"route-C cavity wrote and verified: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
