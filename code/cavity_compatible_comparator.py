#!/usr/bin/env python3
"""Frozen compatible-method comparison for the Re=100 lid-driven cavity."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

for _name in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy.linalg import lu_factor, lu_solve

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))
from project_paths import FIGDIR, RESULTSDIR, ensure_generated_dirs, save_figure
from s6_bmf_hdiv_cavity import (
    CAVITY_REFERENCE100_U_CENTERLINE, CAVITY_REFERENCE100_V_CENTERLINE, RouteCCavityStepper,
    centerline_score, timing_stamp,
)

SCHEMA = "cavity-compatible-comparator-v1"
GRIDS = ((11, 10), (15, 14), (19, 18))
DT = 0.02
REYNOLDS = 100.0
MINIMUM_TIME = 10.0
MAXIMUM_TIME = 60.0
BASE_STEADY_TOLERANCE = 1.0e-6
SENSITIVITY_STEADY_TOLERANCE = 2.0e-7
PHYSICAL_TOLERANCE_PP = 12.0
SENSITIVITY_TOLERANCE_PP = 0.2
DIVERGENCE_TOLERANCE = 2.0e-10
PRESSURE_GAUGE_TOLERANCE = 2.0e-10
RAW_OUTPUT = RESULTSDIR / "cavity_compatible_comparator_raw.json"
SUMMARY_OUTPUT = RESULTSDIR / "cavity_compatible_comparator_summary.json"
GATE_OUTPUT = RESULTSDIR / "cavity_compatible_comparator_gate.json"


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def frozen_config() -> dict[str, Any]:
    return {
        "reynolds": REYNOLDS,
        "pairs": [
            {"bmf_npts": bmf, "mac_cells": mac,
             "velocity_states": 2 * mac * (mac - 1), "pressure_states": mac * mac}
            for bmf, mac in GRIDS
        ],
        "dt": DT,
        "minimum_time": MINIMUM_TIME,
        "maximum_time": MAXIMUM_TIME,
        "base_steady_tolerance": BASE_STEADY_TOLERANCE,
        "sensitivity_steady_tolerance": SENSITIVITY_STEADY_TOLERANCE,
        "physical_tolerance_percentage_points": PHYSICAL_TOLERANCE_PP,
        "sensitivity_tolerance_percentage_points": SENSITIVITY_TOLERANCE_PP,
        "divergence_tolerance": DIVERGENCE_TOLERANCE,
        "pressure_gauge_tolerance": PRESSURE_GAUGE_TOLERANCE,
        "time_integrator": "backward Euler first step; BDF2/AB2 thereafter",
        "timing_metric": "process_time_seconds",
    }


def peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


@dataclass(frozen=True)
class MACState:
    velocity: np.ndarray
    previous_velocity: np.ndarray
    previous_convection: np.ndarray
    pressure: np.ndarray
    step_index: int
    time: float


class MACCavityStepper:
    """Cell-pressure, face-velocity MAC discretization on a uniform square."""

    def __init__(self, cells: int, reynolds: float, dt: float) -> None:
        if cells < 4:
            raise ValueError("MAC cavity requires at least four cells per axis")
        self.cells = int(cells)
        self.h = 1.0 / self.cells
        self.reynolds = float(reynolds)
        self.viscosity = 1.0 / self.reynolds
        self.dt = float(dt)
        self.u_shape = (self.cells - 1, self.cells)
        self.v_shape = (self.cells, self.cells - 1)
        self.nu = int(np.prod(self.u_shape))
        self.nv_component = int(np.prod(self.v_shape))
        self.velocity_size = self.nu + self.nv_component
        self.pressure_size = self.cells * self.cells
        self.laplacian, self.boundary_load = self._assemble_laplacian()
        self.divergence = self._assemble_divergence()
        self.gradient = -self.divergence.T
        self.constant_pressure = np.ones(self.pressure_size)
        self.constant_pressure /= np.linalg.norm(self.constant_pressure)
        self._be_factor = lu_factor(self._saddle_matrix(1.0 / self.dt), check_finite=False)
        self._bdf_factor = lu_factor(self._saddle_matrix(1.5 / self.dt), check_finite=False)

    def _u_index(self, i: int, j: int) -> int:
        return (i - 1) * self.cells + j

    def _v_index(self, i: int, j: int) -> int:
        return self.nu + i * (self.cells - 1) + (j - 1)

    def _p_index(self, i: int, j: int) -> int:
        return i * self.cells + j

    def _assemble_laplacian(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.cells
        h2 = self.h * self.h
        matrix = np.zeros((self.velocity_size, self.velocity_size))
        load = np.zeros(self.velocity_size)
        for i in range(1, n):
            for j in range(n):
                row = self._u_index(i, j)
                matrix[row, row] = (-2.0 + (-3.0 if j in (0, n - 1) else -2.0)) / h2
                if i > 1:
                    matrix[row, self._u_index(i - 1, j)] += 1.0 / h2
                if i < n - 1:
                    matrix[row, self._u_index(i + 1, j)] += 1.0 / h2
                if j > 0:
                    matrix[row, self._u_index(i, j - 1)] += 1.0 / h2
                if j < n - 1:
                    matrix[row, self._u_index(i, j + 1)] += 1.0 / h2
                if j == n - 1:
                    load[row] = 2.0 / h2
        for i in range(n):
            for j in range(1, n):
                row = self._v_index(i, j)
                matrix[row, row] = ((-3.0 if i in (0, n - 1) else -2.0) - 2.0) / h2
                if i > 0:
                    matrix[row, self._v_index(i - 1, j)] += 1.0 / h2
                if i < n - 1:
                    matrix[row, self._v_index(i + 1, j)] += 1.0 / h2
                if j > 1:
                    matrix[row, self._v_index(i, j - 1)] += 1.0 / h2
                if j < n - 1:
                    matrix[row, self._v_index(i, j + 1)] += 1.0 / h2
        return matrix, load

    def _assemble_divergence(self) -> np.ndarray:
        n = self.cells
        matrix = np.zeros((self.pressure_size, self.velocity_size))
        for i in range(n):
            for j in range(n):
                row = self._p_index(i, j)
                if i + 1 < n:
                    matrix[row, self._u_index(i + 1, j)] += 1.0 / self.h
                if i > 0:
                    matrix[row, self._u_index(i, j)] -= 1.0 / self.h
                if j + 1 < n:
                    matrix[row, self._v_index(i, j + 1)] += 1.0 / self.h
                if j > 0:
                    matrix[row, self._v_index(i, j)] -= 1.0 / self.h
        return matrix

    def _saddle_matrix(self, alpha: float) -> np.ndarray:
        return np.block([
            [alpha * np.eye(self.velocity_size) - self.viscosity * self.laplacian,
             self.gradient, np.zeros((self.velocity_size, 1))],
            [self.divergence, np.zeros((self.pressure_size, self.pressure_size)),
             self.constant_pressure[:, None]],
            [np.zeros((1, self.velocity_size)), self.constant_pressure[None, :],
             np.zeros((1, 1))],
        ])

    def initial_state(self) -> MACState:
        velocity = np.zeros(self.velocity_size)
        return MACState(velocity, velocity.copy(), np.zeros_like(velocity),
                        np.zeros(self.pressure_size), 0, 0.0)

    def _full_velocity(self, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = self.cells
        u = np.zeros((n + 1, n))
        v = np.zeros((n, n + 1))
        u[1:n, :] = velocity[:self.nu].reshape(self.u_shape)
        v[:, 1:n] = velocity[self.nu:].reshape(self.v_shape)
        return u, v

    def convection(self, velocity: np.ndarray) -> np.ndarray:
        n, h = self.cells, self.h
        u, v = self._full_velocity(velocity)
        cu = np.zeros(self.u_shape)
        cv = np.zeros(self.v_shape)
        for i in range(1, n):
            for j in range(n):
                right = (0.5 * (u[i, j] + u[i + 1, j])) ** 2
                left = (0.5 * (u[i - 1, j] + u[i, j])) ** 2
                top = 0.0
                bottom = 0.0
                if j + 1 < n:
                    top = 0.5 * (u[i, j] + u[i, j + 1]) * 0.5 * (v[i - 1, j + 1] + v[i, j + 1])
                if j > 0:
                    bottom = 0.5 * (u[i, j - 1] + u[i, j]) * 0.5 * (v[i - 1, j] + v[i, j])
                cu[i - 1, j] = (right - left + top - bottom) / h
        for i in range(n):
            for j in range(1, n):
                top = (0.5 * (v[i, j] + v[i, j + 1])) ** 2
                bottom = (0.5 * (v[i, j - 1] + v[i, j])) ** 2
                right = 0.0
                left = 0.0
                if i + 1 < n:
                    right = 0.5 * (v[i, j] + v[i + 1, j]) * 0.5 * (u[i + 1, j - 1] + u[i + 1, j])
                if i > 0:
                    left = 0.5 * (v[i - 1, j] + v[i, j]) * 0.5 * (u[i, j - 1] + u[i, j])
                cv[i, j - 1] = (right - left + top - bottom) / h
        return np.concatenate([cu.ravel(), cv.ravel()])

    def step(self, state: MACState) -> MACState:
        conv = self.convection(state.velocity)
        if state.step_index == 0:
            history = state.velocity / self.dt - conv
            factor = self._be_factor
        else:
            history = ((2.0 * state.velocity - 0.5 * state.previous_velocity) / self.dt
                       - (2.0 * conv - state.previous_convection))
            factor = self._bdf_factor
        rhs_velocity = history + self.viscosity * self.boundary_load
        rhs = np.concatenate([rhs_velocity, np.zeros(self.pressure_size + 1)])
        answer = lu_solve(factor, rhs, check_finite=False)
        if not np.all(np.isfinite(answer)):
            raise FloatingPointError("non-finite MAC cavity state")
        return MACState(answer[:self.velocity_size], state.velocity.copy(), conv,
                        answer[self.velocity_size:self.velocity_size + self.pressure_size],
                        state.step_index + 1, state.time + self.dt)

    def velocity_rate(self, new: MACState, old: MACState) -> float:
        numerator = np.linalg.norm(new.velocity - old.velocity) / self.dt
        return float(numerator / max(np.linalg.norm(new.velocity), 1.0))

    def centerlines(self, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u, v = self._full_velocity(velocity)
        yu = np.asarray([point for point, _ in CAVITY_REFERENCE100_U_CENTERLINE])
        xv = np.asarray([point for point, _ in CAVITY_REFERENCE100_V_CENTERLINE])
        y_u = (np.arange(self.cells) + 0.5) * self.h
        x_v = (np.arange(self.cells) + 0.5) * self.h
        u_mid = 0.5 * (u[self.cells // 2, :] + u[(self.cells + 1) // 2, :])
        v_mid = 0.5 * (v[:, self.cells // 2] + v[:, (self.cells + 1) // 2])
        u_values = np.interp(yu, np.concatenate([[0.0], y_u, [1.0]]),
                             np.concatenate([[0.0], u_mid, [1.0]]))
        v_values = np.interp(xv, np.concatenate([[0.0], x_v, [1.0]]),
                             np.concatenate([[0.0], v_mid, [0.0]]))
        return u_values, v_values

    def diagnostics(self, state: MACState, steady_rate: float) -> dict[str, Any]:
        velocity = state.velocity
        convection = self.convection(velocity)
        viscous = -self.viscosity * (self.laplacian @ velocity)
        pressure = self.gradient @ state.pressure
        wall = -self.viscosity * self.boundary_load
        spatial = convection + viscous + pressure + wall
        u_values, v_values = self.centerlines(velocity)
        score = centerline_score(u_values, v_values)
        energy_terms = [float(velocity @ term) for term in (convection, viscous, pressure, wall)]
        energy_scale = max(sum(abs(value) for value in energy_terms), 1.0e-300)
        momentum_scale = max(float(np.linalg.norm(convection) + np.linalg.norm(viscous)
                                   + np.linalg.norm(pressure) + np.linalg.norm(wall)), 1.0e-300)
        top_u = velocity[:self.nu].reshape(self.u_shape)[:, -1]
        wall_work = self.viscosity * float(np.sum(2.0 * (1.0 - top_u) / self.h) * self.h)
        return {
            "centerline": score,
            "centerline_discrepancy_percentage_points": 100.0 * score["interior_combined_relative_l2"],
            "u_centerline": [float(value) for value in u_values],
            "v_centerline": [float(value) for value in v_values],
            "divergence_l2": float(np.linalg.norm(self.divergence @ velocity) * self.h),
            "pressure_gauge_residual": abs(float(self.constant_pressure @ state.pressure)),
            "pressure_nullity": self.pressure_size - int(np.linalg.matrix_rank(self.divergence)),
            "normal_trace_max": 0.0,
            "kinetic_energy": 0.5 * self.h * self.h * float(velocity @ velocity),
            "relative_energy_ledger_residual": abs(sum(energy_terms)) / energy_scale,
            "relative_momentum_ledger_residual": float(np.linalg.norm(spatial)) / momentum_scale,
            "convective_energy_production": energy_terms[0],
            "pressure_work": energy_terms[2],
            "wall_work": wall_work,
            "wall_reaction": -wall_work,
            "steady_relative_velocity_rate": steady_rate,
        }


def run_mac_case(cells: int, steady_tolerance: float, *, maximum_time: float = MAXIMUM_TIME) -> dict[str, Any]:
    setup_start = time.process_time()
    stepper = MACCavityStepper(cells, REYNOLDS, DT)
    setup_cpu = time.process_time() - setup_start
    state = stepper.initial_state()
    steady_rate = math.inf
    march_start = time.process_time()
    while state.time < maximum_time - 0.5 * DT:
        old = state
        state = stepper.step(state)
        steady_rate = stepper.velocity_rate(state, old)
        if state.time >= MINIMUM_TIME and steady_rate < steady_tolerance:
            break
    march_cpu = time.process_time() - march_start
    record = stepper.diagnostics(state, steady_rate)
    record.update({
        "method": "MAC2", "grid": cells, "velocity_states": stepper.velocity_size,
        "pressure_states": stepper.pressure_size, "steps": state.step_index,
        "final_time": state.time, "converged": steady_rate < steady_tolerance,
        "steady_tolerance": steady_tolerance, "dt": DT,
        "setup_cpu_seconds": setup_cpu, "march_cpu_seconds": march_cpu,
        "total_cpu_seconds": setup_cpu + march_cpu, "peak_rss_mib": peak_rss_mib(),
    })
    return record


def _bmf_diagnostics(stepper: RouteCCavityStepper, state: Any, steady_rate: float) -> dict[str, Any]:
    record = stepper.diagnostics(state, steady_rate)
    record["centerline_discrepancy_percentage_points"] = 100.0 * record["centerline"]["interior_combined_relative_l2"]
    ux, uz = stepper.formulation._unpack_active(state.velocity)
    cux, cuz, _ledger = stepper.formulation.skew_convection(ux, uz)
    convection = stepper.mass @ stepper.formulation._pack_active(cux, cuz)
    viscous = stepper.viscosity * (stepper.viscous @ state.velocity)
    pressure = stepper.divergence.T @ state.dual_pressure
    wall = -stepper.viscosity * stepper.boundary_load
    spatial = convection + viscous + pressure + wall
    momentum_scale = max(float(sum(np.linalg.norm(term) for term in (convection, viscous, pressure, wall))), 1.0e-300)
    record.update({
        "pressure_gauge_residual": record.pop("dual_pressure_gauge_residual"),
        "pressure_nullity": stepper.nq - int(np.linalg.matrix_rank(stepper.divergence)),
        "relative_energy_ledger_residual": record.pop("relative_steady_energy_residual"),
        "relative_momentum_ledger_residual": float(np.linalg.norm(spatial)) / momentum_scale,
        "pressure_work": float(state.velocity @ pressure),
        "wall_work": float(-state.velocity @ wall),
        "wall_reaction": float(np.sum(wall)),
    })
    return record


def run_bmf_case(npts: int, steady_tolerance: float, *, maximum_time: float = MAXIMUM_TIME) -> dict[str, Any]:
    setup_start = time.process_time()
    stepper = RouteCCavityStepper(npts, 4, REYNOLDS, DT, 64.0)
    setup_cpu = time.process_time() - setup_start
    state = stepper.initial_state()
    steady_rate = math.inf
    march_start = time.process_time()
    while state.time < maximum_time - 0.5 * DT:
        old = state
        state = stepper.step(state)
        steady_rate = stepper.velocity_rate(state, old)
        if state.time >= MINIMUM_TIME and steady_rate < steady_tolerance:
            break
    march_cpu = time.process_time() - march_start
    record = _bmf_diagnostics(stepper, state, steady_rate)
    record.update({
        "method": "BMF4", "grid": npts, "velocity_states": stepper.nv,
        "pressure_states": stepper.nq, "steps": state.step_index,
        "final_time": state.time, "converged": steady_rate < steady_tolerance,
        "steady_tolerance": steady_tolerance, "dt": DT,
        "setup_cpu_seconds": setup_cpu, "march_cpu_seconds": march_cpu,
        "total_cpu_seconds": setup_cpu + march_cpu, "peak_rss_mib": peak_rss_mib(),
    })
    return record


def gate_certificate() -> dict[str, Any]:
    stepper = MACCavityStepper(6, REYNOLDS, DT)
    rng = np.random.default_rng(481)
    rhs_velocity = rng.standard_normal(stepper.velocity_size)
    rhs = np.concatenate([rhs_velocity, np.zeros(stepper.pressure_size + 1)])
    answer = lu_solve(lu_factor(stepper._saddle_matrix(1.0), check_finite=False), rhs, check_finite=False)
    projected_velocity = answer[:stepper.velocity_size]
    projected_pressure = answer[stepper.velocity_size:stepper.velocity_size + stepper.pressure_size]
    short = run_mac_case(6, 0.0, maximum_time=0.20)
    reference_u = np.asarray([value for _, value in CAVITY_REFERENCE100_U_CENTERLINE])
    reference_v = np.asarray([value for _, value in CAVITY_REFERENCE100_V_CENTERLINE])
    certificate = {
        "schema": SCHEMA + "-gate",
        "source_sha256": source_hash(),
        "config_sha256": canonical_hash(frozen_config()),
        "candidate": "second-order staggered MAC",
        "pressure_nullity": stepper.pressure_size - int(np.linalg.matrix_rank(stepper.divergence)),
        "constant_pressure_gradient_max": float(np.max(np.abs(stepper.gradient @ np.ones(stepper.pressure_size)))),
        "projected_divergence_l2": float(np.linalg.norm(stepper.divergence @ projected_velocity) * stepper.h),
        "projected_pressure_gauge_residual": abs(float(stepper.constant_pressure @ projected_pressure)),
        "no_penetration": "boundary-normal face velocities are absent from the active state and fixed to zero",
        "tangential_wall_treatment": "reflected Dirichlet ghosts; unit top lid and zero other tangential traces",
        "nonlinear_short_march_finite": all(math.isfinite(float(short[key])) for key in (
            "kinetic_energy", "divergence_l2", "pressure_gauge_residual", "wall_work")),
        "short_march_divergence_l2": short["divergence_l2"],
        "short_march_pressure_gauge_residual": short["pressure_gauge_residual"],
        "identical_cavity_reference_score_zero": centerline_score(reference_u, reference_v)["combined_relative_l2"],
        "steady_criterion": "first step at t>=10 with relative velocity rate below the frozen tolerance; otherwise t=60",
    }
    verify_gate(certificate)
    return certificate


def verify_gate(record: dict[str, Any]) -> None:
    if record.get("source_sha256") != source_hash() or record.get("config_sha256") != canonical_hash(frozen_config()):
        raise ValueError("comparator gate provenance mismatch")
    if record.get("pressure_nullity") != 1:
        raise AssertionError("comparator pressure nullity gate failed")
    for key, tolerance in (
        ("constant_pressure_gradient_max", 2.0e-12),
        ("projected_divergence_l2", DIVERGENCE_TOLERANCE),
        ("projected_pressure_gauge_residual", PRESSURE_GAUGE_TOLERANCE),
        ("short_march_divergence_l2", DIVERGENCE_TOLERANCE),
        ("short_march_pressure_gauge_residual", PRESSURE_GAUGE_TOLERANCE),
        ("identical_cavity_reference_score_zero", 1.0e-15),
    ):
        if not math.isfinite(float(record.get(key, math.nan))) or float(record[key]) >= tolerance:
            raise AssertionError(f"comparator gate failed: {key}")
    if record.get("nonlinear_short_march_finite") is not True:
        raise AssertionError("comparator nonlinear short-march gate failed")


def build_summary(raw: dict[str, Any]) -> dict[str, Any]:
    rows = []
    by_key = {(run["method"], run["grid"], run["steady_tolerance"]): run for run in raw["runs"]}
    for bmf_grid, mac_grid in GRIDS:
        for method, grid in (("BMF4", bmf_grid), ("MAC2", mac_grid)):
            base = by_key[(method, grid, BASE_STEADY_TOLERANCE)]
            sensitivity = by_key[(method, grid, SENSITIVITY_STEADY_TOLERANCE)]
            delta = abs(base["centerline_discrepancy_percentage_points"] - sensitivity["centerline_discrepancy_percentage_points"])
            rows.append({
                "method": method, "grid": grid,
                "velocity_states": base["velocity_states"], "pressure_states": base["pressure_states"],
                "centerline_discrepancy_percentage_points": base["centerline_discrepancy_percentage_points"],
                "sensitivity_change_percentage_points": delta,
                "physical_accuracy_pass": base["centerline_discrepancy_percentage_points"] <= PHYSICAL_TOLERANCE_PP,
                "sensitivity_pass": delta < SENSITIVITY_TOLERANCE_PP,
                "divergence_l2": base["divergence_l2"], "pressure_nullity": base["pressure_nullity"],
                "pressure_gauge_residual": base["pressure_gauge_residual"],
                "relative_energy_ledger_residual": base["relative_energy_ledger_residual"],
                "relative_momentum_ledger_residual": base["relative_momentum_ledger_residual"],
                "wall_work": base["wall_work"], "wall_reaction": base["wall_reaction"],
                "setup_cpu_seconds": base["setup_cpu_seconds"], "march_cpu_seconds": base["march_cpu_seconds"],
                "total_cpu_seconds": base["total_cpu_seconds"], "peak_rss_mib": base["peak_rss_mib"],
                "steps": base["steps"], "final_time": base["final_time"], "converged": base["converged"],
            })
    eligible = [row for row in rows if row["physical_accuracy_pass"] and row["sensitivity_pass"]]
    decisive = []
    for bmf_grid, mac_grid in GRIDS:
        bmf = next(row for row in rows if row["method"] == "BMF4" and row["grid"] == bmf_grid)
        mac = next(row for row in rows if row["method"] == "MAC2" and row["grid"] == mac_grid)
        if bmf in eligible and mac in eligible:
            decisive.append({
                "bmf_grid": bmf_grid, "mac_grid": mac_grid,
                "bmf_velocity_state_reduction_fraction": 1.0 - bmf["velocity_states"] / mac["velocity_states"],
                "bmf_mission_cost_ratio": bmf["total_cpu_seconds"] / mac["total_cpu_seconds"],
            })
    promoted = any(cell["bmf_velocity_state_reduction_fraction"] >= 0.25 or cell["bmf_mission_cost_ratio"] <= 0.5 for cell in decisive)
    status = "comparative_promotion" if promoted else "validation_only_fallback"
    return {
        "schema": SCHEMA + "-summary", "source_sha256": source_hash(),
        "config_sha256": canonical_hash(frozen_config()), "rows": rows,
        "eligible_matched_pairs": decisive, "claim_status": status,
        "promotion_rule": "both methods pass physical and sensitivity gates, then BMF has >=25% fewer velocity states or <=0.5 mission-cost ratio",
    }


def verify_raw(record: dict[str, Any]) -> None:
    if record.get("schema") != SCHEMA or record.get("config") != frozen_config():
        raise ValueError("comparator raw schema or configuration mismatch")
    if record.get("source_sha256") != source_hash() or record.get("config_sha256") != canonical_hash(frozen_config()):
        raise ValueError("comparator raw provenance mismatch")
    expected = {(method, grid, tol) for bmf, mac in GRIDS for method, grid in (("BMF4", bmf), ("MAC2", mac)) for tol in (BASE_STEADY_TOLERANCE, SENSITIVITY_STEADY_TOLERANCE)}
    actual = {(run.get("method"), run.get("grid"), run.get("steady_tolerance")) for run in record.get("runs", [])}
    if actual != expected:
        raise ValueError("comparator run matrix mismatch")
    for run in record["runs"]:
        if not run.get("converged"):
            raise AssertionError("comparator steady-state gate failed")
        if run["divergence_l2"] >= DIVERGENCE_TOLERANCE or run["pressure_gauge_residual"] >= PRESSURE_GAUGE_TOLERANCE:
            raise AssertionError("comparator compatible-structure gate failed")
        if run["pressure_nullity"] != 1:
            raise AssertionError("comparator pressure-nullity gate failed")
        if len(run["u_centerline"]) != 17 or len(run["v_centerline"]) != 17:
            raise ValueError("comparator centerline coverage mismatch")
        score = centerline_score(np.asarray(run["u_centerline"]), np.asarray(run["v_centerline"]))
        if score != run["centerline"]:
            raise ValueError("comparator centerline arithmetic mismatch")


def make_figure(summary: dict[str, Any]) -> None:
    rows = summary["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    colors = {"BMF4": "#22517a", "MAC2": "#1f6f6f"}
    labels = {"BMF4": r"$\mathrm{BMF}_4$", "MAC2": "MAC2"}
    for method in ("BMF4", "MAC2"):
        selected = [row for row in rows if row["method"] == method]
        axes[0].plot([r["velocity_states"] for r in selected], [r["centerline_discrepancy_percentage_points"] for r in selected], "o-", color=colors[method], label=labels[method])
        axes[1].plot([r["centerline_discrepancy_percentage_points"] for r in selected], [r["total_cpu_seconds"] for r in selected], "o-", color=colors[method], label=labels[method])
    axes[0].axhline(PHYSICAL_TOLERANCE_PP, color="0.45", lw=0.8, ls=(0, (4, 3)), label="12 pp gate")
    axes[0].set_xlabel("velocity states")
    axes[0].set_ylabel("reference discrepancy (percentage points)")
    axes[1].set_xlabel("reference discrepancy (percentage points)")
    axes[1].set_ylabel("mission CPU time (s)")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.grid(alpha=0.18, lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, FIGDIR / "fig_s6_cavity_compatible_cost_accuracy")
    plt.close(fig)


def run_primary() -> dict[str, Any]:
    runs = []
    for bmf, mac in GRIDS:
        for tolerance in (BASE_STEADY_TOLERANCE, SENSITIVITY_STEADY_TOLERANCE):
            for method, grid, runner in (("BMF4", bmf, run_bmf_case), ("MAC2", mac, run_mac_case)):
                print(f"running {method} grid={grid} steady={tolerance:.1e}", flush=True)
                runs.append(runner(grid, tolerance))
    record = {
        "schema": SCHEMA, "config": frozen_config(), "source_sha256": source_hash(),
        "config_sha256": canonical_hash(frozen_config()), "timing": timing_stamp(),
        "runs": runs,
    }
    verify_raw(record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    ensure_generated_dirs()
    if args.gate:
        record = gate_certificate()
        GATE_OUTPUT.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        print(f"wrote {GATE_OUTPUT}")
    elif args.run:
        if RAW_OUTPUT.exists() or SUMMARY_OUTPUT.exists():
            raise FileExistsError("refusing to overwrite comparator evidence")
        raw = run_primary()
        summary = build_summary(raw)
        RAW_OUTPUT.write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
        SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        make_figure(summary)
        print(f"wrote {RAW_OUTPUT} and {SUMMARY_OUTPUT}")
    elif args.verify:
        gate = json.loads(GATE_OUTPUT.read_text())
        raw = json.loads(RAW_OUTPUT.read_text())
        summary = json.loads(SUMMARY_OUTPUT.read_text())
        verify_gate(gate)
        verify_raw(raw)
        if build_summary(raw) != summary:
            raise ValueError("comparator summary does not reproduce from raw evidence")
        print("cavity compatible comparator verify: ALL PASS")
    else:
        parser.error("select --gate, --run, or --verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

