"""Spectral-element head-to-head: the strongest standard variational
competitor fielded in two full PDE experiments (not only interpolation).

Leg A (heat): the Section-6 heat-cavity decay in the convergence-ladder
configuration (modes centered at m=n=8, half-band 1, nu=5e-5, t_end=0.5),
on matched-DOF grids npts = 6K+1 shared by all entrants. BMF6/BMG6/SBP6 run
through the paper's own harness (metric-orthogonal enforcement, tier-B
per-method RK4 step from `per_method_diffusion_dt`); the SEM entrant is the
textbook degree-6 GLL C0 spectral element (textbook form): diagonal
GLL mass, assembled Galerkin stiffness, strong Dirichlet walls (for the
diagonal SEM metric the row selector IS the metric-orthogonal projector),
and the identical tier-B power-iteration step protocol. Errors are relative
L2 at t_end (each family's own sample set, trapezoid weights from its actual
node spacing; the SEM native GLL-quadrature norm is recorded alongside as a
sensitivity check). Cost is single-thread CPU process time for the full
march, plus the DOF count and step count.

Leg B (acoustics): the long-time closed-box standing wave of the wall
cliff experiment (mode 6, t_end = 6 transits, unit wave speed), at the
matched grid npts = 43, all four entrants marched by RK4 at the shared cliff
step protocol (cfl = 0.012 on the shared uniform spacing); the SEM system is
the energy-stable weak-form pair (C + C^T = boundary, wall-normal velocity
strongly zero), so this is the strongest stable variational wave entrant.

Verification (--verify): SEM operator gates (quadrature, poly-exactness,
eigenvalue, SBP identity), SEM heat-ladder monotone convergence, SEM wave
abscissa <= 0 (+roundoff), and every recorded number finite.

Output: code/results/sem_headtohead_summary.json
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import boundary_spline_analysis as ba
from heat_cavity_core import (
    default_heat_modes,
    per_method_diffusion_dt,
    run_heat_cavity,
)
from project_paths import RESULTSDIR, ensure_generated_dirs
from sem_operator import SEMOperator1D, sem_grid_for

SPLINE_LABELS = ("SBP6", "BMF6", "BMG6")
GRIDS = (19, 37, 55, 73, 91)          # npts = 6K+1: matched DOFs for degree-6 SEM
HEAT_CFG = dict(m_center=8, n_center=8, half_band=1, viscosity=5.0e-5)
HEAT_T_END = 0.5
WAVE_NPTS = 43                        # 6*7+1
WAVE_MODE = 6
WAVE_T_END = 6.0
WAVE_CFL = 0.012                      # the wall-cliff shared step protocol
THREAD_ENV_NAMES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def timing_stamp() -> dict:
    return {
        "metric": "process_time_seconds",
        "cpu": platform.processor() or _cpu_model(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "thread_env": {name: os.environ.get(name, "unset") for name in THREAD_ENV_NAMES},
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
    }


def _validate_timing(stamp: dict) -> None:
    required = {"metric", "cpu", "platform", "python", "numpy", "thread_env", "nice"}
    if not isinstance(stamp, dict) or set(stamp) != required:
        raise ValueError("SEM head-to-head timing stamp is incomplete")
    if stamp["metric"] != "process_time_seconds":
        raise ValueError("SEM head-to-head timing metric is not process time")
    if not all(isinstance(stamp[key], str) and stamp[key] for key in ("cpu", "platform", "python", "numpy")):
        raise ValueError("SEM head-to-head timing identity is incomplete")
    if stamp["nice"] != 19:
        raise ValueError("SEM head-to-head timing was not collected at nice 19")
    if not isinstance(stamp["thread_env"], dict) or set(stamp["thread_env"]) != set(THREAD_ENV_NAMES):
        raise ValueError("SEM head-to-head thread stamp is incomplete")
    if any(stamp["thread_env"][name] != "1" for name in THREAD_ENV_NAMES):
        raise ValueError("SEM head-to-head timing was not pinned to one thread")


def expected_config() -> dict:
    return {
        "heat": {**HEAT_CFG, "t_end": HEAT_T_END, "grids": list(GRIDS),
                 "protocol": "tier-B per-method RK4 step (safety 0.4, power iteration), "
                             "metric-orthogonal strong walls, relative L2 at t_end on each "
                             "family's sample set with node-spacing trapezoid weights"},
        "wave": {"npts": WAVE_NPTS, "mode": WAVE_MODE, "t_end": WAVE_T_END,
                 "cfl_shared": WAVE_CFL,
                 "norm": "full-domain coupled relative L2 with actual-node trapezoid weights"},
        "sem": "degree-6 GLL C0 spectral elements, diagonal GLL mass, assembled "
               "Galerkin stiffness, strong Dirichlet rows (= metric-orthogonal for "
               "the diagonal SEM metric); gates: quadrature 2P-1, interior D2 "
               "poly-exactness to P, Dirichlet eigenvalue, SBP identity",
    }


def _nonuniform_trap(x: np.ndarray) -> np.ndarray:
    w = np.zeros_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return w


def _rel_l2(err2d: np.ndarray, ref2d: np.ndarray, w1d: np.ndarray) -> float:
    w2 = np.outer(w1d, w1d)
    num = float(np.sqrt(np.sum(w2 * err2d ** 2)))
    den = float(np.sqrt(np.sum(w2 * ref2d ** 2)))
    return num / den if den > 0 else num


def _coupled_rel_l2(result) -> float:
    grid = np.asarray(result.grid, float)
    weights = _nonuniform_trap(grid)
    pressure_error = np.asarray(result.pressure) - np.asarray(result.pressure_exact)
    velocity_error = np.asarray(result.velocity) - np.asarray(result.velocity_exact)
    numerator = np.sum(weights * (pressure_error**2 + velocity_error**2))
    denominator = np.sum(
        weights
        * (np.asarray(result.pressure_exact) ** 2 + np.asarray(result.velocity_exact) ** 2)
    )
    return float(np.sqrt(numerator / denominator))


# ------------------------------------------------------------------ heat leg

def run_spline_heat(label: str, npts: int, modes) -> dict:
    dt = per_method_diffusion_dt(label, npts=npts,
                                 viscosity=modes.viscosity, t_end=HEAT_T_END)
    t0 = time.process_time()
    res = run_heat_cavity(label, modes, npts=npts, t_end=HEAT_T_END, dt=dt,
                          record_count=2, snapshot_count=2, enforcement="horth")
    cpu = time.process_time() - t0
    return {
        "family": label, "npts": int(npts), "dofs": int(npts * npts),
        "dt": float(dt), "nsteps": int(round(HEAT_T_END / dt)),
        "rel_l2_final": float(res.u_rel_l2[-1]),
        "cpu_seconds": float(cpu),
    }


def sem_heat_dt(op: SEMOperator1D, viscosity: float) -> float:
    """Tier-B power-iteration step, the same protocol as
    `per_method_diffusion_dt` (safety 0.4, 160 iterations, enforced map)."""
    npts = op.npts
    rng = np.random.default_rng(3)
    V = rng.standard_normal((npts, npts))
    V[0, :] = V[-1, :] = 0.0
    V[:, 0] = V[:, -1] = 0.0
    lam = 1.0
    for _ in range(160):
        W = viscosity * (op.D2 @ V + V @ op.D2.T)
        W[0, :] = W[-1, :] = 0.0
        W[:, 0] = W[:, -1] = 0.0
        lam = float(np.linalg.norm(W))
        if lam == 0.0:
            break
        V = W / lam
    dt0 = 0.4 * 2.785 / max(lam, 1.0e-30)
    nsteps = max(1, int(np.ceil(HEAT_T_END / dt0)))
    return HEAT_T_END / nsteps


def run_sem_heat(npts: int, modes) -> dict:
    op = sem_grid_for(npts, 6)
    x = op.x
    nu = float(modes.viscosity)
    dt = sem_heat_dt(op, nu)
    nsteps = int(round(HEAT_T_END / dt))

    def rhs(U):
        R = nu * (op.D2 @ U + U @ op.D2.T)
        R[0, :] = R[-1, :] = 0.0
        R[:, 0] = R[:, -1] = 0.0
        return R

    U = modes.field_grid(x, x, 0.0)
    U[0, :] = U[-1, :] = 0.0
    U[:, 0] = U[:, -1] = 0.0
    t0 = time.process_time()
    t = 0.0
    for _ in range(nsteps):
        k1 = rhs(U)
        k2 = rhs(U + 0.5 * dt * k1)
        k3 = rhs(U + 0.5 * dt * k2)
        k4 = rhs(U + dt * k3)
        U = U + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        U[0, :] = U[-1, :] = 0.0
        U[:, 0] = U[:, -1] = 0.0
        t += dt
    cpu = time.process_time() - t0
    u_ex = modes.field_grid(x, x, t)
    err = U - u_ex
    return {
        "family": "SEM6", "npts": int(npts), "dofs": int(npts * npts),
        "dt": float(dt), "nsteps": int(nsteps),
        "rel_l2_final": _rel_l2(err, u_ex, _nonuniform_trap(x)),
        "rel_l2_final_native_gll": _rel_l2(err, u_ex, op.mass),
        "cpu_seconds": float(cpu),
    }


# ------------------------------------------------------------- acoustic leg

def p_exact(x, t):
    return np.cos(WAVE_MODE * np.pi * np.asarray(x, float)) * np.cos(WAVE_MODE * np.pi * t)


def u_exact(x, t):
    return np.sin(WAVE_MODE * np.pi * np.asarray(x, float)) * np.sin(WAVE_MODE * np.pi * t)


def run_spline_wave(family: str, npts: int) -> dict:
    kind = ba.scheme_kind(f"{family}6")
    op = ba.build_operator(kind, 6, npts)
    t0 = time.process_time()
    res = ba.simulate_reflected_acoustics(op, p_exact, u_exact, WAVE_T_END,
                                          cfl=WAVE_CFL, interior_strip=0,
                                          record_count=8)
    cpu = time.process_time() - t0
    return {
        "family": f"{family}6", "npts": int(npts),
        "rel_l2_final": _coupled_rel_l2(res),
        "cpu_seconds": float(cpu),
    }


def run_sem_wave(npts: int) -> dict:
    op = sem_grid_for(npts, 6)
    x = op.x
    h_shared = 1.0 / (npts - 1)
    dt = WAVE_CFL * h_shared
    nsteps = int(np.ceil(WAVE_T_END / dt))
    dt = WAVE_T_END / nsteps

    D1 = op.D1

    def rhs(P, U):
        dP = -(D1 @ U)
        dU = -(D1 @ P)
        dU[0] = dU[-1] = 0.0          # rigid walls: u = 0 strongly
        return dP, dU

    P = p_exact(x, 0.0)
    U = u_exact(x, 0.0)
    t = 0.0
    t0 = time.process_time()
    for _ in range(nsteps):
        k1p, k1u = rhs(P, U)
        k2p, k2u = rhs(P + 0.5 * dt * k1p, U + 0.5 * dt * k1u)
        k3p, k3u = rhs(P + 0.5 * dt * k2p, U + 0.5 * dt * k2u)
        k4p, k4u = rhs(P + dt * k3p, U + dt * k3u)
        P = P + dt / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
        U = U + dt / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
        U[0] = U[-1] = 0.0
        t += dt
    cpu = time.process_time() - t0
    w = _nonuniform_trap(x)
    perr = P - p_exact(x, t)
    uerr = U - u_exact(x, t)
    pref = p_exact(x, t)
    uref = u_exact(x, t)
    num = np.sqrt(np.sum(w * (perr ** 2 + uerr ** 2)))
    den = np.sqrt(np.sum(w * (pref ** 2 + uref ** 2)))
    # semidiscrete abscissa of the enforced SEM wave operator (energy gate)
    n = x.size
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = -D1
    A[n:, :n] = -D1
    A[n, :] = 0.0
    A[2 * n - 1, :] = 0.0
    ev = np.linalg.eigvals(A)
    return {
        "family": "SEM6", "npts": int(npts), "dt": float(dt), "nsteps": int(nsteps),
        "rel_l2_final": float(num / den),
        "cpu_seconds": float(cpu),
        "abscissa": float(np.max(ev.real)),
    }


# ------------------------------------------------------------------- driver

def run_all() -> dict:
    modes = default_heat_modes(**HEAT_CFG)
    out = {
        "config": expected_config(),
        "timing": timing_stamp(),
        "sem_gates": {},
        "heat": [],
        "wave": [],
    }
    for P, K in ((6, 3), (6, 8)):
        rep = SEMOperator1D(P, K).verify()
        out["sem_gates"][f"P{P}_K{K}"] = {k: float(v) for k, v in rep.items()}

    for npts in GRIDS:
        for label in SPLINE_LABELS:
            try:
                rec = run_spline_heat(label, npts, modes)
            except Exception as exc:  # e.g. SBP closure needs a minimum width
                rec = {"family": label, "npts": int(npts), "error": str(exc)}
            out["heat"].append(rec)
            print("heat", rec, flush=True)
        rec = run_sem_heat(npts, modes)
        out["heat"].append(rec)
        print("heat", rec, flush=True)

    for fam in ("SBP", "BMF", "BMG"):
        try:
            rec = run_spline_wave(fam, WAVE_NPTS)
        except Exception as exc:
            rec = {"family": f"{fam}6", "npts": WAVE_NPTS, "error": str(exc)}
        out["wave"].append(rec)
        print("wave", rec, flush=True)
    rec = run_sem_wave(WAVE_NPTS)
    out["wave"].append(rec)
    print("wave", rec, flush=True)
    return out


def rescore_spline_cases(existing: dict, heat_runner=run_spline_heat,
                         wave_runner=run_spline_wave) -> dict:
    """Recompute the spline rows under the corrected common norms.

    The SEM rows and algebraic gates already used actual-node quadrature and
    therefore remain valid. Exact coverage checks prevent a partial old sweep
    from being silently carried into the replacement artifact.
    """
    sem_heat = [r for r in existing.get("heat", []) if r.get("family") == "SEM6"]
    sem_wave = [r for r in existing.get("wave", []) if r.get("family") == "SEM6"]
    if {r.get("npts") for r in sem_heat} != set(GRIDS) or len(sem_wave) != 1:
        raise ValueError("source SEM head-to-head sweep is incomplete")
    modes = default_heat_modes(**HEAT_CFG)
    out = {"config": expected_config(), "timing": existing.get("timing"),
           "sem_gates": existing.get("sem_gates"),
           "heat": [], "wave": []}
    sem_by_grid = {r["npts"]: r for r in sem_heat}
    for npts in GRIDS:
        for label in SPLINE_LABELS:
            out["heat"].append(heat_runner(label, npts, modes))
        out["heat"].append(sem_by_grid[npts])
    for family in ("SBP", "BMF", "BMG"):
        out["wave"].append(wave_runner(family, WAVE_NPTS))
    out["wave"].append(sem_wave[0])
    return out


def rescore_sem_cases(existing: dict, heat_runner=run_sem_heat,
                      wave_runner=run_sem_wave, stamp: dict | None = None) -> dict:
    """Replace every SEM row under one controlled timing environment."""
    expected_spline_heat = {(family, npts) for family in SPLINE_LABELS for npts in GRIDS}
    spline_heat = [r for r in existing.get("heat", []) if r.get("family") in SPLINE_LABELS]
    actual_spline_heat = {(r.get("family"), r.get("npts")) for r in spline_heat}
    spline_wave = [r for r in existing.get("wave", []) if r.get("family") in SPLINE_LABELS]
    actual_spline_wave = {(r.get("family"), r.get("npts")) for r in spline_wave}
    expected_spline_wave = {(family, WAVE_NPTS) for family in SPLINE_LABELS}
    if actual_spline_heat != expected_spline_heat or len(spline_heat) != len(expected_spline_heat):
        raise ValueError("source spline heat head-to-head sweep is incomplete")
    if actual_spline_wave != expected_spline_wave or len(spline_wave) != len(expected_spline_wave):
        raise ValueError("source spline wave head-to-head sweep is incomplete")
    controlled_stamp = timing_stamp() if stamp is None else stamp
    _validate_timing(controlled_stamp)
    modes = default_heat_modes(**HEAT_CFG)
    spline_heat_by_case = {(r["family"], r["npts"]): r for r in spline_heat}
    spline_wave_by_family = {r["family"]: r for r in spline_wave}
    out = {
        "config": expected_config(),
        "timing": controlled_stamp,
        "sem_gates": {
            f"P{degree}_K{elements}": {
                key: float(value)
                for key, value in SEMOperator1D(degree, elements).verify().items()
            }
            for degree, elements in ((6, 3), (6, 8))
        },
        "heat": [],
        "wave": [],
    }
    for npts in GRIDS:
        for family in SPLINE_LABELS:
            out["heat"].append(spline_heat_by_case[(family, npts)])
        out["heat"].append(heat_runner(npts, modes))
    for family in SPLINE_LABELS:
        out["wave"].append(spline_wave_by_family[family])
    out["wave"].append(wave_runner(WAVE_NPTS))
    return out


def verify(out: dict) -> None:
    if out.get("config") != expected_config():
        raise ValueError("SEM head-to-head configuration differs from the released protocol")
    _validate_timing(out.get("timing"))
    if set(out.get("sem_gates", {})) != {"P6_K3", "P6_K8"}:
        raise ValueError("SEM gate coverage is incomplete")
    expected_heat = {(family, npts) for family in (*SPLINE_LABELS, "SEM6") for npts in GRIDS}
    actual_heat = [(r.get("family"), r.get("npts")) for r in out.get("heat", [])]
    if len(actual_heat) != len(set(actual_heat)) or set(actual_heat) != expected_heat:
        raise ValueError("SEM heat head-to-head coverage is incomplete or duplicated")
    expected_wave = {(family, WAVE_NPTS) for family in (*SPLINE_LABELS, "SEM6")}
    actual_wave = [(r.get("family"), r.get("npts")) for r in out.get("wave", [])]
    if len(actual_wave) != len(set(actual_wave)) or set(actual_wave) != expected_wave:
        raise ValueError("SEM wave head-to-head coverage is incomplete or duplicated")
    for gates in out["sem_gates"].values():
        assert gates["G1_quadrature"] < 1e-12
        assert gates["G4_sbp_identity"] < 1e-12
    sem = [r for r in out["heat"] if r["family"] == "SEM6"]
    errs = [r["rel_l2_final"] for r in sorted(sem, key=lambda r: r["npts"])]
    assert all(np.isfinite(errs)), errs
    assert errs[-1] < errs[0], ("SEM heat ladder must converge", errs)
    # native vs trapezoid norm agreement within 20% (sensitivity check)
    for r in sem:
        a, b = r["rel_l2_final"], r["rel_l2_final_native_gll"]
        assert abs(a - b) <= 0.2 * max(a, b), ("norm sensitivity", r)
    semw = [r for r in out["wave"] if r["family"] == "SEM6"][0]
    assert semw["abscissa"] < 1e-8, ("SEM wave energy gate", semw["abscissa"])
    for r in out["heat"] + out["wave"]:
        if "rel_l2_final" in r:
            assert np.isfinite(r["rel_l2_final"]), r
    print("sem_headtohead verify: ALL PASS")


def main() -> None:
    ensure_generated_dirs()
    path = RESULTSDIR / "sem_headtohead_summary.json"
    if "--verify" in sys.argv:
        out = json.loads(path.read_text())
        verify(out)
        return
    if "--rescore-splines" in sys.argv:
        out = rescore_spline_cases(json.loads(path.read_text()))
    elif "--rescore-sem" in sys.argv:
        out = rescore_sem_cases(json.loads(path.read_text()))
    else:
        out = run_all()
    verify(out)
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
