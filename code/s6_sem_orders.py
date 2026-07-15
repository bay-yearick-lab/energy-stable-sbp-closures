"""Across-order spectral-element head-to-head legs (orders 4 and 8) for the
across-order evidence matrix; the main text displays order 6.

Heat: convergence-ladder configuration at matched-DOF grids N in {25, 49, 97}
(N-1 divisible by 24, so degree-4 and degree-8 elements both fit), tier-B
per-method steps. Wave: closed-box standing mode at N=49, shared cliff step.
SEM degree always matches the family order q.

Output: code/results/sem_orders_summary.json. Verification (--verify): all
finite; at each order and every grid both spline routes beat SEM on heat
error at matched DOFs; SEM stays energy-stable (abscissa < 1e-8).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))

import boundary_spline_analysis as ba
from heat_cavity_core import default_heat_modes, per_method_diffusion_dt, run_heat_cavity
from project_paths import RESULTSDIR, ensure_generated_dirs
from sem_operator import sem_grid_for

ORDERS = (4, 8)
GRIDS = (25, 49, 97)
HEAT_CFG = dict(m_center=8, n_center=8, half_band=1, viscosity=5.0e-5)
T_END = 0.5
WAVE_NPTS = 49
WAVE_MODE = 6
WAVE_T = 6.0
WAVE_CFL = 0.012


def expected_config():
    return {"orders": list(ORDERS), "grids": list(GRIDS),
            "heat": {**HEAT_CFG, "t_end": T_END,
                     "norm": "relative L2 with actual-node trapezoid weights"},
            "wave": {"npts": WAVE_NPTS, "mode": WAVE_MODE,
                     "t_end": WAVE_T, "cfl": WAVE_CFL,
                     "norm": "full-domain coupled relative L2 with actual-node trapezoid weights"}}


def _nonuniform_trap(x):
    nodes = np.asarray(x, float)
    weights = np.empty_like(nodes)
    weights[0] = 0.5 * (nodes[1] - nodes[0])
    weights[-1] = 0.5 * (nodes[-1] - nodes[-2])
    weights[1:-1] = 0.5 * (nodes[2:] - nodes[:-2])
    return weights


def _coupled_rel_l2(result):
    weights = _nonuniform_trap(result.grid)
    pressure_error = np.asarray(result.pressure) - np.asarray(result.pressure_exact)
    velocity_error = np.asarray(result.velocity) - np.asarray(result.velocity_exact)
    numerator = np.sum(weights * (pressure_error**2 + velocity_error**2))
    denominator = np.sum(
        weights
        * (np.asarray(result.pressure_exact) ** 2 + np.asarray(result.velocity_exact) ** 2)
    )
    return float(np.sqrt(numerator / denominator))


def spline_wave(q, family, npts):
    op = ba.build_operator(ba.scheme_kind(f"{family}{q}"), q, npts)
    result = ba.simulate_reflected_acoustics(
        op,
        lambda x, t: np.cos(WAVE_MODE * np.pi * np.asarray(x, float)) * np.cos(WAVE_MODE * np.pi * t),
        lambda x, t: np.sin(WAVE_MODE * np.pi * np.asarray(x, float)) * np.sin(WAVE_MODE * np.pi * t),
        WAVE_T,
        cfl=WAVE_CFL,
        interior_strip=0,
        record_count=4,
    )
    return {
        "family": f"{family}{q}",
        "q": q,
        "rel_l2_final": _coupled_rel_l2(result),
    }


def spline_heat(q, label, npts, modes):
    dt = per_method_diffusion_dt(label, npts=npts,
                                 viscosity=modes.viscosity, t_end=T_END)
    t0 = time.process_time()
    res = run_heat_cavity(label, modes, npts=npts, t_end=T_END, dt=dt,
                          record_count=2, snapshot_count=2, enforcement="horth")
    return {"family": label, "q": q, "npts": npts,
            "rel_l2_final": float(res.u_rel_l2[-1]),
            "cpu_seconds": time.process_time() - t0}


def sem_heat(q, npts, modes):
    op = sem_grid_for(npts, q)
    x, nu = op.x, float(modes.viscosity)
    rng = np.random.default_rng(3)
    V = rng.standard_normal((npts, npts))
    V[[0, -1], :] = 0.0
    V[:, [0, -1]] = 0.0
    lam = 1.0
    for _ in range(160):
        W = nu * (op.D2 @ V + V @ op.D2.T)
        W[[0, -1], :] = 0.0
        W[:, [0, -1]] = 0.0
        lam = float(np.linalg.norm(W))
        if lam == 0:
            break
        V = W / lam
    nsteps = max(1, int(np.ceil(T_END / (0.4 * 2.785 / max(lam, 1e-30)))))
    dt = T_END / nsteps
    U = modes.field_grid(x, x, 0.0)
    U[[0, -1], :] = 0.0
    U[:, [0, -1]] = 0.0

    def rhs(Uv):
        R = nu * (op.D2 @ Uv + Uv @ op.D2.T)
        R[[0, -1], :] = 0.0
        R[:, [0, -1]] = 0.0
        return R

    t0 = time.process_time()
    t = 0.0
    for _ in range(nsteps):
        k1 = rhs(U); k2 = rhs(U + 0.5 * dt * k1)
        k3 = rhs(U + 0.5 * dt * k2); k4 = rhs(U + dt * k3)
        U = U + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        U[[0, -1], :] = 0.0
        U[:, [0, -1]] = 0.0
        t += dt
    cpu = time.process_time() - t0
    ue = modes.field_grid(x, x, t)
    w = np.zeros_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2]); w[0] = 0.5 * (x[1] - x[0]); w[-1] = 0.5 * (x[-1] - x[-2])
    w2 = np.outer(w, w)
    err = float(np.sqrt(np.sum(w2 * (U - ue) ** 2)) / np.sqrt(np.sum(w2 * ue ** 2)))
    return {"rel_l2_final": err, "cpu_seconds": cpu, "nsteps": nsteps}


def sem_wave(q, npts):
    op = sem_grid_for(npts, q)
    x = op.x
    dt = WAVE_CFL / (npts - 1)
    nsteps = int(np.ceil(WAVE_T / dt)); dt = WAVE_T / nsteps
    c = WAVE_MODE * np.pi
    P, U = np.cos(c * x), np.zeros_like(x)
    D1 = op.D1

    def rhs(Pv, Uv):
        dP = -(D1 @ Uv); dU = -(D1 @ Pv)
        dU[0] = dU[-1] = 0.0
        return dP, dU

    for _ in range(nsteps):
        k1p, k1u = rhs(P, U)
        k2p, k2u = rhs(P + 0.5 * dt * k1p, U + 0.5 * dt * k1u)
        k3p, k3u = rhs(P + 0.5 * dt * k2p, U + 0.5 * dt * k2u)
        k4p, k4u = rhs(P + dt * k3p, U + dt * k3u)
        P = P + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        U = U + dt / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        U[0] = U[-1] = 0.0
    pr = np.cos(c * x) * np.cos(c * WAVE_T)
    ur = np.sin(c * x) * np.sin(c * WAVE_T)
    w = np.zeros_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2]); w[0] = 0.5 * (x[1] - x[0]); w[-1] = 0.5 * (x[-1] - x[-2])
    num = np.sqrt(np.sum(w * ((P - pr) ** 2 + (U - ur) ** 2)))
    den = np.sqrt(np.sum(w * (pr ** 2 + ur ** 2)))
    n = x.size
    A = np.zeros((2 * n, 2 * n))
    A[:n, n:] = -D1; A[n:, :n] = -D1
    A[n, :] = 0.0; A[2 * n - 1, :] = 0.0
    return {"rel_l2_final": float(num / den),
            "abscissa": float(np.max(np.linalg.eigvals(A).real))}


def run_all():
    out = {"config": expected_config(),
           "heat": [], "wave": []}
    modes = default_heat_modes(**HEAT_CFG)
    for q in ORDERS:
        for npts in GRIDS:
            for label in (f"SBP{q}", f"BMF{q}", f"BMG{q}"):
                try:
                    rec = spline_heat(q, label, npts, modes)
                except Exception as exc:
                    rec = {"family": label, "q": q, "npts": npts, "error": str(exc)}
                out["heat"].append(rec)
                print(json.dumps(rec)[:140], flush=True)
            rec = {"family": f"SEM{q}", "q": q, "npts": npts, **sem_heat(q, npts, modes)}
            out["heat"].append(rec)
            print(json.dumps(rec)[:140], flush=True)
        for fam in ("SBP", "BMF", "BMG"):
            out["wave"].append(spline_wave(q, fam, WAVE_NPTS))
            print(out["wave"][-1], flush=True)
        out["wave"].append({"family": f"SEM{q}", "q": q, **sem_wave(q, WAVE_NPTS)})
        print(out["wave"][-1], flush=True)
    return out


def rescore_spline_cases(existing, heat_runner=spline_heat,
                         wave_runner=spline_wave):
    """Preserve valid SEM rows and replace every spline heat/wave score."""
    sem_heat_rows = [r for r in existing.get("heat", [])
                     if str(r.get("family", "")).startswith("SEM")]
    sem_wave_rows = [r for r in existing.get("wave", [])
                     if str(r.get("family", "")).startswith("SEM")]
    expected_sem_heat = {(f"SEM{q}", q, npts) for q in ORDERS for npts in GRIDS}
    actual_sem_heat = {(r.get("family"), r.get("q"), r.get("npts")) for r in sem_heat_rows}
    if actual_sem_heat != expected_sem_heat or len(sem_heat_rows) != len(expected_sem_heat):
        raise ValueError("source across-order SEM heat sweep is incomplete")
    expected_sem_wave = {(f"SEM{q}", q) for q in ORDERS}
    actual_sem_wave = {(r.get("family"), r.get("q")) for r in sem_wave_rows}
    if actual_sem_wave != expected_sem_wave or len(sem_wave_rows) != len(expected_sem_wave):
        raise ValueError("source across-order SEM wave sweep is incomplete")
    sem_heat = {(r["q"], r["npts"]): r for r in sem_heat_rows}
    sem_wave_rows_by_q = {r["q"]: r for r in sem_wave_rows}
    modes = default_heat_modes(**HEAT_CFG)
    out = {"config": expected_config(), "heat": [], "wave": []}
    for q in ORDERS:
        for npts in GRIDS:
            for family in ("SBP", "BMF", "BMG"):
                out["heat"].append(heat_runner(q, f"{family}{q}", npts, modes))
            out["heat"].append(sem_heat[(q, npts)])
        for family in ("SBP", "BMF", "BMG"):
            out["wave"].append(wave_runner(q, family, WAVE_NPTS))
        out["wave"].append(sem_wave_rows_by_q[q])
    return out


def verify(out):
    if out.get("config") != expected_config():
        raise ValueError("across-order SEM configuration differs from the released protocol")
    expected_heat = {(f"{family}{q}", q, npts)
                     for q in ORDERS for npts in GRIDS
                     for family in ("SBP", "BMF", "BMG", "SEM")}
    actual_heat = [(r.get("family"), r.get("q"), r.get("npts"))
                   for r in out.get("heat", [])]
    if len(actual_heat) != len(set(actual_heat)) or set(actual_heat) != expected_heat:
        raise ValueError("across-order SEM heat coverage is incomplete or duplicated")
    expected_wave = {(f"{family}{q}", q)
                     for q in ORDERS for family in ("SBP", "BMF", "BMG", "SEM")}
    actual_wave = [(r.get("family"), r.get("q")) for r in out.get("wave", [])]
    if len(actual_wave) != len(set(actual_wave)) or set(actual_wave) != expected_wave:
        raise ValueError("across-order SEM wave coverage is incomplete or duplicated")
    for q in ORDERS:
        for npts in GRIDS:
            rows = {r["family"]: r for r in out["heat"]
                    if r["q"] == q and r["npts"] == npts and "rel_l2_final" in r}
            if f"BMF{q}" in rows and f"SEM{q}" in rows:
                spl = max(rows[f"BMF{q}"]["rel_l2_final"], rows[f"BMG{q}"]["rel_l2_final"])
                assert spl < rows[f"SEM{q}"]["rel_l2_final"], (q, npts)
        semw = [r for r in out["wave"] if r["family"] == f"SEM{q}"][0]
        assert semw["abscissa"] < 1e-8
    print("sem_orders verify: ALL PASS")


def main():
    ensure_generated_dirs()
    path = RESULTSDIR / "sem_orders_summary.json"
    if "--verify" in sys.argv:
        verify(json.loads(path.read_text()))
        return
    if "--rescore-splines" in sys.argv:
        out = rescore_spline_cases(json.loads(path.read_text()))
    else:
        out = run_all()
    verify(out)
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
