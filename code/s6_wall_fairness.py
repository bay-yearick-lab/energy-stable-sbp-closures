#!/usr/bin/env python3
"""Comparator fairness at the wall: published closures, SAT enforcement, narrow D2.

Four repairs are measured here on one grid family and one march protocol.

  K1  The compact family is closed by the published Section 4.1 compact wall rows
      instead of the generic explicit one-sided rows with an identity left
      side. The wall-row coefficients are derived from exactness conditions in
      `boundary_spline_analysis.compact_baseline_wall_row_coefficients`, not transcribed.

  K2  The diagonal-norm SBP first derivative is enforced by simultaneous
      approximation terms rather than by the metric-orthogonal projection, with
      the penalty parameters fixed by the energy method.

  K3  A narrow-stencil compatible second derivative of order six is fielded
      against the wide composition carried by SBP6-BO, whose second derivative
      is shown here to equal `D1 D1` identically.

  K4  Every newly fielded stable closure is marched through the closed-box
      six-transit test of Section 6.1 and compared with the spline routes and
      with the CSB schemes.

  K5  The marched wall-spectrum thresholds of the SAT-enforced operator are
      compared with the projection-enforced ones on the Section-4 code path.

Usage:
  python code/s6_wall_fairness.py --out code/results/wall_fairness_summary.json
  python code/s6_wall_fairness.py --verify code/results/wall_fairness_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import boundary_spline_analysis as ba  # noqa: E402
import baseline_closures as bl  # noqa: E402
from project_paths import RESULTSDIR, ensure_generated_dirs  # noqa: E402
from s6_wall_longtime_cliff import (  # noqa: E402
    ALPHA_F,
    MODE,
    p_exact,
    spectral_abscissa,
    u_exact,
)
from s4_staggered_wallmode import (  # noqa: E402
    TOLS,
    _wsum,
    enforced_pressure_operator,
    threshold_ppw,
)

T_END = 6.0
CFL = 0.012
STRIP = 6
GRIDS = (41, 81)
WALLMODE_NPTS = 129
STABLE_TOL = 1.0e-10
TAUS = (-1.0, -2.0)


# --------------------------------------------------------------- marches

def march_nodal(x: np.ndarray, dmat: np.ndarray, *, t_end: float = T_END,
                cfl: float = CFL, strip: int = STRIP,
                filter_matrix: np.ndarray | None = None) -> dict:
    """Closed-box six-transit march for a nodal first-derivative matrix.

    The protocol is the one `boundary_spline_analysis.simulate_reflected_acoustics`
    runs for the nodal families: state `(p on every node, u on the interior
    nodes)`, rigid-wall velocity traces imposed strongly, RK4 at
    `dt = cfl * h`, standing mode `m = MODE`, unit sound speed. A filter
    matrix, when supplied, is applied once per completed step to both fields,
    the convention of `s6_wall_longtime_cliff.simulate_extra_history`. The
    equivalence with the library path is gated in `verify`.
    """
    npts = x.size
    h = float(x[1] - x[0])
    dt = float(cfl) * h
    nsteps = int(np.ceil(float(t_end) / dt))
    dt = float(t_end) / nsteps
    m_u = npts - 2
    state = np.concatenate([p_exact(x, 0.0), u_exact(x, 0.0)[1:-1]])

    def unpack(y: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(y[:npts], dtype=float)
        u = np.empty(npts, dtype=float)
        u[0] = float(np.asarray(u_exact(x[0], t)))
        u[-1] = float(np.asarray(u_exact(x[-1], t)))
        u[1:-1] = y[npts:npts + m_u]
        return p, u

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        p, u = unpack(y, t)
        return np.concatenate([-dmat @ u, (-dmat @ p)[1:-1]])

    t = 0.0
    peak = 0.0
    for _step in range(nsteps):
        k1 = rhs(t, state)
        k2 = rhs(t + 0.5 * dt, state + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, state + 0.5 * dt * k2)
        k4 = rhs(t + dt, state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        if filter_matrix is not None:
            p_full, u_full = unpack(state, t)
            state = np.concatenate([ba.apply_filter(filter_matrix, p_full),
                                    ba.apply_filter(filter_matrix, u_full)[1:-1]])
        peak = max(peak, float(np.max(np.abs(state))) if np.all(np.isfinite(state)) else np.inf)

    p, u = unpack(state, float(t_end))
    pex = p_exact(x, float(t_end))
    uex = u_exact(x, float(t_end))
    sl = ba.trimmed_interior_slice(npts, strip)
    return {"trimmed": float(ba.coupled_relative_l2(p[sl], pex[sl], u[sl], uex[sl])),
            "untrimmed": float(ba.coupled_relative_l2(p, pex, u, uex)),
            "peak_state": float(peak)}


def march_sat(npts: int, order: int, tau: float, *, t_end: float = T_END,
              cfl: float = CFL, strip: int = STRIP) -> dict:
    """Six-transit march of the SAT-enforced pair; both velocity traces are free."""
    x, gen, _metric = ba.sbp_sat_acoustic_generator(npts, order, 1.0, float(tau))
    h = float(x[1] - x[0])
    dt = float(cfl) * h
    nsteps = int(np.ceil(float(t_end) / dt))
    dt = float(t_end) / nsteps
    y = np.concatenate([p_exact(x, 0.0), u_exact(x, 0.0)])
    for _step in range(nsteps):
        k1 = gen @ y
        k2 = gen @ (y + 0.5 * dt * k1)
        k3 = gen @ (y + 0.5 * dt * k2)
        k4 = gen @ (y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    p = y[:npts]
    u = y[npts:]
    pex = p_exact(x, float(t_end))
    uex = u_exact(x, float(t_end))
    sl = ba.trimmed_interior_slice(npts, strip)
    return {"trimmed": float(ba.coupled_relative_l2(p[sl], pex[sl], u[sl], uex[sl])),
            "untrimmed": float(ba.coupled_relative_l2(p, pex, u, uex)),
            "wall_velocity_residual": float(max(abs(u[0]), abs(u[-1])))}


def march_operator(op, *, t_end: float = T_END, cfl: float = CFL,
                   strip: int = STRIP) -> dict:
    """Six-transit march through the library path, trimmed and untrimmed."""
    res = ba.simulate_reflected_acoustics(op, p_exact, u_exact, t_end,
                                          cfl=cfl, interior_strip=strip,
                                          record_count=200)
    return {"trimmed": float(res.error_history[-1]),
            "untrimmed": float(ba.coupled_relative_l2(
                res.pressure, res.pressure_exact, res.velocity, res.velocity_exact))}


# -------------------------------------------------------------- abscissa

def matrix_abscissa(dmat: np.ndarray) -> float:
    npts = dmat.shape[0]
    zero_p = np.zeros((npts, npts))
    zero_u = np.zeros((npts - 2, npts - 2))
    gen = np.block([[zero_p, -dmat[:, 1:-1]], [-dmat[1:-1, :], zero_u]])
    return float(np.max(np.real(np.linalg.eigvals(gen))))


def sat_abscissa(npts: int, order: int, tau: float) -> float:
    _x, gen, _metric = ba.sbp_sat_acoustic_generator(npts, order, 1.0, float(tau))
    return float(np.max(np.real(np.linalg.eigvals(gen))))


# ------------------------------------------------------------- K1 record

def compact_baseline_closure_record() -> dict:
    classical = {3.0: (-17.0 / 6.0, 3.0 / 2.0, 3.0 / 2.0, -1.0 / 6.0),
                 2.0: (-5.0 / 2.0, 2.0, 1.0 / 2.0, 0.0)}
    out = {"wall_rows": {}, "near_wall_rows": {}}
    for alpha in (3.0, 2.0):
        coeff = ba.compact_baseline_wall_row_coefficients(alpha)
        s = np.arange(coeff.size, dtype=float)
        residuals = {}
        for k in range(6):
            lhs = (1.0 if k == 1 else 0.0) + alpha * (float(k) if k >= 1 else 0.0)
            residuals[f"degree{k}"] = float(coeff @ (s ** k) - lhs)
        out["wall_rows"][f"alpha{alpha:g}"] = {
            "derived": [float(v) for v in coeff],
            "classical": [float(v) for v in classical[alpha]],
            "max_deviation_from_classical": float(np.max(np.abs(coeff - np.asarray(classical[alpha])))),
            "exactness_residual": residuals,
            "formal_order": 4 if abs(residuals["degree4"]) < 1.0e-12 else 3,
        }
    for alpha, half, tag in ((0.25, 1, "pade4"), (1.0 / 3.0, 2, "pade6")):
        coeff = ba.pade_symmetric_row_coefficients(alpha, half)
        s = np.arange(-half, half + 1, dtype=float)
        res = {}
        for k in range(2 * half + 3):
            lhs = alpha * (float(k) * (-1.0) ** (k - 1) if k >= 1 else 0.0) \
                + (1.0 if k == 1 else 0.0) \
                + alpha * (float(k) if k >= 1 else 0.0)
            res[f"degree{k}"] = float(coeff @ (s ** k) - lhs)
        out["near_wall_rows"][tag] = {"alpha": float(alpha),
                                      "derived": [float(v) for v in coeff],
                                      "exactness_residual": res}
    return out


# ------------------------------------------------------------- K2 record

def sat_energy_record(npts: int = 41, order: int = 6) -> dict:
    limits = ba.sbp_sat_penalty_range()
    rec = {"penalty_direction": limits, "identity_residual": {}}
    rng = np.random.default_rng(20260913)
    for tau in TAUS:
        _x, gen, metric = ba.sbp_sat_acoustic_generator(npts, order, 1.0, tau)
        worst = 0.0
        for _trial in range(8):
            z = rng.standard_normal(2 * npts)
            rate = float(2.0 * z @ metric @ (gen @ z))
            predicted = float(2.0 * tau * (z[npts] ** 2 + z[2 * npts - 1] ** 2))
            worst = max(worst, abs(rate - predicted) / max(1.0, abs(predicted)))
        rec["identity_residual"][f"tau{tau:g}"] = worst
    # the same penalty magnitude applied only to the velocity equation, with no
    # pressure-equation term, leaves the indefinite cross term and is unstable
    x, dmat, hmat = ba.sbp_fd_matrix(npts, order, 1.0)
    hinv = np.diag(1.0 / np.diag(hmat))
    e0 = np.zeros((npts, npts)); e0[0, 0] = 1.0
    eN = np.zeros((npts, npts)); eN[-1, -1] = 1.0
    zero = np.zeros((npts, npts))
    naive = np.block([[zero, -dmat], [-dmat, -1.0 * hinv @ (e0 + eN)]])
    rec["velocity_only_penalty_abscissa"] = float(np.max(np.real(np.linalg.eigvals(naive))))
    return rec


# ------------------------------------------------------------- K3 record

def narrow_d2_record(grids=(41, 81)) -> dict:
    _block, selection = ba.narrow_d2_free_block(6)
    out = {"construction": {k: float(v) for k, v in selection.items()},
           "note": ("compatible narrow operator with the published order and "
                    "structure of the narrow compatible SBP second-derivative construction, derived here "
                    "from the accuracy conditions, not their published "
                    "coefficient set"),
           "narrow": {}, "wide_sbp6_bo": {}}
    for npts in grids:
        h = 1.0 / (npts - 1)
        x, d2, hmat, amat = ba.sbp_narrow_second_derivative_matrix(npts, 6, 1.0)
        sym = float(np.max(np.abs(amat - amat.T)) / np.max(np.abs(amat)))
        eig_a = np.sort(np.linalg.eigvalsh(0.5 * (amat + amat.T)))
        dirichlet = d2[1:-1, 1:-1]
        lam = np.linalg.eigvals(dirichlet)
        lam = lam[np.argsort(np.abs(lam))]
        saw = ((-1.0) ** np.arange(npts))[1:-1]
        saw = saw / np.linalg.norm(saw)
        out["narrow"][f"N{npts}"] = {
            "A_symmetry_residual": sym,
            "A_min_eigenvalue": float(eig_a[0]),
            "A_first_nonzero_eigenvalue": float(eig_a[1]),
            "dirichlet_max_abs_eigenvalue": float(np.max(np.abs(lam))),
            "dirichlet_max_abs_times_h2": float(np.max(np.abs(lam)) * h * h),
            "dirichlet_min_abs_eigenvalue": float(np.abs(lam[0])),
            "dirichlet_smallest_six": [float(np.real(v)) for v in lam[:6]],
            "odd_even_damping_rate": float(-np.real(saw @ dirichlet @ saw)),
            "interior_symbol_at_pi": float(sum(
                c * np.cos(np.pi * (j - 3)) for j, c in
                enumerate(ba.NARROW_D2_INTERIOR_ORDER6)) / h ** 2),
        }
        xbo, d2bo = ba.sbp6_bo_second_derivative_matrix(npts, 1.0)
        _x1, d1bo, _h1 = ba.sbp6_bo_fd_matrix(npts, 1.0)
        wide = d2bo[1:-1, 1:-1]
        lam_w = np.linalg.eigvals(wide)
        lam_w = lam_w[np.argsort(np.abs(lam_w))]
        _vals, vecs = np.linalg.eig(wide)
        overlap = np.abs(vecs.conj().T @ saw) / np.linalg.norm(vecs, axis=0)
        order_ov = np.argsort(-overlap)
        out["wide_sbp6_bo"][f"N{npts}"] = {
            "equals_D1_D1_residual": float(np.max(np.abs(d2bo - d1bo @ d1bo))
                                           / np.max(np.abs(d2bo))),
            "dirichlet_max_abs_eigenvalue": float(np.max(np.abs(lam_w))),
            "dirichlet_max_abs_times_h2": float(np.max(np.abs(lam_w)) * h * h),
            "dirichlet_min_abs_eigenvalue": float(np.abs(lam_w[0])),
            "dirichlet_smallest_six": [float(np.real(v)) for v in lam_w[:6]],
            "odd_even_damping_rate": float(-np.real(saw @ wide @ saw)),
            "most_odd_even_eigenvalue": float(np.real(_vals[order_ov[0]])),
            "most_odd_even_overlap": float(overlap[order_ov[0]]),
            "interior_symbol_at_pi": 0.0,
        }
    return out


# ------------------------------------------------------------- K5 record

def sat_second_order_response(npts: int, order: int, tau: float) -> list[dict]:
    """Section-4 marched pressure response for the SAT-enforced pair."""
    x, gen, _metric = ba.sbp_sat_acoustic_generator(npts, order, 1.0, float(tau))
    lp = (gen @ gen)[:npts, :npts]
    w = ba.nodal_trapezoidal_weights(x)
    h_ref = float(np.max(np.diff(x)))
    max_mode = min(npts - 2, int(np.floor((1.0 - 1.0e-12) / h_ref)))
    rows = []
    for m in range(1, max_mode + 1):
        c = m * np.pi
        cm = np.cos(c * x)
        img = lp @ cm
        den = _wsum(w, cm)
        kappa2 = -_wsum(w, img, cm) / den
        rho = float(np.sqrt(max(kappa2, 0.0)) / c)
        rem = img + kappa2 * cm
        eta = float(np.sqrt(max(_wsum(w, rem), 0.0)) / (c ** 2 * np.sqrt(den)))
        rows.append({"mode": m, "ppw": float(2.0 / (m * h_ref)), "rho": rho,
                     "eta": eta, "total": float(np.hypot(rho - 1.0, eta))})
    return rows


def thresholds_from_rows(rows: list[dict]) -> dict:
    ppw = np.asarray([r["ppw"] for r in rows])
    tot = np.asarray([r["total"] for r in rows])
    return {f"tol{t:g}": threshold_ppw(ppw, tot, t) for t in TOLS}


def wallmode_record(npts: int = WALLMODE_NPTS, order: int = 6) -> dict:
    x, lp = enforced_pressure_operator("SBP", order, npts)
    w = ba.nodal_trapezoidal_weights(x)
    h_ref = float(np.max(np.diff(x)))
    max_mode = min(npts - 2, int(np.floor((1.0 - 1.0e-12) / h_ref)))
    rows = []
    for m in range(1, max_mode + 1):
        c = m * np.pi
        cm = np.cos(c * x)
        img = lp @ cm
        den = _wsum(w, cm)
        kappa2 = -_wsum(w, img, cm) / den
        rho = float(np.sqrt(max(kappa2, 0.0)) / c)
        rem = img + kappa2 * cm
        eta = float(np.sqrt(max(_wsum(w, rem), 0.0)) / (c ** 2 * np.sqrt(den)))
        rows.append({"mode": m, "ppw": float(2.0 / (m * h_ref)), "rho": rho,
                     "eta": eta, "total": float(np.hypot(rho - 1.0, eta))})
    out = {"npts": npts, "order": order,
           "projection": thresholds_from_rows(rows), "sat": {}, "relative_shift": {}}
    for tau in TAUS:
        sat_rows = sat_second_order_response(npts, order, tau)
        thr = thresholds_from_rows(sat_rows)
        out["sat"][f"tau{tau:g}"] = thr
        out["relative_shift"][f"tau{tau:g}"] = {
            key: (float("nan") if not np.isfinite(out["projection"][key])
                  else float(abs(thr[key] - out["projection"][key]) / out["projection"][key]))
            for key in thr}
    return out


# ------------------------------------------------------------- assembly

def variant_catalog(npts: int) -> dict:
    """Nodal first-derivative matrices for every competitor at each order."""
    cat: dict[str, dict] = {}
    for q in (4, 6, 8):
        for alpha, tag in ((3.0, "3"), (2.0, "2")):
            op = ba.build_operator_extended(f"C{q}-CB{tag}", q, npts)
            cat[f"C{q}-CB{tag}"] = {"order": q, "matrix": op.matrix, "x": op.x,
                                      "filter": None}
        op = ba.build_operator(ba.scheme_kind(f"C{q}"), q, npts)
        cat[f"C{q}"] = {"order": q, "matrix": op.matrix, "x": op.x, "filter": None}
        opsbp = ba.build_operator(ba.scheme_kind(f"SBP{q}"), q, npts)
        cat[f"SBP{q}"] = {"order": q, "matrix": opsbp.matrix, "x": opsbp.x, "filter": None}
    filt = ba.visbal_gaitonde_filter_matrix(npts, ALPHA_F)
    compact_baseline6 = ba.build_operator_extended("C6-CB3", 6, npts)
    cat["C6F-CB3"] = {"order": 6, "matrix": compact_baseline6.matrix, "x": compact_baseline6.x, "filter": filt}
    c6f = ba.build_operator_extended("C6F", 6, npts)
    cat["C6F"] = {"order": 6, "matrix": c6f.matrix, "x": c6f.x, "filter": filt}
    bo = ba.build_operator_extended("SBP6-BO", 6, npts)
    cat["SBP6-BO"] = {"order": 6, "matrix": bo.matrix, "x": bo.x, "filter": None}
    d_e6, _w, cid_e6 = bl.csb_e6_operator(npts)
    d_t6, _w2, cid_t6 = bl.csb_t6_operator(npts)
    cat["CSB-E6"] = {"order": 6, "matrix": d_e6, "x": np.linspace(0.0, 1.0, npts),
                    "filter": None, "candidate": cid_e6}
    cat["CSB-T6"] = {"order": 6, "matrix": d_t6, "x": np.linspace(0.0, 1.0, npts),
                    "filter": None, "candidate": cid_t6}
    return cat


def run_all() -> dict:
    out = {
           "config": {"t_end": T_END, "cfl": CFL, "mode": MODE, "strip": STRIP,
                      "grids": list(GRIDS), "stable_tolerance": STABLE_TOL,
                      "wallmode_npts": WALLMODE_NPTS, "taus": list(TAUS),
                      "filter_alpha": ALPHA_F},
           "compact_baseline_closures": compact_baseline_closure_record(),
           "sat_energy": sat_energy_record(),
           "narrow_d2": narrow_d2_record(),
           "abscissa": {}, "six_transit": {},
           "wall_spectrum_thresholds": wallmode_record(),
           "closed_box_across_order": {}}

    for npts in GRIDS:
        cat = variant_catalog(npts)
        block = {name: matrix_abscissa(rec["matrix"]) for name, rec in cat.items()}
        for tau in TAUS:
            block[f"SBP6-SAT(tau={tau:g})"] = sat_abscissa(npts, 6, tau)
        for q in (4, 6, 8):
            for kind, tag in (("E", "E"), ("DRP", "DRP"), ("B", "BMF"), ("I", "BMG")):
                block[f"{tag}{q}"] = spectral_abscissa(kind, q, npts)
        out["abscissa"][f"N{npts}"] = {k: float(v) for k, v in sorted(block.items())}

    npts = GRIDS[0]
    cat = variant_catalog(npts)
    six: dict[str, dict] = {}
    for name, rec in sorted(cat.items()):
        abscissa = out["abscissa"][f"N{npts}"][name]
        entry = march_nodal(rec["x"], rec["matrix"], filter_matrix=rec["filter"])
        entry["abscissa_N41"] = float(abscissa)
        entry["abscissa_N81"] = float(out["abscissa"][f"N{GRIDS[1]}"][name])
        entry["stable"] = bool(max(abscissa, out["abscissa"][f"N{GRIDS[1]}"][name]) <= STABLE_TOL)
        entry["order"] = rec["order"]
        six[name] = entry
    for tau in TAUS:
        name = f"SBP6-SAT(tau={tau:g})"
        entry = march_sat(npts, 6, tau)
        entry["abscissa_N41"] = float(out["abscissa"][f"N{npts}"][name])
        entry["abscissa_N81"] = float(out["abscissa"][f"N{GRIDS[1]}"][name])
        entry["stable"] = bool(max(entry["abscissa_N41"], entry["abscissa_N81"]) <= STABLE_TOL)
        entry["order"] = 6
        six[name] = entry
    for q in (4, 6, 8):
        for kind, tag in (("B", "BMF"), ("I", "BMG"), ("E", "E"), ("DRP", "DRP")):
            op = ba.build_operator(kind, q, npts)
            entry = march_operator(op)
            entry["abscissa_N41"] = float(out["abscissa"][f"N{npts}"][f"{tag}{q}"])
            entry["abscissa_N81"] = float(out["abscissa"][f"N{GRIDS[1]}"][f"{tag}{q}"])
            entry["stable"] = bool(max(entry["abscissa_N41"], entry["abscissa_N81"]) <= STABLE_TOL)
            entry["order"] = q
            six[f"{tag}{q}"] = entry
    out["six_transit"] = six

    splines = {4: ("BMF4", "BMG4"), 6: ("BMF6", "BMG6"), 8: ("BMF8", "BMG8")}
    for q in (4, 6, 8):
        spline_pool = {n: six[n] for n in splines[q] if n in six}
        best_spline = min(spline_pool.items(), key=lambda kv: kv[1]["trimmed"])
        comp_pool = {n: r for n, r in six.items()
                     if r["order"] == q and r["stable"] and not n.startswith(("BMF", "BMG"))}
        best_comp = (min(comp_pool.items(), key=lambda kv: kv[1]["trimmed"])
                     if comp_pool else (None, None))
        out["closed_box_across_order"][f"q{q}"] = {
            "spline": {"family": best_spline[0],
                       "trimmed": best_spline[1]["trimmed"],
                       "untrimmed": best_spline[1]["untrimmed"]},
            "competitor": ({"family": best_comp[0],
                            "trimmed": best_comp[1]["trimmed"],
                            "untrimmed": best_comp[1]["untrimmed"]}
                           if best_comp[0] is not None else None),
            "stable_competitors": sorted(comp_pool),
            "spline_leads": (best_comp[0] is None
                             or bool(best_spline[1]["trimmed"] < best_comp[1]["trimmed"])),
        }
    return out


def verify(out: dict, tol: float = 1.0e-10) -> None:
    fresh = run_all()

    def walk(a, b, path=""):
        if isinstance(a, dict):
            assert set(a) == set(b), f"key mismatch at {path}: {set(a) ^ set(b)}"
            for k in a:
                walk(a[k], b[k], f"{path}/{k}")
        elif isinstance(a, list):
            assert len(a) == len(b), f"length mismatch at {path}"
            for i, (u, v) in enumerate(zip(a, b)):
                walk(u, v, f"{path}[{i}]")
        elif isinstance(a, bool) or isinstance(b, bool):
            assert a == b, f"flag mismatch at {path}: {a} vs {b}"
        elif isinstance(a, (int, float)):
            if not (np.isnan(a) and np.isnan(b)):
                scale = max(1.0, abs(a), abs(b))
                assert abs(a - b) <= tol * scale, f"value mismatch at {path}: {a} vs {b}"
        else:
            assert a == b, f"mismatch at {path}: {a} vs {b}"

    walk(out, fresh)
    print(f"wall_fairness verify: ALL PASS at {tol:g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(RESULTSDIR / "wall_fairness_summary.json"))
    ap.add_argument("--verify", default=None,
                    help="recompute everything and compare against this artifact")
    args = ap.parse_args()
    ensure_generated_dirs()
    if args.verify:
        verify(json.loads(Path(args.verify).read_text()))
        return
    out = run_all()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print("abscissa N=41:", {k: f"{v:.2e}" for k, v in out["abscissa"]["N41"].items()})
    print("six-transit trimmed:",
          {k: f"{v['trimmed']:.3e}" for k, v in out["six_transit"].items()})
    for q in (4, 6, 8):
        rec = out["closed_box_across_order"][f"q{q}"]
        comp = rec["competitor"]
        print(f"q={q}: spline {rec['spline']['family']} {rec['spline']['trimmed']:.2e} / "
              f"competitor {comp['family'] if comp else 'none'} "
              f"{comp['trimmed']:.2e}" if comp else f"q={q}: no stable competitor")
    print("wrote", path)


if __name__ == "__main__":
    main()
