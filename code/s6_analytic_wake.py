#!/usr/bin/env python3
"""Unforced incompressible Navier-Stokes analytic wake benchmark.

The driver advances the vorticity-streamfunction form of the two-dimensional
incompressible Navier-Stokes equations to the classical steady analytic wake.
No manufactured forcing is used: the exact solution supplies the inflow,
outflow, lateral boundary traces, and the final reference field.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np
from scipy.linalg import lu_factor, lu_solve

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import boundary_spline_analysis as ba
from project_paths import (
    FIGDIR,
    RESULTSDIR,
    ensure_generated_dirs,
    recommended_process_workers,
    save_figure,
)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.0,
        "axes.labelsize": 10.8,
        "axes.titlesize": 11.4,
        "legend.fontsize": 9.6,
        "xtick.labelsize": 9.4,
        "ytick.labelsize": 9.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.14,
        "grid.linewidth": 0.45,
    }
)


SCHEMES = tuple(ba.comparison_labels(6))
REYNOLDS = (20, 40, 80)
NPTS = 25
DIAG_NPTS = 129
TOLERANCE = 1.0e-8
LENGTH = 1.0
CENTER = 0.5


@dataclass(frozen=True)
class WakeConfig:
    npts: int = NPTS
    diag_npts: int = DIAG_NPTS
    tolerance: float = TOLERANCE


def time_settings(reynolds: int) -> tuple[float, float]:
    """Return the stable explicit-advection step and horizon."""
    if reynolds >= 80:
        return 0.005, 5.0
    return 0.010, 4.0


def analytic_wake_lambda(reynolds: float) -> float:
    re = float(reynolds)
    return float(0.5 * (re - np.sqrt(re * re + 16.0 * np.pi * np.pi)))


def _xy_mesh(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(np.asarray(x, dtype=float), np.asarray(y, dtype=float), indexing="ij")


def exact_fields(x: np.ndarray, y: np.ndarray, reynolds: float) -> dict[str, np.ndarray]:
    """Return the analytic-wake velocity, pressure, streamfunction, and vorticity."""
    lam = analytic_wake_lambda(reynolds)
    xg, yg = _xy_mesh(x, y)
    expx = np.exp(lam * xg)
    siny = np.sin(2.0 * np.pi * yg)
    cosy = np.cos(2.0 * np.pi * yg)
    return {
        "u": 1.0 - expx * cosy,
        "v": (lam / (2.0 * np.pi)) * expx * siny,
        "p": 0.5 * (1.0 - np.exp(2.0 * lam * xg)),
        "psi": yg - (expx / (2.0 * np.pi)) * siny,
        "omega": ((lam * lam) / (2.0 * np.pi) - 2.0 * np.pi) * expx * siny,
    }


def exact_pairs(x: np.ndarray, y: np.ndarray, reynolds: float) -> dict[str, np.ndarray]:
    """Return analytic-wake fields at paired point coordinates."""
    xp = np.asarray(x, dtype=float)
    yp = np.asarray(y, dtype=float)
    lam = analytic_wake_lambda(reynolds)
    expx = np.exp(lam * xp)
    siny = np.sin(2.0 * np.pi * yp)
    cosy = np.cos(2.0 * np.pi * yp)
    return {
        "u": 1.0 - expx * cosy,
        "v": (lam / (2.0 * np.pi)) * expx * siny,
        "psi": yp - (expx / (2.0 * np.pi)) * siny,
        "omega": ((lam * lam) / (2.0 * np.pi) - 2.0 * np.pi) * expx * siny,
    }


def derivative_matrices(op: ba.Operator1D) -> tuple[np.ndarray, np.ndarray]:
    d1 = np.asarray(op.matrix, dtype=float)
    try:
        d2 = np.asarray(op.second_derivative_matrix(), dtype=float)
    except ValueError:
        d2 = d1 @ d1
    return d1, d2


def axis_eval_matrix(op: ba.Operator1D, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    family = ba.scheme_family(op.name)
    if family in ("BMF", "BMG"):
        _, phi, _ = ba.spline_eval_matrices(points, op.x, op.order, getattr(op, "knots"))
        return phi @ np.linalg.inv(getattr(op, "bmat", getattr(op, "_bmat")))
    return ba.local_interpolation_matrix(op.x, points, op.order + 1)


def interp2(op: ba.Operator1D, field: np.ndarray, points: np.ndarray) -> np.ndarray:
    w = axis_eval_matrix(op, points)
    return w @ field @ w.T


def interp_points(
    op: ba.Operator1D,
    field: np.ndarray,
    xp: np.ndarray,
    yp: np.ndarray,
) -> np.ndarray:
    wx = axis_eval_matrix(op, xp)
    wy = axis_eval_matrix(op, yp)
    return np.einsum("mi,ij,mj->m", wx, field, wy, optimize=True)


def boundary_only(field: np.ndarray) -> np.ndarray:
    out = np.zeros_like(field)
    out[0, :] = field[0, :]
    out[-1, :] = field[-1, :]
    out[:, 0] = field[:, 0]
    out[:, -1] = field[:, -1]
    return out


def impose_velocity_boundary(
    u: np.ndarray,
    v: np.ndarray,
    u_exact: np.ndarray,
    v_exact: np.ndarray,
) -> None:
    u[0, :] = u_exact[0, :]
    u[-1, :] = u_exact[-1, :]
    u[:, 0] = u_exact[:, 0]
    u[:, -1] = u_exact[:, -1]
    v[0, :] = v_exact[0, :]
    v[-1, :] = v_exact[-1, :]
    v[:, 0] = v_exact[:, 0]
    v[:, -1] = v_exact[:, -1]


def solve_streamfunction(
    omega: np.ndarray,
    psi_boundary: np.ndarray,
    boundary_laplacian: np.ndarray,
    lu_poisson: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    npts = omega.shape[0]
    interior_shape = (npts - 2, npts - 2)
    rhs = -omega[1:-1, 1:-1].reshape(-1) - boundary_laplacian
    psi = psi_boundary.copy()
    psi[1:-1, 1:-1] = lu_solve(lu_poisson, rhs).reshape(interior_shape)
    return psi


def advance_solution(label: str, reynolds: int, cfg: WakeConfig) -> dict:
    start = time.time()
    cpu_start = time.process_time()
    op = ba.build_operator(ba.scheme_kind(label), ba.scheme_order(label), cfg.npts, length=LENGTH)
    d1, d2 = derivative_matrices(op)
    x = op.x
    exact = exact_fields(x, x, reynolds)
    nu = 1.0 / float(reynolds)
    dt, t_end = time_settings(reynolds)

    npts = x.size
    interior = np.arange(1, npts - 1)
    d2_int = d2[np.ix_(interior, interior)]
    eye_int = np.eye(npts - 2)
    lap_int = np.kron(eye_int, d2_int) + np.kron(d2_int, eye_int)
    lu_poisson = lu_factor(lap_int)
    lu_diffusion = lu_factor(np.eye(lap_int.shape[0]) - dt * nu * lap_int)

    psi_boundary = boundary_only(exact["psi"])
    omega_boundary = boundary_only(exact["omega"])
    psi_boundary_lap = (d2 @ psi_boundary + psi_boundary @ d2.T)[1:-1, 1:-1].reshape(-1)
    omega_boundary_lap = (d2 @ omega_boundary + omega_boundary @ d2.T)[1:-1, 1:-1].reshape(-1)

    omega = omega_boundary.copy()
    max_steps = int(np.ceil(t_end / dt))
    update_inf = np.inf
    completed_steps = max_steps

    for step in range(max_steps):
        psi = solve_streamfunction(omega, psi_boundary, psi_boundary_lap, lu_poisson)
        u = psi @ d1.T
        v = -(d1 @ psi)
        adv = u * (d1 @ omega) + v * (omega @ d1.T)
        old = omega[1:-1, 1:-1].reshape(-1)
        rhs = old + dt * (-adv[1:-1, 1:-1].reshape(-1) + nu * omega_boundary_lap)
        if not np.all(np.isfinite(rhs)):
            raise FloatingPointError(f"{label} Re={reynolds} produced a non-finite vorticity RHS")
        new = lu_solve(lu_diffusion, rhs)
        update_inf = float(np.max(np.abs(new - old)))
        omega[1:-1, 1:-1] = new.reshape((npts - 2, npts - 2))
        completed_steps = step + 1
        if step > 50 and update_inf < cfg.tolerance:
            break

    psi = solve_streamfunction(omega, psi_boundary, psi_boundary_lap, lu_poisson)
    u = psi @ d1.T
    v = -(d1 @ psi)
    impose_velocity_boundary(u, v, exact["u"], exact["v"])
    elapsed = time.time() - start
    cpu_elapsed = time.process_time() - cpu_start

    return {
        "op": op,
        "d1": d1,
        "d2": d2,
        "x": x,
        "u": u,
        "v": v,
        "psi": psi,
        "omega": omega,
        "steps": int(completed_steps),
        "dt": float(dt),
        "t_end": float(t_end),
        "update_inf": float(update_inf),
        "elapsed_s": float(elapsed),
        "cpu_s": float(cpu_elapsed),
    }


def velocity_metrics(
    op: ba.Operator1D,
    u: np.ndarray,
    v: np.ndarray,
    reynolds: int,
    cfg: WakeConfig,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = np.linspace(0.0, LENGTH, cfg.diag_npts)
    weights = ba.trapezoidal_weights(q.size, LENGTH)
    w2 = np.outer(weights, weights)
    u_diag = interp2(op, u, q)
    v_diag = interp2(op, v, q)
    exact = exact_fields(q, q, reynolds)
    err2 = (u_diag - exact["u"]) ** 2 + (v_diag - exact["v"]) ** 2
    ref2 = exact["u"] ** 2 + exact["v"] ** 2
    rel_l2 = float(np.sqrt(np.sum(w2 * err2)) / max(np.sqrt(np.sum(w2 * ref2)), 1.0e-30))
    linf = float(np.max(np.sqrt(err2)))
    return rel_l2, linf, q, u_diag, v_diag, np.sqrt(err2)


def momentum_residual(
    d1: np.ndarray,
    d2: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    reynolds: int,
) -> float:
    nu = 1.0 / float(reynolds)
    ux = d1 @ u
    uy = u @ d1.T
    vx = d1 @ v
    vy = v @ d1.T
    px = d1 @ p
    py = p @ d1.T
    lap_u = d2 @ u + u @ d2.T
    lap_v = d2 @ v + v @ d2.T
    ru = u * ux + v * uy + px - nu * lap_u
    rv = u * vx + v * vy + py - nu * lap_v
    residual = np.sqrt(ru[1:-1, 1:-1] ** 2 + rv[1:-1, 1:-1] ** 2)
    reference = np.sqrt((u * ux + v * uy)[1:-1, 1:-1] ** 2 + (u * vx + v * vy)[1:-1, 1:-1] ** 2)
    return float(np.sqrt(np.mean(residual**2)) / max(np.sqrt(np.mean(reference**2)), 1.0e-30))


def run_case(label: str, reynolds: int, cfg: WakeConfig) -> dict:
    state = advance_solution(label, reynolds, cfg)
    op = state["op"]
    d1 = state["d1"]
    d2 = state["d2"]
    x = state["x"]
    u = state["u"]
    v = state["v"]
    exact_native = exact_fields(x, x, reynolds)
    rel_l2, linf, q, _u_diag, _v_diag, _err_diag = velocity_metrics(op, u, v, reynolds, cfg)
    div = d1 @ u + v @ d1.T
    speed_ref = float(np.max(np.sqrt(exact_native["u"] ** 2 + exact_native["v"] ** 2)))
    div_inf = float(np.max(np.abs(div)) / max(speed_ref, 1.0e-30))
    residual = momentum_residual(d1, d2, u, v, exact_native["p"], reynolds)

    yline = q
    xline = q
    u_mid = interp_points(op, u, np.full_like(yline, CENTER), yline)
    v_mid = interp_points(op, v, xline, np.full_like(xline, CENTER))
    exact_u_mid = exact_pairs(np.full_like(yline, CENTER), yline, reynolds)["u"]
    exact_v_mid = exact_pairs(xline, np.full_like(xline, CENTER), reynolds)["v"]

    return {
        "label": label,
        "family": ba.scheme_family(label),
        "display": ba.scheme_display_name(label),
        "reynolds": int(reynolds),
        "npts": int(cfg.npts),
        "diag_npts": int(cfg.diag_npts),
        "steps": state["steps"],
        "dt": state["dt"],
        "t_end": state["t_end"],
        "update_inf": state["update_inf"],
        "work_per_point": float(op.work_model.total_flops / op.x.size),
        "velocity_rel_l2": rel_l2,
        "velocity_linf": linf,
        "divergence_inf_scaled": div_inf,
        "momentum_residual_rel_rms": residual,
        "elapsed_s": state["elapsed_s"],
        "cpu_s": state["cpu_s"],
        "centerline": {
            "s": q.tolist(),
            "u_xhalf": u_mid.tolist(),
            "u_xhalf_exact": exact_u_mid.tolist(),
            "v_yhalf": v_mid.tolist(),
            "v_yhalf_exact": exact_v_mid.tolist(),
        },
    }


def run_all(
    cfg: WakeConfig,
    reynolds_values: tuple[int, ...] = REYNOLDS,
    max_workers: int | None = None,
) -> list[dict]:
    jobs = [(label, reynolds) for reynolds in reynolds_values for label in SCHEMES]
    if max_workers is None:
        max_workers = recommended_process_workers(len(jobs), cap=6)
    results: list[dict] = []
    print(f"== running {len(jobs)} analytic wake jobs with {max_workers} worker(s) ==")
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_case, label, reynolds, cfg): (label, reynolds) for label, reynolds in jobs}
        for fut in as_completed(futures):
            label, reynolds = futures[fut]
            row = fut.result()
            results.append(row)
            print(
                f"  Re={reynolds:<3d} {label:>5s}: steps={row['steps']:4d} "
                f"vel={row['velocity_rel_l2']:.3e} div={row['divergence_inf_scaled']:.3e} "
                f"res={row['momentum_residual_rel_rms']:.3e}"
            )
    return sorted(results, key=lambda r: (r["reynolds"], SCHEMES.index(r["label"])))


def summarize(results: list[dict], cfg: WakeConfig, reynolds_values: tuple[int, ...]) -> dict:
    by_re: dict[str, list[dict]] = {}
    for row in results:
        by_re.setdefault(str(row["reynolds"]), []).append(row)
    winners = {}
    for reynolds, rows in by_re.items():
        winners[reynolds] = {
            "velocity": min(rows, key=lambda r: r["velocity_rel_l2"])["label"],
            "pointwise": min(rows, key=lambda r: r["velocity_linf"])["label"],
            "divergence": min(rows, key=lambda r: r["divergence_inf_scaled"])["label"],
            "momentum_residual": min(rows, key=lambda r: r["momentum_residual_rel_rms"])["label"],
        }
    return {
        "configuration": {
            "schemes": list(SCHEMES),
            "reynolds": list(reynolds_values),
            "npts": cfg.npts,
            "diag_npts": cfg.diag_npts,
            "domain": [0.0, LENGTH, 0.0, LENGTH],
            "centerline": CENTER,
            "tolerance": cfg.tolerance,
            "lambda": {str(reynolds): analytic_wake_lambda(reynolds) for reynolds in reynolds_values},
            "time_settings": {
                str(reynolds): {"dt": time_settings(reynolds)[0], "t_end": time_settings(reynolds)[1]}
                for reynolds in reynolds_values
            },
        },
        "winners": winners,
        "results": results,
    }


def _style(label: str) -> dict:
    fam = ba.scheme_family(label)
    return {
        "color": ba.DEFAULT_COLORS[fam],
        "marker": ba.DEFAULT_MARKERS[fam],
        "linestyle": ba.DEFAULT_LINESTYLES[fam],
        "label": ba.scheme_display_name(label),
    }


def make_diagnostics_figure(summary: dict) -> None:
    rows = summary["results"]
    re_vals = [int(r) for r in summary["configuration"]["reynolds"]]
    fig, axs = plt.subplots(2, 2, figsize=(11.2, 7.25), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axs.ravel()

    for label in SCHEMES:
        data = [next(r for r in rows if r["label"] == label and r["reynolds"] == reynolds) for reynolds in re_vals]
        st = _style(label)
        ax1.semilogy(
            re_vals,
            [r["velocity_rel_l2"] for r in data],
            color=st["color"],
            marker=st["marker"],
            ls=st["linestyle"],
            lw=2.0,
            ms=5.5,
        )
        ax2.semilogy(
            re_vals,
            [r["divergence_inf_scaled"] for r in data],
            color=st["color"],
            marker=st["marker"],
            ls=st["linestyle"],
            lw=2.0,
            ms=5.5,
        )
        ax4.loglog(
            [r["cpu_s"] for r in data],
            [r["velocity_rel_l2"] for r in data],
            color=st["color"],
            marker=st["marker"],
            ls="none",
            ms=6.2,
        )

    ax1.set_xlabel(r"Reynolds number")
    ax1.set_ylabel(r"final velocity relative $L^2$")
    ax1.set_title(r"(a) Wake accuracy")
    ax1.set_xticks(re_vals)

    ax2.set_xlabel(r"Reynolds number")
    ax2.set_ylabel(r"native $\|\nabla_h\cdot\vec u_h\|_\infty/\|\vec u_{\rm ex}\|_\infty$")
    ax2.set_title(r"(b) Discrete incompressibility")
    ax2.set_xticks(re_vals)

    re_profile = 40
    y = np.asarray(next(r for r in rows if r["label"] == "BMG6" and r["reynolds"] == re_profile)["centerline"]["s"])
    for label in SCHEMES:
        row = next(r for r in rows if r["label"] == label and r["reynolds"] == re_profile)
        st = _style(label)
        u_err = np.abs(
            np.asarray(row["centerline"]["u_xhalf"]) - np.asarray(row["centerline"]["u_xhalf_exact"])
        )
        ax3.semilogx(
            np.maximum(u_err, 1.0e-14),
            y,
            color=st["color"],
            ls=st["linestyle"],
            lw=2.0,
        )
    ax3.set_xlabel(r"$|u_h(1/2,y)-u_{\rm ex}(1/2,y)|$")
    ax3.set_ylabel(r"$y$")
    ax3.set_title(r"(c) Vertical centerline error, $\mathrm{Re}=40$")
    ax3.set_xlim(1.0e-11, 2.0e-2)

    ax4.set_xlabel(r"measured CPU seconds per solve (single thread)")
    ax4.set_ylabel(r"final velocity relative $L^2$")
    ax4.set_title(r"(d) Cost-accuracy placement")

    handles = [
        Line2D(
            [0],
            [0],
            color=ba.DEFAULT_COLORS[ba.scheme_family(label)],
            marker=ba.DEFAULT_MARKERS[ba.scheme_family(label)],
            ls=ba.DEFAULT_LINESTYLES[ba.scheme_family(label)],
            lw=2.0,
            label=ba.scheme_display_name(label),
        )
        for label in SCHEMES
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.035))
    save_figure(fig, "fig_s6_analytic_wake")
    plt.close(fig)


def make_flow_figure(summary: dict, cfg: WakeConfig) -> None:
    label = "BMG6"
    reynolds = 40
    row = next(r for r in summary["results"] if r["label"] == label and r["reynolds"] == reynolds)
    state = advance_solution(label, reynolds, cfg)
    op = state["op"]
    q = np.linspace(0.0, LENGTH, cfg.diag_npts)
    u_diag = interp2(op, state["u"], q)
    v_diag = interp2(op, state["v"], q)
    omega_diag = interp2(op, state["omega"], q)
    exact = exact_fields(q, q, reynolds)
    vel_error = np.sqrt((u_diag - exact["u"]) ** 2 + (v_diag - exact["v"]) ** 2)

    vmax = float(np.nanpercentile(np.abs(omega_diag), 99.5))
    vmax = max(vmax, float(np.max(np.abs(omega_diag))), 1.0e-12)
    err_max = float(np.max(vel_error))
    err_floor = max(err_max * 1.0e-5, 1.0e-13)
    err_plot = np.maximum(vel_error, err_floor)
    exponent = int(np.floor(np.log10(row["velocity_rel_l2"])))
    mantissa = row["velocity_rel_l2"] / (10.0**exponent)
    rel_l2_label = rf"{mantissa:.1f}\times 10^{{{exponent}}}"

    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
    ax0, ax1 = axs
    extent = (0.0, LENGTH, 0.0, LENGTH)

    im0 = ax0.imshow(
        omega_diag.T,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        interpolation="bicubic",
        rasterized=True,
    )
    ax0.streamplot(
        q,
        q,
        u_diag.T,
        v_diag.T,
        color="0.05",
        density=1.35,
        linewidth=0.55,
        arrowsize=0.65,
        minlength=0.05,
        zorder=4,
    )
    ax0.set_title(r"(a) $\mathrm{BMG}_6$ vorticity and streamlines")
    cb0 = fig.colorbar(im0, ax=ax0, shrink=0.88, pad=0.02)
    cb0.set_label(r"$\omega_h$")

    im1 = ax1.imshow(
        err_plot.T,
        origin="lower",
        extent=extent,
        cmap="magma",
        norm=LogNorm(vmin=err_floor, vmax=max(err_max, 10.0 * err_floor)),
        interpolation="bilinear",
        rasterized=True,
    )
    ax1.set_title(
        rf"(b) $\mathrm{{BMG}}_6$ velocity-error magnitude "
        rf"($\mathrm{{Re}}={reynolds}$, $T={row['t_end']:.1f}$, rel. $L^2={rel_l2_label}$)"
    )
    cb1 = fig.colorbar(im1, ax=ax1, shrink=0.88, pad=0.02)
    cb1.set_label(r"$\|\vec u_h-\vec u_{\rm ex}\|_2$")

    for ax in axs:
        ax.add_patch(Rectangle((0.0, 0.0), LENGTH, LENGTH, fill=False, ec="black", lw=1.7, zorder=6))
        ax.set_xlim(0.0, LENGTH)
        ax.set_ylim(0.0, LENGTH)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.grid(False)

    save_figure(fig, "fig_s6_analytic_wake_bmg_flow")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller grid and only Re=40 for smoke checking.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Write the JSON summary but skip figure generation.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum worker processes. Defaults to a conservative core count.",
    )
    parser.add_argument(
        "--from-summary",
        action="store_true",
        help=("Regenerate fig_s6_analytic_wake from the stored summary instead of "
              "rerunning the solves. The summary is read only and is not rewritten; "
              "every plotted field already lives in it."),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_generated_dirs()
    if args.from_summary:
        out = RESULTSDIR / "analytic_wake_summary.json"
        summary = json.loads(out.read_text())
        print(f"== replotting the diagnostics figure from {out} ==")
        make_diagnostics_figure(summary)
        print(f"wrote {FIGDIR / 'fig_s6_analytic_wake.pdf'}")
        return 0
    cfg = WakeConfig(npts=21, diag_npts=81) if args.quick else WakeConfig()
    reynolds_values = (40,) if args.quick else REYNOLDS
    results = run_all(cfg, reynolds_values=reynolds_values, max_workers=args.max_workers)
    summary = summarize(results, cfg, reynolds_values)
    out = RESULTSDIR / "analytic_wake_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    if not args.no_figures:
        make_diagnostics_figure(summary)
        print(f"wrote {FIGDIR / 'fig_s6_analytic_wake.pdf'}")
        make_flow_figure(summary, cfg)
        print(f"wrote {FIGDIR / 'fig_s6_analytic_wake_bmg_flow.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
