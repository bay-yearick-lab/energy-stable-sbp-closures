"""Section 6.10 acoustic cavity benchmark: three-dimensional acoustic cavity
benchmark with an interior observation sphere, driven by a superposition of
box eigenmodes.

This driver is the culminating numerical benchmark of the manuscript.
The governing system is the first-order linearised acoustic PDE on the
closed cube :math:`\\Omega = [0, 1]^3` with acoustically-soft Dirichlet
walls :math:`p = 0` on every face. A smooth observation sphere is held at a
fixed location inside the cavity, and a superposition of soft-wall
sine-eigenmodes of the cube provides both the initial condition and the
closed-form reference against which each scheme is scored at every instant
of the run. Because the exact solution is known at every :math:`(x, y, z, t)`,
the benchmark simultaneously tests:

* 3D volumetric accuracy of the tensor-product derivative operator;
* accuracy of the off-grid sphere-surface reconstruction pipeline at the
  surface nodes (pressure, normal velocity, traction);
* accuracy of the integrated net force :math:`\\vec F(t) = -\\!\\oint p\\,\\vec n\\,dS`
  over the full interval;
* accuracy and stability of the total semi-discrete acoustic energy and
  its companion energy-defect diagnostic.

The accompanying MMS appendix driver ``make_scattering_verify.py`` runs
the same code path across a sequence of grids and records the measured
convergence slopes of the volumetric, surface, and integrated-force
errors. The two are designed so that any issue in the operator assembly,
surface interpolation, or force quadrature shows up in the convergence
appendix before the cavity run at a fixed production grid is evaluated
for its scheme-vs.-scheme ranking.

Publication-grade figures produced here:

* ``fig_s6_scattering_cavity_setup`` -- cavity cartoon with sphere and
  eigenmode drive.
* ``fig_s6_scattering_cavity_snapshots`` -- six-family mid-plane pressure
  snapshots at three representative times against the exact solution.
* ``fig_s6_scattering_cavity_diagnostics`` -- six-panel comparison: field
  error history, sphere-surface pressure error, integrated force error,
  pointwise receiver trace error, energy history, and final surface map.
* ``scattering_cavity.{gif,mp4}`` -- optional mid-plane animation of the
  best scheme over the full run, for the GitHub repository showcase.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Pin BLAS threads before numpy import so process-pool workers don't
# over-subscribe physical cores when each worker spawns its own thread pool.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

import boundary_spline_analysis as ba
from boundary_spline_analysis import scheme_display_name
import scattering_core as sc
from project_paths import FIGDIR, RESULTSDIR, ensure_generated_dirs, recommended_process_workers, save_figure


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.4,
    "axes.labelsize": 11.4,
    "axes.titlesize": 12.2,
    "legend.fontsize": 10.6,
    "xtick.labelsize": 9.4,
    "ytick.labelsize": 9.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.12,
    "grid.linewidth": 0.45,
})


SCHEMES = ba.comparison_labels(6)
C_SOUND = 1.0
RHO0 = 1.0
NPTS = 49
T_END = 0.60
CFL = 0.12
MMS_MODES = [
    (2, 1, 1, 1.00),
    (1, 2, 1, 0.70),
    (1, 1, 2, 0.50),
]
SPHERE_CENTER = (0.42, 0.48, 0.52)
SPHERE_RADIUS = 0.15
N_SURFACE_NODES = 1600
RECEIVER = (NPTS - 2, NPTS - 2, NPTS - 2)
SNAPSHOT_TIMES = (0.15, 0.35, 0.55)
RECORD_CADENCE = 2


def _mms() -> sc.BoxEigenmode:
    return sc.BoxEigenmode(modes=MMS_MODES, c=C_SOUND, rho0=RHO0, length=1.0)


def _run_bmf_staggered_scheme(label: str) -> dict:
    mms = _mms()
    t_start = time.time()
    op = sc.BMFStaggeredOperator3D(label, NPTS)
    x = op.x
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    state = sc.bmf_staggered_exact_state(op, mms, 0.0)
    sphere = sc.build_sphere_surface(op, SPHERE_CENTER, SPHERE_RADIUS, N_SURFACE_NODES)

    dt = CFL * op.h / C_SOUND
    nsteps = int(np.ceil(T_END / dt))
    dt = T_END / nsteps

    times: list[float] = []
    rel_l2_field: list[float] = []
    rel_l2_surf: list[float] = []
    force_err_norm: list[float] = []
    force_num_hist: list[np.ndarray] = []
    force_exact_hist: list[np.ndarray] = []
    receiver_num: list[float] = []
    receiver_exact: list[float] = []
    energy_num: list[float] = []
    energy_exact: list[float] = []

    snap_target = np.asarray(SNAPSHOT_TIMES, dtype=float)
    snap_done = np.zeros(snap_target.size, dtype=bool)
    snap_num: list[np.ndarray] = []
    snap_exact: list[np.ndarray] = []
    snap_surf_num: list[np.ndarray] = []
    snap_surf_exact: list[np.ndarray] = []
    snap_times_out: list[float] = []

    def _record(t: float) -> None:
        P_ex = mms.pressure(X, Y, Z, t)
        err = state.p - P_ex
        denom = float(np.linalg.norm(P_ex))
        times.append(t)
        rel_l2_field.append(float(np.linalg.norm(err) / max(denom, 1.0e-30)))
        p_surf = sphere.interpolate(state.p)
        p_surf_ex = mms.pressure_on_sphere(sphere.points, t)
        denom_s = float(np.linalg.norm(p_surf_ex))
        rel_l2_surf.append(float(np.linalg.norm(p_surf - p_surf_ex) / max(denom_s, 1.0e-30)))
        F_num = sc.integrated_force(sphere, p_surf)
        F_ex = mms.integrated_force_on_sphere(sphere, t)
        force_num_hist.append(F_num)
        force_exact_hist.append(F_ex)
        force_err_norm.append(float(np.linalg.norm(F_num - F_ex)))
        ri, rj, rk = RECEIVER
        receiver_num.append(float(state.p[ri, rj, rk]))
        receiver_exact.append(float(P_ex[ri, rj, rk]))
        energy_num.append(sc.bmf_staggered_energy(state, op, RHO0, C_SOUND))
        exact_state = sc.bmf_staggered_exact_state(op, mms, t)
        energy_exact.append(sc.bmf_staggered_energy(exact_state, op, RHO0, C_SOUND))
        for i, st in enumerate(snap_target):
            if not snap_done[i] and t + 1.0e-12 >= st:
                snap_num.append(state.p.copy())
                snap_exact.append(P_ex.copy())
                snap_surf_num.append(p_surf.copy())
                snap_surf_exact.append(p_surf_ex.copy())
                snap_times_out.append(t)
                snap_done[i] = True

    _record(0.0)
    for step in range(nsteps):
        sc.bmf_staggered_rk4_step(state, dt, op, RHO0, C_SOUND)
        if ((step + 1) % RECORD_CADENCE == 0) or (step + 1 == nsteps):
            _record((step + 1) * dt)

    elapsed = time.time() - t_start
    mid = NPTS // 2
    return {
        "label": label,
        "elapsed_s": float(elapsed),
        "nsteps": int(nsteps),
        "dt": float(dt),
        "h": float(op.h_nominal),
        "h_min": float(op.h),
        "family": op.family,
        "bmf_route": "staggered primal-pressure and derivative-grid velocity complex",
        "times": list(map(float, times)),
        "rel_l2_field": rel_l2_field,
        "rel_l2_surf": rel_l2_surf,
        "force_err_norm": force_err_norm,
        "force_num_hist": [F.tolist() for F in force_num_hist],
        "force_exact_hist": [F.tolist() for F in force_exact_hist],
        "receiver_num": receiver_num,
        "receiver_exact": receiver_exact,
        "energy_num": energy_num,
        "energy_exact": energy_exact,
        "snap_times": list(map(float, snap_times_out)),
        "snap_num_mid": [s[:, :, mid].copy().tolist() for s in snap_num],
        "snap_exact_mid": [s[:, :, mid].copy().tolist() for s in snap_exact],
        "snap_surf_num_final": snap_surf_num[-1].tolist() if snap_surf_num else [],
        "snap_surf_exact_final": snap_surf_exact[-1].tolist() if snap_surf_exact else [],
    }


def _run_scheme(label: str) -> dict:
    if ba.scheme_family(label) == "BMF":
        return _run_bmf_staggered_scheme(label)

    mms = _mms()
    snap = np.asarray(SNAPSHOT_TIMES, dtype=float)
    t_start = time.time()
    op = sc.Operator3D(label, NPTS)
    x = op.x
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    P0 = mms.pressure(X, Y, Z, 0.0)
    U0, V0, W0 = mms.velocity(X, Y, Z, 0.0)
    state = sc.AcousticState(p=P0.copy(), u=U0.copy(), v=V0.copy(), w=W0.copy())
    sc._apply_box_bc(state)

    sphere = sc.build_sphere_surface(op, SPHERE_CENTER, SPHERE_RADIUS, N_SURFACE_NODES)
    metric1d = op.metric_1d

    dt = CFL * op.h / C_SOUND
    nsteps = int(np.ceil(T_END / dt))
    dt = T_END / nsteps

    # Time-history recorders.
    times: list[float] = []
    rel_l2_field: list[float] = []
    rel_l2_surf: list[float] = []
    force_err_norm: list[float] = []
    force_num_hist: list[np.ndarray] = []
    force_exact_hist: list[np.ndarray] = []
    receiver_num: list[float] = []
    receiver_exact: list[float] = []
    energy_num: list[float] = []
    energy_exact: list[float] = []

    snap_target = np.asarray(SNAPSHOT_TIMES, dtype=float)
    snap_done = np.zeros(snap_target.size, dtype=bool)
    snap_num: list[np.ndarray] = []
    snap_exact: list[np.ndarray] = []
    snap_surf_num: list[np.ndarray] = []
    snap_surf_exact: list[np.ndarray] = []
    snap_times_out: list[float] = []

    def _record(t: float) -> None:
        P_ex = mms.pressure(X, Y, Z, t)
        U_ex, V_ex, W_ex = mms.velocity(X, Y, Z, t)
        err = state.p - P_ex
        denom = float(np.linalg.norm(P_ex))
        times.append(t)
        rel_l2_field.append(float(np.linalg.norm(err) / max(denom, 1.0e-30)))
        p_surf = sphere.interpolate(state.p)
        p_surf_ex = mms.pressure_on_sphere(sphere.points, t)
        denom_s = float(np.linalg.norm(p_surf_ex))
        rel_l2_surf.append(float(np.linalg.norm(p_surf - p_surf_ex) / max(denom_s, 1.0e-30)))
        F_num = sc.integrated_force(sphere, p_surf)
        F_ex = mms.integrated_force_on_sphere(sphere, t)
        force_num_hist.append(F_num)
        force_exact_hist.append(F_ex)
        force_err_norm.append(float(np.linalg.norm(F_num - F_ex)))
        ri, rj, rk = RECEIVER
        receiver_num.append(float(state.p[ri, rj, rk]))
        receiver_exact.append(float(P_ex[ri, rj, rk]))
        energy_num.append(sc.energy(state, metric1d, RHO0, C_SOUND))
        exact_state = sc.AcousticState(p=P_ex, u=U_ex, v=V_ex, w=W_ex)
        energy_exact.append(sc.energy(exact_state, metric1d, RHO0, C_SOUND))
        for i, st in enumerate(snap_target):
            if not snap_done[i] and t + 1.0e-12 >= st:
                snap_num.append(state.p.copy())
                snap_exact.append(P_ex.copy())
                snap_surf_num.append(p_surf.copy())
                snap_surf_exact.append(p_surf_ex.copy())
                snap_times_out.append(t)
                snap_done[i] = True

    _record(0.0)
    for step in range(nsteps):
        sc.rk4_step(state, dt, op, RHO0, C_SOUND, None, 0.0, "soft_pressure", None, step * dt)
        if ((step + 1) % RECORD_CADENCE == 0) or (step + 1 == nsteps):
            _record((step + 1) * dt)

    elapsed = time.time() - t_start
    mid = NPTS // 2
    return {
        "label": label,
        "elapsed_s": float(elapsed),
        "nsteps": int(nsteps),
        "dt": float(dt),
        "h": float(op.h_nominal),
        "h_min": float(op.h),
        "family": op.family,
        "times": list(map(float, times)),
        "rel_l2_field": rel_l2_field,
        "rel_l2_surf": rel_l2_surf,
        "force_err_norm": force_err_norm,
        "force_num_hist": [F.tolist() for F in force_num_hist],
        "force_exact_hist": [F.tolist() for F in force_exact_hist],
        "receiver_num": receiver_num,
        "receiver_exact": receiver_exact,
        "energy_num": energy_num,
        "energy_exact": energy_exact,
        "snap_times": list(map(float, snap_times_out)),
        "snap_num_mid": [s[:, :, mid].copy().tolist() for s in snap_num],
        "snap_exact_mid": [s[:, :, mid].copy().tolist() for s in snap_exact],
        "snap_surf_num_final": snap_surf_num[-1].tolist() if snap_surf_num else [],
        "snap_surf_exact_final": snap_surf_exact[-1].tolist() if snap_surf_exact else [],
    }


def run_cavity(max_workers: int | None = None) -> dict:
    jobs = list(SCHEMES)
    if max_workers is None:
        max_workers = recommended_process_workers(len(jobs), cap=2)
    out: dict[str, dict] = {}
    print(f"== cavity NPTS={NPTS} T_end={T_END} cfl={CFL} ==")
    print(f"== using {max_workers} worker process(es) ==")
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_run_scheme, lab): lab for lab in jobs}
        for f in as_completed(futs):
            lab = futs[f]
            r = f.result()
            out[lab] = r
            print(f"  {lab:>5s}  {r['elapsed_s']:6.1f}s  "
                  f"field_final={r['rel_l2_field'][-1]:.3e}  "
                  f"surf_final={r['rel_l2_surf'][-1]:.3e}  "
                  f"F_err_peak={max(r['force_err_norm']):.3e}  "
                  f"E_drift={(r['energy_num'][-1]-r['energy_num'][0])/r['energy_num'][0]:+.3e}")
    return out


def _mid_sphere_mask() -> np.ndarray:
    x = np.linspace(0.0, 1.0, NPTS)
    X, Y = np.meshgrid(x, x, indexing="ij")
    dx = X - SPHERE_CENTER[0]
    dy = Y - SPHERE_CENTER[1]
    return dx * dx + dy * dy <= (SPHERE_RADIUS ** 2)


def _extent() -> list[float]:
    return [0.0, 1.0, 0.0, 1.0]


def _fibonacci_lat_lon(n: int) -> np.ndarray:
    idx = np.arange(n, dtype=float) + 0.5
    theta = np.arccos(1.0 - 2.0 * idx / n)
    phi = np.mod(np.pi * (1.0 + 5.0 ** 0.5) * idx + np.pi, 2.0 * np.pi) - np.pi
    return np.stack([theta, phi], axis=1)


def make_figure_setup(stem: str) -> None:
    """Single-panel setup schematic: the top view carries all needed info.

    The earlier version had a twin ``side view`` panel that was visually
    identical because the configuration is symmetric about
    $z=\tfrac{1}{2}$; it is dropped here and the wall/medium tag is kept
    with the top view.
    """
    fig = plt.figure(figsize=(6.2, 5.8))
    gs = fig.add_gridspec(1, 1, left=0.12, right=0.96, top=0.82, bottom=0.10)
    ax_top = fig.add_subplot(gs[0, 0])

    x = np.linspace(0.0, 1.0, 240)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Z = np.full_like(X, 0.5)
    mms = _mms()
    P = mms.pressure(X, Y, Z, 0.0)
    vmax = float(np.max(np.abs(P))) * 1.05

    ax_top.imshow(P.T, origin="lower", extent=_extent(),
                  cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  alpha=0.60, interpolation="bilinear", zorder=1)
    ax_top.add_patch(Rectangle((0, 0), 1, 1, fill=False,
                               edgecolor="0.15", lw=1.8, zorder=3))
    ax_top.set_xlim(-0.04, 1.04)
    ax_top.set_ylim(-0.04, 1.04)
    ax_top.set_aspect("equal")
    ax_top.set_xticks([0.0, 0.5, 1.0])
    ax_top.set_yticks([0.0, 0.5, 1.0])
    ax_top.set_xlabel(r"$x$")
    ax_top.set_ylabel(r"$y$")

    ax_top.add_patch(Circle(SPHERE_CENTER[:2], SPHERE_RADIUS,
                            facecolor="#d8d8e4", edgecolor="#222",
                            lw=1.6, zorder=4))
    ax_top.text(SPHERE_CENTER[0], SPHERE_CENTER[1] + SPHERE_RADIUS + 0.04,
                "smooth sphere",
                ha="center", fontsize=10.4, color="#222")
    rx = RECEIVER[0] / (NPTS - 1)
    ry = RECEIVER[1] / (NPTS - 1)
    ax_top.plot(rx, ry, marker="s", color="#274a7a", ms=9, zorder=5)
    ax_top.text(rx - 0.02, ry - 0.05, "receiver",
                ha="right", va="top", fontsize=10.0, color="#274a7a")

    ax_top.text(0.5, 1.08,
                r"walls: $p|_{\partial\Omega}=0$   medium: $c=1,\ \rho_0=1$,"
                r"   $\Omega=[0,1]^3$",
                transform=ax_top.transAxes,
                ha="center", fontsize=11.0, color="0.15")
    save_figure(fig, stem)
    plt.close(fig)


def make_figure_snapshots(results: dict, stem: str) -> None:
    nrow = len(SCHEMES)
    ncol = len(SNAPSHOT_TIMES)
    fig = plt.figure(figsize=(3.35 * ncol + 0.6, 2.35 * nrow + 0.7))
    gs = fig.add_gridspec(nrow, ncol, wspace=0.08, hspace=0.18,
                          left=0.08, right=0.99, top=0.93, bottom=0.07)

    # Set the symmetric colour scale from the stable schemes only, so the
    # inter-family contrast among the five stable families is actually visible;
    # unstable families then clip visibly at the scale, which is the message.
    stable_err = []
    for lab in SCHEMES:
        final = results[lab]["rel_l2_field"][-1]
        if final > 1.0e-2:
            continue
        num = np.asarray(results[lab]["snap_num_mid"])
        ex = np.asarray(results[lab]["snap_exact_mid"])
        if num.size:
            stable_err.append(num - ex)
    stacked_err = (np.concatenate([e.ravel() for e in stable_err])
                   if stable_err else np.array([0.0]))
    vmax = float(np.nanpercentile(np.abs(stacked_err), 99.5)) * 1.3
    if vmax < 1.0e-12:
        vmax = 1.0e-10
    norm = Normalize(vmin=-vmax, vmax=vmax)

    mask = _mid_sphere_mask()
    last_im = None
    for i, lab in enumerate(SCHEMES):
        snap_num = np.asarray(results[lab]["snap_num_mid"])
        snap_ex = np.asarray(results[lab]["snap_exact_mid"])
        snap_t = np.asarray(results[lab]["snap_times"])
        for j in range(ncol):
            ax = fig.add_subplot(gs[i, j])
            if snap_num.shape[0] > j:
                err = snap_num[j] - snap_ex[j]
            else:
                err = np.zeros((NPTS, NPTS))
            err_m = np.ma.masked_where(mask, err)
            last_im = ax.imshow(err_m.T, origin="lower", extent=_extent(),
                                cmap="RdBu_r", norm=norm, aspect="equal",
                                interpolation="bilinear")
            ax.add_patch(Circle(SPHERE_CENTER[:2], SPHERE_RADIUS,
                                facecolor="#cfcfe0", edgecolor="#222",
                                lw=1.0, zorder=4))
            ax.set_xticks([]); ax.set_yticks([])
            for pos in ("top", "bottom", "left", "right"):
                ax.spines[pos].set_visible(False)
            if i == 0 and snap_t.size > j:
                ax.set_title(rf"$t = {snap_t[j]:.2f}$",
                             fontsize=11.0, pad=6)
            if j == 0:
                ax.text(-0.06, 0.5, scheme_display_name(lab),
                        transform=ax.transAxes, rotation=90,
                        ha="right", va="center",
                        fontsize=11.4, fontweight="semibold")
    cax = fig.add_axes([0.15, 0.02, 0.72, 0.012])
    if last_im is not None:
        fig.colorbar(last_im, cax=cax, orientation="horizontal",
                     label=r"pointwise error $p_h - p_\mathrm{ex}$ on $z=1/2$")
    save_figure(fig, stem)
    plt.close(fig)


def _cavity_legend(fig) -> None:
    handles = []
    for lab in SCHEMES:
        fam = ba.scheme_family(lab)
        handles.append(Line2D([0], [0], color=ba.DEFAULT_COLORS[fam],
                              ls=ba.DEFAULT_LINESTYLES[fam], lw=2.6,
                              label=scheme_display_name(lab)))
    fig.legend(handles, [h.get_label() for h in handles],
               ncol=len(SCHEMES), loc="upper center",
               bbox_to_anchor=(0.5, 0.995), frameon=False,
               handlelength=3.2, columnspacing=1.8, handletextpad=0.7,
               fontsize=12.0)


def make_figure_diagnostics(results: dict, stem: str) -> None:
    """Render the cavity diagnostics as two three-panel figures.

    The combined six-panel figure used previously mixed accuracy channels
    (volumetric, surface, force) with structural channels (near-corner
    receiver, energy drift, final surface map). The figure is split so
    that the prose can address ``where did the error go?'' on one figure
    and ``is energy preserved?'' on the next.
    """

    # First figure: three time-domain accuracy channels, one panel each.
    fig = plt.figure(figsize=(13.6, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=0.32,
                          left=0.07, right=0.99, top=0.82, bottom=0.17)
    ax_field = fig.add_subplot(gs[0, 0])
    ax_surf = fig.add_subplot(gs[0, 1])
    ax_force = fig.add_subplot(gs[0, 2])

    for lab in SCHEMES:
        fam = ba.scheme_family(lab)
        r = results[lab]
        t = np.asarray(r["times"])
        style = dict(
            color=ba.DEFAULT_COLORS[fam],
            ls=ba.DEFAULT_LINESTYLES[fam], lw=2.1,
        )
        ax_field.semilogy(t, np.maximum(r["rel_l2_field"], 1.0e-18), **style)
        ax_surf.semilogy(t, np.maximum(r["rel_l2_surf"], 1.0e-18), **style)
        ax_force.semilogy(t, np.maximum(r["force_err_norm"], 1.0e-18), **style)

    ax_field.set_title(r"(a) Volumetric $L^2$ error in $p(t)$",
                       fontsize=12.0, fontweight="semibold", pad=6)
    ax_field.set_xlabel(r"time $t$")
    ax_field.set_ylabel(r"$\|p_h(t)-p_\mathrm{ex}(t)\|_2 / \|p_\mathrm{ex}(t)\|_2$")

    ax_surf.set_title(r"(b) Off-grid sphere pressure $L^2$ error",
                      fontsize=12.0, fontweight="semibold", pad=6)
    ax_surf.set_xlabel(r"time $t$")
    ax_surf.set_ylabel(r"$\|p_h(\mathbf{x}_S,t)-p_\mathrm{ex}(\mathbf{x}_S,t)\|_2$ (rel.)")

    ax_force.set_title(r"(c) Integrated force error $\|\vec F_h-\vec F_\mathrm{ex}\|$",
                       fontsize=12.0, fontweight="semibold", pad=6)
    ax_force.set_xlabel(r"time $t$")
    ax_force.set_ylabel(r"$\|\vec F_h(t)-\vec F_\mathrm{ex}(t)\|_2$")

    _cavity_legend(fig)
    save_figure(fig, stem)
    plt.close(fig)

    # Second figure: receiver error, energy drift, and final-time surface map.
    fig2 = plt.figure(figsize=(13.6, 4.6))
    gs2 = fig2.add_gridspec(1, 3, wspace=0.36,
                            left=0.07, right=0.99, top=0.82, bottom=0.17)
    ax_recv = fig2.add_subplot(gs2[0, 0])
    ax_E = fig2.add_subplot(gs2[0, 1])
    ax_surfmap = fig2.add_subplot(gs2[0, 2])

    for lab in SCHEMES:
        fam = ba.scheme_family(lab)
        r = results[lab]
        t = np.asarray(r["times"])
        style = dict(
            color=ba.DEFAULT_COLORS[fam],
            ls=ba.DEFAULT_LINESTYLES[fam], lw=2.1,
        )
        recv_err = np.asarray(r["receiver_num"]) - np.asarray(r["receiver_exact"])
        ax_recv.plot(t, recv_err, **style)
        E0 = r["energy_num"][0]
        dE = (np.asarray(r["energy_num"]) - E0) / max(abs(E0), 1.0e-30)
        ax_E.plot(t, dE, **style)

    ax_recv.set_title(r"(a) Near-corner receiver error",
                      fontsize=12.0, fontweight="semibold", pad=6)
    ax_recv.set_xlabel(r"time $t$")
    ax_recv.set_ylabel(r"$p_h - p_\mathrm{ex}$ at the corner node")

    ax_E.set_title(r"(b) Total energy drift $\Delta E(t)/E(0)$",
                   fontsize=12.0, fontweight="semibold", pad=6)
    ax_E.set_xlabel(r"time $t$")
    ax_E.set_ylabel(r"$\left(E_h(t) - E_h(0)\right)/E_h(0)$")

    finals = {lab: results[lab]["rel_l2_surf"][-1] for lab in SCHEMES}
    best = min(finals, key=finals.get)
    p_surf_num = np.asarray(results[best]["snap_surf_num_final"])
    p_surf_ex = np.asarray(results[best]["snap_surf_exact_final"])
    lat_lon = _fibonacci_lat_lon(p_surf_num.size)
    if p_surf_num.size:
        err = p_surf_num - p_surf_ex
        lim = max(float(np.max(np.abs(err))), 1.0e-18)
        sc1 = ax_surfmap.scatter(lat_lon[:, 1], lat_lon[:, 0],
                                 c=err, cmap="RdBu_r", s=8,
                                 vmin=-lim, vmax=lim)
        ax_surfmap.set_xlabel(r"azimuth $\varphi$ [rad]")
        ax_surfmap.set_ylabel(r"polar $\theta$ [rad]")
        ax_surfmap.set_xlim(-np.pi, np.pi)
        ax_surfmap.set_ylim(0, np.pi)
        fig2.colorbar(sc1, ax=ax_surfmap, fraction=0.04, pad=0.04,
                      label=rf"$p_h - p_\mathrm{{ex}}$  at $t={results[best]['snap_times'][-1]:.2f}$")
        ax_surfmap.set_title(rf"(c) Sphere surface error [{scheme_display_name(best)}]",
                             fontsize=12.0, fontweight="semibold", pad=6)

    _cavity_legend(fig2)
    save_figure(fig2, f"{stem}_receiver_energy")
    plt.close(fig2)


def render_animation(results: dict, stem: str = "scattering_cavity") -> None:
    """Render the lowest-surface-error scheme mid-plane sequence as a GIF + MP4."""
    finals = {lab: results[lab]["rel_l2_surf"][-1] for lab in SCHEMES}
    best = min(finals, key=finals.get)
    print(f"  animation uses scheme {best}")

    n_frames = 36
    frame_times = np.linspace(T_END * 0.05, T_END, n_frames)
    op = sc.Operator3D(best, NPTS)
    x = op.x
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    mms = _mms()
    P0 = mms.pressure(X, Y, Z, 0.0)
    U0, V0, W0 = mms.velocity(X, Y, Z, 0.0)
    state = sc.AcousticState(p=P0.copy(), u=U0.copy(), v=V0.copy(), w=W0.copy())
    sc._apply_box_bc(state)

    dt = CFL * op.h / C_SOUND
    nsteps = int(np.ceil(T_END / dt))
    dt = T_END / nsteps

    frame_idx = np.clip(np.floor(frame_times / dt).astype(int), 1, nsteps)
    want = set(frame_idx.tolist())
    mid = NPTS // 2
    frames = []
    times_out = []
    for step in range(nsteps):
        sc.rk4_step(state, dt, op, RHO0, C_SOUND, None, 0.0, "soft_pressure", None, step * dt)
        if (step + 1) in want:
            frames.append(state.p[:, :, mid].copy())
            times_out.append((step + 1) * dt)
    frames = np.stack(frames, axis=0)
    mask = _mid_sphere_mask()
    vmax = float(np.percentile(np.abs(frames), 99.0))
    if vmax < 1.0e-12:
        vmax = 1.0e-3

    from matplotlib import animation
    fig, ax = plt.subplots(figsize=(5.8, 5.7))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.05)
    frame0 = np.ma.masked_where(mask, frames[0].copy())
    im = ax.imshow(frame0.T, origin="lower", extent=_extent(),
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="bilinear")
    ax.add_patch(Circle(SPHERE_CENTER[:2], SPHERE_RADIUS,
                        facecolor="#cfcfe0", edgecolor="#222",
                        lw=1.0, zorder=4))
    ax.set_xticks([]); ax.set_yticks([])
    title = ax.set_title(rf"{scheme_display_name(best)}   $t = {times_out[0]:.2f}$",
                         fontsize=12.4, pad=6)

    def update(i):
        arr = np.ma.masked_where(mask, frames[i].copy())
        im.set_data(arr.T)
        title.set_text(rf"{scheme_display_name(best)}   $t = {times_out[i]:.2f}$")
        return [im, title]

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                    interval=90, blit=False)
    gif_path = FIGDIR / f"{stem}.gif"
    mp4_path = FIGDIR / f"{stem}.mp4"
    try:
        anim.save(str(gif_path), writer=animation.PillowWriter(fps=11))
        print(f"  wrote {gif_path}")
    except Exception as exc:
        print(f"  (gif write failed: {exc})")
    try:
        anim.save(str(mp4_path), writer=animation.FFMpegWriter(fps=11, bitrate=4000))
        print(f"  wrote {mp4_path}")
    except Exception as exc:
        print(f"  (mp4 write failed: {exc})")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=None,
                        help="Override the process-pool size.")
    parser.add_argument("--with-animation", action="store_true",
                        help="Render the optional GIF/MP4 showcase animation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_generated_dirs()
    print("== Section 6.10 acoustic cavity benchmark ==")
    results = run_cavity(args.workers)

    make_figure_setup("fig_s6_scattering_cavity_setup")
    make_figure_snapshots(results, "fig_s6_scattering_cavity_snapshots")
    make_figure_diagnostics(results, "fig_s6_scattering_cavity_diagnostics")
    if args.with_animation:
        print("== rendering animation ==")
        render_animation(results)
    else:
        print("== skipping animation (not used by the manuscript) ==")

    summary = {
        "configuration": {
            "npts": NPTS, "t_end": T_END, "cfl": CFL,
            "modes": MMS_MODES,
            "sphere_center": list(SPHERE_CENTER),
            "sphere_radius": SPHERE_RADIUS,
            "n_surface_nodes": N_SURFACE_NODES,
            "receiver_index": list(RECEIVER),
            "snapshot_times": list(SNAPSHOT_TIMES),
            "record_cadence": RECORD_CADENCE,
            "c_sound": C_SOUND, "rho0": RHO0,
        },
        "per_scheme_summary": {
            lab: {
                "elapsed_s": results[lab]["elapsed_s"],
                "nsteps": results[lab]["nsteps"],
                "dt": results[lab]["dt"],
                "h": results[lab]["h"],
                "rel_l2_field_final": results[lab]["rel_l2_field"][-1],
                "rel_l2_surf_final": results[lab]["rel_l2_surf"][-1],
                "force_err_peak": float(max(results[lab]["force_err_norm"])),
                "receiver_err_final": float(results[lab]["receiver_num"][-1]
                                             - results[lab]["receiver_exact"][-1]),
                "energy_drift_final": float(
                    (results[lab]["energy_num"][-1] - results[lab]["energy_num"][0])
                    / max(abs(results[lab]["energy_num"][0]), 1.0e-30)),
                "operator_route": results[lab].get("bmf_route", "same-grid first-derivative route"),
            }
            for lab in SCHEMES
        },
    }
    out = RESULTSDIR / "scattering_cavity_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
