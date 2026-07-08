#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import boundary_spline_analysis as ba
from project_paths import RESULTSDIR, ensure_generated_dirs, recommended_process_workers, save_figure

OUT = RESULTSDIR / 'reflected_wave_ppw_sweep_summary.json'
ensure_generated_dirs()

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9.8,
    'axes.labelsize': 10.8,
    'axes.titlesize': 11.5,
    'legend.fontsize': 10.6,
    'xtick.labelsize': 9.0,
    'ytick.labelsize': 9.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 180,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.12,
    'grid.linewidth': 0.45,
})

COLORS = ba.DEFAULT_COLORS
KINDS = [(ba.FAMILY_KINDS[family], family) for family in ba.family_order()]
LINESTYLES = ba.DEFAULT_LINESTYLES
MARKERS = ba.DEFAULT_MARKERS
ORDERS = [4, 6, 8]
PPWS = np.array(list(np.arange(4.5, 8.25, 0.25)) + list(np.arange(8.5, 12.5, 0.5)), dtype=float)

NPTS = 129
X0 = 0.24
SIGMA = 0.05
T_POST = 0.35
INTERIOR_STRIP = 10
CFL = 0.012
REFERENCE_PPW = 6.5


def first_ppw_below(ppw: np.ndarray, errors: np.ndarray, tol: float) -> float:
    idx = np.where(np.asarray(errors, dtype=float) <= float(tol))[0]
    return float(ppw[idx[0]]) if idx.size else float('nan')


def post_bounce_error(op: ba.Operator1D, carrier_ppw: float) -> float:
    h = 1.0 / (NPTS - 1)
    wavelength = float(carrier_ppw) * h
    k_wave = 2.0 * np.pi / wavelength
    pressure_exact, velocity_exact = ba.rigid_wall_acoustics_packet(x0=X0, sigma=SIGMA, k=k_wave)
    post = ba.simulate_reflected_acoustics(
        op,
        pressure_exact,
        velocity_exact,
        t_end=T_POST,
        cfl=CFL,
        interior_strip=INTERIOR_STRIP,
        record_count=16,
    )
    sl = ba.trimmed_interior_slice(post.pressure.size, INTERIOR_STRIP)
    return float(
        ba.coupled_relative_l2(
            post.pressure[sl],
            post.pressure_exact[sl],
            post.velocity[sl],
            post.velocity_exact[sl],
        )
    )


def post_bounce_error_filtered(op: ba.Operator1D, carrier_ppw: float) -> float:
    """Same post-bounce metric as `post_bounce_error`, with the implicit-filter
    F10 filter applied once per completed RK4 step (C6F only)."""
    h = 1.0 / (NPTS - 1)
    wavelength = float(carrier_ppw) * h
    k_wave = 2.0 * np.pi / wavelength
    pressure_exact, velocity_exact = ba.rigid_wall_acoustics_packet(x0=X0, sigma=SIGMA, k=k_wave)
    x = op.x
    npts = x.size
    dt = float(CFL) * h
    nsteps = int(np.ceil(float(T_POST) / dt))
    dt = float(T_POST) / nsteps
    dmat = op.matrix
    m_u = npts - 2
    y = np.concatenate([pressure_exact(x, 0.0), velocity_exact(x, 0.0)[1:-1]])

    def unpack(state, t):
        p = np.asarray(state[:npts], dtype=float)
        u = np.empty(npts, dtype=float)
        u[0] = float(np.asarray(velocity_exact(x[0], t)))
        u[-1] = float(np.asarray(velocity_exact(x[-1], t)))
        u[1:-1] = state[npts:npts + m_u]
        return p, u

    def rhs(t, state):
        p, u = unpack(state, t)
        dp = -dmat @ u
        du = -dmat @ p
        return np.concatenate([dp, du[1:-1]])

    t = 0.0
    for _ in range(nsteps):
        k1 = rhs(t, y)
        k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = rhs(t + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        p_full, u_full = unpack(y, t)
        p_filt = op.apply_filter_to_state(p_full)
        u_filt = op.apply_filter_to_state(u_full)
        y = np.concatenate([p_filt, u_filt[1:-1]])

    p, u = unpack(y, float(T_POST))
    pex, uex = pressure_exact(x, float(T_POST)), velocity_exact(x, float(T_POST))
    sl = ba.trimmed_interior_slice(npts, INTERIOR_STRIP)
    return float(ba.coupled_relative_l2(p[sl], pex[sl], u[sl], uex[sl]))


summary: dict[str, object] = {
    'setup': {
        'npts': NPTS,
        'x0': X0,
        'sigma': SIGMA,
        't_post': T_POST,
        'interior_strip': INTERIOR_STRIP,
        'carrier_ppw_grid': PPWS.tolist(),
        'reference_ppw': REFERENCE_PPW,
        'bmf_route': 'staggered primal-pressure and derivative-grid velocity complex; velocity is reconstructed to the primal grid for the common error metric',
    },
    'orders': {},
}


def evaluate_sweep_curve(q: int, kind: str, name: str) -> tuple[int, str, list[float], dict[str, float | list[float]]]:
    op = ba.build_operator(kind, q, NPTS)
    errors = np.array([post_bounce_error(op, ppw) for ppw in PPWS], dtype=float)
    curve_summary = {
        'post_bounce_error': errors.tolist(),
        'ppw_at_error_0p3': first_ppw_below(PPWS, errors, 0.3),
        'ppw_at_error_0p1': first_ppw_below(PPWS, errors, 0.1),
        'reference_error': float(errors[np.argmin(np.abs(PPWS - REFERENCE_PPW))]),
    }
    return q, name, errors.tolist(), curve_summary


EXTRA_FAMILIES_Q6 = ['C6F', 'SBP6-BO']


def evaluate_extra_sweep_curve(name: str) -> tuple[int, str, list[float], dict[str, float | list[float]]]:
    """C6F / SBP6-BO sweep at q=6 (both order-6-only in this codebase)."""
    op = ba.build_operator_extended(name, 6, NPTS)
    if name == 'C6F':
        errors = np.array([post_bounce_error_filtered(op, ppw) for ppw in PPWS], dtype=float)
    else:
        errors = np.array([post_bounce_error(op, ppw) for ppw in PPWS], dtype=float)
    curve_summary = {
        'post_bounce_error': errors.tolist(),
        'ppw_at_error_0p3': first_ppw_below(PPWS, errors, 0.3),
        'ppw_at_error_0p1': first_ppw_below(PPWS, errors, 0.1),
        'reference_error': float(errors[np.argmin(np.abs(PPWS - REFERENCE_PPW))]),
    }
    return 6, name, errors.tolist(), curve_summary


def run_sweeps(max_workers: int | None = None) -> dict[int, dict[str, dict[str, object]]]:
    jobs = [(q, kind, name) for q in ORDERS for kind, name in KINDS]
    if max_workers is None:
        max_workers = recommended_process_workers(len(jobs) + len(EXTRA_FAMILIES_Q6), cap=6)
    print(f"== reflected PPW sweep using {max_workers} worker process(es) ==")
    out: dict[int, dict[str, dict[str, object]]] = {q: {} for q in ORDERS}
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("fork")) as ex:
        futs = {ex.submit(evaluate_sweep_curve, q, kind, name): (q, name) for q, kind, name in jobs}
        futs.update({ex.submit(evaluate_extra_sweep_curve, name): (6, name) for name in EXTRA_FAMILIES_Q6})
        for fut in as_completed(futs):
            q, name = futs[fut]
            q_out, name_out, errors, curve_summary = fut.result()
            out[q_out][name_out] = {
                'errors': np.asarray(errors, dtype=float),
                'summary': curve_summary,
            }
            print(f"  {name}{q}: reference={curve_summary['reference_error']:.3e}")
    return out


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--workers", type=int, default=None, help="Override the process-pool size.")
ARGS = parser.parse_args()
sweep_results = run_sweeps(ARGS.workers)
all_positive = np.concatenate([
    data['errors'][np.isfinite(data['errors']) & (data['errors'] > 0.0)]
    for qdata in sweep_results.values()
    for data in qdata.values()
])
ymin = 10.0 ** np.floor(np.log10(float(np.min(all_positive))) - 0.15)
ymax = 10.0 ** np.ceil(np.log10(float(np.max(all_positive))) + 0.15)

fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.7), sharey=True)
fig.subplots_adjust(left=0.07, right=0.995, bottom=0.17, top=0.77, wspace=0.18)
for idx, (ax, q) in enumerate(zip(axes, ORDERS, strict=True)):
    qkey = f'q{q}'
    summary['orders'][qkey] = {}
    reference_errors: dict[str, float] = {}
    for kind, name in KINDS:
        data = sweep_results[q][name]
        errors = data['errors']
        ax.plot(
            PPWS,
            errors,
            color=COLORS[name],
            lw=2.3,
            ls=LINESTYLES[name],
            marker=MARKERS[name],
            ms=4.8,
            markevery=3,
            mec='white',
            mew=0.8,
        )
        summary['orders'][qkey][f'{name}{q}'] = data['summary']
        reference_errors[name] = summary['orders'][qkey][f'{name}{q}']['reference_error']
    if q == 6:
        extra_colors = {'C6F': '#e08214', 'SBP6-BO': '#66c2a5'}
        for name in EXTRA_FAMILIES_Q6:
            data = sweep_results[6][name]
            errors = data['errors']
            ax.plot(PPWS, errors, color=extra_colors[name], lw=2.0, ls='--',
                    marker='v', ms=4.4, markevery=3, mec='white', mew=0.7)
            summary['orders'][qkey][name] = data['summary']
    ax.axvline(REFERENCE_PPW, color='0.35', lw=1.1, ls='--')
    ax.text(REFERENCE_PPW + 0.08, 0.91, r'$6.5h$', transform=ax.get_xaxis_transform(), fontsize=8.5, color='0.35')
    ax.set_title(rf'({chr(97 + idx)}) $q={q}$', fontweight='semibold')
    ax.set_xlabel('carrier PPW (lower is harder)')
    ax.set_yscale('log')
    ax.set_xlim(PPWS[0], PPWS[-1])
    ax.set_ylim(ymin, ymax)

axes[0].set_ylabel('post-bounce interior packet error')
handles = [
    Line2D([0], [0], color=COLORS['E'], lw=2.3, label=r'$E_q$'),
    Line2D([0], [0], color=COLORS['C'], lw=2.3, ls='--', marker='s', ms=4.8, label=r'$C_q$'),
    Line2D([0], [0], color=COLORS['SBP'], lw=2.3, ls=LINESTYLES['SBP'], marker=MARKERS['SBP'], ms=4.8, label=r'$\mathrm{SBP}_q$'),
    Line2D([0], [0], color=COLORS['DRP'], lw=2.3, ls=LINESTYLES['DRP'], marker=MARKERS['DRP'], ms=4.8, label=r'$\mathrm{DRP}_q$'),
    Line2D([0], [0], color=COLORS['BMF'], lw=2.3, ls='-.', marker='^', ms=4.8, label=r'$\mathrm{BMF}_q$'),
    Line2D([0], [0], color=COLORS['BMG'], lw=2.3, ls=':', marker='D', ms=4.8, label=r'$\mathrm{BMG}_q$'),
    Line2D([0], [0], color='#e08214', lw=2.0, ls='--', marker='v', ms=4.4, label=r'C6F ($q{=}6$)'),
    Line2D([0], [0], color='#66c2a5', lw=2.0, ls='--', marker='v', ms=4.4, label=r'SBP6-BO ($q{=}6$)'),
]
handles[0].set_marker('o')
handles[0].set_markersize(4.8)
handles[0].set_markeredgecolor('white')
handles[0].set_markeredgewidth(0.8)
fig.legend(
    handles,
    [h.get_label() for h in handles],
    loc='upper center',
    bbox_to_anchor=(0.5, 0.986),
    ncol=4,
    frameon=False,
    handlelength=2.35,
    columnspacing=1.45,
    fontsize=10.4,
)

save_figure(fig, 'fig_s6_reflected_wave_ppw_sweep', pad_inches=0.03)
plt.close(fig)

OUT.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
