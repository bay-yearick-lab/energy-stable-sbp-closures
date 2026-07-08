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

OUT = RESULTSDIR / 'reflected_wave_appendix_summary.json'
ensure_generated_dirs()

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10.0,
    'axes.labelsize': 11.0,
    'axes.titlesize': 11.4,
    'legend.fontsize': 10.8,
    'xtick.labelsize': 9.2,
    'ytick.labelsize': 9.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 180,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.18,
    'grid.linewidth': 0.55,
})

COLORS = {
    'exact': '#111111',
    **ba.DEFAULT_COLORS,
    'C6F': '#e08214',
    'SBP6-BO': '#66c2a5',
}
KINDS = [(ba.FAMILY_KINDS[family], family) for family in ba.family_order()]
ORDERS = [4, 6, 8]

NPTS = 129
X0 = 0.24
CARRIER_PPW = 6.5
SIGMA = 0.05
T_HIST_END = 0.36
T_POST = 0.35
INTERIOR_STRIP = 10
CFL = 0.012
PROFILE_YLIM = (-1.35, 1.35)
PRE_BOUNCE_WINDOW = 0.08

h = 1.0 / (NPTS - 1)
wavelength = CARRIER_PPW * h
k_wave = 2.0 * np.pi / wavelength
t_hit = X0
pressure_exact, velocity_exact = ba.rigid_wall_acoustics_packet(x0=X0, sigma=SIGMA, k=k_wave)
x_sample = ba.build_operator('E', ORDERS[0], NPTS).x
X_LEFT, X_RIGHT = ba.trimmed_interior_bounds(x_sample, INTERIOR_STRIP)

fig, axes = plt.subplots(3, 2, figsize=(11.75, 10.1), sharex='col', sharey='col')
fig.subplots_adjust(left=0.08, right=0.985, top=0.85, bottom=0.08, hspace=0.08, wspace=0.12)
summary: dict[str, object] = {
    'setup': {
        'npts': NPTS,
        'carrier_ppw': CARRIER_PPW,
        'sigma': SIGMA,
        't_hit': t_hit,
        't_post': T_POST,
        'interior_strip': INTERIOR_STRIP,
        'interior_band': [X_LEFT, X_RIGHT],
        'pre_bounce_window': PRE_BOUNCE_WINDOW,
        'bmf_route': 'staggered primal-pressure and derivative-grid velocity complex; velocity is reconstructed to the primal grid for the common error metric',
    },
    'orders': {},
}


def _run_label(q: int, kind: str, name: str):
    label = f'{name}{q}'
    op = ba.build_operator(kind, q, NPTS)
    hist = ba.simulate_reflected_acoustics(
        op,
        pressure_exact,
        velocity_exact,
        t_end=T_HIST_END,
        cfl=CFL,
        interior_strip=INTERIOR_STRIP,
        record_count=170,
    )
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
    row_summary = {
        'post_bounce_error': float(
            ba.coupled_relative_l2(
                post.pressure[sl],
                post.pressure_exact[sl],
                post.velocity[sl],
                post.velocity_exact[sl],
            )
        ),
        'time_history_peak': ba.reflected_history_peak(hist.times, hist.error_history, t_hit, PRE_BOUNCE_WINDOW),
        'operator_route': post.operator_route,
    }
    return q, label, hist, post, row_summary


def _simulate_reflected_acoustics_filtered(op, t_end: float, record_count: int):
    """Same RK4 reflected-acoustics evolution as `ba.simulate_reflected_acoustics`,
    with the tenth-order implicit low-pass (F10) filter applied once per completed RK4 step
    (after the stage combination), matching `CompactFilteredFD.apply_filter_to_state`
    and the convention verified in code/verify_strong_competitors.py. Used only
    for the C6F competitor, which the shared (append-only) module's
    `simulate_reflected_acoustics` does not know how to filter.
    """
    x = op.x
    npts = x.size
    h = x[1] - x[0]
    dt = float(CFL) * float(h)
    nsteps = int(np.ceil(float(t_end) / dt))
    dt = float(t_end) / nsteps
    dmat = op.matrix
    m_u = npts - 2
    sl = ba.trimmed_interior_slice(npts, INTERIOR_STRIP)
    y = np.concatenate([pressure_exact(x, 0.0), velocity_exact(x, 0.0)[1:-1]])
    keep_every = max(1, nsteps // max(2, int(record_count) - 1))
    rec_t, rec_e = [], []

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

    def record(t, state):
        p, u = unpack(state, t)
        pex, uex = pressure_exact(x, t), velocity_exact(x, t)
        rec_t.append(float(t))
        rec_e.append(float(ba.coupled_relative_l2(p[sl], pex[sl], u[sl], uex[sl])))

    record(0.0, y)
    t = 0.0
    for step in range(nsteps):
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
        if step == nsteps - 1 or (step + 1) % keep_every == 0:
            record(t, y)

    p, u = unpack(y, float(t_end))
    pex, uex = pressure_exact(x, float(t_end)), velocity_exact(x, float(t_end))
    return ba.ReflectedAcousticsResult(
        times=np.asarray(rec_t, dtype=float),
        error_history=np.asarray(rec_e, dtype=float),
        pressure=p,
        velocity=u,
        pressure_exact=np.asarray(pex, dtype=float),
        velocity_exact=np.asarray(uex, dtype=float),
        work_per_point=float(op.work_model.total_flops / npts),
        total_work=int(op.work_model.total_flops),
        operator_route="same-grid first-derivative route (filtered)",
        grid=x.copy(),
    )


def _run_extra_label(name: str):
    """C6F / SBP6-BO at q=6 only (both are order-6-only in this codebase)."""
    op = ba.build_operator_extended(name, 6, NPTS)
    if name == 'C6F':
        hist = _simulate_reflected_acoustics_filtered(op, t_end=T_HIST_END, record_count=170)
        post = _simulate_reflected_acoustics_filtered(op, t_end=T_POST, record_count=16)
    else:
        hist = ba.simulate_reflected_acoustics(
            op, pressure_exact, velocity_exact, t_end=T_HIST_END, cfl=CFL,
            interior_strip=INTERIOR_STRIP, record_count=170,
        )
        post = ba.simulate_reflected_acoustics(
            op, pressure_exact, velocity_exact, t_end=T_POST, cfl=CFL,
            interior_strip=INTERIOR_STRIP, record_count=16,
        )
    sl = ba.trimmed_interior_slice(post.pressure.size, INTERIOR_STRIP)
    row_summary = {
        'post_bounce_error': float(
            ba.coupled_relative_l2(
                post.pressure[sl], post.pressure_exact[sl],
                post.velocity[sl], post.velocity_exact[sl],
            )
        ),
        'time_history_peak': ba.reflected_history_peak(hist.times, hist.error_history, t_hit, PRE_BOUNCE_WINDOW),
        'operator_route': post.operator_route,
    }
    return 6, f'{name}', hist, post, row_summary


EXTRA_FAMILIES_Q6 = ['C6F', 'SBP6-BO']


def _run_all(max_workers: int | None = None):
    jobs = [(q, kind, name) for q in ORDERS for kind, name in KINDS]
    if max_workers is None:
        max_workers = recommended_process_workers(len(jobs) + len(EXTRA_FAMILIES_Q6), cap=6)
    print(f"== reflected-wave appendix using {max_workers} worker process(es) ==")
    hist_runs = {q: {} for q in ORDERS}
    post_runs = {q: {} for q in ORDERS}
    row_summaries = {q: {} for q in ORDERS}
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("fork")) as ex:
        futs = {ex.submit(_run_label, q, kind, name): ('std', q, name) for q, kind, name in jobs}
        futs.update({ex.submit(_run_extra_label, name): ('extra', 6, name) for name in EXTRA_FAMILIES_Q6})
        for fut in as_completed(futs):
            q, label, hist, post, row_summary = fut.result()
            hist_runs[q][label] = hist
            post_runs[q][label] = post
            row_summaries[q][label] = row_summary
            print(f"  {label:>8s}: post={row_summary['post_bounce_error']:.3e}")
    return hist_runs, post_runs, row_summaries


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--workers", type=int, default=None, help="Override the process-pool size.")
ARGS = parser.parse_args()
hist_runs, post_runs, row_summaries = _run_all(ARGS.workers)

for row, q in enumerate(ORDERS):
    order_key = f'q{q}'
    summary['orders'][order_key] = {}
    for kind, name in KINDS:
        label = f'{name}{q}'
        summary['orders'][order_key][label] = row_summaries[q][label]
    row_extra_labels: list[str] = []
    if q == 6:
        for name in EXTRA_FAMILIES_Q6:
            summary['orders'][order_key][name] = row_summaries[6][name]
            row_extra_labels.append(name)

    post_data = post_runs[q]
    hist_data = hist_runs[q]
    x = ba.build_operator('E', q, NPTS).x
    sl = ba.trimmed_interior_slice(x.size, INTERIOR_STRIP)
    x_left, x_right = ba.trimmed_interior_bounds(x, INTERIOR_STRIP)
    pex = pressure_exact(x, T_POST)
    ax_profile = axes[row, 0]
    ax_history = axes[row, 1]

    ax_profile.plot(x[sl], pex[sl], color=COLORS['exact'], lw=1.9, ls='--', label='exact')
    for kind, name in KINDS:
        label = f'{name}{q}'
        x_lab = post_data[label].grid if post_data[label].grid is not None else x
        sl_lab = ba.trimmed_interior_slice(x_lab.size, INTERIOR_STRIP)
        y_profile = np.ma.masked_outside(post_data[label].pressure[sl_lab], *PROFILE_YLIM)
        ax_profile.plot(
            x_lab[sl_lab],
            y_profile,
            color=COLORS[name],
            lw=1.8,
            label=label,
        )
    for label in row_extra_labels:
        x_lab = post_data[label].grid if post_data[label].grid is not None else x
        sl_lab = ba.trimmed_interior_slice(x_lab.size, INTERIOR_STRIP)
        y_profile = np.ma.masked_outside(post_data[label].pressure[sl_lab], *PROFILE_YLIM)
        ax_profile.plot(x_lab[sl_lab], y_profile, color=COLORS[label], lw=1.6, ls='--', label=label)
    ax_profile.axvline(x_left, color='0.35', lw=1.0, ls=':')
    ax_profile.set_xlim(x_left, x_right)
    ax_profile.set_ylim(*PROFILE_YLIM)
    ax_profile.set_ylabel(rf'$q={q}$' + '\n' + r'$p(x,t_{\mathrm{post}})$')
    if row == 0:
        ax_profile.set_title('(a) Post-bounce packet on the trimmed interior band', pad=10)
    if row == len(ORDERS) - 1:
        ax_profile.set_xlabel(r'$x$')

    for kind, name in KINDS:
        label = f'{name}{q}'
        tau = hist_data[label].times - t_hit
        yhist = np.maximum(hist_data[label].error_history, 1.0e-12)
        ax_history.plot(
            tau,
            yhist,
            color=COLORS[name],
            lw=1.8,
            label=label,
        )
    for label in row_extra_labels:
        tau = hist_data[label].times - t_hit
        yhist = np.maximum(hist_data[label].error_history, 1.0e-12)
        ax_history.plot(tau, yhist, color=COLORS[label], lw=1.6, ls='--', label=label)
    ax_history.axvline(0.0, color='0.35', ls='--', lw=1.0)
    ax_history.set_xlim(-PRE_BOUNCE_WINDOW, T_HIST_END - t_hit)
    row_positive = np.concatenate([
        np.asarray(hist_data[f'{name}{q}'].error_history, dtype=float)
        for _kind, name in KINDS
    ] + [
        np.asarray(hist_data[label].error_history, dtype=float)
        for label in row_extra_labels
    ])
    row_positive = row_positive[np.isfinite(row_positive) & (row_positive > 0.0)]
    row_ymin = max(1.0e-7, 10.0 ** np.floor(np.log10(float(np.min(row_positive))) - 0.1))
    row_ymax = 10.0 ** np.ceil(np.log10(float(np.max(row_positive))) + 0.1)
    ax_history.set_yscale('log')
    ax_history.set_ylim(row_ymin, row_ymax)
    if row == 0:
        ax_history.set_title('(b) Interior packet error over the bounce window', pad=10)
    if row == len(ORDERS) - 1:
        ax_history.set_xlabel(r'$t - t_{\mathrm{hit}}$')
    ax_history.set_ylabel(r'$\mathcal{E}_{\mathrm{int}}(t)$')
handles = [
    Line2D([0], [0], color=COLORS['exact'], lw=1.9, ls='--', label='exact'),
    Line2D([0], [0], color=COLORS['E'], lw=1.8, label=r'$E_q$'),
    Line2D([0], [0], color=COLORS['C'], lw=1.8, label=r'$C_q$'),
    Line2D([0], [0], color=COLORS['SBP'], lw=1.8, label=r'$\mathrm{SBP}_q$'),
    Line2D([0], [0], color=COLORS['DRP'], lw=1.8, label=r'$\mathrm{DRP}_q$'),
    Line2D([0], [0], color=COLORS['BMF'], lw=1.8, label=r'$\mathrm{BMF}_q$'),
    Line2D([0], [0], color=COLORS['BMG'], lw=1.8, label=r'$\mathrm{BMG}_q$'),
    Line2D([0], [0], color=COLORS['C6F'], lw=1.6, ls='--', label=r'$\mathrm{C6F}$ ($q{=}6$ only)'),
    Line2D([0], [0], color=COLORS['SBP6-BO'], lw=1.6, ls='--', label=r'$\mathrm{SBP6\text{-}BO}$ ($q{=}6$ only)'),
]
fig.legend(
    handles,
    [handle.get_label() for handle in handles],
    loc='upper center',
    bbox_to_anchor=(0.5, 0.986),
    ncol=3,
    frameon=False,
    handlelength=2.35,
    columnspacing=1.45,
    handletextpad=0.6,
    fontsize=10.6,
)

save_figure(fig, 'fig_s6_reflected_wave', pad_inches=0.03)
plt.close(fig)

OUT.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
