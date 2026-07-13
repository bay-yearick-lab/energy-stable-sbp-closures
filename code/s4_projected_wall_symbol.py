#!/usr/bin/env python3
from __future__ import annotations

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import boundary_spline_analysis as ba
from project_paths import RESULTSDIR, ensure_generated_dirs, save_figure

OUT = RESULTSDIR / 'projected_wall_symbol_summary.json'
ensure_generated_dirs()

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 11.0,
    'axes.titlesize': 11.5,
    'legend.fontsize': 9.6,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 180,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.18,
    'grid.linewidth': 0.55,
})

COLORS = {'exact': '#111111', **ba.DEFAULT_COLORS}
QUARTET = ba.comparison_block(6)
NPTS = 129


def threshold(ppw: np.ndarray, metric: np.ndarray, tol: float) -> float:
    idx = np.where(metric <= tol)[0]
    return float(ppw[idx[-1]]) if idx.size else float('nan')


fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.5))
ax_bias, ax_resp = axes[0]
ax_eta, ax_tot = axes[1]
bias_levels: list[np.ndarray] = []

summary: dict[str, object] = {
    'definition': {
        'exact_wall_number': 'omega_m = m pi h_ref, h_ref = max_j(x_{j+1} - x_j)',
        'signed_bias': 'rho_m - 1',
        'response_ratio': 'rho_m = kappa_m / (m pi)',
        'strict_response_error': '|rho_m - 1|',
        'contamination': 'eta_m',
        'total_error': 'sqrt((rho_m - 1)^2 + eta_m^2)',
    },
    'q6': {},
}

for label, kind, degree in QUARTET:
    op = ba.build_operator(kind, degree, NPTS)
    response = ba.wall_mode_response(op)
    ppw = np.array([row.ppw for row in response], dtype=float)
    omega = 2.0 * np.pi / ppw
    omega_ratio = omega / np.pi
    rho = np.array([row.kappa_ratio for row in response], dtype=float)
    bias = rho - 1.0
    strict = np.abs(bias)
    contamination = np.array([row.contamination_l2 for row in response], dtype=float)
    total = np.array([row.rel_l2 for row in response], dtype=float)
    family = ba.scheme_family(label)
    bias_levels.append(bias)

    ax_bias.plot(omega_ratio, bias, lw=2.0, color=COLORS[family], ls=ba.DEFAULT_LINESTYLES[family], label=label)
    ax_resp.plot(omega_ratio, strict, lw=2.0, color=COLORS[family], ls=ba.DEFAULT_LINESTYLES[family], label=label)
    ax_eta.plot(omega_ratio, contamination, lw=2.0, color=COLORS[family], ls=ba.DEFAULT_LINESTYLES[family], label=label)
    ax_tot.plot(omega_ratio, total, lw=2.0, color=COLORS[family], ls=ba.DEFAULT_LINESTYLES[family], label=label)

    summary['q6'][label] = {
        'P1_response': threshold(ppw, strict, 0.01),
        'P5_response': threshold(ppw, strict, 0.05),
        'P1_contamination': threshold(ppw, contamination, 0.01),
        'P5_contamination': threshold(ppw, contamination, 0.05),
        'P1_total': threshold(ppw, total, 0.01),
        'P5_total': threshold(ppw, total, 0.05),
        'max_signed_bias': float(np.max(bias)),
        'min_signed_bias': float(np.min(bias)),
    }

for level, style, color in ((0.0, '-', '0.35'), (0.01, '--', '0.45'), (-0.01, '--', '0.45'), (0.05, ':', '0.60'), (-0.05, ':', '0.60')):
    ax_bias.axhline(level, color=color, lw=1.0, ls=style)

for ax in (ax_resp, ax_eta, ax_tot):
    ax.axhline(0.01, color='0.45', lw=1.0, ls='--')
    ax.axhline(0.05, color='0.60', lw=1.0, ls=':')
    ax.set_yscale('log')
    ax.set_ylim(7.0e-4, 2.0)

ax_bias.set_title(r'(a) Signed target-channel bias')
ax_bias.set_ylabel(r'$\rho_m - 1$')

ax_resp.set_title(r'(b) Strict target-channel error')
ax_resp.set_ylabel(r'$|\rho_m - 1|$')

ax_eta.set_title(r'(c) Orthogonal contamination')
ax_eta.set_ylabel(r'$\eta_m$')

ax_tot.set_title(r'(d) Total wall-mode error')
ax_tot.set_ylabel(r'$\sqrt{(\rho_m-1)^2 + \eta_m^2}$')

for ax in (ax_eta, ax_tot):
    ax.set_xlabel(r'$\omega_m / \pi$')

ax_bias.set_ylim(-1.0, 0.1)

for ax in (ax_bias, ax_resp, ax_eta, ax_tot):
    ax.set_xlim(0.0, 1.0)

handles, labels = ax_bias.get_legend_handles_labels()
fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 0.988), frameon=False)
fig.subplots_adjust(left=0.075, right=0.99, bottom=0.08, top=0.84, wspace=0.26, hspace=0.32)

save_figure(fig, 'fig_s4_projected_wall_symbol', pad_inches=0.03)
plt.close(fig)
OUT.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
