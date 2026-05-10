#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

import boundary_spline_analysis as ba
from project_paths import RESULTSDIR, ensure_generated_dirs, save_figure


ensure_generated_dirs()

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10.4,
    'axes.titlesize': 11.0,
    'axes.labelsize': 10.2,
    'xtick.labelsize': 9.0,
    'ytick.labelsize': 9.0,
    'figure.dpi': 180,
    'savefig.dpi': 300,
})


def basis_curves(knots: np.ndarray, degree: int, sample: np.ndarray) -> np.ndarray:
    nbasis = knots.size - degree - 1
    out = np.zeros((nbasis, sample.size), dtype=float)
    coeffs = np.zeros(nbasis, dtype=float)
    for j in range(nbasis):
        coeffs[:] = 0.0
        coeffs[j] = 1.0
        spline = BSpline(knots, coeffs.copy(), degree, extrapolate=False)
        out[j, :] = np.nan_to_num(spline(sample))
    return out


DEG_FIG = 4
N_FIG = 9
DEG_EX = 3
N_EX = 7


fig, ax = plt.subplots(figsize=(7.6, 3.4), layout='constrained')

knots = ba.open_uniform_knots(N_FIG, DEG_FIG)
greville = ba.greville_from_knots(knots, DEG_FIG, N_FIG)
sample = np.linspace(0.0, 1.0, 801)
curves = basis_curves(knots, DEG_FIG, sample)

ACCENT = '#3f5b8a'

cmap = plt.get_cmap('viridis')
for j in range(N_FIG):
    color = cmap(0.08 + 0.84 * j / max(N_FIG - 1, 1))
    ax.plot(sample, curves[j], color=color, linewidth=1.35, alpha=0.95)

interior_knots = np.unique(knots[(knots > 0.0) & (knots < 1.0)])
for k in interior_knots:
    ax.axvline(k, color='0.72', linewidth=0.55, linestyle=(0, (2.4, 2.2)), zorder=0)

ax.axhline(1.0, color='0.85', linewidth=0.5, linestyle=(0, (1.4, 2.0)), zorder=0)

ax.plot(0.0, 1.0, marker='o', color=ACCENT, markersize=5.5,
        markerfacecolor='white', markeredgewidth=1.2, zorder=6)
ax.plot(1.0, 1.0, marker='o', color=ACCENT, markersize=5.5,
        markerfacecolor='white', markeredgewidth=1.2, zorder=6)

ax.annotate(
    r'$B_0(0)=1$',
    xy=(0.0, 1.0), xytext=(0.075, 0.92),
    fontsize=9.1, color=ACCENT,
    ha='left', va='center',
    arrowprops=dict(arrowstyle='-', color=ACCENT, lw=0.7, shrinkA=2, shrinkB=2),
)
ax.annotate(
    r'$B_{N-1}(1)=1$',
    xy=(1.0, 1.0), xytext=(0.84, 0.92),
    fontsize=9.1, color=ACCENT,
    ha='right', va='center',
    arrowprops=dict(arrowstyle='-', color=ACCENT, lw=0.7, shrinkA=2, shrinkB=2),
)

GREV_STRIP_TOP = -0.06
GREV_STRIP_BOT = -0.155
ax.axhspan(GREV_STRIP_BOT, GREV_STRIP_TOP, xmin=0.0, xmax=1.0,
           facecolor='#f5f1e6', edgecolor='none', zorder=1)
ax.plot([0.0, 1.0], [GREV_STRIP_TOP, GREV_STRIP_TOP], color='0.55', linewidth=0.55, zorder=2)
ax.plot([0.0, 1.0], [GREV_STRIP_BOT, GREV_STRIP_BOT], color='0.55', linewidth=0.55, zorder=2)
ax.text(-0.005, 0.5 * (GREV_STRIP_TOP + GREV_STRIP_BOT),
        'Greville\nnodes', fontsize=8.4, color='0.25',
        ha='right', va='center', linespacing=1.0)

ax.scatter(
    greville, np.full_like(greville, 0.5 * (GREV_STRIP_TOP + GREV_STRIP_BOT)),
    s=24, color='0.08', zorder=5, marker='o',
    edgecolor='white', linewidth=0.7,
)

for j, x in enumerate(greville):
    ax.annotate(
        rf'$x_{{{j}}}$',
        xy=(x, GREV_STRIP_BOT), xytext=(x, GREV_STRIP_BOT - 0.035),
        ha='center', va='top',
        xycoords='data', textcoords='data',
        fontsize=8.4, color='0.15',
    )

ax.set_xlim(-0.05, 1.015)
ax.set_ylim(-0.27, 1.12)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$B_j(x)$')
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.0, 0.5, 1.0])
ax.tick_params(length=2.8, width=0.6, pad=2.2)
for spine in ('top', 'right'):
    ax.spines[spine].set_visible(False)
for spine in ('bottom', 'left'):
    ax.spines[spine].set_linewidth(0.65)
    ax.spines[spine].set_color('0.25')


save_figure(fig, 'fig_s3_clamped_basis', pad_inches=0.03)
plt.close(fig)
print('wrote appendix clamped basis figure')


knots_ex = ba.open_uniform_knots(N_EX, DEG_EX)
nodes_ex, _, bmat_ex, gmat_ex = ba.clamped_bspline_matrices(N_EX, DEG_EX)
mass_ex, adv_ex = ba.bspline_mass_and_advection(N_EX, DEG_EX)
_nodes_check, dmat_bmf_ex = ba.bmf_nodal_operator(N_EX, DEG_EX)
_nodes_check2, dmat_bmg_ex, h_bmg_ex = ba.bmg_sbp_first_derivative_matrix(N_EX, DEG_EX)


def fmt_vec(vec: np.ndarray, digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in vec]


def fmt_mat(mat: np.ndarray, digits: int = 4) -> list[list[float]]:
    return [[round(float(v), digits) for v in row] for row in mat]


worked = {
    'degree': DEG_EX,
    'npts': N_EX,
    'knots': fmt_vec(knots_ex, 6),
    'greville': fmt_vec(nodes_ex, 6),
    'B': fmt_mat(bmat_ex, 4),
    'Bx': fmt_mat(gmat_ex, 4),
    'M': fmt_mat(mass_ex, 5),
    'K': fmt_mat(adv_ex, 5),
    'D_BMF': fmt_mat(dmat_bmf_ex, 3),
    'D_BMG': fmt_mat(dmat_bmg_ex, 3),
    'H_BMG_diag': fmt_vec(np.diag(h_bmg_ex), 5),
}


out_path = RESULTSDIR / 'appendix_worked_example_summary.json'
out_path.write_text(json.dumps(worked, indent=2))
print(f'wrote worked example summary -> {out_path}')
