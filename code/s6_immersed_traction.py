#!/usr/bin/env python3
from __future__ import annotations

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lu_solve

import boundary_spline_analysis as ba
from plotting_helpers import add_theoretical_reference_line
from project_paths import RESULTSDIR, ensure_generated_dirs, save_figure

OUT = RESULTSDIR / 'immersed_traction_summary.json'
ensure_generated_dirs()

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10.2,
    'axes.labelsize': 11.2,
    'axes.titlesize': 12.0,
    'legend.fontsize': 9.8,
    'xtick.labelsize': 9.3,
    'ytick.labelsize': 9.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 180,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.12,
    'grid.linewidth': 0.45,
})

COLORS = ba.DEFAULT_COLORS
LABELS = ba.comparison_labels(6)
FORMAL_ORDER = 6
MARKERS = ba.DEFAULT_MARKERS
LINESTYLES = ba.DEFAULT_LINESTYLES
SNAPSHOT_N = 81

DECAY = 18.0
WAVE = 10.0 * np.pi
ALPHA = 8.0
BETA_NORMAL = 0.2

xc, yc = 0.24, 0.50
radius = 0.11
M = 240
phi = np.linspace(-np.pi, np.pi, M, endpoint=False)
xm = xc + radius * np.cos(phi)
ym = yc + radius * np.sin(phi)
nx = np.cos(phi)
ny = np.sin(phi)
ds = radius * (2.0 * np.pi / M)


def field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.exp(-DECAY * x) * np.sin(WAVE * x) * np.cos(2.0 * np.pi * y)


def grad_x(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.exp(-DECAY * x) * (WAVE * np.cos(WAVE * x) - DECAY * np.sin(WAVE * x)) * np.cos(2.0 * np.pi * y)


def grad_y(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -2.0 * np.pi * np.exp(-DECAY * x) * np.sin(WAVE * x) * np.sin(2.0 * np.pi * y)


def interp2(U: np.ndarray, Wx: np.ndarray, Wy: np.ndarray) -> np.ndarray:
    return np.einsum('ki,ij,kj->k', Wx, U, Wy)


u_exact = field(xm, ym)
normal_exact = grad_x(xm, ym) * nx + grad_y(xm, ym) * ny
traction_exact = ALPHA * u_exact + BETA_NORMAL * normal_exact
force_exact = ds * np.array([
    np.sum(traction_exact * nx),
    np.sum(traction_exact * ny),
])
moment_exact = float(ds * np.sum(xm * traction_exact * ny - ym * traction_exact * nx))

scale_traction = float(np.linalg.norm(traction_exact))
scale_force = float(np.linalg.norm(force_exact))
scale_moment = abs(moment_exact)


def evaluate_scheme(label: str, n: int) -> dict[str, object]:
    kind = ba.scheme_kind(label)
    degree = 6
    op = ba.build_operator(kind, degree, n)
    x = op.x
    X, Y = np.meshgrid(x, x, indexing='ij')
    U = field(X, Y)
    if ba.scheme_family(label) in ('E', 'C', 'SBP', 'DRP'):
        Wx = ba.local_interpolation_matrix(x, xm, degree + 1)
        Wy = ba.local_interpolation_matrix(x, ym, degree + 1)
        value = interp2(U, Wx, Wy)
        gx = interp2(op.differentiate(U), Wx, Wy)
        gy = interp2(U @ op.matrix.T, Wx, Wy)
    elif kind == 'B':
        coeff = lu_solve(op._lu, U)
        coeff = lu_solve(op._lu, coeff.T).T
        _, phix, dphix = ba.spline_eval_matrices(xm, x, degree)
        _, phiy, dphiy = ba.spline_eval_matrices(ym, x, degree)
        value = np.einsum('ki,ij,kj->k', phix, coeff, phiy)
        gx = np.einsum('ki,ij,kj->k', dphix, coeff, phiy)
        gy = np.einsum('ki,ij,kj->k', phix, coeff, dphiy)
    else:
        coeff = lu_solve(op._plu, U)
        coeff = lu_solve(op._plu, coeff.T).T
        gxc = op.differentiate_coefficients(coeff)
        gyc = op.differentiate_coefficients(coeff.transpose(1, 0)).transpose(1, 0)
        _, phix, _ = ba.spline_eval_matrices(xm, x, degree)
        _, phiy, _ = ba.spline_eval_matrices(ym, x, degree)
        value = np.einsum('ki,ij,kj->k', phix, coeff, phiy)
        gx = np.einsum('ki,ij,kj->k', phix, gxc, phiy)
        gy = np.einsum('ki,ij,kj->k', phix, gyc, phiy)
    normal = gx * nx + gy * ny
    traction = ALPHA * value + BETA_NORMAL * normal
    force = ds * np.array([
        np.sum(traction * nx),
        np.sum(traction * ny),
    ])
    moment = float(ds * np.sum(xm * traction * ny - ym * traction * nx))
    return {
        'traction_rel_l2': float(np.linalg.norm(traction - traction_exact) / scale_traction),
        'force_rel': float(np.linalg.norm(force - force_exact) / scale_force),
        'moment_rel': float(abs(moment - moment_exact) / scale_moment),
        'work_2d': float(2.0 * n * op.work_model.total_flops),
        'traction': traction.tolist(),
    }


snapshot = {label: evaluate_scheme(label, SNAPSHOT_N) for label in LABELS}
convergence = {label: {'n': [], 'work_2d': [], 'traction_rel_l2': []} for label in LABELS}
for label in LABELS:
    for n in [33, 49, 65, 81]:
        data = evaluate_scheme(label, n)
        convergence[label]['n'].append(n)
        convergence[label]['work_2d'].append(data['work_2d'])
        convergence[label]['traction_rel_l2'].append(data['traction_rel_l2'])

ratio_tau = snapshot['C6']['traction_rel_l2'] / snapshot['BMF6']['traction_rel_l2']
ratio_force = snapshot['C6']['force_rel'] / snapshot['BMF6']['force_rel']
ratio_moment = snapshot['C6']['moment_rel'] / snapshot['BMF6']['moment_rel']

fig = plt.figure(figsize=(12.45, 5.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.97, 1.02])
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])

theta = phi / np.pi
ax0.plot(theta, traction_exact, color='black', lw=2.3, ls='--', label='exact')
for label in LABELS:
    ax0.plot(theta, np.asarray(snapshot[label]['traction']), lw=2.2, color=COLORS[ba.scheme_family(label)], ls=LINESTYLES[ba.scheme_family(label)], label=label)
ax0.set_xlabel(r'marker angle $\theta / \pi$')
ax0.set_ylabel(r'Robin traction $\tau(\theta)$')
ax0.set_title(r'(a) Traction density along the marker', fontsize=11.2, pad=8, fontweight='semibold')

cats = ['traction\ndensity', 'net\nforce', 'first\nmoment']
centers = np.arange(len(cats), dtype=float)
width = 0.12
offsets = np.linspace(-2.5 * width, 2.5 * width, len(LABELS))
for off, label in zip(offsets, LABELS, strict=True):
    vals = [
        snapshot[label]['traction_rel_l2'],
        snapshot[label]['force_rel'],
        snapshot[label]['moment_rel'],
    ]
    ax1.bar(centers + off, vals, width=width, color=COLORS[ba.scheme_family(label)], alpha=0.92, label=label)
ax1.set_xticks(centers)
ax1.set_xticklabels(cats, linespacing=1.0)
ax1.set_yscale('log')
ax1.set_ylabel(rf'relative error at $N={SNAPSHOT_N}$')
ax1.set_title('(b) Density, force, and moment', fontsize=11.2, pad=8, fontweight='semibold')

for label in LABELS:
    family = ba.scheme_family(label)
    ax2.loglog(
        convergence[label]['work_2d'],
        convergence[label]['traction_rel_l2'],
        marker=MARKERS[family],
        lw=2.3,
        ls=LINESTYLES[family],
        ms=6.0,
        mfc='white',
        mec=COLORS[family],
        mew=1.0,
        color=COLORS[family],
        label=label,
    )
ax2.set_xlabel('recurring 2D transfer work [flops]')
ax2.set_ylabel(r'traction-density $L^2$ relative error')
ax2.set_title('(c) Work versus density error', fontsize=11.2, pad=8, fontweight='semibold')
# The recurring 2D work grows like n^2 on this fixed-geometry sweep, so a
# sixth-order h-asymptotic law becomes a proportional W^{-3} guide here.
add_theoretical_reference_line(
    ax2,
    slope=-0.5 * FORMAL_ORDER,
    label=rf'$\propto \mathcal{{W}}^{{-{FORMAL_ORDER // 2}}}$',
    candidate_regions=('lower right', 'mid right', 'upper right', 'upper left', 'lower left'),
)

handles, labels = ax0.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    ncol=4,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.985),
    frameon=False,
    handlelength=2.3,
    columnspacing=1.2,
    fontsize=10.9,
)
fig.subplots_adjust(left=0.055, right=0.995, bottom=0.17, top=0.76, wspace=0.22)

save_figure(fig, 'fig_s6_immersed_traction', pad_inches=0.03)
plt.close(fig)

summary = {
    'formula': {
        'traction': 'tau = alpha * u + beta_normal * d_n u',
        'alpha': ALPHA,
        'beta_normal': BETA_NORMAL,
    },
    f'snapshot_n{SNAPSHOT_N}': {label: {
        'traction_rel_l2': snapshot[label]['traction_rel_l2'],
        'force_rel': snapshot[label]['force_rel'],
        'moment_rel': snapshot[label]['moment_rel'],
        'work_2d': snapshot[label]['work_2d'],
    } for label in LABELS},
    'convergence': convergence,
    f'ratios_vs_C6_n{SNAPSHOT_N}': {
        'traction_density': float(ratio_tau),
        'net_force': float(ratio_force),
        'first_moment': float(ratio_moment),
    },
}
OUT.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
