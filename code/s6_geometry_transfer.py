#!/usr/bin/env python3
from __future__ import annotations

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.linalg import lu_solve

import boundary_spline_analysis as ba
from plotting_helpers import add_theoretical_reference_line
from project_paths import RESULTSDIR, ensure_generated_dirs, save_figure

OUT = RESULTSDIR / 'geometry_transfer_summary.json'
ensure_generated_dirs()

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10.2,
    'axes.labelsize': 11.2,
    'axes.titlesize': 12.0,
    'legend.fontsize': 10.8,
    'xtick.labelsize': 9.2,
    'ytick.labelsize': 9.2,
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

beta = 18.0
omega = 10.0 * np.pi
xc, yc = 0.24, 0.50
radius = 0.11
M = 240
phi = np.linspace(-np.pi, np.pi, M, endpoint=False)
xm = xc + radius * np.cos(phi)
ym = yc + radius * np.sin(phi)
nx = np.cos(phi)
ny = np.sin(phi)


def field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.exp(-beta * x) * np.sin(omega * x) * np.cos(2.0 * np.pi * y)


def grad_x(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.exp(-beta * x) * (omega * np.cos(omega * x) - beta * np.sin(omega * x)) * np.cos(2.0 * np.pi * y)


def grad_y(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -2.0 * np.pi * np.exp(-beta * x) * np.sin(omega * x) * np.sin(2.0 * np.pi * y)


def interp2(U: np.ndarray, Wx: np.ndarray, Wy: np.ndarray) -> np.ndarray:
    return np.einsum('ki,ij,kj->k', Wx, U, Wy)


ue = field(xm, ym)
nde = grad_x(xm, ym) * nx + grad_y(xm, ym) * ny
scale_u = float(np.linalg.norm(ue))
scale_nd = float(np.linalg.norm(nde))
ds = radius * (2.0 * np.pi / M)
flux_exact = float(ds * np.sum(nde))


def evaluate(label: str, n: int) -> dict[str, float]:
    kind = ba.scheme_kind(label)
    degree = 6
    op = ba.build_operator(kind, degree, n)
    x = op.x
    X, Y = np.meshgrid(x, x, indexing='ij')
    U = field(X, Y)
    if ba.scheme_family(label) in ('E', 'C', 'SBP', 'DRP'):
        Wx = ba.local_interpolation_matrix(x, xm, degree + 1)
        Wy = ba.local_interpolation_matrix(x, ym, degree + 1)
        val = interp2(U, Wx, Wy)
        gx = interp2(op.differentiate(U), Wx, Wy)
        gy = interp2(U @ op.matrix.T, Wx, Wy)
    elif kind == 'B':
        coeff = lu_solve(op._lu, U)
        coeff = lu_solve(op._lu, coeff.T).T
        _, phix, dphix = ba.spline_eval_matrices(xm, x, degree)
        _, phiy, dphiy = ba.spline_eval_matrices(ym, x, degree)
        val = np.einsum('ki,ij,kj->k', phix, coeff, phiy)
        gx = np.einsum('ki,ij,kj->k', dphix, coeff, phiy)
        gy = np.einsum('ki,ij,kj->k', phix, coeff, dphiy)
    else:
        coeff = lu_solve(op._plu, U)
        coeff = lu_solve(op._plu, coeff.T).T
        gxc = op.differentiate_coefficients(coeff)
        gyc = op.differentiate_coefficients(coeff.transpose(1, 0)).transpose(1, 0)
        _, phix, _ = ba.spline_eval_matrices(xm, x, degree)
        _, phiy, _ = ba.spline_eval_matrices(ym, x, degree)
        val = np.einsum('ki,ij,kj->k', phix, coeff, phiy)
        gx = np.einsum('ki,ij,kj->k', phix, gxc, phiy)
        gy = np.einsum('ki,ij,kj->k', phix, gyc, phiy)
    nd = gx * nx + gy * ny
    flux = float(ds * np.sum(nd))
    return {
        'value_rel_l2': float(np.linalg.norm(val - ue) / scale_u),
        'normal_rel_l2': float(np.linalg.norm(nd - nde) / scale_nd),
        'flux_rel': float(abs(flux - flux_exact) / abs(flux_exact)),
        'work_2d': float(2.0 * n * op.work_model.total_flops),
    }


snapshot = {label: evaluate(label, SNAPSHOT_N) for label in LABELS}
convergence = {label: {'n': [], 'work_2d': [], 'normal_rel_l2': []} for label in LABELS}
for label in LABELS:
    for n in [33, 49, 65, 81]:
        data = evaluate(label, n)
        convergence[label]['n'].append(n)
        convergence[label]['work_2d'].append(data['work_2d'])
        convergence[label]['normal_rel_l2'].append(data['normal_rel_l2'])

ratio_value = snapshot['C6']['value_rel_l2'] / snapshot['BMF6']['value_rel_l2']
ratio_normal = snapshot['C6']['normal_rel_l2'] / snapshot['BMF6']['normal_rel_l2']
ratio_flux = snapshot['C6']['flux_rel'] / snapshot['BMF6']['flux_rel']

xplot = np.linspace(0.0, 0.45, 220)
yplot = np.linspace(0.28, 0.72, 220)
Xp, Yp = np.meshgrid(xplot, yplot, indexing='ij')
F = field(Xp, Yp)

setup_fig, setup_ax = plt.subplots(figsize=(6.0, 5.2))
setup_fig.subplots_adjust(left=0.11, right=0.90, bottom=0.12, top=0.90)
im = setup_ax.pcolormesh(xplot, yplot, F.T, shading='auto', cmap='RdBu_r')
setup_ax.plot(np.r_[xm, xm[0]], np.r_[ym, ym[0]], color='black', lw=1.7)
setup_ax.scatter(xm[::6], ym[::6], s=8, color='black', alpha=0.58)
setup_ax.axvline(0.0, color='black', lw=1.9)
sample = slice(None, None, 18)
setup_ax.quiver(
    xm[sample],
    ym[sample],
    0.03 * nx[sample],
    0.03 * ny[sample],
    angles='xy',
    scale_units='xy',
    scale=1.0,
    width=0.004,
    color='0.15',
    alpha=0.82,
)
setup_ax.set_xlim(0.0, 0.45)
setup_ax.set_ylim(0.28, 0.72)
setup_ax.set_aspect('equal', adjustable='box')
setup_ax.set_xlabel(r'$x$')
setup_ax.set_ylabel(r'$y$')
setup_ax.set_title('Immersed-circle setup and sampled field', fontweight='semibold')
setup_cb = setup_fig.colorbar(im, ax=setup_ax, fraction=0.046, pad=0.03)
setup_cb.set_label(r'$u(x,y)$')
save_figure(setup_fig, 'fig_s6_geometry_setup', pad_inches=0.03)
plt.close(setup_fig)

fig, axes = plt.subplots(1, 2, figsize=(10.9, 4.9), gridspec_kw={'wspace': 0.27})
fig.subplots_adjust(left=0.08, right=0.995, bottom=0.18, top=0.77)
ax1, ax2 = axes

cats = ['marker\nvalue', 'normal\nderivative', 'integrated\nnormal flux']
centers = np.arange(len(cats), dtype=float)
width = 0.12
offsets = np.linspace(-2.5 * width, 2.5 * width, len(LABELS))
for off, label in zip(offsets, LABELS, strict=True):
    vals = [snapshot[label]['value_rel_l2'], snapshot[label]['normal_rel_l2'], snapshot[label]['flux_rel']]
    ax1.bar(centers + off, vals, width=width, color=COLORS[ba.scheme_family(label)], alpha=0.92)
ax1.set_xticks(centers)
ax1.set_xticklabels(cats, linespacing=1.0)
ax1.set_yscale('log')
ax1.set_ylabel(rf'relative $H$-error at $N={SNAPSHOT_N}$', labelpad=1.5)
ax1.yaxis.set_label_coords(-0.10, 0.5)
ax1.set_title('(a) Value, normal, and flux', fontweight='semibold')
for label in LABELS:
    family = ba.scheme_family(label)
    ax2.loglog(convergence[label]['work_2d'], convergence[label]['normal_rel_l2'],
               marker=MARKERS[family], lw=2.3, ls=LINESTYLES[family], ms=6.0,
               mfc='white', mec=COLORS[family], mew=1.0, color=COLORS[family], label=label)
ax2.set_xlabel('recurring 2D gradient work [flops]')
ax2.set_ylabel(r'$\|\partial_n u-\partial_n u_{\mathrm{ex}}\|_H / \|\partial_n u_{\mathrm{ex}}\|_H$')
ax2.set_title('(b) Work versus marker-normal error', fontweight='semibold')
# The recurring 2D work is proportional to n^2 on this sweep, so an h^6
# asymptotic error law appears as a work^{-3} guide on these log-log axes.
add_theoretical_reference_line(
    ax2,
    slope=-0.5 * FORMAL_ORDER,
    label=rf'$\propto \mathcal{{W}}^{{-{FORMAL_ORDER // 2}}}$',
    candidate_regions=('lower right', 'mid right', 'upper right', 'upper left', 'lower left'),
)

handles = [
    Line2D([0], [0], color=COLORS[family], lw=2.3, ls=LINESTYLES[family], marker=MARKERS[family], ms=6.0, mfc='white', mec=COLORS[family], mew=1.0, label=f'{family}6')
    for family in ba.family_order()
]
fig.legend(handles, [handle.get_label() for handle in handles], ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.985), frameon=False, handlelength=2.3, columnspacing=1.35, handletextpad=0.58, fontsize=11.4)

save_figure(fig, 'fig_s6_geometry_transfer', pad_inches=0.03)
plt.close(fig)

summary = {
    f'snapshot_n{SNAPSHOT_N}': snapshot,
    'convergence': convergence,
    f'ratios_vs_C6_n{SNAPSHOT_N}': {
        'value': float(ratio_value),
        'normal_derivative': float(ratio_normal),
        'integrated_normal_flux': float(ratio_flux),
    },
}
OUT.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
