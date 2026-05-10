#!/usr/bin/env python3
"""Generate `fig_s3_polynomial_reproduction.pdf`.

Visual companion to Proposition 3.5.  On a `p=4`, `N=9` grid:

    Panel (a):  the spline reconstruction of u(x)=x^4 sits exactly on the
                exact polynomial; an inset twin panel reports the pointwise
                reconstruction error in O(eps_machine).

    Panel (b):  the exact derivative 4 x^3 as a continuous curve, with the
                BMF staggered derivative at the derivative Greville grid
                (blue squares) and the BMG same-grid derivative at the
                primal Greville grid (red diamonds).  Both marker sets sit
                on the curve to numerical precision.

The figure makes the polynomial-reproduction property of the bounded
approximation space visually concrete: every polynomial of degree <= p is
already in S_h^p, so the operators return p x^{p-1} at every Greville point.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

import boundary_spline_analysis as ba
from project_paths import ensure_generated_dirs, save_figure


ensure_generated_dirs()

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10.0,
    'axes.titlesize': 10.4,
    'axes.labelsize': 9.6,
    'xtick.labelsize': 8.6,
    'ytick.labelsize': 8.6,
    'figure.dpi': 180,
    'savefig.dpi': 300,
})


ACCENT = '#2F4858'
EXACT = '0.20'
PANEL_BG = (0.987, 0.985, 0.978)
BMF_COL = ba.DEFAULT_COLORS['BMF']
BMG_COL = ba.DEFAULT_COLORS['BMG']


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL_BG)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('bottom', 'left'):
        ax.spines[s].set_color(ACCENT)
        ax.spines[s].set_linewidth(0.55)
    ax.tick_params(colors=ACCENT, length=2.8, width=0.6)
    ax.grid(True, which='major', alpha=0.18, lw=0.5)


def main() -> None:
    p = 4
    N = 9

    knots = ba.open_uniform_knots(N, p)
    nodes_primal, _ke, bmat, _gmat = ba.clamped_bspline_matrices(N, p)

    u_nodes = nodes_primal**p

    alpha = np.linalg.solve(bmat, u_nodes)

    sample = np.linspace(0.0, 1.0, 1001)
    spline_u = BSpline(knots, alpha, p, extrapolate=False)
    recon = np.nan_to_num(spline_u(sample))
    exact = sample**p
    err = recon - exact

    x_primal, eta_grid, d10, _d01 = ba.bmf_staggered_first_derivative_matrices(N, p)
    bmf_deriv = d10 @ u_nodes
    exact_bmf = p * eta_grid**(p - 1)

    x_bmg, dbmg, _hbmg = ba.bmg_sbp_first_derivative_matrix(N, p)
    bmg_deriv = dbmg @ u_nodes
    exact_bmg = p * x_bmg**(p - 1)

    fig = plt.figure(figsize=(10.0, 4.1), layout='constrained')
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[3.5, 1.0],
        width_ratios=[1.0, 1.0],
        hspace=0.05,
        wspace=0.16,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a_err = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_b = fig.add_subplot(gs[:, 1])

    ax_a.plot(sample, exact, color=EXACT, lw=1.4,
              label=r'exact $u(x)=x^{4}$', zorder=2)
    ax_a.plot(sample, recon, color=ACCENT, lw=1.0, linestyle=(0, (3, 2)),
              label=r'spline reconstruction', zorder=3)
    ax_a.scatter(nodes_primal, u_nodes, s=22,
                 facecolor='white', edgecolor=ACCENT, lw=0.9,
                 zorder=4, label=r'Greville samples')

    ax_a.set_xlim(-0.01, 1.01)
    ax_a.set_ylabel(r'$u(x)$', color=ACCENT)
    ax_a.set_title(r'(a) spline space reproduces $\mathbb{P}_p$ exactly',
                   color=ACCENT, pad=6)
    _style_axis(ax_a)
    ax_a.tick_params(labelbottom=False)
    ax_a.legend(frameon=False, loc='upper left', fontsize=8.6)

    ax_a.text(
        0.96, 0.06,
        rf'$\max\,|u-u_h|={np.max(np.abs(err)):.1e}$',
        transform=ax_a.transAxes,
        ha='right', va='bottom', fontsize=8.4,
        color='0.30', family='serif', style='italic',
    )

    ax_a_err.axhline(0.0, color='0.65', lw=0.6, zorder=0)
    ax_a_err.plot(sample, err, color=ACCENT, lw=0.9, zorder=2)
    ax_a_err.set_ylabel(r'$u-u_h$', fontsize=8.6, color=ACCENT)
    ax_a_err.set_xlabel(r'$x$', color=ACCENT)
    _style_axis(ax_a_err)
    ax_a_err.tick_params(labelsize=8.0)
    err_max = max(np.max(np.abs(err)), 1e-16)
    ax_a_err.set_ylim(-1.5 * err_max, 1.5 * err_max)
    ax_a_err.ticklabel_format(axis='y', style='sci',
                              scilimits=(0, 0), useMathText=True)

    sample_b = np.linspace(0.0, 1.0, 1001)
    exact_d = p * sample_b**(p - 1)
    ax_b.plot(sample_b, exact_d, color=EXACT, lw=1.4,
              label=r'exact $4x^{3}$', zorder=2)
    ax_b.scatter(eta_grid, bmf_deriv, marker='s', s=44,
                 facecolor=BMF_COL, edgecolor='white', lw=0.6,
                 zorder=4,
                 label=r'BMF on $\eta$-grid')
    ax_b.scatter(x_bmg, bmg_deriv, marker='D', s=42,
                 facecolor=BMG_COL, edgecolor='white', lw=0.6,
                 zorder=5,
                 label=r'BMG on primal grid')

    ax_b.set_xlim(-0.01, 1.01)
    ax_b.set_xlabel(r'$x$', color=ACCENT)
    ax_b.set_ylabel(r"$u'(x)$", color=ACCENT)
    ax_b.set_title(r"(b) BMF and BMG return $p\,x^{\,p-1}$ at every Greville point",
                   color=ACCENT, pad=6)
    _style_axis(ax_b)
    ax_b.legend(frameon=False, loc='upper left', fontsize=8.6)

    bmf_err = np.max(np.abs(bmf_deriv - exact_bmf))
    bmg_err = np.max(np.abs(bmg_deriv - exact_bmg))
    ax_b.text(
        0.96, 0.06,
        rf'BMF residual $={bmf_err:.1e}$' '\n' rf'BMG residual $={bmg_err:.1e}$',
        transform=ax_b.transAxes,
        ha='right', va='bottom', fontsize=8.4,
        color='0.30', family='serif', style='italic',
    )

    save_figure(fig, 'fig_s3_polynomial_reproduction', pad_inches=0.05)
    plt.close(fig)
    print('wrote polynomial reproduction figure')


if __name__ == '__main__':
    main()
