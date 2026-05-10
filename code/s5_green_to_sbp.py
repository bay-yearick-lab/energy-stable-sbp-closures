#!/usr/bin/env python3
"""Generate `fig_s5_green_to_sbp.pdf`.

Section 5 opening figure.  Strang-style two-panel figure that walks the
reader from continuous integration by parts to its discrete SBP twin.

Panel (a):  continuous identity on `[0,1]`
            int_0^1 (u v' + u' v) dx = [u v]_0^1
            Two smooth functions are drawn; the wall-trace markers
            highlight the only surviving boundary contribution.

Panel (b):  discrete identity in the natural spline metric
            u^T H D v + (D u)^T H v = -u_0 v_0 + u_{N-1} v_{N-1}
            The same two functions, now sampled on the clamped Greville
            grid, illustrate that exactly the same wall trace survives,
            in algebraic form, after the bilinear and metric algebra.

A slim bridging band below the two panels carries the punchline
``[uv]_0^1 = -0.260 = -u_0 v_0 + u_{N-1} v_{N-1}`` so the matching wall
trace is the visual climax.
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
    'font.size': 12.5,
    'axes.titlesize': 13.5,
    'axes.labelsize': 12.0,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'figure.dpi': 180,
    'savefig.dpi': 300,
})


ACCENT = '#2F4858'
PANEL_BG = (0.987, 0.985, 0.978)
U_COLOR = '#22517a'
V_COLOR = '#a83232'
WALL_COLOR = '#7a4f1d'
TRACE_COLOR = '#1c7c34'


def style_panel(ax, *, ylabel=None, xlabel=None, ylim=None, panel_bg=True,
                show_xticks=True):
    if panel_bg:
        ax.set_facecolor(PANEL_BG)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12.0)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12.0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(0.0, color='0.6', lw=0.6, zorder=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(0.65)
        ax.spines[s].set_color('0.30')
    ax.tick_params(length=3.0, width=0.65, pad=2.5)
    if not show_xticks:
        ax.tick_params(labelbottom=False)
    ax.set_xlim(-0.012, 1.012)


def headline(ax, headline_text, formula_text):
    """Two-line panel title: descriptive headline above, italic formula below."""
    ax.text(0.5, 1.13, headline_text,
            ha='center', va='bottom', transform=ax.transAxes,
            fontsize=12.6, color=ACCENT, family='serif')
    ax.text(0.5, 1.025, formula_text,
            ha='center', va='bottom', transform=ax.transAxes,
            fontsize=11.4, color='0.30', family='serif', style='italic')


def build_test_functions():
    """Smooth, distinct test functions u(x) and v(x) with nontrivial wall traces."""
    u = lambda x: 0.85 * np.sin(np.pi * x) + 0.30 * np.cos(0.5 * np.pi * x) - 0.05
    v = lambda x: 0.50 + 0.55 * np.cos(np.pi * x) - 0.20 * np.sin(2.0 * np.pi * x)
    return u, v


def main() -> None:
    u, v = build_test_functions()
    x_dense = np.linspace(0.0, 1.0, 1001)

    u0, u1 = float(u(0.0)), float(u(1.0))
    v0, v1 = float(v(0.0)), float(v(1.0))
    boundary_term = -u0 * v0 + u1 * v1

    N = 11
    p = 4
    knots = ba.open_uniform_knots(N, p)
    x_primal = ba.greville_from_knots(knots, p, N)
    nodes_primal, _ke, bmat, _gmat = ba.clamped_bspline_matrices(N, p)
    np.testing.assert_allclose(nodes_primal, x_primal, atol=1e-12)

    x_bmg, dbmg, hbmg = ba.bmg_sbp_first_derivative_matrix(N, p)
    np.testing.assert_allclose(x_bmg, x_primal, atol=1e-12)
    H = hbmg

    u_nodes = u(x_primal)
    v_nodes = v(x_primal)
    Du = dbmg @ u_nodes
    Dv = dbmg @ v_nodes

    discrete_lhs = float(u_nodes @ (H @ Dv) + Du @ (H @ v_nodes))
    discrete_rhs = float(-u_nodes[0] * v_nodes[0] + u_nodes[-1] * v_nodes[-1])
    print(f'  continuous boundary term [uv]_0^1 = {boundary_term:.6f}')
    print(f'  discrete LHS u^T H D v + (Du)^T H v = {discrete_lhs:.6e}')
    print(f'  discrete RHS -u_0 v_0 + u_{{N-1}} v_{{N-1}} = {discrete_rhs:.6e}')
    print(f'  closure mismatch (LHS - RHS) = {discrete_lhs - discrete_rhs:.3e}')

    u_dense = u(x_dense)
    v_dense = v(x_dense)
    y_lim = (
        1.18 * float(min(u_dense.min(), v_dense.min(), 0.0) - 0.05),
        1.18 * float(max(u_dense.max(), v_dense.max(), 0.0) + 0.05),
    )
    y_lim_top = (min(y_lim[0], -1.05), max(y_lim[1], 1.55))

    fig = plt.figure(figsize=(13.0, 5.6), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.13],
                          width_ratios=[1.0, 1.0],
                          wspace=0.10, hspace=0.05)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_bridge = fig.add_subplot(gs[1, :])

    style_panel(ax_a, ylabel='amplitude', xlabel=r'$x$', ylim=y_lim_top)
    headline(
        ax_a,
        '(a) continuous: integration by parts leaves only the wall trace',
        r'$\int_0^1 (uv^{\prime}+u^{\prime}v)\,dx = [uv]_0^1$',
    )
    ax_a.axvline(0.0, color=WALL_COLOR, lw=1.2, alpha=0.55, zorder=0)
    ax_a.axvline(1.0, color=WALL_COLOR, lw=1.2, alpha=0.55, zorder=0)
    ax_a.fill_between(x_dense, u_dense * v_dense, 0.0,
                      color=TRACE_COLOR, alpha=0.10, zorder=1,
                      label=r'$uv$')
    ax_a.plot(x_dense, u_dense, color=U_COLOR, lw=1.85, label=r'$u(x)$')
    ax_a.plot(x_dense, v_dense, color=V_COLOR, lw=1.85, label=r'$v(x)$')
    ax_a.plot([0.0], [u0 * v0], marker='*', markersize=16.0,
              color=TRACE_COLOR, markeredgecolor='white', markeredgewidth=0.8,
              zorder=4)
    ax_a.plot([1.0], [u1 * v1], marker='*', markersize=16.0,
              color=TRACE_COLOR, markeredgecolor='white', markeredgewidth=0.8,
              zorder=4)
    ax_a.text(0.03, 0.93, r'wall', ha='left', va='top',
              transform=ax_a.transAxes, fontsize=10.5,
              color=WALL_COLOR, family='serif', style='italic')
    ax_a.text(0.97, 0.93, r'wall', ha='right', va='top',
              transform=ax_a.transAxes, fontsize=10.5,
              color=WALL_COLOR, family='serif', style='italic')
    ax_a.legend(loc='upper right', frameon=False, fontsize=10.5,
                handlelength=1.6, ncol=3, bbox_to_anchor=(1.0, 0.88))

    style_panel(ax_b, xlabel=r'$x$', ylim=y_lim_top)
    headline(
        ax_b,
        '(b) discrete BMG twin: same wall trace survives in the natural metric',
        (r'$u^{\!\top} H Dv + (Du)^{\!\top} H v '
         r'= -u_0 v_0 + u_{N-1} v_{N-1}$'),
    )
    ax_b.axvline(0.0, color=WALL_COLOR, lw=1.2, alpha=0.55, zorder=0)
    ax_b.axvline(1.0, color=WALL_COLOR, lw=1.2, alpha=0.55, zorder=0)
    ax_b.plot(x_dense, u_dense, color=U_COLOR, lw=1.0, alpha=0.55, zorder=1)
    ax_b.plot(x_dense, v_dense, color=V_COLOR, lw=1.0, alpha=0.55, zorder=1)
    ax_b.plot(x_primal, u_nodes, marker='o', linestyle='None',
              markersize=5.6, color=U_COLOR,
              markeredgecolor='white', markeredgewidth=0.6,
              zorder=3, label=r'$u_i$')
    ax_b.plot(x_primal, v_nodes, marker='s', linestyle='None',
              markersize=5.8, color=V_COLOR,
              markeredgecolor='white', markeredgewidth=0.6,
              zorder=3, label=r'$v_i$')
    ax_b.plot([0.0], [u_nodes[0] * v_nodes[0]], marker='*', markersize=16.0,
              color=TRACE_COLOR, markeredgecolor='white', markeredgewidth=0.8,
              zorder=4)
    ax_b.plot([1.0], [u_nodes[-1] * v_nodes[-1]], marker='*', markersize=16.0,
              color=TRACE_COLOR, markeredgecolor='white', markeredgewidth=0.8,
              zorder=4)
    ax_b.text(0.03, 0.93, r'wall', ha='left', va='top',
              transform=ax_b.transAxes, fontsize=10.5,
              color=WALL_COLOR, family='serif', style='italic')
    ax_b.text(0.97, 0.93, r'wall', ha='right', va='top',
              transform=ax_b.transAxes, fontsize=10.5,
              color=WALL_COLOR, family='serif', style='italic')
    ax_b.legend(loc='upper right', frameon=False, fontsize=10.5,
                handlelength=1.4, ncol=2, bbox_to_anchor=(1.0, 0.88))

    # Bridging band: punchline showing the matching wall trace.
    ax_bridge.set_facecolor(PANEL_BG)
    for s in ('top', 'right', 'left', 'bottom'):
        ax_bridge.spines[s].set_visible(False)
    ax_bridge.set_xticks([])
    ax_bridge.set_yticks([])
    ax_bridge.set_xlim(0.0, 1.0)
    ax_bridge.set_ylim(0.0, 1.0)
    ax_bridge.text(
        0.50, 0.55,
        rf'$[uv]_0^1 \approx {boundary_term:.3f} \approx '
        rf'-u_0 v_0 + u_{{N-1}} v_{{N-1}}$',
        ha='center', va='center', fontsize=14.0, color=ACCENT,
    )
    ax_bridge.text(
        0.50, 0.05, 'matching wall trace',
        ha='center', va='center', fontsize=10.6, color='0.35',
        family='serif', style='italic',
    )

    save_figure(fig, 'fig_s5_green_to_sbp', pad_inches=0.05)
    plt.close(fig)
    print('wrote section 5 Green-to-SBP figure')


if __name__ == '__main__':
    main()
