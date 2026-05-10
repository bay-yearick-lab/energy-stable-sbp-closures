#!/usr/bin/env python3
"""Schematic for Subsection 5.3: clamped spline trace operators on both branches.

Two-panel figure that shows how the same primal Greville grid carries the
trace operators on each branch:

  * Dirichlet trace tau_D u = (u_0, u_{N-1}), the clamped endpoint identity.
  * BMF Neumann trace tau_N^{BMF} u, through the staggered gradient onto the
    shortened-degree derivative Greville grid.
  * BMG Neumann trace tau_N^{BMG} u, through the wall rows of D_BMG on the
    same primal grid.

Trace formulas live in the figure caption; the figure shows only the
geometric placement of the trace stars and the route between strips.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
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


PRIMAL = '#22517a'
DERIV = '#a83232'
BMG_COL = '#1f6f6f'
WALL = '#1a1a1a'
DIR_COL = '#22517a'
NEU_COL = '#a83232'
NEU_BMG = '#1f6f6f'
PANEL_BG = '#fbf3df'
PANEL_BG_BMF = '#f6e2dc'
PANEL_BG_BMG = '#dde9e6'
ACCENT = '#2F4858'


def _spline_state(N: int, P: int) -> dict:
    knots_p = ba.open_uniform_knots(N, P)
    x_primal = ba.greville_from_knots(knots_p, P, N)
    knots_pm1 = ba.derivative_space_knots(knots_p)
    eta = ba.derivative_greville_nodes(knots_pm1, P - 1)

    alpha = np.array(
        [0.04, 0.20, 0.50, 0.82, 0.96, 0.83, 0.55, 0.26, 0.05], dtype=float
    )
    assert alpha.size == N
    s = BSpline(knots_p, alpha.copy(), P, extrapolate=False)
    sprime = s.derivative()

    xx = np.linspace(0.0, 1.0, 801)
    s_vals = np.asarray(s(xx), dtype=float)
    sp_vals = np.asarray(sprime(xx), dtype=float)

    u_at_x = np.asarray(s(x_primal), dtype=float)
    q_at_eta = np.asarray(sprime(eta), dtype=float)

    return {
        'x_primal': x_primal,
        'eta': eta,
        'xx': xx,
        's_vals': s_vals,
        'sp_vals': sp_vals,
        'u_at_x': u_at_x,
        'q_at_eta': q_at_eta,
        's_left': float(s(0.0)),
        's_right': float(s(1.0)),
        'sp_left': float(sprime(0.0)),
        'sp_right': float(sprime(1.0)),
    }


def _draw_panel_bg(ax, x0, y0, x1, y1, color=None):
    rect = Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        facecolor=color if color is not None else PANEL_BG,
        edgecolor='none', alpha=0.55, zorder=0,
    )
    ax.add_patch(rect)


def _draw_walls(ax, y_lo, y_hi):
    for xw in (0.0, 1.0):
        ax.plot(
            [xw, xw], [y_lo, y_hi],
            color=WALL, lw=2.0, zorder=5, solid_capstyle='butt',
        )


def draw_bmf_panel(ax, state):
    x_primal = state['x_primal']
    eta = state['eta']
    xx = state['xx']
    s_vals = state['s_vals']
    sp_vals = state['sp_vals']
    u_at_x = state['u_at_x']
    q_at_eta = state['q_at_eta']

    y_primal = 5.0
    y_deriv = 2.0
    amp_p = 1.4
    amp_d = 1.4

    s_min, s_max = float(s_vals.min()), float(s_vals.max())
    sp_min, sp_max = float(sp_vals.min()), float(sp_vals.max())

    def primal_y(values):
        return y_primal + amp_p * (values - s_min) / (s_max - s_min + 1.0e-12)

    def deriv_y(values):
        return y_deriv + amp_d * (values - sp_min) / (sp_max - sp_min + 1.0e-12)

    _draw_panel_bg(ax, -0.25, -0.40, 1.25, y_primal + amp_p + 1.40,
                   color=PANEL_BG_BMF)

    ax.plot(xx, primal_y(s_vals), color=PRIMAL, lw=1.7, zorder=4)
    ax.plot(xx, deriv_y(sp_vals), color=DERIV, lw=1.7, zorder=4)

    ax.plot([0.0, 1.0], [y_primal, y_primal], color='0.55', lw=0.6, zorder=1)
    ax.plot([0.0, 1.0], [y_deriv, y_deriv], color='0.55', lw=0.6, zorder=1)

    for xi, vi in zip(x_primal, u_at_x):
        ax.plot([xi, xi], [y_primal, primal_y(float(vi))],
                color=PRIMAL, lw=0.55, alpha=0.40, zorder=2)
    for ej, vj in zip(eta, q_at_eta):
        ax.plot([ej, ej], [y_deriv, deriv_y(float(vj))],
                color=DERIV, lw=0.55, alpha=0.40, zorder=2)

    ax.scatter(
        x_primal, np.full_like(x_primal, y_primal),
        s=78, marker='o', facecolor=PRIMAL, edgecolor='white', linewidth=1.0,
        zorder=6,
    )
    ax.scatter(
        eta, np.full_like(eta, y_deriv),
        s=82, marker='s', facecolor=DERIV, edgecolor='white', linewidth=1.0,
        zorder=6,
    )

    _draw_walls(ax, 0.55, y_primal + amp_p + 0.55)

    # Dirichlet trace stars on the primal endpoints
    y_DL = float(primal_y(u_at_x[0]))
    y_DR = float(primal_y(u_at_x[-1]))
    ax.scatter(
        [0.0, 1.0], [y_DL, y_DR],
        s=200, marker='*', facecolor=DIR_COL, edgecolor='white', linewidth=1.0,
        zorder=8,
    )

    # Neumann trace stars on the derivative-grid endpoints (extrapolated)
    y_NL = float(deriv_y(state['sp_left']))
    y_NR = float(deriv_y(state['sp_right']))
    ax.scatter(
        [0.0, 1.0], [y_NL, y_NR],
        s=210, marker='*', facecolor=NEU_COL, edgecolor='white', linewidth=1.0,
        zorder=8,
    )
    ax.plot(
        [eta[0], 0.0], [deriv_y(float(q_at_eta[0])), y_NL],
        color=DERIV, lw=0.9, ls=(0, (2.0, 1.4)), alpha=0.45, zorder=3,
    )
    ax.plot(
        [eta[-1], 1.0], [deriv_y(float(q_at_eta[-1])), y_NR],
        color=DERIV, lw=0.9, ls=(0, (2.0, 1.4)), alpha=0.45, zorder=3,
    )

    # Quiet trace labels in the outer margins beside each star.
    ax.text(-0.030, y_DL, r'$\tau_D$', ha='right', va='center',
            fontsize=12.4, color=DIR_COL, family='serif', style='italic')
    ax.text(1.030, y_DR, r'$\tau_D$', ha='left', va='center',
            fontsize=12.4, color=DIR_COL, family='serif', style='italic')
    ax.text(-0.030, y_NL, r'$\tau_N^{\mathrm{BMF}}$', ha='right', va='center',
            fontsize=12.4, color=NEU_COL, family='serif', style='italic')
    ax.text(1.030, y_NR, r'$\tau_N^{\mathrm{BMF}}$', ha='left', va='center',
            fontsize=12.4, color=NEU_COL, family='serif', style='italic')

    # G_BMF arrow connecting primal to derivative strip
    gx = -0.130
    arrow_g = FancyArrowPatch(
        (gx, y_primal - 0.05), (gx, y_deriv + 0.05),
        arrowstyle='-|>', mutation_scale=14, color='0.20', lw=1.4, zorder=5,
    )
    ax.add_patch(arrow_g)
    ax.text(
        gx - 0.020, 0.5 * (y_primal + y_deriv), r'$G_{\mathrm{BMF}}$',
        ha='right', va='center', fontsize=13.0, color='0.10',
    )

    ax.text(
        0.5, y_primal + amp_p + 1.05,
        r"primal Greville $\{x_i\}$,$\quad u_i = s(x_i)$",
        ha='center', va='bottom', fontsize=11.8, color=PRIMAL,
    )
    ax.text(
        0.5, y_deriv - 0.50,
        r"derivative Greville $\{\eta_j\}$,$\quad q_j = s'(\eta_j)$",
        ha='center', va='top', fontsize=11.8, color=DERIV,
    )

    ax.text(0.0, 0.30, r'wall, $x{=}0$', ha='center', va='top', fontsize=11.0)
    ax.text(1.0, 0.30, r'wall, $x{=}1$', ha='center', va='top', fontsize=11.0)

    ax.set_title(r'(a) $\mathrm{BMF}$ branch', loc='center', fontsize=13.0,
                 color=ACCENT, pad=10)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.40, y_primal + amp_p + 1.40)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_bmg_panel(ax, state):
    x_primal = state['x_primal']
    xx = state['xx']
    s_vals = state['s_vals']
    sp_vals = state['sp_vals']
    u_at_x = state['u_at_x']

    N = x_primal.size
    P = 4
    x_check, dbmg, _hbmg = ba.bmg_sbp_first_derivative_matrix(N, P)
    np.testing.assert_allclose(x_check, x_primal, atol=1e-12)
    bmg_at_x = dbmg @ u_at_x
    bmg_left = float(bmg_at_x[0])
    bmg_right = float(bmg_at_x[-1])

    y_primal = 5.0
    y_bmg = 2.0
    amp_p = 1.4
    amp_b = 1.4

    s_min, s_max = float(s_vals.min()), float(s_vals.max())
    bmg_min, bmg_max = (
        float(min(sp_vals.min(), bmg_at_x.min())),
        float(max(sp_vals.max(), bmg_at_x.max())),
    )

    def primal_y(values):
        return y_primal + amp_p * (values - s_min) / (s_max - s_min + 1.0e-12)

    def bmg_y(values):
        return y_bmg + amp_b * (values - bmg_min) / (bmg_max - bmg_min + 1.0e-12)

    _draw_panel_bg(ax, -0.25, -0.40, 1.25, y_primal + amp_p + 1.40,
                   color=PANEL_BG_BMG)

    ax.plot(xx, primal_y(s_vals), color=PRIMAL, lw=1.7, zorder=4)
    ax.plot(xx, bmg_y(sp_vals), color=BMG_COL, lw=1.7, alpha=0.85, zorder=4)

    ax.plot([0.0, 1.0], [y_primal, y_primal], color='0.55', lw=0.6, zorder=1)
    ax.plot([0.0, 1.0], [y_bmg, y_bmg], color='0.55', lw=0.6, zorder=1)

    for xi, vi in zip(x_primal, u_at_x):
        ax.plot([xi, xi], [y_primal, primal_y(float(vi))],
                color=PRIMAL, lw=0.55, alpha=0.40, zorder=2)
    for xi, vi in zip(x_primal, bmg_at_x):
        ax.plot([xi, xi], [y_bmg, bmg_y(float(vi))],
                color=BMG_COL, lw=0.55, alpha=0.40, zorder=2)

    ax.scatter(
        x_primal, np.full_like(x_primal, y_primal),
        s=78, marker='o', facecolor=PRIMAL, edgecolor='white', linewidth=1.0,
        zorder=6,
    )
    ax.scatter(
        x_primal, np.full_like(x_primal, y_bmg),
        s=78, marker='o', facecolor=BMG_COL, edgecolor='white', linewidth=1.0,
        zorder=6,
    )

    _draw_walls(ax, 0.55, y_primal + amp_p + 0.55)

    y_DL = float(primal_y(u_at_x[0]))
    y_DR = float(primal_y(u_at_x[-1]))
    ax.scatter(
        [0.0, 1.0], [y_DL, y_DR],
        s=200, marker='*', facecolor=DIR_COL, edgecolor='white', linewidth=1.0,
        zorder=8,
    )

    y_NL = float(bmg_y(bmg_left))
    y_NR = float(bmg_y(bmg_right))
    ax.scatter(
        [0.0, 1.0], [y_NL, y_NR],
        s=210, marker='*', facecolor=NEU_BMG, edgecolor='white', linewidth=1.0,
        zorder=8,
    )

    ax.text(-0.030, y_DL, r'$\tau_D$', ha='right', va='center',
            fontsize=12.4, color=DIR_COL, family='serif', style='italic')
    ax.text(1.030, y_DR, r'$\tau_D$', ha='left', va='center',
            fontsize=12.4, color=DIR_COL, family='serif', style='italic')
    ax.text(-0.030, y_NL, r'$\tau_N^{\mathrm{BMG}}$', ha='right', va='center',
            fontsize=12.4, color=NEU_BMG, family='serif', style='italic')
    ax.text(1.030, y_NR, r'$\tau_N^{\mathrm{BMG}}$', ha='left', va='center',
            fontsize=12.4, color=NEU_BMG, family='serif', style='italic')

    gx = -0.130
    arrow_g = FancyArrowPatch(
        (gx, y_primal - 0.05), (gx, y_bmg + 0.05),
        arrowstyle='-|>', mutation_scale=14, color='0.20', lw=1.4, zorder=5,
    )
    ax.add_patch(arrow_g)
    ax.text(
        gx - 0.020, 0.5 * (y_primal + y_bmg), r'$D_{\mathrm{BMG}}$',
        ha='right', va='center', fontsize=13.0, color='0.10',
    )

    ax.text(
        0.5, y_primal + amp_p + 1.05,
        r"primal Greville $\{x_i\}$,$\quad u_i = s(x_i)$",
        ha='center', va='bottom', fontsize=11.8, color=PRIMAL,
    )
    ax.text(
        0.5, y_bmg - 0.50,
        r"primal Greville $\{x_i\}$,$\quad (D_{\mathrm{BMG}} u)_i$",
        ha='center', va='top', fontsize=11.8, color=BMG_COL,
    )

    ax.text(0.0, 0.30, r'wall, $x{=}0$', ha='center', va='top', fontsize=11.0)
    ax.text(1.0, 0.30, r'wall, $x{=}1$', ha='center', va='top', fontsize=11.0)

    ax.set_title(r'(b) $\mathrm{BMG}$ branch', loc='center', fontsize=13.0,
                 color=ACCENT, pad=10)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.40, y_primal + amp_p + 1.40)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def main() -> None:
    state = _spline_state(N=9, P=4)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4), layout='constrained')
    draw_bmf_panel(axes[0], state)
    draw_bmg_panel(axes[1], state)

    save_figure(fig, 'fig_s5_strong_bc_traces', pad_inches=0.04)
    plt.close(fig)
    print('wrote strong-bc-traces schematic')


if __name__ == '__main__':
    main()
