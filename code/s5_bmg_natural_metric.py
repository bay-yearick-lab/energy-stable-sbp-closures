#!/usr/bin/env python3
"""Generate `fig_s5_bmg_natural_metric.pdf`.

Section 5 companion figure to `fig_s5_bmf_staggered_cell`.  Where the BMF
schematic shows a horizontal staggered cell, this figure shows the BMG
reduce-solve-expand map on the primal Greville grid:

    primal Greville u_i  ---reduce--->  alpha = B^{-1} u
    coefficient stage    ---solve---->  beta  given by  M beta = K alpha
    coefficient stage    ---expand--->  D_BMG u = B beta

The signed wall trace ``B_partial`` enters through the clamped endpoints
``u_0=s(0)`` and ``u_{N-1}=s(1)`` (labels sit beside the trace stars,
no callout arrows).
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
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
BMG_COL = '#1f6f6f'
WALL = '#1a1a1a'
TRACE = '#1c7c34'
BOX_FACE = '#fbf3df'
BOX_EDGE = '#a8821b'
ACCENT = '#2F4858'


def main() -> None:
    N = 9
    P = 4

    knots = ba.open_uniform_knots(N, P)
    x_primal = ba.greville_from_knots(knots, P, N)

    alpha_seed = np.array(
        [0.04, 0.20, 0.50, 0.82, 0.96, 0.83, 0.55, 0.26, 0.05], dtype=float
    )
    s = BSpline(knots, alpha_seed.copy(), P, extrapolate=False)
    s_prime = s.derivative()

    xx = np.linspace(0.0, 1.0, 801)
    s_vals = np.asarray(s(xx), dtype=float)
    sp_vals = np.asarray(s_prime(xx), dtype=float)

    u_at_x = np.asarray(s(x_primal), dtype=float)
    sp_at_x = np.asarray(s_prime(x_primal), dtype=float)

    x_bmg, dbmg, _hbmg = ba.bmg_sbp_first_derivative_matrix(N, P)
    np.testing.assert_allclose(x_bmg, x_primal, atol=1e-12)
    bmg_at_x = dbmg @ u_at_x

    s_min, s_max = float(s_vals.min()), float(s_vals.max())
    sp_min, sp_max = (
        float(min(sp_vals.min(), bmg_at_x.min())),
        float(max(sp_vals.max(), bmg_at_x.max())),
    )

    y_top = 7.50
    y_bot = 0.80
    amp_top = 1.10
    amp_bot = 1.10
    box_top_y = 5.30
    box_bot_y = 3.00

    def y_field(values):
        return y_top + amp_top * (
            np.asarray(values, dtype=float) - s_min
        ) / (s_max - s_min + 1.0e-12)

    def y_deriv(values):
        return y_bot + amp_bot * (
            np.asarray(values, dtype=float) - sp_min
        ) / (sp_max - sp_min + 1.0e-12)

    fig, ax = plt.subplots(figsize=(11.6, 7.0), layout='constrained')

    # Top tier: primal field strip
    ax.plot(xx, y_field(s_vals), color=PRIMAL, lw=1.9, zorder=4)
    ax.plot([0.0, 1.0], [y_top, y_top], color='0.55', lw=0.7, zorder=1)
    for xi, vi in zip(x_primal, u_at_x):
        ax.plot([xi, xi], [y_top, y_field(float(vi))],
                color=PRIMAL, lw=0.7, alpha=0.5, zorder=2)
    ax.scatter(
        x_primal, np.full_like(x_primal, y_top),
        s=80, marker='o', facecolor=PRIMAL, edgecolor='white',
        linewidth=1.0, zorder=6,
    )

    # Bottom tier: BMG output strip
    ax.plot(xx, y_deriv(sp_vals), color='0.55', lw=1.0, alpha=0.55,
            zorder=3, ls=(0, (3.0, 1.6)))
    ax.plot([0.0, 1.0], [y_bot, y_bot], color='0.55', lw=0.7, zorder=1)
    for xi, vi in zip(x_primal, bmg_at_x):
        ax.plot([xi, xi], [y_bot, y_deriv(float(vi))],
                color=BMG_COL, lw=0.7, alpha=0.55, zorder=2)
    ax.scatter(
        x_primal, [y_deriv(float(v)) for v in bmg_at_x],
        s=80, marker='o', facecolor=BMG_COL, edgecolor='white',
        linewidth=1.0, zorder=6,
    )

    for xw in (0.0, 1.0):
        ax.plot([xw, xw], [0.10, y_top + amp_top + 0.55],
                color=WALL, lw=2.4, zorder=5, solid_capstyle='butt')

    # Slim reduce-solve-expand box: identities only, no per-stage gloss row.
    box_left, box_right = 0.10, 0.90
    box = FancyBboxPatch(
        (box_left, box_bot_y), box_right - box_left, box_top_y - box_bot_y,
        boxstyle='round,pad=0.02,rounding_size=0.04',
        facecolor=BOX_FACE, edgecolor=BOX_EDGE, lw=1.0, zorder=2,
    )
    ax.add_patch(box)

    accent_w = 0.011
    inset = 0.018
    accent = Rectangle(
        (box_left + inset, box_bot_y + 0.18), accent_w,
        (box_top_y - box_bot_y) - 0.36,
        facecolor=BMG_COL, edgecolor='none', alpha=0.95, zorder=4,
    )
    ax.add_patch(accent)

    stage_y = 0.5 * (box_top_y + box_bot_y)
    stage_y_label = stage_y + 0.45
    stage_y_eq = stage_y - 0.05

    ax.text(0.22, stage_y_label, r'reduce',
            ha='center', va='bottom', fontsize=11.4,
            color=ACCENT, style='italic')
    ax.text(0.22, stage_y_eq, r'$\alpha = B^{-1}u$',
            ha='center', va='center', fontsize=14.0, color='0.10')

    arrow_rs = FancyArrowPatch(
        (0.32, stage_y_eq), (0.42, stage_y_eq),
        arrowstyle='-|>', mutation_scale=14, color='0.25', lw=1.3, zorder=5,
    )
    ax.add_patch(arrow_rs)

    ax.text(0.50, stage_y_label, r'solve',
            ha='center', va='bottom', fontsize=11.4,
            color=ACCENT, style='italic')
    ax.text(0.50, stage_y_eq, r'$M\,\beta = K\,\alpha$',
            ha='center', va='center', fontsize=14.0, color='0.10')

    arrow_se = FancyArrowPatch(
        (0.58, stage_y_eq), (0.68, stage_y_eq),
        arrowstyle='-|>', mutation_scale=14, color='0.25', lw=1.3, zorder=5,
    )
    ax.add_patch(arrow_se)

    ax.text(0.78, stage_y_label, r'expand',
            ha='center', va='bottom', fontsize=11.4,
            color=ACCENT, style='italic')
    ax.text(0.78, stage_y_eq, r'$D_{\mathrm{BMG}}u = B\,\beta$',
            ha='center', va='center', fontsize=14.0, color='0.10')

    # One short italic line below the box names the role of the box.
    ax.text(
        0.5, box_bot_y - 0.40,
        r'coefficient stage in the natural spline metric '
        r'$H_{\mathrm{BMG}}=B^{-\!\top}M B^{-1}$',
        ha='center', va='top', fontsize=11.6, color=ACCENT,
        family='serif', style='italic',
    )

    arrow_top = FancyArrowPatch(
        (0.5, y_top - 0.18), (0.5, box_top_y + 0.04),
        arrowstyle='-|>', mutation_scale=15, color='0.20', lw=1.5, zorder=5,
    )
    ax.add_patch(arrow_top)

    arrow_bot = FancyArrowPatch(
        (0.5, box_bot_y - 0.04), (0.5, y_bot + 0.18),
        arrowstyle='-|>', mutation_scale=15, color='0.20', lw=1.5, zorder=5,
    )
    ax.add_patch(arrow_bot)

    # Endpoint trace stars at the wall on the field strip
    y_left_trace = float(y_field(u_at_x[0]))
    y_right_trace = float(y_field(u_at_x[-1]))
    ax.scatter(
        [0.0, 1.0], [y_left_trace, y_right_trace],
        s=200, marker='*', facecolor=TRACE, edgecolor='white',
        linewidth=1.0, zorder=8,
    )

    # Quiet labels in the outer margins beside the stars (no callout arrows).
    ax.text(
        -0.030, y_left_trace, r'$u_0 = s(0)$',
        ha='right', va='center', fontsize=11.4, color=TRACE,
        family='serif', style='italic',
    )
    ax.text(
        1.030, y_right_trace, r'$u_{N-1} = s(1)$',
        ha='left', va='center', fontsize=11.4, color=TRACE,
        family='serif', style='italic',
    )

    # Strip labels (math + italic gloss as separate text calls).
    ax.text(
        0.5, y_top + amp_top + 0.95,
        r"primal Greville $\{x_i\}$,$\quad u_i = s(x_i) \in S_h^{p}$",
        ha='center', va='bottom', fontsize=12.0, color=PRIMAL,
    )
    ax.text(
        0.5, y_top + amp_top + 0.55,
        r"sample input",
        ha='center', va='bottom', fontsize=10.8, color=PRIMAL,
        family='serif', style='italic', alpha=0.75,
    )
    ax.text(
        0.5, y_bot - 0.95,
        r"primal Greville $\{x_i\}$,$\quad (D_{\mathrm{BMG}}u)_i$ "
        r"in $H_{\mathrm{BMG}}$",
        ha='center', va='top', fontsize=12.0, color=BMG_COL,
    )
    ax.text(
        0.5, y_bot - 1.30,
        r"output",
        ha='center', va='top', fontsize=10.8, color=BMG_COL,
        family='serif', style='italic', alpha=0.75,
    )

    ax.text(0.0, -0.25, r'wall, $x{=}0$', ha='center', va='top', fontsize=11.2)
    ax.text(1.0, -0.25, r'wall, $x{=}1$', ha='center', va='top', fontsize=11.2)

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.55, y_top + amp_top + 1.55)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)

    save_figure(fig, 'fig_s5_bmg_natural_metric', pad_inches=0.04)
    plt.close(fig)
    print('wrote BMG natural-metric companion figure')
    print(f'  max |D_BMG u - s\'(x)| = {float(np.max(np.abs(bmg_at_x - sp_at_x))):.3e}')


if __name__ == '__main__':
    main()
