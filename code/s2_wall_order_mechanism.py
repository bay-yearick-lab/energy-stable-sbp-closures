#!/usr/bin/env python3
"""Generate `fig_s2_wall_order_mechanism.pdf`.

Three-panel mechanism drawing for Section 2.  The figure is a Strang-style
blackboard sketch, not a coefficient audit.  All algebra (Taylor series,
worked moment sums) is carried by the prose; this figure only shows the
mechanism.

Panels:

  (a) Interior:    symmetric node pairs around x_i, mirror-pair arcs.
                   Pair cancellation kills even moments structurally.
  (b) Wall:        the same shape clipped at x_0; mirror partners drawn as
                   faint ghost nodes outside the wall.  The pair-cancellation
                   route is unavailable.
  (c) Spline route polynomials enter the clamped open-uniform B-spline
                   space exactly, then exit through the BMF/BMG operators.
                   Polynomial reproduction replaces pair cancellation.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from project_paths import ensure_generated_dirs, save_figure

ensure_generated_dirs()

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8.6,
    'axes.labelsize': 8.6,
    'axes.titlesize': 9.6,
    'xtick.labelsize': 7.6,
    'ytick.labelsize': 7.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 180,
    'savefig.dpi': 300,
})


BLUE = '#1f77b4'
RED = '#d62728'
ACCENT = '#2F4858'
GUIDE = '#D9D2C5'
GHOST = '#B0AFAB'
PANEL_BG = (0.987, 0.985, 0.978)
NODE_SIZE = 17.0


def signed_node(ax: plt.Axes, x: float, y: float, sign: int,
                *, ghost: bool = False) -> None:
    """Draw a small signed node at (x, y) using a marker (display-aspect-safe).

    sign:  +1 for a positive weight, -1 for negative, 0 for the centered
           zero-weight node.
    ghost: render as a hollow gray outline (for missing wall partners).
    """
    if ghost:
        ax.plot([x], [y], marker='o', markerfacecolor='white',
                markeredgecolor=GHOST, markersize=NODE_SIZE,
                markeredgewidth=1.0, linestyle='None', zorder=3)
        return
    if sign == 0:
        ax.plot([x], [y], marker='o', markerfacecolor='white',
                markeredgecolor='black', markersize=NODE_SIZE * 0.55,
                markeredgewidth=0.9, linestyle='None', zorder=3)
        return
    color = BLUE if sign > 0 else RED
    ax.plot([x], [y], marker='o', markerfacecolor=color,
            markeredgecolor=color, markersize=NODE_SIZE,
            linestyle='None', zorder=3)
    glyph = '+' if sign > 0 else r'\mathbf{-}'
    ax.text(x, y, rf'${glyph}$', ha='center', va='center',
            color='white', fontsize=8.4, fontweight='bold', zorder=4)


def panel_frame(ax: plt.Axes, xlim, ylim, title: str) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(PANEL_BG)
    for side in ('left', 'right', 'top', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.set_title(title, pad=8, color=ACCENT, fontsize=10.0)


def fig_a_interior(ax: plt.Axes) -> None:
    """Interior centered row: mirror partners cancel."""
    panel_frame(ax, (-3.7, 3.7), (-1.6, 2.7),
                r'(a) interior: mirror partners cancel')

    ax.axhline(0.0, color='black', lw=0.7, zorder=1)
    nodes = [-3, -2, -1, 0, 1, 2, 3]
    signs = [-1, +1, -1, 0, +1, -1, +1]
    for x, s in zip(nodes, signs):
        signed_node(ax, x, 0.0, s)

    ax.text(0.0, -0.85, r'$x_i$', ha='center', va='top', fontsize=9.0,
            color=ACCENT, family='serif')

    arc_specs = [(1, 0.45), (2, 0.55), (3, 0.65)]
    for k, rad in arc_specs:
        arrow = FancyArrowPatch((-k, 0.32), (k, 0.32),
                                arrowstyle='<->', mutation_scale=8,
                                color=ACCENT, lw=0.7,
                                connectionstyle=f'arc3,rad={rad}')
        ax.add_patch(arrow)

    ax.text(0.0, 1.95, r'mirror pair: $\,w_{-k}=-w_{+k}$',
            ha='center', va='center', fontsize=8.4,
            color=ACCENT, family='serif')
    ax.text(0.0, 1.30,
            r'even moments cancel structurally',
            ha='center', va='center', fontsize=7.6,
            color='0.30', family='serif', style='italic')


def fig_b_wall(ax: plt.Axes) -> None:
    """Wall row: mirror partners are missing."""
    panel_frame(ax, (-3.7, 3.7), (-1.6, 2.7),
                r'(b) wall: partners are missing')

    ax.axhline(0.0, color='black', lw=0.7, zorder=1)

    wall_x = -0.55
    ax.plot([wall_x, wall_x], [-0.55, 0.85], color=ACCENT, lw=1.1, zorder=2)
    ax.text(wall_x - 0.12, 0.65, 'wall', ha='right', va='center',
            fontsize=7.8, color=ACCENT, family='serif', style='italic')

    ghost_nodes = [-3, -2, -1]
    for x in ghost_nodes:
        signed_node(ax, x, 0.0, 0, ghost=True)

    interior_nodes = [0, 1, 2, 3]
    interior_signs = [-1, +1, -1, +1]
    for x, s in zip(interior_nodes, interior_signs):
        signed_node(ax, x, 0.0, s)

    ax.text(0.0, -0.85, r'$x_0$', ha='center', va='top', fontsize=9.0,
            color=ACCENT, family='serif')

    for k, rad in [(1, 0.55), (2, 0.60), (3, 0.65)]:
        arrow = FancyArrowPatch((-k, 0.32), (k, 0.32),
                                arrowstyle='<->', mutation_scale=8,
                                color=GHOST, lw=0.6, linestyle=(0, (3, 2)),
                                connectionstyle=f'arc3,rad={rad}')
        ax.add_patch(arrow)
    ax.text(-2.6, 0.55, 'no partners',
            ha='center', va='center', fontsize=7.4, color=GHOST,
            family='serif', style='italic')

    ax.text(0.0, 1.95,
            r'pair cancellation unavailable',
            ha='center', va='center', fontsize=8.4,
            color=ACCENT, family='serif')
    ax.text(0.0, 1.30,
            r'closure must be tuned (or replaced)',
            ha='center', va='center', fontsize=7.6,
            color='0.30', family='serif', style='italic')


def fig_c_spline(ax: plt.Axes) -> None:
    """Spline route: polynomial reproduction replaces pair cancellation."""
    panel_frame(ax, (0.0, 11.0), (-1.6, 2.7),
                r'(c) spline route: polynomials pass through exactly')

    poly_x = 1.6
    space_x = 5.5
    out_x = 9.4
    box_y = 0.20

    polys = [r'$1$', r'$x$', r'$x^2$', r'$\cdots$', r'$x^{\,p}$']
    poly_y = [1.05, 0.65, 0.25, -0.15, -0.55]
    for label, py in zip(polys, poly_y):
        ax.text(poly_x, py, label, ha='center', va='center',
                fontsize=8.6, family='serif', color='black')
    ax.text(poly_x, 1.55, 'polynomial data',
            ha='center', va='center', fontsize=7.6, color=ACCENT,
            family='serif', style='italic')

    box = FancyBboxPatch((space_x - 1.40, box_y - 1.00), 2.80, 2.00,
                         boxstyle='round,pad=0.04,rounding_size=0.18',
                         linewidth=1.0, edgecolor=ACCENT,
                         facecolor='white', zorder=2)
    ax.add_patch(box)
    ax.text(space_x, box_y + 0.55,
            r'spline space',
            ha='center', va='center', fontsize=8.4, color=ACCENT,
            family='serif')
    ax.text(space_x, box_y + 0.05,
            r'$S_h^{\,p}$',
            ha='center', va='center', fontsize=12.0, color='black',
            family='serif')
    ax.text(space_x, box_y - 0.55,
            r'clamped, open-uniform',
            ha='center', va='center', fontsize=7.0, color='0.40',
            family='serif', style='italic')

    arrow_in = FancyArrowPatch((poly_x + 0.65, box_y + 0.20),
                               (space_x - 1.42, box_y + 0.20),
                               arrowstyle='-|>', mutation_scale=10,
                               color=ACCENT, lw=0.9)
    ax.add_patch(arrow_in)
    ax.text((poly_x + 0.65 + space_x - 1.42) / 2.0, box_y + 0.55,
            r'reconstruct',
            ha='center', va='center', fontsize=7.4, color=ACCENT,
            family='serif', style='italic')

    arrow_out_end = out_x - 1.10
    arrow_out = FancyArrowPatch((space_x + 1.42, box_y + 0.20),
                                (arrow_out_end, box_y + 0.20),
                                arrowstyle='-|>', mutation_scale=10,
                                color=ACCENT, lw=0.9)
    ax.add_patch(arrow_out)
    ax.text((space_x + 1.42 + arrow_out_end) / 2.0, box_y + 0.55,
            r'$D_h$ / eval',
            ha='center', va='center', fontsize=7.4, color=ACCENT,
            family='serif', style='italic')

    right_anchor = 10.85
    ax.text(right_anchor, box_y + 0.55,
            r'exact on $\mathbb{P}_p$',
            ha='right', va='center', fontsize=7.8, color=ACCENT,
            family='serif', style='italic')
    ax.text(right_anchor, box_y - 0.10,
            r'$\delta_{i,\ell}=0,\ \ell\leq p$',
            ha='right', va='center', fontsize=8.8, color=ACCENT,
            family='serif')


def main() -> None:
    fig, axs = plt.subplots(
        1, 3, figsize=(10.0, 2.9), layout='constrained',
        gridspec_kw={'width_ratios': [1.0, 1.0, 1.55], 'wspace': 0.10},
    )
    fig_a_interior(axs[0])
    fig_b_wall(axs[1])
    fig_c_spline(axs[2])

    save_figure(fig, 'fig_s2_wall_order_mechanism', pad_inches=0.05)
    plt.close(fig)
    print('wrote wall-order mechanism figure')


if __name__ == '__main__':
    main()
