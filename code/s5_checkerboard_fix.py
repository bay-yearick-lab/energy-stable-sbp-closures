#!/usr/bin/env python3
"""Generate `fig_s5_checkerboard_fix.pdf`.

Strang-style two-panel comparison that visualizes the structural reason
for staggered placement of pressure and velocity in the merged Section
on B-spline mimetic SBP operators and conservative flow forms.

Panel (a):  collocated lattice with the alternating checkerboard
            pressure pattern.  A standard centered pressure-gradient
            stencil is drawn on a velocity location; the two pressure
            samples that enter that stencil are equal, so the centered
            gradient returns zero.

Panel (b):  staggered MAC placement.  Pressure on cell centers,
            normal velocity on faces.  The face gradient samples the
            jump across one face, returning a nonzero alternating face
            flux.

The figure is intentionally one drawing per panel.  All algebra,
proofs, and product-space details belong in the surrounding text.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    Rectangle,
    FancyBboxPatch,
)

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
P_POS = '#1f5fa5'   # cool blue for +1 pressure samples
P_NEG = '#b8483b'   # warm red for -1 pressure samples
BMF_COL = '#1f5fa5'
WALL = '#1a1a1a'
FACE_FACE = '#fbf3df'
FACE_EDGE = '#a8821b'
FAIL = '#9a3a4a'
PASS = '#2f7d3b'


def style_axes(ax, *, title=None, xlim=(-0.05, 4.05), ylim=(-1.95, 4.25)):
    ax.set_facecolor(PANEL_BG)
    if title:
        ax.set_title(title, color=ACCENT, fontsize=13.5, pad=6.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_checkerboard_pressures(ax, *, with_face=False):
    n = 4
    for i in range(n):
        for j in range(n):
            x_c = i + 0.5
            y_c = j + 0.5
            sign = (-1) ** (i + j)
            colour = P_POS if sign > 0 else P_NEG
            ax.add_patch(Circle((x_c, y_c), 0.16,
                                facecolor=colour, edgecolor='white',
                                linewidth=0.7, zorder=4))
            ax.text(x_c, y_c,
                    r'$+$' if sign > 0 else r'$-$',
                    ha='center', va='center',
                    fontsize=10.6, color='white',
                    fontweight='bold', zorder=5)
    for k in range(n + 1):
        ax.plot([k, k], [0, n], color='0.55', lw=0.45, zorder=1)
        ax.plot([0, n], [k, k], color='0.55', lw=0.45, zorder=1)


def main() -> None:
    fig = plt.figure(figsize=(13.0, 5.7), layout='constrained')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.08)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    style_axes(ax_a, title=r'(a) collocated: centered gradient sees zero on the checkerboard')
    draw_checkerboard_pressures(ax_a)

    cx, cy = 1.5, 2.5
    ax_a.add_patch(Circle((cx, cy), 0.16,
                          facecolor='white', edgecolor=ACCENT,
                          linewidth=1.6, zorder=6))
    ax_a.text(cx - 0.20, cy + 0.32, r'$u_{i,j}$', ha='right', va='bottom',
              fontsize=12.0, color=ACCENT, family='serif')
    arrow_left = FancyArrowPatch((cx - 0.42, cy), (cx + 0.42, cy),
                                 arrowstyle='<->', mutation_scale=13,
                                 color=FAIL, lw=1.6, zorder=7)
    ax_a.add_patch(arrow_left)

    ax_a.text(2.0, -0.30,
              r'collocated: pressure and velocity share the same lattice',
              ha='center', va='top', fontsize=11.5, color='0.30',
              family='serif', style='italic')

    ax_a.text(2.0, -0.95,
              r'centered gradient sees $p_{i+1,j}=p_{i-1,j}=+1$',
              ha='center', va='top', fontsize=12.0, color=FAIL,
              family='serif')
    ax_a.text(2.0, -1.46,
              r'$\Rightarrow\ \dfrac{p_{i+1,j}-p_{i-1,j}}{2h}=0$',
              ha='center', va='top', fontsize=13.0, color=FAIL,
              family='serif', fontweight='bold')

    style_axes(ax_b, title=r'(b) staggered MAC: face gradient resolves the checkerboard')
    draw_checkerboard_pressures(ax_b)

    n = 4
    face_xs = [(i, j + 0.5) for i in range(1, n) for j in range(n)]
    face_ys = [(i + 0.5, j) for i in range(n) for j in range(1, n)]

    for fx, fy in face_xs:
        ax_b.add_patch(Rectangle((fx - 0.06, fy - 0.14),
                                 0.12, 0.28,
                                 facecolor=FACE_FACE, edgecolor=FACE_EDGE,
                                 linewidth=0.6, zorder=2))
    for fx, fy in face_ys:
        ax_b.add_patch(Rectangle((fx - 0.14, fy - 0.06),
                                 0.28, 0.12,
                                 facecolor=FACE_FACE, edgecolor=FACE_EDGE,
                                 linewidth=0.6, zorder=2))

    fx, fy = 2.0, 2.5
    arrow = FancyArrowPatch((fx - 0.20, fy), (fx + 0.20, fy),
                            arrowstyle='-|>', mutation_scale=15,
                            color=PASS, lw=1.9, zorder=7)
    ax_b.add_patch(arrow)
    ax_b.annotate(
        r'$u_1$ on face',
        xy=(fx, fy + 0.05), xytext=(fx, 4.12),
        ha='center', va='bottom', fontsize=11.8, color=PASS,
        family='serif',
        arrowprops=dict(arrowstyle='-', color=PASS, lw=0.9, alpha=0.85,
                        shrinkA=2, shrinkB=4),
    )

    ax_b.text(2.0, -0.30,
              r'staggered: pressure on cells, velocity on faces',
              ha='center', va='top', fontsize=11.5, color='0.30',
              family='serif', style='italic')

    ax_b.text(2.0, -0.95,
              r'face gradient sees $p_{i+1,j}-p_{i,j}=\pm 2$',
              ha='center', va='top', fontsize=12.0, color=PASS,
              family='serif')
    ax_b.text(2.0, -1.46,
              r'$\Rightarrow\ \dfrac{p_{i+1,j}-p_{i,j}}{h}=\pm\dfrac{2}{h}$',
              ha='center', va='top', fontsize=13.0, color=PASS,
              family='serif', fontweight='bold')

    save_figure(fig, 'fig_s5_checkerboard_fix', pad_inches=0.05)
    plt.close(fig)
    print('wrote section 6 checkerboard-fix figure')


if __name__ == '__main__':
    main()
