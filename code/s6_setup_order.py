#!/usr/bin/env python3
"""Setup schematic for the wall-preserved formal-order subsection.

This is a SCHEMATIC of the experimental setup, not a results plot. It shows two
things side by side.

(a) The near-wall grid with the one-sided wall-row stencil footprint of the
    first-derivative operator highlighted, plus plain-English labels for the
    wall-row stencil footprint and the closure moments the wall row must
    reproduce.

(b) The manufactured-solution heat-cavity grid sweep that delivers the dynamical
    order: the closed unit square with Dirichlet walls, a swatch of the exact
    field, and the refining grid sequence along which the measured order is read
    against the spline degree.

This is a schematic only: it carries no equations or defining formulas, only the
geometry, the grid, and plain-English labels.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

from project_paths import ensure_generated_dirs, save_figure

ensure_generated_dirs()
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.2, "axes.labelsize": 11.5, "axes.titlesize": 12.0,
    "legend.fontsize": 9.8, "xtick.labelsize": 9.4, "ytick.labelsize": 9.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 180, "savefig.dpi": 300,
})

WALL = "#2F4858"        # wall / boundary
NODE = "#9aa0a6"        # interior grid node
WALLNODE = "#d62728"    # wall-row node
STENCIL = "#4c72b0"     # one-sided stencil footprint
BAND = "#2ca02c"        # manufactured Gaussian band
INK = "#1a1a1a"


def panel_a(ax: plt.Axes) -> None:
    """Near-wall grid and one-sided wall-row stencil footprint."""
    ax.set_title(r"(a) Wall-row stencil footprint and closure moments")
    nshow = 8
    xs = np.arange(nshow)        # node index from the wall
    y_grid = 0.0

    # wall as a hatched bar at the left
    ax.add_patch(Rectangle((-1.05, -0.55), 0.55, 1.95, facecolor=WALL,
                           edgecolor="none", alpha=0.9, zorder=1))
    ax.text(-0.78, 1.18, "wall", rotation=90, ha="center", va="center",
            color="white", fontsize=10.5)

    # the one-sided wall-row stencil footprint band (over the first few nodes)
    foot = 5
    ax.add_patch(Rectangle((-0.42, -0.34), (foot - 1) + 0.84, 0.68,
                           facecolor=STENCIL, edgecolor="none", alpha=0.16, zorder=1))
    ax.annotate(r"one-sided wall-row footprint of $D$",
                xy=((foot - 1) / 2.0, 0.40), xytext=((foot - 1) / 2.0, 0.95),
                ha="center", va="bottom", color=STENCIL, fontsize=10.0,
                arrowprops=dict(arrowstyle="-[, widthB=4.4, lengthB=0.5",
                                color=STENCIL, lw=1.3))

    # grid nodes
    for k in xs:
        col = WALLNODE if k == 0 else NODE
        ms = 9.5 if k == 0 else 7.0
        ax.plot(k, y_grid, "o", color=col, ms=ms, zorder=4,
                markeredgecolor="white", markeredgewidth=0.8)
    ax.text(0.0, 0.26, r"$x_0$", ha="center", va="bottom",
            color=WALLNODE, fontsize=10.0)
    ax.text(nshow - 1.0, -0.22, r"interior", ha="center", va="top",
            color=NODE, fontsize=9.8)

    # tail dots toward interior
    for k in range(nshow, nshow + 2):
        ax.plot(k - 0.1, y_grid, ".", color=NODE, ms=3.0, zorder=3)

    # spacing brace between first two nodes
    ax.annotate("", xy=(0.0, -0.05), xytext=(1.0, -0.05),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
    ax.text(0.5, -0.135, "grid spacing", ha="center", va="top",
            color=INK, fontsize=9.6)

    # plain-English labels for the wall-row footprint and closure moments
    ax.text((foot - 1) / 2.0, -0.98,
            "wall-row stencil footprint",
            ha="center", va="center", color=INK, fontsize=10.6)
    ax.text((foot - 1) / 2.0, -1.30,
            "closure moments the wall row must reproduce",
            ha="center", va="center", color=STENCIL, fontsize=10.0)

    ax.set_xlim(-1.25, nshow + 0.3)
    ax.set_ylim(-1.55, 1.55)
    ax.axis("off")


def panel_b(ax: plt.Axes) -> None:
    """Manufactured-solution heat-cavity grid sweep on the closed unit square."""
    ax.set_title(r"(b) Manufactured-solution grid sweep of the heat operator")

    # closed unit square domain with Dirichlet walls
    sq = Rectangle((0.0, 0.0), 1.0, 1.0, facecolor="none",
                   edgecolor=WALL, lw=3.0, zorder=3)
    ax.add_patch(sq)
    ax.text(0.5, 1.065, "Dirichlet walls",
            ha="center", va="bottom", color=WALL, fontsize=9.8)
    ax.text(0.02, 0.5, "unit square", rotation=90, ha="right", va="center",
            color=INK, fontsize=9.8, transform=ax.transData)

    # the known exact field: a narrow Gaussian band of sine-sine standing modes
    gx = np.linspace(0, 1, 240)
    gy = np.linspace(0, 1, 240)
    GX, GY = np.meshgrid(gx, gy)
    field = np.zeros_like(GX)
    m0, n0 = 8, 8
    for m in (m0 - 1, m0, m0 + 1):
        for n in (n0 - 1, n0, n0 + 1):
            a = np.exp(-0.5 * ((m - m0) ** 2 + (n - n0) ** 2))
            field += a * np.sin(m * np.pi * GX) * np.sin(n * np.pi * GY)
    ax.contourf(GX, GY, field, levels=14, cmap="RdBu_r", alpha=0.55, zorder=1)

    # grid-sequence refinement marks along the bottom edge (coarse to fine ticks)
    seq = [9, 13, 17, 25]
    base_y = -0.16
    dy = 0.052
    for i, n in enumerate(seq):
        yy = base_y - i * dy
        xs = np.linspace(0.07, 0.93, n)
        ax.plot(xs, np.full_like(xs, yy), "|", color=INK, ms=6.0,
                markeredgewidth=0.9, zorder=2)
        ax.text(0.99, yy, rf"$N$", ha="left", va="center", fontsize=8.4,
                color=INK, alpha=0.0)  # spacer; real label below
    ax.annotate("", xy=(0.04, base_y + 0.018),
                xytext=(0.04, base_y - (len(seq) - 1) * dy - 0.018),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.text(0.0, base_y - (len(seq) - 1) * dy / 2.0,
            r"refine", rotation=90, ha="right", va="center",
            color=INK, fontsize=9.4)
    ax.text(0.5, base_y - (len(seq) - 1) * dy - 0.075,
            "grid sequence, coarse to fine",
            ha="center", va="top", color=INK, fontsize=9.8)

    # plain-English orienting labels: setup name and what is measured
    ax.text(0.5, 1.305,
            "manufactured-solution grid sweep",
            ha="center", va="center", color=INK, fontsize=10.8)
    ax.text(0.5, 1.205,
            "exact field swatch",
            ha="center", va="center", color=BAND, fontsize=9.6)

    ax.text(0.5, -0.50,
            "measured order vs spline degree",
            ha="center", va="center", color=INK, fontsize=10.4)

    ax.set_xlim(-0.10, 1.12)
    ax.set_ylim(-0.58, 1.42)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def main() -> None:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.0),
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})
    panel_a(axA)
    panel_b(axB)
    fig.subplots_adjust(left=0.02, right=0.985, top=0.90, bottom=0.04, wspace=0.12)
    save_figure(fig, "fig_s6_order_setup")
    plt.close(fig)
    print("wrote fig_s6_order_setup.{pdf,png}")


if __name__ == "__main__":
    main()
