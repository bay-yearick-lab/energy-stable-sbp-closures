#!/usr/bin/env python3
"""Figure 3: top-down construction sequence for the two spline routes."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from project_paths import ensure_generated_dirs, save_figure


ensure_generated_dirs()

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.5,
        "figure.dpi": 180,
        "savefig.dpi": 300,
    }
)

PANEL_BG = "#f5f1e6"
INK = "#26333d"
QUIET = "#71808b"
BMF = "#22517a"
BMG = "#a83232"


def _box(
    ax: plt.Axes,
    center: tuple[float, float],
    size: tuple[float, float],
    text: str,
    *,
    edge: str = INK,
    face: str = PANEL_BG,
    linewidth: float = 1.05,
    fontsize: float = 10.2,
) -> FancyBboxPatch:
    x, y = center
    width, height = size
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        mutation_aspect=1.0,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color=INK,
        fontsize=fontsize,
        linespacing=1.22,
        zorder=3,
    )
    return patch


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = QUIET,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.95,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.7, 6.15))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    common_width = 0.48
    common_height = 0.105
    centers = [0.90, 0.715, 0.53, 0.345]
    labels = [
        "clamped open-uniform knots",
        "degree-$p$ B-spline basis",
        "true Greville nodes",
        "sample and coefficient forms\n" r"$B,\ B',\ M,\ K$",
    ]
    for y, label in zip(centers, labels):
        _box(ax, (0.5, y), (common_width, common_height), label)
    for upper, lower in zip(centers[:-1], centers[1:]):
        _arrow(
            ax,
            (0.5, upper - common_height / 2 - 0.006),
            (0.5, lower + common_height / 2 + 0.006),
        )

    ax.text(
        0.5,
        0.255,
        "one bounded coefficient algebra, two routes",
        ha="center",
        va="center",
        color=INK,
        fontsize=10.2,
        fontstyle="italic",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.8},
        zorder=3,
    )

    route_y = 0.115
    route_size = (0.385, 0.145)
    _box(
        ax,
        (0.275, route_y),
        route_size,
        "BMF flux route\nstaggered gradient-divergence pair\nprimal and derivative Greville grids",
        edge=BMF,
        face="#edf3f7",
        linewidth=1.35,
        fontsize=9.5,
    )
    _box(
        ax,
        (0.725, route_y),
        route_size,
        "BMG Galerkin route\nsame-grid derivative $B M^{-1} K B^{-1}$\nnatural spline mass metric",
        edge=BMG,
        face="#f8eeee",
        linewidth=1.35,
        fontsize=9.5,
    )

    fork = (0.5, 0.215)
    _arrow(
        ax,
        (0.5, centers[-1] - common_height / 2 - 0.006),
        fork,
    )
    _arrow(
        ax,
        fork,
        (0.275, route_y + route_size[1] / 2 + 0.008),
        color=BMF,
        connectionstyle="arc3,rad=0.0",
    )
    _arrow(
        ax,
        fork,
        (0.725, route_y + route_size[1] / 2 + 0.008),
        color=BMG,
        connectionstyle="arc3,rad=0.0",
    )

    fig.subplots_adjust(left=0.035, right=0.965, top=0.98, bottom=0.025)
    save_figure(fig, "fig_s3_construction_sequence", pad_inches=0.025)
    plt.close(fig)
    print("wrote section 3 construction-sequence figure")


if __name__ == "__main__":
    main()
