#!/usr/bin/env python3
"""Setup schematic for the mimetic SBP closure subsection (S6 heat cavity).

The primary experiment of the mimetic-SBP subsection is the closed-cavity scalar
heat decay on the unit square: solve the heat equation with homogeneous Dirichlet
walls on all four sides from a smooth initial hump, integrated through a long
parabolic decay, with the scheme-compatible second-derivative closure D^(2)
supplying the discrete Laplacian. This script renders only the SETUP, namely the
square domain, the u=0 Dirichlet condition labelled on each of the four walls, and
the smooth initial state drawn as contours. No result curves are produced here.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

WALL = "#2F4858"          # classyaccent
HUMP = plt.cm.RdBu_r       # smooth field colormap


def smooth_hump(x, y):
    """A single smooth interior hump that vanishes on the walls.

    Built from the lowest Dirichlet sine-sine modes so it is exactly zero on
    dOmega and smooth inside; this is the schematic stand-in for the smooth
    initial state, not a result field.
    """
    base = np.sin(np.pi * x) * np.sin(np.pi * y)
    bump = np.exp(-((x - 0.62) ** 2 + (y - 0.58) ** 2) / (2 * 0.16 ** 2))
    return base * (0.45 + 0.55 * bump)


def draw_dirichlet_hatching(ax) -> None:
    """Uniform outward hatch ticks on all four walls of the unit square."""
    tick_kw = dict(color=WALL, lw=1.05, solid_capstyle="round")
    hatch_len = 0.052
    samples = np.linspace(0.06, 0.94, 13)
    for s in samples:
        ax.plot([s, s - hatch_len], [0.0, -hatch_len], **tick_kw)          # bottom
        ax.plot([s, s + hatch_len], [1.0, 1.0 + hatch_len], **tick_kw)     # top
        ax.plot([0.0, -hatch_len], [s, s - hatch_len], **tick_kw)          # left
        ax.plot([1.0, 1.0 + hatch_len], [s, s + hatch_len], **tick_kw)     # right


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    # --- smooth initial state as contours -------------------------------------
    n = 320
    xx = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xx, xx)
    Z = smooth_hump(X, Y)
    Z = Z / Z.max()

    levels = np.linspace(0.06, 1.0, 9)
    ax.contourf(X, Y, Z, levels=np.linspace(0.0, 1.0, 60), cmap=HUMP,
                alpha=0.92, zorder=1)
    cs = ax.contour(X, Y, Z, levels=levels, colors="#3a3a3a", linewidths=0.55,
                    alpha=0.7, zorder=2)

    # --- domain boundary ------------------------------------------------------
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=WALL,
                               lw=2.4, zorder=4))
    draw_dirichlet_hatching(ax)

    # --- domain label ---------------------------------------------------------
    ax.text(0.13, 0.16, r"$\Omega=(0,1)^2$", fontsize=12.5, color="#1c1c1c",
            zorder=6, ha="left", va="center")
    ax.text(0.62, 0.58, "initial hump", fontsize=11.0, color="#1c1c1c",
            zorder=6, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                      alpha=0.78))

    # --- Dirichlet wall labels, one per side ---------------------------------
    lab_kw = dict(fontsize=11.5, color=WALL, zorder=6)
    ax.text(0.5, -0.135, r"$u=0$", ha="center", va="center", **lab_kw)       # bottom
    ax.text(0.5, 1.135, r"$u=0$", ha="center", va="center", **lab_kw)        # top
    ax.text(-0.135, 0.5, r"$u=0$", ha="center", va="center",
            rotation=90, **lab_kw)                                            # left
    ax.text(1.135, 0.5, r"$u=0$", ha="center", va="center",
            rotation=-90, **lab_kw)                                           # right

    # --- setup title ----------------------------------------------------------
    ax.text(0.5, 1.255, "Closed-cavity heat diffusion setup",
            ha="center", va="center", fontsize=12.8, color="#1c1c1c")

    ax.set_xlim(-0.24, 1.24)
    ax.set_ylim(-0.24, 1.34)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    save_figure(fig, "fig_s6_sbp_setup")
    plt.close(fig)
    print("wrote fig_s6_sbp_setup.(pdf,png)")


if __name__ == "__main__":
    main()
