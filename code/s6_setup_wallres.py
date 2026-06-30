#!/usr/bin/env python3
"""Setup schematic for the spectral-like wall resolution experiment.

This is a SETUP diagram, not a results figure. It shows the operator test that
governs the wall resolution spectrum: the unit interval [0,1] discretized on N
nodes with a rigid wall at x=0, the admissible standing sine input mode s_m,
the first-derivative operator D applied to it, and the target cosine partner
channel c_m onto which the response is projected in the trapezoidal-weight
inner product. No fabricated result curves are drawn; the sine and cosine here
are the analytic input mode and target channel that define the measurement.
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

C_INPUT = "#4c72b0"   # standing sine input mode
C_TARGET = "#d62728"  # target cosine channel
C_WALL = "#3a3a3a"
C_NODE = "#7a7a7a"


def main() -> None:
    m = 3  # an illustrative admissible wall mode for the schematic
    x = np.linspace(0.0, 1.0, 600)
    s = np.sin(m * np.pi * x)
    c = np.cos(m * np.pi * x)

    fig, (axT, axB) = plt.subplots(
        2, 1, figsize=(8.6, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.55},
    )

    # ---- a coarse node set drawn once, shared visual reference ----
    Nshow = 13
    xn = np.linspace(0.0, 1.0, Nshow)

    # ===== top: input standing mode s_m and the operator D =====
    axT.axhline(0.0, color="0.75", lw=0.8, zorder=1)
    axT.plot(x, s, color=C_INPUT, lw=2.2, zorder=3,
             label="admissible standing input mode")
    axT.plot(xn, np.sin(m * np.pi * xn), ls="none", marker="o", ms=4.2,
             mfc="white", mec=C_INPUT, mew=1.3, zorder=4)

    # rigid wall at x=0
    axT.axvline(0.0, color=C_WALL, lw=2.6, zorder=5)
    axT.annotate(r"rigid wall $x=0$", xy=(0.0, -0.62), xytext=(0.085, -0.62),
                 va="center", ha="left", color=C_WALL, fontsize=9.8,
                 arrowprops=dict(arrowstyle="-|>", color=C_WALL, lw=1.3,
                                 shrinkA=0, shrinkB=2))

    # operator annotation: arrow bridging the two panels, label clear of curves
    axT.annotate("", xy=(0.5, -2.55), xytext=(0.5, -1.18),
                 xycoords=("data", "data"), textcoords=("data", "data"),
                 annotation_clip=False,
                 arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.6))
    axT.text(0.5, -1.92, "apply first-derivative operator",
             ha="center", va="center", fontsize=10.2, color="0.25",
             clip_on=False,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"))

    axT.set_ylim(-1.35, 1.35)
    axT.set_ylabel("amplitude")
    axT.legend(loc="upper right", frameon=False, handlelength=1.6,
               borderaxespad=0.2)
    axT.set_title("(a) admissible standing input on $[0,1]$, $N=129$ nodes",
                  loc="left", fontsize=10.6, pad=8)

    # ===== bottom: target cosine channel c_m, projected response =====
    axB.axhline(0.0, color="0.75", lw=0.8, zorder=1)
    axB.fill_between(x, 0.0, c, color=C_TARGET, alpha=0.10, zorder=2)
    axB.plot(x, c, color=C_TARGET, lw=2.2, zorder=3,
             label="target channel")
    axB.plot(xn, np.cos(m * np.pi * xn), ls="none", marker="s", ms=4.0,
             mfc="white", mec=C_TARGET, mew=1.3, zorder=4)

    axB.axvline(0.0, color=C_WALL, lw=2.6, zorder=5)

    axB.set_ylim(-1.35, 1.35)
    axB.set_xlim(-0.015, 1.015)
    axB.set_xlabel(r"$x$")
    axB.set_ylabel("amplitude")
    axB.legend(loc="upper right", frameon=False, handlelength=1.6,
               borderaxespad=0.2)
    axB.set_title(
        "(b) project the response onto the target channel in the weighted inner product",
        loc="left", fontsize=10.6, pad=8)

    # measured functionals box, placed below the axes clear of curves and wall
    axB.text(
        0.5, -2.05,
        "measured: response and orthogonal contamination",
        transform=axB.transData, ha="center", va="center", fontsize=10.2,
        clip_on=False,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f4f4f4", ec="0.6", lw=0.9),
    )

    fig.subplots_adjust(left=0.10, right=0.985, top=0.93, bottom=0.20, hspace=0.62)
    save_figure(fig, "fig_s6_wallres_setup")
    plt.close(fig)
    print("wrote fig_s6_wallres_setup")


if __name__ == "__main__":
    main()
