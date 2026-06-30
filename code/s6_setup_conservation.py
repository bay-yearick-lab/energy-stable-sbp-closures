#!/usr/bin/env python3
"""Setup schematic for the fully conservative flow-forms subsection.

This is a SETUP diagram, not a results plot. It brandishes the governing
problem of the primary experiment in that subsection: the unforced steady
analytic incompressible wake on the unit square, advanced to the closed-form
exact solution imposed as boundary traces. The figure is a single panel showing
the domain [0,1]^2 with the exact-solution boundary traces labelled by words on
all four sides (inflow, downstream wake, and the two periodic-in-y lateral
traces), a few exact streamlines, and the periodic-in-y wake pattern.

The streamlines and the wake shading use the analytic analytic-wake field, which is
the setup field itself, so nothing here is a fabricated result curve. No
equations or defining formulas appear in the figure; those live in the paper
body.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
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

ACCENT = "#2F4858"
INFLOW = "#4c72b0"
OUTFLOW = "#d62728"
LATERAL = "#2ca02c"
WAKE = "#9467bd"


def analytic_wake(Re: float):
    lam = 0.5 * (Re - np.sqrt(Re ** 2 + 16.0 * np.pi ** 2))

    def u(x, y):
        return 1.0 - np.exp(lam * x) * np.cos(2.0 * np.pi * y)

    def v(x, y):
        return (lam / (2.0 * np.pi)) * np.exp(lam * x) * np.sin(2.0 * np.pi * y)

    def psi(x, y):
        return y - np.exp(lam * x) / (2.0 * np.pi) * np.sin(2.0 * np.pi * y)

    return u, v, psi, lam


def main() -> None:
    Re = 40.0
    u, v, psi, lam = analytic_wake(Re)

    fig = plt.figure(figsize=(7.6, 6.0))
    axD = fig.add_subplot(1, 1, 1)

    # ---- Domain, boundary traces, streamlines, wake pattern ----
    n = 320
    xg = np.linspace(0.0, 1.0, n)
    yg = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xg, yg)
    U = u(X, Y)
    V = v(X, Y)
    speed = np.sqrt(U ** 2 + V ** 2)

    # faint speed field to indicate the periodic-in-y wake pattern
    axD.imshow(speed, extent=(0, 1, 0, 1), origin="lower", cmap="Purples",
               alpha=0.32, aspect="auto", zorder=0, vmin=speed.min(), vmax=speed.max())

    # exact streamlines (the setup field, not a result)
    seed = np.array([[0.02, yy] for yy in np.linspace(0.06, 0.94, 9)])
    axD.streamplot(xg, yg, U, V, color=ACCENT, density=2.2, linewidth=0.7,
                   arrowsize=0.7, broken_streamlines=False, start_points=seed,
                   zorder=2)

    # domain box
    axD.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="0.15", lw=1.6, zorder=4)

    # boundary trace markers: color the four edges
    lw_edge = 4.6
    axD.plot([0, 0], [0, 1], color=INFLOW, lw=lw_edge, solid_capstyle="butt", zorder=5)
    axD.plot([1, 1], [0, 1], color=OUTFLOW, lw=lw_edge, solid_capstyle="butt", zorder=5)
    axD.plot([0, 1], [0, 0], color=LATERAL, lw=lw_edge, solid_capstyle="butt", zorder=5)
    axD.plot([0, 1], [1, 1], color=LATERAL, lw=lw_edge, solid_capstyle="butt", zorder=5)

    # edge labels in plain words, placed clear of the box
    axD.text(-0.055, 0.5, "inflow trace", color=INFLOW, rotation=90,
             ha="center", va="center", fontsize=11.0, zorder=6)
    axD.text(1.06, 0.5, "downstream wake trace", color=OUTFLOW, rotation=90,
             ha="center", va="center", fontsize=11.0, zorder=6)
    axD.text(0.5, -0.072, "lateral trace", color=LATERAL,
             ha="center", va="center", fontsize=11.0, zorder=6)
    axD.text(0.5, 1.068, "lateral trace", color=LATERAL,
             ha="center", va="center", fontsize=11.0, zorder=6)

    # periodic-in-y annotation (right side, double arrow)
    arr = FancyArrowPatch((1.20, 0.04), (1.20, 0.96), arrowstyle="<->",
                          mutation_scale=12, color=WAKE, lw=1.3, zorder=6)
    axD.add_patch(arr)
    axD.text(1.245, 0.5, "$y$-periodic wake", color=WAKE, rotation=90,
             ha="center", va="center", fontsize=10.6, zorder=6)

    # interior label (words only, no formulas)
    axD.text(0.5, 0.5,
             "unit-square domain\nexact boundary traces imposed\n"
             "unforced wake, interior started at rest",
             ha="center", va="center", fontsize=10.4, color="0.18", zorder=6,
             bbox=dict(boxstyle="round,pad=0.36", fc="white", ec="0.7", alpha=0.85))

    axD.set_xlim(-0.18, 1.34)
    axD.set_ylim(-0.16, 1.16)
    axD.set_aspect("equal")
    axD.set_xticks([0, 0.5, 1.0])
    axD.set_yticks([0, 0.5, 1.0])
    axD.set_xlabel("$x$")
    axD.set_ylabel("$y$")
    axD.set_title("analytic wake domain and exact boundary traces")
    for s in ("top", "right", "left", "bottom"):
        axD.spines[s].set_visible(False)

    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.085, top=0.93)
    save_figure(fig, "fig_s6_conservation_setup")
    plt.close(fig)
    print("wrote fig_s6_conservation_setup (lambda at Re=40 = %.4f)" % lam)


if __name__ == "__main__":
    main()
