#!/usr/bin/env python3
"""Setup schematic for the exact bounded approximation space subsection.

Brandishes the governing problem of the immersed-circle transfer test: the
prescribed smooth field u(x,y)=exp(-18 x) sin(10 pi x) cos(2 pi y) on the unit
square [0,1]^2 is sampled on the operator's tensor-product node set, and three
off-grid observables are read on a curved interior circle: the value u, the wall
normal derivative grad u . n, and the traction line integral around the circle.

This renders the SETUP only (domain, node set, prescribed field, marker circle,
outward normals, and the measured channels). It fabricates no result curves.
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

# prescribed smooth field
def field(x, y):
    return np.exp(-18.0 * x) * np.sin(10.0 * np.pi * x) * np.cos(2.0 * np.pi * y)

# marker circle geometry (matches the transfer test)
XC, YC, RADIUS = 0.24, 0.50, 0.11
M_MARKERS = 240


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    fig.subplots_adjust(left=0.10, right=0.875, bottom=0.09, top=0.80)

    # field contours on the unit square
    gx = np.linspace(0.0, 1.0, 400)
    gy = np.linspace(0.0, 1.0, 400)
    XX, YY = np.meshgrid(gx, gy)
    FF = field(XX, YY)
    levels = np.linspace(-np.max(np.abs(FF)), np.max(np.abs(FF)), 19)
    cf = ax.contourf(XX, YY, FF, levels=levels, cmap="RdBu_r", alpha=0.92)
    ax.contour(XX, YY, FF, levels=levels, colors="0.35", linewidths=0.35, alpha=0.6)

    # tensor-product node set (sparse, schematic)
    Ng = 11
    nx_pts = np.linspace(0.0, 1.0, Ng)
    NXX, NYY = np.meshgrid(nx_pts, nx_pts)
    ax.scatter(NXX, NYY, s=5.0, color="0.20", alpha=0.55, zorder=3,
               label="node set (tensor-product grid)")

    # interior marker circle
    phi = np.linspace(0.0, 2.0 * np.pi, M_MARKERS, endpoint=False)
    xm = XC + RADIUS * np.cos(phi)
    ym = YC + RADIUS * np.sin(phi)
    ax.plot(np.r_[xm, xm[0]], np.r_[ym, ym[0]], color="black", lw=1.9, zorder=4,
            label="curved interior surface")
    ax.scatter(xm[::15], ym[::15], s=16, color="black", zorder=5,
               label="off-grid markers")

    # a few outward normals
    sample = slice(0, M_MARKERS, 30)
    ax.quiver(xm[sample], ym[sample],
              0.07 * np.cos(phi[sample]), 0.07 * np.sin(phi[sample]),
              angles="xy", scale_units="xy", scale=1.0,
              width=0.006, color="0.10", zorder=6)

    # highlight one marker with a labelled normal arrow
    mk = 4
    xa, ya = xm[mk], ym[mk]
    nxa, nya = np.cos(phi[mk]), np.sin(phi[mk])
    arr = FancyArrowPatch((xa, ya), (xa + 0.135 * nxa, ya + 0.135 * nya),
                          arrowstyle="-|>", mutation_scale=15, lw=2.0,
                          color="#1a1a1a", zorder=7)
    ax.add_patch(arr)
    ax.annotate("normal", (xa + 0.135 * nxa, ya + 0.135 * nya),
                xytext=(7, 4), textcoords="offset points", fontsize=10.5)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Prescribed field sampled on $[0,1]^2$;"
                 + "\n" + "off-grid transfer on the circle",
                 fontweight="semibold", pad=34)

    # colorbar
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("field value")
    cb.outline.set_linewidth(0.6)

    # measured channels as a clean off-axis annotation box (plain words, no formulas)
    obs = ("\n".join([
        "measured at the markers:",
        "value",
        "wall-normal derivative",
        "traction",
    ]))
    ax.text(0.985, 0.015, obs, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.4,
            bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="0.6", lw=0.7,
                      alpha=0.92), zorder=8)

    # legend overhead, outside the axes
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.30), ncol=3,
              frameon=False, handletextpad=0.5, columnspacing=1.5,
              borderaxespad=0.0)

    save_figure(fig, "fig_s6_bounded_setup")
    plt.close(fig)
    print("wrote fig_s6_bounded_setup")


if __name__ == "__main__":
    main()
