#!/usr/bin/env python3
"""Setup schematic for the turbulent wall heat-transfer experiment (S6 Nusselt).

A passive scalar is advected by a developed B-spline channel velocity field at
Re_tau=180 on Omega=[0,2pi] x [0,pi] x [0,2], periodic in x and y, no-slip at the
two walls, sustained by a constant streamwise body force f_x=1 that pins Re_tau
through the discrete momentum balance. The scalar runs between isothermal walls
(theta=0 at z=0, theta=1 at z=2) at Prandtl number 0.71, and the wall-normal
temperature slope read at the wall is the Nusselt number.

This script renders only the SETUP (domain, boundary conditions, the body force,
the clustered coarse wall-normal grid, and what is measured); it carries no
result curves. Numbers shown are problem parameters, not measured outputs.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
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

# ---- colours ----
COLD = "#3b6fb0"     # cold wall, theta = 0
HOT = "#c0392b"      # hot wall, theta = 1
WALL = "#3a3a3a"     # no-slip wall edges
PERIODIC = "#7a7a7a"
FORCE = "#1f7a3d"    # streamwise body-force arrow
GRIDC = "#9aa6b2"    # clustered wall-normal grid
INK = "#1c1c1c"

# isometric-style projection: 3D (x, y, z) -> 2D (X, Y)
AX, AY = 0.52, 0.30          # x-axis screen direction (streamwise, into-right-depth)
BX, BY = 0.86, -0.18         # y-axis screen direction (spanwise, to the right)


def proj(x, y, z):
    return AX * x + BX * y, AY * x + BY * y + z


def clustered_z(nz, Lz):
    """Coarse wall-clustered nodes (cosine / Chebyshev-like), few cells in the
    conductive sublayer, matching the n_z=33 coarse wall grid."""
    j = np.arange(nz)
    zeta = 0.5 * (1.0 - np.cos(np.pi * j / (nz - 1)))
    return Lz * zeta


def main() -> None:
    Lx, Ly, Lz = 2 * np.pi, np.pi, 2.0

    fig, ax = plt.subplots(figsize=(10.4, 5.5))
    ax.set_aspect("equal")
    ax.axis("off")

    # box corners in (x, y, z)
    c = {
        "000": (0, 0, 0), "x00": (Lx, 0, 0), "xy0": (Lx, Ly, 0), "0y0": (0, Ly, 0),
        "00z": (0, 0, Lz), "x0z": (Lx, 0, Lz), "xyz": (Lx, Ly, Lz), "0yz": (0, Ly, Lz),
    }
    P = {k: proj(*v) for k, v in c.items()}

    # ---------- filled faces (painter's order: back to front) ----------
    cmap_cold = LinearSegmentedColormap.from_list("cold", ["#d7e3f2", COLD])
    cmap_hot = LinearSegmentedColormap.from_list("hot", ["#f6d9d4", HOT])

    # bottom wall z=0 : cold isothermal scalar wall theta = 0
    bot = Polygon([P["000"], P["x00"], P["xy0"], P["0y0"]], closed=True,
                  facecolor=COLD, edgecolor=WALL, lw=1.4, alpha=0.55, zorder=1)
    ax.add_patch(bot)
    # top wall z=Lz : hot isothermal scalar wall theta = 1
    top = Polygon([P["00z"], P["x0z"], P["xyz"], P["0yz"]], closed=True,
                  facecolor=HOT, edgecolor=WALL, lw=1.4, alpha=0.42, zorder=6)
    ax.add_patch(top)

    # streamwise (x) periodic face: back face at y=Ly (lighter), front at y=0
    backx = Polygon([P["0y0"], P["xy0"], P["xyz"], P["0yz"]], closed=True,
                    facecolor="#f4f5f7", edgecolor=PERIODIC, lw=1.0,
                    linestyle=(0, (5, 3)), alpha=0.9, zorder=2)
    ax.add_patch(backx)
    # spanwise (y) periodic face at x=Lx (right side)
    rightface = Polygon([P["x00"], P["xy0"], P["xyz"], P["x0z"]], closed=True,
                        facecolor="#eef0f3", edgecolor=PERIODIC, lw=1.0,
                        linestyle=(0, (5, 3)), alpha=0.9, zorder=3)
    ax.add_patch(rightface)
    # front (y=0) face left open to expose the interior; only outline its frame

    # ---------- clustered wall-normal grid on the front (y=0) face ----------
    nz = 17                       # schematic count; clustering matches coarse n_z
    zc = clustered_z(nz, Lz)
    for zi in zc:
        p0 = proj(0, 0, zi)
        p1 = proj(Lx, 0, zi)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GRIDC, lw=0.7, zorder=4)
    # a couple of vertical guides on the front face
    for xi in np.linspace(0, Lx, 5):
        p0 = proj(xi, 0, 0)
        p1 = proj(xi, 0, Lz)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=GRIDC, lw=0.5,
                alpha=0.6, zorder=4)

    # ---------- box edges (drawn over faces) ----------
    edges = [("000", "x00"), ("x00", "xy0"), ("xy0", "0y0"), ("0y0", "000"),
             ("00z", "x0z"), ("x0z", "xyz"), ("xyz", "0yz"), ("0yz", "00z"),
             ("000", "00z"), ("x00", "x0z"), ("xy0", "xyz"), ("0y0", "0yz")]
    for a, b in edges:
        ax.plot([P[a][0], P[b][0]], [P[a][1], P[b][1]],
                color=WALL, lw=1.3, zorder=7, solid_capstyle="round")

    # ---------- streamwise body-force arrow f_x (inside, low, along x) ----------
    fz = 0.34 * Lz
    fy = 0.45 * Ly
    fa = proj(0.28 * Lx, fy, fz)
    fb = proj(0.80 * Lx, fy, fz)
    ax.add_patch(FancyArrowPatch(fa, fb, arrowstyle="-|>", mutation_scale=20,
                                 color=FORCE, lw=2.6, zorder=8))
    # force label: leader down-left to a clean exterior spot below the box front
    flab = proj(0.50 * Lx, fy, fz)
    ax.annotate("streamwise body force",
                xy=(flab[0], flab[1]),
                xytext=(flab[0] + 0.9, flab[1] - 2.35),
                color=FORCE, fontsize=10.4, ha="center", va="top", zorder=9,
                arrowprops=dict(arrowstyle="-", color=FORCE, lw=0.7))

    # ---------- wall-flux measurement marker (Nusselt = wall slope) ----------
    # near-wall normal arrows on the right of the front face, away from the force
    xm = 0.86 * Lx
    for zw, sgn, col in [(0.0, +1, COLD), (Lz, -1, HOT)]:
        a0 = proj(xm, 0, zw)
        a1 = proj(xm, 0, zw + sgn * 0.40)
        ax.add_patch(FancyArrowPatch(a0, a1, arrowstyle="-|>", mutation_scale=12,
                                     color=col, lw=1.8, zorder=9))
    nuanchor = proj(xm, 0, 0.5 * Lz)
    ax.annotate("measure wall heat flux",
                xy=(nuanchor[0], nuanchor[1]),
                xytext=(nuanchor[0] + 2.0, nuanchor[1] + 0.05),
                fontsize=10.6, color=INK, ha="left", va="center", zorder=10,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))

    # ---------- axis triad (far lower-left, outside the box) ----------
    o = proj(0, 0, 0)
    ox, oy = o[0] - 2.4, o[1] - 1.7
    triad = [((AX, AY), r"$x$ streamwise", FORCE, ("left", "center")),
             ((BX, BY), r"$y$ spanwise", PERIODIC, ("left", "top")),
             ((0.0, 1.0), r"$z$ wall-normal", WALL, ("center", "bottom"))]
    L = 0.8
    for (dx, dy), lab, col, (ha_, va_) in triad:
        ax.add_patch(FancyArrowPatch((ox, oy), (ox + L * dx, oy + L * dy),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=col, lw=1.6, zorder=11))
        ax.text(ox + (L + 0.16) * dx, oy + (L + 0.16) * dy, lab, color=col,
                fontsize=8.8, ha=ha_, va=va_, zorder=11)
    ax.text(ox - 0.1, oy - 0.55, r"periodic in $x,y$; no-slip in $z$",
            fontsize=8.4, color="#566573", ha="left", va="top", zorder=11)

    # ---------- wall labels (placed outside the box, no collisions) ----------
    # hot wall label, top
    hl = proj(0.5 * Lx, 0.55 * Ly, Lz)
    ax.annotate(r"hot wall  $z=2:\ \theta=1$  (no-slip)",
                xy=(hl[0], hl[1]), xytext=(hl[0] + 0.6, hl[1] + 1.5),
                fontsize=9.8, color=HOT, ha="center", va="bottom", zorder=11,
                arrowprops=dict(arrowstyle="-", color=HOT, lw=0.7))
    # cold wall label, bottom-right
    cl = proj(Lx, 0.5 * Ly, 0.0)
    ax.annotate(r"cold wall  $z=0:\ \theta=0$  (no-slip)",
                xy=(cl[0], cl[1]), xytext=(cl[0] + 0.55, cl[1] - 1.05),
                fontsize=9.8, color=COLD, ha="left", va="center", zorder=11,
                arrowprops=dict(arrowstyle="-", color=COLD, lw=0.7))

    # domain extent label, top-left
    dl = proj(0, 0.5 * Ly, Lz)
    ax.text(dl[0] - 1.5, dl[1] + 1.5,
            r"$\Omega=[0,2\pi]\times[0,\pi]\times[0,2]$",
            fontsize=11.0, color=INK, ha="center", va="bottom", zorder=11)

    # clustered-grid callout to the left exterior, mid-height
    gc = proj(0.0, 0, clustered_z(nz, Lz)[4])
    ax.annotate("coarse wall-clustered\ngrid  $n_z=33$",
                xy=(gc[0], gc[1]), xytext=(gc[0] - 2.7, gc[1] + 0.9),
                fontsize=8.8, color="#566573", ha="center", va="center", zorder=11,
                arrowprops=dict(arrowstyle="-", color="#566573", lw=0.6))

    # ---------- parameter strip (top, outside, no overlap) ----------
    legend_handles = [
        Line2D([0], [0], color=HOT, lw=6, alpha=0.55, label=r"isothermal hot wall $\theta=1$"),
        Line2D([0], [0], color=COLD, lw=6, alpha=0.6, label=r"isothermal cold wall $\theta=0$"),
        Line2D([0], [0], color=PERIODIC, lw=1.6, ls=(0, (5, 3)), label="periodic faces ($x,y$)"),
        Line2D([0], [0], color=FORCE, lw=2.6, label=r"streamwise body force $f_x$"),
        Line2D([0], [0], color=GRIDC, lw=1.0, label="wall-normal grid"),
    ]
    ax.legend(handles=legend_handles, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, 1.10), frameon=False, fontsize=9.0,
              handlelength=1.6, columnspacing=1.4)

    # parameter tag strip at the bottom (orienting labels only, no equations)
    fig.text(0.5, 0.025,
             r"passive scalar transport      $Re_\tau=180$      $Pr=0.71$",
             ha="center", va="bottom", fontsize=10.2, color=INK)

    # generous limits so nothing clips
    xs = [p[0] for p in P.values()]
    ys = [p[1] for p in P.values()]
    ax.set_xlim(min(xs) - 3.6, max(xs) + 4.6)
    ax.set_ylim(min(ys) - 3.0, max(ys) + 2.2)

    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_figure(fig, "fig_s6_nusselt_setup")
    plt.close(fig)
    print("wrote fig_s6_nusselt_setup")


if __name__ == "__main__":
    main()
