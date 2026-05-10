#!/usr/bin/env python3
"""Schematic for Section 6: spline MAC variable placement for the BMF
incompressible Navier-Stokes formulation.

A two-dimensional view of the staggered tensor-product BMF complex shows
where each primary variable lives:

  * pressure / divergence residual on the primal-by-primal lattice,
  * the velocity component ``u_1`` on the (derivative-x, primal-y) face
    lattice (east-west faces of the scalar control volume),
  * the velocity component ``u_2`` on the (primal-x, derivative-y) face
    lattice (north-south faces of the scalar control volume).

One scalar control volume is highlighted to make the local divergence
equation an explicit four-face flux-difference balance, which is the
content of the local mass conservation statement
in Theorem~\ref{thm:bmf-local-fv} applied direction by direction.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

from project_paths import ensure_generated_dirs, save_figure


ensure_generated_dirs()

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10.6,
    'axes.titlesize': 11.2,
    'figure.dpi': 180,
    'savefig.dpi': 300,
})


# ---- spline data ---------------------------------------------------------
N = 5
xx = np.linspace(0.0, 1.0, N)                           # idealized primal grid
eta = 0.5 * (xx[:-1] + xx[1:])                          # idealized face grid

# Highlight one scalar control volume around the central pressure node
ic, jc = N // 2, N // 2
xc, yc = float(xx[ic]), float(xx[jc])
i_left = ic - 1
i_right = ic
j_bot = jc - 1
j_top = jc
x_l, x_r = float(eta[i_left]), float(eta[i_right])
y_b, y_t = float(eta[j_bot]), float(eta[j_top])
ilab, jlab = 3, 3


# ---- palette -------------------------------------------------------------
PRESSURE = '#22517a'   # blue, primal x primal
U1_COLOR = '#a83232'   # red, deriv-x x primal-y  (east/west faces)
U2_COLOR = '#1c7c34'   # green, primal-x x deriv-y (north/south faces)
CELL_FACE = '#f4ddb1'
CELL_EDGE = '#a8821b'

MATH_FONTSIZE = 12.8
LABEL_HALO = [pe.withStroke(linewidth=2.4, foreground='white')]
ARROW_HALO = [pe.withStroke(linewidth=4.4, foreground='white'), pe.Normal()]


# ---- figure --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.6), layout='constrained')

# Faint background grid lines marking primal / derivative positions
for x in xx:
    ax.axvline(x, color='0.85', lw=0.55, zorder=0)
    ax.axhline(x, color='0.85', lw=0.55, zorder=0)
for x in eta:
    ax.axvline(x, color='0.93', lw=0.55, ls=(0, (1.5, 1.4)), zorder=0)
    ax.axhline(x, color='0.93', lw=0.55, ls=(0, (1.5, 1.4)), zorder=0)

# Pressure nodes (primal x primal)
PX, PY = np.meshgrid(xx, xx, indexing='ij')
ax.scatter(PX.ravel(), PY.ravel(),
           s=44, marker='o', facecolor=PRESSURE,
           edgecolor='white', linewidth=0.8, zorder=4)

# u_1 nodes (deriv-x, primal-y)
U1X, U1Y = np.meshgrid(eta, xx, indexing='ij')
u1_mask = ~(
    (np.isclose(U1X, x_l) | np.isclose(U1X, x_r)) & np.isclose(U1Y, yc)
)
ax.plot(U1X[u1_mask], U1Y[u1_mask], linestyle='none', marker='|',
        markersize=7.5, markeredgewidth=1.55, color=U1_COLOR,
        alpha=0.68, zorder=4)

# u_2 nodes (primal-x, deriv-y)
U2X, U2Y = np.meshgrid(xx, eta, indexing='ij')
u2_mask = ~(
    np.isclose(U2X, xc) & (np.isclose(U2Y, y_b) | np.isclose(U2Y, y_t))
)
ax.plot(U2X[u2_mask], U2Y[u2_mask], linestyle='none', marker='_',
        markersize=7.5, markeredgewidth=1.55, color=U2_COLOR,
        alpha=0.68, zorder=4)


cell = Rectangle(
    (x_l, y_b), x_r - x_l, y_t - y_b,
    facecolor=CELL_FACE, edgecolor=CELL_EDGE,
    lw=1.0, ls=(0, (4.0, 2.0)), alpha=0.42, zorder=1,
)
ax.add_patch(cell)

# Emphasized pressure marker. The four face fluxes are carried by arrows below
# so the schematic does not double-count direction with both markers and arrows.
ax.scatter([xc], [yc], s=110, marker='o', facecolor=PRESSURE,
           edgecolor='white', linewidth=1.0, zorder=6)

# Flux-direction overlay arrows showing signed contribution to the divergence
arrow_kwargs = dict(
    arrowstyle='-|>',
    mutation_scale=21,
    lw=2.7,
    zorder=7,
    alpha=0.96,
    capstyle='round',
    joinstyle='round',
    path_effects=ARROW_HALO,
)
ax.add_patch(FancyArrowPatch(
    (x_l - 0.105, yc), (x_l + 0.050, yc), color=U1_COLOR, **arrow_kwargs,
))
ax.add_patch(FancyArrowPatch(
    (x_r - 0.050, yc), (x_r + 0.105, yc), color=U1_COLOR, **arrow_kwargs,
))
ax.add_patch(FancyArrowPatch(
    (xc, y_b - 0.105), (xc, y_b + 0.050), color=U2_COLOR, **arrow_kwargs,
))
ax.add_patch(FancyArrowPatch(
    (xc, y_t - 0.050), (xc, y_t + 0.105), color=U2_COLOR, **arrow_kwargs,
))

# Cell-node and face-flux annotations
ax.annotate(rf"$p_{{{ilab},{jlab}}}$",
            xy=(xc, yc), xytext=(7, 7), textcoords='offset points',
            fontsize=MATH_FONTSIZE, color=PRESSURE, path_effects=LABEL_HALO)

ax.annotate(rf"$F^{{u_1}}_{{{ilab - 1}+\frac{{1}}{{2}},{jlab}}}$",
            xy=(x_l, yc), xytext=(-17, 18), textcoords='offset points',
            fontsize=MATH_FONTSIZE, color=U1_COLOR, ha='right',
            path_effects=LABEL_HALO)
ax.annotate(rf"$F^{{u_1}}_{{{ilab}+\frac{{1}}{{2}},{jlab}}}$",
            xy=(x_r, yc), xytext=(17, 18), textcoords='offset points',
            fontsize=MATH_FONTSIZE, color=U1_COLOR, ha='left',
            path_effects=LABEL_HALO)
ax.annotate(rf"$F^{{u_2}}_{{{ilab},{jlab - 1}+\frac{{1}}{{2}}}}$",
            xy=(xc, y_b), xytext=(17, -17), textcoords='offset points',
            fontsize=MATH_FONTSIZE, color=U2_COLOR, ha='left',
            path_effects=LABEL_HALO)
ax.annotate(rf"$F^{{u_2}}_{{{ilab},{jlab}+\frac{{1}}{{2}}}}$",
            xy=(xc, y_t), xytext=(17, 14), textcoords='offset points',
            fontsize=MATH_FONTSIZE, color=U2_COLOR, ha='left',
            path_effects=LABEL_HALO)


# Schematic axes polish
view_pad = 0.20
ax.set_xlim(x_l - view_pad, x_r + view_pad)
ax.set_ylim(y_b - view_pad, y_t + view_pad)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)


# ---- compact in-figure legend --------------------------------------------
handles = [
    Line2D([0], [0], marker='o', linestyle='none', color='none',
           markerfacecolor=PRESSURE, markeredgecolor='white', markersize=8,
           label=r"$p\in V_c$ on primal-$x{\times}$primal-$y$"),
    Line2D([0], [0], marker='|', linestyle='none', color=U1_COLOR,
           markeredgewidth=2.0, markersize=12,
           label=r"$u_1\in V_1$ on deriv-$x{\times}$primal-$y$"),
    Line2D([0], [0], marker='_', linestyle='none', color=U2_COLOR,
           markeredgewidth=2.0, markersize=12,
           label=r"$u_2\in V_2$ on primal-$x{\times}$deriv-$y$"),
]
leg = ax.legend(
    handles=handles, loc='upper left',
    bbox_to_anchor=(0.005, 0.998),
    frameon=True, framealpha=0.95, fontsize=8.6,
    handletextpad=0.8, labelspacing=0.55, borderpad=0.6,
)
leg.get_frame().set_edgecolor('0.65')
leg.get_frame().set_linewidth(0.6)


save_figure(fig, 'fig_s5_bmf_mac_grid', pad_inches=0.05)
plt.close(fig)
print('wrote BMF MAC-grid schematic')
