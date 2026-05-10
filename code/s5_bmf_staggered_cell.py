#!/usr/bin/env python3
"""Schematic for Section 5.2: BMF staggered Greville flux complex.

A single-panel illustration that shows why placing the BMF gradient in the
shortened degree-(p-1) derivative spline space, sampled on its own Greville
nodes, makes the SBP identity collapse to a pure boundary term. Two parallel
strips are drawn: the primal Greville grid carrying nodal values
``u_i = s(x_i)`` of a degree-``p`` spline ``s``, and the derivative Greville
grid carrying nodal fluxes ``q_j = s'(eta_j)`` of the exact derivative spline
``s'`` in the shortened space. One representative staggered cell
``[eta_{j-1}, eta_j]`` is highlighted around an interior primal node, and
the two boundary traces ``s'(0)`` and ``s'(1)`` sit at the walls as the wall
extrapolations of the derivative spline. Algebraic identities live in the
figure caption.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.interpolate import BSpline

import boundary_spline_analysis as ba
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


N = 9
P = 4

knots_p = ba.open_uniform_knots(N, P)
x_primal = ba.greville_from_knots(knots_p, P, N)

knots_pm1 = ba.derivative_space_knots(knots_p)
P_DERIV = P - 1
eta = ba.derivative_greville_nodes(knots_pm1, P_DERIV)

alpha = np.array(
    [0.04, 0.20, 0.50, 0.82, 0.96, 0.83, 0.55, 0.26, 0.05], dtype=float
)
assert alpha.size == N

s = BSpline(knots_p, alpha.copy(), P, extrapolate=False)
s_prime = s.derivative()

xx = np.linspace(0.0, 1.0, 801)
s_vals = np.asarray(s(xx), dtype=float)
sp_vals = np.asarray(s_prime(xx), dtype=float)

u_at_x = np.asarray(s(x_primal), dtype=float)
q_at_eta = np.asarray(s_prime(eta), dtype=float)
q_left_wall = float(s_prime(0.0))
q_right_wall = float(s_prime(1.0))


PRIMAL = '#22517a'
DERIV = '#a83232'
WALL = '#1a1a1a'
TRACE = '#1c7c34'
CELL_FACE = '#f4ddb1'
CELL_EDGE = '#a8821b'
PRIMAL_BG = '#dbe7f1'
DERIV_BG = '#f1d9d3'


y_primal = 5.0
y_deriv = 2.0
amp_p = 1.4
amp_d = 1.4

s_min, s_max = float(s_vals.min()), float(s_vals.max())
sp_min, sp_max = float(sp_vals.min()), float(sp_vals.max())


def primal_y(values: np.ndarray | float) -> np.ndarray | float:
    return y_primal + amp_p * (values - s_min) / (s_max - s_min + 1.0e-12)


def deriv_y(values: np.ndarray | float) -> np.ndarray | float:
    return y_deriv + amp_d * (values - sp_min) / (sp_max - sp_min + 1.0e-12)


fig, ax = plt.subplots(figsize=(11.6, 5.4), layout='constrained')


strip_xL, strip_xR = -0.02, 1.02
mid_strip = 0.5 * (y_primal + y_deriv)
primal_strip = Rectangle(
    (strip_xL, mid_strip), strip_xR - strip_xL, (y_primal + amp_p + 1.05) - mid_strip,
    facecolor=PRIMAL_BG, edgecolor='none', alpha=0.55, zorder=0,
)
deriv_strip = Rectangle(
    (strip_xL, y_deriv - 0.95), strip_xR - strip_xL, mid_strip - (y_deriv - 0.95),
    facecolor=DERIV_BG, edgecolor='none', alpha=0.55, zorder=0,
)
ax.add_patch(primal_strip)
ax.add_patch(deriv_strip)


ax.plot(xx, primal_y(s_vals), color=PRIMAL, lw=1.7, zorder=4)
ax.plot(xx, deriv_y(sp_vals), color=DERIV, lw=1.7, zorder=4)

ax.plot([0.0, 1.0], [y_primal, y_primal], color='0.55', lw=0.6, zorder=1)
ax.plot([0.0, 1.0], [y_deriv, y_deriv], color='0.55', lw=0.6, zorder=1)

for xi, vi in zip(x_primal, u_at_x):
    ax.plot([xi, xi], [y_primal, primal_y(float(vi))],
            color=PRIMAL, lw=0.55, alpha=0.45, zorder=2)
for ej, vj in zip(eta, q_at_eta):
    ax.plot([ej, ej], [y_deriv, deriv_y(float(vj))],
            color=DERIV, lw=0.55, alpha=0.45, zorder=2)

ax.scatter(
    x_primal, np.full_like(x_primal, y_primal),
    s=78, marker='o', facecolor=PRIMAL, edgecolor='white', linewidth=1.0, zorder=6,
)
ax.scatter(
    eta, np.full_like(eta, y_deriv),
    s=86, marker='s', facecolor=DERIV, edgecolor='white', linewidth=1.0, zorder=6,
)

for xw in (0.0, 1.0):
    ax.plot([xw, xw], [0.55, y_primal + amp_p + 0.55],
            color=WALL, lw=2.2, zorder=5, solid_capstyle='butt')

i_cell = 4
xi_cell = x_primal[i_cell]
left_idx = max(0, int(np.searchsorted(eta, xi_cell)) - 1)
right_idx = min(eta.size - 1, left_idx + 1)
e_l, e_r = float(eta[left_idx]), float(eta[right_idx])

cell_top = y_primal + amp_p + 0.10
cell_bot = y_deriv - 0.45
cell = Rectangle(
    (e_l, cell_bot), e_r - e_l, cell_top - cell_bot,
    facecolor=CELL_FACE, edgecolor='none', alpha=0.45, zorder=0,
)
ax.add_patch(cell)
for xv in (e_l, e_r):
    ax.plot([xv, xv], [cell_bot, cell_top],
            color=CELL_EDGE, lw=0.7, ls=(0, (3.4, 1.8)), alpha=0.40, zorder=2)

# Quiet italic cell label inside the highlight, just above the primal baseline.
ax.text(
    0.5 * (e_l + e_r), y_primal + amp_p + 0.30,
    r'staggered cell',
    ha='center', va='bottom', fontsize=10.6, color=CELL_EDGE,
    style='italic',
)

# Boundary trace stars at the walls; quiet labels sit beside the stars.
y_left_trace = float(deriv_y(q_left_wall))
y_right_trace = float(deriv_y(q_right_wall))
ax.scatter(
    [0.0, 1.0], [y_left_trace, y_right_trace],
    s=200, marker='*', facecolor=TRACE, edgecolor='white', linewidth=1.0, zorder=8,
)

ax.text(
    0.025, y_left_trace, r"$s^{\prime}(0)$",
    ha='left', va='center', fontsize=11.6, color=TRACE,
    family='serif', style='italic',
)
ax.text(
    0.975, y_right_trace, r"$s^{\prime}(1)$",
    ha='right', va='center', fontsize=11.6, color=TRACE,
    family='serif', style='italic',
)

# Operator arrows: G_{BMF} on the left, D_{BMF} on the right
gx = -0.085
arrow_g = FancyArrowPatch(
    (gx, y_primal - 0.05), (gx, y_deriv + 0.05),
    arrowstyle='-|>', mutation_scale=14, color='0.20', lw=1.4, zorder=5,
)
ax.add_patch(arrow_g)
ax.text(
    gx - 0.020, 0.5 * (y_primal + y_deriv), r'$G_{\mathrm{BMF}}$',
    ha='right', va='center', fontsize=12.5, color='0.10',
)

dx_arrow = 1.085
arrow_d = FancyArrowPatch(
    (dx_arrow, y_deriv + 0.05), (dx_arrow, y_primal - 0.05),
    arrowstyle='-|>', mutation_scale=14, color='0.20', lw=1.4, zorder=5,
)
ax.add_patch(arrow_d)
ax.text(
    dx_arrow + 0.020, 0.5 * (y_primal + y_deriv), r'$D_{\mathrm{BMF}}$',
    ha='left', va='center', fontsize=12.5, color='0.10',
)

ax.text(
    0.5, y_primal + amp_p + 1.05,
    r"primal Greville $\{x_i\}$,$\quad u_i = s(x_i)\in S_h^{p}$",
    ha='center', va='bottom', fontsize=12.0, color=PRIMAL,
)
ax.text(
    0.5, y_deriv - 1.20,
    r"derivative Greville $\{\eta_j\}$,$\quad q_j = s'(\eta_j)\in S_h^{\,p-1}$",
    ha='center', va='top', fontsize=12.0, color=DERIV,
)

ax.text(0.0, 0.25, r'wall, $x{=}0$', ha='center', va='top', fontsize=11.2)
ax.text(1.0, 0.25, r'wall, $x{=}1$', ha='center', va='top', fontsize=11.2)


ax.set_xlim(-0.18, 1.18)
ax.set_ylim(-0.20, y_primal + amp_p + 1.65)
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)


save_figure(fig, 'fig_s5_bmf_staggered_cell', pad_inches=0.04)
plt.close(fig)
print('wrote BMF staggered cell schematic')
