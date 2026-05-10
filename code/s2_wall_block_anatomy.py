#!/usr/bin/env python3
"""Generate `fig_s2_wall_block_anatomy.pdf`.

Compact family cards for Section 2.  Replaces the previous
matrix-block-plus-numeric-strip view with four small cards, one per
matched-order first-derivative family on the $p=q=6$, $N=33$ block.

Each card carries three pieces:

  family label   the matched-order family name and degree.
  wall-row       a small signed-stem footprint of row 0 of the operator.
  moment ladder  cells delta_{0,1}..delta_{0,8}, filled when zero,
                 open when nonzero, with a one-line zero count.
  mechanism tag  one phrase that names how the zeros are obtained
                 (tuned closure, SBP energy closure, spline reproduction).

All numbers are computed at figure-build time from the actual operators in
`boundary_spline_analysis`.  The figure communicates the takeaway without
needing the surrounding caption to explain it.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import boundary_spline_analysis as ba
from project_paths import ensure_generated_dirs, save_figure

ensure_generated_dirs()

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8.6,
    'axes.labelsize': 8.6,
    'axes.titlesize': 10.0,
    'xtick.labelsize': 7.6,
    'ytick.labelsize': 7.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 180,
    'savefig.dpi': 300,
})


BLUE = '#1f77b4'
RED = '#d62728'
ACCENT = '#2F4858'
GUIDE = '#D9D2C5'
PANEL_BG = (0.987, 0.985, 0.978)


N_PTS = 33
DEGREE = 6
MAX_ELL = 8


FAMILIES = [
    ('E', r'$\mathrm{E}_{6}$', 'tuned closure'),
    ('SBP', r'$\mathrm{SBP}_{6}$', 'SBP energy closure'),
    ('B', r'$\mathrm{BMF}_{6}$', 'spline reproduction'),
    ('I', r'$\mathrm{BMG}_{6}$', 'spline reproduction'),
]


def closure_moments(operator: ba.Operator1D, row: int, max_ell: int) -> np.ndarray:
    x = np.asarray(operator.x, dtype=float)
    out = np.zeros(max_ell + 1, dtype=float)
    for ell in range(max_ell + 1):
        u = x ** ell
        du_exact = 0.0 if ell == 0 else ell * (x[row] ** (ell - 1))
        du_num = float(operator.differentiate(u)[row])
        out[ell] = du_num - du_exact
    return out


def wall_row_entries(operator: ba.Operator1D, *, max_show: int = 12,
                     rel_tol: float = 5.0e-4) -> np.ndarray:
    """Return signed wall-row entries normalized by max magnitude.

    Entries below `rel_tol` relative to the largest magnitude are dropped, so
    spline rows that decay geometrically are visually compact.
    """
    row = np.asarray(operator.matrix[0], dtype=float)
    max_abs = float(np.max(np.abs(row)))
    if max_abs == 0.0:
        return np.zeros(0)
    keep = np.where(np.abs(row) >= rel_tol * max_abs)[0]
    keep = keep[:max_show]
    return row[keep] / max_abs


def draw_family_card(ax: plt.Axes, kind: str, label: str,
                     mechanism: str) -> None:
    op = ba.build_operator(kind, DEGREE, N_PTS)
    moments = closure_moments(op, row=0, max_ell=MAX_ELL)
    weights = wall_row_entries(op)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(PANEL_BG)
    for side in ('left', 'right', 'top', 'bottom'):
        ax.spines[side].set_visible(False)

    ax.text(0.04, 0.92, label, ha='left', va='center',
            fontsize=14.0, family='serif', color=ACCENT,
            transform=ax.transAxes)
    ax.text(0.96, 0.92, mechanism, ha='right', va='center',
            fontsize=8.4, family='serif', color='0.30',
            style='italic', transform=ax.transAxes)
    ax.text(0.04, 0.81, r'wall row $i\!=\!0$', ha='left', va='center',
            fontsize=7.4, family='serif', color='0.45',
            style='italic', transform=ax.transAxes)

    n_show = max(weights.size, 1)
    track_left = 0.06
    track_right = 0.94
    track_y = 0.55
    track_w = track_right - track_left
    spacing = track_w / max(n_show - 1, 1) if n_show > 1 else 0.0
    ax.plot([track_left, track_right], [track_y, track_y],
            color='0.55', lw=0.6, zorder=1, transform=ax.transAxes)

    stem_amp = 0.18
    for k, w in enumerate(weights):
        if n_show == 1:
            x = (track_left + track_right) / 2.0
        else:
            x = track_left + k * spacing
        color = BLUE if w >= 0 else RED
        h = stem_amp * float(np.clip(w, -1.0, 1.0))
        ax.plot([x, x], [track_y, track_y + h], color=color, lw=1.6,
                solid_capstyle='round', zorder=3, transform=ax.transAxes)
        ax.plot([x], [track_y + h], marker='o', color=color, markersize=2.6,
                linestyle='None', zorder=3, transform=ax.transAxes)

    ax.text(0.04, 0.32, r'$\delta_{0,\ell}$ ladder',
            ha='left', va='center', fontsize=7.4,
            family='serif', color='0.45', style='italic',
            transform=ax.transAxes)

    ladder_left = 0.06
    ladder_right = 0.94
    ladder_y = 0.18
    cell_w = (ladder_right - ladder_left) / MAX_ELL
    cell_h = 0.10

    zero_count = 0
    first_defect_ell: int | None = None
    for ell in range(1, MAX_ELL + 1):
        cx = ladder_left + (ell - 1) * cell_w
        val = moments[ell]
        is_zero = abs(val) < 5.0e-12
        if is_zero:
            zero_count += 1
            ax.add_patch(plt.Rectangle((cx, ladder_y), cell_w * 0.92, cell_h,
                                       facecolor=ACCENT, edgecolor=ACCENT,
                                       linewidth=0.5, alpha=0.85,
                                       transform=ax.transAxes))
        else:
            ax.add_patch(plt.Rectangle((cx, ladder_y), cell_w * 0.92, cell_h,
                                       facecolor='white', edgecolor='0.55',
                                       linewidth=0.6,
                                       transform=ax.transAxes))
            if first_defect_ell is None:
                first_defect_ell = ell
        ax.text(cx + cell_w * 0.46, ladder_y - 0.05, str(ell),
                ha='center', va='top', fontsize=6.2, color='0.45',
                family='serif', transform=ax.transAxes)

    summary = rf'{zero_count}/{MAX_ELL} structural zeros'
    if first_defect_ell is not None:
        summary = summary + rf'   first defect at $\ell={first_defect_ell}$'
    ax.text(0.5, 0.005, summary, ha='center', va='bottom',
            fontsize=7.6, family='serif', color=ACCENT,
            transform=ax.transAxes)


def main() -> None:
    fig, axs = plt.subplots(2, 2, figsize=(8.6, 4.6), layout='constrained',
                            gridspec_kw={'hspace': 0.20, 'wspace': 0.12})
    for ax, (kind, label, mech) in zip(axs.flat, FAMILIES):
        draw_family_card(ax, kind, label, mech)
    save_figure(fig, 'fig_s2_wall_block_anatomy', pad_inches=0.04)
    plt.close(fig)
    print('wrote wall-block anatomy figure')


if __name__ == '__main__':
    main()
