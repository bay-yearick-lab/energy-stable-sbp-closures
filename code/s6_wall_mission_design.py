#!/usr/bin/env python3
from __future__ import annotations

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

import boundary_spline_analysis as ba
from project_paths import RESULTSDIR, ensure_generated_dirs, save_figure

OUT = RESULTSDIR / 'wall_mission_design_summary.json'
ensure_generated_dirs()

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9.7,
    'axes.labelsize': 10.6,
    'axes.titlesize': 11.4,
    'legend.fontsize': 10.8,
    'xtick.labelsize': 8.7,
    'ytick.labelsize': 8.7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 180,
    'savefig.dpi': 300,
    'axes.grid': False,
})

ORDER_BLOCKS = (4, 6, 8)
FAMILY_ORDER = ba.family_order()
FAMILY_COLORS = ba.DEFAULT_COLORS
NPTS = 129
MISSION_WINDOW = (10.0, 1000.0)
PPW_BUDGET_WINDOW = (2.0, 12.0)
TOL_CYCLES = 0.25


def scheme_label(family: str, q: int) -> str:
    return f'{family}{q}'


def response_requirement(ppw: np.ndarray, response_error: np.ndarray, mission_length: float, tol_cycles: float) -> float:
    tol = float(tol_cycles) / float(mission_length)
    idx = np.where(response_error <= tol)[0]
    return float(ppw[idx[-1]]) if idx.size else float('inf')


def piecewise_winner(
    mission_grid: np.ndarray,
    labels: list[str],
    required_ppw: dict[str, np.ndarray],
    work_map: dict[str, float],
    dim: int,
) -> list[dict[str, float | str]]:
    winners: list[str] = []
    for j in range(mission_grid.size):
        candidates: list[tuple[float, str]] = []
        for label in labels:
            req = required_ppw[label][j]
            if np.isfinite(req):
                candidates.append((work_map[label] * req ** (dim + 1), label))
        winners.append(min(candidates)[1] if candidates else 'none')

    out: list[dict[str, float | str]] = [{'mission_length': float(mission_grid[0]), 'winner': winners[0]}]
    prev = winners[0]
    for mission_length, winner in zip(mission_grid[1:], winners[1:], strict=True):
        if winner != prev:
            out.append({'mission_length': float(mission_length), 'winner': winner})
            prev = winner
    return out


def build_order_data(q: int) -> dict[str, object]:
    labels = [scheme_label(family, q) for family in FAMILY_ORDER]
    ops = {label: ba.build_operator(ba.scheme_kind(label), q, NPTS) for label in labels}
    work_nodal = {label: float(op.work_model.total_flops / op.x.size) for label, op in ops.items()}
    work_coefficient = dict(work_nodal)
    bmg_label = scheme_label('BMG', q)
    work_coefficient[bmg_label] = float(ops[bmg_label].work_model_coefficient.total_flops / ops[bmg_label].x.size)

    curves: dict[str, dict[str, np.ndarray]] = {}
    for label, op in ops.items():
        response = ba.wall_mode_response(op)
        curves[label] = {
            'ppw': np.array([row.ppw for row in response], dtype=float),
            'response_error': np.array([abs(row.kappa_ratio - 1.0) for row in response], dtype=float),
        }

    mission_lengths = np.geomspace(MISSION_WINDOW[0], MISSION_WINDOW[1], 50)
    ppw_budget = np.linspace(PPW_BUDGET_WINDOW[0], PPW_BUDGET_WINDOW[1], 50)

    required_ppw = {
        label: np.array(
            [response_requirement(curves[label]['ppw'], curves[label]['response_error'], mission, TOL_CYCLES) for mission in mission_lengths],
            dtype=float,
        )
        for label in labels
    }

    return {
        'labels': labels,
        'ops': ops,
        'work_nodal': work_nodal,
        'work_coefficient': work_coefficient,
        'curves': curves,
        'mission_lengths': mission_lengths,
        'ppw_budget': ppw_budget,
        'required_ppw': required_ppw,
    }


order_data = {q: build_order_data(q) for q in ORDER_BLOCKS}

fig, axes = plt.subplots(3, 3, figsize=(13.0, 10.15), gridspec_kw={'wspace': 0.14, 'hspace': 0.20})
cmap = ListedColormap([FAMILY_COLORS[family] for family in FAMILY_ORDER])
norm = BoundaryNorm(np.arange(-0.5, len(FAMILY_ORDER) + 0.5, 1.0), cmap.N)

for col, dim in enumerate((1, 2, 3), start=0):
    axes[0, col].set_title(rf'$d={dim}$', pad=10, fontweight='semibold')

for row, q in enumerate(ORDER_BLOCKS):
    data = order_data[q]
    labels = data['labels']
    mission_lengths = data['mission_lengths']
    ppw_budget = data['ppw_budget']
    required_ppw = data['required_ppw']
    work_nodal = data['work_nodal']
    for col, dim in enumerate((1, 2, 3), start=0):
        ax = axes[row, col]
        decision = np.full((ppw_budget.size, mission_lengths.size), -1, dtype=int)
        for i, p_cap in enumerate(ppw_budget):
            for j, _mission in enumerate(mission_lengths):
                candidates: list[tuple[float, int]] = []
                for idx, label in enumerate(labels):
                    req = required_ppw[label][j]
                    if np.isfinite(req) and req <= p_cap:
                        recurring_work = work_nodal[label] * req ** (dim + 1)
                        candidates.append((recurring_work, idx))
                if candidates:
                    decision[i, j] = min(candidates)[1]

        ax.set_facecolor('#e8e6df')
        mission_mesh, budget_mesh = np.meshgrid(mission_lengths, ppw_budget)
        masked = np.ma.masked_where(decision < 0, decision)
        ax.pcolormesh(
            mission_mesh,
            budget_mesh,
            masked,
            cmap=cmap,
            norm=norm,
            shading='nearest',
            alpha=0.97,
        )
        contour_field = np.ma.masked_where(decision < 0, decision.astype(float))
        ax.contour(
            mission_mesh,
            budget_mesh,
            contour_field,
            levels=np.arange(0.5, len(FAMILY_ORDER) - 0.5, 1.0),
            colors='white',
            linewidths=0.65,
            alpha=0.72,
        )
        ax.set_xscale('log')
        ax.set_xlim(*MISSION_WINDOW)
        ax.set_ylim(*PPW_BUDGET_WINDOW)
        if col == 0:
            ax.set_ylabel(rf'$q={q}$' + '\n' + r'available PPW $P_{\max}$')
        if row == len(ORDER_BLOCKS) - 1:
            ax.set_xlabel(r'mission length $M$ [standing-wave periods]')

legend_handles = [
    Patch(facecolor=FAMILY_COLORS['E'], label='explicit'),
    Patch(facecolor=FAMILY_COLORS['C'], label='compact'),
    Patch(facecolor=FAMILY_COLORS['SBP'], label='SBP-SAT'),
    Patch(facecolor=FAMILY_COLORS['DRP'], label='DRP'),
    Patch(facecolor=FAMILY_COLORS['BMF'], label='BMF'),
    Patch(facecolor=FAMILY_COLORS['BMG'], label='BMG'),
    Patch(facecolor='#e8e6df', label='no admissible scheme'),
]
fig.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.992),
    ncol=4,
    frameon=False,
    handlelength=1.9,
    columnspacing=1.45,
    fontsize=11.4,
)
fig.subplots_adjust(left=0.08, right=0.99, top=0.79, bottom=0.09, wspace=0.14, hspace=0.23)
save_figure(fig, 'fig_s6_wall_mission_design', pad_inches=0.03)
plt.close(fig)

summary = {
    'budget': {
        'strict_response_metric': '|rho_m - 1|',
        'tol_cycles': TOL_CYCLES,
        'mission_unit': 'standing-wave periods',
        'work_proxy': 'shared-CFL recurring work ~ work_per_point * P^(d+1)',
    },
    'orders': {},
}

for q in ORDER_BLOCKS:
    data = order_data[q]
    labels = data['labels']
    mission_lengths = data['mission_lengths']
    required_ppw = data['required_ppw']
    work_nodal = data['work_nodal']
    work_coefficient = data['work_coefficient']
    summary['orders'][f'q{q}'] = {
        'labels': labels,
        'work_per_point_nodal': work_nodal,
        'work_per_point_coefficient': {
            scheme_label('BMG', q): work_coefficient[scheme_label('BMG', q)],
        },
        'required_ppw_snapshots': {
            'M=10': {label: float(required_ppw[label][0]) for label in labels},
            'M=100': {label: float(required_ppw[label][np.argmin(np.abs(mission_lengths - 100.0))]) for label in labels},
            'M=1000': {label: float(required_ppw[label][-1]) for label in labels},
        },
        'winner_crossovers_nodal': {
            f'd{dim}': piecewise_winner(mission_lengths, labels, required_ppw, work_nodal, dim) for dim in (1, 2, 3)
        },
        'winner_crossovers_bmg_coefficient': {
            f'd{dim}': piecewise_winner(mission_lengths, labels, required_ppw, work_coefficient, dim) for dim in (1, 2, 3)
        },
    }

OUT.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
