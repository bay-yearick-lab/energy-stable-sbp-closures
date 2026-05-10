#!/usr/bin/env python3
from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import boundary_spline_analysis as ba
from project_paths import RESULTSDIR, ensure_generated_dirs, save_figure

ensure_generated_dirs()

CROSSQUARTET = json.loads((RESULTSDIR / 'crossquartet_client_summary.json').read_text())
MISSION = json.loads((RESULTSDIR / 'wall_mission_design_summary.json').read_text())
OUT_JSON = RESULTSDIR / 'section6_support_summary.json'
OUT_CSV = RESULTSDIR / 'section6_support_table.csv'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10.0,
    'axes.labelsize': 10.8,
    'axes.titlesize': 11.7,
    'legend.fontsize': 10.6,
    'xtick.labelsize': 8.9,
    'ytick.labelsize': 8.9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 180,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.12,
    'grid.linewidth': 0.45,
})

FAMILY_INFO = [
    ('E', 'Explicit', ba.DEFAULT_COLORS['E'], ba.DEFAULT_MARKERS['E'], ba.DEFAULT_LINESTYLES['E'], -0.25),
    ('C', 'Compact', ba.DEFAULT_COLORS['C'], ba.DEFAULT_MARKERS['C'], ba.DEFAULT_LINESTYLES['C'], -0.15),
    ('SBP', 'SBP-SAT', ba.DEFAULT_COLORS['SBP'], ba.DEFAULT_MARKERS['SBP'], ba.DEFAULT_LINESTYLES['SBP'], -0.05),
    ('DRP', 'DRP', ba.DEFAULT_COLORS['DRP'], ba.DEFAULT_MARKERS['DRP'], ba.DEFAULT_LINESTYLES['DRP'], 0.05),
    ('BMF', 'BMF', ba.DEFAULT_COLORS['BMF'], ba.DEFAULT_MARKERS['BMF'], ba.DEFAULT_LINESTYLES['BMF'], 0.15),
    ('BMG', 'BMG', ba.DEFAULT_COLORS['BMG'], ba.DEFAULT_MARKERS['BMG'], ba.DEFAULT_LINESTYLES['BMG'], 0.25),
]
ORDERS = [4, 6, 8]


def values(metric_block: str, metric_key: str, family_prefix: str) -> list[float]:
    out = []
    for q in ORDERS:
        label = f'{family_prefix}{q}'
        out.append(CROSSQUARTET['orders'][f'q{q}'][metric_block][label][metric_key])
    return out


transfer_handles = [
    Line2D([0], [0], color=color, lw=2.25, ls=linestyle, marker=marker, ms=6.2, mfc='white', mec=color, mew=1.0, label=name)
    for _prefix, name, color, marker, linestyle, _xoff in FAMILY_INFO
]


def _plot_support_panel(ax, block: str, key: str, title: str, ylabel: str) -> None:
    for prefix, name, color, marker, linestyle, xoff in FAMILY_INFO:
        series = values(block, key, prefix)
        ax.plot(
            np.asarray(ORDERS, dtype=float) + xoff,
            series,
            marker=marker,
            ms=6.2,
            lw=2.25,
            ls=linestyle,
            color=color,
            label=name,
            mfc='white',
            mec=color,
            mew=1.0,
            clip_on=False,
        )
    ax.set_yscale('log')
    ax.set_xticks(ORDERS)
    ax.set_xlim(3.5, 8.5)
    ax.set_xlabel('matched order $q$')
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8, fontweight='semibold')


def _support_legend(fig) -> None:
    fig.legend(
        transfer_handles,
        [handle.get_label() for handle in transfer_handles],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.985),
        ncol=6,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.4,
        fontsize=11.1,
    )


geom_fig, geom_axes = plt.subplots(1, 2, figsize=(11.35, 4.80))
geom_fig.subplots_adjust(left=0.08, right=0.995, bottom=0.17, top=0.77, wspace=0.24)
_plot_support_panel(
    geom_axes[0], 'geometry_transfer', 'value_rel_l2',
    '(a) Marker value',
    r'$\|u-u_{\mathrm{ex}}\|_H / \|u_{\mathrm{ex}}\|_H$',
)
_plot_support_panel(
    geom_axes[1], 'geometry_transfer', 'normal_rel_l2',
    '(b) Marker normal',
    r'$\|\partial_n u-\partial_n u_{\mathrm{ex}}\|_H / \|\partial_n u_{\mathrm{ex}}\|_H$',
)
_support_legend(geom_fig)
save_figure(geom_fig, 'fig_s6_transfer_support', pad_inches=0.03)
plt.close(geom_fig)

trac_fig, trac_axes = plt.subplots(1, 2, figsize=(11.35, 4.80))
trac_fig.subplots_adjust(left=0.08, right=0.995, bottom=0.17, top=0.77, wspace=0.24)
_plot_support_panel(
    trac_axes[0], 'traction_transfer', 'traction_rel_l2',
    '(a) Traction density',
    r'$\|\tau-\tau_{\mathrm{ex}}\|_H / \|\tau_{\mathrm{ex}}\|_H$',
)
_plot_support_panel(
    trac_axes[1], 'traction_transfer', 'force_rel',
    '(b) Net force',
    r'$|F-F_{\mathrm{ex}}| / |F_{\mathrm{ex}}|$',
)
_support_legend(trac_fig)
save_figure(trac_fig, 'fig_s6_traction_support', pad_inches=0.03)
plt.close(trac_fig)

rows: list[dict[str, object]] = []
summary: dict[str, object] = {'orders': {}}
for q in ORDERS:
    qkey = f'q{q}'
    block = CROSSQUARTET['orders'][qkey]
    mission_block = MISSION['orders'][qkey]
    summary['orders'][qkey] = {
        'strict_response_p1pct': {
            label: MISSION['orders'][qkey]['required_ppw_snapshots']['M=100'][label]
            for label in MISSION['orders'][qkey]['required_ppw_snapshots']['M=100']
        },
        'reflected_wave': {
            label: block['reflected_wave'][label]['post_bounce_error']
            for label in block['reflected_wave']
        },
        'geometry_marker_normal': {
            label: block['geometry_transfer'][label]['normal_rel_l2']
            for label in block['geometry_transfer']
        },
        'traction_density': {
            label: block['traction_transfer'][label]['traction_rel_l2']
            for label in block['traction_transfer']
        },
        'energy_balance': {
            label: block['energy_balance'][label]['max_rel_artificial_rate']
            for label in block['energy_balance']
        },
        'weighted_sbp_residual': {
            label: block['energy_balance'][label]['balance_residual_norm2']
            for label in block['energy_balance']
        },
        'winner_crossovers_nodal': mission_block['winner_crossovers_nodal'],
        'winner_crossovers_bmg_coefficient': mission_block['winner_crossovers_bmg_coefficient'],
    }
    rows.append({
        'q': q,
        'strict_response_at_M100': min(
            mission_block['required_ppw_snapshots']['M=100'].items(),
            key=lambda item: item[1],
        ),
        'post_bounce_packet': min(
            ((label, val['post_bounce_error']) for label, val in block['reflected_wave'].items()),
            key=lambda item: item[1],
        ),
        'marker_normal': min(
            ((label, val['normal_rel_l2']) for label, val in block['geometry_transfer'].items()),
            key=lambda item: item[1],
        ),
        'traction_density': min(
            ((label, val['traction_rel_l2']) for label, val in block['traction_transfer'].items()),
            key=lambda item: item[1],
        ),
        'sample_energy_rate': min(
            ((label, val['max_rel_artificial_rate']) for label, val in block['energy_balance'].items()),
            key=lambda item: item[1],
        ),
        'exact_weighted_sbp': f'BMG{q}, BMF{q}',
    })

with OUT_CSV.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'q',
        'lowest_strict_response_scheme',
        'lowest_strict_response_value_at_M100',
        'lowest_post_bounce_scheme',
        'lowest_post_bounce_error',
        'lowest_marker_normal_scheme',
        'lowest_marker_normal_error',
        'lowest_traction_scheme',
        'lowest_traction_density_error',
        'lowest_sample_energy_scheme',
        'lowest_sample_energy_rate',
        'exact_weighted_sbp_scheme',
    ])
    for row in rows:
        writer.writerow([
            row['q'],
            row['strict_response_at_M100'][0],
            row['strict_response_at_M100'][1],
            row['post_bounce_packet'][0],
            row['post_bounce_packet'][1],
            row['marker_normal'][0],
            row['marker_normal'][1],
            row['traction_density'][0],
            row['traction_density'][1],
            row['sample_energy_rate'][0],
            row['sample_energy_rate'][1],
            row['exact_weighted_sbp'],
        ])

OUT_JSON.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
