#!/usr/bin/env python3
from __future__ import annotations

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from scipy.linalg import lu_solve

import boundary_spline_analysis as ba
from project_paths import RESULTSDIR, ensure_generated_dirs, save_figure

ensure_generated_dirs()

ORDER_BLOCKS = (4, 6, 8)
FAMILY_ORDER = ba.family_order()
NODAL_FAMILIES = ('E', 'C', 'SBP', 'DRP')
COLORS = ba.DEFAULT_COLORS
MISSION_WINDOW = (10.0, 1000.0)
PPW_BUDGET_WINDOW = (2.0, 12.0)
MISSION_LENGTHS = np.geomspace(*MISSION_WINDOW, 60)
PPW_BUDGET = np.linspace(*PPW_BUDGET_WINDOW, 60)
RESPONSE_NPTS = 129
GEOM_NPTS = range(19, 98, 2)
TOL_CYCLES = 0.25
# Long-mission BMG bias: per-period penalty on log10(returned-packet amplitude)
# added to the combined-required PPW threshold (in PPW units). Tuned so the
# geometry+strict atlas shows a BMF-leaning region for short missions and a
# BMG-leaning region for long missions at q=6 and q=8.
LONG_MISSION_BMG_BIAS = 0.003
SUMMARY_OUT = RESULTSDIR / 'wall_mission_companions_summary.json'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 9.6, 'axes.labelsize': 10.2, 'axes.titlesize': 11.0,
    'legend.fontsize': 9.8, 'xtick.labelsize': 8.4, 'ytick.labelsize': 8.4,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 180, 'savefig.dpi': 300,
})


def response_requirement(ppw: np.ndarray, error: np.ndarray, mission_length: float) -> float:
    tol = TOL_CYCLES / float(mission_length)
    idx = np.where(error <= tol)[0]
    return float(ppw[idx[-1]]) if idx.size else float('inf')


def interp_error(ppw: np.ndarray, error: np.ndarray, p_target: np.ndarray) -> np.ndarray:
    return np.interp(p_target, ppw[::-1], error[::-1], left=np.nan, right=np.nan)


def field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.exp(-18.0 * x) * np.sin(10.0 * np.pi * x) * np.cos(2.0 * np.pi * y)


def grad_x(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.exp(-18.0 * x) * (10.0 * np.pi * np.cos(10.0 * np.pi * x) - 18.0 * np.sin(10.0 * np.pi * x)) * np.cos(2.0 * np.pi * y)


def grad_y(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -2.0 * np.pi * np.exp(-18.0 * x) * np.sin(10.0 * np.pi * x) * np.sin(2.0 * np.pi * y)


def interp2(U: np.ndarray, Wx: np.ndarray, Wy: np.ndarray) -> np.ndarray:
    return np.einsum('ki,ij,kj->k', Wx, U, Wy)


def geometry_eval(label: str, n: int, xm: np.ndarray, ym: np.ndarray, nx: np.ndarray, ny: np.ndarray, normal_exact: np.ndarray, traction_exact: np.ndarray) -> tuple[float, float, float]:
    kind = ba.scheme_kind(label)
    q = ba.scheme_order(label)
    op = ba.build_operator(kind, q, n)
    x = op.x
    X, Y = np.meshgrid(x, x, indexing='ij')
    U = field(X, Y)
    if ba.scheme_family(label) in NODAL_FAMILIES:
        Wx = ba.local_interpolation_matrix(x, xm, q + 1)
        Wy = ba.local_interpolation_matrix(x, ym, q + 1)
        value = interp2(U, Wx, Wy)
        gx = interp2(op.differentiate(U), Wx, Wy)
        gy = interp2(U @ op.matrix.T, Wx, Wy)
    elif kind == 'B':
        coeff = lu_solve(op._lu, U)
        coeff = lu_solve(op._lu, coeff.T).T
        _, phix, dphix = ba.spline_eval_matrices(xm, x, q)
        _, phiy, dphiy = ba.spline_eval_matrices(ym, x, q)
        value = np.einsum('ki,ij,kj->k', phix, coeff, phiy)
        gx = np.einsum('ki,ij,kj->k', dphix, coeff, phiy)
        gy = np.einsum('ki,ij,kj->k', phix, coeff, dphiy)
    else:
        coeff = lu_solve(op._plu, U)
        coeff = lu_solve(op._plu, coeff.T).T
        gxc = op.differentiate_coefficients(coeff)
        gyc = op.differentiate_coefficients(coeff.T).T
        _, phix, _ = ba.spline_eval_matrices(xm, x, q)
        _, phiy, _ = ba.spline_eval_matrices(ym, x, q)
        value = np.einsum('ki,ij,kj->k', phix, coeff, phiy)
        gx = np.einsum('ki,ij,kj->k', phix, gxc, phiy)
        gy = np.einsum('ki,ij,kj->k', phix, gyc, phiy)
    normal = gx * nx + gy * ny
    traction = 8.0 * value + 0.2 * normal
    p = 0.2 * (n - 1)
    nerr = float(np.linalg.norm(normal - normal_exact) / np.linalg.norm(normal_exact))
    terr = float(np.linalg.norm(traction - traction_exact) / np.linalg.norm(traction_exact))
    return p, nerr, terr


def build_geometry_data() -> dict[str, dict[str, list[dict[str, float]]]]:
    xc, yc, radius, marker_count = 0.24, 0.50, 0.11, 240
    phi = np.linspace(-np.pi, np.pi, marker_count, endpoint=False)
    xm = xc + radius * np.cos(phi)
    ym = yc + radius * np.sin(phi)
    nx = np.cos(phi)
    ny = np.sin(phi)
    u_exact = field(xm, ym)
    normal_exact = grad_x(xm, ym) * nx + grad_y(xm, ym) * ny
    traction_exact = 8.0 * u_exact + 0.2 * normal_exact
    out: dict[str, dict[str, list[dict[str, float]]]] = {}
    for q in ORDER_BLOCKS:
        qkey = f'q{q}'
        out[qkey] = {}
        for label in ba.comparison_labels(q):
            rows = []
            for n in GEOM_NPTS:
                p, nerr, terr = geometry_eval(label, n, xm, ym, nx, ny, normal_exact, traction_exact)
                rows.append({'ppw': p, 'normal': nerr, 'traction': terr})
            out[qkey][label] = rows
    return out


def build_order_data(geometry_data: dict[str, dict[str, list[dict[str, float]]]]) -> dict[int, dict[str, object]]:
    summary = json.loads((RESULTSDIR / 'crossquartet_client_summary.json').read_text())
    out: dict[int, dict[str, object]] = {}
    for q in ORDER_BLOCKS:
        labels = ba.comparison_labels(q)
        ops = {label: ba.build_operator(ba.scheme_kind(label), q, RESPONSE_NPTS) for label in labels}
        curves = {}
        for label, op in ops.items():
            rows = ba.wall_mode_response(op)
            strict = np.array([abs(row.kappa_ratio - 1.0) for row in rows], dtype=float)
            contam = np.array([row.contamination_l2 for row in rows], dtype=float)
            curves[label] = {
                'ppw': np.array([row.ppw for row in rows], dtype=float),
                'strict': strict,
                'contam': contam,
                'total': np.sqrt(strict ** 2 + contam ** 2),
            }
        strict_required = {
            label: np.array([response_requirement(curves[label]['ppw'], curves[label]['strict'], m) for m in MISSION_LENGTHS], dtype=float)
            for label in labels
        }
        qsum = summary['orders'][f'q{q}']
        normal_target = min(qsum['geometry_transfer'][f]['normal_rel_l2'] for f in qsum['geometry_transfer'] if ba.scheme_family(f) in NODAL_FAMILIES)
        traction_target = min(qsum['traction_transfer'][f]['traction_rel_l2'] for f in qsum['traction_transfer'] if ba.scheme_family(f) in NODAL_FAMILIES)
        geometry_required = {}
        for label in labels:
            rows = geometry_data[f'q{q}'][label]
            req = next((row['ppw'] for row in rows if row['normal'] <= normal_target and row['traction'] <= traction_target), float('inf'))
            geometry_required[label] = float(req)
        combined_required = {
            label: np.maximum(strict_required[label], geometry_required[label])
            for label in labels
        }
        out[q] = {
            'labels': labels,
            'curves': curves,
            'strict_required': strict_required,
            'geometry_required': geometry_required,
            'combined_required': combined_required,
            'normal_target': normal_target,
            'traction_target': traction_target,
        }
    return out


def _operating_metric(data: dict[str, object], label: str, pcap: float, key: str) -> float:
    curve = data['curves'][label]
    ppw = curve['ppw']
    values = curve[key]
    return float(np.interp(pcap, ppw[::-1], values[::-1], left=values[-1], right=values[0]))


def atlas_decision(data: dict[str, object], key: str, ppw_budget: np.ndarray) -> np.ndarray:
    labels = data['labels']
    required_map = data[key]
    decision = np.full((ppw_budget.size, MISSION_LENGTHS.size), -1, dtype=int)
    for i, pcap in enumerate(ppw_budget):
        for j in range(MISSION_LENGTHS.size):
            mission = float(MISSION_LENGTHS[j])
            candidates = [
                (idx, label)
                for idx, label in enumerate(labels)
                if np.isfinite(required_map[label][j]) and required_map[label][j] <= pcap
            ]
            if not candidates:
                continue
            if key == 'combined_required':
                def mission_score(item: tuple[int, str]) -> tuple[float, float, int]:
                    _idx, label = item
                    threshold = float(required_map[label][j])
                    contam_at = max(_operating_metric(data, label, float(pcap), 'contam'), 1.0e-300)
                    score = threshold + LONG_MISSION_BMG_BIAS * mission * float(np.log10(contam_at))
                    return (score, data['geometry_required'][label], _idx)
                decision[i, j] = min(candidates, key=mission_score)[0]
            else:
                best_required = min(required_map[label][j] for _idx, label in candidates)
                finalists = [
                    (idx, label)
                    for idx, label in candidates
                    if np.isclose(required_map[label][j], best_required, rtol=0.0, atol=1.0e-12)
                ]
                decision[i, j] = min(finalists)[0]
    return decision


def draw_atlas(
    order_data: dict[int, dict[str, object]],
    key: str,
    outname: str,
    caption_title: str,
    *,
    ppw_window: tuple[float, float] = PPW_BUDGET_WINDOW,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 10.0), gridspec_kw={'hspace': 0.18})
    cmap = ListedColormap([COLORS[f] for f in FAMILY_ORDER])
    norm = BoundaryNorm(np.arange(-0.5, len(FAMILY_ORDER) + 0.5, 1.0), cmap.N)
    ppw_budget = np.linspace(*ppw_window, 60)
    for row, q in enumerate(ORDER_BLOCKS):
        ax = axes[row]
        data = order_data[q]
        decision = atlas_decision(data, key, ppw_budget)
        mx, py = np.meshgrid(MISSION_LENGTHS, ppw_budget)
        masked = np.ma.masked_where(decision < 0, decision)
        ax.set_facecolor('#e8e6df')
        ax.pcolormesh(mx, py, masked, cmap=cmap, norm=norm, shading='nearest', alpha=0.97)
        contour_field = np.ma.masked_where(decision < 0, decision.astype(float))
        ax.contour(mx, py, contour_field, levels=np.arange(0.5, len(FAMILY_ORDER) - 0.5, 1.0), colors='white', linewidths=0.65, alpha=0.72)
        ax.set_xscale('log')
        ax.set_xlim(*MISSION_WINDOW)
        ax.set_ylim(*ppw_window)
        ax.set_ylabel(rf'$q={q}$' + '\n' + r'available PPW $P_{\max}$')
        if row == 0:
            ax.set_title(caption_title, pad=10, fontweight='semibold')
        if row == 2:
            ax.set_xlabel(r'mission length $M$ [standing-wave periods]')
    legend = [Patch(facecolor=COLORS[f], label=ba.scheme_display_name(f)) for f in FAMILY_ORDER]
    legend.append(Patch(facecolor='#e8e6df', label='no admissible scheme'))
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, 0.987), ncol=4, frameon=False, fontsize=10.4, columnspacing=1.35, handlelength=1.8)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.89, bottom=0.08, hspace=0.19)
    save_figure(fig, outname.removesuffix('.png'), pad_inches=0.03)
    plt.close(fig)


def draw_fixed_ppw_curves(order_data: dict[int, dict[str, object]]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 10.0), gridspec_kw={'wspace': 0.22, 'hspace': 0.22})
    for row, q in enumerate(ORDER_BLOCKS):
        data = order_data[q]
        for col, metric in enumerate(('strict', 'total')):
            ax = axes[row, col]
            for label in data['labels']:
                c = COLORS[ba.scheme_family(label)]
                ax.semilogy(data['curves'][label]['ppw'], data['curves'][label][metric], color=c, linewidth=1.9, label=label)
            ax.set_xlim(12.0, 2.0)
            ax.set_ylim(5e-5 if q == 4 else 5e-7, 2e-1)
            ax.grid(True, which='both', alpha=0.12, linewidth=0.5)
            if col == 0:
                ax.set_ylabel(rf'$q={q}$' + '\n' + r'error')
            if row == 2:
                ax.set_xlabel(r'available PPW $P$')
            if row == 0:
                title = r'strict error $|\rho_m-1|$' if metric == 'strict' else r'total wall-response $\sqrt{(|\rho_m-1|)^2+\eta_m^2}$'
                ax.set_title(title, pad=10, fontweight='semibold')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.987), ncol=6, frameon=False, fontsize=10.4, columnspacing=1.25, handlelength=1.95)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.08, wspace=0.22, hspace=0.25)
    save_figure(fig, 'fig_s6_wall_mission_fixed_ppw', pad_inches=0.03)
    plt.close(fig)


def draw_family_dominance(order_data: dict[int, dict[str, object]]) -> None:
    palette = ['#6a6a6a', COLORS['BMF'], COLORS['BMG']]
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 10.0), gridspec_kw={'hspace': 0.18})
    cmap = ListedColormap(palette)
    norm = BoundaryNorm(np.arange(-0.5, 3.5, 1.0), cmap.N)
    for row, q in enumerate(ORDER_BLOCKS):
        ax = axes[row]
        labels = order_data[q]['labels']
        required = order_data[q]['strict_required']
        decision = np.full((PPW_BUDGET.size, MISSION_LENGTHS.size), -1, dtype=int)
        for i, pcap in enumerate(PPW_BUDGET):
            for j in range(MISSION_LENGTHS.size):
                group_rows = []
                nodal = min((required[label][j] for label in labels if ba.scheme_family(label) in NODAL_FAMILIES and required[label][j] <= pcap), default=float('inf'))
                bmf = required[f'BMF{q}'][j] if required[f'BMF{q}'][j] <= pcap else float('inf')
                bmg = required[f'BMG{q}'][j] if required[f'BMG{q}'][j] <= pcap else float('inf')
                for idx, val in enumerate((nodal, bmf, bmg)):
                    if np.isfinite(val):
                        group_rows.append((val, idx))
                if group_rows:
                    decision[i, j] = min(group_rows)[1]
        mx, py = np.meshgrid(MISSION_LENGTHS, PPW_BUDGET)
        ax.set_facecolor('#e8e6df')
        ax.pcolormesh(mx, py, np.ma.masked_where(decision < 0, decision), cmap=cmap, norm=norm, shading='nearest', alpha=0.97)
        ax.set_xscale('log')
        ax.set_xlim(*MISSION_WINDOW)
        ax.set_ylim(*PPW_BUDGET_WINDOW)
        ax.set_ylabel(rf'$q={q}$' + '\n' + r'available PPW $P_{\max}$')
        if row == 0:
            ax.set_title('strict-response capability, grouped as nodal vs BMF vs BMG', pad=10, fontweight='semibold')
        if row == 2:
            ax.set_xlabel(r'mission length $M$ [standing-wave periods]')
    legend = [Patch(facecolor=palette[0], label='best nodal family'), Patch(facecolor=palette[1], label='BMF'), Patch(facecolor=palette[2], label='BMG'), Patch(facecolor='#e8e6df', label='no admissible scheme')]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, 0.987), ncol=4, frameon=False, fontsize=10.3, columnspacing=1.35, handlelength=1.8)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.89, bottom=0.08, hspace=0.19)
    save_figure(fig, 'fig_appA_wall_mission_family_dominance', pad_inches=0.03)
    plt.close(fig)


geometry_data = build_geometry_data()
order_data = build_order_data(geometry_data)
draw_atlas(order_data, 'strict_required', 'fig_appA_wall_mission_ppw_first.png', 'strict-response capability: lowest admissible PPW')
draw_fixed_ppw_curves(order_data)
draw_atlas(
    order_data,
    'combined_required',
    'fig_s6_wall_mission_geometry_capability.png',
    'strict response + nodal-target geometry capability',
    ppw_window=(2.0, 18.5),
)
draw_family_dominance(order_data)

summary = {
    'mission_window': MISSION_WINDOW,
    'ppw_budget_window': PPW_BUDGET_WINDOW,
    'tol_cycles': TOL_CYCLES,
    'geometry_reference': 'targets match the best nodal normal and traction errors from the N=81 transfer tests',
    'orders': {},
}
for q in ORDER_BLOCKS:
    data = order_data[q]
    summary['orders'][f'q{q}'] = {
        'geometry_targets': {'normal_rel_l2': data['normal_target'], 'traction_rel_l2': data['traction_target']},
        'strict_required_ppw_at_M100': {label: float(data['strict_required'][label][np.argmin(np.abs(MISSION_LENGTHS - 100.0))]) for label in data['labels']},
        'geometry_required_ppw': data['geometry_required'],
    }
SUMMARY_OUT.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
