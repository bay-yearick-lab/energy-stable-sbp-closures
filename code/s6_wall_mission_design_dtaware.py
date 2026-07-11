#!/usr/bin/env python3
"""Dt-aware mission atlas: replace the shared-CFL recurring-work proxy
F_D P^{d+1} with a per-family measured-dt recurring-work model.

`s6_wall_mission_design.py` (Figure 41 / \\ref{fig:mission} in the manuscript)
ranks six families by

    recurring_work = work_per_point * P^(d+1),

where P is the required points-per-wavelength and the extra factor of P
beyond the P^d grid-point count stands in for "one shared explicit time step
per unit physical time, dt ~ h for every family alike." That shared-CFL
assumption prices no family's actual stability limit. This script replaces
it with measured limits from `code/spectral_radius_table.py`
(`code/results/spectral_radius_table.json`), which computes the exact RK4
stability boundary of the operators each family actually marches:

  - hyperbolic missions: the coupled pressure-velocity wave pair of
    `simulate_reflected_acoustics` (staggered d01/d10 pair for BMF), where
    all four SBP-identity carriers (SBP, SBP6-BO, BMF, BMG) turn out to be
    exactly skew-symmetric (spectral abscissa ~1e-14) and the order-tuned
    and dispersion-tuned closures (E, C, DRP) carry a positive abscissa;
  - parabolic missions: the 2D tensor Laplacian of the heat-cavity driver
    (interior-restricted, pairwise-sum spectrum).

Three atlases are drawn:

  1. hyperbolic, explicit RK4:  work = F_D * P^d * P^1 / r_wave
  2. parabolic, explicit RK4:   work = F_D * P^d * P^2 / r_diff
     (the paper's heat cavity advances diffusion explicitly, so this is the
     self-consistent default)
  3. parabolic, implicit diffusion: the practitioner pairing (and the choice
     of this paper's own turbulent-channel solver). Every family's step is
     then accuracy-limited, not stability-limited; dt ~ h with one shared
     constant gives steps/time ~ P for every family, and the family
     differences move entirely into the per-step algebra:
         work = F_impl * P^d * P^1,
     where F_impl is an honest banded-solve estimate measured from each
     family's own assembled matrices (see `implicit_flops_per_point`).

r_wave / r_diff are the measured dt_max ratios against E at matched order
(grid-independent between N=65 and N=129; the N=129 entries are used to
match the mission scripts' NPTS=129). New JSON and figure filenames; the
original `wall_mission_design_summary.json` / `fig_s6_wall_mission_design.*`
are not touched by this script.
"""
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

ensure_generated_dirs()

ORDER_BLOCKS = (4, 6, 8)
FAMILY_ORDER = ba.family_order()
FAMILY_COLORS = ba.DEFAULT_COLORS
NPTS = 129
MISSION_WINDOW = (10.0, 1000.0)
PPW_BUDGET_WINDOW = (2.0, 12.0)
TOL_CYCLES = 0.25
NNZ_TOL = 1.0e-12

SPECTRAL_TABLE = json.loads((RESULTSDIR / 'spectral_radius_table.json').read_text())
_SPECTRAL_NPTS_KEY = str(NPTS)

# Three-tier step data (2026-07-10 correction): tier-B asymptotic RK4 steps
# from the validated certificate. A family with a positive spectral abscissa
# has NO asymptotically stable explicit step; on hyperbolic (long-time
# coherence) missions such a family is inadmissible outright, because the
# inherent e^{alpha t} growth destroys the wall response the mission
# criterion protects (the long-time cliff measures exactly this). Ratios are
# normalized within each order block against the largest stable step there.
TIER_TABLE = {r['label']: r for r in json.loads(
    (RESULTSDIR / 'time_step_cost.json').read_text())['families']}


def dt_ratio(label: str, mission_class: str) -> float | None:
    """Tier-aware step factor. Hyperbolic: tier-B step normalized by the
    block's largest stable step; None when no stable step exists
    (inadmissible). Parabolic: every family is stable and the diffusive
    tier-B ratio against E at matched order is used, as before."""
    if mission_class == 'hyperbolic':
        rec = TIER_TABLE[label]
        dtb = rec['tierB_dt_asymptotic']
        if dtb is None:
            return None
        q = ''.join(ch for ch in label if ch.isdigit())
        block = [r['tierB_dt_asymptotic'] for lab, r in TIER_TABLE.items()
                 if lab.endswith(q) and r['tierB_dt_asymptotic'] is not None]
        return float(dtb / max(block))
    entry = SPECTRAL_TABLE['families'][label][_SPECTRAL_NPTS_KEY]
    return float(entry['diffusion']['dt_max_ratio_vs_E'])


def _half_bandwidth(mat: np.ndarray) -> int:
    rows, cols = np.where(np.abs(mat) > NNZ_TOL)
    return int(max(np.max(rows - cols), np.max(cols - rows)))


def _mean_apply_flops_per_point(mat: np.ndarray) -> float:
    nnz_rows = np.count_nonzero(np.abs(mat) > NNZ_TOL, axis=1)
    return float(np.mean(2 * nnz_rows - 1))


def implicit_flops_per_point(label: str, npts: int) -> tuple[float, dict[str, object]]:
    """Honest banded-solve estimate of implicit-diffusion work per point per
    direction per step (Crank-Nicolson-style: one RHS apply plus one banded
    Helmholtz solve; banded solve priced at the manuscript's own convention
    4*b*n flops for half-bandwidth b, Table 14 / Section E.1).

    - E/DRP: the composed D@D operator is banded (D is banded), so the
      Helmholtz matrix (I - sigma*D2) is banded with the same half-bandwidth;
      cost = apply(D2) + 4*b(D2).
    - SBP: same with the native SBP D2 (banded).
    - C: the compact Pade second-derivative pair (A2 f'' = B2 f, compact-baseline) keeps
      a tridiagonal LHS and a five-point RHS at every order of the family;
      premultiplying the Helmholtz system by A2 gives the banded system
      (A2 - sigma*B2) with half-bandwidth max(b_A2, b_B2). Measured from the
      assembled q=6 matrices (`compact_baseline_c6_d2_matrices`) and reused for q=4,8,
      whose Pade D2 variants share the identical band structure.
    - BMF/BMG: reduce-solve-expand through the clamped spline space: one
      banded solve through B (nodal values -> coefficients), one banded
      coefficient-space Helmholtz solve through (M - sigma*K2) with M the
      spline mass and K2 the stiffness (both half-bandwidth p), one RHS
      apply (M + sigma*K2 row), one B apply back to nodes. Both spline
      branches share the clamped-space band structure, so the same measured
      estimate is used for both (documented approximation; BMF's staggered
      metric forms have the same bandwidths as the primal clamped space).
    """
    fam = ba.scheme_family(label)
    q = ba.scheme_order(label)
    if fam in ('E', 'DRP', 'SBP'):
        op = ba.build_operator(ba.scheme_kind(label), q, npts)
        if fam == 'SBP':
            D2 = op.second_derivative_matrix()
        else:
            D2 = op.matrix @ op.matrix
        b = _half_bandwidth(D2)
        apply_f = _mean_apply_flops_per_point(D2)
        total = apply_f + 4.0 * b
        return total, {'apply': apply_f, 'solve': 4.0 * b, 'half_bandwidth': float(b)}
    if fam == 'C':
        _x, a2, b2 = ba.compact_baseline_c6_d2_matrices(npts)
        b_lhs = max(_half_bandwidth(a2), _half_bandwidth(b2))
        apply_f = _mean_apply_flops_per_point(b2)
        total = apply_f + 4.0 * b_lhs
        return total, {'apply': apply_f, 'solve': 4.0 * b_lhs, 'half_bandwidth': float(b_lhs),
                       'note': 'Pade D2 band structure measured at q=6, reused for q=4,8'}
    if fam in ('BMF', 'BMG'):
        op = ba.build_operator(ba.scheme_kind(label), q, npts)
        bmat = op._bmat  # clamped B-spline evaluation matrix (banded)
        mass, _adv, _weak, stiff = ba.bspline_mass_advection_second_and_stiffness(npts, q)
        b_B = _half_bandwidth(bmat)
        b_MK = max(_half_bandwidth(mass), _half_bandwidth(stiff))
        reduce_f = 4.0 * b_B
        helmholtz_f = 4.0 * b_MK
        apply_f = max(_mean_apply_flops_per_point(mass), _mean_apply_flops_per_point(stiff))
        expand_f = _mean_apply_flops_per_point(bmat)
        total = reduce_f + helmholtz_f + apply_f + expand_f
        return total, {'reduce_solve': reduce_f, 'helmholtz_solve': helmholtz_f,
                       'rhs_apply': apply_f, 'expand_apply': expand_f,
                       'b_B': float(b_B), 'b_MK': float(b_MK)}
    raise ValueError(f'unknown family for implicit model: {label}')


def scheme_label(family: str, q: int) -> str:
    return f'{family}{q}'


def response_requirement(ppw: np.ndarray, response_error: np.ndarray, mission_length: float, tol_cycles: float) -> float:
    tol = float(tol_cycles) / float(mission_length)
    idx = np.where(response_error <= tol)[0]
    return float(ppw[idx[-1]]) if idx.size else float('inf')


def build_order_data(q: int) -> dict[str, object]:
    labels = [scheme_label(family, q) for family in FAMILY_ORDER]
    ops = {label: ba.build_operator(ba.scheme_kind(label), q, NPTS) for label in labels}
    work_nodal = {label: float(op.work_model.total_flops / op.x.size) for label, op in ops.items()}

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

    implicit_work = {}
    implicit_detail = {}
    for label in labels:
        total, detail = implicit_flops_per_point(label, NPTS)
        implicit_work[label] = total
        implicit_detail[label] = detail

    return {
        'labels': labels,
        'work_nodal': work_nodal,
        'curves': curves,
        'mission_lengths': mission_lengths,
        'ppw_budget': ppw_budget,
        'required_ppw': required_ppw,
        'dt_ratio_hyperbolic': {label: dt_ratio(label, 'hyperbolic') for label in labels},
        'dt_ratio_parabolic': {label: dt_ratio(label, 'parabolic') for label in labels},
        'implicit_work': implicit_work,
        'implicit_detail': implicit_detail,
    }


order_data = {q: build_order_data(q) for q in ORDER_BLOCKS}


def dtaware_work(work_per_point: float, req_ppw: float, dim: int, dt_rat: float, k: int) -> float:
    """recurring work per physical time = F * P^d * P^k / dt_ratio(family)."""
    return work_per_point * (req_ppw ** dim) * (req_ppw ** k) / dt_rat


def piecewise_winner(
    mission_grid: np.ndarray,
    labels: list[str],
    required_ppw: dict[str, np.ndarray],
    work_map: dict[str, float],
    dt_ratio_map: dict[str, float],
    dim: int,
    k: int,
) -> list[dict[str, float | str]]:
    winners: list[str] = []
    for j in range(mission_grid.size):
        candidates: list[tuple[float, str]] = []
        for label in labels:
            req = required_ppw[label][j]
            # a family without an asymptotically stable explicit step
            # (dt_ratio None) is inadmissible on this mission class
            if np.isfinite(req) and dt_ratio_map[label] is not None:
                candidates.append((dtaware_work(work_map[label], req, dim, dt_ratio_map[label], k), label))
        winners.append(min(candidates)[1] if candidates else 'none')

    out: list[dict[str, float | str]] = [{'mission_length': float(mission_grid[0]), 'winner': winners[0]}]
    prev = winners[0]
    for mission_length, winner in zip(mission_grid[1:], winners[1:], strict=True):
        if winner != prev:
            out.append({'mission_length': float(mission_length), 'winner': winner})
            prev = winner
    return out


def render_atlas(variant: str, outname: str, title: str) -> dict[str, object]:
    """variant in {'hyperbolic', 'parabolic', 'parabolic_implicit'}."""
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

    def maps_for(data):
        if variant == 'hyperbolic':
            return data['work_nodal'], data['dt_ratio_hyperbolic'], 1
        if variant == 'parabolic':
            return data['work_nodal'], data['dt_ratio_parabolic'], 2
        # parabolic_implicit: accuracy-limited dt ~ h shared by all families
        # (ratio 1), steps/time ~ P, per-step work from banded implicit solves.
        return data['implicit_work'], {label: 1.0 for label in data['labels']}, 1

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
        work_map, dt_ratio_map, k = maps_for(data)
        for col, dim in enumerate((1, 2, 3), start=0):
            ax = axes[row, col]
            decision = np.full((ppw_budget.size, mission_lengths.size), -1, dtype=int)
            for i, p_cap in enumerate(ppw_budget):
                for j, _mission in enumerate(mission_lengths):
                    candidates: list[tuple[float, int]] = []
                    for idx, label in enumerate(labels):
                        req = required_ppw[label][j]
                        if (np.isfinite(req) and req <= p_cap
                                and dt_ratio_map[label] is not None):
                            recurring_work = dtaware_work(work_map[label], req, dim, dt_ratio_map[label], k)
                            candidates.append((recurring_work, idx))
                    if candidates:
                        decision[i, j] = min(candidates)[1]

            ax.set_facecolor('#e8e6df')
            mission_mesh, budget_mesh = np.meshgrid(mission_lengths, ppw_budget)
            masked = np.ma.masked_where(decision < 0, decision)
            ax.pcolormesh(mission_mesh, budget_mesh, masked, cmap=cmap, norm=norm, shading='nearest', alpha=0.97)
            contour_field = np.ma.masked_where(decision < 0, decision.astype(float))
            ax.contour(
                mission_mesh, budget_mesh, contour_field,
                levels=np.arange(0.5, len(FAMILY_ORDER) - 0.5, 1.0),
                colors='white', linewidths=0.65, alpha=0.72,
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
    fig.suptitle(title, y=0.99, fontsize=12.6, fontweight='semibold')
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
        handlelength=1.9,
        columnspacing=1.45,
        fontsize=11.4,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.83, bottom=0.09, wspace=0.14, hspace=0.30)
    save_figure(fig, outname, pad_inches=0.03)
    plt.close(fig)

    work_proxy_desc = {
        'hyperbolic': 'F_D * P^d * P / r_wave, r_wave = measured coupled-wave-pair RK4 dt_max ratio vs E',
        'parabolic': 'F_D * P^d * P^2 / r_diff, r_diff = measured 2D-tensor-Laplacian RK4 dt_max ratio vs E (explicit diffusion)',
        'parabolic_implicit': 'F_impl * P^d * P, dt accuracy-limited (~h, shared constant, all families); '
                              'F_impl = measured banded-solve implicit work per point (see implicit_detail)',
    }[variant]

    summary: dict[str, object] = {
        'budget': {
            'strict_response_metric': '|rho_m - 1|',
            'tol_cycles': TOL_CYCLES,
            'mission_unit': 'standing-wave periods',
            'mission_class': variant,
            'work_proxy': work_proxy_desc,
        },
        'orders': {},
    }
    for q in ORDER_BLOCKS:
        data = order_data[q]
        labels = data['labels']
        mission_lengths = data['mission_lengths']
        required_ppw = data['required_ppw']
        work_map, dt_ratio_map, k = maps_for(data)
        block: dict[str, object] = {
            'labels': labels,
            'work_per_point': work_map,
            'dt_max_ratio_vs_E': dt_ratio_map,
            'required_ppw_snapshots': {
                'M=10': {label: float(required_ppw[label][0]) for label in labels},
                'M=100': {label: float(required_ppw[label][np.argmin(np.abs(mission_lengths - 100.0))]) for label in labels},
                'M=1000': {label: float(required_ppw[label][-1]) for label in labels},
            },
            'winner_crossovers_dtaware': {
                f'd{dim}': piecewise_winner(mission_lengths, labels, required_ppw, work_map, dt_ratio_map, dim, k)
                for dim in (1, 2, 3)
            },
        }
        if variant == 'parabolic_implicit':
            block['implicit_detail'] = data['implicit_detail']
        summary['orders'][f'q{q}'] = block
    return summary


hyperbolic_summary = render_atlas('hyperbolic', 'fig_s6_mission_atlas_dtaware_hyperbolic',
                                   'dt-aware recurring work (hyperbolic mission, coupled wave pair, explicit RK4)')
parabolic_summary = render_atlas('parabolic', 'fig_s6_mission_atlas_dtaware_parabolic',
                                  'dt-aware recurring work (parabolic mission, explicit diffusion, dt ~ $h_{\\min}^2$)')
implicit_summary = render_atlas('parabolic_implicit', 'fig_s6_mission_atlas_dtaware_parabolic_implicit',
                                 'dt-aware recurring work (parabolic mission, implicit diffusion, dt ~ h)')

combined = {
    'hyperbolic': hyperbolic_summary,
    'parabolic': parabolic_summary,
    'parabolic_implicit': implicit_summary,
    'note': (
        'dt-aware companion to wall_mission_design_summary.json (Figure 41 / \\ref{fig:mission}). '
        'dt ratios from code/results/spectral_radius_table.json: coupled acoustic wave pair for the '
        'hyperbolic class, interior-restricted 2D tensor Laplacian for the parabolic class, exact RK4 '
        'stability bisection in both cases. The parabolic_implicit variant prices the practitioner '
        'pairing (implicit diffusion; the choice of the paper\'s own turbulent-channel solver): '
        'accuracy-limited dt shared by every family, banded-solve work measured per family.'
    ),
}

OUT = RESULTSDIR / 'wall_mission_design_dtaware_summary.json'
OUT.write_text(json.dumps(combined, indent=2))
print(f'wrote {OUT}')
print(json.dumps({
    'hyperbolic_q6_d2': hyperbolic_summary['orders']['q6']['winner_crossovers_dtaware']['d2'],
    'parabolic_q6_d2': parabolic_summary['orders']['q6']['winner_crossovers_dtaware']['d2'],
    'parabolic_implicit_q6_d2': implicit_summary['orders']['q6']['winner_crossovers_dtaware']['d2'],
    'implicit_work_q6': implicit_summary['orders']['q6']['work_per_point'],
}, indent=2))
