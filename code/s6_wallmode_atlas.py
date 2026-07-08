#!/usr/bin/env python3
"""Standing-mode atlas for the corrected six-family quartets.

Produces two figures from a single sweep:

* ``fig_s6_wallmode_response.pdf`` is the single-row response view used by
  Section 6 sub-capability (1) target wall response.
* ``fig_s6_wallmode_contamination.pdf`` is the single-row contamination
  view used by sub-capability (1) low wall contamination.

Both figures use the project's cream PANEL_BG, dark ACCENT spines,
and family DEFAULT_COLORS palette so the BMF and BMG curves sit visibly
below the finite-difference families across q in {4, 6, 8}.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

import boundary_spline_analysis as ba
from project_paths import RESULTSDIR, ensure_generated_dirs, save_figure


OUT = RESULTSDIR / 'wallmode_atlas_summary.json'
ensure_generated_dirs()

_PANEL_BG = (0.987, 0.985, 0.978)
_ACCENT = '#2F4858'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10.3,
    'axes.labelsize': 11.0,
    'axes.titlesize': 11.6,
    'legend.fontsize': 10.6,
    'xtick.labelsize': 9.3,
    'ytick.labelsize': 9.3,
    'figure.dpi': 180,
    'savefig.dpi': 300,
})

ORDERS = (4, 6, 8)
NPTS = 129
ALPHA_F = 0.49

LABEL_NAMES = {
    'E': r'$E_q$',
    'C': r'$C_q$',
    'SBP': r'$\mathrm{SBP}_q$',
    'DRP': r'$\mathrm{DRP}_q$',
    'BMF': r'$\mathrm{BMF}_q$',
    'BMG': r'$\mathrm{BMG}_q$',
}
EXTRA_FAMILIES_Q6 = ['C6F', 'SBP6-BO']
EXTRA_COLORS = {'C6F': '#e08214', 'SBP6-BO': '#66c2a5'}
EXTRA_LABEL_NAMES = {'C6F': 'C6F', 'SBP6-BO': 'SBP6-BO'}


def threshold(ppw: np.ndarray, metric: np.ndarray, tol: float) -> float:
    idx = np.where(metric <= tol)[0]
    return float(ppw[idx[-1]]) if idx.size else float('nan')


def wall_mode_response_filtered(op, alpha_f: float = ALPHA_F):
    """Wall-mode response for C6F: the operator-level object composes the
    filter AFTER the derivative, F @ D, not D alone and not F alone.

    Rationale (documented per the brief): this diagnostic measures the
    per-mode response of "the thing that actually enters the RHS/next state"
    in the paper's RK4 usage. In `CompactFilteredFD`'s actual time-marching
    use (see boundary_spline_analysis.py and the reflected-wave/long-time
    drivers), the filter is applied to the STATE once per completed step,
    not to the derivative operator's raw output mid-step; the two honest
    candidates are therefore (i) F@D, treating the filter as if it acted
    immediately on the differentiated signal (an upper bound on the
    single-application filtering cost visible at the operator level), or
    (ii) the full nonlinear effective one-step advance operator (which
    depends on dt/cfl and mixes p and u, and is not a single-mode-diagonal
    object suitable for this per-mode atlas). F@D is chosen because it is
    dt-independent, directly comparable to the other five families' bare D
    (also dt-independent), and gives the more conservative (harsher) picture
    of what the filter costs the standing-mode response, rather than an
    optimistic one computed over many diluted RK sub-steps.
    """
    x = op.x
    npts = x.size
    metric = ba.nodal_trapezoidal_weights(x)
    h = ba.reference_spacing(x)
    filt = ba.visbal_gaitonde_filter_matrix(npts, alpha_f)
    max_mode = min(npts - 2, int(np.floor((1.0 - 1.0e-12) / h)))
    out = []
    for mode in range(1, max_mode + 1):
        s, c, ds = ba.wall_mode_values(x, mode)
        num = filt @ op.differentiate(s)
        denom = ba._quadratic_form(metric, c)
        kappa = ba._quadratic_form(metric, num, c) / denom
        contamination = float(np.sqrt(ba._quadratic_form(metric, num - kappa * c)) / (mode * np.pi * np.sqrt(denom)))
        rel_l2 = float(np.sqrt(ba._quadratic_form(metric, num - ds) / ba._quadratic_form(metric, ds)))
        ppw = float(2.0 / (mode * h))
        out.append(ba.WallModeResponse(mode=mode, ppw=ppw, kappa_ratio=kappa / (mode * np.pi),
                                        contamination_l2=contamination, rel_max=float('nan'), rel_l2=rel_l2))
    return out


def collect_curves() -> tuple[dict, dict]:
    """Run the wall-mode response sweep once across all q and families."""
    summary: dict[str, dict[str, dict[str, float]]] = {}
    curves: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for q in ORDERS:
        summary[f'q{q}'] = {}
        curves[q] = {}
        for label, kind, degree in ba.comparison_block(q):
            op = ba.build_operator(kind, degree, NPTS)
            resp = ba.wall_mode_response(op)
            ppw = np.array([r.ppw for r in resp], dtype=float)
            response = np.array([abs(r.kappa_ratio - 1.0) for r in resp], dtype=float)
            total = np.array([r.rel_l2 for r in resp], dtype=float)
            contamination = np.array([r.contamination_l2 for r in resp], dtype=float)
            family = ba.scheme_family(label)
            curves[q][family] = {
                'ppw': ppw, 'response': response,
                'total': total, 'contamination': contamination,
            }
            metrics = {
                'P1_response': threshold(ppw, response, 0.01),
                'P5_response': threshold(ppw, response, 0.05),
                'P1_total': threshold(ppw, total, 0.01),
                'P5_total': threshold(ppw, total, 0.05),
                'P1_contamination': threshold(ppw, contamination, 0.01),
                'P5_contamination': threshold(ppw, contamination, 0.05),
                'work_per_point_nodal': float(op.work_model.total_flops / op.x.size),
            }
            if label.startswith('BMG'):
                metrics['work_per_point_coefficient'] = float(
                    op.work_model_coefficient.total_flops / op.x.size)
            summary[f'q{q}'][label] = metrics

    # Extra q=6-only competitors (C6F via F@D, SBP6-BO via its own D/metric).
    summary['q6_extra'] = {}
    curves[6]['_extra'] = {}
    for name in EXTRA_FAMILIES_Q6:
        op = ba.build_operator_extended(name, 6, NPTS)
        resp = wall_mode_response_filtered(op) if name == 'C6F' else ba.wall_mode_response(op)
        ppw = np.array([r.ppw for r in resp], dtype=float)
        response = np.array([abs(r.kappa_ratio - 1.0) for r in resp], dtype=float)
        total = np.array([r.rel_l2 for r in resp], dtype=float)
        contamination = np.array([r.contamination_l2 for r in resp], dtype=float)
        curves[6]['_extra'][name] = {'ppw': ppw, 'response': response, 'total': total, 'contamination': contamination}
        summary['q6_extra'][name] = {
            'P1_response': threshold(ppw, response, 0.01),
            'P5_response': threshold(ppw, response, 0.05),
            'P1_total': threshold(ppw, total, 0.01),
            'P5_total': threshold(ppw, total, 0.05),
            'P1_contamination': threshold(ppw, contamination, 0.01),
            'P5_contamination': threshold(ppw, contamination, 0.05),
            'work_per_point_nodal': float(op.work_model.total_flops / op.x.size),
            'note': 'C6F composed as F@D (filter applied to the differentiated signal); see wall_mode_response_filtered docstring' if name == 'C6F' else '',
        }
    return summary, curves


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(_PANEL_BG)
    ax.set_yscale('log')
    ax.set_xlim(2.0, 12.5)
    ax.set_ylim(7.0e-4, 2.0)
    ax.grid(True, which='major', alpha=0.18, lw=0.5)
    ax.grid(True, which='minor', alpha=0.08, lw=0.4)
    for spine in ax.spines.values():
        spine.set_color(_ACCENT)
        spine.set_linewidth(0.55)
    ax.tick_params(colors=_ACCENT)
    ax.axhline(0.01, color=_ACCENT, lw=0.8, ls='--', alpha=0.45)
    ax.axhline(0.05, color=_ACCENT, lw=0.8, ls=':', alpha=0.45)
    ax.text(0.05, 0.93, r'$1\%$', transform=ax.transAxes,
            fontsize=8.4, color=_ACCENT, alpha=0.7)
    ax.text(0.05, 0.78, r'$5\%$', transform=ax.transAxes,
            fontsize=8.4, color=_ACCENT, alpha=0.55)


def _plot_curves(ax: plt.Axes, q: int, curves: dict, key: str) -> None:
    for family in ('E', 'C', 'SBP', 'DRP', 'BMF', 'BMG'):
        if family not in curves[q]:
            continue
        data = curves[q][family]
        ax.plot(
            data['ppw'], data[key],
            color=ba.DEFAULT_COLORS[family],
            ls=ba.DEFAULT_LINESTYLES[family],
            marker=ba.DEFAULT_MARKERS[family],
            lw=2.0, ms=4.4, mec='white', mew=0.7, markevery=12,
            label=LABEL_NAMES[family],
            path_effects=[pe.Stroke(linewidth=2.7, foreground='white',
                                     alpha=0.55), pe.Normal()],
        )
    if q == 6 and '_extra' in curves[q]:
        for name in EXTRA_FAMILIES_Q6:
            data = curves[q]['_extra'][name]
            ax.plot(
                data['ppw'], data[key],
                color=EXTRA_COLORS[name], ls='--', marker='v',
                lw=1.8, ms=4.0, mec='white', mew=0.6, markevery=12,
                label=EXTRA_LABEL_NAMES[name],
                path_effects=[pe.Stroke(linewidth=2.5, foreground='white', alpha=0.55), pe.Normal()],
            )


def _shared_legend(fig, axes_list, *, bbox=(0.5, 0.978)) -> None:
    # Pool handles/labels across all panels (not just the first) and
    # de-duplicate by label, preserving first-seen order. This is required
    # because the q=6-only strengthened competitors (C6F, SBP6-BO) are only
    # plotted on the q=6 axis, so a legend built from axes[0] alone would
    # silently drop them.
    seen: dict = {}
    for ax in axes_list:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
    labels = list(seen.keys())
    handles = [seen[l] for l in labels]
    fig.legend(
        handles, labels, ncol=len(labels), loc='upper center',
        bbox_to_anchor=bbox, frameon=False,
        handlelength=2.4, columnspacing=1.55, fontsize=11.0,
    )


def make_single_row_figure(curves: dict, key: str, ylabel: str, stem: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.25), sharex=True, sharey=True)
    fig.patch.set_facecolor('white')
    for col, q in enumerate(ORDERS):
        ax = axes[col]
        _plot_curves(ax, q, curves, key)
        _style_axis(ax)
        ax.set_title(rf'$q={q}$', color=_ACCENT)
        ax.set_xlabel(r'points per wavelength (lower is harder)', color=_ACCENT)
    axes[0].set_ylabel(ylabel, color=_ACCENT)
    _shared_legend(fig, list(axes), bbox=(0.5, 0.985))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, stem, pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    summary, curves = collect_curves()
    make_single_row_figure(
        curves, 'response',
        r'wall response error $|\rho-1|$',
        'fig_s6_wallmode_response',
    )
    make_single_row_figure(
        curves, 'contamination',
        r'orthogonal contamination $\|c\|_2$',
        'fig_s6_wallmode_contamination',
    )
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
