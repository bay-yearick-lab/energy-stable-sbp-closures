"""Heat-cavity comparison: composed DD vs tailored D2 for the Task-2 competitors.

Adds three labels to the q=6 heat-cavity benchmark that the original
s6_heat_cavity.py / s6_heat_cavity_convergence.py scripts do not carry:

  C6-DD     the composed D@D second derivative for C6 (the ORIGINAL choice
            the paper used for all non-SBP families; kept here, labeled
            explicitly as "composed", for direct comparison).
  C6-D2     the classical compact-difference reference compact second-derivative closure riding on the same
            C6 first-derivative operator (see boundary_spline_analysis.py's
            CompactD2FD / compact_baseline_c6_d2_matrices).
  SBP6-BO   the 2018 boundary-optimized SBP6 construction boundary-optimized SBP6,
            using the compatible SBP second derivative built from its own
            (D, H) pair (sbp6_bo_second_derivative_matrix; the 2018 paper is
            first-derivative only, so no coefficients are transcribed for
            D2 -- the construction itself is the published SBP method).

Classical SBP6 (H^{-1}(-D^T H D - e0 s0 + eN sN), the paper's existing
scheme-level D2) is included unchanged as the baseline for comparison.

Only q=6 is demonstrated: C6-D2 and SBP6-BO are only implemented at order 6
in this codebase (the published tridiagonal family and the "Accurate"
boundary-optimized table used here are both order-6 instances). This is
stated explicitly in the output JSON and here.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import boundary_spline_analysis as ba
from project_paths import FIGDIR, RESULTSDIR, ensure_generated_dirs, recommended_process_workers, save_figure
from heat_cavity_core import (default_heat_modes, per_method_diffusion_dt_extended,
                              run_heat_cavity, run_heat_cavity_extended)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10.2,
    'axes.labelsize': 11.2,
    'axes.titlesize': 12.0,
    'legend.fontsize': 9.8,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 180,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.12,
    'grid.linewidth': 0.45,
})

# Fixed-grid comparison configuration (matches s6_heat_cavity.py's q=6 slice).
FIXED_GRID = {
    "m_center": 28, "n_center": 28, "half_band": 2, "viscosity": 5.0e-5,
    "npts": 81, "t_end": 2.0, "cfl": 0.01, "record_count": 201, "snapshot_count": 61,
}
LABELS_FIXED = ["C6-DD", "C6-D2", "SBP6", "SBP6-BO", "BMF6", "BMG6"]

# MMS convergence-sweep configuration (matches s6_heat_cavity_convergence.py).
CONVERGENCE = {
    "m_center": 8, "n_center": 8, "half_band": 1, "viscosity": 5.0e-5,
    "t_end": 0.5, "cfl": 0.01, "grids": [33, 49, 65, 81, 97, 129],
}
LABELS_CONV = ["C6-DD", "C6-D2", "SBP6", "SBP6-BO"]

COLORS = {
    "C6-DD": "#8fb8de", "C6-D2": "#1f4e79", "SBP6": "#9467bd",
    "SBP6-BO": "#66c2a5", "BMF6": "#2ca02c", "BMG6": "#17a2b8",
}
LINESTYLES = {
    "C6-DD": ":", "C6-D2": "-", "SBP6": "-.", "SBP6-BO": "--", "BMF6": (0, (6.0, 1.6, 1.2, 1.6)), "BMG6": (0, (1.2, 1.2)),
}
MARKERS = {"C6-DD": "o", "C6-D2": "s", "SBP6": "P", "SBP6-BO": "X", "BMF6": "^", "BMG6": "D"}


def _run_fixed(label: str, dt_shared: float):
    # per-method stable step (tier B); dt_shared retained for provenance only
    dt_shared = per_method_diffusion_dt_extended(
        "C6" if label == "C6-DD" else label, npts=int(FIXED_GRID["npts"]),
        viscosity=float(FIXED_GRID["viscosity"]), t_end=float(FIXED_GRID["t_end"]))
    modes = default_heat_modes(
        m_center=FIXED_GRID["m_center"], n_center=FIXED_GRID["n_center"],
        half_band=FIXED_GRID["half_band"], viscosity=FIXED_GRID["viscosity"],
    )
    real_label = "C6" if label == "C6-DD" else label
    r = run_heat_cavity_extended(
        real_label, modes, npts=FIXED_GRID["npts"], t_end=FIXED_GRID["t_end"],
        cfl=FIXED_GRID["cfl"], dt=dt_shared,
        record_count=FIXED_GRID["record_count"], snapshot_count=FIXED_GRID["snapshot_count"],
    ) if label != "C6-DD" else run_heat_cavity(
        "C6", modes, npts=FIXED_GRID["npts"], t_end=FIXED_GRID["t_end"],
        cfl=FIXED_GRID["cfl"], dt=dt_shared,
        record_count=FIXED_GRID["record_count"], snapshot_count=FIXED_GRID["snapshot_count"],
    )
    return label, r


def _shared_dt_fixed() -> float:
    # h_min across all participating labels' grids (SBP6-BO's clustered
    # boundary nodes are typically the binding constraint), matching the
    # fairness convention documented in heat_cavity_core.shared_diffusion_dt.
    npts = FIXED_GRID["npts"]
    grids = [
        ba.build_operator("C", 6, npts).x,
        ba.build_operator("SBP", 6, npts).x,
        ba.build_operator("B", 6, npts).x,
        ba.build_operator("I", 6, npts).x,
        ba.build_operator_extended("SBP6-BO", 6, npts).x,
    ]
    h_min = min(float(np.min(np.diff(x))) for x in grids)
    dt0 = FIXED_GRID["cfl"] * h_min * h_min / FIXED_GRID["viscosity"]
    nsteps = max(1, int(np.ceil(FIXED_GRID["t_end"] / dt0)))
    return FIXED_GRID["t_end"] / nsteps


def run_fixed_grid(max_workers: int | None = None) -> dict:
    dt_shared = _shared_dt_fixed()
    if max_workers is None:
        max_workers = recommended_process_workers(len(LABELS_FIXED), cap=6)
    print(f"== heat-cavity strong-competitor fixed grid: {max_workers} workers, dt={dt_shared:.3e} ==")
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_run_fixed, lab, dt_shared): lab for lab in LABELS_FIXED}
        for fut in as_completed(futs):
            label, r = fut.result()
            results[label] = r
            print(f"  {label:>8s}: steps={r.steps:4d} rel_L2={r.u_rel_l2[-1]:.3e} linf={r.u_linf[-1]:.3e}")
    return results


def _run_conv(label: str, npts: int, dt_shared: float) -> dict:
    dt_shared = per_method_diffusion_dt_extended(
        "C6" if label == "C6-DD" else label, npts=int(npts),
        viscosity=float(CONVERGENCE["viscosity"]), t_end=float(CONVERGENCE["t_end"]))
    modes = default_heat_modes(
        m_center=CONVERGENCE["m_center"], n_center=CONVERGENCE["n_center"],
        half_band=CONVERGENCE["half_band"], viscosity=CONVERGENCE["viscosity"],
    )
    real_label = "C6" if label == "C6-DD" else label
    if label == "C6-DD":
        r = run_heat_cavity(real_label, modes, npts=npts, t_end=CONVERGENCE["t_end"],
                             cfl=CONVERGENCE["cfl"], dt=dt_shared, record_count=2, snapshot_count=2)
    else:
        r = run_heat_cavity_extended(real_label, modes, npts=npts, t_end=CONVERGENCE["t_end"],
                                      cfl=CONVERGENCE["cfl"], dt=dt_shared, record_count=2, snapshot_count=2)
    return {
        "label": label, "npts": int(npts), "h": 1.0 / (int(npts) - 1), "ppw": float(r.ppw),
        "rel_l2_final": float(r.u_rel_l2[-1]), "linf_final": float(r.u_linf[-1]),
    }


def _dt_for_grid(npts: int) -> float:
    grids = [
        ba.build_operator("C", 6, npts).x,
        ba.build_operator("SBP", 6, npts).x,
        ba.build_operator_extended("SBP6-BO", 6, npts).x,
    ]
    h_min = min(float(np.min(np.diff(x))) for x in grids)
    dt0 = CONVERGENCE["cfl"] * h_min * h_min / CONVERGENCE["viscosity"]
    nsteps = max(1, int(np.ceil(CONVERGENCE["t_end"] / dt0)))
    return CONVERGENCE["t_end"] / nsteps


def run_convergence_sweep(max_workers: int | None = None) -> dict[str, list[dict]]:
    dt_per_grid = {n: _dt_for_grid(n) for n in CONVERGENCE["grids"]}
    jobs = [(lab, n) for lab in LABELS_CONV for n in CONVERGENCE["grids"]]
    if max_workers is None:
        max_workers = recommended_process_workers(len(jobs), cap=6)
    print(f"== heat-cavity strong-competitor MMS sweep: {max_workers} workers, {len(jobs)} jobs ==")
    results: dict[str, list[dict]] = {lab: [] for lab in LABELS_CONV}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_run_conv, lab, n, dt_per_grid[n]): (lab, n) for lab, n in jobs}
        for fut in as_completed(futs):
            lab, n = futs[fut]
            res = fut.result()
            results[lab].append(res)
            print(f"  {lab:>8s} N={n:3d} h={res['h']:.5f} ppw={res['ppw']:.2f} rel_L2={res['rel_l2_final']:.3e}")
    for lab in results:
        results[lab].sort(key=lambda r: r["h"], reverse=True)
    return results


def fit_slope(xs: np.ndarray, ys: np.ndarray) -> float:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(ys) & (ys > 0.0) & (ys < 1.0e3)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 2:
        return float('nan')
    half = max(2, len(xs) // 2)
    lx, ly = np.log(xs[-half:]), np.log(ys[-half:])
    a, _ = np.polyfit(lx, ly, 1)
    return float(a)


def merge_fixed_spline_records(existing: dict, fresh: dict) -> dict:
    if set(fresh) != {"BMF6", "BMG6"}:
        raise ValueError("fixed-grid heat rescore must contain BMF6 and BMG6")
    if set(existing) != set(LABELS_FIXED):
        raise ValueError("source fixed-grid heat comparison is incomplete")
    merged = {label: dict(record) for label, record in existing.items()}
    merged.update({label: dict(record) for label, record in fresh.items()})
    return merged


def summarize_fixed_results(fixed_results: dict) -> dict:
    return {
        lab: {
            "final_rel_l2": float(fixed_results[lab].u_rel_l2[-1]),
            "final_linf": float(fixed_results[lab].u_linf[-1]),
            "steps": int(fixed_results[lab].steps),
            "dt": float(fixed_results[lab].dt),
        }
        for lab in LABELS_FIXED
    }


def make_figure(fixed_results: dict, conv_results: dict[str, list[dict]]) -> None:
    fig = plt.figure(figsize=(13.6, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.85], wspace=0.32,
                          left=0.055, right=0.985, top=0.80, bottom=0.16)

    ax_a = fig.add_subplot(gs[0, 0])
    for lab in LABELS_CONV:
        recs = conv_results[lab]
        hs = np.array([r["h"] for r in recs])
        e = np.array([r["rel_l2_final"] for r in recs])
        mask = np.isfinite(e) & (e > 0.0)
        ax_a.loglog(hs[mask], e[mask], color=COLORS[lab], ls=LINESTYLES[lab],
                    marker=MARKERS[lab], lw=2.0, ms=6.0, mfc='white', mec=COLORS[lab], mew=1.1,
                    label=lab)
    ax_a.invert_xaxis()
    ax_a.set_xlabel(r'grid spacing $h=1/(N-1)$')
    ax_a.set_ylabel(r'$\|u-u_\mathrm{exact}\|_2/\|u_\mathrm{exact}\|_2$')
    ax_a.set_title('(a) MMS grid convergence, $q=6$', fontsize=11.6, pad=8)
    ax_a.legend(frameon=False, fontsize=9.4, loc='lower right')

    ax_b = fig.add_subplot(gs[0, 1])
    for lab in LABELS_FIXED:
        r = fixed_results[lab]
        ax_b.semilogy(r.times, np.maximum(r.u_linf, 1.0e-14), color=COLORS[lab],
                      ls=LINESTYLES[lab], lw=2.0, label=lab)
    ax_b.set_xlabel('time $t$')
    ax_b.set_ylabel(r'$\|u-u_\mathrm{exact}\|_\infty$')
    ax_b.set_title('(b) Fixed-grid pointwise error history', fontsize=11.6, pad=8)
    ax_b.set_ylim(top=3.0e0)
    ax_b.legend(frameon=False, fontsize=9.4, ncol=2, loc='lower right')

    ax_c = fig.add_subplot(gs[0, 2])
    pairs = sorted(((lab, float(fixed_results[lab].u_linf[-1])) for lab in LABELS_FIXED), key=lambda kv: kv[1])
    bar_labels = [lab for lab, _ in pairs]
    bar_vals = [v for _, v in pairs]
    bar_colors = [COLORS[lab] for lab, _ in pairs]
    ax_c.bar(bar_labels, bar_vals, color=bar_colors, edgecolor='black', lw=0.55)
    ax_c.set_yscale('log')
    ax_c.set_ylabel(r'final $\|u-u_\mathrm{exact}\|_\infty$')
    ax_c.set_title('(c) Final pointwise peak', fontsize=11.6, pad=8)
    ax_c.tick_params(axis='x', labelrotation=32)
    for tick in ax_c.get_xticklabels():
        tick.set_ha('right')
        tick.set_rotation_mode('anchor')

    save_figure(fig, "fig_s6_heat_cavity_strong_competitors")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--rescore-fixed",
        action="store_true",
        help="Rerun the fixed-grid panel with corrected actual-node weights and preserve the nodal convergence sweep.",
    )
    args = parser.parse_args()
    ensure_generated_dirs()
    print("== fixed-grid comparison ==")
    fixed_results = run_fixed_grid(args.workers)
    out_path = RESULTSDIR / "heat_cavity_strong_competitors_summary.json"
    if args.rescore_fixed:
        source = json.loads(out_path.read_text())
        conv_results = source["convergence_sweeps"]
        print("== preserving uniform-grid convergence sweep ==")
    else:
        print("== MMS convergence sweep ==")
        conv_results = run_convergence_sweep(args.workers)

    measured_slopes = {lab: fit_slope(
        np.array([r["h"] for r in conv_results[lab]]),
        np.array([r["rel_l2_final"] for r in conv_results[lab]]),
    ) for lab in LABELS_CONV}

    out = {
        "note": "q=6 demonstration block only; C6-D2 and SBP6-BO are order-6-only "
                "in this codebase (the published D2 family and the 'Accurate' "
                "boundary-optimized table are both transcribed at q=6).",
        "fixed_grid_configuration": FIXED_GRID,
        "convergence_configuration": CONVERGENCE,
        "fixed_grid": summarize_fixed_results(fixed_results),
        "convergence_sweeps": conv_results,
        "measured_slopes": measured_slopes,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")

    make_figure(fixed_results, conv_results)
    print(f"wrote {FIGDIR}/fig_s6_heat_cavity_strong_competitors.(pdf,png)")

    print("measured slopes:", {k: round(v, 2) for k, v in measured_slopes.items()})
    print("fixed-grid final linf:", {k: f"{out['fixed_grid'][k]['final_linf']:.3e}" for k in LABELS_FIXED})


if __name__ == "__main__":
    main()
