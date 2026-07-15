"""MMS convergence study for the closed-cavity heat-decay benchmark.

Fixes the continuous data: viscosity nu = 5e-5, sine-sine mode band
centred at (m, n) = (8, 8) with a narrow Gaussian weight (so every grid
in the sweep resolves the dominant component to many PPW), and T = 0.5
so the exponential decay leaves a nontrivial field at all grids.

The six matched-order families (E, C, SBP, DRP, BMF, BMG) are integrated
at q in {4, 6, 8} on the shared grid sequence N in
{33, 49, 65, 81, 97, 129} with RK4 under a shared parabolic CFL.
The relative L2 error at T and the maximum pointwise error are recorded.

The output figure is a 2x3 grid:
  Top row    rel_L2 vs h at q = 4, 6, 8 (grid-convergence view).
  Bottom row rel_L2 vs PPW_peak at the same q (resolution view).

Each panel reports a short dotted formal-order reference segment at the
target slope. The shared family legend sits above the panel grid.
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
from boundary_spline_analysis import scheme_display_name
from project_paths import FIGDIR, RESULTSDIR, ensure_generated_dirs, recommended_process_workers, save_figure
from heat_cavity_core import (
    per_method_diffusion_dt,
    default_heat_modes,
    heat_cavity_labels,
    run_heat_cavity,
    shared_diffusion_dt,
)
from plotting_helpers import add_theoretical_reference_line

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10.2,
    'axes.labelsize': 11.2,
    'axes.titlesize': 12.0,
    'legend.fontsize': 10.5,
    'xtick.labelsize': 9.3,
    'ytick.labelsize': 9.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 180,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.12,
    'grid.linewidth': 0.45,
})

CONVERGENCE = {
    "m_center": 8,
    "n_center": 8,
    "half_band": 1,
    "viscosity": 5.0e-5,
    "t_end": 0.5,
    "cfl": 0.01,
    "grids": [33, 49, 65, 81, 97, 129],
    "roster": "six_family",
}
ORDERS = (4, 6, 8)


def _scheme_order(label: str) -> int:
    for q in ORDERS:
        if label.endswith(str(q)):
            return q
    raise ValueError(f"could not infer formal order from label {label!r}")


def _run(label: str, npts: int, dt_shared: float) -> dict:
    modes = default_heat_modes(
        m_center=CONVERGENCE["m_center"],
        n_center=CONVERGENCE["n_center"],
        half_band=CONVERGENCE["half_band"],
        viscosity=CONVERGENCE["viscosity"],
    )
    dt_own = per_method_diffusion_dt(
        label,
        npts=int(npts),
        viscosity=float(CONVERGENCE["viscosity"]),
        t_end=float(CONVERGENCE["t_end"]),
    )
    r = run_heat_cavity(
        label, modes,
        npts=int(npts),
        t_end=float(CONVERGENCE["t_end"]),
        cfl=float(CONVERGENCE["cfl"]),
        dt=float(dt_own),
        record_count=2,
        snapshot_count=2,
    )
    return {
        "label": label,
        "npts": int(npts),
        "h": 1.0 / (int(npts) - 1),
        "ppw": float(r.ppw),
        "rel_l2_final": float(r.u_rel_l2[-1]),
        "linf_final": float(r.u_linf[-1]),
    }


def run_sweep(max_workers: int | None = None) -> dict:
    labels = heat_cavity_labels(CONVERGENCE.get("roster", "sbp_closed"))
    dt_per_grid: dict[int, float] = {}
    for n in CONVERGENCE["grids"]:
        dt_per_grid[int(n)] = shared_diffusion_dt(
            labels, npts=int(n),
            viscosity=float(CONVERGENCE["viscosity"]),
            cfl=float(CONVERGENCE["cfl"]),
            t_end=float(CONVERGENCE["t_end"]),
        )
    jobs = [(lab, n) for lab in labels for n in CONVERGENCE["grids"]]
    if max_workers is None:
        max_workers = recommended_process_workers(len(jobs), cap=8)
    print(f"== using {max_workers} worker process(es) on {len(jobs)} jobs ==")
    results: dict[str, list[dict]] = {lab: [] for lab in labels}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut = {ex.submit(_run, lab, n, dt_per_grid[int(n)]): (lab, n)
               for lab, n in jobs}
        for f in as_completed(fut):
            lab, n = fut[f]
            res = f.result()
            results[lab].append(res)
            print(f"  {lab:>5s}  N={n:3d}  h={res['h']:.5f}  "
                  f"ppw={res['ppw']:.2f}  rel_L2={res['rel_l2_final']:.3e}")
    for lab in results:
        results[lab].sort(key=lambda r: r["h"], reverse=True)
    return results


def run_spline_sweep(max_workers: int | None = None) -> dict[str, list[dict]]:
    labels = ("BMF4", "BMG4", "BMF6", "BMG6", "BMF8", "BMG8")
    all_labels = heat_cavity_labels(CONVERGENCE.get("roster", "sbp_closed"))
    dt_per_grid = {
        int(n): shared_diffusion_dt(
            all_labels, npts=int(n), viscosity=float(CONVERGENCE["viscosity"]),
            cfl=float(CONVERGENCE["cfl"]), t_end=float(CONVERGENCE["t_end"]),
        )
        for n in CONVERGENCE["grids"]
    }
    jobs = [(label, npts) for label in labels for npts in CONVERGENCE["grids"]]
    workers = 1 if max_workers is None else max_workers
    fresh = {label: [] for label in labels}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run, label, npts, dt_per_grid[int(npts)]): (label, npts)
            for label, npts in jobs
        }
        for future in as_completed(futures):
            label, _npts = futures[future]
            fresh[label].append(future.result())
    for label in fresh:
        fresh[label].sort(key=lambda row: row["h"], reverse=True)
    return fresh


def merge_spline_sweeps(existing: dict[str, list[dict]],
                        fresh: dict[str, list[dict]]) -> dict[str, list[dict]]:
    expected_grids = set(CONVERGENCE["grids"])
    spline_labels = {"BMF4", "BMG4", "BMF6", "BMG6", "BMF8", "BMG8"}
    if set(fresh) != spline_labels:
        raise ValueError("corrected heat rescore must contain all six spline families")
    merged = {label: list(rows) for label, rows in existing.items()}
    for label, rows in fresh.items():
        if {row.get("npts") for row in rows} != expected_grids or len(rows) != len(expected_grids):
            raise ValueError(f"incomplete corrected heat sweep for {label}")
        merged[label] = list(rows)
    return merged


def fit_slope(xs: np.ndarray, ys: np.ndarray) -> float:
    """Least-squares slope in log-log, using the finest half of the sweep."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(ys) & (ys > 0.0) & (ys < 1.0e3)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size < 2:
        return float('nan')
    half = max(2, len(xs) // 2)
    lx = np.log(xs[-half:])
    ly = np.log(ys[-half:])
    a, _ = np.polyfit(lx, ly, 1)
    return float(a)


def load_results_from_summary(path: Path) -> dict[str, list[dict]]:
    summary = json.loads(path.read_text())
    results: dict[str, list[dict]] = {}
    for lab, recs in summary["sweeps"].items():
        results[lab] = [
            {
                "label": lab,
                "npts": int(rec["npts"]),
                "h": float(rec["h"]),
                "ppw": float(rec["ppw"]),
                "rel_l2_final": float(rec["rel_l2_final"]),
                "linf_final": float(rec["linf_final"]),
            }
            for rec in recs
        ]
    return results


_PANEL_BG = (0.987, 0.985, 0.978)
_ACCENT = '#2F4858'


def _style_axis(ax) -> None:
    ax.set_facecolor(_PANEL_BG)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('bottom', 'left'):
        ax.spines[s].set_color(_ACCENT)
        ax.spines[s].set_linewidth(0.55)
    ax.tick_params(colors=_ACCENT)
    ax.grid(True, which='major', alpha=0.18, lw=0.5)
    ax.grid(True, which='minor', alpha=0.08, lw=0.4)


FAMILY_ORDER = ('E', 'C', 'SBP', 'DRP', 'BMF', 'BMG')


def _family_legend(fig, *, bbox=(0.5, 0.995)) -> None:
    handles = []
    seen: set[str] = set()
    for fam in FAMILY_ORDER:
        if fam in seen:
            continue
        seen.add(fam)
        handles.append(Line2D(
            [0], [0], color=ba.DEFAULT_COLORS[fam],
            ls=ba.DEFAULT_LINESTYLES[fam], lw=2.4,
            marker=ba.DEFAULT_MARKERS[fam], ms=6.8,
            mfc='white', mec=ba.DEFAULT_COLORS[fam], mew=1.1,
            label={
                'E': r'$E_q$', 'C': r'$C_q$',
                'SBP': r'$\mathrm{SBP}_q$', 'DRP': r'$\mathrm{DRP}_q$',
                'BMF': r'$\mathrm{BMF}_q$', 'BMG': r'$\mathrm{BMG}_q$',
            }[fam],
        ))
    fig.legend(
        handles, [h.get_label() for h in handles],
        ncol=len(handles), loc='upper center',
        bbox_to_anchor=bbox, frameon=False,
        handlelength=2.8, columnspacing=1.6, handletextpad=0.7,
        fontsize=11.6,
    )


def make_figure(results: dict[str, list[dict]], stem: str) -> dict:
    labels = heat_cavity_labels(CONVERGENCE.get("roster", "sbp_closed"))

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6), sharey='row')
    fig.patch.set_facecolor('white')

    measured_slopes: dict[str, float] = {}
    by_q: dict[int, list[str]] = {q: [] for q in ORDERS}
    for lab in labels:
        by_q[_scheme_order(lab)].append(lab)

    for col, q in enumerate(ORDERS):
        ax_h = axes[0, col]
        ax_p = axes[1, col]
        unstable_labels: list[str] = []
        for lab in by_q[q]:
            fam = ba.scheme_family(lab)
            recs = results[lab]
            hs = np.array([r["h"] for r in recs])
            ppw = np.array([r["ppw"] for r in recs])
            e = np.array([r["rel_l2_final"] for r in recs])
            mask = np.isfinite(e) & (e > 0.0) & (e < 1.0e3)
            style = dict(
                color=ba.DEFAULT_COLORS[fam],
                ls=ba.DEFAULT_LINESTYLES[fam],
                marker=ba.DEFAULT_MARKERS[fam],
                lw=2.0, ms=6.5,
                mfc='white', mec=ba.DEFAULT_COLORS[fam], mew=1.1,
            )
            ax_h.loglog(hs[mask], e[mask], **style)
            ax_p.loglog(ppw[mask], e[mask], **style)
            measured_slopes[lab] = fit_slope(hs, e)
            if not np.any(mask):
                fam_root = ba.scheme_family(lab)
                deg = lab[len(fam_root):]
                unstable_labels.append(rf"$\mathrm{{{fam_root}}}_{{{deg}}}$")
        if unstable_labels:
            note = "unstable: " + ", ".join(unstable_labels)
            for ax in (ax_h, ax_p):
                ax.text(
                    0.5, 0.05, note,
                    transform=ax.transAxes,
                    ha='center', va='bottom',
                    fontsize=9.4, color=_ACCENT, fontstyle='italic',
                    family='serif',
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.6},
                )

        ax_h.set_title(rf'$q={q}$: grid convergence',
                       fontsize=11.6, color=_ACCENT, pad=6)
        ax_h.invert_xaxis()
        add_theoretical_reference_line(
            ax_h, slope=q, label=rf'$\propto h^{{{q}}}$',
            candidate_regions=("lower right", "mid right", "upper right",
                               "upper left", "lower left"),
        )
        ax_p.set_title(rf'$q={q}$: resolution view',
                       fontsize=11.6, color=_ACCENT, pad=6)
        add_theoretical_reference_line(
            ax_p, slope=-q,
            label=rf'$\propto \mathrm{{PPW}}^{{-{q}}}$',
            candidate_regions=("lower right", "mid right", "upper right",
                               "upper left", "lower left"),
        )
        _style_axis(ax_h)
        _style_axis(ax_p)
        ax_p.set_xlabel(r'peak points per wavelength', color=_ACCENT)
        if col == 0:
            ax_h.set_ylabel(r'$\|u-u_\mathrm{exact}\|_2 / \|u_\mathrm{exact}\|_2$',
                            color=_ACCENT)
            ax_p.set_ylabel(r'$\|u-u_\mathrm{exact}\|_2 / \|u_\mathrm{exact}\|_2$',
                            color=_ACCENT)
        ax_h.tick_params(labelbottom=True)
    for ax in axes[0, :]:
        ax.set_xlabel(r'grid spacing $h = 1/(N-1)$', color=_ACCENT)

    _family_legend(fig, bbox=(0.5, 0.985))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, stem)
    plt.close(fig)
    return measured_slopes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=None,
                        help="Override the process-pool size.")
    parser.add_argument(
        "--from-summary",
        action="store_true",
        help="Regenerate the figure from the existing summary JSON instead of rerunning the sweep.",
    )
    parser.add_argument(
        "--rescore-splines",
        action="store_true",
        help="Preserve uniform-grid baselines and recompute only BMF/BMG rows with actual-node weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_generated_dirs()
    if args.rescore_splines:
        summary_path = RESULTSDIR / "heat_cavity_convergence_summary.json"
        existing = load_results_from_summary(summary_path)
        fresh = run_spline_sweep(args.workers)
        results = merge_spline_sweeps(existing, fresh)
    elif args.from_summary:
        summary_path = RESULTSDIR / "heat_cavity_convergence_summary.json"
        print(f"== loading existing sweep data from {summary_path} ==")
        results = load_results_from_summary(summary_path)
    else:
        roster = CONVERGENCE.get("roster", "sbp_closed")
        print(f"== MMS convergence sweep for the {roster} roster ==")
        results = run_sweep(args.workers)
    print("== figure ==")
    measured = make_figure(results, "fig_s6_heat_cavity_convergence")
    print(f"wrote {FIGDIR}/fig_s6_heat_cavity_convergence.(pdf,png)")

    labels = heat_cavity_labels(CONVERGENCE.get("roster", "sbp_closed"))
    summary = {
        "configuration": CONVERGENCE,
        "formal_orders_per_label": {lab: _scheme_order(lab) for lab in labels},
        "sweeps": {
            lab: [
                {k: r[k] for k in ("npts", "h", "ppw", "rel_l2_final", "linf_final")}
                for r in results[lab]
            ]
            for lab in labels
        },
        "measured_slopes": measured,
    }
    out = RESULTSDIR / "heat_cavity_convergence_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
