#!/usr/bin/env python3
"""Figure for the coupled lid-driven cavity stress test: spectral benchmark
accuracy anchor, the conservation cliff under under-resolution, and the
extrema convergence ladder. Reads code/cavity/data/ artifacts."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))
from cavity_postprocess import BENCHMARKS  # noqa: E402
from project_paths import ensure_generated_dirs, save_figure  # noqa: E402

DATA = _CODE / "cavity" / "data"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.0,
    "axes.labelsize": 10.8,
    "axes.titlesize": 11.2,
    "legend.fontsize": 9.6,
    "xtick.labelsize": 9.4,
    "ytick.labelsize": 9.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.14,
    "grid.linewidth": 0.5,
})

NCOLORS = {32: "#9ecae1", 48: "#6baed6", 64: "#2171b5", 96: "#08306b"}


def read_xy(path, xc, yc):
    xs, ys = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            xs.append(float(row[xc])); ys.append(float(row[yc]))
    return np.asarray(xs), np.asarray(ys)


def read_budget(prefix):
    path = DATA / f"{prefix}_budgets_raw.csv"
    cols = {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for k in rows[0]:
        cols[k] = np.array([float(r[k]) for r in rows if r[k] != ""])
    return cols


def extrema_from_centerlines(prefix, margin=0.04):
    z, u = read_xy(DATA / f"{prefix}_centerline_u.csv", "z", "u")
    x, w = read_xy(DATA / f"{prefix}_centerline_w.csv", "x", "w")
    umin = float(np.min(u))
    inb = (x > margin) & (x < 1 - margin)
    return {"u_min": umin, "v_min": float(np.min(w[inb])), "v_max": float(np.max(w[inb]))}


def main() -> None:
    ensure_generated_dirs()
    bench = BENCHMARKS["bp1000"]
    ref_u = np.array(bench["u"])          # rows (y, u)
    ref_ex = bench["extrema"]

    grids = [n for n in (32, 48, 64, 96)
             if (DATA / f"cav1000_N{n}cp_centerline_u.csv").exists()]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.4, 3.6))

    # (a) centerline u(0.5, z) against spectral benchmark tabulated points
    for n in grids:
        z, u = read_xy(DATA / f"cav1000_N{n}cp_centerline_u.csv", "z", "u")
        ax1.plot(u, z, "-", lw=1.6, color=NCOLORS[n], label=f"$N={n}$")
    ax1.plot(ref_u[:, 1], ref_u[:, 0], "o", ms=4.6, mfc="none", mec="#d62728",
             mew=1.4, label="spectral benchmark")
    ax1.set_xlabel(r"$u(1/2,z)$")
    ax1.set_ylabel(r"$z$")
    ax1.set_title("(a) Centerline against the spectral benchmark")
    ax1.legend(frameon=False, loc="upper left", handlelength=1.5)

    # (b) conservation cliff: advective KE production, skew vs convective
    for re_, ls in ((3200, "--"), (5000, "-")):
        for form, color in (("skew", "#1f77b4"), ("conv", "#d62728")):
            prefix = f"hh3_{form}_re{re_}_N32"
            if not (DATA / f"{prefix}_budgets_raw.csv").exists():
                continue
            b = read_budget(prefix)
            ax2.semilogy(b["t"][1:], np.abs(b["ke_production"][1:]) + 1e-22,
                         ls, lw=1.5, color=color,
                         label=f"{'conservative' if form == 'skew' else 'convective'}, $\\mathrm{{Re}}={re_}$")
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$|$advective KE production$|$")
    ax2.set_title("(b) Conservation under under-resolution, $N=32$")
    ax2.legend(frameon=False, loc="center right", fontsize=9.2)

    # (c) extrema convergence to the benchmark
    marks = {"u_min": "o", "v_min": "s", "v_max": "^"}
    for key, mk in marks.items():
        errs, ns = [], []
        for n in grids:
            ex = extrema_from_centerlines(f"cav1000_N{n}cp")
            errs.append(abs(ex[key] - ref_ex[key]) / abs(ref_ex[key]))
            ns.append(n)
        ax3.loglog(ns, errs, "-" + mk, ms=5, lw=1.6,
                   label=key.replace("_", " "))
    ax3.set_xlabel(r"$N$ (points per direction)")
    ax3.set_ylabel("relative extremum error")
    ax3.set_title("(c) Benchmark extrema under refinement")
    ax3.set_xticks(grids)
    ax3.set_xticklabels([str(n) for n in grids])
    ax3.legend(frameon=False, loc="lower left")

    fig.tight_layout(w_pad=1.7)
    save_figure(fig, "fig_s6_cavity_stress")
    print("wrote fig_s6_cavity_stress; grids:", grids)

    # machine-readable summary for the manuscript numbers
    out = {"grids": grids, "benchmark": bench["label"], "extrema": {}}
    for n in grids:
        ex = extrema_from_centerlines(f"cav1000_N{n}cp")
        summ = json.loads((DATA / f"cav1000_N{n}cp_run_summary.json").read_text())
        out["extrema"][str(n)] = {
            **ex,
            "rel": {k: abs(ex[k] - ref_ex[k]) / abs(ref_ex[k]) for k in ex},
            "final_ke_production": summ.get("final_ke_production"),
            "final_pressure_work": summ.get("final_pressure_work"),
            "final_dissipation": summ.get("final_dissipation"),
            "final_max_dudt": summ.get("final_max_dudt"),
            "constraint_l2_div": summ.get("final_constraint_l2_div"),
            "cpu_seconds_per_step": summ.get("cpu_seconds_per_step"),
            "peak_rss_mb": summ.get("peak_rss_mb"),
            "T": summ.get("T"),
        }
    for form in ("skew", "conv"):
        for re_ in (3200, 5000):
            p = DATA / f"hh3_{form}_re{re_}_N32_run_summary.json"
            if p.exists():
                s = json.loads(p.read_text())
                out.setdefault("headtohead", {})[f"{form}_re{re_}"] = {
                    "final_ke_production": s["final_ke_production"],
                    "final_pressure_work": s.get("final_pressure_work"),
                    "T": s["T"],
                    "nsteps": s["nsteps"],
                }
    (Path(_CODE) / "results" / "cavity_stress_summary.json").write_text(
        json.dumps(out, indent=1))
    print("wrote results/cavity_stress_summary.json")


if __name__ == "__main__":
    main()
