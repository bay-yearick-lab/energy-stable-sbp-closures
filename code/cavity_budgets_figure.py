#!/usr/bin/env python3
"""
Build the lid-driven cavity budgets figure: centerline validation against
the Re=1000 spectral benchmark plus the live semidiscrete conservation budgets
(kinetic-energy balance residual, momentum balance residual, divergence
defect) over the spin-up to steady state.

Usage: uv run --with numpy --with matplotlib python cavity_budgets_figure.py \
           <outprefix> <benchmark> [data_dir] [results_dir]
"""
import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from cavity_postprocess import BENCHMARKS


def read_csv_cols(path, cols):
    out = {c: [] for c in cols}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            for c in cols:
                out[c].append(float(row[c]))
    return out


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "cav400_N32"
    bench_key = sys.argv[2] if len(sys.argv) > 2 else "cavity_reference400"
    data_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("cavity/data")
    results_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("results")
    # NOTE: scope for this dispatch is code/results/, not the manuscript's
    # figures/ directory (manuscript integration is a separate dispatch).
    fig_dir = results_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    bench = BENCHMARKS[bench_key]
    ref_u, ref_v, ref_label = bench["u"], bench["v"], bench["label"]

    cols = ["t", "KE", "dKEdt", "dissipation", "ke_residual",
            "dPxdt", "dPzdt", "max_weak_div", "max_dudt"]
    # l2_weak_div added in a later driver version; tolerate its absence
    import csv as _csv
    with open(data_dir / f"{prefix}_budgets_raw.csv") as _f:
        has_l2 = "l2_weak_div" in (_csv.DictReader(_f).fieldnames or [])
    if has_l2:
        cols.append("l2_weak_div")
    b = read_csv_cols(data_dir / f"{prefix}_budgets_raw.csv", cols)
    zc, uc = [], []
    with open(data_dir / f"{prefix}_centerline_u.csv") as f:
        rr = csv.DictReader(f)
        for row in rr: zc.append(float(row["z"])); uc.append(float(row["u"]))
    xc, wc = [], []
    with open(data_dir / f"{prefix}_centerline_w.csv") as f:
        rr = csv.DictReader(f)
        for row in rr: xc.append(float(row["x"])); wc.append(float(row["w"]))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # (a) u centerline
    ax = axes[0, 0]
    ax.plot(uc, zc, '-', color='#1f77b4', lw=1.6, label='this work ($\\mathrm{BMG}_4$)')
    rz = [p[0] for p in ref_u]; ru = [p[1] for p in ref_u]
    ax.plot(ru, rz, 'o', color='#d62728', ms=4, mfc='none', label=ref_label)
    ax.set_xlabel('$u(0.5,z)$'); ax.set_ylabel('$z$')
    ax.set_title('(a) horizontal-velocity centerline')
    ax.legend(loc='upper left', frameon=False, fontsize=8)

    # (b) w centerline
    ax = axes[0, 1]
    ax.plot(xc, wc, '-', color='#1f77b4', lw=1.6, label='this work ($\\mathrm{BMG}_4$)')
    rx = [p[0] for p in ref_v]; rv = [p[1] for p in ref_v]
    ax.plot(rx, rv, 'o', color='#d62728', ms=4, mfc='none', label=ref_label)
    ax.set_xlabel('$x$'); ax.set_ylabel('$w(x,0.5)$')
    ax.set_title('(b) vertical-velocity centerline')
    ax.legend(loc='upper left', frameon=False, fontsize=8)

    # (c) KE balance residual (boundary work) + dissipation vs t
    ax = axes[1, 0]
    t = np.array(b["t"])
    ax.plot(t, b["dKEdt"], '-', color='#1f77b4', lw=1.3, label='$dK_h/dt$ (measured)')
    diss = np.array(b["dissipation"])
    ax.plot(t, -diss, '--', color='#2ca02c', lw=1.3, label='$-$dissipation (interior term)')
    ax.plot(t, b["ke_residual"], ':', color='#d62728', lw=1.6, label='residual = lid boundary work')
    ax.set_xlabel('$t$'); ax.set_ylabel('rate')
    ax.set_title('(c) kinetic-energy balance, Thm.~11')
    ax.legend(loc='upper right', frameon=False, fontsize=8)

    # (d) divergence: the CONSERVED weak-constraint l2 norm (Thm 10/11, at CG
    # tolerance) vs the corner-singularity-limited nodal max, both on log scale
    ax = axes[1, 1]
    if has_l2:
        l2 = np.array(b["l2_weak_div"])
        ax.semilogy(t, np.where(l2 > 0, l2, np.nan), '-', color='#1f77b4', lw=1.4,
                    label='$\\|(N_i,\\,\\nabla\\!\\cdot u_h)\\|_2$ (Thm 10/11)')
    mdiv = np.array(b["max_weak_div"])
    ax.semilogy(t, np.where(mdiv > 0, mdiv, np.nan), '--', color='#9467bd', lw=1.2,
                label='nodal max (corner-limited)')
    ax.set_xlabel('$t$'); ax.set_ylabel('divergence measure')
    ax.set_title('(d) discrete incompressibility, Thm.~10/11')
    ax.legend(loc='center right', frameon=False, fontsize=7.5)

    for ax in axes.flat:
        ax.tick_params(labelsize=8)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out_pdf = fig_dir / "fig_s6_cavity_budgets.pdf"
    fig.savefig(out_pdf, dpi=200)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
