#!/usr/bin/env python3
"""Figure for the spectral-element head-to-head: heat ladder at matched DOFs
(error vs DOF and error vs measured CPU) and the long-time wave verdict."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project_paths import FIGDIR, RESULTSDIR, ensure_generated_dirs, save_figure

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.0,
    "axes.labelsize": 10.8,
    "axes.titlesize": 11.2,
    "legend.fontsize": 9.5,
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

COLORS = {"SBP6": "#8c6d31", "BMF6": "#22517a", "BMG6": "#1f6f6f", "SEM6": "#4f7f3f"}
MARKERS = {"SBP6": "s", "BMF6": "o", "BMG6": "D", "SEM6": "^"}
ORDER = ("SBP6", "SEM6", "BMF6", "BMG6")


def main() -> None:
    ensure_generated_dirs()
    d = json.loads((RESULTSDIR / "sem_headtohead_summary.json").read_text())
    heat = [r for r in d["heat"] if "rel_l2_final" in r]
    wave = [r for r in d["wave"] if "rel_l2_final" in r]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.4, 3.55))

    for fam in ORDER:
        rows = sorted((r for r in heat if r["family"] == fam), key=lambda r: r["npts"])
        if not rows:
            continue
        n = [r["dofs"] for r in rows]
        e = [r["rel_l2_final"] for r in rows]
        c = [max(r["cpu_seconds"], 1e-6) for r in rows]
        ax1.loglog(n, e, "-", marker=MARKERS[fam], ms=5, lw=1.7,
                   color=COLORS[fam], label=fam.replace("6", "$6$"))
        ax2.loglog(c, e, "-", marker=MARKERS[fam], ms=5, lw=1.7, color=COLORS[fam])

    ax1.set_xlabel("degrees of freedom (matched across entrants)")
    ax1.set_ylabel(r"relative $L^2$ error at $t_{\mathrm{end}}$")
    ax1.set_title("(a) Heat decay, error against DOFs")
    ax1.legend(frameon=False, loc="lower left", handlelength=1.6)

    ax2.set_xlabel("measured CPU seconds per march (single thread)")
    ax2.set_title("(b) Heat decay, error against cost")

    fams = [r["family"] for r in sorted(wave, key=lambda r: r["rel_l2_final"], reverse=True)]
    errs = [r["rel_l2_final"] for r in sorted(wave, key=lambda r: r["rel_l2_final"], reverse=True)]
    ax3.bar(range(len(fams)), errs,
            color=[COLORS[f] for f in fams], width=0.62)
    ax3.set_yscale("log")
    ax3.set_xticks(range(len(fams)))
    ax3.set_xticklabels([f.replace("6", "$6$") for f in fams])
    ax3.set_ylabel(r"coupled relative $L^2$ error at six transits")
    ax3.set_title("(c) Closed-box acoustics, $N=43$")
    fig.tight_layout(w_pad=1.6)
    save_figure(fig, "fig_s6_sem_headtohead")
    print("wrote fig_s6_sem_headtohead")


if __name__ == "__main__":
    main()
