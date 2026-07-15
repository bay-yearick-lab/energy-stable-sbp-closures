"""Figure fig_s6_wallclock_cost: measured error versus CPU cost.

A measured error-versus-cost curve for the complete heat-cavity solve (operator
assembly + time march at each family's own tier-B stable step under the
metric-orthogonal enforcement). One curve per matched-order family across the
grid ladder; the accuracy a nodal family cannot reach at any grid shown is the
point. CPU process time (single thread) is the load-robust primary metric.
House style: serif STIX, right-end curve labels, no slope triangles.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project_paths import RESULTSDIR, ensure_generated_dirs, save_figure

ensure_generated_dirs()
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.5, "axes.labelsize": 11.5, "axes.titlesize": 12.0,
    "legend.fontsize": 9.5, "xtick.labelsize": 9.6, "ytick.labelsize": 9.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 180, "savefig.dpi": 300,
})

COLORS = {"E6": "#7a7a7a", "C6": "#1f77b4", "SBP6": "#9467bd",
          "DRP6": "#ff7f0e", "BMF6": "#d62728", "BMG6": "#2ca02c"}
DISPLAY = {"E6": "explicit", "C6": "compact", "SBP6": "SBP-SAT",
           "DRP6": "DRP", "BMF6": "BMF", "BMG6": "BMG"}


def main() -> None:
    d = json.loads((RESULTSDIR / "wallclock_cost.json").read_text())
    by = {}
    for c in d["cases"]:
        by.setdefault(c["label"], []).append(c)
    for k in by:
        by[k].sort(key=lambda c: c["npts"])

    # small vertical label offsets to separate families whose end points
    # nearly coincide (compact and DRP land on top of each other)
    yfac = {"E6": 1.0, "C6": 1.35, "SBP6": 1.0, "DRP6": 0.62,
            "BMF6": 0.78, "BMG6": 1.28}
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for lab, rows in by.items():
        t = np.array([r["median_cpu_seconds"] for r in rows])
        e = np.array([r["rel_l2"] for r in rows])
        ax.plot(t, e, "-o", ms=4.5, lw=1.8, color=COLORS[lab])
        ax.annotate(DISPLAY[lab], xy=(t[-1] * 1.06, e[-1] * yfac[lab]),
                    color=COLORS[lab], fontsize=9.2, va="center", ha="left")

    # neutral reference line only; the verdict belongs to the surrounding prose
    ax.axhline(1e-3, color="#999999", ls=":", lw=1.0)
    ax.annotate("$10^{-3}$ reference", xy=(1.6e-3, 1.2e-3),
                fontsize=8.8, color="#666666")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("CPU cost per solve [s, single thread, median of 5]")
    ax.set_ylabel(r"relative $L^2$ error at $T=2$")
    ax.set_xlim(1.5e-3, 0.18)
    ax.set_ylim(1e-6, 1e1)
    fig.tight_layout()
    save_figure(fig, "fig_s6_wallclock_cost")
    print("wrote fig_s6_wallclock_cost")


if __name__ == "__main__":
    main()
