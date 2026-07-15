#!/usr/bin/env python3
"""Render the physical route-C BMF lid-driven-cavity validation."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project_paths import RESULTSDIR, ensure_generated_dirs, save_figure


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.0,
    "axes.labelsize": 10.8,
    "axes.titlesize": 11.2,
    "legend.fontsize": 9.2,
    "xtick.labelsize": 9.4,
    "ytick.labelsize": 9.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 180,
    "savefig.dpi": 300,
})

PANEL_BG = "#fbf7ed"
ROUTE_COLORS = ("#a9c1d5", "#7097b7", "#3f6d94", "#22517a")


def main() -> None:
    ensure_generated_dirs()
    record = json.loads((RESULTSDIR / "bmf_hdiv_cavity_summary.json").read_text())
    runs = sorted(record["runs"], key=lambda row: row["npts"])
    u_reference = np.asarray(record["reference"]["u_centerline"], dtype=float)
    v_reference = np.asarray(record["reference"]["v_centerline"], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.75))
    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.grid(color="#d9d2c3", alpha=0.38, linewidth=0.55)

    ax = axes[0]
    ax.plot(
        u_reference[:, 1], u_reference[:, 0], "o", ms=4.8,
        markerfacecolor="white", markeredgecolor="#252525", markeredgewidth=0.9,
        linestyle="none", label="reference data", zorder=5,
    )
    for color, run in zip(ROUTE_COLORS, runs):
        ax.plot(run["u_centerline"], u_reference[:, 0], color=color, lw=1.55,
                label=fr"BMF$_4$, $N={run['npts']}$")
    ax.set_xlabel(r"horizontal velocity $u(1/2,y)$")
    ax.set_ylabel(r"$y$")
    ax.set_title("(a) Vertical centerline")

    ax = axes[1]
    ax.plot(
        v_reference[:, 0], v_reference[:, 1], "o", ms=4.8,
        markerfacecolor="white", markeredgecolor="#252525", markeredgewidth=0.9,
        linestyle="none", zorder=5,
    )
    for color, run in zip(ROUTE_COLORS, runs):
        ax.plot(v_reference[:, 0], run["v_centerline"], color=color, lw=1.55)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"vertical velocity $v(x,1/2)$")
    ax.set_title("(b) Horizontal centerline")

    ax = axes[2]
    grids = np.asarray([run["npts"] for run in runs])
    full = np.asarray([run["centerline"]["combined_relative_l2"] for run in runs])
    interior = np.asarray([
        run["centerline"]["interior_combined_relative_l2"] for run in runs
    ])
    ax.plot(grids, full, "o-", color="#22517a", lw=1.65, ms=5.2,
            label="full centerlines")
    ax.plot(grids, interior, "s--", color="#1f6f6f", lw=1.45, ms=4.8,
            label="interior points")
    ax.set_xticks(grids)
    ax.set_xlabel(r"grid points per direction $N$")
    ax.set_ylabel(r"combined relative $L^2$ discrepancy")
    ax.set_title("(c) reference discrepancy across grids")
    ax.legend(frameon=False, loc="upper right")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, loc="upper center",
               bbox_to_anchor=(0.5, 1.01), columnspacing=1.15, handlelength=2.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88), w_pad=1.6)
    save_figure(fig, "fig_s6_bmf_hdiv_cavity")
    print("wrote fig_s6_bmf_hdiv_cavity.(pdf,png)")


if __name__ == "__main__":
    main()
