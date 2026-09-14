#!/usr/bin/env python3
"""Optimized native-reconstruction traction frontier on archived surfaces."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from project_paths import RESULTSDIR, ensure_generated_dirs, save_figure

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.7,
    "axes.labelsize": 10.4,
    "axes.titlesize": 10.8,
    "legend.fontsize": 8.9,
    "xtick.labelsize": 9.1,
    "ytick.labelsize": 9.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.14,
    "grid.linewidth": 0.5,
    "figure.dpi": 180,
    "savefig.dpi": 300,
})

SPLINE_COLOR = "#22517a"
SEM_COLOR = "#a83232"
SEM_CLOUD = "#8c8c86"
LINESTYLES = {"median": "-", "p95": "--", "worst": ":"}
SURFACES = (
    ("circle_small", "(a) Small circle"),
    ("circle_large", "(b) Large circle"),
    ("star", "(c) Star-shaped curve"),
)


def _sem_envelope(rows: list[dict], statistic: str) -> tuple[np.ndarray, np.ndarray]:
    by_state = {}
    for row in rows:
        state = row["state_size_1d"]
        by_state[state] = min(by_state.get(state, np.inf), row[statistic])
    states = np.array(sorted(by_state))
    errors = np.minimum.accumulate(np.array([by_state[state] for state in states]))
    return states, errors


def main() -> None:
    ensure_generated_dirs()
    summary = json.loads(
        (RESULTSDIR / "offgrid_sem_geometry_frontier_summary.json").read_text()
    )
    traction = [row for row in summary["rows"] if row["metric"] == "traction"]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.65), sharex=True, sharey=True)

    for axis, (surface, title) in zip(axes, SURFACES, strict=True):
        sem_rows = [
            row for row in traction
            if row["entrant"] == "sem" and row["surface"] == surface
        ]
        unique_sem = {
            (row["degree"], row["elements"]): row for row in sem_rows
        }.values()
        axis.scatter(
            [row["state_size_1d"] for row in unique_sem],
            [row["median"] for row in unique_sem],
            s=8, color=SEM_CLOUD, alpha=0.23, linewidths=0, zorder=1,
        )
        spline_rows = sorted(
            (
                row for row in traction
                if row["entrant"] == "spline_native_BMF6_BMG6"
                and row["surface"] == surface
            ),
            key=lambda row: row["state_size_1d"],
        )
        for statistic, linestyle in LINESTYLES.items():
            states, errors = _sem_envelope(list(unique_sem), statistic)
            axis.semilogy(
                states, errors, linestyle=linestyle, color=SEM_COLOR,
                linewidth=1.55, zorder=2,
            )
            axis.semilogy(
                [row["state_size_1d"] for row in spline_rows],
                [row[statistic] for row in spline_rows],
                linestyle=linestyle, color=SPLINE_COLOR, marker="o",
                markersize=4.0, linewidth=1.65, zorder=3,
            )
        axis.set_title(title)
        axis.set_xlabel("one-dimensional state size")
        axis.set_xlim(2, 63)
        axis.set_xticks((3, 19, 37, 61))
        axis.set_ylim(8.0e-13, 3.0)

    axes[0].set_ylabel("relative traction error")
    legend = [
        Line2D([0], [0], color=SPLINE_COLOR, marker="o", lw=1.65,
               markersize=4.0, label=r"BMF$_6$/BMG$_6$ native reconstruction"),
        Line2D([0], [0], color=SEM_COLOR, lw=1.55,
               label="optimized continuous GLL SEM envelope"),
        Line2D([0], [0], color="#333333", lw=1.4, linestyle="-", label="median"),
        Line2D([0], [0], color="#333333", lw=1.4, linestyle="--", label="95th percentile"),
        Line2D([0], [0], color="#333333", lw=1.4, linestyle=":", label="worst case"),
    ]
    fig.legend(
        handles=legend, loc="upper center", ncol=5, frameon=False,
        bbox_to_anchor=(0.5, 1.01), handlelength=2.2, columnspacing=1.25,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88), w_pad=1.5)
    save_figure(fig, "fig_s6_offgrid_sem_geometry_frontier")
    print("wrote fig_s6_offgrid_sem_geometry_frontier")


if __name__ == "__main__":
    main()
