"""Render the symmetric hp wall-observable frontier figure."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))

from project_paths import FIGDIR, RESULTSDIR, ensure_generated_dirs

SUMMARY = RESULTSDIR / "symmetric_hp_wall_frontier_summary.json"
AUDIT = RESULTSDIR / "symmetric_hp_wall_frontier_normalization_audit.json"

COLORS = {
    "normal_derivative_l2": "#22517a",
    "flux_l2": "#1f6f6f",
    "traction_l2": "#a83232",
}
LABELS = {
    "normal_derivative_l2": "normal derivative",
    "flux_l2": "flux",
    "traction_l2": "traction",
}


def _style_axis(axis) -> None:
    axis.set_facecolor("#fbf7ed")
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#8a8175")
    axis.tick_params(colors="#413c36", labelsize=9)
    axis.grid(axis="y", color="#d8d0c3", linewidth=0.7, alpha=0.55)


def main() -> None:
    ensure_generated_dirs()
    summary = json.loads(SUMMARY.read_text())
    audit = json.loads(AUDIT.read_text())
    rows = [row for row in summary["paired"] if row["objective"] == "wall"]
    metrics = tuple(COLORS)

    figure, axes = plt.subplots(1, 3, figsize=(12.4, 3.75))
    figure.patch.set_facecolor("white")
    for axis in axes:
        _style_axis(axis)

    offsets = np.linspace(-0.18, 0.18, 16)
    for metric_index, metric in enumerate(metrics):
        factors = []
        for row in rows:
            factors.extend(
                row["sem_to_spline_error_factor"][metric][statistic]
                for statistic in ("median", "p95", "worst")
            )
        x = metric_index + np.tile(offsets, 3)
        axes[0].scatter(
            x, factors, s=24, color=COLORS[metric], edgecolor="white",
            linewidth=0.45, alpha=0.84, label=LABELS[metric], zorder=3,
        )
    axes[0].axhline(1.0, color="#4b4742", linewidth=1.0, linestyle=":")
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.3, 6.0e4)
    axes[0].set_xticks(range(3), [LABELS[metric] for metric in metrics], rotation=17)
    axes[0].set_ylabel("SEM / spline error")
    axes[0].set_title("(a) wall-selected frontier", loc="left", fontsize=10.5, fontstyle="italic")

    categories = (
        "spline_only_below_tolerance", "both_below_tolerance",
        "both_above_tolerance", "sem_only_below_tolerance",
    )
    category_labels = ("spline only", "both", "neither", "SEM only")
    category_colors = ("#22517a", "#9fb8ad", "#d2c9b8", "#a83232")
    bottom = np.zeros(3)
    x = np.arange(3)
    for category, label, color in zip(categories, category_labels, category_colors, strict=True):
        values = np.asarray([
            audit["tolerance_materiality"][metric][category] for metric in metrics
        ])
        axes[1].bar(x, values, bottom=bottom, width=0.68, color=color, label=label)
        bottom += values
    axes[1].set_ylim(0, 50)
    axes[1].set_xticks(x, [LABELS[metric] for metric in metrics], rotation=17)
    axes[1].set_ylabel("aggregate cells out of 48")
    axes[1].set_title("(b) declared tolerance", loc="left", fontsize=10.5, fontstyle="italic")
    axes[1].legend(
        frameon=False, fontsize=8.2, ncol=2, loc="upper center",
        bbox_to_anchor=(0.5, -0.22), handlelength=1.2, columnspacing=1.0,
    )

    for metric, marker in (("normal_derivative_l2", "o"), ("traction_l2", "s")):
        span_ratio = np.asarray([
            row["spline_config"]["elements"] / row["sem_config"]["elements"]
            for row in rows
        ])
        factor = np.asarray([
            row["sem_to_spline_error_factor"][metric]["median"] for row in rows
        ])
        exceptional = np.asarray([
            row["resource_kind"] == "mission" and row["resource_cap"] == 20_000_000
            and row["query_count"] == 100 for row in rows
        ])
        axes[2].scatter(
            span_ratio[~exceptional], factor[~exceptional], marker=marker, s=38,
            color=COLORS[metric], edgecolor="white", linewidth=0.55,
            alpha=0.86, label=LABELS[metric], zorder=3,
        )
        axes[2].scatter(
            span_ratio[exceptional], factor[exceptional], marker=marker, s=48,
            facecolor="white", edgecolor=COLORS[metric], linewidth=1.3, zorder=4,
        )
    axes[2].axhline(1.0, color="#4b4742", linewidth=1.0, linestyle=":")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlim(1.05, 11.0)
    axes[2].set_ylim(0.3, 3.0e4)
    axes[2].set_xlabel("spline spans / SEM elements")
    axes[2].set_ylabel("median SEM / spline error")
    axes[2].set_title("(c) continuity dividend", loc="left", fontsize=10.5, fontstyle="italic")
    axes[2].legend(frameon=False, fontsize=8.4, loc="upper left")

    figure.subplots_adjust(left=0.065, right=0.99, top=0.9, bottom=0.26, wspace=0.34)
    for suffix in ("pdf", "png"):
        path = FIGDIR / f"fig_s6_symmetric_hp_wall_frontier.{suffix}"
        figure.savefig(path, dpi=220 if suffix == "png" else None, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    print("wrote symmetric hp wall frontier figure")


if __name__ == "__main__":
    main()
