#!/usr/bin/env python3
"""Render the scalar, truncated-kernel, and full-propagator verdict."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project_paths import RESULTSDIR, ensure_generated_dirs, save_figure


SUMMARIES = (
    ("scalar RMS", RESULTSDIR / "wall_spectrum_heldout_summary.json"),
    ("truncated kernel", RESULTSDIR / "wall_spectrum_kernel_revision_summary.json"),
    ("full propagator", RESULTSDIR / "wall_spectrum_full_propagator_summary.json"),
)
METRICS = ("phase_shift", "waveform_distortion", "leakage")
METRIC_LABELS = ("phase", "distortion", "leakage")
COLORS = ("#22517a", "#a83232", "#1f6f6f")


def main() -> int:
    ensure_generated_dirs()
    records = [(label, json.loads(path.read_text())) for label, path in SUMMARIES]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.4,
        "axes.labelsize": 10.2,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.75))
    for axis, tint in zip(axes, ("#fbf7ee", "#f7f5ef", "#f4f7f5"), strict=True):
        axis.set_facecolor(tint)
        axis.grid(alpha=0.16, linewidth=0.5)

    positions = np.arange(len(records))
    width = 0.22
    for metric_index, (metric, label, color) in enumerate(
        zip(METRICS, METRIC_LABELS, COLORS, strict=True)
    ):
        medians = [record["residual_summary"][metric]["median"] for _name, record in records]
        axes[0].bar(
            positions + (metric_index - 1) * width,
            medians,
            width,
            color=color,
            alpha=0.82,
            label=label,
        )
    axes[0].axhline(0.20, color="0.35", lw=0.9, ls=":")
    axes[0].text(1.52, 0.20, "20%", va="center", ha="left", fontsize=8.2)
    axes[0].set_yscale("log")
    axes[0].set_xticks(positions, [name for name, _record in records], rotation=16)
    axes[0].set_ylabel("median relative residual")
    axes[0].set_title("(a) quantitative calibration", fontstyle="italic")
    axes[0].legend(frameon=False, loc="upper right")

    full = records[-1][1]
    markers = {"phase_shift": "o", "waveform_distortion": "s", "leakage": "^"}
    values = []
    for metric, label, color in zip(METRICS, METRIC_LABELS, COLORS, strict=True):
        predicted = np.asarray([row["predicted"][metric] for row in full["cases"]])
        actual = np.asarray([row["actual"][metric] for row in full["cases"]])
        values.extend(predicted[predicted > 0.0])
        values.extend(actual[actual > 0.0])
        axes[1].scatter(
            predicted,
            actual,
            s=28,
            marker=markers[metric],
            color=color,
            alpha=0.82,
            label=label,
        )
    lower = min(values) / 2.0
    upper = max(values) * 2.0
    axes[1].plot([lower, upper], [lower, upper], color="0.35", lw=0.9)
    axes[1].plot([lower, upper], [0.8 * lower, 0.8 * upper], color="0.62", lw=0.7, ls=":")
    axes[1].plot([lower, upper], [1.2 * lower, 1.2 * upper], color="0.62", lw=0.7, ls=":")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(lower, upper)
    axes[1].set_ylim(lower, upper)
    axes[1].set_xlabel("full-propagator prediction")
    axes[1].set_ylabel("held-out RK4 measurement")
    axes[1].set_title("(b) third untouched family", fontstyle="italic")
    axes[1].legend(frameon=False, loc="lower right")

    checks = ("quantitative_pass", "threshold_side_pass", "family_ordering_pass")
    check_labels = ("quantitative", "threshold side", "family order")
    matrix = np.asarray([
        [1.0 if record["decision"][check] else 0.0 for check in checks]
        for _name, record in records
    ])
    axes[2].imshow(
        matrix,
        cmap=matplotlib.colors.ListedColormap(["#ead8d1", "#cfe1d6"]),
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    axes[2].set_xticks(np.arange(3), check_labels, rotation=16)
    axes[2].set_yticks(np.arange(3), [name for name, _record in records])
    axes[2].set_title("(c) frozen promotion contract", fontstyle="italic")
    axes[2].grid(False)
    for row in range(3):
        for column in range(3):
            axes[2].text(
                column,
                row,
                "pass" if matrix[row, column] else "fail",
                ha="center",
                va="center",
                color="#1c4730" if matrix[row, column] else "#7b3128",
                fontsize=8.7,
            )

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.23, top=0.90, wspace=0.36)
    save_figure(fig, "fig_s6_wall_spectrum_predictor_validation", pad_inches=0.03)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
