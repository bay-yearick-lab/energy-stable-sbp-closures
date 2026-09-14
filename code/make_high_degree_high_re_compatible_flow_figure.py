#!/usr/bin/env python3
"""Render the arbitrary-degree and high-Reynolds compatible-flow summary."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

from project_paths import FIGDIR, RESULTSDIR, ensure_generated_dirs


ALGEBRA = RESULTSDIR / "high_degree_compatible_algebra.json"
MANUFACTURED = RESULTSDIR / "high_degree_manufactured_comparison.json"
TIMING = RESULTSDIR / "high_re_matched_accuracy_timing.json"

BLUE = "#22517a"
TEAL = "#1f6f6f"
RED = "#a83232"
GREEN = "#1c7c34"
CREAM = "#fbf7ed"
INK = "#413c36"
GRID = "#d8d0c3"


def _style_axis(axis, *, logarithmic: bool = False) -> None:
    """Apply the manuscript visual grammar to one data panel."""
    axis.set_facecolor(CREAM)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#8a8175")
    axis.tick_params(colors=INK, labelsize=8.7)
    axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.55)
    if logarithmic:
        axis.set_yscale("log")


def _condition_rows(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the maximum one-dimensional mass condition by degree and family."""
    degrees = np.asarray(sorted({int(row["degree"]) for row in record["rows"]}))
    values = {}
    for method in ("bmf", "compatible_sem"):
        values[method] = np.asarray(
            [
                max(
                    float(row["mass0_condition_number"]),
                    float(row["mass1_condition_number"]),
                )
                for degree in degrees
                for row in record["rows"]
                if row["method"] == method and row["degree"] == degree
            ]
        )
    return degrees, values["bmf"], values["compatible_sem"]


def _manufactured_rows(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the two-span joint error by degree and family."""
    degrees = np.asarray(sorted({int(row["degree"]) for row in record["rows"]}))
    values = {}
    for method in ("bmf", "compatible_sem"):
        values[method] = np.asarray(
            [
                max(
                    float(row["relative_velocity_l2_error"]),
                    float(row["relative_pressure_l2_error"]),
                    float(row["relative_wall_traction_l2_error"]),
                )
                for degree in degrees
                for row in record["rows"]
                if row["method"] == method and row["degree"] == degree
                and row["elements"] == 2
            ]
        )
    return degrees, values["bmf"], values["compatible_sem"]


def _resource_rows(record: dict) -> tuple[list[str], dict[str, np.ndarray]]:
    """Return matched-accuracy comparator-to-BMF resource factors."""
    selections = (
        record["resource_factors"]["1000"]["0.02"],
        record["resource_factors"]["3200"]["0.05"],
    )
    values = {
        "state": np.asarray(
            [row["comparator_to_bmf_state_factor"] for row in selections]
        ),
        "iterations": np.asarray(
            [row["comparator_to_bmf_iteration_factor"] for row in selections]
        ),
        "mission time": np.asarray(
            [row["comparator_to_bmf_time_factor"] for row in selections]
        ),
        "compact factors": np.asarray(
            [row["comparator_to_bmf_operator_factor"] for row in selections]
        ),
    }
    return [r"Re 1000, $2\%$", r"Re 3200, $5\%$"], values


def main() -> None:
    """Render the three-panel source-bound summary."""
    ensure_generated_dirs()
    algebra = json.loads(ALGEBRA.read_text())
    manufactured = json.loads(MANUFACTURED.read_text())
    timing = json.loads(TIMING.read_text())

    figure, axes = plt.subplots(1, 3, figsize=(12.6, 3.75))
    figure.patch.set_facecolor("white")
    _style_axis(axes[0], logarithmic=True)
    _style_axis(axes[1], logarithmic=True)
    _style_axis(axes[2])

    degrees, bmf_condition, sem_condition = _condition_rows(algebra)
    axes[0].plot(degrees, bmf_condition, "o-", color=BLUE, label="BMF")
    axes[0].plot(degrees, sem_condition, "s-", color=TEAL, label="compatible SEM")
    axes[0].set_xticks((2, 4, 6, 8, 10, 12))
    axes[0].set_xlabel("degree")
    axes[0].set_ylabel(r"maximum $\kappa_2(M)$")
    axes[0].set_title(
        "(a) coefficient conditioning",
        loc="left",
        fontsize=10.5,
        fontstyle="italic",
    )

    degrees, bmf_error, sem_error = _manufactured_rows(manufactured)
    axes[1].plot(degrees, bmf_error, "o-", color=BLUE, label="BMF")
    axes[1].plot(degrees, sem_error, "s-", color=TEAL, label="compatible SEM")
    axes[1].set_xticks(degrees)
    axes[1].set_xlabel("degree")
    axes[1].set_ylabel("two-index joint error")
    axes[1].set_title(
        "(b) manufactured wall flow",
        loc="left",
        fontsize=10.5,
        fontstyle="italic",
    )

    labels, resources = _resource_rows(timing)
    x = np.arange(len(labels))
    width = 0.18
    colors = (BLUE, RED, TEAL, GREEN)
    for offset, ((name, values), color) in enumerate(zip(resources.items(), colors)):
        axes[2].bar(
            x + (offset - 1.5) * width,
            values,
            width,
            color=color,
            label=name,
        )
    axes[2].axhline(1.0, color="#4b4742", linewidth=1.0, linestyle=":")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("compatible SEM / BMF")
    axes[2].set_ylim(0.0, 1.75)
    axes[2].set_title(
        "(c) matched flow resources",
        loc="left",
        fontsize=10.5,
        fontstyle="italic",
    )

    handles, legend_labels = [], []
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in legend_labels:
                handles.append(handle)
                legend_labels.append(label)
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=len(legend_labels),
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.99,
        top=0.91,
        bottom=0.25,
        wspace=0.34,
    )
    for suffix in ("pdf", "png"):
        figure.savefig(
            FIGDIR / f"fig_s6_high_degree_high_re_compatible_flow.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
