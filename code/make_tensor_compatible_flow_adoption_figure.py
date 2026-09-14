"""Render the compatible-flow adoption figure from released certificates."""
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

MANUFACTURED = RESULTSDIR / "manufactured_compatible_comparison.json"
SMOOTH_CAVITY = RESULTSDIR / "smooth_cavity_nitsche_traction.json"

BLUE = "#22517a"
TEAL = "#1f6f6f"
RED = "#a83232"
CREAM = "#fbf7ed"
INK = "#413c36"
GRID = "#d8d0c3"


def _style_axis(axis) -> None:
    axis.set_facecolor(CREAM)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#8a8175")
    axis.tick_params(colors=INK, labelsize=9)
    axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.55)
    axis.axhline(1.0, color="#4b4742", linewidth=1.0, linestyle=":", zorder=2)


def _manufactured_ratios(data: dict, targets: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    state = []
    total_time = []
    for target in targets:
        bmf = data["design_selections"]["bmf"][target]
        sem = data["design_selections"]["compatible_sem"][target]
        state.append(sem["state_storage_bytes"] / bmf["state_storage_bytes"])
        total_time.append(sem["total_seconds"] / bmf["total_seconds"])
    return np.asarray(state), np.asarray(total_time)


def _cavity_ratios(data: dict) -> tuple[np.ndarray, np.ndarray]:
    error_advantage = []
    resolved_margin = []
    for score in sorted(data["candidate_scores"], key=lambda item: item["reynolds"]):
        candidates = score["candidate_scores"]
        bmf = candidates["smooth_bmf_p8_k2"]
        sem = candidates["smooth_sem_p8_e2"]
        bmf_error = max(
            bmf[reference]["wall_closure_joint_error"]
            for reference in ("against_bmf_reference", "against_sem_reference")
        )
        sem_error = max(
            sem[reference]["wall_closure_joint_error"]
            for reference in ("against_bmf_reference", "against_sem_reference")
        )
        ordering = score["orderings"]["equal_state_bmf_vs_sem"]
        error_advantage.append(sem_error / bmf_error)
        resolved_margin.append(
            ordering["minimum_margin"] / ordering["reference_disagreement"]
        )
    return np.asarray(error_advantage), np.asarray(resolved_margin)


def _resource_ratios(data: dict) -> tuple[np.ndarray, np.ndarray]:
    rows = {
        (float(row["reynolds"]), row["name"]): row
        for row in data["rows"]
        if row["name"] in {"smooth_bmf_p6_k2", "smooth_bmf_p8_k2", "smooth_sem_p8_e2"}
    }
    comparisons = (
        (100.0, "smooth_bmf_p8_k2"),
        (400.0, "smooth_bmf_p8_k2"),
        (100.0, "smooth_bmf_p6_k2"),
    )
    state = []
    total_time = []
    for reynolds, bmf_name in comparisons:
        bmf = rows[(reynolds, bmf_name)]
        sem = rows[(reynolds, "smooth_sem_p8_e2")]
        state.append(sem["state_storage_bytes"] / bmf["state_storage_bytes"])
        total_time.append(sem["total_seconds"] / bmf["total_seconds"])
    return np.asarray(state), np.asarray(total_time)


def main() -> None:
    ensure_generated_dirs()
    manufactured = json.loads(MANUFACTURED.read_text())
    cavity = json.loads(SMOOTH_CAVITY.read_text())

    figure, axes = plt.subplots(1, 3, figsize=(12.4, 3.7))
    figure.patch.set_facecolor("white")
    for axis in axes:
        _style_axis(axis)

    width = 0.34
    targets = ("1e-05", "1e-08")
    state, total_time = _manufactured_ratios(manufactured, targets)
    x = np.arange(len(targets))
    axes[0].bar(x - width / 2, state, width, color=BLUE, label="state")
    axes[0].bar(x + width / 2, total_time, width, color=TEAL, label="total time")
    axes[0].set_xticks(x, (r"$10^{-5}$", r"$10^{-8}$"))
    axes[0].set_ylabel("SEM / BMF resource")
    axes[0].set_ylim(0, 4.4)
    axes[0].set_title("(a) matched accuracy", loc="left", fontsize=10.5, fontstyle="italic")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")

    error_advantage, resolved_margin = _cavity_ratios(cavity)
    x = np.arange(2)
    axes[1].bar(x - width / 2, error_advantage, width, color=BLUE, label="wall error")
    axes[1].bar(x + width / 2, resolved_margin, width, color=RED, label="margin / reference uncertainty")
    axes[1].set_xticks(x, (r"$\mathrm{Re}=100$", r"$\mathrm{Re}=400$"))
    axes[1].set_ylabel("resolved BMF advantage")
    axes[1].set_ylim(0, 7.0)
    axes[1].set_title("(b) held-out wall closure", loc="left", fontsize=10.5, fontstyle="italic")
    axes[1].legend(frameon=False, fontsize=8.1, loc="upper right")

    state, total_time = _resource_ratios(cavity)
    x = np.arange(3)
    axes[2].bar(x - width / 2, state, width, color=BLUE, label="state")
    axes[2].bar(x + width / 2, total_time, width, color=TEAL, label="total time")
    axes[2].set_xticks(x, ("Re 100\nequal state", "Re 400\nequal state", "Re 100\nreduced state"))
    axes[2].set_ylabel("SEM / BMF resource")
    axes[2].set_ylim(0, 2.0)
    axes[2].set_title("(c) complete flow mission", loc="left", fontsize=10.5, fontstyle="italic")
    axes[2].legend(frameon=False, fontsize=8.5, loc="upper left")

    figure.subplots_adjust(left=0.065, right=0.99, top=0.9, bottom=0.22, wspace=0.34)
    for suffix in ("pdf", "png"):
        path = FIGDIR / f"fig_s6_tensor_compatible_flow_adoption.{suffix}"
        figure.savefig(
            path,
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(figure)
    print("wrote compatible-flow adoption figure")


if __name__ == "__main__":
    main()
