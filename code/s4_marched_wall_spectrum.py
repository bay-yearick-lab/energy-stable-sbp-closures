#!/usr/bin/env python3
"""Render the primary wall spectrum from the operators used in time marching.

The source artifact is produced by ``s4_staggered_wallmode.py``.  The figure
deliberately excludes the same-point BMF reconstruction derivative: that map
is a useful reconciliation diagnostic, but it is not the staggered operator
advanced by the BMF wave solver.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import boundary_spline_analysis as ba
from project_paths import RESULTSDIR, ensure_generated_dirs, save_figure


PANELS = ("signed_bias", "contamination", "total_error")
MARCHED_KEYS = ("E", "C", "SBP", "DRP", "BMF-staggered", "BMG")
DISPLAY = {
    "E": ("E6", "E"),
    "C": ("C6", "C"),
    "SBP": ("SBP6", "SBP"),
    "DRP": ("DRP6", "DRP"),
    "BMF-staggered": ("BMF6 (staggered)", "BMF"),
    "BMG": ("BMG6", "BMG"),
}
PANEL_BG = ("#fbf7ee", "#f7f5ef", "#f4f7f5")
PLOT_COLORS = {
    "E": "#727272",
    "C": "#8c6d31",
    "SBP": "#7a6a96",
    "DRP": "#a05a3a",
    "BMF": "#22517a",
    "BMG": "#1f6f6f",
}


def load_marched_curves(path: Path, *, q: int) -> dict[str, list[dict[str, float]]]:
    source = json.loads(path.read_text())
    block = source.get("second_order", {}).get(f"q{q}", {})
    missing = [key for key in MARCHED_KEYS if key not in block]
    if missing:
        raise ValueError("missing marched spectrum channel(s): " + ", ".join(missing))
    curves: dict[str, list[dict[str, float]]] = {}
    for key in MARCHED_KEYS:
        rows = block[key].get("rows", [])
        if not rows:
            raise ValueError(f"marched spectrum channel {key} has no rows")
        required = {"ppw", "rho", "eta", "total"}
        if any(not required.issubset(row) for row in rows):
            raise ValueError(f"marched spectrum channel {key} has incomplete rows")
        curves[key] = rows
    return curves


def threshold_ppw(
    rows: list[dict[str, float]], field: str, tolerance: float, *, target: float = 0.0
) -> float:
    admissible = [
        float(row["ppw"])
        for row in rows
        if abs(float(row[field]) - target) <= tolerance
    ]
    return min(admissible) if admissible else float("nan")


def render(curves: dict[str, list[dict[str, float]]], *, stem: str) -> dict[str, object]:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.5,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.75), sharex=True)
    for ax, tint in zip(axes, PANEL_BG, strict=True):
        ax.set_facecolor(tint)

    for key, rows in curves.items():
        ppw = np.asarray([row["ppw"] for row in rows], dtype=float)
        omega_ratio = 2.0 / ppw
        rho = np.asarray([row["rho"] for row in rows], dtype=float)
        eta = np.asarray([row["eta"] for row in rows], dtype=float)
        total = np.asarray([row["total"] for row in rows], dtype=float)
        label, family = DISPLAY[key]
        style = dict(
            color=PLOT_COLORS[family],
            ls=ba.DEFAULT_LINESTYLES[family],
            lw=1.9,
            label=label,
        )
        axes[0].plot(omega_ratio, rho - 1.0, **style)
        axes[1].plot(omega_ratio, np.maximum(eta, 1.0e-16), **style)
        axes[2].plot(omega_ratio, np.maximum(total, 1.0e-16), **style)

    for level, line_style, shade in (
        (0.0, "-", "0.35"),
        (0.01, "--", "0.48"),
        (-0.01, "--", "0.48"),
        (0.05, ":", "0.62"),
        (-0.05, ":", "0.62"),
    ):
        axes[0].axhline(level, color=shade, lw=0.9, ls=line_style, zorder=0)
    axes[0].text(1.015, 0.01, "1%", transform=axes[0].get_yaxis_transform(),
                 ha="left", va="center", color="0.38", fontsize=8, clip_on=False)
    axes[0].text(1.015, 0.05, "5%", transform=axes[0].get_yaxis_transform(),
                 ha="left", va="center", color="0.50", fontsize=8, clip_on=False)

    for ax in axes[1:]:
        ax.axhline(0.01, color="0.48", lw=0.9, ls="--", zorder=0)
        ax.axhline(0.05, color="0.62", lw=0.9, ls=":", zorder=0)
        ax.text(1.015, 0.01, "1%", transform=ax.get_yaxis_transform(),
                ha="left", va="center", color="0.38", fontsize=8, clip_on=False)
        ax.text(1.015, 0.05, "5%", transform=ax.get_yaxis_transform(),
                ha="left", va="center", color="0.50", fontsize=8, clip_on=False)
        ax.set_yscale("log")
        ax.set_ylim(1.0e-12, 2.0)

    axes[0].set_title(r"(a) signed target response", fontstyle="italic")
    axes[1].set_title(r"(b) orthogonal contamination", fontstyle="italic")
    axes[2].set_title(r"(c) combined wall-mode error", fontstyle="italic")
    axes[0].set_ylabel(r"$\rho_m-1$")
    axes[1].set_ylabel(r"$\eta_m$")
    axes[2].set_ylabel(r"$\sqrt{(\rho_m-1)^2+\eta_m^2}$")
    axes[0].set_ylim(-1.0, 0.12)
    for ax in axes:
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(r"$\omega_m/\pi=2/\mathrm{PPW}$")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 0.995), frameon=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.78, wspace=0.36)
    save_figure(fig, stem, pad_inches=0.03)
    plt.close(fig)

    return {
        "source": "staggered_wallmode_summary.json:second_order:q6",
        "operators": list(curves),
        "excluded_diagnostic": "BMF-reconstruction",
        "threshold_ppw_1pct": {
            key: {
                "response": threshold_ppw(rows, "rho", 0.01, target=1.0),
                "contamination": threshold_ppw(rows, "eta", 0.01),
                "combined": threshold_ppw(rows, "total", 0.01),
            }
            for key, rows in curves.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTSDIR / "staggered_wallmode_summary.json",
    )
    parser.add_argument("--q", type=int, default=6)
    parser.add_argument("--stem", default="fig_s4_marched_wall_spectrum")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=RESULTSDIR / "marched_wall_spectrum_figure_summary.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.q != 6:
        raise SystemExit("the manuscript comparison figure is defined for q=6")
    ensure_generated_dirs()
    curves = load_marched_curves(args.input, q=args.q)
    summary = render(curves, stem=args.stem)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
