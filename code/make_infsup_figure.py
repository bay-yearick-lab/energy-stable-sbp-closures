"""Figure fig_s5_infsup_sweep: the deep discrete inf-sup sweep.

Reads code/results/infsup_sweep.json (produced by make_infsup_sweep.py) and
renders the two-panel manuscript figure that backs the inf-sup theorem block of
Section 5:

    (a) beta_h versus N (log x) for velocity degrees p = 2..8 at equal order,
        flat against the continuous constant pi;
    (b) |beta_h / pi - 1| versus N (log-log), showing the eigenvalue-
        superconvergent decay toward the roundoff floor, with the observed
        fine-grid slope printed at each curve end.

House style: serif STIX math, pdf.fonttype 42 (no Type-3 fonts), curves
labelled at their right ends in the curve colour (never slope triangles),
legend overhead, zero overlapping text. Saves both PNG and PDF via
project_paths.save_figure.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project_paths import RESULTSDIR, ensure_generated_dirs, save_figure

ensure_generated_dirs()
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.2,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12.0,
        "legend.fontsize": 9.6,
        "xtick.labelsize": 9.4,
        "ytick.labelsize": 9.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
        "savefig.dpi": 300,
    }
)

# A perceptually ordered sequence from the viridis-family so p = 2..8 reads as a
# ladder; distinct enough that the eight curves separate in panel (b).
DEGREE_COLORS = {
    2: "#440154",
    3: "#46327e",
    4: "#365c8d",
    5: "#277f8e",
    6: "#1fa187",
    7: "#4ac16d",
    8: "#a0da39",
}

FLOOR = 1.0e-12  # double-precision eigenvalue roundoff floor (see JSON residuals)


def _load():
    with open(RESULTSDIR / "infsup_sweep.json") as fh:
        data = json.load(fh)
    eq = data["sweep"]["equal_order"]
    pi = data["sweep"]["pi"]
    curves: dict[int, dict[str, np.ndarray]] = {}
    for cell in eq:
        p = cell["pv"]
        curves.setdefault(p, {"N": [], "beta": []})
        curves[p]["N"].append(cell["Nv"])
        curves[p]["beta"].append(cell["beta"])
    for p, c in curves.items():
        order = np.argsort(c["N"])
        c["N"] = np.asarray(c["N"], dtype=float)[order]
        c["beta"] = np.asarray(c["beta"], dtype=float)[order]
    return curves, pi, data


def _fine_slope(logN: np.ndarray, logdev: np.ndarray) -> float | None:
    """Log-log slope on the pre-floor portion of a decay curve.

    Only points above the roundoff floor carry a meaningful rate; a curve that
    has already reached the floor at its coarsest grid returns ``None``.
    """
    mask = logdev > np.log10(10.0 * FLOOR)  # keep well clear of the roundoff floor
    if mask.sum() < 2:
        return None
    lx, ly = logN[mask], logdev[mask]
    slope, _ = np.polyfit(lx, ly, 1)
    return float(slope)


def main() -> None:
    curves, pi, data = _load()
    degrees = sorted(curves)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 4.8))

    # ----- Panel (a): beta_h vs N, flat at pi ---------------------------------
    # Plot the deviation from pi in micro-units so the flat line reads cleanly
    # and the two coarse-grid dips (p = 2, 3) stay on-panel.
    scale = 1.0e6
    for p in degrees:
        c = curves[p]
        axA.plot(
            c["N"],
            (c["beta"] - pi) * scale,
            "-o",
            color=DEGREE_COLORS[p],
            lw=1.5,
            ms=3.6,
            mew=0.0,
            zorder=3,
        )
    axA.axhline(0.0, color="0.35", lw=1.0, ls=(0, (5, 3)), zorder=1)
    axA.set_xscale("log")
    axA.set_xlabel(r"pressure basis count per direction $N$")
    axA.set_ylabel(r"$(\beta_h-\pi)\times 10^{6}$")
    axA.set_xlim(7.5, 620.0)
    axA.set_ylim(-5.1, 0.9)
    # The single coarsest cell (p=2, N=9) is the deepest dip; label the limit
    # line and the approach direction without any slope marker.
    axA.text(
        560.0,
        0.30,
        r"$\beta_h=\pi$ (continuous limit)",
        color="0.30",
        ha="right",
        va="bottom",
        fontsize=10.0,
    )
    axA.annotate(
        r"coarsest cell $p=2,\ N=9$",
        xy=(9.0, (curves[2]["beta"][0] - pi) * scale),
        xytext=(24.0, -4.30),
        color=DEGREE_COLORS[2],
        fontsize=9.2,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=DEGREE_COLORS[2], lw=0.8, shrinkB=3.0),
    )
    axA.text(
        330.0,
        -3.55,
        "approach from below under refinement",
        color="0.35",
        fontsize=8.8,
        ha="center",
        va="center",
    )
    axA.set_title(r"(a) inf-sup constant, equal order")

    # ----- Panel (b): |beta_h/pi - 1| vs N, log-log ---------------------------
    # A curve carries a meaningful decay rate only while it is above the
    # roundoff floor at more than one grid; higher degrees reach the floor at
    # the coarsest grid and carry no slope, so no spurious rate is printed.
    slope_anchor_frac = {2: 0.42, 3: 0.30, 4: 0.20}
    for p in degrees:
        c = curves[p]
        dev = np.abs(c["beta"] / pi - 1.0)
        dev_plot = np.maximum(dev, 0.5 * FLOOR)  # keep exact zeros on the log axis
        axB.plot(
            c["N"],
            dev_plot,
            "-o",
            color=DEGREE_COLORS[p],
            lw=1.5,
            ms=3.6,
            mew=0.0,
            zorder=3,
        )
        logN = np.log10(c["N"])
        logdev = np.log10(np.maximum(dev, 1e-300))
        slope = _fine_slope(logN, logdev)
        if slope is not None and p in slope_anchor_frac:
            # Print the observed slope along the descending part of the curve,
            # in the curve colour, clear of the crowded floor region.
            f = slope_anchor_frac[p]
            xi = f * (np.log10(c["N"][0]) + np.log10(c["N"][-1]))
            xa = 10.0 ** (0.5 * (np.log10(c["N"][0]) + np.log10(c["N"][3])))
            ya = 10.0 ** np.interp(np.log10(xa), logN, logdev)
            axB.annotate(
                rf"$p={p}\!:\ \propto N^{{{slope:.1f}}}$",
                xy=(xa, ya),
                xytext=(6.0, 8.0),
                textcoords="offset points",
                color=DEGREE_COLORS[p],
                fontsize=8.9,
                ha="left",
                va="bottom",
            )
    axB.axhline(FLOOR, color="0.45", lw=1.0, ls=":", zorder=1)
    axB.text(
        9.0,
        FLOOR * 1.4,
        "double-precision floor",
        color="0.40",
        fontsize=8.6,
        ha="left",
        va="bottom",
    )
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlabel(r"pressure basis count per direction $N$")
    axB.set_ylabel(r"$|\beta_h/\pi-1|$")
    axB.set_xlim(7.5, 700.0)
    axB.set_ylim(2.0e-13, 5.0e-6)
    axB.set_title(r"(b) approach to the continuous limit")

    # Overhead legend keyed by degree, kept off both panels.
    handles = [
        plt.Line2D([], [], color=DEGREE_COLORS[p], lw=1.7, marker="o", ms=3.6, mew=0.0)
        for p in degrees
    ]
    labels = [rf"$p={p}$" for p in degrees]
    fig.legend(
        handles,
        labels,
        ncol=len(degrees),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.045),
        frameon=False,
        handlelength=1.6,
        columnspacing=1.3,
    )

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.12, top=0.86, wspace=0.28)
    save_figure(fig, "fig_s5_infsup_sweep")
    plt.close(fig)

    dev_max = data["sweep"]["max_relative_deviation_equal_order"]
    print(
        f"fig_s5_infsup_sweep: {len(degrees)} degrees, "
        f"N up to {int(max(curves[degrees[0]]['N']))}, "
        f"max |beta_h/pi-1| = {dev_max:.2e}"
    )


if __name__ == "__main__":
    main()
