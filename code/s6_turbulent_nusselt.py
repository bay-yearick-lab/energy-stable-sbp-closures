#!/usr/bin/env python3
"""Turbulent wall heat transfer: the wall closure sets the Nusselt number.

A single developed B-spline channel velocity field advects a passive scalar with
isothermal walls. Every scheme shares that velocity and the Fourier x,y treatment;
only the wall-normal operator differs. Two readings separate the families on a
coarse wall grid.

(a) Mean temperature profile: the spline routes hold the near-wall slope while the
    finite-difference wall closures bend away from it in the first cells.
(b) Nusselt number: the spline value sits on the reference; the wall closures
    under- or over-shoot it, since the Nusselt number is exactly the wall slope.

Reads results/turbulent_nusselt.json (written by nusselt_driver.py); numbers are
never hard-coded here.
"""
from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from project_paths import RESULTSDIR, ensure_generated_dirs, save_figure

ensure_generated_dirs()
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.2, "axes.labelsize": 11.5, "axes.titlesize": 12.0,
    "legend.fontsize": 9.8, "xtick.labelsize": 9.4, "ytick.labelsize": 9.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 180, "savefig.dpi": 300,
})

LABEL = {"I": "B-spline (BMG)", "B": "B-spline (BMF)", "C": "compact",
         "SBP": "SBP-SAT", "E": "explicit FD", "DRP": "dispersion-tuned"}
COLOR = {"I": "#2ca02c", "B": "#1f9e4e", "C": "#4c72b0", "SBP": "#8c564b",
         "E": "#7a7a7a", "DRP": "#9467bd"}
ORDER = ["I", "B", "C", "SBP", "E", "DRP"]


def load():
    p = RESULTSDIR / "turbulent_nusselt.json"
    with open(p) as f:
        return json.load(f)


def main():
    d = load()
    sch = d["schemes"]
    kinds = [k for k in ORDER if k in sch]
    Lz = max(max(sch[k]["z"]) for k in kinds)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 4.6),
                                   gridspec_kw={"width_ratios": [1.55, 1.0]})

    # Panel (a): mean temperature profile, near-wall region emphasised
    for k in kinds:
        z = np.asarray(sch[k]["z"])
        th = np.asarray(sch[k]["profile"])
        axA.plot(th, z, color=COLOR[k], lw=2.0, label=LABEL[k])
    axA.set_xlabel(r"mean temperature $\langle\theta\rangle$")
    axA.set_ylabel(r"wall-normal coordinate $z$")
    axA.set_title("(a) Mean temperature profile")
    axA.set_ylim(0, Lz)

    # near-wall inset (bottom wall) where the wall closures separate
    ins = axA.inset_axes([0.55, 0.12, 0.4, 0.42])
    zb = 0.18 * Lz
    for k in kinds:
        z = np.asarray(sch[k]["z"])
        th = np.asarray(sch[k]["profile"])
        m = z <= zb
        ins.plot(th[m], z[m], color=COLOR[k], lw=1.8, marker="o", ms=3.0)
    ins.set_title("near wall", fontsize=8.5)
    ins.tick_params(labelsize=7.5)

    # Panel (b): Nusselt number per scheme, against the reference line
    Nus = [sch[k]["Nu"] for k in kinds]
    xpos = np.arange(len(kinds))
    bars = axB.bar(xpos, Nus, color=[COLOR[k] for k in kinds], width=0.62, zorder=3)
    ref = d.get("reference_Nu")
    if ref:
        axB.axhline(ref, color="k", lw=1.3, ls="--", zorder=2)
        axB.text(len(kinds) - 0.5, ref, f"  reference {ref:.2f}", va="center",
                 ha="left", fontsize=8.8)
    for x, val in zip(xpos, Nus):
        axB.text(x, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9.0)
    axB.set_xticks(xpos)
    axB.set_xticklabels([LABEL[k] for k in kinds], rotation=18, ha="right")
    axB.set_ylabel(r"Nusselt number $Nu$")
    axB.set_title("(b) Wall heat transfer")
    axB.set_ylim(0, max(Nus + ([ref] if ref else [])) * 1.18)

    fig.legend(handles=[Line2D([0], [0], color=COLOR[k], lw=2.2, label=LABEL[k]) for k in kinds],
               ncol=len(kinds), loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, "fig_s6_turbulent_nusselt")
    plt.close(fig)
    print("schemes:", {k: round(sch[k]["Nu"], 3) for k in kinds})
    print("wrote fig_s6_turbulent_nusselt")


if __name__ == "__main__":
    main()
