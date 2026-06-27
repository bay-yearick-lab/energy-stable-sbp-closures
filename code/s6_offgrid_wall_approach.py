#!/usr/bin/env python3
"""Off-grid derivative accuracy up to and at the wall.

Immersed-boundary, embedded-boundary, and wall-coupling calculations need a
derivative at an off-grid location that may sit arbitrarily close to the wall.
A finite-difference family supplies that value by interpolating its nodal
derivative; the spline routes evaluate the analytic derivative of the same
reconstructed field. Two readings separate the families.

(a) Distance sweep at fixed grid: the spline off-grid derivative error is flat
    across distance, including at the wall itself, while the energy-stable
    diagonal-norm SBP closure is dominated by its half-order wall rows and the
    dispersion-tuned closure degrades toward the wall.

(b) Convergence at a fixed near-wall marker: the spline converges at the spline
    degree, the explicit and compact closures track their matched order at a
    fixed offset, and the diagonal-norm SBP closure converges at the lower
    wall-closure rate, so its gap to the spline widens with refinement.

No manufactured boundary model is attached; the off-grid value comes from the
operator's own approximation space.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import boundary_spline_analysis as ba
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

FAMILIES = [("E", "E"), ("C", "C"), ("SBP", "SBP"), ("DRP", "DRP"), ("I", "BMG"), ("B", "BMF")]
COLORS = {"E": "#7a7a7a", "C": "#4c72b0", "SBP": "#8c564b", "DRP": "#9467bd", "BMG": "#2ca02c", "BMF": "#d62728"}
Q = 6

# smooth test field with a nontrivial wall gradient
def f(x):
    return np.exp(-1.5 * x) * np.sin(4 * np.pi * x) + 0.7 * x * x
def df(x):
    return (-1.5 * np.exp(-1.5 * x) * np.sin(4 * np.pi * x)
            + np.exp(-1.5 * x) * 4 * np.pi * np.cos(4 * np.pi * x) + 1.4 * x)


def offgrid_deriv(kind: str, npts: int, markers: np.ndarray) -> np.ndarray:
    op = ba.build_operator(kind, Q, npts)
    x = op.x
    u = f(x)
    if kind in ("B", "I"):                      # spline: analytic derivative of the reconstruction
        _, dvals = op.field_and_derivative(markers, u)
        return np.asarray(dvals)
    du = op.differentiate(u)                     # finite-difference: interpolate the nodal derivative
    w = ba.local_interpolation_matrix(x, markers, Q + 1)
    return w @ du


def rel_err(kind, npts, markers):
    return np.abs(offgrid_deriv(kind, npts, markers) - df(markers)) / np.max(np.abs(df(markers)))


def main() -> None:
    out = {"Q": Q}
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.6))

    # Panel (a): distance sweep at fixed grid
    Na = 49
    h = 1.0 / (Na - 1)
    dphys = np.array([0.0, 0.05, 0.15, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]) * h
    out["distance_sweep"] = {"N": Na, "d_over_h": (dphys / h).tolist()}
    for kind, tag in FAMILIES:
        e = rel_err(kind, Na, np.clip(dphys, 0.0, 1.0))
        out["distance_sweep"][tag] = e.tolist()
        axA.semilogy(dphys / h, np.clip(e, 1e-12, 1e2), color=COLORS[tag], lw=1.9,
                     marker="o", ms=3.2)
    axA.set_xlabel(r"distance from wall $\,d/h$")
    axA.set_ylabel(r"relative off-grid derivative error")
    axA.set_title(f"(a) Approaching the wall ($N={Na}$, $q={Q}$)")
    axA.set_xlim(-0.1, 3.1)

    # Panel (b): convergence at a fixed near-wall marker (rate cliff)
    Ns = [25, 33, 49, 65, 97]
    out["convergence"] = {"N": Ns}
    for kind, tag in FAMILIES:
        errs = []
        for n in Ns:
            d = 0.25 / (n - 1)                  # marker fixed at a quarter cell from the wall
            errs.append(float(rel_err(kind, n, np.array([d]))[0]))
        out["convergence"][tag] = errs
        axB.loglog(Ns, np.clip(errs, 1e-13, 1e2), color=COLORS[tag], lw=1.9, marker="s", ms=3.4)
    # reference slopes
    nref = np.array([Ns[0], Ns[-1]], float)
    for slope, txt, y0 in [(Q, f"$h^{{{Q}}}$", 2e-5), (Q // 2, f"$h^{{{Q//2}}}$", 5e-2)]:
        axB.loglog(nref, y0 * (nref[0] / nref) ** slope, "k:", lw=0.8)
    axB.set_xlabel(r"grid points $N$")
    axB.set_ylabel(r"off-grid derivative error at $d=h/4$")
    axB.set_title("(b) Near-wall convergence rate")

    fig.legend(handles=[Line2D([0], [0], color=COLORS[t], lw=2.0, label=t) for _k, t in FAMILIES],
               ncol=6, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, "fig_s6_offgrid_wall_approach")
    plt.close(fig)
    (RESULTSDIR / "offgrid_wall_approach_summary.json").write_text(json.dumps(out, indent=2))
    print("at-wall (d=0) error:", {t: f"{out['distance_sweep'][t][0]:.1e}" for _k, t in FAMILIES})
    print("finest near-wall error:", {t: f"{out['convergence'][t][-1]:.1e}" for _k, t in FAMILIES})
    print(f"wrote figure + {RESULTSDIR/'offgrid_wall_approach_summary.json'}")


if __name__ == "__main__":
    main()
