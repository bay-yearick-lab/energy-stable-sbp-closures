#!/usr/bin/env python3
"""Exact weighted-SBP energy audit via artificial interior power."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import boundary_spline_analysis as ba
from project_paths import RESULTSDIR, ensure_generated_dirs, save_figure


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.2,
    "axes.labelsize": 11.5,
    "axes.titlesize": 12.1,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 9.4,
    "ytick.labelsize": 9.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.12,
    "grid.linewidth": 0.45,
})


NPTS = 33
TIMES = np.linspace(0.0, 0.40, 500)
U0_FORMULA = "0.32 sin(12 pi x) + 0.24 cos(11 pi x) + 0.10 sin(5 pi x)"
FLOOR = 1.0e-18

# Float64 roundoff range expected for these bilinear products on length-N
# operator stencils: shading marks the presentation band, not a strict bound.
MACHINE_ZERO_TOP = 5.0e-13
VISUAL_FLOOR = 1.0e-17


def audit_matrices_for(label: str, op) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    family = ba.scheme_family(label)
    if family == "BMF":
        ops = ba.bmf_staggered_operators(NPTS, ba.scheme_order(label))
        amat = np.block([
            [np.zeros((NPTS, NPTS), dtype=float), ops.d01],
            [ops.d10, np.zeros((NPTS - 1, NPTS - 1), dtype=float)],
        ])
        hmat = np.block([
            [ops.h0, np.zeros((NPTS, NPTS - 1), dtype=float)],
            [np.zeros((NPTS - 1, NPTS), dtype=float), ops.h1],
        ])
        bmat = np.block([
            [np.zeros((NPTS, NPTS), dtype=float), ops.boundary],
            [ops.boundary.T, np.zeros((NPTS - 1, NPTS - 1), dtype=float)],
        ])
        p0 = 0.32 * np.sin(12.0 * np.pi * ops.x) + 0.24 * np.cos(11.0 * np.pi * ops.x) + 0.10 * np.sin(5.0 * np.pi * ops.x)
        v0 = 0.18 * np.sin(10.0 * np.pi * ops.eta) - 0.11 * np.cos(7.0 * np.pi * ops.eta)
        return hmat, amat, bmat, np.concatenate([p0, v0])
    if family in ("BMG", "SBP"):
        H = op.metric
    else:
        H = np.diag(ba.trapezoidal_weights(NPTS))
    x = op.x
    u0 = 0.32 * np.sin(12.0 * np.pi * x) + 0.24 * np.cos(11.0 * np.pi * x) + 0.10 * np.sin(5.0 * np.pi * x)
    return H, op.matrix, ba.endpoint_flux_matrix(NPTS), u0


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    for i in range(1, y.size):
        out[i] = out[i - 1] + 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return out


def main() -> None:
    ensure_generated_dirs()
    schemes = ba.comparison_block(6)
    summary: dict[str, dict] = {}
    power_curves: dict[str, np.ndarray] = {}

    for label, kind, degree in schemes:
        op = ba.build_operator(kind, degree, NPTS)
        H, D, Bmat, u0 = audit_matrices_for(label, op)
        R = H @ D + D.T @ H - Bmat
        sol = ba.exact_semidiscrete_evolution(D, u0, TIMES)
        # Artificial power quadratic form along the exact semidiscrete trajectory.
        power = np.array([-0.5 * u @ R @ u for u in sol], dtype=float)
        # Natural-norm energy of the evolving state.
        energy = np.array([0.5 * u @ H @ u for u in sol], dtype=float)
        # State-normalised artificial-power rate (per-time, independent of
        # trajectory amplification). This is the actual weighted-SBP
        # defect rate: r(t) = |u^T R u| / (u^T H u). For a scheme that
        # satisfies H D + D^T H = B exactly, r(t) = 0 at every time.
        rate = np.abs(power) / np.maximum(energy, FLOOR)
        cum_rate = cumulative_trapezoid(rate, TIMES)
        power_curves[label] = np.maximum(rate, FLOOR)
        summary[label] = {
            "max_rel_artificial_rate": float(np.max(rate)),
            "cum_rel_artificial_rate": float(cum_rate[-1]),
            "max_abs_artificial_power": float(np.max(np.abs(power))),
            "max_energy": float(np.max(energy)),
            "balance_residual_norm2": float(np.linalg.norm(R, 2)),
        }

    fig = plt.figure(figsize=(13.4, 4.7))
    gs = fig.add_gridspec(1, 3, wspace=0.30, left=0.065, right=0.995, top=0.80, bottom=0.18)
    ax_trace = fig.add_subplot(gs[0, 0])
    ax_work = fig.add_subplot(gs[0, 1])
    ax_resid = fig.add_subplot(gs[0, 2])

    for ax in (ax_trace, ax_work, ax_resid):
        ax.axhspan(VISUAL_FLOOR, MACHINE_ZERO_TOP, color="0.65", alpha=0.15, zorder=0)
        ax.axhline(MACHINE_ZERO_TOP, color="0.45", ls=(0, (4, 3)), lw=0.7, zorder=1)

    for label, _kind, _degree in schemes:
        family = ba.scheme_family(label)
        ax_trace.semilogy(TIMES, power_curves[label], color=ba.DEFAULT_COLORS[family], ls=ba.DEFAULT_LINESTYLES[family], marker=ba.DEFAULT_MARKERS[family], lw=2.15, ms=5.6, markevery=56, mfc="white", mec=ba.DEFAULT_COLORS[family], mew=0.95)
    ax_trace.set_title(r"(a) Instantaneous weighted-SBP defect rate", pad=6, fontweight="semibold")
    ax_trace.set_xlabel(r"time $t$")
    ax_trace.set_ylabel(r"$r(t) = |u^T(HD + D^TH - B)u|\,/\,(u^TH u)$")

    labels = [label for label, _kind, _degree in schemes]
    colors = [ba.DEFAULT_COLORS[ba.scheme_family(label)] for label in labels]
    work_vals = [summary[label]["cum_rel_artificial_rate"] for label in labels]
    resid_vals = [summary[label]["balance_residual_norm2"] for label in labels]
    ax_work.bar(labels, work_vals, color=colors, edgecolor="black", lw=0.5)
    ax_resid.bar(labels, resid_vals, color=colors, edgecolor="black", lw=0.5)
    for ax, ylabel, title in (
        (ax_work, r"$\int_0^T r(t)\,dt$", r"(b) Cumulative weighted-SBP defect rate"),
        (ax_resid, r"$\|HD + D^TH - B\|_2$", r"(c) Algebraic weighted-SBP residual"),
    ):
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=6, fontweight="semibold")
        ax.text(0.985, MACHINE_ZERO_TOP, "machine zero",
                transform=ax.get_yaxis_transform(), fontsize=7.6, color="0.35",
                va="bottom", ha="right",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0})

    handles = [
        Line2D([0], [0], color=ba.DEFAULT_COLORS[ba.scheme_family(label)], ls=ba.DEFAULT_LINESTYLES[ba.scheme_family(label)], marker=ba.DEFAULT_MARKERS[ba.scheme_family(label)], lw=2.3, ms=5.9, mfc="white", mec=ba.DEFAULT_COLORS[ba.scheme_family(label)], mew=0.95, label=label)
        for label in labels
    ]
    fig.legend(handles, [h.get_label() for h in handles], ncol=len(handles), loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False, handlelength=2.8, columnspacing=1.2, fontsize=11.2)
    save_figure(fig, "fig_s6_energy_artificial_power")
    plt.close(fig)

    out = RESULTSDIR / "energy_artificial_power_summary.json"
    out.write_text(json.dumps({"configuration": {"npts": NPTS, "times": [float(TIMES[0]), float(TIMES[-1]), int(TIMES.size)], "initial_state_formula": U0_FORMULA, "bmf_audit": "staggered flux-divergence system with primal pressure and derivative-grid flux"}, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
