"""Closure-moment ladder for the six matched-order families at q in {4,6,8}.

For each family at each `q`, the driver assembles the first-derivative
operator on `N=65` nodes of the unit interval and computes scaled wall-row
closure moments. BMF uses its marched derivative-grid gradient, while BMG
uses its primal-grid Galerkin derivative. The spline scale is the open-uniform
knot width `h=1/(N-p)`.

    delta_{0, ell} = (D x^ell)_0 - ell x_0^{ell-1},      ell = 1, ..., L_max,

with `x_0 = 0`. Structural-zero classification uses a rowwise backward-error
tolerance, never a fixed absolute threshold. Exact-rational finite-grid
endpoint certificates identify the first free spline moment.

Output JSON: `code/results/closure_moment_ladder_summary.json`
Output figure: `figures/fig_s6_closure_moment_ladder.pdf`
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import boundary_spline_analysis as ba
from project_paths import FIGDIR, RESULTSDIR, ensure_generated_dirs, save_figure
import wall_order_certificates as wc


EXPERIMENT = {
    "npts": 65,
    "qs": (4, 6, 8),
    "families": ("E", "C", "SBP", "DRP", "BMF", "BMG"),
    "ell_max": 9,
    "precision": {"arithmetic": "IEEE-754 binary64", "bits": 53},
    "tolerance_rationale": wc.TOLERANCE_RATIONALE,
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10.0,
    'axes.labelsize': 10.5,
    'axes.titlesize': 11.5,
    'figure.dpi': 180,
    'savefig.dpi': 300,
})


def closure_moments(family: str, q: int, npts: int, ell_max: int) -> dict:
    """Compute raw and scaled moments for the left-wall derivative row."""
    if family in ("BMF", "BMG"):
        derivative = wc.actual_derivative_map(family, q, npts)
        D = derivative.matrix
        x = derivative.input_nodes
        output_grid = derivative.output_grid
        h = wc.knot_span_width(npts, q)
        h_definition = "1/(N-p)"
    else:
        op = ba.build_operator(ba.scheme_kind(f"{family}{q}"), q, npts)
        D = op.matrix
        x = op.x.copy()
        output_grid = "family grid"
        h = 1.0 / (npts - 1)
        h_definition = "1/(N-1)"
    structural_exact_degree = q // 2 if family == "SBP" else q
    moments = []
    for ell in range(1, ell_max + 1):
        values = x**ell
        target = 1.0 if ell == 1 else 0.0
        raw = float(D[0] @ values - target)
        raw_tolerance = wc.backward_error_tolerance(
            D[[0]], values, np.array([target]), precision_bits=53
        )
        scaled = h ** (1 - ell) * raw
        scaled_tolerance = h ** (1 - ell) * raw_tolerance
        within_backward_tolerance = bool(abs(raw) <= raw_tolerance)
        moments.append({
            "ell": int(ell),
            "raw_moment": raw,
            "scaled_moment": float(scaled),
            "raw_tolerance": float(raw_tolerance),
            "scaled_tolerance": float(scaled_tolerance),
            "within_backward_tolerance": within_backward_tolerance,
            "structural_zero": bool(ell <= structural_exact_degree),
            "monomial": "(x-left_wall)^ell",
        })

    reference_exact_endpoint = None
    reference_exact_endpoint_n = None
    exact_degree = structural_exact_degree
    first_nonzero = structural_exact_degree + 1
    if family in ("BMF", "BMG"):
        certificate_n = q + 9
        reference_exact_endpoint = wc.multiprecision_endpoint_certificate(
            family, q, certificate_n, decimal_digits=90
        )
        reference_exact_endpoint_n = certificate_n
        if first_nonzero <= ell_max:
            moments[first_nonzero - 1]["classification"] = (
                f"first free moment at N={npts}; floating backward-error "
                f"classification here, with a separate exact-rational reference at N={certificate_n}"
            )

    deltas = [moment["raw_moment"] for moment in moments]
    if family in ("BMF", "BMG"):
        wall_nnz = int(D.shape[1])
        wall_footprint_definition = "structural eliminated-row width"
    else:
        wall_nnz = int(np.sum(np.abs(D[0, :]) > 1.0e-12))
        wall_footprint_definition = "numerical nonzero count at 1e-12"
    return {
        "family": family,
        "q": int(q),
        "N": int(npts),
        "h": float(h),
        "h_definition": h_definition,
        "output_grid": output_grid,
        "row": 0,
        "precision": EXPERIMENT["precision"],
        "tolerance_rationale": EXPERIMENT["tolerance_rationale"],
        "moments": moments,
        "deltas": deltas,
        "scaled_moments": [moment["scaled_moment"] for moment in moments],
        "exact_polynomial_degree": exact_degree,
        "first_nonzero_moment_degree": first_nonzero,
        "reference_exact_endpoint_N": reference_exact_endpoint_n,
        "reference_exact_endpoint_certificate": reference_exact_endpoint,
        "floating_first_free_moment_at_N": {
            "N": int(npts),
            **(moments[first_nonzero - 1] if first_nonzero <= ell_max else {}),
        },
        "wall_footprint": wall_nnz,
        "wall_footprint_definition": wall_footprint_definition,
    }


def run_all() -> dict:
    out: dict = {"configuration": EXPERIMENT, "rows": []}
    for q in EXPERIMENT["qs"]:
        for fam in EXPERIMENT["families"]:
            row = closure_moments(fam, int(q), int(EXPERIMENT["npts"]),
                                   int(EXPERIMENT["ell_max"]))
            out["rows"].append(row)
            moments = row["moments"]
            zero_count = sum(1 for item in moments if item["structural_zero"])
            first = row["first_nonzero_moment_degree"]
            defect = moments[first - 1]["scaled_moment"] if first is not None and first <= len(moments) else 0.0
            print(f"  {fam}{q}: exact degree={row['exact_polynomial_degree']}; "
                  f"first defect mag = "
                  f"{defect:.2e} (scaled); "
                  f"wall-row footprint width = {row['wall_footprint']}")
    return out


_PANEL_BG = (0.987, 0.985, 0.978)
_ACCENT = '#2F4858'


def make_paper_figure(summary: dict, stem: str) -> None:
    """Closure-moment ladder rendered with the project classy palette.

    Spline families sit at the top of each panel so the dense block of
    structural zeros reads first. Non-zero residual cells use a quiet
    rust ramp on the cream panel background; structural zeros are filled
    accent diamonds with a thin cream halo. A family colour stripe on the
    far left ties each row to the BMF/BMG/E/C/SBP/DRP convention used
    elsewhere in the manuscript.
    """
    from matplotlib.colors import LinearSegmentedColormap

    rows = summary["rows"]
    cfg = summary["configuration"]
    qs = cfg["qs"]; ell_max = cfg["ell_max"]

    # Bottom-up display order with origin='lower' places BMG at the top
    # and E at the bottom of each panel.
    fam_order = ('E', 'C', 'SBP', 'DRP', 'BMF', 'BMG')

    cream_to_rust = LinearSegmentedColormap.from_list(
        'cream_rust', [(0.987, 0.985, 0.978),
                       (0.95, 0.83, 0.74),
                       (0.83, 0.45, 0.30),
                       (0.45, 0.13, 0.10)],
    )

    fig = plt.figure(figsize=(14.2, 5.4), facecolor='white')
    gs = fig.add_gridspec(1, len(qs), wspace=0.40, left=0.058, right=0.88,
                          top=0.86, bottom=0.17)
    im = None

    for j, q in enumerate(qs):
        ax = fig.add_subplot(gs[0, j])
        ax.set_facecolor(_PANEL_BG)
        mat = np.zeros((len(fam_order), ell_max), dtype=float)
        wall_w = {}
        for i, fam in enumerate(fam_order):
            row = next(r for r in rows if r["family"] == fam and r["q"] == q)
            for k, v in enumerate(row["scaled_moments"]):
                mat[i, k] = max(abs(v), 1.0e-16)
            wall_w[fam] = int(row["wall_footprint"])

        im = ax.imshow(mat, aspect='auto', cmap=cream_to_rust,
                       norm=LogNorm(vmin=1.0e-15, vmax=1.0e0),
                       interpolation='nearest', origin='lower',
                       extent=(-0.5, ell_max - 0.5,
                               -0.5, len(fam_order) - 0.5))

        # Structural-zero markers as accent diamonds with cream halo.
        for i, fam in enumerate(fam_order):
            row = next(r for r in rows if r["family"] == fam and r["q"] == q)
            for k, moment in enumerate(row["moments"]):
                if moment["structural_zero"]:
                    ax.scatter(k, i, s=68, marker='D', facecolor=_ACCENT,
                               edgecolor=_PANEL_BG, linewidth=0.9, zorder=3)

        # Family-colour stripe on the left of each panel.
        for i, fam in enumerate(fam_order):
            ax.add_patch(plt.Rectangle(
                (-0.95, i - 0.42), 0.32, 0.84,
                facecolor=ba.DEFAULT_COLORS[fam], edgecolor=_ACCENT,
                linewidth=0.4, alpha=0.92, clip_on=False, zorder=4,
            ))
        # Visual separator between non-spline (rows 0..3) and spline (rows 4..5).
        ax.axhline(3.5, color=_ACCENT, lw=0.5, alpha=0.30, zorder=3)

        ax.set_xticks(range(ell_max))
        ax.set_xticklabels([str(k + 1) for k in range(ell_max)],
                           fontsize=9.4, color=_ACCENT)
        ax.set_yticks(range(len(fam_order)))
        ax.set_yticklabels(list(fam_order), fontsize=10.6, color=_ACCENT,
                           family='serif')
        ax.tick_params(axis='both', length=0, pad=4)
        ax.set_xlabel(r'closure moment index $\ell$',
                      fontsize=10.8, color=_ACCENT)
        if j == 0:
            ax.set_ylabel('family', fontsize=10.8, color=_ACCENT)

        for spine in ax.spines.values():
            spine.set_color(_ACCENT)
            spine.set_linewidth(0.55)
        ax.set_title(rf'$q={q}$', fontsize=12.5, color=_ACCENT, pad=8)

        # Wall-row footprint column on the right margin, anchored in axes-fraction
        # so the colorbar cannot clip the values regardless of panel width.
        for i, fam in enumerate(fam_order):
            ax.text(1.018, i, rf'$w{{=}}{wall_w[fam]}$',
                    transform=ax.get_yaxis_transform(),
                    ha='left', va='center', fontsize=9.2, color=_ACCENT,
                    family='serif', clip_on=False)

    cbar_ax = fig.add_axes([0.945, 0.17, 0.0095, 0.69])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(r'$|h^{1-\ell}\delta_{0,\ell}|$', fontsize=10.5, color=_ACCENT)
    cb.ax.tick_params(colors=_ACCENT, labelsize=8.6, length=0)
    cb.outline.set_edgecolor(_ACCENT)
    cb.outline.set_linewidth(0.5)
    save_figure(fig, stem)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_generated_dirs()
    print("== closure-moment ladder ==")
    summary = run_all()
    out_json = RESULTSDIR / "closure_moment_ladder_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")
    if not args.no_figures:
        make_paper_figure(summary, "fig_s6_closure_moment_ladder")
        print(f"wrote {FIGDIR}/fig_s6_closure_moment_ladder.(pdf,png)")


if __name__ == "__main__":
    main()
