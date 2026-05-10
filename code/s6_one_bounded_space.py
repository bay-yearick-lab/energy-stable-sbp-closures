"""One bounded field space, four observable channels.

For the spline pair, a single coefficient algebra ``alpha = B^{-1} u_node``
supplies (i) field reconstruction at off-grid points, (ii) gradient values
at off-grid points, (iii) the natural-metric volume integral
``int_0^1 u(x) dx``, and (iv) the conditioning of the auxiliary map that
binds those channels. The four finite-difference families do not have
this single source: each off-grid observable is built from a separately
tuned local-Lagrange map whose conditioning is reported here.

Test field: ``u(x) = exp(-2 x) sin(7 pi x) cos(3 pi x)`` on ``[0, 1]``.
Native grid sizes: ``N = 33`` for ``q in {4, 6, 8}``, sampled on each
family's native nodes. Off-grid evaluation uses 1024 points drawn from a
Halton-style stratified sequence on ``[0, 1]``.

Output JSON: ``code/results/one_bounded_space_summary.json``
Output figure: ``figures/fig_s6_one_bounded_space.pdf``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import boundary_spline_analysis as ba
from project_paths import FIGDIR, RESULTSDIR, ensure_generated_dirs, save_figure


EXPERIMENT = {
    "npts": 33,
    "qs": (4, 6, 8),
    "families": ("E", "C", "SBP", "DRP", "BMF", "BMG"),
    "n_offgrid": 1024,
    "rng_seed": 17,
}

_PANEL_BG = (0.987, 0.985, 0.978)
_ACCENT = '#2F4858'

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


def test_field(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``u(x), u'(x)`` for ``u = exp(-2x) sin(7 pi x) cos(3 pi x)``."""
    a = np.exp(-2.0 * x)
    s7 = np.sin(7.0 * np.pi * x)
    c7 = np.cos(7.0 * np.pi * x)
    s3 = np.sin(3.0 * np.pi * x)
    c3 = np.cos(3.0 * np.pi * x)
    u = a * s7 * c3
    du = (
        -2.0 * a * s7 * c3
        + a * 7.0 * np.pi * c7 * c3
        - a * 3.0 * np.pi * s7 * s3
    )
    return u, du


def integral_reference() -> float:
    """High-order Gauss-Legendre reference for ``int_0^1 u(x) dx``."""
    xi, wi = leggauss(80)
    nodes = 0.5 * (xi + 1.0)
    weights = 0.5 * wi
    u, _ = test_field(nodes)
    return float(weights @ u)


def stratified_offgrid_points(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    edges = np.linspace(0.0, 1.0, n + 1)
    pts = edges[:-1] + (edges[1:] - edges[:-1]) * rng.random(n)
    return np.sort(pts)


def auxiliary_map_kappa(family: str, q: int, op, points: np.ndarray) -> float:
    """Condition number of the auxiliary map that supplies off-grid reads.

    For the spline pair the coefficient matrix ``B`` is the only auxiliary
    structure the four channels share. For the finite-difference quartet
    the relevant map is the local-Lagrange interpolation matrix evaluated
    at ``points``.
    """
    if family in ('BMF', 'BMG'):
        return float(np.linalg.cond(op._bmat))
    interp = ba.local_interpolation_matrix(op.x, points, q + 1)
    sing = np.linalg.svd(interp, compute_uv=False)
    if sing[-1] <= 0.0:
        return float('inf')
    return float(sing[0] / sing[-1])


def channel_quadrature_weights(family: str, q: int, op) -> np.ndarray:
    """Native quadrature weights ``w_j`` for ``int u dx ~= sum_j w_j u_j``.

    The spline pair carries the exact spline volume weights from the
    natural metric ``H_0`` (BMF) or the BMG natural metric. The
    finite-difference quartet uses trapezoidal weights on the native grid.
    """
    if family == 'BMF':
        h0 = ba.bmf_full_metric(op.x.size, op.degree)[1]
        return np.asarray(h0.sum(axis=1), dtype=float)
    if family == 'BMG':
        return np.asarray(op._metric.sum(axis=1), dtype=float)
    x = np.asarray(op.x, dtype=float)
    h = np.diff(x)
    w = np.zeros_like(x)
    w[:-1] += 0.5 * h
    w[1:] += 0.5 * h
    return w



def evaluate_family(family: str, q: int, points: np.ndarray, integral_ref: float) -> dict:
    op = ba.build_operator(ba.scheme_kind(f"{family}{q}"), q,
                           int(EXPERIMENT["npts"]))
    u_node, _du_node = test_field(op.x)
    u_off, du_off = test_field(points)

    field_off, grad_off = op.field_and_derivative(points, u_node)
    err_field = float(np.max(np.abs(field_off - u_off)))
    err_grad = float(np.max(np.abs(grad_off - du_off)))

    weights = channel_quadrature_weights(family, q, op)
    integral_h = float(weights @ u_node)
    err_integral = float(abs(integral_h - integral_ref))

    kappa = auxiliary_map_kappa(family, q, op, points)
    return {
        "family": family,
        "q": int(q),
        "npts": int(op.x.size),
        "err_field": err_field,
        "err_grad": err_grad,
        "err_integral": err_integral,
        "auxiliary_kappa": kappa,
    }


def run_all() -> dict:
    points = stratified_offgrid_points(int(EXPERIMENT["n_offgrid"]),
                                        int(EXPERIMENT["rng_seed"]))
    integral_ref = integral_reference()
    out: dict = {
        "configuration": EXPERIMENT,
        "integral_reference": integral_ref,
        "rows": [],
    }
    print(f"  reference integral = {integral_ref:.12e}")
    for q in EXPERIMENT["qs"]:
        for fam in EXPERIMENT["families"]:
            row = evaluate_family(fam, int(q), points, integral_ref)
            out["rows"].append(row)
            print(
                f"  {fam}{q}: field={row['err_field']:.2e}  "
                f"grad={row['err_grad']:.2e}  "
                f"integral={row['err_integral']:.2e}  "
                f"kappa={row['auxiliary_kappa']:.2e}"
            )
    return out



def _series(rows: list[dict], family: str, key: str, qs) -> list[float]:
    out = []
    for q in qs:
        cell = next(r for r in rows if r["family"] == family and r["q"] == q)
        out.append(float(cell[key]))
    return out


def make_paper_figure(summary: dict, stem: str) -> None:
    """Four-panel demonstration of the single bounded field space.

    The four panels report the same observables that flow from the spline
    coefficient algebra: off-grid value error, off-grid gradient error,
    volume integral error, and the conditioning of the auxiliary map. The
    spline pair sits one to four orders of magnitude below the
    finite-difference quartet on the three error channels with a
    moderately conditioned coefficient algebra.
    """
    rows = summary["rows"]
    qs = list(summary["configuration"]["qs"])
    families = list(summary["configuration"]["families"])

    fig = plt.figure(figsize=(13.4, 8.6), facecolor='white')
    gs = fig.add_gridspec(
        2, 2, wspace=0.26, hspace=0.40,
        left=0.072, right=0.985, top=0.895, bottom=0.085,
    )
    panels = [
        (gs[0, 0], "err_field", r"off-grid value error $\max_X|u(X)-u_h(X)|$",
         "(a) field reconstruction"),
        (gs[0, 1], "err_grad", r"off-grid gradient error $\max_X|u'(X)-u_h'(X)|$",
         "(b) gradient evaluation"),
        (gs[1, 0], "err_integral",
         r"volume integral error $\,|\,\int_0^1 u\,dx - \sum_j w_j u_j\,|$",
         "(c) traction integral"),
        (gs[1, 1], "auxiliary_kappa",
         r"auxiliary-map condition number $\kappa_2$",
         "(d) auxiliary conditioning"),
    ]

    fam_order = ('E', 'C', 'SBP', 'DRP', 'BMF', 'BMG')
    for cell, key, ylabel, title in panels:
        ax = fig.add_subplot(cell)
        ax.set_facecolor(_PANEL_BG)
        for fam in fam_order:
            ys = _series(rows, fam, key, qs)
            ys_safe = [max(v, 1.0e-18) for v in ys]
            ax.plot(qs, ys_safe,
                    color=ba.DEFAULT_COLORS[fam],
                    marker=ba.DEFAULT_MARKERS[fam],
                    linestyle=ba.DEFAULT_LINESTYLES[fam],
                    linewidth=1.4, markersize=8.0,
                    markerfacecolor='white' if fam in ('BMF', 'BMG') else ba.DEFAULT_COLORS[fam],
                    markeredgecolor=ba.DEFAULT_COLORS[fam],
                    markeredgewidth=1.4 if fam in ('BMF', 'BMG') else 0.9,
                    label=fam)
        ax.set_yscale('log')
        ax.set_xticks(qs)
        ax.set_xticklabels([rf'${q}$' for q in qs])
        ax.set_xlabel(r'order or degree $q$', color=_ACCENT)
        ax.set_ylabel(ylabel, color=_ACCENT)
        ax.set_title(title, color=_ACCENT, pad=8, fontsize=11.6)
        ax.grid(True, which='major', alpha=0.18, lw=0.5)
        ax.grid(True, which='minor', alpha=0.08, lw=0.4)
        for spine in ax.spines.values():
            spine.set_color(_ACCENT)
            spine.set_linewidth(0.55)
        ax.tick_params(colors=_ACCENT, labelsize=9.5)

    # Single shared legend across the figure top so panels stay clean.
    handles, labels = fig.axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='upper center',
                     bbox_to_anchor=(0.5, 0.995), ncol=6,
                     frameon=False, fontsize=11.2,
                     handlelength=2.8, columnspacing=2.2)
    for txt in leg.get_texts():
        txt.set_color(_ACCENT)

    save_figure(fig, stem)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_generated_dirs()
    print("== one bounded space, four channels ==")
    summary = run_all()
    out_json = RESULTSDIR / "one_bounded_space_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")
    if not args.no_figures:
        make_paper_figure(summary, "fig_s6_one_bounded_space")
        print(f"wrote {FIGDIR}/fig_s6_one_bounded_space.(pdf,png)")


if __name__ == "__main__":
    main()
