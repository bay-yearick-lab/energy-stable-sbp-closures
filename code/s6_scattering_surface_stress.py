#!/usr/bin/env python3
"""Oscillatory same-sphere load-transfer stress test for Section 6.11."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.linalg import lu_solve

import boundary_spline_analysis as ba
from boundary_spline_analysis import scheme_display_name
from plotting_helpers import add_theoretical_reference_line
import scattering_core as sc
from project_paths import RESULTSDIR, ensure_generated_dirs, recommended_process_workers, save_figure


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.3,
    "axes.labelsize": 11.4,
    "axes.titlesize": 12.1,
    "legend.fontsize": 10.6,
    "xtick.labelsize": 9.3,
    "ytick.labelsize": 9.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.12,
    "grid.linewidth": 0.45,
})


SCHEMES = ba.comparison_labels(6)
GRIDS = [25, 33, 41, 49]
BAR_GRID = 25
MAP_GRID = 49
T_SAMPLE = 0.35
LOAD_ALPHA = 4.0
SPHERE_CENTER = (0.42, 0.48, 0.52)
SPHERE_RADIUS = 0.15
N_SURFACE_NODES = 1600
MMS_MODES = [(4, 3, 2, 1.00), (3, 4, 2, 0.70), (2, 3, 4, 0.50)]
FORMAL_ORDER = 6


def _mms() -> sc.BoxEigenmode:
    return sc.BoxEigenmode(modes=MMS_MODES, c=1.0, rho0=1.0, length=1.0)


def _solve_tensor_system(tensor: np.ndarray, lu_factor) -> np.ndarray:
    n = tensor.shape[0]
    coeff = lu_solve(lu_factor, tensor.reshape(n, -1)).reshape(n, n, n)
    coeff = np.transpose(lu_solve(lu_factor, np.transpose(coeff, (1, 0, 2)).reshape(n, -1)).reshape(n, n, n), (1, 0, 2))
    coeff = np.transpose(lu_solve(lu_factor, np.transpose(coeff, (2, 0, 1)).reshape(n, -1)).reshape(n, n, n), (1, 2, 0))
    return coeff


def _interp_tensor(field: np.ndarray, wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> np.ndarray:
    tmp = np.einsum("mi,ijk->mjk", wx, field, optimize=True)
    tmp = np.einsum("mj,mjk->mk", wy, tmp, optimize=True)
    return np.einsum("mk,mk->m", wz, tmp, optimize=True)


def _fibonacci_lat_lon(n: int) -> np.ndarray:
    idx = np.arange(n, dtype=float) + 0.5
    theta = np.arccos(1.0 - 2.0 * idx / n)
    phi = np.mod(np.pi * (1.0 + 5.0**0.5) * idx + np.pi, 2.0 * np.pi) - np.pi
    return np.stack([theta, phi], axis=1)


def evaluate_scheme(label: str, npts: int) -> dict:
    kind = ba.scheme_kind(label)
    degree = ba.scheme_order(label)
    family = ba.scheme_family(label)
    op1d = ba.build_operator(kind, degree, npts)
    op3d = sc.Operator3D(label, npts)
    x = op3d.x
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    sphere = sc.build_sphere_surface(op3d, SPHERE_CENTER, SPHERE_RADIUS, N_SURFACE_NODES)
    mms = _mms()
    field = mms.pressure(X, Y, Z, T_SAMPLE)
    p_exact = mms.pressure_on_sphere(sphere.points, T_SAMPLE)
    grad_exact = mms.grad_pressure_on_sphere(sphere.points, T_SAMPLE)
    nd_exact = np.einsum("mj,mj->m", grad_exact, sphere.normals, optimize=True)
    load_exact = nd_exact + LOAD_ALPHA * p_exact
    force_exact = -np.einsum("m,mj,m->j", load_exact, sphere.normals, sphere.weights, optimize=True)
    scale_p = max(float(np.linalg.norm(p_exact)), 1.0e-30)
    scale_nd = max(float(np.linalg.norm(nd_exact)), 1.0e-30)
    scale_load = max(float(np.linalg.norm(load_exact)), 1.0e-30)
    scale_force = max(float(np.linalg.norm(force_exact)), 1.0e-30)

    if family in ("E", "C", "SBP", "DRP"):
        wx = ba.local_interpolation_matrix(x, sphere.points[:, 0], degree + 1)
        wy = ba.local_interpolation_matrix(x, sphere.points[:, 1], degree + 1)
        wz = ba.local_interpolation_matrix(x, sphere.points[:, 2], degree + 1)
        value = _interp_tensor(field, wx, wy, wz)
        gx = _interp_tensor(op3d.dx(field), wx, wy, wz)
        gy = _interp_tensor(op3d.dy(field), wx, wy, wz)
        gz = _interp_tensor(op3d.dz(field), wx, wy, wz)
    elif kind == "B":
        coeff = _solve_tensor_system(field, op1d._lu)
        _, phix, dphix = ba.spline_eval_matrices(sphere.points[:, 0], x, degree)
        _, phiy, dphiy = ba.spline_eval_matrices(sphere.points[:, 1], x, degree)
        _, phiz, dphiz = ba.spline_eval_matrices(sphere.points[:, 2], x, degree)
        value = np.einsum("mi,ijk,mj,mk->m", phix, coeff, phiy, phiz, optimize=True)
        gx = np.einsum("mi,ijk,mj,mk->m", dphix, coeff, phiy, phiz, optimize=True)
        gy = np.einsum("mi,ijk,mj,mk->m", phix, coeff, dphiy, phiz, optimize=True)
        gz = np.einsum("mi,ijk,mj,mk->m", phix, coeff, phiy, dphiz, optimize=True)
    else:
        coeff = _solve_tensor_system(field, op1d._plu)
        gxc = op1d.differentiate_coefficients(coeff.reshape(npts, -1)).reshape(npts, npts, npts)
        gyc = np.transpose(op1d.differentiate_coefficients(np.transpose(coeff, (1, 0, 2)).reshape(npts, -1)).reshape(npts, npts, npts), (1, 0, 2))
        gzc = np.transpose(op1d.differentiate_coefficients(np.transpose(coeff, (2, 0, 1)).reshape(npts, -1)).reshape(npts, npts, npts), (1, 2, 0))
        _, phix, _ = ba.spline_eval_matrices(sphere.points[:, 0], x, degree)
        _, phiy, _ = ba.spline_eval_matrices(sphere.points[:, 1], x, degree)
        _, phiz, _ = ba.spline_eval_matrices(sphere.points[:, 2], x, degree)
        value = np.einsum("mi,ijk,mj,mk->m", phix, coeff, phiy, phiz, optimize=True)
        gx = np.einsum("mi,ijk,mj,mk->m", phix, gxc, phiy, phiz, optimize=True)
        gy = np.einsum("mi,ijk,mj,mk->m", phix, gyc, phiy, phiz, optimize=True)
        gz = np.einsum("mi,ijk,mj,mk->m", phix, gzc, phiy, phiz, optimize=True)

    nd = gx * sphere.normals[:, 0] + gy * sphere.normals[:, 1] + gz * sphere.normals[:, 2]
    load = nd + LOAD_ALPHA * value
    force = -np.einsum("m,mj,m->j", load, sphere.normals, sphere.weights, optimize=True)
    return {
        "label": label,
        "npts": int(npts),
        "h": float(op3d.h_nominal),
        "h_min": float(op3d.h),
        "value_rel_l2": float(np.linalg.norm(value - p_exact) / scale_p),
        "normal_rel_l2": float(np.linalg.norm(nd - nd_exact) / scale_nd),
        "load_rel_l2": float(np.linalg.norm(load - load_exact) / scale_load),
        "force_rel": float(np.linalg.norm(force - force_exact) / scale_force),
        "load_exact": load_exact.tolist() if npts == MAP_GRID else [],
    }


def run_benchmark(max_workers: int | None = None) -> dict[str, dict[int, dict]]:
    jobs = [(label, npts) for label in SCHEMES for npts in GRIDS]
    if max_workers is None:
        max_workers = recommended_process_workers(len(jobs), cap=2)
    results = {label: {} for label in SCHEMES}
    print("== oscillatory same-sphere load stress test ==")
    print(f"== using {max_workers} worker process(es) ==")
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(evaluate_scheme, label, npts): (label, npts) for label, npts in jobs}
        for fut in as_completed(futs):
            label, npts = futs[fut]
            rec = fut.result()
            results[label][npts] = rec
            print(f"  {label:>5s} N={npts:2d}  p={rec['value_rel_l2']:.3e}  dn={rec['normal_rel_l2']:.3e}  load={rec['load_rel_l2']:.3e}  F={rec['force_rel']:.3e}")
    return results


def _family_legend(fig, *, bbox=(0.5, 0.995), fontsize: float = 11.6) -> None:
    handles = [
        Line2D(
            [0], [0],
            color=ba.DEFAULT_COLORS[ba.scheme_family(label)],
            ls=ba.DEFAULT_LINESTYLES[ba.scheme_family(label)],
            marker=ba.DEFAULT_MARKERS[ba.scheme_family(label)],
            lw=2.4, ms=6.2, mfc="white",
            mec=ba.DEFAULT_COLORS[ba.scheme_family(label)],
            mew=1.0,
            label=scheme_display_name(label),
        )
        for label in SCHEMES
    ]
    fig.legend(
        handles, [h.get_label() for h in handles],
        ncol=len(SCHEMES), loc="upper center",
        bbox_to_anchor=bbox, frameon=False,
        handlelength=3.0, columnspacing=1.35,
        handletextpad=0.65, fontsize=fontsize,
    )


def make_figure(results: dict[str, dict[int, dict]]) -> None:
    """Render the load-stress test as a pair of two-panel figures.

    The single four-panel figure used previously was visually dense: the
    reader had to jump between an exact-load map, a coarse-grid bar chart,
    and two refinement plots inside one caption. The figure is split in
    two so the surrounding prose can follow a ``setup and coarse-grid
    split / refinement check'' rhythm without interleaving figures that
    address different questions.
    """

    fig = plt.figure(figsize=(13.2, 4.7))
    gs = fig.add_gridspec(
        1, 2, wspace=0.27, left=0.06, right=0.99, top=0.84, bottom=0.15,
    )
    ax_map = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    lat_lon = _fibonacci_lat_lon(N_SURFACE_NODES)
    load_exact = np.asarray(results["BMF6"][MAP_GRID]["load_exact"], dtype=float)
    lim = max(float(np.max(np.abs(load_exact))), 1.0e-18)
    sca = ax_map.scatter(
        lat_lon[:, 1], lat_lon[:, 0], c=load_exact, cmap="RdBu_r",
        s=8, vmin=-lim, vmax=lim,
    )
    ax_map.set_title(r"(a) Exact oscillatory sphere load $\lambda = \partial_n p + 4p$",
                     pad=6, fontweight="semibold")
    ax_map.set_xlabel(r"azimuth $\varphi$ [rad]")
    ax_map.set_ylabel(r"polar $\theta$ [rad]")
    ax_map.set_xlim(-np.pi, np.pi)
    ax_map.set_ylim(0.0, np.pi)
    fig.colorbar(sca, ax=ax_map, fraction=0.046, pad=0.04,
                 label=rf"$\lambda_{{\rm ex}}$ at $t={T_SAMPLE:.2f}$")

    cats = ["pressure", "normal\nderivative", "load\ndensity", "net\nload"]
    centers = np.arange(len(cats), dtype=float)
    width = 0.11
    offsets = np.linspace(-2.5 * width, 2.5 * width, len(SCHEMES))
    for off, label in zip(offsets, SCHEMES):
        rec = results[label][BAR_GRID]
        family = ba.scheme_family(label)
        vals = [rec["value_rel_l2"], rec["normal_rel_l2"],
                rec["load_rel_l2"], rec["force_rel"]]
        ax_bar.bar(centers + off, vals, width=width,
                   color=ba.DEFAULT_COLORS[family], alpha=0.92)
    ax_bar.set_xticks(centers)
    ax_bar.set_xticklabels(cats, linespacing=1.0)
    ax_bar.set_yscale("log")
    ax_bar.set_ylabel(rf"relative error at ${BAR_GRID}^3$")
    ax_bar.set_title(r"(b) Coarse-grid split: spline reconstruction relative to nodal baselines",
                     pad=6, fontweight="semibold")
    _family_legend(fig)
    save_figure(fig, "fig_s6_scattering_surface_stress")
    plt.close(fig)

    fig2 = plt.figure(figsize=(13.2, 4.9))
    gs2 = fig2.add_gridspec(
        1, 2, wspace=0.27, left=0.06, right=0.99, top=0.84, bottom=0.18,
    )
    ax_load = fig2.add_subplot(gs2[0, 0])
    ax_force = fig2.add_subplot(gs2[0, 1])
    for label in SCHEMES:
        family = ba.scheme_family(label)
        recs = [results[label][npts] for npts in GRIDS]
        hs = np.array([rec["h"] for rec in recs], dtype=float)
        load_err = np.array([rec["load_rel_l2"] for rec in recs], dtype=float)
        force_err = np.array([rec["force_rel"] for rec in recs], dtype=float)
        style = dict(
            color=ba.DEFAULT_COLORS[family], ls=ba.DEFAULT_LINESTYLES[family],
            marker=ba.DEFAULT_MARKERS[family], lw=2.15, ms=6.4,
            mfc="white", mec=ba.DEFAULT_COLORS[family], mew=1.0,
        )
        ax_load.loglog(hs, load_err, **style)
        ax_force.loglog(hs, force_err, **style)
    for ax in (ax_load, ax_force):
        ax.invert_xaxis()
        ax.set_xlabel(r"grid spacing $h = 1/(N-1)$")
    ax_load.set_ylabel(r"relative $L^2$ error in $\lambda$")
    ax_force.set_ylabel(r"relative error in $\vec F_\lambda$")
    ax_load.set_title(r"(a) Load-density refinement on the same curved sphere",
                      pad=6, fontweight="semibold")
    ax_force.set_title(r"(b) Integrated-load refinement on the same curved sphere",
                       pad=6, fontweight="semibold")
    ref_kwargs = dict(
        slope=FORMAL_ORDER,
        label=rf'$\propto h^{FORMAL_ORDER}$',
        candidate_regions=("lower right", "mid right", "upper right", "upper left", "lower left"),
    )
    add_theoretical_reference_line(ax_load, **ref_kwargs)
    add_theoretical_reference_line(ax_force, **ref_kwargs)
    _family_legend(fig2)
    save_figure(fig2, "fig_s6_scattering_surface_stress_refinement")
    plt.close(fig2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=None,
                        help="Override the process-pool size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_generated_dirs()
    results = run_benchmark(args.workers)
    make_figure(results)
    c_bar = results["C6"][BAR_GRID]
    summary = {
        "configuration": {"grids": GRIDS, "bar_grid": BAR_GRID, "map_grid": MAP_GRID, "time_sample": T_SAMPLE, "alpha": LOAD_ALPHA, "sphere_center": list(SPHERE_CENTER), "sphere_radius": SPHERE_RADIUS, "n_surface_nodes": N_SURFACE_NODES, "modes": MMS_MODES},
        "bar_grid": {label: {key: results[label][BAR_GRID][key] for key in ("value_rel_l2", "normal_rel_l2", "load_rel_l2", "force_rel")} for label in SCHEMES},
        "coarse_grid_ratios_vs_C6": {
            label: {
                "pressure": float(c_bar["value_rel_l2"] / max(results[label][BAR_GRID]["value_rel_l2"], 1.0e-30)),
                "normal_derivative": float(c_bar["normal_rel_l2"] / max(results[label][BAR_GRID]["normal_rel_l2"], 1.0e-30)),
                "load_density": float(c_bar["load_rel_l2"] / max(results[label][BAR_GRID]["load_rel_l2"], 1.0e-30)),
                "integrated_load": float(c_bar["force_rel"] / max(results[label][BAR_GRID]["force_rel"], 1.0e-30)),
            }
            for label in SCHEMES
        },
    }
    out = RESULTSDIR / "scattering_surface_stress_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
