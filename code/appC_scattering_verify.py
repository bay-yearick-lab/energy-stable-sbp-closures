"""Same-setup manufactured-solution grid sweep for the Section 6.10 acoustic
cavity benchmark.

The cavity benchmark runs a soft-wall box-eigenmode superposition in a
bounded cubic cavity with an interior observation sphere and diagnoses field,
surface-pressure, force, receiver, and energy behavior. Before the fixed-grid
ranking is trusted, the same solver path is exercised here on the same
soft-wall manufactured solution across a grid sweep.

We run two complementary convergence sweeps:

1. A Cartesian-box sweep checks how the three-dimensional first-order
   acoustic solver (tensor-product operators, box-wall enforcement, RK4
   time integration, total energy accounting) behaves under grid refinement
   on the same benchmark horizon.

2. An off-grid sphere-surface sweep checks the same refinement trend for the
   integrated force :math:`\\vec F(t) = -\\!\\oint p\\,\\vec n\\,dS` computed
   on the exact analytic sphere. This is the same
   same-point off-grid surface pipeline used in the cavity benchmark, so any
   problem in the surface reconstruction or in the surface quadrature
   shows up here rather than being masked by the scattering setup.

The two sweeps share the same exact box-eigenmode data, sphere geometry, time
horizon, and CFL as the cavity benchmark; they differ only in the grid
resolution and in which observable is extracted. This keeps the appendix tied
to the exact same benchmark identity used in the main text.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import boundary_spline_analysis as ba
from boundary_spline_analysis import scheme_display_name
from plotting_helpers import add_theoretical_reference_line
import scattering_core as sc
from project_paths import FIGDIR, RESULTSDIR, ensure_generated_dirs, recommended_process_workers, save_figure


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.4,
    "axes.labelsize": 11.4,
    "axes.titlesize": 12.2,
    "legend.fontsize": 10.4,
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


SCHEMES = ba.comparison_labels(6)
MMS_MODES = [
    (2, 1, 1, 1.00),
    (1, 2, 1, 0.70),
    (1, 1, 2, 0.50),
]
C_SOUND = 1.0
RHO0 = 1.0
T_END = 0.60
CFL = 0.12
SPHERE_CENTER = (0.42, 0.48, 0.52)
SPHERE_RADIUS = 0.15
N_SURFACE_NODES = 1600
GRIDS = [17, 25, 33, 41, 49]
FORMAL_ORDER = 6


def _run_one_bmf_staggered(label: str, npts: int) -> dict:
    mms = sc.BoxEigenmode(modes=MMS_MODES, c=C_SOUND, rho0=RHO0, length=1.0)
    op = sc.BMFStaggeredOperator3D(label, npts)
    x = op.x
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    state = sc.bmf_staggered_exact_state(op, mms, 0.0)
    sphere = sc.build_sphere_surface(op, SPHERE_CENTER, SPHERE_RADIUS, N_SURFACE_NODES)

    h = op.h_nominal
    dt = CFL * op.h / C_SOUND
    nsteps = int(np.ceil(T_END / dt))
    dt = T_END / nsteps

    for _step in range(nsteps):
        sc.bmf_staggered_rk4_step(state, dt, op, RHO0, C_SOUND)

    P_exact = mms.pressure(X, Y, Z, T_END)
    err_field = state.p - P_exact
    rel_l2 = float(np.linalg.norm(err_field) / np.linalg.norm(P_exact))

    p_surf = sphere.interpolate(state.p)
    p_surf_exact = mms.pressure_on_sphere(sphere.points, T_END)
    rel_l2_surf = float(np.linalg.norm(p_surf - p_surf_exact) / np.linalg.norm(p_surf_exact))

    F_num = sc.integrated_force(sphere, p_surf)
    F_exact = mms.integrated_force_on_sphere(sphere, T_END)
    F_err = float(np.linalg.norm(F_num - F_exact) / max(np.linalg.norm(F_exact), 1.0e-30))

    return {
        "label": label,
        "npts": int(npts),
        "h": float(h),
        "h_min": float(op.h),
        "dt": float(dt),
        "nsteps": int(nsteps),
        "rel_l2": rel_l2,
        "rel_l2_surf": rel_l2_surf,
        "force_rel_err": F_err,
        "force_num": F_num.tolist(),
        "force_exact": F_exact.tolist(),
        "operator_route": "staggered primal-pressure and derivative-grid velocity complex",
    }


def _run_one(label: str, npts: int) -> dict:
    if ba.scheme_family(label) == "BMF":
        return _run_one_bmf_staggered(label, npts)

    mms = sc.BoxEigenmode(modes=MMS_MODES, c=C_SOUND, rho0=RHO0, length=1.0)
    op = sc.Operator3D(label, npts)
    x = op.x
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    P0 = mms.pressure(X, Y, Z, 0.0)
    U0, V0, W0 = mms.velocity(X, Y, Z, 0.0)
    state = sc.AcousticState(p=P0.copy(), u=U0.copy(), v=V0.copy(), w=W0.copy())
    sc._apply_box_bc(state)

    sphere = sc.build_sphere_surface(op, SPHERE_CENTER, SPHERE_RADIUS, N_SURFACE_NODES)

    h = op.h_nominal
    dt = CFL * op.h / C_SOUND
    nsteps = int(np.ceil(T_END / dt))
    dt = T_END / nsteps

    for step in range(nsteps):
        sc.rk4_step(state, dt, op, RHO0, C_SOUND, None, 0.0, "soft_pressure", None, step * dt)

    P_exact = mms.pressure(X, Y, Z, T_END)
    err_field = state.p - P_exact
    rel_l2 = float(np.linalg.norm(err_field) / np.linalg.norm(P_exact))

    p_surf = sphere.interpolate(state.p)
    p_surf_exact = mms.pressure_on_sphere(sphere.points, T_END)
    rel_l2_surf = float(np.linalg.norm(p_surf - p_surf_exact) / np.linalg.norm(p_surf_exact))

    F_num = sc.integrated_force(sphere, p_surf)
    F_exact = mms.integrated_force_on_sphere(sphere, T_END)
    F_err = float(np.linalg.norm(F_num - F_exact) / max(np.linalg.norm(F_exact), 1.0e-30))

    return {
        "label": label,
        "npts": int(npts),
        "h": float(h),
        "h_min": float(op.h),
        "dt": float(dt),
        "nsteps": int(nsteps),
        "rel_l2": rel_l2,
        "rel_l2_surf": rel_l2_surf,
        "force_rel_err": F_err,
        "force_num": F_num.tolist(),
        "force_exact": F_exact.tolist(),
    }


def run_sweeps(max_workers: int | None = None) -> dict:
    jobs = [(lab, n) for lab in SCHEMES for n in GRIDS]
    if max_workers is None:
        max_workers = recommended_process_workers(len(jobs), cap=2)
    print(f"== using {max_workers} worker process(es) ==")
    records: dict[str, list[dict]] = {lab: [] for lab in SCHEMES}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_run_one, lab, n): (lab, n) for lab, n in jobs}
        for f in as_completed(futs):
            lab, n = futs[f]
            try:
                r = f.result()
            except Exception as exc:
                print(f"[{lab} N={n}] FAILED: {exc}")
                raise
            records[lab].append(r)
            print(f"  {lab:>5s} N={n:3d}  h={r['h']:.5f}  rel_L2={r['rel_l2']:.3e}  "
                  f"rel_L2_surf={r['rel_l2_surf']:.3e}  F_err={r['force_rel_err']:.3e}")
    for lab in records:
        records[lab].sort(key=lambda r: r["h"], reverse=True)
    return records


def load_records_from_summary(path: Path) -> dict[str, list[dict]]:
    summary = json.loads(path.read_text())
    records: dict[str, list[dict]] = {}
    for lab in SCHEMES:
        recs = [
            {
                "label": lab,
                "npts": int(row["npts"]),
                "h": float(row["h"]),
                "rel_l2": float(row["rel_l2"]),
                "rel_l2_surf": float(row["rel_l2_surf"]),
                "force_rel_err": float(row["force_rel_err"]),
                "force_num": row["force_num"],
                "force_exact": row["force_exact"],
            }
            for row in summary["sweeps"][lab]
        ]
        recs.sort(key=lambda r: r["h"], reverse=True)
        records[lab] = recs
    return records


def fit_slope(xs, ys, tail=3) -> float:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    k = min(tail, xs.size)
    lx = np.log(xs[-k:])
    ly = np.log(ys[-k:])
    a, _ = np.polyfit(lx, ly, 1)
    return float(a)


def _family_legend(fig, labels, *, bbox=(0.5, 0.995), ncol=None) -> None:
    handles = []
    for lab in labels:
        fam = ba.scheme_family(lab)
        handles.append(Line2D(
            [0], [0], color=ba.DEFAULT_COLORS[fam],
            ls=ba.DEFAULT_LINESTYLES[fam], lw=2.6,
            marker=ba.DEFAULT_MARKERS[fam], ms=7.0,
            mfc="white", mec=ba.DEFAULT_COLORS[fam], mew=1.1,
            label=scheme_display_name(lab),
        ))
    fig.legend(
        handles, [h.get_label() for h in handles],
        ncol=(ncol or len(labels)), loc="upper center",
        bbox_to_anchor=bbox, frameon=False,
        handlelength=3.4, columnspacing=1.8, handletextpad=0.8,
        fontsize=12.2,
    )


def make_figure(records: dict[str, list[dict]], stem: str) -> dict:
    labels = SCHEMES

    fig = plt.figure(figsize=(13.6, 5.3))
    gs = fig.add_gridspec(
        1, 3, wspace=0.30,
        left=0.065, right=0.99, top=0.79, bottom=0.15,
    )
    ax_field = fig.add_subplot(gs[0, 0])
    ax_surf = fig.add_subplot(gs[0, 1])
    ax_force = fig.add_subplot(gs[0, 2])

    measured = {}
    for lab in labels:
        fam = ba.scheme_family(lab)
        recs = records[lab]
        hs = np.array([r["h"] for r in recs])
        ef = np.array([r["rel_l2"] for r in recs])
        es = np.array([r["rel_l2_surf"] for r in recs])
        efc = np.array([r["force_rel_err"] for r in recs])
        style = dict(
            color=ba.DEFAULT_COLORS[fam],
            ls=ba.DEFAULT_LINESTYLES[fam],
            marker=ba.DEFAULT_MARKERS[fam],
            lw=2.1, ms=7.2,
            mfc="white", mec=ba.DEFAULT_COLORS[fam], mew=1.2,
        )
        ax_field.loglog(hs, ef, **style)
        ax_surf.loglog(hs, np.maximum(es, 1.0e-15), **style)
        ax_force.loglog(hs, np.maximum(efc, 1.0e-15), **style)
        measured[lab] = {
            "field_slope": fit_slope(hs, ef),
            "surf_slope": fit_slope(hs, np.maximum(es, 1.0e-15)),
            "force_slope": fit_slope(hs, np.maximum(efc, 1.0e-15)),
        }

    for ax in (ax_field, ax_surf, ax_force):
        ax.invert_xaxis()
        ax.set_xlabel(r"grid spacing $h = 1/(N-1)$")

    ax_field.set_ylabel(r"rel. $L^2$ error in $p(T)$ on the grid")
    ax_field.set_title(r"(a) 3D volumetric pressure at $T$",
                       fontsize=12.0, fontweight="semibold", pad=8)
    ax_surf.set_ylabel(r"rel. $L^2$ error in $p(\mathbf{x}_S, T)$ on the sphere")
    ax_surf.set_title(r"(b) Off-grid sphere-surface pressure",
                      fontsize=12.0, fontweight="semibold", pad=8)
    ax_force.set_ylabel(r"rel. error in $\vec F(T) = -\!\oint p\,\vec n\,dS$")
    ax_force.set_title(r"(c) Integrated surface force",
                       fontsize=12.0, fontweight="semibold", pad=8)

    ref_kwargs = dict(
        slope=FORMAL_ORDER,
        label=rf'$\propto h^{FORMAL_ORDER}$',
        candidate_regions=("lower right", "mid right", "upper right", "upper left", "lower left"),
    )
    add_theoretical_reference_line(ax_field, **ref_kwargs)
    add_theoretical_reference_line(ax_surf, **ref_kwargs)
    add_theoretical_reference_line(ax_force, **ref_kwargs)

    _family_legend(fig, labels, bbox=(0.5, 0.995))
    save_figure(fig, stem)
    plt.close(fig)
    return measured


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=None,
                        help="Override the process-pool size.")
    parser.add_argument(
        "--from-summary",
        action="store_true",
        help="Regenerate the figure from the existing summary JSON instead of rerunning the sweep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_generated_dirs()
    if args.from_summary:
        summary_path = RESULTSDIR / "scattering_verify_summary.json"
        print(f"== loading existing sweep data from {summary_path} ==")
        records = load_records_from_summary(summary_path)
    else:
        print("== 3D acoustic cavity same-setup grid sweep ==")
        records = run_sweeps(args.workers)
    print("== figure ==")
    measured = make_figure(records, "fig_appC_scattering_verify")
    print(f"wrote {FIGDIR}/fig_appC_scattering_verify.(pdf,png)")

    summary = {
        "configuration": {
            "modes": MMS_MODES, "c": C_SOUND, "rho0": RHO0,
            "t_end": T_END, "cfl": CFL,
            "sphere_center": list(SPHERE_CENTER),
            "sphere_radius": SPHERE_RADIUS,
            "n_surface_nodes": N_SURFACE_NODES,
            "grids": GRIDS,
        },
        "sweeps": {
            lab: [
                {
                    **{k: r[k] for k in ("npts", "h", "rel_l2", "rel_l2_surf",
                                         "force_rel_err", "force_num", "force_exact")},
                    "operator_route": r.get("operator_route", "same-grid first-derivative route"),
                }
                for r in records[lab]
            ]
            for lab in SCHEMES
        },
        "measured_slopes": measured,
    }
    out = RESULTSDIR / "scattering_verify_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
