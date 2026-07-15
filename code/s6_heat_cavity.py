"""Heat-cavity benchmark driver: 2D heat-cavity sine-mode decay for SBP-closed families.

Runs the SBP-closed heat operators in parallel worker processes, saves a
JSON summary with per-scheme error histories, assembles the static paper
figures, and optionally renders a PNG-frame GIF comparing each numerical
solution against the closed-form exact decay.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

import boundary_spline_analysis as ba
from boundary_spline_analysis import scheme_display_name
from project_paths import FIGDIR, RESULTSDIR, ensure_generated_dirs, recommended_process_workers, save_figure
from heat_cavity_core import (
    per_method_diffusion_dt,
    HeatCavityResult,
    default_heat_modes,
    run_heat_cavity,
    heat_cavity_labels,
    shared_diffusion_dt,
)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10.2,
    'axes.labelsize': 11.2,
    'axes.titlesize': 12.0,
    'legend.fontsize': 9.8,
    'xtick.labelsize': 9.3,
    'ytick.labelsize': 9.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 180,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.12,
    'grid.linewidth': 0.45,
})


EXPERIMENT = {
    "m_center": 28,
    "n_center": 28,
    "half_band": 2,
    "viscosity": 5.0e-5,
    "npts": 81,
    "t_end": 2.0,
    "cfl": 0.01,
    "record_count": 201,
    "snapshot_count": 61,
    "roster": "six_family",
}

FAMILY_COLORS = {
    "SBP6": "#9467bd",
    "BMF6": "#2ca02c",
    "BMG6": "#17a2b8",
}


def _run_one(scheme_label: str, cfg: dict, dt_shared: float) -> HeatCavityResult:
    modes = default_heat_modes(
        m_center=int(cfg["m_center"]),
        n_center=int(cfg["n_center"]),
        half_band=int(cfg["half_band"]),
        viscosity=float(cfg["viscosity"]),
    )
    # per-method stable step (Table tab:dtstep, tier B): a shared step that
    # violates one family's CFL bound measures cost, never spatial accuracy;
    # dt_shared is retained in the signature for the JSON provenance only
    dt_own = per_method_diffusion_dt(
        scheme_label,
        npts=int(cfg["npts"]),
        viscosity=float(cfg["viscosity"]),
        t_end=float(cfg["t_end"]),
    )
    return run_heat_cavity(
        scheme_label,
        modes,
        npts=int(cfg["npts"]),
        t_end=float(cfg["t_end"]),
        cfl=float(cfg["cfl"]),
        dt=float(dt_own),
        record_count=int(cfg["record_count"]),
        snapshot_count=int(cfg["snapshot_count"]),
    )


def run_all_in_parallel(max_workers: int | None = None) -> dict[str, HeatCavityResult]:
    labels = heat_cavity_labels(EXPERIMENT.get("roster", "sbp_closed"))
    dt_shared = shared_diffusion_dt(
        labels,
        npts=int(EXPERIMENT["npts"]),
        viscosity=float(EXPERIMENT["viscosity"]),
        cfl=float(EXPERIMENT["cfl"]),
        t_end=float(EXPERIMENT["t_end"]),
    )
    results: dict[str, HeatCavityResult] = {}
    if max_workers is None:
        max_workers = recommended_process_workers(len(labels), cap=8)
    print(f"== using {max_workers} worker process(es), shared dt = {dt_shared:.3e} "
          f"on {len(labels)} schemes ==")
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_one, lab, EXPERIMENT, dt_shared): lab for lab in labels}
        for f in as_completed(futures):
            lab = futures[f]
            results[lab] = f.result()
            r = results[lab]
            print(f"  {lab:>5s}: steps={r.steps:4d}  rel_L2={r.u_rel_l2[-1]:.3e}  "
                  f"linf={r.u_linf[-1]:.3e}  ppw={r.ppw:.2f}")
    return results


def summarise(results: dict[str, HeatCavityResult]) -> dict:
    labels = heat_cavity_labels(EXPERIMENT.get("roster", "sbp_closed"))
    out = {"configuration": EXPERIMENT, "schemes": {}}
    for lab in labels:
        out["schemes"][lab] = summarise_result(lab, results[lab])
    return out


def summarise_result(label: str, result: HeatCavityResult) -> dict:
    final_linf = float(result.u_linf[-1])
    final_rel_l2 = float(result.u_rel_l2[-1])
    return {
        "display": scheme_display_name(label),
        "stable": bool(np.isfinite(final_linf) and np.isfinite(final_rel_l2)
                       and final_linf < 1.0e2),
        "steps": int(result.steps),
        "dt": float(result.dt),
        "ppw": float(result.ppw),
        "peak_mode": list(result.peak_mode),
        "times": result.times.tolist(),
        "u_rel_l2": result.u_rel_l2.tolist(),
        "u_linf": result.u_linf.tolist(),
        "decay_error": result.decay_error.tolist(),
        "final_rel_l2": final_rel_l2,
        "final_linf": final_linf,
        "max_rel_l2": float(np.max(result.u_rel_l2)),
    }


# ---------------- paper figure (static) ----------------

def _draw_setup_schematic(ax, cfg: dict, peak: tuple[int, int], ppw: float) -> None:
    """Cavity schematic for the 2D heat-decay benchmark.

    All numerics (grid size, viscosity, final time, PPW) are reported in
    the surrounding prose rather than in an on-figure parameter box: the
    schematic itself carries only the domain, the Dirichlet hatching, the
    governing PDE and the symbolic initial condition.
    """
    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(-0.38, 1.28)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    for sp in ax.spines.values():
        sp.set_visible(False)

    sample_m = 6
    sample_n = 6
    grid = 220
    xs = np.linspace(0.0, 1.0, grid)
    ys = np.linspace(0.0, 1.0, grid)
    XS, YS = np.meshgrid(xs, ys, indexing='ij')
    field = np.sin(sample_m * np.pi * XS) * np.sin(sample_n * np.pi * YS)
    ax.pcolormesh(
        XS, YS, field, cmap='RdBu_r',
        vmin=-1.0, vmax=1.0, shading='gouraud',
        alpha=0.50, zorder=1, rasterized=True,
    )

    # Soft inner bevel separating the field from the Dirichlet frame.
    bevel = Rectangle(
        (0.0, 0.0), 1.0, 1.0,
        fill=False, edgecolor='white', lw=2.6, alpha=0.85, zorder=3,
    )
    ax.add_patch(bevel)

    wall_kw = dict(color='black', lw=2.1, zorder=4,
                   solid_capstyle='round', solid_joinstyle='miter')
    ax.plot([0, 1], [0, 0], **wall_kw)
    ax.plot([0, 1], [1, 1], **wall_kw)
    ax.plot([0, 0], [0, 1], **wall_kw)
    ax.plot([1, 1], [0, 1], **wall_kw)

    # Uniform Dirichlet hatching on all four walls.
    tick_kw = dict(color='black', lw=0.85, zorder=4,
                   solid_capstyle='round')
    ntick = 18
    hatch_len = 0.055
    for i in range(ntick):
        s = (i + 0.5) / ntick
        ax.plot([s, s - hatch_len], [0.0, -hatch_len], **tick_kw)
        ax.plot([s, s + hatch_len], [1.0, 1.0 + hatch_len], **tick_kw)
        ax.plot([0.0, -hatch_len], [s, s - hatch_len], **tick_kw)
        ax.plot([1.0, 1.0 + hatch_len], [s, s + hatch_len], **tick_kw)

    ax.text(
        -0.11, -0.11,
        r'$u=0$ on $\partial\Omega$',
        ha='right', va='top', fontsize=12.0, zorder=5,
    )

    ax.text(
        0.5, 1.22,
        r'$u_t = \nu\,(u_{xx}+u_{yy})$',
        ha='center', va='center', fontsize=15.0, zorder=6,
    )
    ax.text(
        0.5, 1.10,
        (r'$u_0(x,y)=\sum_{mn} a_{mn}\sin(m\pi x)\sin(n\pi y),\;'
         r'(m_0,n_0)=(' + f'{peak[0]},{peak[1]}' + r')$'),
        ha='center', va='center', fontsize=12.0, zorder=6,
    )


def _style_for(label: str) -> tuple[str, object]:
    fam = ba.scheme_family(label)
    return ba.DEFAULT_COLORS[fam], ba.DEFAULT_LINESTYLES[fam]


def _style_line_kwargs(lab: str) -> dict:
    fam = ba.scheme_family(lab)
    return dict(
        color=ba.DEFAULT_COLORS[fam],
        ls=ba.DEFAULT_LINESTYLES[fam],
        lw=2.1,
        label=scheme_display_name(lab),
    )


def _make_family_legend(fig, labels: list[str], *,
                        ncol: int | None = None,
                        bbox=(0.5, 0.995)) -> None:
    """Shared top-of-figure family legend, larger than default."""
    from matplotlib.lines import Line2D
    handles = []
    for lab in labels:
        fam = ba.scheme_family(lab)
        handles.append(Line2D(
            [0], [0],
            color=ba.DEFAULT_COLORS[fam],
            ls=ba.DEFAULT_LINESTYLES[fam],
            lw=2.6,
            label=scheme_display_name(lab),
        ))
    fig.legend(
        handles, [h.get_label() for h in handles],
        ncol=ncol or len(labels),
        loc='upper center',
        bbox_to_anchor=bbox,
        frameon=False,
        handlelength=3.4,
        columnspacing=1.8,
        handletextpad=0.8,
        fontsize=12.5,
    )


def make_paper_figure_setup(results: dict[str, HeatCavityResult], stem: str) -> None:
    """Figure 1 of the heat-cavity benchmark: the setup cartoon on its own."""
    labels = heat_cavity_labels()
    cfg = EXPERIMENT
    any_res = results[labels[0]]
    peak = any_res.peak_mode
    ppw = any_res.ppw

    fig = plt.figure(figsize=(8.2, 5.4))
    gs = fig.add_gridspec(1, 1, left=0.08, right=0.92, top=0.93, bottom=0.07)
    ax = fig.add_subplot(gs[0, 0])
    _draw_setup_schematic(ax, cfg, peak, ppw)
    save_figure(fig, stem)
    plt.close(fig)


def make_paper_figure_fields(results: dict[str, HeatCavityResult], stem: str) -> None:
    """Figure 2: exact final field and near-wall error trace at T."""
    labels = heat_cavity_labels()
    cfg = EXPERIMENT
    any_res = results[labels[0]]
    x = np.linspace(0.0, 1.0, any_res.snapshot_u.shape[-1])
    X, Y = np.meshgrid(x, x, indexing='ij')

    fig = plt.figure(figsize=(13.2, 5.2))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.0, 1.25],
        wspace=0.28,
        left=0.065, right=0.985, top=0.80, bottom=0.14,
    )

    # (a) Exact final field
    ax_a = fig.add_subplot(gs[0, 0])
    u_ex_T = any_res.snapshot_u_exact[-1]
    vmax_ex = float(np.max(np.abs(u_ex_T)))
    im = ax_a.pcolormesh(X, Y, u_ex_T, cmap='RdBu_r',
                         vmin=-vmax_ex, vmax=vmax_ex,
                         shading='auto', rasterized=True)
    ax_a.set_aspect('equal', adjustable='box')
    ax_a.set_xticks([0.0, 0.5, 1.0])
    ax_a.set_yticks([0.0, 0.5, 1.0])
    ax_a.set_xlabel(r'$x$')
    ax_a.set_ylabel(r'$y$')
    ax_a.set_title(fr'(a) Exact $u(x,y,T)$ at $T={cfg["t_end"]:.1f}$',
                   fontsize=12.0, fontweight='semibold', pad=8)
    cb = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=9.3)
    cb.set_label(r'$u$', fontsize=10.2)

    # (b) Near-wall error trace
    ax_b = fig.add_subplot(gs[0, 1])
    j_wall = 1
    for lab in labels:
        err = results[lab].snapshot_u[-1] - results[lab].snapshot_u_exact[-1]
        ax_b.plot(x, np.abs(err[:, j_wall]) + 1.0e-18,
                  **_style_line_kwargs(lab))
    ax_b.set_yscale('log')
    ax_b.set_xlabel(r'$x$ along the first interior row above $y=0$')
    ax_b.set_ylabel(r'$|U - u_\mathrm{exact}|$ at $T$')
    ax_b.set_title('(b) Near-wall absolute error at the final time',
                   fontsize=12.0, fontweight='semibold', pad=8)
    ax_b.set_xlim(0.0, 1.0)

    _make_family_legend(fig, labels, bbox=(0.5, 0.995))
    save_figure(fig, stem)
    plt.close(fig)


def make_paper_figure_history(results: dict[str, HeatCavityResult], stem: str) -> None:
    """Figure 3: pointwise and decay histories with final peak ranking."""
    labels = heat_cavity_labels()
    cfg = EXPERIMENT

    fig = plt.figure(figsize=(13.6, 5.2))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.0, 1.0, 0.92],
        wspace=0.32,
        left=0.055, right=0.985, top=0.80, bottom=0.18,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    for lab in labels:
        r = results[lab]
        ax_a.plot(r.times, np.maximum(r.u_linf, 1.0e-14),
                  **_style_line_kwargs(lab))
        ax_b.plot(r.times, np.maximum(r.decay_error, 1.0e-14),
                  **_style_line_kwargs(lab))
    for ax in (ax_a, ax_b):
        ax.set_yscale('log')
        ax.set_xlabel(r'time $t$')
        ax.set_xlim(0.0, cfg['t_end'])
    ax_a.set_ylabel(r'$\|u-u_\mathrm{exact}\|_\infty$')
    ax_b.set_ylabel(r'anti-node decay error')
    ax_a.set_title(r'(a) Pointwise peak error history',
                   fontsize=12.0, fontweight='semibold', pad=8)
    ax_b.set_title(r'(b) Standing-mode decay history',
                   fontsize=12.0, fontweight='semibold', pad=8)

    ax_c = fig.add_subplot(gs[0, 2])
    pairs = sorted(
        ((lab, float(results[lab].u_linf[-1])) for lab in labels),
        key=lambda kv: kv[1],
    )
    best_label, baseline = pairs[0]
    bar_labels = [scheme_display_name(lab) for lab, _ in pairs]
    bar_vals = [v for _, v in pairs]
    bar_colors = [ba.DEFAULT_COLORS[ba.scheme_family(lab)] for lab, _ in pairs]
    bars = ax_c.bar(bar_labels, bar_vals, color=bar_colors,
                    edgecolor='black', lw=0.55)
    ax_c.set_yscale('log')
    ax_c.set_ylabel(r'final $\|u-u_\mathrm{exact}\|_\infty$')
    ax_c.set_title(rf'(c) Final pointwise peak (ratio to {scheme_display_name(best_label)})',
                   fontsize=12.0, fontweight='semibold', pad=8)
    ax_c.tick_params(axis='x', labelrotation=28)
    for tick in ax_c.get_xticklabels():
        tick.set_ha('right')
        tick.set_rotation_mode('anchor')
    ax_c.set_ylim(min(bar_vals) * 0.35, max(bar_vals) * 6.5)
    for b, (lab, v) in zip(bars, pairs):
        ratio = v / baseline
        if lab == best_label:
            tag = 'best'
        elif ratio < 10.0:
            tag = rf'${ratio:.2f}\times$'
        elif ratio < 1000.0:
            tag = rf'${ratio:.1f}\times$'
        else:
            tag = rf'${ratio / 1000.0:.1f}\times 10^3$'
        ax_c.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.18,
                  tag, ha='center', va='bottom', fontsize=9.2)

    _make_family_legend(fig, labels, bbox=(0.5, 0.995))
    save_figure(fig, stem)
    plt.close(fig)


# ---------------- GIF ----------------

_PANEL_BG = (0.987, 0.985, 0.978)
_ACCENT = '#2F4858'


def make_paper_figure_six_family(results: dict[str, HeatCavityResult],
                                 stem: str) -> None:
    """Six-family ranking grouped by q in {4,6,8} for the diffusion-closure demo.

    Panels share y-limits so the spline pair sits on a single horizontal
    floor band across the three orders, and the multi-decade gap to the
    non-spline families is visually constant. The cream panel background
    and ACCENT spine colour follow the project's classy palette.
    """
    qs = (4, 6, 8)
    families = ('E', 'C', 'SBP', 'DRP', 'BMF', 'BMG')

    # Pre-collect per-q values; compute a shared y-range from finite data.
    per_q = []
    all_finite = []
    for q in qs:
        labels = [f"{fam}{q}" for fam in families]
        vals = []
        for lab in labels:
            r = results.get(lab)
            if r is None:
                vals.append(np.nan); continue
            v = float(r.u_linf[-1])
            vals.append(v if np.isfinite(v) and v < 1.0e2 else np.nan)
            if np.isfinite(vals[-1]):
                all_finite.append(vals[-1])
        per_q.append((q, labels, vals))
    if all_finite:
        ymin = min(all_finite) * 0.18
        ymax = max(all_finite) * 9.0
    else:
        ymin, ymax = 1.0e-7, 1.0e1

    fig = plt.figure(figsize=(13.6, 4.9), facecolor='white')
    gs = fig.add_gridspec(1, 3, wspace=0.16, left=0.062, right=0.985,
                          top=0.80, bottom=0.20)
    metric_label = r'final $\|u-u_\mathrm{exact}\|_\infty$'

    for j, (q, labels, vals) in enumerate(per_q):
        ax = fig.add_subplot(gs[0, j])
        ax.set_facecolor(_PANEL_BG)
        positions = np.arange(len(families))
        colors = [ba.DEFAULT_COLORS[ba.scheme_family(lab)] for lab in labels]

        # Spline floor band: lowest finite value in this panel becomes a
        # quiet horizontal stripe so the gap between non-spline and spline
        # rails is visible without annotation.
        spline_vals = [v for lab, v in zip(labels, vals)
                       if ba.scheme_family(lab) in ('BMF', 'BMG') and np.isfinite(v)]
        nonspline_vals = [v for lab, v in zip(labels, vals)
                          if ba.scheme_family(lab) not in ('BMF', 'BMG') and np.isfinite(v)]
        if spline_vals and nonspline_vals:
            sp_max = max(spline_vals)
            ax.axhspan(ymin, sp_max, color='#e8efe6', alpha=0.85, zorder=0)

        bar_vals = [v if np.isfinite(v) else ymin for v in vals]
        bars = ax.bar(positions, bar_vals, color=colors,
                      edgecolor=_ACCENT, lw=0.55, alpha=0.92, zorder=2,
                      width=0.72)
        for p, lab, v, b in zip(positions, labels, vals, bars):
            if not np.isfinite(v):
                ax.text(p, ymin * 2.4, 'unstable',
                        ha='center', va='bottom', fontsize=8.6,
                        color=_ACCENT, fontstyle='italic',
                        family='serif', zorder=4)
                b.set_alpha(0.16)
                b.set_hatch('//')
                b.set_edgecolor('#9a8f88')

        ax.set_yscale('log')
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(positions)
        ax.set_xticklabels(families, fontsize=10.0, color=_ACCENT)
        ax.tick_params(axis='y', colors=_ACCENT, labelsize=9.0)
        for spine in ax.spines.values():
            spine.set_color(_ACCENT)
            spine.set_linewidth(0.55)
        ax.grid(True, axis='y', which='major', color='white',
                linewidth=0.6, alpha=0.85, zorder=1)
        title_str = rf'$q={q}$'
        ax.set_title(title_str, fontsize=12.5, color=_ACCENT, pad=8)
        if j == 0:
            ax.set_ylabel(metric_label, color=_ACCENT)
        else:
            ax.set_yticklabels([])

    save_figure(fig, stem)
    plt.close(fig)


def make_gif(results: dict[str, HeatCavityResult], stem: str) -> None:
    labels = heat_cavity_labels()
    ref_label = min(labels, key=lambda lab: results[lab].snapshot_u.shape[0])
    any_res = results[ref_label]
    npts = any_res.snapshot_u.shape[-1]
    nframes = any_res.snapshot_u.shape[0]
    x = np.linspace(0.0, 1.0, npts)
    X, Y = np.meshgrid(x, x, indexing='ij')

    # Shared error colour scale across all frames & schemes.
    err_max = 0.0
    for lab in labels:
        r = results[lab]
        for f in range(r.snapshot_u.shape[0]):
            err = r.snapshot_u[f] - r.snapshot_u_exact[f]
            err_max = max(err_max, float(np.max(np.abs(err))))
    err_max = max(err_max, 1.0e-6)

    frames: list[Image.Image] = []
    tmpdir = FIGDIR / f"{stem}_frames"
    tmpdir.mkdir(exist_ok=True)
    for f in range(nframes):
        fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.1),
                                 gridspec_kw={'hspace': 0.30, 'wspace': 0.14})
        t_f = any_res.snapshot_times[f]

        u_ex_f = any_res.snapshot_u_exact[f]
        vmax_ex = float(np.max(np.abs(any_res.snapshot_u_exact[0])))
        panel_map = [('exact', axes[0, 0]), ('SBP6', axes[0, 1]), ('BMF6', axes[1, 0]), ('BMG6', axes[1, 1])]
        for panel, ax in panel_map:
            if panel == 'exact':
                ax.pcolormesh(X, Y, u_ex_f, cmap='RdBu_r',
                              vmin=-vmax_ex, vmax=vmax_ex, shading='auto')
                ax.set_title(f'Exact  $u(x,y,t)$\n$t={t_f:.2f}$', fontsize=11)
            else:
                err = results[panel].snapshot_u[f] - results[panel].snapshot_u_exact[f]
                ax.pcolormesh(X, Y, err, cmap='RdBu_r',
                              vmin=-err_max, vmax=err_max, shading='auto')
                fam = scheme_display_name(panel)
                ax.set_title(f'{fam}  $u - u_\\mathrm{{ex}}$\n'
                             f'$\\|\\cdot\\|_\\infty = {np.max(np.abs(err)):.2e}$',
                             fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])

        peak = any_res.peak_mode
        fig.suptitle(
            (f"Heat cavity: $u_t = \\nu \\Delta u$,   "
             f"$u_0 = \\sin({peak[0]}\\pi x)\\sin({peak[1]}\\pi y)$,  "
             f"$N={EXPERIMENT['npts']}$,  "
             f"PPW={any_res.ppw:.2f},  "
             f"colour scale shared: $\\pm${err_max:.2e}"),
            fontsize=12)
        frame_path = tmpdir / f"frame_{f:04d}.png"
        fig.savefig(frame_path, dpi=110, bbox_inches='tight')
        plt.close(fig)
        with Image.open(frame_path) as img:
            frames.append(img.convert('P', palette=Image.ADAPTIVE))

    # Assemble GIF
    gif_path = FIGDIR / f"{stem}.gif"
    frames[0].save(
        gif_path,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=True,
    )
    print(f"wrote {gif_path} ({len(frames)} frames)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=None,
                        help="Override the process-pool size.")
    parser.add_argument("--with-gif", action="store_true",
                        help="Render the optional animated companion GIF.")
    parser.add_argument("--rescore-splines", action="store_true",
                        help="Replace only BMF/BMG summary rows using actual-node weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_generated_dirs()
    out_json = RESULTSDIR / "heat_cavity_summary.json"
    if args.rescore_splines:
        summary = json.loads(out_json.read_text())
        labels = ("BMF4", "BMG4", "BMF6", "BMG6", "BMF8", "BMG8")
        all_labels = heat_cavity_labels(EXPERIMENT.get("roster", "sbp_closed"))
        dt_shared = shared_diffusion_dt(
            all_labels, npts=int(EXPERIMENT["npts"]),
            viscosity=float(EXPERIMENT["viscosity"]), cfl=float(EXPERIMENT["cfl"]),
            t_end=float(EXPERIMENT["t_end"]),
        )
        for label in labels:
            result = _run_one(label, EXPERIMENT, dt_shared)
            summary["schemes"][label] = summarise_result(label, result)
        out_json.write_text(json.dumps(summary, indent=2))
        print(f"rescored six spline summary rows: {out_json}")
        return
    print("== running three SBP-closed heat schemes in parallel ==")
    results = run_all_in_parallel(args.workers)

    summary = summarise(results)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")

    print("== paper figures (split) ==")
    make_paper_figure_setup(results, "fig_s6_heat_cavity_setup")
    print(f"wrote {FIGDIR}/fig_s6_heat_cavity_setup.(pdf,png)")
    make_paper_figure_fields(results, "fig_s6_heat_cavity_fields")
    print(f"wrote {FIGDIR}/fig_s6_heat_cavity_fields.(pdf,png)")
    make_paper_figure_history(results, "fig_s6_heat_cavity_history")
    print(f"wrote {FIGDIR}/fig_s6_heat_cavity_history.(pdf,png)")
    if EXPERIMENT.get("roster", "sbp_closed") == "six_family":
        make_paper_figure_six_family(results, "fig_s6_heat_cavity_six_family")
        print(f"wrote {FIGDIR}/fig_s6_heat_cavity_six_family.(pdf,png)")

    if args.with_gif:
        print("== GIF ==")
        make_gif(results, "heat_cavity")
    else:
        print("== skipping GIF (not used by the manuscript) ==")


if __name__ == "__main__":
    main()
