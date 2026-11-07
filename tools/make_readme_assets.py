"""Build README assets (hero banner, result cards, curated paper figures).

Usage: python tools/make_readme_assets.py [repo-root]
All numbers are read from code/results/*.json; nothing is typed in by hand.
The hero banner and the curated paper panels are cropped from figures/*.png,
which the figure drivers in code/ regenerate (for example
`python code/s6_analytic_wake.py`); missing figures are skipped.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
RES, FIG = SRC / "code" / "results", SRC / "figures"
OUT = Path(__file__).resolve().parents[1] / "assets"
OUT.mkdir(exist_ok=True)
plt.style.use(HERE / "lab.mplstyle")
INK, TEAL, AMBER, SLATE, CORAL, GRAY = "#1B2A41", "#0F9D8A", "#F2A541", "#3D5A80", "#E07A5F", "#8D99AE"

# ---------------------------------------------------------------- card 1: inf-sup
cert = json.load(open(RES / "bmf_hdiv_projection_certificate.json"))
sweep = json.load(open(RES / "infsup_sweep.json"))
fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.6), gridspec_kw=dict(width_ratios=[1, 1.15]))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("tealink", [TEAL, INK])
for p in range(2, 9):
    rows = [c for c in cert["grid_cells"] if c["degree"] == p]
    el = [c["npts"] - p for c in rows]
    err = [max(c["beta_relative_error_to_pi"], 1e-16) for c in rows]
    a.semilogy(el, err, "o-", color=cmap((p - 2) / 6), lw=2, ms=5, label=f"p={p}")
a.axhspan(1e-16, 1e-11, color="#E6E9EF", zorder=0)
a.text(4.2, 3e-14, "round-off floor", color=GRAY, fontsize=9)
a.set_xticks([4, 8, 16]); a.set_xlabel("elements per direction")
a.set_ylabel(r"$|\beta_h/\pi - 1|$"); a.set_ylim(1e-14, 1e-2)
a.set_title("Divergence-conforming pair", fontsize=12)
a.legend(ncol=2, fontsize=8, loc="upper right")
for br, key, tgt, mk, col, lab in [("equal_order", "beta", np.pi, "o", TEAL, r"equal order, target $\pi$ (exact 1-D reduction)"),
                                   ("reduced_order", "beta", np.pi * np.sqrt(2), "s", SLATE, r"reduced order, target $\pi\sqrt{2}$")]:
    r = sweep["sweep"][br]
    N = np.array([c["Nv"] for c in r]); e = np.array([abs(c[key] / tgt - 1) for c in r])
    lob = np.array([c["method"].startswith("lobpcg") for c in r])
    b.scatter(N[~lob], np.maximum(e[~lob], 1e-16), marker=mk, s=18, color=col, alpha=.75, label=lab)
    if lob.any():
        b.scatter(N[lob], np.maximum(e[lob], 1e-16), marker=mk, s=26, facecolor="white", edgecolor=col, lw=1.4)
b.set_xscale("log", base=2); b.set_yscale("log"); b.set_ylim(1e-14, 1e0); b.set_ylabel(r"$|\beta_h/\beta_\infty - 1|$")
b.set_xticks([9, 33, 129, 513]); b.set_xticklabels(["9", "33", "129", "513"])
b.set_xlabel("velocity basis functions per direction")
b.set_title("Galerkin pairing, p = 2-8, n up to 513", fontsize=12)
b.scatter([], [], marker="s", s=26, facecolor="white", edgecolor=SLATE, lw=1.4, label="LOBPCG cells (relative residual < 3e-11)")
b.legend(fontsize=7.5, loc="upper right")
fig.suptitle(r"Inf-sup constant $\beta_h \to \pi$ for every degree p = 2-8 (BMF pair)", fontweight="bold", color=INK, fontsize=14)
fig.tight_layout(); fig.savefig(OUT / "card_infsup.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- card 2: order ladder
hc = json.load(open(RES / "heat_cavity_convergence_summary.json"))["measured_slopes"]
fams = [("E", "explicit", GRAY), ("C", "compact", "#C9CED8"), ("SBP", "SBP", AMBER), ("BMF", "BMF (ours)", TEAL), ("BMG", "BMG (ours)", SLATE)]
fig, ax = plt.subplots(figsize=(8, 3.8))
w = 0.16
for i, (k, lab, col) in enumerate(fams):
    vals = [hc[f"{k}{q}"] for q in (4, 6, 8)]
    xs = np.arange(3) + (i - 2) * w
    bars = ax.bar(xs, vals, w * 0.92, color=col, label=lab, zorder=3)
    if k.startswith("BM"):
        for x, v in zip(xs, vals): ax.text(x, v + 0.15, f"{v:.1f}", ha="center", fontsize=8.5, color=INK, fontweight="bold")
for j, q in enumerate((4, 6, 8)):
    ax.hlines(q, j - 0.45, j + 0.45, colors=INK, linestyles="--", lw=1.2, zorder=4)
ax.text(2.47, 8, "formal\norder", fontsize=8.5, color=INK, va="center")
ax.set_xticks(range(3)); ax.set_xticklabels(["q = 4", "q = 6", "q = 8"]); ax.set_ylabel("measured convergence order")
ax.set_ylim(0, 11); ax.legend(ncol=5, fontsize=9, loc="upper left", bbox_to_anchor=(0, 1.0))
ax.set_title(f"Wall-bounded heat cavity: spline routes converge at {hc['BMF6']:.1f} (q=6) and {hc['BMF8']:.1f} (q=8)", fontsize=12.5)
ax.text(0.99, -0.17, "Measured slope of relative L2 error, grids N = 33-129; dispersion-tuned DRP family diverges and is omitted.",
        transform=ax.transAxes, ha="right", fontsize=7.5, color=GRAY)
fig.tight_layout(); fig.savefig(OUT / "card_convergence_orders.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- card 3: cost vs accuracy
wc = json.load(open(RES / "wallclock_cost.json"))["cases"]
style = {"E6": ("explicit", GRAY, "o"), "C6": ("compact", "#B8BFCC", "o"), "SBP6": ("SBP", AMBER, "D"), "DRP6": ("DRP", CORAL, "^"),
         "BMF6": ("BMF (ours)", TEAL, "o"), "BMG6": ("BMG (ours)", SLATE, "s")}
fig, ax = plt.subplots(figsize=(8, 4.2))
best_nodal = min(c["rel_l2"] for c in wc if not c["label"].startswith("BM"))
best_spline = min(c["rel_l2"] for c in wc if c["label"].startswith("BM"))
for lab, (name, col, mk) in style.items():
    cs = sorted([c for c in wc if c["label"] == lab], key=lambda c: c["npts"])
    lw = 2.6 if lab.startswith("BM") else 1.6
    ax.loglog([c["median_cpu_seconds"] for c in cs], [c["rel_l2"] for c in cs], marker=mk, color=col, lw=lw, ms=6, label=name, zorder=3 if lab.startswith("BM") else 2)
ax.axhline(best_nodal, color=AMBER, ls=":", lw=1.2)
ax.text(0.0021, best_nodal * 0.55, f"best nodal error on ladder: {best_nodal:.1e}", color="#B07A20", fontsize=9)
ax.set_xlabel("single-thread CPU time per complete march [s] (median of 5)")
ax.set_ylabel("relative $L^2$ error at T = 2")
ax.legend(ncol=3, fontsize=9, loc="lower left")
ax.set_title(f"Error vs CPU time: splines reach {best_spline:.1e}; nodal routes stop at {best_nodal:.1e}", fontsize=12.5)
fig.tight_layout(); fig.savefig(OUT / "card_cost_accuracy.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- hero banner
if not (FIG / "fig_s6_analytic_wake_bmg_flow.png").exists():
    print("figures/ not generated; result cards rebuilt, hero and paper panels skipped")
    raise SystemExit(0)
src = Image.open(FIG / "fig_s6_analytic_wake_bmg_flow.png").convert("RGB")
crop = src.crop((134, 76, 1350, 1282))            # plot area of panel (a) only: no titles, ticks, colorbar
fig = plt.figure(figsize=(8, 2.8), dpi=200)
fig.patch.set_facecolor(INK)
axi = fig.add_axes([0.60, 0.0, 0.40, 1.0]); axi.imshow(np.asarray(crop), aspect="auto"); axi.axis("off")
grad = np.linspace(1, 0, 256)[None, :] ** 1.6
axg = fig.add_axes([0.60, 0.0, 0.14, 1.0]); axg.imshow(np.dstack([np.full((1, 256, 3), matplotlib.colors.to_rgb(INK)), grad[..., None]]), aspect="auto"); axg.axis("off")
t = fig.add_axes([0, 0, 0.6, 1]); t.axis("off"); t.set_xlim(0, 1); t.set_ylim(0, 1)
t.text(0.06, 0.74, "energy-stable-sbp-closures", color="white", fontsize=17, fontweight="bold", family="DejaVu Sans Mono")
t.text(0.06, 0.58, "High-order B-spline summation-by-parts wall closures", color="#C9D3E0", fontsize=11.5)
t.text(0.06, 0.44, "Provably energy-stable, discretely conservative,\ninf-sup-stable staggered pressure, formal order kept at the wall", color=TEAL, fontsize=9.8, va="top", linespacing=1.5)
t.text(0.06, 0.06, "Yong Yi Bay  &  Kathleen A. Yearick", color=GRAY, fontsize=9)
fig.text(0.985, 0.025, "BMG (q = 6) vorticity and streamlines, Re = 40 wake", color=INK, fontsize=6.5, ha="right",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8))
plt.rcParams["savefig.bbox"] = "standard"; fig.savefig(OUT / "hero.png", dpi=200, facecolor=INK); plt.close(fig)

# ---------------------------------------------------------------- curated paper figures (downsized)
for name, out in [("fig_s6_analytic_wake_bmg_flow.png", "paper_wake_flow.png"), ("fig_s5_checkerboard_fix.png", "paper_checkerboard.png"),
                  ("fig_s6_heat_cavity_fields.png", "paper_heat_cavity.png"), ("fig_s5_green_to_sbp.png", "paper_green_to_sbp.png")]:
    if not (FIG / name).exists():
        print(f"skip {name} (not generated)"); continue
    im = Image.open(FIG / name).convert("RGB")
    if im.width > 1600: im = im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS)
    im.save(OUT / out, optimize=True)
print("done")
