"""Two-branch Taylor-Green no-slip decay convergence figure
(reads results/twobranch_tgdecay_summary.json).

Panel (a): centerline profile error against the N=129 reference versus grid,
           for BMG and BMF, with p+1 = 5 reference slope; both routes converge
           at high order on the smooth wall-bounded flow.
Panel (b): cross-branch agreement (BMF vs BMG at matched N) and the
           conservation floor (skew production / constraint divergence),
           showing the two independent solvers land on the same solution
           while holding the theorem-level identities at roundoff.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))
from project_paths import RESULTSDIR, save_figure, ensure_generated_dirs

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.0, "axes.labelsize": 10.8, "axes.titlesize": 11.4,
    "legend.fontsize": 9.5, "xtick.labelsize": 9.4, "ytick.labelsize": 9.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 180, "savefig.dpi": 300,
})

BMG_C, BMF_C = "#2ca02c", "#d62728"


def main() -> None:
    ensure_generated_dirs()
    d = json.loads((RESULTSDIR / "twobranch_tgdecay_summary.json").read_text())
    rows = d["results"]

    def series(branch):
        r = sorted([c for c in rows if c["branch"] == branch], key=lambda c: c["N"])
        return np.array([c["N"] for c in r]), np.array([c["u_profile_rel_l2"] for c in r])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.145, wspace=0.28)

    # (a) convergence
    for branch, c, mk in (("BMG", BMG_C, "D"), ("BMF", BMF_C, "^")):
        N, e = series(branch)
        axA.loglog(N, e, marker=mk, color=c, lw=1.5, ms=6, label=f"{branch} route")
    Nref = np.array([17, 65], float)
    e0 = series("BMG")[1][0] * 1.6
    axA.loglog(Nref, e0 * (Nref[0] / Nref) ** 5, "k--", lw=1.0,
               label=r"slope $p{+}1=5$")
    axA.set_xlabel("grid points per direction $N$")
    axA.set_ylabel(r"centerline profile rel. $L^2$ vs $N{=}129$")
    axA.set_title("(a) Both routes converge, no-slip decay")
    axA.set_xticks([17, 25, 33, 49, 65])
    axA.set_xticklabels(["17", "25", "33", "49", "65"])
    from matplotlib.ticker import NullLocator, NullFormatter
    axA.xaxis.set_minor_locator(NullLocator())
    axA.xaxis.set_minor_formatter(NullFormatter())
    axA.legend(frameon=False, loc="lower left")

    # (b) cross-branch agreement + conservation floor
    cb = d.get("cross_branch", {})
    Ns = sorted(int(k) for k in cb)
    axB.loglog(Ns, [cb[str(n)] for n in Ns], marker="o", color="#7f4fa0",
               lw=1.5, ms=6, label="BMF vs BMG (matched $N$)")
    # conservation floor band
    prods = [abs(c.get("ke_production") or 0.0) for c in rows if c["branch"] == "BMG"]
    floor = max([p for p in prods if p > 0] + [1e-17])
    axB.axhspan(1e-18, floor, color="0.85", alpha=0.6)
    axB.text(Ns[len(Ns) // 2], floor * 0.5, "skew production / constraint floor",
             ha="center", va="top", fontsize=8.6, color="0.4")
    axB.set_xlabel("grid points per direction $N$")
    axB.set_ylabel("relative magnitude")
    axB.set_title("(b) Same solution, identities at roundoff")
    axB.set_xticks(Ns); axB.set_xticklabels([str(n) for n in Ns])
    axB.xaxis.set_minor_locator(NullLocator())
    axB.xaxis.set_minor_formatter(NullFormatter())
    axB.set_ylim(1e-18, 1e-2)
    axB.legend(frameon=False, loc="upper right")

    save_figure(fig, "fig_s6_twobranch_tgdecay")
    print("wrote fig_s6_twobranch_tgdecay.(pdf,png)")


if __name__ == "__main__":
    main()
