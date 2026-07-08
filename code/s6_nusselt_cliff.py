"""Turbulent-channel Nusselt energy-stability cliff figure.
(a) Nu(t) at coarse wall resolution n_z=17: compact family diverges exponentially,
    spline and boundary-optimized SBP stay flat.
(b) Nu vs n_z: convergence to the Chebyshev reference; compact diverges at coarse n_z.
"""
from __future__ import annotations
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from project_paths import RESULTSDIR, ensure_generated_dirs, save_figure

ensure_generated_dirs()
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.2, "axes.labelsize": 11.5, "axes.titlesize": 12.0,
    "legend.fontsize": 9.4, "xtick.labelsize": 9.4, "ytick.labelsize": 9.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 180, "savefig.dpi": 300,
})

# scheme -> (label, color, linestyle)
STY = {
    "I":       ("spline (BMG)",      "#2ca02c", "-"),
    "SBP6-BO": ("SBP6-BO",           "#ff7f0e", "-"),
    "SBP":     ("SBP (classical)",   "#8c564b", "--"),
    "C":       ("compact (DD)",      "#4c72b0", "-"),
    "C6-D2":   ("compact (compact D2)", "#9467bd", "-."),
}
ORDER = ["I", "SBP6-BO", "SBP", "C", "C6-D2"]
REF = 4.307   # Chebyshev n_z=97 independent-discretization reference
REF_CI = 0.104

series = json.load(open(os.path.join(RESULTSDIR, "nusselt_cliff_series.json")))
sweep = json.load(open(os.path.join(RESULTSDIR, "nusselt_sweep.json")))

# collect Nu vs nz per scheme from the sweep
def sweep_rows():
    rows = {}
    def walk(o):
        if isinstance(o, dict):
            if "kind" in o and "nz" in o and "Nu" in o:
                rows.setdefault(o["kind"], {})[o["nz"]] = o["Nu"]
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            [walk(v) for v in o]
    walk(sweep)
    return rows
rows = sweep_rows()
NZS = [17, 25, 33, 49, 65]

from matplotlib.lines import Line2D
fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 4.2))

# ---- panel (a): Nu(t) at n_z=17 ----
axA.axhline(REF, color="0.4", lw=0.9, ls=":", zorder=1)
for k in ORDER:
    lab, col, ls = STY[k]
    s = series["schemes"][k]
    t = np.array(s["t_flowthrough"]); nu = np.array(s["Nu_t"])
    axA.plot(t, nu, ls, color=col, lw=1.6, zorder=3)
axA.set_yscale("log")
axA.set_xlim(0, series["schemes"]["I"]["t_flowthrough"][-1] * 1.02)
axA.set_ylim(1.0, 6e10)
axA.set_xlabel("time (flow-through units)")
axA.set_ylabel(r"Nusselt number $Nu(t)$")
axA.set_title("(a)  coarse wall resolution $n_z=17$", loc="left", fontsize=10.5)
axA.annotate("compact family diverges", xy=(6.6, 3e5), xytext=(1.4, 5e7),
             fontsize=8.8, color="#4c72b0",
             arrowprops=dict(arrowstyle="->", color="#4c72b0", lw=0.9))
axA.text(0.2, REF * 1.45, "spline and SBP6-BO hold\nthe reference", fontsize=8.4,
         color="#2ca02c", va="bottom")

# ---- panel (b): Nu vs n_z ----
axB.axhspan(REF - REF_CI, REF + REF_CI, color="0.86", zorder=0)
axB.axhline(REF, color="0.4", lw=0.9, ls=":", zorder=1)
axB.text(66, REF - REF_CI, "Chebyshev\nreference", color="0.4", fontsize=8.0,
         style="italic", ha="left", va="top")
for k in ORDER:
    lab, col, ls = STY[k]
    if k not in rows:
        continue
    ys = [rows[k].get(nz, np.nan) for nz in NZS]
    axB.plot(NZS, ys, ls, color=col, lw=1.6, marker="o", ms=4, zorder=3)
axB.set_yscale("log")
axB.set_xlabel(r"wall-normal resolution $n_z$")
axB.set_ylabel(r"stationary $Nu$")
axB.set_xlim(15, 84)
axB.set_ylim(1.8, 6e9)
axB.set_xticks(NZS)
axB.set_xticklabels([str(n) for n in NZS])
axB.set_title("(b)  convergence vs resolution", loc="left", fontsize=10.5)

# ---- shared overhead legend ----
handles = [Line2D([0], [0], color=STY[k][1], ls=STY[k][2], lw=1.8, label=STY[k][0])
           for k in ORDER]
fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, 1.02), columnspacing=1.4, handlelength=2.0,
           fontsize=9.2)
fig.tight_layout(rect=[0, 0, 1, 0.93])
save_figure(fig, "fig_s6_nusselt_cliff")
print("wrote fig_s6_nusselt_cliff", flush=True)
