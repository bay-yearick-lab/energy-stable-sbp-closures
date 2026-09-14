"""Figure for the localized wall-packet reflection experiment and the
wall-resolution-spectrum calibration.

Regional errors come from the source-bound ``packet_region_rescore.json``
record.  Spectrum thresholds remain in ``packet_reflection_summary.json``.

Panel (a): full-domain coupled error after one full cycle (two wall bounces)
           at q=6 versus grid, all families, no wall trim anywhere.
Panel (b): three-cycle (six-bounce) error at N=97: full-domain bar next to
           wall-strip bar per family, both norms disclosed side by side.
Panel (c): calibration scatter: the spectrum's P1%-eta points-per-wavelength
           threshold against measured waveform distortion (filled) and
           spectral leakage (open), one point per (family, q); the
           predeclared predictor pairing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))
from project_paths import RESULTSDIR, ensure_generated_dirs, save_figure

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.0,
    "axes.labelsize": 10.8,
    "axes.titlesize": 11.4,
    "legend.fontsize": 9.4,
    "xtick.labelsize": 9.4,
    "ytick.labelsize": 9.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 180, "savefig.dpi": 300,
})

FAMS = ["E", "C", "SBP", "DRP", "SEM", "BMF", "BMG"]
COLOR = {"E": "#555555", "C": "#7a6b4f", "SBP": "#7662a6", "DRP": "#b66b2e",
         "SEM": "#4f7f3f", "BMF": "#22517a", "BMG": "#1f6f6f"}
MARK = {"E": "o", "C": "s", "SBP": "P", "DRP": "X", "SEM": "v",
        "BMF": "^", "BMG": "D"}
LABEL = {"E": "explicit", "C": "compact", "SBP": "SBP", "DRP": "DRP",
         "SEM": "SEM", "BMF": "BMF", "BMG": "BMG"}
DESTROYED = 2.0   # relative error above this = solution destroyed


def get(d, fam, q, npts):
    r = d["runs"].get(f"{fam}{q}_N{npts}")
    return None if (r is None or "cycle1" not in r) else r


def main() -> None:
    ensure_generated_dirs()
    d = json.loads((RESULTSDIR / "packet_region_rescore.json").read_text())
    spectrum = json.loads((RESULTSDIR / "packet_reflection_summary.json").read_text())
    grids = [49, 97, 193]

    fig = plt.figure(figsize=(13.6, 4.85))
    gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.055, right=0.985,
                          top=0.78, bottom=0.15)

    # (a) one-cycle full-domain error vs N at q=6
    ax = fig.add_subplot(gs[0, 0])
    for fam in FAMS:
        xs, ys, dead = [], [], []
        for n in grids:
            r = get(d, fam, 6, n)
            if r is None:
                continue
            e = r["cycle1"]["full_rel_l2"]
            if e > DESTROYED:
                dead.append(n)
            else:
                xs.append(n); ys.append(e)
        if xs:
            ax.loglog(xs, ys, marker=MARK[fam], color=COLOR[fam],
                      label=LABEL[fam], lw=1.4, ms=5.5)
        for n in dead:
            ax.plot([n], [DESTROYED], marker="x", color=COLOR[fam], ms=7,
                    clip_on=False)
    ax.axhline(DESTROYED, color="0.85", lw=0.8, zorder=0)
    ax.text(grids[-1], DESTROYED * 1.35, "destroyed", ha="right",
            fontsize=8.8, color="0.45")
    from matplotlib.ticker import NullFormatter, NullLocator
    ax.set_xticks(grids); ax.set_xticklabels([str(g) for g in grids])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("grid points $N$")
    ax.set_ylabel("full-domain rel. $L^2$, one cycle")
    ax.set_title("(a) Returned packet, two bounces, $q=6$")

    # (b) six-bounce full vs strip at N=97, q=6
    ax = fig.add_subplot(gs[0, 1])
    fams_b = [f for f in FAMS if get(d, f, 6, 97) is not None]
    xpos = np.arange(len(fams_b))
    full = [min(get(d, f, 6, 97)["cycle3"]["full_rel_l2"], DESTROYED)
            for f in fams_b]
    strip = [min(get(d, f, 6, 97)["cycle3"]["wall_strip_rel_l2"], DESTROYED)
             for f in fams_b]
    ax.bar(xpos - 0.19, full, width=0.38, color=[COLOR[f] for f in fams_b],
           label="full domain")
    ax.bar(xpos + 0.19, strip, width=0.38,
           color=[COLOR[f] for f in fams_b], alpha=0.45, hatch="//",
           label="wall strip")
    ax.set_yscale("log")
    ax.set_xticks(xpos); ax.set_xticklabels([LABEL[f] for f in fams_b],
                                            rotation=28, ha="right")
    ax.set_ylabel("rel. $L^2$ after six bounces")
    ax.set_title("(b) Six bounces, $N=97$: both norms")
    ax.legend(frameon=False, loc="upper right")

    # (c) calibration: eta threshold vs distortion (filled) / leakage (open)
    ax = fig.add_subplot(gs[0, 2])
    for fam in FAMS:
        for q in (4, 6, 8):
            th = spectrum["thresholds"].get(f"{fam}{q}")
            r = get(d, fam, q, 97)
            if th is None or r is None:
                continue
            x = th.get("P1_eta")
            c1 = r["cycle1"]
            if x is None or not np.isfinite(x) \
                    or c1["waveform_distortion"] > DESTROYED:
                continue
            ax.semilogy([x], [c1["waveform_distortion"]], marker=MARK[fam],
                        color=COLOR[fam], ms=6.0, ls="none")
            if c1["leakage"] <= DESTROYED:
                ax.semilogy([x], [c1["leakage"]], marker=MARK[fam],
                            mfc="none", color=COLOR[fam], ms=6.0, ls="none")
    ax.set_xlabel(r"spectrum threshold $P_{1\%}^{\eta}$ [points per wavelength]")
    ax.set_ylabel("distortion (filled), leakage (open)")
    ax.set_title("(c) Spectrum calibration, $N=97$")

    family_handles = [
        Line2D([0], [0], color=COLOR[fam], marker=MARK[fam], lw=1.4,
               ms=5.5, label=LABEL[fam])
        for fam in FAMS
    ]
    fig.legend(handles=family_handles, frameon=False, ncol=len(FAMS),
               loc="upper center", bbox_to_anchor=(0.5, 0.985),
               columnspacing=1.15, handlelength=1.7)

    save_figure(fig, "fig_s6_packet_reflection")
    print("wrote fig_s6_packet_reflection.(pdf,png)")


if __name__ == "__main__":
    main()
