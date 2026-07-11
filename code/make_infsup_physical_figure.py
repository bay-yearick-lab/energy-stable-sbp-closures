"""Figure fig_s5_infsup_physical: the no-slip saddle census and the stable pair.

Reads code/results/infsup_physical_bc.json (check_infsup_physical_bc.py) and
renders the two-panel figure backing Propositions prop:noslip-census and
prop:divconf-stable:

    (a) beta_h on the constants-only quotient versus grid, viscous energy
        norm, strong no-slip: the divergence-conforming pair (normal trace
        essential) sits FLAT at every degree, while the pressure-two-below
        same-grid pair decays like h^0.7-0.9 and every other pairing sits at
        zero (spurious nulls inside the physical quotient);
    (b) the null-space census: spurious pressure modes beyond the physical
        constant per pairing, exactly invariant across all degrees and grids.

House style: serif STIX math, pdf.fonttype 42, right-end curve labels in the
curve colour, legend overhead, zero overlapping text.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project_paths import RESULTSDIR, ensure_generated_dirs, save_figure

ensure_generated_dirs()
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.2,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12.0,
        "legend.fontsize": 9.2,
        "xtick.labelsize": 9.4,
        "ytick.labelsize": 9.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
        "savefig.dpi": 300,
    }
)

DEGREE_COLORS = {
    2: "#440154",
    3: "#46327e",
    4: "#365c8d",
    5: "#277f8e",
    6: "#1fa187",
    7: "#4ac16d",
    8: "#a0da39",
}


def main() -> None:
    data = json.loads((RESULTSDIR / "infsup_physical_bc.json").read_text())
    cells = [c for c in data["cells"] if "error" not in c]

    def series(branch, offset):
        out = {}
        for c in cells:
            if c["branch"] != branch:
                continue
            if offset is not None and c["p"] - c["pq"] != offset:
                continue
            out.setdefault(c["p"], []).append(
                (c["n_elem"], c["beta_constants_only_quotient"], c["nulldim"]))
        for p in out:
            out[p].sort()
        return out

    rtn = series("rtn", None)
    bmg2 = series("bmg", 2)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.6, 4.0),
                                     gridspec_kw={"width_ratios": [1.25, 1.0]})

    # ------------------------------------------------------------- panel (a)
    for p in sorted(rtn):
        ne = np.array([r[0] for r in rtn[p]], dtype=float)
        bq = np.array([r[1] for r in rtn[p]])
        ax_a.plot(ne, bq, "-o", ms=3.2, lw=1.6, color=DEGREE_COLORS[p],
                  label=f"$p={p}$")
    # only the outermost flat curves get end labels; the ladder is the legend
    for p, dy in ((2, 1.07), (8, 0.93)):
        ne = np.array([r[0] for r in rtn[p]], dtype=float)
        bq = np.array([r[1] for r in rtn[p]])
        ax_a.annotate(f"$p={p}$", xy=(ne[-1] * 1.10, bq[-1] * dy),
                      color=DEGREE_COLORS[p], fontsize=9.0, va="center")
    for p in sorted(bmg2):
        ne = np.array([r[0] for r in bmg2[p]], dtype=float)
        bq = np.array([r[1] for r in bmg2[p]])
        ax_a.plot(ne, bq, "--s", ms=3.0, lw=1.2, color=DEGREE_COLORS[p],
                  alpha=0.55)
    ax_a.annotate("same-grid $(p,\\,p{-}2)$: $\\beta_h$ decays",
                  xy=(20.0, 0.062), fontsize=9.2, color="#555555")
    ax_a.annotate("divergence-conforming pair: $\\beta_h$ grid-uniform",
                  xy=(8.0, 1.13), fontsize=9.2, color="#222222")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(7, 130)
    ax_a.set_ylim(0.05, 1.45)
    ax_a.set_xlabel("elements per direction")
    ax_a.set_ylabel(r"$\beta_h$ (constants-only quotient, $|v|_{H^1}$)")
    ax_a.set_title("(a) no-slip inf-sup constant under refinement",
                   fontsize=11, pad=26)
    ax_a.legend(ncol=7, loc="lower center", bbox_to_anchor=(0.5, 1.005),
                frameon=False, handlelength=1.1, columnspacing=0.8,
                handletextpad=0.4)

    # ------------------------------------------------------------- panel (b)
    pairs = [
        ("same-grid\nequal order", "bmg", 0),
        ("same-grid\n$(p,\\,p{-}1)$", "bmg", 1),
        ("same-grid\n$(p,\\,p{-}2)$", "bmg", 2),
        ("staggered\nflux pair", "bmf", None),
        ("div-conf.,\ntangential strong", "rt", None),
        ("div-conf.,\nnormal essential", "rtn", None),
    ]
    counts, labels, colors = [], [], []
    for label, br, off in pairs:
        sel = [c for c in cells if c["branch"] == br
               and (off is None or c["p"] - c["pq"] == off)]
        nulls = {c["nulldim"] for c in sel}
        assert len(nulls) == 1, (br, off, nulls)
        counts.append(nulls.pop() - 1)          # beyond the physical constant
        labels.append(label)
        colors.append("#b2453c" if counts[-1] > 0 else "#2e7d55")
    ypos = np.arange(len(pairs))[::-1]
    ax_b.barh(ypos, counts, color=colors, height=0.62)
    for y, c in zip(ypos, counts):
        ax_b.annotate(str(c), xy=(c + 0.3, y), va="center", fontsize=10.5,
                      color="#333333")
    ax_b.annotate("(uniform $\\beta_h$)", xy=(1.4, ypos[-1] - 0.02),
                  va="center", fontsize=8.8, color="#2e7d55")
    ax_b.annotate("($\\beta_h$ decays)", xy=(1.4, ypos[2] - 0.02),
                  va="center", fontsize=8.8, color="#7a5a1e")
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels(labels, fontsize=9.0)
    ax_b.set_xlabel("spurious pressure null modes beyond the constant")
    ax_b.set_xlim(0, 14.2)
    ax_b.set_title("(b) null-space census, invariant over all $p$ and grids",
                   fontsize=11)

    fig.tight_layout(w_pad=2.4)
    save_figure(fig, "fig_s5_infsup_physical")
    print("wrote fig_s5_infsup_physical")


if __name__ == "__main__":
    main()
