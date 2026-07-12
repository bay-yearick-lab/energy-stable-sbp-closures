from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "code" / "results" / "offgrid_ensemble.json"
OUT = ROOT / "figures" / "fig_s6_offgrid_ensemble.pdf"


def main() -> None:
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    rows = [row for row in payload["results"] if row["npts"] == 37]
    families = ["E", "C", "SBP", "DRP", "SEM", "BMF", "BMG"]
    labels = ["E6", "C6", "SBP6", "DRP6", "SEM6", "BMF6", "BMG6"]
    colors = {
        "E": "#7f7f7f",
        "C": "#b07d42",
        "SBP": "#6b6b9e",
        "DRP": "#8b6f47",
        "SEM": "#7a5c91",
        "BMF": "#22517a",
        "BMG": "#1f6f6f",
    }
    lookup = {(row["family"], row["metric"]): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), constrained_layout=True)
    for ax, metric, title in zip(axes, ["v", "d"], ["(a) off-grid value", "(b) off-grid derivative"]):
        x = np.arange(len(families))
        med = np.array([lookup[(f, metric)]["median"] for f in families])
        p95 = np.array([lookup[(f, metric)]["p95"] for f in families])
        worst = np.array([lookup[(f, metric)]["worst"] for f in families])
        ax.vlines(x, med, worst, color=[colors[f] for f in families], lw=2.4, alpha=0.42)
        ax.scatter(x, worst, marker="_", s=90, color=[colors[f] for f in families], zorder=3, label="worst")
        ax.scatter(x, p95, marker="o", s=28, facecolors="white", edgecolors=[colors[f] for f in families], linewidths=1.2, zorder=4, label="95th percentile")
        ax.scatter(x, med, marker="o", s=32, color=[colors[f] for f in families], zorder=5, label="median")
        ax.set_yscale("log")
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.set_title(title, loc="left", fontstyle="italic", fontsize=10)
        ax.set_ylabel("relative error")
        ax.grid(axis="y", color="#d9d2c5", lw=0.55, alpha=0.65)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
