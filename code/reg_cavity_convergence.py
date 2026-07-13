#!/usr/bin/env python3
"""Self-convergence of the regularized-lid cavity ladder.

Evaluates each run's velocity field on one shared dense grid with the
INDEPENDENT spline evaluator (verify_cavity_blind machinery), measures the
L2 field difference of each grid against the finest as reference, and
reports the observed Richardson orders. The regularized lid (quartic,
exactly representable in the degree-4 spline space) removes the corner
singularity, so this ladder is the meaningful accuracy statement for the
coupled solver.

Usage: uv run --with numpy,scipy python reg_cavity_convergence.py [N ...]
Writes code/results/reg_cavity_convergence.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))
from verify_cavity_blind import eval_field, load_state  # noqa: E402

PREFIX = "reg400"
GRIDS = [int(a) for a in sys.argv[1:]] or [24, 32, 48, 64]


def dense_fields(prefix: str, pts: np.ndarray):
    nx, nz, p, u, w = load_state(prefix)
    return (eval_field(u, p, pts, pts), eval_field(w, p, pts, pts))


def main() -> None:
    pts = np.linspace(0.0, 1.0, 401)
    ref_n = GRIDS[-1]
    uref, wref = dense_fields(f"{PREFIX}_N{ref_n}", pts)
    scale = float(np.sqrt(np.mean(uref ** 2 + wref ** 2)))
    out = {"reference_grid": ref_n, "grids": GRIDS[:-1], "rows": []}
    errs = []
    for n in GRIDS[:-1]:
        u, w = dense_fields(f"{PREFIX}_N{n}", pts)
        e = float(np.sqrt(np.mean((u - uref) ** 2 + (w - wref) ** 2)) / scale)
        summ = json.loads((_CODE / "cavity" / "data" /
                           f"{PREFIX}_N{n}_run_summary.json").read_text())
        out["rows"].append({"N": n, "rel_l2_vs_ref": e,
                            "u_min": summ["u_centerline_min"],
                            "final_max_dudt": summ["final_max_dudt"]})
        errs.append((n, e))
        print(f"N={n}: rel L2 vs N{ref_n} = {e:.3e}")
    for (n1, e1), (n2, e2) in zip(errs, errs[1:]):
        order = np.log(e1 / e2) / np.log(n2 / n1)
        out.setdefault("orders", []).append({"pair": [n1, n2], "order": float(order)})
        print(f"order({n1}->{n2}) = {order:.2f}")
    (_CODE / "results" / "reg_cavity_convergence.json").write_text(
        json.dumps(out, indent=1))
    print("wrote results/reg_cavity_convergence.json")


if __name__ == "__main__":
    main()
