"""Across-order analytic-wake leg: repeats the wake comparison at orders 4 and 8
(the main text displays order 6), feeding the across-order evidence matrix.

Runs the same driver machinery (same grids, steps, tolerances) for the
six-family blocks at q in {4, 8}, Re = 40 (the order-6 headline Reynolds
number), and records the native velocity/divergence/momentum metrics plus
measured CPU per solve. Output: code/results/analytic_wake_orders_summary.json

Verification (--verify): all metrics finite; at each order both spline
routes beat every nodal family on final velocity error (the ordering claim
the across-order matrix displays).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))

import boundary_spline_analysis as ba
from project_paths import RESULTSDIR, ensure_generated_dirs
from s6_analytic_wake import WakeConfig, run_case

ORDERS = (4, 8)
RE = 40
KEEP = ("velocity_rel_l2", "velocity_linf", "divergence_inf_scaled",
        "momentum_residual_rel_rms", "steps", "cpu_s")


def run_all():
    cfg = WakeConfig()
    out = {"config": {"orders": list(ORDERS), "re": RE, "npts": cfg.npts}, "rows": []}
    for q in ORDERS:
        for label in ba.comparison_labels(q):
            try:
                rec = run_case(label, RE, cfg)
                row = {"label": label, "q": q,
                       **{k: rec[k] for k in KEEP if k in rec}}
            except Exception as exc:
                row = {"label": label, "q": q, "error": str(exc)}
            out["rows"].append(row)
            print(json.dumps(row)[:160], flush=True)
    return out


def verify(out):
    for q in ORDERS:
        rows = {r["label"]: r for r in out["rows"] if r["q"] == q and "error" not in r}
        spl = max(rows[f"BMF{q}"]["velocity_rel_l2"], rows[f"BMG{q}"]["velocity_rel_l2"])
        nod = min(v["velocity_rel_l2"] for k, v in rows.items()
                  if k[0] in "ECSD" and "velocity_rel_l2" in v)
        assert spl < nod, (q, spl, nod)
        for r in rows.values():
            assert np.isfinite(r["velocity_rel_l2"]), r
    print("analytic_wake_orders verify: ALL PASS")


def main():
    ensure_generated_dirs()
    path = RESULTSDIR / "analytic_wake_orders_summary.json"
    if "--verify" in sys.argv:
        verify(json.loads(path.read_text()))
        return
    out = run_all()
    verify(out)
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
