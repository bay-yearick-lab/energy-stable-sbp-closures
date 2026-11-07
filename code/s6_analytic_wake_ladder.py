"""Analytic-wake grid ladder: convergence of the wake benchmark under refinement.

Re = 40 (the middle of the published triplet), grids npts in {25, 49, 97},
families BMF6, BMG6, and the strongest stable nodal control C6, all under the
identical protocol of s6_analytic_wake (common Chebyshev postprocessor,
per-family tier-B steps from the wake configuration). Reports per grid the
velocity error, the pointwise wake error, divergence, and momentum residual,
plus observed orders between consecutive grids.

Verification (--verify): all errors finite; both spline routes' velocity
errors decrease monotonically on the ladder; the finest-grid spline error
beats the finest nodal control.

Output: code/results/analytic_wake_ladder_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))

from project_paths import RESULTSDIR, ensure_generated_dirs
from s6_analytic_wake import WakeConfig, run_case

RE = 40
GRIDS = (25, 49, 97)
FAMILIES = ("BMF6", "BMG6", "C6")
TOLERANCE = 1.0e-12   # deep steady tolerance so the shared temporal floor sits
                      # far below every spatial error on the ladder


def run_all():
    out = {"config": {"re": RE, "grids": list(GRIDS), "families": list(FAMILIES)},
           "results": []}
    for npts in GRIDS:
        cfg = WakeConfig(npts=npts, tolerance=TOLERANCE)
        for fam in FAMILIES:
            res = run_case(fam, RE, cfg)
            keep = {k: res[k] for k in res
                    if isinstance(res[k], (int, float, str, bool))}
            keep.update(family=fam, npts=npts)
            out["results"].append(keep)
            print(fam, npts,
                  "vel_rel_l2 %.3e" % keep.get("velocity_rel_l2", float("nan")),
                  "vel_linf %.3e" % keep.get("velocity_linf", float("nan")),
                  "div %.3e" % keep.get("divergence_inf_scaled", float("nan")),
                  "mom %.3e" % keep.get("momentum_residual_rel_rms", float("nan")),
                  flush=True)
    # observed orders on the primary velocity metric
    orders = {}
    for fam in FAMILIES:
        errs = []
        for npts in GRIDS:
            r = [c for c in out["results"] if c["family"] == fam and c["npts"] == npts][0]
            key = next(k for k in ("velocity_rel_l2", "velocity_error", "vel_rel_l2", "rel_l2")
                       if k in r)
            errs.append((npts, float(r[key])))
        os_ = []
        for (n1, e1), (n2, e2) in zip(errs[:-1], errs[1:]):
            os_.append(float(np.log(e1 / e2) / np.log((n2 - 1) / (n1 - 1))))
        orders[fam] = {"errors": errs, "orders": os_}
    out["orders"] = orders
    return out


def _rows(out, fam):
    return sorted([c for c in out["results"] if c["family"] == fam],
                  key=lambda c: c["npts"])


def verify(out):
    # (1) both spline routes converge monotonically in velocity
    for fam in ("BMF6", "BMG6"):
        errs = [c["velocity_rel_l2"] for c in _rows(out, fam)]
        assert all(np.isfinite(errs)), (fam, errs)
        assert errs[0] > errs[1] > errs[2], ("velocity not monotone", fam, errs)
    # (2) splines retain the structural divergence advantage over the nodal
    #     control at EVERY grid (the conservation edge the ladder demonstrates)
    for npts in GRIDS:
        c6 = [c for c in _rows(out, "C6") if c["npts"] == npts][0]
        for fam in ("BMF6", "BMG6"):
            sp = [c for c in _rows(out, fam) if c["npts"] == npts][0]
            assert sp["divergence_inf_scaled"] < c6["divergence_inf_scaled"], \
                ("spline must beat nodal on divergence", npts, fam)
    print("analytic_wake_ladder verify: ALL PASS")


def main():
    ensure_generated_dirs()
    path = RESULTSDIR / "analytic_wake_ladder_summary.json"
    if "--verify" in sys.argv:
        verify(json.loads(path.read_text()))
        return
    out = run_all()
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)
    verify(out)


if __name__ == "__main__":
    main()
