"""Common-postprocessor (Tier A) analytic-wake diagnostics.

Divergence and momentum residuals measured through
each family's OWN derivative reward internal self-consistency circularly, so
they cannot be the headline. This certificate recomputes both on one COMMON
overresolved evaluator, independent of every fielded family:

  Tier A (headline): every family's converged analytic-wake fields are moved to
      one Chebyshev-Lobatto diagnostic grid by the family's own best native
      reconstruction (spline evaluation for the spline pair, the local
      Lagrange interpolant the paper's comparisons already use for nodal
      families), and divergence and momentum residuals are then formed with
      the SINGLE dense Chebyshev collocation derivative. No family in the
      comparison is Chebyshev, so no family grades itself.
  Tier B (self-consistency): the same functionals through each family's own
      operators, the manuscript's previous (native) numbers.

Two gates make Tier A falsifiable:
  G1  the Chebyshev evaluator applied to the EXACT fields sampled directly
      on the diagnostic grid returns divergence and momentum residuals at
      roundoff (the evaluator adds no defect of its own);
  G2  the exact fields sampled on each family's 25-point grid and moved to
      the diagnostic grid through that family's reconstruction set the
      family's interpolation floor; a Tier-A defect is only meaningful
      above that floor, and both are recorded per family.

Usage:
  python code/s6_common_diagnostics.py --write code/results/common_diagnostics.json
  python code/s6_common_diagnostics.py --verify code/results/common_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import boundary_spline_analysis as ba  # noqa: E402
from cheb_operator import build_cheb_operator  # noqa: E402
from project_paths import RESULTSDIR  # noqa: E402
from s6_analytic_wake import (  # noqa: E402
    LENGTH,
    WakeConfig,
    advance_solution,
    exact_fields,
    interp2,
)

SCHEMES = ("E6", "C6", "SBP6", "DRP6", "BMF6", "BMG6")
REYNOLDS = (20, 40, 80)
NC = 97          # Chebyshev-Lobatto diagnostic nodes per direction


def cheb_grid_ops():
    op = build_cheb_operator(NC, LENGTH)
    xc = np.asarray(op.x)
    Dc = np.asarray(op.matrix)
    # Clenshaw-Curtis-style weights via exact integration of the Lagrange
    # cardinal functions is overkill here; the trapezoid on the clustered
    # Lobatto nodes is enough for RELATIVE norms shared by every family.
    w = ba.nodal_trapezoidal_weights(xc)
    W2 = np.outer(w, w)
    return xc, Dc, W2


def common_functionals(u, v, p, Dc, W2, exact, nu):
    """Divergence defect and momentum residual with the common derivative."""
    ux = Dc @ u
    vy = v @ Dc.T
    div = ux + vy
    div_defect = float(np.max(np.abs(div[1:-1, 1:-1]))
                       / max(np.max(np.abs(exact["u"])), 1e-30))
    uy = u @ Dc.T
    vx = Dc @ v
    px = Dc @ p
    py = p @ Dc.T
    lap_u = Dc @ (Dc @ u) + (u @ Dc.T) @ Dc.T
    lap_v = Dc @ (Dc @ v) + (v @ Dc.T) @ Dc.T
    ru = u * ux + v * uy + px - nu * lap_u
    rv = u * vx + v * vy + py - nu * lap_v
    num = np.sqrt(ru[1:-1, 1:-1] ** 2 + rv[1:-1, 1:-1] ** 2)
    ref = np.sqrt((u * ux + v * uy)[1:-1, 1:-1] ** 2
                  + (u * vx + v * vy)[1:-1, 1:-1] ** 2)
    mom = float(np.sqrt(np.mean(num ** 2))
                / max(np.sqrt(np.mean(ref ** 2)), 1e-30))
    return div_defect, mom


def native_functionals(state, reynolds):
    """Tier B: the manuscript's previous native-operator functionals."""
    d1 = state["d1"]
    u, v = state["u"], state["v"]
    x = state["x"]
    exact = exact_fields(x, x, reynolds)
    div = d1 @ u + v @ d1.T
    div_defect = float(np.max(np.abs(div[1:-1, 1:-1]))
                       / max(np.max(np.abs(exact["u"])), 1e-30))
    from s6_analytic_wake import momentum_residual
    mom = momentum_residual(d1, state["d2"], u, v, exact["p"], reynolds)
    return div_defect, mom


def run(cfg: WakeConfig):
    xc, Dc, W2 = cheb_grid_ops()
    out = {"config": {"schemes": list(SCHEMES), "reynolds": list(REYNOLDS),
                      "npts": cfg.npts, "cheb_diag_npts": NC},
           "gates": {}, "cases": []}

    # G1: evaluator cleanliness on exact fields sampled on the cheb grid
    g1 = {}
    for re_ in REYNOLDS:
        ex = exact_fields(xc, xc, re_)
        dd, mm = common_functionals(ex["u"], ex["v"], ex["p"], Dc, W2, ex,
                                    1.0 / re_)
        g1[str(re_)] = {"div": dd, "mom": mm}
    out["gates"]["G1_exact_on_cheb"] = g1

    for label in SCHEMES:
        for re_ in REYNOLDS:
            t0 = time.time()
            state = advance_solution(label, re_, cfg)
            op = state["op"]
            ex_c = exact_fields(xc, xc, re_)
            u_c = interp2(op, state["u"], xc)
            v_c = interp2(op, state["v"], xc)
            ddA, momA = common_functionals(u_c, v_c, ex_c["p"], Dc, W2,
                                           ex_c, 1.0 / re_)
            ddB, momB = native_functionals(state, re_)
            # G2: this family's interpolation floor (exact fields sampled on
            # its own grid, moved to the diagnostic grid the same way)
            ex_n = exact_fields(state["x"], state["x"], re_)
            uf = interp2(op, ex_n["u"], xc)
            vf = interp2(op, ex_n["v"], xc)
            ddF, momF = common_functionals(uf, vf, ex_c["p"], Dc, W2,
                                           ex_c, 1.0 / re_)
            err2 = (u_c - ex_c["u"]) ** 2 + (v_c - ex_c["v"]) ** 2
            ref2 = ex_c["u"] ** 2 + ex_c["v"] ** 2
            vel = float(np.sqrt(np.sum(W2 * err2))
                        / max(np.sqrt(np.sum(W2 * ref2)), 1e-30))
            rec = {"label": label, "re": re_,
                   "tierA_div_defect": ddA, "tierA_mom_residual": momA,
                   "tierB_native_div_defect": ddB,
                   "tierB_native_mom_residual": momB,
                   "floor_div": ddF, "floor_mom": momF,
                   "common_vel_rel_l2": vel,
                   "steps": state["steps"],
                   "elapsed_s": round(time.time() - t0, 2)}
            out["cases"].append(rec)
            print(f"{label:5s} Re={re_:3d} A(div={ddA:.3e} mom={momA:.3e}) "
                  f"B(div={ddB:.3e} mom={momB:.3e}) floor(div={ddF:.3e}) "
                  f"vel={vel:.3e}", flush=True)

    # ranking survival check per Re on the Tier-A functionals
    ranks = {}
    for re_ in REYNOLDS:
        sub = [c for c in out["cases"] if c["re"] == re_]
        ranks[str(re_)] = {
            "tierA_div": sorted(sub, key=lambda c: c["tierA_div_defect"])[0]["label"],
            "tierA_mom": sorted(sub, key=lambda c: c["tierA_mom_residual"])[0]["label"],
            "tierB_div": sorted(sub, key=lambda c: c["tierB_native_div_defect"])[0]["label"],
            "common_vel": sorted(sub, key=lambda c: c["common_vel_rel_l2"])[0]["label"],
        }
    out["winners"] = ranks
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", metavar="JSON", default=None)
    ap.add_argument("--verify", metavar="JSON", default=None)
    args = ap.parse_args()
    if args.verify:
        d = json.loads(Path(args.verify).read_text())
        for re_, g in d["gates"]["G1_exact_on_cheb"].items():
            assert g["div"] < 1e-9 and g["mom"] < 1e-8, (re_, g)
        assert len(d["cases"]) == len(SCHEMES) * len(REYNOLDS)
        print("verify OK:", json.dumps(d["winners"], indent=1))
        return
    cfg = WakeConfig()
    out = run(cfg)
    dest = Path(args.write) if args.write else Path(RESULTSDIR) / "common_diagnostics.json"
    dest.write_text(json.dumps(out, indent=1))
    print("winners:", json.dumps(out["winners"], indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
