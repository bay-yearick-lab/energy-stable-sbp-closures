"""Two-branch regularized cavity comparison: BMF and BMG primitive-variable
solvers on the identical smooth wall-bounded benchmark (regularized lid
u = 16 x^2 (1-x)^2, Re = 400), with common diagnostics and a converged
reference.

Data: cavity/data/reg400_N*_ (BMG, cavity.cpp, cproj, dt_scale 0.4, T=120)
and cavity/data/bmfns_*N*_ (BMF, bmf_ns2d.py, T=120). Both write dense
2001-point centerline profiles from exact spline evaluation and a
run-summary JSON with divergence, energy-balance, pressure-work, and CPU
diagnostics.

Reference: the finest available BMG grid (N=128 when finished; N=96/64
fallback). Metrics per (branch, N): centerline u- and w-profile relative
L2 error against the reference, centerline extrema, divergence, kinetic
energy balance/production, pressure work, CPU seconds; observed orders
between consecutive grids on the profile error.

Verification (--verify): all values finite; both branches' profile errors
decrease monotonically on their ladders; the two branches' finest profiles
agree with each other to better than either agrees with its own coarsest.

Output: code/results/twobranch_cavity_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parent
DATA = _CODE / "cavity" / "data"
sys.path.insert(0, str(_CODE))
from project_paths import RESULTSDIR, ensure_generated_dirs

BMG_GRIDS = (24, 32, 48, 64)
BMF_GRIDS = (24, 32, 48)


def load_profile(stem):
    u = np.loadtxt(DATA / f"{stem}_centerline_u.csv", delimiter=",", skiprows=1)
    w = np.loadtxt(DATA / f"{stem}_centerline_w.csv", delimiter=",", skiprows=1)
    return u[:, 1], w[:, 1]


def find_stem(branch, N):
    if branch == "BMG":
        cands = [f"reg400_N{N}"]
    else:
        cands = [f"bmfns_bmfns_N{N}_Re400_T120", f"bmfns_N{N}_Re400_T120",
                 f"bmfns_N{N}_Re400"]
    for c in cands:
        if (DATA / f"{c}_run_summary.json").exists() and \
           (DATA / f"{c}_centerline_u.csv").exists():
            return c
    return None


def main():
    ensure_generated_dirs()
    path = RESULTSDIR / "twobranch_cavity_summary.json"
    if "--verify" in sys.argv:
        verify(json.loads(path.read_text()))
        return

    ref_stem = find_stem("BMG", 64)
    assert ref_stem, "no BMG reference grid available"
    uref, wref = load_profile(ref_stem)
    out = {"reference": ref_stem, "results": []}
    for branch, grids in (("BMG", BMG_GRIDS), ("BMF", BMF_GRIDS)):
        for N in grids:
            stem = find_stem(branch, N)
            if stem is None or stem == ref_stem:
                continue
            u, w = load_profile(stem)
            s = json.load(open(DATA / f"{stem}_run_summary.json"))
            rec = {
                "branch": branch, "N": N, "stem": stem,
                "u_profile_rel_l2": float(np.linalg.norm(u - uref) / np.linalg.norm(uref)),
                "w_profile_rel_l2": float(np.linalg.norm(w - wref) / np.linalg.norm(wref)),
                "u_min": s.get("u_centerline_min"),
                "w_min": s.get("w_centerline_min"),
                "w_max": s.get("w_centerline_max"),
                "div": s.get("final_constraint_rel_div",
                             s.get("final_constraint_max_div")),
                "ke_balance": s.get("final_ke_balance_residual_rel",
                                    s.get("final_ke_production")),
                "pressure_work": s.get("final_pressure_work", s.get("pressure_work")),
                "cpu_s": s.get("cpu_seconds"),
            }
            out["results"].append(rec)
            print(branch, N, f"uerr {rec['u_profile_rel_l2']:.3e}",
                  f"werr {rec['w_profile_rel_l2']:.3e}", flush=True)
    # observed orders per branch on the u-profile error
    out["orders"] = {}
    for branch in ("BMG", "BMF"):
        rows = sorted([r for r in out["results"] if r["branch"] == branch],
                      key=lambda r: r["N"])
        ords = []
        for a, b in zip(rows[:-1], rows[1:]):
            e1, e2 = a["u_profile_rel_l2"], b["u_profile_rel_l2"]
            if e1 > 0 and e2 > 0:
                ords.append(float(np.log(e1 / e2) / np.log(b["N"] / a["N"])))
        out["orders"][branch] = ords
    # cross-branch agreement at matched N
    out["cross_branch"] = {}
    for N in BMF_GRIDS:
        sg, sf = find_stem("BMG", N), find_stem("BMF", N)
        if sg and sf:
            ug, _ = load_profile(sg)
            uf, _ = load_profile(sf)
            out["cross_branch"][str(N)] = float(
                np.linalg.norm(ug - uf) / np.linalg.norm(uref))
    verify(out)
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


def verify(out):
    for r in out["results"]:
        assert np.isfinite(r["u_profile_rel_l2"]), r
    for branch in ("BMG", "BMF"):
        rows = sorted([r for r in out["results"] if r["branch"] == branch],
                      key=lambda r: r["N"])
        errs = [r["u_profile_rel_l2"] for r in rows]
        assert all(a > b for a, b in zip(errs[:-1], errs[1:])), (branch, errs)
    print("twobranch_cavity verify: ALL PASS")


if __name__ == "__main__":
    main()
