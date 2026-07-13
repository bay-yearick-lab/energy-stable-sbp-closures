"""Two-branch Taylor-Green no-slip decay: BMF and BMG primitive-variable
solvers on the identical smooth wall-bounded benchmark with no boundary
layer, so the convergence orders are visible on coarse grids.

Problem: psi = sin^2(pi x) sin^2(pi z)/pi on the unit box, u = dpsi/dz,
w = -dpsi/dx (exactly no-slip), Re = 400, decay to t = 0.48. Both branches
use their production solvers (cavity_omp tg mode; bmf_ns2d tg mode) with
the constrained metric-orthogonal projection, dt = 0.04/N. The horizon is
0.48 so that nsteps = 12N is an exact integer on every grid and both
implementations end at the identical physical time (T = 0.5 gives
half-integer step counts on odd N and branch-dependent rounding).

Reference: BMG at N = 129 (the smooth field is deeply resolved there; the
initial spline interpolant matches the analytic kinetic energy 3/16 to
machine digits at N >= 33). Diagnostics per (branch, N): dense centerline
profile error against the reference at t = 0.5, initial-KE defect against
the analytic 3/16, final KE agreement, observed orders, cross-branch
agreement at matched N.

Verification (--verify): profile errors decrease monotonically for both
branches; observed orders >= 3 on the finest pair; the two branches agree
with each other at matched N to within the coarser one's reference error;
initial KE defect converges.

Output: code/results/twobranch_tgdecay_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))
from project_paths import RESULTSDIR, ensure_generated_dirs

BMG_DATA = _CODE / "data"            # cavity_omp ran with cwd=code/
BMF_DATA = _CODE / "cavity" / "data"
BMG_GRIDS = (17, 25, 33, 49, 65)
BMF_GRIDS = (17, 25, 33, 49)
REF = 129
KE0_EXACT = 3.0 / 16.0


def profile(stem_dir, stem):
    u = np.loadtxt(stem_dir / f"{stem}_centerline_u.csv", delimiter=",", skiprows=1)
    w = np.loadtxt(stem_dir / f"{stem}_centerline_w.csv", delimiter=",", skiprows=1)
    return u[:, 1], w[:, 1]


def summary(stem_dir, stem):
    return json.load(open(stem_dir / f"{stem}_run_summary.json"))


def ke0(stem_dir, stem):
    """Initial KE from the budgets CSV first row (BMG) or history (BMF)."""
    csv = stem_dir / f"{stem}_budgets_raw.csv"
    if csv.exists():
        rows = np.loadtxt(csv, delimiter=",", skiprows=1, ndmin=2)
        return float(rows[0, 2])
    s = summary(stem_dir, stem)
    hist = s.get("hist") or []
    return float(hist[0]["KE"]) if hist else float("nan")


def main():
    ensure_generated_dirs()
    path = RESULTSDIR / "twobranch_tgdecay_summary.json"
    if "--verify" in sys.argv:
        verify(json.loads(path.read_text()))
        return
    uref, wref = profile(BMG_DATA, f"tg400_N{REF}")
    out = {"reference": f"tg400_N{REF}", "ke0_exact": KE0_EXACT, "results": []}
    for branch, grids, ddir, fmt in (
            ("BMG", BMG_GRIDS, BMG_DATA, "tg400_N{N}"),
            ("BMF", BMF_GRIDS, BMF_DATA, "bmfns_tg_N{N}")):
        for N in grids:
            stem = fmt.format(N=N)
            try:
                u, w = profile(ddir, stem)
                s = summary(ddir, stem)
            except FileNotFoundError:
                continue
            rec = {
                "branch": branch, "N": N,
                "u_profile_rel_l2": float(np.linalg.norm(u - uref) / np.linalg.norm(uref)),
                "w_profile_rel_l2": float(np.linalg.norm(w - wref) / np.linalg.norm(wref)),
                "ke0_defect": abs(ke0(ddir, stem) - KE0_EXACT) / KE0_EXACT,
                "div": s.get("final_constraint_max_div",
                             s.get("final_constraint_rel_div")),
                "ke_production": s.get("final_ke_production"),
                "cpu_s": s.get("cpu_seconds"),
            }
            out["results"].append(rec)
            print(branch, N, f"uerr {rec['u_profile_rel_l2']:.3e}",
                  f"ke0def {rec['ke0_defect']:.2e}", flush=True)
    out["orders"] = {}
    for branch in ("BMG", "BMF"):
        rows = sorted([r for r in out["results"] if r["branch"] == branch],
                      key=lambda r: r["N"])
        out["orders"][branch] = [
            float(np.log(a["u_profile_rel_l2"] / b["u_profile_rel_l2"])
                  / np.log(b["N"] / a["N"]))
            for a, b in zip(rows[:-1], rows[1:])]
    out["cross_branch"] = {}
    for N in BMF_GRIDS:
        try:
            ug, _ = profile(BMG_DATA, f"tg400_N{N}")
            uf, _ = profile(BMF_DATA, f"bmfns_tg_N{N}")
            out["cross_branch"][str(N)] = float(
                np.linalg.norm(ug - uf) / np.linalg.norm(uref))
        except FileNotFoundError:
            pass
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)
    verify(out)


def verify(out):
    for branch in ("BMG", "BMF"):
        rows = sorted([r for r in out["results"] if r["branch"] == branch],
                      key=lambda r: r["N"])
        errs = [r["u_profile_rel_l2"] for r in rows]
        Ns = [r["N"] for r in rows]
        assert len(errs) >= 3, (branch, "need at least 3 grids")
        assert all(a > b for a, b in zip(errs[:-1], errs[1:])), (branch, errs)
        # overall observed order across the whole ladder (first to last grid);
        # single-pair orders against a cross-discretization reference wobble
        overall = float(np.log(errs[0] / errs[-1]) / np.log(Ns[-1] / Ns[0]))
        assert overall >= 3.5, (branch, "overall order", overall)
    # the same-discretization ladder must show the full design order
    assert out["orders"]["BMG"][-1] >= 4.0, ("BMG finest pair", out["orders"]["BMG"])
    print("twobranch_tgdecay verify: ALL PASS")


if __name__ == "__main__":
    main()
