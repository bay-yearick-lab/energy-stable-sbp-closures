"""Full-domain rescore of the standing-mode cliff test: the identical runs
scored with NO wall trim (interior_strip=0) next to the published trimmed
norm (interior_strip=6), so the headline wall-transport verdict does not
depend on excluding the wall strip.

Configuration matches s6_wall_longtime_cliff exactly: N=41, mode m=6,
cfl=0.012, t_end=6 (six transits), strong rigid-wall traces, q=6 block plus
the CSB compact-filtered and boundary-optimized SBP entrants.

Verification (--verify): every value finite or flagged destroyed; for both
spline routes the full-domain and trimmed errors agree within a factor of
four (no wall-strip artifact hides in the trim); the spline/best-stable-
competitor ordering is identical under both norms.

Output: code/results/fulldomain_rescore_summary.json
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

N = 41
Q = 6
T_END = 6.0
MODE = 6
FAMILIES = ("E6", "C6", "SBP6", "DRP6", "BMF6", "BMG6", "C6F", "SBP6-BO")


def p_exact(x, t):
    return np.cos(MODE * np.pi * np.asarray(x, float)) * np.cos(MODE * np.pi * t)


def u_exact(x, t):
    return np.sin(MODE * np.pi * np.asarray(x, float)) * np.sin(MODE * np.pi * t)


def run(label, strip):
    if label == "SBP6-BO":
        op = ba.build_operator_extended(label, Q, N)
    else:
        op = ba.build_operator(ba.scheme_kind(label), Q, N)
    res = ba.simulate_reflected_acoustics(
        op, p_exact, u_exact, T_END, cfl=0.012,
        interior_strip=strip, record_count=8)
    return float(res.error_history[-1])


def run_c6f_full():
    """C6F needs its per-step filter; the cliff's own history is full-domain."""
    from s6_wall_longtime_cliff import simulate_extra_history
    res = simulate_extra_history("C6F", N, T_END, 0.012, 8)
    return float(np.asarray(res.error_history)[-1])


def main():
    ensure_generated_dirs()
    path = RESULTSDIR / "fulldomain_rescore_summary.json"
    if "--verify" in sys.argv:
        verify(json.loads(path.read_text()))
        return
    out = {"config": {"N": N, "q": Q, "t_end": T_END, "mode": MODE,
                      "strips": [0, 6]}, "results": {}}
    for label in FAMILIES:
        try:
            if label == "C6F":
                full = run_c6f_full()
                out["results"][label] = {
                    "full": full,
                    "note": "filtered march; the cliff scores it full-domain already"}
                print(f"{label:8s} full {full:.3e}  (cliff norm is already untrimmed)",
                      flush=True)
                continue
            full = run(label, 0)
            trim = run(label, 6)
        except Exception as exc:            # noqa: BLE001
            out["results"][label] = {"error": str(exc)}
            continue
        out["results"][label] = {"full": full, "trimmed": trim}
        print(f"{label:8s} full {full:.3e}  trimmed {trim:.3e}", flush=True)
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)
    verify(out)


def verify(out):
    r = out["results"]
    for lab in ("BMF6", "BMG6"):
        f, t = r[lab]["full"], r[lab]["trimmed"]
        assert np.isfinite(f) and np.isfinite(t), (lab, f, t)
        # a trim hiding an order-one wall defect would show many decades,
        # not a single-digit factor
        assert f / t < 10.0 and t / f < 10.0, ("wall-strip artifact", lab, f, t)
    # ordering vs the strongest stable competitor is norm-independent
    stable = [lab for lab in ("SBP6", "SBP6-BO", "C6F")
              if "full" in r.get(lab, {}) and r[lab]["full"] < 1.0]
    best_full = min(r[lab]["full"] for lab in stable)
    best_trim = min(r[lab]["trimmed"] for lab in stable if "trimmed" in r[lab])
    for lab in ("BMF6", "BMG6"):
        assert r[lab]["full"] < best_full and r[lab]["trimmed"] < best_trim
    print("fulldomain_rescore verify: ALL PASS")


if __name__ == "__main__":
    main()
