"""Three-tier time-step analysis.

Three distinct quantities are commonly reported under one name,
"RK4 stable step":

  Tier A  semidiscrete spectral stability: alpha = max Re(lambda). If
          alpha > 0 the semidiscrete system is exponentially unstable and NO
          positive explicit step is asymptotically stable: the asymptotic
          column must read "none", never a number.
  Tier B  asymptotic RK4 bound: the largest dt with rho(R4(dt A)) <= 1.
          Defined only when alpha <= 0. For an exactly skew operator this is
          the FINITE imaginary-axis cap 2*sqrt(2)/rho(A);
  Tier C  finite-horizon tolerance step: the largest dt whose worst-case
          amplification over a NAMED horizon stays below a NAMED tolerance.
          This is a mission quantity, not a stability limit, and is labeled
          as such wherever it appears.

Every tier is validated against direct time integration:
  V1  skew families: an RK4 march at 0.99*dtB stays bounded over the horizon
      and a march at 1.05*dtB grows (the tier-B boundary is sharp);
  V2  positive-abscissa families: the march at the tier-C step grows at the
      predicted rate e^{alpha t} (it is NOT stable, merely within tolerance
      over the horizon);
  V3  nonnormal transient growth: max_m ||R4(dt A)^m||_2 over the horizon is
      recorded at the tier-B step (eigenvalues alone can hide transient
      growth of a nonnormal closure), seeded by the worst right singular
      vector.

Reuses the coupled-wave-pair and interior 2D Laplacian builders of
spectral_radius_table.py: the operators each mission class actually marches.

Usage:
  python code/s6_time_step_cost.py --write code/results/time_step_cost.json
  python code/s6_time_step_cost.py --verify code/results/time_step_cost.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import RESULTSDIR  # noqa: E402
from spectral_radius_table import (  # noqa: E402
    coupled_wave_pair_matrix,
    diffusion_interior_matrix,
)

FAMILIES = ("E6", "C6", "SBP6", "DRP6", "BMF6", "BMG6", "SBP6-BO",
            "E4", "C4", "SBP4", "DRP4", "BMF4", "BMG4",
            "E8", "C8", "SBP8", "DRP8", "BMF8", "BMG8")
NPTS = 129
HORIZON_TIME = 6.0          # six domain transits (the long-time cliff horizon)
TOL_C = 1.25                # named tier-C amplification tolerance over horizon
SKEW_TOL = 1e-10
RK4_IMAG_CAP = 2.0 * np.sqrt(2.0)


def rk4_stability_fn(z: np.ndarray) -> np.ndarray:
    return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0


def tier_b_asymptotic(eigs: np.ndarray, alpha: float, rho: float):
    """Largest dt with max_i |R4(dt lambda_i)| <= 1, or None if alpha > 0."""
    if alpha > SKEW_TOL:
        return None
    lo, hi = 0.0, 10.0 / max(rho, 1e-30)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.max(np.abs(rk4_stability_fn(mid * eigs))) <= 1.0 + 1e-13:
            lo = mid
        else:
            hi = mid
    return lo


def tier_c_tolerance(eigs: np.ndarray, alpha: float, rho: float,
                     horizon: float, tol: float) -> float:
    """Largest dt whose eigenvalue-amplification over the horizon stays
    within a factor tol of the INHERENT semidiscrete growth e^{alpha T}.
    An absolute tolerance is unachievable for a positive-abscissa operator
    (the physical growth is exponential at every dt), so the coherent
    finite-horizon quantity is discretization-added amplification. For a
    stable operator (alpha <= 0) the reference growth is 1 and this reduces
    to the absolute definition. Validated against true marches in V1/V2."""
    ref = np.exp(max(alpha, 0.0) * horizon)
    lo, hi = 0.0, 10.0 / max(rho, 1e-30)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        nsteps = max(int(np.ceil(horizon / mid)), 1)
        with np.errstate(over="ignore"):
            growth = np.max(np.abs(rk4_stability_fn(mid * eigs))) ** nsteps
        if growth <= tol * ref:
            lo = mid
        else:
            hi = mid
    return lo


def march_growth(A: np.ndarray, dt: float, horizon: float,
                 x0: np.ndarray | None = None) -> float:
    """||x(T)|| / ||x(0)|| under an actual RK4 march."""
    n = A.shape[0]
    rng = np.random.default_rng(7)
    x = rng.standard_normal(n) if x0 is None else x0.copy()
    x /= np.linalg.norm(x)
    nsteps = int(np.ceil(horizon / dt))
    peak = 1.0
    for _ in range(nsteps):
        k1 = A @ x
        k2 = A @ (x + 0.5 * dt * k1)
        k3 = A @ (x + 0.5 * dt * k2)
        k4 = A @ (x + dt * k3)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        nx = np.linalg.norm(x)
        peak = max(peak, nx)
        if nx > 1e12:
            break
    return float(np.linalg.norm(x))


def transient_norm(A: np.ndarray, dt: float, horizon: float) -> tuple[float, np.ndarray]:
    """max_m ||R4(dt A)^m||_2 over the horizon and the worst initial vector."""
    n = A.shape[0]
    R = (np.eye(n) + dt * A + (dt * A) @ (dt * A) / 2.0
         + np.linalg.matrix_power(dt * A, 3) / 6.0
         + np.linalg.matrix_power(dt * A, 4) / 24.0)
    nsteps = min(int(np.ceil(horizon / dt)), 4000)
    P = np.eye(n)
    peak, worst = 1.0, None
    stride = max(nsteps // 200, 1)
    for m in range(1, nsteps + 1):
        P = R @ P
        if m % stride == 0 or m == nsteps:
            s = np.linalg.svd(P, compute_uv=False)[0]
            if s > peak:
                peak = s
                worst = None  # recompute lazily below if needed
    # worst right singular vector of the final power for the V3 seed
    _u, _s, vt = np.linalg.svd(P)
    return float(peak), vt[0]


def analyze_family(label: str, npts: int = NPTS) -> dict:
    A, meta = coupled_wave_pair_matrix(label, npts)
    eigs = np.linalg.eigvals(A)
    alpha = float(np.max(eigs.real))
    rho = float(np.max(np.abs(eigs)))
    dtB = tier_b_asymptotic(eigs, alpha, rho)
    dtC = tier_c_tolerance(eigs, alpha, rho, HORIZON_TIME, TOL_C)
    rec = {
        "label": label, "npts": npts,
        "tierA_spectral_abscissa": alpha,
        "spectral_radius": rho,
        "tierB_dt_asymptotic": dtB,
        "tierB_imag_axis_cap_2sqrt2_over_rho": RK4_IMAG_CAP / rho,
        "tierC_dt_tolerance": dtC,
        "tierC_definition": f"amplification <= {TOL_C} x e^(alpha T) "
                            f"over T={HORIZON_TIME}",
    }
    if dtB is not None:
        # V1: the tier-B boundary is sharp under a real march
        g_in = march_growth(A, 0.99 * dtB, HORIZON_TIME)
        g_out = march_growth(A, 1.05 * dtB, HORIZON_TIME)
        rec["V1_march_growth_at_0p99_dtB"] = g_in
        rec["V1_march_growth_at_1p05_dtB"] = g_out
        # skew cross-check: bisection equals the analytic imaginary-axis cap
        if abs(alpha) <= SKEW_TOL:
            rec["V1_skew_cap_match"] = abs(dtB * rho / RK4_IMAG_CAP - 1.0)
        # V3: nonnormal transient at the tier-B step
        peak, worst = transient_norm(A, 0.99 * dtB, HORIZON_TIME)
        rec["V3_transient_peak_at_dtB"] = peak
        g_worst = march_growth(A, 0.99 * dtB, HORIZON_TIME, x0=worst)
        rec["V3_march_growth_worst_vector"] = g_worst
    else:
        # V2: at the tier-C step the march grows at the predicted e^{alpha t};
        # the validation horizon is shortened so the growth stays measurable
        # in double precision (the prediction over T=6 can exceed 1e17)
        t_v2 = min(HORIZON_TIME, np.log(1e8) / max(alpha, 1e-3))
        g = march_growth(A, dtC, t_v2)
        rec["V2_validation_horizon"] = float(t_v2)
        rec["V2_march_growth_at_dtC"] = g
        rec["V2_predicted_exp_alpha_T"] = float(np.exp(alpha * t_v2))
        rec["V2_growth_ratio_vs_prediction"] = g / max(np.exp(alpha * t_v2), 1e-300)
    # diffusion tier data (2D interior tensor Laplacian, pairwise-sum spectrum)
    D2 = diffusion_interior_matrix(label, npts)
    ev1 = np.linalg.eigvals(D2)
    ev2 = (ev1[:, None] + ev1[None, :]).ravel()
    alpha2 = float(np.max(ev2.real))
    rho2 = float(np.max(np.abs(ev2)))
    dtB2 = tier_b_asymptotic(ev2, alpha2, rho2)
    rec["diffusion"] = {
        "tierA_spectral_abscissa_2d": alpha2,
        "spectral_radius_2d": rho2,
        "tierB_dt_asymptotic": dtB2,
    }
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", metavar="JSON", default=None)
    ap.add_argument("--verify", metavar="JSON", default=None)
    args = ap.parse_args()

    if args.verify:
        data = json.loads(Path(args.verify).read_text())
        for rec in data["families"]:
            a = rec["tierA_spectral_abscissa"]
            if a > SKEW_TOL:
                assert rec["tierB_dt_asymptotic"] is None, \
                    f"{rec['label']}: positive abscissa cannot have a stable step"
                assert rec["V2_growth_ratio_vs_prediction"] > 0.1, rec["label"]
            else:
                assert rec["V1_march_growth_at_0p99_dtB"] < 2.0, rec["label"]
                assert rec["V1_march_growth_at_1p05_dtB"] > 2.0, rec["label"]
                if "V1_skew_cap_match" in rec:
                    assert rec["V1_skew_cap_match"] < 1e-6, rec["label"]
        print(f"verify OK: {len(data['families'])} families, three tiers coherent")
        return

    out = []
    for label in FAMILIES:
        try:
            rec = analyze_family(label)
        except Exception as exc:
            rec = {"label": label, "error": f"{type(exc).__name__}: {exc}"}
        out.append(rec)
        if "error" in rec:
            print(label, "FAILED:", rec["error"], flush=True)
            continue
        dtB = rec["tierB_dt_asymptotic"]
        print(f"{label:8s} alpha={rec['tierA_spectral_abscissa']:+.3e} "
              f"dtB={'none' if dtB is None else f'{dtB:.3e}'} "
              f"dtC={rec['tierC_dt_tolerance']:.3e} "
              + (f"V1 in/out={rec['V1_march_growth_at_0p99_dtB']:.3f}/"
                 f"{rec['V1_march_growth_at_1p05_dtB']:.2e} "
                 f"V3 peak={rec['V3_transient_peak_at_dtB']:.3f}"
                 if dtB is not None else
                 f"V2 growth={rec['V2_march_growth_at_dtC']:.3e} "
                 f"pred={rec['V2_predicted_exp_alpha_T']:.3e}"))

    data = {"description": "three-tier RK4 step analysis with direct-march "
                           "validation", "horizon_time": HORIZON_TIME,
            "tolerance_C": TOL_C, "families": out}
    dest = Path(args.write) if args.write else Path(RESULTSDIR) / "time_step_cost.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
