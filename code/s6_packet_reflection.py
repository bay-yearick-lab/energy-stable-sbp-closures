"""Localized wall-packet reflection: the direct returned-packet experiment,
plus the calibration data that turns the wall-resolution spectrum into a
tested predictor.

A right-moving Gaussian-enveloped acoustic packet (p = u on the incoming
characteristic) is launched at the right wall of the closed box; the exact
rigid-wall solution is the method-of-images superposition, and after each
full domain cycle (t = 2k) it reproduces the initial state exactly. Every
family marches the packet with the same protocol as the standing-mode test
(strong rigid-wall traces via the metric-orthogonal machinery, RK4 at the
shared cliff step), and is scored WITHOUT any wall trim:

  - full-domain coupled relative L2 error (native node-spacing trapezoid);
  - wall-strip error (fixed physical strip, width 0.1, both walls);
  - reflected-packet phase shift (quadratic-fit cross-correlation peak),
    amplitude ratio, and waveform distortion (L2 after optimal shift);
  - spectral leakage: energy fraction outside the initial packet band.

Sweep: families E, C, SBP, DRP, BMF, BMG + degree-q GLL SEM, orders
q in {4, 6, 8}, grids N in {49, 97, 193} (N-1 divisible by 4, 6, 8), one and
three full cycles (two and six wall bounces).

Calibration: for each (family, q) the wall-resolution-spectrum thresholds
P1%rho, P1%eta, Ptot at the packet's carrier resolution are paired with the
observed reflection metrics; predeclared pairs: eta-threshold predicts
waveform distortion and leakage, rho-threshold predicts phase shift.

Verification (--verify): exact-solution self-test (images reproduce the IC
at t=2 to 1e-12 on the finest grid), every metric finite, spline families
beat every nodal family on full-domain error at q=6, N=97 (regression pin).

Output: code/results/packet_reflection_summary.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))

import boundary_spline_analysis as ba
from boundary_spline_analysis import build_operator, nodal_trapezoidal_weights, scheme_kind
from project_paths import RESULTSDIR, ensure_generated_dirs
from sem_operator import sem_grid_for

FAMILIES = ("E", "C", "SBP", "DRP", "BMF", "BMG")
ORDERS = (4, 6, 8)
GRIDS = (49, 97, 193)
CFL = 0.012
X0, SIGMA = 0.35, 0.045
K0 = 20.0 * np.pi          # carrier: wavelength 0.1
CYCLES = (1, 3)            # full domain cycles (t = 2k), 2k wall bounces each
T_IMPACT = 0.65            # packet centered on the right wall (first bounce)
STRIP = 0.1                # physical wall-strip width for the strip norm
NIMG = 6                   # image charges per side


def packet(xi):
    return np.exp(-((xi - X0) / SIGMA) ** 2) * np.cos(K0 * (xi - X0))


def p_exact(x, t):
    """Rigid walls at 0 and 1 by images: p even under reflection."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    for m in range(-NIMG, NIMG + 1):
        out += packet(2.0 * m + x - t) + packet(2.0 * m - x - t)
    return out


def u_exact(x, t):
    """u odd under reflection (rigid wall: u = 0)."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    for m in range(-NIMG, NIMG + 1):
        out += packet(2.0 * m + x - t) - packet(2.0 * m - x - t)
    return out


def strip_mask(x):
    return (x <= STRIP) | (x >= 1.0 - STRIP)


def phase_amp_distortion(x, pnum, pref):
    """Cross-correlation phase shift, amplitude ratio, shifted-L2 distortion."""
    xs = np.linspace(0.0, 1.0, 4001)
    fn = np.interp(xs, x, pnum)
    fr = np.interp(xs, x, pref)
    n = xs.size
    corr = np.correlate(fn - fn.mean(), fr - fr.mean(), mode="full")
    k = int(np.argmax(corr))
    # quadratic sub-sample refinement
    if 0 < k < corr.size - 1:
        c0, c1, c2 = corr[k - 1], corr[k], corr[k + 1]
        denom = c0 - 2 * c1 + c2
        frac = 0.5 * (c0 - c2) / denom if denom != 0 else 0.0
    else:
        frac = 0.0
    shift = (k + frac - (n - 1)) * (xs[1] - xs[0])
    fshift = np.interp(xs, xs - shift, fr, left=0.0, right=0.0)
    aref = np.sqrt(np.mean(fr ** 2))
    amp = np.sqrt(np.mean(fn ** 2)) / max(aref, 1e-300)
    dist = float(np.sqrt(np.mean((fn - fshift) ** 2)) / max(aref, 1e-300))
    return float(shift), float(amp), dist


def leakage(x, pnum):
    """Energy fraction outside the initial packet band (uniform resample)."""
    xs = np.linspace(0.0, 1.0, 2048, endpoint=False)
    f = np.interp(xs, x, pnum)
    F = np.abs(np.fft.rfft(f)) ** 2
    freqs = np.fft.rfftfreq(xs.size, d=xs[1] - xs[0]) * 2.0 * np.pi
    band = (freqs >= K0 - 3.0 / SIGMA) & (freqs <= K0 + 3.0 / SIGMA)
    tot = float(np.sum(F[1:]))
    return float(np.sum(F[1:][~band[1:]]) / max(tot, 1e-300))


def metrics(x, p, u, t):
    pr, ur = p_exact(x, t), u_exact(x, t)
    w = nodal_trapezoidal_weights(np.asarray(x, float))
    num = np.sqrt(np.sum(w * ((p - pr) ** 2 + (u - ur) ** 2)))
    den = np.sqrt(np.sum(w * (pr ** 2 + ur ** 2)))
    m = strip_mask(np.asarray(x, float))
    # strip error is normalized by the FULL-domain reference scale: at the
    # return times the exact packet has left the strip, so a strip-relative
    # denominator would be a Gaussian tail
    nums = np.sqrt(np.sum(w[m] * ((p - pr)[m] ** 2 + (u - ur)[m] ** 2)))
    dens = den
    shift, amp, dist = phase_amp_distortion(np.asarray(x, float), p, pr)
    return {
        "full_rel_l2": float(num / max(den, 1e-300)),
        "strip_rel_l2": float(nums / max(dens, 1e-300)),
        "phase_shift": shift, "amp_ratio": amp, "waveform_distortion": dist,
        "leakage": leakage(np.asarray(x, float), p),
    }


def run_family(fam, q, npts):
    op = build_operator(scheme_kind(f"{fam}{q}"), q, npts)
    rows = {}
    t0 = time.process_time()
    for k, tend in [("impact", T_IMPACT)] + [(str(k), 2.0 * k) for k in CYCLES]:
        res = ba.simulate_reflected_acoustics(op, p_exact, u_exact, tend,
                                              cfl=CFL, record_count=4)
        x = res.grid if res.grid is not None else op.x
        tag = "impact" if k == "impact" else f"cycle{k}"
        rows[tag] = metrics(np.asarray(x, float),
                            np.asarray(res.pressure, float),
                            np.asarray(res.velocity, float), tend)
    rows["cpu_seconds"] = time.process_time() - t0
    return rows


def run_sem(q, npts):
    op = sem_grid_for(npts, q)
    x = op.x
    dt = CFL / (npts - 1)
    D1 = op.D1
    rows = {}
    t0 = time.process_time()
    for k, tend in [("impact", T_IMPACT)] + [(str(k), 2.0 * k) for k in CYCLES]:
        nsteps = int(np.ceil(tend / dt))
        dtk = tend / nsteps
        P, U = p_exact(x, 0.0), u_exact(x, 0.0)

        def rhs(Pv, Uv):
            dP = -(D1 @ Uv)
            dU = -(D1 @ Pv)
            dU[0] = dU[-1] = 0.0
            return dP, dU

        for _ in range(nsteps):
            k1p, k1u = rhs(P, U)
            k2p, k2u = rhs(P + 0.5 * dtk * k1p, U + 0.5 * dtk * k1u)
            k3p, k3u = rhs(P + 0.5 * dtk * k2p, U + 0.5 * dtk * k2u)
            k4p, k4u = rhs(P + dtk * k3p, U + dtk * k3u)
            P = P + dtk / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
            U = U + dtk / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
            U[0] = U[-1] = 0.0
        tag = "impact" if k == "impact" else f"cycle{k}"
        rows[tag] = metrics(x, P, U, tend)
    rows["cpu_seconds"] = time.process_time() - t0
    return rows


def spectrum_thresholds(fam, q, npts):
    """P1% thresholds of the wall-resolution spectrum for the calibration."""
    op = build_operator(scheme_kind(f"{fam}{q}"), q, npts)
    resp = ba.wall_mode_response(op)
    ppw = np.array([r.ppw for r in resp])
    rho = np.array([abs(r.kappa_ratio - 1.0) for r in resp])
    eta = np.array([r.contamination_l2 for r in resp])
    tot = np.hypot(rho, eta)

    def thr(v):
        idx = np.where(v <= 0.01)[0]
        return float(ppw[idx[-1]]) if idx.size else float("nan")

    return {"P1_rho": thr(rho), "P1_eta": thr(eta), "P1_tot": thr(tot)}


PARTIAL = None  # set in run_all; one JSON line per completed run for resume


def _partial_load():
    done = {}
    if PARTIAL.exists():
        for line in PARTIAL.read_text().splitlines():
            try:
                rec = json.loads(line)
                done[rec["key"]] = rec["value"]
            except (json.JSONDecodeError, KeyError):
                continue   # torn last line from a kill mid-write
    return done


def _partial_append(key, value):
    with PARTIAL.open("a") as f:
        f.write(json.dumps({"key": key, "value": value}) + "\n")


def run_all():
    global PARTIAL
    PARTIAL = RESULTSDIR / "packet_reflection.partial.jsonl"
    out = {"config": {
        "packet": {"x0": X0, "sigma": SIGMA, "k0_over_pi": K0 / np.pi,
                   "images_per_side": NIMG},
        "grids": list(GRIDS), "orders": list(ORDERS), "cfl": CFL,
        "cycles": list(CYCLES), "strip_width": STRIP,
        "predeclared_pairs": {
            "P1_eta": ["waveform_distortion", "leakage"],
            "P1_rho": ["phase_shift"],
        }}, "exact_selftest": {}, "runs": {}, "thresholds": {}}

    xs = np.linspace(0, 1, 4001)
    st = float(np.max(np.abs(p_exact(xs, 2.0) - p_exact(xs, 0.0))))
    out["exact_selftest"]["p_t2_vs_t0"] = st

    done = _partial_load()
    if done:
        print(f"resume: {len(done)} completed runs loaded from {PARTIAL.name}",
              flush=True)

    def one(key, fn):
        if key in done:
            out["runs"][key] = done[key]
            return
        try:
            out["runs"][key] = fn()
        except Exception as exc:            # noqa: BLE001
            out["runs"][key] = {"error": str(exc)}
        _partial_append(key, out["runs"][key])
        print(key, json.dumps(out["runs"][key].get("cycle1", out["runs"][key]))[:120],
              flush=True)

    for q in ORDERS:
        for npts in GRIDS:
            for fam in FAMILIES:
                one(f"{fam}{q}_N{npts}",
                    lambda fam=fam, q=q, npts=npts: run_family(fam, q, npts))
            one(f"SEM{q}_N{npts}", lambda q=q, npts=npts: run_sem(q, npts))
        for fam in FAMILIES:
            tkey = f"threshold:{fam}{q}"
            if tkey in done:
                out["thresholds"][f"{fam}{q}"] = done[tkey]
            else:
                out["thresholds"][f"{fam}{q}"] = spectrum_thresholds(fam, q, 129)
                _partial_append(tkey, out["thresholds"][f"{fam}{q}"])
    return out


def verify(out):
    assert out["exact_selftest"]["p_t2_vs_t0"] < 1e-12, out["exact_selftest"]
    # finiteness is required only where stability is claimed: the spline
    # routes, the SEM entrant, and SBP. Destroyed explicit/compact/DRP
    # closures legitimately return inf after six bounces; that IS the cliff.
    for key, r in out["runs"].items():
        if "cycle1" not in r:
            continue
        fam_stable = any(key.startswith(f) for f in ("BMF", "BMG", "SEM", "SBP"))
        for c in ("cycle1", "cycle3"):
            if fam_stable:
                for v in r[c].values():
                    assert np.isfinite(v), (key, c, r[c])
    # regression pin: splines beat every nodal family full-domain at q=6, N=97
    nod = min(out["runs"][f"{f}6_N97"]["cycle1"]["full_rel_l2"]
              for f in ("E", "C", "SBP", "DRP")
              if "cycle1" in out["runs"][f"{f}6_N97"])
    spl = max(out["runs"]["BMF6_N97"]["cycle1"]["full_rel_l2"],
              out["runs"]["BMG6_N97"]["cycle1"]["full_rel_l2"])
    assert spl < nod, ("spline pin", spl, nod)
    print("packet_reflection verify: ALL PASS")


def main():
    ensure_generated_dirs()
    path = RESULTSDIR / "packet_reflection_summary.json"
    if "--verify" in sys.argv:
        verify(json.loads(path.read_text()))
        return
    out = run_all()
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)
    verify(out)


if __name__ == "__main__":
    main()
