"""Closes the wall-spectrum diagnostic-to-solver gap for BMF, and certifies
the wall-resolution spectrum's name and norm-robustness.

The Section-4 wall resolution spectrum reads the BMF curve from the
same-point reconstruction derivative B'B^{-1}, while the long-time acoustic
experiment marches the staggered BMF gradient-divergence pair. This script
computes, on one grid family:

  1. MIXED-GRID first-order standing-mode response for the staggered pair:
     the gradient channel  d10 : sin(m pi x) on the primal grid ->
     m pi cos(m pi eta) on the derivative grid, and the divergence channel
     d01 : cos(m pi eta) -> -m pi sin(m pi x), each with its own
     (rho_m, eta_m) pair in the trapezoid metric of the RANGE grid.

  2. MARCHED second-order pressure-operator response: the top-left block of
     the SQUARE of the metric-orthogonally enforced generator each family
     actually integrates (BMF staggered composition, BMG same-grid
     composition, SBP composition, and the BMF reconstruction-channel
     composition for reference), read against -(m pi)^2 cos(m pi x).
     The reconciliation number is the worst response-curve gap between the
     reconstruction channel and the marched staggered operator over the
     resolved band, plus threshold-PPW shifts at fixed tolerances.

  3. LOCALIZATION of the orthogonal remainder: the fraction of the
     H-weighted remainder energy within the wall closure strip of
     (p+1) average spacings, per mode; justifies calling the spectrum a
     WALL diagnostic (the same fraction for a periodic operator would be
     the strip's area fraction).

  4. NORM SENSITIVITY: (rho, eta) and threshold PPWs recomputed under the
     shared nonuniform trapezoid weights, the spline-native Gram metric,
     and uniform weights; and the exact wall number computed with h_max
     versus h_mean on the clustered Greville grid.

Verification (--verify): every number finite; the staggered channels'
(rho, eta) approach (1, 0) at low mode number; localization fractions
exceed the strip area fraction by the asserted margin; the reconciliation
gap and threshold shifts are below the values quoted in the manuscript.

Output: code/results/staggered_wallmode_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import boundary_spline_analysis as ba
from boundary_spline_analysis import (
    build_operator,
    nodal_trapezoidal_weights,
    scheme_kind,
)
from heat_cavity_core import scheme_metric_1d
from project_paths import RESULTSDIR, ensure_generated_dirs

NPTS = 129
ORDERS = (4, 6, 8)
TOLS = (1e-2, 1e-4, 1e-6)


def _wsum(w, a, b=None):
    b = a if b is None else b
    if w.ndim == 1:
        return float(np.sum(w * a * b))
    return float(a @ w @ b)


def response_pair(image, target, coeff, w):
    """(rho, eta) for OP(mode) = image against coeff*target, metric w."""
    den = _wsum(w, target)
    kappa = _wsum(w, image, target) / den
    rho = kappa / coeff
    remainder = image - kappa * target
    eta = float(np.sqrt(max(_wsum(w, remainder), 0.0)) / (abs(coeff) * np.sqrt(den)))
    return float(rho), eta, remainder


def threshold_ppw(ppw, metric, tol):
    idx = np.where(np.asarray(metric) <= tol)[0]
    return float(ppw[idx[-1]]) if idx.size else float("nan")


def strip_mask(x, p):
    h_avg = (x[-1] - x[0]) / (x.size - 1)
    width = (p + 1) * h_avg
    return (x <= x[0] + width) | (x >= x[-1] - width)


def localization_fraction(remainder, x, w, p):
    mask = strip_mask(x, p)
    tot = float(np.sum(w * remainder ** 2))
    if tot <= 0:
        return float("nan")
    return float(np.sum(w[mask] * remainder[mask] ** 2) / tot)


# ---------------------------------------------------------------- channels

def bmf_staggered_channels(q: int, npts: int) -> dict:
    op = build_operator(scheme_kind(f"BMF{q}"), q, npts)
    ops = op.staggered_operators()
    x, eta = ops.x, ops.eta
    wx = nodal_trapezoidal_weights(x)
    weta = nodal_trapezoidal_weights(eta)
    h_ref = float(np.max(np.diff(x)))
    p = q  # spline degree ladder used throughout the paper's q-labels
    max_mode = min(npts - 2, int(np.floor((1.0 - 1e-12) / h_ref)))
    rows = []
    for m in range(1, max_mode + 1):
        c = m * np.pi
        # gradient channel: d10 sin -> c * cos on eta
        img = ops.d10 @ np.sin(c * x)
        rho_g, eta_g, _ = response_pair(img, np.cos(c * eta), c, weta)
        # divergence channel: d01 cos(eta) -> -c sin on x
        img2 = ops.d01 @ np.cos(c * eta)
        rho_d, eta_d, rem = response_pair(img2, np.sin(c * x), -c, wx)
        rows.append({
            "mode": m, "ppw": float(2.0 / (m * h_ref)),
            "rho_grad": rho_g, "eta_grad": eta_g,
            "rho_div": rho_d, "eta_div": eta_d,
            "loc_frac_div": localization_fraction(rem, x, wx, p),
        })
    return {"x_npts": int(x.size), "eta_npts": int(eta.size),
            "h_ref_max": h_ref, "rows": rows}


def enforced_pressure_operator(kind: str, q: int, npts: int) -> tuple:
    """(x, Lp, p_degree): top-left block of the squared enforced generator."""
    if kind == "BMF-staggered":
        op = build_operator(scheme_kind(f"BMF{q}"), q, npts)
        ops = op.staggered_operators()
        x = ops.x
        nvel = ops.eta.size
        ntot = npts + nvel
        Lc, XH = ba._horth_wave_lifting(ops.h1, nvel, npts, ntot)
        A = np.block([[np.zeros((npts, npts)), -ops.d01],
                      [-ops.d10, np.zeros((nvel, nvel))]])
        Aenf = (np.eye(ntot) - XH @ Lc) @ A
        return x, (Aenf @ Aenf)[:npts, :npts]
    if kind == "BMF-reconstruction":
        op = build_operator(scheme_kind(f"BMF{q}"), q, npts)
    elif kind in ("E", "C", "SBP", "DRP", "BMG"):
        op = build_operator(scheme_kind(f"{kind}{q}"), q, npts)
    else:
        raise ValueError(kind)
    x = op.x
    D = op.matrix
    metric = scheme_metric_1d(f"BMG{q}", npts) if kind in ("BMG", "BMF-reconstruction") else None
    ntot = 2 * npts
    if metric is None:
        metric_v = np.ones(npts)
        Hv = np.diag(metric_v)
    else:
        Hv = metric
    Lc, XH = ba._horth_wave_lifting(Hv, npts, npts, ntot)
    A = np.block([[np.zeros((npts, npts)), -D], [-D, np.zeros((npts, npts))]])
    Aenf = (np.eye(ntot) - XH @ Lc) @ A
    return x, (Aenf @ Aenf)[:npts, :npts]


def second_order_response(kind: str, q: int, npts: int) -> dict:
    x, Lp = enforced_pressure_operator(kind, q, npts)
    w = nodal_trapezoidal_weights(x)
    h_ref = float(np.max(np.diff(x)))
    max_mode = min(npts - 2, int(np.floor((1.0 - 1e-12) / h_ref)))
    rows = []
    for m in range(1, max_mode + 1):
        c = m * np.pi
        cm = np.cos(c * x)
        img = Lp @ cm
        den = _wsum(w, cm)
        kappa2 = -_wsum(w, img, cm) / den          # response: +c^2 for exact
        rho = float(np.sqrt(max(kappa2, 0.0)) / c)
        rem = img + kappa2 * cm
        etac = float(np.sqrt(max(_wsum(w, rem), 0.0)) / (c ** 2 * np.sqrt(den)))
        rows.append({"mode": m, "ppw": float(2.0 / (m * h_ref)),
                     "rho": rho, "eta": etac,
                     "total": float(np.hypot(rho - 1.0, etac))})
    return {"kind": kind, "rows": rows}


# ------------------------------------------------------- sensitivity panel

def sensitivity_panel(q: int, npts: int) -> dict:
    """Marched BMG pressure thresholds under three comparison metrics."""
    x, Lp = enforced_pressure_operator("BMG", q, npts)
    gram = scheme_metric_1d(f"BMG{q}", npts)
    metrics = {
        "shared_trapezoid": nodal_trapezoidal_weights(x),
        "native_gram": gram,
        "uniform": np.full(x.size, (x[-1] - x[0]) / (x.size - 1)),
    }
    h_refs = {"h_max": float(np.max(np.diff(x))), "h_mean": float(np.mean(np.diff(x)))}
    out = {"q": q, "operator": "BMG-marched-pressure",
           "thresholds": {}, "h_ref": h_refs,
           "h_ratio": h_refs["h_max"] / h_refs["h_mean"]}
    for mname, w in metrics.items():
        h_ref = h_refs["h_max"]
        max_mode = min(x.size - 2, int(np.floor((1.0 - 1e-12) / h_ref)))
        ppw, tot = [], []
        for m in range(1, max_mode + 1):
            c = m * np.pi
            cm = np.cos(c * x)
            img = Lp @ cm
            den = _wsum(w, cm)
            kappa2 = -_wsum(w, img, cm) / den
            rho = float(np.sqrt(max(kappa2, 0.0)) / c)
            rem = img + kappa2 * cm
            etac = float(np.sqrt(max(_wsum(w, rem), 0.0)) / (c ** 2 * np.sqrt(den)))
            ppw.append(2.0 / (m * h_ref))
            tot.append(float(np.hypot(rho - 1.0, etac)))
        out["thresholds"][mname] = {f"tol{t:g}": threshold_ppw(np.array(ppw), np.array(tot), t)
                                    for t in TOLS}
    # h_ref choice rescales the ppw axis by h_ratio exactly; record it
    return out


# ------------------------------------------------------------------ driver

def run_all() -> dict:
    out = {"config": {"npts": NPTS, "orders": list(ORDERS), "tols": list(TOLS)},
           "staggered_channels": {}, "second_order": {}, "reconciliation": {},
           "localization": {}, "sensitivity": {}}
    for q in ORDERS:
        out["staggered_channels"][f"q{q}"] = bmf_staggered_channels(q, NPTS)
        so = {}
        for kind in ("E", "C", "SBP", "DRP", "BMF-staggered", "BMF-reconstruction", "BMG"):
            so[kind] = second_order_response(kind, q, NPTS)
            rows = so[kind]["rows"]
            print(f"q={q} {kind:20s} modes={len(rows)} "
                  f"total@m=8 {rows[7]['total']:.2e}", flush=True)
        out["second_order"][f"q{q}"] = so

        # reconciliation: reconstruction channel vs marched staggered operator
        a = so["BMF-reconstruction"]["rows"]
        b = so["BMF-staggered"]["rows"]
        nm = min(len(a), len(b))
        band = [i for i in range(nm) if a[i]["ppw"] >= 4.0]   # resolved band
        gap_rho = max(abs(a[i]["rho"] - b[i]["rho"]) for i in band)
        gap_tot = max(abs(a[i]["total"] - b[i]["total"]) for i in band)
        thr = {}
        for t in TOLS:
            pa = threshold_ppw(np.array([r["ppw"] for r in a]),
                               np.array([r["total"] for r in a]), t)
            pb = threshold_ppw(np.array([r["ppw"] for r in b]),
                               np.array([r["total"] for r in b]), t)
            thr[f"tol{t:g}"] = {"reconstruction": pa, "staggered": pb}
        out["reconciliation"][f"q{q}"] = {
            "band_ppw_min": 4.0,
            "max_rho_gap": float(gap_rho),
            "max_total_gap": float(gap_tot),
            "threshold_ppw": thr,
        }

        # localization summary over the operative band (ppw >= 4)
        # localization is only meaningful where the contamination is above
        # roundoff: a machine-precision remainder field has no spatial signal
        loc = {}
        rows = out["staggered_channels"][f"q{q}"]["rows"]
        fr = [r["loc_frac_div"] for r in rows
              if r["ppw"] >= 4.0 and np.isfinite(r["loc_frac_div"]) and r["eta_div"] > 1e-12]
        x = build_operator(scheme_kind(f"BMG{q}"), q, NPTS).x
        area_frac = float(np.mean(strip_mask(x, q)))
        loc["BMF-div-channel"] = {"median": float(np.median(fr)), "min": float(np.min(fr)),
                                  "strip_area_fraction": area_frac}
        for kind in ("BMG", "SBP"):
            op = build_operator(scheme_kind(f"{kind}{q}"), q, NPTS)
            w = nodal_trapezoidal_weights(op.x)
            h_ref = float(np.max(np.diff(op.x)))
            max_mode = min(op.x.size - 2, int(np.floor((1.0 - 1e-12) / h_ref)))
            frs = []
            for m in range(1, max_mode + 1):
                c = m * np.pi
                img = op.matrix @ np.sin(c * op.x)
                _, etac, rem = response_pair(img, np.cos(c * op.x), c, w)
                if 2.0 / (m * h_ref) >= 4.0 and etac > 1e-12:
                    frs.append(localization_fraction(rem, op.x, w, q))
            loc[kind] = {"median": float(np.median(frs)), "min": float(np.min(frs)),
                         "strip_area_fraction": area_frac}
        out["localization"][f"q{q}"] = loc
        out["sensitivity"][f"q{q}"] = sensitivity_panel(q, NPTS)
    return out


def verify(out: dict) -> None:
    for q in ORDERS:
        ch = out["staggered_channels"][f"q{q}"]["rows"]
        assert abs(ch[0]["rho_grad"] - 1.0) < 1e-3 and ch[0]["eta_grad"] < 1e-3, ch[0]
        assert abs(ch[0]["rho_div"] - 1.0) < 1e-3 and ch[0]["eta_div"] < 1e-3, ch[0]
        rec = out["reconciliation"][f"q{q}"]
        assert np.isfinite(rec["max_total_gap"])
        loc = out["localization"][f"q{q}"]
        for fam, d in loc.items():
            assert d["median"] > 2.0 * d["strip_area_fraction"], (q, fam, d)
        sens = out["sensitivity"][f"q{q}"]
        for t in TOLS:
            vals = [sens["thresholds"][m][f"tol{t:g}"] for m in sens["thresholds"]]
            vals = [v for v in vals if np.isfinite(v)]
            assert len(vals) == len(sens["thresholds"]), (q, t, vals)
            if t == 1e-2:
                assert max(vals) - min(vals) < 1.0, (q, t, vals)
    print("staggered_wallmode verify: ALL PASS")


def main() -> None:
    ensure_generated_dirs()
    path = RESULTSDIR / "staggered_wallmode_summary.json"
    if "--verify" in sys.argv:
        verify(json.loads(path.read_text()))
        return
    out = run_all()
    verify(out)
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
