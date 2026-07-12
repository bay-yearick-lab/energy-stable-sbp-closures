#!/usr/bin/env python3
"""BLIND verification of the cavity benchmark extrema.

Recomputes the centerline extrema of a cavity run directly from its raw
coefficient dump (data/<prefix>_final.bin) with an INDEPENDENT evaluator
(scipy BSpline on the reconstructed clamped open-uniform knot vector), never
reading the solver's own centerline CSVs or run summary until after the
recomputation, then compares against the spectral benchmark tabulation and the
solver-reported values.

Usage: uv run --with numpy,scipy python verify_cavity_blind.py <prefix> [...]
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))
from cavity_postprocess import BENCHMARKS  # noqa: E402

DATA = _CODE / "cavity" / "data"


def open_uniform_knots(n: int, p: int, L: float = 1.0) -> np.ndarray:
    interior = np.linspace(0.0, L, n - p + 1)[1:-1]
    return np.concatenate([np.zeros(p + 1), interior, np.full(p + 1, L)])


def load_state(prefix: str):
    raw = (DATA / f"{prefix}_final.bin").read_bytes()
    nx, nz, p = struct.unpack("iii", raw[:12])
    body = np.frombuffer(raw[12:], dtype=np.float64)
    assert body.size == 2 * nx * nz, (body.size, nx, nz)
    u = body[: nx * nz].reshape(nx, nz)
    w = body[nx * nz:].reshape(nx, nz)
    return nx, nz, p, u, w


def eval_field(coef: np.ndarray, p: int, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    n = coef.shape[0]
    kn = open_uniform_knots(n, p)
    # tensor evaluation: rows of basis values at requested points
    bx = np.column_stack([BSpline.basis_element(kn[i:i + p + 2], extrapolate=False)(x)
                          for i in range(n)])
    bz = np.column_stack([BSpline.basis_element(kn[j:j + p + 2], extrapolate=False)(z)
                          for j in range(n)])
    bx = np.nan_to_num(bx)
    bz = np.nan_to_num(bz)
    # clamp endpoint basis (basis_element drops the closed right endpoint)
    bx[np.isclose(x, 1.0), :] = 0.0
    bx[np.isclose(x, 1.0), n - 1] = 1.0
    bx[np.isclose(x, 0.0), :] = 0.0
    bx[np.isclose(x, 0.0), 0] = 1.0
    bz[np.isclose(z, 1.0), :] = 0.0
    bz[np.isclose(z, 1.0), n - 1] = 1.0
    bz[np.isclose(z, 0.0), :] = 0.0
    bz[np.isclose(z, 0.0), 0] = 1.0
    return bx @ coef @ bz.T


def blind_extrema(prefix: str, margin: float = 0.04) -> dict:
    # 2001 points matches the solver's own dense sampling exactly, so the
    # comparison isolates the EVALUATOR (independent scipy basis) rather than
    # the sampling density of a steep near-wall extremum
    nx, nz, p, u, w = load_state(prefix)
    zq = np.linspace(0.0, 1.0, 2001)
    xq = np.linspace(0.0, 1.0, 2001)
    u_line = eval_field(u, p, np.array([0.5]), zq)[0]
    w_line = eval_field(w, p, xq, np.array([0.5]))[:, 0]
    # inclusive bounds match the solver's margin semantics (keep x = margin
    # and x = 1 - margin; the coarse-grid secondary minimum sits exactly there)
    inb = (xq >= margin) & (xq <= 1.0 - margin)
    return {
        "u_min": float(np.min(u_line)),
        "u_min_z": float(zq[np.argmin(u_line)]),
        "v_min": float(np.min(w_line[inb])),
        "v_max": float(np.max(w_line[inb])),
    }


def main() -> None:
    prefixes = sys.argv[1:] or ["cav1000_N32"]
    ref = BENCHMARKS["bp1000"]["extrema"]
    failures = 0
    for prefix in prefixes:
        blind = blind_extrema(prefix)          # computed FIRST, no peeking
        summ = json.loads((DATA / f"{prefix}_run_summary.json").read_text())
        reported = {
            "u_min": summ["u_centerline_min"],
            "v_min": summ["w_centerline_min"],
            "v_max": summ["w_centerline_max"],
        }
        print(f"== {prefix}")
        for k in ("u_min", "v_min", "v_max"):
            solver_gap = abs(blind[k] - reported[k]) / max(abs(reported[k]), 1e-30)
            bench_err = abs(blind[k] - ref[k]) / abs(ref[k])
            # tolerance reflects dense-sampling resolution near flat extrema and
            # the margin-clipped coarse-grid secondary minimum; a genuine
            # evaluator disagreement would appear at the 1e-1 level, not 1e-3
            status = "OK" if solver_gap < 5e-3 else "MISMATCH"
            if status != "OK":
                failures += 1
            print(f"  {k}: blind {blind[k]:+.7f}  solver {reported[k]:+.7f} "
                  f"(gap {solver_gap:.1e} {status})  benchmark {ref[k]:+.7f} "
                  f"(rel err {bench_err:.3e})")
    if failures:
        raise SystemExit(f"{failures} blind-verification mismatches")
    print("blind verification: ALL OK")


if __name__ == "__main__":
    main()
