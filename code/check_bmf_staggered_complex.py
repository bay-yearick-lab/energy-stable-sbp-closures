#!/usr/bin/env python3
"""Numerical smoke check for the staggered BMF SBP complex.

This script is intentionally small: it demonstrates the BMF staggered
gradient-divergence usage pattern without running a PDE benchmark.
"""
from __future__ import annotations

import json

import numpy as np

import boundary_spline_analysis as ba
from project_paths import RESULTSDIR, ensure_generated_dirs


def main() -> None:
    ensure_generated_dirs()
    rows = []
    for degree in range(1, 8):
        for npts in range(degree + 2, degree + 8):
            ops = ba.bmf_staggered_operators(npts, degree)
            residual = ops.h0 @ ops.d01 + ops.d10.T @ ops.h1 - ops.boundary
            _, d_rec = ba.bmf_reconstructed_same_grid_derivative_matrix(npts, degree)
            _, d_raw = ba.bmf_nodal_operator(npts, degree)
            _, d2 = ba.bmf_flux_divergence_second_derivative_matrix(npts, degree)

            x = ops.x
            max_poly_error = 0.0
            for k in range(degree + 1):
                u = x ** k
                exact = np.zeros_like(x) if k < 2 else k * (k - 1) * x ** (k - 2)
                max_poly_error = max(max_poly_error, float(np.linalg.norm(d2 @ u - exact, np.inf)))

            rows.append({
                "degree": degree,
                "npts": npts,
                "staggered_sbp_norm2": float(np.linalg.norm(residual, 2)),
                "reconstruction_norm2": float(np.linalg.norm(d_rec - d_raw, 2)),
                "d2_poly_error_inf": max_poly_error,
            })

    summary = {
        "description": "BMF staggered/staggered SBP smoke check",
        "max_staggered_sbp_norm2": max(row["staggered_sbp_norm2"] for row in rows),
        "max_reconstruction_norm2": max(row["reconstruction_norm2"] for row in rows),
        "max_d2_poly_error_inf": max(row["d2_poly_error_inf"] for row in rows),
        "rows": rows,
    }
    out = RESULTSDIR / "bmf_staggered_complex_check.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "wrote": str(out),
        "max_staggered_sbp_norm2": summary["max_staggered_sbp_norm2"],
        "max_reconstruction_norm2": summary["max_reconstruction_norm2"],
        "max_d2_poly_error_inf": summary["max_d2_poly_error_inf"],
    }, indent=2))


if __name__ == "__main__":
    main()
