"""Off-grid fairness ensemble.

Three controls for the off-grid comparisons: the spline routes use
their native global space while finite differences receive one arbitrarily
chosen local degree-(q+1) Lagrange interpolant; the marker geometry is
cherry-prone; and the natural native-space owners (spectral element /
isogeometric) are absent. This certificate addresses all three in one
artifact:

  * SEEDED ENSEMBLES: 32 random smooth fields (decaying random Fourier
    series) x 64 markers per field (24 log-spaced approach distances per
    wall from 1e-4 to 0.4, plus 16 uniform-random interior points), with
    median, 95th-percentile, and worst-case errors reported, never a single
    chosen marker or field.
  * BEST-OF RECONSTRUCTION for every nodal family: each marker is evaluated
    with local Lagrange interpolants of orders q+1, q+2, and q+3 and the
    BEST result is kept per marker, so the nodal families are fielded at
    (at least) their best plausible local reconstruction.
  * A NATIVE-SPACE COMPETITOR: a spectral-element entrant (Gauss-Lobatto
    Legendre elements of degree 6, barycentric tensor evaluation inside its
    own elements) at matched total degrees of freedom, the natural owner of
    the off-grid regime as specified.
  * MATCHED DOFS: every family carries exactly N degrees of freedom.

Metrics: relative off-grid value error and relative off-grid derivative
error at every (field, marker) pair; aggregated per family and grid.

Usage:
  python code/s6_offgrid_ensemble.py --write code/results/offgrid_ensemble.json
  python code/s6_offgrid_ensemble.py --verify code/results/offgrid_ensemble.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import boundary_spline_analysis as ba  # noqa: E402
from project_paths import RESULTSDIR  # noqa: E402

NFIELDS = 32
NMODES = 8
GRIDS = (19, 37, 61)          # chosen so N-1 is divisible by 6 (SEM elements)
NODAL = ("E", "C", "SBP", "DRP")
Q = 6
SEED = 20260710


def random_fields(nfields: int, rng: np.random.Generator):
    """Decaying random Fourier fields with analytic derivatives."""
    fields = []
    for _ in range(nfields):
        amps = rng.standard_normal(NMODES) / (1.0 + np.arange(NMODES)) ** 2
        phases = rng.uniform(0, 2 * np.pi, NMODES)
        ks = np.pi * (1.0 + np.arange(NMODES))

        def f(x, a=amps, p=phases, k=ks):
            return sum(a[i] * np.sin(k[i] * x + p[i]) for i in range(NMODES))

        def df(x, a=amps, p=phases, k=ks):
            return sum(a[i] * k[i] * np.cos(k[i] * x + p[i]) for i in range(NMODES))

        fields.append((f, df))
    return fields


def marker_set(rng: np.random.Generator):
    d = np.logspace(-4, np.log10(0.4), 24)
    left = d
    right = 1.0 - d
    interior = rng.uniform(0.05, 0.95, 16)
    return np.sort(np.concatenate([left, right, interior]))


# ---------------------------------------------------------------- entrants

def nodal_bestof(kind: str, npts: int, f, df, markers):
    """Best plausible local reconstruction for a nodal family: local Lagrange
    stencils of sizes q+1, q+2, q+3, with the derivative taken BOTH by
    differentiating the local polynomial directly (fd_weights) and by
    interpolating the family's own nodal derivative samples; the best result
    per marker over all routes is kept."""
    op = ba.build_operator(kind, Q, npts)
    x = op.x
    u = f(x)
    du_nodal = op.matrix @ u
    best_v = np.full(markers.size, np.inf)
    best_d = np.full(markers.size, np.inf)
    fex, dex = f(markers), df(markers)
    for order in (Q + 1, Q + 2, Q + 3):
        W = ba.local_interpolation_matrix(x, markers, order)
        best_v = np.minimum(best_v, np.abs(W @ u - fex))
        best_d = np.minimum(best_d, np.abs(W @ du_nodal - dex))
        d_direct = np.empty(markers.size)
        for m, x0 in enumerate(markers):
            idx = ba.local_stencil_indices(x, float(x0), order)
            wts = ba.fd_weights(float(x0), x[idx], 1)
            d_direct[m] = float(wts @ u[idx])
        best_d = np.minimum(best_d, np.abs(d_direct - dex))
    return best_v, best_d


def spline_native(kind: str, npts: int, f, df, markers):
    op = ba.build_operator(ba.scheme_kind(kind + "6"), Q, npts)
    u = f(op.x)
    vals, dvals = op.field_and_derivative(markers, u)
    return np.abs(vals - f(markers)), np.abs(dvals - df(markers))


def sem_native(npts: int, f, df, markers):
    """Spectral-element entrant: degree-6 Gauss-Lobatto Legendre elements,
    matched DOFs (npts = 6K+1), native barycentric evaluation per element."""
    P = 6
    K = (npts - 1) // P
    assert K * P + 1 == npts, (npts, K)
    # GLL nodes of degree P on [-1,1] (P+1 points): roots of (1-x^2)P'_P
    from numpy.polynomial import legendre as L
    inner = L.legder([0] * P + [1])
    gl = np.sort(np.concatenate([[-1.0], np.roots(inner[::-1] if False else
                                                  np.polynomial.legendre.leg2poly(inner)[::-1]).real,
                                 [1.0]]))
    edges = np.linspace(0.0, 1.0, K + 1)
    fex, dex = f(markers), df(markers)
    v_err = np.empty(markers.size)
    d_err = np.empty(markers.size)
    # barycentric weights for the GLL nodes
    wb = np.ones(P + 1)
    for j in range(P + 1):
        wb[j] = 1.0 / np.prod(gl[j] - np.delete(gl, j))
    for m, xm in enumerate(markers):
        e = min(int(xm * K), K - 1)
        a, b = edges[e], edges[e + 1]
        xn = 0.5 * (a + b) + 0.5 * (b - a) * gl
        un = f(xn)
        t = (2.0 * (xm - a) / (b - a)) - 1.0
        diff = t - gl
        if np.any(np.abs(diff) < 1e-14):
            j = int(np.argmin(np.abs(diff)))
            val = un[j]
            # derivative via differentiation matrix row
            drow = np.array([wb[k] / wb[j] / (gl[j] - gl[k]) if k != j else 0.0
                             for k in range(P + 1)])
            drow[j] = -np.sum(drow)
            der = (drow @ un) * 2.0 / (b - a)
        else:
            w = wb / diff
            val = np.sum(w * un) / np.sum(w)
            # derivative of the barycentric form
            s0, s1 = np.sum(w), np.sum(w * un)
            dw = -wb / diff ** 2
            der = (np.sum(dw * un) * s0 - s1 * np.sum(dw)) / s0 ** 2 \
                * 2.0 / (b - a)
        v_err[m] = abs(val - fex[m])
        d_err[m] = abs(der - dex[m])
    return v_err, d_err


# ---------------------------------------------------------------- ensemble

def aggregate(errs, scale):
    e = np.concatenate(errs) / scale
    return {"median": float(np.median(e)), "p95": float(np.quantile(e, 0.95)),
            "worst": float(np.max(e))}


def run(*, verbose: bool = True):
    rng = np.random.default_rng(SEED)
    fields = random_fields(NFIELDS, rng)
    out = {"config": {"nfields": NFIELDS, "grids": list(GRIDS), "q": Q,
                      "markers": "24 log-spaced per wall (1e-4..0.4) + 16 random interior, seeded",
                      "nodal_reconstruction": "best of local Lagrange orders q+1,q+2,q+3 per marker",
                      "sem": "degree-6 GLL elements, matched DOFs, native barycentric"},
           "results": []}
    for npts in GRIDS:
        errs = {}
        for f, df in fields:
            markers = marker_set(rng)
            scale_v = float(np.max(np.abs(f(markers))))
            scale_d = float(np.max(np.abs(df(markers))))
            for kind in NODAL:
                v, d = nodal_bestof(kind, npts, f, df, markers)
                errs.setdefault((kind, "v"), []).append(v / scale_v)
                errs.setdefault((kind, "d"), []).append(d / scale_d)
            for kind in ("BMF", "BMG"):
                v, d = spline_native(kind, npts, f, df, markers)
                errs.setdefault((kind, "v"), []).append(v / scale_v)
                errs.setdefault((kind, "d"), []).append(d / scale_d)
            v, d = sem_native(npts, f, df, markers)
            errs.setdefault(("SEM", "v"), []).append(v / scale_v)
            errs.setdefault(("SEM", "d"), []).append(d / scale_d)
        for (kind, which), lst in sorted(errs.items()):
            rec = {"family": kind, "metric": which, "npts": npts}
            rec.update(aggregate(lst, 1.0))
            out["results"].append(rec)
            if verbose:
                print(f"N={npts} {kind:4s} {which} median={rec['median']:.3e} "
                      f"p95={rec['p95']:.3e} worst={rec['worst']:.3e}", flush=True)
    return out


def verify_record(record: dict, *, recompute: bool = True) -> None:
    """Reject incomplete, internally inconsistent, or numerically altered data."""
    expected_config = {
        "nfields": NFIELDS,
        "grids": list(GRIDS),
        "q": Q,
        "markers": "24 log-spaced per wall (1e-4..0.4) + 16 random interior, seeded",
        "nodal_reconstruction": "best of local Lagrange orders q+1,q+2,q+3 per marker",
        "sem": "degree-6 GLL elements, matched DOFs, native barycentric",
    }
    if record.get("config") != expected_config:
        raise ValueError("off-grid configuration differs from the released protocol")
    rows = record.get("results")
    if not isinstance(rows, list):
        raise ValueError("off-grid results must be a list")
    expected_keys = {
        (family, metric, npts)
        for family in (*NODAL, "BMF", "BMG", "SEM")
        for metric in ("v", "d")
        for npts in GRIDS
    }
    indexed = {}
    for row in rows:
        if set(row) != {"family", "metric", "npts", "median", "p95", "worst"}:
            raise ValueError(f"unexpected off-grid row schema: {row}")
        key = (row["family"], row["metric"], row["npts"])
        if key in indexed:
            raise ValueError(f"duplicate off-grid row: {key}")
        values = np.asarray([row["median"], row["p95"], row["worst"]], float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"nonfinite or negative off-grid error: {key}")
        if not values[0] <= values[1] <= values[2]:
            raise ValueError(f"invalid off-grid quantile order: {key}")
        indexed[key] = row
    if set(indexed) != expected_keys:
        missing = sorted(expected_keys - set(indexed))
        extra = sorted(set(indexed) - expected_keys)
        raise ValueError(f"off-grid row coverage mismatch; missing={missing}, extra={extra}")
    if not recompute:
        return
    fresh = run(verbose=False)
    fresh_rows = {
        (row["family"], row["metric"], row["npts"]): row
        for row in fresh["results"]
    }
    for key in sorted(expected_keys):
        for field in ("median", "p95", "worst"):
            if not np.isclose(indexed[key][field], fresh_rows[key][field],
                              rtol=1.0e-12, atol=1.0e-15):
                raise AssertionError(
                    f"off-grid recomputation mismatch for {key} {field}: "
                    f"stored={indexed[key][field]!r}, recomputed={fresh_rows[key][field]!r}"
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default=None)
    ap.add_argument("--verify", default=None)
    args = ap.parse_args()
    if args.verify:
        d = json.loads(Path(args.verify).read_text())
        verify_record(d, recompute=True)
        print("offgrid_ensemble verify: ALL PASS (42 rows independently recomputed)")
        return
    out = run()
    dest = Path(args.write) if args.write else Path(RESULTSDIR) / "offgrid_ensemble.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
