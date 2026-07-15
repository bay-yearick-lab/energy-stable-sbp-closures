"""Curved-surface off-grid ensemble: values, wall-normal derivatives, and
full tractions on circles and a nontrivial smooth curve, closing the claim
that off-grid evaluation covers "values, normals, tractions" and not only
one-dimensional samples.

Fields: seeded analytic 2D solenoidal velocity fields (stream-function form,
so the exact stress tensor is analytic) plus an analytic pressure. Surfaces:
circles of two radii centered off-grid, plus a smooth star-shaped curve
r(theta) = r0 (1 + 0.18 cos 5 theta) with high curvature; markers sampled at
analytically known positions with exact outward normals.

Entrants at matched total DOFs on [0,1]^2:
  - BMG/BMF spline reconstruction (tensor clamped spline coefficients from
    Greville interpolation of the field samples; values and gradients from
    the exact spline derivative);
  - degree-6 GLL spectral-element native evaluation;
  - nodal oracle baseline: bicubic-through-quintic local Lagrange, best per
    marker (the same generous oracle as the 1D ensemble).

Scored quantities per marker: velocity value error, normal-derivative error
du/dn, traction error t = sigma . n with sigma = -p I + nu (grad u + grad
u^T); statistics: median, 95th percentile, and worst over all seeded fields,
surfaces, and markers. Verification (--verify): an analytic derivative is
checked by an independent finite-difference reconstruction, all values are finite, and spline beats the
oracle nodal baseline on median traction error at N=37, q=6.

Output: code/results/curved_offgrid_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))

from project_paths import RESULTSDIR, ensure_generated_dirs
from verify_cavity_blind import open_uniform_knots
from scipy.interpolate import BSpline
from sem_operator import sem_grid_for

SEED = 20260712
NFIELDS = 12
NMODES = 5
GRIDS = (19, 37, 61)
Q = 6
NU = 0.01


# ---------------------------------------------------------------- fields

def random_field(rng: np.random.Generator):
    """Solenoidal velocity from a random stream function + analytic pressure."""
    a = rng.standard_normal((NMODES, NMODES)) / (1 + np.arange(NMODES)[:, None] + np.arange(NMODES)[None, :]) ** 2
    b = rng.standard_normal((NMODES, NMODES)) / (1 + np.arange(NMODES)[:, None] + np.arange(NMODES)[None, :]) ** 2
    ks = np.pi * (1 + np.arange(NMODES))

    def psi(x, y, dx=0, dy=0):
        out = 0.0
        for i, kx in enumerate(ks):
            for j, ky in enumerate(ks):
                fx = np.sin(kx * x) if dx % 2 == 0 else np.cos(kx * x)
                fy = np.sin(ky * y) if dy % 2 == 0 else np.cos(ky * y)
                sx = (-1) ** (dx // 2) * kx ** dx
                sy = (-1) ** (dy // 2) * ky ** dy
                out = out + a[i, j] * sx * sy * fx * fy
        return out

    def pres(x, y, dx=0, dy=0):
        out = 0.0
        for i, kx in enumerate(ks):
            for j, ky in enumerate(ks):
                fx = np.cos(kx * x) if dx % 2 == 0 else -np.sin(kx * x)
                fy = np.cos(ky * y) if dy % 2 == 0 else -np.sin(ky * y)
                sx = (-1) ** ((dx + 1) // 2) * kx ** dx
                sy = (-1) ** ((dy + 1) // 2) * ky ** dy
                out = out + b[i, j] * sx * sy * fx * fy
        return out

    # velocity u = psi_y, v = -psi_x (solenoidal); all derivatives analytic
    fld = {
        "u": lambda x, y: psi(x, y, 0, 1),
        "v": lambda x, y: -psi(x, y, 1, 0),
        "ux": lambda x, y: psi(x, y, 1, 1),
        "uy": lambda x, y: psi(x, y, 0, 2),
        "vx": lambda x, y: -psi(x, y, 2, 0),
        "vy": lambda x, y: -psi(x, y, 1, 1),
        "p": lambda x, y: pres(x, y, 0, 0),
    }
    return fld


def traction(fld, x, y, nx, ny):
    """t = sigma . n with sigma = -p I + nu (grad u + grad u^T)."""
    p = fld["p"](x, y)
    ux, uy = fld["ux"](x, y), fld["uy"](x, y)
    vx, vy = fld["vx"](x, y), fld["vy"](x, y)
    sxx = -p + 2 * NU * ux
    syy = -p + 2 * NU * vy
    sxy = NU * (uy + vx)
    return sxx * nx + sxy * ny, sxy * nx + syy * ny


# -------------------------------------------------------------- surfaces

def surfaces():
    out = []
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    for name, cx, cy, r0, eps, lobes in (
        ("circle_small", 0.512, 0.487, 0.11, 0.0, 0),
        ("circle_large", 0.503, 0.508, 0.335, 0.0, 0),
        ("star", 0.501, 0.499, 0.24, 0.18, 5),
    ):
        r = r0 * (1 + eps * np.cos(lobes * th))
        drdth = -r0 * eps * lobes * np.sin(lobes * th)
        x = cx + r * np.cos(th)
        y = cy + r * np.sin(th)
        # outward normal of r(theta): ( r cos - r' sin, r sin + r' cos ) x tangent rotation
        tx = drdth * np.cos(th) - r * np.sin(th)
        ty = drdth * np.sin(th) + r * np.cos(th)
        nx, ny = ty, -tx
        nrm = np.hypot(nx, ny)
        nx, ny = nx / nrm, ny / nrm
        out.append({"name": name, "x": x, "y": y, "nx": nx, "ny": ny})
    return out


# -------------------------------------------------------------- entrants

def _basis_matrix(n, p, pts, deriv=0):
    """Rows: basis (or derivative) values at pts; full-knot BSpline handles
    the clamped repeated end-knots that basis_element cannot differentiate."""
    kn = open_uniform_knots(n, p)
    out = np.zeros((pts.size, n))
    e = np.zeros(n)
    for i in range(n):
        e[:] = 0.0
        e[i] = 1.0
        sp = BSpline(kn, e.copy(), p, extrapolate=False)
        if deriv:
            sp = sp.derivative(deriv)
        out[:, i] = np.nan_to_num(sp(pts))
    return out


def spline_eval2d(coef, p, xq, yq, dx=0, dy=0):
    n = coef.shape[0]
    bx = _basis_matrix(n, p, np.asarray(xq, float), dx)
    by = _basis_matrix(n, p, np.asarray(yq, float), dy)
    return np.einsum("mi,ij,mj->m", bx, coef, by)


def spline_entrant(npts, fld, surf):
    """Greville-interpolated clamped spline reconstruction of (u, v, p)."""
    kn = open_uniform_knots(npts, Q)
    grev = np.array([kn[i + 1:i + Q + 1].mean() for i in range(npts)])
    B = _basis_matrix(npts, Q, grev)
    B[0, :] = 0; B[0, 0] = 1
    B[-1, :] = 0; B[-1, -1] = 1
    Binv = np.linalg.inv(B)
    X, Y = np.meshgrid(grev, grev, indexing="ij")
    coefs = {k: Binv @ fld[k](X, Y) @ Binv.T for k in ("u", "v", "p")}
    xq, yq = surf["x"], surf["y"]
    b0x = _basis_matrix(npts, Q, xq); b1x = _basis_matrix(npts, Q, xq, 1)
    b0y = _basis_matrix(npts, Q, yq); b1y = _basis_matrix(npts, Q, yq, 1)
    res = {}
    for k in ("u", "v", "p"):
        C = coefs[k]
        res[k] = np.einsum("mi,ij,mj->m", b0x, C, b0y)
        res[k + "x"] = np.einsum("mi,ij,mj->m", b1x, C, b0y)
        res[k + "y"] = np.einsum("mi,ij,mj->m", b0x, C, b1y)
    return res


def nodal_oracle_entrant(npts, fld, surf):
    """Best-of local Lagrange (orders q, q+1, q+2 per marker), uniform grid."""
    xg = np.linspace(0, 1, npts)
    X, Y = np.meshgrid(xg, xg, indexing="ij")
    grids = {k: fld[k](X, Y) for k in ("u", "v", "p")}

    def lag1d(xq, pts, vals):
        w = np.ones(pts.size)
        for j in range(pts.size):
            w[j] = 1.0 / np.prod(pts[j] - np.delete(pts, j))
        d = xq - pts
        if np.any(np.abs(d) < 1e-14):
            j = int(np.argmin(np.abs(d)))
            val = vals[..., j]
            dv = sum(vals[..., k] * w[k] / w[j] / (pts[j] - pts[k])
                     for k in range(pts.size) if k != j)
            dsum = -sum(w[k] / w[j] / (pts[j] - pts[k]) for k in range(pts.size) if k != j)
            return val, dv + vals[..., j] * dsum
        ww = w / d
        s0 = ww.sum()
        val = (vals * ww).sum(axis=-1) / s0
        dw = -w / d ** 2
        der = ((vals * dw).sum(axis=-1) * s0 - (vals * ww).sum(axis=-1) * dw.sum()) / s0 ** 2
        return val, der

    res = {k + suf: np.zeros(surf["x"].size) for k in ("u", "v", "p") for suf in ("", "x", "y")}
    h = xg[1] - xg[0]
    for m, (a, b) in enumerate(zip(surf["x"], surf["y"])):
        best = {k: None for k in res}
        for width in (Q + 1, Q + 2, Q + 3):
            i0 = min(max(int(a / h) - width // 2, 0), npts - width)
            j0 = min(max(int(b / h) - width // 2, 0), npts - width)
            px, py = xg[i0:i0 + width], xg[j0:j0 + width]
            for k in ("u", "v", "p"):
                sub = grids[k][i0:i0 + width, j0:j0 + width]
                rowv = np.zeros(width); rowd = np.zeros(width)
                for jj in range(width):
                    rowv[jj], rowd[jj] = lag1d(b, py, sub[jj, :])
                val, dx_ = lag1d(a, px, rowv)
                _, dumy = lag1d(a, px, rowv)
                dyv, _ = lag1d(a, px, rowd)
                exact_v = fld[k](a, b)
                for suf, cand in (("", val), ("x", dx_), ("y", dyv)):
                    ek = k + suf
                    exact = {"": fld[k](a, b), "x": fld[k + "x"](a, b) if k + "x" in fld else None,
                             "y": fld[k + "y"](a, b) if k + "y" in fld else None}[suf]
                    if exact is None:
                        continue
                    err = abs(cand - exact)
                    if best[ek] is None or err < abs(best[ek] - exact):
                        best[ek] = cand
        for ek in res:
            res[ek][m] = best[ek] if best[ek] is not None else np.nan
    return res


def score(fld, surf, res):
    x, y, nx, ny = surf["x"], surf["y"], surf["nx"], surf["ny"]
    vex = np.hypot(fld["u"](x, y), fld["v"](x, y))
    scale_v = max(float(np.max(np.abs(vex))), 1e-30)
    verr = np.hypot(res["u"] - fld["u"](x, y), res["v"] - fld["v"](x, y)) / scale_v
    dundn_ex = fld["ux"](x, y) * nx + fld["uy"](x, y) * ny
    dundn = res["ux"] * nx + res["uy"] * ny
    scale_d = max(float(np.max(np.abs(dundn_ex))), 1e-30)
    derr = np.abs(dundn - dundn_ex) / scale_d
    tx_ex, ty_ex = traction(fld, x, y, nx, ny)
    p = res["p"]
    sxx = -p + 2 * NU * res["ux"]
    syy = -p + 2 * NU * res["vy"]
    sxy = NU * (res["uy"] + res["vx"])
    tx = sxx * nx + sxy * ny
    ty = sxy * nx + syy * ny
    scale_t = max(float(np.max(np.hypot(tx_ex, ty_ex))), 1e-30)
    terr = np.hypot(tx - tx_ex, ty - ty_ex) / scale_t
    return verr, derr, terr


def agg(errs):
    e = np.concatenate(errs)
    e = e[np.isfinite(e)]
    return {"median": float(np.median(e)), "p95": float(np.quantile(e, 0.95)),
            "worst": float(np.max(e)), "count": int(e.size)}


def sem_entrant(npts, fld, surf):
    """Degree-6 GLL element native barycentric tensor evaluation."""
    op = sem_grid_for(npts, 6)
    xg = op.x
    X, Y = np.meshgrid(xg, xg, indexing="ij")
    grids = {k: fld[k](X, Y) for k in ("u", "v", "p")}
    K = (npts - 1) // 6
    from sem_operator import gll_nodes_weights, lagrange_diff_matrix
    gl, _ = gll_nodes_weights(6)
    Dref = lagrange_diff_matrix(gl)
    wb = np.ones(7)
    for j in range(7):
        wb[j] = 1.0 / np.prod(gl[j] - np.delete(gl, j))
    edges = np.linspace(0, 1, K + 1)

    def eval1(vals_line, xm):
        e = min(int(xm * K), K - 1)
        a, b = edges[e], edges[e + 1]
        t = 2 * (xm - a) / (b - a) - 1
        seg = vals_line[e * 6:e * 6 + 7]
        d = t - gl
        if np.any(np.abs(d) < 1e-13):
            j = int(np.argmin(np.abs(d)))
            return seg[j], (Dref[j, :] @ seg) * 2 / (b - a)
        w = wb / d
        val = (w * seg).sum() / w.sum()
        dw = -wb / d ** 2
        der = ((dw * seg).sum() * w.sum() - (w * seg).sum() * dw.sum()) / w.sum() ** 2 * 2 / (b - a)
        return val, der

    res = {k + suf: np.zeros(surf["x"].size) for k in ("u", "v", "p") for suf in ("", "x", "y")}
    for m, (a, b) in enumerate(zip(surf["x"], surf["y"])):
        for k in ("u", "v", "p"):
            g = grids[k]
            rowv = np.zeros(npts); rowd = np.zeros(npts)
            for i in range(npts):
                rowv[i], rowd[i] = eval1(g[i, :], b)
            val, dx_ = eval1(rowv, a)
            dyv, _ = eval1(rowd, a)
            res[k][m], res[k + "x"][m], res[k + "y"][m] = val, dx_, dyv
    return res


def run_all(*, verbose: bool = True):
    rng = np.random.default_rng(SEED)
    out = {"config": {"nfields": NFIELDS, "grids": list(GRIDS), "q": Q, "nu": NU,
                      "seed": SEED, "surfaces": [s["name"] for s in surfaces()]},
           "selftest": {}, "results": []}
    # traction self-test: recompute via central differences of the analytic field
    fld = random_field(rng)
    s0 = surfaces()[0]
    tx, ty = traction(fld, s0["x"][:4], s0["y"][:4], s0["nx"][:4], s0["ny"][:4])
    eps = 1e-6
    ux_fd = (fld["u"](s0["x"][:4] + eps, s0["y"][:4]) - fld["u"](s0["x"][:4] - eps, s0["y"][:4])) / (2 * eps)
    out["selftest"]["ux_analytic_vs_fd"] = float(np.max(np.abs(ux_fd - fld["ux"](s0["x"][:4], s0["y"][:4]))))
    flds = [random_field(rng) for _ in range(NFIELDS)]
    for npts in GRIDS:
        errs = {("spline", c): [] for c in ("v", "d", "t")}
        errs.update({("nodal", c): [] for c in ("v", "d", "t")})
        errs.update({("sem", c): [] for c in ("v", "d", "t")})
        for fld in flds:
            for surf in surfaces():
                for name, fn in (("spline", spline_entrant),
                                 ("nodal", nodal_oracle_entrant),
                                 ("sem", sem_entrant)):
                    res = fn(npts, fld, surf)
                    v, d, t = score(fld, surf, res)
                    errs[(name, "v")].append(v)
                    errs[(name, "d")].append(d)
                    errs[(name, "t")].append(t)
        for (name, c), lst in errs.items():
            rec = {"entrant": name, "metric": c, "npts": npts, **agg(lst)}
            out["results"].append(rec)
            if verbose:
                print(rec, flush=True)
    return out


def verify_record(out: dict, *, recompute: bool = True) -> None:
    expected_config = {
        "nfields": NFIELDS,
        "grids": list(GRIDS),
        "q": Q,
        "nu": NU,
        "seed": SEED,
        "surfaces": [surface["name"] for surface in surfaces()],
    }
    if out.get("config") != expected_config:
        raise ValueError("curved off-grid configuration differs from the released protocol")
    selftest = out.get("selftest")
    if not isinstance(selftest, dict) or set(selftest) != {"ux_analytic_vs_fd"}:
        raise ValueError("curved off-grid self-test schema mismatch")
    if not np.isfinite(selftest["ux_analytic_vs_fd"]) or selftest["ux_analytic_vs_fd"] >= 1e-5:
        raise ValueError("curved off-grid analytic derivative self-test failed")
    raw_rows = out.get("results")
    if not isinstance(raw_rows, list):
        raise ValueError("curved off-grid results must be a list")
    expected_keys = {
        (entrant, metric, npts)
        for entrant in ("spline", "nodal", "sem")
        for metric in ("v", "d", "t")
        for npts in GRIDS
    }
    rows = {}
    expected_count = NFIELDS * len(surfaces()) * 48
    for row in raw_rows:
        if set(row) != {"entrant", "metric", "npts", "median", "p95", "worst", "count"}:
            raise ValueError(f"unexpected curved off-grid row schema: {row}")
        key = (row["entrant"], row["metric"], row["npts"])
        if key in rows:
            raise ValueError(f"duplicate curved off-grid row: {key}")
        values = np.asarray([row["median"], row["p95"], row["worst"]], float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"invalid curved off-grid error: {key}")
        if not values[0] <= values[1] <= values[2]:
            raise ValueError(f"invalid curved off-grid quantile order: {key}")
        if row["count"] != expected_count:
            raise ValueError(f"wrong curved off-grid sample count: {key}")
        rows[key] = row
    if set(rows) != expected_keys:
        missing = sorted(expected_keys - set(rows))
        extra = sorted(set(rows) - expected_keys)
        raise ValueError(f"curved off-grid row coverage mismatch; missing={missing}, extra={extra}")
    for c in ("v", "d", "t"):
        if not rows[("spline", c, 37)]["median"] < rows[("nodal", c, 37)]["median"]:
            raise AssertionError(f"spline no longer leads the nodal oracle for curved metric {c}")
    if not recompute:
        return
    fresh = run_all(verbose=False)
    fresh_rows = {
        (row["entrant"], row["metric"], row["npts"]): row
        for row in fresh["results"]
    }
    if not np.isclose(
        selftest["ux_analytic_vs_fd"], fresh["selftest"]["ux_analytic_vs_fd"],
        rtol=1.0e-12, atol=1.0e-15,
    ):
        raise AssertionError("curved off-grid self-test does not reproduce")
    for key in sorted(expected_keys):
        for field in ("median", "p95", "worst"):
            # rtol 1e-8 absorbs round-off differences between BLAS/LAPACK builds.
            if not np.isclose(rows[key][field], fresh_rows[key][field],
                              rtol=1.0e-8, atol=1.0e-15):
                raise AssertionError(
                    f"curved off-grid recomputation mismatch for {key} {field}: "
                    f"stored={rows[key][field]!r}, recomputed={fresh_rows[key][field]!r}"
                )


def verify(out):
    verify_record(out, recompute=True)
    print("curved_offgrid verify: ALL PASS")


def main():
    ensure_generated_dirs()
    path = RESULTSDIR / "curved_offgrid_summary.json"
    if "--verify" in sys.argv:
        verify(json.loads(path.read_text()))
        return
    out = run_all()
    verify(out)
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
