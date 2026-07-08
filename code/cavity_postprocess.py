#!/usr/bin/env python3
"""
Postprocess a cavity.cpp run: compare centerline u(0.5,z) and w(x,0.5) profiles
and the velocity extrema against the Re=1000 spectral benchmark (N=160
spectral solution, converged to 7 digits for the velocities; Tables 2, 4 and 6
of the published benchmark, doi 10.1016/S0045-7930(98)00002-4), cross-checked
against its headline values (u_min=-0.388570 at y=0.1717; v_max=0.376945 at
x=0.1578; v_min=-0.527077 at x=0.9092; primary vortex psi=-0.1189366,
omega=2.067753, at (x,y)=(0.5308,0.5652)).

Usage: uv run --with numpy python cavity_postprocess.py <outprefix> [data_dir] [results_dir]
"""
import sys
import json
import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# Re=1000 spectral benchmark digits (Table 4: u along x=0.5;
# Table 2: v along y=0.5; Table 6: velocity extrema; Table 1: primary-vortex
# center/streamfunction/vorticity).
# ---------------------------------------------------------------------------
BP_U_CENTERLINE = [   # (y, u) at x=0.5, Table 4 column x=0.5
    (1.0000, 1.0000000), (0.9766, 0.6644227), (0.9688, 0.5808359),
    (0.9609, 0.5169277), (0.9531, 0.4723329), (0.8516, 0.3372212),
    (0.7344, 0.1886747), (0.6172, 0.0570178), (0.5000, -0.0620561),
    (0.4531, -0.1081999), (0.2813, -0.2803696), (0.1719, -0.3885690),
    (0.1016, -0.3004561), (0.0703, -0.2228955), (0.0625, -0.2023300),
    (0.0547, -0.1812881), (0.0400, -0.1396601), (0.0300, -0.1090870),
    (0.0200, -0.0759628), (0.0100, -0.0397486), (0.0000, 0.0000000),
]
BP_V_CENTERLINE = [   # (x, v) at y=0.5, Table 2 column y=0.5
    (1.0000, 0.0000000), (0.9688, -0.2279225), (0.9609, -0.2936869),
    (0.9531, -0.3553213), (0.9453, -0.4103754), (0.9063, -0.5264392),
    (0.8594, -0.4264545), (0.8047, -0.3202137), (0.6800, -0.1730887),
    (0.5000, 0.0257995), (0.3500, 0.1882246), (0.2344, 0.3253592),
    (0.2266, 0.3339924), (0.1563, 0.3769189), (0.0938, 0.3330442),
    (0.0781, 0.3099097), (0.0703, 0.2962703), (0.0625, 0.2807056),
    (0.0400, 0.2189314), (0.0300, 0.1796062), (0.0200, 0.1306210),
    (0.0100, 0.0709203), (0.0000, 0.0000000),
]
BP_EXTREMA = {
    "u_min": -0.3885698, "u_min_y": 0.1716968,
    "v_max": 0.3769447, "v_max_x": 0.1578365,
    "v_min": -0.5270773, "v_min_x": 0.9092470,
}
BP_PRIMARY_VORTEX = {
    "psi": -0.118936603, "omega": -2.0677479, "x": 0.53079011, "y": 0.56524055,
}

# ---------------------------------------------------------------------------
# Re=400 cavity reference digits. Table I (u along the vertical centerline
# x=0.5) and Table II (v along the horizontal centerline y=0.5) of the
# standard multigrid reference solution, J. Comput. Phys. 48(3):387-411 (1982),
# doi:10.1016/0021-9991(82)90058-4.
# ---------------------------------------------------------------------------
CAVITY_REFERENCE400_U_CENTERLINE = [   # (y, u) at x=0.5, Re=400
    (1.0000, 1.00000), (0.9766, 0.75837), (0.9688, 0.68439),
    (0.9609, 0.61756), (0.9531, 0.55892), (0.8516, 0.29093),
    (0.7344, 0.16256), (0.6172, 0.02135), (0.5000, -0.11477),
    (0.4531, -0.17119), (0.2813, -0.32726), (0.1719, -0.24299),
    (0.1016, -0.14612), (0.0703, -0.10338), (0.0625, -0.09266),
    (0.0547, -0.08186), (0.0000, 0.00000),
]
CAVITY_REFERENCE400_V_CENTERLINE = [   # (x, v) at y=0.5, Re=400
    (1.00000, 0.00000), (0.9688, -0.12146), (0.9609, -0.15663),
    (0.9531, -0.19254), (0.9453, -0.22847), (0.9063, -0.23827),
    (0.8594, -0.44993), (0.8047, -0.38598), (0.5000, 0.05186),
    (0.2344, 0.30174), (0.2266, 0.30203), (0.1563, 0.28124),
    (0.0938, 0.22965), (0.0781, 0.20920), (0.0703, 0.19713),
    (0.0625, 0.18360), (0.0000, 0.00000),
]
CAVITY_REFERENCE400_EXTREMA = {   # extrema read directly off the tabulated Re=400 columns
    "u_min": -0.32726, "u_min_y": 0.2813,
    "v_max": 0.30203, "v_max_x": 0.2266,
    "v_min": -0.44993, "v_min_x": 0.8594,
}
CAVITY_REFERENCE400_PRIMARY_VORTEX = None   # the reference psi/omega vortex-center table not extracted here

BENCHMARKS = {
    "bp1000": {
        "u": BP_U_CENTERLINE, "v": BP_V_CENTERLINE, "extrema": BP_EXTREMA,
        "vortex": BP_PRIMARY_VORTEX,
        "label": "Re=1000 spectral benchmark",
        "citation": "Benchmark spectral results on the lid-driven cavity flow, "
                    "Computers & Fluids 27(4):421-433 (1998).",
        "source": "DOI 10.1016/S0045-7930(98)00002-4",
    },
    "cavity_reference400": {
        "u": CAVITY_REFERENCE400_U_CENTERLINE, "v": CAVITY_REFERENCE400_V_CENTERLINE, "extrema": CAVITY_REFERENCE400_EXTREMA,
        "vortex": CAVITY_REFERENCE400_PRIMARY_VORTEX,
        "label": "cavity reference tables, Re=400",
        "citation": "High-Re solutions for incompressible "
                    "flow using the Navier-Stokes equations and a multigrid method, "
                    "J. Comput. Phys. 48(3):387-411 (1982).",
        "source": "DOI 10.1016/0021-9991(82)90058-4, Re=400 columns",
    },
}


def read_csv_xy(path, xcol, ycol):
    xs, ys = [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row[xcol])); ys.append(float(row[ycol]))
    return xs, ys


def interp(xq, xs, ys):
    """piecewise-linear interpolation, xs assumed sorted ascending"""
    n = len(xs)
    if xq <= xs[0]: return ys[0]
    if xq >= xs[-1]: return ys[-1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= xq: lo = mid
        else: hi = mid
    t = (xq - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] * (1 - t) + ys[hi] * t


def write_budgets_csv(prefixes, data_dir, results_dir):
    """Merge one or more runs' raw budget histories into the deliverable
    code/results/cavity_budgets.csv, tagged by grid (outprefix)."""
    out_path = results_dir / "cavity_budgets.csv"
    fieldnames = None
    with open(out_path, "w", newline="") as out:
        wtr = None
        for prefix in prefixes:
            src = data_dir / f"{prefix}_budgets_raw.csv"
            if not src.exists():
                continue
            with open(src) as f:
                r = csv.DictReader(f)
                for row in r:
                    if wtr is None:
                        fieldnames = ["grid"] + list(row.keys())
                        wtr = csv.DictWriter(out, fieldnames=fieldnames)
                        wtr.writeheader()
                    row2 = {"grid": prefix}
                    row2.update(row)
                    wtr.writerow(row2)
    print(f"wrote {out_path}")
    return out_path


def main():
    # usage: cavity_postprocess.py <prefix> <benchmark> [data_dir] [results_dir] [extra_prefixes]
    prefix = sys.argv[1] if len(sys.argv) > 1 else "cav400_N32"
    bench_key = sys.argv[2] if len(sys.argv) > 2 else "cavity_reference400"
    data_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("cavity/data")
    results_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("results")
    extra_prefixes = sys.argv[5].split(",") if len(sys.argv) > 5 and sys.argv[5] else []
    results_dir.mkdir(parents=True, exist_ok=True)

    bench = BENCHMARKS[bench_key]
    ref_u, ref_v, ref_ex = bench["u"], bench["v"], bench["extrema"]
    ref_col = bench_key   # column name in the centerline CSV

    write_budgets_csv([prefix] + extra_prefixes, data_dir, results_dir)

    zc, uc = read_csv_xy(data_dir / f"{prefix}_centerline_u.csv", "z", "u")
    xc, wc = read_csv_xy(data_dir / f"{prefix}_centerline_w.csv", "x", "w")
    with open(data_dir / f"{prefix}_run_summary.json") as f:
        run_summary = json.load(f)

    zu = sorted(zip(zc, uc)); zc_s = [p[0] for p in zu]; uc_s = [p[1] for p in zu]
    xw = sorted(zip(xc, wc)); xc_s = [p[0] for p in xw]; wc_s = [p[1] for p in xw]

    # ---- centerline CSV: our profile sampled at the reference tabulated points ----
    rows = []
    for y_r, u_r in ref_u:
        u_ours = interp(y_r, zc_s, uc_s)
        rows.append({"profile": "u(0.5,z)", "coord": y_r,
                     "ours": u_ours, ref_col: u_r, "delta": u_ours - u_r})
    for x_r, v_r in ref_v:
        v_ours = interp(x_r, xc_s, wc_s)
        rows.append({"profile": "w(x,0.5)", "coord": x_r,
                     "ours": v_ours, ref_col: v_r, "delta": v_ours - v_r})
    csv_path = results_dir / "cavity_centerlines.csv"
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["profile", "coord", "ours", ref_col, "delta"])
        wtr.writeheader()
        for row in rows: wtr.writerow(row)

    # ---- extrema. u_min is a clean interior min. v-extrema exclude a wall margin
    # (the outer cells carry the corner-singularity spike, not the physical vortex
    # extremum the reference tabulates at x~0.16-0.23 and x~0.86-0.91). ----
    margin = 0.04
    umin = min(uc_s); umin_z = zc_s[uc_s.index(umin)]
    vin = [(x, v) for x, v in zip(xc_s, wc_s) if margin < x < 1.0 - margin]
    wmin = min(v for _, v in vin); wmin_x = [x for x, v in vin if v == wmin][0]
    wmax = max(v for _, v in vin); wmax_x = [x for x, v in vin if v == wmax][0]

    def relerr(a, b):
        return (a - b) / abs(b) if b != 0 else float("nan")

    summary = {
        "run": run_summary,
        "benchmark": bench["label"],
        "extrema_ours": {"u_min": umin, "u_min_z": umin_z,
                         "v_max": wmax, "v_max_x": wmax_x,
                         "v_min": wmin, "v_min_x": wmin_x},
        "extrema_reference": ref_ex,
        "extrema_delta": {
            "u_min": umin - ref_ex["u_min"], "u_min_z": umin_z - ref_ex["u_min_y"],
            "v_max": wmax - ref_ex["v_max"], "v_max_x": wmax_x - ref_ex["v_max_x"],
            "v_min": wmin - ref_ex["v_min"], "v_min_x": wmin_x - ref_ex["v_min_x"],
        },
        "extrema_relerr": {
            "u_min": relerr(umin, ref_ex["u_min"]),
            "v_max": relerr(wmax, ref_ex["v_max"]),
            "v_min": relerr(wmin, ref_ex["v_min"]),
        },
        "primary_vortex_reference": bench["vortex"],
        "centerline_csv": str(csv_path),
        "reference_citation": bench["citation"],
        "reference_source_verified": bench["source"],
    }
    # fold in the head-to-head KE-production result if present (the capability win)
    keprod = {}
    for g in ("N32", "N48"):
        kp = data_dir / f"ke_production_{g}.json"
        if kp.exists():
            with open(kp) as f:
                keprod[g] = json.load(f)
    if keprod:
        summary["ke_production_head_to_head"] = {
            "note": "advection KE production Pi_adv = sum_i u_i^T <N_i, adv_i(u)> on a "
                    "divergent wall-vanishing field; conservative BMG skew vs non-conservative "
                    "convective, identical grid/operators. skew ~ round-off (any field, "
                    "skew-symmetric cancellation); conv nonzero whenever div u != 0.",
            "by_grid": keprod,
        }
    json_path = results_dir / "cavity_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"\n benchmark: {bench['label']}")
    print(f" u_min: ours={umin:.5f} (z={umin_z:.4f})  ref={ref_ex['u_min']:.5f} (y={ref_ex['u_min_y']:.4f})"
          f"  rel={relerr(umin, ref_ex['u_min']):+.2%}")
    print(f" v_max: ours={wmax:.5f} (x={wmax_x:.4f})  ref={ref_ex['v_max']:.5f} (x={ref_ex['v_max_x']:.4f})"
          f"  rel={relerr(wmax, ref_ex['v_max']):+.2%}")
    print(f" v_min: ours={wmin:.5f} (x={wmin_x:.4f})  ref={ref_ex['v_min']:.5f} (x={ref_ex['v_min_x']:.4f})"
          f"  rel={relerr(wmin, ref_ex['v_min']):+.2%}")


if __name__ == "__main__":
    main()
