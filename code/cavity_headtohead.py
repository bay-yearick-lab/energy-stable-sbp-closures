#!/usr/bin/env python3
"""
Head-to-head accuracy verdict for the lid-driven cavity: does the non-conservative
convective competitor DRIFT from the reference Re=400 benchmark while our
conservative BMG stays on it, or do both match equally well? Reads the two runs'
centerline profiles + summaries and reports the accuracy gap on the tabulated
extrema, plus the KE-production budget difference. Writes
code/results/cavity_headtohead.json.

Usage: uv run --with numpy python cavity_headtohead.py \
           <skew_prefix> <conv_prefix> <benchmark> [data_dir] [results_dir]
"""
import sys
import json
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cavity_postprocess import BENCHMARKS, read_csv_xy, interp


def extrema(prefix, data_dir, ref_ex, margin=0.04):
    zc, uc = read_csv_xy(data_dir / f"{prefix}_centerline_u.csv", "z", "u")
    xc, wc = read_csv_xy(data_dir / f"{prefix}_centerline_w.csv", "x", "w")
    zu = sorted(zip(zc, uc)); zc_s = [p[0] for p in zu]; uc_s = [p[1] for p in zu]
    xw = sorted(zip(xc, wc)); xc_s = [p[0] for p in xw]; wc_s = [p[1] for p in xw]
    umin = min(uc_s); umin_z = zc_s[uc_s.index(umin)]
    vin = [(x, v) for x, v in zip(xc_s, wc_s) if margin < x < 1.0 - margin]
    wmin = min(v for _, v in vin); wmin_x = [x for x, v in vin if v == wmin][0]
    wmax = max(v for _, v in vin); wmax_x = [x for x, v in vin if v == wmax][0]

    def rel(a, b):
        return (a - b) / abs(b) if b else float("nan")
    return {
        "u_min": umin, "u_min_z": umin_z, "u_min_relerr": rel(umin, ref_ex["u_min"]),
        "v_max": wmax, "v_max_x": wmax_x, "v_max_relerr": rel(wmax, ref_ex["v_max"]),
        "v_min": wmin, "v_min_x": wmin_x, "v_min_relerr": rel(wmin, ref_ex["v_min"]),
    }


def load_summary(prefix, data_dir):
    with open(data_dir / f"{prefix}_run_summary.json") as f:
        return json.load(f)


def main():
    skew = sys.argv[1] if len(sys.argv) > 1 else "cav400_N32"
    conv = sys.argv[2] if len(sys.argv) > 2 else "cav400_N32_conv"
    bench_key = sys.argv[3] if len(sys.argv) > 3 else "cavity_reference400"
    data_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("cavity/data")
    results_dir = Path(sys.argv[5]) if len(sys.argv) > 5 else Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    bench = BENCHMARKS[bench_key]
    ref = bench["extrema"]

    ex_skew = extrema(skew, data_dir, ref)
    ex_conv = extrema(conv, data_dir, ref)
    s_skew = load_summary(skew, data_dir)
    s_conv = load_summary(conv, data_dir)

    # accuracy gap: |competitor error| - |our error| on each extremum (positive =>
    # competitor is worse against reference)
    def gap(key):
        return abs(ex_conv[key + "_relerr"]) - abs(ex_skew[key + "_relerr"])
    gaps = {k: gap(k) for k in ("u_min", "v_max", "v_min")}
    worst_gap = max(gaps.values())

    verdict = ("competitor drifts from reference (spurious energy has a measurable accuracy "
               "consequence)" if worst_gap > 0.02 else
               "competitor matches reference about as well (spurious energy stays a budget "
               "diagnostic with no functional accuracy consequence at this Re)")

    out = {
        "benchmark": bench["label"],
        "reference_extrema": ref,
        "ours_BMG_skew": {"extrema": ex_skew, "final_ke_production": s_skew.get("final_ke_production"),
                          "advection": s_skew.get("advection"), "steady_t": s_skew.get("T"),
                          "final_max_dudt": s_skew.get("final_max_dudt")},
        "competitor_convective": {"extrema": ex_conv, "final_ke_production": s_conv.get("final_ke_production"),
                                  "advection": s_conv.get("advection"), "steady_t": s_conv.get("T"),
                                  "final_max_dudt": s_conv.get("final_max_dudt")},
        "accuracy_gap_relerr_conv_minus_skew": gaps,
        "worst_accuracy_gap": worst_gap,
        "verdict": verdict,
    }
    outpath = results_dir / "cavity_headtohead.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)

    print(f"wrote {outpath}\n")
    print(f" benchmark: {bench['label']}")
    print(f" {'quantity':8} {'reference':>10} {'ours(skew)':>12} {'relerr':>9} {'conv':>12} {'relerr':>9}")
    for k, refk in (("u_min", "u_min"), ("v_max", "v_max"), ("v_min", "v_min")):
        print(f" {k:8} {ref[refk]:>10.5f} {ex_skew[k]:>12.5f} {ex_skew[k+'_relerr']:>+8.2%} "
              f"{ex_conv[k]:>12.5f} {ex_conv[k+'_relerr']:>+8.2%}")
    print(f"\n final KE production: ours(skew)={s_skew.get('final_ke_production'):+.3e}  "
          f"conv={s_conv.get('final_ke_production'):+.3e}")
    print(f" worst accuracy gap (|conv err| - |skew err|) = {worst_gap:+.2%}")
    print(f" VERDICT: {verdict}")


if __name__ == "__main__":
    main()
