#!/usr/bin/env python3
"""External validation: wall-resolution sweep of the
turbulent-channel Nusselt number with batch-means uncertainty, run in parallel
across (scheme, n_z) combinations, plus an independent-discretization
(Chebyshev-collocation) reference at high n_z.

Schemes: I (spline, boundary_spline_analysis BoundaryBMG), C (compact FD,
second derivative composed as D1 twice -- as currently built), SBP (SBP-SAT).
n_z in {17, 25, 33, 49, 65}. Reference: CHEB at n_z=97 (dense Chebyshev
collocation, independent of the spline machinery under test).

Parallelized across combos with multiprocessing (fork), one process per combo,
capped to avoid oversubscription (BSPLINE_SWEEP_WORKERS env override; default conservative cap).

Usage:
  python nusselt_sweep.py                # full sweep, background-friendly
  python nusselt_sweep.py --dry-run      # tiny nz list, sanity only
"""
from __future__ import annotations
import sys, os, time, json, argparse
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import numpy as np
import multiprocessing as mp

from nusselt_driver import (load_meta, load_snap, interp_z, nusselt_run, SNAPDIR,
                            warm_theta0, stationarity_gate)
from scalar3d import ScalarChannel3D

try:
    from project_paths import RESULTSDIR, ensure_generated_dirs
    ensure_generated_dirs()
    RESULTS = str(RESULTSDIR)
except Exception:
    RESULTS = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(RESULTS, exist_ok=True)

OUT_PATH = os.path.join(RESULTS, "nusselt_sweep.json")
LOG_PATH = os.path.join(RESULTS, "nusselt_sweep.log")

N_BATCHES = 8   # batch-means CI: split the post-spinup window into this many batches


def batch_means_ci(series: np.ndarray, n_batches: int = N_BATCHES):
    """Split a Nu(t) time series (post-spinup samples) into n_batches contiguous
    batches, average each, and report mean +/- 95% CI (t-distribution) of the
    batch means. Returns (mean, half_width_95, n_batches_used, batch_means)."""
    n = len(series)
    if n < 2 * n_batches:
        n_batches = max(2, n // 4) if n >= 8 else 1
    if n_batches <= 1:
        m = float(np.mean(series)) if n else float("nan")
        return m, float("nan"), 1, [m]
    edges = np.linspace(0, n, n_batches + 1).astype(int)
    means = [float(series[edges[i]:edges[i + 1]].mean()) for i in range(n_batches)]
    means = np.array(means)
    m = float(means.mean())
    s = float(means.std(ddof=1)) if n_batches > 1 else 0.0
    # Student-t 95% two-sided critical values, small-sample table (df = n_batches-1)
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}.get(n_batches - 1, 1.96)
    half = tcrit * s / np.sqrt(n_batches)
    return m, float(half), n_batches, means.tolist()


def _load_snapshots(nsnap, m):
    return [load_snap(k, m) for k in range(nsnap)]


def run_combo(args):
    kind, nz, meta_dict, raw_path = args
    # each worker reloads snapshots from disk (fork-shared memory is not
    # guaranteed once numpy arrays are touched by the parent gc; reloading is
    # cheap -- ~200 files of 2.3MB each -- and keeps workers independent).
    m = meta_dict
    Nx, Ny, Lx, Ly, Lz = m["Nx"], m["Ny"], m["Lx"], m["Ly"], m["Lz"]
    zuni = np.array(m["zuni"])
    Pr = 0.71
    nu = 1.0 / 180.0
    alpha = nu / Pr
    dt = m["dt_snap"] / m["stride"]
    dt_snap = m["dt_snap"]
    nsnap = m["nsnap_actual"]

    t0 = time.time()
    raw = _load_snapshots(nsnap, m)

    order = 6
    S = ScalarChannel3D(kind, order, Nx, Ny, nz, Lx, Ly, Lz, alpha)
    z = S.z
    U = [interp_z(raw[k][0], zuni, z) for k in range(nsnap)]
    V = [interp_z(raw[k][1], zuni, z) for k in range(nsnap)]
    W = [interp_z(raw[k][2], zuni, z) for k in range(nsnap)]

    def vel_at(t, _z, U=U, V=V, W=W):
        s = t / dt_snap
        k = min(int(s), nsnap - 2)
        f = s - k
        return ((1 - f) * U[k] + f * U[k + 1],
                (1 - f) * V[k] + f * V[k + 1],
                (1 - f) * W[k] + f * W[k + 1])

    T = dt_snap * (nsnap - 1)
    nstep = int(round(T / dt))
    # task-3 stationarity fix (wired in by controller after the driver functions
    # were added but never connected): warm-start the scalar from the developed
    # mean profile to remove the cold-start transient, integrate with NO fixed
    # spinup so the full Nu(t) series is returned, then let the stationarity gate
    # select the earliest window whose batch-mean trend is statistically flat.
    # Nu and its CI are computed over that stationary window only.
    theta0 = warm_theta0(z, Lz, Nx, Ny)
    Nu_full, zg, prof, series = nusselt_run(kind, vel_at, Nx, Ny, len(z), Lx, Ly, Lz, alpha,
                                            dt, T=T, spinup=0.0, order=order,
                                            return_series=True, theta0=theta0)
    gate = stationarity_gate(series)
    if gate["passed"]:
        i0 = int(gate["i0"])
    else:
        i0 = len(series) // 2   # no flat window found: last half, flagged untrusted
    stat = series[i0:]
    Nu = float(np.mean(stat))
    mean, half, nb, batch_means = batch_means_ci(stat)
    elapsed = time.time() - t0
    flow_through = T / (Lz)          # Lz/u_bulk ~ O(1) in these units; report T in Lz-units too
    return {
        "kind": kind, "nz": nz, "Nu": Nu, "Nu_mean_batches": mean,
        "ci95": half, "n_batches": nb, "batch_means": batch_means,
        "n_samples": int(len(stat)), "T": float(T), "spinup": 0.0,
        "stationary": {
            "passed": bool(gate["passed"]), "start_frac": gate.get("start_frac"),
            "slope_p": gate.get("slope_p"), "i0": int(i0),
            "n_full": int(len(series)),
        },
        "dt": float(dt), "nstep": int(nstep),
        "wall_time_s": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--nz-list", type=str, default="17,25,33,49,65")
    ap.add_argument("--schemes", type=str, default="I,C,SBP",
                     help="Comma-separated scheme labels to sweep. Task-3C full-strength "
                          "competitors (C6-D2, SBP6-BO) are supported via scalar3d.py's "
                          "build_operator_extended dispatch.")
    ap.add_argument("--no-reference", action="store_true",
                     help="Skip the CHEB nz=97 reference combo (already computed).")
    ap.add_argument("--merge-into-existing", action="store_true",
                     help="Load OUT_PATH if present and merge new results/schemes into it "
                          "instead of overwriting from scratch. Existing result entries for "
                          "(kind, nz) combos not being recomputed are preserved byte-for-byte "
                          "in content (only file-level structure is rewritten).")
    args = ap.parse_args()

    m = load_meta()
    nsnap = len([f for f in os.listdir(SNAPDIR) if f.startswith("snap_") and f.endswith(".bin")])
    m["nsnap_actual"] = nsnap
    m["zuni"] = m["zuni"].tolist()

    nz_list = [int(x) for x in args.nz_list.split(",")] if not args.dry_run else [17, 25]
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]

    existing = None
    if args.merge_into_existing and os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            existing = json.load(f)

    combos = [(kind, nz, m, None) for kind in schemes for nz in nz_list]
    if not args.no_reference:
        combos.append(("CHEB", 97, m, None))  # independent-discretization reference

    default_workers = max(1, min(len(combos), (os.cpu_count() or 4) // 2))
    workers = args.workers or int(os.getenv("BSPLINE_SWEEP_WORKERS", default_workers))
    workers = max(1, min(workers, len(combos)))

    print(f"[nusselt_sweep] {len(combos)} combos, {workers} worker processes, "
          f"nsnap={nsnap}, nz_list={nz_list}", flush=True)

    def write_partial(results, total_wall, done):
        if existing is not None:
            # Merge mode: preserve every existing result entry whose (kind, nz)
            # is not among the combos just (re)computed in this invocation, and
            # union the scheme/nz_list metadata. This lets a second, targeted
            # invocation (e.g. new competitor columns) add keys to the JSON
            # without recomputing or disturbing prior cells.
            new_keys = {(kind, nz) for (kind, nz, _, _) in combos}
            kept = [r for r in existing.get("results", [])
                    if (r["kind"], r["nz"]) not in new_keys]
            merged_results = kept + results
            merged_schemes = list(existing.get("schemes", []))
            for s in schemes:
                if s not in merged_schemes:
                    merged_schemes.append(s)
            merged_nz = sorted(set(existing.get("nz_list", [])) | set(nz_list))
            out = dict(existing)
            out.update({
                "mode": "sweep" if done else "sweep-partial",
                "nz_list": merged_nz,
                "schemes": merged_schemes,
                "n_batches_ci": N_BATCHES,
                "nsnap": nsnap,
                "total_wall_time_s": existing.get("total_wall_time_s", 0.0) + total_wall,
                "workers": workers,
                "n_combos_done": len(merged_results),
                "n_combos_total": len(kept) + len(combos),
                "results": merged_results,
            })
            with open(OUT_PATH, "w") as f:
                json.dump(out, f, indent=2)
            return out
        out = {
            "mode": "sweep" if done else "sweep-partial",
            "Pr": 0.71,
            "Re_tau_by_construction": 180.0,
            "nz_list": nz_list,
            "schemes": schemes,
            "reference": {"kind": "CHEB", "nz": 97},
            "n_batches_ci": N_BATCHES,
            "nsnap": nsnap,
            "total_wall_time_s": total_wall,
            "workers": workers,
            "n_combos_done": len(results),
            "n_combos_total": len(combos),
            "results": results,
        }
        with open(OUT_PATH, "w") as f:
            json.dump(out, f, indent=2)
        return out

    results = []
    t_start = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_combo, combos)):
            elapsed = time.time() - t_start
            print(f"[{i+1}/{len(combos)}] {res['kind']:5s} nz={res['nz']:3d}  "
                  f"Nu={res['Nu']:.4f}  ci95={res['ci95']:.4f}  "
                  f"({res['wall_time_s']:.0f}s this combo, {elapsed:.0f}s total)", flush=True)
            results.append(res)
            write_partial(results, elapsed, done=False)   # incremental checkpoint

    total_wall = time.time() - t_start
    write_partial(results, total_wall, done=True)
    print(f"[nusselt_sweep] wrote {OUT_PATH}  (total wall time {total_wall:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
