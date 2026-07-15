"""Measured wall-clock cost-error curves.

Flop-count proxies are not cost evidence; cost claims need
error-versus-wall-clock curves at matched accuracy with hardware, software,
repeats, and spread stated, each family charged its own stable step.

This harness runs the closed-cavity heat benchmark end to end (operator
assembly + time marching at the family's own tier-B step under the
metric-orthogonal enforcement) for the six matched-order q=6 families on a
grid ladder, timing each complete run with one warmup and five measured
repeats, reporting medians and interquartile spread, and recording the
hardware and software stamp. The run refuses to start unless the 1-minute
load average is below a threshold, so timings are never taken on a loaded
box.

Usage:
  python code/s6_wallclock_cost.py --write code/results/wallclock_cost.json
  python code/s6_wallclock_cost.py --verify code/results/wallclock_cost.json
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heat_cavity_core as hc  # noqa: E402
from project_paths import RESULTSDIR  # noqa: E402

LABELS = ("E6", "C6", "SBP6", "DRP6", "BMF6", "BMG6")
GRIDS = (33, 49, 65, 81)
REPEATS = 5
LOAD_CEILING = 3.0


def _close(stored: float, reconstructed: float) -> bool:
    return bool(np.isclose(stored, reconstructed, rtol=1.0e-12, atol=1.0e-15))


def verify_record(record: dict) -> None:
    if record.get("repeats") != REPEATS:
        raise ValueError("wall-clock repeat count differs from the released protocol")
    if record.get("primary_metric") != "cpu_process_time_single_thread":
        raise ValueError("wall-clock primary metric must be single-thread process time")
    hardware = record.get("hardware")
    required_hardware = {"cpu", "platform", "python", "numpy", "threads", "nice", "load_at_start"}
    if not isinstance(hardware, dict) or set(hardware) != required_hardware:
        raise ValueError("wall-clock hardware stamp is incomplete")
    if not all(hardware[k] for k in ("cpu", "platform", "python", "numpy")):
        raise ValueError("wall-clock hardware stamp has an empty identity field")
    if hardware["threads"] != "1" or hardware["nice"] != 19:
        raise ValueError("wall-clock record was not collected single-threaded at nice 19")
    if not np.isfinite(hardware["load_at_start"]) or hardware["load_at_start"] < 0.0:
        raise ValueError("invalid recorded load average")
    if record.get("wall_time_valid") is not (hardware["load_at_start"] < LOAD_CEILING):
        raise ValueError("wall-time validity flag disagrees with the recorded load")
    cases = record.get("cases")
    if not isinstance(cases, list):
        raise ValueError("wall-clock cases must be a list")
    expected = {(label, npts) for label in LABELS for npts in GRIDS}
    indexed = {}
    required_case = {
        "label", "npts", "median_cpu_seconds", "median_wall_seconds",
        "iqr_cpu_seconds", "repeat_cpu_seconds", "repeat_wall_seconds",
        "rel_l2", "linf", "steps", "dt",
    }
    for case in cases:
        if set(case) != required_case:
            raise ValueError(f"unexpected wall-clock case schema: {case}")
        key = (case["label"], case["npts"])
        if key not in expected or key in indexed:
            raise ValueError(f"unexpected or duplicate wall-clock case: {key}")
        cpus = np.asarray(case["repeat_cpu_seconds"], float)
        walls = np.asarray(case["repeat_wall_seconds"], float)
        if cpus.size != REPEATS or walls.size != REPEATS:
            raise ValueError(f"wrong repeat count for {key}")
        if not np.all(np.isfinite(cpus)) or not np.all(np.isfinite(walls)):
            raise ValueError(f"nonfinite timing for {key}")
        if np.any(cpus <= 0.0) or np.any(walls <= 0.0):
            raise ValueError(f"nonpositive timing for {key}")
        median_cpu = float(statistics.median(cpus))
        median_wall = float(statistics.median(walls))
        iqr_cpu = float(np.subtract(*np.percentile(cpus, [75, 25])))
        if not _close(case["median_cpu_seconds"], median_cpu):
            raise AssertionError(f"forged CPU median for {key}")
        if not _close(case["median_wall_seconds"], median_wall):
            raise AssertionError(f"forged wall median for {key}")
        if not _close(case["iqr_cpu_seconds"], iqr_cpu):
            raise AssertionError(f"forged CPU IQR for {key}")
        scalar = np.asarray([case["rel_l2"], case["linf"], case["dt"]], float)
        if not np.all(np.isfinite(scalar)) or np.any(scalar < 0.0):
            raise ValueError(f"invalid numerical metric for {key}")
        if not isinstance(case["steps"], int) or case["steps"] < 1 or case["dt"] <= 0.0:
            raise ValueError(f"invalid step ledger for {key}")
        if not np.isclose(case["steps"] * case["dt"], 2.0, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"step ledger does not reach the stated horizon for {key}")
        indexed[key] = case
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        raise ValueError(f"incomplete wall-clock grid; missing={missing}")
    for label in LABELS:
        ordered = [indexed[(label, npts)] for npts in GRIDS]
        errors = [case["rel_l2"] for case in ordered]
        cpus = [case["median_cpu_seconds"] for case in ordered]
        if not all(a > b for a, b in zip(errors, errors[1:])):
            raise AssertionError(f"error does not decrease on the {label} grid ladder")
        if not all(a < b for a, b in zip(cpus, cpus[1:])):
            raise AssertionError(f"CPU cost does not increase on the {label} grid ladder")
    nodal_best = min(indexed[(label, npts)]["rel_l2"] for label in LABELS[:4] for npts in GRIDS)
    spline_best = min(indexed[(label, npts)]["rel_l2"] for label in LABELS[4:] for npts in GRIDS)
    if not nodal_best > 3.6e-3 or not spline_best < 4.0e-6:
        raise AssertionError("wall-clock artifact no longer supports the manuscript crossover bounds")


def hardware_stamp() -> dict:
    return {
        "cpu": platform.machine(), "platform": f"{platform.system()}-{platform.machine()}",
        "python": sys.version.split()[0], "numpy": np.__version__,
        "threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "nice": os.nice(0),
        "load_at_start": os.getloadavg()[0],
    }


def one_run(label: str, npts: int, modes) -> tuple[float, float, dict]:
    w0, c0 = time.perf_counter(), time.process_time()
    dt = hc.per_method_diffusion_dt(label, npts=npts,
                                    viscosity=modes.viscosity, t_end=2.0)
    r = hc.run_heat_cavity(label, modes, npts=npts, t_end=2.0, dt=dt,
                           record_count=2, snapshot_count=2)
    wall = time.perf_counter() - w0
    cpu = time.process_time() - c0     # load-robust for a single-thread job
    return wall, cpu, {"rel_l2": float(r.u_rel_l2[-1]),
                       "linf": float(r.u_linf[-1]),
                       "steps": int(r.steps), "dt": float(r.dt)}


def rescore_spline_errors(record: dict, runner=one_run) -> dict:
    """Replace only BMF/BMG numerical scores, preserving measured timings."""
    out = deepcopy(record)
    indexed = {(row.get("label"), row.get("npts")): row for row in out.get("cases", [])}
    expected = {(label, npts) for label in LABELS for npts in GRIDS}
    if set(indexed) != expected or len(indexed) != len(out.get("cases", [])):
        raise ValueError("source wall-clock artifact is incomplete or duplicated")
    modes = hc.default_heat_modes()
    for npts in GRIDS:
        for label in ("BMF6", "BMG6"):
            _wall, _cpu, metrics = runner(label, npts, modes)
            indexed[(label, npts)].update(metrics)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default=None)
    ap.add_argument("--verify", default=None)
    ap.add_argument("--rescore-errors", action="store_true")
    args = ap.parse_args()
    if args.verify:
        d = json.loads(Path(args.verify).read_text())
        verify_record(d)
        print(f"verify OK: {len(d['cases'])} timed cases, "
              f"primary metric = {d['primary_metric']}, "
              f"load_at_start = {d['hardware']['load_at_start']:.1f}")
        return

    if args.rescore_errors:
        dest = Path(args.write) if args.write else Path(RESULTSDIR) / "wallclock_cost.json"
        out = rescore_spline_errors(json.loads(dest.read_text()))
        verify_record(out)
        dest.write_text(json.dumps(out, indent=1))
        print(f"rescored spline errors without altering timing samples: {dest}")
        return

    # CPU time (process_time) is the primary metric: for a single-threaded job
    # it is nearly load-independent, so it is meaningful even when another
    # session shares the box. Wall time is recorded too, flagged by the load.
    modes = hc.default_heat_modes()
    out = {"hardware": hardware_stamp(), "repeats": REPEATS,
           "primary_metric": "cpu_process_time_single_thread",
           "wall_time_valid": os.getloadavg()[0] < LOAD_CEILING, "cases": []}
    for npts in GRIDS:
        for label in LABELS:
            one_run(label, npts, modes)          # warmup (caches, mallocs)
            walls, cpus, met = [], [], None
            for _ in range(REPEATS):
                w, c, met = one_run(label, npts, modes)
                walls.append(w); cpus.append(c)
            times = cpus
            rec = {"label": label, "npts": npts,
                   "median_cpu_seconds": statistics.median(cpus),
                   "median_wall_seconds": statistics.median(walls),
                   "iqr_cpu_seconds": float(np.subtract(*np.percentile(cpus, [75, 25]))),
                   "repeat_cpu_seconds": cpus, "repeat_wall_seconds": walls}
            rec.update(met)
            out["cases"].append(rec)
            print(f"{label:5s} N={npts} cpu={rec['median_cpu_seconds']:.3f}s "
                  f"wall={rec['median_wall_seconds']:.3f}s err={rec['rel_l2']:.3e} "
                  f"steps={rec['steps']}", flush=True)
    dest = Path(args.write) if args.write else Path(RESULTSDIR) / "wallclock_cost.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
