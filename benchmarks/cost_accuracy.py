#!/usr/bin/env python3
"""Cost-accuracy benchmark: complete heat-cavity march at q = 6.

Each family is marched to T = 2 at its own stable RK4 step (operator assembly
included) on the grid ladder N = 33, 49, 65, 81, timed by single-thread CPU
process time (median of --repeats runs after one warmup). Prints relative L2
error against CPU seconds; with --write, stores the table as JSON.

This is the same protocol as code/s6_wallclock_cost.py, whose committed output
code/results/wallclock_cost.json backs the table in benchmarks/README.md.

Usage:  OMP_NUM_THREADS=1 python benchmarks/cost_accuracy.py [--repeats 3] [--write out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import heat_cavity_core as hc  # noqa: E402
from s6_wallclock_cost import GRIDS, LABELS, hardware_stamp, one_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--write", type=Path, default=None)
    args = ap.parse_args()

    modes = hc.default_heat_modes()
    rows = []
    print(f"{'family':>6} {'N':>4} {'CPU [s]':>9} {'steps':>6} {'rel L2':>10}")
    for npts in GRIDS:
        for label in LABELS:
            one_run(label, npts, modes)
            cpus, met = [], None
            for _ in range(args.repeats):
                _wall, cpu, met = one_run(label, npts, modes)
                cpus.append(cpu)
            row = {"label": label, "npts": npts, "median_cpu_seconds": statistics.median(cpus), **met}
            rows.append(row)
            print(f"{label:>6} {npts:>4} {row['median_cpu_seconds']:9.4f} {row['steps']:>6} {row['rel_l2']:10.2e}",
                  flush=True)

    best = {}
    for row in rows:
        fam = "spline" if row["label"].startswith("BM") else "nodal"
        best[fam] = min(best.get(fam, float("inf")), row["rel_l2"])
    print(f"\nbest spline error {best['spline']:.2e}; best nodal error {best['nodal']:.2e}")
    if args.write:
        args.write.write_text(json.dumps({"hardware": hardware_stamp(), "repeats": args.repeats,
                                          "cases": rows}, indent=1))
        print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
