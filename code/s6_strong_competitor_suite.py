"""Strongest-competitor suite: CSB schemes fielded from source.

CSB denotes the published stable conservative boundary schemes (see
baseline_closures.py). This suite fields both their explicit (E6) and compact
(T6) sixth-order schemes at full published strength, from the authors'
machine-readable coefficient release (provenance in baseline_closures.py), and
runs them through the paper's decisive long-time closed-box test.

Per fielded scheme it records: the assembly gates (polynomial exactness at
the measured design orders; the conservation identity on the correct,
uneliminated form), the closed-box wave-pair spectral abscissa of every
published candidate at two grids, and the long-time cliff error of the
best-vortex stable candidate under the exact cliff protocol.

Usage:
  python code/s6_strong_competitor_suite.py --write code/results/strong_competitor_suite.json
  python code/s6_strong_competitor_suite.py --verify code/results/strong_competitor_suite.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_closures as bl  # noqa: E402
from project_paths import RESULTSDIR  # noqa: E402
from s6_wall_longtime_cliff import p_exact, u_exact  # noqa: E402

NPTS_CLIFF = 41
T_END, CFL = 6.0, 0.01


def wave_pair(D: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    A = np.zeros((2 * n - 2, 2 * n - 2))
    A[:n, n:] = -D[:, 1:-1]
    A[n:, :n] = -D[1:-1, :]
    return A


def cliff_error(D: np.ndarray) -> float:
    npts = D.shape[0]
    A = wave_pair(D)
    x = np.linspace(0.0, 1.0, npts)
    h = x[1] - x[0]
    dt = CFL * h
    nsteps = int(np.ceil(T_END / dt))
    dt = T_END / nsteps
    y = np.concatenate([p_exact(x, 0.0), u_exact(x, 0.0)[1:-1]])
    for _ in range(nsteps):
        k1 = A @ y
        k2 = A @ (y + 0.5 * dt * k1)
        k3 = A @ (y + 0.5 * dt * k2)
        k4 = A @ (y + dt * k3)
        y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    p = y[:npts]
    u = np.concatenate([[u_exact(x[0], T_END)], y[npts:], [u_exact(x[-1], T_END)]])
    pex, uex = p_exact(x, T_END), u_exact(x, T_END)
    return float(np.sqrt(
        (np.linalg.norm(p - pex) ** 2 + np.linalg.norm(u - uex) ** 2)
        / (np.linalg.norm(pex) ** 2 + np.linalg.norm(uex) ** 2)))


def field_scheme(name: str, builder, verify, best_pick) -> dict:
    rec = {"scheme": name, "gates": verify()}
    scan = []
    for cid in range(1, 17):
        row = {"candidate": cid}
        for npts in (41, 129):
            D, _w, _ = builder(npts, candidate=cid)
            row[f"abscissa_N{npts}"] = float(
                np.max(np.linalg.eigvals(wave_pair(D)).real))
        scan.append(row)
    rec["candidate_abscissa_scan"] = scan
    stable = [r["candidate"] for r in scan
              if r["abscissa_N41"] < 1e-10 and r["abscissa_N129"] < 1e-10]
    rec["stable_candidates_both_grids"] = stable
    cid = best_pick(stable)
    D, _w, _ = builder(NPTS_CLIFF, candidate=cid)
    rec["fielded_candidate"] = cid
    rec["cliff_error_t6_transits"] = cliff_error(D)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default=None)
    ap.add_argument("--verify", default=None)
    args = ap.parse_args()
    if args.verify:
        d = json.loads(Path(args.verify).read_text())
        for rec in d["schemes"]:
            g = rec["gates"]
            assert all(v < 1e-8 for k, v in g.items() if k != "candidate"), g
            assert rec["stable_candidates_both_grids"], rec["scheme"]
        print("verify OK:", {r["scheme"]: (r["fielded_candidate"],
              f"{r['cliff_error_t6_transits']:.3e}") for r in d["schemes"]})
        return

    # best-vortex candidate among the both-grid-stable set, per scheme
    def best_e6(stable):
        _c, _r, best = bl._load_candidates(6)
        return best if best in stable else stable[0]

    def best_t6(stable):
        _c, _r, best = bl._load_candidates_t6()
        return best if best in stable else stable[0]

    out = {"description": "CSB E6/T6 fielded at full published "
                          "strength through the long-time cliff",
           "provenance": "doi 10.1016/j.dib.2019.104086 machine-readable "
                         "release; see baseline_closures.py",
           "schemes": [
               field_scheme("CSB-E6", bl.csb_e6_operator, bl.verify_assembly, best_e6),
               field_scheme("CSB-T6", bl.csb_t6_operator, bl.verify_assembly_t6, best_t6),
           ]}
    for r in out["schemes"]:
        print(f"{r['scheme']}: stable candidates {r['stable_candidates_both_grids']}, "
              f"fielded {r['fielded_candidate']}, cliff "
              f"{r['cliff_error_t6_transits']:.3e}", flush=True)
    dest = Path(args.write) if args.write else Path(RESULTSDIR) / "strong_competitor_suite.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
