"""Static fairness table: conditioning, bandwidth, and solve counts of every
fielded comparator at the grids Section 6 of `paper/main.tex` uses.

The decision quantity is, per family and per grid,

  * `cond_2` of the family's left-side (solve) matrix, that is, the matrix that
    must be inverted to apply a derivative;
  * `cond_2` of the assembled first- and second-derivative matrices;
  * the bandwidth of every banded factor in the application path;
  * the number of banded solves per derivative application;
  * the nnz of every recorded matrix.

Every operator is taken from the repository's own constructors
(`boundary_spline_analysis`, `baseline_closures`, `sem_operator`). Nothing is
re-derived here. A family whose banded factors are not exposed by any
constructor is recorded with a null condition number and an explicit note
rather than a reconstructed matrix.

Usage:
  python code/s6_comparator_conditioning.py \
      --out code/results/comparator_conditioning.json \
      --tex code/results/comparator_conditioning_table.tex
  python code/s6_comparator_conditioning.py \
      --verify code/results/comparator_conditioning.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import boundary_spline_analysis as bsa  # noqa: E402
import baseline_closures as bl  # noqa: E402
from sem_operator import sem_grid_for  # noqa: E402

Array = np.ndarray

# --- grids ------------------------------------------------------------------
# {41, 81, 161} as instructed, plus the grids Section 6 actually uses:
# closed-box acoustics N=41, analytic-wake 25x25, localized packet N in {49,97,193},
# heat cavity 81x81 (ladder contains 49 and 97).
NPTS_SWEEP = (25, 41, 49, 81, 97, 161, 193)
NPTS_TABLE = 81              # main table grid
NPTS_COMPANION = (41, 81, 161)  # N-independence companion
SEM_DEGREE = 6               # the paper's scalar spectral-element control
SEM_LADDER = (19, 37, 43, 55, 73, 91)  # paper's own scalar-control grids
SEM_TABLE_NPTS = 73          # nearest paper SEM grid to NPTS_TABLE
SEM_COMPANION = (19, 43, 91)   # paper scalar-control grids
BAND_TOL = 1.0e-13
RANK_TOL = 1.0e-10           # relative singular-value floor for the kernel count
SPREAD_TOL = 0.01            # 1 percent, per the contract
SLOPE_TOL = 0.05             # d log cond / d log N, per the contract


# --- matrix statistics ------------------------------------------------------
def matrix_stats(name: str, mat: Array, role: str = "") -> dict:
    """cond_2, bandwidth, nnz and shape of one matrix."""
    a = np.asarray(mat, dtype=float)
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    thr = BAND_TOL * peak
    nz = np.abs(a) > thr
    idx = np.argwhere(nz)
    if idx.size:
        diff = idx[:, 0] - idx[:, 1]
        lower = int(diff.max())
        upper = int((-diff).max())
    else:
        lower = upper = 0
    svals = np.linalg.svd(a, compute_uv=False)
    if a.shape[0] == a.shape[1]:
        cond = float(np.linalg.cond(a, 2))
    else:
        cond = float(svals[0] / svals[-1])
    # Every consistent derivative annihilates constants, so the raw cond_2 of an
    # assembled square D1 or D2 is the condition number of a singular matrix.
    # cond_2_effective is the same ratio restricted to the complement of the
    # numerical null space; kernel_dim reports how many directions were dropped.
    keep = svals[svals > RANK_TOL * svals[0]] if svals[0] > 0.0 else svals
    cond_eff = float(svals[0] / keep[-1]) if keep.size else float("inf")
    return {
        "name": name,
        "role": role,
        "shape": [int(a.shape[0]), int(a.shape[1])],
        "cond_2": cond,
        "cond_2_effective": cond_eff,
        "norm_2": float(svals[0]),
        "numerical_rank": int(keep.size),
        "kernel_dim": int(svals.size - keep.size),
        "bandwidth_lower": lower,
        "bandwidth_upper": upper,
        "bandwidth": int(max(lower, upper)),
        "nnz": int(nz.sum()),
        "density": float(nz.sum()) / float(a.size),
    }


def absent(name: str, note: str) -> dict:
    """Record a factor the repository does not expose. Nothing is invented."""
    return {
        "name": name,
        "role": "left side",
        "shape": None,
        "cond_2": None,
        "cond_2_effective": None,
        "norm_2": None,
        "numerical_rank": None,
        "kernel_dim": None,
        "bandwidth_lower": None,
        "bandwidth_upper": None,
        "bandwidth": None,
        "nnz": None,
        "density": None,
        "missing_constructor": note,
    }


# --- per-family builders ----------------------------------------------------
def build_explicit(order: int, npts: int) -> dict:
    _x, dmat = bsa.explicit_fd_matrix(npts, order)
    return {
        "kind": "explicit finite difference",
        "left_sides": [],
        "left_side_label": "none (explicit)",
        "d1": matrix_stats("D", dmat, "first derivative"),
        "d2": matrix_stats("D D", dmat @ dmat, "second derivative"),
        "solves_per_d1": 0,
        "solves_per_d2": 0,
        "factored_d1": "D1 = D",
        "factored_d2": "D2 = D D",
    }


def build_drp(order: int, npts: int) -> dict:
    _x, dmat = bsa.drp_fd_matrix(npts, order)
    rec = build_explicit(order, npts)
    rec["kind"] = "dispersion-relation-preserving finite difference"
    rec["d1"] = matrix_stats("D", dmat, "first derivative")
    rec["d2"] = matrix_stats("D D", dmat @ dmat, "second derivative")
    return rec


def build_compact(order: int, npts: int) -> dict:
    _x, amat, bmat = bsa.compact_fd_matrices(npts, order)
    d1 = np.linalg.solve(amat, bmat)
    return {
        "kind": "compact (Pade) finite difference",
        "left_sides": [matrix_stats("A", amat, "left side")],
        "left_side_label": "A (tridiagonal)",
        "banded_factors": [matrix_stats("A", amat, "banded factor"),
                           matrix_stats("B", bmat, "banded factor")],
        "d1": matrix_stats("A^{-1} B", d1, "first derivative"),
        "d2": matrix_stats("(A^{-1} B)^2", d1 @ d1, "second derivative"),
        "solves_per_d1": 1,
        "solves_per_d2": 2,
        "factored_d1": "A D1 = B",
        "factored_d2": "D2 = D1 D1 (two tridiagonal solves)",
    }


def build_compact_baseline_c6_d2(npts: int) -> dict:
    _x, a1, b1 = bsa.compact_fd_matrices(npts, 6)
    d1 = np.linalg.solve(a1, b1)
    _x2, a2, b2 = bsa.compact_baseline_c6_d2_matrices(npts)
    d2 = bsa.compact_baseline_c6_d2_operator_matrix(npts)
    return {
        "kind": "compact first derivative with the published tailored compact D2",
        "left_sides": [matrix_stats("A", a1, "left side, first derivative"),
                       matrix_stats("A_{D2}", a2, "left side, second derivative")],
        "left_side_label": "A (tridiagonal)",
        "banded_factors": [matrix_stats("A", a1, "banded factor"),
                           matrix_stats("B", b1, "banded factor"),
                           matrix_stats("A_{D2}", a2, "banded factor"),
                           matrix_stats("B_{D2}", b2, "banded factor")],
        "d1": matrix_stats("A^{-1} B", d1, "first derivative"),
        "d2": matrix_stats("A_{D2}^{-1} B_{D2}", d2, "second derivative"),
        "solves_per_d1": 1,
        "solves_per_d2": 1,
        "factored_d1": "A D1 = B",
        "factored_d2": "A_{D2} D2 = B_{D2}",
    }


def build_sbp(order: int, npts: int) -> dict:
    _x, dmat, hmat = bsa.sbp_fd_matrix(npts, order)
    _x2, d2 = bsa.sbp_second_derivative_matrix(npts, order)
    return {
        "kind": "classical diagonal-norm summation-by-parts",
        "left_sides": [matrix_stats("H", hmat, "left side")],
        "left_side_label": "H (diagonal)",
        "banded_factors": [matrix_stats("H", hmat, "banded factor"),
                           matrix_stats("D", dmat, "banded factor")],
        "d1": matrix_stats("D", dmat, "first derivative"),
        "d2": matrix_stats("H^{-1}(-D^T H D + E)", d2, "second derivative"),
        "solves_per_d1": 0,
        "solves_per_d2": 1,
        "factored_d1": "D1 = D",
        "factored_d2": "D2 = H^{-1}(-D^T H D - e_0 s_0 + e_N s_N); H diagonal",
    }


def build_sbp6_bo(npts: int) -> dict:
    _x, dmat, hmat = bsa.sbp6_bo_fd_matrix(npts)
    _x2, d2 = bsa.sbp6_bo_second_derivative_matrix(npts)
    rec = build_sbp(6, npts)
    rec["kind"] = "boundary-optimized non-equispaced summation-by-parts"
    rec["left_sides"] = [matrix_stats("H", hmat, "left side")]
    rec["banded_factors"] = [matrix_stats("H", hmat, "banded factor"),
                             matrix_stats("D", dmat, "banded factor")]
    rec["d1"] = matrix_stats("D", dmat, "first derivative")
    rec["d2"] = matrix_stats("H^{-1}(-D^T H D + E)", d2, "second derivative")
    return rec


def build_bmg(degree: int, npts: int) -> dict:
    _x, _knots, bmat, _gmat = bsa.clamped_bspline_matrices(npts, degree)
    mass, adv = bsa.bspline_mass_and_advection(npts, degree)
    _x1, d1, hmat = bsa.bmg_sbp_first_derivative_matrix(npts, degree)
    _x2, d2 = bsa.bmg_sbp_second_derivative_matrix(npts, degree)
    return {
        "kind": "same-grid Galerkin derivative in the natural spline mass metric",
        "left_sides": [matrix_stats("B", bmat, "left side (sample)"),
                       matrix_stats("M", mass, "left side (mass)"),
                       matrix_stats("H_{BMG}", hmat, "induced metric B^{-T} M B^{-1}")],
        "left_side_label": "B (sample), M (mass)",
        "banded_factors": [matrix_stats("B", bmat, "banded factor"),
                           matrix_stats("M", mass, "banded factor"),
                           matrix_stats("A", adv, "banded factor")],
        "d1": matrix_stats("B M^{-1} A B^{-1}", d1, "first derivative"),
        "d2": matrix_stats("H^{-1}(-D^T H D + E)", d2, "second derivative"),
        "solves_per_d1": 2,
        "solves_per_d2": 3,
        "factored_d1": "D1 = B M^{-1} A B^{-1} (solves: B, M)",
        "factored_d2": "D2 = -B M^{-1} A^T M^{-1} A B^{-1} + boundary lift (solves: B, M, M)",
    }


def build_bmf(degree: int, npts: int) -> dict:
    ops = bsa.bmf_staggered_operators(npts, degree)
    _xs, d1_same = bsa.bmf_sbp_first_derivative_matrix(npts, degree)
    _x2, d2 = bsa.bmf_sbp_second_derivative_matrix(npts, degree)
    return {
        "kind": "staggered gradient-divergence pair on primal and derivative Greville grids",
        "left_sides": [matrix_stats("B", ops.bmat, "left side (primal sample)"),
                       matrix_stats("H_0", ops.h0, "primal staggered metric"),
                       matrix_stats("H_1", ops.h1, "derivative staggered metric")],
        "left_side_label": "B (sample), H_0, H_1",
        "banded_factors": [matrix_stats("B", ops.bmat, "banded factor"),
                           matrix_stats("Bhat", ops.bhat_eta, "banded factor"),
                           matrix_stats("Delta", ops.delta, "banded factor")],
        "d1": matrix_stats("Bhat Delta B^{-1}", ops.d10, "first derivative (gradient)"),
        "d1_divergence": matrix_stats("D = H_0^{-1}(E - G^T H_1)", ops.d01,
                                      "first derivative (divergence)"),
        "d1_same_point": matrix_stats("B' B^{-1}", d1_same,
                                      "same-point derivative (auxiliary)"),
        "d2": matrix_stats("D G", d2, "second derivative"),
        "solves_per_d1": 1,
        "solves_per_d1_divergence": 4,
        "solves_per_d2": 5,
        "factored_d1": "G = Bhat Delta B^{-1} (solve: B)",
        "factored_d2": "D2 = D G with D = H_0^{-1}(E - G^T H_1); "
                       "H_0^{-1} = B M_0^{-1} B^T, H_1 = Bhat^{-T} M_1 Bhat^{-1} "
                       "(solves: B, Bhat, Bhat, M_0, B)",
    }


def build_csb_e6(npts: int) -> dict:
    dmat, _w, cid = bl.csb_e6_operator(npts)
    return {
        "kind": "CSB stable conservative explicit closure",
        "candidate": int(cid),
        "left_sides": [],
        "left_side_label": "none (explicit)",
        "d1": matrix_stats("D", dmat, "first derivative"),
        "d2": matrix_stats("D D", dmat @ dmat, "second derivative"),
        "solves_per_d1": 0,
        "solves_per_d2": 0,
        "factored_d1": "D1 = D",
        "factored_d2": "D2 = D D",
    }


CSB_T6_NOTE = (
    "baseline_closures.csb_t6_operator returns only the eliminated dense operator "
    "B^{-1} A / h; no repository constructor returns the banded pencil (B, A), "
    "so its condition number is not computed here and is not reconstructed."
)


def build_csb_t6(npts: int) -> dict:
    dmat, _w, cid = bl.csb_t6_operator(npts)
    return {
        "kind": "CSB stable conservative compact closure",
        "candidate": int(cid),
        "left_sides": [absent("B", CSB_T6_NOTE)],
        "left_side_label": "B (not exposed)",
        "banded_factors": [],
        "d1": matrix_stats("B^{-1}A", dmat, "first derivative (eliminated)"),
        "d2": matrix_stats("(B^{-1}A)^2", dmat @ dmat, "second derivative"),
        "solves_per_d1": 1,
        "solves_per_d2": 2,
        "factored_d1": "B D1 = A / h",
        "factored_d2": "D2 = D1 D1",
        "note": CSB_T6_NOTE,
    }


def build_sem(npts: int) -> dict:
    op = sem_grid_for(npts, SEM_DEGREE)
    mass = np.diag(op.mass)
    return {
        "kind": "continuous Gauss-Lobatto-Legendre spectral element",
        "degree": SEM_DEGREE,
        "nelem": int(op.nelem),
        "left_sides": [matrix_stats("M", mass, "left side (diagonal GLL mass)")],
        "left_side_label": "M (diagonal)",
        "banded_factors": [matrix_stats("M", mass, "banded factor"),
                           matrix_stats("K", op._stiff, "banded factor"),
                           matrix_stats("C", op._weakd, "banded factor")],
        "d1": matrix_stats("M^{-1} C", op.D1, "first derivative (weak)"),
        "d2": matrix_stats("-M^{-1} K", op.D2, "second derivative"),
        "solves_per_d1": 1,
        "solves_per_d2": 1,
        "factored_d1": "D1 = M^{-1} C; M diagonal",
        "factored_d2": "D2 = -M^{-1} K; M diagonal",
    }


# --- family registry --------------------------------------------------------
def family_records(npts: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for q in (4, 6, 8):
        out[f"E{q}"] = build_explicit(q, npts)
        out[f"C{q}"] = build_compact(q, npts)
        out[f"SBP{q}"] = build_sbp(q, npts)
        out[f"DRP{q}"] = build_drp(q, npts)
    out["C6-D2"] = build_compact_baseline_c6_d2(npts)
    out["SBP6-BO"] = build_sbp6_bo(npts)
    for p in (4, 6, 8):
        out[f"BMF{p}"] = build_bmf(p, npts)
        out[f"BMG{p}"] = build_bmg(p, npts)
    out["CSB-E6"] = build_csb_e6(npts)
    out["CSB-T6"] = build_csb_t6(npts)
    for label, rec in out.items():
        rec["family"] = label
        rec["npts"] = int(npts)
    return out


TABLE_ORDER = (
    "E4", "E6", "E8",
    "C4", "C6", "C8", "C6-D2",
    "SBP4", "SBP6", "SBP8", "SBP6-BO",
    "DRP4", "DRP6", "DRP8",
    "CSB-E6", "CSB-T6",
    "SEM",
    "BMF4", "BMF6", "BMF8",
    "BMG4", "BMG6", "BMG8",
)
SPLINE_FAMILIES = ("BMF4", "BMF6", "BMF8", "BMG4", "BMG6", "BMG8")


# --- N-dependence analysis --------------------------------------------------
def spread(values: list[float]) -> float:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return 0.0
    return float((max(vals) - min(vals)) / max(vals))


def log_slope(npts: list[int], values: list[float]) -> float | None:
    pairs = [(n, v) for n, v in zip(npts, values)
             if v is not None and np.isfinite(v) and v > 0.0]
    if len(pairs) < 2:
        return None
    ln = np.log([p[0] for p in pairs])
    lv = np.log([p[1] for p in pairs])
    return float(np.polyfit(ln, lv, 1)[0])


def n_dependence(sweep: dict[str, dict[str, dict]]) -> dict:
    grids = sorted(int(n) for n in sweep)
    report: dict[str, dict] = {}
    corners: list[str] = []
    for label in TABLE_ORDER:
        present = [n for n in grids if label in sweep[str(n)]]
        if not present:
            continue
        recs = [sweep[str(n)][label] for n in present]
        entry: dict = {"npts": present}
        for key in ("d1", "d2"):
            raw = [r[key]["cond_2"] for r in recs]
            eff = [r[key]["cond_2_effective"] for r in recs]
            nrm = [r[key]["norm_2"] for r in recs]
            entry[f"cond_{key}"] = raw
            entry[f"cond_effective_{key}"] = eff
            entry[f"norm_{key}"] = nrm
            entry[f"kernel_dim_{key}"] = [r[key]["kernel_dim"] for r in recs]
            entry[f"slope_{key}"] = log_slope(present, raw)
            entry[f"slope_effective_{key}"] = log_slope(present, eff)
            entry[f"slope_norm_{key}"] = log_slope(present, nrm)
        left_entries: dict[str, dict] = {}
        for pos in range(max(len(r["left_sides"]) for r in recs)):
            name = recs[0]["left_sides"][pos]["name"]
            vals = [r["left_sides"][pos]["cond_2"] for r in recs]
            comp = [n for n, v in zip(present, vals) if n in NPTS_COMPANION]
            comp_vals = [v for n, v in zip(present, vals) if n in NPTS_COMPANION]
            left_entries[name] = {
                "cond_2": vals,
                "spread": spread(comp_vals) if comp_vals else None,
                "slope": log_slope(present, vals),
                "companion_npts": comp,
            }
            if label in SPLINE_FAMILIES and comp_vals:
                sp = spread(comp_vals)
                sl = log_slope(present, vals) or 0.0
                if sp > SPREAD_TOL or abs(sl) > SLOPE_TOL:
                    corners.append(
                        f"{label} left side {name}: spread {sp:.3g}, "
                        f"log-log slope {sl:.3g}")
        entry["left_sides"] = left_entries
        report[label] = entry
    return {"per_family": report, "corners": corners}


# --- predeclared predictions (contract section 4) ---------------------------
P1_BAND = (6.0, 50.0)        # spline sample-matrix condition band, p = 4 to 8
P2_BAND = (3.0, 12.0)        # nodal compact left-side band
P3_D1_TOL = 0.15             # |slope - 1| tolerance on the first derivative
P3_D2_TOL = 0.25             # |slope - 2| tolerance on the second derivative


def score_predictions(sweep: dict, analysis: dict) -> dict:
    per = analysis["per_family"]
    p1_entries, p2_entries = {}, {}
    for label in SPLINE_FAMILIES:
        entry = per[label]["left_sides"]
        p1_entries[label] = {
            name: {"spread": info["spread"], "slope": info["slope"],
                   "cond_at_table_npts":
                       sweep[str(NPTS_TABLE)][label]["left_sides"][i]["cond_2"]}
            for i, (name, info) in enumerate(entry.items())}
    for label in ("C4", "C6", "C8"):
        info = per[label]["left_sides"]["A"]
        p2_entries[label] = {
            "spread": info["spread"], "slope": info["slope"],
            "cond_at_table_npts":
                sweep[str(NPTS_TABLE)][label]["left_sides"][0]["cond_2"]}
    b_vals = [v["B"]["cond_at_table_npts"] for v in p1_entries.values()]
    m_vals = [sweep[str(NPTS_TABLE)][f"BMG{p}"]["left_sides"][1]["cond_2"]
              for p in (4, 6, 8)]
    a_vals = [v["cond_at_table_npts"] for v in p2_entries.values()]
    p1_n_indep = all(v["B"]["spread"] is not None and v["B"]["spread"] <= SPREAD_TOL
                     for v in p1_entries.values())
    p1_band = P1_BAND[0] <= min(b_vals) and max(b_vals) <= P1_BAND[1]
    p2_band = P2_BAND[0] <= min(a_vals) and max(a_vals) <= P2_BAND[1]
    p3 = {}
    for label, entry in per.items():
        s1e, s2e = entry.get("slope_effective_d1"), entry.get("slope_effective_d2")
        s1n, s2n = entry.get("slope_norm_d1"), entry.get("slope_norm_d2")
        p3[label] = {
            "slope_effective_d1": s1e, "slope_effective_d2": s2e,
            "slope_norm_d1": s1n, "slope_norm_d2": s2n,
            "held_effective": (s1e is not None and s2e is not None
                               and abs(s1e - 1.0) <= P3_D1_TOL
                               and abs(s2e - 2.0) <= P3_D2_TOL),
            "held_norm": (s1n is not None and s2n is not None
                          and abs(s1n - 1.0) <= P3_D1_TOL
                          and abs(s2n - 2.0) <= P3_D2_TOL),
        }
    return {
        "P1_spline_left_sides": {
            "statement": "spline mass and sample matrices grow with degree but "
                         "are N-independent; BMG p=4..8 about 6 to 50",
            "n_independent": bool(p1_n_indep),
            "sample_matrix_band": [float(min(b_vals)), float(max(b_vals))],
            "sample_matrix_band_held": bool(p1_band),
            "mass_matrix_band": [float(min(m_vals)), float(max(m_vals))],
            "mass_matrix_band_held": bool(P1_BAND[0] <= min(m_vals)
                                          and max(m_vals) <= P1_BAND[1]),
            "detail": p1_entries,
        },
        "P2_nodal_compact_left_sides": {
            "statement": "nodal compact left sides about 3 to 12",
            "band": [float(min(a_vals)), float(max(a_vals))],
            "held": bool(p2_band),
            "detail": p2_entries,
        },
        "P3_derivative_scaling": {
            "statement": "assembled D1 scales like N and D2 like N^2",
            "held_effective_all": all(v["held_effective"] for v in p3.values()),
            "held_norm_all": all(v["held_norm"] for v in p3.values()),
            "exceptions_effective": [k for k, v in p3.items()
                                     if not v["held_effective"]],
            "exceptions_norm": [k for k, v in p3.items() if not v["held_norm"]],
            "detail": p3,
        },
    }


# --- LaTeX ------------------------------------------------------------------
def fmt(value: float | None, sig: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "---"
    if value == 0.0:
        return "$0$"
    exp = int(np.floor(np.log10(abs(value))))
    if -2 <= exp < 4:
        digits = max(0, sig - 1 - exp)
        return f"${value:.{digits}f}$"
    mant = value / 10.0 ** exp
    return f"${mant:.{sig - 1}f}\\times10^{{{exp}}}$"


NAME_TEX = {"Bhat": "\\widehat{B}", "Delta": "\\Delta"}


def tex_name(name: str) -> str:
    return NAME_TEX.get(name, name)


def band_cell(rec: dict) -> str:
    factors = rec.get("banded_factors")
    if factors is None:
        factors = [rec["d1"]]
    parts = []
    for f in factors:
        if f.get("bandwidth") is None:
            continue
        parts.append(f"${tex_name(f['name'])}$~{f['bandwidth']}")
        if len(parts) == 3:
            break
    return ", ".join(parts) if parts else "---"


def left_cell(rec: dict) -> tuple[str, str]:
    lefts = rec["left_sides"]
    if not lefts:
        return "none", "---"
    names = ", ".join(f"${tex_name(l['name'])}$" for l in lefts[:2])
    conds = ", ".join(fmt(l["cond_2"]) for l in lefts[:2])
    return names, conds


def main_table(sweep: dict, sem_rec: dict | None) -> str:
    recs = sweep[str(NPTS_TABLE)]
    lines = [
        "% Generated by code/s6_comparator_conditioning.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Left-side and assembled-derivative $2$-norm condition "
        "numbers, banded-factor bandwidths, and banded solves per $D_1$ and "
        "$D_2$ application, on $[0,1]$ at $N=81$. A star marks the ratio "
        "taken on the complement of the constant and affine null space.}",
        "  \\label{tab:comparator-conditioning}",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\renewcommand{\\arraystretch}{1.18}",
        "  \\begin{tabular*}{0.98\\textwidth}{@{\\extracolsep{\\fill}}"
        "lllrrlc@{}}",
        "    \\toprule",
        "    \\classycap{family} & \\classycap{left side} & "
        "\\classycap{cond left} & \\classycap{cond$^{\\star}D_1$} & "
        "\\classycap{cond$^{\\star}D_2$} & \\classycap{bandwidths} & "
        "\\classycap{solves} \\\\",
        "    \\midrule",
    ]
    for label in TABLE_ORDER:
        if label == "SEM":
            if sem_rec is None:
                continue
            rec = sem_rec
            names, conds = left_cell(rec)
            names = f"{names}, $N={rec['npts']}$"
        else:
            rec = recs[label]
            names, conds = left_cell(rec)
        if label in SPLINE_FAMILIES:
            lines.append("    \\rowcolor{classyaccent!5}")
        solves = f"{rec['solves_per_d1']}\\,/\\,{rec['solves_per_d2']}"
        lines.append(
            f"    {label} & {names} & {conds} & "
            f"{fmt(rec['d1']['cond_2_effective'])} & "
            f"{fmt(rec['d2']['cond_2_effective'])} & "
            f"{band_cell(rec)} & {solves} \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular*}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def companion_table(sweep: dict, sem_sweep: dict) -> str:
    lines = [
        "",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Left-side $2$-norm condition number of each family at "
        "three grids. Spline entries list the sample matrix $B$; nodal "
        "entries list the compact left side $A$ or the norm $H$.}",
        "  \\label{tab:comparator-conditioning-grids}",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{6pt}",
        "  \\renewcommand{\\arraystretch}{1.18}",
        "  \\begin{tabular*}{0.78\\textwidth}{@{\\extracolsep{\\fill}}"
        "llrrrr@{}}",
        "    \\toprule",
        "    \\classycap{family} & \\classycap{left side} & "
        f"\\classycap{{$N={NPTS_COMPANION[0]}$}} & "
        f"\\classycap{{$N={NPTS_COMPANION[1]}$}} & "
        f"\\classycap{{$N={NPTS_COMPANION[2]}$}} & "
        "\\classycap{spread} \\\\",
        "    \\midrule",
    ]
    for label in TABLE_ORDER:
        if label == "SEM":
            grids = [n for n in SEM_COMPANION if str(n) in sem_sweep]
            if not grids:
                continue
            recs = [sem_sweep[str(n)] for n in grids]
            name = f"$M$, $N={'/'.join(str(n) for n in grids)}$"
        else:
            grids = list(NPTS_COMPANION)
            recs = [sweep[str(n)][label] for n in grids]
            if not recs[0]["left_sides"]:
                continue
            name = f"${tex_name(recs[0]['left_sides'][0]['name'])}$"
        vals = [r["left_sides"][0]["cond_2"] for r in recs]
        if any(v is None for v in vals):
            continue
        if label in SPLINE_FAMILIES:
            lines.append("    \\rowcolor{classyaccent!5}")
        cells = " & ".join(fmt(v) for v in vals)
        lines.append(f"    {label} & {name} & {cells} & "
                     f"{100.0 * spread(vals):.2f}\\% \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular*}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


# --- verification -----------------------------------------------------------
# The raw cond_2 of an assembled D1 or D2 is the condition number of a singular
# matrix (constants lie in the kernel), so the reproducible derivative quantity
# is cond_2_effective. Left sides are checked on the raw cond_2.
SPOT_CHECKS = (("BMG6", 81, "d1", "cond_2_effective"),
               ("BMF6", 81, "d2", "cond_2_effective"),
               ("C6", 81, "d1", "cond_2_effective"),
               ("SBP8", 41, "d2", "cond_2_effective"),
               ("E4", 161, "d1", "cond_2_effective"),
               ("CSB-E6", 41, "d1", "cond_2_effective"),
               ("BMG4", 41, "left0", "cond_2"),
               ("BMG4", 161, "left0", "cond_2"),
               ("BMF8", 81, "left0", "cond_2"),
               ("C8", 81, "left0", "cond_2"),
               ("SBP6", 81, "left0", "cond_2"))


def verify(path: Path, rtol: float = 1.0e-8) -> int:
    payload = json.loads(path.read_text())
    sweep = payload["sweep"]
    failures: list[str] = []
    cache: dict[int, dict] = {}
    for label, npts, key, field in SPOT_CHECKS:
        if npts not in cache:
            cache[npts] = family_records(npts)
        if key == "left0":
            fresh = cache[npts][label]["left_sides"][0][field]
            stored = sweep[str(npts)][label]["left_sides"][0][field]
        else:
            fresh = cache[npts][label][key][field]
            stored = sweep[str(npts)][label][key][field]
        rel = abs(fresh - stored) / max(abs(stored), 1.0)
        status = "ok" if rel <= rtol else "FAIL"
        print(f"  {label:<8s} N={npts:<4d} {key:<6s} {field:<17s} "
              f"stored {stored:.8e}  recomputed {fresh:.8e}  "
              f"rel {rel:.2e}  {status}")
        if rel > rtol:
            failures.append(f"{label} N={npts} {key} {field}: rel {rel:.3e}")
    # BMG metric identity, the structural claim behind the H_BMG column
    for npts in (41, 81):
        _x, _k, bmat, _g = bsa.clamped_bspline_matrices(npts, 6)
        mass, _adv = bsa.bspline_mass_and_advection(npts, 6)
        _x1, _d1, hmat = bsa.bmg_sbp_first_derivative_matrix(npts, 6)
        # Multiplied-back form: B^T H B = M. Independent arithmetic from the
        # constructor's own B^{-T} M B^{-1}, so the residual is a real check.
        res = float(np.max(np.abs(bmat.T @ hmat @ bmat - mass)))
        scale = float(np.max(np.abs(mass)))
        print(f"  BMG6    N={npts:<4d} B^T H B = M residual "
              f"{res:.3e} (relative {res / scale:.3e})")
        if res / scale > 1.0e-10:
            failures.append(f"BMG6 N={npts} metric identity {res / scale:.3e}")
    if failures:
        print("VERIFY FAILED:")
        for line in failures:
            print("  " + line)
        return 1
    print("VERIFY OK")
    return 0


# --- driver -----------------------------------------------------------------
def run(out_path: Path, tex_path: Path) -> dict:
    sweep: dict[str, dict] = {}
    for npts in NPTS_SWEEP:
        sweep[str(npts)] = family_records(npts)
    sem_grids = sorted(set(SEM_LADDER)
                       | {n for n in NPTS_SWEEP if (n - 1) % SEM_DEGREE == 0}
                       | {n for n in SEM_COMPANION if (n - 1) % SEM_DEGREE == 0})
    sem_sweep: dict[str, dict] = {}
    for npts in sem_grids:
        rec = build_sem(npts)
        rec["family"] = "SEM"
        rec["npts"] = int(npts)
        sem_sweep[str(npts)] = rec

    analysis = n_dependence(sweep)
    sem_left = [sem_sweep[str(n)]["left_sides"][0]["cond_2"]
                for n in sem_grids]
    sem_entry: dict = {"npts": list(sem_grids),
                       "left_sides": {"M": {
                           "cond_2": sem_left,
                           "spread": spread(sem_left),
                           "slope": log_slope(list(sem_grids), sem_left),
                           "companion_npts": list(sem_grids)}}}
    for key in ("d1", "d2"):
        raw = [sem_sweep[str(n)][key]["cond_2"] for n in sem_grids]
        eff = [sem_sweep[str(n)][key]["cond_2_effective"] for n in sem_grids]
        nrm = [sem_sweep[str(n)][key]["norm_2"] for n in sem_grids]
        sem_entry[f"cond_{key}"] = raw
        sem_entry[f"cond_effective_{key}"] = eff
        sem_entry[f"norm_{key}"] = nrm
        sem_entry[f"kernel_dim_{key}"] = [sem_sweep[str(n)][key]["kernel_dim"]
                                          for n in sem_grids]
        sem_entry[f"slope_{key}"] = log_slope(list(sem_grids), raw)
        sem_entry[f"slope_effective_{key}"] = log_slope(list(sem_grids), eff)
        sem_entry[f"slope_norm_{key}"] = log_slope(list(sem_grids), nrm)
    analysis["per_family"]["SEM"] = sem_entry

    payload = {
        "producer": "code/s6_comparator_conditioning.py",
        "npts_sweep": list(NPTS_SWEEP),
        "npts_table": NPTS_TABLE,
        "npts_companion": list(NPTS_COMPANION),
        "sem_degree": SEM_DEGREE,
        "sem_grids": list(sem_grids),
        "band_tolerance": BAND_TOL,
        "grid_provenance": {
            "closed-box acoustics": 41,
            "analytic wake": 25,
            "localized packet": [49, 97, 193],
            "heat cavity": 81,
            "spectral-element scalar control": list(SEM_LADDER),
        },
        "missing_constructors": {"CSB-T6 left side B": CSB_T6_NOTE},
        "predictions": score_predictions(sweep, analysis),
        "sweep": sweep,
        "sem": sem_sweep,
        "n_dependence": analysis,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")

    sem_table_rec = sem_sweep.get(str(SEM_TABLE_NPTS))
    tex = main_table(sweep, sem_table_rec) + "\n" + companion_table(sweep, sem_sweep)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="code/results/comparator_conditioning.json",
                        help="JSON artifact path")
    parser.add_argument("--tex", default="code/results/comparator_conditioning_table.tex",
                        help="LaTeX table snippet path")
    parser.add_argument("--verify", metavar="PATH", default=None,
                        help="recompute spot values against an existing JSON artifact")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    if args.verify:
        return verify(Path(args.verify) if Path(args.verify).is_absolute()
                      else root / args.verify)
    out_path = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    tex_path = Path(args.tex) if Path(args.tex).is_absolute() else root / args.tex
    payload = run(out_path, tex_path)
    corners = payload["n_dependence"]["corners"]
    print(f"wrote {out_path}")
    print(f"wrote {tex_path}")
    print(f"corners flagged: {len(corners)}")
    for line in corners:
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
