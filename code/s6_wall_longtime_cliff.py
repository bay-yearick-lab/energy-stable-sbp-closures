#!/usr/bin/env python3
"""Long-time wall-bounded acoustic stability-and-accuracy separation.

A closed rigid-wall box conserves acoustic energy exactly, so any growth in a
discrete solution is the wall closure injecting energy. Two operator properties
decide the long-time outcome, and they separate the six matched-order families
into three groups:

  spectral abscissa  max Re(eig) of the semidiscrete acoustic operator. Positive
                     means energy injection and guaranteed long-time blow-up,
                     independent of the time step. The order-tuned (explicit,
                     compact) and dispersion-tuned (DRP) wall closures are
                     positive; the mimetic SBP and spline closures are zero.

  wall-row order     among the energy-stable families, the achievable accuracy is
                     set by the wall-row order. The diagonal-norm SBP closure
                     reaches q/2 at the wall and caps the error; the spline routes
                     reach the spline degree and are orders of magnitude lower.

Only the spline routes are simultaneously energy-stable and high-order at the
wall, so they alone stay both bounded and accurate over long times.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import boundary_spline_analysis as ba
import baseline_closures as bl
from project_paths import RESULTSDIR, ensure_generated_dirs, save_figure

ensure_generated_dirs()

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.2,
    "axes.labelsize": 11.5,
    "axes.titlesize": 12.0,
    "legend.fontsize": 9.8,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 180,
    "savefig.dpi": 300,
})

FAMILIES = [("E", "E"), ("C", "C"), ("SBP", "SBP"), ("DRP", "DRP"), ("I", "BMG"), ("B", "BMF")]
COLORS = {"E": "#7a7a7a", "C": "#4c72b0", "SBP": "#8c564b", "DRP": "#9467bd", "BMG": "#2ca02c", "BMF": "#d62728",
          "C6F": "#e08214", "SBP6-BO": "#66c2a5",
          "CSB-E6": "#a6761d", "CSB-T6": "#b5326d", "C6-CB3": "#0b3d91"}
# Task-2 full-strength competitors, added alongside the original six families.
# Only meaningful at q=6 (both are only defined for order 6 in this codebase);
# they are added as EXTRA curves/bars, not substitutions for C/SBP.
EXTRA_FAMILIES_Q6 = ["C6F", "SBP6-BO"]
# Contract K entrants: the CSB explicit and compact schemes at their
# published strength, and the compact family closed by the published Section 4.1
# wall row instead of the generic explicit one-sided rows.
MATRIX_FAMILIES_Q6 = ["CSB-E6", "CSB-T6", "C6-CB3"]
LINESTYLES = {"CSB-E6": (0, (4.0, 1.5)), "CSB-T6": (0, (1.0, 1.1)),
              "C6-CB3": (0, (5.0, 1.0, 1.0, 1.0))}
N = 41
MODE = 6
ALPHA_F = 0.49
Y_FLOOR = 1.0e-10
Y_CEIL = 1.0e4


def p_exact(x, t):
    return np.cos(MODE * np.pi * np.asarray(x, float)) * np.cos(MODE * np.pi * t)


def u_exact(x, t):
    return np.sin(MODE * np.pi * np.asarray(x, float)) * np.sin(MODE * np.pi * t)


def acoustic_operator(kind: str, q: int, npts: int) -> np.ndarray:
    """Semidiscrete closed-box acoustic operator on the state [p; u_interior]."""
    if kind == "B":  # BMF staggered
        op = ba.bmf_staggered_operators(npts, q)
        d10, d01 = op.d10, op.d01
        mu = np.arange(1, d10.shape[0] - 1)
        nu = mu.size
        return np.block([[np.zeros((npts, npts)), -d01[:, mu]],
                         [-d10[mu, :], np.zeros((nu, nu))]])
    if kind in ("C6F", "SBP6-BO"):
        d = ba.build_operator_extended(kind, 6, npts).matrix
        return np.block([[np.zeros((npts, npts)), -d[:, 1:-1]],
                         [-d[1:-1, :], np.zeros((npts - 2, npts - 2))]])
    d = ba.build_operator(kind, q, npts).matrix
    return np.block([[np.zeros((npts, npts)), -d[:, 1:-1]],
                     [-d[1:-1, :], np.zeros((npts - 2, npts - 2))]])


def spectral_abscissa(kind: str, q: int, npts: int) -> float:
    return float(np.max(np.real(np.linalg.eigvals(acoustic_operator(kind, q, npts)))))


def simulate_extra_history(kind: str, npts: int, t_end: float, cfl: float, record_count: int):
    """Long-time reflected-acoustics history for the extra (q=6-only) competitors.

    C6F applies the tenth-order implicit low-pass (F10) filter once per completed RK4 step
    (after the stage combination), matching `CompactFilteredFD.apply_filter_to_state`;
    SBP6-BO uses `ba.simulate_reflected_acoustics` directly since it has no
    filtering step (a plain SBP-type first-derivative operator).
    """
    op = ba.build_operator_extended(kind, 6, npts)
    if kind != "C6F":
        return ba.simulate_reflected_acoustics(op, p_exact, u_exact, t_end, cfl=cfl, record_count=record_count)

    x = op.x
    h = x[1] - x[0]
    dt = float(cfl) * float(h)
    nsteps = int(np.ceil(float(t_end) / dt))
    dt = float(t_end) / nsteps
    dmat = op.matrix
    m_u = npts - 2
    y = np.concatenate([p_exact(x, 0.0), u_exact(x, 0.0)[1:-1]])
    keep_every = max(1, nsteps // max(2, int(record_count) - 1))

    def unpack(state, t):
        p = np.asarray(state[:npts], dtype=float)
        u = np.empty(npts, dtype=float)
        u[0] = float(np.asarray(u_exact(x[0], t)))
        u[-1] = float(np.asarray(u_exact(x[-1], t)))
        u[1:-1] = state[npts:npts + m_u]
        return p, u

    def rhs(t, state):
        p, u = unpack(state, t)
        dp = -dmat @ u
        du = -dmat @ p
        return np.concatenate([dp, du[1:-1]])

    rec_t, rec_e = [], []

    def record(t, state):
        p, u = unpack(state, t)
        pex, uex = p_exact(x, t), u_exact(x, t)
        rec_t.append(float(t))
        rec_e.append(float(ba.coupled_relative_l2(p, pex, u, uex)))

    record(0.0, y)
    t = 0.0
    for step in range(nsteps):
        k1 = rhs(t, y)
        k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = rhs(t + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        # Filter applied once per completed RK4 step, to the full pressure
        # state and to the (BC-extended) velocity state, then the interior
        # velocity dofs are re-extracted -- the same convention verified in
        # code/verify_strong_competitors.py's closed-box gate.
        p_full, u_full = unpack(y, t)
        p_filt = op.apply_filter_to_state(p_full)
        u_filt = op.apply_filter_to_state(u_full)
        y = np.concatenate([p_filt, u_filt[1:-1]])
        if step == nsteps - 1 or (step + 1) % keep_every == 0:
            record(t, y)

    from types import SimpleNamespace
    return SimpleNamespace(times=np.asarray(rec_t), error_history=np.asarray(rec_e))


def matrix_first_derivative(tag: str, npts: int) -> np.ndarray:
    """First-derivative matrix of a Contract-K entrant on `npts` uniform nodes."""
    if tag == "CSB-E6":
        return bl.csb_e6_operator(npts)[0]
    if tag == "CSB-T6":
        return bl.csb_t6_operator(npts)[0]
    if tag == "C6-CB3":
        return ba.build_operator_extended("C6-CB3", 6, npts).matrix
    raise ValueError(tag)


def simulate_matrix_history(tag: str, npts: int, t_end: float, cfl: float, record_count: int):
    """Closed-box history for an entrant supplied as a bare derivative matrix.

    The protocol is the nodal path of `ba.simulate_reflected_acoustics`:
    strong rigid-wall velocity traces, RK4 at `dt = cfl * h`, and the trimmed
    interior-strip coupled error of the six-family curves.
    """
    dmat = matrix_first_derivative(tag, npts)
    x = np.linspace(0.0, 1.0, npts)
    h = x[1] - x[0]
    dt = float(cfl) * float(h)
    nsteps = int(np.ceil(float(t_end) / dt))
    dt = float(t_end) / nsteps
    m_u = npts - 2
    sl = ba.trimmed_interior_slice(npts, 6)
    y = np.concatenate([p_exact(x, 0.0), u_exact(x, 0.0)[1:-1]])
    keep_every = max(1, nsteps // max(2, int(record_count) - 1))

    def unpack(state, t):
        p = np.asarray(state[:npts], dtype=float)
        u = np.empty(npts, dtype=float)
        u[0] = float(np.asarray(u_exact(x[0], t)))
        u[-1] = float(np.asarray(u_exact(x[-1], t)))
        u[1:-1] = state[npts:npts + m_u]
        return p, u

    def rhs(t, state):
        p, u = unpack(state, t)
        return np.concatenate([-dmat @ u, (-dmat @ p)[1:-1]])

    rec_t, rec_e = [], []

    def record(t, state):
        p, u = unpack(state, t)
        pex, uex = p_exact(x, t), u_exact(x, t)
        rec_t.append(float(t))
        rec_e.append(float(ba.coupled_relative_l2(p[sl], pex[sl], u[sl], uex[sl])))

    record(0.0, y)
    t = 0.0
    for step in range(nsteps):
        k1 = rhs(t, y)
        k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = rhs(t + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        if step == nsteps - 1 or (step + 1) % keep_every == 0:
            record(t, y)

    from types import SimpleNamespace
    return SimpleNamespace(times=np.asarray(rec_t), error_history=np.asarray(rec_e))


def matrix_abscissa(tag: str, npts: int) -> float:
    dmat = matrix_first_derivative(tag, npts)
    op = np.block([[np.zeros((npts, npts)), -dmat[:, 1:-1]],
                   [-dmat[1:-1, :], np.zeros((npts - 2, npts - 2))]])
    return float(np.max(np.real(np.linalg.eigvals(op))))


def main() -> None:
    t_end = 6.0
    out = {"N": N, "mode": MODE, "t_end": t_end, "alpha_f": ALPHA_F,
           "spectral_abscissa": {}, "longtime_error": {}}

    # rigorous, dt-independent energy-stability verdict across orders
    for q in (4, 6, 8):
        out["spectral_abscissa"][str(q)] = {
            tag: spectral_abscissa(kind, q, N) for kind, tag in FAMILIES
        }
    # Extra q=6-only competitors (C6F, SBP6-BO): report their spectral abscissa
    # alongside the original six at q=6 for the same bar panel.
    out["spectral_abscissa"]["6"].update({
        tag: spectral_abscissa(tag, 6, N) for tag in EXTRA_FAMILIES_Q6
    })
    out["spectral_abscissa"]["6"].update({
        tag: matrix_abscissa(tag, N) for tag in MATRIX_FAMILIES_Q6
    })

    # long-time error history at the q=6 block
    q = 6
    histories = {}
    for kind, tag in FAMILIES:
        op = ba.build_operator(kind, q, N)
        res = ba.simulate_reflected_acoustics(op, p_exact, u_exact, t_end, cfl=0.012, record_count=200)
        t = np.asarray(res.times)
        e = np.asarray(res.error_history)
        histories[tag] = (t, e)
        # time at which the relative error first reaches order one (solution destroyed)
        hit = np.where(e >= 1.0)[0]
        out["longtime_error"][tag] = {
            "final": float(e[-1]),
            "max": float(np.max(e)),
            "t_to_order_one": float(t[hit[0]]) if hit.size else None,
        }

    for tag in EXTRA_FAMILIES_Q6:
        res = simulate_extra_history(tag, N, t_end, cfl=0.012, record_count=200)
        t = np.asarray(res.times)
        e = np.asarray(res.error_history)
        histories[tag] = (t, e)
        hit = np.where(e >= 1.0)[0]
        out["longtime_error"][tag] = {
            "final": float(e[-1]),
            "max": float(np.max(e)),
            "t_to_order_one": float(t[hit[0]]) if hit.size else None,
        }

    for tag in MATRIX_FAMILIES_Q6:
        res = simulate_matrix_history(tag, N, t_end, cfl=0.012, record_count=200)
        t = np.asarray(res.times)
        e = np.asarray(res.error_history)
        histories[tag] = (t, e)
        hit = np.where(e >= 1.0)[0]
        out["longtime_error"][tag] = {
            "final": float(e[-1]),
            "max": float(np.max(e)),
            "t_to_order_one": float(t[hit[0]]) if hit.size else None,
        }

    all_tags = [t for _k, t in FAMILIES] + EXTRA_FAMILIES_Q6 + MATRIX_FAMILIES_Q6

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.4, 4.6))

    # Panel (a): long-time error history. The axis is clipped so the stable
    # families are readable; a family that leaves the frame carries an arrow at
    # its exit time instead of compressing every curve into the lower decade.
    for tag in all_tags:
        t, e = histories[tag]
        ls = LINESTYLES.get(tag, "--" if tag in EXTRA_FAMILIES_Q6 else "-")
        e = np.clip(e, Y_FLOOR, 1e300)
        inside = e <= Y_CEIL
        axL.semilogy(t[inside], e[inside], color=COLORS[tag], lw=1.7, ls=ls)
        if not inside.all():
            exit_idx = int(np.argmax(~inside))
            axL.annotate("", xy=(t[exit_idx], Y_CEIL * 0.95),
                         xytext=(t[exit_idx], Y_CEIL * 0.03),
                         arrowprops=dict(arrowstyle="-|>", color=COLORS[tag],
                                         lw=1.3, shrinkA=0.0, shrinkB=0.0))
    axL.axhline(1.0, color="black", lw=0.7, ls=":")
    axL.text(t_end * 0.55, 1.6, "solution destroyed", fontsize=9.0, color="0.25")
    axL.text(t_end * 0.55, Y_CEIL * 0.06, "arrows: error leaves the frame",
             fontsize=9.0, color="0.25")
    axL.set_xlabel("time (wall transit times)")
    axL.set_ylabel(r"relative $L^2$ error of $(p,u)$")
    axL.set_title("(a) Long-time closed-box acoustics")
    axL.set_ylim(Y_FLOOR, Y_CEIL)
    axL.set_xlim(0, t_end)

    # Panel (b): spectral abscissa across orders (q=6 block carries the two extra bars)
    qs = [4, 6, 8]
    tags = [t for _k, t in FAMILIES]
    xb = np.arange(len(tags))
    width = 0.20
    for i, q in enumerate(qs):
        vals = [max(out["spectral_abscissa"][str(q)][t], 1e-16) for t in tags]
        axR.bar(xb + (i - 1) * width, vals, width, label=f"$q={q}$",
                color=["#c44" if v > 1e-6 else "#2ca02c" for v in vals], alpha=0.55 + 0.18 * i)
    extra_tags = EXTRA_FAMILIES_Q6 + MATRIX_FAMILIES_Q6
    extra_xb = np.arange(len(tags), len(tags) + len(extra_tags))
    extra_vals = [max(out["spectral_abscissa"]["6"][t], 1e-16) for t in extra_tags]
    axR.bar(extra_xb, extra_vals, width * 2.2,
            color=["#c44" if v > 1e-6 else "#2ca02c" for v in extra_vals], alpha=0.85,
            hatch="//", edgecolor="0.3", linewidth=0.6)
    axR.set_yscale("log")
    axR.axhline(1e-6, color="black", lw=0.7, ls=":")
    axR.set_xticks(np.concatenate([xb, extra_xb]))
    axR.set_xticklabels(tags + extra_tags, rotation=22, ha="right")
    axR.set_ylabel(r"spectral abscissa $\max\,\mathrm{Re}\,\lambda$")
    axR.set_title("(b) Energy injection rate of the operator")
    axR.set_ylim(1e-16, 1e2)

    handles = [Line2D([0], [0], color=COLORS[t], lw=2.0,
                      ls=LINESTYLES.get(t, "--" if t in EXTRA_FAMILIES_Q6 else "-"),
                      label=t)
               for t in all_tags]
    fig.legend(handles=handles,
               ncol=6, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.07), fontsize=9.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "fig_s6_wall_longtime_cliff")
    plt.close(fig)

    (RESULTSDIR / "wall_longtime_cliff_summary.json").write_text(json.dumps(out, indent=2))
    print("spectral abscissa (q=6):", {t: round(out["spectral_abscissa"]["6"][t], 3) for t in all_tags})
    print("long-time error (final):", {t: f"{out['longtime_error'][t]['final']:.2e}" for t in all_tags})
    print("t to order-one error:", {t: out["longtime_error"][t]["t_to_order_one"] for t in all_tags})
    print(f"wrote figure + {RESULTSDIR/'wall_longtime_cliff_summary.json'}")


if __name__ == "__main__":
    main()
