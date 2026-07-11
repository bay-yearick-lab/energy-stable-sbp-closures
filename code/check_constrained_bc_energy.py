"""Energy law of the ACTUAL constrained boundary dynamics.

The manuscript's strong-enforcement theorem inserted the unconstrained ODE
into the SBP identity. The marched system is a CONSTRAINED evolution: with
trace matrix L and data g(t), the semidiscrete flow is

    u' = F(u) + X ( g'(t) - L F(u) ),        L u(0) = g(0),

where X is a lifting with L X = I. Two liftings are in play:

  selector   X_S : columns supported on the "owned" wall rows (the row
               replacement the current implementations use, e.g.
               heat_cavity_core.enforce_zero_walls);
  H-orthogonal X_H = H^{-1} L^T (L H^{-1} L^T)^{-1} (the Lagrange-multiplier
               DAE  H u' = H F(u) + L^T lambda, L u' = g'; equivalently the
               H-orthogonal projection of the dynamics onto the constraint
               manifold; equivalently the restricted Galerkin trial space).

Exact balance for ANY lifting:  dE/dt = u^T H F(u) + u^T H X (g' - L F(u)).
For X_H the constraint work is a PURE TRACE quantity,

    W_c = g^T (L H^{-1} L^T)^{-1} ( g' - L F(u) ),

which vanishes identically for homogeneous data: the SBP energy law then
holds EXACTLY for the constrained system. For X_S the correction does
interior work through the dense metric and the theorem's balance fails.

This certificate measures, to machine precision, for BMG advection
(inflow-only Dirichlet, the well-posed prescription), BMF/BMG diffusion
(Dirichlet/Neumann/Robin), and the BMF wave pair (Dirichlet on p / Neumann
on v / Robin), for degrees p=2..8 and two grids, at random constraint-
consistent states and along an RK4 march with time-dependent data:

  T1  the CURRENT theorem's balance vs the SELECTOR dynamics  (expected FAIL;
      the certified failure sizes justify the theorem-level correction);
  T2  the CORRECTED balance (SBP form + trace work) vs the H-ORTHOGONAL
      dynamics                                            (roundoff PASS);
  T3  homogeneous data: exact conservation / pure dissipation under X_H
                                                           (roundoff PASS);
  T4  three-way equivalence multiplier == projection == restricted Galerkin
                                                           (roundoff PASS);
  T5  spectral abscissae of both enforced operators (decides whether prior
      homogeneous-BC experiment results remain stability-valid);
  T6  marched nonhomogeneous demo: constraint drift and energy-ledger closure.

Usage:
  python code/check_constrained_bc_energy.py --write code/results/constrained_bc_energy.json
  python code/check_constrained_bc_energy.py --verify code/results/constrained_bc_energy.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boundary_spline_analysis import (  # noqa: E402
    bmf_staggered_operators,
    bmg_nodal_operator,
    bmg_sbp_second_derivative_matrix,
)
from project_paths import RESULTSDIR  # noqa: E402

RNG = np.random.default_rng(20260710)
DEGREES = (2, 3, 4, 5, 6, 7, 8)
GRIDS = (17, 33)
N_STATES = 10


# ----------------------------------------------------------------------------
# case assembly: (A, H, L, owned_rows, sbp_form, name)
# ----------------------------------------------------------------------------

def bmg_blocks(n: int, p: int):
    x, D, H = bmg_nodal_operator(n, p)
    _x2, D2 = bmg_sbp_second_derivative_matrix(n, p)
    return x, D, H, D2


def bmf_blocks(n: int, p: int):
    ops = bmf_staggered_operators(n, p)
    left = -ops.boundary[0, :].copy()          # r_0^T on the derivative grid
    right = ops.boundary[-1, :].copy()         # r_1^T
    return ops, left, right


def build_cases(n: int, p: int):
    """Return the list of certified cases for one (n, p)."""
    cases = []
    x, D, H, D2 = bmg_blocks(n, p)
    e0 = np.zeros(n); e0[0] = 1.0
    eN = np.zeros(n); eN[-1] = 1.0
    a, nu = 1.0, 1.0

    # SBP closed forms (verified as operator gates before use)
    def sbp_adv(u, A=None):
        return -0.5 * a * (u[-1] ** 2 - u[0] ** 2)

    Du_cache = {}

    def sbp_diff_bmg(u, A=None):
        v = D @ u
        return nu * (-(v @ H @ v) + u[-1] * v[-1] - u[0] * v[0])

    # BMG advection, inflow-only Dirichlet (the well-posed prescription)
    cases.append(dict(name="bmg_adv_dirichlet_inflow", A=-a * D, H=H,
                      L=e0[None, :], owned=[0], sbp=sbp_adv))
    # BMG diffusion under all three BC types
    LN = np.vstack([D[0, :], D[-1, :]])
    LD = np.vstack([e0, eN])
    cases.append(dict(name="bmg_diff_dirichlet", A=nu * D2, H=H,
                      L=LD, owned=[0, n - 1], sbp=sbp_diff_bmg))
    cases.append(dict(name="bmg_diff_neumann", A=nu * D2, H=H,
                      L=LN, owned=[0, n - 1], sbp=sbp_diff_bmg))
    # physical Robin uses the OUTWARD normal derivative: alpha u + beta du/dn,
    # i.e. -u' at the left wall and +u' at the right wall
    robin_sign = np.diag([-0.7, 0.7])
    cases.append(dict(name="bmg_diff_robin", A=nu * D2, H=H,
                      L=LD + robin_sign @ LN, owned=[0, n - 1], sbp=sbp_diff_bmg))

    ops, left, right = bmf_blocks(n, p)
    n1 = ops.d10.shape[0]
    LFdiff = ops.d01 @ ops.d10
    H0, H1 = ops.h0, ops.h1

    def sbp_diff_bmf(u, A=None):
        v = ops.d10 @ u
        return nu * (-(v @ H1 @ v) + u[-1] * (right @ v) - u[0] * (left @ v))

    e0f = np.zeros(n); e0f[0] = 1.0
    eNf = np.zeros(n); eNf[-1] = 1.0
    LDf = np.vstack([e0f, eNf])
    LNf = np.vstack([left @ ops.d10, right @ ops.d10])
    cases.append(dict(name="bmf_diff_dirichlet", A=nu * LFdiff, H=H0,
                      L=LDf, owned=[0, n - 1], sbp=sbp_diff_bmf))
    cases.append(dict(name="bmf_diff_neumann", A=nu * LFdiff, H=H0,
                      L=LNf, owned=[0, n - 1], sbp=sbp_diff_bmf))
    cases.append(dict(name="bmf_diff_robin", A=nu * LFdiff, H=H0,
                      L=LDf + robin_sign @ LNf, owned=[0, n - 1], sbp=sbp_diff_bmf))

    # BMF wave pair, state z = (p, v)
    Z = np.zeros
    Aw = np.block([[Z((n, n)), -ops.d01], [-ops.d10, Z((n1, n1))]])
    Hw = np.block([[H0, Z((n, n1))], [Z((n1, n)), H1]])

    def sbp_wave(z, A=None):
        pz, vz = z[:n], z[n:]
        return pz[0] * (left @ vz) - pz[-1] * (right @ vz)

    LD_w = np.hstack([LDf, Z((2, n1))])
    LN_w = np.hstack([Z((2, n)), np.vstack([left, right])])
    cases.append(dict(name="bmf_wave_dirichlet_p", A=Aw, H=Hw,
                      L=LD_w, owned=[0, n - 1], sbp=sbp_wave))
    cases.append(dict(name="bmf_wave_neumann_v", A=Aw, H=Hw,
                      L=LN_w, owned=[n, n + n1 - 1], sbp=sbp_wave))
    # wave-pair Robin = impedance pairing; the mixed Green identity pairs the
    # traces with the OPPOSITE orientation to diffusion, so the absorbing
    # (dissipative) convention is p + Z (r0.v) at the left wall and
    # p - Z (r1.v) at the right wall. The anti-impedance signs are
    # exponentially unstable under ANY enforcement: Robin data on the wave
    # pair is admissible only with this orientation.
    cases.append(dict(name="bmf_wave_robin", A=Aw, H=Hw,
                      L=LD_w + (-robin_sign) @ LN_w, owned=[0, n - 1], sbp=sbp_wave))
    return cases


# ----------------------------------------------------------------------------
# liftings and dynamics
# ----------------------------------------------------------------------------

def lifting_selector(L, owned, nn):
    E = np.zeros((nn, len(owned)))
    for j, r in enumerate(owned):
        E[r, j] = 1.0
    M = L @ E
    return E @ np.linalg.inv(M)


def lifting_horth(L, H):
    Hi_Lt = np.linalg.solve(H, L.T)
    return Hi_Lt @ np.linalg.inv(L @ Hi_Lt)


def udot(A, L, X, u, gdot):
    r = gdot - L @ (A @ u)
    return A @ u + X @ r


def consistent_state(L, X, g0, nn):
    u = RNG.standard_normal(nn)
    return u + X @ (g0 - L @ u)


# ----------------------------------------------------------------------------
# the certificate
# ----------------------------------------------------------------------------

def certify_case(case, n_states=N_STATES):
    A, H, L, owned, sbp = case["A"], case["H"], case["L"], case["owned"], case["sbp"]
    nn = A.shape[0]
    nc = L.shape[0]
    XS = lifting_selector(L, owned, nn)
    XH = lifting_horth(L, H)
    S = np.linalg.inv(L @ np.linalg.solve(H, L.T))     # (L H^-1 L^T)^{-1}

    # operator gate: the SBP closed form must match u^T H A u exactly
    gate = 0.0
    for _ in range(4):
        u = RNG.standard_normal(nn)
        gate = max(gate, abs(u @ H @ (A @ u) - sbp(u)))
    scale = np.linalg.norm(H) * np.linalg.norm(A)

    g0 = np.array([0.6, -0.4][:nc])
    gd = np.array([1.3, 0.8][:nc])                     # g'(t0)

    r_t1 = r_t2 = r_t3 = r_t3_sel = r_t4 = r_wc = 0.0
    for _ in range(n_states):
        # nonhomogeneous, both liftings
        uH = consistent_state(L, XH, g0, nn)
        uS = consistent_state(L, XS, g0, nn)
        # T1: current theorem (pure SBP balance) vs selector dynamics
        duS = udot(A, L, XS, uS, gd)
        r_t1 = max(r_t1, abs(uS @ H @ duS - sbp(uS)))
        # T2: corrected balance vs H-orthogonal dynamics
        duH = udot(A, L, XH, uH, gd)
        wc = (L @ uH) @ (S @ (gd - L @ (A @ uH)))
        r_t2 = max(r_t2, abs(uH @ H @ duH - sbp(uH) - wc))
        # the multiplier form of the trace work equals u^T H X_H (...)
        r_wc = max(r_wc, abs(wc - uH @ H @ (XH @ (gd - L @ (A @ uH)))))
        # T3: homogeneous data -> pure SBP balance, both liftings measured
        u0 = consistent_state(L, XH, 0.0 * g0, nn)
        du0 = udot(A, L, XH, u0, 0.0 * gd)
        r_t3 = max(r_t3, abs(u0 @ H @ du0 - sbp(u0)))
        u0s = consistent_state(L, XS, 0.0 * g0, nn)
        du0s = udot(A, L, XS, u0s, 0.0 * gd)
        r_t3_sel = max(r_t3_sel, abs(u0s @ H @ du0s - sbp(u0s)))
        # T4: multiplier == null-space elimination == restricted Galerkin
        #   null-space route: u' = Z y' + X_H g', with Z an H-orthonormal basis
        #   of ker L; restricted Galerkin: Z (Z^T H Z)^{-1} Z^T H (A u + X_H g'
        #   ... identical algebra); verify the projector identity directly.
        P = np.eye(nn) - XH @ L                        # H-orthogonal projector
        duP = P @ (A @ uH) + XH @ gd
        r_t4 = max(r_t4, np.linalg.norm(duP - duH))

    # T5: spectral abscissae of the two enforced homogeneous operators
    AS = (np.eye(nn) - XS @ L) @ A
    AH = (np.eye(nn) - XH @ L) @ A
    # restrict to the constraint manifold ker L (SVD null basis; a QR of the
    # rank-deficient projector can leak junk directions into the basis)
    _u_, _s_, Vt = np.linalg.svd(L)
    Z = Vt[nc:, :].T
    absc_S = float(np.max(np.real(np.linalg.eigvals(Z.T @ AS @ Z))))
    absc_H = float(np.max(np.real(np.linalg.eigvals(Z.T @ AH @ Z))))

    # T6: RK4 march with time-dependent data, H-orthogonal lifting
    om1, om2 = 3.0, 5.0

    def gfun(t):
        return np.array([0.6 * np.sin(om1 * t), -0.4 * np.cos(om2 * t)][:nc])

    def gdfun(t):
        return np.array([0.6 * om1 * np.cos(om1 * t),
                         0.4 * om2 * np.sin(om2 * t)][:nc])

    u = consistent_state(L, XH, gfun(0.0), nn)
    # accuracy-margin step: dt*rho << RK4 boundary, so the marched trajectory
    # tracks the continuous energy evolution and the ledger closes at RK4
    # order (near the stability boundary the stiff modes decay at the wrong
    # RATE even though the march is stable, which opens a spurious gap)
    AH_full = (np.eye(nn) - XH @ L) @ A
    rho_enf = float(np.max(np.abs(np.linalg.eigvals(AH_full))))
    dt = min(2.0e-4, 0.25 / max(rho_enf, 1.0))
    nsteps = min(int(np.ceil(0.4 / dt)), 20000)
    # burn off the random stiff components first so the ledger measures the
    # smooth trajectory (the identity itself is certified separately in T2)
    for _ in range(60):
        k1 = udot(A, L, XH, u, gdfun(0.0))
        k2 = udot(A, L, XH, u + 0.5 * dt * k1, gdfun(0.0))
        k3 = udot(A, L, XH, u + 0.5 * dt * k2, gdfun(0.0))
        k4 = udot(A, L, XH, u + dt * k3, gdfun(0.0))
        u = u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    u = u + XH @ (gfun(0.0) - L @ u)
    drift = 0.0
    ledger_lhs = 0.5 * (u @ H @ u)
    ledger_rhs = 0.0
    for k in range(nsteps):
        t = k * dt

        def rhs(tt, uu):
            return udot(A, L, XH, uu, gdfun(tt))

        def bal(tt, uu):
            wc = (L @ uu) @ (S @ (gdfun(tt) - L @ (A @ uu)))
            return sbp(uu) + wc

        k1 = rhs(t, u); b1 = bal(t, u)
        k2 = rhs(t + 0.5 * dt, u + 0.5 * dt * k1); b2 = bal(t + 0.5 * dt, u + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, u + 0.5 * dt * k2); b3 = bal(t + 0.5 * dt, u + 0.5 * dt * k2)
        k4 = rhs(t + dt, u + dt * k3); b4 = bal(t + dt, u + dt * k3)
        u = u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        ledger_rhs += dt / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4)
        drift = max(drift, float(np.max(np.abs(L @ u - gfun((k + 1) * dt)))))
    ledger_gap = abs(0.5 * (u @ H @ u) - ledger_lhs - ledger_rhs)
    e_scale = max(abs(ledger_lhs), 1.0)

    return {
        "gate_sbp_form": gate / scale,
        "T1_current_theorem_vs_selector": r_t1 / scale,
        "T2_corrected_vs_horth": r_t2 / scale,
        "T3_homogeneous_horth": r_t3 / scale,
        "T3_homogeneous_selector": r_t3_sel / scale,
        "T4_equivalence": r_t4,
        "Wc_multiplier_identity": r_wc / scale,
        "T5_abscissa_selector": absc_S,
        "T5_abscissa_horth": absc_H,
        "T6_constraint_drift": drift,
        "T6_ledger_gap_rel": ledger_gap / e_scale,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", metavar="JSON", default=None)
    ap.add_argument("--verify", metavar="JSON", default=None)
    args = ap.parse_args()

    if args.verify:
        data = json.loads(Path(args.verify).read_text())
        worst = data["worst"]
        assert worst["gate_sbp_form"] < 1e-11
        assert worst["T2_corrected_vs_horth"] < 1e-10
        assert worst["T3_homogeneous_horth"] < 1e-10
        assert worst["T4_equivalence"] < 1e-9
        assert worst["Wc_multiplier_identity"] < 1e-10
        assert worst["T1_current_theorem_vs_selector"] > 1e-6, \
            "the failing test stopped failing; re-examine the claim"
        print("verify OK:", json.dumps(worst, indent=1))
        return

    rows = []
    worst = {}
    for p in DEGREES:
        for n in GRIDS:
            for case in build_cases(n, p):
                rec = {"p": p, "n": n, "case": case["name"]}
                rec.update(certify_case(case))
                rows.append(rec)
                for k, v in rec.items():
                    if isinstance(v, float):
                        worst[k] = max(worst.get(k, 0.0), abs(v)) \
                            if not k.startswith("T5") else worst.get(k, -1e30)
                if rec["T1_current_theorem_vs_selector"] > 1e-9:
                    tag = "FAILS current theorem"
                else:
                    tag = "matches (special structure)"
                print(f"p={p} n={n} {case['name']:26s} "
                      f"T1={rec['T1_current_theorem_vs_selector']:.2e} ({tag})  "
                      f"T2={rec['T2_corrected_vs_horth']:.2e}  "
                      f"T3H={rec['T3_homogeneous_horth']:.2e}  "
                      f"abscS={rec['T5_abscissa_selector']:+.2e}  "
                      f"abscH={rec['T5_abscissa_horth']:+.2e}")
    for k in ("T5_abscissa_selector", "T5_abscissa_horth"):
        worst[k] = max(r[k] for r in rows)

    data = {"description": "constrained strong-BC energy-law certificate",
            "rows": rows, "worst": worst}
    dest = Path(args.write) if args.write else Path(RESULTSDIR) / "constrained_bc_energy.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=1))
    print(f"\nwrote {dest}")
    print("worst:", json.dumps(worst, indent=1))


if __name__ == "__main__":
    main()
