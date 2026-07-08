#!/usr/bin/env python3
"""S6.7 turbulent wall heat-transfer head-to-head. The developed spline velocity
field (snapshots from snap_vel) advects a passive scalar with isothermal walls.
Every scheme shares that velocity and the Fourier x,y treatment; only the
wall-normal operator differs, so the Nusselt number isolates the wall closure.

Run modes:
  python nusselt_driver.py --self-test     # synthetic div-free field, sanity only
  python nusselt_driver.py                  # real: read code/dns/data/snap/

theta_total = theta_b(z) + theta', theta_b = z/Lz (isothermal walls 0 and 1),
source on the fluctuation = -w * dtheta_b/dz = -w/Lz. Nusselt = wall gradient
normalised by the pure-conduction gradient (1/Lz), so laminar conduction gives 1.
"""
from __future__ import annotations
import sys, os, struct, glob
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import numpy as np
from scalar3d import ScalarChannel3D

SNAPDIR = os.path.join(os.path.dirname(__file__), "dns", "data", "snap")

import json
try:
    from project_paths import RESULTSDIR, ensure_generated_dirs
    ensure_generated_dirs()
    RESULTS = str(RESULTSDIR)
except Exception:
    RESULTS = os.path.dirname(__file__)


def write_results(out):
    p = os.path.join(RESULTS, "turbulent_nusselt.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {p}")


# ----------------------------- snapshot I/O -----------------------------
def load_meta(d=SNAPDIR):
    with open(os.path.join(d, "meta.bin"), "rb") as f:
        Nx, Ny, Nzf = struct.unpack("iii", f.read(12))
        Lx, Ly, Lz, dt_snap = struct.unpack("dddd", f.read(32))
        stride, nsnap = struct.unpack("ii", f.read(8))
        zuni = np.frombuffer(f.read(8 * Nzf), dtype=np.float64).copy()
    return dict(Nx=Nx, Ny=Ny, Nzf=Nzf, Lx=Lx, Ly=Ly, Lz=Lz, dt_snap=dt_snap,
               stride=stride, nsnap=nsnap, zuni=zuni)


def load_snap(k, m, d=SNAPDIR):
    Nx, Ny, Nzf = m["Nx"], m["Ny"], m["Nzf"]
    n = Nx * Ny * Nzf
    with open(os.path.join(d, f"snap_{k:04d}.bin"), "rb") as f:
        a = np.frombuffer(f.read(8 * 3 * n), dtype=np.float64)
    u = a[:n].reshape(Nx, Ny, Nzf)
    v = a[n:2 * n].reshape(Nx, Ny, Nzf)
    w = a[2 * n:].reshape(Nx, Ny, Nzf)
    return u, v, w


def interp_z(field, zsrc, ztgt):
    """interpolate (Nx,Ny,Nzf) on zsrc to (Nx,Ny,nz) on ztgt, per column."""
    Nx, Ny, _ = field.shape
    out = np.empty((Nx, Ny, len(ztgt)))
    fr = field.reshape(Nx * Ny, -1)
    o = out.reshape(Nx * Ny, -1)
    for c in range(Nx * Ny):
        o[c] = np.interp(ztgt, zsrc, fr[c])
    return out


# ----------------------------- warm start -----------------------------
def warm_theta0(z, Lz, Nx, Ny, profile_json=None):
    """Initial scalar FLUCTUATION field from a previously computed turbulent mean
    profile: theta'(x,y,z) = <theta>(z) - z/Lz, uniform
    in x,y. Starting the relaxation from the turbulent mean profile instead of the
    conduction profile removes most of the O(10 time-unit) transient. Source profile: the spline n_z=33 time-mean from
    results/turbulent_nusselt.json (any reasonable developed profile serves; the
    stationarity gate still decides when averaging opens)."""
    if profile_json is None:
        profile_json = os.path.join(RESULTS, "turbulent_nusselt.json")
    with open(profile_json) as f:
        d = json.load(f)
    zp = np.array(d["schemes"]["I"]["z"])
    pp = np.array(d["schemes"]["I"]["profile"])
    prof = np.interp(z, zp, pp)
    thp = prof - z / Lz
    thp[0] = 0.0
    thp[-1] = 0.0
    return np.broadcast_to(thp[None, None, :], (Nx, Ny, len(z))).copy()


# ----------------------------- stationarity gate -----------------------------
def stationarity_gate(series, n_batches=8, alpha=0.05,
                      candidates=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)):
    """Post-hoc stationary-window selection (task-3 fix): for each candidate
    start fraction of the full Nu(t) series, split the remainder into n_batches
    contiguous batch means and OLS-regress them on batch index; the averaging
    window opens at the EARLIEST start whose slope is statistically
    indistinguishable from zero (two-sided t-test via scipy linregress,
    df = n_batches-2, p > alpha). Returns diagnostics for every candidate and
    passed=False if no candidate passes."""
    from scipy import stats
    n = len(series)
    series = np.asarray(series)
    diag = []
    chosen = None
    for c in candidates:
        i0 = int(round(c * n))
        seg = series[i0:]
        if len(seg) < 4 * n_batches:
            continue
        edges = np.linspace(0, len(seg), n_batches + 1).astype(int)
        bm = np.array([seg[edges[i]:edges[i + 1]].mean() for i in range(n_batches)])
        x = np.arange(n_batches, dtype=float)
        res = stats.linregress(x, bm)
        rec = dict(start_frac=float(c), i0=int(i0), slope=float(res.slope),
                   slope_p=float(res.pvalue), batch_means=bm.tolist(),
                   mean=float(bm.mean()))
        diag.append(rec)
        if chosen is None and res.pvalue > alpha:
            chosen = rec
    out = dict(passed=chosen is not None, alpha=alpha, n_batches=n_batches,
               candidates=diag)
    if chosen is not None:
        out.update(start_frac=chosen["start_frac"], i0=chosen["i0"],
                   slope=chosen["slope"], slope_p=chosen["slope_p"])
    return out


# ----------------------------- Nusselt run -----------------------------
def nusselt_run(kind, vel_at, Nx, Ny, nz, Lx, Ly, Lz, alpha, dt, T, spinup, order=6,
                 return_series=False, theta0=None):
    S = ScalarChannel3D(kind, order, Nx, Ny, nz, Lx, Ly, Lz, alpha)
    z = S.z
    base = (z / Lz)[None, None, :]
    th = theta0.copy() if theta0 is not None else np.zeros((Nx, Ny, nz))
    nstep = int(round(T / dt))
    nspin = int(round(spinup / dt))
    acc, cnt = 0.0, 0
    prof = np.zeros(nz)
    series = [] if return_series else None
    for n in range(nstep):
        u, v, w = vel_at(n * dt, S.z)
        src = -w / Lz
        th = S.step(u, v, w, th, dt, src)
        if n >= nspin:
            b, t = S.wall_flux(th + base)
            grad_n = 0.5 * (abs(b) + abs(t))
            acc += grad_n
            prof += (th + base).mean(axis=(0, 1))
            cnt += 1
            if return_series:
                series.append(grad_n * Lz)
    grad = acc / max(cnt, 1)
    if return_series:
        return grad * Lz, z, prof / max(cnt, 1), np.array(series)
    return grad * Lz, z, prof / max(cnt, 1)           # Nu = wall gradient / (1/Lz)


# ----------------------------- self test -----------------------------
def self_test():
    Nx = Ny = 16
    Lx = Ly = 2 * np.pi
    Lz = 2.0
    alpha = 0.05

    def make_vel_at(Lz):
        # steady analytic divergence-free field psi=sin(x)sin(pi z/Lz)
        def vel_at(t, z):
            nz = len(z)
            x = (np.arange(Nx) * Lx / Nx)[:, None, None]
            zz = z[None, None, :]
            u = (np.pi / Lz) * np.sin(x) * np.cos(np.pi * zz / Lz) * np.ones((Nx, Ny, nz))
            v = np.zeros((Nx, Ny, nz))
            w = -np.cos(x) * np.sin(np.pi * zz / Lz) * np.ones((Nx, Ny, nz))
            return u, v, w
        return vel_at

    print(f"{'scheme':6s} {'nz':>4s} {'Nu (sanity)':>14s}")
    out = {"mode": "self-test", "Pr": None, "Re_tau": None, "schemes": {}}
    for kind in ["I", "C", "SBP"]:
        S = ScalarChannel3D(kind, 6, Nx, Ny, 33, Lx, Ly, Lz, alpha)
        nz = len(S.z)
        Nu, z, prof = nusselt_run(kind, make_vel_at(Lz), Nx, Ny, nz, Lx, Ly, Lz,
                                  alpha, dt=2e-3, T=1.5, spinup=0.8)
        out["schemes"][kind] = {"Nu": float(Nu), "z": z.tolist(), "profile": prof.tolist()}
        print(f"{kind:6s} {nz:4d} {Nu:14.4f}   {'finite OK' if np.isfinite(Nu) else 'NON-FINITE'}")
    write_results(out)


# ----------------------------- real run -----------------------------
def real_run():
    m = load_meta()
    print(f"meta: {m['Nx']}x{m['Ny']}, Nzf={m['Nzf']}, nsnap={m['nsnap']}, dt_snap={m['dt_snap']:.4f}")
    nsnap = len(sorted(glob.glob(os.path.join(SNAPDIR, "snap_*.bin"))))
    zuni = m["zuni"]
    Nx, Ny, Lx, Ly, Lz = m["Nx"], m["Ny"], m["Lx"], m["Ly"], m["Lz"]
    Pr = 0.71
    nu = 1.0 / 180.0
    alpha = nu / Pr
    dt = m["dt_snap"] / m["stride"]                    # velocity sub-step
    dt_snap = m["dt_snap"]
    raw = [load_snap(k, m) for k in range(nsnap)]      # all snapshots on the uniform z grid

    def run_one(kind, nz):
        S = ScalarChannel3D(kind, 6, Nx, Ny, nz, Lx, Ly, Lz, alpha)
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
        return nusselt_run(kind, vel_at, Nx, Ny, len(z), Lx, Ly, Lz, alpha, dt,
                           T=T, spinup=0.3 * T)

    NZ_COARSE = int(os.getenv("NZ_COARSE", "33"))     # coarse wall grid (the claim)
    NZ_FINE = int(os.getenv("NZ_FINE", "96"))         # grid-converged reference
    refNu, _, _ = run_one("I", NZ_FINE)               # own converged Nusselt (spline, fine)
    print(f"  reference (spline nz={NZ_FINE}): Nu = {refNu:.4f}")
    out = {"mode": "real", "Pr": Pr, "Re_tau": 1.0 / nu, "nz_coarse": NZ_COARSE,
           "nz_fine": NZ_FINE, "reference_Nu": float(refNu), "schemes": {}}
    for kind in ["I", "C", "SBP"]:
        Nu, zg, prof = run_one(kind, NZ_COARSE)
        out["schemes"][kind] = {"Nu": float(Nu), "z": zg.tolist(), "profile": prof.tolist()}
        print(f"  {kind} (nz={NZ_COARSE}): Nu = {Nu:.4f}  (ref {refNu:.4f}, "
              f"err {100*(Nu-refNu)/refNu:+.1f}%)")
    write_results(out)
    return out


if __name__ == "__main__":
    if "--self-test" in sys.argv or not os.path.exists(os.path.join(SNAPDIR, "meta.bin")):
        print("[self-test: synthetic field]")
        self_test()
    else:
        real_run()
