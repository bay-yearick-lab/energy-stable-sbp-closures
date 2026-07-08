"""Full Nu(t) time series at a coarse wall resolution for the energy-stability
cliff figure: compact family diverges exponentially, spline + boundary-optimized
SBP stay flat. Reuses the stationary-fix driver path (warm-start, no fixed
spinup). Saves the full downsampled series per scheme to results/nusselt_cliff_series.json.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from nusselt_driver import (load_meta, interp_z, nusselt_run, warm_theta0, SNAPDIR, RESULTS)
from nusselt_driver import load_snap
from scalar3d import ScalarChannel3D

NZ = 17
SCHEMES = ["I", "C", "C6-D2", "SBP", "SBP6-BO"]
Pr = 0.71
nu = 1.0 / 180.0
alpha = nu / Pr

m = load_meta()
Nx, Ny, Lx, Ly, Lz = m["Nx"], m["Ny"], m["Lx"], m["Ly"], m["Lz"]
zuni = np.array(m["zuni"])
dt_snap = m["dt_snap"]
dt = dt_snap / m["stride"]
nsnap = len([f for f in os.listdir(SNAPDIR) if f.startswith("snap_") and f.endswith(".bin")])
raw = [load_snap(k, m) for k in range(nsnap)]
T = dt_snap * (nsnap - 1)

out = {"nz": NZ, "Pr": Pr, "Re_tau": 180, "dt_snap": dt_snap, "T": T,
       "flow_through_unit": Lz, "schemes": {}}

for kind in SCHEMES:
    S = ScalarChannel3D(kind, 6, Nx, Ny, NZ, Lx, Ly, Lz, alpha)
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

    theta0 = warm_theta0(z, Lz, Nx, Ny)
    Nu, zg, prof, series = nusselt_run(kind, vel_at, Nx, Ny, NZ, Lx, Ly, Lz, alpha,
                                       dt, T=T, spinup=0.0, order=6,
                                       return_series=True, theta0=theta0)
    series = np.asarray(series)
    # downsample to <=400 points for a clean figure; keep time axis in flow-throughs
    n = len(series)
    idx = np.linspace(0, n - 1, min(n, 400)).astype(int)
    t_axis = (idx * dt) / Lz            # flow-through units (Lz/u_bulk ~ 1)
    out["schemes"][kind] = {
        "t_flowthrough": t_axis.tolist(),
        "Nu_t": series[idx].tolist(),
        "Nu_final": float(Nu),
        "diverged": bool(not np.isfinite(series[-1]) or abs(series[-1]) > 1e3),
    }
    print(f"{kind:8s} nz={NZ}: Nu_final={Nu:.4g}  last={series[-1]:.4g}  "
          f"diverged={out['schemes'][kind]['diverged']}", flush=True)

with open(os.path.join(RESULTS, "nusselt_cliff_series.json"), "w") as f:
    json.dump(out, f)
print("wrote results/nusselt_cliff_series.json", flush=True)
