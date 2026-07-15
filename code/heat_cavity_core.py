"""Heat-cavity core: 2D heat equation in a rigid-wall cavity.

The PDE is the scalar heat equation on the unit square,

    u_t = nu ( u_xx + u_yy ),     (x, y) in (0, 1)^2,

with homogeneous Dirichlet walls

    u(0, y, t) = u(1, y, t) = u(x, 0, t) = u(x, 1, t) = 0,

and a sine-sine standing-wave family of exact solutions

    u_mn(x, y, t) = exp(-nu pi^2 (m^2 + n^2) t) * sin(m pi x) sin(n pi y).

Both sin(m pi x) and sin(n pi y) are identically zero at their own walls,
so strong Dirichlet enforcement is trivially consistent and the
boundary is neither a source of stability concerns nor a forcing term.
The only ingredient that varies across schemes is the tensor-product
discrete Laplacian built from each scheme's one-dimensional
second-derivative operator,

    Lap_h U = D2 @ U + U @ D2.T,

and the benchmark is restricted to the families in this repository that
provide such an operator: SBP, BMF, and BMG. This is where the wall
closure shows up. For BMF the operator is the staggered flux-divergence
closure D_BMF @ G_BMF proved in Section 5. In the code API those matrices are
stored as d01 and d10. For BMG the
operator is the compatible natural-metric Galerkin closure proved in
Section 6. The direct sampled curvature map and the weak-curvature
Galerkin map remain available separately as algebraic reconstruction
objects and are not used as the scheme-level diffusion operators.
"""

from __future__ import annotations

import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from boundary_spline_analysis import (
    Array,
    Operator1D,
    build_operator,
    scheme_family,
    scheme_kind,
    scheme_order,
    second_derivative_matrix,
    trapezoidal_weights,
)


def diffusion_second_derivative(op: Operator1D, scheme_label: str) -> Array:
    """Return the second-derivative operator used by the heat solver.

    The SBP-closed families (`SBP_q`, `BMF_p`, `BMG_p`) supply a scheme-level
    second derivative through their `second_derivative_matrix()` method:
    SBP_q returns the standard SBP D2 closure, BMF_p returns the staggered
    flux-divergence composition `D01 D10`, and BMG_p returns the natural-
    metric Galerkin closure proven in Section 5. The non-SBP families
    (`E_q`, `C_q`, `DRP_q`) do not supply a scheme-level D2; for them the
    natural same-grid second derivative is the composition `D D` of the
    first-derivative matrix with itself, which is the operator those
    families would otherwise use without a tailored boundary closure.
    """
    family = scheme_family(scheme_label)
    if family in ('SBP', 'BMF', 'BMG'):
        return second_derivative_matrix(op)
    if family in ('E', 'C', 'DRP'):
        D = op.matrix
        return D @ D
    raise ValueError(f"unknown scheme family: {family} for label {scheme_label}")


def diffusion_second_derivative_extended(op: Operator1D, scheme_label: str) -> Array:
    """Task-2 addendum: composed-DD vs tailored D2 for the strengthened competitors.

    `C6-D2` uses the published compact second-derivative closure (via
    `CompactD2FD.second_derivative_matrix()`); `SBP6-BO` uses the compatible
    SBP second derivative built from its own (D, H); `C6F` has no tailored
    second derivative (the filter operates on the state, not the Laplacian),
    so it falls back to composed DD, same as bare C6. `diffusion_second_derivative`
    above is untouched; this is a new, separate entry point.
    """
    if scheme_label in ('C6-D2', 'SBP6-BO'):
        return op.second_derivative_matrix()
    if scheme_label == 'C6F':
        D = op.matrix
        return D @ D
    return diffusion_second_derivative(op, scheme_label)


def run_heat_cavity_extended(
    scheme_label: str,
    modes: HeatSineMode,
    *,
    npts: int = 65,
    t_end: float = 3.0,
    cfl: float = 0.1,
    dt: float | None = None,
    record_count: int = 200,
    snapshot_count: int = 81,
):
    """Same evolution as `run_heat_cavity`, routed through
    `build_operator_extended`/`diffusion_second_derivative_extended` for the
    Task-2 competitor labels ('C6-D2', 'SBP6-BO', 'C6F'); falls back to the
    ordinary path for the original six-family labels.
    """
    import boundary_spline_analysis as ba

    if scheme_label in ('C6-D2', 'SBP6-BO', 'C6F'):
        op = ba.build_operator_extended(scheme_label, 6, npts)
    else:
        kind = scheme_kind(scheme_label)
        order = scheme_order(scheme_label)
        op = build_operator(kind, order, npts)
    D = diffusion_second_derivative_extended(op, scheme_label)
    x = op.x
    y = x.copy()
    nu = float(modes.viscosity)
    # extended labels are all nodal (diagonal metric): the fast path IS the
    # metric-orthogonal enforcement; spline labels get the dense projector
    proj = make_wall_projector(
        scheme_metric_1d(scheme_label, npts)
        if scheme_label not in ('C6-D2', 'SBP6-BO', 'C6F') else None,
        npts)

    if dt is not None:
        nsteps = max(1, int(np.ceil(t_end / float(dt))))
        dt = t_end / nsteps
    else:
        h = float(x[1] - x[0])
        dt0 = cfl * h * h / max(nu, 1.0e-30)
        nsteps = int(np.ceil(t_end / dt0))
        dt = t_end / nsteps

    U = modes.field_grid(x, y, 0.0)
    enforce_zero_walls(U)
    w2d = np.outer(node_trapezoid_weights(x), node_trapezoid_weights(y))

    rec_every = max(1, nsteps // max(2, record_count - 1))
    snap_every = max(1, nsteps // max(2, snapshot_count - 1))

    rec_t = [0.0]
    rec_u_l2 = [0.0]
    rec_u_inf = [0.0]
    rec_dec = [0.0]
    snap_t = [0.0]
    snap_u = [U.copy()]
    snap_uex = [U.copy()]

    def anti_node():
        mpeak, npeak = modes.peak_mode()
        ix = int(round((npts - 1) / (2.0 * mpeak)))
        iy = int(round((npts - 1) / (2.0 * npeak)))
        return ix, iy

    ix, iy = anti_node()
    t = 0.0
    for step in range(nsteps):
        U = rk4_step_horth(D, U, dt, nu, proj)
        t += dt
        is_last = step == nsteps - 1
        if is_last or (step + 1) % rec_every == 0:
            u_ex = modes.field_grid(x, y, t)
            err = U - u_ex
            rec_t.append(t)
            rec_u_l2.append(_l2_rel(err, u_ex, w2d))
            rec_u_inf.append(float(np.max(np.abs(err))))
            rec_dec.append(abs(float(U[ix, iy]) - float(u_ex[ix, iy])))
        if is_last or (step + 1) % snap_every == 0:
            snap_t.append(t)
            snap_u.append(U.copy())
            snap_uex.append(modes.field_grid(x, y, t))

    peak = modes.peak_mode()
    return HeatCavityResult(
        scheme=scheme_label,
        npts=int(npts),
        times=np.asarray(rec_t),
        u_rel_l2=np.asarray(rec_u_l2),
        u_linf=np.asarray(rec_u_inf),
        decay_error=np.asarray(rec_dec),
        snapshot_times=np.asarray(snap_t),
        snapshot_u=np.stack(snap_u, axis=0),
        snapshot_u_exact=np.stack(snap_uex, axis=0),
        final_u=U.copy(),
        final_u_exact=modes.field_grid(x, y, t),
        peak_mode=peak,
        ppw=ppw_for(max(peak), npts, 1.0),
        steps=int(nsteps),
        dt=float(dt),
        t_end=float(t_end),
    )


@dataclass(frozen=True)
class HeatSineMode:
    modes_m: Array
    modes_n: Array
    amplitudes: Array
    viscosity: float = 1.0e-4

    def decay(self, t: float) -> Array:
        return np.exp(-self.viscosity * np.pi * np.pi
                      * (self.modes_m.astype(float) ** 2
                         + self.modes_n.astype(float) ** 2) * float(t))

    def field_grid(self, x: Array, y: Array, t: float) -> Array:
        mpi = self.modes_m.astype(float) * np.pi
        npi = self.modes_n.astype(float) * np.pi
        sx = np.sin(mpi[:, None] * x[None, :])
        sy = np.sin(npi[:, None] * y[None, :])
        return np.einsum('k,k,ki,kj->ij', self.amplitudes, self.decay(t), sx, sy)

    def derivative_x_grid(self, x: Array, y: Array, t: float) -> Array:
        mpi = self.modes_m.astype(float) * np.pi
        npi = self.modes_n.astype(float) * np.pi
        cx = np.cos(mpi[:, None] * x[None, :])
        sy = np.sin(npi[:, None] * y[None, :])
        return np.einsum('k,k,k,ki,kj->ij', self.amplitudes * mpi, self.decay(t),
                         np.ones(self.amplitudes.size), cx, sy)

    def peak_mode(self) -> tuple[int, int]:
        idx = int(np.argmax(np.abs(self.amplitudes)))
        return int(self.modes_m[idx]), int(self.modes_n[idx])


def default_heat_modes(
    m_center: int = 24,
    n_center: int = 24,
    half_band: int = 2,
    viscosity: float = 1.0e-4,
    seed: int = 0,
) -> HeatSineMode:
    rng = np.random.default_rng(int(seed))
    ms: list[int] = []
    ns: list[int] = []
    amps: list[float] = []
    for m in range(max(1, m_center - half_band), m_center + half_band + 1):
        for n in range(max(1, n_center - half_band), n_center + half_band + 1):
            w = np.exp(-(((m - m_center) / max(half_band, 1)) ** 2
                         + ((n - n_center) / max(half_band, 1)) ** 2))
            jitter = 1.0 + 0.05 * rng.standard_normal()
            ms.append(m)
            ns.append(n)
            amps.append(float(w * jitter))
    ms_arr = np.asarray(ms, dtype=int)
    ns_arr = np.asarray(ns, dtype=int)
    a_arr = np.asarray(amps, dtype=float)
    a_arr /= np.max(np.abs(a_arr))
    return HeatSineMode(modes_m=ms_arr, modes_n=ns_arr, amplitudes=a_arr,
                        viscosity=float(viscosity))


def enforce_zero_walls(field: Array) -> None:
    field[0, :] = 0.0
    field[-1, :] = 0.0
    field[:, 0] = 0.0
    field[:, -1] = 0.0


def scheme_metric_1d(scheme_label: str, npts: int) -> Array | None:
    """1D energy metric of the scheme, or None when it is diagonal.

    For a DIAGONAL metric the H-orthogonal boundary projection of a
    wall-value constraint coincides with zeroing the wall rows, so the
    fast path (enforce_zero_walls) is already the metric-orthogonal
    enforcement of Definition def:strong-enforcement. Only the two dense
    spline metrics need the explicit projector.
    """
    from boundary_spline_analysis import (bmf_staggered_metric_matrices,
                                          bmg_nodal_operator)
    fam = scheme_family(scheme_label)
    order = scheme_order(scheme_label)
    if fam == 'BMG':
        return bmg_nodal_operator(npts, order)[2].copy()
    if fam == 'BMF':
        return bmf_staggered_metric_matrices(npts, order)[0].copy()
    return None


def make_wall_projector(metric: Array | None, npts: int):
    """Return the H-orthogonal projector onto {residuals with zero wall
    values} for the tensor metric H (x) H, as a callable on N x N residual
    matrices. metric=None returns the diagonal fast path."""
    if metric is None:
        def proj_diag(R: Array) -> Array:
            out = R.copy()
            enforce_zero_walls(out)
            return out
        return proj_diag

    Hi = np.linalg.inv(metric)
    idx = np.zeros((npts, npts), dtype=bool)
    idx[0, :] = idx[-1, :] = idx[:, 0] = idx[:, -1] = True
    bi, bj = np.nonzero(idx)
    # M_b[a, b] = (Hi (x) Hi) restricted to the boundary node set
    Mb = Hi[np.ix_(bi, bi)] * Hi[np.ix_(bj, bj)]
    from scipy.linalg import cho_factor, cho_solve
    cho = cho_factor(0.5 * (Mb + Mb.T))

    def proj(R: Array) -> Array:
        b = R[bi, bj]
        y = cho_solve(cho, b)
        Y = np.zeros_like(R)
        Y[bi, bj] = y
        return R - Hi @ Y @ Hi.T

    return proj


def rk4_step_horth(D: Array, U: Array, dt: float, nu: float, proj) -> Array:
    """RK4 on the metric-orthogonal constrained flow u' = Pi (nu Lap u).

    Every stage derivative is projected, so the wall values of the state
    are invariant by construction; the final enforcement is roundoff
    hygiene only.
    """
    k1 = proj(heat_rhs(D, U, nu))
    k2 = proj(heat_rhs(D, U + 0.5 * dt * k1, nu))
    k3 = proj(heat_rhs(D, U + 0.5 * dt * k2, nu))
    k4 = proj(heat_rhs(D, U + dt * k3, nu))
    U_new = U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    enforce_zero_walls(U_new)
    return U_new


def per_method_diffusion_dt_extended(
    scheme_label: str,
    *,
    npts: int,
    viscosity: float,
    t_end: float,
    safety: float = 0.4,
) -> float:
    """per_method_diffusion_dt for the extended competitor labels
    ('C6-D2', 'SBP6-BO', 'C6F'), which are nodal (diagonal metric)."""
    if scheme_label not in ('C6-D2', 'SBP6-BO', 'C6F'):
        return per_method_diffusion_dt(scheme_label, npts=npts,
                                       viscosity=viscosity, t_end=t_end,
                                       safety=safety)
    import boundary_spline_analysis as ba
    op = ba.build_operator_extended(scheme_label, 6, npts)
    D = diffusion_second_derivative_extended(op, scheme_label)
    proj = make_wall_projector(None, npts)
    rng = np.random.default_rng(3)
    V = proj(rng.standard_normal((npts, npts)))
    lam = 1.0
    for _ in range(160):
        W = proj(heat_rhs(D, V, viscosity))
        lam = float(np.linalg.norm(W))
        if lam == 0.0:
            break
        V = W / lam
    dt0 = safety * 2.785 / max(lam, 1.0e-30)
    nsteps = max(1, int(np.ceil(t_end / dt0)))
    return float(t_end) / nsteps


def per_method_diffusion_dt(
    scheme_label: str,
    *,
    npts: int,
    viscosity: float,
    t_end: float,
    safety: float = 0.4,
) -> float:
    """Per-method RK4 step: safety x (RK4 real-axis bound) / rho of the
    ENFORCED 2D diffusion operator, measured by power iteration on the
    exact marched map. This is the tier-B asymptotic step of the operator
    each method actually marches; a shared step that violates one method's
    bound measures cost, never spatial instability."""
    op = build_operator(scheme_kind(scheme_label), scheme_order(scheme_label), npts)
    D = diffusion_second_derivative(op, scheme_label)
    proj = make_wall_projector(scheme_metric_1d(scheme_label, npts), npts)
    rng = np.random.default_rng(3)
    V = proj(rng.standard_normal((npts, npts)))
    lam = 1.0
    for _ in range(160):
        W = proj(heat_rhs(D, V, viscosity))
        lam = float(np.linalg.norm(W))
        if lam == 0.0:
            break
        V = W / lam
    dt0 = safety * 2.785 / max(lam, 1.0e-30)
    nsteps = max(1, int(np.ceil(t_end / dt0)))
    return float(t_end) / nsteps


def laplacian_h(D: Array, field: Array) -> Array:
    return (D @ field) + (field @ D.T)


def heat_rhs(D: Array, U: Array, nu: float) -> Array:
    return nu * laplacian_h(D, U)


def rk4_step(D: Array, U: Array, dt: float, nu: float) -> Array:
    k1 = heat_rhs(D, U, nu)

    U1 = U + 0.5 * dt * k1
    enforce_zero_walls(U1)
    k2 = heat_rhs(D, U1, nu)

    U2 = U + 0.5 * dt * k2
    enforce_zero_walls(U2)
    k3 = heat_rhs(D, U2, nu)

    U3 = U + dt * k3
    enforce_zero_walls(U3)
    k4 = heat_rhs(D, U3, nu)

    U_new = U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    enforce_zero_walls(U_new)
    return U_new


def _trap2d(npts: int, length: float = 1.0) -> Array:
    w = trapezoidal_weights(npts, length)
    return np.outer(w, w)


def node_trapezoid_weights(nodes: Array) -> Array:
    """Composite-trapezoid weights for the actual, possibly nonuniform, nodes."""
    x = np.asarray(nodes, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("nodes must be a finite one-dimensional array of length at least two")
    dx = np.diff(x)
    if np.any(dx <= 0.0):
        raise ValueError("nodes must be strictly increasing")
    weights = np.empty_like(x)
    weights[0] = 0.5 * dx[0]
    weights[-1] = 0.5 * dx[-1]
    weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    return weights


def _l2_rel(err: Array, ref: Array, w2d: Array) -> float:
    num = float(np.sqrt(np.sum(w2d * err ** 2)))
    den = float(np.sqrt(np.sum(w2d * ref ** 2)))
    return num / den if den > 0.0 else num


def ppw_for(mode_peak: int, npts: int, length: float = 1.0) -> float:
    h = length / (npts - 1)
    wavelength = 2.0 / mode_peak
    return float(wavelength / h)


@dataclass
class HeatCavityResult:
    scheme: str
    npts: int
    times: Array
    u_rel_l2: Array
    u_linf: Array
    decay_error: Array            # |peak(U_num) - decay_exact|
    snapshot_times: Array
    snapshot_u: Array             # (F, Nx, Ny)
    snapshot_u_exact: Array       # (F, Nx, Ny)
    final_u: Array
    final_u_exact: Array
    peak_mode: tuple[int, int]
    ppw: float
    steps: int
    dt: float
    t_end: float


def run_heat_cavity(
    scheme_label: str,
    modes: HeatSineMode,
    *,
    npts: int = 65,
    t_end: float = 3.0,
    cfl: float = 0.1,
    dt: float | None = None,
    record_count: int = 200,
    snapshot_count: int = 81,
    enforcement: str = 'horth',
) -> HeatCavityResult:
    kind = scheme_kind(scheme_label)
    order = scheme_order(scheme_label)
    op = build_operator(kind, order, npts)
    D = diffusion_second_derivative(op, scheme_label)
    x = op.x
    y = x.copy()
    nu = float(modes.viscosity)
    # metric-orthogonal enforcement (Definition def:strong-enforcement); the
    # projector is the diagonal fast path for every nodal family and the
    # dense-metric projector for the two spline families. 'selector' keeps
    # the legacy zero-the-wall-rows lifting for comparison runs only.
    proj = make_wall_projector(
        scheme_metric_1d(scheme_label, npts) if enforcement == 'horth' else None,
        npts)

    if dt is not None:
        # Externally supplied dt: every family in a fair comparison uses
        # the same time step so reported errors reflect spatial accuracy
        # rather than mismatched temporal resolution.
        nsteps = max(1, int(np.ceil(t_end / float(dt))))
        dt = t_end / nsteps
    else:
        # Family-local CFL: dt <= cfl * h^2 / nu (safe for RK4 on the
        # diffusion operator). Backward-compatible fallback.
        h = float(x[1] - x[0])
        dt0 = cfl * h * h / max(nu, 1.0e-30)
        nsteps = int(np.ceil(t_end / dt0))
        dt = t_end / nsteps

    U = modes.field_grid(x, y, 0.0)
    enforce_zero_walls(U)

    weights = node_trapezoid_weights(x)
    w2d = np.outer(weights, weights)

    rec_every = max(1, nsteps // max(2, record_count - 1))
    snap_every = max(1, nsteps // max(2, snapshot_count - 1))

    rec_t = [0.0]
    rec_u_l2 = [0.0]
    rec_u_inf = [0.0]
    rec_dec = [0.0]

    snap_t = [0.0]
    snap_u = [U.copy()]
    snap_uex = [U.copy()]

    # Peak amplitude of |exact field| happens at the anti-node of the
    # dominant mode; track the numerical peak relative to that exact
    # anti-node location over time — an informative scalar for videos.
    def anti_node():
        mpeak, npeak = modes.peak_mode()
        ix = int(round((npts - 1) / (2.0 * mpeak)))
        iy = int(round((npts - 1) / (2.0 * npeak)))
        return ix, iy

    ix, iy = anti_node()

    t = 0.0
    for step in range(nsteps):
        U = rk4_step_horth(D, U, dt, nu, proj)
        t += dt
        is_last = step == nsteps - 1
        if is_last or (step + 1) % rec_every == 0:
            u_ex = modes.field_grid(x, y, t)
            err = U - u_ex
            rec_t.append(t)
            rec_u_l2.append(_l2_rel(err, u_ex, w2d))
            rec_u_inf.append(float(np.max(np.abs(err))))
            rec_dec.append(abs(float(U[ix, iy]) - float(u_ex[ix, iy])))
        if is_last or (step + 1) % snap_every == 0:
            snap_t.append(t)
            snap_u.append(U.copy())
            snap_uex.append(modes.field_grid(x, y, t))

    peak = modes.peak_mode()
    return HeatCavityResult(
        scheme=scheme_label,
        npts=int(npts),
        times=np.asarray(rec_t),
        u_rel_l2=np.asarray(rec_u_l2),
        u_linf=np.asarray(rec_u_inf),
        decay_error=np.asarray(rec_dec),
        snapshot_times=np.asarray(snap_t),
        snapshot_u=np.stack(snap_u, axis=0),
        snapshot_u_exact=np.stack(snap_uex, axis=0),
        final_u=U.copy(),
        final_u_exact=modes.field_grid(x, y, t),
        peak_mode=peak,
        ppw=ppw_for(max(peak), npts, 1.0),
        steps=int(nsteps),
        dt=float(dt),
        t_end=float(t_end),
    )


_SBP_CLOSED_LABELS: list[str] = ["SBP6", "BMF6", "BMG6"]
_SIX_FAMILY_LABELS: list[str] = [
    'E4', 'C4', 'SBP4', 'DRP4', 'BMF4', 'BMG4',
    'E6', 'C6', 'SBP6', 'DRP6', 'BMF6', 'BMG6',
    'E8', 'C8', 'SBP8', 'DRP8', 'BMF8', 'BMG8',
]


def heat_cavity_labels(roster: str = 'sbp_closed') -> list[str]:
    """Return the list of scheme labels for the heat-cavity benchmark.

    The legacy roster `sbp_closed` returns the three SBP-closed families
    (SBP6, BMF6, BMG6) used in earlier drafts. The roster `six_family`
    returns all six matched-order families at q in {4, 6, 8}, with the
    composed same-grid second derivative used for E, C, DRP.
    """
    if roster == 'sbp_closed':
        return list(_SBP_CLOSED_LABELS)
    if roster == 'six_family':
        return list(_SIX_FAMILY_LABELS)
    raise ValueError(f"unknown roster: {roster}")


def shared_diffusion_dt(
    labels: list[str],
    *,
    npts: int,
    viscosity: float,
    cfl: float,
    t_end: float,
) -> float:
    """Return the dt that all families share for a fair comparison.

    The dt is `cfl * h_min^2 / nu` where `h_min` is the smallest grid
    spacing over the union of all participating families' grids. Greville
    clustering on the spline pair sets the binding constraint on the spline
    routes and the same dt is then imposed on the uniform-grid routes so
    every family takes the same number of RK4 steps and reported errors
    reflect spatial accuracy rather than mismatched temporal resolution.
    """
    h_min = float('inf')
    for lab in labels:
        op = build_operator(scheme_kind(lab), scheme_order(lab), npts)
        h_min = min(h_min, float(np.min(np.diff(op.x))))
    dt0 = cfl * h_min * h_min / max(float(viscosity), 1.0e-30)
    nsteps = max(1, int(np.ceil(t_end / dt0)))
    return float(t_end) / nsteps
