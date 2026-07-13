// cavity.cpp
//
// STANDARD (unregularized) lid-driven cavity, Re=1000, unit square, exercising
// the same-grid BMG conservative primitive-variable incompressible Navier-Stokes
// formulation of Section 5.7 / Theorem 11 of the paper (WallNS2D:
// skew-symmetric split Galerkin advection, compatible vector Laplacian, weak-div-free
// Schur-complement projection, strong Dirichlet traces). Only the BMG same-grid
// route is exposed as a primitive-variable NS loop in the source library (no BMF
// staggered primitive-variable NS loop exists there); this app runs
// the BMG route only.
//
// No-slip on three walls; TOP wall carries the strong Dirichlet trace u=1,w=0
// (the lid), not the regularized (polynomial-tapered) lid -- the Re=1000 spectral
// benchmark tabulates the standard cavity, so this app matches that anchor exactly.
//
// Reports, at every output step:
//   - kinetic energy K_h = 1/2 (u^T H u + w^T H w)
//   - dissipation D_h = nu * sum_r (G_r u_i)^T H_r (G_r u_i)   (interior term of
//     the KE balance, Theorem 11); the residual dK/dt - (-D_h) is the boundary
//     work supplied by the moving lid (a strong trace, not a natural BC, so this
//     is measured as a residual, not asserted a priori)
//   - momentum balance residual: d/dt (1^T H u, 1^T H w) vs the same measured
//     residual decomposition
//   - divergence defect max|weak div u|
//
// Usage: bin/cavity [N=64] [p=4] [Re=1000] [T=40] [dt_scale=1.0] [out] [wall_budget_s=0] [skew|conv] [restart.bin|-]
//   dt_scale multiplies the base explicit-stable dt (used for the dt, dt/2 pair
//   in the time-integrator note). wall_budget_s>0 stops the run early (after the
//   in-progress step) once that many wall-clock seconds have elapsed, so a run
//   sharing a contended machine always exits with valid, honestly-labeled
//   output (steady_step=-1, final_max_dudt reported) instead of being killed
//   mid-write; 0 means no budget (run the full nsteps or until steady).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <limits>
#include <sys/resource.h>
#include "wall_ns2d.h"

static const double PI = 3.14159265358979323846;

// Lid-driven cavity boundary condition: u = g_lid(x), w = 0 on the TOP wall
// (z = Lz, j = Z.n-1); u = w = 0 on the other three walls. Overwrites
// WallNS2D::no_slip, which zeros every boundary DOF including the lid.
// Default lid profile: the standard unregularized u = 1 (coefficients all 1
// by the partition of unity). With "reg" the lid is the smooth regularized
// profile u(x) = 16 x^2 (1-x)^2 (zero value and slope at the corners, so the
// exact solution is H1-regular and functional convergence is meaningful);
// its spline lid coefficients are the 1D Greville interpolant, computed once.
static std::vector<double> LIDCOEF;   // per-i lid coefficient (size nx)

static void build_lid_coef(WallNS2D& S, bool regularized) {
    const int nx = S.X.n;
    LIDCOEF.assign(nx, 1.0);
    if (!regularized) return;
    DenseMatrix B(nx);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < nx; ++j) B(i, j) = S.X.B(i, j);
    std::vector<double> f(nx), a;
    for (int i = 0; i < nx; ++i) {
        const double x = S.X.grev[i];
        f[i] = 16.0 * x * x * (1.0 - x) * (1.0 - x);
    }
    dense_lu_solve(B, f, a);
    LIDCOEF = a;
}

static void lid_bc(WallNS2D& S) {
    const int nx = S.X.n, nz = S.Z.n;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < nz; ++j) {
            const bool bnd = (i == 0 || i == nx - 1 || j == 0 || j == nz - 1);
            if (!bnd) continue;
            const bool lid = (j == nz - 1);
            S.u.at(i, j) = lid ? LIDCOEF[i] : 0.0;
            S.w.at(i, j) = 0.0;
        }
    }
}
// homogeneous version for RK stage increments (du, dw are rates, not fields:
// the strongly-enforced boundary rows carry zero rate once the BC value is
// fixed for all t>0)
static void lid_bc_rate(WallNS2D& S, Field2D& du, Field2D& dw) {
    const int nx = S.X.n, nz = S.Z.n;
    for (int i = 0; i < nx; ++i) for (int j = 0; j < nz; ++j) {
        const bool bnd = (i == 0 || i == nx - 1 || j == 0 || j == nz - 1);
        if (bnd) { du.at(i, j) = 0.0; dw.at(i, j) = 0.0; }
    }
    (void)S;
}

// step() re-implemented from wall_ns2d.h's WallNS2D::step but calling lid_bc
// instead of no_slip on the fields (rates are homogeneous at every RK stage
// because the lid value is a constant-in-time Dirichlet datum).
// project without the library's own no_slip pin (which would zero the lid too),
// then re-enforce the CAVITY boundary data (u=1 on the lid, zero elsewhere).
// This is the same fractional-step wall-splitting pattern as WallNS2D::project's
// own pin=true branch, substituting the cavity BC for the no-slip BC in that slot.
static void cavity_project(WallNS2D& S);

// ---------------------------------------------------------------------------
// CONSTRAINED projection (Definition 3, metric-orthogonal): the H-nearest
// field that satisfies the weak divergence constraint AND the exact boundary
// trace simultaneously, replacing the fractional-step project-then-pin whose
// standing wall defect weakens the driven vortex under refinement.
//   min ||v-u*||_H  s.t.  D v = 0,  B v = g
//   v = P(u* - H^{-1}B^T lambda),  S_b lambda = B P u* - g,
//   S_b = B P H^{-1} B^T,  P = I - H^{-1}D^T L^+ D  (fast-diagonalized).
// S_b is built column by column once and LU-factored once; each application
// costs two fast projections and one dense back-substitution.
struct ConstrainedProjector {
    WallNS2D& S;
    std::vector<int> bidx;              // boundary flat indices (one component)
    int m1 = 0, m = 0;
    std::vector<double> eig, evec, eval_;   // eigendecomposition of S_b
    double cutoff = 0.0;
    std::vector<double> g;              // boundary targets [u-part; w-part]

    // div-free projection of the CURRENT (S.u, S.w), no pin
    void pfree() { S.project(false); }

    explicit ConstrainedProjector(WallNS2D& S_) : S(S_) {
        const int nx = S.X.n, nz = S.Z.n;
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < nz; ++j)
                if (S.is_bndry(i, j)) bidx.push_back(i * nz + j);
        m1 = (int)bidx.size();
        m = 2 * m1;
        g.assign(m, 0.0);
        for (int k = 0; k < m1; ++k) {
            const int i = bidx[k] / nz, j = bidx[k] % nz;
            g[k] = (j == nz - 1) ? LIDCOEF[i] : 0.0;   // lid profile; walls 0; w all 0
        }
        // build S_b = B P H^{-1} B^T column by column
        std::vector<double> Sb((size_t)m * m, 0.0);
        Field2D su(nx, nz), sw(nx, nz);
        for (int c = 0; c < m; ++c) {
            for (double& v : S.u.a) v = 0.0;
            for (double& v : S.w.a) v = 0.0;
            if (c < m1) S.u.a[bidx[c]] = 1.0; else S.w.a[bidx[c - m1]] = 1.0;
            mass_solve2d(S.X, S.Z, S.u);
            mass_solve2d(S.X, S.Z, S.w);
            pfree();
            for (int r = 0; r < m1; ++r) {
                Sb[(size_t)r * m + c] = S.u.a[bidx[r]];
                Sb[(size_t)(r + m1) * m + c] = S.w.a[bidx[r]];
            }
        }
        // S_b is symmetric PSD with near-null directions at the lid corners
        // (boundary-trace combinations a weak-div-free spline field cannot
        // reach; the discontinuous-lid pathology). A plain LU amplifies those
        // directions catastrophically, so the solve is the eigendecomposition
        // pseudo-inverse with a relative cutoff: reachable constraints are met
        // exactly, unreachable corner combinations are left at their
        // least-squares residual.
        for (int i = 0; i < m; ++i)                          // symmetrize roundoff
            for (int j = i + 1; j < m; ++j) {
                const double v = 0.5 * (Sb[(size_t)i * m + j] + Sb[(size_t)j * m + i]);
                Sb[(size_t)i * m + j] = v; Sb[(size_t)j * m + i] = v;
            }
        eig = Sb;                                            // Jacobi in place
        evec.assign((size_t)m * m, 0.0);
        for (int i = 0; i < m; ++i) evec[(size_t)i * m + i] = 1.0;
        for (int sweep = 0; sweep < 80; ++sweep) {
            double off = 0, dia = 0;
            for (int i = 0; i < m; ++i) {
                dia += eig[(size_t)i * m + i] * eig[(size_t)i * m + i];
                for (int j = i + 1; j < m; ++j) off += eig[(size_t)i * m + j] * eig[(size_t)i * m + j];
            }
            if (off <= 1e-26 * std::fmax(dia, 1e-300)) break;
            for (int pp = 0; pp < m; ++pp) for (int q = pp + 1; q < m; ++q) {
                const double apq = eig[(size_t)pp * m + q];
                if (std::fabs(apq) < 1e-300) continue;
                const double app = eig[(size_t)pp * m + pp], aqq = eig[(size_t)q * m + q];
                const double th = 0.5 * (aqq - app) / apq;
                const double tsg = (th >= 0) ? 1.0 : -1.0;
                const double tj = tsg / (std::fabs(th) + std::sqrt(th * th + 1.0));
                const double cj = 1.0 / std::sqrt(tj * tj + 1.0), sj = tj * cj;
                for (int k = 0; k < m; ++k) {
                    const double a1 = eig[(size_t)k * m + pp], a2 = eig[(size_t)k * m + q];
                    eig[(size_t)k * m + pp] = cj * a1 - sj * a2;
                    eig[(size_t)k * m + q] = sj * a1 + cj * a2;
                }
                for (int k = 0; k < m; ++k) {
                    const double a1 = eig[(size_t)pp * m + k], a2 = eig[(size_t)q * m + k];
                    eig[(size_t)pp * m + k] = cj * a1 - sj * a2;
                    eig[(size_t)q * m + k] = sj * a1 + cj * a2;
                }
                for (int k = 0; k < m; ++k) {
                    const double a1 = evec[(size_t)k * m + pp], a2 = evec[(size_t)k * m + q];
                    evec[(size_t)k * m + pp] = cj * a1 - sj * a2;
                    evec[(size_t)k * m + q] = sj * a1 + cj * a2;
                }
            }
        }
        eval_.resize(m);
        double emax = 0;
        for (int i = 0; i < m; ++i) {
            eval_[i] = eig[(size_t)i * m + i];
            emax = std::fmax(emax, eval_[i]);
        }
        // relative truncation cutoff, overridable for the sensitivity study
        double relcut = 1e-8;
        if (const char* env = std::getenv("CPROJ_CUTOFF")) relcut = std::atof(env);
        int kept = 0;
        for (int i = 0; i < m; ++i) if (eval_[i] > relcut * emax) ++kept;
        std::printf(" constrained projector: m=%d, eig range [%.2e, %.2e], kept %d/%d modes\n",
                    m, *std::min_element(eval_.begin(), eval_.end()), emax, kept, m);
        cutoff = relcut * emax;
        // the setup used the state arrays as scratch; hand back a clean state
        for (double& v : S.u.a) v = 0.0;
        for (double& v : S.w.a) v = 0.0;
    }

    void lusolve(std::vector<double>& b) const {
        // pseudo-inverse apply: b <- V diag(1/eval on kept modes) V^T b
        std::vector<double> y(m, 0.0);
        for (int j = 0; j < m; ++j) {
            if (eval_[j] <= cutoff) continue;
            double s = 0;
            for (int i = 0; i < m; ++i) s += evec[(size_t)i * m + j] * b[i];
            y[j] = s / eval_[j];
        }
        for (int i = 0; i < m; ++i) {
            double s = 0;
            for (int j = 0; j < m; ++j) s += evec[(size_t)i * m + j] * y[j];
            b[i] = s;
        }
    }

    // replace (S.u, S.w) by the constrained projection of themselves
    void apply() {
        const int nz = S.Z.n;
        pfree();                                            // (u,w) <- P u*
        std::vector<double> rhs(m);
        for (int r = 0; r < m1; ++r) {
            rhs[r] = S.u.a[bidx[r]] - g[r];
            rhs[r + m1] = S.w.a[bidx[r]] - g[r + m1];
        }
        lusolve(rhs);                                       // lambda
        // u <- P u* - P H^{-1} B^T lambda: build the correction, project, subtract
        Field2D cu(S.X.n, nz), cw(S.X.n, nz);
        cu.a.swap(S.u.a); cw.a.swap(S.w.a);                 // stash P u*
        for (double& v : S.u.a) v = 0.0;
        for (double& v : S.w.a) v = 0.0;
        for (int r = 0; r < m1; ++r) {
            S.u.a[bidx[r]] = rhs[r];
            S.w.a[bidx[r]] = rhs[r + m1];
        }
        mass_solve2d(S.X, S.Z, S.u);
        mass_solve2d(S.X, S.Z, S.w);
        pfree();                                            // P H^{-1}B^T lambda
        for (size_t q = 0; q < S.u.a.size(); ++q) {
            S.u.a[q] = cu.a[q] - S.u.a[q];
            S.w.a[q] = cw.a[q] - S.w.a[q];
        }
    }
};

static ConstrainedProjector* CPROJ = nullptr;

// pin-after-project fractional step (legacy), or the Definition-3 constrained
// projection when enabled by the "cproj" argument
static void cavity_project(WallNS2D& S) {
    if (CPROJ) { CPROJ->apply(); return; }
    S.project(false); lid_bc(S);
}

static void cavity_step(WallNS2D& S, double dt) {
    S.u0 = S.u; S.w0 = S.w;
    S.rhs(S.ku, S.kw); if (S.pin_rates) lid_bc_rate(S, S.ku, S.kw);
    for (size_t q = 0; q < S.u.a.size(); ++q) {
        S.u.a[q] = S.u0.a[q] + dt * S.ku.a[q];
        S.w.a[q] = S.w0.a[q] + dt * S.kw.a[q];
    }
    lid_bc(S); cavity_project(S);
    S.rhs(S.ku, S.kw); if (S.pin_rates) lid_bc_rate(S, S.ku, S.kw);
    for (size_t q = 0; q < S.u.a.size(); ++q) {
        S.u.a[q] = 0.75 * S.u0.a[q] + 0.25 * (S.u.a[q] + dt * S.ku.a[q]);
        S.w.a[q] = 0.75 * S.w0.a[q] + 0.25 * (S.w.a[q] + dt * S.kw.a[q]);
    }
    lid_bc(S); cavity_project(S);
    S.rhs(S.ku, S.kw); if (S.pin_rates) lid_bc_rate(S, S.ku, S.kw);
    for (size_t q = 0; q < S.u.a.size(); ++q) {
        S.u.a[q] = (1.0 / 3.0) * S.u0.a[q] + (2.0 / 3.0) * (S.u.a[q] + dt * S.ku.a[q]);
        S.w.a[q] = (1.0 / 3.0) * S.w0.a[q] + (2.0 / 3.0) * (S.w.a[q] + dt * S.kw.a[q]);
    }
    lid_bc(S); cavity_project(S);
}

// dissipation D_h = nu * sum_r (G_r u_i)^T H_r (G_r u_i) = nu * (u^T Stiff u + w^T Stiff w)
// (the interior term of the KE balance; stiff() applies Sx(x)Mz + Mx(x)Sz, the
// weak Laplacian bilinear form, so u^T stiff(u) = (grad-form) u^T H_r G_r term).
static double dissipation(WallNS2D& S) {
    S.stiff(S.u, S.t1);
    double d = 0; for (size_t q = 0; q < S.u.a.size(); ++q) d += S.u.a[q] * S.t1.a[q];
    S.stiff(S.w, S.t1);
    for (size_t q = 0; q < S.w.a.size(); ++q) d += S.w.a[q] * S.t1.a[q];
    return S.nu * d;
}
// Advective kinetic-energy PRODUCTION rate: the raw quadratic form
//   Pi_adv = u_i^T C_i(u),   C_i = the assembled advection term (skew or conv).
// This is the rate at which the convective term feeds the discrete KE budget
// (dK/dt gets a -Pi_adv contribution; the H-metric mass solve is not applied
// here because C_i is already the H-weighted Galerkin residual <N_i, adv>).
// Theorem 10/11: for the energy-conserving skew form Pi_adv = 0 to round-off
// (the interior injects nothing; only the boundary trace carries flux). For the
// non-conservative convective form Pi_adv != 0 -- the SPURIOUS energy production
// the conservative scheme removes and the collocated/central scheme cannot.
static double ke_production(WallNS2D& S) {
    Field2D Cu(S.X.n, S.Z.n), Cw(S.X.n, S.Z.n);
    if (S.convective) S.advect_conv(Cu, Cw); else S.advect_skew(Cu, Cw);
    double p = 0;
    for (size_t q = 0; q < S.u.a.size(); ++q) p += S.u.a[q] * Cu.a[q] + S.w.a[q] * Cw.a[q];
    return p;   // = sum_i u_i^T <N, adv_i>  (spurious KE production, up to boundary flux)
}
// total discrete momentum components P_x = 1^T H u, P_z = 1^T H w  (H = Mx(x)Mz)
static void momentum(WallNS2D& S, double& Px, double& Pz) {
    apply_z(S.Z.M, S.u, S.t1); apply_x(S.X.M, S.t1, S.t2);
    Px = 0; for (double v : S.t2.a) Px += v;
    apply_z(S.Z.M, S.w, S.t1); apply_x(S.X.M, S.t1, S.t2);
    Pz = 0; for (double v : S.t2.a) Pz += v;
}
// The CONSERVED quantity of Theorems 10/11: the discrete (weak/moment) divergence
// constraint (q, div u_h) = 0 that the Schur projection enforces. weak_div gives
// the moment vector d_i = (N_i, div u_h); its l2 norm ||d||_2 is the machine-
// precision measure of the constraint, driven to the projection's CG tolerance.
// This is distinct from max|d| (the nodal max), which is corner-singularity-
// limited by the top-corner lid velocity discontinuity, NOT a conservation
// failure. Both are reported.
static double weak_div_l2(WallNS2D& S) {
    S.weak_div(S.u, S.w, S.dvg);
    double s = 0; for (double v : S.dvg.a) s += v * v;
    return std::sqrt(s);
}
// The exact Theorem 10/11 constraint measure: the residual (N_i, div u_h) of the
// field the Schur projection produces, BEFORE the fractional-step boundary pin
// re-writes the lid rows. Measured non-destructively (save, project without pin,
// measure, restore). This is the (q, div u_h)=0 constraint the projection solves
// to CG tolerance; it is machine-precision small, distinct from the post-pin
// divergence (which carries the small fractional-step wall-splitting error).
static void constraint_div(WallNS2D& S, double& l2, double& mx) {
    std::vector<double> us = S.u.a, ws = S.w.a;
    S.project(false);
    S.weak_div(S.u, S.w, S.dvg);
    l2 = 0; mx = 0;
    for (double v : S.dvg.a) { l2 += v * v; mx = std::fmax(mx, std::fabs(v)); }
    l2 = std::sqrt(l2);
    S.u.a = us; S.w.a = ws;
}

// Measured pressure-work rate, the Theorem 10/11 pressure channel: the discrete
// pressure of the last projection (p = phi/dt up to the projection scaling)
// paired with the weak divergence of the field as evolved (including the
// fractional-step lid pin). Identically zero in exact arithmetic on the
// constrained field; measured, it is capped by |phi| times the divergence
// defect, and is reported alongside the dissipation for scale.
static double pressure_work(WallNS2D& S, double dt) {
    S.weak_div(S.u, S.w, S.dvg);
    double pw = 0;
    for (size_t q = 0; q < S.phi.a.size(); ++q) pw += S.phi.a[q] * S.dvg.a[q];
    return pw / dt;
}

// State file: header nx,nz,p (3x int32), then t (double, checkpoint format
// only), then u then w coefficient arrays. The legacy <prefix>_final.bin dumps
// omit t; both are accepted on restart (legacy resumes with t0 = 0).
static void write_state(const std::string& path, WallNS2D& S, double t, bool with_t) {
    const std::string tmp = path + ".tmp";
    FILE* fd = std::fopen(tmp.c_str(), "wb");
    if (!fd) { std::fprintf(stderr, "cannot write %s\n", tmp.c_str()); return; }
    int nx = S.X.n, nz = S.Z.n, pp = S.X.p;
    std::fwrite(&nx, 4, 1, fd); std::fwrite(&nz, 4, 1, fd); std::fwrite(&pp, 4, 1, fd);
    if (with_t) std::fwrite(&t, 8, 1, fd);
    std::fwrite(S.u.a.data(), 8, S.u.a.size(), fd);
    std::fwrite(S.w.a.data(), 8, S.w.a.size(), fd);
    std::fclose(fd);
    std::rename(tmp.c_str(), path.c_str());
}

// returns t0 (0.0 for legacy files without a time stamp); exits on mismatch
static double read_state(const std::string& path, WallNS2D& S) {
    FILE* fr = std::fopen(path.c_str(), "rb");
    if (!fr) { std::fprintf(stderr, "restart file %s not found\n", path.c_str()); std::exit(1); }
    int rnx = 0, rnz = 0, rp = 0;
    if (std::fread(&rnx, 4, 1, fr) != 1 || std::fread(&rnz, 4, 1, fr) != 1 || std::fread(&rp, 4, 1, fr) != 1) {
        std::fprintf(stderr, "restart header read failed\n"); std::exit(1);
    }
    if (rnx != S.X.n || rnz != S.Z.n || rp != S.X.p) {
        std::fprintf(stderr, "restart mismatch: file %dx%d p=%d vs run %dx%d p=%d\n",
                     rnx, rnz, rp, S.X.n, S.Z.n, S.X.p);
        std::exit(1);
    }
    const long pos = std::ftell(fr);
    std::fseek(fr, 0, SEEK_END);
    const long sz = std::ftell(fr);
    std::fseek(fr, pos, SEEK_SET);
    const long payload = (long)(S.u.a.size() + S.w.a.size()) * 8;
    double t0 = 0.0;
    if (sz - pos == payload + 8) {
        if (std::fread(&t0, 8, 1, fr) != 1) { std::fprintf(stderr, "restart t read failed\n"); std::exit(1); }
    } else if (sz - pos != payload) {
        std::fprintf(stderr, "restart size mismatch (%ld bytes of payload, expected %ld or %ld)\n",
                     sz - pos, payload, payload + 8);
        std::exit(1);
    }
    if (std::fread(S.u.a.data(), 8, S.u.a.size(), fr) != S.u.a.size() ||
        std::fread(S.w.a.data(), 8, S.w.a.size(), fr) != S.w.a.size()) {
        std::fprintf(stderr, "restart payload read failed\n"); std::exit(1);
    }
    std::fclose(fr);
    return t0;
}

// EXACT spline evaluation of a coefficient field at arbitrary (x,z):
// u_h(x,z) = sum_{i,j} alpha_{ij} N_i(x) N_j(z). Uses the de Boor span + basis
// evaluation, so it captures sub-Greville-node curvature (the sharp near-wall
// velocity dip) that a nodal bilinear interpolant would miss. This is the right
// tool for centerline profiles and their extrema.
struct SplineEval {
    WallNS2D& S;
    SplineEval(WallNS2D& S_) : S(S_) {}
    // evaluate a coefficient field (u or w) at (x,z)
    double eval(const Field2D& coef, double x, double z) const {
        const Axis& X = S.X; const Axis& Z = S.Z;
        std::vector<std::vector<double>> dx, dz;
        int sx = find_span(X.n - 1, X.p, x, X.knots);
        int sz = find_span(Z.n - 1, Z.p, z, Z.knots);
        ders_basis_funs(sx, x, X.p, 0, X.knots, dx);
        ders_basis_funs(sz, z, Z.p, 0, Z.knots, dz);
        double val = 0.0;
        for (int a = 0; a <= X.p; ++a) {
            const int gi = sx - X.p + a;
            for (int b = 0; b <= Z.p; ++b) {
                const int gj = sz - Z.p + b;
                val += coef.at(gi, gj) * dx[0][a] * dz[0][b];
            }
        }
        return val;
    }
    double u_at(double x, double z) const { return eval(S.u, x, z); }
    double w_at(double x, double z) const { return eval(S.w, x, z); }
};

int main(int argc, char** argv) {
    const int    N        = (argc > 1) ? std::atoi(argv[1]) : 64;
    const int    p        = (argc > 2) ? std::atoi(argv[2]) : 4;
    const double Re       = (argc > 3) ? std::atof(argv[3]) : 1000.0;
    const double T        = (argc > 4) ? std::atof(argv[4]) : 40.0;
    const double dt_scale = (argc > 5) ? std::atof(argv[5]) : 1.0;
    const std::string outprefix = (argc > 6) ? argv[6] : "cavity";
    const double wall_budget_s = (argc > 7) ? std::atof(argv[7]) : 0.0;
    // argv[8]: advection form. "conv" = NON-CONSERVATIVE competitor (u.grad u),
    // anything else / absent = the paper's energy-conserving skew BMG scheme.
    const bool use_conv = (argc > 8) && (std::string(argv[8]) == "conv");
    // argv[9]: restart state file ("-" or absent = fresh start). T is TOTAL
    // physical time: a restart marches from the checkpoint's t up to T.
    const std::string restart_path = (argc > 9 && std::string(argv[9]) != "-") ? argv[9] : "";
    const auto wall_t0 = std::chrono::steady_clock::now();

    const double Lx = 1.0, Lz = 1.0;
    const double nu = 1.0 / Re;

    WallNS2D S(N, N, p, Lx, Lz, nu);
    S.convective = use_conv;
    // CAVITY_CG=1 falls back to the CG Schur solve (cross-verification of the
    // direct fast-diagonalization path; identical up to CG tolerance)
    S.direct_schur = (std::getenv("CAVITY_CG") == nullptr);
    // argv[10] == "cproj": Definition-3 constrained projection (div-free AND
    // exact boundary trace in one metric-orthogonal solve); built before any
    // state is loaded because the setup uses the state arrays as scratch
    const bool use_cproj = (argc > 10) && (std::string(argv[10]) == "cproj");
    // CAVITY_NO_ADVECT=1: Stokes-only march (forensic probe isolating the
    // viscous + projection stack from the advection assembly)
    if (std::getenv("CAVITY_NO_ADVECT")) S.advect = false;
    // argv[11] == "reg": smooth regularized lid u = 16 x^2 (1-x)^2
    // argv[11] == "tg":  Taylor-Green-like decay in an all-no-slip box:
    //                    zero lid, IC from psi = sin^2(pi x) sin^2(pi z) / pi
    const bool use_reg = (argc > 11) && (std::string(argv[11]) == "reg");
    const bool use_tg  = (argc > 11) && (std::string(argv[11]) == "tg");
    build_lid_coef(S, use_reg);
    if (use_tg) LIDCOEF.assign(LIDCOEF.size(), 0.0);   // no-slip on all four walls
    if (use_cproj) CPROJ = new ConstrainedProjector(S);
    // CPROJ_FREE_RATES=1 (constrained projection only): stage rates evolve
    // freely and the projection alone owns the boundary manifold, removing
    // the selector rate-zeroing suspected of capping coupled-order accuracy
    const bool free_rates = use_cproj && std::getenv("CPROJ_FREE_RATES");
    S.pin_rates = !free_rates;
    double t_start = 0.0;
    if (!restart_path.empty()) {
        // the checkpoint stores the exact beginning-of-step (post-projection,
        // post-pin) state; re-projecting here would perturb it by the
        // fractional-step wall defect, so only the (idempotent) lid rows are
        // re-asserted
        t_start = read_state(restart_path, S);
        lid_bc(S);
    } else {
        if (use_tg) {
            // u = dpsi/dz, w = -dpsi/dx of psi = sin^2(pi x) sin^2(pi z)/pi:
            // both velocity components vanish identically on every wall, so
            // the analytic field is exactly no-slip and the projection only
            // removes its small discrete divergence. The state arrays hold
            // spline COEFFICIENTS, so the Greville samples are interpolated:
            // c = Bx^{-1} U Bz^{-T}, one dense solve per line.
            const int nx = S.X.n, nz = S.Z.n;
            for (int i = 0; i < nx; ++i) for (int j = 0; j < nz; ++j) {
                const double x = S.X.grev[i], z = S.Z.grev[j];
                const double sx = std::sin(M_PI * x), cx = std::cos(M_PI * x);
                const double sz = std::sin(M_PI * z), cz = std::cos(M_PI * z);
                S.u.at(i, j) = 2.0 * sx * sx * sz * cz;
                S.w.at(i, j) = -2.0 * sx * cx * sz * sz;
            }
            DenseMatrix Bx(nx), Bz(nz);
            for (int i = 0; i < nx; ++i) for (int j = 0; j < nx; ++j) Bx(i, j) = S.X.B(i, j);
            for (int i = 0; i < nz; ++i) for (int j = 0; j < nz; ++j) Bz(i, j) = S.Z.B(i, j);
            std::vector<double> line, sol;
            for (Field2D* F : {&S.u, &S.w}) {
                line.assign(nx, 0.0);
                for (int j = 0; j < nz; ++j) {            // x-direction solves
                    for (int i = 0; i < nx; ++i) line[i] = F->at(i, j);
                    dense_lu_solve(Bx, line, sol);
                    for (int i = 0; i < nx; ++i) F->at(i, j) = sol[i];
                }
                line.assign(nz, 0.0);
                for (int i = 0; i < nx; ++i) {            // z-direction solves
                    for (int j = 0; j < nz; ++j) line[j] = F->at(i, j);
                    dense_lu_solve(Bz, line, sol);
                    for (int j = 0; j < nz; ++j) F->at(i, j) = sol[j];
                }
            }
        }
        lid_bc(S); cavity_project(S);
    }

    const double dz = Lz / N;
    const double dt_base = 0.10 * dz;     // advective-CFL-ish base step (unit lid speed)
    const double dt = dt_scale * dt_base;
    const int nsteps = std::max(0, (int)std::llround((T - t_start) / dt));

    std::printf("=================================================================\n");
    std::printf(" Lid-driven cavity (BMG same-grid primitive NS, Theorem 11)\n");
    std::printf(" advection = %s\n", use_conv ? "NON-CONSERVATIVE convective u.grad u (COMPETITOR)"
                                              : "energy-conserving skew BMG (paper scheme)");
    std::printf(" N=%d^2  p=%d  Re=%.1f  nu=%.6e  dt=%.6e (scale=%.3f)  nsteps=%d\n",
                N, p, Re, nu, dt, dt_scale, nsteps);
    if (!restart_path.empty())
        std::printf(" RESTART from %s at t=%.6f\n", restart_path.c_str(), t_start);
    std::printf("=================================================================\n");
    std::fflush(stdout);

    const std::string csv_path = std::string("data/") + outprefix + "_budgets_raw.csv";
    bool csv_has_rows = false;
    if (!restart_path.empty()) {
        if (FILE* pr = std::fopen(csv_path.c_str(), "r")) {
            std::fseek(pr, 0, SEEK_END);
            csv_has_rows = std::ftell(pr) > 0;
            std::fclose(pr);
        }
    }
    FILE* budget_csv = std::fopen(csv_path.c_str(), csv_has_rows ? "a" : "w");
    if (!csv_has_rows)
        std::fprintf(budget_csv, "step,t,KE,dKEdt,dissipation,ke_residual,ke_production,Px,Pz,dPxdt,dPzdt,"
                                 "max_weak_div,l2_weak_div,constraint_l2_div,constraint_max_div,max_dudt,"
                                 "pressure_work\n");

    double KE_prev = S.kinetic_energy();
    double Px_prev, Pz_prev; momentum(S, Px_prev, Pz_prev);
    double t = t_start;
    long steps_done = 0;
    const int report_every = std::max(1, nsteps / 200);
    const int ckpt_every = std::max(1000, report_every * 10);
    const std::string ckpt_path = std::string("data/") + outprefix + "_ckpt.bin";

    std::vector<double> u_prev = S.u.a, w_prev = S.w.a;
    double max_dudt = 1.0;
    int converged_step = -1;
    double steady_time = -1.0;
    const double STEADY_TOL = 1e-9;   // max|du/dt| below this => steady

    for (int n = 0; n <= nsteps; ++n) {
        if (n % report_every == 0 || n == nsteps) {
            const double KE = S.kinetic_energy();
            const double dKEdt = (n == 0) ? 0.0 : (KE - KE_prev) / (dt * report_every);
            const double diss = dissipation(S);
            const double ke_residual = dKEdt - (-diss);   // measured boundary work (lid)
            const double ke_prod = ke_production(S);       // spurious KE production (0 for skew)
            double Px, Pz; momentum(S, Px, Pz);
            const double dPxdt = (n == 0) ? 0.0 : (Px - Px_prev) / (dt * report_every);
            const double dPzdt = (n == 0) ? 0.0 : (Pz - Pz_prev) / (dt * report_every);
            const double mdiv = S.max_weak_div();
            const double l2div = weak_div_l2(S);
            double cdiv_l2, cdiv_mx; constraint_div(S, cdiv_l2, cdiv_mx);
            const double pwork = pressure_work(S, dt);
            std::fprintf(budget_csv, "%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n",
                         n, t, KE, dKEdt, diss, ke_residual, ke_prod, Px, Pz, dPxdt, dPzdt, mdiv, l2div, cdiv_l2, cdiv_mx, max_dudt, pwork);
            std::fflush(budget_csv);
            if (n % (report_every * 10) == 0 || n == nsteps) {
                std::printf("  n=%7d  t=%8.4f  KE=%.6e  dKE/dt=%+.3e  KEprod=%+.3e  constraint|div|2=%.2e  postdiv=%.2e  max|du/dt|=%.3e\n",
                            n, t, KE, dKEdt, ke_prod, cdiv_l2, mdiv, max_dudt);
                std::fflush(stdout);
            }
            KE_prev = KE; Px_prev = Px; Pz_prev = Pz;
        }
        if (wall_budget_s > 0.0) {
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_t0).count();
            if (elapsed > wall_budget_s) {
                std::printf(" ** wall-clock budget (%.0fs) reached at step %d (t=%.4f); stopping early **\n", wall_budget_s, n, t);
                std::fflush(stdout);
                break;
            }
        }
        if (n < nsteps) {
            cavity_step(S, dt);
            t += dt;
            ++steps_done;
            // NaN-safe maximum: fmax silently DROPS NaN operands, which once
            // turned a blown-up state into max|du/dt| = 0 and a false steady
            // exit; explicit comparisons keep NaN sticky and abort the march
            double md = 0;
            bool state_nan = false;
            for (size_t q = 0; q < S.u.a.size(); ++q) {
                const double av = std::fabs(S.u.a[q] - u_prev[q]) / dt;
                if (std::isnan(av)) { state_nan = true; break; }
                if (av > md) md = av;
            }
            if (!state_nan) for (size_t q = 0; q < S.w.a.size(); ++q) {
                const double av = std::fabs(S.w.a[q] - w_prev[q]) / dt;
                if (std::isnan(av)) { state_nan = true; break; }
                if (av > md) md = av;
            }
            if (state_nan) {
                std::printf(" ** NaN state detected at step %d, t=%.4f; aborting march **\n", n, t);
                std::fflush(stdout);
                max_dudt = std::numeric_limits<double>::quiet_NaN();
                break;
            }
            max_dudt = md;
            u_prev = S.u.a; w_prev = S.w.a;
            if (n % ckpt_every == 0 && n > 0) write_state(ckpt_path, S, t, true);
            if (converged_step < 0 && max_dudt < STEADY_TOL) {
                converged_step = n;
                steady_time = t;
                std::printf(" ** steady (max|du/dt| < %.1e) at step %d, t=%.4f; stopping march **\n",
                            STEADY_TOL, n, t);
                std::fflush(stdout);
                break;
            }
        }
    }
    std::fclose(budget_csv);
    write_state(ckpt_path, S, t, true);

    std::printf("-----------------------------------------------------------------\n");
    if (converged_step >= 0)
        std::printf(" STEADY by step %d (t=%.4f): max|du/dt| < %.1e\n", converged_step, steady_time, STEADY_TOL);
    else
        std::printf(" NOT yet below tol=%.1e; final max|du/dt|=%.3e\n", STEADY_TOL, max_dudt);
    const double final_l2div = weak_div_l2(S);
    const double final_maxdiv = S.max_weak_div();
    double final_cdiv_l2, final_cdiv_mx; constraint_div(S, final_cdiv_l2, final_cdiv_mx);
    std::printf(" Thm 10/11 constraint (post-projection): ||(N_i,div u)||_2 = %.3e , max = %.3e  [machine precision]\n",
                final_cdiv_l2, final_cdiv_mx);
    std::printf(" post-step field divergence (incl. fractional-step wall pin): ||.||_2 = %.3e , max = %.3e\n",
                final_l2div, final_maxdiv);

    // ---- centerline extraction: u(x=0.5, z), w(x, z=0.5) ----
    // Dense EXACT spline evaluation (Ndense points per line) so the sharp
    // near-wall dip and the physical vortex extrema are captured, not the coarse
    // Greville-node values that a nodal interpolant would give.
    SplineEval se(S);
    const int Ndense = 2001;
    std::vector<double> zc(Ndense), uc(Ndense), xc(Ndense), wc(Ndense);
    for (int k = 0; k < Ndense; ++k) {
        const double s = (double)k / (Ndense - 1);
        zc[k] = s * Lz; uc[k] = se.u_at(0.5 * Lx, zc[k]);
        xc[k] = s * Lx; wc[k] = se.w_at(xc[k], 0.5 * Lz);
    }

    FILE* fc = std::fopen((std::string("data/") + outprefix + "_centerline_u.csv").c_str(), "w");
    std::fprintf(fc, "z,u\n");
    for (int k = 0; k < Ndense; ++k) std::fprintf(fc, "%.10e,%.10e\n", zc[k], uc[k]);
    std::fclose(fc);
    FILE* fw = std::fopen((std::string("data/") + outprefix + "_centerline_w.csv").c_str(), "w");
    std::fprintf(fw, "x,w\n");
    for (int k = 0; k < Ndense; ++k) std::fprintf(fw, "%.10e,%.10e\n", xc[k], wc[k]);
    std::fclose(fw);

    // extrema. u_min (vertical centerline) is a clean interior minimum. The v
    // (=w) extrema are searched with a wall margin excluded: the outermost cells
    // carry the top-corner lid-velocity-discontinuity spike (a singularity, not
    // the physical secondary-vortex extremum that the reference tables tabulate at
    // x~0.16-0.23 and x~0.86-0.91). Margin = 0.04 sits well inside those.
    const double margin = 0.04;
    double umin = 1e300, umin_z = 0, wmin = 1e300, wmin_x = 0, wmax = -1e300, wmax_x = 0;
    for (int k = 0; k < Ndense; ++k) if (uc[k] < umin) { umin = uc[k]; umin_z = zc[k]; }
    for (int k = 0; k < Ndense; ++k) {
        if (xc[k] < margin * Lx || xc[k] > (1.0 - margin) * Lx) continue;
        if (wc[k] < wmin) { wmin = wc[k]; wmin_x = xc[k]; }
        if (wc[k] > wmax) { wmax = wc[k]; wmax_x = xc[k]; }
    }
    std::printf(" centerline u_min=%.6f at z=%.4f ; w_min=%.6f at x=%.4f ; w_max=%.6f at x=%.4f\n",
                umin, umin_z, wmin, wmin_x, wmax, wmax_x);

    // ---- final field dump for restart / two-grid comparison (legacy format) ----
    write_state(std::string("data/") + outprefix + "_final.bin", S, t, false);

    // ---- cost instrumentation: wall time, single-process CPU time, memory ----
    const double wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_t0).count();
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    const double cpu_seconds = ru.ru_utime.tv_sec + 1e-6 * ru.ru_utime.tv_usec
                             + ru.ru_stime.tv_sec + 1e-6 * ru.ru_stime.tv_usec;
    const double peak_rss_mb = ru.ru_maxrss / 1024.0;
    std::printf(" cost: %ld steps, %.1f s wall, %.1f s CPU (%.4f s CPU/step), peak RSS %.1f MB\n",
                steps_done, wall_seconds, cpu_seconds,
                steps_done > 0 ? cpu_seconds / steps_done : 0.0, peak_rss_mb);

    // machine-readable summary (extend in postprocessing with spectral benchmark deltas)
    FILE* fj = std::fopen((std::string("data/") + outprefix + "_run_summary.json").c_str(), "w");
    std::fprintf(fj, "{\n");
    std::fprintf(fj, "  \"N\": %d, \"p\": %d, \"Re\": %.1f, \"nu\": %.10e,\n", N, p, Re, nu);
    std::fprintf(fj, "  \"advection\": \"%s\",\n", use_conv ? "convective_nonconservative" : "skew_energy_conserving");
    std::fprintf(fj, "  \"dt\": %.10e, \"dt_scale\": %.6f, \"nsteps\": %d, \"T\": %.6f, \"t_start\": %.6f,\n",
                 dt, dt_scale, nsteps, t, t_start);
    std::fprintf(fj, "  \"restart_from\": \"%s\",\n", restart_path.c_str());
  std::fprintf(fj, "  \"lid_profile\": \"%s\",\n", use_reg ? "regularized_16x2_1mx2" : "unregularized_unit");
    std::fprintf(fj, "  \"steps_done\": %ld, \"wall_seconds\": %.3f, \"cpu_seconds\": %.3f,\n",
                 steps_done, wall_seconds, cpu_seconds);
    std::fprintf(fj, "  \"cpu_seconds_per_step\": %.6e, \"peak_rss_mb\": %.1f,\n",
                 steps_done > 0 ? cpu_seconds / steps_done : 0.0, peak_rss_mb);
    std::fprintf(fj, "  \"final_ke_production\": %.10e,\n", ke_production(S));
    std::fprintf(fj, "  \"final_pressure_work\": %.10e,\n", pressure_work(S, dt));
    std::fprintf(fj, "  \"final_dissipation\": %.10e,\n", dissipation(S));
    std::fprintf(fj, "  \"steady_step\": %d, \"steady_t\": %.6f, \"steady_tol\": %.3e, \"final_max_dudt\": %.6e,\n",
                converged_step, steady_time, STEADY_TOL, max_dudt);
    std::fprintf(fj, "  \"final_constraint_l2_div\": %.6e, \"final_constraint_max_div\": %.6e,\n", final_cdiv_l2, final_cdiv_mx);
    std::fprintf(fj, "  \"final_poststep_l2_div\": %.6e, \"final_poststep_max_div\": %.6e,\n", final_l2div, final_maxdiv);
    std::fprintf(fj, "  \"u_centerline_min\": %.10f, \"u_centerline_min_z\": %.10f,\n", umin, umin_z);
    std::fprintf(fj, "  \"w_centerline_min\": %.10f, \"w_centerline_min_x\": %.10f,\n", wmin, wmin_x);
    std::fprintf(fj, "  \"w_centerline_max\": %.10f, \"w_centerline_max_x\": %.10f\n", wmax, wmax_x);
    std::fprintf(fj, "}\n");
    std::fclose(fj);

    std::printf(" wrote data/%s_{budgets_raw,centerline_u,centerline_w,run_summary,final.bin}\n", outprefix.c_str());
    return 0;
}
