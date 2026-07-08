// dt_note.cpp
//
// Fully-discrete time-integration budget note for the lid-driven cavity
// (Remark after Theorem 11: the RK time integrator is not exactly conservative,
// so the small measured energy drift is the time-integration budget at the
// stated step, not a violation of the spatial balance).
//
// Procedure: spin the cavity up to a near-steady state at the base step, then
// from that SAME state advance a fixed physical interval twice -- once at dt,
// once at dt/2 -- and report the energy-identity residual at both. The residual
// is the deviation of the discrete kinetic-energy change from the semidiscrete
// balance dK/dt = W_boundary - dissipation over the interval; at fixed spatial
// discretization it is purely the time-integration error and shrinks with dt.
//
//   bin/dt_note [N=32] [p=4] [Re=400] [Tspin=20] [Tmeasure=1.0]
//
// Reuses the cavity step/projection/budget helpers by #include of the app TU is
// not possible, so the minimal loop is reproduced here (kept in sync with
// cavity.cpp: same lid BC, same cavity_project, same SSP-RK3).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "wall_ns2d.h"

static void lid_bc(WallNS2D& S) {
    const int nx = S.X.n, nz = S.Z.n;
    for (int i = 0; i < nx; ++i) for (int j = 0; j < nz; ++j) {
        const bool bnd = (i == 0 || i == nx - 1 || j == 0 || j == nz - 1);
        if (!bnd) continue;
        S.u.at(i, j) = (j == nz - 1) ? 1.0 : 0.0;
        S.w.at(i, j) = 0.0;
    }
}
static void lid_bc_rate(WallNS2D& S, Field2D& du, Field2D& dw) {
    const int nx = S.X.n, nz = S.Z.n;
    for (int i = 0; i < nx; ++i) for (int j = 0; j < nz; ++j) {
        const bool bnd = (i == 0 || i == nx - 1 || j == 0 || j == nz - 1);
        if (bnd) { du.at(i, j) = 0.0; dw.at(i, j) = 0.0; }
    }
}
static void cavity_project(WallNS2D& S) { S.project(false); lid_bc(S); }
static void cavity_step(WallNS2D& S, double dt) {
    S.u0 = S.u; S.w0 = S.w;
    S.rhs(S.ku, S.kw); lid_bc_rate(S, S.ku, S.kw);
    for (size_t q = 0; q < S.u.a.size(); ++q) { S.u.a[q] = S.u0.a[q] + dt * S.ku.a[q]; S.w.a[q] = S.w0.a[q] + dt * S.kw.a[q]; }
    lid_bc(S); cavity_project(S);
    S.rhs(S.ku, S.kw); lid_bc_rate(S, S.ku, S.kw);
    for (size_t q = 0; q < S.u.a.size(); ++q) { S.u.a[q] = 0.75 * S.u0.a[q] + 0.25 * (S.u.a[q] + dt * S.ku.a[q]); S.w.a[q] = 0.75 * S.w0.a[q] + 0.25 * (S.w.a[q] + dt * S.kw.a[q]); }
    lid_bc(S); cavity_project(S);
    S.rhs(S.ku, S.kw); lid_bc_rate(S, S.ku, S.kw);
    for (size_t q = 0; q < S.u.a.size(); ++q) { S.u.a[q] = (1.0/3.0) * S.u0.a[q] + (2.0/3.0) * (S.u.a[q] + dt * S.ku.a[q]); S.w.a[q] = (1.0/3.0) * S.w0.a[q] + (2.0/3.0) * (S.w.a[q] + dt * S.kw.a[q]); }
    lid_bc(S); cavity_project(S);
}
static double dissipation(WallNS2D& S) {
    S.stiff(S.u, S.t1);
    double d = 0; for (size_t q = 0; q < S.u.a.size(); ++q) d += S.u.a[q] * S.t1.a[q];
    S.stiff(S.w, S.t1);
    for (size_t q = 0; q < S.w.a.size(); ++q) d += S.w.a[q] * S.t1.a[q];
    return S.nu * d;
}

// Advance a fixed physical interval Tmeasure at step dt from the given state,
// accumulating the time-integrated semidiscrete balance
//   integral( dK/dt - (W_lid - dissipation) ) dt
// measured discretely as [ K(end) - K(start) ] - integral( W_lid - dissipation ).
// The lid boundary work W_lid is not known independently; instead we report the
// residual of the DISCRETE balance dK_step = dt * (W_lid - dissipation): at true
// steady state W_lid = dissipation and dK -> 0, so the surviving |dK| over the
// window is the time-integration budget. We report max |dK/dt| sample and the
// net KE change over the window, at dt and dt/2.
struct WindowStats { double dKE; double max_abs_dKEdt; double diss_mean; };
static WindowStats run_window(WallNS2D& S, double dt, double Tmeasure) {
    const int steps = (int)std::llround(Tmeasure / dt);
    const double KE0 = S.kinetic_energy();
    double max_rate = 0.0, diss_sum = 0.0; int nsamp = 0;
    double KEprev = KE0;
    for (int n = 0; n < steps; ++n) {
        cavity_step(S, dt);
        if (n % 10 == 0) {
            const double KE = S.kinetic_energy();
            const double rate = (KE - KEprev) / (dt * 10);
            max_rate = std::fmax(max_rate, std::fabs(rate));
            diss_sum += dissipation(S); nsamp++;
            KEprev = KE;
        }
    }
    const double KEend = S.kinetic_energy();
    return { KEend - KE0, max_rate, diss_sum / std::max(1, nsamp) };
}

int main(int argc, char** argv) {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 32;
    const int p = (argc > 2) ? std::atoi(argv[2]) : 4;
    const double Re = (argc > 3) ? std::atof(argv[3]) : 400.0;
    const double Tspin = (argc > 4) ? std::atof(argv[4]) : 20.0;
    const double Tmeasure = (argc > 5) ? std::atof(argv[5]) : 1.0;
    const double Lx = 1.0, Lz = 1.0, nu = 1.0 / Re;
    const double dz = Lz / N, dt = 0.10 * dz;

    std::printf("=================================================================\n");
    std::printf(" dt-note: time-integration budget, cavity N=%d^2 p=%d Re=%.0f\n", N, p, Re);
    std::printf(" spin-up to t=%.1f at dt=%.3e, then measure window T=%.2f at dt and dt/2\n", Tspin, dt, Tmeasure);
    std::printf("=================================================================\n");
    std::fflush(stdout);

    WallNS2D S(N, N, p, Lx, Lz, nu);
    lid_bc(S); cavity_project(S);
    const int spin = (int)std::llround(Tspin / dt);
    for (int n = 0; n < spin; ++n) cavity_step(S, dt);
    std::printf(" spun up: KE=%.8e  max|du/dt|(proxy via last step) computed in window\n", S.kinetic_energy());
    std::fflush(stdout);

    // snapshot the spun-up state
    std::vector<double> us = S.u.a, ws = S.w.a;

    // window at dt
    S.u.a = us; S.w.a = ws;
    WindowStats a = run_window(S, dt, Tmeasure);
    // window at dt/2 from the SAME state
    S.u.a = us; S.w.a = ws;
    WindowStats b = run_window(S, 0.5 * dt, Tmeasure);

    std::printf("-----------------------------------------------------------------\n");
    std::printf(" window T=%.2f  (steady-state stretch; KE should be ~constant)\n", Tmeasure);
    std::printf("   dt   = %.6e :  net dKE = %+.6e   max|dKE/dt| = %.6e\n", dt, a.dKE, a.max_abs_dKEdt);
    std::printf("   dt/2 = %.6e :  net dKE = %+.6e   max|dKE/dt| = %.6e\n", 0.5 * dt, b.dKE, b.max_abs_dKEdt);
    const double ratio_dKE = (b.dKE != 0.0) ? a.dKE / b.dKE : 0.0;
    const double ratio_rate = (b.max_abs_dKEdt != 0.0) ? a.max_abs_dKEdt / b.max_abs_dKEdt : 0.0;
    std::printf("   ratio(dt : dt/2)  net dKE = %.3f   max|dKE/dt| = %.3f   (SSP-RK3 => ~8x for 3rd order)\n",
                ratio_dKE, ratio_rate);

    // machine-readable
    FILE* fj = std::fopen("data/dt_note.json", "w");
    std::fprintf(fj, "{\n");
    std::fprintf(fj, "  \"N\": %d, \"p\": %d, \"Re\": %.1f, \"Tspin\": %.4f, \"Tmeasure\": %.4f,\n", N, p, Re, Tspin, Tmeasure);
    std::fprintf(fj, "  \"dt\": %.10e, \"dt_half\": %.10e,\n", dt, 0.5 * dt);
    std::fprintf(fj, "  \"net_dKE_dt\": %.10e, \"net_dKE_dt_half\": %.10e, \"ratio_net_dKE\": %.6f,\n", a.dKE, b.dKE, ratio_dKE);
    std::fprintf(fj, "  \"max_abs_dKEdt_dt\": %.10e, \"max_abs_dKEdt_dt_half\": %.10e, \"ratio_rate\": %.6f\n",
                a.max_abs_dKEdt, b.max_abs_dKEdt, ratio_rate);
    std::fprintf(fj, "}\n");
    std::fclose(fj);
    std::printf(" wrote data/dt_note.json\n");
    return 0;
}
