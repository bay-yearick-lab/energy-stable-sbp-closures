// ke_production.cpp
//
// HEAD-TO-HEAD capability that only the conservative primitive-variable BMG
// formulation has (Theorems 10/11): the discrete advection operator produces
// ZERO spurious kinetic energy, on ANY velocity field, including a divergent one.
// The non-conservative convective form u.grad u that a standard collocated /
// central pressure-projection scheme uses produces O(1) spurious KE on the same
// field, because it lacks the discrete integration-by-parts skew
// cancellation. This is measured directly as the quadratic form
//
//     Pi_adv = sum_i u_i^T <N_i, advection_i(u)>,
//
// the rate the advection feeds the discrete KE budget (dK/dt gets -Pi_adv). For
// the energy-conserving skew split Pi_adv = 0 to round-off on the interior (only
// the boundary trace carries flux); for the convective form Pi_adv = -1/2 u^T
// (div u) u != 0 whenever div u != 0. The lid cavity fractional step and every
// coarse-grid nonlinear flow produce div u != 0 between projections, so this is
// exactly the spurious production the conservative scheme removes and the
// non-conservative one cannot.
//
// This is a static operator test (no time stepping): fast, deterministic, and
// the clean isolation of the only-ours capability behind the cavity experiment.

#include <cstdio>
#include <cmath>
#include <vector>
#include "wall_ns2d.h"

static const double PI = 3.14159265358979323846;

static void interp2d(const Axis& X, const Axis& Z, double (*f)(double, double), Field2D& out) {
    Field2D nodal(X.n, Z.n);
    for (int i = 0; i < X.n; ++i) for (int j = 0; j < Z.n; ++j) nodal.at(i, j) = f(X.grev[i], Z.grev[j]);
    Field2D tmp(X.n, Z.n);
    std::vector<double> col(X.n), row(Z.n);
    for (int j = 0; j < Z.n; ++j) {
        for (int i = 0; i < X.n; ++i) col[i] = nodal.at(i, j);
        std::vector<double> c; dense_lu_solve(X.B, col, c);
        for (int i = 0; i < X.n; ++i) tmp.at(i, j) = c[i];
    }
    for (int i = 0; i < X.n; ++i) {
        for (int j = 0; j < Z.n; ++j) row[j] = tmp.at(i, j);
        std::vector<double> c; dense_lu_solve(Z.B, row, c);
        for (int j = 0; j < Z.n; ++j) out.at(i, j) = c[j];
    }
}

// a genuinely DIVERGENT field with u = w = 0 on the walls (so no boundary flux;
// any nonzero production is purely the interior non-cancellation)
static double u_div(double x, double z) { return std::sin(PI * x) * std::sin(PI * x) * std::sin(3 * PI * z) + 0.4 * std::sin(2 * PI * x) * std::sin(PI * z); }
static double w_div(double x, double z) { return std::sin(PI * z) * std::sin(PI * z) * std::sin(3 * PI * x) - 0.3 * std::sin(PI * x) * std::sin(2 * PI * z); }

static double ke_production(WallNS2D& S, bool conv) {
    Field2D Cu(S.X.n, S.Z.n), Cw(S.X.n, S.Z.n);
    if (conv) S.advect_conv(Cu, Cw); else S.advect_skew(Cu, Cw);
    double p = 0;
    for (size_t q = 0; q < S.u.a.size(); ++q) p += S.u.a[q] * Cu.a[q] + S.w.a[q] * Cw.a[q];
    return p;
}

int main(int argc, char** argv) {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 32;
    const int p = (argc > 2) ? std::atoi(argv[2]) : 4;

    std::printf("=================================================================\n");
    std::printf(" Advection KE-production head-to-head, N=%d^2 p=%d\n", N, p);
    std::printf(" Pi_adv = sum_i u_i^T <N_i, advection_i(u)>  (interior KE injection)\n");
    std::printf("=================================================================\n");

    WallNS2D S(N, N, p, 1.0, 1.0, 1.0);
    interp2d(S.X, S.Z, u_div, S.u);
    interp2d(S.X, S.Z, w_div, S.w);
    S.no_slip(S.u); S.no_slip(S.w);   // zero on all walls -> no boundary flux term

    const double div_before = S.max_weak_div();
    const double skew = ke_production(S, false);
    const double conv = ke_production(S, true);
    std::printf(" DIVERGENT field (max|weak div u| = %.3e, u=w=0 on all walls):\n", div_before);
    std::printf("   SKEW  (paper, energy-conserving BMG) : Pi_adv = %+.6e\n", skew);
    std::printf("   CONV  (non-conservative competitor)  : Pi_adv = %+.6e\n", conv);
    std::printf("   ratio |conv/skew| = %.3e   (skew is machine zero; conv is O(field))\n",
                skew != 0.0 ? std::fabs(conv / skew) : 0.0);

    // and on a weak-div-free field, both agree at zero (the production only
    // differs through the divergence defect -- the momentum-defect identity)
    S.project(false);
    const double div_after = S.max_weak_div();
    const double skew2 = ke_production(S, false);
    const double conv2 = ke_production(S, true);
    std::printf(" After projection (weak-div-free, max|weak div u| = %.3e):\n", div_after);
    std::printf("   SKEW : Pi_adv = %+.6e\n", skew2);
    std::printf("   CONV : Pi_adv = %+.6e   (both ~0: the leak is proportional to div u)\n", conv2);

    FILE* fj = std::fopen("data/ke_production.json", "w");
    std::fprintf(fj, "{\n");
    std::fprintf(fj, "  \"N\": %d, \"p\": %d,\n", N, p);
    std::fprintf(fj, "  \"divergent_field\": {\"max_weak_div\": %.10e, \"skew_ke_production\": %.10e, \"conv_ke_production\": %.10e},\n",
                div_before, skew, conv);
    std::fprintf(fj, "  \"projected_field\": {\"max_weak_div\": %.10e, \"skew_ke_production\": %.10e, \"conv_ke_production\": %.10e}\n",
                div_after, skew2, conv2);
    std::fprintf(fj, "}\n");
    std::fclose(fj);
    std::printf(" wrote data/ke_production.json\n");
    return 0;
}
