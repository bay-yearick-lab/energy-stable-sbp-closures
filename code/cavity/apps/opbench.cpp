// opbench.cpp
//
// Implementation-performance microbenchmark for the clamped B-spline operator
// stack, single thread, CPU process time (load-robust on a shared box).
//
// 1D, for degree p in {4,6,8} and N up to 4097:
//   assemble_s   full Axis construction: knots, Greville, sampled basis, the
//                M/K/S coefficient bilinear forms, and the banded LU of M
//   factor_s     re-factoring the banded mass LU alone (amortized part)
//   msolve_s     one banded mass solve M^{-1} v (the recurring per-step work)
//   kapply_s     one bandwidth-limited K application (O(N p) recurring work)
//   band_bytes   storage actually needed by the banded representation
//                ((2p+1) N doubles per form + banded LU)
//   dense_bytes  N^2 doubles, the storage the dense convenience layout uses
//
// 2D (p=4), the coupled Navier-Stokes stack the cavity experiment runs:
//   setup_s      WallNS2D construction: axes + the one-time fast-diagonalization
//                Schur eigendecomposition (the amortized direct-solver setup)
//   step_s       one full RK3 step with three projections (recurring cost)
//   rss_mb       peak resident set
//
// Output: JSON on stdout (redirect to code/results/operator_cost_scaling.json).
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <chrono>
#include <sys/resource.h>
#include "wall_ns2d.h"

static double cpu_now() {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

// banded K apply: out = K v using only the 2p+1 central diagonals
static void band_apply(const Mat& K, int p, const std::vector<double>& v,
                       std::vector<double>& out) {
    const int n = K.r;
    for (int i = 0; i < n; ++i) {
        const int j0 = std::max(0, i - p), j1 = std::min(n - 1, i + p);
        double s = 0;
        for (int j = j0; j <= j1; ++j) s += K(i, j) * v[j];
        out[i] = s;
    }
}

int main(int argc, char** argv) {
    // mode "1d" or "2d" (default both). Run the 2D leg in its OWN process for
    // an uncontaminated peak-RSS reading (getrusage reports the high-water
    // mark, and the 1D leg's dense scratch forms at N=4097 dominate it).
    const std::string mode = (argc > 1) ? argv[1] : "both";
    const bool do1d = (mode == "1d" || mode == "both");
    const bool do2d = (mode == "2d" || mode == "both");
    const bool do3d = (mode == "3d");
    std::printf("{\n \"oned\": [\n");
    bool first = true;
    if (do1d) for (int p : {4, 6, 8}) {
        for (int N : {65, 129, 257, 513, 1025, 2049, 4097}) {
            double t0 = cpu_now();
            Axis ax(N, p, 1.0);
            const double assemble_s = cpu_now() - t0;

            // re-factor timing (banded LU of M)
            DenseMatrix Md(N);
            for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) Md(i, j) = ax.M(i, j);
            t0 = cpu_now();
            int fr = 3;
            for (int r = 0; r < fr; ++r) {
                BandSolver bs; bs.set_from_dense(Md, p, p); bs.factor();
            }
            const double factor_s = (cpu_now() - t0) / fr;

            std::vector<double> v(N, 1.0), out(N);
            // banded mass solve
            int reps = std::max(20, 200000 / N);
            t0 = cpu_now();
            for (int r = 0; r < reps; ++r) ax.Mfac.solve_inplace(v);
            const double msolve_s = (cpu_now() - t0) / reps;
            // banded K apply
            for (int i = 0; i < N; ++i) v[i] = 1.0;
            t0 = cpu_now();
            for (int r = 0; r < reps; ++r) band_apply(ax.K, p, v, out);
            const double kapply_s = (cpu_now() - t0) / reps;

            const long band_bytes = (long)(2 * p + 1) * N * 8 * 3;  // M, K + LU
            const long dense_bytes = (long)N * N * 8;
            std::printf("%s  {\"p\": %d, \"N\": %d, \"assemble_s\": %.6e, \"factor_s\": %.6e, "
                        "\"msolve_s\": %.6e, \"kapply_s\": %.6e, \"band_bytes\": %ld, "
                        "\"dense_bytes\": %ld}",
                        first ? "" : ",\n", p, N, assemble_s, factor_s, msolve_s, kapply_s,
                        band_bytes, dense_bytes);
            first = false;
            std::fflush(stdout);
        }
    }
    std::printf("\n ],\n \"twod\": [\n");
    first = true;
    if (do2d) for (int N : {32, 64, 96, 128}) {
        struct rusage ru0; getrusage(RUSAGE_SELF, &ru0);
        double t0 = cpu_now();
        WallNS2D S(N, N, 4, 1.0, 1.0, 1e-3);
        const double setup_s = cpu_now() - t0;
        // developed-ish state: a smooth divergence-free-ish start, then warmup
        for (int i = 0; i < S.X.n; ++i) for (int j = 0; j < S.Z.n; ++j) {
            const double x = S.X.grev[i], z = S.Z.grev[j];
            S.u.at(i, j) = std::sin(3.14159265358979 * x) * std::cos(3.14159265358979 * z);
            S.w.at(i, j) = -std::cos(3.14159265358979 * x) * std::sin(3.14159265358979 * z);
        }
        S.no_slip(S.u); S.no_slip(S.w); S.project(true);
        const double dt = 0.1 / N;
        for (int k = 0; k < 3; ++k) S.step(dt);
        const int nst = 20;
        t0 = cpu_now();
        for (int k = 0; k < nst; ++k) S.step(dt);
        const double step_s = (cpu_now() - t0) / nst;
        struct rusage ru; getrusage(RUSAGE_SELF, &ru);
        std::printf("%s  {\"N\": %d, \"p\": 4, \"setup_s\": %.6e, \"step_s\": %.6e, "
                    "\"rss_mb\": %.1f, \"rss_delta_mb\": %.1f}",
                    first ? "" : ",\n", N, setup_s, step_s, ru.ru_maxrss / 1024.0,
                    (ru.ru_maxrss - ru0.ru_maxrss) / 1024.0);
        first = false;
        std::fflush(stdout);
    }
    std::printf("\n ],\n \"threed\": [\n");
    first = true;
    // 3D bounded tensor-product diffusion scaling ("3d" mode): one heat step
    // = per direction, N^2 line kernels (banded mass solve + banded K apply)
    // over an N^3 coefficient block, OpenMP over lines; plus the projection
    // transform proxy (one dense NxN transform per direction over the block,
    // the fast-diagonalization pattern). Measured, not extrapolated.
    if (do3d) for (int N : {32, 48, 64, 96}) {
        Axis ax(N, 4, 1.0);
        std::vector<double> blk((size_t)N * N * N, 1.0), tmp(blk.size());
        const double mem_mb = 2.0 * blk.size() * 8.0 / 1048576.0;
        for (int nt : {1, 4, 8, 16}) {
#ifdef _OPENMP
            omp_set_num_threads(nt);
#endif
            const int reps = 3;
            double t0 = cpu_now();
            const auto w0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) {
                // three directional sweeps of line kernels
                for (int dir = 0; dir < 3; ++dir) {
#pragma omp parallel
                    {
                        std::vector<double> line(N), out(N);
#pragma omp for schedule(static)
                        for (int l = 0; l < N * N; ++l) {
                            const int a = l / N, b = l % N;
                            for (int k = 0; k < N; ++k) {
                                const size_t idx = dir == 0 ? ((size_t)k * N + a) * N + b
                                               : dir == 1 ? ((size_t)a * N + k) * N + b
                                                          : ((size_t)a * N + b) * N + k;
                                line[k] = blk[idx];
                            }
                            ax.Mfac.solve_inplace(line);
                            const int pdeg = 4;
                            for (int i = 0; i < N; ++i) {
                                const int j0 = std::max(0, i - pdeg), j1 = std::min(N - 1, i + pdeg);
                                double sacc = 0;
                                for (int j = j0; j <= j1; ++j) sacc += ax.K(i, j) * line[j];
                                out[i] = sacc;
                            }
                            for (int k = 0; k < N; ++k) {
                                const size_t idx = dir == 0 ? ((size_t)k * N + a) * N + b
                                               : dir == 1 ? ((size_t)a * N + k) * N + b
                                                          : ((size_t)a * N + b) * N + k;
                                tmp[idx] = out[k];
                            }
                        }
                    }
                }
            }
            const double step_cpu = (cpu_now() - t0) / reps;
            const double step_wall = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - w0).count() / reps;
            std::printf("%s  {\"N\": %d, \"threads\": %d, \"step_cpu_s\": %.4e, "
                        "\"step_wall_s\": %.4e, \"block_mem_mb\": %.1f}",
                        first ? "" : ",\n", N, nt, step_cpu, step_wall, mem_mb);
            first = false;
            std::fflush(stdout);
        }
    }
    std::printf("\n ]\n}\n");
    return 0;
}
