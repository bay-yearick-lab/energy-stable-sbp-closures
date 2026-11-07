# Benchmarks

All timings are single-thread CPU process time on one x86_64 desktop core (Python 3.14, NumPy 2.4; C++ built with g++ 13, `-O3`). CPU process time is used because it is insensitive to other load on the machine.

## Cost against accuracy: heat cavity, q = 6

`python benchmarks/cost_accuracy.py` (same protocol as `code/s6_wallclock_cost.py`). Complete march of the wall-bounded heat cavity to T = 2, operator assembly included, each family at its own stable RK4 step, median of five runs after one warmup. Each entry is relative L² error / CPU time. Committed data: `code/results/wallclock_cost.json`.

| N | E6 | C6 | SBP6 | DRP6 | BMF6 | BMG6 |
|---|---|---|---|---|---|---|
| 33 | 4.6e+00 / 2.0 ms | 3.1e+00 / 2.4 ms | 5.3e+00 / 2.3 ms | 4.1e+00 / 2.3 ms | 5.9e-02 / 8.3 ms | 7.5e-01 / 8.4 ms |
| 49 | 3.7e-01 / 4.6 ms | 2.8e-01 / 4.5 ms | 1.2e+00 / 4.7 ms | 2.8e-01 / 4.6 ms | 5.8e-04 / 20.8 ms | 1.6e-03 / 21.2 ms |
| 65 | 6.0e-02 / 7.9 ms | 3.5e-02 / 8.1 ms | 4.0e-01 / 8.0 ms | 2.5e-02 / 8.8 ms | 3.4e-05 / 51.7 ms | 1.0e-04 / 51.2 ms |
| 81 | 1.8e-02 / 14.3 ms | 3.8e-03 / 15.2 ms | 4.6e-02 / 15.8 ms | 3.7e-03 / 15.4 ms | 3.7e-06 / 114.3 ms | 1.2e-05 / 112.6 ms |

BMF6 at N = 49 (21 ms) is 6 to 80 times more accurate than every nodal family at N = 81 (14 to 16 ms), and below 4e-3 error the spline routes are the only ones on this ladder.

## Banded operator stack (C++)

`python code/operator_cost_benchmark.py` compiles `code/cavity/apps/opbench.cpp` against the header-only library in `code/cavity/include/` and times the 1-D clamped B-spline operators. Committed data: `code/results/operator_cost_scaling.json`.

| p | N | assembly | banded M solve | K apply | banded M + K + LU | dense N² |
|---|---|---|---|---|---|---|
| 4 | 513 | 7.6 ms | 7.5 µs | 1.5 µs | 0.11 MB | 2.1 MB |
| 4 | 4097 | 583 ms | 59 µs | 17 µs | 0.88 MB | 134 MB |
| 6 | 513 | 6.4 ms | 7.7 µs | 1.3 µs | 0.16 MB | 2.1 MB |
| 6 | 4097 | 591 ms | 62 µs | 21 µs | 1.28 MB | 134 MB |
| 8 | 513 | 4.1 ms | 8.1 µs | 1.5 µs | 0.21 MB | 2.1 MB |
| 8 | 4097 | 601 ms | 66 µs | 27 µs | 1.67 MB | 134 MB |

Solve and apply costs grow linearly in N (O(Np) work and storage). In d dimensions every operator application is d·N^(d−1) independent line solves of this kind.

## Other drivers

| Script | Measures |
|---|---|
| `code/benchmark_tensor_compatible_scaling.py` | tensor-compatible residual cost and storage scaling in 2-D and 3-D |
| `code/benchmark_tensor_pressure_step.py` | complete pressure-coupled cavity step |
| `code/benchmark_tensor_linear_components.py` | matrix-free 3-D linear residual components |
| `code/make_infsup_sweep.py` | matrix-free LOBPCG inf-sup sweep, 112 cells up to 513 basis functions per direction |
