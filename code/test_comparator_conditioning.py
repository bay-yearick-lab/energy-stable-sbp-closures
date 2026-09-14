"""Unit gates for the comparator-conditioning fairness table.

Two structural claims carry the spline columns of the table and are checked
here directly against the repository's own constructors:

  G1  the BMG metric identity `H_BMG = B^{-T} M B^{-1}`, which is what makes
      the reported `H_BMG` column a derived object rather than a third
      independent left side;
  G2  `cond_2(M)` at degree `p = 4` is independent of `N` to one percent
      between `N = 41` and `N = 161`, the predeclared N-independence claim.

The remaining checks guard the bandwidth and kernel bookkeeping the table
prints.
"""
from __future__ import annotations

import unittest

import numpy as np

import boundary_spline_analysis as bsa
import s6_comparator_conditioning as cc


class BMGMetricIdentityTests(unittest.TestCase):
    def test_bmg_metric_equals_b_inverse_transpose_mass_b_inverse(self):
        for npts, degree in ((41, 4), (81, 6), (41, 8)):
            with self.subTest(npts=npts, degree=degree):
                _x, _knots, bmat, _gmat = bsa.clamped_bspline_matrices(npts, degree)
                mass, _adv = bsa.bspline_mass_and_advection(npts, degree)
                _x1, _dmat, hmat = bsa.bmg_sbp_first_derivative_matrix(npts, degree)
                # Multiplied-back form B^T H B = M, arithmetically independent
                # of the constructor's own B^{-T} M B^{-1} assembly.
                residual = float(np.max(np.abs(bmat.T @ hmat @ bmat - mass)))
                self.assertLess(residual / float(np.max(np.abs(mass))), 1.0e-11)
                # Solve-based route: H B = B^{-T} M, again different arithmetic.
                residual = float(np.max(np.abs(
                    hmat @ bmat - np.linalg.solve(bmat.T, mass))))
                self.assertLess(residual / float(np.max(np.abs(hmat @ bmat))),
                                1.0e-10)

    def test_bmg_metric_is_symmetric_positive_definite(self):
        _x, _d, hmat = bsa.bmg_sbp_first_derivative_matrix(41, 6)
        self.assertLess(float(np.max(np.abs(hmat - hmat.T))),
                        1.0e-12 * float(np.max(np.abs(hmat))))
        self.assertGreater(float(np.linalg.eigvalsh(hmat).min()), 0.0)


class MassConditionNumberTests(unittest.TestCase):
    def test_degree_four_mass_condition_is_n_independent_to_one_percent(self):
        conds = []
        for npts in (41, 161):
            mass, _adv = bsa.bspline_mass_and_advection(npts, 4)
            conds.append(float(np.linalg.cond(mass, 2)))
        spread = (max(conds) - min(conds)) / max(conds)
        self.assertLess(spread, 0.01, f"cond_2(M) spread {spread:.4g} at p=4: {conds}")

    def test_sample_matrix_condition_is_n_independent_for_every_degree(self):
        for degree in (4, 6, 8):
            with self.subTest(degree=degree):
                conds = []
                for npts in (41, 161):
                    _x, _k, bmat, _g = bsa.clamped_bspline_matrices(npts, degree)
                    conds.append(float(np.linalg.cond(bmat, 2)))
                spread = (max(conds) - min(conds)) / max(conds)
                self.assertLess(spread, 0.01)


class MatrixStatsTests(unittest.TestCase):
    def test_bandwidth_and_nnz_of_a_known_tridiagonal(self):
        mat = np.diag(np.ones(9)) + np.diag(0.25 * np.ones(8), 1) \
            + np.diag(0.25 * np.ones(8), -1)
        stats = cc.matrix_stats("T", mat)
        self.assertEqual(stats["bandwidth"], 1)
        self.assertEqual(stats["nnz"], 9 + 8 + 8)
        self.assertEqual(stats["kernel_dim"], 0)

    def test_assembled_first_derivative_has_the_constant_kernel(self):
        _x, dmat = bsa.explicit_fd_matrix(41, 6)
        stats = cc.matrix_stats("D", dmat)
        self.assertEqual(stats["kernel_dim"], 1)
        self.assertEqual(stats["numerical_rank"], 40)
        self.assertLess(stats["cond_2_effective"], stats["cond_2"])

    def test_bmf_staggered_gradient_has_full_row_rank(self):
        ops = bsa.bmf_staggered_operators(41, 6)
        stats = cc.matrix_stats("G", ops.d10)
        self.assertEqual(stats["shape"], [40, 41])
        self.assertEqual(stats["kernel_dim"], 0)
        self.assertAlmostEqual(stats["cond_2"], stats["cond_2_effective"], places=6)


if __name__ == "__main__":
    unittest.main()
