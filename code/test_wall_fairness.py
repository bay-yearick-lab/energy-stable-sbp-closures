"""Unit gates for the wall-fairness comparator repairs.

Four structural claims carry the new comparators and are checked here against
the repository's own constructors:

  G1  every new compact row is exact on polynomials up to its design degree,
      and the derived compact-baseline wall coefficients equal the classical values;
  G2  the SAT-enforced acoustic generator satisfies the energy identity
      `d/dt (p^T H p + u^T H u) = 2 tau (u_0^2 + u_N^2) <= 0` on random data;
  G3  the narrow second derivative has a symmetric positive semidefinite `A`
      whose null space is exactly the constants;
  G4  the narrow second derivative is exact on quadratics at every row.

The registry display name of the SBP family is checked too, since the class
carries no penalty term and the incumbent label claimed one.
"""
from __future__ import annotations

import unittest

import numpy as np

import boundary_spline_analysis as bsa
import s6_wall_fairness as wf


class CompactBaselineClosureTests(unittest.TestCase):
    def test_derived_wall_row_matches_the_classical_coefficients(self):
        expected = {3.0: (-17.0 / 6.0, 3.0 / 2.0, 3.0 / 2.0, -1.0 / 6.0),
                    2.0: (-5.0 / 2.0, 2.0, 1.0 / 2.0, 0.0)}
        for alpha, classical in expected.items():
            with self.subTest(alpha=alpha):
                coeff = bsa.compact_baseline_wall_row_coefficients(alpha)
                self.assertLess(float(np.max(np.abs(coeff - np.asarray(classical)))), 1.0e-12)

    def test_wall_row_design_order(self):
        s = np.arange(4, dtype=float)
        for alpha, order in ((3.0, 4), (2.0, 3)):
            with self.subTest(alpha=alpha):
                coeff = bsa.compact_baseline_wall_row_coefficients(alpha)
                for k in range(order + 1):
                    lhs = (1.0 if k == 1 else 0.0) + alpha * (float(k) if k >= 1 else 0.0)
                    self.assertLess(abs(float(coeff @ (s ** k)) - lhs), 1.0e-11,
                                    msg=f"alpha={alpha} degree={k}")
                k = order + 1
                lhs = (1.0 if k == 1 else 0.0) + alpha * (float(k) if k >= 1 else 0.0)
                self.assertGreater(abs(float(coeff @ (s ** k)) - lhs), 1.0e-6,
                                   msg="the row must fail one degree above its design order")

    def test_near_wall_pade_rows_hit_their_design_order(self):
        for alpha, half, order in ((0.25, 1, 4), (1.0 / 3.0, 2, 6)):
            with self.subTest(alpha=alpha):
                coeff = bsa.pade_symmetric_row_coefficients(alpha, half)
                s = np.arange(-half, half + 1, dtype=float)
                for k in range(order + 1):
                    def deriv(t: float) -> float:
                        return float(k) * t ** (k - 1) if k >= 1 else 0.0
                    lhs = alpha * deriv(-1.0) + deriv(0.0) + alpha * deriv(1.0)
                    self.assertLess(abs(float(coeff @ (s ** k)) - lhs), 1.0e-11,
                                    msg=f"alpha={alpha} degree={k}")

    def test_assembled_rows_are_compact_and_exact_at_their_design_orders(self):
        """Row-by-row exactness of the implicit pair `A f' = B f`.

        The eliminated operator `A^{-1} B` is dense, so the wall row's own
        order limits the global accuracy; the design order of each row is a
        statement about the row, which is what is gated here. The ladder is
        wall row four, second node four, third node six.
        """
        npts = 61
        design = {0: 4, 1: 4, 2: 6}
        for order in (4, 6, 8):
            with self.subTest(order=order):
                x, amat, bmat = bsa.compact_fd_matrices_compact_baseline(npts, order, 1.0, 3.0)
                offsets = np.argwhere(np.abs(amat) > 1.0e-14)
                bandwidth = int(np.max(np.abs(offsets[:, 0] - offsets[:, 1])))
                self.assertEqual(bandwidth, 1, "the compact-baseline closure keeps a tridiagonal left side")
                self.assertGreater(abs(amat[0, 1]), 1.0e-12,
                                   "the wall row must not fall back to an identity left side")
                half = bsa._compact_rhs_half_width(order)
                for degree in range(order + 1):
                    f = x ** degree
                    df = degree * x ** (degree - 1) if degree > 0 else np.zeros_like(x)
                    residual = np.abs(amat @ df - bmat @ f)
                    self.assertLess(float(np.max(residual[half:-half])), 1.0e-9,
                                    msg=f"interior row, degree {degree}")
                    for row in range(half):
                        if degree <= design[row]:
                            self.assertLess(float(residual[row]), 1.0e-9,
                                            msg=f"row {row}, degree {degree}")


class SATEnergyTests(unittest.TestCase):
    def test_sbp_identity_of_the_underlying_pair(self):
        for npts, order in ((41, 4), (41, 6), (61, 8)):
            with self.subTest(npts=npts, order=order):
                _x, dmat, hmat = bsa.sbp_fd_matrix(npts, order)
                bmat = hmat @ dmat + dmat.T @ hmat
                expected = np.zeros((npts, npts))
                expected[0, 0] = -1.0
                expected[-1, -1] = 1.0
                self.assertLess(float(np.max(np.abs(bmat - expected))), 1.0e-12)

    def test_energy_rate_matches_the_derivation_and_is_nonpositive(self):
        rng = np.random.default_rng(11)
        for tau in (-1.0, -2.0):
            for npts in (41, 61):
                with self.subTest(tau=tau, npts=npts):
                    _x, gen, metric = bsa.sbp_sat_acoustic_generator(npts, 6, 1.0, tau)
                    for _trial in range(6):
                        z = rng.standard_normal(2 * npts)
                        rate = float(2.0 * z @ metric @ (gen @ z))
                        predicted = 2.0 * tau * (z[npts] ** 2 + z[2 * npts - 1] ** 2)
                        self.assertLess(abs(rate - predicted), 1.0e-10 * max(1.0, abs(predicted)))
                        self.assertLessEqual(rate, 1.0e-12)

    def test_penalty_makes_the_closed_box_generator_nonexpansive(self):
        for tau in (-1.0, -2.0):
            for npts in (41, 81):
                with self.subTest(tau=tau, npts=npts):
                    self.assertLessEqual(wf.sat_abscissa(npts, 6, tau), 1.0e-12)

    def test_velocity_only_penalty_without_the_pressure_term_is_unstable(self):
        record = wf.sat_energy_record()
        self.assertGreater(record["velocity_only_penalty_abscissa"], 1.0e-3)


class NarrowSecondDerivativeTests(unittest.TestCase):
    def test_A_is_symmetric_positive_semidefinite_with_constant_null_space(self):
        for npts in (41, 61):
            with self.subTest(npts=npts):
                _x, _d2, _h, amat = bsa.sbp_narrow_second_derivative_matrix(npts, 6)
                scale = float(np.max(np.abs(amat)))
                self.assertLess(float(np.max(np.abs(amat - amat.T))) / scale, 1.0e-13)
                eig = np.sort(np.linalg.eigvalsh(0.5 * (amat + amat.T)))
                self.assertGreater(eig[0], -1.0e-9 * scale)
                self.assertGreater(eig[1], 1.0e-3)
                ones = np.ones(npts)
                self.assertLess(float(np.max(np.abs(amat @ ones))) / scale, 1.0e-12)

    def test_second_derivative_is_exact_on_quadratics(self):
        for npts in (41, 61):
            with self.subTest(npts=npts):
                x, d2, _h, _a = bsa.sbp_narrow_second_derivative_matrix(npts, 6)
                for degree, value in ((0, 0.0), (1, 0.0), (2, 2.0)):
                    residual = float(np.max(np.abs(d2 @ (x ** degree) - value)))
                    self.assertLess(residual, 1.0e-8, msg=f"degree {degree}")

    def test_interior_stencil_is_narrow_and_sixth_order(self):
        npts = 61
        _x, d2, _h, _a = bsa.sbp_narrow_second_derivative_matrix(npts, 6)
        row = d2[npts // 2]
        support = np.flatnonzero(np.abs(row) > 1.0e-10)
        self.assertEqual(int(support[-1] - support[0]) + 1, 7)
        x = np.linspace(0.0, 1.0, npts)
        for degree in range(2, 8):
            exact = degree * (degree - 1) * x ** (degree - 2)
            residual = float(np.max(np.abs((d2 @ (x ** degree) - exact)[8:-8])))
            self.assertLess(residual, 1.0e-6, msg=f"interior degree {degree}")

    def test_incumbent_sbp6_bo_second_derivative_is_the_wide_composition(self):
        npts = 41
        _x, d2 = bsa.sbp6_bo_second_derivative_matrix(npts, 1.0)
        _x1, d1, _h = bsa.sbp6_bo_fd_matrix(npts, 1.0)
        residual = float(np.max(np.abs(d2 - d1 @ d1)) / np.max(np.abs(d2)))
        self.assertLess(residual, 1.0e-12)


class MarchProtocolTests(unittest.TestCase):
    def test_local_march_reproduces_the_library_path(self):
        npts = 41
        for label, kind in (("SBP6", "SBP"), ("C6", "C")):
            with self.subTest(label=label):
                op = bsa.build_operator(kind, 6, npts)
                local = wf.march_nodal(op.x, op.matrix, t_end=0.3)
                res = bsa.simulate_reflected_acoustics(
                    op, wf.p_exact, wf.u_exact, 0.3, cfl=wf.CFL,
                    interior_strip=wf.STRIP, record_count=4)
                self.assertLess(abs(local["trimmed"] - float(res.error_history[-1]))
                                / max(1.0e-14, abs(local["trimmed"])), 1.0e-10)


class RegistryTests(unittest.TestCase):
    def test_sbp_family_is_not_displayed_as_sat(self):
        self.assertEqual(bsa.scheme_display_name("SBP6"), "SBP")
        self.assertEqual(bsa.FAMILY_DISPLAY_NAMES["SBP"], "SBP")


if __name__ == "__main__":
    unittest.main()
