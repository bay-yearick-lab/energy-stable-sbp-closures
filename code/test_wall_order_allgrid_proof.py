from __future__ import annotations

from fractions import Fraction
import unittest

import numpy as np

import boundary_spline_analysis as ba
import wall_order_allgrid_proof as proof


class WallOrderAllGridProofTests(unittest.TestCase):
    def test_exact_collocation_matrix_matches_production_operator(self) -> None:
        for degree in (4, 6, 8):
            npts = degree + 9
            exact = proof.exact_collocation_matrix(degree, npts)
            floating = np.array(
                [[float(value) for value in row] for row in exact], dtype=float
            )
            production = ba.clamped_bspline_matrices(npts, degree, 1.0)[2]
            np.testing.assert_allclose(floating, production, rtol=0.0, atol=2.0e-14)

    def test_characteristic_root_intervals_account_for_every_root(self) -> None:
        for degree in (4, 6, 8):
            blocks = proof.characteristic_root_certificates(degree)
            self.assertEqual(len(blocks["collocation"]["inside_intervals"]), degree // 2)
            self.assertEqual(len(blocks["mass"]["inside_intervals"]), degree)
            for name in ("collocation", "mass"):
                block = blocks[name]
                self.assertTrue(block["palindromic"])
                self.assertTrue(block["all_roots_accounted_for"])
                self.assertTrue(block["no_unit_circle_root"])
                for left, right in block["inside_intervals"]:
                    self.assertLess(Fraction(left), Fraction(right))
                    self.assertGreaterEqual(Fraction(left), Fraction(-1))
                    self.assertLess(Fraction(right), Fraction(0))

    def test_exact_supersolutions_prove_uniform_one_and_infinity_norm_bounds(self) -> None:
        for degree in (4, 6, 8):
            certificate = proof.uniform_stability_certificate(degree)
            self.assertTrue(certificate["all_N_ge_p_plus_1"])
            self.assertTrue(certificate["inverse_checkerboard_nonnegative"])
            self.assertGreater(Fraction(certificate["minimum_exact_margin"]), 0)
            self.assertGreater(Fraction(certificate["inverse_inf_bound"]), 0)
            self.assertGreater(Fraction(certificate["inverse_one_bound"]), 0)

    def test_halfline_endpoint_constants_exclude_zero_for_both_routes(self) -> None:
        expected_sign = {
            ("BMF", 4): -1, ("BMF", 6): 1, ("BMF", 8): 1,
            ("BMG", 4): -1, ("BMG", 6): 1, ("BMG", 8): 1,
        }
        for (family, degree), sign in expected_sign.items():
            certificate = proof.halfline_endpoint_certificate(family, degree)
            lower = Fraction(certificate["constant_interval"][0])
            upper = Fraction(certificate["constant_interval"][1])
            self.assertLess(lower, upper)
            self.assertTrue(upper < 0 if sign < 0 else lower > 0)
            self.assertTrue(certificate["interval_excludes_zero"])
            self.assertTrue(certificate["endpoint_weights_localize_exponentially"])
            self.assertLess(Fraction(certificate["localization_rate_upper"]), 1)


if __name__ == "__main__":
    unittest.main()
