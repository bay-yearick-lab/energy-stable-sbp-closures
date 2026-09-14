from __future__ import annotations

import unittest

import numpy as np

import cavity_compatible_comparator as comparator


class CompatibleCavityComparatorTests(unittest.TestCase):
    def test_frozen_pairs_have_exactly_matched_state_counts(self):
        for bmf_npts, mac_cells in comparator.GRIDS:
            bmf = comparator.RouteCCavityStepper(bmf_npts, 4, 100.0, comparator.DT, 64.0)
            mac = comparator.MACCavityStepper(mac_cells, 100.0, comparator.DT)
            self.assertEqual(bmf.nv, mac.velocity_size)
            self.assertEqual(bmf.nq, mac.pressure_size)

    def test_mac_has_one_pressure_null_mode(self):
        mac = comparator.MACCavityStepper(6, 100.0, comparator.DT)
        self.assertEqual(mac.pressure_size - np.linalg.matrix_rank(mac.divergence), 1)
        self.assertLess(np.max(np.abs(mac.gradient @ np.ones(mac.pressure_size))), 2.0e-12)

    def test_gate_certificate_passes(self):
        certificate = comparator.gate_certificate()
        comparator.verify_gate(certificate)

    def test_cavity_reference_score_uses_shared_arithmetic(self):
        u = np.asarray([value for _, value in comparator.CAVITY_REFERENCE100_U_CENTERLINE])
        v = np.asarray([value for _, value in comparator.CAVITY_REFERENCE100_V_CENTERLINE])
        self.assertEqual(comparator.centerline_score(u, v)["combined_relative_l2"], 0.0)


if __name__ == "__main__":
    unittest.main()

