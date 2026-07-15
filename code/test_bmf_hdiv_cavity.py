from __future__ import annotations

import copy
import unittest

import numpy as np

import s6_bmf_hdiv_cavity as cavity


class RouteCCavityTests(unittest.TestCase):
    def test_cavity_reference_re100_reference_has_both_centerlines_and_endpoints(self):
        self.assertEqual(len(cavity.CAVITY_REFERENCE100_U_CENTERLINE), 17)
        self.assertEqual(len(cavity.CAVITY_REFERENCE100_V_CENTERLINE), 17)
        self.assertEqual(cavity.CAVITY_REFERENCE100_U_CENTERLINE[0], (0.0, 0.0))
        self.assertEqual(cavity.CAVITY_REFERENCE100_U_CENTERLINE[-1], (1.0, 1.0))
        self.assertEqual(cavity.CAVITY_REFERENCE100_V_CENTERLINE[0], (0.0, 0.0))
        self.assertEqual(cavity.CAVITY_REFERENCE100_V_CENTERLINE[-1], (1.0, 0.0))

    def test_one_step_preserves_exact_discrete_mass_and_normal_trace(self):
        stepper = cavity.RouteCCavityStepper(npts=7, degree=4, reynolds=100.0, dt=0.02)
        state = stepper.initial_state()
        advanced = stepper.step(state)
        self.assertTrue(np.all(np.isfinite(advanced.velocity)))
        ux, uz = stepper.formulation._unpack_active(advanced.velocity)
        divergence = stepper.formulation.core.divergence(ux, uz)
        self.assertLess(stepper.formulation.core.pressure_norm(divergence), 2.0e-11)
        self.assertLess(stepper.formulation.core.normal_trace_max(ux, uz), 2.0e-13)

    def test_centerline_score_is_zero_for_reference_values(self):
        u = np.asarray([value for _coordinate, value in cavity.CAVITY_REFERENCE100_U_CENTERLINE])
        v = np.asarray([value for _coordinate, value in cavity.CAVITY_REFERENCE100_V_CENTERLINE])
        score = cavity.centerline_score(u, v)
        self.assertEqual(score["combined_relative_l2"], 0.0)
        self.assertEqual(score["maximum_absolute_error"], 0.0)

    def test_verifier_rejects_forged_divergence(self):
        record = cavity.smoke_record()
        cavity.verify_record(record, require_production=False)
        forged = copy.deepcopy(record)
        forged["runs"][0]["divergence_l2"] = 1.0
        with self.assertRaisesRegex((AssertionError, ValueError), "divergence"):
            cavity.verify_record(forged, require_production=False)


if __name__ == "__main__":
    unittest.main()
