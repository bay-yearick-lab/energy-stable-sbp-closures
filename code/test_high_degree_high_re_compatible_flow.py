#!/usr/bin/env python3
"""Regression tests for the high-degree and high-Reynolds study driver."""
from __future__ import annotations

import unittest

import numpy as np

import high_degree_high_re_compatible_flow as study


class HighDegreeHighReCompatibleFlowTests(unittest.TestCase):
    """Validate matrix construction, exact-sequence reach, and diagnostics."""

    def test_degree_twelve_exact_sequence_for_both_families(self) -> None:
        """Degree-twelve factors preserve the one-dimensional derivative sequence."""
        for method in ("bmf", "compatible_sem"):
            flow = study._flow(method, 12, 2)
            residual = flow.axis.d0 - flow.axis.b1 @ flow.axis.incidence
            relative = np.linalg.norm(residual.toarray()) / np.linalg.norm(
                flow.axis.d0.toarray()
            )
            self.assertLess(relative, study.ALGEBRA_TOLERANCE)

    def test_degree_twelve_sem_endpoint_trace_is_exact(self) -> None:
        """Barycentric endpoint evaluation preserves exact normal trace ownership."""
        flow = study._flow("compatible_sem", 12, 2)
        expected = np.zeros_like(flow.axis.trace0)
        expected[0, 0] = 1.0
        expected[1, -1] = 1.0
        np.testing.assert_array_equal(flow.axis.trace0, expected)

    def test_degree_twelve_pressure_action_has_one_null_direction(self) -> None:
        """The active degree-twelve pressure action retains only the constant null."""
        for method in ("bmf", "compatible_sem"):
            flow = study._flow(method, 12, 2)
            gradient = study._pressure_gradient_matrix(flow)
            singular_values = np.linalg.svd(gradient, compute_uv=False)
            tolerance = (
                singular_values[0]
                * max(gradient.shape)
                * np.finfo(float).eps
            )
            rank = int(np.sum(singular_values > tolerance))
            self.assertEqual(gradient.shape[1] - rank, 1)

    def test_identical_flow_records_have_zero_joint_difference(self) -> None:
        """Every state and wall diagnostic vanishes under self-comparison."""
        row = {
            "samples": {
                "ux": [[1.0]],
                "uz": [[2.0]],
                "pressure": [[3.0]],
                "native_traction": {
                    wall: {
                        "traction_x": [1.0],
                        "traction_z": [2.0],
                        "normal_flux": [0.0],
                    }
                    for wall in ("x0", "x1", "z0", "z1")
                },
                "strong_traction": {
                    wall: {
                        "traction_x": [1.0],
                        "traction_z": [2.0],
                        "normal_flux": [0.0],
                    }
                    for wall in ("x0", "x1", "z0", "z1")
                },
            },
            "native_lid_reaction_statistics": {"mean": 1.0},
            "integrated_native_wall_work": 2.0,
        }
        comparison = study._comparison(row, row)
        self.assertTrue(all(value == 0.0 for value in comparison.values()))


if __name__ == "__main__":
    unittest.main()
