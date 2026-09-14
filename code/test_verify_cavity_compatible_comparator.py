from __future__ import annotations

import unittest

import verify_cavity_compatible_comparator as verifier


def _row(method: str, grid: int, states: int, error: float) -> dict:
    return {
        "method": method,
        "grid": grid,
        "velocity_states": states,
        "centerline_discrepancy_percentage_points": error,
        "physical_accuracy_pass": True,
        "sensitivity_pass": True,
        "invariants_pass": True,
    }


class MatchedAccuracyAnalysisTests(unittest.TestCase):
    def test_cross_grid_state_advantage_is_not_lost_to_equal_state_pairing(self):
        rows = [
            _row("BMF4", 11, 180, 2.6054),
            _row("MAC2", 10, 180, 10.6900),
            _row("MAC2", 18, 612, 3.1987),
        ]
        pairs = verifier.matched_accuracy_pairs(rows)
        strongest = pairs[0]
        self.assertEqual((strongest["bmf_grid"], strongest["mac_grid"]), (11, 18))
        self.assertAlmostEqual(
            strongest["bmf_velocity_state_reduction_fraction"],
            1.0 - 180.0 / 612.0,
        )
        self.assertGreater(strongest["bmf_velocity_state_reduction_fraction"], 0.25)

    def test_less_accurate_bmf_pair_is_ineligible(self):
        rows = [
            _row("BMF4", 11, 180, 3.3),
            _row("MAC2", 18, 612, 3.2),
        ]
        self.assertEqual(verifier.matched_accuracy_pairs(rows), [])

    def test_failed_sensitivity_excludes_cross_grid_pair(self):
        bmf = _row("BMF4", 11, 180, 2.6)
        bmf["sensitivity_pass"] = False
        rows = [bmf, _row("MAC2", 18, 612, 3.2)]
        self.assertEqual(verifier.matched_accuracy_pairs(rows), [])


if __name__ == "__main__":
    unittest.main()
