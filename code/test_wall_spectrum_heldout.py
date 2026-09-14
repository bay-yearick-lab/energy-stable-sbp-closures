"""Focused tests for the frozen wall-spectrum held-out protocol."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import s6_wall_spectrum_heldout as heldout
import s6_wall_spectrum_heldout_runner as runner
import s6_wall_spectrum_kernel_revision as revision
import s6_wall_spectrum_full_propagator as full_propagator


class HeldoutProtocolTests(unittest.TestCase):
    def test_exact_images_return_after_one_cycle(self) -> None:
        pressure, velocity = heldout.exact_pair(0.055, 16.0 * np.pi)
        x = np.linspace(0.0, 1.0, 1001)
        self.assertLess(np.max(np.abs(pressure(x, 2.0) - pressure(x, 0.0))), 1.0e-12)
        self.assertLess(np.max(np.abs(velocity(x, 2.0) - velocity(x, 0.0))), 1.0e-12)

    def test_primary_matrix_has_sixty_unique_cases(self) -> None:
        identifiers = {
            heldout.case_id(sigma, carrier, encounters, method)
            for sigma in heldout.HOLDOUT_SIGMAS
            for carrier in heldout.HOLDOUT_K0_OVER_PI
            for encounters in heldout.HOLDOUT_ENCOUNTERS
            for method in heldout.METHODS
        }
        self.assertEqual(len(identifiers), 60)

    def test_tampered_prediction_binding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "prediction.json"
            raw = root / "raw.json"
            prediction.write_text(json.dumps({"artifact_state": "immutable_pre_result_predictions", "predictions": []}))
            raw.write_text(json.dumps({"prediction_sha256": "forged", "results": []}))
            with self.assertRaises(ValueError):
                heldout.summarize(prediction, raw, root / "summary.json")

    def test_bl_t6_declared_candidate_is_stable_on_primary_grid(self) -> None:
        operator = heldout.dense_operator("CSB-T6", heldout.NPTS)
        derivative = operator.matrix
        npts = derivative.shape[0]
        generator = np.block([
            [np.zeros((npts, npts)), -derivative[:, 1:-1]],
            [-derivative[1:-1, :], np.zeros((npts - 2, npts - 2))],
        ])
        self.assertLess(np.max(np.linalg.eigvals(generator).real), 1.0e-9)

    def test_rk4_step_map_matches_four_stage_update(self) -> None:
        matrix = np.array([[0.0, -2.0], [3.0, 0.0]])
        state = np.array([0.4, -0.7])
        step = 0.013
        k1 = matrix @ state
        k2 = matrix @ (state + 0.5 * step * k1)
        k3 = matrix @ (state + 0.5 * step * k2)
        k4 = matrix @ (state + step * k3)
        staged = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        mapped = runner.rk4_step_map(matrix, step) @ state
        np.testing.assert_allclose(mapped, staged, rtol=2.0e-15, atol=2.0e-15)

    def test_revision_matrix_is_fresh_and_finite(self) -> None:
        self.assertTrue(set(revision.SIGMAS).isdisjoint(heldout.HOLDOUT_SIGMAS))
        self.assertTrue(set(revision.CARRIERS).isdisjoint(heldout.HOLDOUT_K0_OVER_PI))
        self.assertTrue(set(revision.ENCOUNTERS).isdisjoint(heldout.HOLDOUT_ENCOUNTERS))
        identifiers = {
            heldout.case_id(sigma, carrier, encounters, method)
            for sigma in revision.SIGMAS
            for carrier in revision.CARRIERS
            for encounters in revision.ENCOUNTERS
            for method in heldout.METHODS
        }
        self.assertEqual(len(identifiers), 60)

    def _assert_accelerated_matches_serial(self, method: str, npts: int) -> None:
        sigma = 0.075
        carrier = 12.0
        time_value = 0.025
        exact_pressure, exact_velocity = heldout.exact_pair(sigma, carrier * np.pi)
        if method in ("BMF6", "BMG6", "SBP6"):
            operator = heldout.ba.build_operator(
                heldout.ba.scheme_kind(method), heldout.ORDER, npts
            )
        else:
            operator = heldout.dense_operator(method, npts)
        serial = heldout.ba.simulate_reflected_acoustics(
            operator,
            exact_pressure,
            exact_velocity,
            time_value,
            cfl=heldout.CFL,
            record_count=2,
        )
        data, accelerated_state, _steps = runner.matrix_power_state(
            method, npts, sigma, carrier, time_value
        )
        accelerated_p, accelerated_u = runner.unpack_state(
            method, data, accelerated_state
        )
        np.testing.assert_allclose(
            accelerated_p, serial.pressure, rtol=2.0e-11, atol=2.0e-12
        )
        np.testing.assert_allclose(
            accelerated_u, serial.velocity, rtol=2.0e-11, atol=2.0e-12
        )
        if method == "BMF6":
            npressure = int(data["pressure_size"])
            constraint_values = accelerated_state[[npressure, accelerated_state.size - 1]]
        elif method == "BMG6":
            npressure = int(data["pressure_size"])
            constraint_values = accelerated_state[[npressure, accelerated_state.size - 1]]
        else:
            constraint_values = accelerated_u[[0, -1]]
        np.testing.assert_allclose(constraint_values, 0.0, rtol=0.0, atol=2.0e-13)
        serial_metrics = heldout.score_fields(
            np.asarray(serial.grid, dtype=float),
            np.asarray(serial.pressure, dtype=float),
            np.asarray(serial.velocity, dtype=float),
            exact_pressure,
            exact_velocity,
            time_value,
            carrier * np.pi,
            sigma,
        )
        accelerated_metrics = heldout.score_fields(
            np.asarray(data["x"], dtype=float),
            accelerated_p,
            accelerated_u,
            exact_pressure,
            exact_velocity,
            time_value,
            carrier * np.pi,
            sigma,
        )
        for metric in (
            "phase_shift",
            "waveform_distortion",
            "leakage",
            "leakage_excess",
            "full_rel_l2",
        ):
            self.assertAlmostEqual(
                accelerated_metrics[metric], serial_metrics[metric], delta=2.0e-11
            )

    def test_accelerated_samegrid_sbp_faithfulness(self) -> None:
        self._assert_accelerated_matches_serial("SBP6", 25)

    def test_accelerated_horth_bmg_faithfulness(self) -> None:
        self._assert_accelerated_matches_serial("BMG6", 25)

    def test_accelerated_horth_bmf_faithfulness(self) -> None:
        self._assert_accelerated_matches_serial("BMF6", 25)

    def test_full_propagator_matches_direct_exponential_and_constraints(self) -> None:
        from scipy.linalg import expm

        sigma = 0.075
        carrier = 12.0
        time_value = 0.037
        for method in ("SBP6", "BMG6", "BMF6"):
            decomposition = full_propagator.factor(method)
            data = decomposition["data"]
            initial = runner.initial_states(data, [(sigma, carrier)])
            predicted, _cost = full_propagator.propagate(
                decomposition, initial, time_value
            )
            direct = expm(data["matrix"] * time_value) @ initial
            np.testing.assert_allclose(
                predicted, direct, rtol=3.0e-11, atol=3.0e-12
            )
            pressure, velocity = runner.unpack_state(
                method, data, predicted[:, 0]
            )
            if method in ("BMF6", "BMG6"):
                npressure = int(data["pressure_size"])
                constraints = predicted[[npressure, predicted.shape[0] - 1], 0]
            else:
                constraints = velocity[[0, -1]]
            np.testing.assert_allclose(constraints, 0.0, rtol=0.0, atol=5.0e-12)

    def test_full_propagator_family_is_untouched(self) -> None:
        used_sigmas = set(heldout.HOLDOUT_SIGMAS) | set(revision.SIGMAS) | {heldout.reported.SIGMA}
        used_carriers = set(heldout.HOLDOUT_K0_OVER_PI) | set(revision.CARRIERS) | {heldout.reported.K0 / np.pi}
        used_encounters = set(heldout.HOLDOUT_ENCOUNTERS) | set(revision.ENCOUNTERS) | {2, 6}
        self.assertTrue(set(full_propagator.SIGMAS).isdisjoint(used_sigmas))
        self.assertTrue(set(full_propagator.CARRIERS).isdisjoint(used_carriers))
        self.assertTrue(set(full_propagator.ENCOUNTERS).isdisjoint(used_encounters))


if __name__ == "__main__":
    unittest.main()
