#!/usr/bin/env python3
"""Regression gates for the mixed compatible spectral-element comparator."""
from __future__ import annotations

import unittest

import numpy as np

from compatible_sem_flow import CompatibleSEMFlow2D
from tensor_compatible_flow import TensorCompatibleSaddle2D, VelocityPair


class CompatibleSEMFlowTests(unittest.TestCase):
    def test_histopolant_derivative_factorization_and_bandwidth(self) -> None:
        for degree in (2, 4, 6, 8):
            with self.subTest(degree=degree):
                flow = CompatibleSEMFlow2D(degree, 3)
                self.assertLess(flow.exact_sequence_residual(), 3.0e-14)
                self.assertLessEqual(
                    flow.axis.storage_record()["maximum_factor_bandwidth"],
                    2 * degree,
                )

    def test_compatible_residual_preserves_divergence_and_energy(self) -> None:
        flow = CompatibleSEMFlow2D(4, 2)
        rng = np.random.default_rng(442)
        ux = rng.standard_normal(flow.ux_shape)
        uz = rng.standard_normal(flow.uz_shape)
        ux[[0, -1], :] = 0.0
        uz[:, [0, -1]] = 0.0
        divergence = flow.divergence(ux, uz)
        evaluated_divergence = flow.axis.b1 @ divergence @ flow.axis.b1.T
        velocity = flow.quadrature_velocity(ux, uz)
        pointwise_divergence = (
            flow.axis.d0 @ ux @ flow.axis.b1.T
            + flow.axis.b1 @ uz @ flow.axis.d0.T
        )
        self.assertLess(
            np.linalg.norm(evaluated_divergence - pointwise_divergence)
            / max(np.linalg.norm(pointwise_divergence), 1.0e-300),
            2.0e-12,
        )
        convection = flow.convection_dual(ux, uz)
        energy = float(np.sum(ux * convection.ux) + np.sum(uz * convection.uz))
        scale = (
            np.linalg.norm(ux) * np.linalg.norm(convection.ux)
            + np.linalg.norm(uz) * np.linalg.norm(convection.uz)
        )
        self.assertLess(abs(energy) / max(scale, 1.0e-300), 2.0e-12)
        self.assertEqual(velocity.ux.shape, pointwise_divergence.shape)

    def test_requested_point_factors_reproduce_quadrature_factors(self) -> None:
        flow = CompatibleSEMFlow2D(6, 3)
        primal, derivative = flow.evaluation_factors(flow.axis.q)
        self.assertLess(
            np.linalg.norm((primal - flow.axis.b0).toarray()),
            2.0e-13,
        )
        self.assertLess(
            np.linalg.norm((derivative - flow.axis.b1).toarray()),
            2.0e-12,
        )

    def test_matrix_free_saddle_matches_dense_comparator_oracle(self) -> None:
        flow = CompatibleSEMFlow2D(4, 2)
        solver = TensorCompatibleSaddle2D(
            flow,
            alpha=1.0,
            viscosity=0.1,
            relative_tolerance=1.0e-11,
            maximum_iterations=6000,
        )
        rng = np.random.default_rng(421)
        rhs = VelocityPair(
            ux=rng.standard_normal(flow.ux_shape),
            uz=rng.standard_normal(flow.uz_shape),
        )
        actual = solver.solve(rhs)
        oracle = solver.dense_oracle(rhs)
        velocity_error = (
            np.linalg.norm(actual.ux - oracle.ux)
            + np.linalg.norm(actual.uz - oracle.uz)
        ) / max(np.linalg.norm(oracle.ux) + np.linalg.norm(oracle.uz), 1.0e-300)
        pressure_error = np.linalg.norm(
            actual.dual_pressure - oracle.dual_pressure
        ) / max(np.linalg.norm(oracle.dual_pressure), 1.0e-300)
        self.assertLess(velocity_error, 2.0e-11)
        self.assertLess(pressure_error, 1.0e-10)
        self.assertLess(actual.divergence_relative_residual, 2.0e-11)
        self.assertLess(actual.pressure_gauge_relative_residual, 2.0e-11)


if __name__ == "__main__":
    unittest.main()
