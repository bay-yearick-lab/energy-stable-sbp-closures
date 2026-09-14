#!/usr/bin/env python3
"""Regression gates for coefficient-space tensor-compatible flow operators."""
from __future__ import annotations

import unittest

import numpy as np

from bmf_hdiv_ns2d import BMFHDivNS2D
from compatible_sem_flow import CompatibleSEMFlow2D
from tensor_compatible_cavity import TensorCompatibleCavityStepper2D
from tensor_compatible_flow import (
    TensorCompatibleFlow2D,
    TensorCompatibleFlow3D,
    TensorCompatibleSaddle2D,
    VelocityPair,
    relative_pair_error,
)


def _active_pair(pair: VelocityPair) -> VelocityPair:
    return VelocityPair(ux=pair.ux[1:-1, :], uz=pair.uz[:, 1:-1])


def _legacy_active_pair(stepper: BMFHDivNS2D, vector: np.ndarray) -> VelocityPair:
    split = (stepper.npts - 2) * (stepper.npts - 1)
    return VelocityPair(
        ux=np.asarray(vector[:split]).reshape(stepper.npts - 2, stepper.npts - 1),
        uz=np.asarray(vector[split:]).reshape(stepper.npts - 1, stepper.npts - 2),
    )


class TensorCompatibleFlowTests(unittest.TestCase):
    def test_matrix_free_saddle_matches_dense_coefficient_oracle(self) -> None:
        flow = TensorCompatibleFlow2D(7, 4)
        solver = TensorCompatibleSaddle2D(
            flow,
            alpha=50.0,
            viscosity=0.01,
            relative_tolerance=1.0e-11,
        )
        rng = np.random.default_rng(704)
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
        self.assertLess(actual.relative_residual, 2.0e-11)
        self.assertLess(velocity_error, 2.0e-11)
        self.assertLess(pressure_error, 1.0e-10)
        self.assertLess(actual.divergence_relative_residual, 2.0e-11)
        self.assertLess(actual.pressure_gauge_relative_residual, 2.0e-11)
        self.assertEqual(
            solver.storage_record()["multidimensional_dense_operator_entries"],
            0,
        )

        primal_pressure = solver.primal_pressure(actual.dual_pressure)
        reconstructed_dual = flow.axis.mass1 @ primal_pressure @ flow.axis.mass1.T
        self.assertLess(
            np.linalg.norm(reconstructed_dual - actual.dual_pressure)
            / max(np.linalg.norm(actual.dual_pressure), 1.0e-300),
            2.0e-12,
        )

    def test_recurring_cavity_step_preserves_exact_constraints_and_ledger(self) -> None:
        stepper = TensorCompatibleCavityStepper2D(
            npts=7,
            degree=4,
            reynolds=100.0,
            dt=0.02,
            relative_tolerance=1.0e-11,
        )
        state = stepper.initial_state()
        for _ in range(3):
            result = stepper.step(state)
            state = result.state
            self.assertLess(result.saddle.relative_residual, 2.0e-11)
            self.assertLess(result.saddle.divergence_relative_residual, 2.0e-11)
            self.assertLess(
                result.saddle.pressure_gauge_relative_residual,
                2.0e-11,
            )
            self.assertLess(result.energy["relative_balance_residual"], 5.0e-9)
            stepper.flow.require_no_penetration(state.ux, state.uz)
        observable = stepper.native_wall_observable(state)
        self.assertEqual(observable["maximum_normal_flux"], 0.0)
        self.assertTrue(np.isfinite(observable["lid_reaction"]))
        self.assertEqual(
            stepper.storage_record()["multidimensional_dense_operator_entries"],
            0,
        )

    def test_compatible_sem_uses_the_same_recurring_step(self) -> None:
        flow = CompatibleSEMFlow2D(degree=4, nelem=2)
        stepper = TensorCompatibleCavityStepper2D(
            npts=flow.npts,
            degree=flow.degree,
            reynolds=100.0,
            dt=0.02,
            flow=flow,
        )
        result = stepper.step(stepper.initial_state())
        self.assertLess(result.saddle.relative_residual, 1.0e-11)
        self.assertLess(result.saddle.divergence_relative_residual, 2.0e-11)
        self.assertLess(result.saddle.pressure_gauge_relative_residual, 2.0e-11)
        self.assertLess(result.energy["relative_balance_residual"], 5.0e-9)
        self.assertLess(
            stepper.native_wall_observable(result.state)["maximum_normal_flux"],
            2.0e-13,
        )

    def test_all_residual_channels_match_dense_nodal_oracle(self) -> None:
        for degree in (2, 4, 6, 8):
            with self.subTest(degree=degree):
                npts = degree + 5
                tensor = TensorCompatibleFlow2D(
                    npts=npts,
                    degree=degree,
                    penalty_gamma=4.0 * degree**2,
                )
                dense = BMFHDivNS2D(
                    npts=npts,
                    degree=degree,
                    penalty_gamma=4.0 * degree**2,
                )
                rng = np.random.default_rng(800 + degree)
                ux = rng.standard_normal(tensor.ux_shape)
                uz = rng.standard_normal(tensor.uz_shape)
                ux[[0, -1], :] = 0.0
                uz[:, [0, -1]] = 0.0
                dual_pressure = rng.standard_normal(tensor.pressure_shape)
                zero = lambda x, z: np.zeros(np.broadcast_shapes(np.shape(x), np.shape(z)))
                lid = lambda x, z: np.broadcast_to(
                    np.isclose(np.asarray(z, dtype=float), 1.0).astype(float),
                    np.broadcast_shapes(np.shape(x), np.shape(z)),
                )
                channels = tensor.channel_residuals(
                    ux, uz, dual_pressure, lid, zero
                )

                nodal = tensor.nodal_velocity(ux, uz)
                active_nodal = dense._pack_active(nodal.ux, nodal.uz)

                mass_nodal = dense.velocity_mass @ active_nodal
                mass_active = _legacy_active_pair(dense, mass_nodal)
                mass_oracle = tensor.coefficient_dual_from_nodal(
                    mass_active.ux, mass_active.uz
                )
                self.assertLess(
                    relative_pair_error(_active_pair(channels.mass), mass_oracle),
                    2.0e-12,
                )

                viscous_nodal = dense.viscous_matrix @ active_nodal
                viscous_active = _legacy_active_pair(dense, viscous_nodal)
                viscous_oracle = tensor.coefficient_dual_from_nodal(
                    viscous_active.ux, viscous_active.uz
                )
                self.assertLess(
                    relative_pair_error(_active_pair(channels.viscous), viscous_oracle),
                    2.0e-12,
                )

                nodal_dual_pressure = tensor.nodal_dual_pressure(dual_pressure)
                pressure_nodal = dense.divergence_matrix.T @ nodal_dual_pressure.ravel()
                pressure_active = _legacy_active_pair(dense, pressure_nodal)
                pressure_oracle = tensor.coefficient_dual_from_nodal(
                    pressure_active.ux, pressure_active.uz
                )
                self.assertLess(
                    relative_pair_error(_active_pair(channels.pressure), pressure_oracle),
                    2.0e-12,
                )

                cux, cuz, _ledger = dense.skew_convection(nodal.ux, nodal.uz)
                full_convection = np.concatenate([cux.ravel(), cuz.ravel()])
                full_dual = dense.velocity_mass_full @ full_convection
                full_split = int(np.prod(dense.core.ux_shape))
                full_ux_dual = full_dual[:full_split].reshape(dense.core.ux_shape)
                full_uz_dual = full_dual[full_split:].reshape(dense.core.uz_shape)
                convection_oracle = VelocityPair(
                    ux=np.asarray(
                        tensor.axis.greville0.T
                        @ full_ux_dual
                        @ tensor.axis.greville1
                    ),
                    uz=np.asarray(
                        tensor.axis.greville1.T
                        @ full_uz_dual
                        @ tensor.axis.greville0
                    ),
                )
                self.assertLess(
                    relative_pair_error(
                        channels.convection, convection_oracle
                    ),
                    2.0e-12,
                )

                historical_active = dense.velocity_mass @ dense._pack_active(cux, cuz)
                historical_pair = _legacy_active_pair(dense, historical_active)
                historical_coefficient = tensor.coefficient_dual_from_nodal(
                    historical_pair.ux, historical_pair.uz
                )
                self.assertGreater(
                    relative_pair_error(
                        _active_pair(channels.convection), historical_coefficient
                    ),
                    1.0e-6,
                )

                wall_nodal = dense._assemble_viscous_rhs(zero, zero, lid, zero)
                wall_active = _legacy_active_pair(dense, wall_nodal)
                wall_oracle = tensor.coefficient_dual_from_nodal(
                    wall_active.ux, wall_active.uz
                )
                self.assertLess(
                    relative_pair_error(_active_pair(channels.wall), wall_oracle),
                    2.0e-12,
                )

                divergence_nodal = dense.core.divergence(nodal.ux, nodal.uz)
                divergence_oracle = (
                    tensor.axis.greville1 @ channels.divergence
                    @ tensor.axis.greville1.T
                )
                difference = np.linalg.norm(divergence_nodal - divergence_oracle)
                scale = max(np.linalg.norm(divergence_nodal), 1.0e-300)
                self.assertLess(difference / scale, 2.0e-12)

                energy = tensor.convective_energy_production(ux, uz)
                energy_scale = np.linalg.norm(ux) * np.linalg.norm(
                    channels.convection.ux
                ) + np.linalg.norm(uz) * np.linalg.norm(channels.convection.uz)
                self.assertLess(abs(energy) / max(energy_scale, 1.0e-300), 2.0e-12)

    def test_storage_uses_fixed_bandwidth_factors(self) -> None:
        for degree in (2, 4, 6, 8):
            operator = TensorCompatibleFlow2D(degree + 9, degree)
            record = operator.storage_record()
            self.assertLessEqual(record["maximum_factor_bandwidth"], degree)
            self.assertLess(
                record["stored_factor_entries"],
                32 * degree * operator.npts**2,
            )

    def test_three_dimensional_residual_matches_dense_tensor_oracle(self) -> None:
        operator = TensorCompatibleFlow3D(7, 4)
        axis = operator.axis
        rng = np.random.default_rng(417)
        ux = rng.standard_normal(operator.ux_shape)
        uy = rng.standard_normal(operator.uy_shape)
        uz = rng.standard_normal(operator.uz_shape)
        ux[[0, -1], :, :] = 0.0
        uy[:, [0, -1], :] = 0.0
        uz[:, :, [0, -1]] = 0.0
        pressure = rng.standard_normal(operator.pressure_shape)
        channels = operator.channel_residuals(ux, uy, uz, pressure)

        def apply(factors, field):
            arrays = tuple(
                factor.toarray() if hasattr(factor, "toarray") else np.asarray(factor)
                for factor in factors
            )
            return np.einsum(
                "ia,jb,kc,abc->ijk",
                *arrays,
                field,
                optimize=True,
            )

        def moments(factors, values):
            arrays = tuple(
                factor.toarray() if hasattr(factor, "toarray") else np.asarray(factor)
                for factor in factors
            )
            return np.einsum(
                "ia,jb,kc,ijk->abc",
                *arrays,
                values,
                optimize=True,
            )

        basis = (
            (axis.b0, axis.b1, axis.b1),
            (axis.b1, axis.b0, axis.b1),
            (axis.b1, axis.b1, axis.b0),
        )
        derivative = (
            (axis.d0, axis.d1, axis.d1),
            (axis.d1, axis.d0, axis.d1),
            (axis.d1, axis.d1, axis.d0),
        )
        fields = (ux, uy, uz)
        values = tuple(apply(basis[index], fields[index]) for index in range(3))
        weight = axis.w[:, None, None] * axis.w[None, :, None] * axis.w[None, None, :]
        convection_oracle = []
        for component in range(3):
            advective = np.zeros_like(values[component])
            conservative = np.zeros_like(fields[component])
            for direction in range(3):
                trial = list(basis[component])
                trial[direction] = derivative[component][direction]
                advective += values[direction] * apply(tuple(trial), fields[component])
                test = list(basis[component])
                test[direction] = derivative[component][direction]
                conservative += moments(
                    tuple(test), weight * values[direction] * values[component]
                )
            convection_oracle.append(
                0.5 * (moments(basis[component], weight * advective) - conservative)
            )

        mass_oracle = (
            apply((axis.mass0, axis.mass1, axis.mass1), ux),
            apply((axis.mass1, axis.mass0, axis.mass1), uy),
            apply((axis.mass1, axis.mass1, axis.mass0), uz),
        )
        viscous_oracle = (
            apply((axis.stiffness0, axis.mass1, axis.mass1), ux)
            + apply((axis.mass0, operator.tangential_factor, axis.mass1), ux)
            + apply((axis.mass0, axis.mass1, operator.tangential_factor), ux),
            apply((operator.tangential_factor, axis.mass0, axis.mass1), uy)
            + apply((axis.mass1, axis.stiffness0, axis.mass1), uy)
            + apply((axis.mass1, axis.mass0, operator.tangential_factor), uy),
            apply((operator.tangential_factor, axis.mass1, axis.mass0), uz)
            + apply((axis.mass1, operator.tangential_factor, axis.mass0), uz)
            + apply((axis.mass1, axis.mass1, axis.stiffness0), uz),
        )

        def relative(actual, expected):
            return np.linalg.norm(actual - expected) / max(
                np.linalg.norm(expected), 1.0e-300
            )

        for actual, expected in zip(
            (channels.mass.ux, channels.mass.uy, channels.mass.uz), mass_oracle
        ):
            self.assertLess(relative(actual, expected), 2.0e-12)
        for actual, expected in zip(
            (channels.viscous.ux, channels.viscous.uy, channels.viscous.uz),
            viscous_oracle,
        ):
            self.assertLess(relative(actual, expected), 2.0e-12)
        for actual, expected in zip(
            (
                channels.convection.ux,
                channels.convection.uy,
                channels.convection.uz,
            ),
            convection_oracle,
        ):
            self.assertLess(relative(actual, expected), 2.0e-12)

        divergence_oracle = apply(
            (axis.incidence, np.eye(axis.derivative_size), np.eye(axis.derivative_size)),
            ux,
        )
        divergence_oracle += apply(
            (np.eye(axis.derivative_size), axis.incidence, np.eye(axis.derivative_size)),
            uy,
        )
        divergence_oracle += apply(
            (np.eye(axis.derivative_size), np.eye(axis.derivative_size), axis.incidence),
            uz,
        )
        self.assertLess(relative(channels.divergence, divergence_oracle), 2.0e-12)

        energy = operator.convective_energy_production(ux, uy, uz)
        energy_scale = sum(
            np.linalg.norm(field) * np.linalg.norm(moment)
            for field, moment in zip(
                fields,
                (
                    channels.convection.ux,
                    channels.convection.uy,
                    channels.convection.uz,
                ),
            )
        )
        self.assertLess(abs(energy) / max(energy_scale, 1.0e-300), 2.0e-12)


if __name__ == "__main__":
    unittest.main()
