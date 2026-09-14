#!/usr/bin/env python3
"""Faithful matrix-free BDF2/AB2 cavity step in coefficient coordinates."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from tensor_compatible_flow import (
    Array,
    SaddleSolution2D,
    TensorCompatibleFlow2D,
    TensorCompatibleSaddle2D,
    ScalarField,
    VelocityPair,
)


@dataclass(frozen=True)
class TensorCavityState2D:
    """Coefficient state and one-step history for BDF2/AB2 time integration."""

    ux: Array
    uz: Array
    previous_ux: Array
    previous_uz: Array
    previous_convection_ux: Array
    previous_convection_uz: Array
    dual_pressure: Array
    step_index: int
    time: float


@dataclass(frozen=True)
class TensorCavityStep2D:
    """Advanced state with saddle and discrete energy diagnostics."""

    state: TensorCavityState2D
    saddle: SaddleSolution2D
    energy: dict[str, float]


class TensorCompatibleCavityStepper2D:
    """Reusable pressure-coupled step with direct weak convective moments."""

    def __init__(
        self,
        npts: int,
        degree: int,
        reynolds: float,
        dt: float,
        penalty_gamma: float | None = None,
        relative_tolerance: float = 1.0e-11,
        maximum_iterations: int = 4000,
        pressure_preconditioner: str = "product_schur",
        flow: TensorCompatibleFlow2D | None = None,
        lid_velocity: ScalarField | None = None,
    ) -> None:
        self.reynolds = float(reynolds)
        self.dt = float(dt)
        if not np.isfinite(self.reynolds) or self.reynolds <= 0.0:
            raise ValueError("reynolds must be finite and positive")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.viscosity = 1.0 / self.reynolds
        if flow is None:
            self.flow = TensorCompatibleFlow2D(
                npts=npts,
                degree=degree,
                penalty_gamma=penalty_gamma,
            )
        else:
            if flow.npts != int(npts) or flow.degree != int(degree):
                raise ValueError("provided flow must match npts and degree")
            self.flow = flow
        self.backward_euler = TensorCompatibleSaddle2D(
            self.flow,
            alpha=1.0 / self.dt,
            viscosity=self.viscosity,
            relative_tolerance=relative_tolerance,
            maximum_iterations=maximum_iterations,
            pressure_preconditioner=pressure_preconditioner,
        )
        self.bdf2 = TensorCompatibleSaddle2D(
            self.flow,
            alpha=1.5 / self.dt,
            viscosity=self.viscosity,
            relative_tolerance=relative_tolerance,
            maximum_iterations=maximum_iterations,
            pressure_preconditioner=pressure_preconditioner,
        )

        def zero(x: Array, z: Array) -> Array:
            return np.zeros(np.broadcast_shapes(np.shape(x), np.shape(z)))

        def lid(x: Array, z: Array) -> Array:
            shape = np.broadcast_shapes(np.shape(x), np.shape(z))
            return np.broadcast_to(
                np.isclose(np.asarray(z, dtype=float), self.flow.length).astype(float),
                shape,
            )

        self.lid_velocity = lid if lid_velocity is None else lid_velocity
        self.wall_load = self.flow.wall_load(self.lid_velocity, zero)

    def initial_state(self) -> TensorCavityState2D:
        ux = np.zeros(self.flow.ux_shape)
        uz = np.zeros(self.flow.uz_shape)
        return TensorCavityState2D(
            ux=ux,
            uz=uz,
            previous_ux=ux.copy(),
            previous_uz=uz.copy(),
            previous_convection_ux=np.zeros_like(ux),
            previous_convection_uz=np.zeros_like(uz),
            dual_pressure=np.zeros(self.flow.pressure_shape),
            step_index=0,
            time=0.0,
        )

    def convection(self, state: TensorCavityState2D) -> VelocityPair:
        return self.flow.convection_dual(state.ux, state.uz)

    def assemble_rhs(
        self,
        state: TensorCavityState2D,
        convection: VelocityPair,
    ) -> VelocityPair:
        if state.step_index == 0:
            history_ux = state.ux / self.dt
            history_uz = state.uz / self.dt
            convection_ux = convection.ux
            convection_uz = convection.uz
        else:
            history_ux = (2.0 * state.ux - 0.5 * state.previous_ux) / self.dt
            history_uz = (2.0 * state.uz - 0.5 * state.previous_uz) / self.dt
            convection_ux = 2.0 * convection.ux - state.previous_convection_ux
            convection_uz = 2.0 * convection.uz - state.previous_convection_uz
        history = self.flow.mass_action(history_ux, history_uz)
        return VelocityPair(
            ux=history.ux - convection_ux + self.viscosity * self.wall_load.ux,
            uz=history.uz - convection_uz + self.viscosity * self.wall_load.uz,
        )

    def solve_rhs(
        self,
        state: TensorCavityState2D,
        rhs: VelocityPair,
    ) -> SaddleSolution2D:
        solver = self.backward_euler if state.step_index == 0 else self.bdf2
        return solver.solve(rhs)

    @staticmethod
    def _duality(velocity: VelocityPair, dual: VelocityPair) -> float:
        return float(np.sum(velocity.ux * dual.ux) + np.sum(velocity.uz * dual.uz))

    def _energy_ledger(
        self,
        old: TensorCavityState2D,
        new_velocity: VelocityPair,
        convection: VelocityPair,
        saddle: SaddleSolution2D,
    ) -> dict[str, float]:
        mass_new = self.flow.mass_action(new_velocity.ux, new_velocity.uz)
        if old.step_index == 0:
            rate = VelocityPair(
                ux=(new_velocity.ux - old.ux) / self.dt,
                uz=(new_velocity.uz - old.uz) / self.dt,
            )
            explicit_convection = convection
        else:
            rate = VelocityPair(
                ux=(
                    1.5 * new_velocity.ux
                    - 2.0 * old.ux
                    + 0.5 * old.previous_ux
                )
                / self.dt,
                uz=(
                    1.5 * new_velocity.uz
                    - 2.0 * old.uz
                    + 0.5 * old.previous_uz
                )
                / self.dt,
            )
            explicit_convection = VelocityPair(
                ux=2.0 * convection.ux - old.previous_convection_ux,
                uz=2.0 * convection.uz - old.previous_convection_uz,
            )
        temporal = self._duality(rate, mass_new)
        convective = self._duality(new_velocity, explicit_convection)
        viscous_dual = self.flow.viscous_action(new_velocity.ux, new_velocity.uz)
        viscous = self.viscosity * self._duality(new_velocity, viscous_dual)
        axis = self.flow.axis
        boundary_factor = self.flow.tangential_factor - axis.stiffness1
        ux_boundary = axis.mass0 @ new_velocity.ux
        ux_boundary = np.asarray(boundary_factor @ ux_boundary.T).T
        uz_boundary = boundary_factor @ new_velocity.uz
        uz_boundary = np.asarray(axis.mass0 @ uz_boundary.T).T
        boundary_enforcement = self.viscosity * self._duality(
            new_velocity,
            VelocityPair(ux=ux_boundary, uz=uz_boundary),
        )
        pressure_dual = self.flow.pressure_action(saddle.dual_pressure)
        pressure = self._duality(new_velocity, pressure_dual)
        wall = self.viscosity * self._duality(new_velocity, self.wall_load)
        volume_viscous = viscous - boundary_enforcement
        net_wall = wall - boundary_enforcement
        terms = (temporal, convective, volume_viscous, pressure, -net_wall)
        scale = max(sum(abs(value) for value in terms), 1.0e-300)
        balance = sum(terms)
        return {
            "temporal_power": temporal,
            "explicit_convective_power": convective,
            "viscous_power": viscous,
            "volume_viscous_power": volume_viscous,
            "boundary_enforcement_power": boundary_enforcement,
            "pressure_power": pressure,
            "nitsche_wall_power": wall,
            "net_wall_power": net_wall,
            "relative_balance_residual": abs(balance) / scale,
            "kinetic_energy": self.flow.kinetic_energy(
                new_velocity.ux, new_velocity.uz
            ),
        }

    def step(self, state: TensorCavityState2D) -> TensorCavityStep2D:
        self.flow.require_no_penetration(state.ux, state.uz)
        convection = self.convection(state)
        rhs = self.assemble_rhs(state, convection)
        saddle = self.solve_rhs(state, rhs)
        velocity = VelocityPair(ux=saddle.ux, uz=saddle.uz)
        energy = self._energy_ledger(state, velocity, convection, saddle)
        new_state = TensorCavityState2D(
            ux=saddle.ux,
            uz=saddle.uz,
            previous_ux=state.ux.copy(),
            previous_uz=state.uz.copy(),
            previous_convection_ux=convection.ux,
            previous_convection_uz=convection.uz,
            dual_pressure=saddle.dual_pressure,
            step_index=state.step_index + 1,
            time=state.time + self.dt,
        )
        return TensorCavityStep2D(state=new_state, saddle=saddle, energy=energy)

    def native_wall_observable(
        self,
        state: TensorCavityState2D,
    ) -> dict[str, float | str]:
        return self._wall_observable(
            self.strong_wall_traction(state),
            "strong",
        )

    def strong_wall_traction(
        self,
        state: TensorCavityState2D,
    ) -> dict[str, dict[str, Array]]:
        """Return the derivative-plus-pressure traction of the current state."""
        solver = self.backward_euler if state.step_index <= 1 else self.bdf2
        primal_pressure = solver.primal_pressure(state.dual_pressure)
        return self.flow.wall_traction(
            state.ux,
            state.uz,
            primal_pressure,
            viscosity=self.viscosity,
        )

    def nitsche_consistent_wall_traction(
        self,
        state: TensorCavityState2D,
    ) -> dict[str, dict[str, Array]]:
        """Return the boundary flux represented by the symmetric Nitsche residual."""
        traction = {
            name: {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in wall.items()
            }
            for name, wall in self.strong_wall_traction(state).items()
        }
        axis = self.flow.axis
        correction_scale = self.viscosity * self.flow.penalty
        for side, name in ((0, "x0"), (1, "x1")):
            trace_coefficients = axis.trace1[[side]] @ state.uz
            tangential_trace = np.asarray(
                axis.b0 @ np.asarray(trace_coefficients).T
            ).ravel()
            traction[name]["traction_z"] -= correction_scale * tangential_trace
        for side, name, coordinate in (
            (0, "z0", 0.0),
            (1, "z1", self.flow.length),
        ):
            tangential_trace = np.asarray(axis.b0 @ state.ux)
            tangential_trace = np.asarray(
                tangential_trace @ axis.trace1[[side]].T
            ).ravel()
            prescribed = np.asarray(
                self.lid_velocity(axis.q, np.full_like(axis.q, coordinate)),
                dtype=float,
            )
            traction[name]["traction_x"] -= correction_scale * (
                tangential_trace - prescribed
            )
        return traction

    def nitsche_wall_observable(
        self,
        state: TensorCavityState2D,
    ) -> dict[str, float | str]:
        """Return reaction and work from the residual-consistent wall flux."""
        return self._wall_observable(
            self.nitsche_consistent_wall_traction(state),
            "nitsche_consistent",
        )

    def _wall_observable(
        self,
        traction: dict[str, dict[str, Array]],
        estimator: str,
    ) -> dict[str, float | str]:
        """Integrate lid reaction, lid power, and normal flux."""
        top = traction["z1"]
        reaction = -float(np.sum(self.flow.axis.w * top["traction_x"]))
        lid_values = np.asarray(
            self.lid_velocity(
                self.flow.axis.q,
                np.full_like(self.flow.axis.q, self.flow.length),
            ),
            dtype=float,
        )
        power = -float(
            np.sum(self.flow.axis.w * top["traction_x"] * lid_values)
        )
        maximum_flux = max(
            float(np.max(np.abs(wall["normal_flux"])))
            for wall in traction.values()
        )
        return {
            "estimator": estimator,
            "lid_reaction": reaction,
            "lid_power": power,
            "maximum_normal_flux": maximum_flux,
        }

    def velocity_rate(
        self,
        new: TensorCavityState2D,
        old: TensorCavityState2D,
    ) -> float:
        difference = VelocityPair(ux=new.ux - old.ux, uz=new.uz - old.uz)
        mass = self.flow.mass_action(difference.ux, difference.uz)
        numerator = math.sqrt(max(self._duality(difference, mass), 0.0)) / self.dt
        denominator = max(
            math.sqrt(max(2.0 * self.flow.kinetic_energy(new.ux, new.uz), 0.0)),
            1.0,
        )
        return numerator / denominator

    def storage_record(self) -> dict[str, int]:
        backward_euler = self.backward_euler.storage_record()
        bdf2 = self.bdf2.storage_record()
        return {
            "flow_factor_entries": self.flow.storage_record()["stored_factor_entries"],
            "time_factor_entries": backward_euler["one_dimensional_factor_entries"]
            + bdf2["one_dimensional_factor_entries"],
            "pressure_preconditioner_entries": backward_euler[
                "pressure_preconditioner_entries"
            ]
            + bdf2["pressure_preconditioner_entries"],
            "multidimensional_dense_operator_entries": 0,
        }
