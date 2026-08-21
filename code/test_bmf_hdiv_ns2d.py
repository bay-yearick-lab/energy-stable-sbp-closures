from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bmf_hdiv_ns2d import BMFHDivNS2D
import check_bmf_hdiv_ns2d as checker
from check_bmf_hdiv_ns2d import build_certificate, verify_certificate


def _normal_zero_state(solver: BMFHDivNS2D, seed: int = 13):
    rng = np.random.default_rng(seed)
    ux = rng.standard_normal(solver.core.ux_shape)
    uz = rng.standard_normal(solver.core.uz_shape)
    ux[[0, -1], :] = 0.0
    uz[:, [0, -1]] = 0.0
    return ux, uz


def _projected_state(solver: BMFHDivNS2D, seed: int = 13):
    ux, uz = _normal_zero_state(solver, seed)
    projected = solver.core.project(ux, uz)
    return projected.ux, projected.uz


def _all_self_consistent_forgery_cases(valid: dict) -> list[dict]:
    cases: list[dict] = []
    extra = copy.deepcopy(valid)
    extra["fabricated"] = True
    cases.append(extra)

    zero = copy.deepcopy(valid)
    for state in zero["random_projected_states"]:
        for key in ("pressure_adjoint_unprojected", "pressure_cancellation_projected"):
            state[key].update(
                velocity_gradient_work=0.0,
                divergence_pressure_work=0.0,
                adjoint_residual=0.0,
                raw_scale=1.0,
                relative_adjoint_residual=0.0,
                relative_pressure_work=0.0,
            )
        state["convection_projected"].update(
            pointwise_divergence_l2_squared=0.0,
            pointwise_divergence_scale_squared=1.0,
            relative_pointwise_divergence=0.0,
            kinetic_energy_production=0.0,
            kinetic_energy_scale=1.0,
            relative_kinetic_energy_production=0.0,
            convective_momentum_flux_x_defect=0.0,
            convective_momentum_flux_z_defect=0.0,
            convective_momentum_flux_defect=0.0,
            convective_momentum_flux_scale=1.0,
            relative_convective_momentum_flux_defect=0.0,
        )
        state["content_sha256"] = checker.record_content_sha256(state)
    cases.append(zero)

    cloned = copy.deepcopy(valid)
    source = cloned["random_projected_states"][0]
    for target in cloned["random_projected_states"][1:]:
        for key in (
            "pressure_adjoint_unprojected",
            "pressure_cancellation_projected",
            "convection_projected",
        ):
            target[key] = copy.deepcopy(source[key])
        target["content_sha256"] = checker.record_content_sha256(target)
    cases.append(cloned)

    nitsche = copy.deepcopy(valid)
    gate = nitsche["nitsche"]
    gate.update(
        matrix_asymmetry_l2_squared=0.0,
        matrix_l2_squared=1.0,
        minimum_eigenvalue=1.0,
        maximum_eigenvalue=2.0,
    )
    gate["polynomial_consistency"].update(
        residual_l2_squared=0.0,
        load_l2_squared=1.0,
        tangential_trace_l2_squared=1.0,
    )
    gate["content_sha256"] = checker.record_content_sha256(gate)
    cases.append(nitsche)

    scope = copy.deepcopy(valid)
    scope["scope"] = {"not_established": "nothing"}
    scope["proof_debts"] = ["done", "done", "done"]
    cases.append(scope)

    negative = copy.deepcopy(valid)
    for record in negative["stokes_refinement"]:
        record["divergence_l2_squared"] = -1.0
        record["linear_residual_l2_squared"] = -1.0
        record["normal_trace_max"] = -1.0
        record["content_sha256"] = checker.record_content_sha256(record)
    cases.append(negative)

    missing = copy.deepcopy(valid)
    missing["random_projected_states"][0].pop("convection_projected")
    cases.append(missing)

    digest = copy.deepcopy(valid)
    digest["random_projected_states"][0]["input_sha256"] = "0" * 64
    cases.append(digest)
    return cases


class TangentialNitscheTests(unittest.TestCase):
    def test_matrix_is_symmetric_and_coercive_with_normal_traces_eliminated(self) -> None:
        solver = BMFHDivNS2D(npts=8, degree=3)
        asymmetry = np.linalg.norm(solver.viscous_matrix - solver.viscous_matrix.T, ord=np.inf)
        scale = np.linalg.norm(solver.viscous_matrix, ord=np.inf)
        eigenvalues = np.linalg.eigvalsh(solver.viscous_matrix)
        self.assertLess(asymmetry, 2.0e-13 * scale)
        self.assertGreater(eigenvalues[0], 1.0e-3)
        self.assertEqual(solver.penalty_gamma, 4.0 * solver.degree**2)

    def test_polynomial_consistency_includes_nonzero_tangential_wall_data(self) -> None:
        solver = BMFHDivNS2D(npts=8, degree=3)
        result = solver.polynomial_nitsche_consistency()
        self.assertGreater(result["tangential_trace_l2_squared"], 1.0e-2)
        self.assertLess(
            result["residual_l2_squared"] / result["load_l2_squared"],
            2.0e-24,
        )

    def test_tangential_deletion_creates_four_extra_pressure_nulls(self) -> None:
        solver = BMFHDivNS2D(npts=8, degree=3)
        rank = solver.divergence_rank_certificate()
        self.assertEqual(rank["weak_tangential_nullity"], 1)
        self.assertEqual(rank["strong_tangential_nullity"], 5)
        self.assertEqual(rank["weak_tangential_rank"], solver.core.pressure_size - 1)
        self.assertEqual(rank["weak_matrix_change_from_core"], 0.0)


class ConservationTests(unittest.TestCase):
    def test_adversarial_degree_refinement_sweep_has_nonzero_tangential_trace(self) -> None:
        for degree in (2, 3, 4, 6):
            for elements in (3, 5):
                with self.subTest(degree=degree, elements=elements):
                    solver = BMFHDivNS2D(npts=degree + elements, degree=degree)
                    ux, uz = _projected_state(
                        solver,
                        seed=1000 + 100 * degree + elements,
                    )
                    rng = np.random.default_rng(1001 + 100 * degree + elements)
                    pressure = solver.core.remove_pressure_mean(
                        rng.standard_normal(solver.core.pressure_shape)
                    )
                    convection = solver.skew_convection(ux, uz)[2]
                    momentum = solver.momentum_reaction_ledger(ux, uz, pressure)
                    pressure_work = solver.pressure_work_ledger(ux, uz, pressure)
                    self.assertGreater(solver.tangential_trace_l2_squared(ux, uz), 1.0e-4)
                    self.assertEqual(convection["normal_trace_max"], 0.0)
                    self.assertLess(convection["relative_pointwise_divergence"], 1.0e-12)
                    self.assertLess(convection["relative_kinetic_energy_production"], 1.0e-12)
                    self.assertLess(momentum["relative_momentum_balance_defect"], 1.0e-12)
                    self.assertLess(pressure_work["relative_pressure_work"], 1.0e-12)

    def test_pressure_adjoint_is_nontrivial_before_projection_and_cancels_after(self) -> None:
        solver = BMFHDivNS2D(npts=8, degree=3)
        ux, uz = _normal_zero_state(solver, seed=5)
        rng = np.random.default_rng(6)
        pressure = solver.core.remove_pressure_mean(
            rng.standard_normal(solver.core.pressure_shape)
        )
        unprojected = solver.pressure_work_ledger(ux, uz, pressure)
        projected = solver.core.project(ux, uz)
        cancelled = solver.pressure_work_ledger(
            projected.ux,
            projected.uz,
            pressure,
        )
        self.assertLess(unprojected["relative_adjoint_residual"], 2.0e-13)
        self.assertGreater(unprojected["relative_pressure_work"], 1.0e-3)
        self.assertLess(cancelled["relative_adjoint_residual"], 2.0e-13)
        self.assertLess(cancelled["relative_pressure_work"], 2.0e-12)
        self.assertGreater(unprojected["raw_scale"], 1.0e-8)

    def test_exact_quadrature_skew_convection_closes_energy_and_momentum(self) -> None:
        solver = BMFHDivNS2D(npts=8, degree=3)
        ux, uz = _projected_state(solver, seed=19)
        cux, cuz, ledger = solver.skew_convection(ux, uz)
        self.assertEqual(cux.shape, solver.core.ux_shape)
        self.assertEqual(cuz.shape, solver.core.uz_shape)
        self.assertLess(ledger["relative_pointwise_divergence"], 3.0e-12)
        self.assertLess(ledger["relative_kinetic_energy_production"], 3.0e-12)
        self.assertLess(
            ledger["relative_convective_momentum_flux_defect"],
            3.0e-12,
        )
        self.assertLess(ledger["normal_trace_max"], 2.0e-13)
        self.assertLess(ledger["relative_conservative_identity_residual"], 3.0e-13)
        self.assertGreaterEqual(
            ledger["gauss_exact_polynomial_degree"],
            ledger["maximum_galerkin_integrand_degree"],
        )
        self.assertGreater(ledger["kinetic_energy_scale"], 1.0e-8)
        self.assertGreater(ledger["convective_momentum_flux_scale"], 1.0e-8)

    def test_strong_normal_constraint_has_an_exact_wall_reaction_momentum_ledger(self) -> None:
        solver = BMFHDivNS2D(npts=8, degree=3)
        ux, uz = _projected_state(solver, seed=23)
        rng = np.random.default_rng(24)
        pressure = solver.core.remove_pressure_mean(
            rng.standard_normal(solver.core.pressure_shape)
        )

        ledger = solver.momentum_reaction_ledger(ux, uz, pressure)

        self.assertLess(ledger["relative_active_equation_residual"], 3.0e-12)
        self.assertLess(ledger["relative_constraint_support_defect"], 3.0e-12)
        self.assertLess(ledger["relative_momentum_balance_defect"], 3.0e-12)
        self.assertLess(ledger["relative_convective_projection_moment_defect"], 3.0e-12)
        self.assertGreater(ledger["constraint_reaction_scale"], 1.0e-8)


class StokesManufacturedSolutionTests(unittest.TestCase):
    def test_nonzero_tangential_wall_stokes_solution_converges(self) -> None:
        records = [
            BMFHDivNS2D(npts=npts, degree=2).solve_manufactured_stokes()
            for npts in (6, 8, 11)
        ]
        errors = [np.sqrt(record["velocity_error_l2_squared"]) for record in records]
        pressure_errors = [
            np.sqrt(record["pressure_mod_constant_error_l2_squared"])
            for record in records
        ]
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertGreater(pressure_errors[0], pressure_errors[1])
        self.assertGreater(pressure_errors[1], pressure_errors[2])
        rates = [
            np.log(errors[i - 1] / errors[i])
            / np.log(records[i - 1]["mesh_size"] / records[i]["mesh_size"])
            for i in (1, 2)
        ]
        self.assertGreater(min(rates), 1.7)
        pressure_rates = [
            np.log(pressure_errors[i - 1] / pressure_errors[i])
            / np.log(records[i - 1]["mesh_size"] / records[i]["mesh_size"])
            for i in (1, 2)
        ]
        self.assertGreater(min(pressure_rates), 1.7)
        for record in records:
            self.assertGreater(record["tangential_data_l2_squared"], 1.0)
            self.assertLess(record["divergence_l2_squared"], 2.0e-20)
            self.assertLess(record["normal_trace_max"], 2.0e-13)
            self.assertEqual(record["pressure_unknown"], "dual pressure H_Q p")
            self.assertGreater(record["pressure_exact_l2_squared"], 1.0e-2)


class CertificateIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.valid = build_certificate()

    def test_certificate_records_raw_norms_and_exact_formulation_metadata(self) -> None:
        verify_certificate(self.valid)
        self.assertEqual(self.valid["schema"], "bmf-hdiv-ns2d-c5-c8-v4")
        self.assertEqual(self.valid["formulation"]["normal_trace"], "strong")
        self.assertEqual(self.valid["formulation"]["tangential_trace"], "symmetric Nitsche")
        self.assertEqual(
            [record["seed"] for record in self.valid["random_projected_states"]],
            [41, 43, 47],
        )
        self.assertEqual(len(self.valid["nitsche_coercivity_sweep"]), 8)
        self.assertEqual(len(self.valid["adversarial_conservation_sweep"]), 8)
        self.assertGreater(
            min(record["minimum_eigenvalue"] for record in self.valid["nitsche_coercivity_sweep"]),
            0.1,
        )
        for record in self.valid["random_projected_states"]:
            self.assertRegex(record["input_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(record["content_sha256"], r"^[0-9a-f]{64}$")
            self.assertIn("pressure_adjoint_unprojected", record)
            self.assertIn("pressure_cancellation_projected", record)
            self.assertIn("convection_projected", record)
            self.assertIn("momentum_reaction", record)
        for record in self.valid["stokes_refinement"]:
            for key in (
                "velocity_error_l2_squared",
                "velocity_exact_l2_squared",
                "divergence_l2_squared",
                "tangential_data_l2_squared",
                "pressure_mod_constant_error_l2_squared",
                "pressure_exact_l2_squared",
            ):
                self.assertIn(key, record)
            self.assertRegex(record["input_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(record["content_sha256"], r"^[0-9a-f]{64}$")

    def test_verifier_rejects_nan_forged_metadata_and_wrong_derived_rate(self) -> None:
        nan_case = copy.deepcopy(self.valid)
        nan_case["stokes_refinement"][0]["velocity_error_l2_squared"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            verify_certificate(nan_case)

        forged = copy.deepcopy(self.valid)
        forged["formulation"]["tangential_trace"] = "strong deletion"
        with self.assertRaisesRegex(ValueError, "metadata"):
            verify_certificate(forged)

        wrong = copy.deepcopy(self.valid)
        wrong["stokes_observed_orders"][0] *= 0.5
        with self.assertRaisesRegex(ValueError, "derived"):
            verify_certificate(wrong)

    def test_verifier_rejects_forged_small_pressure_and_momentum_ratios(self) -> None:
        pressure = copy.deepcopy(self.valid)
        pressure["random_projected_states"][0]["pressure_cancellation_projected"][
            "relative_pressure_work"
        ] = 1.0e-13
        pressure["random_projected_states"][0]["content_sha256"] = (
            checker.record_content_sha256(pressure["random_projected_states"][0])
        )
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_certificate(pressure)

        momentum = copy.deepcopy(self.valid)
        momentum["random_projected_states"][0]["convection_projected"][
            "relative_convective_momentum_flux_defect"
        ] = 1.0e-13
        momentum["random_projected_states"][0]["content_sha256"] = (
            checker.record_content_sha256(momentum["random_projected_states"][0])
        )
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_certificate(momentum)

    def test_deterministic_rebuild_rejects_every_self_consistent_forgery(self) -> None:
        cases: list[dict] = []

        extra = copy.deepcopy(self.valid)
        extra["fabricated"] = True
        cases.append(extra)

        zero = copy.deepcopy(self.valid)
        for state in zero["random_projected_states"]:
            for key in ("pressure_adjoint_unprojected", "pressure_cancellation_projected"):
                lane = state[key]
                lane.update(
                    velocity_gradient_work=0.0,
                    divergence_pressure_work=0.0,
                    adjoint_residual=0.0,
                    raw_scale=1.0,
                    relative_adjoint_residual=0.0,
                    relative_pressure_work=0.0,
                )
            convection = state["convection_projected"]
            convection.update(
                pointwise_divergence_l2_squared=0.0,
                pointwise_divergence_scale_squared=1.0,
                relative_pointwise_divergence=0.0,
                kinetic_energy_production=0.0,
                kinetic_energy_scale=1.0,
                relative_kinetic_energy_production=0.0,
                convective_momentum_flux_x_defect=0.0,
                convective_momentum_flux_z_defect=0.0,
                convective_momentum_flux_defect=0.0,
                convective_momentum_flux_scale=1.0,
                relative_convective_momentum_flux_defect=0.0,
            )
            state["content_sha256"] = checker.record_content_sha256(state)
        cases.append(zero)

        cloned = copy.deepcopy(self.valid)
        source = cloned["random_projected_states"][0]
        for target in cloned["random_projected_states"][1:]:
            for key in (
                "pressure_adjoint_unprojected",
                "pressure_cancellation_projected",
                "convection_projected",
            ):
                target[key] = copy.deepcopy(source[key])
            target["content_sha256"] = checker.record_content_sha256(target)
        cases.append(cloned)

        nitsche = copy.deepcopy(self.valid)
        gate = nitsche["nitsche"]
        gate.update(
            matrix_asymmetry_l2_squared=0.0,
            matrix_l2_squared=1.0,
            minimum_eigenvalue=1.0,
            maximum_eigenvalue=2.0,
        )
        gate["polynomial_consistency"].update(
            residual_l2_squared=0.0,
            load_l2_squared=1.0,
            tangential_trace_l2_squared=1.0,
        )
        gate["content_sha256"] = checker.record_content_sha256(gate)
        cases.append(nitsche)

        scope = copy.deepcopy(self.valid)
        scope["scope"] = {"not_established": "nothing"}
        scope["proof_debts"] = ["done", "done", "done"]
        cases.append(scope)

        negative = copy.deepcopy(self.valid)
        for record in negative["stokes_refinement"]:
            record["divergence_l2_squared"] = -1.0
            record["linear_residual_l2_squared"] = -1.0
            record["normal_trace_max"] = -1.0
            record["content_sha256"] = checker.record_content_sha256(record)
        cases.append(negative)

        missing = copy.deepcopy(self.valid)
        missing["random_projected_states"][0].pop("convection_projected")
        cases.append(missing)

        digest = copy.deepcopy(self.valid)
        digest["random_projected_states"][0]["input_sha256"] = "0" * 64
        cases.append(digest)

        for forged in cases:
            with self.assertRaises(ValueError):
                verify_certificate(forged)

    def test_optimized_python_rejects_every_self_consistent_forgery(self) -> None:
        cases = _all_self_consistent_forgery_cases(self.valid)

        with tempfile.TemporaryDirectory() as directory:
            for index, forged in enumerate(cases):
                path = Path(directory) / f"broken-{index}.json"
                path.write_text(json.dumps(forged, allow_nan=False))
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-O",
                        str(HERE / "check_bmf_hdiv_ns2d.py"),
                        "--verify",
                        str(path),
                    ],
                    cwd=HERE.parent,
                    text=True,
                    capture_output=True,
                    env={
                        **os.environ,
                        "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    },
                )
                self.assertNotEqual(completed.returncode, 0)

    def test_optimized_python_rejects_a_forged_certificate(self) -> None:
        broken = copy.deepcopy(self.valid)
        broken["formulation"]["pressure_space"] = "equal order"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text(json.dumps(broken, allow_nan=False))
            completed = subprocess.run(
                [
                    sys.executable,
                    "-O",
                    str(HERE / "check_bmf_hdiv_ns2d.py"),
                    "--verify",
                    str(path),
                ],
                cwd=HERE.parent,
                text=True,
                capture_output=True,
                env={
                    **os.environ,
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("metadata", completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
