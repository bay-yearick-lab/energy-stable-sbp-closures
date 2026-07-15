from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bmf_hdiv_projection import BMFHDivProjection2D
from check_bmf_hdiv_projection import (
    assemble_independent,
    build_certificate,
    compare_independent_projection,
    verify_certificate,
)


def _sample_velocity(core: BMFHDivProjection2D, seed: int = 9) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ux = rng.standard_normal(core.ux_shape)
    uz = rng.standard_normal(core.uz_shape)
    ux[[0, -1], :] = 0.0
    uz[:, [0, -1]] = 0.0
    return ux, uz


class BMFHDivProjectionTests(unittest.TestCase):
    def test_spaces_and_exact_divergence_have_the_declared_shapes(self) -> None:
        core = BMFHDivProjection2D(npts=11, degree=4)
        self.assertEqual(core.ux_shape, (11, 10))
        self.assertEqual(core.uz_shape, (10, 11))
        self.assertEqual(core.pressure_shape, (10, 10))
        ux, uz = _sample_velocity(core)
        expected = core.d10 @ ux + uz @ core.d10.T
        np.testing.assert_allclose(core.divergence(ux, uz), expected, rtol=0.0, atol=0.0)

    def test_adjoint_gradient_satisfies_the_natural_metric_identity(self) -> None:
        core = BMFHDivProjection2D(npts=10, degree=3)
        ux, uz = _sample_velocity(core, seed=2)
        rng = np.random.default_rng(3)
        pressure = rng.standard_normal(core.pressure_shape)
        gx, gz = core.adjoint_gradient(pressure)
        lhs = core.velocity_inner((ux, uz), (gx, gz))
        rhs = core.pressure_inner(core.divergence(ux, uz), pressure)
        self.assertLess(abs(lhs - rhs), 2.0e-12 * max(1.0, abs(rhs)))

    def test_tensor_and_dense_quotient_solves_agree_modulo_constants(self) -> None:
        core = BMFHDivProjection2D(npts=10, degree=4)
        ux, uz = _sample_velocity(core, seed=5)
        rhs = core.divergence(ux, uz)
        pt = core.solve_pressure(rhs, method="tensor")
        pd = core.solve_pressure(rhs, method="dense")
        difference = core.remove_pressure_mean(pt - pd)
        rel = core.pressure_norm(difference) / max(core.pressure_norm(pd), 1.0e-30)
        self.assertLess(rel, 1.0e-12)

    def test_projection_annihilates_divergence_and_is_metric_idempotent(self) -> None:
        core = BMFHDivProjection2D(npts=11, degree=4)
        ux, uz = _sample_velocity(core, seed=7)
        first = core.project(ux, uz, method="tensor")
        second = core.project(first.ux, first.uz, method="tensor")
        initial_div = core.pressure_norm(core.divergence(ux, uz))
        final_div = core.pressure_norm(core.divergence(first.ux, first.uz))
        delta = core.velocity_norm((second.ux - first.ux, second.uz - first.uz))
        scale = core.velocity_norm((first.ux, first.uz))
        self.assertLess(final_div / initial_div, 1.0e-12)
        self.assertLess(delta / max(scale, 1.0e-30), 1.0e-12)
        self.assertLess(core.normal_trace_max(first.ux, first.uz), 1.0e-13)

    def test_pressure_schur_has_only_the_physical_constant_null(self) -> None:
        core = BMFHDivProjection2D(npts=11, degree=4)
        certificate = core.kernel_certificate()
        self.assertEqual(certificate.nullity, 1)
        self.assertGreaterEqual(certificate.constant_overlap, 1.0 - 1.0e-12)

    def test_divergence_range_is_the_mean_zero_pressure_space(self) -> None:
        for degree in (2, 4, 6, 8):
            core = BMFHDivProjection2D(npts=degree + 5, degree=degree)
            self.assertEqual(core.divergence_rank(), core.pressure_size - 1)

    def test_beta_tracks_the_neumann_constant_pi(self) -> None:
        degree_two_errors = [
            abs(BMFHDivProjection2D(npts=npts, degree=2).beta() / np.pi - 1.0)
            for npts in (7, 11, 19)
        ]
        self.assertGreater(degree_two_errors[0], degree_two_errors[1])
        self.assertGreater(degree_two_errors[1], degree_two_errors[2])
        self.assertLess(degree_two_errors[-1], 1.0e-6)
        for degree in (4, 6, 8):
            core = BMFHDivProjection2D(npts=degree + 9, degree=degree)
            self.assertLess(abs(core.beta() / np.pi - 1.0), 1.0e-8)

    def test_projection_rejects_nonzero_normal_trace(self) -> None:
        core = BMFHDivProjection2D(npts=9, degree=3)
        ux, uz = _sample_velocity(core)
        ux[0, 2] = 1.0e-4
        with self.assertRaisesRegex(ValueError, "normal trace"):
            core.project(ux, uz)

    def test_pressure_solve_rejects_an_incompatible_rhs(self) -> None:
        core = BMFHDivProjection2D(npts=9, degree=3)
        with self.assertRaisesRegex(ValueError, "mean-zero"):
            core.solve_pressure(np.ones(core.pressure_shape))

    def test_nonfinite_constructor_and_input_data_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            BMFHDivProjection2D(npts=9, degree=3, length=np.nan)
        core = BMFHDivProjection2D(npts=9, degree=3)
        ux, uz = _sample_velocity(core)
        ux[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            core.project(ux, uz)
        pressure = np.zeros(core.pressure_shape)
        pressure[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            core.solve_pressure(pressure)

    def test_nonfinite_internal_metric_fails_with_a_named_error(self) -> None:
        import bmf_hdiv_projection as module

        valid = module.bmf_staggered_operators(9, 3, 1.0)
        broken = SimpleNamespace(**vars(valid))
        broken.h1 = np.asarray(valid.h1).copy()
        broken.h1[0, 0] = np.nan
        with patch.object(module, "bmf_staggered_operators", return_value=broken):
            with self.assertRaisesRegex(ValueError, "finite.*h1"):
                BMFHDivProjection2D(npts=9, degree=3)

    def test_constructor_rejects_a_nonconstant_endpoint_cokernel(self) -> None:
        import bmf_hdiv_projection as module

        valid = module.bmf_staggered_operators(9, 3, 1.0)
        broken = SimpleNamespace(**vars(valid))
        broken.d10 = np.asarray(valid.d10).copy()
        broken.h0 = np.asarray(valid.h0).copy()
        broken.h1 = np.asarray(valid.h1).copy()
        broken.d10[0, 1] += 10.0
        with patch.object(module, "bmf_staggered_operators", return_value=broken):
            with self.assertRaisesRegex(ValueError, "physical constant"):
                BMFHDivProjection2D(npts=9, degree=3)

    def test_constructor_rejects_a_rank_deficient_endpoint_derivative(self) -> None:
        import bmf_hdiv_projection as module

        valid = module.bmf_staggered_operators(9, 3, 1.0)
        broken = SimpleNamespace(**vars(valid))
        broken.d10 = np.asarray(valid.d10).copy()
        broken.h0 = np.asarray(valid.h0).copy()
        broken.h1 = np.asarray(valid.h1).copy()
        broken.d10[:, 1] = 0.0
        with patch.object(module, "bmf_staggered_operators", return_value=broken):
            with self.assertRaisesRegex(ValueError, "rank"):
                BMFHDivProjection2D(npts=9, degree=3)

    def test_constructor_validates_the_raw_null_and_positive_gap(self) -> None:
        import bmf_hdiv_projection as module

        original_eigh = module.eigh

        def missing_null(*args, **kwargs):
            values, vectors = original_eigh(*args, **kwargs)
            values = np.asarray(values).copy()
            values[0] = 0.5 * values[1]
            return values, vectors

        with patch.object(module, "eigh", side_effect=missing_null):
            with self.assertRaisesRegex(ValueError, "raw null"):
                BMFHDivProjection2D(npts=9, degree=3)

        def missing_gap(*args, **kwargs):
            values, vectors = original_eigh(*args, **kwargs)
            values = np.asarray(values).copy()
            values[1] = 0.0
            return values, vectors

        with patch.object(module, "eigh", side_effect=missing_gap):
            with self.assertRaisesRegex(ValueError, "positive spectral gap"):
                BMFHDivProjection2D(npts=9, degree=3)


class IndependentAssemblyTests(unittest.TestCase):
    def test_direct_coefficient_assembly_matches_rank_kernel_and_beta(self) -> None:
        for degree, npts in ((2, 10), (4, 12), (6, 14), (8, 16)):
            direct = assemble_independent(npts=npts, degree=degree)
            production = BMFHDivProjection2D(npts=npts, degree=degree)
            self.assertEqual(direct.divergence_rank(), direct.pressure_size - 1)
            kernel = direct.kernel_certificate()
            self.assertEqual(kernel["nullity"], 1)
            self.assertGreaterEqual(kernel["constant_overlap"], 1.0 - 1.0e-12)
            self.assertLess(abs(direct.beta() - production.beta()), 2.0e-11)

    def test_direct_and_production_projections_agree(self) -> None:
        result = compare_independent_projection(npts=10, degree=4, seed=17)
        self.assertLess(result["velocity_relative_difference"], 1.0e-12)
        self.assertLess(result["pressure_mod_constant_relative_difference"], 1.0e-11)
        self.assertLess(result["direct_postprojection_relative_divergence"], 1.0e-12)
        self.assertLess(result["production_postprojection_relative_divergence"], 1.0e-12)

    def test_verifier_rejects_a_corrupted_projection_gate(self) -> None:
        valid = build_certificate()
        verify_certificate(valid)
        broken = copy.deepcopy(valid)
        lane = broken["projection_adversaries"][0]["production"]
        lane["post_divergence_metric_squared"] = (
            lane["initial_divergence_metric_squared"] * 1.0e-10
        )
        lane["postprojection_relative_divergence"] = 1.0e-5
        with self.assertRaisesRegex(ValueError, "divergence"):
            verify_certificate(broken)

    def test_independent_projection_fails_closed_on_nonfinite_data(self) -> None:
        direct = assemble_independent(npts=9, degree=4)
        velocity = np.zeros(direct.velocity_metric.shape[0])
        velocity[0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite.*velocity"):
            direct.project_free_vector(velocity)

        direct = assemble_independent(npts=9, degree=4)
        direct.divergence_matrix[0, 0] = np.inf
        velocity = np.zeros(direct.velocity_metric.shape[0])
        with self.assertRaisesRegex(ValueError, "finite.*divergence"):
            direct.project_free_vector(velocity)


class CertificateIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.valid = build_certificate()

    def test_certificate_has_exact_metadata_and_dual_path_adversaries(self) -> None:
        self.assertEqual(self.valid["schema_version"], 2)
        formulation = self.valid["formulation"]
        if "domain" not in formulation:
            self.fail("exact domain metadata is missing")
        self.assertEqual(formulation["domain"], "unit square [0,1]^2")
        self.assertIn("natural L2", formulation["velocity_metric"])
        self.assertIn("natural L2", formulation["pressure_metric"])
        self.assertEqual(
            formulation["continuum_comparator"],
            "Neumann pressure Laplacian on the unit square modulo constants; beta=pi",
        )
        self.assertEqual(
            {record["name"] for record in self.valid["projection_adversaries"]},
            {"smooth", "checkerboard", "wall_localized", "old_nu"},
        )
        for record in self.valid["projection_adversaries"]:
            self.assertRegex(record["input_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(record["content_sha256"], r"^[0-9a-f]{64}$")
            self.assertIn("production", record)
            self.assertIn("independent", record)
            for path in ("production", "independent"):
                lane = record[path]
                self.assertIn("post_divergence_metric_squared", lane)
                self.assertIn("initial_divergence_metric_squared", lane)
                self.assertIn("post_divergence_euclidean_squared", lane)
                self.assertIn("initial_divergence_euclidean_squared", lane)

    def test_finite_certificate_points_to_completed_analytic_proofs(self) -> None:
        obligations = self.valid["proof_obligations_beyond_finite_certificate"]
        joined = " ".join(obligations.values()).lower()
        self.assertNotIn("remains required", joined)
        self.assertNotIn("must still be written", joined)
        self.assertIn("article proposition", obligations["C1"].lower())
        self.assertIn("article theorem", obligations["C4"].lower())
        self.assertIn("separate integration certificate", obligations["C3"].lower())

    def test_verifier_rejects_nan_missing_grid_and_forged_metadata(self) -> None:
        nan_case = copy.deepcopy(self.valid)
        nan_case["grid_cells"][0]["beta"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            verify_certificate(nan_case)

        missing_grid = copy.deepcopy(self.valid)
        missing_grid["grid_cells"].pop()
        with self.assertRaisesRegex(ValueError, "grid"):
            verify_certificate(missing_grid)

        forged_metadata = copy.deepcopy(self.valid)
        forged_metadata["formulation"]["domain"] = "periodic torus"
        with self.assertRaisesRegex(ValueError, "metadata"):
            verify_certificate(forged_metadata)

    def test_verifier_rejects_missing_adversary_and_wrong_derived_ratio(self) -> None:
        missing = copy.deepcopy(self.valid)
        missing["projection_adversaries"].pop()
        with self.assertRaisesRegex(ValueError, "adversar"):
            verify_certificate(missing)

        wrong_ratio = copy.deepcopy(self.valid)
        wrong_ratio["projection_adversaries"][0]["production"][
            "postprojection_relative_divergence"
        ] *= 0.5
        with self.assertRaisesRegex(ValueError, "derived ratio"):
            verify_certificate(wrong_ratio)

    def test_verifier_rejects_euclidean_norm_forgery(self) -> None:
        forged = copy.deepcopy(self.valid)
        if "production" not in forged["projection_adversaries"][0]:
            self.fail("production raw-metric lane is missing")
        lane = forged["projection_adversaries"][0]["production"]
        lane["post_divergence_metric_squared"] = lane[
            "post_divergence_euclidean_squared"
        ]
        lane["initial_divergence_metric_squared"] = lane[
            "initial_divergence_euclidean_squared"
        ]
        lane["postprojection_relative_divergence"] = lane[
            "postprojection_euclidean_relative_divergence"
        ]
        lane["metric_vs_euclidean_initial_divergence_ratio"] = 1.0
        with self.assertRaisesRegex(ValueError, "natural metric"):
            verify_certificate(forged)

    def test_deterministic_rebuild_rejects_scaled_euclidean_forgery(self) -> None:
        import check_bmf_hdiv_projection as module

        forged = copy.deepcopy(self.valid)
        record = forged["projection_adversaries"][0]
        lane = record["production"]
        factor = 2.0
        lane["post_divergence_metric_squared"] = (
            factor * lane["post_divergence_euclidean_squared"]
        )
        lane["initial_divergence_metric_squared"] = (
            factor * lane["initial_divergence_euclidean_squared"]
        )
        lane["postprojection_relative_divergence"] = lane[
            "postprojection_euclidean_relative_divergence"
        ]
        lane["idempotence_metric_squared"] = (
            factor * lane["idempotence_euclidean_squared"]
        )
        lane["input_velocity_metric_squared"] = (
            factor * lane["input_velocity_euclidean_squared"]
        )
        lane["idempotence_relative"] = lane["idempotence_euclidean_relative"]
        lane["metric_vs_euclidean_initial_divergence_ratio"] = factor
        lane["metric_vs_euclidean_input_velocity_ratio"] = factor
        if hasattr(module, "adversary_content_sha256"):
            record["content_sha256"] = module.adversary_content_sha256(record)
        with self.assertRaisesRegex(ValueError, "deterministic rebuild"):
            verify_certificate(forged)

    def test_deterministic_rebuild_binds_each_name_to_its_adversary_content(self) -> None:
        import check_bmf_hdiv_projection as module

        forged = copy.deepcopy(self.valid)
        target = forged["projection_adversaries"][0]
        source = forged["projection_adversaries"][1]
        target["production"] = copy.deepcopy(source["production"])
        target["independent"] = copy.deepcopy(source["independent"])
        if hasattr(module, "adversary_content_sha256"):
            target["content_sha256"] = module.adversary_content_sha256(target)
        with self.assertRaisesRegex(ValueError, "deterministic rebuild"):
            verify_certificate(forged)

    def test_verifier_rejects_a_forged_canonical_adversary_digest(self) -> None:
        forged = copy.deepcopy(self.valid)
        forged["projection_adversaries"][0]["input_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "canonical input digest"):
            verify_certificate(forged)

    def test_optimized_python_still_rejects_a_missing_grid(self) -> None:
        broken = copy.deepcopy(self.valid)
        broken["grid_cells"].pop()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text(json.dumps(broken, allow_nan=False))
            completed = subprocess.run(
                [
                    sys.executable,
                    "-O",
                    str(HERE / "check_bmf_hdiv_projection.py"),
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
        self.assertIn("grid", completed.stderr + completed.stdout)

    def test_json_read_and_write_reject_nonfinite_constants(self) -> None:
        import check_bmf_hdiv_projection as module

        self.assertTrue(
            hasattr(module, "write_json_strict"),
            "strict JSON writer API is missing",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nan.json"
            path.write_text('{"beta": NaN}')
            with self.assertRaisesRegex(ValueError, "non-finite JSON"):
                module.load_json_strict(path)
            broken = copy.deepcopy(self.valid)
            broken["grid_cells"][0]["beta"] = float("nan")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                module.write_json_strict(path, broken)


if __name__ == "__main__":
    unittest.main()
