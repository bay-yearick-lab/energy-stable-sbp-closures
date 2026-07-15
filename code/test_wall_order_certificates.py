from __future__ import annotations

import unittest

import numpy as np

import boundary_spline_analysis as ba
import wall_order_certificates as wc


class WallOrderCertificateTests(unittest.TestCase):
    def test_w1_actual_operator_identities_and_bmf_endpoint_rows(self) -> None:
        rng = np.random.default_rng(20260714)
        for degree in (4, 6, 8):
            for npts in (degree + 9, degree + 13, degree + 17):
                u = rng.standard_normal(npts)

                bmf = wc.actual_derivative_map("BMF", degree, npts)
                bmf_op = ba.build_operator("B", degree, npts)
                direct_bmf = bmf_op.field_and_derivative(bmf.output_nodes, u)[1]
                bmf_tol = wc.backward_error_tolerance(
                    bmf.matrix, u, direct_bmf, precision_bits=53
                )
                self.assertLessEqual(
                    np.max(np.abs(bmf.matrix @ u - direct_bmf)), bmf_tol
                )
                np.testing.assert_allclose(
                    bmf.matrix[[0, -1]],
                    bmf_op.matrix[[0, -1]],
                    rtol=0.0,
                    atol=50.0 * np.finfo(float).eps * np.linalg.norm(bmf_op.matrix, np.inf),
                )

                bmg = wc.actual_derivative_map("BMG", degree, npts)
                bmg_op = ba.build_operator("I", degree, npts)
                direct_bmg = bmg_op.field_and_derivative(bmg.output_nodes, u)[1]
                bmg_tol = wc.backward_error_tolerance(
                    bmg.matrix, u, direct_bmg, precision_bits=53
                )
                self.assertLessEqual(
                    np.max(np.abs(bmg.matrix @ u - direct_bmg)), bmg_tol
                )

    def test_w1_identity_certificate_records_residuals_and_tolerances(self) -> None:
        for family in ("BMF", "BMG"):
            certificate = wc.operator_identity_certificate(family, 8, 25)
            self.assertTrue(certificate["identity_passes"])
            self.assertEqual(certificate["comparison"], "full operator matrix")
            self.assertEqual(certificate["coordinate_vectors_compared"], 25)
            self.assertIn("128 eps", certificate["tolerance_rationale"])
            self.assertLessEqual(
                certificate["identity_residual_inf"], certificate["identity_tolerance"]
            )
            self.assertEqual(certificate["precision"]["bits"], 53)
            if family == "BMF":
                self.assertTrue(certificate["endpoint_rows_match_same_point"])
                self.assertLessEqual(
                    certificate["endpoint_row_residual_inf"],
                    certificate["endpoint_row_tolerance"],
                )

        assembled = np.eye(5)
        factorized = assembled.copy()
        self.assertTrue(wc.matrix_identity_metrics(assembled, factorized)["passes"])
        factorized[2, 3] += 1.0e-6
        mutated = wc.matrix_identity_metrics(assembled, factorized)
        self.assertFalse(mutated["passes"])
        self.assertGreater(mutated["residual_inf"], mutated["tolerance"])

    def test_w2_scaled_moments_use_knot_width_and_condition_aware_tolerance(self) -> None:
        for family in ("BMF", "BMG"):
            for degree in (4, 6, 8):
                for npts in (degree + 9, degree + 13, degree + 17):
                    certificate = wc.floating_moment_certificate(family, degree, npts)
                    self.assertEqual(certificate["h_definition"], "1/(N-p)")
                    self.assertEqual(certificate["output_grid"],
                                     "derivative Greville" if family == "BMF" else "primal Greville")
                    self.assertEqual(certificate["precision"]["bits"], 53)
                    self.assertIn("backward", certificate["tolerance_rationale"].lower())
                    for row in certificate["rows"]:
                        for moment in row["moments"][:degree + 1]:
                            self.assertTrue(moment["within_backward_tolerance"])
                            self.assertTrue(moment["structural_zero"])
                            self.assertIn("raw_moment", moment)
                            self.assertIn("scaled_moment", moment)
                            self.assertIn("raw_tolerance", moment)

    def test_w6_actual_grid_convergence_and_w7_roundoff_guard(self) -> None:
        ns_map = {
            4: (17, 25, 33, 49, 65),
            6: (17, 25, 33, 49, 65),
            8: (17, 25, 33, 49, 65),
        }
        for family in ("BMF", "BMG"):
            for degree, npts in ns_map.items():
                result = wc.actual_grid_convergence(family, degree, npts)
                self.assertEqual(result["output_grid"],
                                 "derivative Greville" if family == "BMF" else "primal Greville")
                self.assertGreaterEqual(result["fit_point_count"], 3)
                self.assertTrue(result["roundoff_floor_guard_applied"])
                self.assertIn("32 eps", result["tolerance_rationale"])
                self.assertNotIn("tail_order", result)
                self.assertGreater(result["pre_roundoff_fit_order"], degree - 0.75)
                self.assertLess(result["pre_roundoff_fit_order"], degree + 1.25)
                self.assertEqual(result["fit_N_count"], result["fit_point_count"])
                self.assertEqual(result["fit_N_min"], min(result["fit_N"]))
                self.assertEqual(result["fit_N_max"], max(result["fit_N"]))
                self.assertTrue(all(
                    error > floor
                    for error, floor, used in zip(
                        result["errors_inf"], result["roundoff_floors"], result["fit_mask"]
                    )
                    if used
                ))

        fit = wc.fit_pre_roundoff_orders(
            hs=np.array([1 / 8, 1 / 16, 1 / 32, 1 / 64], dtype=float),
            errors=np.array([2.0**-8, 2.0**-12, 2.0**-16, 1.0e-16]),
            floors=np.array([1.0e-18, 1.0e-18, 1.0e-18, 2.0e-16]),
        )
        self.assertEqual(fit["fit_mask"], [True, True, True, False])
        self.assertAlmostEqual(fit["pre_roundoff_fit_order"], 4.0, places=12)

    def test_w8_schema_marks_p8_first_free_wall_moment(self) -> None:
        for family in ("BMF", "BMG"):
            certificate = wc.multiprecision_endpoint_certificate(
                family, degree=8, npts=17, decimal_digits=90
            )
            self.assertGreaterEqual(certificate["precision"]["decimal_digits"], 80)
            self.assertEqual(certificate["h_definition"], "1/(N-p)")
            self.assertEqual(certificate["exact_polynomial_degree"], 8)
            self.assertEqual(certificate["first_nonzero_moment_degree"], 9)
            for side in ("left", "right"):
                interval = certificate["endpoints"][side]["scaled_moment_interval"]
                self.assertTrue(interval[1] < 0.0 or interval[0] > 0.0)
                self.assertTrue(certificate["endpoints"][side]["exact_rational_nonzero"])
                self.assertEqual(
                    certificate["endpoints"][side]["interval_excludes_zero"],
                    interval[1] < 0.0 or interval[0] > 0.0,
                )
                self.assertEqual(certificate["endpoints"][side]["row"],
                                 0 if side == "left" else certificate["output_size"] - 1)

    def test_w3_finite_grid_exact_endpoint_sequence_never_crosses_zero(self) -> None:
        for family in ("BMF", "BMG"):
            for degree in (4, 6, 8):
                sequence = wc.multiprecision_endpoint_sequence(
                    family, degree, (degree + 9, degree + 17, degree + 25),
                    decimal_digits=90,
                )
                self.assertEqual(sequence["certificate_count"], 3)
                self.assertTrue(sequence["all_intervals_exclude_zero"])
                self.assertFalse(sequence["proves_all_N_limit"])
                self.assertTrue(all(
                    item["endpoints"]["left"]["interval_excludes_zero"]
                    and item["endpoints"]["right"]["interval_excludes_zero"]
                    for item in sequence["certificates"]
                ))

    def test_formal_order_and_closure_drivers_consume_corrected_schema(self) -> None:
        import check_formal_order as formal
        import s6_closure_moment_ladder as ladder

        formal_summary = formal.build_formal_order_summary(
            degrees=(8,), convergence_degrees=(8,),
            ns_map={8: (17, 25, 33, 49, 65)},
            include_exact_endpoints=True,
        )
        for family in ("BMF", "BMG"):
            row = formal_summary["polynomial_exactness"][f"{family}8"]
            self.assertEqual(row["exact_polynomial_degree"], 8)
            self.assertEqual(row["first_nonzero_moment_degree"], 9)
            self.assertEqual(row["output_grid"],
                             "derivative Greville" if family == "BMF" else "primal Greville")
            self.assertTrue(row["finite_grid_exact_endpoint_certificate"])
            self.assertTrue(row["all_rows_exact_through_p_within_backward_tolerance"])
            self.assertNotIn("all_rows_exact_through_p_within_tolerance", row)

        row = ladder.closure_moments("BMF", 8, 17, 9)
        self.assertEqual(row["h_definition"], "1/(N-p)")
        self.assertEqual(row["output_grid"], "derivative Greville")
        self.assertEqual(row["exact_polynomial_degree"], 8)
        self.assertEqual(row["first_nonzero_moment_degree"], 9)
        self.assertIn("scaled_moment", row["moments"][8])
        self.assertFalse(row["moments"][8]["structural_zero"])
        self.assertFalse(row["moments"][8]["within_backward_tolerance"])
        self.assertEqual(row["wall_footprint"], 17)
        self.assertEqual(row["wall_footprint_definition"], "structural eliminated-row width")
        self.assertNotIn("finite_grid_exact_endpoint_certificate", row)
        self.assertEqual(row["reference_exact_endpoint_N"], 17)
        self.assertIn("reference_exact_endpoint_certificate", row)
        self.assertEqual(row["floating_first_free_moment_at_N"]["N"], 17)

        csv_text = formal.render_formal_order_csv(formal_summary)
        self.assertNotIn("\r", csv_text)
        self.assertIn("pre_roundoff_fit_order", csv_text.splitlines()[0])
        self.assertNotIn("tail_order", csv_text.splitlines()[0])
        self.assertTrue(csv_text.endswith("\n"))


if __name__ == "__main__":
    unittest.main()
