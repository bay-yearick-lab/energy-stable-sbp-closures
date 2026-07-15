from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import boundary_spline_analysis as bsa
import heat_cavity_core as hcc
import scattering_core as sc


class BoundarySplineAnalysisTests(unittest.TestCase):
    def test_bmf_staggered_sbp_identity_for_representative_degrees(self) -> None:
        for npts, degree in ((17, 2), (33, 4), (33, 6), (49, 10)):
            ops = bsa.bmf_staggered_operators(npts, degree)
            resid = np.linalg.norm(ops.h0 @ ops.d01 + ops.d10.T @ ops.h1 - ops.boundary, 2)
            self.assertEqual(ops.d10.shape, (npts - 1, npts))
            self.assertEqual(ops.d01.shape, (npts, npts - 1))
            self.assertLess(resid, 1.0e-8)
            self.assertGreater(np.linalg.eigvalsh(ops.h0).min(), 1.0e-12)
            self.assertGreater(np.linalg.eigvalsh(ops.h1).min(), 1.0e-12)

    def test_bmf_natural_metric_is_not_same_grid_sbp_norm(self) -> None:
        npts, degree = 33, 6
        dmat = bsa.bmf_nodal_operator(npts, degree)[1]
        h0 = bsa.bmf_full_metric(npts, degree)[1]
        resid = np.linalg.norm(h0 @ dmat + dmat.T @ h0 - bsa.endpoint_flux_matrix(npts), 2)
        self.assertGreater(resid, 1.0e-2)

    def test_bmf_staggered_derivative_reconstructs_same_point_map(self) -> None:
        npts, degree = 33, 6
        dpt = bsa.bmf_nodal_operator(npts, degree)[1]
        drecon = bsa.bmf_reconstructed_same_grid_derivative_matrix(npts, degree)[1]
        self.assertLess(np.linalg.norm(dpt - drecon, ord=np.inf), 1.0e-11)

    def test_bmg_metric_satisfies_weighted_sbp_identity(self) -> None:
        npts, degree = 33, 6
        op = bsa.BoundaryBMG(npts, degree)
        resid = np.linalg.norm(op.metric @ op.matrix + op.matrix.T @ op.metric - bsa.endpoint_flux_matrix(npts), 2)
        mineig = np.linalg.eigvalsh(op.metric).min()
        self.assertLess(resid, 1.0e-8)
        self.assertGreater(mineig, 1.0e-4)

    def test_bmg_theorem_facing_first_derivative_matches_operator_matrix(self) -> None:
        npts, degree = 33, 6
        _x, dmat, metric = bsa.bmg_sbp_first_derivative_matrix(npts, degree)
        op = bsa.BoundaryBMG(npts, degree)
        self.assertLess(np.linalg.norm(dmat - op.matrix, ord=np.inf), 1.0e-11)
        self.assertLess(np.linalg.norm(metric - op.metric, ord=np.inf), 1.0e-13)

    def test_wall_mode_response_uses_reference_spacing_for_bmf(self) -> None:
        npts, degree = 33, 6
        op = bsa.ClampedBSpline(npts, degree)
        response = bsa.wall_mode_response(op)
        href = bsa.reference_spacing(op.x)
        expected_mode = min(npts - 2, math.floor((1.0 - 1.0e-12) / href))
        self.assertEqual(response[-1].mode, expected_mode)
        self.assertAlmostEqual(2.0 / response[-1].ppw, expected_mode * href, places=12)

    def test_operator3d_uses_operator_grid_for_bmf_and_bmg(self) -> None:
        for label in ("BMF6", "BMG6"):
            op3d = sc.Operator3D(label, 33)
            self.assertTrue(np.allclose(op3d.x, op3d._op1d.x))
            self.assertLess(op3d.h, op3d.h_nominal)

    def test_bmf_curvature_map_reproduces_polynomials(self) -> None:
        op = bsa.ClampedBSpline(17, 4)
        d2 = op.curvature_matrix()
        for k in range(5):
            u = op.x ** k
            exact = np.zeros_like(op.x) if k < 2 else k * (k - 1) * op.x ** (k - 2)
            err = np.linalg.norm(d2 @ u - exact, ord=np.inf)
            self.assertLess(err, 1.0e-10)

    def test_bmf_canonical_first_derivative_preserves_constants_and_polynomials(self) -> None:
        x, d1 = bsa.bmf_sbp_first_derivative_matrix(17, 4)
        self.assertLess(np.linalg.norm(d1 @ np.ones_like(x), ord=np.inf), 1.0e-10)
        for k in range(5):
            u = x ** k
            exact = np.zeros_like(x) if k == 0 else k * x ** (k - 1)
            err = np.linalg.norm(d1 @ u - exact, ord=np.inf)
            self.assertLess(err, 1.0e-10)

    def test_bmf_scheme_second_derivative_is_staggered_divergence_gradient(self) -> None:
        op = bsa.ClampedBSpline(17, 4)
        ops = bsa.bmf_staggered_operators(17, 4)
        d2 = bsa.second_derivative_matrix(op)
        expected = ops.d01 @ ops.d10
        self.assertLess(np.linalg.norm(d2 - expected, ord=np.inf), 1.0e-12)
        resid = np.linalg.norm(ops.h0 @ ops.d01 + ops.d10.T @ ops.h1 - ops.boundary, ord=np.inf)
        self.assertLess(resid, 1.0e-10)
        for k in range(5):
            u = op.x ** k
            exact = np.zeros_like(op.x) if k < 2 else k * (k - 1) * op.x ** (k - 2)
            err = np.linalg.norm(d2 @ u - exact, ord=np.inf)
            self.assertLess(err, 1.0e-10)

    def test_bmg_sbp_second_derivative_matches_compatible_identity(self) -> None:
        op = bsa.BoundaryBMG(17, 4)
        d2 = bsa.second_derivative_matrix(op)
        expected = bsa.bmg_sbp_second_derivative_matrix(17, 4)[1]
        e0 = np.zeros((17, 1), dtype=float)
        eN = np.zeros((17, 1), dtype=float)
        e0[0, 0] = 1.0
        eN[-1, 0] = 1.0
        self.assertLess(np.linalg.norm(d2 - expected, ord=np.inf), 1.0e-12)
        resid = np.linalg.norm(op.metric @ d2 + op.matrix.T @ op.metric @ op.matrix + e0 @ op.matrix[[0], :] - eN @ op.matrix[[-1], :], ord=np.inf)
        self.assertLess(resid, 1.0e-10)
        self.assertLess(np.linalg.norm(d2 @ np.ones_like(op.x), ord=np.inf), 1.0e-10)
        for k in range(5):
            u = op.x ** k
            exact = np.zeros_like(op.x) if k < 2 else k * (k - 1) * op.x ** (k - 2)
            err = np.linalg.norm(d2 @ u - exact, ord=np.inf)
            self.assertLess(err, 1.0e-10)

    def test_only_families_with_second_derivatives_enter_heat_cavity_benchmark(self) -> None:
        self.assertEqual(hcc.heat_cavity_labels(), ['SBP6', 'BMF6', 'BMG6'])

    def test_operator_without_second_derivative_raises(self) -> None:
        bmf = bsa.ClampedBSpline(17, 4)
        bmg = bsa.BoundaryBMG(17, 4)
        sbp = bsa.SBPSAT(17, 4)
        efd = bsa.ExplicitFD(17, 6)
        self.assertEqual(bsa.second_derivative_matrix(bmf).shape, (17, 17))
        self.assertEqual(bsa.second_derivative_matrix(bmg).shape, (17, 17))
        self.assertEqual(bsa.second_derivative_matrix(sbp).shape, (17, 17))
        with self.assertRaises(ValueError):
            bsa.second_derivative_matrix(efd)

    def test_heat_cavity_uses_scheme_second_derivatives(self) -> None:
        modes = hcc.default_heat_modes(m_center=4, n_center=4, half_band=1, viscosity=1.0e-4, seed=1)
        for label in ('SBP4', 'BMF4', 'BMG4'):
            result = hcc.run_heat_cavity(label, modes, npts=17, t_end=0.02, cfl=0.05, record_count=2, snapshot_count=2)
            self.assertEqual(result.scheme, label)
            self.assertTrue(np.all(np.isfinite(result.u_rel_l2)))

    def test_heat_cavity_scores_nonuniform_greville_nodes_with_their_actual_spacing(self) -> None:
        modes = hcc.default_heat_modes(
            m_center=6, n_center=6, half_band=1, viscosity=1.0e-4, seed=2
        )
        result = hcc.run_heat_cavity(
            "BMF4", modes, npts=17, t_end=0.05, dt=0.05,
            record_count=2, snapshot_count=2,
        )
        x = bsa.build_operator(bsa.scheme_kind("BMF4"), 4, 17).x
        w = hcc.node_trapezoid_weights(x)
        error = result.final_u - result.final_u_exact
        manual = hcc._l2_rel(error, result.final_u_exact, np.outer(w, w))

        self.assertAlmostEqual(result.u_rel_l2[-1], manual, places=14)
        self.assertFalse(np.allclose(w, bsa.trapezoidal_weights(17)))

    def test_reflected_acoustics_bmf_uses_staggered_route(self) -> None:
        op = bsa.ClampedBSpline(25, 4)
        pressure_exact, velocity_exact = bsa.rigid_wall_acoustics_packet(
            x0=0.24,
            sigma=0.05,
            k=2.0 * np.pi / (6.5 / 24.0),
        )
        result = bsa.simulate_reflected_acoustics(
            op,
            pressure_exact,
            velocity_exact,
            t_end=0.04,
            cfl=0.02,
            interior_strip=4,
            record_count=3,
        )
        self.assertEqual(result.pressure.shape, op.x.shape)
        self.assertEqual(result.velocity.shape, op.x.shape)
        self.assertTrue(np.all(np.isfinite(result.error_history)))
        self.assertEqual(result.operator_route, "staggered primal-pressure and derivative-grid velocity complex")

    def test_reflected_acoustics_does_not_clamp_pressure_trace(self) -> None:
        def pressure_exact(x, t):
            return np.sin(np.pi * np.asarray(x, dtype=float)) + float(t)

        def velocity_exact(x, _t):
            return np.zeros_like(np.asarray(x, dtype=float), dtype=float)

        for op in (bsa.ExplicitFD(17, 4), bsa.ClampedBSpline(17, 4)):
            result = bsa.simulate_reflected_acoustics(
                op,
                pressure_exact,
                velocity_exact,
                t_end=0.02,
                cfl=0.02,
                interior_strip=3,
                record_count=3,
            )
            exact_endpoint = float(np.asarray(pressure_exact(result.grid[0], 0.02)))
            self.assertGreater(abs(result.pressure[0] - exact_endpoint), 1.0e-4)

    def test_bmf_staggered_3d_acoustic_shapes_and_energy(self) -> None:
        op = sc.BMFStaggeredOperator3D("BMF4", 11)
        mms = sc.BoxEigenmode(modes=[(1, 1, 1, 1.0)], c=1.0, rho0=1.0, length=1.0)
        state = sc.bmf_staggered_exact_state(op, mms, 0.1)
        self.assertEqual(state.p.shape, (11, 11, 11))
        self.assertEqual(state.u.shape, (10, 11, 11))
        self.assertEqual(state.v.shape, (11, 10, 11))
        self.assertEqual(state.w.shape, (11, 11, 10))
        energy0 = sc.bmf_staggered_energy(state, op, rho0=1.0, c=1.0)
        self.assertGreater(energy0, 0.0)
        sc.bmf_staggered_rk4_step(state, 1.0e-4, op, rho0=1.0, c=1.0)
        self.assertTrue(np.all(np.isfinite(state.p)))
        self.assertAlmostEqual(float(np.max(np.abs(state.p[0, :, :]))), 0.0, places=13)


if __name__ == "__main__":
    unittest.main()
