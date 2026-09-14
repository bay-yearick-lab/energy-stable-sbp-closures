#!/usr/bin/env python3
"""Regression tests for exact universal wall-transfer certificates."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

import boundary_spline_analysis as ba
import universal_wall_transfer as uwt


class UniversalWallTransfer(unittest.TestCase):
    """Exact certificates and float reconstructions for both routes."""

    def test_exact_assembly_matches_quadrature(self) -> None:
        """Match exact unit-cell matrices to the established quadrature assembly."""
        for degree, npts in ((2, 17), (6, 33), (12, 57)):
            cells = npts - degree
            exact = uwt.exact_coefficient_matrices(npts, degree)
            mass, advection, _weak, stiffness = (
                ba.bspline_mass_advection_second_and_stiffness(npts, degree)
            )
            exact_mass = uwt._float_exact(exact["mass"], 1.0 / cells)
            exact_advection = uwt._float_exact(exact["advection"])
            exact_stiffness = uwt._float_exact(exact["stiffness"], cells)
            assert np.linalg.norm(exact_mass - mass) < 2.0e-13
            assert np.linalg.norm(exact_advection - advection) < 2.0e-12
            assert np.linalg.norm(exact_stiffness - stiffness) < 2.0e-10


    def test_rank_formulas_and_proof_certificates(self) -> None:
        """Verify every exact formula and nonzero minor on the declared ladder."""
        artifact = json.loads(uwt.STRUCTURE_PATH.read_text())
        assert len(artifact["records"]) == 33
        for record in artifact["records"]:
            degree = record["degree"]
            corrections = record["coefficient_corrections"]
            for name in ("mass", "advection", "stiffness"):
                assert corrections[name]["exact_rank"] == 4 * degree - 2
            assert corrections["incidence"]["exact_rank"] == 2 * degree - 2
            assert (
                record["bmg_acoustic_pencil"]["exact_common_left_rank"]
                == 8 * degree - 6
            )
            assert (
                record["bmf_mixed_acoustic_pencil"]["exact_common_left_rank"]
                == 8 * degree - 8
            )
        assert len(artifact["proof_certificates"]) == 11
        for certificate in artifact["proof_certificates"]:
            degree = certificate["degree"]
            for determinant in certificate["operator_left_minor_determinants"].values():
                assert determinant["nonzero"]
                assert determinant["sha256"]
            assert certificate["incidence"]["rank"] == 2 * degree - 2
            assert certificate["bmg_acoustic_pencil"]["rank"] == 8 * degree - 6
            assert certificate["bmf_mixed_acoustic_pencil"]["rank"] == 8 * degree - 8
            assert (
                certificate["bmg_acoustic_pencil"]["explicit_operator"][
                    "explicit_correction_rank"
                ]
                == 8 * degree - 6
            )
            assert (
                certificate["bmf_mixed_acoustic_pencil"]["explicit_operator"][
                    "explicit_correction_rank"
                ]
                == 6 * degree - 6
            )


    def test_resolution_independence(self) -> None:
        """Require identical exact ranks across all three resolutions at fixed degree."""
        artifact = json.loads(uwt.STRUCTURE_PATH.read_text())
        for degree in uwt.DEGREES:
            records = [row for row in artifact["records"] if row["degree"] == degree]
            assert {row["npts"] for row in records} == {
                4 * degree + 9,
                6 * degree + 9,
                8 * degree + 9,
            }
            for name in ("mass", "advection", "stiffness", "incidence"):
                assert len(
                    {
                        row["coefficient_corrections"][name]["exact_rank"]
                        for row in records
                    }
                ) == 1


    def test_boundary_channel_resolvent_certificate(self) -> None:
        """Verify the two-route Woodbury certificate and minimal dimensions."""
        artifact = json.loads(uwt.RESOLVENT_PATH.read_text())
        assert artifact["decision"]["accepted"]
        assert (
            artifact["decision"]["maximum_resolvent_relative_residual"]
            <= uwt.STRUCTURAL_TOLERANCE
        )
        assert len(artifact["records"]) == 24
        for record in artifact["records"]:
            degree = record["degree"]
            expected = 8 * degree - 6 if record["route"] == "BMG" else 8 * degree - 8
            assert record["boundary_channel_dimension"] == expected
            assert record["expected_minimal_common_pencil_dimension"] == expected
            assert record["pencil_identity_residual"] <= uwt.STRUCTURAL_TOLERANCE
            assert record["metric_skew_residual"] <= uwt.STRUCTURAL_TOLERANCE
            for point in record["points"]:
                assert (
                    point["energy_scaled_resolvent_relative_residual"]
                    <= uwt.STRUCTURAL_TOLERANCE
                )


    def test_two_route_semigroup_factorizations(self) -> None:
        """Reconstruct both generators with their smallest exact transfer sizes."""
        for builder, expected in (
            (uwt.bmg_float_system, lambda degree: 8 * degree - 6),
            (uwt.bmf_float_system, lambda degree: 6 * degree - 6),
        ):
            for degree, npts in ((2, 25), (6, 41), (12, 65)):
                system = builder(degree, npts)
                residual = np.linalg.norm(
                    system["matrix"] - system["reconstructed"]
                ) / np.linalg.norm(system["matrix"])
                assert system["semigroup_transfer_dimension"] == expected(degree)
                assert residual <= uwt.STRUCTURAL_TOLERANCE
                construction = uwt.data_states(system, heldout=False)
                heldout = uwt.data_states(system, heldout=True)
                successor = uwt.data_states(system, heldout=True, successor=True)
                assert set(construction) == set(uwt.DATA_FAMILIES) | {
                    "phase_adversarial_pair"
                }
                assert set(heldout) == set(uwt.HELDOUT_DATA_FAMILIES)
                assert set(successor) == set(uwt.SUCCESSOR_HELDOUT_DATA_FAMILIES)
                assert {
                    value.tobytes() for value in heldout.values()
                }.isdisjoint({value.tobytes() for value in successor.values()})


    def test_tensor_face_line_certificate(self) -> None:
        """Verify degree-sized normal channels under Cartesian face lifting."""
        artifact = json.loads(uwt.TENSOR_PATH.read_text())
        assert artifact["accepted"]
        assert artifact["maximum_relative_residual"] <= uwt.STRUCTURAL_TOLERANCE
        assert len(artifact["records"]) == 12
        for record in artifact["records"]:
            degree = record["degree"]
            line_rank = (
                8 * degree - 6 if record["route"] == "BMG" else 6 * degree - 6
            )
            assert record["normal_line_channel_dimension"] == line_rank
            assert record["lifted_channel_dimension"] == (
                record["tangential_lines"] * line_rank
            )
            assert record["certified_exact_lifted_rank"] == (
                record["tangential_lines"] * line_rank
            )
            assert record["double_precision_effective_rank"] <= (
                record["certified_exact_lifted_rank"]
            )


if __name__ == "__main__":
    unittest.main()
