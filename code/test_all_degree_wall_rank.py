#!/usr/bin/env python3
"""Regression tests for the all-degree wall-rank factorization."""
from __future__ import annotations

import hashlib
import json
import unittest
from fractions import Fraction
from pathlib import Path

import all_degree_wall_rank as awr
import universal_wall_transfer as uwt


def _fraction_mod(value: Fraction, prime: int) -> int:
    return (
        value.numerator
        * pow(value.denominator % prime, prime - 2, prime)
        % prime
    )


def _anchor_row_core(matrix: list[list[object]], bandwidth: int) -> list[list[object]]:
    """Reproduce the superseded anchor-row extraction for contrast only."""
    rows = len(matrix)
    anchor = rows // 2
    values = {
        offset: matrix[anchor][anchor + offset]
        for offset in range(-bandwidth, bandwidth + 1)
    }
    zero = type(matrix[0][0])(0)
    return [
        [values.get(column - row, zero) for column in range(rows)]
        for row in range(rows)
    ]


class CardinalCoreExtraction(unittest.TestCase):
    """Lock the comparator to the cardinal same-spacing Toeplitz section."""

    def test_small_size_core_matches_large_size_core(self) -> None:
        """Require the N=4p-2 core to equal the N=8p+9 core entrywise."""
        prime = awr.DEFAULT_PRIMES[0]
        for degree in (2, 3):
            small = 4 * degree - 2
            large = 8 * degree + 9
            exact_large = uwt.exact_coefficient_matrices(large, degree)
            modular_large = awr.modular_coefficient_matrices(
                large, degree, prime
            )
            for name in ("mass", "advection", "stiffness"):
                reference = uwt.exact_toeplitz_core(
                    exact_large[name], degree
                )
                observed = uwt.exact_cardinal_toeplitz_core(
                    small, degree, name
                )
                self.assertEqual(
                    observed,
                    [row[:small] for row in reference[:small]],
                    f"exact {name} core differs at p={degree}",
                )
            for name in ("mass", "advection"):
                reference = awr.toeplitz_core(modular_large[name], degree)
                observed = awr.modular_cardinal_toeplitz_core(
                    small, degree, prime, name
                )
                self.assertEqual(
                    observed,
                    [row[:small] for row in reference[:small]],
                    f"modular {name} core differs at p={degree}",
                )

    def test_derivative_core_matches_large_size_core(self) -> None:
        """Require the degree-1 velocity mass core to be size independent."""
        prime = awr.DEFAULT_PRIMES[0]
        for degree in (2, 3):
            small = 4 * degree - 3
            large = 8 * degree + 9
            reference = uwt.exact_toeplitz_core(
                uwt.exact_coefficient_matrices(large, degree - 1)["mass"],
                degree - 1,
            )
            observed = uwt.exact_cardinal_toeplitz_core(
                small, degree - 1, "mass"
            )
            self.assertEqual(
                observed, [row[:small] for row in reference[:small]]
            )
            modular_reference = awr.toeplitz_core(
                awr.modular_coefficient_matrices(large, degree - 1, prime)[
                    "mass"
                ],
                degree - 1,
            )
            modular_observed = awr.modular_cardinal_toeplitz_core(
                small, degree - 1, prime, "mass"
            )
            self.assertEqual(
                modular_observed,
                [row[:small] for row in modular_reference[:small]],
            )

    def test_anchor_row_core_is_contaminated_below_4p_minus_1(self) -> None:
        """Show the superseded anchor-row comparator differs at N=4p-2."""
        for degree in (2, 3):
            small = 4 * degree - 2
            mass = uwt.exact_coefficient_matrices(small, degree)["mass"]
            self.assertNotEqual(
                _anchor_row_core(mass, degree),
                uwt.exact_cardinal_toeplitz_core(small, degree, "mass"),
            )

    def test_cardinal_row_values_reject_small_matrices(self) -> None:
        """Require an anchor at least 2p+1 indices from both walls."""
        for degree in (2, 3, 4):
            mass = uwt.exact_coefficient_matrices(4 * degree + 2, degree)[
                "mass"
            ]
            with self.assertRaises(ValueError):
                uwt.cardinal_row_values(mass, degree)
            admissible = uwt.exact_coefficient_matrices(
                4 * degree + 3, degree
            )["mass"]
            band = uwt.cardinal_row_values(admissible, degree)
            self.assertEqual(len(band), 2 * degree + 1)

    def test_reference_size_choice_is_immaterial(self) -> None:
        """Require identical cardinal bands at 4p+3, 4p+9, and 8p+9."""
        for degree in (2, 3, 5):
            bands = set()
            for npts in (4 * degree + 3, 4 * degree + 9, 8 * degree + 9):
                matrix = uwt.exact_coefficient_matrices(npts, degree)["mass"]
                values = uwt.cardinal_row_values(matrix, degree)
                bands.add(
                    tuple(
                        values[offset]
                        for offset in range(-degree, degree + 1)
                    )
                )
            self.assertEqual(len(bands), 1)


class AllDegreeWallRank(unittest.TestCase):
    """Exact controls for the arbitrary-degree wall-rank formulas."""

    def test_modular_assembly_matches_rational_assembly(self) -> None:
        """Match modular Cox-de Boor matrices to reduced rational matrices."""
        prime = awr.DEFAULT_PRIMES[0]
        for degree, npts in ((2, 7), (4, 15), (7, 27)):
            rational = uwt.exact_coefficient_matrices(npts, degree)
            modular = awr.modular_coefficient_matrices(npts, degree, prime)
            for name in ("mass", "advection", "incidence"):
                reduced = [
                    [_fraction_mod(value, prime) for value in row]
                    for row in rational[name]
                ]
                self.assertEqual(modular[name], reduced)

    def test_cumulative_mass_factors(self) -> None:
        """Verify the triangular cumulative-defect mass factors exactly."""
        for degree in range(2, 9):
            result = awr.rational_mass_triangular_control(
                degree, 4 * degree + 3
            )
            self.assertTrue(result["triangular"])
            self.assertTrue(result["diagonal_nonzero"])
            self.assertTrue(result["diagonal_negative"])
            self.assertTrue(result["partition_null"])
            self.assertTrue(result["partition_quadratic_negative"])

    def test_piecewise_rank_formulas(self) -> None:
        """Recompute overlap, transition, and noninteracting ranks exactly."""
        prime = awr.DEFAULT_PRIMES[0]
        for degree in (2, 3, 5, 8, 12):
            for npts in (
                4 * degree - 3,
                4 * degree - 2,
                4 * degree - 1,
                4 * degree,
                4 * degree + 1,
            ):
                self.assertEqual(
                    awr.modular_control(degree, npts, prime),
                    awr.predicted_ranks(degree, npts),
                    f"rank formula failed at p={degree}, N={npts}",
                )

    def test_rational_lane_matches_modular_lane(self) -> None:
        """Agree between exact rationals and one prime field at small sizes."""
        prime = awr.DEFAULT_PRIMES[0]
        for degree in (2, 3, 4):
            for npts in (4 * degree - 3, 4 * degree - 2, 4 * degree - 1):
                exact = awr.rational_control(degree, npts)
                modular = awr.modular_control(degree, npts, prime)
                self.assertEqual(
                    exact,
                    {key: modular[key] for key in exact},
                    f"lane disagreement at p={degree}, N={npts}",
                )

    def test_saved_adversarial_controls(self) -> None:
        """Require two-prime agreement across the persisted control grid."""
        artifact = json.loads(awr.DEFAULT_OUTPUT.read_text())
        self.assertEqual(artifact["schema"], "all-degree-wall-rank-v3")
        self.assertIn("core-extraction correction", artifact["revision"])
        self.assertEqual(artifact["degree_range"], [2, 10])
        self.assertEqual(tuple(artifact["primes"]), awr.DEFAULT_PRIMES)
        self.assertEqual(
            artifact["hypotheses"], {"degree": "p >= 2", "size": "N >= 4p-3"}
        )
        self.assertEqual(artifact["noninteracting_range"], "N >= 4p")
        self.assertEqual(
            artifact["sharp_range"],
            {
                "bmg_pencil": "N >= 4p-1",
                "bmf_pencil": "N >= 4p-1",
                "bmf_explicit": "N >= 4p-1",
                "bmg_explicit": "N >= 4p",
            },
        )
        self.assertEqual(artifact["noninteracting_ranks"]["bmg_explicit"], "8p-6")
        self.assertEqual(artifact["noninteracting_ranks"]["bmf_explicit"], "6p-6")
        self.assertEqual(
            artifact["transition_at_N_4p_minus_2"],
            {
                "mass_correction": "N",
                "bmg_pencil": "8p-6",
                "bmg_explicit": "8p-8",
                "bmf_pencil": "8p-8",
                "bmf_explicit_x": "4p-5",
                "bmf_explicit_y": "2p-2",
                "bmf_explicit": "6p-7",
            },
        )
        self.assertEqual(artifact["deviations"], [])
        self.assertEqual(
            max(record["degree"] for record in artifact["records"]), 10
        )
        for record in artifact["records"]:
            expected = record["predicted"]
            self.assertEqual(
                expected,
                awr.predicted_ranks(record["degree"], record["npts"]),
            )
            for observed in record["prime_results"].values():
                self.assertEqual(observed, expected)
            rational = record.get("rational_result")
            if rational is not None:
                self.assertEqual(
                    rational, {key: expected[key] for key in rational}
                )
        for name, digest in artifact["source_sha256"].items():
            path = Path(awr.__file__).with_name(name)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), digest
            )


if __name__ == "__main__":
    unittest.main()
