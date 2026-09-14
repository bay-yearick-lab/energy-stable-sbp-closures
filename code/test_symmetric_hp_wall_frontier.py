"""Controls for the symmetric hp wall-observable frontier."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest

import numpy as np

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))

import s6_symmetric_hp_wall_frontier as frontier
from s6_offgrid_sem_geometry_frontier import _barycentric_rows, _basis_matrix


class SymmetricHPWallFrontierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specification = json.loads(frontier.SPEC_PATH.read_text())

    def test_configuration_matrix_is_symmetric_and_complete(self) -> None:
        configs = frontier.configurations(self.specification)
        self.assertEqual(len(configs), 425)
        self.assertEqual(len(configs), len({row["key"] for row in configs}))
        for family in ("spline", "sem"):
            self.assertEqual(
                {row["degree"] for row in configs if row["family"] == family},
                set(self.specification["degrees"]),
            )
        self.assertTrue(all(row["n"] <= 49 for row in configs))

    def test_same_partition_space_inclusion_and_dimension_dividend(self) -> None:
        degree, elements = 3, 4
        spline_n = elements + degree
        sem_n = elements * degree + 1
        points = np.linspace(0.0, 1.0, 301)
        spline_basis = _basis_matrix(spline_n, degree, points)
        sem_basis, _ = _barycentric_rows(points, degree, elements)
        coefficient, *_ = np.linalg.lstsq(sem_basis, spline_basis, rcond=None)
        residual = np.linalg.norm(sem_basis @ coefficient - spline_basis, ord=np.inf)
        self.assertLess(residual, 2.0e-12)
        self.assertEqual(sem_n - spline_n, (degree - 1) * (elements - 1))

    def test_equal_state_spaces_are_distinct(self) -> None:
        degree, n = 3, 13
        spline_basis = _basis_matrix(n, degree, np.linspace(0.0, 1.0, 401))
        sem_basis, _ = _barycentric_rows(np.linspace(0.0, 1.0, 401), degree, 4)
        self.assertEqual(np.linalg.matrix_rank(spline_basis), n)
        self.assertEqual(np.linalg.matrix_rank(sem_basis), n)
        self.assertGreater(np.linalg.matrix_rank(np.column_stack([spline_basis, sem_basis])), n)

    def test_design_and_heldout_fields_are_disjoint(self) -> None:
        design = frontier.build_fields(self.specification, "design")
        heldout = frontier.build_fields(self.specification, "heldout")
        self.assertEqual(len(design), 16)
        self.assertEqual(len(heldout), 16)
        design_identity = {(field.field_class, field.parameters) for field in design}
        heldout_identity = {(field.field_class, field.parameters) for field in heldout}
        self.assertFalse(design_identity & heldout_identity)
        self.assertLess(frontier._derivative_gate(design), 2.0e-6)
        self.assertLess(frontier._derivative_gate(heldout), 2.0e-6)

    def test_native_reconstruction_smoke(self) -> None:
        fields = frontier.build_fields(self.specification, "design")[:2]
        samples = frontier.sampling_record(self.specification, fields)
        cases = [
            {"key": "spline:p6:k7", "family": "spline", "degree": 6, "elements": 7, "n": 13},
            {"key": "sem:p6:e2", "family": "sem", "degree": 6, "elements": 2, "n": 13},
        ]
        for config in cases:
            rows, gate = frontier.evaluate_configuration(
                config, fields, samples, self.specification,
            )
            self.assertEqual(len(rows), len(fields))
            self.assertLess(max(gate.values()), 2.0e-10)
            for row in rows:
                values = np.asarray(list(row["metrics"].values()))
                self.assertTrue(np.all(np.isfinite(values)))
                self.assertTrue(np.all(values >= 0.0))

    def test_cost_accounting_activates_repeated_queries(self) -> None:
        config = {"key": "spline:p6:k19", "family": "spline", "degree": 6, "elements": 19, "n": 25}
        cost = frontier.configuration_cost(config, self.specification)
        self.assertEqual(cost["scalar_state"], 3 * 25**2)
        self.assertGreater(cost["state_prepare_work"], 0)
        self.assertGreater(cost["marginal_query_work"], cost["state_prepare_work"])
        entries = frontier._resource_entries(config, cost, self.specification)
        missions = [entry for entry in entries if entry["kind"] == "mission"]
        by_query = {}
        for entry in missions:
            by_query.setdefault(entry["query_count"], entry["value"])
        self.assertLess(by_query[1], by_query[10])
        self.assertLess(by_query[10], by_query[100])


if __name__ == "__main__":
    unittest.main()
