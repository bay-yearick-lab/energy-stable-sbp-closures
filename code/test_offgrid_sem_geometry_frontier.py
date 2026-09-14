"""Independent controls for the off-grid hp-SEM geometry frontier."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))

import s6_offgrid_sem_geometry_frontier as frontier
from s6_curved_offgrid import random_field, sem_entrant, surfaces


def test_admissible_matrix() -> None:
    expected_counts = {19: 34, 37: 74, 61: 124}
    for budget, expected_count in expected_counts.items():
        configurations = frontier._admissible_sem_configurations(budget)
        assert len(configurations) == expected_count
        assert len(configurations) == len(set(configurations))
        assert all(2 <= degree <= 12 for degree, _ in configurations)
        assert all(elements * degree + 1 <= budget for degree, elements in configurations)
        for degree in range(2, 13):
            observed = [elements for current_degree, elements in configurations if current_degree == degree]
            assert observed == list(range(1, (budget - 1) // degree + 1))


def test_polynomial_reproduction() -> None:
    archived_surfaces = surfaces()
    entrants = [
        frontier.SplineReconstruction(19),
        frontier.SplineReconstruction(61),
        frontier.SpectralElementReconstruction(2, 30),
        frontier.SpectralElementReconstruction(7, 8),
        frontier.SpectralElementReconstruction(12, 5),
    ]
    for entrant in entrants:
        gate = frontier._polynomial_gate(entrant, archived_surfaces)
        assert gate["value_max_abs"] < 2.0e-10
        assert gate["derivative_max_abs"] < 2.0e-10


def test_native_sem_matches_archived_degree_six() -> None:
    rng = np.random.default_rng(20260712)
    random_field(rng)
    field = random_field(rng)
    surface = surfaces()[2]
    native = frontier.SpectralElementReconstruction(6, 3)
    sampled = frontier._sample_state(field, native.grid)
    result = native.plan(surface["x"], surface["y"]).evaluate(sampled)
    archived = sem_entrant(19, field, surface)
    for name in ("u", "v", "p", "ux", "uy", "vx", "vy"):
        assert np.allclose(result[name], archived[name], rtol=2.0e-12, atol=2.0e-12)


def test_analytic_and_orientation_gates() -> None:
    rng = np.random.default_rng(20260712)
    random_field(rng)
    fields = [random_field(rng) for _ in range(12)]
    gate = frontier._analytic_gates(fields, surfaces())
    assert gate["normal_unit_max_abs"] < 2.0e-14
    assert gate["normal_outward_dot_min"] > 0.02
    assert gate["analytic_derivative_fd_max_abs"] < 2.0e-7
    assert gate["analytic_divergence_max_abs"] < 2.0e-12


def main() -> None:
    test_admissible_matrix()
    test_polynomial_reproduction()
    test_native_sem_matches_archived_degree_six()
    test_analytic_and_orientation_gates()
    print("offgrid SEM geometry frontier tests: ALL PASS")


if __name__ == "__main__":
    main()
