#!/usr/bin/env python3
"""Adversarial tests for persisted numerical-evidence verifiers."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import s6_hpsem_frontier as hpsem
import s6_offgrid_ensemble as offgrid
import s6_wallclock_cost as wallclock
import s6_curved_offgrid as curved
import s6_sem_headtohead as semhead
import s6_sem_orders as semorders
import s6_heat_cavity_convergence as heatconv
import s6_heat_cavity_strong_competitors as heatstrong


RESULTS = Path(__file__).resolve().parent / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


class OffgridVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = load("offgrid_ensemble.json")

    def test_released_record_recomputes_exactly(self):
        offgrid.verify_record(self.record, recompute=True)

    def test_forged_error_is_rejected(self):
        forged = deepcopy(self.record)
        forged["results"][0]["median"] *= 0.5
        with self.assertRaises((AssertionError, ValueError)):
            offgrid.verify_record(forged, recompute=True)

    def test_missing_family_metric_grid_row_is_rejected_before_recompute(self):
        forged = deepcopy(self.record)
        forged["results"].pop()
        with self.assertRaises((AssertionError, ValueError)):
            offgrid.verify_record(forged, recompute=False)


class HpsemVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = load("hpsem_frontier_summary.json")

    def test_released_frontier_is_reconstructed_from_raw_cases(self):
        hpsem.verify_record(self.record)

    def test_forged_frontier_winner_is_rejected(self):
        forged = deepcopy(self.record)
        forged["frontier"]["heat_sem"]["1e-08"]["by_cpu"]["cpu"] *= 0.5
        with self.assertRaises((AssertionError, ValueError)):
            hpsem.verify_record(forged)

    def test_missing_sweep_case_is_rejected(self):
        forged = deepcopy(self.record)
        forged["sem_heat"].pop()
        with self.assertRaises((AssertionError, ValueError)):
            hpsem.verify_record(forged)

    def test_spline_only_rescore_preserves_sem_cases_and_replaces_every_spline_case(self):
        old = deepcopy(self.record)
        sem_heat = deepcopy(old["sem_heat"])
        sem_wave = deepcopy(old["sem_wave"])

        def fake_heat(label, npts, _modes):
            return 1.0 / npts, 0.01 * npts, npts

        def fake_wave(label, npts):
            return 2.0 / npts, 0.02 * npts, npts

        rescored = hpsem.rescore_splines(
            old,
            checkpoint=None,
            heat_case=fake_heat,
            wave_case=fake_wave,
        )

        self.assertEqual(rescored["sem_heat"], sem_heat)
        self.assertEqual(rescored["sem_wave"], sem_wave)
        self.assertEqual(len(rescored["spline_heat"]), 42)
        self.assertEqual(len(rescored["spline_wave"]), 42)
        self.assertEqual(
            rescored["config"]["wave_norm"],
            "full-domain coupled relative L2 with actual-node trapezoid weights",
        )
        self.assertTrue(all(row["err"] != old["spline_heat"][i]["err"]
                            for i, row in enumerate(rescored["spline_heat"])))

    def test_sem_only_rescore_preserves_spline_cases_and_attaches_timing_provenance(self):
        old = deepcopy(self.record)
        new_heat = deepcopy(old["sem_heat"])
        new_wave = deepcopy(old["sem_wave"])
        for row in new_heat + new_wave:
            row["cpu"] += 0.125
        timing = {
            "metric": "process_time_seconds",
            "cpu": "test cpu",
            "platform": "test platform",
            "python": "3.test",
            "numpy": "2.test",
            "thread_env": {name: "1" for name in hpsem.THREAD_ENV_NAMES},
            "nice": 19,
        }
        rescored = hpsem.replace_sem_sweeps(old, new_heat, new_wave, timing)
        self.assertEqual(rescored["spline_heat"], old["spline_heat"])
        self.assertEqual(rescored["spline_wave"], old["spline_wave"])
        self.assertEqual(rescored["sem_heat"], new_heat)
        self.assertEqual(rescored["sem_wave"], new_wave)
        self.assertEqual(rescored["timing"], timing)


class CurvedOffgridVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = load("curved_offgrid_summary.json")

    def test_released_curved_surface_record_recomputes_exactly(self):
        curved.verify_record(self.record, recompute=True)

    def test_forged_traction_error_is_rejected(self):
        forged = deepcopy(self.record)
        row = next(r for r in forged["results"] if r["metric"] == "t")
        row["worst"] *= 0.5
        with self.assertRaises((AssertionError, ValueError)):
            curved.verify_record(forged, recompute=True)

    def test_missing_curved_surface_row_is_rejected_before_recompute(self):
        forged = deepcopy(self.record)
        forged["results"].pop()
        with self.assertRaises((AssertionError, ValueError)):
            curved.verify_record(forged, recompute=False)

class WallclockVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = load("wallclock_cost.json")

    def test_released_statistics_recompute_from_repeats(self):
        wallclock.verify_record(self.record)

    def test_forged_median_is_rejected(self):
        forged = deepcopy(self.record)
        forged["cases"][0]["median_cpu_seconds"] *= 0.5
        with self.assertRaises((AssertionError, ValueError)):
            wallclock.verify_record(forged)

    def test_missing_label_grid_case_is_rejected(self):
        forged = deepcopy(self.record)
        forged["cases"].pop()
        with self.assertRaises((AssertionError, ValueError)):
            wallclock.verify_record(forged)

    def test_error_only_rescore_preserves_every_timing_sample(self):
        before = deepcopy(self.record)

        def fake_run(label, npts, _modes):
            return 99.0, 88.0, {"rel_l2": 1.0 / npts, "linf": 2.0 / npts,
                                "steps": 4, "dt": 0.5}

        after = wallclock.rescore_spline_errors(before, runner=fake_run)
        for old, new in zip(self.record["cases"], after["cases"]):
            for key in ("median_cpu_seconds", "median_wall_seconds", "iqr_cpu_seconds",
                        "repeat_cpu_seconds", "repeat_wall_seconds"):
                self.assertEqual(new[key], old[key])
            if old["label"] in ("BMF6", "BMG6"):
                self.assertEqual(new["rel_l2"], 1.0 / old["npts"])
            else:
                self.assertEqual(new["rel_l2"], old["rel_l2"])


class SemAcousticScoringTests(unittest.TestCase):
    @staticmethod
    def fake_result():
        return SimpleNamespace(
            error_history=[99.0],
            grid=[0.0, 0.1, 1.0],
            pressure=[2.0, 1.0, 1.0],
            velocity=[0.0, 2.0, 0.0],
            pressure_exact=[1.0, 1.0, 1.0],
            velocity_exact=[0.0, 0.0, 0.0],
        )

    def test_hpsem_spline_wave_uses_full_domain_actual_node_weights(self):
        seen = {}

        def simulate(*args, **kwargs):
            seen.update(kwargs)
            return self.fake_result()

        with patch.object(hpsem.ba, "build_operator", return_value=object()), \
             patch.object(hpsem.ba, "simulate_reflected_acoustics", side_effect=simulate):
            error, _cpu, _npts = hpsem.spline_wave_case("BMF4", 17)

        weights = hpsem._nonuniform_trap(np.array([0.0, 0.1, 1.0]))
        expected = ((weights[0] * 1.0 + weights[1] * 4.0) / weights.sum()) ** 0.5
        self.assertEqual(seen["interior_strip"], 0)
        self.assertAlmostEqual(error, expected, places=14)

    def test_fixed_sem_headtohead_uses_the_same_full_domain_score(self):
        seen = {}

        def simulate(*args, **kwargs):
            seen.update(kwargs)
            return self.fake_result()

        with patch.object(semhead.ba, "build_operator", return_value=object()), \
             patch.object(semhead.ba, "simulate_reflected_acoustics", side_effect=simulate):
            record = semhead.run_spline_wave("BMF", 43)

        weights = semhead._nonuniform_trap(np.array([0.0, 0.1, 1.0]))
        expected = ((weights[0] * 1.0 + weights[1] * 4.0) / weights.sum()) ** 0.5
        self.assertEqual(seen["interior_strip"], 0)
        self.assertAlmostEqual(record["rel_l2_final"], expected, places=14)

    def test_across_order_sem_comparison_uses_the_same_full_domain_score(self):
        seen = {}

        def simulate(*args, **kwargs):
            seen.update(kwargs)
            return self.fake_result()

        with patch.object(semorders.ba, "build_operator", return_value=object()), \
             patch.object(semorders.ba, "simulate_reflected_acoustics", side_effect=simulate):
            record = semorders.spline_wave(4, "BMF", 49)

        weights = semorders._nonuniform_trap(np.array([0.0, 0.1, 1.0]))
        expected = ((weights[0] * 1.0 + weights[1] * 4.0) / weights.sum()) ** 0.5
        self.assertEqual(seen["interior_strip"], 0)
        self.assertAlmostEqual(record["rel_l2_final"], expected, places=14)

    def test_fixed_headtohead_rescore_preserves_sem_and_replaces_spline_rows(self):
        old = load("sem_headtohead_summary.json")

        def fake_heat(label, npts, _modes):
            return {"family": label, "npts": npts, "dofs": npts * npts,
                    "dt": 0.1, "nsteps": 5, "rel_l2_final": 1.0 / npts,
                    "cpu_seconds": 0.1}

        def fake_wave(family, npts):
            return {"family": f"{family}6", "npts": npts,
                    "rel_l2_final": 2.0 / npts, "cpu_seconds": 0.2}

        rescored = semhead.rescore_spline_cases(old, fake_heat, fake_wave)
        self.assertEqual(
            [r for r in rescored["heat"] if r["family"] == "SEM6"],
            [r for r in old["heat"] if r["family"] == "SEM6"],
        )
        self.assertEqual(
            [r for r in rescored["wave"] if r["family"] == "SEM6"],
            [r for r in old["wave"] if r["family"] == "SEM6"],
        )
        self.assertEqual(len(rescored["heat"]), 20)
        self.assertEqual(len(rescored["wave"]), 4)

    def test_fixed_sem_only_rescore_preserves_splines_and_attaches_timing(self):
        old = load("sem_headtohead_summary.json")
        stamp = {
            "metric": "process_time_seconds",
            "cpu": "test cpu",
            "platform": "test platform",
            "python": "3.test",
            "numpy": "2.test",
            "thread_env": {name: "1" for name in semhead.THREAD_ENV_NAMES},
            "nice": 19,
        }

        def fake_heat(npts, _modes):
            return {
                "family": "SEM6", "npts": npts, "dofs": npts * npts,
                "dt": 0.01, "nsteps": 50, "rel_l2_final": 1.0 / npts,
                "rel_l2_final_native_gll": 1.01 / npts, "cpu_seconds": 0.5,
            }

        def fake_wave(npts):
            return {
                "family": "SEM6", "npts": npts, "dt": 0.001,
                "nsteps": 6000, "rel_l2_final": 1.0e-4,
                "cpu_seconds": 1.0, "abscissa": 0.0,
            }

        rescored = semhead.rescore_sem_cases(
            old, heat_runner=fake_heat, wave_runner=fake_wave, stamp=stamp
        )
        self.assertEqual(
            [row for row in rescored["heat"] if row["family"] != "SEM6"],
            [row for row in old["heat"] if row["family"] != "SEM6"],
        )
        self.assertEqual(
            [row for row in rescored["wave"] if row["family"] != "SEM6"],
            [row for row in old["wave"] if row["family"] != "SEM6"],
        )
        self.assertEqual(rescored["timing"], stamp)

    def test_across_order_rescore_preserves_sem_and_replaces_spline_rows(self):
        old = load("sem_orders_summary.json")

        def fake_heat(q, label, npts, _modes):
            return {"family": label, "q": q, "npts": npts,
                    "rel_l2_final": 1.0 / npts, "cpu_seconds": 0.1}

        def fake_wave(q, family, npts):
            return {"family": f"{family}{q}", "q": q,
                    "rel_l2_final": 2.0 / npts}

        rescored = semorders.rescore_spline_cases(old, fake_heat, fake_wave)
        self.assertEqual(
            [r for r in rescored["heat"] if r["family"].startswith("SEM")],
            [r for r in old["heat"] if r["family"].startswith("SEM")],
        )
        self.assertEqual(
            [r for r in rescored["wave"] if r["family"].startswith("SEM")],
            [r for r in old["wave"] if r["family"].startswith("SEM")],
        )
        self.assertEqual(len(rescored["heat"]), 24)
        self.assertEqual(len(rescored["wave"]), 8)


class HeatRescoreMergeTests(unittest.TestCase):
    def test_convergence_merge_replaces_only_spline_sweeps(self):
        old = {label: [{"label": label, "npts": n, "rel_l2_final": 9.0}
                       for n in heatconv.CONVERGENCE["grids"]]
               for label in heatconv.heat_cavity_labels("six_family")}
        fresh = {label: [{"label": label, "npts": n, "rel_l2_final": 1.0 / n}
                         for n in heatconv.CONVERGENCE["grids"]]
                 for label in ("BMF4", "BMG4", "BMF6", "BMG6", "BMF8", "BMG8")}
        merged = heatconv.merge_spline_sweeps(old, fresh)
        self.assertEqual(merged["SBP8"], old["SBP8"])
        self.assertEqual(merged["BMF8"], fresh["BMF8"])

    def test_strong_fixed_grid_merge_replaces_only_two_spline_records(self):
        old = {label: {"final_rel_l2": 9.0, "final_linf": 8.0,
                       "steps": 1, "dt": 0.5}
               for label in heatstrong.LABELS_FIXED}
        fresh = {label: {"final_rel_l2": 1.0, "final_linf": 2.0,
                         "steps": 3, "dt": 0.25}
                 for label in ("BMF6", "BMG6")}
        merged = heatstrong.merge_fixed_spline_records(old, fresh)
        self.assertEqual(merged["SBP6-BO"], old["SBP6-BO"])
        self.assertEqual(merged["BMF6"], fresh["BMF6"])

if __name__ == "__main__":
    unittest.main()
