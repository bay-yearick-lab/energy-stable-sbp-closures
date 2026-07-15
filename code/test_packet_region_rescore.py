#!/usr/bin/env python3
"""Unit tests for the localized-packet physical-region rescore."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

import s6_packet_region_rescore as rescore


def valid_record(family="BMF"):
    config = rescore.build_config(
        families=(family,), orders=(4,), grids=(17,), cycles=(1,),
        strip_width=0.1, smoke=True
    )
    x = rescore.expected_grid(family, 4, 17)
    pref = np.cos(np.pi * x)
    uref = np.sin(np.pi * x)
    metric = rescore.region_l2_diagnostics(
        x, pref + 0.01, uref - 0.02, pref, uref, strip_width=0.1
    )
    metric.update(
        status="ok",
        grid_provenance=rescore.grid_provenance(x, 0.1),
        phase_shift=0.0,
        amp_ratio=1.0,
        waveform_distortion=0.01,
        leakage=0.001,
    )
    run = {"impact": deepcopy(metric), "cycle1": deepcopy(metric), "cpu_seconds": 0.1}
    key = f"{family}4_N17"
    return rescore.record_envelope(
        config, {key: run}, rescore.exact_recurrence_selftest()
    )


def synthetic_process_worker(spec):
    """Cheap picklable process worker that exercises the real adapter."""
    original_metrics = rescore.packet_source.metrics
    original_cycles = rescore.packet_source.CYCLES
    x = rescore.expected_grid(spec["family"], spec["order"], spec["grid_size"])
    time_value = 0.31
    pref = rescore.packet_source.p_exact(x, time_value)
    uref = rescore.packet_source.u_exact(x, time_value)
    pressure = pref + 0.01
    velocity = uref - 0.02

    def fake_runner(family, order, grid_size):
        metric = rescore.packet_source.metrics(
            x, pressure, velocity, time_value
        )
        out = {"impact": metric, "cpu_seconds": 0.0}
        for cycle in spec["cycles"]:
            out[f"cycle{cycle}"] = deepcopy(metric)
        return out

    run = rescore._run_source(
        spec["family"],
        spec["order"],
        spec["grid_size"],
        tuple(spec["cycles"]),
        spec["strip_width"],
        runner=fake_runner,
    )
    assert rescore.packet_source.metrics is original_metrics
    assert rescore.packet_source.CYCLES == original_cycles
    return spec["key"], run


def worker_must_not_run(spec):
    raise AssertionError(f"checkpointed case was rerun: {spec['key']}")


class RegionMaskTests(unittest.TestCase):
    def test_wall_and_interior_masks_are_exact_complements(self):
        x = np.array([0.0, 0.1, 0.1000001, 0.6, 0.8999999, 0.9, 1.0])

        masks = rescore.region_masks(x, 0.1)

        np.testing.assert_array_equal(masks["full"], np.ones(x.size, dtype=bool))
        expected_wall = (x <= 0.1) | (x >= 0.9)
        np.testing.assert_array_equal(masks["wall_strip"], expected_wall)
        np.testing.assert_array_equal(masks["interior"], ~expected_wall)
        np.testing.assert_array_equal(
            masks["wall_strip"] | masks["interior"], masks["full"]
        )
        self.assertFalse(np.any(masks["wall_strip"] & masks["interior"]))

    def test_default_width_mask_matches_source_driver_exactly(self):
        x = np.linspace(0.0, 1.0, 101)

        masks = rescore.region_masks(x, rescore.packet_source.STRIP)

        np.testing.assert_array_equal(
            masks["wall_strip"], rescore.packet_source.strip_mask(x)
        )

    def test_invalid_strip_widths_are_rejected(self):
        x = np.linspace(0.0, 1.0, 5)
        for width in (0.0, -0.1, 0.5, 0.9, 1.0, 1.1, float("nan"), float("inf")):
            with self.subTest(width=width), self.assertRaises(ValueError):
                rescore.region_masks(x, width)


class RegionDiagnosticTests(unittest.TestCase):
    def test_raw_numerators_and_denominators_partition_full_domain(self):
        x = np.linspace(0.0, 1.0, 5)
        pref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        uref = np.array([0.5, -0.5, 1.0, -1.0, 0.25])
        p = pref + np.array([0.1, -0.2, 0.3, -0.4, 0.5])
        u = uref + np.array([-0.3, 0.2, -0.1, 0.4, -0.5])

        got = rescore.region_l2_diagnostics(
            x, p, u, pref, uref, strip_width=0.25
        )

        regions = got["regions"]
        self.assertAlmostEqual(
            regions["full"]["numerator_sq"],
            regions["wall_strip"]["numerator_sq"]
            + regions["interior"]["numerator_sq"],
            places=14,
        )
        self.assertAlmostEqual(
            regions["full"]["denominator_sq"],
            regions["wall_strip"]["denominator_sq"]
            + regions["interior"]["denominator_sq"],
            places=14,
        )

    def test_analytic_scaling_has_expected_full_normalized_contributions(self):
        x = np.linspace(0.0, 1.0, 5)
        pref = np.ones_like(x)
        uref = np.zeros_like(x)
        p = 3.0 * pref
        u = uref.copy()

        got = rescore.region_l2_diagnostics(
            x, p, u, pref, uref, strip_width=0.25
        )

        self.assertAlmostEqual(got["full_rel_l2"], 2.0, places=14)
        self.assertAlmostEqual(got["wall_strip_rel_l2"], math.sqrt(3.0), places=14)
        self.assertAlmostEqual(got["interior_rel_l2"], 1.0, places=14)
        self.assertAlmostEqual(
            got["wall_strip_rel_l2"] ** 2 + got["interior_rel_l2"] ** 2,
            got["full_rel_l2"] ** 2,
            places=14,
        )


class RecordContractTests(unittest.TestCase):
    def test_config_rejects_invalid_route_order_and_grid_combinations(self):
        invalid = (
            {"families": ("BMF",), "orders": (2,), "grids": (17,)},
            {"families": ("BMF",), "orders": (4,), "grids": (12,)},
            {"families": ("BMG",), "orders": (4,), "grids": (12,)},
            {"families": ("SEM",), "orders": (4,), "grids": (18,)},
            {"families": ("SEM",), "orders": (8,), "grids": (8,)},
        )
        for selection in invalid:
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                rescore.build_config(
                    **selection, cycles=(1,), strip_width=0.1, smoke=True
                )

    def test_config_accepts_route_admissible_bmf_and_sem_grid(self):
        config = rescore.build_config(
            families=("BMF", "SEM"), orders=(4,), grids=(17,), cycles=(1,),
            strip_width=0.1, smoke=True
        )

        self.assertEqual(config["families"], ["BMF", "SEM"])
        self.assertEqual(config["grids"], [17])

    def test_config_rejects_duplicate_selectors_before_key_assembly(self):
        duplicates = (
            {"families": ("BMF", "BMF"), "orders": (4,), "grids": (17,), "cycles": (1,)},
            {"families": ("BMF",), "orders": (4, 4), "grids": (17,), "cycles": (1,)},
            {"families": ("BMF",), "orders": (4,), "grids": (17, 17), "cycles": (1,)},
            {"families": ("BMF",), "orders": (4,), "grids": (17,), "cycles": (1, 1)},
        )
        for selection in duplicates:
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                rescore.build_config(**selection, strip_width=0.1, smoke=True)

    def test_smoke_selection_leaves_a_nonempty_solver_interior(self):
        families, orders, grids, cycles = rescore.smoke_selection()

        self.assertEqual((families, orders, cycles), (("BMF", "SEM"), (4,), (1,)))
        self.assertGreater(grids[0], 2 * 6)

    def test_parser_accepts_two_process_workers(self):
        args = rescore.parse_args(["--workers", "2"])

        self.assertEqual(args.workers, 2)

    def test_record_envelope_has_schema_and_source_provenance(self):
        config = rescore.build_config(
            families=("BMF",), orders=(4,), grids=(17,), cycles=(1,),
            strip_width=0.1, smoke=True
        )

        record = rescore.record_envelope(config, runs={}, exact_selftest={})

        self.assertEqual(record["schema"], "packet-region-rescore-v2")
        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(
            record["provenance"]["source_driver"], "code/s6_packet_reflection.py"
        )
        self.assertEqual(
            record["provenance"]["driver"], "code/s6_packet_region_rescore.py"
        )
        self.assertEqual(len(record["provenance"]["driver_sha256"]), 64)
        self.assertEqual(len(record["provenance"]["source_sha256"]), 64)
        self.assertEqual(
            record["provenance"]["region_rules"],
            {
                "wall_strip": "(x <= strip_width) | (x >= 1 - strip_width)",
                "interior": "complement of wall_strip",
            },
        )
        self.assertEqual(
            record["provenance"]["relative_l2_normalization"],
            "full-domain exact squared norm",
        )
        self.assertTrue(record["config"]["smoke"])

    def test_write_new_json_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json"
            path.write_text('{"sentinel": true}\n')

            with self.assertRaises(FileExistsError):
                rescore.write_new_json(path, {"replacement": True})

            self.assertEqual(json.loads(path.read_text()), {"sentinel": True})

    def test_write_new_json_rejects_nonfinite_values_before_creating_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json"

            with self.assertRaises(ValueError):
                rescore.write_new_json(path, {"bad": float("nan")})

            self.assertFalse(path.exists())


class VerifierAdversarialTests(unittest.TestCase):
    def assert_rejected(self, record):
        with self.assertRaises((AssertionError, KeyError, ValueError)):
            rescore.verify_record(record)

    def test_valid_synthetic_record_passes(self):
        rescore.verify_record(valid_record())

    def test_missing_and_extra_run_keys_are_rejected(self):
        missing = valid_record()
        missing["runs"] = {}
        self.assert_rejected(missing)

        extra = valid_record()
        extra["runs"]["BMF4_N999"] = deepcopy(extra["runs"]["BMF4_N17"])
        self.assert_rejected(extra)

    def test_missing_and_extra_tags_are_rejected(self):
        missing = valid_record()
        missing["runs"]["BMF4_N17"].pop("cycle1")
        self.assert_rejected(missing)

        extra = valid_record()
        extra["runs"]["BMF4_N17"]["cycle999"] = deepcopy(
            extra["runs"]["BMF4_N17"]["impact"]
        )
        self.assert_rejected(extra)

    def test_forged_source_hash_is_rejected(self):
        record = valid_record()
        record["provenance"]["source_sha256"] = "0" * 64
        self.assert_rejected(record)

    def test_forged_exact_recurrence_scalar_is_rejected(self):
        record = valid_record()
        record["exact_selftest"]["p_t2_vs_t0"] = 1.0e-16

        self.assert_rejected(record)

    def test_optimized_interpreter_rejects_forged_hash_and_scaled_norm(self):
        records = []
        forged_hash = valid_record()
        forged_hash["provenance"]["source_sha256"] = "0" * 64
        records.append(forged_hash)
        scaled_norm = valid_record()
        scaled_norm["runs"]["BMF4_N17"]["cycle1"]["full_rel_l2"] *= 7.0
        records.append(scaled_norm)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with tempfile.TemporaryDirectory() as tmp:
            for index, record in enumerate(records):
                path = Path(tmp) / f"forged-{index}.json"
                path.write_text(json.dumps(record, allow_nan=False))

                result = subprocess.run(
                    [
                        sys.executable,
                        "-O",
                        str(Path(rescore.__file__).resolve()),
                        "--verify",
                        "--output",
                        str(path),
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )

                with self.subTest(index=index, stdout=result.stdout, stderr=result.stderr):
                    self.assertNotEqual(result.returncode, 0)

    def test_wrong_normalized_and_raw_fields_are_rejected(self):
        cases = []
        wrong_top = valid_record()
        wrong_top["runs"]["BMF4_N17"]["cycle1"]["full_rel_l2"] *= 2.0
        cases.append(wrong_top)

        wrong_region = valid_record()
        wrong_region["runs"]["BMF4_N17"]["cycle1"]["regions"]["interior"][
            "full_reference_rel_l2"
        ] *= 2.0
        cases.append(wrong_region)

        wrong_denominator = valid_record()
        wrong_denominator["runs"]["BMF4_N17"]["cycle1"]["regions"]["wall_strip"][
            "normalization_denominator_sq"
        ] *= 2.0
        cases.append(wrong_denominator)

        missing_raw = valid_record()
        missing_raw["runs"]["BMF4_N17"]["cycle1"]["regions"]["full"].pop(
            "numerator_sq"
        )
        cases.append(missing_raw)

        negative_raw = valid_record()
        negative_raw["runs"]["BMF4_N17"]["cycle1"]["regions"]["full"][
            "numerator_sq"
        ] = -1.0
        cases.append(negative_raw)

        for record in cases:
            with self.subTest(case=cases.index(record)):
                self.assert_rejected(record)

    def test_zero_counts_cannot_impersonate_configured_grid(self):
        record = valid_record()
        regions = record["runs"]["BMF4_N17"]["cycle1"]["regions"]
        for region in regions.values():
            region["point_count"] = 0

        self.assert_rejected(record)

    def test_forged_coordinate_and_mask_hashes_are_rejected(self):
        records = []
        coordinate = valid_record()
        coordinate["runs"]["BMF4_N17"]["cycle1"]["grid_provenance"][
            "coordinate_sha256"
        ] = "0" * 64
        records.append(coordinate)
        mask = valid_record()
        mask["runs"]["BMF4_N17"]["cycle1"]["grid_provenance"][
            "wall_mask_sha256"
        ] = "f" * 64
        records.append(mask)

        for record in records:
            self.assert_rejected(record)

    def test_nonfinite_values_are_rejected_for_every_family(self):
        for family in rescore.ALL_FAMILIES:
            record = valid_record(family)
            record["runs"][f"{family}4_N17"]["cycle1"]["full_rel_l2"] = float("nan")
            with self.subTest(family=family):
                self.assert_rejected(record)

    def test_nodal_instability_uses_explicit_finite_status_record(self):
        record = valid_record("E")
        provenance = deepcopy(
            record["runs"]["E4_N17"]["cycle1"]["grid_provenance"]
        )
        record["runs"]["E4_N17"]["cycle1"] = {
            "status": "unstable_nonfinite",
            "reason": "nonfinite numerical state",
            "grid_provenance": provenance,
        }

        rescore.verify_record(record)


class FiniteStatusTests(unittest.TestCase):
    def test_packet_metrics_records_coordinate_and_mask_hashes(self):
        x = np.linspace(0.0, 1.0, 17)
        pressure = rescore.packet_source.p_exact(x, 0.31)
        velocity = rescore.packet_source.u_exact(x, 0.31)

        metric = rescore.packet_metrics(x, pressure, velocity, 0.31, 0.1)

        provenance = metric["grid_provenance"]
        self.assertEqual(provenance["point_count"], 17)
        self.assertEqual(provenance["wall_point_count"], 4)
        self.assertEqual(provenance["interior_point_count"], 13)
        for field in (
            "coordinate_sha256", "wall_mask_sha256", "interior_mask_sha256"
        ):
            self.assertEqual(len(provenance[field]), 64)

    def test_packet_metrics_converts_nonfinite_state_to_explicit_status(self):
        x = np.linspace(0.0, 1.0, 17)
        pressure = np.full_like(x, np.nan)
        velocity = np.zeros_like(x)

        metric = rescore.packet_metrics(x, pressure, velocity, 0.31, 0.1)

        self.assertEqual(metric["status"], "unstable_nonfinite")
        self.assertEqual(metric["reason"], "nonfinite numerical state")
        self.assertEqual(metric["grid_provenance"]["point_count"], 17)


class AdapterContractTests(unittest.TestCase):
    def test_adapter_preserves_source_metrics_and_restores_state_after_success(self):
        original_metrics = rescore.packet_source.metrics
        original_cycles = rescore.packet_source.CYCLES
        x = np.linspace(0.0, 1.0, 17)
        time_value = 0.31
        pref = rescore.packet_source.p_exact(x, time_value)
        uref = rescore.packet_source.u_exact(x, time_value)
        pressure = pref + 0.01 * np.cos(2.0 * np.pi * x)
        velocity = uref - 0.02 * np.sin(np.pi * x)
        expected = original_metrics(x, pressure, velocity, time_value)
        seen = {}

        def fake_runner(family, order, grid_size):
            seen.update(family=family, order=order, grid_size=grid_size)
            seen["cycles"] = rescore.packet_source.CYCLES
            metric = rescore.packet_source.metrics(
                x, pressure, velocity, time_value
            )
            return {"impact": metric, "cycle1": deepcopy(metric), "cpu_seconds": 0.0}

        got = rescore._run_source(
            "BMF", 4, 17, (1,), 0.1, runner=fake_runner
        )

        self.assertEqual(seen, {"family": "BMF", "order": 4, "grid_size": 17, "cycles": (1,)})
        for key in (
            "full_rel_l2", "phase_shift", "amp_ratio", "waveform_distortion", "leakage"
        ):
            self.assertAlmostEqual(got["impact"][key], expected[key], places=14)
        self.assertIs(rescore.packet_source.metrics, original_metrics)
        self.assertEqual(rescore.packet_source.CYCLES, original_cycles)

    def test_adapter_restores_source_state_after_runner_exception(self):
        original_metrics = rescore.packet_source.metrics
        original_cycles = rescore.packet_source.CYCLES

        def failing_runner(family, order, grid_size):
            self.assertIsNot(rescore.packet_source.metrics, original_metrics)
            self.assertEqual(rescore.packet_source.CYCLES, (1,))
            raise RuntimeError("injected runner failure")

        with self.assertRaisesRegex(RuntimeError, "injected runner failure"):
            rescore._run_source(
                "SEM", 4, 17, (1,), 0.1, runner=failing_runner
            )

        self.assertIs(rescore.packet_source.metrics, original_metrics)
        self.assertEqual(rescore.packet_source.CYCLES, original_cycles)


class ParallelCheckpointTests(unittest.TestCase):
    def test_resume_rejects_nonfinite_nodal_checkpoint_line(self):
        record = valid_record("E")
        config = record["config"]
        run = record["runs"]["E4_N17"]
        run["cycle1"]["full_rel_l2"] = float("nan")
        entry = {
            "key": "E4_N17",
            "config_sha256": rescore.config_sha256(config),
            "source_sha256": rescore.hashlib.sha256(
                rescore.SOURCE_DRIVER.read_bytes()
            ).hexdigest(),
            "run": run,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "packet.partial.jsonl"
            path.write_text(json.dumps(entry, allow_nan=True) + "\n")

            loaded = rescore.load_checkpoint(path, config)

        self.assertEqual(loaded, {})

    def test_checkpoint_resume_tolerates_torn_and_rejects_invalid_lines(self):
        record = valid_record()
        config = record["config"]
        key = "BMF4_N17"
        run = record["runs"][key]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "packet.partial.jsonl"
            rescore.append_checkpoint(path, key, run, config)
            forged = rescore.checkpoint_entry(key, run, config)
            forged["source_sha256"] = "0" * 64
            with path.open("a") as handle:
                handle.write(json.dumps(forged) + "\n")
                handle.write('{"key":"torn')

            loaded = rescore.load_checkpoint(path, config)

        self.assertEqual(loaded, {key: run})

    def test_append_checkpoint_rejects_invalid_run_before_writing(self):
        record = valid_record()
        config = record["config"]
        key = "BMF4_N17"
        invalid = deepcopy(record["runs"][key])
        invalid["cycle1"]["full_rel_l2"] *= 2.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "packet.partial.jsonl"

            with self.assertRaises((AssertionError, ValueError, KeyError)):
                rescore.append_checkpoint(path, key, invalid, config)

            self.assertFalse(path.exists())

    def test_two_process_workers_assemble_deterministically_and_resume(self):
        config = rescore.build_config(
            families=("BMF", "SEM"), orders=(4,), grids=(17,), cycles=(1,),
            strip_width=0.1, smoke=True
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "packet.partial.jsonl"
            runs = rescore.execute_cases(
                config,
                workers=2,
                checkpoint_path=checkpoint,
                worker_fn=synthetic_process_worker,
            )
            resumed = rescore.execute_cases(
                config,
                workers=2,
                checkpoint_path=checkpoint,
                worker_fn=worker_must_not_run,
            )

        self.assertEqual(list(runs), ["BMF4_N17", "SEM4_N17"])
        self.assertEqual(resumed, runs)

if __name__ == "__main__":
    unittest.main()
