from __future__ import annotations

import copy
import shutil
import unittest

import operator_cost_benchmark as benchmark


def valid_record() -> dict:
    rows = []
    for degree in benchmark.DEGREES:
        for npts in benchmark.GRID_SIZES:
            rows.append({
                "p": degree,
                "N": npts,
                "assemble_s": 1.0,
                "factor_s": 0.1,
                "msolve_s": 0.01,
                "kapply_s": 0.001,
                "band_bytes": (2 * degree + 1) * npts * 8 * 3,
                "dense_bytes": npts * npts * 8,
            })
    stamp = {
        "cpu": "test cpu",
        "platform": "test platform",
        "python": "3.test",
        "threads": {name: "1" for name in benchmark.THREAD_ENV_NAMES},
        "nice": 19,
        "load_at_start": 1.0,
    }
    return benchmark.build_record(rows, stamp)


@unittest.skipUnless(shutil.which("g++"), "g++ is required to stamp the C++ build")
class OperatorCostBenchmarkTests(unittest.TestCase):
    def test_released_schema_accepts_complete_controlled_record(self):
        benchmark.verify_record(valid_record())

    def test_missing_case_is_rejected(self):
        forged = valid_record()
        forged["oned"].pop()
        with self.assertRaisesRegex(ValueError, "coverage"):
            benchmark.verify_record(forged)

    def test_storage_arithmetic_tamper_is_rejected(self):
        forged = copy.deepcopy(valid_record())
        forged["oned"][0]["band_bytes"] += 8
        with self.assertRaisesRegex(ValueError, "storage arithmetic"):
            benchmark.verify_record(forged)

    def test_uncontrolled_thread_environment_is_rejected(self):
        forged = copy.deepcopy(valid_record())
        forged["hardware"]["threads"]["OMP_NUM_THREADS"] = "8"
        with self.assertRaisesRegex(ValueError, "not pinned"):
            benchmark.verify_record(forged)

    def test_transitive_source_tamper_is_rejected(self):
        forged = copy.deepcopy(valid_record())
        forged["provenance"]["sources"]["code/cavity/include/linalg.h"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            benchmark.verify_record(forged)

    def test_build_recipe_tamper_is_rejected(self):
        forged = copy.deepcopy(valid_record())
        forged["provenance"]["compiler"]["command"][1] = "-O0"
        with self.assertRaisesRegex(ValueError, "compiler command mismatch"):
            benchmark.verify_record(forged)


if __name__ == "__main__":
    unittest.main()
