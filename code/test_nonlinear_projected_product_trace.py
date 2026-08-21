from __future__ import annotations

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from s6_nonlinear_product_audit import projected_product_trace_leak_bmg


class ProjectedProductTraceTests(unittest.TestCase):
    def test_zero_advecting_trace_does_not_zero_projected_boundary_work(self) -> None:
        expected_leak = {
            9: 2.0e-5,
            17: 3.0e-7,
            33: 5.0e-9,
        }
        for npts, lower_bound in expected_leak.items():
            with self.subTest(npts=npts):
                record = projected_product_trace_leak_bmg(npts, 4)
                self.assertEqual(record["advecting_left_trace"], 0.0)
                self.assertLess(abs(record["advecting_right_trace"]), 2.0e-31)
                self.assertGreater(abs(record["state_left_trace"]), 0.9)
                self.assertGreater(abs(record["state_right_trace"]), 0.9)
                self.assertGreater(
                    max(
                        abs(record["projected_product_left_trace"]),
                        abs(record["projected_product_right_trace"]),
                    ),
                    lower_bound,
                )
                self.assertGreater(abs(record["split_power"]), lower_bound / 4.0)
                self.assertLess(record["relative_boundary_identity_residual"], 1.0e-10)


if __name__ == "__main__":
    unittest.main()
