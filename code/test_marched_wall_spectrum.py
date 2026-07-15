from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import s4_marched_wall_spectrum as spectrum


class MarchedWallSpectrumTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summary_path = Path(__file__).resolve().parent / "results" / "staggered_wallmode_summary.json"

    def test_loads_only_operators_that_are_actually_marched(self) -> None:
        data = spectrum.load_marched_curves(self.summary_path, q=6)
        self.assertEqual(
            set(data),
            {"E", "C", "SBP", "DRP", "BMF-staggered", "BMG"},
        )
        self.assertNotIn("BMF-reconstruction", data)
        for rows in data.values():
            self.assertGreater(len(rows), 20)
            self.assertTrue(all({"ppw", "rho", "eta", "total"} <= set(row) for row in rows))

    def test_rejects_a_summary_without_the_marched_bmf_channel(self) -> None:
        source = json.loads(self.summary_path.read_text())
        del source["second_order"]["q6"]["BMF-staggered"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "BMF-staggered"):
                spectrum.load_marched_curves(path, q=6)

    def test_figure_contract_has_three_nonredundant_panels(self) -> None:
        self.assertEqual(
            spectrum.PANELS,
            ("signed_bias", "contamination", "total_error"),
        )

    def test_norm_sensitivity_uses_the_marched_pressure_operator(self) -> None:
        source = json.loads(self.summary_path.read_text())
        self.assertEqual(
            source["sensitivity"]["q6"]["operator"],
            "BMG-marched-pressure",
        )


if __name__ == "__main__":
    unittest.main()
