from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import verify_wall_order_artifacts as verify
import run_all_tests


class PersistedWallOrderArtifactVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.canonical = verify.canonical_artifact_bytes()

    def _write_canonical(self, directory: Path) -> None:
        for name, payload in self.canonical.items():
            (directory / name).write_bytes(payload)

    def test_canonical_artifacts_pass_schema_semantics_and_byte_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_canonical(directory)
            self.assertEqual(verify.verify_artifact_directory(directory), [])

    def test_default_bounded_suite_invokes_persisted_artifact_verifier(self) -> None:
        commands = {command.name: command for command in run_all_tests.default_commands()}
        self.assertIn("wall-order-artifacts", commands)
        self.assertTrue(any(
            part.endswith("verify_wall_order_artifacts.py")
            for part in commands["wall-order-artifacts"].argv
        ))

    def test_numeric_and_fraction_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_canonical(directory)
            wall_path = directory / "wall_order_certificate_summary.json"
            data = json.loads(wall_path.read_text(encoding="utf-8"))
            key = sorted(data["operator_identities"])[0]
            data["operator_identities"][key]["identity_residual_inf"] = 1.0
            exact_key = sorted(data["exact_endpoint_moments"])[0]
            data["exact_endpoint_moments"][exact_key]["certificates"][0]["endpoints"]["left"][
                "exact_rational_numerator"
            ] = "1"
            wall_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            errors = verify.verify_artifact_directory(directory)
            self.assertTrue(any("identity" in error.lower() for error in errors), errors)
            self.assertTrue(any("fraction" in error.lower() for error in errors), errors)

    def test_byte_only_staleness_and_csv_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_canonical(directory)
            json_path = directory / "formal_order_summary.json"
            json_path.write_bytes(json_path.read_bytes() + b"\n")
            csv_path = directory / "formal_order_table.csv"
            csv_path.write_text(
                csv_path.read_text(encoding="utf-8").replace(
                    "pre_roundoff_fit_order", "pre_roundoff_fit_ordex", 1
                ),
                encoding="utf-8",
            )
            errors = verify.verify_artifact_directory(directory)
            self.assertTrue(any("byte" in error.lower() for error in errors), errors)
            self.assertTrue(any("csv" in error.lower() for error in errors), errors)

    def test_cli_integrity_checks_remain_active_under_python_O(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_canonical(directory)
            path = directory / "closure_moment_ladder_summary.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["rows"][0]["moments"][0]["within_backward_tolerance"] = False
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    "-O",
                    str(Path(__file__).with_name("verify_wall_order_artifacts.py")),
                    "--artifact-dir",
                    str(directory),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("within_backward_tolerance", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
