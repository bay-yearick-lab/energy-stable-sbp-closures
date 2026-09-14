#!/usr/bin/env python3
"""Run the repository's bounded verification suite.

The default suite is intentionally a check suite, not a full paper rebuild:
it runs unit tests and small algebraic smoke checks under a per-command
timeout. Paper figure generation remains opt-in because several manuscript
drivers are production experiments rather than quick tests.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TIMEOUT_S = 600


@dataclass(frozen=True)
class TestCommand:
    name: str
    argv: tuple[str, ...]
    description: str
    generated_paths: tuple[Path, ...] = ()
    # Raw production records (> 2 MB) kept outside the repository; the
    # command runs when they are placed in code/results/.
    requires: tuple[Path, ...] = ()


def default_commands() -> list[TestCommand]:
    py = sys.executable
    return [
        TestCommand(
            name="unit",
            argv=(py, "-m", "unittest", "discover", "-s", "code", "-p", "test_*.py"),
            description=(
                "unit tests for spline operators, heat core, acoustic core, "
                "the all-degree wall-rank controls, and the universal "
                "wall-transfer certificates"
            ),
        ),
        TestCommand(
            name="bmf-staggered",
            argv=(py, "code/check_bmf_staggered_complex.py"),
            description="BMF staggered SBP and flux-divergence smoke check",
            generated_paths=(Path("code/results/bmf_staggered_complex_check.json"),),
        ),
        TestCommand(
            name="formal-order",
            argv=(py, "code/check_formal_order.py"),
            description="bounded polynomial exactness and formal-order summary",
            generated_paths=(
                Path("code/results/formal_order_summary.json"),
                Path("code/results/formal_order_table.csv"),
            ),
        ),
        TestCommand(
            name="wall-order-artifacts",
            argv=(py, "code/verify_wall_order_artifacts.py"),
            description=(
                "independent schema, arithmetic, classification, and staleness "
                "verification for persisted wall-order artifacts"
            ),
        ),
        TestCommand(
            name="offgrid-sem-geometry-unit",
            argv=(py, "code/test_offgrid_sem_geometry_frontier.py"),
            description="native spline and hp-SEM matrix, polynomial, stress, and orientation controls",
        ),
        TestCommand(
            name="offgrid-sem-geometry-artifacts",
            argv=(py, "code/verify_offgrid_sem_geometry_frontier.py"),
            description="independent raw, summary, frontier, promotion, and archived-SEM verification",
            requires=(Path("code/results/offgrid_sem_geometry_frontier_raw.json"),),
        ),
        TestCommand(
            name="symmetric-hp-wall-artifacts",
            argv=(py, "code/verify_symmetric_hp_wall_frontier.py"),
            description="symmetric design, selector, held-out, classwise, and frontier reconstruction",
            requires=(Path("code/results/symmetric_hp_wall_frontier_design.json"),),
        ),
        TestCommand(
            name="symmetric-hp-wall-normalization",
            argv=(py, "code/audit_symmetric_hp_wall_losses.py"),
            description="wall-exception, absolute-error, normalization-floor, and selector audit",
            generated_paths=(Path("code/results/symmetric_hp_wall_frontier_normalization_audit.json"),),
            requires=(Path("code/results/symmetric_hp_wall_frontier_design.json"),),
        ),
        TestCommand(
            name="high-degree-high-re-flow",
            argv=(py, "code/high_degree_high_re_compatible_flow.py", "verify-all"),
            description="source-bound H1-H5 degree, Reynolds, invariant, lineage, and resource checks",
            requires=(
                Path("code/results/high_re_smooth_cavity_comparison.json"),
                Path("code/results/high_re_matched_accuracy_timing.json"),
            ),
        ),
        TestCommand(
            name="analytic-wake-quick",
            argv=(py, "code/s6_analytic_wake.py", "--quick", "--no-figures"),
            description="bounded unforced incompressible Navier-Stokes wake smoke check",
            generated_paths=(Path("code/results/analytic_wake_summary.json"),),
        ),
    ]


def optional_commands(include_paper_build: bool) -> list[TestCommand]:
    commands: list[TestCommand] = []
    if include_paper_build:
        commands.append(
            TestCommand(
                name="paper-build",
                argv=("./build_paper.sh",),
                description="LaTeX build through the repository cleanup script",
            )
        )
    return commands


def snapshot_paths(paths: tuple[Path, ...]) -> dict[Path, bytes | None]:
    snapshot: dict[Path, bytes | None] = {}
    for path in paths:
        full_path = ROOT / path
        snapshot[path] = full_path.read_bytes() if full_path.exists() else None
    return snapshot


def restore_paths(snapshot: dict[Path, bytes | None]) -> None:
    for path, content in snapshot.items():
        full_path = ROOT / path
        if content is None:
            if full_path.exists():
                full_path.unlink()
            continue
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(content)


def run_command(command: TestCommand, timeout_s: int) -> tuple[bool, float]:
    print(f"\n== {command.name}: {command.description} ==")
    missing = [str(path) for path in command.requires if not (ROOT / path).exists()]
    if missing:
        print(f"SKIP {command.name}: needs {', '.join(missing)}")
        return True, 0.0
    print("$ " + " ".join(command.argv))
    t0 = time.monotonic()
    snapshot = snapshot_paths(command.generated_paths)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            command.argv,
            cwd=ROOT,
            env=env,
            timeout=timeout_s,
            text=True,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        restore_paths(snapshot)
        print(f"TIMEOUT after {elapsed:.1f}s: {command.name} exceeded {timeout_s}s")
        return False, elapsed
    restore_paths(snapshot)
    elapsed = time.monotonic() - t0
    if proc.returncode == 0:
        print(f"PASS {command.name} in {elapsed:.1f}s")
        return True, elapsed
    print(f"FAIL {command.name} in {elapsed:.1f}s with exit code {proc.returncode}")
    return False, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        help=f"Per-command timeout in seconds. Default: {DEFAULT_TIMEOUT_S}.",
    )
    parser.add_argument(
        "--include-paper-build",
        action="store_true",
        help="Also run ./build_paper.sh under the same per-command timeout.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List commands without running them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    commands = default_commands() + optional_commands(args.include_paper_build)
    if args.list:
        for command in commands:
            print(f"{command.name:>14s}: {' '.join(command.argv)}")
        return 0

    print(f"Repository: {ROOT}")
    print(f"Per-command timeout: {args.timeout}s")
    print(f"Commands: {', '.join(command.name for command in commands)}")

    failures: list[str] = []
    total_start = time.monotonic()
    timings: list[tuple[str, float]] = []
    for command in commands:
        ok, elapsed = run_command(command, args.timeout)
        timings.append((command.name, elapsed))
        if not ok:
            failures.append(command.name)
    total_elapsed = time.monotonic() - total_start

    print("\n== summary ==")
    for name, elapsed in timings:
        print(f"{name:>14s}: {elapsed:7.1f}s")
    print(f"{'total':>14s}: {total_elapsed:7.1f}s")

    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print("All requested checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
