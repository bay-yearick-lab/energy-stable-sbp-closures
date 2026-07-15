#!/usr/bin/env python3
"""Run and verify the released one-dimensional operator-cost benchmark.

The C++ benchmark measures CPU process time, so unrelated system load does not
enter the reported assembly, factorization, solve, or banded-application
times.  This wrapper binds those numbers to the exact source and executable,
records the execution environment, and validates every derived storage value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DRIVER = Path(__file__).resolve()
SOURCE = ROOT / "code" / "cavity" / "apps" / "opbench.cpp"
BINARY = ROOT / "code" / "cavity" / "bin" / "opbench"
DEFAULT_OUTPUT = ROOT / "code" / "results" / "operator_cost_scaling.json"
SCHEMA = "operator-cost-benchmark-v2"
SOURCE_FILES = (
    SOURCE,
    ROOT / "code" / "cavity" / "include" / "wall_ns2d.h",
    ROOT / "code" / "cavity" / "include" / "wall2d.h",
    ROOT / "code" / "cavity" / "include" / "bspline.h",
    ROOT / "code" / "cavity" / "include" / "linalg.h",
)
BUILD_COMMAND = (
    "g++", "-O3", "-std=c++17", "-fopenmp", "-Icode/cavity/include",
    "code/cavity/apps/opbench.cpp", "-o", "code/cavity/bin/opbench",
)
DEGREES = (4, 6, 8)
GRID_SIZES = (65, 129, 257, 513, 1025, 2049, 4097)
THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _compiler_version() -> str:
    completed = subprocess.run(
        [BUILD_COMMAND[0], "--version"], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    first_line = completed.stdout.splitlines()[0].strip()
    _require(bool(first_line), "compiler version is empty")
    return first_line


def compile_benchmark() -> None:
    for path in SOURCE_FILES:
        _require(path.is_file(), f"missing benchmark source: {path}")
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(BUILD_COMMAND, cwd=ROOT, check=True)
    _require(BINARY.is_file() and os.access(BINARY, os.X_OK), "benchmark build did not produce an executable")


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def execution_stamp() -> dict[str, Any]:
    return {
        "cpu": _cpu_name(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "threads": {name: os.environ.get(name) for name in THREAD_ENV_NAMES},
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "load_at_start": os.getloadavg()[0],
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_positive(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} is not numeric")
    number = float(value)
    _require(math.isfinite(number) and number > 0.0, f"{label} is not finite and positive")
    return number


def parse_raw_output(payload: str) -> list[dict[str, Any]]:
    raw = json.loads(payload, parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON constant: {value}")
    ))
    _require(set(raw) == {"oned", "twod", "threed"}, "unexpected opbench output schema")
    _require(raw["twod"] == [] and raw["threed"] == [], "one-dimensional run returned other benchmark modes")
    rows = raw["oned"]
    _require(isinstance(rows, list), "one-dimensional result is not a list")
    return rows


def build_record(rows: list[dict[str, Any]], stamp: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "protocol": {
            "mode": "1d",
            "timing_metric": "CPU process time",
            "reported_statistic": "per-operation mean over the repetitions fixed in opbench.cpp",
            "thread_policy": "single-thread execution",
            "load_policy": "CPU process time is retained under shared-machine load; load is recorded but is not a wall-time validity gate",
        },
        "hardware": stamp,
        "provenance": {
            "driver": "code/operator_cost_benchmark.py",
            "driver_sha256": _sha256(DRIVER),
            "sources": {_relative(path): _sha256(path) for path in SOURCE_FILES},
            "compiler": {
                "executable": BUILD_COMMAND[0],
                "version": _compiler_version(),
                "command": list(BUILD_COMMAND),
            },
            "executable": "code/cavity/bin/opbench",
            "executable_sha256": _sha256(BINARY),
            "command": ["code/cavity/bin/opbench", "1d"],
        },
        "oned": rows,
    }


def verify_record(record: dict[str, Any]) -> None:
    _require(isinstance(record, dict), "benchmark record is not an object")
    _require(set(record) == {"schema", "protocol", "hardware", "provenance", "oned"}, "benchmark record keys differ from the released schema")
    _require(record["schema"] == SCHEMA, "benchmark schema mismatch")
    protocol = record["protocol"]
    _require(protocol == {
        "mode": "1d",
        "timing_metric": "CPU process time",
        "reported_statistic": "per-operation mean over the repetitions fixed in opbench.cpp",
        "thread_policy": "single-thread execution",
        "load_policy": "CPU process time is retained under shared-machine load; load is recorded but is not a wall-time validity gate",
    }, "benchmark protocol mismatch")

    hardware = record["hardware"]
    _require(isinstance(hardware, dict) and set(hardware) == {
        "cpu", "platform", "python", "threads", "nice", "load_at_start"
    }, "benchmark hardware stamp is incomplete")
    _require(all(isinstance(hardware[key], str) and hardware[key] for key in ("cpu", "platform", "python")), "benchmark hardware identity is incomplete")
    _require(hardware["nice"] == 19, "benchmark was not run at nice 19")
    load = hardware["load_at_start"]
    _require(
        isinstance(load, (int, float)) and not isinstance(load, bool)
        and math.isfinite(float(load)) and float(load) >= 0.0,
        "hardware.load_at_start is not finite and nonnegative",
    )
    threads = hardware["threads"]
    _require(isinstance(threads, dict) and set(threads) == set(THREAD_ENV_NAMES), "thread stamp is incomplete")
    _require(all(threads[name] == "1" for name in THREAD_ENV_NAMES), "benchmark thread environment was not pinned to one")

    provenance = record["provenance"]
    _require(set(provenance) == {
        "driver", "driver_sha256", "sources", "compiler", "executable",
        "executable_sha256", "command",
    }, "benchmark provenance schema mismatch")
    _require(provenance.get("driver") == "code/operator_cost_benchmark.py", "driver path mismatch")
    _require(provenance.get("driver_sha256") == _sha256(DRIVER), "driver hash mismatch")
    expected_sources = {_relative(path): _sha256(path) for path in SOURCE_FILES}
    _require(provenance.get("sources") == expected_sources, "transitive source hash mismatch")
    compiler = provenance.get("compiler")
    _require(isinstance(compiler, dict) and set(compiler) == {
        "executable", "version", "command"
    }, "compiler provenance is incomplete")
    _require(compiler["executable"] == BUILD_COMMAND[0], "compiler executable mismatch")
    _require(isinstance(compiler["version"], str) and compiler["version"], "compiler version is empty")
    _require(compiler["command"] == list(BUILD_COMMAND), "compiler command mismatch")
    _require(provenance.get("executable") == "code/cavity/bin/opbench", "executable path mismatch")
    _require(provenance.get("command") == ["code/cavity/bin/opbench", "1d"], "command mismatch")
    executable_hash = provenance.get("executable_sha256")
    _require(isinstance(executable_hash, str) and len(executable_hash) == 64, "executable hash is malformed")
    if BINARY.is_file():
        _require(executable_hash == _sha256(BINARY), "executable hash mismatch")

    rows = record["oned"]
    _require(isinstance(rows, list), "oned is not a list")
    expected = {(degree, npts) for degree in DEGREES for npts in GRID_SIZES}
    actual: set[tuple[int, int]] = set()
    required = {"p", "N", "assemble_s", "factor_s", "msolve_s", "kapply_s", "band_bytes", "dense_bytes"}
    for index, row in enumerate(rows):
        _require(isinstance(row, dict) and set(row) == required, f"oned[{index}] schema mismatch")
        degree, npts = row["p"], row["N"]
        _require(isinstance(degree, int) and isinstance(npts, int), f"oned[{index}] grid metadata is not integer")
        key = (degree, npts)
        _require(key not in actual, f"duplicate one-dimensional case {key}")
        actual.add(key)
        for metric in ("assemble_s", "factor_s", "msolve_s", "kapply_s"):
            _finite_positive(row[metric], f"oned[{index}].{metric}")
        _require(row["band_bytes"] == (2 * degree + 1) * npts * 8 * 3, f"oned[{index}] band storage arithmetic mismatch")
        _require(row["dense_bytes"] == npts * npts * 8, f"oned[{index}] dense storage arithmetic mismatch")
    _require(actual == expected, f"one-dimensional case coverage mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")


def run_benchmark() -> dict[str, Any]:
    compile_benchmark()
    stamp = execution_stamp()
    _require(stamp["nice"] == 19, "run the benchmark under nice -n 19")
    _require(all(stamp["threads"][name] == "1" for name in THREAD_ENV_NAMES), "set every supported thread environment variable to 1")
    completed = subprocess.run(
        [str(BINARY), "1d"], cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    record = build_record(parse_raw_output(completed.stdout), stamp)
    verify_record(record)
    return record


def load_json_strict(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(), parse_constant=lambda token: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON constant: {token}")
    ))
    _require(isinstance(value, dict), "benchmark JSON root is not an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.verify is not None:
        verify_record(load_json_strict(args.verify))
        print(f"operator-cost benchmark verified: {args.verify}")
        return 0
    record = run_benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(f"operator-cost benchmark wrote and verified: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
