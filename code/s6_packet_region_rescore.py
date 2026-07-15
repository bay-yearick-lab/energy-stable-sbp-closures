#!/usr/bin/env python3
"""Same-run physical-region diagnostics for the localized wall packet.

The numerical marches, exact image solution, family labels, formal orders,
grids, cycle times, Courant number, and Runge--Kutta protocol are delegated to
``s6_packet_reflection.py``.  This driver changes only the scoring callback:
each returned field is scored on the full domain, the physical strip
``(x <= strip_width) | (x >= 1 - strip_width)``, and its exact complement.

All three errors use the full-domain exact norm.  This preserves the original
packet-study convention and avoids normalization by a Gaussian tail after the
packet has left the wall strip.  Raw regional squared numerators and exact
denominators are retained so the partition can be checked independently.

The default output is a new record, ``packet_region_rescore.json``.  Existing
files are never overwritten.  ``--smoke --workers 2`` runs BMF4 and SEM4 on
17 points for one cycle; ``--verify`` checks a record without marching.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))

import s6_packet_reflection as packet_source
from project_paths import RESULTSDIR, ensure_generated_dirs

SCHEMA = "packet-region-rescore-v2"
SCHEMA_VERSION = 2
SOURCE_DRIVER = _CODE / "s6_packet_reflection.py"
DRIVER = Path(__file__).resolve()
DEFAULT_OUTPUT = RESULTSDIR / "packet_region_rescore.json"
ALL_FAMILIES = tuple(packet_source.FAMILIES) + ("SEM",)
STABLE_FAMILIES = frozenset(("BMF", "BMG", "SBP", "SEM"))
REGION_RULES = {
    "wall_strip": "(x <= strip_width) | (x >= 1 - strip_width)",
    "interior": "complement of wall_strip",
}
METRIC_KEYS = frozenset(
    (
        "full_rel_l2",
        "wall_strip_rel_l2",
        "interior_rel_l2",
        "regions",
        "phase_shift",
        "amp_ratio",
        "waveform_distortion",
        "leakage",
        "status",
        "grid_provenance",
    )
)
UNSTABLE_METRIC_KEYS = frozenset(("status", "reason", "grid_provenance"))
GRID_PROVENANCE_KEYS = frozenset(
    (
        "point_count",
        "wall_point_count",
        "interior_point_count",
        "coordinate_sha256",
        "wall_mask_sha256",
        "interior_mask_sha256",
    )
)
REGION_KEYS = frozenset(
    (
        "mask_rule",
        "point_count",
        "numerator_sq",
        "denominator_sq",
        "full_reference_rel_l2",
        "normalization_denominator_sq",
    )
)


def smoke_selection() -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return two tiny source-admissible BMF and SEM process-smoke cases."""
    return ("BMF", "SEM"), (4,), (17,), (1,)


def region_masks(x: np.ndarray, strip_width: float) -> dict[str, np.ndarray]:
    """Return the full, two-wall-strip, and complementary interior masks."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a one-dimensional grid with at least two points")
    if not np.isfinite(strip_width) or not 0.0 < strip_width < 0.5:
        raise ValueError("strip_width must be finite and strictly between 0 and 0.5")
    wall = (x <= strip_width) | (x >= 1.0 - strip_width)
    interior = ~wall
    return {
        "full": np.ones(x.size, dtype=bool),
        "wall_strip": wall,
        "interior": interior,
    }


def _array_sha256(values: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    header = f"{array.dtype.str}:{array.shape}".encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def grid_provenance(x: np.ndarray, strip_width: float) -> dict[str, Any]:
    """Bind a metric to its coordinates and exact two-wall region masks."""
    x = np.asarray(x, dtype=float)
    masks = region_masks(x, strip_width)
    return {
        "point_count": int(x.size),
        "wall_point_count": int(np.count_nonzero(masks["wall_strip"])),
        "interior_point_count": int(np.count_nonzero(masks["interior"])),
        "coordinate_sha256": _array_sha256(x, "<f8"),
        "wall_mask_sha256": _array_sha256(masks["wall_strip"], "u1"),
        "interior_mask_sha256": _array_sha256(masks["interior"], "u1"),
    }


@lru_cache(maxsize=None)
def _expected_grid_tuple(family: str, order: int, grid_size: int) -> tuple[float, ...]:
    if family == "SEM":
        x = packet_source.sem_grid_for(grid_size, order).x
    else:
        operator = packet_source.build_operator(
            packet_source.scheme_kind(f"{family}{order}"), order, grid_size
        )
        x = operator.x
    return tuple(float(value) for value in np.asarray(x, dtype=float))


def expected_grid(family: str, order: int, grid_size: int) -> np.ndarray:
    """Reconstruct the exact source grid implied by one configured route."""
    return np.asarray(_expected_grid_tuple(family, order, grid_size), dtype=float)


def region_l2_diagnostics(
    x: np.ndarray,
    pressure: np.ndarray,
    velocity: np.ndarray,
    pressure_exact: np.ndarray,
    velocity_exact: np.ndarray,
    strip_width: float,
) -> dict[str, Any]:
    """Compute full-normalized coupled L2 contributions on three regions."""
    x = np.asarray(x, dtype=float)
    arrays = [
        np.asarray(a, dtype=float)
        for a in (pressure, velocity, pressure_exact, velocity_exact)
    ]
    if any(a.shape != x.shape for a in arrays):
        raise ValueError("all numerical and exact fields must match the grid shape")
    p, u, pref, uref = arrays
    masks = region_masks(x, strip_width)
    weights = packet_source.nodal_trapezoidal_weights(x)
    error_density = (p - pref) ** 2 + (u - uref) ** 2
    exact_density = pref ** 2 + uref ** 2

    regions: dict[str, dict[str, Any]] = {}
    for name, mask in masks.items():
        regions[name] = {
            "mask_rule": {
                "full": "all grid points",
                "wall_strip": REGION_RULES["wall_strip"],
                "interior": REGION_RULES["interior"],
            }[name],
            "point_count": int(np.count_nonzero(mask)),
            "numerator_sq": float(np.sum(weights[mask] * error_density[mask])),
            "denominator_sq": float(np.sum(weights[mask] * exact_density[mask])),
        }

    full_denominator_sq = regions["full"]["denominator_sq"]
    scale = max(full_denominator_sq, 1.0e-300)
    for region in regions.values():
        region["full_reference_rel_l2"] = float(
            np.sqrt(region["numerator_sq"] / scale)
        )
        region["normalization_denominator_sq"] = float(full_denominator_sq)

    return {
        "full_rel_l2": regions["full"]["full_reference_rel_l2"],
        "wall_strip_rel_l2": regions["wall_strip"]["full_reference_rel_l2"],
        "interior_rel_l2": regions["interior"]["full_reference_rel_l2"],
        "regions": regions,
    }


def packet_metrics(
    x: np.ndarray,
    pressure: np.ndarray,
    velocity: np.ndarray,
    time_value: float,
    strip_width: float,
) -> dict[str, Any]:
    """Score one source-driver state without changing its march."""
    x = np.asarray(x, dtype=float)
    p = np.asarray(pressure, dtype=float)
    u = np.asarray(velocity, dtype=float)
    provenance = grid_provenance(x, strip_width)
    pref = packet_source.p_exact(x, time_value)
    uref = packet_source.u_exact(x, time_value)
    if not all(np.all(np.isfinite(value)) for value in (x, p, u, pref, uref)):
        return {
            "status": "unstable_nonfinite",
            "reason": "nonfinite numerical state",
            "grid_provenance": provenance,
        }
    out = region_l2_diagnostics(x, p, u, pref, uref, strip_width)
    shift, amplitude, distortion = packet_source.phase_amp_distortion(x, p, pref)
    out.update(
        {
            "status": "ok",
            "grid_provenance": provenance,
            "phase_shift": shift,
            "amp_ratio": amplitude,
            "waveform_distortion": distortion,
            "leakage": packet_source.leakage(x, p),
        }
    )
    try:
        _require_finite_tree(out)
    except RecordValidationError:
        return {
            "status": "unstable_nonfinite",
            "reason": "nonfinite derived metric",
            "grid_provenance": provenance,
        }
    return out


def _run_source(
    family: str,
    order: int,
    grid_size: int,
    cycles: tuple[int, ...],
    strip_width: float,
    runner=None,
) -> dict[str, Any]:
    """Call the original serial packet runner with a temporary metric callback."""
    original_metrics = packet_source.metrics
    original_cycles = packet_source.CYCLES
    packet_source.metrics = lambda x, p, u, t: packet_metrics(
        x, p, u, t, strip_width
    )
    packet_source.CYCLES = cycles
    try:
        if runner is not None:
            return runner(family, order, grid_size)
        if family == "SEM":
            return packet_source.run_sem(order, grid_size)
        return packet_source.run_family(family, order, grid_size)
    finally:
        packet_source.metrics = original_metrics
        packet_source.CYCLES = original_cycles


def build_config(
    *,
    families: Iterable[str],
    orders: Iterable[int],
    grids: Iterable[int],
    cycles: Iterable[int],
    strip_width: float,
    smoke: bool,
) -> dict[str, Any]:
    """Validate and serialize the selected source-study parameter block."""
    families = tuple(str(value) for value in families)
    orders = tuple(int(value) for value in orders)
    grids = tuple(int(value) for value in grids)
    cycles = tuple(int(value) for value in cycles)
    region_masks(np.array([0.0, 1.0]), strip_width)
    unknown = sorted(set(families) - set(ALL_FAMILIES))
    if unknown:
        raise ValueError(f"unknown packet families: {unknown}")
    if not families or not orders or not grids or not cycles:
        raise ValueError("families, orders, grids, and cycles must be nonempty")
    for name, values in (
        ("families", families), ("orders", orders), ("grids", grids), ("cycles", cycles)
    ):
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must not contain duplicates")
    invalid_orders = sorted(set(orders) - set(packet_source.ORDERS))
    if invalid_orders:
        raise ValueError(f"orders must preserve the source selection {packet_source.ORDERS}: {invalid_orders}")
    for family in families:
        for order in orders:
            for grid in grids:
                if family == "SEM":
                    if grid <= order or (grid - 1) % order != 0:
                        raise ValueError(
                            f"SEM{order}_N{grid} requires N > order and order | (N - 1)"
                        )
                elif grid < 13:
                    raise ValueError(
                        f"{family}{order}_N{grid} cannot retain the source six-point interior strip"
                    )
    if any(cycle <= 0 for cycle in cycles):
        raise ValueError("cycles must be positive")
    return {
        "families": list(families),
        "orders": list(orders),
        "grids": list(grids),
        "cycles": list(cycles),
        "strip_width": float(strip_width),
        "smoke": bool(smoke),
        "packet": {
            "x0": packet_source.X0,
            "sigma": packet_source.SIGMA,
            "k0_over_pi": packet_source.K0 / np.pi,
            "images_per_side": packet_source.NIMG,
        },
        "cfl": packet_source.CFL,
        "impact_time": packet_source.T_IMPACT,
    }


def record_envelope(
    config: dict[str, Any], runs: dict[str, Any], exact_selftest: dict[str, float]
) -> dict[str, Any]:
    """Wrap runs with the immutable schema and source provenance."""
    source_hash = hashlib.sha256(SOURCE_DRIVER.read_bytes()).hexdigest()
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "driver": "code/s6_packet_region_rescore.py",
            "driver_sha256": hashlib.sha256(DRIVER.read_bytes()).hexdigest(),
            "source_driver": "code/s6_packet_reflection.py",
            "source_sha256": source_hash,
            "march_reuse": ["run_family", "run_sem"],
            "exact_image_reuse": ["p_exact", "u_exact"],
            "region_rules": dict(REGION_RULES),
            "relative_l2_normalization": "full-domain exact squared norm",
        },
        "config": config,
        "exact_selftest": exact_selftest,
        "runs": runs,
    }


def exact_recurrence_selftest() -> dict[str, Any]:
    """Recompute the source image-solution recurrence on its fixed audit grid."""
    x = np.linspace(0.0, 1.0, 4001)
    return {
        "grid_point_count": int(x.size),
        "coordinate_sha256": _array_sha256(x, "<f8"),
        "p_t2_vs_t0": float(
            np.max(np.abs(packet_source.p_exact(x, 2.0) - packet_source.p_exact(x, 0.0)))
        ),
        "u_t2_vs_t0": float(
            np.max(np.abs(packet_source.u_exact(x, 2.0) - packet_source.u_exact(x, 0.0)))
        ),
    }


def run_all(
    config: dict[str, Any],
    *,
    workers: int = 1,
    checkpoint_path: Path | None = None,
    worker_fn=None,
) -> dict[str, Any]:
    """Execute or resume source marches and assemble their deterministic record."""
    exact_selftest = exact_recurrence_selftest()
    runs = execute_cases(
        config,
        workers=workers,
        checkpoint_path=checkpoint_path,
        worker_fn=worker_fn or _execute_case,
    )
    return record_envelope(config, runs, exact_selftest)


def _close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=5.0e-13, atol=1.0e-300))


class RecordValidationError(ValueError):
    """Raised when an evidence record violates its finite schema or invariants."""


def _require(condition: bool, *details: Any) -> None:
    """Raise an optimization-proof validation error when a condition fails."""
    if not condition:
        message = "; ".join(str(detail) for detail in details) or "record validation failed"
        raise RecordValidationError(message)


def _require_finite_tree(value: Any, path: str = "record") -> None:
    """Reject every nonfinite numeric leaf, independent of scheme family."""
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise RecordValidationError(f"nonfinite numeric value at {path}")


def expected_cases(config: dict[str, Any]) -> dict[str, str]:
    """Return the deterministic run-key to family map implied by a config."""
    return {
        f"{family}{order}_N{grid_size}": family
        for order in config["orders"]
        for grid_size in config["grids"]
        for family in config["families"]
    }


def case_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Serialize cases in the same deterministic order used by final records."""
    return [
        {
            "key": f"{family}{order}_N{grid_size}",
            "family": family,
            "order": order,
            "grid_size": grid_size,
            "cycles": list(config["cycles"]),
            "strip_width": config["strip_width"],
        }
        for order in config["orders"]
        for grid_size in config["grids"]
        for family in config["families"]
    ]


def _execute_case(spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Process-pool entry point for one source-isolated packet march."""
    run = _run_source(
        spec["family"],
        spec["order"],
        spec["grid_size"],
        tuple(spec["cycles"]),
        spec["strip_width"],
    )
    return spec["key"], run


def _verify_grid_provenance(
    key: str,
    tag: str,
    stored: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    _require(set(stored) == GRID_PROVENANCE_KEYS, key, tag, "grid provenance keys")
    _require(stored == expected, key, tag, "grid provenance mismatch")


def _verify_metric(
    key: str,
    tag: str,
    metric: dict[str, Any],
    stable: bool,
    expected_provenance: dict[str, Any],
) -> None:
    _verify_grid_provenance(
        key, tag, metric.get("grid_provenance", {}), expected_provenance
    )
    if metric.get("status") == "unstable_nonfinite":
        _require(not stable, key, tag, "stable family marked unstable")
        _require(set(metric) == UNSTABLE_METRIC_KEYS, key, tag, "unstable metric keys")
        _require(
            metric["reason"] in {
                "nonfinite numerical state",
                "nonfinite derived metric",
            },
            key,
            tag,
            "unstable reason",
        )
        return
    _require(set(metric) == METRIC_KEYS, key, tag, "metric keys", set(metric))
    _require(metric["status"] == "ok", key, tag, "metric status")
    scalar_names = (
        "full_rel_l2",
        "wall_strip_rel_l2",
        "interior_rel_l2",
        "phase_shift",
        "amp_ratio",
        "waveform_distortion",
        "leakage",
    )
    if stable:
        _require(
            all(np.isfinite(metric[name]) for name in scalar_names),
            key,
            tag,
            "nonfinite stable metric",
        )
    regions = metric["regions"]
    _require(
        set(regions) == {"full", "wall_strip", "interior"},
        key,
        tag,
        "region keys",
    )
    expected_rules = {
        "full": "all grid points",
        "wall_strip": REGION_RULES["wall_strip"],
        "interior": REGION_RULES["interior"],
    }
    for name, region in regions.items():
        _require(set(region) == REGION_KEYS, key, tag, name, "raw keys")
        _require(
            region["mask_rule"] == expected_rules[name], key, tag, name, "mask rule"
        )
        _require(
            isinstance(region["point_count"], int) and region["point_count"] >= 0,
            key,
            tag,
            name,
            "point count",
        )
        raw = (
            region["numerator_sq"],
            region["denominator_sq"],
            region["full_reference_rel_l2"],
            region["normalization_denominator_sq"],
        )
        if stable:
            _require(
                all(np.isfinite(value) for value in raw),
                key,
                tag,
                name,
                "nonfinite stable raw field",
            )
        for value_name in ("numerator_sq", "denominator_sq"):
            value = region[value_name]
            if np.isfinite(value):
                _require(value >= 0.0, key, tag, name, value_name, value)

    full = regions["full"]
    wall = regions["wall_strip"]
    interior = regions["interior"]
    _require(
        full["point_count"] == expected_provenance["point_count"],
        key,
        tag,
        "full point count",
    )
    _require(
        wall["point_count"] == expected_provenance["wall_point_count"],
        key,
        tag,
        "wall point count",
    )
    _require(
        interior["point_count"] == expected_provenance["interior_point_count"],
        key,
        tag,
        "interior point count",
    )
    _require(
        full["point_count"] == wall["point_count"] + interior["point_count"],
        key,
        tag,
        "point-count partition",
    )
    for field in ("numerator_sq", "denominator_sq"):
        values = (full[field], wall[field], interior[field])
        if all(np.isfinite(value) for value in values):
            _require(
                _close(values[0], values[1] + values[2]),
                key,
                tag,
                field,
                values,
            )

    full_denominator_sq = full["denominator_sq"]
    if np.isfinite(full_denominator_sq):
        _require(full_denominator_sq > 0.0, key, tag, "full denominator")
        scale = max(full_denominator_sq, 1.0e-300)
        for name, region in regions.items():
            _require(
                _close(region["normalization_denominator_sq"], full_denominator_sq),
                key,
                tag,
                name,
                "normalization denominator",
            )
            if np.isfinite(region["numerator_sq"]):
                expected_rel = float(np.sqrt(region["numerator_sq"] / scale))
                _require(
                    _close(region["full_reference_rel_l2"], expected_rel),
                    key,
                    tag,
                    name,
                    "regional relative norm",
                )
        for top_name, region_name in (
            ("full_rel_l2", "full"),
            ("wall_strip_rel_l2", "wall_strip"),
            ("interior_rel_l2", "interior"),
        ):
            if np.isfinite(metric[top_name]):
                _require(
                    _close(metric[top_name], regions[region_name]["full_reference_rel_l2"]),
                    key,
                    tag,
                    top_name,
                )


def _verify_run(key: str, family: str, run: dict[str, Any], config: dict[str, Any]) -> None:
    expected_tags = {"impact", "cpu_seconds"} | {
        f"cycle{cycle}" for cycle in config["cycles"]
    }
    _require(set(run) == expected_tags, key, "run tags", set(run), expected_tags)
    _require(
        np.isfinite(run["cpu_seconds"]) and run["cpu_seconds"] >= 0.0,
        key,
        "cpu_seconds",
    )
    stable = family in STABLE_FAMILIES
    specs = {spec["key"]: spec for spec in case_specs(config)}
    _require(key in specs, key, "missing configured case spec")
    spec = specs[key]
    _require(spec["family"] == family, key, family, "family mismatch")
    provenance = grid_provenance(
        expected_grid(family, spec["order"], spec["grid_size"]),
        spec["strip_width"],
    )
    for tag in sorted(expected_tags - {"cpu_seconds"}):
        _verify_metric(key, tag, run[tag], stable, provenance)


def config_sha256(config: dict[str, Any]) -> str:
    """Fingerprint the exact experiment selection stored in checkpoint lines."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def checkpoint_entry(
    key: str, run: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Validate and wrap one completed run for append-only persistence."""
    cases = expected_cases(config)
    _require(key in cases, key, "unexpected checkpoint key")
    _verify_run(key, cases[key], run, config)
    entry = {
        "key": key,
        "config_sha256": config_sha256(config),
        "source_sha256": hashlib.sha256(SOURCE_DRIVER.read_bytes()).hexdigest(),
        "run": run,
    }
    _require_finite_tree(entry, "checkpoint")
    return entry


def append_checkpoint(
    path: Path, key: str, run: dict[str, Any], config: dict[str, Any]
) -> None:
    """Append one prevalidated run and flush it durably before returning."""
    entry = checkpoint_entry(key, run, config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(entry, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_checkpoint(path: Path | None, config: dict[str, Any]) -> dict[str, Any]:
    """Load only complete, current-source, arithmetically valid checkpoint lines."""
    if path is None or not Path(path).exists():
        return {}
    expected = expected_cases(config)
    wanted_config_hash = config_sha256(config)
    wanted_source_hash = hashlib.sha256(SOURCE_DRIVER.read_bytes()).hexdigest()
    done: dict[str, Any] = {}
    for line in Path(path).read_text().splitlines():
        try:
            entry = json.loads(line)
            _require_finite_tree(entry, "checkpoint")
            _require(
                set(entry) == {"key", "config_sha256", "source_sha256", "run"},
                "checkpoint keys",
            )
            _require(entry["config_sha256"] == wanted_config_hash, "checkpoint config hash")
            _require(entry["source_sha256"] == wanted_source_hash, "checkpoint source hash")
            key = entry["key"]
            _require(key in expected, key, "unexpected checkpoint key")
            _verify_run(key, expected[key], entry["run"], config)
            if key in done and done[key] != entry["run"]:
                raise ValueError(f"conflicting valid checkpoint records for {key}")
            done[key] = entry["run"]
        except (RecordValidationError, json.JSONDecodeError, KeyError, TypeError):
            continue
    return done


def execute_cases(
    config: dict[str, Any],
    *,
    workers: int,
    checkpoint_path: Path | None,
    worker_fn=_execute_case,
) -> dict[str, Any]:
    """Run missing cases in processes, checkpoint in the parent, assemble in order."""
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    specs = case_specs(config)
    done = load_checkpoint(checkpoint_path, config)
    pending = [spec for spec in specs if spec["key"] not in done]

    def accept(spec: dict[str, Any], result: tuple[str, dict[str, Any]]) -> None:
        key, run = result
        _require(key == spec["key"], "worker returned wrong key", key, spec["key"])
        family = spec["family"]
        _verify_run(key, family, run, config)
        if checkpoint_path is not None:
            append_checkpoint(checkpoint_path, key, run, config)
        done[key] = run
        print(key, json.dumps(run.get("cycle1", run))[:120], flush=True)

    if workers == 1:
        for spec in pending:
            accept(spec, worker_fn(spec))
    elif pending:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as executor:
            futures = {executor.submit(worker_fn, spec): spec for spec in pending}
            for future in as_completed(futures):
                accept(futures[future], future.result())

    return {spec["key"]: done[spec["key"]] for spec in specs}


def verify_record(record: dict[str, Any]) -> None:
    """Verify experiment identity, provenance, completeness, and raw arithmetic."""
    _require_finite_tree(record)
    _require(record["schema"] == SCHEMA, "schema", record["schema"])
    _require(record["schema_version"] == SCHEMA_VERSION, "schema version")
    config = record["config"]
    rebuilt = build_config(
        families=config["families"],
        orders=config["orders"],
        grids=config["grids"],
        cycles=config["cycles"],
        strip_width=config["strip_width"],
        smoke=config["smoke"],
    )
    _require(config == rebuilt, "stored configuration differs from the source protocol")
    current_hash = hashlib.sha256(SOURCE_DRIVER.read_bytes()).hexdigest()
    _require(
        record["provenance"]["driver_sha256"]
        == hashlib.sha256(DRIVER.read_bytes()).hexdigest(),
        "rescore driver hash mismatch",
    )
    _require(
        record["provenance"]["source_sha256"] == current_hash,
        "source hash mismatch",
    )
    _require(
        record["provenance"]["source_driver"] == "code/s6_packet_reflection.py",
        "source driver",
    )
    _require(record["provenance"]["region_rules"] == REGION_RULES, "region rules")
    _require(
        record["provenance"]["relative_l2_normalization"]
        == "full-domain exact squared norm",
        "relative L2 normalization",
    )
    expected_recurrence = exact_recurrence_selftest()
    _require(
        record["exact_selftest"] == expected_recurrence,
        "stored exact recurrence differs from source recomputation",
    )
    _require(
        expected_recurrence["p_t2_vs_t0"] < 1.0e-12
        and expected_recurrence["u_t2_vs_t0"] < 1.0e-12,
        "exact recurrence threshold",
    )
    cases = expected_cases(config)
    _require(
        set(record["runs"]) == set(cases),
        "run keys",
        set(record["runs"]),
        set(cases),
    )
    for key, family in cases.items():
        _verify_run(key, family, record["runs"][key], config)
    print("packet_region_rescore verify: ALL PASS")


def write_new_json(path: Path, record: dict[str, Any]) -> None:
    """Write a new JSON record and fail if any file already occupies its path."""
    path = Path(path)
    _require_finite_tree(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(record, handle, indent=1, allow_nan=False)
        handle.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", default=list(ALL_FAMILIES))
    parser.add_argument("--orders", nargs="+", type=int, default=list(packet_source.ORDERS))
    parser.add_argument("--grids", nargs="+", type=int, default=list(packet_source.GRIDS))
    parser.add_argument("--cycles", nargs="+", type=int, default=list(packet_source.CYCLES))
    parser.add_argument("--strip-width", type=float, default=packet_source.STRIP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_generated_dirs()
    if args.verify:
        verify_record(json.loads(args.output.read_text()))
        return
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite final record: {args.output}")
    if args.smoke:
        args.families, args.orders, args.grids, args.cycles = smoke_selection()
    config = build_config(
        families=args.families,
        orders=args.orders,
        grids=args.grids,
        cycles=args.cycles,
        strip_width=args.strip_width,
        smoke=args.smoke,
    )
    checkpoint = args.checkpoint or Path(str(args.output) + ".partial.jsonl")
    record = run_all(
        config, workers=args.workers, checkpoint_path=checkpoint
    )
    verify_record(record)
    write_new_json(args.output, record)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
