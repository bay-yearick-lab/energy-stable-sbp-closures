#!/usr/bin/env python3
"""Certify and apply degree-sized wall-transfer representations.

The coefficient generalized pencils preserve the banded cardinal interior.
Repeated endpoint knots change only degree-sized boundary neighborhoods. The
resulting low-rank correction supplies an exact Woodbury resolvent and a
universal semigroup representation for arbitrary finite-energy data.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.linalg import block_diag, cholesky, expm, solve_triangular

import boundary_spline_analysis as ba


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "code" / "results"
STRUCTURE_PATH = RESULTS / "universal_wall_transfer_structure.json"
RESOLVENT_PATH = RESULTS / "universal_wall_transfer_resolvent.json"
PREDICTION_PATH = RESULTS / "universal_wall_transfer_predictions.json"
RAW_PATH = RESULTS / "universal_wall_transfer_raw.json"
SUMMARY_PATH = RESULTS / "universal_wall_transfer_summary.json"
SUCCESSOR_PREDICTION_PATH = RESULTS / "universal_wall_transfer_successor_predictions.json"
SUCCESSOR_RAW_PATH = RESULTS / "universal_wall_transfer_successor_raw.json"
SUCCESSOR_SUMMARY_PATH = RESULTS / "universal_wall_transfer_successor_summary.json"
TENSOR_PATH = RESULTS / "universal_wall_transfer_tensor.json"
SCHEMA = "universal-wall-transfer-v1"
STRUCTURAL_TOLERANCE = 5.0e-12
TYPICAL_TOLERANCE = 0.10
WORST_TOLERANCE = 0.20
DEGREES = tuple(range(2, 13))
PREDICTOR_DEGREES = (2, 4, 6, 8, 10, 12)
ENCOUNTERS = tuple(range(1, 9))
DATA_FAMILIES = (
    "frozen_gaussian",
    "chirped",
    "multi_carrier",
    "phase_adversarial",
    "random_broadband",
)
HELDOUT_DATA_FAMILIES = (
    "heldout_chirped_mixture",
    "heldout_random_phase",
)
SUCCESSOR_HELDOUT_DATA_FAMILIES = (
    "successor_chirped_mixture",
    "successor_random_phase",
)


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _poly_add(left: list[Fraction], right: list[Fraction]) -> list[Fraction]:
    size = max(len(left), len(right))
    out = [Fraction(0) for _ in range(size)]
    for index, value in enumerate(left):
        out[index] += value
    for index, value in enumerate(right):
        out[index] += value
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def _poly_linear(
    coefficients: list[Fraction], constant: Fraction, slope: Fraction
) -> list[Fraction]:
    out = [Fraction(0) for _ in range(len(coefficients) + 1)]
    for degree, value in enumerate(coefficients):
        out[degree] += constant * value
        out[degree + 1] += slope * value
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def _poly_derivative(coefficients: list[Fraction]) -> list[Fraction]:
    if len(coefficients) == 1:
        return [Fraction(0)]
    return [Fraction(index) * coefficients[index] for index in range(1, len(coefficients))]


def _poly_product_integral(
    left: list[Fraction], right: list[Fraction]
) -> Fraction:
    total = Fraction(0)
    for i, left_value in enumerate(left):
        for j, right_value in enumerate(right):
            total += left_value * right_value / Fraction(i + j + 1)
    return total


def exact_open_uniform_basis(
    npts: int, degree: int
) -> tuple[list[int], list[dict[int, list[Fraction]]]]:
    """Return exact piecewise polynomials on a unit-cell knot sequence."""
    p = int(degree)
    cells = int(npts) - p
    if cells < 1:
        raise ValueError("npts must exceed degree")
    knots = [0] * (p + 1) + list(range(1, cells)) + [cells] * (p + 1)
    previous: list[dict[int, list[Fraction]]] = []
    for index in range(len(knots) - 1):
        pieces: dict[int, list[Fraction]] = {}
        if knots[index + 1] > knots[index]:
            pieces[knots[index]] = [Fraction(1)]
        previous.append(pieces)
    for current_degree in range(1, p + 1):
        count = len(knots) - current_degree - 1
        current: list[dict[int, list[Fraction]]] = []
        for index in range(count):
            pieces: dict[int, list[Fraction]] = {}
            left_denominator = knots[index + current_degree] - knots[index]
            right_denominator = (
                knots[index + current_degree + 1] - knots[index + 1]
            )
            cell_ids = set(previous[index]) | set(previous[index + 1])
            for cell in cell_ids:
                polynomial = [Fraction(0)]
                if left_denominator and cell in previous[index]:
                    polynomial = _poly_add(
                        polynomial,
                        _poly_linear(
                            previous[index][cell],
                            Fraction(cell - knots[index], left_denominator),
                            Fraction(1, left_denominator),
                        ),
                    )
                if right_denominator and cell in previous[index + 1]:
                    polynomial = _poly_add(
                        polynomial,
                        _poly_linear(
                            previous[index + 1][cell],
                            Fraction(
                                knots[index + current_degree + 1] - cell,
                                right_denominator,
                            ),
                            Fraction(-1, right_denominator),
                        ),
                    )
                if any(polynomial):
                    pieces[cell] = polynomial
            current.append(pieces)
        previous = current
    if len(previous) != npts:
        raise AssertionError("basis count mismatch")
    return knots, previous


def exact_coefficient_matrices(
    npts: int, degree: int
) -> dict[str, list[list[Fraction]]]:
    """Assemble exact dimensionless mass, advection, and stiffness matrices."""
    knots, basis = exact_open_uniform_basis(npts, degree)
    size = int(npts)
    mass = [[Fraction(0) for _ in range(size)] for _ in range(size)]
    advection = [[Fraction(0) for _ in range(size)] for _ in range(size)]
    stiffness = [[Fraction(0) for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(max(0, i - degree), min(size, i + degree + 1)):
            shared = set(basis[i]) & set(basis[j])
            for cell in shared:
                left = basis[i][cell]
                right = basis[j][cell]
                dleft = _poly_derivative(left)
                dright = _poly_derivative(right)
                mass[i][j] += _poly_product_integral(left, right)
                advection[i][j] += _poly_product_integral(left, dright)
                stiffness[i][j] += _poly_product_integral(dleft, dright)
    incidence = [
        [Fraction(0) for _ in range(size)] for _ in range(size - 1)
    ]
    for i in range(size - 1):
        denominator = knots[i + degree + 1] - knots[i + 1]
        scale = Fraction(degree, denominator)
        incidence[i][i] = -scale
        incidence[i][i + 1] = scale
    return {
        "mass": mass,
        "advection": advection,
        "stiffness": stiffness,
        "incidence": incidence,
    }


def cardinal_reference_npts(degree: int) -> int:
    """Return a size whose central row is a genuinely cardinal interior row.

    An anchor row carries the translation-invariant cardinal values only when
    it and every row within one bandwidth sit at least one full support away
    from both repeated-knot endpoints, which requires an index distance of at
    least 2p+1 from each wall. The smallest admissible size is 4p+3, and the
    value 4p+9 clears the distance with margin at every admissible degree.
    Every size at or above 4p+3 yields the identical cardinal band, so the
    reference size is a cost choice rather than a modeling choice.
    """
    return 4 * int(degree) + 9


def cardinal_row_values(
    source: list[list[Fraction]], bandwidth: int
) -> dict[int, Fraction]:
    """Read the cardinal band values from a verified interior anchor row.

    The check rejects any anchor closer than 2p+1 indices to either wall,
    which is exactly the regime in which the repeated endpoint knots reach
    the anchor band and destroy translation invariance.
    """
    rows = len(source)
    columns = len(source[0])
    anchor = rows // 2
    margin = 2 * bandwidth + 1
    if anchor - margin < 0 or anchor + margin > columns - 1:
        raise ValueError(
            "matrix is too small for a cardinal anchor row; supply a larger "
            "reference matrix or use exact_cardinal_toeplitz_core"
        )
    return {
        offset: source[anchor][anchor + offset]
        for offset in range(-bandwidth, bandwidth + 1)
    }


def exact_toeplitz_core(
    matrix: list[list[Fraction]],
    bandwidth: int,
    *,
    reference: list[list[Fraction]] | None = None,
) -> list[list[Fraction]]:
    """Extend one cardinal interior row across a finite Toeplitz section.

    The anchor row is validated to lie at least 2p+1 indices from both walls,
    so the extracted band is the cardinal same-spacing sequence rather than a
    row contaminated by the far wall. When ``matrix`` itself is too small, the
    caller passes a larger ``reference`` matrix of the same family and degree,
    whose central row supplies the cardinal band.
    """
    rows = len(matrix)
    columns = len(matrix[0])
    if rows != columns:
        raise ValueError("Toeplitz extraction requires a square matrix")
    values = cardinal_row_values(
        matrix if reference is None else reference, bandwidth
    )
    return [
        [values.get(j - i, Fraction(0)) for j in range(columns)]
        for i in range(rows)
    ]


@lru_cache(maxsize=None)
def _exact_cardinal_band(degree: int, name: str) -> tuple[Fraction, ...]:
    """Cache the cardinal band of one coefficient family at one degree."""
    reference = exact_coefficient_matrices(
        cardinal_reference_npts(degree), degree
    )[name]
    values = cardinal_row_values(reference, degree)
    return tuple(values[offset] for offset in range(-degree, degree + 1))


def exact_cardinal_toeplitz_core(
    npts: int, degree: int, name: str = "mass"
) -> list[list[Fraction]]:
    """Return the cardinal Toeplitz section of one family at any size.

    The band is always read from the reference size of
    ``cardinal_reference_npts(degree)``, so the section is the same-spacing
    cardinal comparator even when ``npts`` is too small to contain a cardinal
    interior row of its own.
    """
    band = _exact_cardinal_band(int(degree), name)
    values = {
        offset: band[offset + degree] for offset in range(-degree, degree + 1)
    }
    return [
        [values.get(j - i, Fraction(0)) for j in range(npts)]
        for i in range(npts)
    ]


def exact_incidence_core(npts: int) -> list[list[Fraction]]:
    """Return the translation-invariant coefficient derivative incidence."""
    matrix = [[Fraction(0) for _ in range(npts)] for _ in range(npts - 1)]
    for row in range(npts - 1):
        matrix[row][row] = Fraction(-1)
        matrix[row][row + 1] = Fraction(1)
    return matrix


def exact_subtract(
    left: list[list[Fraction]], right: list[list[Fraction]]
) -> list[list[Fraction]]:
    """Subtract exact matrices of identical shape."""
    return [
        [a - b for a, b in zip(left_row, right_row, strict=True)]
        for left_row, right_row in zip(left, right, strict=True)
    ]


def exact_rank(matrix: list[list[Fraction]]) -> int:
    """Compute rank by fraction-preserving Gaussian elimination."""
    if not matrix or not matrix[0]:
        return 0
    active_rows = [row[:] for row in matrix if any(row)]
    if not active_rows:
        return 0
    row = 0
    columns = len(active_rows[0])
    for column in range(columns):
        pivot = next(
            (candidate for candidate in range(row, len(active_rows)) if active_rows[candidate][column]),
            None,
        )
        if pivot is None:
            continue
        active_rows[row], active_rows[pivot] = active_rows[pivot], active_rows[row]
        pivot_value = active_rows[row][column]
        active_rows[row] = [value / pivot_value for value in active_rows[row]]
        for candidate in range(len(active_rows)):
            if candidate == row or not active_rows[candidate][column]:
                continue
            factor = active_rows[candidate][column]
            active_rows[candidate] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(
                    active_rows[candidate], active_rows[row], strict=True
                )
            ]
        row += 1
        if row == len(active_rows):
            break
    return row


def exact_pivot_columns(matrix: list[list[Fraction]]) -> list[int]:
    """Return pivot columns from fraction-preserving row reduction."""
    if not matrix or not matrix[0]:
        return []
    active_rows = [row[:] for row in matrix if any(row)]
    if not active_rows:
        return []
    row = 0
    pivots: list[int] = []
    for column in range(len(active_rows[0])):
        pivot = next(
            (candidate for candidate in range(row, len(active_rows)) if active_rows[candidate][column]),
            None,
        )
        if pivot is None:
            continue
        active_rows[row], active_rows[pivot] = active_rows[pivot], active_rows[row]
        pivot_value = active_rows[row][column]
        active_rows[row] = [value / pivot_value for value in active_rows[row]]
        for candidate in range(len(active_rows)):
            if candidate == row or not active_rows[candidate][column]:
                continue
            factor = active_rows[candidate][column]
            active_rows[candidate] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(
                    active_rows[candidate], active_rows[row], strict=True
                )
            ]
        pivots.append(column)
        row += 1
        if row == len(active_rows):
            break
    return pivots


def exact_determinant(matrix: list[list[Fraction]]) -> Fraction:
    """Compute a square determinant by exact fraction elimination."""
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("determinant requires a square matrix")
    work = [row[:] for row in matrix]
    determinant = Fraction(1)
    sign = 1
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if work[row][column]), None
        )
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            sign *= -1
        pivot_value = work[column][column]
        determinant *= pivot_value
        for row in range(column + 1, size):
            if not work[row][column]:
                continue
            factor = work[row][column] / pivot_value
            for entry in range(column + 1, size):
                work[row][entry] -= factor * work[column][entry]
            work[row][column] = Fraction(0)
    return sign * determinant


def modular_rank(
    matrix: list[list[Fraction]], prime: int = 1_000_000_007
) -> int:
    """Compute a rational rank lower certificate over one prime field."""
    work = [
        [
            (value.numerator % prime)
            * pow(value.denominator % prime, prime - 2, prime)
            % prime
            for value in row
        ]
        for row in matrix
    ]
    if not work:
        return 0
    row = 0
    for column in range(len(work[0])):
        pivot = next(
            (candidate for candidate in range(row, len(work)) if work[candidate][column]),
            None,
        )
        if pivot is None:
            continue
        work[row], work[pivot] = work[pivot], work[row]
        inverse = pow(work[row][column], prime - 2, prime)
        work[row] = [(value * inverse) % prime for value in work[row]]
        for candidate in range(row + 1, len(work)):
            if not work[candidate][column]:
                continue
            factor = work[candidate][column]
            work[candidate] = [
                (value - factor * pivot_value) % prime
                for value, pivot_value in zip(
                    work[candidate], work[row], strict=True
                )
            ]
        row += 1
        if row == len(work):
            break
    return row


def explicit_correction_rank_certificate(
    metric: list[list[Fraction]],
    generator: list[list[Fraction]],
    core_metric: list[list[Fraction]],
    core_generator: list[list[Fraction]],
) -> dict[str, int]:
    """Certify the explicit Schur-complement rank without forming an inverse."""
    size = len(metric)
    block = [
        core_metric[row] + core_generator[row] for row in range(size)
    ] + [metric[row] + generator[row] for row in range(size)]
    prime = 1_000_000_007
    block_rank = modular_rank(block, prime)
    return {
        "prime": prime,
        "block_rank": block_rank,
        "core_dimension": size,
        "explicit_correction_rank": block_rank - size,
    }


def _nonzero_rows(matrix: list[list[Fraction]]) -> list[int]:
    return [index for index, row in enumerate(matrix) if any(row)]


def _nonzero_columns(matrix: list[list[Fraction]]) -> list[int]:
    if not matrix:
        return []
    return [
        column
        for column in range(len(matrix[0]))
        if any(row[column] for row in matrix)
    ]


def _active_exact_matrix(
    matrix: list[list[Fraction]], rows: Iterable[int] | None = None
) -> list[list[Fraction]]:
    selected_rows = list(rows) if rows is not None else _nonzero_rows(matrix)
    selected_columns = _nonzero_columns(matrix)
    return [[matrix[i][j] for j in selected_columns] for i in selected_rows]


def _float_matrix(matrix: list[list[Fraction]], scale: float = 1.0) -> np.ndarray:
    return scale * np.asarray(
        [[float(value) for value in row] for row in matrix], dtype=float
    )


def numerical_rank(matrix: np.ndarray) -> tuple[int, float, np.ndarray]:
    """Return numerical rank, tolerance, and singular values."""
    singular = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    if singular.size == 0 or singular[0] == 0.0:
        return 0, 0.0, singular
    tolerance = (
        100.0
        * max(matrix.shape)
        * np.finfo(float).eps
        * float(singular[0])
    )
    return int(np.count_nonzero(singular > tolerance)), tolerance, singular


def exact_block_diagonal(
    first: list[list[Fraction]], second: list[list[Fraction]]
) -> list[list[Fraction]]:
    """Assemble an exact two-block diagonal matrix."""
    n0 = len(first)
    n1 = len(second)
    out = [[Fraction(0) for _ in range(n0 + n1)] for _ in range(n0 + n1)]
    for i in range(n0):
        out[i][:n0] = first[i][:]
    for i in range(n1):
        out[n0 + i][n0:] = second[i][:]
    return out


def exact_bmg_acoustic_pencil(
    mass: list[list[Fraction]],
    advection: list[list[Fraction]],
) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    """Assemble the rigid-wall BMG acoustic generalized pencil."""
    npts = len(mass)
    interior = list(range(1, npts - 1))
    velocity_mass = [[mass[i][j] for j in interior] for i in interior]
    metric = exact_block_diagonal(mass, velocity_mass)
    size = npts + len(interior)
    generator = [[Fraction(0) for _ in range(size)] for _ in range(size)]
    for i in range(npts):
        for local_j, j in enumerate(interior):
            generator[i][npts + local_j] = -advection[i][j]
            generator[npts + local_j][i] = -advection[j][i]
    return metric, generator


def exact_bmf_acoustic_pencil(
    mass0: list[list[Fraction]],
    mass1: list[list[Fraction]],
    incidence: list[list[Fraction]],
) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    """Assemble the rigid-wall mixed BMF coefficient pencil."""
    n0 = len(mass0)
    n1 = len(mass1)
    interior = list(range(1, n1 - 1))
    velocity_mass = [[mass1[i][j] for j in interior] for i in interior]
    metric = exact_block_diagonal(mass0, velocity_mass)
    size = n0 + len(interior)
    generator = [[Fraction(0) for _ in range(size)] for _ in range(size)]
    delta_t_mass = [
        [
            sum(incidence[k][i] * mass1[k][j] for k in range(n1))
            for j in interior
        ]
        for i in range(n0)
    ]
    mass_delta = [
        [
            sum(mass1[i][k] * incidence[k][j] for k in range(n1))
            for j in range(n0)
        ]
        for i in interior
    ]
    for i in range(n0):
        for j in range(len(interior)):
            generator[i][n0 + j] = delta_t_mass[i][j]
            generator[n0 + j][i] = -mass_delta[j][i]
    return metric, generator


def correction_record(
    matrix: list[list[Fraction]], core: list[list[Fraction]]
) -> dict[str, Any]:
    """Summarize exact support, exact rank, and numerical singular decay."""
    correction = exact_subtract(matrix, core)
    active = _active_exact_matrix(correction)
    floating = _float_matrix(active) if active and active[0] else np.zeros((0, 0))
    numeric_rank, tolerance, singular = numerical_rank(floating)
    return {
        "exact_rank": exact_rank(active),
        "numerical_rank": numeric_rank,
        "numerical_rank_tolerance": tolerance,
        "row_support": _nonzero_rows(correction),
        "column_support": _nonzero_columns(correction),
        "active_shape": list(floating.shape),
        "singular_values": singular.tolist(),
    }


def common_pencil_record(
    metric: list[list[Fraction]],
    generator: list[list[Fraction]],
    core_metric: list[list[Fraction]],
    core_generator: list[list[Fraction]],
) -> dict[str, Any]:
    """Summarize the common boundary range of a generalized pencil."""
    delta_metric = exact_subtract(metric, core_metric)
    delta_generator = exact_subtract(generator, core_generator)
    active_rows = sorted(
        set(_nonzero_rows(delta_metric)) | set(_nonzero_rows(delta_generator))
    )
    concatenated = [
        delta_metric[row] + delta_generator[row] for row in active_rows
    ]
    floating = _float_matrix(concatenated)
    numeric_rank, tolerance, singular = numerical_rank(floating)
    return {
        "exact_common_left_rank": exact_rank(concatenated),
        "numerical_common_left_rank": numeric_rank,
        "numerical_rank_tolerance": tolerance,
        "row_support": active_rows,
        "row_support_count": len(active_rows),
        "metric_column_support": _nonzero_columns(delta_metric),
        "generator_column_support": _nonzero_columns(delta_generator),
        "singular_values": singular.tolist(),
    }


def _fraction_certificate(value: Fraction) -> dict[str, Any]:
    encoded = f"{value.numerator}/{value.denominator}".encode()
    return {
        "nonzero": bool(value),
        "sign": 0 if value == 0 else (1 if value > 0 else -1),
        "numerator_digits": len(str(abs(value.numerator))),
        "denominator_digits": len(str(value.denominator)),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def rank_proof_certificate(degree: int) -> dict[str, Any]:
    """Create exact minors that certify every claimed rank at one degree."""
    npts = 4 * degree + 9
    matrices = exact_coefficient_matrices(npts, degree)
    cores = {
        name: exact_cardinal_toeplitz_core(npts, degree, name)
        for name in ("mass", "advection", "stiffness")
    }
    left = list(range(2 * degree - 1))
    operator_minors = {}
    for name in ("mass", "advection", "stiffness"):
        correction = exact_subtract(matrices[name], cores[name])
        minor = [[correction[i][j] for j in left] for i in left]
        operator_minors[name] = _fraction_certificate(exact_determinant(minor))

    incidence = exact_subtract(
        matrices["incidence"], exact_incidence_core(npts)
    )
    incidence_rows = _nonzero_rows(incidence)
    incidence_pivots = exact_pivot_columns(incidence)

    bmg_metric, bmg_generator = exact_bmg_acoustic_pencil(
        matrices["mass"], matrices["advection"]
    )
    bmg_core_metric, bmg_core_generator = exact_bmg_acoustic_pencil(
        cores["mass"], cores["advection"]
    )
    bmg_delta_metric = exact_subtract(bmg_metric, bmg_core_metric)
    bmg_delta_generator = exact_subtract(bmg_generator, bmg_core_generator)
    bmg_rows = sorted(
        set(_nonzero_rows(bmg_delta_metric))
        | set(_nonzero_rows(bmg_delta_generator))
    )
    bmg_concatenated = [
        bmg_delta_metric[row] + bmg_delta_generator[row] for row in bmg_rows
    ]

    derivative_matrices = exact_coefficient_matrices(npts - 1, degree - 1)
    derivative_core = exact_cardinal_toeplitz_core(
        npts - 1, degree - 1, "mass"
    )
    bmf_metric, bmf_generator = exact_bmf_acoustic_pencil(
        matrices["mass"],
        derivative_matrices["mass"],
        matrices["incidence"],
    )
    bmf_core_metric, bmf_core_generator = exact_bmf_acoustic_pencil(
        cores["mass"], derivative_core, exact_incidence_core(npts)
    )
    bmf_delta_metric = exact_subtract(bmf_metric, bmf_core_metric)
    bmf_delta_generator = exact_subtract(bmf_generator, bmf_core_generator)
    bmf_rows = sorted(
        set(_nonzero_rows(bmf_delta_metric))
        | set(_nonzero_rows(bmf_delta_generator))
    )
    bmf_concatenated = [
        bmf_delta_metric[row] + bmf_delta_generator[row] for row in bmf_rows
    ]
    return {
        "degree": degree,
        "certificate_npts": npts,
        "left_endpoint_minor_size": 2 * degree - 1,
        "operator_left_minor_determinants": operator_minors,
        "incidence": {
            "nonzero_row_count": len(incidence_rows),
            "pivot_columns": incidence_pivots,
            "rank": len(incidence_pivots),
        },
        "bmg_acoustic_pencil": {
            "common_row_support_count": len(bmg_rows),
            "pivot_columns": exact_pivot_columns(bmg_concatenated),
            "rank": exact_rank(bmg_concatenated),
            "explicit_operator": explicit_correction_rank_certificate(
                bmg_metric,
                bmg_generator,
                bmg_core_metric,
                bmg_core_generator,
            ),
        },
        "bmf_mixed_acoustic_pencil": {
            "common_row_support_count": len(bmf_rows),
            "pivot_columns": exact_pivot_columns(bmf_concatenated),
            "rank": exact_rank(bmf_concatenated),
            "explicit_operator": explicit_correction_rank_certificate(
                bmf_metric,
                bmf_generator,
                bmf_core_metric,
                bmf_core_generator,
            ),
        },
    }


def structure_atlas(output_path: Path) -> dict[str, Any]:
    """Build the exact degree and resolution atlas for both spline routes."""
    records: list[dict[str, Any]] = []
    for degree in DEGREES:
        sizes = (4 * degree + 9, 6 * degree + 9, 8 * degree + 9)
        for npts in sizes:
            matrices = exact_coefficient_matrices(npts, degree)
            cores = {
                name: exact_cardinal_toeplitz_core(npts, degree, name)
                for name in ("mass", "advection", "stiffness")
            }
            incidence_core = exact_incidence_core(npts)
            matrix_records = {
                name: correction_record(matrices[name], cores[name])
                for name in ("mass", "advection", "stiffness")
            }
            matrix_records["incidence"] = correction_record(
                matrices["incidence"], incidence_core
            )

            bmg_metric, bmg_generator = exact_bmg_acoustic_pencil(
                matrices["mass"], matrices["advection"]
            )
            bmg_core_metric, bmg_core_generator = exact_bmg_acoustic_pencil(
                cores["mass"], cores["advection"]
            )
            derivative_matrices = exact_coefficient_matrices(npts - 1, degree - 1)
            derivative_mass_core = exact_cardinal_toeplitz_core(
                npts - 1, degree - 1, "mass"
            )
            bmf_metric, bmf_generator = exact_bmf_acoustic_pencil(
                matrices["mass"],
                derivative_matrices["mass"],
                matrices["incidence"],
            )
            bmf_core_metric, bmf_core_generator = exact_bmf_acoustic_pencil(
                cores["mass"], derivative_mass_core, incidence_core
            )
            records.append(
                {
                    "degree": degree,
                    "npts": npts,
                    "cell_count": npts - degree,
                    "coefficient_corrections": matrix_records,
                    "bmg_acoustic_pencil": common_pencil_record(
                        bmg_metric,
                        bmg_generator,
                        bmg_core_metric,
                        bmg_core_generator,
                    ),
                    "bmf_mixed_acoustic_pencil": common_pencil_record(
                        bmf_metric,
                        bmf_generator,
                        bmf_core_metric,
                        bmf_core_generator,
                    ),
                }
            )
            print(f"structure p={degree} N={npts} complete", flush=True)
    result = {
        "schema": SCHEMA,
        "artifact_state": "verified_structure_atlas",
        "coordinate_decision": "coefficient generalized pencil, with BMF retained as a mixed primal and derivative-space system",
        "normalization": {
            "knot_cells": "unit cells used for exact rational rank",
            "unit_interval": "mass scales by h, stiffness by 1/h, and advection is unchanged; ranks and relative singular decay are invariant",
        },
        "rank_definition": "exact fraction-preserving elimination on the active correction and a 100 times dimension times machine-epsilon numerical threshold",
        "proved_formulas_on_declared_degree_range": {
            "degrees": [2, 12],
            "mass_advection_stiffness_rank": "4p-2",
            "incidence_rank": "2p-2",
            "bmg_acoustic_common_left_rank": "8p-6",
            "bmf_mixed_acoustic_common_left_rank": "8p-8",
            "bmg_explicit_semigroup_transfer_rank": "8p-6",
            "bmf_explicit_semigroup_transfer_rank": "6p-6",
            "resolution_condition": "N at least 4p+9 in the certified atlas; endpoint support and rank are unchanged across N=4p+9, 6p+9, and 8p+9",
        },
        "proof_certificates": [
            rank_proof_certificate(degree) for degree in DEGREES
        ],
        "records": records,
        "source_sha256": sha256(Path(__file__)),
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _float_exact(matrix: list[list[Fraction]], scale: float = 1.0) -> np.ndarray:
    return scale * np.array(
        [[float(value) for value in row] for row in matrix], dtype=float
    )


def bmg_float_system(degree: int, npts: int) -> dict[str, Any]:
    """Return bounded and interior-core BMG acoustic coefficient systems."""
    matrices = exact_coefficient_matrices(npts, degree)
    mass_core = exact_cardinal_toeplitz_core(npts, degree, "mass")
    advection_core = exact_cardinal_toeplitz_core(npts, degree, "advection")
    cells = npts - degree
    mass = _float_exact(matrices["mass"], 1.0 / cells)
    advection = _float_exact(matrices["advection"])
    core_mass = _float_exact(mass_core, 1.0 / cells)
    core_advection = _float_exact(advection_core)
    interior = np.arange(1, npts - 1)
    metric = block_diag(mass, mass[np.ix_(interior, interior)])
    core_metric = block_diag(
        core_mass, core_mass[np.ix_(interior, interior)]
    )
    generator = np.block(
        [
            [np.zeros((npts, npts)), -advection[:, interior]],
            [-advection[np.ix_(interior, np.arange(npts))], np.zeros((npts - 2, npts - 2))],
        ]
    )
    core_generator = np.block(
        [
            [np.zeros((npts, npts)), -core_advection[:, interior]],
            [-core_advection[np.ix_(interior, np.arange(npts))], np.zeros((npts - 2, npts - 2))],
        ]
    )
    matrix = np.linalg.solve(metric, generator)
    core_matrix = np.linalg.solve(core_metric, core_generator)
    correction = matrix - core_matrix
    singular_left, singular, singular_right = np.linalg.svd(
        correction, full_matrices=False
    )
    rank_tolerance = (
        100.0
        * correction.shape[0]
        * np.finfo(float).eps
        * max(float(singular[0]), 1.0)
    )
    rank = int(np.count_nonzero(singular > rank_tolerance))
    exact_metric, exact_generator = exact_bmg_acoustic_pencil(
        matrices["mass"], matrices["advection"]
    )
    exact_core_metric, exact_core_generator = exact_bmg_acoustic_pencil(
        mass_core, advection_core
    )
    exact_delta_metric = exact_subtract(exact_metric, exact_core_metric)
    exact_delta_generator = exact_subtract(
        exact_generator, exact_core_generator
    )
    row_support = np.asarray(
        sorted(
            set(_nonzero_rows(exact_delta_metric))
            | set(_nonzero_rows(exact_delta_generator))
        ),
        dtype=int,
    )
    semigroup_rank = 8 * degree - 6
    transfer_left = singular_left[:, :semigroup_rank] * singular[:semigroup_rank]
    transfer_right = singular_right[:semigroup_rank, :]
    reconstructed = core_matrix + transfer_left @ transfer_right
    return {
        "route": "BMG",
        "degree": degree,
        "npts": npts,
        "x": ba.clamped_bspline_matrices(npts, degree)[0],
        "bmat": ba.clamped_bspline_matrices(npts, degree)[2],
        "metric": metric,
        "generator_pencil": generator,
        "matrix": matrix,
        "core_metric": core_metric,
        "core_generator_pencil": core_generator,
        "core_matrix": core_matrix,
        "transfer_left": transfer_left,
        "transfer_right": transfer_right,
        "correction_singular_values": singular,
        "rank": rank,
        "rank_tolerance": rank_tolerance,
        "boundary_channel_dimension": int(row_support.size),
        "semigroup_transfer_dimension": semigroup_rank,
        "boundary_row_support": row_support,
        "reconstructed": reconstructed,
    }


def bmf_float_system(degree: int, npts: int) -> dict[str, Any]:
    """Return bounded and interior-core BMF mixed acoustic systems."""
    primal = exact_coefficient_matrices(npts, degree)
    derivative = exact_coefficient_matrices(npts - 1, degree - 1)
    primal_mass_core = exact_cardinal_toeplitz_core(npts, degree, "mass")
    derivative_mass_core = exact_cardinal_toeplitz_core(
        npts - 1, degree - 1, "mass"
    )
    incidence_core = exact_incidence_core(npts)
    exact_metric, exact_generator = exact_bmf_acoustic_pencil(
        primal["mass"], derivative["mass"], primal["incidence"]
    )
    exact_core_metric, exact_core_generator = exact_bmf_acoustic_pencil(
        primal_mass_core, derivative_mass_core, incidence_core
    )
    cells = npts - degree
    metric = _float_exact(exact_metric, 1.0 / cells)
    generator = _float_exact(exact_generator)
    core_metric = _float_exact(exact_core_metric, 1.0 / cells)
    core_generator = _float_exact(exact_core_generator)
    matrix = np.linalg.solve(metric, generator)
    core_matrix = np.linalg.solve(core_metric, core_generator)
    correction = matrix - core_matrix
    singular_left, singular, singular_right = np.linalg.svd(
        correction, full_matrices=False
    )
    rank_tolerance = (
        100.0
        * correction.shape[0]
        * np.finfo(float).eps
        * max(float(singular[0]), 1.0)
    )
    rank = int(np.count_nonzero(singular > rank_tolerance))
    exact_delta_metric = exact_subtract(exact_metric, exact_core_metric)
    exact_delta_generator = exact_subtract(
        exact_generator, exact_core_generator
    )
    row_support = np.asarray(
        sorted(
            set(_nonzero_rows(exact_delta_metric))
            | set(_nonzero_rows(exact_delta_generator))
        ),
        dtype=int,
    )
    delta_metric = metric - core_metric
    delta_generator = generator - core_generator
    selector = np.eye(matrix.shape[0])[row_support, :]
    semigroup_rank = 6 * degree - 6
    transfer_left = singular_left[:, :semigroup_rank] * singular[:semigroup_rank]
    transfer_right = singular_right[:semigroup_rank, :]
    reconstructed = core_matrix + transfer_left @ transfer_right
    x, _knots, bmat, _gmat = ba.clamped_bspline_matrices(npts, degree)
    derivative_knots = ba.derivative_space_knots(
        ba.open_uniform_knots(npts, degree)
    )
    derivative_nodes = ba.derivative_greville_nodes(
        derivative_knots, degree - 1
    )
    derivative_bmat = ba.bspline_evaluation_matrix(
        derivative_knots, degree - 1, derivative_nodes
    )
    return {
        "route": "BMF",
        "degree": degree,
        "npts": npts,
        "x": x,
        "bmat": bmat,
        "velocity_bmat": derivative_bmat,
        "velocity_x": derivative_nodes,
        "velocity_to_pressure": ba.bspline_evaluation_matrix(
            derivative_knots, degree - 1, x
        ),
        "metric": metric,
        "generator_pencil": generator,
        "matrix": matrix,
        "core_metric": core_metric,
        "core_generator_pencil": core_generator,
        "core_matrix": core_matrix,
        "transfer_left": transfer_left,
        "transfer_right": transfer_right,
        "correction_singular_values": singular,
        "rank": rank,
        "rank_tolerance": rank_tolerance,
        "boundary_channel_dimension": int(row_support.size),
        "semigroup_transfer_dimension": semigroup_rank,
        "boundary_row_support": row_support,
        "reconstructed": reconstructed,
    }


def woodbury_resolvent(
    core: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    value: complex,
) -> np.ndarray:
    """Evaluate the exact low-rank resolvent through the boundary system."""
    core_resolvent = np.linalg.inv(value * np.eye(core.shape[0]) - core)
    middle = np.linalg.inv(
        np.eye(right.shape[0]) - right @ core_resolvent @ left
    )
    return core_resolvent + core_resolvent @ left @ middle @ right @ core_resolvent


def refined_solve(
    matrix: np.ndarray, right_hand_side: np.ndarray, iterations: int = 3
) -> np.ndarray:
    """Solve a dense system with bounded residual refinement."""
    solution = np.linalg.solve(matrix, right_hand_side)
    for _ in range(iterations):
        residual = right_hand_side - matrix @ solution
        correction = np.linalg.solve(matrix, residual)
        candidate = solution + correction
        if np.linalg.norm(right_hand_side - matrix @ candidate) >= np.linalg.norm(
            residual
        ):
            break
        solution = candidate
    return solution


def pencil_woodbury_resolvent(
    system: dict[str, Any], value: complex
) -> np.ndarray:
    """Evaluate the exact generalized-pencil boundary resolvent."""
    metric = system["metric"]
    generator = system["generator_pencil"]
    core_metric = system["core_metric"]
    core_generator = system["core_generator_pencil"]
    row_support = np.asarray(system["boundary_row_support"], dtype=int)
    selector = np.eye(metric.shape[0])[row_support, :]
    left = selector.T
    metric_coordinates = selector @ (metric - core_metric)
    generator_coordinates = selector @ (generator - core_generator)
    core_pencil = value * core_metric - core_generator
    core_inverse = refined_solve(core_pencil, np.eye(core_pencil.shape[0]))
    boundary_coordinates = value * metric_coordinates - generator_coordinates
    boundary_matrix = (
        np.eye(row_support.size)
        + boundary_coordinates @ core_inverse @ left
    )
    return core_inverse - (
        core_inverse
        @ left
        @ refined_solve(
            boundary_matrix, boundary_coordinates @ core_inverse
        )
    )


def mass_scaled_pencil_woodbury_resolvent(
    system: dict[str, Any], value: complex
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate direct and boundary resolvents in the physical energy norm."""
    metric = system["metric"]
    generator = system["generator_pencil"]
    core_metric = system["core_metric"]
    core_generator = system["core_generator_pencil"]
    row_support = np.asarray(system["boundary_row_support"], dtype=int)
    selector = np.eye(metric.shape[0])[row_support, :]
    left = selector.T
    metric_coordinates = selector @ (metric - core_metric)
    generator_coordinates = selector @ (generator - core_generator)
    boundary_coordinates = value * metric_coordinates - generator_coordinates
    factor = cholesky(metric, lower=True)

    def two_sided_scale(matrix: np.ndarray) -> np.ndarray:
        left_scaled = solve_triangular(factor, matrix, lower=True)
        return solve_triangular(
            factor, left_scaled.T, lower=True
        ).T

    scaled_full = two_sided_scale(value * metric - generator)
    scaled_core = two_sided_scale(value * core_metric - core_generator)
    scaled_left = solve_triangular(factor, left, lower=True)
    scaled_coordinates = solve_triangular(
        factor, boundary_coordinates.T, lower=True
    ).T
    core_inverse = refined_solve(scaled_core, np.eye(scaled_core.shape[0]))
    boundary_matrix = (
        np.eye(row_support.size)
        + scaled_coordinates @ core_inverse @ scaled_left
    )
    woodbury = core_inverse - (
        core_inverse
        @ scaled_left
        @ refined_solve(
            boundary_matrix, scaled_coordinates @ core_inverse
        )
    )
    return woodbury, refined_solve(
        scaled_full, np.eye(scaled_full.shape[0])
    )


def resolvent_certificate(output_path: Path) -> dict[str, Any]:
    """Certify the exact boundary-channel Woodbury identity before propagation."""
    points = (complex(0.75, 0.37), complex(1.5, 2.25), complex(3.0, 7.0))
    records = []
    for route, builder in (("BMG", bmg_float_system), ("BMF", bmf_float_system)):
        for degree in PREDICTOR_DEGREES:
          for npts in (4 * degree + 17, 8 * degree + 17):
            system = builder(degree, npts)
            metric = system["metric"]
            generator = system["generator_pencil"]
            matrix = system["matrix"]
            skew_residual = np.linalg.norm(generator + generator.T) / max(
                np.linalg.norm(generator), 1.0e-300
            )
            metric_skew_residual = np.linalg.norm(
                matrix.T @ metric + metric @ matrix
            ) / max(np.linalg.norm(metric @ matrix), 1.0e-300)
            pencil_reconstruction = (
                system["core_generator_pencil"]
                + (
                    generator - system["core_generator_pencil"]
                )
            )
            metric_reconstruction = (
                system["core_metric"]
                + (metric - system["core_metric"])
            )
            pencil_identity_residual = max(
                np.linalg.norm(pencil_reconstruction - generator)
                / max(np.linalg.norm(generator), 1.0e-300),
                np.linalg.norm(metric_reconstruction - metric)
                / max(np.linalg.norm(metric), 1.0e-300),
            )
            point_records = []
            for point in points:
                woodbury = pencil_woodbury_resolvent(system, point)
                direct = np.linalg.inv(point * metric - generator)
                coefficient_forward_residual = np.linalg.norm(woodbury - direct) / max(
                    np.linalg.norm(direct), 1.0e-300
                )
                energy_woodbury, energy_direct = (
                    mass_scaled_pencil_woodbury_resolvent(system, point)
                )
                energy_residual = np.linalg.norm(
                    energy_woodbury - energy_direct
                ) / max(np.linalg.norm(energy_direct), 1.0e-300)
                state_from_pencil = woodbury @ metric
                state_direct = np.linalg.inv(
                    point * np.eye(matrix.shape[0]) - matrix
                )
                coefficient_state_residual = np.linalg.norm(
                    state_from_pencil - state_direct
                ) / max(np.linalg.norm(state_direct), 1.0e-300)
                point_records.append(
                    {
                        "z": [point.real, point.imag],
                        "coefficient_generalized_forward_residual": coefficient_forward_residual,
                        "coefficient_state_forward_residual": coefficient_state_residual,
                        "energy_scaled_resolvent_relative_residual": energy_residual,
                        "energy_resolvent_upper_bound": 1.0 / point.real,
                    }
                )
            records.append(
                {
                    "route": route,
                    "degree": degree,
                    "npts": npts,
                    "state_dimension": int(matrix.shape[0]),
                    "boundary_channel_dimension": int(
                        system["boundary_channel_dimension"]
                    ),
                    "expected_minimal_common_pencil_dimension": (
                        8 * degree - 6 if route == "BMG" else 8 * degree - 8
                    ),
                    "pencil_identity_residual": pencil_identity_residual,
                    "skew_pencil_residual": skew_residual,
                    "metric_skew_residual": metric_skew_residual,
                    "points": point_records,
                }
            )
            print(
                f"resolvent {route} p={degree} N={npts} complete",
                flush=True,
            )
    maximum_residual = max(
        value
        for record in records
        for point in record["points"]
        for value in (point["energy_scaled_resolvent_relative_residual"],)
    )
    maximum_coefficient_forward_residual = max(
        value
        for record in records
        for point in record["points"]
        for value in (
            point["coefficient_generalized_forward_residual"],
            point["coefficient_state_forward_residual"],
        )
    )
    maximum_structure = max(
        value
        for record in records
        for value in (
            record["pencil_identity_residual"],
            record["skew_pencil_residual"],
            record["metric_skew_residual"],
        )
    )
    result = {
        "schema": SCHEMA,
        "artifact_state": "verified_boundary_channel_resolvent",
        "formula": "Q0^-1 - Q0^-1 U (I + C(z) Q0^-1 U)^-1 C(z) Q0^-1",
        "records": records,
        "decision": {
            "maximum_resolvent_relative_residual": maximum_residual,
            "maximum_coefficient_forward_residual": maximum_coefficient_forward_residual,
            "maximum_structural_relative_residual": maximum_structure,
            "tolerance": STRUCTURAL_TOLERANCE,
            "accepted": bool(
                maximum_residual <= STRUCTURAL_TOLERANCE
                and maximum_structure <= STRUCTURAL_TOLERANCE
            ),
        },
        "source_sha256": sha256(Path(__file__)),
        "structure_sha256": sha256(STRUCTURE_PATH),
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _coefficients(system: dict[str, Any], values: np.ndarray) -> np.ndarray:
    return np.linalg.solve(system["bmat"], np.asarray(values, dtype=float))


def _velocity_coefficients(
    system: dict[str, Any], values: np.ndarray
) -> np.ndarray:
    """Interpolate velocity in the route's natural coefficient space."""
    if system["route"] == "BMG":
        return np.linalg.solve(
            system["bmat"], np.asarray(values, dtype=float)
        )
    return np.linalg.solve(
        system["velocity_bmat"], np.asarray(values, dtype=float)
    )


def phase_adversarial_pair(
    system: dict[str, Any], band_max: int
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Construct equal-magnitude modal states separated by wall coupling."""
    x = np.asarray(system["x"], dtype=float)
    npts = int(system["npts"])
    dimension = system["matrix"].shape[0]
    mode_count = max(4, min(band_max, dimension // 4))
    columns = []
    for mode in range(1, mode_count + 1):
        pressure = _coefficients(system, np.cos(mode * np.pi * x))
        velocity_x = (
            x if system["route"] == "BMG" else system["velocity_x"]
        )
        velocity = _velocity_coefficients(
            system, np.sin(mode * np.pi * velocity_x)
        )[1:-1]
        velocity_size = system["matrix"].shape[0] - npts
        columns.append(np.concatenate([pressure, np.zeros(velocity_size)]))
        columns.append(np.concatenate([np.zeros(npts), velocity]))
    modal = np.column_stack(columns)
    gram = modal.T @ system["metric"] @ modal
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    keep = eigenvalues > 1.0e-12 * np.max(eigenvalues)
    orthogonal = modal @ (eigenvectors[:, keep] / np.sqrt(eigenvalues[keep]))
    modal_propagator = (
        orthogonal.T
        @ system["metric"]
        @ expm(system["matrix"])
        @ orthogonal
    )
    off_diagonal = modal_propagator - np.diag(np.diag(modal_propagator))
    row, column = np.unravel_index(
        np.argmax(np.abs(off_diagonal)), off_diagonal.shape
    )
    first = np.zeros(orthogonal.shape[1])
    second = np.zeros(orthogonal.shape[1])
    first[row] = second[row] = 1.0 / math.sqrt(2.0)
    first[column] = 1.0 / math.sqrt(2.0)
    second[column] = -1.0 / math.sqrt(2.0)
    first_state = orthogonal @ first
    second_state = orthogonal @ second
    output_first = modal_propagator @ first
    output_second = modal_propagator @ second
    certificate: dict[str, float | int] = {
        "mode_row": int(row),
        "mode_column": int(column),
        "maximum_off_diagonal_transfer": float(abs(off_diagonal[row, column])),
        "initial_modal_magnitude_difference": float(
            np.max(np.abs(np.abs(first) - np.abs(second)))
        ),
        "target_modal_output_separation": float(
            abs(output_first[row] - output_second[row])
        ),
    }
    return first_state, second_state, certificate


def data_states(
    system: dict[str, Any], heldout: bool, successor: bool = False
) -> dict[str, np.ndarray]:
    """Build construction or held-out broadband coefficient states."""
    x = np.asarray(system["x"], dtype=float)
    degree = int(system["degree"])
    npts = int(system["npts"])
    band_max = max(2, min((npts - degree) // 2, npts - degree - 2))
    rng = np.random.default_rng(
        77123 if heldout and successor else (90317 if heldout else 24051)
    )

    if heldout:
        if successor:
            chirped = np.exp(-((x - 0.57) / 0.13) ** 2) * np.cos(
                2.0
                * np.pi
                * (
                    0.27 * band_max * x
                    + 0.31 * band_max * (x - 0.57) ** 2
                    + 0.19
                )
            )
        else:
            chirped = np.exp(-((x - 0.61) / 0.17) ** 2) * np.cos(
                2.0
                * np.pi
                * (
                    0.31 * band_max * x
                    + 0.23 * band_max * (x - 0.61) ** 2
                )
            )
        phase_modes = np.arange(1, band_max + 1)
        phase = rng.uniform(-np.pi, np.pi, size=band_max)
        amplitude = 1.0 / np.sqrt(phase_modes)
        random_field = np.sum(
            amplitude[:, None]
            * np.cos(phase_modes[:, None] * np.pi * x[None, :] + phase[:, None]),
            axis=0,
        )
        if successor:
            values = {
                "successor_chirped_mixture": chirped
                + 0.29 * np.cos(0.67 * band_max * np.pi * x - 0.53),
                "successor_random_phase": random_field,
            }
        else:
            values = {
                "heldout_chirped_mixture": chirped
                + 0.31 * np.cos(0.73 * band_max * np.pi * x + 0.41),
                "heldout_random_phase": random_field,
            }
    else:
        center = 0.34
        width = 0.11
        carrier = max(2, int(round(0.42 * band_max)))
        frozen = np.exp(-((x - center) / width) ** 2) * np.cos(
            carrier * np.pi * (x - center)
        )
        chirped = np.exp(-((x - 0.43) / 0.15) ** 2) * np.cos(
            2.0 * np.pi * (0.22 * band_max * x + 0.18 * band_max * (x - 0.43) ** 2)
        )
        carriers = sorted({max(1, band_max // 5), max(2, band_max // 2), band_max})
        multi = sum(
            weight * np.cos(mode * np.pi * x + phase)
            for weight, mode, phase in zip(
                (1.0, 0.63, 0.37), carriers, (0.0, 0.7, -1.1), strict=True
            )
        )
        modes = np.arange(1, band_max + 1)
        random_phase = rng.uniform(-np.pi, np.pi, size=band_max)
        random_field = np.sum(
            (1.0 / modes)[:, None]
            * np.cos(modes[:, None] * np.pi * x[None, :] + random_phase[:, None]),
            axis=0,
        )
        values = {
            "frozen_gaussian": frozen,
            "chirped": chirped,
            "multi_carrier": multi,
            "random_broadband": random_field,
        }

    states: dict[str, np.ndarray] = {}
    for name, pressure_values in values.items():
        velocity_x = (
            x if system["route"] == "BMG" else system["velocity_x"]
        )
        pressure_velocity_grid = np.interp(velocity_x, x, pressure_values)
        velocity_values = velocity_x * (1.0 - velocity_x) * (
            0.47 * pressure_velocity_grid
            + 0.19 * np.sin((band_max + 1) * np.pi * velocity_x)
        )
        pressure_coefficients = _coefficients(system, pressure_values)
        velocity_coefficients = _velocity_coefficients(
            system, velocity_values
        )[1:-1]
        state = np.concatenate([pressure_coefficients, velocity_coefficients])
        energy = math.sqrt(float(state @ system["metric"] @ state))
        states[name] = state / energy

    if not heldout:
        first, second, _certificate = phase_adversarial_pair(system, band_max)
        states["phase_adversarial"] = first
        states["phase_adversarial_pair"] = second
    return states


def state_observables(
    system: dict[str, Any], state: np.ndarray, initial_energy: float
) -> dict[str, float]:
    """Evaluate energy, traces, phase, amplitude, and wall-local content."""
    npts = int(system["npts"])
    pressure_coefficients = state[:npts]
    velocity_size = npts if system["route"] == "BMG" else npts - 1
    velocity_coefficients = np.zeros(velocity_size)
    velocity_coefficients[1:-1] = state[npts:]
    pressure = system["bmat"] @ pressure_coefficients
    velocity = (
        system["bmat"] @ velocity_coefficients
        if system["route"] == "BMG"
        else system["velocity_to_pressure"] @ velocity_coefficients
    )
    x = np.asarray(system["x"], dtype=float)
    energy = math.sqrt(max(float(state @ system["metric"] @ state), 0.0))
    carrier = max(1, min((npts - system["degree"]) // 3, npts // 3))
    cosine = np.cos(carrier * np.pi * x)
    sine = np.sin(carrier * np.pi * x)
    cosine_component = float(np.dot(pressure, cosine))
    sine_component = float(np.dot(pressure, sine))
    wall_mask = (x <= 0.15) | (x >= 0.85)
    density = pressure**2 + velocity**2
    left_mask = x <= 0.15
    right_mask = x >= 0.85
    wall_energy = float(
        np.trapezoid(density[left_mask], x[left_mask])
        + np.trapezoid(density[right_mask], x[right_mask])
    )
    return {
        "energy": energy,
        "transient_amplification": energy / initial_energy,
        "wall_reflected_energy": wall_energy,
        "phase": math.atan2(sine_component, cosine_component),
        "amplitude": math.hypot(cosine_component, sine_component),
        "wall_work": float(pressure[-1] * velocity[-1] - pressure[0] * velocity[0]),
        "left_pressure": float(pressure[0]),
        "right_pressure": float(pressure[-1]),
    }


def _case_id(
    route: str, degree: int, npts: int, family: str, encounter: int
) -> str:
    return f"{route}_p{degree}_N{npts}_{family}_t{encounter}"


def encode_state(state: np.ndarray) -> str:
    """Encode a floating state as deterministic little-endian bytes."""
    values = np.asarray(state, dtype="<f8")
    return base64.b64encode(values.tobytes()).decode("ascii")


def decode_state(encoded: str, dimension: int) -> np.ndarray:
    """Decode a deterministic little-endian floating state."""
    values = np.frombuffer(base64.b64decode(encoded), dtype="<f8")
    if values.size != dimension:
        raise ValueError("encoded state dimension mismatch")
    return values.copy()


def freeze_predictions(
    output_path: Path, successor: bool = False
) -> dict[str, Any]:
    """Freeze compressed predictions before direct held-out propagation."""
    if output_path.exists():
        raise FileExistsError(output_path)
    cases: list[dict[str, Any]] = []
    systems: list[dict[str, Any]] = []
    resolvent_artifact = json.loads(RESOLVENT_PATH.read_text())
    resolvent_by_system = {
        (row["route"], row["degree"], row["npts"]): max(
            point["energy_scaled_resolvent_relative_residual"]
            for point in row["points"]
        )
        for row in resolvent_artifact["records"]
    }
    for route, builder in (("BMG", bmg_float_system), ("BMF", bmf_float_system)):
        for degree, npts in (
            (degree, npts)
            for degree in PREDICTOR_DEGREES
            for npts in (4 * degree + 17, 8 * degree + 17)
        ):
            system = builder(degree, npts)
            construction_states = data_states(system, heldout=False)
            heldout_states = data_states(
                system, heldout=True, successor=successor
            )
            heldout_families = (
                SUCCESSOR_HELDOUT_DATA_FAMILIES
                if successor
                else HELDOUT_DATA_FAMILIES
            )
            states = {**construction_states, **heldout_states}
            band_max = max(2, min((npts - degree) // 2, npts - degree - 2))
            _first, _second, adversarial = phase_adversarial_pair(
                system, band_max
            )
            identity_residual = np.linalg.norm(
                system["matrix"] - system["reconstructed"]
            ) / max(np.linalg.norm(system["matrix"]), 1.0e-300)
            resolvent_residual = resolvent_by_system[(route, degree, npts)]
            metric_skew = np.linalg.norm(
                system["matrix"].T @ system["metric"]
                + system["metric"] @ system["matrix"]
            ) / max(np.linalg.norm(system["metric"] @ system["matrix"]), 1.0e-300)
            systems.append(
                {
                    "route": route,
                    "degree": degree,
                    "npts": npts,
                    "state_dimension": int(system["matrix"].shape[0]),
                    "explicit_correction_numerical_rank": int(system["rank"]),
                    "boundary_channel_dimension": int(
                        system["boundary_channel_dimension"]
                    ),
                    "semigroup_transfer_dimension": int(
                        system["semigroup_transfer_dimension"]
                    ),
                    "boundary_channels_over_degree": system[
                        "boundary_channel_dimension"
                    ]
                    / degree,
                    "generator_identity_residual": identity_residual,
                    "woodbury_resolvent_residual": resolvent_residual,
                    "metric_skew_residual": metric_skew,
                    "correction_singular_values": system["correction_singular_values"].tolist(),
                    "scalar_impossibility_certificate": adversarial,
                }
            )
            initial_energies = {
                family: math.sqrt(float(initial @ system["metric"] @ initial))
                for family, initial in states.items()
            }
            for encounter in ENCOUNTERS:
                propagator = expm(system["reconstructed"] * encounter)
                for family, initial in states.items():
                    final = propagator @ initial
                    cases.append(
                        {
                            "case_id": _case_id(
                                route, degree, npts, family, encounter
                            ),
                            "route": route,
                            "degree": degree,
                            "npts": npts,
                            "family": family,
                            "evidence_partition": "heldout"
                            if family in heldout_families
                            else "construction_control",
                            "encounters": encounter,
                            "initial_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                            "predicted_state_sha256": hashlib.sha256(final.tobytes()).hexdigest(),
                            "state_dimension": int(final.size),
                            "predicted_state_base64": encode_state(final),
                            "predicted_observables": state_observables(
                                system, final, initial_energies[family]
                            ),
                        }
                    )
            print(
                f"freeze {route} p={degree} N={npts} complete",
                flush=True,
            )
    heldout_initial_hashes = sorted(
        {
            case["initial_sha256"]
            for case in cases
            if case["evidence_partition"] == "heldout"
        }
    )
    predecessor: dict[str, Any] | None = None
    novelty_certificate: dict[str, Any] | None = None
    if successor:
        first_prediction = json.loads(PREDICTION_PATH.read_text())
        first_hashes = {
            case["initial_sha256"]
            for case in first_prediction["cases"]
            if case["evidence_partition"] == "heldout"
        }
        overlap = first_hashes & set(heldout_initial_hashes)
        if overlap:
            raise AssertionError("successor held-out states overlap the first protocol")
        predecessor = {
            "prediction_sha256": sha256(PREDICTION_PATH),
            "raw_sha256": sha256(RAW_PATH),
            "summary_sha256": sha256(SUMMARY_PATH),
        }
        novelty_certificate = {
            "compared_prediction_sha256": sha256(PREDICTION_PATH),
            "overlap_count": 0,
            "new_unique_initial_hash_count": len(heldout_initial_hashes),
            "new_initial_sha256": heldout_initial_hashes,
        }
    record = {
        "schema": SCHEMA,
        "revision": 2 if successor else 1,
        "protocol": "broadband-successor-v2" if successor else "broadband-first-v1",
        "artifact_state": "immutable_pre_result_predictions",
        "predictor": "Toeplitz interior generator plus compressed degree-sized boundary transfer",
        "factorization": "rank-revealing minimal explicit correction with 8p-6 BMG channels and 6p-6 BMF channels",
        "construction_data_excluded": list(heldout_families),
        "heldout_case_contract": {
            "degrees": list(PREDICTOR_DEGREES),
            "routes": ["BMG", "BMF"],
            "resolutions": "N equals 4p+17 and 8p+17",
            "families": list(heldout_families),
            "encounters": list(ENCOUNTERS),
            "heldout_case_count": 2
            * len(PREDICTOR_DEGREES)
            * 2
            * len(heldout_families)
            * len(ENCOUNTERS),
            "total_case_count": len(cases),
        },
        "construction_controls": {
            "families": list(DATA_FAMILIES) + ["phase_adversarial_pair"],
            "information_use": "the operator compression uses no packet data; controls diagnose universality and scalar nonidentifiability",
        },
        "tolerances": {
            "structural_relative": STRUCTURAL_TOLERANCE,
            "typical_observable_relative": TYPICAL_TOLERANCE,
            "worst_observable_relative": WORST_TOLERANCE,
            "absolute_floors": {
                "energy": 1.0e-12,
                "transient_amplification": 1.0e-12,
                "wall_reflected_energy": 1.0e-12,
                "phase": 1.0e-10,
                "amplitude": 1.0e-10,
                "wall_work": 1.0e-12,
                "left_pressure": 1.0e-10,
                "right_pressure": 1.0e-10,
                "full_field": 1.0e-12,
            },
        },
        "promotion": "all structural identities pass; typical and worst observable errors remain below frozen limits; every full-field state error remains below structural tolerance",
        "systems": systems,
        "cases": cases,
        "heldout_initial_sha256": heldout_initial_hashes,
        "predecessor": predecessor,
        "heldout_novelty_certificate": novelty_certificate,
        "source_sha256": sha256(Path(__file__)),
        "structure_sha256": sha256(STRUCTURE_PATH),
        "resolvent_sha256": sha256(RESOLVENT_PATH),
    }
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def run_heldout(
    prediction_path: Path, output_path: Path, successor: bool = False
) -> dict[str, Any]:
    """Evaluate the untouched broadband states with the direct bounded system."""
    if output_path.exists():
        raise FileExistsError(output_path)
    prediction = json.loads(prediction_path.read_text())
    revision = int(prediction.get("revision", 1))
    if successor != (revision == 2):
        raise ValueError("run protocol does not match the frozen prediction revision")
    expected = {case["case_id"] for case in prediction["cases"]}
    results: list[dict[str, Any]] = []
    for route, builder in (("BMG", bmg_float_system), ("BMF", bmf_float_system)):
        for degree, npts in (
            (degree, npts)
            for degree in PREDICTOR_DEGREES
            for npts in (4 * degree + 17, 8 * degree + 17)
        ):
            system = builder(degree, npts)
            states = {
                **data_states(system, heldout=False),
                **data_states(system, heldout=True, successor=successor),
            }
            initial_energies = {
                family: math.sqrt(float(initial @ system["metric"] @ initial))
                for family, initial in states.items()
            }
            for encounter in ENCOUNTERS:
                propagator = expm(system["matrix"] * encounter)
                for family, initial in states.items():
                    final = propagator @ initial
                    results.append(
                        {
                            "case_id": _case_id(
                                route, degree, npts, family, encounter
                            ),
                            "route": route,
                            "initial_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                            "state_dimension": int(final.size),
                            "measured_state_base64": encode_state(final),
                            "measured_observables": state_observables(
                                system, final, initial_energies[family]
                            ),
                        }
                    )
            print(
                f"heldout {route} p={degree} N={npts} complete",
                flush=True,
            )
    if {row["case_id"] for row in results} != expected:
        raise AssertionError("held-out case set does not match the prediction")
    record = {
        "schema": SCHEMA,
        "artifact_state": "heldout_direct_results",
        "prediction_sha256": sha256(prediction_path),
        "source_sha256": sha256(Path(__file__)),
        "results": results,
    }
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def _wrapped_phase_error(predicted: float, measured: float) -> float:
    return abs(math.atan2(math.sin(predicted - measured), math.cos(predicted - measured)))


def summarize(
    prediction_path: Path, raw_path: Path, output_path: Path
) -> dict[str, Any]:
    """Apply the frozen universal-predictor decision contract."""
    prediction = json.loads(prediction_path.read_text())
    raw = json.loads(raw_path.read_text())
    predicted = {row["case_id"]: row for row in prediction["cases"]}
    measured = {row["case_id"]: row for row in raw["results"]}
    floors = prediction["tolerances"]["absolute_floors"]
    rows = []
    observable_errors: list[float] = []
    full_field_errors: list[float] = []
    system_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    for identifier in sorted(predicted):
        pcase = predicted[identifier]
        rcase = measured[identifier]
        pstate = decode_state(
            pcase["predicted_state_base64"], pcase["state_dimension"]
        )
        rstate = decode_state(
            rcase["measured_state_base64"], rcase["state_dimension"]
        )
        system_key = (pcase["route"], pcase["degree"], pcase["npts"])
        if system_key not in system_cache:
            builder = (
                bmg_float_system if pcase["route"] == "BMG" else bmf_float_system
            )
            system_cache[system_key] = builder(pcase["degree"], pcase["npts"])
        metric = system_cache[system_key]["metric"]
        difference = pstate - rstate
        full_field = math.sqrt(max(float(difference @ metric @ difference), 0.0)) / max(
            math.sqrt(max(float(rstate @ metric @ rstate), 0.0)),
            floors["full_field"],
        )
        errors = {}
        for name, pvalue in pcase["predicted_observables"].items():
            rvalue = rcase["measured_observables"][name]
            absolute = (
                _wrapped_phase_error(pvalue, rvalue)
                if name == "phase"
                else abs(pvalue - rvalue)
            )
            relative = absolute / max(abs(rvalue), floors[name])
            errors[name] = {"absolute": absolute, "relative": relative}
            observable_errors.append(relative)
        full_field_errors.append(full_field)
        rows.append(
            {
                "case_id": identifier,
                "full_field_relative_error": full_field,
                "observable_errors": errors,
            }
        )
    structural_failures = [
        row
        for row in prediction["systems"]
        if row["generator_identity_residual"] > STRUCTURAL_TOLERANCE
        or row["woodbury_resolvent_residual"] > STRUCTURAL_TOLERANCE
        or row["metric_skew_residual"] > STRUCTURAL_TOLERANCE
    ]
    typical = float(np.median(observable_errors))
    worst = float(np.max(observable_errors))
    worst_field = float(np.max(full_field_errors))
    promotion = (
        not structural_failures
        and typical <= TYPICAL_TOLERANCE
        and worst <= WORST_TOLERANCE
        and worst_field <= STRUCTURAL_TOLERANCE
    )
    record = {
        "schema": SCHEMA,
        "artifact_state": "verified_summary",
        "prediction_sha256": sha256(prediction_path),
        "raw_sha256": sha256(raw_path),
        "decision": {
            "promotion_earned": promotion,
            "claim_status": "accepted_universal_boundary_transfer" if promotion else "universal_boundary_transfer_not_promoted",
            "typical_observable_relative_error": typical,
            "worst_observable_relative_error": worst,
            "worst_full_field_relative_error": worst_field,
            "structural_failure_count": len(structural_failures),
        },
        "cases": rows,
    }
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def verify(
    structure_path: Path,
    prediction_path: Path,
    raw_path: Path,
    summary_path: Path,
) -> None:
    """Verify artifact binding, complete matrices, and all frozen decisions."""
    structure = json.loads(structure_path.read_text())
    prediction = json.loads(prediction_path.read_text())
    raw = json.loads(raw_path.read_text())
    summary = json.loads(summary_path.read_text())
    assert structure["schema"] == SCHEMA
    assert prediction["structure_sha256"] == sha256(structure_path)
    assert prediction["resolvent_sha256"] == sha256(RESOLVENT_PATH)
    assert raw["prediction_sha256"] == sha256(prediction_path)
    assert summary["prediction_sha256"] == sha256(prediction_path)
    assert summary["raw_sha256"] == sha256(raw_path)
    expected_structure = {
        (degree, npts)
        for degree in DEGREES
        for npts in (4 * degree + 9, 6 * degree + 9, 8 * degree + 9)
    }
    assert {
        (row["degree"], row["npts"]) for row in structure["records"]
    } == expected_structure
    expected_cases = {row["case_id"] for row in prediction["cases"]}
    assert len(expected_cases) == prediction["heldout_case_contract"][
        "total_case_count"
    ]
    assert {row["case_id"] for row in prediction["cases"]} == expected_cases
    assert {row["case_id"] for row in raw["results"]} == expected_cases
    assert {row["case_id"] for row in summary["cases"]} == expected_cases
    assert summary["decision"]["promotion_earned"]
    print("universal_wall_transfer verify: ALL PASS")


def tensor_certificate(output_path: Path) -> dict[str, Any]:
    """Certify the Cartesian face-line lift of each boundary transfer."""
    records: list[dict[str, Any]] = []
    rng = np.random.default_rng(48117)
    for route, builder in (("BMG", bmg_float_system), ("BMF", bmf_float_system)):
        for degree in (2, 6, 12):
            npts = 4 * degree + 17
            system = builder(degree, npts)
            correction = system["matrix"] - system["core_matrix"]
            transfer = system["transfer_left"] @ system["transfer_right"]
            line_rank = int(system["semigroup_transfer_dimension"])
            for tangential_lines in (3, 5):
                identity = np.eye(tangential_lines)
                direct = np.kron(identity, correction)
                lifted_left = np.kron(identity, system["transfer_left"])
                lifted_right = np.kron(identity, system["transfer_right"])
                reconstructed = lifted_left @ lifted_right
                scale = max(np.linalg.norm(direct), 1.0e-300)
                matrix_residual = np.linalg.norm(direct - reconstructed) / scale
                vector = rng.standard_normal(direct.shape[1])
                application_residual = np.linalg.norm(
                    direct @ vector - lifted_left @ (lifted_right @ vector)
                ) / max(np.linalg.norm(direct @ vector), 1.0e-300)
                factor_residual = np.linalg.norm(correction - transfer) / max(
                    np.linalg.norm(correction), 1.0e-300
                )
                records.append(
                    {
                        "route": route,
                        "degree": degree,
                        "npts": npts,
                        "tangential_lines": tangential_lines,
                        "normal_line_channel_dimension": line_rank,
                        "lifted_channel_dimension": tangential_lines * line_rank,
                        "certified_exact_lifted_rank": tangential_lines * line_rank,
                        "double_precision_effective_rank": int(
                            np.linalg.matrix_rank(
                                reconstructed,
                                tol=system["rank_tolerance"],
                            )
                        ),
                        "one_dimensional_factor_residual": factor_residual,
                        "matrix_identity_residual": matrix_residual,
                        "random_application_residual": application_residual,
                    }
                )
    maximum_residual = max(
        max(
            row["one_dimensional_factor_residual"],
            row["matrix_identity_residual"],
            row["random_application_residual"],
        )
        for row in records
    )
    record = {
        "schema": SCHEMA,
        "artifact_state": "verified_tensor_face_line_certificate",
        "scope": "Cartesian tensor-product face lift without cut-cell or curvilinear geometry",
        "identity": "I_t tensor (A-A0) = (I_t tensor B)(I_t tensor C)",
        "tolerance": STRUCTURAL_TOLERANCE,
        "accepted": bool(maximum_residual <= STRUCTURAL_TOLERANCE),
        "maximum_relative_residual": maximum_residual,
        "records": records,
        "source_sha256": sha256(Path(__file__)),
    }
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    structure = commands.add_parser("structure")
    structure.add_argument("--output", type=Path, default=STRUCTURE_PATH)
    resolvent = commands.add_parser("resolvent")
    resolvent.add_argument("--output", type=Path, default=RESOLVENT_PATH)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--output", type=Path, default=PREDICTION_PATH)
    run = commands.add_parser("run")
    run.add_argument("--prediction", type=Path, default=PREDICTION_PATH)
    run.add_argument("--output", type=Path, default=RAW_PATH)
    successor_freeze = commands.add_parser("freeze-successor")
    successor_freeze.add_argument(
        "--output", type=Path, default=SUCCESSOR_PREDICTION_PATH
    )
    successor_run = commands.add_parser("run-successor")
    successor_run.add_argument(
        "--prediction", type=Path, default=SUCCESSOR_PREDICTION_PATH
    )
    successor_run.add_argument("--output", type=Path, default=SUCCESSOR_RAW_PATH)
    successor_summary = commands.add_parser("summarize-successor")
    successor_summary.add_argument(
        "--prediction", type=Path, default=SUCCESSOR_PREDICTION_PATH
    )
    successor_summary.add_argument("--raw", type=Path, default=SUCCESSOR_RAW_PATH)
    successor_summary.add_argument(
        "--output", type=Path, default=SUCCESSOR_SUMMARY_PATH
    )
    tensor = commands.add_parser("tensor")
    tensor.add_argument("--output", type=Path, default=TENSOR_PATH)
    summary = commands.add_parser("summarize")
    summary.add_argument("--prediction", type=Path, default=PREDICTION_PATH)
    summary.add_argument("--raw", type=Path, default=RAW_PATH)
    summary.add_argument("--output", type=Path, default=SUMMARY_PATH)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--structure", type=Path, default=STRUCTURE_PATH)
    verify_parser.add_argument("--prediction", type=Path, default=PREDICTION_PATH)
    verify_parser.add_argument("--raw", type=Path, default=RAW_PATH)
    verify_parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "structure":
        result = structure_atlas(args.output)
        print(f"wrote {args.output} with {len(result['records'])} records")
    elif args.command == "resolvent":
        result = resolvent_certificate(args.output)
        print(json.dumps(result["decision"], indent=2, sort_keys=True))
    elif args.command == "freeze":
        result = freeze_predictions(args.output)
        print(f"wrote {args.output} with {len(result['cases'])} predictions")
    elif args.command == "freeze-successor":
        result = freeze_predictions(args.output, successor=True)
        print(f"wrote {args.output} with {len(result['cases'])} predictions")
    elif args.command == "run":
        result = run_heldout(args.prediction, args.output)
        print(f"wrote {args.output} with {len(result['results'])} results")
    elif args.command == "run-successor":
        result = run_heldout(args.prediction, args.output, successor=True)
        print(f"wrote {args.output} with {len(result['results'])} results")
    elif args.command == "summarize":
        result = summarize(args.prediction, args.raw, args.output)
        print(json.dumps(result["decision"], indent=2, sort_keys=True))
    elif args.command == "summarize-successor":
        result = summarize(args.prediction, args.raw, args.output)
        print(json.dumps(result["decision"], indent=2, sort_keys=True))
    elif args.command == "tensor":
        result = tensor_certificate(args.output)
        print(json.dumps({
            "accepted": result["accepted"],
            "maximum_relative_residual": result["maximum_relative_residual"],
        }, indent=2, sort_keys=True))
    else:
        verify(args.structure, args.prediction, args.raw, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
