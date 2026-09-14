#!/usr/bin/env python3
"""Verify arbitrary-degree wall-rank formulas in exact finite fields.

The modular assembly is the rational Cox-de Boor construction reduced modulo
a prime. Two independent primes provide adversarial controls without the large
integer growth of fraction elimination at high degree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import universal_wall_transfer as uwt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "code" / "results" / "all_degree_wall_rank_controls.json"
DEFAULT_PRIMES = (1_000_000_007, 1_000_000_009)


def _poly_add(left: list[int], right: list[int], prime: int) -> list[int]:
    size = max(len(left), len(right))
    result = [0] * size
    for index, value in enumerate(left):
        result[index] = (result[index] + value) % prime
    for index, value in enumerate(right):
        result[index] = (result[index] + value) % prime
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def _poly_linear(
    coefficients: list[int], constant: int, slope: int, prime: int
) -> list[int]:
    result = [0] * (len(coefficients) + 1)
    for degree, value in enumerate(coefficients):
        result[degree] = (result[degree] + constant * value) % prime
        result[degree + 1] = (result[degree + 1] + slope * value) % prime
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def _poly_derivative(coefficients: list[int], prime: int) -> list[int]:
    if len(coefficients) == 1:
        return [0]
    return [
        index * coefficients[index] % prime
        for index in range(1, len(coefficients))
    ]


def _poly_product_integral(
    left: list[int], right: list[int], prime: int
) -> int:
    total = 0
    for i, left_value in enumerate(left):
        for j, right_value in enumerate(right):
            total += left_value * right_value * pow(i + j + 1, prime - 2, prime)
    return total % prime


def modular_open_uniform_basis(
    npts: int, degree: int, prime: int
) -> tuple[list[int], list[dict[int, list[int]]]]:
    """Return unit-cell B-spline polynomials over one prime field."""
    cells = npts - degree
    if cells < 1:
        raise ValueError("npts must exceed degree")
    knots = [0] * (degree + 1) + list(range(1, cells)) + [cells] * (degree + 1)
    previous: list[dict[int, list[int]]] = []
    for index in range(len(knots) - 1):
        pieces: dict[int, list[int]] = {}
        if knots[index + 1] > knots[index]:
            pieces[knots[index]] = [1]
        previous.append(pieces)
    for current_degree in range(1, degree + 1):
        count = len(knots) - current_degree - 1
        current: list[dict[int, list[int]]] = []
        for index in range(count):
            pieces: dict[int, list[int]] = {}
            left_denominator = knots[index + current_degree] - knots[index]
            right_denominator = knots[index + current_degree + 1] - knots[index + 1]
            for cell in set(previous[index]) | set(previous[index + 1]):
                polynomial = [0]
                if left_denominator and cell in previous[index]:
                    inverse = pow(left_denominator, prime - 2, prime)
                    polynomial = _poly_add(
                        polynomial,
                        _poly_linear(
                            previous[index][cell],
                            (cell - knots[index]) * inverse % prime,
                            inverse,
                            prime,
                        ),
                        prime,
                    )
                if right_denominator and cell in previous[index + 1]:
                    inverse = pow(right_denominator, prime - 2, prime)
                    polynomial = _poly_add(
                        polynomial,
                        _poly_linear(
                            previous[index + 1][cell],
                            (knots[index + current_degree + 1] - cell)
                            * inverse
                            % prime,
                            -inverse % prime,
                            prime,
                        ),
                        prime,
                    )
                if any(polynomial):
                    pieces[cell] = polynomial
            current.append(pieces)
        previous = current
    if len(previous) != npts:
        raise AssertionError("basis count mismatch")
    return knots, previous


def modular_coefficient_matrices(
    npts: int, degree: int, prime: int
) -> dict[str, list[list[int]]]:
    """Assemble mass, advection, and incidence matrices modulo one prime."""
    knots, basis = modular_open_uniform_basis(npts, degree, prime)
    mass = [[0] * npts for _ in range(npts)]
    advection = [[0] * npts for _ in range(npts)]
    for i in range(npts):
        for j in range(max(0, i - degree), min(npts, i + degree + 1)):
            for cell in set(basis[i]) & set(basis[j]):
                left = basis[i][cell]
                right = basis[j][cell]
                mass[i][j] = (
                    mass[i][j] + _poly_product_integral(left, right, prime)
                ) % prime
                advection[i][j] = (
                    advection[i][j]
                    + _poly_product_integral(
                        left, _poly_derivative(right, prime), prime
                    )
                ) % prime
    incidence = [[0] * npts for _ in range(npts - 1)]
    for row in range(npts - 1):
        denominator = knots[row + degree + 1] - knots[row + 1]
        scale = degree * pow(denominator, prime - 2, prime) % prime
        incidence[row][row] = -scale % prime
        incidence[row][row + 1] = scale
    return {"mass": mass, "advection": advection, "incidence": incidence}


def toeplitz_core(
    matrix: list[list[int]],
    bandwidth: int,
    *,
    reference: list[list[int]] | None = None,
) -> list[list[int]]:
    """Extend one cardinal row across a finite Toeplitz section.

    The anchor row is validated to lie at least 2p+1 indices from both walls.
    Below that distance the clamped endpoint knots reach the anchor band, and
    the extracted section is no longer the same-spacing cardinal comparator
    the wall-transfer theorem defines. Small sizes therefore pass a larger
    ``reference`` matrix of the same family and degree.
    """
    rows = len(matrix)
    values = uwt.cardinal_row_values(
        matrix if reference is None else reference, bandwidth
    )
    return [
        [values.get(j - i, 0) for j in range(rows)]
        for i in range(rows)
    ]


@lru_cache(maxsize=None)
def _modular_cardinal_band(
    degree: int, prime: int, name: str
) -> tuple[int, ...]:
    """Cache the cardinal band of one modular coefficient family."""
    reference = modular_coefficient_matrices(
        uwt.cardinal_reference_npts(degree), degree, prime
    )[name]
    values = uwt.cardinal_row_values(reference, degree)
    return tuple(values[offset] for offset in range(-degree, degree + 1))


def modular_cardinal_toeplitz_core(
    npts: int, degree: int, prime: int, name: str = "mass"
) -> list[list[int]]:
    """Return the cardinal Toeplitz section of one family at any size.

    The band comes from the central row of a reference matrix of size
    ``uwt.cardinal_reference_npts(degree)``, which is large enough that the
    anchor row and its whole band are translation-invariant cardinal rows.
    """
    band = _modular_cardinal_band(int(degree), int(prime), name)
    values = {
        offset: band[offset + degree] for offset in range(-degree, degree + 1)
    }
    return [
        [values.get(j - i, 0) for j in range(npts)] for i in range(npts)
    ]


def incidence_core(npts: int, prime: int) -> list[list[int]]:
    """Return the constant cardinal derivative incidence."""
    matrix = [[0] * npts for _ in range(npts - 1)]
    for row in range(npts - 1):
        matrix[row][row] = -1 % prime
        matrix[row][row + 1] = 1
    return matrix


def subtract(
    left: list[list[int]], right: list[list[int]], prime: int
) -> list[list[int]]:
    """Subtract equal-shaped matrices modulo one prime."""
    return [
        [(a - b) % prime for a, b in zip(left_row, right_row, strict=True)]
        for left_row, right_row in zip(left, right, strict=True)
    ]


def block_diagonal(
    first: list[list[int]], second: list[list[int]]
) -> list[list[int]]:
    """Assemble a two-block diagonal matrix."""
    n0 = len(first)
    n1 = len(second)
    result = [[0] * (n0 + n1) for _ in range(n0 + n1)]
    for row in range(n0):
        result[row][:n0] = first[row][:]
    for row in range(n1):
        result[n0 + row][n0:] = second[row][:]
    return result


def bmg_pencil(
    mass: list[list[int]], advection: list[list[int]], prime: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Assemble the rigid-wall BMG coefficient pencil."""
    npts = len(mass)
    interior = list(range(1, npts - 1))
    velocity_mass = [[mass[i][j] for j in interior] for i in interior]
    metric = block_diagonal(mass, velocity_mass)
    size = len(metric)
    generator = [[0] * size for _ in range(size)]
    for i in range(npts):
        for local_j, j in enumerate(interior):
            generator[i][npts + local_j] = -advection[i][j] % prime
            generator[npts + local_j][i] = -advection[j][i] % prime
    return metric, generator


def bmf_pencil(
    mass0: list[list[int]],
    mass1: list[list[int]],
    incidence: list[list[int]],
    prime: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Assemble the rigid-wall mixed BMF coefficient pencil."""
    n0 = len(mass0)
    n1 = len(mass1)
    interior = list(range(1, n1 - 1))
    velocity_mass = [[mass1[i][j] for j in interior] for i in interior]
    metric = block_diagonal(mass0, velocity_mass)
    size = len(metric)
    generator = [[0] * size for _ in range(size)]
    for i in range(n0):
        for local_j, j in enumerate(interior):
            upper = sum(
                incidence[k][i] * mass1[k][j] for k in range(n1)
            ) % prime
            lower = sum(
                mass1[j][k] * incidence[k][i] for k in range(n1)
            ) % prime
            generator[i][n0 + local_j] = upper
            generator[n0 + local_j][i] = -lower % prime
    return metric, generator


def rank_mod(matrix: list[list[int]], prime: int) -> int:
    """Return matrix rank over one prime field."""
    work = [row[:] for row in matrix if any(row)]
    if not work:
        return 0
    pivot_row = 0
    for column in range(len(work[0])):
        pivot = next(
            (
                row
                for row in range(pivot_row, len(work))
                if work[row][column] % prime
            ),
            None,
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        inverse = pow(work[pivot_row][column] % prime, prime - 2, prime)
        work[pivot_row] = [value * inverse % prime for value in work[pivot_row]]
        for row in range(pivot_row + 1, len(work)):
            factor = work[row][column] % prime
            if factor:
                work[row] = [
                    (value - factor * pivot_value) % prime
                    for value, pivot_value in zip(
                        work[row], work[pivot_row], strict=True
                    )
                ]
        pivot_row += 1
        if pivot_row == len(work):
            break
    return pivot_row


def predicted_ranks(degree: int, npts: int) -> dict[str, int]:
    """Return the piecewise all-degree rank formulas.

    The formulas are stated against the cardinal same-spacing Toeplitz
    comparator of ``modular_cardinal_toeplitz_core``. Earlier revisions read
    the comparator from the anchor row of the finite clamped matrix, which is
    contaminated by the far wall for N at most 4p-2 and produced different
    small-size values.
    """
    return {
        "mass_correction": min(npts, 4 * degree - 2),
        "bmg_pencil": min(2 * npts - 2, 8 * degree - 6),
        "bmg_explicit": min(
            2 * npts - 4,
            npts + 4 * degree - 6,
            8 * degree - 6,
        ),
        "bmf_pencil": min(2 * npts - 3, 8 * degree - 8),
        "bmf_explicit_x": min(npts - 3, 4 * degree - 4),
        "bmf_explicit_y": 2 * degree - 2,
        "bmf_explicit": min(npts + 2 * degree - 5, 6 * degree - 6),
    }


def modular_control(degree: int, npts: int, prime: int) -> dict[str, int]:
    """Evaluate all three exact ranks at one degree, size, and prime."""
    primal = modular_coefficient_matrices(npts, degree, prime)
    primal_mass_core = modular_cardinal_toeplitz_core(
        npts, degree, prime, "mass"
    )
    primal_advection_core = modular_cardinal_toeplitz_core(
        npts, degree, prime, "advection"
    )
    mass_correction_rank = rank_mod(
        subtract(primal["mass"], primal_mass_core, prime), prime
    )
    bmg_metric, bmg_generator = bmg_pencil(
        primal["mass"], primal["advection"], prime
    )
    bmg_core_metric, bmg_core_generator = bmg_pencil(
        primal_mass_core, primal_advection_core, prime
    )
    bmg_delta_metric = subtract(bmg_metric, bmg_core_metric, prime)
    bmg_delta_generator = subtract(bmg_generator, bmg_core_generator, prime)
    bmg_common = [
        metric_row + generator_row
        for metric_row, generator_row in zip(
            bmg_delta_metric, bmg_delta_generator, strict=True
        )
    ]
    bmg_size = len(bmg_metric)
    bmg_explicit_block = [
        bmg_core_metric[row] + bmg_core_generator[row]
        for row in range(bmg_size)
    ] + [bmg_metric[row] + bmg_generator[row] for row in range(bmg_size)]
    bmg_explicit_rank = rank_mod(bmg_explicit_block, prime) - bmg_size
    derivative = modular_coefficient_matrices(npts - 1, degree - 1, prime)
    derivative_mass_core = modular_cardinal_toeplitz_core(
        npts - 1, degree - 1, prime, "mass"
    )
    bmf_metric, bmf_generator = bmf_pencil(
        primal["mass"], derivative["mass"], primal["incidence"], prime
    )
    bmf_core_metric, bmf_core_generator = bmf_pencil(
        primal_mass_core,
        derivative_mass_core,
        incidence_core(npts, prime),
        prime,
    )
    bmf_delta_metric = subtract(bmf_metric, bmf_core_metric, prime)
    bmf_delta_generator = subtract(bmf_generator, bmf_core_generator, prime)
    bmf_common = [
        metric_row + generator_row
        for metric_row, generator_row in zip(
            bmf_delta_metric, bmf_delta_generator, strict=True
        )
    ]
    primal_size = npts
    velocity_size = len(bmf_metric) - primal_size
    primal_mass = [row[:primal_size] for row in bmf_metric[:primal_size]]
    primal_mass_core_block = [
        row[:primal_size] for row in bmf_core_metric[:primal_size]
    ]
    velocity_mass = [row[primal_size:] for row in bmf_metric[primal_size:]]
    velocity_mass_core = [
        row[primal_size:] for row in bmf_core_metric[primal_size:]
    ]
    upper = [row[primal_size:] for row in bmf_generator[:primal_size]]
    upper_core = [
        row[primal_size:] for row in bmf_core_generator[:primal_size]
    ]
    lower = [row[:primal_size] for row in bmf_generator[primal_size:]]
    lower_core = [
        row[:primal_size] for row in bmf_core_generator[primal_size:]
    ]
    x_schur = [
        primal_mass_core_block[row] + upper_core[row]
        for row in range(primal_size)
    ] + [
        primal_mass[row] + upper[row] for row in range(primal_size)
    ]
    y_schur = [
        velocity_mass_core[row] + lower_core[row]
        for row in range(velocity_size)
    ] + [
        velocity_mass[row] + lower[row] for row in range(velocity_size)
    ]
    size = len(bmf_metric)
    explicit_block = [
        bmf_core_metric[row] + bmf_core_generator[row] for row in range(size)
    ] + [bmf_metric[row] + bmf_generator[row] for row in range(size)]
    explicit_x_rank = rank_mod(x_schur, prime) - primal_size
    explicit_y_rank = rank_mod(y_schur, prime) - velocity_size
    explicit_rank = rank_mod(explicit_block, prime) - size
    if explicit_rank != explicit_x_rank + explicit_y_rank:
        raise AssertionError("explicit BMF block-rank sum mismatch")
    return {
        "mass_correction": mass_correction_rank,
        "bmg_pencil": rank_mod(bmg_common, prime),
        "bmg_explicit": bmg_explicit_rank,
        "bmf_pencil": rank_mod(bmf_common, prime),
        "bmf_explicit_x": explicit_x_rank,
        "bmf_explicit_y": explicit_y_rank,
        "bmf_explicit": explicit_rank,
    }


def rational_mass_triangular_control(degree: int, npts: int) -> dict[str, object]:
    """Check the cumulative-defect triangular mass argument over fractions."""
    matrices = uwt.exact_coefficient_matrices(npts, degree)
    core = uwt.exact_cardinal_toeplitz_core(npts, degree, "mass")
    correction = uwt.exact_subtract(matrices["mass"], core)
    cross = [
        [correction[row][column] for column in range(degree)]
        for row in range(degree, 2 * degree - 1)
    ]
    cumulative = [
        [sum(row[column:]) for column in range(1, degree)]
        for row in cross
    ]
    triangular = all(
        cumulative[row][column] == 0
        for row in range(degree - 1)
        for column in range(degree - 1)
        if column < row
    )
    diagonal = [cumulative[index][index] for index in range(degree - 1)]
    partition_null = all(sum(row) == 0 for row in cross)
    leading = range(degree)
    leading_block = [
        [correction[row][column] for column in leading] for row in leading
    ]
    partition_quadratic = sum(
        leading_block[row][column]
        for row in range(degree)
        for column in range(degree)
    )
    return {
        "triangular": triangular,
        "diagonal_nonzero": all(value != 0 for value in diagonal),
        "diagonal_negative": all(value < 0 for value in diagonal),
        "partition_null": partition_null,
        "partition_quadratic_negative": partition_quadratic < 0,
        "diagonal": [f"{value.numerator}/{value.denominator}" for value in diagonal],
        "partition_quadratic": (
            f"{partition_quadratic.numerator}/{partition_quadratic.denominator}"
        ),
    }


def rational_control(degree: int, npts: int) -> dict[str, int]:
    """Evaluate the same five wall-correction ranks over exact rationals."""
    primal = uwt.exact_coefficient_matrices(npts, degree)
    derivative = uwt.exact_coefficient_matrices(npts - 1, degree - 1)
    mass_core = uwt.exact_cardinal_toeplitz_core(npts, degree, "mass")
    advection_core = uwt.exact_cardinal_toeplitz_core(
        npts, degree, "advection"
    )
    derivative_mass_core = uwt.exact_cardinal_toeplitz_core(
        npts - 1, degree - 1, "mass"
    )
    mass_correction = uwt.exact_rank(
        uwt.exact_subtract(primal["mass"], mass_core)
    )
    bmg_metric, bmg_generator = uwt.exact_bmg_acoustic_pencil(
        primal["mass"], primal["advection"]
    )
    bmg_core_metric, bmg_core_generator = uwt.exact_bmg_acoustic_pencil(
        mass_core, advection_core
    )
    bmg_pencil_rank = uwt.exact_rank(
        [
            metric_row + generator_row
            for metric_row, generator_row in zip(
                uwt.exact_subtract(bmg_metric, bmg_core_metric),
                uwt.exact_subtract(bmg_generator, bmg_core_generator),
                strict=True,
            )
        ]
    )
    bmg_size = len(bmg_metric)
    bmg_explicit = (
        uwt.exact_rank(
            [
                bmg_core_metric[row] + bmg_core_generator[row]
                for row in range(bmg_size)
            ]
            + [
                bmg_metric[row] + bmg_generator[row]
                for row in range(bmg_size)
            ]
        )
        - bmg_size
    )
    bmf_metric, bmf_generator = uwt.exact_bmf_acoustic_pencil(
        primal["mass"], derivative["mass"], primal["incidence"]
    )
    bmf_core_metric, bmf_core_generator = uwt.exact_bmf_acoustic_pencil(
        mass_core, derivative_mass_core, uwt.exact_incidence_core(npts)
    )
    bmf_pencil_rank = uwt.exact_rank(
        [
            metric_row + generator_row
            for metric_row, generator_row in zip(
                uwt.exact_subtract(bmf_metric, bmf_core_metric),
                uwt.exact_subtract(bmf_generator, bmf_core_generator),
                strict=True,
            )
        ]
    )
    bmf_size = len(bmf_metric)
    bmf_explicit = (
        uwt.exact_rank(
            [
                bmf_core_metric[row] + bmf_core_generator[row]
                for row in range(bmf_size)
            ]
            + [
                bmf_metric[row] + bmf_generator[row]
                for row in range(bmf_size)
            ]
        )
        - bmf_size
    )
    return {
        "mass_correction": mass_correction,
        "bmg_pencil": bmg_pencil_rank,
        "bmg_explicit": bmg_explicit,
        "bmf_pencil": bmf_pencil_rank,
        "bmf_explicit": bmf_explicit,
    }


def control_sizes(degree: int) -> tuple[int, ...]:
    """Return the overlap, transition, and noninteracting control sizes.

    The first six sizes bracket the two transitions of the corrected
    formulas: the pencil ranks leave the full-row-rank branch at N=4p-2, the
    explicit BMF rank reaches its constant at N=4p-1, and the explicit BMG
    rank reaches its constant at N=4p. The last two sizes repeat the
    noninteracting regime at the resolutions used by the exact atlas.
    """
    return tuple(
        sorted(
            {
                4 * degree - 3,
                4 * degree - 2,
                4 * degree - 1,
                4 * degree,
                4 * degree + 1,
                4 * degree + 2,
                6 * degree + 9,
                8 * degree + 9,
            }
        )
    )


def build_controls(
    degrees: Iterable[int],
    primes: tuple[int, ...] = DEFAULT_PRIMES,
    rational_max_degree: int = 8,
    strict: bool = True,
) -> dict[str, object]:
    """Build exact controls across the requested degree and size set."""
    degrees = tuple(degrees)
    records: list[dict[str, object]] = []
    rational_records: list[dict[str, object]] = []
    deviations: list[dict[str, object]] = []
    for degree in degrees:
        if degree <= rational_max_degree:
            rational = rational_mass_triangular_control(degree, 4 * degree + 3)
            if not all(
                rational[key]
                for key in (
                    "triangular",
                    "diagonal_nonzero",
                    "diagonal_negative",
                    "partition_null",
                    "partition_quadratic_negative",
                )
            ):
                raise AssertionError(f"mass endpoint control failed at p={degree}")
            rational_records.append({"degree": degree, **rational})
        for npts in control_sizes(degree):
            expected = predicted_ranks(degree, npts)
            prime_results = {
                str(prime): modular_control(degree, npts, prime)
                for prime in primes
            }
            record: dict[str, object] = {
                "degree": degree,
                "npts": npts,
                "predicted": expected,
                "prime_results": prime_results,
            }
            observed_lanes = [
                (f"prime {prime}", result)
                for prime, result in prime_results.items()
            ]
            if degree <= rational_max_degree:
                exact = rational_control(degree, npts)
                record["rational_result"] = exact
                observed_lanes.append(("exact rational", exact))
            for lane, observed in observed_lanes:
                mismatch = {
                    key: [expected[key], value]
                    for key, value in observed.items()
                    if value != expected[key]
                }
                if mismatch:
                    deviations.append(
                        {
                            "degree": degree,
                            "npts": npts,
                            "lane": lane,
                            "predicted_versus_observed": mismatch,
                        }
                    )
            records.append(record)
        print(f"all-degree wall ranks p={degree} complete", flush=True)
    if deviations and strict:
        raise AssertionError(
            f"rank formula deviations: {json.dumps(deviations, sort_keys=True)}"
        )
    return {
        "schema": "all-degree-wall-rank-v3",
        "revision": (
            "2026-09-13 core-extraction correction. Earlier revisions read "
            "the same-spacing cardinal comparator from the anchor row N//2 of "
            "the finite clamped matrix. For N at most 4p-2 that row lies "
            "within 2p of the far wall, so its band mixes non-cardinal basis "
            "functions and the comparator was not the cardinal Toeplitz "
            "section of the theorem. Every core is now read from the central "
            "row of a reference matrix of size 4p+9, where the anchor row and "
            "its whole band are translation-invariant. Every reference size at "
            "or above 4p+3 yields the identical cardinal band, 8p+9 included, "
            "so the reference size is a cost choice. The corrected small-N "
            "controls replace the superseded values bmg_pencil=8p-8 and "
            "bmf_pencil=8p-9 at N=4p-2; results for N at least 4p-1 are "
            "unchanged."
        ),
        "degree_range": [min(degrees), max(degrees)],
        "primes": list(primes),
        "rational_max_degree": rational_max_degree,
        "rational_mass_controls": rational_records,
        "piecewise_formulas": {
            "mass_correction": "min(N, 4p-2)",
            "bmg_pencil": "min(2N-2, 8p-6)",
            "bmg_explicit": "min(2N-4, N+4p-6, 8p-6)",
            "bmf_pencil": "min(2N-3, 8p-8)",
            "bmf_explicit_x": "min(N-3, 4p-4)",
            "bmf_explicit_y": "2p-2",
            "bmf_explicit": "min(N+2p-5, 6p-6)",
        },
        "hypotheses": {"degree": "p >= 2", "size": "N >= 4p-3"},
        "noninteracting_range": "N >= 4p",
        "sharp_range": {
            "bmg_pencil": "N >= 4p-1",
            "bmf_pencil": "N >= 4p-1",
            "bmf_explicit": "N >= 4p-1",
            "bmg_explicit": "N >= 4p",
        },
        "noninteracting_ranks": {
            "mass_correction": "4p-2",
            "bmg_pencil": "8p-6",
            "bmg_explicit": "8p-6",
            "bmf_pencil": "8p-8",
            "bmf_explicit_x": "4p-4",
            "bmf_explicit_y": "2p-2",
            "bmf_explicit": "6p-6",
        },
        "transition_at_N_4p_minus_2": {
            "mass_correction": "N",
            "bmg_pencil": "8p-6",
            "bmg_explicit": "8p-8",
            "bmf_pencil": "8p-8",
            "bmf_explicit_x": "4p-5",
            "bmf_explicit_y": "2p-2",
            "bmf_explicit": "6p-7",
        },
        "superseded_transition_at_N_4p_minus_2": {
            "bmg_pencil": "8p-8",
            "bmf_pencil": "8p-9",
            "bmf_explicit_x": "4p-5",
            "bmf_explicit_y": "2p-2",
            "bmf_explicit": "6p-7",
        },
        "deviations": deviations,
        "records": records,
        "source_sha256": {
            "all_degree_wall_rank.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "universal_wall_transfer.py": hashlib.sha256(
                (Path(__file__).with_name("universal_wall_transfer.py")).read_bytes()
            ).hexdigest(),
            "test_all_degree_wall_rank.py": hashlib.sha256(
                (Path(__file__).with_name("test_all_degree_wall_rank.py")).read_bytes()
            ).hexdigest(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-degree", type=int, default=2)
    parser.add_argument("--max-degree", type=int, default=10)
    parser.add_argument("--rational-max-degree", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-deviations",
        action="store_true",
        help="record formula deviations in the artifact instead of failing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_degree < 2 or args.max_degree < args.min_degree:
        raise ValueError("degree range must satisfy 2 <= min <= max")
    degrees = tuple(range(args.min_degree, args.max_degree + 1))
    result = build_controls(
        degrees,
        rational_max_degree=args.rational_max_degree,
        strict=not args.allow_deviations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
