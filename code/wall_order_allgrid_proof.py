#!/usr/bin/env python3
"""Exact and interval certificates for the all-grid wall-order proof.

This module is deliberately independent of the floating operator assembly used
by the manuscript.  Its arithmetic layer uses :class:`fractions.Fraction`.
Floating-point root finders only propose brackets; every accepted bracket is
certified by exact rational sign tests and a degree count.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import math
from typing import Any, Iterable

import numpy as np


DEGREES = (4, 6, 8)


def _scaled_open_uniform_knots(npts: int, degree: int) -> list[Fraction]:
    spans = int(npts) - int(degree)
    if spans < 1:
        raise ValueError("npts must be at least degree + 1")
    return (
        [Fraction(0)] * (degree + 1)
        + [Fraction(j) for j in range(1, spans)]
        + [Fraction(spans)] * (degree + 1)
    )


def _greville_nodes(knots: list[Fraction], degree: int) -> list[Fraction]:
    nbasis = len(knots) - degree - 1
    return [
        sum(knots[i + 1:i + degree + 1], Fraction(0)) / degree
        for i in range(nbasis)
    ]


def _bspline_values(
    knots: list[Fraction], degree: int, point: Fraction
) -> list[Fraction]:
    values = [Fraction(0) for _ in range(len(knots) - 1)]
    if point == knots[-1]:
        positive = [i for i in range(len(knots) - 1) if knots[i] < knots[i + 1]]
        values[positive[-1]] = Fraction(1)
    else:
        for i in range(len(values)):
            if knots[i] <= point < knots[i + 1]:
                values[i] = Fraction(1)
                break
    for order in range(1, degree + 1):
        next_values: list[Fraction] = []
        for i in range(len(values) - 1):
            left_den = knots[i + order] - knots[i]
            right_den = knots[i + order + 1] - knots[i + 1]
            left = (
                Fraction(0)
                if left_den == 0
                else (point - knots[i]) * values[i] / left_den
            )
            right = (
                Fraction(0)
                if right_den == 0
                else (knots[i + order + 1] - point) * values[i + 1] / right_den
            )
            next_values.append(left + right)
        values = next_values
    return values


@lru_cache(maxsize=None)
def exact_collocation_matrix(degree: int, npts: int) -> list[list[Fraction]]:
    knots = _scaled_open_uniform_knots(npts, degree)
    nodes = _greville_nodes(knots, degree)
    return [_bspline_values(knots, degree, node) for node in nodes]


def _fraction_solve(
    matrix: list[list[Fraction]], rhs: list[list[Fraction]]
) -> list[list[Fraction]]:
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be nonempty and square")
    nrhs = len(rhs[0])
    augmented = [list(matrix[i]) + list(rhs[i]) for i in range(n)]
    for column in range(n):
        pivot = next(
            (row for row in range(column, n) if augmented[row][column] != 0),
            None,
        )
        if pivot is None:
            raise ValueError("singular exact matrix")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [entry / scale for entry in augmented[column]]
        for row in range(n):
            if row == column or augmented[row][column] == 0:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                augmented[row][j] - factor * augmented[column][j]
                for j in range(n + nrhs)
            ]
    return [row[n:] for row in augmented]


def _local_basis_polynomials(
    knots: list[Fraction], degree: int
) -> list[tuple[Fraction, list[list[Fraction]]]]:
    nbasis = len(knots) - degree - 1
    samples = [Fraction(j + 1, degree + 2) for j in range(degree + 1)]
    vandermonde = [
        [sample**power for power in range(degree + 1)] for sample in samples
    ]
    spans: list[tuple[Fraction, list[list[Fraction]]]] = []
    for left, right in zip(knots[:-1], knots[1:]):
        if right <= left:
            continue
        width = right - left
        sampled = [
            _bspline_values(knots, degree, left + width * sample)
            for sample in samples
        ]
        by_power = _fraction_solve(vandermonde, sampled)
        by_basis = [
            [by_power[power][basis] for power in range(degree + 1)]
            for basis in range(nbasis)
        ]
        spans.append((width, by_basis))
    return spans


@lru_cache(maxsize=None)
def exact_mass_and_advection(
    degree: int, npts: int
) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    """Return mass and ``int B_i B'_j`` on the mesh scaled to unit spans."""
    knots = _scaled_open_uniform_knots(npts, degree)
    nbasis = npts
    mass = [[Fraction(0) for _ in range(nbasis)] for _ in range(nbasis)]
    advection = [[Fraction(0) for _ in range(nbasis)] for _ in range(nbasis)]
    for width, coefficients in _local_basis_polynomials(knots, degree):
        active = [i for i, coefficient in enumerate(coefficients) if any(coefficient)]
        for i in active:
            ci = coefficients[i]
            for j in active:
                cj = coefficients[j]
                mass[i][j] += width * sum(
                    (
                        ci[a] * cj[b] / Fraction(a + b + 1)
                        for a in range(degree + 1)
                        for b in range(degree + 1)
                    ),
                    Fraction(0),
                )
                advection[i][j] += sum(
                    (
                        ci[a] * b * cj[b] / Fraction(a + b)
                        for a in range(degree + 1)
                        for b in range(1, degree + 1)
                    ),
                    Fraction(0),
                )
    return mass, advection


def _trim_polynomial(coefficients: Iterable[Fraction]) -> list[Fraction]:
    out = list(coefficients)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    while len(out) > 1 and out[0] == 0:
        out.pop(0)
    return out


def _interior_symbol_coefficients(
    degree: int, kind: str
) -> tuple[list[Fraction], list[int]]:
    npts = 6 * degree + 9
    row = 3 * degree + 4
    if kind == "collocation":
        matrix = exact_collocation_matrix(degree, npts)
    elif kind == "mass":
        matrix = exact_mass_and_advection(degree, npts)[0]
    else:
        raise ValueError(kind)
    nonzero = [j for j, value in enumerate(matrix[row]) if value != 0]
    offsets = list(range(nonzero[0] - row, nonzero[-1] - row + 1))
    coefficients = [matrix[row][row + offset] for offset in offsets]
    return coefficients, offsets


def _poly_value(coefficients: list[Fraction], x: Fraction) -> Fraction:
    value = Fraction(0)
    for coefficient in reversed(coefficients):
        value = value * x + coefficient
    return value


def _candidate_negative_roots(coefficients: list[Fraction]) -> list[float]:
    floating = np.array([float(value) for value in coefficients], dtype=float)
    roots = np.roots(floating[::-1])
    candidates = sorted(
        (float(root.real) for root in roots if abs(root.imag) < 1.0e-7 and -1.0 < root.real < 0.0)
    )
    return candidates


def _exact_root_bracket(
    coefficients: list[Fraction], candidate: float
) -> tuple[Fraction, Fraction]:
    scale = max(abs(candidate), 1.0e-18)
    for exponent in range(8, 17):
        radius = scale * 10.0 ** (-exponent) + 10.0 ** (-exponent - 18)
        left = Fraction(str(candidate - radius))
        right = Fraction(str(candidate + radius))
        if left < -1:
            left = Fraction(-1)
        if right >= 0:
            right = Fraction(-1, 10**30)
        fleft = _poly_value(coefficients, left)
        fright = _poly_value(coefficients, right)
        if fleft != 0 and fright != 0 and (fleft < 0) != (fright < 0):
            # The p=8 BMG half-line system couples the mass and collocation
            # decaying modes and amplifies characteristic-root uncertainty by
            # roughly forty decimal digits.  A 110-step bracket is therefore
            # adequate for the separate symbols but not for the coupled
            # endpoint constant.  These are exact rational bisections, so the
            # tighter bracket adds no floating-point assumption to the proof.
            for _ in range(360):
                midpoint = (left + right) / 2
                fmid = _poly_value(coefficients, midpoint)
                if fmid == 0:
                    return midpoint, midpoint
                if (fleft < 0) != (fmid < 0):
                    right = midpoint
                    fright = fmid
                else:
                    left = midpoint
                    fleft = fmid
            return left, right
    raise RuntimeError(f"could not certify root bracket near {candidate}")


def _root_block(degree: int, kind: str) -> dict[str, Any]:
    coefficients, offsets = _interior_symbol_coefficients(degree, kind)
    expected_inside = degree // 2 if kind == "collocation" else degree
    candidates = _candidate_negative_roots(coefficients)

    # Extremely small mass-symbol roots can be lost by a double-precision
    # polynomial root finder.  Exact sign scans on a logarithmic grid recover
    # them before the final exact degree count.
    if len(candidates) < expected_inside:
        scan = [Fraction(-1, 10**power) for power in range(0, 40)]
        for left, right in zip(scan[:-1], scan[1:]):
            fleft = _poly_value(coefficients, left)
            fright = _poly_value(coefficients, right)
            if fleft != 0 and fright != 0 and (fleft < 0) != (fright < 0):
                candidates.append(0.5 * (float(left) + float(right)))
    candidates = sorted(candidates)
    brackets = [_exact_root_bracket(coefficients, root) for root in candidates]
    brackets = sorted(set(brackets))
    if len(brackets) != expected_inside:
        raise RuntimeError(
            f"{kind} p={degree}: found {len(brackets)} of {expected_inside} inside roots"
        )
    if any(right >= 0 or left < -1 for left, right in brackets):
        raise RuntimeError("inside-root bracket left (-1,0)")
    if any(
        _poly_value(coefficients, left) * _poly_value(coefficients, right) >= 0
        for left, right in brackets
    ):
        raise RuntimeError("root bracket lacks an exact sign change")
    palindromic = coefficients == list(reversed(coefficients))
    if not palindromic:
        raise RuntimeError("interior symbol is not exactly palindromic")
    # Every inside interval contains a distinct real root. Palindromicity gives
    # a reciprocal root outside the closed unit disk. Their total equals the
    # polynomial degree, so no root remains unaccounted for or on |z|=1.
    all_accounted = 2 * len(brackets) == len(coefficients) - 1
    return {
        "kind": kind,
        "degree": degree,
        "offsets": offsets,
        "coefficients": [str(value) for value in coefficients],
        "palindromic": palindromic,
        "inside_intervals": [[str(left), str(right)] for left, right in brackets],
        "all_roots_accounted_for": all_accounted,
        "no_unit_circle_root": all_accounted,
        "maximum_inside_modulus_upper": str(max(-left for left, _ in brackets)),
    }


@lru_cache(maxsize=None)
def characteristic_root_certificates(degree: int) -> dict[str, Any]:
    if degree not in DEGREES:
        raise ValueError(f"degree must be one of {DEGREES}")
    return {
        "collocation": _root_block(degree, "collocation"),
        "mass": _root_block(degree, "mass"),
    }


def _checkerboard_matrix(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    return [
        [((-1) ** (i + j)) * value for j, value in enumerate(row)]
        for i, row in enumerate(matrix)
    ]


def _matvec(
    matrix: list[list[Fraction]], vector: list[Fraction]
) -> list[Fraction]:
    return [
        sum((value * entry for value, entry in zip(row, vector)), Fraction(0))
        for row in matrix
    ]


def _transpose(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    return [list(column) for column in zip(*matrix)]


def _rationalize_positive_vector(
    values: np.ndarray, *, scale: Fraction = Fraction(2), denominator: int = 10**10
) -> list[Fraction]:
    out = [
        Fraction(float(scale) * float(value)).limit_denominator(denominator)
        for value in values
    ]
    if any(value <= 0 for value in out):
        raise RuntimeError("proposed checkerboard supersolution is not positive")
    return out


def _finite_supersolution(
    degree: int, npts: int, *, transpose: bool
) -> tuple[list[Fraction], Fraction]:
    collocation = exact_collocation_matrix(degree, npts)
    checkerboard = _checkerboard_matrix(collocation)
    operator = _transpose(checkerboard) if transpose else checkerboard
    floating = np.array(
        [[float(value) for value in row] for row in operator], dtype=float
    )
    exact_rhs_solution = np.linalg.solve(floating, np.ones(npts, dtype=float))
    vector = _rationalize_positive_vector(exact_rhs_solution)
    residual = [value - 1 for value in _matvec(operator, vector)]
    minimum = min(residual)
    if minimum <= 0:
        # A larger rational scaling retains the proposed shape but gives a full
        # unit of exact slack.  Floating arithmetic never decides acceptance.
        vector = [2 * value for value in vector]
        residual = [value - 1 for value in _matvec(operator, vector)]
        minimum = min(residual)
    if minimum <= 0:
        raise RuntimeError(f"failed exact finite supersolution at p={degree}, N={npts}")
    return vector, minimum


def _alternating_symbol_value(degree: int) -> Fraction:
    coefficients, offsets = _interior_symbol_coefficients(degree, "collocation")
    return sum(
        (
            coefficient if offset % 2 == 0 else -coefficient
            for coefficient, offset in zip(coefficients, offsets)
        ),
        Fraction(0),
    )


@lru_cache(maxsize=None)
def _template_profile(
    degree: int, transpose: bool, depth: int
) -> tuple[Fraction, ...]:
    # The boundary profile must be independent of N for the translation
    # argument.  This fixed reference is far enough that the sampled left
    # profile is insensitive to the reflected right boundary at binary64
    # proposal accuracy; exact inequalities below decide validity.
    reference_n = 4 * (depth + degree + 4) + 1
    collocation = exact_collocation_matrix(degree, reference_n)
    checkerboard = _checkerboard_matrix(collocation)
    operator = _transpose(checkerboard) if transpose else checkerboard
    floating = np.array(
        [[float(value) for value in row] for row in operator], dtype=float
    )
    solution = np.linalg.solve(floating, np.ones(reference_n, dtype=float))
    boundary_values = _rationalize_positive_vector(solution[:depth])
    core = Fraction(2, 1) / _alternating_symbol_value(degree)
    return tuple(boundary_values + [core])


def _template_vector(
    degree: int, npts: int, *, transpose: bool, depth: int
) -> list[Fraction]:
    profile = _template_profile(degree, transpose, depth)
    return [profile[min(i, npts - 1 - i, depth)] for i in range(npts)]


def _exact_template_audit(
    degree: int, *, transpose: bool, depth: int, threshold: int
) -> tuple[Fraction, Fraction]:
    minimum_margin: Fraction | None = None
    maximum_entry = Fraction(0)
    # Both parities and every possible band interaction with the transition
    # are represented in this finite window.  Beyond it, rows are translations
    # of a checked transition row or a pure Toeplitz row acting on the constant
    # core profile.
    for npts in range(threshold, threshold + 2 * degree + 6):
        collocation = exact_collocation_matrix(degree, npts)
        checkerboard = _checkerboard_matrix(collocation)
        operator = _transpose(checkerboard) if transpose else checkerboard
        vector = _template_vector(
            degree, npts, transpose=transpose, depth=depth
        )
        margins = [value - 1 for value in _matvec(operator, vector)]
        local_minimum = min(margins)
        if local_minimum <= 0:
            raise RuntimeError(
                f"template supersolution failed at p={degree}, N={npts}, "
                f"transpose={transpose}: {local_minimum}"
            )
        minimum_margin = (
            local_minimum
            if minimum_margin is None
            else min(minimum_margin, local_minimum)
        )
        maximum_entry = max(maximum_entry, max(vector))
    if minimum_margin is None:
        raise RuntimeError("empty template audit")
    return minimum_margin, maximum_entry


def uniform_stability_certificate(degree: int) -> dict[str, Any]:
    """Return an exact supersolution proof for every ``N >= degree + 1``.

    Total positivity of B-spline collocation gives checkerboard nonnegativity
    of the inverse by the cofactor formula.  If ``A=S B S`` and ``A v >= 1``
    with ``v>0``, then ``A^{-1}1 <= v``.  Applying this once to ``A`` and once
    to ``A^T`` bounds the infinity and one norms of ``B^{-1}``.
    """
    if degree not in DEGREES:
        raise ValueError(f"degree must be one of {DEGREES}")
    depth = degree + 16
    threshold = 2 * (depth + degree + 2) + 1
    inf_margin, asymptotic_inf = _exact_template_audit(
        degree, transpose=False, depth=depth, threshold=threshold
    )
    one_margin, asymptotic_one = _exact_template_audit(
        degree, transpose=True, depth=depth, threshold=threshold
    )

    inverse_inf_bound = asymptotic_inf
    inverse_one_bound = asymptotic_one
    minimum_margin = min(inf_margin, one_margin)
    audited_small_grids = 0
    for npts in range(degree + 1, threshold):
        for transpose in (False, True):
            vector, margin = _finite_supersolution(
                degree, npts, transpose=transpose
            )
            minimum_margin = min(minimum_margin, margin)
            if transpose:
                inverse_one_bound = max(inverse_one_bound, max(vector))
            else:
                inverse_inf_bound = max(inverse_inf_bound, max(vector))
        audited_small_grids += 1

    half_bandwidth = degree - 1
    core_margin = (
        _alternating_symbol_value(degree)
        * (Fraction(2) / _alternating_symbol_value(degree))
        - 1
    )
    minimum_margin = min(minimum_margin, core_margin)
    return {
        "degree": degree,
        "all_N_ge_p_plus_1": True,
        "inverse_checkerboard_nonnegative": True,
        "checkerboard_reason": (
            "B-spline collocation is totally positive; complementary minors "
            "in the cofactor formula give (-1)^(i+j)(B^-1)_ij >= 0"
        ),
        "template_depth": depth,
        "large_grid_threshold": threshold,
        "large_grid_window_checked": [threshold, threshold + 2 * degree + 5],
        "maximum_half_bandwidth": half_bandwidth,
        "translation_argument": (
            "past the checked window each row is either a reflected boundary/"
            "transition row already checked or an interior Toeplitz row acting "
            "on the constant core"
        ),
        "small_grid_count_checked": audited_small_grids,
        "minimum_exact_margin": str(minimum_margin),
        "inverse_inf_bound": str(inverse_inf_bound),
        "inverse_one_bound": str(inverse_one_bound),
        "asymptotic_inverse_inf_bound": str(asymptotic_inf),
        "asymptotic_inverse_one_bound": str(asymptotic_one),
        "interior_alternating_symbol": str(_alternating_symbol_value(degree)),
    }


class _Interval:
    __slots__ = ("lo", "hi")
    # Exact root bisection supplies more than 100 guard digits to the coupled
    # p=8 BMG solve.  Retaining 120 outward-rounded decimal digits is enough to
    # survive its conditioning while avoiding the extreme rational-integer
    # growth caused by the former 180-digit arithmetic.  Acceptance still
    # requires every pivot interval and the final endpoint interval to exclude
    # zero.
    ROUND_DIGITS = 120

    def __init__(self, lo: Fraction | int, hi: Fraction | int | None = None) -> None:
        self.lo = Fraction(lo)
        self.hi = self.lo if hi is None else Fraction(hi)
        if self.lo > self.hi:
            raise ValueError("reversed interval")

    @staticmethod
    def coerce(value: "_Interval | Fraction | int") -> "_Interval":
        return value if isinstance(value, _Interval) else _Interval(Fraction(value))

    @classmethod
    def rounded(cls, lo: Fraction, hi: Fraction) -> "_Interval":
        scale = 10**cls.ROUND_DIGITS
        lower_integer = (lo.numerator * scale) // lo.denominator
        upper_integer = -((-hi.numerator * scale) // hi.denominator)
        return cls(Fraction(lower_integer, scale), Fraction(upper_integer, scale))

    def __add__(self, other: "_Interval | Fraction | int") -> "_Interval":
        rhs = self.coerce(other)
        return self.rounded(self.lo + rhs.lo, self.hi + rhs.hi)

    __radd__ = __add__

    def __neg__(self) -> "_Interval":
        return _Interval(-self.hi, -self.lo)

    def __sub__(self, other: "_Interval | Fraction | int") -> "_Interval":
        return self + (-self.coerce(other))

    def __rsub__(self, other: "_Interval | Fraction | int") -> "_Interval":
        return self.coerce(other) - self

    def __mul__(self, other: "_Interval | Fraction | int") -> "_Interval":
        rhs = self.coerce(other)
        products = (
            self.lo * rhs.lo,
            self.lo * rhs.hi,
            self.hi * rhs.lo,
            self.hi * rhs.hi,
        )
        return self.rounded(min(products), max(products))

    __rmul__ = __mul__

    def reciprocal(self) -> "_Interval":
        if self.lo <= 0 <= self.hi:
            raise ZeroDivisionError("interval contains zero")
        return self.rounded(1 / self.hi, 1 / self.lo)

    def __truediv__(self, other: "_Interval | Fraction | int") -> "_Interval":
        return self * self.coerce(other).reciprocal()

    def __pow__(self, exponent: int) -> "_Interval":
        if exponent < 0:
            return (self.reciprocal()) ** (-exponent)
        result = _Interval(1)
        base = self
        power = int(exponent)
        while power:
            if power & 1:
                result = result * base
            base = base * base
            power //= 2
        return result

    def midpoint_float(self) -> float:
        return 0.5 * (float(self.lo) + float(self.hi))

    def magnitude_upper(self) -> Fraction:
        return max(abs(self.lo), abs(self.hi))


def _interval_sum(values: Iterable[_Interval]) -> _Interval:
    total = _Interval(0)
    for value in values:
        total = total + value
    return total


def _interval_linear_solve(
    matrix: list[list[_Interval]], rhs: list[_Interval]
) -> list[_Interval]:
    n = len(matrix)
    a = [[_Interval(entry.lo, entry.hi) for entry in row] for row in matrix]
    b = [_Interval(entry.lo, entry.hi) for entry in rhs]
    for column in range(n):
        pivot = max(
            range(column, n), key=lambda row: abs(a[row][column].midpoint_float())
        )
        if a[pivot][column].lo <= 0 <= a[pivot][column].hi:
            raise RuntimeError(f"interval pivot contains zero at column {column}")
        if pivot != column:
            a[column], a[pivot] = a[pivot], a[column]
            b[column], b[pivot] = b[pivot], b[column]
        for row in range(column + 1, n):
            if a[row][column].lo == 0 and a[row][column].hi == 0:
                continue
            factor = a[row][column] / a[column][column]
            for j in range(column, n):
                a[row][j] = a[row][j] - factor * a[column][j]
            b[row] = b[row] - factor * b[column]
    solution = [_Interval(0) for _ in range(n)]
    for row in range(n - 1, -1, -1):
        value = b[row] - _interval_sum(
            a[row][j] * solution[j] for j in range(row + 1, n)
        )
        solution[row] = value / a[row][row]
    return solution


def _root_intervals(degree: int, kind: str) -> list[_Interval]:
    block = characteristic_root_certificates(degree)[kind]
    return [
        _Interval(Fraction(left), Fraction(right))
        for left, right in block["inside_intervals"]
    ]


def _mode_value(
    boundary_values: list[_Interval], amplitudes: list[_Interval],
    roots: list[_Interval], tail_start: int, index: int,
) -> _Interval:
    if index < tail_start:
        return boundary_values[index]
    return _interval_sum(
        amplitude * (root ** index)
        for amplitude, root in zip(amplitudes, roots)
    )


def _bmf_halfline_weights(degree: int) -> tuple[list[_Interval], list[_Interval], int]:
    roots = _root_intervals(degree, "collocation")
    root_count = len(roots)
    tail_start = degree
    equation_count = tail_start + root_count
    npts = 8 * degree + 25
    collocation = exact_collocation_matrix(degree, npts)
    matrix: list[list[_Interval]] = []
    rhs: list[_Interval] = []
    for column in range(equation_count):
        row: list[_Interval] = []
        for index in range(tail_start):
            row.append(_Interval(collocation[index][column]))
        for root in roots:
            row.append(_interval_sum(
                _Interval(collocation[index][column]) * (root ** index)
                for index in range(tail_start, npts)
                if collocation[index][column] != 0
            ))
        matrix.append(row)
        rhs.append(_Interval(degree * (-1 if column == 0 else 1 if column == 1 else 0)))
    solution = _interval_linear_solve(matrix, rhs)
    return solution[:tail_start], solution[tail_start:], tail_start


def _mass_halfline_solution(
    degree: int,
) -> tuple[list[_Interval], list[_Interval], int, list[list[Fraction]], list[list[Fraction]]]:
    roots = _root_intervals(degree, "mass")
    tail_start = degree
    equation_count = 2 * degree
    npts = 10 * degree + 25
    mass, advection = exact_mass_and_advection(degree, npts)
    matrix: list[list[_Interval]] = []
    rhs: list[_Interval] = []
    for equation in range(equation_count):
        row: list[_Interval] = []
        for index in range(tail_start):
            row.append(_Interval(mass[equation][index]))
        for root in roots:
            row.append(_interval_sum(
                _Interval(mass[equation][index]) * (root ** index)
                for index in range(tail_start, npts)
                if mass[equation][index] != 0
            ))
        matrix.append(row)
        rhs.append(_Interval(1 if equation == 0 else 0))
    solution = _interval_linear_solve(matrix, rhs)
    return (
        solution[:tail_start], solution[tail_start:], tail_start,
        mass, advection,
    )


def _bmg_halfline_weights(degree: int) -> tuple[list[_Interval], list[_Interval], list[_Interval], int]:
    z_boundary, z_amplitudes, z_tail_start, _mass, advection = (
        _mass_halfline_solution(degree)
    )
    mass_roots = _root_intervals(degree, "mass")
    collocation_roots = _root_intervals(degree, "collocation")
    all_roots = collocation_roots + mass_roots
    q_tail_start = 2 * degree
    equation_count = q_tail_start + len(all_roots)
    npts = len(advection)
    collocation = exact_collocation_matrix(degree, npts)

    def z(index: int) -> _Interval:
        return _mode_value(
            z_boundary, z_amplitudes, mass_roots, z_tail_start, index
        )

    def w(column: int) -> _Interval:
        return _interval_sum(
            _Interval(advection[index][column]) * z(index)
            for index in range(npts)
            if advection[index][column] != 0
        )

    matrix: list[list[_Interval]] = []
    rhs: list[_Interval] = []
    for column in range(equation_count):
        row: list[_Interval] = []
        for index in range(q_tail_start):
            row.append(_Interval(collocation[index][column]))
        for root in all_roots:
            row.append(_interval_sum(
                _Interval(collocation[index][column]) * (root ** index)
                for index in range(q_tail_start, npts)
                if collocation[index][column] != 0
            ))
        matrix.append(row)
        rhs.append(w(column))
    solution = _interval_linear_solve(matrix, rhs)
    return (
        solution[:q_tail_start], solution[q_tail_start:],
        all_roots, q_tail_start,
    )


def _geometric_polynomial_tail_bound(
    rate: Fraction, first_index: int, power: int
) -> Fraction:
    first = Fraction(first_index) ** power * rate**first_index
    ratio = rate * Fraction(first_index + 1, first_index) ** power
    if ratio >= 1:
        raise RuntimeError("tail ratio is not contractive")
    return first / (1 - ratio)


def _endpoint_constant_interval(
    degree: int, boundary: list[_Interval], amplitudes: list[_Interval],
    roots: list[_Interval], tail_start: int,
) -> _Interval:
    npts = 10 * degree + 25
    nodes = _greville_nodes(_scaled_open_uniform_knots(npts, degree), degree)
    power = degree + 1
    result = _interval_sum(
        boundary[index] * _Interval(nodes[index] ** power)
        for index in range(tail_start)
    )
    cutoff = 260
    for amplitude, root in zip(amplitudes, roots):
        partial = _interval_sum(
            (root ** index) * _Interval(
                (Fraction(2 * index - degree + 1, 2)) ** power
            )
            for index in range(tail_start, cutoff)
        )
        rate = root.magnitude_upper()
        tail = _geometric_polynomial_tail_bound(rate, cutoff, power)
        series = partial + _Interval(-tail, tail)
        result = result + amplitude * series
    return result


def _decimal_outward(interval: _Interval, digits: int = 14) -> _Interval:
    scale = 10**digits
    lower_integer = (interval.lo.numerator * scale) // interval.lo.denominator
    upper_integer = -(
        (-interval.hi.numerator * scale) // interval.hi.denominator
    )
    return _Interval(
        Fraction(lower_integer, scale), Fraction(upper_integer, scale)
    )


def halfline_endpoint_certificate(family: str, degree: int) -> dict[str, Any]:
    family = family.upper()
    if degree not in DEGREES or family not in ("BMF", "BMG"):
        raise ValueError("family must be BMF/BMG and degree one of 4, 6, 8")
    if family == "BMF":
        boundary, amplitudes, tail_start = _bmf_halfline_weights(degree)
        roots = _root_intervals(degree, "collocation")
    else:
        boundary, amplitudes, roots, tail_start = _bmg_halfline_weights(degree)
    constant = _endpoint_constant_interval(
        degree, boundary, amplitudes, roots, tail_start
    )
    reported_constant = _decimal_outward(constant)
    rate = max(root.magnitude_upper() for root in roots)
    excludes_zero = bool(constant.hi < 0 or constant.lo > 0)
    if not excludes_zero:
        raise RuntimeError(
            f"{family}{degree} endpoint interval does not exclude zero: "
            f"[{float(constant.lo)}, {float(constant.hi)}]"
        )
    return {
        "family": family,
        "degree": degree,
        "constant_interval": [
            str(reported_constant.lo), str(reported_constant.hi)
        ],
        "constant_interval_decimal": [
            float(reported_constant.lo), float(reported_constant.hi)
        ],
        "interval_excludes_zero": excludes_zero,
        "tail_start": tail_start,
        "tail_mode_count": len(roots),
        "localization_rate_upper": str(rate),
        "endpoint_weights_localize_exponentially": bool(rate < 1),
        "constant_definition": (
            "limit of h^{-p} times the endpoint first-derivative error on "
            "u(x)=x^(p+1); the leading coefficient for a smooth field is "
            "this constant divided by (p+1)!"
        ),
    }
