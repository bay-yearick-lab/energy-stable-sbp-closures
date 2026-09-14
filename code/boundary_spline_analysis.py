#!/usr/bin/env python3
from __future__ import annotations

import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import minimize

Array = np.ndarray

COMPARISON_FAMILIES: tuple[tuple[str, str, str], ...] = (
    ('E', 'E', 'explicit'),
    ('C', 'C', 'compact'),
    ('SBP', 'SBP', 'SBP'),
    ('DRP', 'DRP', 'DRP'),
    ('BMF', 'B', 'BMF'),
    ('BMG', 'I', 'BMG'),
)
FAMILY_DISPLAY_NAMES = {family: display for family, _kind, display in COMPARISON_FAMILIES}
FAMILY_KINDS = {family: kind for family, kind, _display in COMPARISON_FAMILIES}
NUMERAL_MEANINGS = {
    'E': 'order',
    'C': 'order',
    'SBP': 'order',
    'DRP': 'order',
    'BMF': 'degree',
    'BMG': 'degree',
}
DEFAULT_COLORS = {
    'E': '#6a6a6a',
    'C': '#1f77b4',
    'SBP': '#9467bd',
    'DRP': '#ff7f0e',
    'BMF': '#d62728',
    'BMG': '#2ca02c',
}
DEFAULT_MARKERS = {
    'E': 'o',
    'C': 's',
    'SBP': 'P',
    'DRP': 'X',
    'BMF': '^',
    'BMG': 'D',
}
DEFAULT_LINESTYLES = {
    'E': '-',
    'C': '--',
    'SBP': '-.',
    'DRP': ':',
    'BMF': (0, (6.0, 1.6, 1.2, 1.6)),
    'BMG': (0, (1.2, 1.2)),
}
DEFAULT_XOFFSETS = {
    'E': -0.25,
    'C': -0.15,
    'SBP': -0.05,
    'DRP': 0.05,
    'BMF': 0.15,
    'BMG': 0.25,
}


def _as_2d(u: Array) -> tuple[Array, bool]:
    arr = np.asarray(u, dtype=float)
    if arr.ndim == 1:
        return arr[:, None], True
    if arr.ndim == 2:
        return arr, False
    raise ValueError("input must be 1D or 2D")


def family_order() -> tuple[str, ...]:
    return tuple(family for family, _kind, _display in COMPARISON_FAMILIES)


def scheme_family(label: str) -> str:
    upper = str(label).upper()
    for family in sorted(FAMILY_KINDS, key=len, reverse=True):
        if upper.startswith(family):
            return family
    raise ValueError(f'unknown scheme label: {label}')


def scheme_order(label: str) -> int:
    family = scheme_family(label)
    return int(str(label)[len(family):])


def scheme_kind(label: str) -> str:
    return FAMILY_KINDS[scheme_family(label)]


def scheme_display_name(label_or_family: str) -> str:
    family = label_or_family if label_or_family in FAMILY_DISPLAY_NAMES else scheme_family(label_or_family)
    return FAMILY_DISPLAY_NAMES[family]


def numeral_means(label: str) -> str:
    return NUMERAL_MEANINGS[scheme_family(label)]


def comparison_block(order: int, families: Sequence[str] | None = None) -> list[tuple[str, str, int]]:
    selected = tuple(families) if families is not None else family_order()
    return [(f'{family}{int(order)}', FAMILY_KINDS[family], int(order)) for family in selected]


def comparison_labels(order: int, families: Sequence[str] | None = None) -> list[str]:
    return [label for label, _kind, _order in comparison_block(order, families=families)]


def family_style_maps() -> tuple[dict[str, str], dict[str, str], dict[str, str | tuple[float, ...] | tuple[int, tuple[float, ...]]]]:
    return DEFAULT_COLORS.copy(), DEFAULT_MARKERS.copy(), DEFAULT_LINESTYLES.copy()


@dataclass(frozen=True)
class WorkModel:
    grid_apply_flops: int
    solve_flops: int
    total_flops: int
    storage_nnz: int


@dataclass(frozen=True)
class WallModeResponse:
    mode: int
    ppw: float
    kappa_ratio: float
    contamination_l2: float
    rel_max: float
    rel_l2: float


class Operator1D:
    name: str
    order: int

    @property
    def x(self) -> Array:
        raise NotImplementedError

    @property
    def matrix(self) -> Array:
        raise NotImplementedError

    @property
    def work_model(self) -> WorkModel:
        raise NotImplementedError

    def differentiate(self, u: Array) -> Array:
        raise NotImplementedError

    def second_derivative_matrix(self) -> Array:
        raise ValueError(f"{self.name} does not provide a second-derivative operator")

    def field_and_derivative(self, points: Array, u: Array) -> tuple[Array, Array]:
        raise NotImplementedError


@lru_cache(maxsize=None)
def uniform_grid(npts: int, length: float = 1.0) -> Array:
    return np.linspace(0.0, float(length), int(npts), dtype=float)


@lru_cache(maxsize=None)
def trapezoidal_weights(npts: int, length: float = 1.0) -> Array:
    x = uniform_grid(npts, length)
    h = x[1] - x[0]
    w = np.ones(npts, dtype=float)
    w[0] = 0.5
    w[-1] = 0.5
    return h * w


def nodal_trapezoidal_weights(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a one-dimensional grid with at least two points")
    dx = np.diff(x)
    if np.any(dx <= 0.0):
        raise ValueError("x must be strictly increasing")
    w = np.empty_like(x)
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    if x.size > 2:
        w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return w


def reference_spacing(x: Array) -> float:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a one-dimensional grid with at least two points")
    dx = np.diff(x)
    if np.any(dx <= 0.0):
        raise ValueError("x must be strictly increasing")
    return float(np.max(dx))


@lru_cache(maxsize=None)
def endpoint_flux_matrix(npts: int) -> Array:
    mat = np.zeros((npts, npts), dtype=float)
    mat[-1, -1] = 1.0
    mat[0, 0] = -1.0
    return mat



def fd_weights(x0: float, x_nodes: Array, deriv_order: int = 1) -> Array:
    x_nodes = np.asarray(x_nodes, dtype=float)
    n = x_nodes.size
    vand = np.vander(x_nodes - x0, N=n, increasing=True).T
    rhs = np.zeros(n, dtype=float)
    rhs[deriv_order] = math.factorial(deriv_order)
    return np.linalg.solve(vand, rhs)


@lru_cache(maxsize=None)
def explicit_fd_matrix(npts: int, order: int = 6, length: float = 1.0) -> tuple[Array, Array]:
    if order % 2 or order < 2:
        raise ValueError("order must be a positive even integer")
    x = uniform_grid(npts, length)
    radius = order // 2
    stencil_size = order + 1
    dmat = np.zeros((npts, npts), dtype=float)
    for j in range(radius):
        dmat[j, :stencil_size] = fd_weights(float(x[j]), x[:stencil_size], 1)
    for j in range(radius, npts - radius):
        idx = np.arange(j - radius, j + radius + 1)
        dmat[j, idx] = fd_weights(float(x[j]), x[idx], 1)
    for j in range(npts - radius, npts):
        dmat[j, -stencil_size:] = fd_weights(float(x[j]), x[-stencil_size:], 1)
    return x.copy(), dmat



def drp_interior_coefficients(order: int) -> tuple[float, ...]:
    if order == 4:
        return (0.67880327, -0.08962729)
    if order == 6:
        return (0.77793115, -0.17388691, 0.02338713)
    if order == 8:
        return (0.84149635, -0.24532989, 0.06081891, -0.00839807)
    raise ValueError("supported DRP orders are 4, 6, and 8")


@lru_cache(maxsize=None)
def drp_fd_matrix(npts: int, order: int = 6, length: float = 1.0) -> tuple[Array, Array]:
    x = uniform_grid(npts, length)
    h = x[1] - x[0]
    upper = drp_interior_coefficients(order)
    radius = len(upper)
    stencil_size = 2 * radius + 1
    dmat = np.zeros((npts, npts), dtype=float)
    for j in range(radius):
        dmat[j, :stencil_size] = fd_weights(float(x[j]), x[:stencil_size], 1)
    for j in range(radius, npts - radius):
        for k, coef in enumerate(upper, start=1):
            dmat[j, j - k] = -coef / h
            dmat[j, j + k] = coef / h
    for j in range(npts - radius, npts):
        dmat[j, -stencil_size:] = fd_weights(float(x[j]), x[-stencil_size:], 1)
    return x.copy(), dmat



def sbp_first_derivative_coefficients(order: int) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...], tuple[float, ...]]:
    if order == 4:
        left_rows = (
            (-24.0 / 17.0, 59.0 / 34.0, -4.0 / 17.0, -3.0 / 34.0),
            (-1.0 / 2.0, 0.0, 1.0 / 2.0),
            (4.0 / 43.0, -59.0 / 86.0, 0.0, 59.0 / 86.0, -4.0 / 43.0),
            (3.0 / 98.0, 0.0, -59.0 / 98.0, 0.0, 32.0 / 49.0, -4.0 / 49.0),
        )
        upper = (2.0 / 3.0, -1.0 / 12.0)
        weights = (17.0 / 48.0, 59.0 / 48.0, 43.0 / 48.0, 49.0 / 48.0)
        return left_rows, upper, weights
    if order == 6:
        left_rows = (
            (-21600.0 / 13649.0, 104009.0 / 54596.0, 30443.0 / 81894.0, -33311.0 / 27298.0, 16863.0 / 27298.0, -15025.0 / 163788.0),
            (-104009.0 / 240260.0, 0.0, -311.0 / 72078.0, 20229.0 / 24026.0, -24337.0 / 48052.0, 36661.0 / 360390.0),
            (-30443.0 / 162660.0, 311.0 / 32532.0, 0.0, -11155.0 / 16266.0, 41287.0 / 32532.0, -21999.0 / 54220.0),
            (33311.0 / 107180.0, -20229.0 / 21436.0, 485.0 / 1398.0, 0.0, 4147.0 / 21436.0, 25427.0 / 321540.0, 72.0 / 5359.0),
            (-16863.0 / 78770.0, 24337.0 / 31508.0, -41287.0 / 47262.0, -4147.0 / 15754.0, 0.0, 342523.0 / 472620.0, -1296.0 / 7877.0, 144.0 / 7877.0),
            (15025.0 / 525612.0, -36661.0 / 262806.0, 21999.0 / 87602.0, -25427.0 / 262806.0, -342523.0 / 525612.0, 0.0, 32400.0 / 43801.0, -6480.0 / 43801.0, 720.0 / 43801.0),
        )
        upper = (3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0)
        weights = (13649.0 / 43200.0, 12013.0 / 8640.0, 2711.0 / 4320.0, 5359.0 / 4320.0, 7877.0 / 8640.0, 43801.0 / 43200.0)
        return left_rows, upper, weights
    if order == 8:
        left_rows = (
            (-2540160.0 / 1498139.0, 5544277.0 / 5992556.0, 198794991.0 / 29962780.0, -256916579.0 / 17977668.0, 20708767.0 / 1498139.0, -41004357.0 / 5992556.0, 27390659.0 / 17977668.0, -2323531.0 / 29962780.0),
            (-5544277.0 / 31004596.0, 0.0, -85002381.0 / 22146140.0, 49607267.0 / 4429228.0, -165990199.0 / 13287684.0, 7655859.0 / 1107307.0, -7568311.0 / 4429228.0, 48319961.0 / 465068940.0),
            (-66264997.0 / 8719620.0, 9444709.0 / 415220.0, 0.0, -20335981.0 / 249132.0, 32320879.0 / 249132.0, -35518713.0 / 415220.0, 2502774.0 / 103805.0, -3177073.0 / 1743924.0),
            (256916579.0 / 109619916.0, -49607267.0 / 5219996.0, 61007943.0 / 5219996.0, 0.0, -68748371.0 / 5219996.0, 65088123.0 / 5219996.0, -66558305.0 / 15659988.0, 3870214.0 / 9134993.0),
            (-20708767.0 / 2096689.0, 165990199.0 / 3594324.0, -96962637.0 / 1198108.0, 68748371.0 / 1198108.0, 0.0, -27294549.0 / 1198108.0, 14054993.0 / 1198108.0, -42678199.0 / 25160268.0, -2592.0 / 299527.0),
            (13668119.0 / 8660148.0, -850651.0 / 103097.0, 35518713.0 / 2061940.0, -21696041.0 / 1237164.0, 9098183.0 / 1237164.0, 0.0, -231661.0 / 412388.0, 7120007.0 / 43300740.0, 3072.0 / 103097.0, -288.0 / 103097.0),
            (-27390659.0 / 56287644.0, 7568311.0 / 2680364.0, -22524966.0 / 3350455.0, 66558305.0 / 8041092.0, -14054993.0 / 2680364.0, 2084949.0 / 2680364.0, 0.0, 70710683.0 / 93812740.0, -145152.0 / 670091.0, 27648.0 / 670091.0, -2592.0 / 670091.0),
            (2323531.0 / 102554780.0, -48319961.0 / 307664340.0, 9531219.0 / 20510956.0, -3870214.0 / 5127739.0, 2246221.0 / 3238572.0, -21360021.0 / 102554780.0, -70710683.0 / 102554780.0, 0.0, 4064256.0 / 5127739.0, -1016064.0 / 5127739.0, 193536.0 / 5127739.0, -18144.0 / 5127739.0),
        )
        upper = (4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0)
        weights = (1498139.0 / 5080320.0, 1107307.0 / 725760.0, 20761.0 / 80640.0, 1304999.0 / 725760.0, 299527.0 / 725760.0, 103097.0 / 80640.0, 670091.0 / 725760.0, 5127739.0 / 5080320.0)
        return left_rows, upper, weights
    raise ValueError("supported SBP orders are 4, 6, and 8")


@lru_cache(maxsize=None)
def sbp_fd_matrix(npts: int, order: int = 6, length: float = 1.0) -> tuple[Array, Array, Array]:
    x = uniform_grid(npts, length)
    h = x[1] - x[0]
    left_rows, upper, weights = sbp_first_derivative_coefficients(order)
    radius = len(upper)
    nclosure = len(weights)
    if npts < 2 * nclosure + 1:
        raise ValueError(f"need at least {2 * nclosure + 1} points for SBP{order}")
    dmat = np.zeros((npts, npts), dtype=float)
    for j, row in enumerate(left_rows):
        dmat[j, :len(row)] = np.asarray(row, dtype=float) / h
    for j in range(nclosure, npts - nclosure):
        for k, coef in enumerate(upper, start=1):
            dmat[j, j - k] = -coef / h
            dmat[j, j + k] = coef / h
    for idx in range(nclosure):
        row = np.asarray(left_rows[nclosure - 1 - idx], dtype=float)
        dmat[npts - nclosure + idx, npts - row.size:] = -row[::-1] / h
    diag = np.ones(npts, dtype=float)
    diag[:nclosure] = np.asarray(weights, dtype=float)
    diag[-nclosure:] = np.asarray(weights[::-1], dtype=float)
    metric = h * np.diag(diag)
    return x.copy(), dmat, metric



def compact_interior_coefficients(order: int) -> tuple[float, float, float, float]:
    if order == 4:
        return 1.0 / 4.0, 3.0 / 2.0, 0.0, 0.0
    if order == 6:
        return 1.0 / 3.0, 14.0 / 9.0, 1.0 / 9.0, 0.0
    if order == 8:
        return 3.0 / 8.0, 25.0 / 16.0, 1.0 / 5.0, -1.0 / 80.0
    raise ValueError("supported compact orders are 4, 6, and 8")


@lru_cache(maxsize=None)
def compact_fd_matrices(npts: int, order: int = 6, length: float = 1.0) -> tuple[Array, Array, Array]:
    x = uniform_grid(npts, length)
    h = x[1] - x[0]
    alpha, a, b, c = compact_interior_coefficients(order)
    radius = order // 2
    stencil_size = order + 1
    amat = np.zeros((npts, npts), dtype=float)
    bmat = np.zeros((npts, npts), dtype=float)
    for j in range(radius):
        amat[j, j] = 1.0
        bmat[j, :stencil_size] = fd_weights(float(x[j]), x[:stencil_size], 1)
    for j in range(radius, npts - radius):
        amat[j, j - 1] = alpha
        amat[j, j] = 1.0
        amat[j, j + 1] = alpha
        bmat[j, j - 1] = -a / (2.0 * h)
        bmat[j, j + 1] = a / (2.0 * h)
        if abs(b) > 0.0:
            bmat[j, j - 2] = -b / (4.0 * h)
            bmat[j, j + 2] = b / (4.0 * h)
        if abs(c) > 0.0:
            bmat[j, j - 3] = -c / (6.0 * h)
            bmat[j, j + 3] = c / (6.0 * h)
    for j in range(npts - radius, npts):
        amat[j, j] = 1.0
        bmat[j, -stencil_size:] = fd_weights(float(x[j]), x[-stencil_size:], 1)
    return x.copy(), amat, bmat



def interpolation_knots(x: Array, degree: int) -> Array:
    x = np.asarray(x, dtype=float)
    n = x.size - 1
    p = int(degree)
    if n < p:
        raise ValueError("need at least degree+1 sample points")
    knots = np.empty(n + p + 2, dtype=float)
    knots[: p + 1] = x[0]
    knots[n + 1 :] = x[-1]
    for j in range(1, n - p + 1):
        knots[j + p] = np.mean(x[j : j + p])
    return knots


@lru_cache(maxsize=None)
def open_uniform_knots(npts: int, degree: int, length: float = 1.0) -> Array:
    npts = int(npts)
    p = int(degree)
    if npts <= p:
        raise ValueError("need at least degree+1 basis functions")
    interior = npts - p - 1
    left = np.zeros(p + 1, dtype=float)
    right = np.full(p + 1, float(length), dtype=float)
    if interior <= 0:
        return np.concatenate([left, right])
    mids = np.linspace(0.0, float(length), interior + 2, dtype=float)[1:-1]
    return np.concatenate([left, mids, right])


def greville_from_knots(knots: Array, degree: int, nbasis: int | None = None) -> Array:
    knots = np.asarray(knots, dtype=float)
    p = int(degree)
    nbas = int(nbasis) if nbasis is not None else knots.size - p - 1
    pts = np.zeros(nbas, dtype=float)
    for i in range(nbas):
        if p == 0:
            pts[i] = knots[i]
        else:
            pts[i] = np.sum(knots[i + 1 : i + p + 1]) / p
    pts[0] = knots[0]
    pts[-1] = knots[-1]
    return pts


def matrices_from_knots_and_nodes(knots: Array, nodes: Array, degree: int) -> tuple[Array, Array]:
    knots = np.asarray(knots, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    nbasis = nodes.size
    bmat = np.zeros((nbasis, nbasis), dtype=float)
    gmat = np.zeros((nbasis, nbasis), dtype=float)
    coeffs = np.zeros(nbasis, dtype=float)
    for i in range(nbasis):
        coeffs[:] = 0.0
        coeffs[i] = 1.0
        basis = BSpline(knots, coeffs.copy(), degree, extrapolate=False)
        bmat[:, i] = basis(nodes)
        gmat[:, i] = basis.derivative()(nodes)
    return bmat, gmat


def pack_sym(mat: Array) -> Array:
    rows, cols = np.triu_indices(mat.shape[0])
    return np.asarray(mat, dtype=float)[rows, cols]


def unpack_sym(vec: Array, n: int) -> Array:
    out = np.zeros((n, n), dtype=float)
    rows, cols = np.triu_indices(n)
    out[rows, cols] = vec
    out[cols, rows] = vec
    return out


def sym_constraint_matrix(dmat: Array) -> Array:
    dmat = np.asarray(dmat, dtype=float)
    n = dmat.shape[0]
    cols = []
    for k in range(n * (n + 1) // 2):
        basis = np.zeros(n * (n + 1) // 2, dtype=float)
        basis[k] = 1.0
        hmat = unpack_sym(basis, n)
        cols.append(pack_sym(hmat @ dmat + dmat.T @ hmat))
    return np.column_stack(cols)


def affine_solution_space(dmat: Array, boundary: Array) -> tuple[Array, Array]:
    amat = sym_constraint_matrix(dmat)
    bvec = pack_sym(boundary)
    umat, sing, vhat = np.linalg.svd(amat, full_matrices=False)
    tol = max(amat.shape) * sing[0] * np.finfo(float).eps * 20.0
    rank = int((sing > tol).sum())
    x0 = vhat[:rank].T @ ((umat[:, :rank].T @ bvec) / sing[:rank])
    null_basis = vhat[rank:].T.copy()
    return x0, null_basis


def metric_from_coords(zvec: Array, x0: Array, null_basis: Array, npts: int) -> Array:
    xvec = x0 + (null_basis @ zvec if null_basis.size else 0.0)
    hmat = unpack_sym(xvec, npts)
    return 0.5 * (hmat + hmat.T)


def _project_to_affine(vec: Array, x0: Array, null_basis: Array) -> Array:
    if null_basis.size == 0:
        return x0.copy()
    return x0 + null_basis @ (null_basis.T @ (vec - x0))


def _spectral_margin_penalty(zvec: Array, x0: Array, null_basis: Array, npts: int, floor: float) -> float:
    eig = np.linalg.eigvalsh(metric_from_coords(zvec, x0, null_basis, npts))
    deficit = np.clip(floor - eig, 0.0, None)
    return float(deficit @ deficit + 1.0e-12 * (zvec @ zvec))


def _maximize_min_eig(zvec: Array, x0: Array, null_basis: Array, npts: int, floor: float) -> float:
    eig = np.linalg.eigvalsh(metric_from_coords(zvec, x0, null_basis, npts))
    deficit = np.clip(floor - eig, 0.0, None)
    return float(-eig.min() + 1.0e6 * (deficit @ deficit) + 1.0e-12 * (zvec @ zvec))


def solve_full_metric(dmat: Array, boundary: Array, href: Array) -> tuple[Array, float, float]:
    npts = dmat.shape[0]
    x0, null_basis = affine_solution_space(dmat, boundary)
    href_vec = pack_sym(0.5 * (np.asarray(href, dtype=float) + np.asarray(href, dtype=float).T))
    xvec = _project_to_affine(href_vec, x0, null_basis)

    def spd_project(vec: Array, floor: float = 1.0e-10) -> Array:
        htmp = unpack_sym(vec, npts)
        eigvals, eigvecs = np.linalg.eigh(0.5 * (htmp + htmp.T))
        eigvals = np.maximum(eigvals, floor)
        return pack_sym(eigvecs @ np.diag(eigvals) @ eigvecs.T)

    for _ in range(200):
        xold = xvec.copy()
        xvec = _project_to_affine(xvec, x0, null_basis)
        resid = np.linalg.norm(xvec - _project_to_affine(xvec, x0, null_basis))
        mineig = np.linalg.eigvalsh(unpack_sym(xvec, npts)).min()
        if resid < 1.0e-10 and mineig > 1.0e-10:
            break
        xvec = spd_project(xvec)
        if np.linalg.norm(xvec - xold) < 1.0e-12:
            break

    xvec = _project_to_affine(xvec, x0, null_basis)
    hmat = unpack_sym(xvec, npts)
    eigmin = float(np.linalg.eigvalsh(hmat).min())

    if eigmin <= 1.0e-10 and null_basis.size:
        ref_z = null_basis.T @ (href_vec - x0)
        warm_z = null_basis.T @ (xvec - x0)
        starts = [ref_z, warm_z, np.zeros_like(ref_z)]
        rng = np.random.default_rng(0)
        for scale in (1.0e-3, 1.0e-2, 1.0e-1):
            starts.append(ref_z + scale * rng.standard_normal(ref_z.shape))
        best_h = hmat.copy()
        best_z = warm_z.copy()
        best_margin = eigmin
        for start in starts:
            sol = minimize(
                _spectral_margin_penalty,
                start,
                args=(x0, null_basis, npts, 1.0e-10),
                method='Powell',
                options={'maxiter': 120, 'xtol': 1.0e-4, 'ftol': 1.0e-10},
            )
            cand = metric_from_coords(sol.x, x0, null_basis, npts)
            margin = float(np.linalg.eigvalsh(cand).min())
            if margin > best_margin:
                best_h = cand
                best_z = sol.x.copy()
                best_margin = margin
        if best_margin > 1.0e-10:
            sol = minimize(
                _maximize_min_eig,
                best_z,
                args=(x0, null_basis, npts, 1.0e-10),
                method='Powell',
                options={'maxiter': 120, 'xtol': 1.0e-4, 'ftol': 1.0e-10},
            )
            cand = metric_from_coords(sol.x, x0, null_basis, npts)
            margin = float(np.linalg.eigvalsh(cand).min())
            if margin > best_margin:
                best_h = cand
                best_margin = margin
        hmat = best_h
        eigmin = best_margin

    hmat = 0.5 * (hmat + hmat.T)
    resid = float(np.linalg.norm(hmat @ dmat + dmat.T @ hmat - boundary, 2))
    eigmin = float(np.linalg.eigvalsh(hmat).min())
    return hmat, resid, eigmin


@lru_cache(maxsize=None)
def clamped_bspline_matrices(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array, Array, Array]:
    knots = open_uniform_knots(npts, degree, length)
    x = greville_from_knots(knots, degree, npts)
    bmat, gmat = matrices_from_knots_and_nodes(knots, x, degree)
    return x.copy(), knots.copy(), bmat, gmat


@lru_cache(maxsize=None)
def clamped_bspline_curvature_matrix(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array]:
    x, knots, _bmat, _gmat = clamped_bspline_matrices(npts, degree, length)
    nbasis = npts
    hmat = np.zeros((npts, nbasis), dtype=float)
    coeffs = np.zeros(nbasis, dtype=float)
    for i in range(nbasis):
        coeffs[:] = 0.0
        coeffs[i] = 1.0
        basis = BSpline(knots, coeffs.copy(), degree, extrapolate=False)
        hmat[:, i] = basis.derivative(2)(x)
    return x.copy(), hmat


@lru_cache(maxsize=None)
def bmf_curvature_matrix(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array]:
    x, _knots, bmat, _gmat = clamped_bspline_matrices(npts, degree, length)
    _x, hmat = clamped_bspline_curvature_matrix(npts, degree, length)
    plu = lu_factor(bmat)
    curvmat = lu_solve(plu, hmat.T, trans=1).T
    return x.copy(), curvmat


@lru_cache(maxsize=None)
def bmf_direct_second_derivative_matrix(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array]:
    return bmf_curvature_matrix(npts, degree, length)


@lru_cache(maxsize=None)
def bspline_mass_and_advection(npts: int, degree: int, length: float = 1.0, quad_order: int = 40) -> tuple[Array, Array]:
    mass, adv, _weak_d2, _stiffness = bspline_mass_advection_second_and_stiffness(
        npts, degree, length, quad_order
    )
    return mass.copy(), adv.copy()


@lru_cache(maxsize=None)
def bspline_mass_advection_second_and_stiffness(
    npts: int,
    degree: int,
    length: float = 1.0,
    quad_order: int = 40,
) -> tuple[Array, Array, Array, Array]:
    _, knots, _, _ = clamped_bspline_matrices(npts, degree, length)
    nbasis = npts
    coeffs = np.zeros(nbasis, dtype=float)
    basis = []
    dbasis = []
    d2basis = []
    for i in range(nbasis):
        coeffs[:] = 0.0
        coeffs[i] = 1.0
        b = BSpline(knots, coeffs.copy(), degree, extrapolate=False)
        basis.append(b)
        dbasis.append(b.derivative())
        d2basis.append(b.derivative(2))
    uk = np.unique(knots)
    spans = [(a, b) for a, b in zip(uk[:-1], uk[1:]) if b > a + 1.0e-14]
    xi, wi = leggauss(quad_order)
    mass = np.zeros((nbasis, nbasis), dtype=float)
    adv = np.zeros((nbasis, nbasis), dtype=float)
    weak_d2 = np.zeros((nbasis, nbasis), dtype=float)
    stiffness = np.zeros((nbasis, nbasis), dtype=float)
    for a, b in spans:
        pts = 0.5 * (b - a) * xi + 0.5 * (a + b)
        w = 0.5 * (b - a) * wi
        vals = np.stack([f(pts) for f in basis], axis=0)
        dvals = np.stack([f(pts) for f in dbasis], axis=0)
        d2vals = np.stack([f(pts) for f in d2basis], axis=0)
        wmat = np.diag(w)
        mass += vals @ wmat @ vals.T
        adv += vals @ wmat @ dvals.T
        weak_d2 += vals @ wmat @ d2vals.T
        stiffness += dvals @ wmat @ dvals.T
    return mass, adv, weak_d2, stiffness


@dataclass(frozen=True)
class BMFStaggeredOperators:
    """Primal/derivative-grid operators for the staggered BMF SBP complex."""

    x: Array
    eta: Array
    d10: Array
    d01: Array
    h0: Array
    h1: Array
    boundary: Array
    bmat: Array
    bhat_eta: Array
    bhat_x: Array
    delta: Array


def bspline_evaluation_matrix(knots: Array, degree: int, nodes: Array, derivative: int = 0) -> Array:
    knots = np.asarray(knots, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    p = int(degree)
    nbasis = knots.size - p - 1
    if derivative > p:
        return np.zeros((nodes.size, nbasis), dtype=float)
    mat = np.zeros((nodes.size, nbasis), dtype=float)
    coeffs = np.zeros(nbasis, dtype=float)
    for i in range(nbasis):
        coeffs[:] = 0.0
        coeffs[i] = 1.0
        basis = BSpline(knots, coeffs.copy(), p, extrapolate=False)
        if derivative:
            basis = basis.derivative(derivative)
        vals = np.asarray(basis(nodes), dtype=float)
        mat[:, i] = np.where(np.isfinite(vals), vals, 0.0)
    return mat


def derivative_space_knots(knots: Array) -> Array:
    knots = np.asarray(knots, dtype=float)
    if knots.size < 4:
        raise ValueError("not enough knots to form a derivative space")
    return knots[1:-1].copy()


def derivative_greville_nodes(knots: Array, degree: int) -> Array:
    knots = np.asarray(knots, dtype=float)
    p = int(degree)
    nbasis = knots.size - p - 1
    if p == 0:
        return 0.5 * (knots[:nbasis] + knots[1:nbasis + 1])
    return greville_from_knots(knots, p, nbasis)


def bspline_mass_matrix_from_knots(knots: Array, degree: int, quad_order: int = 40) -> Array:
    knots = np.asarray(knots, dtype=float)
    p = int(degree)
    nbasis = knots.size - p - 1
    coeffs = np.zeros(nbasis, dtype=float)
    basis = []
    for i in range(nbasis):
        coeffs[:] = 0.0
        coeffs[i] = 1.0
        basis.append(BSpline(knots, coeffs.copy(), p, extrapolate=False))
    uk = np.unique(knots)
    spans = [(a, b) for a, b in zip(uk[:-1], uk[1:]) if b > a + 1.0e-14]
    xi, wi = leggauss(quad_order)
    mass = np.zeros((nbasis, nbasis), dtype=float)
    for a, b in spans:
        pts = 0.5 * (b - a) * xi + 0.5 * (a + b)
        w = 0.5 * (b - a) * wi
        vals = np.stack([f(pts) for f in basis], axis=0)
        mass += vals @ np.diag(w) @ vals.T
    return mass


def bspline_derivative_incidence_matrix(knots: Array, degree: int) -> Array:
    knots = np.asarray(knots, dtype=float)
    p = int(degree)
    npts = knots.size - p - 1
    if p < 1:
        raise ValueError("degree must be at least one for the derivative incidence matrix")
    delta = np.zeros((npts - 1, npts), dtype=float)
    for i in range(npts - 1):
        denom = knots[i + p + 1] - knots[i + 1]
        if denom <= 0.0:
            raise ValueError("degenerate knot span in derivative incidence matrix")
        scale = p / denom
        delta[i, i] = -scale
        delta[i, i + 1] = scale
    return delta


@lru_cache(maxsize=None)
def bmf_staggered_operators(npts: int, degree: int, length: float = 1.0) -> BMFStaggeredOperators:
    x, knots, bmat, _gmat = clamped_bspline_matrices(npts, degree, length)
    if degree < 1:
        raise ValueError("staggered BMF requires degree at least one")
    hknots = derivative_space_knots(knots)
    hdegree = int(degree) - 1
    eta = derivative_greville_nodes(hknots, hdegree)
    bhat_eta = bspline_evaluation_matrix(hknots, hdegree, eta)
    bhat_x = bspline_evaluation_matrix(hknots, hdegree, x)
    delta = bspline_derivative_incidence_matrix(knots, degree)

    mass0 = bspline_mass_matrix_from_knots(knots, degree)
    mass1 = bspline_mass_matrix_from_knots(hknots, hdegree)

    binv = np.linalg.inv(bmat)
    bhatinv = np.linalg.inv(bhat_eta)
    h0 = 0.5 * (binv.T @ mass0 @ binv + (binv.T @ mass0 @ binv).T)
    h1 = 0.5 * (bhatinv.T @ mass1 @ bhatinv + (bhatinv.T @ mass1 @ bhatinv).T)

    d10 = bhat_eta @ delta @ binv

    left = bspline_evaluation_matrix(hknots, hdegree, np.array([0.0], dtype=float)) @ bhatinv
    right = bspline_evaluation_matrix(hknots, hdegree, np.array([float(length)], dtype=float)) @ bhatinv
    e0 = np.zeros((npts, 1), dtype=float)
    eN = np.zeros((npts, 1), dtype=float)
    e0[0, 0] = 1.0
    eN[-1, 0] = 1.0
    boundary = -e0 @ left + eN @ right
    d01 = np.linalg.solve(h0, boundary - d10.T @ h1)
    return BMFStaggeredOperators(
        x=x.copy(),
        eta=eta.copy(),
        d10=d10.copy(),
        d01=d01.copy(),
        h0=h0.copy(),
        h1=h1.copy(),
        boundary=boundary.copy(),
        bmat=bmat.copy(),
        bhat_eta=bhat_eta.copy(),
        bhat_x=bhat_x.copy(),
        delta=delta.copy(),
    )


@lru_cache(maxsize=None)
def bmf_staggered_first_derivative_matrices(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array, Array, Array]:
    ops = bmf_staggered_operators(npts, degree, length)
    return ops.x.copy(), ops.eta.copy(), ops.d10.copy(), ops.d01.copy()


@lru_cache(maxsize=None)
def bmf_staggered_metric_matrices(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array, Array]:
    ops = bmf_staggered_operators(npts, degree, length)
    return ops.h0.copy(), ops.h1.copy(), ops.boundary.copy()


@lru_cache(maxsize=None)
def bmf_reconstructed_same_grid_derivative_matrix(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array]:
    ops = bmf_staggered_operators(npts, degree, length)
    bhatinv = np.linalg.inv(ops.bhat_eta)
    dmat = ops.bhat_x @ bhatinv @ ops.d10
    return ops.x.copy(), dmat


@lru_cache(maxsize=None)
def bmf_flux_divergence_second_derivative_matrix(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array]:
    ops = bmf_staggered_operators(npts, degree, length)
    return ops.x.copy(), ops.d01 @ ops.d10


@lru_cache(maxsize=None)
def bmf_staggered_sbp_residual(npts: int, degree: int, length: float = 1.0) -> Array:
    ops = bmf_staggered_operators(npts, degree, length)
    return ops.h0 @ ops.d01 + ops.d10.T @ ops.h1 - ops.boundary


@lru_cache(maxsize=None)
def bmg_nodal_operator(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array, Array]:
    x, _, bmat, _ = clamped_bspline_matrices(npts, degree, length)
    mass, adv = bspline_mass_and_advection(npts, degree, length)
    binv = np.linalg.inv(bmat)
    dmat = bmat @ np.linalg.solve(mass, adv) @ binv
    metric = binv.T @ mass @ binv
    return x.copy(), dmat, metric


@lru_cache(maxsize=None)
def bmf_nodal_operator(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array]:
    x, _, bmat, gmat = clamped_bspline_matrices(npts, degree, length)
    plu = lu_factor(bmat)
    dmat = lu_solve(plu, gmat.T, trans=1).T
    return x.copy(), dmat


@lru_cache(maxsize=None)
def bmf_polynomial_projector(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array]:
    x, _, _, _ = clamped_bspline_matrices(npts, degree, length)
    poly_cols = max(int(degree), 1)
    vand = np.vander(x, N=poly_cols, increasing=True)
    qmat, _rmat = np.linalg.qr(vand, mode='reduced')
    proj = np.eye(npts, dtype=float) - qmat @ qmat.T
    return x.copy(), 0.5 * (proj + proj.T)


@lru_cache(maxsize=None)
def bmf_riesz_map(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array]:
    x, _, bmat, _ = clamped_bspline_matrices(npts, degree, length)
    hmat = bmf_full_metric(npts, degree, length)[1]
    nmat = bmat.T @ hmat @ bmat
    return x.copy(), 0.5 * (nmat + nmat.T)


@lru_cache(maxsize=None)
def bmf_reconstruction_derivative_matrix(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array]:
    """Return the same-point reconstruction derivative of the clamped basis.

    The matrix is the reconstruction map B'B^{-1}, which differentiates the
    spline reconstruction and samples the result at the same points. It is not
    the BMF summation-by-parts operator: no metric accompanies it and no
    discrete integration-by-parts identity is claimed for it.
    """
    return bmf_nodal_operator(npts, degree, length)


@lru_cache(maxsize=None)
def clamped_boundary_trace_rows(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array]:
    """Return the first and last rows of the same-point reconstruction derivative.

    The rows come from the reconstruction map B'B^{-1}, not from the BMF
    summation-by-parts operator.
    """
    _x, dmat = bmf_reconstruction_derivative_matrix(npts, degree, length)
    return dmat[[0], :].copy(), dmat[[-1], :].copy()


@lru_cache(maxsize=None)
def bmf_sbp_first_derivative_matrix(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array]:
    """Alias retained for existing call sites.

    The returned matrix is the same-point reconstruction derivative
    B'B^{-1} of bmf_reconstruction_derivative_matrix, not the BMF
    summation-by-parts operator.
    """
    return bmf_reconstruction_derivative_matrix(npts, degree, length)


@lru_cache(maxsize=None)
def bmg_sbp_first_derivative_matrix(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array, Array]:
    return bmg_nodal_operator(npts, degree, length)


@lru_cache(maxsize=None)
def bmf_weak_curvature_second_derivative_matrix(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array]:
    x, _, bmat, _ = clamped_bspline_matrices(npts, degree, length)
    _x, nmat = bmf_riesz_map(npts, degree, length)
    _mass, _adv, weak_d2, _stiffness = bspline_mass_advection_second_and_stiffness(npts, degree, length)
    binv = np.linalg.inv(bmat)
    d2mat = bmat @ np.linalg.solve(nmat, weak_d2) @ binv
    return x.copy(), d2mat


@lru_cache(maxsize=None)
def bmf_sbp_second_derivative_matrix(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array]:
    return bmf_flux_divergence_second_derivative_matrix(npts, degree, length)


@lru_cache(maxsize=None)
def bmg_weak_curvature_second_derivative_matrix(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array]:
    x, _, bmat, _ = clamped_bspline_matrices(npts, degree, length)
    mass, _adv, weak_d2, _stiffness = bspline_mass_advection_second_and_stiffness(npts, degree, length)
    binv = np.linalg.inv(bmat)
    d2mat = bmat @ np.linalg.solve(mass, weak_d2) @ binv
    return x.copy(), d2mat


@lru_cache(maxsize=None)
def bmg_sbp_second_derivative_matrix(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array]:
    x, dmat, hmat = bmg_sbp_first_derivative_matrix(npts, degree, length)
    e0 = np.zeros((npts, 1), dtype=float)
    eN = np.zeros((npts, 1), dtype=float)
    e0[0, 0] = 1.0
    eN[-1, 0] = 1.0
    d2mat = np.linalg.solve(hmat, -dmat.T @ hmat @ dmat - e0 @ dmat[[0], :] + eN @ dmat[[-1], :])
    return x.copy(), d2mat


@lru_cache(maxsize=None)
def sbp_boundary_trace_rows(npts: int, order: int, length: float = 1.0) -> tuple[Array, Array]:
    _x, dmat, _metric = sbp_fd_matrix(npts, order, length)
    return dmat[[0], :].copy(), dmat[[-1], :].copy()


@lru_cache(maxsize=None)
def sbp_second_derivative_matrix(npts: int, order: int, length: float = 1.0) -> tuple[Array, Array]:
    x, dmat, hmat = sbp_fd_matrix(npts, order, length)
    s0, sN = sbp_boundary_trace_rows(npts, order, length)
    e0 = np.zeros((npts, 1), dtype=float)
    eN = np.zeros((npts, 1), dtype=float)
    e0[0, 0] = 1.0
    eN[-1, 0] = 1.0
    d2mat = np.linalg.solve(hmat, -dmat.T @ hmat @ dmat - e0 @ s0 + eN @ sN)
    return x.copy(), d2mat


@lru_cache(maxsize=None)
def clamped_greville_reference_metric(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array]:
    _, _, bmat, _ = clamped_bspline_matrices(npts, degree, length)
    mass, _ = bspline_mass_and_advection(npts, degree, length)
    binv = np.linalg.inv(bmat)
    metric = binv.T @ mass @ binv
    return bmat.copy(), 0.5 * (metric + metric.T)


@lru_cache(maxsize=None)
def bmf_full_metric(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array]:
    ops = bmf_staggered_operators(npts, degree, length)
    return ops.x.copy(), ops.h0.copy()


def clamped_greville_metric(npts: int, degree: int, length: float = 1.0) -> tuple[Array, Array]:
    return bmf_full_metric(npts, degree, length)



def spline_eval_matrices(points: Array, x_grid: Array, degree: int, knots: Array | None = None) -> tuple[Array, Array, Array]:
    points = np.asarray(points, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    if knots is None:
        length = float(x_grid[-1] - x_grid[0])
        knots = open_uniform_knots(x_grid.size, degree, length) + x_grid[0]
    else:
        knots = np.asarray(knots, dtype=float)
    nbasis = x_grid.size
    phi = np.zeros((points.size, nbasis), dtype=float)
    dphi = np.zeros((points.size, nbasis), dtype=float)
    coeffs = np.zeros(nbasis, dtype=float)
    for i in range(nbasis):
        coeffs[:] = 0.0
        coeffs[i] = 1.0
        basis = BSpline(knots, coeffs.copy(), degree, extrapolate=False)
        phi[:, i] = basis(points)
        dphi[:, i] = basis.derivative()(points)
    return knots, phi, dphi



def local_stencil_indices(x_grid: Array, x0: float, stencil_size: int) -> Array:
    n = x_grid.size
    idx = np.searchsorted(x_grid, x0) - 1
    left = max(0, min(n - stencil_size, idx - stencil_size // 2))
    return np.arange(left, left + stencil_size)



def barycentric_weights(x_nodes: Array) -> Array:
    x_nodes = np.asarray(x_nodes, dtype=float)
    n = x_nodes.size
    w = np.ones(n, dtype=float)
    for j in range(n):
        for k in range(n):
            if k != j:
                w[j] /= x_nodes[j] - x_nodes[k]
    return w



def local_interpolation_matrix(x_grid: Array, points: Array, stencil_size: int) -> Array:
    x_grid = np.asarray(x_grid, dtype=float)
    points = np.asarray(points, dtype=float)
    mat = np.zeros((points.size, x_grid.size), dtype=float)
    for row, x0 in enumerate(points):
        idx = local_stencil_indices(x_grid, float(x0), stencil_size)
        x_nodes = x_grid[idx]
        diffs = x0 - x_nodes
        if np.any(np.abs(diffs) < 1.0e-14):
            loc = np.argmin(np.abs(diffs))
            mat[row, idx[loc]] = 1.0
            continue
        w = barycentric_weights(x_nodes)
        temp = w / diffs
        temp /= np.sum(temp)
        mat[row, idx] = temp
    return mat


class ExplicitFD(Operator1D):
    def __init__(self, npts: int, order: int, length: float = 1.0) -> None:
        x, dmat = explicit_fd_matrix(npts, order, length)
        self.name = f"E{order}"
        self.order = int(order)
        self._x = x
        self._dmat = dmat
        storage = int(np.count_nonzero(np.abs(dmat) > 1.0e-14))
        grid = int(sum(2 * np.count_nonzero(np.abs(row) > 1.0e-14) - 1 for row in dmat))
        self._work = WorkModel(grid_apply_flops=grid, solve_flops=0, total_flops=grid, storage_nnz=storage)

    @property
    def x(self) -> Array:
        return self._x

    @property
    def matrix(self) -> Array:
        return self._dmat

    @property
    def work_model(self) -> WorkModel:
        return self._work

    def differentiate(self, u: Array) -> Array:
        arr, squeeze = _as_2d(u)
        out = self._dmat @ arr
        return out[:, 0] if squeeze else out

    def field_and_derivative(self, points: Array, u: Array) -> tuple[Array, Array]:
        interp = local_interpolation_matrix(self._x, np.asarray(points, dtype=float), self.order + 1)
        du = self.differentiate(u)
        arr_u, squeeze = _as_2d(u)
        arr_du, _ = _as_2d(du)
        f = interp @ arr_u
        g = interp @ arr_du
        if squeeze:
            return f[:, 0], g[:, 0]
        return f, g


class CompactFD(Operator1D):
    def __init__(self, npts: int, order: int, length: float = 1.0) -> None:
        x, amat, bmat = compact_fd_matrices(npts, order, length)
        self.name = f"C{order}"
        self.order = int(order)
        self._x = x
        self._amat = amat
        self._bmat = bmat
        self._lu = lu_factor(amat)
        self._dmat = lu_solve(self._lu, bmat)
        storage = int(np.count_nonzero(np.abs(amat) > 1.0e-14) + np.count_nonzero(np.abs(bmat) > 1.0e-14))
        rhs_flops = int(sum(2 * np.count_nonzero(np.abs(row) > 1.0e-14) - 1 for row in bmat))
        solve_flops = 4 * npts
        self._work = WorkModel(grid_apply_flops=rhs_flops, solve_flops=solve_flops, total_flops=rhs_flops + solve_flops, storage_nnz=storage)

    @property
    def x(self) -> Array:
        return self._x

    @property
    def matrix(self) -> Array:
        return self._dmat

    @property
    def work_model(self) -> WorkModel:
        return self._work

    def differentiate(self, u: Array) -> Array:
        arr, squeeze = _as_2d(u)
        rhs = self._bmat @ arr
        out = lu_solve(self._lu, rhs)
        return out[:, 0] if squeeze else out

    def field_and_derivative(self, points: Array, u: Array) -> tuple[Array, Array]:
        interp = local_interpolation_matrix(self._x, np.asarray(points, dtype=float), self.order + 1)
        du = self.differentiate(u)
        arr_u, squeeze = _as_2d(u)
        arr_du, _ = _as_2d(du)
        f = interp @ arr_u
        g = interp @ arr_du
        if squeeze:
            return f[:, 0], g[:, 0]
        return f, g


class SBPSAT(Operator1D):
    def __init__(self, npts: int, order: int, length: float = 1.0) -> None:
        x, dmat, metric = sbp_fd_matrix(npts, order, length)
        self.name = f"SBP{order}"
        self.order = int(order)
        self._x = x
        self._dmat = dmat
        self._metric = metric
        self._d2mat: Array | None = None
        storage = int(np.count_nonzero(np.abs(dmat) > 1.0e-14) + np.count_nonzero(np.abs(metric) > 1.0e-14))
        grid = int(sum(2 * np.count_nonzero(np.abs(row) > 1.0e-14) - 1 for row in dmat))
        self._work = WorkModel(grid_apply_flops=grid, solve_flops=0, total_flops=grid, storage_nnz=storage)

    @property
    def x(self) -> Array:
        return self._x

    @property
    def matrix(self) -> Array:
        return self._dmat

    @property
    def metric(self) -> Array:
        return self._metric

    @property
    def work_model(self) -> WorkModel:
        return self._work

    def second_derivative_matrix(self) -> Array:
        if self._d2mat is None:
            self._d2mat = sbp_second_derivative_matrix(self._x.size, self.order, self._x[-1] - self._x[0])[1]
        return self._d2mat

    def differentiate(self, u: Array) -> Array:
        arr, squeeze = _as_2d(u)
        out = self._dmat @ arr
        return out[:, 0] if squeeze else out

    def field_and_derivative(self, points: Array, u: Array) -> tuple[Array, Array]:
        interp = local_interpolation_matrix(self._x, np.asarray(points, dtype=float), self.order + 1)
        du = self.differentiate(u)
        arr_u, squeeze = _as_2d(u)
        arr_du, _ = _as_2d(du)
        f = interp @ arr_u
        g = interp @ arr_du
        if squeeze:
            return f[:, 0], g[:, 0]
        return f, g


class DRP(Operator1D):
    def __init__(self, npts: int, order: int, length: float = 1.0) -> None:
        x, dmat = drp_fd_matrix(npts, order, length)
        self.name = f"DRP{order}"
        self.order = int(order)
        self._x = x
        self._dmat = dmat
        storage = int(np.count_nonzero(np.abs(dmat) > 1.0e-14))
        grid = int(sum(2 * np.count_nonzero(np.abs(row) > 1.0e-14) - 1 for row in dmat))
        self._work = WorkModel(grid_apply_flops=grid, solve_flops=0, total_flops=grid, storage_nnz=storage)

    @property
    def x(self) -> Array:
        return self._x

    @property
    def matrix(self) -> Array:
        return self._dmat

    @property
    def work_model(self) -> WorkModel:
        return self._work

    def differentiate(self, u: Array) -> Array:
        arr, squeeze = _as_2d(u)
        out = self._dmat @ arr
        return out[:, 0] if squeeze else out

    def field_and_derivative(self, points: Array, u: Array) -> tuple[Array, Array]:
        interp = local_interpolation_matrix(self._x, np.asarray(points, dtype=float), self.order + 1)
        du = self.differentiate(u)
        arr_u, squeeze = _as_2d(u)
        arr_du, _ = _as_2d(du)
        f = interp @ arr_u
        g = interp @ arr_du
        if squeeze:
            return f[:, 0], g[:, 0]
        return f, g


class ClampedBSpline(Operator1D):
    def __init__(self, npts: int, degree: int, length: float = 1.0) -> None:
        x, knots, bmat, gmat = clamped_bspline_matrices(npts, degree, length)
        self.name = f"BMF{degree}"
        self.order = int(degree)
        self.degree = int(degree)
        self.length = float(length)
        self._x = x
        self._knots = knots
        self._bmat = bmat
        self._gmat = gmat
        self._lu = lu_factor(bmat)
        self._dmat = lu_solve(self._lu, gmat.T, trans=1).T
        self._curvmat: Array | None = None
        self._d2mat: Array | None = None
        storage = int(np.count_nonzero(np.abs(bmat) > 1.0e-14) + np.count_nonzero(np.abs(gmat) > 1.0e-14))
        apply_flops = int(sum(2 * np.count_nonzero(np.abs(row) > 1.0e-14) - 1 for row in gmat))
        rows, cols = np.where(np.abs(bmat) > 1.0e-14)
        bw = int(max(np.max(rows - cols), np.max(cols - rows)))
        solve_flops = 4 * bw * npts
        self._work = WorkModel(grid_apply_flops=apply_flops, solve_flops=solve_flops, total_flops=apply_flops + solve_flops, storage_nnz=storage)

    @property
    def x(self) -> Array:
        return self._x

    @property
    def knots(self) -> Array:
        return self._knots

    @property
    def bmat(self) -> Array:
        return self._bmat

    @property
    def gmat(self) -> Array:
        return self._gmat

    @property
    def matrix(self) -> Array:
        return self._dmat

    @property
    def work_model(self) -> WorkModel:
        return self._work

    @property
    def metric(self) -> Array:
        """Primal natural metric H0 used by the staggered BMF complex.

        The square same-grid matrix is an exact reconstruction derivative.
        The BMF SBP identity uses ``staggered_operators``.
        """
        return bmf_full_metric(self._x.size, self.degree, self.length)[1]

    def staggered_operators(self) -> BMFStaggeredOperators:
        """Return the staggered BMF SBP blocks.

        Manuscript notation calls d10 the gradient G_BMF and d01 the
        divergence D_BMF; the attribute names are retained for API stability.
        """
        return bmf_staggered_operators(self._x.size, self.degree, self.length)

    def staggered_gradient(self, u: Array) -> Array:
        """Apply the BMF staggered gradient to primal data."""
        return self.staggered_operators().d10 @ np.asarray(u, dtype=float)

    def staggered_divergence(self, q: Array) -> Array:
        """Apply the BMF staggered divergence to derivative-grid flux data."""
        return self.staggered_operators().d01 @ np.asarray(q, dtype=float)

    def coefficients(self, u: Array) -> Array:
        arr, squeeze = _as_2d(u)
        coeffs = lu_solve(self._lu, arr)
        return coeffs[:, 0] if squeeze else coeffs

    def differentiate(self, u: Array) -> Array:
        coeffs, squeeze = _as_2d(self.coefficients(u))
        out = self._gmat @ coeffs
        return out[:, 0] if squeeze else out

    def curvature_matrix(self) -> Array:
        if self._curvmat is None:
            self._curvmat = bmf_curvature_matrix(self._x.size, self.degree, self.length)[1]
        return self._curvmat

    def second_derivative_matrix(self) -> Array:
        if self._d2mat is None:
            self._d2mat = bmf_sbp_second_derivative_matrix(self._x.size, self.degree, self.length)[1]
        return self._d2mat

    def field_and_derivative(self, points: Array, u: Array) -> tuple[Array, Array]:
        coeffs, squeeze = _as_2d(self.coefficients(u))
        _, phi, dphi = spline_eval_matrices(np.asarray(points, dtype=float), self._x, self.degree, self._knots)
        f = phi @ coeffs
        g = dphi @ coeffs
        if squeeze:
            return f[:, 0], g[:, 0]
        return f, g



class BoundaryBMG(Operator1D):
    def __init__(self, npts: int, degree: int, length: float = 1.0) -> None:
        x, knots, bmat, gmat = clamped_bspline_matrices(npts, degree, length)
        mass, adv = bspline_mass_and_advection(npts, degree, length)
        self.name = f"BMG{degree}"
        self.order = int(degree)
        self.degree = int(degree)
        self.length = float(length)
        self._x = x
        self._knots = knots
        self._bmat = bmat
        self._gmat = gmat
        self._mass = mass
        self._adv = adv
        self._plu = lu_factor(bmat)
        self._mlu = lu_factor(mass)
        self._binv = lu_solve(self._plu, np.eye(npts))
        self._dmat = bmat @ lu_solve(self._mlu, adv) @ self._binv
        self._d2mat_sbp: Array | None = None
        self._metric = self._binv.T @ mass @ self._binv
        storage = int(np.count_nonzero(np.abs(bmat) > 1.0e-14) + np.count_nonzero(np.abs(mass) > 1.0e-14) + np.count_nonzero(np.abs(adv) > 1.0e-14))
        rhs_flops = int(sum(2 * np.count_nonzero(np.abs(row) > 1.0e-14) - 1 for row in adv))
        brow, bcol = np.where(np.abs(bmat) > 1.0e-14)
        b_bw = int(max(np.max(brow - bcol), np.max(bcol - brow)))
        mrow, mcol = np.where(np.abs(mass) > 1.0e-14)
        m_bw = int(max(np.max(mrow - mcol), np.max(mcol - mrow)))
        p_apply = int(sum(2 * np.count_nonzero(np.abs(row) > 1.0e-14) - 1 for row in bmat))
        nodal_solve = 4 * b_bw * npts
        mass_solve = 4 * m_bw * npts
        self._work_nodal = WorkModel(grid_apply_flops=rhs_flops + p_apply,
                                     solve_flops=nodal_solve + mass_solve,
                                     total_flops=rhs_flops + p_apply + nodal_solve + mass_solve,
                                     storage_nnz=storage)
        self._work_coef = WorkModel(grid_apply_flops=rhs_flops,
                                    solve_flops=mass_solve,
                                    total_flops=rhs_flops + mass_solve,
                                    storage_nnz=int(np.count_nonzero(np.abs(mass) > 1.0e-14) + np.count_nonzero(np.abs(adv) > 1.0e-14)))

    @property
    def x(self) -> Array:
        return self._x

    @property
    def knots(self) -> Array:
        return self._knots

    @property
    def mass(self) -> Array:
        return self._mass

    @property
    def adv(self) -> Array:
        return self._adv

    @property
    def metric(self) -> Array:
        return self._metric

    @property
    def matrix(self) -> Array:
        return self._dmat

    @property
    def work_model(self) -> WorkModel:
        return self._work_nodal

    @property
    def work_model_coefficient(self) -> WorkModel:
        return self._work_coef

    def coefficients(self, u: Array) -> Array:
        arr, squeeze = _as_2d(u)
        coeffs = lu_solve(self._plu, arr)
        return coeffs[:, 0] if squeeze else coeffs

    def differentiate_coefficients(self, c: Array) -> Array:
        arr, squeeze = _as_2d(c)
        rhs = self._adv @ arr
        g = lu_solve(self._mlu, rhs)
        return g[:, 0] if squeeze else g

    def differentiate(self, u: Array) -> Array:
        coeffs = self.coefficients(u)
        g = self.differentiate_coefficients(coeffs)
        arr, squeeze = _as_2d(g)
        out = self._bmat @ arr
        return out[:, 0] if squeeze else out

    def second_derivative_matrix(self) -> Array:
        if self._d2mat_sbp is None:
            self._d2mat_sbp = bmg_sbp_second_derivative_matrix(self._x.size, self.degree, self.length)[1]
        return self._d2mat_sbp

    def field_and_derivative(self, points: Array, u: Array) -> tuple[Array, Array]:
        coeffs = self.coefficients(u)
        gcoeff = self.differentiate_coefficients(coeffs)
        carr, squeeze = _as_2d(coeffs)
        garr, _ = _as_2d(gcoeff)
        _, phi, _ = spline_eval_matrices(np.asarray(points, dtype=float), self._x, self.degree, self._knots)
        field = phi @ carr
        deriv = phi @ garr
        if squeeze:
            return field[:, 0], deriv[:, 0]
        return field, deriv


def second_derivative_matrix(operator: Operator1D) -> Array:
    return operator.second_derivative_matrix()


def build_operator(kind: str, order: int, npts: int, length: float = 1.0) -> Operator1D:
    kind = kind.upper()
    if kind == 'E':
        return ExplicitFD(npts, order, length)
    if kind == 'C':
        return CompactFD(npts, order, length)
    if kind in ('S', 'SBP'):
        return SBPSAT(npts, order, length)
    if kind in ('D', 'DRP'):
        return DRP(npts, order, length)
    if kind == 'B':
        return ClampedBSpline(npts, order, length)
    if kind == 'I':
        return BoundaryBMG(npts, order, length)
    raise ValueError("kind must be E, C, SBP/S, DRP/D, B, or I")



def wall_mode_values(x: Array, mode: int) -> tuple[Array, Array, Array]:
    x = np.asarray(x, dtype=float)
    s = np.sin(mode * np.pi * x)
    c = np.cos(mode * np.pi * x)
    ds = mode * np.pi * c
    return s, c, ds


def _quadratic_form(metric: Array, u: Array, v: Array | None = None) -> float:
    m = np.asarray(metric, dtype=float)
    a = np.asarray(u, dtype=float)
    b = a if v is None else np.asarray(v, dtype=float)
    if m.ndim == 1:
        return float(np.sum(m * a * b))
    return float(a @ m @ b)



def wall_mode_response(operator: Operator1D, mode_max: int | None = None, length: float = 1.0) -> list[WallModeResponse]:
    x = operator.x
    npts = x.size
    metric = nodal_trapezoidal_weights(x)
    h = reference_spacing(x)
    max_mode = int(mode_max) if mode_max is not None else min(npts - 2, int(np.floor((1.0 - 1.0e-12) / h)))
    out: list[WallModeResponse] = []
    for mode in range(1, max_mode + 1):
        s, c, ds = wall_mode_values(x, mode)
        num = operator.differentiate(s)
        denom = _quadratic_form(metric, c)
        kappa = _quadratic_form(metric, num, c) / denom
        contamination = float(np.sqrt(_quadratic_form(metric, num - kappa * c)) / (mode * np.pi * np.sqrt(denom)))
        rel_max = float(np.max(np.abs(num - ds)) / (mode * np.pi))
        rel_l2 = float(np.sqrt(_quadratic_form(metric, num - ds) / _quadratic_form(metric, ds)))
        ppw = float(2.0 / (mode * h))
        out.append(WallModeResponse(mode=mode, ppw=ppw, kappa_ratio=kappa / (mode * np.pi), contamination_l2=contamination, rel_max=rel_max, rel_l2=rel_l2))
    return out



def standing_wave_error_map(operator: Operator1D, mode_max: int | None = None) -> tuple[Array, Array, Array, Array]:
    x = operator.x
    npts = x.size
    max_mode = int(mode_max) if mode_max is not None else npts - 2
    modes = np.arange(1, max_mode + 1, dtype=int)
    errors = np.zeros((modes.size, npts), dtype=float)
    rel_max = np.zeros(modes.size, dtype=float)
    rel_l2 = np.zeros(modes.size, dtype=float)
    for row, mode in enumerate(modes):
        s, _, ds = wall_mode_values(x, int(mode))
        num = operator.differentiate(s)
        scale = mode * np.pi
        pointwise = np.abs(num - ds) / (scale + 1.0e-30)
        errors[row, :] = pointwise
        rel_max[row] = float(np.max(pointwise))
        rel_l2[row] = float(np.linalg.norm(num - ds) / np.linalg.norm(ds))
    ppw = 2.0 / (modes * (x[1] - x[0]))
    return modes, ppw, errors, np.vstack([rel_max, rel_l2])



def threshold_ppw(ppw: Array, metric: Array, tol: float) -> float:
    idx = np.where(metric <= tol)[0]
    if idx.size == 0:
        return float('nan')
    return float(ppw[idx[-1]])



def boundary_layer_function(beta: float = 40.0, omega: float = 8.0 * np.pi):
    def f(x: Array) -> Array:
        x_arr = np.asarray(x, dtype=float)
        return np.exp(-beta * x_arr) * np.sin(omega * x_arr)

    def df(x: Array) -> Array:
        x_arr = np.asarray(x, dtype=float)
        return np.exp(-beta * x_arr) * (omega * np.cos(omega * x_arr) - beta * np.sin(omega * x_arr))

    return f, df



def mirror_packet_family(sigma: float = 0.06, mode: int = 18):
    k = mode * np.pi

    def f(x: Array, xi: float) -> Array:
        x_arr = np.asarray(x, dtype=float)
        y = x_arr - xi
        y_img = x_arr + xi
        return np.exp(-0.5 * (y / sigma) ** 2) * np.cos(k * y) - np.exp(-0.5 * (y_img / sigma) ** 2) * np.cos(k * y_img)

    def df(x: Array, xi: float) -> Array:
        x_arr = np.asarray(x, dtype=float)
        y = x_arr - xi
        y_img = x_arr + xi
        q = np.exp(-0.5 * (y / sigma) ** 2)
        q_img = np.exp(-0.5 * (y_img / sigma) ** 2)
        first = q * (-(y / sigma ** 2) * np.cos(k * y) - k * np.sin(k * y))
        second = q_img * (-(y_img / sigma ** 2) * np.cos(k * y_img) - k * np.sin(k * y_img))
        return first - second

    return f, df



def l2_relative_error(num: Array, exact: Array) -> float:
    num = np.asarray(num, dtype=float)
    exact = np.asarray(exact, dtype=float)
    denom = np.linalg.norm(exact)
    return float(np.linalg.norm(num - exact) / denom)



def max_abs_error(num: Array, exact: Array) -> float:
    return float(np.max(np.abs(np.asarray(num, dtype=float) - np.asarray(exact, dtype=float))))



def observed_order(grid_sizes: Sequence[int], errors: Sequence[float]) -> list[float]:
    sizes = list(grid_sizes)
    errs = list(errors)
    out: list[float] = []
    for i in range(1, len(errs)):
        h_ratio = (sizes[i] - 1) / (sizes[i - 1] - 1)
        out.append(float(np.log(errs[i - 1] / errs[i]) / np.log(h_ratio)))
    return out



def diagonal_reference_line(x0: float, y0: float, slope: float, xvals: Array) -> Array:
    x = np.asarray(xvals, dtype=float)
    return y0 * (x / x0) ** slope



def row_sum_norm(mat: Array) -> float:
    return float(np.linalg.norm(np.sum(np.asarray(mat, dtype=float), axis=1), ord=np.inf))



def weighted_sbp_residual(dmat: Array, metric: Array, npts: int | None = None) -> Array:
    arr = metric @ dmat + dmat.T @ metric
    if npts is None:
        npts = arr.shape[0]
    return arr - endpoint_flux_matrix(npts)



def rigid_wall_acoustics_packet(x0: float, sigma: float, k: float):
    """Exact reflected packet for 1D linear acoustics on x in [0, 1].

    The system is
        p_t + u_x = 0,
        u_t + p_x = 0,
    with the rigid-wall condition u(0, t) = 0.  The returned pressure and
    velocity fields are generated by an image pair so that the left-going
    incident packet reflects into a right-going packet.
    """

    def g(z: Array) -> Array:
        zz = np.asarray(z, dtype=float)
        y = zz - float(x0)
        return np.exp(-0.5 * (y / float(sigma)) ** 2) * np.cos(float(k) * y)

    def pressure(x: Array, t: float) -> Array:
        xx = np.asarray(x, dtype=float)
        return g(xx + float(t)) + g(float(t) - xx)

    def velocity(x: Array, t: float) -> Array:
        xx = np.asarray(x, dtype=float)
        return g(float(t) - xx) - g(xx + float(t))

    return pressure, velocity


def coupled_relative_l2(num_a: Array, exact_a: Array, num_b: Array, exact_b: Array) -> float:
    num_a = np.asarray(num_a, dtype=float)
    exact_a = np.asarray(exact_a, dtype=float)
    num_b = np.asarray(num_b, dtype=float)
    exact_b = np.asarray(exact_b, dtype=float)
    numer = np.linalg.norm(num_a - exact_a) ** 2 + np.linalg.norm(num_b - exact_b) ** 2
    denom = np.linalg.norm(exact_a) ** 2 + np.linalg.norm(exact_b) ** 2
    return float(np.sqrt(numer / denom))


def trimmed_interior_slice(npts: int, interior_strip: int) -> slice:
    strip = int(interior_strip)
    npts = int(npts)
    if strip < 0:
        raise ValueError('interior_strip must be nonnegative')
    if 2 * strip >= npts:
        raise ValueError('interior_strip trims away the full domain')
    return slice(strip, npts - strip)


def trimmed_interior_bounds(x: Array, interior_strip: int) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    sl = trimmed_interior_slice(x.size, interior_strip)
    return float(x[sl.start]), float(x[sl.stop - 1])


def reflected_history_peak(times: Array, error_history: Array, t_hit: float, pre_bounce_window: float) -> float:
    tau = np.asarray(times, dtype=float) - float(t_hit)
    errs = np.asarray(error_history, dtype=float)
    mask = tau >= -float(pre_bounce_window)
    if not np.any(mask):
        raise ValueError('bounce window does not contain any recorded samples')
    return float(np.max(errs[mask]))


@dataclass(frozen=True)
class ReflectedAcousticsResult:
    times: Array
    error_history: Array
    pressure: Array
    velocity: Array
    pressure_exact: Array
    velocity_exact: Array
    work_per_point: float
    total_work: int
    operator_route: str = "same-grid first-derivative route"
    grid: Array | None = None



def simulate_reflected_acoustics(
    operator: Operator1D,
    pressure_exact,
    velocity_exact,
    t_end: float,
    *,
    cfl: float = 0.012,
    interior_strip: int = 6,
    record_count: int = 160,
) -> ReflectedAcousticsResult:
    """Strong-BC RK4 evolution for the reflected 1D acoustics packet.

    The operator itself is the only ingredient that changes between runs.
    The rigid-wall velocity trace is imposed strongly, while pressure is
    advanced at every pressure node. This keeps the wall treatment aligned
    with the acoustic boundary condition instead of injecting exact pressure
    data at the boundary.
    For BMF the evolution uses the staggered flux complex:
    pressure lives on the primal Greville grid, velocity lives on the
    derivative Greville grid, and the returned velocity is reconstructed to
    the primal grid only for the common post-bounce error metric.
    """
    if scheme_family(operator.name) == "BMF":
        return _simulate_reflected_acoustics_bmf_staggered(
            operator,
            pressure_exact,
            velocity_exact,
            t_end,
            cfl=cfl,
            interior_strip=interior_strip,
            record_count=record_count,
        )
    if scheme_family(operator.name) == "BMG":
        # dense natural metric: the velocity trace is enforced by the
        # metric-orthogonal projection of Definition def:strong-enforcement
        # (for the diagonal-metric nodal families below, the selector
        # injection IS that projection, so their path is unchanged)
        return _simulate_reflected_acoustics_horth_samegrid(
            operator,
            pressure_exact,
            velocity_exact,
            t_end,
            cfl=cfl,
            interior_strip=interior_strip,
            record_count=record_count,
        )

    x = operator.x
    npts = x.size
    if npts < 4:
        raise ValueError('need at least four grid points')
    h = x[1] - x[0]
    dt = float(cfl) * float(h)
    nsteps = int(np.ceil(float(t_end) / dt))
    dt = float(t_end) / nsteps
    dmat = operator.matrix
    m_u = npts - 2
    sl = trimmed_interior_slice(npts, interior_strip)
    y = np.concatenate([pressure_exact(x, 0.0), velocity_exact(x, 0.0)[1:-1]])

    keep_every = max(1, nsteps // max(2, int(record_count) - 1))
    rec_t = []
    rec_e = []

    def unpack(state: Array, t: float) -> tuple[Array, Array]:
        p = np.asarray(state[:npts], dtype=float)
        u = np.empty(npts, dtype=float)
        u[0] = float(np.asarray(velocity_exact(x[0], t)))
        u[-1] = float(np.asarray(velocity_exact(x[-1], t)))
        u[1:-1] = state[npts:npts + m_u]
        return p, u

    def rhs(t: float, state: Array) -> Array:
        p, u = unpack(state, t)
        dp = -dmat @ u
        du = -dmat @ p
        return np.concatenate([dp, du[1:-1]])

    def record(t: float, state: Array) -> None:
        p, u = unpack(state, t)
        pex = pressure_exact(x, t)
        uex = velocity_exact(x, t)
        err = coupled_relative_l2(p[sl], pex[sl], u[sl], uex[sl])
        rec_t.append(float(t))
        rec_e.append(float(err))

    record(0.0, y)
    t = 0.0
    for step in range(nsteps):
        k1 = rhs(t, y)
        k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = rhs(t + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        if step == nsteps - 1 or (step + 1) % keep_every == 0:
            record(t, y)

    p, u = unpack(y, float(t_end))
    pex = pressure_exact(x, float(t_end))
    uex = velocity_exact(x, float(t_end))
    return ReflectedAcousticsResult(
        times=np.asarray(rec_t, dtype=float),
        error_history=np.asarray(rec_e, dtype=float),
        pressure=p,
        velocity=u,
        pressure_exact=np.asarray(pex, dtype=float),
        velocity_exact=np.asarray(uex, dtype=float),
        work_per_point=float(operator.work_model.total_flops / npts),
        total_work=int(operator.work_model.total_flops),
        operator_route="same-grid first-derivative route",
        grid=x.copy(),
    )


def _horth_wave_lifting(metric_v: Array, nv: int, offset: int, ntot: int) -> tuple[Array, Array]:
    """H-orthogonal lifting for the two velocity-trace rows of a wave pair.

    The state is [p; v] with v of size nv starting at `offset` in a state of
    size ntot; the traces are the first and last velocity entries; the
    energy metric on the velocity block is `metric_v` (dense for the spline
    routes). Returns (L, X_H) with L X_H = I."""
    L = np.zeros((2, ntot))
    L[0, offset] = 1.0
    L[1, offset + nv - 1] = 1.0
    E = np.zeros((nv, 2))
    E[0, 0] = 1.0
    E[-1, 1] = 1.0
    HiE = np.linalg.solve(metric_v, E)
    XH = np.zeros((ntot, 2))
    XH[offset:offset + nv, :] = HiE @ np.linalg.inv(E.T @ HiE)
    return L, XH


def _simulate_reflected_acoustics_horth_samegrid(
    operator: Operator1D,
    pressure_exact,
    velocity_exact,
    t_end: float,
    *,
    cfl: float,
    interior_strip: int,
    record_count: int,
) -> ReflectedAcousticsResult:
    """Same-grid wave pair marched under metric-orthogonal strong enforcement
    of the velocity trace (Definition def:strong-enforcement): the full state
    evolves as z' = A z + X_H (g' - L A z), which keeps L z = g exactly and
    carries the exact energy balance of thm:strong-bc-sbp."""
    x = operator.x
    npts = x.size
    if npts < 4:
        raise ValueError('need at least four grid points')
    h = x[1] - x[0]
    dt = float(cfl) * float(h)
    nsteps = int(np.ceil(float(t_end) / dt))
    dt = float(t_end) / nsteps
    dmat = operator.matrix
    metric = bmg_nodal_operator(npts, operator.order)[2]
    Z = np.zeros((npts, npts))
    A = np.block([[Z, -dmat], [-dmat, Z]])
    L, XH = _horth_wave_lifting(metric, npts, npts, 2 * npts)
    sl = trimmed_interior_slice(npts, interior_strip)

    def g(t: float) -> Array:
        return np.array([float(np.asarray(velocity_exact(x[0], t))),
                         float(np.asarray(velocity_exact(x[-1], t)))])

    def gdot(t: float, eps: float = 1.0e-6) -> Array:
        return (g(t + eps) - g(t - eps)) / (2.0 * eps)

    def rhs(t: float, z: Array) -> Array:
        Az = A @ z
        return Az + XH @ (gdot(t) - L @ Az)

    y = np.concatenate([pressure_exact(x, 0.0), velocity_exact(x, 0.0)])
    keep_every = max(1, nsteps // max(2, int(record_count) - 1))
    rec_t: list[float] = []
    rec_e: list[float] = []

    def record(t: float, z: Array) -> None:
        p = z[:npts]
        u = z[npts:]
        pex = pressure_exact(x, t)
        uex = velocity_exact(x, t)
        rec_t.append(float(t))
        rec_e.append(float(coupled_relative_l2(p[sl], pex[sl], u[sl], uex[sl])))

    record(0.0, y)
    t = 0.0
    for step in range(nsteps):
        k1 = rhs(t, y)
        k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = rhs(t + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        y = y + XH @ (g(t) - L @ y)      # roundoff hygiene on the constraint
        if step == nsteps - 1 or (step + 1) % keep_every == 0:
            record(t, y)

    p = y[:npts]
    u = y[npts:]
    pex = pressure_exact(x, float(t_end))
    uex = velocity_exact(x, float(t_end))
    return ReflectedAcousticsResult(
        times=np.asarray(rec_t, dtype=float),
        error_history=np.asarray(rec_e, dtype=float),
        pressure=np.asarray(p, dtype=float),
        velocity=np.asarray(u, dtype=float),
        pressure_exact=np.asarray(pex, dtype=float),
        velocity_exact=np.asarray(uex, dtype=float),
        work_per_point=float(operator.work_model.total_flops / npts),
        total_work=int(operator.work_model.total_flops),
        operator_route="same-grid first-derivative route, metric-orthogonal enforcement",
        grid=x.copy(),
    )


def _simulate_reflected_acoustics_bmf_staggered(
    operator: Operator1D,
    pressure_exact,
    velocity_exact,
    t_end: float,
    *,
    cfl: float,
    interior_strip: int,
    record_count: int,
) -> ReflectedAcousticsResult:
    """Reflected acoustics evolved with the staggered BMF SBP pair."""
    if not isinstance(operator, ClampedBSpline):
        raise TypeError("BMF staggered reflected acoustics requires ClampedBSpline")

    ops = operator.staggered_operators()
    x = ops.x
    eta = ops.eta
    npts = x.size
    nvel = eta.size
    if npts < 4 or nvel < 3:
        raise ValueError("need at least four primal points for staggered BMF acoustics")

    h = min(float(np.min(np.diff(x))), float(np.min(np.diff(eta))))
    dt = float(cfl) * h
    nsteps = int(np.ceil(float(t_end) / dt))
    dt = float(t_end) / nsteps

    # Reconstruct derivative-grid velocity values to the primal grid only
    # after the staggered update, so every family is scored on the same
    # pressure/velocity locations without changing the BMF evolution.
    eta_to_x = ops.bhat_x @ np.linalg.inv(ops.bhat_eta)

    sl = trimmed_interior_slice(npts, interior_strip)
    # full staggered state [p; v]; the velocity trace is enforced by the
    # metric-orthogonal projection of Definition def:strong-enforcement in
    # the dense derivative-grid metric h1
    y = np.concatenate([
        pressure_exact(x, 0.0),
        velocity_exact(eta, 0.0),
    ])
    Lc, XH = _horth_wave_lifting(ops.h1, nvel, npts, npts + nvel)
    Zpp = np.zeros((npts, npts))
    Zvv = np.zeros((nvel, nvel))
    Afull = np.block([[Zpp, -ops.d01], [-ops.d10, Zvv]])

    keep_every = max(1, nsteps // max(2, int(record_count) - 1))
    rec_t: list[float] = []
    rec_e: list[float] = []

    def gfun(t: float) -> Array:
        return np.array([float(np.asarray(velocity_exact(eta[0], t))),
                         float(np.asarray(velocity_exact(eta[-1], t)))])

    def gdot(t: float, eps: float = 1.0e-6) -> Array:
        return (gfun(t + eps) - gfun(t - eps)) / (2.0 * eps)

    def unpack(state: Array, t: float) -> tuple[Array, Array]:
        p = np.asarray(state[:npts], dtype=float)
        u_eta = np.asarray(state[npts:npts + nvel], dtype=float)
        return p, u_eta

    def reconstruct_velocity(u_eta: Array) -> Array:
        return eta_to_x @ np.asarray(u_eta, dtype=float)

    def rhs(t: float, state: Array) -> Array:
        Az = Afull @ state
        return Az + XH @ (gdot(t) - Lc @ Az)

    def record(t: float, state: Array) -> None:
        p, u_eta = unpack(state, t)
        u = reconstruct_velocity(u_eta)
        pex = pressure_exact(x, t)
        uex = velocity_exact(x, t)
        err = coupled_relative_l2(p[sl], pex[sl], u[sl], uex[sl])
        rec_t.append(float(t))
        rec_e.append(float(err))

    record(0.0, y)
    t = 0.0
    for step in range(nsteps):
        k1 = rhs(t, y)
        k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = rhs(t + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        y = y + XH @ (gfun(t) - Lc @ y)  # roundoff hygiene on the constraint
        if step == nsteps - 1 or (step + 1) % keep_every == 0:
            record(t, y)

    p, u_eta = unpack(y, float(t_end))
    u = reconstruct_velocity(u_eta)
    pex = pressure_exact(x, float(t_end))
    uex = velocity_exact(x, float(t_end))
    work_total = int(np.count_nonzero(np.abs(ops.d10) > 1.0e-14) * 2
                     + np.count_nonzero(np.abs(ops.d01) > 1.0e-14) * 2)
    return ReflectedAcousticsResult(
        times=np.asarray(rec_t, dtype=float),
        error_history=np.asarray(rec_e, dtype=float),
        pressure=p,
        velocity=u,
        pressure_exact=np.asarray(pex, dtype=float),
        velocity_exact=np.asarray(uex, dtype=float),
        work_per_point=float(work_total / npts),
        total_work=work_total,
        operator_route="staggered primal-pressure and derivative-grid velocity complex",
        grid=x.copy(),
    )


def exact_semidiscrete_evolution(dmat: Array, u0: Array, times: Array) -> Array:
    from scipy.sparse.linalg import expm_multiply

    tvals = np.asarray(times, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    if tvals.ndim != 1 or tvals.size == 0:
        raise ValueError("times must be a nonempty one-dimensional array")
    if tvals.size == 1:
        return expm_multiply(-float(tvals[0]) * dmat, u0)[None, :]
    steps = np.diff(tvals)
    if np.allclose(steps, steps[0], rtol=1.0e-12, atol=1.0e-14):
        return np.asarray(
            expm_multiply(
                -np.asarray(dmat, dtype=float),
                u0,
                start=float(tvals[0]),
                stop=float(tvals[-1]),
                num=tvals.size,
                endpoint=True,
            ),
            dtype=float,
        )
    return np.asarray([expm_multiply(-float(t) * dmat, u0) for t in tvals], dtype=float)


def energy_budget_defect(dmat: Array, metric: Array, u0: Array, times: Array) -> tuple[Array, Array, Array]:
    tvals = np.asarray(times, dtype=float)
    sol = exact_semidiscrete_evolution(dmat, np.asarray(u0, dtype=float), tvals)
    B = endpoint_flux_matrix(dmat.shape[0])
    energy = np.array([0.5 * u @ metric @ u for u in sol], dtype=float)
    boundary_flux = np.array([-0.5 * u @ B @ u for u in sol], dtype=float)
    cumulative = np.zeros_like(tvals)
    for i in range(1, tvals.size):
        cumulative[i] = cumulative[i - 1] + 0.5 * (boundary_flux[i] + boundary_flux[i - 1]) * (tvals[i] - tvals[i - 1])
    defect = energy - energy[0] - cumulative
    return sol, energy, defect


# =====================================================================================
# Full-strength finite-difference competitors.
#
# APPEND-ONLY. Everything below is new (new builder functions, new Operator1D
# subclasses, new scheme labels 'C6F' / 'SBP6-BO' / 'C6-D2'). Nothing above this
# banner is renamed, reordered, or altered, and no existing function's return
# value changes for any existing call site. This keeps the module safe to import
# from long-running sibling processes while this addendum is written.
#
# Three competitors are added, each verified against a primary or primary-adjacent
# source (see code/verify_strong_competitors.py and .superpowers/sdd/task-2-report.md
# for the gate outputs and exact citations):
#
#   C6F     Standard sixth-order compact (Pade) first derivative (the existing
#           `compact_fd_matrices(order=6)` interior + one-sided closures already in
#           this module) with the
#           implicit tenth-order low-pass filter applied once per RK stage-update
#           (i.e. once per full RK4 step, after the stage combination -- see
#           `CompactFilteredFD` below and its use in the experiment drivers).
#
#   SBP6-BO Boundary-optimized diagonal-norm SBP6 (JCP 374, 2018), transcribed
#           from the open-source SummationByPartsOperators.jl coefficient tables
#           (order-6 "Accurate" boundary-optimized set). This
#           is the "Accurate" variant (the paper's higher-accuracy family; the
#           "Minimal" variant trades some accuracy for a narrower stencil).
#
#   C6-D2   the classical compact-difference reference sixth-order compact second-derivative interior
#           scheme (eq. 2.2.7: alpha=2/11, a=12/11, b=3/11) with the one-sided
#           boundary closures given in Sec. 4.3: the third-order compact wall
#           row (eq. 4.3.1) at node 1, and the standard fourth-order Pade second-
#           derivative formula (eq. 2.2.6 family at alpha=1/10) applied directly
#           at node 2, which the paper states explicitly ("at node 2 ... the standard
#           Pade form is used").
# =====================================================================================


# ---------------------------------------------------------------------------
# C6-D2: the classical compact-difference reference compact second-derivative operator.
# ---------------------------------------------------------------------------

def compact_baseline_c6_d2_interior_coefficients() -> tuple[float, float, float]:
    """the classical compact-difference reference eq. (2.2.7): sixth-order tridiagonal compact second derivative.

    alpha * (f''_{i-1} + f''_{i+1}) + f''_i
        = a * (f_{i+1} - 2 f_i + f_{i-1}) / h^2
        + b * (f_{i+2} - 2 f_i + f_{i-2}) / (4 h^2)

    Returns (alpha, a, b); the published c = 0 for this member of the family.
    """
    return 2.0 / 11.0, 12.0 / 11.0, 3.0 / 11.0


@lru_cache(maxsize=None)
def compact_baseline_c6_d2_matrices(npts: int, length: float = 1.0) -> tuple[Array, Array, Array]:
    """Assemble the (A, B) pencil for the published compact D2: A @ f'' = B @ f.

    Interior rows use eq. (2.2.7). Node 2 (and its mirror at N-3) use the
    standard fourth-order Pade compact D2 (the published eq. 2.2.6 family at
    alpha=1/10, i.e. a=6/5, b=0), which the reference states explicitly is the
    conventional choice there since node 2 already has full symmetric
    three-point access. Node 1 (and its mirror at N-2, the wall rows) use
    the published third-order compact boundary closure, eq. (4.3.1):

        f''_1 + 11 f''_2 = (13 f_1 - 27 f_2 + 15 f_3 - f_4) / h^2.
    """
    x = uniform_grid(npts, length)
    h = x[1] - x[0]
    if npts < 6:
        raise ValueError("need at least six points for the compact C6-D2 closure")
    alpha, a, b = compact_baseline_c6_d2_interior_coefficients()
    # Node-2 standard Pade D2 (alpha=1/10, a=6/5, b=0; eq. 2.2.6 at alpha=1/10).
    alpha2, a2 = 0.1, 1.2

    amat = np.zeros((npts, npts), dtype=float)
    bmat = np.zeros((npts, npts), dtype=float)

    # Wall row (node 1, index 0): eq. (4.3.1), f''_1 + 11 f''_2 = (13 f1 -27 f2 +15 f3 - f4)/h^2.
    amat[0, 0] = 1.0
    amat[0, 1] = 11.0
    bmat[0, 0] = 13.0 / h ** 2
    bmat[0, 1] = -27.0 / h ** 2
    bmat[0, 2] = 15.0 / h ** 2
    bmat[0, 3] = -1.0 / h ** 2

    # Node 2 (index 1): standard fourth-order Pade D2, symmetric tridiagonal row.
    amat[1, 0] = alpha2
    amat[1, 1] = 1.0
    amat[1, 2] = alpha2
    bmat[1, 0] = a2 / h ** 2
    bmat[1, 1] = -2.0 * a2 / h ** 2
    bmat[1, 2] = a2 / h ** 2

    # Interior rows (indices 2 .. npts-3): eq. (2.2.7).
    for j in range(2, npts - 2):
        amat[j, j - 1] = alpha
        amat[j, j] = 1.0
        amat[j, j + 1] = alpha
        bmat[j, j - 1] += a / h ** 2
        bmat[j, j] += -2.0 * a / h ** 2
        bmat[j, j + 1] += a / h ** 2
        bmat[j, j - 2] += b / (4.0 * h ** 2)
        bmat[j, j] += -2.0 * b / (4.0 * h ** 2)
        bmat[j, j + 2] += b / (4.0 * h ** 2)

    # Mirror closures at the right wall.
    amat[-1, -1] = 1.0
    amat[-1, -2] = 11.0
    bmat[-1, -1] = 13.0 / h ** 2
    bmat[-1, -2] = -27.0 / h ** 2
    bmat[-1, -3] = 15.0 / h ** 2
    bmat[-1, -4] = -1.0 / h ** 2

    amat[-2, -1] = alpha2
    amat[-2, -2] = 1.0
    amat[-2, -3] = alpha2
    bmat[-2, -1] = a2 / h ** 2
    bmat[-2, -2] = -2.0 * a2 / h ** 2
    bmat[-2, -3] = a2 / h ** 2

    return x.copy(), amat, bmat


def compact_baseline_c6_d2_operator_matrix(npts: int, length: float = 1.0) -> Array:
    """Dense D2 matrix for the published compact second derivative: solve A f'' = B f."""
    _x, amat, bmat = compact_baseline_c6_d2_matrices(npts, length)
    lu = lu_factor(amat)
    return lu_solve(lu, bmat)


class CompactD2FD(CompactFD):
    """C6 first derivative (unchanged) plus the published compact D2 as second derivative.

    Subclasses the existing `CompactFD` so the first-derivative route (and its
    wall-mode / reflected-acoustics behaviour) is byte-for-byte identical to the
    already-verified `C6` operator; the only addition is a scheme-level
    `second_derivative_matrix()` so the heat-cavity driver can select the published
    tailored D2 instead of the composed `D @ D` fallback.
    """

    def __init__(self, npts: int, order: int, length: float = 1.0) -> None:
        if int(order) != 6:
            raise ValueError("C6-D2 is only defined for order=6 in this codebase")
        super().__init__(npts, order, length)
        self.name = "C6-D2"
        self._compact_baseline_d2: Array | None = None

    def second_derivative_matrix(self) -> Array:
        if self._compact_baseline_d2 is None:
            self._compact_baseline_d2 = compact_baseline_c6_d2_operator_matrix(self._x.size, self._x[-1] - self._x[0])
        return self._compact_baseline_d2


# ---------------------------------------------------------------------------
# SBP6-BO: the 2018 boundary-optimized SBP6 construction boundary-optimized,
# non-equispaced diagonal-norm SBP6 operator ("Accurate" variant).
# ---------------------------------------------------------------------------

def sbp6_bo_reference_nodes() -> tuple[float, ...]:
    """Non-equispaced boundary-node offsets (in units of the interior spacing).

    Transcribed from the order-6 "Accurate" boundary-optimized grid in the
    open-source SummationByPartsOperators.jl coefficient tables ("Boundary
    optimized diagonal-norm SBP operators," J. Comput. Phys. 374 (2018)
    1261-1266). Offsets are relative to the wall node x_0=0, in units
    where the asymptotic interior spacing is 1; the last three offsets are
    exactly 1.0 apart, confirming the grid returns to uniform spacing away
    from the wall.
    """
    return (
        0.0000000000000e+00,
        4.4090263368623e-01,
        1.2855984345073e+00,
        2.2638953951239e+00,
        3.2638953951239e+00,
        4.2638953951239e+00,
        5.2638953951239e+00,
    )


def sbp6_bo_boundary_rows() -> tuple[tuple[float, ...], ...]:
    """Left-boundary first-derivative rows for SBP6-BO (rows d1..d6), Delta_x=1.

    Transcribed from `first_derivative_coefficients` (order=6 branch) in the
    same source file. Each row gives the (non-equispaced) FD weights at the
    corresponding boundary node; divide by the physical Delta_x to use on an
    actual grid (matching the convention already used by `sbp_fd_matrix` in
    this module for the equispaced classical SBP operator).
    """
    return (
        (-3.837232862086773, 5.068376175016296, -1.6963756420892036, 0.5851295073874213,
         -0.13275449453716695, 0.012857316309426029),
        (-0.9591958116075092, 0.0, 1.2687130518352143, -0.3860731957925361,
         0.08345276954506707, -0.006896813980236088),
        (0.23226893958753303, -0.9178972823223088, 0.0, 0.8585728949642006,
         -0.19840158580452047, 0.0254570335750956),
        (-0.07693308450770664, 0.2682207441654631, -0.8244592785032373, 0.0,
         0.7749305933808132, -0.15857634324473752, 0.016817368709408833),
        (0.017248604775379643, -0.057293724956320936, 0.1882698460378496, -0.7657841952075574,
         0.0, 0.7505104744873311, -0.14956988077877112, 0.01661887564208568),
        (-0.0016761725437960326, 0.004750928272020083, -0.02423857479897864, 0.15723370478311835,
         -0.753044107168037, 0.0, 0.7503740531217644, -0.15007481062435288, 0.016674978958261432),
    )


def sbp6_bo_interior_upper_coefficients() -> tuple[float, float, float]:
    """Interior sixth-order central stencil (identical to the classical SBP6/C6 family)."""
    return 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0


def sbp6_bo_boundary_weights() -> tuple[float, ...]:
    """Diagonal-norm weights H_ii/Delta_x for the six left-boundary nodes (order 6).

    Transcribed from the same source file's `left_weights` for order=6.
    """
    return (
        0.13030223027124,
        0.68851501587715,
        0.95166202564389,
        0.99103890475697,
        1.0028757074552,
        0.99950151111941,
    )


@lru_cache(maxsize=None)
def sbp6_bo_grid(npts: int, length: float = 1.0) -> Array:
    """Boundary-optimized non-equispaced grid on [0, length] with npts nodes.

    Six clustered nodes at each wall (offsets from `sbp6_bo_reference_nodes`),
    uniform spacing in the interior, mirrored at the right wall. The uniform
    spacing Delta_x is fixed by requiring the total length match `length`.
    """
    offsets = np.asarray(sbp6_bo_reference_nodes(), dtype=float)
    m = offsets.size  # 7 boundary nodes (indices 0..6), node 6 already at unit interior spacing
    if npts < 2 * m + 1:
        raise ValueError(f"need at least {2 * m + 1} points for SBP6-BO")
    # (npts-1) total physical intervals; interior (uniform) intervals count:
    n_interior_gaps = (npts - 1) - 2 * (m - 1)
    if n_interior_gaps < 1:
        raise ValueError("grid too coarse for SBP6-BO boundary clustering")
    total_ref_span = 2.0 * offsets[-1] + n_interior_gaps
    dx = float(length) / total_ref_span
    left = offsets * dx
    right = float(length) - offsets[::-1] * dx
    n_mid = npts - 2 * m
    if n_mid > 0:
        mid = left[-1] + dx * np.arange(1, n_mid + 1)
        x = np.concatenate([left, mid, right])
    else:
        x = np.concatenate([left, right])
    return x


@lru_cache(maxsize=None)
def sbp6_bo_fd_matrix(npts: int, length: float = 1.0) -> tuple[Array, Array, Array]:
    """Assemble the SBP6-BO first-derivative matrix and diagonal norm H.

    Mirrors the structure of `sbp_fd_matrix` for the classical equispaced SBP
    operator already in this module: dense D on the non-equispaced grid plus
    a diagonal metric H such that H D + D^T H - B = 0 (B the boundary bilinear
    form diag(-1, 0, ..., 0, 1)).
    """
    x = sbp6_bo_grid(npts, length)
    offsets = np.asarray(sbp6_bo_reference_nodes(), dtype=float)
    m_grid = offsets.size
    total_ref_span = 2.0 * offsets[-1] + ((npts - 1) - 2 * (m_grid - 1))
    dx = float(length) / total_ref_span

    rows = sbp6_bo_boundary_rows()
    upper = sbp6_bo_interior_upper_coefficients()
    weights = sbp6_bo_boundary_weights()
    m = len(rows)  # 6 boundary-closure rows/weights (distinct from the 7-entry grid offsets)

    dmat = np.zeros((npts, npts), dtype=float)
    for j, row in enumerate(rows):
        dmat[j, :len(row)] = np.asarray(row, dtype=float) / dx
    for j in range(m, npts - m):
        for k, coef in enumerate(upper, start=1):
            dmat[j, j - k] = -coef / dx
            dmat[j, j + k] = coef / dx
    for idx in range(m):
        row = np.asarray(rows[m - 1 - idx], dtype=float)
        dmat[npts - m + idx, npts - row.size:] = -row[::-1] / dx

    diag = np.ones(npts, dtype=float)
    diag[:m] = np.asarray(weights, dtype=float)
    diag[-m:] = np.asarray(weights[::-1], dtype=float)
    metric = dx * np.diag(diag)
    return x.copy(), dmat, metric


@lru_cache(maxsize=None)
def sbp6_bo_second_derivative_matrix(npts: int, length: float = 1.0) -> tuple[Array, Array]:
    """Compatible SBP second derivative built from SBP6-BO's own (D, H).

    The the 2018 boundary-optimized SBP6 construction paper is a first-derivative
    paper; it does not publish a second-derivative operator. The standard
    SBP construction (already used elsewhere in this module for the
    classical operator, see `sbp_second_derivative_matrix`) builds a
    compatible D2 from ANY first-derivative SBP pair (D, H) via
        D2 = H^{-1} ( -D^T H D - e0 s0 + eN sN ),
    where s0/sN are the operator's own boundary derivative-trace rows. This
    is a derived, not transcribed, second-derivative operator: it inherits
    SBP6-BO's non-equispaced node placement and diagonal norm exactly, and
    the construction itself (not new coefficients) is the published method.
    """
    x, dmat, hmat = sbp6_bo_fd_matrix(npts, length)
    s0 = dmat[[0], :].copy()
    sN = dmat[[-1], :].copy()
    e0 = np.zeros((npts, 1), dtype=float)
    eN = np.zeros((npts, 1), dtype=float)
    e0[0, 0] = 1.0
    eN[-1, 0] = 1.0
    d2mat = np.linalg.solve(hmat, -dmat.T @ hmat @ dmat - e0 @ s0 + eN @ sN)
    return x.copy(), d2mat


class SBPBoundaryOptimized(Operator1D):
    """SBP6-BO: the 2018 boundary-optimized SBP6 construction boundary-optimized SBP6."""

    def __init__(self, npts: int, order: int, length: float = 1.0) -> None:
        if int(order) != 6:
            raise ValueError("SBP6-BO is only defined for order=6 in this codebase")
        x, dmat, metric = sbp6_bo_fd_matrix(npts, length)
        self.name = "SBP6-BO"
        self.order = 6
        self._x = x
        self._dmat = dmat
        self._metric = metric
        self._d2mat: Array | None = None
        storage = int(np.count_nonzero(np.abs(dmat) > 1.0e-14) + np.count_nonzero(np.abs(metric) > 1.0e-14))
        grid = int(sum(2 * np.count_nonzero(np.abs(row) > 1.0e-14) - 1 for row in dmat))
        self._work = WorkModel(grid_apply_flops=grid, solve_flops=0, total_flops=grid, storage_nnz=storage)

    @property
    def x(self) -> Array:
        return self._x

    @property
    def matrix(self) -> Array:
        return self._dmat

    @property
    def metric(self) -> Array:
        return self._metric

    @property
    def work_model(self) -> WorkModel:
        return self._work

    def second_derivative_matrix(self) -> Array:
        if self._d2mat is None:
            self._d2mat = sbp6_bo_second_derivative_matrix(self._x.size, self._x[-1] - self._x[0])[1]
        return self._d2mat

    def differentiate(self, u: Array) -> Array:
        arr, squeeze = _as_2d(u)
        out = self._dmat @ arr
        return out[:, 0] if squeeze else out

    def field_and_derivative(self, points: Array, u: Array) -> tuple[Array, Array]:
        interp = local_interpolation_matrix(self._x, np.asarray(points, dtype=float), self.order + 1)
        du = self.differentiate(u)
        arr_u, squeeze = _as_2d(u)
        arr_du, _ = _as_2d(du)
        f = interp @ arr_u
        g = interp @ arr_du
        if squeeze:
            return f[:, 0], g[:, 0]
        return f, g


# ---------------------------------------------------------------------------
# C6F: standard C6 interior/closures plus the implicit
# tenth-order low-pass filter (JCP 181, 2002; AFRL-VA-WP-TR-1998-3060 gives the
# coefficient tables in closed form).
# ---------------------------------------------------------------------------

def visbal_gaitonde_f10_interior_coefficients(alpha_f: float) -> tuple[float, ...]:
    """Interior tenth-order tridiagonal filter coefficients a0..a5 (Table 2.14 family).

    Derived and cross-checked independently (see code/verify_strong_competitors.py):
    solve for a0..a5 in
        alpha_f (phihat_{i-1}+phihat_{i+1}) + phihat_i = sum_{n=0}^{5} a_n cos(n k) [Fourier symbol]
    subject to SF(0)=1 (consistency) and SF(pi)=0 (total removal of the
    grid-Nyquist mode), which are the two conditions the filter construction imposes
    before Taylor-matching the remaining four equations. At alpha_f=0 this
    reduces exactly to the well-known explicit tenth-order filter table
    (193/256, 105/256, -15/64, 45/512, -5/256, 1/512), which matches the
    published af=0 row and is the standard cross-check for this construction.
    """
    af = float(alpha_f)
    a0 = 193.0 / 256.0 + (63.0 / 128.0) * af
    a1 = 105.0 / 256.0 + (151.0 / 128.0) * af
    a2 = -15.0 / 64.0 + (15.0 / 32.0) * af
    a3 = 45.0 / 512.0 - (45.0 / 256.0) * af
    a4 = -5.0 / 256.0 + (5.0 / 128.0) * af
    a5 = 1.0 / 512.0 - (1.0 / 256.0) * af
    return a0, a1, a2, a3, a4, a5


# Boundary filter formulas at points 1-5 (order of accuracy 3,4,6,8,9 respectively;
# the standard "practice" choice of decreasing filter order toward
# the wall, transcribed from Tables 2.15/2.16/2.17/2.18/2.19 of the compact-filter report
# "High-order schemes for Navier-Stokes equations: algorithm and implementation
# into FDL3DI," AFRL-VA-WP-TR-1998-3060 (1998), freely available via DTIC
# (archive.org item DTIC_ADA364301). Each row gives (coefficients a..k) applied to
# phi_1..phi_11 (or as many as are nonzero), i.e. the filtered value at point p is
# phihat_p solved from  alpha_f*phihat_{p-1} + phihat_p + alpha_f*phihat_{p+1}
# = a*phi_1 + b*phi_2 + ... (with phihat_0, phihat_{-1} etc. absent at the wall).
# The favorable-to-the-competitor choice (per the task's instruction to pick the
# strongest defensible variant) uses the HIGHEST tabulated order at each of the
# five near-boundary points: FB1,10 / FB2,10 / FB3,10 / FB4,10 / FB5,10.
def visbal_gaitonde_f10_boundary_rows(alpha_f: float) -> tuple[tuple[float, ...], ...]:
    """Rows FB1,10 .. FB5,10, re-transcribed from 300-dpi crops of the source
    tables (compact-filter report, Tables 2.15-2.19;
    freely available via archive.org item DTIC_ADA364301). Each entry below
    was cross-read against a magnified crop of the corresponding table cell;
    see code/verify_strong_competitors.py for the row-sum-to-1 (consistency)
    and 6th/8th-order MMS checks used to catch any remaining transcription
    slip.
    """
    af = float(alpha_f)
    row1 = (  # FB1,10 (Table 2.19, point 1)
        (1023.0 + af) / 1024.0,
        (5.0 + 507.0 * af) / 512.0,
        45.0 * (-1.0 + af) / 1024.0,
        15.0 * (1.0 - af) / 128.0,
        105.0 * (-1.0 + af) / 512.0,
        63.0 * (1.0 - af) / 256.0,
        105.0 * (-1.0 + af) / 512.0,
        15.0 * (1.0 - af) / 128.0,
        45.0 * (-1.0 + af) / 1024.0,
        5.0 * (1.0 - af) / 512.0,
        (-1.0 + af) / 1024.0,
    )
    row2 = (  # FB2,10 (Table 2.18, point 2)
        (1.0 + 1022.0 * af) / 1024.0,
        (507.0 + 10.0 * af) / 512.0,
        (45.0 + 934.0 * af) / 1024.0,
        15.0 * (-1.0 + 2.0 * af) / 128.0,
        105.0 * (1.0 - 2.0 * af) / 512.0,
        63.0 * (-1.0 + 2.0 * af) / 256.0,
        105.0 * (1.0 - 2.0 * af) / 512.0,
        15.0 * (-1.0 + 2.0 * af) / 128.0,
        45.0 * (1.0 - 2.0 * af) / 1024.0,
        5.0 * (-1.0 + 2.0 * af) / 512.0,
        (1.0 - 2.0 * af) / 1024.0,
    )
    row3 = (  # FB3,10 (Table 2.17, point 3)
        (-1.0 + 2.0 * af) / 1024.0,
        (5.0 + 502.0 * af) / 512.0,
        (979.0 + 90.0 * af) / 1024.0,
        (15.0 + 98.0 * af) / 128.0,
        105.0 * (-1.0 + 2.0 * af) / 512.0,
        63.0 * (1.0 - 2.0 * af) / 256.0,
        105.0 * (-1.0 + 2.0 * af) / 512.0,
        15.0 * (1.0 - 2.0 * af) / 128.0,
        45.0 * (-1.0 + 2.0 * af) / 1024.0,
        5.0 * (1.0 - 2.0 * af) / 512.0,
        (-1.0 + 2.0 * af) / 1024.0,
    )
    row4 = (  # FB4,10 (Table 2.16, point 4)
        (1.0 - 2.0 * af) / 1024.0,
        5.0 * (-1.0 + 2.0 * af) / 512.0,
        (45.0 + 934.0 * af) / 1024.0,
        (113.0 + 30.0 * af) / 128.0,
        (105.0 + 302.0 * af) / 512.0,
        63.0 * (-1.0 + 2.0 * af) / 256.0,
        105.0 * (1.0 - 2.0 * af) / 512.0,
        15.0 * (-1.0 + 2.0 * af) / 128.0,
        45.0 * (1.0 - 2.0 * af) / 1024.0,
        5.0 * (-1.0 + 2.0 * af) / 512.0,
        (1.0 - 2.0 * af) / 1024.0,
    )
    row5 = (  # FB5,10 (Table 2.15, point 5)
        (-1.0 + 2.0 * af) / 1024.0,
        5.0 * (1.0 - 2.0 * af) / 512.0,
        45.0 * (-1.0 + 2.0 * af) / 1024.0,
        (15.0 + 98.0 * af) / 128.0,
        (407.0 + 210.0 * af) / 512.0,
        (63.0 + 130.0 * af) / 256.0,
        105.0 * (-1.0 + 2.0 * af) / 512.0,
        15.0 * (1.0 - 2.0 * af) / 128.0,
        45.0 * (-1.0 + 2.0 * af) / 1024.0,
        5.0 * (1.0 - 2.0 * af) / 512.0,
        (-1.0 + 2.0 * af) / 1024.0,
    )
    return row1, row2, row3, row4, row5


@lru_cache(maxsize=None)
def visbal_gaitonde_filter_matrix(npts: int, alpha_f: float = 0.49) -> Array:
    """Dense filter operator F such that phihat = F @ phi (tenth order, one-sided walls).

    Assembled once and cached; `CompactFilteredFD` applies this once per RK4
    step (see its docstring for exactly where in the step).
    """
    af = float(alpha_f)
    if not (0.0 < af <= 0.5):
        raise ValueError("alpha_f must lie in (0, 0.5]")
    a = visbal_gaitonde_f10_interior_coefficients(af)
    boundary_rows = visbal_gaitonde_f10_boundary_rows(af)

    lhs = np.eye(npts, dtype=float)
    rhs = np.zeros((npts, npts), dtype=float)

    m = len(boundary_rows)  # 5 near-boundary rows per wall
    for p in range(m):
        row = boundary_rows[p]
        lhs[p, p] = 1.0
        if p > 0:
            lhs[p, p - 1] = af
        if p + 1 < npts:
            lhs[p, p + 1] = af
        width = min(len(row), npts)
        rhs[p, :width] = np.asarray(row[:width], dtype=float)

    for j in range(m, npts - m):
        lhs[j, j - 1] = af
        lhs[j, j] = 1.0
        lhs[j, j + 1] = af
        rhs[j, j] = a[0]
        for n in range(1, 6):
            rhs[j, j - n] += 0.5 * a[n]
            rhs[j, j + n] += 0.5 * a[n]

    for p in range(m):
        row = boundary_rows[p]
        i = npts - 1 - p
        lhs[i, i] = 1.0
        if i - 1 >= 0:
            lhs[i, i - 1] = af
        if i + 1 < npts:
            lhs[i, i + 1] = af
        width = min(len(row), npts)
        rhs[i, npts - width:] = np.asarray(row[:width], dtype=float)[::-1]

    lu = lu_factor(lhs)
    return lu_solve(lu, rhs)


def apply_filter(filter_matrix: Array, u: Array) -> Array:
    arr, squeeze = _as_2d(u)
    out = filter_matrix @ arr
    return out[:, 0] if squeeze else out


def filter_transfer_function(alpha_f: float, k: Array) -> Array:
    """Interior (periodic-grid) filter symbol SF(k) for gate-(a)/(b) checks."""
    af = float(alpha_f)
    a = visbal_gaitonde_f10_interior_coefficients(af)
    kk = np.asarray(k, dtype=float)
    numer = a[0] + sum(a[n] * np.cos(n * kk) for n in range(1, 6))
    denom = 1.0 + 2.0 * af * np.cos(kk)
    return numer / denom


class CompactFilteredFD(CompactFD):
    """C6F: the existing C6 operator plus a per-RK4-step tenth-order implicit low-pass (F10) filter.

    The differentiation route (`matrix`, `differentiate`) is byte-identical to
    `CompactFD` (order-6 Pade interior + its existing one-sided closures); the
    filter is a SEPARATE operator applied to the solution state, not folded into
    the derivative matrix. `apply_filter_to_state` is called by the experiment
    drivers once per completed RK4 step (after the k1..k4 combination that
    advances the state to t+dt, i.e. filtering acts on y_{n+1}, not on the RK
    stage values k1..k4 themselves). This is the standard implicit-filter
    practice: filter the updated solution once per full step, not per stage.
    """

    def __init__(self, npts: int, order: int, length: float = 1.0, alpha_f: float = 0.49) -> None:
        if int(order) != 6:
            raise ValueError("C6F is only defined for order=6 in this codebase")
        super().__init__(npts, order, length)
        self.name = "C6F"
        self.alpha_f = float(alpha_f)
        self._filter: Array | None = None

    @property
    def filter_matrix(self) -> Array:
        if self._filter is None:
            self._filter = visbal_gaitonde_filter_matrix(self._x.size, self.alpha_f)
        return self._filter

    def apply_filter_to_state(self, u: Array) -> Array:
        return apply_filter(self.filter_matrix, u)


def build_operator_extended(kind: str, order: int, npts: int, length: float = 1.0) -> Operator1D:
    """Dispatcher for the full-strength competitor labels, kept separate from `build_operator`.

    `build_operator` (above) is untouched. This companion function recognizes
    the new scheme kinds so experiment scripts can opt in explicitly:
    'C6F' (filtered compact), 'SBP6-BO' (boundary-optimized SBP6), 'C6-D2'
    (compact-baseline second derivative riding on the ordinary C6 first
    derivative), and 'Cq-CB3'/'Cq-CB2' (the Pade interior closed by the published
    Section 4.1 compact wall rows at wall alpha 3 and 2).
    Any kind not in this list is forwarded to the untouched `build_operator`.
    """
    key = str(kind).upper()
    if key in ('C6-CB', 'C6-CB3'):
        return CompactBaselineFD(npts, 6, length, wall_alpha=3.0)
    if key == 'C6-CB2':
        return CompactBaselineFD(npts, 6, length, wall_alpha=2.0)
    if key in ('C4-CB', 'C4-CB3'):
        return CompactBaselineFD(npts, 4, length, wall_alpha=3.0)
    if key == 'C4-CB2':
        return CompactBaselineFD(npts, 4, length, wall_alpha=2.0)
    if key in ('C8-CB', 'C8-CB3'):
        return CompactBaselineFD(npts, 8, length, wall_alpha=3.0)
    if key == 'C8-CB2':
        return CompactBaselineFD(npts, 8, length, wall_alpha=2.0)
    if key == 'C6F':
        return CompactFilteredFD(npts, 6, length)
    if key == 'SBP6-BO':
        return SBPBoundaryOptimized(npts, 6, length)
    if key == 'C6-D2':
        return CompactD2FD(npts, 6, length)
    return build_operator(kind, order, npts, length)


# ---------------------------------------------------------------------------
# Contract K (comparator fairness at the wall):
#   * the classical compact-difference reference Section 4.1 compact boundary closures for the C_q family,
#   * simultaneous-approximation-term (SAT) enforcement of the closed acoustic
#     pair on the diagonal-norm SBP first derivative,
#   * a narrow-stencil compatible SBP second derivative of order six.
# Every coefficient below is derived from exactness conditions inside this
# module; none is transcribed from a table.
# ---------------------------------------------------------------------------


def compact_baseline_wall_row_coefficients(wall_alpha: float, ncol: int = 4) -> Array:
    """Right-side coefficients of the published one-sided compact wall row.

    The row is
        f'_1 + alpha f'_2 = (1/h) sum_{j=1}^{ncol} coeff_j f_j ,
    with nodes at s = 0, 1, ..., ncol-1 in units of h. The coefficients are
    obtained by requiring exactness on the monomials s^k, k = 0, ..., ncol-1,
    which is a square linear system in (a, b, c, d) for a given alpha. For
    alpha = 3 the solution is the classical fourth-order row
    (-17/6, 3/2, 3/2, -1/6); for alpha = 2 it is the third-order row
    (-5/2, 2, 1/2, 0). The fourth-order condition on s^4 is met only by
    alpha = 3, which the unit tests check directly.
    """
    alpha = float(wall_alpha)
    ncol = int(ncol)
    s = np.arange(ncol, dtype=float)
    lhs = np.vstack([s ** k for k in range(ncol)])
    rhs = np.empty(ncol, dtype=float)
    for k in range(ncol):
        d_at_0 = 1.0 if k == 1 else 0.0
        d_at_1 = float(k) if k >= 1 else 0.0
        rhs[k] = d_at_0 + alpha * d_at_1
    return np.linalg.solve(lhs, rhs)


def pade_symmetric_row_coefficients(alpha: float, half_width: int) -> Array:
    """Right-side coefficients of a symmetric tridiagonal Pade row.

    The row is
        alpha f'_{i-1} + f'_i + alpha f'_{i+1}
            = (1/h) sum_{k=-half_width}^{half_width} coeff_k f_{i+k},
    derived by exactness on s^m for m up to 2*half_width. For
    half_width = 1 the solution at alpha = 1/4 is the standard fourth-order
    Pade row with right side (-3/4, 0, 3/4); for half_width = 2 at
    alpha = 1/3 it is the standard sixth-order row.
    """
    half = int(half_width)
    s = np.arange(-half, half + 1, dtype=float)
    size = s.size
    lhs = np.vstack([s ** k for k in range(size)])
    rhs = np.empty(size, dtype=float)
    for k in range(size):
        def deriv(t: float) -> float:
            return float(k) * t ** (k - 1) if k >= 1 else 0.0
        rhs[k] = float(alpha) * deriv(-1.0) + deriv(0.0) + float(alpha) * deriv(1.0)
    return np.linalg.solve(lhs, rhs)


def _compact_rhs_half_width(order: int) -> int:
    _alpha, _a, b, c = compact_interior_coefficients(order)
    if abs(c) > 0.0:
        return 3
    if abs(b) > 0.0:
        return 2
    return 1


@lru_cache(maxsize=None)
def compact_fd_matrices_compact_baseline(npts: int, order: int = 6, length: float = 1.0,
                             wall_alpha: float = 3.0) -> tuple[Array, Array, Array]:
    """Compact first derivative with the published Section 4.1 boundary closures.

    `compact_fd_matrices` above is left untouched: it pairs the Pade interior
    with generic explicit one-sided rows and an identity left side on the first
    q/2 rows, a closure that was not published. This companion keeps the same
    Pade interior but closes the wall with the published compact rows:

      row 1      f'_1 + alpha f'_2 = (a f_1 + b f_2 + c f_3 + d f_4)/h,
                 alpha = `wall_alpha` (3 for the fourth-order row, 2 for the
                 third-order row), coefficients from
                 `compact_baseline_wall_row_coefficients`;
      row 2      the standard fourth-order Pade row, alpha = 1/4;
      row 3      the standard sixth-order Pade row, alpha = 1/3, used only
                 when the interior right side reaches three neighbors (q = 8).

    The right wall is the antisymmetric mirror. The returned pair (A, B)
    satisfies A f' = B f, so the eliminated operator is A^{-1} B and the left
    side is tridiagonal at every row, unlike the identity rows of the
    incumbent closure.
    """
    npts = int(npts)
    order = int(order)
    x = uniform_grid(npts, length)
    h = float(x[1] - x[0])
    alpha, a, b, c = compact_interior_coefficients(order)
    half = _compact_rhs_half_width(order)
    if npts < 4 * half + 6:
        raise ValueError(f"need at least {4 * half + 6} points for the compact-baseline closure at order {order}")
    amat = np.zeros((npts, npts), dtype=float)
    bmat = np.zeros((npts, npts), dtype=float)
    for j in range(half, npts - half):
        amat[j, j - 1] = alpha
        amat[j, j] = 1.0
        amat[j, j + 1] = alpha
        bmat[j, j - 1] = -a / (2.0 * h)
        bmat[j, j + 1] = a / (2.0 * h)
        if abs(b) > 0.0:
            bmat[j, j - 2] = -b / (4.0 * h)
            bmat[j, j + 2] = b / (4.0 * h)
        if abs(c) > 0.0:
            bmat[j, j - 3] = -c / (6.0 * h)
            bmat[j, j + 3] = c / (6.0 * h)
    wall = compact_baseline_wall_row_coefficients(wall_alpha)
    amat[0, 0] = 1.0
    amat[0, 1] = float(wall_alpha)
    bmat[0, :wall.size] = wall / h
    amat[npts - 1, npts - 1] = 1.0
    amat[npts - 1, npts - 2] = float(wall_alpha)
    bmat[npts - 1, npts - wall.size:] = -wall[::-1] / h
    ladder = ((1.0 / 4.0, 1), (1.0 / 3.0, 2))
    for j in range(1, half):
        row_alpha, row_half = ladder[j - 1]
        coeff = pade_symmetric_row_coefficients(row_alpha, row_half)
        amat[j, j - 1] = row_alpha
        amat[j, j] = 1.0
        amat[j, j + 1] = row_alpha
        bmat[j, j - row_half:j + row_half + 1] = coeff / h
        jj = npts - 1 - j
        amat[jj, jj - 1] = row_alpha
        amat[jj, jj] = 1.0
        amat[jj, jj + 1] = row_alpha
        bmat[jj, jj - row_half:jj + row_half + 1] = -coeff[::-1] / h
    return x.copy(), amat, bmat


class CompactBaselineFD(Operator1D):
    """Compact family closed by the published Section 4.1 boundary rows."""

    def __init__(self, npts: int, order: int, length: float = 1.0,
                 wall_alpha: float = 3.0) -> None:
        x, amat, bmat = compact_fd_matrices_compact_baseline(npts, order, length, float(wall_alpha))
        self.name = f"C{order}-CB{int(round(float(wall_alpha)))}"
        self.order = int(order)
        self.wall_alpha = float(wall_alpha)
        self._x = x
        self._amat = amat
        self._bmat = bmat
        self._lu = lu_factor(amat)
        self._dmat = lu_solve(self._lu, bmat)
        storage = int(np.count_nonzero(np.abs(amat) > 1.0e-14) + np.count_nonzero(np.abs(bmat) > 1.0e-14))
        rhs_flops = int(sum(2 * np.count_nonzero(np.abs(row) > 1.0e-14) - 1 for row in bmat))
        solve_flops = 4 * npts
        self._work = WorkModel(grid_apply_flops=rhs_flops, solve_flops=solve_flops,
                               total_flops=rhs_flops + solve_flops, storage_nnz=storage)

    @property
    def x(self) -> Array:
        return self._x

    @property
    def matrix(self) -> Array:
        return self._dmat

    @property
    def work_model(self) -> WorkModel:
        return self._work

    def differentiate(self, u: Array) -> Array:
        arr, squeeze = _as_2d(u)
        out = lu_solve(self._lu, self._bmat @ arr)
        return out[:, 0] if squeeze else out

    def field_and_derivative(self, points: Array, u: Array) -> tuple[Array, Array]:
        interp = local_interpolation_matrix(self._x, np.asarray(points, dtype=float), self.order + 1)
        du = self.differentiate(u)
        arr_u, squeeze = _as_2d(u)
        arr_du, _ = _as_2d(du)
        f = interp @ arr_u
        g = interp @ arr_du
        if squeeze:
            return f[:, 0], g[:, 0]
        return f, g


def sbp_sat_penalty_range() -> dict[str, float]:
    """Admissible SAT parameters for the rigid-wall acoustic pair.

    Energy method. With `H D + D^T H = diag(-1, 0, ..., 0, 1)` and the state
    `z = (p, u)` carrying every node of both fields,

        p_t = -D u + H^{-1}(theta_0 e_0 u_0 + theta_N e_N u_N),
        u_t = -D p + H^{-1}(tau_0 e_0 u_0 + tau_N e_N u_N),

    the discrete energy `E = p^T H p + u^T H u` obeys

        dE/dt = 2 (1 + theta_0) p_0 u_0 + 2 (theta_N - 1) p_N u_N
                + 2 tau_0 u_0^2 + 2 tau_N u_N^2 .

    The boundary pressure traces are unconstrained, so a rate that is
    nonpositive for every state forces `theta_0 = -1` and `theta_N = +1`;
    the remaining freedom is `tau_0, tau_N <= 0`. The endpoint `tau = 0` is
    energy neutral and leaves the boundary residual undamped. The smallest
    strictly damping penalty scaled by the incoming characteristic speed
    `|lambda| = 1` is `tau = -1`, at which the penalty direction is exactly
    the incoming characteristic eigenvector; `tau = -2` is the doubled value
    and remains admissible.
    """
    return {"theta_left": -1.0, "theta_right": 1.0,
            "tau_max": 0.0, "tau_characteristic": -1.0, "tau_doubled": -2.0}


@lru_cache(maxsize=None)
def sbp_sat_acoustic_generator(npts: int, order: int = 6, length: float = 1.0,
                               tau: float = -1.0) -> tuple[Array, Array, Array]:
    """SAT-enforced closed-box acoustic generator on the diagonal-norm SBP pair.

    Returns `(x, generator, metric)` where `generator` is the `2*npts` square
    matrix of `z_t = A z` for `z = (p, u)` with `u = 0` imposed weakly at both
    walls by simultaneous approximation terms, and `metric` is the block
    energy norm `diag(H, H)`. The penalty parameters follow
    `sbp_sat_penalty_range`: `theta_0 = -1`, `theta_N = +1`, and `tau <= 0`
    on the velocity equation at both walls.
    """
    npts = int(npts)
    x, dmat, hmat = sbp_fd_matrix(npts, order, length)
    hinv = np.diag(1.0 / np.diag(hmat))
    limits = sbp_sat_penalty_range()
    e0 = np.zeros((npts, npts), dtype=float)
    eN = np.zeros((npts, npts), dtype=float)
    e0[0, 0] = 1.0
    eN[-1, -1] = 1.0
    pen_p = hinv @ (limits["theta_left"] * e0 + limits["theta_right"] * eN)
    pen_u = float(tau) * hinv @ (e0 + eN)
    zero = np.zeros((npts, npts), dtype=float)
    gen = np.block([[zero, -dmat + pen_p], [-dmat, pen_u]])
    metric = np.block([[hmat, zero], [zero, hmat]])
    return x.copy(), gen, metric


NARROW_D2_INTERIOR_ORDER6 = (1.0 / 90.0, -3.0 / 20.0, 3.0 / 2.0, -49.0 / 18.0,
                             3.0 / 2.0, -3.0 / 20.0, 1.0 / 90.0)
_NARROW_D2_REFERENCE_NPTS = 41
_NARROW_D2_BLOCK = 6
_NARROW_D2_BOUNDARY_ROWS = 6
_NARROW_D2_BOUNDARY_DEGREE = 4
_NARROW_D2_TRACE_POINTS = 6


def _narrow_d2_interior_entry(offset: int) -> float:
    if abs(offset) > 3:
        return 0.0
    return -NARROW_D2_INTERIOR_ORDER6[offset + 3]


def _narrow_d2_norm_diagonal(npts: int, order: int) -> Array:
    _rows, _upper, weights = sbp_first_derivative_coefficients(order)
    w = np.asarray(weights, dtype=float)
    diag = np.ones(int(npts), dtype=float)
    diag[:w.size] = w
    diag[-w.size:] = w[::-1]
    return diag


def _narrow_d2_assemble_A(block: Array, npts: int) -> Array:
    """Symmetric A on `npts` nodes in units h = 1 from the free boundary block."""
    npts = int(npts)
    amat = np.zeros((npts, npts), dtype=float)
    for i in range(npts):
        for off in range(-3, 4):
            j = i + off
            if 0 <= j < npts:
                amat[i, j] = _narrow_d2_interior_entry(off)
    nb = _NARROW_D2_BLOCK
    for i in range(nb):
        for j in range(i, nb):
            v = float(block[i, j])
            amat[i, j] = v
            amat[j, i] = v
            amat[npts - 1 - i, npts - 1 - j] = v
            amat[npts - 1 - j, npts - 1 - i] = v
    return amat


def _narrow_d2_accuracy_system(order: int) -> tuple[Array, Array, list[tuple[int, int]]]:
    """Linear accuracy conditions for the free block, in units h = 1.

    The conditions are `(A v_k)_i = (B S v_k)_i - H_i v_k''(s_i)` for the
    boundary rows `i < 6` and polynomial degrees `k <= 4` (boundary order
    three for the second derivative). A shifted and scaled monomial basis is
    used so the Vandermonde stays well conditioned.
    """
    npts = _NARROW_D2_REFERENCE_NPTS
    nb = _NARROW_D2_BLOCK
    diag = _narrow_d2_norm_diagonal(npts, order)
    s = np.arange(npts, dtype=float)
    center = 0.5 * (nb + 3.0)
    scale = float(max(nb, 4))
    t = (s - center) / scale
    pairs = [(i, j) for i in range(nb) for j in range(i, nb)]
    index = {p: k for k, p in enumerate(pairs)}
    rows: list[Array] = []
    rhs: list[float] = []
    for i in range(_NARROW_D2_BOUNDARY_ROWS):
        for k in range(_NARROW_D2_BOUNDARY_DEGREE + 1):
            coef = np.zeros(len(pairs), dtype=float)
            fixed = 0.0
            values = t ** k
            for j in range(npts):
                if i < nb and j < nb:
                    coef[index[(min(i, j), max(i, j))]] += values[j]
                else:
                    fixed += _narrow_d2_interior_entry(j - i) * values[j]
            trace = (float(k) * t[0] ** (k - 1) / scale) if k >= 1 else 0.0
            curv = (float(k * (k - 1)) * t[i] ** (k - 2) / scale ** 2) if k >= 2 else 0.0
            target = -(trace if i == 0 else 0.0) - diag[i] * curv
            rows.append(coef)
            rhs.append(target - fixed)
    return np.asarray(rows), np.asarray(rhs), pairs


def _narrow_d2_trace_rows(npts: int) -> tuple[Array, Array]:
    """One-sided wall derivative rows in units h = 1 (six points, order five)."""
    s = np.arange(npts, dtype=float)
    left = fd_weights(0.0, s[:_NARROW_D2_TRACE_POINTS], 1)
    right = -left[::-1]
    return left, right


def _narrow_d2_operator_h1(block: Array, npts: int, order: int) -> tuple[Array, Array, Array]:
    amat = _narrow_d2_assemble_A(block, npts)
    diag = _narrow_d2_norm_diagonal(npts, order)
    left, right = _narrow_d2_trace_rows(npts)
    bs = np.zeros((npts, npts), dtype=float)
    bs[0, :left.size] = -left
    bs[-1, -right.size:] = right
    d2 = (1.0 / diag)[:, None] * (-amat + bs)
    return amat, np.diag(diag), d2


@lru_cache(maxsize=None)
def narrow_d2_free_block(order: int = 6) -> tuple[Array, dict[str, float]]:
    """Free boundary block of the narrow second derivative, and its selection record.

    The accuracy conditions leave a one-parameter family. The fielded member
    minimizes the spectral radius of the Dirichlet-enforced operator over the
    parameters for which A is positive semidefinite, and, among the parameters
    within one part in a thousand of that minimum, takes the one that
    maximizes the first nonzero eigenvalue of A, so the null space of A stays
    exactly the constants.
    """
    if int(order) != 6:
        raise ValueError("the narrow second derivative is built here at order six only")
    mat, rhs, pairs = _narrow_d2_accuracy_system(order)
    particular, _res, rank, _sv = np.linalg.lstsq(mat, rhs, rcond=None)
    residual = float(np.linalg.norm(mat @ particular - rhs) / max(1.0, float(np.linalg.norm(rhs))))
    _u, _s, vt = np.linalg.svd(mat)
    null = vt[rank:].T
    if null.shape[1] != 1:
        raise RuntimeError(f"expected a one-parameter family, found dimension {null.shape[1]}")
    direction = null[:, 0]
    # The SVD fixes the null vector only up to sign; pin it so the parameter
    # window below is the same on every LAPACK build.
    if direction[0] > 0.0:
        direction = -direction
    npts = _NARROW_D2_REFERENCE_NPTS
    nb = _NARROW_D2_BLOCK

    def block_of(param: float) -> Array:
        coef = particular + direction * float(param)
        out = np.zeros((nb, nb), dtype=float)
        for (i, j), v in zip(pairs, coef):
            out[i, j] = v
            out[j, i] = v
        return out

    samples = []
    for param in np.linspace(-20.0, -0.5, 391):
        amat, _h, d2 = _narrow_d2_operator_h1(block_of(param), npts, order)
        eig = np.sort(np.linalg.eigvalsh(0.5 * (amat + amat.T)))
        if eig[0] < -1.0e-10:
            continue
        radius = float(np.max(np.abs(np.linalg.eigvals(d2[1:-1, 1:-1]))))
        samples.append((float(param), float(eig[1]), radius))
    if not samples:
        raise RuntimeError("no positive semidefinite member of the narrow-D2 family was found")
    arr = np.asarray(samples)
    best_radius = float(np.min(arr[:, 2]))
    band = arr[arr[:, 2] <= best_radius * 1.001]
    pick = band[int(np.argmax(band[:, 1]))]
    record = {"residual_relative": residual,
              "family_dimension": 1.0,
              "parameter": float(pick[0]),
              "first_nonzero_eigenvalue_of_A": float(pick[1]),
              "dirichlet_spectral_radius_h1": float(pick[2]),
              "minimum_spectral_radius_h1": best_radius,
              "reference_npts": float(npts)}
    return block_of(float(pick[0])), record


@lru_cache(maxsize=None)
def sbp_narrow_second_derivative_matrix(npts: int, order: int = 6,
                                        length: float = 1.0) -> tuple[Array, Array, Array, Array]:
    """Narrow-stencil compatible SBP second derivative of order six.

    Returns `(x, d2, hmat, amat)` with `d2 = H^{-1}(-A + B S)`, `A = A^T`
    positive semidefinite with the constants as its only null direction, `B`
    the endpoint selector, and `S` the one-sided wall derivative rows. The
    interior stencil is the standard seven-point sixth-order second
    derivative, so the operator is narrow; the incumbent `SBP6-BO` second
    derivative in this module is instead the wide composition.

    This is a compatible narrow operator with the published order and
    structure of the narrow compatible SBP second-derivative construction, derived here from the
    accuracy conditions; it is not their published coefficient set.
    """
    npts = int(npts)
    order = int(order)
    if npts < 4 * _NARROW_D2_BLOCK + 1:
        raise ValueError(f"need at least {4 * _NARROW_D2_BLOCK + 1} points for the narrow D2")
    block, _record = narrow_d2_free_block(order)
    amat_h1, hmat_h1, d2_h1 = _narrow_d2_operator_h1(block, npts, order)
    x = uniform_grid(npts, length)
    h = float(x[1] - x[0])
    return x.copy(), d2_h1 / h ** 2, hmat_h1 * h, amat_h1 / h
