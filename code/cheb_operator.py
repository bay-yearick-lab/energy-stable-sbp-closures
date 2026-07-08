#!/usr/bin/env python3
"""Chebyshev-collocation wall-normal operator: an INDEPENDENT discretization
(no B-spline machinery, no shared code path with boundary_spline_analysis) used
as an external reference for the turbulent-Nusselt resolution study. Standard Chebyshev-Gauss-Lobatto differentiation matrices (Trefethen,
Spectral Methods in MATLAB, 2000, program cheb.m), mapped from the canonical
[-1,1] node set to the physical wall-normal interval [0, Lz] and re-sorted to
ascending z to match the calling convention of scalar3d.z_operators (z[0]=0
bottom wall, z[-1]=Lz top wall; D1 @ u gives du/dz; D2 = D1 @ D1, dense).

This is spectrally accurate for smooth fields, so at n_z=97 (interior points
95) it is converged far beyond any of the finite-order schemes under test and
serves as a reference that does not share a single line of implementation with
the spline operator it is judging.
"""
from __future__ import annotations
import numpy as np


def cheb(n: int):
    """Trefethen's cheb.m: n+1 Chebyshev-Gauss-Lobatto points on [-1,1] and the
    (n+1)x(n+1) differentiation matrix, x descending from 1 to -1."""
    if n == 0:
        return np.array([1.0]), np.array([[0.0]])
    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.ones(n + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * (-1.0) ** np.arange(n + 1)
    X = np.tile(x, (n + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
    D = D - np.diag(D.sum(axis=1))
    return x, D


class ChebOperator:
    """Dense Chebyshev-collocation operator on nz nodes mapped to [0, Lz].
    Matches the (x, matrix, second_derivative_matrix()) interface consumed by
    scalar3d.z_operators via build_operator-like duck typing."""

    def __init__(self, npts: int, length: float = 1.0):
        n = int(npts) - 1
        xi, Dxi = cheb(n)                      # xi descending in [-1,1]
        # map [-1,1] -> [0,length], z = length*(1-xi)/2 (so xi=1 -> z=0, xi=-1 -> z=length)
        z = length * (1.0 - xi) / 2.0
        dz_dxi = -length / 2.0
        D1_desc = Dxi / dz_dxi                 # chain rule, still in descending-z order
        order = np.argsort(z)                  # ascending z: 0 (bottom) ... length (top)
        self._x = z[order]
        self._D1 = D1_desc[np.ix_(order, order)]
        self._D2 = self._D1 @ self._D1
        self.name = f"CHEB{npts}"
        self.order = npts - 1

    @property
    def x(self):
        return self._x

    @property
    def matrix(self):
        return self._D1

    def second_derivative_matrix(self):
        return self._D2


def build_cheb_operator(npts: int, length: float = 1.0) -> ChebOperator:
    return ChebOperator(npts, length)


if __name__ == "__main__":
    # quick self-check: differentiate a smooth test function against exact derivatives
    Lz = 2.0
    for n in [9, 17, 33]:
        op = build_cheb_operator(n, Lz)
        z = op.x
        f = np.sin(np.pi * z / Lz) * (z * (Lz - z))
        df_exact = (np.pi / Lz) * np.cos(np.pi * z / Lz) * (z * (Lz - z)) + np.sin(np.pi * z / Lz) * (Lz - 2 * z)
        df = op.matrix @ f
        err = np.max(np.abs(df - df_exact)) / np.max(np.abs(df_exact))
        print(f"n={n:3d}  z[0]={z[0]:.6f}  z[-1]={z[-1]:.6f}  deriv rel err={err:.3e}")
