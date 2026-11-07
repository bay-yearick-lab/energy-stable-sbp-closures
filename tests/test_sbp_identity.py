"""Summation-by-parts identities hold to round-off for every spline degree."""
import numpy as np
import pytest

import boundary_spline_analysis as bsa

DEGREES = range(2, 9)


def _boundary(npts: int) -> np.ndarray:
    b = np.zeros((npts, npts))
    b[0, 0], b[-1, -1] = -1.0, 1.0
    return b


@pytest.mark.parametrize("p", DEGREES)
def test_bmg_sbp_identity(p):
    """H D + D^T H = diag(-1, 0, ..., 0, 1) for the Galerkin derivative."""
    npts = 4 * p + 9
    _x, dmat, hmat = bsa.bmg_nodal_operator(npts, p)
    assert np.allclose(hmat, hmat.T, atol=1e-14)
    assert np.linalg.eigvalsh(hmat).min() > 0.0
    residual = hmat @ dmat + dmat.T @ hmat - _boundary(npts)
    assert np.abs(residual).max() < 1e-12


@pytest.mark.parametrize("p", DEGREES)
def test_bmf_staggered_sbp_identity(p):
    """H0 D01 + D10^T H1 = B for the staggered gradient-divergence pair."""
    npts = 4 * p + 9
    assert np.abs(bsa.bmf_staggered_sbp_residual(npts, p)).max() < 1e-13


@pytest.mark.parametrize("p", (3, 5, 7))
def test_bmg_energy_conservation(p):
    """Skew part of H D carries only the wall flux, so u^T H D u = (u_N^2 - u_0^2)/2."""
    npts = 4 * p + 9
    _x, dmat, hmat = bsa.bmg_nodal_operator(npts, p)
    u = np.random.default_rng(p).standard_normal(npts)
    assert abs(u @ hmat @ dmat @ u - 0.5 * (u[-1] ** 2 - u[0] ** 2)) < 1e-12 * (u @ u)


@pytest.mark.parametrize("p", (2, 4, 6))
def test_bmg_differentiates_polynomials_exactly(p):
    """The derivative is exact on polynomials up to the spline degree."""
    npts = 4 * p + 9
    x, dmat, _h = bsa.bmg_nodal_operator(npts, p)
    for k in range(1, p + 1):
        assert np.abs(dmat @ x**k - k * x ** (k - 1)).max() < 1e-9
