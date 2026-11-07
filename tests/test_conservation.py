"""Discrete mass conservation of the divergence-conforming pressure projection."""
import numpy as np
import pytest

from bmf_hdiv_projection import BMFHDivProjection2D


@pytest.mark.parametrize("p", (2, 4, 6))
def test_projection_divergence_free_to_roundoff(p):
    """A random velocity with zero normal trace is projected to discrete
    divergence zero, and the projection is idempotent."""
    core = BMFHDivProjection2D(12 + p, p)
    rng = np.random.default_rng(p)
    ux = rng.standard_normal(core.ux_shape)
    uz = rng.standard_normal(core.uz_shape)
    ux[[0, -1], :] = 0.0
    uz[:, [0, -1]] = 0.0
    first = core.project(ux, uz)
    scale = max(np.abs(ux).max(), np.abs(uz).max())
    assert np.abs(core.divergence(first.ux, first.uz)).max() < 1e-12 * scale
    assert core.normal_trace_max(first.ux, first.uz) < 1e-13
    second = core.project(first.ux, first.uz)
    assert np.abs(second.ux - first.ux).max() < 1e-12 * scale
    assert np.abs(second.uz - first.uz).max() < 1e-12 * scale
