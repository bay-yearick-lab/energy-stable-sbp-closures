"""Inf-sup constant of the divergence-conforming BMF pair converges to pi."""
import numpy as np
import pytest

from bmf_hdiv_projection import BMFHDivProjection2D


@pytest.mark.parametrize("p", range(2, 9))
def test_infsup_bounded_below_and_converging(p):
    """beta_h >= pi at every resolution, the distance to pi decreases with h,
    and the pressure Schur complement has the constant as its only null vector."""
    errors = []
    for elements in (4, 8, 16):
        core = BMFHDivProjection2D(elements + p, p)
        beta = core.beta()
        assert beta >= np.pi * (1.0 - 1e-13)
        errors.append(abs(beta / np.pi - 1.0))
        kernel = core.kernel_certificate()
        assert kernel.nullity == 1
        assert kernel.constant_overlap > 1.0 - 1e-10
    floor = 1e-11
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse or fine < floor


@pytest.mark.parametrize("p, bound", [(2, 1.1e-6), (3, 1e-9), (4, 1e-11)])
def test_infsup_values_at_16_elements(p, bound):
    """Reported values: |beta_h/pi - 1| = 1.0e-6 (p=2), 9.7e-10 (p=3), round-off for p >= 4."""
    beta = BMFHDivProjection2D(16 + p, p).beta()
    assert abs(beta / np.pi - 1.0) < bound
