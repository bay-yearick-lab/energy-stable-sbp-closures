"""Measured convergence orders: splines keep formal order at the wall, classical SBP does not."""
import numpy as np
import pytest

import boundary_spline_analysis as bsa


def _slope(h, err):
    return float(np.polyfit(np.log(h), np.log(err), 1)[0])


@pytest.mark.parametrize("p, minimum", [(3, 3.5), (5, 5.5)])
def test_bmg_first_derivative_order(p, minimum):
    """Mass-metric error of the BMG derivative of a smooth non-periodic function."""
    f = lambda x: np.sin(2.5 * np.pi * x + 0.3)
    df = lambda x: 2.5 * np.pi * np.cos(2.5 * np.pi * x + 0.3)
    grids = (33, 49, 65)
    errs = []
    for n in grids:
        x, dmat, hmat = bsa.bmg_nodal_operator(n, p)
        e = dmat @ f(x) - df(x)
        errs.append(np.sqrt(e @ hmat @ e))
    h = 1.0 / (np.array(grids) - 1.0)
    assert _slope(h, np.array(errs)) > minimum


@pytest.mark.parametrize("label, minimum, maximum", [
    ("BMF6", 6.5, None), ("BMG6", 6.5, None), ("SBP6", None, 4.5),
])
def test_heat_cavity_order_q6(label, minimum, maximum):
    """Wall-bounded heat cavity at q = 6, fitted on N = 81, 97, 129.
    Stored result: BMF 7.1, BMG 7.1, SBP 3.7."""
    import s6_heat_cavity_convergence as hc

    runs = [hc._run(label, n, 0.0) for n in (81, 97, 129)]
    h = np.array([r["h"] for r in runs])
    err = np.array([r["rel_l2_final"] for r in runs])
    slope = _slope(h, err)
    if minimum is not None:
        assert slope > minimum
    if maximum is not None:
        assert slope < maximum


def test_heat_cavity_accuracy_gap():
    """On the same grid, the spline error is orders of magnitude below classical SBP."""
    import s6_heat_cavity_convergence as hc

    spline = hc._run("BMF6", 65, 0.0)["rel_l2_final"]
    sbp = hc._run("SBP6", 65, 0.0)["rel_l2_final"]
    assert spline < 1e-3 * sbp
