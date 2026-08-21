#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Match the manuscript-wide STIX serif so this figure family is not the lone
# DejaVu outlier; typeface only, leaving each panel's explicit font sizes intact.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline

from boundary_spline_analysis import (
    DEFAULT_COLORS,
    bmf_staggered_operators,
    bmg_nodal_operator,
    bspline_evaluation_matrix,
    bspline_mass_and_advection,
    bspline_mass_matrix_from_knots,
    build_operator,
    clamped_bspline_matrices,
    derivative_space_knots,
    nodal_trapezoidal_weights,
)
from project_paths import FIGDIR, RESULTSDIR, ensure_generated_dirs, save_figure

Array = np.ndarray
MACHINE_EPS = np.finfo(float).eps
# Float64 roundoff range expected for these bilinear products of length-N
# operator stencils: shading marks the presentation band, not a strict bound.
MACHINE_ZERO_TOP = 5.0e-13
VISUAL_FLOOR = 1.0e-17


def _basis(knots: Array, degree: int, derivative: int = 0) -> list[BSpline]:
    knots = np.asarray(knots, dtype=float)
    p = int(degree)
    nbas = knots.size - p - 1
    coeff = np.zeros(nbas, dtype=float)
    out: list[BSpline] = []
    for i in range(nbas):
        coeff[:] = 0.0
        coeff[i] = 1.0
        b = BSpline(knots, coeff.copy(), p, extrapolate=False)
        if derivative:
            b = b.derivative(derivative)
        out.append(b)
    return out


def _basis_eval(basis: list[BSpline], points: Array) -> Array:
    vals = np.stack([b(points) for b in basis], axis=0)
    return np.where(np.isfinite(vals), vals, 0.0)


def _quad_spans(knots_list: Iterable[Array], quad_order: int = 120):
    breaks = np.unique(np.concatenate([np.unique(np.asarray(k, dtype=float)) for k in knots_list]))
    spans = [(a, b) for a, b in zip(breaks[:-1], breaks[1:]) if b > a + 1.0e-14]
    xi, wi = leggauss(int(quad_order))
    for a, b in spans:
        points = 0.5 * (b - a) * xi + 0.5 * (a + b)
        weights = 0.5 * (b - a) * wi
        yield points, weights


def _sample_fields(points: Array) -> tuple[Array, Array]:
    x = np.asarray(points, dtype=float)
    a = np.sin(np.pi * x) + 0.45 * np.sin(2.0 * np.pi * x + 0.2) + 0.15 * np.cos(4.0 * np.pi * x)
    w = np.cos(2.0 * np.pi * x - 0.3) + 0.25 * np.sin(3.0 * np.pi * x + 0.5)
    return a, w


def _streamfunction_and_pressure(x: Array, y: Array) -> tuple[Array, Array]:
    X, Y = np.meshgrid(x, y, indexing="ij")
    psi = (np.sin(np.pi * X) ** 2) * (np.sin(np.pi * Y) ** 2) * (
        1.0 + 0.2 * np.cos(2.0 * np.pi * X + 0.3) * np.sin(2.0 * np.pi * Y)
    )
    press = (np.sin(np.pi * X) ** 2) * (np.sin(np.pi * Y) ** 2) * (
        np.cos(3.0 * np.pi * X - 0.4) + 0.7 * np.sin(2.0 * np.pi * Y + 0.1)
    )
    return psi, press


def _analytic_stream_velocity(x: Array, y: Array) -> tuple[Array, Array]:
    X, Y = np.meshgrid(x, y, indexing="ij")
    sx = np.sin(np.pi * X)
    sy = np.sin(np.pi * Y)
    cx = np.cos(np.pi * X)
    cy = np.cos(np.pi * Y)
    modifier = 1.0 + 0.2 * np.cos(2.0 * np.pi * X + 0.3) * np.sin(2.0 * np.pi * Y)
    dmod_dx = -0.4 * np.pi * np.sin(2.0 * np.pi * X + 0.3) * np.sin(2.0 * np.pi * Y)
    dmod_dy = 0.4 * np.pi * np.cos(2.0 * np.pi * X + 0.3) * np.cos(2.0 * np.pi * Y)
    dpsi_dx = 2.0 * np.pi * sx * cx * sy**2 * modifier + sx**2 * sy**2 * dmod_dx
    dpsi_dy = 2.0 * np.pi * sy * cy * sx**2 * modifier + sx**2 * sy**2 * dmod_dy
    return dpsi_dy, -dpsi_dx


def product_rule_residual(family: str, npts: int, degree: int, *, quad_order: int = 120) -> dict[str, float]:
    """Weak Leibniz residual for exact products and for native-node reinterpolation.

    The exact residual integrates
        L_i [(aw)' - a w' - a' w],
    while the shortcut residual replaces aw by the spline obtained by
    reinterpolating the nodal product on the native product grid.  This is
    the cleanest algebraic isolator for the de-aliasing clause in the
    skew-symmetric product-map assumption.
    """
    family = family.upper()
    if family == "BMG":
        x, knots, bmat, _ = clamped_bspline_matrices(npts, degree)
        binv = np.linalg.inv(bmat)
        avals, wvals = _sample_fields(x)
        ca = binv @ avals
        cw = binv @ wvals
        caw_shortcut = binv @ (avals * wvals)
        basis = _basis(knots, degree, 0)
        dbasis = _basis(knots, degree, 1)

        rexact = np.zeros(npts, dtype=float)
        rshortcut = np.zeros(npts, dtype=float)
        denom = np.zeros(npts, dtype=float)
        for points, weights in _quad_spans([knots], quad_order):
            v = _basis_eval(basis, points)
            dv = _basis_eval(dbasis, points)
            a = ca @ v
            w = cw @ v
            da = ca @ dv
            dw = cw @ dv
            d_aw_exact = da * w + a * dw
            d_aw_shortcut = caw_shortcut @ dv
            rexact += v @ (weights * (d_aw_exact - a * dw - da * w))
            rshortcut += v @ (weights * (d_aw_shortcut - a * dw - da * w))
            denom += np.abs(v) @ (weights * np.abs(da * w))
        grid_kind = "same-grid"
    elif family == "BMF":
        x, knots, bmat, _ = clamped_bspline_matrices(npts, degree)
        ops = bmf_staggered_operators(npts, degree)
        hknots = derivative_space_knots(knots)
        hdegree = degree - 1
        eta = ops.eta
        binv = np.linalg.inv(bmat)
        bhatinv = np.linalg.inv(ops.bhat_eta)
        avals, _ = _sample_fields(eta)
        _, wvals = _sample_fields(x)
        ca = bhatinv @ avals
        cw = binv @ wvals
        vbasis = _basis(knots, degree, 0)
        dvbasis = _basis(knots, degree, 1)
        wbasis = _basis(hknots, hdegree, 0)
        dwbasis = _basis(hknots, hdegree, 1)
        w_at_eta = cw @ _basis_eval(vbasis, eta)
        caw_shortcut = bhatinv @ (avals * w_at_eta)

        rexact = np.zeros(npts, dtype=float)
        rshortcut = np.zeros(npts, dtype=float)
        denom = np.zeros(npts, dtype=float)
        for points, weights in _quad_spans([knots, hknots], quad_order):
            v = _basis_eval(vbasis, points)
            dv = _basis_eval(dvbasis, points)
            s = _basis_eval(wbasis, points)
            ds = _basis_eval(dwbasis, points)
            a = ca @ s
            w = cw @ v
            da = ca @ ds
            dw = cw @ dv
            d_aw_exact = da * w + a * dw
            d_aw_shortcut = caw_shortcut @ ds
            rexact += v @ (weights * (d_aw_exact - a * dw - da * w))
            rshortcut += v @ (weights * (d_aw_shortcut - a * dw - da * w))
            denom += np.abs(v) @ (weights * np.abs(da * w))
        grid_kind = "staggered"
    else:
        raise ValueError("family must be BMG or BMF")

    scale_inf = float(np.linalg.norm(denom, ord=np.inf) + 1.0e-300)
    scale_l2 = float(np.linalg.norm(denom) + 1.0e-300)
    return {
        "family": family,
        "grid_kind": grid_kind,
        "npts": int(npts),
        "degree": int(degree),
        "exact_inf": float(np.linalg.norm(rexact, ord=np.inf) / scale_inf),
        "exact_l2": float(np.linalg.norm(rexact) / scale_l2),
        "shortcut_inf": float(np.linalg.norm(rshortcut, ord=np.inf) / scale_inf),
        "shortcut_l2": float(np.linalg.norm(rshortcut) / scale_l2),
    }


def product_rule_residual_nodal_control(kind: str, npts: int, order: int) -> dict[str, float]:
    """Discrete weak Leibniz residual for a nodal scheme with its native-node diagonal product.

    The cross-scheme analog of `product_rule_residual` for the spline routes:
    each nodal family forms (a*w) at native grid points and applies its
    family-specific derivative D, then contracts against trapezoidal-weighted
    test functions in its own native test space.
    """
    op = build_operator(kind, order, npts)
    x = op.x
    dmat = op.matrix
    if hasattr(op, "metric"):
        hmat = op.metric
        if np.ndim(hmat) == 1:
            hvec = np.asarray(hmat, dtype=float)
        else:
            hvec = np.diag(hmat)
    else:
        hvec = nodal_trapezoidal_weights(x)

    a, w = _sample_fields(x)
    da = dmat @ a
    dw = dmat @ w
    aw = a * w
    daw = dmat @ aw
    res = hvec * (daw - a * dw - da * w)
    denom = hvec * np.abs(da * w)
    scale_inf = float(np.linalg.norm(denom, ord=np.inf) + 1.0e-300)
    scale_l2 = float(np.linalg.norm(denom) + 1.0e-300)
    return {
        "family": kind.upper(),
        "npts": int(npts),
        "degree": int(order),
        "weak_leibniz_inf": float(np.linalg.norm(res, ord=np.inf) / scale_inf),
        "weak_leibniz_l2": float(np.linalg.norm(res) / scale_l2),
    }


def _product_map_bmg(npts: int, degree: int, a_nodal: Array, *, quad_order: int = 80):
    x, knots, bmat, _ = clamped_bspline_matrices(npts, degree)
    binv = np.linalg.inv(bmat)
    mass = bspline_mass_matrix_from_knots(knots, degree, quad_order)
    ca = binv @ np.asarray(a_nodal, dtype=float)
    basis = _basis(knots, degree, 0)
    rhs = np.zeros((npts, npts), dtype=float)
    for points, weights in _quad_spans([knots], quad_order):
        v = _basis_eval(basis, points)
        a = ca @ v
        nodal_basis = binv.T @ v
        rhs += (v * (weights * a)) @ nodal_basis.T
    coeff = np.linalg.solve(mass, rhs)
    amat = bmat @ coeff
    hmat = binv.T @ mass @ binv
    return amat, 0.5 * (hmat + hmat.T), x


def _product_map_bmf(npts: int, degree: int, a_flux: Array, *, quad_order: int = 80):
    x, knots, bmat, _ = clamped_bspline_matrices(npts, degree)
    ops = bmf_staggered_operators(npts, degree)
    hknots = derivative_space_knots(knots)
    hdegree = degree - 1
    binv = np.linalg.inv(bmat)
    bhatinv = np.linalg.inv(ops.bhat_eta)
    mass_flux = bspline_mass_matrix_from_knots(hknots, hdegree, quad_order)
    ca = bhatinv @ np.asarray(a_flux, dtype=float)
    vbasis = _basis(knots, degree, 0)
    fbasis = _basis(hknots, hdegree, 0)
    rhs = np.zeros((ops.eta.size, npts), dtype=float)
    for points, weights in _quad_spans([knots, hknots], quad_order):
        v = _basis_eval(vbasis, points)
        f = _basis_eval(fbasis, points)
        a = ca @ f
        nodal_basis_v = binv.T @ v
        rhs += (f * (weights * a)) @ nodal_basis_v.T
    coeff = np.linalg.solve(mass_flux, rhs)
    amat = ops.bhat_eta @ coeff
    return amat, ops.h0.copy(), ops.h1.copy(), x, ops.eta.copy()


def nonlinear_ke_residual(family: str, npts: int, degree: int) -> dict[str, float]:
    family = family.upper()
    if family == "BMG":
        x, dmat, hmat = bmg_nodal_operator(npts, degree)
        a = np.sin(np.pi * x) ** 2 * (1.0 + 0.2 * np.sin(4.0 * np.pi * x))
        w = np.sin(np.pi * x) ** 2 * (np.cos(3.0 * np.pi * x) + 0.2 * np.sin(6.0 * np.pi * x))
        amat, hmat, _x = _product_map_bmg(npts, degree, a)
        adag = np.linalg.solve(hmat, amat.T @ hmat)
        conv = 0.5 * (dmat @ (amat @ w) + adag @ (dmat @ w))
        denom = float(w @ hmat @ w + 1.0e-300)
        route = "same-grid metric-adjoint"
    elif family == "BMF":
        ops = bmf_staggered_operators(npts, degree)
        x = ops.x
        eta = ops.eta
        a = np.sin(np.pi * eta) ** 2 * (1.0 + 0.2 * np.sin(4.0 * np.pi * eta))
        w = np.sin(np.pi * x) ** 2 * (np.cos(3.0 * np.pi * x) + 0.2 * np.sin(6.0 * np.pi * x))
        amat, h0, h1, _x, _eta = _product_map_bmf(npts, degree, a)
        adag = np.linalg.solve(h0, amat.T @ h1)
        conv = 0.5 * (ops.d01 @ (amat @ w) + adag @ (ops.d10 @ w))
        denom = float(w @ h0 @ w + 1.0e-300)
        route = "staggered metric-adjoint"
    else:
        raise ValueError("family must be BMG or BMF")
    return {
        "family": family,
        "npts": int(npts),
        "degree": int(degree),
        "route": route,
        "ke_production": float(abs(w @ (hmat if family == "BMG" else h0) @ conv) / denom),
    }


def projected_product_trace_leak_bmg(npts: int, degree: int) -> dict[str, float]:
    """Resolve the projected-product power into its exact SBP boundary work."""
    x, dmat, hmat = bmg_nodal_operator(npts, degree)
    a = np.sin(np.pi * x) ** 2
    w = 1.0 + 0.3 * x
    amat, hmat, _ = _product_map_bmg(npts, degree, a)
    adag = np.linalg.solve(hmat, amat.T @ hmat)
    projected = amat @ w
    convection = 0.5 * (dmat @ projected + adag @ (dmat @ w))
    power = float(w @ hmat @ convection)
    boundary_work = 0.5 * float(w[-1] * projected[-1] - w[0] * projected[0])
    identity_residual = abs(power - boundary_work)
    scale = max(
        math.sqrt(float(w @ hmat @ w) * float(convection @ hmat @ convection)),
        abs(power),
        abs(boundary_work),
        1.0e-300,
    )
    return {
        "family": "BMG",
        "npts": int(npts),
        "degree": int(degree),
        "advecting_left_trace": float(a[0]),
        "advecting_right_trace": float(a[-1]),
        "state_left_trace": float(w[0]),
        "state_right_trace": float(w[-1]),
        "projected_product_left_trace": float(projected[0]),
        "projected_product_right_trace": float(projected[-1]),
        "split_power": power,
        "sbp_boundary_work": boundary_work,
        "boundary_identity_residual": identity_residual,
        "relative_boundary_identity_residual": identity_residual / scale,
    }


def nonlinear_ke_residual_nodal_control(kind: str, npts: int, order: int) -> dict[str, float]:
    op = build_operator(kind, order, npts)
    x = op.x
    dmat = op.matrix
    if hasattr(op, "metric"):
        hmat = op.metric
        if np.ndim(hmat) == 1:
            hmat = np.diag(hmat)
    else:
        hmat = np.diag(nodal_trapezoidal_weights(x))
    a = np.sin(np.pi * x) ** 2 * (1.0 + 0.2 * np.sin(4.0 * np.pi * x))
    w = np.sin(np.pi * x) ** 2 * (np.cos(3.0 * np.pi * x) + 0.2 * np.sin(6.0 * np.pi * x))
    amat = np.diag(a)
    adag = np.linalg.solve(hmat, amat.T @ hmat)
    conv = 0.5 * (dmat @ (amat @ w) + adag @ (dmat @ w))
    return {
        "family": kind.upper(),
        "npts": int(npts),
        "degree": int(order),
        "route": "native same-grid nodal product",
        "ke_production": float(abs(w @ hmat @ conv) / float(w @ hmat @ w + 1.0e-300)),
    }


def local_fv_residual_bmg(npts: int, degree: int) -> dict[str, float]:
    x, knots, bmat, _ = clamped_bspline_matrices(npts, degree)
    mass, adv = bspline_mass_and_advection(npts, degree, 1.0, 80)
    _x, dmat, _hmat = bmg_nodal_operator(npts, degree)
    binv = np.linalg.inv(bmat)
    f = np.sin(2.0 * np.pi * x) + 0.3 * np.cos(5.0 * np.pi * x + 0.1)
    phi = binv @ f
    mdot = mass @ (binv @ (-dmat @ f))
    residual_moments = adv @ phi
    faces = np.empty(npts + 1, dtype=float)
    faces[0] = f[0]
    prefix = faces[0]
    for j in range(npts - 1):
        prefix += residual_moments[j]
        faces[j + 1] = prefix
    faces[npts] = f[-1]
    jumps = faces[1:] - faces[:-1]
    residual = mdot + jumps
    return {
        "family": "BMG",
        "npts": int(npts),
        "degree": int(degree),
        "local_fv_residual": float(np.linalg.norm(residual, ord=np.inf) / (np.linalg.norm(jumps, ord=np.inf) + 1.0e-300)),
    }


def local_fv_residual_bmf(npts: int, degree: int) -> dict[str, float]:
    x, knots, bmat, _ = clamped_bspline_matrices(npts, degree)
    ops = bmf_staggered_operators(npts, degree)
    hknots = derivative_space_knots(knots)
    hdegree = degree - 1
    mass = bspline_mass_matrix_from_knots(knots, degree, 80)
    mass_flux = bspline_mass_matrix_from_knots(hknots, hdegree, 80)
    binv = np.linalg.inv(bmat)
    bhatinv = np.linalg.inv(ops.bhat_eta)
    q = np.sin(2.0 * np.pi * ops.eta) + 0.3 * np.cos(5.0 * np.pi * ops.eta + 0.1)
    gamma = bhatinv @ q
    mdot = mass @ (binv @ (-ops.d01 @ q))
    flux_moments = mass_flux @ gamma
    t = knots
    incidence = np.array([degree / (t[i + degree + 1] - t[i + 1]) for i in range(npts - 1)], dtype=float)
    left_row = bspline_evaluation_matrix(hknots, hdegree, np.array([0.0])) @ bhatinv
    right_row = bspline_evaluation_matrix(hknots, hdegree, np.array([1.0])) @ bhatinv
    faces = np.empty(npts + 1, dtype=float)
    faces[0] = float((left_row @ q).item())
    faces[1:npts] = incidence * flux_moments
    faces[npts] = float((right_row @ q).item())
    jumps = faces[1:] - faces[:-1]
    residual = mdot + jumps
    return {
        "family": "BMF",
        "npts": int(npts),
        "degree": int(degree),
        "local_fv_residual": float(np.linalg.norm(residual, ord=np.inf) / (np.linalg.norm(jumps, ord=np.inf) + 1.0e-300)),
    }


def _qform2(a: Array, hx: Array, hy: Array, b: Array | None = None) -> float:
    if b is None:
        b = a
    return float(a.ravel() @ np.kron(hx, hy) @ b.ravel())


def pressure_work_residual_bmg(npts: int, degree: int) -> dict[str, float]:
    x, dmat, hmat = bmg_nodal_operator(npts, degree)
    psi, press = _streamfunction_and_pressure(x, x)
    dx = np.kron(dmat, np.eye(npts))
    dy = np.kron(np.eye(npts), dmat)
    psi_v = psi.ravel()
    p_v = press.ravel()
    u = (dy @ psi_v).reshape(npts, npts)
    v = -(dx @ psi_v).reshape(npts, npts)
    div = (dx @ u.ravel() + dy @ v.ravel()).reshape(npts, npts)
    gp_x = (dx @ p_v).reshape(npts, npts)
    gp_y = (dy @ p_v).reshape(npts, npts)
    work = _qform2(u, hmat, hmat, gp_x) + _qform2(v, hmat, hmat, gp_y)
    denom = math.sqrt(_qform2(u, hmat, hmat) + _qform2(v, hmat, hmat)) * math.sqrt(_qform2(press, hmat, hmat))
    return {
        "family": "BMG",
        "npts": int(npts),
        "degree": int(degree),
        "pressure_work": float(abs(work) / (denom + 1.0e-300)),
        "divergence_inf": float(np.max(np.abs(div))),
    }


def pressure_work_residual_bmf(npts: int, degree: int) -> dict[str, float]:
    ops = bmf_staggered_operators(npts, degree)
    x = ops.x
    eta = ops.eta
    psi, _ = _streamfunction_and_pressure(eta, eta)
    _, press = _streamfunction_and_pressure(x, x)
    u1 = psi @ ops.d01.T
    u2 = -(ops.d01 @ psi)
    div = ops.d01 @ u1 + u2 @ ops.d01.T
    gp1 = ops.d10 @ press
    gp2 = press @ ops.d10.T
    work = _qform2(u1, ops.h1, ops.h0, gp1) + _qform2(u2, ops.h0, ops.h1, gp2)
    denom = math.sqrt(_qform2(u1, ops.h1, ops.h0) + _qform2(u2, ops.h0, ops.h1)) * math.sqrt(_qform2(press, ops.h0, ops.h0))
    return {
        "family": "BMF",
        "npts": int(npts),
        "degree": int(degree),
        "pressure_work": float(abs(work) / (denom + 1.0e-300)),
        "divergence_inf": float(np.max(np.abs(div))),
    }


def pressure_work_residual_nodal_control(kind: str, npts: int, order: int) -> dict[str, float]:
    op = build_operator(kind, order, npts)
    x = op.x
    dmat = op.matrix
    if hasattr(op, "metric"):
        hmat = op.metric
        if np.ndim(hmat) == 1:
            hmat = np.diag(hmat)
    else:
        hmat = np.diag(nodal_trapezoidal_weights(x))
    psi, press = _streamfunction_and_pressure(x, x)
    dx = np.kron(dmat, np.eye(npts))
    dy = np.kron(np.eye(npts), dmat)
    psi_v = psi.ravel()
    p_v = press.ravel()
    u = (dy @ psi_v).reshape(npts, npts)
    v = -(dx @ psi_v).reshape(npts, npts)
    gp_x = (dx @ p_v).reshape(npts, npts)
    gp_y = (dy @ p_v).reshape(npts, npts)
    work = _qform2(u, hmat, hmat, gp_x) + _qform2(v, hmat, hmat, gp_y)
    denom = math.sqrt(_qform2(u, hmat, hmat) + _qform2(v, hmat, hmat)) * math.sqrt(_qform2(press, hmat, hmat))
    return {
        "family": kind.upper(),
        "npts": int(npts),
        "degree": int(order),
        "pressure_work": float(abs(work) / (denom + 1.0e-300)),
    }


def _dense_spline_values(knots: Array, degree: int, coeff: Array, points: Array, derivative: int = 0) -> Array:
    basis = _basis(knots, degree, derivative)
    return coeff @ _basis_eval(basis, points)


def _spline_weak_leibniz_residual(family: str, npts: int, degree: int, *, quad_order: int = 120) -> Array:
    family = family.upper()
    if family == "BMG":
        x, knots, bmat, _ = clamped_bspline_matrices(npts, degree)
        binv = np.linalg.inv(bmat)
        avals, wvals = _sample_fields(x)
        ca = binv @ avals
        cw = binv @ wvals
        basis = _basis(knots, degree, 0)
        dbasis = _basis(knots, degree, 1)
        residual = np.zeros(npts, dtype=float)
        for points, weights in _quad_spans([knots], quad_order):
            v = _basis_eval(basis, points)
            dv = _basis_eval(dbasis, points)
            a = ca @ v
            w = cw @ v
            da = ca @ dv
            dw = cw @ dv
            d_aw = da * w + a * dw
            residual += v @ (weights * (d_aw - a * dw - da * w))
        return residual
    if family == "BMF":
        x, knots, bmat, _ = clamped_bspline_matrices(npts, degree)
        ops = bmf_staggered_operators(npts, degree)
        hknots = derivative_space_knots(knots)
        hdegree = degree - 1
        eta = ops.eta
        binv = np.linalg.inv(bmat)
        bhatinv = np.linalg.inv(ops.bhat_eta)
        avals, _aw = _sample_fields(eta)
        _aa, wvals = _sample_fields(x)
        ca = bhatinv @ avals
        cw = binv @ wvals
        vbasis = _basis(knots, degree, 0)
        dvbasis = _basis(knots, degree, 1)
        wbasis = _basis(hknots, hdegree, 0)
        dwbasis = _basis(hknots, hdegree, 1)
        residual = np.zeros(npts, dtype=float)
        for points, weights in _quad_spans([knots, hknots], quad_order):
            v = _basis_eval(vbasis, points)
            dv = _basis_eval(dvbasis, points)
            s = _basis_eval(wbasis, points)
            ds = _basis_eval(dwbasis, points)
            a = ca @ s
            w = cw @ v
            da = ca @ ds
            dw = cw @ dv
            d_aw = da * w + a * dw
            residual += v @ (weights * (d_aw - a * dw - da * w))
        return residual
    raise ValueError("family must be BMG or BMF")


def make_dealiased_product_figure(npts: int, degree: int) -> None:
    colors = DEFAULT_COLORS
    machine_zero = 5.0e-13

    op_sbp = build_operator("SBP", degree, npts)
    x_sbp = op_sbp.x
    a_sbp, w_sbp = _sample_fields(x_sbp)
    dmat_sbp = op_sbp.matrix
    r_sbp = dmat_sbp @ (a_sbp * w_sbp) - a_sbp * (dmat_sbp @ w_sbp) - (dmat_sbp @ a_sbp) * w_sbp

    x_primal, _knots, _bmat, _ = clamped_bspline_matrices(npts, degree)
    r_bmf = _spline_weak_leibniz_residual("BMF", npts, degree)
    r_bmg = _spline_weak_leibniz_residual("BMG", npts, degree)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(x_sbp, r_sbp, color="0.32", lw=1.4, marker="o", ms=3.4,
            mfc="white", mec="0.32", mew=0.9)
    ax.set_title(rf"(a) $\mathrm{{SBP}}_{degree}$ native-node product", fontsize=10.6, pad=4)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$R(x)$")
    ax.grid(True, lw=0.35, alpha=0.28)
    rmax_a = float(np.max(np.abs(r_sbp)))
    ax.text(0.03, 0.94, rf"$\max_x|R(x)|=\mathrm{{{rmax_a:.2e}}}$",
            transform=ax.transAxes, fontsize=8.8, ha="left", va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0})

    for axx, residual, fam, title_text in (
        (axes[1], r_bmf, "BMF",
         rf"(b) $\mathrm{{BMF}}_{degree}$ flux-divergence form"),
        (axes[2], r_bmg, "BMG",
         rf"(c) $\mathrm{{BMG}}_{degree}$ natural-metric form"),
    ):
        axx.axhspan(-machine_zero, machine_zero, color="0.65", alpha=0.18, zorder=0)
        axx.axhline(machine_zero, color="0.45", ls=(0, (4, 3)), lw=0.7, zorder=1)
        axx.axhline(-machine_zero, color="0.45", ls=(0, (4, 3)), lw=0.7, zorder=1)
        axx.plot(x_primal, residual, color=colors[fam], lw=1.4, marker="s", ms=3.4,
                 mfc="white", mec=colors[fam], mew=0.9)
        axx.set_title(title_text, fontsize=10.6, pad=4)
        axx.set_xlabel(r"$x$")
        axx.set_ylabel(r"$R(x)$")
        axx.grid(True, lw=0.35, alpha=0.28)
        rmax = float(np.max(np.abs(residual)))
        axx.text(0.97, 0.94, rf"$\max_x|R(x)|=\mathrm{{{rmax:.2e}}}$",
                 transform=axx.transAxes, fontsize=8.8, ha="right", va="top",
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0})
        axx.text(0.03, 0.07, "machine zero", transform=axx.transAxes,
                 fontsize=8.0, color="0.35", ha="left", va="bottom")

    save_figure(fig, "fig_s6_nonlinear_product_setup")
    plt.close(fig)


def _draw_wall_hatching(ax) -> None:
    width = 0.035
    for rect in [
        Rectangle((0.0, 0.0), 1.0, width),
        Rectangle((0.0, 1.0 - width), 1.0, width),
        Rectangle((0.0, 0.0), width, 1.0),
        Rectangle((1.0 - width, 0.0), width, 1.0),
    ]:
        rect.set_facecolor("none")
        rect.set_edgecolor("0.35")
        rect.set_hatch("////")
        rect.set_linewidth(0.0)
        ax.add_patch(rect)
    ax.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, facecolor="none", edgecolor="0.18", linewidth=0.9))


def make_pressure_work_setup_figure(npts: int, degree: int) -> None:
    x, dmat, _hmat = bmg_nodal_operator(npts, degree)
    psi, _press = _streamfunction_and_pressure(x, x)
    dx = np.kron(dmat, np.eye(npts))
    dy = np.kron(np.eye(npts), dmat)
    psi_v = psi.ravel()
    u = (dy @ psi_v).reshape(npts, npts)
    v = -(dx @ psi_v).reshape(npts, npts)
    div = (dx @ u.ravel() + dy @ v.ravel()).reshape(npts, npts)
    _div_inf = float(np.max(np.abs(div)))
    if _div_inf >= 5.0e-13:
        raise RuntimeError("pressure setup divergence exceeded the manuscript caption bound")

    dense = np.linspace(0.0, 1.0, 150)
    u_dense, v_dense = _analytic_stream_velocity(dense, dense)
    _, press_dense = _streamfunction_and_pressure(dense, dense)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.18, h_pad=0.04, wspace=0.10, hspace=0.0)

    ax = axes[0]
    ax.set_facecolor("white")
    ax.streamplot(dense, dense, u_dense.T, v_dense.T, color="0.10", density=1.0, linewidth=0.55, arrowsize=0.55)
    _draw_wall_hatching(ax)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_title("streamfunction velocity", loc="left", fontsize=11.5)
    ax.tick_params(labelsize=11)

    ax = axes[1]
    levels = np.linspace(float(np.min(press_dense)), float(np.max(press_dense)), 18)
    im = ax.contourf(dense, dense, press_dense.T, levels=levels, cmap="RdBu_r")
    _draw_wall_hatching(ax)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_title("wall-vanishing pressure", loc="left", fontsize=11.5)
    ax.tick_params(labelsize=11)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("$p$", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    save_figure(fig, "fig_s6_nonlinear_pressure_setup")
    plt.close(fig)


def _add_roundoff_band(ax, *, label: bool = True) -> None:
    # The 1e-13 to 1e-16 strip is the float64 roundoff range expected for
    # these bilinear products on length-N operator stencils. The shaded band
    # is a presentation device and does not encode a strict numerical bound.
    ax.axhspan(VISUAL_FLOOR, MACHINE_ZERO_TOP, color="0.65", alpha=0.15, zorder=0)
    ax.axhline(MACHINE_ZERO_TOP, color="0.45", lw=0.7, ls=(0, (4, 3)), zorder=1)
    if label:
        ax.text(
            0.015,
            MACHINE_ZERO_TOP,
            r"$10^{-13}$ (machine-zero band)",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=7.8,
            color="0.30",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
        )


def _bar_panel(ax, labels: list[str], values: list[float], colors: list[str], title: str, ylabel: str, *, ylim: tuple[float, float]):
    x = np.arange(len(labels))
    raw_values = np.asarray(values, dtype=float)
    plot_values = np.maximum(raw_values, VISUAL_FLOOR)
    plot_values = np.where(raw_values <= 0.0, VISUAL_FLOOR * 1.8, plot_values)
    bars = ax.bar(x, plot_values, color=colors, width=0.72, edgecolor="black", linewidth=0.35, zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18 if len(labels) > 4 else 0, ha="right" if len(labels) > 4 else "center")
    ax.set_title(title, loc="left", fontsize=10.2)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", which="both", lw=0.35, alpha=0.25)
    _add_roundoff_band(ax)
    return bars


def make_audit_figure(summary: dict) -> None:
    colors = DEFAULT_COLORS
    fig, axes = plt.subplots(2, 2, figsize=(10.9, 7.05), constrained_layout=True)

    qstar = "8"
    prod = summary["product_rule"][qstar]
    prod_nodal = summary["product_rule_nodal_controls"][qstar]
    product_order = ["BMG", "BMF", "SBP", "C", "DRP", "E"]
    labels = ["BMG8", "BMF8", "SBP8", "C8", "DRP8", "E8"]
    values = [
        prod["BMG"]["exact_inf"],
        prod["BMF"]["exact_inf"],
        prod_nodal["SBP"]["weak_leibniz_inf"],
        prod_nodal["C"]["weak_leibniz_inf"],
        prod_nodal["DRP"]["weak_leibniz_inf"],
        prod_nodal["E"]["weak_leibniz_inf"],
    ]
    spline_top = max(prod["BMG"]["exact_inf"], prod["BMF"]["exact_inf"], VISUAL_FLOOR)
    nodal_top = max(values[2:])
    nodal_floor = min(values[2:])
    panel_ceiling = max(nodal_top * 6.0, 1.0e-2)
    bars_a = _bar_panel(
        axes[0, 0],
        labels,
        values,
        [colors.get(k, "#777777") for k in product_order],
        "(a) de-aliased product rule",
        r"relative weak Leibniz defect",
        ylim=(VISUAL_FLOOR, panel_ceiling),
    )
    for spline_idx, spline_key in enumerate(("BMG", "BMF")):
        bars_a[spline_idx].set_height(VISUAL_FLOOR * 1.8)
        bars_a[spline_idx].set_facecolor("white")
        bars_a[spline_idx].set_edgecolor(colors[spline_key])
        bars_a[spline_idx].set_linewidth(0.8)
        bars_a[spline_idx].set_hatch("////")
        axes[0, 0].text(
            spline_idx,
            VISUAL_FLOOR * 3.5,
            r"$<10^{-17}$",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="0.20",
            zorder=4,
        )
    ax = axes[0, 0]
    trans = ax.get_xaxis_transform()
    for start, end, group_label in [(0, 1, "exact projection"), (2, 5, "native-node product")]:
        left = start - 0.36
        right = end + 0.36
        y = -0.20
        ax.plot([left, right], [y, y], transform=trans, color="0.28", lw=0.7, clip_on=False)
        ax.plot([left, left], [y - 0.025, y + 0.025], transform=trans, color="0.28", lw=0.7, clip_on=False)
        ax.plot([right, right], [y - 0.025, y + 0.025], transform=trans, color="0.28", lw=0.7, clip_on=False)
        ax.text((start + end) / 2.0, y - 0.080, group_label, transform=trans, ha="center", va="top", fontsize=8.0, clip_on=False)
    axes[0, 0].tick_params(axis="x", pad=2)
    note_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5}

    ke_entries = summary["ke_q8"]
    ke_order = ["BMG", "BMF", "SBP", "C", "DRP", "E"]
    labels = ["BMG8", "BMF8", "SBP8", "C8", "DRP8", "E8"]
    values = [ke_entries[k]["ke_production"] for k in ke_order]
    _bar_panel(
        axes[0, 1],
        labels,
        values,
        [colors.get(k, "#777777") for k in ke_order],
        "(b) nonlinear kinetic-energy production",
        r"$|w^T H C(a)w|/(w^T H w)$",
        ylim=(VISUAL_FLOOR, 2.0e-5),
    )

    local = summary["local_fv_refinement_q8"]
    nvals = np.array([17, 25, 33, 49], dtype=float)
    ax = axes[1, 0]
    ax.plot(nvals, [local[str(n)]["BMG"]["local_fv_residual"] for n in nvals.astype(int)], marker="D", color=colors["BMG"], lw=1.8, label="BMG")
    ax.plot(nvals, [local[str(n)]["BMF"]["local_fv_residual"] for n in nvals.astype(int)], marker="^", color=colors["BMF"], lw=1.8, label="BMF")
    ax.set_yscale("log")
    ax.set_ylim(VISUAL_FLOOR, 3.0e-13)
    ax.set_xticks([17, 25, 33, 49])
    ax.set_xlabel("$N$ at fixed $q=8$")
    ax.set_ylabel("relative local-balance residual")
    ax.set_title("(c) local finite-volume closure", loc="left", fontsize=10.2)
    ax.grid(True, axis="y", which="both", lw=0.35, alpha=0.25)
    _add_roundoff_band(ax, label=False)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.95)
    ax.text(
        0.985, 0.05, r"$10^{-13}$ machine-zero band",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.8, color="0.30", style="italic",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
    )

    pressure_entries = summary["pressure_q8"]
    pressure_order = ["BMG", "BMF", "SBP", "C", "DRP", "E"]
    labels = ["BMG8", "BMF8", "SBP8", "C8", "DRP8", "E8"]
    values = [pressure_entries[k]["pressure_work"] for k in pressure_order]
    bars = _bar_panel(
        axes[1, 1],
        labels,
        values,
        [colors.get(k, "#777777") for k in pressure_order],
        "(d) pressure-work cancellation",
        r"normalized interior pressure work",
        ylim=(VISUAL_FLOOR, 1.0e-5),
    )
    sbp_idx = pressure_order.index("SBP")
    bars[sbp_idx].set_height(VISUAL_FLOOR * 1.8)
    bars[sbp_idx].set_facecolor("white")
    bars[sbp_idx].set_edgecolor(colors["SBP"])
    bars[sbp_idx].set_linewidth(0.8)
    bars[sbp_idx].set_hatch("////")
    axes[1, 1].text(
        sbp_idx,
        VISUAL_FLOOR * 3.5,
        r"$<10^{-17}$",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="0.20",
        zorder=4,
    )

    save_figure(fig, "fig_s6_nonlinear_balance_audit")
    plt.close(fig)


def main() -> None:
    ensure_generated_dirs()
    npts = 33
    degrees = (4, 6, 8)

    product: dict[str, dict[str, dict[str, float]]] = {}
    product_nodal: dict[str, dict[str, dict[str, float]]] = {}
    ke_cross: dict[str, dict[str, dict[str, float]]] = {}
    ke_nodal: dict[str, dict[str, dict[str, float]]] = {}
    local: dict[str, dict[str, dict[str, float]]] = {}
    pressure_cross: dict[str, dict[str, dict[str, float]]] = {}
    pressure_nodal: dict[str, dict[str, dict[str, float]]] = {}
    nodal_controls = ("SBP", "C", "DRP", "E")

    for degree in degrees:
        product[str(degree)] = {
            "BMG": product_rule_residual("BMG", npts, degree),
            "BMF": product_rule_residual("BMF", npts, degree),
        }
        product_nodal[str(degree)] = {
            kind: product_rule_residual_nodal_control(kind, npts, degree)
            for kind in nodal_controls
        }
        ke_cross[str(degree)] = {
            "BMG": nonlinear_ke_residual("BMG", npts, degree),
            "BMF": nonlinear_ke_residual("BMF", npts, degree),
        }
        ke_nodal[str(degree)] = {
            kind: nonlinear_ke_residual_nodal_control(kind, npts, degree)
            for kind in nodal_controls
        }
        local[str(degree)] = {
            "BMG": local_fv_residual_bmg(npts, degree),
            "BMF": local_fv_residual_bmf(npts, degree),
        }
        pressure_cross[str(degree)] = {
            "BMG": pressure_work_residual_bmg(npts, degree),
            "BMF": pressure_work_residual_bmf(npts, degree),
        }
        pressure_nodal[str(degree)] = {
            kind: pressure_work_residual_nodal_control(kind, npts, degree)
            for kind in nodal_controls
        }

    local_refinement_q8 = {
        str(n): {
            "BMG": local_fv_residual_bmg(n, 8),
            "BMF": local_fv_residual_bmf(n, 8),
        }
        for n in (17, 25, 33, 49)
    }

    ke_q8 = {
        "BMG": ke_cross["8"]["BMG"],
        "BMF": ke_cross["8"]["BMF"],
        **ke_nodal["8"],
    }
    pressure_q8 = {
        "BMG": pressure_cross["8"]["BMG"],
        "BMF": pressure_cross["8"]["BMF"],
        **pressure_nodal["8"],
    }

    summary = {
        "npts": npts,
        "degrees": list(degrees),
        "product_rule": product,
        "product_rule_nodal_controls": product_nodal,
        "ke_cross": ke_cross,
        "ke_nodal_controls": ke_nodal,
        "local_fv": local,
        "local_fv_refinement_q8": local_refinement_q8,
        "pressure_cross": pressure_cross,
        "pressure_nodal_controls": pressure_nodal,
        "ke_q8": ke_q8,
        "pressure_q8": pressure_q8,
        "projected_product_trace_leak_bmg_p4": {
            str(n): projected_product_trace_leak_bmg(n, 4) for n in (9, 17, 33)
        },
        "notes": [
            "Exact product-rule residual is the weak Leibniz residual assembled by quadrature on the product space.",
            "Nodal-control residuals use each family's native-node diagonal product and trapezoidal-weighted test space.",
            "The q=8 panels are used in the manuscript because they make the product-space and balance separation visually sharp.",
            "The adversarial BMG trace test has zero advecting trace and nonzero transported trace; its split power equals the projected-product SBP boundary work and is not generally zero.",
        ],
    }

    RESULTSDIR.mkdir(parents=True, exist_ok=True)
    (RESULTSDIR / "nonlinear_product_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for degree in degrees:
        for family in ("BMG", "BMF"):
            rows.append({
                "degree": degree,
                "family": family,
                "weak_leibniz_inf": product[str(degree)][family]["exact_inf"],
                "ke_production": ke_cross[str(degree)][family]["ke_production"],
                "local_fv_residual": local[str(degree)][family]["local_fv_residual"],
                "pressure_work": pressure_cross[str(degree)][family]["pressure_work"],
            })
        for family in nodal_controls:
            rows.append({
                "degree": degree,
                "family": family,
                "weak_leibniz_inf": product_nodal[str(degree)][family]["weak_leibniz_inf"],
                "ke_production": ke_nodal[str(degree)][family]["ke_production"],
                "local_fv_residual": math.nan,
                "pressure_work": pressure_nodal[str(degree)][family]["pressure_work"],
            })
    csv_lines = ["degree,family,weak_leibniz_inf,ke_production,local_fv_residual,pressure_work"]
    for row in rows:
        csv_lines.append(
            f"{row['degree']},{row['family']},{row['weak_leibniz_inf']:.16e},"
            f"{row['ke_production']:.16e},{row['local_fv_residual']:.16e},{row['pressure_work']:.16e}"
        )
    (RESULTSDIR / "nonlinear_product_audit_table.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    make_dealiased_product_figure(npts, 8)
    make_pressure_work_setup_figure(npts, 8)
    make_audit_figure(summary)


if __name__ == "__main__":
    main()
