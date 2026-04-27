#!/usr/bin/env python3
from __future__ import annotations

import csv
import json

import numpy as np
from scipy.linalg import lu_solve

import boundary_spline_analysis as ba
from project_paths import RESULTSDIR, ensure_generated_dirs

OUT_JSON = RESULTSDIR / 'crossquartet_client_summary.json'
OUT_CSV = RESULTSDIR / 'crossquartet_client_table.csv'
ensure_generated_dirs()

ORDERS = (4, 6, 8)
NPTS_REFLECT = 129
NPTS_MARKER = 81
NPTS_ENERGY = 65
TIMES = np.linspace(0.0, 0.20, 500)
ENERGY_FLOOR = 1.0e-18

x0 = 0.24
carrier_ppw = 6.5
h_reflect = 1.0 / (NPTS_REFLECT - 1)
wavelength = carrier_ppw * h_reflect
k_wave = 2.0 * np.pi / wavelength
sigma = 0.05
t_hit = x0
t_post = 0.35
t_hist_end = 0.36
interior_strip = 10
pre_bounce_window = 0.08

beta = 18.0
omega = 10.0 * np.pi
xc, yc = 0.24, 0.50
radius = 0.11
marker_count = 240
phi = np.linspace(-np.pi, np.pi, marker_count, endpoint=False)
xm = xc + radius * np.cos(phi)
ym = yc + radius * np.sin(phi)
nx = np.cos(phi)
ny = np.sin(phi)
ds = radius * (2.0 * np.pi / marker_count)

ALPHA = 8.0
BETA_NORMAL = 0.2


def field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.exp(-beta * x) * np.sin(omega * x) * np.cos(2.0 * np.pi * y)


def grad_x(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.exp(-beta * x) * (omega * np.cos(omega * x) - beta * np.sin(omega * x)) * np.cos(2.0 * np.pi * y)


def grad_y(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -2.0 * np.pi * np.exp(-beta * x) * np.sin(omega * x) * np.sin(2.0 * np.pi * y)


def interp2(U: np.ndarray, Wx: np.ndarray, Wy: np.ndarray) -> np.ndarray:
    return np.einsum('ki,ij,kj->k', Wx, U, Wy)


u_marker_exact = field(xm, ym)
normal_marker_exact = grad_x(xm, ym) * nx + grad_y(xm, ym) * ny
traction_exact = ALPHA * u_marker_exact + BETA_NORMAL * normal_marker_exact
force_exact = ds * np.array([
    np.sum(traction_exact * nx),
    np.sum(traction_exact * ny),
])
moment_exact = float(ds * np.sum(xm * traction_exact * ny - ym * traction_exact * nx))
scale_u = float(np.linalg.norm(u_marker_exact))
scale_normal = float(np.linalg.norm(normal_marker_exact))
scale_traction = float(np.linalg.norm(traction_exact))
scale_force = float(np.linalg.norm(force_exact))
scale_moment = abs(moment_exact)
flux_exact = float(ds * np.sum(normal_marker_exact))


def quartet(q: int) -> list[tuple[str, str, int]]:
    return ba.comparison_block(q)


def evaluate_marker_scheme(kind: str, degree: int, n: int) -> dict[str, float]:
    op = ba.build_operator(kind, degree, n)
    x = op.x
    X, Y = np.meshgrid(x, x, indexing='ij')
    U = field(X, Y)
    if kind in ('E', 'C', 'SBP', 'DRP'):
        Wx = ba.local_interpolation_matrix(x, xm, degree + 1)
        Wy = ba.local_interpolation_matrix(x, ym, degree + 1)
        value = interp2(U, Wx, Wy)
        gx = interp2(op.differentiate(U), Wx, Wy)
        gy = interp2(U @ op.matrix.T, Wx, Wy)
    elif kind == 'B':
        coeff = lu_solve(op._lu, U)
        coeff = lu_solve(op._lu, coeff.T).T
        _, phix, dphix = ba.spline_eval_matrices(xm, x, degree)
        _, phiy, dphiy = ba.spline_eval_matrices(ym, x, degree)
        value = np.einsum('ki,ij,kj->k', phix, coeff, phiy)
        gx = np.einsum('ki,ij,kj->k', dphix, coeff, phiy)
        gy = np.einsum('ki,ij,kj->k', phix, coeff, dphiy)
    else:
        coeff = lu_solve(op._plu, U)
        coeff = lu_solve(op._plu, coeff.T).T
        gxc = op.differentiate_coefficients(coeff)
        gyc = op.differentiate_coefficients(coeff.transpose(1, 0)).transpose(1, 0)
        _, phix, _ = ba.spline_eval_matrices(xm, x, degree)
        _, phiy, _ = ba.spline_eval_matrices(ym, x, degree)
        value = np.einsum('ki,ij,kj->k', phix, coeff, phiy)
        gx = np.einsum('ki,ij,kj->k', phix, gxc, phiy)
        gy = np.einsum('ki,ij,kj->k', phix, gyc, phiy)
    normal = gx * nx + gy * ny
    flux = float(ds * np.sum(normal))
    traction = ALPHA * value + BETA_NORMAL * normal
    force = ds * np.array([
        np.sum(traction * nx),
        np.sum(traction * ny),
    ])
    moment = float(ds * np.sum(xm * traction * ny - ym * traction * nx))
    return {
        'value_rel_l2': float(np.linalg.norm(value - u_marker_exact) / scale_u),
        'normal_rel_l2': float(np.linalg.norm(normal - normal_marker_exact) / scale_normal),
        'flux_rel': float(abs(flux - flux_exact) / abs(flux_exact)),
        'traction_rel_l2': float(np.linalg.norm(traction - traction_exact) / scale_traction),
        'force_rel': float(np.linalg.norm(force - force_exact) / scale_force),
        'moment_rel': float(abs(moment - moment_exact) / scale_moment),
        'work_2d': float(2.0 * n * op.work_model.total_flops),
    }


def _energy_initial_state(x: np.ndarray) -> np.ndarray:
    return 1.0 + 0.2 * np.sin(8.0 * np.pi * x) + 0.15 * np.cos(9.0 * np.pi * x) + 0.1 * x


def evaluate_energy_scheme(kind: str, degree: int) -> dict[str, float]:
    if kind == 'B':
        ops = ba.bmf_staggered_operators(NPTS_ENERGY, degree)
        H = np.block([
            [ops.h0, np.zeros((ops.h0.shape[0], ops.h1.shape[1]))],
            [np.zeros((ops.h1.shape[0], ops.h0.shape[1])), ops.h1],
        ])
        D = np.block([
            [np.zeros((ops.d01.shape[0], ops.d01.shape[0])), ops.d01],
            [ops.d10, np.zeros((ops.d10.shape[0], ops.d10.shape[0]))],
        ])
        Bmat = np.block([
            [np.zeros((ops.boundary.shape[0], ops.boundary.shape[0])), ops.boundary],
            [ops.boundary.T, np.zeros((ops.boundary.shape[1], ops.boundary.shape[1]))],
        ])
        u0 = np.concatenate([_energy_initial_state(ops.x), _energy_initial_state(ops.eta)])
        audit = 'staggered'
    else:
        op = ba.build_operator(kind, degree, NPTS_ENERGY)
        if kind in ('I', 'SBP'):
            H = op.metric
        else:
            H = np.diag(ba.trapezoidal_weights(NPTS_ENERGY))
        D = op.matrix
        Bmat = ba.endpoint_flux_matrix(NPTS_ENERGY)
        u0 = _energy_initial_state(op.x)
        audit = 'square'

    sol = ba.exact_semidiscrete_evolution(D, u0, TIMES)
    energy = np.array([0.5 * u @ H @ u for u in sol], dtype=float)
    flux = np.array([-0.5 * u @ Bmat @ u for u in sol], dtype=float)
    cumulative = np.zeros_like(TIMES)
    for i in range(1, TIMES.size):
        cumulative[i] = cumulative[i - 1] + 0.5 * (flux[i] + flux[i - 1]) * (TIMES[i] - TIMES[i - 1])
    defect = energy - energy[0] - cumulative
    residual_matrix = H @ D + D.T @ H - Bmat
    residual = float(np.linalg.norm(residual_matrix, 2))
    rate = np.array([
        abs(float(u @ residual_matrix @ u)) / max(float(u @ H @ u), ENERGY_FLOOR)
        for u in sol
    ], dtype=float)
    cum_rate = np.zeros_like(TIMES)
    for i in range(1, TIMES.size):
        cum_rate[i] = cum_rate[i - 1] + 0.5 * (rate[i] + rate[i - 1]) * (TIMES[i] - TIMES[i - 1])
    return {
        'audit': audit,
        'balance_residual_norm2': residual,
        'max_rel_artificial_rate': float(np.max(rate)),
        'cum_rel_artificial_rate': float(cum_rate[-1]),
        'max_defect': float(np.max(np.abs(defect))),
        'final_defect': float(defect[-1]),
    }


pressure_exact, velocity_exact = ba.rigid_wall_acoustics_packet(x0=x0, sigma=sigma, k=k_wave)


def evaluate_reflected_scheme(kind: str, degree: int) -> dict[str, float]:
    op = ba.build_operator(kind, degree, NPTS_REFLECT)
    hist = ba.simulate_reflected_acoustics(
        op,
        pressure_exact,
        velocity_exact,
        t_end=t_hist_end,
        cfl=0.012,
        interior_strip=interior_strip,
        record_count=170,
    )
    post = ba.simulate_reflected_acoustics(
        op,
        pressure_exact,
        velocity_exact,
        t_end=t_post,
        cfl=0.012,
        interior_strip=interior_strip,
        record_count=16,
    )
    sl = ba.trimmed_interior_slice(post.pressure.size, interior_strip)
    return {
        'post_bounce_error': float(
            ba.coupled_relative_l2(
                post.pressure[sl],
                post.pressure_exact[sl],
                post.velocity[sl],
                post.velocity_exact[sl],
            )
        ),
        'time_history_peak': ba.reflected_history_peak(hist.times, hist.error_history, t_hit, pre_bounce_window),
    }


def min_label(metric_map: dict[str, dict[str, float]], key: str) -> str:
    return min(metric_map, key=lambda label: metric_map[label][key])


summary: dict[str, object] = {
    'orders': {},
    'setup': {
        'reflected_wave': {
            'npts': NPTS_REFLECT,
            'carrier_ppw': carrier_ppw,
            'sigma': sigma,
            't_hit': t_hit,
            't_post': t_post,
            'pre_bounce_window': pre_bounce_window,
            'bmf_route': 'staggered primal-pressure and derivative-grid velocity complex; velocity is reconstructed to the primal grid for the common error metric',
        },
        'marker_transfer': {
            'npts': NPTS_MARKER,
            'center': [xc, yc],
            'radius': radius,
            'marker_count': marker_count,
            'traction_formula': 'tau = 8u + 0.2 d_n u',
        },
        'energy': {
            'npts': NPTS_ENERGY,
            'time_window': [float(TIMES[0]), float(TIMES[-1])],
            'bmf_audit': 'staggered flux-divergence system on primal and derivative Greville grids',
        },
    },
}

csv_rows: list[dict[str, float | int | str]] = []
winner_rows: list[dict[str, str | int | float]] = []

for q in ORDERS:
    order_key = f'q{q}'
    order_summary: dict[str, object] = {
        'reflected_wave': {},
        'geometry_transfer': {},
        'traction_transfer': {},
        'energy_balance': {},
    }
    for label, kind, degree in quartet(q):
        reflected = evaluate_reflected_scheme(kind, degree)
        marker = evaluate_marker_scheme(kind, degree, NPTS_MARKER)
        energy = evaluate_energy_scheme(kind, degree)
        order_summary['reflected_wave'][label] = reflected
        order_summary['geometry_transfer'][label] = {
            'value_rel_l2': marker['value_rel_l2'],
            'normal_rel_l2': marker['normal_rel_l2'],
            'flux_rel': marker['flux_rel'],
            'work_2d': marker['work_2d'],
        }
        order_summary['traction_transfer'][label] = {
            'traction_rel_l2': marker['traction_rel_l2'],
            'force_rel': marker['force_rel'],
            'moment_rel': marker['moment_rel'],
            'work_2d': marker['work_2d'],
        }
        order_summary['energy_balance'][label] = energy
        csv_rows.append({
            'q': q,
            'scheme': label,
            'post_bounce_error': reflected['post_bounce_error'],
            'time_history_peak': reflected['time_history_peak'],
            'marker_value_error': marker['value_rel_l2'],
            'marker_normal_error': marker['normal_rel_l2'],
            'marker_flux_error': marker['flux_rel'],
            'traction_density_error': marker['traction_rel_l2'],
            'traction_force_error': marker['force_rel'],
            'traction_moment_error': marker['moment_rel'],
            'energy_max_rel_artificial_rate': energy['max_rel_artificial_rate'],
            'energy_balance_residual_norm2': energy['balance_residual_norm2'],
        })

    reflected_winner = min_label(order_summary['reflected_wave'], 'post_bounce_error')
    geometry_winner = min_label(order_summary['geometry_transfer'], 'normal_rel_l2')
    traction_winner = min_label(order_summary['traction_transfer'], 'traction_rel_l2')
    energy_winner = min_label(order_summary['energy_balance'], 'max_rel_artificial_rate')
    winner_rows.extend([
        {'q': q, 'client': 'reflected_wave', 'winner': reflected_winner, 'metric': 'post_bounce_error', 'value': order_summary['reflected_wave'][reflected_winner]['post_bounce_error']},
        {'q': q, 'client': 'geometry_normals', 'winner': geometry_winner, 'metric': 'normal_rel_l2', 'value': order_summary['geometry_transfer'][geometry_winner]['normal_rel_l2']},
        {'q': q, 'client': 'traction_density', 'winner': traction_winner, 'metric': 'traction_rel_l2', 'value': order_summary['traction_transfer'][traction_winner]['traction_rel_l2']},
        {'q': q, 'client': 'energy_balance', 'winner': energy_winner, 'metric': 'max_rel_artificial_rate', 'value': order_summary['energy_balance'][energy_winner]['max_rel_artificial_rate']},
    ])
    summary['orders'][order_key] = order_summary

summary['winner_rows'] = winner_rows

with OUT_CSV.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
    writer.writeheader()
    writer.writerows(csv_rows)

OUT_JSON.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
