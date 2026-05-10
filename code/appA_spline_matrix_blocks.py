#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import boundary_spline_analysis as ba
from project_paths import ensure_generated_dirs, save_figure

ensure_generated_dirs()

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10.0,
    'axes.labelsize': 10.8,
    'axes.titlesize': 11.0,
    'xtick.labelsize': 8.8,
    'ytick.labelsize': 8.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 180,
    'savefig.dpi': 300,
})


def _nonzero_mask(mat: np.ndarray, tol: float = 1.0e-14) -> np.ndarray:
    arr = np.asarray(mat, dtype=float)
    return np.abs(arr) > tol


def _sampled_derivative_support_mask(B: np.ndarray, Bx: np.ndarray, tol: float = 1.0e-14) -> np.ndarray:
    return _nonzero_mask(B, tol=tol) | _nonzero_mask(Bx, tol=tol)


NPTS = 33
DEGREE = 6
_x, _knots, B, Bx = ba.clamped_bspline_matrices(NPTS, DEGREE)
M, K, W, S = ba.bspline_mass_advection_second_and_stiffness(NPTS, DEGREE)

fig, axs = plt.subplots(2, 3, figsize=(12.1, 7.9), layout='constrained')
panels = [
    (axs[0, 0], B, '#1f77b4', r'(a) Nonzero pattern of $B_{ij}=N_j(x_i)$'),
    (axs[0, 1], _sampled_derivative_support_mask(B, Bx), '#d62728', r"(b) Structural support of $(B')_{ij}=N_j'(x_i)$"),
    (axs[0, 2], M, '#1f77b4', r'(c) Nonzero pattern of $M_{ij}=\int N_iN_j$'),
    (axs[1, 0], K, '#d62728', r"(d) Nonzero pattern of $K_{ij}=\int N_iN_j'$") ,
    (axs[1, 1], W, '#9467bd', r"(e) Nonzero pattern of $W_{ij}=\int N_iN_j''$") ,
    (axs[1, 2], S, '#2ca02c', r"(f) Nonzero pattern of $S_{ij}=\int N_i'N_j'$") ,
]

for ax, mat, color, title in panels:
    mask = _nonzero_mask(mat)
    ax.spy(mask, markersize=3.8, color=color, marker='s')
    ax.set_title(title, pad=6)
    ax.set_xlabel('column index')
    ax.set_ylabel('row index')
    ax.set_xticks([0, NPTS // 2, NPTS - 1])
    ax.set_yticks([0, NPTS // 2, NPTS - 1])

save_figure(fig, 'fig_appA_spline_matrix_blocks', pad_inches=0.03)
plt.close(fig)
print('wrote spline matrix blocks')