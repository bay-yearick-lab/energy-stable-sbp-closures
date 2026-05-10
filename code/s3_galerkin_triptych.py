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
    'axes.labelsize': 10.0,
    'axes.titlesize': 10.4,
    'xtick.labelsize': 8.8,
    'ytick.labelsize': 8.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 180,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.18,
    'grid.linewidth': 0.55,
})


BLUE = '#1f77b4'
RED = '#d62728'


def _signed_support_panel(
    ax: plt.Axes,
    mat: np.ndarray,
    title: str,
    *,
    xlabel: str = 'basis index',
    ylabel: str = 'basis index',
) -> None:
    scale = max(float(np.max(np.abs(mat))), 1.0)
    tol = 1.0e-12 * scale
    positive = mat > tol
    negative = mat < -tol

    if np.any(positive):
        ax.spy(positive, markersize=2.7, color=BLUE, marker='s')
    if np.any(negative):
        ax.spy(negative, markersize=2.7, color=RED, marker='s')

    nrows, ncols = mat.shape
    ax.set_title(title, pad=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)
    ax.set_xticks(np.arange(0, ncols, 5))
    ax.set_yticks(np.arange(0, nrows, 5))
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('bottom')
    ax.grid(True, color='0.78', linewidth=0.45, alpha=0.35)


NPTS = 33
DEGREE = 3

_x, _knots, _bmat, _bx = ba.clamped_bspline_matrices(NPTS, DEGREE)
mass, adv, _weak_d2, _stiffness = ba.bspline_mass_advection_second_and_stiffness(NPTS, DEGREE)
green = adv + adv.T

fig, axs = plt.subplots(1, 3, figsize=(10.9, 3.85), layout='constrained')

_signed_support_panel(axs[0], mass, r'(a) coefficient mass $M$')
_signed_support_panel(axs[1], adv, r"(b) coefficient derivative $K$")
_signed_support_panel(axs[2], green, r"(c) boundary form $K+K^T$")

save_figure(fig, 'fig_s3_galerkin_triptych', pad_inches=0.04)
plt.close(fig)
print('wrote Galerkin coefficient triptych')
