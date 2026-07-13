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
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9.7,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 180,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.18,
    'grid.linewidth': 0.55,
})

n = 65
mode = 26
op = ba.build_operator('C', 6, n)
x = op.x
s, c, ds = ba.wall_mode_values(x, mode)
num = op.differentiate(s)
weights = ba.trapezoidal_weights(n)
kappa = float(np.sum(weights * num * c) / np.sum(weights * c * c))
proj = kappa * c
resid = num - proj
eta = float(np.sqrt(np.sum(weights * resid ** 2)) / np.sqrt(np.sum(weights * ds ** 2)))
rho_minus_1_signed = float(kappa / (mode * np.pi) - 1.0)
combined = float(np.sqrt(rho_minus_1_signed ** 2 + eta ** 2))
ppw = 2.0 / (mode * (x[1] - x[0]))
resid_rel = resid / (mode * np.pi)

print(f'  PPW            ~ {ppw:.2f}')
print(f'  |rho_m - 1|    = {abs(rho_minus_1_signed):.4f}')
print(f'  eta_m          = {eta:.4f}')
print(f'  combined error = {combined:.4f}')

fig, axs = plt.subplots(1, 3, figsize=(13.0, 3.6), layout='constrained')

axs[0].plot(x, s, lw=2.0, color='#1f77b4')
axs[0].set_title('(a) standing wall mode $s_m$')
axs[0].set_xlabel(r'$x$')
axs[0].set_ylabel('amplitude')

axs[1].plot(x, ds, lw=2.0, color='black', label=r'exact $m\pi c_m$')
axs[1].plot(x, proj, lw=2.0, color='#d62728', ls='--', label=r'$\kappa_m c_m$')
axs[1].set_title('(b) response projection onto $c_m$')
axs[1].set_xlabel(r'$x$')
axs[1].legend(loc='upper right', framealpha=0.95)

axs[2].axhline(0.0, lw=1.0, color='0.35')
axs[2].plot(x, resid_rel, lw=2.0, color='#2ca02c', label=r'$r_m / (m\pi)$')
axs[2].set_title('(c) orthogonal contamination remainder')
axs[2].set_xlabel(r'$x$')
axs[2].legend(loc='upper right', framealpha=0.95)

save_figure(fig, 'fig_s4_wallmode_decomp', pad_inches=0.03)
plt.close(fig)
print('wrote wallmode decomp')
