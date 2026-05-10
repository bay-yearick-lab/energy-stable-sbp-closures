#!/usr/bin/env python3
"""Generate `fig_s4_periodic_vs_bounded.pdf`.

Section 4 opening figure.  Strang-style two-panel comparison of how the
discrete derivative `D` acts on a sine mode in two regimes:

    Panel (a):  periodic grid, no walls.
                The mode is a Fourier eigenvector of the periodic derivative,
                so `D s_m = k_mod * c_m` to numerical precision.
                One scalar describes the action of `D` on this mode.

    Panel (b):  clamped bounded interval with walls.
                The mode is no longer an eigenvector of `D`.
                Decomposing `D s_m` in the discrete weighted L^2 metric gives
                `D s_m = kappa_m c_m + r_m`, with `r_m` orthogonal to `c_m`
                in that metric and visibly nonzero near the walls.
                Two coordinates `(rho_m, eta_m)` are needed to characterize
                the action.

The same mode `f(x) = sin(m pi x)` is used in both panels, and at a
comparable points-per-wavelength budget, so the difference between the two
panels is purely the absence or presence of walls.  The figure carries no
coefficient algebra; the prose carries the full standing-mode apparatus.
"""
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
    'font.size': 9.6,
    'axes.titlesize': 10.4,
    'axes.labelsize': 9.6,
    'xtick.labelsize': 8.4,
    'ytick.labelsize': 8.4,
    'figure.dpi': 180,
    'savefig.dpi': 300,
})


ACCENT = '#2F4858'
PANEL_BG = (0.987, 0.985, 0.978)
SINE_COLOR = '#1f77b4'
COSINE_COLOR = '#111111'
KAPPA_COLOR = '#b8483b'
REMAIN_COLOR = '#2ca02c'
WALL_COLOR = '#7a4f1d'


def fft_derivative_periodic(values: np.ndarray, length: float = 1.0) -> np.ndarray:
    """Spectrally accurate periodic derivative on a uniform grid."""
    n = len(values)
    k = np.fft.fftfreq(n, d=length / n) * 2.0 * np.pi
    F = np.fft.fft(values)
    return np.real(np.fft.ifft(1j * k * F))


def style_panel(ax, *, title=None, ylabel=None, ylim=None,
                show_xticks=True, panel_bg=True):
    if panel_bg:
        ax.set_facecolor(PANEL_BG)
    if title:
        ax.set_title(title, color=ACCENT, fontsize=10.4, pad=4)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.4)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(0.0, color='0.6', lw=0.5, zorder=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(0.55)
        ax.spines[s].set_color('0.30')
    ax.tick_params(length=2.6, width=0.55, pad=2.0)
    if not show_xticks:
        ax.tick_params(labelbottom=False)
    ax.set_xlim(-0.005, 1.005)


def main() -> None:
    mode = 12
    periodic_N = 32
    bounded_N = 33

    x_periodic = np.linspace(0.0, 1.0, periodic_N, endpoint=False)
    s_periodic = np.sin(mode * np.pi * x_periodic)
    Ds_periodic = fft_derivative_periodic(s_periodic, length=1.0)
    cos_sample_periodic = np.cos(mode * np.pi * x_periodic)
    k_mod = float(np.sum(Ds_periodic * cos_sample_periodic) /
                  np.sum(cos_sample_periodic * cos_sample_periodic))
    cos_amp = mode * np.pi
    periodic_residual_rel = (Ds_periodic - k_mod * cos_sample_periodic) / cos_amp

    op = ba.build_operator('C', 6, bounded_N)
    x_bounded = op.x
    s_bounded, c_bounded, ds_bounded = ba.wall_mode_values(x_bounded, mode)
    Ds_bounded = op.differentiate(s_bounded)
    weights = ba.trapezoidal_weights(bounded_N)
    kappa = float(np.sum(weights * Ds_bounded * c_bounded) /
                  np.sum(weights * c_bounded * c_bounded))
    proj = kappa * c_bounded
    remainder = Ds_bounded - proj
    bounded_residual_rel = remainder / cos_amp
    rho = kappa / cos_amp
    eta = float(np.sqrt(np.sum(weights * remainder ** 2)) /
                np.sqrt(np.sum(weights * ds_bounded ** 2)))
    h_ref = float(np.max(np.diff(x_bounded)))
    ppw_bounded = 2.0 / (mode * h_ref)

    x_dense = np.linspace(0.0, 1.0, 1001)
    s_dense = np.sin(mode * np.pi * x_dense)
    c_dense = np.cos(mode * np.pi * x_dense)
    s_lim = 1.22

    res_max = max(np.max(np.abs(periodic_residual_rel)),
                  np.max(np.abs(bounded_residual_rel)))
    res_lim = 1.25 * max(res_max, 0.02)

    fig = plt.figure(figsize=(11.2, 5.4), layout='constrained')
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.0, 1.0, 0.78],
        width_ratios=[1.0, 1.0],
        hspace=0.18,
        wspace=0.10,
    )

    ax_a_top = fig.add_subplot(gs[0, 0])
    ax_a_mid = fig.add_subplot(gs[1, 0], sharex=ax_a_top)
    ax_a_bot = fig.add_subplot(gs[2, 0], sharex=ax_a_top)
    ax_b_top = fig.add_subplot(gs[0, 1])
    ax_b_mid = fig.add_subplot(gs[1, 1], sharex=ax_b_top)
    ax_b_bot = fig.add_subplot(gs[2, 1], sharex=ax_b_top)

    title_a = '(a) periodic grid: one eigenmap captures the response'
    title_b = '(b) bounded grid: walls couple modes into a residual'

    style_panel(ax_a_top,
                title=title_a,
                ylabel=r'$s_m(x)=\sin(m\pi x)$',
                ylim=(-s_lim, s_lim),
                show_xticks=False)
    ax_a_top.plot(x_dense, s_dense, color=SINE_COLOR, lw=1.0, alpha=0.55,
                  zorder=1)
    ax_a_top.plot(x_periodic, s_periodic, marker='o', linestyle='None',
                  markersize=3.6, color=SINE_COLOR,
                  markeredgecolor='white', markeredgewidth=0.4, zorder=3)

    style_panel(ax_a_mid,
                ylabel=r"$Ds_m / (m\pi)$",
                ylim=(-s_lim, s_lim),
                show_xticks=False)
    ax_a_mid.plot(x_dense, c_dense, color=COSINE_COLOR, lw=1.1, alpha=0.55,
                  label=r'exact $c_m$', zorder=1)
    ax_a_mid.plot(x_periodic, Ds_periodic / cos_amp, marker='s',
                  linestyle='None', markersize=4.2, color=KAPPA_COLOR,
                  markeredgecolor='white', markeredgewidth=0.4,
                  label=r'$Ds_m/(m\pi)$', zorder=3)
    ax_a_mid.legend(frameon=False, loc='upper right', fontsize=8.2,
                    handlelength=1.6, ncol=2)

    style_panel(ax_a_bot,
                ylabel=r"$r_m / (m\pi)$",
                ylim=(-res_lim, res_lim))
    ax_a_bot.axhline(0.0, color='0.6', lw=0.5, zorder=0)
    for xi, ri in zip(x_periodic, periodic_residual_rel):
        ax_a_bot.plot([xi, xi], [0.0, ri], color=REMAIN_COLOR, lw=0.8,
                      alpha=0.85, zorder=2)
    ax_a_bot.plot(x_periodic, periodic_residual_rel, marker='o',
                  linestyle='None', markersize=3.0, color=REMAIN_COLOR,
                  markeredgecolor='white', markeredgewidth=0.4, zorder=3)
    ax_a_bot.set_xlabel(r'$x$', fontsize=9.4)
    ax_a_bot.text(0.985, 0.86,
                  rf'$\|r_m\|_\infty/(m\pi)={np.max(np.abs(periodic_residual_rel)):.1e}$',
                  ha='right', va='top', transform=ax_a_bot.transAxes,
                  fontsize=8.4, color=ACCENT, family='serif', style='italic')

    style_panel(ax_b_top,
                title=title_b,
                ylabel=r'$s_m(x)=\sin(m\pi x)$',
                ylim=(-s_lim, s_lim),
                show_xticks=False)
    ax_b_top.axvline(0.0, color=WALL_COLOR, lw=1.0, alpha=0.55, zorder=0)
    ax_b_top.axvline(1.0, color=WALL_COLOR, lw=1.0, alpha=0.55, zorder=0)
    ax_b_top.plot(x_dense, s_dense, color=SINE_COLOR, lw=1.0, alpha=0.55,
                  zorder=1)
    ax_b_top.plot(x_bounded, s_bounded, marker='o', linestyle='None',
                  markersize=3.6, color=SINE_COLOR,
                  markeredgecolor='white', markeredgewidth=0.4, zorder=3)
    ax_b_top.text(0.012, 0.92, r'wall', ha='left', va='top',
                  transform=ax_b_top.transAxes, fontsize=8.0,
                  color=WALL_COLOR, family='serif', style='italic')
    ax_b_top.text(0.988, 0.92, r'wall', ha='right', va='top',
                  transform=ax_b_top.transAxes, fontsize=8.0,
                  color=WALL_COLOR, family='serif', style='italic')

    style_panel(ax_b_mid,
                ylabel=r"$Ds_m / (m\pi)$",
                ylim=(-s_lim, s_lim),
                show_xticks=False)
    ax_b_mid.axvline(0.0, color=WALL_COLOR, lw=1.0, alpha=0.55, zorder=0)
    ax_b_mid.axvline(1.0, color=WALL_COLOR, lw=1.0, alpha=0.55, zorder=0)
    ax_b_mid.plot(x_dense, c_dense, color=COSINE_COLOR, lw=1.1, alpha=0.55,
                  label=r'exact $c_m$', zorder=1)
    ax_b_mid.plot(x_bounded, proj / cos_amp, color=KAPPA_COLOR, lw=1.0,
                  linestyle=(0, (4, 2)), label=r'$\kappa_m c_m/(m\pi)$',
                  zorder=2)
    ax_b_mid.legend(frameon=False, loc='upper right', fontsize=8.2,
                    handlelength=1.6, ncol=2)

    style_panel(ax_b_bot,
                ylabel=r"$r_m / (m\pi)$",
                ylim=(-res_lim, res_lim))
    ax_b_bot.axhline(0.0, color='0.6', lw=0.5, zorder=0)
    ax_b_bot.axvline(0.0, color=WALL_COLOR, lw=1.0, alpha=0.55, zorder=0)
    ax_b_bot.axvline(1.0, color=WALL_COLOR, lw=1.0, alpha=0.55, zorder=0)
    for xi, ri in zip(x_bounded, bounded_residual_rel):
        ax_b_bot.plot([xi, xi], [0.0, ri], color=REMAIN_COLOR, lw=0.9,
                      alpha=0.9, zorder=2)
    ax_b_bot.plot(x_bounded, bounded_residual_rel, marker='o',
                  linestyle='None', markersize=3.0, color=REMAIN_COLOR,
                  markeredgecolor='white', markeredgewidth=0.4, zorder=3)
    ax_b_bot.set_xlabel(r'$x$', fontsize=9.4)
    ax_b_bot.text(0.985, 0.86,
                  rf'$\eta_m={eta:.3f}$',
                  ha='right', va='top', transform=ax_b_bot.transAxes,
                  fontsize=8.4, color=ACCENT, family='serif', style='italic')

    fig.text(0.265, 0.012,
             rf'one number says it all: '
             rf'$k_{{\mathrm{{mod}}}}/(m\pi)={k_mod/cos_amp:.4f}$',
             ha='center', va='bottom', fontsize=8.8,
             color=ACCENT, family='serif', style='italic')
    fig.text(0.755, 0.012,
             rf'two numbers: $\rho_m={rho:.3f},\ \eta_m={eta:.3f}$ '
             rf'(at $\mathrm{{PPW}}\!\approx\!{ppw_bounded:.1f}$)',
             ha='center', va='bottom', fontsize=8.8,
             color=ACCENT, family='serif', style='italic')

    save_figure(fig, 'fig_s4_periodic_vs_bounded', pad_inches=0.05)
    plt.close(fig)
    print('wrote section 4 periodic vs bounded modes figure')
    print(f'  k_mod / (m*pi) = {k_mod / cos_amp:.6f}')
    print(f'  rho_m = {rho:.6f}, eta_m = {eta:.6f}, '
          f'PPW = {ppw_bounded:.3f}')


if __name__ == '__main__':
    main()
