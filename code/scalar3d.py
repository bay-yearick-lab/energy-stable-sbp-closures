#!/usr/bin/env python3
"""Three-dimensional passive-scalar transport for the S6.7 wall-heat-transfer
head-to-head. The flow field is shared; the streamwise/spanwise (x,y) treatment is
shared Fourier spectral; only the wall-normal (z) operator differs between schemes.
That isolates the wall closure as the single variable behind the Nusselt number.

  d(theta)/dt + u . grad theta = alpha lap theta,   isothermal walls.

x,y: periodic, Fourier spectral derivatives (identical for every scheme).
z:   each scheme's nodal first/second-derivative operators (wall closure built in),
     taken from boundary_spline_analysis.build_operator.
time: SSP-RK3 for skew-symmetric advection (+ source), backward-Euler implicit
     diffusion solved mode-by-mode in the eigenbasis of the wall-normal operator.
Nusselt: the wall heat flux is read with the same scheme's first-derivative wall
     row, so the wall closure under test sets both the transport and the readout.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import numpy as np
import boundary_spline_analysis as ba
from cheb_operator import build_cheb_operator


def z_operators(kind, order, nz, Lz):
    if kind.upper() in ("CHEB", "CHEBYSHEV"):
        op = build_cheb_operator(nz, Lz)
    elif kind.upper() in ("C6-D2", "SBP6-BO", "C6F") and hasattr(ba, "build_operator_extended"):
        # Task-3C full-strength competitors (compact C6-D2, boundary-optimized
        # SBP6-BO, filtered C6F), appended to boundary_spline_analysis.py and
        # dispatched through the companion build_operator_extended so the
        # original build_operator dispatcher stays untouched.
        op = ba.build_operator_extended(kind, order, nz, Lz)
    else:
        op = ba.build_operator(kind, order, nz, Lz)
    z = np.asarray(op.x, dtype=float)
    D1 = np.asarray(op.matrix, dtype=float)
    try:
        D2 = np.asarray(op.second_derivative_matrix(), dtype=float)
    except Exception:
        D2 = D1 @ D1
    return z, D1, D2


class ScalarChannel3D:
    def __init__(self, kind, order, Nx, Ny, nz, Lx, Ly, Lz, alpha):
        self.kind = kind
        self.Nx, self.Ny, self.nz = Nx, Ny, nz
        self.Lx, self.Ly, self.Lz, self.alpha = Lx, Ly, Lz, alpha
        self.z, self.D1, self.D2 = z_operators(kind, order, nz, Lz)
        # x,y spectral wavenumbers (broadcast to (Nx,Ny,1))
        kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
        self.KX = kx[:, None, None]
        self.KY = ky[None, :, None]
        self.K2 = (kx[:, None] ** 2 + ky[None, :] ** 2)[:, :, None]   # (Nx,Ny,1)
        # interior wall-normal eigenbasis for implicit diffusion (Dirichlet walls)
        self.intr = np.arange(1, nz - 1)
        D2i = self.D2[np.ix_(self.intr, self.intr)]
        lam, V = np.linalg.eig(D2i)
        self.lam = lam.real if np.allclose(lam.imag, 0, atol=1e-8) else lam
        self.V = V
        self.Vinv = np.linalg.inv(V)

    # ---- spectral helpers ----
    def ddx(self, f):
        return np.real(np.fft.ifft2(1j * self.KX * np.fft.fft2(f, axes=(0, 1)), axes=(0, 1)))

    def ddy(self, f):
        return np.real(np.fft.ifft2(1j * self.KY * np.fft.fft2(f, axes=(0, 1)), axes=(0, 1)))

    def ddz(self, f):
        return f @ self.D1.T

    def apply_walls(self, f):
        f[:, :, 0] = 0.0
        f[:, :, -1] = 0.0
        return f

    # ---- skew-symmetric advection residual C(theta) ----
    def advect(self, u, v, w, th):
        tx, ty, tz = self.ddx(th), self.ddy(th), self.ddz(th)
        adv = u * tx + v * ty + w * tz
        div = self.ddx(u * th) + self.ddy(v * th) + (w * th) @ self.D1.T
        return 0.5 * (adv + div)

    def adv_rhs(self, u, v, w, th, src):
        return -self.advect(u, v, w, th) + src

    # ---- backward-Euler implicit diffusion in the wall-normal eigenbasis ----
    def diffuse(self, th, dt):
        a = self.alpha
        hat = np.fft.fft2(th, axes=(0, 1))                      # (Nx,Ny,nz) complex
        hi = hat[:, :, self.intr]                               # interior in z
        c = hi @ self.Vinv.T                                    # to eigenbasis
        denom = (1.0 + dt * a * self.K2) - dt * a * self.lam[None, None, :]
        c = c / denom
        hi = c @ self.V.T
        hat[:, :, self.intr] = hi
        hat[:, :, 0] = 0.0
        hat[:, :, -1] = 0.0
        return np.real(np.fft.ifft2(hat, axes=(0, 1)))

    # ---- one step: SSP-RK3 advection (+ source) then implicit diffusion ----
    def step(self, u, v, w, th, dt, src=0.0):
        s = src if np.ndim(src) else np.full_like(th, src)
        th0 = th
        t1 = self.apply_walls(th0 + dt * self.adv_rhs(u, v, w, th0, s))
        t2 = self.apply_walls(0.75 * th0 + 0.25 * t1 + 0.25 * dt * self.adv_rhs(u, v, w, t1, s))
        th = self.apply_walls((1.0 / 3.0) * th0 + (2.0 / 3.0) * t2 + (2.0 / 3.0) * dt * self.adv_rhs(u, v, w, t2, s))
        return self.diffuse(th, dt)

    # ---- Nusselt ingredient: plane-averaged wall heat flux of the total field ----
    def wall_flux(self, th_total):
        dthdz = th_total @ self.D1.T                           # (Nx,Ny,nz)
        bot = dthdz[:, :, 0].mean()
        top = dthdz[:, :, -1].mean()
        return bot, top


# ----------------------------- verification -----------------------------
def mms_diffusion(kind):
    Nx = Ny = 16
    nz = 33
    Lx = Ly = 2 * np.pi
    Lz = 2.0
    alpha = 0.05
    S = ScalarChannel3D(kind, 6, Nx, Ny, nz, Lx, Ly, Lz, alpha)
    x = (np.arange(Nx) * Lx / Nx)[:, None, None]
    y = (np.arange(Ny) * Ly / Ny)[None, :, None]
    z = S.z[None, None, :]
    kx, ky = 2.0, 1.0
    shape = np.sin(kx * x) * np.sin(ky * y) * np.sin(np.pi * z / Lz)
    lam = kx ** 2 + ky ** 2 + (np.pi / Lz) ** 2
    th = shape.copy()
    u = np.zeros_like(th)
    dt, T = 1e-3, 0.3
    for _ in range(int(round(T / dt))):
        th = S.step(u, u, u, th, dt)
    t = int(round(T / dt)) * dt
    exact = np.exp(-alpha * lam * t) * shape
    return np.max(np.abs(th - exact)) / np.max(np.abs(exact))


def mms_constant(kind):
    # analytic pointwise-divergence-free field must leave a constant scalar untouched
    Nx = Ny = 16
    nz = 33
    Lx = Ly = 2 * np.pi
    Lz = 2.0
    S = ScalarChannel3D(kind, 6, Nx, Ny, nz, Lx, Ly, Lz, 0.0)
    x = (np.arange(Nx) * Lx / Nx)[:, None, None]
    z = S.z[None, None, :]
    # psi = sin(x) sin(pi z/Lz): u = d psi/dz, w = -d psi/dx  (div-free in x,z)
    u = (np.pi / Lz) * np.sin(x) * np.cos(np.pi * z / Lz) * np.ones((Nx, Ny, nz))
    v = np.zeros((Nx, Ny, nz))
    w = -np.cos(x) * np.sin(np.pi * z / Lz) * np.ones((Nx, Ny, nz))
    th = np.ones((Nx, Ny, nz))
    C = S.advect(u, v, w, th)
    return np.max(np.abs(C))


if __name__ == "__main__":
    print(f"{'scheme':6s} {'diffusion MMS':>16s} {'const annihil.':>16s}")
    for k in ["E", "C", "SBP", "DRP", "B", "I", "CHEB"]:
        try:
            print(f"{k:6s} {mms_diffusion(k):16.2e} {mms_constant(k):16.2e}")
        except Exception as e:
            print(f"{k:6s}  FAIL: {e}")
