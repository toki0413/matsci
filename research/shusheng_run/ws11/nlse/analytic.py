"""Analytic solution, invariants and grid utilities for the 1-D focusing NLSE.

Equation:   i psi_t + psi_xx + 2 |psi|^2 psi = 0.
"""

from __future__ import annotations

import numpy as np

from . import params as P

# --------------------------------------------------------------------------- #
# Exact travelling-wave solution
# --------------------------------------------------------------------------- #
def exact(x: np.ndarray, t: float, A: float = P.A, v: float = P.V) -> np.ndarray:
    """Exact one-soliton solution at time t on array x.

        psi(x,t) = A sech[A(x - v t)] exp{ i [ v x/2 + (A^2 - v^2/4) t ] }
    """
    xi = A * (x - v * t)
    env = A / np.cosh(xi)                     # A sech(xi)
    phase = 0.5 * v * x + (A * A - 0.25 * v * v) * t
    return env * np.exp(1j * phase)


def intensity(x: np.ndarray, t: float) -> np.ndarray:
    """|psi(x,t)|^2 = A^2 sech^2[A(x - v t)]."""
    A = P.A
    return (A / np.cosh(A * (x - P.V * t))) ** 2


# --------------------------------------------------------------------------- #
# Invariants (exact values on the whole real line)
# --------------------------------------------------------------------------- #
def energy_exact(A: float = P.A, v: float = P.V) -> float:
    """E = int (|psi_x|^2 - |psi|^4) dx = A (v^2/2 - 2 A^2/3)."""
    return A * (0.5 * v * v - 2.0 / 3.0 * A * A)


def momentum_exact(A: float = P.A, v: float = P.V) -> float:
    """P = Im int conj(psi) psi_x dx = A v."""
    return A * v


def energy_numerical(psi: np.ndarray, dx: float) -> float:
    """E = int (|psi_x|^2 - |psi|^4) dx by second-order finite differences.

    Periodic boundary conditions are assumed (the grid is the FFT domain).
    """
    # |psi_x|^2 via central differences, periodic
    dpsi = np.roll(psi, -1) - np.roll(psi, 1)
    dpsi /= 2.0 * dx
    kinetic = np.sum(np.abs(dpsi) ** 2) * dx
    potential = np.sum(np.abs(psi) ** 4) * dx
    return kinetic - potential


def momentum_numerical(psi: np.ndarray, dx: float) -> float:
    """P = Im int conj(psi) psi_x dx, central differences, periodic."""
    dpsi = np.roll(psi, -1) - np.roll(psi, 1)
    dpsi /= 2.0 * dx
    return float(np.imag(np.sum(np.conj(psi) * dpsi)) * dx)


# --------------------------------------------------------------------------- #
# Grid / refinement machinery
# --------------------------------------------------------------------------- #
def level_params(ell: int, N0: int = P.N0, dt0: float = P.DT0, L: float = P.L):
    """Return (N, dx, dt, x, k) for refinement level ell (0-based).

        dx_ell = (2L / N0) * 2^{-ell},   dt_ell = dt0 * 2^{-ell}.
    """
    N = N0 * (1 << ell)
    dx = 2.0 * L / N
    dt = dt0 / (1 << ell)
    x = np.arange(N, dtype=float) * dx - L          # [-L, L)  (periodic grid)
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)       # Fourier wavenumbers
    return N, dx, dt, x, k


def l2_relative(psi_num: np.ndarray, psi_ex: np.ndarray) -> float:
    """Relative L2 error  ||psi_num - psi_ex||_2 / ||psi_ex||_2  on the grid."""
    num = np.linalg.norm(psi_num - psi_ex)
    den = np.linalg.norm(psi_ex)
    return float(num / den) if den > 0 else float("nan")


def richardson_order(err_coarse: float, err_fine: float, ratio: float = 2.0) -> float:
    """Observed convergence order given a refinement ratio `ratio` (here 2).

        p = log2(err_coarse / err_fine)   when ratio = 2.
    """
    if err_fine <= 0.0 or err_coarse <= 0.0:
        return float("nan")
    return float(np.log2(err_coarse / err_fine) / np.log2(ratio))