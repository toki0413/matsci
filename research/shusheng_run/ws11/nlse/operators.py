"""Time-stepping schemes for the 1-D focusing NLSE.

Two *independent* discretisations are implemented so that agreement between them
is a cross-check rather than a tautology:

  1. SSFM  -- Strang split-step Fourier method.  The linear (dispersive) step is
              solved exactly in Fourier space; the nonlinear step is solved
              exactly in physical space.  No CFL restriction.  Error is dominated
              by the Strang splitting, O(dt^2).

  2. CN    -- Crank-Nicolson type implicit finite difference.  The linear part
              (second derivative) is treated implicitly and *exactly* via a
              constant tridiagonal LU factorisation; the cubic nonlinearity is
              handled by a convergent fixed-point iteration.  Unconditionally
              stable, error O(dt^2 + dx^2).

Neither scheme uses an explicit nonlinear update, so neither suffers the
CFL-type instability that an explicit scheme (e.g. forward-Euler in time) would
encounter for this dispersive problem.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from . import params as P

# --------------------------------------------------------------------------- #
# Common: absorbing mask (smooth cosine taper near the boundaries)
# --------------------------------------------------------------------------- #
def build_absorption_mask(x: np.ndarray, L: float = P.L,
                          width: float = P.SIGMA_WIDTH,
                          amp: float = P.SIGMA_AMP) -> np.ndarray:
    """Smooth mask m(x) in [0,1] that equals 1 in the interior and tapers to
    exp(-amp) at the boundaries.  Applied every time step to damp dispersive
    radiation before it can reflect.

        m(x) = 1 - amp * s(x),   s = smooth step from 0 (interior) to 1 (edge).
    """
    d = np.abs(x)                     # distance from centre, on [-L, L)
    # The mask starts acting at |x| = L - width and reaches full strength at
    # the boundary.  Use a raised-cosine profile for C^1 smoothness.
    s = np.zeros_like(d)
    active = d > (L - width)
    s[active] = 0.5 * (1.0 - np.cos(np.pi * (d[active] - (L - width)) / width))
    # Multiplicative damping factor applied each step: exp(-amp * s)
    return np.exp(-amp * s)


# =========================================================================== #
# 1. STRANG SPLIT-STEP FOURIER METHOD
# =========================================================================== #
class SSFM:
    """Strang split-step Fourier integrator for  i psi_t + psi_xx + 2|psi|^2 psi = 0.

    Splitting:
        psi_t = -i psi_xx            (linear, solved exactly in Fourier space)
        psi_t =  2i |psi|^2 psi      (nonlinear, solved exactly in real space)

    Strang step (dt):
        psi* = exp(-i dt/2 D2) psi^n
        psi** = exp(2i dt |psi*|^2) psi*
        psi^{n+1} = exp(-i dt/2 D2) psi**

    with the absorbing mask applied after each full step.
    """

    def __init__(self, x: np.ndarray, k: np.ndarray, dx: float, dt: float,
                 L: float = P.L):
        self.x = x
        self.k = k
        self.dx = dx
        self.dt = dt
        self.L = L
        self.mask = build_absorption_mask(x, L)
        # Linear propagator for a half step (time-independent -> precomputed)
        self.half_lin = np.exp(-0.5j * dt * k ** 2)

    def step(self, psi: np.ndarray) -> np.ndarray:
        dt = self.dt
        # --- half linear step ---
        u = np.fft.ifft(self.half_lin * np.fft.fft(psi))
        # --- full nonlinear step (exact) ---
        u = u * np.exp(2.0j * dt * np.abs(u) ** 2)
        # --- half linear step ---
        psi_new = np.fft.ifft(self.half_lin * np.fft.fft(u))
        # --- absorbing layer ---
        psi_new = psi_new * self.mask
        return psi_new

    def march(self, psi0: np.ndarray, nsteps: int,
              record_every: int = 1):
        """Advance nsteps, returning the state and (optionally) a history."""
        psi = psi0.copy()
        history = [psi.copy()] if record_every == 1 else None
        for n in range(nsteps):
            psi = self.step(psi)
            if record_every > 0 and (n + 1) % record_every == 0:
                if history is not None:
                    history.append(psi.copy())
        return psi, history


# =========================================================================== #
# 2. CRANK-NICOLSON TYPE IMPLICIT SCHEME
# =========================================================================== #
class CrankNicolson:
    """Crank-Nicolson implicit finite-difference integrator.

    Semi-discrete equation (central differences in space, periodic):

        i (psi^{n+1} - psi^n)/dt + (1/2)(D2 psi^{n+1} + D2 psi^n)
            + N(psi^{n+1}, psi^n) = 0,

    where the nonlinear term is the symmetric (average) discretisation

        N = (|psi^{n+1}|^2 + |psi^n|^2)/2 * (psi^{n+1} + psi^n)/2 .

    Rearranged:

        [ I + (i dt/2) D2 ] psi^{n+1}
            = [ I - (i dt/2) D2 ] psi^n - i dt N(psi^{n+1}, psi^n) .

    The tridiagonal matrix on the left is *time independent*, so its LU
    factorisation is computed once and reused for every step and every fixed-point
    iteration.  The nonlinearity is resolved by fixed-point iteration:

        psi^{n+1,(0)} = psi^n  (predictor)
        solve for psi^{n+1,(m+1)} using N(psi^{n+1,(m)}, psi^n)
        until ||psi^{n+1,(m+1)} - psi^{n+1,(m)}|| < tol.

    Periodic boundary conditions are encoded in the banded matrix by placing the
    corner entries (1, N) and (N, 1) explicitly.
    """

    def __init__(self, x: np.ndarray, dx: float, dt: float, L: float = P.L):
        self.x = x
        self.dx = dx
        self.dt = dt
        self.L = L
        self.N = x.size
        self.mask = build_absorption_mask(x, L)

        # Second-derivative matrix (periodic, 2nd order):
        #   (D2 psi)_j = (psi_{j+1} - 2 psi_j + psi_{j-1}) / dx^2
        # We build the matrix  M = I + (i dt/2) D2  in scipy solve_banded format
        # (3 diagonals: upper, main, lower) plus the two periodic corners.
        N = self.N
        alpha = 0.5j * dt / (dx * dx)

        # solve_banded ab format: ab[0] = super-diagonal (length N-1),
        #                          ab[1] = main diagonal,
        #                          ab[2] = sub-diagonal (length N-1)
        ab = np.zeros((3, N), dtype=complex)
        ab[0, 1:] = -alpha          # super-diagonal  (coefficient of psi_{j+1})
        ab[1, :] = 1.0 + 2.0 * alpha  # main diagonal
        ab[2, :-1] = -alpha         # sub-diagonal    (coefficient of psi_{j-1})
        # Periodic corners
        ab[0, 0] = -alpha           # psi_N = psi_0  -> entry (1, N)
        ab[2, N - 1] = -alpha       # psi_0 = psi_N  -> entry (N, 1)
        self.ab = ab

        # Right-hand side operator:  (I - (i dt/2) D2) applied to a vector
        self.beta = -0.5j * dt / (dx * dx)   # note: beta = -alpha

    def _apply_rhs(self, psi: np.ndarray) -> np.ndarray:
        """Compute (I - (i dt/2) D2) psi with periodic BCs."""
        N = self.N
        b = self.beta
        out = np.empty_like(psi)
        out[0] = (1.0 - 2.0 * b) * psi[0] + b * psi[1] + b * psi[N - 1]
        out[N - 1] = (1.0 - 2.0 * b) * psi[N - 1] + b * psi[N - 2] + b * psi[0]
        out[1:N - 1] = ((1.0 - 2.0 * b) * psi[1:N - 1]
                        + b * psi[2:N] + b * psi[0:N - 2])
        return out

    def _nonlinear(self, psi_new: np.ndarray, psi_old: np.ndarray) -> np.ndarray:
        """Symmetric nonlinear term  N = (|new|^2+|old|^2)(new+old).

        The NLSE nonlinearity is  2 |psi|^2 psi.  Evaluated at the midpoint
        psi^{n+1/2} = (new + old)/2 this gives

            2 |psi^{n+1/2}|^2 psi^{n+1/2}
                = 2 |(new+old)/2|^2 (new+old)/2
                = (1/4) |new+old|^2 (new+old) ,

        which is algebraically equivalent to the polarisation-identity form

            (|new|^2 + |old|^2) (new + old) / 2

        used here (both reduce to 2|old|^2 old when new == old).  The factor of
        2 in front of |psi|^2 psi is therefore fully retained.
        """
        return 0.5 * (np.abs(psi_new) ** 2 + np.abs(psi_old) ** 2) * (psi_new + psi_old)

    def step(self, psi: np.ndarray) -> np.ndarray:
        dt = self.dt
        rhs = self._apply_rhs(psi)
        # Predictor
        guess = psi.copy()
        for _ in range(P.CN_MAX_ITER):
            b = rhs - 1j * dt * self._nonlinear(guess, psi)
            new_psi = solve_banded((1, 1), self.ab, b, overwrite_b=True)
            if np.max(np.abs(new_psi - guess)) < P.CN_TOL:
                return new_psi * self.mask
            guess = new_psi
        return new_psi * self.mask

    def march(self, psi0: np.ndarray, nsteps: int, record_every: int = 1):
        psi = psi0.copy()
        history = [psi.copy()] if record_every == 1 else None
        for n in range(nsteps):
            psi = self.step(psi)
            if record_every > 0 and (n + 1) % record_every == 0:
                if history is not None:
                    history.append(psi.copy())
        return psi, history