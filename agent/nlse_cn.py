"""Crank-Nicolson solver for the 1-D focusing NLSE.

  i u_t + u_xx + 2 |u|^2 u = 0

Discretisation (periodic grid, spacing h):
  i (u^{n+1} - u^n)/dt + (D2 u^{n+1} + D2 u^n)/2 + (|u^{n+1}|^2 u^{n+1} + |u^n|^2 u^n) = 0

Rearranged:
  [ I - i dt/2 D2 - i dt G^{(m)} ] u^{n+1,(m+1)}
      = [ I + i dt/2 D2 ] u^n + i dt |u^n|^2 u^n

with G^{(m)} = diag( (|u^{n+1,(m)}|^2 + |u^n|^2)/2 ).
The matrix is cyclic-tridiagonal (periodic) -> solved with scipy.linalg.solve_banded
after a Sherman-Morrison rank-2 correction for the periodic wrap.
"""

from __future__ import annotations

import numpy as np


def cyclic_thomas(a, b, c, d):
    """Solve a cyclic tridiagonal system  A x = d.

    a = sub-diagonal; a[0] is the corner element A[n-1, 0]
    b = main diagonal
    c = super-diagonal; c[n-1] is the corner element A[0, n-1]
    d = right-hand side

    Decomposition (Sherman-Morrison-Woodbury, rank-2):
        A = M + beta e_0 e_{n-1}^T + alpha e_{n-1} e_0^T
        with alpha = a[0], beta = c[n-1], and M the same tridiagonal matrix
        but with the two corners zeroed (a[0]=c[n-1]=0).
    Requires three non-cyclic tridiagonal solves.
    """
    n = len(b)
    alpha = a[0]  # A[n-1, 0]
    beta = c[n - 1]  # A[0, n-1]

    aM = a.copy()
    cM = c.copy()
    aM[0] = 0.0
    cM[n - 1] = 0.0

    x = tri_solve(aM, b, cM, d)  # M^{-1} d

    e0 = np.zeros(n, dtype=complex)
    e0[0] = 1.0
    en = np.zeros(n, dtype=complex)
    en[n - 1] = 1.0
    m_inv_e0 = tri_solve(aM, b, cM, e0)
    m_inv_en = tri_solve(aM, b, cM, en)

    # M^{-1} U,  U = [beta e_0, alpha e_{n-1}]
    Uinv = np.column_stack([beta * m_inv_e0, alpha * m_inv_en])

    # V^T M^{-1} d,  V = [e_{n-1}, e_0]
    w = np.array([x[n - 1], x[0]], dtype=complex)

    # V^T M^{-1} U  (2x2)
    K = np.array(
        [
            [beta * m_inv_e0[n - 1], alpha * m_inv_en[n - 1]],
            [beta * m_inv_e0[0], alpha * m_inv_en[0]],
        ],
        dtype=complex,
    )
    S = np.eye(2, dtype=complex) + K
    lam = np.linalg.solve(S, w)
    return x - Uinv @ lam


class CrankNicolson:
    def __init__(self, x, dx, dt, L_box, nl_iter=12, nl_tol=1e-13, sponge_w=0.15, sponge_s=4.0):
        self.x = x
        self.dx = dx
        self.dt = dt
        self.nx = len(x)
        self.nl_iter = nl_iter
        self.nl_tol = nl_tol
        h2 = dx * dx
        mu = 1j * dt / (2.0 * h2)  # i*dt/(2 h^2)
        # LHS: (I - mu*A)  with A = discrete Laplacian (-2 on diag, +1 off)
        #   -> main = 1 + 2 mu, off = -mu
        self._lhs_main_const = np.full(self.nx, 1.0 + 2.0 * mu)
        self._lhs_off_const = np.full(self.nx, -mu)
        # RHS: (I + mu*A) u^n  -> main = 1 - 2 mu, off = mu
        self._rhs_main = np.full(self.nx, 1.0 - 2.0 * mu)
        self._rhs_off = np.full(self.nx, mu)
        self.mask = _sponge(L_box, self.nx, sponge_w, sponge_s)

    def _lap(self, u):
        return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / self.dx**2

    def step(self, u):
        dt, nx = self.dt, self.nx
        u_old = u
        lap = self._lap(u_old)
        rhs = u_old + 1j * dt / 2.0 * lap + 1j * dt * np.abs(u_old) ** 2 * u_old

        u_new = u_old.copy()
        for it in range(self.nl_iter):
            G = 0.5 * (np.abs(u_new) ** 2 + np.abs(u_old) ** 2)
            main = self._lhs_main_const.copy() - 1j * dt * G
            off = self._lhs_off_const.copy()
            a = np.roll(off, 1)   # sub-diagonal (corner at a[0])
            c = np.roll(off, -1)  # super-diagonal (corner at c[n-1])
            u_next = cyclic_thomas(a, main, c, rhs)
            if np.max(np.abs(u_next - u_new)) < self.nl_tol:
                u_new = u_next
                break
            u_new = u_next
        return u_new * self.mask

    def integrate(self, u0, nsteps, record_every=1):
        u = u0.copy()
        traj = [u.copy()]
        for n in range(1, nsteps + 1):
            u = self.step(u)
            if n % record_every == 0:
                traj.append(u.copy())
        return u, traj


def _sponge(L, nx, width_frac, strength):
    x = np.linspace(-L, L, nx, endpoint=False)
    sigma = np.zeros(nx)
    w = L * width_frac
    for i in range(nx):
        r = abs(x[i]) - (L - w)
        if r > 0:
            sigma[i] = strength * (r / w) ** 2
    return np.exp(-sigma)


if __name__ == "__main__":
    # sanity check of the cyclic solver against a dense solve
    rng = np.random.default_rng(0)
    for n in (8, 16, 64, 256):
        a = rng.standard_normal(n) + 0j
        b = rng.standard_normal(n) + 5 + 0j
        c = rng.standard_normal(n) + 0j
        d = rng.standard_normal(n) + 0j
        M = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
        M[0, n - 1] = c[n - 1]
        M[n - 1, 0] = a[0]
        x1 = cyclic_thomas(a, b, c, d)
        x2 = np.linalg.solve(M, d)
        print(f"n={n:4d}  max err = {np.max(np.abs(x1 - x2)):.3e}")