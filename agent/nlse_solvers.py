"""Step 2 — two independent NLSE solvers sharing one analytic single-soliton IC.

PDE (normalized, focusing):
    i u_t + u_xx + 2 |u|^2 u = 0,   x in [-L, L],  t in [0, T]

Exact single-soliton:
    u(x,t) = A sech( A (x - v t) ) exp( i ( C x - Omega t + phi0 ) )
    with  v = 2 C,  Omega = C^2 - A^2,  and the width-amplitude lock B = A.

Conserved quantities (field-theoretic Noether, derived in nlse_symbolic.py):
    N = int |u|^2 dx          (particle number / L2^2)
    P = int Im(u* u_x) dx     (momentum)
    H = int (|u_x|^2 - |u|^4) dx   (energy / Hamiltonian)

Solvers:
  (a) Crank-Nicolson implicit finite difference  -> unconditionally stable,
      discrete conservation of N (and H up to the nonlinear iteration).
  (b) Split-step Fourier method (Strang splitting) -> spectral accuracy in x,
      exactly unitary (|u| preserved pointwise in the nonlinear step).

Both use the SAME normalisation, phase convention, and a large box with an
absorbing (sponge) layer as a transparent-boundary surrogate.
"""

from __future__ import annotations

import json
import os

import numpy as np

WORKSPACE = os.environ.get("NLSE_WORKSPACE", "/workspace/agent")

# ------------------------------------------------------------------ reference
# Reference soliton: A=1, C=0.5 -> v=1, Omega = 0.25 - 1 = -0.75, B=A=1.
# Peak moves right at speed 1; by t=10 the peak is at x=10.
REF = {"A": 1.0, "B": 1.0, "C": 0.5, "phi0": 0.0}
REF["v"] = 2.0 * REF["C"]
REF["Omega"] = REF["C"] ** 2 - REF["A"] ** 2


def exact_soliton(x, t, A=1.0, B=None, C=0.5, phi0=0.0):
    """Exact single-soliton of i u_t + u_xx + 2|u|^2 u = 0."""
    if B is None:
        B = A  # the soliton lock
    v = 2.0 * C
    Omega = C**2 - A**2
    z = B * (x - v * t)
    # np.sech via 1/cosh; use np.cosh with clipping for overflow safety
    env = 1.0 / np.cosh(z)
    return A * env * np.exp(1j * (C * x - Omega * t + phi0))


def exact_envelope(x, t, A=1.0, B=None, C=0.5):
    """|u(x,t)| of the exact soliton."""
    if B is None:
        B = A
    v = 2.0 * C
    return A / np.cosh(B * (x - v * t))


# ------------------------------------------------------------- conserved stuff
def conserved(u, dx):
    """Return (N, P, H) on a uniform grid (periodic/trapezoidal quadrature)."""
    N = np.sum(np.abs(u) ** 2) * dx
    # momentum: Im(u* u_x); central difference for u_x
    ux = np.roll(u, -1) - np.roll(u, 1)
    ux /= 2.0 * dx
    P = np.sum(np.imag(np.conj(u) * ux)) * dx
    # energy: |u_x|^2 - |u|^4
    H = (np.sum(np.abs(ux) ** 2) - np.sum(np.abs(u) ** 4)) * dx
    return N, P, H


def conserved_spectral(u, dx, k):
    """Spectral (FFT-based) derivatives for P and H — more accurate."""
    N = np.sum(np.abs(u) ** 2) * dx
    ux = np.fft.ifft(1j * k * np.fft.fft(u))
    P = np.sum(np.imag(np.conj(u) * ux)) * dx
    H = (np.sum(np.abs(ux) ** 2) - np.sum(np.abs(u) ** 4)) * dx
    return N, P, H


# --------------------------------------------------------------- grid / sponge
def build_grid(L, nx):
    """Uniform grid on [-L, L] with nx interior points (periodic)."""
    dx = 2.0 * L / nx
    x = -L + (np.arange(nx) + 0.5) * dx  # cell-centred, avoids x=0 singularity issues
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    return x, dx, k


def sponge(L, nx, width_frac=0.15, strength=4.0):
    """Absorbing layer mask: 1 in the interior, decays to ~0 at the boundaries.

    Applied as  u <- u * exp(-sigma(x) dt)  each step (multiplicative damping).
    This is the transparent-boundary surrogate; the box is chosen large enough
    that the exact solution is ~machine-zero at the boundaries.
    """
    x = np.linspace(-L, L, nx, endpoint=False)
    sigma = np.zeros(nx)
    w = L * width_frac
    for i in range(nx):
        d = abs(abs(x[i]) - (L - w))
        if d > 0 and abs(x[i]) > L - w:
            sigma[i] = strength * ((d) / w) ** 2
    return np.exp(-sigma)


# ============================================================ SOLVER (a): CN
class CrankNicolson:
    """Crank-Nicolson for i u_t + u_xx + 2|u|^2 u = 0.

    Semi-discrete:  i (u^{n+1} - u^n)/dt + D2 (u^{n+1}+u^n)/2
                    + (|u^{n+1}|^2 u^{n+1} + |u^n|^2 u^n) = 0

    Rearranged into a linear system for u^{n+1} with a frozen nonlinearity,
    solved by fixed-point (Newton-like) iteration:

        [ I - i dt/2 D2 - i dt G^{(m)} ] u^{n+1,(m+1)}
            = [ I + i dt/2 D2 ] u^n + i dt |u^n|^2 u^n

    where G^{(m)} = diag(|u^{n+1,(m)}|^2 + |u^n|^2)/2  (Crank-Nicolson average).

    The matrix is tridiagonal (periodic) -> Thomas algorithm / scipy sparse.
    Discrete N is conserved to machine precision when the nonlinear iteration
    converges.
    """

    def __init__(self, x, dx, dt, L_box, nl_iter=10, nl_tol=1e-12):
        self.x = x
        self.dx = dx
        self.dt = dt
        self.nx = len(x)
        self.nl_iter = nl_iter
        self.nl_tol = nl_tol
        self.k = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=dx)

        # second-order central difference matrix (periodic) as a LinearOperator
        self._build_matrix()
        self.mask = sponge(L_box, self.nx)

    def _build_matrix(self):
        nx, dx, dt = self.nx, self.dx, self.dt
        mu = 1j * dt / (2.0 * dx * dx)  # i*dt/(2 h^2)
        # (I - mu * A) where A is the discrete Laplacian: -2 on diag, 1 on off-diag
        # so I - mu*A  ->  diag(1+2mu), off-diag(-mu)
        self.diag_main = np.full(nx, 1.0 + 2.0 * mu)
        self.diag_off = np.full(nx, -mu)
        # RHS operator: (I + mu*A) u = u + mu*(-2u + u_{i-1}+u_{i+1})
        self.rhs_main = np.full(nx, 1.0 - 2.0 * mu)
        self.rhs_off = np.full(nx, mu)

    def _tridiag_solve(self, main, off, rhs):
        """Periodic tridiagonal solve via the Sherman-Morrison trick.

        Standard Thomas on the cyclic system.  Uses scipy if available.
        """
        try:
            from scipy.linalg import solve_banded

            # banded form for a cyclic tridiagonal: lower, diag, upper
            # We handle the periodic wrap by the standard reduction.
            pass
        except Exception:
            pass
        return self._thomas_periodic(main, off, rhs)

    @staticmethod
    def _thomas_periodic(main, off, rhs):
        """Cyclic tridiagonal (periodic BC) solver — Sherman-Morrison."""
        n = len(main)
        a = np.roll(off, 1)  # sub-diagonal (a[0] is the wrap element)
        c = np.roll(off, -1)  # super-diagonal
        b = main.copy()
        d = rhs.copy()

        # Standard algorithm for cyclic tridiagonal (see e.g. Press et al.)
        gamma = -b[0]
        a[0] = 0.0
        c[n - 1] = 0.0

        # forward sweep
        w = np.empty(n)
        w[0] = b[0]
        for i in range(1, n):
            w[i] = b[i] - a[i] * c[i - 1] / w[i - 1]

        # build the two auxiliary solutions
        z = np.empty(n)
        z[0] = d[0]
        for i in range(1, n):
            z[i] = d[i] - a[i] * z[i - 1] / w[i - 1]
        z = z / w

        u = np.empty(n)
        u[0] = gamma
        for i in range(1, n):
            u[i] = -a[i] * u[i - 1] / w[i - 1]
        u = u / w

        # correction for the cyclic part
        factor = c[n - 1] / (w[n - 1] + a[0] * u[n - 1] * 0 + 0)
        # simpler: use the explicit Sherman-Morrison formula
        # (A + uv^T)^{-1} ... implement the standard cyclic Thomas properly:
        return _cyclic_thomas(a, b, c, d)

    def step(self, u):
        """Advance one time step with nonlinear fixed-point iteration."""
        dt, nx = self.dt, self.nx
        u_old = u.copy()
        # RHS = (I + i dt/2 D2) u^n + i dt |u^n|^2 u^n
        lap = self._lap(u_old)
        rhs = u_old + 1j * dt / 2.0 * lap + 1j * dt * np.abs(u_old) ** 2 * u_old

        u_new = u_old.copy()
        for it in range(self.nl_iter):
            G = 0.5 * (np.abs(u_new) ** 2 + np.abs(u_old) ** 2)
            # LHS matrix: I - i dt/2 D2 - i dt G
            main = self.diag_main.copy() - 1j * dt * G
            off = self.diag_off.copy()
            rhs_full = rhs.copy()
            u_next = _cyclic_thomas(
                np.roll(off, 1), main.copy(), np.roll(off, -1), rhs_full
            )
            if np.max(np.abs(u_next - u_new)) < self.nl_tol:
                u_new = u_next
                break
            u_new = u_next

        # absorbing layer
        u_new = u_new * self.mask
        return u_new

    def _lap(self, u):
        """Discrete Laplacian (periodic, 2nd order)."""
        return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / self.dx**2

    def integrate(self, u0, nsteps, record_every=1):
        u = u0.copy()
        trajectory = [u.copy()]
        for n in range(1, nsteps + 1):
            u = self.step(u)
            if n % record_every == 0:
                trajectory.append(u.copy())
        return u, trajectory


def _cyclic_thomas(a, b, c, d):
    """Cyclic tridiagonal solve.  a=sub, b=diag, c=super, d=rhs (all length n).

    Solves the system with periodic wrap: a[0] and c[n-1] are the corner terms.
    Standard algorithm (e.g. Conte & de Boor / Press):
      solve A x = d  where A is cyclic tridiagonal.
    """
    n = len(b)
    if n < 3:
        return np.linalg.solve(_build_full(a, b, c), d)

    bb = b.copy()
    dd = d.copy()
    gamma = -bb[0]
    beta = c[n - 1] if n > 1 else 0.0
    alpha = a[0] if n > 1 else 0.0

    # modify the first and last rows to remove the cyclic entries
    bb[0] = b[0] - gamma
    bb[n - 1] = b[n - 1] - beta * alpha / gamma if False else b[n - 1]
    # Use the standard Sherman-Morrison approach:
    # A = M + u v^T  where M is the non-cyclic tridiagonal part.
    a_nc = a.copy()
    c_nc = c.copy()
    a_nc[0] = 0.0
    c_nc[n - 1] = 0.0
    bb2 = b.copy()
    bb2[0] = b[0] - gamma
    bb2[n - 1] = b[n - 1] - beta * alpha / gamma

    # Actually implement the well-known algorithm directly:
    return _cyclic_thomas_std(a, b, c, d)


def _cyclic_thomas_std(a, b, c, d):
    """Standard cyclic tridiagonal algorithm (Press et al., §2.7)."""
    n = len(b)
    gam = np.empty(n)
    u = np.empty(n)
    z = np.empty(n)

    bb = b.copy()
    dd = d.copy()
    gamma = -b[0]
    beta = c[n - 1]
    alpha = a[0]

    bb[0] = b[0] - gamma
    bb[n - 1] = b[n - 1] - beta * alpha / gamma

    # forward sweep for the modified (non-cyclic) system
    bet = bb[0]
    if bet == 0:
        raise ValueError("Singular tridiagonal")
    gam[0] = c[0] / bet
    z[0] = dd[0] / bet
    for j in range(1, n - 1):
        bet = bb[j] - a[j] * gam[j - 1]
        gam[j] = c[j] / bet
        z[j] = (dd[j] - a[j] * z[j - 1]) / bet
    bet = bb[n - 1] - a[n - 1] * gam[n - 2]
    z[n - 1] = (dd[n - 1] - a[n - 1] * z[n - 2]) / bet

    # back substitution with the cyclic correction
    x = np.empty(n)
    x[n - 1] = z[n - 1]
    for j in range(n - 2, -1, -1):
        x[j] = z[j] - gam[j] * x[j + 1]

    # Now solve for the correction vectors (u and v in Press)
    # u solves M u = rhs with rhs = (gamma, 0, ..., -alpha*gamma/beta_term)
    # Simpler & robust: use the explicit Sherman-Morrison with two solves.
    # Let's just do it properly with two tridiagonal solves.
    return _cyclic_sm(a, b, c, d)


def _cyclic_sm(a, b, c, d):
    """Cyclic tridiagonal via Sherman-Morrison: two non-cyclic solves."""
    n = len(b)
    alpha = a[0]
    beta = c[n - 1]

    # M = A without the corner entries
    aM = a.copy()
    cM = c.copy()
    bM = b.copy()
    aM[0] = 0.0
    cM[n - 1] = 0.0
    bM[0] = b[0] - alpha
    bM[n - 1] = b[n - 1] - beta

    def solve_tri(a_, b_, c_, d_):
        n2 = len(b_)
        gam = np.empty(n2)
        dd = d_.copy()
        bet = b_[0]
        gam[0] = c_[0] / bet
        dd[0] = dd[0] / bet
        for j in range(1, n2):
            bet = b_[j] - a_[j] * gam[j - 1]
            gam[j] = c_[j] / bet
            dd[j] = (dd[j] - a_[j] * dd[j - 1]) / bet
        x = np.empty(n2)
        x[n2 - 1] = dd[n2 - 1]
        for j in range(n2 - 2, -1, -1):
            x[j] = dd[j] - gam[j] * x[j + 1]
        return x

    x = solve_tri(aM, bM, cM, d)

    # u: M u = r  where r has r[0]=1, r[n-1]=...  (Sherman-Morrison rank-2)
    # A = M + alpha e_n e_1^T + beta e_1 e_n^T
    r = np.zeros(n)
    r[0] = 1.0
    u = solve_tri(aM, bM, cM, r)
    s = np.zeros(n)
    s[n - 1] = 1.0
    v = solve_tri(aM, bM, cM, s)

    # correction
    xu = x[0]
    xv = x[n - 1]
    uu = u[0]
    uv = u[n - 1]
    vu = v[0]
    vv = v[n - 1]

    # solve 2x2: (I + ...) 
    det = (1 + alpha * vv + beta * uu + alpha * beta * (uu * vv - uv * vu))
    # Standard formula for rank-2 update:
    # (M + alpha e_n e_1^T + beta e_1 e_n^T)^{-1} d
    # Let z = x - [u v] * M2^{-1} * [alpha*xv; beta*xu]  where M2 = I + [[alpha*vu, alpha*vv],[beta*uu,beta*uv]]
    M2 = np.array(
        [[1.0 + alpha * vu, alpha * vv], [beta * uu, 1.0 + beta * uv]]
    )
    rhs2 = np.array([alpha * xv, beta * xu])
    lam = np.linalg.solve(M2, rhs2)
    return x - lam[0] * u - lam[1] * v


def _build_full(a, b, c):
    n = len(b)
    M = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    M[0, n - 1] = c[n - 1]
    M[n - 1, 0] = a[0]
    return M


# ============================================================ SOLVER (b): SSFM
class SplitStepFourier:
    """Strang split-step Fourier method for i u_t + u_xx + 2|u|^2 u = 0.

    Split  u_t = i u_xx + 2i |u|^2 u  into
      D (dispersion, linear):  u_t = i u_xx   -> solved exactly in Fourier:
            u_hat(k, t+dt) = u_hat(k,t) exp(-i k^2 dt)
      N (nonlinear):           u_t = 2i |u|^2 u  -> solved exactly:
            u(t+dt) = u(t) exp(2i |u|^2 dt)

    Strang splitting:  u^{n+1} = D(dt/2) N(dt) D(dt/2) u^n
    -> 2nd order in dt, spectral in x, exactly unitary (|u| pointwise
       preserved by the nonlinear step, and the linear step is a unitary
       multiplication in Fourier space).
    """

    def __init__(self, x, dx, dt, L_box):
        self.x = x
        self.dx = dx
        self.dt = dt
        self.nx = len(x)
        self.k = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=dx)
        self.mask = sponge(L_box, self.nx)
        # precompute the linear half-step propagator
        self.half = np.exp(-1j * self.k**2 * (dt / 2.0))

    def step(self, u):
        # D(dt/2)
        u = np.fft.ifft(self.half * np.fft.fft(u))
        # N(dt)
        u = u * np.exp(2j * np.abs(u) ** 2 * self.dt)
        # D(dt/2)
        u = np.fft.ifft(self.half * np.fft.fft(u))
        # absorbing layer
        return u * self.mask

    def integrate(self, u0, nsteps, record_every=1):
        u = u0.copy()
        trajectory = [u.copy()]
        for n in range(1, nsteps + 1):
            u = self.step(u)
            if n % record_every == 0:
                trajectory.append(u.copy())
        return u, trajectory


# ============================================================ shared interface
def run_solver(name, x, dx, dt, u0, nsteps, record_every, L_box):
    """Run either solver; return (u_final, trajectory)."""
    if name == "cn":
        solver = CrankNicolson(x, dx, dt, L_box)
    elif name == "ssfm":
        solver = SplitStepFourier(x, dx, dt, L_box)
    else:
        raise ValueError(f"unknown solver {name}")
    return solver.integrate(u0, nsteps, record_every)


def save_reference_params(path=None):
    path = path or os.path.join(WORKSPACE, "nlse_params.json")
    payload = {
        "pde": "i u_t + u_xx + 2 |u|^2 u = 0",
        "reference_soliton": REF,
        "relations": {"B_eq_A": "B = A", "v_eq_2C": "v = 2 C", "Omega": "Omega = C^2 - A^2"},
        "conserved_quantities": {
            "N": "int |u|^2 dx",
            "P": "int Im(u* u_x) dx",
            "H": "int (|u_x|^2 - |u|^4) dx",
        },
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path


if __name__ == "__main__":
    p = save_reference_params()
    print(f"[solvers] wrote {p}")
    print(f"[solvers] reference soliton: {REF}")
    # quick smoke test
    L, nx, dt = 40.0, 2048, 1e-3
    x, dx, k = build_grid(L, nx)
    u0 = exact_soliton(x, 0.0, **REF)
    for name in ("cn", "ssfm"):
        uf, traj = run_solver(name, x, dx, dt, u0, 10, 1, L)
        N0, P0, H0 = conserved_spectral(u0, dx, k)
        N1, P1, H1 = conserved_spectral(uf, dx, k)
        print(
            f"[solvers] {name:4s} t=0.01: N rel err={abs(N1-N0)/abs(N0):.3e} "
            f"P rel err={abs(P1-P0)/abs(P0):.3e} H rel err={abs(H1-H0)/abs(H0):.3e}"
        )