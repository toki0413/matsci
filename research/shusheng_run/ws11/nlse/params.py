"""Physical and numerical parameters for the 1-D NLSE soliton verification pipeline.

Governing equation (focusing NLSE, standard normalization)

        i psi_t + psi_xx + 2 |psi|^2 psi = 0 ,   x in [-L, L], t in [0, T]

Exact travelling-wave (ground-state soliton) solution

        psi(x,t) = A sech[A(x - v t)] exp{ i [ v x/2 + (A^2 - v^2/4) t ] }

Parameter constraints (derived from the exact solution):
  * A > 0                       (amplitude / inverse width)
  * v in R                      (group velocity of the envelope)
  * The ansatz solves the NLSE identically for ANY A>0, v in R.
  * Energy E = A (v^2/2 - 2 A^2/3) is POSITIVE only when |v| > 2 A / sqrt(3).
  * Momentum P = A v is non-zero only when v != 0.
  * We therefore require  |v| > 2A/sqrt(3)  and  v != 0  so that both the energy
    and momentum conservation checks are well-posed and non-trivial.

Default working point:  A = 1, v = 2  ->  E = 4/3, P = 2.
The soliton centre at t = T is at x = v T = 20, so a half-width L = 32 leaves a
distance d = 12 to the boundary; the evanescent tail there is ~exp(-12) = 6e-6,
which sets the height of the boundary-reflection plateau (see report).
"""

# --------------------------------------------------------------------------- #
# Physical parameters
# --------------------------------------------------------------------------- #
A = 1.0          # soliton amplitude (inverse width)
V = 2.0          # soliton group velocity
L = 32.0         # computational half-width  -> domain [-L, L]
T_END = 10.0     # final integration time

# Derived physical invariants (exact, on the whole real line)
E0_EXACT = A * (V ** 2 / 2.0 - 2.0 * A ** 2 / 3.0)   # = A(v^2/2 - 2A^2/3)
P0_EXACT = A * V                                      # = A v

assert E0_EXACT > 0, "E0 must be > 0 for a meaningful relative energy drift"
assert P0_EXACT != 0, "P0 must be != 0 for a meaningful relative momentum drift"

# Times at which the L2 error is sampled
T_CHECK = [0.0, 1.0, 5.0, 10.0]

# --------------------------------------------------------------------------- #
# Convergence / refinement study parameters
# --------------------------------------------------------------------------- #
# Number of joint (dx, dt) refinement levels.
N_REFINE = 10

# Base grid: N0 points over the FULL domain [-L, L] (periodic FFT grid).
# dx0 = 2L / N0.  Each level halves dx and halves dt.
N0 = 256

# Base time step.  Chosen so that the base-level SSFM error is comfortably
# above the boundary plateau but the finest level is well below it.
DT0 = 1e-3

# Scaling rule:  dx -> dx/2,  dt -> dt/2  at each refinement level.
#   dx_ell = dx0 * 2^{-ell},   dt_ell = dt0 * 2^{-ell},   ell = 0..N_REFINE-1
# With dt ~ dx the SSFM (Strang) splitting error O(dt^2) dominates the
# exponentially-small spatial (spectral) error, so the observed convergence
# order in dt is 2.  For the Crank-Nicolson scheme the error is O(dt^2 + dx^2);
# with dt ~ dx both contributions scale like 2^{-2ell}, again giving order 2.
#
# A stricter rule dt ~ dx^2 would make the CN spatial error O(dx^2) dominate
# (temporal O(dt^2)=O(dx^4)) and still yield order 2, but at a prohibitive cost
# for 10 levels (dt_9 ~ 1e-12).  The dt ~ dx rule is the practical choice that
# keeps the 10-level study tractable while preserving a clean O(h^2) law.

# --------------------------------------------------------------------------- #
# Absorbing / boundary treatment
# --------------------------------------------------------------------------- #
# The domain is periodic for the FFT-based SSFM.  A physical absorbing layer is
# implemented as a smooth mask applied every step (see operators.py).  Its
# purpose is NOT to remove the soliton (which never reaches the boundary) but to
# damp the tiny dispersive radiation that would otherwise reflect and pollute
# the L2 error, thereby allowing the plateau to be attributed unambiguously to
# the *exact* solution's truncated tail rather than to numerical artefacts.
SIGMA_WIDTH = 4.0      # width of the absorbing mask (in units of dx, see impl.)
SIGMA_AMP = 0.15       # absorption strength per step (tuned, see report)

# --------------------------------------------------------------------------- #
# Tolerances to be certified at the finest level
# --------------------------------------------------------------------------- #
TOL_L2 = 1e-6      # L2 relative error
TOL_E = 1e-8       # |dE / E0|
TOL_P = 1e-7       # |dP / P0|

# --------------------------------------------------------------------------- #
# Crank-Nicolson non-linear solver settings
# --------------------------------------------------------------------------- #
CN_MAX_ITER = 20      # fixed-point iterations per time step
CN_TOL = 1e-14        # fixed-point convergence tolerance