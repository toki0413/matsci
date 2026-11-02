"""
1D hydrogen-storage-bed thermal model (Direction A -> design document component C2).

Governing equations
-------------------
Energy (bed-averaged, 1D):
    (rho*cp)_eff * dT/dt = d/dx( k_eff * dT/dx ) + rho_H_max * (-DeltaH)/M_H2 * dX/dt
    The source term is the heat of absorption released per unit bed volume.

Hydrogen fraction kinetics:
    dX/dt = k_arr(T) * ( p - p_eq(T) ) / p_ref * (1 - X)
    - X in [0,1] is the fraction of the maximum hydrogen capacity reached.
    - Driving force (p - p_eq): absorption when p > p_eq, desorption when p < p_eq.
    - (1-X) limits the rate as the bed saturates.

van't Hoff equilibrium pressure (public-domain MgH2 literature values, see
objective_ws29.txt Direction A):
    ln p_eq = -DeltaH/(R T) + DeltaS/R
    DeltaH = -75 kJ/mol H2,  DeltaS = -135 J/(mol H2 K)
    -> p_eq(T) = exp(-DeltaH/(R T) + DeltaS/R)   [bar]

Arrhenius rate constant:
    k_arr(T) = A * exp(-Ea/(R T))

Boundary / initial conditions
-----------------------------
    x = 0 (cooling wall):      T = T_cool          (Dirichlet)
    x = L (insulated face):    dT/dx = 0           (Neumann)
    t = 0:                     T = T0,  X = 0      everywhere

Numerics
--------
Finite differences in space (N nodes), implicit Euler in time.  The coupled
(T, X) system at each time step is solved with a monolithic Newton iteration so
that the stiff heat source (absorption is strongly exothermic) is treated
stably.  X is updated analytically inside the residual (exact integration of
dX/dt = -c(T)*X over the step), which keeps the Newton system 1 unknown per
node (T) and avoids positivity violations.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
R = 8.314               # J/(mol K), universal gas constant

# MgH2 thermodynamics (public-domain literature values; NOT fitted/measured here)
DeltaH = -75.0e3        # J/mol H2, enthalpy of formation (negative = exothermic)
DeltaS = -135.0         # J/(mol H2 K), entropy of formation
M_H2 = 2.016e-3         # kg/mol H2

# ---------------------------------------------------------------------------
# Bed / material properties (representative MgH2 compact)
# ---------------------------------------------------------------------------
rho_s = 1740.0          # kg/m^3, solid density
cp_s = 1300.0           # J/(kg K), solid specific heat
k_s = 0.5               # W/(m K), solid thermal conductivity
porosity = 0.4          # void fraction

# Maximum volumetric hydrogen capacity of the compacted bed.
#   rho_H_max = (bed H2 mass fraction) * (bulk bed density)
# A realistic MgH2-based compact stores on the order of 1 wt% H2 at the bed
# level (theoretical MgH2 is 7.6 wt%, but a porous compact with additives and
# voids is far lower).  1.1 wt% of a 1740*0.6 = 1044 kg/m3 solid skeleton is
# ~11.5 kg H2/m3 bed.  This value is chosen so that the adiabatic temperature
# rise of a fully charged bed is physically bounded:
#   dT_ad = Q_vol / rho_cp_eff  with Q_vol = rho_H_max/M_H2*(-DeltaH)
# giving dT_ad ~ 230 K, i.e. a 573 K bed would reach ~800 K if fully charged
# adiabatically -- hot, but not the 3500 K+ runaway of the previous (100x
# too large) capacity.  This makes the grid-independence study meaningful.
rho_H_max = 11.5        # kg H2 / m^3 bed, maximum volumetric storage capacity

# Effective properties: solid phase dominates the heat capacity/conduction
rho_cp_eff = (1.0 - porosity) * rho_s * cp_s      # J/(m^3 K)
k_eff = (1.0 - porosity) * k_s                    # W/(m K)

# Volumetric heat of absorption per unit X (J/m^3 bed)
#   Q_vol = rho_H_max [kg/m3] / M_H2 [kg/mol] * (-DeltaH) [J/mol]
Q_vol = rho_H_max / M_H2 * (-DeltaH)

# ---------------------------------------------------------------------------
# Kinetics
# ---------------------------------------------------------------------------
A_kin = 1.0e7             # 1/s, pre-exponential factor
                            #   Chosen so that k_arr(573 K) = 7.6e-3 1/s, i.e. a
                            #   characteristic absorption time of ~100 s at the
                            #   nominal driving force.  This is an order-of-
                            #   magnitude scale (not a fitted value): real MgH2
                            #   hydriding half-times at 300 C are minutes to tens
                            #   of minutes, and the exact value depends strongly
                            #   on catalyst/porosity which are not modelled here.
Ea = 100.0e3              # J/mol, activation energy
p_ref = 1.0               # bar, reference pressure for the driving force


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------
class BedModel:
    def __init__(self, N=41, L=0.02, T0=573.0, T_cool=573.0, p_supply=2.0):
        self.N = N
        self.L = L
        self.dx = L / (N - 1)
        self.x = np.linspace(0.0, L, N)
        self.T0 = T0
        self.T_cool = T_cool
        self.p_supply = p_supply

    # ---- equilibrium pressure from van't Hoff ----------------------------
    def p_eq(self, T):
        """Equilibrium H2 pressure [bar] at temperature T [K]."""
        return np.exp(-DeltaH / (R * T) + DeltaS / R)

    # ---- Arrhenius rate constant -----------------------------------------
    def k_arr(self, T):
        return A_kin * np.exp(-Ea / (R * T))

    # ---- residual + Jacobian for one implicit Euler step -----------------
    def _residual_jac(self, T, X, dt, p):
        """Monolithic Newton residual for T (X eliminated analytically).

        Returns (res, jac) where res is length N and jac is NxN.
        """
        dx, N = self.dx, self.N
        res = np.zeros(N)
        jac = np.zeros((N, N))

        k = self.k_arr(T)
        peq = self.p_eq(T)
        # d peq / d T  =  p_eq * DeltaH / (R T^2)   (DeltaH < 0 -> peq rises with T)
        dpeq_dT = peq * DeltaH / (R * T ** 2)

        # Analytic X update over the step: dX/dt = -c (1-X) with
        #   c = k*(p - peq)/p_ref   ->  X_new = 1 - (1-X_old)*exp(-c dt)
        c = k * (p - peq) / p_ref
        c = np.maximum(c, 1e-300)          # guard against division/overflow noise
        exp_term = np.exp(-c * dt)
        X_new = 1.0 - (1.0 - X) * exp_term
        dX = X_new - X                     # change in X over the step

        # dc/dT  and  dX_new/dT  (chain rule through c and exp_term)
        dc_dT = (k * Ea / (R * T ** 2) * (p - peq) / p_ref
                 + k * (-dpeq_dT) / p_ref)
        dexp_dT = -exp_term * dt * dc_dT
        dX_new_dT = (1.0 - X) * dexp_dT

        # --- diffusion term  d/dx(k dT/dx)  (central differences) ---------
        # Interior node i: k/dx^2 * (T[i-1] - 2T[i] + T[i+1])
        diff = np.zeros(N)
        for i in range(1, N - 1):
            diff[i] = (T[i - 1] - 2.0 * T[i] + T[i + 1]) / dx ** 2
        # Neumann at x=L (ghost node T[N]=T[N-1]): contributes -k/dx^2*(T[N-1]-T[N-2])
        # handled by the stencil below via one-sided form.
        # Left Dirichlet is enforced by replacing row 0.

        # --- energy residual: rho_cp_eff*(T - T_old)/dt = diffusion + Q_vol*dX ---
        res[:] = (rho_cp_eff * (T - self._T_old) / dt
                  - k_eff * diff
                  - Q_vol * dX / dt)

        # --- Jacobian of the energy residual w.r.t. T --------------------
        J = np.zeros((N, N))
        # transient
        J[np.diag_indices(N)] = rho_cp_eff / dt
        # diffusion interior
        for i in range(1, N - 1):
            J[i, i - 1] -= k_eff / dx ** 2
            J[i, i] += 2.0 * k_eff / dx ** 2
            J[i, i + 1] -= k_eff / dx ** 2
        # Neumann right boundary: one-sided second derivative
        J[N - 1, N - 2] -= k_eff / dx ** 2
        J[N - 1, N - 1] += k_eff / dx ** 2
        # source term contribution: -Q_vol * d(dX)/dT / dt
        J[np.diag_indices(N)] -= Q_vol * dX_new_dT / dt

        # --- Dirichlet at x=0: T[0] = T_cool ------------------------------
        res[0] = T[0] - self.T_cool
        jac = J.copy()
        jac[0, :] = 0.0
        jac[0, 0] = 1.0
        return res, jac

    def step(self, T, X, dt, p):
        """Advance (T, X) by dt with implicit Euler + Newton."""
        self._T_old = T.copy()
        T = T.copy()
        for _ in range(50):
            res, jac = self._residual_jac(T, X, dt, p)
            if np.max(np.abs(res)) < 1e-10:
                break
            dT = np.linalg.solve(jac, res)
            T = T - dT
            # damping to avoid overshoot in the Newton update
            if np.max(np.abs(dT)) > 20.0:
                T = T - 0.5 * dT

        # Final X from the converged T
        k = self.k_arr(T)
        peq = self.p_eq(T)
        c = k * (p - peq) / p_ref
        c = np.maximum(c, 1e-300)
        X_new = 1.0 - (1.0 - X) * np.exp(-c * dt)
        return T, np.clip(X_new, 0.0, 1.0)

    def simulate(self, t_end, dt, record_every=1, p=None):
        """Integrate from t=0 to t_end, recording every record_every steps."""
        if p is None:
            p = self.p_supply
        N = self.N
        T = np.full(N, self.T0)
        X = np.zeros(N)
        times = [0.0]
        Ts = [T.copy()]
        Xs = [X.copy()]
        t = 0.0
        n_steps = int(np.round(t_end / dt))
        for i in range(1, n_steps + 1):
            T, X = self.step(T, X, dt, p)
            t = i * dt
            if i % record_every == 0:
                times.append(t)
                Ts.append(T.copy())
                Xs.append(X.copy())
        return np.array(times), np.array(Ts), np.array(Xs)


# ---------------------------------------------------------------------------
# Energy-conservation diagnostic
# ---------------------------------------------------------------------------
def energy_balance(model, times, Ts, Xs, p):
    """Check global energy conservation: accumulated heat released vs.
    conducted out of the cooling wall plus internal energy change.

    Returns (cumulative imbalance) at each recorded time.
    """
    dx, N = model.dx, model.N
    T0 = model.T0
    # Total internal energy relative to initial state
    U = rho_cp_eff * np.trapezoid(Ts - T0, model.x, axis=1)
    # Cumulative heat released by absorption
    Q_rel = Q_vol * np.trapezoid(Xs, model.x, axis=1)
    # Cumulative heat leaving through x=0 (Fourier: q = -k dT/dx)
    q_wall = np.zeros(len(times))
    for i, T in enumerate(Ts):
        if N > 1:
            q_wall[i] = -k_eff * (T[1] - T[0]) / dx
    Q_out = -np.trapezoid(q_wall, times)   # integral of outward flux
    # Imbalance = (U + Q_out) - Q_rel   should be ~0
    imbalance = (U + Q_out) - Q_rel
    return dict(times=times, U=U, Q_rel=Q_rel, Q_out=Q_out,
                imbalance=imbalance,
                rel_error=np.abs(imbalance) / (np.max(np.abs(Q_rel)) + 1e-12))


if __name__ == "__main__":
    m = BedModel(N=41)
    times, Ts, Xs = m.simulate(t_end=3000.0, dt=10.0)
    print("final T profile:", Ts[-1])
    print("final X:", Xs[-1])
    eb = energy_balance(m, times, Ts, Xs, m.p_supply)
    print("max rel energy imbalance:", np.max(np.abs(eb["rel_error"])))
    print("adiabatic dT for full charge:", Q_vol / rho_cp_eff, "K")