#!/usr/bin/env python3
"""
CO Oxidation on Pt — LH Kinetic Parameter Sensitivity Analysis
Self-contained script using only numpy and matplotlib (Agg backend).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ──────────────────────────────────────────────
# 0. Physical constants & fixed parameters
# ──────────────────────────────────────────────
R = 8.314                     # J/mol/K
P_CO = 0.01                   # bar
P_O2 = 0.05                   # bar

# Known / fixed from previous fit
Ea_true    = 85000.0          # J/mol
K_CO0_true = 0.03
K_O2_0_true = 0.008
dH_CO      = -140000.0        # J/mol  (fixed)
dH_O2      = -150000.0        # J/mol  (fixed)

# Data
T_C = np.array([200, 250, 300, 350, 400, 450], dtype=float)   # °C
X   = np.array([5, 12, 28, 95, 31, 15], dtype=float) / 100.0  # conversion fraction
T   = T_C + 273.15            # K

# ──────────────────────────────────────────────
# 1. LH rate model
# ──────────────────────────────────────────────
def lh_rate(Tk, Ea, K_CO0_val, K_O2_0_val):
    """Return reaction rate r(T) in s⁻¹ (pre-exponential absorbed into k0)."""
    k   = np.exp(-Ea / (R * Tk))
    KCO = K_CO0_val * np.exp(-dH_CO / (R * Tk))
    KO2 = K_O2_0_val * np.exp(-dH_O2 / (R * Tk))
    sqrt_term = np.sqrt(KO2 * P_O2)
    denom = (KCO * P_CO + sqrt_term + 1.0) ** 2
    r = k * sqrt_term * P_CO / denom
    return r

def pfr_conversion(Tk, Ea, K_CO0_val, K_O2_0_val, tau):
    """PFR model: X = 1 - exp(-r * tau)"""
    r = lh_rate(Tk, Ea, K_CO0_val, K_O2_0_val)
    return 1.0 - np.exp(-r * tau)

# Determine tau so that X(350°C) = 0.95
T350 = 350.0 + 273.15
r350 = lh_rate(T350, Ea_true, K_CO0_true, K_O2_0_true)
tau = -np.log(1.0 - 0.95) / r350
print(f"tau = {tau:.4f} s  (so that X(350°C) = 0.95)")
print()

# ──────────────────────────────────────────────
# 2. Loss function for fitting
# ──────────────────────────────────────────────
def loss_func(params, T_vals, X_obs):
    """params = [Ea, K_CO0, K_O2_0]"""
    Ea_v, K0_v, KO0_v = params
    X_pred = pfr_conversion(T_vals, Ea_v, K0_v, KO0_v, tau)
    res = X_pred - X_obs
    return np.sum(res ** 2)

def fit_params(T_vals, X_obs, initial_guess=None):
    """Simple Nelder-Mead fit (no scipy needed)."""
    if initial_guess is None:
        guess = np.array([Ea_true, K_CO0_true, K_O2_0_true])
    else:
        guess = np.array(initial_guess, dtype=float)

    # Nelder-Mead simplex
    n = len(guess)
    simplex = np.zeros((n + 1, n))
    simplex[0] = guess
    for i in range(n):
        delta = np.zeros(n)
        delta[i] = 0.05 * abs(guess[i]) + 1e-6
        simplex[i + 1] = guess + delta

    f_simplex = np.array([loss_func(s, T_vals, X_obs) for s in simplex])

    max_iter = 5000
    tol = 1e-10
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho   = 0.5   # contraction
    sigma = 0.5   # shrink

    for iteration in range(max_iter):
        # Order
        idx = np.argsort(f_simplex)
        simplex = simplex[idx]
        f_simplex = f_simplex[idx]

        if np.std(f_simplex) < tol * (1.0 + np.mean(np.abs(f_simplex))):
            break

        x_bar = np.mean(simplex[:-1], axis=0)

        # Reflection
        x_r = x_bar + alpha * (x_bar - simplex[-1])
        f_r = loss_func(x_r, T_vals, X_obs)

        if f_simplex[0] <= f_r < f_simplex[-2]:
            simplex[-1] = x_r
            f_simplex[-1] = f_r
        elif f_r < f_simplex[0]:
            # Expansion
            x_e = x_bar + gamma * (x_r - x_bar)
            f_e = loss_func(x_e, T_vals, X_obs)
            if f_e < f_r:
                simplex[-1] = x_e
                f_simplex[-1] = f_e
            else:
                simplex[-1] = x_r
                f_simplex[-1] = f_r
        else:
            # Contraction
            x_c = x_bar + rho * (simplex[-1] - x_bar)
            f_c = loss_func(x_c, T_vals, X_obs)
            if f_c < f_simplex[-1]:
                simplex[-1] = x_c
                f_simplex[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_simplex[i] = loss_func(simplex[i], T_vals, X_obs)

    best_idx = np.argmin(f_simplex)
    return simplex[best_idx], f_simplex[best_idx]

# ──────────────────────────────────────────────
# 3. Task 1: Noise sensitivity (Monte Carlo)
# ──────────────────────────────────────────────
print("=" * 70)
print("TASK 1: Noise Sensitivity — Monte Carlo (1000 trials, ±5% noise on 400°C & 450°C)")
print("=" * 70)

np.random.seed(42)
n_mc = 1000
noise_level = 0.05  # 5%

# Indices of the noisy points
noise_idx = [4, 5]  # 400°C (index 4), 450°C (index 5)

fitted_Ea   = np.zeros(n_mc)
fitted_KCO0 = np.zeros(n_mc)
fitted_KO20 = np.zeros(n_mc)

for i in range(n_mc):
    X_noisy = X.copy()
    for j in noise_idx:
        X_noisy[j] = X[j] * (1.0 + noise_level * np.random.randn())
        # Clip to [0, 1]
        X_noisy[j] = np.clip(X_noisy[j], 0.0, 1.0)

    p, _ = fit_params(T, X_noisy)
    fitted_Ea[i]   = p[0]
    fitted_KCO0[i] = p[1]
    fitted_KO20[i] = p[2]

mean_Ea   = np.mean(fitted_Ea)
std_Ea    = np.std(fitted_Ea)
mean_KCO0 = np.mean(fitted_KCO0)
std_KCO0  = np.std(fitted_KCO0)
mean_KO20 = np.mean(fitted_KO20)
std_KO20  = np.std(fitted_KO20)

print(f"  Ea:     mean = {mean_Ea:.1f} J/mol,  std = {std_Ea:.1f} J/mol  (true = {Ea_true})")
print(f"  K_CO0:  mean = {mean_KCO0:.6f},      std = {std_KCO0:.6f}  (true = {K_CO0_true})")
print(f"  K_O2_0: mean = {mean_KO20:.6f},      std = {std_KO20:.6f}  (true = {K_O2_0_true})")
print()

# ──────────────────────────────────────────────
# 4. Task 2: Data point requirement
# ──────────────────────────────────────────────
print("=" * 70)
print("TASK 2: Data Point Requirement — Ea std vs N (500 trials each)")
print("=" * 70)

np.random.seed(123)
n_trials = 500
Ns = [3, 4, 5, 6]
Ea_std_vs_N = []

for N in Ns:
    Ea_fits_N = np.zeros(n_trials)
    for i in range(n_trials):
        idx = np.random.choice(6, N, replace=False)
        T_sub = T[idx]
        X_sub = X[idx]
        p, _ = fit_params(T_sub, X_sub)
        Ea_fits_N[i] = p[0]
    std_val = np.std(Ea_fits_N)
    Ea_std_vs_N.append(std_val)
    print(f"  N = {N}:  Ea std = {std_val:.1f} J/mol")

print()

# ──────────────────────────────────────────────
# 5. Task 3: Optimal temperature placement
# ──────────────────────────────────────────────
print("=" * 70)
print("TASK 3: Optimal Temperature Placement (3 extra points)")
print("=" * 70)

# Fisher Information Matrix approach
# We compute the FIM at the true parameters using finite-difference gradients.
# FIM_ij = sum_k (1/sigma_k^2) * (dX/dtheta_i) * (dX/dtheta_j)
# We assume uniform sigma = 0.02 (2% conversion uncertainty).

np.random.seed(456)

def compute_fim(T_vals, sigma_y=0.02):
    """Compute Fisher Information Matrix for parameters [Ea, K_CO0, K_O2_0]."""
    eps = 1e-5
    n_param = 3
    n_pts = len(T_vals)
    grad = np.zeros((n_pts, n_param))

    params0 = np.array([Ea_true, K_CO0_true, K_O2_0_true])

    for j in range(n_param):
        params_plus = params0.copy()
        params_minus = params0.copy()
        h = max(eps, eps * abs(params0[j]))
        params_plus[j] += h
        params_minus[j] -= h
        Xp = pfr_conversion(T_vals, params_plus[0], params_plus[1], params_plus[2], tau)
        Xm = pfr_conversion(T_vals, params_minus[0], params_minus[1], params_minus[2], tau)
        grad[:, j] = (Xp - Xm) / (2.0 * h)

    FIM = np.zeros((n_param, n_param))
    for k in range(n_pts):
        FIM += np.outer(grad[k], grad[k]) / (sigma_y ** 2)
    return FIM, grad

# Candidate temperatures (in °C) to scan
candidates_C = np.arange(150, 501, 10)  # 150 to 500°C in 10°C steps
candidates_K = candidates_C + 273.15

# We already have 6 points. We'll add 3 candidate points.
# For efficiency, we do a random search over triplets
n_random_triplets = 2000
all_temps = []

for i in range(n_random_triplets):
    t3 = np.random.choice(candidates_K, 3, replace=False)
    all_temps.append(t3)

# Also add some heuristic temperatures
for extra in [[150, 200, 250], [200, 300, 400], [150, 250, 350],
              [400, 450, 500], [200, 350, 500], [150, 300, 500],
              [250, 350, 450]]:
    all_temps.append(np.array(extra) + 273.15)

best_std_Ea = float('inf')
best_temps = None

for t3 in all_temps:
    T_all = np.concatenate([T, t3])
    FIM, _ = compute_fim(T_all)
    try:
        cov = np.linalg.inv(FIM)
        std_Ea_candidate = np.sqrt(cov[0, 0])
        if std_Ea_candidate < best_std_Ea:
            best_std_Ea = std_Ea_candidate
            best_temps = t3.copy()
    except np.linalg.LinAlgError:
        continue

print(f"  Best 3 additional temperatures: {np.sort(best_temps - 273.15)} °C")
print(f"  Expected Ea std with these added: {best_std_Ea:.1f} J/mol")

# Also compute baseline std without extra points
FIM_base, _ = compute_fim(T)
cov_base = np.linalg.inv(FIM_base)
std_Ea_base = np.sqrt(cov_base[0, 0])
print(f"  Baseline Ea std (6 points only): {std_Ea_base:.1f} J/mol")
print()

# ──────────────────────────────────────────────
# 6. Task 4: Correlation matrix
# ──────────────────────────────────────────────
print("=" * 70)
print("TASK 4: Correlation Matrix of Fitted Parameters")
print("=" * 70)

FIM_full, _ = compute_fim(T)
cov_full = np.linalg.inv(FIM_full)
corr = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        corr[i, j] = cov_full[i, j] / np.sqrt(cov_full[i, i] * cov_full[j, j])

param_names = ['Ea', 'K_CO0', 'K_O2_0']
print(f"  {'':>10} {'Ea':>10} {'K_CO0':>10} {'K_O2_0':>10}")
for i, name in enumerate(param_names):
    print(f"  {name:>10} {corr[i, 0]:>10.4f} {corr[i, 1]:>10.4f} {corr[i, 2]:>10.4f}")
print()

# Covariance matrix
print("  Covariance matrix:")
print(f"  {'':>10} {'Ea':>15} {'K_CO0':>15} {'K_O2_0':>15}")
for i, name in enumerate(param_names):
    print(f"  {name:>10} {cov_full[i, 0]:>15.6e} {cov_full[i, 1]:>15.6e} {cov_full[i, 2]:>15.6e}")
print()

# ──────────────────────────────────────────────
# 7. Plotting
# ──────────────────────────────────────────────
print("=" * 70)
print("Saving plot to co_ox_sensitivity.png")
print("=" * 70)

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                       hspace=0.30, wspace=0.30)

# ── Panel 1: Fitted Ea distribution (noise sensitivity) ──
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(fitted_Ea, bins=40, color='steelblue', edgecolor='white', alpha=0.85,
         density=True)
ax1.axvline(Ea_true, color='red', linestyle='--', linewidth=2, label=f'True Ea = {Ea_true:.0f}')
ax1.axvline(mean_Ea, color='darkgreen', linestyle=':', linewidth=2,
            label=f'Mean = {mean_Ea:.0f}')
ax1.set_xlabel('Fitted Ea (J/mol)', fontsize=12)
ax1.set_ylabel('Probability Density', fontsize=12)
ax1.set_title('Panel 1: Ea Distribution (Noise Sensitivity, 1000 MC)', fontsize=13)
ax1.legend(fontsize=10)
ax1.text(0.98, 0.95, f'std = {std_Ea:.0f} J/mol', transform=ax1.transAxes,
         ha='right', va='top', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ── Panel 2: Ea std vs N ──
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(Ns, Ea_std_vs_N, 'o-', color='crimson', linewidth=2.5, markersize=10)
ax2.set_xlabel('Number of Data Points (N)', fontsize=12)
ax2.set_ylabel('Ea Standard Deviation (J/mol)', fontsize=12)
ax2.set_title('Panel 2: Ea Std vs Number of Data Points', fontsize=13)
ax2.set_xticks(Ns)
ax2.grid(True, alpha=0.3)
for N, std_val in zip(Ns, Ea_std_vs_N):
    ax2.annotate(f'{std_val:.0f}', (N, std_val), textcoords='offset points',
                 xytext=(0, 12), ha='center', fontsize=10)

# ── Panel 3: Fitted curves vs data with error bands ──
ax3 = fig.add_subplot(gs[1, :])

# Compute the fitted curve from the noise-sensitivity mean parameters
T_smooth = np.linspace(150, 500, 200) + 273.15
X_smooth_mean = pfr_conversion(T_smooth, mean_Ea, mean_KCO0, mean_KO20, tau)
X_smooth_true = pfr_conversion(T_smooth, Ea_true, K_CO0_true, K_O2_0_true, tau)

# Compute error bands from MC fits
n_band_samples = 200
X_curves = np.zeros((n_band_samples, len(T_smooth)))
for i in range(n_band_samples):
    ea_i   = np.random.choice(fitted_Ea, 1)[0]
    kc0_i  = np.random.choice(fitted_KCO0, 1)[0]
    ko0_i  = np.random.choice(fitted_KO20, 1)[0]
    X_curves[i] = pfr_conversion(T_smooth, ea_i, kc0_i, ko0_i, tau)

X_lower = np.percentile(X_curves, 2.5, axis=0)
X_upper = np.percentile(X_curves, 97.5, axis=0)

ax3.fill_between(T_smooth - 273.15, X_lower * 100, X_upper * 100,
                 color='steelblue', alpha=0.25, label='95% CI (MC)')
ax3.plot(T_smooth - 273.15, X_smooth_mean * 100, 'b-', linewidth=2.5,
         label=f'Fitted (mean)')
ax3.plot(T_smooth - 273.15, X_smooth_true * 100, 'r--', linewidth=2,
         label=f'True parameters')
ax3.plot(T_C, X * 100, 'o', color='darkorange', markersize=10, zorder=5,
         label='Data')
# Highlight the noisy points
ax3.plot(T_C[noise_idx], X[noise_idx] * 100, 's', color='red',
         markersize=12, fillstyle='none', markeredgewidth=2.5,
         label='Noisy points (400, 450°C)')

ax3.set_xlabel('Temperature (°C)', fontsize=12)
ax3.set_ylabel('Conversion X (%)', fontsize=12)
ax3.set_title('Panel 3: Fitted Curve vs Data with Error Bands', fontsize=13)
ax3.legend(fontsize=10, loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-5, 105)

plt.suptitle('CO Oxidation on Pt — LH Kinetic Parameter Sensitivity Analysis',
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig('co_ox_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()

print("Plot saved successfully!")
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"tau (space time) = {tau:.4f} s")
print()
print(f"Task 1 — Noise Sensitivity (5% on 400°C, 450°C):")
print(f"  Ea     = {mean_Ea:.1f} ± {std_Ea:.1f} J/mol")
print(f"  K_CO0  = {mean_KCO0:.6f} ± {std_KCO0:.6f}")
print(f"  K_O2_0 = {mean_KO20:.6f} ± {std_KO20:.6f}")
print()
print(f"Task 2 — Data Point Requirement:")
for N, std_val in zip(Ns, Ea_std_vs_N):
    print(f"  N={N}: Ea std = {std_val:.1f} J/mol")
print()
print(f"Task 3 — Optimal Temperature Placement:")
print(f"  Best 3 extra T: {np.sort(best_temps - 273.15)} °C")
print(f"  Expected Ea std: {best_std_Ea:.1f} J/mol (baseline: {std_Ea_base:.1f})")
print()
print(f"Task 4 — Correlation Matrix:")
for i, name in enumerate(param_names):
    print(f"  {name}: {corr[i]}")
