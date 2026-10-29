#!/usr/bin/env python3
"""
Arrhenius parameter estimation with bootstrap analysis.

Reads benchmark_t50.csv, performs bootstrap resampling, fits Arrhenius
parameters (Ea, p) using log-log linear regression and nonlinear least squares,
compares Fick diffusion model via AIC/BIC, and computes predictions.
"""

import csv
import json
import math
import numpy as np
from scipy import optimize, stats

# Constants
R_GAS = 8.314  # J/(mol·K)
BOOTSTRAP_ITERATIONS = 50
RANDOM_SEED = 42
PREDICT_T = 650.0  # K
PREDICT_R = 2e-6   # m


def read_csv(filepath):
    """Read CSV file and return arrays of temperature, radius, t50."""
    temps = []
    radii = []
    t50s = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            temps.append(float(row['temperature_K']))
            radii.append(float(row['radius_m']))
            t50s.append(float(row['t50_s']))
    return np.array(temps), np.array(radii), np.array(t50s)


def arrhenius_model(A, p, Ea, T, r):
    """Arrhenius model: t50 = A * r^p * exp(Ea/(R*T))."""
    return A * (r ** p) * np.exp(Ea / (R_GAS * T))


def fick_model(A, Ea, T, r):
    """Fick diffusion model (p=2): t50 = A * r^2 * exp(Ea/(R*T))."""
    return arrhenius_model(A, 2.0, Ea, T, r)


def log_log_regression(T, r, t50):
    """Fit Arrhenius parameters using log-log linear regression.

    Model: ln(t50) = ln(A) + p*ln(r) + Ea/(R*T)
    Rewrite: ln(t50) = beta0 + beta1*ln(r) + beta2*(1/T)
    where beta0 = ln(A), beta1 = p, beta2 = Ea/R.
    """
    # Design matrix for linear regression: [1, ln(r), 1/T]
    X = np.column_stack([np.ones(len(T)), np.log(r), 1.0 / T])
    y = np.log(t50)

    # Solve normal equations: X^T X beta = X^T y
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)

    # Extract parameters
    A_est = np.exp(beta[0])
    p_est = beta[1]
    Ea_est = beta[2] * R_GAS

    return A_est, p_est, Ea_est


def nonlinear_least_squares(T, r, t50, A0=None, p0=None, Ea0=None):
    """Fit Arrhenius parameters using nonlinear least squares.

    Uses scipy.optimize.curve_fit with bounds to ensure positive parameters.
    """
    # Provide initial guesses if not given
    if A0 is None or p0 is None or Ea0 is None:
        # Use linear regression as initial guess
        A0, p0, Ea0 = log_log_regression(T, r, t50)

    # Define bounds: A > 0, p can be any real, Ea > 0
    bounds = ([0, -np.inf, 0], [np.inf, np.inf, np.inf])

    try:
        popt, pcov = optimize.curve_fit(
            lambda T_r, A, p, Ea: arrhenius_model(A, p, Ea, T_r[:, 0], T_r[:, 1]),
            np.column_stack([T, r]), t50,
            p0=[A0, p0, Ea0],
            bounds=bounds,
            maxfev=10000
        )
        A_fit, p_fit, Ea_fit = popt
    except Exception as e:
        # Fallback to linear regression if NLS fails
        print(f"  NLS failed: {e}, falling back to linear regression")
        A_fit, p_fit, Ea_fit = A0, p0, Ea0

    return A_fit, p_fit, Ea_fit


def compute_rss(model_func, params, T, r, t50):
    """Compute residual sum of squares on original t50 scale.
    
    For models with signature model_func(A, p, Ea, T, r): params = [A, p, Ea]
    For models with signature model_func(A, Ea, T, r): params = [A, Ea]
    """
    if len(params) == 3:
        A, p, Ea = params
        pred = model_func(A, p, Ea, T, r)
    elif len(params) == 2:
        A, Ea = params
        pred = model_func(A, Ea, T, r)
    else:
        raise ValueError(f"Unexpected number of parameters: {len(params)}")
    residuals = t50 - pred
    return np.sum(residuals ** 2)


def compute_aic_bic(n, rss, k):
    """Compute AIC and BIC for a model.

    AIC = n * ln(RSS/n) + 2*k
    BIC = n * ln(RSS/n) + k * ln(n)
    """
    ln_rss_n = math.log(rss / n)
    aic = n * ln_rss_n + 2 * k
    bic = n * ln_rss_n + k * math.log(n)
    return aic, bic


def fit_fick_model(T, r, t50, A0=None, Ea0=None):
    """Fit Fick diffusion model (p=2 fixed) using nonlinear least squares."""
    if A0 is None or Ea0 is None:
        # Use NLS results as initial guess
        A0, p0, Ea0 = log_log_regression(T, r, t50)
        A0 = A0  # Keep A from log-log
        # Use Ea from NLS or log-log
        if Ea0 is None:
            Ea0 = Ea_ll if 'Ea_ll' in dir() else Ea_ll

    def fick_wrapper(T_r, A, Ea):
        return A * (T_r[:, 1] ** 2) * np.exp(Ea / (R_GAS * T_r[:, 0]))

    try:
        popt, _ = optimize.curve_fit(
            fick_wrapper,
            np.column_stack([T, r]), t50,
            p0=[A0, Ea0],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=10000
        )
        A_fit, Ea_fit = popt
        return A_fit, Ea_fit
    except Exception as e:
        print(f"  Fick model fitting failed: {e}")
        return None, None


def bootstrap_analysis(T, r, t50):
    """Perform bootstrap resampling and model fitting."""
    np.random.seed(RANDOM_SEED)
    n = len(T)

    # Storage for bootstrap results
    bootstrap_log = []

    for i in range(BOOTSTRAP_ITERATIONS):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        T_boot = T[indices]
        r_boot = r[indices]
        t50_boot = t50[indices]

        # Fit using log-log linear regression
        A_ll, p_ll, Ea_ll = log_log_regression(T_boot, r_boot, t50_boot)

        # Fit using nonlinear least squares
        A_nls, p_nls, Ea_nls = nonlinear_least_squares(T_boot, r_boot, t50_boot,
                                                       A0=A_ll, p0=p_ll, Ea0=Ea_ll)

        # Compute RSS and AIC/BIC for NLS fit (on original scale)
        rss_nls = compute_rss(arrhenius_model, [A_nls, p_nls, Ea_nls], T_boot, r_boot, t50_boot)
        aic_nls, bic_nls = compute_aic_bic(n, rss_nls, 3)

        # Store bootstrap iteration results
        bootstrap_log.append({
            "iteration": i + 1,
            "A_loglog": float(A_ll),
            "p_loglog": float(p_ll),
            "Ea_loglog": float(Ea_ll),
            "A_nls": float(A_nls),
            "p_nls": float(p_nls),
            "Ea_nls": float(Ea_nls),
            "RSS_NLS": float(rss_nls),
            "AIC_NLS": float(aic_nls),
            "BIC_NLS": float(bic_nls)
        })

    return bootstrap_log


def main():
    print("Reading data...")
    T, r, t50 = read_csv('data/benchmark_t50.csv')
    n = len(T)
    print(f"  Loaded {n} data points")

    # Fit on full dataset using both methods
    print("Fitting Arrhenius model on full dataset...")
    A_ll, p_ll, Ea_ll = log_log_regression(T, r, t50)
    print(f"  Log-log regression: A={A_ll:.6e}, p={p_ll:.6f}, Ea={Ea_ll:.6f} J/mol")

    A_nls, p_nls, Ea_nls = nonlinear_least_squares(T, r, t50)
    print(f"  NLS: A={A_nls:.6e}, p={p_nls:.6f}, Ea={Ea_nls:.6f} J/mol")

    # Bootstrap analysis
    print(f"Performing {BOOTSTRAP_ITERATIONS} bootstrap iterations...")
    bootstrap_log = bootstrap_analysis(T, r, t50)

    # Compute statistics from bootstrap
    A_ll_vals = [b["A_loglog"] for b in bootstrap_log]
    p_ll_vals = [b["p_loglog"] for b in bootstrap_log]
    Ea_ll_vals = [b["Ea_loglog"] for b in bootstrap_log]
    A_nls_vals = [b["A_nls"] for b in bootstrap_log]
    p_nls_vals = [b["p_nls"] for b in bootstrap_log]
    Ea_nls_vals = [b["Ea_nls"] for b in bootstrap_log]

    # Compute mean and std for each parameter
    def mean_std(vals):
        return float(np.mean(vals)), float(np.std(vals))

    A_ll_mean, A_ll_std = mean_std(A_ll_vals)
    p_ll_mean, p_ll_std = mean_std(p_ll_vals)
    Ea_ll_mean, Ea_ll_std = mean_std(Ea_ll_vals)
    A_nls_mean, A_nls_std = mean_std(A_nls_vals)
    p_nls_mean, p_nls_std = mean_std(p_nls_vals)
    Ea_nls_mean, Ea_nls_std = mean_std(Ea_nls_vals)

    print(f"  Bootstrap log-log: A={A_ll_mean:.6e}±{A_ll_std:.6e}, p={p_ll_mean:.6f}±{p_ll_std:.6f}, Ea={Ea_ll_mean:.6f}±{Ea_ll_std:.6f}")
    print(f"  Bootstrap NLS: A={A_nls_mean:.6e}±{A_nls_std:.6e}, p={p_nls_mean:.6f}±{p_nls_std:.6f}, Ea={Ea_nls_mean:.6f}±{Ea_nls_std:.6f}")

    # AIC/BIC comparison on full dataset
    print("Computing AIC/BIC for model comparison...")
    # Arrhenius model (3 parameters: A, p, Ea)
    rss_arrhenius = compute_rss(arrhenius_model, [A_nls, p_nls, Ea_nls], T, r, t50)
    aic_arrhenius, bic_arrhenius = compute_aic_bic(n, rss_arrhenius, 3)

    # Fick model (2 parameters: A, Ea, with p=2 fixed)
    # Fit Fick model using NLS with p=2 fixed
    def fick_wrapper(T_r, A, Ea):
        return A * (T_r[:, 1] ** 2) * np.exp(Ea / (R_GAS * T_r[:, 0]))

    try:
        popt_fick, _ = optimize.curve_fit(
            fick_wrapper,
            np.column_stack([T, r]), t50,
            p0=[A_nls, Ea_nls],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=10000
        )
        A_fick, Ea_fick = popt_fick
        # Compute RSS directly for Fick model
        pred_fick = fick_wrapper(np.column_stack([T, r]), A_fick, Ea_fick)
        rss_fick = float(np.sum((t50 - pred_fick) ** 2))
        aic_fick, bic_fick = compute_aic_bic(n, rss_fick, 2)
        print(f"  Fick model (p=2): A={A_fick:.6e}, Ea={Ea_fick:.6f} J/mol")
        print(f"  Fick RSS={rss_fick:.6e}, AIC={aic_fick:.6f}, BIC={bic_fick:.6f}")
    except Exception as e:
        print(f"  Fick model fitting failed: {e}")
        rss_fick = np.inf
        aic_fick = np.inf
        bic_fick = np.inf

    # Prediction at T=650K, r=2e-6m
    print(f"Computing prediction at T={PREDICT_T}K, r={PREDICT_R}m...")
    t50_pred_ll = arrhenius_model(A_ll, p_ll, Ea_ll, PREDICT_T, PREDICT_R)
    t50_pred_nls = arrhenius_model(A_nls, p_nls, Ea_nls, PREDICT_T, PREDICT_R)

    # Bootstrap prediction intervals
    pred_ll_bootstrap = [arrhenius_model(b["A_loglog"], b["p_loglog"], b["Ea_loglog"], PREDICT_T, PREDICT_R) for b in bootstrap_log]
    pred_nls_bootstrap = [arrhenius_model(b["A_nls"], b["p_nls"], b["Ea_nls"], PREDICT_T, PREDICT_R) for b in bootstrap_log]

    pred_ll_mean = float(np.mean(pred_ll_bootstrap))
    pred_ll_std = float(np.std(pred_ll_bootstrap))
    pred_nls_mean = float(np.mean(pred_nls_bootstrap))
    pred_nls_std = float(np.std(pred_nls_bootstrap))

    print(f"  Log-log prediction: t50={t50_pred_ll:.6f}s (bootstrap mean={pred_ll_mean:.6f}±{pred_ll_std:.6f})")
    print(f"  NLS prediction: t50={t50_pred_nls:.6f}s (bootstrap mean={pred_nls_mean:.6f}±{pred_nls_std:.6f})")

    # Prepare results
    results = {
        "full_dataset_fit": {
            "log_log_regression": {
                "A": A_ll,
                "p": p_ll,
                "Ea_J_per_mol": Ea_ll
            },
            "nonlinear_least_squares": {
                "A": A_nls,
                "p": p_nls,
                "Ea_J_per_mol": Ea_nls
            }
        },
        "bootstrap_statistics": {
            "log_log": {
                "A_mean": A_ll_mean,
                "A_std": A_ll_std,
                "p_mean": p_ll_mean,
                "p_std": p_ll_std,
                "Ea_mean": Ea_ll_mean,
                "Ea_std": Ea_ll_std
            },
            "nls": {
                "A_mean": A_nls_mean,
                "A_std": A_nls_std,
                "p_mean": p_nls_mean,
                "p_std": p_nls_std,
                "Ea_mean": Ea_nls_mean,
                "Ea_std": Ea_nls_std
            }
        },
        "model_comparison": {
            "arrhenius_model": {
                "RSS": float(rss_arrhenius),
                "AIC": float(aic_arrhenius),
                "BIC": float(bic_arrhenius),
                "n_parameters": 3
            },
            "fick_model": {
                "RSS": float(rss_fick),
                "AIC": float(aic_fick),
                "BIC": float(bic_fick),
                "n_parameters": 2,
                "fixed_p": 2.0
            },
            "aic_difference": float(aic_arrhenius - aic_fick),
            "bic_difference": float(bic_arrhenius - bic_fick)
        },
        "prediction": {
            "temperature_K": PREDICT_T,
            "radius_m": PREDICT_R,
            "log_log_prediction": {
                "t50_seconds": t50_pred_ll,
                "bootstrap_mean": pred_ll_mean,
                "bootstrap_std": pred_ll_std
            },
            "nls_prediction": {
                "t50_seconds": t50_pred_nls,
                "bootstrap_mean": pred_nls_mean,
                "bootstrap_std": pred_nls_std
            }
        },
        "metadata": {
            "n_data_points": n,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "random_seed": RANDOM_SEED,
            "gas_constant_J_per_mol_K": R_GAS
        }
    }

    # Write results.json
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Written results.json")

    # Write loop_log.jsonl
    with open('loop_log.jsonl', 'w') as f:
        for entry in bootstrap_log:
            f.write(json.dumps(entry) + '\n')
    print("Written loop_log.jsonl")

    # Write report.md
    report = f"""# Arrhenius Parameter Analysis Report

## Data Summary
- Number of data points: {n}
- Bootstrap iterations: {BOOTSTRAP_ITERATIONS}
- Random seed: {RANDOM_SEED}

## Full Dataset Fit Results

### Log-Log Linear Regression
- Pre-exponential factor A: {A_ll:.6e}
- Radius scaling exponent p: {p_ll:.6f}
- Activation energy Ea: {Ea_ll:.6f} J/mol ({Ea_ll/1000:.2f} kJ/mol)

### Nonlinear Least Squares
- Pre-exponential factor A: {A_nls:.6e}
- Radius scaling exponent p: {p_nls:.6f}
- Activation energy Ea: {Ea_nls:.6f} J/mol ({Ea_nls/1000:.2f} kJ/mol)

## Bootstrap Statistics (50 iterations)

### Log-Log Regression
- A: {A_ll_mean:.6e} ± {A_ll_std:.6e}
- p: {p_ll_mean:.6f} ± {p_ll_std:.6f}
- Ea: {Ea_ll_mean:.6f} ± {Ea_ll_std:.6f} J/mol

### Nonlinear Least Squares
- A: {A_nls_mean:.6e} ± {A_nls_std:.6e}
- p: {p_nls_mean:.6f} ± {p_nls_std:.6f}
- Ea: {Ea_nls_mean:.6f} ± {Ea_nls_std:.6f} J/mol

## Model Comparison (AIC/BIC)

| Model | RSS | AIC | BIC | Parameters |
|-------|-----|-----|-----|------------|
| Arrhenius (free p) | {rss_arrhenius:.6e} | {aic_arrhenius:.6f} | {bic_arrhenius:.6f} | 3 |
| Fick (p=2 fixed) | {rss_fick:.6e} | {aic_fick:.6f} | {bic_fick:.6f} | 2 |

- AIC difference (Arrhenius - Fick): {aic_arrhenius - aic_fick:.6f}
- BIC difference (Arrhenius - Fick): {bic_arrhenius - bic_fick:.6f}

**Interpretation:**
- Negative AIC/BIC difference favors the Arrhenius model (lower AIC/BIC is better).
- Positive difference favors the simpler Fick model.
- The magnitude indicates the strength of evidence.

## Prediction at T=650K, r=2×10⁻⁶m

### Log-Log Regression
- Predicted t50: {t50_pred_ll:.6f} s
- Bootstrap mean: {pred_ll_mean:.6f} s
- Bootstrap std: {pred_ll_std:.6f} s

### Nonlinear Least Squares
- Predicted t50: {t50_pred_nls:.6f} s
- Bootstrap mean: {pred_nls_mean:.6f} s
- Bootstrap std: {pred_nls_std:.6f} s

## Conclusion
The Arrhenius model with free radius scaling exponent p provides a flexible
description of the temperature and radius dependence of t50. The bootstrap
analysis quantifies the uncertainty in the parameter estimates. Model comparison
via AIC/BIC helps determine whether the simpler Fick model (p=2) is sufficient
or if the additional parameter p significantly improves the fit.
"""

    with open('report.md', 'w') as f:
        f.write(report)
    print("Written report.md")

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()