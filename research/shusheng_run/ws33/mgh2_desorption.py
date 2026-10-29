#!/usr/bin/env python3
"""
MgH2 Hydrogen Desorption Simulation
Models hydrogen diffusion-controlled desorption from MgH2 particles
using Fick's second law with Arrhenius temperature dependence.

This simulation calculates the fraction of hydrogen released from
spherical MgH2 particles as a function of time, temperature, and
particle size.
"""

import numpy as np
from scipy.optimize import brentq
import json
import os
import csv

# Physical constants
R = 8.314  # Gas constant (J/mol·K)

# MgH2 material properties (typical values from literature)
# Source: Various studies on MgH2 hydrogen storage kinetics
MgH2_PROPS = {
    'D0': 1.0e-5,  # Pre-exponential factor (m^2/s) - typical range 1e-6 to 1e-4
    'Ea': 70000,   # Activation energy (J/mol) - typical range 60-80 kJ/mol
    'rho': 1738,   # Density (kg/m^3)
    'capacity_wt': 7.6,  # Theoretical H capacity (wt%)
}

def diffusion_coefficient(T):
    """
    Calculate diffusion coefficient using Arrhenius equation:
    D = D0 * exp(-Ea / (R * T))
    """
    D0 = MgH2_PROPS['D0']
    Ea = MgH2_PROPS['Ea']
    return D0 * np.exp(-Ea / (R * T))

def spherical_diffusion_fraction(r, D, t, n_terms=50):
    """
    Calculate the fraction of hydrogen released from a spherical particle
    using the analytical solution to Fick's second law.

    For a sphere with constant surface concentration (C=0), the solution is:
    F(t) = 1 - (6/pi^2) * sum_{n=1}^{inf} (1/n^2) * exp(-n^2 * pi^2 * D * t / r^2)

    where F(t) is the fraction of total hydrogen released at time t.

    Parameters:
    -----------
    r : float
        Particle radius (m)
    D : float
        Diffusion coefficient (m^2/s)
    t : float or array-like
        Time(s) (s)
    n_terms : int
        Number of terms in the series expansion (default: 50)

    Returns:
    --------
    F : float or array-like
        Fraction of hydrogen released (0 to 1)
    """
    t_arr = np.atleast_1d(t)
    F = np.zeros_like(t_arr, dtype=float)
    for n in range(1, n_terms + 1):
        exponent = -n**2 * np.pi**2 * D * t_arr / r**2
        F += (1 / n**2) * np.exp(exponent)
    F = 1 - (6 / np.pi**2) * F
    # Return scalar if input was scalar
    if np.isscalar(t):
        return float(F[0]) if F.size > 0 else 0.0
    return F

def spherical_diffusion_fraction_derivative(r, D, t, n_terms=50):
    """
    Calculate the derivative of the diffusion fraction (desorption rate).
    dF/dt = (6 * D * pi) / (r^2) * sum_{n=1}^{inf} exp(-n^2 * pi^2 * D * t / r^2)
    """
    t_arr = np.atleast_1d(t)
    dFdt = np.zeros_like(t_arr, dtype=float)
    for n in range(1, n_terms + 1):
        exponent = -n**2 * np.pi**2 * D * t_arr / r**2
        dFdt += np.exp(exponent)
    dFdt = (6 * D * np.pi / r**2) * dFdt
    if np.isscalar(t):
        return float(dFdt[0]) if dFdt.size > 0 else 0.0
    return dFdt

def calculate_hydrogen_mass(r, capacity_wt=None):
    """
    Calculate the total hydrogen mass stored in a spherical MgH2 particle.
    """
    if capacity_wt is None:
        capacity_wt = MgH2_PROPS['capacity_wt']
    V = (4/3) * np.pi * r**3
    m_MgH2 = MgH2_PROPS['rho'] * V
    m_H2 = m_MgH2 * capacity_wt / 100.0
    return m_H2

def time_to_fraction(r, D, target_fraction=0.5, t_max=1e6, n_terms=50):
    """
    Calculate the time required to reach a specified fraction of hydrogen release.
    Uses root-finding to solve F(t) = target_fraction.
    """
    def F_minus_target(t):
        return spherical_diffusion_fraction(r, D, t, n_terms) - target_fraction
    try:
        # Check if target is already reached at t=0
        F0 = spherical_diffusion_fraction(r, D, 0, n_terms)
        if F0 >= target_fraction:
            return 0.0
        # Check if target is reached within t_max
        Fmax = spherical_diffusion_fraction(r, D, t_max, n_terms)
        if Fmax < target_fraction:
            return None  # Target not reached within t_max
        # Find root using brentq
        t_half = brentq(F_minus_target, 1e-10, t_max)
        return float(t_half)
    except:
        return None

def simulate_desorption(T, r_list, t_max=3600, n_points=100):
    """
    Simulate hydrogen desorption for multiple particle sizes at a given temperature.
    """
    D = float(diffusion_coefficient(T))
    t = np.linspace(0, t_max, n_points)
    results = {}
    for r in r_list:
        r = float(r)
        F = spherical_diffusion_fraction(r, D, t)
        m_H2_total = calculate_hydrogen_mass(r)
        m_H2_released = m_H2_total * F
        key = f'r_{r:.2e}'
        results[key] = {
            'radius_m': r,
            'time_s': t.tolist(),
            'fraction_released': F.tolist(),
            'hydrogen_released_g': (m_H2_released * 1000).tolist(),
            'diffusion_coefficient_m2s': D,
        }
    return t, results

def simulate_temperature_dependence(r, T_list, t_max=3600, n_points=100):
    """
    Simulate hydrogen desorption at different temperatures for a fixed particle size.
    """
    t = np.linspace(0, t_max, n_points)
    results = {}
    for T in T_list:
        T = float(T)
        D = float(diffusion_coefficient(T))
        F = spherical_diffusion_fraction(r, D, t)
        m_H2_total = calculate_hydrogen_mass(r)
        m_H2_released = m_H2_total * F
        key = f'T_{int(T)}K'
        results[key] = {
            'temperature_K': T,
            'time_s': t.tolist(),
            'fraction_released': F.tolist(),
            'hydrogen_released_g': (m_H2_released * 1000).tolist(),
            'diffusion_coefficient_m2s': D,
        }
    return t, results

def main():
    print("=" * 60)
    print("MgH2 Hydrogen Desorption Simulation")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  D0 = {MgH2_PROPS['D0']:.2e} m^2/s")
    print(f"  Ea = {MgH2_PROPS['Ea']/1000:.0f} kJ/mol")
    print(f"  rho = {MgH2_PROPS['rho']} kg/m^3")
    print(f"  Capacity = {MgH2_PROPS['capacity_wt']} wt%")

    # Define simulation parameters
    r_list = np.array([100e-9, 500e-9, 1e-6, 5e-6, 10e-6])  # 100 nm to 10 um
    T_list = np.array([400, 450, 500, 550, 600, 650, 700])
    t_max = 3600  # 1 hour

    print(f"\nParticle sizes: {[f'{r*1e9:.0f} nm' for r in r_list]}")
    print(f"Temperatures: {T_list} K")
    print(f"Max time: {t_max/60:.1f} min")

    all_results = {}

    # Temperature dependence at fixed particle size (r = 1 um)
    r_fixed = 1e-6
    print(f"\nSimulating temperature dependence at r = {r_fixed*1e9:.0f} um...")
    t_temp, results_temp = simulate_temperature_dependence(r_fixed, T_list, t_max, 100)
    all_results.update(results_temp)

    # Size dependence at fixed temperature (T = 550 K)
    T_fixed = 550
    print(f"\nSimulating size dependence at T = {T_fixed} K...")
    t_size, results_size = simulate_desorption(T_fixed, r_list, t_max, 100)
    all_results.update(results_size)

    # Parameter sweep with 50% times
    print(f"\nRunning parameter sweep (t = {t_max/60:.1f} min)...")
    sweep_results = {}
    for T in T_list[::2]:
        for r in r_list[::2]:
            D = float(diffusion_coefficient(T))
            F = spherical_diffusion_fraction(r, D, t_max)
            m_H2 = calculate_hydrogen_mass(r)
            key = f'T{T}_r{r:.0e}'
            t50 = time_to_fraction(r, D, target_fraction=0.5, t_max=t_max)
            sweep_results[key] = {
                'temperature_K': float(T),
                'radius_m': float(r),
                'diffusion_coefficient_m2s': D,
                'fraction_released_at_tmax': float(F),
                'hydrogen_released_g': float(m_H2 * F * 1000),
                'time_to_50_percent_s': t50,
                'time_to_50_percent_min': t50/60 if t50 else None,
            }
            if t50 is not None:
                print(f"  T={T}K, r={r*1e9:.0f}nm: t_50%={t50:.3f}s = {t50/60:.2f}min")
            else:
                print(f"  T={T}K, r={r*1e9:.0f}nm: t_50% > {t_max/60:.1f}min (not reached)")

    all_results.update(sweep_results)

    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)

    print(f"\n--- Temperature Dependence (r = 1 um) ---")
    for T in T_list:
        T = float(T)
        D = float(diffusion_coefficient(T))
        F = spherical_diffusion_fraction(r_fixed, D, t_max)
        print(f"  T={T:4.0f} K: D={D:.2e} m^2/s, F(t_max)={F:.3f}")

    print(f"\n--- Size Dependence (T = 550 K) ---")
    T_fixed = float(T_fixed)
    D_fixed = float(diffusion_coefficient(T_fixed))
    for r in r_list:
        r = float(r)
        F = spherical_diffusion_fraction(r, D_fixed, t_max)
        print(f"  r={r*1e9:4.0f} nm: D={D_fixed:.2e} m^2/s, F(t_max)={F:.3f}")

    print(f"\n--- Parameter Sweep (t = {t_max/60:.1f} min) ---")
    for key in sorted(sweep_results.keys()):
        val = sweep_results[key]
        print(f"  T={val['temperature_K']}K, r={val['radius_m']*1e9:.0f}nm: "
              f"F={val['fraction_released_at_tmax']:.3f}, "
              f"t_50%={val['time_to_50_percent_min']}min")

    # Save results
    output_dir = "/workspace/research/shusheng_run/ws33/results"
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, "mgh2_desorption_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Save summary data for plotting
    summary_data = {
        'particle_sizes_nm': [float(r*1e9) for r in r_list],
        'temperatures_K': [float(T) for T in T_list],
        'diffusion_coefficients_at_550K': [float(diffusion_coefficient(T_fixed)) for _ in r_list],
        'fraction_released_at_550K': [float(spherical_diffusion_fraction(r, diffusion_coefficient(T_fixed), t_max)) for r in r_list],
        'sweep_results': sweep_results,
    }

    summary_file = os.path.join(output_dir, "mgh2_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary saved to: {summary_file}")

    # Save CSV files for easy plotting
    # Temperature dependence CSV
    temp_csv = os.path.join(output_dir, "mgh2_temp_dependence.csv")
    with open(temp_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Temperature (K)', 'Diffusion Coeff (m^2/s)', 'Fraction Released (t_max)'])
        for T in T_list:
            T = float(T)
            D = float(diffusion_coefficient(T))
            F = spherical_diffusion_fraction(r_fixed, D, t_max)
            writer.writerow([T, D, F])
    print(f"Temperature dependence data saved to: {temp_csv}")

    # Size dependence CSV
    size_csv = os.path.join(output_dir, "mgh2_size_dependence.csv")
    with open(size_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Particle Size (nm)', 'Diffusion Coeff (m^2/s)', 'Fraction Released (t_max)'])
        for r in r_list:
            r = float(r)
            F = spherical_diffusion_fraction(r, D_fixed, t_max)
            writer.writerow([r*1e9, D_fixed, F])
    print(f"Size dependence data saved to: {size_csv}")

    # 50% release time CSV
    time50_csv = os.path.join(output_dir, "mgh2_t50_sweep.csv")
    with open(time50_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Temperature (K)', 'Particle Size (nm)', 't_50% (min)'])
        for key in sorted(sweep_results.keys()):
            val = sweep_results[key]
            writer.writerow([val['temperature_K'], val['radius_m']*1e9, val['time_to_50_percent_min']])
    print(f"50% release time data saved to: {time50_csv}")

    print("\nSimulation complete!")

if __name__ == "__main__":
    main()