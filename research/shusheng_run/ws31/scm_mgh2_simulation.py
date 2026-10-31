#!/usr/bin/env python3
"""
Shrinking Core Model (SCM) for MgH2 Hydrogen Absorption
Computational Workflow with 50+ Cycles and Resolution Validation

This script implements a physically-grounded SCM simulation for MgH2 hydrogen
absorption with explicit kinetic parameters from literature ranges.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import json
import warnings
warnings.filterwarnings('ignore')


class MgH2SCM:
    """
    Shrinking Core Model for MgH2 Hydrogen Absorption
    
    The SCM describes the reaction of Mg with H2 to form MgH2, where the
    reaction front moves inward from the particle surface.
    
    Key assumptions:
    - Spherical particles
    - Isothermal conditions
    - Constant H2 pressure
    - Reaction-controlled or diffusion-controlled kinetics
    """
    
    def __init__(self, 
                 initial_radius: float = 50e-6,  # m (50 microns)
                 temperature: float = 573,  # K (300 C)
                 h2_pressure: float = 1.0,  # MPa
                 reaction_order: int = 1,
                 mechanism: str = 'reaction'):
        """
        Initialize the SCM model with physical parameters.
        
        Parameters:
        -----------
        initial_radius : float
            Initial particle radius in meters
        temperature : float
            Temperature in Kelvin
        h2_pressure : float
            H2 pressure in MPa
        reaction_order : int
            Reaction order with respect to H2 pressure
        mechanism : str
            'reaction' for reaction-controlled, 'diffusion' for diffusion-controlled
        """
        self.initial_radius = initial_radius
        self.temperature = temperature
        self.h2_pressure = h2_pressure
        self.reaction_order = reaction_order
        self.mechanism = mechanism
        
        # Material properties (from literature)
        self.rho_Mg = 1740.0  # kg/m^3 (density of Mg)
        self.rho_MgH2 = 1680.0  # kg/m^3 (density of MgH2)
        self.M_Mg = 24.305  # g/mol (molar mass of Mg)
        self.M_H2 = 2.016  # g/mol (molar mass of H2)
        self.M_MgH2 = 26.321  # g/mol (molar mass of MgH2)
        
        # Kinetic parameters (from literature ranges)
        # Based on: B. Bogdanovic, M. Schwickardi, J. Alloys Compd. (1997)
        # and other literature on MgH2 hydrogenation kinetics
        self.A_reaction = 1.5e8  # m/s (pre-exponential factor for reaction)
        self.Ea_reaction = 45000  # J/mol (activation energy for reaction)
        self.D0 = 1.2e-5  # m^2/s (pre-exponential for diffusion)
        self.Ea_diff = 65000  # J/mol (activation energy for diffusion)
        
        # Gas constant
        self.R = 8.314  # J/(mol·K)
        
        # Calculate rate constants at given temperature
        self.k_rate = self._calculate_k_rate()
        self.D_eff = self._calculate_D_eff()
    
    def _calculate_k_rate(self) -> float:
        """Calculate reaction rate constant using Arrhenius equation."""
        if self.mechanism == 'reaction':
            return self.A_reaction * np.exp(-self.Ea_reaction / (self.R * self.temperature))
        return 0.0
    
    def _calculate_D_eff(self) -> float:
        """Calculate effective diffusion coefficient using Arrhenius equation."""
        if self.mechanism == 'diffusion':
            return self.D0 * np.exp(-self.Ea_diff / (self.R * self.temperature))
        return 0.0
    
    def _calculate_conversion(self, r_core: float) -> float:
        """
        Calculate conversion based on core radius.
        
        Conversion X = 1 - (r_core / R_initial)^3
        """
        if r_core <= 0:
            return 1.0
        return 1.0 - (r_core / self.initial_radius) ** 3
    
    def _calculate_reaction_rate(self, r_core: float) -> float:
        """
        Calculate reaction rate at current core radius.
        
        For reaction-controlled mechanism:
        dX/dt = k * P_H2^n * (1 - X)^(2/3)
        
        For diffusion-controlled mechanism:
        dX/dt = k * P_H2^n * (1 - (1 - X)^(1/3))
        """
        X = self._calculate_conversion(r_core)
        
        if self.mechanism == 'reaction':
            # Reaction-controlled: rate proportional to unreacted surface area
            # dX/dt = k * P_H2^n * (1 - X)^(2/3)
            rate = self.k_rate * (self.h2_pressure ** self.reaction_order) * ((1 - X) ** (2/3))
        else:
            # Diffusion-controlled: rate proportional to diffusion through product layer
            # dX/dt = k * P_H2^n * (1 - (1 - X)^(1/3))
            rate = self.k_rate * (self.h2_pressure ** self.reaction_order) * (1 - (1 - X) ** (1/3))
        
        return rate
    
    def _calculate_diffusion_rate(self, r_core: float) -> float:
        """
        Calculate diffusion rate at current core radius.
        
        For diffusion through product layer:
        dX/dt = (2 * D_eff * C_H2) / (rho_MgH2 * R_initial^2) * (1 / (1/X - 1))
        """
        X = self._calculate_conversion(r_core)
        
        if X >= 1.0 or X <= 0:
            return 0.0
        
        # Concentration of H2 at surface (ideal gas approximation)
        C_H2 = self.h2_pressure * 1e6 / (self.R * self.temperature)  # mol/m^3
        
        # Diffusion rate through product layer
        rate = (2 * self.D_eff * C_H2) / (self.rho_MgH2 * self.initial_radius ** 2) * (1 / (1/X - 1))
        
        return rate
    
    def simulate_cycle(self, 
                      t_max: float = 3600,  # seconds (1 hour)
                      dt: float = 1.0,  # time step
                      method: str = 'euler') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate one absorption cycle using the SCM.
        
        Parameters:
        -----------
        t_max : float
            Maximum simulation time in seconds
        dt : float
            Time step in seconds
        method : str
            Integration method: 'euler' or 'rk4'
        
        Returns:
        --------
        time : np.ndarray
            Time array
        conversion : np.ndarray
            Conversion array
        core_radius : np.ndarray
            Core radius array
        """
        n_steps = int(t_max / dt)
        time = np.zeros(n_steps + 1)
        conversion = np.zeros(n_steps + 1)
        core_radius = np.zeros(n_steps + 1)
        
        # Initial conditions
        r_core = self.initial_radius
        conversion[0] = 0.0
        core_radius[0] = r_core
        
        for i in range(n_steps):
            if self.mechanism == 'reaction':
                rate = self._calculate_reaction_rate(r_core)
            else:
                rate = self._calculate_diffusion_rate(r_core)
            
            # Update core radius using d(r_core)/dt = -rate * (rho_MgH2 / rho_Mg) * (R_initial / 3)
            # This comes from stoichiometry and geometry
            dr_core_dt = -rate * (self.rho_MgH2 / self.rho_Mg) * (self.initial_radius / 3)
            
            if method == 'euler':
                r_core_new = r_core + dr_core_dt * dt
            elif method == 'rk4':
                # RK4 integration for better accuracy
                k1 = dr_core_dt
                k2 = self._calculate_dr_core_dt(r_core + k1 * dt / 2)
                k3 = self._calculate_dr_core_dt(r_core + k2 * dt / 2)
                k4 = self._calculate_dr_core_dt(r_core + k3 * dt)
                r_core_new = r_core + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                raise ValueError(f"Unknown integration method: {method}")
            
            # Ensure core radius doesn't go negative
            r_core_new = max(r_core_new, 0.0)
            r_core = r_core_new
            
            # Calculate conversion
            X = self._calculate_conversion(r_core)
            
            # Store results
            time[i + 1] = time[i] + dt
            conversion[i + 1] = X
            core_radius[i + 1] = r_core
        
        return time, conversion, core_radius
    
    def _calculate_dr_core_dt(self, r_core: float) -> float:
        """Calculate derivative of core radius for RK4."""
        if self.mechanism == 'reaction':
            rate = self._calculate_reaction_rate(r_core)
        else:
            rate = self._calculate_diffusion_rate(r_core)
        
        return -rate * (self.rho_MgH2 / self.rho_Mg) * (self.initial_radius / 3)
    
    def simulate_multiple_cycles(self, 
                                n_cycles: int = 50,
                                t_max: float = 3600,
                                dt: float = 1.0,
                                method: str = 'euler',
                                cycle_variation: float = 0.1) -> Dict:
        """
        Simulate multiple absorption cycles with slight variations to simulate
        real-world cycling behavior (particle size changes, surface conditions, etc.).
        
        Parameters:
        -----------
        n_cycles : int
            Number of cycles to simulate
        t_max : float
            Maximum time per cycle in seconds
        dt : float
            Time step in seconds
        method : str
            Integration method
        cycle_variation : float
            Fractional variation in parameters between cycles (0 to 1)
        
        Returns:
        --------
        results : dict
            Dictionary containing all cycle data
        """
        all_results = {
            'time': [],
            'conversion': [],
            'core_radius': [],
            'cycle': [],
            'initial_radius': [],
            'temperature': [],
            'h2_pressure': []
        }
        
        for cycle in range(n_cycles):
            # Introduce slight variations to simulate real cycling behavior
            # Particle size may change due to pulverization or agglomeration
            radius_variation = 1.0 + (np.random.rand() - 0.5) * 2 * cycle_variation
            temp_variation = 1.0 + (np.random.rand() - 0.5) * 0.02 * cycle_variation  # ±2%
            pressure_variation = 1.0 + (np.random.rand() - 0.5) * 0.05 * cycle_variation  # ±5%
            
            # Update parameters for this cycle
            original_radius = self.initial_radius
            original_temp = self.temperature
            original_pressure = self.h2_pressure
            
            self.initial_radius = original_radius * radius_variation
            self.temperature = original_temp * temp_variation
            self.h2_pressure = original_pressure * pressure_variation
            
            # Simulate this cycle
            time, conversion, core_radius = self.simulate_cycle(t_max, dt, method)
            
            # Store results
            all_results['time'].append(time)
            all_results['conversion'].append(conversion)
            all_results['core_radius'].append(core_radius)
            all_results['cycle'].append(np.ones_like(time) * (cycle + 1))
            all_results['initial_radius'].append(np.ones_like(time) * self.initial_radius)
            all_results['temperature'].append(np.ones_like(time) * self.temperature)
            all_results['h2_pressure'].append(np.ones_like(time) * self.h2_pressure)
            
            # Restore original parameters
            self.initial_radius = original_radius
            self.temperature = original_temp
            self.h2_pressure = original_pressure
        
        # Convert to arrays
        for key in all_results:
            all_results[key] = np.vstack(all_results[key]) if isinstance(all_results[key][0], np.ndarray) else all_results[key]
        
        return all_results
    
    def get_resolution_comparison(self, 
                                 resolutions: List[float] = [0.5, 1.0, 2.0, 5.0]) -> Dict:
        """
        Perform resolution comparison for independent validation.
        
        Test different time step resolutions to ensure numerical convergence.
        
        Parameters:
        -----------
        resolutions : list
            Relative time step resolutions to test (1.0 is default)
        
        Returns:
        --------
        comparison : dict
            Results from different resolution simulations
        """
        comparison = {}
        default_results = None
        
        for res in resolutions:
            dt = 1.0 / res  # Adjust time step based on resolution
            
            # Store original dt
            original_dt = 1.0
            
            # Temporarily adjust dt for this simulation
            # We'll use a smaller dt for higher resolution
            self._temp_dt = original_dt / res
            
            # Simulate with this resolution
            time, conversion, core_radius = self.simulate_cycle(t_max=3600, dt=self._temp_dt, method='rk4')
            
            comparison[f'resolution_{res}'] = {
                'time': time,
                'conversion': conversion,
                'core_radius': core_radius,
                'dt': self._temp_dt
            }
            
            # Store default for comparison
            if res == 1.0:
                default_results = conversion
        
        # Calculate errors relative to default resolution
        if default_results is not None:
            for res in resolutions:
                if res != 1.0:
                    conv = comparison[f'resolution_{res}']['conversion']
                    # Interpolate to same time points if needed
                    max_len = max(len(default_results), len(conv))
                    if len(conv) < max_len:
                        conv = np.interp(np.linspace(0, time[-1], max_len), 
                                        comparison[f'resolution_{res}']['time'], conv)
                    error = np.abs(conv - default_results[:len(conv)])
                    comparison[f'resolution_{res}']['max_error'] = np.max(error)
                    comparison[f'resolution_{res}']['rmse'] = np.sqrt(np.mean(error**2))
        
        return comparison


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create synthetic sample data for analysis when experimental data is unavailable.
    
    This generates physically-grounded synthetic data based on the SCM model.
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate realistic parameter distributions
    radii = np.random.lognormal(mean=np.log(50e-6), sigma=0.3, size=n_samples)  # log-normal size distribution
    temperatures = np.random.normal(loc=573, scale=20, size=n_samples)  # ±20K around 300C
    pressures = np.random.uniform(0.5, 2.0, size=n_samples)  # 0.5-2.0 MPa
    
    # Create DataFrame
    data = pd.DataFrame({
        'radius_m': radii,
        'temperature_K': temperatures,
        'pressure_MPa': pressures
    })
    
    return data


def analyze_cycle_results(results: Dict, output_dir: str = './results') -> Dict:
    """
    Analyze the cycle simulation results and extract key metrics.
    
    Parameters:
    -----------
    results : dict
        Results from simulate_multiple_cycles
    output_dir : str
        Directory to save analysis results
    
    Returns:
    --------
    analysis : dict
        Dictionary containing analysis metrics
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    n_cycles = int(np.max(results['cycle']))
    
    analysis = {
        'cycle_metrics': [],
        'average_conversion_time': [],
        'final_conversion': [],
        'cycle_statistics': {}
    }
    
    for cycle in range(1, n_cycles + 1):
        mask = results['cycle'] == cycle
        time = results['time'][mask]
        conversion = results['conversion'][mask]
        radius = results['initial_radius'][mask]
        
        # Find time to reach 50% and 90% conversion
        t_50 = None
        t_90 = None
        
        for i, X in enumerate(conversion):
            if t_50 is None and X >= 0.5:
                t_50 = time[i]
            if t_90 is None and X >= 0.9:
                t_90 = time[i]
                break
        
        # Final conversion
        X_final = conversion[-1]
        
        # Calculate average rate
        if t_90 is not None and t_90 > 0:
            avg_rate = 0.9 / t_90
        else:
            avg_rate = 0.0
        
        cycle_metrics = {
            'cycle': cycle,
            't_50': t_50 if t_50 is not None else np.nan,
            't_90': t_90 if t_90 is not None else np.nan,
            'final_conversion': X_final,
            'average_rate': avg_rate,
            'mean_radius': np.mean(radius),
            'mean_temperature': np.mean(results['temperature'][mask]),
            'mean_pressure': np.mean(results['h2_pressure'][mask])
        }
        
        analysis['cycle_metrics'].append(cycle_metrics)
        analysis['average_conversion_time'].append(t_90 if t_90 is not None else np.nan)
        analysis['final_conversion'].append(X_final)
    
    # Calculate statistics across cycles
    analysis['cycle_statistics'] = {
        'n_cycles': n_cycles,
        'avg_t_50': np.nanmean([m['t_50'] for m in analysis['cycle_metrics']]),
        'avg_t_90': np.nanmean([m['t_90'] for m in analysis['cycle_metrics']]),
        'avg_final_conversion': np.mean(analysis['final_conversion']),
        'std_final_conversion': np.std(analysis['final_conversion']),
        'avg_rate': np.mean([m['average_rate'] for m in analysis['cycle_metrics']])
    }
    
    # Save cycle metrics to CSV
    metrics_df = pd.DataFrame(analysis['cycle_metrics'])
    metrics_df.to_csv(f'{output_dir}/cycle_metrics.csv', index=False)
    
    # Save summary statistics
    summary = {
        'cycle_statistics': analysis['cycle_statistics'],
        'parameter_ranges': {
            'radius_m': [np.min(results['initial_radius']), np.max(results['initial_radius'])],
            'temperature_K': [np.min(results['temperature']), np.max(results['temperature'])],
            'pressure_MPa': [np.min(results['h2_pressure']), np.max(results['h2_pressure'])]
        }
    }
    
    with open(f'{output_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return analysis


def main():
    """
    Main function to run the complete SCM workflow.
    """
    print("=" * 60)
    print("Shrinking Core Model (SCM) for MgH2 Hydrogen Absorption")
    print("=" * 60)
    
    # Create output directory
    import os
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./data', exist_ok=True)
    
    # Initialize model with reaction-controlled mechanism (typical for MgH2 at moderate temperatures)
    print("\nInitializing SCM model...")
    model = MgH2SCM(
        initial_radius=50e-6,  # 50 microns
        temperature=573,  # 300°C
        h2_pressure=1.0,  # 1 MPa
        reaction_order=1,
        mechanism='reaction'
    )
    
    print(f"  Initial radius: {model.initial_radius*1e6:.1f} μm")
    print(f"  Temperature: {model.temperature:.1f} K ({model.temperature-273.15:.1f}°C)")
    print(f"  H2 Pressure: {model.h2_pressure:.1f} MPa")
    print(f"  Mechanism: {model.mechanism}")
    print(f"  Reaction rate constant: {model.k_rate:.2e} m/s")
    
    # Single cycle simulation for validation
    print("\nSimulating single cycle (validation)...")
    time, conversion, core_radius = model.simulate_cycle(t_max=3600, dt=1.0, method='rk4')
    
    # Save single cycle results
    single_cycle_data = pd.DataFrame({
        'time_s': time,
        'conversion': conversion,
        'core_radius_m': core_radius
    })
    single_cycle_data.to_csv('./results/single_cycle.csv', index=False)
    print(f"  Single cycle completed: {conversion[-1]:.3f} conversion in {time[-1]:.1f} s")
    
    # Multiple cycle simulation (50+ cycles)
    print("\n" + "=" * 60)
    print("Simulating multiple cycles (50 cycles)...")
    print("=" * 60)
    
    n_cycles = 52  # More than 50 as requested
    results = model.simulate_multiple_cycles(
        n_cycles=n_cycles,
        t_max=3600,
        dt=1.0,
        method='rk4',
        cycle_variation=0.15
    )
    
    print(f"  Simulated {n_cycles} cycles with parameter variations")
    
    # Analyze results
    print("\nAnalyzing cycle results...")
    analysis = analyze_cycle_results(results, output_dir='./results')
    
    print(f"\nCycle Statistics:")
    print(f"  Average time to 50% conversion: {analysis['cycle_statistics']['avg_t_50']:.1f} s")
    print(f"  Average time to 90% conversion: {analysis['cycle_statistics']['avg_t_90']:.1f} s")
    print(f"  Average final conversion: {analysis['cycle_statistics']['avg_final_conversion']:.3f}")
    print(f"  Std of final conversion: {analysis['cycle_statistics']['std_final_conversion']:.3f}")
    print(f"  Average reaction rate: {analysis['cycle_statistics']['avg_rate']:.6f} s⁻¹")
    
    # Resolution comparison for validation
    print("\n" + "=" * 60)
    print("Performing resolution comparison (validation)...")
    print("=" * 60)
    
    comparison = model.get_resolution_comparison(resolutions=[0.5, 1.0, 2.0, 5.0])
    
    print("\nResolution Comparison Results:")
    for key, value in comparison.items():
        if isinstance(value, dict) and 'max_error' in value:
            print(f"  {key}: max_error={value['max_error']:.6e}, rmse={value['rmse']:.6e}")
    
    # Save comparison results
    comparison_data = []
    for key, value in comparison.items():
        if isinstance(value, dict) and 'time' in value:
            for i in range(len(value['time'])):
                comparison_data.append({
                    'resolution': key,
                    'time': value['time'][i],
                    'conversion': value['conversion'][i],
                    'core_radius': value['core_radius'][i]
                })
    
    comp_df = pd.DataFrame(comparison_data)
    comp_df.to_csv('./results/resolution_comparison.csv', index=False)
    
    # Create sample data for additional analysis
    print("\nCreating synthetic sample data...")
    sample_data = create_sample_data(n_samples=2000)
    sample_data.to_csv('./data/sample_data.csv', index=False)
    print(f"  Generated {len(sample_data)} samples")
    
    # Save model parameters
    params = {
        'initial_radius_m': model.initial_radius,
        'temperature_K': model.temperature,
        'h2_pressure_MPa': model.h2_pressure,
        'reaction_order': model.reaction_order,
        'mechanism': model.mechanism,
        'rho_Mg_kg_m3': model.rho_Mg,
        'rho_MgH2_kg_m3': model.rho_MgH2,
        'M_Mg_g_mol': model.M_Mg,
        'M_H2_g_mol': model.M_H2,
        'M_MgH2_g_mol': model.M_MgH2,
        'A_reaction_m_s': model.A_reaction,
        'Ea_reaction_J_mol': model.Ea_reaction,
        'D0_m2_s': model.D0,
        'Ea_diff_J_mol': model.Ea_diff,
        'R_J_mol_K': model.R
    }
    
    with open('./results/model_parameters.json', 'w') as f:
        json.dump(params, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Workflow completed successfully!")
    print("=" * 60)
    print("\nOutput files:")
    print("  ./results/single_cycle.csv - Single cycle simulation data")
    print("  ./results/cycle_metrics.csv - Cycle-by-cycle metrics")
    print("  ./results/summary.json - Summary statistics")
    print("  ./results/resolution_comparison.csv - Resolution comparison data")
    print("  ./results/model_parameters.json - Model parameters")
    print("  ./data/sample_data.csv - Synthetic sample data")
    
    return analysis, model, results


if __name__ == '__main__':
    main()