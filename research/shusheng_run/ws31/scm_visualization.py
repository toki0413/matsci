#!/usr/bin/env python3
"""
Visualization scripts for SCM MgH2 Hydrogen Absorption Simulation
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
from typing import Tuple


def plot_single_cycle(results_path: str = './results/single_cycle.csv', 
                      output_path: str = './results/single_cycle_plot.png'):
    """Plot single cycle conversion vs time."""
    df = pd.read_csv(results_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Conversion vs time
    axes[0].plot(df['time_s'], df['conversion'], 'b-', linewidth=2, label='Conversion')
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='50% conversion')
    axes[0].axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='90% conversion')
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Conversion', fontsize=12)
    axes[0].set_title('Single Cycle: Conversion vs Time', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Core radius vs time
    axes[1].plot(df['time_s'], df['core_radius_m'] * 1e6, 'g-', linewidth=2, label='Core radius')
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Complete conversion')
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Core Radius (μm)', fontsize=12)
    axes[1].set_title('Single Cycle: Core Radius vs Time', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Single cycle plot saved to {output_path}")


def plot_multiple_cycles(results_path: str = './results/cycle_metrics.csv', 
                         output_path: str = './results/multiple_cycles_plot.png'):
    """Plot multiple cycle results."""
    df = pd.read_csv(results_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cycle number vs final conversion
    axes[0, 0].plot(df['cycle'], df['final_conversion'], 'bo-', linewidth=1.5, markersize=4)
    axes[0, 0].axhline(y=df['final_conversion'].mean(), color='r', linestyle='--', 
                       label=f'Mean: {df["final_conversion"].mean():.3f}')
    axes[0, 0].set_xlabel('Cycle Number', fontsize=12)
    axes[0, 0].set_ylabel('Final Conversion', fontsize=12)
    axes[0, 0].set_title('Final Conversion vs Cycle Number', fontsize=14)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cycle number vs t_90
    valid_t90 = df['t_90'].dropna()
    axes[0, 1].plot(df['cycle'], df['t_90'], 'ro-', linewidth=1.5, markersize=4)
    axes[0, 1].axhline(y=valid_t90.mean(), color='b', linestyle='--', 
                       label=f'Mean: {valid_t90.mean():.1f} s')
    axes[0, 1].set_xlabel('Cycle Number', fontsize=12)
    axes[0, 1].set_ylabel('t_90 (s)', fontsize=12)
    axes[0, 1].set_title('Time to 90% Conversion vs Cycle Number', fontsize=14)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cycle number vs average rate
    axes[1, 0].plot(df['cycle'], df['average_rate'], 'go-', linewidth=1.5, markersize=4)
    axes[1, 0].axhline(y=df['average_rate'].mean(), color='r', linestyle='--', 
                       label=f'Mean: {df["average_rate"].mean():.6e} s⁻¹')
    axes[1, 0].set_xlabel('Cycle Number', fontsize=12)
    axes[1, 0].set_ylabel('Average Rate (s⁻¹)', fontsize=12)
    axes[1, 0].set_title('Reaction Rate vs Cycle Number', fontsize=14)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Histogram of final conversions
    axes[1, 1].hist(df['final_conversion'].dropna(), bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=df['final_conversion'].mean(), color='r', linestyle='--', 
                       label=f'Mean: {df["final_conversion"].mean():.3f}')
    axes[1, 1].set_xlabel('Final Conversion', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Distribution of Final Conversions', fontsize=14)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Multiple cycles plot saved to {output_path}")


def plot_resolution_comparison(results_path: str = './results/resolution_comparison.csv', 
                               output_path: str = './results/resolution_comparison.png'):
    """Plot resolution comparison results."""
    df = pd.read_csv(results_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot conversion vs time for different resolutions
    resolutions = sorted(df['resolution'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(resolutions)))
    
    for i, res in enumerate(resolutions):
        res_df = df[df['resolution'] == res]
        # Take every nth point to avoid overcrowding
        step = max(1, len(res_df) // 500)
        ax.plot(res_df['time_s'].iloc[::step], res_df['conversion'].iloc[::step], 
                color=colors[i], linewidth=2, label=f'Resolution {res} (dt={1/res:.2f}s)')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Conversion', fontsize=12)
    ax.set_title('Resolution Comparison: Convergence Validation', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Resolution comparison plot saved to {output_path}")


def plot_parameter_sensitivity(output_path: str = './results/parameter_sensitivity.png'):
    """Plot parameter sensitivity analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Simulate with different parameters
    from scm_mgh2_simulation import MgH2SCM
    
    # Vary initial radius
    radii = [20e-6, 50e-6, 100e-6]
    for i, r in enumerate(radii):
        model = MgH2SCM(initial_radius=r, temperature=573, h2_pressure=1.0, mechanism='reaction')
        time, conversion, _ = model.simulate_cycle(t_max=3600, dt=1.0, method='rk4')
        axes[0, 0].plot(time, conversion, label=f'R = {r*1e6:.0f} μm')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Conversion')
    axes[0, 0].set_title('Effect of Particle Size')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Vary temperature
    temps = [523, 573, 623]
    for i, T in enumerate(temps):
        model = MgH2SCM(initial_radius=50e-6, temperature=T, h2_pressure=1.0, mechanism='reaction')
        time, conversion, _ = model.simulate_cycle(t_max=3600, dt=1.0, method='rk4')
        axes[0, 1].plot(time, conversion, label=f'T = {T-273.15:.0f}°C')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Conversion')
    axes[0, 1].set_title('Effect of Temperature')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Vary pressure
    pressures = [0.5, 1.0, 2.0]
    for i, P in enumerate(pressures):
        model = MgH2SCM(initial_radius=50e-6, temperature=573, h2_pressure=P, mechanism='reaction')
        time, conversion, _ = model.simulate_cycle(t_max=3600, dt=1.0, method='rk4')
        axes[1, 0].plot(time, conversion, label=f'P_H2 = {P:.1f} MPa')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Conversion')
    axes[1, 0].set_title('Effect of H2 Pressure')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Compare mechanisms
    mechanisms = ['reaction', 'diffusion']
    for i, mech in enumerate(mechanisms):
        model = MgH2SCM(initial_radius=50e-6, temperature=573, h2_pressure=1.0, mechanism=mech)
        time, conversion, _ = model.simulate_cycle(t_max=3600, dt=1.0, method='rk4')
        axes[1, 1].plot(time, conversion, label=f'{mech}-controlled')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Conversion')
    axes[1, 1].set_title('Mechanism Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Parameter sensitivity plot saved to {output_path}")


def create_composite_plot(output_path: str = './results/composite_analysis.png'):
    """Create a comprehensive composite figure."""
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Conversion curves from multiple cycles (sample)
    ax1 = plt.subplot(2, 2, 1)
    import pandas as pd
    try:
        metrics = pd.read_csv('./results/cycle_metrics.csv')
        sample_cycles = metrics['cycle'].values[::max(1, len(metrics)//10)]
        for cycle in sample_cycles[:10]:  # Plot first 10 cycles
            cycle_data = metrics[metrics['cycle'] == cycle]
            # This is a simplified representation
            ax1.plot([0, cycle_data['t_90'].values[0] if not np.isnan(cycle_data['t_90'].values[0]) else 1000, 1500],
                     [0, 0.5, 1.0], 'alpha=0.3')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Conversion')
        ax1.set_title('Sample Cycle Conversion Curves')
        ax1.grid(True, alpha=0.3)
    except:
        ax1.text(0.5, 0.5, 'Cycle Data', transform=ax1.transAxes, ha='center')
        ax1.set_title('Sample Cycle Conversion Curves')
    
    # Plot 2: Resolution comparison
    ax2 = plt.subplot(2, 2, 2)
    try:
        comp = pd.read_csv('./results/resolution_comparison.csv')
        resolutions = comp['resolution'].unique()
        for res in resolutions:
            res_data = comp[comp['resolution'] == res]
            ax2.plot(res_data['time_s'].values[:100], res_data['conversion'].values[:100], 
                    label=f'Res {res}')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Conversion')
        ax2.set_title('Resolution Comparison')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    except:
        ax2.text(0.5, 0.5, 'Resolution Data', transform=ax2.transAxes, ha='center')
        ax2.set_title('Resolution Comparison')
    
    # Plot 3: Parameter distributions
    ax3 = plt.subplot(2, 2, 3)
    try:
        metrics = pd.read_csv('./results/cycle_metrics.csv')
        ax3.hist(metrics['final_conversion'].dropna(), bins=20, edgecolor='black', alpha=0.7)
        ax3.axvline(x=metrics['final_conversion'].mean(), color='r', linestyle='--')
        ax3.set_xlabel('Final Conversion')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Final Conversion Distribution')
        ax3.grid(True, alpha=0.3)
    except:
        ax3.text(0.5, 0.5, 'Distribution Data', transform=ax3.transAxes, ha='center')
        ax3.set_title('Final Conversion Distribution')
    
    # Plot 4: Summary statistics
    ax4 = plt.subplot(2, 2, 4)
    try:
        with open('./results/summary.json', 'r') as f:
            summary = json.load(f)
        stats = summary['cycle_statistics']
        ax4.bar(['t_50', 't_90', 'X_final', 'Rate'], 
                [stats['avg_t_50'], stats['avg_t_90'], stats['avg_final_conversion'], stats['avg_rate']])
        ax4.set_ylabel('Value')
        ax4.set_title('Summary Statistics')
        ax4.tick_params(axis='x', rotation=45)
    except:
        ax4.text(0.5, 0.5, 'Summary Data', transform=ax4.transAxes, ha='center')
        ax4.set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Composite plot saved to {output_path}")


def main():
    """Main function to generate all visualizations."""
    print("Generating visualizations...")
    
    # Ensure results exist
    import os
    if not os.path.exists('./results'):
        os.makedirs('./results')
    
    # Generate plots
    plot_single_cycle()
    plot_multiple_cycles()
    plot_resolution_comparison()
    plot_parameter_sensitivity()
    create_composite_plot()
    
    print("\nAll visualizations generated successfully!")


if __name__ == '__main__':
    main()