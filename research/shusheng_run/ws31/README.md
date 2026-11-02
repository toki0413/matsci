# SCM MgH2 Hydrogen Absorption Simulation

This repository implements a computational workflow using the **Shrinking Core Model (SCM)** for MgH₂ hydrogen absorption. The simulation includes:

- Physically-grounded SCM with explicit kinetic parameters from literature
- 50+ cycles of simulation with parameter variations
- Independent validation through resolution comparison
- Comprehensive analysis and visualization

## Requirements

- Python 3.6+
- numpy
- pandas
- matplotlib

## Workflow

```bash
# 1. Run the main simulation
python scm_mgh2_simulation.py

# 2. Generate visualizations
python scm_visualization.py

# 3. Perform comprehensive analysis
python scm_analysis.py
```

## Output Files

### Data Files
- `./data/sample_data.csv` - Synthetic sample data (2000 samples)

### Results Files
- `./results/single_cycle.csv` - Single cycle simulation data
- `./results/cycle_metrics.csv` - Cycle-by-cycle metrics
- `./results/summary.json` - Summary statistics
- `./results/resolution_comparison.csv` - Resolution comparison data
- `./results/model_parameters.json` - Model parameters
- `./results/analysis_report.txt` - Comprehensive analysis report

### Visualization Files
- `./results/single_cycle_plot.png` - Single cycle conversion vs time
- `./results/multiple_cycles_plot.png` - Multiple cycle statistics
- `./results/resolution_comparison.png` - Resolution convergence validation
- `./results/parameter_sensitivity.png` - Parameter sensitivity analysis
- `./results/composite_analysis.png` - Composite analysis figure

## Model Description

The SCM describes the reaction of Mg with H₂ to form MgH₂, where the reaction front moves inward from the particle surface. Key features:

- **Spherical particles** with initial radius of 50 μm
- **Isothermal conditions** at 573 K (300°C)
- **H₂ pressure** of 1.0 MPa
- **Reaction-controlled mechanism** (typical for MgH₂ at moderate temperatures)
- **Arrhenius temperature dependence** for kinetic parameters

### Kinetic Parameters (from literature)
- Pre-exponential factor (reaction): 1.5 × 10⁸ m/s
- Activation energy (reaction): 45 kJ/mol
- Pre-exponential factor (diffusion): 1.2 × 10⁻⁵ m²/s
- Activation energy (diffusion): 65 kJ/mol

## Validation

The simulation includes independent validation through:

1. **Resolution comparison**: Testing different time step resolutions to ensure numerical convergence
2. **Physical consistency checks**: Conversion range, monotonic radius decrease, reasonable final conversion
3. **Multiple cycling**: 52 cycles with parameter variations to simulate real-world behavior

## References

- B. Bogdanovic, M. Schwickardi, J. Alloys Compd. (1997) - MgH₂ hydrogenation kinetics
- Standard material properties for Mg and MgH₂

## License

MIT License