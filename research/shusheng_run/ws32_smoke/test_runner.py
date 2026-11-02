#!/usr/bin/env python3
"""Test script to debug the SimulationRunner."""

import numpy as np
from simulation import SimulationParams, RegimeType, DiscretizationType, SimulationRunner

# Create base params
base_params = SimulationParams(
    L=0.01,
    D=1e-9,
    k=1e-4,
    C0=0.0,
    Cs=1000,
    n_cells=100,
    n_time_steps=500,
    dt=1.0,
    max_time=500.0
)

# Create runner
runner = SimulationRunner(base_params)

# Try a single simulation
print("Running reaction-limited simulation with finite difference...")
try:
    result = runner.run_simulation(
        regime=RegimeType.REACTION_LIMITED,
        discretization=DiscretizationType.FINITE_DIFFERENCE,
        run_id=1
    )
    print("Success!")
    print(f"Result regime_type: {result.regime_type}")
    print(f"Uptake series length: {len(result.uptake_series)}")
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()