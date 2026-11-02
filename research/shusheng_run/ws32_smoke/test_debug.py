#!/usr/bin/env python3
"""Test script to debug the simulation error."""

import numpy as np
from simulation import SimulationParams, RegimeType, DiscretizationType, FiniteDifferenceSolver

# Create basic params
params = SimulationParams(
    L=0.01,
    D=1e-9,
    k=1e-4,
    C0=0.0,
    Cs=1000,
    n_cells=10,
    n_time_steps=10,
    dt=1.0,
    max_time=10.0
)

print("Params created:")
print(f"  Cs type: {type(params.Cs)}, value: {params.Cs}")
print(f"  k type: {type(params.k)}, value: {params.k}")
print(f"  D type: {type(params.D)}, value: {params.D}")
print(f"  n_cells type: {type(params.n_cells)}, value: {params.n_cells}")

# Create solver
solver = FiniteDifferenceSolver(params)

print("\nSolver created:")
print(f"  Cs type: {type(solver.Cs)}, value: {solver.Cs}")
print(f"  k type: {type(solver.k)}, value: {solver.k}")
print(f"  C type: {type(solver.C)}, shape: {solver.C.shape}")
print(f"  C[0] type: {type(solver.C[0])}")

# Test reaction rate
print("\nTesting reaction_rate:")
try:
    result = solver.reaction_rate(solver.C)
    print(f"  result type: {type(result)}, shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
except Exception as e:
    print(f"  Error: {e}")

# Test step
print("\nTesting step:")
try:
    residual = solver.step()
    print(f"  residual: {residual}")
except Exception as e:
    import traceback
    print(f"  Error: {e}")
    traceback.print_exc()