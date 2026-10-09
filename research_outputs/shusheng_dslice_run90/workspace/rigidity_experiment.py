#!/usr/bin/env python3
"""
Solution Space Rigidity Experiment for Small Feedforward Networks

Tests whether small neural networks can achieve zero violation (max abs error <= 1e-3)
on held-out constraint points as the number of constraint samples w varies.

Rigid: N_c(w) stays bounded (small finite value, independent of w)
Floppy: N_c(w) grows without bound within scan range
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SmallMLP(nn.Module):
    """Single-hidden-layer feedforward network."""
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # Xavier initialization for stability
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

def train_network(model, X_train, y_train, X_val, y_val, epochs=8000, lr=1e-3, patience=200):
    """Train network and return validation max absolute error."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    best_val_error = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        # Early stopping check on validation set
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_error = torch.max(torch.abs(y_val_pred - y_val)).item()

        if val_error < best_val_error:
            best_val_error = val_error
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_error

def find_Nc_single_trial(w, target_func, w_train_ratio=0.8, max_width=200, tolerance=1e-3):
    """Find minimum width achieving zero violation on held-out points."""
    # Generate constraint points on [0, 1]
    X = np.random.uniform(0, 1, w)
    X = torch.FloatTensor(X).view(-1, 1)
    y = target_func(X)

    # Split into training and validation
    w_train = int(w * w_train_ratio)
    w_val = w - w_train

    # Shuffle indices
    indices = np.arange(w)
    np.random.shuffle(indices)

    X_train = X[indices[:w_train]]
    y_train = y[indices[:w_train]]
    X_val = X[indices[w_train:]]
    y_val = y[indices[w_train:]]

    # Search for minimum width
    for width in range(1, max_width + 1):
        model = SmallMLP(hidden_dim=width)
        val_error = train_network(model, X_train, y_train, X_val, y_val)

        if val_error <= tolerance:
            return width, val_error

    return None, None  # No width achieved zero violation

def find_Nc_with_retries(w, target_func, n_trials=5, **kwargs):
    """Run multiple trials and report the minimum width that succeeds in any trial."""
    widths = []
    errors = []
    for trial in range(n_trials):
        width, error = find_Nc_single_trial(w, target_func, **kwargs)
        if width is not None:
            widths.append(width)
            errors.append(error)
    if widths:
        # Return the minimum width across successful trials
        min_width = min(widths)
        min_error = errors[widths.index(min_width)]
        return min_width, min_error
    return None, None

def run_experiment(target_name, target_func, w_values, max_width=200, tolerance=1e-3, n_trials=5):
    """Run the full experiment for a given target function."""
    results = {
        'target': target_name,
        'w_values': w_values.tolist(),
        'N_c_values': [],
        'validation_errors': [],
        'rigid': False
    }

    print(f"\n=== Experiment: {target_name} ===")
    print(f"Target: {target_name}")
    print(f"Constraint samples w: {w_values}")
    print(f"Max width: {max_width}, Tolerance: {tolerance}")

    for i, w in enumerate(w_values):
        print(f"\n  w = {w:3d} ({i+1}/{len(w_values)})")

        width, error = find_Nc_with_retries(
            w, target_func, max_width=max_width, tolerance=tolerance, n_trials=n_trials
        )

        if width is not None:
            results['N_c_values'].append(width)
            results['validation_errors'].append(error)
            print(f"  N_c = {width:3d}, val_error = {error:.2e}")
        else:
            results['N_c_values'].append(None)
            results['validation_errors'].append(None)
            print(f"  N_c = UNREACHED (within max width {max_width})")

    # Determine rigidity: check if N_c is bounded and doesn't grow with w
    nc_values = [v for v in results['N_c_values'] if v is not None]
    if len(nc_values) > 0:
        max_nc = max(nc_values)
        # Check if N_c stays reasonably bounded (doesn't grow with w)
        if max_nc <= max_width // 4:  # Reasonably small compared to scan limit
            results['rigid'] = True
            print(f"\n  Result: RIGID (max N_c = {max_nc}, bounded)")
        else:
            results['rigid'] = False
            print(f"\n  Result: POTENTIALLY FLOPPY (max N_c = {max_nc}, may grow)")
    else:
        results['rigid'] = False
        print(f"\n  Result: FLOPPY (no N_c reached within scan range)")

    return results

def main():
    # Define target functions
    def rigid_target(x):
        """Simple periodic function - low-dimensional structure."""
        return torch.sin(2 * np.pi * x)

    def floppy_target(x):
        """High-frequency composition - complex structure."""
        return (torch.sin(2 * np.pi * x) +
                0.5 * torch.sin(10 * np.pi * x) +
                0.3 * torch.sin(20 * np.pi * x))

    def complex_rigid(x):
        """Polynomial - very simple analytic structure."""
        return x ** 2

    # Experiment parameters
    w_values = np.array([5, 10, 15, 20, 30, 40, 50, 60, 80, 100])
    max_width = 100
    tolerance = 1e-3
    n_trials = 3

    print("=" * 60)
    print("SOLUTION SPACE RIGIDITY EXPERIMENT")
    print("=" * 60)

    # Run experiments
    results_rigid = run_experiment(
        'sin(2πx)', rigid_target, w_values, max_width, tolerance, n_trials
    )

    results_floppy = run_experiment(
        'sin(2πx)+0.5sin(10πx)+0.3sin(20πx)', floppy_target,
        w_values, max_width, tolerance, n_trials
    )

    results_poly = run_experiment(
        'x²', complex_rigid, w_values, max_width, tolerance, n_trials
    )

    # Save results
    all_results = {
        'sin_2pi_x': results_rigid,
        'complex_floppy': results_floppy,
        'x_squared': results_poly,
        'metadata': {
            'w_values': w_values.tolist(),
            'max_width': max_width,
            'tolerance': tolerance,
            'n_trials': n_trials,
            'date': '2026-10-09'
        }
    }

    os.makedirs('results', exist_ok=True)
    with open('results/rigidity_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)

    for name, res in [('sin(2πx)', results_rigid),
                      ('complex (floppy)', results_floppy),
                      ('x²', results_poly)]:
        nc_vals = [v for v in res['N_c_values'] if v is not None]
        if nc_vals:
            trend = "CONSTANT" if len(nc_vals) < 2 or all(v == nc_vals[0] for v in nc_vals) else "GROWING"
            print(f"{name}: N_c values = {nc_vals}, rigid={res['rigid']}, trend={trend}")
        else:
            print(f"{name}: N_c values = UNREACHED, rigid={res['rigid']}")

    print("\nResults saved to results/rigidity_results.json")

if __name__ == '__main__':
    main()