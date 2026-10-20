#!/usr/bin/env python3
"""Experiment for testing solution space rigidity via MLP capacity scanning."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MLP(nn.Module):
    """Small feedforward network with configurable hidden width."""
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, num_layers=2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def generate_ode_constraints(n_samples, n_out=10, x_range=(-1, 1), freq=np.pi):
    """
    Generate ODE constraints: f''(x) + omega^2 * f(x) = 0 (harmonic oscillator).
    The solution space is 2-dimensional: f(x) = A*cos(omega*x) + B*sin(omega*x).
    This is a RIGID solution space - we expect N_c(w) to be constant.
    """
    x_train = np.linspace(x_range[0], x_range[1], n_samples)
    x_out = np.linspace(x_range[0], x_range[1], n_out)
    
    # True solution: f(x) = cos(omega*x) (one particular solution)
    # The constraint is f'' + omega^2*f = 0
    # We'll sample the ODE residual at training points
    
    # For training: we need to satisfy f''(x_i) + omega^2*f(x_i) = 0
    # We'll use finite differences for the second derivative
    h = x_train[1] - x_train[0]
    f_true = np.cos(freq * x_train)
    f_true_out = np.cos(freq * x_out)
    
    # Approximate second derivative using central differences
    f_double_prime = np.zeros_like(f_true)
    for i in range(1, len(f_true) - 1):
        f_double_prime[i] = (f_true[i+1] - 2*f_true[i] + f_true[i-1]) / (h**2)
    # Handle boundaries with forward/backward differences
    f_double_prime[0] = (f_true[2] - 2*f_true[1] + f_true[0]) / (h**2)
    f_double_prime[-1] = (f_true[-1] - 2*f_true[-2] + f_true[-3]) / (h**2)
    
    # The constraint is that the residual should be zero
    # We'll train the network to make the residual close to zero
    # The target for the network output is the true solution
    # (since the true solution satisfies the ODE)
    
    return (
        torch.tensor(x_train, dtype=torch.float32).view(-1, 1),
        torch.tensor(f_double_prime + freq**2 * f_true, dtype=torch.float32).view(-1, 1),  # target residual should be 0
        torch.tensor(x_out, dtype=torch.float32).view(-1, 1),
        torch.tensor(np.zeros_like(x_out), dtype=torch.float32).view(-1, 1),  # target residual at out points
        f_true_out  # true solution for reference
    )

def generate_polynomial_constraints(n_samples, n_out=10, degree=5, x_range=(-1, 1)):
    """
    Generate polynomial interpolation constraints: f(x_i) = p(x_i) where p is a degree-5 polynomial.
    The solution space is infinite-dimensional (any function passing through w points).
    This is a FAT solution space - we expect N_c(w) to grow with w.
    """
    x_train = np.linspace(x_range[0], x_range[1], n_samples)
    x_out = np.linspace(x_range[0], x_range[1], n_out)
    
    # True polynomial: p(x) = 1 + 0.5*x - 0.3*x^2 + 0.2*x^3 - 0.1*x^4 + 0.05*x^5
    true_coeffs = [1.0, 0.5, -0.3, 0.2, -0.1, 0.05]
    y_train = sum(c * (x_train ** i) for i, c in enumerate(true_coeffs))
    y_out = sum(c * (x_out ** i) for i, c in enumerate(true_coeffs))
    
    return (
        torch.tensor(x_train, dtype=torch.float32).view(-1, 1),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
        torch.tensor(x_out, dtype=torch.float32).view(-1, 1),
        torch.tensor(y_out, dtype=torch.float32).view(-1, 1)
    )

def generate_oscillatory_constraints(n_samples, n_out=10, freq=5, x_range=(-1, 1)):
    """Generate oscillatory (sinusoidal) constraints with high frequency."""
    x_train = np.linspace(x_range[0], x_range[1], n_samples)
    x_out = np.linspace(x_range[0], x_range[1], n_out)
    
    y_train = np.sin(freq * np.pi * x_train)
    y_out = np.sin(freq * np.pi * x_out)
    
    return (
        torch.tensor(x_train, dtype=torch.float32).view(-1, 1),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
        torch.tensor(x_out, dtype=torch.float32).view(-1, 1),
        torch.tensor(y_out, dtype=torch.float32).view(-1, 1)
    )

def train_network(model, x_train, y_train, x_out, y_out, epochs=5000, lr=1e-3):
    """Train network and return max absolute error on out-of-sample points."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate on out-of-sample
    with torch.no_grad():
        y_out_pred = model(x_out)
        max_error = torch.max(torch.abs(y_out_pred - y_out)).item()
    
    # Also compute train MSE for reference
    with torch.no_grad():
        train_mse = criterion(model(x_train), y_train).item()
    
    return max_error, train_mse

def find_min_width(constraint_gen, width_range=list(range(8, 128, 8)), 
                   n_out=10, max_epochs=8000, tol=1e-3):
    """Find minimum width that achieves zero violation on held-out points."""
    min_width = None
    errors = []
    train_mses = []
    
    for width in width_range:
        # Generate constraints
        x_train, y_train, x_out, y_out = constraint_gen(n_samples=20, n_out=n_out)
        
        # Create and train network
        model = MLP(hidden_dim=width)
        error, train_mse = train_network(model, x_train, y_train, x_out, y_out, 
                                         epochs=max_epochs, lr=1e-3)
        errors.append(error)
        train_mses.append(train_mse)
        
        if error <= tol:
            if min_width is None:
                min_width = width
            print(f"  Width={width:3d}: train_mse={train_mse:.6e}, out_error={error:.6e} (VIOLATION OK)")
        else:
            print(f"  Width={width:3d}: train_mse={train_mse:.6e}, out_error={error:.6e} (VIOLATION)")
    
    return min_width, errors, train_mses

def scan_Nc_w(constraint_gen, w_values, n_out=15, width_range=list(range(8, 128, 8)), 
              max_epochs=8000, tol=1e-3):
    """Scan N_c(w) for different values of w (number of constraint samples)."""
    Nc_results = []
    all_errors = []
    
    print(f"\n  Scanning w values: {w_values}")
    for w in w_values:
        print(f"\n  w = {w} (constraint sample count):")
        min_width, errors, train_mses = find_min_width(
            constraint_gen, 
            width_range=width_range, 
            n_out=n_out, 
            max_epochs=max_epochs, 
            tol=tol,
            n_samples=w  # pass n_samples to constraint_gen
        )
        if min_width is None:
            Nc_results.append(float('inf'))
            print(f"  N_c({w}) = INF (no width achieved zero violation, max error: {max(errors):.6e})")
        else:
            Nc_results.append(min_width)
            print(f"  N_c({w}) = {min_width}")
        all_errors.append(errors)
    
    return Nc_results, all_errors

def main():
    print("=" * 70)
    print("Testing Solution Space Rigidity via MLP Capacity Scanning")
    print("=" * 70)
    
    # Varying w (constraint sample count)
    w_values = [6, 8, 12, 16, 20, 24, 30]
    width_range = list(range(8, 64, 8))  # narrower range for faster testing
    
    # Test case 1: ODE constraints (rigid solution space - 2D)
    print("\n" + "=" * 70)
    print("TEST CASE 1: ODE Constraints (Harmonic Oscillator)")
    print("  Theory: Solution space is 2D (A*cos + B*sin) - RIGID")
    print("=" * 70)
    Nc_ode, errors_ode = scan_Nc_w(
        generate_ode_constraints, 
        w_values=w_values, 
        n_out=15, 
        width_range=width_range,
        max_epochs=6000,
        tol=1e-3
    )
    
    # Test case 2: Polynomial interpolation (fat solution space - infinite dimensional)
    print("\n" + "=" * 70)
    print("TEST CASE 2: Polynomial Interpolation Constraints")
    print("  Theory: Any function through w points - FAT (infinite-dimensional)")
    print("=" * 70)
    Nc_poly, errors_poly = scan_Nc_w(
        generate_polynomial_constraints, 
        w_values=w_values, 
        n_out=15, 
        width_range=width_range,
        max_epochs=6000,
        tol=1e-3
    )
    
    # Test case 3: Oscillatory (high frequency - challenging landscape)
    print("\n" + "=" * 70)
    print("TEST CASE 3: High-Frequency Oscillatory Constraints")
    print("  Theory: sin(5*pi*x) - requires sufficient capacity to resolve frequency")
    print("=" * 70)
    Nc_osc, errors_osc = scan_Nc_w(
        generate_oscillatory_constraints, 
        w_values=w_values, 
        n_out=15, 
        width_range=width_range,
        max_epochs=6000,
        tol=1e-3
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF N_c(w) VALUES")
    print("=" * 70)
    print(f"{'w':>6} {'N_c(ODE)':>12} {'N_c(poly)':>12} {'N_c(osc)':>12}")
    print("-" * 45)
    for i, w in enumerate(w_values):
        print(f"{w:>6} {Nc_ode[i]:>12} {Nc_poly[i]:>12} {Nc_osc[i]:>12}")
    
    # Save results
    results = {
        'w_values': w_values,
        'Nc_ode': Nc_ode,
        'Nc_poly': Nc_poly,
        'Nc_osc': Nc_osc
    }
    np.savez('/tmp/hypothesis_test_results', **results)
    print("\nResults saved to /tmp/hypothesis_test_results.npz")

if __name__ == '__main__':
    main()