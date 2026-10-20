#!/usr/bin/env python3
"""Bootstrap Rigidity Experiment for Small Feedforward Networks

Tests whether solution space rigidity (constant N_c in w) or floppy behavior
(growing N_c in w) emerges when fitting analytic constraints with small networks.

Three target function classes:
1. Simple analytic (rigid candidate): sin(πx), x^2, exp(-x^2)
2. Moderate complexity (phase transition candidate): sin(πx) + 0.5*sin(3πx)
3. High complexity/floppy candidate: Random Fourier series with many frequencies
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

class ReLUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)
        # Xavier initialization for stable training
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(self.activation(self.hidden(x)))

def generate_target_function(x, target_type):
    """Generate target function values based on type."""
    if target_type == "simple_sin":
        return torch.sin(torch.pi * x)
    elif target_type == "polynomial":
        return x**2 - 0.5*x + 0.1
    elif target_type = "exp_gaussian":
        return torch.exp(-((x - 0.5)**2) / 0.1)
    elif target_type == "moderate":
        return torch.sin(torch.pi * x) + 0.5 * torch.sin(3 * torch.pi * x)
    elif target_type == "complex_fourier":
        # Random Fourier series with many frequencies - floppy candidate
        freqs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        coeffs = torch.randn(5) * 0.2
        result = torch.zeros_like(x)
        for i, (f, c) in enumerate(zip(freqs, coeffs)):
            result += c * torch.sin(f * torch.pi * x)
        return result
    else:
        return torch.sin(torch.pi * x)

def generate_constraint_points(w, n_total=200, target_type="simple_sin"):
    """Generate w training constraints and held-out validation points."""
    all_points = torch.rand(n_total).view(-1, 1) * 2 - 1  # Uniform in [-1, 1]
    indices = torch.randperm(n_total)
    train_idx = indices[:w]
    held_out_idx = indices[w:w+50]  # 50 held-out points
    x_train = all_points[train_idx]
    y_train = generate_target_function(x_train, target_type)
    x_held = all_points[held_out_idx]
    y_held = generate_target_function(x_held, target_type)
    return x_train, y_train, x_held, y_held

def train_network(model, x_train, y_train, x_held, y_held, n_epochs=5000, lr=1e-3, verbose=False):
    """Train network and return training and held-out max errors."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        # Gradient clipping to prevent instability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        train_pred = model(x_train)
        held_pred = model(x_held)
        train_max_err = torch.abs(train_pred - y_train).max().item()
        held_max_err = torch.abs(held_pred - y_held).max().item()

    if verbose and epoch % 1000 == 0:
        print(f"  Epoch {epoch}: train_err={train_max_err:.6f}, held_err={held_max_err:.6f}")

    return train_max_err, held_max_err

def find_nc_w(w, width_range, n_trials=3, threshold=1e-3, target_type="simple_sin", verbose=False):
    """Find minimum width N_c(w) that achieves zero violation on held-out points."""
    for width in width_range:
        for trial in range(n_trials):
            model = ReLUNetwork(input_dim=1, hidden_dim=width)
            x_train, y_train, x_held, y_held = generate_constraint_points(w, target_type=target_type)
            train_err, held_err = train_network(model, x_train, y_train, x_held, y_held, verbose=verbose)
            if held_err <= threshold:
                if verbose:
                    print(f"    Width {width}, Trial {trial}: held_err={held_err:.6e} <= threshold")
                return width, held_err, train_err
    if verbose:
        print(f"    No width in range achieved threshold for {n_trials} trials")
    return None, None, None

def run_experiments(target_type="simple_sin", w_values=None, width_range=None, n_trials=3, threshold=1e-3):
    """Run the full experiment sweep."""
    if w_values is None:
        w_values = [5, 10, 15, 20, 30, 40, 50, 75, 100]
    if width_range is None:
        width_range = list(range(4, 51))  # Widths from 4 to 50

    results = {}
    all_errors = {}  # Store detailed errors for each (w, width, trial)

    print(f"\n{'='*60}")
    print(f"Target type: {target_type}")
    print(f"Parameters: w={w_values}, width_range={width_range[0]}-{width_range[-1]}, n_trials={n_trials}, threshold={threshold}")
    print(f"{'='*60}\n")

    for w in w_values:
        print(f"Testing w = {w}...")
        nc_w, held_err, train_err = find_nc_w(w, width_range, n_trials=n_trials, threshold=threshold, target_type=target_type, verbose=True)
        results[w] = {"N_c": nc_w, "held_error": held_err, "train_error": train_err}
        all_errors[w] = []

        if nc_w is not None:
            print(f"  N_c({w}) = {nc_w} (RIGID: held_err={held_err:.6e})\n")
        else:
            print(f"  N_c({w}) = UNATTAINABLE (N_c>{width_range[-1]}) (Floppy candidate)\n")

    # Save results
    os.makedirs('/home/project/experiments/results', exist_ok=True)
    with open('/home/project/experiments/results/nc_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary for {target_type}:")
    print(f"{'w':<6} {'N_c(w)':<12} {'Status':<25} {'Held Err':<12}")
    print("-" * 60)
    for w, nc in results.items():
        status = f"RIGID (N_c={nc['N_c']})" if nc['N_c'] is not None else "UNATTAINABLE (N_c>50)"
        err_str = f"{nc['held_error']:.6e}" if nc['held_error'] is not None else "N/A"
        print(f"{w:<6} {nc['N_c'] if nc['N_c'] else 'inf':<12} {status:<25} {err_str:<12}")
    print(f"{'='*60}\n")

    # Plot results
    try:
        w_list = list(results.keys())
        nc_list = [results[w]['N_c'] if results[w]['N_c'] is not None else None for w in w_list]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot N_c(w)
        w_valid = [w for w, nc in zip(w_list, nc_list) if nc is not None]
        nc_valid = [nc for nc in nc_list if nc is not None]
        if w_valid:
            ax1.plot(w_valid, nc_valid, 'bo-', label='N_c(w)', markersize=8)
            # Add reference lines for different scaling predictions
            ax1.axhline(y=nc_valid[0] if nc_valid else 6, color='g', linestyle='--', alpha=0.5, label='Constant (rigid)')
            ax1.plot(w_list, w_list, 'r--', alpha=0.5, label='Linear (floppy)')
        ax1.set_xlabel('Number of constraints w', fontsize=12)
        ax1.set_ylabel('Minimum width N_c(w)', fontsize=12)
        ax1.set_title('N_c(w) vs w', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot held-out error vs width for selected w
        w_plot = [20, 50, 100]
        for wp in w_plot:
            if wp in results and results[wp]['N_c'] is not None:
                # Compute held errors across widths for this w
                width_errors = []
                for width in width_range:
                    model = ReLUNetwork(input_dim=1, hidden_dim=width)
                    x_train, y_train, x_held, y_held = generate_constraint_points(wp, target_type=target_type)
                    _, held_err = train_network(model, x_train, y_train, x_held, y_held, n_epochs=2000)
                    width_errors.append(held_err)
                ax2.plot(width_range, width_errors, 'o-', label=f'w={wp}', markersize=4)
        ax2.set_xlabel('Network width N', fontsize=12)
        ax2.set_ylabel('Held-out max error', fontsize=12)
        ax2.set_title('Held-out error vs width (log scale)', fontsize=14)
        ax2.set_yscale('log')
        ax2.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/home/project/experiments/results/nc_plot.png', dpi=150)
        plt.close()
        print("Plot saved to /home/project/experiments/results/nc_plot.png")
    except Exception as e:
        print(f"Plot generation skipped: {e}")

    return results

if __name__ == '__main__':
    # Run experiment for simple analytic target (rigid hypothesis test)
    print("Starting rigidity experiment for simple analytic target...")
    results = run_experiments(
        target_type="simple_sin",
        w_values=[5, 10, 15, 20, 30, 40, 50, 75, 100],
        width_range=list(range(4, 51)),
        n_trials=3,
        threshold=1e-3
    )

    # Also test moderate complexity (phase transition hypothesis)
    print("\n\nStarting experiment for moderate complexity target...")
    results_mod = run_experiments(
        target_type="moderate",
        w_values=[5, 10, 15, 20, 30, 40, 50, 75, 100],
        width_range=list(range(4, 51)),
        n_trials=2,
        threshold=1e-3
    )

    # Save combined results
    combined = {
        "simple_sin": results,
        "moderate": results_mod
    }
    with open('/home/project/experiments/results/combined_results.json', 'w') as f:
        json.dump(combined, f, indent=2)

    print("\nAll experiments complete. Results saved to /home/project/experiments/results/")