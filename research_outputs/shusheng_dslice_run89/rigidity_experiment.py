#!/usr/bin/env python3
"""Bootstrap Rigidity Experiment for Small Feedforward Networks

Tests whether solution space rigidity (constant N_c in w) or floppy behavior
(growing N_c in w) emerges when fitting analytic constraints with small networks.

Three target function classes:
1. Simple analytic (rigid candidate): sin(πx)
2. Moderate complexity (phase transition candidate): sin(πx) + 0.5*sin(10πx)
3. High complexity/floppy candidate: sin(πx) + 0.5*sin(10πx) + 0.3*sin(50πx)

The experiment measures N_c(w) = minimum width achieving zero violation
(max abs error <= 1e-3) on held-out points, as a function of constraint count w.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import sys

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class ReLUNetwork(nn.Module):
    """1-hidden-layer ReLU network."""
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
    elif target_type == "moderate":
        return torch.sin(torch.pi * x) + 0.5 * torch.sin(10 * torch.pi * x)
    elif target_type == "complex":
        return torch.sin(torch.pi * x) + 0.5 * torch.sin(10 * torch.pi * x) + 0.3 * torch.sin(50 * torch.pi * x)
    else:
        return torch.sin(torch.pi * x)


def generate_constraint_points(w, n_total=200, target_type="simple_sin"):
    """Generate w training constraints and held-out validation points.
    
    Args:
        w: Number of training constraint points
        n_total: Total number of points to sample from
        target_type: Type of target function
    
    Returns:
        x_train, y_train, x_held, y_held
    """
    # Sample points uniformly from [-1, 1]
    all_points = torch.rand(n_total).view(-1, 1) * 2 - 1
    indices = torch.randperm(n_total)
    
    # Training points
    train_idx = indices[:w]
    # Held-out validation points (after training points)
    held_out_idx = indices[w:w+50]  # 50 held-out points
    
    x_train = all_points[train_idx]
    y_train = generate_target_function(x_train, target_type)
    x_held = all_points[held_out_idx]
    y_held = generate_target_function(x_held, target_type)
    
    return x_train, y_train, x_held, y_held


def train_network(model, x_train, y_train, x_held, y_held, n_epochs=5000, lr=1e-3, verbose=False):
    """Train network and return training and held-out max errors.
    
    Uses Adam optimizer with cosine annealing and gradient clipping.
    """
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
        print(f"  Epoch {epoch}: train_err={train_max_err:.6e}, held_err={held_max_err:.6e}")

    return train_max_err, held_max_err


def find_nc_w(w, width_range, n_trials=3, threshold=1e-3, target_type="simple_sin", verbose=False):
    """Find minimum width N_c(w) that achieves zero violation on held-out points.
    
    Args:
        w: Number of training constraints
        width_range: List of widths to test
        n_trials: Number of random seed trials per width
        threshold: Zero violation threshold (max abs error)
        target_type: Type of target function
        verbose: Whether to print debug info
    
    Returns:
        nc_w (int or None): Minimum width achieving threshold, or None if not attainable
        held_err (float or None): Held-out error at the minimum width
        train_err (float or None): Training error at the minimum width
    """
    for width in width_range:
        for trial in range(n_trials):
            # Set seed for reproducibility across trials
            torch.manual_seed(42 + w * 1000 + width * 10 + trial)
            np.random.seed(42 + w * 1000 + width * 10 + trial)
            
            model = ReLUNetwork(input_dim=1, hidden_dim=width)
            x_train, y_train, x_held, y_held = generate_constraint_points(w, target_type=target_type)
            train_err, held_err = train_network(model, x_train, y_train, x_held, y_held, verbose=False)
            
            if held_err <= threshold:
                if verbose:
                    print(f"    Width {width}, Trial {trial}: held_err={held_err:.6e} <= threshold {threshold}")
                return width, held_err, train_err
    
    if verbose:
        print(f"    No width in range {width_range[0]}-{width_range[-1]} achieved threshold for {n_trials} trials")
    return None, None, None


def run_experiments(target_type="simple_sin", w_values=None, width_range=None, n_trials=3, threshold=1e-3, output_dir=None):
    """Run the full experiment sweep.
    
    Args:
        target_type: Type of target function
        w_values: List of constraint counts to test
        width_range: List of network widths to test
        n_trials: Number of random seed trials per (w, width) pair
        threshold: Zero violation threshold
        output_dir: Directory to save results
    
    Returns:
        results: Dictionary with N_c(w) for each w
    """
    if w_values is None:
        w_values = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
    if width_range is None:
        width_range = list(range(4, 101))  # Widths from 4 to 100

    results = {}
    all_errors = {}  # Store detailed errors for each (w, width, trial)

    print(f"\n{'='*70}")
    print(f"Target type: {target_type}")
    print(f"Parameters: w={w_values}, width_range={width_range[0]}-{width_range[-1]}, n_trials={n_trials}, threshold={threshold}")
    print(f"{'='*70}\n")

    for w in w_values:
        print(f"Testing w = {w}...")
        nc_w, held_err, train_err = find_nc_w(w, width_range, n_trials=n_trials, threshold=threshold, target_type=target_type, verbose=False)
        results[w] = {"N_c": nc_w, "held_error": held_err, "train_error": train_err}
        all_errors[w] = []

        if nc_w is not None:
            print(f"  N_c({w}) = {nc_w} (RIGID: held_err={held_err:.6e})\n")
        else:
            print(f"  N_c({w}) = UNATTAINABLE (N_c>{width_range[-1]}) (Floppy candidate)\n")

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "nc_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        # Print summary
        print(f"\n{'='*70}")
        print(f"Summary for {target_type}:")
        print(f"{'w':<6} {'N_c(w)':<12} {'Status':<25} {'Held Err':<12}")
        print("-" * 70)
        for w, nc in sorted(results.items()):
            status = f"RIGID (N_c={nc['N_c']})" if nc['N_c'] is not None else "UNATTAINABLE (N_c>100)"
            err_str = f"{nc['held_error']:.6e}" if nc['held_error'] is not None else "N/A"
            print(f"{w:<6} {nc['N_c'] if nc['N_c'] else 'inf':<12} {status:<25} {err_str:<12}")
        print(f"{'='*70}\n")

    return results


def fit_scaling_powerlaw(w_values, nc_values):
    """Fit N_c(w) = a * w^b + c using least squares on log scale.
    
    Returns:
        a, b, c: Fitted parameters
        r_squared: R-squared value
    """
    # Filter out None values
    valid_pairs = [(w, nc) for w, nc in zip(w_values, nc_values) if nc is not None]
    if len(valid_pairs) < 3:
        return None, None, None, 0.0
    
    w_valid, nc_valid = zip(*valid_pairs)
    w_valid = np.array(w_valid, dtype=float)
    nc_valid = np.array(nc_valid, dtype=float)
    
    # Fit log(N_c - c) = log(a) + b * log(w) for different c values
    best_b = None
    best_a = None
    best_c = 0
    best_r2 = 0.0
    
    # Try different offset values
    for c in np.linspace(0, max(nc_valid) * 0.5, 20):
        nc_shifted = nc_valid - c
        if np.any(nc_shifted <= 0):
            continue
        log_w = np.log(w_valid)
        log_nc = np.log(nc_shifted)
        
        # Linear fit
        coeffs = np.polyfit(log_w, log_nc, 1)
        b = coeffs[0]
        a = np.exp(coeffs[1])
        
        # Compute R-squared
        log_nc_pred = np.polyval(coeffs, log_w)
        ss_res = np.sum((log_nc - log_nc_pred) ** 2)
        ss_tot = np.sum((log_nc - np.mean(log_nc)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        if r2 > best_r2:
            best_r2 = r2
            best_b = b
            best_a = a
            best_c = c
    
    if best_b is None:
        return None, None, None, 0.0
    
    return best_a, best_b, best_c, best_r2


if __name__ == "__main__":
    # Run experiments for different target types
    output_dir = "/workspace/research_outputs/shusheng_dslice_run89/experiments/results"
    
    target_types = ["simple_sin", "moderate", "complex"]
    all_results = {}
    
    for target_type in target_types:
        print(f"\n{'#'*70}")
        print(f"EXPERIMENT: {target_type}")
        print(f"{'#'*70}\n")
        
        results = run_experiments(
            target_type=target_type,
            w_values=[10, 20, 50, 100, 200, 300, 500, 750, 1000],
            width_range=list(range(4, 101)),
            n_trials=3,
            threshold=1e-3,
            output_dir=output_dir
        )
        all_results[target_type] = results
        
        # Fit scaling law for this target type
        w_vals = list(results.keys())
        nc_vals = [results[w]["N_c"] for w in w_vals]
        a, b, c, r2 = fit_scaling_powerlaw(w_vals, nc_vals)
        
        if a is not None:
            print(f"\nScaling fit for {target_type}: N_c(w) = {a:.2f} * w^{b:.2f} + {c:.2f}")
            print(f"R-squared: {r2:.4f}")
            if abs(b) < 0.1:
                print("-> Consistent with RIGID (constant N_c)")
            elif abs(b - 0.5) < 0.1:
                print("-> Consistent with OPTIMIZATION hypothesis (sqrt(w) scaling)")
            elif abs(b - 1.0) < 0.1:
                print("-> Consistent with linear scaling (strongly floppy)")
            else:
                print(f"-> Scaling exponent b = {b:.2f}")
        else:
            print(f"\nCould not fit scaling law for {target_type} (insufficient data points)")

    # Save combined results
    combined_output = os.path.join(output_dir, "combined_results.json")
    with open(combined_output, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll experiments complete. Results saved to {output_dir}/")