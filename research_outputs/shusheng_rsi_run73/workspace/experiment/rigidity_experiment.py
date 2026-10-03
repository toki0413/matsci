#!/usr/bin/env python3
"""
Bootstrap Solution Space Rigidity Experiment for Small Feedforward Networks

Tests whether the minimum network width N_c(w) needed to achieve zero violation
(max absolute error ≤ 1e-3) on held-out constraint points grows with the
number of constraint samples w.

Three hypotheses tested:
- Hypothesis 1 (Computation/Barron): N_c(w) = constant → RIGID
- Hypothesis 2 (Optimization/Landscape): N_c(w) ∝ w → FAT
- Hypothesis 3 (Geometry/Manifold): N_c(w) ∝ log w → INTERMEDIATE
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import sys
from typing import Tuple, List, Optional, Dict

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ==================== Configuration ====================

# Multiple constraint functions to test - all analytic on [-1, 1]
def f1(x: np.ndarray) -> np.ndarray:
    """Smooth analytic function: sin(πx) + cos(2πx)"""
    return np.sin(np.pi * x) + np.cos(2 * np.pi * x)

def f2(x: np.ndarray) -> np.ndarray:
    """Exponential: exp(-x²) (Gaussian)"""
    return np.exp(-x * x)

def f3(x: np.ndarray) -> np.ndarray:
    """Rational: 1/(1 + 25x²) (Runge-like, analytic on [-1,1])"""
    return 1.0 / (1.0 + 25 * x * x)

def f4(x: np.ndarray) -> np.ndarray:
    """Polynomial: 5x⁴ - 6x² + 1"""
    return 5 * x**4 - 6 * x**2 + 1

def f5(x: np.ndarray) -> np.ndarray:
    """Sum of sinusoids with different frequencies"""
    return np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x) + 0.3 * np.sin(10 * np.pi * x)

FUNCTIONS = {
    'f1': f1,
    'f2': f2,
    'f3': f3,
    'f4': f4,
    'f5': f5,
}

# Network architecture: Single hidden layer ReLU
class SingleLayerReLU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.hidden(x))
        return self.output(x)

# ==================== Experiment Functions ====================

def generate_samples(target_func, w: int, dim: int = 1, held_out_ratio: float = 0.2,
                     domain: Tuple[float, float] = (-1.0, 1.0)) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate w constraint samples and held-out validation samples."""
    x = np.random.uniform(domain[0], domain[1], w)
    y = target_func(x)
    
    # Shuffle and split into training and held-out validation
    indices = np.random.permutation(w)
    val_size = max(1, int(w * held_out_ratio))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    x_train = x[train_idx].reshape(-1, 1)
    y_train = y[train_idx].reshape(-1, 1)
    x_val = x[val_idx].reshape(-1, 1)
    y_val = y[val_idx].reshape(-1, 1)
    
    # Additional held-out test set for zero violation check (larger, independent)
    x_test = np.random.uniform(domain[0], domain[1], 500).reshape(-1, 1)
    y_test = target_func(x_test)
    
    return (
        torch.FloatTensor(x_train), torch.FloatTensor(y_train),
        torch.FloatTensor(x_val), torch.FloatTensor(y_val),
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

def train_network(model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor,
                  max_epochs: int = 5000, lr: float = 0.01) -> Tuple[float, float]:
    """Train network with early stopping and return training max error."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience = 100
    patience_counter = 0
    
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        # Early stopping check
        with torch.no_grad():
            train_pred = model(x_train)
            train_loss = float(criterion(train_pred, y_train).item())
        
        if train_loss < best_val_loss * 0.99:
            best_val_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Final evaluation
    with torch.no_grad():
        train_pred = model(x_train)
        train_max_err = float(torch.max(torch.abs(train_pred - y_train)).item())
    
    return train_max_err

def find_min_width(target_func, w: int, n_range: List[int], held_out_ratio: float = 0.2,
                   zero_violation_threshold: float = 1e-3, n_trials: int = 3,
                   domain: Tuple[float, float] = (-1.0, 1.0)) -> Optional[int]:
    """Find minimum width n that achieves zero violation on held-out set across trials."""
    for n in n_range:
        success = False
        for trial in range(n_trials):
            x_train, y_train, x_val, y_val, x_test, y_test = generate_samples(
                target_func, w, dim=1, held_out_ratio=held_out_ratio, domain=domain
            )
            
            model = SingleLayerReLU(input_dim=1, hidden_dim=n, output_dim=1)
            train_max_err = train_network(model, x_train, y_train, max_epochs=3000, lr=0.01)
            
            # Evaluate on held-out validation set for zero violation check
            with torch.no_grad():
                val_pred = model(x_val)
                val_max_err = float(torch.max(torch.abs(val_pred - y_val)).item())
            
            if val_max_err <= zero_violation_threshold:
                success = True
                break
        
        if success:
            return n
    
    return None  # Width not reached in scan range

def run_experiment(func_name: str = 'f1', w_values: List[int] = None,
                   n_range: List[int] = None, held_out_ratio: float = 0.2,
                   zero_violation_threshold: float = 1e-3, n_trials: int = 3) -> Dict:
    """Run the full experiment across different w values for a given function."""
    if w_values is None:
        w_values = [10, 20, 50, 100, 200, 500, 1000]
    if n_range is None:
        n_range = list(range(1, 201))  # Fine-grained scan from 1 to 200
    
    target_func = FUNCTIONS[func_name]
    results = {'w_values': [], 'N_c_values': [], 'details': []}
    
    print(f"\n{'='*60}")
    print(f"Experiment: {func_name} - {target_func.__doc__}")
    print(f"w values: {w_values}")
    print(f"Width range: {n_range[0]} to {n_range[-1]}")
    print(f"Zero violation threshold: {zero_violation_threshold}")
    print(f"Trials per (w,n): {n_trials}")
    print(f"{'='*60}\n")
    
    for w in w_values:
        print(f"Testing w = {w}...")
        N_c = find_min_width(
            target_func, w, n_range, held_out_ratio, zero_violation_threshold, n_trials
        )
        results['w_values'].append(w)
        nc_val = N_c if N_c is not None else n_range[-1] + 1  # Use max_width+1 for unreachable
        results['N_c_values'].append(nc_val)
        results['details'].append({'w': w, 'N_c': N_c})
        
        if N_c is not None:
            print(f"  N_c({w}) = {N_c} (RIGID)")
        else:
            print(f"  N_c({w}) = UNREACHED (FAT in scan range)")
        print()
    
    return results

def analyze_scaling(results: Dict) -> str:
    """Analyze the scaling of N_c(w) and determine which hypothesis is supported."""
    w_vals = np.array(results['w_values'], dtype=float)
    nc_vals = np.array(results['N_c_values'], dtype=float)
    
    analysis_lines = []
    analysis_lines.append("Scaling Analysis:")
    analysis_lines.append(f"  w values: {w_vals}")
    analysis_lines.append(f"  N_c values: {nc_vals}")
    analysis_lines.append("")
    
    # Check for rigid (constant N_c)
    unique_nc = set(nc for nc in nc_vals if nc <= max(results['N_c_values']) - 1)
    if len(unique_nc) == 1 and all(nc <= max(results['N_c_values']) - 1 for nc in nc_vals):
        constant_nc = list(unique_nc)[0]
        analysis_lines.append(f"RIGID: N_c is constant ({constant_nc}) across all w values.")
        analysis_lines.append(f"  Supports Hypothesis 1 (Computation/Barron's Theorem).")
        return "\n".join(analysis_lines)
    
    # Check for fat (unreachable in scan range)
    if all(nc > max(results['N_c_values']) - 1 for nc in nc_vals):
        analysis_lines.append(f"FAT: No width in scan range achieved zero violation for any w.")
        analysis_lines.append(f"  Solution space appears fat or function too complex for this architecture.")
        return "\n".join(analysis_lines)
    
    # Fit scaling for reachable values
    reachable_mask = nc_vals <= max(results['N_c_values']) - 1
    w_reachable = w_vals[reachable_mask]
    nc_reachable = nc_vals[reachable_mask]
    
    if len(w_reachable) < 3:
        analysis_lines.append("Insufficient data points for scaling analysis.")
        return "\n".join(analysis_lines)
    
    # Linear fit: N_c = a*w + b
    coeffs_linear = np.polyfit(w_reachable, nc_reachable, 1)
    linear_pred = np.polyval(coeffs_linear, w_reachable)
    ss_res_linear = np.sum((nc_reachable - linear_pred) ** 2)
    ss_tot_linear = np.sum((nc_reachable - np.mean(nc_reachable)) ** 2)
    linear_r2 = 1 - ss_res_linear / ss_tot_linear if ss_tot_linear > 0 else 0
    
    # Log fit: N_c = a*log(w) + b
    log_w = np.log(w_reachable)
    coeffs_log = np.polyfit(log_w, nc_reachable, 1)
    log_pred = np.polyval(coeffs_log, log_w)
    ss_res_log = np.sum((nc_reachable - log_pred) ** 2)
    ss_tot_log = np.sum((nc_reachable - np.mean(nc_reachable)) ** 2)
    log_r2 = 1 - ss_res_log / ss_tot_log if ss_tot_log > 0 else 0
    
    # Constant fit: N_c = c
    constant_val = np.mean(nc_reachable)
    ss_res_const = np.sum((nc_reachable - constant_val) ** 2)
    ss_tot_const = np.sum((nc_reachable - np.mean(nc_reachable)) ** 2)
    const_r2 = 1 - ss_res_const / ss_tot_const if ss_tot_const > 0 else 0
    
    analysis_lines.append(f"  Linear fit:    N_c = {coeffs_linear[0]:.4f}*w + {coeffs_linear[1]:.4f}, R² = {linear_r2:.4f}")
    analysis_lines.append(f"  Log fit:       N_c = {coeffs_log[0]:.4f}*log(w) + {coeffs_log[1]:.4f}, R² = {log_r2:.4f}")
    analysis_lines.append(f"  Constant fit:  N_c = {constant_val:.4f}, R² = {const_r2:.4f}")
    analysis_lines.append("")
    
    # Determine best fit
    best_r2 = max(linear_r2, log_r2, const_r2)
    if best_r2 == const_r2 and const_r2 > 0.9:
        analysis_lines.append("CONSTANT (RIGID): N_c does not grow with w.")
        analysis_lines.append("  Supports Hypothesis 1 (Computation/Barron's Theorem).")
    elif best_r2 == log_r2 and log_r2 > 0.9:
        analysis_lines.append("LOGARITHMIC (INTERMEDIATE): N_c grows as log(w).")
        analysis_lines.append("  Supports Hypothesis 3 (Geometry/Manifold Curvature).")
    elif best_r2 == linear_r2 and linear_r2 > 0.9:
        analysis_lines.append("LINEAR (FAT): N_c grows linearly with w.")
        analysis_lines.append("  Supports Hypothesis 2 (Optimization/Landscape Complexity).")
    else:
        analysis_lines.append("No clear scaling pattern observed. Data may be noisy.")
        analysis_lines.append(f"  Best R² = {best_r2:.4f} (linear: {linear_r2:.4f}, log: {log_r2:.4f}, const: {const_r2:.4f})")
    
    return "\n".join(analysis_lines)

def main():
    """Run experiments for multiple constraint functions."""
    os.makedirs('/workspace/experiment/results', exist_ok=True)
    
    # Test all five functions
    func_names = list(FUNCTIONS.keys())
    all_results = {}
    
    for func_name in func_names:
        print(f"\n{'#'*60}")
        print(f"Running experiment for {func_name}")
        print(f"{'#'*60}\n")
        
        results = run_experiment(func_name=func_name)
        all_results[func_name] = results
        
        # Analyze and print results
        analysis = analyze_scaling(results)
        print(analysis)
        print()
        
        # Save results
        with open(f'/workspace/experiment/results/{func_name}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        with open(f'/workspace/experiment/results/{func_name}_analysis.txt', 'w') as f:
            f.write(analysis)
    
    # Save combined results
    combined = {
        'functions': func_names,
        'results': all_results
    }
    with open('/workspace/experiment/results/combined_results.json', 'w') as f:
        json.dump(combined, f, indent=2)
    
    print("\n" + "#"*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("#"*60)

if __name__ == "__main__":
    main()