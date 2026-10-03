#!/usr/bin/env python3
"""
Numerical Experiment: Solution Space Rigidity in Small Feedforward Networks

Tests whether the minimum network width N_c(w) needed to achieve zero violation
(generalization error ≤ 1e-3) on held-out constraint points grows with the
number of constraint samples w.

Three hypotheses tested:
- Hypothesis 1 (Computation): N_c(w) ∝ w (linear growth) → fat
- Hypothesis 2 (Geometry): N_c(w) = constant → rigid
- Hypothesis 3 (Optimization): N_c(w) ∝ log(w) (logarithmic growth) → intermediate
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import json
from typing import Tuple, List, Optional

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ==================== Configuration ====================

# Constraint function: smooth analytic function on [0,1]^2
def f(x: np.ndarray) -> np.ndarray:
    """Analytic constraint function: sin(πx1) + cos(πx2)"""
    return np.sin(np.pi * x[:, 0:1]) + np.cos(np.pi * x[:, 1:2])

def f_torch(x: torch.Tensor) -> torch.Tensor:
    """Torch version of f"""
    return torch.sin(np.pi * x[:, 0:1]) + torch.cos(np.pi * x[:, 1:2])

# Network architecture: Single hidden layer ReLU
class FeedforwardNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.hidden(x))
        return self.output(h)

# ==================== Experiment Functions ====================

def generate_samples(w: int, dim: int = 2, test_size: int = 1000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate w constraint samples and test samples."""
    # Constraint samples
    X_train = torch.rand(w, dim)
    y_train = f_torch(X_train)
    
    # Hold-out validation from training (20%)
    val_size = int(w * 0.2)
    if val_size < 5:
        val_size = 5
    indices = torch.randperm(w)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    X_train = X_train[train_idx]
    y_train = y_train[train_idx]
    
    # Test samples (uniformly distributed)
    X_test = torch.rand(test_size, dim)
    y_test = f_torch(X_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_network(
    model: nn.Module,
    X: torch.Tensor, y: torch.Tensor,
    X_val: torch.Tensor, y_val: torch.Tensor,
    epochs: int = 10000,
    lr: float = 1e-3
) -> Tuple[float, float]:
    """Train network and return training and validation max errors."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        train_pred = model(X)
        val_pred = model(X_val)
        train_max_err = torch.max(torch.abs(train_pred - y)).item()
        val_max_err = torch.max(torch.abs(val_pred - y_val)).item()
    
    return train_max_err, val_max_err

def find_nc_w(
    w: int,
    dim: int = 2,
    h_min: int = 10,
    h_max: int = 200,
    h_steps: List[int] = None,
    test_size: int = 1000
) -> Optional[int]:
    """Find minimum width h that achieves zero violation (max error ≤ 1e-3) on test set."""
    if h_steps is None:
        h_steps = [10, 20, 50, 100, 150, 200]
    
    for h in h_steps:
        # Generate samples
        X_train, y_train, X_val, y_val, X_test, y_test = generate_samples(w, dim, test_size)
        
        # Create and train network
        model = FeedforwardNet(dim, h, 1)
        train_err, val_err = train_network(model, X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        with torch.no_grad():
            test_pred = model(X_test)
            test_max_err = torch.max(torch.abs(test_pred - y_test)).item()
        
        # Check zero violation condition
        if test_max_err <= 1e-3:
            print(f"  w={w}, h={h}: train_err={train_err:.6f}, val_err={val_err:.6f}, test_max_err={test_max_err:.6f} ✓")
            return h
    
    print(f"  w={w}: No width in [{h_min}, {h_max}] achieved zero violation (best test_err={test_max_err:.6f})")
    return None  # Indicates N_c not reachable

def run_experiment() -> dict:
    """Run the full experiment across different w values."""
    results = {
        'w_values': [],
        'N_c_values': [],
        'details': []
    }
    
    w_values = [10, 50, 100, 500, 1000]
    h_min, h_max = 10, 200
    
    for w in w_values:
        print(f"\n=== Testing w = {w} ===")
        nc = find_nc_w(w, h_min=h_max, h_max=h_max)
        results['w_values'].append(w)
        results['N_c_values'].append(nc if nc is not None else h_max + 1)
        results['details'].append({'w': w, 'N_c': nc})
    
    return results

def analyze_results(results: dict) -> str:
    """Analyze results and determine which hypothesis is supported."""
    w_vals = results['w_values']
    nc_vals = results['N_c_values']
    
    analysis = []
    analysis.append(f"Experiment Results:")
    analysis.append(f"  w values: {w_vals}")
    analysis.append(f"  N_c values: {nc_vals}")
    analysis.append("")
    
    # Check for constant (rigid)
    unique_nc = set(nc for nc in nc_vals if nc is not None)
    if len(unique_nc) == 1 and None not in nc_vals:
        analysis.append(f"RIGID: N_c is constant ({list(unique_nc)[0]}) across all w values.")
        analysis.append(f"  Supports Hypothesis 2 (Geometry/Manifold).")
    elif None in nc_vals:
        # Check if all widths failed (fat)
        if all(nc is None for nc in nc_vals):
            analysis.append(f"FAT: No width in scan range achieved zero violation for any w.")
            analysis.append(f"  Solution space may be fat or function too complex.")
        else:
            analysis.append(f"Partial: Some w values reached N_c, others did not.")
    else:
        # Try to fit scaling
        w_arr = np.array(w_vals, dtype=float)
        nc_arr = np.array([nc if nc is not None else np.nan for nc in nc_vals])
        
        # Linear fit
        valid = ~np.isnan(nc_arr)
        if np.sum(valid) >= 3:
            w_valid = w_arr[valid]
            nc_valid = nc_arr[valid]
            
            # Linear: N_c = a*w + b
            coeffs_linear = np.polyfit(w_valid, nc_valid, 1)
            linear_pred = np.polyval(coeffs_linear, w_valid)
            linear_r2 = 1 - np.sum((nc_valid - linear_pred)**2) / np.sum((nc_valid - np.mean(nc_valid))**2)
            
            # Log fit: N_c = a*log(w) + b
            log_w = np.log(w_valid)
            coeffs_log = np.polyfit(log_w, nc_valid, 1)
            log_pred = np.polyval(coeffs_log, log_w)
            log_r2 = 1 - np.sum((nc_valid - log_pred)**2) / np.sum((nc_valid - np.mean(nc_valid))**2)
            
            analysis.append(f"Scaling Analysis:")
            analysis.append(f"  Linear fit: N_c = {coeffs_linear[0]:.4f}*w + {coeffs_linear[1]:.4f}, R² = {linear_r2:.4f}")
            analysis.append(f"  Log fit:    N_c = {coeffs_log[0]:.4f}*log(w) + {coeffs_log[1]:.4f}, R² = {log_r2:.4f}")
            
            if log_r2 > linear_r2 and log_r2 > 0.9:
                analysis.append(f"  Supports Hypothesis 3 (Optimization): logarithmic growth.")
            elif linear_r2 > 0.9:
                analysis.append(f"  Supports Hypothesis 1 (Computation): linear growth.")
            else:
                analysis.append(f"  Neither fit is good; data may be noisy or require more w values.")
    
    return "\n".join(analysis)

if __name__ == "__main__":
    os.makedirs("/workspace/results", exist_ok=True)
    
    print("Starting solution space rigidity experiment...")
    print(f"Function: f(x) = sin(πx₁) + cos(πx₂)")
    print(f"Width scan: h ∈ [10, 200]")
    print(f"Constraint samples w ∈ [10, 50, 100, 500, 1000]")
    print(f"Zero violation: max absolute error ≤ 1e-3 on 1000 test points\n")
    
    results = run_experiment()
    
    analysis = analyze_results(results)
    print("\n" + analysis)
    
    # Save results
    with open("/workspace/results/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open("/workspace/results/analysis.txt", "w") as f:
        f.write(analysis)
    
    print("\nResults saved to /workspace/results/")