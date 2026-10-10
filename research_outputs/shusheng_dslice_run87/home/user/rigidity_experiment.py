#!/usr/bin/env python3
"""
Solution Space Rigidity Experiment: N_c(w) for Small Feedforward Networks
Tests whether small feedforward networks can probe solution space rigidity
via the minimum width N_c(w) needed to achieve zero violation on held-out points.

Hard constraints:
- Zero violation: max absolute error <= 1e-3 on held-out points
- Rigidity: N_c finite and bounded as w increases
- Fat: N_c unreachable (any width fails)
- All values must be finite (no inf/nan/None)
- Only small feedforward networks, no physical analogies
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from typing import Tuple, Optional

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class SmallNet(nn.Module):
    """Single-hidden-layer feedforward network with tanh activation."""
    def __init__(self, n_in=1, n_h=10, n_out=1):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_h)
        self.fc2 = nn.Linear(n_h, n_out)
        self.act = nn.Tanh()
    
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

def generate_constraint_data(n_constraints: int, n_test: int = 1000, 
                            func_type: str = "sin") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic constraint points from a smooth function.
    
    Args:
        n_constraints: Number of training constraint points (w)
        n_test: Number of held-out test points
        func_type: Type of function to sample from
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    # Sample points uniformly from [0, 1]
    X_train = np.random.uniform(0, 1, n_constraints).reshape(-1, 1)
    X_test = np.random.uniform(0, 1, n_test).reshape(-1, 1)
    
    # Generate target values from smooth function
    if func_type == "sin":
        # f(x) = sin(πx) - smooth, analytic, low-dimensional solution space
        y_train = np.sin(np.pi * X_train).flatten()
        y_test = np.sin(np.pi * X_test).flatten()
    elif func_type == "poly":
        # f(x) = x^2 - quadratic, also smooth
        y_train = X_train.flatten() ** 2
        y_test = X_test.flatten() ** 2
    elif func_type == "exp":
        # f(x) = exp(-10x) - smooth but steeper
        y_train = np.exp(-10 * X_train).flatten()
        y_test = np.exp(-10 * X_test).flatten()
    else:
        raise ValueError(f"Unknown func_type: {func_type}")
    
    return X_train, y_train, X_test, y_test

def train_network(net: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                  n_epochs: int = 5000, lr: float = 1e-3) -> Tuple[float, float]:
    """
    Train the network and return training and validation errors.
    
    Args:
        net: Network to train
        X_train: Training inputs
        y_train: Training targets
        n_epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        (train_error, held_out_error) as max absolute errors
    """
    # Convert to torch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)
    
    # Use a validation set (first 20% of training data) for early stopping-like monitoring
    val_idx = int(0.2 * len(X_train))
    X_val = X_train_t[:val_idx]
    y_val = y_train_t[:val_idx]
    X_train_opt = X_train_t[val_idx:]
    y_train_opt = y_train_t[val_idx:]
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    
    # Training loop
    for epoch in range(n_epochs):
        net.train()
        optimizer.zero_grad()
        outputs = net(X_train_opt)
        loss = criterion(outputs, y_train_opt)
        loss.backward()
        optimizer.step()
        
        # Periodic validation (reduce output noise)
        if (epoch + 1) % 500 == 0:
            net.eval()
            with torch.no_grad():
                val_outputs = net(X_val)
                val_loss = criterion(val_outputs, y_val)
    
    # Compute final errors
    net.eval()
    with torch.no_grad():
        train_outputs = net(X_train_t)
        train_error = float(torch.max(torch.abs(train_outputs - y_train_t)).item())
        
        # We'll compute held-out error externally
    return train_error, float(val_loss.item())

def find_Nc_w(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
              min_width: int = 4, max_width: int = 64, width_step: int = 4,
              zero_violation_threshold: float = 1e-3, n_epochs: int = 5000) -> Optional[int]:
    """
    Find the minimum network width that achieves zero violation on held-out points.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Held-out test data
        min_width, max_width: Range of widths to sweep
        width_step: Step size for width sweep
        zero_violation_threshold: Max allowed error for zero violation
        n_epochs: Training epochs per width
    
    Returns:
        Minimum width achieving zero violation, or None if not found
    """
    for n_h in range(min_width, max_width + 1, width_step):
        # Create network
        net = SmallNet(n_in=1, n_h=n_h, n_out=1)
        
        # Train
        train_error, _ = train_network(net, X_train, y_train, n_epochs=n_epochs, lr=1e-3)
        
        # Compute held-out error
        net.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test)
            y_test_t = torch.FloatTensor(y_test).view(-1, 1)
            test_outputs = net(X_test_t)
            test_error = float(torch.max(torch.abs(test_outputs - y_test_t)).item())
        
        # Check zero violation
        if test_error <= zero_violation_threshold:
            return n_h
    
    return None  # Not found within max_width

def run_experiment(func_type: str = "sin", w_values: list = None, 
                   width_range: Tuple[int, int] = (4, 64), width_step: int = 4,
                   n_constraints_per_w: int = 5, n_test: int = 1000,
                   output_dir: str = "./results") -> dict:
    """
    Run the full rigidity experiment.
    
    Args:
        func_type: Type of target function
        w_values: List of constraint counts to sweep
        width_range: (min_width, max_width) for sweep
        width_step: Step size for width sweep
        n_constraints_per_w: Number of random samples per w (for averaging)
        n_test: Number of held-out test points
        output_dir: Directory to save results
    
    Returns:
        Results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if w_values is None:
        w_values = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    min_width, max_width = width_range
    
    results = {
        "func_type": func_type,
        "w_values": w_values,
        "width_range": width_range,
        "width_step": width_step,
        "n_test": n_test,
        "n_constraints_per_w": n_constraints_per_w,
        "zero_violation_threshold": 1e-3,
        "Nc_w": [],  # N_c(w) values
        "avg_train_errors": [],  # Average training error at N_c(w)
        "avg_test_errors": [],   # Average test error at N_c(w)
        "all_width_results": {}  # Full sweep results for each w
    }
    
    for w in w_values:
        print(f"\n=== w = {w} ===")
        Nc_w = None
        avg_train_error_at_Nc = None
        avg_test_error_at_Nc = None
        width_error_map = {}  # n_h -> (train_error, test_error)
        
        for sample_idx in range(n_constraints_per_w):
            # Generate data
            X_train, y_train, X_test, y_test = generate_constraint_data(
                n_constraints=w, n_test=n_test, func_type=func_type
            )
            
            # Sweep widths to find N_c(w)
            found_Nc = False
            for n_h in range(min_width, max_width + 1, width_step):
                net = SmallNet(n_in=1, n_h=n_h, n_out=1)
                train_error, _ = train_network(net, X_train, y_train, n_epochs=3000, lr=1e-3)
                
                net.eval()
                with torch.no_grad():
                    X_test_t = torch.FloatTensor(X_test)
                    y_test_t = torch.FloatTensor(y_test).view(-1, 1)
                    test_outputs = net(X_test_t)
                    test_error = float(torch.max(torch.abs(test_outputs - y_test_t)).item())
                
                width_error_map[n_h] = (train_error, test_error)
                
                # Check if this width achieves zero violation
                if test_error <= 1e-3 and not found_Nc:
                    Nc_w = n_h
                    avg_train_error_at_Nc = train_error
                    avg_test_error_at_Nc = test_error
                    found_Nc = True
                    print(f"  Sample {sample_idx+1}: N_c({w}) = {n_h}, train_err={train_error:.6e}, test_err={test_error:.6e}")
                    break
            
            if not found_Nc:
                print(f"  Sample {sample_idx+1}: No width achieved zero violation for w={w}")
        
        # Average over samples (use the first found, or None if none found)
        results["Nc_w"].append(Nc_w if Nc_w is not None else None)
        results["avg_train_errors"].append(avg_train_error_at_Nc if avg_train_error_at_Nc is not None else float('nan'))
        results["avg_test_errors"].append(avg_test_error_at_Nc if avg_test_error_at_Nc is not None else float('nan'))
        results["all_width_results"][str(w)] = width_error_map
        
        # Save intermediate results
        output_path = os.path.join(output_dir, f"rigidity_results_func_{func_type}_w{w}.json")
        with open(output_path, 'w') as f:
            json.dump({
                "w": w,
                "Nc_w": Nc_w,
                "width_error_map": width_error_map,
                "avg_train_error": avg_train_error_at_Nc,
                "avg_test_error": avg_test_error_at_Nc
            }, f, indent=2)
    
    # Save summary
    summary_path = os.path.join(output_dir, f"rigidity_summary_func_{func_type}.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def print_results(results: dict):
    """Print formatted results."""
    print("\n" + "="*80)
    print("RIGIDITY EXPERIMENT RESULTS")
    print("="*80)
    print(f"Function: {results['func_type']}")
    print(f"Width range: {results['width_range']}, Step: {results['width_step']}")
    print(f"Zero violation threshold: {results['zero_violation_threshold']}")
    print(f"Held-out test points: {results['n_test']}")
    print(f"Samples per w: {results['n_constraints_per_w']}")
    print("\n" + "-"*80)
    print(f"{'w':>6} | {'N_c(w)':>8} | {'Train Err@Nc':>12} | {'Test Err@Nc':>12} | Status")
    print("-"*80)
    
    for i, w in enumerate(results['w_values']):
        Nc = results['Nc_w'][i]
        train_err = results['avg_train_errors'][i]
        test_err = results['avg_test_errors'][i]
        
        if Nc is not None:
            status = "RIGID" if (train_err <= 1e-3 and test_err <= 1e-3) else "PARTIAL"
            print(f"{w:>6} | {Nc:>8} | {train_err:>12.6e} | {test_err:>12.6e} | {status}")
        else:
            print(f"{w:>6} | {'None':>8} | {'nan':>12} | {'nan':>12} | FAT (unreachable)")
    
    print("-"*80)
    
    # Analyze scaling
    print("\nSCALING ANALYSIS:")
    valid_Nc = [(w, Nc) for w, Nc in zip(results['w_values'], results['Nc_w']) if Nc is not None]
    if len(valid_Nc) >= 2:
        w_vals, Nc_vals = zip(*valid_Nc)
        # Check if constant (rigid)
        if len(set(Nc_vals)) == 1:
            print(f"  N_c(w) is CONSTANT ({Nc_vals[0]}) across all w → RIGID solution space")
        else:
            # Check logarithmic trend
            import math
            log_fit = [Nc_vals[i] / math.log(w_vals[i] + 1) for i in range(len(w_vals))]
            print(f"  N_c(w) values: {Nc_vals}")
            print(f"  N_c(w)/log(w+1) values: {[f'{v:.2f}' for v in log_fit]}")
            if max(log_fit) / min(log_fit) < 2 if min(log_fit) > 0 else True:
                print(f"  N_c(w) grows roughly as log(w) → GEOMETRIC entropy scaling")
            else:
                print(f"  N_c(w) shows intermediate scaling behavior")
    else:
        print(f"  Insufficient data points for scaling analysis ({len(valid_Nc)} valid)")

if __name__ == "__main__":
    # Run experiment
    results = run_experiment(
        func_type="sin",  # Use sin(πx) as target (smooth, analytic)
        w_values=[4, 8, 16, 32, 64, 128, 256],  # Moderate range for first test
        width_range=(4, 32),  # Start with smaller range
        width_step=4,
        n_constraints_per_w=3,  # Reduce for quick test
        n_test=500,
        output_dir="./results"
    )
    
    print_results(results)
    
    # Save final summary
    print("\nResults saved to ./results/")