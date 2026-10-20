#!/usr/bin/env python3
"""
Solution Space Rigidity Experiment for Small Feedforward Networks

This experiment tests whether small feedforward networks can serve as probes
for bootstrap solution space rigidity.

Definitions:
- w = number of constraint samples (training points)
- N_c(w) = minimum hidden layer width needed to achieve "zero violation"
          on held-out constraint points
- Zero violation = max absolute error <= 1e-3 on out-of-sample constraints
- Rigid = exists small finite N_c that does NOT grow with w
- Fat = any width in scan range fails to achieve zero violation (N_c unreachable)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import json
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class FeedforwardNetwork(nn.Module):
    """Small feedforward network with configurable hidden width."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1, num_layers: int = 2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def generate_constraint_function(func_type: str, n_samples: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a constraint function and sample points.
    
    We use smooth analytic functions that have well-defined solution spaces.
    Different function types represent different constraint classes.
    """
    if func_type == "polynomial":
        # f(x) = sin(2πx) + 0.5*sin(4πx) - smooth analytic function
        x = torch.rand(n_samples, 1, device=device) * 2 - 1  # [-1, 1]
        y = torch.sin(2 * np.pi * x) + 0.5 * torch.sin(4 * np.pi * x)
    elif func_type == "rational":
        # f(x) = 1 / (1 + 25*x^2) - classic Runge function
        x = torch.rand(n_samples, 1, device=device) * 2 - 1
        y = 1.0 / (1.0 + 25.0 * x**2)
    elif func_type == "exponential":
        # f(x) = exp(-x^2) - Gaussian
        x = torch.rand(n_samples, 1, device=device) * 4 - 2
        y = torch.exp(-x**2)
    elif func_type == "sinusoidal":
        # f(x) = sin(5x) - highly oscillatory
        x = torch.rand(n_samples, 1, device=device) * np.pi
        y = torch.sin(5 * x)
    else:
        raise ValueError(f"Unknown function type: {func_type}")
    
    return x, y

def train_network(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    n_epochs: int = 5000,
    lr: float = 1e-3
) -> Tuple[float, float]:
    """
    Train a network and return training and validation errors.
    
    Returns:
        (train_max_error, val_max_error) where error is max absolute error
    """
    model.to(device)
    model.train()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate on both training and validation sets
    model.eval()
    with torch.no_grad():
        train_pred = model(x_train)
        val_pred = model(x_val)
        train_max_error = torch.max(torch.abs(train_pred - y_train)).item()
        val_max_error = torch.max(torch.abs(val_pred - y_val)).item()
    
    return train_max_error, val_max_error

def find_minimum_width(
    func_type: str,
    w: int,
    width_range: List[int],
    device: torch.device,
    n_trials: int = 3,
    zero_violation_threshold: float = 1e-3
) -> Optional[int]:
    """
    Find the minimum width that achieves zero violation on held-out points.
    
    For each width in width_range, train n_trials networks and check if any
    achieves zero violation on validation points.
    
    Returns:
        Minimum width that achieves zero violation, or None if none succeed
    """
    n_val = w // 4 if w > 4 else 5  # Validation set size
    
    for width in width_range:
        for trial in range(n_trials):
            # Generate constraint points
            x, y = generate_constraint_function(func_type, w + n_val, device)
            
            # Split into train and validation
            indices = torch.randperm(w + n_val)
            train_idx = indices[:w]
            val_idx = indices[w:w + n_val]
            
            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]
            
            # Create and train network
            model = FeedforwardNetwork(input_dim=1, hidden_dim=width, output_dim=1, num_layers=2)
            train_error, val_error = train_network(
                model, x_train, y_train, x_val, y_val, device
            )
            
            # Check if zero violation achieved on validation
            if val_error <= zero_violation_threshold:
                print(f"  w={w}, width={width}, trial={trial}: "
                      f"train_err={train_error:.6e}, val_err={val_error:.6e} "
                      f"-> ZERO VIOLATION ACHIEVED")
                return width
        
        print(f"  w={w}, width={width}: "
              f"best val_err across {n_trials} trials not achieved")
    
    return None  # No width achieved zero violation

def run_experiment(
    func_type: str,
    w_values: List[int],
    width_range: List[int],
    device: torch.device
) -> Dict[str, Any]:
    """
    Run the full experiment for a given function type.
    
    Returns:
        Dictionary containing results for all w values
    """
    results = {
        "function_type": func_type,
        "w_values": w_values,
        "width_range": width_range,
        "nc_values": [],  # N_c(w) values
        "rigid": False,
        "details": []
    }
    
    print(f"\n{'='*60}")
    print(f"Experiment: {func_type} function")
    print(f"{'='*60}")
    
    for w in w_values:
        print(f"\nw = {w}")
        nc = find_minimum_width(func_type, w, width_range, device)
        results["nc_values"].append(nc)
        
        detail = {
            "w": w,
            "N_c_w": nc,
            "rigid": nc is not None and nc < max(width_range) // 2  # heuristic for "small"
        }
        results["details"].append(detail)
        
        if nc is not None:
            print(f"N_c({w}) = {nc} (rigid candidate)")
        else:
            print(f"N_c({w}) = UNREACHABLE (fat candidate)")
    
    # Determine rigidity: N_c should be constant and small across all w
    nc_values = [nc for nc in results["nc_values"] if nc is not None]
    if len(nc_values) == len(w_values) and len(nc_values) > 0:
        # Check if N_c is roughly constant (not growing with w)
        mean_nc = np.mean(nc_values)
        std_nc = np.std(nc_values)
        if std_nc / mean_nc < 0.5 and mean_nc < max(width_range) // 2:
            results["rigid"] = True
            print(f"\nResult: RIGID family - N_c stays approximately constant at ~{mean_nc:.0f}")
        else:
            results["rigid"] = False
            print(f"\nResult: FAT or mixed - N_c varies across w values")
    else:
        results["rigid"] = False
        print(f"\nResult: FAT family - N_c unreachable for some w values")
    
    return results

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Experiment parameters
    func_type = "polynomial"  # Try different function types
    w_values = [10, 20, 50, 100, 200]  # Number of constraint samples
    width_range = list(range(4, 51, 4))  # Hidden widths from 4 to 48
    
    # Run experiment
    results = run_experiment(
        func_type=func_type,
        w_values=w_values,
        width_range=width_range,
        device=device
    )
    
    # Save results
    output_dir = "/workspace/research_outputs/shusheng_rsi_run73/results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"rigidity_experiment_{func_type}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Function type: {func_type}")
    print(f"W values: {w_values}")
    print(f"Width range: {width_range}")
    print(f"N_c(w) values: {results['nc_values']}")
    print(f"Rigid classification: {'RIGID' if results['rigid'] else 'FAT'}")

if __name__ == "__main__":
    main()