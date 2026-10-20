#!/usr/bin/env python3
"""
Numerical experiment for solution space rigidity in small feedforward networks.

Tests whether the minimum network width N_c(w) needed to achieve zero violation
on held-out constraint points scales as constant, log(w), or linear in w.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Optional
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class FeedforwardNet(nn.Module):
    """Simple feedforward network with one hidden layer."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h = self.activation(self.hidden(x))
        return self.output(h)

def generate_constraints(
    n_samples: int,
    x_range: Tuple[float, float] = (0.0, 1.0),
    target_func=None,
    noise_level: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate constraint points (x, y).
    
    Parameters:
    - n_samples: Number of constraint points
    - x_range: Range of x values
    - target_func: Function y = f(x) that defines the constraints
    - noise_level: Add Gaussian noise to y values
    
    Returns:
    - x: Array of shape (n_samples, 1)
    - y: Array of shape (n_samples, 1)
    """
    x = np.random.uniform(x_range[0], x_range[1], n_samples).reshape(-1, 1)
    if target_func is None:
        # Arbitrary constraints (fat solution space)
        y = np.random.randn(n_samples, 1) * 0.5
    else:
        y = target_func(x).reshape(-1, 1)
        if noise_level > 0:
            y += np.random.randn(n_samples, 1) * noise_level
    return x, y

def train_network(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    n_epochs: int = 5000,
    lr: float = 0.01,
    device: str = "cpu"
) -> float:
    """Train the network and return training MSE."""
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        train_pred = model(x_train)
        train_mse = criterion(train_pred, y_train).item()
    
    return train_mse

def evaluate_network(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    device: str = "cpu"
) -> float:
    """Evaluate network on test points and return max absolute error."""
    model.to(device)
    model.eval()
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    
    with torch.no_grad():
        outputs = model(x_test)
        errors = torch.abs(outputs - y_test).cpu().numpy()
        max_error = np.max(errors)
    
    return max_error

def find_min_width(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    width_range: List[int],
    max_error_threshold: float = 1e-3,
    n_trials: int = 3
) -> Optional[int]:
    """
    Find the minimum width that achieves zero violation on held-out points.
    
    Returns:
    - Minimum width if found, None if not achievable in the range.
    """
    device = "cpu"  # Use CPU for simplicity
    
    for width in width_range:
        best_max_error = float('inf')
        
        for trial in range(n_trials):
            # Initialize network with random seed for reproducibility
            torch.manual_seed(42 + trial * 1000 + width)
            model = FeedforwardNet(input_dim=1, hidden_dim=width, output_dim=1)
            
            # Train
            train_mse = train_network(model, x_train, y_train, n_epochs=3000, lr=0.01, device=device)
            
            # Evaluate on test set
            max_error = evaluate_network(model, x_test, y_test, device=device)
            best_max_error = min(best_max_error, max_error)
            
            if max_error <= max_error_threshold:
                print(f"  Width {width}: trial {trial}, max_error={max_error:.2e} (SUCCESS)")
                return width
        
        print(f"  Width {width}: best max_error={best_max_error:.2e} (failed)")
    
    return None  # Not achievable in the scanned range

def run_experiment(
    w_values: List[int],
    width_range: List[int],
    n_test_points: int = 50,
    n_trials_per_w: int = 3,
    target_func=None,
    noise_level: float = 0.0
) -> dict:
    """
    Run the full experiment for multiple w values.
    
    Returns:
    - Dictionary containing results for each w.
    """
    results = {}
    x_range = (0.0, 1.0)
    
    for w in w_values:
        print(f"\n=== Testing w = {w} ===")
        N_c_w = None
        
        for trial in range(n_trials_per_w):
            # Generate training constraints
            x_train, y_train = generate_constraints(
                n_samples=w, x_range=x_range, target_func=target_func, noise_level=noise_level
            )
            
            # Generate held-out test constraints (different random points)
            x_test, y_test = generate_constraints(
                n_samples=n_test_points, x_range=x_range, target_func=target_func, noise_level=noise_level
            )
            
            # Convert to tensors
            x_train_t = torch.FloatTensor(x_train)
            y_train_t = torch.FloatTensor(y_train)
            x_test_t = torch.FloatTensor(x_test)
            y_test_t = torch.FloatTensor(y_test)
            
            # Find minimum width
            N_c_w = find_min_width(
                x_train_t, y_train_t, x_test_t, y_test_t,
                width_range=width_range, max_error_threshold=1e-3, n_trials=1
            )
            
            if N_c_w is not None:
                break
        
        if N_c_w is None:
            print(f"w={w}: N_c(w) = INF (not achievable in width range)")
            results[w] = {"N_c": float('inf'), "status": "not_achievable"}
        else:
            print(f"w={w}: N_c(w) = {N_c_w}")
            results[w] = {"N_c": N_c_w, "status": "achievable"}
    
    return results

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

if __name__ == "__main__":
    os.makedirs("/workspace/research_outputs/shusheng_rsi_run75/results", exist_ok=True)
    
    # Experiment parameters
    w_values = [5, 10, 20, 50, 100, 200, 500]  # Constraint sample sizes
    width_range = list(range(5, 201, 5))  # Hidden layer widths from 5 to 200
    n_test_points = 100  # Number of held-out constraint points
    
    # Scenario 1: Rigid case - constraints from a linear function (2D solution space)
    def linear_func(x):
        return 2.0 * x + 1.0  # f(x) = 2x + 1
    
    print("="*60)
    print("SCENARIO 1: Rigid Case - Linear Constraints (2D solution space)")
    print("="*60)
    results_rigid = run_experiment(
        w_values=w_values,
        width_range=width_range,
        n_test_points=n_test_points,
        target_func=linear_func,
        noise_level=0.0
    )
    
    # Save results
    import json
    with open("/workspace/research_outputs/shusheng_rsi_run75/results/rigid_case.json", "w") as f:
        json.dump(results_rigid, f, indent=2)
    
    # Scenario 2: Fat case - constraints from a quadratic function (higher-dimensional solution space needed)
    def quadratic_func(x):
        return 3.0 * x**2 + 2.0 * x + 1.0  # f(x) = 3x^2 + 2x + 1
    
    print("\n" + "="*60)
    print("SCENARIO 2: Fat Case - Quadratic Constraints (3D solution space)")
    print("="*60)
    results_fat = run_experiment(
        w_values=w_values,
        width_range=width_range,
        n_test_points=n_test_points,
        target_func=quadratic_func,
        noise_level=0.0
    )
    
    with open("/workspace/research_outputs/shusheng_rsi_run75/results/fat_case.json", "w") as f:
        json.dump(results_fat, f, indent=2)
    
    # Scenario 3: Fat case - arbitrary random constraints (very high-dimensional solution space)
    print("\n" + "="*60)
    print("SCENARIO 3: Fat Case - Arbitrary Random Constraints (High-D solution space)")
    print("="*60)
    results_random = run_experiment(
        w_values=w_values,
        width_range=width_range,
        n_test_points=n_test_points,
        target_func=None,  # Random constraints
        noise_level=0.0
    )
    
    with open("/workspace/research_outputs/shusheng_rsi_run75/results/random_case.json", "w") as f:
        json.dump(results_random, f, indent=2)
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)