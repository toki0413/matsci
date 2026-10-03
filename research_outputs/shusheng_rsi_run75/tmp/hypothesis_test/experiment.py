#!/usr/bin/env python3
"""
Bootstrap Solution Space Rigidity Experiment
Testing whether small feedforward networks can probe solution space rigidity.

Constraint family: Linear ODE f''(x) + ω²f(x) = 0 (harmonic oscillator)
Solution space dimension: 2 (rigid by construction)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ODEDataset(Dataset):
    """Dataset for ODE constraint evaluation."""
    def __init__(self, x_points, omega=1.0):
        self.x = torch.tensor(x_points, dtype=torch.float32).unsqueeze(1)
        self.omega = omega
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        # ODE constraint: f''(x) + omega^2 * f(x) = 0
        # We'll approximate f'' using finite differences in training
        return x

class SimpleMLP(nn.Module):
    """Simple feedforward network."""
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1, n_layers=2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class SecondDerivativeLoss(nn.Module):
    """Loss for ODE constraints using finite differences."""
    def __init__(self, omega=1.0, delta_x=0.01):
        super().__init__()
        self.omega = omega
        self.delta_x = delta_x
        
    def forward(self, model, x_points):
        """Compute ODE loss: f''(x) + omega^2 * f(x)."""
        x = x_points.unsqueeze(0)  # Add batch dim
        f = model(x)
        
        # Finite difference approximation of f''
        x_plus = x + self.delta_x
        x_minus = x - self.delta_x
        f_plus = model(x_plus)
        f_minus = model(x_minus)
        f_second = (f_plus - 2*f + f_minus) / (self.delta_x ** 2)
        
        ode_residual = f_second + self.omega**2 * f
        return torch.mean(ode_residual**2)

def train_network(model, dataset, epochs=2000, lr=0.01):
    """Train a network on ODE constraints."""
    criterion = SecondDerivativeLoss(omega=1.0, delta_x=0.01)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    x_all = dataset.x.numpy().flatten()
    np.random.shuffle(x_all)
    
    for epoch in range(epochs):
        # Sample subset for training
        idx = np.random.choice(len(x_all), size=len(x_all)//2, replace=False)
        x_train = torch.tensor(x_all[idx], dtype=torch.float32).unsqueeze(1)
        
        loss = criterion(model, x_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"  Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return loss.item()

def evaluate_zero_violation(model, held_out_x, threshold=1e-3):
    """Check if model achieves zero violation on held-out points."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(held_out_x, dtype=torch.float32).unsqueeze(1)
        f_pred = model(x_tensor)
        
        # Compute ODE residual at held-out points
        delta_x = 0.01
        x_plus = x_tensor + delta_x
        x_minus = x_tensor - delta_x
        f_plus = model(x_plus)
        f_minus = model(x_minus)
        f_second = (f_plus - 2*f_pred + f_minus) / (delta_x ** 2)
        ode_residual = f_second + f_pred  # omega=1
        
        max_abs_error = torch.abs(ode_residual).max().item()
    return max_abs_error <= threshold, max_abs_error

def find_min_width(w_constraints, width_range, held_out_ratio=0.2, n_trials=3):
    """
    Find minimum network width that achieves zero violation.
    
    Args:
        w_constraints: Number of constraint samples
        width_range: List of widths to test
        held_out_ratio: Fraction of points to hold out for evaluation
        n_trials: Number of trials per width (take best)
    
    Returns:
        Minimum width achieving zero violation, or None (unreachable)
    """
    # Generate constraint points
    x_domain = np.linspace(-2*np.pi, 2*np.pi, 1000)
    np.random.shuffle(x_domain)
    x_constraints = x_domain[:w_constraints]
    
    # Hold out some points for evaluation
    eval_pool = x_domain[w_constraints:w_constraints + int(len(x_domain) * held_out_ratio)]
    
    min_width = None
    
    for width in width_range:
        best_error = float('inf')
        for trial in range(n_trials):
            model = SimpleMLP(hidden_dim=width)
            train_network(model, ODEDataset(x_constraints), epochs=2000, lr=0.01)
            
            # Evaluate on held-out points
            zero_viol, max_err = evaluate_zero_violation(model, eval_pool)
            best_error = min(best_error, max_err)
            
            if zero_viol:
                print(f"  Width {width}: Zero violation achieved (trial {trial})")
                min_width = width
                break
        
        if min_width is not None:
            break
    
    return min_width

def main():
    """Run the main experiment."""
    print("=" * 60)
    print("Bootstrap Solution Space Rigidity Experiment")
    print("Constraint: ODE f''(x) + f(x) = 0 (harmonic oscillator)")
    print("Known solution space dimension: 2 (rigid)")
    print("=" * 60)
    
    # Scan parameters
    w_values = [5, 10, 20, 50, 100, 200]  # Number of constraint samples
    width_range = [4, 8, 16, 32, 64, 128, 256]  # Hidden layer widths
    
    results = []
    
    for w in w_values:
        print(f"\nw = {w}: Finding N_c(w)...")
        N_c = find_min_width(w, width_range, held_out_ratio=0.3, n_trials=2)
        
        if N_c is None:
            print(f"  N_c(w) = UNREACHABLE (fat)")
            results.append({'w': w, 'N_c': None})
        else:
            print(f"  N_c({w}) = {N_c} (rigid)")
            results.append({'w': w, 'N_c': N_c})
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary of N_c(w) values:")
    print("=" * 60)
    for r in results:
        if r['N_c'] is None:
            print(f"  w={r['w']}: N_c = UNREACHABLE")
        else:
            print(f"  w={r['w']}: N_c = {r['N_c']}")
    
    # Determine rigidity classification
    rigid_N_c = [r['N_c'] for r in results if r['N_c'] is not None]
    if len(rigid_N_c) > 0 and max(rigid_N_c) <= 32:  # Small finite N_c
        print("\nClassification: RIGID (solution space is low-dimensional)")
    else:
        print("\nClassification: FAT (solution space is high-dimensional or N_c unreachable)")

if __name__ == "__main__":
    main()