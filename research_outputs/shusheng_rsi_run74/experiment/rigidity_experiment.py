#!/usr/bin/env python3
"""
Bootstrap Rigidity Probe for Small Feedforward Networks

Tests whether the minimum network width N_c(w) needed to satisfy w analytic
constraints on held-out points scales as:
- Constant (rigid), Linear (floppy), or Sublinear power-law (geometric intermediate)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# 1. Define Analytic Constraint Functions
# ============================================================

class AnalyticFunction:
    """Base class for analytic functions with point evaluation constraints."""
    
    def __init__(self, domain: Tuple[float, float] = (-1.0, 1.0)):
        self.domain = domain
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def sample_constraints(self, n: int, train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Sample n constraint points uniformly from domain."""
        x = np.random.uniform(self.domain[0], self.domain[1], n)
        y = self.evaluate(x)
        return x, y


class PolynomialFunc(AnalyticFunc):
    """Polynomial function of degree d."""
    
    def __init__(self, degree: int = 3, coeffs: Optional[np.ndarray] = None, domain: Tuple[float, float] = (-1.0, 1.0)):
        super().__init__(domain)
        self.degree = degree
        if coeffs is None:
            coeffs = np.random.uniform(-1, 1, degree + 1)
        self.coeffs = coeffs
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.polyval(self.coeffs, x)


class TrigFunc(AnalyticFunc):
    """Trigonometric function combination."""
    
    def __init__(self, domain: Tuple[float, float] = (-np.pi, np.pi)):
        super().__init__(domain)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.sin(3 * x) + 0.5 * np.cos(5 * x) + 0.3 * np.sin(x)


class MixedFunc(AnalyticFunc):
    """Mixed analytic function (polynomial + trigonometric)."""
    
    def __init__(self, domain: Tuple[float, float] = (-1.0, 1.0)):
        super().__init__(domain)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return x**3 - 2*x + 0.5 * np.sin(4 * np.pi * x)


# ============================================================
# 2. Single-Hidden-Layer MLP
# ============================================================

class SingleLayerMLP(nn.Module):
    """Single-hidden-layer feedforward network with ReLU activation."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


# ============================================================
# 3. Training and Evaluation
# ============================================================

def train_mlp(
    model: SingleLayerMLP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    max_epochs: int = 5000,
    lr: float = 1e-3,
    patience: int = 100
) -> Tuple[float, float, int]:
    """Train MLP and return validation error and epochs used."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_error = float('inf')
    patience_counter = 0
    epochs_used = 0
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation (held-out constraints)
        model.eval()
        with torch.no_grad():
            y_val_pred = model(x_val)
            val_error = torch.max(torch.abs(y_val_pred - y_val)).item()
        
        epochs_used = epoch + 1
        
        # Early stopping
        if val_error < best_val_error:
            best_val_error = val_error
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_val_error, epochs_used


def find_nc_w(
    func: AnalyticFunc,
    w: int,
    width_range: List[int],
    n_trials: int = 3,
    max_epochs: int = 5000,
    tolerance: float = 1e-3
) -> Optional[int]:
    """
    Find minimum width N_c(w) that achieves zero violation on held-out constraints.
    
    Returns the minimum width where max abs error <= tolerance on validation set,
    or None if no width in the range succeeds.
    """
    n_total = w * 3  # Total points (training + validation)
    
    for trial in range(n_trials):
        # Sample constraint points
        x_all, y_all = func.sample_constraints(n_total)
        
        # Split into training (2w/3) and validation (w/3)
        n_train = int(2 * w / 3)
        n_val = w - n_train
        
        x_train = torch.tensor(x_all[:n_train], dtype=torch.float32).view(-1, 1)
        y_train = torch.tensor(y_all[:n_train], dtype=torch.float32).view(-1, 1)
        x_val = torch.tensor(x_all[n_train:n_train + n_val], dtype=torch.float32).view(-1, 1)
        y_val = torch.tensor(y_all[n_train:n_train + n_val], dtype=torch.float32).view(-1, 1)
        
        # Test each width in range
        for width in width_range:
            model = SingleLayerMLP(input_dim=1, hidden_dim=width, output_dim=1)
            val_error, epochs = train_mlp(model, x_train, y_train, x_val, y_val, 
                                          max_epochs=max_epochs, lr=1e-3)
            
            if val_error <= tolerance:
                # Success on this trial - check if consistent
                return width
    
    # Failed on all trials for this width
    return None


# ============================================================
# 4. Main Experiment
# ============================================================

def run_experiment():
    """Run the full rigidity scan experiment."""
    
    # Configuration
    widths_to_test = [4, 8, 16, 32, 64, 128, 256, 512, 1024]  # Search range for N_c
    w_values = [5, 10, 20, 40, 80, 160, 320]  # Constraint sample sizes
    n_trials_per_w = 2  # Number of random constraint sets per w
    max_epochs = 5000
    tolerance = 1e-3
    
    # Use mixed analytic function (polynomial + trigonometric)
    func = MixedFunc(domain=(-1.0, 1.0))
    
    results = {}
    
    print(f"Starting rigidity scan with w values: {w_values}")
    print(f"Width range: {widths_to_test}")
    print(f"Tolerance: {tolerance}")
    print(f"Function: MixedFunc (polynomial + trigonometric)")
    print()
    
    for w in w_values:
        print(f"Testing w = {w}:")
        nc_values = []
        
        for trial in range(n_trials_per_w):
            nc = find_nc_w(func, w, widths_to_test, n_trials=1, 
                         max_epochs=max_epochs, tolerance=tolerance)
            if nc is not None:
                nc_values.append(nc)
                print(f"  Trial {trial+1}: N_c = {nc}")
            else:
                print(f"  Trial {trial+1}: No width achieved zero violation (N_c > {widths_to_test[-1]})")
        
        if nc_values:
            nc_w = min(nc_values)  # Take minimum across trials
            results[w] = nc_w
            print(f"  => N_c({w}) = {nc_w} (minimum across trials)")
        else:
            results[w] = None
            print(f"  => N_c({w}) = None (unreachable in scan range)")
        
        print()
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    # Save as JSON
    with open('results/rigidity_results.json', 'w') as f:
        json.dump({
            'widths_to_test': widths_to_test,
            'w_values': w_values,
            'results': results,
            'tolerance': tolerance,
            'function_type': 'MixedFunc'
        }, f, indent=2)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    w_kept = [w for w in results if results[w] is not None]
    nc_kept = [results[w] for w in w_kept]
    
    if w_kept:
        plt.scatter(w_kept, nc_kept, color='blue', s=100, label='N_c(w)', zorder=5)
        plt.plot(w_kept, nc_kept, 'b-', linewidth=2, alpha=0.7)
        
        # Try power-law fit: N_c = a * w^alpha
        log_w = np.log(w_kept)
        log_nc = np.log(nc_kept)
        coeffs = np.polyfit(log_w, log_nc, 1)
        alpha = coeffs[0]
        a = np.exp(coeffs[1])
        w_smooth = np.array(w_kept)
        nc_fit = a * w_smooth ** alpha
        
        plt.plot(w_smooth, nc_fit, 'r--', linewidth=2, 
                label=f'Power-law fit: N_c = {a:.2f} * w^{alpha:.3f}')
        print(f"\nPower-law fit: N_c = {a:.4f} * w^{alpha:.4f}")
        
        # Check if alpha is close to 0 (rigid), 1 (floppy), or intermediate (geometric)
        if alpha < 0.3:
            regime = "RIGID (constant-like)"
        elif alpha > 0.7:
            regime = "FLOPPY (linear-like)"
        else:
            regime = "GEOMETRIC INTERMEDIATE (sublinear)"
        print(f"Regime: {regime} (alpha = {alpha:.3f})")
        
        plt.xlabel('Number of constraints w', fontsize=12)
        plt.ylabel('Minimum width N_c(w)', fontsize=12)
        plt.title('Bootstrap Rigidity Probe: N_c(w) vs w', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('results/rigidity_scan.png', dpi=150)
        plt.close()
    
    print("\nResults saved to results/")
    return results


if __name__ == '__main__':
    results = run_experiment()