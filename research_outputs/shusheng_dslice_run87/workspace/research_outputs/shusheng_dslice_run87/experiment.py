"""
Experiment: Measuring N_c(w) for solution space rigidity probing
by small feedforward networks.

Problem setup:
- w = number of constraint samples (points (x_i, y_i) that the network must satisfy)
- N_c(w) = minimum hidden layer width achieving max abs error <= 1e-3 on held-out points
- Zero violation: max absolute error on held-out constraints <= 1e-3

We test three competing predictions for N_c(w):
1. Algebraic (Structure): N_c = O(1) — constant in w
2. Geometric (Manifold): N_c ∝ w — linear in w  
3. Measure (Statistical): N_c ∝ sqrt(w) — square-root in w
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    # Constraint sampling
    'w_values': [10, 25, 50, 100, 200, 500],  # number of constraint samples
    'n_held_out': 100,  # number of held-out constraint points
    'x_min': 0.0,
    'x_max': 1.0,
    
    # Network architecture
    'width_range': (4, 64),  # (min_width, max_width) for sweep
    'n_widths': 15,  # number of widths to test
    
    # Training
    'n_epochs': 500,  # reduced for faster testing
    'batch_size': 32,
    'learning_rate': 1e-3,
    'zero_violation_threshold': 1e-3,
    
    # Function family to constrain (ground truth)
    'function_type': 'polynomial',  # 'polynomial', 'sine', 'piecewise'
    'poly_degree': 3,
}

os.makedirs('/workspace/research_outputs/shusheng_dslice_run87/results', exist_ok=True)

# ============================================================
# Function families to constrain
# ============================================================

def true_function(x, func_type, **kwargs):
    """The ground-truth function that the network should learn."""
    if func_type == 'polynomial':
        degree = kwargs.get('poly_degree', 3)
        # Generate a random polynomial of given degree
        coeffs = np.random.randn(degree + 1)
        return np.polyval(coeffs, x)
    elif func_type == 'sine':
        return np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)
    elif func_type == 'piecewise':
        # Piecewise function with discontinuities in derivative
        return np.where(x < 0.5, x, 1 - x)
    else:
        return np.sin(2 * np.pi * x)

# ============================================================
# Neural Network Model
# ============================================================

class SimpleReLU(nn.Module):
    """Simple feedforward network with ReLU activations."""
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1, n_layers=2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i in range(n_layers):
            if i == n_layers - 1:
                layers.append(nn.Linear(prev_dim, output_dim))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
            prev_dim = hidden_dim if i < n_layers - 1 else output_dim
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ============================================================
# Data Generation
# ============================================================

def generate_constraint_data(w, func_type, **kwargs):
    """Generate w constraint points (x, y) from the true function."""
    x = np.random.uniform(CONFIG['x_min'], CONFIG['x_max'], w)
    y = true_function(x, func_type, **kwargs)
    return torch.tensor(x, dtype=torch.float32).view(-1, 1), torch.tensor(y, dtype=torch.float32).view(-1, 1)

def generate_held_out_data(n, func_type, **kwargs):
    """Generate held-out constraint points."""
    x = np.random.uniform(CONFIG['x_min'], CONFIG['x_max'], n)
    y = true_function(x, func_type, **kwargs)
    return torch.tensor(x, dtype=torch.float32).view(-1, 1), torch.tensor(y, dtype=torch.float32).view(-1, 1)

# ============================================================
# Training and Evaluation
# ============================================================

def train_network(model, x_train, y_train, x_val, y_val, n_epochs, lr):
    """Train the network and return validation loss."""
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    # Evaluate on held-out set
    model.eval()
    with torch.no_grad():
        val_outputs = model(x_val)
        val_loss = criterion(val_outputs, y_val).item()
        max_abs_error = torch.max(torch.abs(val_outputs - y_val)).item()
    
    return val_loss, max_abs_error

def find_min_width(w, func_type, **kwargs):
    """Find the minimum width achieving zero violation on held-out points."""
    x_train, y_train = generate_constraint_data(w, func_type, **kwargs)
    x_val, y_val = generate_held_out_data(CONFIG['n_held_out'], func_type, **kwargs)
    
    width_range = CONFIG['width_range']
    n_widths = CONFIG['n_widths']
    widths = np.linspace(width_range[0], width_range[1], n_widths, dtype=int)
    
    min_width = None
    min_width_val = float('inf')
    
    for H in widths:
        model = SimpleReLU(input_dim=1, hidden_dim=H, output_dim=1, n_layers=2)
        val_loss, max_abs_error = train_network(
            model, x_train, y_train, x_val, y_val,
            CONFIG['n_epochs'], CONFIG['learning_rate']
        )
        
        if max_abs_error <= CONFIG['zero_violation_threshold']:
            if H < min_width_val or min_width is None:
                min_width_val = H
                min_width = H
        
        # Print progress
        status = "✓" if max_abs_error <= CONFIG['zero_violation_threshold'] else "✗"
        print(f"  w={w:4d}, H={H:3d}, max_err={max_abs_error:.2e}, {status}")
    
    return min_width

# ============================================================
# Main Experiment
# ============================================================

def main():
    print(f"Starting experiment with function type: {CONFIG['function_type']}")
    print(f"Testing w values: {CONFIG['w_values']}")
    print(f"Width range: {CONFIG['width_range']}, {CONFIG['n_widths']} widths")
    print(f"Zero violation threshold: {CONFIG['zero_violation_threshold']}")
    print()
    
    results = {}
    
    for w in CONFIG['w_values']:
        print(f"\n=== Testing w = {w} ===")
        min_width = find_min_width(w, CONFIG['function_type'], poly_degree=CONFIG['poly_degree'])
        results[w] = min_width
        print(f"  Minimum width for zero violation: {min_width}")
    
    # Save results
    results_path = '/workspace/research_outputs/shusheng_dslice_run87/results/results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Fit scaling laws
    fit_scaling_laws(results)

def fit_scaling_laws(results):
    """Fit different scaling laws to the measured N_c(w) values."""
    w_vals = np.array(list(results.keys()), dtype=float)
    Nc_vals = np.array([results[w] for w in w_vals], dtype=float)
    
    # Remove any inf/None values
    valid = ~np.isinf(Nc_vals) & ~np.isnan(Nc_vals)
    w_vals = w_vals[valid]
    Nc_vals = Nc_vals[valid]
    
    if len(w_vals) < 3:
        print("Not enough valid data points for fitting.")
        return
    
    print("\n--- Scaling Law Fitting ---")
    print(f"Data: w={w_vals}, N_c={Nc_vals}")
    
    # Fit: N_c = a * log(w) + b (logarithmic)
    log_w = np.log(w_vals)
    log_fit = np.polyfit(log_w, Nc_vals, 1)
    log_a, log_b = log_fit
    log_pred = np.polyval(log_fit, log_w)
    log_r2 = 1 - np.sum((Nc_vals - log_pred)**2) / np.sum((Nc_vals - np.mean(Nc_vals))**2)
    print(f"Logarithmic: N_c = {log_a:.3f} * log(w) + {log_b:.3f}, R² = {log_r2:.4f}")
    
    # Fit: N_c = a * sqrt(w) + b (square-root)
    sqrt_w = np.sqrt(w_vals)
    sqrt_fit = np.polyfit(sqrt_w, Nc_vals, 1)
    sqrt_a, sqrt_b = sqrt_fit
    sqrt_pred = np.polyval(sqrt_fit, sqrt_w)
    sqrt_r2 = 1 - np.sum((Nc_vals - sqrt_pred)**2) / np.sum((Nc_vals - np.mean(Nc_vals))**2)
    print(f"Square-root: N_c = {sqrt_a:.3f} * sqrt(w) + {sqrt_b:.3f}, R² = {sqrt_r2:.4f}")
    
    # Fit: N_c = a * w + b (linear)
    lin_fit = np.polyfit(w_vals, Nc_vals, 1)
    lin_a, lin_b = lin_fit
    lin_pred = np.polyval(lin_fit, w_vals)
    lin_r2 = 1 - np.sum((Nc_vals - lin_pred)**2) / np.sum((Nc_vals - np.mean(Nc_vals))**2)
    print(f"Linear:      N_c = {lin_a:.4f} * w + {lin_b:.3f}, R² = {lin_r2:.4f}")
    
    # Fit: N_c = constant (horizontal line)
    const_fit = np.mean(Nc_vals)
    const_r2 = 1 - np.sum((Nc_vals - const_fit)**2) / np.sum((Nc_vals - np.mean(Nc_vals))**2)
    print(f"Constant:    N_c = {const_fit:.3f}, R² = {const_r2:.4f}")
    
    # Determine best fit
    r2s = {
        'logarithmic': log_r2,
        'square-root': sqrt_r2,
        'linear': lin_r2,
        'constant': const_r2
    }
    best_fit = max(r2s, key=r2s.get)
    print(f"\nBest fit: {best_fit} (R² = {r2s[best_fit]:.4f})")
    
    # Plot results
    plot_scaling_laws(w_vals, Nc_vals, log_w, log_pred, sqrt_w, sqrt_pred, w_vals, lin_pred, const_fit)

def plot_scaling_laws(w_vals, Nc_vals, log_w, log_pred, sqrt_w, sqrt_pred, w_vals_lin, lin_pred, const_fit):
    """Plot the scaling law fits."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: N_c vs w (log scale for x-axis)
    ax1 = axes[0]
    ax1.scatter(w_vals, Nc_vals, color='blue', s=100, zorder=5, label='Measured N_c(w)')
    ax1.plot(w_vals, log_pred, 'r--', label=f'Log fit (R²={log_r2:.2f})')
    ax1.plot(w_vals, sqrt_pred, 'g--', label=f'Sqrt fit (R²={sqrt_r2:.2f})')
    ax1.plot(w_vals, lin_pred, 'm--', label=f'Linear fit (R²={lin_r2:.2f})')
    ax1.axhline(y=const_fit, color='orange', linestyle=':', label=f'Constant (R²={const_r2:.2f})')
    ax1.set_xscale('log')
    ax1.set_xlabel('w (constraint samples)', fontsize=12)
    ax1.set_ylabel('N_c(w) (min width)', fontsize=12)
    ax1.set_title('N_c(w) vs w (log scale)', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: N_c vs transformed variables
    ax2 = axes[1]
    ax2.scatter(np.log(w_vals), Nc_vals, color='blue', s=100, zorder=5, label='Data')
    ax2.plot(log_w, log_pred, 'r-', linewidth=2, label=f'Log fit (R²={log_r2:.2f})')
    ax2.scatter(np.sqrt(w_vals), Nc_vals, color='green', s=100, zorder=5, label='Data')
    ax2.plot(sqrt_w, sqrt_pred, 'g-', linewidth=2, label=f'Sqrt fit (R²={sqrt_r2:.2f})')
    ax2.scatter(w_vals, Nc_vals, color='red', s=100, zorder=5, label='Data')
    ax2.plot(w_vals_lin, lin_pred, 'm-', linewidth=2, label=f'Linear fit (R²={lin_r2:.2f})')
    ax2.axhline(y=const_fit, color='orange', linestyle='--', linewidth=2, label=f'Constant (R²={const_r2:.2f})')
    ax2.set_xlabel('Transformed variable', fontsize=12)
    ax2.set_ylabel('N_c(w) (min width)', fontsize=12)
    ax2.set_title('N_c(w) vs transformed variables', fontsize=14)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/research_outputs/shusheng_dslice_run87/results/scaling_fit.png', dpi=150)
    plt.close()
    print("Plot saved to /workspace/research_outputs/shusheng_dslice_run87/results/scaling_fit.png")

if __name__ == '__main__':
    main()