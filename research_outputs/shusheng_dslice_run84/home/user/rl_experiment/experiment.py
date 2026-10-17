#!/usr/bin/env python3
"""
Experiment: Testing whether small feedforward networks can probe solution space rigidity.

This script implements the core experiment:
- Define a family of analytic constraint functions (polynomials, trig functions, etc.)
- For each constraint function, sample w constraint points (training set)
- Hold out a separate set of validation points (held-out constraint points)
- Train small feedforward networks of varying hidden widths
- Find the minimum width N_c that achieves zero violation (max abs error <= 1e-3)
  on the held-out points
- Report N_c(w) values and determine if the solution space is rigid or fat

The experiment uses two families of functions:
1. Low-complexity (rigid): Simple polynomials, low-degree functions
2. High-complexity (fat): High-frequency oscillatory functions, high-degree polynomials
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
from typing import Tuple, List, Optional

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Zero violation threshold
ZERO_VIOLATION_THRESHOLD = 1e-3

# Search range for network width
MIN_WIDTH = 1
MAX_WIDTH = 256  # Upper bound for the scan

# Training configuration
NUM_EPOCHS = 1000
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
PATIENCE = 50  # Early stopping patience

class SimpleFeedforward(nn.Module):
    """Simple feedforward network with ReLU activations."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        ])
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        # Add more layers if hidden_dim > 1
        if hidden_dim > 1:
            self.layers.insert(1, nn.Linear(hidden_dim, hidden_dim))
            self.layers.insert(2, nn.ReLU())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def generate_constraint_function(func_type: str, n_points: int, x_range: Tuple[float, float] = (0.0, 1.0)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a constraint function and sample points.
    
    Args:
        func_type: Type of function ('poly', 'trig', 'high_freq', 'piecewise')
        n_points: Number of points to sample
        x_range: Range of x values
    
    Returns:
        Tuple of (x_values, y_values)
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    
    if func_type == 'poly':
        # Low-degree polynomial (rigid)
        y = 0.5 * x**2 + 0.3 * x + 0.1
    elif func_type == 'trig':
        # Simple trigonometric function (rigid)
        y = np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x)
    elif func_type == 'high_freq':
        # High-frequency oscillatory function (fat)
        y = np.sin(20 * np.pi * x) * np.exp(-5 * x)
    elif func_type == 'piecewise':
        # Piecewise function with discontinuities (fat)
        y = np.where(x < 0.5, x, 1 - x) + 0.1 * np.sin(50 * np.pi * x)
    else:
        raise ValueError(f"Unknown func_type: {func_type}")
    
    return x, y

def create_dataset(x: np.ndarray, y: np.ndarray, train_frac: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create training and held-out validation datasets.
    
    Args:
        x: Input features
        y: Target values
        train_frac: Fraction of points to use for training
    
    Returns:
        Tuple of (x_train, y_train, x_val, y_val)
    """
    n = len(x)
    indices = np.random.permutation(n)
    train_end = int(n * train_frac)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:]
    
    x_train = torch.tensor(x[train_idx], dtype=torch.float32).reshape(-1, 1)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32).reshape(-1, 1)
    x_val = torch.tensor(x[val_idx], dtype=torch.float32).reshape(-1, 1)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32).reshape(-1, 1)
    
    return x_train, y_train, x_val, y_val

def train_network(model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, 
                 x_val: torch.Tensor, y_val: torch.Tensor, epochs: int = 1000,
                 lr: float = 1e-3, patience: int = 50) -> float:
    """
    Train a network and return the validation error.
    
    Args:
        model: Neural network model
        x_train, y_train: Training data
        x_val, y_val: Validation data
        epochs: Number of training epochs
        lr: Learning rate
        patience: Early stopping patience
    
    Returns:
        Maximum absolute error on validation set
    """
    model.to(DEVICE)
    x_train = x_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    x_val = x_val.to(DEVICE)
    y_val = y_val.to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_error = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_error = torch.max(torch.abs(val_outputs - y_val)).item()
        
        if val_error < best_val_error:
            best_val_error = val_error
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_val_error

def find_minimum_width(func_type: str, w: int, n_val: int = 50) -> Tuple[Optional[int], float]:
    """
    Find the minimum network width that achieves zero violation on held-out points.
    
    Args:
        func_type: Type of constraint function
        w: Number of constraint samples (training points)
        n_val: Number of validation points
    
    Returns:
        Tuple of (minimum_width, validation_error)
        If no width achieves zero violation, returns (None, final_error)
    """
    # Generate the constraint function
    x_full, y_full = generate_constraint_function(func_type, w + n_val)
    
    # Create training and validation sets
    x_train, y_train, x_val, y_val = create_dataset(x_full, y_full, train_frac=w/(w+n_val))
    
    # Search for minimum width
    for width in range(MIN_WIDTH, MAX_WIDTH + 1, max(1, MAX_WIDTH // 20)):
        # Create and train network
        model = SimpleFeedforward(input_dim=1, hidden_dim=width)
        val_error = train_network(model, x_train, y_train, x_val, y_val, 
                                  epochs=NUM_EPOCHS, lr=LEARNING_RATE, 
                                  patience=PATIENCE)
        
        # Check for zero violation
        if val_error <= ZERO_VIOLATION_THRESHOLD:
            return width, val_error
    
    # No width achieved zero violation - return the best error found
    # Try the maximum width one more time to get a final error
    model = SimpleFeedforward(input_dim=1, hidden_dim=MAX_WIDTH)
    final_error = train_network(model, x_train, y_train, x_val, y_val,
                               epochs=NUM_EPOCHS, lr=LEARNING_RATE,
                               patience=PATIENCE)
    return None, final_error

def run_experiment(func_types: List[str], w_values: List[int]) -> dict:
    """
    Run the full experiment across multiple function types and w values.
    
    Args:
        func_types: List of function types to test
        w_values: List of constraint sample counts to scan
    
    Returns:
        Dictionary with results
    """
    results = {
        'func_types': func_types,
        'w_values': w_values,
        'results': {}
    }
    
    for func_type in func_types:
        results['results'][func_type] = []
        print(f"\n=== Testing function type: {func_type} ===")
        
        for w in w_values:
            min_width, val_error = find_minimum_width(func_type, w)
            
            result_entry = {
                'w': w,
                'min_width': min_width,
                'validation_error': val_error,
                'zero_violation': min_width is not None
            }
            results['results'][func_type].append(result_entry)
            
            status = "ZERO VIOLATION" if min_width is not else "NO ZERO VIOLATION"
            print(f"  w={w}: N_c={min_width if min_width else 'None (max={MAX_WIDTH})'}, error={val_error:.6e} [{status}]")
        
        # Determine if rigid or fat
        rigid_widths = [r['min_width'] for r in results['results'][func_type] if r['min_width'] is not None]
        if rigid_widths and all(w <= 20 for w in rigid_widths):  # Threshold for "small" N_c
            classification = "RIGID"
        elif rigid_widths:
            classification = "MEDIUM"
        else:
            classification = "FAT"
        print(f"  Classification: {classification}")
    
    return results

def plot_results(results: dict, output_path: str = '/home/user/rl_experiment/results.png'):
    """Plot the N_c(w) results."""
    func_types = results['func_types']
    w_values = results['w_values']
    
    fig, axes = plt.subplots(1, len(func_types), figsize=(5*len(func_types), 5), 
                            sharey=True)
    if len(func_types) == 1:
        axes = [axes]
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, func_type in enumerate(func_types):
        ax = axes[idx]
        r_results = results['results'][func_type]
        
        widths = []
        errors = []
        w_list = []
        
        for r in r_results:
            if r['min_width'] is not None:
                widths.append(r['min_width'])
                errors.append(r['validation_error'])
                w_list.append(r['w'])
            else:
                widths.append(MAX_WIDTH + 10)  # Mark as beyond range
                errors.append(r['validation_error'])
                w_list.append(r['w'])
        
        ax.plot(w_list, widths, 'o-', color=colors[idx % len(colors)], 
                label=f'{func_type} (N_c)', markersize=8)
        ax.set_xlabel('w (constraint samples)')
        ax.set_ylabel('N_c (hidden width)')
        ax.set_title(f'{func_type}')
        ax.axhline(y=ZERO_VIOLATION_THRESHOLD * 10, color='gray', linestyle='--', 
                   alpha=0.5, label='Zero violation threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Solution Space Rigidity: N_c(w) vs w')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults plot saved to {output_path}")
    
    # Also plot error vs w
    fig, axes = plt.subplots(1, len(func_types), figsize=(5*len(func_types), 5), 
                            sharey=True)
    if len(func_types) == 1:
        axes = [axes]
    
    for idx, func_type in enumerate(func_types):
        ax = axes[idx]
        r_results = results['results'][func_type]
        
        errors = []
        w_list = []
        
        for r in r_results:
            errors.append(r['validation_error'])
            w_list.append(r['w'])
        
        ax.plot(w_list, errors, 's-', color=colors[idx % len(colors)], 
                label=f'{func_type} error', markersize=8)
        ax.set_xlabel('w (constraint samples)')
        ax.set_ylabel('Validation error (max abs)')
        ax.set_title(f'{func_type} - Error')
        ax.axhline(y=ZERO_VIOLATION_THRESHOLD, color='red', linestyle='--', 
                   alpha=0.5, label=f'Zero violation threshold ({ZERO_VIOLATION_THRESHOLD})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.suptitle('Solution Space Rigidity: Validation Error vs w')
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_error.png'), dpi=150, bbox_inches='tight')
    print(f"Error plot saved to {output_path.replace('.png', '_error.png')}")

def main():
    """Main entry point."""
    print("=" * 60)
    print("Solution Space Rigidity Experiment")
    print("=" * 60)
    
    # Define function types to test (rigid vs fat families)
    func_types = ['poly', 'trig', 'high_freq', 'piecewise']
    
    # Define w values to scan
    w_values = [5, 10, 20, 50, 100, 200, 500]
    
    # Run experiment
    results = run_experiment(func_types, w_values)
    
    # Save results to JSON
    output_dir = '/home/user/rl_experiment'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_dir}/results.json")
    
    # Plot results
    plot_results(results)
    
    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE: N_c(w) Values")
    print("=" * 60)
    
    table_header = f"{'Function Type':<15}"
    for w in w_values:
        table_header += f"{w:>8}"
    print(table_header)
    
    for func_type in func_types:
        row = f"{func_type:<15}"
        r_results = results['results'][func_type]
        for r in r_results:
            if r['min_width'] is not None:
                row += f"{r['min_width']:>8}"
            else:
                row += f"{'None':>8}"
        print(row)
    
    print("\n" + "=" * 60)
    print("CONCLUSION: Determining Rigidity vs Fatness")
    print("=" * 60)
    
    for func_type in func_types:
        r_results = results['results'][func_type]
        rigid_widths = [r['min_width'] for r in r_results if r['min_width'] is not None]
        
        if rigid_widths:
            max_nc = max(rigid_widths)
            if max_nc <= 20:  # Threshold for "small" N_c
                print(f"{func_type}: RIGID (N_c stays small, max={max_nc})")
            else:
                print(f"{func_type}: MEDIUM (N_c grows but finite, max={max_nc})")
        else:
            print(f"{func_type}: FAT (N_c not reachable within search range)")
    
    print("\nExperiment complete!")

if __name__ == '__main__':
    main()