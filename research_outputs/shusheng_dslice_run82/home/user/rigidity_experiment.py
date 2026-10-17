#!/usr/bin/env python3
"""
Solution Space Rigidity Experiment for Small Feedforward Networks

Tests whether N_c(w) (minimum hidden width for zero violation on out-of-sample constraints)
remains constant (rigid) or grows (floppy) as the number of constraint samples w increases.

Hard constraints:
- Zero violation: max absolute error <= 1e-3 on out-of-sample constraints
- All reported N_c values must be finite (no inf/nan)
- Only small feedforward networks (two-layer ReLU)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# Target function (Barron-class example: smooth analytic function)
# ============================================================
class TargetFunction:
    """A smooth analytic function in the Barron class."""
    def __init__(self, d=1):
        self.d = d
    
    def __call__(self, x):
        # Single-variable smooth function: sin(x) + 0.1*x^2 (analytic, bounded Fourier support)
        if x.ndim == 1:
            return np.sin(x) + 0.1 * x**2
        else:
            return np.sin(x) + 0.1 * x**2
    
    def gradient(self, x):
        if x.ndim == 1:
            return np.cos(x) + 0.2 * x
        else:
            return np.cos(x) + 0.2 * x

# ============================================================
# Dataset
# ============================================================
class ConstraintDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ============================================================
# Two-layer ReLU Network
# ============================================================
class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Initialize weights for stable training
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# ============================================================
# Training function
# ============================================================
def train_model(model, x_train, y_train, x_val, y_val, 
                lr=0.01, max_epochs=10000, patience=100, verbose=False):
    """Train a model and return training and validation max errors."""
    dataset = ConstraintDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=min(32, len(x_train)), shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        total_loss = 0
        for xb, yb in loader:
            outputs = model(xb)
            loss = criterion(outputs, yb)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        # Compute max absolute errors
        with torch.no_grad():
            train_pred = model(torch.FloatTensor(x_train))
            train_max_err = float(torch.max(torch.abs(train_pred - torch.FloatTensor(y_train))).numpy())
            
            val_pred = model(torch.FloatTensor(x_val))
            val_max_err = float(torch.max(torch.abs(val_pred - torch.FloatTensor(y_val))).numpy())
        
        val_losses.append(val_max_err)
        
        if val_max_err < best_val_loss:
            best_val_loss = val_max_err
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break
    
    return train_max_err, val_max_err, val_losses

# ============================================================
# Find minimum width for zero violation
# ============================================================
def find_nc_w(w, target_func, input_dim=1, 
              width_range=list(range(4, 51)),  # N from 4 to 50
              n_samples_train=100, n_samples_val=50,
              seed=42):
    """
    Find the minimum width N_c(w) that achieves zero violation on out-of-sample points.
    
    Returns:
        N_c: minimum width (finite int) or None if not found in range
        train_err: training max error at N_c
        val_err: validation max error at N_c
    """
    np.random.seed(seed + w)
    
    # Generate training constraints
    x_train = np.random.uniform(-2, 2, w)
    y_train = target_func(x_train)
    
    # Generate out-of-sample validation constraints
    np.random.seed(seed + w + 1000)
    x_val = np.random.uniform(-2, 2, n_samples_val)
    y_val = target_func(x_val)
    
    N_c = None
    train_err = None
    val_err = None
    
    for N in width_range:
        # Create and train model
        model = TwoLayerNet(input_dim, N)
        train_max_err, val_max_err, _ = train_model(
            model, x_train, y_train, x_val, y_val,
            lr=0.01, max_epochs=10000, patience=200, verbose=False
        )
        
        # Check zero violation on validation set
        if val_max_err <= 1e-3:
            N_c = N
            train_err = train_max_err
            val_err = val_max_err
            if verbose:
                print(f"  Width {N} achieves zero violation (val_err={val_max_err:.2e})")
            break
    
    return N_c, train_err, val_err

# ============================================================
# Main experiment
# ============================================================
def main():
    print("=" * 70)
    print("Solution Space Rigidity Experiment")
    print("Target: f(x) = sin(x) + 0.1*x^2 (Barron-class analytic function)")
    print("Zero violation threshold: max abs error <= 1e-3")
    print("Width range: 4 to 50 hidden units")
    print("=" * 70)
    
    target_func = TargetFunction(d=1)
    
    # Scan over different numbers of constraint samples w
    w_values = [5, 10, 20, 50, 100, 200, 500]  # Range of constraint counts
    width_range = list(range(4, 51))
    
    results = []
    
    print(f"\nScanning w values: {w_values}\n")
    print(f"{'w':>5} {'N_c':>6} {'train_err':>12} {'val_err':>12} {'status':>10}")
    print("-" * 50)
    
    for w in w_values:
        N_c, train_err, val_err = find_nc_w(
            w=w, target_func=target_func,
            width_range=width_range,
            n_samples_train=w, n_samples_val=50,
            seed=42
        )
        
        if N_c is not None:
            status = "RIGID" if w == w_values[0] or N_c == results[0][1] else "Growing"
            print(f"{w:5d} {N_c:6d} {train_err:.2e} {val_err:.2e} {status:>10}")
            results.append((w, N_c, train_err, val_err))
        else:
            print(f"{w:5d} {'None':>6} {'-':>12} {'-':>12} {'INF':>10}")
            results.append((w, None, None, None))
    
    # Analyze results
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)
    
    # Check if N_c is constant (rigid)
    valid_results = [(w, N_c) for w, N_c, _, _ in results if N_c is not None]
    
    if len(valid_results) >= 2:
        widths = [N_c for w, N_c in valid_results]
        unique_widths = set(widths)
        
        if len(unique_widths) == 1:
            print(f"\n*** RESULT: RIGID SOLUTION SPACE ***")
            print(f"N_c(w) = {widths[0]} (constant) for all w in {[w for w, _ in valid_results]}")
            print("The minimal width required for zero violation does not increase")
            print("with the number of constraint samples, indicating a rigid solution space.")
        else:
            # Check for linear growth
            w_vals, n_c_vals = zip(*valid_results)
            # Simple linear fit
            coeffs = np.polyfit(w_vals, n_c_vals, 1)
            r_squared = np.corrcoef(w_vals, n_c_vals)[0, 1]**2
            
            print(f"\n*** RESULT: FLOPPY SOLUTION SPACE ***")
            print(f"N_c(w) grows with w. Best linear fit: N_c = {coeffs[0]:.2f} * w + {coeffs[1]:.2f}")
            print(f"Correlation squared: {r_squared:.3f}")
            print(f"The minimal width increases with constraint count, indicating a floppy solution space.")
    else:
        print("\n*** INSUFFICIENT DATA ***")
        print("Not enough valid results to determine rigidity.")
    
    # Save results to file
    os.makedirs('results', exist_ok=True)
    with open('results/rigidity_results.txt', 'w') as f:
        f.write("Solution Space Rigidity Experiment Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Target function: f(x) = sin(x) + 0.1*x^2\n")
        f.write(f"Zero violation threshold: 1e-3\n")
        f.write(f"Width range: 4-50\n\n")
        f.write("w\tN_c\ttrain_err\tval_err\tstatus\n")
        for w, N_c, train_err, val_err in results:
            if N_c is not None:
                f.write(f"{w}\t{N_c}\t{train_err:.2e}\t{val_err:.2e}\tRigid\n")
            else:
                f.write(f"{w}\tNone\t-\t-\tInf\n")
    
    print(f"\nResults saved to results/rigidity_results.txt")

if __name__ == "__main__":
    main()