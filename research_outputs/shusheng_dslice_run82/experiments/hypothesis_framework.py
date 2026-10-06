#!/usr/bin/env python3
"""
Framework for testing mutually discriminable hypotheses on solution space rigidity.
Tests three different constraint generation processes:
1. Manifold constraints (Geometry dimension)
2. GP constraints with finite smoothness (Measure dimension)  
3. Random/unstructured constraints (Optimization dimension)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# Target functions for different constraint generation processes
# ============================================================

class ManifoldTarget:
    """Target function on a 1D circle embedded in 2D."""
    def __init__(self, d=2):
        self.d = d
    
    def __call__(self, x):
        # x shape: (w, d) - points on circle in 2D
        # Project to angle, then apply smooth function
        angles = np.arctan2(x[:, 1], x[:, 0])
        return np.sin(angles) + 0.1 * angles**2

class GPMatern32Target:
    """Sample from GP with Matérn-3/2 kernel."""
    def __init__(self, d=1, sigma=1.0, length_scale=1.0, seed=42):
        self.d = d
        self.sigma = sigma
        self.ls = length_scale
        self.rng = np.random.default_rng(seed)
    
    def __call__(self, x):
        # Simple 1D Matérn-3/2 GP sample using spectral representation
        # For simplicity, use a smooth deterministic proxy
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        # Matérn-3/2 kernel: k(r) = sigma^2 * (1 + sqrt(3)*r/ls) * exp(-sqrt(3)*r/ls)
        # Use a function in the RKHS: f(x) = sum a_i * k(x, x_i)
        n_samples = min(50, len(x))
        centers = self.rng.uniform(-2, 2, n_samples)
        coeffs = self.rng.normal(0, 1, n_samples)
        result = np.zeros(len(x))
        for i, xi in enumerate(x):
            r = np.abs(xi - centers)
            k_val = self.sigma**2 * (1 + np.sqrt(3) * r / self.ls) * np.exp(-np.sqrt(3) * r / self.ls)
            result[i] = np.sum(coeffs * k_val)
        return result if len(x) > 1 else result[0]

class RandomTarget:
    """Random high-frequency function (unstructured constraints)."""
    def __init__(self, d=1, seed=42):
        self.d = d
        self.rng = np.random.default_rng(seed)
    
    def __call__(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        # Sum of many high-frequency sinusoids
        freqs = self.rng.uniform(5, 20, 20)
        coeffs = self.rng.normal(0, 1, 20)
        result = np.zeros(len(x))
        for f, c in zip(freqs, coeffs):
            result += c * np.sin(f * x[:, 0])
        return result

# ============================================================
# Neural Network and Training
# ============================================================

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

def train_model(model, x_train, y_train, x_val, y_val, lr=0.01, max_epochs=10000, patience=100):
    dataset = ConstraintDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=min(32, len(x_train)), shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    best_val_err = float('inf')
    patience_counter = 0
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        for xb, yb in loader:
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(x_val))
            val_max_err = float(torch.max(torch.abs(val_pred - torch.FloatTensor(y_val))).numpy())
        
        if val_max_err < best_val_err:
            best_val_err = val_max_err
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    with torch.no_grad():
        train_pred = model(torch.FloatTensor(x_train))
        train_max_err = float(torch.max(torch.abs(train_pred - torch.FloatTensor(y_train))).numpy())
    
    return train_max_err, best_val_err

class ConstraintDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

# ============================================================
# Find N_c(w)
# ============================================================

def find_nc_w(w, target_func, input_dim=1, width_range=list(range(4, 51)), 
              n_samples_val=50, seed=42, activation='relu'):
    np.random.seed(seed + w)
    
    # Generate training constraints
    x_train = np.random.uniform(-2, 2, (w, input_dim))
    y_train = target_func(x_train)
    
    # Generate out-of-sample validation constraints
    np.random.seed(seed + w + 1000)
    x_val = np.random.uniform(-2, 2, (n_samples_val, input_dim))
    y_val = target_func(x_val)
    
    for N in width_range:
        model = TwoLayerNet(input_dim, N, activation=activation)
        train_err, val_err = train_model(model, x_train, y_train, x_val, y_val, 
                                          lr=0.01, max_epochs=10000, patience=200)
        
        if val_err <= 1e-3:
            return N, train_err, val_err
    
    return None, None, None

# ============================================================
# Main experiment runner
# ============================================================

def run_hypothesis_test(hypothesis_type, w_values, activation='relu'):
    """Run experiment for a specific hypothesis type."""
    print(f"\n{'='*70}")
    print(f"Testing Hypothesis: {hypothesis_type}")
    print(f"{'='*70}")
    
    if hypothesis_type == 'manifold':
        target = ManifoldTarget(d=2)
        input_dim = 2
    elif hypothesis_type == 'gp_matern32':
        target = GPMatern32Target(d=1, seed=42)
        input_dim = 1
    elif hypothesis_type == 'random':
        target = RandomTarget(d=1, seed=42)
        input_dim = 1
    else:
        raise ValueError(f"Unknown hypothesis type: {hypothesis_type}")
    
    results = []
    print(f"\nw values: {w_values}")
    print(f"{'w':>5} {'N_c':>6} {'train_err':>12} {'val_err':>12} {'status':>10}")
    print("-" * 50)
    
    for w in w_values:
        N_c, train_err, val_err = find_nc_w(w, target, input_dim=input_dim, 
                                            width_range=list(range(4, 51)),
                                            n_samples_val=50, seed=42,
                                            activation=activation)
        
        if N_c is not None:
            print(f"{w:5d} {N_c:6d} {train_err:.2e} {val_err:.2e} {'FOUND':>10}")
            results.append((w, N_c, train_err, val_err))
        else:
            print(f"{w:5d} {'None':>6} {'-':>12} {'-':>12} {'NOT FOUND':>10}")
            results.append((w, None, None, None))
    
    # Analyze scaling
    valid_results = [(w, N_c) for w, N_c, _, _ in results if N_c is not None]
    if len(valid_results) >= 2:
        w_vals, n_c_vals = zip(*valid_results)
        coeffs = np.polyfit(np.log(w_vals), np.log(n_c_vals), 1)
        beta = coeffs[0]
        print(f"\nScaling exponent β = {beta:.3f} (from log-log fit)")
        return beta, results
    return None, results

if __name__ == "__main__":
    w_values = [10, 20, 50, 100, 200, 500]
    
    for htype in ['manifold', 'gp_matern32', 'random']:
        beta, results = run_hypothesis_test(htype, w_values)
        print(f"\nFinal result for {htype}: β = {beta}")
