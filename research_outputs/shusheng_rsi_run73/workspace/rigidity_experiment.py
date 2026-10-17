#!/usr/bin/env python3
"""
Solution Space Rigidity Experiment for Small Feedforward Networks

Tests whether the minimum network width N_c needed to achieve zero violation
(max abs error <= 1e-3) on held-out constraint points grows with the number
of constraint samples w.

Three target families:
1. Polynomial (algebraic/computation) - expected rigid (constant N_c)
2. Low-dimensional manifold (geometry) - expected slow growth
3. Oscillatory function (optimization) - expected fat (growing N_c)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Optional, Dict
import json
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class FeedforwardNet(nn.Module):
    """Small feedforward network with ReLU activation."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def generate_polynomial_samples(x: np.ndarray, d: int) -> np.ndarray:
    """Generate polynomial target of degree d with random coefficients."""
    coeffs = np.random.randn(d + 1)
    # Construct polynomial using Horner's method
    y = np.zeros_like(x)
    for i in range(d + 1):
        y = y * x + coeffs[i]
    return y, coeffs

def generate_manifold_samples(x: np.ndarray, k: int = 2, dim: int = 5) -> np.ndarray:
    """Generate target on a k-dimensional manifold embedded in R^dim."""
    # Project onto a k-dimensional subspace
    subspace = np.random.randn(dim, k)
    subspace, _ = np.linalg.qr(subspace)  # Orthogonalize
    x_reduced = x @ subspace  # Map to k-dim
    # Apply a smooth function on the reduced space
    y = np.sum(x_reduced**2, axis=1)  # Quadratic form on manifold
    return y

def generate_oscillatory_samples(x: np.ndarray, freq: int = 10) -> np.ndarray:
    """Generate highly oscillatory target function."""
    y = np.sin(freq * x * np.pi) * np.exp(-x**2 / 4)
    return y

def train_network(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 5000,
    lr: float = 1e-3,
    batch_size: int = 32
) -> Tuple[float, float]:
    """Train network and return train and validation errors."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        # Mini-batch training
        indices = np.random.permutation(len(x_train))[:batch_size]
        x_batch = x_train[indices]
        y_batch = y_train[indices]

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    # Evaluate on full train and val sets
    model.eval()
    with torch.no_grad():
        y_train_pred = model(x_train)
        y_val_pred = model(x_val)

        train_mse = criterion(y_train_pred, y_train).item()
        val_mse = criterion(y_val_pred, y_val).item()

        train_mae = torch.abs(y_train_pred - y_train).mean().item()
        val_mae = torch.abs(y_val_pred - y_val).max().item()  # max abs error for validation

    return train_mae, val_mae

def find_nc_w(
    target_func,
    w: int,
    width_range: List[int],
    n_trials: int = 3,
    max_abs_error_threshold: float = 1e-3,
    val_ratio: float = 0.2
) -> Optional[int]:
    """
    Find the minimum network width N that achieves zero violation on held-out points.
    Returns None if no width in the range achieves zero violation.
    """
    for width in width_range:
        for trial in range(n_trials):
            # Generate samples
            x_all = np.random.uniform(-1, 1, w + int(w * val_ratio))
            np.random.shuffle(x_all)

            x_train = x_all[:w]
            x_val = x_all[w:]

            y_train = target_func(x_train)
            y_val = target_func(x_val)

            # Convert to tensors
            x_train_t = torch.tensor(x_train, dtype=torch.float32).view(-1, 1)
            y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            x_val_t = torch.tensor(x_val, dtype=torch.float32).view(-1, 1)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

            # Create and train network
            model = FeedforwardNet(input_dim=1, hidden_dim=width)
            train_mae, val_mae = train_network(
                model, x_train_t, y_train_t, x_val_t, y_val_t,
                epochs=5000, lr=1e-3, batch_size=min(32, w)
            )

            # Check zero violation
            if val_mae <= max_abs_error_threshold:
                return width

    return None  # No width achieved zero violation

def run_experiment():
    """Run the full rigidity experiment."""
    # Scan parameters
    w_values = [10, 20, 50, 100, 200, 500, 1000]
    width_range = list(range(5, 201))  # Widths from 5 to 200
    max_abs_error_threshold = 1e-3

    # Target families
    targets = {
        'polynomial_d1': (lambda x: generate_polynomial_samples(x, d=1)[0], 'Polynomial deg=1'),
        'polynomial_d2': (lambda x: generate_polynomial_samples(x, d=2)[0], 'Polynomial deg=2'),
        'polynomial_d3': (lambda x: generate_polynomial_samples(x, d=3)[0], 'Polynomial deg=3'),
        'manifold_k2': (lambda x: generate_manifold_samples(x, k=2, dim=5)[0], 'Manifold k=2'),
        'oscillatory_freq5': (lambda x: generate_oscillatory_samples(x, freq=5)[0], 'Oscillatory freq=5'),
        'oscillatory_freq10': (lambda x: generate_oscillatory_samples(x, freq=10)[0], 'Oscillatory freq=10'),
    }

    results = {}

    for name, (target_func, desc) in targets.items():
        print(f"\n{'='*60}")
        print(f"Target: {desc} ({name})")
        print(f"{'='*60}")

        nc_w = []
        for w in w_values:
            nc = find_nc_w(target_func, w, width_range, n_trials=2, max_abs_error_threshold=max_abs_error_threshold)
            nc_w.append(nc)
            print(f"  w={w:4d}: N_c = {nc if nc is not None else 'None (no width reached threshold)'}")

        results[name] = {
            'description': desc,
            'w_values': w_values,
            'nc_w': nc_w,
            'rigid': all(nc is not None and nc <= 20 for nc in nc_w),  # heuristic for rigid
            'fat': all(nc is None for nc in nc_w)  # heuristic for fat
        }

        # Save intermediate results
        os.makedirs('/workspace/results', exist_ok=True)
        with open('/workspace/results/rigidity_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, res in results.items():
        nc_w = res['nc_w']
        finite_nc = [nc for nc in nc_w if nc is not None]
        if finite_nc:
            trend = "constant" if len(set(finite_nc)) == 1 else "growing"
            print(f"{name}: N_c values = {finite_nc}, trend={trend}, rigid={res['rigid']}, fat={res['fat']}")
        else:
            print(f"{name}: N_c values = all None, trend=none, rigid=False, fat=True")

    return results

if __name__ == '__main__':
    results = run_experiment()