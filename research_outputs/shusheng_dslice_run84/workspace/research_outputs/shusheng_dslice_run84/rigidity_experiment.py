#!/usr/bin/env python3
"""
Solution Space Rigidity Experiment for Small Feedforward Networks

This experiment probes whether small feedforward networks can serve as
probes of solution space rigidity through their generalization behavior.

Key definitions:
- w = number of constraint samples (points (x_i, y_i) the network must fit)
- N_c(w) = minimum hidden layer width achieving zero violation on held-out points
- Zero violation = max absolute error on held-out points <= 1e-3
- Rigid: N_c(w) is small and constant in w
- Fat: N_c(w) grows without bound or is unattainable within scan range
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Optional, Dict, Any
import json
import os

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configuration
ZERO_VIOLATION_THRESHOLD = 1e-3
MAX_WIDTH = 256  # Maximum width to scan
WIDTH_START = 4
WIDTH_STEP = 4
W_MIN = 5
W_MAX = 100
W_STEP = 5
MAX_EPOCHS = 5000
LEARNING_RATE = 1e-3
HOLDOUT_RATIO = 0.2
NUM_TRIALS = 3


class NeuralNetwork(nn.Module):
    """3-layer feedforward network with optional skip connection."""

    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, use_skip=False, activation='tanh'):
        super().__init__()
        self.use_skip = use_skip
        if activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Tanh()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, output_dim)
        )

        if use_skip:
            self.skip = nn.Linear(input_dim, output_dim)
        else:
            self.skip = None

    def forward(self, x):
        if self.skip is not None:
            return self.network(x) + self.skip(x)
        return self.network(x)


def train_network(model: nn.Module, train_x: np.ndarray, train_y: np.ndarray,
                  val_x: np.ndarray, val_y: np.ndarray, lr: float = 1e-3,
                  max_epochs: int = 5000) -> Tuple[float, float, List[float]]:
    """Train network and return training and validation errors."""
    x_train = torch.FloatTensor(train_x).view(-1, 1)
    y_train = torch.FloatTensor(train_y).view(-1, 1)
    x_val = torch.FloatTensor(val_x).view(-1, 1)
    y_val = torch.FloatTensor(val_y).view(-1, 1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_errors = []
    val_errors = []

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_err = torch.mean(torch.abs(outputs - y_train)).item()
            val_outputs = model(x_val)
            val_err = torch.mean(torch.abs(val_outputs - y_val)).item()

        train_errors.append(train_err)
        val_errors.append(val_err)

        # Early stopping if validation error is very small
        if val_err < 1e-6 and epoch > 100:
            break

    # Final evaluation
    with torch.no_grad():
        train_final = torch.mean(torch.abs(model(x_train) - y_train)).item()
        val_final = torch.mean(torch.abs(model(x_val) - y_val)).item()
        max_val_err = torch.max(torch.abs(model(x_val) - y_val)).item()

    return train_final, val_final, max_val_err, train_errors, val_errors


def find_minimum_width(func_name: str, w: int, width_range: List[int],
                       holdout_ratio: float = 0.2, num_trials: int = 3,
                       max_epochs: int = 5000) -> Optional[int]:
    """Find minimum width that achieves zero violation on held-out points."""
    x_all = np.linspace(0, 1, 1000)  # Full domain for evaluation

    for width in width_range:
        trial_max_errors = []
        for trial in range(num_trials):
            # Generate constraint points
            np.random.seed(42 + w * 1000 + width * 100 + trial * 10)
            train_indices = np.random.choice(len(x_all), w, replace=False)
            x_train = x_all[train_indices]

            # Get function values
            if func_name == 'polynomial':
                y_train = 0.5 + 2.0 * x_train - 1.0 * x_train**2 + 0.5 * x_train**3
            elif func_name == 'sine':
                y_train = np.sin(2 * np.pi * x_train * 10)
            elif func_name == 'exponential':
                y_train = np.exp(-5 * x_train) * np.cos(2 * np.pi * x_train)
            elif func_name == 'mixed':
                y_train = (0.5 + 2.0 * x_train - 1.0 * x_train**2 + 0.5 * x_train**3 +
                           0.3 * np.sin(20 * np.pi * x_train))
            else:
                y_train = 0.5 + 2.0 * x_train - 1.0 * x_train**2 + 0.5 * x_train**3

            # Split into train and holdout
            holdout_size = int(len(x_train) * holdout_ratio)
            if holdout_size < 1:
                holdout_size = 1
            holdout_idx = np.random.choice(len(x_train), holdout_size, replace=False)
            train_idx = np.array([i for i in range(len(x_train)) if i not in holdout_idx])

            x_train_final = x_all[train_indices[train_idx]]
            y_train_final = y_train[train_idx]
            x_holdout = x_all[train_indices[holdout_idx]]
            y_holdout = y_train[holdout_idx]

            # Also create a dense test set for evaluation
            x_test = x_all

            # Build and train network
            model = NeuralNetwork(input_dim=1, hidden_dim=width, output_dim=1)
            train_err, val_err, max_val_err, _, _ = train_network(
                model, x_train_final, y_train_final, x_holdout, y_holdout,
                lr=LEARNING_RATE, max_epochs=max_epochs
            )

            trial_max_errors.append(max_val_err)

            if max_val_err <= ZERO_VIOLATION_THRESHOLD:
                print(f"  w={w}, width={width}, trial={trial}, max_val_err={max_val_err:.6e}")

        # Check if all trials achieved zero violation
        if all(err <= ZERO_VIOLATION_THRESHOLD for err in trial_max_errors):
            print(f"  w={w}: N_c={width} (all trials achieved zero violation)")
            return width

        # Check if best trial is close to threshold
        best_err = min(trial_max_errors)
        print(f"  w={w}, width={width}, best_err={best_err:.6e} (not yet zero violation)")

    return None  # No width achieved zero violation


def run_experiment(func_name: str) -> Dict[str, Any]:
    """Run full experiment for a given function."""
    print(f"\n{'='*60}")
    print(f"Experiment: {func_name}")
    print(f"{'='*60}")

    width_range = list(range(WIDTH_START, MAX_WIDTH + 1, WIDTH_STEP))
    w_values = list(range(W_MIN, W_MAX + 1, W_STEP))

    results = {
        'function': func_name,
        'width_range': width_range,
        'w_values': w_values,
        'N_c_values': [],  # N_c(w) for each w
        'training_errors': [],  # Training errors at N_c
        'holdout_errors': [],   # Holdout errors at N_c
    }

    for w in w_values:
        print(f"\nScanning w = {w}...")
        N_c = find_minimum_width(func_name, w, width_range,
                                holdout_ratio=HOLDOUT_RATIO,
                                num_trials=NUM_TRIALS,
                                max_epochs=MAX_EPOCHS)

        results['N_c_values'].append(N_c if N_c is not None else None)

        # Also record errors at the best width for this w
        if N_c is not None:
            # Re-run with best width to get final error stats
            np.random.seed(42 + w * 1000)
            x_all = np.linspace(0, 1, 1000)
            train_indices = np.random.choice(len(x_all), w, replace=False)
            x_train = x_all[train_indices]

            if func_name == 'polynomial':
                y_train = 0.5 + 2.0 * x_train - 1.0 * x_train**2 + 0.5 * x_train**3
            elif func_name == 'sine':
                y_train = np.sin(2 * np.pi * x_train * 10)
            elif func_name == 'exponential':
                y_train = np.exp(-5 * x_train) * np.cos(2 * np.pi * x_train)
            elif func_name == 'mixed':
                y_train = (0.5 + 2.0 * x_train - 1.0 * x_train**2 + 0.5 * x_train**3 +
                           0.3 * np.sin(20 * np.pi * x_train))

            holdout_size = int(w * HOLDOUT_RATIO)
            if holdout_size < 1:
                holdout_size = 1
            holdout_idx = np.random.choice(len(x_train), holdout_size, replace=False)
            train_idx = np.array([i for i in range(len(x_train)) if i not in holdout_idx])

            x_train_final = x_all[train_indices[train_idx]]
            y_train_final = y_train[train_idx]
            x_holdout = x_all[train_indices[holdout_idx]]
            y_holdout = y_train[holdout_idx]

            model = NeuralNetwork(input_dim=1, hidden_dim=N_c, output_dim=1)
            train_err, val_err, max_val_err, _, _ = train_network(
                model, x_train_final, y_train_final, x_holdout, y_holdout,
                lr=LEARNING_RATE, max_epochs=MAX_EPOCHS
            )

            results['training_errors'].append(train_err)
            results['holdout_errors'].append(max_val_err)

            print(f"  Final results: w={w}, N_c={N_c}, train_err={train_err:.6e}, max_val_err={max_val_err:.6e}")
        else:
            results['training_errors'].append(None)
            results['holdout_errors'].append(None)
            print(f"  w={w}: No width achieved zero violation (N_c not reachable)")

    return results


def save_results(results: Dict[str], filepath: str):
    """Save results to JSON file."""
    # Convert None to string for JSON serialization
    serializable = {}
    for k, v in results.items():
        if isinstance(v, list):
            serializable[k] = [x if x is not None else 'None' for x in v]
        else:
            serializable[k] = v
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {filepath}")


def main():
    """Run all experiments."""
    func_names = ['polynomial', 'sine', 'exponential', 'mixed']
    all_results = {}

    for func_name in func_names:
        results = run_experiment(func_name)
        all_results[func_name] = results
        filepath = f'rigidity_results_{func_name}.json'
        save_results(results, filepath)

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE: N_c(w) values")
    print("="*60)
    print(f"{'Function':<15} {'W':>5} {'N_c':>8}")
    print("-" * 30)
    for func_name in func_names:
        results = all_results[func_name]
        for i, w in enumerate(results['w_values']):
            N_c = results['N_c_values'][i]
            N_c_str = N_c if N_c is not None else 'inf'
            print(f"{func_name:<15} {w:>5} {N_c_str:>8}")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()