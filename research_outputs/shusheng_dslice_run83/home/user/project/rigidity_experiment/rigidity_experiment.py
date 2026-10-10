#!/usr/bin/env python3
"""
Solution Space Rigidity Experiment for Small Feedforward Networks

Tests whether N_c(w) (minimum width achieving zero violation on held-out constraints)
scales with constraint count w. Three competing predictions:
  - Rigidity (structure): N_c(w) = constant
  - Floppy (optimization): N_c(w) ~ linear in w
  - Intermediate (measure): N_c(w) ~ logarithmic or sublinear in w

All experiments use ReLU single-hidden-layer networks trained with L-BFGS/BFGS
for precise convergence to the zero-violation threshold.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
import json
import os
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
np.random.seed(42)
torch.manual_seed(42)

# Zero violation threshold
EPSILON = 1e-3

# Network architecture search
WIDTH_MIN = 1
WIDTH_MAX = 50  # Scan this range for N_c
WIDTH_STEP = 1

# Constraint sampling
W_VALUES = [10, 20, 50, 100, 200, 500, 1000]  # constraint counts
X_TRAIN_MIN, X_TRAIN_MAX = -2.0, 2.0
X_VAL_MIN, X_VAL_MAX = -2.0, 2.0
VAL_RATIO = 0.2  # proportion of samples held out for validation

# Target function families (algebraic structure)
TARGET_FAMILIES = {
    'poly3': lambda x: 0.5 * x**3 - 0.5 * x + 0.2 * np.sin(5 * x),  # degree-3 polynomial + sinusoid
    'poly2': lambda x: x**2 - 1.0,  # simple quadratic
    'sin_sum': lambda x: np.sin(x) + 0.5 * np.sin(3 * x),  # sum of sinusoids
}

# Training settings
MAX_EPOCHS = 5000
LEARNING_RATE = 0.01
PATIENCE = 100


class SingleLayerReLU(nn.Module):
    """Single-hidden-layer ReLU feedforward network."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x


def generate_target_data(
    target_func, n_samples: int, x_min: float, x_max: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate input-output pairs from a target function."""
    x = np.linspace(x_min, x_max, n_samples)
    y = target_func(x)
    return x.reshape(-1, 1), y


def train_network(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    device: str = 'cpu',
    max_epochs: int = MAX_EPOCHS,
) -> float:
    """Train a network using L-BFGS for precise convergence."""
    model.to(device)
    model.train()

    x_torch = torch.FloatTensor(x_train).to(device)
    y_torch = torch.FloatTensor(y_train).view(-1, 1).to(device)

    # Use L-BFGS via scipy for better convergence on small problems
    # Convert parameters to flat array
    params = list(model.parameters())
    param_sizes = [p.numel() for p in params]
    flat_params = torch.cat([p.view(-1) for p in params]).detach().numpy()

    def closure():
        optimizer.zero_grad()
        output = model(x_torch)
        loss = nn.MSELoss()(output, y_torch)
        loss.backward()
        return loss.item()

    optimizer = optim.LBFGS(model.parameters(), lr=0.1)

    for epoch in range(max_epochs):
        optimizer.step(closure)
        if epoch % 1000 == 0:
            loss_val = closure()
            if loss_val < 1e-10:
                break

    # Convert back to model parameters
    flat_params = torch.cat([p.view(-1) for p in params]).detach().numpy()
    offset = 0
    for i, p in enumerate(params):
        p.data = torch.FlatParameterView(flat_params[offset:offset + param_sizes[i]], p.shape)
        offset += param_sizes[i]

    return float(closure())


def validate_model(
    model: nn.Module,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: str = 'cpu',
) -> float:
    """Compute maximum absolute error on validation set."""
    model.eval()
    with torch.no_grad():
        x_torch = torch.FloatTensor(x_val).to(device)
        y_torch = torch.FloatTensor(y_val).view(-1, 1).to(device)
        output = model(x_torch)
        errors = torch.abs(output - y_val).numpy().flatten()
        return float(np.max(errors))


def find_nc_w(
    target_func,
    w: int,
    device: str = 'cpu',
    n_trials: int = 3,
) -> Optional[int]:
    """
    Find N_c(w): minimum width achieving zero violation on held-out set.
    Returns None if no width in range achieves zero violation.
    """
    # Generate w constraint samples
    x_train, y_train = generate_target_data(
        target_func, w, X_TRAIN_MIN, X_TRAIN_MAX
    )

    # Hold out a validation set (same distribution)
    n_val = int(w * VAL_RATIO)
    if n_val < 5:
        n_val = 5

    # Create validation indices
    val_indices = np.random.choice(w, size=n_val, replace=False)
    train_indices = np.setdiff1d(np.arange(w), val_indices)

    x_val = x_train[val_indices]
    y_val = y_train[val_indices]
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]

    # Scan widths
    for width in range(WIDTH_MIN, WIDTH_MAX + 1, WIDTH_STEP):
        zero_violation_found = False

        for trial in range(n_trials):
            # Initialize network with specific width
            model = SingleLayerReLU(input_dim=1, hidden_dim=width, output_dim=1)

            # Initialize weights carefully
            nn.init.xavier_uniform_(model.hidden.weight)
            nn.init.zeros_(model.hidden.bias)
            nn.init.xavier_uniform_(model.output.weight)
            nn.init.zeros_(model.output.bias)

            # Train
            train_error = train_network(model, x_train, y_train, device=device)

            # Validate
            val_error = validate_model(model, x_val, y_val, device=device)

            if val_error <= EPSILON:
                zero_violation_found = True
                break

        if zero_violation_found:
            return width  # This is N_c(w)

    return None  # No width achieved zero violation


def run_experiment(
    target_name: str,
    target_func,
    output_dir: str = '/home/user/project/rigidity_experiment',
) -> Dict:
    """Run full experiment for one target function."""
    os.makedirs(output_dir, exist_ok=True)

    result = {
        'target': target_name,
        'function': target_func.__name__,
        'Nc_w_values': [],
        'w_values': [],
        'experiment_config': {
            'width_range': [WIDTH_MIN, WIDTH_MAX],
            'w_values': W_VALUES,
            'epsilon': EPSILON,
        },
    }

    print(f"\n=== Running experiment for {target_name} ===")

    for w in W_VALUES:
        print(f"  w = {w}: Finding N_c(w)...")
        nc = find_nc_w(target_func, w)
        result['w_values'].append(w)
        result['Nc_w_values'].append(nc if nc is not None else -1)  # -1 indicates not found
        print(f"    N_c(w) = {nc}")

    # Save results
    output_file = os.path.join(output_dir, f'rigidity_{target_name}.json')
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {output_file}")
    return result


def analyze_results(results: List[Dict]) -> None:
    """Analyze and compare results across target functions."""
    print("\n=== Analysis ===")

    for result in results:
        target = result['target']
        w_vals = result['w_values']
        nc_vals = result['Nc_w_values']

        print(f"\nTarget: {target}")
        print(f"  w: {w_vals}")
        print(f"  N_c(w): {nc_vals}")

        # Check for rigidity (constant N_c)
        valid_nc = [nc for nc in nc_vals if nc > 0]
        if len(valid_nc) >= 2:
            nc_mean = np.mean(valid_nc)
            nc_std = np.std(valid_nc)
            print(f"  Valid N_c values: {valid_nc}")
            print(f"  Mean: {nc_mean:.2f}, Std: {nc_std:.2f}")

            # Check if approximately constant
            if nc_std < 1.0 and nc_mean <= WIDTH_MAX:
                print("  => CONSISTENT WITH RIGIDITY (N_c constant)")
            else:
                print("  => NOT CONSISTENT WITH RIGIDITY")

        # Check for linear growth
        if len(valid_nc) >= 3 and all(nc > 0 for nc in valid_nc):
            w_arr = np.array([w_vals[i] for i in range(len(w_vals)) if nc_vals[i] > 0])
            nc_arr = np.array([nc_vals[i] for i in range(len(nc_vals)) if nc_vals[i] > 0])
            # Simple linear fit
            coeffs = np.polyfit(w_arr, nc_arr, 1)
            print(f"  Linear fit: N_c = {coeffs[0]:.4f} * w + {coeffs[1]:.4f}")
            if abs(coeffs[0]) > 0.1:  # Significant slope
                print("  => CONSISTENT WITH LINEAR GROWTH (floppy)")

        # Check for logarithmic growth
        if len(valid_nc) >= 3 and all(nc > 0 for nc in valid_nc):
            w_arr = np.array([w_vals[i] for i in range(len(w_vals)) if nc_vals[i] > 0])
            nc_arr = np.array([nc_vals[i] for i in range(len(nc_vals)) if nc_vals[i] > 0])
            # Log fit
            log_coeffs = np.polyfit(np.log(w_arr + 1), nc_arr, 1)
            print(f"  Log fit: N_c = {log_coeffs[0]:.4f} * log(w+1) + {log_coeffs[1]:.4f}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    device = 'cpu'  # Use GPU if available and if torch.cuda.is_available()

    # Run experiments for all target families
    all_results = []
    for name, func in TARGET_FAMILIES.items():
        result = run_experiment(name, func, device=device)
        all_results.append(result)

    # Analyze results
    analyze_results(all_results)

    print("\n=== Experiment Complete ===")