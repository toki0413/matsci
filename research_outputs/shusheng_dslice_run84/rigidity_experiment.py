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
MIN_WIDTH = 4    # Minimum width to scan
WIDTH_STEP = 4   # Step size for width scan
MAX_W = 500      # Maximum number of constraint samples
MIN_W = 5        # Minimum number of constraint samples
W_STEP = 5       # Step size for w scan
NUM_TRIALS = 3   # Number of random trials per (w, width) combination
MAX_EPOCHS = 5000  # Maximum training epochs
LEARNING_RATE = 1e-3
PATIENCE = 100   # Early stopping patience

# Analytic function classes to test
class AnalyticFunction:
    """Base class for analytic functions used as constraint targets."""
    
    def __init__(self, name: str, dim: int = 1):
        self.name = name
        self.dim = dim
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the function at input x."""
        raise NotImplementedError
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)


class PolynomialFunction(AnalyticFunction):
    """Low-degree polynomial - represents a rigid solution space."""
    
    def __init__(self, coeffs: List[float]):
        super().__init__(f"poly_{len(coeffs)-1}")
        self.coeffs = np.array(coeffs)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        # Polynomial: c0 + c1*x + c2*x^2 + ...
        result = np.zeros_like(x, dtype=np.float64)
        for i, c in enumerate(self.coeffs):
            result += c * (x ** i)
        return result


class TrigonometricFunction(AnalyticFunction):
    """Smooth trigonometric function - moderate complexity."""
    
    def __init__(self, freq: float = 1.0, amp: float = 1.0):
        super().__init__(f"sin_{freq}_{amp}")
        self.freq = freq
        self.amp = amp
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.amp * np.sin(self.freq * x)


class HighFrequencyTrigFunction(AnalyticFunction):
    """High-frequency trigonometric function - represents a fat solution space."""
    
    def __init__(self, freq: float = 10.0, amp: float = 1.0):
        super().__init__(f"high_freq_sin_{freq}_{amp}")
        self.freq = freq
        self.amp = amp
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.amp * np.sin(self.freq * x)


class PiecewiseSmoothFunction(AnalyticFunction):
    """Piecewise smooth function with moderate complexity."""
    
    def __init__(self, n_pieces: int = 3):
        super().__init__(f"piecewise_{n_pieces}")
        self.n_pieces = n_pieces
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        # Create a piecewise function with different polynomials in each segment
        result = np.zeros_like(x, dtype=np.float64)
        for i in range(self.n_pieces):
            mask = (x >= i / self.n_pieces) & (x < (i + 1) / self.n_pieces)
            if np.any(mask):
                # Different polynomial in each segment
                coeffs = [0.1 * i, 0.5, 0.2 * i]
                result[mask] = coeffs[0] + coeffs[1] * x[mask] + coeffs[2] * x[mask]**2
        # Handle the last point
        mask = x >= (self.n_pieces - 1) / self.n_pieces
        if np.any(mask):
            coeffs = [0.1 * (self.n_pieces - 1), 0.5, 0.2 * (self.n_pieces - 1)]
            result[mask] = coeffs[0] + coeffs[1] * x[mask] + coeffs[2] * x[mask]**2
        return result


class NeuralNetwork(torch.nn.Module):
    """Small feedforward network with configurable hidden width."""
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, output_dim: int = 1, 
                 activation: str = "tanh"):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(torch.nn.Linear(input_dim, hidden_dim))
        if activation == "tanh":
            layers.append(torch.nn.Tanh())
        elif activation == "relu":
            layers.append(torch.nn.ReLU())
        elif activation == "sigmoid":
            layers.append(torch.nn.Sigmoid())
        
        # Hidden layers
        for _ in range(2):  # Fixed depth of 3 layers (input + 2 hidden)
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            if activation == "tanh":
                layers.append(torch.nn.Tanh())
            elif activation == "relu":
                layers.append(torch.nn.ReLU())
            elif activation == "sigmoid":
                layers.append(torch.nn.Sigmoid())
        
        # Output layer
        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        
        self.network = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def generate_constraint_points(func: AnalyticFunction, w: int, 
                              x_range: Tuple[float, float] = (0.0, 1.0),
                              test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate w constraint points and hold-out test points.
    
    Returns:
        train_x, train_y: training constraint points and targets
        test_x, test_y: held-out test points for generalization check
    """
    np.random.seed()  # Seed based on time for variety
    
    # Generate training points
    train_x = np.random.uniform(x_range[0], x_range[1], w)
    train_y = func.evaluate(train_x)
    
    # Generate test points (held-out)
    n_test = int(w * test_ratio)
    test_x = np.random.uniform(x_range[0], x_range[1], n_test)
    test_y = func.evaluate(test_x)
    
    return train_x, train_y, test_x, test_y


def train_network(model: torch.nn.Module, train_x: np.ndarray, train_y: np.ndarray,
                  test_x: np.ndarray, test_y: np.ndarray, 
                  epochs: int = MAX_EPOCHS, lr: float = LEARNING_RATE) -> Tuple[float, float]:
    """
    Train the network and return training and test errors.
    
    Returns:
        train_max_error, test_max_error (maximum absolute error)
    """
    device = torch.device("cpu")  # Use CPU for simplicity
    
    model.to(device)
    model.train()
    
    # Convert to tensors
    x_train = torch.FloatTensor(train_x).to(device).view(-1, 1)
    y_train = torch.FloatTensor(train_y).to(device).view(-1, 1)
    x_test = torch.FloatTensor(test_x).to(device).view(-1, 1)
    y_test = torch.FloatTensor(test_y).to(device).view(-1, 1)
    
    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop with early stopping
    best_test_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Early stopping check
        with torch.no_grad():
            test_outputs = model(x_test)
            test_loss = criterion(test_outputs, y_test).item()
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
    
    # Calculate maximum absolute errors
    with torch.no_grad():
        train_pred = model(x_train)
        test_pred = model(x_test)
        
        train_max_error = float(torch.max(torch.abs(train_pred - y_train)).item())
        test_max_error = float(torch.max(torch.abs(test_pred - y_test)).item())
    
    return train_max_error, test_max_error


def find_n_c_w(func: AnalyticFunction, w: int, width_range: Tuple[int, int] = (4, 256),
               width_step: int = 4, num_trials: int = NUM_TRIALS) -> Optional[int]:
    """
    Find the minimum width N_c that achieves zero violation on held-out points.
    
    Returns:
        N_c (minimum width) or None if not achievable within width_range
    """
    min_width, max_width = width_range
    
    for width in range(min_width, max_width + 1, width_step):
        all_trials_pass = True
        
        for trial in range(num_trials):
            # Generate constraint points
            train_x, train_y, test_x, test_y = generate_constraint_points(func, w)
            
            # Create and train network
            model = NeuralNetwork(input_dim=1, hidden_dim=width)
            train_error, test_error = train_network(model, train_x, train_y, test_x, test_y)
            
            # Check zero violation condition
            if test_error > ZERO_VIOLATION_THRESHOLD:
                all_trials_pass = False
                break
        
        if all_trials_pass:
            return width  # Return the minimum width that works
    
    return None  # None means not achievable within scan range


def run_experiment(func: AnalyticFunction, output_dir: str = ".") -> Dict[str, Any]:
    """
    Run the full experiment for a given analytic function.
    
    Returns:
        Dictionary with results including N_c(w) values
    """
    os.makedirs(output_dir, exist_ok=True)
    
    w_values = list(range(MIN_W, MAX_W + 1, W_STEP))
    results = {
        "function_name": func.name,
        "function_type": type(func).__name__,
        "w_values": w_values,
        "N_c_w": [],  # N_c for each w
        "summary": {}
    }
    
    print(f"\n{'='*60}")
    print(f"Experiment: {func.name}")
    print(f"{'='*60}")
    
    for w in w_values:
        N_c = find_n_c_w(func, w)
        results["N_c_w"].append(N_c)
        
        status = f"{N_c}" if N_c is not None else "INF (unattainable)"
        print(f"w={w:4d}: N_c = {status}")
        
        # Store intermediate results
        results["summary"][f"w_{w}"] = {
            "N_c": N_c,
            "attainable": N_c is not None
        }
    
    # Calculate summary statistics
    attainable_N_c = [nc for nc in results["N_c_w"] if nc is not None]
    if attainable_N_c:
        results["summary"]["mean_N_c"] = np.mean(attainable_N_c)
        results["summary"]["min_N_c"] = min(attainable_N_c)
        results["summary"]["max_N_c"] = max(attainable_N_c)
        
        # Check if N_c is constant in w (rigid) vs growing (fat)
        if len(attainable_N_c) > 1:
            results["summary"]["N_c_constant"] = (max(attainable_N_c) - min(attainable_N_c)) == 0
    
    # Save results
    result_file = os.path.join(output_dir, f"rigidity_{func.name}.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {result_file}")
    return results


def main():
    """Main function to run all experiments."""
    output_dir = "/workspace/research_outputs/shusheng_dslice_run84"
    
    # Define the analytic functions to test
    # Function 1: Low-degree polynomial - expected to be RIGID
    func1 = PolynomialFunction(coeffs=[0.5, 2.0, -1.0, 0.5])  # Quadratic + cubic
    
    # Function 2: Low-frequency sine - expected to be RIGID or borderline
    func2 = TrigonometricFunction(freq=2.0, amp=1.0)
    
    # Function 3: High-frequency sine - expected to be FAT
    func3 = HighFrequencyTrigFunction(freq=15.0, amp=1.0)
    
    # Function 4: Piecewise smooth - intermediate case
    func4 = PiecewiseSmoothFunction(n_pieces=4)
    
    functions = [func1, func2, func3, func4]
    names = ["poly_3", "sin_2_1", "high_freq_sin_15_1", "piecewise_4"]
    
    all_results = {}
    
    for i, func in enumerate(functions):
        print(f"\n\n{'#'*60} Testing {func.name} {'#'*60}\n")
        result = run_experiment(func, output_dir)
        all_results[names[i]] = result
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE: N_c(w) across functions and constraint counts")
    print(f"{'='*80}")
    
    # Create summary table
    print(f"\n{'Function':<25}", end="")
    for w in list(range(MIN_W, MAX_W + 1, W_STEP))[:8]:  # Show first 8 w values
        print(f"{w:>8}", end="")
    print()
    
    for name in names:
        result = all_results[name]
        print(f"{name:<25}", end="")
        for nc in result["N_c_w"][:8]:  # Show first 8 N_c values
            status = f"{nc:>4}" if nc is not None else " INF"
            print(status, end="")
        print()
    
    print(f"\n{'='*80}")
    
    # Determine rigidity/fatness for each function
    print("\nRIGIDITY/FATNESS DETERMINATION:")
    print("-" * 80)
    
    for name in names:
        result = all_results[name]
        attainable = [nc for nc in result["N_c_w"] if nc is not None]
        
        if not attainable:
            print(f"{name}: FAT (N_c unattainable for all w in scan range)")
        elif len(attainable) > 0 and max(attainable) - min(attainable) == 0:
            print(f"{name}: RIGID (N_c = {attainable[0]} constant across w)")
        else:
            # Check if N_c grows with w
            w_values = result["w_values"][:len(attainable)]
            growth_rate = (max(attainable) - min(attainable)) / (max(w_values) - min(w_values)) if len(w_values) > 1 else 0
            if growth_rate < 0.1:  # Slow growth
                print(f"{name}: BORDERLINE (N_c ranges from {min(attainable)} to {max(attainable)})")
            else:
                print(f"{name}: FAT (N_c grows from {min(attainable)} to {max(attainable)})")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()