#!/usr/bin/env python3
"""
Rigidity Experiment: Neural Network Solution Space Rigidity Probe
Tests whether small feedforward networks can achieve zero violation
on held-out constraint points as the number of constraints w varies.
"""

import numpy as np
import sys
from typing import Tuple, Optional, List

# Set random seed for reproducibility
np.random.seed(42)


class SimpleMLP:
    """Simple MLP implemented from scratch using numpy."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        """Initialize MLP with one hidden layer."""
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Xavier/Glorot initialization
        W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        b1 = np.zeros(hidden_dim)
        W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        b2 = np.zeros(output_dim)
        
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with ReLU activation."""
        # x shape: (batch, input_dim)
        z1 = x @ self.W1 + self.b1  # (batch, hidden_dim)
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ self.W2 + self.b2  # (batch, output_dim)
        return z2
    
    def get_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.W1, self.b1, self.W2, self.b2
    
    def set_params(self, W1, b1, W2, b2):
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              lr: float = 1e-3, max_epochs: int = 10000, 
              tol: float = 1e-8, verbose: bool = False) -> List[float]:
        """Train the MLP using gradient descent."""
        losses = []
        n_samples = X.shape[0]
        
        for epoch in range(max_epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute MSE loss
            loss = np.mean((y_pred - y) ** 2)
            losses.append(loss)
            
            if verbose and epoch % 1000 == 0:
                print(f"  Epoch {epoch}, Loss = {loss:.6e}")
            
            # Check convergence
            if loss < tol:
                break
            
            # Backward pass
            m = n_samples
            dy_pred = 2 * (y_pred - y) / m  # derivative of MSE
            
            # Output layer gradients
            dW2 = dy_pred.T @ self.a1 if hasattr(self, 'a1') else None
            if dW2 is None:
                # Need to recompute forward pass for gradients
                z1 = X @ self.W1 + self.b1
                a1 = np.maximum(0, z1)
                self.a1 = a1
                z2 = a1 @ self.W2 + self.b2
                dy_pred = 2 * (z2 - y) / m
                dW2 = a1.T @ dy_pred  # (hidden_dim, output_dim)
            db2 = np.sum(dy_pred, axis=0)
            
            # Hidden layer gradients
            da1 = dy_pred @ self.W2.T
            dz1 = da1 * (self.a1 > 0)  # ReLU derivative
            dW1 = X.T @ dz1  # (input_dim, hidden_dim)
            db1 = np.sum(dz1, axis=0)
            
            # Update parameters
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
        
        return losses
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate on input X."""
        return self.forward(X)


def generate_constraint_function(func_type: str, n_points: int, 
                                  x_range: Tuple[float, float] = (0.0, 1.0)) -> Tuple[np.ndarray, np.ndarray]:
    """Generate constraint points from a synthetic function."""
    x = np.linspace(x_range[0], x_range[1], n_points)
    
    if func_type == "polynomial":
        # f(x) = 1 + 2x - 3x^2 + 4x^3 (degree 3 polynomial)
        y = 1 + 2*x - 3*x**2 + 4*x**3
    elif func_type == "sin":
        # f(x) = sin(π*x) - has algebraic structure with known zeros
        y = np.sin(np.pi * x)
    elif func_type == "sin_squared":
        # f(x) = sin^2(π*x) = (1 - cos(2π*x))/2
        y = np.sin(np.pi * x)**2
    elif func_type == "exp":
        # f(x) = exp(x) - transcendental, no simple algebraic relation
        y = np.exp(x)
    elif func_type == "rational":
        # f(x) = 1/(1 + x^2) - rational function
        y = 1 / (1 + x**2)
    else:
        raise ValueError(f"Unknown function type: {func_type}")
    
    return x, y


def find_min_width(X_train: np.ndarray, y_train: np.ndarray,
                   X_holdout: np.ndarray, y_holdout: np.ndarray,
                   max_width: int = 64, width_step: int = 1) -> Optional[int]:
    """
    Find the minimum hidden width that achieves zero violation on holdout points.
    Zero violation: max absolute error <= 1e-3 on holdout points.
    """
    widths = list(range(1, max_width + 1, width_step))
    
    for width in widths:
        # Initialize and train network
        net = SimpleMLP(input_dim=1, hidden_dim=width, output_dim=1)
        
        try:
            losses = net.train(X_train, y_train, lr=1e-3, max_epochs=5000, 
                              tol=1e-10, verbose=False)
            
            # Evaluate on holdout
            y_pred_holdout = net.evaluate(X_holdout)
            max_error = np.max(np.abs(y_pred_holdout - y_holdout))
            
            if max_error <= 1e-3:
                return width
                
        except Exception as e:
            print(f"  Width {width} failed: {e}")
            continue
    
    return None  # No width achieved zero violation


def run_experiment(func_type: str, w: int, max_width: int = 64, 
                   holdout_ratio: float = 0.2) -> dict:
    """Run a single experiment for a given w."""
    # Generate all constraint points
    all_n = int(w / (1 - holdout_ratio))
    x_all, y_all = generate_constraint_function(func_type, all_n)
    
    # Split into training and holdout
    indices = np.arange(len(x_all))
    np.random.shuffle(indices)
    
    train_n = int(w)
    train_indices = indices[:train_n]
    holdout_indices = indices[train_n:]
    
    X_train = x_all[train_indices].reshape(-1, 1)
    y_train = y_all[train_indices]
    X_holdout = x_all[holdout_indices].reshape(-1, 1)
    y_holdout = y_all[holdout_indices]
    
    # Find minimum width
    N_c = find_min_width(X_train, y_train, X_holdout, y_holdout, max_width=max_width)
    
    # Also record the best error for widths up to max_width
    best_error = float('inf')
    best_width = None
    
    for width in range(1, max_width + 1):
        try:
            net = SimpleMLP(input_dim=1, hidden_dim=width, output_dim=1)
            net.train(X_train, y_train, lr=1e-3, max_epochs=5000, tol=1e-10, verbose=False)
            y_pred_holdout = net.evaluate(X_holdout)
            max_error = np.max(np.abs(y_pred_holdout - y_holdout))
            if max_error < best_error:
                best_error = max_error
                best_width = width
        except:
            continue
    
    return {
        'w': w,
        'func_type': func_type,
        'N_c': N_c,
        'best_error': best_error,
        'best_width': best_width,
        'x_train': X_train,
        'y_train': y_train,
        'x_holdout': X_holdout,
        'y_holdout': y_holdout
    }


def main():
    """Run the rigidity experiment sweep."""
    print("Rigidity Experiment: Neural Network Solution Space Rigidity Probe")
    print("=" * 70)
    
    # Parameters
    func_type = "polynomial"  # Try "sin", "polynomial", "exp", etc.
    w_values = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
    max_width = 64
    
    print(f"\nFunction type: {func_type}")
    print(f"Width range: 1 to {max_width}")
    print(f"Held-out ratio: 20%")
    print(f"Zero violation threshold: 1e-3")
    print(f"w values: {w_values}\n")
    
    results = []
    
    for w in w_values:
        print(f"Testing w = {w}...")
        result = run_experiment(func_type, w, max_width=max_width)
        results.append(result)
        
        if result['N_c'] is not None:
            print(f"  N_c({w}) = {result['N_c']} (zero violation achieved)")
        else:
            print(f"  N_c({w}) = UNBOUNDED (no width <= {max_width} achieved zero violation)")
            print(f"  Best error at width {result['best_width']}: {result['best_error']:.6e}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("Summary Table:")
    print(f"{'w':>5} {'N_c':>8} {'Status':<20}")
    print("-" * 35)
    for r in results:
        status = f"{r['N_c']}" if r['N_c'] is not None else "UNBOUNDED"
        print(f"{r['w']:>5} {r['N_c']:>8} {status:<20}")
    
    # Analyze trend
    print("\n" + "=" * 70)
    print("Analysis:")
    
    N_c_values = [r['N_c'] for r in results if r['N_c'] is not None]
    unbounded_count = sum(1 for r in results if r['N_c'] is None)
    
    if unbounded_count > 0:
        print(f"WARNING: {unbounded_count} value(s) of w reached UNBOUNDED (N_c > {max_width})")
    
    if N_c_values:
        min_Nc = min(N_c_values)
        max_Nc = max(N_c_values)
        mean_Nc = np.mean(N_c_values)
        print(f"N_c values: {N_c_values}")
        print(f"Min N_c = {min_Nc}, Max N_c = {max_Nc}, Mean = {mean_Nc:.2f}")
        
        # Check if N_c is roughly constant
        if max_Nc - min_Nc <= 2:
            print("Conclusion: N_c is approximately CONSTANT in w -> RIGID family")
        else:
            print("Conclusion: N_c varies with w -> Need further analysis")
    
    # Save results
    import json
    with open('/tmp/rigidity_results.json', 'w') as f:
        json.dump({
            'func_type': func_type,
            'w_values': w_values,
            'max_width': max_width,
            'results': [{
                'w': r['w'],
                'N_c': r['N_c'],
                'best_error': r['best_error'],
                'best_width': r['best_width']
            } for r in results]
        }, f, indent=2)
    
    print(f"\nResults saved to /tmp/rigidity_results.json")
    
    # Try multiple function types if polynomial is UNBOUNDED
    if unbounded_count > 0 and func_type == "polynomial":
        print("\n\nTrying 'sin' function (algebraic structure with trig identities)...")
        for w in w_values[:5]:  # Just test a few
            result = run_experiment("sin", w, max_width=max_width)
            print(f"  w={w}: N_c={result['N_c']}, best_err={result['best_error']:.6e}")


if __name__ == "__main__":
    main()