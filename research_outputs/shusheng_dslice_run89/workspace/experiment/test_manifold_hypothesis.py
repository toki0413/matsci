#!/usr/bin/env python3
"""
Test the Manifold Intrinsicality Hypothesis (Hypothesis B).

Prediction: N_c(w) ∝ log w (logarithmic growth)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import json

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# Configuration
# ============================================================
n = 10  # ambient dimension
k = 2   # intrinsic dimension (manifold dimension)
epsilon = 1e-3  # zero-violation threshold
N_candidates = [10, 20, 50, 100, 200, 500, 1000]  # network widths to test
w_values = [10, 20, 50, 100, 200, 500, 1000]  # constraint counts
v = 100  # validation points

# ============================================================
# Manifold Definition: 2D torus embedded in R^10
# ============================================================
def generate_torus_samples(n_samples, theta1=None, theta2=None):
    """Generate samples uniformly distributed on a 2D torus embedded in R^10.
    
    The torus is parameterized by (theta1, theta2) in [0, 2π)^2.
    We embed it in R^10 by using the first 4 coordinates for the torus
    and setting the remaining 6 coordinates to 0 (or small perturbations).
    
    Torus embedding in R^4:
    x1 = (R + r*cos(theta2)) * cos(theta1)
    x2 = (R + r*cos(theta2)) * sin(theta1)
    x3 = r*sin(theta2) * cos(theta1)  (optional variation)
    x4 = r*sin(theta2) * sin(theta1)  (optional variation)
    
    For simplicity, use a standard torus in R^4 and zero out the rest.
    """
    if theta1 is None or theta2 is None:
        theta1 = np.random.uniform(0, 2 * np.pi, n_samples)
        theta2 = np.random.uniform(0, 2 * np.pi, n_samples)
    
    # Standard torus parameters
    R = 2.0  # major radius
    r = 1.0  # minor radius
    
    # Embedding in R^4 (first 4 dimensions)
    x1 = (R + r * np.cos(theta2)) * np.cos(theta1)
    x2 = (R + r * np.cos(theta2)) * np.sin(theta1)
    x3 = r * np.sin(theta2) * np.cos(theta1)
    x4 = r * np.sin(theta2) * np.sin(theta1)
    
    # For R^10, fill remaining dimensions with zeros (small noise for numerical stability)
    X = np.zeros((n_samples, n))
    X[:, :4] = np.column_stack([x1, x2, x3, x4])
    
    # Add small noise to make it more realistic (but still approximately on manifold)
    X += np.random.normal(0, 1e-5, X.shape)
    
    return theta1, theta2, X

def constraint_function(X):
    """f(x) = sin(2π * x_1) where x_1 is the first coordinate of x.
    
    This is a smooth function defined on the manifold.
    """
    return np.sin(2 * np.pi * X[:, 0])

# ============================================================
# Neural Network Architecture
# ============================================================
def create_regressor(input_dim, hidden_dim):
    """Create a single-hidden-layer ReLU network."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden_dim, activation='relu'),
        layers.Dense(1)
    ])
    return model

# ============================================================
# Training Function
# ============================================================
def train_and_evaluate(X_train, y_train, X_val, y_val, hidden_dim, epochs=2000, batch_size=32):
    """Train a network and return validation error."""
    model = create_regressor(X_train.shape[1], hidden_dim)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=[]
    )
    
    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True,
        min_delta=1e-6
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Get validation predictions
    y_pred_val = model.predict(X_val, verbose=0).flatten()
    max_val_error = np.max(np.abs(y_pred_val - y_val))
    
    # Also get training error
    y_pred_train = model.predict(X_train, verbose=0).flatten()
    max_train_error = np.max(np.abs(y_pred_train - y_train))
    
    return max_val_error, max_train_error, model

# ============================================================
# Main Experiment
# ============================================================
print(f"Starting Manifold Intrinsicality Hypothesis Test")
print(f"Manifold: 2D torus in R^{n}, k={k}, n={n}")
print(f"Zero-violation threshold: {epsilon}")
print(f"Testing w values: {w_values}")
print(f"Testing widths: {N_candidates}")
print()

results = {}

for w in w_values:
    print(f"\n=== w = {w} ===")
    
    # Generate training samples on the manifold
    theta1_train, theta2_train, X_train = generate_torus_samples(w)
    y_train = constraint_function(X_train)
    
    # Generate validation samples on the manifold (different random seed)
    theta1_val, theta2_val, X_val = generate_torus_samples(v)
    y_val = constraint_function(X_val)
    
    # Normalize inputs (optional, but helps training)
    X_train_mean = X_train.mean(axis=0, keepdims=True)
    X_train_std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - X_train_mean) / X_train_std
    X_val = (X_val - X_train_mean) / X_train_std
    
    # Find minimum N_c(w)
    N_c_w = None
    best_N = None
    best_error = float('inf')
    
    for N in N_candidates:
        start_time = time.time()
        val_error, train_error, model = train_and_evaluate(
            X_train, y_train, X_val, y_val, hidden_dim=N
        )
        elapsed = time.time() - start_time
        
        if val_error <= epsilon:
            if N_c_w is None:
                N_c_w = N
                best_N = N
                best_error = val_error
            print(f"  N={N}: val_error={val_error:.6e} (within threshold), time={elapsed:.2f}s")
        else:
            print(f"  N={N}: val_error={val_error:.6e} (above threshold), time={elapsed:.2f}s")
    
    if N_c_w is None:
        N_c_w = None  # Not achievable within tested range
        print(f"  No width achieved threshold. Best error: {best_error:.6e}")
    
    results[w] = {
        'N_c': N_c_w,
        'best_N': best_N,
        'best_error': best_error if best_error != float('inf') else None,
        'w': w
    }
    print(f"  N_c({w}) = {N_c_w}")

# ============================================================
# Analysis: Check if N_c(w) scales as log w
# ============================================================
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

print(f"\n{'w':>6} {'N_c(w)':>10} {'log(w)':>10}")
print("-"*30)

log_w_values = []
N_c_values = []

for w in w_values:
    result = results[w]
    log_w = np.log(w)
    N_c = result['N_c']
    log_w_values.append(log_w)
    if N_c is not None:
        N_c_values.append(N_c)
        print(f"{w:>6} {N_c:>10} {log_w:>10.4f}")
    else:
        print(f"{w:>6} {'None':>10} {log_w:>10.4f}")

# ============================================================
# Check for logarithmic scaling
# ============================================================
if len(N_c_values) >= 2:
    print("\n" + "="*60)
    print("SCALING ANALYSIS")
    print("="*60)
    
    # Fit linear model: N_c = a * log(w) + b
    coeffs = np.polyfit(log_w_values, N_c_values, 1)
    a, b = coeffs
    log_N_c_pred = a * np.array(log_w_values) + b
    
    print(f"\nLinear fit: N_c = {a:.2f} * log(w) + {b:.2f}")
    
    # Calculate R-squared
    ss_res = np.sum((np.array(N_c_values) - log_N_c_pred) ** 2)
    ss_tot = np.sum((np.array(N_c_values) - np.mean(N_c_values)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    print(f"R² = {r_squared:.4f}")
    
    # Check if slope is approximately proportional to k=2
    print(f"\nExpected slope (proportional to k={k}): ~{2 * 1.0:.1f} (rough estimate)")
    print(f"Actual slope: {a:.2f}")
    
    # Compare with constant prediction (Hypothesis A)
    # If N_c were constant, the slope would be close to 0
    if abs(a) < 0.5:
        print("Note: Small slope suggests possible constant behavior (Hypothesis A)")
    elif a > 1:
        print("Note: Significant positive slope suggests logarithmic growth (Hypothesis B)")
    
    # Compare with linear prediction (Hypothesis C)
    # If N_c were linear in w, log-log plot would have slope 1
    # But we're plotting N_c vs log(w), so linear in w would be exponential here
    print("\nInterpretation:")
    print("- Hypothesis A (constant): N_c should be flat across w values")
    print("- Hypothesis B (logarithmic): N_c should increase roughly linearly with log(w)")
    print("- Hypothesis C (linear): N_c should increase much faster (exponential in log(w))")

# ============================================================
# Save results
# ============================================================
output = {
    'w_values': w_values,
    'N_candidates': N_candidates,
    'results': results,
    'fit_coefficients': {'slope': float(a), 'intercept': float(b)},
    'R_squared': float(r_squared) if 'r_squared' in dir() else None
}

with open('/workspace/experiment/results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to /workspace/experiment/results.json")