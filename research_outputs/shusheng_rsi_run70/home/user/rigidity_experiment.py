#!/usr/bin/env python3
"""
Bootstrap Solution Space Rigidity Experiment for Small Feedforward Networks.

Tests the scaling of N_c(w) = minimum hidden layer width achieving zero violation
(max absolute error <= 1e-3) on held-out constraint points as the number of
constraint points w increases.

Three competing predictions:
  - Algebraic (structure): N_c(w) = O(1)  -> flat on log-log
  - Geometric (manifold): N_c(w) = O(w)   -> slope 1 on log-log
  - Statistical (measure): N_c(w) = O(sqrt(w)) -> slope 0.5 on log-log
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    # Target function
    'target_type': 'sin',  # 'sin' or 'poly' for polynomial
    'domain_min': 0.0,
    'domain_max': 1.0,
    
    # Constraint points
    'w_values': [10, 20, 50, 100, 200],  # number of constraint samples
    'holdout_ratio': 0.2,  # fraction of w as held-out
    
    # Network architecture
    'min_width': 4,
    'max_width': 256,
    'width_step': 2,  # multiply by 2 each step
    'hidden_layers': 1,  # single hidden layer (small feedforward)
    'activation': 'tanh',
    
    # Training
    'n_seeds': 3,  # multiple seeds per (w, N) to address initialization sensitivity
    'max_epochs': 5000,
    'batch_size': 32,
    'lr': 1e-3,
    'tolerance': 1e-3,  # zero violation threshold
    
    # Output
    'output_dir': './rigidity_results',
}

# Create output directory
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ============================================================
# Target Function Definition
# ============================================================
def target_function(x, config):
    """Evaluate the target function at points x."""
    if config['target_type'] == 'sin':
        return torch.sin(2 * np.pi * x)
    elif config['target_type'] == 'poly':
        # f(x) = x^2 - x + 0.25 (a simple quadratic)
        return x**2 - x + 0.25
    else:
        raise ValueError(f"Unknown target type: {config['target_type']}")

def target_derivative(x, config):
    """Optional: derivative of target function (for potential loss terms)."""
    if config['target_type'] == 'sin':
        return 2 * np.pi * torch.cos(2 * np.pi * x)
    elif config['target_type'] == 'poly':
        return 2 * x - 1
    else:
        raise ValueError(f"Unknown target type: {config['target_type']}")

# ============================================================
# Neural Network Model
# ============================================================
class SimpleFFN(nn.Module):
    """Simple feedforward network with one hidden layer."""
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            act = nn.Tanh()
        elif activation == 'relu':
            act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

# ============================================================
# Data Generation
# ============================================================
def generate_constraint_points(w, config):
    """Generate w constraint points from the target function."""
    x_train = np.random.uniform(config['domain_min'], config['domain_max'], w)
    y_train = target_function(torch.tensor(x_train, dtype=torch.float32), config).numpy()
    
    # Split into training and held-out
    n_holdout = int(w * config['holdout_ratio'])
    n_train = w - n_holdout
    
    indices = np.arange(w)
    np.random.shuffle(indices)
    
    holdout_idx = indices[:n_holdout]
    train_idx = indices[n_holdout:]
    
    x_train_all = x_train[train_idx]
    y_train_all = y_train[train_idx]
    x_holdout = x_train[holdout_idx]
    y_holdout = y_train[holdout_idx]
    
    return {
        'x_train': torch.tensor(x_train_all, dtype=torch.float32).view(-1, 1),
        'y_train': torch.tensor(y_train_all, dtype=torch.float32).view(-1, 1),
        'x_holdout': torch.tensor(x_holdout, dtype=torch.float32).view(-1, 1),
        'y_holdout': torch.tensor(y_holdout, dtype=torch.float32).view(-1, 1),
    }

# ============================================================
# Training and Evaluation
# ============================================================
def train_and_evaluate(seed, w, N, config):
    """Train a network of width N on w constraints and evaluate on held-out set."""
    torch.manual_seed(seed)
    
    # Generate data
    data = generate_constraint_points(w, config)
    x_train, y_train = data['x_train'], data['y_train']
    x_holdout, y_holdout = data['x_holdout'], data['y_holdout']
    
    # Create model
    model = SimpleFFN(input_dim=1, hidden_dim=N, output_dim=1, activation=config['activation'])
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # DataLoader
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Training loop
    model.train()
    for epoch in range(config['max_epochs']):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Early stopping if loss is very small
        if epoch_loss / len(loader) < 1e-10:
            break
    
    # Evaluate on held-out set
    model.eval()
    with torch.no_grad():
        y_pred_holdout = model(x_holdout)
        errors = torch.abs(y_pred_holdout - y_holdout).numpy()
        max_error = float(np.max(errors))
        train_loss = float(epoch_loss / len(loader))
    
    return {
        'max_error': max_error,
        'train_loss': train_loss,
        'zero_violation': max_error <= config['tolerance'],
    }

# ============================================================
# Main Experiment
# ============================================================
def run_experiment():
    """Run the full experiment sweeping over w and N."""
    config = CONFIG
    results = {}
    
    print(f"Starting experiment with target: {config['target_type']}")
    print(f"Constraint counts w: {config['w_values']}")
    print(f"Width range: {config['min_width']} to {config['max_width']}")
    print(f"Seeds per config: {config['n_seeds']}")
    print(f"Tolerance: {config['tolerance']}")
    print()
    
    # Sweep over w values
    for w in config['w_values']:
        results[w] = {}
        print(f"\nProcessing w = {w}...")
        
        # Sweep over width N
        N_values = []
        N_val = config['min_width']
        while N_val <= config['max_width']:
            N_values.append(N_val)
            N_val *= config['width_step']
        
        for N in N_values:
            results[w][N] = {}
            best_zero_violation = False
            best_seed = None
            
            # Test multiple seeds to address initialization sensitivity
            for seed in range(config['n_seeds']):
                result = train_and_evaluate(seed, w, N, config)
                results[w][N][seed] = result
                
                if result['zero_violation']:
                    best_zero_violation = True
                    best_seed = seed
                    break  # Found a seed that works
            
            results[w][N]['best_zero_violation'] = best_zero_violation
            results[w][N]['best_seed'] = best_seed
            
            # Print progress
            if best_zero_violation:
                print(f"  w={w}, N={N}: ZERO VIOLATION achieved (seed={best_seed}, max_error={results[w][N][best_seed]['max_error']:.2e})")
            else:
                # Report best seed's max error
                best_seed_temp = results[w][N].get('best_seed', 0)
                if best_seed_temp is None:
                    best_seed_temp = 0
                max_err = results[w][N][best_seed_temp]['max_error']
                print(f"  w={w}, N={N}: No zero violation (best seed {best_seed_temp}, max_error={max_err:.2e})")
        
        # Determine N_c(w): minimum N achieving zero violation across any seed
        Nc_w = None
        for N in N_values:
            if results[w][N]['best_zero_violation']:
                Nc_w = N
                break
        results[w]['N_c'] = Nc_w
        print(f"  N_c({w}) = {Nc_w}")
    
    # Save results
    output_path = os.path.join(config['output_dir'], 'results.json')
    with open(output_path, 'w') as f:
        # Convert numpy types to Python native types for JSON serialization
        results_serializable = {}
        for w in results:
            results_serializable[str(w)] = {}
            for N in results[w]:
                results_serializable[str(w)][str(N)] = {}
                for seed in results[w][N]:
                    if isinstance(seed, dict) or seed in ['best_zero_violation', 'best_seed']:
                        results_serializable[str(w)][str(N)][seed] = results[w][N][seed]
                    else:
                        results_serializable[str(w)][str(N)][seed] = {
                            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                            for k, v in results[w][N][seed].items()
                        }
        results_serializable['N_c'] = {str(k): v for k, v in results.items() if k not in [w for w in results if isinstance(w, int)]}
        # Better serialization
        final_results = {}
        for key in results:
            if key == 'N_c':
                final_results[key] = results[key]
            elif isinstance(key, int):
                final_results[str(key)] = {}
                for subkey in results[key]:
                    if isinstance(subkey, dict) or subkey in ['best_zero_violation', 'best_seed']:
                        final_results[str(key)][subkey] = results[key][subkey]
                    else:
                        final_results[str(key)][str(subkey)] = {
                            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                            for k, v in results[key][subkey].items()
                        }
        
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")
    return results

# ============================================================
# Analysis and Visualization
# ============================================================
def analyze_and_plot(results, config):
    """Analyze results and generate plots."""
    output_dir = config['output_dir']
    
    # Extract N_c values
    w_values = sorted([int(w) for w in results.keys() if w != 'N_c' and isinstance(w, (str, int))])
    Nc_values = []
    
    for w in w_values:
        wc_str = str(w)
        if wc_str in results and 'N_c' in results[wc_str]:
            Nc = results[wc_str]['N_c']
            if Nc is not None and isinstance(Nc, (int, float)) and np.isfinite(Nc):
                Nc_values.append(Nc)
            else:
                Nc_values.append(None)
        else:
            Nc_values.append(None)
    
    print("\nN_c(w) values:")
    for w, Nc in zip(w_values, Nc_values):
        print(f"  w={w}: N_c={Nc}")
    
    # Fit scaling laws
    w_finite = [w for w, Nc in zip(w_values, Nc_values) if Nc is not None and np.isfinite(Nc)]
    Nc_finite = [Nc for Nc in zip(w_values, Nc_values) if Nc is not None and np.isfinite(Nc)]
    
    if len(w_finite) >= 2:
        # Log-log fit
        log_w = np.log(w_finite)
        log_Nc = np.log(Nc_finite)
        
        # Fit log(Nc) = a + b * log(w) => Nc = exp(a) * w^b
        coeffs = np.polyfit(log_w, log_Nc, 1)
        exponent = coeffs[0]
        prefactor = np.exp(coeffs[1])
        
        print(f"\nLog-log fit: N_c(w) ≈ {prefactor:.2f} * w^{exponent:.2f}")
        
        # Compare with predictions
        predictions = {
            'constant (structure)': 0.0,
            'square-root (measure)': 0.5,
            'linear (geometry)': 1.0,
        }
        
        print("\nComparison with predictions:")
        for pred_name, pred_exp in predictions.items():
            diff = abs(exponent - pred_exp)
            print(f"  {pred_name}: predicted exponent = {pred_exp}, fitted = {exponent:.2f}, diff = {diff:.2f}")
    else:
        print("\nNot enough finite data points for fitting.")
    
    # Plot 1: N_c vs w (log-log scale)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Log-log plot of N_c(w)
    w_plot = [w for w, Nc in zip(w_values, Nc_values) if Nc is not None and np.isfinite(Nc)]
    Nc_plot = [Nc for Nc in zip(w_values, Nc_values) if Nc is not None and np.isfinite(Nc)]
    
    if w_plot:
        axes[0].loglog(w_plot, Nc_plot, 'bo-', label='N_c(w)', markersize=8)
        
        # Plot reference lines
        w_ref = np.array(w_plot)
        axes[0].loglog(w_ref, np.full_like(w_ref, Nc_plot[0]), 'r--', label='Constant (O(1))')
        axes[0].loglog(w_ref, prefactor * w_ref**exponent, 'g--', label=f'Fitted: w^{exponent:.2f}')
        axes[0].loglog(w_ref, prefactor * np.sqrt(w_ref), 'm--', label='Square-root (O(√w))')
        axes[0].loglog(w_ref, prefactor * w_ref, 'c--', label='Linear (O(w))')
        
        axes[0].set_xlabel('w (constraint count)')
        axes[0].set_ylabel('N_c(w) (minimum width)')
        axes[0].set_title('Solution Space Rigidity: N_c(w) Scaling')
        axes[0].legend()
        axes[0].grid(True, which='both', alpha=0.3)
    
    # Plot 2: Example error vs width for fixed w
    if w_values:
        w_sample = w_values[len(w_values) // 2]  # middle w value
        N_values = []
        max_errors = []
        
        for N in range(config['min_width'], config['max_width'] + 1, max(1, config['min_width'] // 4)):
            # Get best result for this (w, N)
            if str(w_sample) in results and str(N) in results[str(w_sample)]:
                best_seed = results[str(w_sample)][str(N)].get('best_seed')
                if best_seed is not None and best_seed in results[str(w_sample)][str(N)]:
                    err = results[str(w_sample)][str(N)][best_seed]['max_error']
                    N_values.append(N)
                    max_errors.append(err)
        
        if N_values:
            axes[1].semilogy(N_values, max_errors, 'ro-', label=f'w={w_sample}', markersize=6)
            axes[1].axhline(y=config['tolerance'], color='k', linestyle='--', label=f'Tolerance ({config['tolerance']})')
            axes[1].set_xlabel('Network width N')
            axes[1].set_ylabel('Max absolute error on held-out set')
            axes[1].set_title(f'Error vs Width for w={w_sample}')
            axes[1].legend()
            axes[1].grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'rigidity_analysis.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nAnalysis plot saved to {plot_path}")
    
    return {
        'w_values': w_values,
        'Nc_values': Nc_values,
        'exponent': exponent if 'exponent' in dir() else None,
        'prefactor': prefactor if 'prefactor' in dir() else None,
    }

# ============================================================
# Main Execution
# ============================================================
if __name__ == '__main__':
    results = run_experiment()
    analysis = analyze_and_plot(results, CONFIG)
    
    # Summary summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Target function: {CONFIG['target_type']}")
    print(f"Zero violation tolerance: {CONFIG['tolerance']}")
    print(f"Constraint counts tested: {CONFIG['w_values']}")
    print(f"Width range: {CONFIG['min_width']} - {CONFIG['max_width']}")
    print(f"\nN_c(w) results:")
    for w, Nc in zip(analysis['w_values'], analysis['Nc_values']):
        print(f"  w={w}: N_c={Nc}")
    
    if analysis['exponent'] is not None:
        print(f"\nFitted scaling: N_c(w) ≈ {analysis['prefactor']:.2f} × w^{analysis['exponent']:.2f}")
        print(f"Closest prediction: ", end="")
        predictions = {
            'constant (structure)': 0.0,
            'square-root (measure)': 0.5,
            'linear (geometry)': 1.0,
        }
        closest = min(predictions, key=lambda k: abs(analysis['exponent'] - predictions[k]))
        print(closest)
    print("="*60)