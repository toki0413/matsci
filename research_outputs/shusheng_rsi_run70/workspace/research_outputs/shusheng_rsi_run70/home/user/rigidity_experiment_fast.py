#!/usr/bin/env python3
"""Fast version of the rigidity experiment for quick testing."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

torch.manual_seed(42)
np.random.seed(42)

# Fast configuration
CONFIG = {
    'target_type': 'sin',
    'domain_min': 0.0,
    'domain_max': 1.0,
    'w_values': [10, 20, 50],  # fewer w values for speed
    'holdout_ratio': 0.2,
    'min_width': 4,
    'max_width': 64,  # smaller range
    'width_step': 2,
    'hidden_layers': 1,
    'activation': 'tanh',
    'n_seeds': 2,  # fewer seeds
    'max_epochs': 1000,  # fewer epochs
    'batch_size': 16,
    'lr': 1e-3,
    'tolerance': 1e-3,
    'output_dir': './rigidity_results_fast',
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

def target_function(x, config):
    if config['target_type'] == 'sin':
        return torch.sin(2 * np.pi * x)
    elif config['target_type'] == 'poly':
        return x**2 - x + 0.25

class SimpleFFN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, activation='tanh'):
        super().__init__()
        act = nn.Tanh() if activation == 'tanh' else nn.ReLU()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.layers(x)

def generate_constraint_points(w, config):
    x_train = np.random.uniform(config['domain_min'], config['domain_max'], w)
    y_train = target_function(torch.tensor(x_train, dtype=torch.float32), config).numpy()
    n_holdout = int(w * config['holdout_ratio'])
    n_train = w - n_holdout
    indices = np.arange(w)
    np.random.shuffle(indices)
    holdout_idx = indices[:n_holdout]
    train_idx = indices[n_holdout:]
    return {
        'x_train': torch.tensor(x_train[train_idx], dtype=torch.float32).view(-1, 1),
        'y_train': torch.tensor(y_train[train_idx], dtype=torch.float32).view(-1, 1),
        'x_holdout': torch.tensor(x_train[holdout_idx], dtype=torch.float32).view(-1, 1),
        'y_holdout': torch.tensor(y_train[holdout_idx], dtype=torch.float32).view(-1, 1),
    }

def train_and_evaluate(seed, w, N, config):
    torch.manual_seed(seed)
    data = generate_constraint_points(w, config)
    x_train, y_train = data['x_train'], data['y_train']
    x_holdout, y_holdout = data['x_holdout'], data['y_holdout']
    
    model = SimpleFFN(input_dim=1, hidden_dim=N, output_dim=1, activation=config['activation'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    for epoch in range(config['max_epochs']):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if loss.item() < 1e-10:
            break
    
    model.eval()
    with torch.no_grad():
        y_pred = model(x_holdout)
        errors = torch.abs(y_pred - y_holdout).numpy()
        max_error = float(np.max(errors))
    
    return {
        'max_error': max_error,
        'zero_violation': max_error <= config['tolerance'],
    }

def run_experiment():
    config = CONFIG
    results = {}
    
    print(f"Fast experiment: target={config['target_type']}, w={config['w_values']}, width {config['min_width']}-{config['max_width']}")
    
    for w in config['w_values']:
        results[str(w)] = {}
        print(f"\nw = {w}:")
        
        N_values = []
        N_val = config['min_width']
        while N_val <= config['max_width']:
            N_values.append(N_val)
            N_val *= config['width_step']
        
        Nc_w = None
        for N in N_values:
            results[str(w)][str(N)] = {}
            best_zero = False
            for seed in range(config['n_seeds']):
                result = train_and_evaluate(seed, w, N, config)
                results[str(w)][str(N)][str(seed)] = result
                if result['zero_violation']:
                    best_zero = True
                    break
            results[str(w)][str(N)]['best_zero'] = best_zero
            if result['zero_violation'] and Nc_w is None:
                Nc_w = N
            print(f"  N={N}: best_zero={best_zero}, N_c so far={Nc_w}")
        results[str(w)]['N_c'] = Nc_w
        print(f"  N_c({w}) = {Nc_w}")
    
    # Save
    with open(os.path.join(config['output_dir'], 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Analyze
    w_vals = [int(w) for w in results.keys() if w != 'N_c' and isinstance(w, (str, int))]
    Nc_vals = []
    for w in sorted(w_vals):
        wc = str(w)
        if wc in results and results[wc].get('N_c') is not None:
            Nc_vals.append(results[wc]['N_c'])
    
    print("\nN_c(w) results:")
    for w, Nc in zip(sorted(w_vals), Nc_vals):
        print(f"  w={w}: N_c={Nc}")
    
    if len(w_vals) >= 2 and all(Nc is not None for Nc in Nc_vals):
        w_finite = np.array(sorted(w_vals))
        Nc_finite = np.array(Nc_vals, dtype=float)
        log_w = np.log(w_finite)
        log_Nc = np.log(Nc_finite)
        coeffs = np.polyfit(log_w, log_Nc, 1)
        exponent = coeffs[0]
        prefactor = np.exp(coeffs[1])
        print(f"\nFitted: N_c(w) ≈ {prefactor:.2f} * w^{exponent:.2f}")
        print("Predictions: constant=0.0, sqrt=0.5, linear=1.0")
        print(f"Closest: {'constant' if abs(exponent) < 0.25 else 'sqrt' if abs(exponent - 0.5) < 0.25 else 'linear'}")
    
    print(f"\nResults saved to {config['output_dir']}")

if __name__ == '__main__':
    run_experiment()