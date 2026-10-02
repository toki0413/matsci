#!/usr/bin/env python3
"""Improved experiment with better optimization."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json

torch.manual_seed(42)
np.random.seed(42)

# Configuration - better optimization settings
CONFIG = {
    'target_type': 'poly',  # f(x) = x^2 - x + 0.25
    'target_func': lambda x: x**2 - x + 0.25,
    'domain_min': 0.0,
    'domain_max': 1.0,
    'w_values': [10, 20, 50, 100],
    'holdout_ratio': 0.2,
    'min_width': 4,
    'max_width': 256,
    'width_step': 2,
    'hidden_layers': 1,
    'activation': 'relu',  # ReLU often works better for polynomials
    'n_seeds': 2,  # Two seeds for better statistics
    'max_epochs': 10000,
    'batch_size': 32,
    'lr': 1e-1,  # Higher learning rate
    'tolerance': 1e-3,
    'use_scheduler': True,
}

def target_function(x, config):
    return config['target_func'](x)

class SimpleFFN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, activation='relu'):
        super().__init__()
        act = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.layers(x)

def generate_constraint_points(w, config):
    x_train = np.random.uniform(config['domain_min'], config['domain_max'], w)
    y_true = target_function(torch.tensor(x_train, dtype=torch.float32), config).numpy()
    n_holdout = int(w * config['holdout_ratio'])
    n_train = w - n_holdout
    indices = np.arange(w)
    np.random.shuffle(indices)
    holdout_idx = indices[:n_holdout]
    train_idx = indices[n_holdout:]
    return {
        'x_train': torch.tensor(x_train[train_idx], dtype=torch.float32).view(-1, 1),
        'y_train': torch.tensor(y_true[train_idx], dtype=torch.float32).view(-1, 1),
        'x_holdout': torch.tensor(x_train[holdout_idx], dtype=torch.float32).view(-1, 1),
        'y_holdout': torch.tensor(y_true[holdout_idx], dtype=torch.float32).view(-1, 1),
    }

def train_and_evaluate(seed, w, N, config):
    torch.manual_seed(seed)
    data = generate_constraint_points(w, config)
    x_train, y_train = data['x_train'], data['y_train']
    x_holdout, y_holdout = data['x_holdout'], data['y_holdout']
    
    model = SimpleFFN(input_dim=1, hidden_dim=N, output_dim=1, activation=config['activation'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000, min_lr=1e-6) if config['use_scheduler'] else None
    
    for epoch in range(config['max_epochs']):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if scheduler:
            scheduler.step(loss)
    
    model.eval()
    with torch.no_grad():
        y_pred_train = model(x_train)
        y_pred_hold = model(x_holdout)
        train_errors = torch.abs(y_pred_train - y_train).numpy()
        holdout_errors = torch.abs(y_pred_hold - y_holdout).numpy()
        train_max_err = float(np.max(train_errors))
        holdout_max_err = float(np.max(holdout_errors))
    
    return {
        'max_error': holdout_max_err,
        'train_max_err': train_max_err,
        'zero_violation': holdout_max_err <= config['tolerance'],
        'train_errors': train_errors.tolist(),
        'holdout_errors': holdout_errors.tolist(),
    }

# Quick test with different N values
print("Testing polynomial target with ReLU, lr=0.1")
for N in [4, 8, 16, 32, 64, 128]:
    result = train_and_evaluate(0, 10, N, CONFIG)
    print(f"w=10, N={N}: train_max_err={result['train_max_err']:.6f}, holdout_max_err={result['max_error']:.6f}, zero_violation={result['zero_violation']}")

# Check if any width achieves zero violation
any_zero = any(train_and_evaluate(0, 10, N, CONFIG)['zero_violation'] for N in [4, 8, 16, 32, 64, 128, 256])
print(f"\nAny width achieved zero violation for w=10? {any_zero}")

if not any_zero:
    print("\nNeed to adjust settings. Trying with even higher learning rate...")
    CONFIG['lr'] = 0.5
    for N in [4, 8, 16, 32, 64]:
        result = train_and_evaluate(0, 10, N, CONFIG)
        print(f"w=10, N={N}: train_max_err={result['train_max_err']:.6f}, holdout_max_err={result['max_error']:.6f}, zero_violation={result['zero_violation']}")

# Full experiment if we have at least one zero violation
result_first = train_and_evaluate(0, 10, 64, CONFIG)
if result_first['zero_violation']:
    print("\nPolynomial target now works! Running full experiment...")
    results = {}
    for w in CONFIG['w_values']:
        results[str(w)] = {}
        print(f"\nw = {w}:")
        N_values = []
        N_val = CONFIG['min_width']
        while N_val <= CONFIG['max_width']:
            N_values.append(N_val)
            N_val *= CONFIG['width_step']
        Nc_w = None
        for N in N_values:
            results[str(w)][str(N)] = {}
            best_zero = False
            best_result = None
            for seed in range(CONFIG['n_seeds']):
                result = train_and_evaluate(seed, w, N, CONFIG)
                results[str(w)][str(N)][str(seed)] = result
                if result['zero_violation']:
                    best_zero = True
                    best_result = result
                    break
            results[str(w)][str(N)]['best_zero'] = best_zero
            if best_result and Nc_w is None:
                Nc_w = N
            print(f"  N={N}: best_zero={best_zero}, train_err={best_result['train_max_err'] if best_result else 'N/A':.6f}, holdout_err={best_result['max_error'] if best_result else 'N/A':.6f}, N_c so far={Nc_w}")
        results[str(w)]['N_c'] = Nc_w
        print(f"  N_c({w}) = {Nc_w}")
    
    with open('./rigidity_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nSummary:")
    w_vals = sorted([int(w) for w in results.keys()])
    for w in w_vals:
        wc = str(w)
        Nc = results[wc].get('N_c')
        print(f"w={w}: N_c={Nc}")
    
    # Try to fit scaling
    import numpy as np
    w_vals_fit = [int(w) for w in results.keys() if results[str(w)]['N_c'] is not None]
    Nc_vals = [results[str(w)]['N_c'] for w in w_vals_fit]
    if len(w_vals_fit) >= 2:
        log_w = np.log(w_vals_fit)
        log_Nc = np.log(Nc_vals)
        coeffs = np.polyfit(log_w, log_Nc, 1)
        print(f"\nScaling fit: N_c ~ w^({coeffs[0]:.3f})")
        print(f"Expected: constant (0), linear (1), sqrt (0.5)")
else:
    print("\nStill not achieving zero violation. Need more debugging.")
    print(f"Best result for w=10, N=256: {train_and_evaluate(0, 10, 256, CONFIG)}")