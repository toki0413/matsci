#!/usr/bin/env python3
"""Simpler test with constant target to verify setup."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
np.random.seed(42)

# Configuration
CONFIG = {
    'target_type': 'constant',
    'target_value': 0.5,
    'domain_min': 0.0,
    'domain_max': 1.0,
    'w_values': [10],
    'holdout_ratio': 0.2,
    'min_width': 4,
    'max_width': 16,
    'width_step': 2,
    'hidden_layers': 1,
    'activation': 'tanh',
    'n_seeds': 1,
    'max_epochs': 1000,
    'batch_size': 16,
    'lr': 1e-2,
    'tolerance': 1e-3,
}

def target_function(x, config):
    if config['target_type'] == 'constant':
        return torch.full_like(x, config['target_value'])
    elif config['target_type'] == 'poly':
        return x**2 - x + 0.25
    elif config['target_type'] == 'sin':
        return torch.sin(2 * np.pi * x)

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
    
    for epoch in range(config['max_epochs']):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
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
    }

# Test constant target
print("Testing constant target f(x) = 0.5")
result = train_and_evaluate(0, 10, 16, CONFIG)
print(f"w=10, N=16: train_max_err={result['train_max_err']:.6f}, holdout_max_err={result['max_error']:.6f}, zero_violation={result['zero_violation']}")

# Test polynomial
print("\nTesting polynomial target f(x) = x^2 - x + 0.25")
CONFIG_poly = {
    'target_type': 'poly',
    'domain_min': 0.0,
    'domain_max': 1.0,
    'w_values': [10, 20, 50],
    'holdout_ratio': 0.2,
    'min_width': 4,
    'max_width': 128,
    'width_step': 2,
    'hidden_layers': 1,
    'activation': 'tanh',
    'n_seeds': 1,
    'max_epochs': 5000,
    'batch_size': 16,
    'lr': 1e-2,
    'tolerance': 1e-3,
}

result = train_and_evaluate(0, 10, 64, CONFIG_poly)
print(f"w=10, N=64: train_max_err={result['train_max_err']:.6f}, holdout_max_err={result['max_error']:.6f}, zero_violation={result['zero_violation']}")

if result['zero_violation']:
    print("\nPolynomial target works! Running full experiment...")
    results = {}
    for w in CONFIG_poly['w_values']:
        results[str(w)] = {}
        print(f"\nw = {w}:")
        N_values = []
        N_val = CONFIG_poly['min_width']
        while N_val <= CONFIG_poly['max_width']:
            N_values.append(N_val)
            N_val *= CONFIG_poly['width_step']
        Nc_w = None
        for N in N_values:
            results[str(w)][str(N)] = {}
            best_zero = False
            for seed in range(CONFIG_poly['n_seeds']):
                result = train_and_evaluate(seed, w, N, CONFIG_poly)
                results[str(w)][str(N)][str(seed)] = result
                if result['zero_violation']:
                    best_zero = True
                    break
            results[str(w)][str(N)]['best_zero'] = best_zero
            if result['zero_violation'] and Nc_w is None:
                Nc_w = N
            print(f"  N={N}: best_zero={best_zero}, train_err={result['train_max_err']:.6f}, holdout_err={result['max_error']:.6f}, N_c so far={Nc_w}")
        results[str(w)]['N_c'] = Nc_w
        print(f"  N_c({w}) = {Nc_w}")
    
    import json
    with open('./rigidity_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nSummary:")
    w_vals = [int(w) for w in sorted(results.keys())]
    for w in w_vals:
        wc = str(w)
        Nc = results[wc].get('N_c')
        print(f"w={w}: N_c={Nc}")
else:
    print("\nPolynomial target not achieving zero violation. Need to debug.")
    print(f"Train max err: {result['train_max_err']:.6f}, Holdout max err: {result['max_error']:.6f}")