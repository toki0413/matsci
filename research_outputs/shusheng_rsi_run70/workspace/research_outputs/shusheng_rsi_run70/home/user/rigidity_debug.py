#!/usr/bin/env python3
"""Debug version to understand the training behavior."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
np.random.seed(42)

# Configuration
CONFIG = {
    'target_type': 'poly',  # Try polynomial first - easier to fit
    'domain_min': 0.0,
    'domain_max': 1.0,
    'w_values': [10],
    'holdout_ratio': 0.2,
    'min_width': 4,
    'max_width': 128,
    'width_step': 2,
    'hidden_layers': 1,
    'activation': 'tanh',
    'n_seeds': 1,
    'max_epochs': 5000,
    'batch_size': 16,
    'lr': 1e-3,
    'tolerance': 1e-3,
}

def target_function(x, config):
    if config['target_type'] == 'sin':
        return torch.sin(2 * np.pi * x)
    elif config['target_type'] == 'poly':
        # Simple polynomial: x^2 - x + 0.25 = (x-0.5)^2
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
        'x_all': torch.tensor(x_train, dtype=torch.float32).view(-1, 1),
        'y_all': torch.tensor(y_true, dtype=torch.float32).view(-1, 1),
    }

def train_and_evaluate(seed, w, N, config):
    torch.manual_seed(seed)
    data = generate_constraint_points(w, config)
    x_train, y_train = data['x_train'], data['y_train']
    x_holdout, y_holdout = data['x_holdout'], data['y_holdout']
    x_all, y_all = data['x_all'], data['y_all']
    
    model = SimpleFFN(input_dim=1, hidden_dim=N, output_dim=1, activation=config['activation'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    train_losses = []
    for epoch in range(config['max_epochs']):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            with torch.no_grad():
                y_pred_hold = model(x_holdout)
                err = torch.abs(y_pred_hold - y_holdout).numpy().max()
                print(f"  seed={seed}, w={w}, N={N}, epoch={epoch}, train_loss={loss.item():.6f}, holdout_max_err={err:.6f}")
    
    model.eval()
    with torch.no_grad():
        y_pred = model(x_holdout)
        errors = torch.abs(y_pred - y_holdout).numpy()
        max_error = float(np.max(errors))
        train_max_err = float(torch.abs(model(x_train) - y_train).numpy().max())
    
    return {
        'max_error': max_error,
        'train_max_err': train_max_err,
        'zero_violation': max_error <= config['tolerance'],
        'train_loss_final': train_losses[-1] if train_losses else None,
    }

# Quick debug: test with w=10, N=64, seed=0
print("Debug test: w=10, N=64, seed=0")
result = train_and_evaluate(0, 10, 64, CONFIG)
print(f"Result: max_error={result['max_error']:.6f}, train_max_err={result['train_max_err']:.6f}, zero_violation={result['zero_violation']}")

# Now run full experiment
print("\n" + "="*60)
print("FULL EXPERIMENT")
print("="*60)

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
        for seed in range(CONFIG['n_seeds']):
            result = train_and_evaluate(seed, w, N, CONFIG)
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

# Save results
import json
with open('./rigidity_debug_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
w_vals = [int(w) for w in sorted(results.keys())]
Nc_vals = []
for w in w_vals:
    wc = str(w)
    if wc in results and results[wc].get('N_c') is not None:
        Nc_vals.append(results[wc]['N_c'])
    else:
        Nc_vals.append(None)

for w, Nc in zip(w_vals, Nc_vals):
    print(f"w={w}: N_c={Nc}")

if all(Nc is None for Nc in Nc_vals):
    print("\nAll N_c are None - no width achieved zero violation in the scanned range.")
    print("This suggests the solution space may be 'fat' in this range, OR")
    print("the target function is too difficult for the network architecture.")