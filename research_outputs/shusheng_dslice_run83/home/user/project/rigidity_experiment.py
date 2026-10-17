#!/usr/bin/env python3
"""Solution Space Rigidity Experiment for Small Feedforward Networks."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import json
import os

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

CONFIG = {
    'input_dim': 2,
    'output_dim': 1,
    'zero_violation_threshold': 1e-3,
    'max_width': 80,
    'min_width': 4,
    'constraint_counts': [5, 10, 15, 20, 30, 40, 50, 60, 70, 80],
    'epochs': 200,
    'batch_size': 16,
    'learning_rate': 1e-3,
    'holdout_ratio': 0.2,
    'n_trials': 2,
}

def generate_true_function(x):
    return np.sin(np.pi * x[:, 0:1]) + np.cos(np.pi * x[:, 1:2]) + 0.5 * x[:, 0:1] * x[:, 1:2]

def generate_constraint_samples(n_samples, input_dim=2, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, size=(n_samples, input_dim))
    y = generate_true_function(X)
    return X, y

class SmallFeedforward(nn.Module):
    def __init__(self, input_dim, hidden_N, output_dim, activation=nn.ReLU):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_N)
        self.act = activation()
        self.output = nn.Linear(hidden_N, output_dim)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
    
    def forward(self, x):
        x = self.act(self.hidden(x))
        return self.output(x)

def train_network(model, X_train, y_train, X_val, y_val, epochs=200, lr=1e-3, batch_size=16):
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        train_pred = model(torch.FloatTensor(X_train))
        val_pred = model(torch.FloatTensor(X_val))
        train_max_err = float(torch.max(torch.abs(train_pred - torch.FloatTensor(y_train))).item())
        val_max_err = float(torch.max(torch.abs(val_pred - torch.FloatTensor(y_val))).item())
        train_mse = float(criterion(train_pred, torch.FloatTensor(y_train)).item())
        val_mse = float(criterion(val_pred, torch.FloatTensor(y_val)).item())
    
    return train_mse, val_mse, train_max_err, val_max_err

def find_minimal_width(w, max_width=80, min_width=4, holdout_ratio=0.2, epochs=200, lr=1e-3, seed=42):
    rng = np.random.RandomState(seed)
    X_all, y_all = generate_constraint_samples(w, 2, seed)
    n_holdout = int(w * holdout_ratio)
    n_train = w - n_holdout
    indices = np.arange(w)
    rng.shuffle(indices)
    holdout_idx = indices[:n_holdout]
    train_idx = indices[n_holdout:]
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_hold, y_hold = X_all[holdout_idx], y_all[holdout_idx]
    
    for N in range(min_width, max_width + 1):
        model = SmallFeedforward(2, N, 1)
        train_mse, val_mse, train_max, val_max = train_network(
            model, X_train, y_train, X_hold, y_hold,
            epochs=epochs, lr=lr, batch_size=16
        )
        if val_max <= 1e-3:
            return N, train_max, val_max, True
    return -1, 0, 0, False

def run_experiment():
    print("SOLUTION SPACE RIGIDITY EXPERIMENT")
    print(f"Config: {CONFIG}")
    
    results = []
    for w in CONFIG['constraint_counts']:
        print(f"\nw = {w}")
        widths_achieved = []
        train_errors = []
        val_errors = []
        successes = []
        
        for trial in range(CONFIG['n_trials']):
            seed = 42 + w * 1000 + trial * 100
            N_c, train_err, val_err, success = find_minimal_width(
                w=w, max_width=CONFIG['max_width'], min_width=CONFIG['min_width'],
                holdout_ratio=CONFIG['holdout_ratio'], epochs=CONFIG['epochs'],
                lr=CONFIG['learning_rate'], seed=seed
            )
            widths_achieved.append(N_c)
            train_errors.append(train_err)
            val_errors.append(val_err)
            successes.append(success)
            print(f"  Trial {trial+1}: N_c={N_c}, train_err={train_err:.6e}, val_err={val_err:.6e}, success={success}")
        
        successful_N = [N for N, s in zip(widths_achieved, successes) if s]
        N_c_median = int(np.median(successful_N)) if successful_N else -1
        avg_train_err = np.mean([e for e, s in zip(train_errors, successes) if s]) if any(successes) else float('inf')
        avg_val_err = np.mean([e for e, s in zip(val_errors, successes) if s]) if any(successes) else float('inf')
        
        results.append({'w': w, 'N_c': N_c_median, 'successes': sum(successes), 'n_trials': CONFIG['n_trials'],
                       'train_error': avg_train_err, 'val_error': avg_val_err, 'widths_achieved': widths_achieved})
        print(f"  Summary: N_c(w)={N_c_median}, success_rate={sum(successes)}/{CONFIG['n_trials']}")
    
    print(f"\nRESULTS TABLE:")
    print(f"{'w':>6} {'N_c(w)':>8} {'Success Rate':>12} {'Train Err':>12} {'Val Err':>12}")
    for r in results:
        print(f"{r['w']:>6} {r['N_c']:>8} {r['successes']}/{r['n_trials']:>10} {r['train_error']:>12.2e} {r['val_error']:>12.2e}")
    
    w_vals = [r['w'] for r in results if r['N_c'] > 0]
    Nc_vals = [r['N_c'] for r in results if r['N_c'] > 0]
    
    if len(w_vals) >= 2:
        Nc_range = max(Nc_vals) - min(Nc_vals)
        print(f"\nN_c range: {Nc_range}")
        if len(w_vals) >= 3:
            coeffs = np.polyfit(w_vals, Nc_vals, 1)
            print(f"Linear fit: N_c = {coeffs[0]:.4f}*w + {coeffs[1]:.4f}")
            if abs(coeffs[0]) < 0.1:
                print("-> Suggests RIGID regime")
            elif abs(coeffs[0]) > 0.3:
                print("-> Suggests FLOPPY regime (linear growth)")
            else:
                print("-> Intermediate")
    
    os.makedirs('/home/user/project/results', exist_ok=True)
    with open('/home/user/project/results/rigidity_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /home/user/project/results/rigidity_results.json")
    
    return results

if __name__ == '__main__':
    run_experiment()