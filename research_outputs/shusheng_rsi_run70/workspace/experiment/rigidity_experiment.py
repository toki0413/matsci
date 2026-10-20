#!/usr/bin/env python3
"""Bootstrap Solution Space Rigidity Experiment for Small Feedforward Networks"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

np.random.seed(42)
torch.manual_seed(42)

class SingleLayerReLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.hidden(x))
        return self.output(x)

def generate_constraints(target_func, n_points, domain=(0.0, 1.0)):
    x = np.random.uniform(domain[0], domain[1], n_points)
    y = target_func(x)
    return x, y

def generate_held_out(target_func, n_points, domain=(0.0, 1.0)):
    x = np.random.uniform(domain[0], domain[1], n_points)
    y = target_func(x)
    return x, y

def train_network(model, x_train, y_train, x_val, y_val, max_epochs=1000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        y_train_pred = model(x_train)
        y_val_pred = model(x_val)
        train_max_err = float(torch.max(torch.abs(y_train_pred - y_train)).item())
        val_max_err = float(torch.max(torch.abs(y_val_pred - y_val)).item())
    return train_max_err, val_max_err

def find_min_width(w, n_range, target_func, held_out_ratio=0.2, max_epochs=2000,
                   zero_violation_threshold=1e-3, n_trials=3):
    for n in n_range:
        success = False
        for trial in range(n_trials):
            x_train, y_train = generate_constraints(target_func, w)
            x_val, y_val = generate_held_out(target_func, int(w * held_out_ratio))
            x_train_t = torch.FloatTensor(x_train).view(-1, 1)
            y_train_t = torch.FloatTensor(y_train).view(-1, 1)
            x_val_t = torch.FloatTensor(x_val).view(-1, 1)
            y_val_t = torch.FloatTensor(y_val).view(-1, 1)
            model = SingleLayerReLU(input_dim=1, hidden_dim=n, output_dim=1)
            train_err, val_err = train_network(model, x_train_t, y_train_t, x_val_t, y_val_t,
                                              max_epochs=max_epochs, lr=0.01)
            if val_err <= zero_violation_threshold:
                success = True
                break
        if success:
            return n
    return None

def run_experiment():
    target_func = lambda x: np.sin(2 * np.pi * x)
    w_values = [10, 20, 50, 100, 200, 500, 1000]
    n_range = list(range(8, 513, 8))
    held_out_ratio = 0.2
    max_epochs = 2000
    zero_violation_threshold = 1e-3
    n_trials = 3

    print(f"Experiment: sin(2*pi*x) fitting")
    print(f"w values: {w_values}")
    print(f"Width range: {n_range[0]} to {n_range[-1]}")
    print(f"Zero violation threshold: {zero_violation_threshold}\n")

    results = {}
    for w in w_values:
        print(f"Testing w = {w}...")
        N_c = find_min_width(w=w, n_range=n_range, target_func=target_func,
                             held_out_ratio=held_out_ratio, max_epochs=max_epochs,
                             zero_violation_threshold=zero_violation_threshold, n_trials=n_trials)
        results[w] = N_c
        if N_c is not None:
            print(f"  N_c({w}) = {N_c}")
        else:
            print(f"  N_c({w}) = UNREACHED")
        print()

    os.makedirs('/workspace/experiment/results', exist_ok=True)
    with open('/workspace/experiment/results/ricidity_results.json', 'w') as f:
        json.dump({'w_values': w_values, 'N_c': results}, f, indent=2)

    print("Results saved.")
    print("\nSummary:")
    for w, N_c in results.items():
        status = "RIGID" if N_c is not None else "FAT"
        print(f"  w={w}: N_c={N_c if N_c else 'unreachable'} -> {status}")

    return results

if __name__ == "__main__":
    run_experiment()