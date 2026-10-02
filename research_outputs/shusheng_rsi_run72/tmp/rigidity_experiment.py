#!/usr/bin/env python3
"""Bootstrap Rigidity Experiment for Small Feedforward Networks"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os

torch.manual_seed(42)
np.random.seed(42)

class ReLUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return self.output(self.activation(self.hidden(x)))

def generate_target_function(x):
    return torch.sin(torch.pi * x)

def generate_constraint_points(w, n_total=100):
    all_points = torch.rand(n_total).view(-1, 1)
    indices = torch.randperm(n_total)
    train_idx = indices[:w]
    held_out_idx = indices[w:]
    x_train = all_points[train_idx]
    y_train = generate_target_function(x_train)
    x_held = all_points[held_out_idx]
    y_held = generate_target_function(x_held)
    return x_train, y_train, x_held, y_held

def train_network(model, x_train, y_train, x_held, y_held, n_epochs=5000, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_pred = model(x_train)
        held_pred = model(x_held)
        train_max_err = torch.abs(train_pred - y_train).max().item()
        held_max_err = torch.abs(held_pred - y_held).max().item()
    return train_max_err, held_max_err

def find_nc_w(w, width_range, n_trials=2, threshold=1e-3):
    for width in width_range:
        for trial in range(n_trials):
            model = ReLUNetwork(input_dim=1, hidden_dim=width)
            x_train, y_train, x_held, y_held = generate_constraint_points(w)
            train_err, held_err = train_network(model, x_train, y_train, x_held, y_held)
            if held_err <= threshold:
                return width
    return None

def run_experiments():
    w_values = [5, 10, 20, 30, 50, 75, 100]
    width_range = list(range(5, 51))
    n_trials = 2
    threshold = 1e-3
    
    results = {}
    print(f"Parameters: w={w_values}, width_range={width_range[0]}-{width_range[-1]}, threshold={threshold}\n")
    
    for w in w_values:
        print(f"Testing w = {w}...")
        nc_w = find_nc_w(w, width_range, n_trials=n_trials, threshold=threshold)
        results[w] = nc_w
        if nc_w is not None:
            print(f"  N_c({w}) = {nc_w} (RIGID)\n")
        else:
            print(f"  N_c({w}) = UNATTAINABLE (N_c>{width_range[-1]})\n")
    
    os.makedirs('/tmp/experiments_results', exist_ok=True)
    with open('/tmp/experiments_results/nc_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nSummary:")
    print(f"{'w':<6} {'N_c(w)':<12} {'Status':<20}")
    print("-" * 40)
    for w, nc in results.items():
        if nc is not None:
            status = f"RIGID (N_c={nc})"
        else:
            status = f"UNATTAINABLE (N_c>{width_range[-1]})"
        print(f"{w:<6} {nc if nc else 'inf':<12} {status:<20}")

if __name__ == '__main__':
    run_experiments()