#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os

torch.manual_seed(42)
np.random.seed(42)

class FeedforwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def generate_constraint(func_type, n_samples, device):
    x = torch.rand(n_samples, 1, device=device) * 2 - 1
    if func_type == "polynomial":
        y = torch.sin(2 * np.pi * x) + 0.5 * torch.sin(4 * np.pi * x)
    elif func_type == "rational":
        y = 1.0 / (1.0 + 25.0 * x**2)
    elif func_type == "exponential":
        x = torch.rand(n_samples, 1, device=device) * 4 - 2
        y = torch.exp(-x**2)
    elif func_type == "sinusoidal":
        x = torch.rand(n_samples, 1, device=device) * np.pi
        y = torch.sin(5 * x)
    else:
        raise ValueError(f"Unknown function type: {func_type}")
    return x, y

def train_network(model, x_train, y_train, x_val, y_val, device, n_epochs=5000, lr=1e-3):
    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        train_pred = model(x_train)
        val_pred = model(x_val)
        train_max_error = torch.max(torch.abs(train_pred - y_train)).item()
        val_max_error = torch.max(torch.abs(val_pred - y_val)).item()
    
    return train_max_error, val_max_error

def find_minimum_width(func_type, w, width_range, device, n_trials=3, threshold=1e-3):
    n_val = w // 4 if w > 4 else 5
    
    for width in width_range:
        for trial in range(n_trials):
            x, y = generate_constraint(func_type, w + n_val, device)
            indices = torch.randperm(w + n_val)
            train_idx = indices[:w]
            val_idx = indices[w:w + n_val]
            
            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]
            
            model = FeedforwardNetwork(input_dim=1, hidden_dim=width, output_dim=1, num_layers=2)
            train_error, val_error = train_network(model, x_train, y_train, x_val, y_val, device)
            
            if val_error <= threshold:
                return width
    return None

def run_experiment(func_type, w_values, width_range, device):
    results = {
        "function_type": func_type,
        "w_values": w_values,
        "width_range": width_range,
        "nc_values": [],
        "details": []
    }
    
    print(f"\n{'='*60}")
    print(f"Experiment: {func_type} function")
    print(f"{'='*60}")
    
    for w in w_values:
        print(f"\nw = {w}")
        nc = find_minimum_width(func_type, w, width_range, device)
        results["nc_values"].append(nc)
        
        detail = {"w": w, "N_c_w": nc}
        results["details"].append(detail)
        
        if nc is not None:
            print(f"N_c({w}) = {nc}")
        else:
            print(f"N_c({w}) = UNREACHABLE")
    
    output_dir = "/workspace/research_outputs/shusheng_rsi_run73/results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"rigidity_experiment_{func_type}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    func_type = "polynomial"
    w_values = [10, 20, 50, 100, 200]
    width_range = list(range(4, 51, 4))
    
    results = run_experiment(func_type, w_values, width_range, device)
    
    print(f"\nN_c(w) values: {results['nc_values']}")

if __name__ == "__main__":
    main()
