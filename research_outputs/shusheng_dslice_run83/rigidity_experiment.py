#!/usr/bin/env python3
"""
Solution Space Rigidity Experiment for Small Feedforward Networks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

np.random.seed(42)
torch.manual_seed(42)

class TargetFunction:
    def __init__(self, d=1):
        self.d = d
    def __call__(self, x):
        return np.sin(x) + 0.1 * x**2

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def train_model(model, x_train, y_train, x_val, y_val, lr=0.01, max_epochs=10000, patience=100):
    x_train = np.asarray(x_train).reshape(-1, 1)
    x_val = np.asarray(x_val).reshape(-1, 1)
    y_train = np.asarray(y_train).reshape(-1, 1)
    y_val = np.asarray(y_val).reshape(-1, 1)
    
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    loader = torch.utils.data.DataLoader(dataset, batch_size=min(32, len(x_train)), shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        for xb, yb in loader:
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            train_pred = model(torch.FloatTensor(x_train))
            train_max_err = float(torch.max(torch.abs(train_pred - torch.FloatTensor(y_train))).numpy())
            val_pred = model(torch.FloatTensor(x_val))
            val_max_err = float(torch.max(torch.abs(val_pred - torch.FloatTensor(y_val))).numpy())
        
        if val_max_err < best_val_loss:
            best_val_loss = val_max_err
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return train_max_err, val_max_err

def find_nc_w(w, target_func, input_dim=1, width_range=list(range(4, 101)), n_samples_val=50, seed=42):
    np.random.seed(seed + w)
    x_train = np.random.uniform(-2, 2, w)
    y_train = target_func(x_train)
    np.random.seed(seed + w + 1000)
    x_val = np.random.uniform(-2, 2, n_samples_val)
    y_val = target_func(x_val)
    
    N_c = None
    train_err = None
    val_err = None
    
    for N in width_range:
        model = TwoLayerNet(input_dim, N)
        train_max_err, val_max_err = train_model(model, x_train, y_train, x_val, y_val, lr=0.01, max_epochs=10000, patience=200)
        if val_max_err <= 1e-3:
            N_c = N
            train_err = train_max_err
            val_err = val_max_err
            break
    
    return N_c, train_err, val_err

def main():
    print("=" * 70)
    print("Solution Space Rigidity Experiment")
    print("Target: f(x) = sin(x) + 0.1*x^2")
    print("Zero violation: max abs error <= 1e-3")
    print("Width range: 4-100")
    print("=" * 70)
    
    target_func = TargetFunction(d=1)
    w_values = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
    width_range = list(range(4, 101))
    
    results = []
    print(f"\nw values: {w_values}\n")
    print(f"{'w':>5} {'N_c':>6} {'train_err':>12} {'val_err':>12} {'status':>10}")
    print("-" * 50)
    
    for w in w_values:
        N_c, train_err, val_err = find_nc_w(w, target_func, width_range=width_range, seed=42)
        if N_c is not None:
            print(f"{w:5d} {N_c:6d} {train_err:.2e} {val_err:.2e} {'FOUND':>10}")
            results.append((w, N_c, train_err, val_err))
        else:
            print(f"{w:5d} {'None':>6} {'-':>12} {'-':>12} {'NOT FOUND':>10}")
            results.append((w, None, None, None))
    
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)
    
    valid_results = [(w, N_c) for w, N_c, _, _ in results if N_c is not None]
    
    if len(valid_results) >= 2:
        widths = [N_c for w, N_c in valid_results]
        unique_widths = set(widths)
        
        if len(unique_widths) == 1:
            print(f"\n*** RESULT: RIGID SOLUTION SPACE ***")
            print(f"N_c(w) = {widths[0]} (constant) for all w in {[w for w, _ in valid_results]}")
        else:
            w_vals, n_c_vals = zip(*valid_results)
            coeffs_linear = np.polyfit(w_vals, n_c_vals, 1)
            r_squared_linear = np.corrcoef(w_vals, n_c_vals)[0, 1]**2
            print(f"\n*** RESULT: N_c(w) varies with w ***")
            print(f"Linear fit: N_c = {coeffs_linear[0]:.3f} * w + {coeffs_linear[1]:.3f} (R^2 = {r_squared_linear:.3f})")
            print(f"Widths: {widths}")
    else:
        print("\n*** INSUFFICIENT DATA ***")
    
    os.makedirs('results', exist_ok=True)
    with open('results/rigidity_results.txt', 'w') as f:
        f.write("Solution Space Rigidity Experiment Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Target: f(x) = sin(x) + 0.1*x^2\n")
        f.write("Threshold: 1e-3\n")
        f.write("w\tN_c\ttrain_err\tval_err\n")
        for w, N_c, train_err, val_err in results:
            if N_c is not None:
                f.write(f"{w}\t{N_c}\t{train_err:.2e}\t{val_err:.2e}\n")
    
    print(f"\nResults saved to results/rigidity_results.txt")

if __name__ == "__main__":
    main()
