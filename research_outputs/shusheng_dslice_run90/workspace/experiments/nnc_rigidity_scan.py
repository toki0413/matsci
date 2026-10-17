#!/usr/bin/env python3
"""Neural Network Capacity Scan for Solution Space Rigidity."""

import numpy as np
from scipy.optimize import minimize
import math

# ============================================================
# Primitive functions (copied from scaffold)
# ============================================================

def _as_2d(a):
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a

def poly_basis(X, n):
    X = _as_2d(X)
    x = X[:, 0]
    powers = np.arange(int(n))
    return x[:, None] ** powers[None, :]

def zero_violation(max_abs_err, tol=1e-3):
    try:
        e = float(max_abs_err)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(e):
        return False
    return e <= float(tol)

def mlp_fit(X, y, h, seeds=3, maxiter=3000, maxfun=30000, seed=0):
    X = _as_2d(X)
    y = _as_2d(y)
    h = int(h)
    d = int(X.shape[1])
    N = int(X.shape[0])
    n = d * h + h + h + 1

    def unpack(p):
        i = 0
        W1 = p[i:i + d * h].reshape(d, h); i += d * h
        b1 = p[i:i + h]; i += h
        W2 = p[i:i + h].reshape(h, 1); i += h
        b2 = p[i:i + 1].reshape(1, 1)
        return W1, b1, W2, b2

    Xm, Xs = X.mean(0), X.std(0) + 1e-9
    Xn = (X - Xm) / Xs
    ym, ys = float(y.mean()), float(y.std()) + 1e-9
    yn = (y - ym) / ys

    def loss(p):
        W1, b1, W2, b2 = unpack(p)
        out = np.tanh(Xn @ W1 + b1) @ W2 + b2
        return float(np.mean((out - yn) ** 2))

    def grad(p):
        W1, b1, W2, b2 = unpack(p)
        H = np.tanh(Xn @ W1 + b1)
        g = 2.0 * (H @ W2 + b2 - yn) / N
        gW2 = H.T @ g
        gb2 = g.sum(0)
        gA = (g @ W2.T) * (1.0 - H ** 2)
        gW1 = Xn.T @ gA
        gb1 = gA.sum(0)
        o = np.empty(n)
        i = 0
        o[i:i + d * h] = gW1.reshape(-1); i += d * h
        o[i:i + h] = gb1; i += h
        o[i:i + h] = gW2.reshape(-1); i += h
        o[i:i + 1] = gb2
        return o

    best_fun = float('inf')
    best_p = None
    opts = {'maxiter': int(maxiter), 'ftol': 1e-14, 'gtol': 1e-10,
            'maxfun': int(maxfun)}
    for s in range(int(seeds)):
        rng = np.random.default_rng(int(seed) + 101 * s + 7 * h + d)
        p0 = rng.standard_normal(n) * 0.5
        res = minimize(loss, p0, jac=grad, method='L-BFGS-B', options=opts)
        if float(res.fun) < best_fun:
            best_fun = float(res.fun)
            best_p = res.x
    return {'params': best_p.tolist(), 'h': h, 'd': d, 'train_mse': best_fun,
            'scaler': {'Xm': Xm.tolist(), 'Xs': Xs.tolist(),
                       'ym': ym, 'ys': ys}}

def mlp_predict(model, X):
    X = _as_2d(X)
    p = np.asarray(model['params'], dtype=float)
    h = int(model['h'])
    d = int(model['d'])
    sc = model.get('scaler')
    if sc is None:
        Xn = X
    else:
        X = (X - np.asarray(sc['Xm'], dtype=float)) / np.asarray(sc['Xs'], dtype=float)
        Xn = X
    i = 0
    W1 = p[i:i + d * h].reshape(d, h); i += d * h
    b1 = p[i:i + h]; i += h
    W2 = p[i:i + h].reshape(h, 1); i += h
    b2 = p[i:i + 1].reshape(1, 1)
    out = np.tanh(Xn @ W1 + b1) @ W2 + b2
    if sc is not None:
        out = out * float(sc['ys']) + float(sc['ym'])
    return out

# ============================================================
# Constraint families
# ============================================================

def family(kind, w, seed):
    rng = np.random.default_rng(seed)

    if kind == 'rigid':
        # Rigid: simple smooth function f(x) = sin(pi * x)
        # Low-dimensional (1-dimensional function family), constraints are consistent
        X = np.linspace(0.05, 0.95, w).reshape(-1, 1)
        Xv = np.linspace(0.02, 0.98, 200).reshape(-1, 1)
        y = np.sin(np.pi * X[:, 0]).reshape(-1, 1)
        yv = np.sin(np.pi * Xv[:, 0]).reshape(-1, 1)
        return {'X': X, 'y': y, 'Xv': Xv, 'yv': yv}

    elif kind == 'fat':
        # Fat: Fourier series with w terms - complexity grows with w
        # f_w(x) = sum_{k=1}^{w} (1/k) * sin(k * pi * x)
        # As w increases, more high-frequency components are added
        X = np.linspace(0.05, 0.95, w).reshape(-1, 1)
        Xv = np.linspace(0.02, 0.98, 200).reshape(-1, 1)
        y = np.zeros((w, 1))
        yv = np.zeros((200, 1))
        for k in range(1, w + 1):
            y[:, 0] += (1.0 / k) * np.sin(k * np.pi * X[:, 0])
            yv[:, 0] += (1.0 / k) * np.sin(k * np.pi * Xv[:, 0])
        return {'X': X, 'y': y, 'Xv': Xv, 'yv': yv}

    else:
        raise ValueError(f"Unknown kind: {kind}")

# ============================================================
# Main experiment
# ============================================================

def run(cfg):
    seed = int(cfg.get('seed', 42))

    # Scan parameters
    w_values = [2, 4, 8, 16]  # constraint points (w)
    h_values = [2, 4, 8, 16, 32, 64]  # hidden layer widths (scan range)
    max_h = max(h_values)

    # Results: N_c[kind] = list of N_c values for each w
    N_c_rigid = []
    N_c_fat = []
    err_rigid = []
    err_fat = []

    for kind, target_list in [('rigid', N_c_rigid), ('fat', N_c_fat)]:
        N_c_list = []
        err_list = []

        for w_idx, w in enumerate(w_values):
            min_Nc = None
            min_err = float('inf')

            for h in h_values:
                # Get data for this (kind, w, seed)
                data = family(kind, w, seed=seed + w_idx * 1000 + h)
                X, y = data['X'], data['y']
                Xv, yv = data['Xv'], data['yv']

                # Train network
                model = mlp_fit(X, y, h=h, seeds=3, seed=seed)

                # Evaluate on held-out validation set
                yv_pred = mlp_predict(model, Xv)
                max_err = float(np.max(np.abs(yv_pred - yv)))

                # Check zero violation using the fixed criterion
                if zero_violation(max_err):
                    if min_Nc is None or h < min_Nc:
                        min_Nc = h
                        min_err = max_err

            # Record result
            if min_Nc is None:
                # Zero violation not achieved in scan range
                N_c_val = max_h + 1  # Report as upper bound (finite number)
                err_list.append(min_err if min_err != float('inf') else max_err)
            else:
                N_c_val = min_Nc
                err_list.append(min_err)

            N_c_list.append(N_c_val)
            print(f"  {kind}, w={w}, N_c={N_c_val}, err={min_err:.6e}")

        if kind == 'rigid':
            N_c_rigid = N_c_list
            err_rigid = err_list
        else:
            N_c_fat = N_c_list
            err_fat = err_list

    # Build output
    summary = {
        'note': (f"Rigid N_c(w)={N_c_rigid}, Fat N_c(w)={N_c_fat} | "
                 f"Rigid err={err_rigid}, Fat err={err_fat}"),
        'rigid_Nc': N_c_rigid,
        'fat_Nc': N_c_fat,
        'rigid_max_err': err_rigid,
        'fat_max_err': err_fat,
        'w_values': w_values,
        'h_values': h_values
    }

    objectives = {
        'rigid_Nc': [N_c_rigid],
        'fat_Nc': [N_c_fat],
        'rigid_err': [err_rigid],
        'fat_err': [err_fat],
        'score': 0.5
    }

    return {
        'success': True,
        'summary': summary,
        'objectives': objectives
    }

# ============================================================
# Run experiment
# ============================================================

if __name__ == '__main__':
    cfg = {'seed': 42}
    result = run(cfg)
    print("\n=== Summary ===")
    print(f"Rigid N_c(w): {result['summary']['rigid_Nc']}")
    print(f"Fat N_c(w): {result['summary']['fat_Nc']}")
    print(f"Rigid errors: {result['summary']['rigid_max_err']}")
    print(f"Fat errors: {result['summary']['fat_max_err']}")