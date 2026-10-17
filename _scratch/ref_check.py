import sys
sys.path.insert(0, "/workspace/agent")
from huginn.research.code_lab import sandbox_run

CODE = '''
import numpy as np

def _fit(X, y, h, seed, steps=4000, lr=0.05):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((X.shape[1], h)) * 0.3
    b1 = np.zeros((1, h))
    W2 = rng.standard_normal((h, 1)) * 0.3
    b2 = np.zeros((1, 1))
    for _ in range(steps):
        H = np.tanh(X @ W1 + b1)
        out = H @ W2 + b2
        d = (out - y) / X.shape[0]
        W2 = W2 - lr * (H.T @ d)
        b2 = b2 - lr * d.sum(axis=0, keepdims=True)
        dh = (d @ W2.T) * (1 - H ** 2)
        W1 = W1 - lr * (X.T @ dh)
        b1 = b1 - lr * dh.sum(axis=0, keepdims=True)
    return (W1, b1, W2, b2)

def _pred(X, p):
    W1, b1, W2, b2 = p
    return np.tanh(X @ W1 + b1) @ W2 + b2

def run(cfg):
    seed = int(cfg.get('seed', 0))
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, (10, 1))
    y = np.sin(np.pi * X)
    Xv = np.linspace(0.0, 1.0, 25).reshape(-1, 1)
    yv = np.sin(np.pi * Xv)
    widths = [4, 8, 16, 32, 64]
    nc = -1
    errs = {}
    for h in widths:
        p = _fit(X, y, h, seed)
        e = float(np.max(np.abs(_pred(Xv, p) - yv)))
        errs['h%d' % h] = e
        if e < 1e-2 and nc < 0:
            nc = h
    return {"success": True,
            "summary": {"n_c": nc, "val_errs": errs},
            "objectives": {"n_c_reached": 1.0 if nc > 0 else 0.0}}
'''

res, reason = sandbox_run(CODE, {"seed": 0}, timeout=60)
print("reason:", reason)
print("res:", res)