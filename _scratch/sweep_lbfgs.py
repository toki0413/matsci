import numpy as np
from scipy.optimize import minimize

def build(h, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((1, h)) * 0.5, np.zeros((1, h)),
            rng.standard_normal((h, 1)) * 0.5, np.zeros((1, 1))]

def unpack(p, h):
    W1 = p[:h].reshape(1, h); b1 = p[h:2*h].reshape(1, h)
    W2 = p[2*h:2*h+h].reshape(h, 1); b2 = p[2*h+h:2*h+h+1].reshape(1, 1)
    return W1, b1, W2, b2

def make(h, X, y):
    def loss(p):
        W1, b1, W2, b2 = unpack(p, h)
        out = np.tanh(X @ W1 + b1) @ W2 + b2
        r = (out - y).ravel()
        return float(np.mean(r ** 2))
    return loss

def pred(p, h, X):
    W1, b1, W2, b2 = unpack(p, h)
    return np.tanh(X @ W1 + b1) @ W2 + b2

rng = np.random.default_rng(0)
X = rng.uniform(0.0, 1.0, (10, 1)); y = np.sin(np.pi * X)
Xv = np.linspace(0.0, 1.0, 25).reshape(-1, 1); yv = np.sin(np.pi * Xv)

for h in [2, 4, 8, 16]:
    best = 1e9
    for s in [0, 1, 2]:
        p0 = np.concatenate([a.ravel() for a in build(h, s)])
        r = minimize(make(h, X, y), p0, method="L-BFGS-B",
                     options={"maxiter": 50000, "ftol": 1e-15, "gtol": 1e-12})
        e = float(np.max(np.abs(pred(r.x, h, Xv) - yv)))
        best = min(best, e)
    print(f"h={h} best_maxval_err={best:.2e}")