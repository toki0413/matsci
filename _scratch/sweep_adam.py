import numpy as np

def fit_adam(X, y, h, seed, steps=3000, lr=0.01, init=0.5):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((X.shape[1], h)) * init
    b1 = np.zeros((1, h))
    W2 = rng.standard_normal((h, 1)) * init
    b2 = np.zeros((1, 1))
    ps = [W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    b1_, b2_, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        H = np.tanh(X @ ps[0] + ps[1])
        out = H @ ps[2] + ps[3]
        d = (out - y) / X.shape[0]
        gW2 = H.T @ d
        gb2 = d.sum(axis=0, keepdims=True)
        dh = (d @ ps[2].T) * (1 - H ** 2)
        gW1 = X.T @ dh
        gb1 = dh.sum(axis=0, keepdims=True)
        gs = [gW1, gb1, gW2, gb2]
        for i in range(4):
            m[i] = b1_ * m[i] + (1 - b1_) * gs[i]
            v[i] = b2_ * v[i] + (1 - b2_) * (gs[i] ** 2)
            mh = m[i] / (1 - b1_ ** t)
            vh = v[i] / (1 - b2_ ** t)
            ps[i] = ps[i] - lr * mh / (np.sqrt(vh) + eps)
    return ps

def pred(X, ps):
    return np.tanh(X @ ps[0] + ps[1]) @ ps[2] + ps[3]

for kind in ["sin", "linear", "tanhlin"]:
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, (10, 1))
    if kind == "sin":
        y = np.sin(np.pi * X)
    elif kind == "linear":
        y = 0.5 + X
    else:
        y = np.tanh(3 * X - 1.5)
    Xv = np.linspace(0.0, 1.0, 25).reshape(-1, 1)
    yv = np.sin(np.pi * Xv) if kind == "sin" else (0.5 + Xv if kind == "linear" else np.tanh(3 * Xv - 1.5))
    errs = {}
    for h in [4, 8, 16, 32, 64]:
        ps = fit_adam(X, y, h, 0)
        errs[h] = round(float(np.max(np.abs(pred(Xv, ps) - yv))), 5)
    print(kind, errs)