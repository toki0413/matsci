import numpy as np

def fit(X, y, h, seed, steps, lr, init):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((X.shape[1], h)) * init
    b1 = np.zeros((1, h))
    W2 = rng.standard_normal((h, 1)) * init
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

def pred(X, p):
    W1, b1, W2, b2 = p
    return np.tanh(X @ W1 + b1) @ W2 + b2

rng = np.random.default_rng(0)
X = rng.uniform(0.0, 1.0, (10, 1))
y = np.sin(np.pi * X)
Xv = np.linspace(0.0, 1.0, 25).reshape(-1, 1)
yv = np.sin(np.pi * Xv)

for steps, lr, init in [(4000,0.05,0.3),(10000,0.1,0.3),(20000,0.1,1.0),(20000,0.2,1.0),(20000,0.05,1.0)]:
    errs = {}
    for h in [4,8,16,32,64]:
        p = fit(X,y,h,0,steps,lr,init)
        errs[h] = round(float(np.max(np.abs(pred(Xv,p)-yv))),4)
    print(f"steps={steps} lr={lr} init={init} -> {errs}")