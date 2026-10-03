"""诊断 2: 小宽度(h=2..8)下, 更多起点/更严容差能否把 sin(pi x) 拟合到 1e-5."""
import numpy as np
from scipy.optimize import minimize

w = 20
X = np.linspace(0.05, 0.95, w).reshape(-1, 1)
y = np.sin(np.pi * X)
Xv = np.linspace(0.02, 0.98, 200).reshape(-1, 1)
yv = np.sin(np.pi * Xv)

Xm, Xs = X.mean(0), X.std(0) + 1e-9
Xn = (X - Xm) / Xs
ym, ys = float(y.mean()), float(y.std()) + 1e-9
yn = (y - ym) / ys


def fit(h, starts, opt):
    d = 1
    n = d * h + h + h + 1

    def unpack(p):
        i = 0
        W1 = p[i:i + d * h].reshape(d, h); i += d * h
        b1 = p[i:i + h]; i += h
        W2 = p[i:i + h].reshape(h, 1); i += h
        b2 = p[i:i + 1].reshape(1, 1)
        return W1, b1, W2, b2

    def loss(p):
        W1, b1, W2, b2 = unpack(p)
        out = np.tanh(Xn @ W1 + b1) @ W2 + b2
        return float(np.mean((out - yn) ** 2))

    best, bp = np.inf, None
    for s in range(starts):
        rng = np.random.default_rng(1000 + 7 * h + s)
        p0 = rng.standard_normal(n) * 0.5
        r = minimize(loss, p0, method="L-BFGS-B", options=opt)
        if r.fun < best:
            best, bp = float(r.fun), r.x
    W1, b1, W2, b2 = unpack(bp)
    def pred(Xq):
        Xq = (Xq - Xm) / Xs
        return (np.tanh(Xq @ W1 + b1) @ W2 + b2) * ys + ym
    return best, float(np.max(np.abs(pred(X) - y))), float(np.max(np.abs(pred(Xv) - yv)))


tight = {"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-12, "maxfun": 200000}
loose = {"maxiter": 20000}
print("== default opts ==")
for h in (2, 4, 8):
    for st in (8, 32):
        m, tr, ho = fit(h, st, loose)
        print("  h=%d st=%2d mse=%.2e tr=%.2e ho=%.2e" % (h, st, m, tr, ho))
print("== tight opts ==")
for h in (2, 4, 8):
    for st in (8, 32):
        m, tr, ho = fit(h, st, tight)
        print("  h=%d st=%2d mse=%.2e tr=%.2e ho=%.2e" % (h, st, m, tr, ho))