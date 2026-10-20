"""诊断 4: 容差 & 解析梯度 对 收敛速度/精度 的影响."""
import time
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


def make(h, analytic):
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
        return float(np.mean((np.tanh(Xn @ W1 + b1) @ W2 + b2 - yn) ** 2))

    def grad(p):
        W1, b1, W2, b2 = unpack(p)
        A = Xn @ W1 + b1
        H = np.tanh(A)
        r = H @ W2 + b2 - yn
        g = 2.0 * r / Xn.shape[0]
        gW2 = H.T @ g
        gb2 = g.sum(0)
        gA = (g @ W2.T) * (1 - H ** 2)
        gW1 = Xn.T @ gA
        gb1 = gA.sum(0)
        out = np.empty(n)
        i = 0
        out[i:i + d * h] = gW1.reshape(-1); i += d * h
        out[i:i + h] = gb1; i += h
        out[i:i + h] = gW2.reshape(-1); i += h
        out[i:i + 1] = gb2
        return out
    return unpack, loss, grad


def run(h, analytic, gi, fi, mi=4000, mf=60000, starts=3):
    unpack, loss, grad = make(h, analytic)
    best, bp, nfev = np.inf, None, 0
    t0 = time.time()
    for s in range(starts):
        rng = np.random.default_rng(1000 + 7 * h + s)
        p0 = rng.standard_normal(len_ := (1 * h + h + h + 1)) * 0.5
        kw = {"jac": grad} if analytic else {}
        r = minimize(loss, p0, method="L-BFGS-B", **kw,
                     options={"maxiter": mi, "maxfun": mf, "ftol": fi, "gtol": gi})
        nfev += r.nfev
        if r.fun < best:
            best, bp = float(r.fun), r.x
    W1, b1, W2, b2 = unpack(bp)
    def pred(Xq):
        return (np.tanh(((Xq - Xm) / Xs) @ W1 + b1) @ W2 + b2) * ys + ym
    tr = float(np.max(np.abs(pred(X) - y)))
    ho = float(np.max(np.abs(pred(Xv) - yv)))
    return best, tr, ho, nfev, time.time() - t0


for h in (2, 4, 8):
    for tag, an, gi, fi in (("fd gtol1e-5", False, 1e-5, 2.2e-9),
                            ("fd gtol1e-10", False, 1e-10, 1e-14),
                            ("an gtol1e-10", True, 1e-10, 1e-14),
                            ("an gtol1e-12", True, 1e-12, 1e-15)):
        mse, tr, ho, nfev, dt = run(h, an, gi, fi)
        print("h=%d %-13s mse=%.2e tr=%.2e ho=%.2e nfev=%d %.2fs"
              % (h, tag, mse, tr, ho, nfev, dt))