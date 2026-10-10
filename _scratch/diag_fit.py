"""诊断 mlp_fit 收敛: 为什么简单正弦族拟合不到 1e-6."""
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location(
    "network_rigidity", "/workspace/examples/codelab_scaffolds/network_rigidity.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

w = 20
X = np.linspace(0.05, 0.95, w).reshape(-1, 1)
y = np.sin(np.pi * X)
Xv = np.linspace(0.02, 0.98, 200).reshape(-1, 1)
yv = np.sin(np.pi * Xv)

for h in (8, 16, 32):
    for ns in (1, 3, 8):
        m = mod.mlp_fit(X, y, h, seeds=ns, maxiter=20000, seed=0)
        tr = float(np.max(np.abs(mod.mlp_predict(m, X) - y)))
        ho = float(np.max(np.abs(mod.mlp_predict(m, Xv) - yv)))
        print("h=%2d starts=%d  mse=%.2e  train_max=%.2e  ho_max=%.2e"
              % (h, ns, m["train_mse"], tr, ho))