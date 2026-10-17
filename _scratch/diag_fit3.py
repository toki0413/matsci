"""诊断 3: 不同 maxiter/maxfun 下 rigid 收敛性与单次耗时 (含 fat 最坏情况)."""
import importlib.util
import time
import numpy as np

spec = importlib.util.spec_from_file_location(
    "network_rigidity", "/workspace/examples/codelab_scaffolds/network_rigidity.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ns = {"np": np, **mod.primitives()}
exec(mod.TEMPLATE, ns)
fam = ns["family"]

print("== rigid 收敛 vs maxiter/maxfun ==")
d = fam("rigid", 20, 0)
X, y, Xv, yv = d["X"], d["y"], d["Xv"], d["yv"]
for mi, mf in ((20000, 200000), (3000, 30000), (1200, 12000)):
    for h in (2, 4, 8):
        t0 = time.time()
        m = mod.mlp_fit(X, y, h, seeds=3, maxiter=mi, maxfun=mf, seed=0)
        tr = float(np.max(np.abs(mod.mlp_predict(m, X) - y)))
        ho = float(np.max(np.abs(mod.mlp_predict(m, Xv) - yv)))
        print("  mi=%5d mf=%6d h=%d tr=%.2e ho=%.2e %.1fs"
              % (mi, mf, h, tr, ho, time.time() - t0))

print("== fat 最坏情况耗时 (h=64, 3 starts) ==")
d = fam("fat", 20, 0)
X, y, Xv, yv = d["X"], d["y"], d["Xv"], d["yv"]
for mi, mf in ((20000, 200000), (3000, 30000), (1200, 12000)):
    t0 = time.time()
    m = mod.mlp_fit(X, y, 64, seeds=3, maxiter=mi, maxfun=mf, seed=0)
    print("  mi=%5d mf=%6d h=64  %.1fs" % (mi, mf, time.time() - t0))