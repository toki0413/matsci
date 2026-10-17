"""探针判别性诊断: 给 canonical rigid/fat 族, 测机械 (mlp_fit/zero_violation) 能否分开."""
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location(
    "network_rigidity", "/workspace/examples/codelab_scaffolds/network_rigidity.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

WIDTHS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def family(kind, w, seed):
    rng = np.random.default_rng(seed)
    X = np.linspace(0.05, 0.95, w).reshape(-1, 1)
    Xv = np.linspace(0.02, 0.98, 200).reshape(-1, 1)
    x, xv = X[:, 0], Xv[:, 0]
    if kind == "rigid":
        y, yv = np.sin(np.pi * x), np.sin(np.pi * xv)
    else:
        c = rng.standard_normal(15)
        y = sum(c[k] * x ** k for k in range(15))
        yv = sum(c[k] * xv ** k for k in range(15))
    return {"X": X, "y": y.reshape(-1, 1), "Xv": Xv, "yv": yv.reshape(-1, 1)}


for kind in ("rigid", "fat"):
    for w in (6, 10, 20, 40):
        d = family(kind, w, 0)
        errs = {}
        Nc = None
        for h in WIDTHS:
            m = mod.mlp_fit(d["X"], d["y"], h, seeds=3, seed=0)
            err = float(np.max(np.abs(mod.mlp_predict(m, d["Xv"]) - d["yv"])))
            errs[h] = err
            if Nc is None and mod.zero_violation(err):
                Nc = h
        print(f"{kind:5s} w={w:3d}  N_c={str(Nc):>4s}  "
              f"errs=" + " ".join(f"h{h}={errs[h]:.1e}" for h in WIDTHS))