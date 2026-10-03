"""计时: 完整默认扫描 (解析梯度 + fit_starts=2) 逐 (kind,w) 打印耗时与结果."""
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

t_all = time.time()
for kind in ("rigid", "fat"):
    for w in (5, 10, 20):
        t0 = time.time()
        r = mod.capacity_scan(fam, kinds=(kind,), ws=(w,))
        s = r["summary"]
        print("%-5s w=%-3d %6.1fs  Nc=%s  %s"
              % (kind, w, time.time() - t0, s["Nc"][kind], s["trend"][kind]))
        row = ["h%d:tr%.1e/ho%.1e" % (h, s["rows"]["%s_w%d_h%d" % (kind, w, h)]["train_err"],
                                      s["rows"]["%s_w%d_h%d" % (kind, w, h)]["heldout_err"])
               for h in (2, 4, 8, 16, 32, 64)]
        print("      " + "  ".join(row))
print("TOTAL %.1fs" % (time.time() - t_all))