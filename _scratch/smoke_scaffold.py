"""离线冒烟: 验证修复后的脚手架模板可跑通 (小网格, 不触发广播错误)."""
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location(
    "network_rigidity", "/workspace/examples/codelab_scaffolds/network_rigidity.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# 1) 模板自身语法与形状
ns = {"np": np, **mod.primitives()}
exec(mod.TEMPLATE, ns)
family, run = ns["family"], ns["run"]

for kind in ("rigid", "fat"):
    for w in (5, 10):
        d = family(kind, w, 0)
        assert d["X"].shape == (w, 1), (kind, w, d["X"].shape)
        assert d["y"].shape == (w, 1), (kind, w, d["y"].shape)
        assert d["yv"].shape[1] == 1
        print(kind, "w=", w, "std(y)=%.3f" % float(np.std(d["y"])),
              "std(yv)=%.3f" % float(np.std(d["yv"])))

# 2) 小网格跑通 capacity_scan (含 fat 分支, 之前必炸的分支)
res = mod.capacity_scan(family, ws=(5, 10), widths=(4, 16), seeds=1)
print("success:", res["success"])
summ = res["summary"]
print("trend:", summ["trend"])
print("Nc:", summ["Nc"])
for k, v in summ["rows"].items():
    print("  ", k, "tr=%.2e ho=%.2e" % (v["train_err"], v["heldout_err"]))