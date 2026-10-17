"""验证: 修复后的脚手架用**默认参数**跑模板示例的完整容量扫描."""
import importlib.util
import json
import time
import numpy as np

spec = importlib.util.spec_from_file_location(
    "network_rigidity", "/workspace/examples/codelab_scaffolds/network_rigidity.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ns = {"np": np, **mod.primitives()}
exec(mod.TEMPLATE, ns)

t0 = time.time()
res = mod.capacity_scan(ns["family"], seed=0)
print("elapsed %.1fs" % (time.time() - t0), "success:", res["success"])
s = res["summary"]
print("trend:", s["trend"])
print("Nc:", s["Nc"])
print("anchor:", s["anchor"])
print("overlap:", s["heldout_overlap"], "warnings:", s["warnings"])
for k in sorted(s["rows"]):
    v = s["rows"][k]
    print("  %-18s tr=%.2e ho=%.2e" % (k, v["train_err"], v["heldout_err"]))