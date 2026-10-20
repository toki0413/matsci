"""刚性示例可达性探针: 模板 rigid 族在大 w / 大宽度下是否可达零违规."""
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location(
    "network_rigidity", "/workspace/examples/codelab_scaffolds/network_rigidity.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ns = {"np": np, **mod.primitives()}
exec(mod.TEMPLATE, ns)
family = ns["family"]

res = mod.capacity_scan(family, kinds=("rigid",), ws=(10, 20, 40),
                        widths=(8, 16, 32, 64), seeds=2, seed=0)
print("trend:", res["summary"]["trend"], "Nc:", res["summary"]["Nc"])
for k, v in res["summary"]["rows"].items():
    print("  ", k, "tr=%.2e ho=%.2e" % (v["train_err"], v["heldout_err"]))
print("anchor:", res["summary"]["anchor"])