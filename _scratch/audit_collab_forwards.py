"""跨协作对象审计: __getattr__ 转发到引擎的 self._x(...) 调用, 引擎是否真有.

对每个协作对象文件, 静态收集 self._method(...) 调用名, 减去本类/基类定义的名字,
剩下的须能在 AutoloopEngine (含薄委托) 上解析. 解析不到 = 死码 (AttributeError).
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/agent")

FILES = [
    "plan_check", "engine_act", "engine_reflect", "hypothesis_loop",
    "engine_control", "engine_observe", "cognitive_loop", "visual_inspect",
    "engine_perceive",
]
BASE = Path("/workspace/agent/huginn/autoloop")

from huginn.autoloop.engine import AutoloopEngine  # noqa: E402

engine_dir = set(dir(AutoloopEngine))


def _local_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


for stem in FILES:
    p = BASE / f"{stem}.py"
    if not p.exists():
        continue
    tree = ast.parse(p.read_text(encoding="utf-8"))
    local = _local_names(tree)
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            v = node.func.value
            if isinstance(v, ast.Name) and v.id == "self":
                called.add(node.func.attr)
    dead = sorted(
        n for n in called
        if not n.startswith("__")
        and n not in local
        and n not in engine_dir
    )
    if dead:
        print(f"[{stem}] 引擎上解析不到 (可能死码): {dead}")
    else:
        print(f"[{stem}] OK")