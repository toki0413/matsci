"""系统性审计: 协作对象的 __getattr__/__setattr__ 转发完整性.

三类缺陷 (run88 暴露的 _is_code_experiment 死码属第 3 类):
  1. _OWN_ATTRS 里列了但类上没定义 → 赋值语义错 (本对象方法被误转发回引擎).
  2. 类上定义了协作方法但漏进 _OWN_ATTRS → 测试/覆写时被误写回引擎.
  3. 类内部 self._x() 调用的名字, 既不在本类, 也不在引擎 (薄委托缺失) → 运行时
     AttributeError / 静默降级 (死码).
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/agent")

from huginn.autoloop.engine import AutoloopEngine  # noqa: E402

BASE = Path("/workspace/agent/huginn/autoloop")
COLLABS = [
    "plan_check", "engine_act", "engine_reflect", "hypothesis_loop",
    "engine_control", "engine_observe", "cognitive_loop", "visual_inspect",
    "engine_perceive",
]
engine_dir = set(dir(AutoloopEngine))


def _class_defs(tree: ast.Module):
    """返回 {class_name: (method_names, own_attrs, self_calls)}."""
    out = {}
    for node in tree.body:  # 只看模块顶层类
        if not isinstance(node, ast.ClassDef):
            continue
        methods: set[str] = set()
        own: set[str] = set()
        self_calls: set[str] = set()
        for sub in ast.walk(node):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.add(sub.name)
            if isinstance(sub, ast.Assign):
                for t in sub.targets:
                    if isinstance(t, ast.Name) and t.id == "_OWN_ATTRS":
                        if isinstance(sub.value, ast.Call):
                            for a in sub.value.args:
                                if isinstance(a, ast.Set):
                                    for el in a.elts:
                                        if isinstance(el, ast.Constant):
                                            own.add(el.value)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                v = sub.func.value
                if isinstance(v, ast.Name) and v.id == "self":
                    self_calls.add(sub.func.attr)
        out[node.name] = (methods, own, self_calls)
    return out


for stem in COLLABS:
    p = BASE / f"{stem}.py"
    if not p.exists():
        continue
    tree = ast.parse(p.read_text(encoding="utf-8"))
    for cls, (methods, own, self_calls) in _class_defs(tree).items():
        if not own:  # 非转发协作对象
            continue
        # 1. _OWN_ATTRS 里但类上没定义
        ghost = sorted(own - methods)
        # 2. 类上定义的协作方法漏进 _OWN_ATTRS (排除 __init__ 等 dunder)
        missing_own = sorted(
            m for m in methods
            if not m.startswith("__") and m not in own
        )
        # 3. self._x() 死码: 既不在本类, 也不在引擎
        dead = sorted(
            c for c in self_calls
            if not c.startswith("__") and c not in methods and c not in engine_dir
        )
        tag = f"{stem}.{cls}"
        if ghost:
            print(f"[{tag}] _OWN_ATTRS 幽灵 (列了但未定义): {ghost}")
        if missing_own:
            print(f"[{tag}] 类方法未列入 _OWN_ATTRS: {missing_own}")
        if dead:
            print(f"[{tag}] self._x() 引擎上解析不到 (可能死码): {dead}")
print("审计完成")