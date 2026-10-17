"""审计: PlanCheck 经 self._x 转发的调用, 引擎是否真有对应方法.

背景: PlanCheck.__getattr__ 把未定义属性读转发到 engine. 若 engine 也没定义
(未做薄委托), 调用会 AttributeError —— 补丁就成死码 (run88 报告域漂移根因).
这里静态扫 plan_check.py 里的 self._method(...) 调用, 逐个在 AutoloopEngine
上试解析.
"""
import ast
import re
import sys

sys.path.insert(0, "/workspace/agent")

SRC = "/workspace/agent/huginn/autoloop/plan_check.py"
text = open(SRC, encoding="utf-8").read()
tree = ast.parse(text)

called: set[str] = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        v = node.func.value
        if isinstance(v, ast.Name) and v.id == "self":
            called.add(node.func.attr)

from huginn.autoloop.engine import AutoloopEngine  # noqa: E402

# 引擎上真实可解析的名字: 类属性 + 各协作对象里存在的方法也因其 __getattr__ 有效
have = set(dir(AutoloopEngine))
public_missing = []
for name in sorted(called):
    if name.startswith("__"):
        continue
    if name in have:
        continue
    # 试试引擎实例能否通过 __getattr__ 拿到 (协作对象转发)
    public_missing.append(name)

print("self._x() 调用总数:", len(called))
print("引擎类不可直接解析的:", public_missing)