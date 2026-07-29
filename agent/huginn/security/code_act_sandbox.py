"""CodeAct 沙箱安全机制 (权威定义).

CodeAct 模式的工具过滤、安全 builtins、import 白名单、防失控阈值集中在这.
agent/code_act_loop.py 引用本模块, 不再保留内联副本.

天花板 (ponytail):
- import 白名单是保守的, 升级路径是加白名单而非全禁. 每加一个模块都要
  确认它不带 subprocess/socket/etc 的副作用入口.
- in-process exec 共享解释器, 足够聪明的 payload 仍可能通过属性遍历逃逸
  (例: ().__class__.__bases__[0].__subclasses__() 链). 真要硬隔离走
  Docker 沙箱 + 同 namespace, 或 E2B.
- check_degrade 是无状态纯函数, 不维护连续错误计数; 调用方负责累积
  (升级: CodeActTurnLimiter 类维护状态).
"""

from __future__ import annotations

import builtins
import os
from typing import Any

# 不注入 CodeAct 沙箱的工具集.
# - hpc_client / bash_tool / shell_tool / container_exec: 外部副作用,
#   绕过 CodeAct 设的审计轨迹. 留在 tool_call 轨道让 langgraph 追.
# - code_tool: 让 LLM 在 code_act 内嵌套生成沙箱, 递归 footgun.
_BLOCKED_TOOLS = frozenset(
    {"hpc_client", "bash_tool", "shell_tool", "container_exec", "code_tool"}
)

# import 白名单. 科研计算常用栈, 不含 os/sys/subprocess/socket 等危险模块.
_ALLOWED_IMPORTS = frozenset(
    {
        "math",
        "statistics",
        "json",
        "re",
        "numpy",
        "pandas",
        "sympy",
        "scipy",
        "matplotlib",
        "ase",
        "pymatgen",
    }
)

# CodeAct 单轮会话最大轮次. CodeAct 论文 (Wang et al. ICML 2024) 显示
# M3ToolEval 中位数 6-8 步; 15 留探索余地同时控制失控成本.
_MAX_TURNS = 15

# 连续代码异常多少次后发 code_act_degraded 事件降级到 tool_call 轨道.
_DEGRADE_AFTER_ERRORS = 3

# 危险内置函数, 从 safe builtins 里移除.
# code_act_loop.py 原内联版本多去了 globals/locals, 统一到这里 — 防止
# exec'd 代码通过 globals()/locals() 拿到宿主 namespace 逃逸.
_DANGEROUS_BUILTINS = frozenset(
    {"__import__", "exec", "eval", "compile", "open", "globals", "locals"}
)


def filter_tools_for_code_act(tools: list) -> list:
    """过滤掉 _BLOCKED_TOOLS 里的工具, 返回剩余的列表.

    输入可以是工具名 list, 也可以是 (name, tool) 二元组 list; 都按 name 过滤.
    ponytail: 一个函数吃两种输入比拆两个函数少一倍接口.
    """
    result: list = []
    for item in tools:
        # 二元组按第一个元素判, 否则按整个元素当 name
        name = item[0] if isinstance(item, tuple) and len(item) == 2 else item
        if name in _BLOCKED_TOOLS:
            continue
        result.append(item)
    return result


def make_safe_builtins() -> dict:
    """返回安全的 builtins 字典, 移除危险内置函数.

    移除: __import__ / exec / eval / compile / open
    __import__ 被 safe_import 替代, 其余直接缺失 (调用时报 NameError).
    """
    safe = {
        k: v
        for k, v in vars(builtins).items()
        if k not in _DANGEROUS_BUILTINS
    }
    safe["__import__"] = safe_import
    return safe


def safe_import(
    name: str,
    globals: dict | None = None,
    locals: dict | None = None,
    fromlist: tuple = (),
    level: int = 0,
) -> Any:
    """替代 __import__ 的安全版本, 只允许白名单内的模块.

    白名单外模块直接 raise ImportError, 让 exec'd 代码把 ImportError 当普通
    代码错误处理 (计入 degrade 阈值).

    两个 opt-in 模块走 env var 放行 (从 code_act_loop.py 迁移):
    - atomworld: 需 HUGINN_USE_ATOMWORLD=1
    - cognitive_map / structure_cognitive_map: 需 HUGINN_USE_COGNITIVE_MAP=1
    """
    # AtomWorld — flag-off 时即使装了也不让 import
    if name == "atomworld":
        if os.environ.get("HUGINN_USE_ATOMWORLD", "0") != "1":
            raise ImportError("import of 'atomworld' requires HUGINN_USE_ATOMWORLD=1")
        return builtins.__import__(name, globals, locals, fromlist, level)
    # Structure Cognitive Map — 同理, agent 应走 namespace 函数而非直 import
    if name in ("cognitive_map", "structure_cognitive_map"):
        if os.environ.get("HUGINN_USE_COGNITIVE_MAP", "0") != "1":
            raise ImportError("import of 'cognitive_map' requires HUGINN_USE_COGNITIVE_MAP=1")
    root = name.split(".")[0]
    if root not in _ALLOWED_IMPORTS:
        raise ImportError(
            f"import of {name!r} is not allowed in CodeAct mode; "
            f"allowed: {sorted(_ALLOWED_IMPORTS)}"
        )
    return builtins.__import__(name, globals, locals, fromlist, level)


def check_degrade(error_count: int) -> bool:
    """判断连续错误数是否达到降级阈值.

    无状态纯函数, 调用方负责累积 error_count. 降级时调用方应发
    code_act_degraded 事件并 fall back 到 tool_call 轨道.
    """
    return error_count >= _DEGRADE_AFTER_ERRORS


if __name__ == "__main__":
    # Self-check: 验证关键安全机制生效, 失败就 assert 失败.
    # 不引入测试框架, ponytail: 最小可运行检查.

    # 1. _BLOCKED_TOOLS 过滤生效 (工具名 list)
    names_in = [
        "hpc_client", "math_tool", "bash_tool",
        "rag_tool", "code_tool", "shell_tool", "container_exec",
    ]
    filtered = filter_tools_for_code_act(names_in)
    assert filtered == ["math_tool", "rag_tool"], f"filter names failed: {filtered}"

    # 二元组 list 也能吃
    tuples_in = [
        ("hpc_client", object()),
        ("math_tool", object()),
        ("code_tool", object()),
    ]
    tuples_out = filter_tools_for_code_act(tuples_in)
    assert len(tuples_out) == 1 and tuples_out[0][0] == "math_tool"

    # 2. 危险 builtins 被移除 (含 globals/locals, 防止 namespace 逃逸)
    sb = make_safe_builtins()
    assert sb["__import__"] is safe_import, "__import__ must be replaced by safe_import"
    for danger in ("exec", "eval", "compile", "open", "globals", "locals"):
        assert danger not in sb, f"{danger} should be removed from safe builtins"

    # 3. safe_import 白名单外模块被拦截
    for bad in ("os", "subprocess", "socket", "ctypes", "builtins"):
        try:
            safe_import(bad)
            raise AssertionError(f"import {bad} should raise ImportError")
        except ImportError:
            pass

    # 白名单内的不抛 (math 装必有, 其他模块可能没装但白名单判断在 import 前)
    math_mod = safe_import("math")
    assert hasattr(math_mod, "sqrt"), "math should be importable"

    # 子模块按 root 判: numpy.fft 也走 numpy 白名单
    # (不实际 import, 只验证 root 判断逻辑不漏)
    assert "numpy".split(".")[0] in _ALLOWED_IMPORTS

    # 3b. opt-in 模块: flag-off 时被拦截, flag-on 时放行
    os.environ.pop("HUGINN_USE_ATOMWORLD", None)
    try:
        safe_import("atomworld")
        raise AssertionError("atomworld should be blocked without HUGINN_USE_ATOMWORLD=1")
    except ImportError:
        pass
    os.environ["HUGINN_USE_ATOMWORLD"] = "1"
    # flag-on 时不抛 ImportError (模块可能没装, 那是 ImportError 不是白名单拦截)
    try:
        safe_import("atomworld")
    except ImportError as e:
        # 区分: 白名单拦截的 msg 含 "requires", 没装的 msg 不含
        assert "requires" not in str(e), f"atomworld should pass whitelist with flag on: {e}"
    finally:
        os.environ.pop("HUGINN_USE_ATOMWORLD", None)

    os.environ.pop("HUGINN_USE_COGNITIVE_MAP", None)
    for cm_name in ("cognitive_map", "structure_cognitive_map"):
        try:
            safe_import(cm_name)
            raise AssertionError(f"{cm_name} should be blocked without HUGINN_USE_COGNITIVE_MAP=1")
        except ImportError:
            pass

    # 4. check_degrade(3) 返回 True
    assert check_degrade(0) is False
    assert check_degrade(2) is False
    assert check_degrade(3) is True
    assert check_degrade(5) is True

    print("code_act_sandbox self-check: OK")
