"""共享标量提取 — 对账/判据门的统一取数语义.

DRY: ``replan_gate`` / ``early_stop_gate`` / ``decision_gate`` / ``program`` 曾各自
复制一份完全相同的 ``_first_scalar`` 实现, 任何语义调整都要同步改四处, 且极易
漂移成"同名字段在四个门里取到不同的值". 收敛到此处后全仓只有一个决策源.
"""
from __future__ import annotations

from typing import Any

# 优先取常见量纲判定名 (温度/分数/数值/实测值), 找不到再深度优先折返.
_PREFERRED_KEYS = ("T_eq_K", "T", "score", "value", "y", "actual", "S_Wm2")


def first_scalar(node: Any) -> float | None:
    """从任意结构里取第一个可量标量; 取不到返回 None (调用方跳过, 不伪造).

    bool 先于 int 判断 —— Python 里 ``True`` 是 int 子类, 当作 1.0 会污染对账.
    """
    if isinstance(node, bool):
        return None
    if isinstance(node, int | float):
        return float(node)
    if isinstance(node, dict):
        for k in _PREFERRED_KEYS:
            if k in node:
                v = first_scalar(node[k])
                if v is not None:
                    return v
        for v in node.values():
            s = first_scalar(v)
            if s is not None:
                return s
    elif isinstance(node, list | tuple):
        for v in node:
            s = first_scalar(v)
            if s is not None:
                return s
    return None
