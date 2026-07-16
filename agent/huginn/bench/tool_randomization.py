"""Tool order randomization for benchmark harness.

Inkling 训练时随机化 tool set 和 schema 减少 agent 对特定 harness 的过拟合.
这里在 benchmark 层做最小版本: 打乱工具列表顺序, 让 agent 不能靠"排在前面的
优先调"取巧, 必须按 description 判断该用哪个工具.

只在 bench harness 层做, 不侵入核心 ToolRegistry. 生产 run() 不受影响.
ponytail: 只做顺序随机化. 升级路径:
  1. description jitter (加随机后缀打破原文记忆) — 需 wrap StructuredTool
  2. 参数名别名 (path↔file_path) — 需改 args_schema 保持调用兼容
  3. 工具子集采样 (每次只暴露 80%) — 会破坏可复现性, 需特殊处理
"""

from __future__ import annotations

import random
from typing import Any


def randomize_tool_order(tools: list[Any], seed: int | None = None) -> list[Any]:
    """Shuffle tool list order. Same tools, different LLM-visible ordering.

    OpenAI/Anthropic function calling 把 tools 数组传给 LLM, LLM 对顺序有
   轻微偏好 (靠前的略容易被选). 打乱后每轮 benchmark 看到的顺序不同,
    强制 agent 按 description 内容判断, 而不是靠位置走捷径.
    """
    rng = random.Random(seed)
    shuffled = list(tools)
    rng.shuffle(shuffled)
    return shuffled
