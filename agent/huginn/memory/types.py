"""类型化记忆分类 — 借鉴 Claude Code 的 memdir 设计。

每条记忆都带有一个语义类型，便于在写入时按主题归档、读取时按类型过滤，
也让索引文件能按类型分组展示。
"""

from __future__ import annotations

from enum import Enum


class MemoryType(str, Enum):
    """记忆的语义类型。

    继承 str 让枚举值可以直接序列化为 JSON / 文件路径片段，
    不需要额外调用 ``.value``。
    """

    USER = "user"  # 用户偏好、习惯
    FEEDBACK = "feedback"  # 用户反馈、纠正
    PROJECT = "project"  # 项目状态、决策
    REFERENCE = "reference"  # 技术参考、文档
    CALCULATION = "calculation"  # 材料计算结果（材料科学专用）


# 写入提示词：在向 LLM 解释某种类型应该存什么时使用
TYPE_PROMPTS: dict[MemoryType, str] = {
    MemoryType.USER: "Store user preferences, working style, and recurring requests.",
    MemoryType.FEEDBACK: "Store corrections and feedback the user gave about your work.",
    MemoryType.PROJECT: "Store project state, key decisions, and current goals.",
    MemoryType.REFERENCE: "Store technical references, formulas, and best practices.",
    MemoryType.CALCULATION: "Store calculation parameters, convergence results, and material properties.",
}


__all__ = ["MemoryType", "TYPE_PROMPTS"]
