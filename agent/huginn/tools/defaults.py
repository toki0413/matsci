"""工具默认值 —— Fail-closed 设计。

参考 Claude Code 的 Tool.ts 思路：所有元数据默认值都按"最不安全"的取向给出，
工具必须显式声明自己是只读 / 非破坏性 / 可并发，否则一律按需要确认处理。
这样可以避免新工具被无意中加进自动批准白名单带来的安全风险。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Awaitable


@dataclass
class ToolMetadata:
    """Fail-closed 默认值 —— 未显式声明即视为不安全。

    Attributes:
        is_read_only: 是否只读（无副作用）。默认 False，工具必须显式声明只读。
        is_destructive: 是否会删除 / 覆盖用户数据。默认 False。
        is_concurrency_safe: 是否可并发执行。默认 False，避免竞态。
        requires_confirmation: 是否需要用户确认。默认 True，新工具默认走 ASK 流程。
        category: 工具分类标签，便于按类别批量配置权限。
    """

    is_read_only: bool = False
    is_destructive: bool = False
    is_concurrency_safe: bool = False
    requires_confirmation: bool = True
    category: str = "general"


def build_tool(definition: dict) -> dict:
    """用 spread 语义合并默认值与工具定义，确保每个工具都有完整接口。

    `definition` 中显式提供的字段会覆盖默认值，从而实现"显式优先"。
    返回的字典至少包含：
      - metadata: ToolMetadata 实例（如 definition 未提供则用默认值）
      - user_facing_name: 返回工具展示名的可调用对象
      - definition 中的其它所有原始字段

    Args:
        definition: 工具定义字典，至少应包含 "name" 字段。

    Returns:
        合并后的工具定义字典。
    """
    defaults: dict[str, Any] = {
        "metadata": ToolMetadata(),
        # 展示名默认回落到内部 name，避免缺字段时报错
        "user_facing_name": lambda: definition.get("name", "unknown"),
    }
    # 注意：definition 在后，因此其字段覆盖默认值
    return {**defaults, **definition}


def merge_metadata(
    base: ToolMetadata | None,
    overrides: dict[str, Any] | None,
) -> ToolMetadata:
    """把 overrides 中的字段合并进 base，生成新的 ToolMetadata。

    工具类常用类属性（read_only / destructive）回填到 metadata，用这个工具
    可以保留 base 的其余字段不被覆盖。
    """
    if base is None:
        base = ToolMetadata()
    if not overrides:
        return replace(base)
    # 只接受 ToolMetadata 已声明的字段，避免脏数据
    valid_keys = {f for f in ToolMetadata.__dataclass_fields__}
    clean = {k: v for k, v in overrides.items() if k in valid_keys}
    return replace(base, **clean)
