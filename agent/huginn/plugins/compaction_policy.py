"""Compaction 保护策略注册表 —— "Everything is a Plugin" 形态 B 在 compaction 的落地.

compaction 是性能敏感路径 (O(n)), 不能用事件分发 (形态 A), 用同步策略注册表
(形态 B): 第三方把"哪些消息受保护 / 哪些块永不裁剪 / 哪些内容 marker 标 root"
声明成 CompactionPolicy 注册进来, compact_messages / summarize_compact_messages
聚合所有策略得出最终保护集, 不再需要改核心模块加保护。

形态说明: 与 prompt 段一致, 用 StrategyRegistry.ordered() 聚合全部策略并做并集
(union), 而非 resolve 单个 — 因为多个插件各自声明保护, 需要叠加而非二选一。
聚合是 O(P) (P = 已注册策略数, 通常个位), 相对 compaction 的 O(n) 可忽略。
"""

from __future__ import annotations

from dataclasses import dataclass

from huginn.plugins.strategy import StrategyRegistry

# 内置默认保护 (与原 context.py 模块级常量一致, 保证默认行为不变).
_DEFAULT_PROTECTED_ROLES = frozenset({"system"})
_DEFAULT_NEVER_TRIM_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})


@dataclass(frozen=True)
class CompactionPolicy:
    """一次 compaction 的保护声明.

    Attributes:
        protected_roles: 这些 role 的消息永不参与摘要/裁剪 (如 system).
        never_trim_block_types: 含这些 content block type 的消息永不裁剪
            (Anthropic thinking/redacted_thinking 带 signature, 裁了后续 400).
        root_content_markers: 内容含这些 marker 的消息永不被 drop (如 checklist
            标题). 与调用方传入的 root_content_markers 取并集.
    """

    protected_roles: frozenset[str] = _DEFAULT_PROTECTED_ROLES
    never_trim_block_types: frozenset[str] = _DEFAULT_NEVER_TRIM_BLOCK_TYPES
    root_content_markers: tuple[str, ...] = ()


_registry: StrategyRegistry[CompactionPolicy] = StrategyRegistry[CompactionPolicy]()


def register_compaction_policy(
    name: str,
    policy: CompactionPolicy,
    priority: int = 0,
) -> None:
    """注册一个 compaction 保护策略.

    同名注册覆盖 (priority 高者 / 同 priority 后注册者生效, 见 resolve 语义).
    各策略之间是并集叠加: 新注册策略补充受保护 role / block type / marker.
    """
    _registry.register(name, policy, priority=priority)


def unregister_compaction_policy(name: str) -> int:
    """移除策略, 返回移除条数."""
    return _registry.unregister(name)


def _aggregate() -> CompactionPolicy:
    protected = set(_DEFAULT_PROTECTED_ROLES)
    never = set(_DEFAULT_NEVER_TRIM_BLOCK_TYPES)
    markers: list[str] = []
    for _name, _priority, policy in _registry.ordered():
        protected |= set(policy.protected_roles)
        never |= set(policy.never_trim_block_types)
        markers.extend(policy.root_content_markers)
    return CompactionPolicy(
        protected_roles=frozenset(protected),
        never_trim_block_types=frozenset(never),
        root_content_markers=tuple(markers),
    )


def protected_roles() -> frozenset[str]:
    """聚合后的受保护 role 集合."""
    return _aggregate().protected_roles


def never_trim_block_types() -> frozenset[str]:
    """聚合后的"永不裁剪"content block type 集合."""
    return _aggregate().never_trim_block_types


def root_content_markers() -> tuple[str, ...]:
    """聚合后的 root content markers (策略贡献部分, 不含调用方传入的)."""
    return _aggregate().root_content_markers


def clear_policies() -> None:
    """清空注册表 (测试用)."""
    _registry.clear()


__all__ = [
    "CompactionPolicy",
    "clear_policies",
    "never_trim_block_types",
    "protected_roles",
    "register_compaction_policy",
    "root_content_markers",
    "unregister_compaction_policy",
]