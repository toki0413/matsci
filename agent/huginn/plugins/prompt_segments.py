"""Prompt 段插件注册表 —— "Everything is a Plugin" 形态 B 在 prompt 体系的落地.

每个 prompt 段 (persona/mode/phase/metacog/tools/thinking/safety) 是一个独立的、
可注册、可替换、可按 priority 排序的段插件。build_prompt 从注册表按 priority
顺序组装 system_prompt, 不再硬编码六段。

形态说明: 用同步策略注册表 (形态 B) 而非事件钩子 (形态 A)。因为 build_prompt
是主路径同步调用, 引入 async 事件分发会把同步链改成异步, 风险高。注册表
register() 是 O(1), assemble 是 O(N) 纯顺序拼接, 零异步开销、零副作用。

段插件签名: fn(mode: str, phase: str, metacog_state: str, system_prompt: str | None) -> str
返回空串表示跳过该段。第三方可注册新段或覆盖内置段 (同 name 后注册者或更高
priority 生效, 见 StrategyRegistry.resolve 语义)。
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from huginn.plugins.strategy import StrategyRegistry

logger = logging.getLogger(__name__)

# 段签名: 输入三元 (mode/phase/metacog) + runtime persona, 返回该段文本 (空串 = 跳过).
PromptSegmentFn = Callable[[str, str, str, str | None], str]

# 内置段优先级 (升序 = 拼接顺序). 与设计文档 §3.2 约定一致:
#   0-99 框架保留基础段 (persona/mode/phase/metacog/tools)
#   100-199 框架保留动态段 (thinking)
#   safety 固定最后.
_PRIORITY = {
    "persona": 10,
    "mode": 20,
    "phase": 30,
    "metacog": 40,
    "tools": 50,
    "multimodal": 55,  # 多模态引导: 紧挨 tools, 先于 writing
    "writing": 60,  # 写作引导: 夹在 tools 与 thinking 之间
    "thinking": 100,
    "safety": 200,
}

# 模块级注册表: 名字 -> 段插件.
_registry: StrategyRegistry[PromptSegmentFn] = StrategyRegistry[PromptSegmentFn]()


def register_prompt_segment(
    name: str,
    fn: PromptSegmentFn,
    priority: int | None = None,
) -> None:
    """注册一个 prompt 段插件。

    Args:
        name: 段名, 如 "persona" / "tools". 覆盖内置段用同名注册.
        fn: 段渲染函数, 见 PromptSegmentFn 签名.
        priority: 覆盖内置默认优先级; None 时用 _PRIORITY 表, 未知名默认 0.
    """
    if priority is None:
        priority = _PRIORITY.get(name, 0)
    _registry.register(name, fn, priority=priority)


def unregister_prompt_segment(name: str) -> int:
    """移除段插件, 返回移除条数."""
    return _registry.unregister(name)


def assemble_prompt_segments(
    mode: str,
    phase: str,
    metacog_state: str,
    system_prompt: str | None = None,
) -> str:
    """按 priority 升序依次执行全部段插件, 拼接非空结果为 system_prompt.

    同名段插件去重: priority 高者生效, 同 priority 后注册者生效 (覆盖内置段).
    单个段异常被隔离 (跳过该段), 保证 build_prompt 永不因插件抛异常.
    """
    # 去重: name -> (priority, 升序 idx, fn). 升序列表内同 priority 保持注册顺序,
    # 故 idx 越大 = 越晚注册. 保留 priority 更高者, 同 priority 保留更晚注册者.
    best: dict[str, tuple[int, int, PromptSegmentFn]] = {}
    for idx, (name, priority, fn) in enumerate(_registry.ordered()):
        prev = best.get(name)
        if prev is None or priority > prev[0] or (priority == prev[0] and idx > prev[1]):
            best[name] = (priority, idx, fn)

    parts: list[str] = []
    for name, _priority, fn in sorted(best.values()):
        try:
            text = fn(mode, phase, metacog_state, system_prompt)
        except Exception:
            logger.warning("prompt segment %s failed, skipped", name, exc_info=True)
            continue
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def render_prompt_segment(
    name: str,
    mode: str,
    phase: str,
    metacog_state: str,
    system_prompt: str | None = None,
) -> str:
    """只渲染单个指定段插件, 返回其文本 (未注册/异常返回空串).

    供非 build_prompt 路径 (如 context.py 主路径) 在需要时单独注入某一段,
    保持"同一段逻辑只实现一次、可被插件替换"的单一来源语义。
    """
    fn = _registry.resolve(name)
    if fn is None:
        logger.debug("prompt segment %s not registered, skip", name)
        return ""
    try:
        return fn(mode, phase, metacog_state, system_prompt) or ""
    except Exception:
        logger.warning("prompt segment %s failed, skipped", name, exc_info=True)
        return ""


def clear_registry() -> None:
    """清空注册表 (测试用)."""
    _registry.clear()


def registered_prompt_segments() -> list[str]:
    """返回当前注册的 prompt 段名 (去重后), 供 Catalog 枚举 kind=prompt."""
    seen: set[str] = set()
    for name, _priority, _fn in _registry.ordered():
        seen.add(name)
    return sorted(seen)


__all__ = [
    "PromptSegmentFn",
    "assemble_prompt_segments",
    "clear_registry",
    "register_prompt_segment",
    "registered_prompt_segments",
    "render_prompt_segment",
    "unregister_prompt_segment",
]
