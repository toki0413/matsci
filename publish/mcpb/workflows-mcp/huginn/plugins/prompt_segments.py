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

# 框架骨架段名 (由 agent.build_prompt 注册). 它们不属于插件贡献: 非 build_prompt
# 的主路径 (autoloop 的 _llm_chat 直接拼 persona system prompt) 只消费**插件段**,
# 需据此排除骨架段, 否则会把 persona/phase 等重复注入.
_FRAMEWORK_SEGMENTS = frozenset({
    "persona", "mode", "phase", "metacog", "tools",
    "multimodal", "writing", "thinking", "safety",
})

# 段生效面声明 (声明面**单一事实来源**): 段名 -> 该段**必须**在哪些生产路径生效.
# 取值 ∈ {"main","s7","autoloop"}:
#   main     = agent 主会话路径 (context._effective_system_prompt)
#   s7       = s7 自修改态 (build_prompt 组装**全部**段 → 每个段都必须含 s7)
#   autoloop = 自主循环路径 (assemble_plugin_prompt_segments 只取**插件段**)
# 门禁 (tests/test_prompt_contract.py) 断言: 每个 registered 段都有声明, 且各路径的
# **生效集合**与声明一致. 未声明 = 孤儿段 (红), 不得静默失效.
# 用**字面量 tuple** (非 frozenset 调用) 以便 weightless 门禁用 ast 直接解析.
_APPLIES_TO: dict[str, tuple[str, ...]] = {
    "persona": ("s7",),
    "mode": ("s7",),
    "phase": ("s7",),
    "metacog": ("s7",),
    "tools": ("s7",),
    # 以下三段此前只在 s7 生效, 主路径静默缺席 (反防御写作 / 多模态引导 / 外部思考):
    # 声明进 main 后由 render_prompt_segments_for_path("main") 真正注入.
    "multimodal": ("main", "s7"),
    "writing": ("main", "s7"),
    "thinking": ("main", "s7"),
    # safety: 主路径的安全规范由 huginn/prompts.py §8 承载, 这里只声明 s7;
    # 两份重复文本由 test_prompt_contract 的**同步门**守住 (不得各自漂移, 对齐 MATH_DEPTH 范式).
    "safety": ("s7",),
}

_APPLICABLE_PATHS = frozenset({"main", "s7", "autoloop"})

# 未声明占位 (空元组): 门禁判为"未声明"→红. 第三方插件段须显式传 applies_to.
_UNDECLARED: tuple[str, ...] = ()

# 运行时生效面: 段名 -> 声明路径 (register 时填充; 内置段默认取 _APPLIES_TO).
_applicability: dict[str, tuple[str, ...]] = {}

# 模块级注册表: 名字 -> 段插件.
_registry: StrategyRegistry[PromptSegmentFn] = StrategyRegistry[PromptSegmentFn]()


def register_prompt_segment(
    name: str,
    fn: PromptSegmentFn,
    priority: int | None = None,
    applies_to: tuple[str, ...] | None = None,
) -> None:
    """注册一个 prompt 段插件。

    Args:
        name: 段名, 如 "persona" / "tools". 覆盖内置段用同名注册.
        fn: 段渲染函数, 见 PromptSegmentFn 签名.
        priority: 覆盖内置默认优先级; None 时用 _PRIORITY 表, 未知名默认 0.
        applies_to: 该段须生效的生产路径 (子集 ∈ {"main","s7","autoloop"}).
            省略时取内置声明表 `_APPLIES_TO` (内置段); 未知段省略 = 未声明
            (`_UNDECLARED`), 会被 `tests/test_prompt_contract.py` 判为孤儿段 (红),
            故第三方插件段必须显式声明 (可见决策).
    """
    if priority is None:
        priority = _PRIORITY.get(name, 0)
    if applies_to is not None:
        _applicability[name] = tuple(applies_to)
    else:
        _applicability.setdefault(name, _APPLIES_TO.get(name, _UNDECLARED))
    _registry.register(name, fn, priority=priority)


def _applicability_of(name: str) -> tuple[str, ...]:
    """段名 -> 生效路径声明 (运行时表优先, 回落内置声明表, 再回落未声明)."""
    return _applicability.get(name, _APPLIES_TO.get(name, _UNDECLARED))


def segment_applicability() -> dict[str, tuple[str, ...]]:
    """返回 registered 段名 -> 生效路径声明 (空元组 = 未声明).

    供 `tests/test_prompt_contract.py` 与治理台账核对"声明面".
    """
    return {name: _applicability_of(name) for name in registered_prompt_segments()}


def unregister_prompt_segment(name: str) -> int:
    """移除段插件, 返回移除条数."""
    return _registry.unregister(name)


def _assemble(
    mode: str,
    phase: str,
    metacog_state: str,
    system_prompt: str | None,
    include: Callable[[str], bool],
) -> str:
    """段组装的单一实现: 去重 + 按 priority 升序渲染 + 拼接非空结果.

    include(name) 为 False 的段名被跳过 (供"只取插件段""只取某路径生效段"的
    调用方过滤). 去重: name -> (priority, 升序 idx, fn). 升序列表内同 priority
    保持注册顺序, 故 idx 越大 = 越晚注册. 保留 priority 更高者, 同 priority
    保留更晚注册者. 单个段异常被隔离 (跳过该段), 组装永不因插件抛异常.
    """
    best: dict[str, tuple[int, int, PromptSegmentFn]] = {}
    for idx, (name, priority, fn) in enumerate(_registry.ordered()):
        if not include(name):
            continue
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
    return _assemble(mode, phase, metacog_state, system_prompt, lambda _name: True)


def assemble_plugin_prompt_segments(
    mode: str,
    phase: str,
    metacog_state: str,
    system_prompt: str | None = None,
) -> str:
    """只组装**插件贡献**的 prompt 段 (排除框架骨架段 _FRAMEWORK_SEGMENTS).

    供绕过 build_prompt 的主路径 (autoloop 的 _llm_chat 直接拼 persona system
    prompt) 消费插件段 —— 否则注册进注册表的插件段在这些路径上永不生效.
    """
    return _assemble(
        mode, phase, metacog_state, system_prompt,
        lambda name: name not in _FRAMEWORK_SEGMENTS,
    )


def render_prompt_segments_for_path(
    path: str,
    mode: str,
    phase: str,
    metacog_state: str,
    system_prompt: str | None = None,
) -> str:
    """按**生效声明**渲染在给定生产路径上应生效的段 (供 agent 主路径消费).

    与 assemble_prompt_segments 的区别: 后者渲染**全部**段 (s7 全量组装); 本函数
    只渲染 `_APPLIES_TO` / `applies_to` 声明里含 `path` 的段 (含对 path 有声明的
    第三方插件段). 修掉"主路径只渲染 thinking"的硬编码漏项 —— writing / multimodal
    / 插件段不再在主路径静默缺席 (孤儿段就此消除). 未知 path 抛 ValueError
    (调用方写错即崩, 不静默).
    """
    if path not in _APPLICABLE_PATHS:
        raise ValueError(
            f"unknown prompt path: {path!r}; expected one of {sorted(_APPLICABLE_PATHS)}"
        )
    return _assemble(
        mode, phase, metacog_state, system_prompt,
        lambda name: path in _applicability_of(name),
    )


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
    _applicability.clear()


def registered_prompt_segments() -> list[str]:
    """返回当前注册的 prompt 段名 (去重后), 供 Catalog 枚举 kind=prompt."""
    seen: set[str] = set()
    for name, _priority, _fn in _registry.ordered():
        seen.add(name)
    return sorted(seen)


__all__ = [
    "PromptSegmentFn",
    "assemble_plugin_prompt_segments",
    "assemble_prompt_segments",
    "clear_registry",
    "register_prompt_segment",
    "registered_prompt_segments",
    "render_prompt_segment",
    "render_prompt_segments_for_path",
    "segment_applicability",
    "unregister_prompt_segment",
]
