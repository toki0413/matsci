"""事件驱动认知纪律守护 (M2).

背景: 常驻系统提示 (``HUGINN_SYSTEM_PROMPT``) 承载了完整认知纪律, 对弱本地模型
是必要补偿; 对顶尖大模型则是"常驻开销 + 认知摩擦" (oh-my-pi 的 harness 结论)。
极简模式/均衡模式把纪律从"常驻"降为"事件驱动" —— 平时不占 token, 只在检测到
模型偏离纪律时才即时注入一条紧凑提醒。

本模块只做"检测 → 生成提醒"的纯函数, 不碰消息持久化 / 循环控制; 注入动作由
streaming.py 在发送前调用 ``inject_discipline_reminder`` 完成。

安全层不在此列: 物理 precheck / 数据完整性 / 不伪造 等红线在除 event 模式外的
prompt 里始终保留 (见 model_tier 的"裁架构不裁安全"原则)。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 事件驱动模式下的即时纪律提醒 (紧凑, 不重复常驻大段).
_EVENT_REMINDERS = {
    "tool_failure": (
        "[Discipline] A tool call failed. Inspect the scene with read/list "
        "tools BEFORE explaining, then try a different path. Do not silently "
        "fall back to memory or fabricate a result."
    ),
}


def discipline_mode() -> str:
    """当前档位的认知纪律模式: "always" / "event".

    读取失败时回退 "always" (最保守, 保留常驻纪律), 保证不因档位读取异常
    而让纪律静默消失。
    """
    try:
        from huginn.plugins.model_tier import get_profile

        return get_profile().cognitive_discipline
    except Exception:
        logger.debug("discipline_mode fallback to always", exc_info=True)
        return "always"


def deviation_kind(last_message: Any) -> str | None:
    """检测最后一条消息是否触发纪律提醒.

    目前只识别工具失败 (ToolMessage content 以 "Error" 开头)。
    返回提醒 kind 或 None (无偏离 / 无法判定)。
    """
    if last_message is None:
        return None
    content = getattr(last_message, "content", None)
    if isinstance(content, str) and content.startswith("Error"):
        return "tool_failure"
    return None


def event_reminder(kind: str) -> str:
    """按 kind 取即时纪律提醒文本; 未知 kind 返回空串."""
    return _EVENT_REMINDERS.get(kind, "")


def inject_discipline_reminder(messages: list[Any]) -> list[Any]:
    """event 模式下, 若对话尾部显示纪律偏离, 追加一条即时提醒.

    - 非 event 模式: 原样返回 (常驻纪律已覆盖, 不额外注入).
    - 检测不到偏离 / 提醒已存在 / 无消息: 原样返回.
    - 否则追加一条 ``HumanMessage`` 提醒 (对聊天模型友好, 与 bench 注入
      fix_prompt/CONTINUE_MSG 的中断式纠正同构).
    纯函数, 不抛异常; 失败即原样返回。
    """
    if discipline_mode() != "event":
        return messages
    if not messages:
        return messages
    try:
        last = messages[-1]
        kind = deviation_kind(last)
        if kind is None:
            return messages
        reminder = event_reminder(kind)
        if not reminder:
            return messages
        # 防重复: 最近 3 条里已有一条提醒就跳过 (注入的消息会持久化进下一轮).
        for m in messages[-3:]:
            _c = getattr(m, "content", None)
            if isinstance(_c, str) and "[Discipline]" in _c:
                return messages
        from langchain_core.messages import HumanMessage

        logger.info("event-driven discipline reminder injected (%s)", kind)
        return [*messages, HumanMessage(content=reminder)]
    except Exception:
        logger.debug("inject_discipline_reminder skipped", exc_info=True)
        return messages


__all__ = [
    "deviation_kind",
    "discipline_mode",
    "event_reminder",
    "inject_discipline_reminder",
]