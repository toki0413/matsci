"""Intuition capture — preserve vague research signals without judgment.

研究人员最初的直觉、跨领域类比、未成型的技术偏好, 往往不符合"fact"或
"insight"的严格标准, 但对后续 hypothesis 演化至关重要. 这些信号如果不
主动捕捉, 会在对话中被冲掉; 如果走标准 importance 评分, 又会因为"证据
不足"被 decay 掉.

设计 (借鉴 Open WebUI 路径化记忆 + 用户 profile "without judgment or filtering"):
- detect_intuition(message): 轻量关键词识别, 不做语义判断
- capture(): 存 longterm, tier=long (永久), path=sessions/{id}/intuitions
- recall_intuitions(): 按 path 拉回, 给 hypothesis 阶段做 hint
- 不打分, 不 decay, 不过滤 — 原样保留, 让用户后续自己取舍
"""
from __future__ import annotations

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# 模糊意图信号词 (中英双语). 命中任一就触发捕捉.
# ponytail: 关键词会有误报, 但宁可多存也不漏 — 反正 tier=long 不 decay,
# recall 时按 path 过滤, 噪声可控. 升级: 用 LLM 做意图分类.
_INTUITION_SIGNALS = re.compile(
    r"(?:这就像|类似于|好像|感觉是|直觉是|猜测|不妨|或许|灵感|"
    r"it's like|similar to|feels like|intuition|analogy|"
    r"what if|suppose|imagine|wonder if|reminds me of)",
    re.IGNORECASE,
)

# 跨领域类比信号: "X 的 Y 就像 Z 的 W" 句式
_CROSS_DOMAIN = re.compile(
    r"(.+?)的(.+?)(?:就像|类似于|好比)(.+?)的(.+)",
)


def detect_intuition(message: str) -> dict[str, Any] | None:
    """从用户消息里提取模糊意图信号. 无信号返回 None.

    返回 dict:
      - kind: "intuition" | "cross_domain_analogy" | "vague_preference"
      - signal: 匹配到的信号词
      - analogy: (跨领域类比时) 提取的 (src_domain, src_aspect, dst_domain, dst_aspect)
      - raw: 原始消息片段 (信号词前后 100 字符上下文)
    """
    if not message or len(message) < 5:
        return None

    m = _CROSS_DOMAIN.search(message)
    if m:
        return {
            "kind": "cross_domain_analogy",
            "signal": m.group(0),
            "analogy": {
                "src_domain": m.group(1).strip(),
                "src_aspect": m.group(2).strip(),
                "dst_domain": m.group(3).strip(),
                "dst_aspect": m.group(4).strip(),
            },
            "raw": _excerpt(message, m.start()),
        }

    m = _INTUITION_SIGNALS.search(message)
    if m:
        kind = "intuition"
        sig = m.group(0).lower()
        if any(k in sig for k in ("偏好", "preference", "倾向")):
            kind = "vague_preference"
        return {
            "kind": kind,
            "signal": m.group(0),
            "raw": _excerpt(message, m.start()),
        }

    return None


def _excerpt(message: str, pos: int, window: int = 100) -> str:
    """信号词前后 window 字符的上下文."""
    start = max(0, pos - window)
    end = min(len(message), pos + window)
    snippet = message[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(message):
        snippet = snippet + "..."
    return snippet


def capture(
    memory_manager: Any,
    message: str,
    session_id: str,
) -> str | None:
    """检测并捕捉模糊意图. 返回 memory_id, 无信号返回 None.

    存为 tier=long (永久, 不 decay), path=sessions/{session_id}/intuitions.
    category="intuition" 与标准 fact/insight 区分, recall 时可单独拉.
    """
    signal = detect_intuition(message)
    if signal is None:
        return None

    # 原样保留, 不做摘要/改写 — 用户 profile "without judgment or filtering"
    content = signal["raw"]
    tags = [signal["kind"]]
    if "analogy" in signal:
        tags.append(signal["analogy"]["src_domain"])
        tags.append(signal["analogy"]["dst_domain"])

    try:
        return memory_manager.remember(
            content=content,
            category="intuition",
            tags=tags,
            importance=1.0,  # 最高 importance, 但 tier=long 不 decay
            tier="long",
            path=f"sessions/{session_id}/intuitions",
        )
    except Exception:
        logger.debug("intuition capture failed", exc_info=True)
        return None


def recall_intuitions(
    memory_manager: Any,
    session_id: str | None = None,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """拉回直觉信号. 不传 session_id 拉所有会话的.

    给 hypothesis 阶段做 hint: "你之前提到过 X 类似 Y, 要不要往这个方向想?"
    """
    path = f"sessions/{session_id}/intuitions" if session_id else None
    try:
        return memory_manager.recall(
            query="",  # 空查询走 path 过滤
            category="intuition",
            top_k=top_k,
            path=path,
        )
    except Exception:
        logger.debug("intuition recall failed", exc_info=True)
        return []
