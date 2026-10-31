"""共享的 "第一个平衡 JSON 对象" 提取.

DRY: ``metacog.self_model`` 与 ``metacog.imagination`` 曾各复制一份逐字相同的
``_parse_first_json``. self_model 的注释称复制是为了规避循环依赖, 但仓库的严格
import 图里并不存在该环 —— 复制只会让容错语义多出一个漂移点.
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def parse_first_json(text: str) -> dict | None:
    """从 text 抓第一个平衡的 ``{...}`` JSON 对象 (支持嵌套). 失败返回 None."""
    if not text:
        return None
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    logger.debug("best-effort op failed", exc_info=True)
                    return None
    return None
