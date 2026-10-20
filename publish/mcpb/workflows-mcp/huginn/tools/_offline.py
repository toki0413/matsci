"""离线开关: `HUGINN_DISABLE_WEB_SEARCH` 的唯一解析点.

为什么需要: 需要在"离线 / CI"下短路真实联网的有三处 —— `web_search_tool`、
`agentic_search_tool`、literature 层 (`literature/_http.py`)。此前每一处都自行
解析同一个环境变量的真值集合, 任一处漂移 (例如只在一处补上 `"y"`) 就会造成
"某个工具已离线、另一个仍在联网" 的不一致 —— 正是 literature 曾因守卫只覆盖
部分入口而静默联网、拖垮 CI 的同类根因。把解析收在这里, 让真值契约只有一个来源。

约定 (与既有行为逐字一致): 仅 `1` / `true` / `yes` / `on` (大小写不敏感) 视为开启。
"""

from __future__ import annotations

import os

# 真值集合: 唯一来源. 改这里即全局生效.
_TRUTHY = ("1", "true", "yes", "on")


def web_search_disabled() -> bool:
    """`HUGINN_DISABLE_WEB_SEARCH` 为真时返回 True (离线 / CI 短路真实联网)."""
    return os.environ.get("HUGINN_DISABLE_WEB_SEARCH", "").lower() in _TRUTHY
