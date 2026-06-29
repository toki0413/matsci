"""互动层 —— 实时反馈 / 中途干预 / 主动提问 / 进度展示。

四个模块都是可选注入: 没接到 agent loop 时零开销, 接进去后通过
SSE 或 HTTP 路由把 agent 内部状态透给前端。
"""

from huginn.interaction.streaming import StreamInterceptor
from huginn.interaction.interrupt import InterruptEvent, InterruptManager
from huginn.interaction.clarification import ClarificationManager
from huginn.interaction.progress import ProgressTracker

__all__ = [
    "StreamInterceptor",
    "InterruptEvent",
    "InterruptManager",
    "ClarificationManager",
    "ProgressTracker",
]
