"""Huginn plugin package for third-party / external integrations.

两套东西并存:
  - 老的 AutoresearchTool / science-skills bridge (外部 skill 集成)
  - 新的 AstrBot 风格插件框架 (registry / event_bus / loader / permissions)
后续老代码逐步迁到新框架, 现在不删不动。
"""

from __future__ import annotations

__all__ = [
    "AutoresearchTool",
    # 新插件框架
    "StarHandlerRegistry",
    "EventBus",
    "PluginLoader",
    "PluginMetadata",
    "PermissionChecker",
    "PluginPermission",
]

from huginn.plugins.autoresearch import AutoresearchTool
from huginn.plugins.event_bus import EventBus
from huginn.plugins.loader import PluginLoader
from huginn.plugins.metadata import PluginMetadata
from huginn.plugins.permissions import PermissionChecker, PluginPermission
from huginn.plugins.registry import StarHandlerRegistry
