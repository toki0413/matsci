"""Catalog 统一清单项 —— 只描述"有哪些接入项", 不存执行逻辑.

CatalogEntry 是跨 5 类注册表 (mcp/plugin/skill/tool/prompt/model) 的统一视图:
id 唯一, kind + origin 标注来源, registered_names 记录在某类注册表里实际挂了哪些名字.
执行仍委托既有注册表, 控制面只管"罗列 / 启停 / 追踪".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# 支持的接入种类. 与各注册表的 kind 路由一一对应:
#   mcp    -> MCPClientManager
#   tool   -> ToolRegistry
#   model  -> ModelRegistry
#   skill  -> SkillRegistry
#   prompt -> prompt_segments
#   plugin -> plugins/loader + StarHandlerRegistry
KINDS = ("mcp", "plugin", "skill", "tool", "prompt", "model")

# 来源. discover_all 去重时按该优先级: 值越大越优先保留.
#   builtin   -> lifespan/servers 目录内置 server
#   .mcp.json -> 仓库根 .mcp.json
#   config    -> HuginnConfig.mcp_servers
#   dirs      -> plugins_dir / skills_dir 目录扫描
#   api       -> 运行时 API/CLI 注册
ORIGINS = ("builtin", "dirs", ".mcp.json", "config", "api")
ORIGIN_PRIORITY = {o: i for i, o in enumerate(ORIGINS)}


@dataclass
class CatalogEntry:
    """一条统一接入项."""

    id: str  # f"{kind}:{name}" 唯一键
    kind: str  # 见 KINDS
    name: str
    origin: str  # 见 ORIGINS
    enabled: bool = True
    version: str | None = None
    # 在该 kind 注册表里实际生效的名字 (同一个 server 可能注册到多个注册面).
    registered_names: list[str] = field(default_factory=list)
    # 附加信息 (transport/command/description 等), 不承诺 schema 稳定.
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_entry_id(kind: str, name: str) -> str:
    return f"{kind}:{name}"