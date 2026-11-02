"""Catalog 控制面 —— 对 5 类注册表提供统一"发现 → 追踪 → 启停"编排.

只做控制面, 不存执行逻辑:
- 发现: 采集各注册表的实时状态, 归一成 CatalogEntry.
- 追踪: 持有去重后的视图, 记录来源 (origin) 与启停状态.
- 启停/卸载: 本骨架只改追踪视图的 enabled/移除; 真正把工具从 agent 工具列表
  摘除/恢复 (HuginnTool unregister + MCP 重连) 是 E 清单第 5 步范畴.

`ponytail:` set_enabled/uninstall 目前只翻转追踪标记, 不触碰底层注册表 —— 是有意
收窄的一期边界, 升级路径就是接 reconcile (把 enabled=False 落到各注册表的 unregister/
禁用上)。这样骨架期不会误改线上注册状态。
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huginn.catalog.models import (
    KINDS,
    ORIGIN_PRIORITY,
    CatalogEntry,
    make_entry_id,
)

logger = logging.getLogger(__name__)

# kind -> 采集该 kind 所需的外部 handle 键 (None = 无需 handle, 从全局注册表读).
# 这张表是 `discover_handles` / `_kind_collectors` / `discover_all` 三者之间的契约:
# 新增 kind 必须显式声明它依赖哪个 handle, 门禁 (tests/test_arch_catalog_handles.py)
# 断言三者的键集双向闭合 —— 否则又会回到「忘了传 handle → 该 kind 结构上永不出现」.
KIND_HANDLE_SPECS: dict[str, str | None] = {
    "mcp": "mcp_manager",
    "model": "model_registry",
    "plugin": "plugins_dir",
    "tool": None,
    "skill": None,
    "prompt": None,
    "workflow": None,
}

# discover_all 接受的 handle 键全集.
_HANDLE_KEYS = frozenset(v for v in KIND_HANDLE_SPECS.values() if v)


def discover_handles(ctx: Any) -> dict[str, Any]:
    """从 ServerContext 派生 discover_all 的 handles (MCP / 模型 / 插件目录).

    只做**读取**: 缺哪项就给 None, 对应 kind 如实缺席, 不伪造条目. 集中在此是为了让
    census / lifespan / REST 路由共用同一份"从上下文取哪些 handle"的知识, 避免各处
    手写一半 handles 导致 discover_all 覆盖式采集时把其它 kind 冲掉.
    """
    handles: dict[str, Any] = {"mcp_manager": getattr(ctx, "mcp_manager", None)}
    factory = getattr(ctx, "agent_factory", None)
    handles["model_registry"] = getattr(factory, "model_registry", None)
    try:
        from huginn.plugins.loader import DEFAULT_PLUGINS_DIR

        handles["plugins_dir"] = DEFAULT_PLUGINS_DIR
    except Exception:  # 非致命: 插件目录不可解析时该 kind 缺席
        logger.debug("plugins_dir unavailable for catalog discovery", exc_info=True)
        handles["plugins_dir"] = None
    return handles


def _collect_tools() -> list[CatalogEntry]:
    """从 ToolRegistry 采集内置工具."""
    from huginn.tools.registry import ToolRegistry

    out: list[CatalogEntry] = []
    for name in ToolRegistry.list_tools():
        out.append(CatalogEntry(
            id=make_entry_id("tool", name),
            kind="tool",
            name=name,
            origin="builtin",
            registered_names=[name],
        ))
    return out


def _collect_models(model_registry: Any | None) -> list[CatalogEntry]:
    """从 ModelRegistry 采集配置的模型"""
    if model_registry is None:
        return []
    out: list[CatalogEntry] = []
    try:
        refs = model_registry.list()
    except Exception:  # pragma: no cover - 注册表未就绪, 跳过
        return []
    for ref in refs:
        alias = ref.alias
        out.append(CatalogEntry(
            id=make_entry_id("model", alias),
            kind="model",
            name=alias,
            origin="config",
            enabled=bool(getattr(ref, "enabled", True)),
            registered_names=[alias],
            meta={"provider": ref.provider, "model": ref.model},
        ))
    return out


def _collect_skills() -> list[CatalogEntry]:
    """从 SkillRegistry 采集已注册 skill."""
    from huginn.skills.registry import SkillRegistry

    out: list[CatalogEntry] = []
    for name in SkillRegistry.list_skills():
        out.append(CatalogEntry(
            id=make_entry_id("skill", name),
            kind="skill",
            name=name,
            origin="dirs",
            registered_names=[name],
        ))
    return out


def _collect_prompts() -> list[CatalogEntry]:
    """从 prompt_segments 采集 prompt 段."""
    from huginn.plugins.prompt_segments import registered_prompt_segments

    out: list[CatalogEntry] = []
    for name in registered_prompt_segments():
        out.append(CatalogEntry(
            id=make_entry_id("prompt", name),
            kind="prompt",
            name=name,
            origin="builtin",
            registered_names=[name],
        ))
    return out


def _collect_mcps(mcp_manager: Any | None) -> list[CatalogEntry]:
    """从 MCPClientManager.list_servers() 采集已注册 MCP server (含来源).

    origin 由各接入源在注册时标注 (builtin/.mcp.json/config/api); 未标注者回退
    runtime. 这也是来源归一的同一数据源, 避免 catalog 另起读出多份.
    """
    if mcp_manager is None:
        return []
    out: list[CatalogEntry] = []
    try:
        servers = mcp_manager.list_servers() or []
    except Exception:  # pragma: no cover - 管理器未就绪, 跳过
        return []
    for item in servers:
        name = item.get("name")
        if not name:
            continue
        origin = item.get("origin") or "runtime"
        out.append(CatalogEntry(
            id=make_entry_id("mcp", name),
            kind="mcp",
            name=name,
            origin=origin,
            enabled=True,
            registered_names=[name],
        ))
    return out


def _collect_plugins(plugins_dir: str | Path | None) -> list[CatalogEntry]:
    """从接入目录 (含 manifest.yaml/metadata.yaml/SKILL.md) 采集插件/skill/tool.

    统一走 `plugins.manifest.discover` 归一, kind 由 manifest.yaml 显式声明或
    按目录内容推断; 这是统一接入路径 (E 清单第 7 步) 的清单侧落地.
    """
    from huginn.plugins.manifest import discover

    if plugins_dir is None:
        return []
    out: list[CatalogEntry] = []
    try:
        specs = discover(plugins_dir)
    except Exception:  # pragma: no cover - 目录不可读, 跳过
        return []
    for spec in specs:
        kind = spec.get("kind") or "plugin"
        name = spec.get("name") or ""
        if not name:
            continue
        if kind not in KINDS:
            kind = "plugin"
        out.append(CatalogEntry(
            id=make_entry_id(kind, name),
            kind=kind,
            name=name,
            origin="dirs",
            version=spec.get("version"),
            registered_names=[name],
            meta=spec,
        ))
    return out


def _collect_workflows() -> list[CatalogEntry]:
    """从 WorkflowRegistry 采集命名工作流 (内置模板 + 已注册脚本)."""
    from huginn.workflows.registry import WorkflowRegistry

    WorkflowRegistry.register_builtin_templates()
    out: list[CatalogEntry] = []
    for it in WorkflowRegistry.manifest():
        name = it.get("name")
        if not name:
            continue
        out.append(CatalogEntry(
            id=make_entry_id("workflow", name),
            kind="workflow",
            name=name,
            origin="builtin",
            registered_names=[name],
            meta={
                "kind": it.get("kind"),
                "category": it.get("category", ""),
                "n_stages": it.get("n_stages", 0),
                "n_subtasks": it.get("n_subtasks", 0),
                "params": it.get("params", []),
            },
        ))
    return out


# 各 kind 的采集函数: kind -> callable(handles, ...) -> list[CatalogEntry].
def _kind_collectors(handles: dict[str, Any]) -> dict[str, list[CatalogEntry]]:
    return {
        "tool": _collect_tools(),
        "model": _collect_models(handles.get("model_registry")),
        "skill": _collect_skills(),
        "plugin": _collect_plugins(handles.get("plugins_dir")),
        "prompt": _collect_prompts(),
        "mcp": _collect_mcps(handles.get("mcp_manager")),
        "workflow": _collect_workflows(),
    }


@dataclass
class CatalogManager:
    """统一清单控制面. 挂 server_context.catalog.

    只编排"发现 → 登记 → 启停追踪 → 快照", 不持有执行逻辑.
    """

    # id -> CatalogEntry, 追踪视图 (discover_all 覆盖, 保留用户启停状态).
    _entries: dict[str, CatalogEntry] = field(default_factory=dict)
    _lock: Any = field(default_factory=threading.Lock)

    def discover_all(self, **handles: Any) -> list[CatalogEntry]:
        """从各注册表实时采集, 去重 (origin 优先级高者胜), 合并进追踪视图.

        返回排序后的最新清单. 已存在同名条目的 enabled 状态被保留, 这样
        set_enabled 标记的"用户禁用"不会在 rediscover 时被冲掉.
        """
        raw: dict[str, CatalogEntry] = {}
        unknown = sorted(set(handles) - _HANDLE_KEYS)
        if unknown:
            # 误传参此前被静默忽略 = 「传错了也不算错」; 显式拒绝让接入方早暴露.
            raise TypeError(
                f"unknown catalog handle(s): {unknown}; "
                f"accepted: {sorted(_HANDLE_KEYS)}"
            )
        for _kind, entries in _kind_collectors(handles).items():
            for e in entries:
                prev = raw.get(e.id)
                if prev is None or ORIGIN_PRIORITY[e.origin] > ORIGIN_PRIORITY[prev.origin]:
                    raw[e.id] = e
        with self._lock:
            merged: dict[str, CatalogEntry] = {}
            for eid, entry in raw.items():
                old = self._entries.get(eid)
                if old is not None:
                    # 覆盖采集字段, 保留用户设置的 enabled.
                    for f in ("version", "meta", "registered_names", "origin"):
                        setattr(entry, f, getattr(old, f))
                    entry.enabled = old.enabled
                merged[eid] = entry
            self._entries = merged
            return sorted(merged.values(), key=lambda e: (e.kind, e.name))

    def list(self) -> list[CatalogEntry]:
        with self._lock:
            return sorted(self._entries.values(), key=lambda e: (e.kind, e.name))

    def get(self, entry_id: str) -> CatalogEntry | None:
        with self._lock:
            return self._entries.get(entry_id)

    def set_enabled(self, entry_id: str, enabled: bool) -> bool:
        """翻转追踪视图的启停标记. 返回是否命中.

        `ponytail:` 一期只改标记, 不动底层注册表; 接 reconcile 后 enabled=False
        才会真正把工具/MCP 从 agent 列表摘除 (见 E 清单第 5 步).
        """
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                return False
            entry.enabled = enabled
            return True

    def uninstall(self, entry_id: str) -> bool:
        """从追踪视图移除一项. 返回是否命中.

        `ponytail:` 一期只从清单移除追踪, 不调用底层注册表 unregister; 底层
        摘除归第 5 步 reconcile.
        """
        with self._lock:
            return self._entries.pop(entry_id, None) is not None

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """全量快照 (可序列化), 供审计 / 恢复."""
        with self._lock:
            return {eid: e.to_dict() for eid, e in self._entries.items()}

    def restore(self, snap: dict[str, dict[str, Any]]) -> None:
        """从快照恢复追踪视图."""
        with self._lock:
            self._entries = {}
            for eid, data in (snap or {}).items():
                try:
                    self._entries[eid] = CatalogEntry(**data)
                except Exception:  # pragma: no cover - 快照格式不兼容, 跳过该项
                    continue
