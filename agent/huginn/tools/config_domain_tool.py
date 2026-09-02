"""通用配置域对话器 — 让 HuginnConfig 的任意配置域都能通过对话读写.

背景: ``ConfigWizardTool`` 只封装了"模型/隐私/feature"三个域的引导流程; 用户要的是
**更广泛意义上的对话改配置** — persona、团队、HPC、MCP、记忆、预算、材料库 key 等
全部 16 组配置域都能在对话里改, 且改完热生效.

本工具不做逐域手写 action (13+ 域会爆炸且难维护), 而是**反射 ``HuginnConfig`` 的
dataclass 字段**自动生成可对话的字段清单, 用三层保护取代手写:

- ``DENY``   : 敏感/不可安全改的标量 (api_key/hpc_password/encryption_password/材料库key),
  set 被拒, get 或列表展示一律脱敏 — 防止泄露与误改.
- ``READ_ONLY``: 复杂结构字段 (models/agents/feature_flags/mcp_servers/queue_map…) 只读,
  v1 不在对话里直接 set.
- ``EDITABLE``: 标量 (bool/int/float/str 含 Literal 枚举) 可读可写, 类型自动校验.

写盘走模块级 ``persist_config_with_hot_reload`` (与 ConfigWizardTool 共用), 因此任何域
的 set 都即刻热生效. 后续可在 set 前接 ``control_safety`` 策略闸门 + 配置回滚.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import os
import typing
from pathlib import Path
from typing import Any, Literal, Union

from pydantic import BaseModel, Field

import huginn.config as config_module
from huginn.config import HuginnConfig
from huginn.core_types import ToolResult, ValidationResult
from huginn.tools.base import HuginnTool
from huginn.tools.config_wizard_tool import persist_config_with_hot_reload

logger = logging.getLogger(__name__)

# 敏感字段: set 一律拒绝, get/列表展示脱敏.
DENY_SET_FIELDS = frozenset(
    {
        "api_key",
        "hpc_password",
        "hpc_key_path",
        "encryption_password",
        "encryption_key_file",
        "mp_api_key",
        "oqmd_api_key",
    }
)

# 需要整体跳过的字段 (不可序列化回调 / 版本号不宜对话改).
_SKIP_FIELDS = frozenset({"approval_callback", "config_version"})

# 字段 → 配置域 分组 (精确匹配优先, 前缀其次, 未命中归 other).
_EXACT_GROUP: dict[str, str] = {
    "provider": "llm",
    "model": "llm",
    "base_url": "llm",
    "thinking": "llm",
    "max_tokens": "llm",
    "models": "llm",
    "agents": "agents",
    "team_mode_enabled": "agents",
    "max_concurrent_subagents": "agents",
    "plan_auto_confirm": "agents",
    "auto_approve": "agents",
    "enable_exploration": "agents",
    "max_parallel_branches": "agents",
    "persona": "agents",
    "rag_enabled": "agents",
    "vasp_executable": "compute",
    "lammps_executable": "compute",
    "execution_backend": "compute",
    "container_runtime": "compute",
    "container_image": "compute",
    "feature_flags": "flags",
    "local_only_mode": "security",
    "allow_local_bash": "security",
    "prompt_cache_control": "budget",
    "max_tool_output_tokens": "budget",
    "context_budget_tokens": "budget",
    "tool_compression_max_tokens": "budget",
    "checkpointer_path": "persistence",
    "telemetry_enabled": "persistence",
    "extreme_dispatch": "advanced",
    "pet_name": "pet",
    "pet_personality": "pet",
    "pet_accessories": "pet",
    "workspace": "workspace",
}
_PREFIX_GROUP: dict[str, str] = {
    "hpc_": "hpc",
    "mcp_": "mcp",
    "encryption_": "security",
    "privacy_": "security",
    "persona_": "agents",
    "kg_": "kg",
    "memory_": "memory",
    "ollama_": "llm",
    "wm_": "advanced",
    "em_": "advanced",
    "pm_": "advanced",
    "phys_": "advanced",
}
_SIMPLE_TYPES = (bool, int, float, str)
_COMPLEX_ORIGINS = (list, dict)


def _assign_group(name: str) -> str:
    if name in _EXACT_GROUP:
        return _EXACT_GROUP[name]
    for prefix, group in _PREFIX_GROUP.items():
        if name.startswith(prefix):
            return group
    return "other"


def _is_simple(ann: Any) -> bool:
    """标量 (含 Literal / Optional[标量]) → 可编辑; list/dict/Callable → 否."""
    origin = typing.get_origin(ann)
    args = typing.get_args(ann)
    if origin is Union and type(None) in args:
        return any(_is_simple(a) for a in args if a is not type(None))
    if origin is Literal:
        return True
    if origin in _COMPLEX_ORIGINS:
        return False
    return not origin and ann in _SIMPLE_TYPES


_TYPE_HINTS = typing.get_type_hints(HuginnConfig)


def _type_name(ann: Any, name: str) -> str:
    del name
    origin = typing.get_origin(ann)
    args = typing.get_args(ann)
    if origin is Literal:
        return "enum:" + "|".join(str(a) for a in args)
    if origin in _COMPLEX_ORIGINS:
        return origin.__name__
    if ann in _SIMPLE_TYPES:
        return ann.__name__
    return "str"


def _field_kind(name: str) -> str:
    """字段可编辑性: "deny" (敏感) / "readonly" (复杂…) / "editable"."""
    if name in DENY_SET_FIELDS:
        return "deny"
    if name in _SKIP_FIELDS:
        return "readonly"
    ann = _TYPE_HINTS.get(name)
    if ann is None or not _is_simple(ann):
        return "readonly"
    return "editable"


# 运行时构建一次字段元数据 (名字/分组/可编辑性/类型名), 供 list/get 复用.
FIELD_META: list[dict[str, Any]] = [
    {
        "name": f.name,
        "group": _assign_group(f.name),
        "kind": _field_kind(f.name),
        "type": _type_name(_TYPE_HINTS.get(f.name), f.name),
    }
    for f in dataclasses.fields(HuginnConfig)
]


def _mask(value: Any) -> Any:
    return "******" if isinstance(value, str) else value


def _resolve_config_path(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    raw = os.environ.get("HUGINN_CONFIG_FILE")
    if raw:
        return Path(raw)
    return Path(os.environ.get("HUGINN_WORKSPACE", ".")) / "huginn.toml"


def _load_config(path: Path) -> HuginnConfig:
    # 优先基于现有缓存配置(继承 models/api_key), 避免与 auth-loss guard 对照时
    # 因 from_env 空 models 而误判"会丢失 key". 浅拷贝即可——我们只改标量字段,
    # models 列表仍共享引用并原样写回.
    cached = config_module._config_cache
    if cached is not None:
        return copy.copy(cached)
    if path.exists():
        try:
            return HuginnConfig.load(path)
        except Exception:
            return HuginnConfig.from_env()
    return HuginnConfig.from_env()


def _coerce(value: Any, ann: Any) -> Any:
    """按字段类型注解把对话里的字符串值转成正确类型, 并校验 Literal 枚举."""
    origin = typing.get_origin(ann)
    args = typing.get_args(ann)
    if origin is Union and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if not non_none:
            return None
        if value is None or str(value).strip().lower() in ("", "none"):
            return None
        return _coerce(value, non_none[0])
    if origin is Literal:
        sv = str(value)
        if sv not in args:
            raise ValueError(f"{sv!r} 不在可选值 {list(args)}")
        return sv
    if ann in (bool, int, float, str):
        if ann is bool:
            if isinstance(value, bool):
                return value
            low = str(value).strip().lower()
            if low in ("true", "1", "yes", "on"):
                return True
            if low in ("false", "0", "no", "off"):
                return False
            raise ValueError(f"非法布尔值: {value!r} (可选 true/false)")
        if ann is int:
            return int(value)
        if ann is float:
            return float(value)
        return str(value)
    return value


class ConfigDomainInput(BaseModel):
    action: Literal["list_fields", "get_field", "set_field"] = Field(
        ...,
        description="list_fields 列出全部可对话字段; get_field 读单个; set_field 改单个",
    )
    field: str | None = Field(
        default=None, description="字段名 (get_field/set_field 用)"
    )
    value: Any | None = Field(default=None, description="字段新值 (set_field 用)")
    config_path: str | None = Field(
        default=None, description="配置文件路径, 默认 HUGINN_CONFIG_FILE 或 huginn.toml"
    )


class ConfigDomainTool(HuginnTool):
    """通用配置域对话器 — 反射 HuginnConfig, 任意标量字段可读可写并热生效."""

    name = "config_domain_tool"
    category = "meta"
    description = (
        "Conversational config editor for any Huginn setting: persona, team, HPC, MCP, "
        "memory, budget, material-db keys, privacy, etc. Actions: list_fields (list all "
        "editable settings grouped by domain), get_field (read one), set_field (change one, "
        "applied immediately). Sensitive fields are masked; complex fields are read-only."
    )
    input_schema = ConfigDomainInput
    read_only = False

    async def validate_input(
        self, args: ConfigDomainInput, context: Any = None
    ) -> ValidationResult:
        if args.action in ("get_field", "set_field") and not args.field:
            return ValidationResult(
                result=False,
                message=f"{args.action} 需要 field 参数",
            )
        if args.action == "set_field" and args.value is None:
            return ValidationResult(result=False, message="set_field 需要 value 参数")
        return ValidationResult(result=True)

    async def call(self, args: dict[str, Any], context: Any = None) -> ToolResult:
        try:
            inp = ConfigDomainInput(**args)
            if inp.action == "list_fields":
                return self._list_fields(inp.config_path)
            if inp.action == "get_field":
                return self._get_field(inp.field or "", inp.config_path)
            if inp.action == "set_field":
                return self._set_field(inp.field or "", inp.value, inp.config_path)
            return ToolResult(
                data=None,
                success=False,
                error=f"未知 action: {inp.action}",
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=str(e))

    # ── list_fields ───────────────────────────────────────────
    def _list_fields(self, config_path: str | None) -> ToolResult:
        cfg = _load_config(_resolve_config_path(config_path))
        by_group: dict[str, list[dict[str, Any]]] = {}
        for meta in FIELD_META:
            name = meta["name"]
            kind = meta["kind"]
            entry = {
                "field": name,
                "type": meta["type"],
                "kind": kind,
            }
            if kind == "editable":
                entry["current"] = _mask(getattr(cfg, name, None))
            # deny 字段只标敏感, 不展示当前值原文
            if kind == "deny":
                entry["current"] = "******"
            by_group.setdefault(meta["group"], []).append(entry)
        domains = {g: by_group[g] for g in sorted(by_group)}
        return ToolResult(
            data={
                "domains": domains,
                "domain_count": len(domains),
                "message": (
                    "任选 editable 字段调用 set_field 修改; deny=敏感只读+脱敏, "
                    "readonly=复杂结构暂不对话改. 修改即热生效, 无需重启."
                ),
            },
            success=True,
        )

    # ── get_field ─────────────────────────────────────────────
    def _get_field(self, field: str, config_path: str | None) -> ToolResult:
        meta = self._find_meta(field)
        cfg = _load_config(_resolve_config_path(config_path))
        value = getattr(cfg, field, None)
        if meta and meta["kind"] == "deny":
            value = "******"
        return ToolResult(
            data={
                "field": field,
                "value": value,
                "kind": meta["kind"] if meta else "unknown",
            },
            success=True,
        )

    # ── set_field ─────────────────────────────────────────────
    def _set_field(self, field: str, value: Any, config_path: str | None) -> ToolResult:
        meta = self._find_meta(field)
        if meta is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"未知配置字段: {field!r}",
            )
        if meta["kind"] != "editable":
            return ToolResult(
                data=None,
                success=False,
                error=(
                    f"字段 {field!r} 不可对话修改 "
                    f"(kind={meta['kind']}, 敏感或复杂只读)"
                ),
            )
        ann = _TYPE_HINTS.get(field)
        try:
            converted = _coerce(value, ann)
        except (TypeError, ValueError) as e:
            return ToolResult(data=None, success=False, error=f"值校验失败: {e}")

        path = _resolve_config_path(config_path)
        cfg = _load_config(path)
        setattr(cfg, field, converted)
        try:
            persist_config_with_hot_reload(cfg, path)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"保存失败: {e}")

        return ToolResult(
            data={
                "field": field,
                "applied": converted,
                "config_path": str(path),
                "message": f"已将 {field} 设为 {converted!r}, 即刻生效.",
            },
            success=True,
        )

    @staticmethod
    def _find_meta(field: str) -> dict[str, Any] | None:
        return next((m for m in FIELD_META if m["name"] == field), None)
