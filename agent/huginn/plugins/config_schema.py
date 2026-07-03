"""Schema 驱动的插件配置 —— AstrBot 的 _conf_schema.json 模式。

每个插件目录可放一个 _conf_schema.json (跟 metadata.yaml 并列), 声明可配置
选项的类型 / 默认值 / 取值范围 / UI 提示。框架用它来:
  1. 生成默认配置
  2. 校验用户给的配置
  3. (未来) 自动生成前端配置表单

Schema 格式 (JSON Schema 子集):
{
  "encut": {
    "type": "number",
    "default": 520,
    "description": "Plane-wave cutoff energy (eV)",
    "min": 200,
    "max": 2000
  },
  "xc_functional": {
    "type": "string",
    "default": "PBE",
    "enum": ["PBE", "LDA", "HSE06", "PBEsol"],
    "description": "Exchange-correlation functional"
  }
}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("huginn.plugin_config_schema")

SCHEMA_FILENAME = "_conf_schema.json"


@dataclass
class ConfigField:
    """单个可配置选项。"""
    name: str
    type: str  # "string" | "number" | "boolean" | "integer" | "array" | "object"
    default: Any = None
    description: str = ""
    enum: list[Any] | None = None
    min: float | None = None
    max: float | None = None
    required: bool = False


def load_schema(plugin_dir: str | Path) -> dict[str, ConfigField] | None:
    """从插件目录读 _conf_schema.json。

    没有 schema 文件时返回 None。
    """
    schema_path = Path(plugin_dir) / SCHEMA_FILENAME
    if not schema_path.exists():
        return None

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load schema %s: %s", schema_path, e)
        return None

    fields: dict[str, ConfigField] = {}
    for name, spec in raw.items():
        fields[name] = ConfigField(
            name=name,
            type=spec.get("type", "string"),
            default=spec.get("default"),
            description=spec.get("description", ""),
            enum=spec.get("enum"),
            min=spec.get("min"),
            max=spec.get("max"),
            required=spec.get("required", False),
        )
    return fields


def generate_defaults(schema: dict[str, ConfigField]) -> dict[str, Any]:
    """按 schema 生成默认配置 dict。

    显式声明了 default 就用 default; 否则按类型给一个零值。
    """
    defaults: dict[str, Any] = {}
    for name, field in schema.items():
        if field.default is not None:
            defaults[name] = field.default
        elif field.type == "string":
            defaults[name] = ""
        elif field.type in ("number", "integer"):
            defaults[name] = 0
        elif field.type == "boolean":
            defaults[name] = False
        elif field.type == "array":
            defaults[name] = []
        elif field.type == "object":
            defaults[name] = {}
    return defaults


def validate_config(
    config: dict[str, Any],
    schema: dict[str, ConfigField],
) -> list[str]:
    """拿配置 dict 跟 schema 对一遍。

    返回校验错误信息列表 (空列表 = 合法)。
    """
    errors: list[str] = []

    # 必填字段检查
    for name, field in schema.items():
        if field.required and name not in config:
            errors.append(f"Missing required field: {name}")

    # 类型和约束检查
    for name, value in config.items():
        if name not in schema:
            continue  # 未知字段放行 (向前兼容)

        field = schema[name]

        # 类型检查
        if field.type == "string" and not isinstance(value, str):
            errors.append(f"{name}: expected string, got {type(value).__name__}")
        elif field.type == "number" and not isinstance(value, (int, float)):
            errors.append(f"{name}: expected number, got {type(value).__name__}")
        elif field.type == "integer" and not isinstance(value, int):
            errors.append(f"{name}: expected integer, got {type(value).__name__}")
        elif field.type == "boolean" and not isinstance(value, bool):
            errors.append(f"{name}: expected boolean, got {type(value).__name__}")
        elif field.type == "array" and not isinstance(value, list):
            errors.append(f"{name}: expected array, got {type(value).__name__}")

        # 枚举检查
        if field.enum is not None and value not in field.enum:
            errors.append(f"{name}: value '{value}' not in {field.enum}")

        # 范围检查
        if field.min is not None and isinstance(value, (int, float)) and value < field.min:
            errors.append(f"{name}: value {value} below minimum {field.min}")
        if field.max is not None and isinstance(value, (int, float)) and value > field.max:
            errors.append(f"{name}: value {value} above maximum {field.max}")

    return errors


def merge_defaults(
    user_config: dict[str, Any],
    schema: dict[str, ConfigField],
) -> dict[str, Any]:
    """用户配置跟 schema 默认值合并 (用户值优先)。"""
    defaults = generate_defaults(schema)
    merged = {**defaults, **user_config}
    return merged


__all__ = [
    "ConfigField",
    "SCHEMA_FILENAME",
    "load_schema",
    "generate_defaults",
    "validate_config",
    "merge_defaults",
]
