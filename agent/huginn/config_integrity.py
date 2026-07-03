"""Configuration integrity self-healing — inspired by AstrBot.

AstrBot's AstrBotConfig.check_config_integrity() recursively compares
the on-disk config against DEFAULT_CONFIG, auto-fills missing keys,
removes orphan keys, and fixes type mismatches. This module brings
the same safety to HuginnConfig.

Key difference from AstrBot: HuginnConfig is a dataclass (not a dict),
so we work with dict representations (to_dict/from_dict) and add
a version field for migration tracking.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import MISSING, fields
from typing import Any

logger = logging.getLogger(__name__)

# 当前配置 schema 版本
CONFIG_VERSION = 1


def _get_default_config() -> dict[str, Any]:
    """从 HuginnConfig 的 dataclass 默认值生成参考配置字典.

    遍历所有字段, 取 default 或 default_factory() 的值,
    最后补上 config_version.
    """
    from huginn.config import HuginnConfig

    defaults: dict[str, Any] = {}
    for f in fields(HuginnConfig):
        if f.default is not MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not MISSING:
            try:
                defaults[f.name] = f.default_factory()
            except Exception:
                # 工厂函数本身报错就跳过, 不影响其余字段
                pass
    defaults["config_version"] = CONFIG_VERSION
    return defaults


def check_config_integrity(
    stored: dict[str, Any],
    reference: dict[str, Any] | None = None,
    *,
    remove_orphans: bool = True,
    fix_types: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """递归比对配置完整性, 参考 AstrBotConfig.check_config_integrity().

    Args:
        stored: 从磁盘加载的配置字典
        reference: 参考默认配置; None 时用 HuginnConfig 默认值
        remove_orphans: True 时删除 reference 里不存在的键
        fix_types: True 时修正类型不匹配(用默认值替换)

    Returns:
        (healed_config, list_of_changes)
    """
    if reference is None:
        reference = _get_default_config()

    changes: list[str] = []
    healed = dict(stored)

    # 补全缺失键 / 修正类型
    for key, default_value in reference.items():
        if key not in healed:
            healed[key] = default_value
            changes.append(f"added missing key '{key}'")
        elif fix_types and isinstance(default_value, dict) and not isinstance(
            healed[key], dict
        ):
            healed[key] = default_value
            changes.append(f"fixed type mismatch for '{key}' (expected dict)")
        elif isinstance(default_value, dict) and isinstance(healed[key], dict):
            # 默认值非空才递归; 空字典视为不透明容器(用户自填键),
            # 递归会误删 hpc_queue_map / feature_flags / mcp_servers 等用户数据
            if default_value:
                nested_healed, nested_changes = check_config_integrity(
                    healed[key],
                    default_value,
                    remove_orphans=remove_orphans,
                    fix_types=fix_types,
                )
                if nested_changes:
                    healed[key] = nested_healed
                    for nc in nested_changes:
                        changes.append(f"'{key}.{nc}'")

    # 删除孤儿键
    if remove_orphans:
        orphan_keys = [k for k in healed if k not in reference]
        for k in orphan_keys:
            # 下划线开头的键视为私有扩展, 保留不动
            if not k.startswith("_"):
                del healed[k]
                changes.append(f"removed orphan key '{k}'")

    # 确保版本号存在
    healed["config_version"] = CONFIG_VERSION

    return healed, changes


def migrate_config(stored: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """把旧版本配置迁移到当前版本.

    Args:
        stored: 从磁盘加载的配置字典

    Returns:
        (migrated_config, list_of_migration_notes)
    """
    notes: list[str] = []
    version = stored.get("config_version", 0)

    if version < 1:
        # v0 → v1: 引入 config_version 字段, 确保新增字段存在
        notes.append("migrated config from v0 to v1 (added config_version field)")

    # 未来迁移: if version < 2: ...

    # 迁移完跑一遍完整性校验, 补全缺失键 / 清理孤儿键
    healed, integrity_changes = check_config_integrity(stored)
    notes.extend(integrity_changes)

    return healed, notes


def save_with_healing(
    config_dict: dict[str, Any],
    path: str,
    *,
    format: str = "toml",
) -> list[str]:
    """先自愈再原子写入.

    1. 对 config_dict 跑 check_config_integrity()
    2. 用 _atomic_write (tmp + os.replace) 落盘

    Returns: 自愈过程中产生的变更列表
    """
    healed, changes = check_config_integrity(config_dict)

    from huginn.config import _atomic_write

    _atomic_write(pathlib.Path(path), healed, format=format)

    if changes:
        logger.info("Config self-healed: %s", "; ".join(changes))

    return changes
