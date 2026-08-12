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

import contextlib
import logging
import pathlib
from dataclasses import MISSING, fields
from typing import Any, Callable

logger = logging.getLogger(__name__)

# 当前配置 schema 版本。
# 门禁 (tests/test_arch_config_migration.py) 强制：MIGRATIONS 必须恰好覆盖
# 0..CONFIG_VERSION-1 每一个源版本，且与 HuginnConfig.config_version 一致。
CONFIG_VERSION = 1


def _migrate_v0_to_v1(stored: dict[str, Any]) -> dict[str, Any]:
    """v0 → v1：引入 config_version 字段。旧配置无版本号，语义上视为 v0。"""
    data = dict(stored)
    data.setdefault("config_version", 1)
    return data


# 迁移引擎：每个条目 key = 源版本，value = 把该版本配置升到 (key+1) 的迁移函数。
#
# 新增配置 schema 变更时的标准动作（缺一不可，否则 CI 变红）：
#   1. 把 CONFIG_VERSION 递增；
#   2. 写一个 migrate_vX_to_vY(stored) 函数并注册进 MIGRATIONS；
#   3. 在 tests/test_config_integrity.py 补一个从旧版本升级的迁移测试。
#
# 禁止"只改字段不升版本"——那会让旧配置静默丢失新字段语义。
MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    0: _migrate_v0_to_v1,
}


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
            with contextlib.suppress(Exception):
                # 工厂函数本身报错就跳过, 不影响其余字段
                defaults[f.name] = f.default_factory()
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
        elif (
            fix_types
            and isinstance(default_value, dict)
            and not isinstance(healed[key], dict)
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

    按 MIGRATIONS 注册表，把 config_version 从当前值一路升到 CONFIG_VERSION。

    Args:
        stored: 从磁盘加载的配置字典

    Returns:
        (migrated_config, list_of_migration_notes)
    """
    notes: list[str] = []
    migrated = dict(stored)
    version = int(migrated.get("config_version", 0))

    # 逐级迁移：0 → 1 → ... → CONFIG_VERSION
    while version < CONFIG_VERSION:
        fn = MIGRATIONS.get(version)
        if fn is None:
            # 缺迁移函数：保持原样并记录，不硬崩（由门禁在 CI 拦截缺注册）。
            notes.append(f"no migration registered for v{version}->v{version+1}")
            break
        migrated = fn(migrated)
        version += 1
        migrated["config_version"] = version
        notes.append(f"migrated config from v{version - 1} to v{version}")

    # 迁移完跑一遍完整性校验, 补全缺失键 / 清理孤儿键
    healed, integrity_changes = check_config_integrity(migrated)
    notes.extend(integrity_changes)

    return healed, notes


def save_with_healing(
    config: dict[str, Any],
    path: str | pathlib.Path,
    *,
    format: str = "json",
) -> list[str]:
    """写入前先自愈, 再原子落盘.

    把 check_config_integrity + _atomic_write 串起来:
    先用默认配置作 reference 比对, 补全缺失键 / 删孤儿键 / 修类型,
    然后原子写入. 返回 healing 阶段产生的变更列表.

    Args:
        config: 待写入的配置字典 (会被 healing 修改后的副本覆盖)
        path: 目标文件路径
        format: "json" 或 "toml"

    Returns:
        变更说明列表 (空 = 配置本来就很干净)
    """
    from huginn.config import _atomic_write

    healed, changes = check_config_integrity(config)
    _atomic_write(pathlib.Path(path), healed, format)
    return changes
