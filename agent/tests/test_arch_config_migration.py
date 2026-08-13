"""配置 schema 迁移门禁：版本递增必须配套迁移函数。

规则 (CI fast-fail)：
  R1  MIGRATIONS 必须恰好覆盖 0..CONFIG_VERSION-1 每一个源版本 —— 每个版本升级
      都要有对应的迁移函数，禁止"只改字段不升版本"或"升了版本没写迁移"。
  R2  CONFIG_VERSION 必须与 HuginnConfig.config_version 默认值一致 —— 两处版本号
      是同一 schema 的别名，漂移会在 CI 里立刻拦截。
  R3  从 (CONFIG_VERSION-1) 升上来的迁移函数必须是"实迁移"：把旧配置跑一遍能稳定
      到达 CONFIG_VERSION，且幂等（重复跑不产生额外迁移 note）。

设计：
  - 纯逻辑 + 只读，不落盘。weightless，适合在 CI fast-fail 阶段先跑。
  - MIGRATIONS / CONFIG_VERSION 定义在 huginn/config_integrity.py。
"""
from __future__ import annotations

import sys

import pytest

from huginn.config import HuginnConfig
from huginn.config_integrity import CONFIG_VERSION, MIGRATIONS, migrate_config


def test_migrations_cover_every_version_gap():
    """R1：0..CONFIG_VERSION-1 每个源版本都必须注册迁移函数。"""
    required = set(range(CONFIG_VERSION))  # 0,1,...,CONFIG_VERSION-1
    registered = set(MIGRATIONS.keys())

    missing = required - registered
    assert not missing, (
        "以下配置版本缺迁移函数，请为每个版本写 migrate_vX_to_vY 并注册进 "
        "huginn/config_integrity.py::MIGRATIONS：\n  " + ", ".join(sorted(missing))
    )

    extra = registered - required
    assert not extra, (
        "MIGRATIONS 里注册了超出 CONFIG_VERSION 范围的版本，版本号不连续：\n  "
        + ", ".join(sorted(extra))
    )


def test_config_version_matches_dataclass_default():
    """R2：config_integrity.CONFIG_VERSION 与 HuginnConfig.config_version 一致。"""
    dataclass_default = HuginnConfig.__dataclass_fields__["config_version"].default
    assert dataclass_default == CONFIG_VERSION, (
        f"huginn/config_integrity.py::CONFIG_VERSION={CONFIG_VERSION} "
        f"与 HuginnConfig.config_version 默认值={dataclass_default} 不一致。"
        "两处版本号必须同步（同一 schema 的别名）。"
    )


def test_migration_from_previous_version_is_real_and_idempotent():
    """R3：从 CONFIG_VERSION-1 升级的迁移是"实迁移"且幂等。"""
    if CONFIG_VERSION <= 1:
        # 尚无历史版本可迁移，跳过（v1 是初始版本）。
        pytest.skip("CONFIG_VERSION==1，无历史版本迁移可验证")

    # 造一个上一版本的配置（只带版本号，其余交给完整性校验补全）
    prev = {"config_version": CONFIG_VERSION - 1}

    migrated, notes = migrate_config(prev)
    assert migrated["config_version"] == CONFIG_VERSION, (
        f"从 v{CONFIG_VERSION-1} 迁移后未到达 v{CONFIG_VERSION}，迁移函数没有真正生效"
    )

    # 幂等：已到达当前版本后，再跑一遍不应产生新的迁移 note
    _, notes2 = migrate_config(migrated)
    migration_notes = [n for n in notes2 if n.startswith("migrated config from")]
    assert migration_notes == [], (
        f"迁移不幂等：已到 v{CONFIG_VERSION} 的配置再跑一遍仍产生迁移 note：{migration_notes}"
    )


def test_migration_gate_self_test():
    """门禁自检：确认版本覆盖逻辑能抓到已知错误形态。"""
    assert set(range(1)) == {0}  # 平凡自检，防止 range 语义被改坏
    # 模拟"升了版本但没写迁移"：CONFIG_VERSION=2 时 MIGRATIONS 缺 key=1
    fake_required = set(range(2))
    fake_registered = {0}
    assert (fake_required - fake_registered) == {1}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-o", "addopts="]))
