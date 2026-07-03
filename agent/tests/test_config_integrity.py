"""Tests for AstrBot-inspired config integrity self-healing.

覆盖点:
- check_config_integrity: 缺失键补全 / 孤儿键删除 / 类型修正 / 嵌套递归
- migrate_config: v0 → v1 迁移
- HuginnConfig.check_and_heal: 端到端临时文件自愈
- _get_default_config: 返回所有字段 + config_version
- 下划线前缀孤儿键保留
- 原子写失败不损坏原文件
"""

from __future__ import annotations

import json
import os
from dataclasses import fields as dc_fields

import pytest

from huginn.config import HuginnConfig, _atomic_write
from huginn.config_integrity import (
    CONFIG_VERSION,
    _get_default_config,
    check_config_integrity,
    migrate_config,
    save_with_healing,
)


# ---------------------------------------------------------------------------
# check_config_integrity
# ---------------------------------------------------------------------------
class TestCheckConfigIntegrity:
    def test_missing_key_added(self):
        """reference 里有但 stored 里没有的键会被补上."""
        reference = {"a": 1, "b": 2}
        stored = {"a": 1}
        healed, changes = check_config_integrity(stored, reference)
        assert healed["b"] == 2
        assert any("added missing key 'b'" in c for c in changes)

    def test_orphan_removed(self):
        """stored 里有但 reference 里没有的孤儿键会被删掉."""
        reference = {"a": 1}
        stored = {"a": 1, "extra": 99}
        healed, changes = check_config_integrity(stored, reference)
        assert "extra" not in healed
        assert any("removed orphan key 'extra'" in c for c in changes)

    def test_type_mismatch_fixed(self):
        """默认值是 dict 但 stored 给了 str, 会被替换成默认值."""
        reference = {"section": {"x": 1}}
        stored = {"section": "not a dict"}
        healed, changes = check_config_integrity(stored, reference, fix_types=True)
        assert healed["section"] == {"x": 1}
        assert any("type mismatch" in c for c in changes)

    def test_type_mismatch_not_fixed_when_disabled(self):
        """fix_types=False 时不修类型."""
        reference = {"section": {"x": 1}}
        stored = {"section": "not a dict"}
        healed, changes = check_config_integrity(stored, reference, fix_types=False)
        assert healed["section"] == "not a dict"
        assert not any("type mismatch" in c for c in changes)

    def test_nested_dict_recursive(self):
        """嵌套字典会递归补全缺失键 + 删除孤儿键, 变更带层级前缀."""
        reference = {"outer": {"inner": 1, "required": True}}
        stored = {"outer": {"inner": 1, "stale": "x"}}
        healed, changes = check_config_integrity(stored, reference)
        assert healed["outer"]["required"] is True
        assert "stale" not in healed["outer"]
        assert any("'outer." in c for c in changes)

    def test_empty_default_dict_is_opaque(self):
        """默认值是空 dict 时视为不透明容器, 不动用户填的键.

        防止 hpc_queue_map / feature_flags / mcp_servers 被清空.
        """
        reference = {"queue_map": {}}
        stored = {"queue_map": {"gpu": "gpu_queue"}}
        healed, changes = check_config_integrity(stored, reference)
        assert healed["queue_map"] == {"gpu": "gpu_queue"}
        assert not any("queue_map" in c and "removed" in c for c in changes)

    def test_orphan_with_underscore_preserved(self):
        """下划线开头的孤儿键(私有扩展)不会被删除."""
        reference = {"a": 1}
        stored = {"a": 1, "_private_ext": "keep"}
        healed, changes = check_config_integrity(stored, reference)
        assert "_private_ext" in healed
        assert not any("_private_ext" in c for c in changes)

    def test_no_changes_on_clean_config(self):
        """配置和默认值完全一致时不产生变更."""
        reference = {"a": 1, "b": 2}
        stored = {"a": 1, "b": 2}
        healed, changes = check_config_integrity(stored, reference)
        assert changes == []
        assert healed["a"] == 1
        assert healed["b"] == 2
        # config_version 总会被无条件写入
        assert healed["config_version"] == CONFIG_VERSION

    def test_version_always_set(self):
        """无论输入有没有 config_version, 输出一定带当前版本号."""
        reference = {"a": 1}
        stored = {"a": 1}
        healed, _ = check_config_integrity(stored, reference)
        assert healed["config_version"] == CONFIG_VERSION


# ---------------------------------------------------------------------------
# migrate_config
# ---------------------------------------------------------------------------
class TestMigrateConfig:
    def test_v0_to_v1_migration(self):
        """没有 config_version 的旧配置会被迁移到 v1."""
        stored = {"provider": "openai", "model": "gpt4"}
        migrated, notes = migrate_config(stored)
        assert migrated["config_version"] == CONFIG_VERSION
        assert any("v0 to v1" in n for n in notes)

    def test_already_v1_no_migration_note(self):
        """已经是 v1 的配置不产生迁移说明."""
        stored = {"config_version": 1}
        migrated, notes = migrate_config(stored)
        assert migrated["config_version"] == 1
        assert not any("v0 to v1" in n for n in notes)

    def test_migration_fills_missing_keys(self):
        """迁移后会顺带补全缺失字段."""
        stored = {"config_version": 0, "provider": "openai"}
        migrated, notes = migrate_config(stored)
        # 迁移后应该有 HuginnConfig 的字段
        assert "ollama_host" in migrated
        assert any("v0 to v1" in n for n in notes)


# ---------------------------------------------------------------------------
# _get_default_config
# ---------------------------------------------------------------------------
class TestGetDefaultConfig:
    def test_returns_all_fields_plus_version(self):
        """默认字典包含所有 HuginnConfig 字段 + config_version."""
        defaults = _get_default_config()
        for f in dc_fields(HuginnConfig):
            assert f.name in defaults, f"字段 {f.name} 缺失"
        assert defaults["config_version"] == CONFIG_VERSION

    def test_defaults_match_dataclass_defaults(self):
        """关键默认值应与 dataclass 声明一致."""
        defaults = _get_default_config()
        assert defaults["provider"] == "default"
        assert defaults["team_mode_enabled"] is False
        assert defaults["models"] == []
        assert defaults["hpc_queue_map"] == {}
        assert defaults["feature_flags"] == {}
        assert defaults["config_version"] == 1

    def test_no_missing_sentinel_in_defaults(self):
        """工厂字段不应该出现 MISSING 哨兵值."""
        from dataclasses import MISSING

        defaults = _get_default_config()
        for key, val in defaults.items():
            assert val is not MISSING, f"字段 {key} 的值是 MISSING"


# ---------------------------------------------------------------------------
# HuginnConfig.check_and_heal (端到端)
# ---------------------------------------------------------------------------
class TestCheckAndHeal:
    def test_heals_missing_keys_in_temp_file(self, tmp_path):
        """临时配置文件缺失键会被补全并写回."""
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"provider": "openai"}), encoding="utf-8")

        changes = HuginnConfig.check_and_heal(path)

        assert len(changes) > 0
        assert any("config_version" in c for c in changes)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["config_version"] == CONFIG_VERSION
        assert data["provider"] == "openai"

    def test_removes_orphan_keys(self, tmp_path):
        """端到端: 孤儿键会被删除."""
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"provider": "openai", "ghost_key": "boo"}),
            encoding="utf-8",
        )

        changes = HuginnConfig.check_and_heal(path)

        assert any("ghost_key" in c and "removed" in c for c in changes)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "ghost_key" not in data

    def test_no_changes_on_complete_config(self, tmp_path):
        """和默认值一致的配置不产生变更."""
        defaults = _get_default_config()
        path = tmp_path / "config.json"
        path.write_text(json.dumps(defaults), encoding="utf-8")

        changes = HuginnConfig.check_and_heal(path)

        assert changes == []

    def test_nonexistent_path_returns_empty(self, tmp_path):
        """路径不存在时返回空列表."""
        path = tmp_path / "nope.json"
        assert HuginnConfig.check_and_heal(path) == []

    def test_none_path_returns_empty(self):
        """path=None 返回空列表."""
        assert HuginnConfig.check_and_heal(None) == []

    def test_preserves_user_dict_fields(self, tmp_path):
        """自愈不会清空用户填的 hpc_queue_map / feature_flags / mcp_servers."""
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "hpc_queue_map": {"gpu": "gpuq"},
                    "feature_flags": {"experimental": True},
                    "mcp_servers": {"foo": {"command": "bar"}},
                }
            ),
            encoding="utf-8",
        )

        HuginnConfig.check_and_heal(path)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["hpc_queue_map"] == {"gpu": "gpuq"}
        assert data["feature_flags"] == {"experimental": True}
        assert data["mcp_servers"] == {"foo": {"command": "bar"}}

    def test_underscore_orphan_preserved_in_file(self, tmp_path):
        """端到端: 下划线前缀的孤儿键在文件里也保留."""
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"provider": "openai", "_custom_ext": "secret"}),
            encoding="utf-8",
        )

        HuginnConfig.check_and_heal(path)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["_custom_ext"] == "secret"

    def test_toml_file_supported(self, tmp_path):
        """TOML 格式配置也能自愈."""
        try:
            import toml  # noqa: F401
        except ImportError:
            pytest.skip("toml not installed")

        path = tmp_path / "config.toml"
        path.write_text('provider = "openai"\nghost = "x"\n', encoding="utf-8")

        changes = HuginnConfig.check_and_heal(path)

        assert any("ghost" in c and "removed" in c for c in changes)
        import toml as _toml

        data = _toml.loads(path.read_text(encoding="utf-8"))
        assert "ghost" not in data
        assert data["config_version"] == CONFIG_VERSION

    def test_corrupt_file_returns_empty(self, tmp_path):
        """损坏的配置文件(无法解析)返回空列表, 不抛异常."""
        path = tmp_path / "config.json"
        path.write_text("{ this is not valid json", encoding="utf-8")

        changes = HuginnConfig.check_and_heal(path)

        assert changes == []
        # 原文件没被改
        assert path.read_text(encoding="utf-8") == "{ this is not valid json"


# ---------------------------------------------------------------------------
# save_with_healing
# ---------------------------------------------------------------------------
class TestSaveWithHealing:
    def test_heals_before_write(self, tmp_path):
        """写入前先自愈, 孤儿键不落盘."""
        path = tmp_path / "out.json"
        changes = save_with_healing(
            {"provider": "openai", "orphan": 1}, str(path), format="json"
        )
        assert any("orphan" in c and "removed" in c for c in changes)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "orphan" not in data
        assert data["config_version"] == CONFIG_VERSION

    def test_clean_config_no_changes(self, tmp_path):
        """完整配置写入时不产生变更."""
        defaults = _get_default_config()
        path = tmp_path / "out.json"
        changes = save_with_healing(defaults, str(path), format="json")
        assert changes == []


# ---------------------------------------------------------------------------
# 原子写安全性
# ---------------------------------------------------------------------------
class TestAtomicWriteSafety:
    def test_atomic_write_no_corruption_on_success(self, tmp_path):
        """正常原子写入不损坏文件."""
        path = tmp_path / "cfg.json"
        _atomic_write(path, {"a": 1}, format="json")
        assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}

    def test_atomic_write_cleans_tmp_on_failure(self, tmp_path, monkeypatch):
        """os.replace 失败时清理临时文件, 原文件不受影响."""
        path = tmp_path / "cfg.json"
        path.write_text('{"original": true}', encoding="utf-8")

        def boom(src, dst):
            raise OSError("simulated replace failure")

        monkeypatch.setattr(os, "replace", boom)

        with pytest.raises(OSError):
            _atomic_write(path, {"a": 1}, format="json")

        # 原文件应该还在且没被破坏
        assert json.loads(path.read_text(encoding="utf-8")) == {"original": True}
        # 不应该残留临时文件
        tmps = list(tmp_path.glob("*.tmp.*"))
        assert tmps == []
