"""通用配置域对话器 — 反射 HuginnConfig 任意标量域可对话读写 + 热生效.

覆盖三层保护: editable 可改/热生效, deny(敏感)拒改+脱敏, 复杂只读; 以及
类型 coerce 与 Literal 枚举校验.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import huginn.config as config_module
from huginn.tools.config_domain_tool import ConfigDomainTool


@pytest.fixture(autouse=True)
def _isolated_config_cache(monkeypatch):
    """每个用例前重置全局配置缓存, 避免跨用例触发 save 的 auth-loss guard."""
    monkeypatch.setattr(config_module, "_config_cache", None)
    monkeypatch.setattr(config_module, "_config_cache_path", None)
    yield


# 横跨多个配置域的样例, 证明"任意域"而非仅 LLM 配置:
#   team_mode(agents) / hpc_default_nodes(hpc) / pet_name(pet) / max_parallel_branches(agents)
#   extreme_dispatch(advanced) / wm_summarize(advanced, str) / hpc_scheduler(hpc, Literal)


async def test_list_fields_groups_multi_domain():
    tool = ConfigDomainTool()
    result = await tool.call({"action": "list_fields"}, context=None)
    assert result.success is True
    domains = result.data["domains"]
    # 至少能看到非 LLM 的配置域
    assert any("hpc" in d for d in domains)
    # 找到跨域的可编辑字段
    fields = {e["field"]: e for g in domains.values() for e in g}
    assert fields["team_mode_enabled"]["kind"] == "editable"
    assert fields["hpc_default_nodes"]["kind"] == "editable"
    assert fields["pet_name"]["kind"] == "editable"
    # wm_summarize 是 str 字段(可编辑), 枚举演示用 hpc_scheduler (Literal)
    assert fields["wm_summarize"]["kind"] == "editable"
    assert fields["wm_summarize"]["type"] == "str"
    assert (
        fields["hpc_scheduler"]["kind"] == "editable"
        and "enum" in fields["hpc_scheduler"]["type"]
    )


async def test_set_editable_persists_and_hot_reloads(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.toml"
    fake_ctx = SimpleNamespace(
        agent="x", agent_factory="x", planner_agent="x", orchestrator="x"
    )
    monkeypatch.setattr("huginn.server_core.get_context", lambda: fake_ctx)
    monkeypatch.setattr("huginn.pet.configure_pet", lambda *a, **k: None)

    tool = ConfigDomainTool()
    # 改 HPC 域字段 — 属于"更广泛配置", 不是 LLM provider.
    result = await tool.call(
        {
            "action": "set_field",
            "field": "hpc_default_nodes",
            "value": "4",
            "config_path": str(cfg_path),
        },
        context=None,
    )
    assert result.success is True
    assert result.data["applied"] == 4
    assert cfg_path.exists()
    assert "hpc_default_nodes" in cfg_path.read_text()
    assert fake_ctx.agent is None  # 热重建被触发


async def test_set_pet_and_bool(tmp_path):
    tool = ConfigDomainTool()
    # pet_name (str)
    r1 = await tool.call(
        {
            "action": "set_field",
            "field": "pet_name",
            "value": "raven",
            "config_path": str(tmp_path / "a.toml"),
        },
        context=None,
    )
    assert r1.success is True and r1.data["applied"] == "raven"
    # team_mode_enabled (bool from string)
    r2 = await tool.call(
        {
            "action": "set_field",
            "field": "team_mode_enabled",
            "value": "true",
            "config_path": str(tmp_path / "b.toml"),
        },
        context=None,
    )
    assert r2.success is True and r2.data["applied"] is True


async def test_literal_enum_rejected(tmp_path):
    tool = ConfigDomainTool()
    # hpc_scheduler 是 Literal 枚举; 非法值应被 _coerce 拒绝
    r = await tool.call(
        {
            "action": "set_field",
            "field": "hpc_scheduler",
            "value": "bogus_value",
            "config_path": str(tmp_path / "x.toml"),
        },
        context=None,
    )
    assert r.success is False
    assert "可选值" in r.error or "校验" in r.error


async def test_deny_sensitive_field_refused_and_masked(tmp_path):
    tool = ConfigDomainTool()
    cfg_path = tmp_path / "sec.toml"
    # set 被拒
    r = await tool.call(
        {
            "action": "set_field",
            "field": "hpc_password",
            "value": "secret",
            "config_path": str(cfg_path),
        },
        context=None,
    )
    assert r.success is False
    assert "不可对话修改" in r.error
    # get 脱敏
    g = await tool.call(
        {"action": "get_field", "field": "api_key", "config_path": str(cfg_path)},
        context=None,
    )
    assert g.success is True
    assert g.data["value"] == "******"


async def test_unknown_field_error():
    tool = ConfigDomainTool()
    r = await tool.call(
        {
            "action": "set_field",
            "field": "no_such",
            "value": 1,
            "config_path": "x.toml",
        },
        context=None,
    )
    assert r.success is False
    assert "未知配置字段" in r.error
