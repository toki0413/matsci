"""Tests for AgentProfileConfig.system_prompt_override field.

Covers the AstrBot-inspired session-level system prompt override: the field
defaults to None (use persona prompt) and round-trips through HuginnConfig's
to_dict()/from_dict() serialization.
"""

from __future__ import annotations

from dataclasses import fields

from huginn.config import AgentProfileConfig, HuginnConfig


# ---------------------------------------------------------------------------
# Field presence & default
# ---------------------------------------------------------------------------


def test_field_exists_with_default_none():
    """AgentProfileConfig 必须有 system_prompt_override 字段, 默认 None。"""
    field_names = {f.name for f in fields(AgentProfileConfig)}
    assert "system_prompt_override" in field_names

    profile = AgentProfileConfig(id="lead")
    assert profile.system_prompt_override is None


def test_field_accepts_string():
    profile = AgentProfileConfig(id="lead", system_prompt_override="You are a coder.")
    assert profile.system_prompt_override == "You are a coder."


def test_default_does_not_mutate_across_instances():
    """默认 None 是不可变值, 不会出现实例间共享可变默认的坑。"""
    a = AgentProfileConfig(id="lead")
    b = AgentProfileConfig(id="coder")
    a.system_prompt_override = "override-a"
    assert b.system_prompt_override is None


# ---------------------------------------------------------------------------
# to_dict() serialization (HuginnConfig is the serializer for agents)
# ---------------------------------------------------------------------------


def _config_with_agent(**agent_kwargs) -> HuginnConfig:
    agent = AgentProfileConfig(id="lead", **agent_kwargs)
    return HuginnConfig(agents=[agent])


def test_to_dict_includes_system_prompt_override():
    cfg = _config_with_agent(system_prompt_override="custom system prompt")
    data = cfg.to_dict(mask_key=False)

    agent_dicts = data["agents"]
    assert len(agent_dicts) == 1
    assert agent_dicts[0]["system_prompt_override"] == "custom system prompt"


def test_to_dict_serializes_none_default():
    cfg = _config_with_agent()
    data = cfg.to_dict(mask_key=False)

    assert data["agents"][0]["system_prompt_override"] is None


# ---------------------------------------------------------------------------
# Round-trip: to_dict() -> from_dict()
# ---------------------------------------------------------------------------


def test_round_trip_preserves_override_string():
    original = _config_with_agent(system_prompt_override="You are a VASP expert.")
    data = original.to_dict(mask_key=False)

    restored = HuginnConfig.from_dict(data)
    assert len(restored.agents) == 1
    assert restored.agents[0].system_prompt_override == "You are a VASP expert."


def test_round_trip_preserves_none_default():
    original = _config_with_agent()
    data = original.to_dict(mask_key=False)

    restored = HuginnConfig.from_dict(data)
    assert restored.agents[0].system_prompt_override is None


def test_from_dict_loads_override_from_raw_dict():
    """直接喂一个带 system_prompt_override 的 agents 列表, from_dict 要能还原。"""
    raw = {
        "agents": [
            {
                "id": "lead",
                "name": "Lead",
                "model_alias": "default",
                "persona": "coder",
                "tools": ["bash"],
                "enabled": True,
                "max_steps": 8,
                "system_prompt_override": "override-from-raw",
            }
        ]
    }
    cfg = HuginnConfig.from_dict(raw)
    assert cfg.agents[0].system_prompt_override == "override-from-raw"
    # 其他字段也要正常带上
    assert cfg.agents[0].persona == "coder"
    assert cfg.agents[0].max_steps == 8


def test_from_dict_legacy_dict_without_override_defaults_none():
    """老配置文件里没有这个键, 加载后应该退回默认 None, 不报错。"""
    raw = {
        "agents": [
            {
                "id": "lead",
                "name": "Lead",
                "model_alias": "default",
            }
        ]
    }
    cfg = HuginnConfig.from_dict(raw)
    assert cfg.agents[0].system_prompt_override is None


def test_round_trip_multiple_agents_with_mixed_overrides():
    original = HuginnConfig(
        agents=[
            AgentProfileConfig(id="lead", system_prompt_override="lead prompt"),
            AgentProfileConfig(id="coder"),  # None
            AgentProfileConfig(id="reviewer", system_prompt_override="reviewer prompt"),
        ]
    )
    data = original.to_dict(mask_key=False)
    restored = HuginnConfig.from_dict(data)

    assert [a.id for a in restored.agents] == ["lead", "coder", "reviewer"]
    assert restored.agents[0].system_prompt_override == "lead prompt"
    assert restored.agents[1].system_prompt_override is None
    assert restored.agents[2].system_prompt_override == "reviewer prompt"
