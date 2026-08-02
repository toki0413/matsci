"""Step 3 (wire-perception-channel-and-authority): NonAuthoritativeProvenance.

验证子 agent 权限隔离:
1. TeamMember.get_agent() 强制 auto_approve_all=False (不继承主会话高权限)
2. VISION/CRITIC role 按 _ROLE_TOOL_FILTER 限定工具子集
3. approval_callback 从 HuginnConfig 经 build_agent_kwargs 流到子 agent
4. Checkpoint.auto_approve 字段 round-trip + 旧格式兼容
5. resume_from_checkpoint 带 agent 参数时恢复 auto_approve
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from huginn.agents.team import _ROLE_TOOL_FILTER, TeamMember, TeamRole
from huginn.config import HuginnConfig
from huginn.permissions import PermissionConfig
from huginn.runtime.checkpoint import (
    Checkpoint,
    _from_dict,
    load_checkpoint,
    resume_from_checkpoint,
    save_checkpoint,
)


# ── 1. TeamMember 权限降级 ──────────────────────────────────────────


def _make_member(role: TeamRole, config: HuginnConfig) -> TeamMember:
    return TeamMember(
        name=f"test-{role.value}",
        profile_id=role.value,
        role=role,
        model_name="",
        _config=config,
    )


def test_subagent_auto_approve_forced_false(monkeypatch):
    """子 agent 不继承主会话 auto_approve=True 高权限."""
    cfg = HuginnConfig(auto_approve=True)

    captured = {}

    class _StubAgent:
        def __init__(self, **kw):
            captured["kwargs"] = kw
            self._permission_config = PermissionConfig(auto_approve_all=True)

    monkeypatch.setattr(
        "huginn.agent.HuginnAgent.from_config",
        classmethod(lambda cls, config, profile_id="lead", **ov: _StubAgent(**ov)),
    )

    member = _make_member(TeamRole.PLANNER, cfg)
    agent = member.get_agent()
    assert agent._permission_config.auto_approve_all is False, \
        "子 agent auto_approve_all 必须被强制降级为 False"


def test_subagent_preserves_other_perm_fields(monkeypatch):
    """降级只翻 auto_approve_all, 不动 path_rules / rcb_mode."""
    cfg = HuginnConfig()

    class _StubAgent:
        def __init__(self, **kw):
            self._permission_config = PermissionConfig(
                auto_approve_all=True,
                rcb_mode=True,
                path_rules=[("secret.env", "deny")],
            )

    monkeypatch.setattr(
        "huginn.agent.HuginnAgent.from_config",
        classmethod(lambda cls, config, profile_id="lead", **ov: _StubAgent(**ov)),
    )

    member = _make_member(TeamRole.SCIENTIST, cfg)
    agent = member.get_agent()
    assert agent._permission_config.auto_approve_all is False
    assert agent._permission_config.rcb_mode is True, "rcb_mode 不应被降级"
    assert agent._permission_config.path_rules == [("secret.env", "deny")], \
        "path_rules 不应被降级"


# ── 2. Role tool filter ─────────────────────────────────────────────


def test_vision_role_gets_limited_tools(monkeypatch):
    """VISION member 只给 vision_describe / image_analysis_tool / file_read_tool."""
    cfg = HuginnConfig()
    captured = {}

    class _StubAgent:
        def __init__(self, **kw):
            captured["tool_filter"] = kw.get("tool_filter")
            self._permission_config = PermissionConfig()

    monkeypatch.setattr(
        "huginn.agent.HuginnAgent.from_config",
        classmethod(lambda cls, config, profile_id="lead", **ov: _StubAgent(**ov)),
    )

    member = _make_member(TeamRole.VISION, cfg)
    member.get_agent()
    assert captured["tool_filter"] == _ROLE_TOOL_FILTER[TeamRole.VISION]
    assert "bash_tool" not in captured["tool_filter"]
    assert "vasp_tool" not in captured["tool_filter"]
    assert "code_tool" not in captured["tool_filter"]


def test_planner_role_no_filter(monkeypatch):
    """PLANNER role 不限工具 (None = 继承 profile)."""
    cfg = HuginnConfig()
    captured = {}

    class _StubAgent:
        def __init__(self, **kw):
            captured["tool_filter"] = kw.get("tool_filter")
            self._permission_config = PermissionConfig()

    monkeypatch.setattr(
        "huginn.agent.HuginnAgent.from_config",
        classmethod(lambda cls, config, profile_id="lead", **ov: _StubAgent(**ov)),
    )

    member = _make_member(TeamRole.PLANNER, cfg)
    member.get_agent()
    assert captured["tool_filter"] is None, "PLANNER 不应有 tool_filter"


def test_critic_role_gets_readonly_tools(monkeypatch):
    """CRITIC 只给只读工具."""
    cfg = HuginnConfig()
    captured = {}

    class _StubAgent:
        def __init__(self, **kw):
            captured["tool_filter"] = kw.get("tool_filter")
            self._permission_config = PermissionConfig()

    monkeypatch.setattr(
        "huginn.agent.HuginnAgent.from_config",
        classmethod(lambda cls, config, profile_id="lead", **ov: _StubAgent(**ov)),
    )

    member = _make_member(TeamRole.CRITIC, cfg)
    member.get_agent()
    assert "file_write_tool" not in captured["tool_filter"]
    assert "file_read_tool" in captured["tool_filter"]


# ── 3. approval_callback 流转 ───────────────────────────────────────


def test_approval_callback_in_build_agent_kwargs():
    """build_agent_kwargs 输出 approval_callback (Task 3.3 bug fix)."""
    cb = lambda tool_name, args: True  # noqa: E731
    cfg = HuginnConfig(approval_callback=cb)
    kwargs = cfg.build_agent_kwargs()
    assert "approval_callback" in kwargs, "build_agent_kwargs 必须输出 approval_callback"
    assert kwargs["approval_callback"] is cb


def test_approval_callback_none_by_default():
    """默认无 callback, 不阻塞现有行为."""
    cfg = HuginnConfig()
    kwargs = cfg.build_agent_kwargs()
    assert kwargs["approval_callback"] is None


# ── 4. Checkpoint auto_approve round-trip ───────────────────────────


def test_checkpoint_auto_approve_roundtrip():
    """auto_approve 字段 save → load round-trip."""
    ws = Path(tempfile.mkdtemp(prefix="huginn_p4_")) / "ws"
    ws.mkdir()
    try:
        save_checkpoint(
            task_id="t1", step_id=1, phase="execute", workspace=ws,
            context_digest="x", memory_cursor=None,
            target_chain_progress={}, prospective_queue=[],
            auto_approve=True,
        )
        loaded = load_checkpoint("t1", ws, step_id=1)
        assert loaded is not None
        assert loaded.auto_approve is True, "auto_approve=True 应 round-trip"

        save_checkpoint(
            task_id="t2", step_id=1, phase="execute", workspace=ws,
            context_digest="x", memory_cursor=None,
            target_chain_progress={}, prospective_queue=[],
            auto_approve=False,
        )
        loaded2 = load_checkpoint("t2", ws, step_id=1)
        assert loaded2 is not None
        assert loaded2.auto_approve is False, "auto_approve=False 应 round-trip"
    finally:
        shutil.rmtree(ws.parent, ignore_errors=True)


def test_checkpoint_auto_approve_default_none():
    """不传 auto_approve 时默认 None."""
    ws = Path(tempfile.mkdtemp(prefix="huginn_p4_")) / "ws"
    ws.mkdir()
    try:
        save_checkpoint(
            task_id="t1", step_id=1, phase="execute", workspace=ws,
            context_digest="x", memory_cursor=None,
            target_chain_progress={}, prospective_queue=[],
        )
        loaded = load_checkpoint("t1", ws, step_id=1)
        assert loaded is not None
        assert loaded.auto_approve is None
    finally:
        shutil.rmtree(ws.parent, ignore_errors=True)


# ── 5. 旧格式 checkpoint 兼容 ───────────────────────────────────────


def test_old_checkpoint_without_auto_approve_loads():
    """旧格式 checkpoint (无 auto_approve 字段) 加载不报错, 默认 None."""
    ws = Path(tempfile.mkdtemp(prefix="huginn_p4_")) / "ws"
    ws.mkdir()
    try:
        cp_path = ws / ".huginn" / "checkpoints" / "old_task" / "step_1.json"
        cp_path.parent.mkdir(parents=True, exist_ok=True)
        old_data = {
            "task_id": "old_task",
            "step_id": 1,
            "phase": "execute",
            "context_digest": "abc",
            "memory_cursor": None,
            "target_chain_progress": {},
            "prospective_queue": [],
            "audit_hash_head": "0" * 64,
            "saved_at": 1234567890.0,
            # 注意: 没有 auto_approve 字段 (旧格式)
        }
        cp_path.write_text(json.dumps(old_data), encoding="utf-8")

        loaded = load_checkpoint("old_task", ws, step_id=1)
        assert loaded is not None
        assert loaded.auto_approve is None, "旧格式 checkpoint auto_approve 应为 None"
    finally:
        shutil.rmtree(ws.parent, ignore_errors=True)


def test_resume_does_not_override_when_auto_approve_none():
    """旧格式 checkpoint (auto_approve=None) resume 时不覆盖 agent 当前配置."""
    ws = Path(tempfile.mkdtemp(prefix="huginn_p4_")) / "ws"
    ws.mkdir()
    try:
        save_checkpoint(
            task_id="t1", step_id=1, phase="execute", workspace=ws,
            context_digest="x", memory_cursor=None,
            target_chain_progress={}, prospective_queue=[],
            # auto_approve 不传 → None (模拟旧格式)
        )
        cp = load_checkpoint("t1", ws, step_id=1)
        assert cp is not None

        agent = MagicMock()
        agent._permission_config = PermissionConfig(auto_approve_all=True)
        resume_from_checkpoint(cp, ws, agent=agent)
        assert agent._permission_config.auto_approve_all is True, \
            "旧格式 checkpoint 不应覆盖 agent 当前 auto_approve"
    finally:
        shutil.rmtree(ws.parent, ignore_errors=True)


def test_resume_restores_auto_approve_when_present():
    """新格式 checkpoint (auto_approve=False) resume 时恢复到 agent."""
    ws = Path(tempfile.mkdtemp(prefix="huginn_p4_")) / "ws"
    ws.mkdir()
    try:
        save_checkpoint(
            task_id="t1", step_id=1, phase="execute", workspace=ws,
            context_digest="x", memory_cursor=None,
            target_chain_progress={}, prospective_queue=[],
            auto_approve=False,
        )
        cp = load_checkpoint("t1", ws, step_id=1)
        assert cp is not None

        agent = MagicMock()
        agent._permission_config = PermissionConfig(auto_approve_all=True)
        resume_from_checkpoint(cp, ws, agent=agent)
        assert agent._permission_config.auto_approve_all is False, \
            "新格式 checkpoint auto_approve=False 应恢复到 agent"
    finally:
        shutil.rmtree(ws.parent, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
