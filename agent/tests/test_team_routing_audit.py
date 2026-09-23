"""Tests for ModelCaps 路由决策的可审计面.

核心诉求: /team/v2/members 只给"谁承担哪个角色"的结果, 看不出"为什么是这个
模型". 本测试锁定 routing_audit() 的留痕语义 —— 每个角色一条, 记下硬性能力
要求 / 加分项 / 决策类型 / 选中模型 / 候选评估 (含落选原因), 且 trace 必须与
实际阵容一致 (不重新路由).
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest

from huginn.agents.team import ModelTeam, TeamRole
from huginn.config import AgentProfileConfig, HuginnConfig, ModelConfig


def _cfg(
    models: list[tuple[str, str]],
    agents: list[tuple[str, str]],
) -> HuginnConfig:
    """构造一个最小 config: models 是 (alias, model), agents 是 (id, alias)."""
    return HuginnConfig(
        models=[
            ModelConfig(alias=a, provider="default", model=m) for a, m in models
        ],
        agents=[AgentProfileConfig(id=i, model_alias=a) for i, a in agents],
    )


def _row(audit: dict[str, Any], role: str) -> dict[str, Any]:
    for r in audit["roles"]:
        if r["role"] == role:
            return r
    raise AssertionError(f"审计表里没有 {role}: {[r['role'] for r in audit['roles']]}")


def _cand(row: dict[str, Any], profile: str) -> dict[str, Any]:
    for c in row["candidates"]:
        if c["profile"] == profile:
            return c
    raise AssertionError(f"{row['role']} 候选里没有 {profile}")


class TestSingleModelMode:
    def test_all_roles_use_the_only_profile(self):
        cfg = _cfg([("m1", "claude-sonnet-5")], [("lead", "m1")])
        audit = ModelTeam.from_config(cfg).routing_audit()

        assert audit["count"] == len(list(TeamRole))
        assert audit["decision_types"] == ["single_model"]
        for row in audit["roles"]:
            assert row["decision"] == "single_model"
            assert row["chosen_model"] == "claude-sonnet-5"


class TestIdMatch:
    def test_role_named_profile_binds_directly(self):
        cfg = _cfg([("m1", "claude-sonnet-5")], [("planner", "m1"), ("lead", "m1")])
        audit = ModelTeam.from_config(cfg).routing_audit()

        planner = _row(audit, "planner")
        assert planner["decision"] == "id_match"
        assert planner["chosen_profile"] == "planner"
        assert planner["chosen_model"] == "claude-sonnet-5"


class TestCapabilityMatch:
    def test_best_capability_wins_and_loser_records_reason(self):
        cfg = _cfg(
            [("strong", "claude-sonnet-5"), ("codeonly", "deepseek-coder")],
            [("alpha", "strong"), ("beta", "codeonly")],
        )
        audit = ModelTeam.from_config(cfg).routing_audit()

        planner = _row(audit, "planner")
        assert planner["decision"] == "capability_match"
        assert planner["required"] == ["reasoning"]
        assert planner["chosen_profile"] == "alpha"

        # 落选候选: 记下缺失的硬性能力, 不给分
        beta = _cand(planner, "beta")
        assert beta["passed"] is False
        assert beta["missing_required"] == ["reasoning"]
        assert beta["score"] is None

        # 选中候选: 有分, 且能看出它满足硬性要求
        alpha = _cand(planner, "alpha")
        assert alpha["passed"] is True
        assert alpha["score"] > 0

    def test_candidate_caps_snapshot_includes_context_window(self):
        cfg = _cfg(
            [("strong", "claude-sonnet-5"), ("codeonly", "deepseek-coder")],
            [("alpha", "strong"), ("beta", "codeonly")],
        )
        audit = ModelTeam.from_config(cfg).routing_audit()

        alpha = _cand(_row(audit, "planner"), "alpha")
        assert alpha["caps"]["context_window"] == 1_000_000
        assert alpha["caps"]["reasoning"] is True


class TestFallbackAndUnfilled:
    def _cfg_tools_only(self) -> HuginnConfig:
        # 两个都不带 reasoning 的 profile: planner 硬性要求 reasoning → 无人可用
        return _cfg(
            [("c1", "deepseek-coder"), ("c2", "deepseek-coder")],
            [("alpha", "c1"), ("beta", "c2")],
        )

    def test_unfilled_when_nothing_left_to_borrow(self):
        audit = ModelTeam.from_config(self._cfg_tools_only()).routing_audit()

        planner = _row(audit, "planner")
        assert planner["decision"] == "unfilled"
        assert planner["chosen_profile"] is None
        # 候选都因为缺 reasoning 落选
        assert all(c["missing_required"] == ["reasoning"] for c in planner["candidates"])

    def test_fallback_borrows_after_candidates_exhausted(self):
        audit = ModelTeam.from_config(self._cfg_tools_only()).routing_audit()

        executor = _row(audit, "executor")
        assert executor["decision"] == "fallback"
        assert executor["chosen_profile"] is not None
        assert executor["note"]


class TestTraceMatchesRoster:
    def test_every_member_role_has_a_trace_row(self):
        cfg = _cfg(
            [("strong", "claude-sonnet-5"), ("codeonly", "deepseek-coder")],
            [("alpha", "strong"), ("beta", "codeonly")],
        )
        team = ModelTeam.from_config(cfg)
        audit = team.routing_audit()

        traced = {r["role"] for r in audit["roles"]}
        for member in team.list_members():
            assert member["role"] in traced

    def test_chosen_profile_matches_actual_member(self):
        cfg = _cfg(
            [("strong", "claude-sonnet-5"), ("codeonly", "deepseek-coder")],
            [("alpha", "strong"), ("beta", "codeonly")],
        )
        team = ModelTeam.from_config(cfg)
        audit = team.routing_audit()

        actual = {m["role"]: m for m in team.list_members()}
        for row in audit["roles"]:
            if row["chosen_profile"] is None:
                continue
            assert actual[row["role"]]["profile"] == row["chosen_profile"]

    def test_manual_team_has_empty_trace(self):
        """手工构造的 team 没有 trace, 审计面退化为空表而非报错."""
        audit = ModelTeam([]).routing_audit()
        assert audit["count"] == 0
        assert audit["roles"] == []


#: 端点测试需要完整 app (fastapi + mcp). app_client 是 module scope,
#: 会在函数级 fixture 之前构造, 所以只能用 skipif 在 setup 前拦住.
_HAS_APP = all(
    importlib.util.find_spec(m) is not None for m in ("fastapi", "mcp")
)


class TestEndpoint:
    @pytest.mark.skipif(not _HAS_APP, reason="fastapi/mcp not installed")
    def test_routing_endpoint_returns_audit(self, app_client):
        r = app_client.get("/team/v2/routing")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert isinstance(data["roles"], list)
        assert "count" in data
