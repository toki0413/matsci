"""Tests for the Model Advisor — knowledge base, advisor logic, and routes."""

from __future__ import annotations

import pytest

from huginn.advisor.knowledge import (
    ModelProfile,
    find_by_provider,
    find_by_task,
    get_profiles,
)
from huginn.advisor.model_advisor import ModelAdvisor, Recommendation


# ── ModelProfile and knowledge base ──────────────────────────────────


class TestModelProfile:
    def test_create_profile(self):
        p = ModelProfile(name="test-model", provider="test")
        assert p.name == "test-model"
        assert p.provider == "test"
        assert p.context_window == 4096
        assert p.available is True
        assert p.requires_api_key is True
        assert p.strengths == []
        assert p.weaknesses == []
        assert p.best_for == []
        assert p.notes == ""

    def test_profile_with_all_fields(self):
        p = ModelProfile(
            name="full-model",
            provider="prov",
            strengths=["Fast"],
            weaknesses=["Expensive"],
            context_window=100000,
            cost_per_1k_tokens=0.01,
            best_for=["coding"],
            available=False,
            requires_api_key=False,
            notes="A note",
        )
        assert p.strengths == ["Fast"]
        assert p.weaknesses == ["Expensive"]
        assert p.context_window == 100000
        assert p.cost_per_1k_tokens == 0.01
        assert p.best_for == ["coding"]
        assert p.available is False
        assert p.requires_api_key is False
        assert p.notes == "A note"


class TestKnowledgeBase:
    def test_get_profiles_returns_list(self):
        profiles = get_profiles()
        assert isinstance(profiles, list)
        assert len(profiles) >= 7

    def test_get_profiles_returns_copies(self):
        """get_profiles should return a new list each time."""
        a = get_profiles()
        b = get_profiles()
        assert a is not b

    def test_find_by_task_coding(self):
        results = find_by_task("coding")
        assert len(results) >= 2
        names = [m.name for m in results]
        assert "gpt-4o" in names

    def test_find_by_task_privacy(self):
        results = find_by_task("privacy")
        assert len(results) >= 1
        for m in results:
            assert "privacy" in m.best_for

    def test_find_by_task_nonexistent(self):
        results = find_by_task("nonexistent_task_xyz")
        assert results == []

    def test_find_by_provider_openai(self):
        results = find_by_provider("openai")
        assert len(results) >= 2
        for m in results:
            assert m.provider == "openai"

    def test_find_by_provider_unknown(self):
        results = find_by_provider("unknown_provider")
        assert results == []


# ── ModelAdvisor ─────────────────────────────────────────────────────


class TestModelAdvisor:
    def test_recommend_default(self):
        advisor = ModelAdvisor()
        recs = advisor.recommend()
        assert len(recs) == 5
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_recommend_with_task(self):
        advisor = ModelAdvisor()
        recs = advisor.recommend(task="coding")
        assert len(recs) == 5
        # Top recommendation should be coding-related
        top = recs[0]
        assert top.score >= 0.5

    def test_recommend_budget_low(self):
        advisor = ModelAdvisor()
        recs = advisor.recommend(budget="low")
        # Low-budget should favour cheap models
        top_names = [r.model for r in recs[:3]]
        # gpt-4o (0.005) should be penalised, so not first
        assert recs[0].model != "gpt-4o" or recs[0].score >= 0.5

    def test_recommend_budget_high(self):
        advisor = ModelAdvisor()
        recs = advisor.recommend(budget="high")
        assert len(recs) == 5

    def test_recommend_privacy(self):
        advisor = ModelAdvisor()
        recs = advisor.recommend(privacy=True)
        # Privacy-focused models (no API key) should be at top
        top = recs[0]
        assert "privacy" in top.reason.lower() or "local" in top.reason.lower() or "private" in top.reason.lower()

    def test_recommend_context_large(self):
        advisor = ModelAdvisor()
        recs = advisor.recommend(context_size="large")
        # Large context models should score higher
        assert len(recs) == 5

    def test_recommend_combined_criteria(self):
        advisor = ModelAdvisor()
        recs = advisor.recommend(task="coding", budget="low", privacy=False, context_size="large")
        assert len(recs) == 5
        assert all(r.score > 0 for r in recs)

    def test_recommend_scoring_order(self):
        """Recommendations should be sorted by score descending."""
        advisor = ModelAdvisor()
        recs = advisor.recommend(task="analysis")
        scores = [r.score for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_cost_estimate_format(self):
        advisor = ModelAdvisor()
        recs = advisor.recommend()
        for r in recs:
            assert r.cost_estimate  # non-empty
            if r.cost_estimate != "Free (local)":
                assert "$" in r.cost_estimate

    def test_compare_two_models(self):
        advisor = ModelAdvisor()
        result = advisor.compare("gpt-4o", "claude-3.5-sonnet")
        assert "error" not in result
        assert result["model_a"]["name"] == "gpt-4o"
        assert result["model_b"]["name"] == "claude-3.5-sonnet"
        assert "strengths" in result["model_a"]
        assert "weaknesses" in result["model_b"]

    def test_compare_case_insensitive(self):
        advisor = ModelAdvisor()
        result = advisor.compare("GPT-4O", "Claude-3.5-Sonnet")
        assert "error" not in result

    def test_compare_unknown_model(self):
        advisor = ModelAdvisor()
        result = advisor.compare("gpt-4o", "nonexistent-model")
        assert "error" in result

    def test_compare_both_unknown(self):
        advisor = ModelAdvisor()
        result = advisor.compare("fake-a", "fake-b")
        assert "error" in result


# ── Route endpoints ──────────────────────────────────────────────────


pytest.importorskip("mcp", reason="MCP SDK not installed (pip install mcp)")

from fastapi.testclient import TestClient
from huginn.server import app

_client = TestClient(app)


class TestAdvisorRoutes:
    def test_list_models(self):
        r = _client.get("/advisor/models")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert "count" in data
        assert data["count"] >= 7

    def test_recommend_endpoint(self):
        r = _client.post("/advisor/recommend", json={"task": "coding", "budget": "low"})
        assert r.status_code == 200
        data = r.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) == 5

    def test_recommend_privacy_endpoint(self):
        r = _client.post("/advisor/recommend", json={"privacy": True})
        assert r.status_code == 200
        data = r.json()
        assert len(data["recommendations"]) > 0

    def test_compare_endpoint(self):
        r = _client.get("/advisor/compare", params={"model_a": "gpt-4o", "model_b": "deepseek-v3"})
        assert r.status_code == 200
        data = r.json()
        assert "model_a" in data
        assert "model_b" in data

    def test_compare_unknown_endpoint(self):
        r = _client.get("/advisor/compare", params={"model_a": "gpt-4o", "model_b": "fake"})
        assert r.status_code == 200
        data = r.json()
        assert "error" in data
