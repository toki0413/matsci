"""Tests for the multi-LLM model router."""

from __future__ import annotations

import pytest

from huginn.agent import HuginnAgent
from huginn.models.registry import (
    compact_threshold_for,
    get_model_capabilities,
)
from huginn.models.router import ModelRouter, classify_band


class _FakeModel:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"FakeModel({self.name})"


class TestModelRouter:
    def test_select_by_tag_priority(self):
        router = ModelRouter()
        router.register(
            "cheap", _FakeModel("cheap"), tags={"cheap", "summarize"}, priority=1
        )
        router.register(
            "strong", _FakeModel("strong"), tags={"science", "reasoning"}, priority=5
        )
        router.register("default", _FakeModel("default"), tags={"default"}, priority=0)

        assert router.select("science").name == "strong"
        assert router.select("summarize").name == "cheap"
        assert router.select("unknown").name == "default"

    def test_prefer_cheap(self):
        router = ModelRouter()
        router.register(
            "a", _FakeModel("a"), tags={"default"}, cost_input=10, priority=5
        )
        router.register(
            "b", _FakeModel("b"), tags={"default"}, cost_input=1, priority=1
        )

        assert router.select("default").name == "a"
        assert router.select("default", prefer_cheap=True).name == "b"

    def test_empty_router_raises(self):
        with pytest.raises(RuntimeError):
            ModelRouter().select("default")

    def test_from_env_skips_invalid(self, monkeypatch):
        monkeypatch.setenv("HUGINN_MODEL_CHEAP", "openai:gpt-4o-mini")
        # No OPENAI_API_KEY, so registration should be skipped.
        router = ModelRouter.from_env()
        assert "cheap" not in router.list_models()


class TestBandRouting:
    """双吸引子 band 路由: 外部路由器量化到稳定带 (spec/react), 避开 mixed."""

    def test_classify_band_quantizes_stable_bands_only(self):
        # mixed 永不作为输出 — 相变陷阱应被避让.
        assert classify_band("science") == "spec"
        assert classify_band("reasoning") == "spec"
        assert classify_band("summarize") == "react"
        assert classify_band("planning") == "react"
        assert classify_band(None, signals="hypothesize") == "spec"
        assert classify_band("execute") == "react"

    def test_select_band_prefers_stable_band_model(self):
        r = ModelRouter()
        r.register("spec_a", _FakeModel("spec_a"), tags={"agent"}, bands={"spec"})
        r.register("react_b", _FakeModel("react_b"), tags={"agent"}, bands={"react"})
        assert r.select_band("spec").name == "spec_a"
        assert r.select_band("react").name == "react_b"
        assert r.list_bands() == {"spec": ["spec_a"], "react": ["react_b"]}

    def test_select_band_no_cross_band_fallback(self):
        # 只有 spec 带模型时, react 应落到通用模型, 不回退到 spec (相变区).
        r = ModelRouter()
        r.register("spec_a", _FakeModel("spec_a"), bands={"spec"})
        r.register("generic", _FakeModel("generic"))
        assert r.select_band("react").name == "generic"

    def test_select_band_mixed_is_trap_not_target(self):
        # mixed 是相变陷阱, 不能作为路由目标 — 应走通用回退而非报错.
        r = ModelRouter()
        r.register("generic", _FakeModel("generic"))
        assert r.select_band("mixed").name == "generic"
        assert r.select_band(None).name == "generic"

    def test_register_drops_mixed_band(self):
        r = ModelRouter()
        entry = r.register("m", _FakeModel("m"), bands={"spec", "mixed"})
        assert entry.bands == {"spec"}


class TestHuginnAgentRouter:
    def test_select_model_uses_router(self):
        router = ModelRouter()
        router.register("m", _FakeModel("routed"), tags={"agent"})
        agent = HuginnAgent(model_router=router)
        assert agent.select_model("agent").name == "routed"

    def test_select_model_falls_back_to_single_model(self):
        model = _FakeModel("single")
        agent = HuginnAgent(model=model)
        assert agent.select_model("agent").name == "single"

    def test_no_model_raises(self):
        agent = HuginnAgent()
        with pytest.raises(RuntimeError):
            agent.select_model("agent")


class TestCompactThresholdFor:
    """成本自适应压缩阈值 (G34b): 按模型长上下文成本模式决定主动压缩时机.

    linear(KDA)/hybrid(SALA) 线性注意力骨架便宜 → 推迟压缩, 避免
    '用信息换本不必省的算力'; quadratic(旧全注意力) 维持既往 60%.
    """

    def test_unknown_model_fails_closed_to_60(self):
        assert compact_threshold_for("totally-unknown-model-xyz") == 60

    def test_quadratic_stays_at_legacy_60(self):
        assert compact_threshold_for("gpt-4o") == 60

    def test_kda_linear_pushes_threshold_to_80(self):
        # Kimi K2/K3 走 KDA 线性注意力: 默认 linear → 80
        assert compact_threshold_for("kimi-k2.6") == 75
        assert compact_threshold_for("kimi-k3") == 85

    def test_sala_hybrid_mid_threshold(self):
        # MiniCPM-SALA 稀疏×线性混合: hybrid → 75
        assert compact_threshold_for("minicpm-sala") == 75

    def test_env_override_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("HUGINN_COMPACT_THRESHOLD", "90")
        assert compact_threshold_for("gpt-4o") == 90
        assert compact_threshold_for("kimi-k3") == 90
        monkeypatch.delenv("HUGINN_COMPACT_THRESHOLD")

    def test_invalid_env_ignored(self, monkeypatch):
        monkeypatch.setenv("HUGINN_COMPACT_THRESHOLD", "abc")
        assert compact_threshold_for("kimi-k3") == 85
        monkeypatch.delenv("HUGINN_COMPACT_THRESHOLD")

    def test_minicpm5_short_window_is_quadratic(self):
        caps = get_model_capabilities("minicpm5")
        assert caps.long_context_cost_mode == "quadratic"
        assert caps.context_window == 8192
