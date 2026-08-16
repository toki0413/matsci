"""Tests for the model-tier (minimal mode) feature.

Covers the TierProfileStore singleton, per-tier profile defaults, runtime
switching via ``set_tier``, and the peripheral RouteConfig model-tier
endpoints (pure function level, no heavy app boot).
"""

from __future__ import annotations

import os

import pytest

from huginn.plugins.model_tier import (
    ModelTier,
    TierProfileStore,
    get_profile,
    get_store,
    get_tier,
    set_tier,
)


@pytest.fixture(autouse=True)
def _reset_default_tier():
    """每个用例都从 FULL 出发, 避免用例间通过单例相互污染."""
    get_store().set_tier(ModelTier.FULL)


def test_default_tier_from_env():
    os.environ["HUGINN_MODEL_TIER"] = "minimal"
    try:
        store = TierProfileStore()
        assert store.tier is ModelTier.MINIMAL
    finally:
        os.environ.pop("HUGINN_MODEL_TIER", None)


def test_default_tier_invalid_falls_back_to_full():
    os.environ["HUGINN_MODEL_TIER"] = "bogus"
    try:
        store = TierProfileStore()
        assert store.tier is ModelTier.FULL
    finally:
        os.environ.pop("HUGINN_MODEL_TIER", None)


def test_full_profile_keeps_full_orchestration():
    set_tier(ModelTier.FULL)
    p = get_profile()
    assert p.use_phase_machine is True
    assert p.use_plan_gating is True
    assert p.cognitive_discipline == "always"
    assert p.compaction_tier == "heavy"
    assert p.external_thinking is True


def test_balanced_profile_uses_event_driven_discipline():
    set_tier(ModelTier.BALANCED)
    p = get_profile()
    assert p.use_phase_machine is True
    assert p.cognitive_discipline == "event"
    assert p.compaction_tier == "medium"


def test_minimal_profile_strips_phase_and_plan_gating():
    set_tier(ModelTier.MINIMAL)
    p = get_profile()
    assert p.use_phase_machine is False
    assert p.use_plan_gating is False
    assert p.cognitive_discipline == "event"
    assert p.compaction_tier == "light"
    assert p.external_thinking is False


def test_set_tier_switches_runtime_value():
    set_tier(ModelTier.MINIMAL)
    assert get_tier() is ModelTier.MINIMAL
    set_tier(ModelTier.FULL)
    assert get_tier() is ModelTier.FULL


def test_endpoint_helpers_are_registered_on_router():
    """config 路由组应挂载 GET/POST /config/model-tier 两个端点."""
    from huginn.routes.config import router

    paths = [r.path for r in router.routes]
    assert paths.count("/config/model-tier") == 2