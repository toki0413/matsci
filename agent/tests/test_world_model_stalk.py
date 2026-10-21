"""世界模型 stalk 注册表测试：last-value、租户隔离、标签保留."""
from __future__ import annotations

from huginn.research.world_model_stalk import (
    clear_world_model_stalk,
    get_world_model_stalk,
    record_world_model_stalk,
)


def test_record_and_get_stalk():
    clear_world_model_stalk("t1", "s1")
    record = record_world_model_stalk(
        world_model_id="law.exoplanet",
        worldview="physics_causal",
        domain="exoplanet",
        contract_version="v1",
        predicted={"T_eq_K": 200.0},
        observed={"T_eq_K": 210.0},
        prediction_error=0.05,
        epoch=3,
        tenant_id="t1",
        session_id="s1",
    )
    assert record["worldview"] == "physics_causal"
    got = get_world_model_stalk("t1", "s1")
    assert got is not None
    assert got["predicted"]["T_eq_K"] == 200.0
    assert got["epoch"] == 3


def test_tenant_isolation():
    record_world_model_stalk(
        world_model_id="m",
        worldview="physics_causal",
        domain="d",
        contract_version="v1",
        tenant_id="t1",
        session_id="s1",
    )
    assert get_world_model_stalk("t2", "s1") is None
