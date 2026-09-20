"""Fusion 能力维度 —— 无 torch 依赖部分.

覆盖:
  - DIMENSIONS 含 fusion
  - fusion.fused_kv 内置探针注册 + 默认如实报告 available=False (诚实边界)
  - 外部提供 FusionChannel 实现覆盖挂载 (同名注册), 可查询
  - @capability("fusion", ...) 装饰器维度合法

torch 相关 (C2CProjector 参考实现) 在 test_c2c_projector.py, 无 torch 时 skip.
"""
from __future__ import annotations

import pytest

from huginn.capabilities.capability import DIMENSIONS, capability
from huginn.capabilities.registry import CapabilityMountRegistry


@pytest.fixture(autouse=True)
def _clean_caps():
    snap = dict(CapabilityMountRegistry._caps)  # type: ignore[attr-defined]
    CapabilityMountRegistry.clear()
    yield
    CapabilityMountRegistry._caps = snap  # type: ignore[attr-defined]


def test_fusion_dimension_registered_in_dims():
    assert "fusion" in DIMENSIONS


def test_fusion_probe_registered_and_frankly_unavailable():
    from huginn.capabilities.fusion import register_fusion_capabilities
    from huginn.capabilities.registry import get_shared_capability_registry

    register_fusion_capabilities()
    reg = get_shared_capability_registry()

    cap = reg.get("fusion", "fused_kv")
    assert cap is not None
    assert cap.dimension == "fusion"
    result = cap.impl({})
    # 诚实边界: 默认链路拿不到 KV-Cache, 必须 available=False, 不能假装支持
    assert result.available is False
    assert result.reason


def test_fusion_channel_override_registration():
    """外部拥有本地模型 + projector 的运行环境可按同名覆盖探针."""
    from huginn.capabilities.capability import CapabilityMetadata
    from huginn.capabilities.fusion import (
        FusionChannel,
        FusionStatus,
        register_fusion_capabilities,
    )
    from huginn.capabilities.registry import get_shared_capability_registry

    register_fusion_capabilities()

    class _LocalChannel(FusionChannel):
        def probe(self):
            return FusionStatus(available=True, reason="local models owned", source="local")

        def fuse(self, source_kv, target_kv, **kwargs):
            return target_kv  # 参考: 原样透传 target

    reg = get_shared_capability_registry()
    reg.register(CapabilityMetadata(dimension="fusion", name="fused_kv", impl=_LocalChannel, plugin_name="local"))

    cap = reg.get("fusion", "fused_kv")
    assert cap.impl().probe().available is True
    assert cap.impl().probe().source == "local"


def test_fusion_capability_decorator_accepted():
    """@capability("fusion", ...) 不出 ValueError → 维度合法."""

    @capability("fusion", "fused_kv")
    def _fuse():
        return None  # pragma: no cover

    assert callable(_fuse)
