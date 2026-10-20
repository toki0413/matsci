"""Fusion 能力维度 —— 模型间隐表示 (KV-KV) 语义融合通道的接缝.

对标 C2C (Cache-to-Cache, arxiv:2510.03215) 的通信范式: 不用文本中转, 而是把
source 模型的 KV-Cache 直接投影、融合进 target 模型的 KV-Cache. ``fusion``
维度声明"本 harness 支持这种能力", 但**不默认加载任何投影网络**.

设计约束 (诚实边界):
  - `A FusionChannel 是本地模型运行的桥**: 投影网络需要物理访问两端模型的 KV-Cache
    (DynamicCache), 走 API 推理的默认链路拿不到内部表示. 因此能力实现必须"按需挂载",
    由实际拥有本地模型 + projector 权重的运行环境提供 (research/ 给了纯 torch 参考实现).
  - `A FusionChannel 与 loop/session 等维度不同**: 它不改变 agent 的编排流, 只暴露
    "能否融合 / 如何融合" 的协议契约. 默认内置是一个**探针**, 如实报告当前链路是否可达,
    而不是假装能融合.
  - 这里只定义协议与状态探针, 不 import torch, 保证维度本身零重型依赖.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from huginn.capabilities.capability import CapabilityMetadata
from huginn.capabilities.registry import get_shared_capability_registry

logger = logging.getLogger(__name__)


@dataclass
class FusionStatus:
    """融合通道的可达性状态 (探针结果)."""

    available: bool                       # 是否真正具备融合能力
    reason: str                           # 可达/不可达的说明
    source: str = "builtin:probe"         # 谁挂载的
    details: dict[str, Any] = field(default_factory=dict)


class FusionChannel(ABC):
    """KV-KV 语义融合通道协议.

    提供方实现并挂载到 ``fusion.fused_kv`` 能力上. 契约:
      - ``probe()``    : 报告该通道是否可用及原因;
      - ``fuse(...)``  : source_kv / target_kv 各自为 (key, value), 返回融合后的 KV.
    具体张量形状由运行时约定 (见 research/ 参考实现), 本协议不 bound 到任何张量库.
    """

    @abstractmethod
    def probe(self) -> FusionStatus:
        raise NotImplementedError

    @abstractmethod
    def fuse(
        self,
        source_kv: tuple[Any, Any],
        target_kv: tuple[Any, Any],
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        raise NotImplementedError


# ── 内置探针能力 ──────────────────────────────────────────────────

def _probe_fused_kv(ctx: dict[str, Any] | None = None) -> FusionStatus:
    """默认 (未挂载) 的融合通道极具: 如实报告不可用.

    语义融合需要本地模型 KV-Cache 访问权; 走 API 推理的默认链路拿不到,
    因此如实返回 available=False, 而非假装支持 —— 保持可证伪/诚实边界.
    """
    return FusionStatus(
        available=False,
        reason=(
            "default fusion channel is a seam/probe: KV-KV fusion needs local "
            "model KV-Cache access (DynamicCache) + projector weights, which the "
            "default API-backed chain cannot see. Mount a FusionChannel impl via "
            "research.C2CProjector or a runtime that owns the models."
        ),
    )


def _mk(dimension: str, name: str, impl: Any) -> CapabilityMetadata:
    return CapabilityMetadata(dimension=dimension, name=name, impl=impl, plugin_name="builtin")


def register_fusion_capabilities() -> None:
    """注册 fusion 内置能力到共享 registry (幂等: 同名覆盖).

    注意: 只注册"探针", 不注册任何投影网络. 真正可用的 channel 由外部运行环境
    (拥有本地模型 + projector 权重) 用同维度 name ``fused_kv`` 覆盖注册.
    """
    reg = get_shared_capability_registry()
    reg.register(_mk("fusion", "fused_kv", _probe_fused_kv))


__all__ = [
    "FusionStatus", "FusionChannel",
    "register_fusion_capabilities", "_probe_fused_kv",
]
