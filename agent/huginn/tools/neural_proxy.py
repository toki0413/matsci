"""Neural PDE proxy — Transolver 风格的神经 PDE 代理模型占位.

这是一个 placeholder: 接口已经定好 (load_model / predict), 但真正的
Transolver 权重需要离线训练后再 load_model 进来. torch 不可用时整体降级,
上层 (multi_fidelity_tool) 会退回纯 FEM / GP 路径.

ponytail: 这里没有实现真正的 Transolver 架构. 实际部署需要:
  1. 离线在 (mesh, bc) -> solution 对上训练 Transolver
  2. 把权重 dump 到 model_path
  3. load_model 加载后 predict 才返回有意义的解
当前 predict 在 torch 可用但没加载权重时返回零场 + 降级标记, 调用方应据此
决定是否信任该结果.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# torch / transolver 都是可选依赖, 缺了就降级
try:
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - 环境相关
    _TORCH_AVAILABLE = False

try:
    import transolver  # type: ignore  # noqa: F401

    _TRANSOLVER_AVAILABLE = True
except Exception:
    _TRANSOLVER_AVAILABLE = False


@dataclass
class ProxySolution:
    """神经代理预测的解场 + 元信息."""

    field: np.ndarray  # 解场, shape 跟 mesh 节点数对齐
    available: bool = True  # False = 降级, 调用方应回退 FEM
    reason: str = ""
    backend: str = "transolver"  # 实际后端名, 方便日志区分
    meta: dict[str, Any] = field(default_factory=dict)


class NeuralPDEProxy:
    """Transolver 风格神经 PDE 代理模型.

    接口:
        load_model(model_path)  -> 加载训练好的权重
        predict(mesh, bc)        -> 返回 ProxySolution

    torch 不可用或权重未加载时, predict 返回 available=False 的解,
    让 multi_fidelity_tool 自动回退到 FEM 校验路径.
    """

    def __init__(self, device: str = "cpu") -> None:
        self._model: Any = None
        self._model_path: str | None = None
        self._loaded = False
        self._device = device

    # ── 可用性探测 ──────────────────────────────────────────────

    @staticmethod
    def available() -> bool:
        """torch + transolver 都在才算真正可用."""
        return _TORCH_AVAILABLE and _TRANSOLVER_AVAILABLE

    @staticmethod
    def status() -> str:
        """给人看的可用性描述, 方便降级日志."""
        if _TORCH_AVAILABLE and _TRANSOLVER_AVAILABLE:
            return "transolver backend ready"
        missing = []
        if not _TORCH_AVAILABLE:
            missing.append("torch")
        if not _TRANSOLVER_AVAILABLE:
            missing.append("transolver")
        return f"neural proxy not available (missing: {', '.join(missing)}), use FEM"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── 接口 ────────────────────────────────────────────────────

    def load_model(self, model_path: str) -> None:
        """加载训练好的 Transolver 权重.

        ponytail: 真正实现里这里应该 torch.load + 构造 Transolver 网络 + load_state_dict.
        当前只记路径并标记 loaded, predict 会据此决定返回零场还是真解.
        """
        if not _TORCH_AVAILABLE:
            # 没有 torch, load 直接标记不可用, 不抛异常让上层降级
            self._loaded = False
            self._model_path = model_path
            return

        self._model_path = model_path
        try:
            # 占位: 真正的 Transolver 权重加载在这里
            # self._model = Transolver(...); self._model.load_state_dict(torch.load(model_path))
            self._model = None
            self._loaded = True
        except Exception:
            self._loaded = False

    def predict(self, mesh: dict, bc: dict) -> ProxySolution:
        """对给定 mesh + boundary_conditions 做快速神经预估.

        mesh:  {'nodes': np.ndarray (n, d), 'elements': ...}
        bc:    {'type': 'dirichlet', 'values': ...}
        """
        if not _TORCH_AVAILABLE:
            return ProxySolution(
                field=np.array([]),
                available=False,
                reason="torch not available",
                backend="none",
            )

        # 拿节点数, 至少能给个形状对齐的零场
        nodes = mesh.get("nodes") if isinstance(mesh, dict) else mesh
        try:
            n_nodes = int(np.asarray(nodes).shape[0])
        except Exception:
            n_nodes = 0

        if not self._loaded or self._model is None:
            # 权重没加载, 返回零场并标记降级
            return ProxySolution(
                field=np.zeros(n_nodes),
                available=False,
                reason="model weights not loaded",
                backend="transolver",
                meta={"n_nodes": n_nodes},
            )

        # ponytail: 真正的 Transolver forward 在这里
        # features = self._encode_mesh(mesh, bc)
        # out = self._model(features.to(self._device))
        # return ProxySolution(field=out.cpu().numpy(), ...)
        # 占位: 用 bc 强度的均匀场近似, 至少形状对
        bc_val = 0.0
        if isinstance(bc, dict):
            vals = bc.get("values")
            if vals is not None:
                try:
                    bc_val = float(np.mean(vals))
                except Exception:
                    bc_val = 0.0
        field = np.full(n_nodes, bc_val)

        return ProxySolution(
            field=field,
            available=True,
            backend="transolver",
            meta={"n_nodes": n_nodes, "note": "placeholder uniform field"},
        )
