"""Haptic Property Layer — 近场力学感知层.

神经科学依据: 主动触觉探索 (haptic exploration). 盲人用手看物体时, 手指运动
轨迹本身就是认知过程. 对应 MCMC proposal 不应只靠几何相似性 (SE(3)), 还应靠
力学相似性 — 摸到一个软的东西后倾向于继续摸软的.

封装材料的近场力学属性 (弹性/声子/表面能/应力应变/热响应/密度), 给 agent
"触觉". 石墨 vs 金刚石: 视觉 (结构) 上都是碳, 触觉 (硬度) 上天差地别 —
这正是 single-modality (CIF only) 认知抓不到的.

接入点: StructureCognitiveMap.haptic (Step 2) + HypothesisManifold._haptic_layers
(Step 3) + mcmc_step 力学引导 proposal (Step 4) + cross_modal_check 跨模态验证
(Step 5).

ponytail: 复用 mechanics.py 的 ElasticTensor + numpy, 不引新依赖.
ceiling: haptic_distance 用相对范数差代理真度量, 升级路径接 Mahalanobis (按
协方差加权) 或学习式 embedding 距离.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from huginn.mechanics import ElasticTensor

# 子模态权重 — elastic/phonon 是主导 (各 0.3), surface/thermal 次之 (各 0.15),
# density 弱 (0.1). stress_strain 不参与距离 (曲线对齐需 DTW, 留升级路径).
_HAPTIC_WEIGHTS: dict[str, float] = {
    "elastic": 0.3,
    "phonon": 0.3,
    "surface": 0.15,
    "thermal": 0.15,
    "density": 0.1,
}

_EPS = 1e-12


@dataclass
class HapticPropertyLayer:
    """近场力学属性层 — 材料的 "触觉".

    所有字段可选: 实际场景里 DFT 算全很贵, ML 只给部分, empirical 给标量.
    缺失子模态在 haptic_distance 里跳过 + 权重重新归一化.

    Attributes:
        elastic: 弹性刚度张量 (6x6 Voigt, GPa), 复用 mechanics.ElasticTensor.
        phonon_freqs: 声子频率向量 (THz), 振动模式指纹.
        surface_energy: 表面能 (J/m²), 纹理.
        stress_strain: 应力-应变曲线 (dict, 不参与距离, 留作扩展).
        thermal: {"thermal_expansion": float, "thermal_conductivity": float} 等.
        density: 密度 (g/cm³).
        source: 数据来源 (DFT/ML/empirical).
        confidence: 数据置信度 [0, 1].
    """

    elastic: ElasticTensor | None = None
    phonon_freqs: np.ndarray | None = None
    surface_energy: float | None = None
    stress_strain: dict | None = None
    thermal: dict | None = None
    density: float | None = None
    source: str = "DFT"
    confidence: float = 1.0

    # ── 序列化 ──────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict. ElasticTensor.C → list-of-list, phonon → list."""
        return {
            "elastic": (
                self.elastic.C.tolist() if self.elastic is not None else None
            ),
            "phonon_freqs": (
                self.phonon_freqs.tolist()
                if self.phonon_freqs is not None
                else None
            ),
            "surface_energy": self.surface_energy,
            "stress_strain": self.stress_strain,
            "thermal": self.thermal,
            "density": self.density,
            "source": self.source,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HapticPropertyLayer:
        """反序列化. elastic list → ElasticTensor, phonon list → ndarray."""
        elastic = None
        if d.get("elastic") is not None:
            elastic = ElasticTensor(C=np.asarray(d["elastic"], dtype=float))
        phonon = None
        if d.get("phonon_freqs") is not None:
            phonon = np.asarray(d["phonon_freqs"], dtype=float)
        return cls(
            elastic=elastic,
            phonon_freqs=phonon,
            surface_energy=d.get("surface_energy"),
            stress_strain=d.get("stress_strain"),
            thermal=d.get("thermal"),
            density=d.get("density"),
            source=d.get("source", "DFT"),
            confidence=float(d.get("confidence", 1.0)),
        )

    # ── 距离 ────────────────────────────────────────────────

    def haptic_distance(self, other: HapticPropertyLayer) -> float:
        """加权 L2 距离, 各子模态相对范数归一化后加权.

        缺失子模态跳过 + 权重重新归一化. 都缺失返回 0.0.
        返回值落在 [0, 1] 附近 (相对差上界 ≈ 1), threshold 0.5 用于 cross_modal.

        ponytail: 相对范数差 ||a-b|| / (||a||+||b||) 代理真度量.
        ceiling: 不按协方差加权 (Mahalanobis), 不做曲线对齐 (DTW for stress_strain).
        """
        dists: dict[str, float] = {}
        weights: dict[str, float] = {}

        # elastic: Frobenius 范数差
        if self.elastic is not None and other.elastic is not None:
            ca = self.elastic.C
            cb = other.elastic.C
            if ca.shape == cb.shape:
                diff = float(np.linalg.norm(ca - cb))
                scale = float(np.linalg.norm(ca)) + float(np.linalg.norm(cb))
                dists["elastic"] = diff / (scale + _EPS)
                weights["elastic"] = _HAPTIC_WEIGHTS["elastic"]

        # phonon: 频率向量 L2
        if (
            self.phonon_freqs is not None
            and other.phonon_freqs is not None
        ):
            fa = np.asarray(self.phonon_freqs, dtype=float)
            fb = np.asarray(other.phonon_freqs, dtype=float)
            if fa.shape == fb.shape:
                diff = float(np.linalg.norm(fa - fb))
                scale = float(np.linalg.norm(fa)) + float(np.linalg.norm(fb))
                dists["phonon"] = diff / (scale + _EPS)
                weights["phonon"] = _HAPTIC_WEIGHTS["phonon"]

        # surface_energy: 标量绝对差
        if (
            self.surface_energy is not None
            and other.surface_energy is not None
        ):
            a = float(self.surface_energy)
            b = float(other.surface_energy)
            dists["surface"] = abs(a - b) / (abs(a) + abs(b) + _EPS)
            weights["surface"] = _HAPTIC_WEIGHTS["surface"]

        # thermal: 公共 key 的值向量 L2
        if self.thermal is not None and other.thermal is not None:
            common = set(self.thermal) & set(other.thermal)
            if common:
                va = np.array([float(self.thermal[k]) for k in common])
                vb = np.array([float(other.thermal[k]) for k in common])
                diff = float(np.linalg.norm(va - vb))
                scale = float(np.linalg.norm(va)) + float(np.linalg.norm(vb))
                dists["thermal"] = diff / (scale + _EPS)
                weights["thermal"] = _HAPTIC_WEIGHTS["thermal"]

        # density: 标量绝对差
        if self.density is not None and other.density is not None:
            a = float(self.density)
            b = float(other.density)
            dists["density"] = abs(a - b) / (abs(a) + abs(b) + _EPS)
            weights["density"] = _HAPTIC_WEIGHTS["density"]

        if not dists:
            return 0.0

        total_w = sum(weights.values())
        if total_w <= 0.0:
            return 0.0
        return sum(dists[k] * weights[k] for k in dists) / total_w


# ── self-check ─────────────────────────────────────────────────

def _selfcheck() -> None:
    """Assert-based demo: 创建 + 序列化 round-trip + 距离单调性."""
    # 1. 空 layer
    h = HapticPropertyLayer()
    assert h.elastic is None and h.density is None
    assert h.haptic_distance(HapticPropertyLayer()) == 0.0

    # 2. density-only 距离 > 0, 自距离 = 0
    a = HapticPropertyLayer(density=1.0)
    b = HapticPropertyLayer(density=2.0)
    d = a.haptic_distance(b)
    assert d > 0.0, f"density diff should be > 0, got {d}"
    assert a.haptic_distance(a) == 0.0

    # 3. 序列化 round-trip (elastic + phonon + density)
    C = np.eye(6) * 100.0
    src = HapticPropertyLayer(
        elastic=ElasticTensor(C=C),
        phonon_freqs=np.array([1.0, 2.0, 3.0]),
        density=5.0,
        source="ML",
        confidence=0.8,
    )
    d_dict = src.to_dict()
    assert d_dict["elastic"] is not None and len(d_dict["elastic"]) == 6
    assert d_dict["phonon_freqs"] == [1.0, 2.0, 3.0]
    rt = HapticPropertyLayer.from_dict(d_dict)
    assert rt.source == "ML" and rt.confidence == 0.8
    assert np.allclose(rt.elastic.C, C)
    assert np.allclose(rt.phonon_freqs, [1.0, 2.0, 3.0])
    assert rt.density == 5.0
    # round-trip 后自距离 = 0
    assert src.haptic_distance(rt) == 0.0

    # 4. 缺失子模态权重重归一化: 只 density 时距离用 density 权重 (归一化到 1)
    #    两个 density 差 1 倍 → 相对差 = 1/(1+2) ≈ 0.333
    only_d1 = HapticPropertyLayer(density=1.0)
    only_d2 = HapticPropertyLayer(density=2.0)
    d_only = only_d1.haptic_distance(only_d2)
    assert abs(d_only - (1.0 / 3.0)) < 1e-9, f"re-normalized density dist: {d_only}"

    # 5. 全子模态: 距离单调 (相同=0, 不同>0)
    C2 = np.eye(6) * 200.0
    full_a = HapticPropertyLayer(
        elastic=ElasticTensor(C=np.eye(6) * 100.0),
        phonon_freqs=np.array([1.0, 2.0]),
        surface_energy=1.0,
        thermal={"thermal_expansion": 1.0, "thermal_conductivity": 2.0},
        density=5.0,
    )
    full_b = HapticPropertyLayer(
        elastic=ElasticTensor(C=C2),
        phonon_freqs=np.array([2.0, 4.0]),
        surface_energy=2.0,
        thermal={"thermal_expansion": 2.0, "thermal_conductivity": 4.0},
        density=10.0,
    )
    assert full_a.haptic_distance(full_a) == 0.0
    assert full_a.haptic_distance(full_b) > 0.0
    # 各子模态相对差都是 1/3 → 加权后也是 1/3
    d_full = full_a.haptic_distance(full_b)
    assert abs(d_full - (1.0 / 3.0)) < 1e-9, f"uniform relative diff: {d_full}"

    print("✓ haptic_property_layer self-check passed")
    print("  empty distance = 0.0")
    print(f"  density-only distance = {d_only:.4f}")
    print(f"  full-layer uniform distance = {d_full:.4f}")
    print(f"  round-trip elastic C shape = {rt.elastic.C.shape}")


if __name__ == "__main__":
    _selfcheck()
