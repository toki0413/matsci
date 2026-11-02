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


def _rel_norm_dist(va: np.ndarray, vb: np.ndarray) -> float:
    diff = float(np.linalg.norm(va - vb))
    scale = float(np.linalg.norm(va)) + float(np.linalg.norm(vb))
    return diff / (scale + _EPS)


def _elastic_dist(ea, eb) -> float | None:
    if ea is None or eb is None:
        return None
    ca, cb = ea.C, eb.C
    if ca.shape != cb.shape:
        return None
    return _rel_norm_dist(ca, cb)


def _phonon_dist(fa, fb) -> float | None:
    if fa is None or fb is None:
        return None
    fa = np.asarray(fa, dtype=float)
    fb = np.asarray(fb, dtype=float)
    if fa.shape != fb.shape:
        return None
    return _rel_norm_dist(fa, fb)


def _scalar_dist(a, b) -> float | None:
    if a is None or b is None:
        return None
    a = float(a)
    b = float(b)
    return abs(a - b) / (abs(a) + abs(b) + _EPS)


def _thermal_dist(ta, tb) -> float | None:
    if ta is None or tb is None:
        return None
    common = set(ta) & set(tb)
    if not common:
        return None
    va = np.array([float(ta[k]) for k in common])
    vb = np.array([float(tb[k]) for k in common])
    return _rel_norm_dist(va, vb)


def _record(
    dists: dict[str, float], weights: dict[str, float], key: str, value: float | None
) -> None:
    if value is not None:
        dists[key] = value
        weights[key] = _HAPTIC_WEIGHTS[key]


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
            "elastic": (self.elastic.C.tolist() if self.elastic is not None else None),
            "phonon_freqs": (
                self.phonon_freqs.tolist() if self.phonon_freqs is not None else None
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
        _record(dists, weights, "elastic", _elastic_dist(self.elastic, other.elastic))
        # phonon: 频率向量 L2
        _record(
            dists,
            weights,
            "phonon",
            _phonon_dist(self.phonon_freqs, other.phonon_freqs),
        )
        # surface_energy: 标量绝对差
        _record(
            dists,
            weights,
            "surface",
            _scalar_dist(self.surface_energy, other.surface_energy),
        )
        # thermal: 公共 key 的值向量 L2
        _record(dists, weights, "thermal", _thermal_dist(self.thermal, other.thermal))
        # density: 标量绝对差
        _record(dists, weights, "density", _scalar_dist(self.density, other.density))

        if not dists:
            return 0.0

        total_w = sum(weights.values())
        if total_w <= 0.0:
            return 0.0
        return sum(dists[k] * weights[k] for k in dists) / total_w


# ── self-check ─────────────────────────────────────────────────


def _check_empty_and_density() -> None:
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


def _check_roundtrip() -> HapticPropertyLayer:
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
    return rt


def _check_partial_renorm() -> float:
    # 4. 缺失子模态权重重归一化: 只 density 时距离用 density 权重 (归一化到 1)
    #    两个 density 差 1 倍 → 相对差 = 1/(1+2) ≈ 0.333
    only_d1 = HapticPropertyLayer(density=1.0)
    only_d2 = HapticPropertyLayer(density=2.0)
    d_only = only_d1.haptic_distance(only_d2)
    assert abs(d_only - (1.0 / 3.0)) < 1e-9, f"re-normalized density dist: {d_only}"
    return d_only


def _check_full() -> float:
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
    return d_full


def _selfcheck() -> None:
    """Assert-based demo: 创建 + 序列化 round-trip + 距离单调性."""
    _check_empty_and_density()
    rt = _check_roundtrip()
    d_only = _check_partial_renorm()
    d_full = _check_full()

    print("✓ haptic_property_layer self-check passed")
    print("  empty distance = 0.0")
    print(f"  density-only distance = {d_only:.4f}")
    print(f"  full-layer uniform distance = {d_full:.4f}")
    print(f"  round-trip elastic C shape = {rt.elastic.C.shape}")


if __name__ == "__main__":
    _selfcheck()
