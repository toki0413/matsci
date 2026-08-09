"""HapticDescriptor — 力学 latent space (对齐层 Step 2).

从 HapticPropertyLayer 提取 12 维力学描述符: 弹性张量特征值 / 体模量 / 剪切模量 /
声子频率统计 / 表面能 / 热膨胀. 跟 StructureDescriptor 配对, 喂给 AlignmentFunction
(Step 4) 学 structure -> haptic 对齐.

缺失字段是常态 — DFT 算全很贵, ML 只给部分, empirical 只给标量. 全部填 0, 不报错.
mask() 返回哪些维度是真实数据 vs 填充, 让 AlignmentFunction (Step 4) 的 uncertainty
加权 (填充维度不确定性大).

ponytail: elastic 用 eigvalsh (对称矩阵), 不做群论约化. bulk/shear 用 Voigt 平均
(上界), 不做 Reuss-Hill 平均 — 单界够用作 descriptor.
ceiling: Voigt 上界偏高, 升级路径接 VRH 平均 (mechanics.ElasticTensor 已有 reuss_moduli).
"""
from __future__ import annotations

import contextlib

import numpy as np

from huginn.metacog.haptic_property_layer import HapticPropertyLayer
from huginn.metacog.latent_space import LatentSpace


class HapticDescriptor(LatentSpace):
    """力学 latent space — 12 维.

    维度布局:
      [0:6]  elastic_eigenvalues  弹性张量 6 个特征值 (升序, 缺失填 0)
      [6]    bulk_modulus         Voigt 体模量 (GPa, 缺失填 0)
      [7]    shear_modulus        Voigt 剪切模量 (GPa, 缺失填 0)
      [8]    phonon_mean          声子频率均值 (THz, 缺失填 0)
      [9]    phonon_min           声子频率最小值 (THz, 缺失填 0)
      [10]   surface_energy       表面能 (J/m², 缺失填 0)
      [11]   thermal_expansion    热膨胀系数 (1/K, 缺失填 0)
    """

    @property
    def dim(self) -> int:
        return 12

    @property
    def name(self) -> str:
        return "haptic"

    def encode(self, layer: HapticPropertyLayer) -> np.ndarray:
        """编码 HapticPropertyLayer -> 12 维. 缺失字段填 0, 不报错."""
        vec = np.zeros(12, dtype=float)

        # [0:6] elastic eigenvalues
        if layer.elastic is not None:
            C = np.asarray(layer.elastic.C, dtype=float)
            if C.shape == (6, 6):
                # 对称化防数值噪声 — Voigt 张量理应对称
                C_sym = (C + C.T) / 2.0
                with contextlib.suppress(np.linalg.LinAlgError):
                    vec[0:6] = np.sort(np.linalg.eigvalsh(C_sym))
            elif C.ndim == 1 and C.size >= 6:
                # Voigt 6 向量直接取前 6 个
                vec[0:6] = np.sort(C[:6])

        # [6] bulk_modulus, [7] shear_modulus — Voigt 平均
        if layer.elastic is not None:
            try:
                moduli = layer.elastic.voigt_moduli()
                vec[6] = float(moduli["bulk_modulus_voigt"])
                vec[7] = float(moduli["shear_modulus_voigt"])
            except Exception:
                pass

        # [8] phonon_mean, [9] phonon_min
        if layer.phonon_freqs is not None:
            freqs = np.asarray(layer.phonon_freqs, dtype=float)
            if freqs.size > 0:
                vec[8] = float(np.mean(freqs))
                vec[9] = float(np.min(freqs))

        # [10] surface_energy
        if layer.surface_energy is not None:
            vec[10] = float(layer.surface_energy)

        # [11] thermal_expansion
        if layer.thermal is not None:
            vec[11] = float(layer.thermal.get("thermal_expansion", 0.0))

        return vec

    def mask(self, layer: HapticPropertyLayer) -> np.ndarray:
        """返回 12 维 bool mask, True = 该维度有真实数据.

        给 AlignmentFunction (Step 4) 的 uncertainty 加权用: 填充维度 (False)
        不确定性大, 真实维度 (True) 不确定性小.
        """
        m = np.zeros(12, dtype=bool)

        if layer.elastic is not None:
            C = np.asarray(layer.elastic.C, dtype=float)
            if C.shape == (6, 6) or (C.ndim == 1 and C.size >= 6):
                m[0:6] = True
                m[6:8] = True  # bulk/shear 也从 elastic 算

        if layer.phonon_freqs is not None:
            freqs = np.asarray(layer.phonon_freqs, dtype=float)
            if freqs.size > 0:
                m[8:10] = True

        if layer.surface_energy is not None:
            m[10] = True

        if layer.thermal is not None and "thermal_expansion" in layer.thermal:
            m[11] = True

        return m


# ── self-check ─────────────────────────────────────────────────

def _selfcheck() -> None:
    """assert demo: 空 layer 全 0, 全字段无 0, mask 一致, 距离单调."""
    from huginn.mechanics import ElasticTensor

    hd = HapticDescriptor()
    assert hd.dim == 12
    assert hd.name == "haptic"

    # 空 layer — 12 维全 0, mask 全 False
    empty = HapticPropertyLayer()
    v_empty = hd.encode(empty)
    assert v_empty.shape == (12,), f"expected (12,), got {v_empty.shape}"
    assert np.all(v_empty == 0.0), f"empty should be all 0: {v_empty}"
    assert not hd.mask(empty).any()
    assert hd.distance(empty, empty) < 1e-9

    # 全字段 layer
    C = np.diag([300.0, 200.0, 150.0, 100.0, 80.0, 60.0])
    C[0, 1] = C[1, 0] = 50.0
    C[0, 2] = C[2, 0] = 40.0
    C[1, 2] = C[2, 1] = 30.0
    full = HapticPropertyLayer(
        elastic=ElasticTensor(C=C),
        phonon_freqs=np.array([2.0, 5.0, 8.0, 12.0]),
        surface_energy=1.5,
        thermal={"thermal_expansion": 1.2e-5, "thermal_conductivity": 50.0},
        density=7.8,
    )
    v_full = hd.encode(full)
    assert v_full.shape == (12,)
    assert np.isfinite(v_full).all()
    assert v_full[0:6].all(), f"elastic eigenvalues non-zero: {v_full[0:6]}"
    assert v_full[6] > 0 and v_full[7] > 0, f"bulk/shear > 0: {v_full[6:8]}"
    assert abs(v_full[8] - 6.75) < 1e-9, f"phonon mean: {v_full[8]}"  # (2+5+8+12)/4
    assert abs(v_full[9] - 2.0) < 1e-9, f"phonon min: {v_full[9]}"
    assert abs(v_full[10] - 1.5) < 1e-9
    assert abs(v_full[11] - 1.2e-5) < 1e-15
    assert hd.mask(full).all(), f"full mask all True: {hd.mask(full)}"

    # 部分字段 — 只有 elastic
    partial = HapticPropertyLayer(elastic=ElasticTensor(C=np.eye(6) * 100.0))
    v_part = hd.encode(partial)
    assert v_part[0:6].all()              # elastic 有
    assert v_part[6] > 0 and v_part[7] > 0  # bulk/shear 有
    assert v_part[8] == 0.0 and v_part[9] == 0.0  # phonon 无
    assert v_part[10] == 0.0 and v_part[11] == 0.0  # surface/thermal 无
    m_part = hd.mask(partial)
    assert m_part[0:8].all() and not m_part[8:12].any()

    # 距离单调
    assert hd.distance(empty, full) > 0.0
    assert hd.distance(full, full) < 1e-9

    print("✓ haptic_descriptor self-check passed")
    print(f"  empty vec    = {v_empty}")
    print(f"  full vec     = {np.round(v_full, 4).tolist()}")
    print(f"  full mask    = {hd.mask(full).tolist()}")
    print(f"  partial mask = {m_part.tolist()}")
    print(f"  d(empty, full) = {hd.distance(empty, full):.4f}")


if __name__ == "__main__":
    _selfcheck()
