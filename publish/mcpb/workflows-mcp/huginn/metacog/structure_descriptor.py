"""StructureDescriptor — 结构 latent space (对齐层 Step 2).

从 StructureCognitiveMap 提取 16 维物理描述符: 配位数 / 键角统计 / 堆积系数 /
密度 / 空间群 / 近邻距离直方图. 跟 HapticDescriptor 配对, 喂给 AlignmentFunction
(Step 4) 学 structure -> haptic 对齐, surprise 驱动发现 (Step 6).

SE(3) 不变: 配位数 / 键角 / 距离直方图都不依赖绝对坐标朝向, 旋转平移后 descriptor
不变. 这是对齐层能用 structure 预测 haptic 的前提 (haptic 也是 SE(3) 不变的).

ponytail: 描述符全是 numpy 一行能算的物理量, 不上 GNN / 晶体图卷积.
ceiling: 对 supercell 敏感 (直方图 bin 是固定范围), 不区分键类型. 升级路径接
pymatgen StructureFeatures 或可学习 embedding.
"""
from __future__ import annotations

import numpy as np

from huginn.metacog.latent_space import LatentSpace
from huginn.metacog.structure_cognitive_map import StructureCognitiveMap

# 粗原子半径 (Å) — 共价/金属混合值, 算 packing_fraction 代理用.
# ponytail: 单值够用, 不查真共价/金属半径表. 对未知元素用默认值.
_ATOMIC_RADIUS: dict[str, float] = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.69,
    "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41,
    "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
    "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
    "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01,
    "W": 1.39, "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Pb": 1.46, "Bi": 1.48,
}
_DEFAULT_RADIUS = 1.5  # 未知元素 fallback

# 粗原子质量 (amu) — 算 density 代理用.
_ATOMIC_MASS: dict[str, float] = {
    "H": 1.008, "He": 4.003, "Li": 6.94, "Be": 9.012, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305,
    "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996,
    "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.63, "As": 74.922, "Se": 78.971, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.95,
    "Tc": 98.0, "Ru": 101.07, "Rh": 102.906, "Pd": 106.42, "Ag": 107.868, "Cd": 112.414,
    "In": 114.818, "Sn": 118.71, "Sb": 121.76, "Te": 127.6, "I": 126.904, "Xe": 131.293,
    "Cs": 132.905, "Ba": 137.327, "La": 138.905, "Ce": 140.116, "Pr": 140.908, "Nd": 144.242,
    "W": 183.84, "Pt": 195.084, "Au": 196.967, "Hg": 200.592, "Pb": 207.2, "Bi": 208.98,
}
_DEFAULT_MASS = 50.0  # 未知元素 fallback

# 近邻距离直方图固定范围 (Å). 默认 cutoff=3.0, 留余量到 5.0.
# ponytail: 固定范围让 GP 核函数 bin 跨样本对齐. ceiling: cutoff > 5 的远邻全落末 bin.
_HIST_RANGE = (0.0, 5.0)
_HIST_BINS = 10


class StructureDescriptor(LatentSpace):
    """结构 latent space — 16 维物理描述符.

    维度布局:
      [0]     coordination_number  配位数均值
      [1]     bond_angle_mean      键角均值 (度)
      [2]     bond_angle_std       键角标准差
      [3]     packing_fraction     堆积系数 (原子体积和 / 胞体积)
      [4]     density              密度 (g/cm³, metadata 优先, 否则质量/体积算)
      [5]     spacegroup_number    归一化到 [0,1] (÷230), 缺失填 0
      [6:16]  neighbor_distance_histogram  10 bins, 归一化到 sum=1
    """

    @property
    def dim(self) -> int:
        return 16

    @property
    def name(self) -> str:
        return "structure"

    def encode(self, cmap: StructureCognitiveMap) -> np.ndarray:
        """编码 StructureCognitiveMap -> 16 维. 缺字段填 0, 不报错."""
        vec = np.zeros(16, dtype=float)
        n = len(cmap.species)

        # [0] 配位数均值
        if n > 0:
            cn = [len(cmap.adjacency.get(i, [])) for i in range(n)]
            vec[0] = float(np.mean(cn))

        # [1-2] 键角均值/std
        angles = self._collect_bond_angles(cmap)
        if angles:
            vec[1] = float(np.mean(angles))
            vec[2] = float(np.std(angles))

        # [3] packing_fraction
        vec[3] = self._packing_fraction(cmap)

        # [4] density
        vec[4] = self._density(cmap)

        # [5] spacegroup_number 归一化
        sg = cmap.metadata.get("spacegroup_number")
        if sg is not None:
            vec[5] = float(sg) / 230.0

        # [6:16] 近邻距离直方图
        vec[6:16] = self._neighbor_distance_histogram(cmap)

        return vec

    # ── 各维度计算 ──────────────────────────────────────────

    def _collect_bond_angles(self, cmap: StructureCognitiveMap) -> list[float]:
        """收集所有 i-j-k 键角 (j 是顶点, i/k 是 j 的邻居对)."""
        angles: list[float] = []
        n = len(cmap.species)
        for j in range(n):
            nbs = cmap.adjacency.get(j, [])
            if len(nbs) < 2:
                continue
            for ii in range(len(nbs)):
                for kk in range(ii + 1, len(nbs)):
                    try:
                        angles.append(cmap.query_angle(nbs[ii], j, nbs[kk]))
                    except Exception:
                        continue
        return angles

    def _packing_fraction(self, cmap: StructureCognitiveMap) -> float:
        """堆积系数 = 原子体积和 / 胞体积. 无 lattice 返回 0."""
        if cmap.lattice is None:
            return 0.0
        vol_cell = float(abs(np.linalg.det(cmap.lattice)))
        if vol_cell < 1e-12:
            return 0.0
        vol_atoms = 0.0
        for sp in cmap.species:
            r = _ATOMIC_RADIUS.get(sp, _DEFAULT_RADIUS)
            vol_atoms += (4.0 / 3.0) * np.pi * (r ** 3)
        return float(vol_atoms / vol_cell)

    def _density(self, cmap: StructureCognitiveMap) -> float:
        """密度 g/cm³. metadata 优先, 否则质量/体积算. 算不出返回 0."""
        d = cmap.metadata.get("density")
        if d is not None:
            return float(d)
        if cmap.lattice is None:
            return 0.0
        vol_ang3 = float(abs(np.linalg.det(cmap.lattice)))
        if vol_ang3 < 1e-12:
            return 0.0
        total_mass_amu = sum(_ATOMIC_MASS.get(sp, _DEFAULT_MASS) for sp in cmap.species)
        # 1 amu = 1.66054e-24 g, 1 Å³ = 1e-24 cm³ → amu/Å³ × 1.66054 = g/cm³
        return float(total_mass_amu * 1.66054 / vol_ang3)

    def _neighbor_distance_histogram(self, cmap: StructureCognitiveMap) -> np.ndarray:
        """近邻距离直方图 (10 bins, 归一化 sum=1). 无近邻返回全 0."""
        dists: list[float] = []
        n = len(cmap.species)
        for i in range(n):
            for j in cmap.adjacency.get(i, []):
                if j > i:  # 每对只算一次
                    try:
                        dists.append(cmap.query_distance(i, j))
                    except Exception:
                        continue
        if not dists:
            return np.zeros(_HIST_BINS, dtype=float)
        hist, _ = np.histogram(dists, bins=_HIST_BINS, range=_HIST_RANGE)
        total = hist.sum()
        if total > 0:
            hist = hist / total
        return hist.astype(float)


# ── self-check ─────────────────────────────────────────────────

def _selfcheck() -> None:
    """assert demo: encode 16 维 + 物理量合理 + 缺字段安全降级 + 距离单调."""
    sd = StructureDescriptor()
    assert sd.dim == 16
    assert sd.name == "structure"

    # 2 原子 NaCl-like 结构 (有 lattice)
    m1 = StructureCognitiveMap.from_coords(
        species=["Na", "Cl"],
        coords=np.array([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]]),
        lattice=np.eye(3) * 5.6,
        cutoff=3.0,
    )
    v1 = sd.encode(m1)
    assert v1.shape == (16,), f"expected (16,), got {v1.shape}"
    assert np.isfinite(v1).all(), f"non-finite: {v1}"
    # Na-Cl 距离 2.8 < cutoff 3.0 → 每个原子 1 个邻居 → 配位数均值 1.0
    assert abs(v1[0] - 1.0) < 1e-9, f"coordination expected 1.0, got {v1[0]}"
    assert v1[3] > 0, f"packing_fraction > 0: {v1[3]}"  # 有 lattice
    assert v1[4] > 0, f"density > 0: {v1[4]}"           # 有 lattice
    assert v1[6:16].sum() > 0, f"histogram all zero: {v1[6:16]}"  # 1 条近邻

    # 单原子结构 (无键角 / 无近邻) — 安全降级全 0
    m_single = StructureCognitiveMap.from_coords(
        species=["Fe"], coords=np.array([[0.0, 0.0, 0.0]]), cutoff=3.0
    )
    v_single = sd.encode(m_single)
    assert v_single.shape == (16,)
    assert np.isfinite(v_single).all()
    assert v_single[0] == 0.0          # 无邻居
    assert v_single[1] == 0.0 and v_single[2] == 0.0  # 无键角
    assert v_single[6:16].sum() == 0.0  # 无近邻距离

    # 距离: 相同 = 0, 不同 > 0
    assert sd.distance(m1, m1) < 1e-9
    m2 = StructureCognitiveMap.from_coords(
        species=["Fe", "Fe", "Fe"],
        coords=np.array([[0, 0, 0], [2.0, 0, 0], [0, 2.0, 0]]),
        cutoff=3.0,
    )
    assert sd.distance(m1, m2) > 0.0

    # spacegroup 归一化: metadata 有值时除以 230
    m_sg = StructureCognitiveMap.from_coords(
        species=["Si"], coords=np.array([[0, 0, 0]]), cutoff=3.0
    )
    m_sg.metadata["spacegroup_number"] = 227  # Fd-3m 金刚石
    v_sg = sd.encode(m_sg)
    assert abs(v_sg[5] - 227.0 / 230.0) < 1e-9, f"spacegroup norm: {v_sg[5]}"

    print("✓ structure_descriptor self-check passed")
    print(f"  NaCl-like vec[0:6] = {np.round(v1[:6], 4).tolist()}")
    print(f"  NaCl-like hist     = {np.round(v1[6:16], 4).tolist()}")
    print(f"  single-atom vec    = {np.round(v_single[:6], 4).tolist()}")
    print(f"  d(m1, m1) = {sd.distance(m1, m1):.6f}")
    print(f"  d(m1, m2) = {sd.distance(m1, m2):.4f}")


if __name__ == "__main__":
    _selfcheck()
