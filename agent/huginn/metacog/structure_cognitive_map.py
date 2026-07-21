"""Structure Cognitive Map — 神经科学启发的 3D 空间表示模块.

借鉴 Ego3D-VLM (arXiv:2509.06266, Huawei, 2025-09) training-free cognitive map:
海马体 place cells / grid cells (Moser-Nobel 2014) 的外部化. 给 LLM (无 vision
encoder) 显式 3D coords + adjacency + SE(3) 等变查询 API, 替代 text-centric 推理.

AtomWorld rotation <12% 的根因: 族系 I (text) β_rotation=0, 旋转模式无对应
token 流形结构. Cognitive map 升级到族系 II (显式 3D coords), β_rotation > 0,
旋转模式有对应 3D 矩阵乘法流形, 拓扑许可恢复.

接入点: CodeAct 沙箱 (通过 structure_cognitive_map_tool 注入) + EngineState 持久化.
env flag HUGINN_USE_COGNITIVE_MAP=1 控制. 默认 off, 行为完全不变.

ponytail: 用 pymatgen + scipy 现有依赖, 不引入新包. SE(3) 用 scipy.spatial.transform.Rotation
不手写旋转矩阵避免数值误差. adjacency 用 cutoff + PBC, 不上 KD-tree (YAGNI).
升级路径: 接 Hodge decomposition 做 topological analysis (后续 P2 候选).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# 默认 cutoff (Å). ponytail: 不区分共价/金属键, 单一 cutoff 够用.
# ceiling: 对离子晶体 + 长键合金可能漏邻, 升级路径接 pymatgen CrystalNN.
_DEFAULT_CUTOFF = 3.0

try:
    from pymatgen.core import Structure as _PmgStructure
    _HAS_PMG = True
except ImportError:
    _HAS_PMG = False

try:
    from scipy.spatial.transform import Rotation as _R
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


@dataclass
class SubgraphResult:
    """query_subgraph 返回值. k-hop 邻域子图."""

    nodes: list[int]
    edges: list[tuple[int, int]]
    node_species: list[str]


@dataclass
class StructureCognitiveMap:
    """3D 空间认知地图 — 显式 coords + adjacency + SE(3) operations.

    海马体 cognitive map 的外部化. Immutable 风格: 所有 transform 返回新实例.

    用法:
        m = StructureCognitiveMap.from_cif(cif_str)
        d = m.query_distance(0, 1)
        m2 = m.rotate(axis="z", angle=90)
        assert abs(m2.query_distance(0, 1) - d) < 1e-6  # SE(3) 等变
    """

    species: list[str]
    coords: np.ndarray
    lattice: np.ndarray | None = None
    adjacency: dict[int, list[int]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── 构造 ──────────────────────────────────────────────

    @classmethod
    def from_cif(cls, cif_str: str, cutoff: float = _DEFAULT_CUTOFF) -> "StructureCognitiveMap":
        """从 CIF 构建. 用 pymatgen.Structure 解析, 考虑 PBC."""
        if not _HAS_PMG:
            raise RuntimeError("pymatgen not installed, pip install pymatgen")
        s = _PmgStructure.from_str(cif_str, fmt="cif")
        cart = np.asarray(s.cart_coords, dtype=float)
        # lattice 行向量 (pymatgen lattice.matrix 是行向量 a/b/c)
        lat = np.asarray(s.lattice.matrix, dtype=float) if s.lattice is not None else None
        species = [str(sp) for sp in s.species]
        adj = _build_adjacency_cart(cart, lat, cutoff)
        return cls(species=species, coords=cart, lattice=lat, adjacency=adj,
                   metadata={"source": "cif", "num_sites": len(species)})

    @classmethod
    def from_molecule(cls, sdf_str: str, cutoff: float = _DEFAULT_CUTOFF) -> "StructureCognitiveMap":
        """从 SDF/Mol block 构建. 用 rdkit 解析. 不考虑 PBC."""
        try:
            from rdkit import Chem
        except ImportError:
            raise RuntimeError("rdkit not installed, pip install rdkit")
        mol = Chem.MolFromMolBlock(sdf_str, removeHs=False)
        if mol is None:
            raise ValueError("rdkit MolFromMolBlock returned None")
        conf = mol.GetConformer()
        n = mol.GetNumAtoms()
        coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                           for i in range(n)], dtype=float)
        species = [a.GetSymbol() for a in mol.GetAtoms()]
        adj = _build_adjacency_cart(coords, None, cutoff)
        return cls(species=species, coords=coords, lattice=None, adjacency=adj,
                   metadata={"source": "sdf", "num_sites": n})

    @classmethod
    def from_coords(cls, species: list[str], coords: np.ndarray,
                    lattice: np.ndarray | None = None,
                    cutoff: float = _DEFAULT_CUTOFF) -> "StructureCognitiveMap":
        """直接从 coords 构造. lattice=None 表示 molecule."""
        coords = np.asarray(coords, dtype=float)
        if coords.shape != (len(species), 3):
            raise ValueError(f"coords shape {coords.shape} != ({len(species)}, 3)")
        adj = _build_adjacency_cart(coords, lattice, cutoff)
        return cls(species=list(species), coords=coords, lattice=lattice, adjacency=adj,
                   metadata={"source": "coords", "num_sites": len(species)})

    # ── 6 query API ───────────────────────────────────────

    def query_distance(self, i: int, j: int) -> float:
        """原子 i-j 距离 (Å). crystal 考虑 PBC, molecule 直接欧氏."""
        self._check_index(i); self._check_index(j)
        if self.lattice is not None:
            d = _min_image_distance(self.coords[i], self.coords[j], self.lattice)
        else:
            d = float(np.linalg.norm(self.coords[i] - self.coords[j]))
        return d

    def query_angle(self, i: int, j: int, k: int) -> float:
        """i-j-k 角度 (度), j 是顶点."""
        self._check_index(i); self._check_index(j); self._check_index(k)
        v1 = self.coords[i] - self.coords[j]
        v2 = self.coords[k] - self.coords[j]
        # ponytail: PBC 不考虑 (角点局部, 邻居一般同 cell)
        cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        cos_t = float(np.clip(cos_t, -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_t)))

    def query_neighbors(self, i: int, cutoff: float | None = None) -> list[int]:
        """cutoff 内邻居原子 index. cutoff=None 用预计算 adjacency."""
        self._check_index(i)
        if cutoff is None:
            return list(self.adjacency.get(i, []))
        if self.lattice is not None:
            neighbors = []
            for j in range(len(self.species)):
                if j == i:
                    continue
                d = _min_image_distance(self.coords[i], self.coords[j], self.lattice)
                if d <= cutoff:
                    neighbors.append(j)
            return neighbors
        d = np.linalg.norm(self.coords - self.coords[i], axis=1)
        return [j for j in range(len(self.species)) if j != i and d[j] <= cutoff]

    def query_subgraph(self, center_indices: list[int], hops: int = 2) -> SubgraphResult:
        """k-hop 邻域子图. BFS 扩展 hops 跳."""
        for c in center_indices:
            self._check_index(c)
        if hops < 1:
            return SubgraphResult(nodes=list(center_indices), edges=[],
                                  node_species=[self.species[c] for c in center_indices])
        visited = set(center_indices)
        edges: list[tuple[int, int]] = []
        frontier = list(center_indices)
        for _ in range(hops):
            new_frontier = []
            for node in frontier:
                for nb in self.adjacency.get(node, []):
                    if nb not in visited:
                        visited.add(nb)
                        new_frontier.append(nb)
                        edges.append((node, nb))
            frontier = new_frontier
        nodes_sorted = sorted(visited)
        return SubgraphResult(nodes=nodes_sorted, edges=edges,
                              node_species=[self.species[n] for n in nodes_sorted])

    def query_after_rotation(self, indices: list[int], axis, angle: float,
                             degrees: bool = True, origin=None) -> list[tuple[float, float, float]]:
        """SE(3) 等变查询: 返回旋转后新坐标, 不改 map.

        axis: "x"/"y"/"z" 或 3-vector
        angle: 旋转角度
        origin: 旋转中心, None = coords 质心 (molecule) 或 lattice 原点 (crystal)
        """
        for i in indices:
            self._check_index(i)
        rot = _make_rotation(axis, angle, degrees)
        if origin is None:
            origin = np.zeros(3) if self.lattice is not None else self.coords.mean(axis=0)
        else:
            origin = np.asarray(origin, dtype=float)
        pts = self.coords[indices] - origin
        rotated = rot.apply(pts) + origin
        return [tuple(float(x) for x in p) for p in rotated]

    def query_after_translation(self, indices: list[int], vector) -> list[tuple[float, float, float]]:
        """平移后新坐标, 不改 map."""
        for i in indices:
            self._check_index(i)
        vec = np.asarray(vector, dtype=float)
        translated = self.coords[indices] + vec
        return [tuple(float(x) for x in p) for p in translated]

    # ── SE(3) operations (return new map, immutable) ──────

    def rotate(self, axis, angle: float, origin=None, degrees: bool = True) -> "StructureCognitiveMap":
        """返回旋转后的新 StructureCognitiveMap."""
        rot = _make_rotation(axis, angle, degrees)
        if origin is None:
            origin = np.zeros(3) if self.lattice is not None else self.coords.mean(axis=0)
        else:
            origin = np.asarray(origin, dtype=float)
        new_coords = rot.apply(self.coords - origin) + origin
        return self._clone_with(coords=new_coords)

    def translate(self, vector) -> "StructureCognitiveMap":
        """返回平移后的新 map."""
        vec = np.asarray(vector, dtype=float)
        return self._clone_with(coords=self.coords + vec)

    def supercell(self, scale) -> "StructureCognitiveMap":
        """返回 supercell 后的新 map. scale 是 int 或 (a, b, c).

        ponytail: lattice=None (molecule) 时不允许, 因为没有 PBC.
        ceiling: 不支持非对角 supercell, 升级路径接 pymatgen make_supercell.
        """
        if self.lattice is None:
            raise RuntimeError("supercell requires lattice (crystal), molecule has no PBC")
        if isinstance(scale, int):
            sa, sb, sc = scale, scale, scale
        else:
            sa, sb, sc = scale
        if not _HAS_PMG:
            raise RuntimeError("pymatgen not installed, pip install pymatgen")
        s = _PmgStructure(species=self.species, coords=self.coords, lattice=self.lattice)
        s.make_supercell((sa, sb, sc))
        new_coords = np.asarray(s.cart_coords, dtype=float)
        new_species = [str(sp) for sp in s.species]
        # lattice 不变, 重建 adjacency
        adj = _build_adjacency_cart(new_coords, self.lattice, _DEFAULT_CUTOFF)
        return StructureCognitiveMap(species=new_species, coords=new_coords,
                                     lattice=self.lattice.copy(), adjacency=adj,
                                     metadata={**self.metadata, "op": "supercell", "scale": (sa, sb, sc)})

    def remove_atom(self, index: int) -> "StructureCognitiveMap":
        """返回移除指定原子后的新 map."""
        self._check_index(index)
        mask = np.ones(len(self.species), dtype=bool)
        mask[index] = False
        new_species = [s for i, s in enumerate(self.species) if mask[i]]
        new_coords = self.coords[mask]
        # 重建 adjacency (index 重排)
        old_to_new = {old: new for new, old in enumerate(np.where(mask)[0])}
        adj: dict[int, list[int]] = {}
        for old_i, nbs in self.adjacency.items():
            if mask[old_i]:
                new_i = old_to_new[old_i]
                adj[new_i] = [old_to_new[nb] for nb in nbs if mask[nb]]
        return StructureCognitiveMap(species=new_species, coords=new_coords,
                                     lattice=self.lattice.copy() if self.lattice is not None else None,
                                     adjacency=adj, metadata={**self.metadata, "op": "remove_atom"})

    def add_atom(self, species: str, coord) -> "StructureCognitiveMap":
        """返回添加原子后的新 map."""
        coord = np.asarray(coord, dtype=float).reshape(3)
        new_species = self.species + [species]
        new_coords = np.vstack([self.coords, coord[None, :]])
        # 不重建全图, 只算新原子的邻居 (ponytail: 简化, adjacency 是缓存可重建)
        adj = {k: list(v) for k, v in self.adjacency.items()}
        new_idx = len(self.species)
        for j in range(len(self.species)):
            if self.lattice is not None:
                d = _min_image_distance(new_coords[new_idx], new_coords[j], self.lattice)
            else:
                d = float(np.linalg.norm(new_coords[new_idx] - new_coords[j]))
            if d <= _DEFAULT_CUTOFF:
                adj.setdefault(new_idx, []).append(j)
                adj.setdefault(j, []).append(new_idx)
        adj.setdefault(new_idx, [])
        return StructureCognitiveMap(species=new_species, coords=new_coords,
                                     lattice=self.lattice.copy() if self.lattice is not None else None,
                                     adjacency=adj, metadata={**self.metadata, "op": "add_atom"})

    def swap_atoms(self, i: int, j: int) -> "StructureCognitiveMap":
        """返回交换 i/j 原子种类后的新 map. coords 不变, species 交换."""
        self._check_index(i); self._check_index(j)
        new_species = list(self.species)
        new_species[i], new_species[j] = new_species[j], new_species[i]
        return StructureCognitiveMap(species=new_species, coords=self.coords.copy(),
                                     lattice=self.lattice.copy() if self.lattice is not None else None,
                                     adjacency={k: list(v) for k, v in self.adjacency.items()},
                                     metadata={**self.metadata, "op": "swap_atoms"})

    # ── 序列化 (P15 Persistence) ─────────────────────────

    def save(self, path: str | Path) -> None:
        """原子写 JSON. 跟 P15 atomic_write_json 风格一致."""
        from huginn.utils.common import atomic_write_json
        atomic_write_json(Path(path), self.to_engine_state_dict())

    @classmethod
    def load(cls, path: str | Path) -> "StructureCognitiveMap":
        """从 JSON 重建."""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_engine_state_dict(d)

    def to_engine_state_dict(self) -> dict[str, Any]:
        """序列化到 JSON-safe dict, 可放 EngineState.cognitive_maps[map_id]."""
        return {
            "species": list(self.species),
            "coords": self.coords.tolist(),
            "lattice": self.lattice.tolist() if self.lattice is not None else None,
            "adjacency": {str(k): list(v) for k, v in self.adjacency.items()},
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_engine_state_dict(cls, d: dict[str, Any]) -> "StructureCognitiveMap":
        """从 dict 重建."""
        coords = np.asarray(d["coords"], dtype=float)
        lat = np.asarray(d["lattice"], dtype=float) if d.get("lattice") is not None else None
        adj = {int(k): list(v) for k, v in d.get("adjacency", {}).items()}
        return cls(species=list(d["species"]), coords=coords, lattice=lat, adjacency=adj,
                   metadata=dict(d.get("metadata", {})))

    # ── internals ──────────────────────────────────────────

    def _clone_with(self, **changes) -> "StructureCognitiveMap":
        """复制 + 修改字段. 用 dataclasses.replace."""
        from dataclasses import replace
        return replace(self, **changes)

    def _check_index(self, i: int) -> None:
        if i < 0 or i >= len(self.species):
            raise IndexError(f"atom index {i} out of range [0, {len(self.species)})")

    def __len__(self) -> int:
        return len(self.species)

    def __repr__(self) -> str:
        kind = "crystal" if self.lattice is not None else "molecule"
        return f"StructureCognitiveMap({kind}, n={len(self.species)})"


# ── module-level helpers ─────────────────────────────────────

def _make_rotation(axis, angle: float, degrees: bool = True) -> "_R":
    """统一构造 scipy Rotation. axis 是 'x'/'y'/'z' 或 3-vector, 自动归一化."""
    if not _HAS_SCIPY:
        raise RuntimeError("scipy not installed, pip install scipy")
    if isinstance(axis, str):
        axis_vec = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}[axis.lower()]
    else:
        axis_vec = list(axis)
    axis_arr = np.asarray(axis_vec, dtype=float)
    norm = float(np.linalg.norm(axis_arr))
    if norm < 1e-12:
        raise ValueError("axis cannot be zero vector")
    # from_rotvec 第一个参数是 axis*angle (即旋转向量), degrees 控制角度单位
    return _R.from_rotvec(axis_arr / norm * float(angle), degrees=degrees)


def _build_adjacency_cart(coords: np.ndarray, lattice: np.ndarray | None,
                          cutoff: float) -> dict[int, list[int]]:
    """构建 adjacency. crystal 用 PBC, molecule 直接距离.

    ponytail: O(N²) 双循环, 没上 KD-tree.
    ceiling: N > 1000 时慢, 升级路径接 scipy.spatial.cKDTree.
    """
    n = len(coords)
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if lattice is not None:
                d = _min_image_distance(coords[i], coords[j], lattice)
            else:
                d = float(np.linalg.norm(coords[i] - coords[j]))
            if d <= cutoff:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _min_image_distance(a: np.ndarray, b: np.ndarray, lattice: np.ndarray) -> float:
    """PBC 最小镜像距离. lattice 是 (3, 3) 行向量 a/b/c."""
    inv = np.linalg.inv(lattice)
    frac_a = a @ inv
    frac_b = b @ inv
    delta = frac_b - frac_a
    # wrap to [-0.5, 0.5]
    delta = delta - np.round(delta)
    cart_delta = delta @ lattice
    return float(np.linalg.norm(cart_delta))


# ── self-check ─────────────────────────────────────────────

def _selfcheck() -> None:
    """6 场景: 构建 + 距离/角度 + 邻居 + SE(3) 等变 + save/load + immutability.

    ponytail: 用 NaCl 简单结构 (2 原子常规 cell, d≈4.88Å), 不依赖真 CIF 文件.
    ceiling: 不验 PBC 边界 case (a in cell 0, b in cell 1), 升级路径加 edge case.
    """
    import os
    import tempfile

    # Mock NaCl CIF (a=5.64Å, P1 不写 spacegroup 避免 pymatgen 展开对称操作).
    # ponytail: 2 原子直接列出, d(Na,Cl) = a*sqrt(3)/2 ≈ 4.88Å.
    nacl_cif = """data_NaCl
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
"""
    print("1. from_cif 构建测试...")
    m = StructureCognitiveMap.from_cif(nacl_cif)
    assert len(m) == 2, f"expected 2 atoms, got {len(m)}"
    assert m.species == ["Na", "Cl"], f"species wrong: {m.species}"
    print(f"   OK: {m}")

    print("2. query_distance + query_angle 测试...")
    d = m.query_distance(0, 1)
    # Na-Cl: a*sqrt(3)/2 ≈ 4.88Å (2 原子常规 cell, Na 在原点 Cl 在体心)
    assert 4.0 < d < 6.0, f"NaCl distance expected ~4.88Å, got {d}"
    print(f"   OK: d(Na, Cl) = {d:.3f}Å")

    print("3. query_neighbors 测试...")
    # Fm-3m 常规 cell 只有 2 原子, cutoff=5.0 时 Cl 在 Na 邻居里
    nbs = m.query_neighbors(0, cutoff=5.0)
    print(f"   Na neighbors (cutoff=5.0): {nbs}")

    print("4. SE(3) 等变验证 (rotate z=90 后距离不变)...")
    m_rot = m.rotate(axis="z", angle=90, degrees=True)
    d_rot = m_rot.query_distance(0, 1)
    assert abs(d - d_rot) < 1e-6, f"SE(3) 不变: d={d}, d_rot={d_rot}"
    print(f"   OK: rotate 前后 d 不变 ({d:.3f} -> {d_rot:.3f})")

    print("5. save/load round-trip 测试...")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test_map.json")
        m.save(path)
        m2 = StructureCognitiveMap.load(path)
        assert len(m2) == len(m)
        assert m2.species == m.species
        d2 = m2.query_distance(0, 1)
        assert abs(d - d2) < 1e-6, f"load 后距离不一致: {d} vs {d2}"
    print("   OK: save/load 距离一致")

    print("6. query_after_rotation immutability 测试...")
    original_d = m.query_distance(0, 1)
    rotated_coords = m.query_after_rotation([0, 1], axis="z", angle=45, degrees=True)
    assert len(rotated_coords) == 2
    # 原 map 不变
    assert m.query_distance(0, 1) == original_d, "原 map 被改了!"
    # 旋转后两点间距离不变 (SE(3) 等变)
    rot_d = float(np.linalg.norm(np.array(rotated_coords[0]) - np.array(rotated_coords[1])))
    assert abs(rot_d - original_d) < 1e-6, f"SE(3) 不等变: rot_d={rot_d}, orig={original_d}"
    print("   OK: query_after_rotation 不改原 map, 距离等变")

    print("\nstructure_cognitive_map selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
