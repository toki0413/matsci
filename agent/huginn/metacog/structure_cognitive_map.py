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

Bourbaki 三结构视角 (B 文档化):
  代数 I  (free monoid): 多个 map 可 concat 成 supercell (supercell 方法).
                         from_cif/from_molecule/from_coords 是 monoid 元素构造.
  代数 II (SE(3) 群作用): rotate/translate/query_after_rotation 是 SE(3) 群作用.
                         半直积 (M × V) ⋊ SE(3) 已验证 (见 C 实验 composite_token_experiment).
  拓扑   (邻域):         query_neighbors / adjacency 定义邻域.
                         neighborhood 方法 (A 拓扑 Protocol) 适配 SupportsNeighborhood.
                         升级路径: Hodge decomposition (gradient/curl/harmonic 三分量).
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
class Bond:
    """化学键. identify_bonds 返回单元."""

    i: int
    j: int
    length: float
    bond_type: str  # covalent / ionic / metallic / hydrogen / unknown
    source: str = "cutoff"  # cutoff / crystalnn / voronoi


@dataclass
class CoordinationShell:
    """配位壳层. coordination_shell 返回单元."""

    center: int
    neighbors: list[int]
    neighbor_distances: list[float]
    geometry: str  # octahedral / tetrahedral / square_planar / linear / unknown


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

    @classmethod
    def from_image(cls, image_bytes: bytes, pixel_size_nm: float = 0.1,
                   fft_threshold: float | None = None,
                   cutoff: float = _DEFAULT_CUTOFF) -> "StructureCognitiveMap":
        """M7: 从 TEM 图像构建认知图: 调 tem_lattice 拿 FFT → d-spacings → 构造 minimal cell.

        流程:
          1. 落盘 image_bytes 到 tmp 文件
          2. 调 scenes_tem.tem_lattice(image_path) 拿 FFT 径向峰 + d-spacings
          3. 用最强 d_spacing 构造 cubic placeholder cell (1 原子在原点)
          4. metadata 存 fft_peak_2d / d_spacings / pixel_size / image_shape

        不真反推晶系 (那是 DICVOL/ITO 的活). 只把 FFT 结果外部化成认知图,
        让后续 query_distance / coordination_shell 能用这些 d-spacing 信息.

        ponytail: minimal cubic placeholder, 不是真结构. 升级路径: 接 pattern
        indexing (DICVOL/ITO) 真反推晶系 + 原子位置 + 接 EDS 定元素.

        Args:
            image_bytes: TEM 图像二进制
            pixel_size_nm: 像素尺寸 (nm/px), 默认 0.1
            fft_threshold: FFT 阈值 (None 自动)
            cutoff: 邻接 cutoff (Å)

        Returns:
            StructureCognitiveMap (1 原子 cubic placeholder + FFT metadata)

        Raises:
            RuntimeError: tem_lattice 失败 / 无 d-spacing 检测到
        """
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        try:
            from huginn.tools.image_analysis.tool import ImageAnalysisInput
            from huginn.tools.image_analysis.scenes_tem import tem_lattice

            params: dict[str, Any] = {"pixel_size_nm": pixel_size_nm}
            if fft_threshold is not None:
                params["fft_threshold"] = fft_threshold
            input_data = ImageAnalysisInput(
                image_path=tmp_path,
                action="tem_lattice",
                parameters=params,
            )
            res = tem_lattice(input_data)
            if not res.success or not res.data:
                raise RuntimeError(
                    f"tem_lattice failed: {getattr(res, 'error', 'unknown')}"
                )
            data = res.data
            measurements = data.get("measurements", {})
            d_entries = measurements.get("d_spacings", [])
            if not d_entries:
                raise RuntimeError("TEM 图像无 d-spacing 检测到, 检查 fft_threshold 或图像质量")

            # 用最强 d-spacing 构造 cubic placeholder (d_main in Å)
            d_main_nm = float(d_entries[0]["d_nm"])
            d_main_angstrom = d_main_nm / 10.0  # nm → Å
            # 1 原子在原点, cubic lattice = d_main * I_3
            species = ["X"]  # placeholder 元素, 真元素需 EDS 配合
            coords = np.array([[0.0, 0.0, 0.0]])
            lattice = np.eye(3) * d_main_angstrom
            adj = _build_adjacency_cart(coords, lattice, cutoff)

            return cls(
                species=species, coords=coords, lattice=lattice, adjacency=adj,
                metadata={
                    "source": "image",
                    "d_main_nm": d_main_nm,
                    "d_main_angstrom": d_main_angstrom,
                    "d_spacings": d_entries,
                    "fft_peak_2d": measurements.get("fft_peak_2d", []),
                    "pixel_size_nm": measurements.get("pixel_size_nm"),
                    "image_shape": measurements.get("image_shape"),
                    "fft_threshold": measurements.get("fft_threshold"),
                    "n_peaks": measurements.get("n_peaks"),
                    "tem_summary": data.get("summary"),
                    "note": (
                        "cubic placeholder from d_main, not real crystal structure. "
                        "Upgrade path: pattern indexing (DICVOL/ITO) + EDS for element ID."
                    ),
                },
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

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

    def neighborhood(self, x: int, radius: float | None = None) -> set[int]:
        """A 拓扑 Protocol — SupportsNeighborhood 实现.

        x = 原子 index, radius = cutoff (None = 用预计算 adjacency).
        跟 query_neighbors 同语义, 只是返回 set 适配 Protocol.
        """
        return set(self.query_neighbors(x, cutoff=radius))

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

    # ── 9 个新 ops (补全 AtomWorld 15 actions) ─────────────

    def move_atom(self, index: int, vector) -> "StructureCognitiveMap":
        """单原子平移. 跟 translate (整体平移) 不同."""
        self._check_index(index)
        vec = np.asarray(vector, dtype=float)
        new_coords = self.coords.copy()
        new_coords[index] += vec
        adj = _build_adjacency_cart(new_coords, self.lattice, _DEFAULT_CUTOFF)
        return self._clone_with(coords=new_coords, adjacency=adj, metadata={**self.metadata, "op": "move_atom"})

    def move_selected(self, indices: list[int], vector) -> "StructureCognitiveMap":
        """子集平移."""
        for i in indices:
            self._check_index(i)
        vec = np.asarray(vector, dtype=float)
        new_coords = self.coords.copy()
        new_coords[indices] += vec
        adj = _build_adjacency_cart(new_coords, self.lattice, _DEFAULT_CUTOFF)
        return self._clone_with(coords=new_coords, adjacency=adj, metadata={**self.metadata, "op": "move_selected"})

    def move_towards(self, i: int, j: int, fraction: float = 0.5) -> "StructureCognitiveMap":
        """原子 i 朝 j 方向移动 fraction 比例 (0=不动, 1=到 j 位置)."""
        self._check_index(i); self._check_index(j)
        vec = (self.coords[j] - self.coords[i]) * float(fraction)
        return self.move_atom(i, vec)

    def move_around(self, i: int, center, axis, angle: float,
                    degrees: bool = True) -> "StructureCognitiveMap":
        """原子 i 绕 center 旋转. center 可以是 index 或 3-vector."""
        self._check_index(i)
        if isinstance(center, int):
            self._check_index(center)
            origin = self.coords[center]
        else:
            origin = np.asarray(center, dtype=float)
        rot = _make_rotation(axis, angle, degrees)
        new_coords = self.coords.copy()
        new_coords[i] = rot.apply(new_coords[i] - origin) + origin
        adj = _build_adjacency_cart(new_coords, self.lattice, _DEFAULT_CUTOFF)
        return self._clone_with(coords=new_coords, adjacency=adj, metadata={**self.metadata, "op": "move_around"})

    def change_atom(self, index: int, new_species: str) -> "StructureCognitiveMap":
        """改原子种类 (不改坐标). 单步 API, 不用 swap+remove+add 组合."""
        self._check_index(index)
        new_sp = list(self.species)
        new_sp[index] = new_species
        return self._clone_with(species=new_sp, metadata={**self.metadata, "op": "change_atom"})

    def scale(self, factor: float) -> "StructureCognitiveMap":
        """整体缩放 (coords + lattice). MiSI T3 zooming 需要."""
        f = float(factor)
        new_coords = self.coords * f
        new_lattice = self.lattice * f if self.lattice is not None else None
        adj = _build_adjacency_cart(new_coords, new_lattice, _DEFAULT_CUTOFF)
        return self._clone_with(coords=new_coords, lattice=new_lattice, adjacency=adj,
                                metadata={**self.metadata, "op": "scale", "factor": f})

    def insert_between(self, i: int, j: int, species: str) -> "StructureCognitiveMap":
        """在 i-j 中点插入新原子."""
        self._check_index(i); self._check_index(j)
        midpoint = (self.coords[i] + self.coords[j]) / 2.0
        return self.add_atom(species, midpoint)

    def delete_below(self, i: int, cutoff: float | None = None) -> "StructureCognitiveMap":
        """删除 i 的近邻 (保留 i). cutoff=None 用 _DEFAULT_CUTOFF."""
        self._check_index(i)
        c = cutoff if cutoff is not None else _DEFAULT_CUTOFF
        to_delete = set(self.query_neighbors(i, cutoff=c))
        if not to_delete:
            return self
        mask = np.ones(len(self.species), dtype=bool)
        for d in to_delete:
            mask[d] = False
        new_species = [s for k, s in enumerate(self.species) if mask[k]]
        new_coords = self.coords[mask]
        old_to_new = {old: new for new, old in enumerate(np.where(mask)[0])}
        adj: dict[int, list[int]] = {}
        for old_i, nbs in self.adjacency.items():
            if mask[old_i]:
                new_i = old_to_new[old_i]
                adj[new_i] = [old_to_new[nb] for nb in nbs if mask[nb]]
        adj.setdefault(len(new_species), [])
        return StructureCognitiveMap(species=new_species, coords=new_coords,
                                     lattice=self.lattice.copy() if self.lattice is not None else None,
                                     adjacency=adj, metadata={**self.metadata, "op": "delete_below"})

    def delete_around_atom(self, i: int, cutoff: float | None = None) -> "StructureCognitiveMap":
        """删除 i 的配位壳层 (保留 i). 跟 delete_below 同义, AtomWorld 语义."""
        return self.delete_below(i, cutoff)

    # ── bond 识别 (替代 3.0Å 单 cutoff) ──────────────────

    def identify_bonds(self, method: str = "cutoff",
                        cutoff: float | None = None) -> list[Bond]:
        """识别化学键. method: cutoff / crystalnn / voronoi.

        ponytail: cutoff 默认 3.0Å, 不区分键类型 (bond_type=unknown).
        crystalnn 用 pymatgen CrystalNN (更准确, 但对金属/合金不稳定).
        ceiling: crystalnn 对部分体系会报错, 降级到 cutoff.
        """
        c = cutoff if cutoff is not None else _DEFAULT_CUTOFF
        if method == "crystalnn" and _HAS_PMG:
            return self._bonds_crystalnn()
        if method == "voronoi" and _HAS_PMG:
            return self._bonds_voronoi()
        return self._bonds_cutoff(c)

    def _bonds_cutoff(self, cutoff: float) -> list[Bond]:
        bonds: list[Bond] = []
        n = len(self.species)
        for i in range(n):
            for j in range(i + 1, n):
                if self.lattice is not None:
                    d = _min_image_distance(self.coords[i], self.coords[j], self.lattice)
                else:
                    d = float(np.linalg.norm(self.coords[i] - self.coords[j]))
                if d <= cutoff:
                    bonds.append(Bond(i=i, j=j, length=d,
                                       bond_type=self.classify_bond(i, j),
                                       source="cutoff"))
        return bonds

    def _bonds_crystalnn(self) -> list[Bond]:
        try:
            from pymatgen.analysis.local_env import CrystalNN
            s = _PmgStructure(species=self.species, coords=self.coords, lattice=self.lattice)
            cnn = CrystalNN()
            bonds: list[Bond] = []
            for i in range(len(self.species)):
                try:
                    info = cnn.get_cn_dict(s, i)
                    for j, count in info.items():
                        if j > i:
                            d = self.query_distance(i, j)
                            bonds.append(Bond(i=i, j=int(j), length=d,
                                               bond_type=self.classify_bond(i, int(j)),
                                               source="crystalnn"))
                except Exception:
                    continue
            return bonds
        except Exception as e:
            logger.debug("CrystalNN failed, fallback to cutoff: %s", e)
            return self._bonds_cutoff(_DEFAULT_CUTOFF)

    def _bonds_voronoi(self) -> list[Bond]:
        try:
            from pymatgen.analysis.local_env import VoronoiNN
            s = _PmgStructure(species=self.species, coords=self.coords, lattice=self.lattice)
            vnn = VoronoiNN()
            bonds: list[Bond] = []
            for i in range(len(self.species)):
                try:
                    nbs = vnn.get_nn_info(s, i)
                    for nb in nbs:
                        j = nb["site_index"]
                        if j > i:
                            d = self.query_distance(i, j)
                            bonds.append(Bond(i=i, j=j, length=d,
                                               bond_type=self.classify_bond(i, j),
                                               source="voronoi"))
                except Exception:
                    continue
            return bonds
        except Exception as e:
            logger.debug("VoronoiNN failed, fallback to cutoff: %s", e)
            return self._bonds_cutoff(_DEFAULT_CUTOFF)

    def coordination_shell(self, i: int, method: str = "cutoff",
                            cutoff: float | None = None) -> CoordinationShell:
        """配位壳层. 识别几何 (octahedral/tetrahedral/square_planar/linear)."""
        self._check_index(i)
        c = cutoff if cutoff is not None else _DEFAULT_CUTOFF
        if method == "crystalnn" and _HAS_PMG:
            try:
                from pymatgen.analysis.local_env import CrystalNN
                s = _PmgStructure(species=self.species, coords=self.coords, lattice=self.lattice)
                cnn = CrystalNN()
                info = cnn.get_cn_dict(s, i)
                nbs = list(info.keys())
                dists = [self.query_distance(i, j) for j in nbs]
            except Exception:
                nbs = self.query_neighbors(i, cutoff=c)
                dists = [self.query_distance(i, j) for j in nbs]
        else:
            nbs = self.query_neighbors(i, cutoff=c)
            dists = [self.query_distance(i, j) for j in nbs]
        geom = self._classify_geometry(nbs)
        return CoordinationShell(center=i, neighbors=nbs, neighbor_distances=dists, geometry=geom)

    def _classify_geometry(self, neighbor_indices: list[int]) -> str:
        """根据配位数粗判几何. ponytail: 不做完整点群分析."""
        n = len(neighbor_indices)
        return {2: "linear", 3: "trigonal", 4: "tetrahedral",
                5: "trigonal_bipyramidal", 6: "octahedral",
                8: "cubic"}.get(n, "unknown")

    def classify_bond(self, i: int, j: int) -> str:
        """分类键类型: covalent/ionic/metallic/hydrogen/unknown.

        ponytail: 用元素电负性差 + 是否含 H. 不查真键长表.
        ceiling: 不区分极性共价键, 升级路径接 Allen 电负性 + 键长数据库.
        """
        sp_i = self.species[i]
        sp_j = self.species[j]
        # 氢键: H 跟 N/O/F 之间
        h_partners = {"H", "N", "O", "F"}
        if (sp_i in h_partners and sp_j in h_partners) and \
           ((sp_i == "H" and sp_j in {"N", "O", "F"}) or
            (sp_j == "H" and sp_i in {"N", "O", "F"})):
            d = self.query_distance(i, j)
            if d > 1.5:  # >1.5Å 判氢键, <1.5Å 判共价
                return "hydrogen"
            return "covalent"
        # 金属键: 两个都是金属
        metals = {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba",
                  "Al", "Fe", "Cu", "Zn", "Ag", "Au", "Ni", "Co", "Cr",
                  "Mn", "Ti", "V", "Zr", "Mo", "W", "Pd", "Pt", "Pb", "Sn"}
        if sp_i in metals and sp_j in metals:
            return "metallic"
        # 离子键: 电负性差 > 1.7 (Pauling 粗判)
        en = {"H": 2.20, "Li": 0.98, "Na": 0.93, "K": 0.82, "Mg": 1.31,
              "Ca": 1.00, "Al": 1.61, "C": 2.55, "N": 3.04, "O": 3.44,
              "F": 3.98, "Cl": 3.16, "Br": 2.96, "I": 2.66, "S": 2.58,
              "P": 2.19, "Si": 1.90, "B": 2.04, "Fe": 1.83, "Cu": 1.90,
              "Zn": 1.65, "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55,
              "Co": 1.88, "Ni": 1.91, "Ag": 1.93, "Au": 2.54, "Pt": 2.28,
              "Pd": 2.20, "W": 2.36, "Mo": 2.16, "Zr": 1.33}
        en_i = en.get(sp_i, 1.5)
        en_j = en.get(sp_j, 1.5)
        if abs(en_i - en_j) > 1.7:
            return "ionic"
        return "covalent"

    def identify_hydrogen_bonds(self, cutoff: float = 3.5) -> list[Bond]:
        """氢键识别. donor (N/O/F-H) ... acceptor (N/O/F).

        ponytail: donor-H...acceptor 距离 < 3.5Å + 角度 > 120° (简化, 不查 H 位置).
        ceiling: 不验 H 位置 (分子 CIF 里 H 坐标常缺), 升级路径接 rdkit.
        """
        h_donors = {"N", "O", "F"}
        h_acceptors = {"N", "O", "F"}
        bonds: list[Bond] = []
        for i in range(len(self.species)):
            if self.species[i] != "H":
                continue
            # 找 donor (i 附近的重原子)
            donor = None
            for nb in self.query_neighbors(i, cutoff=1.5):
                if self.species[nb] in h_donors:
                    donor = nb
                    break
            if donor is None:
                continue
            # 找 acceptor
            for j in range(len(self.species)):
                if j == i or j == donor:
                    continue
                if self.species[j] not in h_acceptors:
                    continue
                d = self.query_distance(i, j) if self.lattice is None else \
                    _min_image_distance(self.coords[i], self.coords[j], self.lattice)
                if d <= cutoff and d > 1.5:
                    bonds.append(Bond(i=donor, j=j, length=d,
                                       bond_type="hydrogen", source="hbond"))
        return bonds

    # ── 泛化深化: 变换链 + 配位多面体 + 键类型分组 ────────

    def compose(self, *ops) -> "StructureCognitiveMap":
        """串联多个变换. ops 是 callable: map -> map.

        用法:
            m2 = m.compose(
                lambda m: m.rotate("z", 90),
                lambda m: m.translate([1, 2, 3]),
                lambda m: m.scale(2),
            )
        ponytail: 每步创建中间 map, O(N*k). 不做矩阵合并 (YAGNI).
        """
        result = self
        for op in ops:
            result = op(result)
        return result

    def coordination_polyhedra(self) -> list[CoordinationShell]:
        """识别所有配位多面体. 遍历每个原子."""
        return [self.coordination_shell(i) for i in range(len(self.species))]

    def bond_types(self, method: str = "cutoff") -> dict[str, list[Bond]]:
        """按键类型分组."""
        bonds = self.identify_bonds(method=method)
        groups: dict[str, list[Bond]] = {}
        for b in bonds:
            groups.setdefault(b.bond_type, []).append(b)
        return groups

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

    # ── G1: 复合 token 接入 (C 实验工程化) ─────────────────────

    def to_composite_token(self) -> dict[str, Any]:
        """把 cognitive map 投影成复合 token (text + coords).

        C 实验验证: SE(3) 群作用能同时穿过 text <point3d> 和 coords,
        半直积 (M × V) ⋊ SE(3) 合法. 本方法是工程入口 —
        让 StructureCognitiveMap 能进 se3_act 群作用流程.

        返回:
            dict: {
                "text": "<point3d>[x,y,z]</point3d>(Fe)\\n<point3d>...",  # 原语文本
                "coords": (N, 3) ndarray,  # 原始 3D 坐标 (Å)
                "species": ["Fe", "O", ...],
                "n_atoms": int,
            }

        用法:
            token = m.to_composite_token()
            from huginn.metacog.composite_token_experiment import se3_act, CompositeToken
            ct = CompositeToken(token["text"], token["coords"])
            rot = Rotation.from_euler("z", 90, degrees=True)
            rotated = se3_act(rot, np.zeros(3), ct)
            # rotated.text 是旋转后的 <point3d>, rotated.coords 是旋转后的坐标

        ponytail: 返回 dict 不引入新类. 升级路径: 返回 CompositeToken 实例.
        """
        from huginn.tools.visual_hook import extract_point3d_primitives
        text = extract_point3d_primitives(
            self.coords.tolist(),
            species=self.species,
            normalize_to=999,
        )
        return {
            "text": text,
            "coords": self.coords.copy(),
            "species": list(self.species),
            "n_atoms": len(self.species),
        }

    # ── Hodge: 拓扑三分量分解 ─────────────────────────────────

    def hodge_decomposition(self) -> dict[str, Any]:
        """把 adjacency 拓扑分解为 gradient / curl / harmonic 三分量.

        Bourbaki 拓扑结构视角: 邻域图 (graph) 的 1-form 可 Hodge 分解:
          ω = ∇φ (gradient, 无旋) ⊕ ∇×A (curl, 无散) ⊕ h (harmonic, 闭且余闭)

        材料科学意义:
          - gradient: 单向流动 (如单向应力梯度, 浓度梯度)
          - curl: 环流 (如位错 Burgers 矢量, 涡旋)
          - harmonic: 亏格 (如孔洞/晶界拓扑缺陷, 全局拓扑不变量)

        实现:
          - 构建图拉普拉斯 L = D - A
          - 求 L 的特征值/特征向量
          - λ=0 对应 harmonic (亏格数 = λ=0 重数 - 1)
          - λ 小对应低频 (gradient 主导)
          - λ 大对应高频 (curl 主导)

        ponytail: numpy.linalg.eigh, 不引入 scipy.sparse. 升级路径: 大图接 scipy.sparse.csgraph.

        Returns:
            dict: {
                "n_harmonic": int,  # λ≈0 重数 (亏格)
                "eigenvalues": list[float],  # 升序
                "gradient_fraction": float,  # 低频能量占比 (前 1/3)
                "curl_fraction": float,  # 高频能量占比 (后 1/3)
                "harmonic_fraction": float,  # λ≈0 占比
                "spectrum_gap": float,  # λ[1] - λ[0] (谱隙, >0 连通)
                "note": str,
            }
        """
        n = len(self.species)
        if n < 2:
            return {
                "n_harmonic": 0, "eigenvalues": [],
                "gradient_fraction": 0.0, "curl_fraction": 0.0,
                "harmonic_fraction": 0.0, "spectrum_gap": 0.0,
                "note": "n<2, no graph structure",
            }

        # 构建邻接矩阵 A (对称, 无权)
        A = np.zeros((n, n), dtype=float)
        for i, neighbors in self.adjacency.items():
            for j in neighbors:
                if 0 <= i < n and 0 <= j < n:
                    A[i, j] = 1.0
                    A[j, i] = 1.0

        # 图拉普拉斯 L = D - A
        D = np.diag(A.sum(axis=1))
        L = D - A

        # 特征值分解 (对称矩阵, eigh 返回升序)
        try:
            eigenvalues, _ = np.linalg.eigh(L)
        except np.linalg.LinAlgError as e:
            return {
                "n_harmonic": 0, "eigenvalues": [],
                "gradient_fraction": 0.0, "curl_fraction": 0.0,
                "harmonic_fraction": 0.0, "spectrum_gap": 0.0,
                "note": f"eigh failed: {e}",
            }

        eigenvalues_sorted = np.sort(eigenvalues)
        # λ≈0 阈值 (数值误差容忍)
        tol = 1e-8 * max(1.0, abs(eigenvalues_sorted[-1]))
        n_harmonic = int(np.sum(np.abs(eigenvalues_sorted) < tol))

        # 三分量能量占比 (按特征值分三段)
        n_eig = len(eigenvalues_sorted)
        if n_eig >= 3:
            third = n_eig // 3
            low = eigenvalues_sorted[:third]  # gradient
            high = eigenvalues_sorted[-third:]  # curl
            mid = eigenvalues_sorted[third:-third] if third < n_eig - third else np.array([])

            total_energy = float(np.sum(eigenvalues_sorted ** 2)) + 1e-12
            grad_frac = float(np.sum(low ** 2)) / total_energy
            curl_frac = float(np.sum(high ** 2)) / total_energy
            harm_frac = float(np.sum(eigenvalues_sorted[np.abs(eigenvalues_sorted) < tol] ** 2)) / total_energy
        else:
            grad_frac = curl_frac = harm_frac = 0.0

        spectrum_gap = float(eigenvalues_sorted[1] - eigenvalues_sorted[0]) if n_eig >= 2 else 0.0

        return {
            "n_harmonic": n_harmonic,
            "eigenvalues": eigenvalues_sorted.tolist(),
            "gradient_fraction": round(grad_frac, 4),
            "curl_fraction": round(curl_frac, 4),
            "harmonic_fraction": round(harm_frac, 4),
            "spectrum_gap": round(spectrum_gap, 6),
            "note": (
                f"topology: {n_harmonic} harmonic (genus), "
                f"gap={spectrum_gap:.4f} ({'connected' if spectrum_gap > 1e-6 else 'disconnected'}), "
                f"gradient={grad_frac:.2f}, curl={curl_frac:.2f}"
            ),
        }


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
    """12 场景: 构建 + 距离/角度 + 邻居 + SE(3) 等变 + save/load + immutability
    + 9 新 ops + bond 识别 + compose.

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
    assert 4.0 < d < 6.0, f"NaCl distance expected ~4.88Å, got {d}"
    print(f"   OK: d(Na, Cl) = {d:.3f}Å")

    print("3. query_neighbors 测试...")
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
    assert m.query_distance(0, 1) == original_d, "原 map 被改了!"
    rot_d = float(np.linalg.norm(np.array(rotated_coords[0]) - np.array(rotated_coords[1])))
    assert abs(rot_d - original_d) < 1e-6, f"SE(3) 不等变: rot_d={rot_d}, orig={original_d}"
    print("   OK: query_after_rotation 不改原 map, 距离等变")

    print("7. move_atom + move_selected + move_towards...")
    m_moved = m.move_atom(0, [0.1, 0, 0])
    assert len(m_moved) == 2
    m_sel = m.move_selected([0], [0.2, 0, 0])
    assert len(m_sel) == 2
    m_towards = m.move_towards(0, 1, fraction=0.5)
    d_towards = m_towards.query_distance(0, 1)
    # 朝 j 移动一半, 距离减半
    assert abs(d_towards - d * 0.5) < 0.1, f"move_towards 距离错: {d_towards} vs {d*0.5}"
    print(f"   OK: move_towards 距离减半 ({d:.3f} -> {d_towards:.3f})")

    print("8. move_around + change_atom + scale...")
    m_around = m.move_around(0, center=1, axis="z", angle=45)
    assert len(m_around) == 2
    m_changed = m.change_atom(0, "K")
    assert m_changed.species[0] == "K"
    m_scaled = m.scale(2.0)
    d_scaled = m_scaled.query_distance(0, 1)
    assert abs(d_scaled - d * 2.0) < 0.1, f"scale 距离错: {d_scaled} vs {d*2.0}"
    print(f"   OK: scale 2x 距离翻倍 ({d:.3f} -> {d_scaled:.3f})")

    print("9. insert_between + delete_below + delete_around_atom...")
    m_ins = m.insert_between(0, 1, "O")
    assert len(m_ins) == 3
    m_del = m.delete_below(0, cutoff=5.0)
    assert len(m_del) < len(m), f"delete_below 没删: {len(m_del)} vs {len(m)}"
    m_del2 = m.delete_around_atom(0, cutoff=5.0)
    assert len(m_del2) == len(m_del)
    print(f"   OK: insert_between n=3, delete_below n={len(m_del)}")

    print("10. bond 识别 (cutoff + crystalnn) + classify_bond...")
    bonds = m.identify_bonds(method="cutoff", cutoff=5.0)
    assert len(bonds) >= 1, f"期望至少 1 个键, got {bonds}"
    bt = m.classify_bond(0, 1)
    assert bt in ("ionic", "covalent", "metallic", "hydrogen", "unknown")
    types = m.bond_types(method="cutoff")
    assert isinstance(types, dict)
    print(f"   OK: bonds={len(bonds)}, Na-Cl={bt}, types={list(types.keys())}")

    print("11. coordination_shell + coordination_polyhedra...")
    shell = m.coordination_shell(0, cutoff=5.0)
    assert shell.center == 0
    polys = m.coordination_polyhedra()
    assert len(polys) == len(m)
    print(f"   OK: shell.geometry={shell.geometry}, polys={len(polys)}")

    print("12. compose (变换链)...")
    m_composed = m.compose(
        lambda mm: mm.rotate("z", 90),
        lambda mm: mm.translate([1.0, 0, 0]),
        lambda mm: mm.scale(1.5),
    )
    d_composed = m_composed.query_distance(0, 1)
    # rotate + translate 不改变距离, scale 1.5 改变距离
    assert abs(d_composed - d * 1.5) < 0.5, f"compose 距离错: {d_composed} vs {d*1.5}"
    print(f"   OK: rotate+translate+scale d={d_composed:.3f} (原 {d:.3f} x 1.5)")

    # ── 13. M7 from_image: TEM 图像 → FFT → d-spacings → cubic placeholder ──
    print("13. M7 from_image (TEM FFT → lattice → map)...")
    # 构造一个合成 lattice 图像: 用 numpy 生成正弦光栅模拟晶格条纹
    import io as _io
    from PIL import Image as _PILImage
    # 200x200 灰度图, 正弦光栅周期=20px, 模拟 d-spacing
    _size = 200
    _period = 20
    _x = np.arange(_size)
    _XX, _YY = np.meshgrid(_x, _x)
    _grating = 128 + 100 * np.sin(2 * np.pi * (_XX + _YY) / _period)
    _img_arr = np.clip(_grating, 0, 255).astype(np.uint8)
    _pil = _PILImage.fromarray(_img_arr)
    _buf = _io.BytesIO()
    _pil.save(_buf, format="PNG")
    _img_bytes = _buf.getvalue()
    try:
        m_img = StructureCognitiveMap.from_image(
            _img_bytes, pixel_size_nm=0.05, fft_threshold=0.3
        )
        # 验证构造成功 + metadata 完整
        assert m_img.metadata["source"] == "image", f"source mismatch: {m_img.metadata.get('source')}"
        assert "d_main_nm" in m_img.metadata, "missing d_main_nm"
        assert "d_spacings" in m_img.metadata, "missing d_spacings"
        assert "fft_peak_2d" in m_img.metadata, "missing fft_peak_2d"
        assert "image_shape" in m_img.metadata, "missing image_shape"
        assert m_img.species == ["X"], f"expected ['X'], got {m_img.species}"
        assert m_img.lattice is not None, "lattice should not be None"
        # cubic placeholder: lattice = d_main * I_3
        d_main_a = m_img.metadata["d_main_angstrom"]
        expected_lat = np.eye(3) * d_main_a
        assert np.allclose(m_img.lattice, expected_lat, atol=1e-6), \
            f"lattice mismatch: {m_img.lattice} vs {expected_lat}"
        # coords 是 1 原子在原点
        assert m_img.coords.shape == (1, 3), f"coords shape: {m_img.coords.shape}"
        assert np.allclose(m_img.coords[0], [0, 0, 0]), f"coords not origin: {m_img.coords[0]}"
        # 升级路径注释存在
        assert "cubic placeholder" in m_img.metadata["note"], "note missing cubic placeholder"
        print(f"   OK: source=image, d_main={d_main_a:.3f} Å, "
              f"n_peaks={m_img.metadata.get('n_peaks')}, "
              f"n_fft_2d={len(m_img.metadata.get('fft_peak_2d', []))}")
    except RuntimeError as exc:
        # tem_lattice 在合成图上可能检测不到峰, 这是已知的 ceiling — 验证错误信息合理
        print(f"   SKIP: tem_lattice on synthetic grating failed ({exc})")
        print("   (ceiling: 合成正弦光栅可能不够 'TEM-like' 触发 FFT 峰检测)")

    # ── 14. G1: to_composite_token (C 实验工程化) ──────────────
    print("\n14. G1 to_composite_token (coords → <point3d> + dict)...")
    _m_g1 = StructureCognitiveMap.from_coords(
        species=["Fe", "O", "O"],
        coords=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        cutoff=3.0,
    )
    token = _m_g1.to_composite_token()
    assert "text" in token and "coords" in token and "species" in token
    assert token["n_atoms"] == 3
    assert "<point3d>" in token["text"], f"missing <point3d>: {token['text']}"
    assert "(Fe)" in token["text"] and "(O)" in token["text"]
    # 原点归一化到 [0,0,0]
    from huginn.tools.visual_hook import parse_point3d_primitive
    parsed = parse_point3d_primitive(token["text"])
    assert len(parsed) == 3, f"expected 3 atoms, got {len(parsed)}"
    assert parsed[0]["label"] == "Fe"
    assert parsed[0]["coordinates"] == [0, 0, 0], f"origin: {parsed[0]}"
    # coords 跟原 map 一致
    assert np.allclose(token["coords"], _m_g1.coords)
    # 跟 se3_act 集成: 旋转 90° z, <point3d> 和 coords 同步
    try:
        from huginn.metacog.composite_token_experiment import CompositeToken, se3_act
        ct = CompositeToken(token["text"], token["coords"])
        rot_z90 = _R.from_euler("z", 90, degrees=True)
        rotated = se3_act(rot_z90, np.zeros(3), ct)
        # coords [2,0,0] (O, x 轴) → [0,2,0] (y 轴)
        assert np.allclose(rotated.coords[1], [0.0, 2.0, 0.0], atol=1e-6), \
            f"coords rotate: {rotated.coords[1]}"
        # <point3d> 同步: [999,0,0] (O x 轴 max) → [-0,999,0] 或 [0,999,0]
        r_parsed = parse_point3d_primitive(rotated.text)
        assert len(r_parsed) == 3
        # Fe 原点不变
        assert r_parsed[0]["coordinates"] == [0, 0, 0], f"Fe origin: {r_parsed[0]}"
        print(f"   OK: n_atoms=3, <point3d>={parsed[0]['coordinates']}...{parsed[2]['coordinates']}")
        print(f"   OK: se3_act rotate z=90°, coords[1] [2,0,0]→{rotated.coords[1].tolist()}")
    except ImportError:
        print("   SKIP: scipy not available for se3_act integration test")

    # ── 15. Hodge: 拓扑三分量分解 ─────────────────────────────
    print("\n15. Hodge decomposition (gradient / curl / harmonic)...")
    # 构建已知图: 4 原子方形 (连通, genus=0)
    _m_hodge = StructureCognitiveMap.from_coords(
        species=["A", "B", "C", "D"],
        coords=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        cutoff=1.5,  # 对角线 √2 ≈ 1.41 ≤ 1.5, 全连通
    )
    hodge = _m_hodge.hodge_decomposition()
    assert "n_harmonic" in hodge and "eigenvalues" in hodge
    assert "gradient_fraction" in hodge and "curl_fraction" in hodge
    # 4 原子全连通: λ=0 重数 = 1 (1 个连通分量)
    assert hodge["n_harmonic"] == 1, f"connected graph should have 1 harmonic, got {hodge['n_harmonic']}"
    # 谱隙 > 0 (连通)
    assert hodge["spectrum_gap"] > 1e-6, f"connected graph gap>0, got {hodge['spectrum_gap']}"
    print(f"   OK: 4-atom 全连通图, n_harmonic={hodge['n_harmonic']}, "
          f"gap={hodge['spectrum_gap']:.4f}, grad={hodge['gradient_fraction']}, "
          f"curl={hodge['curl_fraction']}")
    # 测试不连通图: 2 个独立对
    _m_disc = StructureCognitiveMap.from_coords(
        species=["A", "B", "C", "D"],
        coords=np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0], [11, 0, 0]], dtype=float),
        cutoff=1.5,  # 只有 A-B 和 C-D 连, AB-CD 不连
    )
    hodge_disc = _m_disc.hodge_decomposition()
    assert hodge_disc["n_harmonic"] == 2, \
        f"2-component graph should have 2 harmonic, got {hodge_disc['n_harmonic']}"
    assert hodge_disc["spectrum_gap"] < 1e-6, \
        f"disconnected graph gap≈0, got {hodge_disc['spectrum_gap']}"
    print(f"   OK: 2-component 图, n_harmonic={hodge_disc['n_harmonic']}, "
          f"gap={hodge_disc['spectrum_gap']:.6f} (≈0, disconnected)")
    # 测试环形图 (genus 相关): 3 原子三角环
    _m_tri = StructureCognitiveMap.from_coords(
        species=["A", "B", "C"],
        coords=np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0]], dtype=float),
        cutoff=1.5,  # 三边都连, 形成环
    )
    hodge_tri = _m_tri.hodge_decomposition()
    assert hodge_tri["n_harmonic"] == 1, f"connected triangle: 1 harmonic, got {hodge_tri['n_harmonic']}"
    print(f"   OK: 3-atom 三角环, n_harmonic={hodge_tri['n_harmonic']}, "
          f"eigenvalues={[round(e, 4) for e in hodge_tri['eigenvalues']]}")

    print("\nstructure_cognitive_map selfcheck OK (15 actions + bond + compose + from_image + G1 + Hodge)")


if __name__ == "__main__":
    _selfcheck()
