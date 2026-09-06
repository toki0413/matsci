#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MIN_MDANALYSIS_VERSION = (2, 10, 0)
RECORDS_FILENAME = "solvation_environment_records.csv"
DISTRIBUTION_FILENAME = "solvation_environment_distribution.csv"
SUMMARY_FILENAME = "solvation_environment_summary.json"
PLOT_FILENAME = "fig_coordination_environment_polar.png"
DISTRIBUTION_PLOT_FILENAME = "fig_coordination_environment_distribution.png"


@dataclass(frozen=True)
class MoleculeDefinition:
    """保存一个 TPR molecule 的稳定拓扑定义。

    功能目的：
        将逐帧不变的 molecule identity、species 和 PBC 展开路径预先整理好。
    输入参数：
        molnum：TPR molecule 编号；moltype：TPR molecule 类型；atom_indices：全局原子索引；
        traversal_edges：按父节点已访问顺序排列的成键遍历边。
    返回值：
        不可变的数据对象，供逐帧 COG 计算复用。
    关键流程：
        多原子分子的遍历边由 bonds 的广度优先搜索产生。
    可能报错或边界情况：
        本类本身不校验连通性；连通性由 build_molecule_definitions 负责。
    """

    molnum: int
    moltype: str
    atom_indices: tuple[int, ...]
    traversal_edges: tuple[tuple[int, int], ...]


def require_mdanalysis() -> Any:
    """导入并检查 MDAnalysis 版本。

    功能目的：确保 TPR 当前坐标读取和本脚本使用的 API 具有明确版本基线。
    输入参数：无。
    返回值：已导入的 MDAnalysis 模块。
    关键流程：从版本字符串提取前三个数字并与 2.10.0 比较。
    可能报错或边界情况：未安装或版本过低时抛出 RuntimeError，不修改环境。
    """

    try:
        import MDAnalysis as mda
    except ImportError as exc:
        raise RuntimeError("需要 MDAnalysis >= 2.10.0；当前环境未安装 MDAnalysis。") from exc

    version_numbers = tuple(int(item) for item in re.findall(r"\d+", mda.__version__)[:3])
    padded_version = version_numbers + (0,) * (3 - len(version_numbers))
    if padded_version < MIN_MDANALYSIS_VERSION:
        required = ".".join(str(item) for item in MIN_MDANALYSIS_VERSION)
        raise RuntimeError(f"需要 MDAnalysis >= {required}；当前版本为 {mda.__version__}。")
    return mda


def stable_unique(values: Sequence[Any]) -> list[Any]:
    """按首次出现顺序返回唯一值列表。"""

    return list(dict.fromkeys(values))


def get_atom_attribute(atoms: Any, name: str) -> np.ndarray | None:
    """安全读取 AtomGroup 拓扑属性。

    功能目的：让 inspect 能在属性缺失时继续报告其他拓扑信息。
    输入参数：atoms 为 MDAnalysis AtomGroup，name 为属性名。
    返回值：属性的 NumPy 数组；属性不存在时返回 None。
    关键流程：捕获 MDAnalysis 的 NoDataError 及普通属性错误。
    可能报错或边界情况：不会吞掉数组转换后的数据错误。
    """

    try:
        return np.asarray(getattr(atoms, name))
    except (AttributeError, RuntimeError):
        return None
    except Exception as exc:
        if exc.__class__.__name__ == "NoDataError":
            return None
        raise


def load_universe(tpr: Path, xtc: Path | None = None, gro: Path | None = None) -> Any:
    """从 TPR 与可选坐标/轨迹文件构建 MDAnalysis Universe。

    功能目的：统一 inspect 与 analyze 的 GROMACS 输入加载逻辑。
    输入参数：tpr 为必需拓扑；xtc 与 gro 可选且互斥。
    返回值：MDAnalysis Universe。
    关键流程：仅 TPR 时尝试读取 TPR 内当前帧；提供 XTC/GRO 时由 TPR 提供拓扑。
    可能报错或边界情况：文件缺失、XTC/GRO 同时提供或坐标不可读时给出明确异常。
    """

    mda = require_mdanalysis()
    if not tpr.is_file():
        raise FileNotFoundError(f"TPR 文件不存在: {tpr}")
    if xtc is not None and gro is not None:
        raise ValueError("--xtc 与 --gro 互斥，不能同时提供。")
    coordinate_path = xtc if xtc is not None else gro
    if coordinate_path is not None and not coordinate_path.is_file():
        raise FileNotFoundError(f"坐标/轨迹文件不存在: {coordinate_path}")

    try:
        if coordinate_path is None:
            universe = mda.Universe(str(tpr))
        else:
            universe = mda.Universe(str(tpr), str(coordinate_path))
    except Exception as exc:
        source = str(coordinate_path) if coordinate_path is not None else "TPR 当前帧"
        raise RuntimeError(f"无法读取 GROMACS 输入（TPR={tpr}，坐标={source}）：{exc}") from exc
    return universe


def build_inspection_report(universe: Any, tpr: Path) -> dict[str, Any]:
    """汇总 TPR 中供 Agent 判断中心的候选名称和拓扑能力。

    功能目的：只读报告 moltype、resname、atomname 及分析所需能力，不计算配位环境。
    输入参数：已加载的 Universe 和 TPR 路径。
    返回值：可直接序列化为 JSON 的检查报告。
    关键流程：分别按 molecule、residue 和 atom name 聚合，并检查 molnum、bonds、box、坐标。
    可能报错或边界情况：拓扑属性缺失时在 capabilities 中标为 false，仍尽量输出其余信息。
    """

    atoms = universe.atoms
    molnums = get_atom_attribute(atoms, "molnums")
    moltypes = get_atom_attribute(atoms, "moltypes")
    names = get_atom_attribute(atoms, "names")
    resnames = get_atom_attribute(atoms, "resnames")

    molecule_rows: list[dict[str, Any]] = []
    if molnums is not None and moltypes is not None:
        # 一次遍历完成 molnum 分组，避免大量溶剂分子时为每个 molecule 重扫全体系。
        atom_indices_by_molnum: dict[int, list[int]] = defaultdict(list)
        for atom_index, molnum_value in enumerate(molnums.tolist()):
            atom_indices_by_molnum[int(molnum_value)].append(atom_index)
        for molnum, grouped_indices in atom_indices_by_molnum.items():
            atom_indices = np.asarray(grouped_indices, dtype=int)
            molecule_moltypes = stable_unique([str(value) for value in moltypes[atom_indices].tolist()])
            molecule_rows.append(
                {
                    "molnum": molnum,
                    "moltype": molecule_moltypes[0] if len(molecule_moltypes) == 1 else molecule_moltypes,
                    "atom_count": int(len(atom_indices)),
                    "resnames": sorted(set(str(value) for value in resnames[atom_indices].tolist())) if resnames is not None else [],
                    "atomnames": sorted(set(str(value) for value in names[atom_indices].tolist())) if names is not None else [],
                }
            )

    moltype_summary: list[dict[str, Any]] = []
    grouped_molecules: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in molecule_rows:
        moltype = row["moltype"]
        grouped_molecules[str(moltype)].append(row)
    for moltype in sorted(grouped_molecules):
        rows = grouped_molecules[moltype]
        moltype_summary.append(
            {
                "moltype": moltype,
                "molecule_count": len(rows),
                "atom_count_per_molecule": sorted(set(int(row["atom_count"]) for row in rows)),
                "resnames": sorted({name for row in rows for name in row["resnames"]}),
                "atomnames": sorted({name for row in rows for name in row["atomnames"]}),
            }
        )

    residue_summary: list[dict[str, Any]] = []
    for resname in sorted(set(str(value) for value in resnames.tolist())) if resnames is not None else []:
        selected = atoms[resnames.astype(str) == resname]
        selected_moltypes = get_atom_attribute(selected, "moltypes")
        residue_summary.append(
            {
                "resname": resname,
                "residue_count": len(stable_unique([int(value) for value in selected.resindices.tolist()])),
                "atom_count": int(len(selected)),
                "moltypes": sorted(set(str(value) for value in selected_moltypes.tolist())) if selected_moltypes is not None else [],
                "atomnames": sorted(set(str(value) for value in selected.names.tolist())) if names is not None else [],
            }
        )

    atomname_summary: list[dict[str, Any]] = []
    for atomname in sorted(set(str(value) for value in names.tolist())) if names is not None else []:
        mask = names.astype(str) == atomname
        atomname_summary.append(
            {
                "atomname": atomname,
                "count": int(np.count_nonzero(mask)),
                "resnames": sorted(set(str(value) for value in resnames[mask].tolist())) if resnames is not None else [],
                "moltypes": sorted(set(str(value) for value in moltypes[mask].tolist())) if moltypes is not None else [],
            }
        )

    try:
        bond_count = int(len(universe.bonds))
        bonds_available = True
    except Exception:
        bond_count = 0
        bonds_available = False

    coordinates_available = False
    box_available = False
    frame_count: int | None = None
    coordinate_error: str | None = None
    try:
        frame_count = int(len(universe.trajectory))
        positions = np.asarray(universe.atoms.positions, dtype=float)
        coordinates_available = positions.shape == (len(atoms), 3) and bool(np.all(np.isfinite(positions)))
        dimensions = np.asarray(universe.trajectory.ts.dimensions, dtype=float)
        box_available = dimensions.shape == (6,) and bool(np.all(np.isfinite(dimensions))) and bool(np.all(dimensions[:3] > 0))
    except Exception as exc:
        coordinate_error = str(exc)

    return {
        "schema_version": 1,
        "mode": "inspect",
        "tpr": str(tpr.expanduser().resolve()),
        "counts": {
            "atoms": int(len(atoms)),
            "residues": int(len(universe.residues)),
            "molecules": len(molecule_rows) if molnums is not None else None,
            "bonds": bond_count,
            "frames": frame_count,
        },
        "capabilities": {
            "molnum": molnums is not None,
            "moltype": moltypes is not None,
            "bonds": bonds_available,
            "current_coordinates": coordinates_available,
            "box": box_available,
        },
        "coordinate_error": coordinate_error,
        "moltypes": moltype_summary,
        "resnames": residue_summary,
        "atomnames": atomname_summary,
    }


def build_molecule_definitions(universe: Any) -> list[MoleculeDefinition]:
    """由 TPR molnum、moltype 与 bonds 构建完整 molecule 定义。

    功能目的：建立所有邻居分子的唯一身份，并准备逐帧 PBC 展开路径。
    输入参数：含 TPR 拓扑的 MDAnalysis Universe。
    返回值：按 molecule 首次出现顺序排列的 MoleculeDefinition 列表。
    关键流程：按 molnum 分组；每个多原子 molecule 用内部 bonds 做广度优先搜索。
    可能报错或边界情况：缺少 molnum/moltype、单个 molecule 含多个 moltype、或多原子 molecule 不连通时抛错。
    """

    atoms = universe.atoms
    molnums = get_atom_attribute(atoms, "molnums")
    moltypes = get_atom_attribute(atoms, "moltypes")
    if molnums is None or moltypes is None:
        raise ValueError("TPR 必须同时提供 molnum 与 moltype；请先运行 inspect 检查拓扑能力。")

    try:
        bond_indices = np.asarray(universe.bonds.indices, dtype=int)
    except Exception:
        bond_indices = np.empty((0, 2), dtype=int)

    # 原子和键均只分组一次，避免 n_molecules 次全量扫描造成二次复杂度。
    molnums_int = np.asarray(molnums, dtype=int)
    atom_indices_by_molnum: dict[int, list[int]] = defaultdict(list)
    for atom_index, molnum_value in enumerate(molnums_int.tolist()):
        atom_indices_by_molnum[int(molnum_value)].append(atom_index)
    bonds_by_molnum: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for left, right in bond_indices.tolist():
        left_i, right_i = int(left), int(right)
        if molnums_int[left_i] == molnums_int[right_i]:
            bonds_by_molnum[int(molnums_int[left_i])].append((left_i, right_i))

    molecule_definitions: list[MoleculeDefinition] = []
    for molnum, grouped_indices in atom_indices_by_molnum.items():
        atom_indices = np.asarray(grouped_indices, dtype=int)
        molecule_moltypes = stable_unique([str(value) for value in moltypes[atom_indices].tolist()])
        if len(molecule_moltypes) != 1:
            raise ValueError(f"molnum={molnum} 对应多个 moltype: {molecule_moltypes}")

        traversal_edges: list[tuple[int, int]] = []
        if len(atom_indices) > 1:
            atom_set = set(int(value) for value in atom_indices.tolist())
            adjacency: dict[int, list[int]] = {index: [] for index in atom_set}
            for left_i, right_i in bonds_by_molnum.get(molnum, []):
                adjacency[left_i].append(right_i)
                adjacency[right_i].append(left_i)

            root = int(atom_indices[0])
            visited = {root}
            queue: deque[int] = deque([root])
            while queue:
                parent = queue.popleft()
                for child in sorted(adjacency[parent]):
                    if child in visited:
                        continue
                    visited.add(child)
                    traversal_edges.append((parent, child))
                    queue.append(child)
            if visited != atom_set:
                missing = sorted(atom_set - visited)
                raise ValueError(
                    f"molnum={molnum} ({molecule_moltypes[0]}) 是多原子分子，但 TPR bonds 不连通；"
                    f"无法可靠恢复 PBC 下的完整分子。未连通原子索引: {missing[:10]}"
                )

        molecule_definitions.append(
            MoleculeDefinition(
                molnum=molnum,
                moltype=molecule_moltypes[0],
                atom_indices=tuple(int(value) for value in atom_indices.tolist()),
                traversal_edges=tuple(traversal_edges),
            )
        )
    if not molecule_definitions:
        raise ValueError("TPR 中未发现 molecule。")
    return molecule_definitions


def validate_box_and_cutoff(timestep: Any, cutoff_angstrom: float) -> np.ndarray:
    """验证当前帧周期盒并检查 cutoff 的最小镜像约束。

    功能目的：避免在盒信息缺失或 cutoff 过大时产生歧义距离。
    输入参数：MDAnalysis Timestep 与单位为 Å 的 cutoff。
    返回值：MDAnalysis distances API 使用的六元 dimensions 数组。
    关键流程：由 triclinic_vectors 得到三条晶格向量长度，并要求 cutoff 不超过最短者的一半。
    可能报错或边界情况：非有限盒、退化盒、非正 cutoff 或超出半盒时抛错。
    """

    from MDAnalysis.lib.mdamath import triclinic_vectors

    dimensions = np.asarray(timestep.dimensions, dtype=float)
    if dimensions.shape != (6,) or not np.all(np.isfinite(dimensions)) or np.any(dimensions[:3] <= 0):
        raise ValueError(f"frame={timestep.frame} 缺少有效周期盒 dimensions。")
    vectors = np.asarray(triclinic_vectors(dimensions), dtype=float)
    vector_lengths = np.linalg.norm(vectors, axis=1)
    if vectors.shape != (3, 3) or np.any(~np.isfinite(vector_lengths)) or np.any(vector_lengths <= 0):
        raise ValueError(f"frame={timestep.frame} 的周期盒退化，无法计算最小镜像距离。")
    if not math.isfinite(cutoff_angstrom) or cutoff_angstrom <= 0:
        raise ValueError("第一配位层半径必须大于 0。")
    maximum_cutoff = 0.5 * float(np.min(vector_lengths))
    if cutoff_angstrom > maximum_cutoff + 1e-9:
        raise ValueError(
            f"frame={timestep.frame} 的 cutoff={cutoff_angstrom / 10.0:.6g} nm 超过"
            f"最短晶格向量一半 {maximum_cutoff / 10.0:.6g} nm。"
        )
    return dimensions


def calculate_molecule_cogs(
    positions: np.ndarray,
    dimensions: np.ndarray,
    molecule_definitions: Sequence[MoleculeDefinition],
) -> np.ndarray:
    """在 PBC 下恢复完整分子并计算几何中心。

    功能目的：为所有 neighbor molecule 生成每帧唯一的 COG 距离对象。
    输入参数：positions 为 Å 坐标；dimensions 为周期盒；molecule_definitions 为预构建拓扑。
    返回值：形状为 (n_molecules, 3) 的 COG 数组，单位 Å。
    关键流程：单原子直接取坐标；多原子沿成键遍历边逐步应用最小镜像位移后求算术平均。
    可能报错或边界情况：坐标形状错误、非有限坐标或拓扑索引越界时抛错。
    """

    from MDAnalysis.lib.distances import minimize_vectors

    coordinates = np.asarray(positions, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.all(np.isfinite(coordinates)):
        raise ValueError("当前帧坐标必须是有限的 (n_atoms, 3) 数组。")

    cogs = np.empty((len(molecule_definitions), 3), dtype=float)
    for molecule_index, molecule in enumerate(molecule_definitions):
        atom_indices = np.asarray(molecule.atom_indices, dtype=int)
        if len(atom_indices) == 1:
            cogs[molecule_index] = coordinates[atom_indices[0]]
            continue

        unwrapped: dict[int, np.ndarray] = {int(atom_indices[0]): coordinates[atom_indices[0]].copy()}
        for parent, child in molecule.traversal_edges:
            displacement = coordinates[child] - coordinates[parent]
            minimum_image = minimize_vectors(displacement[np.newaxis, :], dimensions)[0]
            unwrapped[child] = unwrapped[parent] + minimum_image
        cogs[molecule_index] = np.mean([unwrapped[int(index)] for index in atom_indices], axis=0)
    return cogs


def choose_frame_indices(total_frames: int, stride: int | None, n_frames: int | None) -> list[int]:
    """生成确定性的轨迹抽帧索引。

    功能目的：实现默认逐帧、指定 stride 或均匀 n_frames 三种互斥采样。
    输入参数：total_frames 为存储帧数；stride 与 n_frames 至多一个非空。
    返回值：升序、无重复的零基轨迹帧索引。
    关键流程：stride 从第 0 帧起取样；n_frames 用包含首尾的等间隔四舍五入索引。
    可能报错或边界情况：帧数为空、参数冲突、非正参数或 n_frames 超过存储帧数时抛错。
    """

    if total_frames <= 0:
        raise ValueError("输入不包含可用坐标帧。")
    if stride is not None and n_frames is not None:
        raise ValueError("stride 与 n_frames 互斥，不能同时指定。")
    if stride is not None:
        if stride <= 0:
            raise ValueError("stride 必须是正整数。")
        return list(range(0, total_frames, stride))
    if n_frames is not None:
        if n_frames <= 0:
            raise ValueError("n_frames 必须是正整数。")
        if n_frames > total_frames:
            raise ValueError(f"n_frames={n_frames} 超过存储帧数 {total_frames}。")
        if n_frames == 1:
            return [0]
        indices = np.rint(np.linspace(0, total_frames - 1, n_frames)).astype(int).tolist()
        if len(set(indices)) != n_frames:
            raise RuntimeError(f"无法为 {total_frames} 帧生成 {n_frames} 个唯一均匀索引。")
        return indices
    return list(range(total_frames))


def resolve_centers(
    universe: Any,
    center_mode: str,
    center_selection: str,
    molecule_definitions: Sequence[MoleculeDefinition],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """把已确认的 MDAnalysis selection 解析为稳定中心列表。

    功能目的：脚本只执行明确 selection，不承担自然语言或化学别名判断。
    输入参数：Universe、atom/molecule 模式、selection 字符串和 molecule 定义。
    返回值：逐中心元数据列表与用于 summary 的解析报告。
    关键流程：atom 模式逐原子建中心；molecule 模式由选中原子的 molnum 去重并扩展到完整分子。
    可能报错或边界情况：selection 无效/为空、模式非法或选中原子缺少 molnum 时抛错。
    """

    if center_mode not in {"atom", "molecule"}:
        raise ValueError("center_mode 必须是 atom 或 molecule。")
    try:
        selected = universe.select_atoms(center_selection)
    except Exception as exc:
        raise ValueError(f"无效的 MDAnalysis center selection {center_selection!r}: {exc}") from exc
    if len(selected) == 0:
        raise ValueError(f"center selection 未选中任何原子: {center_selection!r}")

    selected_molnums = get_atom_attribute(selected, "molnums")
    selected_moltypes = get_atom_attribute(selected, "moltypes")
    if selected_molnums is None or selected_moltypes is None:
        raise ValueError("中心 selection 的原子缺少 TPR molnum/moltype。")

    molecule_by_molnum = {definition.molnum: definition for definition in molecule_definitions}
    centers: list[dict[str, Any]] = []
    if center_mode == "atom":
        for atom in selected:
            atom_molnum = int(atom.molnum)
            centers.append(
                {
                    "center_id": f"atom:{int(atom.index)}",
                    "atom_index": int(atom.index),
                    "molnum": atom_molnum,
                    "moltype": str(atom.moltype),
                    "atomname": str(atom.name),
                    "resname": str(atom.resname),
                }
            )
    else:
        for molnum in stable_unique([int(value) for value in selected_molnums.tolist()]):
            if molnum not in molecule_by_molnum:
                raise ValueError(f"center selection 解析到未知 molnum={molnum}。")
            molecule = molecule_by_molnum[molnum]
            centers.append(
                {
                    "center_id": f"molecule:{molnum}",
                    "atom_index": None,
                    "molnum": molnum,
                    "moltype": molecule.moltype,
                    "atomname": None,
                    "resname": None,
                }
            )

    if center_mode == "atom":
        resolved_atoms = selected
    else:
        resolved_molnums = {int(center["molnum"]) for center in centers}
        all_molnums = get_atom_attribute(universe.atoms, "molnums")
        if all_molnums is None:
            raise ValueError("无法读取完整 molecule 的 molnum。")
        resolved_mask = np.asarray([int(value) in resolved_molnums for value in all_molnums], dtype=bool)
        resolved_atoms = universe.atoms[resolved_mask]

    resolution = {
        "center_mode": center_mode,
        "center_selection": center_selection,
        "selected_atom_count": int(len(selected)),
        "center_count": len(centers),
        "selected_moltypes": sorted(set(str(value) for value in selected_moltypes.tolist())),
        "selected_resnames": sorted(set(str(value) for value in selected.resnames.tolist())),
        "selected_atomnames": sorted(set(str(value) for value in selected.names.tolist())),
        "resolved_atom_count": int(len(resolved_atoms)),
        "actual_moltypes": sorted(set(str(value) for value in resolved_atoms.moltypes.tolist())),
        "actual_resnames": sorted(set(str(value) for value in resolved_atoms.resnames.tolist())),
        "actual_atomnames": sorted(set(str(value) for value in resolved_atoms.names.tolist())),
        "distance_definition": "atom-to-molecule-COG" if center_mode == "atom" else "molecule-COG-to-molecule-COG",
    }
    return centers, resolution


def build_distribution(
    records: list[dict[str, Any]], species_order: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[tuple[int, ...], int]]:
    """按完整 species 计数向量归并溶剂化环境。

    功能目的：把每个中心-帧事件转为环境类型的次数与比例分布。
    输入参数：records 为事件记录；species_order 决定向量顺序。
    返回值：排序后的环境列表，以及 composition tuple 到 environment id 的映射。
    关键流程：按出现次数降序、向量字典序升序稳定排序，再分配从 1 开始的 id。
    可能报错或边界情况：records 或 species_order 为空时抛错。
    """

    if not records:
        raise ValueError("没有中心-帧事件，无法构建环境分布。")
    if not species_order:
        raise ValueError("species_order 为空，无法定义环境组成。")
    counts = Counter(tuple(int(record["composition"][species]) for species in species_order) for record in records)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    total = len(records)
    environments: list[dict[str, Any]] = []
    vector_to_id: dict[tuple[int, ...], int] = {}
    for environment_id, (vector, count) in enumerate(ordered, start=1):
        composition_full = {species: int(value) for species, value in zip(species_order, vector)}
        vector_json = json.dumps(list(vector), ensure_ascii=False, separators=(",", ":"))
        vector_hash = hashlib.sha256(vector_json.encode("utf-8")).hexdigest()
        environments.append(
            {
                "env_id": environment_id,
                "count": int(count),
                "fraction": float(count / total),
                "composition": {key: value for key, value in composition_full.items() if value != 0},
                "composition_full": composition_full,
                "signature": {
                    "species_order_vector": list(vector),
                    "species_order_vector_repr": vector_json,
                    "species_order_vector_hash": vector_hash,
                },
            }
        )
        vector_to_id[vector] = environment_id
    return environments, vector_to_id


def analyze_universe(
    universe: Any,
    rdf_radius_nm: float,
    center_mode: str,
    center_selection: str,
    stride: int | None = None,
    n_frames: int | None = None,
) -> dict[str, Any]:
    """对已加载的 GROMACS Universe 计算固定半径溶剂化环境。

    功能目的：实现与文件输出解耦的核心统计，便于复用和测试。
    输入参数：Universe、半径 nm、中心模式/selection，以及互斥抽帧参数。
    返回值：包含 records、environments、中心解析、采样和验证信息的字典。
    关键流程：构建 molecule 拓扑；逐帧算 PBC-safe COG；稀疏 cutoff 搜索；排除自身 molnum；归并组成。
    可能报错或边界情况：半径、拓扑、坐标、盒、selection 或多原子分子连通性不合法时抛错。
    """

    from MDAnalysis.lib.distances import capped_distance

    if not math.isfinite(rdf_radius_nm) or rdf_radius_nm <= 0:
        raise ValueError("rdf_radius_nm 必须是大于 0 的有限数。")
    cutoff_angstrom = float(rdf_radius_nm) * 10.0
    molecule_definitions = build_molecule_definitions(universe)
    molnum_to_index = {definition.molnum: index for index, definition in enumerate(molecule_definitions)}
    species_order = stable_unique([definition.moltype for definition in molecule_definitions])
    centers, center_resolution = resolve_centers(
        universe=universe,
        center_mode=center_mode,
        center_selection=center_selection,
        molecule_definitions=molecule_definitions,
    )
    frame_indices = choose_frame_indices(len(universe.trajectory), stride=stride, n_frames=n_frames)

    records: list[dict[str, Any]] = []
    self_pairs_excluded = 0
    for sample_index, trajectory_frame_index in enumerate(frame_indices):
        timestep = universe.trajectory[trajectory_frame_index]
        dimensions = validate_box_and_cutoff(timestep, cutoff_angstrom)
        molecule_cogs = calculate_molecule_cogs(universe.atoms.positions, dimensions, molecule_definitions)

        if center_mode == "atom":
            center_positions = np.asarray(
                [universe.atoms[int(center["atom_index"])].position for center in centers], dtype=float
            )
        else:
            center_positions = np.asarray(
                [molecule_cogs[molnum_to_index[int(center["molnum"])]] for center in centers], dtype=float
            )

        pairs = capped_distance(
            center_positions,
            molecule_cogs,
            max_cutoff=cutoff_angstrom,
            box=dimensions,
            return_distances=False,
        )
        neighbor_indices: list[set[int]] = [set() for _ in centers]
        for center_index, molecule_index in np.asarray(pairs, dtype=int).reshape(-1, 2):
            molecule = molecule_definitions[int(molecule_index)]
            if molecule.molnum == int(centers[int(center_index)]["molnum"]):
                self_pairs_excluded += 1
                continue
            neighbor_indices[int(center_index)].add(int(molecule_index))

        time_ps = float(timestep.time) if timestep.time is not None and math.isfinite(float(timestep.time)) else None
        for center_index, center in enumerate(centers):
            composition = dict.fromkeys(species_order, 0)
            for molecule_index in sorted(neighbor_indices[center_index]):
                composition[molecule_definitions[molecule_index].moltype] += 1
            records.append(
                {
                    "event_id": f"frame:{int(timestep.frame)}__{center['center_id']}",
                    "sample_index": sample_index,
                    "frame_index": int(timestep.frame),
                    "time_ps": time_ps,
                    "center_id": center["center_id"],
                    "center_mode": center_mode,
                    "center_atom_index": center["atom_index"],
                    "center_molnum": center["molnum"],
                    "center_moltype": center["moltype"],
                    "center_atomname": center["atomname"],
                    "center_resname": center["resname"],
                    "total_neighbor_molecules": int(sum(composition.values())),
                    "composition": composition,
                }
            )

    environments, vector_to_id = build_distribution(records, species_order)
    for record in records:
        vector = tuple(int(record["composition"][species]) for species in species_order)
        record["environment_type_id"] = vector_to_id[vector]

    expected_events = len(centers) * len(frame_indices)
    fraction_sum = float(sum(environment["fraction"] for environment in environments))
    validation = {
        "expected_total_events": expected_events,
        "actual_total_events": len(records),
        "total_event_count_matches": len(records) == expected_events,
        "fraction_sum": fraction_sum,
        "fraction_sum_is_one": math.isclose(fraction_sum, 1.0, rel_tol=0.0, abs_tol=1e-12),
        "self_molecule_pairs_excluded": self_pairs_excluded,
        "neighbor_identity": "TPR molnum",
        "species_identity": "TPR moltype",
        "molecule_cog_uses_pbc_bond_reconstruction": True,
    }
    if not validation["total_event_count_matches"] or not validation["fraction_sum_is_one"]:
        raise RuntimeError(f"内部统计不变量失败: {validation}")

    return {
        "records": records,
        "environments": environments,
        "species_order": species_order,
        "center_resolution": center_resolution,
        "sampling": {
            "stored_frame_count": int(len(universe.trajectory)),
            "sampled_frame_count": len(frame_indices),
            "frame_indices": frame_indices,
            "stride": stride if stride is not None else (1 if n_frames is None else None),
            "n_frames": n_frames,
        },
        "validation": validation,
    }


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    """写出 RFC 4180、UTF-8 BOM、CRLF 的 Excel 友好 CSV。

    功能目的：统一满足 macOS/Windows Excel 的编码和换行要求。
    输入参数：输出路径、稳定列顺序、行字典。
    返回值：None。
    关键流程：newline 置空并由 csv.DictWriter 明确输出 CRLF 与最小必要引号。
    可能报错或边界情况：父目录会自动创建；磁盘和权限错误原样抛出。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="raise",
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\r\n",
            doublequote=True,
        )
        writer.writeheader()
        writer.writerows(rows)


def build_plot_summary(summary_payload: dict[str, Any], max_environment_types: int = 12) -> tuple[dict[str, Any], dict[str, Any]]:
    """为高多样性环境构建可读的极坐标绘图摘要。

    功能目的：完整统计类型过多时，仅在图中保留最高频类型，并将长尾合并为 Other。
    输入参数：完整 summary_payload；max_environment_types 为图中最多扇区数，包含 Other。
    返回值：供绘图函数使用的浅拷贝摘要，以及记录截断策略的元数据。
    关键流程：统计 JSON/CSV 不变；只替换绘图副本中的 environments 列表。
    可能报错或边界情况：max_environment_types 小于 2 时抛错；环境数未超限时不聚合。
    """

    if max_environment_types < 2:
        raise ValueError("max_environment_types 必须至少为 2。")
    environments = list(summary_payload.get("environments", []))
    plot_payload = dict(summary_payload)
    if len(environments) <= max_environment_types:
        plotting_metadata = {
            "total_environment_types": len(environments),
            "displayed_environment_types": len(environments),
            "top_environment_types_shown": len(environments),
            "other_environment_types_aggregated": 0,
            "other_count": 0,
            "other_fraction": 0.0,
        }
        return plot_payload, plotting_metadata

    top_count = max_environment_types - 1
    retained = [dict(environment) for environment in environments[:top_count]]
    omitted = environments[top_count:]
    other_count = int(sum(int(environment["count"]) for environment in omitted))
    total_entries = int(summary_payload.get("summary", {}).get("total_entries", 0))
    species_order = [str(value) for value in summary_payload.get("species_order", [])]
    retained.append(
        {
            "env_id": "Other",
            "count": other_count,
            "fraction": float(other_count / total_entries) if total_entries else 0.0,
            "composition": {},
            "composition_full": dict.fromkeys(species_order, 0),
            "display_label": "Other",
        }
    )
    plot_payload["environments"] = retained
    plotting_metadata = {
        "total_environment_types": len(environments),
        "displayed_environment_types": len(retained),
        "top_environment_types_shown": top_count,
        "other_environment_types_aggregated": len(omitted),
        "other_count": other_count,
        "other_fraction": float(other_count / total_entries) if total_entries else 0.0,
    }
    return plot_payload, plotting_metadata


def plot_solvation_environment_distribution(
    summary_payload: dict[str, Any],
    output: bool = True,
    figsize: tuple[float, float] = (7.0, 4.6),
    figname: str = DISTRIBUTION_PLOT_FILENAME,
    max_environment_types: int = 12,
) -> dict[str, Any]:
    """绘制高频溶剂化环境及长尾合并项的水平比例图。

    功能目的：为环境类型较多的体系提供比极坐标图更易读的主结果图。
    输入参数：完整 summary；output/figsize/figname 控制导出；max_environment_types 包含 Other。
    返回值：实际采用的绘图聚合元数据。
    关键流程：复用 build_plot_summary；组成标签仅显示非零 species；比例始终以全部事件为分母。
    可能报错或边界情况：没有环境、绘图依赖不可用或输出路径不可写时抛错。
    """

    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    plot_payload, plotting_metadata = build_plot_summary(summary_payload, max_environment_types=max_environment_types)
    environments = list(plot_payload.get("environments", []))
    if not environments:
        raise ValueError("summary 中没有 environments，无法绘图。")

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
            "axes.unicode_minus": False,
        }
    )

    labels: list[str] = []
    percentages: list[float] = []
    for environment in environments:
        if environment.get("display_label") == "Other":
            omitted_count = int(plotting_metadata["other_environment_types_aggregated"])
            label = f"Other ({omitted_count} types)"
        else:
            composition = environment.get("composition", {})
            label = ", ".join(f"{species}={int(count)}" for species, count in composition.items()) or "No neighbors"
        labels.append(label)
        percentages.append(100.0 * float(environment["fraction"]))

    values = np.asarray(percentages, dtype=float)
    if np.allclose(values.max(), values.min()):
        color_values = np.full(len(values), 0.58)
    else:
        color_values = 0.35 + 0.45 * (values - values.min()) / max(values.max() - values.min(), 1e-12)
    colors = [colormaps.get_cmap("Blues")(value) for value in color_values]
    if plotting_metadata["other_environment_types_aggregated"]:
        colors[-1] = (0.72, 0.72, 0.72, 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    y_positions = np.arange(len(environments))
    ax.barh(y_positions, values, color=colors, edgecolor="none", height=0.68)
    ax.set_yticks(y_positions, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("Fraction of center–frame events (%)")
    ax.set_ylabel("Solvation environment")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.5, width=0.75)
    maximum = max(float(np.max(values)), 1.0)
    ax.set_xlim(0.0, maximum * 1.18)
    for y_position, value in zip(y_positions, values):
        ax.text(value + maximum * 0.015, y_position, f"{value:.1f}%", ha="left", va="center")
    fig.tight_layout()
    if output:
        output_path = Path(figname)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return plotting_metadata


def format_environment_vector_label(environment: dict[str, Any], species_order: Sequence[str]) -> str:
    """把环境组成格式化为稳定的连字符向量标签。"""

    if environment.get("display_label") is not None:
        return str(environment["display_label"])
    values = [int(environment.get("composition_full", {}).get(species, 0)) for species in species_order]
    return "-".join(str(value) for value in values)


def plot_coordination_polar_from_summary(
    summary_payload: dict[str, Any],
    output: bool = True,
    figsize: tuple[float, float] = (7.0, 4.6),
    figname: str = PLOT_FILENAME,
    dpi: int = 600,
    gamma: float = 0.55,
) -> None:
    """绘制固定半径环境类型的极坐标比例图。

    功能目的：保留与原配位分析流程一致的极坐标表达，便于快速比较高频环境。
    输入参数：summary_payload 为绘图摘要；output/figsize/figname 控制导出；dpi/gamma 控制分辨率和柱高压缩。
    返回值：None。
    关键流程：按全事件比例绘制扇形柱；组成标签遵循 species_order；支持聚合后的 Other 标签。
    可能报错或边界情况：没有环境、绘图依赖不可用或输出路径不可写时抛错。
    """

    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    environments = list(summary_payload.get("environments", []))
    if not environments:
        raise ValueError("summary 中没有 environments，无法绘图。")
    species_order = [str(value) for value in summary_payload.get("summary", {}).get("df_name_order", [])]
    total_entries = max(1, int(summary_payload.get("summary", {}).get("total_entries", 0)))
    counts = np.asarray([int(environment["count"]) for environment in environments], dtype=float)
    percentages = 100.0 * counts / float(total_entries)
    labels = [format_environment_vector_label(environment, species_order) for environment in environments]

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": dpi,
            "axes.unicode_minus": False,
        }
    )

    environment_count = len(environments)
    angles = np.linspace(0, 2.0 * np.pi, environment_count, endpoint=False)
    width = 2.0 * np.pi / max(1, environment_count) * 0.88
    maximum_percentage = max(float(np.max(percentages)), 1e-12)
    heights = (percentages / maximum_percentage) ** gamma * maximum_percentage
    inner_radius = 0.60 * maximum_percentage

    if np.allclose(percentages.max(), percentages.min()):
        color_values = np.full(environment_count, 0.55)
    else:
        normalized = (percentages - percentages.min()) / max(percentages.max() - percentages.min(), 1e-12)
        color_values = 0.35 + 0.45 * normalized
    colors = [colormaps.get_cmap("Blues")(value) for value in color_values]

    figure = plt.figure(figsize=figsize)
    axis = figure.add_subplot(111, polar=True)
    axis.bar(angles, heights, width=width, bottom=inner_radius, color=colors, linewidth=0, edgecolor="none")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    axis.spines["polar"].set_visible(False)
    axis.text(0.0, 0.0, "_".join(species_order), ha="center", va="center")

    label_offset = 0.04 * maximum_percentage
    for angle, height, label, percentage in zip(angles, heights, labels, percentages):
        degree = float(np.degrees(angle))
        rotation = degree - 90.0
        horizontal_alignment = "left"
        if 90.0 < degree < 270.0:
            rotation += 180.0
            horizontal_alignment = "right"
        axis.text(
            angle,
            inner_radius + height + label_offset,
            label,
            rotation=rotation,
            rotation_mode="anchor",
            ha=horizontal_alignment,
            va="center",
        )
        axis.text(angle, inner_radius + 0.52 * height, f"{percentage:.1f}%", ha="center", va="center")

    figure.tight_layout()
    if output:
        output_path = Path(figname)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def write_analysis_outputs(
    result: dict[str, Any],
    output_dir: Path,
    settings: dict[str, Any],
    plot: bool,
) -> dict[str, Path]:
    """将核心统计结果写为约定 CSV、JSON 和可选 PNG。

    功能目的：集中控制输出 schema、编码、文件名和绘图依赖边界。
    输入参数：analyze_universe 结果、输出目录、已确认 CLI 设置、是否绘图。
    返回值：逻辑输出名到绝对路径的映射。
    关键流程：先分配环境 id 后展开 species 列；JSON 使用稳定键结构；仅 plot=true 时导入绘图模块。
    可能报错或边界情况：输出目录不可写或绘图依赖缺失时抛出明确异常。
    """

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    species_order = [str(value) for value in result["species_order"]]
    species_columns = [f"count::{species}" for species in species_order]

    record_fields = [
        "event_id",
        "sample_index",
        "frame_index",
        "time_ps",
        "center_id",
        "center_mode",
        "center_atom_index",
        "center_molnum",
        "center_moltype",
        "center_atomname",
        "center_resname",
        "total_neighbor_molecules",
        "environment_type_id",
        "composition_json",
        *species_columns,
    ]
    record_rows: list[dict[str, Any]] = []
    for record in result["records"]:
        row = {key: record.get(key) for key in record_fields if not key.startswith("count::") and key != "composition_json"}
        row["composition_json"] = json.dumps(record["composition"], ensure_ascii=False, separators=(",", ":"))
        for species, column in zip(species_order, species_columns):
            row[column] = int(record["composition"][species])
        record_rows.append(row)

    distribution_fields = [
        "environment_type_id",
        "count",
        "fraction",
        "total_neighbor_molecules",
        "composition_vector",
        "composition_json",
        *species_columns,
    ]
    distribution_rows: list[dict[str, Any]] = []
    for environment in result["environments"]:
        composition = environment["composition_full"]
        row = {
            "environment_type_id": int(environment["env_id"]),
            "count": int(environment["count"]),
            "fraction": float(environment["fraction"]),
            "total_neighbor_molecules": int(sum(composition.values())),
            "composition_vector": environment["signature"]["species_order_vector_repr"],
            "composition_json": json.dumps(composition, ensure_ascii=False, separators=(",", ":")),
        }
        for species, column in zip(species_order, species_columns):
            row[column] = int(composition[species])
        distribution_rows.append(row)

    records_path = output_dir / RECORDS_FILENAME
    distribution_path = output_dir / DISTRIBUTION_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    write_csv_rows(records_path, record_fields, record_rows)
    write_csv_rows(distribution_path, distribution_fields, distribution_rows)

    output_paths: dict[str, Path] = {
        "records_csv": records_path,
        "distribution_csv": distribution_path,
        "summary_json": summary_path,
    }
    summary_payload = {
        "schema_version": 1,
        "method": {
            "name": "GROMACS fixed-radius solvation environment",
            "distance_definition": result["center_resolution"]["distance_definition"],
            "center_molecule_site": "geometric center (COG)",
            "neighbor_site": "molecule geometric center (COG)",
            "distance_unit": "nm",
        },
        "confirmed_settings": settings,
        "center_resolution": result["center_resolution"],
        "species_order": species_order,
        "sampling": result["sampling"],
        "summary": {
            "total_entries": len(result["records"]),
            "center_count": int(result["center_resolution"]["center_count"]),
            "sampled_frame_count": int(result["sampling"]["sampled_frame_count"]),
            "unique_environments": len(result["environments"]),
            "df_name_order": species_order,
        },
        "environments": result["environments"],
        "validation": result["validation"],
        "outputs": {key: str(path) for key, path in output_paths.items()},
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if plot:
        try:
            plot_path = output_dir / PLOT_FILENAME
            plot_payload, plotting_metadata = build_plot_summary(summary_payload, max_environment_types=12)
            plot_coordination_polar_from_summary(
                plot_payload,
                output=True,
                figsize=(7.0, 4.6),
                figname=str(plot_path),
                dpi=600,
            )
            output_paths["plot_png"] = plot_path
            summary_payload["outputs"]["plot_png"] = str(plot_path)
            summary_payload["plotting"] = plotting_metadata
            distribution_plot_path = output_dir / DISTRIBUTION_PLOT_FILENAME
            plot_solvation_environment_distribution(
                summary_payload,
                output=True,
                figsize=(7.0, 4.6),
                figname=str(distribution_plot_path),
                max_environment_types=12,
            )
            output_paths["distribution_plot_png"] = distribution_plot_path
            summary_payload["outputs"]["distribution_plot_png"] = str(distribution_plot_path)
            summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except ImportError as exc:
            raise RuntimeError(f"绘图依赖不可用；核心 CSV/JSON 已写出，但无法生成 PNG: {exc}") from exc
    return output_paths


def run_inspect(args: argparse.Namespace) -> int:
    """执行只读 TPR 检查并把 JSON 写到标准输出。"""

    universe = load_universe(args.tpr)
    report = build_inspection_report(universe, args.tpr)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def run_analyze(args: argparse.Namespace) -> int:
    """执行已由用户确认的固定半径统计并报告输出路径。"""

    universe = load_universe(args.tpr, xtc=args.xtc, gro=args.gro)
    try:
        # 只给 TPR 时必须确认其当前帧确实含坐标；不做静默降级或猜测。
        _ = np.asarray(universe.atoms.positions, dtype=float)
    except Exception as exc:
        if args.xtc is None and args.gro is None:
            raise RuntimeError("TPR 当前坐标不可读；请补充 --gro 或 --xtc 后重新确认设置。") from exc
        raise

    result = analyze_universe(
        universe=universe,
        rdf_radius_nm=args.rdf_radius_nm,
        center_mode=args.center_mode,
        center_selection=args.center_selection,
        stride=args.stride,
        n_frames=args.n_frames,
    )
    source_mode = "trajectory" if args.xtc is not None else "snapshot"
    coordinate_path = args.xtc if args.xtc is not None else args.gro
    settings = {
        "tpr": str(args.tpr.expanduser().resolve()),
        "xtc": str(args.xtc.expanduser().resolve()) if args.xtc is not None else None,
        "gro": str(args.gro.expanduser().resolve()) if args.gro is not None else None,
        "source_mode": source_mode,
        "coordinate_source": str(coordinate_path.expanduser().resolve()) if coordinate_path is not None else "TPR current frame",
        "rdf_radius_nm": float(args.rdf_radius_nm),
        "center_mode": args.center_mode,
        "center_selection": args.center_selection,
        "stride": args.stride if args.stride is not None else (1 if args.n_frames is None else None),
        "n_frames": args.n_frames,
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "plot": bool(args.plot),
    }
    output_paths = write_analysis_outputs(result, args.output_dir, settings=settings, plot=args.plot)
    print(json.dumps({key: str(value) for key, value in output_paths.items()}, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析 inspect/analyze 两个确定性子命令。"""

    parser = argparse.ArgumentParser(
        description="Inspect a GROMACS TPR or analyze fixed-radius molecular-COG solvation environments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Read only: list TPR names and topology capabilities.")
    inspect_parser.add_argument("--tpr", type=Path, required=True)
    inspect_parser.set_defaults(handler=run_inspect)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze settings that the user has already confirmed.")
    analyze_parser.add_argument("--tpr", type=Path, required=True)
    analyze_parser.add_argument("--rdf-radius-nm", type=float, required=True)
    analyze_parser.add_argument("--center-mode", choices=("atom", "molecule"), required=True)
    analyze_parser.add_argument("--center-selection", required=True)
    coordinates = analyze_parser.add_mutually_exclusive_group()
    coordinates.add_argument("--xtc", type=Path)
    coordinates.add_argument("--gro", type=Path)
    sampling = analyze_parser.add_mutually_exclusive_group()
    sampling.add_argument("--stride", type=int)
    sampling.add_argument("--n-frames", type=int)
    analyze_parser.add_argument("--output-dir", type=Path, required=True)
    analyze_parser.add_argument("--plot", action="store_true")
    analyze_parser.set_defaults(handler=run_analyze)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口；将预期错误转为简洁、可操作的消息。"""

    args = parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
