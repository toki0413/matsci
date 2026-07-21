"""Structure Cognitive Map tool — CodeAct 沙箱注入 4 函数.

借鉴 Ego3D-VLM (arXiv:2509.06266) training-free cognitive map. 给 DeepSeek
(无 vision encoder) 一个 3D 空间查询 API, SE(3) 等变 (旋转 = 矩阵乘法), 替代
text-centric 推理.

env flag HUGINN_USE_COGNITIVE_MAP=1 控制 CodeAct 注入. 默认 off, 行为完全不变.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from huginn.metacog.structure_cognitive_map import StructureCognitiveMap

logger = logging.getLogger(__name__)

# 全局 map registry: map_id -> StructureCognitiveMap
# ponytail: 进程内 dict, 不上 disk cache. 跟 atomworld_tool 不一样 (atomworld 无状态)
# ceiling: 多进程不共享, 升级路径接 P15 EngineState.cognitive_maps 持久化.
_MAPS: dict[str, StructureCognitiveMap] = {}


def is_available() -> bool:
    """cognitive map 依赖 pymatgen + scipy, 检查是否装."""
    try:
        import pymatgen  # noqa: F401
        import scipy  # noqa: F401
        return True
    except ImportError:
        return False


def cognitive_map_from_cif(cif_str: str) -> str:
    """从 CIF 构建 cognitive map, 返 map_id."""
    m = StructureCognitiveMap.from_cif(cif_str)
    map_id = f"map_{uuid.uuid4().hex[:8]}"
    _MAPS[map_id] = m
    return map_id


def cognitive_map_query(map_id: str, query_type: str, **params: Any) -> dict:
    """统一查询接口.

    query_type in {"distance", "angle", "neighbors", "subgraph",
                   "after_rotation", "after_translation"}
    """
    m = _get_map(map_id)
    if query_type == "distance":
        return {"distance": m.query_distance(params["i"], params["j"])}
    elif query_type == "angle":
        return {"angle": m.query_angle(params["i"], params["j"], params["k"])}
    elif query_type == "neighbors":
        cutoff = params.get("cutoff")
        return {"neighbors": m.query_neighbors(params["i"], cutoff=cutoff)}
    elif query_type == "subgraph":
        result = m.query_subgraph(params["centers"], hops=params.get("hops", 2))
        return {"nodes": result.nodes, "edges": result.edges, "species": result.node_species}
    elif query_type == "after_rotation":
        coords = m.query_after_rotation(params["indices"], params["axis"],
                                        params["angle"], degrees=params.get("degrees", True),
                                        origin=params.get("origin"))
        return {"coords": coords}
    elif query_type == "after_translation":
        coords = m.query_after_translation(params["indices"], params["vector"])
        return {"coords": coords}
    else:
        raise ValueError(f"unknown query_type: {query_type}")


def cognitive_map_transform(map_id: str, op: str, **params: Any) -> str:
    """SE(3) transform, 返新 map_id.

    op in {"rotate", "translate", "supercell", "remove_atom", "add_atom", "swap_atoms"}
    """
    m = _get_map(map_id)
    if op == "rotate":
        new_m = m.rotate(params["axis"], params["angle"],
                         origin=params.get("origin"), degrees=params.get("degrees", True))
    elif op == "translate":
        new_m = m.translate(params["vector"])
    elif op == "supercell":
        new_m = m.supercell(params["scale"])
    elif op == "remove_atom":
        new_m = m.remove_atom(params["index"])
    elif op == "add_atom":
        new_m = m.add_atom(params["species"], params["coord"])
    elif op == "swap_atoms":
        new_m = m.swap_atoms(params["i"], params["j"])
    else:
        raise ValueError(f"unknown op: {op}")
    new_id = f"map_{uuid.uuid4().hex[:8]}"
    _MAPS[new_id] = new_m
    return new_id


def cognitive_map_to_text(map_id: str, max_atoms: int = 50) -> str:
    """text summary 给 LLM 推理. species + 前 max_atoms 原子 coords + lattice."""
    m = _get_map(map_id)
    lines = [f"Structure Cognitive Map (n={len(m)}, "
             f"{'crystal' if m.lattice is not None else 'molecule'})"]
    if m.lattice is not None:
        lines.append("Lattice:")
        for i, row in enumerate(m.lattice):
            lines.append(f"  [{'abc'[i]}] = ({row[0]:.3f}, {row[1]:.3f}, {row[2]:.3f}) Å")
    lines.append(f"Atoms (first {min(max_atoms, len(m))}/{len(m)}):")
    for i, (sp, c) in enumerate(zip(m.species[:max_atoms], m.coords[:max_atoms])):
        lines.append(f"  [{i}] {sp}: ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}) Å")
    if m.adjacency:
        n_edges = sum(len(v) for v in m.adjacency.values()) // 2
        lines.append(f"Adjacency: {n_edges} edges")
    return "\n".join(lines)


def _get_map(map_id: str) -> StructureCognitiveMap:
    if map_id not in _MAPS:
        raise KeyError(f"unknown map_id: {map_id}")
    return _MAPS[map_id]


def _selfcheck() -> None:
    """3 场景: from_cif + query + transform."""
    # NaCl CIF (P1, 不写 spacegroup 避免 pymatgen 展开)
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
    print("1. cognitive_map_from_cif + to_text...")
    map_id = cognitive_map_from_cif(nacl_cif)
    assert map_id.startswith("map_")
    text = cognitive_map_to_text(map_id)
    assert "Na" in text
    print(f"   OK: map_id={map_id}")

    print("2. cognitive_map_query...")
    result = cognitive_map_query(map_id, "distance", i=0, j=1)
    assert "distance" in result
    print(f"   OK: d(0,1)={result['distance']:.3f}")

    print("3. cognitive_map_transform (rotate z=90, SE(3) 等变)...")
    new_id = cognitive_map_transform(map_id, "rotate", axis="z", angle=90)
    assert new_id != map_id
    d_orig = cognitive_map_query(map_id, "distance", i=0, j=1)["distance"]
    d_rot = cognitive_map_query(new_id, "distance", i=0, j=1)["distance"]
    assert abs(d_orig - d_rot) < 1e-6, f"SE(3) 不等变: {d_orig} vs {d_rot}"
    print(f"   OK: rotate 前后 d 不变 ({d_orig:.3f} -> {d_rot:.3f})")

    print("\nstructure_cognitive_map_tool selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
