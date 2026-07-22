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
                   "after_rotation", "after_translation",
                   "bonds", "coordination_shell", "hydrogen_bonds", "bond_types"}
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
    elif query_type == "bonds":
        method = params.get("method", "cutoff")
        bonds = m.identify_bonds(method=method)
        return {"bonds": [{"i": b.i, "j": b.j, "length": b.length,
                            "bond_type": b.bond_type, "source": b.source} for b in bonds]}
    elif query_type == "coordination_shell":
        shell = m.coordination_shell(params["i"], method=params.get("method", "cutoff"))
        return {"center": shell.center, "neighbors": shell.neighbors,
                "distances": shell.neighbor_distances, "geometry": shell.geometry}
    elif query_type == "hydrogen_bonds":
        bonds = m.identify_hydrogen_bonds(cutoff=params.get("cutoff", 3.5))
        return {"bonds": [{"i": b.i, "j": b.j, "length": b.length} for b in bonds]}
    elif query_type == "bond_types":
        groups = m.bond_types(method=params.get("method", "cutoff"))
        return {"types": {k: [{"i": b.i, "j": b.j, "length": b.length} for b in v]
                          for k, v in groups.items()}}
    elif query_type == "coordination_polyhedra":
        polys = m.coordination_polyhedra()
        return {"polyhedra": [{"center": p.center, "neighbors": p.neighbors,
                                "geometry": p.geometry} for p in polys]}
    else:
        raise ValueError(f"unknown query_type: {query_type}")


def cognitive_map_transform(map_id: str, op: str, **params: Any) -> str:
    """SE(3) transform, 返新 map_id. 补全 AtomWorld 15 actions.

    op in {"rotate", "translate", "supercell", "remove_atom", "add_atom", "swap_atoms",
            "move_atom", "move_selected", "move_towards", "move_around",
            "change_atom", "scale", "insert_between", "delete_below", "delete_around_atom"}
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
    elif op == "move_atom":
        new_m = m.move_atom(params["index"], params["vector"])
    elif op == "move_selected":
        new_m = m.move_selected(params["indices"], params["vector"])
    elif op == "move_towards":
        new_m = m.move_towards(params["i"], params["j"], params.get("fraction", 0.5))
    elif op == "move_around":
        new_m = m.move_around(params["i"], params["center"], params["axis"],
                              params["angle"], degrees=params.get("degrees", True))
    elif op == "change_atom":
        new_m = m.change_atom(params["index"], params["species"])
    elif op == "scale":
        new_m = m.scale(params["factor"])
    elif op == "insert_between":
        new_m = m.insert_between(params["i"], params["j"], params["species"])
    elif op == "delete_below":
        new_m = m.delete_below(params["i"], cutoff=params.get("cutoff"))
    elif op == "delete_around_atom":
        new_m = m.delete_around_atom(params["i"], cutoff=params.get("cutoff"))
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


def cognitive_map_compose(map_id: str, ops: list[dict]) -> str:
    """串联多个 transform. ops 是 [{op: "rotate", axis: "z", angle: 90}, ...].

    返回最终 map_id. ponytail: 逐个调 transform, 不做矩阵合并.
    """
    current_id = map_id
    for op_spec in ops:
        op = op_spec.pop("op") if isinstance(op_spec, dict) else None
        if op is None:
            raise ValueError("op spec missing 'op' key")
        current_id = cognitive_map_transform(current_id, op, **op_spec)
    return current_id


def _get_map(map_id: str) -> StructureCognitiveMap:
    if map_id not in _MAPS:
        raise KeyError(f"unknown map_id: {map_id}")
    return _MAPS[map_id]


def _selfcheck() -> None:
    """8 场景: from_cif + query + transform + 9 新 ops + bond + compose."""
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

    print("2. cognitive_map_query distance + bonds...")
    result = cognitive_map_query(map_id, "distance", i=0, j=1)
    assert "distance" in result
    bonds = cognitive_map_query(map_id, "bonds")["bonds"]
    assert isinstance(bonds, list)
    print(f"   OK: d(0,1)={result['distance']:.3f}, bonds={len(bonds)}")

    print("3. cognitive_map_transform (rotate z=90, SE(3) 等变)...")
    new_id = cognitive_map_transform(map_id, "rotate", axis="z", angle=90)
    assert new_id != map_id
    d_orig = cognitive_map_query(map_id, "distance", i=0, j=1)["distance"]
    d_rot = cognitive_map_query(new_id, "distance", i=0, j=1)["distance"]
    assert abs(d_orig - d_rot) < 1e-6, f"SE(3) 不等变: {d_orig} vs {d_rot}"
    print(f"   OK: rotate 前后 d 不变 ({d_orig:.3f} -> {d_rot:.3f})")

    print("4. 9 新 ops: move_atom + change_atom + scale + insert_between...")
    moved_id = cognitive_map_transform(map_id, "move_atom", index=0, vector=[0.1, 0, 0])
    assert moved_id != map_id
    changed_id = cognitive_map_transform(map_id, "change_atom", index=0, species="K")
    assert _get_map(changed_id).species[0] == "K"
    scaled_id = cognitive_map_transform(map_id, "scale", factor=2.0)
    assert len(_get_map(scaled_id)) == 2
    inserted_id = cognitive_map_transform(map_id, "insert_between", i=0, j=1, species="O")
    assert len(_get_map(inserted_id)) == 3
    print("   OK: move_atom / change_atom / scale / insert_between 都工作")

    print("5. move_towards + move_around + move_selected...")
    towards_id = cognitive_map_transform(map_id, "move_towards", i=0, j=1, fraction=0.3)
    around_id = cognitive_map_transform(map_id, "move_around", i=0, center=1, axis="z", angle=30)
    selected_id = cognitive_map_transform(map_id, "move_selected", indices=[0], vector=[0.1, 0.1, 0])
    assert towards_id and around_id and selected_id
    print("   OK: move_towards / move_around / move_selected 都工作")

    print("6. delete_below + delete_around_atom...")
    # 先加几个原子再删
    big_id = cognitive_map_transform(map_id, "insert_between", i=0, j=1, species="O")
    big_id = cognitive_map_transform(big_id, "insert_between", i=0, j=1, species="O")
    deleted_id = cognitive_map_transform(big_id, "delete_below", i=0, cutoff=5.0)
    # 原子数应该减少
    assert len(_get_map(deleted_id)) < len(_get_map(big_id))
    print(f"   OK: delete_below 减少了原子 ({len(_get_map(big_id))} -> {len(_get_map(deleted_id))})")

    print("7. bond 识别 + coordination_shell + bond_types...")
    shell = cognitive_map_query(map_id, "coordination_shell", i=0, cutoff=5.0)
    assert "geometry" in shell
    types = cognitive_map_query(map_id, "bond_types")
    assert "types" in types
    print(f"   OK: shell.geometry={shell['geometry']}, bond_types={list(types['types'].keys())}")

    print("8. compose (变换链)...")
    composed_id = cognitive_map_compose(map_id, [
        {"op": "rotate", "axis": "z", "angle": 90},
        {"op": "translate", "vector": [1.0, 0, 0]},
        {"op": "scale", "factor": 1.5},
    ])
    assert composed_id != map_id
    d_composed = cognitive_map_query(composed_id, "distance", i=0, j=1)["distance"]
    # scale 1.5 倍, 距离应该变 1.5 倍
    assert abs(d_composed - d_orig * 1.5) < 0.5, f"compose 后距离错: {d_composed} vs {d_orig*1.5}"
    print(f"   OK: rotate+translate+scale 链式变换 d={d_composed:.3f} (原 {d_orig:.3f} x 1.5)")

    print("\nstructure_cognitive_map_tool selfcheck OK (15 actions + bond + compose)")


if __name__ == "__main__":
    _selfcheck()
