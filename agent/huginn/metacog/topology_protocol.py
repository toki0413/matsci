"""A. 拓扑 Protocol — Bourbaki 母结构视角下的统一邻域接口.

布尔巴基三母结构 (代数 / 序 / 拓扑) 中, 拓扑结构最弱也最通用: 只要求
"邻域" 概念存在. 视觉 token / 文本 token / 3D 结构 / 历史记忆虽然载体不同,
但都有"距离 ≤ radius 的点集"概念. 本模块把这种共识显式化成 Protocol.

  StructureCognitiveMap: neighborhood(i, cutoff) -> {原子 index}
  ImageIndex:           neighborhood(query, top_k) -> {image_id}
  hippocampus history:   neighborhood(query, top_k) -> {entry}
  M6 box primitives:     neighborhood(label, max_area) -> {box}

Protocol 用 runtime_checkable, duck typing 不强求继承. 各类只需加一个
`neighborhood` 方法 (一行 delegation 到已有的 query_neighbors / search).

跟 C 实验 (composite_token_experiment) 的关系:
  拓扑结构是三结构兼容叠加的最小公约数. 先统一拓扑接口,
  SE(3) 群作用 (代数 II) 才有合法的作用空间.

接入点:
  - 结构模块: 结构实现 neighborhood (delegation to query_neighbors)
  - perception: ImageIndex 实现 neighborhood (delegation to search)
  - metacog: hippocampus + box 通过 adapter 函数封装
  - env flag HUGINN_USE_TOPOLOGY=1 控制 selfcheck, 默认 off
"""
from __future__ import annotations

import json
import os
from typing import Any, Protocol, runtime_checkable

_TOPOLOGY_FLAG = "HUGINN_USE_TOPOLOGY"


def use_topology() -> bool:
    """HUGINN_USE_TOPOLOGY=1 才开 (selfcheck 用). 默认 off."""
    return os.environ.get(_TOPOLOGY_FLAG, "0") == "1"


# ── Protocol ────────────────────────────────────────────────────────────────


@runtime_checkable
class SupportsNeighborhood(Protocol):
    """统一拓扑接口: 邻域查询.

    任何有"距离 ≤ radius 的点集"概念的对象都可实现. duck typing, 不强求继承.
    实现: 加一个 `neighborhood(x, radius=None) -> set` 方法即可.

    Args:
        x: 查询点 (类型由实现定: int / str / np.ndarray / dict)
        radius: 邻域半径 (None = 用对象默认)

    Returns:
        set: 邻域内点的集合 (元素类型由实现定)
    """

    def neighborhood(self, x: Any, radius: float | None = None) -> set[Any]:
        """返回 x 的邻域 (距离 ≤ radius 的点集)."""
        ...


# ── Adapter wrappers (for function-style modules) ─────────────────────────


class HippocampusView:
    """把 hippocampus 的 list[str] history 包装成满足 SupportsNeighborhood.

    hippocampus 是函数式 (record/recall/forget 操作 list), 不是类.
    本 wrapper 持有 history 引用, 暴露 neighborhood 方法, 让 Protocol 能套上.
    """

    def __init__(self, history: list[str], tau_s: float = 3600.0) -> None:
        self._history = history
        self._tau_s = tau_s

    def neighborhood(self, x: Any, radius: float | None = None) -> set[Any]:
        """x = query string, radius = top_k (None=5).

        Returns:
            set of entry JSON strings (top_k 相似记忆)
        """
        from huginn.metacog.visual_hippocampus import recall
        top_k = int(radius) if radius is not None else 5
        results = recall(
            self._history, query=x if isinstance(x, str) else None,
            text_query=x if isinstance(x, str) else None,
            top_k=top_k, decay=True, tau_s=self._tau_s,
        )
        return {json.dumps(r["entry"], ensure_ascii=False) for r in results}


class BoxPrimitivesView:
    """把 M6 box primitives (list[dict]) 包装成满足 SupportsNeighborhood.

    M6 extract_box_primitives 返回 str, parse_box_primitive 返回 list[dict].
    本 wrapper 持有 boxes list, 暴露 neighborhood 方法.
    """

    def __init__(self, boxes: list[dict[str, Any]]) -> None:
        self._boxes = boxes

    def neighborhood(self, x: Any, radius: float | None = None) -> set[Any]:
        """x = (cx, cy) 中心坐标, radius = 搜索半径 (像素, normalized 0-999).

        Returns:
            set of box label strings (中心在 radius 内的 boxes)
        """
        if not isinstance(x, (tuple, list)) or len(x) < 2:
            return set()
        cx, cy = float(x[0]), float(x[1])
        max_dist = float(radius) if radius is not None else 200.0
        result: set[str] = set()
        for b in self._boxes:
            coords = b.get("coordinates", [])
            if len(coords) < 4:
                continue
            bx = (coords[0] + coords[2]) / 2.0
            by = (coords[1] + coords[3]) / 2.0
            dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
            if dist <= max_dist:
                lbl = b.get("label", "unknown")
                result.add(lbl)
        return result


# ── 通用 neighborhood helper ──────────────────────────────────────────────


def neighborhood_of(obj: Any, x: Any, radius: float | None = None) -> set[Any]:
    """通用邻域查询: duck typing.

    优先级:
      1. obj 有 neighborhood 方法 → 直接调 (SupportsNeighborhood)
      2. obj 有 query_neighbors → delegation (StructureCognitiveMap 兼容)
      3. obj 有 search → delegation (ImageIndex 兼容)
      4. obj 是 list[str] → HippocampusView 包装
      5. obj 是 list[dict] 含 coordinates → BoxPrimitivesView 包装
      6. 其他 → TypeError

    ponytail: 用 hasattr duck typing, 不上 isinstance. 升级路径: 真接 Protocol.
    """
    if hasattr(obj, "neighborhood"):
        return obj.neighborhood(x, radius)
    if hasattr(obj, "query_neighbors"):
        # StructureCognitiveMap: x=atom_idx, radius=cutoff
        return set(obj.query_neighbors(x, cutoff=radius))
    if hasattr(obj, "search"):
        # ImageIndex: x=query/path/bytes, radius=top_k
        top_k = int(radius) if radius is not None else 5
        results = obj.search(query=x, top_k=top_k)
        return {r.get("path") for r in results}
    if isinstance(obj, list) and obj:
        first = obj[0]
        if isinstance(first, str):
            # hippocampus history (JSON lines)
            return HippocampusView(obj).neighborhood(x, radius)
        if isinstance(first, dict) and "coordinates" in first:
            # M6 boxes
            return BoxPrimitivesView(obj).neighborhood(x, radius)
    raise TypeError(
        f"cannot compute neighborhood of {type(obj).__name__}, "
        f"obj must have neighborhood/query_neighbors/search method or be a "
        f"list[str] (hippocampus) / list[dict] (boxes)"
    )


# ── selfcheck ──────────────────────────────────────────────────────────────


def _selfcheck() -> None:
    """A selfcheck: Protocol + 4 类对象的邻域查询."""
    print("=" * 70)
    print("A 拓扑 Protocol: 统一邻域接口验证")
    print("=" * 70)
    print()

    # 1. StructureCognitiveMap (已有类, 直接满足 Protocol)
    print("[1] StructureCognitiveMap (delegation to query_neighbors)")
    import numpy as np
    from huginn.metacog.structure_cognitive_map import StructureCognitiveMap
    # 4 原子方形: (0,0,0) (1,0,0) (0,1,0) (1,1,0), cutoff=1.5
    m = StructureCognitiveMap.from_coords(
        species=["A", "B", "C", "D"],
        coords=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        cutoff=1.5,
    )
    # 原子 0 的邻域 (cutoff=1.5): 应包含 1, 2 (距离 1.0), 不含 3 (距离 √2 ≈ 1.41 ≤ 1.5)
    n = m.neighborhood(0, radius=1.5)
    assert isinstance(n, set), f"neighborhood must return set, got {type(n)}"
    assert 1 in n and 2 in n, f"expected 1,2 in neighborhood, got {n}"
    print(f"    atom 0 邻域 (cutoff=1.5): {sorted(n)}")
    # 验证 isinstance(Protocol) — runtime_checkable
    assert isinstance(m, SupportsNeighborhood), "StructureCognitiveMap should satisfy Protocol"
    print(f"    isinstance(SupportsNeighborhood): {isinstance(m, SupportsNeighborhood)}")
    print()

    # 2. ImageIndex (已有类, delegation to search)
    print("[2] ImageIndex (delegation to search)")
    from huginn.perception.image_index import ImageIndex
    idx = ImageIndex()  # in-memory, no encoder
    idx.set_encoder(None)  # 离线模式, 关掉 CLIP lazy load
    idx.add_image_bytes(b"fake1", metadata={"caption": "lattice 4Å cubic"})
    idx.add_image_bytes(b"fake2", metadata={"caption": "particles 10"})
    idx.add_image_bytes(b"fake3", metadata={"caption": "spectrum 3 peaks"})
    # text_query 搜 "lattice"
    n = idx.neighborhood("lattice", radius=3)
    assert isinstance(n, set)
    print(f"    text_query='lattice' 邻域: {len(n)} matches")
    assert isinstance(idx, SupportsNeighborhood), "ImageIndex should satisfy Protocol"
    print(f"    isinstance(SupportsNeighborhood): {isinstance(idx, SupportsNeighborhood)}")
    print()

    # 3. hippocampus history (list[str], adapter)
    print("[3] hippocampus history (adapter wrapper)")
    from huginn.metacog.visual_hippocampus import record
    h: list[str] = []
    record(h, "[bands] peak=2.5 lattice", ts=100.0)
    record(h, "[lattice] d=4Å cubic", ts=200.0)
    record(h, "[particles] n=10 detected", ts=300.0)
    view = HippocampusView(h)
    n = view.neighborhood("lattice", radius=2)
    assert isinstance(n, set)
    assert len(n) >= 1, f"expected ≥1 lattice match, got {n}"
    print(f"    query='lattice' 邻域: {len(n)} entries")
    assert isinstance(view, SupportsNeighborhood), "HippocampusView should satisfy Protocol"
    print(f"    isinstance(SupportsNeighborhood): {isinstance(view, SupportsNeighborhood)}")
    print()

    # 4. M6 boxes (list[dict], adapter)
    print("[4] M6 box primitives (adapter wrapper)")
    boxes = [
        {"label": "region0", "coordinates": [100, 100, 200, 200]},
        {"label": "region1", "coordinates": [500, 500, 600, 600]},
        {"label": "region2", "coordinates": [150, 150, 250, 250]},
    ]
    view = BoxPrimitivesView(boxes)
    # 中心 (150, 150), 半径 100 → 应含 region0 (中心 150,150), region2 (中心 200,200)
    n = view.neighborhood((150, 150), radius=100)
    assert isinstance(n, set)
    assert "region0" in n, f"region0 should be in neighborhood, got {n}"
    assert "region2" in n, f"region2 should be in neighborhood, got {n}"
    assert "region1" not in n, f"region1 (中心 550,550) 距离太远, got {n}"
    print(f"    中心 (150,150), 半径 100 → 邻域: {sorted(n)}")
    assert isinstance(view, SupportsNeighborhood), "BoxPrimitivesView should satisfy Protocol"
    print(f"    isinstance(SupportsNeighborhood): {isinstance(view, SupportsNeighborhood)}")
    print()

    # 5. 通用 helper: neighborhood_of 自动 dispatch
    print("[5] neighborhood_of 通用 dispatch")
    # cognitive_map
    n = neighborhood_of(m, 0, radius=1.5)
    assert 1 in n and 2 in n
    print(f"    cognitive_map → {sorted(n)}")
    # image_index
    n = neighborhood_of(idx, "lattice", radius=3)
    print(f"    image_index → {len(n)} matches")
    # hippocampus
    n = neighborhood_of(h, "lattice", radius=2)
    print(f"    hippocampus → {len(n)} entries")
    # boxes
    n = neighborhood_of(boxes, (150, 150), radius=100)
    print(f"    boxes → {sorted(n)}")
    print()

    # 6. 同一接口, 不同载体 — 拓扑统一性
    print("[6] 拓扑统一性: 同一 neighborhood 接口, 4 种不同载体")
    providers = [
        ("StructureCognitiveMap", m, 0, 1.5),
        ("ImageIndex", idx, "lattice", 3),
        ("HippocampusView", HippocampusView(h), "lattice", 2),
        ("BoxPrimitivesView", BoxPrimitivesView(boxes), (150, 150), 100),
    ]
    for name, obj, x, r in providers:
        n = obj.neighborhood(x, radius=r)
        assert isinstance(n, set)
        print(f"    {name:25s}.neighborhood({x!r}, {r}) → {type(n).__name__} of size {len(n)}")
    print()

    print("=" * 70)
    print("结论")
    print("=" * 70)
    print()
    print("4 类对象 (cognitive_map / image_index / hippocampus / M6 boxes)")
    print("都满足 SupportsNeighborhood Protocol, 同一 neighborhood(x, radius) 接口.")
    print()
    print("布尔巴基拓扑结构视角: 不同载体的邻域概念被显式统一.")
    print("这是三结构兼容叠加的最小公约数 (C 实验已验证 SE(3) 代数能穿过).")
    print()
    print("A TOPOLOGY PROTOCOL ALL CHECKS PASSED")


if __name__ == "__main__":
    _selfcheck()
