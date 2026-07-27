"""子任务 DAG — 拓扑序 + 最大反链 + 关键路径.

治 spec 天花板 "并行 dispatch 无依赖感知, 4 个 subagent 可能探同路径".

数学: 子任务建为 DAG G=(V, E), v ∈ V 是子任务, (u,v) ∈ E 表示 u 输出是 v 输入.
  - 拓扑序 (Kahn): O(V+E), 决定执行顺序
  - 并行度 = 最大反链 (Dilworth): 最小链覆盖数 = 最大反链大小
  - 关键路径: DAG 最长路径, wall-clock 下限

Dilworth 定理: 有限偏序集的最大反链大小 = 最小链覆盖数.
对 DAG, 把偏序关系 (u ≤ v iff 存在路径 u→v) 看成偏序集, antichain = 互相
不可达的节点集 (可并行执行). 最小链覆盖 = 用最少的链覆盖所有节点.

接入: dispatch_parallel 接受 tasks + dependencies, 建 DAG, 按拓扑分层,
同层 antichain 内并行 dispatch.

不做 (YAGNI):
  - PERT 加权 (节点耗时) — LLM 不预知 subagent 耗时, 无权 DAG 够用
  - 并发资源约束 (machine limit) — 用 budget_decomp 的 parallel 硬 cap

天花板: 依赖来源优先 LLM 显式填, 缺失时用 provenance registry 反推
(见 infer_dependencies_from_provenance). 升级: 完全无 provenance 时退化到全并行.
"""
from __future__ import annotations

from typing import Any


class TaskDAG:
    """子任务有向无环图.

    用法:
        dag = TaskDAG(tasks=["A","B","C","D","E"],
                      dependencies=[("A","B"),("A","C"),("B","D"),("C","D"),("D","E")])
        order = dag.topological_order()    # [A, B, C, D, E] 或 [A, C, B, D, E]
        width = dag.antichain_width()       # 2 ({B,C} 可并行)
        cp = dag.critical_path()            # [A, B, D, E] 长度 4
        layers = dag.parallel_layers()      # [[A], [B,C], [D], [E]] 按层并行
    """

    def __init__(
        self,
        tasks: list[str],
        dependencies: list[tuple[str, str]] | None = None,
    ) -> None:
        self.tasks = list(tasks)
        self.deps = list(dependencies or [])
        # 邻接表 + 入度
        self._adj: dict[str, list[str]] = {t: [] for t in self.tasks}
        self._indegree: dict[str, int] = {t: 0 for t in self.tasks}
        for u, v in self.deps:
            if u not in self._adj or v not in self._adj:
                raise ValueError(f"dependency ({u},{v}) 引用不存在的 task")
            self._adj[u].append(v)
            self._indegree[v] += 1
        # 环检测 (Kahn 副产品: 拓扑序长度 < 节点数则有环)
        if len(self.topological_order()) != len(self.tasks):
            raise ValueError("DAG 有环, 无法拓扑排序")

    def topological_order(self) -> list[str]:
        """Kahn 算法拓扑排序. O(V+E).

        ponytail: 多次调用重复算, 不缓存 — DAG 通常小 (<20 节点), 无所谓.
        升级: 大 DAG 时缓存 + invalidate on add.
        """
        indeg = dict(self._indegree)
        queue = [t for t in self.tasks if indeg[t] == 0]
        order: list[str] = []
        while queue:
            # 取入度 0 的节点 (不排序, 保持插入顺序稳定)
            node = queue.pop(0)
            order.append(node)
            for nxt in self._adj[node]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
        return order

    def antichain_width(self) -> int:
        """最大反链大小 (Dilworth 定理) = 最小链覆盖数.

        算法: 把 DAG 传递闭包看成偏序集, 二分图匹配求最小链覆盖.
        最小链覆盖 = V - 最大匹配数 (Kőnig 定理在 DAG 上的应用).

        ponytail: networkx.dag_longest_path 不直接给 antichain, 用传递闭包 +
        二分图匹配. 升级: Hopcroft-Karp 替代匈牙利 (大 DAG 时).
        """
        import networkx as nx
        g = nx.DiGraph()
        g.add_nodes_from(self.tasks)
        g.add_edges_from(self.deps)
        # 传递闭包: u ≤ v iff 存在路径 u→v
        tc = nx.transitive_closure(g)
        # 二分图: 左右各一份节点, 边 (u_left, v_right) iff u ≤ v 且 u != v
        # 最小链覆盖 = V - 最大匹配数
        # ponytail: networkx 没有直接的最小路径覆盖, 手动建二分图 + maximum_matching
        bipartite = nx.Graph()
        for t in self.tasks:
            bipartite.add_node((t, "L"))
            bipartite.add_node((t, "R"))
        for u in tc.nodes():
            for v in tc.nodes():
                if u != v and nx.has_path(tc, u, v):
                    bipartite.add_edge((u, "L"), (v, "R"))
        matching = nx.algorithms.matching.max_weight_matching(bipartite, maxcardinality=True)
        # 最大匹配数 = len(matching) (每条匹配边覆盖一个链覆盖的前驱关系)
        max_match = len(matching)
        return len(self.tasks) - max_match

    def critical_path(self) -> list[str]:
        """DAG 最长路径 (关键路径), wall-clock 下限.

        ponytail: networkx.dag_longest_path 直接用. 无权图 = 边数最长.
        升级: 加权 (节点耗时) 时换 dag_longest_path_length.
        """
        import networkx as nx
        g = nx.DiGraph()
        g.add_nodes_from(self.tasks)
        g.add_edges_from(self.deps)
        return nx.dag_longest_path(g)

    def parallel_layers(self) -> list[list[str]]:
        """按拓扑分层, 同层 antichain 内可并行.

        返回 [[layer0], [layer1], ...], layer0 无依赖可先跑, layer1 依赖 layer0, ...
        每层内节点互相不可达 (antichain), 可并行 dispatch.

        ponytail: Kahn 变种, 每轮取所有入度 0 的节点作为一层.
        """
        indeg = dict(self._indegree)
        layers: list[list[str]] = []
        remaining = set(self.tasks)
        while remaining:
            # 当前层 = 所有入度 0 且未处理的节点
            layer = [t for t in self.tasks if indeg[t] == 0 and t in remaining]
            if not layer:
                break  # 防御性, 有环时已 __init__ 拦截
            layers.append(layer)
            for node in layer:
                remaining.discard(node)
                for nxt in self._adj[node]:
                    indeg[nxt] -= 1
        return layers


# ── provenance 反推依赖 (Task 26.8) ────────────────────────

# 路径字段名跟 register_tool_output 对齐, 不另立标准
_PROV_PATH_FIELDS = (
    "file_path", "working_dir", "poscar_path", "structure_file",
    "output_file", "outcar_path", "trajectory_file", "saved_to",
)


def _extract_paths_from_tool_input(tool_input: dict) -> set[str]:
    """从 tool_input 顶层抓路径值. 嵌套结构不递归 — register_tool_output
    自己也只扫顶层, 跟它对齐."""
    paths: set[str] = set()
    if not isinstance(tool_input, dict):
        return paths
    for k in _PROV_PATH_FIELDS:
        v = tool_input.get(k)
        if isinstance(v, str) and v:
            paths.add(v)
    return paths


def infer_dependencies_from_provenance(
    tasks: list[dict],
    provenance_registry,
) -> list[tuple[str, str]]:
    """从 provenance 溯源链反推 task 间依赖 (后验补全).

    对每个 task B, 抽它 tool_input 里出现的路径, 在 registry 里 find_by_path
    反查; 命中则该路径是某个上游 task 的产出. 再用 get_lineage 拿溯源链,
    链上每个上游产出的 file_path 反查"哪个 task 的 tool_input 提到过它",
    从而定位上游 task A, 加 A→B 边.

    LLM 填的 dependencies 当先验, 这里推出来的当后验, 取并集即可.
    LLM 完全不填时, 只要 registry 有数据也能建出正确 DAG.

    Args:
        tasks: list of dict, 每个含 id (str) 和 tool_input (dict).
        provenance_registry: ProvenanceRegistry 实例 (或任何有 find_by_path
            和 get_lineage 的对象).

    Returns:
        list of (A_id, B_id) tuples, A 是 B 的上游.
    """
    if provenance_registry is None:
        return []

    # task_id -> tool_input 里出现的路径集合
    task_paths: dict[str, set[str]] = {}
    # path -> 引用过它的 task_id 集合 (反查上游产出归属哪个 task)
    path_to_tasks: dict[str, set[str]] = {}
    for t in tasks:
        tid = t["id"]
        paths = _extract_paths_from_tool_input(t.get("tool_input") or {})
        task_paths[tid] = paths
        for p in paths:
            path_to_tasks.setdefault(p, set()).add(tid)

    inferred: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for t in tasks:
        bid = t["id"]
        for p in task_paths.get(bid, ()):
            entry = provenance_registry.find_by_path(p)
            if entry is None:
                # registry 里没记录, 判不了上游
                continue
            # get_lineage 返回 p 的溯源链 (含 p 自己), 链头是最新产出
            chain = provenance_registry.get_lineage(p, depth=5)
            for e in chain:
                upstream = e.file_path
                if upstream == p:
                    # p 本身是 B 消费的产出, 上游归属看链后续 entry
                    continue
                for aid in path_to_tasks.get(upstream, ()):
                    if aid != bid:
                        edge = (aid, bid)
                        if edge not in seen:
                            seen.add(edge)
                            inferred.append(edge)
    return inferred


def build_dag_with_provenance(
    tasks: list[dict],
    provenance_registry,
    explicit_deps: list[tuple[str, str]] | None = None,
) -> TaskDAG:
    """LLM 显式 dep 当先验, provenance 推断当后验, 取并集建 DAG."""
    task_ids = [t["id"] for t in tasks]
    inferred = infer_dependencies_from_provenance(tasks, provenance_registry)
    merged: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for d in (explicit_deps or []) + inferred:
        if d not in seen:
            seen.add(d)
            merged.append(d)
    return TaskDAG(tasks=task_ids, dependencies=merged)


# ── selfcheck ──────────────────────────────────────────────

if __name__ == "__main__":
    # 5 节点 DAG: A→B, A→C, B→D, C→D, D→E
    #  A
    # / \
    # B  C
    # \ /
    #  D
    #  |
    #  E
    dag = TaskDAG(
        tasks=["A", "B", "C", "D", "E"],
        dependencies=[("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")],
    )

    # 1. 拓扑序合法 (A 必须在 B/C 前, B/C 在 D 前, D 在 E 前)
    order = dag.topological_order()
    assert order[0] == "A", f"A 应第一, got {order}"
    assert order[-1] == "E", f"E 应最后, got {order}"
    assert order.index("B") < order.index("D"), f"B 应在 D 前, got {order}"
    assert order.index("C") < order.index("D"), f"C 应在 D 前, got {order}"
    assert len(order) == 5
    print(f"[ok] 拓扑序: {order}")

    # 2. 最大反链 = 2 ({B, C} 互相不可达, 可并行)
    width = dag.antichain_width()
    assert width == 2, f"antichain_width 应 2 ({{B,C}}), got {width}"
    print(f"[ok] antichain_width = {width} ({{B,C}} 可并行)")

    # 3. 关键路径长度 4 (A→B→D→E 或 A→C→D→E, 都 4 节点)
    cp = dag.critical_path()
    assert len(cp) == 4, f"关键路径应 4 节点, got {cp}"
    assert cp[0] == "A" and cp[-1] == "E", f"起点 A 终点 E, got {cp}"
    assert "D" in cp, f"D 应在关键路径, got {cp}"
    print(f"[ok] 关键路径: {cp} (长度 {len(cp)})")

    # 4. parallel_layers: [[A], [B,C], [D], [E]]
    layers = dag.parallel_layers()
    assert layers == [["A"], ["B", "C"], ["D"], ["E"]], f"layers 错误: {layers}"
    print(f"[ok] parallel_layers: {layers}")

    # 5. 环检测: A→B→A 应 raise
    try:
        TaskDAG(tasks=["A", "B"], dependencies=[("A", "B"), ("B", "A")])
        raise AssertionError("环 DAG 应 raise ValueError")
    except ValueError as e:
        assert "环" in str(e), f"错误信息应含 '环', got {e}"
        print(f"[ok] 环检测: {e}")

    # 6. 无依赖 DAG: 所有节点同层, antichain_width = N
    dag2 = TaskDAG(tasks=["X", "Y", "Z"])
    assert dag2.antichain_width() == 3, "无依赖 DAG antichain 应 3"
    assert dag2.parallel_layers() == [["X", "Y", "Z"]], "无依赖应单层"
    assert dag2.critical_path() == ["X"], f"无依赖关键路径单节点, got {dag2.critical_path()}"
    print(f"[ok] 无依赖 DAG: width={dag2.antichain_width()}, layers={dag2.parallel_layers()}")

    # 7. 线性链 A→B→C: antichain=1, critical_path=3
    dag3 = TaskDAG(tasks=["A", "B", "C"], dependencies=[("A", "B"), ("B", "C")])
    assert dag3.antichain_width() == 1, "线性链 antichain 应 1"
    assert dag3.critical_path() == ["A", "B", "C"], "线性链关键路径全长"
    assert dag3.parallel_layers() == [["A"], ["B"], ["C"]], "线性链每层 1 节点"
    print(f"[ok] 线性链 A→B→C: width=1, cp=3, layers=[[A],[B],[C]]")

    # 8. provenance 自动推断: B 引用 A 输出但 LLM 未填 dependencies
    #    验证 LLM 完全不填 dependencies 时也能建出正确 DAG
    import os
    import tempfile
    os.environ["HUGINN_CACHE_DIR"] = tempfile.mkdtemp(prefix="huginn_prov_")
    from huginn.provenance.registry import ProvenanceRegistry
    reg = ProvenanceRegistry()
    # task A 产出 A_out.cif (无上游)
    reg.register(file_path="/tmp/A_out.cif", produced_by="relax_tool",
                 input_files=[], file_format="cif")
    # task M 消费 A_out.cif, 产出 M_out.cif
    reg.register(file_path="/tmp/M_out.cif", produced_by="static_tool",
                 input_files=["/tmp/A_out.cif"], file_format="cif")
    tasks_no_dep = [
        {"id": "A", "tool_input": {"output_file": "/tmp/A_out.cif"}},
        {"id": "M", "tool_input": {"output_file": "/tmp/M_out.cif",
                                    "file_path": "/tmp/A_out.cif"}},
        {"id": "B", "tool_input": {"file_path": "/tmp/M_out.cif"}},
    ]
    inferred = infer_dependencies_from_provenance(tasks_no_dep, reg)
    assert ("A", "M") in inferred, f"应推断 A→M, got {inferred}"
    assert ("M", "B") in inferred, f"应推断 M→B, got {inferred}"
    assert ("A", "B") in inferred, f"应推断 A→B (溯源链传递), got {inferred}"
    # build_dag_with_provenance: explicit_deps 留空也能建出有依赖的 DAG
    dag4 = build_dag_with_provenance(tasks_no_dep, reg, explicit_deps=[])
    order4 = dag4.topological_order()
    assert order4.index("A") < order4.index("M") < order4.index("B"), \
        f"拓扑序应 A 先 M 中 B 后, got {order4}"
    print(f"[ok] provenance 推断: {inferred}, 拓扑序 {order4}")

    print("[task_dag] self-check OK (8/8)")
