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
  - DAG 自动依赖推断 — LLM 在 dispatch_parallel 时填 dependencies 字段
  - 并发资源约束 (machine limit) — 用 budget_decomp 的 parallel 硬 cap

天花板: 假设依赖已知. 升级: LLM 不填 dependencies 时退化到全并行.
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

    print("[task_dag] self-check OK (7/7)")
