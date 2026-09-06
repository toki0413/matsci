"""科学结论超图 (ClaimHypergraph) — 结论-前提依赖的 n 元语义层.

为什么需要超图 (高阶网络视角, Fujita & Smarandache 2026 向下闭包判据):
  科学结论通常是"合取依赖" — 结论 C 要成立, 前提 {A, B, D} 必须同时成立.
  这违反向下闭包 (simplicial complex 强制超集交互⇒子集也存在): 单取 {A, B}
  不构成任何独立成立的结论. 因此结论依赖是超图而非普通二元图/单纯复形.

  现有 ProjectKnowledgeGraph 只存二元边, 把 A→C, B→C, D→C 分开存会丢失
  "合取"语义 — 无法区分"任意一个被挑战就推翻 C"(OR) 与"全部成立才支持 C"
  (AND). ClaimHypergraph 补上这一层:

    - 超边 = {conclusion, premises, mode: AND|OR, condition(适用边界),
              evidence_strength(证据强度), source, status}
    - impact_propagation: 挑战源头假设 → 沿超边前向传播, 计算受影响下游结论
    - citation_cycles: 依赖环检测 (自指环, β₁>0 的拓扑签名), 标记"引用链惯性"

持久化: 独立 claim_hypergraph.json, 沿用 atomic_write_json, 不动 project_kg.json.
ponytail: 轻量 dict/list 实现, 零新依赖. 超图规模 < 10K 边时性能可接受.
升级路径: 需要严格拓扑不变量时换成 TopoNetX SimplicialComplex.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from huginn.utils.common import atomic_write_json

# 结论状态机 — research_log 的 refuted/superseded 语义复用到文献结论层
STATUS_ACCEPTED = "accepted"
STATUS_CONTESTED = "contested"
STATUS_SUPERSEDED = "superseded"
STATUS_RECHECK = "recheck"
VALID_STATUSES = {STATUS_ACCEPTED, STATUS_CONTESTED, STATUS_SUPERSEDED, STATUS_RECHECK}

MODE_AND = "AND"
MODE_OR = "OR"
VALID_MODES = {MODE_AND, MODE_OR}

# 影响传播的受影响程度档位
IMPACT_RECHECK = "recheck"       # AND 超边: 前提被挑战 → 结论需重查
IMPACT_WEAKENED = "weakened"     # OR 超边: 单前提被挑战 → 结论削弱但保留
IMPACT_ROOT = "root"             # 被挑战的源头结论自身


@dataclass
class ClaimHyperedge:
    """一条结论-前提超边 (n 元). conclusion 是被支持/可被挑战的结论节点.

    premises 里的每一项可以是:
      - 另一个 CLAIM 节点 id (结论依赖结论)
      - 一个前提实体节点 id (Material/Tool/Method/Literature...)
    mode = AND 表示所有前提同时成立才支持 conclusion; OR 表示任一前提成立即可.
    condition 记录适用边界 ("只在 XX 条件下成立") — 同结论不同条件=不同超边.
    evidence_strength ∈ [0,1]: 直接实验/复现 > 单样本 > 理论推导 (由上层评估).
    """

    conclusion: str
    premises: list[str] = field(default_factory=list)
    mode: str = MODE_AND
    condition: str = ""
    evidence_strength: float = 0.5
    source: str = "auto"
    status: str = STATUS_ACCEPTED
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode 必须是 {sorted(VALID_MODES)}, got {self.mode!r}")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status 必须是 {sorted(VALID_STATUSES)}, got {self.status!r}")
        self.premises = [p for p in self.premises if p]
        if not self.premises:
            raise ValueError("premises 至少需要 1 个前提")


class ClaimHypergraph:
    """科学结论超图 — 结论-前提 n 元依赖, 支持挑战传播与自指环检测.

    Usage::

        chg = ClaimHypergraph(Path(workspace))
        chg.add_claim("CLAIM:..., 结论X", ["CLAIM:..., 假设A", "Fact:..., 数据B"],
                      mode="AND", condition="低温区间", evidence_strength=0.9,
                      source="doi:...")
        report = chg.impact_propagation("CLAIM:..., 假设A")
        cycles = chg.citation_cycles()
    """

    FILENAME = "claim_hypergraph.json"

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / self.FILENAME
        self._edges: list[ClaimHyperedge] = []
        self._lock = threading.RLock()
        if self.path.exists():
            self.load()

    # ── 持久化 ──

    def load(self) -> None:
        with self._lock:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._edges = [
                ClaimHyperedge(**e) for e in data.get("edges", [])
            ]

    def save(self) -> None:
        with self._lock:
            atomic_write_json(
                self.path,
                {"edges": [asdict(e) for e in self._edges]},
                indent=2,
            )

    # ── 超边写入 ──

    def add_claim(
        self,
        conclusion: str,
        premises: list[str],
        *,
        mode: str = MODE_AND,
        condition: str = "",
        evidence_strength: float = 0.5,
        source: str = "auto",
        status: str = STATUS_ACCEPTED,
    ) -> ClaimHyperedge:
        """写入一条结论-前提超边并返回.

        同一 (conclusion, mode, condition) 已存在时更新证据强度/来源, 不重复堆边.
        """
        he = ClaimHyperedge(
            conclusion=conclusion,
            premises=premises,
            mode=mode,
            condition=condition,
            evidence_strength=evidence_strength,
            source=source,
            status=status,
        )
        with self._lock:
            for existing in self._edges:
                if (
                    existing.conclusion == he.conclusion
                    and existing.mode == he.mode
                    and existing.condition == he.condition
                ):
                    existing.premises = he.premises
                    existing.evidence_strength = he.evidence_strength
                    existing.source = he.source
                    if he.status != STATUS_ACCEPTED:
                        existing.status = he.status
                    return existing
            self._edges.append(he)
        return he

    def update_status(self, claim_id: str, status: str) -> None:
        """更新某结论在所有超边里的状态 (accepted/contested/superseded/recheck)."""
        if status not in VALID_STATUSES:
            raise ValueError(f"status 必须是 {sorted(VALID_STATUSES)}, got {status!r}")
        with self._lock:
            changed = False
            for e in self._edges:
                if e.conclusion == claim_id and e.status != status:
                    e.status = status
                    changed = True
            if changed:
                self.save()

    def get_edges_for_conclusion(self, claim_id: str) -> list[ClaimHyperedge]:
        """返回以 claim_id 为结论的所有超边."""
        with self._lock:
            return [e for e in self._edges if e.conclusion == claim_id]

    def get_edges_with_premise(self, claim_id: str) -> list[ClaimHyperedge]:
        """返回把 claim_id 当前提的所有超边 (它的下游消费者)."""
        with self._lock:
            return [e for e in self._edges if claim_id in e.premises]

    # ── 依赖查询 (回溯/前向) ──

    def upstream_premises(self, claim_id: str, recursive: bool = False) -> list[str]:
        """回溯: 返回结论的源头前提. recursive=True 时沿前提链递归到根."""
        with self._lock:
            direct: set[str] = set()
            for e in self.get_edges_for_conclusion(claim_id):
                direct.update(e.premises)
            if not recursive:
                return sorted(direct)
            seen: set[str] = set(direct)
            stack = list(direct)
            while stack:
                node = stack.pop()
                for e in self.get_edges_for_conclusion(node):
                    for p in e.premises:
                        if p not in seen:
                            seen.add(p)
                            stack.append(p)
            return sorted(seen)

    def downstream_claims(self, claim_id: str, recursive: bool = False) -> list[tuple[str, int]]:
        """前向: 返回直接依赖 claim_id 的下游结论 [(claim, depth)].

        depth 表示离源头的依赖距离 (直接依赖=1). recursive=True 时继续往下.
        """
        with self._lock:
            result: list[tuple[str, int]] = []
            seen: set[str] = set()
            stack: list[tuple[str, int]] = [(claim_id, 0)]
            while stack:
                node, depth = stack.pop()
                for e in self.get_edges_with_premise(node):
                    if e.conclusion == node:
                        continue  # 防自指边死循环
                    if e.conclusion not in seen:
                        seen.add(e.conclusion)
                        result.append((e.conclusion, depth + 1))
                        if recursive:
                            stack.append((e.conclusion, depth + 1))
            result.sort(key=lambda x: (x[1], x[0]))
            return result

    # ── 挑战传播 (debug 人类认知的核心) ──

    def impact_propagation(self, challenged_claim: str, max_hops: int = 10) -> dict[str, Any]:
        """挑战源头 → 沿超边前向传播, 计算受影响下游结论.

        传播规则 (高阶网络 AND/OR 语义):
          - AND 超边: 前提被挑战 ⇒ 结论受影响 (recheck) — 合取依赖下任一前提
            不成立即推翻结论. 继续向下传播.
          - OR  超边: 单前提被挑战 ⇒ 结论削弱 (weakened) 但保留 — 还有其他
            前提支撑. 记录但**不**继续传播 (还有替代支撑).
          - 自指边 (conclusion == premise) 直接跳过, 不传播.

        返回 ImpactReport::

            {
              "challenged": challenged_claim,
              "root": {...},                       # 被挑战源头自身状态
              "affected": [                        # 受影响下游, 按 depth 排序
                 {"claim": ..., "impact": "recheck|weakened", "depth": N,
                  "via": [前提路径], "edge": {超边快照}}
              ],
              "unaffected_count": N,               # 走 OR 超边但未被推翻的
              "hops": M,
            }
        """
        with self._lock:
            if not any(
                e.conclusion == challenged_claim or challenged_claim in e.premises
                for e in self._edges
            ):
                return {
                    "challenged": challenged_claim,
                    "root": None,
                    "affected": [],
                    "unaffected_count": 0,
                    "hops": 0,
                }

            root_status = next(
                (e.status for e in self._edges if e.conclusion == challenged_claim),
                STATUS_RECHECK,
            )
            affected: list[dict[str, Any]] = []
            visited: set[str] = set()
            stack: list[tuple[str, int, list[str]]] = [
                (challenged_claim, 0, [challenged_claim])
            ]
            hops = 0
            while stack and hops < max_hops:
                node, depth, path = stack.pop()
                hops += 1
                for e in self.get_edges_with_premise(node):
                    if e.conclusion == node or e.conclusion in visited:
                        continue
                    if e.mode == MODE_AND:
                        impact = IMPACT_RECHECK
                        visited.add(e.conclusion)
                        affected.append({
                            "claim": e.conclusion,
                            "impact": impact,
                            "depth": depth + 1,
                            "via": path + [e.conclusion],
                            "edge": asdict(e),
                        })
                        stack.append((e.conclusion, depth + 1, path + [e.conclusion]))
                    else:  # OR: 削弱不推翻, 不向下传播
                        # 仍可能影响但非决定性 — 记录为 weakened
                        visited.add(e.conclusion)
                        affected.append({
                            "claim": e.conclusion,
                            "impact": IMPACT_WEAKENED,
                            "depth": depth + 1,
                            "via": path + [e.conclusion],
                            "edge": asdict(e),
                        })

            # OR 超边产生的 weakened 不阻断后续 AND 传播: 上面已统一入队,
            # 这里只需按 depth 排序输出.
            affected.sort(key=lambda x: (x["depth"], x["claim"]))
            return {
                "challenged": challenged_claim,
                "root": {
                    "claim": challenged_claim,
                    "status": root_status,
                    "impact": IMPACT_ROOT,
                    "depth": 0,
                },
                "affected": affected,
                "unaffected_count": 0,
                "hops": hops,
            }

    # ── 自指环检测 (β₁ 拓扑签名) ──

    def citation_cycles(self) -> list[list[str]]:
        """检测依赖环 (自指环): 结论 A 依赖 B, B 依赖 A ... 强连通分量 > 1 节点.

        对应用户观察的"引用链惯性": 一个几十年前的源头判断被后续论文不断引用,
        形成自我印证的环. 环上的结论建议标记为 recheck — 它们是 Hodge 分解里
        无源无汇的"调和分量", 标准图 Laplacian 不可见.

        返回: 环列表, 每个环是 [claim1, claim2, ...] (依赖方向沿边).
        """
        with self._lock:
            # 建二元投影: 结论 → 它依赖的前提 (含前提是另一结论的情况)
            dep: dict[str, set[str]] = {}
            for e in self._edges:
                dep.setdefault(e.conclusion, set()).update(
                    p for p in e.premises if p != e.conclusion
                )

            # Tarjan SCC (networkx 已有, 直接复用)
            import networkx as nx

            g = nx.DiGraph()
            for c, pres in dep.items():
                for p in pres:
                    if p in dep:  # 只关心结论节点间的依赖, 跳过叶子前提实体
                        g.add_edge(c, p)
            cycles: list[list[str]] = []
            for comp in nx.strongly_connected_components(g):
                if len(comp) > 1:
                    cycles.append(sorted(comp))
            return cycles

    def stats(self) -> dict[str, Any]:
        with self._lock:
            statuses: dict[str, int] = {}
            modes: dict[str, int] = {}
            for e in self._edges:
                statuses[e.status] = statuses.get(e.status, 0) + 1
                modes[e.mode] = modes.get(e.mode, 0) + 1
            return {
                "edges": len(self._edges),
                "conclusions": len({e.conclusion for e in self._edges}),
                "statuses": statuses,
                "modes": modes,
                "cycles": len(self.citation_cycles()),
            }


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        chg = ClaimHypergraph(tmp)
        # 合取依赖: 结论X 依赖 假设A + 数据B (AND)
        chg.add_claim("CLAIM:X", ["CLAIM:A", "Fact:B"], mode="AND",
                      condition="低温区间", evidence_strength=0.9, source="doi:1")
        # OR 依赖: 结论Y 依赖 数据B 或 数据B2
        chg.add_claim("CLAIM:Y", ["Fact:B", "Fact:B2"], mode="OR",
                      evidence_strength=0.6, source="doi:2")
        # 结论Z 依赖 结论X (结论依赖结论)
        chg.add_claim("CLAIM:Z", ["CLAIM:X"], mode="AND", source="doi:3")

        # 挑战源头 A → X (AND recheck) → Z (AND recheck)
        r = chg.impact_propagation("CLAIM:A")
        assert r["root"]["claim"] == "CLAIM:A", r
        claims = {a["claim"]: a["impact"] for a in r["affected"]}
        assert claims["CLAIM:X"] == "recheck", claims
        assert claims["CLAIM:Z"] == "recheck", claims
        # Y 走 OR, 前提 B 不是被挑战者, 不出现
        assert "CLAIM:Y" not in claims, claims
        print("1. impact_propagation AND 链传播 OK")

        # 挑战 B (Fact) → X (AND recheck) + Y (OR weakened)
        r2 = chg.impact_propagation("Fact:B")
        claims2 = {a["claim"]: a["impact"] for a in r2["affected"]}
        assert claims2["CLAIM:X"] == "recheck", claims2
        assert claims2["CLAIM:Y"] == "weakened", claims2
        print("2. OR weakened 不推翻 OK")

        # 自指环: A → A 环 (2 节点), 无其他环
        chg2 = ClaimHypergraph(tempfile.mkdtemp(prefix="chg_cycle_"))
        chg2.add_claim("CLAIM:A", ["CLAIM:B"], mode="AND")
        chg2.add_claim("CLAIM:B", ["CLAIM:A"], mode="AND")
        cycles = chg2.citation_cycles()
        assert len(cycles) == 1 and len(cycles[0]) == 2, cycles
        print("3. 自指环检测 OK")

        # 持久化 round-trip
        chg.save()
        chg3 = ClaimHypergraph(tmp)
        assert len(chg3._edges) == 3, len(chg3._edges)
        print("4. 持久化 round-trip OK")

        # status 更新
        chg.update_status("CLAIM:X", "contested")
        assert chg.get_edges_for_conclusion("CLAIM:X")[0].status == "contested"
        print("5. status 更新 OK")

        print("all claim-hypergraph checks passed")
