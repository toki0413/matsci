"""Hypothesis 闭环 — 假设图 + 失败实验驱动的假设修正.

R5 (W4): 把研究假设组织成图, 节点是假设, 边是 support / refute / derive
三种关系. 实验失败 (refute) 时调 RedTeamReviewer 审查失败原因, 生成修正假设
入队, 形成"假设-实验-修正"闭环.

跟 CampaignManager 协同: campaign 里每个 Experiment 绑定一个 hypothesis_id,
实验跑完调 support/refute 更新图状态, refute 触发 refine_failed 产出新假设.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


HypothesisStatus = Literal["untested", "supported", "refuted", "superseded"]
EdgeType = Literal["support", "refute", "derive", "pivot"]


# ── data structures ──────────────────────────────────────────────────────────


@dataclass
class HypothesisNode:
    """图中的一个假设节点."""

    id: str
    statement: str
    rationale: str = ""
    testable_prediction: str = ""
    status: HypothesisStatus = "untested"
    parent_id: str | None = None  # derive 边的源
    evidence: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    # refine_failed 时记录 red-team findings, 方便回溯
    refinement_basis: list[dict[str, Any]] = field(default_factory=list)
    # v11: 假设依赖的核心维度 (composition/temperature/defect/structure/transport).
    # 关键词命中抽取, 非语义. ponytail: 升级路径接 LLM 判定.
    dimension: str = ""
    # v11: pivot 兄弟组 id — 同一失败假设 pivot 出的多个候选共享一个 group.
    # ponytail: 字段驱动, 非 LLM 判定. None = 无兄弟.
    sibling_group_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "rationale": self.rationale,
            "testable_prediction": self.testable_prediction,
            "status": self.status,
            "parent_id": self.parent_id,
            "evidence": dict(self.evidence),
            "created_at": self.created_at,
            "refinement_basis": list(self.refinement_basis),
            "dimension": self.dimension,
            "sibling_group_id": self.sibling_group_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HypothesisNode":
        return cls(
            id=d["id"],
            statement=d.get("statement", ""),
            rationale=d.get("rationale", ""),
            testable_prediction=d.get("testable_prediction", ""),
            status=d.get("status", "untested"),
            parent_id=d.get("parent_id"),
            evidence=d.get("evidence", {}),
            created_at=d.get("created_at", ""),
            refinement_basis=d.get("refinement_basis", []),
            dimension=d.get("dimension", ""),
            sibling_group_id=d.get("sibling_group_id"),
        )


@dataclass
class HypothesisEdge:
    """节点间的关系边."""

    from_id: str
    to_id: str
    edge_type: EdgeType
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_id": self.from_id,
            "to_id": self.to_id,
            "edge_type": self.edge_type,
            "evidence": dict(self.evidence),
        }


# ── 异常 ────────────────────────────────────────────────────────────────────


class HypothesisGraphError(Exception):
    """图操作错误: 节点不存在 / 重复 / 非法状态转移."""


# v11: 假设维度关键词表 — 中英文命中, 非语义. ponytail: 升级路径接 LLM 判定.
# 不新建 DimensionDetector 组件, 只在 add_hypothesis 时命中.
_DIMENSION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "composition": ("ca/si", "ca_si", "al2o3", "掺杂", "doping", "alloy",
                    "composition", "ratio", "化学计量", "stoichiometry"),
    "temperature": ("温度", "temperature", "thermal", "退火", "annealing",
                    "t-dependent", "phase transition", "相变"),
    "defect": ("缺陷", "defect", "vacancy", "空位", "dislocation",
               "位错", "interface", "界面", "itz"),
    "structure": ("结构", "structure", "crystal", "晶体", "lattice",
                  "晶格", "symmetry", "对称", "phase", "相"),
    "transport": ("输运", "diffusion", "扩散", "conductivity",
                  "电导", "mobility", "迁移率", "percolation"),
}


def _extract_dimension(statement: str) -> str:
    """从假设陈述抽 dimension, 命中第一个返回. ponytail: 字符串 in 匹配, 非 embedding."""
    if not statement:
        return ""
    low = statement.lower()
    for dim, keywords in _DIMENSION_KEYWORDS.items():
        for kw in keywords:
            if kw in low:
                return dim
    return ""


# ── graph ────────────────────────────────────────────────────────────────────


class HypothesisGraph:
    """假设图: 节点 + 边 + 失败驱动的假设修正.

    典型流程:
        graph = HypothesisGraph()
        h1 = graph.add_hypothesis("如果掺杂增加, 带隙减小", prediction="...")
        # 实验跑完, 结果不支持
        graph.refute(h1, evidence={"result": "带隙反而增加"})
        # 触发修正
        h2 = graph.refine_failed(h1, evidence={"result": "带隙反而增加"})
        # h2 是新假设, parent=h1, status=untested, 进 campaign 队列
    """

    # 交叉授粉延迟: 分量成熟度达到此阈值才允许跨分量 derive/pivot
    _CROSS_POLLINATION_MATURITY_THRESHOLD = 0.6

    def __init__(self) -> None:
        self._nodes: dict[str, HypothesisNode] = {}
        self._edges: list[HypothesisEdge] = []
        self._events: list[dict[str, Any]] = []
        # 2-单纯形: frozenset[node_id, ev_key_1, ev_key_2] 表示"N 条独立证据作为整体支撑假设".
        # dual_covered 命中时自动注册. 满足 downward closure: 任意 1-子集 (节点本身) 也在图里.
        # ponytail: 用 frozenset 模拟, 不引入新依赖. 升级: SimplicialComplex (gudhi/TopoNetX) 当 >2-ary 关系变常见.
        self._simplicials: set[frozenset[str]] = set()
        # ponytail: in-memory event log, 不持久化. 升级: 写入 session event log
        # (Anthropic Managed Agents: Session as event log, 可 resume/replay/debug)

    def _record_event(self, event_type: str, node_id: str | None = None,
                      **payload: Any) -> None:
        """记录结构化事件到 event log (append-only, 不删除).
        用于回放/调试/状态恢复. 失败学第12节: 把失败变成可回放材料."""
        self._events.append({
            "event": event_type,
            "node_id": node_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            **payload,
        })

    def _log_research(self, record_type: str, title: str, content: str,
                      parent_id: str | None = None, status: str = "proposed",
                      tags: list[str] | None = None) -> None:
        """把假设生命周期事件写到结构化研究日志. 出错只 log debug, 不影响主流程."""
        try:
            from huginn.research_log import get_research_log
            get_research_log().add(
                record_type=record_type,
                title=title,
                content=content,
                parent_id=parent_id,
                status=status,
                tags=tags or [],
            )
        except Exception as e:
            logging.getLogger(__name__).debug(
                "research log write failed: %s", e,
            )

    # ── 节点 ─────────────────────────────────────────────────────────

    def add_hypothesis(
        self,
        statement: str,
        rationale: str = "",
        testable_prediction: str = "",
        parent_id: str | None = None,
    ) -> str:
        """新增假设节点, 返回 node id. parent_id 非空时自动加 derive 边."""
        if not statement.strip():
            raise HypothesisGraphError("假设陈述不能为空")
        # 先查 parent 再加节点, 避免失败时留下孤儿节点
        if parent_id is not None:
            self._check_node(parent_id)
        node_id = f"h_{uuid.uuid4().hex[:8]}"
        node = HypothesisNode(
            id=node_id,
            statement=statement,
            rationale=rationale,
            testable_prediction=testable_prediction,
            parent_id=parent_id,
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            dimension=_extract_dimension(statement),
        )
        self._nodes[node_id] = node
        if parent_id is not None:
            # 交叉授粉延迟: 跨分量 derive 需两端分量都成熟.
            # 新节点是 singleton, _is_cross_component_edge 排除单节点 → 正常 derive 不受影响.
            # 防的是未来直接连两个已有节点的场景.
            ok, reason = self._check_cross_pollination_readiness(parent_id, node_id)
            if not ok:
                logging.getLogger(__name__).info(
                    "derive 被交叉授粉延迟拒绝: %s -> %s, %s",
                    parent_id, node_id, reason,
                )
                del self._nodes[node_id]
                return None
            self._edges.append(HypothesisEdge(
                from_id=parent_id, to_id=node_id, edge_type="derive",
                evidence={"rationale": rationale},
            ))
        self._record_event("add", node_id, statement=statement,
                           parent_id=parent_id)
        self._log_research(
            "conjecture", node.statement[:80],
            f"{node.statement}\n\nRationale: {node.rationale}\n"
            f"Prediction: {node.testable_prediction}",
            parent_id=parent_id, status="proposed",
            tags=["autoloop", "hypothesis"],
        )
        return node_id

    def get(self, node_id: str) -> HypothesisNode:
        self._check_node(node_id)
        return self._nodes[node_id]

    def all_nodes(self) -> list[HypothesisNode]:
        return list(self._nodes.values())

    def cluster_by_dimension(self) -> dict[str, list[HypothesisNode]]:
        """按 dimension 二维分组, 返回 cluster_id -> nodes.

        cluster_id = f"{dimension}" (空 dimension 归到 "unknown").
        ponytail: 关键词命中 dimension, 非语义. 升级路径: LLM 判定 dimension (v8+).
        跟 _metacog_classify_family (engine.py) 同范式, 不引入 embedding.
        """
        clusters: dict[str, list[HypothesisNode]] = {}
        for node in self._nodes.values():
            key = node.dimension or "unknown"
            clusters.setdefault(key, []).append(node)
        return clusters

    def siblings(self, node_id: str) -> list[HypothesisNode]:
        """返回同 sibling_group_id 的兄弟节点 (不含自身).

        v11: pivot 时同一失败假设产出的多个候选共享 sibling_group_id.
        ponytail: 字段驱动, 非 LLM 判定. None 视为无兄弟.
        """
        node = self.get(node_id)
        if not node.sibling_group_id:
            return []
        return [
            n for n in self._nodes.values()
            if n.sibling_group_id == node.sibling_group_id and n.id != node_id
        ]

    def frontier(self) -> list[HypothesisNode]:
        """未测试的假设 (campaign 该排队的)."""
        return [n for n in self._nodes.values() if n.status == "untested"]

    def supported(self) -> list[HypothesisNode]:
        return [n for n in self._nodes.values() if n.status == "supported"]

    def refuted(self) -> list[HypothesisNode]:
        return [n for n in self._nodes.values() if n.status == "refuted"]

    def events(self) -> list[dict[str, Any]]:
        """返回事件日志副本 (append-only, 调用方不应修改).
        用于回放/调试: 重放事件可重建图状态."""
        return list(self._events)

    # ── 状态转移 ─────────────────────────────────────────────────────

    def support(self, node_id: str, evidence: dict[str, Any]) -> None:
        """标记假设被实验支持."""
        self._check_node(node_id)
        node = self._nodes[node_id]
        if node.status == "refuted":
            raise HypothesisGraphError(
                f"节点 {node_id} 已被反驳, 不能再标记为 supported"
            )
        node.status = "supported"
        node.evidence = {**node.evidence, **evidence}
        self._edges.append(HypothesisEdge(
            from_id=node_id, to_id=node_id, edge_type="support", evidence=evidence,
        ))
        self._record_event("support", node_id,
                           modality=evidence.get("modality"),
                           data_source=evidence.get("data_source"))
        self._log_research(
            "verification", f"验证通过: {node.statement[:60]}",
            f"假设 {node_id} 被实验支持.\n\n"
            f"Statement: {node.statement}\nEvidence: {evidence}",
            parent_id=node_id, status="verified",
            tags=["autoloop", "verified"],
        )

    def refute(self, node_id: str, evidence: dict[str, Any]) -> None:
        """标记假设被实验反驳. 不自动生成修正假设 — 调 refine_failed 才生成."""
        self._check_node(node_id)
        node = self._nodes[node_id]
        if node.status == "supported":
            raise HypothesisGraphError(
                f"节点 {node_id} 已被支持, 不能再标记为 refuted"
            )
        node.status = "refuted"
        node.evidence = {**node.evidence, **evidence}
        self._edges.append(HypothesisEdge(
            from_id=node_id, to_id=node_id, edge_type="refute", evidence=evidence,
        ))
        self._record_event("refute", node_id,
                           reason=str(evidence.get("errors", ""))[:200])
        self._log_research(
            "counterexample", f"反驳: {node.statement[:60]}",
            f"假设 {node_id} 被实验反驳.\n\n"
            f"Statement: {node.statement}\nEvidence: {evidence}",
            parent_id=node_id, status="refuted",
            tags=["autoloop", "refuted"],
        )

    def supersede(self, node_id: str) -> None:
        """标记假设被衍生假设取代 (refine_failed 后旧假设变 superseded)."""
        self._check_node(node_id)
        node = self._nodes[node_id]
        # 源状态校验: untested 节点不该被 supersede (没测过就取代无意义)
        if node.status == "untested":
            raise HypothesisGraphError(
                f"节点 {node_id} 未测试, 不能直接 supersede (应先 refute/support)"
            )
        node.status = "superseded"
        self._record_event("supersede", node_id)
        # 对称: add/support/refute/refine/pivot 都写 research_log, supersede 补上
        self._log_research(
            "counterexample", f"取代: {node.statement[:60]}",
            f"假设 {node_id} 被 refine 后的衍生假设取代.\n\n"
            f"Statement: {node.statement}\nStatus: superseded",
            parent_id=node_id, status="superseded",
            tags=["autoloop", "superseded"],
        )

    # ── 失败驱动的假设修正 ───────────────────────────────────────────

    def pivot(
        self,
        failed_node_id: str,
        evidence: dict[str, Any],
        model: Any | None = None,
        objective: str = "",
        n_best: int = 2,
    ) -> str:
        """战略转向: 放弃当前假设方向, 生成全新假设.

        v11: N-best speculative — n_best=2 并行采样, 主候选返回, 备候选进图
        标 sibling_group_id 共享. 复用 _hypothesize 的 N=2 思路, 但 pivot 是
        sync 方法, 这里用顺序调用 (LLM 调用是瓶颈, 顺序 vs 并行差异小).
        ponytail: 不 async 化 pivot (会传染到调用方), 顺序 N-best 够用.

        与 refine_failed 的区别: refine 在原假设基础上修正参数或条件,
        pivot 彻底换一个方向. 在 refine 次数耗尽时触发.

        流程:
        1. 收集所有已反驳假设的失败模式, 避免重走老路
        2. 调 LLM 生成 N 个候选 (主温度 + 高温), 主候选返回, 备候选进图
           - 不可用时回退到模板: 换一个变量维度
        3. 新假设不继承 parent (不是 derive 关系, 是 pivot 关系)
        4. v11: N 个候选共享 sibling_group_id, siblings() 可查
        """
        self._check_node(failed_node_id)
        failed_node = self._nodes[failed_node_id]

        # 收集所有 refuted 节点的 statement, 让 LLM 知道哪些路走不通
        failed_statements = [
            n.statement for n in self._nodes.values()
            if n.status in ("refuted", "superseded")
        ]

        # v11: N-best 采样 — 主候选 (默认温度) + 备候选 (高温=1.0)
        # ponytail: 顺序调用, 不 async. LLM invoke 是同步阻塞, 并行需 asyncio.
        # 升级路径: async pivot + asyncio.gather (v12 候选, 要改调用方).
        main_statement: str = ""
        backup_statements: list[str] = []
        if model is not None and self._is_real_model(model):
            main_statement = self._llm_pivot(
                failed_node.statement, failed_statements, evidence, objective, model,
            )
            if n_best >= 2:
                try:
                    hot_model = model.bind(temperature=1.0)
                    _backup = self._llm_pivot(
                        failed_node.statement, failed_statements, evidence, objective, hot_model,
                    )
                    if _backup and _backup != main_statement:
                        backup_statements.append(_backup)
                except Exception:
                    pass  # not all model wrappers support bind
        else:
            main_statement = self._template_pivot(
                failed_node.statement, failed_statements,
            )

        # v11: 生成 sibling_group_id, N 个候选共享
        _sibling_group = f"sg_{uuid.uuid4().hex[:8]}" if backup_statements else None

        # 加节点 + pivot 边 (不是 derive, 是独立的 pivot 关系)
        new_id = self.add_hypothesis(
            statement=main_statement,
            rationale=f"战略转向: refine 次数耗尽, 放弃 {failed_node_id} 方向",
        )
        # v11: 主候选标 sibling_group_id
        if _sibling_group:
            self._nodes[new_id].sibling_group_id = _sibling_group

        # v11: 备候选进图, 标 sibling_group_id + evidence={"candidate_role":"backup"}
        for _backup_stmt in backup_statements:
            try:
                _backup_id = self.add_hypothesis(
                    statement=_backup_stmt,
                    rationale=f"backup pivot candidate (sibling of {new_id})",
                )
                if _backup_id:
                    self._nodes[_backup_id].sibling_group_id = _sibling_group
                    self._nodes[_backup_id].evidence = {
                        **self._nodes[_backup_id].evidence,
                        "candidate_role": "backup",
                    }
            except Exception:
                pass  # backup 失败不阻塞主候选

        # 交叉授粉延迟: pivot 跨分量需两端分量都成熟.
        # new_id 是 singleton, _is_cross_component_edge 排除单节点 → 正常 pivot 不受影响.
        ok, reason = self._check_cross_pollination_readiness(
            failed_node_id, new_id,
        )
        if not ok:
            logging.getLogger(__name__).info(
                "pivot 被交叉授粉延迟拒绝: %s -> %s, %s",
                failed_node_id, new_id, reason,
            )
            return None
        # 把 pivot 关系记到边里 — 用 "pivot" 类型, 不污染 children() 的 derive 过滤
        self._edges.append(HypothesisEdge(
            from_id=failed_node_id, to_id=new_id, edge_type="pivot",
            evidence={"reason": "max_refines_reached"},
        ))
        self._record_event(
            "pivot", new_id,
            from_node=failed_node_id,
            failed_count=len(failed_statements),
        )
        self._log_research(
            "conjecture", f"PIVOT: {main_statement[:60]}",
            f"战略转向 — refine 次数耗尽后放弃原方向.\n\n"
            f"新假设: {main_statement}\n"
            f"放弃方向: {failed_node.statement}\n"
            f"已尝试的失败假设数: {len(failed_statements)}\n"
            f"backup 候选数: {len(backup_statements)}",
            parent_id=new_id, status="proposed",
            tags=["autoloop", "pivot"],
        )

        # v12: AlphaEvolve crossover — sibling_group 有主+备时, 尝试组合产生第 3 候选.
        # ponytail: 复用 model.invoke, 失败 non-fatal. child 通过 derive 边连 parent_a.
        # 升级路径: mutation (单候选局部扰动) — YAGNI, crossover 已够闭环, benchmark 驱动再加.
        if _sibling_group and backup_statements and model is not None and self._is_real_model(model):
            try:
                _backup_id = next(
                    (n.id for n in self._nodes.values()
                     if n.sibling_group_id == _sibling_group
                     and n.evidence.get("candidate_role") == "backup"),
                    None,
                )
                if _backup_id:
                    _child_id = self.crossover(
                        new_id, _backup_id, model, objective,
                    )
                    if _child_id:
                        self._nodes[_child_id].sibling_group_id = _sibling_group
            except Exception:
                logger.debug("v12 crossover after pivot failed (non-fatal)", exc_info=True)

        return new_id

    def crossover(
        self,
        parent_a_id: str,
        parent_b_id: str,
        model: Any,
        objective: str = "",
    ) -> str | None:
        """v12: AlphaEvolve crossover — 组合两个 parent 假设的优势产生 child.

        sibling_group 里的两个候选 (主+备) 组合产生第 3 候选, 让 frontier 选优.
        child 标 evidence={"crossover_parents": [a, b], "candidate_role": "crossover"},
        通过 derive 边连 parent_a (crossover 是组合不是转向, derive 语义合适).

        ponytail: 复用 _llm_pivot 的 model.invoke 模式, 不新建 EvolutionEngine.
        失败 (无 model / LLM 异常 / child 同质) 返回 None, non-fatal.
        """
        if not self._is_real_model(model):
            return None
        self._check_node(parent_a_id)
        self._check_node(parent_b_id)
        a = self._nodes[parent_a_id]
        b = self._nodes[parent_b_id]

        prompt = (
            f"研究目标: {objective}\n\n"
            f"假设 A: {a.statement}\n  理由: {a.rationale}\n\n"
            f"假设 B: {b.statement}\n  理由: {b.rationale}\n\n"
            "组合 A 和 B 的优势, 生成一个新假设 — 继承 A 在某方面的优点和 "
            "B 在另一方面的优点. 必须可测试且新颖 (非简单合并).\n"
            "用一句话陈述新假设:"
        )
        try:
            resp = model.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            stmt = text.strip().split("\n")[0].strip()
            stmt = stmt.lstrip("- *•").strip().strip('"\'')
        except Exception:
            return None

        if not stmt or stmt == a.statement or stmt == b.statement:
            return None

        child_id = self.add_hypothesis(
            statement=stmt,
            rationale=f"crossover of {parent_a_id} + {parent_b_id}",
            parent_id=parent_a_id,  # derive 边
        )
        if child_id:
            self._nodes[child_id].evidence = {
                **self._nodes[child_id].evidence,
                "crossover_parents": [parent_a_id, parent_b_id],
                "candidate_role": "crossover",
            }
            self._record_event(
                "crossover", child_id,
                parents=[parent_a_id, parent_b_id],
            )
            self._log_research(
                "conjecture", f"CROSSOVER: {stmt[:60]}",
                f"AlphaEvolve crossover — 组合两个 parent 优势.\n\n"
                f"child: {stmt}\n"
                f"parent_a: {a.statement}\n"
                f"parent_b: {b.statement}",
                parent_id=child_id, status="proposed",
                tags=["autoloop", "crossover"],
            )
        return child_id

    def _llm_pivot(
        self,
        failed_statement: str,
        failed_statements: list[str],
        evidence: dict[str, Any],
        objective: str,
        model: Any,
    ) -> str:
        """调 LLM 生成一个全新方向的假设."""
        failed_list = "\n".join(f"  - {s}" for s in failed_statements[:8])
        prompt = (
            f"研究目标: {objective}\n\n"
            f"已尝试但失败的假设:\n{failed_list}\n\n"
            f"最新失败: {failed_statement}\n"
            f"失败证据: {evidence}\n\n"
            "以上方向都不行. 请提出一个完全不同的假设方向 — "
            "不是修正参数, 而是换一个变量维度或方法论.\n"
            "用一句话陈述新假设:"
        )
        try:
            resp = model.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            # 取第一行, 去掉引号和前缀
            line = text.strip().split("\n")[0].strip()
            return line.lstrip("- *•").strip().strip('"\'')
        except Exception:
            return self._template_pivot(failed_statement, failed_statements)

    @staticmethod
    def _template_pivot(
        failed_statement: str,
        failed_statements: list[str],
    ) -> str:
        """无 LLM 时的降级: 换一个变量维度."""
        # 简单启发: 把原假设的关键词反一下
        return (
            f"与之前方向不同: 在排除了 {len(failed_statements)} 个假设后, "
            "考虑从另一个物理量或方法角度重新切入问题"
        )

    def refine_failed(
        self,
        node_id: str,
        evidence: dict[str, Any],
        model: Any | None = None,
        block_registry: Any | None = None,
        method_family: str | None = None,
        proposed_mechanism_type: str | None = None,
    ) -> str:
        """对失败的假设生成修正假设, 返回新 node id.

        流程:
        1. 调 RedTeamReviewer 审查原假设 + 失败证据, 拿 findings
        2. 基于 findings 生成修正假设陈述
           - model 可用时调 LLM 生成
           - 不可用时用 findings 的 mitigation 做模板拼接
        3. (可选) 阻塞-重启协议: 若 block_registry + method_family 给定,
           且该族有阻塞路线, 把新假设作为重启提议过 try_reopen.
           拒绝 (still_blocked / equivalent_to_previous) 时不加新节点,
           返回原 node_id, 调用方据此知道 refine 被阻塞.
        4. 新假设 parent_id = 失败节点, 旧节点标 superseded
        5. 返回新 node id, 调用方 (CampaignManager) 把它进队列
        """
        self._check_node(node_id)
        node = self._nodes[node_id]
        if node.status != "refuted":
            raise HypothesisGraphError(
                f"节点 {node_id} 状态为 {node.status}, 只有 refuted 才能 refine"
            )

        # 1. red-team 审查
        from huginn.autoloop.red_team import RedTeamReviewer

        reviewer = RedTeamReviewer(model=model)
        report = reviewer.review(
            "hypothesize", "plan",
            {"hypothesis": node.statement, "evidence": evidence},
        )
        findings = [f.to_dict() for f in report.findings]

        # 把 red-team 发现的障碍记到研究日志
        obstacle_summary = "; ".join(
            f.get("description", f.get("severity", "unknown"))
            for f in findings
        ) if findings else "无具体发现"
        self._log_research(
            "obstacle", f"障碍识别: {node.statement[:50]}",
            f"假设 {node_id} 在修正时识别到障碍:\n\n"
            f"原假设: {node.statement}\n"
            f"失败证据: {evidence}\n"
            f"Red-team 发现: {obstacle_summary}",
            parent_id=node_id, status="in_progress",
            tags=["autoloop", "refine", "red-team"],
        )

        # 2. 生成修正假设
        if model is not None and self._is_real_model(model):
            new_statement = self._llm_refine(node.statement, findings, evidence, model)
        else:
            new_statement = self._template_refine(node.statement, findings)

        # 3. 阻塞-重启协议: 检查该族是否有阻塞路线
        # 新假设作为 "proposed mechanism" 过 try_reopen, 防止换名重启死路线.
        # 拒绝时返回原 node_id, 不加新节点, 不 supersede 旧节点.
        if block_registry is not None and method_family is not None:
            blocked_routes = block_registry.list_blocked(family_id=method_family)
            if blocked_routes:
                # 取最近一条阻塞路线尝试重启
                route = blocked_routes[0]
                mech_type = proposed_mechanism_type or "new_construction"  # type: ignore[arg-type]
                attempt = block_registry.try_reopen(
                    route_id=route.route_id,
                    proposed_mechanism=new_statement,
                    proposer_agent=f"refine:{node_id}",
                    mechanism_type=mech_type,  # type: ignore[arg-type]
                )
                if attempt.verdict != "reopen":
                    # 重启被拒: 换名归约或机制类型不匹配, 不加新节点
                    self._log_research(
                        "obstacle", f"阻塞路线拒绝重启: {route.route_id}",
                        f"假设 {node_id} 的修正被阻塞路线 {route.route_id} 拒绝.\n\n"
                        f"提议机制: {new_statement}\n"
                        f"拒绝原因: {attempt.verdict}\n"
                        f"阻塞原因: {route.block_reason}\n"
                        f"需要: {route.required_mechanism_description}",
                        parent_id=node_id, status="blocked",
                        tags=["autoloop", "refine", "block_registry"],
                    )
                    return node_id  # 调用方据此知道 refine 被阻塞

        # 4. 加节点 + 标旧节点 superseded
        new_id = self.add_hypothesis(
            statement=new_statement,
            rationale=f"修正自 {node_id}: {node.statement}",
            testable_prediction=node.testable_prediction,
            parent_id=node_id,
        )
        self._nodes[new_id].refinement_basis = findings
        self._record_event(
            "refine", new_id,
            from_node=node_id,
            findings_count=len(findings),
        )
        self.supersede(node_id)
        self._log_research(
            "proof_attempt", f"修正假设: {new_statement[:60]}",
            f"基于障碍分析修正假设.\n\n"
            f"新假设: {new_statement}\n"
            f"原假设: {node.statement}\n"
            f"修正依据: {obstacle_summary}",
            parent_id=new_id, status="proposed",
            tags=["autoloop", "refine"],
        )
        return new_id

    # ── 双覆盖查询 ───────────────────────────────────────────────────

    def needs_dual_coverage(self, node_id: str) -> bool:
        """节点是否需要双模态覆盖 (割边判定).

        升级自启发式 → networkx.articulation_points 精确判定.
        割边 (articulation point): 删除后图分量数增加的节点.
        在假设图上, 割点是关键路径枢纽 — 若它被幻觉/压缩, 下游全断.
        """
        self._check_node(node_id)
        articulation = self._articulation_points()
        return node_id in articulation

    def _articulation_points(self) -> set[str]:
        """计算当前图的所有割点 (articulation points).

        ponytail: O(V+E) Tarjan 算法 (networkx 实现). 节点 <3 时
        直接返回空集 (无割点可能). derive/support/refute 边都计入无向图.
        """
        if len(self._nodes) < 3:
            return set()
        try:
            import networkx as nx

            g = nx.Graph()
            g.add_nodes_from(self._nodes.keys())
            for e in self._edges:
                # 自环 (support/refute 边 from==to) 不影响割点判定, 跳过
                if e.from_id != e.to_id:
                    g.add_edge(e.from_id, e.to_id)
            return set(nx.articulation_points(g))
        except Exception:
            # networkx 不可用时降级到启发式
            return {
                e.from_id for e in self._edges
                if e.edge_type == "derive"
                and any(e2.from_id == e.to_id for e2 in self._edges
                        if e2.edge_type == "derive")
            }

    def dual_covered(self, node_id: str) -> bool:
        """节点是否被 ≥2 种独立模态支撑 (via support 边的 modality 字段).

        独立性判定:
        - modality 必须不同 (deductive ≠ numeric)
        - 若 support 边带 data_source 字段, 则 data_source 也必须不同
          (防 IPI: 两条边若来自同一被污染数据源, 是假双覆盖)

        命中时自动注册 2-单纯形 {node, ev_key_1, ev_key_2}, 把"N 条证据
        作为整体支撑"的语义显式化. 之前靠组合判定隐式表达, 下游无法区分
        "碰巧有 2 条 support 边"和"2 条证据形成独立支撑整体".

        ponytail: 'deductive' 与 'numeric' 是软独立 — GP 数值验证与符号
        推导基底不同, 但仍是同模型权重. 真独立需跨模型/跨模态, 等幻觉
        断裂数据再升级. data_source 检查是 IPI 防御的硬约束."""
        self._check_node(node_id)
        support_edges = [
            e for e in self._edges
            if e.from_id == node_id
            and e.to_id == node_id
            and e.edge_type == "support"
            and e.evidence.get("modality")
        ]
        if len(support_edges) < 2:
            return False

        modalities = {e.evidence.get("modality") for e in support_edges}
        if len(modalities) < 2:
            return False

        # 若任一 support 边带 data_source, 检查来源独立性
        sources = {
            e.evidence.get("data_source")
            for e in support_edges
            if e.evidence.get("data_source")
        }
        # 有 data_source 标签时, 必须有 ≥2 个不同来源
        # 没有 data_source 标签时, 退回到只检查 modality (向后兼容)
        if sources and len(sources) < 2:
            return False

        # 注册 2-单纯形: 取前两条独立 support 边的 modality 作为 ev_key
        # ponytail: 只存 frozenset, 不存边的完整引用 — 避免边删除后悬空指针
        ev_keys = [f"ev:{node_id}:{e.evidence['modality']}" for e in support_edges[:2]]
        self._simplicials.add(frozenset({node_id, *ev_keys}))
        return True

    def simplicial_faces(self, node_id: str) -> list[frozenset[str]]:
        """返回包含该节点的所有 2-单纯形 (作为整体支撑关系的显式记录)."""
        return [s for s in self._simplicials if node_id in s]

    # ── 连通分量监控 ───────────────────────────────────────────────────

    def connected_components(self) -> list[set[str]]:
        """弱连通分量列表, 按 size 降序.

        support/refute/derive/pivot 都算连接. 自环边 (support/refute 的
        from==to) 不影响连通性, 跳过.
        """
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(self._nodes.keys())
        for e in self._edges:
            if e.from_id != e.to_id:
                g.add_edge(e.from_id, e.to_id)
        return sorted(nx.connected_components(g), key=len, reverse=True)

    def component_count(self) -> int:
        """连通分量数."""
        return len(self.connected_components())

    def component_maturity(self, component_node_ids: set[str]) -> dict[str, Any]:
        """分量的成熟度指标.

        size:           节点数
        depth:          最长派生链长度 (沿 parent_id 链的最长路径)
        has_refuted:    是否含 refuted 节点 (暴露缺口)
        has_blocked_route: 是否关联 block_registry 阻塞路线 (无入参时 False)
        audited:        是否所有节点都过等价性审计 (evidence 里有 equivalence_verdict)
        maturity_score: 0.0-1.0, has_refuted +0.3, audited +0.3, depth>=2 +0.4
        """
        nodes = [
            self._nodes[nid] for nid in component_node_ids
            if nid in self._nodes
        ]
        # 最长派生链: 在分量内沿 parent_id 走的最长路径
        depth = 0
        for n in nodes:
            chain_len = 0
            cur: HypothesisNode | None = n
            seen: set[str] = set()
            while cur is not None and cur.id not in seen:
                seen.add(cur.id)
                chain_len += 1
                if cur.parent_id and cur.parent_id in component_node_ids:
                    cur = self._nodes.get(cur.parent_id)
                else:
                    cur = None
            if chain_len > depth:
                depth = chain_len

        has_refuted = any(n.status == "refuted" for n in nodes)
        # 无 block_registry 入参, 默认 False
        has_blocked_route = False
        audited = bool(nodes) and all(
            "equivalence_verdict" in n.evidence for n in nodes
        )
        score = 0.0
        if has_refuted:
            score += 0.3
        if audited:
            score += 0.3
        if depth >= 2:
            score += 0.4
        return {
            "size": len(nodes),
            "depth": depth,
            "has_refuted": has_refuted,
            "has_blocked_route": has_blocked_route,
            "audited": audited,
            "maturity_score": score,
        }

    def component_representative(self, component_node_ids: set[str]) -> str | None:
        """选分量代表节点, 防单分量靠节点数主导.

        优先级: supported > untested > 其他, 同级取 created_at 最新的.
        """
        nodes = [
            self._nodes[nid] for nid in component_node_ids
            if nid in self._nodes
        ]
        if not nodes:
            return None
        supported = [n for n in nodes if n.status == "supported"]
        if supported:
            return max(supported, key=lambda n: n.created_at).id
        untested = [n for n in nodes if n.status == "untested"]
        if untested:
            return max(untested, key=lambda n: n.created_at).id
        return max(nodes, key=lambda n: n.created_at).id

    def is_collapsed(self, min_components: int = 2) -> bool:
        """拓扑坍缩: 连通分量数 < min_components."""
        return self.component_count() < min_components

    def _is_cross_component_edge(self, from_id: str, to_id: str) -> bool:
        """检查 from 和 to 是否在不同连通分量.

        ponytail: 排除单节点分量 — singleton 要么是刚加的新节点 (接枝到已有线),
        要么是还没发展的孤立假设. 交叉授粉指两条已发展的线 (size>=2) 之间搭桥.
        不排除的话 add_hypothesis/pivot 每次加新节点都会被误判 (新节点必然是
        singleton, 永远不成熟), 整个 derive/pivot 流程就废了.
        """
        components = self.connected_components()
        from_comp = None
        to_comp = None
        for comp in components:
            if from_id in comp:
                from_comp = comp
            if to_id in comp:
                to_comp = comp
        if from_comp is None or to_comp is None:
            return False
        if len(from_comp) == 1 or len(to_comp) == 1:
            return False
        return from_comp is not to_comp

    def _check_cross_pollination_readiness(
        self, from_id: str, to_id: str,
    ) -> tuple[bool, str]:
        """检查跨分量边是否允许 (交叉授粉延迟).

        两分量都成熟 (maturity_score >= 阈值) 才允许.
        对应 prompt: "仅在独立 agent 已将其发展到足以暴露其真正优势和缺口
        后才进行交叉授粉".
        """
        if not self._is_cross_component_edge(from_id, to_id):
            return True, ""  # 同分量或单节点, 放行

        components = self.connected_components()
        for comp in components:
            if from_id in comp or to_id in comp:
                maturity = self.component_maturity(comp)
                if maturity["maturity_score"] < self._CROSS_POLLINATION_MATURITY_THRESHOLD:
                    return False, (
                        f"交叉授粉延迟: 分量 (size={maturity['size']}) "
                        f"成熟度 {maturity['maturity_score']:.2f} < 阈值 "
                        f"{self._CROSS_POLLINATION_MATURITY_THRESHOLD}, "
                        f"暂不允许跨分量边"
                    )
        return True, ""

    # ── 边查询 ───────────────────────────────────────────────────────

    def edges(self) -> list[HypothesisEdge]:
        return list(self._edges)

    def children(self, node_id: str) -> list[HypothesisNode]:
        """直接衍生子节点."""
        self._check_node(node_id)
        child_ids = {
            e.to_id for e in self._edges
            if e.from_id == node_id and e.edge_type == "derive"
        }
        return [self._nodes[c] for c in child_ids if c in self._nodes]

    def derivation_chain(self, node_id: str) -> list[HypothesisNode]:
        """从根到指定节点的衍生链."""
        self._check_node(node_id)
        chain: list[HypothesisNode] = []
        current: str | None = node_id
        seen: set[str] = set()
        while current is not None and current not in seen:
            seen.add(current)
            chain.append(self._nodes[current])
            current = self._nodes[current].parent_id
        chain.reverse()
        return chain

    # ── 序列化 ───────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
            # 2-单纯形: list[list[str]] (frozenset 不可 JSON 序列化)
            "simplicials": [sorted(s) for s in self._simplicials],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HypothesisGraph":
        graph = cls()
        for nd in d.get("nodes", []):
            node = HypothesisNode.from_dict(nd)
            graph._nodes[node.id] = node
        for ed in d.get("edges", []):
            graph._edges.append(HypothesisEdge(
                from_id=ed["from_id"],
                to_id=ed["to_id"],
                edge_type=ed["edge_type"],
                evidence=ed.get("evidence", {}),
            ))
        for s in d.get("simplicials", []):
            graph._simplicials.add(frozenset(s))
        return graph

    # ── 内部 ─────────────────────────────────────────────────────────

    def _check_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            raise HypothesisGraphError(f"节点 {node_id} 不存在")

    @staticmethod
    def _is_real_model(model: Any) -> bool:
        return not hasattr(model, "_mock_name")

    def _template_refine(
        self, original: str, findings: list[dict[str, Any]]
    ) -> str:
        """无 LLM 时用 findings 的 mitigation 做模板拼接."""
        if not findings:
            return f"修正假设: {original} (考虑未覆盖的边界条件后重新表述)"
        mitigations = [f["mitigation"] for f in findings if f.get("mitigation")]
        if not mitigations:
            return f"修正假设: {original} (根据失败证据调整预期关系)"
        basis = "; ".join(mitigations[:3])
        return f"修正假设: {original} — 已纳入修正: {basis}"

    def _llm_refine(
        self,
        original: str,
        findings: list[dict[str, Any]],
        evidence: dict[str, Any],
        model: Any,
    ) -> str:
        """调 LLM 生成修正假设. 失败时降级到模板."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            findings_text = "\n".join(
                f"- [{f.get('severity', '?')}] {f.get('description', '')}"
                f" → {f.get('mitigation', '')}"
                for f in findings
            )
            evidence_text = str(evidence)[:500]
            messages = [
                SystemMessage(content=(
                    "You are a research hypothesis refiner. Given a refuted "
                    "hypothesis, the red-team findings, and the experimental "
                    "evidence, produce ONE revised hypothesis statement that "
                    "addresses the identified weaknesses. Output only the "
                    "statement, no preamble."
                )),
                HumanMessage(content=(
                    f"原假设: {original}\n"
                    f"Red-team 发现:\n{findings_text}\n"
                    f"实验证据: {evidence_text}\n"
                    f"修正假设:"
                )),
            ]
            import asyncio

            try:
                asyncio.get_running_loop()
                # 已有 loop 时不能 asyncio.run, 同步拿不了
                resp = model.invoke(messages)
            except RuntimeError:
                resp = asyncio.run(model.ainvoke(messages))
            return str(resp.content).strip()
        except Exception:
            return self._template_refine(original, findings)


def _selfcheck_connected_components() -> None:
    """连通分量监控自检 — 构造 2 个不相交分量验证基本逻辑."""
    g = HypothesisGraph()
    # 分量 1: 3 节点 derive 链
    h1 = g.add_hypothesis("h1 statement")
    h2 = g.add_hypothesis("h2 statement", parent_id=h1)
    h3 = g.add_hypothesis("h3 statement", parent_id=h2)
    # 分量 2: 2 节点 derive 链
    h4 = g.add_hypothesis("h4 statement")
    h5 = g.add_hypothesis("h5 statement", parent_id=h4)

    # 给不同的 created_at, 让 "最新" 优先级可判定
    g.get(h1).created_at = "2024-01-01T00:00:01Z"
    g.get(h2).created_at = "2024-01-01T00:00:02Z"
    g.get(h3).created_at = "2024-01-01T00:00:03Z"
    g.get(h4).created_at = "2024-01-01T00:00:04Z"
    g.get(h5).created_at = "2024-01-01T00:00:05Z"

    # 分量数
    assert g.component_count() == 2, f"expected 2, got {g.component_count()}"

    comps = g.connected_components()
    assert len(comps) == 2
    assert len(comps[0]) >= len(comps[1])  # 降序
    assert len(comps[0]) == 3, f"largest should have 3 nodes, got {len(comps[0])}"

    # 成熟度指标字段
    m = g.component_maturity(comps[0])
    expected_keys = {
        "size", "depth", "has_refuted", "has_blocked_route",
        "audited", "maturity_score",
    }
    assert set(m.keys()) == expected_keys, f"keys mismatch: {set(m.keys())}"
    assert m["size"] == 3
    assert m["depth"] == 3, f"expected depth 3 (h1->h2->h3), got {m['depth']}"
    assert m["has_refuted"] is False
    assert m["has_blocked_route"] is False
    assert m["audited"] is False  # evidence 里没有 equivalence_verdict
    # depth>=2 -> +0.4, 其余 False
    assert abs(m["maturity_score"] - 0.4) < 1e-9

    # 坍缩: 2 < 3 -> True, 2 < 2 -> False
    assert g.is_collapsed(min_components=3) is True
    assert g.is_collapsed(min_components=2) is False

    # 代表: 全 untested -> 最新 (h3)
    rep = g.component_representative(comps[0])
    assert rep is not None and rep in comps[0]
    assert rep == h3, f"expected newest untested h3, got {rep}"

    # 代表: 标 h2 为 supported 后应优先选 h2
    g.get(h2).status = "supported"
    rep2 = g.component_representative(comps[0])
    assert rep2 == h2, f"expected supported h2, got {rep2}"

    # 空集
    assert g.component_representative(set()) is None

    # ── 交叉授粉延迟 ─────────────────────────────────────────────
    # 构造 2 个不成熟的多节点分量, 验证跨分量边被拒绝; 成熟后才放行
    g2 = HypothesisGraph()
    # 分量 A: a1 -> a2 (depth=2, 无 refute 无 audit → score=0.4 < 0.6)
    a1 = g2.add_hypothesis("a1")
    a2 = g2.add_hypothesis("a2", parent_id=a1)
    # 分量 B: b1 -> b2 (同样不成熟)
    b1 = g2.add_hypothesis("b1")
    b2 = g2.add_hypothesis("b2", parent_id=b1)

    # 两个多节点 + 不同分量 → 是跨分量边
    assert g2._is_cross_component_edge(a1, b1) is True
    # 同分量不算跨分量
    assert g2._is_cross_component_edge(a1, a2) is False
    # singleton 不算跨分量 (新节点 / 孤立假设)
    g3 = HypothesisGraph()
    s1 = g3.add_hypothesis("s1")
    s2 = g3.add_hypothesis("s2")
    assert g3._is_cross_component_edge(s1, s2) is False

    # 两端都不成熟 → 拒绝
    ok, reason = g2._check_cross_pollination_readiness(a1, b1)
    assert ok is False, f"不成熟分量应拒绝, got ok={ok}"
    assert "交叉授粉延迟" in reason

    # 让 A 成熟 (refute a2 → has_refuted +0.3 → score=0.7 >= 0.6)
    g2.refute(a2, evidence={"r": "test"})
    # B 仍不成熟 → 仍拒绝
    ok2, reason2 = g2._check_cross_pollination_readiness(a1, b1)
    assert ok2 is False, f"B 不成熟应拒绝, got ok={ok2}"
    assert "交叉授粉延迟" in reason2

    # 让 B 也成熟
    g2.refute(b2, evidence={"r": "test"})
    ok3, reason3 = g2._check_cross_pollination_readiness(a1, b1)
    assert ok3 is True, f"两边都成熟应放行, got ok={ok3}"
    assert reason3 == ""

    print("OK: connected_components selfcheck passed")


if __name__ == "__main__":
    _selfcheck_connected_components()
