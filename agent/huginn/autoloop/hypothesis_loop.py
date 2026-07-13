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
EdgeType = Literal["support", "refute", "derive"]


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

    def __init__(self) -> None:
        self._nodes: dict[str, HypothesisNode] = {}
        self._edges: list[HypothesisEdge] = []

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
        )
        self._nodes[node_id] = node
        if parent_id is not None:
            self._edges.append(HypothesisEdge(
                from_id=parent_id, to_id=node_id, edge_type="derive",
            ))
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

    def frontier(self) -> list[HypothesisNode]:
        """未测试的假设 (campaign 该排队的)."""
        return [n for n in self._nodes.values() if n.status == "untested"]

    def supported(self) -> list[HypothesisNode]:
        return [n for n in self._nodes.values() if n.status == "supported"]

    def refuted(self) -> list[HypothesisNode]:
        return [n for n in self._nodes.values() if n.status == "refuted"]

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
        self._nodes[node_id].status = "superseded"

    # ── 失败驱动的假设修正 ───────────────────────────────────────────

    def pivot(
        self,
        failed_node_id: str,
        evidence: dict[str, Any],
        model: Any | None = None,
        objective: str = "",
    ) -> str:
        """战略转向: 放弃当前假设方向, 生成全新假设.

        与 refine_failed 的区别: refine 在原假设基础上修正参数或条件,
        pivot 彻底换一个方向. 在 refine 次数耗尽时触发.

        流程:
        1. 收集所有已反驳假设的失败模式, 避免重走老路
        2. 调 LLM 生成一个与失败方向不同的新假设
           - 不可用时回退到模板: 换一个变量维度
        3. 新假设不继承 parent (不是 derive 关系, 是 pivot 关系)
        """
        self._check_node(failed_node_id)
        failed_node = self._nodes[failed_node_id]

        # 收集所有 refuted 节点的 statement, 让 LLM 知道哪些路走不通
        failed_statements = [
            n.statement for n in self._nodes.values()
            if n.status in ("refuted", "superseded")
        ]

        # 调 LLM 生成战略转向
        if model is not None and self._is_real_model(model):
            new_statement = self._llm_pivot(
                failed_node.statement, failed_statements, evidence, objective, model,
            )
        else:
            new_statement = self._template_pivot(
                failed_node.statement, failed_statements,
            )

        # 加节点 + pivot 边 (不是 derive, 是独立的 pivot 关系)
        new_id = self.add_hypothesis(
            statement=new_statement,
            rationale=f"战略转向: refine 次数耗尽, 放弃 {failed_node_id} 方向",
        )
        # 把 pivot 关系记到边里
        self._edges.append(HypothesisEdge(
            from_id=failed_node_id, to_id=new_id, edge_type="derive",
            evidence={"pivot": True, "reason": "max_refines_reached"},
        ))
        self._log_research(
            "conjecture", f"PIVOT: {new_statement[:60]}",
            f"战略转向 — refine 次数耗尽后放弃原方向.\n\n"
            f"新假设: {new_statement}\n"
            f"放弃方向: {failed_node.statement}\n"
            f"已尝试的失败假设数: {len(failed_statements)}",
            parent_id=new_id, status="proposed",
            tags=["autoloop", "pivot"],
        )
        return new_id

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
    ) -> str:
        """对失败的假设生成修正假设, 返回新 node id.

        流程:
        1. 调 RedTeamReviewer 审查原假设 + 失败证据, 拿 findings
        2. 基于 findings 生成修正假设陈述
           - model 可用时调 LLM 生成
           - 不可用时用 findings 的 mitigation 做模板拼接
        3. 新假设 parent_id = 失败节点, 旧节点标 superseded
        4. 返回新 node id, 调用方 (CampaignManager) 把它进队列
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

        # 3. 加节点 + 标旧节点 superseded
        new_id = self.add_hypothesis(
            statement=new_statement,
            rationale=f"修正自 {node_id}: {node.statement}",
            testable_prediction=node.testable_prediction,
            parent_id=node_id,
        )
        self._nodes[new_id].refinement_basis = findings
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
        """节点是否需要双模态覆盖.
        启发式: 是路径分叉点 (有 derive 子节点) 或处于 derivation_chain 深处.
        ponytail: 启发式判定, 复杂图下可能漏检割边. 升级:
        networkx.articulation_points, 当节点 >50 时换."""
        self._check_node(node_id)
        has_derive_children = any(
            e.from_id == node_id and e.edge_type == "derive"
            for e in self._edges
        )
        chain = self.derivation_chain(node_id)
        return has_derive_children or len(chain) >= 2

    def dual_covered(self, node_id: str) -> bool:
        """节点是否被 ≥2 种独立模态支撑 (via support 边的 modality 字段).
        ponytail: 'deductive' 与 'numeric' 是软独立 — GP 数值验证与符号
        推导基底不同, 但仍是同模型权重. 真独立需跨模型/跨模态, 等幻觉
        断裂数据再升级."""
        self._check_node(node_id)
        modalities = {
            e.evidence.get("modality")
            for e in self._edges
            if e.from_id == node_id
            and e.to_id == node_id
            and e.edge_type == "support"
            and e.evidence.get("modality")
        }
        return len(modalities) >= 2

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
