"""可解释性统一入口 — 把分散的观测栈拼装成端到端解释 (第 3 项改进).

Huginn 的审计/溯源/事件系统各自独立: audit 记录"谁做了什么", provenance 记录
"产出了什么文件/关键值", event bus 记录领域事件. 本模块是它们之上的 facade,
提供一个统一的 ``explain(goal)`` 接口, 把零散观测拼合成一条可读的时间线 +
依赖 DAG + 关键发现, 回答"这个目标是怎么一步步达到的".

设计:
- **不修改任何后端** (audit/provenance/events 保持原样), 只在上层做归一化拼装.
- 通过 Protocol provider 注入, 默认接受 AuditLogger + ProvenanceRegistry,
  便于测试用 fake provider 验证整合逻辑.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class ExplainStep:
    """时间线上的一个归一化观测点 (审计事件或产物记录)."""

    ts: float
    kind: str  # "audit" | "artifact"
    actor: str = ""
    action: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    produced_by: str = ""
    file_path: str = ""


@dataclass
class Explanation:
    """一次 explain() 的结果: 时间线 + 依赖 DAG + 关键发现."""

    goal: str
    steps: list[ExplainStep] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    key_findings: dict[str, Any] = field(default_factory=dict)

    def timeline(self) -> list[ExplainStep]:
        """按时间升序返回观测点."""
        return sorted(self.steps, key=lambda s: s.ts)


class AuditProvider(Protocol):
    """审计日志查询面 (AuditLogger 满足)."""

    def query(
        self,
        *,
        event_type: str | None = None,
        actor: str | None = None,
        action: str | None = None,
        session_id: str | None = None,
        tool: str | None = None,
        since: float | str | None = None,
        until: float | str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]: ...


class ProvenanceProvider(Protocol):
    """溯源注册表查询面 (ProvenanceRegistry 满足)."""

    def search(self, query_str: str) -> list[dict[str, Any]]: ...

    def recent(self, n: int = 10) -> list[Any]: ...


class Explainer:
    """统一解释入口: 整合 audit + provenance 观测, 回答"目标如何达成"."""

    def __init__(
        self,
        audit: AuditProvider | None = None,
        provenance: ProvenanceProvider | None = None,
    ) -> None:
        self._audit = audit
        self._provenance = provenance

    # ── 主入口 ───────────────────────────────────────────────────
    def explain(
        self,
        goal: str | None = None,
        *,
        actor: str | None = None,
        tool: str | None = None,
        limit: int = 200,
    ) -> Explanation:
        """针对一个目标/关键词拼装端到端解释.

        ``goal`` 同时作为审计 action 子串与溯源搜索词; ``tool``/``actor`` 可
        进一步收窄. 汇总各后端的观测, 得到时间线、产出、依赖边与关键发现.
        """
        queries = goal or ""
        steps: list[ExplainStep] = []
        artifacts: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        key_findings: dict[str, Any] = {}

        # 1) 审计时间线: 谁做了什么
        if self._audit is not None:
            for ev in self._audit.query(
                action=queries or None,
                actor=actor,
                tool=tool,
                limit=limit,
            ):
                steps.append(
                    ExplainStep(
                        ts=_to_ts(ev.get("timestamp")),
                        kind="audit",
                        actor=ev.get("actor", ""),
                        action=ev.get("action", ""),
                        detail=ev.get("details") or {},
                    )
                )

        # 2) 溯源产出: 产出了什么文件/关键值
        if self._provenance is not None:
            if queries:
                entries = self._provenance.search(queries)
            else:
                entries = [self._as_dict(e) for e in self._provenance.recent(limit)]
            artifacts = list(entries)
            for a in entries:
                steps.append(
                    ExplainStep(
                        ts=a.get("produced_at", 0.0),
                        kind="artifact",
                        produced_by=a.get("produced_by", ""),
                        action=a.get("produced_by", ""),
                        file_path=a.get("file_path", ""),
                        detail=a,
                    )
                )
                # 依赖边: 产出文件 → 其输入文件
                for inp in a.get("input_files") or []:
                    edges.append(
                        {
                            "from_file": inp,
                            "to_file": a["file_path"],
                            "produced_by": a.get("produced_by", ""),
                        }
                    )
                for k, v in (a.get("key_properties") or {}).items():
                    key_findings.setdefault(k, v)

        return Explanation(
            goal=queries,
            steps=steps,
            artifacts=artifacts,
            edges=edges,
            key_findings=key_findings,
        )

    @staticmethod
    def _as_dict(e: Any) -> dict[str, Any]:
        """溯源条目可能是 entry 对象 (有 to_dict) 或已是 dict, 统一为 dict."""
        return e if isinstance(e, dict) else e.to_dict()


def _to_ts(t: Any) -> float:
    """把审计事件的时间戳 (ISO 字符串或 float) 归一化为 float."""
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, str):
        try:
            from datetime import datetime, timezone

            return datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0