"""Blue-team 防守校验 — 与 RedTeamReviewer 对偶的防守侧评审.

概念框架借鉴 *Awesome Cyber Security University* 的红/蓝双轨: 红队主动进攻
(尝试突破边界), 蓝队断言控制按预期生效且**不静默失效**. 只有单边的攻击面是
治理盲区 —— 红队找到反驳, 却无人确认"已有的防守是否真的在动".

本评审器注入 PhaseGateHook 做第二个 reviewer_fn (在红队之后跑). 与红队的
对抗性反驳互补, 蓝队只问一件事: **被声明的控制是否真的在位且有效**. 三类盲区:

- provenance_gap: 外部来源内容 (``source_class == "external_content"``) 被当作
  既定事实推进, 却没有 user_input / tool_output 的独立佐证 → 高 (来源信任控制失效,
  这是提示注入路径).
- no_redundancy: 声称阳性结果 (``tests_passed == True``) 但没有任何独立校验
  (cross / baseline / independent) → 中 (defense-in-depth 缺失, 单点控制).
- guard_integrity: 守卫自己报告不可信 (``_visual_self_check`` 低置信 /
  ``physics_audit`` warning) 但流程仍按结论推进 → 中 (守卫失效被忽略).

确定性规则实现, 无 LLM 依赖. 只有 provenance_gap 会阻断 (高), 因此正常证据
(无 external_content) 不受影响 —— 与红队"只在攻击形状的证据上阻断"同构.

ReviewerFn 接口: ``__call__(from, to, evidence) -> (approved: bool, reason: str)``
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from huginn.autoloop.phase_gate import _collect_source_classes

logger = logging.getLogger(__name__)


# 与红队对齐: 评审同样只在这两个转移点触发 (成对覆盖).
_REVIEW_TRANSITIONS: set[tuple[str, str]] = {
    ("hypothesize", "plan"),
    ("validate", "learn"),
}

# 独立校验的标记词 (证据里出现即认为有冗余控制).
_INDEPENDENT_MARKERS = (
    "cross", "交叉", "baseline", "基准", "independent", "独立",
    "replicat", "复现", "second_method", "对照组",
)


@dataclass
class BlueTeamFinding:
    """一条防守侧发现."""

    category: str  # provenance_gap | no_redundancy | guard_integrity
    description: str
    severity: str  # high | medium | low
    mitigation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "description": self.description,
            "severity": self.severity,
            "mitigation": self.mitigation,
        }


@dataclass
class BlueTeamReport:
    """一次 blue-team 校验的完整报告."""

    transition: tuple[str, str]
    findings: list[BlueTeamFinding] = field(default_factory=list)
    summary: str = ""

    @property
    def has_blocking(self) -> bool:
        return any(f.severity == "high" for f in self.findings)

    @property
    def n_findings(self) -> int:
        return len(self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition": list(self.transition),
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "has_blocking": self.has_blocking,
            "n_findings": self.n_findings,
        }


class BlueTeamReviewer:
    """防守校验器, 注入 PhaseGateHook 做第二个 reviewer_fn (红队之后).

    断言被声明的控制真正在位且有效. 只有 provenance_gap (外部内容无佐证被当
    事实) 会阻断; 其余为 medium/low 提示.
    """

    def __init__(
        self,
        enabled_transitions: set[tuple[str, str]] | None = None,
    ) -> None:
        self._enabled = enabled_transitions or _REVIEW_TRANSITIONS
        self._last_report: BlueTeamReport | None = None

    def __call__(
        self, from_phase: str, to_phase: str, evidence: dict[str, Any]
    ) -> tuple[bool, str]:
        """ReviewerFn 接口: 返回 (approved, reason)."""
        if (from_phase, to_phase) not in self._enabled:
            return True, ""
        report = self.review(from_phase, to_phase, evidence)
        self._last_report = report
        if report.has_blocking:
            return False, report.summary
        return True, ""

    def review(
        self, from_phase: str, to_phase: str, evidence: dict[str, Any]
    ) -> BlueTeamReport:
        """执行防守校验, 返回 BlueTeamReport."""
        transition = (from_phase, to_phase)
        findings: list[BlueTeamFinding] = []
        findings.extend(self._check_provenance(evidence))
        findings.extend(self._check_redundancy(evidence))
        findings.extend(self._check_guard_integrity(evidence))
        summary = self._build_summary(findings, transition)
        return BlueTeamReport(transition=transition, findings=findings, summary=summary)

    # ── 控制检查 ────────────────────────────────────────────────────

    @staticmethod
    def _check_provenance(evidence: dict[str, Any]) -> list[BlueTeamFinding]:
        """来源信任控制: 外部内容未经独立佐证即被推进 → 高 (阻断)."""
        classes = _collect_source_classes(evidence)
        if not classes:
            return []
        counter = Counter(classes)
        if counter.get("external_content", 0) == 0:
            return []
        corroborated = counter.get("user_input", 0) + counter.get("tool_output", 0)
        if corroborated:
            return [BlueTeamFinding(
                category="provenance_gap",
                description=(
                    "证据含外部来源内容 (external_content), 但已有 user_input/"
                    "tool_output 佐证. 仍需确认佐证独立于被注入内容."
                ),
                severity="low",
                mitigation="确认佐证来源不是同一外部内容的变体",
            )]
        return [BlueTeamFinding(
            category="provenance_gap",
            description=(
                f"证据含 {counter['external_content']} 处外部来源内容 "
                "(external_content) 但无 user_input/tool_output 独立佐证, "
                "外部内容被当作既定事实推进 —— 来源信任控制失效, 提示注入路径."
            ),
            severity="high",
            mitigation="补独立验证: 用本地工具复算/用户确认, 或降级为待验证假设",
        )]

    @staticmethod
    def _check_redundancy(evidence: dict[str, Any]) -> list[BlueTeamFinding]:
        """冗余控制: 声称阳性但无独立校验 → 中 (defense-in-depth 缺失)."""
        if evidence.get("tests_passed") is not True:
            return []
        blob = str(evidence).lower()
        if any(m.lower() in blob for m in _INDEPENDENT_MARKERS):
            return []
        return [BlueTeamFinding(
            category="no_redundancy",
            description=(
                "声称验证通过 (tests_passed=True) 但无独立校验/基准/交叉验证, "
                "单一控制无冗余 —— 控制静默失效时无可发现信号."
            ),
            severity="medium",
            mitigation="加交叉验证或基准对照, 让控制失效可被独立观测",
        )]

    @staticmethod
    def _check_guard_integrity(evidence: dict[str, Any]) -> list[BlueTeamFinding]:
        """守卫自身完整性: 守卫报告不可信但流程仍推进 → 中."""
        out: list[BlueTeamFinding] = []

        vsc = evidence.get("_visual_self_check")
        if isinstance(vsc, dict):
            conf = float(vsc.get("confidence", 1.0))
            if conf < 0.3:
                out.append(BlueTeamFinding(
                    category="guard_integrity",
                    description=(
                        f"可视化守卫自报置信度低 (confidence={conf:.2f}), "
                        "守卫自身不可信, 却被当作有效控制."
                    ),
                    severity="medium",
                    mitigation="复核守卫输出或作废该守卫结论",
                ))

        pa = evidence.get("physics_audit")
        if isinstance(pa, dict):
            warnings = [f for f in (pa.get("findings") or [])
                        if str(f.get("severity", "")).lower() in ("warning", "warn")]
            if warnings:
                out.append(BlueTeamFinding(
                    category="guard_integrity",
                    description=(
                        f"物理守卫报 {len(warnings)} 条 warning 但流程仍推进, "
                        "守卫发现被忽略 (控制静默失效)."
                    ),
                    severity="medium",
                    mitigation="确认物理警告后再采信, 或修复后重跑",
                ))

        return out

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _build_summary(
        findings: list[BlueTeamFinding], transition: tuple[str, str]
    ) -> str:
        if not findings:
            return ""
        from_str, to_str = transition
        parts = [f"Blue-team 校验 {from_str}→{to_str}: {len(findings)} 条发现."]
        for f in findings:
            line = f"  [{f.severity}] {f.category}: {f.description}"
            if f.mitigation:
                line += f" → 修复: {f.mitigation}"
            parts.append(line)
        return "\n".join(parts)


__all__ = [
    "BlueTeamReviewer",
    "BlueTeamReport",
    "BlueTeamFinding",
]
