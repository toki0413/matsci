"""Grader 层 — 把现有物理/量纲校验器统一成"打分"接口.

autoloop 的奖励回流需要一个稳定的 (score, passed, reward) 三元组来驱动
evolution. PhysicsAuditor / DimensionalValidator 各自吐 findings / check
result, 结构不一致; Grader 就是这层适配: 每个校验器包成一个 Grader,
registry 统一跑 evaluate_all 拿回同构的 GraderResult 列表.

只做包装, 不重复实现校验逻辑——校验规则仍归原校验器.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from huginn.execution.physics_auditor import AuditReport, PhysicsAuditor
from huginn.validation.dimensional import DimensionalCheckResult, DimensionalValidator
from huginn.validation.research import RedTeamReviewer


@dataclass
class GraderResult:
    """一次打分的结果, 喂给 autoloop reward 通道用的同构结构."""

    name: str
    score: float          # 0.0 ~ 1.0, 综合分
    passed: bool          # 是否达到"可接受"门槛 (无 error / 量纲一致)
    reward: float = 0.0   # 奖励信号, 默认等于 score, 调用方可加权
    checks: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""

    def __post_init__(self) -> None:
        # reward 没显式给就用 score, 避免每个调用方都写一遍
        if self.reward == 0.0 and self.score > 0.0:
            self.reward = self.score


# 严重度 -> 分数权重. error 直接判 0, warning 折半, info 不扣分
_SEV_SCORE = {"error": 0.0, "warning": 0.5, "info": 1.0}


class PhysicsGrader:
    """包 PhysicsAuditor: 审计计算结果, findings 折算成 score."""

    name = "physics"

    def __init__(self, auditor: PhysicsAuditor | None = None) -> None:
        self._auditor = auditor or PhysicsAuditor()

    def evaluate(self, data: dict[str, Any]) -> GraderResult:
        tool = data.get("tool_name", "unknown")
        action = data.get("action", "")
        parsed = data.get("parsed") or data.get("result_data") or {}
        params = data.get("input_params") or {}
        report: AuditReport = self._auditor.audit(tool, action, parsed, params)

        findings = report.findings
        if not findings:
            return GraderResult(
                name=self.name, score=1.0, passed=True,
                message="no findings", checks=[],
            )
        # 加权平均: error 拉低, warning 折半, info 满分
        total = sum(_SEV_SCORE.get(f.severity, 0.5) for f in findings)
        score = round(total / len(findings), 4)
        passed = not report.has_errors
        checks = [f.to_dict() for f in findings]
        msg = "; ".join(f.message for f in findings)
        return GraderResult(
            name=self.name, score=score, passed=passed,
            checks=checks, message=msg,
        )

    # 让实例可直接当 callable 塞进 registry
    def __call__(self, data: dict[str, Any]) -> GraderResult:
        return self.evaluate(data)


class DimensionalGrader:
    """包 DimensionalValidator: 校验方程量纲一致性."""

    name = "dimensional"

    def __init__(self, validator: DimensionalValidator | None = None) -> None:
        self._validator = validator or DimensionalValidator()

    def evaluate(self, data: dict[str, Any]) -> GraderResult:
        lhs = data.get("lhs_quantities") or []
        rhs = data.get("rhs_quantities") or []
        eq_name = data.get("equation_name", "")
        if not lhs or not rhs:
            return GraderResult(
                name=self.name, score=0.0, passed=False,
                message="missing lhs_quantities / rhs_quantities",
            )
        res: DimensionalCheckResult = self._validator.check_equation(
            lhs, rhs, equation_name=eq_name,
        )
        score = 1.0 if res.consistent else 0.0
        return GraderResult(
            name=self.name, score=score, passed=res.consistent,
            checks=[{
                "equation": res.equation,
                "consistent": res.consistent,
                "lhs_dimensions": res.lhs_dimensions,
                "rhs_dimensions": res.rhs_dimensions,
                "notes": res.notes,
            }],
            message="; ".join(res.notes),
        )

    def __call__(self, data: dict[str, Any]) -> GraderResult:
        return self.evaluate(data)


class RedTeamGrader:
    """包 RedTeamReviewer: 对抗性审查研究逻辑.

    data: {"from_phase", "to_phase", "evidence"}
    """

    name = "red_team"

    def __init__(self, reviewer: RedTeamReviewer | None = None) -> None:
        self._reviewer = reviewer or RedTeamReviewer()

    def evaluate(self, data: dict[str, Any]) -> GraderResult:
        from_phase = data.get("from_phase", "")
        to_phase = data.get("to_phase", "")
        evidence = data.get("evidence", {})
        report = self._reviewer.review(from_phase, to_phase, evidence)

        findings = report.findings
        if not findings:
            return GraderResult(
                name=self.name, score=1.0, passed=True,
                message="no findings",
            )
        # blocking -> 0, 每条发现扣 0.1
        if report.has_blocking:
            score = 0.0
        else:
            score = max(0.0, 1.0 - 0.1 * len(findings))
        checks = [
            {"severity": f.severity, "description": f.description}
            for f in findings
        ]
        return GraderResult(
            name=self.name, score=score,
            passed=not report.has_blocking,
            checks=checks,
            message="; ".join(f.description for f in findings),
        )

    def __call__(self, data: dict[str, Any]) -> GraderResult:
        return self.evaluate(data)


class HallucinationGrader:
    """断言式幻觉检测 — 纯规则检查, 不调 LLM.

    data: {"text": str, "expected_facts": list[str] (可选)}
    """

    name = "hallucination"

    _PATTERNS: list[tuple[str, str]] = [
        (r"(?:百分之百|100\s*%|绝对|毫无疑问|确定无疑|万无一失)", "overconfident assertion"),
        (r"\d+\.\d{6,}", "suspicious precision without citation"),
        (r"室温超导", "unlikely physics claim without evidence"),
    ]

    def __init__(self) -> None:
        self._compiled = [
            (re.compile(p, re.IGNORECASE), label) for p, label in self._PATTERNS
        ]

    def evaluate(self, data: dict[str, Any]) -> GraderResult:
        text = data.get("text") or data.get("output") or ""
        if not isinstance(text, str):
            text = str(text)
        issues: list[str] = []

        for pattern, label in self._compiled:
            matches = pattern.findall(text)
            if matches:
                issues.append(f"{label}: {len(matches)} occurrence(s)")

        for fact in data.get("expected_facts", []):
            if fact.lower() not in text.lower():
                issues.append(f"missing expected fact: {fact}")

        n_issues = len(issues)
        score = max(0.0, 1.0 - 0.2 * n_issues)
        return GraderResult(
            name=self.name, score=score, passed=n_issues == 0,
            checks=[{"issue": i} for i in issues],
            message="; ".join(issues),
        )

    def __call__(self, data: dict[str, Any]) -> GraderResult:
        return self.evaluate(data)


class GraderRegistry:
    """注册多个 Grader, 统一跑 evaluate_all 拿同构结果."""

    def __init__(self) -> None:
        self._graders: dict[str, Callable[[dict[str, Any]], GraderResult]] = {}

    def register(
        self,
        name: str,
        grader: Callable[[dict[str, Any]], GraderResult],
    ) -> None:
        self._graders[name] = grader

    def evaluate_all(self, data: dict[str, Any]) -> list[GraderResult]:
        return [g(data) for g in self._graders.values()]

    def names(self) -> list[str]:
        return list(self._graders)


def default_registry() -> GraderRegistry:
    """预注册 physics + dimensional + hallucination 三个内置 grader."""
    reg = GraderRegistry()
    reg.register("physics", PhysicsGrader())
    reg.register("dimensional", DimensionalGrader())
    reg.register("hallucination", HallucinationGrader())
    return reg


__all__ = [
    "GraderResult",
    "PhysicsGrader",
    "DimensionalGrader",
    "RedTeamGrader",
    "HallucinationGrader",
    "GraderRegistry",
    "default_registry",
]
