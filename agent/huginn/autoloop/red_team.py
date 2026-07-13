"""Red-team 对抗性审查 — 在关键阶段转移点生成反驳意见.

R3 (W3): 作为 PhaseGateHook 的 reviewer_fn 注入. 在 hypothesize→plan 和
validate→learn 两个转移点触发, 生成对抗性反驳 (隐含前提 / 混淆变量 / 替代解释 /
方法论缺陷). 高严重度未消解 → blocked.

两种模式:
- rule-based (默认, model=None): 按规则检查 evidence 里的常见问题, 确定性, 测试用.
- LLM-enhanced (model 传入): 规则检查 + LLM 生成更深层的对抗性意见.

ReviewerFn 接口: __call__(from, to, evidence) -> (approved: bool, reason: str)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any
import logging
logger = logging.getLogger(__name__)


def _dominant_source_class(evidence: dict[str, Any]) -> str:
    """从 evidence 递归收集 source_class, 返回占比最高的.

    用于 LLM 没标 source_class 的 finding 自动派生. 没找到返回 "".
    ponytail: depth=20 防异常深度 (evidence 实际嵌套 < 10).
    """
    from huginn.autoloop.phase_gate import _collect_source_classes
    classes = _collect_source_classes(evidence)
    if not classes:
        return ""
    return Counter(classes).most_common(1)[0][0]


# Phase-gate 用的阶段名是字符串字面量, 不是 ResearchPhase enum
# (见 phase_gate.py._DEFAULT_EVIDENCE_REQUIREMENTS)
_REVIEW_TRANSITIONS: set[tuple[str, str]] = {
    ("hypothesize", "plan"),
    ("validate", "learn"),
}


# ── data structures ──────────────────────────────────────────────────────────


@dataclass
class RedTeamFinding:
    """一条对抗性发现."""

    category: str  # hidden_assumption | confounder | alternative_explanation | methodology_gap
    description: str
    severity: str  # high | medium | low
    mitigation: str = ""
    # ARGUS: 这条 finding 基于哪类来源得出的判断.
    # user_input / tool_output / external_content / agent_generated 四选一.
    # ponytail: LLM 自报 + evidence 自动派生兜底. 空串时通过 effective_source_class 兜底.
    source_class: str = ""

    @property
    def effective_source_class(self) -> str:
        """空串时兜底为 agent_generated (LLM 没声明就是 agent 自己的判断)."""
        return self.source_class or "agent_generated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "description": self.description,
            "severity": self.severity,
            "mitigation": self.mitigation,
            "source_class": self.effective_source_class,
        }


@dataclass
class RedTeamReport:
    """一次 red-team 审查的完整报告."""

    transition: tuple[str, str]
    findings: list[RedTeamFinding] = field(default_factory=list)
    summary: str = ""

    @property
    def has_blocking(self) -> bool:
        """有 high 严重度发现 → 阻断."""
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


# ── reviewer ─────────────────────────────────────────────────────────────────


class RedTeamReviewer:
    """对抗性审查器, 注入 PhaseGateHook 做 reviewer_fn.

    在 hypothesize→plan 检查假设的隐含前提和可证伪性;
    在 validate→learn 检查验证的充分性和替代解释.
    其他转移点直接放行.

    model 不为 None 时会尝试 LLM 增强 (生成更深层反驳), 失败则降级到纯规则.
    """

    def __init__(
        self,
        model: Any | None = None,
        enabled_transitions: set[tuple[str, str]] | None = None,
        failure_mode_registry: Any | None = None,
        critic_model: Any | None = None,
    ) -> None:
        self._model = model
        # 跨模型审查: 优先用 critic_model 做对抗, 避免同模型自审 (confirmation bias).
        # None 时 fallback 到 self._model, 保持向后兼容.
        # ponytail: 单 critic, 升级路径: 并行多 critic 投票 + Dempster 合成.
        self._critic_model = critic_model
        self._enabled = enabled_transitions or _REVIEW_TRANSITIONS
        self._last_report: RedTeamReport | None = None
        # 领域失败模式注册表 (材料科学具体陷阱). None 时用默认单例.
        # ponytail: 懒导入避免循环依赖, 测试可注入自定义 registry
        if failure_mode_registry is None:
            from huginn.metacog.failure_modes import DEFAULT_REGISTRY
            self._failure_registry = DEFAULT_REGISTRY
        else:
            self._failure_registry = failure_mode_registry

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
    ) -> RedTeamReport:
        """执行审查, 返回 RedTeamReport."""
        transition = (from_phase, to_phase)
        if from_phase == "hypothesize":
            findings = self._review_hypothesis(evidence)
        elif from_phase == "validate":
            findings = self._review_validation(evidence)
        else:
            findings = []

        # LLM 增强 (可选, 失败不阻塞).
        # 跨模型审查: 优先用 critic_model, fallback 到 self._model.
        # 同模型自审有 confirmation bias, 但比无 LLM 增强好.
        # ponytail: 单 critic 同步调用. 升级: 异步并行多 critic 投票.
        model_for_review = self._critic_model or self._model
        if model_for_review is not None and not hasattr(model_for_review, "_mock_name"):
            try:
                findings.extend(self._llm_findings(from_phase, to_phase, evidence))
            except Exception:
                logger.debug("extend failed", exc_info=True)  # LLM 挂了用规则结果

        # 领域失败模式扫描 (材料科学具体陷阱: 数据泄漏 / 单位混乱 / 对称性 / ...)
        # 把通用规则检查无法覆盖的领域具体失败模式补上.
        findings.extend(self._domain_failure_scan(evidence))

        summary = self._build_summary(findings, transition)
        return RedTeamReport(transition=transition, findings=findings, summary=summary)

    def _domain_failure_scan(self, evidence: dict[str, Any]) -> list[RedTeamFinding]:
        """扫描材料科学领域失败模式清单, 把命中的转成 RedTeamFinding.

        severity 映射: block→high (会阻断), warn→medium, info→low.
        这样 first-principles-violation (warn) 不会硬阻断, 对齐用户
        '先警告再 force proceed' 偏好.
        """
        if self._failure_registry is None:
            return []
        # evidence 里可能带 method_family 标记, 没有就当通用
        family_id = evidence.get("method_family")
        hit_modes = self._failure_registry.scan(evidence, family_id=family_id)
        sev_map = {"block": "high", "warn": "medium", "info": "low"}
        out: list[RedTeamFinding] = []
        for mode in hit_modes:
            out.append(RedTeamFinding(
                category="methodology_gap",  # 复用现有 category, 不改 enum
                description=f"[{mode.id}] {mode.description}",
                severity=sev_map.get(mode.severity, "medium"),
                mitigation=mode.mitigation,
            ))
        return out

    # ── 规则审查 ────────────────────────────────────────────────────

    def _review_hypothesis(self, evidence: dict[str, Any]) -> list[RedTeamFinding]:
        """审查假设: 隐含前提 / 可证伪性 / 混淆变量."""
        findings: list[RedTeamFinding] = []
        hyp = self._extract_hypothesis(evidence)
        if not hyp:
            findings.append(RedTeamFinding(
                category="methodology_gap",
                description="假设为空或未明确表述, 无法做对抗性审查",
                severity="high",
                mitigation="明确写出可检验的假设, 包括自变量、因变量、预期关系",
            ))
            return findings

        # 可证伪性: 有没有 if-then 结构
        falsifiable_markers = ["如果", "if ", "当", "when ", "则", "then", "若", "假设"]
        if not any(m in hyp.lower() for m in falsifiable_markers):
            findings.append(RedTeamFinding(
                category="methodology_gap",
                description="假设缺乏可证伪的预测结构 (if-then 形式), 难以实验反驳",
                severity="medium",
                mitigation="重写为 '如果 X 成立, 则应观察到 Y' 的形式",
            ))

        # 隐含前提: 长假设没提边界条件 (中文紧凑, 阈值比英文低)
        if len(hyp) > 30 and not any(
            m in hyp.lower() for m in ["前提", "assumption", "given", "assuming", "条件", "范围"]
        ):
            findings.append(RedTeamFinding(
                category="hidden_assumption",
                description="假设较长但未显式列出边界条件, 可能遗漏关键隐含前提",
                severity="medium",
                mitigation="列出温度范围、尺度限制、理想化条件等隐含假设",
            ))

        # 混淆变量: 没提控制变量
        if not any(m in hyp.lower() for m in ["控制", "control", "固定", "fixed", "排除"]):
            findings.append(RedTeamFinding(
                category="confounder",
                description="未提及控制变量, 可能存在混淆变量影响因果归因",
                severity="low",
                mitigation="列出可能的混淆变量及控制策略",
            ))

        return findings

    def _review_validation(self, evidence: dict[str, Any]) -> list[RedTeamFinding]:
        """审查验证: 测试通过 / 替代解释 / 收敛性 / 物理 oracle 警告."""
        findings: list[RedTeamFinding] = []

        tests_passed = evidence.get("tests_passed")
        if tests_passed is False:
            findings.append(RedTeamFinding(
                category="methodology_gap",
                description="验证未通过测试, 结论不可靠",
                severity="high",
                mitigation="修复失败项, 或降低结论强度并标注局限性",
            ))

        # 物理 oracle findings: PhaseGate 已在 has_errors=True 时硬阻断, 这里只看 warnings.
        # 把 warning 级 finding 提升为 medium, 让红队报告对物理警告不再失明.
        # ponytail: 只看 severity=warning, error 已被 PhaseGate 拦截. 升级: DS 合成置信度.
        pa = evidence.get("physics_audit")
        if isinstance(pa, dict):
            for f in (pa.get("findings") or []):
                sev = str(f.get("severity", "")).lower()
                if sev in ("warning", "warn"):
                    findings.append(RedTeamFinding(
                        category="methodology_gap",
                        description=f"Physics audit warning: {f.get('category', '?')} — {f.get('message', '')}",
                        severity="medium",
                        mitigation="确认物理合理性后再采信结果, 或修复后重跑",
                    ))

        # 视觉自验证: 工具产出的数值数据有 _visual_self_check 字段 (Nullmax 启发).
        # low confidence 或有 caveats 时, 红队报告标注可视化结论不可信.
        vsc = evidence.get("_visual_self_check")
        if isinstance(vsc, dict):
            conf = float(vsc.get("confidence", 1.0))
            caveats = vsc.get("caveats") or []
            if conf < 0.3 or any("too_few" in c or "low_snr" in c for c in caveats):
                findings.append(RedTeamFinding(
                    category="methodology_gap",
                    description=f"可视化数据置信度低 (confidence={conf:.2f}): {'; '.join(caveats[:3])}",
                    severity="medium",
                    mitigation="增加数据点 / 检查数据质量 / 用替代方法交叉验证可视化结论",
                ))

        mode = str(evidence.get("mode", ""))
        # 单一方法验证: 没有交叉验证
        if mode and not any(
            m in str(evidence).lower() for m in ["cross", "交叉", "对比", "baseline", "基准"]
        ):
            findings.append(RedTeamFinding(
                category="alternative_explanation",
                description="仅用单一方法验证, 未排除替代解释 (参数巧合 / 代码 bug / 数据泄漏)",
                severity="medium",
                mitigation="用独立方法或基准交叉验证, 排除替代解释",
            ))

        return findings

    # ── LLM 增强 ────────────────────────────────────────────────────

    def _is_real_model(self) -> bool:
        """检测是不是 MagicMock (测试注入的)."""
        return not hasattr(self._model, "_mock_name")

    def _llm_findings(
        self, from_phase: str, to_phase: str, evidence: dict[str, Any]
    ) -> list[RedTeamFinding]:
        """用 LLM 生成对抗性意见. 失败返回空列表 (调用方 try/except)."""
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = self._build_llm_prompt(from_phase, to_phase, evidence)
        messages = [
            SystemMessage(content=(
                "你是红队审查员 (red-team reviewer). 任务: 对下面的研究证据做对抗性审查, "
                "找出隐含前提、混淆变量、替代解释、方法论缺陷. "
                "输出 JSON 数组, 每条: {category, description, severity, mitigation, source_class}. "
                "category ∈ hidden_assumption|confounder|alternative_explanation|methodology_gap. "
                "severity ∈ high|medium|low. "
                "source_class ∈ user_input|tool_output|external_content, "
                "标这条 finding 基于哪类来源得出 (ARGUS 影响溯源用). 没问题就输出 []. "
                "反向激励: 假设证据有错, 找最可能失败的点, 不要验证正确性."
            )),
            HumanMessage(content=prompt),
        ]
        # 用同步 invoke 避免在 async 引擎上下文里 run_until_complete 报错.
        # 跨模型审查: 优先 critic_model, fallback 主 model (与 review() 同源选择).
        critic = self._critic_model or self._model
        resp = critic.invoke(messages)
        text = str(resp.content).strip()
        findings = self._parse_llm_findings(text)
        # ARGUS: LLM 没标 source_class 的 finding, 从 evidence 自动派生 dominant.
        # ponytail: 调用方诚实标记 + 自动兜底. 升级: 按参数值路径精确溯源.
        dominant = _dominant_source_class(evidence)
        if dominant:
            for f in findings:
                if not f.source_class:
                    f.source_class = dominant
        return findings

    def _build_llm_prompt(
        self, from_phase: str, to_phase: str, evidence: dict[str, Any]
    ) -> str:
        import json

        return (
            f"阶段转移: {from_phase} → {to_phase}\n"
            f"证据: {json.dumps(evidence, ensure_ascii=False, default=str)}\n\n"
            f"请做对抗性审查."
        )

    @staticmethod
    def _parse_llm_findings(text: str) -> list[RedTeamFinding]:
        """解析 LLM 返回的 JSON 数组. 解析失败返回空列表."""
        import json

        try:
            # 去掉可能的 markdown 代码块包裹
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            items = json.loads(text)
            findings = []
            for item in items:
                findings.append(RedTeamFinding(
                    category=item.get("category", "methodology_gap"),
                    description=item.get("description", ""),
                    severity=item.get("severity", "medium"),
                    mitigation=item.get("mitigation", ""),
                    source_class=item.get("source_class", ""),
                ))
            return findings
        except (json.JSONDecodeError, TypeError):
            return []

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_hypothesis(evidence: dict[str, Any]) -> str:
        """从 evidence 里抽假设文本."""
        for key in ("hypothesis", "value", "description"):
            val = evidence.get(key)
            if val and isinstance(val, str):
                return val
        return ""

    @staticmethod
    def _build_summary(findings: list[RedTeamFinding], transition: tuple[str, str]) -> str:
        if not findings:
            return ""
        from_str, to_str = transition
        parts = [f"Red-team 审查 {from_str}→{to_str}: {len(findings)} 条发现."]
        for f in findings:
            parts.append(f"  [{f.severity}] {f.category}: {f.description}")
        return "\n".join(parts)


__all__ = [
    "RedTeamReviewer",
    "RedTeamReport",
    "RedTeamFinding",
]
