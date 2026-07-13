"""Phase-gate hooks for the autoloop engine.

每个阶段转移可以挂在硬性证据上. 证据缺失时门阻断, engine 跳到下一轮迭代,
把 feedback 留在共享状态里供 agent 下一轮参考, 避免半成品往后流.

设计要点:
- PhaseGate: 一次门评估的结果 (status / missing / feedback)
- PhaseGateConfig: 每个转移需要的证据 key 清单
- PhaseGateHook: 纯评估器, 无状态, 可注入 reviewer_fn 做主观审查
- PhaseGateState: 进程内共享单例, 连接 engine (写) 与 PhaseTool (读/写)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal
import logging
logger = logging.getLogger(__name__)



GateStatus = Literal["pending", "approved", "blocked", "rejected"]


@dataclass
class PhaseGate:
    """一次阶段转移门评估的结果."""

    from_phase: str
    to_phase: str
    status: GateStatus
    required_evidence: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    feedback: str = ""
    reviewer: str | None = None

    @property
    def is_blocked(self) -> bool:
        return self.status in ("blocked", "rejected")

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_phase": self.from_phase,
            "to_phase": self.to_phase,
            "status": self.status,
            "required_evidence": list(self.required_evidence),
            "missing_evidence": list(self.missing_evidence),
            "feedback": self.feedback,
            "reviewer": self.reviewer,
        }


# 默认硬性证据清单: (from, to) -> evidence dict 里必须出现的 key
# 这些 key 缺失或为空就阻断, 防止空 plan / 空结果往后传
# Phase names must match huginn.autoloop.engine.AUTOLOOP_PHASES
_DEFAULT_EVIDENCE_REQUIREMENTS: dict[tuple[str, str], list[str]] = {
    ("hypothesize", "plan"): ["hypothesis"],
    ("plan", "execute"): ["mode", "description"],
    ("execute", "validate"): ["mode"],
    ("validate", "learn"): ["tests_passed"],
    ("learn", "report"): [],  # report 总是放行
}


class PhaseGateConfig:
    """每个阶段转移的硬性证据要求."""

    def __init__(
        self,
        requirements: dict[tuple[str, str], list[str]] | None = None,
    ):
        self.requirements = requirements or dict(_DEFAULT_EVIDENCE_REQUIREMENTS)

    def required_for(self, from_phase: str, to_phase: str) -> list[str]:
        return self.requirements.get((from_phase, to_phase), [])

    def add_requirement(
        self, from_phase: str, to_phase: str, keys: list[str]
    ) -> None:
        self.requirements[(from_phase, to_phase)] = keys


# reviewer_fn 签名: (from, to, evidence) -> (approved: bool, reason: str)
ReviewerFn = Callable[[str, str, dict[str, Any]], tuple[bool, str]]

# math_checker_fn 签名: (evidence) -> (passed, feedback, details)
# 用于在硬性证据齐全后做论文级数学证据合成 (Dempster-Shafer).
MathCheckerFn = Callable[[dict[str, Any]], tuple[bool, str, dict[str, Any]]]


# ── Dempster-Shafer 证据合成 ──────────────────────────────────────
#
# 每个 evidence source 给出 (m_pass, m_fail, m_uncertainty) 三元组,
# 满足 m_pass + m_fail + m_unc = 1. 多源用 Dempster 合成规则合并:
#   m12(A) = Σ_{B∩C=A} m1(B)·m2(C) / (1 - K),  K = Σ_{B∩C=∅} m1(B)·m2(C)
# 对二元假设空间 {pass, fail}:
#   K = m_pass1·m_fail2 + m_fail1·m_pass2          (冲突)
#   m_pass  = (m_pass1·m_pass2 + m_pass1·m_unc2 + m_unc1·m_pass2) / (1-K)
#   m_fail  = (m_fail1·m_fail2 + m_fail1·m_unc2 + m_unc1·m_fail2) / (1-K)
#   m_unc   = (m_unc1·m_unc2) / (1-K)
# K=1 表示完全冲突, 不可合成.


class DempsterShaferCombiner:
    """Dempster-Shafer 证据合成. 输入若干 (m_pass, m_fail, m_unc) 三元组,
    输出合并后的三元组."""

    @staticmethod
    def combine_pair(
        m1: tuple[float, float, float],
        m2: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        m_p1, m_f1, m_u1 = m1
        m_p2, m_f2, m_u2 = m2
        K = m_p1 * m_f2 + m_f1 * m_p2
        if K >= 1.0:
            # 完全冲突 — 强制归约到全 fail
            return (0.0, 1.0, 0.0)
        denom = 1.0 - K
        new_pass = (m_p1 * m_p2 + m_p1 * m_u2 + m_u1 * m_p2) / denom
        new_fail = (m_f1 * m_f2 + m_f1 * m_u2 + m_u1 * m_f2) / denom
        new_unc = (m_u1 * m_u2) / denom
        # 数值稳定: 归一化到和=1 (浮点误差可能让和稍微偏离)
        total = new_pass + new_fail + new_unc
        if total > 0:
            new_pass /= total
            new_fail /= total
            new_unc /= total
        else:
            return (0.0, 1.0, 0.0)
        return (new_pass, new_fail, new_unc)

    @staticmethod
    def combine(masses: list[tuple[float, float, float]]) -> tuple[float, float, float]:
        if not masses:
            return (0.0, 0.0, 1.0)  # 无证据 → 全不确定
        result = masses[0]
        for m in masses[1:]:
            result = DempsterShaferCombiner.combine_pair(result, m)
        return result


class MathEvidenceChecker:
    """数学证据检查器 — 论文级 Dempster-Shafer 合成.

    evidence dict 里可能出现以下数学证据 key (由 engine._validate 填充):
      - conservation_law: {"verified": bool, "current": str, "symmetry": str}
      - dimensional_consistent: bool
      - pde_classification: {"consistent": bool, "expected": str, "actual": str}
      - sobol_top_features: {"hypothesis_covers_top": bool, "top_features": list, "hypothesis_features": list}
      - constraint_check: {"all_passed": bool, "violations": list}

    每个 source 给一个 (m_pass, m_fail, m_unc) 三元组, 用 Dempster 合并.
    belief(pass) > threshold 才放行; 否则 block 并把 feedback 拼进 _speculator_hint.
    无任何数学证据 key 时不阻断 (让硬性证据检查决定).
    """

    DEFAULT_THRESHOLD = 0.5

    # 每个 source 的 mass assignment:
    # (verified → m_pass, m_fail, m_unc)  vs  (failed → m_pass, m_fail, m_unc)
    _SOURCE_MASSES = {
        "conservation_law": ((0.70, 0.05, 0.25), (0.05, 0.80, 0.15)),
        "dimensional_consistent": ((0.60, 0.10, 0.30), (0.05, 0.85, 0.10)),
        "pde_classification": ((0.65, 0.05, 0.30), (0.05, 0.75, 0.20)),
        "sobol_top_features": ((0.55, 0.10, 0.35), (0.10, 0.60, 0.30)),
        "constraint_check": ((0.60, 0.10, 0.30), (0.05, 0.80, 0.15)),
    }

    # dual_coverage source: 节点需双覆盖时, 实际是否双覆盖
    # ponytail: masses 与上面 source 同结构, 直接 append 进 masses 列表
    _DUAL_COVERED_MASS = (0.55, 0.10, 0.35)
    _DUAL_NOT_COVERED_MASS = (0.05, 0.80, 0.15)

    def __init__(
        self,
        threshold: float | None = None,
        graph: Any | None = None,
        hypothesis_id_key: str = "hypothesis_id",
    ):
        self.threshold = (
            threshold if threshold is not None else self.DEFAULT_THRESHOLD
        )
        self._graph = graph
        self._hyp_id_key = hypothesis_id_key

    def _extract_source_outcome(self, key: str, value: Any) -> bool | None:
        """从 evidence value 里抽出 verified/consistent/passed 布尔."""
        if isinstance(value, bool):
            return value
        if isinstance(value, dict):
            for k in ("verified", "consistent", "all_passed", "hypothesis_covers_top", "passed"):
                if k in value:
                    return bool(value[k])
        return None

    def __call__(self, evidence: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        masses: list[tuple[float, float, float]] = []
        sources: list[str] = []

        for key, (pass_mass, fail_mass) in self._SOURCE_MASSES.items():
            if key not in evidence:
                continue
            value = evidence[key]
            outcome = self._extract_source_outcome(key, value)
            if outcome is None:
                continue
            masses.append(pass_mass if outcome else fail_mass)
            sources.append(key)

        # dual_coverage: 节点需双覆盖时, 实际是否被双模态支撑.
        # 作为第6条 source 进入 DS 合成 — covered 加分, not-covered 扣分.
        # graph 未注入或 hyp_id 缺失则跳过, 不影响原逻辑.
        if self._graph is not None:
            hyp_id = evidence.get(self._hyp_id_key)
            if hyp_id:
                try:
                    if self._graph.needs_dual_coverage(hyp_id):
                        covered = self._graph.dual_covered(hyp_id)
                        masses.append(
                            self._DUAL_COVERED_MASS if covered
                            else self._DUAL_NOT_COVERED_MASS
                        )
                        sources.append("dual_coverage")
                except Exception:
                    logger.debug("dual_coverage check failed", exc_info=True)

        if not masses:
            # 无数学证据 — 不阻断, 让硬性证据检查决定
            return (True, "no math evidence (skipped)", {"sources": []})

        m_pass, m_fail, m_unc = DempsterShaferCombiner.combine(masses)
        passed = m_pass > self.threshold

        details = {
            "sources": sources,
            "n_sources": len(sources),
            "belief_pass": round(m_pass, 4),
            "belief_fail": round(m_fail, 4),
            "uncertainty": round(m_unc, 4),
            "threshold": self.threshold,
        }

        if passed:
            feedback = (
                f"Math evidence passed: belief_pass={m_pass:.3f} > {self.threshold}, "
                f"sources={sources}"
            )
        else:
            feedback = (
                f"Math evidence insufficient: belief_pass={m_pass:.3f} <= {self.threshold}, "
                f"belief_fail={m_fail:.3f}, uncertainty={m_unc:.3f}, sources={sources}. "
                f"Run symbolic_math_tool (pde_classify/euler_lagrange/noether/diffgeo_metric) "
                f"and symbolic_regression_tool (sobol_indices/constraint_check) to "
                f"strengthen math evidence before re-attempting validate→learn."
            )

        return (passed, feedback, details)


class PhaseGateHook:
    """评估阶段转移门. 纯评估, 无运行时状态.

    证据不足直接返回 blocked, 不抛异常. caller 决定怎么处理.
    可选注入 reviewer_fn 做主观审查 (LLM 调用), 不传就只做硬性检查.
    可选注入 math_checker 做论文级 Dempster-Shafer 证据合成, 在硬性证据
    齐全后评估数学证据是否足够支撑阶段推进.
    """

    def __init__(
        self,
        config: PhaseGateConfig | None = None,
        reviewer_fn: ReviewerFn | None = None,
        math_checker: MathCheckerFn | None = None,
    ):
        self.config = config or PhaseGateConfig()
        self._reviewer_fn = reviewer_fn
        self._math_checker = math_checker

    def evaluate(
        self,
        from_phase: str,
        to_phase: str,
        evidence: dict[str, Any] | Any,
    ) -> PhaseGate:
        required = self.config.required_for(from_phase, to_phase)
        # evidence 可能是 dict 或裸值 (如 hypothesis str), 统一成 dict
        if not isinstance(evidence, dict):
            evidence = {"value": evidence}

        missing = [
            k
            for k in required
            if k not in evidence or evidence[k] in (None, "", [], {})
        ]

        if missing:
            feedback = (
                f"阶段转移 {from_phase}→{to_phase} 被阻断: 缺少证据 {missing}. "
                f"已有证据 keys: {list(evidence.keys())}. "
                f"补齐后再推进, 或用 phase_tool override 强制放行."
            )
            return PhaseGate(
                from_phase=from_phase,
                to_phase=to_phase,
                status="blocked",
                required_evidence=required,
                missing_evidence=missing,
                feedback=feedback,
            )

        # 硬性证据齐全, 走 math_checker (可选, 论文级 Dempster-Shafer 合成).
        # 仅在 validate→learn 转移时触发 — 这是数学证据最有意义的检查点.
        if self._math_checker is not None and from_phase == "validate" and to_phase == "learn":
            try:
                math_passed, math_feedback, math_details = self._math_checker(evidence)
                if not math_passed:
                    return PhaseGate(
                        from_phase=from_phase,
                        to_phase=to_phase,
                        status="blocked",
                        required_evidence=required,
                        missing_evidence=[],
                        feedback=f"Math evidence blocked: {math_feedback}",
                        reviewer="math_checker",
                    )
            except Exception:
                # math_checker 挂了不阻断, 降级放行
                pass

        # 走 reviewer (可选). reviewer 挂了不阻断, 降级放行.
        if self._reviewer_fn is not None:
            try:
                approved, reason = self._reviewer_fn(from_phase, to_phase, evidence)
                if not approved:
                    return PhaseGate(
                        from_phase=from_phase,
                        to_phase=to_phase,
                        status="rejected",
                        required_evidence=required,
                        feedback=f"Reviewer 拒绝: {reason}",
                        reviewer="reviewer",
                    )
            except Exception:
                logger.debug("reviewer fn failed", exc_info=True)

        return PhaseGate(
            from_phase=from_phase,
            to_phase=to_phase,
            status="approved",
            required_evidence=required,
        )


# ── 共享状态: 连接 engine 与 PhaseTool ──────────────────────────


class PhaseGateState:
    """进程内共享的 phase-gate 运行时状态.

    engine 写 (pending_transition / history), PhaseTool 读+写
    (submit_evidence / override). 单例通过 get_shared_phase_gate_state 拿.
    """

    def __init__(self) -> None:
        self.history: list[PhaseGate] = []
        self.pending_transition: tuple[str, str] | None = None
        self.submitted_evidence: dict[str, Any] = {}
        self.overrides: set[tuple[str, str]] = set()
        # override 元数据 (并行于 overrides set): 谁/何时/为何 override
        # ponytail: 并行结构, set.add 不写 meta 则 meta 缺. 升级: set→dict 合并
        self.override_meta: dict[tuple[str, str], dict] = {}

    def reset(self) -> None:
        self.history.clear()
        self.pending_transition = None
        self.submitted_evidence.clear()
        self.overrides.clear()
        self.override_meta.clear()

    def last_gate(self) -> PhaseGate | None:
        return self.history[-1] if self.history else None


_shared_state: PhaseGateState | None = None
_shared_lock = __import__("threading").Lock()


def get_shared_phase_gate_state() -> PhaseGateState:
    global _shared_state
    if _shared_state is None:
        with _shared_lock:
            if _shared_state is None:
                _shared_state = PhaseGateState()
    return _shared_state


def set_shared_phase_gate_state(state: PhaseGateState | None) -> None:
    """测试用: 注入干净实例或清空."""
    global _shared_state
    _shared_state = state
