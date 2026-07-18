"""Phase-gate hooks for the autoloop engine.

缺证据默认走 advisory: warning + feedback, 不阻断 transition, 避免 agent
"curl override gradient" (绕过 gate 而非补证据). human_checkpoint_phases
仍硬阻断 — 人工 checkpoint 必须用户 override 才能放行.

设计要点:
- PhaseGate: 一次门评估的结果 (status / missing / feedback)
- PhaseGateConfig: 每个转移需要的证据 key 清单
- PhaseGateHook: 纯评估器, 无状态, 可注入 reviewer_fn 做主观审查
- PhaseGateState: 进程内共享单例, 连接 engine (写) 与 PhaseTool (读/写)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Literal
import logging
logger = logging.getLogger(__name__)


def _load_checkpoint_config() -> set[tuple[str, str]]:
    """从 HUGINN_HUMAN_CHECKPOINT_PHASES 环境变量加载 checkpoint 配置.

    格式: "plan:execute,validate:learn"
    空或未设置时返回空集 (不启用任何 checkpoint).
    """
    raw = os.environ.get("HUGINN_HUMAN_CHECKPOINT_PHASES", "").strip()
    if not raw:
        return set()
    result: set[tuple[str, str]] = set()
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        frm, to = pair.split(":", 1)
        result.add((frm.strip(), to.strip()))
    return result


def _load_hard_checkpoint_config() -> set[tuple[str, str]]:
    """从 HUGINN_HARD_CHECKPOINT_PHASES 加载硬性 checkpoint 配置.

    硬性 checkpoint 与普通 checkpoint 的区别: 不可超时自动放行.
    普通 checkpoint 600s 后 force proceed, 硬性的一直阻塞到用户显式 override.
    用户仍可 override 强制放行 (符合 "warnings first, option to force proceed" 偏好),
    只是不会被超时偷偷放过去.

    格式同 _load_checkpoint_config: "plan:execute,validate:learn"
    """
    raw = os.environ.get("HUGINN_HARD_CHECKPOINT_PHASES", "").strip()
    if not raw:
        return set()
    result: set[tuple[str, str]] = set()
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        frm, to = pair.split(":", 1)
        result.add((frm.strip(), to.strip()))
    return result



GateStatus = Literal["pending", "approved", "blocked", "rejected"]


def _has_external_source(obj: Any, _depth: int = 0) -> bool:
    """递归扫 dict/list 找 source_class=external_content. ponytail: 写死 depth=50 (实际 evidence 嵌套 < 10)."""
    if _depth > 50 or not isinstance(obj, (dict, list, tuple)):
        return False
    if isinstance(obj, dict) and obj.get("source_class") == "external_content":
        return True
    if isinstance(obj, dict):
        return any(_has_external_source(v, _depth + 1) for v in obj.values())
    return any(_has_external_source(v, _depth + 1) for v in obj)


def _collect_source_classes(obj: Any, _depth: int = 0) -> list[str]:
    """递归收集所有 source_class 值. ponytail: depth=50 防异常深度."""
    if _depth > 50 or not isinstance(obj, (dict, list, tuple)):
        return []
    found: list[str] = []
    if isinstance(obj, dict):
        sc = obj.get("source_class")
        if isinstance(sc, str) and sc:
            found.append(sc)
        for v in obj.values():
            found.extend(_collect_source_classes(v, _depth + 1))
    else:
        for v in obj:
            found.extend(_collect_source_classes(v, _depth + 1))
    return found


# 每个 source_class 的 (m_pass, m_fail, m_uncertainty) 先验.
# ponytail: 取值基于经验, 未做大规模数据校准. 升级: 从历史任务拟合.
_SOURCE_CLASS_MASSES: dict[str, tuple[float, float, float]] = {
    "user_input":         (0.70, 0.05, 0.25),   # 用户指令高可信
    "tool_output":        (0.60, 0.20, 0.20),   # 工具输出中等可信
    "external_content":   (0.30, 0.40, 0.30),   # 外部内容可信度低 (可能被注入)
    "agent_generated":    (0.50, 0.30, 0.20),   # agent 生成默认
}


def _argus_confidence(evidence: dict[str, Any]) -> tuple[float, str]:
    """DS 合成 evidence 里所有 source_class 的 (m_pass, m_fail, m_unc).

    返回 (confidence, dominant_source_class):
    - confidence: DS 合成后的 m_pass, 作为 evidence 整体可信度
    - dominant_source_class: 占比最高的来源类 (feedback 文本用)
    ponytail: 无 source_class 时返回 (1.0, "") 不降级.
    """
    classes = _collect_source_classes(evidence)
    if not classes:
        return (1.0, "")
    masses = [
        _SOURCE_CLASS_MASSES.get(c, _SOURCE_CLASS_MASSES["agent_generated"])
        for c in classes
    ]
    combined = DempsterShaferCombiner.combine(masses)
    # 统计 dominant
    from collections import Counter
    counter = Counter(classes)
    dominant = counter.most_common(1)[0][0]
    return (combined[0], dominant)


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
    # OAK 启发: trace_id 贯穿, 让前端能按 trace 聚合而非线性展示
    trace_id: str | None = None
    parent_trace_id: str | None = None  # fork 时记录父 trace

    @property
    def is_blocked(self) -> bool:
        return self.status in ("blocked", "rejected")

    def __post_init__(self) -> None:
        """遥测: 把每个 PhaseGate 决策写到 HUGINN_TELEMETRY_PATH (jsonl).
        ponytail: 用 __post_init__ 拦截所有 5 个 return 点, 0 处业务代码改动.
        失败静默 — 遥测挂了不影响 phase gate 本身. 升级: 加 evidence_keys.
        """
        path = os.environ.get("HUGINN_TELEMETRY_PATH", "")
        if not path:
            return
        try:
            import json
            import time
            record = {
                "ts": time.time(),
                "from_phase": self.from_phase,
                "to_phase": self.to_phase,
                "status": self.status,
                "missing": self.missing_evidence,
                "reviewer": self.reviewer,
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_phase": self.from_phase,
            "to_phase": self.to_phase,
            "status": self.status,
            "required_evidence": list(self.required_evidence),
            "missing_evidence": list(self.missing_evidence),
            "feedback": self.feedback,
            "reviewer": self.reviewer,
            "trace_id": self.trace_id,
            "parent_trace_id": self.parent_trace_id,
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

    @staticmethod
    def conflict(masses: list[tuple[float, float, float]]) -> float:
        """返回 DS 合成的全局冲突度 K ∈ [0, 1).

        标准 Dempster 规则在 K→1 时 m_pass 异常放大 (DS 痛点). 调用方
        检测 K > 0.5 后应改用 combine_robust (Smets 折扣) 降权不可信源.
        之前 combine 不返回 K, 调用方无法感知冲突 — 算子层的高阶合成
        盲跑, 高冲突场景下结论失真.
        """
        if len(masses) < 2:
            return 0.0
        result = masses[0]
        max_k = 0.0
        for m in masses[1:]:
            m_p1, m_f1, _ = result
            m_p2, m_f2, _ = m
            k = m_p1 * m_f2 + m_f1 * m_p2
            max_k = max(max_k, k)
            result = DempsterShaferCombiner.combine_pair(result, m)
        return max_k

    @staticmethod
    def combine_robust(
        masses: list[tuple[float, float, float]],
        weights: list[float] | None = None,
    ) -> tuple[float, float, float]:
        """Smets 折扣规则: m'(A) = w·m(A), m'(Theta) = w·m(Theta) + (1-w).

        高冲突 (K > 0.5) 时 Dempster 规则放大 m_pass, 用折扣规则降权
        不可信源更稳. weights=None 时等价于 combine (w=1.0).
        算法复用 evidence_fusion_tool._weighted_combine 的 Smets 实现,
        避免两套独立 DS 实现逻辑漂移.
        """
        if not masses:
            return (0.0, 0.0, 1.0)
        if weights is None:
            weights = [1.0] * len(masses)
        # 折扣每个 mass
        discounted: list[tuple[float, float, float]] = []
        for (m_p, m_f, m_u), w in zip(masses, weights):
            if w >= 1.0 - 1e-9:
                discounted.append((m_p, m_f, m_u))
                continue
            # m'(pass) = w·m(pass), m'(fail) = w·m(fail), m'(unc) = w·m(unc) + (1-w)
            discounted.append((m_p * w, m_f * w, m_u * w + (1.0 - w)))
        return DempsterShaferCombiner.combine(discounted)


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
        # v11: Hard-veto 短路 — 守恒律违反 / 量纲不一致直接 reject, 绕过 DS 合成.
        # ponytail: 只对 verified == False 触发 (确定判负), verified == None / 缺失走原 DS.
        # 不新建 SMT 组件, 只路由已有 BourbakiTool / symbolic_math_tool 输出.
        # v12: constraint_check.violations 非空也走 hard-veto (强信号, 误判风险低).
        _conservation = evidence.get("conservation_law")
        if isinstance(_conservation, dict) and _conservation.get("verified") is False:
            return (
                False,
                "hard-veto: conservation law violated (BourbakiTool verified=False)",
                {"hard_veto": "conservation_law", "sources": ["conservation_law"]},
            )
        _dimensional = evidence.get("dimensional_consistent")
        if _dimensional is False:
            return (
                False,
                "hard-veto: dimensional inconsistency (symbolic_math_tool)",
                {"hard_veto": "dimensional_inconsistent", "sources": ["dimensional_consistent"]},
            )
        # v12: constraint_check.violations 非空 → hard-veto
        # ponytail: truthy 判定 (空 list / 空字符串 / None 都不触发), 复用已有 symbolic_regression_tool 输出.
        _constraint = evidence.get("constraint_check")
        if isinstance(_constraint, dict):
            _violations = _constraint.get("violations")
            if _violations:
                return (
                    False,
                    f"hard-veto: constraint violations ({len(_violations) if hasattr(_violations, '__len__') else '?'} 条): "
                    f"{str(_violations)[:120]}",
                    {"hard_veto": "constraint_violations", "sources": ["constraint_check"]},
                )

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

    缺证据默认走 advisory (warning + feedback, 不阻断), 避免 agent 绕开
    gate 而不补证据. 人工 checkpoint 仍硬阻断 — 由 human_checkpoint_phases
    显式配置或从 shared state 懒查.
    可选注入 reviewer_fn 做主观审查 (LLM 调用), 不传就只做硬性检查.
    可选注入 math_checker 做论文级 Dempster-Shafer 证据合成, 在硬性证据
    齐全后评估数学证据是否足够支撑阶段推进.
    """

    def __init__(
        self,
        config: PhaseGateConfig | None = None,
        reviewer_fn: ReviewerFn | None = None,
        math_checker: MathCheckerFn | None = None,
        human_checkpoint_phases: set[tuple[str, str]] | None = None,
    ):
        self.config = config or PhaseGateConfig()
        self._reviewer_fn = reviewer_fn
        self._math_checker = math_checker
        # None = 评估时懒查 shared state; 显式传 set 时以调用方为准 (测试用)
        self._human_checkpoint_phases = human_checkpoint_phases

    def _is_human_checkpoint(self, from_phase: str, to_phase: str) -> bool:
        """该转移是否需人工 checkpoint (硬阻断, advisory 不放行).

        未显式注入时懒查 shared state — engine 默认场景无需改 __init__ 签名.
        已被 override 的转移不算 checkpoint (用户已决策).
        """
        if self._human_checkpoint_phases is not None:
            return (from_phase, to_phase) in self._human_checkpoint_phases
        try:
            state = get_shared_phase_gate_state()
            return state.needs_human_checkpoint(from_phase, to_phase)
        except Exception:
            return False

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
            # 人工 checkpoint 仍硬阻断: 让 engine 走 hitl 流程, 用户必须 override
            # 才能放行. 缺证据时不能 advisory 跳过 — checkpoint 就是 checkpoint.
            if self._is_human_checkpoint(from_phase, to_phase):
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
            # advisory: 缺证据只 warning + feedback, 不阻断 transition.
            # ponytail: ceiling 是 agent 可能忽略 warning 继续推进, 升级路径:
            # PhaseGateState 累计同一 (from,to) advisory 反馈 N 次后强制回退到 blocked.
            logger.warning(
                "phase_gate advisory: %s→%s 缺证据 %s, 已放行 (非 checkpoint)",
                from_phase,
                to_phase,
                missing,
            )
            return PhaseGate(
                from_phase=from_phase,
                to_phase=to_phase,
                status="approved",
                required_evidence=required,
                missing_evidence=missing,
                feedback=(
                    f"advisory: {from_phase}→{to_phase} 缺少证据 {missing}, "
                    f"已有 keys: {list(evidence.keys())}. "
                    f"未阻断, 下游应补齐后再推进."
                ),
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

        # 物理 oracle 否决: simulator tool 把 PhysicsAuditor 结果填进 evidence
        # ["physics_audit"] (dict 含 has_errors bool). 有 error 直接 rejected,
        # 优先级高于 reviewer — 物理 plausibility 是 first-principles 不可妥协.
        # ponytail: 仅检查 has_errors, 不解析 findings. 升级: 按 severity 加权 + DS 合成.
        _pa = evidence.get("physics_audit")
        if isinstance(_pa, dict) and _pa.get("has_errors"):
            return PhaseGate(
                from_phase=from_phase,
                to_phase=to_phase,
                status="rejected",
                required_evidence=required,
                missing_evidence=[],
                feedback=(
                    "Physics oracle rejected: tool output contains physical "
                    "errors (AuditReport.has_errors=True). Fix the unphysical "
                    "values (e.g. negative band gap, non-converged SCF, "
                    "thermodynamic violation) before re-attempting the phase transition."
                ),
                reviewer="physics_oracle",
            )

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
            feedback=_argus_feedback(evidence),
            reviewer="argus_provenance" if _has_external_source(evidence) else None,
        )


def _argus_feedback(evidence: dict[str, Any]) -> str:
    """ARGUS 降级提示: evidence dominant=external_content 或 confidence<0.5 时附加.

    不阻断 (status 仍 approved), 只提示下游 agent 注意来源可信度.
    ponytail: 用 Dempster-Shafer 合成各 source_class 的先验 mass, 输出 m_pass 作 confidence.
    升级: 从历史任务拟合 _SOURCE_CLASS_MASSES 的先验值 (当前为经验值).
    """
    confidence, dominant = _argus_confidence(evidence)
    if not dominant:
        return ""
    # 只在 dominant=external_content 或 confidence < 0.5 时返回 feedback
    # (user_input / tool_output dominant 时不加噪声)
    if dominant != "external_content" and confidence >= 0.5:
        return ""
    return (
        f"ARGUS provenance: dominant source_class={dominant}, "
        f"DS confidence={confidence:.3f}. "
        f"建议补充 user_input 或 tool_output 来源的独立验证."
    )


# ── 共享状态: 连接 engine 与 PhaseTool ──────────────────────────


class PhaseGateState:
    """进程内共享的 phase-gate 运行时状态.

    engine 写 (pending_transition / history), PhaseTool 读+写
    (submit_evidence / override). 单例通过 get_shared_phase_gate_state 拿.

    human_checkpoint_phases: 配置哪些 (from, to) 转移需要人工 checkpoint.
        命中时 engine 不自动推进, 而是把 pending_human_review 设为该转移,
        yield phase_checkpoint 事件给 UI 层. 用户通过 override 或
        submit_evidence + resume 推进. 借鉴 LangGraph interrupt_before 模式.
    pending_human_review: 当前等待人工审查的转移, None 表示无.
    """

    def __init__(self) -> None:
        self.history: list[PhaseGate] = []
        self.pending_transition: tuple[str, str] | None = None
        self.submitted_evidence: dict[str, Any] = {}
        self.overrides: set[tuple[str, str]] = set()
        # override 元数据 (并行于 overrides set): 谁/何时/为何 override
        # ponytail: 并行结构, set.add 不写 meta 则 meta 缺. 升级: set→dict 合并
        self.override_meta: dict[tuple[str, str], dict] = {}
        # Human-in-the-loop checkpoint 配置 + 运行时状态
        # 从 HUGINN_HUMAN_CHECKPOINT_PHASEES=plan:execute,validate:learn 加载
        self.human_checkpoint_phases: set[tuple[str, str]] = _load_checkpoint_config()
        # 硬性 checkpoint: 不可超时自动放行, 一直阻塞到用户显式 override
        self.hard_checkpoint_phases: set[tuple[str, str]] = _load_hard_checkpoint_config()
        self.pending_human_review: tuple[str, str] | None = None

    def reset(self) -> None:
        self.history.clear()
        self.pending_transition = None
        self.submitted_evidence.clear()
        self.overrides.clear()
        self.override_meta.clear()
        self.pending_human_review = None

    def reset_runtime(self) -> None:
        """只清每轮瞬态 (history / pending / submitted_evidence),
        保留 caller 配置的 overrides 和 override_meta.

        engine.run_cognitive() 开头调这个, 不用 reset() — reset() 会清掉 caller
        在 run 之前预设的 override, 破坏 "用户预先放行某转移" 的用法.
        """
        self.history.clear()
        self.pending_transition = None
        self.submitted_evidence.clear()
        self.pending_human_review = None

    def last_gate(self) -> PhaseGate | None:
        return self.history[-1] if self.history else None

    def needs_human_checkpoint(self, from_phase: str, to_phase: str) -> bool:
        """该转移是否配置了人工 checkpoint (未被 override 覆盖).

        硬性 checkpoint 也算人工 checkpoint — 只是阻塞强度不同 (见 is_hard_checkpoint).
        """
        if (from_phase, to_phase) in self.overrides:
            return False
        key = (from_phase, to_phase)
        return key in self.human_checkpoint_phases or key in self.hard_checkpoint_phases

    def is_hard_checkpoint(self, from_phase: str, to_phase: str) -> bool:
        """该转移是否配置为硬性 checkpoint (不可超时自动放行)."""
        return (from_phase, to_phase) in self.hard_checkpoint_phases


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
