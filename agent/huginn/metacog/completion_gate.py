"""CompletionGate — 三审放行门.

把散在 engine reflect_fn L1859-1915 (F2 completion audit + F17 GoalJudge)
和 run() L1878-1945 双写的三处顺序拼装, 收敛成一个 review() 调用.

三审顺序:
- 审1 Criteria (规则): GoalScheduler.check_completion 命中 → 进审2
- 审2 Metacog (Auditor): CompletionAuditor 反完成审计, 命中 → block
- 审3 GoalJudge (LLM/规则): 每 N 轮 + 末轮, achieved → 进审2, gaps → hint

run() 那边留 AV1, 本版只接 reflect_fn. 失败 fallback 到原 F2+F17 顺序拼装,
由 engine 调用方根据 flag 自行决定.

flag: HUGINN_USE_COMPLETION_GATE=1 开启, 默认 off.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class GateDecision:
    """三审放行门决策."""

    status: str  # pass / block / gaps_hint / pending
    reason: str = ""  # 阻断原因或 gaps 文本
    category: str = ""  # criteria_match / metacog_block / goal_judge_achieved / goal_judge_gaps / no_criteria
    should_stop: bool = False  # 终极停机信号
    should_complete_goal: bool = False  # 标 goal.status=completed


@dataclass
class GateContext:
    """review 需要的状态快照, 调用方从 engine self 上凑."""

    iteration: int
    max_iterations: int
    families_explored: int
    live_components: int
    last_raw_hypothesis: str = ""
    objective: str = ""


class CompletionGate:
    """三审放行门 — 编排 Criteria + Metacog + GoalJudge.

    auditor_factory: 返回 CompletionAuditor 的 callable, 懒构造复用 engine 配置.
    goal_judge_llm: GoalJudge 用的 LLM, None 走规则版 (对齐 F17 现状).
    judge_every_n: GoalJudge 调用频率, 默认 3 (对齐 F17).
    """

    def __init__(
        self,
        auditor_factory: Callable[[], Any] | None = None,
        goal_judge_llm: Any | None = None,
        judge_every_n: int = 3,
    ) -> None:
        self._auditor_factory = auditor_factory
        self._goal_judge_llm = goal_judge_llm
        self._judge_every_n = judge_every_n

    def review(
        self,
        goal: Any | None,
        validation: Any,
        ctx: GateContext,
    ) -> GateDecision:
        """主入口. goal=None 或无 criteria → 直接 pass (no_criteria)."""
        if goal is None:
            return GateDecision(status="pass", category="no_goal")

        # 审1: Criteria (规则) — GoalScheduler.check_completion
        criteria_hit = False
        try:
            from huginn.autoloop.goal_scheduler import GoalScheduler
            criteria_hit = GoalScheduler.check_completion(goal, validation)
        except Exception:
            logger.debug("审1 Criteria failed", exc_info=True)
            criteria_hit = False

        # 审3 时机: criteria 命中 → 必跑; miss → 每 N 轮 + 末轮
        is_judge_round = (
            criteria_hit
            or ctx.iteration % self._judge_every_n == self._judge_every_n - 1
            or ctx.iteration >= ctx.max_iterations - 1
        )

        if criteria_hit:
            # 审1 命中 → 进审2
            return self._run_metacog(ctx, category_on_pass="criteria_match")

        if not is_judge_round:
            # 审1 miss + 非 judge 轮 → 啥都不做
            return GateDecision(status="pass", category="no_criteria")

        # 审3: GoalJudge (LLM/规则)
        try:
            from huginn.evaluation.goal_judge import GoalJudge
            _judge = GoalJudge(llm=self._goal_judge_llm)
            _final_text = str(
                (validation or {}).get("summary")
                or (validation or {}).get("result_data")
                or ""
            )
            _gj = _judge.judge(
                getattr(goal, "objective", "") or "",
                None,
                _final_text,
            )
        except Exception:
            logger.debug("审3 GoalJudge failed", exc_info=True)
            return GateDecision(status="pass", category="goal_judge_failed")

        if _gj.get("achieved"):
            # achieved → 进审2
            return self._run_metacog(ctx, category_on_pass="goal_judge_achieved")

        # gaps → hint, 不停机
        gaps = _gj.get("gaps") or []
        if gaps:
            _gap_hint = "; ".join(str(g) for g in gaps[:3])
            return GateDecision(
                status="gaps_hint",
                reason=_gap_hint,
                category="goal_judge_gaps",
            )

        # achieved=False 无 gaps — 罕见, 不停
        return GateDecision(status="pass", category="goal_judge_not_achieved")

    def _run_metacog(
        self,
        ctx: GateContext,
        *,
        category_on_pass: str,
    ) -> GateDecision:
        """审2: CompletionAuditor 反完成审计. 出错放行 (advisory)."""
        try:
            if self._auditor_factory is None:
                # 无 factory — 默认放行, 留升级路径
                return GateDecision(
                    status="pass",
                    category=category_on_pass,
                    should_stop=True,
                    should_complete_goal=True,
                )
            auditor = self._auditor_factory()
            # 从 raw hypothesis 提 UNEXPLORED: 块
            unexplored = ""
            raw = ctx.last_raw_hypothesis or ""
            if "UNEXPLORED:" in raw:
                unexplored = raw.split("UNEXPLORED:", 1)[1].strip()
                for marker in ["\n\nHYPOTHESIS", "\n\nSELECTED", "\n\nRATIONALE"]:
                    if marker in unexplored:
                        unexplored = unexplored.split(marker)[0].strip()
                        break
            checklist = auditor.audit(
                iteration=ctx.iteration,
                families_explored=ctx.families_explored,
                live_components=ctx.live_components,
                total_iterations=ctx.max_iterations,
                candidate_finding="",  # ctx 里没有, 升级路径加
                original_problem=ctx.objective,
                unexplored_declaration=unexplored,
            )
            if checklist.is_complete:
                return GateDecision(
                    status="pass",
                    category=category_on_pass,
                    should_stop=True,
                    should_complete_goal=True,
                )
            return GateDecision(
                status="block",
                reason=checklist.block_reason(),
                category="metacog_block",
            )
        except Exception:
            logger.debug("审2 Metacog failed, advisory pass", exc_info=True)
            return GateDecision(
                status="pass",
                category=category_on_pass,
                should_stop=True,
                should_complete_goal=True,
            )


# === 自检 ===

if __name__ == "__main__":
    from dataclasses import dataclass as _dc

    @_dc
    class _Goal:
        objective: str = "predict X"
        success_criteria: list = field(default_factory=lambda: ["X"])
        status: str = "active"
        metadata: dict = field(default_factory=dict)

    # 1) goal=None → pass, no_goal
    gate = CompletionGate()
    d = gate.review(None, {}, GateContext(iteration=1, max_iterations=10, families_explored=1, live_components=1))
    assert d.status == "pass" and d.category == "no_goal"
    assert not d.should_stop and not d.should_complete_goal

    # 2) criteria 命中 + 无 auditor_factory → pass + complete + stop
    g = _Goal(success_criteria=["X"])
    val = {"summary": "X achieved"}
    d = gate.review(g, val, GateContext(iteration=5, max_iterations=10, families_explored=2, live_components=2))
    assert d.status == "pass", f"criteria hit → pass, got {d.status}"
    assert d.category == "criteria_match"
    assert d.should_complete_goal and d.should_stop

    # 3) criteria miss + 非 judge 轮 → pass, no_criteria
    g2 = _Goal(success_criteria=["Z"])  # Z 不在 val 里
    d = gate.review(g2, val, GateContext(iteration=1, max_iterations=10, families_explored=1, live_components=1))
    assert d.status == "pass" and d.category == "no_criteria"
    assert not d.should_stop

    # 4) criteria miss + 末轮 (iteration >= max-1) → GoalJudge 规则版
    #    final_text 不含 objective 关键词 → gaps_hint 或 not_achieved
    d = gate.review(
        g2, {"summary": "unrelated output"},
        GateContext(iteration=9, max_iterations=10, families_explored=1, live_components=1, objective="predict Z"),
    )
    assert d.category in ("goal_judge_gaps", "goal_judge_not_achieved"), f"末轮 judge → gaps/not_achieved, got {d.category}"
    assert not d.should_stop, "not achieved 不应停机"

    # 5) criteria miss + 每 3 轮 (iteration % 3 == 2) → GoalJudge 触发
    d = gate.review(
        g2, {"summary": "unrelated"},
        GateContext(iteration=2, max_iterations=10, families_explored=1, live_components=1, objective="predict Z"),
    )
    # iteration=2, judge_every_n=3 → 2 % 3 == 2 → is_judge_round
    assert d.category in ("goal_judge_gaps", "goal_judge_not_achieved"), f"每3轮 judge, got {d.category}"

    # 6) GoalJudge achieved + 无 auditor_factory → pass + complete + stop
    # success_criteria=["QQQ"] 不命中, 但 GoalJudge 规则版 achieved:
    # objective 关键词在 final_output 里且 len>20
    long_text = "predict ZZZ with detailed analysis and computation results here"
    g4 = _Goal(success_criteria=["QQQ"], objective="predict ZZZ")
    d = gate.review(
        g4, {"summary": long_text},
        GateContext(iteration=9, max_iterations=10, families_explored=1, live_components=1, objective="predict ZZZ"),
    )
    assert d.category == "goal_judge_achieved", f"achieved + 无 auditor → pass+complete, got {d.category}"
    assert d.should_complete_goal and d.should_stop

    # 7) 有 auditor_factory + is_complete=False → block
    @_dc
    class _Checklist:
        is_complete: bool = False
        def block_reason(self) -> str:
            return "effort low"
    class _MockAuditor:
        def audit(self, **kw):
            return _Checklist(is_complete=False)
    gate2 = CompletionGate(auditor_factory=lambda: _MockAuditor())
    d = gate2.review(
        _Goal(success_criteria=["X"]), {"summary": "X done"},
        GateContext(iteration=5, max_iterations=10, families_explored=1, live_components=1),
    )
    assert d.status == "block" and d.category == "metacog_block"
    assert "effort low" in d.reason
    assert not d.should_stop and not d.should_complete_goal

    # 8) 有 auditor_factory + is_complete=True → pass + complete + stop
    class _MockAuditor2:
        def audit(self, **kw):
            return _Checklist(is_complete=True)
    gate3 = CompletionGate(auditor_factory=lambda: _MockAuditor2())
    d = gate3.review(
        _Goal(success_criteria=["X"]), {"summary": "X done"},
        GateContext(iteration=5, max_iterations=10, families_explored=2, live_components=2),
    )
    assert d.status == "pass" and d.category == "criteria_match"
    assert d.should_complete_goal and d.should_stop

    print("all self-checks passed")
