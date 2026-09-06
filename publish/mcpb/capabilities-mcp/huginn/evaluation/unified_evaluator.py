"""UnifiedEvaluator — 评估逻辑统一接口.

把分散在 4 个包的评估器结果聚合成统一的决策信号:
  - evaluation/core.py        : MCDA (AHP/TOPSIS/VIKOR) → EvaluationResult
  - evaluation/goal_judge.py  : GoalJudge 纵向判定 → {achieved, score, evidence, gaps}
  - validation/grader.py      : 8 种 Grader → GraderResult
  - metacog/step_evaluator.py : StepEvaluator → StepEvaluation

四套结果格式互不兼容, UnifiedEvaluationResult 把它们归一为统一字段:
  score (0-1) / achieved (bool) / evidence (list[str]) / gaps (list[str]) /
  category (str) / source (str).

设计原则:
  - 所有适配方法和 evaluate 分支都用 try/except 保护, 失败返回默认结果, 不抛异常.
  - 子评估器按需懒导入, 避免 numpy/langchain 等重依赖在模块加载时被拉起.
  - 适配方法用 duck-typing (getattr / dict.get), 兼容 dataclass 和 dict 两种形态.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# 聚合默认门槛: 加权 score >= 该值视为 achieved
_DEFAULT_THRESHOLD = 0.5

# on_track 字符串 → score 的映射 (StepEvaluator 用)
_ON_TRACK_SCORE: dict[str, float] = {"true": 1.0, "unsure": 0.5, "false": 0.0}


@dataclass
class UnifiedEvaluationResult:
    """统一评估结果. 所有子评估器结果都归一到这里.

    Attributes:
        score: 综合得分, 范围 [0, 1], 越高越好.
        achieved: 是否达到目标 / 通过门槛.
        evidence: 支持达成的证据条目列表.
        gaps: 未达成项 / 缺口列表.
        category: 评估类别 (mcda / goal_judge / grader / step / unified).
        source: 来源评估器名 (如 GoalJudge / physics / entropy-topsis).
    """

    score: float = 0.0
    achieved: bool = False
    evidence: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    category: str = "unified"
    source: str = "UnifiedEvaluator"

    def __post_init__(self) -> None:
        # 钳到 [0, 1], 防止子评估器越界污染聚合
        if self.score < 0.0:
            self.score = 0.0
        elif self.score > 1.0:
            self.score = 1.0


class UnifiedEvaluator:
    """统一评估器: 内部调用各子评估器并聚合成统一决策信号.

    持有可选的子评估器实例 (GoalJudge / GraderRegistry). evaluate() 根据
    context 里提供的输入调用对应子评估器, 把结果转成 UnifiedEvaluationResult,
    再加权聚合成一个最终结果. 任一子评估器失败都不影响其它分支.

    适配方法 from_*() 是静态方法, 可独立调用, 把单个子结果转成统一格式.
    """

    def __init__(
        self,
        goal_judge: Any | None = None,
        grader_registry: Any | None = None,
        llm: Any | None = None,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        # goal_judge: 可传入已配置的 GoalJudge 实例; None 时按需构造
        self._goal_judge = goal_judge
        # grader_registry: 可传入已注册的 GraderRegistry; None 时按需构造
        self._grader_registry = grader_registry
        # llm: 可选, 传给 GoalJudge / ValidityJudge 做 LLM 评估
        self._llm = llm
        # 聚合 achieved 门槛
        self._threshold = threshold

    # ── 适配方法: 子结果 → UnifiedEvaluationResult ───────────────

    @staticmethod
    def from_mcda(result: Any, alternative: str | None = None) -> UnifiedEvaluationResult:
        """从 MCDA EvaluationResult 转换.

        取指定 alternative 的分; 未指定时取 ranking 第一名; 都没有则取首个 score.

        Args:
            result: EvaluationResult (有 scores/ranking/method 属性) 或等价 dict.
            alternative: 指定要取分的备选项名; None 时自动选排名第一.

        Returns:
            归一后的 UnifiedEvaluationResult, category="mcda".
        """
        try:
            if isinstance(result, dict):
                scores = result.get("scores", {}) or {}
                ranking = result.get("ranking", []) or []
                method = result.get("method", "mcda")
            else:
                scores = getattr(result, "scores", None) or {}
                ranking = getattr(result, "ranking", []) or []
                method = getattr(result, "method", "mcda")

            # 选 alternative: 显式指定 > 排名第一 > 首个 score 键
            alt = alternative
            if alt is None and ranking:
                alt = ranking[0]
            if alt is None and scores:
                alt = next(iter(scores), None)

            score = float(scores.get(alt, 0.0)) if alt is not None else 0.0
            achieved = score >= _DEFAULT_THRESHOLD
            ranking_str = " > ".join(ranking) if ranking else "(空)"
            evidence = [f"method={method}", f"alternative={alt}", f"ranking={ranking_str}"]
            gaps = (
                []
                if achieved
                else [f"{alt} score {score:.3f} 低于门槛 {_DEFAULT_THRESHOLD}"]
            )
            return UnifiedEvaluationResult(
                score=score,
                achieved=achieved,
                evidence=evidence,
                gaps=gaps,
                category="mcda",
                source=method,
            )
        except Exception as exc:
            logger.warning("from_mcda 转换失败: %s", exc)
            return UnifiedEvaluationResult(category="mcda", source="mcda_error")

    @staticmethod
    def from_goal_judge(result: Any) -> UnifiedEvaluationResult:
        """从 GoalJudge.judge() 返回结果转换.

        期望字段: {achieved: bool, score: float, evidence: list, gaps: list}.

        Args:
            result: GoalJudge 返回的 dict, 或含同等属性的对象.

        Returns:
            归一后的 UnifiedEvaluationResult, category="goal_judge", source="GoalJudge".
        """
        try:
            if isinstance(result, dict):
                achieved = bool(result.get("achieved", False))
                score = float(result.get("score", 0.0))
                evidence = list(result.get("evidence", []) or [])
                gaps = list(result.get("gaps", []) or [])
            else:
                achieved = bool(getattr(result, "achieved", False))
                score = float(getattr(result, "score", 0.0))
                evidence = list(getattr(result, "evidence", []) or [])
                gaps = list(getattr(result, "gaps", []) or [])
            return UnifiedEvaluationResult(
                score=score,
                achieved=achieved,
                evidence=[str(e) for e in evidence],
                gaps=[str(g) for g in gaps],
                category="goal_judge",
                source="GoalJudge",
            )
        except Exception as exc:
            logger.warning("from_goal_judge 转换失败: %s", exc)
            return UnifiedEvaluationResult(
                category="goal_judge", source="goal_judge_error"
            )

    @staticmethod
    def from_grader(result: Any) -> UnifiedEvaluationResult:
        """从 GraderResult 转换.

        期望字段: name (str), score (0-1), passed (bool), checks (list[dict]),
        message (str). 兼容 8 种 Grader (physics/dimensional/red_team/
        hallucination/matworld_bench/literature/validity/materials_bounds).

        Args:
            result: GraderResult 对象或等价 dict.

        Returns:
            归一后的 UnifiedEvaluationResult, category="grader", source=grader.name.
        """
        try:
            if isinstance(result, dict):
                name = result.get("name", "grader")
                score = float(result.get("score", 0.0))
                passed = bool(result.get("passed", False))
                checks = result.get("checks", []) or []
                message = result.get("message", "") or ""
            else:
                name = getattr(result, "name", "grader")
                score = float(getattr(result, "score", 0.0))
                passed = bool(getattr(result, "passed", False))
                checks = getattr(result, "checks", []) or []
                message = getattr(result, "message", "") or ""

            evidence = [f"grader={name} score={score:.3f}"]
            if message:
                evidence.append(message)
            # 把 checks 里的关键字段也作为证据
            for chk in checks:
                if isinstance(chk, dict):
                    detail = (
                        chk.get("issue")
                        or chk.get("violation")
                        or chk.get("description")
                        or chk.get("reason")
                        or ""
                    )
                    if detail:
                        evidence.append(str(detail))
                elif chk:
                    evidence.append(str(chk))

            if passed:
                gaps = []
            else:
                gaps = [f"{name} 未通过: {message}"] if message else [f"{name} 未通过"]

            return UnifiedEvaluationResult(
                score=score,
                achieved=passed,
                evidence=evidence,
                gaps=gaps,
                category="grader",
                source=name,
            )
        except Exception as exc:
            logger.warning("from_grader 转换失败: %s", exc)
            return UnifiedEvaluationResult(category="grader", source="grader_error")

    @staticmethod
    def from_step_evaluator(result: Any) -> UnifiedEvaluationResult:
        """从 StepEvaluation 转换.

        期望字段: step_id, on_track (true/false/unsure), structure_check
        (passed/failed/soft_warning/not_applicable), evidence_quality
        (high/medium/low/unknown), deviation (str), target_chain_ref (str|None).

        score 由 on_track 映射: true=1.0 / unsure=0.5 / false=0.0.
        achieved = (on_track == "true").

        Args:
            result: StepEvaluation 对象或含 on_track 键的 dict.

        Returns:
            归一后的 UnifiedEvaluationResult, category="step", source="StepEvaluator".
        """
        try:
            if isinstance(result, dict):
                step_id = result.get("step_id", -1)
                on_track = str(result.get("on_track", "unsure")).lower()
                struct = result.get("structure_check", "not_applicable")
                eq = result.get("evidence_quality", "unknown")
                deviation = result.get("deviation", "") or ""
                target_ref = result.get("target_chain_ref")
            else:
                step_id = getattr(result, "step_id", -1)
                on_track = str(getattr(result, "on_track", "unsure")).lower()
                struct = getattr(result, "structure_check", "not_applicable")
                eq = getattr(result, "evidence_quality", "unknown")
                deviation = getattr(result, "deviation", "") or ""
                target_ref = getattr(result, "target_chain_ref", None)

            score = _ON_TRACK_SCORE.get(on_track, 0.5)
            achieved = on_track == "true"

            evidence = [
                f"step={step_id} on_track={on_track}",
                f"structure_check={struct} evidence_quality={eq}",
            ]
            if target_ref:
                evidence.append(f"target_chain={target_ref}")

            gaps: list[str] = []
            if deviation:
                gaps.append(str(deviation))
            if struct == "failed":
                gaps.append("结构不变量检查失败")
            if eq == "low":
                gaps.append("证据质量低")

            return UnifiedEvaluationResult(
                score=score,
                achieved=achieved,
                evidence=evidence,
                gaps=gaps,
                category="step",
                source="StepEvaluator",
            )
        except Exception as exc:
            logger.warning("from_step_evaluator 转换失败: %s", exc)
            return UnifiedEvaluationResult(
                category="step", source="step_error"
            )

    # ── 聚合主入口 ───────────────────────────────────────────────

    def evaluate(self, context: dict[str, Any]) -> UnifiedEvaluationResult:
        """聚合评估: 内部调用各子评估器并聚合成统一决策信号.

        context 可包含以下键 (都可选, 缺哪个跳过哪个, 失败返回默认不抛):
          - mcda: EvaluationResult, 或 MCDA 输入 dict
                  {alternatives, criteria, matrix, directions, weight_method,
                   eval_method, ahp_matrix, eval_kwargs, alternative}
          - goal_judge: GoalJudge 结果 dict {achieved, score, evidence, gaps}
                  或输入 dict {objective, trajectory, final_output}
          - grader: GraderResult / list[GraderResult] / GraderResult 等价 dict
                  / grader data dict (传给 GraderRegistry.evaluate_all)
          - step: StepEvaluation / list[StepEvaluation] / 含 on_track 的 dict
                  / step 评估输入 dict {meta_trace_entry, target_chains, ...}

        Args:
            context: 各子评估器的输入或预计算结果.

        Returns:
            聚合后的 UnifiedEvaluationResult. 无任何可用子结果时返回默认值.
        """
        sub_results: list[UnifiedEvaluationResult] = []

        # MCDA 分支
        mcda_ctx = context.get("mcda")
        if mcda_ctx is not None:
            res = self._eval_mcda(mcda_ctx)
            if res is not None:
                sub_results.append(res)

        # GoalJudge 分支
        gj_ctx = context.get("goal_judge")
        if gj_ctx is not None:
            res = self._eval_goal_judge(gj_ctx)
            if res is not None:
                sub_results.append(res)

        # Grader 分支
        grader_ctx = context.get("grader")
        if grader_ctx is not None:
            res = self._eval_grader(grader_ctx)
            if res is not None:
                sub_results.append(res)

        # Step 分支
        step_ctx = context.get("step")
        if step_ctx is not None:
            res = self._eval_step(step_ctx)
            if res is not None:
                sub_results.append(res)

        if not sub_results:
            return self._default_result()

        return self._aggregate(sub_results)

    # ── 各子评估器调用 (内部, try/except 保护) ────────────────────

    def _eval_mcda(self, ctx: Any) -> UnifiedEvaluationResult | None:
        """调用 MCDA 评估器. ctx 可以是 EvaluationResult 或 MCDA 输入 dict."""
        try:
            # 已经是 EvaluationResult (duck-typing: 有 scores/ranking 属性)
            if hasattr(ctx, "scores") and hasattr(ctx, "ranking"):
                return self.from_mcda(ctx)
            # dict 输入: 调 evaluation.core.evaluate
            if isinstance(ctx, dict):
                from huginn.evaluation.core import evaluate as mcda_evaluate

                result = mcda_evaluate(
                    alternatives=ctx["alternatives"],
                    criteria=ctx["criteria"],
                    matrix=ctx["matrix"],
                    directions=ctx.get("directions"),
                    weight_method=ctx.get("weight_method", "entropy"),
                    eval_method=ctx.get("eval_method", "topsis"),
                    ahp_matrix=ctx.get("ahp_matrix"),
                    eval_kwargs=ctx.get("eval_kwargs"),
                )
                return self.from_mcda(result, alternative=ctx.get("alternative"))
            logger.debug("_eval_mcda: 不支持的 ctx 类型 %s", type(ctx).__name__)
            return None
        except Exception as exc:
            logger.warning("_eval_mcda 失败: %s", exc)
            return UnifiedEvaluationResult(category="mcda", source="mcda_error")

    def _eval_goal_judge(self, ctx: Any) -> UnifiedEvaluationResult | None:
        """调用 GoalJudge. ctx 可以是 judge 结果 dict 或输入 {objective, ...}."""
        try:
            # 已经是 judge 结果 (含 achieved/score 键, 且不是输入 dict)
            if (
                isinstance(ctx, dict)
                and ("achieved" in ctx or "score" in ctx)
                and "objective" not in ctx
            ):
                return self.from_goal_judge(ctx)
            # 输入 dict: 调 GoalJudge.judge
            if isinstance(ctx, dict):
                judge = self._goal_judge
                if judge is None:
                    from huginn.evaluation.goal_judge import GoalJudge

                    judge = GoalJudge(llm=self._llm)
                result = judge.judge(
                    objective=ctx.get("objective", ""),
                    trajectory=ctx.get("trajectory"),
                    final_output=ctx.get("final_output", ""),
                )
                return self.from_goal_judge(result)
            # duck-typing: 有 achieved 属性的对象
            if hasattr(ctx, "achieved"):
                return self.from_goal_judge(ctx)
            logger.debug("_eval_goal_judge: 不支持的 ctx 类型 %s", type(ctx).__name__)
            return None
        except Exception as exc:
            logger.warning("_eval_goal_judge 失败: %s", exc)
            return UnifiedEvaluationResult(
                category="goal_judge", source="goal_judge_error"
            )

    def _eval_grader(self, ctx: Any) -> UnifiedEvaluationResult | None:
        """调用 Grader. ctx 可以是 GraderResult / list / data dict."""
        try:
            # 单个 GraderResult (duck-typing: 有 name/score/passed 属性)
            if (
                hasattr(ctx, "name")
                and hasattr(ctx, "score")
                and hasattr(ctx, "passed")
            ):
                return self.from_grader(ctx)
            # list[GraderResult]: 逐个转换后聚合
            if isinstance(ctx, (list, tuple)) and ctx:
                results = [self.from_grader(g) for g in ctx]
                return self._aggregate(results)
            # dict: GraderResult 的 dict 形式 or grader data dict
            if isinstance(ctx, dict):
                # 看起来是 GraderResult 序列化 (含 name+score+passed)
                if "name" in ctx and "score" in ctx and "passed" in ctx:
                    return self.from_grader(ctx)
                # 否则当作 grader data 调 registry
                registry = self._grader_registry
                if registry is None:
                    from huginn.validation.grader import default_registry

                    registry = default_registry(model=self._llm)
                grader_results = registry.evaluate_all(ctx)
                if not grader_results:
                    return None
                unified = [self.from_grader(g) for g in grader_results]
                return self._aggregate(unified)
            logger.debug("_eval_grader: 不支持的 ctx 类型 %s", type(ctx).__name__)
            return None
        except Exception as exc:
            logger.warning("_eval_grader 失败: %s", exc)
            return UnifiedEvaluationResult(category="grader", source="grader_error")

    def _eval_step(self, ctx: Any) -> UnifiedEvaluationResult | None:
        """调用 StepEvaluator. ctx 可以是 StepEvaluation / list / 输入 dict."""
        try:
            # 单个 StepEvaluation (duck-typing: 有 on_track 属性)
            if hasattr(ctx, "on_track"):
                return self.from_step_evaluator(ctx)
            # list[StepEvaluation]: 逐个转换后聚合
            if isinstance(ctx, (list, tuple)) and ctx:
                results = [self.from_step_evaluator(s) for s in ctx]
                return self._aggregate(results)
            # dict: 已是评估结果 (含 on_track) 或 step 评估输入
            if isinstance(ctx, dict):
                if "on_track" in ctx:
                    return self.from_step_evaluator(ctx)
                # 输入: {meta_trace_entry, target_chains, ...}
                from huginn.metacog.step_evaluator import evaluate_step

                result = evaluate_step(
                    meta_trace_entry=ctx.get("meta_trace_entry") or ctx,
                    target_chains=ctx.get("target_chains") or [],
                    verification_signals=ctx.get("verification_signals"),
                    memory=ctx.get("memory"),
                    kb=ctx.get("kb"),
                    persona=ctx.get("persona"),
                    model=ctx.get("model") or self._llm,
                )
                return self.from_step_evaluator(result)
            logger.debug("_eval_step: 不支持的 ctx 类型 %s", type(ctx).__name__)
            return None
        except Exception as exc:
            logger.warning("_eval_step 失败: %s", exc)
            return UnifiedEvaluationResult(category="step", source="step_error")

    # ── 聚合 / 默认值 ───────────────────────────────────────────

    def _aggregate(
        self, results: list[UnifiedEvaluationResult]
    ) -> UnifiedEvaluationResult:
        """加权聚合多个子结果. 等权平均 score, achieved 用门槛+多数投票."""
        if not results:
            return self._default_result()
        n = len(results)
        # 等权平均 score
        score = sum(r.score for r in results) / n
        # achieved: 加权分超门槛, 或多数 achieved (含等票)
        n_achieved = sum(1 for r in results if r.achieved)
        majority = n_achieved * 2 >= n
        achieved = score >= self._threshold or majority
        # 合并 evidence / gaps, 带来源前缀避免混淆
        evidence = [f"[{r.source}] {e}" for r in results for e in r.evidence]
        gaps = [f"[{r.source}] {g}" for r in results for g in r.gaps]
        sources = [r.source for r in results]
        return UnifiedEvaluationResult(
            score=score,
            achieved=achieved,
            evidence=evidence,
            gaps=gaps,
            category="unified",
            source=",".join(sources),
        )

    def _default_result(self) -> UnifiedEvaluationResult:
        """无任何子结果时的默认值."""
        return UnifiedEvaluationResult(
            score=0.0,
            achieved=False,
            evidence=["无可用评估结果"],
            gaps=["未提供任何评估输入"],
            category="unified",
            source="default",
        )


__all__ = ["UnifiedEvaluationResult", "UnifiedEvaluator"]


# ── 自检 ───────────────────────────────────────────────────────

if __name__ == "__main__":
    from types import SimpleNamespace

    # 1) UnifiedEvaluationResult 钳值与默认字段
    _r = UnifiedEvaluationResult(score=1.5, achieved=True)
    assert _r.score == 1.0, f"clamp high 失败: {_r.score}"
    _r = UnifiedEvaluationResult(score=-0.5)
    assert _r.score == 0.0, f"clamp low 失败: {_r.score}"
    assert _r.evidence == [] and _r.gaps == [], "默认 list 应为空"

    # 2) from_mcda — mock EvaluationResult
    _mcda = SimpleNamespace(
        method="entropy-topsis",
        scores={"A": 0.8, "B": 0.3},
        ranking=["A", "B"],
        weights={"c1": 0.5},
    )
    _u = UnifiedEvaluator.from_mcda(_mcda)
    assert _u.score == 0.8, f"mcda score 失败: {_u.score}"
    assert _u.achieved is True, f"mcda achieved 失败: {_u.achieved}"
    assert _u.category == "mcda" and _u.source == "entropy-topsis"
    # 指定 alternative
    _u2 = UnifiedEvaluator.from_mcda(_mcda, alternative="B")
    assert _u2.score == 0.3 and _u2.achieved is False, f"mcda B 失败: {_u2.score}"
    # None 输入不抛 (返回零分结果)
    _u3 = UnifiedEvaluator.from_mcda(None)
    assert _u3.score == 0.0 and _u3.achieved is False

    # 3) from_goal_judge — dict 形式
    _gj = {
        "achieved": True,
        "score": 0.85,
        "evidence": ["数值合理"],
        "gaps": [],
        "reasoning": "ok",
    }
    _u = UnifiedEvaluator.from_goal_judge(_gj)
    assert _u.score == 0.85 and _u.achieved is True
    assert _u.evidence == ["数值合理"]
    assert _u.category == "goal_judge" and _u.source == "GoalJudge"
    # 对象形式
    _gj_obj = SimpleNamespace(achieved=False, score=0.2, evidence=[], gaps=["无机制解释"])
    _u = UnifiedEvaluator.from_goal_judge(_gj_obj)
    assert _u.achieved is False and _u.gaps == ["无机制解释"]

    # 4) from_grader — GraderResult mock
    _gr = SimpleNamespace(
        name="physics", score=0.6, passed=True,
        checks=[{"issue": "ok"}], message="no findings",
    )
    _u = UnifiedEvaluator.from_grader(_gr)
    assert _u.score == 0.6 and _u.achieved is True
    assert _u.source == "physics" and _u.category == "grader"
    assert any("no findings" in _e for _e in _u.evidence)
    # 未通过
    _gr2 = SimpleNamespace(name="dimensional", score=0.0, passed=False, checks=[], message="量纲不一致")
    _u = UnifiedEvaluator.from_grader(_gr2)
    assert _u.achieved is False and _u.gaps, "未通过应有 gaps"

    # 5) from_step_evaluator — StepEvaluation mock
    _step = SimpleNamespace(
        step_id=3, on_track="true", structure_check="passed",
        evidence_quality="high", deviation="", target_chain_ref="T1",
    )
    _u = UnifiedEvaluator.from_step_evaluator(_step)
    assert _u.score == 1.0 and _u.achieved is True
    assert _u.category == "step"
    assert any("step=3" in _e for _e in _u.evidence)
    # unsure
    _step2 = SimpleNamespace(
        step_id=4, on_track="unsure", structure_check="not_applicable",
        evidence_quality="unknown", deviation="机械信号不足", target_chain_ref=None,
    )
    _u = UnifiedEvaluator.from_step_evaluator(_step2)
    assert _u.score == 0.5 and _u.achieved is False
    assert "机械信号不足" in _u.gaps
    # false + 结构失败
    _step3 = SimpleNamespace(
        step_id=5, on_track="false", structure_check="failed",
        evidence_quality="low", deviation="脱轨", target_chain_ref=None,
    )
    _u = UnifiedEvaluator.from_step_evaluator(_step3)
    assert _u.score == 0.0 and _u.achieved is False
    assert "结构不变量检查失败" in _u.gaps
    assert "证据质量低" in _u.gaps

    # 6) UnifiedEvaluator.evaluate 聚合
    _ev = UnifiedEvaluator()
    _ctx = {
        "goal_judge": {"achieved": True, "score": 0.8, "evidence": ["e1"], "gaps": []},
        "grader": [
            SimpleNamespace(name="physics", score=0.7, passed=True, checks=[], message="ok"),
            SimpleNamespace(name="dim", score=0.0, passed=False, checks=[], message="bad"),
        ],
        "step": SimpleNamespace(
            step_id=1, on_track="true", structure_check="passed",
            evidence_quality="high", deviation="", target_chain_ref="T",
        ),
    }
    _u = _ev.evaluate(_ctx)
    # grader list 内部先聚合: (0.7+0.0)/2=0.35; 整体: (0.8+0.35+1.0)/3≈0.717
    assert 0.5 < _u.score < 0.9, f"聚合 score 失败: {_u.score}"
    assert _u.category == "unified"
    assert "GoalJudge" in _u.source and "physics" in _u.source
    assert _u.achieved is True

    # 7) 无输入 → 默认结果
    _u = _ev.evaluate({})
    assert _u.score == 0.0 and _u.achieved is False
    assert _u.source == "default"

    # 8) 异常输入不抛 (各分支走 try/except 返回 None / 默认)
    _u = _ev.evaluate({
        "mcda": object(),
        "goal_judge": [1, 2, 3],
        "grader": 12345,
        "step": 99,
    })
    assert isinstance(_u, UnifiedEvaluationResult)
    assert _u.source == "default", f"全异常应回默认: {_u.source}"

    # 9) 单分支也能跑 (只给 grader data dict 之外的 GraderResult list)
    _u = _ev.evaluate({
        "grader": [SimpleNamespace(name="hallucination", score=1.0, passed=True, checks=[], message="clean")],
    })
    assert _u.score == 1.0 and _u.achieved is True
    assert _u.source == "hallucination"

    # 10) from_grader 的 dict 形式 (序列化 GraderResult)
    _u = UnifiedEvaluator.from_grader({
        "name": "materials_bounds", "score": 0.0, "passed": False,
        "checks": [{"violation": "band_gap_eV=50 outside [0, 10]"}],
        "message": "band_gap_eV=50 outside [0, 10]",
    })
    assert _u.score == 0.0 and _u.achieved is False
    assert _u.source == "materials_bounds"
    assert any("band_gap" in _e for _e in _u.evidence)

    print("UnifiedEvaluator 自检通过")
