"""Goal achievement judge — 拿原始目标 + 完整轨迹 + 最终产出做端到端判定.

AutoML-Agent 的多阶段验证启发: 不只看中间步骤是否报错, 而是让独立 LLM
对照原始 objective 判断最终产出是否真正达成了目标.

与 JudgeEvaluator 的区别:
  - JudgeEvaluator 做 A/B 横向比较 (哪个更好)
  - GoalJudge 做纵向判定 (目标达成了没有)

用法:
    judge = GoalJudge(llm=some_chat_model)
    result = judge.judge(
        objective="计算 Si 的间接带隙并给出物理机制解释",
        trajectory=trajectory_dict,  # save_trajectory 的输出
        final_output="Si 的间接带隙约为 1.17 eV...",
    )
    # result = {"achieved": True, "score": 0.85, "evidence": [...], "gaps": [...]}
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


_GOAL_JUDGE_PROMPT = """你是一个严格的材料科学研究目标评审员。

## 原始研究目标
{objective}

## Agent 的完整执行轨迹 (工具调用序列)
{trajectory_summary}

## Agent 的最终产出
{final_output}

## 评审要求
请判断 agent 的最终产出是否真正达成了原始目标。考虑:
1. 是否回答了目标中提出的所有问题?
2. 数值结果是否合理 (物理常识)?
3. 是否有物理机制解释 (而非纯数值)?
4. 是否有明确的结论 (而非含糊其辞)?

如果目标包含实验验证/计算验证, 产出中是否包含验证结果?

输出 JSON (不要 markdown 代码块):
{{
    "achieved": true/false,
    "score": 0.0-1.0,
    "evidence": ["支持达成的证据1", "证据2"],
    "gaps": ["未达成的方面1", "方面2"],
    "reasoning": "简要说明判定理由"
}}"""


def _summarize_trajectory(trajectory: dict[str, Any] | None) -> str:
    """把轨迹压缩成 prompt 可读的摘要."""
    if trajectory is None:
        return "(无轨迹数据)"

    tool_calls = trajectory.get("tool_calls") or []
    if not tool_calls:
        # 也许直接是 phases 列表
        phases = trajectory.get("phases") or []
        if phases:
            lines = [f"- {p}" for p in phases]
            return f"执行了 {len(phases)} 个阶段:\n" + "\n".join(lines)
        return "(无工具调用记录)"

    lines = []
    for i, tc in enumerate(tool_calls, 1):
        tool = tc.get("tool", "?")
        success = tc.get("success", True)
        status = "✓" if success else "✗"
        result_preview = str(tc.get("result", ""))[:200]
        lines.append(f"{i}. [{status}] {tool}: {result_preview}")

    return f"共 {len(tool_calls)} 个工具调用:\n" + "\n".join(lines)


class GoalJudge:
    """端到端目标达成判定器.

    用独立 LLM 对照原始 objective 评判最终产出, 输出 achieved/score/evidence/gaps.
    没有 LLM 时降级到规则判定 (产出非空 + 包含目标关键词).
    """

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    def judge(
        self,
        objective: str,
        trajectory: dict[str, Any] | None = None,
        final_output: str = "",
    ) -> dict[str, Any]:
        """判定目标是否达成.

        Returns:
            {"achieved": bool, "score": float, "evidence": list, "gaps": list}
        """
        traj_summary = _summarize_trajectory(trajectory)

        # 没有 LLM 就走规则降级
        if self._llm is None:
            return self._rule_based_judge(objective, final_output)

        prompt = _GOAL_JUDGE_PROMPT.format(
            objective=objective,
            trajectory_summary=traj_summary,
            final_output=final_output[:4000],
        )

        try:
            response = self._invoke_llm(prompt)
            result = self._parse_response(response)
            if result is not None:
                return result
            # JSON 解析失败, 降级到规则
            return self._rule_based_judge(objective, final_output)
        except Exception as e:
            logger.warning("GoalJudge LLM invoke failed: %s, falling back to rules", e)
            return self._rule_based_judge(objective, final_output)

    def _invoke_llm(self, prompt: str) -> str:
        """调 LLM, 兼容 langchain ChatModel 和 raw callable."""
        if hasattr(self._llm, "invoke"):
            # langchain ChatModel
            from langchain_core.messages import HumanMessage
            result = self._llm.invoke([HumanMessage(content=prompt)])
            return result.content if hasattr(result, "content") else str(result)
        elif callable(self._llm):
            return str(self._llm(prompt))
        raise RuntimeError("LLM is neither a ChatModel nor callable")

    def _parse_response(self, response: str) -> dict[str, Any]:
        """解析 LLM 返回的 JSON, 解析失败时回退到规则判定."""
        # 去掉可能的 markdown 代码块包裹
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            data = json.loads(text)
            return {
                "achieved": bool(data.get("achieved", False)),
                "score": float(data.get("score", 0.0)),
                "evidence": data.get("evidence", []),
                "gaps": data.get("gaps", []),
                "reasoning": data.get("reasoning", ""),
            }
        except (json.JSONDecodeError, ValueError):
            # JSON 解析失败, 降级到规则判定
            logger.warning("GoalJudge JSON parse failed, falling back to rules")
            return None  # signal fallback needed

    def _rule_based_judge(self, objective: str, final_output: str) -> dict[str, Any]:
        """无 LLM 时的降级判定: 基于关键词覆盖度."""
        if not final_output or not final_output.strip():
            return {
                "achieved": False,
                "score": 0.0,
                "evidence": [],
                "gaps": ["no output produced"],
                "reasoning": "rule-based: empty output",
            }

        # 从 objective 抽关键词 (去掉停用词)
        stop = {"的", "和", "与", "在", "了", "是", "并", "求", "计算", "分析",
                "给出", "提供", "the", "a", "an", "of", "and", "to", "for", "in",
                "compute", "calculate", "find", "predict", "is", "are", "was", "were"}
        words = [w for w in objective.lower().split()
                 if len(w) > 1 and w.lower() not in stop]
        if not words:
            words = [objective.lower()]

        output_lower = final_output.lower()
        covered = sum(1 for w in words if w in output_lower)
        coverage = covered / len(words) if words else 0.0

        achieved = coverage >= 0.5 and len(final_output) > 20
        gaps = [] if achieved else [
            f"only {covered}/{len(words)} keywords matched"
        ] if coverage < 0.5 else [
            "output too short"
        ]

        return {
            "achieved": achieved,
            "score": round(coverage, 4),
            "evidence": [f"keyword coverage: {covered}/{len(words)}"],
            "gaps": gaps,
            "reasoning": "rule-based fallback: keyword coverage",
        }


__all__ = ["GoalJudge"]
