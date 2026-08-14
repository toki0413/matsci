"""Epistemic gate — IOED (illusion of explanatory depth) guard for the autoloop.

存在目的: 长程科研任务里, agent 很容易在"能复述一个事实"和"真正理解机制"
之间产生错觉深度 (IOED): 自信地做出断言, 但实际没有证据支撑, 也没有标注
不确定性. 本模块在 autoloop 的 ``_validate`` 阶段加一道纯启发式闸门, 检查
执行结果里是否出现"强断言 + 证据缺失"的组合, 若是则输出一个
``epistemic_gap`` 警告, 让下游 (report / 下一轮 hypothesis) 看到知识缺口.

设计约束:
- 纯启发式, 默认不调 LLM (零额外成本), 永不抛异常.
- 只"报告"缺口, 不改 ``tests_passed`` / ``constraints_satisfied`` — 它是
  可见性字段, 不是硬 gate, 避免误伤正常通过的执行.
- 与系统提示的 Epistemic Honesty 指令 (prompts.py principle 7) 配合:
  指令告诉模型要诚实, 这道闸门在模型失灵时兜底暴露缺口.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# 强断言信号词 — 出现即表示结果在"下结论", 而非报告原始数据.
_CONFIDENT_CLAIM_RE = re.compile(
    r"\b(?:is|are|equals?|proves?|shows?|demonstrates?|determined|"
    r"conclusively|definitively|confirmed|verified|the reason is|"
    r"indicates? that|caused by|due to|leads to|results in)\b",
    re.IGNORECASE,
)

# 不确定性/诚实标注信号词 — 出现即表示 agent 已意识到局限, 无需强提示.
_UNCERTAINTY_RE = re.compile(
    r"\b(?:uncertain|unknown|uncertainty|approximately|estimated|"
    r"roughly|~|may|might|could|possibly|suggest[s]?|not sure|"
    r"needs? (?:verification|validation|confirmation)|i don'?t know|"
    r"有待|需验证|不确定|未知|可能|需要验证|超出.*知识)\b",
    re.IGNORECASE,
)

# 证据信号 — 结果里出现数值/引用/benchmark 即视为有支撑.
_EVIDENCE_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:eV|eV/atom|kJ|GPa|MPa|K|nm|Å|angstrom|%|"
    r"cm-1|g/cm|mol|W/m|S/cm|J|m|s|this work|ref\.|doi|benchmark|"
    r"MATERIALS|database|verified|measured|calculated)\b",
    re.IGNORECASE,
)


def _text_of(value: Any, limit: int = 4000) -> str:
    """把任意执行结果字段压成可分析的纯文本."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, (dict, list)):
        try:
            import json

            return json.dumps(value, ensure_ascii=False, default=str)[:limit]
        except Exception:
            return str(value)[:limit]
    return str(value)[:limit]


def _extract_claim_text(execution_result: Any) -> str:
    """从执行结果里抽取"结论/断言"文本, 用于判断是否下结论."""
    if not isinstance(execution_result, dict):
        return _text_of(execution_result)
    for key in (
        "final_answer", "conclusion", "summary", "answer",
        "result_data", "description", "result",
    ):
        if key in execution_result:
            return _text_of(execution_result[key])
    # 兜底: 整个结果压成文本
    return _text_of(execution_result)


def _evidence_present(execution_result: Any, results: dict[str, Any]) -> bool:
    """是否有可辨识的支撑证据: 数值/引用/benchmark/物理校验."""
    blob = _text_of(execution_result) + " " + _text_of(results)
    if _EVIDENCE_RE.search(blob):
        return True
    # 显式校验结果也算证据
    for key in ("r_phys", "benchmarks", "grader_scores", "generative_verify",
                "physics_validation", "math_validation", "eval_summary",
                "literature_claims"):
        v = results.get(key)
        if v:
            return True
    return False


def detect_epistemic_gap(
    execution_result: Any,
    results: dict[str, Any],
) -> dict[str, Any] | None:
    """检测执行结果里的"强断言 + 证据缺失"知识缺口.

    Returns:
        有缺口时返回 ``{"overconfidence": bool, "advice": str, ...}`` dict,
        否则返回 None. 永不抛异常.
    """
    try:
        claim_text = _extract_claim_text(execution_result)
        if not claim_text.strip():
            return None

        has_confident = bool(_CONFIDENT_CLAIM_RE.search(claim_text))
        has_uncertainty = bool(_UNCERTAINTY_RE.search(claim_text))
        has_evidence = _evidence_present(execution_result, results)

        # 强断言 + 无证据 + 无不确定性标注 → 疑似 IOED 缺口
        if has_confident and not has_evidence and not has_uncertainty:
            return {
                "overconfidence": True,
                "has_evidence": False,
                "advice": (
                    "结论包含强断言但缺少可验证证据(数值/引用/校验), 且未标注"
                    "不确定性. 请在最终交付中区分 established / estimated / "
                    "unknown: 明确本轮是通过计算、检索还是回忆得出该结论, "
                    "并指出需要哪些额外验证."
                ),
            }
        # 强断言 + 有证据, 但完全没有不确定性标注 → 轻度提示补齐局限说明
        if has_confident and has_evidence and not has_uncertainty:
            return {
                "overconfidence": False,
                "has_evidence": True,
                "advice": (
                    "结论有证据支撑, 但建议补一句不确定性/适用边界说明, "
                    "明确估计误差与尚未验证的假设."
                ),
            }
        return None
    except Exception as exc:  # pragma: no cover - 防御性兜底
        logger.debug("epistemic gap detection skipped: %s", exc)
        return None