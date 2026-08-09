"""通用对抗式审查 — 独立 LLM 调用做 skeptical reviewer.

从 cli/rcb_critique.py 迁移到此, 消除 agent 通用代码对 RCB 模块的依赖.
adversarial_critique / critique_decision / format_critique_for_agent
都是通用元认知能力, 不绑定任何 benchmark 框架.
"""
from __future__ import annotations

import difflib
import json
import logging
from dataclasses import dataclass
from dataclasses import field as _dc_field
from typing import Any

logger = logging.getLogger(__name__)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        text = text[nl + 1:] if nl > 0 else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


async def adversarial_critique(
    model: Any = None,
    report: str = "",
    checklist: str = "",
    *,
    mode: str = "object",
    proposal: str = "",
    system_prompt_summary: str = "",
    recent_rejections: list[str] | None = None,
    llm_client: Any = None,
) -> dict[str, Any]:
    """独立 LLM 调用做 skeptical reviewer — 消除 confirmation bias.

    mode="object": critique report
    mode="meta": critique agent 自修改提案 (L2 元元认知层)
    """
    if mode == "meta":
        for prev in recent_rejections or []:
            ratio = difflib.SequenceMatcher(None, proposal, prev).ratio()
            if ratio > 0.8:
                logger.info("meta_critique early_reject: similarity=%.2f", ratio)
                return {
                    "verdict": "reject",
                    "reason": f"similar to past rejection (similarity={ratio:.2f})",
                    "expected_utility_delta": 0.0,
                    "early_reject": True,
                }
        from langchain_core.messages import HumanMessage, SystemMessage
        client = llm_client if llm_client is not None else model
        meta_system = SystemMessage(content=(
            "你是 META-REVIEWER 评估 agent 的自修改提案.\n"
            "你的工作是判断这个提案是否会真正改进 agent 的效用, "
            "还是会污染 gradient 或引入坏习惯.\n\n"
            "评估维度:\n"
            "1. 是否污染 gradient (例如加 CRITICAL: always use X) — 这是 σ₄ lesson\n"
            "2. 是否与最近 rejection 相似 (相似度 > 0.8 直接 reject)\n"
            "3. 是否与现有 stable_principles 冲突\n"
            "4. expected_utility_delta 是否为正\n\n"
            "输出严格 JSON: "
            '{"verdict": "accept"|"reject", "reason": "...", "expected_utility_delta": float}'
        ))
        rejections_block = "\n".join(f"- {r}" for r in (recent_rejections or [])) or "(none)"
        meta_human = HumanMessage(content=(
            f"## Proposal\n{proposal}\n\n"
            f"## Current system prompt summary\n{system_prompt_summary or '(empty)'}\n\n"
            f"## Recent rejections (do not repeat)\n{rejections_block}\n\n"
            "Output ONLY the JSON object."
        ))
        try:
            resp = await client.ainvoke([meta_system, meta_human])
            text = resp.content if hasattr(resp, "content") else str(resp)
            text = _strip_code_fences(text)
            result = json.loads(text)
            result.setdefault("verdict", "reject")
            result.setdefault("reason", "no reason provided")
            result.setdefault("expected_utility_delta", 0.0)
            result["early_reject"] = False
            logger.info("meta_critique: verdict=%s", result["verdict"])
            return result
        except Exception as e:
            logger.warning("meta_critique failed: %s", e)
            return {
                "verdict": "reject",
                "reason": f"meta_critique error: {e}",
                "expected_utility_delta": 0.0,
                "early_reject": False,
                "error": str(e),
            }

    # === object mode ===
    from langchain_core.messages import HumanMessage, SystemMessage
    system = SystemMessage(content=(
        "You are a SKEPTICAL SCIENTIFIC REVIEWER who wants to score this report LOW. "
        "Your job is to find FLAWS. Be adversarial. "
        "Output ONLY a valid JSON object. No markdown fences, no preamble."
    ))
    human = HumanMessage(content=(
        f"## Methodology Checklist (from Step 1)\n{checklist}\n\n"
        f"## Report to Critique\n{report}\n\n"
        "## Your Task (output JSON only)\n"
        "1. \"implausible_metrics\": metrics where report value is BETTER than paper baseline. "
        "Format: [{\"metric\": name, \"paper\": value, \"yours\": value, \"red_flag\": why}]. "
        "Empty list if none.\n"
        "2. \"silent_substitutions\": [EXACT] components silently replaced with simpler alternatives. "
        "Format: [{\"component\": name, \"expected\": what, \"actual\": what}]. Empty list if none.\n"
        "3. \"missing_components\": checklist items absent from report. Empty list if none.\n"
        "4. \"overall_verdict\": \"pass\" | \"fix_needed\" | \"fail\"\n\n"
        "Output ONLY the JSON object."
    ))
    try:
        resp = await model.ainvoke([system, human])
        text = resp.content if hasattr(resp, "content") else str(resp)
        text = _strip_code_fences(text)
        result = json.loads(text)
        result.setdefault("implausible_metrics", [])
        result.setdefault("silent_substitutions", [])
        result.setdefault("missing_components", [])
        result.setdefault("overall_verdict", "fix_needed")
        logger.info("adversarial_critique: verdict=%s", result["overall_verdict"])
        return result
    except Exception as e:
        logger.warning("adversarial_critique failed: %s", e)
        return {
            "implausible_metrics": [],
            "silent_substitutions": [],
            "missing_components": [],
            "overall_verdict": "fix_needed",
            "error": str(e),
        }


@dataclass
class Decision:
    """可 critique 的决策 — mode 切换 / phase 转移 / tool 选择."""
    kind: str  # "mode_switch" | "phase_transition" | "tool_select"
    frm: str
    to: str
    rationale: str = ""
    metadata: dict[str, Any] = _dc_field(default_factory=dict)


@dataclass
class CritiqueResult:
    """critique_decision 的返回 — 含 red_flags + suggestions + 总体 verdict + gap_type."""
    verdict: str  # "accept" | "reject" | "fix_needed"
    red_flags: list[str] = _dc_field(default_factory=list)
    suggestions: list[str] = _dc_field(default_factory=list)
    reason: str = ""
    gap_type: str = "none"


_VALID_MODES = frozenset({"chat", "plan", "research"})
_VALID_PHASES = frozenset({
    "perceive", "hypothesize", "plan", "execute", "validate", "learn", "report",
})


def _template_critique_decision(decision: Decision, context: dict[str, Any]) -> CritiqueResult:
    """模板 critique — 无 LLM 时走规则."""
    red_flags: list[str] = []
    suggestions: list[str] = []

    if decision.kind not in ("mode_switch", "phase_transition", "tool_select"):
        return CritiqueResult(
            verdict="reject",
            red_flags=[f"unknown decision kind: {decision.kind}"],
            reason="invalid kind",
        )

    if decision.kind == "mode_switch":
        if decision.frm and decision.frm not in _VALID_MODES:
            red_flags.append(f"frm mode '{decision.frm}' not in {_VALID_MODES}")
        if decision.to not in _VALID_MODES:
            red_flags.append(f"to mode '{decision.to}' not in {_VALID_MODES}")
    elif decision.kind == "phase_transition":
        if decision.frm and decision.frm not in _VALID_PHASES:
            red_flags.append(f"frm phase '{decision.frm}' not in {_VALID_PHASES}")
        if decision.to not in _VALID_PHASES:
            red_flags.append(f"to phase '{decision.to}' not in {_VALID_PHASES}")
        if decision.frm in _VALID_PHASES and decision.to in _VALID_PHASES:
            list(_VALID_PHASES)
            if decision.frm == "report" and decision.to == "execute" and not context.get("force"):
                red_flags.append("report → execute backtrack without force=True")
                suggestions.append("if refining after report, pass context['force']=True")

    if not decision.rationale.strip():
        red_flags.append("empty rationale — decision without justification")
        suggestions.append("provide rationale explaining why this decision is needed")

    if red_flags:
        return CritiqueResult(
            verdict="fix_needed",
            red_flags=red_flags,
            suggestions=suggestions,
            reason="template rules failed",
        )
    return CritiqueResult(verdict="accept", reason="template rules passed")


async def critique_decision(
    decision: Decision,
    context: dict[str, Any] | None = None,
    *,
    model: Any = None,
    llm_client: Any = None,
) -> CritiqueResult:
    """L3 decision critique — 扩展 adversarial_critique 到 mode/phase/tool."""
    context = context or {}

    template_result = _template_critique_decision(decision, context)
    if template_result.verdict == "reject":
        return template_result

    client = llm_client if llm_client is not None else model
    if client is None or not hasattr(client, "ainvoke"):
        return template_result

    from langchain_core.messages import HumanMessage, SystemMessage
    system = SystemMessage(content=(
        "你是 DECISION AUDITOR 评估 agent 的 mode/phase/tool 决策.\n"
        "你的工作是找 FLAWS — 不合理的转移、跳阶段、tool 选择错误.\n"
        "输出严格 JSON: "
        '{"verdict": "accept"|"reject"|"fix_needed", '
        '"red_flags": [string], "suggestions": [string], "reason": string, '
        '"gap_type": "numeric_recompute"|"exact_component_missing"|"text_description"|"none"}\n'
        "gap_type 分类:\n"
        "  - numeric_recompute: 数值需重算\n"
        "  - exact_component_missing: 标记组件缺失\n"
        "  - text_description: 仅文字描述不足 — 不触发回退\n"
        "  - none: 无 gap, verdict=pass 时必为 none"
    ))
    ctx_str = json.dumps(context, ensure_ascii=False, default=str)[:2000]
    human = HumanMessage(content=(
        f"## Decision\n"
        f"kind: {decision.kind}\n"
        f"from: {decision.frm}\n"
        f"to: {decision.to}\n"
        f"rationale: {decision.rationale}\n\n"
        f"## Context\n{ctx_str}\n\n"
        "Output ONLY the JSON object."
    ))
    try:
        resp = await client.ainvoke([system, human])
        text = resp.content if hasattr(resp, "content") else str(resp)
        text = _strip_code_fences(text)
        data = json.loads(text)
        _gap = data.get("gap_type", "none")
        if _gap not in ("numeric_recompute", "exact_component_missing",
                        "text_description", "none"):
            _gap = "none"
        return CritiqueResult(
            verdict=data.get("verdict", "fix_needed"),
            red_flags=list(data.get("red_flags", [])),
            suggestions=list(data.get("suggestions", [])),
            reason=data.get("reason", ""),
            gap_type=_gap,
        )
    except Exception as e:
        logger.warning("critique_decision LLM path failed: %s", e)
        return template_result


def format_critique_for_agent(critique: dict[str, Any]) -> str:
    """把 critique 结果格式化成 agent 可读的修复指令."""
    lines = ["ADVERSARIAL CRITIQUE RESULTS (from independent reviewer):\n"]
    verdict = critique.get("overall_verdict", "fix_needed")
    lines.append(f"Overall verdict: {verdict.upper()}\n")

    implausible = critique.get("implausible_metrics", [])
    if implausible:
        lines.append("## RED FLAG — Implausible Metrics (better than paper)")
        for m in implausible:
            lines.append(
                f"  - {m.get('metric', '?')}: paper={m.get('paper', '?')}, "
                f"yours={m.get('yours', '?')} — {m.get('red_flag', 'investigate')}"
            )
        lines.append("")

    recomputed = critique.get("recomputed_red_flags", [])
    if recomputed:
        lines.append("## RED FLAG — Numeric Claim vs Recomputed Mismatch (fabrication suspect)")
        for m in recomputed:
            lines.append(
                f"  - {m.get('metric', '?')}: report={m.get('claimed', '?')}, "
                f"recomputed={m.get('recomputed', '?')} — {m.get('red_flag', 'investigate')}"
            )
        lines.append("")

    subs = critique.get("silent_substitutions", [])
    if subs:
        lines.append("## RED FLAG — Silent Methodology Substitutions")
        for s in subs:
            lines.append(
                f"  - {s.get('component', '?')}: expected={s.get('expected', '?')}, "
                f"actual={s.get('actual', '?')}"
            )
        lines.append("")

    missing = critique.get("missing_components", [])
    if missing:
        lines.append("## Missing Components")
        for c in missing:
            lines.append(f"  - {c}")
        lines.append("")

    lines.append("## Fix Instructions:")
    lines.append("- RED FLAG metric: investigate cause (data leakage? wrong split? bug?). "
                 "Fix the bug or document honestly.")
    lines.append("- SILENT SUBSTITUTION: implement [EXACT] component as-specified, "
                 "try >=2 approaches before giving up.")
    lines.append("- MISSING COMPONENT: implement now, or add to Limitations with error evidence.")
    lines.append("- OVERWRITE report/report.md with fixes using file_write_tool.")
    return "\n".join(lines)
