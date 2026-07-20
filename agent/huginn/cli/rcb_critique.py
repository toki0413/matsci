"""rcb_runner 拆分: adversarial_critique + Decision/CritiqueResult + format_critique_for_agent.

抽自 rcb_runner.py L56-513, 单一职责 = 独立 LLM 调用做 skeptical reviewer.

ponytail: 不引新依赖, 不改逻辑, 纯 import 抽取. 测试靠 rcb_runner 集成路径.
"""
from __future__ import annotations

import difflib
import json
import logging
from dataclasses import dataclass, field as _dc_field
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

    mode="object": critique report (原逻辑不变)
    mode="meta": critique agent 自修改提案 (L2 元元认知层, 哥德尔机 proof
                 verifier 弱化版 — LLM judge 替代形式化 proof)

    不让 agent 自检, 因为 agent 写的 report 它自己不会判造假.
    用独立 LLM 调用 (新 system prompt, 无对话历史) 做 adversarial review.
    返回结构化 JSON, format_critique_for_agent() 把它转成 agent 可读的修复指令.
    """
    if mode == "meta":
        # 早期拒绝查重 — 命中直接返回, 不调 LLM 省 token
        # ponytail: 天花板是 difflib 字符串相似度 (抓不到同义改写这类语义近义,
        #           "always use X" 改写成 "X must be used" 漏判);
        #           升级路径换 embedding 相似度 (sentence-transformers cosine > 0.85)
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
        # 调 LLM 做完整评估 — 复用现有 ainvoke 模式, 同 client 不同 system prompt
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

    # === object mode (原逻辑不变) ===
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


# === G42: critique_decision — v4 预留接口第 1 项 ===
# 扩展 adversarial_critique 到 mode/phase/tool 决策层. 不破坏 object/meta 双层.
# ponytail: 用同样的独立 LLM 调用模式, 不引入新组件. 升级路径是接 LeanInterface
# 对决策做形式化验证 (mode/phase 转移是可形式化的有限状态机).


@dataclass
class Decision:
    """可 critique 的决策 — mode 切换 / phase 转移 / tool 选择.

    frm/to 用字符串而非 enum, 因为 mode/phase/tool 名字都是有限字符串集合,
    不必为每种决策维护单独 enum. 调用方负责传合法值.
    """
    kind: str  # "mode_switch" | "phase_transition" | "tool_select"
    frm: str
    to: str
    rationale: str = ""
    metadata: dict[str, Any] = _dc_field(default_factory=dict)


@dataclass
class CritiqueResult:
    """critique_decision 的返回 — 含 red_flags + suggestions + 总体 verdict + gap_type.

    gap_type (v14 Task 7): 标识 gap 类型, 驱动 Step3→Step2 回退判定.
      - numeric_recompute: 数值需重算 (MAE 算错 / exclusion curve 数据需重跑)
      - exact_component_missing: [EXACT] 标记组件缺失 (graph VAE 缺失 / Bayesian 未实现)
      - text_description: 仅文字描述不足 (methodology 段太短) — 不触发回退
      - none: verdict=pass, 无 gap
    """
    verdict: str  # "accept" | "reject" | "fix_needed"
    red_flags: list[str] = _dc_field(default_factory=list)
    suggestions: list[str] = _dc_field(default_factory=list)
    reason: str = ""
    gap_type: str = "none"  # v14 Task 7: 驱动 Step3→Step2 拓扑许可回退


# 决策类型的合法 from/to 集合 — 模板 critique 用, LLM 路径不依赖
_VALID_MODES = frozenset({"chat", "plan", "research"})
_VALID_PHASES = frozenset({
    "perceive", "hypothesize", "plan", "execute", "validate", "learn", "report",
})


def _template_critique_decision(decision: Decision, context: dict[str, Any]) -> CritiqueResult:
    """模板 critique — 无 LLM 时走规则. 三层检查:
    1. kind 合法
    2. frm/to 在对应合法集合里
    3. rationale 非空 (无理由的决策不可 critique)
    ponytail: 这是廉价兜底, 真正的 critique 走 LLM 路径.
    """
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
        # 7-phase pipeline: 只能向前走或回 perceive, 不能跳过 execute → report
        if decision.frm in _VALID_PHASES and decision.to in _VALID_PHASES:
            phases_order = list(_VALID_PHASES)
            # report 不能跳回 execute (除非显式 force)
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
    """L3 decision critique — 扩展 adversarial_critique 到 mode/phase/tool.

    不破坏现有 object/meta 双层 critique. 这是 v4 预留的第 1 项接口实现.
    复用同样独立 LLM 调用模式: 新 system prompt, 无对话历史, 输出结构化 JSON.

    model=None 时走模板规则 (测试用); model 传入时优先调 LLM, 失败降级到模板.
    """
    context = context or {}

    # 先走模板兜底 — 模板都过不了的不必浪费 LLM 调用
    template_result = _template_critique_decision(decision, context)
    if template_result.verdict == "reject":
        return template_result

    client = llm_client if llm_client is not None else model
    if client is None or not hasattr(client, "ainvoke"):
        return template_result

    # LLM 路径 — 让 skeptical reviewer 看决策是否合理
    from langchain_core.messages import HumanMessage, SystemMessage
    system = SystemMessage(content=(
        "你是 DECISION AUDITOR 评估 agent 的 mode/phase/tool 决策.\n"
        "你的工作是找 FLAWS — 不合理的转移、跳阶段、tool 选择错误.\n"
        "输出严格 JSON: "
        '{"verdict": "accept"|"reject"|"fix_needed", '
        '"red_flags": [string], "suggestions": [string], "reason": string, '
        '"gap_type": "numeric_recompute"|"exact_component_missing"|"text_description"|"none"}\n'
        "gap_type 分类 (v14 Task 7, 驱动 Step3→Step2 回退判定):\n"
        "  - numeric_recompute: 数值需重算 (MAE 算错 / exclusion curve 数据需重跑)\n"
        "  - exact_component_missing: [EXACT] 标记组件缺失 (graph VAE 缺失 / Bayesian 未实现)\n"
        "  - text_description: 仅文字描述不足 (methodology 段太短) — 不触发回退\n"
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
        # v14 Task 7: gap_type 校验 — 非法值回退 none, 避免 _should_retry_execute 误判
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

    # G28: 数值重算出的 red flags — report 写的数 vs outputs/ 实际数据对不上
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
