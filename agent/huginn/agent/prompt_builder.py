"""Prompt builder — 三元 (mode/phase/metacog) aware prompt 构造器.

v4 Task 6 (G18): 统一 persona/mode/phase/metacog/tools/safety 六段构造,
替代 context.py 里散落的拼接逻辑. 当前为最小核心, context.py 仍维护完整 persona
(Task 7 才做委托迁移).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ponytail: 最小核心段, 完整 persona 仍由 context.py 维护;
# 升级路径是逐步迁移 persona 到此
def persona_segment() -> str:
    return (
        "## PERSONA\n"
        "You are a materials-science research companion. Help the user design, "
        "run, and interpret simulations and experiments with rigor."
    )


_MODE_INSTRUCTIONS = {
    "chat": (
        "Conversational assistance. Answer directly; avoid heavy simulation "
        "tooling unless the user explicitly asks for it."
    ),
    "research": (
        "Systematic research mode. Cite literature for claims, quantify "
        "uncertainty, compare results to published values, flag unexpected "
        "results as potential discoveries, write findings to the knowledge base."
    ),
    "code": (
        "Code-act mode. Solve tasks by writing and executing code in the "
        "sandbox; verify each step before reporting."
    ),
    "fusion": (
        "Fusion mode. Integrate evidence across simulation, experiment, and "
        "literature into a coherent conclusion; reconcile contradictions explicitly."
    ),
}


def mode_segment(mode: str) -> str:
    instr = _MODE_INSTRUCTIONS.get(mode)
    if not instr:
        return ""
    return f"## MODE: {mode.upper()}\n{instr}"


# ponytail: phase 引导为最小版, 升级路径是从 cognitive_engine.py 迁移完整 PHASE_PROMPTS
_PHASE_GUIDE = {
    "perceive": "Perceive: gather observations; read the problem and existing data.",
    "hypothesize": "Hypothesize: propose falsifiable hypotheses grounded in theory.",
    "plan": "Plan: design an experiment or simulation protocol with checkpoints.",
    "execute": "Execute: run the plan step by step, capturing raw outputs.",
    "validate": "Validate: cross-check results against physics, references, sanity bounds.",
    "learn": "Learn: extract transferable principles; write to memory/knowledge base.",
    "report": "Report: summarize methods, results, uncertainty, and open questions.",
}


def phase_segment(phase: str) -> str:
    guide = _PHASE_GUIDE.get(phase)
    if not guide:
        return ""
    return f"## PHASE: {phase.upper()}\n{guide}"


def metacog_segment(metacog_state: str) -> str:
    # 非自修改态不注入; 即使 metacog_state 不认识也不抛异常
    if metacog_state != "s7_self_modify":
        return ""
    parts = [
        "## METACOG: SELF-MODIFY",
        "You are in S7 self-modify. Call self_observe to read recent failure "
        "patterns before proposing any change. Propose one concrete modification "
        "(a new stable principle, a prompt patch, or a tool preference change).",
    ]
    try:
        from huginn.memory import load_stable_principles

        principles = load_stable_principles()
        if principles:
            parts.append("### STABLE_PRINCIPLES")
            parts.extend(f"- {p}" for p in principles)
    except Exception:
        # 文件缺失/损坏/import 失败都不应让 build_prompt 抛异常
        logger.debug("stable_principles load skipped in prompt_builder", exc_info=True)
    return "\n".join(parts)


# ponytail: tools_segment 不做复杂过滤, 避免重复 tools/__init__.py 的 schema 生成逻辑;
# 升级路径是加 mode/phase/state 过滤规则映射
def tools_segment(mode: str, phase: str, metacog_state: str) -> str:
    return "## TOOLS\nTools available per current mode/phase/state."


# ponytail: 最小核心段, 完整 persona 仍由 context.py 维护;
# 升级路径是逐步迁移 persona 到此
def safety_segment() -> str:
    return (
        "## SAFETY\n"
        "- Physics precheck: verify inputs are physically plausible before "
        "launching any simulation.\n"
        "- Data integrity: never overwrite or delete user data without explicit "
        "confirmation.\n"
        "- No fabrication: if a result is missing or a tool fails, report it "
        "honestly — do not invent values."
    )


def build_prompt(mode: str, phase: str, metacog_state: str) -> str:
    """构造三元 aware system prompt.

    任意未知 mode/phase/metacog_state 都不会抛异常 — 对应段直接跳过.
    """
    segments = [
        persona_segment(),
        mode_segment(mode),
        phase_segment(phase),
        metacog_segment(metacog_state),
        tools_segment(mode, phase, metacog_state),
        safety_segment(),
    ]
    return "\n\n".join(s for s in segments if s)


if __name__ == "__main__":
    p = build_prompt("research", "execute", "s4_construct")
    assert p and len(p) > 50, "build_prompt returned empty/short"
    assert "research" in p.lower() or "execute" in p.lower(), "mode/phase missing"
    # 未知 metacog_state 不应抛异常
    p2 = build_prompt("chat", "perceive", "unknown_state")
    assert p2 and len(p2) > 50
    # s7 应触发 self-modify 段 (即使 principles 文件不存在, try/except 兜底)
    p3 = build_prompt("research", "validate", "s7_self_modify")
    assert "SELF-MODIFY" in p3
    print("OK")
