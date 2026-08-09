"""Prompt builder — 三元 (mode/phase/metacog) aware prompt 构造器.

v4 Task 6 (G18): 统一 persona/mode/phase/metacog/tools/safety 六段构造,
替代 context.py 里散落的拼接逻辑.

迁移状态:
- persona: persona_segment() 接受可选 system_prompt 参数, context.py 的 s7
  委托路径已传入 self.system_prompt. env/taste/missing_backends/stable_principles
  等会话级动态层仍由 context.py 维护 (与 agent 实例状态强耦合, 不进 prompt_builder).
- phase: phase_segment() 已整合 phases.py 的完整 PHASE_PROMPTS (经 autoloop↔
  ResearchPhase 双向映射), v6 G51 结构关系语义对齐补充单独维护.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# 迁移完成: persona_segment 接受可选 system_prompt 参数, 调用方 (context.py 的
# s7 委托路径) 传入 runtime persona 时优先使用; 未传入则回退到内置最小 persona.
# env/taste/missing_backends/stable_principles 等会话级动态层仍由 context.py 维护,
# 因其与 agent 实例状态 (workspace / _cached_taste / _csm) 强耦合, 不进 prompt_builder.
def persona_segment(system_prompt: str | None = None) -> str:
    if system_prompt:
        return f"## PERSONA\n{system_prompt}"
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


# 迁移完成: phase_segment 整合 phases.py 的完整 PHASE_PROMPTS (经 autoloop↔
# ResearchPhase 双向映射同时支持两种调用约定). v6 G51 结构关系语义对齐补充
# 单独维护 (不进 phases.py, 因 phases.py 是通用层).
# v6 G51: hypothesize / validate 加结构关系语义对齐 — 先识别物理结构, 再断言同构保持
_PHASE_G51_NOTES = {
    "hypothesis": (
        "v6 G51: before proposing, identify the physical structure "
        "(relation_type + implementor_slots) of the target problem; "
        "if a source-domain analogue exists, assert structure preservation "
        "via validate_structure_preservation."
    ),
    "validation": (
        "v6 G51: if the hypothesis carried a PhysicalStructure, re-assert "
        "isomorphism on the observed results; emit structure_violation signal "
        "via SignalHub when preservation breaks."
    ),
}


def _resolve_research_phase(phase: str):
    """将 phase 字符串解析为 ResearchPhase, 兼容 autoloop 名和 ResearchPhase 值.

    返回 None 表示无法识别. OPEN 可识别 (返回 ResearchPhase.OPEN), 其 PHASE_PROMPTS
    为空串, 由调用方 phase_segment 处理为空输出.
    """
    from huginn.phases import AUTOLOOP_TO_PHASE, ResearchPhase
    # 先按 autoloop 名映射 (perceive/hypothesize/plan/execute/validate/learn/report)
    rp = AUTOLOOP_TO_PHASE.get(phase)
    if rp is not None:
        return rp
    # 再按 ResearchPhase 值直接解析 (literature/hypothesis/planning/execution/.../open)
    try:
        return ResearchPhase(phase)
    except (ValueError, TypeError):
        return None


def phase_segment(phase: str) -> str:
    """构造 phase 引导段.

    迁移完成: 整合 phases.py 的完整 PHASE_PROMPTS, 并叠加 v6 G51 结构关系语义对齐补充.
    兼容 autoloop phase 名 (perceive/...) 与 ResearchPhase 值 (literature/...).
    未知 phase 或 OPEN 返回空串, 不抛异常.
    """
    rp = _resolve_research_phase(phase)
    if rp is None:
        return ""
    from huginn.phases import PHASE_PROMPTS
    rich = PHASE_PROMPTS.get(rp, "")
    # G51 补充按 ResearchPhase value 查 (hypothesis/validation)
    g51 = _PHASE_G51_NOTES.get(rp.value, "")
    if not rich and not g51:
        return ""
    parts = []
    if rich:
        parts.append(rich)
    if g51:
        parts.append(g51)
    return "\n".join(parts)


def metacog_segment(metacog_state: str) -> str:
    # 非自修改态不注入; 即使 metacog_state 不认识也不抛异常
    if metacog_state != "s7_self_modify":
        return ""
    parts = [
        "## METACOG: SELF-MODIFY",
        "You are in S7 self-modify. Call self_observe to read recent failure "
        "patterns before proposing any change. Propose one concrete modification "
        "(a new stable principle, a prompt patch, or a tool preference change).",
        # v6 G52: 结构主义反思 — 若近期失败携带 structure_relation_type,
        # 用 enumerate_implementors 看是否有未试过的 implementor 替换,
        # 再决定是否更新 recall_strategy 表的 structure_relation_type 条目.
        "v6 G52: if recent failures carry a structure_relation_type, run "
        "enumerate_implementors on the affected PhysicalStructure to see "
        "whether an untried implementor substitution exists; consider "
        "updating recall_strategy entries tagged with that relation_type.",
    ]
    try:
        from huginn.memory import load_stable_principles

        principles = load_stable_principles()[:5]
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


# ponytail: safety 段保持最小; 不涉及 persona/phase 迁移.
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


def build_prompt(
    mode: str,
    phase: str,
    metacog_state: str,
    system_prompt: str | None = None,
) -> str:
    """构造三元 aware system prompt.

    任意未知 mode/phase/metacog_state 都不会抛异常 — 对应段直接跳过.
    system_prompt 可选: 传入时 persona_segment 使用 runtime persona (迁移自
    context.py 的 self.system_prompt); 未传入则回退到内置最小 persona (向后兼容).
    """
    segments = [
        persona_segment(system_prompt),
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
    # persona 迁移: 传入 system_prompt 时应替换内置最小 persona
    p4 = build_prompt("chat", "execute", "s0_blank", system_prompt="My custom persona.")
    assert "My custom persona." in p4
    print("OK")
