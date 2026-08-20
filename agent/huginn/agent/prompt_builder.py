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
    "extreme": (
        "Extreme mode. Long-horizon task at maximum capability: unlock all "
        "tools, persist intermediate results, follow the goal to completion, "
        "cite literature, quantify uncertainty, and flag potential discoveries."
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
        logger.debug("best-effort op failed", exc_info=True)
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


# 反防御性写作引导 (anti-defensive-writing). 目标是让论文/报告主张先行、
# 措辞精确, 避免"过度免责 / 以限制开头 / 堆叠 hedges / not-X-but-Y"这类
# 削弱表达又没增加准确度的写法。必要限制(scope/method/safety)保留一次放到
# 对的位置, 不四处散落。
def multimodal_segment() -> str:
    """多模态引导 —— 面向文本模型兜底路径 (DeepSeek Flash vision=False).

    当图像只能走 CV_TOOLS 分支时, 明确告诉模型"该直接看还是调定量工具"的取舍,
    把静默降级改成可见决策点。与 writing 段同机制挂进 prompt_segments 注册表。
    """
    return (
        "## MULTIMODAL\n"
        "- Prefer native vision: if the current model supports image input, "
        "read the image directly before reaching for tools.\n"
        "- Text model path: when native vision is unavailable, pick one explicit "
        "route — ① delegate to a vision-specialized agent, ② retrieve similar "
        "images from visual memory, ③ call image_analysis_tool for quantitative "
        "measurements.\n"
        "- No silent degradation: never describe an image without either native "
        "visual understanding or an explicit tool-based analysis."
    )


def writing_segment() -> str:
    return (
        "## WRITING\n"
        "- Lead with the claim: say what the analysis shows, proposes, or "
        "contributes before any caveat.\n"
        "- Write as the author explaining an argument, not negotiating with a "
        "hypothetical critic.\n"
        "- Convert defensive framing into positive scope: state what the work "
        "examines, not what it does not claim.\n"
        "- Replace hedging (may/could/potentially) with precise claims backed by "
        "scope and evidence strength; if uncertainty is real, name its source.\n"
        "- Keep necessary limits (scope, method, safety, accuracy) once, in the "
        "right section; drop disclaimers that add no precision.\n"
        "- Present data and conclusions honestly: state what a single mechanism "
        "can and cannot explain without apologizing for the data."
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

    "Everything is a Plugin" (形态 B): 六段已注册为段插件, 这里从注册表按
    priority 组装。若注册表为空 (import 异常等极端情况), 回退到硬编码拼接,
    保证行为不变。
    """
    from huginn.plugins.prompt_segments import assemble_prompt_segments

    assembled = assemble_prompt_segments(mode, phase, metacog_state, system_prompt)
    if assembled:
        return assembled
    # 回退: 注册表空时保持旧硬编码行为 (向后兼容).
    segments = [
        persona_segment(system_prompt),
        mode_segment(mode),
        phase_segment(phase),
        metacog_segment(metacog_state),
        tools_segment(mode, phase, metacog_state),
        writing_segment(),
        safety_segment(),
    ]
    return "\n\n".join(s for s in segments if s)


# ── "Everything is a Plugin" 形态 B: 内置六段注册为段插件 ──────────────
# 统一段签名 (mode, phase, metacog_state, system_prompt) -> str, 与
# plugins/prompt_segments.PromptSegmentFn 一致。注册发生在模块 import 时,
# 供 build_prompt 从注册表组装; 段内部逻辑不变, 仅加薄适配层。

def _persona_plugin(mode, phase, metacog_state, system_prompt):
    return persona_segment(system_prompt)


def _mode_plugin(mode, phase, metacog_state, system_prompt):
    return mode_segment(mode)


def _phase_plugin(mode, phase, metacog_state, system_prompt):
    return phase_segment(phase)


def _metacog_plugin(mode, phase, metacog_state, system_prompt):
    return metacog_segment(metacog_state)


def _tools_plugin(mode, phase, metacog_state, system_prompt):
    return tools_segment(mode, phase, metacog_state)


def _multimodal_plugin(mode, phase, metacog_state, system_prompt):
    return multimodal_segment()


def _writing_plugin(mode, phase, metacog_state, system_prompt):
    return writing_segment()


def _safety_plugin(mode, phase, metacog_state, system_prompt):
    return safety_segment()


def _thinking_plugin(mode, phase, metacog_state, system_prompt):
    """External Thinking 段 — 迁移自 context.py 的 feature-flag 注入.

    flag 开启时要求模型在动手前先用 deep_think 工具写分析。默认关, 不改变
    默认行为。fail-open: flag 层异常不输出, 保证 prompt 构建永不崩。
    """
    try:
        from huginn.feature_flags import FeatureFlags

        if not FeatureFlags.shared().is_enabled("external_thinking"):
            return ""
    except Exception:
        return ""
    return (
        "## External Thinking\n"
        "Before you answer, modify code, or call other tools, "
        "first call the `deep_think` tool and write your "
        "step-by-step analysis into its `analysis` argument. "
        "This is an external scratchpad — your analysis is recorded "
        "and returned to the developer, but is not echoed as part of "
        "your visible answer.\n"
        "Deepen it with the structured protocol when applicable: "
        "`phase` = think | plan | pre_action | reflect (stage your "
        "reasoning), and fill `hypothesis` (the claim), `evidence` "
        "(derivation), and `estimate` (a quantitative prediction, "
        "with units/range) so the prediction can be verified after "
        "execution and distilled as reusable knowledge. A verified "
        "pre_action estimate is the strongest signal you can leave. "
        "Then complete the task using that analysis."
    )


def _register_builtin_segments() -> None:
    from huginn.plugins.prompt_segments import register_prompt_segment

    register_prompt_segment("persona", _persona_plugin)
    register_prompt_segment("mode", _mode_plugin)
    register_prompt_segment("phase", _phase_plugin)
    register_prompt_segment("metacog", _metacog_plugin)
    register_prompt_segment("tools", _tools_plugin)
    register_prompt_segment("multimodal", _multimodal_plugin)
    register_prompt_segment("writing", _writing_plugin)
    # thinking 段: priority 100, 位于 tools(50) 与 safety(200) 之间.
    register_prompt_segment("thinking", _thinking_plugin)
    register_prompt_segment("safety", _safety_plugin)


_register_builtin_segments()


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
    # writing 段应注入反防御写作引导, 且位于 safety 之前
    p5 = build_prompt("research", "execute", "s4_construct")
    assert "WRITING" in p5 and "Lead with the claim" in p5
    assert p5.index("WRITING") < p5.index("SAFETY")
    print("OK")
