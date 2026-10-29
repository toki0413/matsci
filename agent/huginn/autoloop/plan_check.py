"""PlanCheck — plan_check 方法族协作对象, 从 engine.py 下沉.

P1 slim-down: 22 个 plan_check 方法从 engine.py 迁入. 原以 mixin class 多继承
接入, 去 mixin 阶段6 改为普通类 PlanCheck, 引擎经组合持有 self._plan_checker
= PlanCheck(self), 保留同名薄委托方法 → 既有 self.method() 调用点零改动.

方法经转发访问 engine 状态 (self 未定义属性落回 engine):
字段如 _plan_check_patterns / _plan_check_history / _plan_check_warnings /
_plan_check_last_result / _scene_tag_extra_keywords / _iteration / workspace,
方法如 _maybe_clarify / _llm_chat / _build_memory_text / _build_kb_text /
_apply_block_patches / _trim_to_budget 等均转发到引擎.

设计关键 (ponytail):
- 方法体大量读写引擎状态(字段+方法) → 「全属性转发」: __getattr__ 把未定义
  属性读转发到 engine, __setattr__ 转发写. 字段/方法留引擎不复制.
- 防递归: __getattr__ 用 object.__getattribute__ 直达 engine 实例属性
  (engine==self 的测试 mock 场景不递归); __setattr__ 在 engine is self 时直写
  实例 dict.
- 对 engine.py 模块级符号用方法内 lazy import (_get_math_signals), 避免 circular.
- 协作对象不额外持有业务状态 (除 engine 引用).

调用点: cognitive_loop 调 _load_plan_check_patterns(); engine_act._plan
调 _build_plan_prompt / _parse_plan / _override_plan_mode / _plan_check_and_refine.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from typing import Any

from huginn.autoloop._facade import EngineFacade
from huginn.autoloop.signals import routing_surprise
from huginn.memory.longterm import load_stable_principles
from huginn.utils.json_extract import extract_json_text
from huginn.utils.runtime import HUGINN_DIR_NAME

logger = logging.getLogger(__name__)

def _get_math_signals():
    """Delayed import to avoid circular dependency (engine imports PlanCheck)."""
    from huginn.autoloop.engine import _MATH_SIGNALS

    return _MATH_SIGNALS

# planner 提示教的 MODE 候选基线. 不要在源码里把它改成 f-string 占位符:
# contract_audit 的 workflow 面用正则静态解析本文件里 "MODE:" 后的候选枚举
# (huginn/cli/contract_audit.py), 占位符会让该门禁看不到任何候选 mode → 变红.
# VISTA 借鉴的两个实验 mode 由 _plan_mode_enum/_extend_mode_enum 在运行时按 flag 追加.
_BASE_MODE_ENUM = "coder|workflow|explore|skill|visual_inspect"

# _pp_scan 的文本字段前缀 → (状态键, 是否小写化). 顺序与旧 elif 链一致.
_PLAN_TEXT_FIELDS: dict[str, tuple[str, bool]] = {
    "MODE:": ("mode", True),
    "DESCRIPTION:": ("description", False),
    "SKILL:": ("skill_name", False),
    "PREDICTION:": ("prediction", False),
}

# _parse_plan 的可选字段映射 (有序: 决定 plan dict 的键插入顺序).
_PLAN_OPTIONAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("skill_name", "skill"),
    ("prediction", "expected_prediction"),
    ("slots", "prediction_inputs"),
    ("formula", "plan_formula"),
    ("files", "target_files"),
)

# ---------------------------------------------------------------------------
# plan prompt 拼接片段 (_build_plan_prompt 外提; 文本内容与拼接顺序逐字节不变)
# ---------------------------------------------------------------------------

_PC_PROTOCOL_CHECK_PROMPT = """Protocol completeness check (RCBench failure mode: experimental protocol mismatch):
Before finalizing, verify your plan covers all necessary steps:
- For DFT: structure optimization BEFORE property calculation? Convergence test (encut/kpoints)?
- For MD: equilibration BEFORE production run? Timestep appropriate for the system?
- For analysis: raw data processing BEFORE interpretation? Reference comparison?
- Are computational parameters appropriate for the target property (e.g. HSE06 for band gap, not PBE)?
- Cross-check against domain knowledge above: any known methodological requirements?
If a step is missing, add it to DESCRIPTION.

"""

_PC_SYMBOLIC_MATH_HINT_PROMPT = """When the hypothesis involves a PDE / variational principle / curved
geometry, consider the symbolic_math_tool actions listed in the math
depth block above — but numerical solvers are equally valid.

"""

_PC_ALIGNMENT_GATE_PROMPT = """Alignment gate (plan_check will reject the plan if this fails):
- Name the exact quantity / structure / assertion the hypothesis makes, and point to
  the step in DESCRIPTION that measures or computes THAT thing.
- A numeric scan (capacity / error / scaling) only tests a claim that is itself numeric.
  If the hypothesis asserts something else (topological / algebraic / existence /
  uniqueness), either pick a MODE + workflow that computes that object symbolically, or
  restate the hypothesis to a numerically testable form that the plan actually implements.
- Never reuse a previous round's DESCRIPTION when the hypothesis has changed. A plan that
  ignores the current hypothesis is invalid even if the code runs and returns numbers.

"""

# contract_audit 的 workflow 面用正则静态解析本文件里 "MODE:" 后的候选枚举
# (huginn/cli/contract_audit.py), 故下面字面量必须留在源码, 不要改成 f-string 占位符.
_PC_OUTPUT_FORMAT_PROMPT = """Respond in this exact format:
MODE: <coder|workflow|explore|skill|visual_inspect>
DESCRIPTION: <brief description of what to do>
SKILL: <composite skill name, only if MODE is skill>
FILES: <OPTIONAL, comma-separated repo-relative paths or globs you intend to modify this round, e.g. "src/a.py, tests/test_a.py". Used only for an intent-scope reward audit (changes outside this set are flagged). Omit if you don't yet know which files you'll touch.>
PREDICTION: <what you expect the result to look like — be specific: "energy ~ -X eV", "converges in ~N steps", "band gap ~X eV". This prediction will be compared against actual results to measure surprise.>
SLOTS: <OPTIONAL, only for method/numerical objectives where inputs are known BEFORE execution. List the given input params and the formula you will apply, e.g. "a=3; b=4; formula=RMS(a,b)". CRITICAL: only plan-time known GIVENS and the operation — NEVER write the predicted output value here. Leaking the answer here corrupts the surprise signal.>
"""

def _pc_task_prompt(
    hypothesis: str, context: dict[str, Any], extra_mode_lines: str
) -> str:
    """role/task 段: 假设 + 上下文 + mode 说明 + 实验 mode 附加行."""
    return f"""Given the hypothesis: "{hypothesis}"

Context:
{json.dumps(context, indent=2, ensure_ascii=False)[:1000]}

Choose ONE mode and describe the plan:
- coder: modify code/files to fix or improve something
- workflow: run a computational simulation pipeline
- explore: search a design space for optimal parameters
- skill: use a pre-built composite skill pipeline (band structure, mechanical properties, MD, etc.)
- visual_inspect: interactively inspect visual data (zoom into chart region, measure data points, annotate structure). Use this when you need to examine previous results more carefully before deciding next steps. Available actions: zoom, measure, annotate, compare.
{extra_mode_lines}
"""

def _pc_plan_body(
    hypothesis: str, context: dict[str, Any], extra_mode_lines: str
) -> str:
    """prompt 主体: role/task + rules(协议完整性/符号数学/对齐门) + output format."""
    return (
        _pc_task_prompt(hypothesis, context, extra_mode_lines)
        + _PC_PROTOCOL_CHECK_PROMPT
        + _PC_SYMBOLIC_MATH_HINT_PROMPT
        + _PC_ALIGNMENT_GATE_PROMPT
        + _PC_OUTPUT_FORMAT_PROMPT
    )

def _pc_retrieval_blocks(
    pc: Any, hypothesis: str
) -> tuple[str, str, str, str, str, str]:
    """KB / KG / 长期记忆 / PM / metacog / 稳定原则 六块 (无内容时为空串)."""
    # 同 hypothesize: 用 hypothesis 串检索 KB, 把参考块喂给 planner
    kb_block = pc._build_kb_text(query=hypothesis)
    if kb_block:
        kb_block = f"\n{kb_block}\n"
    # KG 检索: 用 hypothesis 当 query, 看看已有实体里有没有相关的
    kg_block = pc._build_kg_text(query=hypothesis)
    if kg_block:
        kg_block = f"\n{kg_block}\n"
    # 长期记忆检索 (同 hypothesize)
    mem_block = pc._build_memory_text(query=hypothesis)
    if mem_block:
        mem_block = f"\n{mem_block}\n"
    # C2: PM 层 trajectory_match 召回 (同 hypothesize, 极限模式才开)
    pm_block = pc._build_pm_text()
    if pm_block:
        pm_block = f"\n{pm_block}\n"
    # C2: metacog 信号注入 (同 hypothesize) — target_chain + prospective
    metacog_block = pc._build_metacog_block(include_prospective=True)
    if metacog_block:
        metacog_block = f"\n{metacog_block}\n"
    # H0: stable_principles 注入 (同 hypothesize, 修 P3 断链)
    try:
        _principles = load_stable_principles()[:5]
    except Exception:  # 防御: 原则加载失败用空表
        _principles = []
    principles_block = "\n".join(f"- {p}" for p in _principles) if _principles else ""
    if principles_block:
        principles_block = (
            f"\n### Stable Principles (procedural memory)\n{principles_block}\n"
        )
    return kb_block, kg_block, mem_block, pm_block, metacog_block, principles_block

def _pc_world_model_block(pc: Any, hypothesis: str) -> str:
    """P1 Task 5: world model 预测注入 (懒路, toggle HUGINN_WORLD_MODEL)."""
    block = pc._build_world_model_block(hypothesis)
    if block:
        block = f"\n### World Model Prediction (analogy-based)\n{block}"
    return block

def _pc_visual_block(pc: Any) -> str:
    """视觉基元注入 (同 hypothesize)."""
    block = getattr(pc, "_last_visual_context", "")
    if block:
        block = f"\n### Visual Primitives (from last tool output)\n{block}\n"
    return block

def _pc_math_block(pc: Any, hypothesis: str, context: dict[str, Any]) -> str:
    """条件化 math_block (同 hypothesize)."""
    hyp_blob = (
        hypothesis.lower() + json.dumps(context, ensure_ascii=False).lower()[:500]
    )
    # P21/dsh: 相关任务链 → 降级引导 (math_block 不注入).
    related_chain = pc._is_related_chain(context.get("changed_files", []))
    return (
        pc._MATH_DEPTH_PROMPT_BLOCK
        if not related_chain and any(s in hyp_blob for s in _get_math_signals())
        else ""
    )

def _pc_acceptance_block(pc: Any) -> str:
    """当前交付物验收缺口 → plan prompt 块 (修复目标就写在这里).

    ws35 实测: 门只把缺口报给了 decider (下一步选什么), 而"怎么修"由 plan 编写;
    plan prompt 里没有缺口时, 计划仍按自造键名走, 再被判红 (ws35 iter 8 的
    results.json 用 h1..h5 嵌套 → 17 个必需键全缺). 缺口必须到作者手里.
    """
    from huginn.autoloop.deliverables import check_manifest, render_acceptance_gap

    gate_fn = getattr(pc, "_deliverables_ok", None)
    if callable(gate_fn):
        gate = gate_fn()
    else:
        manifest = getattr(pc, "_deliverable_manifest", None) or []
        if not manifest:
            return ""
        gate = check_manifest(
            getattr(pc, "workspace", None),
            manifest,
            getattr(pc, "_deliverable_contract", None),
        )
    gap = render_acceptance_gap(gate)
    if not gap:
        return ""
    return (
        "\nAcceptance (deterministic engine gate; the plan must close these gaps):\n"
        f"{gap}\n"
    )


def _pc_learning_blocks(pc: Any, hypothesis: str) -> tuple[str, str]:
    """evolution 学到的 skills + prompt patches 提示行."""
    # Inject learned skills + prompt patches from evolution engine.
    # This is the "use what you learned" half of the Learn→Plan loop.
    skill_hints = ""
    patch_hints = ""
    try:
        evolution = pc._get_evolution()
        skills = evolution.get_relevant_skills(hypothesis)
        if skills:
            skill_lines = [f"  - {s.name}: {s.description}" for s in skills[:3]]
            skill_hints = (
                "\nLearned skills (from past iterations):\n"
                + "\n".join(skill_lines)
                + "\n"
            )
        patches = evolution.get_prompt_patches()
        if patches:
            patch_hints = (
                "\nLearned patches:\n"
                + "\n".join(f"  - {p}" for p in patches[:3])
                + "\n"
            )
    except Exception:  # 防御: 技能/补丁拉取失败不阻断提示
        logger.warning(
            "error in _build_plan_prompt: evolution skill/patch fetch failed",
            exc_info=True,
        )
    return skill_hints, patch_hints

def _pc_composite_block(hypothesis: str) -> str:
    """条件化 composite skills 列表 (仿真/计算类 hypothesis 才注入)."""
    # Inject matching composite skills — lets the LLM pick a pre-built
    # multi-tool pipeline instead of improvising from scratch.
    # 条件化: 只在 hypothesis 涉及仿真/计算/材料性质时注入, coder-only
    # 任务不需要 composite skill 列表. 节省 ~500 tokens.
    composite_block = ""
    hyp_lower = hypothesis.lower()
    _workflow_signals = (
        "workflow",
        "simulation",
        "band",
        "dos",
        "phonon",
        "mechanical",
        "thermal",
        "optical",
        "dft",
        "vasp",
        "lammps",
        "md ",
        "structure",
        "property",
        "energy",
        "convergence",
        "optimize",
        "calc",
    )
    if any(s in hyp_lower for s in _workflow_signals):
        try:
            from huginn.skills.composite import _ensure_registered
            from huginn.skills.registry import SkillRegistry

            _ensure_registered()
            matches = SkillRegistry.search(hypothesis)
            if not matches:
                matches = SkillRegistry.get_all_definitions()
            if matches:
                lines = [s.to_prompt() for s in matches[:4]]
                composite_block = (
                    "\nAvailable composite skills (prefer these over manual workflow):\n"
                    + "\n\n".join(lines)
                    + "\n"
                )
        except Exception:  # 防御: 组合技能查找失败跳过
            logger.debug("composite skill lookup failed", exc_info=True)
    return composite_block

def _pc_pipeline_block(pc: Any) -> str:
    """Pipeline 建议: 基于 provenance 规则推荐下一步工具."""
    # 42 条领域规则, 零 LLM 调用. 让 plan 知道"这类任务通常下一步是 X".
    pipeline_block = ""
    try:
        from huginn.provenance.pipeline import SimulationPipeline

        pipeline = SimulationPipeline(pc.kg.root if hasattr(pc.kg, "root") else None)
        # 用上一轮的 execution_result 触发 suggest_next
        last_result = getattr(pc, "_last_execution_result", None)
        if last_result and isinstance(last_result, dict):
            tool_name = last_result.get("_tool_name", "")
            suggestions = pipeline.suggest_next(
                tool_name=tool_name,
                tool_input=last_result.get("_tool_input", {}),
                tool_output=last_result.get("result", last_result),
            )
            if suggestions:
                s_lines = [
                    f"  - {s.tool_hint}: {s.description}" for s in suggestions[:3]
                ]
                pipeline_block = (
                    "\nPipeline suggestions (based on provenance):\n"
                    + "\n".join(s_lines)
                    + "\n"
                )
    except Exception:  # 防御: 建议失败不阻塞流程
        logger.debug(
            "best-effort op failed", exc_info=True
        )  # pipeline 是 advisory, 失败不阻塞
    return pipeline_block

# 场景关键词表 (快路径). 顺序即优先级: dft > md > workflow, 之后按 mode 判
# skill, 再 fem; 自动发现的关键词 (_scene_tag_extra_keywords) 走最后.
_SCENE_KEYWORDS_HEAD: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "dft",
        ("vasp", "scf", "band", "dos", "dft", "qe", "cp2k", "gaussian", "orca"),
    ),
    (
        "md",
        (
            "lammps",
            "molecular dynamics",
            "minimize",
            "nvt",
            "npt",
            "md ",
            "gromacs",
            "openmm",
        ),
    ),
    ("workflow", ("workflow", "pipeline", "orchestrat")),
)
_SCENE_KEYWORDS_TAIL: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fem", ("fenics", "abaqus", "comsol", "openfoam", "fem", "elmer")),
)

# scene_tag 自动发现的停用词 (英文功能词 + 高频动作词)
_PC_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "run",
        "then",
        "calc",
        "calculate",
        "using",
        "use",
        "plan",
        "step",
    }
)

def _match_scene_keywords(
    desc: str, table: tuple[tuple[str, tuple[str, ...]], ...]
) -> str | None:
    """按表顺序匹配关键词; 命中返回标签."""
    for label, keywords in table:
        if any(kw in desc for kw in keywords):
            return label
    return None

def _collect_other_scene_descs(history: Any) -> list[str]:
    """收集 scene='other' 的 plan descriptions (小写)."""
    out: list[str] = []
    for c in history:
        snapshot = c.get("plan_snapshot") or {}
        if c.get("scene_tag") == "other" and snapshot.get("description"):
            out.append(snapshot["description"].lower())
    return out

def _count_plan_ngrams(descs: list[str]) -> tuple[Counter[str], Counter[str]]:
    """unigram (>=4 chars, 去停用词) + bigram 词频."""
    word_counts: Counter[str] = Counter()
    bigram_counts: Counter[str] = Counter()
    for desc in descs:
        words = [
            w for w in re.findall(r"[a-z][a-z0-9_]{3,}", desc) if w not in _PC_STOPWORDS
        ]
        for word in words:
            word_counts[word] += 1
        # bigram: 相邻两词组合
        for i in range(len(words) - 1):
            bigram_counts[f"{words[i]} {words[i + 1]}"] += 1
    return word_counts, bigram_counts

def _apply_discovered_tags(
    target: dict[str, set[str]], counts: Counter[str], limit: int
) -> None:
    """把 >=3 次的高频 n-gram 写进 extra_keywords (label 用下划线连接)."""
    for gram, count in counts.most_common(limit):
        if count < 3:
            continue
        label = f"auto_{gram.replace(' ', '_')}"
        target.setdefault(label, set()).add(gram)

# 抓 "<number> <unit>" 量, e.g. "210 GPa" / "1.5e3 kg/m3"
_DIM_QUANTITY_RE = re.compile(
    r"([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s+([A-Za-z][A-Za-z0-9/\^\-\*\.\(\)]+)"
)

def _dimensional_text(plan: Any, hypothesis: str) -> str:
    """拼 plan + hypothesis 文本 (取 description/prediction 字段)."""
    parts = [hypothesis or ""]
    for k in ("description", "expected_prediction", "prediction"):
        v = plan.get(k) if isinstance(plan, dict) else None
        if isinstance(v, str) and v:
            parts.append(v)
    return "\n".join(parts)

def _dimensional_line_warning(validator: Any, line: str) -> str | None:
    """单行等式量纲校验; 不可解析或量纲一致时返回 None."""
    if "=" not in line:
        return None
    lhs, rhs = line.split("=", 1)
    lhs_qs = [f"{m[0]} {m[1]}" for m in _DIM_QUANTITY_RE.findall(lhs)]
    rhs_qs = [f"{m[0]} {m[1]}" for m in _DIM_QUANTITY_RE.findall(rhs)]
    if not lhs_qs or not rhs_qs:
        return None
    try:
        result = validator.check_equation(
            lhs_qs, rhs_qs, equation_name=line.strip()[:80]
        )
    except Exception:  # 防御: 解析失败跳过该条继续
        logger.debug("best-effort op failed", exc_info=True)
        return None
    if result.consistent:
        return None
    return (
        f"dimensional inconsistency: '{line.strip()[:80]}' "
        f"LHS={result.lhs_dimensions} RHS={result.rhs_dimensions}"
    )

def _pcp_render_prompt(
    hypothesis: str,
    plan: dict[str, Any],
    failure_modes: str,
    scene: str,
    similar_text: str,
    dim_warnings_text: str,
) -> str:
    """反向规划识别器 prompt 模板."""
    return f"""你是反向规划识别器 (KRCL 启发). 判断以下 plan 执行后能否达成 hypothesis.

# 目标 (hypothesis)
{hypothesis}

# 当前 plan
MODE: {plan.get("mode", "coder")}
DESCRIPTION: {plan.get("description", "")}
PREDICTION: {plan.get("expected_prediction", "N/A")}

# 已知失败模式 (避免重蹈覆辙)
{failure_modes or "N/A"}

# 同场景历史失败 (scene={scene}, 跨 run 积累)
{similar_text}

# 量纲预检查警告 (L2 dimensional pre-check, v6 G57)
{dim_warnings_text}

# 任务
判断这个 plan 执行后能否达成 hypothesis. 严格检查:
- MODE 是否匹配任务类型 (coder 写代码 / workflow 跑流程 / explore 探索 / skill 复合技能)
- DESCRIPTION 是否覆盖 hypothesis 的关键要求
- PREDICTION 是否可验证 (能跑出数值/结构/代码对比)
- 是否遗漏必要前置步骤 (如 band 前需 SCF / MD 前需 minimize / elastic 前需 relax)
- 是否重复了"同场景历史失败"里列出的坑
- 量纲预检查有警告时, 把它列入 risks

# 判定边界 (硬性, 误判会卡死整个循环)
- DESCRIPTION 是**计划文本**, 不是源码. 代码/脚本在 execute 阶段才写出.
  **不得**因为 DESCRIPTION 里没有导入语句、可运行源码、print 语句、完整函数体
  而判 invalid —— ws35 实测此误判导致 13 连拒、循环空转.
- 只有以下情况才判 invalid: 计划与 hypothesis 不对齐 (测的不是同一个量)、
  缺必要前置步骤、PREDICTION 不可验证、方法在本环境不可执行.
- 判 invalid 必须能在 missing_steps 里给出**可执行**的补充项; 给不出来就判 valid.

输出 JSON (不要其他文本):
{{
  "is_valid": true 或 false,
  "confidence": 0.0 到 1.0 (对判断的置信度, 1.0=非常确定, 0.5=模棱两可, 0.0=完全没把握),
  "reason": "为什么 valid / invalid",
  "missing_steps": ["如果 invalid, 缺少哪些步骤"],
  "risks": ["潜在风险"]
}}"""

def _extract_json_object(text: str) -> tuple[str, str]:
    """括号配平法抓第一个完整 JSON 对象; 返回 (对象文本, 错误原因).

    薄封装: 实现见 huginn/utils/json_extract.py, 这里保留原二元组契约.
    """
    blob, err = extract_json_text(text, prefer="first", openers=("{",))
    return blob, f"{err}, skip" if err else ""

def _rp_render_prompt(
    plan: dict[str, Any],
    hypothesis: str,
    check: dict[str, Any],
    scene: str,
    few_shot_block: str,
) -> str:
    """plan 重建 prompt 模板."""
    return f"""之前的 plan 未通过反向校验. 根据反馈重新生成.

# 目标
{hypothesis}

# 之前的 plan
MODE: {plan.get("mode", "coder")}
DESCRIPTION: {plan.get("description", "")}

# 校验反馈
reason: {check.get("reason", "")}
missing_steps: {check.get("missing_steps", [])}
risks: {check.get("risks", [])}

# 同场景成功示例 (scene={scene}, 跨 iteration 积累, 仅供参考结构)
{few_shot_block}

# 任务
根据反馈重新生成 plan. 参考成功示例的结构 (不要照抄内容). 严格按格式输出:
MODE: <coder|workflow|explore|skill|visual_inspect>
DESCRIPTION: <brief description>
SKILL: <composite skill name, only if MODE is skill>
PREDICTION: <预期结果, 用于后续 validate 对比>"""

class PlanCheck(EngineFacade):
    """plan_check 方法族协作对象.

    未定义的属性读写经 __getattr__/__setattr__ 转发到 self.engine —
    引擎字段/方法不会被复制两份, 方法体零改动、行为完全等价.
    """

    def _build_subgoal_block(self) -> str:
        """从 agent 或 self 上读 sub_goals, 注入到 prompt."""
        sgs = getattr(self, "_sub_goals", None) or []
        if not sgs:
            return ""
        lines = ["\n### Active Sub-goal Constraints (from /subgoal)"]
        for i, sg in enumerate(sgs, 1):
            lines.append(f"{i}. {sg}")
        lines.append("### End Sub-goal Constraints\n")
        return "\n".join(lines)

    def _plan_mode_enum(self) -> str:
        """计划格式里的 MODE 枚举. VISTA 借鉴的两个 mode 仅对应 flag 开时才列出.

        默认全关 → 返回值与历史硬编码完全一致 (_BASE_MODE_ENUM), 提示词向后兼容.
        """
        modes = _BASE_MODE_ENUM.split("|")
        try:
            from huginn.feature_flags import FeatureFlags

            ff = FeatureFlags.shared()
            if ff.is_enabled("trace_inspect"):
                modes.append("trace_inspect")
            if ff.is_enabled("visual_frame_memory"):
                modes.append("frame_inspect")
        except Exception:  # 防御: flag 层异常 → 回落历史枚举
            logger.debug("plan mode enum flag read failed", exc_info=True)
        return "|".join(modes)

    def _extend_mode_enum(self, prompt: str) -> str:
        """把提示词里的基线 MODE 枚举按 feature flag 展开 (flag 关时原样返回).

        提示词模板里保留字面量 ``MODE: <coder|workflow|...>`` 供 contract_audit
        静态门读取; 这里只在 flag 开启时把它替换成含新 mode 的枚举.
        """
        enum = self._plan_mode_enum()
        if enum == _BASE_MODE_ENUM:
            return prompt
        return prompt.replace(f"MODE: <{_BASE_MODE_ENUM}>", f"MODE: <{enum}>")

    def _plan_extra_mode_lines(self) -> str:
        """VISTA 借鉴 mode 的说明行; 对应 flag 关时为空串 (提示词与历史一致)."""
        lines: list[str] = []
        try:
            from huginn.feature_flags import FeatureFlags

            ff = FeatureFlags.shared()
            if ff.is_enabled("trace_inspect"):
                lines.append(
                    "- trace_inspect: recall your OWN past execution traces "
                    "(process-level) by keyword/tool/intent. Read-only. Put a JSON object "
                    'in DESCRIPTION, e.g. {"query":"timeout","tool":"code_lab","limit":8}. '
                    "Use it to reuse earlier runs' numbers/errors instead of re-running."
                )
            if ff.is_enabled("visual_frame_memory"):
                lines.append(
                    "- frame_inspect: re-view a previously captured frame losslessly. "
                    'DESCRIPTION JSON: {"action":"view|region|pixels","frame_id":N,'
                    '"box":[x0,y0,x1,y1],"points":[[x,y],...],"normalized":false}. '
                    "region crops the ORIGINAL pixels faithfully; pixels returns exact RGB "
                    "— use it to read values off a figure without re-rendering."
                )
        except Exception:  # 防御: flag 层异常 → 不追加
            logger.debug("plan extra mode lines failed", exc_info=True)
        return "\n".join(lines)

    def _build_plan_prompt(self, hypothesis: str, context: dict[str, Any]) -> str:
        (
            kb_block,
            kg_block,
            mem_block,
            pm_block,
            metacog_block,
            principles_block,
        ) = _pc_retrieval_blocks(self, hypothesis)
        world_model_block = _pc_world_model_block(self, hypothesis)
        visual_block = _pc_visual_block(self)
        math_block = _pc_math_block(self, hypothesis, context)

        skill_hints, patch_hints = _pc_learning_blocks(self, hypothesis)
        composite_block = _pc_composite_block(hypothesis)
        pipeline_block = _pc_pipeline_block(self)

        # VISTA 借鉴: 两个实验 mode 的说明 (flag 关时为空串); 枚举由
        # _extend_mode_enum 在返回前按 flag 展开 (模板保留基线字面量).
        extra_mode_lines = self._plan_extra_mode_lines()

        # objective 原文 + 契约独立成块: plan 阶段的 context 只截 1000 字符,
        # 7KB 任务书 (H1–H5 定义/阈值/交付物键名) 到不了模型 → 模型按自造的
        # 假设规划, 于是 plan_check 反复判 misalign (ws35 实测 13 连拒).
        from huginn.autoloop.deliverables import objective_block_for

        blocks = self._apply_block_patches(
            [
                ("body", _pc_plan_body(hypothesis, context, extra_mode_lines)),
                ("objective", objective_block_for(self)),
                ("acceptance", _pc_acceptance_block(self)),
                ("math", math_block),
                ("kg", kg_block),
                ("visual", visual_block),
                ("kb", kb_block),
                ("principles", principles_block),
                ("world_model", world_model_block),
                ("mem", mem_block),
                ("pm", pm_block),
                ("metacog", metacog_block),
                # PMK 三路立场 + 一致性标签 (HUGINN_PMK_INJECT off 或全空时返空串)
                ("pmk", self._build_pmk_block(context)),
                ("skill", skill_hints + patch_hints),
                ("composite", composite_block),
                ("pipeline", pipeline_block),
                ("subgoal", self._build_subgoal_block()),
                ("ctx_hint", self._plan_context_hint()),
            ],
            "plan",
        )
        return self._extend_mode_enum(self._trim_to_budget(blocks, phase="plan"))

    def _plan_context_hint(self) -> str:
        """B: 把上下文信号转成 plan prompt 提示文本 (软路由).

        让 LLM 知道当前图状态/失败次数/refine 次数, 倾向选验证型 mode.
        硬路由在 _override_plan_mode 里做.
        """
        hints = []
        # 割点节点需要双覆盖 → 倾向选能跑验证的 mode
        try:
            current_hyp = getattr(self, "_current_hyp_id_for_plan", None)
            if current_hyp and self.hypothesis_graph.needs_dual_coverage(current_hyp):
                hints.append(
                    "CRITICAL: 当前假设是图的关键割点, 需要双模态验证. "
                    "优先选 workflow/skill 跑符号验证, 不要只选 coder."
                )
        except Exception:  # 防御: 双覆盖提示失败跳过
            logger.debug("dual coverage hint skipped", exc_info=True)
        # 连续失败 → 倾向换方向
        cf = getattr(self, "_consecutive_failures", 0)
        if cf >= 3:
            hints.append(
                f"WARNING: 已连续失败 {cf} 次. 考虑 explore 换参数空间, "
                "或换一个完全不同的方法论."
            )
        # refine 次数多 → 假设可能方向错
        rc = getattr(self, "_refine_count", 0)
        if rc >= 3:
            hints.append(f"NOTE: 已 refine {rc} 次. 如果再失败可能需要 pivot 换方向.")
        # surprise 高 → 预测误差大, 倾向 explore 重新假设.
        # v31: 读秩归一信号 routing_surprise() —— 原始 surprise 在 jaccard 回落时
        # 饱和于 1.0, 会把"恒定"误当"高", 每轮都提示 explore.
        surprise = routing_surprise(self)
        if surprise > 0.5:
            hints.append(
                f"NOTE: 预测误差大 (surprise={surprise:.2f}). "
                "预测与实际差异显著, 考虑 explore 重新假设或换方法论."
            )
        if not hints:
            return ""
        return "\n\nContext signals:\n" + "\n".join(f"- {h}" for h in hints) + "\n"

    def _override_plan_mode(self, plan: dict[str, Any]) -> dict[str, Any]:
        """B: 硬路由 — 在 LLM 选完 mode 后, 根据硬性规则覆盖.

        只在极端情况覆盖, 不破坏 LLM 的常规选择:
        - needs_dual_coverage=True → mode 不能是 coder (必须能跑验证)
        - consecutive_failures >= 5 或 surprise > 0.9 → 强制 explore

        覆盖记到 PhaseGateState.history 补审计缺口 (reviewer="auto_router"),
        plan["override_reason"] 留结构化标记供调用方读取.

        ponytail: 只覆盖极端情况, 常规让 LLM 决定.
        budget tier 已由 _check_budget 处理, 这里不重复.
        升级: campaign 队列状态 (queue 满则 workflow 批量验证).
        """
        current_mode = plan.get("mode", "coder")
        current_mode = self._ovr_force_coder(plan, current_mode)
        current_mode = self._ovr_code_experiment(plan, current_mode)
        self._ovr_dual_coverage(plan, current_mode)
        self._ovr_force_explore(plan)
        return plan

    def _ovr_force_coder(self, plan: dict[str, Any], current_mode: str) -> str:
        """写码类目标钉死 coder; 返回(可能更新后的)当前 mode.

        execute 得有"手". explore 模式只有注册了设计空间才真跑, 否则秒回一个
        空 pod (0.0s, 无 script 无数值), 而 report 照旧把 LLM 现编的数字当
        "Results"写出去. 目标/plan 明确要求"写并运行代码"时钉死 coder
        (Write+Bash 真执行真取数), 不让 explore 接管.
        """
        if current_mode == "coder" or not self._asks_to_write_and_run_code(plan):
            return current_mode
        plan["mode"] = "coder"
        plan["override_reason"] = "code_task_force_coder"
        plan["description"] = (
            f"[auto-routed: 写码目标需真实执行] {plan.get('description', '')}"
        )
        logger.info("override mode %s→coder: 目标要求写并运行代码", current_mode)
        self._log_plan_override("code_task_force_coder", "目标要求写并运行代码")
        return "coder"

    def _ovr_code_experiment(self, plan: dict[str, Any], current_mode: str) -> str:
        """计算实验目标不套"物理管线"模式; 返回(可能更新后的)当前 mode.

        workflow 只认 cfd/fea/qc/symbolic/dft 几类模板, 把纯 ML/数学命题(如
        "解空间刚性")塞进去 → 报告被写成 DFT workflow 的 Methods/Results
        (run88 实测域漂移). 目标是代码实验时改走真实执行路径: coder
        (Write+Bash 真写码真执行) 而非再叫一次 code_lab —— execute 的实验
        快路径已经先试过 Code Lab, 这里换一条真实执行路径推进, 不重复空转.
        """
        if current_mode != "workflow" or not self._is_code_experiment_plan(plan):
            return current_mode
        plan["mode"] = "coder"
        plan["override_reason"] = "code_experiment_not_physics_workflow"
        plan["description"] = (
            f"[auto-routed: 代码实验走真实执行] {plan.get('description', '')}"
        )
        logger.info("override mode %s→coder: 目标是代码实验, 非物理管线", current_mode)
        self._log_plan_override(
            "code_experiment_not_physics_workflow", "代码实验不套物理 workflow 模板"
        )
        return "coder"

    def _ovr_dual_coverage(self, plan: dict[str, Any], current_mode: str) -> None:
        """割点节点: 强制非 coder mode (需能跑验证).

        代码实验的"验证模态"就是 Code Lab 真跑, 故代码实验走 code_lab,
        其余走 workflow.
        """
        try:
            current_hyp = getattr(self, "_current_hyp_id_for_plan", None)
            if not (
                current_hyp
                and self.hypothesis_graph.needs_dual_coverage(current_hyp)
                and current_mode == "coder"
            ):
                return
            dual_mode = (
                "code_lab" if self._is_code_experiment_plan(plan) else "workflow"
            )
            plan["mode"] = dual_mode
            plan["override_reason"] = "cut_vertex_dual_coverage"
            plan["description"] = (
                f"[auto-routed: 割点需双覆盖] {plan.get('description', '')}"
            )
            logger.info(
                "override mode coder→%s for cut vertex %s", dual_mode, current_hyp
            )
            self._log_plan_override(
                "cut_vertex_dual_coverage", f"割点 {current_hyp} 需双覆盖"
            )
        except Exception:  # 防御: 双覆盖改写失败跳过
            logger.debug("dual coverage override skipped", exc_info=True)

    def _ovr_force_explore(self, plan: dict[str, Any]) -> None:
        """连败/surprise 强制 explore (合并条件, 共享覆盖路径)."""
        cf = getattr(self, "_consecutive_failures", 0)
        # v31: 硬阈值改读秩归一信号. 原始 surprise 在 jaccard 回落时饱和 1.0 ⇒
        # 恒 >0.9 ⇒ 每轮强制 explore(路由退化). 秩下恒定信号 ≈0.5, 不再误触发.
        surprise = routing_surprise(self)
        explore_reasons: list[str] = []
        if cf >= 5:
            explore_reasons.append(f"连续失败{cf}次")
        if surprise > 0.9:
            explore_reasons.append(f"surprise={surprise:.2f}")
        if not explore_reasons or plan["mode"] == "explore":
            return
        reason = "+".join(explore_reasons)
        plan["mode"] = "explore"
        plan["override_reason"] = "force_explore"
        plan["description"] = f"[auto-routed: {reason}] {plan.get('description', '')}"
        logger.info("override mode →explore: %s", reason)
        self._log_plan_override("force_explore", reason)

    # 强标记: 明确"写并运行代码"的措辞. 不用裸 "python"/"代码" 防误伤
    # (workflow/explore 目标里出现 python 很常见, 不该被强行改道).
    _CODE_TASK_MARKERS = (
        "写并运行",
        "编写并运行",
        "写代码",
        "编写代码",
        "运行代码",
        "现写",
        "写脚本",
        "写一个脚本",
        "写个脚本",
        "python 脚本",
        "python脚本",
        "write and run",
        "write a script",
        "run the code",
        "write code and run",
    )

    _NEGATION_WINDOW_RE = re.compile(
        r"(?:不要|禁止|勿|无需|不必|不得|do\s+not|don['’]?t|without|no\s+need)"
        r"[^,.;。；\n]{0,40}"
    )

    @classmethod
    def _strip_negated_windows(cls, text: str) -> str:
        """剥离否定窗口, 避免"不要写代码"命中"写代码"标记."""
        return cls._NEGATION_WINDOW_RE.sub(" ", text or "")

    def _asks_to_write_and_run_code(self, plan: dict[str, Any]) -> bool:
        """目标/plan 是否明确要求"写并运行代码" (否定句先剥离)."""
        blob = " ".join(
            str(x)
            for x in (
                getattr(self, "_objective", "") or "",
                plan.get("description", "") or "",
            )
        ).lower()
        blob = self._strip_negated_windows(blob)
        return any(m in blob for m in self._CODE_TASK_MARKERS)

    def _is_code_experiment_plan(self, plan: dict[str, Any]) -> bool:
        """目标/plan 是否为"需亲手写代码真跑的计算实验" (命题无关).

        复用引擎的 ``_is_code_experiment`` 词表(单一出处, 不在这里另立一份), 同时
        看 objective 与 plan 描述 —— plan 常只写动作短语, 实验意图待在 objective.
        引擎缺该方法(测试 mock)时降级 False, 不误改路由.
        """
        try:
            fn = self._is_code_experiment  # __getattr__ 转发到引擎
        except AttributeError:
            return False
        if not callable(fn):
            return False
        blob = " ".join(
            str(x)
            for x in (
                getattr(self, "_objective", "") or "",
                plan.get("description", "") or "",
            )
        )
        return bool(fn(blob))

    def _log_plan_override(self, reason_code: str, reason_text: str) -> None:
        """把 mode 覆盖记到 PhaseGateState.history, 补审计缺口.

        _override_plan_mode 之前只 logger.info, PhaseGate.history 不知道发生过
        覆盖. 现在复用 history 通道, reviewer="auto_router" 标记来源.
        失败不阻塞 (测试/无 phase_gate_hook 场景).
        """
        try:
            from huginn.autoloop.phase_gate import (
                PhaseGate,
                get_shared_phase_gate_state,
            )

            state = get_shared_phase_gate_state()
            state.history.append(
                PhaseGate(
                    from_phase="plan",
                    to_phase="plan",
                    status="approved",
                    feedback=f"[auto-routed] {reason_code}: {reason_text}",
                    reviewer="auto_router",
                )
            )
        except Exception:  # 防御: 记录改写失败不阻塞
            logger.debug("log plan override failed", exc_info=True)

    @staticmethod
    def _pp_slots(line: str) -> tuple[list[dict], str]:
        """SLOTS: 行 → (槽位列表, 公式); 解析失败回落空."""
        try:
            from huginn.jepa_slots import parse_slots_line

            return parse_slots_line(line.replace("SLOTS:", "", 1))
        except Exception:  # noqa: BLE001 — 槽解析失败回落空, 不阻塞计划落盘
            return [], ""

    @staticmethod
    def _pp_files(line: str) -> list[str]:
        """FILES: 行 → 文件名列表 (逗号/空格分隔)."""
        return [
            f.strip()
            for f in line.replace("FILES:", "", 1).replace(",", " ").split()
            if f.strip()
        ]

    @staticmethod
    def _pp_scan(response: str) -> dict[str, Any]:
        """逐行扫 MODE/DESCRIPTION/SKILL/PREDICTION/SLOTS/FILES 字段."""
        st: dict[str, Any] = {
            "mode": "coder",
            "description": response.strip(),
            "skill_name": "",
            "prediction": "",
            "slots": [],
            "formula": "",
            "files": [],
        }
        for line in response.split("\n"):
            prefix = next((p for p in _PLAN_TEXT_FIELDS if line.startswith(p)), None)
            if prefix is not None:
                field, lower = _PLAN_TEXT_FIELDS[prefix]
                value = line.replace(prefix, "").strip()
                st[field] = value.lower() if lower else value
                continue
            if line.startswith("SLOTS:"):
                st["slots"], st["formula"] = PlanCheck._pp_slots(line)
            elif line.startswith("FILES:"):
                st["files"] = PlanCheck._pp_files(line)
        return st

    @staticmethod
    def _pp_desc_after_field(response: str) -> str:
        """DESCRIPTION 值落在下一行时的补捞.

        模型常把 ``DESCRIPTION:`` 单独占一行、正文另起（尤其长描述）。
        ``_pp_scan`` 逐行取值 → 拿到空串 → execute 收到空任务（ws14 实测
        plan→execute 缺证据 ['description']，coder 只能瞎逛）。这里把
        ``DESCRIPTION:`` 之后、下一个字段行之前的正文整体当描述。
        """
        lines = response.split("\n")
        idx = next(
            (i for i, l in enumerate(lines) if l.startswith("DESCRIPTION:")), None
        )
        if idx is None:
            return ""
        head = lines[idx].replace("DESCRIPTION:", "", 1).strip()
        if head:
            return head
        tail: list[str] = []
        for line in lines[idx + 1 :]:
            if line.startswith(tuple(_PLAN_TEXT_FIELDS)) or line.startswith(
                ("SLOTS:", "FILES:")
            ):
                break
            tail.append(line)
        return "\n".join(tail).strip()

    def _parse_plan(self, response: str) -> dict[str, Any]:
        """Parse LLM plan response."""
        st = PlanCheck._pp_scan(response)
        plan = {"mode": st["mode"], "description": st["description"]}
        if not str(plan["description"]).strip():
            plan["description"] = PlanCheck._pp_desc_after_field(response)
        for src, dst in _PLAN_OPTIONAL_FIELDS:
            if st[src]:
                plan[dst] = st[src]
        return plan

    # ── KRCL plan check (反向校验 + 闭环重生成) ─────────────────
    # 磐石100 KRCL 启发: 正向神经规划器生成 plan → 反向符号规划识别器校验
    # → 失败反馈重生成. ponytail: 单 LLM 反向校验, 不上 PDDL solver.
    # ceiling: LLM 自校验有同模型盲点, 不如 KRCL 的符号识别器硬.
    # 升级路径: 接 BourbakiTool.check_conservation 做符号反推 (需 Lean 成熟).
    async def _plan_check_and_refine(
        self,
        plan: dict[str, Any],
        hypothesis: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """KRCL 闭环: 反向校验 plan, 失败反馈 LLM 重生成, 超限不阻塞.

        phase-aware: iteration tier (open/medium/light) + plan 复杂度综合
          判定. open 或 skip 跳过校验, medium 只校验不 refine, light 完整闭环.
          复杂 plan 即使 open tier 也升级到 medium (要校验), 简单 plan 即使
          light tier 也降级到 skip (阈值 0.25: explore+20chars desc 能触发,
          coder 的简单任务仍校验因为涉及代码改动).
        自适应: 按 scene_tag 分桶的最近 5 次 success rate 微调 max_refines —
          >=80% 放宽 (-1), <=20% 收紧 (+1), 样本不足走 baseline.
        不暴露: check 结果只存 self._plan_check_last_result / _warnings /
          _plan_check_patterns, 不塞回 plan dict (plan 会进 prompt, 塞了
          等于喂 LLM 元信息).
        失败模式记忆: 失败记到 _plan_check_patterns, 跨 run JSON 持久化,
          下次同场景 plan 来了注入 prompt 让 LLM 重点避开.
        连续失败澄清: 同 scene 连续 3 次失败 + scene != "other" -> 触发
          _maybe_clarify 问用户 (physical_precheck 同款, 不阻塞).

        失败不拦截 (physical_precheck 同款), warning 留痕给 _validate.
        """
        # trivial plan (description 太短) 跳过, 不浪费 LLM 调用
        desc = plan.get("description", "")
        if len(desc) < 20:
            return plan
        tier = self._plan_check_tier(plan)
        if tier in ("open", "skip"):
            logger.debug(
                "plan_check skipped (tier=%s, iter=%d)",
                tier,
                self._iteration,
            )
            return plan
        scene = self._plan_check_scene_tag(plan)
        max_refines = self._plan_check_max_refines(tier, scene)
        return await self._pcr_loop(plan, hypothesis, context, tier, scene, max_refines)

    async def _pcr_loop(
        self,
        plan: dict[str, Any],
        hypothesis: str,
        context: dict[str, Any],
        tier: str,
        scene: str,
        max_refines: int,
    ) -> dict[str, Any]:
        """校验-重生成循环; 失败时按需抬高 max_refines."""
        attempt = 0
        # 用 while 而非 for range(max_refines+1): 失败时可能需要临时抬高预算
        # (见下), for 的 range 在进入循环时就固定了, 抬了也无效.
        while attempt <= max_refines:
            try:
                check = await self._plan_check(plan, hypothesis, context)
            except Exception as e:
                logger.debug("plan_check LLM call failed: %s", e)
                return plan
            self._pcr_record(plan, check, scene)
            if check.get("is_valid", True):
                if self._pcr_pass(check, attempt, max_refines, tier, scene):
                    return plan
            else:
                _done, max_refines = await self._pcr_fail(
                    plan, check, attempt, max_refines, tier, scene, hypothesis
                )
                if _done:
                    return plan
            logger.info(
                "plan_check refining (attempt %d, tier=%s, scene=%s, conf=%.2f): %s",
                attempt,
                tier,
                scene,
                float(check.get("confidence", 0.8)),
                check.get("reason"),
            )
            plan = await self._refine_plan(plan, check, hypothesis, context)
            attempt += 1
        return plan

    def _pcr_record(
        self, plan: dict[str, Any], check: dict[str, Any], scene: str
    ) -> None:
        """给 check 打 scene_tag 并存引擎状态; 成功时存 plan_snapshot 喂 refine."""
        check["scene_tag"] = scene
        if check.get("is_valid", True):
            check["plan_snapshot"] = {
                "mode": plan.get("mode", ""),
                "description": plan.get("description", "")[:200],
            }
        self._plan_check_last_result = check
        self._plan_check_history.append(check)
        # 历史窗口截断, 保留最近 20 条防无限增长
        if len(self._plan_check_history) > 20:
            del self._plan_check_history[: len(self._plan_check_history) - 20]

    def _pcr_pass(
        self,
        check: dict[str, Any],
        attempt: int,
        max_refines: int,
        tier: str,
        scene: str,
    ) -> bool:
        """通过分支: 放行返回 True; 低置信返回 False 继续 refine."""
        confidence = float(check.get("confidence", 0.8))
        # confidence 分级: 低置信通过 (<0.5) 强制 refine 一次, 防 LLM 没看懂
        # 就放行; 高置信直接通过.
        if confidence >= 0.5 or attempt >= max_refines or max_refines == 0:
            logger.info(
                "plan_check passed (attempt %d, tier=%s, scene=%s, conf=%.2f)",
                attempt,
                tier,
                scene,
                confidence,
            )
            # 每 5 次校验触发一次 scene_tag 自动发现 (低成本, 不阻塞)
            if len(self._plan_check_history) % 5 == 0:
                self._discover_scene_tags()
            return True
        logger.info(
            "plan_check passed but low confidence (conf=%.2f), refining",
            confidence,
        )
        return False

    async def _pcr_fail(
        self,
        plan: dict[str, Any],
        check: dict[str, Any],
        attempt: int,
        max_refines: int,
        tier: str,
        scene: str,
        hypothesis: str,
    ) -> tuple[bool, int]:
        """失败分支: 返回 (是否放行, 更新后的 max_refines)."""
        # 失败: 记到 patterns (跨 run 持久化, 喂下次 prompt)
        self._record_plan_check_failure(plan, check, scene, hypothesis)
        confidence = float(check.get("confidence", 0.8))
        # is_valid=False 是明确的「plan 达不成 hypothesis」信号, 必须重建.
        # 低置信只降级记 warning (checker 自己没把握), 仍走 refine 直到用尽.
        if confidence < 0.3:
            self._plan_check_warnings.append(
                f"[{scene}] {check.get('reason', 'unknown')} "
                f"(low_conf={confidence:.2f})"
            )
        # 预算耗尽且已重建过至少一次 -> 放行
        if attempt >= max_refines and attempt >= 1:
            reason = check.get("reason", "unknown")
            self._plan_check_warnings.append(f"[{scene}] {reason}")
            logger.warning(
                "plan_check failed (tier=%s, scene=%s, max_refines=%d): %s",
                tier,
                scene,
                max_refines,
                reason,
            )
            # 连续失败触发主动澄清 (不阻塞, 用户可 force_proceed)
            await self._maybe_trigger_plan_check_clarify(scene, reason, plan)
            return True, max_refines
        if attempt >= max_refines:
            # 自适应 ewma 放宽会把 max_refines 压到 0 (bucket 多数通过时
            # baseline-1), 首次失败就被放行 → 明知达不成 hypothesis 的 plan
            # 原样执行 (run45/46: misalign 永不纠正). 首次失败时强制给一次
            # 重建预算, 通过路径不受影响.
            max_refines += 1
        return False, max_refines

    async def _maybe_trigger_plan_check_clarify(
        self,
        scene: str,
        reason: str,
        plan: dict[str, Any],
    ) -> None:
        """连续 N 次同场景失败 + 场景已知 -> 问用户方向.

        ponytail: 阈值 3 写死, 跟 validation_fail 同款; 不阻塞, 异常吞掉.
        ceiling: 阈值靠拍; "other" 场景没上下文给用户, 直接跳过.
        """
        if scene == "other":
            return
        # 数最近连续失败 (同 scene, 遇到第一条成功就断)
        recent_fails = 0
        for c in reversed(self._plan_check_history):
            if c.get("scene_tag") == scene and not c.get("is_valid", True):
                recent_fails += 1
            else:
                break
        if recent_fails < 3:
            return
        try:
            await self._maybe_clarify(
                "plan_check_fail",
                {
                    "scene": scene,
                    "reason": reason,
                    "consecutive_fails": recent_fails,
                    "plan": plan,
                },
            )
        except Exception as e:
            logger.debug("plan_check clarify failed: %s", e)

    def _plan_check_tier(self, plan: dict[str, Any] | None = None) -> str:
        """phase-aware tier: iteration + plan 复杂度综合判定.

        iteration baseline: open (1-10) / medium (11-30) / light (31+).
        跟 ProgressiveBudget.default() 边界对齐, 但解耦 — budget 关了
        plan_check 仍按 iteration 判 phase.
        plan 复杂度修正 (plan 传入时):
          - 复杂 plan (score >= upgrade_threshold) 即使 open tier 也升级到 medium
          - 简单 plan (score < downgrade_threshold) 即使 light tier 也降级到 skip
        阈值分场景校准: DFT/MD/workflow 各有自己的 success rate, 不会互相带偏.
        ponytail: 阈值从 _plan_check_complexity_thresholds(scene) 取, 不是写死.
        ceiling: 校准靠历史 success rate, 样本不足走默认 0.7/0.25;
          边界跟 ProgressiveBudget 重复一份.
        升级路径: ProgressiveBudget 暴露 tier_of(n) -> label, 这里复用;
                  阈值用 Bayesian 更新而非简单 success rate.
        """
        n = getattr(self, "_iteration", 0)
        if n <= 10:
            base = "open"
        elif n <= 30:
            base = "medium"
        else:
            base = "light"
        if plan is None:
            return base
        complexity = self._plan_check_complexity(plan)
        scene = self._plan_check_scene_tag(plan)
        upgrade_t, downgrade_t = self._plan_check_complexity_thresholds(scene)
        if complexity >= upgrade_t and base == "open":
            return "medium"
        if complexity < downgrade_t and base == "light":
            return "skip"
        return base

    def _plan_check_complexity_thresholds(self, scene: str = "") -> tuple[float, float]:
        """用历史 success rate 自动校准复杂度阈值, 分场景.

        默认: upgrade=0.7 (复杂 plan 升级到 medium), downgrade=0.25 (简单
        plan 降级到 skip).
        分场景校准: 同 scene_tag 的最近 10 条 plan_check 的 success rate
          >=0.8 (一直成功) -> upgrade 放宽到 0.8, downgrade 收紧到 0.15
            (成功率高, 只拦最复杂的, 简单的不轻易跳过)
          <=0.2 (一直失败) -> upgrade 收紧到 0.6, downgrade 放宽到 0.35
            (失败率高, 多拦一些, 简单的也更容易跳过不浪费 LLM)
          样本 <5 走默认, 早期不误判. 未知场景 (scene 无历史) 走全局.
        ponytail: 线性插值, 不上 Bayesian; 阈值钳制在 [0.4, 0.9] / [0.1, 0.4].
        ceiling: 线性插值过于简单; 场景样本不足时回退全局.
        升级路径: 上 Bayesian 更新带先验; 场景用 embedding 聚类而非关键词.
        """
        history = getattr(self, "_plan_check_history", [])
        if scene:
            bucket = [c for c in history if c.get("scene_tag") == scene]
        else:
            bucket = history
        if len(bucket) < 5:
            # 场景样本不足, 回退全局; 全局也不足, 走默认
            if scene and len(history) >= 5:
                bucket = history
            else:
                return (0.7, 0.25)
        recent = bucket[-10:]
        success_rate = sum(1 for c in recent if c.get("is_valid", True)) / len(recent)
        if success_rate >= 0.8:
            return (0.8, 0.15)
        if success_rate <= 0.2:
            return (0.6, 0.35)
        return (0.7, 0.25)

    @staticmethod
    def _hyp_key(hypothesis: str) -> str:
        """假设指纹: 归一化后取 md5 前 12 位, 用于把"历史失败"限定在同一假设.

        scene_tag 太粗 (capacity scan / 拓扑 / 上同调 plan 全落 "other"), 单靠
        scene 过滤会让 checker 把上一个假设的失败判定当成当前 plan 的"已知坑",
        反复回灌同一结论 (run45 的自我污染死循环). 按假设指纹隔离即断链.
        """
        norm = re.sub(r"\s+", " ", (hypothesis or "").strip().lower())
        return hashlib.md5(norm.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]

    def _plan_check_scene_tag(self, plan: dict[str, Any]) -> str:
        """从 plan 抽场景标签, 给失败模式记忆和分桶自适应用.

        写死的关键词表 + 自动发现的关键词 (_scene_tag_extra_keywords) 互补.
        ponytail: 关键词匹配, 不上 embedding.
        ceiling: 写死的关键词表要手动加新仿真器; 自动发现靠高频词统计,
          新场景需要 >=3 次出现才会被识别.
        升级路径: 用 plan_check_history 聚类自动发现 scene_tag (无监督).
        """
        desc = (plan.get("description", "") + " " + plan.get("mode", "")).lower()
        # 写死的关键词表 (快路径)
        hit = _match_scene_keywords(desc, _SCENE_KEYWORDS_HEAD)
        if hit:
            return hit
        if plan.get("mode") == "skill":
            return "skill"
        hit = _match_scene_keywords(desc, _SCENE_KEYWORDS_TAIL)
        if hit:
            return hit
        # 自动发现的关键词 (慢路径, 跨 run 积累)
        for label, keywords in getattr(self, "_scene_tag_extra_keywords", {}).items():
            if any(kw in desc for kw in keywords):
                return label
        return "other"

    def _discover_scene_tags(self) -> None:
        """从 _plan_check_history 里 scene='other' 的 plans 做关键词统计,
        发现高频词 (>=3 次) 自动加到 _scene_tag_extra_keywords.

        命中未知场景 — 新仿真器/新任务类型不用手动改关键词表, 跑几次
        plan_check 后自动归类.
        双重识别: unigram (>=4 chars) + bigram (两词短语, 如 "phase diagram",
        "neb chain"), 更准地捕获多词术语.
        ponytail: 简单词频统计, 不上 TF-IDF/embedding.
        ceiling: 只统计 scene='other' 的 plans, 已归类的不参与; 阈值 3 靠拍;
          只取英文, 中文/数字不参与; bigram 不去介词/停用词组合.
        升级路径: 上 TF-IDF 或 embedding 聚类, 识别任意长度 n-gram.
        """
        other_descs = _collect_other_scene_descs(
            getattr(self, "_plan_check_history", [])
        )
        if len(other_descs) < 3:
            return  # 样本不足, 不触发发现
        word_counts, bigram_counts = _count_plan_ngrams(other_descs)
        # 高频词 (>=3 次) 与高频 bigram 都进 extra_keywords (bigram 更具体,
        # 如 "phase diagram" -> auto_phase_diagram, 关键词 "phase diagram")
        _apply_discovered_tags(self._scene_tag_extra_keywords, word_counts, 10)
        _apply_discovered_tags(self._scene_tag_extra_keywords, bigram_counts, 5)

    def _plan_check_complexity(self, plan: dict[str, Any]) -> float:
        """plan 复杂度评分 [0, 1], 跟 tier 一起决定是否校验.

        维度: description 长度 (0.3) + mode 复杂度 (0.4) + 有无 prediction
        (0.15) + 同场景历史失败数 (0.15, 踩过坑的要复查).
        ponytail: 启发式打分, 不上结构化解析.
        ceiling: description 长度不代表真复杂度, 长描述可能是废话.
        升级路径: 解析 plan 的 step 数 (需要结构化 plan schema).
        """
        score = 0.0
        desc = plan.get("description", "")
        score += min(len(desc), 50) / 50 * 0.3
        mode = plan.get("mode", "coder")
        score += {"workflow": 0.4, "skill": 0.3, "coder": 0.2, "explore": 0.1}.get(
            mode, 0.2
        )
        if plan.get("expected_prediction"):
            score += 0.15
        scene = self._plan_check_scene_tag(plan)
        similar_fails = sum(
            1
            for p in getattr(self, "_plan_check_patterns", [])
            if p.get("scene_tag") == scene
        )
        score += min(similar_fails, 3) / 3 * 0.15
        return min(score, 1.0)

    def _plan_check_max_refines(self, tier: str, scene: str = "") -> int:
        """自适应: 按场景分桶的 EWMA success rate 微调 max_refines.

        baseline: medium=0 (只校验不 refine), light=1 (完整闭环).
        分桶: 同 scene_tag 的最近 5 次, EWMA 加权 (alpha 根据桶大小自适应)
          >=80% 放宽 (baseline-1, 最低 0), <=20% 收紧 (baseline+1, 最高 2).
        alpha 自适应: 桶 3-4 条用 alpha=0.3 (老样本权重大, 样本少要稳),
          桶 5 条用 alpha=0.4 (近期权重大, 样本足要敏感).
        样本 <3 走 baseline, 早期不误判. 未知场景 (scene 无历史) 走全局.
        ponytail: EWMA 简单指数加权; alpha 分两档, 不上 decay schedule.
        ceiling: 桶太小 (<5 条) EWMA 不稳, 但样本不足走 baseline 兜底;
          alpha 分档靠拍, 没数据校准.
        升级路径: alpha 用 cross-validation 自动选; 或上 Bayesian 更新.
        """
        baseline = {"medium": 0, "light": 1}.get(tier, 1)
        history = getattr(self, "_plan_check_history", [])
        bucket = (
            [c for c in history if c.get("scene_tag") == scene] if scene else history
        )
        if len(bucket) < 3:
            return baseline
        recent = bucket[-5:]
        # alpha 自适应: 桶小用低 alpha (稳), 桶大用高 alpha (敏感)
        alpha = 0.3 if len(recent) < 5 else 0.4
        weights = [
            alpha * (1 - alpha) ** (len(recent) - 1 - i) for i in range(len(recent))
        ]
        total_w = sum(weights)
        if total_w == 0:
            return baseline
        ewma_success = (
            sum(
                w * (1.0 if c.get("is_valid", True) else 0.0)
                for w, c in zip(weights, recent)
            )
            / total_w
        )
        if ewma_success >= 0.8:
            return max(0, baseline - 1)
        if ewma_success <= 0.2:
            return min(2, baseline + 1)
        return baseline

    async def _plan_check(
        self,
        plan: dict[str, Any],
        hypothesis: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """单次反向校验: 让 LLM 判断 plan 执行后能否达成 hypothesis.

        用 task='verification' 让 model_router 路由到独立验证模型,
        避免正向/反向用同一个模型 (同模型有同盲点).

        v6 G57: DeepMind 三层 validation — L1 LLM plan_check (本方法) +
        L2 dimensional pre-check (_dimensional_pre_check) +
        L3 physical_precheck (PRE_TOOL_USE hook).
        量纲不一致不直接淘汰 plan, 只追加到 risks + dimensional_warnings,
        让 LLM judge 看到后决定.
        """
        # L2: dimensional pre-check — 先跑, 把 warnings 拼进 prompt 上下文
        dim_warnings = self._dimensional_pre_check(plan, hypothesis)
        if dim_warnings:
            context = dict(context)
            context["dimensional_warnings"] = "\n".join(dim_warnings)

        prompt = self._build_plan_check_prompt(plan, hypothesis, context)
        response = await self._llm_chat(
            prompt,
            persona_name="default",
            task="verification",
        )
        result = self._parse_plan_check(response)
        if dim_warnings:
            result["dimensional_warnings"] = dim_warnings
            existing = result.get("risks") or []
            existing.extend(dim_warnings)
            result["risks"] = existing
            self._plan_check_warnings.extend(dim_warnings)
        return result

    def _dimensional_pre_check(
        self,
        plan: dict[str, Any],
        hypothesis: str,
    ) -> list[str]:
        """L2 dimensional pre-check — 扫 plan + hypothesis 里的等式, 验量纲.

        ponytail: regex 抓 "<number> <unit>" 量 + "=" 等式, 调 DimensionalValidator.
        只在能解析出两侧都带量纲的等式时跑; 否则跳过 (不误报).
        ceiling: 简单 regex 抓不住复杂表达式 (函数调用 / 多行推导);
        升级路径: sympy 解析 + 单位推断.
        """
        warnings: list[str] = []
        try:
            from huginn.validation.dimensional import DimensionalValidator
        except Exception:  # 防御: 校验异常返回既有警告
            return warnings
        text = _dimensional_text(plan, hypothesis)
        if "=" not in text:
            return warnings
        validator = DimensionalValidator()
        for line in text.splitlines():
            warning = _dimensional_line_warning(validator, line)
            if warning:
                warnings.append(warning)
        return warnings

    def _pcp_similar_history(
        self, plan: dict[str, Any], hypothesis: str
    ) -> tuple[str, str]:
        """同假设历史失败模式 → (scene, 渲染文本).

        旧版只按 scene_tag 抽, 而 scene="other" 是万能桶 (capacity scan /
        拓扑 / 上同调 plan 全落这里), 于是 checker 把自己上一轮的判定当"已知
        坑"原样回灌, 锁死结论 (run45 46/46 全失败的自我污染). 现按假设指纹
        隔离 + reason 去重: 只喂真正同一假设下、且互不重复的失败信息.
        """
        scene = self._plan_check_scene_tag(plan)
        hyp_key = self._hyp_key(hypothesis)
        same_hyp = [
            p
            for p in getattr(self, "_plan_check_patterns", [])
            if p.get("scene_tag") == scene and p.get("hyp_key") == hyp_key
        ]
        seen_reasons: set[str] = set()
        similar: list[dict[str, Any]] = []
        for p in reversed(same_hyp):
            r = str(p.get("reason") or "")[:120]
            if r in seen_reasons:
                continue
            seen_reasons.add(r)
            similar.append(p)
            if len(similar) >= 3:
                break
        similar.reverse()
        if not similar:
            return scene, "N/A"
        return scene, "\n".join(
            f"- {str(p['reason'])[:240]} "
            f"(缺: {', '.join(p.get('missing_steps', [])) or 'N/A'})"
            for p in similar
        )

    def _build_plan_check_prompt(
        self,
        plan: dict[str, Any],
        hypothesis: str,
        context: dict[str, Any],
    ) -> str:
        """反向规划识别器 prompt: 判断 plan 能否达成 hypothesis."""
        # 从 context 抽最近失败模式, 帮 LLM 避开已知坑
        failure_modes = context.get("failure_modes", "")
        if not failure_modes and self._speculator_hint:
            failure_modes = self._speculator_hint[-500:]
        scene, similar_text = self._pcp_similar_history(plan, hypothesis)
        # v6 G57: L2 dimensional pre-check 警告 (若 context 带了)
        dim_warnings_text = context.get("dimensional_warnings", "") or "N/A"
        return _pcp_render_prompt(
            hypothesis, plan, failure_modes, scene, similar_text, dim_warnings_text
        )

    def _record_plan_check_failure(
        self,
        plan: dict[str, Any],
        check: dict[str, Any],
        scene: str,
        hypothesis: str = "",
    ) -> None:
        """失败模式记到 patterns, 跨 run 持久化给下次注入 prompt.

        ponytail: 内存 append + 同步 dump JSON, 量小 (<=50 条) 写快.
        ceiling: 同步写盘, 高频失败时可能拖慢; description 截断 200 chars.
        升级路径: 后台 async flush, 或上 SQLite.
        """
        self._plan_check_patterns.append(
            {
                "scene_tag": scene,
                "hyp_key": self._hyp_key(hypothesis),
                "reason": check.get("reason", "unknown"),
                "missing_steps": check.get("missing_steps", []),
                "mode": plan.get("mode", ""),
                "description": plan.get("description", "")[:200],
            }
        )
        if len(self._plan_check_patterns) > 50:
            del self._plan_check_patterns[: len(self._plan_check_patterns) - 50]
        self._save_plan_check_patterns()

    def _load_plan_check_patterns(self) -> None:
        """跨 run 加载历史失败模式.

        ponytail: JSON 文件, 不上 DB; 只在 _prepare_run 调一次.
        ceiling: 文件可能被外部篡改, 解析失败静默回退.
        """
        path = self.workspace / HUGINN_DIR_NAME / "plan_check_patterns.json"
        if not path.exists():
            return
        try:
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._plan_check_patterns = data[-50:]
                logger.info(
                    "loaded %d plan_check patterns from %s",
                    len(self._plan_check_patterns),
                    path,
                )
        except Exception as e:
            logger.debug("load plan_check_patterns failed: %s", e)

    def _save_plan_check_patterns(self) -> None:
        """dump 失败模式到 workspace, 跨 run 积累.

        ponytail: 同步写, 量小 (<=50 条); 跟 skill_evolver 历史持久化同款.
        """
        path = self.workspace / HUGINN_DIR_NAME / "plan_check_patterns.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            import json

            path.write_text(
                json.dumps(
                    self._plan_check_patterns[-50:], ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug("save plan_check_patterns failed: %s", e)

    def _parse_plan_check(self, response: str) -> dict[str, Any]:
        """解析反向校验 JSON — 括号配平法 (ValidityJudge._parse_verdict 同款).

        解析失败返回 is_valid=True (跳过校验, 不阻塞).
        """
        blob, err = _extract_json_object(response)
        if err:
            return {"is_valid": True, "reason": err}
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            return {"is_valid": True, "reason": "json parse failed, skip"}
        # 字段补全, 保证下游一致
        obj.setdefault("is_valid", True)
        obj.setdefault("confidence", 0.8)  # 默认高置信, 不误触发 refine
        obj.setdefault("reason", "")
        obj.setdefault("missing_steps", [])
        obj.setdefault("risks", [])
        return obj

    def _rp_few_shot(self, plan: dict[str, Any]) -> tuple[str, str]:
        """抽同场景最近 1 条成功 plan 做 few-shot; 返回 (scene, 渲染块)."""
        scene = self._plan_check_scene_tag(plan)
        success_example = None
        for c in reversed(getattr(self, "_plan_check_history", [])):
            if (
                c.get("is_valid")
                and c.get("scene_tag") == scene
                and c.get("plan_snapshot")
            ):
                success_example = c["plan_snapshot"]
                break
        if not success_example:
            return scene, "N/A"
        return scene, (
            f"MODE: {success_example.get('mode', 'coder')}\n"
            f"DESCRIPTION: {success_example.get('description', '')[:200]}"
        )

    async def _refine_plan(
        self,
        plan: dict[str, Any],
        check: dict[str, Any],
        hypothesis: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """根据反向校验反馈, 让 LLM 重新生成 plan (保留 plan_id).

        few-shot: 从 _plan_check_history 抽同场景最近 1 条成功 plan 塞进
        prompt, 让 LLM 知道'上次同场景怎么成功的'. 命中长程任务 — 跨 iteration
        积累的成功经验不再丢失.
        """
        scene, few_shot_block = self._rp_few_shot(plan)
        prompt = _rp_render_prompt(plan, hypothesis, check, scene, few_shot_block)
        prompt = self._extend_mode_enum(prompt)
        try:
            response = await self._llm_chat(
                prompt,
                persona_name="default",
                task="planning",
            )
            new_plan = self._parse_plan(response)
            new_plan = self._override_plan_mode(new_plan)
            # 保留 plan_id (如果有), 让 PlanStore 能跟踪同一 plan 的演进
            if "plan_id" in plan:
                new_plan["plan_id"] = plan["plan_id"]
            return new_plan
        except Exception as e:
            logger.debug("plan refine failed: %s", e)
            return plan
