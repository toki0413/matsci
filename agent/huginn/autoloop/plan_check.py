"""PlanCheckMixin - plan_check 方法族, 从 engine.py 下沉.

P1 slim-down: 22 个 plan_check 方法从 engine.py 迁入, 定义为 mixin class.
engine 通过多继承接入, 方法内通过 self 访问 engine 状态字段
(_plan_check_patterns / _plan_check_history / _plan_check_warnings /
_plan_check_last_result / _scene_tag_extra_keywords / _iteration / workspace)
和 engine 方法 (_maybe_clarify / _llm_chat / _build_memory_text /
_build_metacog_block / _build_hypothesis_prompt 等).

调用点: engine._prepare_run 调 _load_plan_check_patterns();
engine._plan 调 _plan_check_and_refine().
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any

from huginn.memory.longterm import load_stable_principles

logger = logging.getLogger(__name__)


class PlanCheckMixin:
    """plan_check 方法族. 通过 self 访问 engine 状态."""

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

    def _build_plan_prompt(self, hypothesis: str, context: dict[str, Any]) -> str:
        # 同 hypothesize: 用 hypothesis 串检索 KB, 把参考块喂给 planner
        kb_block = self._build_kb_text(query=hypothesis)
        if kb_block:
            kb_block = f"\n{kb_block}\n"
        # KG 检索: 用 hypothesis 当 query, 看看已有实体里有没有相关的
        kg_block = self._build_kg_text(query=hypothesis)
        if kg_block:
            kg_block = f"\n{kg_block}\n"
        # 长期记忆检索 (同 hypothesize)
        mem_block = self._build_memory_text(query=hypothesis)
        if mem_block:
            mem_block = f"\n{mem_block}\n"
        # C2: PM 层 trajectory_match 召回 (同 hypothesize, 极限模式才开)
        pm_block = self._build_pm_text()
        if pm_block:
            pm_block = f"\n{pm_block}\n"
        # C2: metacog 信号注入 (同 hypothesize) — target_chain + prospective
        metacog_block = self._build_metacog_block(include_prospective=True)
        if metacog_block:
            metacog_block = f"\n{metacog_block}\n"
        # H0: stable_principles 注入 (同 hypothesize, 修 P3 断链)
        try:
            _principles = load_stable_principles()[:5]
        except Exception:
            _principles = []
        principles_block = (
            "\n".join(f"- {p}" for p in _principles) if _principles else ""
        )
        if principles_block:
            principles_block = (
                f"\n### Stable Principles (procedural memory)\n{principles_block}\n"
            )
        # P1 Task 5: world model 预测注入 (懒路, toggle HUGINN_WORLD_MODEL)
        world_model_block = self._build_world_model_block(hypothesis)
        if world_model_block:
            world_model_block = f"\n### World Model Prediction (analogy-based)\n{world_model_block}"

        # 视觉基元注入 (同 hypothesize)
        visual_block = getattr(self, "_last_visual_context", "")
        if visual_block:
            visual_block = (                f"\n### Visual Primitives (from last tool output)\n{visual_block}\n"
            )
        # 条件化 math_block (同 hypothesize)
        hyp_blob = (
            hypothesis.lower() + json.dumps(context, ensure_ascii=False).lower()[:500]
        )
        math_block = (
            self._MATH_DEPTH_PROMPT_BLOCK
            if any(s in hyp_blob for s in _MATH_SIGNALS)
            else ""
        )

        # Inject learned skills + prompt patches from evolution engine.
        # This is the "use what you learned" half of the Learn→Plan loop.
        skill_hints = ""
        patch_hints = ""
        try:
            evolution = self._get_evolution()
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
        except Exception:
            logger.warning(
                "error in _build_plan_prompt: evolution skill/patch fetch failed",
                exc_info=True,
            )

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
            except Exception:
                logger.debug("composite skill lookup failed", exc_info=True)

        # Pipeline 建议: 基于 provenance 规则推荐下一步工具.
        # 42 条领域规则, 零 LLM 调用. 让 plan 知道"这类任务通常下一步是 X".
        pipeline_block = ""
        try:
            from huginn.provenance.pipeline import SimulationPipeline

            pipeline = SimulationPipeline(
                self.kg.root if hasattr(self.kg, "root") else None
            )
            # 用上一轮的 execution_result 触发 suggest_next
            last_result = getattr(self, "_last_execution_result", None)
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
        except Exception:
            pass  # pipeline 是 advisory, 失败不阻塞

        blocks = self._apply_block_patches(
            [
                (
                    "body",
                    f"""Given the hypothesis: "{hypothesis}"

Context:
{json.dumps(context, indent=2, ensure_ascii=False)[:1000]}

Choose ONE mode and describe the plan:
- coder: modify code/files to fix or improve something
- workflow: run a computational simulation pipeline
- explore: search a design space for optimal parameters
- skill: use a pre-built composite skill pipeline (band structure, mechanical properties, MD, etc.)
- visual_inspect: interactively inspect visual data (zoom into chart region, measure data points, annotate structure). Use this when you need to examine previous results more carefully before deciding next steps. Available actions: zoom, measure, annotate, compare.

Protocol completeness check (RCBench failure mode: experimental protocol mismatch):
Before finalizing, verify your plan covers all necessary steps:
- For DFT: structure optimization BEFORE property calculation? Convergence test (encut/kpoints)?
- For MD: equilibration BEFORE production run? Timestep appropriate for the system?
- For analysis: raw data processing BEFORE interpretation? Reference comparison?
- Are computational parameters appropriate for the target property (e.g. HSE06 for band gap, not PBE)?
- Cross-check against domain knowledge above: any known methodological requirements?
If a step is missing, add it to DESCRIPTION.

When the hypothesis involves a PDE / variational principle / curved
geometry, consider the symbolic_math_tool actions listed in the math
depth block above — but numerical solvers are equally valid.

Respond in this exact format:
MODE: <coder|workflow|explore|skill>
DESCRIPTION: <brief description of what to do>
SKILL: <composite skill name, only if MODE is skill>
PREDICTION: <what you expect the result to look like — be specific: "energy ~ -X eV", "converges in ~N steps", "band gap ~X eV". This prediction will be compared against actual results to measure surprise.>
""",
                ),
                ("math", math_block),
                ("kg", kg_block),
                ("visual", visual_block),
                ("kb", kb_block),
                ("principles", principles_block),
                ("world_model", world_model_block),
                ("mem", mem_block),
                ("pm", pm_block),
                ("metacog", metacog_block),
                ("skill", skill_hints + patch_hints),
                ("composite", composite_block),
                ("pipeline", pipeline_block),
                ("subgoal", self._build_subgoal_block()),
                ("ctx_hint", self._plan_context_hint()),
            ],
            "plan",
        )
        return self._trim_to_budget(blocks, phase="plan")

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
        except Exception:
            pass
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
        # surprise 高 → 预测误差大, 倾向 explore 重新假设
        surprise = getattr(self, "_last_surprise", 0.0)
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
        # 割点节点: 强制非 coder mode
        try:
            current_hyp = getattr(self, "_current_hyp_id_for_plan", None)
            if (
                current_hyp
                and self.hypothesis_graph.needs_dual_coverage(current_hyp)
                and current_mode == "coder"
            ):
                plan["mode"] = "workflow"
                plan["override_reason"] = "cut_vertex_dual_coverage"
                plan["description"] = (
                    f"[auto-routed: 割点需双覆盖] {plan.get('description', '')}"
                )
                logger.info(
                    "override mode coder→workflow for cut vertex %s", current_hyp
                )
                self._log_plan_override(
                    "cut_vertex_dual_coverage", f"割点 {current_hyp} 需双覆盖"
                )
        except Exception:
            pass
        # 连败/surprise 强制 explore (合并条件, 共享覆盖路径)
        cf = getattr(self, "_consecutive_failures", 0)
        surprise = getattr(self, "_last_surprise", 0.0)
        explore_reasons: list[str] = []
        if cf >= 5:
            explore_reasons.append(f"连续失败{cf}次")
        if surprise > 0.9:
            explore_reasons.append(f"surprise={surprise:.2f}")
        if explore_reasons and plan["mode"] != "explore":
            reason = "+".join(explore_reasons)
            plan["mode"] = "explore"
            plan["override_reason"] = "force_explore"
            plan["description"] = (
                f"[auto-routed: {reason}] {plan.get('description', '')}"
            )
            logger.info("override mode →explore: %s", reason)
            self._log_plan_override("force_explore", reason)
        return plan

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
        except Exception:
            logger.debug("log plan override failed", exc_info=True)

    def _parse_plan(self, response: str) -> dict[str, Any]:
        """Parse LLM plan response."""
        mode = "coder"
        description = response.strip()
        skill_name = ""
        prediction = ""

        for line in response.split("\n"):
            if line.startswith("MODE:"):
                mode = line.replace("MODE:", "").strip().lower()
            elif line.startswith("DESCRIPTION:"):
                description = line.replace("DESCRIPTION:", "").strip()
            elif line.startswith("SKILL:"):
                skill_name = line.replace("SKILL:", "").strip()
            elif line.startswith("PREDICTION:"):
                prediction = line.replace("PREDICTION:", "").strip()

        plan = {"mode": mode, "description": description}
        if skill_name:
            plan["skill"] = skill_name
        if prediction:
            plan["expected_prediction"] = prediction
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
        for attempt in range(max_refines + 1):
            try:
                check = await self._plan_check(plan, hypothesis, context)
            except Exception as e:
                logger.debug("plan_check LLM call failed: %s", e)
                return plan
            # 给 check 打 scene_tag, 喂分桶自适应; 不暴露: 存引擎状态.
            # 成功时存 plan_snapshot, 喂 _refine_plan few-shot.
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
            if check.get("is_valid", True):
                # confidence 分级: 低置信通过 (<0.5) 强制 refine 一次, 防 LLM
                # 没看懂就放行; 高置信直接通过.
                confidence = float(check.get("confidence", 0.8))
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
                    return plan
                logger.info(
                    "plan_check passed but low confidence (conf=%.2f), refining",
                    confidence,
                )
            else:
                # 失败: 记到 patterns (跨 run 持久化, 喂下次 prompt)
                self._record_plan_check_failure(plan, check, scene)
                # confidence 分级: 低置信失败 (<0.3) 跳过 refine, LLM 都没把握
                # 判断, refine 可能也是瞎改, 直接 warning + 触发澄清更靠谱.
                confidence = float(check.get("confidence", 0.8))
                if confidence < 0.3:
                    reason = check.get("reason", "unknown")
                    self._plan_check_warnings.append(
                        f"[{scene}] {reason} (low_conf={confidence:.2f})"
                    )
                    logger.warning(
                        "plan_check failed low-conf (tier=%s, scene=%s, conf=%.2f): %s",
                        tier,
                        scene,
                        confidence,
                        reason,
                    )
                    await self._maybe_trigger_plan_check_clarify(scene, reason, plan)
                    return plan
                if attempt >= max_refines:
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
                    await self._maybe_trigger_plan_check_clarify(
                        scene,
                        reason,
                        plan,
                    )
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
        return plan

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
        if any(
            kw in desc
            for kw in [
                "vasp",
                "scf",
                "band",
                "dos",
                "dft",
                "qe",
                "cp2k",
                "gaussian",
                "orca",
            ]
        ):
            return "dft"
        if any(
            kw in desc
            for kw in [
                "lammps",
                "molecular dynamics",
                "minimize",
                "nvt",
                "npt",
                "md ",
                "gromacs",
                "openmm",
            ]
        ):
            return "md"
        if any(kw in desc for kw in ["workflow", "pipeline", "orchestrat"]):
            return "workflow"
        if plan.get("mode") == "skill":
            return "skill"
        if any(
            kw in desc
            for kw in ["fenics", "abaqus", "comsol", "openfoam", "fem", "elmer"]
        ):
            return "fem"
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
        import re
        from collections import Counter

        # 收集 scene='other' 的 plan descriptions
        other_descs: list[str] = []
        for c in getattr(self, "_plan_check_history", []):
            snapshot = c.get("plan_snapshot") or {}
            if c.get("scene_tag") == "other" and snapshot.get("description"):
                other_descs.append(snapshot["description"].lower())
        if len(other_descs) < 3:
            return  # 样本不足, 不触发发现
        # 统计英文单词词频 (>=4 chars, 过滤停用词)
        stop = {
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
        word_counts: Counter[str] = Counter()
        # bigram 词频 (两词短语, 用空格连接)
        bigram_counts: Counter[str] = Counter()
        for desc in other_descs:
            words = [
                w for w in re.findall(r"[a-z][a-z0-9_]{3,}", desc) if w not in stop
            ]
            for word in words:
                word_counts[word] += 1
            # bigram: 相邻两词组合
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                bigram_counts[bigram] += 1
        # 高频词 (>=3 次) 加到 extra_keywords, 用 word 本身做 label
        for word, count in word_counts.most_common(10):
            if count >= 3:
                label = f"auto_{word}"
                self._scene_tag_extra_keywords.setdefault(label, set()).add(word)
        # 高频 bigram (>=3 次) 加到 extra_keywords, 用下划线连接做 label
        # (如 "phase diagram" -> auto_phase_diagram, 关键词 "phase diagram")
        for bigram, count in bigram_counts.most_common(5):
            if count >= 3:
                label = f"auto_{bigram.replace(' ', '_')}"
                self._scene_tag_extra_keywords.setdefault(label, set()).add(bigram)

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
        except Exception:
            return warnings

        # 拼 plan + hypothesis 文本
        text_parts = [hypothesis or ""]
        for k in ("description", "expected_prediction", "prediction"):
            v = plan.get(k) if isinstance(plan, dict) else None
            if isinstance(v, str) and v:
                text_parts.append(v)
        text = "\n".join(text_parts)
        if "=" not in text:
            return warnings

        validator = DimensionalValidator()
        # 抓 "<number> <unit>" 量, e.g. "210 GPa" / "1.5e3 kg/m3"
        qty_re = re.compile(
            r"([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s+([A-Za-z][A-Za-z0-9/\^\-\*\.\(\)]+)"
        )
        # 按行 + 按 "=" 切等式
        for line in text.splitlines():
            if "=" not in line:
                continue
            lhs, rhs = line.split("=", 1)
            lhs_qs = [f"{m[0]} {m[1]}" for m in qty_re.findall(lhs)]
            rhs_qs = [f"{m[0]} {m[1]}" for m in qty_re.findall(rhs)]
            if not lhs_qs or not rhs_qs:
                continue
            try:
                result = validator.check_equation(lhs_qs, rhs_qs, equation_name=line.strip()[:80])
                if not result.consistent:
                    warnings.append(
                        f"dimensional inconsistency: '{line.strip()[:80]}' "
                        f"LHS={result.lhs_dimensions} RHS={result.rhs_dimensions}"
                    )
            except Exception:
                # 解析失败静默跳过 — 量纲库不全不该阻塞 plan_check
                continue
        return warnings

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
        # 同场景历史失败模式 (跨 run 积累, 最近 3 条) — 让 LLM 重点避开
        scene = self._plan_check_scene_tag(plan)
        similar = [
            p
            for p in getattr(self, "_plan_check_patterns", [])
            if p.get("scene_tag") == scene
        ][-3:]
        if similar:
            similar_text = "\n".join(
                f"- {p['reason']} (缺: {', '.join(p.get('missing_steps', [])) or 'N/A'})"
                for p in similar
            )
        else:
            similar_text = "N/A"
        # v6 G57: L2 dimensional pre-check 警告 (若 context 带了)
        dim_warnings_text = context.get("dimensional_warnings", "") or "N/A"
        return f"""你是反向规划识别器 (KRCL 启发). 判断以下 plan 执行后能否达成 hypothesis.

# 目标 (hypothesis)
{hypothesis}

# 当前 plan
MODE: {plan.get('mode', 'coder')}
DESCRIPTION: {plan.get('description', '')}
PREDICTION: {plan.get('expected_prediction', 'N/A')}

# 已知失败模式 (避免重蹈覆辙)
{failure_modes or 'N/A'}

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

输出 JSON (不要其他文本):
{{
  "is_valid": true 或 false,
  "confidence": 0.0 到 1.0 (对判断的置信度, 1.0=非常确定, 0.5=模棱两可, 0.0=完全没把握),
  "reason": "为什么 valid / invalid",
  "missing_steps": ["如果 invalid, 缺少哪些步骤"],
  "risks": ["潜在风险"]
}}"""

    def _record_plan_check_failure(
        self,
        plan: dict[str, Any],
        check: dict[str, Any],
        scene: str,
    ) -> None:
        """失败模式记到 patterns, 跨 run 持久化给下次注入 prompt.

        ponytail: 内存 append + 同步 dump JSON, 量小 (<=50 条) 写快.
        ceiling: 同步写盘, 高频失败时可能拖慢; description 截断 200 chars.
        升级路径: 后台 async flush, 或上 SQLite.
        """
        self._plan_check_patterns.append(
            {
                "scene_tag": scene,
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
        path = self.workspace / ".huginn" / "plan_check_patterns.json"
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
        path = self.workspace / ".huginn" / "plan_check_patterns.json"
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
        import json

        start = response.find("{")
        if start < 0:
            return {"is_valid": True, "reason": "no json, skip"}
        depth = 0
        for i, ch in enumerate(response[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(response[start : i + 1])
                        # 字段补全, 保证下游一致
                        obj.setdefault("is_valid", True)
                        obj.setdefault("confidence", 0.8)  # 默认高置信, 不误触发 refine
                        obj.setdefault("reason", "")
                        obj.setdefault("missing_steps", [])
                        obj.setdefault("risks", [])
                        return obj
                    except json.JSONDecodeError:
                        return {"is_valid": True, "reason": "json parse failed, skip"}
        return {"is_valid": True, "reason": "no closing brace, skip"}

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
        # 抽同场景最近 1 条成功 plan (is_valid=True, scene_tag 相同)
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
        few_shot_block = "N/A"
        if success_example:
            few_shot_block = (
                f"MODE: {success_example.get('mode', 'coder')}\n"
                f"DESCRIPTION: {success_example.get('description', '')[:200]}"
            )
        prompt = f"""之前的 plan 未通过反向校验. 根据反馈重新生成.

# 目标
{hypothesis}

# 之前的 plan
MODE: {plan.get('mode', 'coder')}
DESCRIPTION: {plan.get('description', '')}

# 校验反馈
reason: {check.get('reason', '')}
missing_steps: {check.get('missing_steps', [])}
risks: {check.get('risks', [])}

# 同场景成功示例 (scene={scene}, 跨 iteration 积累, 仅供参考结构)
{few_shot_block}

# 任务
根据反馈重新生成 plan. 参考成功示例的结构 (不要照抄内容). 严格按格式输出:
MODE: <coder|workflow|explore|skill|visual_inspect>
DESCRIPTION: <brief description>
SKILL: <composite skill name, only if MODE is skill>
PREDICTION: <预期结果, 用于后续 validate 对比>"""
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
