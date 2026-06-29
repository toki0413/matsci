"""Autoloop Engine — the main autonomous loop for Huginn.

Ties together exploration, coder, workflow, benchmark, and report
into a single closed-loop ecosystem:

    Perceive → Hypothesize → Plan → Execute → Validate → Learn → Report

Usage:
    engine = AutoloopEngine(workspace=Path("."))
    asyncio.run(engine.run(objective="Optimize C-S-H defect kinetics"))
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from huginn.bench.runner import BenchmarkRunner
from huginn.coder.loop import CoderRunner
from huginn.config import get_settings
from huginn.exploration.orchestrator import ExplorationOrchestrator
from huginn.exploration.strategies import ParetoPruningStrategy
from huginn.interaction.progress import ProgressTracker, get_progress_tracker
from huginn.kg.builder import ProjectKnowledgeGraph
from huginn.llm import get_model
from huginn.memory.manager import MemoryManager
from huginn.tools.report_tool import ReportTool
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult
from huginn.workflows.engine import WorkflowEngine
from huginn.workflows.templates import standard_dft_workflow


# 7 阶段 → persona 分派表. None 表示该阶段不走 LLM persona 注入
# (比如 Execute 直接调 workflow, 不需要 persona 影响输出).
# Hypothesize 用 default, 真正的 persona 在 _hypothesize 里按研究类型动态选.
_PHASE_PERSONAS: dict[str, str | None] = {
    "perceive": "default",
    "hypothesize": None,  # 动态选 dft_expert / md_expert, 见 _hypothesize
    "plan": "default",
    "execute": None,  # 直接调 workflow / coder, 不走 LLM persona
    "validate": "reviewer",  # 关键: 校验阶段用 reviewer persona 做批判性审视
    "learn": "default",
    "report": "tutor",  # 教学风格输出
}


@dataclass
class LoopPhase:
    """A single phase in the autonomous loop."""

    name: str
    status: str = "pending"  # pending | running | completed | failed
    start_time: float | None = None
    end_time: float | None = None
    result: Any = None
    error: str | None = None


@dataclass
class AutoloopResult:
    """Result of a full autonomous loop iteration."""

    run_id: str
    objective: str
    phases: list[LoopPhase]
    success: bool
    report_path: str | None = None
    total_time_seconds: float = 0.0


class AutoloopEngine:
    """Main autonomous loop engine.

    Orchestrates perception, hypothesis generation, planning, execution,
    validation, learning, and reporting into a single cohesive loop.
    """

    def __init__(self, workspace: str | Path | None = None):
        self.workspace = Path(workspace or ".").resolve()
        self.settings = get_settings()
        self.model = get_model(self.settings)
        self.memory = MemoryManager()
        self.kg = ProjectKnowledgeGraph()
        self.report_tool = ReportTool()

        # Sub-engines
        self.explorer = ExplorationOrchestrator(
            strategy=ParetoPruningStrategy(),
            max_parallel=3,
        )
        self.workflow_engine = WorkflowEngine(
            tool_registry=None,  # Will use default tool registry
        )
        self.coder = CoderRunner()

        self._should_stop = False
        self._iteration = 0
        # Evolution engine 懒加载——只在 _learn 真正用到时初始化
        self._evolution = None
        # PersonaManager 懒加载 — 避免实例化时就扫描 .huginn/personas 目录
        self._persona_manager = None
        # 进度跟踪: 默认走进程级单例, 跟 WorkflowEngine 共享, 让 /tasks
        # 路由能汇总所有引擎的进度. 测试时可注入独立 tracker 隔离.
        self.progress_tracker: ProgressTracker | None = None
        # 投机执行 hint: on_turn_start 写入, _build_*_prompt 读出注入 LLM
        self._speculator_hint: str = ""

    def _get_evolution(self):
        """懒加载 EvolutionEngine, 避免实例化时就拉起日志和规则文件。"""
        if self._evolution is None:
            from huginn.evolution.engine import EvolutionEngine
            from huginn.evolution.logger import ExecutionLogger

            self._evolution = EvolutionEngine(logger=ExecutionLogger())
        return self._evolution

    def _get_persona_manager(self):
        """懒加载 PersonaManager, 实例化时才扫描 persona 文件."""
        if self._persona_manager is None:
            from huginn.personas import PersonaManager

            self._persona_manager = PersonaManager(workspace=self.workspace)
        return self._persona_manager

    def _persona_system_prompt(self, persona_name: str | None) -> str:
        """取 persona 的 system prompt. 找不到就返回空串, 不报错."""
        if not persona_name:
            return ""
        try:
            persona = self._get_persona_manager().get(persona_name)
            return persona.system_prompt or ""
        except Exception:
            return ""

    @staticmethod
    def _phase_persona(phase_name: str) -> str | None:
        """查表拿阶段对应的 persona 名."""
        return _PHASE_PERSONAS.get(phase_name)

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    async def run(self, objective: str, max_iterations: int = 5) -> AutoloopResult:
        """Run the full autonomous loop for the given objective."""
        run_id = f"loop_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        phases: list[LoopPhase] = []

        self._iteration = 0
        self._should_stop = False

        # 投机执行: 拿 top-3 意图, 高置信度时预热工具缓存
        # 预测只是 hint, 不强制, LLM 可以无视
        self._speculator_hint = ""
        try:
            from huginn.agents.speculator import on_turn_start

            spec_result = on_turn_start(objective)
            self._speculator_hint = spec_result.get("hint", "")
            if spec_result.get("predictions"):
                print(f"[Autoloop] Speculator: {self._speculator_hint}")
        except Exception as exc:
            print(f"[Autoloop] Speculator skipped: {exc}")

        # 进度跟踪: 每轮迭代 6 个阶段 + 最终 report, 总步数按迭代数算
        tracker = self.progress_tracker or get_progress_tracker()
        total_steps = max_iterations * 6 + 1  # 6 phase/iter + 1 report
        progress_task_id = f"autoloop:{run_id}"
        tracker.start_task(
            task_id=progress_task_id,
            description=f"autoloop: {objective[:80]}",
            total_steps=total_steps,
            stage_labels=["perceive", "hypothesize", "plan", "execute", "validate", "learn", "report"],
            engine_kind="autoloop",
            metadata={"run_id": run_id, "objective": objective[:200]},
        )
        completed_steps = 0

        while self._iteration < max_iterations and not self._should_stop:
            self._iteration += 1
            print(f"\n[Autoloop] Iteration {self._iteration}/{max_iterations}: {objective}")

            # 1. Perceive
            phase = self._run_phase("perceive", self._perceive)
            phases.append(phase)
            completed_steps += 1
            tracker.update(progress_task_id, current_step=completed_steps,
                           current_label=f"iter {self._iteration}: perceive ({phase.status})")
            if not phase.result:
                print("  → No changes detected, waiting...")
                await asyncio.sleep(2)
                continue

            context = phase.result

            # 2. Hypothesize
            phase = await self._run_phase_async("hypothesize", self._hypothesize, context)
            phases.append(phase)
            completed_steps += 1
            tracker.update(progress_task_id, current_step=completed_steps,
                           current_label=f"iter {self._iteration}: hypothesize ({phase.status})")
            hypothesis = phase.result
            if not hypothesis:
                print("  → No hypothesis generated, skipping iteration")
                continue
            print(f"  → Hypothesis: {hypothesis}")

            # 3. Plan
            phase = await self._run_phase_async("plan", self._plan, hypothesis, context)
            phases.append(phase)
            completed_steps += 1
            tracker.update(progress_task_id, current_step=completed_steps,
                           current_label=f"iter {self._iteration}: plan ({phase.status})")
            plan = phase.result
            if not plan:
                print("  → No plan generated, skipping iteration")
                continue
            print(f"  → Plan: {plan['mode']} | {plan['description']}")

            # 4. Execute
            phase = await self._run_phase_async("execute", self._execute, plan, context)
            phases.append(phase)
            completed_steps += 1
            tracker.update(progress_task_id, current_step=completed_steps,
                           current_label=f"iter {self._iteration}: execute ({phase.status})")
            execution_result = phase.result
            if phase.error:
                print(f"  → Execution failed: {phase.error}")
                continue
            print(f"  → Execution complete: {execution_result}")

            # 5. Validate
            phase = await self._run_phase_async("validate", self._validate, execution_result)
            phases.append(phase)
            completed_steps += 1
            tracker.update(progress_task_id, current_step=completed_steps,
                           current_label=f"iter {self._iteration}: validate ({phase.status})")
            validation = phase.result
            print(f"  → Validation: {validation}")

            # 6. Learn
            phase = await self._run_phase_async("learn", self._learn, hypothesis, plan, validation)
            phases.append(phase)
            completed_steps += 1
            tracker.update(progress_task_id, current_step=completed_steps,
                           current_label=f"iter {self._iteration}: learn ({phase.status})")
            print(f"  → Learning complete")

        # 7. Report
        total_time = time.time() - start_time
        report_phase = await self._run_phase_async(
            "report", self._report, objective, phases, total_time
        )
        phases.append(report_phase)
        completed_steps += 1
        tracker.update(progress_task_id, current_step=completed_steps,
                       current_label=f"report ({report_phase.status})")

        # 标记完成: 只要 report 跑完就算 completed, 即使中间有 phase failed
        if report_phase.status == "completed":
            tracker.complete(progress_task_id, result={"report_path": report_phase.result})
        else:
            tracker.fail(progress_task_id, f"report phase failed: {report_phase.error}")

        return AutoloopResult(
            run_id=run_id,
            objective=objective,
            phases=phases,
            success=all(p.status == "completed" for p in phases[-7:]),
            report_path=report_phase.result,
            total_time_seconds=total_time,
        )

    def stop(self) -> None:
        """Signal the loop to stop at the next safe point."""
        self._should_stop = True

    # ──────────────────────────────────────────────────────────────
    # Phase implementations
    # ──────────────────────────────────────────────────────────────

    def _perceive(self) -> dict[str, Any] | None:
        """Perceive the workspace using the multi-modal perception layer."""
        from huginn.perception import PerceptionLayer
        
        perception = PerceptionLayer(self.workspace)
        perception.start()
        snapshot = perception.get_snapshot()
        perception.stop()
        
        context = snapshot.to_context()
        if not snapshot.has_activity():
            return None
        return context
    def _perceive_legacy(self) -> dict[str, Any] | None:
        """Legacy perceive (fallback)."""
        changed_files = []
        git_diff = ""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=self.workspace, capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                changed_files = [line.strip() for line in result.stdout.strip().split("\n")]
                git_diff = subprocess.run(
                    ["git", "diff", "--stat"],
                    cwd=self.workspace, capture_output=True, text=True, timeout=10,
                ).stdout
        except Exception:
            pass
        error_patterns = []
        for log_file in self.workspace.rglob("*.log"):
            if log_file.stat().st_mtime > time.time() - 3600:
                try:
                    content = log_file.read_text(errors="ignore")
                    if "ERROR" in content or "FAIL" in content:
                        error_patterns.append(f"{log_file.name}: {content[:200]}")
                except Exception:
                    pass
        if not changed_files and not error_patterns:
            return None
        return {
            "changed_files": changed_files,
            "git_diff": git_diff,
            "error_patterns": error_patterns,
            "timestamp": datetime.now().isoformat(),
        }

    async def _hypothesize(self, context: dict[str, Any]) -> str | None:
        """Generate a hypothesis from perceived context."""
        # Use knowledge graph + LLM to generate hypothesis
        prompt = self._build_hypothesis_prompt(context)
        # 按研究类型选 persona: MD 类用 md_expert, 默认走 dft_expert.
        # 这俩 persona 在 personas.py 内置, 直接取就行.
        persona_name = self._pick_hypothesis_persona(context)
        try:
            response = await self._llm_chat(prompt, persona_name=persona_name)
            return response.strip()
        except Exception:
            return None

    @staticmethod
    def _pick_hypothesis_persona(context: dict[str, Any]) -> str:
        """根据 context 内容判断走 DFT 还是 MD 专家 persona."""
        blob = json.dumps(context, ensure_ascii=False).lower()
        md_markers = ("md", "lammps", "molecular dynamics", "nvt", "npt", "md_steps")
        if any(m in blob for m in md_markers):
            return "md_expert"
        return "dft_expert"

    async def _plan(self, hypothesis: str, context: dict[str, Any]) -> dict[str, Any] | None:
        """Generate a plan from hypothesis."""
        # Determine which mode to use: coder, workflow, or exploration
        prompt = self._build_plan_prompt(hypothesis, context)
        try:
            response = await self._llm_chat(prompt, persona_name="default")
            plan = self._parse_plan(response)
            return plan
        except Exception:
            return None

    async def _execute(self, plan: dict[str, Any], context: dict[str, Any]) -> Any:
        """Execute the plan using the appropriate sub-engine."""
        mode = plan.get("mode", "coder")
        description = plan.get("description", "")

        if mode == "coder":
            # Use CoderRunner to modify code
            return await self._execute_coder(description, context)
        elif mode == "workflow":
            # Use WorkflowEngine to run computational pipeline
            return await self._execute_workflow(description, context)
        elif mode == "explore":
            # Use ExplorationOrchestrator to search design space
            return await self._execute_explore(description, context)
        else:
            raise ValueError(f"Unknown plan mode: {mode}")

    async def _validate(self, execution_result: Any) -> dict[str, Any]:
        """Validate execution results using benchmarks and constraints."""
        # Run quick validation checks
        results = {
            "tests_passed": False,
            "constraints_satisfied": False,
            "benchmarks": {},
        }

        # 物理校验: 执行结果带物理数据时跑 validate_tool 拿 R_phys
        # 这是阶段4 单轨奖励回流的入口——R_phys 会传给 _learn 喂 evolution
        if isinstance(execution_result, dict):
            r_phys = execution_result.get("r_phys")
            if r_phys is None:
                result_type = execution_result.get("result_type")
                result_data = execution_result.get("result_data")
                if result_type and result_data:
                    try:
                        from huginn.tools.validate_tool import (
                            ValidateTool,
                            ValidateToolInput,
                        )

                        validator = ValidateTool()
                        tool_ctx = ToolContext(
                            session_id=f"validate_{uuid.uuid4().hex[:8]}",
                            workspace=str(self.workspace),
                            config=self.settings,
                        )
                        vr = await validator.call(
                            ValidateToolInput(
                                result_type=result_type,
                                result_data=result_data,
                            ),
                            tool_ctx,
                        )
                        if vr.success and vr.data:
                            r_phys = vr.data.get("r_phys")
                            results["physics_validation"] = vr.data
                    except Exception as e:
                        results["physics_validation_error"] = str(e)
            if r_phys is not None:
                results["r_phys"] = r_phys

        # Try to run pytest on modified files
        try:
            import subprocess
            result = subprocess.run(
                ["python", "-m", "pytest", "-x", "-q", "--tb=line"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=60,
            )
            results["tests_passed"] = result.returncode == 0
            results["test_output"] = result.stdout + result.stderr
        except Exception as e:
            results["test_output"] = f"Test execution error: {e}"

        # Run bench if available
        try:
            runner = BenchmarkRunner()
            report = runner.run(categories=["math", "coding"])
            results["benchmarks"] = {
                "passed": report.passed,
                "failed": report.failed,
                "skipped": report.skipped,
            }
        except Exception:
            pass

        # Reviewer persona 批判性审视: 让 LLM 戴 reviewer 帽子点评本次结果.
        # 这是 validate 阶段接入对话层的关键点 — reviewer persona 的 system
        # prompt 会强约束 LLM 走"挑毛病 + 提改进"的语气, 而不是默认的助手语气.
        # 失败不影响 validation 流程, 只是不带 reviewer_critique 字段.
        try:
            critique = await self._llm_chat(
                self._build_reviewer_prompt(execution_result, results),
                persona_name="reviewer",
            )
            if critique and critique.strip():
                results["reviewer_critique"] = critique.strip()
        except Exception as e:
            results["reviewer_critique_error"] = str(e)

        return results

    @staticmethod
    def _build_reviewer_prompt(execution_result: Any, results: dict[str, Any]) -> str:
        """构造让 reviewer persona 点评执行结果的 prompt."""
        try:
            exec_blob = json.dumps(execution_result, ensure_ascii=False, default=str)[:1500]
        except Exception:
            exec_blob = str(execution_result)[:1500]
        try:
            res_blob = json.dumps(results, ensure_ascii=False, default=str)[:1500]
        except Exception:
            res_blob = str(results)[:1500]
        return (
            "Below is the execution result and validation summary from an "
            "autonomous materials-science research loop iteration.\n\n"
            f"Execution result:\n{exec_blob}\n\n"
            f"Validation summary:\n{res_blob}\n\n"
            "As a critical peer reviewer, point out:\n"
            "1. Any methodological weakness or missing convergence check.\n"
            "2. Whether the result is reproducible and benchmarked.\n"
            "3. Concrete next-step improvements.\n"
            "Be concise and direct."
        )

    async def _learn(self, hypothesis: str, plan: dict[str, Any], validation: dict[str, Any]) -> None:
        """Learn from iteration results — update memory, knowledge graph, evolution rules."""
        r_phys = validation.get("r_phys") if isinstance(validation, dict) else None

        # Log to memory
        self.memory.add_message(
            "system",
            {
                "iteration": self._iteration,
                "hypothesis": hypothesis,
                "plan": plan,
                "validation": validation,
                "r_phys": r_phys,
            },
        )

        # 奖励回流: 把 R_phys 喂给 evolution engine, 驱动基于奖励的进化
        # 这是阶段4 单轨的核心闭环——物理校验分数真正影响 agent 后续行为
        if r_phys is not None:
            try:
                evolution = self._get_evolution()
                # 记录本次迭代的 reward, 供 evolve_from_rewards 消费
                evolution.logger.log_tool_call(
                    session_id=f"loop_{self._iteration}",
                    tool_name=plan.get("mode", "unknown"),
                    tool_input={"hypothesis": hypothesis, "plan": plan},
                    result=validation,
                    reward=r_phys,
                )
                reward_result = evolution.evolve_from_rewards()
                n_skills = len(reward_result["high_reward_skills"])
                n_patches = len(reward_result["low_reward_patches"])
                if n_skills or n_patches:
                    print(
                        f"  → Reward evolution: +{n_skills} skills, +{n_patches} patches (R_phys={r_phys:.2f})"
                    )
            except Exception as e:
                print(f"  → Reward evolution failed: {e}")

    async def _report(self, objective: str, phases: list[LoopPhase], total_time: float) -> str | None:
        """Generate a final report summarizing the loop."""
        report_data = {
            "objective": objective,
            "run_id": f"loop_{uuid.uuid4().hex[:8]}",
            "total_time_seconds": total_time,
            "phases": [
                {
                    "name": p.name,
                    "status": p.status,
                    "duration": (p.end_time or 0) - (p.start_time or 0) if p.start_time and p.end_time else 0,
                    "error": p.error,
                }
                for p in phases
            ],
        }

        # Report 阶段接入 tutor persona: 让 LLM 用教学口吻写一段总结,
        # 帮助用户理解这轮 loop 做了什么、为什么这么做. 失败就退化为纯表格报告.
        tutor_narrative = ""
        try:
            tutor_narrative = await self._llm_chat(
                self._build_tutor_report_prompt(report_data),
                persona_name="tutor",
            )
            tutor_narrative = (tutor_narrative or "").strip()
        except Exception:
            tutor_narrative = ""

        # Save markdown report to workspace
        report_path = self.workspace / f"huginn_autoloop_report_{report_data['run_id']}.md"
        report_content = self._render_report(report_data)
        if tutor_narrative:
            report_content += "\n\n## Tutor's Summary\n\n" + tutor_narrative + "\n"
        report_path.write_text(report_content, encoding="utf-8")

        return str(report_path)

    @staticmethod
    def _build_tutor_report_prompt(report_data: dict[str, Any]) -> str:
        """构造让 tutor persona 写教学口吻总结的 prompt."""
        try:
            phases_blob = json.dumps(report_data["phases"], ensure_ascii=False)[:1200]
        except Exception:
            phases_blob = str(report_data.get("phases", ""))[:1200]
        return (
            "You just supervised an autonomous research loop. Summarize for a "
            "graduate student what happened, in a patient, pedagogical tone.\n\n"
            f"Objective: {report_data['objective']}\n"
            f"Total time: {report_data['total_time_seconds']:.1f}s\n"
            f"Phases:\n{phases_blob}\n\n"
            "Cover:\n"
            "- What the loop tried to achieve and why each phase matters.\n"
            "- Any phase that failed, and what a student should learn from it.\n"
            "- One concrete suggestion for the next iteration.\n"
            "Keep it under 200 words."
        )

    # ──────────────────────────────────────────────────────────────
    # Execution helpers
    # ──────────────────────────────────────────────────────────────

    async def _execute_coder(self, description: str, context: dict[str, Any]) -> dict[str, Any]:
        """Execute a coder task."""
        # Build a coding prompt from the description and context
        prompt = f"""Task: {description}

Context:
- Changed files: {context.get('changed_files', [])}
- Git diff: {context.get('git_diff', '')[:500]}

Please modify the code to address this task."""

        # Run CoderRunner
        # (Simplified — in production this would use the full CoderRunner loop)
        return {"mode": "coder", "prompt": prompt, "status": "submitted"}

    async def _execute_workflow(self, description: str, context: dict[str, Any]) -> dict[str, Any]:
        """Execute a workflow task."""
        # For now, use a standard DFT workflow as example
        # In production, dynamically select workflow template based on description
        try:
            # Find structure files in workspace
            structure_files = list(self.workspace.rglob("*.cif")) + list(self.workspace.rglob("*.poscar")) + list(self.workspace.rglob("*.vasp"))
            structure_path = str(structure_files[0]) if structure_files else "structure.cif"

            stages = standard_dft_workflow(structure_path, engine="vasp")
            tool_context = ToolContext(
                session_id=f"workflow_{uuid.uuid4().hex[:8]}",
                workspace=str(self.workspace),
                config=self.settings,
            )
            result = await self.workflow_engine.execute(stages, tool_context)
            return {"mode": "workflow", "success": result.success, "stages": len(stages)}
        except Exception as e:
            return {"mode": "workflow", "success": False, "error": str(e)}

    async def _execute_explore(self, description: str, context: dict[str, Any]) -> dict[str, Any]:
        """Execute an exploration task."""
        try:
            result = await self.explorer.explore(
                objective=description,
                initial_branches=[
                    {"name": "baseline", "hypothesis": f"Baseline for: {description}"}
                ],
                max_iterations=5,
            )
            return {
                "mode": "explore",
                "n_explored": result.n_branches_explored,
                "n_pruned": result.n_branches_pruned,
                "convergence": result.convergence_reason,
            }
        except Exception as e:
            return {"mode": "explore", "success": False, "error": str(e)}

    # ──────────────────────────────────────────────────────────────
    # LLM helpers
    # ──────────────────────────────────────────────────────────────

    async def _llm_chat(self, prompt: str, persona_name: str | None = None) -> str:
        """Send a prompt to the LLM and return the response.

        persona_name 不为空时, 把对应 persona 的 system prompt 作为
        SystemMessage 插在最前, 实现"每阶段开始注入 persona system prompt".
        persona 找不到就退化为不注入, 行为跟改动前一致.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages: list[Any] = []
        if persona_name:
            sys_prompt = self._persona_system_prompt(persona_name)
            if sys_prompt:
                messages.append(SystemMessage(content=sys_prompt))
        messages.append(HumanMessage(content=prompt))
        response = await self.model.ainvoke(messages)
        return str(response.content)

    def _build_hypothesis_prompt(self, context: dict[str, Any]) -> str:
        # 投机执行 hint: 基于历史预测的下一步意图, 注入给 LLM 参考
        # 预测只是 hint, LLM 可以无视, 不强制
        hint_block = ""
        if self._speculator_hint:
            hint_block = f"\nSpeculator hint (advisory, may be ignored): {self._speculator_hint}\n"
        return f"""You are an autonomous material science research agent.
{hint_block}
Perceived context:
{json.dumps(context, indent=2, ensure_ascii=False)[:2000]}

Generate a single, testable hypothesis about what should be done next.
The hypothesis should be a single sentence, concrete and actionable.

Hypothesis:"""

    def _build_plan_prompt(self, hypothesis: str, context: dict[str, Any]) -> str:
        return f"""Given the hypothesis: "{hypothesis}"

Context:
{json.dumps(context, indent=2, ensure_ascii=False)[:1000]}

Choose ONE mode and describe the plan:
- coder: modify code/files to fix or improve something
- workflow: run a computational simulation pipeline
- explore: search a design space for optimal parameters

Respond in this exact format:
MODE: <coder|workflow|explore>
DESCRIPTION: <brief description of what to do>
"""

    def _parse_plan(self, response: str) -> dict[str, Any]:
        """Parse LLM plan response."""
        mode = "coder"
        description = response.strip()

        for line in response.split("\n"):
            if line.startswith("MODE:"):
                mode = line.replace("MODE:", "").strip().lower()
            elif line.startswith("DESCRIPTION:"):
                description = line.replace("DESCRIPTION:", "").strip()

        return {"mode": mode, "description": description}

    # ──────────────────────────────────────────────────────────────
    # Phase runner utilities
    # ──────────────────────────────────────────────────────────────

    def _run_phase(self, name: str, fn, *args) -> LoopPhase:
        """Run a synchronous phase function."""
        phase = LoopPhase(name=name)
        phase.start_time = time.time()
        phase.status = "running"
        try:
            phase.result = fn(*args)
            phase.status = "completed"
        except Exception as e:
            phase.status = "failed"
            phase.error = str(e)
        phase.end_time = time.time()
        return phase

    async def _run_phase_async(self, name: str, fn, *args) -> LoopPhase:
        """Run an async phase function."""
        phase = LoopPhase(name=name)
        phase.start_time = time.time()
        phase.status = "running"
        try:
            phase.result = await fn(*args)
            phase.status = "completed"
        except Exception as e:
            phase.status = "failed"
            phase.error = str(e)
        phase.end_time = time.time()
        return phase

    # ──────────────────────────────────────────────────────────────
    # Report rendering
    # ──────────────────────────────────────────────────────────────

    def _render_report(self, data: dict[str, Any]) -> str:
        """Render a markdown report."""
        lines = [
            f"# Huginn Autoloop Report",
            f"",
            f"**Objective:** {data['objective']}",
            f"**Run ID:** {data['run_id']}",
            f"**Total Time:** {data['total_time_seconds']:.1f}s",
            f"",
            f"## Phases",
            f"",
            f"| Phase | Status | Duration (s) | Error |",
            f"|-------|--------|--------------|-------|",
        ]
        for p in data["phases"]:
            lines.append(f"| {p['name']} | {p['status']} | {p['duration']:.1f} | {p['error'] or ''} |")
        lines.append("")
        lines.append("---")
        lines.append("Generated by Huginn Autoloop Engine")
        return "\n".join(lines)
