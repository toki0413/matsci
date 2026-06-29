"""Computational workflow engine.

Orchestrates multi-stage computational pipelines with dependency management,
validation, and retry policies.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from huginn.interaction.progress import ProgressTracker, get_progress_tracker
from huginn.queue import InMemoryTaskBackend, TaskBackend
from huginn.types import (
    BudgetDecision,
    BudgetPolicy,
    CostEstimate,
    ToolContext,
    ToolResult,
)
from huginn.workflows.checkpoint import WorkflowCheckpoint
from huginn.workflows.stages import (
    ComputationalStage,
    RetryPolicy,
    ValidationRule,
    WorkflowResult,
)

# Re-export dataclasses so existing imports keep working.
__all__ = [
    "WorkflowEngine",
    "ComputationalStage",
    "ValidationRule",
    "RetryPolicy",
    "WorkflowResult",
]


class WorkflowEngine:
    """Engine for executing computational workflows."""

    def __init__(
        self,
        tool_registry: Any,
        budget_policy: BudgetPolicy | None = None,
        task_backend: TaskBackend | None = None,
        persona_manager: Any | None = None,
        long_term_memory: Any | None = None,
        skill_registry: Any | None = None,
        progress_tracker: ProgressTracker | None = None,
    ):
        self.registry = tool_registry
        self.budget_policy = budget_policy
        self.task_backend = task_backend or InMemoryTaskBackend()
        self.task_backend.register_task(
            "huginn.workflow.stage", self._execute_stage_sync
        )
        # 对话层组件 — 不传就跳过对应注入, 保持老 workflow 行为.
        # 传了才会在 _apply_stage_context 里被读取, 这样老调用方完全无感.
        self.persona_manager = persona_manager
        self.long_term_memory = long_term_memory
        self.skill_registry = skill_registry
        # 进度跟踪: 默认走进程级单例, 让 /tasks 路由能汇总所有引擎的进度.
        # 测试时可以传独立 tracker 隔离. None 时退化为不跟踪 (老行为).
        self.progress_tracker = progress_tracker

    async def execute(
        self,
        stages: list[ComputationalStage],
        context: ToolContext,
        checkpoint_path: str | Path | None = None,
        budget_policy: BudgetPolicy | None = None,
    ) -> WorkflowResult:
        """Execute a workflow with topological ordering and parallelization.

        If ``checkpoint_path`` is provided, the engine writes a snapshot after
        every stage transition so the workflow can be resumed later.
        """

        stage_map = {s.id: s for s in stages}
        completed: set[str] = {s.id for s in stages if s.status == "completed"}
        failed: set[str] = {s.id for s in stages if s.status == "failed"}
        outputs: dict[str, Any] = {
            s.id: s.result.data
            for s in stages
            if s.status == "completed" and s.result is not None
        }

        start_time = datetime.now()

        # 登记进度任务: 用 session_id + uuid 区分, 避免跟其它 workflow 撞
        tracker = self.progress_tracker or get_progress_tracker()
        progress_task_id = f"{context.session_id}:workflow:{uuid.uuid4().hex[:8]}"
        stage_labels = [s.id for s in stages]
        tracker.start_task(
            task_id=progress_task_id,
            description=f"workflow with {len(stages)} stages",
            total_steps=len(stages),
            stage_labels=stage_labels,
            engine_kind="workflow",
            metadata={"session_id": context.session_id},
        )

        def _maybe_checkpoint() -> None:
            if checkpoint_path is not None:
                WorkflowCheckpoint(
                    stages=list(stage_map.values()), outputs=outputs
                ).save(checkpoint_path)

        while len(completed) + len(failed) < len(stages):
            # Find stages whose dependencies are all satisfied
            ready = [
                s
                for s in stages
                if s.id not in completed
                and s.id not in failed
                and all(dep in completed for dep in s.dependencies)
            ]

            if not ready:
                # Deadlock or all remaining blocked by failures
                remaining = [
                    s.id for s in stages if s.id not in completed and s.id not in failed
                ]
                _maybe_checkpoint()
                tracker.fail(
                    progress_task_id,
                    f"Workflow blocked: stages {remaining} have unsatisfied dependencies",
                )
                return WorkflowResult(
                    success=False,
                    stages=stage_map,
                    outputs=outputs,
                    error=f"Workflow blocked: stages {remaining} have unsatisfied dependencies",
                )

            # Budget check for ready stages
            effective_budget = budget_policy or self.budget_policy
            allowed_ready: list[ComputationalStage] = []
            for stage in ready:
                if effective_budget is None:
                    allowed_ready.append(stage)
                    continue
                estimate = self._estimate_stage_cost(stage)
                decision, reason = effective_budget.check(estimate)
                if decision == BudgetDecision.DENY:
                    stage.status = "failed"
                    stage.result = ToolResult(
                        data=None,
                        success=False,
                        error=f"Budget denied: {reason}",
                    )
                    failed.add(stage.id)
                elif decision == BudgetDecision.WARN:
                    # Proceed but surface the warning
                    stage.tool_input.setdefault("__budget_warnings", []).append(reason)
                    allowed_ready.append(stage)
                else:
                    allowed_ready.append(stage)

            if not allowed_ready:
                _maybe_checkpoint()
                continue

            # Execute ready stages in parallel, optionally through a task backend.
            if self.task_backend is not None:
                results = await self._dispatch_stages(allowed_ready, context, outputs)
            else:
                tasks = [
                    self._execute_stage(s, context, outputs) for s in allowed_ready
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

            for stage, result in zip(allowed_ready, results):
                if isinstance(result, Exception):
                    stage.status = "failed"
                    failed.add(stage.id)
                elif result.success:
                    stage.status = "completed"
                    stage.result = result
                    outputs[stage.id] = result.data
                    completed.add(stage.id)
                else:
                    # Check retry policy
                    if self._should_retry(stage, result.error):
                        stage.status = "pending"
                        stage.attempts += 1

                        # Auto-diagnose and apply fixes if enabled
                        if stage.retry_policy.auto_diagnose:
                            await self._diagnose_and_fix(stage, result, context)

                        # Backoff delay
                        delay = stage.retry_policy.backoff_factor**stage.attempts
                        await asyncio.sleep(min(delay, 30))  # Cap at 30s
                    else:
                        stage.status = "failed"
                        failed.add(stage.id)

                _maybe_checkpoint()
                # 每个 stage 落定后更新进度 (含失败和重试中的)
                tracker.update(
                    progress_task_id,
                    current_step=len(completed) + len(failed),
                    current_label=f"stage {stage.id}: {stage.status}",
                    metadata={
                        "completed": sorted(completed),
                        "failed": sorted(failed),
                    },
                )

        _maybe_checkpoint()
        total_time = (datetime.now() - start_time).total_seconds()
        success = len(failed) == 0

        # 标记整个 workflow 完成 / 失败
        if success:
            tracker.complete(progress_task_id, result={"outputs": list(outputs.keys())})
        else:
            tracker.fail(progress_task_id, f"Stages failed: {sorted(failed)}")

        return WorkflowResult(
            success=success,
            stages=stage_map,
            outputs=outputs,
            error=f"Stages failed: {failed}" if failed else None,
            total_walltime=total_time,
        )

    async def resume(
        self,
        stages: list[ComputationalStage],
        context: ToolContext,
        checkpoint_path: str | Path,
        budget_policy: BudgetPolicy | None = None,
    ) -> WorkflowResult:
        """Resume a workflow from a saved checkpoint.

        The provided ``stages`` are used as the template; their runtime state is
        overlaid with whatever is stored in the checkpoint.
        """
        checkpoint = WorkflowCheckpoint.load(checkpoint_path)
        checkpoint_map = {s.id: s for s in checkpoint.stages}

        restored_stages: list[ComputationalStage] = []
        for stage in stages:
            if stage.id in checkpoint_map:
                restored_stages.append(checkpoint_map[stage.id])
            else:
                restored_stages.append(stage)

        return await self.execute(
            restored_stages,
            context,
            checkpoint_path=checkpoint_path,
            budget_policy=budget_policy,
        )

    def _should_retry(self, stage: ComputationalStage, error: str | None) -> bool:
        """Check whether a failed stage should be retried."""
        if stage.attempts >= stage.retry_policy.max_retries:
            return False

        error_lower = (error or "").lower()
        retry_on = set(stage.retry_policy.retry_on)
        if "any" in retry_on:
            return True

        if "timeout" in retry_on and any(
            tag in error_lower for tag in ("timeout", "timed out", "walltime")
        ):
            return True

        if "oom" in retry_on and any(
            tag in error_lower for tag in ("memory", "oom", "out of memory")
        ):
            return True

        if "remote_failure" in retry_on and self._is_remote_failure(error):
            return True

        # Default: retry on any non-empty error when policy allows it
        return bool(error)

    @staticmethod
    def _is_remote_failure(error: str | None) -> bool:
        """Detect transient remote-execution failures worth retrying."""
        if not error:
            return False
        error_lower = error.lower()
        remote_markers = (
            "ssh",
            "connection",
            "connection refused",
            "timed out",
            "timeout",
            "eof occurred",
            "temporarily unavailable",
            "slurm",
            "pbs",
            "qsub",
            "sbatch",
            "remote execution failed",
            "node failure",
            "job killed",
        )
        return any(marker in error_lower for marker in remote_markers)

    def _estimate_stage_cost(self, stage: ComputationalStage) -> CostEstimate:
        """Heuristic cost estimate for a workflow stage.

        Uses explicit resource hints in tool_input when available; otherwise
        falls back to rough per-tool defaults.
        """
        inp = stage.tool_input
        tool_lower = stage.tool.lower()

        walltime_hours = float(
            inp.get("walltime_hours")
            or inp.get("walltime", "24:00:00").split(":")[0]
            or 1.0
        )
        nodes = int(inp.get("nodes", 1))
        ntasks = int(inp.get("ntasks_per_node", inp.get("cores", 4)))
        memory_gb = float(inp.get("memory_gb", nodes * ntasks * 2))
        storage_gb = float(inp.get("storage_gb", 1.0))

        if "vasp" in tool_lower:
            # Larger k-grid / encut / MD steps scale CPU work
            encut = float(inp.get("encut", 520))
            kpts = inp.get("kpoints", "1 1 1")
            n_kpts = 1
            if isinstance(kpts, str):
                try:
                    parts = [int(x) for x in kpts.split() if x.isdigit()]
                    n_kpts = max(
                        1, (parts[1] * parts[2] * parts[3]) if len(parts) >= 4 else 1
                    )
                except Exception:
                    n_kpts = 1
            walltime_hours *= max(1.0, encut / 520.0) * max(1.0, n_kpts / 8.0)
        elif "lammps" in tool_lower:
            n_steps = int(inp.get("n_steps", 1000))
            walltime_hours *= max(1.0, n_steps / 1000.0)
        elif "aimd" in tool_lower or "md" in tool_lower:
            n_steps = int(inp.get("n_steps", inp.get("md_steps", 1000)))
            walltime_hours *= max(1.0, n_steps / 1000.0)

        cpu_hours = nodes * ntasks * walltime_hours
        gpu_hours = (
            walltime_hours if inp.get("queue") == "gpu" or "gpu" in tool_lower else 0.0
        )

        return CostEstimate(
            cpu_hours=cpu_hours,
            gpu_hours=gpu_hours,
            memory_gb=memory_gb,
            storage_gb=storage_gb,
            walltime_hours=walltime_hours,
        )

    async def _execute_stage(
        self,
        stage: ComputationalStage,
        context: ToolContext,
        available_outputs: dict[str, Any],
    ) -> ToolResult:
        """Execute a single stage."""
        stage.status = "running"
        stage.started_at = datetime.now()

        # Resolve inputs from dependency outputs
        tool_input = self._resolve_inputs(stage.tool_input, available_outputs)

        # 接入对话层组件 — persona / emotion / memory / skill 白名单.
        # 这些字段不填 (默认 None) 就完全跳过, 老 workflow 行为不变.
        tool_input, ctx_error = self._apply_stage_context(stage, tool_input, context)
        if ctx_error is not None:
            stage.completed_at = datetime.now()
            return ToolResult(data=None, success=False, error=ctx_error)

        tool = self.registry.get(stage.tool)
        if not tool:
            return ToolResult(
                data=None, success=False, error=f"Tool '{stage.tool}' not found"
            )

        # Convert dict to tool's Pydantic input schema
        if hasattr(tool, "input_schema") and tool.input_schema:
            try:
                tool_input = tool.input_schema(**tool_input)
            except Exception as e:
                stage.completed_at = datetime.now()
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Invalid input for '{stage.tool}': {e}",
                )

        # Validate and execute
        result = await tool.call(tool_input, context)

        stage.completed_at = datetime.now()

        # Apply validation
        if result.success and stage.validation:
            valid = self._validate(result.data, stage.validation)
            if not valid:
                result.success = False
                result.error = f"Validation failed: {stage.validation.check}"

        return result

    def _execute_stage_sync(
        self,
        stage: ComputationalStage,
        context: ToolContext,
        available_outputs: dict[str, Any],
    ) -> ToolResult:
        """Synchronous wrapper used by task backends to run an async stage."""
        return asyncio.run(self._execute_stage(stage, context, available_outputs))

    async def _dispatch_stages(
        self,
        stages: list[ComputationalStage],
        context: ToolContext,
        available_outputs: dict[str, Any],
    ) -> list[Any]:
        """Submit ready stages to the task backend and await their results."""
        task_ids: list[tuple[ComputationalStage, str]] = []
        for stage in stages:
            task_id = self.task_backend.send_task(
                "huginn.workflow.stage",
                args=(stage, context, available_outputs),
                task_id=f"{context.session_id}:{stage.id}:{stage.attempts}",
            )
            task_ids.append((stage, task_id))

        raw_results = await asyncio.gather(
            *[
                asyncio.to_thread(self.task_backend.wait_for, task_id)
                for _, task_id in task_ids
            ],
            return_exceptions=True,
        )

        results: list[Any] = []
        for (_, task_id), raw in zip(task_ids, raw_results):
            if isinstance(raw, Exception):
                results.append(raw)
                continue
            from huginn.queue.base import TaskResult

            if not isinstance(raw, TaskResult):
                results.append(RuntimeError(f"Unexpected task result for {task_id}"))
                continue
            if raw.status == "SUCCESS":
                results.append(raw.result)
            else:
                results.append(RuntimeError(raw.error or f"Task {task_id} failed"))
        return results

    def _resolve_inputs(
        self, tool_input: dict[str, Any], available_outputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve input references like '${stage_id.output_key}'."""
        resolved = {}
        for key, value in tool_input.items():
            if (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
                # Reference to another stage's output
                ref = value[2:-1]  # stage_id.output_key
                parts = ref.split(".")
                stage_id = parts[0]
                output_key = ".".join(parts[1:]) if len(parts) > 1 else None

                if stage_id in available_outputs:
                    stage_output = available_outputs[stage_id]
                    if output_key and isinstance(stage_output, dict):
                        resolved[key] = stage_output.get(output_key)
                    else:
                        resolved[key] = stage_output
                else:
                    resolved[key] = value  # Leave unresolved
            else:
                resolved[key] = value
        return resolved

    def _apply_stage_context(
        self,
        stage: ComputationalStage,
        tool_input: dict[str, Any],
        context: ToolContext,
    ) -> tuple[dict[str, Any], str | None]:
        """把 stage 声明的对话层组件注入到 tool_input 里.

        返回 (maybe_modified_tool_input, error). error 非 None 表示该 stage
        不应该执行 (目前只有 skill 白名单不通过会触发), 调用方直接返回失败即可.

        四个字段全是 opt-in: stage 不设或引擎没装对应组件, 就完全跳过,
        老 workflow 的执行路径不变.
        """
        # skill 白名单: stage.tool 必须在所列 skill 的 required_tools 集合里.
        # skill 名查不到就忽略 — 不卡 stage, 避免改名后老 workflow 挂掉.
        if stage.skill_context and self.skill_registry is not None:
            allowed_tools: set[str] = set()
            for skill_name in stage.skill_context:
                skill = self.skill_registry.get(skill_name)
                if skill is not None:
                    allowed_tools.update(getattr(skill, "required_tools", []) or [])
            if allowed_tools and stage.tool not in allowed_tools:
                return tool_input, (
                    f"Stage tool '{stage.tool}' not allowed by skill_context "
                    f"(allowed: {sorted(allowed_tools)})"
                )

        # persona: 取 system prompt, 作为附加上下文塞进 tool_input.
        # 工具自己决定要不要消费 (跟 __diagnosis 一个套路).
        if stage.persona and self.persona_manager is not None:
            try:
                persona = self.persona_manager.get(stage.persona)
                if persona and persona.system_prompt:
                    tool_input["__persona_prompt"] = persona.system_prompt
            except Exception:
                # persona 找不到不要把 stage 整个挂掉, 跟现有诊断钩子的容错一致
                pass

        # emotion_state: dict 转 EmotionState, 生成情绪片段注入.
        # 不走持久化, 只用临时 tracker 复用 context_prompt 的逻辑.
        if stage.emotion_state:
            snippet = self._emotion_state_to_prompt(stage.emotion_state)
            if snippet:
                tool_input["__emotion_context"] = snippet

        # memory_scope: 从 LongTermMemory 检索相关记忆, 摘要注入.
        # 走 FTS 即可, 不强求语义检索 (语义检索依赖 vector store, 不一定在线).
        if stage.memory_scope and self.long_term_memory is not None:
            try:
                memories = self.long_term_memory.retrieve(
                    stage.memory_scope, top_k=3, semantic=False
                )
                if memories:
                    tool_input["__memory_context"] = memories
            except Exception:
                pass

        return tool_input, None

    @staticmethod
    def _emotion_state_to_prompt(state_dict: dict[str, Any]) -> str:
        """把 emotion_state dict 转成可注入的简短情绪片段.

        复用 EmotionTracker.context_prompt 的措辞逻辑, 但不落盘, 也不动
        用户 workspace 下的 emotion 文件. 拿不到相关模块就直接返回空串.
        """
        try:
            import tempfile

            from huginn.persona_emotion import EmotionState, EmotionTracker

            state = EmotionState.from_dict(state_dict)
            # 用临时目录当 workspace, 避免在用户项目下创建 .huginn/emotion/
            tracker = EmotionTracker(
                persona_name="_workflow_stage",
                workspace=tempfile.gettempdir(),
            )
            tracker._state = state
            return tracker.context_prompt()
        except Exception:
            return ""

    async def _diagnose_and_fix(
        self, stage: ComputationalStage, result: ToolResult, context: ToolContext
    ) -> None:
        """Diagnose stage failure and apply automatic fixes to tool_input.

        Uses diagnose_tool to analyze error messages, then maps suggested
        fixes to the stage's tool_input for the next retry attempt.
        Also queries Sobko knowledge base for domain-specific guidance.
        """
        if not stage.retry_policy.auto_diagnose:
            return

        error_msg = result.error or "Unknown error"

        # Detect software and calculation type from stage context
        software = self._detect_software_from_stage(stage)
        calc_type = self._detect_calculation_type_from_stage(stage)

        # Call diagnose_tool with error message
        diagnose_tool = self.registry.get("diagnose_tool")
        if diagnose_tool:
            try:
                from huginn.tools.diagnose_tool import DiagnoseInput as DiagnoseToolInput

                diag_input = DiagnoseToolInput(
                    error_message=error_msg,
                    software=software,
                    calculation_type=calc_type,
                    context=json.dumps(stage.tool_input, ensure_ascii=False)[:500],
                )
                diag_result = await diagnose_tool.call(diag_input, context)

                if diag_result.success and diag_result.data:
                    report = diag_result.data
                    findings = report.get("findings", [])
                    general_advice = report.get("general_advice", [])
                    next_steps = report.get("recommended_next_steps", [])

                    # Store diagnosis in stage metadata for logging
                    stage.tool_input["__diagnosis"] = {
                        "findings": findings[:3],
                        "advice": general_advice[:3],
                        "next_steps": next_steps[:3],
                    }

                    # Apply heuristic auto-fixes based on findings
                    if stage.retry_policy.apply_auto_fix:
                        fixes = self._extract_fixes_from_diagnosis(report, software)
                        if fixes:
                            self._apply_fixes_to_stage(stage, fixes, software)

            except Exception:
                pass  # Silently ignore diagnosis errors

        # Also query Sobko knowledge base for additional context
        rag_tool = self.registry.get("rag_tool")
        if rag_tool:
            try:
                from huginn.rag.rag_tool import RAGToolInput

                rag_input = RAGToolInput(
                    action="search",
                    query=f"{software} {error_msg[:100]} 解决方法",
                    top_k=3,
                )
                rag_result = await rag_tool.call(rag_input, context)
                if rag_result.success and rag_result.data:
                    results = rag_result.data.get("results", [])
                    if results:
                        kb_hints = [r.get("text", "")[:200] for r in results[:2]]
                        stage.tool_input["__sobko_hints"] = kb_hints
            except Exception:
                pass

    def _detect_software_from_stage(self, stage: ComputationalStage) -> str | None:
        """Detect which software this stage uses."""
        tool_lower = stage.tool.lower()
        if "vasp" in tool_lower:
            return "VASP"
        if "lammps" in tool_lower:
            return "LAMMPS"
        if "gaussian" in tool_lower or "g16" in tool_lower:
            return "Gaussian"
        if "orca" in tool_lower:
            return "ORCA"
        if "multiwfn" in tool_lower:
            return "Multiwfn"
        if "cp2k" in tool_lower:
            return "CP2K"
        if (
            "qe" in tool_lower
            or "quantum espresso" in tool_lower
            or "pw.x" in tool_lower
        ):
            return "QuantumESPRESSO"
        if "gromacs" in tool_lower:
            return "GROMACS"
        # Try to infer from tool_input
        params = json.dumps(stage.tool_input).lower()
        for sw in [
            "gaussian",
            "orca",
            "vasp",
            "lammps",
            "cp2k",
            "qe",
            "gromacs",
            "multiwfn",
        ]:
            if sw in params:
                return sw.title()
        return None

    def _detect_calculation_type_from_stage(
        self, stage: ComputationalStage
    ) -> str | None:
        """Detect calculation type from stage context."""
        params = json.dumps(stage.tool_input).lower()
        if any(k in params for k in ["scf", "dft", "pbe", "b3lyp"]):
            return "DFT"
        if any(k in params for k in ["md", "nvt", "npt", "molecular dynamics"]):
            return "MD"
        if any(k in params for k in ["tddft", "excited", "cis"]):
            return "TDDFT"
        if any(k in params for k in ["opt", "relax", "geometry"]):
            return "geometry_optimization"
        if any(k in params for k in ["band", "dos"]):
            return "band_structure"
        return None

    def _extract_fixes_from_diagnosis(
        self, report: dict[str, Any], software: str | None
    ) -> dict[str, Any]:
        """Extract actionable fixes from diagnosis report."""
        fixes: dict[str, Any] = {}
        all_text = " ".join(
            f.get("text", "") for f in report.get("findings", [])
        ).lower()

        # VASP-specific fixes
        if software == "VASP":
            if "converg" in all_text or "scf" in all_text:
                fixes["ALGO"] = "Normal"
                fixes["NELMIN"] = 6
            if "memory" in all_text or "oom" in all_text:
                fixes["NCORE"] = 4
            if "band" in all_text:
                fixes["NBANDS"] = "__increase_20pct"

        # Gaussian-specific fixes
        elif software == "Gaussian":
            if "converg" in all_text:
                fixes["scf"] = "(xqc,MaxCycle=128)"
            if "td" in all_text or "excited" in all_text:
                fixes["IOp(9/40)"] = 4

        # LAMMPS-specific fixes
        elif software == "LAMMPS":
            if "timestep" in all_text:
                fixes["timestep"] = "__reduce_half"
            if "neighbor" in all_text:
                fixes["neighbor"] = "0.3 bin"

        # General fixes
        if "basis" in all_text or "diffuse" in all_text:
            fixes["__recommend_diffuse"] = True
        if "smear" in all_text:
            fixes["__check_smearing"] = True
        if "grid" in all_text:
            fixes["__increase_grid"] = True

        return fixes

    def _apply_fixes_to_stage(
        self, stage: ComputationalStage, fixes: dict[str, Any], software: str | None
    ) -> None:
        """Apply extracted fixes to stage tool_input."""
        if software == "VASP":
            overrides = stage.tool_input.get("incar_overrides", {})
            if not isinstance(overrides, dict):
                overrides = {}
            for key, value in fixes.items():
                if not key.startswith("_"):
                    overrides[key] = value
            stage.tool_input["incar_overrides"] = overrides
            stage.tool_input["__diagnosis_applied"] = list(fixes.keys())

        elif software in ("Gaussian", "ORCA"):
            params = stage.tool_input.get("params", {})
            if not isinstance(params, dict):
                params = {}
            for key, value in fixes.items():
                if not key.startswith("_"):
                    params[key] = value
            stage.tool_input["params"] = params
            stage.tool_input["__diagnosis_applied"] = list(fixes.keys())

        elif software == "LAMMPS":
            existing = stage.tool_input.get("fixes", {})
            if not isinstance(existing, dict):
                existing = {}
            existing.update(fixes)
            stage.tool_input["fixes"] = existing
            stage.tool_input["__diagnosis_applied"] = list(existing.keys())

        else:
            # Generic: store in metadata
            stage.tool_input["__auto_fixes"] = fixes

    # Registry for custom validation functions
    _custom_validators: dict[str, Any] = {}

    def register_validator(self, name: str, fn: Any) -> None:
        """Register a custom validation function for workflow rules."""
        self._custom_validators[name] = fn

    def _validate(self, data: Any, rule: ValidationRule) -> bool:
        """Apply validation rule to stage output."""
        if rule.check == "convergence":
            if isinstance(data, dict):
                return data.get("converged", False)
            return False

        if rule.check == "energy_sign":
            if isinstance(data, dict) and "energy" in data:
                return data["energy"] < 0
            return True  # Can't validate, assume ok

        if rule.check == "force_threshold":
            if isinstance(data, dict) and "max_force" in data:
                threshold = rule.threshold or 0.01
                return data["max_force"] < threshold
            return True

        if rule.check == "custom" and rule.custom_fn:
            fn = self._custom_validators.get(rule.custom_fn)
            if fn is None:
                # Try resolving as a dotted module path
                fn = self._resolve_custom_fn(rule.custom_fn)
            if fn is not None:
                try:
                    if rule.threshold is not None:
                        return bool(fn(data, threshold=rule.threshold))
                    return bool(fn(data))
                except Exception:
                    return False
            # Unknown validator — fail closed for safety
            return False

        return True  # Unknown check type — pass through

    @staticmethod
    def _resolve_custom_fn(dotted_name: str) -> Any | None:
        """Resolve a dotted name like 'huginn.validators.check_energy' to a callable."""
        parts = dotted_name.rsplit(".", 1)
        if len(parts) != 2:
            return None
        module_path, fn_name = parts
        try:
            import importlib

            mod = importlib.import_module(module_path)
            return getattr(mod, fn_name, None)
        except (ImportError, AttributeError):
            return None
