"""Computational workflow engine.

Orchestrates multi-stage computational pipelines with dependency management,
validation, and retry policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Any
from datetime import datetime
import asyncio

from huginn.types import ToolContext, ToolResult


@dataclass
class ValidationRule:
    """Rule for validating stage output."""
    check: Literal["convergence", "energy_sign", "force_threshold", "custom"]
    threshold: float | None = None
    custom_fn: str | None = None  # Name of validation function


@dataclass
class RetryPolicy:
    """Retry policy for failed stages."""
    max_retries: int = 2
    backoff_factor: float = 2.0
    retry_on: list[Literal["convergence_fail", "timeout", "oom", "any"]] = field(
        default_factory=lambda: ["convergence_fail", "timeout"]
    )
    auto_diagnose: bool = True  # Whether to call diagnose_tool before retry
    apply_auto_fix: bool = True  # Whether to apply suggested fixes from diagnosis


@dataclass
class ComputationalStage:
    """Single stage in a computational workflow."""
    id: str
    name: str
    tool: str  # Tool name to invoke
    tool_input: dict[str, Any]
    dependencies: list[str] = field(default_factory=list)
    validation: ValidationRule | None = None
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    
    # Execution state
    status: Literal["pending", "running", "completed", "failed", "skipped"] = "pending"
    result: ToolResult | None = None
    attempts: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class WorkflowResult:
    """Result of a complete workflow execution."""
    success: bool
    stages: dict[str, ComputationalStage]
    outputs: dict[str, Any]
    error: str | None = None
    total_walltime: float = 0.0  # seconds


class WorkflowEngine:
    """Engine for executing computational workflows."""
    
    def __init__(self, tool_registry: Any):
        self.registry = tool_registry
    
    async def execute(
        self,
        stages: list[ComputationalStage],
        context: ToolContext
    ) -> WorkflowResult:
        """Execute a workflow with topological ordering and parallelization."""
        
        stage_map = {s.id: s for s in stages}
        completed: set[str] = set()
        failed: set[str] = set()
        outputs: dict[str, Any] = {}
        
        start_time = datetime.now()
        
        while len(completed) + len(failed) < len(stages):
            # Find stages whose dependencies are all satisfied
            ready = [
                s for s in stages
                if s.id not in completed and s.id not in failed
                and all(dep in completed for dep in s.dependencies)
            ]
            
            if not ready:
                # Deadlock or all remaining blocked by failures
                remaining = [s.id for s in stages if s.id not in completed and s.id not in failed]
                return WorkflowResult(
                    success=False,
                    stages=stage_map,
                    outputs=outputs,
                    error=f"Workflow blocked: stages {remaining} have unsatisfied dependencies"
                )
            
            # Execute ready stages in parallel
            tasks = [self._execute_stage(s, context, outputs) for s in ready]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for stage, result in zip(ready, results):
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
                    if stage.attempts < stage.retry_policy.max_retries:
                        stage.status = "pending"
                        stage.attempts += 1
                        
                        # Auto-diagnose and apply fixes if enabled
                        if stage.retry_policy.auto_diagnose:
                            await self._diagnose_and_fix(stage, result, context)
                        
                        # Backoff delay
                        delay = stage.retry_policy.backoff_factor ** stage.attempts
                        await asyncio.sleep(min(delay, 30))  # Cap at 30s
                    else:
                        stage.status = "failed"
                        failed.add(stage.id)
        
        total_time = (datetime.now() - start_time).total_seconds()
        success = len(failed) == 0
        
        return WorkflowResult(
            success=success,
            stages=stage_map,
            outputs=outputs,
            error=f"Stages failed: {failed}" if failed else None,
            total_walltime=total_time
        )
    
    async def _execute_stage(
        self,
        stage: ComputationalStage,
        context: ToolContext,
        available_outputs: dict[str, Any]
    ) -> ToolResult:
        """Execute a single stage."""
        stage.status = "running"
        stage.started_at = datetime.now()
        
        # Resolve inputs from dependency outputs
        tool_input = self._resolve_inputs(stage.tool_input, available_outputs)
        
        tool = self.registry.get(stage.tool)
        if not tool:
            return ToolResult(
                data=None,
                success=False,
                error=f"Tool '{stage.tool}' not found"
            )
        
        # Convert dict to tool's Pydantic input schema
        if hasattr(tool, 'input_schema') and tool.input_schema:
            try:
                tool_input = tool.input_schema(**tool_input)
            except Exception as e:
                stage.completed_at = datetime.now()
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Invalid input for '{stage.tool}': {e}"
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
    
    def _resolve_inputs(
        self,
        tool_input: dict[str, Any],
        available_outputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve input references like '${stage_id.output_key}'."""
        resolved = {}
        for key, value in tool_input.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
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
    
    async def _diagnose_and_fix(
        self,
        stage: ComputationalStage,
        result: ToolResult,
        context: ToolContext
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
                from huginn.tools.diagnose_tool import DiagnoseToolInput
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
        if "qe" in tool_lower or "quantum espresso" in tool_lower or "pw.x" in tool_lower:
            return "QuantumESPRESSO"
        if "gromacs" in tool_lower:
            return "GROMACS"
        # Try to infer from tool_input
        params = json.dumps(stage.tool_input).lower()
        for sw in ["gaussian", "orca", "vasp", "lammps", "cp2k", "qe", "gromacs", "multiwfn"]:
            if sw in params:
                return sw.title()
        return None
    
    def _detect_calculation_type_from_stage(self, stage: ComputationalStage) -> str | None:
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
        self,
        stage: ComputationalStage,
        fixes: dict[str, Any],
        software: str | None
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
        
        return True  # Custom validation not yet implemented
