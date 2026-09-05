"""Execution Orchestrator — turns workflow stages into real actions.

Replaces the "describe what to do" pattern with "actually do it".

Key capabilities:
  1. Dependency resolution: stages run in topological order
  2. Async execution: I/O-bound stages run in parallel where possible
  3. Progress tracking: each stage reports status, stdout, stderr
  4. Failure isolation: one stage failure doesn't cascade unless specified
  5. Checkpointing: save/resume long-running workflows
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from huginn.execution.compute_router import ComputeRouter

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Result of executing a single workflow stage."""

    stage_id: str
    stage_name: str
    tool_name: str
    success: bool
    output_data: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    stdout: str = ""
    stderr: str = ""
    walltime_seconds: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    retry_count: int = 0
    auto_fixed: bool = False
    fix_applied: str | None = None
    execution_target: str | None = None
    route_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowExecutionRecord:
    """Complete record of a workflow execution."""

    workflow_name: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at: str | None = None
    stage_results: list[StageResult] = field(default_factory=list)
    overall_success: bool = False
    working_directory: str = ""


class ExecutionOrchestrator:
    """Orchestrates the execution of multi-stage computational workflows.

    Usage:
        orch = ExecutionOrchestrator(working_dir="./my_calc")
        stages = [
            {"id": "relax", "tool": "vasp_tool", "action": "relax", "params": {...}},
            {"id": "band", "tool": "vasp_tool", "action": "band", "params": {...},
             "depends_on": ["relax"]},
        ]
        record = await orch.run(stages)
    """

    def __init__(
        self,
        working_dir: str = "",
        tool_registry: Any = None,
        enable_autofix: bool = True,
        max_retries: int = 2,
        compute_router: Any = None,
        local_only: bool = False,
    ):
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        # tool_registry 三种形态:
        #   None  → 默认接全局 ToolRegistry (classmethod .get), 启动时已注册好工具
        #   dict[str, Callable] → 老风格, 直接调 tool_fn(action=, **params)
        #   ToolRegistry 类 → 新风格, .get(name) 返回 HuginnTool, 用 .call(args, ctx)
        if tool_registry is None:
            from huginn.tools.registry import ToolRegistry as _GlobalToolRegistry
            tool_registry = _GlobalToolRegistry
        self.tool_registry = tool_registry
        self.enable_autofix = enable_autofix
        self.max_retries = max_retries
        # ComputeRouter — auto-selects local vs HPC. Default instance so
        # route() always runs even when caller doesn't inject one.
        self.compute_router = compute_router or ComputeRouter()
        # M2 细粒度计算权限 + 预算. 默认 None (FeatureFlags.compute_policy 关则不
        # 实例化, 不改变现有执行); 首次 enforce 时按 flag 惰性构建.
        self.compute_policy: Any = None
        # M3 设备私有化: local_only=True 时路由强制远程回落本地, 数据不离开设备.
        self.local_only = local_only
        self._execution_history: list[WorkflowExecutionRecord] = []

    def register_tool(self, name: str, fn: Callable) -> None:
        """Register a tool function for execution.

        仅对 dict-style registry 有意义; ToolRegistry 类有自己的 .register.
        """
        if isinstance(self.tool_registry, dict):
            self.tool_registry[name] = fn
        else:
            # ToolRegistry 类模式有自己的 .register, register_tool 是 no-op,
            # 不需要 warning — 打 warning 会让调用方误以为出错了。debug 记录.
            logger.debug(
                "register_tool no-op: tool_registry is %s, not a dict",
                type(self.tool_registry).__name__,
            )

    def _audit_compute_route(
        self,
        stage_id: str,
        tool_name: str,
        action: str,
        params: dict[str, Any],
        target: str | None,
        reason: str | None,
    ) -> None:
        """M1 分流审计: 把一次计算路由决策写进 append-only audit 日志.

        记录 tool/action/目标/原因 + 参数哈希 (不落明文参, 防泄敏), 带 trace_id。
        best-effort: audit 层不可用/异常只 debug 记, 绝不阻断执行 (与 routing 本身
        "失败不阻塞" 同一哲学)。文案经 AuditLogger._redact_details 二次脱敏。
        """
        try:
            from huginn.security.audit import get_audit_logger

            get_audit_logger().log(
                event_type="compute_route",
                actor="orchestrator",
                action=f"{tool_name}:{action}",
                details={
                    "stage_id": stage_id,
                    "tool": tool_name,
                    "action": action,
                    "target": target,
                    "reason": reason,
                },
                input_data=str(params),
            )
        except Exception:
            logger.debug("compute_route audit write failed", exc_info=True)

    def _decide_call_privacy(self, tool_name: str, params: dict[str, Any]) -> Any:
        """P1(修正版): 判定本次调用承载的数据是否敏感 (需强制本地).

        内容 = tool 名 + 参数 (含路径/结构/凭证式字段)。不敏感 → 返回敏感=False,
        正常任务照常放行。判定失败/不可用 → None (fail-open 放行, 不阻塞)。
        """
        try:
            from huginn.execution.privacy_decision import decide_privacy

            # 用户自由度出口: 调用方可在 params 里带 privacy_override ("allow"/"local")
            # 显式覆盖本次私有化判定; 不在合法取值则落回默认规则.
            override = params.get("privacy_override")
            if override not in ("allow", "local"):
                override = None
            content = f"{tool_name} {params}"
            return decide_privacy(
                content, conservative_temporary=self.local_only, override=override
            )
        except Exception:
            logger.debug("privacy classify failed, fail-open allow", exc_info=True)
            return None

    def _audit_privacy_enforce(
        self,
        tool_name: str,
        params: dict[str, Any],
        decision: Any,
    ) -> None:
        """把一次"敏感数据 → 强制本地"的私有化覆盖落审计 (含 signal/reason)."""
        try:
            from huginn.security.audit import get_audit_logger

            get_audit_logger().log(
                event_type="privacy_enforce",
                actor="orchestrator",
                action=f"{tool_name}",
                details={
                    "tool": tool_name,
                    "sensitive": decision.sensitive,
                    "signal": decision.signal,
                    "reason": decision.reason,
                },
                input_data=str(params),
            )
        except Exception:
            logger.debug("privacy_enforce audit write failed", exc_info=True)

    def _enforce_compute_policy(
        self,
        tool_name: str,
        target: str | None,
        actor: str,
        params: dict[str, Any],
    ) -> Any:
        """M2 细粒度权限 + 预算: 按 (tool×target×actor×heavy) 判定一次调用.

        FeatureFlags.compute_policy 默认关 ⇒ 返回 None (不拦截, 不改变现有行为)。
        开启后返回 PolicyVerdict; 决策落 audit. 任何异常 fail-open (只 debug, 不阻断),
        保持与路由一致"故障不阻塞执行"哲学。
        """
        try:
            from huginn.feature_flags import FeatureFlags

            if not FeatureFlags.shared().is_enabled("compute_policy"):
                return None

            scaling = "generic"
            if self.compute_router is not None:
                scaling = self.compute_router.spec_for(tool_name).scaling

            from huginn.execution.compute_policy import ComputePolicy

            if self.compute_policy is None:
                self.compute_policy = ComputePolicy()
            verdict = self.compute_policy.enforce(
                tool_name, target, actor, scaling=scaling
            )
            try:
                from huginn.security.audit import get_audit_logger

                get_audit_logger().log(
                    event_type="compute_policy",
                    actor=actor,
                    action=f"{tool_name}:{target}",
                    details={
                        "tool": tool_name,
                        "target": target,
                        "allowed": verdict.allowed,
                        "requires_approval": verdict.requires_approval,
                        "reason": verdict.reason,
                    },
                )
            except Exception:
                logger.debug("compute_policy audit write failed", exc_info=True)
            return verdict
        except Exception:
            logger.debug("compute_policy enforce failed, fail-open", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    async def run(
        self,
        stages: list[dict[str, Any]],
        workflow_name: str = "unnamed_workflow",
    ) -> WorkflowExecutionRecord:
        """Execute a workflow defined as a list of stages.

        Args:
            stages: List of stage dicts with keys:
                id, tool, action, params, depends_on (optional)
            workflow_name: Identifier for this workflow run
        """
        record = WorkflowExecutionRecord(
            workflow_name=workflow_name,
            working_directory=str(self.working_dir),
        )

        # Build dependency graph
        graph = self._build_dependency_graph(stages)
        completed: set[str] = set()
        results_by_id: dict[str, StageResult] = {}

        # Execute stages in waves (topological order via BFS)
        pending = {s["id"] for s in stages}
        while pending:
            # Find stages whose dependencies are all satisfied
            ready = [
                sid
                for sid in pending
                if all(d in completed for d in graph.get(sid, []))
            ]
            if not ready:
                # Cycle detected or missing dependencies
                unresolved = [sid for sid in pending if sid not in ready]
                for sid in unresolved:
                    stage = self._find_stage(stages, sid)
                    result = StageResult(
                        stage_id=sid,
                        stage_name=stage.get("name", sid),
                        tool_name=stage.get("tool", "unknown"),
                        success=False,
                        error_message="Dependencies unresolved (possible cycle or missing stage)",
                    )
                    record.stage_results.append(result)
                    completed.add(sid)
                break

            # Execute ready stages concurrently
            stage_objs = [self._find_stage(stages, sid) for sid in ready]
            coros = [self._execute_stage(s, results_by_id) for s in stage_objs]
            stage_results = await asyncio.gather(*coros, return_exceptions=True)

            for sid, res in zip(ready, stage_results):
                pending.remove(sid)
                completed.add(sid)

                if isinstance(res, Exception):
                    result = StageResult(
                        stage_id=sid,
                        stage_name=self._find_stage(stages, sid).get("name", sid),
                        tool_name=self._find_stage(stages, sid).get("tool", "unknown"),
                        success=False,
                        error_message=str(res),
                    )
                else:
                    result = res

                results_by_id[sid] = result
                record.stage_results.append(result)

                # If autofix enabled and stage failed, try to fix and retry
                if (
                    self.enable_autofix
                    and not result.success
                    and result.retry_count < self.max_retries
                ):
                    fixed = await self._attempt_autofix(
                        result, self._find_stage(stages, sid)
                    )
                    if fixed:
                        # Re-execute with fix
                        retry_result = await self._execute_stage(
                            self._find_stage(stages, sid),
                            results_by_id,
                            retry_count=result.retry_count + 1,
                        )
                        retry_result.retry_count = result.retry_count + 1
                        retry_result.auto_fixed = True
                        retry_result.fix_applied = result.fix_applied
                        results_by_id[sid] = retry_result
                        # Replace in record
                        record.stage_results[-1] = retry_result

        record.finished_at = datetime.now().isoformat()
        record.overall_success = all(r.success for r in record.stage_results)
        self._execution_history.append(record)
        self._save_checkpoint(record)
        return record

    async def _execute_stage(
        self,
        stage: dict[str, Any],
        previous_results: dict[str, StageResult],
        retry_count: int = 0,
    ) -> StageResult:
        """Execute a single stage."""
        stage_id = stage["id"]
        stage_name = stage.get("name", stage_id)
        tool_name = stage.get("tool", "unknown")
        action = stage.get("action", "")
        params = dict(stage.get("params", {}))

        # Substitute dependency outputs into params
        params = self._resolve_param_refs(params, previous_results)

        # ComputeRouter: auto-select local vs HPC execution target.
        # Decision is stored on the StageResult, not injected into params —
        # most tools don't accept **kwargs and would choke on extra keys.
        _route_target: str | None = None
        _route_reason: str | None = None
        if self.compute_router is not None:
            try:
                _route = self.compute_router.route(tool_name, action, params)
                _route_target = _route.target
                _route_reason = _route.reason
                logger.debug(
                    "stage %s routed to %s: %s",
                    stage_id,
                    _route_target,
                    _route_reason,
                )
                # M1 分流审计: 路由决策落 audit (hash 链 + 脱敏), 供 per-tool×target
                # 报表与"该 hpc 却 local"偏差排查. best-effort, 失败不阻断执行.
                self._audit_compute_route(
                    stage_id, tool_name, action, params, _route_target, _route_reason
                )
            except Exception:
                logger.debug("route failed", exc_info=True)  # routing failure shouldn't block execution

        # P1(修正版) 私有化: 逐调用敏感判定覆盖远程目标.
        # 只有"这条数据的敏感度"决定是否强制本地, 全局开关不阻塞正常任务:
        #   敏感 → hpc/remote 改投 local (数据不出设备);
        #   不敏感 → 照常放行 (普通 DFT/MD 任务不受 local_only 妨碍).
        _privacy_pd = self._decide_call_privacy(tool_name, params)
        if _privacy_pd is not None:
            _privacy_override_signal = _privacy_pd.signal in (
                "user_override_allow",
                "user_override_local",
            )
            if _privacy_pd.sensitive and _route_target in ("hpc", "remote"):
                _route_target = "local"
                _route_reason = f"privacy override: {_privacy_pd.reason}"
                self._audit_privacy_enforce(tool_name, params, _privacy_pd)
            elif _privacy_override_signal:
                # 用户显式 override (允许外发 / 要求留本地): 即便未改目标也落审计,
                # 让"用户自由度"透明可追溯, 并给合规回放证据.
                self._audit_privacy_enforce(tool_name, params, _privacy_pd)

        # M2 细粒度权限 + 预算 (flag 门; 默认关 => _enforce 返回 None 不过滤).
        _verdict = self._enforce_compute_policy(
            tool_name, _route_target, "system", params
        )
        if _verdict is not None and not _verdict.allowed:
            return StageResult(
                stage_id=stage_id,
                stage_name=stage_name,
                tool_name=tool_name,
                success=False,
                error_message=f"blocked by compute policy: {_verdict.reason}",
                started_at=datetime.now().isoformat(),
                finished_at=datetime.now().isoformat(),
                execution_target=_route_target,
                route_reason=_route_reason,
            )
        if _verdict is not None and _verdict.requires_approval:
            # orchestrator 无交互审批回调 → fail-secure: 标记需审批即拦截, 留审计.
            return StageResult(
                stage_id=stage_id,
                stage_name=stage_name,
                tool_name=tool_name,
                success=False,
                error_message=f"compute policy requires approval: {_verdict.reason}",
                started_at=datetime.now().isoformat(),
                finished_at=datetime.now().isoformat(),
                execution_target=_route_target,
                route_reason=_route_reason,
            )

        started = datetime.now().isoformat()
        t0 = time.time()

        # Find and call the tool
        tool = self.tool_registry.get(tool_name)
        if tool is None:
            return StageResult(
                stage_id=stage_id,
                stage_name=stage_name,
                tool_name=tool_name,
                success=False,
                error_message=f"Tool '{tool_name}' not found in registry",
                started_at=started,
                finished_at=datetime.now().isoformat(),
                walltime_seconds=time.time() - t0,
                execution_target=_route_target,
                route_reason=_route_reason,
            )

        try:
            output = await self._invoke_tool(tool, action, params, workflow_name=stage_id)
            walltime = time.time() - t0
            return StageResult(
                stage_id=stage_id,
                stage_name=stage_name,
                tool_name=tool_name,
                success=True,
                output_data=self._serialize_output(output),
                started_at=started,
                finished_at=datetime.now().isoformat(),
                walltime_seconds=walltime,
                retry_count=retry_count,
                execution_target=_route_target,
                route_reason=_route_reason,
            )
        except Exception as e:
            walltime = time.time() - t0
            return StageResult(
                stage_id=stage_id,
                stage_name=stage_name,
                tool_name=tool_name,
                success=False,
                error_message=str(e),
                started_at=started,
                finished_at=datetime.now().isoformat(),
                walltime_seconds=walltime,
                retry_count=retry_count,
                execution_target=_route_target,
                route_reason=_route_reason,
            )

    async def _invoke_tool(
        self,
        tool: Any,
        action: str,
        params: dict[str, Any],
        workflow_name: str = "",
    ) -> Any:
        """统一工具调用入口, 屏蔽 dict-callable 和 HuginnTool 两种形态.

        - HuginnTool (有 .call 和 .input_schema): 构造 input_schema 实例,
          带 action 时一并塞进去, 走 await tool.call(args, ctx).
        - dict-callable: 老路径 tool_fn(action=, **params).
        """
        # HuginnTool 分支: 有 .call 方法 (async) + .input_schema
        if hasattr(tool, "call") and hasattr(tool, "input_schema"):
            from huginn.core_types import ToolContext

            schema = tool.input_schema
            kwargs = dict(params)
            if action and schema is not None:
                # 多数工具的 input schema 有 action 字段, 没有的话 try-except 兜底
                try:
                    args_obj = schema(action=action, **kwargs)
                except Exception:
                    args_obj = schema(**kwargs)
            elif schema is not None:
                args_obj = schema(**kwargs)
            else:
                args_obj = kwargs
            ctx = ToolContext(
                session_id=workflow_name or "orchestrator",
                workspace=str(self.working_dir),
            )
            result = await tool.call(args_obj, ctx)
            # ToolResult → 取 data; 失败抛上去让外层标 failed
            if hasattr(result, "success") and not result.success:
                raise RuntimeError(result.error or "tool call failed")
            return getattr(result, "data", result)

        # dict-callable 分支 (老行为)
        if asyncio.iscoroutinefunction(tool):
            return await tool(action=action, **params)
        return tool(action=action, **params)

    # ------------------------------------------------------------------
    # Auto-fix integration
    # ------------------------------------------------------------------

    async def _attempt_autofix(
        self,
        failed_result: StageResult,
        stage: dict[str, Any],
    ) -> bool:
        """Attempt to automatically fix a failed stage."""
        # Import autofix logic
        try:
            from huginn.execution.autofix import AutoFixLoop

            fixer = AutoFixLoop()
            fixed_params = fixer.apply_fix(
                tool_name=failed_result.tool_name,
                error=failed_result.error_message or "",
                current_params=stage.get("params", {}),
            )
            if fixed_params:
                stage["params"] = fixed_params
                failed_result.fix_applied = str(fixed_params)
                return True
        except Exception:
            logger.debug("attempt autofix failed", exc_info=True)
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_dependency_graph(
        self, stages: list[dict[str, Any]]
    ) -> dict[str, list[str]]:
        graph = {}
        for s in stages:
            deps = s.get("depends_on", [])
            graph[s["id"]] = deps if isinstance(deps, list) else [deps] if deps else []
        return graph

    def _find_stage(
        self, stages: list[dict[str, Any]], stage_id: str
    ) -> dict[str, Any]:
        for s in stages:
            if s.get("id") == stage_id:
                return s
        return {"id": stage_id, "tool": "unknown", "params": {}}

    def _resolve_param_refs(
        self,
        params: dict[str, Any],
        previous_results: dict[str, StageResult],
    ) -> dict[str, Any]:
        """Replace ${stage_id.output_key} references with actual values."""
        resolved = {}
        for key, val in params.items():
            if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
                # Format: ${stage_id.output_key}
                inner = val[2:-1]
                if "." in inner:
                    sid, out_key = inner.split(".", 1)
                    if sid in previous_results:
                        resolved[key] = previous_results[sid].output_data.get(
                            out_key, val
                        )
                    else:
                        resolved[key] = val
                else:
                    resolved[key] = previous_results.get(
                        inner, StageResult(inner, "", "", False)
                    ).output_data
            else:
                resolved[key] = val
        return resolved

    def _serialize_output(self, output: Any) -> dict[str, Any]:
        """Convert tool output to a serializable dict."""
        if isinstance(output, dict):
            return output
        if hasattr(output, "model_dump"):
            return output.model_dump()
        if hasattr(output, "__dict__"):
            return output.__dict__
        return {"raw": str(output)}

    def _save_checkpoint(self, record: WorkflowExecutionRecord) -> None:
        """Save execution record to disk for resumability."""
        checkpoint_dir = self.working_dir / ".checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = checkpoint_dir / f"{record.workflow_name}_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(record), f, ensure_ascii=False, indent=2)

    def list_checkpoints(self) -> list[Path]:
        """List available checkpoint files."""
        checkpoint_dir = self.working_dir / ".checkpoints"
        if not checkpoint_dir.exists():
            return []
        return sorted(
            checkpoint_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )

    def load_checkpoint(self, path: Path) -> WorkflowExecutionRecord | None:
        """Load a workflow execution from checkpoint."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return WorkflowExecutionRecord(**data)
