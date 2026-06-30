"""系统资源诊断工具 —— agent 主动查系统健康状态的入口。

和 diagnose_tool (计算化学错误知识库) 不同，这个工具看的是运行时系统资源：
CPU/内存/磁盘/swap 的实时负载，谁在吃资源，以及该怎么处理。

背后的逻辑全在 huginn.diagnostics.system_health.SystemHealthMonitor，
工具本身只做薄封装 + 输入校验。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult, ValidationResult


class SystemDiagnosticInput(BaseModel):
    action: Literal[
        "snapshot",
        "diagnose",
        "recent_anomalies",
        "top_processes",
    ] = Field(
        ...,
        description="snapshot: 当前资源快照; diagnose: 诊断异常+根因; "
        "recent_anomalies: 最近记录的异常事件; top_processes: 吃资源最多的进程.",
    )
    top_n: int = Field(
        default=10,
        description="top_processes action 返回的进程数 (最多 50).",
    )
    by: Literal["cpu", "memory"] = Field(
        default="cpu",
        description="top_processes 按什么排序.",
    )


class SystemDiagnosticTool(HuginnTool):
    """查看系统资源健康状态，诊断 CPU/内存/磁盘负载异常。"""

    name = "system_diagnostic_tool"
    category = "core"
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset({ResearchPhase.VALIDATION}),
    )
    description = (
        "Monitor and diagnose system resource anomalies (CPU, memory, disk, swap). "
        "Returns real-time metrics, root-cause analysis with top consumers, "
        "and actionable recommendations. Read-only — auto-remediation is "
        "controlled by the system_health_auto_fix feature flag."
    )
    input_schema = SystemDiagnosticInput

    def is_read_only(self, args: SystemDiagnosticInput) -> bool:
        return True

    async def validate_input(
        self, args: SystemDiagnosticInput, context: ToolContext | None = None
    ) -> ValidationResult:
        if args.action == "top_processes" and (args.top_n < 1 or args.top_n > 50):
            return ValidationResult(
                result=False,
                message="top_n must be between 1 and 50.",
            )
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = SystemDiagnosticInput(**args)
        from huginn.diagnostics.system_health import SystemHealthMonitor

        monitor = SystemHealthMonitor.shared()

        if input_data.action == "snapshot":
            metrics = monitor.snapshot()
            return ToolResult(data=metrics.to_dict(), success=True)

        if input_data.action == "diagnose":
            events = monitor.diagnose()
            if not events:
                return ToolResult(
                    data={
                        "status": "healthy",
                        "message": "No resource anomalies detected.",
                        "metrics": monitor.snapshot().to_dict(),
                    },
                    success=True,
                )
            return ToolResult(
                data={
                    "status": "anomaly",
                    "anomaly_count": len(events),
                    "anomalies": [ev.to_dict() for ev in events],
                    "metrics": monitor.snapshot().to_dict(),
                },
                success=True,
            )

        if input_data.action == "recent_anomalies":
            events = monitor.recent_anomalies(limit=20)
            return ToolResult(
                data={
                    "count": len(events),
                    "anomalies": [ev.to_dict() for ev in events],
                },
                success=True,
            )

        if input_data.action == "top_processes":
            procs = monitor.top_processes(n=input_data.top_n, by=input_data.by)
            return ToolResult(
                data={
                    "by": input_data.by,
                    "count": len(procs),
                    "processes": procs,
                },
                success=True,
            )

        return ToolResult(
            data=None,
            success=False,
            error=f"Unknown action: {input_data.action}",
        )
