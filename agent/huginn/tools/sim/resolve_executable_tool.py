"""让用户/agent 注册仿真软件安装路径的工具.

当 sim tool 返回 needs_resolution 时, agent 调这个工具把用户给的路径存下来,
后续调用自动命中缓存.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult


class ResolveExecutableInput(BaseModel):
    action: Literal["resolve", "register", "list", "install_hint"] = Field(
        default="resolve"
    )
    tool_name: str = Field(
        default="",
        description="仿真工具名称: vasp / lammps / qe / cp2k / gaussian / orca / ... (list action 可省略)",
    )
    path: str | None = Field(
        default=None,
        description="用户提供的本地安装路径 (文件或目录). action=register 时必填.",
    )


class ResolveExecutableTool(HuginnTool):
    """查找/注册仿真软件的可执行文件路径."""

    name = "resolve_executable_tool"
    category = "sim"
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset({ResearchPhase.PLANNING}),
    )
    description = (
        "查找仿真软件 (VASP/LAMMPS/QE/...) 的可执行文件路径, "
        "或注册用户提供的路径. 当仿真工具找不到可执行文件时, "
        "用此工具传入用户提供的路径."
    )
    input_schema = ResolveExecutableInput

    def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = ResolveExecutableInput(**args)

        from huginn.tools.sim.executable_resolver import get_resolver, ResolutionRequest

        resolver = get_resolver()

        if input_data.action == "resolve":
            result = resolver.resolve(input_data.tool_name)
            if isinstance(result, str):
                return ToolResult(
                    data={
                        "found": True,
                        "path": result,
                        "tool_name": input_data.tool_name,
                    },
                    success=True,
                )
            return ToolResult(
                data={
                    "found": False,
                    "tool_name": input_data.tool_name,
                    "resolution_request": result.to_dict(),
                },
                success=True,
            )

        if input_data.action == "register":
            if not input_data.path:
                return ToolResult(
                    data=None,
                    success=False,
                    error="register action requires a 'path' field.",
                )
            ok = resolver.register_path(input_data.tool_name, input_data.path)
            return ToolResult(
                data={
                    "registered": ok,
                    "tool_name": input_data.tool_name,
                    "path": input_data.path,
                },
                success=ok,
                error=None if ok else f"Path not found: {input_data.path}",
            )

        if input_data.action == "list":
            from huginn.tools.sim.executable_resolver import _REGISTRY
            tools = {}
            for name, spec in _REGISTRY.items():
                r = resolver.resolve(name)
                tools[name] = {
                    "found": isinstance(r, str),
                    "path": r if isinstance(r, str) else None,
                    "env_vars": list(spec.env_vars),
                    "conda_package": spec.conda_package,
                    "license_required": spec.license_required,
                }
            return ToolResult(data={"tools": tools}, success=True)

        if input_data.action == "install_hint":
            from huginn.tools.sim.executable_resolver import _REGISTRY
            spec = _REGISTRY.get(input_data.tool_name)
            if spec is None:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Unknown tool: {input_data.tool_name}",
                )
            return ToolResult(
                data={
                    "tool_name": input_data.tool_name,
                    "install_hint": spec.install_hint,
                    "conda_package": spec.conda_package,
                    "license_required": spec.license_required,
                    "install_command": resolver.get_install_command(input_data.tool_name),
                },
                success=True,
            )

        return ToolResult(
            data=None, success=False, error=f"Unknown action: {input_data.action}"
        )
