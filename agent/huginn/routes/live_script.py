"""Live script execution routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/live", tags=["live-script"])


class ScriptRequest(BaseModel):
    script: str = Field(..., description="Python script to execute")
    variables: dict[str, Any] = Field(
        default_factory=dict, description="Variables to inject"
    )
    timeout: float = Field(default=30.0, ge=1.0, le=120.0)


class ScriptResponse(BaseModel):
    success: bool
    stdout: str = ""
    stderr: str = ""
    result_value: Any = None
    execution_time_ms: float = 0
    error: str | None = None


@router.post("/execute", response_model=ScriptResponse)
async def execute_script(request: ScriptRequest) -> ScriptResponse:
    """Execute a Python script in the sandboxed environment."""
    from huginn.security.script_runner import ScriptRunner

    runner = ScriptRunner(timeout=request.timeout)
    result = await runner.execute(request.script, request.variables)
    return ScriptResponse(
        success=result.success,
        stdout=result.stdout,
        stderr=result.stderr,
        result_value=result.result_value,
        execution_time_ms=result.execution_time_ms,
        error=result.error,
    )


@router.get("/capabilities")
async def get_capabilities() -> dict[str, Any]:
    """Return sandbox capabilities (allowed builtins, blocked imports)."""
    from huginn.security.script_runner import _BLOCKED_IMPORTS, _SAFE_BUILTINS

    return {
        "safe_builtins": sorted(_SAFE_BUILTINS),
        "blocked_imports": sorted(_BLOCKED_IMPORTS),
        "max_timeout": 120,
        "default_timeout": 30,
    }
