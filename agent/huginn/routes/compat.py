"""math-anything compatibility stubs."""

from __future__ import annotations

import os
import tempfile
from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["compat"])


@router.get("/firewall/status")
async def firewall_status() -> dict[str, Any]:
    return {"enabled": False}


@router.post("/sandbox/execute")
async def sandbox_execute(params: dict[str, Any]) -> dict[str, Any]:
    """Execute Python code in a sandbox."""
    code = params.get("code", "")
    timeout = params.get("timeout_seconds", 10)

    # Pre-validate code against restricted execution policy
    try:
        from huginn.security import RestrictedPythonError, validate_code

        validate_code(code)
    except RestrictedPythonError as e:
        return {"success": False, "error": f"Policy violation: {e}"}

    try:
        import subprocess

        from huginn.security import SandboxConfig, SandboxExecutor

        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            tmp_path = f.name

        sandbox = SandboxExecutor(
            SandboxConfig(
                allowed_executables={"python", "python3"},
                default_timeout=min(float(timeout), 300.0),
                max_timeout=300.0,
                max_output_bytes=10 * 1024 * 1024,
            )
        )
        sb_result = sandbox.run(
            ["python", tmp_path],
            timeout=min(float(timeout), 300.0),
        )
        result = sb_result

        os.unlink(tmp_path)

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Execution timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/analyze/symmetry")
async def analyze_symmetry(params: dict[str, Any]) -> dict[str, Any]:
    return {"error": "Not yet implemented. Install spglib for symmetry analysis."}


@router.post("/analyze/spectral")
async def analyze_spectral(params: dict[str, Any]) -> dict[str, Any]:
    return {"error": "Not yet implemented"}


@router.post("/analyze/dynamics")
async def analyze_dynamics(params: dict[str, Any]) -> dict[str, Any]:
    return {"error": "Not yet implemented"}


@router.post("/analyze/tda")
async def analyze_tda(params: dict[str, Any]) -> dict[str, Any]:
    return {"error": "Not yet implemented"}


@router.post("/analyze/sindy")
async def analyze_sindy(params: dict[str, Any]) -> dict[str, Any]:
    return {"error": "Not yet implemented"}


@router.post("/viz/dos")
async def viz_dos(params: dict[str, Any]) -> dict[str, Any]:
    return {"fallback": True, "html": "<div>DOS visualization stub</div>"}


@router.post("/viz/phase")
async def viz_phase(params: dict[str, Any]) -> dict[str, Any]:
    return {"fallback": True, "html": "<div>Phase portrait stub</div>"}


@router.post("/viz/persistence")
async def viz_persistence(params: dict[str, Any]) -> dict[str, Any]:
    return {"fallback": True, "html": "<div>Persistence diagram stub</div>"}


@router.post("/viz/sindy")
async def viz_sindy(params: dict[str, Any]) -> dict[str, Any]:
    return {"fallback": True, "html": "<div>SINDy visualization stub</div>"}
