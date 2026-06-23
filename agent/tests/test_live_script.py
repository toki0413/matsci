"""Tests for live script execution (ScriptRunner + routes)."""

from __future__ import annotations

import asyncio

import pytest

from huginn.security.script_runner import (
    _BLOCKED_IMPORTS,
    _SAFE_BUILTINS,
    ScriptResult,
    ScriptRunner,
)


# ---------------------------------------------------------------------------
# ScriptRunner — direct tests
# ---------------------------------------------------------------------------


class TestScriptRunnerBasic:
    """Basic script execution tests."""

    @pytest.fixture()
    def runner(self) -> ScriptRunner:
        return ScriptRunner(timeout=10.0)

    @pytest.mark.asyncio
    async def test_simple_print(self, runner: ScriptRunner) -> None:
        result = await runner.execute("print('hello world')")
        assert result.success is True
        assert "hello world" in result.stdout
        assert result.error is None

    @pytest.mark.asyncio
    async def test_variables_injection(self, runner: ScriptRunner) -> None:
        script = "result = x + y"
        result = await runner.execute(script, variables={"x": 10, "y": 20})
        assert result.success is True
        assert result.result_value == "30"

    @pytest.mark.asyncio
    async def test_result_variable_capture(self, runner: ScriptRunner) -> None:
        script = "result = [1, 2, 3]"
        result = await runner.execute(script)
        assert result.success is True
        assert result.result_value == "[1, 2, 3]"

    @pytest.mark.asyncio
    async def test_no_result_variable(self, runner: ScriptRunner) -> None:
        script = "x = 42"
        result = await runner.execute(script)
        assert result.success is True
        assert result.result_value is None

    @pytest.mark.asyncio
    async def test_empty_script(self, runner: ScriptRunner) -> None:
        result = await runner.execute("")
        assert result.success is True
        assert result.stdout == ""

    @pytest.mark.asyncio
    async def test_multiline_script(self, runner: ScriptRunner) -> None:
        script = """
total = 0
for i in range(5):
    total += i
print(f"total={total}")
result = total
"""
        result = await runner.execute(script)
        assert result.success is True
        assert "total=10" in result.stdout
        assert result.result_value == "10"

    @pytest.mark.asyncio
    async def test_execution_time_recorded(self, runner: ScriptRunner) -> None:
        result = await runner.execute("x = 1")
        assert result.execution_time_ms >= 0


class TestScriptRunnerSafety:
    """Security / sandbox tests."""

    @pytest.fixture()
    def runner(self) -> ScriptRunner:
        return ScriptRunner(timeout=10.0)

    @pytest.mark.asyncio
    async def test_blocked_import_os(self, runner: ScriptRunner) -> None:
        result = await runner.execute("import os")
        assert result.success is False
        assert "not allowed" in result.error

    @pytest.mark.asyncio
    async def test_blocked_import_subprocess(self, runner: ScriptRunner) -> None:
        result = await runner.execute("import subprocess")
        assert result.success is False
        assert "not allowed" in result.error

    @pytest.mark.asyncio
    async def test_blocked_import_sys(self, runner: ScriptRunner) -> None:
        result = await runner.execute("import sys")
        assert result.success is False
        assert "not allowed" in result.error

    @pytest.mark.asyncio
    async def test_safe_math_available(self, runner: ScriptRunner) -> None:
        result = await runner.execute("result = sqrt(16)")
        assert result.success is True
        assert result.result_value == "4.0"

    @pytest.mark.asyncio
    async def test_safe_math_pi(self, runner: ScriptRunner) -> None:
        result = await runner.execute("result = pi")
        assert result.success is True
        assert "3.14" in result.result_value

    @pytest.mark.asyncio
    async def test_syntax_error(self, runner: ScriptRunner) -> None:
        result = await runner.execute("def foo(")
        assert result.success is False
        assert "Syntax error" in result.error

    @pytest.mark.asyncio
    async def test_runtime_error(self, runner: ScriptRunner) -> None:
        result = await runner.execute("1 / 0")
        assert result.success is False
        assert "ZeroDivisionError" in result.error


# ---------------------------------------------------------------------------
# ScriptResult dataclass
# ---------------------------------------------------------------------------


class TestScriptResult:
    def test_defaults(self) -> None:
        r = ScriptResult(success=True)
        assert r.stdout == ""
        assert r.stderr == ""
        assert r.result_value is None
        assert r.execution_time_ms == 0
        assert r.error is None


# ---------------------------------------------------------------------------
# Route endpoints
# ---------------------------------------------------------------------------


class TestLiveScriptRoutes:
    """Test the /live/* FastAPI endpoints."""

    @pytest.fixture()
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from huginn.routes.live_script import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_execute_simple(self, client) -> None:
        resp = client.post(
            "/live/execute",
            json={"script": "print('hi')", "variables": {}, "timeout": 10},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "hi" in data["stdout"]

    def test_execute_with_variables(self, client) -> None:
        resp = client.post(
            "/live/execute",
            json={
                "script": "result = a * b",
                "variables": {"a": 6, "b": 7},
                "timeout": 10,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["result_value"] == "42"

    def test_execute_blocked_import(self, client) -> None:
        resp = client.post(
            "/live/execute",
            json={"script": "import os", "timeout": 10},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "not allowed" in data["error"]

    def test_capabilities(self, client) -> None:
        resp = client.get("/live/capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert "safe_builtins" in data
        assert "blocked_imports" in data
        assert data["max_timeout"] == 120
        assert data["default_timeout"] == 30
        assert "print" in data["safe_builtins"]
        assert "os" in data["blocked_imports"]

    def test_execute_timeout_too_large(self, client) -> None:
        resp = client.post(
            "/live/execute",
            json={"script": "x=1", "timeout": 999},
        )
        assert resp.status_code == 422  # validation error from pydantic


# ---------------------------------------------------------------------------
# Timeout — keep LAST so zombie thread doesn't block other tests
# ---------------------------------------------------------------------------


class TestScriptRunnerTimeout:
    """Timeout handling — placed last to avoid thread-pool interference."""

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        runner = ScriptRunner(timeout=0.5)
        # Use time.sleep (releases GIL) with short duration
        result = await runner.execute("import time\ntime.sleep(2)")
        assert result.success is False
        assert "timed out" in result.error
