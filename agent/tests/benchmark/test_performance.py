"""Performance benchmarks for Huginn — concurrency, memory, timeout.

Run with: pytest tests/benchmark/ --benchmark-only
Requires: pytest-benchmark, memory-profiler (listed in dev dependencies)
"""

from __future__ import annotations

import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from huginn.agent import HuginnAgent
from huginn.self_improvement import BenchmarkCase, BenchmarkSuite
from huginn.memory.manager import MemoryManager
from huginn.security import SandboxConfig, SandboxExecutor, SandboxResult
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult
from huginn.workflows.engine import WorkflowEngine

CTX = ToolContext(session_id="benchmark", workspace=".")


# ── Fixture: lightweight agent ─────────────────────────────────────────────

@pytest.fixture
def light_agent(tmp_path: Path):
    from huginn.memory.longterm import LongTermMemory
    longterm = LongTermMemory(db_path=tmp_path / "memory.db")
    return HuginnAgent(memory_manager=MemoryManager(longterm=longterm))


# ── 1. Tool call throughput (serial vs parallel) ───────────────────────────

class _EchoTool(HuginnTool):
    name = "echo"
    description = "Echo input"

    async def call(self, args, context):
        await asyncio.sleep(0.001)  # 1 ms simulated work
        return ToolResult(data={"echo": args.get("msg", "")}, success=True)


class TestToolConcurrency:
    def test_serial_tool_calls(self, benchmark):
        tool = _EchoTool()
        def _serial():
            async def _inner():
                for i in range(50):
                    await tool.call({"msg": f"x{i}"}, CTX)
            asyncio.run(_inner())
        benchmark(_serial)

    def test_parallel_tool_calls(self, benchmark):
        tool = _EchoTool()
        def _parallel():
            async def _inner():
                await asyncio.gather(*[
                    tool.call({"msg": f"x{i}"}, CTX) for i in range(50)
                ])
            asyncio.run(_inner())
        benchmark(_parallel)


# ── 2. Sandbox timeout & cancellation ──────────────────────────────────────

class TestSandboxTimeout:
    def test_sandbox_fast_command(self, benchmark):
        cfg = SandboxConfig(allowed_executables={"python", "python3"})
        sandbox = SandboxExecutor(cfg)
        benchmark(sandbox.run, [sys.executable, "-c", "print(1)"])

    def test_sandbox_timeout_result(self):
        cfg = SandboxConfig(allowed_executables={"python", "python3"}, max_timeout=1.0)
        sandbox = SandboxExecutor(cfg)
        start = time.monotonic()
        result = sandbox.run(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout=0.5,
        )
        elapsed = time.monotonic() - start
        assert result.success is False
        assert result.returncode == -1
        assert elapsed < 1.5  # should return near timeout, not sleep 5 s

    def test_sandbox_output_truncation(self):
        cfg = SandboxConfig(
            allowed_executables={"python", "python3"},
            max_output_bytes=100,
        )
        sandbox = SandboxExecutor(cfg)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "x" * 500
            mock_run.return_value.stderr = ""
            result = sandbox.run([sys.executable, "-c", "print(1)"])
        assert "truncated" in result.stdout


# ── 3. Memory pressure (large data transfer) ────────────────────────────────

class TestMemoryPressure:
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="memory_profiler CLI differs on Windows",
    )
    def test_memory_with_large_structure(self, benchmark):
        """Benchmark creating / passing a large JSON structure through a tool."""
        import json

        large_dict = {
            "atoms": [
                {"id": i, "x": i * 0.1, "y": i * 0.2, "z": i * 0.3}
                for i in range(50_000)
            ]
        }
        raw = json.dumps(large_dict)

        def _tool_with_large_data():
            async def _inner():
                tool = _EchoTool()
                return await tool.call({"msg": raw}, CTX)
            asyncio.run(_inner())

        benchmark(_tool_with_large_data)

    def test_memory_with_benchmark_suite(self, benchmark):
        """Benchmark suite evaluation with 100 cases."""
        suite = BenchmarkSuite()
        for i in range(100):
            suite.add(BenchmarkCase(
                task=f"task_{i}",
                expected_keywords=[f"kw{i}"],
            ))

        class _FastAgent:
            async def chat(self, task, thread_id="default"):
                yield {"messages": [type("Msg", (), {"content": f"kw{task.split('_')[1]}"})()]}

        def _run_suite():
            asyncio.run(suite.run(_FastAgent()))

        benchmark(_run_suite)


# ── 4. Workflow engine throughput ───────────────────────────────────────────

class TestWorkflowThroughput:
    def test_simple_workflow(self, benchmark):
        from huginn.workflows.engine import WorkflowEngine
        from huginn.tools.registry import ToolRegistry
        engine = WorkflowEngine(ToolRegistry)
        # execute is async, but we can benchmark the engine init at least
        benchmark(lambda: WorkflowEngine(ToolRegistry))


# ── 5. Auth / API overhead ─────────────────────────────────────────────────

class TestAuthOverhead:
    def test_api_key_comparison(self, benchmark):
        from huginn.security.auth import secrets_match
        benchmark(secrets_match, "a" * 64, "a" * 64)

    def test_api_key_comparison_different(self, benchmark):
        from huginn.security.auth import secrets_match
        benchmark(secrets_match, "a" * 64, "b" * 64)


# ── 6. Audit log throughput ────────────────────────────────────────────────

class TestAuditThroughput:
    def test_audit_log_append(self, benchmark, tmp_path: Path):
        from huginn.security import AuditLogger
        log = AuditLogger(str(tmp_path / "audit.jsonl"))

        def _append():
            for i in range(100):
                log.log("tool_call", "agent", f"action_{i}")

        benchmark(_append)

    def test_audit_chain_verify(self, benchmark, tmp_path: Path):
        from huginn.security import AuditLogger
        log = AuditLogger(str(tmp_path / "audit.jsonl"))
        for i in range(500):
            log.log("tool_call", "agent", f"action_{i}")
        benchmark(log.verify_chain)


# ── 7. FastAPI endpoint latency (if server tests exist) ───────────────────

class TestServerLatency:
    @pytest.mark.skipif(True, reason="Requires running server; run manually")
    def test_health_endpoint_latency(self, benchmark):
        """Placeholder: requires actual HTTP client against running server."""
        pass


# ── 8. LLM / agent overhead (mocked) ───────────────────────────────────────

class TestAgentOverhead:
    def test_agent_tool_registry_lookup(self, benchmark):
        from huginn.tools.registry import ToolRegistry
        benchmark(ToolRegistry.list_tools)

    def test_agent_memory_retrieval(self, benchmark, light_agent):
        benchmark(light_agent.memory.recall, "silicon band gap")
