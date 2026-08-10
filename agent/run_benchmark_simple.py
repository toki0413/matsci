#!/usr/bin/env python3
"""简易性能基准测试 (不依赖 pytest-benchmark, 避免内存压力).

测量关键性能指标:
- 工具调用吞吐 (serial/parallel)
- 沙箱执行延迟
- 审计日志写入吞吐
- API key 比较延迟
- 大结构内存占用

输出 JSON 报告: benchmark_simple.json
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# 确保能 import huginn
sys.path.insert(0, str(Path(__file__).parent))

RESULTS: dict = {"tests": [], "env": {"python": sys.version, "pid": os.getpid()}}


def record(name: str, elapsed: float, ops: int, unit: str = "ops/s", extra: dict | None = None):
    rate = ops / elapsed if elapsed > 0 else float("inf")
    result = {
        "name": name,
        "elapsed_s": round(elapsed, 6),
        "ops": ops,
        "rate": round(rate, 2),
        "unit": unit,
    }
    if extra:
        result["extra"] = extra
    RESULTS["tests"].append(result)
    print(f"  {name}: {elapsed:.4f}s ({rate:.0f} {unit})")


def bench_tool_serial():
    """串行工具调用吞吐."""
    from huginn.tools.base import HuginnTool
    from huginn.types import ToolContext, ToolResult

    class EchoTool(HuginnTool):
        name = "echo"
        description = "Echo"
        input_schema = dict

        async def call(self, args, context):
            await asyncio.sleep(0.001)
            return ToolResult(data={"echo": args.get("msg", "")}, success=True)

    tool = EchoTool()
    ctx = ToolContext(session_id="bench", workspace=".")
    N = 50

    async def _run():
        for i in range(N):
            await tool.call({"msg": f"x{i}"}, ctx)

    start = time.monotonic()
    asyncio.run(_run())
    elapsed = time.monotonic() - start
    record("tool_serial_50", elapsed, N)


def bench_tool_parallel():
    """并行工具调用吞吐."""
    from huginn.tools.base import HuginnTool
    from huginn.types import ToolContext, ToolResult

    class EchoTool(HuginnTool):
        name = "echo"
        description = "Echo"
        input_schema = dict

        async def call(self, args, context):
            await asyncio.sleep(0.001)
            return ToolResult(data={"echo": args.get("msg", "")}, success=True)

    tool = EchoTool()
    ctx = ToolContext(session_id="bench", workspace=".")
    N = 50

    async def _run():
        await asyncio.gather(*[tool.call({"msg": f"x{i}"}, ctx) for i in range(N)])

    start = time.monotonic()
    asyncio.run(_run())
    elapsed = time.monotonic() - start
    record("tool_parallel_50", elapsed, N)


def bench_sandbox_exec():
    """沙箱执行延迟."""
    from huginn.security import SandboxConfig, SandboxExecutor

    cfg = SandboxConfig(allowed_executables={"python", "python3"})
    sandbox = SandboxExecutor(cfg)
    N = 10
    start = time.monotonic()
    for _ in range(N):
        try:
            result = sandbox.run([sys.executable, "-c", "print(1)"])
            assert result.success, result.error
        except OSError as e:
            if e.errno == 12:  # ENOMEM
                record("sandbox_exec_10", 0, 0, extra={"skipped": "OOM"})
                return
            raise
    elapsed = time.monotonic() - start
    record("sandbox_exec_10", elapsed, N)


def bench_audit_log(tmp_path: Path):
    """审计日志写入吞吐."""
    from huginn.security import AuditLogger

    log = AuditLogger(str(tmp_path / "audit.jsonl"))
    N = 100
    start = time.monotonic()
    for i in range(N):
        log.log("tool_call", "agent", f"action_{i}")
    elapsed = time.monotonic() - start
    record("audit_log_100", elapsed, N)


def bench_api_key_compare():
    """API key 恒定时间比较延迟."""
    from huginn.security.auth import secrets_match

    a = "a" * 64
    b = "a" * 64
    N = 10000
    start = time.monotonic()
    for _ in range(N):
        secrets_match(a, b)
    elapsed = time.monotonic() - start
    record("api_key_match_10k", elapsed, N, unit="compares/s")


def bench_workflow_init():
    """工作流引擎初始化延迟."""
    from huginn.tools.registry import ToolRegistry
    from huginn.workflows.engine import WorkflowEngine

    N = 20
    start = time.monotonic()
    for _ in range(N):
        WorkflowEngine(ToolRegistry)
    elapsed = time.monotonic() - start
    record("workflow_init_20", elapsed, N)


def bench_large_json():
    """大 JSON 结构处理."""
    import json as json_mod

    large_dict = {
        "atoms": [
            {"id": i, "x": i * 0.1, "y": i * 0.2, "z": i * 0.3}
            for i in range(10_000)
        ]
    }
    N = 5
    start = time.monotonic()
    for _ in range(N):
        raw = json_mod.dumps(large_dict)
        json_mod.loads(raw)
    elapsed = time.monotonic() - start
    record("large_json_10k_5", elapsed, N, unit="serializations/s")


def main():
    import tempfile

    tmp_path = Path(tempfile.mkdtemp(prefix="huginn_bench_"))
    print(f">>> 性能基准测试 (pid={os.getpid()})")
    print(f">>> Python: {sys.version.split()[0]}")
    print(f">>> 临时目录: {tmp_path}")
    print()

    tests = [
        ("工具调用-串行", bench_tool_serial),
        ("工具调用-并行", bench_tool_parallel),
        ("沙箱执行", bench_sandbox_exec),
        ("审计日志", lambda: bench_audit_log(tmp_path)),
        ("API key 比较", bench_api_key_compare),
        ("工作流初始化", bench_workflow_init),
        ("大JSON处理", bench_large_json),
    ]

    for label, fn in tests:
        print(f"[{label}]")
        try:
            fn()
        except Exception as e:
            print(f"  FAILED: {e}")
            RESULTS["tests"].append({"name": label, "error": str(e)})
        print()

    # 保存报告
    report_path = Path("benchmark_simple.json")
    with open(report_path, "w") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    print(f">>> 报告已保存: {report_path}")

    # 汇总
    print("\n=== 汇总 ===")
    for t in RESULTS["tests"]:
        if "error" in t:
            print(f"  FAIL  {t['name']}: {t['error']}")
        else:
            print(f"  OK    {t['name']}: {t['rate']} {t['unit']}")


if __name__ == "__main__":
    main()
