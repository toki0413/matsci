"""Agent 一轮回复后自动跑 pytest 的 Stop 钩子.

只在 workspace 里有 pytest 配置文件时触发, 跑 pytest --tb=short -q,
60s 超时. 结果摘要塞进 ctx.metadata['test_summary'], 不阻断 agent.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from huginn.hooks import HookContext

logger = logging.getLogger(__name__)

# 有任一文件就认为 workspace 启用了 pytest
_PYTEST_MARKERS = ("pytest.ini", "pyproject.toml", "conftest.py", "setup.cfg")
_TEST_TIMEOUT = 60  # 秒


async def test_stop_hook(ctx: HookContext) -> HookContext | None:
    """STOP 事件触发后跑 pytest, 把摘要写进 metadata."""
    workspace = ctx.metadata.get("workspace") or "."
    ws = Path(workspace)
    if not _has_pytest_config(ws):
        return None

    # pytest 没装就静默跳过
    try:
        import pytest  # noqa: F401
    except ImportError:
        return None

    try:
        summary = await _run_pytest(ws)
    except Exception:
        logger.warning("test_stop_hook failed", exc_info=True)
        return None

    if summary:
        ctx.metadata["test_summary"] = summary
    return None


def _has_pytest_config(ws: Path) -> bool:
    """workspace 里有没有 pytest 配置文件."""
    try:
        return any((ws / name).is_file() for name in _PYTEST_MARKERS)
    except Exception:
        return False


async def _run_pytest(ws: Path) -> str:
    """跑 pytest --tb=short -q, 60s 超时, 返回简短摘要."""
    cmd = [sys.executable, "-m", "pytest", "--tb=short", "-q"]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(ws),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except (FileNotFoundError, OSError):
        return ""

    try:
        stdout, _ = await asyncio.wait_for(
            proc.communicate(), timeout=_TEST_TIMEOUT
        )
    except asyncio.TimeoutError:
        # 超时就杀掉, 别让 pytest 一直挂着
        proc.kill()
        await proc.wait()
        return f"pytest timed out after {_TEST_TIMEOUT}s"

    output = stdout.decode("utf-8", errors="replace")
    return _extract_summary(output)


def _extract_summary(output: str) -> str:
    """从 pytest 输出里抓最后一行摘要 (passed/failed/error)."""
    lines = output.splitlines()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        if any(kw in line for kw in ("passed", "failed", "error", "no tests ran")):
            return line
    return lines[-1].strip() if lines else ""
