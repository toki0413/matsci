"""文件改动后自动 lint 的 PostToolUse 钩子.

只在 .py 文件被 file_edit_tool / file_write_tool 改动后触发.
优先用 ruff check, 没装 ruff 就退到 py_compile 做语法检查.
不阻断工具执行, 只把 lint 结果塞进 ctx.metadata['lint_errors'].
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
from pathlib import Path

from huginn.hooks import HookContext

logger = logging.getLogger(__name__)

# 这些工具会改文件, 改完需要 lint
_LINT_TRIGGERS = {"file_edit_tool", "file_write_tool"}

# 单文件 lint 最长等 30s, 再久就放过去别卡 agent
_LINT_TIMEOUT = 30


async def lint_post_tool_hook(ctx: HookContext) -> HookContext | None:
    """文件改动后自动 lint, 结果写进 metadata, 不阻断主流程."""
    if ctx.tool_name not in _LINT_TRIGGERS:
        return None
    args = ctx.args if isinstance(ctx.args, dict) else {}
    file_path = args.get("file_path", "") or args.get("path", "")
    if not isinstance(file_path, str) or not file_path.endswith(".py"):
        return None

    path = Path(file_path)
    if not path.is_file():
        return None

    try:
        errors = await _run_lint(path)
    except Exception:
        # lint 失败不能影响 agent 主流程
        logger.warning("lint hook failed for %s", file_path, exc_info=True)
        return None

    if errors:
        ctx.metadata["lint_errors"] = errors
        ctx.metadata["lint_file"] = str(path)
    return None


async def _run_lint(path: Path) -> list[str]:
    """跑 ruff (有就用) 或 py_compile (兜底), 返回错误行列表."""
    ruff_bin = shutil.which("ruff")
    if ruff_bin:
        return await _run_ruff(path, ruff_bin)
    # ruff 没装就退到 py_compile, 至少能抓语法错
    return await _run_py_compile(path)


async def _run_ruff(path: Path, ruff_bin: str) -> list[str]:
    """ruff check, 收集报错行."""
    cmd = [ruff_bin, "check", "--no-fix", str(path)]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_LINT_TIMEOUT
        )
    except (FileNotFoundError, asyncio.TimeoutError):
        return []

    output = (stdout + stderr).decode("utf-8", errors="replace")
    if proc.returncode == 0:
        return []
    return [line for line in output.splitlines() if line.strip()]


async def _run_py_compile(path: Path) -> list[str]:
    """没 ruff 时用 py_compile 做语法检查."""
    # 用子进程跑, 避免污染当前解释器状态
    code = (
        "import py_compile, sys; "
        f"py_compile.compile(r'{path}', doraise=True)"
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_LINT_TIMEOUT
        )
    except (FileNotFoundError, asyncio.TimeoutError):
        return []

    if proc.returncode == 0:
        return []
    err = stderr.decode("utf-8", errors="replace")
    return [line for line in err.splitlines() if line.strip()]
