"""Bash tool — run shell commands inside the workspace.

Used by the Coder agent to run tests, builds, and git operations.
Always requires approval.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.security import ContainerExecutor, SandboxError, SandboxExecutor, get_executor
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class BashToolInput(BaseModel):
    action: Literal["run", "stream"] = Field(default="run")
    command: list[str] = Field(..., description="Command as a list of arguments")
    working_dir: str | None = Field(default=None)
    timeout: float = Field(default=300.0, gt=0)
    capture_output: bool = Field(default=True)
    stream: bool = Field(
        default=False,
        description="Stream stdout/stderr line-by-line while the command runs",
    )


# 常见错误的修复建议, 让 agent 知道下一步该做什么而不是干等 CONTINUE_MSG
# ponytail: 只覆盖高频模式, 不做 NLP. 升级: LLM 生成建议.
def _suggest_fix(returncode: int, stderr: str, stdout: str, command: list[str]) -> str:
    """从 stderr 提取常见错误模式, 返回具体修复建议."""
    s = (stderr or "") + (stdout or "")
    sl = s.lower()
    if "modulenotfounderror" in sl or "no module named" in sl:
        m = ""
        for line in s.splitlines():
            if "No module named" in line:
                m = line.split("named")[-1].strip().strip("'\"")
                break
        return f"ModuleNotFoundError: pip install {m} in bash_tool, then re-run."
    if "syntaxerror" in sl:
        return "SyntaxError: re-read the .py file, fix the syntax, re-run."
    if "filenotfounderror" in sl or "no such file" in sl:
        return "FileNotFoundError: check path with glob/grep, or create the file first via code_tool."
    if "importerror" in sl:
        return "ImportError: check if the module exists in workspace, or pip install it."
    if "timed out" in sl or "timeout" in sl:
        return f"Timeout: reduce iterations or split the task. Command was: {' '.join(command[:5])}."
    if "attributeerror" in sl:
        return "AttributeError: check the class/module API. Use dir() or help() in code_tool to inspect."
    if "valueerror" in sl or "typeerror" in sl:
        return "Value/TypeError: check input shapes/types. Use code_tool to print them before the failing line."
    if "runtimeerror" in sl and "cuda" in sl:
        return "CUDA RuntimeError: fall back to CPU (device='cpu'), or reduce batch size."
    if returncode != 0 and not s.strip():
        return "Command failed with no output. Check if the executable exists and is in PATH."
    return ""


# 从 stdout 提取进度行, 让 agent 快速看训练曲线 / 执行进度
# ponytail: 不是真正流式 IO (那需要 async generator), 而是从已捕获的 stdout
# 提取关键行. 升级路径: Popen 逐行读 + async yield.
_PROGRESS_KEYWORDS = (
    "loss", "epoch", "step", "it/s", "s/it", "accuracy", "error",
    "traceback", "exception", "warning", "complete", "done", "finished",
    "epoch:", "step:", "iter", "train", "val", "test", "metric", "score",
)


def _extract_progress(stdout: str, max_lines: int = 50) -> list[str]:
    """从 stdout 提取含进度关键词的行, 最多 max_lines 行.

    训练日志通常有大量 print, agent 只需看 loss/epoch/step 趋势.
    提取后 agent 能快速判断训练是否收敛 / 哪步出错.
    """
    if not stdout:
        return []
    lines = stdout.splitlines()
    progress = []
    for line in lines:
        ll = line.lower().strip()
        if not ll:
            continue
        # 含进度关键词的行, 或 Error/Traceback 块
        if any(kw in ll for kw in _PROGRESS_KEYWORDS):
            progress.append(line.rstrip())
            if len(progress) >= max_lines:
                break
    return progress


class BashTool(HuginnTool):
    """Run shell commands in the workspace."""

    name = "bash_tool"
    category = "core"
    description = (
        "Run a shell command as a list of arguments inside the workspace. "
        "Use for tests, builds, git, and other command-line tasks."
    )
    destructive = True
    input_schema = BashToolInput

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = BashToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )

        if not input_data.command:
            return ToolResult(data=None, success=False, error="Empty command.")

        # Use the Rust sandbox runner when the compiled extension is available.
        # ponytail: HUGINN_NO_RUST_SANDBOX=1 跳过 Rust sandbox — 它在某些场景
        # (RDKit+sklearn GPR) 会静默崩溃返回空 stderr, 导致 "Unknown error".
        # 升级: 修 Rust 侧的崩溃根因 (可能 fork/exec 或内存限制).
        if os.environ.get("HUGINN_NO_RUST_SANDBOX", "").lower() not in ("1", "true", "yes"):
            try:
                from huginn_ext.sandbox import (
                    run_sandboxed,  # type: ignore[import-not-found]
                )

                allowed_base_dirs = [str(work_dir.resolve()), str(Path.cwd().resolve())]
                result = run_sandboxed(
                    command=input_data.command[0],
                    args=input_data.command[1:],
                    cwd=str(work_dir),
                    timeout=input_data.timeout,
                    allowed_base_dirs=allowed_base_dirs,
                )
                if not result["success"]:
                    error = (
                        result.get("stderr")
                        or result.get("message")
                        or "Sandboxed command failed."
                    )
                else:
                    error = None
                return ToolResult(
                    data={
                        "command": input_data.command,
                        "returncode": result["returncode"],
                        "stdout": result["stdout"],
                        "stderr": result["stderr"],
                        "message": result["message"],
                        "timed_out": result["timed_out"],
                        "suggest_fix": _suggest_fix(result["returncode"], result["stderr"], result["stdout"], input_data.command) if not result["success"] else "",
                        "stream_progress": _extract_progress(result["stdout"]),
                    },
                    success=result["success"],
                    error=error,
                )
            except Exception:
                # Rust extension not available; proceed to the configured backend.
                pass

        try:
            executor = get_executor()
        except SandboxError as exc:
            return ToolResult(
                data=None, success=False, error=f"Execution blocked: {exc}"
            )

        if isinstance(executor, ContainerExecutor):
            result = executor.run(
                input_data.command,
                cwd=work_dir,
                timeout=input_data.timeout,
                capture_output=True,
                text=True,
            )
            return ToolResult(
                data={
                    "command": input_data.command,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "message": (
                        "Command succeeded." if result.success else "Command failed."
                    ),
                    "container": True,
                    "suggest_fix": _suggest_fix(result.returncode, result.stderr, result.stdout, input_data.command) if not result.success else "",
                    "stream_progress": _extract_progress(result.stdout),
                },
                success=result.success,
            )

        # SandboxExecutor path — uses executable whitelist + work-dir validation.
        if isinstance(executor, SandboxExecutor):
            try:
                result = executor.run(
                    input_data.command,
                    cwd=work_dir,
                    timeout=input_data.timeout,
                    capture_output=input_data.capture_output,
                    text=True,
                )
                return ToolResult(
                    data={
                        "command": input_data.command,
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "message": (
                            "Command succeeded."
                            if result.returncode == 0
                            else "Command failed."
                        ),
                        "sandbox": True,
                        "suggest_fix": _suggest_fix(result.returncode, result.stderr, result.stdout, input_data.command) if result.returncode != 0 else "",
                        "stream_progress": _extract_progress(result.stdout),
                    },
                    success=result.returncode == 0,
                )
            except SandboxError as e:
                return ToolResult(
                    data=None, success=False,
                    error=f"Sandbox blocked command: {e}",
                )
            except Exception as e:
                return ToolResult(
                    data=None, success=False,
                    error=f"Sandbox execution failed: {e}",
                )
