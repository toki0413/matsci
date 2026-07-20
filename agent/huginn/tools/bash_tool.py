"""Bash tool — run shell commands inside the workspace.

Used by the Coder agent to run tests, builds, and git operations.
Always requires approval.
"""

from __future__ import annotations

import os
import re
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


# v14 Task 9: bash 重活识别 heuristic
# ponytail: 关键词匹配, 不解析命令语义. 升级路径: shellparse + 历史执行时长 ML 估时.
_PY_LONG_RUN_KEYWORDS = ("train", "fit", "epoch")


def _is_heavy_bash(command: list[str]) -> tuple[bool, str]:
    """识别 bash 重活: 返回 (is_heavy, reason).

    判据 (满足任一):
      - python 跑 .py 脚本 + 含 train/fit/epoch 关键词 (典型训练任务)
      - jupyter / notebook 启动 (长任务)
    pip install / ls / cat 这类短命令不算重活.
    """
    if not command:
        return False, ""
    # 拼成字符串做包含匹配, list[str] → "python train.py --epochs 100"
    cmd_str = " ".join(command)
    cmd_lower = cmd_str.lower()

    # jupyter / notebook 启动 → 重活
    if "jupyter" in cmd_lower or "notebook" in cmd_lower:
        return True, "jupyter/notebook long-running task"

    # python 跑 .py + 训练关键词 → 重活
    # ponytail: 不用 re, 简单 substring 即可. 升级: shlex 解析精确判定.
    has_python = "python" in cmd_lower
    has_py_file = bool(re.search(r"\.py\b", cmd_str))
    if has_python and has_py_file:
        if any(kw in cmd_lower for kw in _PY_LONG_RUN_KEYWORDS):
            return True, "python training/fitting task (train/fit/epoch keyword)"

    return False, ""


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
        return "FileNotFoundError: check path with glob/grep, or create the file first via file_write_tool (code_tool sandbox blocks open())."
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

        # v14 Task 9: bash 重活识别 + 自动 dispatch 给 Support subagent
        # ponytail: 失败时降级到原行为 (直接执行 bash), 不阻塞主流程.
        is_heavy, reason = _is_heavy_bash(input_data.command)
        if is_heavy and os.environ.get("HUGINN_CORE_SUPPORT_PROTOCOL", "1") == "1":
            # v14 Task 13: PersistentTerminal 路径优先 — 长任务不被枪毙.
            from huginn.tools.persistent_terminal import (
                resolve_persistent_terminal_flag,
            )
            if resolve_persistent_terminal_flag(None):
                dispatched = _dispatch_to_support_bash_persistent(
                    input_data.command, context, reason
                )
                if dispatched is not None:
                    return dispatched
            else:
                dispatched = await _dispatch_to_support_bash(
                    input_data.command, context, reason
                )
                if dispatched is not None:
                    return dispatched

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


async def _dispatch_to_support_bash(
    command: list[str],
    context: ToolContext | None,
    reason: str,
) -> ToolResult | None:
    """把 bash_tool 重活 rewrite 为 subagent_tool dispatch.

    BashTool.call 是 async, 直接 await SubagentTool.call.
    返回 None 表示 dispatch 没成立 (context 缺 agent_factory 等), 调用方应降级.

    返回的 ToolResult 带 metadata={"dispatched_to_support": True, ...},
    rcb_runner 主循环看到后写 cochain_type="curl" trace entry.
    """
    # context 必须有 agent_factory 才能起子 agent, 没有就降级
    if context is None or getattr(context, "agent_factory", None) is None:
        return None

    try:
        from huginn.tools.subagent_tool import SubagentTool

        support_input = {
            "action": "dispatch",
            "spec_name": "support",
            "task": (
                "Run this shell command in isolation and return a JSON finding "
                "with key results / evidence / limitations / artifacts.\n\n"
                f"$ {' '.join(command)}"
            ),
        }
        result = await SubagentTool().call(support_input, context)
    except Exception:
        # best-effort: dispatch 失败就降级到本地执行, 不阻塞主流程
        return None

    # v14 Task 10: Čech H¹ 一致性检查 — Support finding vs Core context (原 command 作为 proxy).
    # ponytail: 真正 Core context 拿不到, 用触发 dispatch 的 command 当 claim proxy.
    # 升级路径: 从 ToolContext.memory_manager 取最近 N 条 Core message 做 core_context.
    if result.success and isinstance(result.data, dict):
        finding = result.data.get("summary")
        if finding:
            from huginn.agents.subagent import (
                _check_finding_consistency,
                _write_support_rejection,
            )
            core_context = " ".join(command)
            h1_zero, h1_reason = _check_finding_consistency(finding, core_context)
            result.metadata["h1_status"] = "zero" if h1_zero else "nonzero"
            result.metadata["h1_reason"] = h1_reason
            if not h1_zero:
                _write_support_rejection(
                    context.workspace, finding, h1_reason, core_context,
                )
                result.data = {
                    "summary": None,
                    "message": (
                        f"Support finding rejected: {h1_reason}. "
                        "Finding written to .huginn/support_rejections.jsonl for later review."
                    ),
                }
                result.metadata["h1_obstruction"] = True
                result.metadata["rejection_reason"] = h1_reason

    result.metadata["dispatched_to_support"] = True
    result.metadata["dispatch_reason"] = reason
    return result


def _dispatch_to_support_bash_persistent(
    command: list[str],
    context: ToolContext | None,
    reason: str,
) -> ToolResult | None:
    """v14 Task 13: bash 重活走 PersistentTerminal — 启 session 跑原 command, 立即返回.

    不等 Support 完成就返回 session_id, Core 通过 poll_support_session 续轮.
    返回 None 表示启动 session 失败, 调用方降级.

    ponytail: 直接把 command list 交给 PersistentTerminal.start — Windows 上
    _SubprocessHandle 走 shell=False, list 模式更安全. 升级路径: 跟
    _dispatch_to_support_persistent 一样跑 SubagentDispatch 而非裸 command.
    """
    workspace = None
    if context is not None and getattr(context, "workspace", None):
        workspace = str(context.workspace)

    try:
        from huginn.tools.persistent_terminal import get_default_terminal

        terminal = get_default_terminal()
        session_id = terminal.start(command, cwd=workspace)
    except Exception:
        return None

    result = ToolResult(
        data={
            "session_id": session_id,
            "status": "started",
            "summary": None,
            "message": (
                f"Support bash task started in persistent session {session_id}. "
                "Use poll_support_session(session_id) to read incremental output."
            ),
        },
        success=True,
    )
    result.metadata["dispatched_to_support"] = True
    result.metadata["dispatched_via_persistent_terminal"] = True
    result.metadata["session_id"] = session_id
    result.metadata["dispatch_reason"] = reason
    return result


def self_check_v14_task9_bash() -> None:
    """v14 Task 9 bash: 重活识别 self-check."""
    # python + train.py + train keyword → 重活
    is_heavy, reason = _is_heavy_bash(["python", "train.py", "--epochs", "100"])
    assert is_heavy, "train.py 应识别为重活"
    assert "train" in reason or "fit" in reason or "epoch" in reason, reason

    # python + fit.py → 重活
    is_heavy, _ = _is_heavy_bash(["python", "fit_model.py"])
    assert is_heavy, "fit_model.py 应识别为重活"

    # jupyter notebook → 重活
    is_heavy, reason = _is_heavy_bash(["jupyter", "notebook"])
    assert is_heavy and "jupyter" in reason, reason

    # pip install → 不是重活
    is_heavy, _ = _is_heavy_bash(["pip", "install", "torch"])
    assert not is_heavy, "pip install 不应识别为重活"

    # 普通短命令 → 不是重活
    is_heavy, _ = _is_heavy_bash(["ls", "-la"])
    assert not is_heavy, "ls 不应识别为重活"

    # python 跑 .py 但无训练关键词 → 不是重活 (e.g. python hello.py)
    is_heavy, _ = _is_heavy_bash(["python", "hello.py"])
    assert not is_heavy, "无训练关键词的 python .py 不应识别为重活"

    print("[CHECK v14 Task 9 bash] heavy bash detection OK")


if __name__ == "__main__":
    self_check_v14_task9_bash()
