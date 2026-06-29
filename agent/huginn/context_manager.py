"""Context management — memoized system/user context, git status, token window.

参考 Claude Code 的 context.ts + utils/context.ts 设计：
- 模型上下文窗口与最大输出 tokens 的查询表
- 实时计算上下文使用百分比
- 缓存的系统上下文（日期 + git 状态），对话期间只计算一次
- 用户上下文复用 project_context.py 的逻辑，读取 `.huginn.md` / `AGENTS.md`
"""

from __future__ import annotations

import functools
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .project_context import load_project_context

# 模型上下文窗口大小（按输入侧 tokens 计）
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-haiku-4": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "deepseek-chat": 128_000,
    "deepseek-coder": 128_000,
    "deepseek-reasoner": 128_000,
    "moonshot-v1-8k": 8_000,
    "moonshot-v1-32k": 32_000,
    "moonshot-v1-128k": 128_000,
    "kimi-k2-0905-preview": 256_000,
    "glm-4.6": 200_000,
    "glm-4-plus": 128_000,
    "qwen-max": 32_000,
    "qwen2.5-72b": 128_000,
    "llama-3.1-405b": 128_000,
}

# 模型最大输出 tokens: (默认值, 上限)
MODEL_MAX_OUTPUT_TOKENS: dict[str, tuple[int, int]] = {
    "claude-sonnet-4-6": (8192, 64000),
    "claude-opus-4-6": (8192, 32000),
    "claude-haiku-4": (8192, 8192),
    "gpt-4o": (16384, 16384),
    "gpt-4o-mini": (16384, 16384),
    "gpt-4.1": (32768, 32768),
    "gpt-4.1-mini": (16384, 16384),
    "deepseek-chat": (8192, 8192),
    "deepseek-coder": (8192, 8192),
    "deepseek-reasoner": (16384, 16384),
    "moonshot-v1-8k": (2048, 4096),
    "moonshot-v1-32k": (4096, 8192),
    "moonshot-v1-128k": (8192, 8192),
    "kimi-k2-0905-preview": (8192, 8192),
    "glm-4.6": (4096, 8192),
    "glm-4-plus": (4096, 8192),
    "qwen-max": (8192, 8192),
    "qwen2.5-72b": (8192, 8192),
    "llama-3.1-405b": (4096, 8192),
}

# 安全默认值：未知模型按 128k 上下文、4k 输出处理
_DEFAULT_CONTEXT_WINDOW = 128_000
_DEFAULT_MAX_OUTPUT_TOKENS: tuple[int, int] = (4096, 8192)

# git 子进程超时（秒），避免在大型仓库或网络挂载盘上卡死
_GIT_TIMEOUT = 5


def get_context_window(model: str) -> int:
    """解析模型的上下文窗口大小。

    优先级：
      1. 环境变量 ``HUGINN_MAX_CONTEXT_TOKENS``（手动覆盖）
      2. ``MODEL_CONTEXT_WINDOWS`` 中按子串匹配
      3. 回落到 ``_DEFAULT_CONTEXT_WINDOW``
    """
    if env := os.environ.get("HUGINN_MAX_CONTEXT_TOKENS"):
        try:
            return int(env)
        except ValueError:
            # 环境变量配置错误时忽略，按表查询
            pass
    model_lower = model.lower()
    for key, size in MODEL_CONTEXT_WINDOWS.items():
        if key in model_lower:
            return size
    return _DEFAULT_CONTEXT_WINDOW


def get_model_max_output_tokens(model: str) -> tuple[int, int]:
    """返回 ``(default, upper_limit)`` 输出 tokens 配置。"""
    model_lower = model.lower()
    for key, vals in MODEL_MAX_OUTPUT_TOKENS.items():
        if key in model_lower:
            return vals
    return _DEFAULT_MAX_OUTPUT_TOKENS


def calculate_context_usage(usage: dict, window: int) -> dict:
    """实时计算上下文使用百分比。

    ``usage`` 一般来自 LLM 返回的 token 统计，支持以下字段（缺失按 0 处理）：
      - ``input_tokens``
      - ``cache_read_input_tokens``
      - ``cache_creation_input_tokens``
    """
    total = (
        usage.get("input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
    )
    used_pct = min(100, round(total / window * 100)) if window > 0 else 0
    return {
        "used": used_pct,
        "remaining": 100 - used_pct,
        "total_tokens": total,
        "window": window,
    }


def format_context_usage(usage: dict, window: int) -> str:
    """格式化上下文使用情况为可读字符串（带进度条）。"""
    info = calculate_context_usage(usage, window)
    bar_len = 40
    filled = int(bar_len * info["used"] / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    return f"[{bar}] {info['used']:.1f}% ({info['total_tokens']:,}/{window:,})"


def _run_git(args: list[str], cwd: str) -> str:
    """在 workspace 下执行一条 git 命令，失败返回空串。"""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def get_git_status(workspace: str, max_chars: int = 2000) -> str | None:
    """获取 git 仓库状态摘要，拼接 branch / status / 最近提交。

    非 git 仓库返回 ``None``；输出截断到 ``max_chars`` 防止上下文膨胀。
    """
    workspace_path = Path(workspace).expanduser().resolve()
    # 探测是否为 git 仓库，避免在普通目录里报错刷屏
    rev_parse = _run_git(["rev-parse", "--is-inside-work-tree"], str(workspace_path))
    if rev_parse != "true":
        return None

    branch = _run_git(["branch", "--show-current"], str(workspace_path))
    status = _run_git(["status", "--short"], str(workspace_path))
    log = _run_git(["log", "--oneline", "-n", "5"], str(workspace_path))

    sections: list[str] = []
    if branch:
        sections.append(f"branch: {branch}")
    if status:
        sections.append("status:")
        sections.append(status)
    if log:
        sections.append("recent:")
        sections.append(log)

    if not sections:
        return None

    output = "\n".join(sections)
    if len(output) > max_chars:
        output = output[:max_chars] + "\n... (truncated)"
    return output


@functools.lru_cache(maxsize=4)
def get_system_context(workspace: str) -> dict[str, str]:
    """缓存的系统上下文，对话期间只计算一次。

    包含日期和 git 状态。``lru_cache`` 保证同一 workspace 不会重复拉取 git。
    """
    ctx: dict[str, str] = {
        "currentDate": f"Today's date is {datetime.now().strftime('%Y-%m-%d')}."
    }
    git_status = get_git_status(workspace)
    if git_status:
        ctx["gitStatus"] = git_status
    return ctx


def get_user_context(workspace: str) -> dict[str, str]:
    """用户上下文（项目特定的上下文）。

    复用 ``project_context.load_project_context`` 读取 `.huginn.md` / `AGENTS.md`，
    未配置则返回空字典。
    """
    content = load_project_context(workspace)
    if not content:
        return {}
    return {"projectContext": content}


def reset_context_cache() -> None:
    """清空 ``get_system_context`` 的 LRU 缓存。

    在新对话开始或 workspace 切换时调用，确保 git 状态不被旧值污染。
    """
    get_system_context.cache_clear()


__all__ = [
    "MODEL_CONTEXT_WINDOWS",
    "MODEL_MAX_OUTPUT_TOKENS",
    "get_context_window",
    "get_model_max_output_tokens",
    "calculate_context_usage",
    "format_context_usage",
    "get_git_status",
    "get_system_context",
    "get_user_context",
    "reset_context_cache",
]
