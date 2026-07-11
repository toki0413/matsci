"""Slash 命令处理器 — 在 chat 循环里拦截 / 开头的命令。

命令不送给 agent, 直接在本地执行, 类似 Claude Code 的 /help /compact 之类。
自定义命令 (/foo) 会被展开成 prompt 返回给 agent。
"""

from __future__ import annotations

import json
import os
import subprocess
from contextlib import suppress
from pathlib import Path
from typing import Any
import logging
logger = logging.getLogger(__name__)



def _get_console(console: Any | None = None):
    """拿一个可用的 Console, 没传就用 rich 默认的。"""
    if console is not None:
        return console
    from rich.console import Console

    return Console()


def _get_workspace(ctx: Any | None = None) -> Path:
    """从 ctx 挖 workspace 路径, 没有就用当前目录。"""
    if ctx is not None:
        ws = getattr(ctx, "workspace", None)
        if ws is not None:
            return Path(ws)
    return Path(".")


# ── 内置命令名, 自定义命令不能跟这些重名 ──────────────────────────────

_BUILTIN_COMMANDS: frozenset[str] = frozenset(
    {
        "help",
        "compact",
        "clear",
        "context",
        "cost",
        "undo",
        "tools",
        "sessions",
        "bg",
        "step",
        "map",
        "plan",
        "research",
        "subgoal",
    }
)


def _print_help(console: Any) -> None:
    """列出所有可用的 slash 命令。"""
    help_text = """[bold]可用命令:[/bold]

  [cyan]/help[/cyan]      显示本帮助
  [cyan]/compact[/cyan]   手动触发上下文压缩 (promote session summary)
  [cyan]/clear[/cyan]     清空当前会话上下文
  [cyan]/context[/cyan]   显示上下文使用情况
  [cyan]/cost[/cyan]      显示当前会话的 token 消耗统计
  [cyan]/undo[/cyan]      回滚到上一个 checkpoint (或 git stash / 撤销最后一条消息)
  [cyan]/tools[/cyan]     列出当前已注册的工具
  [cyan]/sessions[/cyan]  列出历史会话
  [cyan]/bg[/cyan]        后台任务: /bg <objective> | /bg list | /bg status <id> | /bg stop <id> | /bg result <id>
  [cyan]/step[/cyan]      单步执行: /step on | /step off | /step (切换)
  [cyan]/map[/cyan]       显示代码库地图: /map | /map <query>
  [cyan]/plan[/cyan]     进入计划模式 (只规划不执行, 写工具需确认)
  [cyan]/research[/cyan] 进入研究模式 (全工具, 数学深度引导, Deli 管线)
  [cyan]/subgoal[/cyan]  追加子目标约束: /subgoal <约束描述> (不中断当前任务)"""
    console.print(help_text)

    # 顺便列一下当前 workspace 的自定义命令
    try:
        from huginn.cli.custom_commands import list_custom_commands

        customs = list_custom_commands()
        if customs:
            console.print("\n[bold]自定义命令:[/bold]")
            for name in customs:
                console.print(f"  [cyan]/{name}[/cyan]")
    except Exception:
        # 加载失败不影响 help
        pass


def _handle_compact(agent: Any, console: Any) -> None:
    """手动触发上下文压缩, 把 session summary 提升到 mid tier。"""
    if agent is None:
        console.print("[yellow]Agent 未初始化, 无法压缩[/yellow]")
        return
    memory = getattr(agent, "memory", None)
    if memory is None:
        console.print("[yellow]没有 memory_manager, 无法压缩[/yellow]")
        return
    try:
        memory_id = memory.promote_session_summary(tier="mid")
        console.print(f"[green]✓[/green] 会话已压缩, memory_id={memory_id}")
    except Exception as e:
        console.print(f"[red]压缩失败: {e}[/red]")


def _handle_clear(agent: Any, console: Any) -> None:
    """清空当前会话上下文。"""
    if agent is None:
        console.print("[yellow]Agent 未初始化[/yellow]")
        return
    memory = getattr(agent, "memory", None)
    if memory is None:
        console.print("[yellow]没有 memory_manager[/yellow]")
        return
    try:
        memory.clear_session()
        console.print("[green]✓[/green] 会话上下文已清空")
    except Exception as e:
        console.print(f"[red]清空失败: {e}[/red]")


def _handle_context(agent: Any, console: Any) -> None:
    """显示上下文使用情况, 调用 context_manager 的两个工具函数。"""
    if agent is None:
        console.print("[yellow]Agent 未初始化[/yellow]")
        return
    try:
        from huginn.context_manager import (
            format_context_usage,
            get_context_window,
        )
    except Exception as e:
        console.print(f"[red]无法导入 context_manager: {e}[/red]")
        return

    # 从 agent 上拿模型名和最近一次的 usage 统计
    model_name = _resolve_model_name(agent)
    window = get_context_window(model_name) if model_name else 128_000
    usage = getattr(agent, "_last_cache_stats", {}) or {}

    if not usage:
        # 没有缓存统计就退而求其次, 显示 0
        usage = {"input_tokens": 0}

    try:
        text = format_context_usage(usage, window)
        console.print(f"[blue]模型:[/blue] {model_name or 'unknown'}")
        console.print(f"[blue]窗口:[/blue] {window:,} tokens")
        console.print(f"[blue]使用:[/blue] {text}")
    except Exception as e:
        console.print(f"[red]计算上下文使用失败: {e}[/red]")


def _resolve_model_name(agent: Any) -> str:
    """从 agent 上挖出当前模型名, 给 context_window 查表用。"""
    # 优先用 model_router 上注册的 alias
    router = getattr(agent, "model_router", None)
    if router is not None:
        # ModelRouter 一般有 _providers 或类似结构, 取第一个可用 alias
        providers = getattr(router, "_providers", None)
        if providers and isinstance(providers, dict) and providers:
            # 取第一个 key 当模型名近似值
            first_key = next(iter(providers))
            return str(first_key)
    model = getattr(agent, "model", None)
    if model is not None:
        # langchain 的 model 对象一般有 model_name / model 属性
        for attr in ("model_name", "model", "deployment", "name"):
            val = getattr(model, attr, None)
            if isinstance(val, str) and val:
                return val
    return ""


def _handle_cost(agent: Any, console: Any) -> None:
    """从 agent._telemetry_collector 聚合 token 消耗统计。"""
    if agent is None:
        console.print("[yellow]Agent 未初始化[/yellow]")
        return
    collector = getattr(agent, "_telemetry_collector", None)
    if collector is None:
        console.print("[yellow]没有 telemetry collector[/yellow]")
        return

    try:
        summary = collector.summary()
    except Exception as e:
        console.print(f"[red]telemetry summary 失败: {e}[/red]")
        return

    by_name = summary.get("by_name", {})
    if not by_name:
        console.print("[dim]还没有 telemetry 数据[/dim]")
        return

    # 把 span 数和总耗时打印出来, 让用户看到 agent 跑了多少轮
    console.print(f"[bold]总 span 数:[/bold] {summary.get('total_spans', 0)}")
    console.print("[bold]按操作分类:[/bold]")
    for name, info in by_name.items():
        count = info.get("count", 0)
        duration = info.get("duration_ms", 0)
        console.print(
            f"  [cyan]{name:30s}[/cyan]  "
            f"调用 {count:>4d} 次  "
            f"耗时 {duration / 1000:.2f}s"
        )

    # 缓存统计里通常有 token 用量
    cache_stats = getattr(agent, "_last_cache_stats", {}) or {}
    if cache_stats:
        console.print("[bold]Token 用量 (最近一次):[/bold]")
        for key, val in cache_stats.items():
            if isinstance(val, (int, float)) and val:
                console.print(f"  [cyan]{key:30s}[/cyan]  {val:,}")


# ── /undo ────────────────────────────────────────────────────────────


def _detect_server_url() -> str | None:
    """探测 huginn server 地址。

    优先用环境变量 HUGINN_SERVER_URL, 没设就试默认端口 8001。
    探测失败返回 None。
    """
    url = os.environ.get("HUGINN_SERVER_URL")
    if url:
        return url.rstrip("/")

    # 试默认端口, 1.5s 超时, 没响应就当没 server
    default = "http://127.0.0.1:8001"
    try:
        import urllib.request

        req = urllib.request.Request(f"{default}/health", method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            if resp.status == 200:
                return default
    except Exception:
        logger.debug("detect server url failed", exc_info=True)
    return None


def _undo_via_routes_api(console: Any) -> bool:
    """走 HTTP routes API 回滚最近一个 checkpoint。

    成功返回 True, 服务器不可用或没 checkpoint 返回 False。
    调用的是 routes/checkpoints.py 里的 reject 端点。
    """
    base = _detect_server_url()
    if base is None:
        return False

    import urllib.request
    import urllib.error

    try:
        # 先列所有 checkpoint, 找最后一个 (没法直接拿最后一个, 只能 list)
        # routes 里没有 list all 端点, 但 server_core._checkpoints 是 dict,
        # 我们直接走内存路径就行。这里试一下 GET /checkpoints 看有没有 list 端点
        # 没有就走 _undo_via_memory
        with urllib.request.urlopen(
            urllib.request.Request(f"{base}/checkpoints", method="GET"),
            timeout=2.0,
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
        return False

    # 如果 server 返回了 checkpoint 列表, 拿最后一个 reject
    items = data if isinstance(data, list) else data.get("checkpoints", [])
    if not items:
        return False

    # 列表项可能是 str (id) 或 dict (含 id)
    last = items[-1]
    cp_id = last if isinstance(last, str) else last.get("id")
    if not cp_id:
        return False

    try:
        req = urllib.request.Request(
            f"{base}/checkpoints/{cp_id}/reject",
            method="POST",
            data=b"",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        if result.get("success"):
            console.print(
                f"[green]✓[/green] 已通过 routes API 回滚到 checkpoint {cp_id}"
            )
            return True
        console.print(f"[yellow]routes API 返回: {result}[/yellow]")
        return True
    except Exception as e:
        console.print(f"[yellow]routes API 调用失败: {e}[/yellow]")
        return False


def _undo_via_memory_checkpoints(console: Any) -> bool:
    """直接从 server_core._checkpoints 拿最后一个 checkpoint 回滚。

    CLI 模式下没有 server 跑着, _checkpoints 通常是空的, 这种情况返回 False
    让上层走 git stash 兜底。
    """
    try:
        from huginn.server_core import _checkpoints, _state_lock
    except Exception as e:
        console.print(f"[yellow]checkpoint 服务不可用: {e}[/yellow]")
        return False

    with _state_lock:
        if not _checkpoints:
            return False
        # dict 保序, 最后插入的就是最近的 checkpoint
        cp_id = next(reversed(_checkpoints))
        base, snapshot = _checkpoints.pop(cp_id)

    # 把 snapshot 里的文件写回去, 同时删掉 snapshot 之后新增的文件
    # 逻辑跟 routes/checkpoints.py 的 reject_checkpoint 一致
    try:
        current = _snapshot_directory_for_cli(base)
        for rel, content in snapshot.items():
            if rel in current:
                path = base / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
        # 删掉 snapshot 之后才出现的文件
        for rel in current:
            if rel not in snapshot:
                path = base / rel
                with suppress(Exception):
                    path.unlink()
        console.print(
            f"[green]✓[/green] 已回滚到 checkpoint {cp_id} "
            f"(base={base}, {len(snapshot)} 个文件)"
        )
        return True
    except Exception as e:
        console.print(f"[red]回滚失败: {e}[/red]")
        return False


def _undo_via_git_stash(console: Any, workspace: Path) -> bool:
    """用 git stash 把工作区改动暂存起来, 相当于回到上次 commit。

    只在 git 仓库里有效, 没有 git 或没有改动返回 False。
    """
    if not (workspace / ".git").exists():
        return False

    # 先看有没有改动, 没改动就不用 stash
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

    if result.returncode != 0:
        return False
    if not result.stdout.strip():
        console.print("[dim]git 工作区是干净的, 没东西可撤销[/dim]")
        return True

    # git stash 暂存当前改动, 用户后续可以 git stash pop 找回
    try:
        stash_result = subprocess.run(
            ["git", "stash", "push", "-m", "huginn-undo"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except subprocess.TimeoutExpired:
        console.print("[red]git stash 超时[/red]")
        return False

    if stash_result.returncode != 0:
        console.print(f"[red]git stash 失败: {stash_result.stderr.strip()}[/red]")
        return False

    console.print(
        f"[green]✓[/green] 已 git stash 工作区改动 "
        f"(可用 [cyan]git stash pop[/cyan] 找回)"
    )
    return True


def _undo_last_message(agent: Any, console: Any) -> bool:
    """兜底方案: 从内存 session 里弹出最后一条消息对。

    只影响当前进程的内存, 不动持久化的 checkpoint。
    如果最后一条是 assistant, 把它和前面的 user 一起弹掉。
    """
    if agent is None:
        return False
    memory = getattr(agent, "memory", None)
    if memory is None:
        return False
    session = getattr(memory, "session", None)
    if session is None:
        return False
    messages = getattr(session, "messages", None)
    if not messages:
        return False

    popped = []
    # 先弹最后一条 (一般是 assistant 回复)
    popped.append(messages.pop())
    # 如果上一条是 user, 也一起弹掉, 这样下一轮对话不会带着这条
    if messages and getattr(messages[-1], "role", None) == "user":
        popped.append(messages.pop())

    console.print(
        f"[green]✓[/green] 已从内存 session 弹出 {len(popped)} 条消息 "
        f"(不影响持久化 checkpoint)"
    )
    return True


def _handle_undo(agent: Any, console: Any, ctx: Any | None = None) -> None:
    """回滚到上一个 checkpoint。

    优先级:
      1. routes API (HTTP, 跨进程)
      2. server_core._checkpoints 内存 (CLI 同进程)
      3. git stash (工作区改动暂存)
      4. 撤销最后一条消息 (仅内存 session)
    """
    # 1. 先试 routes API, 跨进程最可靠
    if _undo_via_routes_api(console):
        return

    # 2. 内存 checkpoint, CLI 模式下用这个
    if _undo_via_memory_checkpoints(console):
        return

    workspace = _get_workspace(ctx)

    # 3. git stash 兜底
    if _undo_via_git_stash(console, workspace):
        return

    # 4. 撤销最后一条消息, 最后的兜底
    if _undo_last_message(agent, console):
        return

    console.print(
        "[yellow]没有可撤销的内容[/yellow] "
        "(无 checkpoint / 非 git 仓库 / session 为空)"
    )


def _snapshot_directory_for_cli(base):
    """跟 server_core._snapshot_directory 同样的逻辑, 但不依赖 server 跑起来。

    只读文本文件, 二进制跳过。
    """
    base = Path(base)
    snapshot: dict[str, str] = {}
    if not base.exists():
        return snapshot
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        # 跳过 .git / __pycache__ / .venv 之类噪音目录
        parts = path.relative_to(base).parts
        if any(p.startswith(".git") or p == "__pycache__" or p == ".venv" for p in parts):
            continue
        try:
            snapshot[str(path.relative_to(base)).replace("\\", "/")] = path.read_text(
                encoding="utf-8"
            )
        except (UnicodeDecodeError, OSError):
            # 二进制文件跳过
            continue
    return snapshot


def _handle_tools(agent: Any, console: Any) -> None:
    """列出当前已注册的工具。"""
    if agent is None:
        console.print("[yellow]Agent 未初始化[/yellow]")
        return
    tools = getattr(agent, "langchain_tools", [])
    if not tools:
        console.print("[dim]还没有注册工具[/dim]")
        return
    console.print(f"[bold]已注册 {len(tools)} 个工具:[/bold]")
    for t in tools:
        name = getattr(t, "name", None) or getattr(t, "__name__", str(t))
        desc = getattr(t, "description", "") or ""
        # 描述太长就截断, 一行能放下
        if len(desc) > 60:
            desc = desc[:57] + "..."
        console.print(f"  [cyan]{name:30s}[/cyan]  {desc}")


def _handle_step(agent: Any, args: str, console: Any) -> None:
    """切换单步执行模式。

    /step on    开启: 每个工具调用完暂停, chat 流会 yield 一个 tool_break 标记
    /step off   关闭: 恢复正常的一次性流式输出
    /step       无参数就切换当前状态
    """
    if agent is None:
        console.print("[yellow]Agent 未初始化[/yellow]")
        return
    arg = args.strip().lower()
    current = bool(getattr(agent, "_break_after_tool", False))
    if arg == "on":
        agent._break_after_tool = True
        state = "开启"
    elif arg == "off":
        agent._break_after_tool = False
        state = "关闭"
    else:
        # 无参数: 切换, 方便快速 toggle
        agent._break_after_tool = not current
        state = "开启" if agent._break_after_tool else "关闭"
    console.print(f"[green]✓[/green] 单步模式已{state}")


def _handle_sessions(console: Any) -> None:
    """列出历史会话, 复用 sessions 命令里的 _list_threads。"""
    try:
        from huginn.cli.commands.sessions import _get_db_path, _list_threads
    except Exception as e:
        console.print(f"[yellow]无法加载 sessions 模块: {e}[/yellow]")
        return

    db_path = _get_db_path()
    threads = _list_threads(db_path)
    if not threads:
        console.print("[yellow]没有历史会话[/yellow]")
        return

    console.print(f"[bold]历史会话 (共 {len(threads)} 个):[/bold]")
    for t in threads:
        thread_id = t["thread_id"]
        created = t.get("created")
        created_str = created.strftime("%Y-%m-%d %H:%M") if created else "—"
        messages = t.get("messages", 0)
        preview = t.get("preview", "") or ""
        console.print(
            f"  [cyan]{thread_id}[/cyan]  "
            f"[green]{created_str}[/green]  "
            f"{messages} cp  "
            f"[dim]{preview}[/dim]"
        )


# ── /map 代码库地图 ──────────────────────────────────────────────────


def _handle_map(command: str, console: Any, ctx: Any | None = None) -> None:
    """显示 repo map — 用 tree-sitter 抽符号 + PageRank 排序。

    /map            全局 top-N 符号
    /map <query>    跟 query 相关的符号 (符号名/文件名片段都行)
    """
    workspace = _get_workspace(ctx)
    parts = command.split(None, 1)
    query = parts[1].strip() if len(parts) > 1 else ""

    try:
        from huginn.coder.repo_map import RepoMap
    except Exception as e:
        console.print(f"[red]无法加载 RepoMap: {e}[/red]")
        return

    try:
        rm = RepoMap(workspace)
        rm.build()
        text = rm.get_map(query=query or None)
    except Exception as e:
        console.print(f"[red]生成 repo map 失败: {e}[/red]")
        return

    if not text:
        console.print("[yellow]repo map 为空[/yellow]")
        return

    # 标题行单独打, 正文走 dim 让符号行不那么扎眼
    if query:
        console.print(f"[bold]代码库地图 (query={query}):[/bold]")
    else:
        console.print("[bold]代码库地图 (global):[/bold]")
    console.print(text)


# ── /bg 后台任务快捷方式 ─────────────────────────────────────────────


_BG_SUBCOMMANDS = frozenset({"list", "status", "stop", "result"})


def _handle_bg(command: str, agent: Any, console: Any) -> None:
    """处理 /bg 快捷方式, 转发到 BackgroundTaskManager。

    /bg <objective>     启动后台任务 (用当前 chat 的 agent)
    /bg list            列出所有任务
    /bg status <id>     查看任务状态
    /bg stop <id>       停止任务
    /bg result <id>     查看任务结果
    """
    from huginn.cli.commands.background import BackgroundTaskManager

    body = command.lstrip("/")
    parts = body.split(None, 2)
    # parts[0] == "bg"
    sub = parts[1].lower() if len(parts) > 1 else ""
    arg = parts[2] if len(parts) > 2 else ""

    manager = BackgroundTaskManager.get_instance()

    if sub == "list":
        tasks = manager.list_tasks()
        if not tasks:
            console.print("[yellow]没有后台任务[/yellow]")
            return
        console.print(f"[bold]后台任务 (共 {len(tasks)} 个):[/bold]")
        for t in tasks:
            status = t.get("status", "?")
            console.print(
                f"  [cyan]{t.get('task_id', '')}[/cyan]  "
                f"[{status}]  "
                f"{t.get('objective', '')[:40]}"
            )
        return

    if sub == "status":
        if not arg:
            console.print("[yellow]用法: /bg status <task_id>[/yellow]")
            return
        task = manager.get_status(arg)
        if task is None:
            console.print(f"[red]任务 {arg} 不存在[/red]")
            return
        for k, v in task.items():
            console.print(f"[cyan]{k}[/cyan]: {v}")
        return

    if sub == "stop":
        if not arg:
            console.print("[yellow]用法: /bg stop <task_id>[/yellow]")
            return
        ok, msg = manager.stop(arg)
        if ok:
            console.print(f"[green]✓[/green] 任务 {arg} {msg}")
        else:
            console.print(f"[yellow]任务 {arg}: {msg}[/yellow]")
        return

    if sub == "result":
        if not arg:
            console.print("[yellow]用法: /bg result <task_id>[/yellow]")
            return
        text = manager.get_result(arg)
        if text is None:
            console.print(f"[yellow]任务 {arg} 还没有结果[/yellow]")
            return
        console.print(text)
        return

    # 不是已知子命令, 把 sub + arg 拼起来当 objective
    # 即 /bg 筛选100种钙钛矿材料 -> objective = "筛选100种钙钛矿材料"
    objective = " ".join(parts[1:]).strip()
    if not objective:
        console.print(
            "[yellow]用法:[/yellow]\n"
            "  [cyan]/bg <objective>[/cyan]      启动后台任务\n"
            "  [cyan]/bg list[/cyan]             列出所有任务\n"
            "  [cyan]/bg status <id>[/cyan]      查看状态\n"
            "  [cyan]/bg stop <id>[/cyan]        停止任务\n"
            "  [cyan]/bg result <id>[/cyan]      查看结果"
        )
        return

    if agent is None:
        console.print("[yellow]Agent 未初始化, 无法启动后台任务[/yellow]")
        return

    task_id = manager.start(objective, agent=agent)
    console.print(f"[green]✓[/green] 后台任务已启动, id={task_id}")
    console.print(
        f"[dim]用 [cyan]/bg status {task_id}[/cyan] 查看进度, "
        f"[cyan]/bg result {task_id}[/cyan] 看结果[/dim]"
    )


# ── 自定义命令 ───────────────────────────────────────────────────────


def _try_custom_command(
    command: str,
    ctx: Any | None = None,
    console: Any | None = None,
) -> str | None:
    """检查是否是自定义命令, 是的话返回展开后的 prompt。

    不是自定义命令返回 None, 让上层继续处理。
    """
    try:
        from huginn.cli.custom_commands import resolve_custom_command
    except Exception:
        return None

    workspace = _get_workspace(ctx)
    try:
        return resolve_custom_command(command, workspace=str(workspace))
    except Exception as e:
        if console is not None:
            console.print(f"[yellow]自定义命令加载失败: {e}[/yellow]")
        return None


# ── 主入口 ──────────────────────────────────────────────────────────


def _handle_subgoal(command: str, agent: Any, console: Any) -> None:
    """/subgoal <text> — 追加子目标约束到当前任务.

    存到 agent._sub_goals, context builder 会注入到后续 prompt.
    不中断当前任务, 不需要重开 goal.
    """
    parts = command.split(None, 1)
    if len(parts) < 2 or not parts[1].strip():
        console.print("[yellow]用法: /subgoal <约束描述>[/yellow]")
        console.print("[dim]例: /subgoal 所有修改不改动原有接口[/dim]")
        existing = getattr(agent, "_sub_goals", [])
        if existing:
            console.print(f"[dim]当前子目标 ({len(existing)}):[/dim]")
            for i, sg in enumerate(existing, 1):
                console.print(f"  [dim]{i}. {sg}[/dim]")
        return

    text = parts[1].strip()
    if agent is None:
        console.print("[yellow]Agent 未初始化[/yellow]")
        return
    if not hasattr(agent, "_sub_goals"):
        agent._sub_goals = []
    agent._sub_goals.append(text)
    console.print(f"[green]✓[/green] 子目标已追加 ({len(agent._sub_goals)}): {text}")
    console.print("[dim]后续 LLM 调用会自动包含此约束[/dim]")


def handle_slash_command(
    command: str,
    agent: Any | None = None,
    ctx: Any | None = None,
    console: Any | None = None,
) -> str | None:
    """处理 / 开头的 slash 命令。

    返回值:
      - None: 命令已处理, 不需要发给 agent
      - str: 替换用户输入, 把这个字符串发给 agent (自定义命令走这条路)
    """
    con = _get_console(console)
    # 去掉开头 /, 按空格切出命令名和参数
    body = command.lstrip("/")
    parts = body.split(None, 1)
    name = parts[0].lower() if parts else ""
    # args = parts[1] if len(parts) > 1 else ""

    if name == "help":
        _print_help(con)
        return None

    if name == "compact":
        _handle_compact(agent, con)
        return None

    if name == "clear":
        _handle_clear(agent, con)
        return None

    if name == "context":
        _handle_context(agent, con)
        return None

    if name == "cost":
        _handle_cost(agent, con)
        return None

    if name == "undo":
        _handle_undo(agent, con, ctx=ctx)
        return None

    if name == "tools":
        _handle_tools(agent, con)
        return None

    if name == "sessions":
        _handle_sessions(con)
        return None

    if name == "bg":
        _handle_bg(command, agent, con)
        return None

    if name == "step":
        # /step on | /step off | /step
        arg = parts[1] if len(parts) > 1 else ""
        _handle_step(agent, arg, con)
        return None

    if name == "map":
        _handle_map(command, con, ctx=ctx)
        return None

    if name == "plan":
        if agent is not None and hasattr(agent, "set_mode"):
            agent.set_mode("plan")
            con.print("[green]✓[/green] Plan mode: write tools require confirmation")
        else:
            con.print("[yellow]Agent 未初始化[/yellow]")
        return None

    if name == "research":
        if agent is not None and hasattr(agent, "set_mode"):
            agent.set_mode("research")
            con.print("[green]✓[/green] Research mode: all tools, math depth guide, Deli pipeline")
        else:
            con.print("[yellow]Agent 未初始化[/yellow]")
        return None

    if name == "subgoal":
        _handle_subgoal(command, agent, con)
        return None

    # 不在内置命令里, 试自定义命令
    if name and name not in _BUILTIN_COMMANDS:
        expanded = _try_custom_command(command, ctx=ctx, console=con)
        if expanded is not None:
            return expanded

    con.print(f"[yellow]未知命令: /{name}[/yellow]")
    con.print("[dim]输入 /help 查看可用命令[/dim]")
    return None


__all__ = ["handle_slash_command"]
