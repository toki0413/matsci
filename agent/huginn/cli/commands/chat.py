"""Interactive chat command."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import click
from langchain_core.messages import ToolMessage

from huginn import __version__
from huginn.cli.context import CliContext, build_agent_from_ctx, init_mcp, shutdown_mcp
from huginn.cli.design_system import get_design_system
from huginn.cli.input_parser import has_at_references, parse_at_references
from huginn.cli.slash_commands import handle_slash_command
from huginn.pet import get_pet_avatar


def _make_approval_callback(ctx: CliContext):
    """Interactive approval prompt for ASK-mode tools."""

    def callback(tool_name: str, reason: str) -> bool:
        # 走统一 design_system, 保证视觉一致 + 可扩展 diff 预览
        return get_design_system().permission_request(tool=tool_name, reason=reason)

    return callback


def _warn_missing_api_keys(ctx: CliContext) -> None:
    """启动时检查 API key, 有缺失就提示用户跑 huginn api-keys。

    只看常见的服务 (tavily/openai/anthropic 等), 全配了就不提示。
    """
    try:
        from huginn.cli.api_key_setup import check_api_keys
    except Exception:
        # 模块加载失败别影响 chat 启动
        return

    status = check_api_keys()
    missing = [name for name, ok in status.items() if not ok]
    if not missing:
        return

    ctx.console.print(
        "[yellow]提示: 运行 huginn api-keys 配置 API key[/yellow]"
    )
    ctx.console.print(f"[dim]未配置: {', '.join(missing)}[/dim]")


def _looks_like_diff(content: str) -> bool:
    """启发式判断文本是否是 unified diff。

    同时出现 --- / +++ 行, 或含 @@ hunk 头, 就认为是 diff。
    """
    has_minus = has_plus = has_hunk = False
    for line in content.splitlines():
        if line.startswith("---"):
            has_minus = True
        elif line.startswith("+++"):
            has_plus = True
        elif line.startswith("@@"):
            has_hunk = True
    return (has_minus and has_plus) or has_hunk


def _extract_diff_from_messages(messages: list) -> str | None:
    """从 ToolMessage 列表中提取 diff 文本。

    扫描所有工具消息, 把看起来是 diff 的内容拼起来返回。
    没有匹配返回 None, 不影响正常消息展示。
    """
    if not messages:
        return None
    diff_chunks: list[str] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        content = getattr(msg, "content", "")
        if not isinstance(content, str) or not content:
            continue
        if _looks_like_diff(content):
            diff_chunks.append(content)
    if not diff_chunks:
        return None
    return "\n".join(diff_chunks)


@click.command()
@click.pass_obj
def chat(ctx: CliContext) -> None:
    """Start interactive chat with the Agent."""
    ds = get_design_system()
    avatar = get_pet_avatar()
    ds.dialog(
        title="Welcome",
        content=(
            f"[dim]{avatar}[/dim]\n"
            f"[bold blue]Huginn[/bold blue] v{__version__}\n"
            f"Workspace: {ctx.workspace}\n"
            f"Type your materials science questions or 'exit' to quit."
        ),
    )

    # plan / yolo 模式提示, 让用户知道当前处于什么状态
    if ctx.plan_mode:
        ctx.console.print("[dim]Plan mode: 只读模式, 写操作需确认[/dim]")
    if ctx.yolo:
        ctx.console.print("[yellow]YOLO mode: 所有操作自动执行[/yellow]")
    if ctx.resume_thread_id:
        ctx.console.print(
            f"[cyan]Resuming session: {ctx.resume_thread_id}[/cyan]"
        )

    agent = build_agent_from_ctx(ctx, approval_callback=_make_approval_callback(ctx))
    if agent is None:
        ds.warning("No provider configured.")
        ds.info("Examples:")
        ctx.console.print("  huginn chat --provider openai --model gpt-4o")
        ctx.console.print(
            "  huginn chat --provider ollama --ollama-url http://localhost:11434"
        )
        ctx.console.print(
            "  huginn chat --provider vllm --base-url http://localhost:8000/v1 --model llama-3-8b"
        )
        ctx.console.print("  huginn chat --provider moonshot --model moonshot-v1-8k")
        ctx.console.print("  huginn chat --provider dashscope --model qwen-max")
        ctx.console.print("  huginn chat --config huginn.toml")
    else:
        agent.register_tools_from_registry()
        ds.success(f"Agent initialized with {len(agent.langchain_tools)} tools")

    # 启动时检测缺失的 API key, 提示用户跑 huginn api-keys 配置
    _warn_missing_api_keys(ctx)

    # Initialize MCP servers
    try:
        cfg = ctx.load_config()
        asyncio.run(init_mcp(cfg.abaqus_mcp_server))
        if agent:
            agent.register_tools_from_registry()
            ds.success(f"Total tools: {len(agent.langchain_tools)}")
    except Exception as e:
        ds.warning(f"MCP init skipped: {e}")

    # --prompt headless 模式: 跑一次就退出, 不进交互循环
    if ctx.prompt_text:
        _run_one_turn(ctx, agent, ctx.prompt_text, ds)
        with contextlib.suppress(Exception):
            asyncio.run(shutdown_mcp())
        return

    # Chat loop
    try:
        while True:
            try:
                user_input = ctx.console.input("[bold green]You:[/bold green] ")
            except (EOFError, KeyboardInterrupt):
                ds.warning("Goodbye!")
                break

            if user_input.lower() in ("exit", "quit", "q"):
                ds.warning("Goodbye!")
                break

            if not user_input.strip():
                continue

            # slash 命令: / 开头的不送给 agent, 本地处理
            if user_input.startswith("/"):
                handle_slash_command(
                    user_input,
                    agent=agent,
                    ctx=ctx,
                    console=ctx.console,
                )
                continue

            # 没配 agent 就别往下走了, 直接提示
            if agent is None:
                ds.error("Agent not available. Please check configuration.")
                continue

            _run_one_turn(ctx, agent, user_input, ds)
    finally:
        with contextlib.suppress(Exception):
            asyncio.run(shutdown_mcp())


def _run_one_turn(
    ctx: CliContext,
    agent: Any | None,
    user_input: str,
    ds: Any,
) -> None:
    """跑一轮对话: 解析 @ 引用 → 调 agent → 渲染结果。

    单独抽出来是为了让交互循环和 --prompt headless 模式复用同一段逻辑。
    """
    if agent is None:
        ds.error("Agent not available. Please check configuration.")
        return

    # @file / @url 引用: 文本里有 @ 才解析, 避免每次都跑正则
    message = user_input
    if has_at_references(user_input):
        try:
            message = parse_at_references(user_input, workspace=str(ctx.workspace))
        except Exception as e:
            # 解析失败不阻塞对话, 用原始输入继续
            ds.warning(f"@ 引用解析失败 ({e}), 用原始输入继续")

    ds.info("Thinking...")
    try:
        result = agent.invoke(message)

        messages = result.get("messages", [])

        # 工具产生的文件改动以 diff 形式预览, 方便用户审阅
        diff_text = _extract_diff_from_messages(messages)
        if diff_text:
            ds.info("文件改动预览:")
            ds.render_diff(diff_text)

        if messages:
            last_msg = messages[-1]
            content = (
                last_msg.content
                if hasattr(last_msg, "content")
                else str(last_msg)
            )
            ds.render_message("assistant", content)
        else:
            ds.info(str(result))

    except Exception as e:
        ds.error(str(e))
