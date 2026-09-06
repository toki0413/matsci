"""Interactive chat command."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from typing import Any

import click
from langchain_core.messages import AIMessage, ToolMessage
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

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


def _ai_message_view(msg: Any) -> tuple[str, str, list[str]]:
    """拆解 AIMessage 给渲染层用。

    返回 (kind, text, tool_names):
    - kind='thought': 带工具调用的推理, content 一般是 LLM 的思考/计划
    - kind='answer': 纯文本回复, 作为最终答复正文
    text 取 content 的文本部分; 多模态 content blocks 会被拍平成纯文本。
    """
    tcs = getattr(msg, "tool_calls", None) or []
    content = msg.content
    if isinstance(content, list):
        # 多模态 content: 拼出 text blocks, 非 text 的(图片等)直接丢掉
        content = "".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    kind = "thought" if tcs else "answer"
    names = [tc.get("name", "unknown") for tc in tcs]
    return kind, content or "", names


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

    优先走流式 (agent.chat async generator); 不支持就退回 invoke 阻塞模式。
    交互循环和 --prompt headless 模式都复用这里。
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

    thread_id = ctx.resume_thread_id or "default"

    # agent.chat 是 async generator 才走流式; 否则降级到 invoke
    chat_fn = getattr(agent, "chat", None)
    if chat_fn is not None and inspect.isasyncgenfunction(chat_fn):
        try:
            _run_streaming_turn(ctx, agent, message, ds, thread_id)
            return
        except KeyboardInterrupt:
            # Ctrl+C 只中断当前这轮生成, 不退出 chat 循环
            ds.warning("[interrupted by user]")
            return
        except Exception as e:
            # 流式挂了别直接抛, 降级到阻塞模式再试一次
            ds.warning(f"流式输出异常 ({e}), 退回阻塞模式")

    _run_blocking_turn(ctx, agent, message, ds, thread_id)


def _run_streaming_turn(
    ctx: CliContext,
    agent: Any,
    message: str,
    ds: Any,
    thread_id: str,
) -> None:
    """流式渲染一轮对话。

    agent.chat() 用 stream_mode="values" 产出完整 state 快照, messages 列表是
    累积的; 用 shown 记录已处理条数, 只渲染每轮新增的消息。
    工具调用 / 思考过程走 console.print (留在 Live 区域上方), 回复正文走 Live
    边到边显示, 结束后再用 Markdown 重新渲染最终答复。
    """
    console = ctx.console

    async def _drive() -> tuple[str, list]:
        shown = 0
        assistant_parts: list[str] = []
        final_messages: list = []
        with Live(
            Text("Thinking…", style="dim italic"),
            console=console,
            refresh_per_second=12,
            transient=True,
        ) as live:
            async for state in agent.chat(message, thread_id=thread_id):
                # 单步 / 思考循环终止标记: 拆出真实 state 再继续
                if isinstance(state, dict) and (
                    state.get("tool_break") or state.get("thought_loop_terminated")
                ):
                    state = state.get("state") or {}

                if isinstance(state, dict):
                    final_messages = state.get("messages", [])

                new_msgs = final_messages[shown:]
                shown = len(final_messages)
                for msg in new_msgs:
                    if isinstance(msg, AIMessage):
                        kind, text, tool_names = _ai_message_view(msg)
                        if kind == "thought":
                            # 推理 / 计划: dim 显示, 不进 Live
                            if text.strip():
                                console.print(f"[dim italic]{text}[/dim italic]")
                            for name in tool_names:
                                console.print(
                                    f"[dim]→ 调用 [bold]{name}[/bold][/dim]"
                                )
                        elif text:
                            assistant_parts.append(text)
                            live.update(Text(text, style="magenta"))
                    elif isinstance(msg, ToolMessage):
                        name = getattr(msg, "name", None) or "tool"
                        ok = not (
                            isinstance(msg.content, str)
                            and msg.content.startswith("Error")
                        )
                        mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
                        console.print(f"{mark} {name}")

        return "".join(assistant_parts), final_messages

    final_text, final_messages = asyncio.run(_drive())

    # 工具产生的文件改动以 diff 形式预览, 方便用户审阅
    diff_text = _extract_diff_from_messages(final_messages)
    if diff_text:
        ds.info("文件改动预览:")
        ds.render_diff(diff_text)

    # 流式时只显示了纯文本, 结束后用 Markdown 重新渲染最终答复
    if final_text.strip():
        console.print()
        console.print("[bold magenta]Huginn:[/bold magenta]")
        console.print(Markdown(final_text))


def _run_blocking_turn(
    ctx: CliContext,
    agent: Any,
    message: str,
    ds: Any,
    thread_id: str,
) -> None:
    """阻塞模式: invoke 一次性拿结果。流式不可用时的降级路径。"""
    ds.info("Thinking...")
    try:
        result = agent.invoke(message, thread_id=thread_id)
        messages = result.get("messages", []) if isinstance(result, dict) else []

        # 工具产生的文件改动以 diff 形式预览
        diff_text = _extract_diff_from_messages(messages)
        if diff_text:
            ds.info("文件改动预览:")
            ds.render_diff(diff_text)

        if messages:
            last_msg = messages[-1]
            content = (
                last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            )
            if not isinstance(content, str):
                content = str(content)
            # 阻塞模式也走 Markdown 渲染, 跟流式保持一致
            ctx.console.print("[bold magenta]Huginn:[/bold magenta]")
            ctx.console.print(Markdown(content))
        else:
            ds.info(str(result))

    except KeyboardInterrupt:
        # Ctrl+C 中断当前这轮, 不退出 chat 循环
        ds.warning("[interrupted by user]")
    except Exception as e:
        ds.error(str(e))


if __name__ == "__main__":
    # 纯函数自检: AIMessage 分类逻辑, 不依赖 agent / Rich
    class _FakeMsg:
        def __init__(self, content, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []

    # 纯文本回复 → answer
    k, t, names = _ai_message_view(_FakeMsg("hello world"))
    assert k == "answer" and t == "hello world" and names == [], (k, t, names)

    # 带工具调用 → thought, 文本是推理过程
    k, t, names = _ai_message_view(
        _FakeMsg("先搜一下", tool_calls=[{"name": "search", "args": {}}])
    )
    assert k == "thought" and t == "先搜一下" and names == ["search"], (k, t, names)

    # 多模态 content blocks → 拍平成纯文本
    k, t, _ = _ai_message_view(
        _FakeMsg([{"type": "text", "text": "hi"}, {"type": "image"}])
    )
    assert k == "answer" and t == "hi", (k, t)

    print("chat.py self-check OK")
