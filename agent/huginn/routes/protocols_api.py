"""对外协议适配端点 —— OpenAI Responses 兼容面 (缺口① P1-4).

三段式 handler (对齐 deepseek-recipe ``server-py/main.py``):

1. **prepare**: 解析请求体 → ``ResponsesRequest.convert()`` → 共享 IR ``ChatRequest``;
2. **run**: 调 huginn agent (流式从 agent 产出 token 流, 非流式取终态);
3. **respond**: token 流喂 ``OutputParser`` → 有序 ``OutputChunk`` 流 →
   ``ResponsesChunkGenerator.generate()`` → SSE / 终态 JSON.

不新增传输层、不引第三方 SDK: 复用既有 SSE 与 auth 中间件. 注册进
``routes.ALL_ROUTERS`` 后由 ``include_v1_routes`` 落到 ``POST /v1/responses``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import queue
from collections.abc import AsyncIterator, Iterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from huginn.protocols.ir import ChatRequest, Role
from huginn.protocols.openai_responses import ResponsesRequest
from huginn.protocols.stream import OutputChunk, ProtocolEvent, Token
from huginn.server_core import get_agent_factory

router = APIRouter(tags=["protocols"])

logger = logging.getLogger(__name__)

_SSE_HEADERS = {
    # 关掉 nginx buffer, 否则 SSE 会被攒一批再发 (延迟很大).
    "X-Accel-Buffering": "no",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}

_ROLE_LABEL: dict[Role, str] = {
    Role.SYSTEM: "system",
    Role.USER: "user",
    Role.ASSISTANT: "assistant",
    Role.TOOL: "tool",
}


@router.post("/responses")
async def responses(params: dict[str, Any]) -> Any:
    """``POST /v1/responses`` —— Responses 协议入口.

    请求方向经 ``ResponsesRequest.convert()`` 归一进共享 IR; 响应方向经
    ``OutputParser`` + ``ResponsesChunkGenerator`` 映射回 Responses 事件.
    """
    # ── prepare: 协议请求 -> 共享 IR ──────────────────────────────
    try:
        request = ResponsesRequest(params)
        ir = request.convert()
    except ValueError as exc:
        return _error(str(exc), status_code=400)

    agent_id = str(params.get("agent_id") or "default")
    thread_id = str(params.get("thread_id") or params.get("user") or "default")

    # ── run + respond ─────────────────────────────────────────────
    if not ir.stream:
        try:
            return JSONResponse(await _collect_final(request, ir, agent_id, thread_id))
        except Exception as exc:  # 防御: agent 侧异常收敛为协议错误, 不裸抛 500
            logger.error("responses 非流式路径失败", exc_info=True)
            return _error(f"agent error: {exc}", status_code=500)

    return StreamingResponse(
        _sse_stream(request, ir, agent_id, thread_id),
        media_type="text/event-stream",
        headers=dict(_SSE_HEADERS),
    )


# ── respond: 非流式 (取终态 response.completed) ──────────────────────


async def _collect_final(
    request: ResponsesRequest,
    ir: ChatRequest,
    agent_id: str,
    thread_id: str,
) -> dict:
    """跑完 agent, 把整段输出过一次解析+映射, 取终态事件 data."""
    parser = request.output_parser()
    chunks: list[OutputChunk] = []
    async for token in _agent_tokens(ir, agent_id, thread_id):
        chunks.extend(parser.feed(token))
    chunks.extend(parser.finish())
    final: ProtocolEvent | None = None
    for event in request.chunk_generator().generate(chunks):
        final = event
    return final.data if final is not None else {}


# ── respond: 流式 (SSE) ─────────────────────────────────────────────


async def _sse_stream(
    request: ResponsesRequest,
    ir: ChatRequest,
    agent_id: str,
    thread_id: str,
) -> AsyncIterator[str]:
    """流式: agent token 流 -> parser -> chunk 流 -> 协议事件 -> SSE.

    ``ChunkGenerator.generate()`` 是同步生成器 (吃 ``Iterable[OutputChunk]``),
    agent 是异步 token 流 —— 用线程桥接: agent 协程把 token 塞进线程安全队列,
    工作线程阻塞式 ``next()`` 拉取并同步推进解析+映射, 事件经 ``call_soon_threadsafe``
    回到事件循环. 这样 chunk 到达即出事件, 不是攒完再发.
    """
    loop = asyncio.get_running_loop()
    token_q: queue.Queue[Token | None] = queue.Queue()
    sse_q: asyncio.Queue[ProtocolEvent | None] = asyncio.Queue()
    producer = asyncio.create_task(_produce_tokens(ir, agent_id, thread_id, token_q))

    def _chunks() -> Iterator[OutputChunk]:
        parser = request.output_parser()
        while True:
            token = token_q.get()
            if token is None:
                break
            yield from parser.feed(token)
        yield from parser.finish()

    def _drive() -> None:
        try:
            for event in request.chunk_generator().generate(_chunks()):
                loop.call_soon_threadsafe(sse_q.put_nowait, event)
        except Exception as exc:  # 防御: 桥接线程异常也须让消费者收到终止信号
            logger.error("responses 流式桥接失败", exc_info=True)
            loop.call_soon_threadsafe(
                sse_q.put_nowait,
                ProtocolEvent("error", {"type": "error", "message": str(exc)}),
            )
        finally:
            loop.call_soon_threadsafe(sse_q.put_nowait, None)

    driver = loop.run_in_executor(None, _drive)
    try:
        while True:
            event = await sse_q.get()
            if event is None:
                break
            yield _sse_line(event)
    finally:
        # 客户端断开: 取消 agent 协程, 并放终止哨兵解开线程队列的阻塞.
        if not producer.done():
            producer.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await producer
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await driver


async def _produce_tokens(
    ir: ChatRequest,
    agent_id: str,
    thread_id: str,
    token_q: queue.Queue[Token | None],
) -> None:
    """跑 agent, 把每个 AI 文本段作为普通 token 塞进桥接队列."""
    try:
        async for token in _agent_tokens(ir, agent_id, thread_id):
            token_q.put(token)
    finally:
        token_q.put(None)


async def _agent_tokens(
    ir: ChatRequest, agent_id: str, thread_id: str
) -> AsyncIterator[Token]:
    """驱动 huginn agent 并把 AI 输出转成 ``Token`` (普通 token, 不触发标记)."""
    factory = get_agent_factory()
    agent = factory.create(agent_id, thread_id=thread_id, max_tokens=ir.max_tokens)
    async for state in agent.chat(_render_prompt(ir), thread_id=thread_id):
        messages = state.get("messages", []) if isinstance(state, dict) else []
        for message in messages:
            if getattr(message, "type", "") != "ai":
                continue
            content = getattr(message, "content", "")
            if isinstance(content, str) and content:
                yield Token(content, is_special=False)


# ── helpers ─────────────────────────────────────────────────────────


def _render_prompt(ir: ChatRequest) -> str:
    """把共享 IR 折成一行给 agent 的输入串.

    huginn agent 是带自身会话记忆的对话体, 只接单条 message —— 这里做**协议无关**
    的极简折行 (非 DeepSeek 的 prompt 渲染): 单条 user 文本直送原文, 其余按角色
    前缀拼成 transcript.
    """
    if len(ir.messages) == 1 and ir.messages[0].role is Role.USER:
        parts = ir.messages[0].parts
        if len(parts) <= 1:
            return parts[0].text if parts else ""
    lines: list[str] = []
    for message in ir.messages:
        text = " ".join(part.text for part in message.parts if part.text)
        if message.tool_call_id:
            text = f"[tool:{message.tool_call_id}] {text}"
        label = _ROLE_LABEL.get(message.role, str(message.role))
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


def _sse_line(event: ProtocolEvent) -> str:
    """Responses SSE: 事件类型内嵌在 JSON, 故只发 ``data:`` 行."""
    return f"data: {json.dumps(event.data, ensure_ascii=False)}\n\n"


def _error(message: str, *, status_code: int) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "invalid_request_error"}},
        status_code=status_code,
    )
