"""互动层 HTTP 路由 —— 实时反馈 / 中途干预 / 主动提问 / 进度展示.

四个维度的端点都集中在这一个文件里, 方便前端和路由表管理:
- 实时反馈: POST /agents/{id}/chat/stream  (SSE 流式 chat)
- 中途干预: POST /agents/{id}/interrupt  +  GET /agents/{id}/interrupt/status
- 主动提问: GET /clarifications  +  POST /clarifications/{qid}/resolve
- 进度展示: GET /tasks  +  GET /tasks/{id}  +  GET /tasks/stream (SSE)

SSE 选型: 项目已有 WS, 但 SSE 更简单 (单向推送 + 自动重连), 这里
的流式场景都是 server → client 单向, 没必要走 WS 双向.
"""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from huginn.interaction.clarification import get_clarification_manager
from huginn.interaction.interrupt import (
    InterruptCancelled,
    InterruptEvent,
    get_interrupt_manager,
)
from huginn.interaction.progress import get_progress_tracker
from huginn.interaction.streaming import StreamInterceptor
from huginn.routes.schemas import ChatRequest
from huginn.server_core import get_agent_factory

router = APIRouter(tags=["interaction"])

logger = logging.getLogger(__name__)


# ── 实时反馈: SSE 流式 chat ──────────────────────────────────────


@router.post("/agents/{agent_id}/chat/stream")
async def chat_stream(agent_id: str, params: dict[str, Any]) -> StreamingResponse:
    """流式 chat 端点. 跟 /agents/{id}/chat 一致的入参, 但走 SSE 返回.

    body 跟 /chat 一样: {message, thread_id, timeout, ...}.
    SSE 事件类型:
    - token:        LLM 输出的 token
    - thought:      agent 内部思考 (<thought> 标签)
    - plan:         agent 当前计划 (<plan> 标签)
    - tool_start:   工具开始执行
    - tool_end:     工具结束
    - done:         流式结束 (带 final_text)
    - error:        异常
    - interrupt:    被用户中断 (带 reason)
    """
    # Validate the request body before spinning up the SSE stream.
    try:
        req = ChatRequest.model_validate(params)
    except ValidationError as exc:
        return JSONResponse(
            {"error": f"Invalid request: {exc.errors()}"}, status_code=422
        )

    message = req.content
    thread_id = req.thread_id
    timeout = float(params.get("timeout", 300))

    interceptor = StreamInterceptor(thread_id=thread_id)
    interrupt_mgr = get_interrupt_manager()
    # 进来就清掉历史残留干预, 避免上一轮的 pause 还挂着
    interrupt_mgr.clear_interrupt(thread_id)

    async def _run_agent() -> None:
        """后台跑 agent, 把事件喂给 interceptor."""
        try:
            if message:
                # 走跟 /chat 一样的 factory 路径, 拿到 agent 后调 chat()
                factory = get_agent_factory()
                agent = factory.create(
                    agent_id,
                    thread_id=thread_id,
                    thinking=req.thinking,
                    max_tokens=req.max_tokens,
                )
                # sidecar serve 模式自动批准工具, 跟 /chat 行为一致
                agent._permission_config.auto_approve_all = True

                final_text = ""
                try:
                    # agent.chat 是 async generator, 每轮 yield 一个 state
                    # 把 state 里的 AIMessage 内容喂给 interceptor.on_llm_token
                    async for state in agent.chat(message, thread_id=thread_id):
                        # 命中中断就停
                        evt = interrupt_mgr.check_interrupt(thread_id)
                        if evt is not None:
                            if evt.type == "cancel":
                                interceptor.on_error(
                                    f"cancelled by user: {evt.message}"
                                )
                                return
                            if evt.type == "modify":
                                # modify: 把用户的新消息当 token 喂回去,
                                # 让前端看到 agent 收到了修改意见
                                interceptor.on_thought(
                                    f"[用户修改] {evt.message}"
                                )
                        # 暂停态: 阻塞到 resume
                        await interrupt_mgr.wait_if_paused(thread_id)

                        msgs = state.get("messages", []) if isinstance(state, dict) else []
                        for msg in msgs:
                            # AIMessage 的 content 当 token 流推
                            content = getattr(msg, "content", "")
                            msg_type = getattr(msg, "type", "")
                            if msg_type == "ai" and isinstance(content, str) and content:
                                interceptor.on_llm_token(content)
                                final_text = content
                            # ToolMessage 当 tool_end 推 (简化: 不拦截 start)
                            elif msg_type == "tool":
                                name = getattr(msg, "name", "unknown")
                                interceptor.on_tool_end(name, content, 0.0)
                    interceptor.on_done(final_text)
                except InterruptCancelled:
                    interceptor.on_error("cancelled by user")
                except Exception as exc:
                    interceptor.on_error(str(exc))
            else:
                interceptor.on_error("empty message")
        except Exception as exc:
            logger.error("unexpected error", exc_info=True)
            interceptor.on_error(f"server error: {exc}")

    # 后台任务: 跑 agent, 不阻塞 SSE 流的产出
    task = asyncio.create_task(_run_agent())

    async def _sse() -> Any:
        """SSE 生成器: 把 interceptor 的事件流吐出去."""
        try:
            async for chunk in interceptor.to_event_stream():
                yield chunk
        finally:
            # SSE 客户端断开时把后台任务也取消, 避免 agent 白跑
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    return StreamingResponse(
        _sse(),
        media_type="text/event-stream",
        headers={
            # 关闭 nginx 的 buffer, 否则 SSE 会被攒一批再发, 延迟很大
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ── 中途干预 ─────────────────────────────────────────────────────


@router.post("/agents/{agent_id}/interrupt")
async def submit_interrupt(agent_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """提交一次干预. body: {type, message, thread_id}.

    type: pause | resume | cancel | modify
    message: 用户给的修改意见 / 取消原因 (可选)
    thread_id: 必填, 要干预哪个会话
    """
    thread_id = params.get("thread_id") or params.get("id") or "default"
    evt_type = str(params.get("type", "")).lower()
    message = str(params.get("message", ""))

    try:
        evt = InterruptEvent(type=evt_type, message=message)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    mgr = get_interrupt_manager()
    mgr.submit_interrupt(thread_id, evt)
    return {
        "success": True,
        "agent_id": agent_id,
        "thread_id": thread_id,
        "type": evt.type,
    }


@router.get("/agents/{agent_id}/interrupt/status")
async def interrupt_status(agent_id: str, thread_id: str = "default") -> dict[str, Any]:
    """查干预状态. 用 query param 传 thread_id: ?thread_id=xxx."""
    mgr = get_interrupt_manager()
    status = mgr.status(thread_id)
    status["agent_id"] = agent_id
    return status


# ── 主动提问 ─────────────────────────────────────────────────────


@router.get("/clarifications")
async def list_clarifications(thread_id: str | None = None) -> dict[str, Any]:
    """列出待回答的提问. ?thread_id=xxx 只看某个会话的."""
    mgr = get_clarification_manager()
    if thread_id:
        questions = mgr.list_pending(thread_id)
    else:
        questions = mgr.list_all_pending()
    return {"success": True, "count": len(questions), "questions": questions}


@router.post("/clarifications/{question_id}/resolve")
async def resolve_clarification(
    question_id: str, params: dict[str, Any]
) -> dict[str, Any]:
    """回答一个提问. body: {answer}."""
    answer = str(params.get("answer", ""))
    if not answer:
        return {"success": False, "error": "answer is required"}
    mgr = get_clarification_manager()
    ok = mgr.resolve(question_id, answer)
    return {"success": ok, "question_id": question_id}


@router.post("/agents/{agent_id}/clarifications/resolve")
async def resolve_clarification_by_thread(
    agent_id: str, params: dict[str, Any]
) -> dict[str, Any]:
    """按 thread_id 回答最早的未答提问. body: {answer, thread_id}.

    前端不知道 question_id 时用这个端点 (按 thread_id 路由到最早的问题).
    """
    answer = str(params.get("answer", ""))
    thread_id = params.get("thread_id") or "default"
    if not answer:
        return {"success": False, "error": "answer is required"}
    mgr = get_clarification_manager()
    ok = mgr.resolve_thread(thread_id, answer)
    return {"success": ok, "agent_id": agent_id, "thread_id": thread_id}


# ── 进度展示 ─────────────────────────────────────────────────────


@router.get("/tasks")
async def list_tasks(active_only: bool = False) -> dict[str, Any]:
    """列出所有任务进度. ?active_only=true 只看未完成的."""
    tracker = get_progress_tracker()
    if active_only:
        tasks = tracker.list_active()
    else:
        tasks = tracker.list_all()
    return {"success": True, "count": len(tasks), "tasks": tasks}


@router.get("/tasks/{task_id}")
async def get_task(task_id: str) -> dict[str, Any]:
    """拿单个任务的进度."""
    tracker = get_progress_tracker()
    status = tracker.get_status(task_id)
    if status is None:
        return {"success": False, "error": f"task '{task_id}' not found"}
    return {"success": True, "task": status}


@router.get("/tasks/stream")
async def tasks_stream() -> StreamingResponse:
    """SSE 流式进度更新. 每次有任务 update 都推一条.

    事件类型:
    - snapshot:  连接时全量快照 (每个任务一条)
    - update:    任务状态更新
    - heartbeat: 心跳 (15s 一次, 保持连接)
    """
    tracker = get_progress_tracker()

    async def _sse() -> Any:
        async for chunk in tracker.to_event_stream():
            yield chunk

    return StreamingResponse(
        _sse(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str) -> dict[str, Any]:
    """取消一个任务 (标记状态, 不真的杀进程 —— 进程级取消由 InterruptManager 管)."""
    tracker = get_progress_tracker()
    result = tracker.cancel(task_id)
    if result is None:
        return {"success": False, "error": f"task '{task_id}' not found"}
    return {"success": True, "task": result.to_dict()}
