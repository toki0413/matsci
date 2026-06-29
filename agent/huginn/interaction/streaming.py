"""流式反馈拦截器 —— 把 LLM token / 工具调用 / 思考过程实时透给前端。

设计: StreamInterceptor 是一个事件收集器, agent loop 通过 on_* 回调
把事件塞进来; to_event_stream() 把累积的事件转成 SSE 异步生成器, 让
FastAPI StreamingResponse 直接吐给前端。

支持两类特殊标签: <thought> ... </thought> 表示内部思考, <plan> ... </plan>
表示当前步骤计划. LLM 输出里只要带这两个标签, 就会被拆出来单独推送,
不会混到普通 token 流里。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

# 抓 <thought>...</thought> 和 <plan>...</plan> 块.
# 用非贪婪匹配 + DOTALL 让多行内容也能整体抠出来.
_THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL | re.IGNORECASE)
_PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL | re.IGNORECASE)


@dataclass
class StreamEvent:
    """一条流式事件. type 决定前端怎么渲染."""

    type: str  # token | tool_start | tool_end | thought | plan | done | error
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        """序列化成 SSE 一行: `event: <type>\ndata: <json>\n\n`."""
        payload = {"type": self.type, "ts": self.timestamp, **self.data}
        return f"event: {self.type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


class StreamInterceptor:
    """收集 agent 思考过程事件, 暴露成 SSE 流.

    用法 (agent loop 侧):
        interceptor = StreamInterceptor(thread_id="t1")
        # LLM 流式输出 token 时:
        interceptor.on_llm_token(tok)
        # 工具调用前后:
        interceptor.on_tool_start("vasp_tool", {"structure": "Si.cif"})
        interceptor.on_tool_end("vasp_tool", {"energy": -0.5}, 12.3)

    用法 (路由侧):
        async def stream():
            async for sse in interceptor.to_event_stream():
                yield sse
        return StreamingResponse(stream(), media_type="text/event-stream")

    token 缓冲: LLM 输出里可能夹着 <thought>/<plan> 标签, 直接按字符
    喂给前端会把标签拆碎. 这里先攒一段 buffer, 检测到完整标签后整块
    发 thought/plan 事件, 没命中标签的纯文本再按 token 发出去.
    """

    def __init__(self, thread_id: str = "default", buffer_size: int = 8192):
        self.thread_id = thread_id
        # 已生成但还没被消费的事件队列. SSE 消费端按 FIFO 读.
        self._events: deque[StreamEvent] = deque()
        # 标记流是否结束 (done/error). 关闭后不再接收新事件.
        self._closed = False
        # 唤醒等待中的 SSE 消费者: 新事件来了或者流关闭了都触发.
        self._signal = asyncio.Event()
        # token 缓冲: 攒够一段再扫标签, 避免逐字符正则性能炸.
        self._token_buf = ""
        self._buffer_cap = buffer_size
        # 当前是否处在 <thought>/<plan> 块内部, 用来判断要不要走标签路径.
        self._in_thought = False
        self._in_plan = False

    # ── 公共回调 ────────────────────────────────────────────────

    def on_llm_token(self, token: str) -> None:
        """LLM 流式输出一个 token. 内部会拆 thought/plan 标签."""
        if self._closed or not token:
            return
        self._token_buf += token
        # 缓冲太长也没必要等标签, 直接吐出来避免前端等太久
        if len(self._token_buf) > self._buffer_cap:
            self._flush_token_buf()
            return
        # 命中结束标签才拆, 起始标签留到 flush 时再判断, 简化状态机
        if "</thought>" in self._token_buf.lower():
            self._extract_tagged(_THOUGHT_RE, "thought")
        elif "</plan>" in self._token_buf.lower():
            self._extract_tagged(_PLAN_RE, "plan")
        # 没命中标签也不立刻发, 等下一个 token 或者显式 flush

    def on_tool_start(self, tool_name: str, tool_input: dict[str, Any]) -> None:
        """工具开始执行. tool_input 可能含敏感字段, 调用方自己脱敏."""
        # 先把残留 token 吐掉, 保证前端先看到完整文本再看工具卡片
        self._flush_token_buf()
        self._push(StreamEvent(
            type="tool_start",
            data={
                "tool": tool_name,
                "input": _safe_serialize(tool_input),
            },
        ))

    def on_tool_end(
        self, tool_name: str, result: Any, dt: float
    ) -> None:
        """工具结束. dt 是耗时(秒)."""
        self._push(StreamEvent(
            type="tool_end",
            data={
                "tool": tool_name,
                "result": _safe_serialize(result),
                "dt_seconds": round(float(dt), 3),
            },
        ))

    def on_thought(self, thought: str) -> None:
        """显式推送一段思考. 走这条路径就不必再走 token 缓冲."""
        thought = (thought or "").strip()
        if not thought:
            return
        self._push(StreamEvent(type="thought", data={"text": thought}))

    def on_plan(self, plan: str) -> None:
        """显式推送一段计划."""
        plan = (plan or "").strip()
        if not plan:
            return
        self._push(StreamEvent(type="plan", data={"text": plan}))

    def on_done(self, final_text: str | None = None) -> None:
        """流式结束. 通知 SSE 消费端关闭连接."""
        self._flush_token_buf()
        data = {} if final_text is None else {"final_text": final_text}
        self._push(StreamEvent(type="done", data=data))
        self._closed = True

    def on_error(self, message: str) -> None:
        """异常路径. 推一条 error 事件后关闭."""
        self._flush_token_buf()
        self._push(StreamEvent(type="error", data={"message": message}))
        self._closed = True

    # ── SSE 输出 ───────────────────────────────────────────────

    async def to_event_stream(self) -> AsyncGenerator[str, None]:
        """转成 SSE 字符串异步生成器.

        一直吐事件直到 on_done / on_error 被调用且队列清空.
        队列空但流没关时, 挂在 _signal 上等新事件, 不空轮询.
        """
        while True:
            # 把当前队列里的事件全吐干净
            while self._events:
                yield self._events.popleft().to_sse()
            if self._closed:
                # 流已关闭, 最后再扫一眼队列避免丢尾巴
                while self._events:
                    yield self._events.popleft().to_sse()
                return
            # 队列空 + 流没关: 等新事件唤醒
            self._signal.clear()
            await self._signal.wait()

    # ── 内部工具 ───────────────────────────────────────────────

    def _push(self, event: StreamEvent) -> None:
        """压一条事件并唤醒 SSE 消费端."""
        self._events.append(event)
        self._signal.set()

    def _flush_token_buf(self) -> None:
        """把缓冲里的纯 token 一次性发出去."""
        if not self._token_buf:
            return
        # 发之前最后扫一次标签, 避免漏掉刚攒满还没拆的
        if _THOUGHT_RE.search(self._token_buf):
            self._extract_tagged(_THOUGHT_RE, "thought")
        if _PLAN_RE.search(self._token_buf):
            self._extract_tagged(_PLAN_RE, "plan")
        if self._token_buf:
            self._push(StreamEvent(
                type="token",
                data={"text": self._token_buf},
            ))
            self._token_buf = ""

    def _extract_tagged(self, pattern: re.Pattern, label: str) -> None:
        """从缓冲里抠出 <label>...</label> 块, 推一条事件, 剩余文本留在缓冲.

        被抠掉的部分不会出现在 token 流里, 这样前端不会看到原始标签.
        """
        matches = list(pattern.finditer(self._token_buf))
        if not matches:
            return
        # 重组: 标签前的文本 + 标签内文本(发事件) + 标签后的文本
        rebuilt = ""
        cursor = 0
        for m in matches:
            before = self._token_buf[cursor:m.start()]
            rebuilt += before
            inner = m.group(1).strip()
            if inner:
                self._push(StreamEvent(type=label, data={"text": inner}))
            cursor = m.end()
        rebuilt += self._token_buf[cursor:]
        self._token_buf = rebuilt


def _safe_serialize(obj: Any, max_len: int = 2000) -> Any:
    """把任意对象转成 JSON 安全的结构, 顺手截断超长字符串.

    工具的 result 可能是 numpy 数组 / 大 dict / 自定义对象, 直接
    json.dumps 会炸. 这里先做一轮 best-effort 转换, 超长的字符串
    就截断, 避免单个事件把 SSE 队列撑爆.
    """
    try:
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            if isinstance(obj, str) and len(obj) > max_len:
                return obj[:max_len] + f"...<+{len(obj) - max_len} chars>"
            return obj
        if isinstance(obj, dict):
            return {str(k): _safe_serialize(v, max_len) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_safe_serialize(v, max_len) for v in list(obj)[:50]]
        # 兜底: 走 str, 失败就给个占位
        text = str(obj)
        if len(text) > max_len:
            text = text[:max_len] + f"...<+{len(text) - max_len} chars>"
        return text
    except Exception:
        return "<unserializable>"
