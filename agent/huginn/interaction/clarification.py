"""主动提问管理 —— agent 不确定时主动问用户, 而不是瞎猜.

跟 clarify_questions_hook 的区别: 那个 hook 在 USER_PROMPT_SUBMIT 阶段
做规则匹配, 命中关键词才追问; 这个 ClarificationManager 是给 agent
自己用的 —— LLM 在执行过程中发现信息不足, 调 clarification_tool 主动
提问, 阻塞等待用户回答.

阻塞怎么实现: ask() 内部挂一个 asyncio.Future, HTTP 路由调 resolve()
时 set_result, ask 返回. 超时 (默认 5 分钟) 自动返回默认值, 避免
agent 永远卡死. 没有运行中的事件循环时 (比如同步调用) 直接返回超时
占位, 不阻塞.

触发条件由 LLM 自己判断, prompts.py 里给了引导:
- 任务描述模糊 ("算一下" 没说算什么)
- 参数不明确 (ENCUT 没说取多少)
- 多个可能路径 (DFT 还是 ML 势)
- 长任务前确认 (VASP 预计 2h)
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ClarificationQuestion:
    """一个待回答的问题."""

    question_id: str
    thread_id: str
    question: str
    options: list[str] = field(default_factory=list)
    context: str = ""
    # 默认值: 超时或用户跳过时用它
    default_answer: str = ""
    created_at: float = field(default_factory=time.time)
    # 回答相关
    answer: Optional[str] = None
    answered_at: Optional[float] = None
    # 提问时的上下文快照, 给前端展示用
    metadata: dict[str, Any] = field(default_factory=dict)


class ClarificationManager:
    """管理 agent 主动提问的全生命周期.

    用法 (agent 工具内):
        mgr = get_clarification_manager()
        if mgr.should_ask("param_ambiguous", context):
            answer = await mgr.ask(
                thread_id="t1",
                question="ENCUT 取 400 还是 520?",
                options=["400", "520", "auto"],
                default_answer="520",
            )
            # answer 是用户回答的字符串, 或者超时退回的 default_answer

    用法 (HTTP 路由, 用户回答时):
        mgr.resolve(question_id, user_answer)
        # 或者按 thread_id 解 (前端只知道 thread_id 时):
        mgr.resolve_thread(thread_id, user_answer)
    """

    def __init__(self, default_timeout: float = 300.0) -> None:
        # 默认 5 分钟超时, 跟任务要求一致
        self.default_timeout = default_timeout
        # question_id -> ClarificationQuestion (含 Future)
        self._questions: dict[str, ClarificationQuestion] = {}
        # thread_id -> 待回答问题列表 (一个 thread 可能连续问多个)
        self._by_thread: dict[str, list[str]] = {}
        # Future 池: question_id -> asyncio.Future. 单独存避免 dataclass
        # 里塞 Future 走 dataclass 默认序列化时出问题.
        self._futures: dict[str, asyncio.Future] = {}
        self._lock = threading.Lock()

    # ── 判断要不要问 ───────────────────────────────────────────

    def should_ask(
        self, question_type: str, context: dict[str, Any] | None = None
    ) -> bool:
        """判断是否需要问. 简单启发式, 真正的判断由 LLM 在 prompt 引导下做.

        这里只做几个硬规则, 避免 agent 在不该问的时候瞎问:
        - 同一 thread 已经有 3 个未回答的问题: 别再问了, 用默认值继续.
        - question_type 不在已知列表里: 放行, 让 LLM 自由发挥.

        已知 question_type: task_vague / param_ambiguous / multi_path /
        long_task_confirm / scope_unclear.
        """
        ctx = context or {}
        thread_id = ctx.get("thread_id")
        if thread_id:
            with self._lock:
                pending = len(self._by_thread.get(thread_id, []))
            if pending >= 3:
                # 已经攒了 3 个没回答的, 再问也是堆着, 不如用默认值
                return False
        return True

    # ── agent 侧: 提问并等回答 ─────────────────────────────────

    async def ask(
        self,
        thread_id: str,
        question: str,
        options: list[str] | None = None,
        context: str = "",
        default_answer: str = "",
        timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """提一个问题, 阻塞等用户回答. 超时返回 default_answer.

        Returns:
            用户回答的字符串. 如果用户跳过 / 超时, 返回 default_answer.
        """
        qid = f"q_{uuid.uuid4().hex[:10]}"
        timeout_val = self.default_timeout if timeout is None else float(timeout)
        q = ClarificationQuestion(
            question_id=qid,
            thread_id=thread_id,
            question=question,
            options=list(options) if options else [],
            context=context,
            default_answer=default_answer or "",
            metadata=metadata or {},
        )
        # 拿当前 loop 的 Future. 没有运行中的 loop 就直接退化为超时返回,
        # 不强行 asyncio.run —— 那会创建新 loop 跟 agent loop 打架.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 同步上下文里调, 没法阻塞等. 直接走默认值, 但还是把问题
            # 登记进 _questions, 让前端能看到 agent 想问什么.
            with self._lock:
                self._questions[qid] = q
                self._by_thread.setdefault(thread_id, []).append(qid)
            q.answer = default_answer or ""
            q.answered_at = time.time()
            return q.answer

        fut: asyncio.Future = loop.create_future()
        with self._lock:
            self._questions[qid] = q
            self._by_thread.setdefault(thread_id, []).append(qid)
            self._futures[qid] = fut

        try:
            # 挂着等回答, 超时退回 default
            answer = await asyncio.wait_for(fut, timeout=timeout_val)
            return str(answer)
        except asyncio.TimeoutError:
            q.answer = default_answer or ""
            q.answered_at = time.time()
            return q.answer
        except asyncio.CancelledError:
            # agent loop 被取消时跟着退, Future 自动清理
            with self._lock:
                self._futures.pop(qid, None)
            raise
        finally:
            # 不管成功还是超时, 都把 Future 清掉, 防止内存泄漏
            with self._lock:
                self._futures.pop(qid, None)

    # ── 用户侧: 回答 ───────────────────────────────────────────

    def resolve(self, question_id: str, answer: str) -> bool:
        """按 question_id 回答. 返回 True 表示命中且已回答."""
        with self._lock:
            q = self._questions.get(question_id)
            fut = self._futures.get(question_id)
            if q is None or q.answer is not None:
                # 问题不存在或已被回答 (超时也算已回答)
                return False
            q.answer = answer
            q.answered_at = time.time()
            # 从 thread 待答列表里移除
            pending = self._by_thread.get(q.thread_id, [])
            if question_id in pending:
                pending.remove(question_id)
        # set_result 必须在锁外调, 避免在锁里 await 唤醒的协程
        if fut is not None and not fut.done():
            try:
                fut.set_result(answer)
            except asyncio.InvalidStateError:
                # 已经被 wait_for 超时 set 过了, 忽略
                pass
        return True

    def resolve_thread(self, thread_id: str, answer: str) -> bool:
        """按 thread_id 回答最早的未答问题. 前端不知道 qid 时用这个."""
        with self._lock:
            pending = self._by_thread.get(thread_id, [])
            if not pending:
                return False
            qid = pending[0]
        return self.resolve(qid, answer)

    # ── 查询 ───────────────────────────────────────────────────

    def list_pending(self, thread_id: str) -> list[dict[str, Any]]:
        """列出某 thread 下所有未答问题. 给前端轮询/SSE 用."""
        with self._lock:
            qids = list(self._by_thread.get(thread_id, []))
            result = []
            for qid in qids:
                q = self._questions.get(qid)
                if q and q.answer is None:
                    result.append(self._question_to_dict(q))
        return result

    def list_all_pending(self) -> list[dict[str, Any]]:
        """列出所有 thread 的未答问题. 给 /clarifications 路由用."""
        with self._lock:
            thread_ids = list(self._by_thread.keys())
        out: list[dict[str, Any]] = []
        for tid in thread_ids:
            out.extend(self.list_pending(tid))
        return out

    @staticmethod
    def _question_to_dict(q: ClarificationQuestion) -> dict[str, Any]:
        return {
            "question_id": q.question_id,
            "thread_id": q.thread_id,
            "question": q.question,
            "options": q.options,
            "context": q.context,
            "created_at": q.created_at,
            "metadata": q.metadata,
        }


# ── 进程级单例 ──────────────────────────────────────────────
#
# 跟 InterruptManager 一样的思路: 路由层和 agent loop 共享同一份
# 问题队列. 多 worker 各自一份, 同一 thread 的提问和回答必须落在
# 同一 worker 上才有效 —— 这跟 chat 请求的 thread_id 路由一致.

_singleton: ClarificationManager | None = None
_singleton_lock = threading.Lock()


def get_clarification_manager() -> ClarificationManager:
    """拿进程级 ClarificationManager 单例."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = ClarificationManager()
    return _singleton
