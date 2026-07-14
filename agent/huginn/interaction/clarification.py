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
import logging
logger = logging.getLogger(__name__)



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
        # 提问统计: {question_type: {"asked": N, "answered": N, "timed_out": N}}
        # 用来调优触发阈值 — 某类问题超时率高说明问得不对, 下次少问
        self._stats: dict[str, dict[str, int]] = {}
        # 每线程上次提问时间, 给 should_ask_contextual 的 cooldown 用
        self._last_ask_time: dict[str, float] = {}

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

    def should_ask_contextual(
        self, question_type: str, context: dict[str, Any] | None = None
    ) -> bool:
        """上下文感知的触发判断, 比 should_ask 多看几个维度.

        额外考虑:
        - consecutive_failures: autoloop 连续失败 3+ 次时强制触发 (绕过 cooldown)
        - cost_estimate: 预估成本 > 阈值时强制触发 (绕过 cooldown)
        - cooldown: 同一 thread 60s 内不重复问同类问题
        - 超时率: 某类问题超时率 > 70% 时降频 (少问)
        """
        if not self.should_ask(question_type, context):
            return False

        ctx = context or {}
        thread_id = ctx.get("thread_id", "")

        # 强制触发: 连续失败 3+ 次或高成本, 绕过 cooldown 直接问
        failures = ctx.get("consecutive_failures", 0)
        if failures >= 3:
            return True

        cost_hours = ctx.get("cost_estimate_hours", 0)
        if cost_hours and cost_hours >= 1.0:
            return True

        # cooldown: 60s 内同 thread 不重复问
        with self._lock:
            last = self._last_ask_time.get(thread_id, 0)
        if time.time() - last < 60:
            return False

        # 超时率高的类型降频: 这类问题用户大概率不回答, 少问
        stats = self._stats.get(question_type, {})
        asked = stats.get("asked", 0)
        timed_out = stats.get("timed_out", 0)
        if asked >= 5 and timed_out / asked > 0.7:
            return False

        return True

    def generate_question(
        self,
        context: dict[str, Any],
        model: Any | None = None,
    ) -> tuple[str, list[str], str]:
        """根据上下文生成提问文案 + 选项 + 默认值.

        有 LLM 时用 LLM 生成上下文相关的问题; 没有时走模板.
        返回 (question, options, default_answer).

        context 里应该带:
        - question_type: task_vague / param_ambiguous / multi_path / cost_confirm
        - phase: 当前在哪个阶段 (plan / validate / ...)
        - summary: 当前阶段的摘要
        - options_hint: 可选的选项提示 (如 ["GGA-PBE", "HSE06"])
        """
        qtype = context.get("question_type", "task_vague")
        phase = context.get("phase", "")
        summary = context.get("summary", "")

        # 有真实 LLM 时让它生成更好的问题
        if model is not None and not hasattr(model, "_mock_name"):
            try:
                return self._llm_generate_question(model, qtype, phase, summary, context)
            except Exception:
                logger.debug("llm generate question failed", exc_info=True)  # 降级到模板

        # 模板兜底
        return self._template_question(qtype, summary, context)

    def _llm_generate_question(
        self, model: Any, qtype: str, phase: str,
        summary: str, context: dict[str, Any],
    ) -> tuple[str, list[str], str]:
        """调 LLM 生成上下文相关的提问. 失败时抛异常让上层降级."""
        from langchain_core.messages import HumanMessage, SystemMessage

        options_hint = context.get("options_hint", [])
        options_str = " / ".join(options_hint) if options_hint else "N/A"

        sys_prompt = (
            "You generate a single concise clarifying question for a materials "
            "science research agent. The question must be specific to the context, "
            "offer clear options when possible, and include a safe default. "
            "Reply in the same language as the summary. "
            f"Output format: QUESTION\\nOPTIONS (comma-separated)\\nDEFAULT"
        )
        user_prompt = (
            f"Phase: {phase}\n"
            f"Question type: {qtype}\n"
            f"Context summary: {summary[:500]}\n"
            f"Available options hint: {options_str}\n"
            "Generate one question, options, and a safe default."
        )
        resp = model.ainvoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
        # model.ainvoke 返回 coroutine, 但这里可能不在 async 上下文
        # 如果是 coroutine, 抛异常让上层走模板
        if asyncio.iscoroutine(resp):
            raise RuntimeError("async model in sync generate_question")
        text = str(resp.content).strip()
        parts = text.split("\n")
        question = parts[0].strip() if parts else text
        options = [o.strip() for o in parts[1].split(",")] if len(parts) > 1 else []
        default = parts[2].strip() if len(parts) > 2 else (options[0] if options else "")
        return question, options, default

    @staticmethod
    def _template_question(
        qtype: str, summary: str, context: dict[str, Any],
    ) -> tuple[str, list[str], str]:
        """无 LLM 时的模板提问. 按类型走不同模板."""
        options_hint = context.get("options_hint", [])

        if qtype == "cost_confirm":
            tool = context.get("tool", "计算")
            q = f"即将执行 {tool} 计算, 预计耗时较长. 确认执行？"
            return q, ["确认执行", "调整参数", "取消"], "确认执行"

        if qtype == "multi_path":
            paths = options_hint or ["方案A", "方案B"]
            q = f"检测到多个可选路径: {' / '.join(paths)}. 请选择？"
            return q, paths, paths[0] if paths else ""

        if qtype == "param_ambiguous":
            param = context.get("param", "关键参数")
            q = f"{param} 未明确指定. 请提供具体值？"
            return q, options_hint or [], options_hint[0] if options_hint else ""

        if qtype == "validation_fail":
            fails = context.get("consecutive_failures", 1)
            q = (
                f"已连续 {fails} 轮验证未通过. "
                f"当前结果: {summary[:200]}. "
                "建议方向？"
            )
            return q, [
                "修正假设重新实验",
                "调整计算参数",
                "换一种方法",
                "继续当前路径",
            ], "继续当前路径"

        # task_vague 兜底
        q = f"当前任务描述不够具体: {summary[:200]}. 请补充目标或输入？"
        return q, [], ""

    def _record_stats(self, question_type: str, outcome: str) -> None:
        """记录提问结果 (answered / timed_out / skipped)."""
        with self._lock:
            s = self._stats.setdefault(question_type, {"asked": 0, "answered": 0, "timed_out": 0})
            s["asked"] += 1
            if outcome in ("answered", "timed_out"):
                s[outcome] += 1

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
            self._last_ask_time[thread_id] = time.time()

        qtype = (metadata or {}).get("question_type", "agent_initiated")
        try:
            # 挂着等回答, 超时退回 default
            answer = await asyncio.wait_for(fut, timeout=timeout_val)
            self._record_stats(qtype, "answered")
            return str(answer)
        except asyncio.TimeoutError:
            q.answer = default_answer or ""
            q.answered_at = time.time()
            self._record_stats(qtype, "timed_out")
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


# ── 轻量声明式触发 (chat 入口用, 不阻塞) ─────────────────────────
#
# ClarificationManager 是给 LLM 主动调 clarification_tool 用的 (阻塞式).
# should_ask_clarification 是给 chat() 入口用的声明式预检 — 在 agent loop
# 之前扫一遍用户消息, 命中模糊信号就 yield clarification_request 事件,
# 不阻塞, 用户可选择回答或忽略继续. 两者互补: 一个事前提示, 一个事中追问.
# 用户 profile: "more questioning环节" + "capturing vague intuitions".

import re as _re

_ACTION_VERBS = _re.compile(
    r"(?:计算|模拟|分析|优化|生成|查找|搜索|对比|预测|拟合|"
    r"跑|做|算|查|看|画|写|改|删|建|run|calc|simul|analy|"
    r"optimi|generat|search|find|compar|predict|fit|plot|"
    r"write|create|build|delete|update)",
    _re.IGNORECASE,
)

_ANALOGY_SIGNAL = _re.compile(
    r"(?:这就像|类似于|好像.*一样|好比|reminds me of|it's like)",
    _re.IGNORECASE,
)

_MATERIAL_PATTERN = _re.compile(
    r"\b(?:[A-Z][a-z]?\d?(?:[A-Z][a-z]?\d?)*|"
    r"perovskite|MXene|MOF|COF|"
    r"钙钛矿|石墨烯|氮化镓|碳化硅|氧化物|合金|"
    r"GaN|SiC|TiO2|ZnO|Fe2O3|Cu2O|MoS2|WSe2|hBN)\b"
)


def should_ask_clarification(
    message: str,
    session_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """声明式模糊意图检测. 命中返回 clarification dict, 否则 None.

    三档触发 (任一命中即问):
      1. cross_domain_analogy: 跨领域类比 → 问是机制映射还是比喻
      2. no_action_verb: 无明确动词 + 短消息 → 问想做什么
      3. new_material: 首次提到材料体系 → 问具体操作

    不做语义判断, 纯关键词. 宁可误问也不漏问 — 反正不阻塞.
    """
    if not message or len(message) < 3:
        return None

    if _ANALOGY_SIGNAL.search(message):
        return {
            "reason": "cross_domain_analogy",
            "suggestion": [
                "你想把这个机制的物理原理迁移过来吗?",
                "还是只是打个比方帮助理解?",
                "需要我查一下两个领域的相关性吗?",
            ],
            "raw": message[:120],
        }

    if not _ACTION_VERBS.search(message) and len(message.split()) <= 8:
        return {
            "reason": "no_action_verb",
            "suggestion": [
                "你是想让我计算/模拟这个体系?",
                "还是查找相关文献/数据?",
                "或者只是记录这个想法供后续参考?",
            ],
            "raw": message[:120],
        }

    mat_match = _MATERIAL_PATTERN.search(message)
    if mat_match and session_history is not None:
        material = mat_match.group(0)
        seen_before = any(
            material in (m.get("content", "") or "")
            for m in session_history[-20:]
        )
        if not seen_before:
            return {
                "reason": "new_material",
                "material": material,
                "suggestion": [
                    f"首次提到 {material}, 你想做什么?",
                    f"查 {material} 的结构/性质数据?",
                    f"用 {material} 跑模拟 (VASP/LAMMPS)?",
                    f"只是记录, 不需要现在处理?",
                ],
                "raw": message[:120],
            }

    return None
