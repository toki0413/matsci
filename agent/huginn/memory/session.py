"""Session memory — short-term context for active conversations.

Stores messages, tool calls, and reasoning traces for the current session.
Automatically compacts when context grows too large.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from huginn.types import AgentMessage, ToolResult


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation."""

    tool_name: str
    input_args: dict[str, Any]
    result: ToolResult | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    call_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class SessionContext:
    """Mutable context for the current agent session."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    messages: list[AgentMessage] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    reasoning_trace: list[str] = field(default_factory=list)
    user_preferences: dict[str, Any] = field(default_factory=dict)

    # Context compaction settings
    max_messages: int = 100
    max_tool_calls: int = 50
    max_reasoning_lines: int = 200

    # C1: WM sliding window — 极限模式才开, 平常 _compact_if_needed 走原逻辑.
    # ponytail: token 估算用 char//4 近似 (英文 ~4 char/token, 中文偏密但近似够用).
    # 升级路径: 装 tiktoken 后换精确计数.
    # 默认值从 env 读, 前端 Settings 可调.
    token_budget: int = field(
        default_factory=lambda: int(os.environ.get("HUGINN_WM_TOKEN_BUDGET", "8192"))
    )
    # summarize 触发周期: 每 N 次 add_message 检查一次 (避免每次都算 token).
    # ponytail: 简单计数器, 不引 debounce 抽象.
    _summarize_check_every: int = field(
        default_factory=lambda: int(os.environ.get("HUGINN_WM_SUMMARIZE_EVERY_N", "5"))
    )
    _add_count: int = 0
    # 历次 summarize 结果, 外部 (manager) 可读后写入 EM.
    summaries: list[str] = field(default_factory=list)
    # 最近一次 sliding window summarize 的时间戳 (ISO 字符串), 给前端面板显示.
    # None 表示从未 summarize 过. ponytail: 不存 history list, 只记最后一次够用.
    last_summarize_at: str | None = None
    # 可选 EM sink: manager 注入 callable(summary_text, metadata_dict).
    # 不注入时 summary 只存在 self.summaries 里, 不写 EM.
    episodic_sink: Callable[[str, dict[str, Any]], None] | None = None

    def add_message(
        self, message: AgentMessage | str, content: str | None = None
    ) -> None:
        if isinstance(message, str) and content is not None:
            message = AgentMessage(role=message, content=content)
        self.messages.append(message)
        self._add_count += 1
        self._compact_if_needed()

    def add_tool_call(self, record: ToolCallRecord) -> None:
        self.tool_calls.append(record)
        if len(self.tool_calls) > self.max_tool_calls:
            # Keep most recent, archive oldest
            self.tool_calls = self.tool_calls[-self.max_tool_calls :]

    def add_reasoning(self, text: str) -> None:
        self.reasoning_trace.append(text)
        if len(self.reasoning_trace) > self.max_reasoning_lines:
            # Summarize oldest entries
            self.reasoning_trace = self.reasoning_trace[-self.max_reasoning_lines :]

    def get_recent_messages(self, n: int = 10) -> list[AgentMessage]:
        return self.messages[-n:]

    def get_recent_tool_calls(self, n: int = 5) -> list[ToolCallRecord]:
        return self.tool_calls[-n:]

    def _compact_if_needed(self) -> None:
        # C1: 极限模式 + token 超预算 → sliding window summarize, 推到 EM.
        # 非极限模式: 走原逻辑 (keep system + recent).
        if os.environ.get("HUGINN_EXTREME_DISPATCH", "0").lower() in ("1", "true"):
            # 每 N 次 add 才检查一次 token, 避免每次都算
            if self._add_count % self._summarize_check_every != 0:
                return
            if self._estimate_tokens() > self.token_budget:
                self._sliding_window_compact()
                return
        if len(self.messages) > self.max_messages:
            # Strategy: keep first system message, then most recent
            system_msgs = [m for m in self.messages if m.role == "system"]
            recent = self.messages[-(self.max_messages - len(system_msgs)) :]
            self.messages = system_msgs + recent

    def _estimate_tokens(self) -> int:
        """粗估当前 messages 的 token 数. ponytail: char//4 近似."""
        total = 0
        for m in self.messages:
            c = m.content
            if isinstance(c, str):
                total += len(c)
            else:
                try:
                    total += len(json.dumps(c, ensure_ascii=False))
                except Exception:
                    total += 0
        return total // 4

    def _sliding_window_compact(self) -> None:
        """C1: 把超出预算的旧 messages summarize 成一条, 替换原 messages.

        保留 system + 最近 N 条不动, 中间段调 summarize_window 压缩.
        summary 写入 self.summaries, 若 episodic_sink 存在则同步推到 EM.
        """
        if len(self.messages) < 10:
            return  # 太短不压
        system_msgs = [m for m in self.messages if m.role == "system"]
        non_system = [m for m in self.messages if m.role != "system"]
        if len(non_system) < 8:
            return
        # 保留最近 1/3 不动, 压缩前 2/3
        keep_recent = max(4, len(non_system) // 3)
        to_summarize = non_system[:-keep_recent]
        recent = non_system[-keep_recent:]
        if not to_summarize:
            return
        summary = self.summarize_window(to_summarize)
        if not summary:
            # summarize 失败, fallback 走原逻辑
            recent = non_system[-(self.max_messages - len(system_msgs)) :]
            self.messages = system_msgs + recent
            return
        self.summaries.append(summary)
        # 记最近一次 summarize 时间, 给前端 Memory 层级面板显示
        self.last_summarize_at = datetime.now().isoformat()
        if self.episodic_sink is not None:
            try:
                self.episodic_sink(summary, {
                    "source": "wm_sliding_window",
                    "session_id": self.session_id,
                    "summarized_count": len(to_summarize),
                })
            except Exception:
                pass  # EM sink 失败不阻塞 session
        # summary 作为 system message 注回, 让后续 LLM 看到压缩上下文
        from huginn.types import AgentMessage as _AM
        summary_msg = _AM(
            role="system",
            content=f"[Compacted context summary]\n{summary}",
        )
        self.messages = system_msgs + [summary_msg] + recent

    def summarize_window(
        self,
        messages: list[AgentMessage],
        llm_chat_fn: Callable[[str], Any] | None = None,
    ) -> str:
        """C1: summarize 一段 messages. 默认 rule-based (0 LLM), env 切换策略.

        env HUGINN_WM_SUMMARIZE: rule (默认) / ngram / llm / hybrid.
        ponytail: rule 复用 _summarize_trajectory 模式 (tool_calls 名字序列
        + assistant 首句 + tool result 关键字段). 升级路径: ngram/llm.
        """
        strategy = os.environ.get("HUGINN_WM_SUMMARIZE", "rule").lower()
        if strategy == "llm" and llm_chat_fn is not None:
            return self._summarize_window_llm(messages, llm_chat_fn)
        if strategy == "hybrid" and llm_chat_fn is not None:
            # 偶尔 LLM (每 5 次), 平常 rule
            if (self._add_count // self._summarize_check_every) % 5 == 0:
                return self._summarize_window_llm(messages, llm_chat_fn)
            return self._summarize_window_rule(messages)
        if strategy == "ngram":
            return self._summarize_window_ngram(messages)
        return self._summarize_window_rule(messages)

    def _summarize_window_rule(self, messages: list[AgentMessage]) -> str:
        """rule-based: 0 LLM 调用. 抽 tool 名字 + assistant 首句 + tool result 关键字段.

        复用 trajectory_pattern._summarize_trajectory 的模式 (不引依赖,
        本地实现一份). 输出 ~500 token 文本.
        """
        tool_names: list[str] = []
        assistant_first_lines: list[str] = []
        tool_key_fields: list[str] = []
        # 关键字段 regex: r_phys / energy / success / convergence / band_gap
        key_field_re = re.compile(
            r'"(r_phys|energy|success|convergence|band_gap|forces|stress|diffusion|formation_energy)"\s*:\s*([^,}\s]+)'
        )
        for m in messages:
            c = m.content
            if isinstance(c, dict):
                # assistant 带 tool_calls: 抽 tool 名字
                tcs = c.get("tool_calls") or []
                for tc in tcs:
                    if isinstance(tc, dict):
                        name = tc.get("function", {}).get("name") or tc.get("name", "?")
                        tool_names.append(str(name))
                # 文本内容
                txt = c.get("content") or ""
                if isinstance(txt, str) and txt.strip():
                    first_line = txt.strip().split("\n", 1)[0][:100]
                    if first_line:
                        assistant_first_lines.append(first_line)
            elif isinstance(c, str):
                if m.role == "assistant" and c.strip():
                    first_line = c.strip().split("\n", 1)[0][:100]
                    if first_line:
                        assistant_first_lines.append(first_line)
                elif m.role == "tool":
                    # 抽关键字段
                    for k, v in key_field_re.findall(c):
                        tool_key_fields.append(f"{k}={v[:80]}")
                    # 没 key field 也截前 200 字符
                    if not key_field_re.search(c) and len(tool_key_fields) < 10:
                        tool_key_fields.append(c.strip()[:200])
        parts = [f"[rule-based summary of {len(messages)} messages]"]
        if tool_names:
            parts.append(f"Tools: {', '.join(tool_names[:20])}")
        if assistant_first_lines:
            parts.append("Assistant turns:")
            for line in assistant_first_lines[:8]:
                parts.append(f"  - {line}")
        if tool_key_fields:
            parts.append("Key results:")
            for kf in tool_key_fields[:10]:
                parts.append(f"  - {kf}")
        return "\n".join(parts)

    def _summarize_window_ngram(self, messages: list[AgentMessage]) -> str:
        """ngram: 0 LLM. 抽高频 2-3 gram top-N. ponytail: 简单 Counter."""
        from collections import Counter
        text_parts: list[str] = []
        for m in messages:
            c = m.content
            if isinstance(c, str):
                text_parts.append(c)
            elif isinstance(c, dict):
                txt = c.get("content") or ""
                if isinstance(txt, str):
                    text_parts.append(txt)
        text = " ".join(text_parts)
        # 简单 2-gram (按词)
        words = re.findall(r'\S+', text)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        top = Counter(bigrams).most_common(15)
        parts = [f"[ngram summary of {len(messages)} messages, top-15 bigrams]"]
        for gram, cnt in top:
            parts.append(f"  ({cnt}x) {gram}")
        return "\n".join(parts)

    def _summarize_window_llm(
        self,
        messages: list[AgentMessage],
        llm_chat_fn: Callable[[str], Any],
    ) -> str:
        """LLM summarize. ponytail: 不 await, 假设 llm_chat_fn 同步或已 wrap.
        失败返空串, 让上层 fallback rule.
        """
        try:
            text_parts: list[str] = []
            for m in messages:
                c = m.content
                if isinstance(c, str):
                    text_parts.append(f"[{m.role}] {c}")
                elif isinstance(c, dict):
                    txt = c.get("content") or ""
                    if isinstance(txt, str):
                        text_parts.append(f"[{m.role}] {txt}")
            joined = "\n".join(text_parts)[:6000]
            prompt = (
                "Summarize the following agent messages into a concise context "
                "(key actions, results, current state). Under 500 tokens.\n\n"
                + joined
            )
            resp = llm_chat_fn(prompt)
            if isinstance(resp, str) and resp.strip():
                return f"[llm summary]\n{resp.strip()}"
        except Exception:
            pass
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "message_count": len(self.messages),
            "tool_call_count": len(self.tool_calls),
            "user_preferences": self.user_preferences,
        }

    def export_full(self) -> dict[str, Any]:
        """Export complete session for serialization."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "messages": [
                {
                    "role": m.role,
                    "content": (
                        m.content
                        if isinstance(m.content, str)
                        else json.dumps(m.content)
                    ),
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in self.messages
            ],
            "tool_calls": [
                {
                    "tool_name": t.tool_name,
                    "input_args": t.input_args,
                    "success": t.result.success if t.result else None,
                    "timestamp": t.timestamp.isoformat(),
                    "call_id": t.call_id,
                }
                for t in self.tool_calls
            ],
        }


def _run_c1_selfcheck() -> None:
    """C1: WM sliding window + rule-based summarize selfcheck.

    覆盖:
    - 非极限模式: 走原逻辑 (max_messages 截断), summaries 始终空
    - 极限模式 + token 超预算: 触发 sliding_window_compact, summaries 非空
    - rule 策略: 抽出 tool 名字 / assistant 首句 / tool key field
    - ngram 策略: 抽 bigram
    - llm 策略: 调注入的 fake llm_chat_fn
    - episodic_sink 回调被调用
    """
    import os as _os
    from huginn.types import AgentMessage

    # ── 非极限模式: summaries 应始终空, 走原 max_messages 逻辑 ──
    _os.environ.pop("HUGINN_EXTREME_DISPATCH", None)
    _os.environ.pop("HUGINN_WM_SUMMARIZE", None)
    sc = SessionContext(max_messages=5)
    for i in range(20):
        sc.add_message(AgentMessage(role="user", content=f"msg {i}" * 50))
    assert len(sc.summaries) == 0, "非极限模式不应触发 summarize"
    assert len(sc.messages) <= 5 + 1, f"原逻辑应截到 max_messages, got {len(sc.messages)}"
    print("C1-A non-extreme: no summarize, max_messages truncate OK")

    # ── 极限模式 + token 超预算: 触发 sliding window ──
    _os.environ["HUGINN_EXTREME_DISPATCH"] = "1"
    sc2 = SessionContext(token_budget=200)  # 故意调小, 加几条就超
    sc2._summarize_check_every = 1  # 每次 add 都检查 (测试用)
    for i in range(20):
        sc2.add_message(AgentMessage(
            role="assistant",
            content=f"Assistant turn {i}. " + "x" * 200,
        ))
    assert len(sc2.summaries) >= 1, (
        f"极限模式 + 超预算应触发 summarize, got {len(sc2.summaries)} summaries"
    )
    # summary 应注入回 messages (作为 system message)
    has_summary_msg = any(
        isinstance(m.content, str) and "[Compacted context summary]" in m.content
        for m in sc2.messages
    )
    assert has_summary_msg, "compact 后应有 summary system message"
    print(f"C1-B extreme + over budget: {len(sc2.summaries)} summaries, summary msg injected OK")

    # ── rule 策略: 抽 tool 名字 + assistant 首句 + tool key field ──
    _os.environ["HUGINN_WM_SUMMARIZE"] = "rule"
    msgs = [
        AgentMessage(role="assistant", content={
            "content": "Let me run DFT calculation.",
            "tool_calls": [{"function": {"name": "run_dft"}}],
        }),
        AgentMessage(role="tool", content='{"energy": -4.5, "success": true, "r_phys": 0.85}'),
        AgentMessage(role="assistant", content="The calculation converged."),
    ]
    out = sc2._summarize_window_rule(msgs)
    assert "run_dft" in out, f"rule 应抽 tool 名字, got: {out!r}"
    assert "Let me run DFT" in out, f"rule 应抽 assistant 首句, got: {out!r}"
    assert "energy=-4.5" in out, f"rule 应抽 key field, got: {out!r}"
    assert "r_phys=0.85" in out, f"rule 应抽 r_phys, got: {out!r}"
    print("C1-C rule strategy: tool + assistant + key fields OK")

    # ── ngram 策略: 抽 bigram ──
    _os.environ["HUGINN_WM_SUMMARIZE"] = "ngram"
    msgs2 = [
        AgentMessage(role="user", content="band gap calculation band gap result band gap"),
    ]
    out2 = sc2._summarize_window_ngram(msgs2)
    assert "band gap" in out2, f"ngram 应抓 'band gap' bigram, got: {out2!r}"
    assert "(3x)" in out2 or "(2x)" in out2, f"ngram 应显示频次, got: {out2!r}"
    print("C1-D ngram strategy: bigram count OK")

    # ── llm 策略: 调注入的 fake llm_chat_fn ──
    _os.environ["HUGINN_WM_SUMMARIZE"] = "llm"
    def _fake_llm(prompt: str) -> str:
        return "FAKE LLM SUMMARY"
    out3 = sc2.summarize_window(msgs, llm_chat_fn=_fake_llm)
    assert "FAKE LLM SUMMARY" in out3, f"llm 策略应用 fake 返回, got: {out3!r}"
    print("C1-E llm strategy: injected fake llm OK")

    # ── llm 策略但 llm_chat_fn=None: fallback 到 rule ──
    out4 = sc2.summarize_window(msgs, llm_chat_fn=None)
    assert "rule-based summary" in out4, (
        f"llm 策略无 chat_fn 应 fallback rule, got: {out4!r}"
    )
    print("C1-F llm strategy without chat_fn → rule fallback OK")

    # ── episodic_sink 回调被调用 ──
    _os.environ["HUGINN_WM_SUMMARIZE"] = "rule"
    sink_calls: list[tuple[str, dict]] = []
    def _sink(text: str, meta: dict) -> None:
        sink_calls.append((text, meta))
    sc3 = SessionContext(token_budget=100)
    sc3._summarize_check_every = 1
    sc3.episodic_sink = _sink
    for i in range(15):
        sc3.add_message(AgentMessage(role="user", content=f"msg {i} " + "y" * 200))
    assert len(sink_calls) >= 1, (
        f"episodic_sink 应被调用, got {len(sink_calls)} calls"
    )
    assert all(isinstance(t, str) and t for t, _ in sink_calls), "sink 应收非空 str"
    assert all(m.get("source") == "wm_sliding_window" for _, m in sink_calls), (
        "sink metadata source 应为 wm_sliding_window"
    )
    print(f"C1-G episodic_sink: {len(sink_calls)} calls, metadata OK")

    _os.environ.pop("HUGINN_EXTREME_DISPATCH", None)
    _os.environ.pop("HUGINN_WM_SUMMARIZE", None)
    print("SessionContext C1 selfcheck OK")


if __name__ == "__main__":
    _run_c1_selfcheck()
