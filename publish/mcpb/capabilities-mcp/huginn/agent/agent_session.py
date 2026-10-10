"""AgentSession —— 库式会话内核 (Pi ``pi-agent-core`` 的类比).

目的: 让 huginn 能被当**库**嵌入, 而不只是当 fastapi server 跑. 宿主
``from huginn.agent.agent_session import AgentSession`` 即可开一轮循环,
与 server/routes 完全解耦 —— 这对应 Pi 的"OpenClaw import pi-agent-core 当引擎".

本门面只做薄封装, 复用 HuginnAgent 既有的会话树 / 流式 / 分支机制:
  - :meth:`prompt`     一轮对话 (阻塞, 返回末态)
  - :meth:`astream`    流式事件 (async generator, 复用 ``chat``)
  - :meth:`mode`       切换 pi 极简内核 / 默认
  - :meth:`set_system_prompt`  换系统提示 (Pi 的 SYSTEM.md 控制)
  - :meth:`set_code_mode`      开/关 Code Mode (DSH 的 PTC: 模型写代码编排多步工具)
  - :meth:`trajectory`         按来源读取本次运行的可审计轨迹 (DSH 的 trajectory view)
  - :meth:`fork` / :meth:`rewind` / :meth:`branches`  会话树 (Pi 的 branch/replay)

不新增任何中心机制, 零性能 / 零注册表改动。
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger(__name__)

# DSH trajectory: 会话事件 kind → 来源分类 (对标 DeepSeek Harness 的
# "每个上下文注入都按来源可审计"). 消费已有的 huginn.events.session_log 事件流.
_TRAJECTORY_SOURCE: dict[str, str] = {
    "message": "model",
    "reasoning": "model",
    "model_change": "model",
    "tool_call": "tools",
    "tool_result": "tools",
    "phase_change": "governance",
    "cognitive_mode_change": "governance",
    "branch_summary": "governance",
    "autoloop_phase_change": "governance",
    "compaction": "context",
    "reset_boundary": "context",
    "file_hash_mismatch": "context",
    "context_injection": "context",
}


def _trajectory_source(kind: str) -> str:
    return _TRAJECTORY_SOURCE.get(kind, "custom")


class AgentSession:
    """把 HuginnAgent 包成一个可嵌入、可传会话树的会话内核."""

    def __init__(
        self,
        agent: Any | None = None,
        *,
        config: Any | None = None,
        thread_id: str = "default",
        profile_id: str = "lead",
    ) -> None:
        if agent is None:
            agent = self._build_agent(config, profile_id=profile_id)
        self.agent = agent
        # thread_id 兜底: 优先用 agent 自带的 (若它持有会话线程).
        self.thread_id = getattr(agent, "thread_id", None) or thread_id
        # P1 能力维度: 当前选中的 loop (None = 默认 langgraph agent.chat).
        self._loop: Any | None = None

    # ── 构造 ───────────────────────────────────────────────────────

    @staticmethod
    def _build_agent(config: Any | None, profile_id: str) -> Any:
        if config is None:
            from huginn.config import HuginnConfig

            config = HuginnConfig.from_env()
        from huginn.agent.core import HuginnAgent

        return HuginnAgent.from_config(config, profile_id=profile_id)

    @classmethod
    def from_config(
        cls, config: Any, thread_id: str = "default", profile_id: str = "lead"
    ) -> AgentSession:
        """Headless 建链: 从 HuginnConfig 构造, 不经 server."""
        return cls(config=config, thread_id=thread_id, profile_id=profile_id)

    @classmethod
    def attach(cls, agent: Any) -> AgentSession:
        """包装一个已构造的 agent (沿用其线程/内存/工具)."""
        return cls(agent=agent)

    # ── 对话 ───────────────────────────────────────────────────────

    def prompt(self, message: str, *, thread_id: str | None = None) -> dict[str, Any]:
        """跑一轮对话, 返回末态 dict (阻塞). thread_id 缺省用本会话."""
        return self.agent.invoke(message, thread_id or self.thread_id)

    def astream(
        self, message: str, *, thread_id: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """流式事件 (async generator). 优先当前选中 loop 能力, 否则走 agent.chat."""
        if self._loop is not None:
            return self._loop(self.agent, message, thread_id or self.thread_id)
        return self.agent.chat(message, thread_id or self.thread_id)

    def aprompt(
        self, message: str, *, thread_id: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """异步逐事件迭代 (等价 astream; 提供更直觉的命名)."""
        return self.astream(message, thread_id=thread_id)

    # ── 模式 / 提示控制 (Pi 的 SYSTEM.md) ──────────────────────────

    def mode(self, name: str) -> dict[str, Any]:
        """切换内核模式: ``"pi"`` 极简 / ``"default"`` 全量. 作用于工具注册表可见面."""
        from huginn.modes.pi import apply_pi_mode, exit_pi_mode, pi_active

        if name == "pi":
            return apply_pi_mode()
        if name == "default":
            return exit_pi_mode()
        if name == "status":
            return {"mode": "pi" if pi_active() else "default"}
        raise ValueError(f"unknown mode {name!r}; expected 'pi'/'default'/'status'")

    def set_system_prompt(self, text: str) -> None:
        """替换系统提示并令图/工具描述缓存重建 (Pi 的 SYSTEM.md 热切换)."""
        self.agent.system_prompt = (text or "").strip()
        self.agent._agent_graph = None
        if hasattr(self.agent, "_invalidate_tool_description_cache"):
            self.agent._invalidate_tool_description_cache()

    # ── 会话树 (Pi 的 branch / rewind) ─────────────────────────────

    def fork(self, from_node_id: str | None = None) -> dict[str, Any]:
        """从当前 (或指定) 节点分叉出一条新分支, 返回分支信息."""
        return self.agent.fork_conversation(from_node_id)

    def rewind(self, node_id: str) -> dict[str, Any]:
        """回退到会话树的某个先辈节点并从那里继续."""
        return self.agent.switch_branch(node_id)

    def branches(self) -> dict[str, Any]:
        """当前会话树的所有分支."""
        return self.agent.conversation_branches()

    # ── DSH 对齐: Code Mode + trajectory 审计 ──────────────────────

    def set_code_mode(self, on: bool) -> dict[str, Any]:
        """开关 Code Mode (DSH 的 PTC): 模型写 Python 程序编排多步工具调用,
        而非逐轮 JSON tool_call. 复用既有 code_act_loop, 纯透传."""
        self.agent.mode = "code_act" if on else "tool_call"
        return {"code_mode": bool(on), "agent_mode": self.agent.mode}

    # ── 能力维度（P1）─────────────────────────────────────────────

    def _ensure_capabilities(self) -> Any:
        from huginn.capabilities.builtin import register_builtin_capabilities
        from huginn.capabilities.registry import get_shared_capability_registry

        register_builtin_capabilities()   # 幂等
        return get_shared_capability_registry()

    def set_loop(self, name: str) -> dict[str, Any]:
        """热切换编排循环 (loop 能力维度): 支持内置 name=default/code_act 及
        插件 mount 的 loop 能力. astream 随后的调用走该循环."""
        reg = self._ensure_capabilities()
        meta = reg.get("loop", name)
        if meta is None:
            raise ValueError(f"unknown loop capability {name!r}; available {reg.list_names('loop')}")
        self._loop = meta.impl
        return {"loop": name, "plugin": meta.plugin_name}

    def loops(self) -> list[tuple[str, str]]:
        return self._ensure_capabilities().list_names("loop")

    def sysprompt_sections(self) -> list[dict[str, Any]]:
        """装配插件贡献的系统提示 section (DSH `ctx.systemPrompt` 动态拼装).

        每个注册的 ``sysprompt`` 能力贡献一段文本; 贡献的段也会以
        ``sysprompt:<name>`` 来源落注入审计事件, 进 trajectory 按来源可查.
        """
        reg = self._ensure_capabilities()
        sections: list[dict[str, Any]] = []
        for meta in reg.list("sysprompt"):
            try:
                text = meta.impl({"thread_id": self.thread_id})
            except Exception:  # noqa: BLE001 — 单个 section 失败不阻断装配
                logger.debug("sysprompt capability %s failed", meta.name, exc_info=True)
                continue
            if not text:
                continue
            source = f"sysprompt:{meta.name}"
            sections.append({"capability": meta.name, "source": source, "text": text})
            self.inject(source, {"section_len": len(str(text))})
        return sections

    def trajectory(
        self, *, leaf_id: str | None = None, log: Any | None = None
    ) -> dict[str, Any]:
        """读本次会话的可审计轨迹, 按来源分组 (DSH trajectory view).

        数据来自既有的 append-only SessionEventLog (system prompts / 推理 /
        tool_call|result / 阶段切换 / 压缩 等事件). ``log`` 可注入 (测试用假 log);
        缺省打开本会话的事件日志, 无日志时 fail-open 返回空.
        """
        if log is None:
            log = self._open_session_log()
        evs = log.events_on_path(leaf_id) if log is not None else []
        by_source: dict[str, list[Any]] = {}
        for e in evs:
            src = self._event_source(e)
            by_source.setdefault(src, []).append(
                e.to_dict() if hasattr(e, "to_dict") else e
            )
        return {
            "thread_id": self.thread_id,
            "count": len(evs),
            "by_source": by_source,
            "events": [e.to_dict() if hasattr(e, "to_dict") else e for e in evs],
        }

    @staticmethod
    def _event_source(e: Any) -> str:
        """事件来源: 优先取 ``payload.source`` (真注入来源), 回落 kind→source."""
        if isinstance(e, dict):
            payload = e.get("payload") or {}
            kind = str(e.get("kind", ""))
        else:
            payload = getattr(e, "payload", None) or {}
            kind = str(getattr(e, "kind", ""))
        src = (payload or {}).get("source")
        return str(src) if src else _trajectory_source(kind)

    def inject(self, source: str, payload: dict[str, Any] | None = None) -> None:
        """记录一次上下文注入到本会话轨迹 (DSH 按来源审计). fail-open, 不阻断."""
        try:
            from huginn.events.session_writer import record_injection

            record_injection(self.thread_id, source, payload)
        except Exception:  # noqa: BLE001 — trajectory 注入写失败不阻断主流程
            return

    def _open_session_log(self) -> Any | None:
        try:
            from huginn.events.session_log import SessionEventLog

            return SessionEventLog.open(self.thread_id, load=True)
        except Exception:  # noqa: BLE001 — 事件日志不可用 → trajectory 返回空, 不阻断
            return None

    # ── 生命周期 ───────────────────────────────────────────────────

    def close(self) -> None:
        self.agent.close()

    def __enter__(self) -> AgentSession:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


__all__ = ["AgentSession"]
