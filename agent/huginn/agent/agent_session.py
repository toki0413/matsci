"""AgentSession —— 库式会话内核 (Pi ``pi-agent-core`` 的类比).

目的: 让 huginn 能被当**库**嵌入, 而不只是当 fastapi server 跑. 宿主
``from huginn.agent.agent_session import AgentSession`` 即可开一轮循环,
与 server/routes 完全解耦 —— 这对应 Pi 的"OpenClaw import pi-agent-core 当引擎".

本门面只做薄封装, 复用 HuginnAgent 既有的会话树 / 流式 / 分支机制:
  - :meth:`prompt`     一轮对话 (阻塞, 返回末态)
  - :meth:`astream`    流式事件 (async generator, 复用 ``chat``)
  - :meth:`mode`       切换 pi 极简内核 / 默认
  - :meth:`set_system_prompt`  换系统提示 (Pi 的 SYSTEM.md 控制)
  - :meth:`fork` / :meth:`rewind` / :meth:`branches`  会话树 (Pi 的 branch/replay)

不新增任何中心机制, 零性能 / 零注册表改动。
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any


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
        """流式事件 (async generator). 复用 HuginnAgent.chat 的事件流."""
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

    # ── 生命周期 ───────────────────────────────────────────────────

    def close(self) -> None:
        self.agent.close()

    def __enter__(self) -> AgentSession:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


__all__ = ["AgentSession"]
