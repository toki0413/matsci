"""内置能力示例 —— 证明 loop / sysprompt 维度可 mount.

``plugin_name="builtin"``: 卸载流程会把它当普通插件撤销, 但默认注册进共享 registry,
让 AgentSession 开箱即用.

- ``loop.default``    : 默认 langgraph 循环 (直接透传 agent.chat 事件流)
- ``loop.code_act``   : CodeAct 循环 (模型写 Python 驱工具, 复用 code_act_loop)
- ``sysprompt.builtin``: 演示插件贡献一段 prompt section
- ``fusion.fused_kv`` : KV-KV 语义融合通道 (C2C 范式) 的探针; 默认如实报告不可用,
                        由拥有本地模型 + projector 的运行环境按需覆盖挂载
"""
from __future__ import annotations

from typing import Any

from huginn.capabilities.capability import CapabilityMetadata
from huginn.capabilities.registry import get_shared_capability_registry

# ── loop ────────────────────────────────────────────────────────────

async def _loop_default(agent: Any, message: str, thread_id: str = "default"):
    """内置默认循环: 直接复用 agent.chat 的事件流 (langgraph tool_call)."""
    async for ev in agent.chat(message, thread_id):
        yield ev


async def _loop_code_act(agent: Any, message: str, thread_id: str = "default"):
    """CodeAct 循环: LLM 输出 Python 块编排多步工具 (嵌套结果留在执行局部)."""
    from huginn.agent.code_act_loop import run_code_act_turn

    async for ev in run_code_act_turn(agent, message, thread_id):
        yield ev


# ── sysprompt ───────────────────────────────────────────────────────

def _sysprompt_builtin(ctx: dict[str, Any] | None = None) -> str | None:
    """演示: 插件贡献一段系统提示 section (返回 None 表示不贡献)."""
    return "## 内置 sysprompt 能力示例: 显式声明你的身份/规范."


def _mk(dimension: str, name: str, impl: Any) -> CapabilityMetadata:
    return CapabilityMetadata(dimension=dimension, name=name, impl=impl, plugin_name="builtin")


def register_builtin_capabilities() -> None:
    """注册内置能力到共享 registry (幂等: 同名覆盖)."""
    from huginn.capabilities.fusion import register_fusion_capabilities

    register_fusion_capabilities()
    reg = get_shared_capability_registry()
    reg.register(
        _mk("loop", "default", _loop_default),
        _mk("loop", "code_act", _loop_code_act),
        _mk("sysprompt", "builtin", _sysprompt_builtin),
    )


__all__ = ["register_builtin_capabilities", "_loop_default", "_loop_code_act", "_sysprompt_builtin"]
