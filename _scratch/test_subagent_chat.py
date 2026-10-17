"""隔离: 子智能体 chat 为何返回 'in prompt processing error'.

对比 (a) 无工具 agent, (b) 有工具 agent, (c) 直接 model.invoke 带 tools.
"""
import asyncio
import os
import sys

sys.path.insert(0, "/workspace/agent")

os.environ.setdefault("HUGINN_PROVIDER", "internlm")
os.environ.setdefault("HUGINN_MODEL", "intern-s2-preview")

TASK = "Say exactly: HELLO_RIGIDITY. Nothing else."


async def run_agent_chat(factory, *, with_tools: bool) -> None:
    agent = factory.create(
        profile_id="lead",
        thread_id=f"probe_{with_tools}",
        system_prompt_override="You are a concise assistant.",
    )
    if not with_tools:
        agent.langchain_tools.clear()
    print(f"[with_tools={with_tools}] n_tools={len(agent.langchain_tools)}")
    last = None
    async for state in agent.chat(TASK, f"probe_{with_tools}"):
        if isinstance(state, dict) and state.get("messages"):
            last = state
    if last is None:
        print("  NO messages state")
        return
    msgs = last["messages"]
    print(f"  n_messages={len(msgs)}")
    for i, m in enumerate(msgs):
        c = getattr(m, "content", None)
        s = str(c)
        print(f"    [{i}] {type(m).__name__} len={len(s)} head={s[:220]!r} tail={s[-160:]!r}")


def run_raw_with_tools(factory) -> None:
    from langchain_core.messages import HumanMessage, SystemMessage

    model = factory.model_registry.resolve(factory.model_registry.default_alias())
    tools = [
        {
            "type": "function",
            "function": {
                "name": "file_read_tool",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    m = model.bind_tools(tools) if hasattr(model, "bind_tools") else model
    r = m.invoke([
        SystemMessage(content="You are a concise assistant. Do not call tools."),
        HumanMessage(content=TASK),
    ])
    print(f"[raw bind_tools] content={getattr(r, 'content', None)!r}")


async def main() -> None:
    from huginn.server_core import get_agent_factory

    factory = get_agent_factory()
    await run_agent_chat(factory, with_tools=False)
    run_raw_with_tools(factory)
    await run_agent_chat(factory, with_tools=True)


if __name__ == "__main__":
    asyncio.run(main())