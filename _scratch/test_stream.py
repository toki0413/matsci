"""探针: internlm 的 invoke vs astream, 以及被 persona 装饰后的 prompt."""
import asyncio
import os
import sys

sys.path.insert(0, "/workspace/agent")

os.environ.setdefault("HUGINN_PROVIDER", "internlm")
os.environ.setdefault("HUGINN_MODEL", "intern-s2-preview")


async def main() -> None:
    from langchain_core.messages import HumanMessage, SystemMessage

    from huginn.server_core import get_agent_factory

    factory = get_agent_factory()
    model = factory.model_registry.resolve(factory.model_registry.default_alias())

    msgs = [
        SystemMessage(content="You are a concise assistant."),
        HumanMessage(content="Say exactly: HELLO_RIGIDITY. Nothing else."),
    ]

    r = model.invoke(msgs)
    print("invoke ->", repr(r.content))

    parts = []
    async for chunk in model.astream(msgs):
        parts.append(getattr(chunk, "content", ""))
    print("astream ->", repr("".join(parts)))

    # 模拟 persona 装饰后的 human 内容
    decorated = (
        "[Current inner state] You feel slightly down. Let this subtly colour your "
        "tone; do not mention these feelings explicitly.\n\n"
        "### Structural Coordinates (L1)\nexploration\n\n"
        "Say exactly: HELLO_RIGIDITY. Nothing else."
    )
    r2 = model.invoke([
        SystemMessage(content="You are a concise assistant."),
        HumanMessage(content=decorated),
    ])
    print("invoke(decorated) ->", repr(r2.content))

    r3 = model.invoke([
        SystemMessage(content="You are a concise assistant.\n" * 200),
        HumanMessage(content="Say exactly: HELLO_RIGIDITY. Nothing else."),
    ])
    print("invoke(huge system) ->", repr(r3.content))


if __name__ == "__main__":
    asyncio.run(main())