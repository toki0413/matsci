"""直接打一次 LLM, 看是否所有回复都是 'in prompt processing error' (provider 层问题)."""
import os
import sys

sys.path.insert(0, "/workspace/agent")


def main() -> None:
    from langchain_core.messages import HumanMessage, SystemMessage

    from huginn.server_core import get_agent_factory

    factory = get_agent_factory()
    alias = factory.model_registry.default_alias()
    print("default_alias=", alias)
    model = factory.model_registry.resolve(alias)
    for i in range(3):
        r = model.invoke([
            SystemMessage(content="You are a concise assistant."),
            HumanMessage(content="Reply with exactly one short sentence: what is 2+2?"),
        ])
        print(f"[{i}] content={r.content!r}")
        print(f"[{i}] meta={getattr(r, 'response_metadata', None)}")
        print(f"[{i}] usage={getattr(r, 'usage_metadata', None)}")


if __name__ == "__main__":
    main()