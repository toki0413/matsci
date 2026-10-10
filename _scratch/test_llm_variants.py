"""探针: 什么 prompt 会让模型吐出 'in prompt processing error'."""
import os
import sys

sys.path.insert(0, "/workspace/agent")

LONG = (
    "Propose one concrete, testable hypothesis for: small feedforward networks as a "
    "probe of solution-space rigidity. Define N_c(w) as the minimal hidden width for "
    "zero violation (max abs error <= 1e-3) on held-out constraint points. Decide the "
    "experiment family, samples, scan, and script. Justify the design. Output a single "
    "hypothesis statement, no prose. " * 3
)


def main() -> None:
    from langchain_core.messages import HumanMessage, SystemMessage

    from huginn.server_core import get_agent_factory

    factory = get_agent_factory()
    model = factory.model_registry.resolve(factory.model_registry.default_alias())

    cases = {
        "plain_short": (
            "You are a concise assistant.", "Reply with exactly one sentence: what is 2+2?"
        ),
        "explore_sys_short": (
            "You are an exploration agent. Read files, search code, and summarize findings. "
            "Do not modify anything.",
            "Propose one concrete, testable hypothesis for: small feedforward nets as a "
            "probe of solution-space rigidity. Be brief.",
        ),
        "explore_sys_long": (
            "You are an exploration agent. Read files, search code, and summarize findings. "
            "Do not modify anything.",
            LONG,
        ),
        "sys_only": (
            "You are an exploration agent. Read files, search code, and summarize findings. "
            "Do not modify anything.",
            "Say hello.",
        ),
    }
    for name, (s, h) in cases.items():
        try:
            r = model.invoke([SystemMessage(content=s), HumanMessage(content=h)])
            print(f"[{name}] {r.content!r}")
        except Exception as exc:
            print(f"[{name}] EXC {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()