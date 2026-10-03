"""定位 branch incubator ok=0: 直接派发一次 explore 子智能体, 打印 success/error/summary."""
import asyncio
import os
import sys

sys.path.insert(0, "/workspace/agent")

os.environ.setdefault("HUGINN_PROVIDER", "internlm")
os.environ.setdefault("HUGINN_MODEL", "intern-s2-preview")


async def main() -> None:
    from huginn.agents.subagent import SubagentDispatch
    from huginn.server_core import get_agent_factory

    factory = get_agent_factory()
    print("factory:", factory)

    d = SubagentDispatch()
    res = await d.dispatch(
        "explore",
        "Propose one concrete, testable hypothesis for: small feedforward nets as a "
        "probe of solution-space rigidity. Be brief.",
        context={"agent_factory": factory},
    )
    print("success=", res.success)
    print("error=", res.error)
    print("summary_len=", len(res.summary or ""))
    print("summary[:400]=", (res.summary or "")[:400])
    print("tokens=", res.tokens_used)
    print("tool_calls=", len(res.tool_calls))


if __name__ == "__main__":
    asyncio.run(main())