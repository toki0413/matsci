"""Self-check for code_act_loop — run with: python -m huginn.agent.code_act_selfcheck

Ponytail rule: non-trivial logic leaves ONE runnable check. CodeAct's
non-trivial parts are (1) the degrade-on-3-errors contract and (2) the
normal-termination-when-no-code contract. Pure helpers (extract_blocks,
safe_import, tool_signature) are obvious enough to skip.
"""
import asyncio
import sys

from langchain_core.messages import AIMessage


class _MockModel:
    """Yields canned responses in sequence."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def ainvoke(self, messages):
        resp = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return AIMessage(content=resp)


class _MockAgent:
    def __init__(self, model):
        self.model = model
        class _SS:
            workspace = "."
            audit_logger = None
        self.session_state = _SS()

    def select_model(self, task):
        return self.model


async def _collect(agent, message):
    from huginn.agent.code_act_loop import run_code_act_turn
    out = []
    async for ev in run_code_act_turn(agent, message):
        out.append(ev)
    return out


def test_degrade_after_three_errors():
    """3 consecutive code errors → degrade event."""
    agent = _MockAgent(_MockModel(["```python\n1/0\n```"] * 10))
    events = asyncio.run(_collect(agent, "test"))
    types = [e["type"] for e in events]
    assert "code_act_degraded" in types, f"expected degrade, got: {types}"
    errs = [e for e in events if e["type"] == "code_executed" and e.get("error")]
    assert len(errs) == 3, f"expected 3 errors before degrade, got {len(errs)}"


def test_normal_termination():
    """LLM stops emitting code → final event, no degrade."""
    agent = _MockAgent(_MockModel([
        "```python\nprint('working')\n```",
        "The answer is 42.",
    ]))
    events = asyncio.run(_collect(agent, "test"))
    types = [e["type"] for e in events]
    assert "final" in types
    assert "code_act_degraded" not in types
    assert "42" in [e for e in events if e["type"] == "final"][0]["content"]


if __name__ == "__main__":
    test_degrade_after_three_errors()
    print("PASS: degrade_after_three_errors")
    test_normal_termination()
    print("PASS: normal_termination")
    print("ALL CHECKS PASSED")
    sys.exit(0)
