"""ponytail self-check: HUGINN_SKIP_CSM=1 时 _run_post_turn_reflection 早退."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "agent"))

os.environ.pop("HUGINN_SKIP_CSM", None)
from huginn.agent.reflection import ReflectionMixin

class MockSession:
    tool_results_this_turn = [{"tool_name": "bash_tool", "content": "ok"}]

class MockAgent(ReflectionMixin):
    def __init__(self):
        self._session_state = MockSession()

agent = MockAgent()

os.environ["HUGINN_SKIP_CSM"] = "1"
try:
    agent._run_post_turn_reflection()
    print("self-check passed: HUGINN_SKIP_CSM=1 early-returns without error")
except AttributeError as e:
    assert False, f"skip failed — hit uninitialized attr: {e}"
