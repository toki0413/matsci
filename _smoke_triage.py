"""ponytail self-check: triage_prompt 不再写 dummy 模板."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "agent"))

from huginn.bench.orchestrator import _triage_prompt
prompt = _triage_prompt({"train.py", "report.md"})
# 旧模板特征: "50-line skeleton with main() that saves dummy output"
assert "50-line skeleton" not in prompt, "old skeleton template still present"
assert "saves dummy" not in prompt, "old dummy-save template still present"
assert "You decide" in prompt or "decide" in prompt, "no advisory hint"
assert "REAL stubs" in prompt or "real" in prompt.lower(), "no real-stub guidance"
print("self-check passed: triage_prompt is advisory, not dummy-template")
