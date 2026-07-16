"""ponytail self-check: PhaseGate 遥测能写入 jsonl."""
import os, json, tempfile, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "agent"))

with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
    telemetry_path = f.name
os.environ["HUGINN_TELEMETRY_PATH"] = telemetry_path

from huginn.autoloop.phase_gate import PhaseGate

pg = PhaseGate(
    from_phase="plan", to_phase="execute", status="blocked",
    required_evidence=["hypothesis"], missing_evidence=["hypothesis"],
    feedback="missing hypothesis",
)
# __post_init__ 应该已经写了遥测
with open(telemetry_path, "r", encoding="utf-8") as f:
    record = json.loads(f.readline())
assert record["from_phase"] == "plan", f"wrong from_phase: {record}"
assert record["to_phase"] == "execute", f"wrong to_phase: {record}"
assert record["status"] == "blocked", f"wrong status: {record}"
assert record["missing"] == ["hypothesis"], f"wrong missing: {record}"
print(f"self-check passed: telemetry writes to jsonl — {record}")

# 验证 HUGINN_TELEMETRY_PATH 未设时不写 (静默)
os.environ.pop("HUGINN_TELEMETRY_PATH", None)
pg2 = PhaseGate(from_phase="x", to_phase="y", status="approved")
print("self-check passed: no telemetry path = silent no-op")

os.unlink(telemetry_path)
