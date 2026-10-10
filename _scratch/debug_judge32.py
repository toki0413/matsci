"""Inspect the raw GoalJudge LLM response on the run32 report."""
import glob
import os
from pathlib import Path

os.environ.setdefault("HUGINN_PROVIDER", "internlm")
os.environ.setdefault("HUGINN_MODEL", "intern-s2-preview")
os.environ.setdefault(
    "INTERNLM_API_KEY", "sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba"
)

from huginn.autoloop import AutoloopEngine  # noqa: E402
from huginn.evaluation import goal_judge as gj  # noqa: E402

ws = Path("/workspace/research_outputs/shusheng_rsi_run32")
eng = AutoloopEngine(workspace=ws)
model = eng.verification_model or eng.model

f = glob.glob(str(ws / "huginn_autoloop_report_*.md"))[0]
t = Path(f).read_text(encoding="utf-8")
body = t[t.find("## Research Report"):]
obj = (ws / "objective.txt").read_text(encoding="utf-8")

prompt = gj._GOAL_JUDGE_PROMPT.format(
    objective=obj,
    trajectory_summary=gj._summarize_trajectory(None),
    final_output=body[:12000],
)
print("prompt chars:", len(prompt))
judge = gj.GoalJudge(llm=model)
raw = judge._invoke_llm(prompt)
print("=== RAW RESPONSE (len=%d) ===" % len(raw))
print(raw[:3000])