"""Validate the GoalJudge fix on the existing run32 report (no autoloop rerun)."""
import glob
import os
from pathlib import Path

os.environ.setdefault("HUGINN_PROVIDER", "internlm")
os.environ.setdefault("HUGINN_MODEL", "intern-s2-preview")
os.environ.setdefault(
    "INTERNLM_API_KEY", "sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba"
)

from huginn.autoloop import AutoloopEngine  # noqa: E402
from huginn.evaluation.goal_judge import GoalJudge  # noqa: E402

ws = Path("/workspace/research_outputs/shusheng_rsi_run32")
eng = AutoloopEngine(workspace=ws)
model = eng.verification_model or eng.model

f = glob.glob(str(ws / "huginn_autoloop_report_*.md"))[0]
t = Path(f).read_text(encoding="utf-8")
cut = t.find("## Research Report")
body = t[cut:] if cut > 0 else t

obj = (ws / "objective.txt").read_text(encoding="utf-8")
print("report file:", Path(f).name, "| body chars:", len(body))

res = GoalJudge(llm=model).judge(objective=obj, trajectory=None, final_output=body)
import json

print(json.dumps(res, ensure_ascii=False, indent=1))