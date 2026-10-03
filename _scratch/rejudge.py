"""Re-judge an existing autoloop report with the fixed GoalJudge pipeline.

Usage: python rejudge.py <run_dir>
Reads <run_dir>/objective.txt and the newest huginn_autoloop_report_*.md,
strips the autoloop preamble (as cognitive_loop does post-fix), and calls GoalJudge.
"""
import glob
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HUGINN_PROVIDER", "internlm")
os.environ.setdefault("HUGINN_MODEL", "intern-s2-preview")
os.environ.setdefault(
    "INTERNLM_API_KEY", "sk-pGqiBxoW3z84Pk80jF3bZB1UK4awwOVVpHtE68hwrQ0hQlba"
)

from huginn.autoloop import AutoloopEngine  # noqa: E402
from huginn.evaluation.goal_judge import GoalJudge  # noqa: E402

ws = Path(sys.argv[1])
eng = AutoloopEngine(workspace=ws)
model = eng.verification_model or eng.model

files = sorted(glob.glob(str(ws / "huginn_autoloop_report_*.md")))
f = files[-1]
t = Path(f).read_text(encoding="utf-8")
cut = t.find("## Research Report")
body = t[cut:] if cut > 0 else t
obj = (ws / "objective.txt").read_text(encoding="utf-8")

print("run dir:", ws.name, "| report:", Path(f).name, "| body chars:", len(body))
res = GoalJudge(llm=model).judge(objective=obj, trajectory=None, final_output=body)
print(json.dumps(res, ensure_ascii=False, indent=1))