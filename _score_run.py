import os
import sys

os.environ["DEEPSEEK_API_KEY"] = "sk-462dae89e16941e4b05c62f0e8e76aa6"
os.environ["JUDGE_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]
os.environ["JUDGE_API_BASE"] = "https://api.deepseek.com/v1"
os.environ["JUDGE_MODEL_NAME"] = "deepseek-chat"

sys.path.insert(0, "ResearchClawBench")
from evaluation.score import score_workspace

r = score_workspace(r"ResearchClawBench\workspaces\Astronomy_000_20260720_204310")
print("TOTAL:", r.get("total_score"))
print("ITEMS:")
for i in r.get("items", []):
    t = i.get("type", "text")
    w = i.get("weight", "?")
    s = i.get("score", "?")
    rs = (i.get("reasoning") or "")[:200]
    print(f"  [{t}] w={w} score={s} - {rs}")
