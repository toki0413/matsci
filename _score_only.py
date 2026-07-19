"""直接对已有 workspace 跑 score, 不重跑 agent."""
import os, sys
from pathlib import Path

# 用 DeepSeek 作 judge
os.environ["JUDGE_API_KEY"] = os.environ.get("DEEPSEEK_API_KEY", "")
os.environ["JUDGE_API_BASE"] = "https://api.deepseek.com"
os.environ["JUDGE_MODEL_NAME"] = "deepseek-chat"
os.environ["JUDGE_SKIP_VISION"] = "1"

sys.path.insert(0, str(Path(__file__).parent / "ResearchClawBench"))

# deepseek-chat 不支持 image_url, 直接走 image_prompt 会 400 → 非 JSON → score=0.
# monkey-patch _score_single_item: image 项 fallback 到 text prompt, 只看 report 文本.
# 升级路径: 换支持视觉的 judge (gpt-4o/claude/qwen-vl-max) 后移除此 patch.
import evaluation.score as _score_mod
_orig_score_single = _score_mod._score_single_item

def _patched_score_single(agent, report_text, item, target_image_path, generated_images, instructions):
    if item.get("type") == "image":
        item = dict(item)
        item["type"] = "text"
    return _orig_score_single(agent, report_text, item, None, [], instructions)

_score_mod._score_single_item = _patched_score_single

from evaluation.score import score_workspace

ws = sys.argv[1] if len(sys.argv) > 1 else r"c:/Users/wanzh/Desktop/matsci-agent/ResearchClawBench/workspaces/Astronomy_000_20260718_221927"
result = score_workspace(ws)
import json
print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
